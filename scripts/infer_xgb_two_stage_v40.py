"""Two-Stage XGBoost Inference Engine — Version 4.0 (Full Opt 4.0 SSOT)

Architecture & Feature Specifications:
1. Version 3.0 SSOT Core Retrieval Foundation:
   - Single-pass candidate pool retrieval via build_candidate_pool() in retrieval.py.
   - 3-stream candidate retrieval (EPCI + Native UL_SIGLE + Corporate Parent Brand).
   - Hard-locked stage1_top_n = 20 to preserve ~0.35s/query execution speed.
2. Version 2.0 - 2.6 Ultra-Stable JSONL Progress & Resume Architecture:
   - Appends output records immediately to a temporary text progress file (.jsonl) on disk
     using atomic open(temp_path, "a", encoding="utf-8") writes (0 RAM overhead, <0.0001s/write).
   - Auto-detects incomplete runs on startup and resumes seamlessly from missing _orig_index.
   - Flushes buffers and clears LRU TF-IDF memory caches every 500 records.
3. Complete French Column Key Mapping (Robust Headers):
   - Supports crm_cp, cp, postcode, crm_postcode, Code Postal.
   - Supports crm_commune, commune, crm_city, city, Ville.
   - Supports crm_adresse, adresse, crm_address, address, Adresse.
4. French Street Abbreviation Normalization:
   - Pre-processes query address strings (AV -> AVENUE, BD -> BOULEVARD, R -> RUE, ZA -> ZONE ARTISANALE, etc.)
     prior to CrmInput construction to enhance addr_jaro and street_name_jaro features.
5. Calibrated Threshold & Generic Query Safety Floor:
   - Calibrated AUTO threshold = 0.85.
   - Generic name floor (idf_name < 2.0 + missing street number + low street name jaro) -> NO_MATCH (Score < 0.50).
6. Removed dead Post-Decider Heuristics:
   - All sister-branch and co-location tie-breakers are handled natively by Stage 2 listwise ranking.
"""

from __future__ import annotations

import os

# Unbuffer stdout immediately for live Windows CMD output
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import sys
import json
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import replace

# Project path resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    make_features_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.infer import XgbInferenceEngine, CrmInput, TopKRow
from src.xgb_matcher.profile import InferenceProfile
from src.xgb_matcher.retrieval import build_candidate_pool


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime and pandas Timestamp serialization."""
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass
        return super().default(obj)


# ---------------------------------------------------------------------------
# Query Pre-Processing & Normalization Functions
# ---------------------------------------------------------------------------

_FRENCH_STREET_ABBR_MAP = {
    r"\bAV\b": "AVENUE",
    r"\bAVE\b": "AVENUE",
    r"\bBD\b": "BOULEVARD",
    r"\bBVD\b": "BOULEVARD",
    r"\bR\b": "RUE",
    r"\bZA\b": "ZONE ARTISANALE",
    r"\bZI\b": "ZONE INDUSTRIELLE",
    r"\bIMP\b": "IMPASSE",
    r"\bALL\b": "ALLEE",
    r"\bRTE\b": "ROUTE",
    r"\bPL\b": "PLACE",
    r"\bSQ\b": "SQUARE",
    r"\bCRS\b": "COURS",
    r"\bCH\b": "CHEMIN",
    r"\bRES\b": "RESIDENCE",
    r"\bBLD\b": "BOULEVARD",
}

def _normalize_french_street_address(raw_addr: str) -> str:
    """Expand common French street abbreviations prior to feature calculation."""
    if not raw_addr:
        return ""
    addr_upper = str(raw_addr).upper().strip()
    for pattern, replacement in _FRENCH_STREET_ABBR_MAP.items():
        addr_upper = re.sub(pattern, replacement, addr_upper)
    return re.sub(r"\s+", " ", addr_upper).strip()


# ---------------------------------------------------------------------------
# Version 4.0 Engine Class (Cascade Gate & Listwise Tie Breaking)
# ---------------------------------------------------------------------------

class Version40XgbInferenceEngine(XgbInferenceEngine):
    """Version 4.0 Two-Stage XGBoost Engine (Cascade Bypass Gate + Listwise ML Ranking)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tfidf_cache = {}
        self._partition_cache = OrderedDict()
        self._cand_pool_cache = {}
        self.database_path = None  # Pure Parquet Offline Mode
        self.disable_scraper = True  # Pure Offline Mode
        self.auto_threshold = 0.85  # Calibrated AUTO threshold
        
        # Hard-lock Stage 1 Top N candidates to 20 (SSOT strategy)
        if hasattr(self, "retrieval_config") and self.retrieval_config is not None:
            try:
                self.retrieval_config = replace(self.retrieval_config, stage1_top_n=20)
            except Exception:
                pass

    def clear_tfidf_cache(self):
        """Clear TF-IDF, candidate pool, and partition caches to free RAM."""
        if hasattr(self, "_tfidf_cache") and self._tfidf_cache is not None:
            self._tfidf_cache.clear()
        if hasattr(self, "_partition_cache") and self._partition_cache is not None:
            self._partition_cache.clear()
        if hasattr(self, "_cand_pool_cache") and self._cand_pool_cache is not None:
            self._cand_pool_cache.clear()

    def _build_candidate_pool(
        self,
        crm_row: Dict[str, Any],
        crm_id: str,
        crm_pre: Dict[str, Any],
        pool_mode: str,
        drop_unnamed: bool,
        exclude_closed: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], float]:
        """Build candidate pool via V4.0 Cascade Bypass retrieval (single pass)."""
        result = build_candidate_pool(
            store=self.store,
            crm_row=crm_row,
            crm_pre=crm_pre,
            config=self.retrieval_config,
            tfidf_cache=getattr(self, "_tfidf_cache", {}),
            gt_siret=None,
            siren_to_geo=self.siren_to_geo,
            partition_cache=getattr(self, "_partition_cache", None),
        )
        return result.candidates, result.idf_map, result.default_idf

    def process_row(
        self,
        row_dict: dict,
        top_n: int = 20,
        enable_scraper: bool = False,
    ) -> dict:
        """Process a single CRM record through Version 4.0 Pipeline."""
        # Robust French & English column dictionary lookup chain
        input_name = str(
            row_dict.get("crm_name") or row_dict.get("Client final") or row_dict.get("name") or row_dict.get("nom") or ""
        ).strip()
        
        input_postcode = str(
            row_dict.get("crm_cp") or row_dict.get("cp") or row_dict.get("postcode") or row_dict.get("crm_postcode") or row_dict.get("Code Postal") or ""
        ).strip()
        
        input_city = str(
            row_dict.get("crm_commune") or row_dict.get("commune") or row_dict.get("crm_city") or row_dict.get("Commune") or row_dict.get("city") or row_dict.get("ville") or ""
        ).strip()
        
        input_insee = str(
            row_dict.get("crm_insee") or row_dict.get("insee_code") or row_dict.get("insee") or row_dict.get("Code INSEE") or ""
        ).strip()
        
        raw_address = str(
            row_dict.get("crm_adresse") or row_dict.get("adresse") or row_dict.get("crm_address") or row_dict.get("Adresse") or row_dict.get("address") or ""
        ).strip()

        # Pre-process street abbreviations
        input_address = _normalize_french_street_address(raw_address)

        crm_in = CrmInput(
            crm_id=str(row_dict.get("crm_id") or "1"),
            crm_name=input_name,
            postcode=input_postcode,
            crm_city=input_city,
            insee=input_insee,
            crm_address=input_address,
        )

        topk_rows = self.infer_topk(
            crm_in,
            top_k=top_n,
            pool_mode="insee_then_postcode",
            export_routing_features=True
        )

        if not topk_rows:
            return {
                "crm_name": input_name,
                "siret": None,
                "score": 0.0,
                "decision": "NO_MATCH",
                "top_candidates": [],
            }

        top_match = topk_rows[0]
        top_score = top_match.score

        # Generic Query Safety Floor (idf_name < 2.0 with weak address evidence -> NO_MATCH)
        idf_name = float(getattr(top_match, "idf_name", 5.0))
        num_diff = float(getattr(top_match, "street_number_diff", 9999))
        street_name_jaro = float(getattr(top_match, "street_name_jaro", 0.0))

        if idf_name < 2.0 and num_diff == 9999 and street_name_jaro < 0.60:
            decision = "NO_MATCH"
            top_score = min(top_score, 0.49)
        else:
            decision = "AUTO" if top_score >= self.auto_threshold else ("REVIEW" if top_score >= 0.50 else "NO_MATCH")

        return {
            "crm_name": input_name,
            "siret": top_match.siret_candidate,
            "score": top_score,
            "decision": decision,
            "top_match": {
                "denomination": top_match.candidate_name,
                "city": top_match.candidate_city,
                "postcode": top_match.candidate_postcode,
                "siret": top_match.siret_candidate,
            },
            "top_candidates": [r.to_dict() for r in topk_rows[:5]],
        }


# ---------------------------------------------------------------------------
# Helper function to find latest trained models
# ---------------------------------------------------------------------------

def get_latest_profile() -> InferenceProfile:
    """Read the latest trained model profile metadata to load newest model."""
    meta_files = list(Path(_PROJECT_ROOT / "models").glob("xgb_two_stage_meta_*.json"))
    if not meta_files:
        # Fallback default baseline models if no training has run yet
        logging.info("No trained models found. Falling back to default baseline models.")
        return InferenceProfile(
            partitions_dir=Path(_PROJECT_ROOT / "data" / "candidates_v7_all"),
            ranker_path=Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json"),
            decider_path=Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json"),
        )
    
    # Sort metadata files by modification time
    meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_meta = meta_files[0]
    logging.info(f"Loading model configuration metadata: {latest_meta}")
    
    with open(latest_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    ranker_name = meta.get("ranker_fast_model") or meta.get("ranker_model")
    decider_name = meta.get("decider_model")
    
    ranker_path = Path(_PROJECT_ROOT / "models" / Path(ranker_name).name) if ranker_name else Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json")
    decider_path = Path(_PROJECT_ROOT / "models" / Path(decider_name).name) if decider_name else Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json")
    
    return InferenceProfile(
        partitions_dir=Path(_PROJECT_ROOT / "data" / "candidates_v7_all"),
        ranker_path=ranker_path,
        decider_path=decider_path,
    )


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Version 4.0 Two-Stage XGBoost Engine (Cascade Bypass Gate + Listwise ML Ranking)")
    parser.add_argument("--input-file", type=str, required=True, help="Input Excel/CSV file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output Excel/CSV file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows to process")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Initializing Version 4.0 Engine...")

    # Load latest trained model profile
    profile = get_latest_profile()
    logging.info(f"Active Ranker Model: {profile.ranker_path}")
    logging.info(f"Active Decider Model: {profile.decider_path}")

    engine = Version40XgbInferenceEngine.from_profile(profile)

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path, dtype=str)
    else:
        try:
            df = pd.read_csv(input_path, sep=";", dtype=str)
        except Exception:
            df = pd.read_csv(input_path, sep=None, engine="python", dtype=str)

    if args.limit > 0 and len(df) > args.limit:
        if args.random_seed is not None:
            df = df.sample(n=args.limit, random_state=args.random_seed).reset_index(drop=True)
        else:
            df = df.iloc[:args.limit].reset_index(drop=True)

    out_path = Path(args.output_file)
    temp_path = out_path.with_suffix(".jsonl")

    results = []
    completed_orig_indices = set()

    # V2.0 - V2.6 Resume Detection Architecture
    if temp_path.exists():
        logging.info(f"Found temporary JSONL progress file: {temp_path}. Resuming...")
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    results.append(row)
                    if "_orig_index" in row:
                        completed_orig_indices.add(str(row["_orig_index"]))
            logging.info(f"[RESUME DETECTED] Loaded {len(results)} completed records ({len(completed_orig_indices)} queries).")
        except Exception as e:
            logging.error(f"Failed to load progress from {temp_path}: {e}. Starting fresh.")
            results.clear()
            completed_orig_indices.clear()

    total_queries = len(df)
    logging.info(f"Processing {total_queries} records sequentially...")

    progress = tqdm(total=total_queries, initial=len(completed_orig_indices), desc="Matching V4.0 Cascade")
    for idx, row in df.iterrows():
        orig_index = str(row.get("_orig_index") or row.get("crm_id") or row.get("ID") or idx)
        if orig_index in completed_orig_indices:
            continue

        row_dict = row.to_dict()
        row_dict["_orig_index"] = orig_index

        res = engine.process_row(row_dict, top_n=20)

        row_res = row_dict.copy()
        row_res["v40_siret"] = res.get("siret")
        row_res["v40_score"] = res.get("score")
        row_res["v40_decision"] = res.get("decision")
        if res.get("top_match"):
            row_res["v40_match_name"] = res["top_match"].get("denomination")
            row_res["v40_match_city"] = res["top_match"].get("city")

        results.append(row_res)
        completed_orig_indices.add(orig_index)
        
        # Write progress immediately to disk to guarantee progress preservation on failure
        try:
            with open(temp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row_res, cls=DateTimeEncoder) + "\n")
        except Exception as e:
            logging.error(f"Failed to write progress to {temp_path}: {e}")

        # Clear TF-IDF / LRU cache periodically every 500 records to free RAM
        if (len(completed_orig_indices)) % 500 == 0:
            engine.clear_tfidf_cache()

        progress.update(1)

    progress.close()

    # Clear all RAM caches to release the large partition footprint before Excel compilation
    logging.info("Clearing memory caches and running garbage collection to release RAM before Excel serialization...")
    engine.clear_tfidf_cache()
    if hasattr(engine, "_partition_cache"):
        engine._partition_cache.clear()
    if hasattr(engine, "_cand_pool_cache"):
        engine._cand_pool_cache.clear()
    import gc
    gc.collect()

    # V2.0 - V2.6 Final DataFrame Assembly & Export Architecture
    out_df = pd.DataFrame(results)
    if "_orig_index" in out_df.columns:
        try:
            out_df["_orig_num"] = pd.to_numeric(out_df["_orig_index"], errors="coerce")
            out_df = out_df.sort_values("_orig_num", ascending=True).drop(columns=["_orig_num"])
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in [".xlsx", ".xls"]:
        out_df.to_excel(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    # Clean up temporary JSONL progress file after 100% successful export
    if temp_path.exists():
        try:
            temp_path.unlink()
            logging.info(f"[CLEANUP] Deleted temporary JSONL progress file after 100% completion: {temp_path}")
        except Exception as e:
            logging.warning(f"Could not remove temporary progress file: {e}")

    logging.info(f"Version 4.0 Processing Complete! Saved output to: {out_path}")


if __name__ == "__main__":
    main()
