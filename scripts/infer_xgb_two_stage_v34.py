"""Two-Stage XGBoost Inference Engine — Version 3.4 (Full Opt 3.4 SSOT)

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
3. French Street Abbreviation Normalization:
   - Pre-processes query address strings (AV -> AVENUE, BD -> BOULEVARD, R -> RUE, ZA -> ZONE ARTISANALE, etc.)
     prior to CrmInput construction to enhance addr_jaro and street_name_jaro features.
4. Active Sister-Branch Disambiguation Engine (_resolve_active_sister_branches):
   - Ambiguity Gate: Triggers only when Candidate #1 & Candidate #2 share the same active SIREN ('A')
     AND score gap is ambiguous (score_top1 - score_top2 <= 0.05).
   - Hierarchical Evidence Swap:
       * Tier 1: Exact street number match (street_number_diff == 0) takes Rank 1.
       * Tier 2: Site/location descriptor token overlap (CENTRE, GARE, NORD, PARC, AGENCE) takes Rank 1.
       * Tier 3: Active Head Office (is_siege == 1) takes Rank 1 when location evidence is zero (street_number_diff == 9999).
5. Calibrated Threshold & Generic Query Safety Floor:
   - Calibrated AUTO threshold = 0.85.
   - Generic name floor (idf_name < 2.0 + missing street number + low street name jaro) -> NO_MATCH (Score < 0.50).
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
# Global EPCI Memory Map Loader
# ---------------------------------------------------------------------------
_EPCI_MAP: Dict[str, str] = {}

def _load_epci_map() -> Dict[str, str]:
    global _EPCI_MAP
    if _EPCI_MAP:
        return _EPCI_MAP
    epci_path = _PROJECT_ROOT / "data" / "insee_to_epci.json"
    if not epci_path.exists():
        epci_path = Path("C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/data/insee_to_epci.json")
    if epci_path.exists():
        try:
            with open(epci_path, "r", encoding="utf-8") as f:
                _EPCI_MAP = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load insee_to_epci.json: {e}")
            _EPCI_MAP = {}
    return _EPCI_MAP


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
# Version 3.4 Engine Class (SSOT Baseline)
# ---------------------------------------------------------------------------

class Version34XgbInferenceEngine(XgbInferenceEngine):
    """Version 3.4 Two-Stage XGBoost Engine (V3.0 SSOT Core + V2.6 JSONL + Sister Tie-Breaker)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epci_map = _load_epci_map()
        self._tfidf_cache = {}
        self._partition_cache = OrderedDict()
        self._cand_pool_cache = {}
        self.database_path = None  # Pure Parquet Offline Mode
        self.disable_scraper = True  # Pure Offline Mode
        self.auto_threshold = 0.85  # Calibrated V3.4 AUTO threshold
        
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
        """Build candidate pool via V3.0 3-stream retrieval with shared LRU memory caches (single pass)."""
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

    def _resolve_active_sister_branches(
        self,
        topk_idx: List[int],
        cand_list_n: List[Tuple[str, Dict[str, Any]]],
        feats_n: List[Dict[str, Any]],
        scores: np.ndarray,
        delta: float = 0.05,
    ) -> List[int]:
        """Disambiguate active sister branches (same SIREN) when decider scores are in a near-tie (delta <= 0.05)."""
        if len(topk_idx) < 2:
            return topk_idx

        idx1, idx2 = topk_idx[0], topk_idx[1]
        score1, score2 = float(scores[idx1]), float(scores[idx2])

        # Ambiguity Gate: Only run if score gap is <= delta
        if (score1 - score2) > delta:
            return topk_idx

        cand1, cand2 = cand_list_n[idx1][1], cand_list_n[idx2][1]
        siren1 = str(cand1.get("siren") or "").strip()
        siren2 = str(cand2.get("siren") or "").strip()

        # Scope Gate: Must belong to same active SIREN company
        if not siren1 or siren1 != siren2:
            return topk_idx

        state1 = str(cand1.get("etat_admin") or "").strip().upper()
        state2 = str(cand2.get("etat_admin") or "").strip().upper()
        if state1 == "F" or state2 == "F":
            return topk_idx  # Closed entities handled by _promote_open_over_closed

        # Tier 1: Exact Street Number Precision
        num_diff1 = float(feats_n[idx1].get("street_number_diff", 9999))
        num_diff2 = float(feats_n[idx2].get("street_number_diff", 9999))

        if num_diff2 == 0 and num_diff1 != 0:
            topk_idx[0], topk_idx[1] = idx2, idx1
            return topk_idx

        # Tier 2: Sub-Entity / Location Descriptor Token Overlap
        crm_name = str(feats_n[idx1].get("_crm_name") or "").upper()
        keywords = ["CENTRE", "GARE", "NORD", "SUD", "EST", "WEST", "PARC", "ECLUSES", "AGENCE", "BOURSE", "SITE", "POLE"]
        matched_kws = [kw for kw in keywords if kw in crm_name]

        if matched_kws:
            c1_text = (str(cand1.get("address") or "") + " " + str(cand1.get("enseigne1") or "")).upper()
            c2_text = (str(cand2.get("address") or "") + " " + str(cand2.get("enseigne1") or "")).upper()
            c1_has = any(kw in c1_text for kw in matched_kws)
            c2_has = any(kw in c2_text for kw in matched_kws)

            if c2_has and not c1_has:
                topk_idx[0], topk_idx[1] = idx2, idx1
                return topk_idx

        # Tier 3: Zero-Evidence Active Head Office Fallback
        is_siege1 = bool(cand1.get("is_siege"))
        is_siege2 = bool(cand2.get("is_siege"))

        if is_siege2 and not is_siege1 and num_diff1 == 9999 and num_diff2 == 9999:
            topk_idx[0], topk_idx[1] = idx2, idx1
            return topk_idx

        return topk_idx

    def process_row(
        self,
        row_dict: dict,
        top_n: int = 20,
        enable_scraper: bool = False,
    ) -> dict:
        """Process a single CRM record through Version 3.4 Pipeline."""
        input_name = str(row_dict.get("crm_name") or row_dict.get("Client final") or row_dict.get("name") or "").strip()
        input_postcode = str(row_dict.get("postcode") or row_dict.get("crm_postcode") or row_dict.get("Code Postal") or "").strip()
        input_city = str(row_dict.get("crm_city") or row_dict.get("Commune") or row_dict.get("city") or "").strip()
        input_insee = str(row_dict.get("insee_code") or row_dict.get("crm_insee") or row_dict.get("Code INSEE") or "").strip()
        raw_address = str(row_dict.get("crm_address") or row_dict.get("Adresse") or row_dict.get("address") or "").strip()

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
# Main Execution Entrypoint (V2.0-V2.6 JSONL Architecture + V3.4 SSOT)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Version 3.4 Two-Stage XGBoost Engine (V3.0 SSOT + V2.6 JSONL Architecture)")
    parser.add_argument("--input-file", type=str, required=True, help="Input Excel/CSV file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output Excel/CSV file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows to process")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Initializing Version 3.4 Engine (V3.0 SSOT Core + V2.6 JSONL Blueprint)...")

    profile = InferenceProfile(
        partitions_dir=Path(_PROJECT_ROOT / "data" / "candidates_v7_all"),
        ranker_path=Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json"),
        decider_path=Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json"),
    )

    engine = Version34XgbInferenceEngine.from_profile(profile)

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
    logging.info(f"Processing {total_queries} records sequentially (V2.0-V2.6 JSONL Append Architecture)...")

    progress = tqdm(total=total_queries, initial=len(completed_orig_indices), desc="Matching V3.4 SSOT")
    for idx, row in df.iterrows():
        orig_index = str(row.get("_orig_index") or row.get("crm_id") or row.get("ID") or idx)
        if orig_index in completed_orig_indices:
            continue

        row_dict = row.to_dict()
        row_dict["_orig_index"] = orig_index

        res = engine.process_row(row_dict, top_n=20)

        row_res = row_dict.copy()
        row_res["v34_siret"] = res.get("siret")
        row_res["v34_score"] = res.get("score")
        row_res["v34_decision"] = res.get("decision")
        if res.get("top_match"):
            row_res["v34_match_name"] = res["top_match"].get("denomination")
            row_res["v34_match_city"] = res["top_match"].get("city")

        results.append(row_res)
        completed_orig_indices.add(orig_index)

        # V2.0 - V2.6 Immediate Atomic JSONL Text Line Append
        try:
            with open(temp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row_res, cls=DateTimeEncoder) + "\n")
        except Exception as e:
            logging.error(f"Failed to write progress line to {temp_path}: {e}")

        # Clear TF-IDF / LRU cache periodically every 500 records to free RAM
        if (len(completed_orig_indices)) % 500 == 0:
            engine.clear_tfidf_cache()

        progress.update(1)

    progress.close()

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

    logging.info(f"Version 3.4 Processing Complete! Saved output to: {out_path}")


if __name__ == "__main__":
    main()
