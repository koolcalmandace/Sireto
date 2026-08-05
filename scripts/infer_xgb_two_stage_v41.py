"""Two-Stage XGBoost Inference Engine — Version 4.1 (Full Parallel Opt 4.1 SSOT)

Features:
1. Multi-Processing Parallel Runner: Runs concurrent.futures.ProcessPoolExecutor with 4 workers.
2. CPU Thread Lock: Forces PyTorch and matrix operations to a single CPU thread to keep system responsive.
3. Expanded Candidate Database: Dynamically checks for data/candidates_v7_expanded and overrides store path.
4. Jaro-Winkler C++ Acceleration: Inherits features.py's RapidFuzz string comparison.
5. Resume-Safe Progress Preserver: Writes each query result immediately to disk.
"""

from __future__ import annotations

import os
# Set thread limits BEFORE importing libraries to ensure they take effect
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

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


# ---------------------------------------------------------------------------
# Global Worker State for Multiprocessing
# ---------------------------------------------------------------------------
_engine = None

def init_worker(profile_dict: dict):
    """Initialize a single worker process with its own inference engine instance."""
    global _engine
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load profile parameters
    profile = InferenceProfile(
        partitions_dir=Path(profile_dict["partitions_dir"]),
        ranker_path=Path(profile_dict["ranker_path"]),
        decider_path=Path(profile_dict["decider_path"]),
    )
    
    # Force single-threaded PyTorch in workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
        
    _engine = Version41XgbInferenceEngine.from_profile(profile)

def process_single_row(args: tuple) -> tuple:
    """Worker task function to process a single query row."""
    idx, row_dict = args
    global _engine
    if _engine is None:
        raise RuntimeError("Worker engine not initialized!")
        
    res = _engine.process_row(row_dict, top_n=20)
    
    row_res = row_dict.copy()
    row_res["v41_siret"] = res.get("siret")
    row_res["v41_score"] = res.get("score")
    row_res["v41_decision"] = res.get("decision")
    if res.get("top_match"):
        row_res["v41_match_name"] = res["top_match"].get("denomination")
        row_res["v41_match_city"] = res["top_match"].get("city")
        
    return idx, row_res


# ---------------------------------------------------------------------------
# DateTime JSON Serialization
# ---------------------------------------------------------------------------
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
# Version 4.1 Engine Class (Cascade Gate & Listwise Tie Breaking)
# ---------------------------------------------------------------------------
class Version41XgbInferenceEngine(XgbInferenceEngine):
    """Version 4.1 Two-Stage XGBoost Engine (Cascade Bypass Gate + Listwise ML Ranking)."""

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
        """Build candidate pool via V4.1 Cascade Bypass retrieval (single pass)."""
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
        """Process a single CRM record through Version 4.1 Pipeline."""
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
        logging.info("No trained models found. Falling back to default baseline models.")
        ranker_path = Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json")
        decider_path = Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json")
    else:
        meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_meta = meta_files[0]
        logging.info(f"Loading model configuration metadata: {latest_meta}")
        
        with open(latest_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        ranker_name = meta.get("ranker_fast_model") or meta.get("ranker_model")
        decider_name = meta.get("decider_model")
        
        ranker_path = Path(_PROJECT_ROOT / "models" / Path(ranker_name).name) if ranker_name else Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json")
        decider_path = Path(_PROJECT_ROOT / "models" / Path(decider_name).name) if decider_name else Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json")
    
    # Dynamic Candidate Database Scope Override (P0 Isolation)
    expanded_dir = Path(_PROJECT_ROOT / "data" / "candidates_v7_expanded")
    partitions_dir = expanded_dir if expanded_dir.exists() else Path(_PROJECT_ROOT / "data" / "candidates_v7_all")
    logging.info(f"[DATABASE DETECTED] Using active candidate store: {partitions_dir}")
    
    return InferenceProfile(
        partitions_dir=partitions_dir,
        ranker_path=ranker_path,
        decider_path=decider_path,
    )


# ---------------------------------------------------------------------------
# Main Execution Entrypoint (Parallel Multi-Processing)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Version 4.1 Parallel XGBoost Engine (CPU Thread Lock + Process Pool Concurrency)")
    parser.add_argument("--input-file", type=str, required=True, help="Input Excel/CSV file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output Excel/CSV file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows to process")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Initializing Version 4.1 Parallel Engine...")

    profile = get_latest_profile()
    logging.info(f"Active Ranker Model: {profile.ranker_path}")
    logging.info(f"Active Decider Model: {profile.decider_path}")

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

    # Resume Detection
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

    rows_to_process = []
    for idx, row in df.iterrows():
        orig_index = str(row.get("_orig_index") or row.get("crm_id") or row.get("ID") or idx)
        if orig_index in completed_orig_indices:
            continue
        row_dict = row.to_dict()
        row_dict["_orig_index"] = orig_index
        rows_to_process.append((idx, row_dict))

    total_queries = len(df)
    
    if rows_to_process:
        import concurrent.futures
        
        # Serialize profile attributes for pickle compatibility in child processes
        profile_dict = {
            "partitions_dir": str(profile.partitions_dir),
            "ranker_path": str(profile.ranker_path),
            "decider_path": str(profile.decider_path),
        }
        
        # Concurrency Worker Cap (safely default to 2 workers to control RAM footprint; configurable via environment variable)
        num_workers = int(os.environ.get("XGB_INFER_WORKERS", "2"))
        logging.info(f"Spawning {num_workers} concurrent workers for parallel batch matching...")
        
        progress = tqdm(total=total_queries, initial=len(completed_orig_indices), desc="Matching V4.2", mininterval=5.0, dynamic_ncols=True)
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(profile_dict,)
        ) as executor:
            futures = {executor.submit(process_single_row, item): item for item in rows_to_process}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    _, row_res = future.result()
                    results.append(row_res)
                    completed_orig_indices.add(row_res["_orig_index"])
                    
                    # Write immediately to disk (atomic resume-safety)
                    with open(temp_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row_res, cls=DateTimeEncoder) + "\n")
                except Exception as e:
                    logging.error(f"Error processing row future: {e}")
                    
                progress.update(1)
                
        progress.close()
    else:
        logging.info("All records already processed.")

    # Export to final format
    if results:
        out_df = pd.DataFrame(results)
        if "_orig_index" in out_df.columns:
            try:
                out_df["_orig_num"] = pd.to_numeric(out_df["_orig_index"], errors="coerce")
                out_df = out_df.sort_values("_orig_num", ascending=True).drop(columns=["_orig_num"])
            except Exception:
                pass
                
        logging.info(f"Exporting final results to: {out_path}")
        if out_path.suffix.lower() in [".xlsx", ".xls"]:
            out_df.to_excel(out_path, index=False)
        else:
            out_df.to_csv(out_path, sep=";", index=False)
            
        # Clean up temp file upon successful completion
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
                
        logging.info("Version 4.1 matching run finished successfully!")
    else:
        logging.warning("No results generated.")


if __name__ == "__main__":
    main()
