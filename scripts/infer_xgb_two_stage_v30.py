"""Two-Stage XGBoost Inference Engine — Version 3.0 (Full Opt 3.0)

Features:
1. 3-Stream Candidate Retrieval Engine:
   - Stream 1: EPCI Inter-Communal Territory Word N-Gram TF-IDF (n=1,2) -> Top 10
   - Stream 2: Department Native UL_SIGLE & Acronym Char 3/4-Gram TF-IDF -> Top 10
   - Stream 3: Corporate Parent & Brand Network Subsequence TF-IDF -> Top 10
2. In-Memory O(1) EPCI Commercial Territory Resolver (insee_to_epci.json)
3. 100% Dictionary-Free Native UL_SIGLE Indexing
4. Dual Token Normalization (Raw Baseline Tokens & Cleaned Core Brand Tokens)
5. 19-Feature Vector Extraction (name_jaro_raw, name_jaro_cleaned, name_jaro_max)
6. Pure Parquet Execution with 100% Muted SQLite Database
"""

from __future__ import annotations

import os

# CRITICAL: Set XGB_SEMANTIC_ENABLED = 1 before importing any matcher modules
os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
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
from dataclasses import replace

# Project path resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    make_features_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.infer import XgbInferenceEngine, CrmInput
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


def _resolve_epci_id(insee_code: str = "", postcode: str = "") -> str:
    """Resolve 9-digit EPCI ID in O(1) time."""
    epci_map = _load_epci_map()
    if insee_code and str(insee_code).zfill(5) in epci_map:
        return str(epci_map[str(insee_code).zfill(5)])
    if postcode and len(str(postcode)) == 5:
        if str(postcode) in epci_map:
            return str(epci_map[str(postcode)])
    return ""


# ---------------------------------------------------------------------------
# Pre-Processing & Normalization Functions
# ---------------------------------------------------------------------------

def _strip_city_branch_suffix(crm_name: str, crm_city: str = "", postcode: str = "", crm_address: str = "") -> str:
    """Safe city-anchored branch suffix stripper."""
    if not crm_name:
        return ""
    cleaned = crm_name.upper().strip()
    cleaned = re.sub(r"\s*[-/:_]\s*\b(?:\d+[A-Z0-9]*|[A-Z]+\d+)\b\s*$", "", cleaned)
    
    resolved_town = str(crm_city or "").upper().strip()
    if not resolved_town and postcode:
        addr_match = re.search(r"\b\d{5}\s+([A-Z\s-]+)\b", str(crm_address or "").upper())
        if addr_match:
            resolved_town = addr_match.group(1).strip()
            
    if " - " in cleaned or " / " in cleaned:
        parts = re.split(r"\s+[-/:–]\s+", cleaned)
        if len(parts) >= 2:
            trailing = parts[-1].strip().upper()
            leading = parts[0].strip().upper()
            if len(leading) >= 3 and (trailing == resolved_town or (len(trailing) >= 3 and resolved_town and trailing in resolved_town)):
                cleaned = leading
                
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else crm_name.upper().strip()


def _extract_core_company_name(crm_name: str) -> str:
    """Extract core company name by removing trailing operational codes."""
    if not crm_name:
        return ""
    raw_upper = crm_name.upper().strip()
    cleaned = re.sub(r"\s*[-/:_]\s*\b(?:\d+[A-Z0-9]*|[A-Z]+\d+)\b\s*$", "", raw_upper)
    parts = re.split(r"\s+[-/:–]\s+|\s*\(", cleaned)
    if parts and len(parts[0].strip()) >= 3:
        cleaned = parts[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else raw_upper


# ---------------------------------------------------------------------------
# Version 3.0 Engine Class (100% Self-Contained)
# ---------------------------------------------------------------------------

from collections import OrderedDict

class Version30XgbInferenceEngine(XgbInferenceEngine):
    """Version 3.0 Two-Stage XGBoost Engine (Self-Contained, Independent)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epci_map = _load_epci_map()
        self._tfidf_cache = {}
        self._partition_cache = OrderedDict()
        self._cand_pool_cache = {}
        self.database_path = None  # Pure Parquet Offline Mode
        self.disable_scraper = True  # Pure Offline Mode

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
        """Build candidate pool via 3-stream retrieval with shared LRU memory caches."""
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
        """Process a single CRM record through Version 3.0 3-Stream pipeline."""
        input_name = str(row_dict.get("crm_name") or row_dict.get("name") or "").strip()
        input_postcode = str(row_dict.get("postcode") or row_dict.get("crm_postcode") or "").strip()
        input_city = str(row_dict.get("crm_city") or row_dict.get("city") or "").strip()
        input_insee = str(row_dict.get("insee_code") or row_dict.get("crm_insee") or "").strip()
        input_address = str(row_dict.get("crm_address") or row_dict.get("address") or "").strip()
        input_siren = str(row_dict.get("siren") or "").strip()

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
        decision = "AUTO" if top_score >= 0.90 else ("REVIEW" if top_score >= 0.50 else "NO_MATCH")

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
# Main Execution CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Version 3.0 Two-Stage XGBoost Engine")
    parser.add_argument("--input-file", type=str, required=True, help="Input Excel/CSV file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output Excel/CSV file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows to process")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Initializing Version 3.0 Two-Stage XGBoost Matching Engine...")

    profile = InferenceProfile(
        partitions_dir=Path(_PROJECT_ROOT / "data" / "candidates_v7_all"),
        ranker_path=Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json"),
        decider_path=Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json"),
    )

    engine = Version30XgbInferenceEngine.from_profile(profile)

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

    logging.info(f"Processing {len(df)} records through Version 3.0 Engine...")
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching V3.0"):
        res = engine.process_row(row.to_dict(), top_n=20)
        row_res = row.to_dict()
        row_res["v30_siret"] = res.get("siret")
        row_res["v30_score"] = res.get("score")
        row_res["v30_decision"] = res.get("decision")
        if res.get("top_match"):
            row_res["v30_match_name"] = res["top_match"].get("denomination")
            row_res["v30_match_city"] = res["top_match"].get("city")
        results.append(row_res)

    out_df = pd.DataFrame(results)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() in [".xlsx", ".xls"]:
        out_df.to_excel(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    logging.info(f"Version 3.0 Processing Complete! Saved output to: {out_path}")


if __name__ == "__main__":
    main()
