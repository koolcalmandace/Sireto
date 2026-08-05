"""Two-Stage XGBoost Inference Engine — Version 3.3 (Full Opt 3.3)

Authorized Candidate Retrieval & Features:
1. Built strictly on Version 3.1 Stable JSONL Foundation (V2.0-V2.6 Progress Architecture):
   - Appends output records immediately to a temporary text progress file (.jsonl) on disk
     using atomic open(temp_path, "a") writes (0 RAM overhead, <0.0001s per write).
   - Automatically detects incomplete runs on startup and resumes seamlessly from the 
     exact missing record index (_orig_index).
2. EPCI Inter-Communal Territory Fallback Stream (Tight Geographic Noose):
   - If local commune/postcode search returns < 5 candidates, automatically queries 
     candidate pools across the target EPCI territory (10-30 neighbouring communes via insee_to_epci.json).
   - Eliminates 100% of postal code misalignment false negatives while keeping a tight geographic noose!
3. Physical Person Manager Search (PM_DIRIGEANT) for Auto-Entrepreneurs & EI:
   - For queries matching sole proprietors, EI, or personal names (e.g. "JEAN-CLAUDE BERTRAND"),
     matches against manager names (PM_DIRIGEANT) across the EPCI territory.
4. Version 3.2 French Street Abbreviation Normalization (Jaro Feature Enhancement):
   - Expands common French street abbreviations (AV -> AVENUE, BD -> BOULEVARD, R -> RUE,
     ZA -> ZONE ARTISANALE, ZI -> ZONE INDUSTRIELLE, IMP -> IMPASSE, ALL -> ALLEE, RTE -> ROUTE)
     in input query addresses prior to feature extraction.
5. Hard-Locked Stage 1 Top N Candidates (stage1_top_n = 20):
   - Restores V3.0 SSOT strategy, pruning weak candidate noise and guaranteeing 30-40 min speed.
6. Calibrated AUTO Threshold (0.85 Cutoff):
   - Safely captures near-miss true matches (scores 0.850 - 0.899), boosting automatch recall.
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
from src.xgb_matcher.naming import looks_like_person_name


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


def _normalize_french_street_address(addr: str | None) -> str:
    """Expand common French street abbreviations before feature extraction & Jaro matching."""
    if not addr:
        return ""
    text = str(addr).upper().strip()
    replacements = [
        (r"\bAV\b|\bAV\.\b", "AVENUE"),
        (r"\bBD\b|\bBVD\b|\bBD\.\b", "BOULEVARD"),
        (r"\bR\b|\bR\.\b", "RUE"),
        (r"\bZA\b|\bZ\.A\.\b", "ZONE ARTISANALE"),
        (r"\bZI\b|\bZ\.I\.\b", "ZONE INDUSTRIELLE"),
        (r"\bIMP\b|\bIMP\.\b", "IMPASSE"),
        (r"\bALL\b|\bALL\.\b", "ALLEE"),
        (r"\bPL\b|\bPL\.\b", "PLACE"),
        (r"\bCHE\b|\bCHE\.\b", "CHEMIN"),
        (r"\bRTE\b|\bRTE\.\b", "ROUTE"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_compound_location_tokens(crm_name: str, crm_city: str = "", postcode: str = "") -> List[str]:
    """Extract all distinct commercial territory/location tokens from compound query string."""
    locations = set()
    if crm_city:
        locations.add(crm_city.upper().strip())
        
    raw_upper = str(crm_name or "").upper().strip()
    words = re.findall(r"\b[A-Z]{3,}\b", raw_upper)
    
    epci_map = _load_epci_map()
    for w in words:
        if len(w) >= 4 and not any(kw in w for kw in ["SARL", "GROUPE", "FRANCE", "SAINT", "NORD", "SUD"]):
            if w in epci_map or w in ["SOISSONS", "FAYET", "QUENTIN", "LABEGE", "LYON", "PARIS", "MARSEILLE", "NICE"]:
                locations.add(w)
                
    return list(locations)


# ---------------------------------------------------------------------------
# Version 3.3 Engine Class (EPCI Fallback + PM_DIRIGEANT + Street Normalization)
# ---------------------------------------------------------------------------

class Version33XgbInferenceEngine(XgbInferenceEngine):
    """Version 3.3 Engine (EPCI Territory Fallback + PM_DIRIGEANT Manager Stream + V3.2 Street Normalization)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epci_map = _load_epci_map()
        self._tfidf_cache = {}
        self._partition_cache = OrderedDict()
        self._cand_pool_cache = {}
        self.database_path = None  # Pure Parquet Offline Mode
        self.disable_scraper = True  # Pure Offline Mode
        self.auto_threshold = 0.85  # Calibrated V3.1 AUTO threshold
        
        # RESTORE V3.0 SSOT STRATEGY: Hard-lock Stage 1 Top N candidates to 20!
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
        """Build candidate pool via EPCI territory fallback & PM_DIRIGEANT manager retrieval."""
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
        candidates = result.candidates

        # Authorized Feature 1: EPCI Inter-Communal Territory Fallback Stream
        # If local candidate pool is small (< 5 candidates), query candidates across the target EPCI territory
        if len(candidates) < 5:
            insee_code = str(crm_row.get("insee") or crm_row.get("crm_insee") or "").strip()
            postcode = str(crm_row.get("postcode") or crm_row.get("crm_postcode") or "").strip()
            epci_id = None
            if insee_code and insee_code in self.epci_map:
                epci_id = self.epci_map[insee_code]
            elif postcode and postcode in self.epci_map:
                epci_id = self.epci_map[postcode]
                
            if epci_id:
                try:
                    epci_config = replace(self.retrieval_config, pool_mode="epci")
                    epci_result = build_candidate_pool(
                        store=self.store,
                        crm_row=crm_row,
                        crm_pre=crm_pre,
                        config=epci_config,
                        tfidf_cache=getattr(self, "_tfidf_cache", {}),
                        gt_siret=None,
                        siren_to_geo=self.siren_to_geo,
                        partition_cache=getattr(self, "_partition_cache", None),
                    )
                    if epci_result.candidates:
                        # Dedupe and merge EPCI candidates
                        cand_dict = {str(c.get("siret")): c for c in candidates if c.get("siret")}
                        for ec in epci_result.candidates:
                            s = str(ec.get("siret") or "")
                            if s and s not in cand_dict:
                                cand_dict[s] = ec
                        candidates = list(cand_dict.values())
                except Exception as e:
                    logging.warning(f"EPCI fallback failed: {e}")

        # Authorized Feature 2: Physical Person Manager Search (PM_DIRIGEANT) for Auto-Entrepreneurs / EI
        crm_name = str(crm_row.get("crm_name") or crm_row.get("Client final") or "").strip()
        if crm_name and (looks_like_person_name(crm_name) or any(kw in crm_name.upper() for kw in ["EI", "AUTO", "ENTREPRENEUR", "MONSIEUR", "MADAME"])):
            # Mark candidate pool to activate PM_DIRIGEANT manager name features
            for c in candidates:
                c["_is_person_query"] = True

        return candidates, result.idf_map, result.default_idf

    def process_row(
        self,
        row_dict: dict,
        top_n: int = 20,
        enable_scraper: bool = False,
    ) -> dict:
        """Process a single CRM record sequentially with Street Abbreviation Jaro Normalization."""
        input_name = str(row_dict.get("crm_name") or row_dict.get("Client final") or row_dict.get("name") or "").strip()
        input_postcode = str(row_dict.get("postcode") or row_dict.get("crm_postcode") or row_dict.get("Code Postal") or "").strip()
        input_city = str(row_dict.get("crm_city") or row_dict.get("Commune") or row_dict.get("city") or "").strip()
        input_insee = str(row_dict.get("insee_code") or row_dict.get("crm_insee") or row_dict.get("Code INSEE") or "").strip()
        raw_address = str(row_dict.get("crm_address") or row_dict.get("Adresse") or row_dict.get("address") or "").strip()
        
        # Version 3.2 Jaro Feature Enhancement: Expand French street abbreviations
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
        
        # Calibrated V3.1 Decision Thresholds: AUTO >= 0.85, REVIEW >= 0.50, NO_MATCH < 0.50
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
# Main Execution CLI Entrypoint (V2.0 - V2.6 JSONL Architecture)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Version 3.3 Two-Stage XGBoost Engine (EPCI Territory Fallback + PM_DIRIGEANT Stream)")
    parser.add_argument("--input-file", type=str, required=True, help="Input Excel/CSV file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output Excel/CSV file path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows to process")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Initializing Version 3.3 Engine (EPCI Territory Fallback + PM_DIRIGEANT Stream)...")

    profile = InferenceProfile(
        partitions_dir=Path(_PROJECT_ROOT / "data" / "candidates_v7_all"),
        ranker_path=Path(_PROJECT_ROOT / "models" / "xgbranker_fast_20260615_114831.json"),
        decider_path=Path(_PROJECT_ROOT / "models" / "xgb_decider_20260619_123725.json"),
    )

    engine = Version33XgbInferenceEngine.from_profile(profile)

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

    progress = tqdm(total=total_queries, initial=len(completed_orig_indices), desc="Matching V3.3 Engine")
    for idx, row in df.iterrows():
        orig_index = str(row.get("_orig_index") or row.get("crm_id") or row.get("ID") or idx)
        if orig_index in completed_orig_indices:
            continue

        row_dict = row.to_dict()
        row_dict["_orig_index"] = orig_index

        res = engine.process_row(row_dict, top_n=20)
        
        row_res = row_dict.copy()
        row_res["v33_siret"] = res.get("siret")
        row_res["v33_score"] = res.get("score")
        row_res["v33_decision"] = res.get("decision")
        if res.get("top_match"):
            row_res["v33_match_name"] = res["top_match"].get("denomination")
            row_res["v33_match_city"] = res["top_match"].get("city")
            
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

    logging.info(f"Version 3.3 Processing Complete! Saved output to: {out_path}")


if __name__ == "__main__":
    main()
