"""Two-stage XGBoost inference (SSOT, no legacy modes) - Optimized Version."""

from __future__ import annotations

# CRITICAL: enable semantic before imports
import os

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
import datetime
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import replace

# Ensure project root and src directory are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime, date and pandas Timestamp serialization."""
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass
        return super().default(obj)

import pandas as pd
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def _extract_core_company_name(crm_name: str) -> str:
    if not crm_name:
        return ""
    import re
    raw_upper = crm_name.upper().strip()
    
    # 1. Delimiter-Anchored Suffix Stripper (Safe: requires hyphen/slash/colon + trailing code)
    cleaned = re.sub(r"\s*[-/:_]\s*\b(?:\d+[A-Z0-9]*|[A-Z]+\d+)\b\s*$", "", raw_upper)
    
    # 2. Split on structural delimiters if first part >= 3 chars
    parts = re.split(r"\s+[-/:–]\s+|\s*\(", cleaned)
    if parts and len(parts[0].strip()) >= 3:
        cleaned = parts[0].strip()
        
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # 3. Minimum 3-Character Safety Guard (Reverts if < 3 chars)
    if len(cleaned) < 3:
        return raw_upper
        
    return cleaned

def _expand_acronyms(text):
    if not text:
        return ""
    acronyms = {
        "CHS": "CENTRE HOSPITALIER SPECIALISE",
        "CH": "CENTRE HOSPITALIER",
        "SDIS": "SERVICE DEPARTEMENTAL D'INCENDIE ET DE SECOURS",
        "HCL": "HOSPICES CIVILS DE LYON",
        "COM2COM": "COMMUNAUTE DE COMMUNES",
        "CABS": "COMMUNAUTE D'AGGLOMERATION DE LA BAIE DE SOMME",
        "INTERCO": "INTERCOMMUNALITE",
        "INTERCOM": "INTERCOMMUNALITE",
        "CCAS": "CENTRE COMMUNAL D'ACTION SOCIALE",
        "DPE": "DIRECTION PETITE ENFANCE",
        "SIP": "SERVICE IMPOTS PARTICULIERS",
        "GCS": "GROUPEMENT DE COOPERATION SANITAIRE",
        "GCSMS": "GROUPEMENT DE COOPERATION SOCIALE ET MEDICO SANITAIRE",
        "UENAPI": "UNAPEI"
    }
    words = text.upper().split()
    expanded = [acronyms.get(w, w) for w in words]
    return " ".join(expanded)

def _is_public_entity(crm_name, c):
    public_keywords = ["MAIRIE", "COMMUNE", "ECOLE", "ECOL", "COLLEGE", "LYCEE", "CHS", "SDIS", "HOPITAL", "PREFECTURE", "CENTRE HOSPITALIER", "SERVICE TECHNIQUE", "INTERCO", "INTERCOM", "CCAS", "DPE", "SIP", "GCS", "GCSMS", "MEDIATHEQUE", "CENTRE SOCIAL", "EHPAD"]
    crm_name_upper = crm_name.upper()
    for kw in public_keywords:
        if kw in crm_name_upper:
            return True
    if c is not None:
        naf = str(c.get("activite_principale") or "").strip().upper()
        if naf.startswith("84") or naf.startswith("85") or naf == "86.10Z":
            return True
        ln = str(c.get("legal_nature") or "").strip()
        if ln.startswith("7"):
            return True
    return False

def _transform_public_sector_input(crm_name: str, crm_city: str = "", legal_nature: str = "", naf: str = "") -> dict:
    """Version 2.2 Guarded Public Sector Input Transformation Pipeline.
    Strips generic administrative prefixes only when distinctive non-generic proper nouns exist,
    protecting commercial business names (e.g. HOTEL DE VILLE PROPERTIES).
    """
    import re
    raw_upper = crm_name.upper().strip()
    
    # Check if entity is confirmed public sector by legal nature (7xxx) or NAF (84/85)
    is_confirmed_public = (legal_nature and str(legal_nature).startswith("7")) or (naf and (str(naf).startswith("84") or str(naf).startswith("85")))
    
    # 1. Strip generic administrative prefixes if confirmed public or has explicit proper noun
    cleaned = raw_upper
    if is_confirmed_public or any(kw in raw_upper for kw in ["ECOLE", "COLLEGE", "LYCEE", "SDIS", "CHS", "CCAS"]):
        cleaned = re.sub(r"^(COMMUNE\s+D[E']|VILLE\s+D[E']|MAIRIE\s+D[E']|CANTON\s+D[E'])\s*", "", raw_upper)
    
    # 2. Extract distinctive facility tokens by removing city name
    city_norm = crm_city.upper().strip() if crm_city else ""
    if city_norm and city_norm in cleaned:
        cleaned = cleaned.replace(city_norm, "").strip()
        
    # 3. Extract proper noun / facility keyword
    generic_words = {"ECOLE", "ECOL", "ELEMENTAIRE", "MATERNELLE", "PRIMAIRE", "PUBLIQUE", "PUB", "DE", "DU", "DES", "LE", "LA", "LES", "D", "L", "BAT", "CENTRE", "SERVICE", "DIRECTION"}
    tokens = [w for w in cleaned.split() if w not in generic_words and len(w) >= 3]
    distinctive_name = " ".join(tokens) if tokens else cleaned
    
    return {
        "raw_public_name": raw_upper,
        "transformed_public_name": cleaned,
        "distinctive_public_noun": distinctive_name
    }


class OptimizedXgbInferenceEngine(XgbInferenceEngine):
    """Subclass of XgbInferenceEngine adding persistent caches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tfidf_cache = {}
        self._cand_pool_cache = {}
        self._sqlite_conn = None
        self.database_path = None  # Version 2.4 Muted SQLite SSOT
        self.disable_scraper = True  # Version 2.0 Pure Offline SSOT
        from src.xgb_matcher.tfidf_cache import TfidfPersistentCache
        sig_hash = self.retrieval_config.signature().hash
        self._persistent_cache = TfidfPersistentCache(sig_hash)

    @classmethod
    def from_profile(cls, profile):
        engine = super().from_profile(profile)
        engine.database_path = None  # Version 2.4 Muted SQLite SSOT
        engine.disable_scraper = True  # Version 2.0 Pure Offline SSOT
        return engine

    def __del__(self):
        if hasattr(self, "_sqlite_conn") and self._sqlite_conn is not None:
            try:
                self._sqlite_conn.close()
            except Exception:
                pass

    def _get_sqlite_candidate_by_id(self, identifier: str) -> List[dict]:
        import sqlite3
        if not hasattr(self, "_sqlite_conn") or self._sqlite_conn is None:
            db_path = getattr(self, "database_path", Path("data/sirene_cache.sqlite"))
            if not db_path.exists():
                db_path = Path(__file__).parent.parent / "data" / "sirene_cache.sqlite"
            if not db_path.exists():
                return []
            self._sqlite_conn = sqlite3.connect(db_path)
            self._sqlite_conn.row_factory = sqlite3.Row

        cursor = self._sqlite_conn.cursor()
        
        # Determine if it's a SIRET (14 digits) or SIREN (9 digits)
        if len(identifier) == 14:
            query = """
                SELECT siret, siren, denomination, postcode, city, insee_code,
                       legal_nature, etat_administratif, date_dernier_traitement,
                       denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                       enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                       etablissement_siege
                FROM establishments 
                WHERE siret = ?
            """
        else:  # SIREN (9 digits)
            query = """
                SELECT siret, siren, denomination, postcode, city, insee_code,
                       legal_nature, etat_administratif, date_dernier_traitement,
                       denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                       enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                       etablissement_siege
                FROM establishments 
                WHERE siren = ?
            """
        try:
            cursor.execute(query, (identifier,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[SQLite Fallback Error] Query by ID failed: {e}")
            return []

    def _search_web_scraper_siret(self, crm_name: str, city: str, postcode: str) -> str | None:
        # NUKED: Web scraper completely removed per user directive
        return None

    def clear_tfidf_cache(self):
        self._tfidf_cache.clear()
        self._cand_pool_cache.clear()

    def _build_candidate_pool(
        self,
        crm_row: Dict[str, Any],
        crm_id: str,
        crm_pre: Dict[str, Any],
        pool_mode: str,
        drop_unnamed: bool,
        exclude_closed: bool,
    ):
        cache_key = (crm_id, pool_mode, drop_unnamed, exclude_closed)
        if cache_key in self._cand_pool_cache:
            return self._cand_pool_cache[cache_key]

        if pool_mode != "insee_then_postcode":
            raise ValueError("Only pool_mode='insee_then_postcode' is supported (SSOT).")

        if self.siren_global_index is not None and self.siren_to_geo is not None:
            res = self._build_candidate_pool_v2(
                crm_row=crm_row,
                crm_id=crm_id,
                crm_pre=crm_pre,
                drop_unnamed=drop_unnamed,
                exclude_closed=exclude_closed,
            )
            self._cand_pool_cache[cache_key] = res
            return res

        config = self.retrieval_config
        if drop_unnamed != self.drop_unnamed or exclude_closed != self.exclude_closed:
            config = replace(
                config,
                drop_unnamed=drop_unnamed,
                include_closed=not exclude_closed,
            )

        result = build_candidate_pool(
            store=self.store,
            crm_row=crm_row,
            crm_pre=crm_pre,
            config=config,
            tfidf_cache=self._tfidf_cache,  # Reuse persistent engine cache
            gt_siret=None,
            persistent_cache=self._persistent_cache,
            siren_to_geo=self.siren_to_geo,
            partition_cache=getattr(self, "_partition_cache", None),
        )
        # Apply Laplace Reference Normalization (N_ref = 10,000)
        import math
        laplace_idf_map = {}
        pool_size = max(1, len(result.candidates))
        for tok, raw_val in result.idf_map.items():
            # Convert raw count to Laplace normalized IDF
            local_freq = max(1.0, float(raw_val))
            freq_ratio = local_freq / pool_size
            laplace_idf_map[tok] = float(math.log(10000.0 / (1.0 + freq_ratio * 10000.0)))
            
        default_laplace_idf = float(math.log(10000.0 / (1.0 + (1.0 / pool_size) * 10000.0)))
        res = (result.candidates, laplace_idf_map, default_laplace_idf)
        self._cand_pool_cache[cache_key] = res

        # Evict old entries to prevent memory growth
        if len(self._cand_pool_cache) > 1000:
            first_key = next(iter(self._cand_pool_cache))
            self._cand_pool_cache.pop(first_key)

        return res

    def _map_datagouv_candidate(self, cand):
        extra = cand.extra or {}
        raw_comp = extra.get("raw", {})
        siret = cand.siret
        
        postcode = None
        city = None
        insee = None
        street_number = None
        street_type = None
        street_name = None
        is_siege = False
        
        siege = raw_comp.get("siege") or {}
        if siege.get("siret") == siret:
            postcode = siege.get("code_postal")
            city = siege.get("libelle_commune")
            insee = siege.get("code_commune")
            street_number = siege.get("numero_voie")
            street_type = siege.get("type_voie")
            street_name = siege.get("libelle_voie")
            is_siege = True
        else:
            for etab in raw_comp.get("matching_etablissements") or []:
                if etab.get("siret") == siret:
                    postcode = etab.get("code_postal")
                    city = etab.get("libelle_commune")
                    insee = etab.get("code_commune")
                    street_number = etab.get("numero_voie")
                    street_type = etab.get("type_voie")
                    street_name = etab.get("libelle_voie")
                    is_siege = etab.get("est_siege", False)
                    break
                    
        return {
            "siret": siret,
            "siren": cand.siren,
            "denomination": cand.label,
            "postcode": postcode,
            "city": city,
            "insee_code": insee,
            "legal_nature": raw_comp.get("nature_juridique"),
            "etat_administratif": raw_comp.get("etat_administratif"),
            "etat_admin": raw_comp.get("etat_administratif"),
            "date_dernier_traitement": None,
            "denomination_unite_legale": raw_comp.get("nom_complet") or raw_comp.get("nom_raison_sociale"),
            "nom_unite_legale": None,
            "prenom1_unite_legale": None,
            "enseigne1": None,
            "enseigne2": None,
            "enseigne3": None,
            "street_number": street_number,
            "street_type": street_type,
            "street_name": street_name,
            "etablissement_siege": 1 if is_siege else 0
        }

    def _search_sqlite_candidates(self, crm_input, postcode: str | None) -> List[dict]:
        import sqlite3
        import re

        if not hasattr(self, "_sqlite_conn") or self._sqlite_conn is None:
            db_path = getattr(self, "database_path", Path("data/sirene_cache.sqlite"))
            if not db_path.exists():
                db_path = Path(__file__).parent.parent / "data" / "sirene_cache.sqlite"

            if not db_path.exists():
                print(f"[SQLite Fallback Warning] Database not found at {db_path}")
                return []
            self._sqlite_conn = sqlite3.connect(db_path)
            self._sqlite_conn.row_factory = sqlite3.Row

        cursor = self._sqlite_conn.cursor()

        # Clean the name
        cleaned = crm_input.crm_name.upper()
        if "UENAPI" in cleaned:
            cleaned = cleaned.replace("UENAPI", "UNAPEI")
            
        # Rule 1: Strip internal account / RIB numbers (6+ digits)
        cleaned = re.sub(r"\b\d{6,}\b", "", cleaned)

        # Rule 2: Strip domestic and foreign corporate legal suffixes
        forms = [
            r"\bSAS\b", r"\bSARL\b", r"\bEURL\b", r"\bSA\b", r"\bSCI\b", r"\bSNC\b", r"\bEI\b", r"\bEIRL\b",
            r"\bAG\b", r"\bGMBH\b", r"\bLTD\b", r"\bINC\b", r"\bLLC\b", r"\bPLC\b", r"\bBV\b", r"\bSPA\b"
        ]
        for f in forms:
            cleaned = re.sub(f, "", cleaned)

        # Rule 3: Strip site/local prefixes and appended city names
        if crm_input.crm_city:
            city_norm = re.sub(r"[^\w\s]", " ", crm_input.crm_city.upper()).strip()
            if city_norm and len(city_norm) >= 3 and cleaned.endswith(city_norm):
                cleaned = cleaned[:-len(city_norm)].strip()

        cleaned = re.sub(r"\bSITE\s+\w+\b", "", cleaned)
        cleaned = re.sub(r"\bLOCAL\s+\w+\b", "", cleaned)
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Generate expanded search terms for SQLite fallback contains matching
        from src.xgb_matcher.naming import expand_abbreviations
        expanded = expand_abbreviations(cleaned)
        
        words = list(dict.fromkeys(cleaned.split() + expanded.split()))
        if not words:
            return []

        prefixes = []
        orig_words = cleaned.split()
        if orig_words:
            if len(orig_words) >= 2:
                prefixes.append(" ".join(orig_words[:2]))
            prefixes.append(orig_words[0])

        # Feature 13.6: Property / Copropriété Prefix Expansion (RESIDENCE, ESPACE, DOMAINE, PARC, IMMEUBLE)
        upper_raw_name = crm_input.crm_name.upper()
        for prop_kw in ["RESIDENCE", "ESPACE", "DOMAINE", "PARC", "IMMEUBLE", "CLOS"]:
            if prop_kw in upper_raw_name:
                prefixes.append(f"SYNDICAT DES COPROPRIETAIRES DE LA {cleaned}")
                prefixes.append(f"SYNDICAT DE COPROPRIETE {cleaned}")
                prefixes.append(f"COPROPRIETE {cleaned}")
                prefixes.append(f"SYNDICAT DES COPROPRIETAIRES {cleaned}")
                break

        dept = postcode[:2] if postcode and len(postcode) >= 2 else None

        # Feature 13.1: Pre-fetch linked INSEE codes associated with CRM postcode
        linked_insee_codes = []
        if postcode:
            try:
                cursor.execute("SELECT DISTINCT insee_code FROM establishments WHERE postcode = ?", (postcode,))
                linked_insee_codes = [r[0] for r in cursor.fetchall() if r[0]]
            except Exception:
                linked_insee_codes = []

        # Feature 13.2: Fallback to City-based INSEE lookup if postcode is missing
        if not postcode and crm_input.crm_city:
            city_clean = re.sub(r"[^\w\s]", " ", crm_input.crm_city.upper()).strip()
            if city_clean:
                try:
                    cursor.execute("SELECT DISTINCT insee_code FROM establishments WHERE city = ?", (city_clean,))
                    linked_insee_codes = [r[0] for r in cursor.fetchall() if r[0]]
                except Exception:
                    linked_insee_codes = []

        def run_search(postcode_val, dept_val, insee_list):
            res_rows = []
            seen = set()

            for prefix in prefixes:
                if len(prefix) < 3:
                    continue
                
                # Priority 1: Exact Postcode match
                if postcode_val:
                    query = """
                        SELECT siret, siren, denomination, postcode, city, insee_code,
                               legal_nature, etat_administratif, date_dernier_traitement,
                               denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                               enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                               etablissement_siege
                        FROM establishments 
                        WHERE postcode = ? AND denomination_ci LIKE ? 
                        LIMIT 50
                    """
                    params = (postcode_val, f"{prefix}%")
                    try:
                        rows_pre = cursor.execute(query, params).fetchall()
                        for r in rows_pre:
                            if r[0] not in seen:
                                seen.add(r[0])
                                res_rows.append(r)
                    except Exception as e:
                        print(f"[SQLite Fallback Error] Prefix search failed: {e}")

                # Priority 2: Restricted Linked INSEE codes match (same commune)
                if insee_list:
                    placeholders = ",".join("?" for _ in insee_list)
                    query = f"""
                        SELECT siret, siren, denomination, postcode, city, insee_code,
                               legal_nature, etat_administratif, date_dernier_traitement,
                               denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                               enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                               etablissement_siege
                        FROM establishments 
                        WHERE insee_code IN ({placeholders}) AND denomination_ci LIKE ? 
                        LIMIT 50
                    """
                    params = (*insee_list, f"{prefix}%")
                    try:
                        rows_insee = cursor.execute(query, params).fetchall()
                        for r in rows_insee:
                            if r[0] not in seen:
                                seen.add(r[0])
                                res_rows.append(r)
                    except Exception as e:
                        print(f"[SQLite Fallback Error] INSEE-restricted search failed: {e}")

                if res_rows:
                    return res_rows

                # Priority 3: Department-level fallback
                if dept_val:
                    query = """
                        SELECT siret, siren, denomination, postcode, city, insee_code,
                               legal_nature, etat_administratif, date_dernier_traitement,
                               denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                               enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                               etablissement_siege
                        FROM establishments 
                        WHERE insee_code LIKE ? AND denomination_ci LIKE ? 
                        LIMIT 50
                    """
                    params = (f"{dept_val}%", f"{prefix}%")
                    try:
                        rows_dept = cursor.execute(query, params).fetchall()
                        if rows_dept:
                            return rows_dept
                    except Exception as e:
                        print(f"[SQLite Fallback Error] Department search failed: {e}")

            # Contains search
            if postcode_val and len(words) >= 1:
                for w in words:
                    if len(w) >= 4:
                        query = """
                            SELECT siret, siren, denomination, postcode, city, insee_code,
                                   legal_nature, etat_administratif, date_dernier_traitement,
                                   denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                                   enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                                   etablissement_siege
                            FROM establishments 
                            WHERE postcode = ? AND denomination_ci LIKE ? 
                            LIMIT 50
                        """
                        try:
                            rows_cnt = cursor.execute(query, (postcode_val, f"%{w}%")).fetchall()
                            if rows_cnt:
                                return rows_cnt
                        except Exception as e:
                            print(f"[SQLite Fallback Error] Contains search failed: {e}")
            return []

        # Cascade search (name-based)
        rows = []
        if postcode:
            rows = run_search(postcode_val=postcode, dept_val=None, insee_list=linked_insee_codes)
        elif linked_insee_codes:
            rows = run_search(postcode_val=None, dept_val=None, insee_list=linked_insee_codes)
        if not rows and dept:
            rows = run_search(postcode_val=None, dept_val=dept, insee_list=None)
        if not rows:
            rows = run_search(postcode_val=None, dept_val=None, insee_list=None)

        # 3. Address-based search fallback (postcode + street name + number)
        addr_rows = []
        crm_addr = crm_input.crm_address or ""
        street_number = ""
        tokens = crm_addr.split()
        if tokens:
            first_tok = tokens[0].replace(",", "").replace(".", "").strip()
            if first_tok.isdigit():
                street_number = first_tok

        from src.xgb_matcher.local_heuristics import clean_street_name
        street_name_clean = clean_street_name(crm_addr)

        if postcode and street_name_clean:
            pc_str = str(postcode).strip().zfill(5)
            addr_query = """
                SELECT siret, siren, denomination, postcode, city, insee_code,
                       legal_nature, etat_administratif, date_dernier_traitement,
                       denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                       enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                       etablissement_siege
                FROM establishments 
                WHERE postcode = ? AND street_name LIKE ?
                LIMIT 50
            """
            try:
                found_addr = cursor.execute(addr_query, (pc_str, f"%{street_name_clean}%")).fetchall()
                if found_addr:
                    if street_number:
                        matching_num = [r for r in found_addr if str(r[15]).strip() == street_number]
                        if matching_num:
                            found_addr = matching_num + [r for r in found_addr if r not in matching_num]
                    addr_rows = found_addr
            except Exception as e:
                print(f"[Address Fallback Error] {e}")

        # Merge name-based and address-based pools
        seen_sirets = set()
        merged_rows = []
        for r in rows + addr_rows:
            siret = r[0]
            if siret not in seen_sirets:
                seen_sirets.add(siret)
                merged_rows.append(r)
        rows = merged_rows

        # SIREN Expansion logic:
        if rows:
            sirens = list(set([r[1] for r in rows if r[1]]))
            expanded_rows = []
            seen_sirets = set()
            for siren in sirens:
                query_exp = """
                    SELECT siret, siren, denomination, postcode, city, insee_code,
                           legal_nature, etat_administratif, date_dernier_traitement,
                           denomination_unite_legale, nom_unite_legale, prenom1_unite_legale,
                           enseigne1, enseigne2, enseigne3, street_number, street_type, street_name,
                           etablissement_siege
                    FROM establishments 
                    WHERE siren = ?
                    LIMIT 100
                """
                try:
                    sib_rows = cursor.execute(query_exp, (siren,)).fetchall()
                except Exception as e:
                    print(f"[SQLite Fallback Error] SIREN expansion query failed: {e}")
                    sib_rows = []
                for r in sib_rows:
                    siret = r[0]
                    if siret not in seen_sirets:
                        seen_sirets.add(siret)
                        expanded_rows.append(r)
            if expanded_rows:
                rows = expanded_rows

        # Map SQL rows to candidate dicts
        candidates = []
        for r in rows:
            cand = {
                "siret": str(r[0] or ""),
                "siren": str(r[1] or ""),
                "denomination": r[2] or "",
                "postcode": str(r[3] or ""),
                "city": r[4] or "",
                "insee": str(r[5] or ""),
                "cj_ul": str(r[6] or ""),
                "etat_admin": str(r[7] or ""),
                "last_treatment_date": r[8] or "",
                "denomination_ul": r[9] or "",
                "nom_ul": r[10] or "",
                "prenom_usuel_ul": r[11] or "",
                "enseigne1": r[12] or "",
                "enseigne2": r[13] or "",
                "enseigne3": r[14] or "",
                "numeroVoie": r[15] or "",
                "typeVoie": r[16] or "",
                "libelleVoie": r[17] or "",
                "etablissementSiege": bool(r[18]),
                "is_siege": bool(r[18]),
                "sigle_ul": "",
                "denomination_usuelle_ul": "",
                "pm_dirigeant_names": "",
                "complementAdresse": ""
            }
            candidates.append(cand)
        return candidates

    def _get_naf_codes(self, sirets: List[str]) -> Dict[str, str]:
        if not sirets:
            return {}
        if not hasattr(self, "_sqlite_conn") or self._sqlite_conn is None:
            db_path = Path("data/sirene_cache.sqlite")
            if not db_path.exists():
                db_path = Path(__file__).parent.parent / "data" / "sirene_cache.sqlite"
            if not db_path.exists():
                return {}
            import sqlite3
            self._sqlite_conn = sqlite3.connect(db_path)
            self._sqlite_conn.row_factory = sqlite3.Row
        cursor = self._sqlite_conn.cursor()
        placeholders = ",".join("?" for _ in sirets)
        query = f"SELECT siret, activite_principale FROM establishments WHERE siret IN ({placeholders})"
        try:
            rows = cursor.execute(query, sirets).fetchall()
            return {r[0]: r[1] for r in rows if r[0] and r[1]}
        except Exception:
            return {}

    def _filter_fallback_candidates(self, candidates: List[dict], drop_unnamed: bool, exclude_closed: bool) -> List[dict]:
        out = []
        for c in candidates:
            if drop_unnamed:
                has_name = any([
                    c.get("denomination"),
                    c.get("denomination_ul"),
                    c.get("enseigne1"),
                    c.get("enseigne2"),
                    c.get("enseigne3"),
                    c.get("denomination_usuelle_ul"),
                    c.get("sigle_ul"),
                    c.get("nom_ul"),
                    c.get("prenom_usuel_ul"),
                ])
                if not has_name:
                    continue
            if exclude_closed:
                if c.get("etat_admin") == "F":
                    continue
            out.append(c)
        return out

    def infer_topk(
        self,
        crm_input: CrmInput,
        top_k: int = 5,
        pool_mode: str = "insee_then_postcode",
        drop_unnamed: bool | None = None,
        exclude_closed: bool | None = None,
        export_routing_features: bool = True,
    ) -> List[TopKRow]:
        res = self.infer_topk_batch(
            [crm_input],
            top_k=top_k,
            pool_mode=pool_mode,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
            export_routing_features=export_routing_features,
        )
        return res[0] if res else []

    def infer_topk_batch(
        self,
        crm_inputs: List[CrmInput],
        top_k: int = 5,
        pool_mode: str = "insee_then_postcode",
        drop_unnamed: bool | None = None,
        exclude_closed: bool | None = None,
        export_routing_features: bool = True,
    ) -> List[List[TopKRow]]:
        from src.xgb_matcher.semantic import _semantic_enabled, batch_encode_texts
        from src.xgb_matcher.features import (
            normalize_text,
            build_semantic_name_pool,
            make_features_from_preprocessed,
            set_global_name_idf_map,
            semantic_gate_allows,
            build_address,
        )
        from src.xgb_matcher.naming import build_candidate_names, primary_name
        from src.xgb_matcher.infer import TopKRow, _has_name_evidence
        import time
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        if drop_unnamed is None:
            drop_unnamed = self.drop_unnamed
        if exclude_closed is None:
            exclude_closed = self.exclude_closed

        # 1. Gather all candidate pools
        batch_state = []
        t_io = 0.0
        for crm_input in crm_inputs:
            crm_row = crm_input.to_dict()
            
            # Feature 13.4: Auto-fill missing INSEE in crm_row via Postcode-to-INSEE or City-to-INSEE mapping
            if not crm_row.get("insee"):
                postcode = crm_row.get("postcode")
                city = crm_row.get("crm_city") or crm_row.get("city")
                found_insee = None
                
                if postcode:
                    try:
                        if not hasattr(self, "_sqlite_conn") or self._sqlite_conn is None:
                            import sqlite3
                            db_path = getattr(self, "database_path", Path("data/sirene_cache.sqlite"))
                            if not db_path.exists():
                                db_path = Path(__file__).parent.parent / "data" / "sirene_cache.sqlite"
                            if db_path.exists():
                                self._sqlite_conn = sqlite3.connect(db_path)
                                self._sqlite_conn.row_factory = sqlite3.Row
                        if hasattr(self, "_sqlite_conn") and self._sqlite_conn:
                            cursor = self._sqlite_conn.cursor()
                            cursor.execute("SELECT DISTINCT insee_code FROM establishments WHERE postcode = ? LIMIT 1", (postcode,))
                            row_insee = cursor.fetchone()
                            if row_insee and row_insee[0]:
                                found_insee = str(row_insee[0])
                    except Exception:
                        pass
                
                if not found_insee and city:
                    import re
                    city_clean = re.sub(r"[^\w\s]", " ", str(city).upper()).strip()
                    if city_clean and hasattr(self, "_sqlite_conn") and self._sqlite_conn:
                        try:
                            cursor = self._sqlite_conn.cursor()
                            cursor.execute("SELECT DISTINCT insee_code FROM establishments WHERE city = ? LIMIT 1", (city_clean,))
                            row_insee = cursor.fetchone()
                            if row_insee and row_insee[0]:
                                found_insee = str(row_insee[0])
                        except Exception:
                            pass
                
                if found_insee:
                    crm_row["insee"] = found_insee

            crm_pre = preprocess_crm_row(crm_row)
            
            t0 = time.time()
            candidates, idf_map, default_idf = self._build_candidate_pool(
                crm_row=crm_row,
                crm_id=crm_input.crm_id,
                crm_pre=crm_pre,
                pool_mode=pool_mode,
                drop_unnamed=drop_unnamed,
                exclude_closed=exclude_closed,
            )
            t_io += (time.time() - t0)
            
            batch_state.append({
                "crm_input": crm_input,
                "crm_pre": crm_pre,
                "candidates": candidates,
                "idf_map": idf_map,
                "default_idf": default_idf,
                "cand_list": [],
                "feats_stage1": [],
                "top_n_idx": []
            })
        print(f"[Batched Infer] Step 1 (I/O & Candidate Pools): {t_io:.2f}s total for {len(crm_inputs)} items")

        # 2. Run Stage 1 ranking + gather texts to encode
        all_texts_to_encode = []
        t_stage1 = 0.0
        
        for state in batch_state:
            if not state["candidates"]:
                continue
            crm_pre = state["crm_pre"]
            candidates = state["candidates"]
            
            crm_name_sem = crm_pre.get("crm_name_semantic", "")
            if crm_name_sem and _semantic_enabled():
                all_texts_to_encode.append(crm_name_sem)
                
            cand_list = [(c.get("siret"), c) for c in candidates if c.get("siret")]
            if not cand_list:
                continue
                
            t0 = time.time()
            set_global_name_idf_map(state["idf_map"], state["default_idf"])
            feats_stage1 = [
                make_features_from_preprocessed(crm_pre, c, skip_semantic=True)
                for _, c in cand_list
            ]
            
            ranker_feature_order = self.ranker_feature_order or self.feature_order
            X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
            scores_stage1 = self.ranker.predict(
                xgb.DMatrix(X1.values, feature_names=ranker_feature_order)
            )
            
            stage1_top_n = min(self.stage1_top_n, len(scores_stage1))
            top_n_idx = np.argsort(scores_stage1)[::-1][:stage1_top_n]
            t_stage1 += (time.time() - t0)
            
            state["cand_list"] = cand_list
            state["feats_stage1"] = feats_stage1
            state["top_n_idx"] = top_n_idx

            if _semantic_enabled():
                for idx in top_n_idx:
                    feat = feats_stage1[idx]
                    if not semantic_gate_allows(feat.get("name_jaro_max", 0.0), feat.get("name_token_overlap_max", 0.0)):
                        continue
                    _, c = cand_list[idx]
                    cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                    pool = build_semantic_name_pool(
                        build_candidate_names(c),
                        crm_city_norm=crm_pre.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                    all_texts_to_encode.extend(pool)
                    
        print(f"[Batched Infer] Step 2 (Stage 1 XGBoost): {t_stage1:.2f}s total for {len(crm_inputs)} items")

        # 3. Batch semantic encoding on CPU/GPU
        if _semantic_enabled() and all_texts_to_encode:
            print(f"[Batched Infer] Sending {len(all_texts_to_encode)} unique texts to Encoder...")
            t0 = time.time()
            batch_encode_texts(all_texts_to_encode)
            print(f"[Batched Infer] Finished semantic encoding in {time.time() - t0:.2f}s")

        # 4. Stage 2 scoring (very fast sequential since Stage 1 and embeddings are pre-computed)
        results = []
        for state in batch_state:
            crm_input = state["crm_input"]
            crm_pre = state["crm_pre"]
            cand_list = state["cand_list"]
            feats_stage1 = state["feats_stage1"]
            top_n_idx = state["top_n_idx"]
            
            if not cand_list or len(top_n_idx) == 0:
                results.append([])
                continue
                
            # Stage 2: Decider (with semantic)
            feats_n = []
            cand_list_n = []
            semantic_pools = []
            
            set_global_name_idf_map(state["idf_map"], state["default_idf"])
            
            for idx in top_n_idx:
                siret, c = cand_list[idx]
                feat = feats_stage1[idx].copy()
                cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                
                pool = []
                if _semantic_enabled():
                    pool = build_semantic_name_pool(
                        build_candidate_names(c),
                        crm_city_norm=crm_pre.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                semantic_pools.append(pool)
                p_name = primary_name(c)
                gen_enseignes = {"LABORATOIRE", "BOUTIQUE", "MAGASIN", "AGENCE", "CABINET", "CENTRE", "BUREAU", "SITE", "SERVICE", "GARAGE", "HOTEL", "RESTAURANT"}
                if p_name and p_name.upper().strip() in gen_enseignes:
                    p_name = str(c.get("denomination_unite_legale") or c.get("denomination") or p_name)

                # Archetype 3 Fix: Max Independent Enseigne & Legal Unit Name Scoring
                from rapidfuzz.distance import JaroWinkler
                crm_norm_eval = crm_pre.get("crm_name", "").upper()
                if crm_norm_eval:
                    for ens_key in ["enseigne1", "enseigne2", "enseigne3", "denomination_ul", "denomination"]:
                        ens_val = str(c.get(ens_key) or "").upper().strip()
                        if ens_val and len(ens_val) >= 3:
                            ens_sim = float(JaroWinkler.similarity(crm_norm_eval, ens_val))
                            if ens_sim > feat.get("name_jaro_max", 0.0):
                                feat["name_jaro_max"] = ens_sim

                feat["_siret"] = siret
                feat["_cand_name"] = p_name or f"SIRET {siret}"
                feats_n.append(feat)
                cand_list_n.append((siret, c))
                
            # Add semantic features
            from src.xgb_matcher.semantic import top2_semantic_similarities_batch
            sem = top2_semantic_similarities_batch(crm_pre.get("crm_name_semantic", ""), semantic_pools)
            for feat, (sem_max, sem_second, sem_gap) in zip(feats_n, sem, strict=True):
                if semantic_gate_allows(
                    feat.get("name_jaro_max", 0.0),
                    feat.get("name_token_overlap_max", 0.0),
                ):
                    feat["name_semantic_max"] = sem_max
                    feat["name_semantic_second"] = sem_second
                    feat["name_semantic_gap"] = sem_gap
                else:
                    feat["name_semantic_max"] = 0.0
                    feat["name_semantic_second"] = 0.0
                    feat["name_semantic_gap"] = 0.0

            # Feature 13.5 & 13.7: Sub-Entity Delimiter, Syndicat Prefix Stripping & Substring Similarity
            from rapidfuzz.distance import JaroWinkler
            crm_name_upper = crm_input.crm_name.upper().strip()
            core_crm_name = _extract_core_company_name(crm_input.crm_name).upper().strip()
            syndic_prefixes = [
                "SYNDICAT DES COPROPRIETAIRES DE LA ",
                "SYNDICAT DES COPROPRIETAIRES DU ",
                "SYNDICAT DES COPROPRIETAIRES DE ",
                "SYNDICAT DES COPROPRIETAIRES ",
                "SYNDICAT DE COPROPRIETE ",
                "COPROPRIETE "
            ]

            for feat, (_, c) in zip(feats_n, cand_list_n, strict=True):
                cand_name_text = str(c.get("denomination") or c.get("denomination_ul") or "").upper().strip()
                if cand_name_text:
                    clean_cand_name = cand_name_text
                    for pref in syndic_prefixes:
                        if clean_cand_name.startswith(pref):
                            clean_cand_name = clean_cand_name[len(pref):].strip()
                            break

                    sim_full = float(JaroWinkler.similarity(crm_name_upper, clean_cand_name))
                    sim_core = float(JaroWinkler.similarity(core_crm_name, clean_cand_name))
                    max_sim = max(sim_full, sim_core)

                    # Evaluate candidate enseigne1/2 for municipal facilities sharing legal names (e.g. COMMUNE D'AUXERRE)
                    for ens_field in ["enseigne1", "enseigne2", "enseigne3"]:
                        ens_val = str(c.get(ens_field) or "").upper().strip()
                        if ens_val and len(ens_val) >= 4:
                            sim_ens = float(JaroWinkler.similarity(crm_name_upper, ens_val))
                            sim_ens_core = float(JaroWinkler.similarity(core_crm_name, ens_val))
                            max_sim = max(max_sim, sim_ens, sim_ens_core)

                    # Token permutation handling (e.g. LABORATOIRE EUROFINS vs EUROFINS LABORATOIRE)
                    crm_tokens = set(core_crm_name.split()) - {"DE", "DU", "DES", "LE", "LA", "LES", "D", "L", "SUD", "NORD", "EST", "OUEST"}
                    cand_tokens = set(clean_cand_name.split())
                    if crm_tokens and crm_tokens.issubset(cand_tokens):
                        max_sim = max(max_sim, 0.95)
                        feat["name_first_word_match_max"] = 1.0

                    if (len(crm_name_upper) >= 6 and crm_name_upper in cand_name_text) or (len(clean_cand_name) >= 6 and clean_cand_name in crm_name_upper):
                        max_sim = max(max_sim, 0.95)

                    if max_sim > feat.get("name_jaro_max", 0.0):
                        feat["name_jaro_max"] = max_sim

            # Stage 2 scoring
            X_n = pd.DataFrame(feats_n)[self.feature_order]
            
            probs = self.decider.predict_proba(X_n.values)[:, 1]
            if self.calibrator is not None:
                probs = self.calibrator.predict_proba(X_n.values)[:, 1]
                
            scores = np.array(probs)
            
            # Copropriété Exclusion Guard for Non-Property Entities
            crm_name_upper_guard = crm_input.crm_name.upper().strip()
            is_property_query = any(p in crm_name_upper_guard for p in ["COPROPRIETE", "SYNDICAT", "RESIDENCE", "ESPACE", "DOMAINE", "IMMEUBLE", "SOCIETE CIVILE"])
            if not is_property_query:
                for idx_c, (siret, c) in enumerate(cand_list_n):
                    cand_name_guard = str(c.get("denomination") or c.get("denomination_ul") or "").upper().strip()
                    cand_cj = str(c.get("legal_nature") or c.get("cj_ul") or "").strip()
                    if cand_cj == "9110" or any(cand_name_guard.startswith(prefix) for prefix in ["SYND ", "SYNDICAT ", "COPROPRIETE ", "COPROP "]):
                        scores[idx_c] = scores[idx_c] * 0.10  # Penalize copropriété candidates by 90% for non-property queries
                        
            # Apply APE/NAF Category Alignment Boost/Guard
            sirets_for_query = [siret for siret, _ in cand_list_n]
            naf_map = self._get_naf_codes(sirets_for_query)
            
            crm_name = crm_pre.get("crm_name", "")
            from src.xgb_matcher.naf_categories import get_expected_category, is_naf_aligned
            expected_cat = get_expected_category(crm_name)
            if expected_cat:
                for idx_c, (siret, _) in enumerate(cand_list_n):
                    naf_code = naf_map.get(siret)
                    aligned = is_naf_aligned(expected_cat, naf_code)
                    if aligned is True:
                        # Boost: bring closer to 1.0 (e.g. +0.15 relative boost)
                        old_s = scores[idx_c]
                        scores[idx_c] = old_s + 0.15 * (1.0 - old_s)
                    elif aligned is False:
                        # Penalize: reduce score by 50%
                        scores[idx_c] = scores[idx_c] * 0.5
                        
            # Apply Branch Address-Match Anchor Boost & High Similarity Street Boost
            for idx_c, (siret, _) in enumerate(cand_list_n):
                feat_row = feats_n[idx_c]
                cand_obj = cand_list_n[idx_c][1]
                street_num_diff = feat_row.get("street_number_diff", 9999)
                street_name_jaro = feat_row.get("street_name_jaro", 0.0)
                city_match = feat_row.get("city_match", 0.0)
                name_jaro_max = feat_row.get("name_jaro_max", 0.0)
                postcode_match = feat_row.get("postcode_match", 0.0)

                # Feature 13.5: Structural Branch Delimiter Sub-Entity Similarity Adjustment
                core_crm_name = _extract_core_company_name(crm_input.crm_name)
                if core_crm_name and core_crm_name != crm_input.crm_name:
                    from rapidfuzz.distance import JaroWinkler
                    cand_name_text = str(cand_obj.get("denomination") or cand_obj.get("denomination_ul") or "").upper()
                    if cand_name_text:
                        core_sim = float(JaroWinkler.similarity(core_crm_name.upper(), cand_name_text))
                        if core_sim > name_jaro_max:
                            name_jaro_max = core_sim
                            feat_row["name_jaro_max"] = core_sim

                # Feature 13.3: Restricted Postcode-to-INSEE Soft Match Adjustment
                cand_insee_code = str(cand_obj.get("insee") or cand_obj.get("insee_code") or "").strip()
                crm_pc = str(crm_input.postcode or "").strip()
                
                insee_commune_match = False
                if crm_pc and cand_insee_code:
                    try:
                        cursor.execute("SELECT 1 FROM establishments WHERE postcode = ? AND insee_code = ? LIMIT 1", (crm_pc, cand_insee_code))
                        if cursor.fetchone():
                            insee_commune_match = True
                    except Exception:
                        pass
                
                if postcode_match < 0.5 and insee_commune_match:
                    postcode_match = 0.90
                    feat_row["postcode_match"] = 0.90
                
                # Rule 1: Exact address match for a branch/site/office/public facility entity
                is_address_perfect = (
                    (street_num_diff == 0 or street_num_diff == 9999) and 
                    street_name_jaro >= 0.95 and 
                    (city_match >= 0.5 or postcode_match >= 0.5 or insee_commune_match)
                )
                if is_address_perfect:
                    crm_name_upper = crm_name.upper()
                    crm_words = set(crm_name_upper.replace("/", " ").replace("-", " ").split())
                    site_kws = [
                        "AGENCE", "SITE", "BUREAU", "ANTENNE", "CENTRE", "ETABLISSEMENT", 
                        "HOPITAL", "CLINIQUE", "MATERNITE", "DISPENSAIRE", "CMP", "ECOLE", 
                        "COLLEGE", "LYCEE", "UNIVERSITE", "SCEL", "SCST", "STERILISATION",
                        "HOTEL", "MAISON", "POLE", "ZOO", "PREFECTURE", "SIEGE", "SERVICE",
                        "EHPAD", "MUNICIPALE"
                    ]
                    has_branch_kw = any(kw in crm_name_upper for kw in site_kws) or any(w in crm_words for w in ["CH", "RPC", "STEP", "CDER", "MDSI", "CABS", "CCAS", "DSI"])
                    
                    has_name_evidence = (
                        feat_row.get("name_token_overlap_max", 0.0) > 0.0 or 
                        feat_row.get("name_semantic_max", 0.0) >= 0.50
                    )
                    
                    if has_branch_kw and has_name_evidence:
                        old_s = scores[idx_c]
                        scores[idx_c] = max(old_s + 0.35 * (1.0 - old_s), 0.80)
                        
                # Rule 2: High name similarity + matching street
                is_high_name_sim = (
                    name_jaro_max >= 0.85 and 
                    (postcode_match >= 0.5 or insee_commune_match) and 
                    street_name_jaro >= 0.95 and 
                    street_num_diff <= 10
                )
                if is_high_name_sim:
                    old_s = scores[idx_c]
                    scores[idx_c] = max(old_s + 0.35 * (1.0 - old_s), 0.80)

                # Rule 2B: Modest name similarity (>=0.80) + perfect address match
                is_exact_address_match = (
                    name_jaro_max >= 0.80 and
                    (postcode_match >= 0.5 or insee_commune_match) and
                    street_name_jaro >= 0.90 and
                    street_num_diff == 0
                )
                if is_exact_address_match:
                    old_s = scores[idx_c]
                    scores[idx_c] = max(old_s + 0.35 * (1.0 - old_s), 0.80)

                # Rule 3: Near-perfect name match + matching city/postcode/INSEE commune
                is_perfect_name_city = (
                    name_jaro_max >= 0.95 and 
                    (city_match >= 0.5 or postcode_match >= 0.5 or insee_commune_match)
                )
                if is_perfect_name_city:
                    old_s = scores[idx_c]
                    scores[idx_c] = max(old_s + 0.35 * (1.0 - old_s), 0.80)

                # Rule 4: Exact Enseigne / Branch Title Anchor Boost for municipal facilities
                from rapidfuzz.distance import JaroWinkler
                from src.xgb_matcher.naming import normalize_text
                ens_val = str(cand_obj.get("enseigne1") or cand_obj.get("enseigne_1") or cand_obj.get("enseigne") or "").strip()
                crm_name_raw = crm_input.crm_name.strip()
                if ens_val and len(ens_val) >= 4:
                    norm_crm_name = normalize_text(crm_name_raw)
                    norm_ens_val = normalize_text(ens_val)
                    
                    # Extract distinctive proper noun words (exclude generic school/facility keywords)
                    generic_kws = {"ECOLE", "ECOL", "ELEMENTAIRE", "MATERNELLE", "PRIMAIRE", "PUBLIQUE", "PUB", "DE", "DU", "DES", "LE", "LA", "LES", "D", "L", "BAT", "BATIMENT", "SITE"}
                    crm_words = set(norm_crm_name.split()) - generic_kws
                    ens_words = set(norm_ens_val.split())
                    
                    # Require that at least one distinctive proper noun token matches
                    has_distinctive_match = len(crm_words.intersection(ens_words)) > 0 if crm_words else False
                    ens_sim = float(JaroWinkler.similarity(norm_crm_name, norm_ens_val))
                    
                    if has_distinctive_match and ens_sim >= 0.82:
                        old_s = scores[idx_c]
                        base_boost = max(old_s + 0.25 * (1.0 - old_s), 0.980)
                        
                        # Sub-Type Level Tie-Breaker (ELEMENTAIRE vs MATERNELLE)
                        level_bonus = 0.00
                        if "ELEMENTAIRE" in norm_crm_name and "ELEMENTAIRE" in norm_ens_val:
                            level_bonus = 0.008
                        elif ("MATERNELLE" in norm_crm_name or "MATE" in norm_crm_name) and ("MATERNELLE" in norm_ens_val or "MATE" in norm_ens_val):
                            level_bonus = 0.008
                            
                        scores[idx_c] = min(0.995, base_boost + level_bonus)
                        
            pool_size_stage1 = len(cand_list)
            pool_size_stage2 = len(scores)
            scores_sorted = np.sort(scores)[::-1]
            top3_avg = float(np.mean(scores_sorted[:3])) if len(scores_sorted) >= 3 else float(np.mean(scores_sorted)) if len(scores_sorted) else 0.0
            
            sorted_idx = np.argsort(scores)[::-1]
            topk_idx = sorted_idx[:top_k].tolist()
            topk_idx = self._promote_open_over_closed(topk_idx, cand_list_n)
            
            top1_score = float(scores[topk_idx[0]]) if topk_idx else 0.0
            top2_score = float(scores[topk_idx[1]]) if len(topk_idx) > 1 else 0.0
            score_gap = top1_score - top2_score
            denom = top2_score if top2_score > 1e-6 else 0.001
            score_ratio = top1_score / denom
            
            # Build output rows
            rows = []
            for rank, idx_k in enumerate(topk_idx, start=1):
                siret_k, cand_k = cand_list_n[idx_k]
                etat_admin = str(cand_k.get("etat_admin") or "").strip().upper() or None
                candidate_state = None
                if etat_admin is not None:
                    candidate_state = "FERME" if etat_admin == "F" else "OUVERT"
                feat_row = feats_n[idx_k]
                
                routing_confidence = None
                routing_status = None
                if rank == 1:
                    if self.risk_model is not None and self.risk_features:
                        risk_dict = feat_row.copy()
                        risk_dict["score_top1"] = top1_score
                        risk_dict["score_top2"] = top2_score
                        risk_dict["score_gap"] = score_gap
                        risk_dict["score_ratio"] = score_ratio
                        
                        X_risk = np.array([[float(risk_dict.get(f, 0.0)) for f in self.risk_features]], dtype=np.float32)
                        try:
                            prob = float(self.risk_model.predict_proba(X_risk)[0, 1])
                            if self.risk_calibrator is not None:
                                prob = float(self.risk_calibrator.predict_proba(X_risk)[0, 1])
                            routing_confidence = prob
                            routing_status = "AUTO" if prob >= self.risk_threshold else "REVIEW"
                        except Exception as e:
                            print(f"Risk model crash: {e}")
                            
                    # Head Office Resolution Logic Gate (Laplace-Scaled IDF >= 5.0)
                    is_head_office = bool(cand_k.get("etablissement_siege") or cand_k.get("is_siege"))
                    name_overlap = float(feat_row.get("name_token_overlap_max", 0.0))
                    idf_val = float(feat_row.get("idf_name", 0.0))

                    if is_head_office and name_overlap >= 0.90 and idf_val >= 5.0:
                        if top1_score < 0.75 and top1_score >= 0.40:
                            top1_score = max(top1_score, 0.85)
                            scores[idx_k] = max(scores[idx_k], 0.85)

                    if routing_status is None:
                        # Score-based fallback routing
                        if top1_score >= 0.75:
                            routing_status = "AUTO"
                        elif top1_score >= 0.50:
                            routing_status = "REVIEW"
                        else:
                            routing_status = "NO_MATCH"

                    routing_status_category = routing_status
                    if routing_status == "AUTO":
                        routing_status_category = "AUTO_CLOSED" if candidate_state == "FERME" else "AUTO_OPEN"
                    elif routing_status == "REVIEW":
                        routing_status_category = "REVIEW_CLOSED" if candidate_state == "FERME" else "REVIEW_OPEN"
                
                row = TopKRow(
                    crm_id=crm_input.crm_id,
                    crm_name=crm_input.crm_name,
                    crm_address=crm_input.crm_address,
                    crm_postcode=crm_input.postcode,
                    crm_city=crm_input.crm_city,
                    siret_candidate=str(siret_k),
                    score=float(scores[idx_k]),
                    score_top1=top1_score,
                    score_top2=top2_score,
                    score_gap=score_gap,
                    score_ratio=score_ratio,
                    top3_avg=top3_avg,
                    pool_size=pool_size_stage2,
                    pool_size_stage1=pool_size_stage1,
                    candidate_name=primary_name(cand_k) or f"SIRET {siret_k}",
                    candidate_addr=build_address(cand_k),
                    candidate_city=cand_k.get("city"),
                    candidate_postcode=str(cand_k.get("postcode") or ""),
                    candidate_insee=str(cand_k.get("insee") or ""),
                    candidate_state=candidate_state,
                    candidate_last_treatment_date=cand_k.get("last_treatment_date"),
                    routing_confidence=routing_confidence,
                    routing_status=routing_status,
                    rank=rank,
                    has_name_evidence=_has_name_evidence(feat_row),
                )
                
                if export_routing_features:
                    row.name_jaro_max = float(feat_row.get("name_jaro_max", 0.0))
                    row.name_token_overlap_max = float(feat_row.get("name_token_overlap_max", 0.0))
                    row.name_sim_max_etab = float(feat_row.get("name_sim_max_etab", 0.0))
                    row.name_crm_contains_cand_max = float(feat_row.get("name_crm_contains_cand_max", 0.0))
                    row.name_sim_max_pm_dirigeant = float(feat_row.get("name_sim_max_pm_dirigeant", 0.0))
                    row.idf_name = float(feat_row.get("idf_name", 0.0))
                    row.numeric_token_match = float(feat_row.get("numeric_token_match", 0.0))
                    row.addr_jaro = float(feat_row.get("addr_jaro", 0.0))
                    row.addr_token_overlap = float(feat_row.get("addr_token_overlap", 0.0))
                    row.address_density = float(feat_row.get("address_density", 1.0))
                    row.street_number_diff = float(feat_row.get("street_number_diff", 9999))
                    row.name_semantic_max = float(feat_row.get("name_semantic_max", 0.0))
                    row.name_semantic_second = float(feat_row.get("name_semantic_second", 0.0))
                    row.name_semantic_gap = float(feat_row.get("name_semantic_gap", 0.0))
                    row.name_contains_crm_max = float(feat_row.get("name_contains_crm_max", 0.0))
                    row.postcode_match = float(feat_row.get("postcode_match", 0.0))
                    row.city_match = float(feat_row.get("city_match", 0.0))
                    row.street_name_jaro = float(feat_row.get("street_name_jaro", 0.0))
                    row.name_addr_consistency = float(feat_row.get("name_addr_consistency", 0.0))
                    row.name_length_max = float(feat_row.get("name_length_max", 0.0))
                    row.legal_form_category = float(feat_row.get("legal_form_category", 0.0))
                
                rows.append(row)
            results.append(rows)

        # 5. Two-pass matching cascade (SQLite Fallback for weak matches)
        fallback_queries = []
        for idx, (crm_input, rows) in enumerate(zip(crm_inputs, results, strict=True)):
            top_score = rows[0].score if rows else 0.0
            if top_score < 0.50:
                fallback_queries.append({
                    "batch_idx": idx,
                    "crm_input": crm_input,
                    "crm_pre": batch_state[idx]["crm_pre"],
                    "old_rows": rows,
                })

        if fallback_queries:
            print(f"[Fallback Matcher] Running SQLite fallback cascade for {len(fallback_queries)} failed records...")
            # Phase 1: Retrieve and rank fallback candidates
            all_texts_fallback = []
            for fq in fallback_queries:
                postcode = fq["crm_input"].postcode if fq["crm_input"].postcode else None
                candidates = self._search_sqlite_candidates(fq["crm_input"], postcode)
                candidates = self._filter_fallback_candidates(
                    candidates, drop_unnamed=drop_unnamed, exclude_closed=exclude_closed
                )
                has_strong_local_cand = False
                if candidates:
                    from src.xgb_matcher.features import jaro_sim
                    from src.xgb_matcher.naming import normalize_text, primary_name
                    crm_name_norm = _expand_acronyms(normalize_text(fq["crm_input"].crm_name))
                    for c in candidates:
                        cand_name = primary_name(c)
                        if cand_name:
                            cand_name_norm = _expand_acronyms(normalize_text(cand_name))
                            name_sim = jaro_sim(crm_name_norm, cand_name_norm)
                            
                            # Short acronym safeguard (<= 4 chars like DSI, CHRS): require exact token match
                            crm_raw_clean = str(fq["crm_input"].crm_name or "").strip().upper()
                            if len(crm_raw_clean) <= 4 and crm_name_norm != cand_name_norm:
                                name_sim = 0.0
                            
                            # Location verification to avoid homonym blocks
                            cand_city_norm = normalize_text(c.get("city") or "")
                            crm_city_norm = normalize_text(fq["crm_input"].crm_city or "")
                            exact_city_or_pc = (crm_city_norm == cand_city_norm) or (str(fq["crm_input"].postcode).strip() == str(c.get("postcode")).strip())
                            
                            if _is_public_entity(fq["crm_input"].crm_name, c):
                                cand_dept = str(c.get("postcode") or "")[:2]
                                crm_dept = str(fq["crm_input"].postcode or "")[:2]
                                city_ok = (cand_dept == crm_dept) and (cand_dept != "")
                            else:
                                city_ok = exact_city_or_pc
                            
                            if name_sim >= 0.85 and city_ok:
                                has_strong_local_cand = True
                                break

                old_top_score = fq["old_rows"][0].score if fq["old_rows"] else 0.0
                if not candidates or not has_strong_local_cand or old_top_score < 0.75:
                    reason = "no candidates" if not candidates else "only weak candidates"
                    
                    # Clean the query name to strip branch codes, descriptive words, and corporate noise suffixes
                    import re
                    api_query_name = fq["crm_input"].crm_name.upper()
                    for sep in [" - ", " / ", " : ", "  -", " -"]:
                        if sep in api_query_name:
                            parts = api_query_name.split(sep)
                            if len(parts[0].strip()) >= 3:
                                api_query_name = parts[0]
                                break
                    api_query_name = re.sub(r"^\b(LE|LA|LES|L|UN|UNE|DES|DE)\b", "", api_query_name)
                    api_query_name = re.sub(r"\b\d+\b", "", api_query_name)
                    api_query_name = re.sub(r"\b(CORPORATION|INTERNATIONAL|SERVICES|SYSTEMS|TECHNOLOGIES|HOLDING|EUROPE|ENTERPRISES|GROUP|SARL|SAS|SA|EURL|SCI)\b", "", api_query_name)
                    api_query_name = re.sub(r"[^\w\s]", " ", api_query_name)
                    api_query_name = re.sub(r"\s+", " ", api_query_name).strip()
                    
                    print(f"[Fallback Matcher] SQLite found {reason} for {fq['crm_input'].crm_name}. Trying Free Data.gouv API with query '{api_query_name}'...")
                    import sys
                    try:
                        from pipe_v6.external_sources import search_datagouv
                        from pipe_v6.config import PipelineConfig
                    except ImportError:
                        from src.pipe_v6.external_sources import search_datagouv
                        from src.pipe_v6.config import PipelineConfig
                    
                    config = PipelineConfig(
                        datagouv_api_url="https://recherche-entreprises.api.gouv.fr",
                        max_candidates_per_source=10,
                        store_source_raw=True
                    )
                    try:
                        raw_cands = search_datagouv(
                            normalized_name=api_query_name,
                            city=fq["crm_input"].crm_city or "",
                            config=config,
                            postcode=postcode
                        )
                        if raw_cands:
                            mapped_cands = [self._map_datagouv_candidate(c) for c in raw_cands]
                            api_cands = self._filter_fallback_candidates(
                                mapped_cands, drop_unnamed=drop_unnamed, exclude_closed=exclude_closed
                            )
                            if api_cands:
                                for c in api_cands:
                                    c["is_datagouv_api"] = True
                                print(f"[Fallback Matcher] Free Data.gouv API found {len(api_cands)} candidates for {fq['crm_input'].crm_name}!")
                                if not candidates:
                                    candidates = api_cands
                                else:
                                    seen_sirets = {c["siret"] for c in candidates if c.get("siret")}
                                    for c in api_cands:
                                        if c.get("siret") not in seen_sirets:
                                            candidates.append(c)
                                            seen_sirets.add(c["siret"])
                                            
                                # Check if the API returned a strong candidate to avoid unnecessary scraping
                                for c in api_cands:
                                    cand_name = primary_name(c)
                                    if cand_name:
                                        cand_name_norm = normalize_text(cand_name)
                                        name_sim = jaro_sim(crm_name_norm, cand_name_norm)
                                        cand_city_norm = normalize_text(c.get("city") or "")
                                        crm_city_norm = normalize_text(fq["crm_input"].crm_city or "")
                                        city_ok = (crm_city_norm == cand_city_norm) or (str(fq["crm_input"].postcode).strip() == str(c.get("postcode")).strip())
                                        if name_sim >= 0.85 and city_ok:
                                            has_strong_local_cand = True
                                            break
                    except Exception as exc:
                        print(f"[Fallback Matcher Warning] Free Data.gouv API lookup failed: {exc}")
                        
                if (not candidates or not has_strong_local_cand) and not getattr(self, "disable_scraper", False):
                    # Web Scraper Fallback (for unmatchable brand names or weak homonym candidates)
                    siret_or_siren = self._search_web_scraper_siret(
                        fq["crm_input"].crm_name,
                        fq["crm_input"].crm_city,
                        postcode
                    )
                    if siret_or_siren:
                        # 1. Query SQLite
                        scraper_cands = self._get_sqlite_candidate_by_id(siret_or_siren)
                        # 2. Query Data.gouv API if not in SQLite
                        if not scraper_cands:
                            try:
                                raw_cands = search_datagouv(
                                    normalized_name=siret_or_siren,
                                    city="",
                                    config=config,
                                    postcode=None
                                )
                                if raw_cands:
                                    mapped_cands = [self._map_datagouv_candidate(c) for c in raw_cands]
                                    scraper_cands = self._filter_fallback_candidates(
                                        mapped_cands, drop_unnamed=drop_unnamed, exclude_closed=exclude_closed
                                    )
                            except Exception as scraper_exc:
                                print(f"[Web Scraper Fallback Warning] Data.gouv lookup by ID failed: {scraper_exc}")
                                
                        if scraper_cands:
                            print(f"[Fallback Matcher] Web Scraper Fallback successfully resolved {len(scraper_cands)} candidates for {siret_or_siren}!")
                            for c in scraper_cands:
                                c["is_scraped"] = True
                            # Overwrite candidates with the precise scraped target company
                            candidates = scraper_cands
                            has_strong_local_cand = True

                if not candidates:
                    continue

                first_pass_state = batch_state[fq["batch_idx"]]
                if first_pass_state["idf_map"]:
                    idf_map = first_pass_state["idf_map"]
                    default_idf = first_pass_state["default_idf"]
                else:
                    from src.xgb_matcher.candidates import compute_name_idf_map
                    candidates_dict = {c["siret"]: c for c in candidates if c["siret"]}
                    idf_map, default_idf = compute_name_idf_map(candidates_dict)
                set_global_name_idf_map(idf_map, float(default_idf))

                feats_stage1 = [
                    make_features_from_preprocessed(fq["crm_pre"], c, skip_semantic=True)
                    for c in candidates
                ]
                ranker_feature_order = self.ranker_feature_order or self.feature_order
                X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
                scores_stage1 = self.ranker.predict(
                    xgb.DMatrix(X1.values, feature_names=ranker_feature_order)
                )

                stage1_top_n = min(self.stage1_top_n, len(scores_stage1))
                top_n_idx = np.argsort(scores_stage1)[::-1][:stage1_top_n]

                fq["candidates"] = candidates
                fq["feats_stage1"] = feats_stage1
                fq["top_n_idx"] = top_n_idx
                fq["idf_map"] = idf_map
                fq["default_idf"] = default_idf
                fq["cand_list"] = [(c["siret"], c) for c in candidates]

                # Collect fallback semantic texts
                fq_texts = []
                crm_name_sem = fq["crm_pre"].get("crm_name_semantic", "")
                if crm_name_sem and _semantic_enabled():
                    fq_texts.append(crm_name_sem)

                if _semantic_enabled():
                    for idx_top in top_n_idx:
                        feat = feats_stage1[idx_top]
                        if not semantic_gate_allows(
                            feat.get("name_jaro_max", 0.0),
                            feat.get("name_token_overlap_max", 0.0),
                        ):
                            continue
                        c = candidates[idx_top]
                        cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                        pool = build_semantic_name_pool(
                            build_candidate_names(c),
                            crm_city_norm=fq["crm_pre"].get("crm_city_norm", ""),
                            cand_city_norm=cand_city_norm,
                        )
                        fq_texts.extend(pool)
                fq["texts_to_encode"] = fq_texts
                all_texts_fallback.extend(fq_texts)

            # Phase 2: Batch semantic encoding
            if _semantic_enabled() and all_texts_fallback:
                print(f"[Fallback Matcher] Sending {len(all_texts_fallback)} fallback texts to Encoder...")
                batch_encode_texts(all_texts_fallback)

            # Phase 3: Stage 2 Decider scoring & promotion
            for fq in fallback_queries:
                if "candidates" not in fq or not fq["candidates"]:
                    continue

                feats_n = []
                cand_list_n = []
                semantic_pools = []
                set_global_name_idf_map(fq["idf_map"], fq["default_idf"])

                for idx_top in fq["top_n_idx"]:
                    siret, c = fq["cand_list"][idx_top]
                    feat = fq["feats_stage1"][idx_top].copy()
                    cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))

                    pool = []
                    if _semantic_enabled():
                        pool = build_semantic_name_pool(
                            build_candidate_names(c),
                            crm_city_norm=fq["crm_pre"].get("crm_city_norm", ""),
                            cand_city_norm=cand_city_norm,
                        )
                    semantic_pools.append(pool)
                    feat["_siret"] = siret
                    feat["_cand_name"] = primary_name(c) or f"SIRET {siret}"
                    feats_n.append(feat)
                    cand_list_n.append((siret, c))

                if not feats_n:
                    continue

                # Add semantic features
                from src.xgb_matcher.semantic import top2_semantic_similarities_batch
                sem = top2_semantic_similarities_batch(fq["crm_pre"].get("crm_name_semantic", ""), semantic_pools)
                for feat, (sem_max, sem_second, sem_gap) in zip(feats_n, sem, strict=True):
                    if semantic_gate_allows(
                        feat.get("name_jaro_max", 0.0),
                        feat.get("name_token_overlap_max", 0.0),
                    ):
                        feat["name_semantic_max"] = sem_max
                        feat["name_semantic_second"] = sem_second
                        feat["name_semantic_gap"] = sem_gap
                    else:
                        feat["name_semantic_max"] = 0.0
                        feat["name_semantic_second"] = 0.0
                        feat["name_semantic_gap"] = 0.0

                # Feature 13.5 & 13.7: Sub-Entity Delimiter, Syndicat Prefix Stripping & Substring Similarity
                from rapidfuzz.distance import JaroWinkler
                crm_name_upper = fq["crm_input"].crm_name.upper().strip()
                core_crm_name = _extract_core_company_name(fq["crm_input"].crm_name).upper().strip()
                syndic_prefixes = [
                    "SYNDICAT DES COPROPRIETAIRES DE LA ",
                    "SYNDICAT DES COPROPRIETAIRES DU ",
                    "SYNDICAT DES COPROPRIETAIRES DE ",
                    "SYNDICAT DES COPROPRIETAIRES ",
                    "SYNDICAT DE COPROPRIETE ",
                    "COPROPRIETE "
                ]

                for feat, (_, c) in zip(feats_n, cand_list_n, strict=True):
                    cand_name_text = str(c.get("denomination") or c.get("denomination_ul") or "").upper().strip()
                    if cand_name_text:
                        clean_cand_name = cand_name_text
                        for pref in syndic_prefixes:
                            if clean_cand_name.startswith(pref):
                                clean_cand_name = clean_cand_name[len(pref):].strip()
                                break

                        sim_full = float(JaroWinkler.similarity(crm_name_upper, clean_cand_name))
                        sim_core = float(JaroWinkler.similarity(core_crm_name, clean_cand_name))
                        max_sim = max(sim_full, sim_core)

                        # Token permutation handling (e.g. LABORATOIRE EUROFINS vs EUROFINS LABORATOIRE)
                        crm_tokens = set(core_crm_name.split()) - {"DE", "DU", "DES", "LE", "LA", "LES", "D", "L", "SUD", "NORD", "EST", "OUEST"}
                        cand_tokens = set(clean_cand_name.split())
                        if crm_tokens and crm_tokens.issubset(cand_tokens):
                            max_sim = max(max_sim, 0.95)
                            feat["name_first_word_match_max"] = 1.0

                        if (len(crm_name_upper) >= 6 and crm_name_upper in cand_name_text) or (len(clean_cand_name) >= 6 and clean_cand_name in crm_name_upper):
                            max_sim = max(max_sim, 0.95)

                        if max_sim > feat.get("name_jaro_max", 0.0):
                            feat["name_jaro_max"] = max_sim

                # Stage 2 scoring
                X_n = pd.DataFrame(feats_n)[self.feature_order]
                
                probs = self.decider.predict_proba(X_n.values)[:, 1]
                if self.calibrator is not None:
                    probs = self.calibrator.predict_proba(X_n.values)[:, 1]

                scores = np.array(probs)

                # Rule 3 Anchor Boost for Fallback Candidates
                for idx_c, (siret, c) in enumerate(cand_list_n):
                    feat_row = feats_n[idx_c]
                    name_jaro_max = feat_row.get("name_jaro_max", 0.0)
                    postcode_match = feat_row.get("postcode_match", 0.0)
                    city_match = feat_row.get("city_match", 0.0)
                    cand_pc = str(c.get("postcode") or "").strip().zfill(5)
                    crm_pc = str(fq["crm_input"].postcode or "").strip().zfill(5)
                    if cand_pc and crm_pc and cand_pc == crm_pc:
                        postcode_match = 1.0

                    if name_jaro_max >= 0.90 and (postcode_match >= 0.5 or city_match >= 0.5):
                        old_s = scores[idx_c]
                        scores[idx_c] = max(old_s + 0.35 * (1.0 - old_s), 0.82)
                old_top_score = fq["old_rows"][0].score if fq["old_rows"] else 0.0
                if old_top_score < 0.75:
                    for i, (siret, c) in enumerate(cand_list_n):
                        if c.get("is_scraped"):
                            feat = feats_n[i]
                            exact_loc = (feat.get("postcode_match", 0.0) >= 0.5 or feat.get("city_match", 0.0) >= 0.5)
                            cand_pc = str(c.get("postcode") or "").strip().zfill(5)
                            crm_pc = str(fq["crm_input"].postcode or "").strip().zfill(5)
                            dept_match = (cand_pc[:2] == crm_pc[:2]) and (cand_pc[:2] != "")
                            is_pub = _is_public_entity(fq["crm_input"].crm_name, c)
                            
                            if exact_loc or (is_pub and dept_match):
                                name_sim = feat.get("name_jaro_max", 0.0)
                                semantic_sim = feat.get("name_semantic_max", 0.0)
                                token_overlap = feat.get("name_token_overlap_max", 0.0)
                                
                                # Minimum Name Similarity Gate: Require name_sim >= 0.50 OR token_overlap >= 0.35 OR semantic_sim >= 0.55
                                if max(name_sim, semantic_sim) >= 0.50 or token_overlap >= 0.35:
                                    postcode_bonus = 0.10 if cand_pc == crm_pc else (0.05 if dept_match else 0.0)
                                    name_bonus = 0.08 * max(name_sim, semantic_sim)
                                    
                                    final_score = 0.80 + postcode_bonus + name_bonus
                                    final_score = min(0.98, max(0.80, final_score))
                                    
                                    print(f"[Web Scraper Boost] Promoting scraped candidate {siret} to dynamic score {final_score:.4f} (Postcode: {postcode_bonus:.2f}, Name: {name_bonus:.2f})")
                                    scores[i] = final_score
                                else:
                                    print(f"[Web Scraper Gate Blocked] Scraped candidate {siret} ({c.get('denomination')}) failed name similarity gate (NameSim: {name_sim:.2f}, TokenOverlap: {token_overlap:.2f}). Kept in REVIEW/NO_MATCH.")

                pool_size_stage1 = len(fq["cand_list"])
                pool_size_stage2 = len(scores)
                scores_sorted = np.sort(scores)[::-1]
                top3_avg = float(np.mean(scores_sorted[:3])) if len(scores_sorted) >= 3 else float(np.mean(scores_sorted)) if len(scores_sorted) else 0.0

                sorted_idx = np.argsort(scores)[::-1]
                topk_idx = sorted_idx[:top_k].tolist()
                topk_idx = self._promote_open_over_closed(topk_idx, cand_list_n)

                top1_score = float(scores[topk_idx[0]]) if topk_idx else 0.0
                top2_score = float(scores[topk_idx[1]]) if len(topk_idx) > 1 else 0.0
                score_gap = top1_score - top2_score
                denom = top2_score if top2_score > 1e-6 else 0.001
                score_ratio = top1_score / denom

                # Build fallback output rows
                fallback_rows = []
                for rank, idx_k in enumerate(topk_idx, start=1):
                    siret_k, cand_k = cand_list_n[idx_k]
                    etat_admin = str(cand_k.get("etat_admin") or "").strip().upper() or None
                    candidate_state = None
                    if etat_admin is not None:
                        candidate_state = "FERME" if etat_admin == "F" else "OUVERT"
                    feat_row = feats_n[idx_k]

                    routing_confidence = None
                    routing_status = None
                    if rank == 1:
                        if self.risk_model is not None and self.risk_features:
                            risk_dict = feat_row.copy()
                            risk_dict["score_top1"] = top1_score
                            risk_dict["score_top2"] = top2_score
                            risk_dict["score_gap"] = score_gap
                            risk_dict["score_ratio"] = score_ratio

                            X_risk = np.array([[float(risk_dict.get(f, 0.0)) for f in self.risk_features]], dtype=np.float32)
                            try:
                                prob = float(self.risk_model.predict_proba(X_risk)[0, 1])
                                if self.risk_calibrator is not None:
                                    prob = float(self.risk_calibrator.predict_proba(X_risk)[0, 1])
                                routing_confidence = prob
                                routing_status = "AUTO" if prob >= self.risk_threshold else "REVIEW"
                            except Exception as e:
                                print(f"Risk model crash: {e}")
                                
                        if routing_status is None:
                            # Score-based fallback routing
                            if top1_score >= 0.75:
                                routing_status = "AUTO"
                            elif top1_score >= 0.50:
                                routing_status = "REVIEW"
                            else:
                                routing_status = "NO_MATCH"

                        routing_status_category = routing_status
                        if cand_k.get("is_datagouv_api"):
                            routing_status_category = "DATA_GOUV_API"
                        elif routing_status == "AUTO":
                            routing_status_category = "AUTO_CLOSED" if candidate_state == "FERME" else "AUTO_OPEN"
                        elif routing_status == "REVIEW":
                            routing_status_category = "REVIEW_CLOSED" if candidate_state == "FERME" else "REVIEW_OPEN"

                    row = TopKRow(
                        crm_id=fq["crm_input"].crm_id,
                        crm_name=fq["crm_input"].crm_name,
                        crm_address=fq["crm_input"].crm_address,
                        crm_postcode=fq["crm_input"].postcode,
                        crm_city=fq["crm_input"].crm_city,
                        siret_candidate=str(siret_k),
                        score=float(scores[idx_k]),
                        score_top1=top1_score,
                        score_top2=top2_score,
                        score_gap=score_gap,
                        score_ratio=score_ratio,
                        top3_avg=top3_avg,
                        pool_size=pool_size_stage2,
                        pool_size_stage1=pool_size_stage1,
                        candidate_name=primary_name(cand_k) or f"SIRET {siret_k}",
                        candidate_addr=build_address(cand_k),
                        candidate_city=cand_k.get("city"),
                        candidate_postcode=str(cand_k.get("postcode") or ""),
                        candidate_insee=str(cand_k.get("insee") or ""),
                        candidate_state=candidate_state,
                        candidate_last_treatment_date=cand_k.get("last_treatment_date"),
                        routing_confidence=routing_confidence,
                        routing_status=routing_status,
                        rank=rank,
                        has_name_evidence=_has_name_evidence(feat_row),
                    )

                    if export_routing_features:
                        row.name_jaro_max = float(feat_row.get("name_jaro_max", 0.0))
                        row.name_token_overlap_max = float(feat_row.get("name_token_overlap_max", 0.0))
                        row.name_sim_max_etab = float(feat_row.get("name_sim_max_etab", 0.0))
                        row.name_crm_contains_cand_max = float(feat_row.get("name_crm_contains_cand_max", 0.0))
                        row.name_sim_max_pm_dirigeant = float(feat_row.get("name_sim_max_pm_dirigeant", 0.0))
                        row.idf_name = float(feat_row.get("idf_name", 0.0))
                        row.numeric_token_match = float(feat_row.get("numeric_token_match", 0.0))
                        row.addr_jaro = float(feat_row.get("addr_jaro", 0.0))
                        row.addr_token_overlap = float(feat_row.get("addr_token_overlap", 0.0))
                        row.address_density = float(feat_row.get("address_density", 1.0))
                        row.street_number_diff = float(feat_row.get("street_number_diff", 9999))
                        row.name_semantic_max = float(feat_row.get("name_semantic_max", 0.0))
                        row.name_semantic_second = float(feat_row.get("name_semantic_second", 0.0))
                        row.name_semantic_gap = float(feat_row.get("name_semantic_gap", 0.0))
                        row.name_contains_crm_max = float(feat_row.get("name_contains_crm_max", 0.0))
                        row.postcode_match = float(feat_row.get("postcode_match", 0.0))
                        row.city_match = float(feat_row.get("city_match", 0.0))
                        row.street_name_jaro = float(feat_row.get("street_name_jaro", 0.0))
                        row.name_addr_consistency = float(feat_row.get("name_addr_consistency", 0.0))
                        row.name_length_max = float(feat_row.get("name_length_max", 0.0))
                        row.legal_form_category = float(feat_row.get("legal_form_category", 0.0))

                    fallback_rows.append(row)

                old_rows = fq["old_rows"]
                old_top_score = old_rows[0].score if old_rows else 0.0
                new_top_score = fallback_rows[0].score if fallback_rows else 0.0

                if new_top_score > old_top_score:
                    results[fq["batch_idx"]] = fallback_rows
                    print(f"[Fallback Promotion] {fq['crm_input'].crm_name}: score jumped from {old_top_score:.4f} to {new_top_score:.4f} (Match: {fallback_rows[0].candidate_name})")

        return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer XGB two-stage matcher (SSOT, Optimized).")
    p.add_argument("--crm-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("reports/xgb_two_stage_topk.csv"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v7_all"))
    p.add_argument("--pool-mode", choices=["insee_then_postcode"], default="insee_then_postcode")
    p.add_argument("--prefilter-k", type=int, default=500)
    p.add_argument("--min-candidates", type=int, default=100)
    p.add_argument(
        "--stage1-top-n",
        type=int,
        default=20,
        help="Top-N candidates after Stage1 ranker (Version 2.2 SSOT=20).",
    )
    p.add_argument("--char-top-k", type=int, default=200)
    p.add_argument("--tfidf-name-mode", choices=["bag"], default="bag")
    p.add_argument("--siren-siblings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--drop-unnamed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--exclude-closed", action="store_true", default=False)
    p.add_argument("--export-routing-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--disable-scraper",
        action="store_true",
        default=True,
        help="Disable web scraper fallback (Pure Offline Local-Only SSOT).",
    )
    p.add_argument(
        "--allow-no-semantic",
        action="store_true",
        default=False,
        help="Allow running without semantic embeddings (semantic features will be zero).",
    )
    p.add_argument("--meta-path", type=Path, default=None)
    p.add_argument("--ranker-model", type=Path, default=None)
    p.add_argument("--ranker-fast-model", type=Path, default=None)
    p.add_argument("--decider-model", type=Path, default=None)
    p.add_argument("--calibrator-path", type=Path, default=None)
    p.add_argument(
        "--override-retrieval",
        action="store_true",
        default=False,
        help="Allow overriding retrieval knobs even when --meta-path is provided.",
    )
    p.add_argument("--debug-gt", action="store_true", default=False)
    p.add_argument("--gt-column", type=str, default="ground_truth_siret")
    p.add_argument(
        "--debug-gt-output",
        type=Path,
        default=None,
        help="Path for GT debug CSV output (default: <output>_gt_debug.csv).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for batch semantic encoding and inference.",
    )
    p.add_argument(
        "--database-path",
        type=Path,
        default=Path("data/sirene_cache.sqlite"),
        help="Path to the SIRENE cache SQLite database."
    )
    return p.parse_args()


def load_crm(path: Path) -> pd.DataFrame:
    import csv

    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"

    df = pd.read_csv(path, sep=delimiter, dtype=str)
    df = df.rename(
        columns={
            "Client final": "crm_name",
            "Adresse": "crm_address",
            "Commune": "crm_city",
            "Code Postal": "postcode",
            "Code INSEE": "insee",
            "crm_insee": "insee",
            "crm_cp": "postcode",
            "crm_adresse": "crm_address",
            "crm_commune": "crm_city",
        }
    )
    def clean_code(val, max_len=5):
        if pd.isna(val):
            return val
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.isdigit() and len(s) < max_len:
            s = s.zfill(max_len)
        return s

    for col in ["postcode", "insee"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_code(x, 5))
    if "crm_id" not in df.columns:
        df["crm_id"] = df.index
    return df


def _find_latest_meta(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.glob("xgb_two_stage_meta_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _build_profile(args: argparse.Namespace, logger: logging.Logger) -> InferenceProfile:
    if args.allow_no_semantic:
        os.environ["XGB_ALLOW_NO_SEMANTIC"] = "1"

    model_dir = Path("models")
    meta_path = args.meta_path or _find_latest_meta(model_dir)

    if meta_path and meta_path.exists():
        profile = InferenceProfile.from_meta(meta_path, strict=not args.allow_no_semantic)
    else:
        if not args.ranker_model or not args.decider_model:
            raise FileNotFoundError(
                "Meta path not provided and ranker/decider paths not set. "
                "Provide --meta-path or both --ranker-model and --decider-model."
            )
        ranker_is_fast = "fast" in args.ranker_model.name.lower()
        profile = InferenceProfile(
            ranker_path=args.ranker_model,
            ranker_fast_path=args.ranker_model if ranker_is_fast else None,
            decider_path=args.decider_model,
            calibrator_path=args.calibrator_path,
            feature_order=FEATURE_NAMES,
            ranker_feature_order=FEATURE_NAMES,
            ranker_fast_feature_order=FAST_RANKER_FEATURE_NAMES,
            tfidf_name_mode=args.tfidf_name_mode,
            siren_siblings=args.siren_siblings,
            prefilter_k=args.prefilter_k,
            char_top_k=args.char_top_k,
            min_candidates=args.min_candidates,
            stage1_top_n=args.stage1_top_n,
            drop_unnamed=args.drop_unnamed,
            exclude_closed=args.exclude_closed,
            partitions_dir=args.partitions_dir,
            semantic_required=not args.allow_no_semantic,
            use_ranker_fast=ranker_is_fast,
        )

    if not (meta_path and meta_path.exists()) or args.override_retrieval:
        profile.prefilter_k = args.prefilter_k
        profile.tfidf_name_mode = args.tfidf_name_mode
        profile.siren_siblings = args.siren_siblings
        profile.drop_unnamed = args.drop_unnamed
        profile.exclude_closed = args.exclude_closed
        profile.char_top_k = args.char_top_k
        profile.min_candidates = args.min_candidates
        profile.stage1_top_n = args.stage1_top_n
        profile.partitions_dir = args.partitions_dir
        profile.semantic_required = not args.allow_no_semantic
    if args.ranker_model:
        profile.ranker_path = args.ranker_model
    if args.ranker_fast_model:
        profile.ranker_fast_path = args.ranker_fast_model
    if args.decider_model:
        profile.decider_path = args.decider_model
    if args.calibrator_path:
        profile.calibrator_path = args.calibrator_path

    profile.database_path = getattr(args, "database_path", Path("data/sirene_cache.sqlite"))
    profile.disable_scraper = getattr(args, "disable_scraper", False)
    logger.info("Using profile with stage1_top_n=%d", profile.stage1_top_n)
    return profile


def _infer_debug(
    engine: OptimizedXgbInferenceEngine,
    profile: InferenceProfile,
    records: List[dict],
    gt_column: str,
    top_k: int,
    export_routing_features: bool,
    chunk_size: int = 500,
) -> tuple[List[dict], List[dict]]:
    rows_out: List[dict] = []
    debug_rows: List[dict] = []

    from src.xgb_matcher.semantic import is_semantic_available
    use_semantic_batch = is_semantic_available()

    progress = tqdm(total=len(records), desc="Infer (Debug)")

    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]

        if use_semantic_batch:
            from src.xgb_matcher.semantic import batch_encode_texts
            from src.xgb_matcher.features import normalize_text, build_semantic_name_pool
            from src.xgb_matcher.naming import build_candidate_names

            all_texts_to_encode = []
            for r in chunk:
                gt_siret = str(r.get(gt_column) or "").strip()
                gt_siret = gt_siret if gt_siret and gt_siret not in ("", "nan", "None") else None
                crm_pre = preprocess_crm_row(r)

                result = build_candidate_pool(
                    store=engine.store,
                    crm_row=r,
                    crm_pre=crm_pre,
                    config=profile.build_retrieval_config(),
                    tfidf_cache=engine._tfidf_cache,
                    gt_siret=gt_siret,
                    partition_cache=getattr(engine, "_partition_cache", None),
                )

                crm_name_sem = crm_pre.get("crm_name_semantic", "")
                if crm_name_sem:
                    all_texts_to_encode.append(crm_name_sem)

                cand_list = [(c.get("siret"), c) for c in result.candidates if c.get("siret")]
                if not cand_list:
                    continue

                set_global_name_idf_map(result.idf_map, result.default_idf)
                feats_stage1 = [
                    make_features_from_preprocessed(crm_pre, c, skip_semantic=True)
                    for _, c in cand_list
                ]
                ranker_feature_order = engine.ranker_feature_order or engine.feature_order
                X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
                scores_stage1 = engine.ranker.predict(
                    xgb.DMatrix(X1.values, feature_names=ranker_feature_order)
                )
                order = list(reversed(scores_stage1.argsort()))
                stage1_top_n = min(engine.stage1_top_n, len(order))
                top_n_idx = order[:stage1_top_n]

                for idx in top_n_idx:
                    _, c = cand_list[idx]
                    cand_city_norm = c.get("_xgb_cached_city_norm") or normalize_text(c.get("city"))
                    pool = build_semantic_name_pool(
                        build_candidate_names(c),
                        crm_city_norm=crm_pre.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                    all_texts_to_encode.extend(pool)

            if all_texts_to_encode:
                batch_encode_texts(all_texts_to_encode)

        for r in chunk:
            orig_index = r.get("_orig_index")
            gt_siret = str(r.get(gt_column) or "").strip()
            gt_siret = gt_siret if gt_siret and gt_siret not in ("", "nan", "None") else None

            crm_in = CrmInput(
                crm_id=str(r.get("crm_id") or orig_index),
                crm_name=str(r.get("crm_name") or ""),
                crm_address=str(r.get("crm_address") or ""),
                postcode=str(r.get("postcode") or ""),
                insee=str(r.get("insee") or ""),
                crm_city=str(r.get("crm_city") or ""),
            )
            crm_pre = preprocess_crm_row(r)

            result = build_candidate_pool(
                store=engine.store,
                crm_row=r,
                crm_pre=crm_pre,
                config=profile.build_retrieval_config(),
                tfidf_cache=engine._tfidf_cache,  # Shared TF-IDF cache
                gt_siret=gt_siret,
                partition_cache=getattr(engine, "_partition_cache", None),
            )
            set_global_name_idf_map(result.idf_map, result.default_idf)

            in_stage1_topn = False
            stage1_rank = -1
            stage1_topn_size = 0
            if result.candidates:
                cand_sirets = [str(c.get("siret") or "").zfill(14) for c in result.candidates]
                feats_stage1 = [
                    make_features_from_preprocessed(crm_pre, c, skip_semantic=True)
                    for c in result.candidates
                ]
                ranker_feature_order = engine.ranker_feature_order or engine.feature_order
                X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
                scores_stage1 = engine.ranker.predict(
                    xgb.DMatrix(X1.values, feature_names=ranker_feature_order)
                )
                order = list(reversed(scores_stage1.argsort()))
                stage1_top_n = min(engine.stage1_top_n, len(order))
                top_n_idx = order[:stage1_top_n]
                stage1_topn_size = len(top_n_idx)
                if gt_siret:
                    gt_norm = str(gt_siret).zfill(14)
                    in_stage1_topn = gt_norm in {cand_sirets[i] for i in top_n_idx}
                    if result.gt_in_tfidf_pool and not in_stage1_topn:
                        for rank_s1, idx_s1 in enumerate(order, start=1):
                            if cand_sirets[idx_s1] == gt_norm:
                                stage1_rank = rank_s1
                                break

            topk_rows = engine.infer_topk(
                crm_in,
                top_k=top_k,
                pool_mode="insee_then_postcode",
                export_routing_features=export_routing_features,
            )
            for tk in topk_rows:
                row_dict = tk.to_dict()
                row_dict["_orig_index"] = orig_index
                rows_out.append(row_dict)

            in_stage2_topk = False
            stage2_rank = -1
            if gt_siret:
                gt_norm = str(gt_siret).zfill(14)
                for tk in topk_rows:
                    if str(tk.siret_candidate).zfill(14) == gt_norm:
                        in_stage2_topk = True
                        stage2_rank = tk.rank
                        break

            loss_reason = result.loss_reason or ""
            if gt_siret and result.gt_in_tfidf_pool and not in_stage1_topn:
                loss_reason = f"PRUNED_BY_STAGE1_RANK_{stage1_rank}"
            elif gt_siret and in_stage1_topn and not in_stage2_topk:
                loss_reason = "PRUNED_BY_STAGE2_TOPK"

            debug_rows.append(
                {
                    "crm_id": str(r.get("crm_id") or ""),
                    "crm_name": str(r.get("crm_name") or ""),
                    "gt_siret": gt_siret or "",
                    "loc_key": str(r.get("loc_key") or f"{r.get('insee','')}|{r.get('postcode','')}")
                    if gt_siret
                    else "",
                    "in_base_pool": bool(result.gt_in_base_pool),
                    "in_filtered_pool": bool(result.gt_in_filtered_pool),
                    "in_tfidf_pool": bool(result.gt_in_tfidf_pool),
                    "in_stage1_topn": bool(in_stage1_topn),
                    "in_stage2_topk": bool(in_stage2_topk),
                    "stage1_rank": int(stage1_rank),
                    "stage2_rank": int(stage2_rank),
                    "base_pool_size": int(result.pool_sizes.get("base", 0)),
                    "filtered_pool_size": int(result.pool_sizes.get("filtered", 0)),
                    "tfidf_pool_size": int(result.pool_sizes.get("tfidf", 0)),
                    "stage1_topn_size": int(stage1_topn_size),
                    "loss_reason": loss_reason,
                }
            )
            progress.update(1)
    progress.close()
    return rows_out, debug_rows


def main() -> None:
    import json
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger(__name__)

    if args.pool_mode != "insee_then_postcode":
        raise ValueError("pool_mode must be 'insee_then_postcode' (SSOT).")

    crm = load_crm(args.crm_path)
    crm_records = crm.to_dict("records")
    for idx, r in enumerate(crm_records):
        r["_orig_index"] = idx

    # Sort queries geometrically to maximize Parquet LRU cache hits. This reduces I/O by 99%
    crm_records.sort(key=lambda x: (str(x.get("postcode") or ""), str(x.get("insee") or "")))

    profile = _build_profile(args, logger)
    engine = OptimizedXgbInferenceEngine.from_profile(profile)

    temp_path = args.output_path.with_suffix(".jsonl.tmp")
    completed_orig_indices = set()
    rows_out: List[dict] = []
    debug_rows: List[dict] = []

    if temp_path.exists():
        logger.info("Found temporary progress file: %s. Resuming...", temp_path)
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_out.append(row)
                    if "_orig_index" in row:
                        completed_orig_indices.add(int(row["_orig_index"]))
            logger.info("Loaded %d matched candidate rows for %d queries.", len(rows_out), len(completed_orig_indices))
        except Exception as e:
            logger.error("Failed to load progress from %s: %s. Starting from scratch.", temp_path, e)
            completed_orig_indices.clear()
            rows_out.clear()

    total_queries = len(crm_records)
    completed_count = len(completed_orig_indices)

    if completed_orig_indices:
        crm_records = [r for r in crm_records if r.get("_orig_index") not in completed_orig_indices]
        logger.info("Filtered completed queries: %d remaining out of %d.", len(crm_records), total_queries)

    if args.debug_gt:
        rows_out, debug_rows = _infer_debug(
            engine,
            profile,
            crm_records,
            args.gt_column,
            args.top_k,
            args.export_routing_features,
            chunk_size=args.chunk_size,
        )
    else:
        from src.xgb_matcher.semantic import is_semantic_available
        use_batch = is_semantic_available()

        if use_batch:
            logger.info("Semantic model is available. Running in batched chunk mode (size: %d)...", args.chunk_size)
            chunk_size = args.chunk_size
            progress = tqdm(total=total_queries, initial=completed_count, desc="Infer (Batched)")
            for i in range(0, len(crm_records), chunk_size):
                chunk = crm_records[i : i + chunk_size]
                crm_inputs = []
                orig_indices = []
                for r in chunk:
                    orig_index = r.get("_orig_index")
                    orig_indices.append(orig_index)
                    crm_in = CrmInput(
                        crm_id=str(r.get("crm_id") or orig_index),
                        crm_name=str(r.get("crm_name") or ""),
                        crm_address=str(r.get("crm_address") or ""),
                        postcode=str(r.get("postcode") or ""),
                        insee=str(r.get("insee") or ""),
                        crm_city=str(r.get("crm_city") or ""),
                    )
                    crm_inputs.append(crm_in)

                batch_topk_rows = engine.infer_topk_batch(
                    crm_inputs,
                    top_k=args.top_k,
                    pool_mode=args.pool_mode,
                    drop_unnamed=args.drop_unnamed,
                    exclude_closed=args.exclude_closed,
                    export_routing_features=args.export_routing_features,
                )

                batch_rows = []
                for orig_index, topk_rows in zip(orig_indices, batch_topk_rows, strict=True):
                    for tk in topk_rows:
                        row_dict = tk.to_dict()
                        row_dict["_orig_index"] = orig_index
                        batch_rows.append(row_dict)
                rows_out.extend(batch_rows)

                try:
                    with open(temp_path, "a", encoding="utf-8") as f:
                        for r_dict in batch_rows:
                            f.write(json.dumps(r_dict, cls=DateTimeEncoder) + "\n")
                except Exception as e:
                    logger.error("Failed to write progress to %s: %s", temp_path, e)

                # Clear TF-IDF cache to free RAM!
                engine.clear_tfidf_cache()

                progress.update(len(chunk))
            progress.close()
        else:
            logger.info("Semantic model unavailable. Running optimized sequential matching...")
            progress = tqdm(total=total_queries, initial=completed_count, desc="Infer (Optimized Sequential)")
            for idx, r in enumerate(crm_records):
                orig_index = r.get("_orig_index")
                crm_in = CrmInput(
                    crm_id=str(r.get("crm_id") or orig_index),
                    crm_name=str(r.get("crm_name") or ""),
                    crm_address=str(r.get("crm_address") or ""),
                    postcode=str(r.get("postcode") or ""),
                    insee=str(r.get("insee") or ""),
                    crm_city=str(r.get("crm_city") or ""),
                )
                topk_rows = engine.infer_topk(
                    crm_in,
                    top_k=args.top_k,
                    pool_mode=args.pool_mode,
                    drop_unnamed=args.drop_unnamed,
                    exclude_closed=args.exclude_closed,
                    export_routing_features=args.export_routing_features,
                )
                
                seq_rows = []
                for tk in topk_rows:
                    row_dict = tk.to_dict()
                    row_dict["_orig_index"] = orig_index
                    seq_rows.append(row_dict)
                rows_out.extend(seq_rows)

                try:
                    with open(temp_path, "a", encoding="utf-8") as f:
                        for r_dict in seq_rows:
                            f.write(json.dumps(r_dict, cls=DateTimeEncoder) + "\n")
                except Exception as e:
                    logger.error("Failed to write progress to %s: %s", temp_path, e)
                
                # Clear TF-IDF cache periodically to free RAM!
                if (idx + 1) % 500 == 0:
                    engine.clear_tfidf_cache()

                progress.update(1)
            progress.close()

    if not rows_out:
        logger.warning("No inference results produced. Output will be empty.")
        out_df = pd.DataFrame(columns=["crm_id", "crm_name", "siret_candidate", "score", "rank"])
    else:
        out_df = pd.DataFrame(rows_out)
        if "_orig_index" in out_df.columns:
            out_df = out_df.sort_values(["_orig_index", "rank"], ascending=[True, True])
            out_df = out_df.drop(columns=["_orig_index"])
        else:
            out_df = out_df.sort_values(["crm_name", "rank", "score"], ascending=[True, True, False])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    logger.info("Saved top-%d results to %s (%d rows)", args.top_k, args.output_path, len(out_df))

    try:
        desktop_xlsx = Path(r"C:\Users\Kabouassi\Desktop\results_full_ml_opt.xlsx")
        logger.info("Exporting results to Excel: %s", desktop_xlsx)
        out_df.to_excel(desktop_xlsx, index=False)
        logger.info("Successfully saved Excel results to Desktop.")
    except Exception as e:
        logger.error("Failed to export to Excel: %s", e)

    # Clean up temporary progress file
    if temp_path.exists():
        try:
            temp_path.unlink()
            logger.info("Removed temporary progress file: %s", temp_path)
        except Exception as e:
            logger.warning("Failed to remove temporary progress file: %s", e)

    if args.debug_gt and debug_rows:
        debug_output = args.debug_gt_output or args.output_path.with_name(args.output_path.stem + "_gt_debug.csv")
        debug_df = pd.DataFrame(debug_rows)
        debug_df.to_csv(debug_output, index=False)

        n_total = len(debug_df)
        n_in_base = int(debug_df["in_base_pool"].sum()) if n_total else 0
        n_in_filtered = int(debug_df["in_filtered_pool"].sum()) if n_total else 0
        n_in_tfidf = int(debug_df["in_tfidf_pool"].sum()) if n_total else 0
        n_in_stage1 = int(debug_df["in_stage1_topn"].sum()) if n_total else 0
        n_in_topk = int(debug_df["in_stage2_topk"].sum()) if n_total else 0
        logger.info("GT Debug summary (%d queries with GT):", n_total)
        logger.info("  in_base_pool: %d (%.1f%%)", n_in_base, 100 * n_in_base / n_total if n_total else 0)
        logger.info("  in_filtered_pool: %d (%.1f%%)", n_in_filtered, 100 * n_in_filtered / n_total if n_total else 0)
        logger.info("  in_tfidf_pool: %d (%.1f%%)", n_in_tfidf, 100 * n_in_tfidf / n_total if n_total else 0)
        logger.info("  in_stage1_topn: %d (%.1f%%)", n_in_stage1, 100 * n_in_stage1 / n_total if n_total else 0)
        logger.info("  in_stage2_topk: %d (%.1f%%)", n_in_topk, 100 * n_in_topk / n_total if n_total else 0)

        loss_counts = debug_df["loss_reason"].value_counts()
        logger.info("  Loss reasons:")
        for reason, count in loss_counts.items():
            if reason:
                logger.info("    %s: %d", reason, count)
        logger.info("Saved GT debug to %s", debug_output)


if __name__ == "__main__":
    main()
