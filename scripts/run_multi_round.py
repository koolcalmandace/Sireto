#!/usr/bin/env python3
"""
Multi-round address matching script.
Runs a 5-round matching workflow on the CRM dataset without modifying the core pipe_v6 logic.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import unicodedata
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipe_v6.config import load_config
from pipe_v6.logging_utils import setup_logging
from pipe_v6.commune_detection import CommuneKey
from pipe_v6.external_sources import search_datagouv
from pipe_v6.crm_loader import load_crm

LOGGER = logging.getLogger("multi_round_matcher")


# ============================================================================
# Core Normalization and Collapsing Helpers
# ============================================================================

def clean_name(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").upper()
    s = re.sub(r"\s+", " ", s).strip()
    for tok in ["SAS", "SARL", "SASU", "SA", "ASSOCIATION", "ENTREPRISE", "SOCIETE", "AGENCE", "SITE", "BUREAU", "ANTENNE", "DELEGATION", "DIRECTION", "SERVICE", "INTERNATIONAL", "FRANCE", "GROUP", "GROUPE", "HOLDING", "DEVELOPPEMENT", "DISTRIBUTION", "EUROPE"]:
        s = re.sub(r"\b" + tok + r"\b", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def clean_street_sig(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").upper()
    tokens = s.split()
    street_types = {"RUE", "AVENUE", "BOULEVARD", "ALLEE", "ROUTE", "CHEMIN", "PLACE", "SQUARE", "IMPASSE", "COURS", "AV", "BD", "R", "RTE", "PL"}
    filtered = []
    for t in tokens:
        if t in street_types:
            continue
        if re.match(r"^\d+$", t):  # skip numbers or postcodes
            continue
        filtered.append(t)
    res = "".join(filtered)
    res = re.sub(r"[^A-Z0-9]", "", res)
    return res


def get_street_number(addr_str: str) -> str:
    if not addr_str or pd.isna(addr_str):
        return ""
    tokens = str(addr_str).split()
    if tokens and re.match(r"^\d+$", tokens[0]) and tokens[0] != "0":
        return tokens[0]
    for t in tokens:
        if re.match(r"^\d+$", t) and t != "0" and len(t) < 5:
            return t
    return ""


def street_numbers_conflict(num1: str, num2: str) -> bool:
    """Returns True if both street numbers are present and they represent different houses (diff > 2)."""
    if not num1 or not num2:
        return False
    
    # Extract only the digits to handle suffixes like 'bis', 'ter', 'B'
    d1 = re.sub(r"\D", "", str(num1))
    d2 = re.sub(r"\D", "", str(num2))
    
    if not d1 or not d2:
        return False
        
    try:
        val1 = int(d1)
        val2 = int(d2)
        # Difference greater than 2 indicates a clear conflict on the street
        return abs(val1 - val2) > 2
    except ValueError:
        return d1 != d2


def collapse_address_ultra(number: str, street_name: str, address_full: str) -> str:
    """Collapses an address down to its core number and alphanumeric signature (e.g. '5 RUE DU BAC' -> '5BAC')."""
    num = str(number).strip().upper() if pd.notna(number) else ""
    if num == "NAN" or num == "0" or not num:
        num = ""
    
    # Extract numbers from address_full if num is empty
    if not num and address_full and pd.notna(address_full):
        m = re.match(r"^\d+", str(address_full).strip())
        if m:
            num = m.group(0)
            
    street = str(street_name or address_full or "").upper()
    street = unicodedata.normalize("NFKD", street).encode("ascii", "ignore").decode("ascii")
    
    # Remove French street types
    for tok in ["RUE", "AVENUE", "BOULEVARD", "ALLEE", "ROUTE", "CHEMIN", "PLACE", "SQUARE", "IMPASSE", "COURS", "AV", "BD", "R", "RTE", "PL"]:
        street = re.sub(r"\b" + tok + r"\b", "", street)
    # Remove common French prepositions/articles
    for tok in ["DE", "DU", "LA", "LE", "DES", "LES", "ET", "EN", "AU", "AUX"]:
        street = re.sub(r"\b" + tok + r"\b", "", street)
    # Remove all non-alphanumeric
    street = re.sub(r"[^A-Z0-9]", "", street)
    return f"{num}{street}"


# ============================================================================
# Candidate Retrieval
# ============================================================================

def get_candidates(row: Any, conn: sqlite3.Connection, config: Any) -> List[Dict[str, Any]]:
    postcode = row.get("postcode")
    insee = row.get("insee_code") or row.get("insee")
    city = row.get("city")
    
    crm_name = str(row.get("crm_name") or "")
    crm_city = str(city or "")

    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row

    rows = []
    if postcode and pd.notna(postcode):
        pc_str = str(postcode).strip().zfill(5)
        cursor.execute(
            "SELECT siret, siren, denomination, denomination_unite_legale, nom_unite_legale, prenom1_unite_legale, "
            "enseigne1, enseigne2, enseigne3, street_number, street_name, address_full, postcode, city, insee_code "
            "FROM establishments WHERE postcode = ?",
            (pc_str,)
        )
        rows = cursor.fetchall()

    if not rows and insee and pd.notna(insee):
        insee_str = str(insee).strip().zfill(5)
        cursor.execute(
            "SELECT siret, siren, denomination, denomination_unite_legale, nom_unite_legale, prenom1_unite_legale, "
            "enseigne1, enseigne2, enseigne3, street_number, street_name, address_full, postcode, city, insee_code "
            "FROM establishments WHERE insee_code = ?",
            (insee_str,)
        )
        rows = cursor.fetchall()

    if not rows and crm_city:
        city_clean = clean_name(crm_city)
        cursor.execute(
            "SELECT siret, siren, denomination, denomination_unite_legale, nom_unite_legale, prenom1_unite_legale, "
            "enseigne1, enseigne2, enseigne3, street_number, street_name, address_full, postcode, city, insee_code "
            "FROM establishments"
        )
        all_est = cursor.fetchall()
        rows = [r for r in all_est if r["city"] and clean_name(r["city"]) == city_clean]

    candidates = [dict(r) for r in rows]

    # Check if there is any candidate with a decent name match locally
    crm_name_clean = clean_name(crm_name)
    has_decent_name_match = False
    for c in candidates:
        c_names = [
            c.get("denomination"),
            c.get("denomination_unite_legale"),
            c.get("enseigne1"),
            c.get("enseigne2"),
            c.get("enseigne3")
        ]
        if c.get("nom_unite_legale"):
            c_names.append(c["nom_unite_legale"])
        c_names = [v for v in c_names if v]
        for v in c_names:
            ratio = SequenceMatcher(None, crm_name_clean, clean_name(v)).ratio()
            if ratio >= 0.75:
                has_decent_name_match = True
                break
        if has_decent_name_match:
            break

    # Live API query if no local candidates found OR no local candidate has a decent name match
    if (not candidates or not has_decent_name_match) and config.datagouv_api_url:
        try:
            LOGGER.debug("No decent local name match for '%s', calling search_datagouv live API...", crm_name)
            raw_cands = search_datagouv(crm_name, crm_city, config, postcode=postcode)
            for rc in raw_cands:
                if not any(c["siret"] == rc.siret for c in candidates):
                    candidates.append({
                        "siret": rc.siret,
                        "siren": rc.siren,
                        "denomination": rc.label,
                        "denomination_unite_legale": rc.label,
                        "nom_unite_legale": None,
                        "prenom1_unite_legale": None,
                        "enseigne1": None,
                        "enseigne2": None,
                        "enseigne3": None,
                        "street_number": rc.extra.get("numero"),
                        "street_name": rc.extra.get("libelle_voie"),
                        "address_full": rc.extra.get("adresse"),
                        "postcode": postcode,
                        "city": crm_city,
                        "insee_code": insee,
                    })
        except Exception as e:
            LOGGER.error("Live fallback search failed: %s", e)

    return candidates


# ============================================================================
# Multi-Round Matching Engine
# ============================================================================

def match_row_multi_round(row: Any, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Matches a single CRM row against candidates in 5 progressive rounds."""
    crm_name = str(row.get("crm_name") or "")
    crm_name_clean = clean_name(crm_name)
    
    crm_street_num = str(row.get("street_number") or "").strip()
    if crm_street_num in ["nan", "<NA>", "None"]:
        crm_street_num = ""
    crm_street_name = str(row.get("street_name") or "").strip()
    if crm_street_name in ["nan", "<NA>", "None"]:
        crm_street_name = ""
        
    crm_address_full = f"{crm_street_num} {crm_street_name}".strip()
    
    crm_num = get_street_number(crm_street_num or crm_address_full)
    crm_sig = clean_street_sig(crm_street_name or crm_address_full)
    crm_collapsed = collapse_address_ultra(crm_street_num, crm_street_name, crm_address_full)
    
    # Guess CRM category
    crm_name_upper = crm_name.upper()
    if any(k in crm_name_upper for k in ["COLLEGE", "MAIRIE", "ECOLE", "UNIVERSITE", "LYCEE", "PUBLIC", "COMMUNE", "SNCF", "RATP", "LA POSTE"]):
        crm_category = "PUBLIC"
    elif "SANS ABRI" not in crm_name_upper and any(k in crm_name_upper for k in ["ARMOIRE", "NRO", "ABRI", "STATION", "PARKING", "ECLAIRAGE", "PYLONE", "TRANSFORMATEUR"]):
        crm_category = "EQUIPEMENT_URBAIN"
    else:
        crm_category = "PRIVE"

    # Pre-clean CRM address string
    crm_addr_clean = unicodedata.normalize("NFKD", crm_address_full).encode("ascii", "ignore").decode("ascii").upper()
    crm_addr_clean = re.sub(r"[^A-Z0-9 ]", "", crm_addr_clean)
    crm_addr_clean = re.sub(r"\s+", " ", crm_addr_clean).strip()

    best_match = {
        "status": "NO_MATCH",
        "siret": None,
        "siren": None,
        "name": None,
        "score": 0.0,
        "reason": "No candidates available"
    }
    
    # Short-circuit EQUIPEMENT_URBAIN
    if crm_category == "EQUIPEMENT_URBAIN":
        return {
            "status": "NO_MATCH",
            "siret": None,
            "siren": None,
            "name": None,
            "score": 0.0,
            "reason": "EQUIPEMENT_URBAIN: short-circuit"
        }

    if not candidates:
        return best_match

    # Check if there is any candidate with exact name match
    has_any_exact_name = False
    for c in candidates:
        c_names = [c.get("denomination"), c.get("denomination_unite_legale"), c.get("enseigne1"), c.get("enseigne2"), c.get("enseigne3")]
        if c.get("nom_unite_legale"):
            c_names.append(c["nom_unite_legale"])
        c_names = [v for v in c_names if v]
        for v in c_names:
            if crm_name_clean == clean_name(v):
                has_any_exact_name = True
                break

    best_conf = -1.0
    
    for c in candidates:
        # Pre-clean candidate address string
        cand_addr_clean = unicodedata.normalize("NFKD", c.get("address_full") or "").encode("ascii", "ignore").decode("ascii").upper()
        cand_addr_clean = re.sub(r"[^A-Z0-9 ]", "", cand_addr_clean)
        cand_addr_clean = re.sub(r"\s+", " ", cand_addr_clean).strip()

        # Name variants
        c_names = [
            c.get("denomination"),
            c.get("denomination_unite_legale"),
            c.get("enseigne1"),
            c.get("enseigne2"),
            c.get("enseigne3")
        ]
        if c.get("nom_unite_legale"):
            if c.get("prenom1_unite_legale"):
                c_names.append(f"{c['nom_unite_legale']} {c['prenom1_unite_legale']}")
            else:
                c_names.append(c["nom_unite_legale"])
        c_names = [v for v in c_names if v]
        clean_variants = [clean_name(v) for v in c_names]

        max_name_ratio = 0.0
        is_exact_name = False
        is_substring_name = False

        for v_clean in clean_variants:
            if not v_clean:
                continue
            if crm_name_clean == v_clean:
                is_exact_name = True
                max_name_ratio = 1.0
                break
            ratio = SequenceMatcher(None, crm_name_clean, v_clean).ratio()
            if len(crm_name_clean) >= 3 and len(v_clean) >= 3 and (crm_name_clean in v_clean or v_clean in crm_name_clean):
                ratio = max(ratio, 0.80)
                is_substring_name = True
            if ratio > max_name_ratio:
                max_name_ratio = ratio

        if is_exact_name:
            name_score = 0.30
        elif max_name_ratio > 0.80 or is_substring_name:
            name_score = 0.30
        elif max_name_ratio > 0.55:
            name_score = 0.20
        elif max_name_ratio > 0.30:
            name_score = 0.10
        else:
            name_score = 0.0

        # Address scoring
        c_num = get_street_number(c.get("street_number") or c.get("address_full"))
        c_sig = clean_street_sig(c.get("street_name") or c.get("address_full"))

        # Check street number tolerance <= 2
        num_match = False
        if not crm_num or not c_num:
            num_match = True
        else:
            try:
                num_match = abs(int(crm_num) - int(c_num)) <= 2
            except ValueError:
                num_match = (crm_num == c_num)

        # Check street signature match
        street_sig_match = False
        if crm_sig and c_sig:
            if crm_sig == c_sig:
                street_sig_match = True
            elif len(crm_sig) >= 5 and len(c_sig) >= 5:
                if crm_sig in c_sig or c_sig in crm_sig:
                    street_sig_match = True
                else:
                    sig_ratio = SequenceMatcher(None, crm_sig, c_sig).ratio()
                    if sig_ratio > 0.60:
                        street_sig_match = True

        # Check postcode match (same department)
        crm_pc = row.get("postcode")
        cand_pc = c.get("postcode")
        pc_match = False
        if crm_pc and cand_pc:
            pc_match = (str(crm_pc)[:2] == str(cand_pc)[:2])

        # Evaluate physical street number conflict
        conflict = street_numbers_conflict(crm_num, c_num)

        if conflict:
            address_score = 0.15  # Cap the address score when numbers conflict
        elif crm_addr_clean and cand_addr_clean == crm_addr_clean:
            address_score = 0.50
        elif street_sig_match and pc_match and num_match:
            address_score = 0.50
        else:
            addr_ratio = SequenceMatcher(None, crm_addr_clean, cand_addr_clean).ratio() if crm_addr_clean and cand_addr_clean else 0.0
            if addr_ratio > 0.85:
                address_score = 0.50
            elif addr_ratio > 0.70:
                address_score = 0.30
            elif addr_ratio > 0.40:
                address_score = 0.15
            else:
                address_score = 0.0

        # Multi-source score (assume 0.20 if in cache or DataGouv)
        multisource_score = 0.20

        # Category score
        cand_category = "PRIVE"  # Default
        if c.get("denomination"):
            c_upper = c["denomination"].upper()
            if any(k in c_upper for k in ["COLLEGE", "MAIRIE", "ECOLE", "UNIVERSITE", "LYCEE", "PUBLIC", "COMMUNE"]):
                cand_category = "PUBLIC"
        category_score = 0.10 if cand_category == crm_category else 0.0

        conf = address_score + name_score + multisource_score + category_score

        # Boost confidence if exact/very close address AND name matches moderately well
        if address_score == 0.50 and not conflict and (max_name_ratio >= 0.50 or is_substring_name):
            conf += 0.20

        # Boost confidence if exact name match in the same postcode
        if is_exact_name and crm_pc and cand_pc and str(crm_pc).strip() == str(cand_pc).strip():
            conf += 0.05

        # Boost Sitiv/Commune mapping
        if "SITIV" in crm_name_upper:
            cand_name_upper = (c.get("denomination") or "").upper()
            if "COMMUNE" in cand_name_upper or "MAIRIE" in cand_name_upper:
                if address_score == 0.50:
                    conf = max(conf, 0.85)

        is_street_number_match = (crm_num and c_num and crm_num == c_num and crm_num != "0")
        if has_any_exact_name and not is_exact_name and max_name_ratio < 0.60:
            if not is_street_number_match:
                conf -= 0.20

        # Tiny tie-breaker boost for physical street number matches
        if is_street_number_match:
            conf += 0.01

        conf = max(0.0, min(1.0, conf))

        # ----------------------------------------------------
        # MULTI-ROUND OVERRIDES & RESCUES
        # ----------------------------------------------------
        round_applied = "Standard Scoring"

        # Round 4: Street Signature Match (ignored house numbers)
        # Only allow rescue if there is no conflict, or the name matches exactly
        if conf < 0.85:
            if street_sig_match and pc_match:
                if not conflict or is_exact_name:
                    if is_exact_name or max_name_ratio >= 0.80 or is_substring_name:
                        conf = max(conf, 0.85)
                        round_applied = "Round 4 Rescue: Street Sig + Strong Name Match"

        # Round 5: Collapsed Address Match ("5BAC" format)
        if conf < 0.85:
            c_collapsed = collapse_address_ultra(c.get("street_number"), c.get("street_name"), c.get("address_full"))
            collapsed_match = False
            if crm_collapsed and c_collapsed:
                if crm_collapsed in c_collapsed or c_collapsed in crm_collapsed:
                    collapsed_match = True
                else:
                    col_ratio = SequenceMatcher(None, crm_collapsed, c_collapsed).ratio()
                    if col_ratio > 0.80:
                        collapsed_match = True
            
            # If the street numbers conflict, we do NOT allow collapsed match unless the name is exact!
            if collapsed_match:
                if conflict and not is_exact_name:
                    collapsed_match = False
                        
            if collapsed_match and (max_name_ratio >= 0.70 or is_substring_name):
                conf = max(conf, 0.85)
                round_applied = f"Round 5 Rescue: Collapsed Address Match ('{crm_collapsed}' vs '{c_collapsed}')"

        # Name-based score capping to prevent false positives when names are completely different
        if not is_exact_name and not is_substring_name:
            if max_name_ratio < 0.40:
                conf = min(conf, 0.50)  # Complete name mismatch -> NO_MATCH

        if conf > best_conf:
            best_conf = conf
            best_match = {
                "status": "MATCH" if conf >= 0.85 else ("REVIEW" if conf >= 0.60 else "NO_MATCH"),
                "siret": c.get("siret"),
                "siren": c.get("siren"),
                "name": c.get("denomination") or c.get("denomination_unite_legale") or c.get("nom_unite_legale"),
                "score": round(conf, 2),
                "reason": round_applied + f" (BaseConf={conf:.2f})"
            }

    return best_match



# ============================================================================
# CRM Column & Separator Auto-detection
# ============================================================================

def detect_crm_columns_and_sep(csv_path: Path) -> tuple[dict[str, str] | None, str, str]:
    """Detects separator and column map for the given CRM CSV file."""
    first_line = ""
    for enc in ["utf-8", "latin-1"]:
        try:
            with open(csv_path, "r", encoding=enc) as f:
                first_line = f.readline()
            if first_line:
                break
        except Exception:
            continue

    if not first_line:
        return None, ";", "french_default"

    # Detect separator
    if ";" in first_line:
        sep = ";"
    elif "," in first_line:
        sep = ","
    else:
        sep = ";"  # fallback

    headers = [h.strip('"').strip("'").strip() for h in first_line.split(sep)]

    if "crm_id" in headers:
        column_map = {
            "crm_id": "crm_id",
            "name": "crm_name",
            "street": "crm_adresse",
            "city": "crm_commune",
            "postcode": "crm_cp",
            "insee": "crm_insee",
        }
        return column_map, sep, "database"
    
    return None, sep, "french"


# ============================================================================
# Main Runner Flow
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Multi-Round matching pipeline on CRM dataset")
    parser.add_argument("--crm-path", type=Path, default=Path("data/testcrm/data_56_subset_corbas_decines.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/reports_normalized/results_multi_round.csv"))
    parser.add_argument("--output-xlsx", type=Path, default=Path("C:/Users/Kabouassi/Desktop/results_multi_round.xlsx"))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    # Setup config & connection
    config = load_config(None)
    setup_logging(config)

    db_path = Path(config.sqlite_path)
    if not db_path.exists():
        LOGGER.error("SQLite database cache not found at: %s", db_path)
        return 1

    conn = sqlite3.connect(db_path)
    LOGGER.info("Connected to SIRENE SQLite Cache: %s", db_path)

    # Detect separator and columns automatically
    column_map, sep, detected_style = detect_crm_columns_and_sep(args.crm_path)
    LOGGER.info("Detected CRM style '%s' with separator '%s'", detected_style, sep)

    # Preprocess CRM path to fill missing crm_id values if any
    # This avoids "ValueError: CRM data contains empty 'crm_id' values after normalization"
    temp_crm_path = args.crm_path
    has_temp_file = False
    try:
        # Determine encoding first
        input_enc = "utf-8"
        for enc in ["utf-8", "latin-1"]:
            try:
                pd.read_csv(args.crm_path, sep=sep, encoding=enc, nrows=5)
                input_enc = enc
                break
            except Exception:
                continue

        df_raw = pd.read_csv(args.crm_path, sep=sep, dtype=str, encoding=input_enc)
        
        # Determine crm_id column name
        crm_id_col = "N° Service"  # Default French
        if column_map and "crm_id" in column_map:
            crm_id_col = column_map["crm_id"]
        elif "crm_id" in df_raw.columns:
            crm_id_col = "crm_id"
            
        if crm_id_col in df_raw.columns:
            null_mask = df_raw[crm_id_col].isna() | (df_raw[crm_id_col].str.strip() == "")
            if null_mask.any():
                LOGGER.info("Found %d empty/null crm_id values. Generating temporary unique IDs...", null_mask.sum())
                df_raw.loc[null_mask, crm_id_col] = [f"GEN_ID_{i:06d}" for i in range(null_mask.sum())]
                temp_crm_path = args.crm_path.parent / f"{args.crm_path.stem}_cleaned.csv"
                df_raw.to_csv(temp_crm_path, sep=sep, index=False, encoding=input_enc)
                has_temp_file = True
    except Exception as e:
        LOGGER.warning("Failed to preprocess CRM file to clean null crm_id values: %s. Proceeding directly.", e)

    try:
        try:
            df_crm = load_crm(temp_crm_path, column_map=column_map, encoding="utf-8", sep=sep)
        except Exception as e:
            LOGGER.info("Loading CRM with UTF-8 failed (%s), trying Latin-1...", e)
            df_crm = load_crm(temp_crm_path, column_map=column_map, encoding="latin-1", sep=sep)
    finally:
        if has_temp_file and temp_crm_path.exists():
            try:
                temp_crm_path.unlink()
                LOGGER.info("Cleaned up temporary preprocessed CRM file: %s", temp_crm_path)
            except Exception as e:
                LOGGER.warning("Failed to delete temporary file %s: %s", temp_crm_path, e)

    if args.max_rows:
        df_crm = df_crm.head(args.max_rows)
    LOGGER.info("Loaded CRM file via crm_loader: %s (%d rows)", args.crm_path, len(df_crm))

    results = []
    
    for idx, row in tqdm(df_crm.iterrows(), total=len(df_crm), desc="Matching Multi-Round"):
        crm_id = row.get("crm_id")
        crm_name = row.get("crm_name")
        
        # Get candidate set
        cands = get_candidates(row, conn, config)
        
        # Match using our 5 rounds
        match_decision = match_row_multi_round(row, cands)
        
        results.append({
            "crm_id": crm_id,
            "crm_name": crm_name,
            "street_number": row.get("street_number"),
            "street_name": row.get("street_name"),
            "postcode": row.get("postcode"),
            "city": row.get("city"),
            "status": match_decision["status"],
            "matched_siret": match_decision["siret"],
            "matched_siren": match_decision["siren"],
            "matched_name": match_decision["name"],
            "confidence": match_decision["score"],
            "reason": match_decision["reason"]
        })

    conn.close()

    # Save outputs
    df_out = pd.DataFrame(results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False, sep=";")
    LOGGER.info("Saved CSV results to: %s", args.output_csv)

    try:
        # Convert to Excel directly on Desktop
        df_out.to_excel(args.output_xlsx, index=False)
        LOGGER.info("Saved Excel results directly to Desktop: %s", args.output_xlsx)
    except Exception as e:
        LOGGER.error("Failed to export Excel to Desktop: %s", e)

    # Calculate match statistics
    stats = df_out["status"].value_counts()
    total = len(df_out)
    
    print("\n" + "="*50)
    print("           MULTI-ROUND MATCH STATISTICS")
    print("="*50)
    for status, count in stats.items():
        pct = (count / total) * 100
        print(f" {status:<10} : {count:<5} ({pct:.1f}%)")
    print("="*50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
