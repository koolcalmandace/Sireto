"""
Shared feature engineering functions for XGBoost SIRET matcher.

This module is used by both training and inference scripts to ensure
consistency (no train/serve skew).

Features computed:
  - name_jaro: Jaro-Winkler similarity between normalized names
  - name_levenshtein: Normalized Levenshtein similarity between names
  - addr_jaro: Jaro-Winkler similarity between normalized addresses
  - addr_levenshtein: Normalized Levenshtein similarity between addresses
  - postcode_match: 1 if postcodes match exactly, 0 otherwise
  - city_match: 1 if normalized city names match, 0 otherwise
  - street_number_diff: Absolute difference between street numbers (9999 if unavailable)
  - name_token_overlap: Ratio of common words between names
  - name_first_word_match: 1 if first words match
  - name_contains_crm: 1 if candidate name contains CRM name
  - name_crm_contains_cand: 1 if CRM name contains candidate name
  - acronym_match: 1 if acronym matches expanded form
  - idf_name: Average IDF of overlapping name tokens
  - numeric_token_match: Numeric token overlap between names
  - addr_token_overlap: Ratio of common words in addresses
  - address_density: Number of candidates sharing the same address (per location)
  - street_name_jaro: Jaro similarity on street name only (no number)
  - legal_form_category: Encoded legal form category (PUBLIC/PRIVE/INCONNU)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Set

import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Inlined from pipe_v6.category_mapping to avoid circular imports
LEGAL_NATURE_MAPPING = {
    # PUBLIC
    "7111": "PUBLIC", "7112": "PUBLIC", "7120": "PUBLIC", "7150": "PUBLIC", "7160": "PUBLIC",
    "7210": "PUBLIC", "7220": "PUBLIC", "7230": "PUBLIC", "7312": "PUBLIC", "7313": "PUBLIC",
    "7314": "PUBLIC", "7321": "PUBLIC", "7331": "PUBLIC", "7344": "PUBLIC", "7345": "PUBLIC",
    "7346": "PUBLIC", "7347": "PUBLIC", "7348": "PUBLIC", "7354": "PUBLIC", "7389": "PUBLIC",
    "5532": "PUBLIC", "5542": "PUBLIC", "5552": "PUBLIC", "5558": "PUBLIC",
    "7171": "PUBLIC", "7353": "PUBLIC", "7361": "PUBLIC",
    # PRIVE
    "2110": "PRIVE", "5660": "PRIVE", "6596": "PRIVE",
    "5410": "PRIVE", "5415": "PRIVE", "5499": "PRIVE", "5505": "PRIVE", "5510": "PRIVE",
    "5560": "PRIVE", "5599": "PRIVE", "5710": "PRIVE", "5720": "PRIVE", "5770": "PRIVE", "5785": "PRIVE",
    "1000": "PRIVE", "1100": "PRIVE", "1200": "PRIVE", "1300": "PRIVE", "1400": "PRIVE",
    "1500": "PRIVE", "1600": "PRIVE",
    "9220": "PRIVE", "9230": "PRIVE", "9240": "PRIVE", "9260": "PRIVE", "9300": "PRIVE",
    "5202": "PRIVE", "5485": "PRIVE", "5699": "PRIVE", "6540": "PRIVE", "6541": "PRIVE",
    "6589": "PRIVE", "6599": "PRIVE", "6220": "PRIVE", "9150": "PRIVE", "3120": "PRIVE",
}

def map_legal_to_category(legal_nature: str | None) -> str:
    """Convert SIRENE legal nature code to PUBLIC/PRIVE/INCONNU."""
    if not legal_nature:
        return "INCONNU"
    return LEGAL_NATURE_MAPPING.get(legal_nature, "INCONNU")


from .communes_nouvelles import resolve_commune_nouvelle
from .naming import (
    CandidateName,
    NameSource,
    build_candidate_names,
    looks_like_person_name,
    normalize_name,
    normalize_text,
)


# Thresholds for city-like detection
CITY_OVERLAP_THRESHOLD = 0.7
CITY_MAX_TOKENS = 3
SEMANTIC_GATE_JARO = float(os.getenv("XGB_SEMANTIC_GATE_JARO", "0.90"))
SEMANTIC_GATE_ENABLED = os.getenv("XGB_SEMANTIC_GATE_ENABLED", "0") == "1"
SEMANTIC_GATE_JARO_MIN = float(os.getenv("XGB_SEMANTIC_GATE_JARO_MIN", "0.50"))
SEMANTIC_GATE_TOKEN_MIN = float(os.getenv("XGB_SEMANTIC_GATE_TOKEN_MIN", "0.20"))

# Stopwords to down-weight generic name overlap (avoid LES/DU false positives)
NAME_STOPWORDS: Set[str] = {
    "LES", "DU", "DE", "LA", "LE", "DES", "D", "L", "AUX", "AU",
}

# Global IDF map for name tokens (set by candidate loader)
_GLOBAL_NAME_IDF: Dict[str, float] = {}
_GLOBAL_NAME_IDF_DEFAULT: float = 0.0


def set_global_name_idf_map(idf_map: Mapping[str, float] | None, default_idf: float | None = None) -> None:
    """Register a global IDF map for name tokens (used by idf_name feature)."""
    global _GLOBAL_NAME_IDF, _GLOBAL_NAME_IDF_DEFAULT
    _GLOBAL_NAME_IDF = dict(idf_map) if idf_map else {}
    if default_idf is None:
        if _GLOBAL_NAME_IDF:
            _GLOBAL_NAME_IDF_DEFAULT = max(_GLOBAL_NAME_IDF.values())
        else:
            _GLOBAL_NAME_IDF_DEFAULT = 0.0
    else:
        _GLOBAL_NAME_IDF_DEFAULT = float(default_idf)


def semantic_gate_allows(name_jaro_max: float | None, name_token_overlap_max: float | None) -> bool:
    """Return True if semantic features are allowed under lexical gating."""
    if not SEMANTIC_GATE_ENABLED:
        return True
    try:
        jaro = float(name_jaro_max or 0.0)
    except Exception:
        jaro = 0.0
    try:
        tok = float(name_token_overlap_max or 0.0)
    except Exception:
        tok = 0.0
    return not (jaro < SEMANTIC_GATE_JARO_MIN and tok < SEMANTIC_GATE_TOKEN_MIN)


# Feature names in order (used for training and inference)
FEATURE_NAMES: List[str] = [
    # Aggregated name similarities (bag of names)
    "has_any_name",
    "name_count",
    "name_jaro_max",
    "name_jaro_second",
    "name_jaro_gap",
    "name_levenshtein_max",
    "name_token_overlap_max",
    "idf_name",
    "numeric_token_match",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "name_sim_max_pm_dirigeant",
    "type_of_max_name",
    "is_ul_name_max",
    "is_sigle_max",
    "name_length_max",
    "has_person_name",
    "person_name_jaro_max",
    "name_city_overlap_max",
    "name_is_city_like_max",
    "name_semantic_max",
    "name_semantic_second",
    "name_semantic_gap",
    # Address / location features (unchanged)
    "addr_jaro",
    "addr_levenshtein",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "address_density",
    "street_name_jaro",
    "name_addr_consistency",
    "legal_form_category",
    # New features from reranking rules (Phase 3)
    "is_siege",           # D9: head office indicator
    "is_association",     # B1: association candidate indicator
    "alias_match",        # D6: alias in parentheses match
    "token_overlap_ul",   # D11: token overlap with UL names
    "ul_vs_pm_indicator", # D7: UL vs PM dirigeant source indicator
    "is_crm_school",      # Context: CRM query refers to a school
    # V6 discriminant features
    "geo_exact_match",    # INSEE CRM == INSEE candidat (1 if match, 0 otherwise)
    "name_norm_exact",    # Exact match after normalization (1 if identical, 0 otherwise)
    "street_number_match", # Street number exact match (1 if match, 0 otherwise)
    # V8 new interaction features for colocataires, homonymes, multi-site SIREN
    "addr_unsupported_by_name",    # addr_jaro * (1 - name_jaro_max) -- penalizes address-only match
    "name_density_penalty",        # address_density * (1 - name_jaro_max) -- penalizes many candidates with weak name
    "addr_jaro_per_density",       # addr_jaro / log(1 + address_density) -- normalizes address quality by density
    "postcode_match_without_addr", # postcode_match * (1 - street_name_jaro) -- detects geographic homonymes
    "full_addr_match_score",       # weighted address completeness: 0.5*street_name_jaro + 0.3*street_number_match + 0.2*postcode_match
    "name_jaro_vs_enseigne",       # Jaro between CRM name and establishment-level name (enseigne), not UL name
    "name_city_suffix_match",      # ratio of CRM city tokens appearing in candidate enseigne
    "is_copropriete",              # Syndicate of co-owners indicator (cj_ul == '9110')
    "sidecar_match",               # 1.0 if identified by Sidecar (trade name, brand, SSDA anchor), 0.0 otherwise
    "is_active",                   # 1.0 if establishment is active (etat_admin == 'A'), 0.0 otherwise
    # BAN spatial features (Base Adresse Nationale continuous GPS coordinates)
    "spatial_distance_km",        # Equirectangular GPS distance between CRM and candidate address (km, 0.0 to 99.0)
    "spatial_is_very_close",      # 1.0 if distance < 0.100 km (~100m), 0.0 otherwise
    "spatial_is_same_block",      # 1.0 if distance < 0.300 km (~300m), 0.0 otherwise
    "spatial_has_coords",         # 1.0 if both CRM and candidate addresses geocoded successfully, 0.0 otherwise
    # V9 generic mathematical & categorical features (Stage 1 NDCG@50 & retention)
    "exact_address_anchor",       # 1.0 if street_number_match == 1.0 and street_name_jaro >= 0.85, 0.0 otherwise
    "is_commune_authority",       # 1.0 if candidate is local authority (cj in 72xx, 73xx) and in same commune (geo_exact_match == 1.0)
    "acronym_mutual_token_match", # 1.0 if CRM tokens contain acronym of cand or cand tokens/sigle contain acronym of CRM
    # Stage 2 Discriminant Features (Hit@1 Maximization)
    "delta_addr_vs_siege",        # addr concordance differential vs head office ([-1.0, 1.0])
    "naf_vector_similarity",      # continuous cosine similarity between CRM name and official INSEE NAF
    "civil_name_jaro",            # dual permutation Jaro similarity for natural persons (Nom Prenom / Prenom Nom)
]

SEMANTIC_FEATURE_NAMES: List[str] = [
    name for name in FEATURE_NAMES if name.startswith("name_semantic_")
]

FAST_RANKER_FEATURE_SET: Set[str] = {
    "has_any_name",
    "name_count",
    "name_jaro_max",
    "name_jaro_second",
    "name_jaro_gap",
    "name_levenshtein_max",
    "name_token_overlap_max",
    "idf_name",
    "numeric_token_match",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "name_sim_max_pm_dirigeant",
    "is_ul_name_max",
    "is_sigle_max",
    "name_length_max",
    "addr_jaro",
    "addr_levenshtein",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "address_density",
    "street_name_jaro",
    "name_addr_consistency",
    "geo_exact_match",
    "name_norm_exact",
    "street_number_match",
    "name_jaro_vs_enseigne",
    "alias_match",
    "sidecar_match",
    "is_active",
    "is_siege",
    "is_association",
    "addr_unsupported_by_name",
    "full_addr_match_score",
    "spatial_distance_km",
    "spatial_is_very_close",
    "spatial_is_same_block",
    "spatial_has_coords",
    "exact_address_anchor",
    "is_commune_authority",
    "acronym_mutual_token_match",
}

FAST_RANKER_FEATURE_NAMES: List[str] = [
    name for name in FEATURE_NAMES if name in FAST_RANKER_FEATURE_SET
]


# Official INSEE Legal Classification (Generic, Data-Driven)
# Associations are officially defined under INSEE category 92xx (9210, 9220, 9230, 9240, 9260)
INSEE_ASSOCIATION_CJ_PREFIX: str = "92"


# --------------------------------------------------------------------------- #
# Similarity functions
# --------------------------------------------------------------------------- #


def jaro_sim(a: str, b: str) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    Returns 1.0 if both empty, 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


def levenshtein_norm(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein similarity (1 - distance/max_len).

    Returns 1.0 if both empty, 0.0 if only one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


# --------------------------------------------------------------------------- #
# CRM name cleanup
# --------------------------------------------------------------------------- #


def _remove_tokens(text: str, tokens_to_remove: Set[str]) -> str:
    if not text:
        return ""
    tokens = text.split()
    filtered = [t for t in tokens if t not in tokens_to_remove]
    return " ".join(filtered)


def strip_location_from_crm_name(crm_row: pd.Series, preserve_case: bool = False) -> str:
    """Remove city/CP/INSEE tokens from the CRM name to avoid city bias.

    Keeps the normalized version; if everything is removed, fall back to the original
    normalized name to avoid empty strings.
    """

    original_norm = normalize_name(crm_row.get("crm_name", ""), uppercase=not preserve_case)
    if not original_norm:
        return ""

    city = normalize_text(crm_row.get("crm_city_addr") or crm_row.get("crm_city"))
    postcode = normalize_text(crm_row.get("postcode"))
    insee = normalize_text(crm_row.get("insee"))

    tokens_to_remove: Set[str] = set()
    if city:
        tokens_to_remove.update(city.split())
    if postcode:
        tokens_to_remove.add(postcode)
    if insee:
        tokens_to_remove.add(insee)

    # Use a case-insensitive match for removal if preserving case
    if preserve_case:
        tokens = original_norm.split()
        filtered = [t for t in tokens if t.upper() not in tokens_to_remove]
        cleaned = " ".join(filtered)
    else:
        cleaned = _remove_tokens(original_norm, tokens_to_remove)
        
    return cleaned or original_norm


# --------------------------------------------------------------------------- #
# Address building
# --------------------------------------------------------------------------- #


def build_address(cand: dict) -> str:
    """
    Build normalized address from SIRENE candidate fields.

    Uses: numeroVoie + typeVoie + libelleVoie
    Falls back to complementAdresse if main fields are empty.

    Args:
        cand: Dictionary with SIRENE candidate data

    Returns:
        Normalized address string
    """
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_addr_norm")
        if isinstance(cached, str):
            return cached

    parts = [
        cand.get("numeroVoie"),
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    addr = " ".join(str(x) for x in parts if x and str(x).strip())
    if not addr and cand.get("complementAdresse"):
        addr = str(cand["complementAdresse"])
    addr_norm = normalize_text(addr)
    if isinstance(cand, dict):
        cand["_xgb_cached_addr_norm"] = addr_norm
    return addr_norm


CLEAN_COMPLEMENT_REGEX = re.compile(
    r"\b(BP\s*\d+|CEDEX\s*\d*|ETAGE\s*\d+|CS\s*\d+|BOITE\s*\d+)\b",
    re.IGNORECASE,
)


def build_address_for_retrieval(cand: dict) -> str:
    """
    Build address string for candidate retrieval (TF-IDF), incorporating
    cleaned complementAdresse to capture business parks, industrial zones,
    and lieux-dits without corrupting street-level matching.
    """
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_addr_retrieval")
        if isinstance(cached, str):
            return cached

    parts = [
        cand.get("numeroVoie"),
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    voie = " ".join(str(x) for x in parts if x and str(x).strip())
    comp = str(cand.get("complementAdresse") or "").strip()
    if comp:
        comp_clean = CLEAN_COMPLEMENT_REGEX.sub(" ", comp)
        comp_clean = re.sub(r"\s+", " ", comp_clean).strip()
        if comp_clean:
            voie = f"{voie} {comp_clean}".strip() if voie else comp_clean
    addr_norm = normalize_text(voie)
    if isinstance(cand, dict):
        cand["_xgb_cached_addr_retrieval"] = addr_norm
    return addr_norm


# --------------------------------------------------------------------------- #
# Street type normalization (BAN official nomenclature)
# --------------------------------------------------------------------------- #

_STREET_TYPE_NORMALIZATION: Dict[str, str] = {
    "AV": "AVENUE",
    "BD": "BOULEVARD",
    "BLVD": "BOULEVARD",
    "R": "RUE",
    "RTE": "ROUTE",
    "IMP": "IMPASSE",
    "ALL": "ALLEE",
    "CHE": "CHEMIN",
    "CRS": "COURS",
    "PL": "PLACE",
    "QU": "QUAI",
    "RES": "RESIDENCE",
    "SQ": "SQUARE",
    "ZA": "ZONE ARTISANALE",
    "ZI": "ZONE INDUSTRIELLE",
    "ZAC": "ZONE D AMENAGEMENT CONCERTE",
}
_STREET_TYPE_REGEX = re.compile(
    r"\b(" + "|".join(sorted(_STREET_TYPE_NORMALIZATION.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_street_type(address: str | None) -> str:
    """Normalize French street type abbreviations according to BAN standard."""
    if not address:
        return ""
    return _STREET_TYPE_REGEX.sub(
        lambda m: _STREET_TYPE_NORMALIZATION[m.group(0).upper()],
        str(address),
    )


def parse_street_numbers(num_str: str | None) -> Set[int]:
    """Parse integer street numbers from a number or range string (e.g. '66-68' -> {66, 68})."""
    if not num_str:
        return set()
    nums: Set[int] = set()
    raw_parts = re.findall(r"\d+", str(num_str))
    for part in raw_parts:
        try:
            nums.add(int(part))
        except ValueError:
            pass
    # If range like '6-8' or '6 a 10', expand intermediate values if range is short (<= 8)
    if len(raw_parts) == 2:
        try:
            p0, p1 = int(raw_parts[0]), int(raw_parts[1])
            if 0 < p1 - p0 <= 8:
                step = 2 if (p0 % 2 == p1 % 2) else 1
                nums.update(range(p0, p1 + 1, step))
        except ValueError:
            pass
    return nums


def extract_street_number(addr: str | None) -> str | None:
    """
    Extract street number or number range from address string.

    Handles:
      - Ranges like '66-68', '66/68', '66 - 68' -> returns '66-68'
      - Standard leading numbers like '12', '12B', '12 BIS'
      - Multi-position numbers when preceded by building/BP prefixes (e.g. 'Batiment B, 14 rue de Paris')

    Args:
        addr: Address string

    Returns:
        Street number or range string, or None if not found
    """
    if not addr:
        return None
    addr_str = str(addr).strip()
    if not addr_str:
        return None

    # 1. Range check (e.g. '66-68' or '66 - 68' or '66/68')
    range_match = re.search(r"\b(\d+)\s*[-/]\s*(\d+)\b", addr_str)
    if range_match:
        return f"{range_match.group(1)}-{range_match.group(2)}"

    # 2. Leading digits (standard: '12 rue de la Paix', '12B rue...')
    lead_match = re.match(r"^(\d+)", addr_str)
    if lead_match:
        return lead_match.group(1)

    # 3. Clean multi-position fallback (filtering BP, Batiment, Etage, CS, etc.)
    cleaned = re.sub(
        r"\b(?:BP|CS|BOITE\s+POSTALE|BATIMENT|BAT|ETAGE|PORTE|LOCAL|IMMEUBLE)\s*[A-Z0-9-]+\b",
        "",
        addr_str,
        flags=re.IGNORECASE,
    )
    multi_match = re.search(r"(?<!\d)([1-9]\d{0,3})(?!\d)", cleaned)
    if multi_match:
        return multi_match.group(1)

    return None


def get_street_name(cand: dict) -> str:
    """
    Get only the street name part (typeVoie + libelleVoie), excluding number.

    Used for street_name_jaro feature to compare street names independently.
    """
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_street_name_norm")
        if isinstance(cached, str):
            return cached
    parts = [
        cand.get("typeVoie"),
        cand.get("libelleVoie"),
    ]
    norm = normalize_text(" ".join(str(x) for x in parts if x and str(x).strip()))
    if isinstance(cand, dict):
        cand["_xgb_cached_street_name_norm"] = norm
    return norm


def _candidate_city_norm(cand: dict) -> str:
    if isinstance(cand, dict):
        cached = cand.get("_xgb_cached_city_norm")
        if isinstance(cached, str):
            return cached
    norm = normalize_text(cand.get("city"))
    if isinstance(cand, dict):
        cand["_xgb_cached_city_norm"] = norm
    return norm


def extract_street_name_from_address(addr: str | None) -> str:
    """
    Extract street name from full address string (remove leading number).

    Args:
        addr: Full address string

    Returns:
        Street name without the leading number
    """
    if not addr:
        return ""
    addr_str = normalize_text(addr)
    # Remove leading number and any bis/ter suffix with word boundary protection
    cleaned = re.sub(r"^\d+\s*(?:BIS|TER|QUATER|[BTQ])?\b\s*", "", addr_str, flags=re.IGNORECASE)
    return cleaned.strip()


# --------------------------------------------------------------------------- #
# Matching functions
# --------------------------------------------------------------------------- #


def postal_match(cp1: str | None, cp2: str | None) -> int:
    """Check if two postal codes match exactly."""
    if not cp1 or not cp2:
        return 0
    return int(str(cp1).strip() == str(cp2).strip())


def city_match(c1: str | None, c2: str | None) -> int:
    """Check if two city names match after normalization."""
    if not c1 or not c2:
        return 0
    return int(normalize_text(c1) == normalize_text(c2))


def street_number_diff(n1: str | None, n2: str | None) -> float:
    """
    Compute absolute difference between street numbers.
    Supports intervals (e.g. '66-68' vs '68' -> 0.0).

    Returns 9999.0 if either number is unavailable or invalid.
    """
    try:
        nums1 = parse_street_numbers(n1)
        nums2 = parse_street_numbers(n2)
        if nums1 and nums2:
            if nums1 & nums2:
                return 0.0
            return float(min(abs(c1 - c2) for c1 in nums1 for c2 in nums2))
        return 9999.0
    except (TypeError, ValueError):
        return 9999.0


def _address_density_for_candidate(crm: Mapping[str, Any], cand: dict) -> float:
    """Return address density for the candidate, preferring INSEE-based grouping."""
    insee = crm.get("insee")
    if insee:
        density = cand.get("_xgb_addr_density_insee")
        if density is not None:
            return float(density)
    density = cand.get("_xgb_addr_density_cp")
    if density is not None:
        return float(density)
    density = cand.get("address_density")
    if density is not None:
        return float(density)
    return 1.0


# --------------------------------------------------------------------------- #
# Token-based features (new for improved ranking)
# --------------------------------------------------------------------------- #


def _tokenize(
    text: str,
    *,
    stopwords: Set[str] | None = None,
    min_len: int = 1,
) -> Set[str]:
    """Split normalized text into set of tokens (words), with optional filtering."""
    if not text:
        return set()
    tokens = text.split()
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    return set(tokens)


def token_overlap(
    a: str,
    b: str,
    *,
    stopwords: Set[str] | None = None,
    min_len: int = 1,
) -> float:
    """
    Compute ratio of common tokens between two strings.

    Returns |A ∩ B| / |A ∪ B| (Jaccard similarity on tokens).
    """
    tokens_a = _tokenize(a, stopwords=stopwords, min_len=min_len)
    tokens_b = _tokenize(b, stopwords=stopwords, min_len=min_len)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def first_word_match(a: str, b: str) -> int:
    """
    Check if the first word of both strings matches.

    Useful for "SARL X" vs "X SARL" patterns.
    """
    if not a or not b:
        return 0
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return 0
    return int(words_a[0] == words_b[0])


def contains_check(container: str, contained: str) -> int:
    """
    Check if container string contains the contained string.

    Useful for detecting "GE FRUITS" in "GE FRUITS DISTRIBUTION".
    """
    if not container or not contained:
        return 0
    return int(contained in container)


def _idf_overlap(a: str, b: str) -> float:
    """Average IDF of overlapping name tokens (0 if no overlap or no IDF map)."""
    if not _GLOBAL_NAME_IDF:
        return 0.0
    tokens_a = _tokenize(a, stopwords=NAME_STOPWORDS, min_len=2)
    tokens_b = _tokenize(b, stopwords=NAME_STOPWORDS, min_len=2)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0
    return sum(_GLOBAL_NAME_IDF.get(tok, _GLOBAL_NAME_IDF_DEFAULT) for tok in overlap) / len(overlap)


def numeric_token_match(a: str, b: str) -> float:
    """Overlap ratio of numeric tokens between two strings."""
    nums_a = set(re.findall(r"\d+", a or ""))
    nums_b = set(re.findall(r"\d+", b or ""))
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / max(len(nums_a), len(nums_b))


def _extract_acronym(text: str) -> str:
    """Extract initials from a multi-word string, ignoring small words and common city names."""
    if not text:
        return ""
    
    # Common French stopwords and specific geography to ignore in acronyms
    STOPWORDS = {"DE", "DU", "LA", "LE", "DES", "ET", "LES", "D", "L", "AUX", "AU"}
    GEOGRAPHY = {"LYON", "CORBAS", "DECINES", "MEYZIEU", "VILLEURBANNE"}
    
    words = text.split()
    if len(words) <= 1:
        return ""
        
    initials = []
    for w in words:
        w_norm = w.strip().upper()
        if not w_norm:
            continue
        if w_norm in STOPWORDS or w_norm in GEOGRAPHY:
            continue
        initials.append(w_norm[0])
        
    return "".join(initials)


def _is_acronym(text: str) -> bool:
    """Check if text looks like an acronym (2-6 uppercase letters, no spaces)."""
    if not text:
        return False
    # After normalization, acronyms are uppercase words with 2-6 chars
    return bool(re.match(r"^[A-Z]{2,6}$", text.replace(" ", "")))


def acronym_match(a: str, b: str) -> int:
    """
    Check if one string is the acronym of the other.

    Handles cases like "SNCF" ↔ "SOCIETE NATIONALE DES CHEMINS DE FER".
    """
    if not a or not b:
        return 0

    norm_a = a.replace(" ", "").upper()
    norm_b = b.replace(" ", "").upper()
    a_is_acronym = _is_acronym(norm_a)
    b_is_acronym = _is_acronym(norm_b)

    # Both are acronyms: exact match (fixes C5 bug where SNCF vs SNCF returned 0)
    if a_is_acronym and b_is_acronym:
        return int(norm_a == norm_b)

    if a_is_acronym and not b_is_acronym:
        query_acro = norm_a
        cand_acro = _extract_acronym(b)
        
        # 1. Exact match
        if query_acro == cand_acro:
            return 1
        # 2. Prefix match (e.g. ASL matches Association Syndicale Libre ...)
        if len(query_acro) >= 2 and cand_acro.startswith(query_acro):
            return 1
        return 0
    elif b_is_acronym and not a_is_acronym:
        query_acro = norm_b
        cand_acro = _extract_acronym(a)
        if query_acro == cand_acro or (len(query_acro) >= 2 and cand_acro.startswith(query_acro)):
            return 1

    return 0


# --------------------------------------------------------------------------- #
# Aggregation helpers for bag-of-names strategy
# --------------------------------------------------------------------------- #


_SOURCE_ENCODING = {
    NameSource.NONE: 0,
    NameSource.ETAB_ENSEIGNE: 1,
    NameSource.ETAB_DENOM: 2,
    NameSource.ETAB_ENSEIGNE_X: 3,
    NameSource.PM_DIRIGEANT: 4,
    NameSource.UL_SIGLE: 5,
    NameSource.UL_DENOM_USUELLE: 6,
    NameSource.UL_DENOM: 7,
    NameSource.PERSON_NAME: 8,
}


def _encode_source(source: NameSource | str) -> int:
    """Map NameSource to an ordinal value for model consumption."""
    return _SOURCE_ENCODING.get(source, 0)


def _init_name_feature_defaults() -> Dict[str, float]:
    """Default values when no candidate names are available."""
    return {
        "has_any_name": 0.0,
        "name_count": 0.0,
        "name_jaro_max": 0.0,
        "name_jaro_second": 0.0,
        "name_jaro_gap": 0.0,
        "name_levenshtein_max": 0.0,
        "name_token_overlap_max": 0.0,
        "idf_name": 0.0,
        "numeric_token_match": 0.0,
        "name_first_word_match_max": 0.0,
        "name_contains_crm_max": 0.0,
        "name_crm_contains_cand_max": 0.0,
        "acronym_match_max": 0.0,
        "name_sim_max_etab": 0.0,
        "name_sim_max_ul": 0.0,
        "name_sim_max_sigle": 0.0,
        "name_sim_max_pm_dirigeant": 0.0,
        "type_of_max_name": 0.0,
        "is_ul_name_max": 0.0,
        "is_sigle_max": 0.0,
        "name_length_max": 0.0,
        "has_person_name": 0.0,
        "person_name_jaro_max": 0.0,
        "name_city_overlap_max": 0.0,
        "name_is_city_like_max": 0.0,
        "name_semantic_max": 0.0,
        "name_semantic_second": 0.0,
        "name_semantic_gap": 0.0,
    }


# --------------------------------------------------------------------------- #
# Main feature computation
# --------------------------------------------------------------------------- #

def preprocess_crm_row(crm_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Precompute CRM-side normalizations once per row (for faster inference)."""
    crm_address_raw = str(crm_row.get("crm_address") or crm_row.get("crm_adresse") or crm_row.get("adresse") or "")
    crm_address_norm = normalize_street_type(crm_address_raw)
    crm_raw_name = str(crm_row.get("crm_name") or "")
    crm_name = strip_location_from_crm_name(crm_row)
    crm_city = crm_row.get("crm_city_addr") or crm_row.get("crm_city", "")

    # Multi-representation on delimiters (' - ', ' / ', ' // ') on raw name before hyphens are stripped into spaces
    crm_name_root = crm_name
    delim_match = re.search(r"\s+[-/]\s+", crm_raw_name)
    if delim_match:
        root_candidate = crm_raw_name[:delim_match.start()].strip()
        if len(root_candidate) >= 3:
            crm_name_root = strip_location_from_crm_name({**dict(crm_row), "crm_name": root_candidate})

    # Generic activity context is handled continuously by Pilier 3 NAF vector projection.
    # Keyword-based school detection deprecated in favor of NAF 85.xx continuous embeddings.
    is_crm_school = 0.0

    # D6: alias extraction (normalized, filtered)
    crm_raw = crm_row.get("crm_name", "")
    aliases = [normalize_text(m) for m in re.findall(r"\(([^)]+)\)", str(crm_raw))]
    blacklist = {"SIEGE", "BAT", "BATIMENT", "ETAGE", "LOCAL", "AGENCE", "RDC"}
    aliases = [a for a in aliases if len(a) >= 3 and a not in blacklist and not a.isdigit()]

    # D11: CRM token set (normalized + stopword filtered)
    crm_tokens = {tok for tok in crm_name.split() if len(tok) >= 3 and tok not in NAME_STOPWORDS}
    crm_tokens -= {"GROUPE", "SOCIETE", "HOLDING"}

    return {
        "crm_name": crm_name,
        "crm_name_root": crm_name_root,
        "crm_name_semantic": strip_location_from_crm_name(crm_row, preserve_case=True),
        "crm_addr": normalize_text(crm_address_norm),
        "crm_city": crm_city,
        "crm_city_norm": normalize_text(crm_city),
        "crm_street_num": extract_street_number(crm_address_raw),
        "crm_street_name": extract_street_name_from_address(crm_address_raw),
        "postcode": crm_row.get("postcode") or crm_row.get("crm_cp"),
        "insee": crm_row.get("insee") or crm_row.get("crm_insee"),
        "crm_insee": crm_row.get("crm_insee") or crm_row.get("insee"),
        "crm_cp": crm_row.get("crm_cp") or crm_row.get("postcode"),
        "crm_address": crm_address_raw,
        "aliases": aliases,
        "crm_tokens": crm_tokens,
        "is_crm_school": is_crm_school,
    }


def build_semantic_name_pool(
    candidate_names: List[CandidateName],
    *,
    crm_city_norm: str,
    cand_city_norm: str,
) -> List[str]:
    """Build the semantic candidate-name pool with the same rules as make_features()."""
    if not candidate_names:
        return []

    pool: List[str] = []
    for nm in candidate_names:
        city_overlap = max(
            token_overlap(nm.text, cand_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
            token_overlap(nm.text, crm_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
        )
        is_city_like = city_overlap > CITY_OVERLAP_THRESHOLD and len(nm.text.split()) <= CITY_MAX_TOKENS
        is_person_nm = nm.source == NameSource.PERSON_NAME or looks_like_person_name(nm.text)
        if not is_city_like and not is_person_nm:
            pool.append(nm.text)

    if pool:
        return pool
    return [nm.text for nm in candidate_names]


def make_features_from_preprocessed(
    crm: Mapping[str, Any],
    cand: dict,
    skip_semantic: bool = False,
) -> Dict[str, float]:
    """
    Compute all similarity features between a CRM row and a SIRENE candidate.

    This function is used by both training and inference to ensure consistency.

    Args:
        crm: Preprocessed CRM dict (see preprocess_crm_row)
        cand: Dictionary with SIRENE candidate data
        skip_semantic: If True, skip expensive semantic features (for stage-1 ranking)

    Returns:
        Dictionary with all feature values
    """
    # Normalize inputs
    crm_name = crm.get("crm_name", "")
    crm_addr = crm.get("crm_addr", "")
    cand_addr = build_address(cand)

    # Bag of names for the candidate
    candidate_names: List[CandidateName] = build_candidate_names(cand)

    # Street components
    crm_street_num = crm.get("crm_street_num")
    cand_street_num = cand.get("numeroVoie") or cand.get("num_voie")
    if cand_street_num:
        cand_street_num = str(cand_street_num)

    crm_street_name = crm.get("crm_street_name", "")
    cand_street_name = get_street_name(cand)

    crm_city = crm.get("crm_city", "")
    crm_city_norm = crm.get("crm_city_norm", "")
    cand_city_norm = _candidate_city_norm(cand)
    is_crm_school = float(crm.get("is_crm_school", 0.0))

    # ---------------- Name similarities (aggregated) -----------------
    name_features = _init_name_feature_defaults()

    if candidate_names:
        sims = []
        cand_texts = [nm.text for nm in candidate_names]
        if crm_name and cand_texts and all(cand_texts):
            jaro_scores = rf_process.cdist(
                [crm_name],
                cand_texts,
                scorer=JaroWinkler.similarity,
                processor=None,
            )
            lev_dists = rf_process.cdist(
                [crm_name],
                cand_texts,
                scorer=Levenshtein.distance,
                processor=None,
            )
            jaro_scores = np.asarray(jaro_scores)[0]
            lev_dists = np.asarray(lev_dists)[0]
            max_lens = np.array([max(len(crm_name), len(t)) for t in cand_texts], dtype=float)
            lev_scores = 1.0 - (lev_dists / max_lens)
        else:
            jaro_scores = [jaro_sim(crm_name, nm.text) for nm in candidate_names]
            lev_scores = [levenshtein_norm(crm_name, nm.text) for nm in candidate_names]

        crm_name_root = crm.get("crm_name_root")
        has_root = bool(crm_name_root and crm_name_root != crm_name)

        for idx, nm in enumerate(candidate_names):
            sim_jaro = float(jaro_scores[idx])
            sim_lev = float(lev_scores[idx])
            tok = token_overlap(crm_name, nm.text, stopwords=NAME_STOPWORDS, min_len=2)
            if has_root:
                sim_jaro = max(sim_jaro, float(jaro_sim(crm_name_root, nm.text)))
                sim_lev = max(sim_lev, float(levenshtein_norm(crm_name_root, nm.text)))
                tok = max(tok, float(token_overlap(crm_name_root, nm.text, stopwords=NAME_STOPWORDS, min_len=2)))
            fw = first_word_match(crm_name, nm.text)
            contains_crm = contains_check(nm.text, crm_name)
            crm_contains = contains_check(crm_name, nm.text)
            acr = acronym_match(crm_name, nm.text)
            city_overlap = max(
                token_overlap(nm.text, cand_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
                token_overlap(nm.text, crm_city_norm, stopwords=NAME_STOPWORDS, min_len=2),
            )
            is_city_like = city_overlap > CITY_OVERLAP_THRESHOLD and len(nm.text.split()) <= CITY_MAX_TOKENS
            is_person_nm = nm.source == NameSource.PERSON_NAME or looks_like_person_name(nm.text)

            sims.append(
                {
                    "nm": nm,
                    "jaro": sim_jaro,
                    "lev": sim_lev,
                    "tok": tok,
                "fw": float(fw),
                "contains_crm": float(contains_crm),
                "crm_contains": float(crm_contains),
                "acronym": float(acr),
                "city_overlap": city_overlap,
                "is_city_like": is_city_like,
                "is_person_nm": is_person_nm,
            }
            )

        # Global maxima
        name_features["has_any_name"] = 1.0
        name_features["name_count"] = float(len(candidate_names))

        sims_sorted = sorted(sims, key=lambda x: x["jaro"], reverse=True)

        # Option hard: prefer non city-like names for the maxima when available
        non_city_sims = [s for s in sims_sorted if not s["is_city_like"]]
        sims_for_max = non_city_sims or sims_sorted

        top = sims_for_max[0]
        second = sims_for_max[1]["jaro"] if len(sims_for_max) > 1 else 0.0

        idf_name = _idf_overlap(crm_name, top["nm"].text)
        numeric_match = numeric_token_match(crm_name, top["nm"].text)

        name_features.update(
            {
                "name_jaro_max": top["jaro"],
                "name_jaro_second": second,
                "name_jaro_gap": top["jaro"] - second,
                "name_levenshtein_max": top["lev"],
                "name_token_overlap_max": max(s["tok"] for s in sims),
                "idf_name": idf_name,
                "numeric_token_match": numeric_match,
                "name_first_word_match_max": max(s["fw"] for s in sims),
                "name_contains_crm_max": max(s["contains_crm"] for s in sims),
                "name_crm_contains_cand_max": max(s["crm_contains"] for s in sims),
                "acronym_match_max": max(s["acronym"] for s in sims),
                "type_of_max_name": float(_encode_source(top["nm"].source)),
                "is_ul_name_max": float(top["nm"].is_ul_name),
                "is_sigle_max": float(top["nm"].is_sigle),
                "name_length_max": float(len(top["nm"].text)),
                "has_person_name": float(any(s["is_person_nm"] for s in sims)),
                "person_name_jaro_max": max([s["jaro"] for s in sims if s["is_person_nm"]] or [0.0]),
                "name_city_overlap_max": max(s["city_overlap"] for s in sims),
                "name_is_city_like_max": float(max(s["is_city_like"] for s in sims)),
            }
        )

        # Source-specific maxima (based on Jaro)
        name_features["name_sim_max_etab"] = max(
            [s["jaro"] for s in sims if not s["nm"].is_ul_name] or [0.0]
        )
        name_features["name_sim_max_ul"] = max(
            [s["jaro"] for s in sims if s["nm"].is_ul_name] or [0.0]
        )
        name_features["name_sim_max_sigle"] = max(
            [s["jaro"] for s in sims if s["nm"].is_sigle] or [0.0]
        )
        name_features["name_sim_max_pm_dirigeant"] = max(
            [s["jaro"] for s in sims if s["nm"].source == NameSource.PM_DIRIGEANT] or [0.0]
        )

        semantic_pool = [
            s["nm"].text
            for s in sims
            if not s["is_city_like"] and not s["is_person_nm"]
        ]
        if not semantic_pool:
            semantic_pool = [s["nm"].text for s in sims]
        
        # Use original case version for semantic embedding (CamelCase split needs it)
        crm_name_semantic = crm.get("crm_name_semantic", "")
        
        # Calculate semantic scores using batch GPU encoding (skip if requested or gated)
        semantic_allowed = (not skip_semantic) and semantic_gate_allows(
            name_features.get("name_jaro_max", 0.0),
            name_features.get("name_token_overlap_max", 0.0),
        )
        if semantic_allowed:
            from .semantic import top2_semantic_similarities
            sem_max, sem_second, sem_gap = top2_semantic_similarities(crm_name_semantic, semantic_pool)
            name_features["name_semantic_max"] = sem_max
            name_features["name_semantic_second"] = sem_second
            name_features["name_semantic_gap"] = sem_gap

    # ---------------- Address & location features -----------------
    addr_features = {
        "addr_jaro": jaro_sim(crm_addr, cand_addr),
        "addr_levenshtein": levenshtein_norm(crm_addr, cand_addr),
        "postcode_match": float(postal_match(crm.get("postcode"), cand.get("postcode"))),
        "city_match": float(city_match(crm_city, cand.get("city"))),
        "street_number_diff": street_number_diff(crm_street_num, cand_street_num),
        "addr_token_overlap": token_overlap(crm_addr, cand_addr),
        "address_density": _address_density_for_candidate(crm, cand),
        "street_name_jaro": jaro_sim(crm_street_name, cand_street_name),
    }

    features = {**name_features, **addr_features}
    features["name_addr_consistency"] = name_features["name_jaro_max"] * addr_features["addr_jaro"]
    legal_category = map_legal_to_category(cand.get("cj_ul"))
    if legal_category == "PUBLIC":
        features["legal_form_category"] = 1.0
    elif legal_category == "PRIVE":
        features["legal_form_category"] = -1.0
    else:
        features["legal_form_category"] = 0.0

    # ---------------- New features from reranking rules (Phase 3) -----------------
    
    # Get primary candidate name for new features
    from .naming import primary_name
    primary_cand_name = primary_name(cand) or ""
    
    # D9: is_siege - head office indicator
    features["is_siege"] = float(cand.get("is_siege", False))
    
    # is_active: factual activity indicator (1.0 if active 'A', 0.0 otherwise)
    etat_val = str(cand.get("etat_admin") or cand.get("etatAdministratifEtablissement") or "").strip().upper()
    features["is_active"] = float(etat_val == "A")
    
    # B2: is_copropriete - syndicate of co-owners indicator (cj_ul == '9110')
    # Continuous feature fed to gradient boosting trees, coexisting with macro-category
    # legal_form_category to allow continuous non-destructive weighting without arbitrary hard exclusions.
    features["is_copropriete"] = float(str(cand.get("cj_ul") or "").strip() == "9110")
    
    # B1: is_association - official INSEE legal category for associations (92xx)
    # Generic, data-driven approach replacing fragile keyword lists
    cand_cj = str(cand.get("cj_ul") or "").strip()
    features["is_association"] = float(cand_cj.startswith(INSEE_ASSOCIATION_CJ_PREFIX))
    
    # D6: alias_match - check if aliases in parentheses match candidate name
    aliases = crm.get("aliases") or []
    alias_match_score = 0.0
    if aliases and primary_cand_name:
        cand_norm = normalize_text(primary_cand_name)
        for alias in aliases:
            if jaro_sim(alias, cand_norm) >= 0.80 or alias in cand_norm:
                alias_match_score = 1.0
                break
    features["alias_match"] = alias_match_score
    
    # D11: token_overlap_ul - token overlap with UL names including substring matches
    ul_tokens: Set[str] = set()
    for field in ["denomination_ul", "denomination_usuelle_ul", "sigle_ul"]:
        val = cand.get(field)
        if val:
            ul_tokens.update(normalize_text(str(val)).split())
    
    crm_tokens = crm.get("crm_tokens") or set()
    
    if not crm_tokens:
        features["token_overlap_ul"] = 0.0
    else:
        # Direct token match
        common_tokens = crm_tokens & ul_tokens
        
        # Substring match: check if a candidate token is IN a CRM token 
        # (for CamelCase like DIGITBOXING) or vice-versa.
        substring_matches = set()
        for cand_tok in ul_tokens:
            if len(cand_tok) >= 4:
                for crm_tok in crm_tokens:
                    if (len(crm_tok) >= 6 and cand_tok in crm_tok) or (len(crm_tok) >= 4 and crm_tok in cand_tok):
                        substring_matches.add(cand_tok)
                        break
        
        all_matches = common_tokens | substring_matches
        features["token_overlap_ul"] = len(all_matches) / len(crm_tokens)
    
    # D7: ul_vs_pm_indicator - 1 if best match is from UL, 0 if from PM, 0.5 otherwise
    type_max = features.get("type_of_max_name", 0.0)
    is_ul_max = features.get("is_ul_name_max", 0.0)
    if is_ul_max == 1.0:
        features["ul_vs_pm_indicator"] = 1.0
    elif type_max == 4.0:  # PM_DIRIGEANT
        features["ul_vs_pm_indicator"] = 0.0
    else:
        features["ul_vs_pm_indicator"] = 0.5
        
    # Context features
    features["is_crm_school"] = is_crm_school

    # -------------------------------------------------------------------------
    # V6 discriminant features
    # -------------------------------------------------------------------------
    
    # geo_exact_match: INSEE CRM == INSEE candidat (with Communes Nouvelles resolution)
    crm_insee = crm.get("insee")
    cand_insee = cand.get("codeCommuneEtablissement") or cand.get("insee")
    if crm_insee and cand_insee:
        crm_s = str(crm_insee).strip()
        cand_s = str(cand_insee).strip()
        if crm_s == cand_s:
            features["geo_exact_match"] = 1.0
        else:
            crm_res = resolve_commune_nouvelle(crm_s)
            cand_res = resolve_commune_nouvelle(cand_s)
            features["geo_exact_match"] = float(bool(crm_res and cand_res and crm_res == cand_res))
    else:
        features["geo_exact_match"] = 0.0

    # Sidecar match indicator (Option A: feature-driven, trade name / SSDA anchor)
    features["sidecar_match"] = float(bool(cand.get("_is_sidecar_match") or cand.get("is_sidecar_match")))
    
    # name_norm_exact: Exact match after normalization
    # Compare best candidate name (by jaro) to CRM name (both normalized)
    best_cand_name_text = ""
    if candidate_names:
        best_cand_name_text = max(candidate_names, key=lambda nm: jaro_sim(crm_name, nm.text)).text
    features["name_norm_exact"] = float(crm_name == best_cand_name_text and crm_name != "")
    
    # street_number_match: Street number exact match (supports multi-number intervals)
    if crm_street_num and cand_street_num:
        crm_nums = parse_street_numbers(crm_street_num)
        cand_nums = parse_street_numbers(cand_street_num)
        if crm_nums and cand_nums and (crm_nums & cand_nums):
            features["street_number_match"] = 1.0
        else:
            features["street_number_match"] = 0.0
    else:
        features["street_number_match"] = 0.0

    # -------------------------------------------------------------------------
    # V8 new interaction features for colocataires, homonymes, multi-site SIREN
    # -------------------------------------------------------------------------

    # Feature A: addr_unsupported_by_name
    # Penalizes high address similarity without name corroboration (colocataire signal)
    features["addr_unsupported_by_name"] = (
        addr_features["addr_jaro"] * (1.0 - name_features["name_jaro_max"])
    )

    # Feature B: name_density_penalty
    # Penalizes high address density + weak name match (colocataire in dense areas)
    density_norm = min(addr_features["address_density"], 100.0) / 100.0
    features["name_density_penalty"] = (
        density_norm * (1.0 - name_features["name_jaro_max"])
    )

    # Feature C: addr_jaro_per_density
    # Normalizes address quality by the density of the location
    import math
    density = max(1.0, addr_features["address_density"])
    features["addr_jaro_per_density"] = addr_features["addr_jaro"] / math.log1p(density)

    # Feature D: postcode_match_without_addr
    # Detects geographic homonymes: correct postcode but wrong street address
    features["postcode_match_without_addr"] = (
        addr_features["postcode_match"] * (1.0 - addr_features["street_name_jaro"])
    )

    # Feature E: full_addr_match_score
    # Weighted address completeness (street > street_number > postcode)
    features["full_addr_match_score"] = (
        0.5 * addr_features["street_name_jaro"]
        + 0.3 * features["street_number_match"]
        + 0.2 * addr_features["postcode_match"]
    )

    # Feature F: name_jaro_vs_enseigne
    # Jaro similarity against establishment-level name (enseigne), not UL name
    # This helps disambiguate multi-site SIRENs where siblings share the same UL denomination
    enseigne1 = cand.get("enseigne1") or cand.get("denomination") or ""
    enseigne1_norm = normalize_name(enseigne1) if enseigne1 else ""
    features["name_jaro_vs_enseigne"] = jaro_sim(crm_name, enseigne1_norm) if enseigne1_norm else 0.0

    # Feature G: name_city_suffix_match
    # Ratio of CRM city tokens appearing in candidate establishment name (enseigne)
    # Helps with multi-site SIREN: ACTEMIUM CAEN vs ACTEMIUM METZ
    crm_city_tokens = set(crm_city_norm.split()) - NAME_STOPWORDS if crm_city_norm else set()
    enseigne_tokens = set(normalize_text(enseigne1).split()) if enseigne1 else set()

    if crm_city_tokens and enseigne_tokens:
        city_in_enseigne = len(crm_city_tokens & enseigne_tokens) / len(crm_city_tokens)
    else:
        city_in_enseigne = 0.0
    features["name_city_suffix_match"] = city_in_enseigne

    # BAN spatial features default (for isolated/single calls)
    features["spatial_distance_km"] = 99.0
    features["spatial_is_very_close"] = 0.0
    features["spatial_is_same_block"] = 0.0
    features["spatial_has_coords"] = 0.0

    # -------------------------------------------------------------------------
    # V9 generic mathematical & categorical features (NDCG@50 alignment)
    # -------------------------------------------------------------------------
    # 1. exact_address_anchor: strict physical concordance (exact street number + street name)
    features["exact_address_anchor"] = float(
        features.get("street_number_match", 0.0) == 1.0 and addr_features.get("street_name_jaro", 0.0) >= 0.85
    )

    # 2. is_commune_authority: structural INSEE legal category (72xx = Communes, 73xx = EPCI) in same commune
    cand_cj = str(cand.get("cj_ul") or "").strip()
    is_authority_cj = cand_cj.startswith("72") or cand_cj.startswith("73")
    features["is_commune_authority"] = float(is_authority_cj and features.get("geo_exact_match", 0.0) == 1.0)

    # 3. acronym_mutual_token_match: generic mathematical initialism detection
    crm_name_raw = normalize_text(str(crm.get("crm_name") or ""))
    crm_words = [w for w in crm_name_raw.split() if len(w) >= 1 and not w.isdigit()]
    cand_name_fields = [
        cand.get("denomination"),
        cand.get("denomination_ul"),
        cand.get("enseigne1"),
        cand.get("nom_ul"),
    ]
    cand_words_raw = " ".join(str(f) for f in cand_name_fields if f)
    cand_words = [w for w in normalize_text(cand_words_raw).split() if len(w) >= 1 and not w.isdigit()]

    acronym_mutual_score = 0.0
    cand_sigle = str(cand.get("sigle") or cand.get("sigle_ul") or "").strip().upper()
    if cand_sigle and (cand_sigle in crm_words or cand_sigle in (crm.get("crm_tokens") or set())):
        acronym_mutual_score = 1.0
    elif crm_words and cand_words:
        crm_acronym = "".join(w[0] for w in crm_words)
        cand_acronym = "".join(w[0] for w in cand_words)
        cand_set = set(cand_words)
        crm_set = set(crm_words)
        if len(crm_acronym) >= 2 and (crm_acronym in cand_set or crm_acronym == cand_sigle):
            acronym_mutual_score = 1.0
        elif len(cand_acronym) >= 2 and (cand_acronym in crm_set):
            acronym_mutual_score = 1.0
    features["acronym_mutual_token_match"] = acronym_mutual_score

    # -------------------------------------------------------------------------
    # Stage 2 Discriminant Features (Hit@1 Maximization)
    # -------------------------------------------------------------------------
    features["civil_name_jaro"] = compute_civil_name_jaro(crm.get("crm_name_norm", ""), cand)
    features["naf_vector_similarity"] = compute_naf_vector_similarity(crm.get("crm_name", ""), cand)
    features["delta_addr_vs_siege"] = 0.0  # Populated in batch context

    return features


def compute_civil_name_jaro(crm_name_norm: str, cand: Dict[str, Any]) -> float:
    """
    Compute continuous Jaro similarity testing both permutations for natural persons:
    - 'PRENOM NOM' vs CRM
    - 'NOM PRENOM' vs CRM
    - 'NOM' vs CRM
    Also strips common civil titles (DR, DOCTEUR, ME, CABINET) to catch medical/legal practitioners.
    Returns 0.0 if neither prenom nor nom is present.
    """
    crm_clean = normalize_text(crm_name_norm)
    if not crm_clean:
        return 0.0
    prenom = str(cand.get("prenom_usuel_ul") or cand.get("prenom1") or "").strip()
    nom = str(cand.get("nom_ul") or cand.get("nom") or "").strip()
    if not nom and not prenom:
        return 0.0

    cand_p_n = normalize_text(f"{prenom} {nom}") if prenom and nom else ""
    cand_n_p = normalize_text(f"{nom} {prenom}") if prenom and nom else ""
    cand_nom = normalize_text(nom) if nom else ""

    sim_p_n = jaro_sim(crm_clean, cand_p_n) if cand_p_n else 0.0
    sim_n_p = jaro_sim(crm_clean, cand_n_p) if cand_n_p else 0.0
    sim_nom = jaro_sim(crm_clean, cand_nom) if cand_nom else 0.0

    best_sim = max(sim_p_n, sim_n_p, sim_nom)

    # Check without common civil/medical/legal titles
    crm_words = crm_clean.split()
    if len(crm_words) >= 2 and crm_words[0] in {"DR", "DOCTEUR", "ME", "MAITRE", "CABINET", "MR", "MME", "MONSIEUR", "MADAME"}:
        crm_stripped = " ".join(crm_words[1:])
        if cand_p_n:
            best_sim = max(best_sim, jaro_sim(crm_stripped, cand_p_n))
        if cand_n_p:
            best_sim = max(best_sim, jaro_sim(crm_stripped, cand_n_p))
        if cand_nom:
            best_sim = max(best_sim, jaro_sim(crm_stripped, cand_nom))

    return float(best_sim)


def compute_naf_vector_similarity(crm_name: str, cand: Dict[str, Any]) -> float:
    """
    Compute continuous cosine similarity between CRM query text and candidate's NAF code.
    """
    if not crm_name:
        return 0.0
    raw_naf = (
        cand.get("activitePrincipaleEtablissement")
        or cand.get("activitePrincipaleUniteLegale")
        or cand.get("apet")
        or cand.get("apen")
        or cand.get("activitePrincipale")
        or ""
    )
    if not raw_naf:
        return 0.0
    try:
        from .naf_vector_engine import NafVectorEngine
        return float(NafVectorEngine.get_instance().get_naf_similarity(crm_name, str(raw_naf)))
    except Exception:
        return 0.0


def compute_delta_addr_vs_siege_batch(
    candidates: List[Dict[str, Any]],
    feats_list: List[Dict[str, float]],
) -> List[float]:
    """
    Compute delta_addr_vs_siege for each candidate in the batch.
    For secondary establishments:
        addr_score(cand) - addr_score(siege_of_same_siren)
    For head offices (siege):
        addr_score(siege) - max(addr_score(secondary) for secondary of same siren)
    If no comparison is possible (single establishment for siren): 0.0.
    Bounded in [-1.0, 1.0].
    """
    n = len(candidates)
    if n == 0:
        return []

    addr_scores: List[float] = []
    sirens: List[str] = []
    is_sieges: List[bool] = []

    for i in range(n):
        f = feats_list[i]
        c = candidates[i]
        score = f.get("street_name_jaro", 0.0) + 0.3 * f.get("street_number_match", 0.0)
        addr_scores.append(score)

        siret = str(c.get("siret") or "").strip().replace(" ", "").zfill(14)
        siren = siret[:9] if len(siret) >= 9 else ""
        sirens.append(siren)

        is_siege = bool(f.get("is_siege", 0.0) >= 0.5 or c.get("is_siege", False))
        is_sieges.append(is_siege)

    siren_to_indices: Dict[str, List[int]] = {}
    for i, siren in enumerate(sirens):
        if siren:
            siren_to_indices.setdefault(siren, []).append(i)

    deltas = [0.0] * n
    for siren, idxs in siren_to_indices.items():
        if len(idxs) <= 1:
            continue

        siege_idx = next((i for i in idxs if is_sieges[i]), None)
        secondary_idxs = [i for i in idxs if not is_sieges[i]]

        if siege_idx is not None and secondary_idxs:
            siege_score = addr_scores[siege_idx]
            max_sec_score = max(addr_scores[i] for i in secondary_idxs)

            for i in secondary_idxs:
                diff = addr_scores[i] - siege_score
                deltas[i] = max(-1.0, min(1.0, float(diff)))

            diff_siege = siege_score - max_sec_score
            deltas[siege_idx] = max(-1.0, min(1.0, float(diff_siege)))

    return deltas


_SPATIAL_RESOLVER = None


def get_spatial_resolver():
    """Module-level singleton for BAN SpatialResolver."""
    global _SPATIAL_RESOLVER
    if _SPATIAL_RESOLVER is None:
        try:
            from .spatial import SpatialResolver
            res = SpatialResolver()
            if res.is_available:
                _SPATIAL_RESOLVER = res
        except Exception:
            _SPATIAL_RESOLVER = None
    return _SPATIAL_RESOLVER


def make_features(crm_row: pd.Series, cand: dict, skip_semantic: bool = False) -> Dict[str, float]:
    """Backward-compatible wrapper (builds CRM preprocessing on the fly)."""
    crm = preprocess_crm_row(crm_row)
    return make_features_from_preprocessed(crm, cand, skip_semantic=skip_semantic)


# ---------------------------------------------------------------------------
# Axe 2: Matrix Vector Blocks & Batch Representation
# ---------------------------------------------------------------------------

TEXT_FEATURE_NAMES: List[str] = [
    "has_any_name", "name_count", "name_jaro_max", "name_jaro_second", "name_jaro_gap",
    "name_levenshtein_max", "name_token_overlap_max", "idf_name", "numeric_token_match",
    "name_first_word_match_max", "name_contains_crm_max", "name_crm_contains_cand_max",
    "acronym_match_max", "name_sim_max_etab", "name_sim_max_ul", "name_sim_max_sigle",
    "name_sim_max_pm_dirigeant", "type_of_max_name", "is_ul_name_max", "is_sigle_max",
    "name_length_max", "has_person_name", "person_name_jaro_max", "name_city_overlap_max",
    "name_is_city_like_max", "name_norm_exact", "alias_match", "token_overlap_ul",
    "ul_vs_pm_indicator", "name_jaro_vs_enseigne", "name_city_suffix_match",
    "acronym_mutual_token_match", "civil_name_jaro",
]

GEO_FEATURE_NAMES: List[str] = [
    "addr_jaro", "addr_levenshtein", "postcode_match", "city_match",
    "street_number_diff", "addr_token_overlap", "address_density",
    "street_name_jaro", "geo_exact_match", "street_number_match",
]

SPATIAL_FEATURE_NAMES: List[str] = [
    "spatial_distance_km",
    "spatial_is_very_close",
    "spatial_is_same_block",
    "spatial_has_coords",
]

ADMIN_FEATURE_NAMES: List[str] = [
    "legal_form_category", "is_siege", "is_association", "is_copropriete", "is_crm_school", "sidecar_match", "is_active",
    "is_commune_authority", "naf_vector_similarity",
]

INTERACTION_FEATURE_NAMES: List[str] = [
    "name_addr_consistency", "addr_unsupported_by_name", "name_density_penalty",
    "addr_jaro_per_density", "postcode_match_without_addr", "full_addr_match_score",
    "exact_address_anchor", "delta_addr_vs_siege",
]

_IDX_TEXT = [FEATURE_NAMES.index(fn) for fn in TEXT_FEATURE_NAMES]
_IDX_GEO = [FEATURE_NAMES.index(fn) for fn in GEO_FEATURE_NAMES]
_IDX_SPATIAL = [FEATURE_NAMES.index(fn) for fn in SPATIAL_FEATURE_NAMES]
_IDX_ADMIN = [FEATURE_NAMES.index(fn) for fn in ADMIN_FEATURE_NAMES]
_IDX_INTERACT = [FEATURE_NAMES.index(fn) for fn in INTERACTION_FEATURE_NAMES]


def make_features_batch(
    crm_pre: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    skip_semantic: bool = False,
    spatial_resolver: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute features for a batch of candidates for a single CRM query.
    Returns:
        features_matrix: np.ndarray of shape (N, len(FEATURE_NAMES))
        vector_blocks: dict of sub-matrices
    """
    if not candidates:
        empty_mat = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        return empty_mat, {
            "V_text": empty_mat[:, _IDX_TEXT],
            "V_geo": empty_mat[:, _IDX_GEO],
            "V_spatial": empty_mat[:, _IDX_SPATIAL],
            "V_admin": empty_mat[:, _IDX_ADMIN],
            "V_interact": empty_mat[:, _IDX_INTERACT],
        }

    feats_list = [
        make_features_from_preprocessed(crm_pre, cand, skip_semantic=skip_semantic)
        for cand in candidates
    ]

    # 1. Batch delta_addr_vs_siege
    deltas = compute_delta_addr_vs_siege_batch(candidates, feats_list)
    for f, d in zip(feats_list, deltas):
        f["delta_addr_vs_siege"] = d

    # 2. Optimized batch NAF similarity
    crm_name = str(crm_pre.get("crm_name") or "").strip()
    if crm_name:
        try:
            from .naf_vector_engine import NafVectorEngine
            raw_nafs = [
                str(
                    c.get("activitePrincipaleEtablissement")
                    or c.get("activitePrincipaleUniteLegale")
                    or c.get("apet")
                    or c.get("apen")
                    or c.get("activitePrincipale")
                    or ""
                )
                for c in candidates
            ]
            naf_sims = NafVectorEngine.get_instance().get_batch_naf_similarities(crm_name, raw_nafs)
            for f, n_sim in zip(feats_list, naf_sims):
                f["naf_vector_similarity"] = n_sim
        except Exception:
            pass

    matrix = np.array(
        [[f[fn] for fn in FEATURE_NAMES] for f in feats_list],
        dtype=np.float32,
    )

    resolver = spatial_resolver if spatial_resolver is not None else get_spatial_resolver()
    if resolver and resolver.is_available:
        try:
            sp_mat = resolver.compute_spatial_features(crm_pre, candidates)
            matrix[:, _IDX_SPATIAL] = sp_mat
        except Exception as e:
            logger.debug("Failed computing batch spatial features: %s", e)

    blocks = {
        "V_text": matrix[:, _IDX_TEXT],
        "V_geo": matrix[:, _IDX_GEO],
        "V_spatial": matrix[:, _IDX_SPATIAL],
        "V_admin": matrix[:, _IDX_ADMIN],
        "V_interact": matrix[:, _IDX_INTERACT],
    }
    return matrix, blocks

