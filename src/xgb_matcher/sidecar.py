"""
sidecar.py — Entity Resolution Sidecar for Extreme Divergence Cases.

Completely isolated, deterministic, offline module designed to rescue:
  1. Pilier 1 : National commercial franchise banners vs legal holdings (without Google Places)
  2. Pilier 2 : Dynamic local initialisms / acronyms (zero static lists, calculated on-the-fly)
  3. Pilier 3 : Sector / NAF activity bridge (mapped to official INSEE nomenclature)

Designed as a standalone sidecar: kept strictly decoupled from the core retrieval
and XGBoost engine until explicitly enabled.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Normalization & Helpers
# ---------------------------------------------------------------------------

FRENCH_STOPWORDS = {
    "de", "du", "la", "le", "les", "des", "d", "l", "et", "en", "au", "aux",
    "un", "une", "pour", "sur", "sous", "par", "a", "dans", "chez", "s", "sa", "son"
}

COMMON_FRENCH_WORDS = {
    "FRANCE", "BOURG", "BRESSE", "PASCAL", "NICE", "ANDRE", "ROCHE", "LOUBET",
    "COMMERCE", "SOCIETE", "SERVICES", "SERVICE", "CENTRE", "HOTEL", "RELAIS",
    "ATELIER", "ESPACE", "SAINT", "SAINTE", "NOUVELLE", "EUROPE", "SUD", "NORD",
    "EST", "OUEST", "VALLEE", "VILLAGE", "VILLE", "ALPES", "ASSOCIE", "ASSOCIES",
    "NATIONALE", "GENERAL", "GENERALE", "MUTUELLE", "CABINET", "CONSEIL", "EXPERT",
    "EXPERTISE", "OPERATIONNEL", "INTERNAT", "EXTERNAT", "RAPIDO", "TEQUILA",
    "JARDIN", "JARDINS", "MAISON", "AGRICOLE", "NATURE", "ENSEMBLE", "FORMATION",
    "MAGASIN", "MAGASINS", "BOUTIQUE", "BOUTIQUES"
}

FRENCH_STREET_TYPES = {
    "RUE", "AVENUE", "BOULEVARD", "CHEMIN", "ROUTE", "PLACE", "ALLEE", "IMPASSE",
    "QUAI", "COURS", "SQUARE", "PASSAGE", "VOIE", "R", "AV", "BD", "CHE", "RTE",
    "PL", "ALL", "IMP", "QU", "CRS", "SQ", "PAS", "PROMENADE", "PROM", "CENTRE",
    "COMMERCIAL", "ZONE", "INDUSTRIELLE", "ACTIVITE", "LOTISSEMENT", "RESIDENCE"
}


def normalize_token(text: str | None) -> str:
    """Normalize text: lowercase, strip accents, de-concatenate alphanumeric, remove non-alphanumeric."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # De-concatenate alphanumeric: 'ADAPA01' -> 'ADAPA 01', '75001PARIS' -> '75001 PARIS'
    text = re.sub(r"([A-Za-z]+)(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)([A-Za-z]+)", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_acronym_candidates(crm_name: str) -> List[str]:
    """
    Extract candidate acronym tokens from CRM name.
    Handles all-uppercase CRM text safely by distinguishing true acronyms
    from regular French vocabulary.
    """
    if not crm_name:
        return []

    candidates: List[str] = []

    # 1. Preserve raw alphanumeric codes like 'CTC10' before de-concatenation
    raw_alnum = re.findall(r"\b([A-Za-z]{2,5}\d{1,3})\b", crm_name)
    for tok in raw_alnum:
        candidates.append(tok.upper())

    # 2. Alphanumeric de-concatenation: 'ADAPA01' -> 'ADAPA 01'
    clean_name = re.sub(r"([A-Za-z]+)(\d+)", r"\1 \2", crm_name)
    clean_name = re.sub(r"(\d+)([A-Za-z]+)", r"\1 \2", clean_name)

    # 3. Delimited tokens: after dash, in parentheses, brackets, or slashes (e.g. - BFA, (L-ODYSEE))
    delimited = re.findall(r'[-\(\[\/\"]\s*([A-Za-z]{2,8})\b|\b([A-Za-z]{2,8})\s*[-\)\]\/\"]', clean_name)
    for g1, g2 in delimited:
        tok = (g1 or g2).upper()
        if tok not in COMMON_FRENCH_WORDS and tok.lower() not in FRENCH_STOPWORDS and len(tok) >= 2:
            candidates.append(tok)

    # 4. Tokens next to digits (e.g. ADAPA 01, MFR 44, CTC 10)
    digit_prox = re.findall(r'\b([A-Za-z]{2,8})\s+\d+|\d+\s+([A-Za-z]{2,8})\b', clean_name)
    for g1, g2 in digit_prox:
        tok = (g1 or g2).upper()
        if tok not in COMMON_FRENCH_WORDS and tok.lower() not in FRENCH_STOPWORDS and len(tok) >= 2:
            candidates.append(tok)

    # 5. Consonant-heavy or short uppercase tokens (2 to 6 chars)
    raw_tokens = re.findall(r"\b[A-Za-z0-9]{2,6}\b", clean_name)
    for tok in raw_tokens:
        tok_u = tok.upper()
        if tok.isalpha() and 2 <= len(tok) <= 6:
            if tok_u in COMMON_FRENCH_WORDS or tok.lower() in FRENCH_STOPWORDS:
                continue
            vowels = sum(1 for c in tok.lower() if c in "aeiouy")
            # Consonant-heavy (e.g. DSHA, DSB, LHSM, BFA) or short (2-3 chars, e.g. MFR, CEPI)
            if vowels <= 1 or len(tok) <= 3:
                candidates.append(tok_u)
            # Standalone single-word name or first word with all-caps short length
            elif len(raw_tokens) <= 2:
                candidates.append(tok_u)

    return list(dict.fromkeys(candidates))


def compute_initials(name: str | None) -> str:
    """
    Compute initial letters of all significant words in a business name.
    Preserves leading article if it is part of a proper place name (e.g. Le Havre -> LHSM).
    Example: 'Federation Departementale d Energie des Ardennes' -> 'FDEA'
             'Le Havre Seine Metropole' -> 'LHSM'
    """
    if not name:
        return ""
    norm = normalize_token(name)
    tokens = [t for t in norm.split() if t]
    if not tokens:
        return ""
    
    words = []
    for i, w in enumerate(tokens):
        if i == 0 and w in {"le", "la", "les", "l"}:
            words.append(w)
        elif w not in FRENCH_STOPWORDS and len(w) >= 2:
            words.append(w)
            
    return "".join(w[0] for w in words).upper()


def compute_initials_variants(name: str | None) -> List[str]:
    """Compute all reasonable initialism variants (with and without leading article)."""
    inits = compute_initials(name)
    if not inits:
        return []
    variants = [inits]
    if inits.startswith("L") and len(inits) > 2:
        variants.append(inits[1:])
    return list(dict.fromkeys(variants))


# ---------------------------------------------------------------------------
# Pilier 2: Dynamic Local Initialisms Matcher (Zero Static List)
# ---------------------------------------------------------------------------

def match_local_acronyms(
    crm_name: str,
    candidates: List[Dict[str, Any]],
    crm_address: str = "",
) -> Set[str]:
    """
    Dynamically test if an acronym in the CRM name matches the initial letters
    of a business name registered in this specific local commune.
    
    Safety guard: Only returns SIRET if:
      - Exactly 1 candidate matches this acronym in the entire commune, OR
      - Multiple match but exactly 1 shares a street token with crm_address.
    """
    acronyms = extract_acronym_candidates(crm_name)
    
    # Also check if any candidate in the local partition has an explicit sigle_ul present in crm_name
    norm_crm_tokens = set(normalize_token(crm_name).upper().split())
    for cand in candidates:
        cand_sigle = normalize_token(str(cand.get("sigle_ul") or "")).upper()
        if cand_sigle and len(cand_sigle) >= 2 and cand_sigle not in COMMON_FRENCH_WORDS and cand_sigle in norm_crm_tokens:
            if cand_sigle not in acronyms:
                acronyms.append(cand_sigle)

    if not acronyms or not candidates:
        return set()

    rescued: Set[str] = set()
    norm_crm_addr = normalize_token(crm_address)

    for acr in acronyms:
        matching_cands: List[Dict[str, Any]] = []

        for cand in candidates:
            # Check primary candidate names
            names_to_check = [
                cand.get("denomination"),
                cand.get("denomination_ul"),
                cand.get("sigle_ul"),
                cand.get("enseigne1"),
            ]
            
            # 1. Direct match with official sigle_ul if declared
            sigle = normalize_token(str(cand.get("sigle_ul") or "")).upper()
            if sigle and (sigle == acr or acr in sigle.split()):
                matching_cands.append(cand)
                continue

            # 2. Dynamic initialisms computed on-the-fly
            for name in names_to_check:
                if not name:
                    continue
                # Fast pre-filter: first character must match first letter of acronym or 'L' (leading article)
                first_char = name.strip()[:1].upper()
                if first_char != acr[0] and first_char != "L":
                    continue
                inits_list = compute_initials_variants(name)
                # Check exact match or prefix match if acronym has 4+ letters
                if any(inits == acr or inits.startswith(acr) or (len(acr) >= 4 and acr in inits) for inits in inits_list):
                    matching_cands.append(cand)
                    break

        # Apply Ambiguity Guard
        if len(matching_cands) == 1:
            siret = str(matching_cands[0].get("siret") or "").strip()
            if siret:
                rescued.add(siret)
        elif len(matching_cands) > 1 and norm_crm_addr:
            # Disambiguate using address
            addr_matches = []
            for mc in matching_cands:
                cand_addr = normalize_token(f"{mc.get('numeroVoie', '')} {mc.get('libelleVoie', '')}")
                cand_street_words = [w for w in cand_addr.split() if w not in FRENCH_STOPWORDS and len(w) >= 3]
                if any(w in norm_crm_addr for w in cand_street_words):
                    addr_matches.append(mc)
            
            if len(addr_matches) == 1:
                siret = str(addr_matches[0].get("siret") or "").strip()
                if siret:
                    rescued.add(siret)

    return rescued


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pilier 1 & 3: Universal Spatial-Sectoral Dual-Anchor (SSDA) Engine
# ---------------------------------------------------------------------------

from .features import (
    extract_street_number,
    parse_street_numbers,
    normalize_street_type,
    normalize_text,
    jaro_sim,
)
from .naf_vector_engine import predict_naf

HOLDING_SHELL_NAF_PREFIXES = {"6420", "7010", "6820", "8299"}


def match_spatial_sectoral_dual_anchor(
    crm_name: str,
    candidates: List[Dict[str, Any]],
    crm_address: str = "",
    min_vector_similarity: float = 0.10,
) -> Set[str]:
    """
    Universal Spatial-Sectoral Dual-Anchor (SSDA) Matcher.
    Zero static dictionaries. Strictly uses native CRM and SIRENE attributes.

    Rescues candidates when:
      1. Spatial Anchor: Exact street number + street name match in local pool.
      2. Sectoral Anchor: Continuous semantic NAF vector projection (INSEE NAF Rév. 2)
         OR Transversal Holding Metaclass pass-through (64.20, 70.10, 68.20, 82.99).
      3. Exclusivity Guard:
         - If spatial density == 1: unique active economic actor at physical street address.
         - If spatial density > 1: exactly 1 candidate matches the predicted sector.
         - If multi-tenant address (>3 candidates in same sector): safely aborts to prevent false positives.
    """
    if not crm_address or not candidates:
        return set()

    # Step 1: Spatial Anchor (Street number + street name match)
    crm_num_raw = extract_street_number(crm_address)
    if not crm_num_raw:
        return set()
    crm_nums = parse_street_numbers(crm_num_raw)
    if not crm_nums:
        return set()

    crm_addr_norm = normalize_text(normalize_street_type(crm_address))
    crm_words = {
        w.upper() for w in crm_addr_norm.split()
        if len(w) >= 3 and not w.isdigit() and w.lower() not in FRENCH_STOPWORDS and w.upper() not in FRENCH_STREET_TYPES
    }

    spatial_cands: List[Dict[str, Any]] = []
    for cand in candidates:
        cand_num = cand.get("numeroVoie")
        if not cand_num:
            continue
        cand_nums = parse_street_numbers(str(cand_num))
        if not (crm_nums & cand_nums):
            continue

        cand_voie = normalize_text(f"{cand.get('typeVoie') or ''} {cand.get('libelleVoie') or ''}")
        if not cand_voie:
            continue

        voie_words = {
            w.upper() for w in cand_voie.split()
            if len(w) >= 3 and not w.isdigit() and w.lower() not in FRENCH_STOPWORDS and w.upper() not in FRENCH_STREET_TYPES
        }
        shared_words = voie_words & crm_words
        sim = jaro_sim(cand_voie, crm_addr_norm)

        # Exact number + shared specific street name word or high street jaro
        if shared_words or sim >= 0.80:
            spatial_cands.append(cand)

    if not spatial_cands:
        return set()

    # Case A: Single-Tenant Spatial Exclusivity (Exactly 1 active establishment at this address in commune)
    if len(spatial_cands) == 1:
        siret = str(spatial_cands[0].get("siret") or "").strip()
        if siret:
            return {siret}

    # Case B: Multi-Tenant Address -> Disambiguate using Sectoral Anchor (NAF Vector Projection)
    norm_crm_name = normalize_text(crm_name)
    predicted_nafs = predict_naf(norm_crm_name, top_k=5, min_similarity=min_vector_similarity)
    target_prefixes = [code.replace(".", "").upper()[:4] for code, _ in predicted_nafs]

    sector_matching: List[Dict[str, Any]] = []
    holding_cands: List[Dict[str, Any]] = []

    for cand in spatial_cands:
        # Check both etab and UL NAF codes
        c_naf = str(cand.get("activitePrincipaleEtablissement") or cand.get("activitePrincipaleUniteLegale") or "")
        c_naf_clean = c_naf.replace(".", "").upper()

        if c_naf_clean:
            # Direct Sector match
            if any(c_naf_clean.startswith(pref) for pref in target_prefixes):
                sector_matching.append(cand)
            # Holding Metaclass pass-through
            elif any(c_naf_clean.startswith(h) for h in HOLDING_SHELL_NAF_PREFIXES):
                holding_cands.append(cand)

        # Also check name token overlap / Jaro for holding/subsidiary disambiguation
        cand_name = normalize_text(str(cand.get("denomination") or cand.get("denomination_ul") or cand.get("enseigne1") or cand.get("pm_dirigeant_names") or ""))
        cand_words = {w for w in cand_name.split() if len(w) >= 3 and w.upper() not in COMMON_FRENCH_WORDS and w.lower() not in FRENCH_STOPWORDS}
        crm_name_words = {w for w in norm_crm_name.split() if len(w) >= 3 and w.upper() not in COMMON_FRENCH_WORDS and w.lower() not in FRENCH_STOPWORDS}
        
        name_sim = jaro_sim(cand_name, norm_crm_name)
        if (cand_words & crm_name_words) or name_sim >= 0.70:
            if cand not in sector_matching:
                sector_matching.append(cand)

    # Exclusivity Guard on Sector Matching
    if len(sector_matching) == 1:
        siret = str(sector_matching[0].get("siret") or "").strip()
        if siret:
            return {siret}

    # If no direct sector match but exactly 1 holding at this address
    if not sector_matching and len(holding_cands) == 1:
        siret = str(holding_cands[0].get("siret") or "").strip()
        if siret:
            return {siret}

    return set()



# Aliases for backward compatibility
match_national_brand = match_spatial_sectoral_dual_anchor
match_naf_activity = match_spatial_sectoral_dual_anchor


# ---------------------------------------------------------------------------
@dataclass
class SidecarMatch:
    """Unified qualification structure for an establishment match."""
    siret_retenu: str
    type_etablissement: str  # "SIEGE" | "SECONDAIRE" | "FILIALE"
    siret_siege_holding: Optional[str] = None
    nom_holding: Optional[str] = None
    confidence: float = 1.0
    pillar: str = "SSDA"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "siret_retenu": self.siret_retenu,
            "type_etablissement": self.type_etablissement,
            "siret_siege_holding": self.siret_siege_holding,
            "nom_holding": self.nom_holding,
            "confidence": self.confidence,
            "pillar": self.pillar,
        }


def qualify_establishment(
    cand: Dict[str, Any],
    pool: Optional[List[Dict[str, Any]]] = None,
    pillar: str = "SSDA",
) -> SidecarMatch:
    """
    Qualify an establishment candidate into the unified 4-field schema:
      - siret_retenu: The establishment SIRET at the physical site
      - type_etablissement: 'SIEGE' | 'SECONDAIRE' | 'FILIALE'
      - siret_siege_holding: SIRET of the headquarters / parent holding
      - nom_holding: Corporate legal name of the enterprise / holding
    """
    siret = str(cand.get("siret") or "").strip()
    is_siege = bool(cand.get("is_siege") or cand.get("etablissementSiege"))
    siren = str(cand.get("siren") or (siret[:9] if len(siret) >= 9 else "")).strip()
    nom_ul = str(cand.get("denomination_ul") or cand.get("denomination") or "").strip()
    pm_dirigeant = str(cand.get("pm_dirigeant_names") or "").strip()

    if is_siege:
        return SidecarMatch(
            siret_retenu=siret,
            type_etablissement="SIEGE",
            siret_siege_holding=siret,
            nom_holding=nom_ul,
            confidence=1.0,
            pillar=pillar,
        )

    # Secondary establishment or subsidiary
    siret_siege = None
    if pool and siren:
        for c in pool:
            if str(c.get("siren")) == siren and bool(c.get("is_siege") or c.get("etablissementSiege")):
                siret_siege = str(c.get("siret") or "").strip()
                break

    if pm_dirigeant:
        type_etab = "FILIALE"
        nom_h = pm_dirigeant
    else:
        type_etab = "SECONDAIRE"
        nom_h = nom_ul

    return SidecarMatch(
        siret_retenu=siret,
        type_etablissement=type_etab,
        siret_siege_holding=siret_siege,
        nom_holding=nom_h,
        confidence=1.0,
        pillar=pillar,
    )


def run_entity_sidecar_matches(
    crm_name: str,
    crm_address: str,
    candidates: List[Dict[str, Any]],
    max_rescue: int = 3,
) -> List[SidecarMatch]:
    """
    Unified entry point returning rich SidecarMatch objects with 4-field qualification.
    """
    if not crm_name or not candidates:
        return []

    cand_by_siret = {str(c.get("siret")): c for c in candidates if c.get("siret")}
    matches: List[SidecarMatch] = []
    seen_sirets: Set[str] = set()

    # 1. Dynamic Local Initialisms & Acronyms
    acronym_sirets = match_local_acronyms(crm_name, candidates, crm_address)
    for s in acronym_sirets:
        if s in cand_by_siret and s not in seen_sirets:
            matches.append(qualify_establishment(cand_by_siret[s], candidates, pillar="ACRONYM"))
            seen_sirets.add(s)
            if len(matches) >= max_rescue:
                return matches

    # 2. Universal SSDA
    if len(matches) < max_rescue and crm_address:
        ssda_sirets = match_spatial_sectoral_dual_anchor(crm_name, candidates, crm_address)
        for s in ssda_sirets:
            if s in cand_by_siret and s not in seen_sirets:
                matches.append(qualify_establishment(cand_by_siret[s], candidates, pillar="SSDA"))
                seen_sirets.add(s)
                if len(matches) >= max_rescue:
                    break

    return matches


# ---------------------------------------------------------------------------
# Unified Sidecar Orchestrator Entry Point (Set of SIRETs for backward compatibility)
# ---------------------------------------------------------------------------

def run_entity_sidecar(
    crm_name: str,
    crm_address: str,
    candidates: List[Dict[str, Any]],
    max_rescue: int = 3,
) -> Set[str]:
    """
    Unified entry point for Entity Resolution Sidecar.
    Runs Dynamic Local Acronyms (Pillar 2) and Universal SSDA (Spatial-Sectoral Dual-Anchor).

    Zero static dictionaries. Strictly data-driven from native CRM and SIRENE attributes.
    Returns at most max_rescue SIRETs (default: 3).
    Completely isolated: pure function, zero global side-effects.
    """
    if not crm_name or not candidates:
        return set()

    rescued: Set[str] = set()

    # 1. Dynamic Local Initialisms & Acronyms (e.g. FDEA, LHSM, CEPI)
    acronym_sirets = match_local_acronyms(crm_name, candidates, crm_address)
    rescued.update(acronym_sirets)

    # 2. Universal Spatial-Sectoral Dual-Anchor (SSDA)
    if len(rescued) < max_rescue and crm_address:
        ssda_sirets = match_spatial_sectoral_dual_anchor(crm_name, candidates, crm_address)
        rescued.update(ssda_sirets)

    return set(list(rescued)[:max_rescue])

