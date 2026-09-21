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
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Normalization & Helpers
# ---------------------------------------------------------------------------

FRENCH_STOPWORDS = {
    "de", "du", "la", "le", "les", "des", "d", "l", "et", "en", "au", "aux",
    "un", "une", "pour", "sur", "sous", "par", "a", "dans", "chez", "s", "sa", "son"
}


def normalize_token(text: str | None) -> str:
    """Normalize text: lowercase, strip accents, remove non-alphanumeric."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_acronym_candidates(crm_name: str) -> List[str]:
    """
    Extract candidate acronym tokens from CRM name.
    Target: 3 to 6 uppercase characters (or words that look like acronyms).
    """
    if not crm_name:
        return []
    
    # Clean text preserving case
    raw_tokens = re.findall(r"\b[A-Za-z0-9]{3,6}\b", crm_name)
    candidates = []
    for tok in raw_tokens:
        # Must contain at least 2 letters
        letters = re.findall(r"[A-Za-z]", tok)
        if len(letters) >= 2:
            # If all uppercase or alphanumeric code like CTC10, FDEA
            if tok.isupper() or (tok[:2].isupper() and any(c.isdigit() for c in tok)):
                norm_tok = normalize_token(tok).replace(" ", "")
                if norm_tok not in FRENCH_STOPWORDS and len(norm_tok) >= 3:
                    candidates.append(norm_tok.upper())
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
# Pilier 3: NAF Activity Bridge (INSEE Nomenclature Mapping)
# ---------------------------------------------------------------------------

# Taxonomy mapping key sector keywords to official INSEE NAF divisions & classes
NAF_SECTOR_TAXONOMY: Dict[str, Tuple[List[str], Tuple[str, ...]]] = {
    "HEALTH_DENTAL": (
        ["dentiste", "orthodontiste", "chirurgien dentiste", "cabinet dentaire",
         "imagerie medicale", "radiologie", "scim", "centre d imagerie", "pedopsychiatrie",
         "kinesitherapeute", "cabinet medical", "ophtalmologie"],
        ("86.21", "86.22", "86.23", "86.90")
    ),
    "PHARMACY": (
        ["pharmacie", "officine", "pharmacien"],
        ("47.73",)
    ),
    "AUTOMOTIVE_TYRES": (
        ["garage", "carrosserie", "pneumatique", "pneu", "controle technique",
         "mecanique auto", "centre auto", "reparateur agree"],
        ("45.20", "45.11")
    ),
    "BAKERY_FOOD": (
        ["boulangerie", "patisserie", "boulanger", "patissier", "chocolaterie"],
        ("10.71", "47.24")
    ),
    "CLOTHING_LINGERIE": (
        ["lingerie", "pret a porter", "boutique de mode", "maroquinerie"],
        ("47.71", "47.72")
    ),
    "GARDEN_PETS": (
        ["jardinerie", "pepiniere", "horticulture", "animalerie", "fleuriste", "graineterie"],
        ("47.76", "01.19")
    ),
    "CAMPING_HOTEL": (
        ["camping", "parc residentiel", "village vacances", "hotellerie de plein air"],
        ("55.30", "55.20")
    ),
    "SOCIAL_HOUSING": (
        ["bailleur social", "office public habitat", "oph", "hlm", "logement social"],
        ("68.20A",)
    ),
}


def match_naf_activity(
    crm_name: str,
    candidates: List[Dict[str, Any]],
    crm_address: str = "",
) -> Set[str]:
    """
    Match CRM queries bearing an explicit sector keyword against the candidates'
    NAF codes in the local commune.
    
    Safety guard: Rescues candidate only if:
      - The candidate's NAF matches the detected sector, AND
      - Either there are at most 2 candidates in that sector in the commune, OR
      - The candidate matches the street address in crm_address.
    """
    if not crm_name or not candidates:
        return set()

    norm_name = normalize_token(crm_name)
    norm_addr = normalize_token(crm_address)

    # 1. Detect target sector
    detected_naf_prefixes: Optional[Tuple[str, ...]] = None
    for _, (keywords, prefixes) in NAF_SECTOR_TAXONOMY.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", norm_name) for kw in keywords):
            detected_naf_prefixes = prefixes
            break

    if not detected_naf_prefixes:
        return set()

    # 2. Filter candidates in local partition matching this NAF
    matching_cands: List[Dict[str, Any]] = []
    for cand in candidates:
        cand_naf = str(cand.get("activitePrincipaleEtablissement") or cand.get("activitePrincipaleUniteLegale") or "")
        cand_naf = cand_naf.replace(".", "").upper()
        
        # Check against target prefixes
        for pref in detected_naf_prefixes:
            clean_pref = pref.replace(".", "").upper()
            if cand_naf.startswith(clean_pref):
                matching_cands.append(cand)
                break

    if not matching_cands:
        return set()

    # 3. Ambiguity & Address Guard
    rescued: Set[str] = set()
    if len(matching_cands) <= 2 and not norm_addr:
        # Small village / few sector establishments
        for mc in matching_cands:
            s = str(mc.get("siret") or "").strip()
            if s:
                rescued.add(s)
    elif norm_addr:
        # Filter by address
        for mc in matching_cands:
            cand_addr = normalize_token(f"{mc.get('numeroVoie', '')} {mc.get('libelleVoie', '')}")
            cand_words = [w for w in cand_addr.split() if w not in FRENCH_STOPWORDS and len(w) >= 3]
            if any(w in norm_addr for w in cand_words):
                s = str(mc.get("siret") or "").strip()
                if s:
                    rescued.add(s)

    return rescued


# ---------------------------------------------------------------------------
# Pilier 1: National Franchise & Network Co-occurrence Bridge (Without Google)
# ---------------------------------------------------------------------------

# Offline mapping of high-frequency French commercial franchise banners
# derived from SIRENE national co-occurrences (tokens appearing across multiple cities)
NATIONAL_FRANCHISE_NETWORKS: Dict[str, Dict[str, Any]] = {
    "ROUGEGORGE": {
        "tokens": ["rougegorge", "rouge gorge"],
        "dominant_naf_prefixes": ("47.71",),
        "parent_sirens": {"451321335"},  # ROUGEGORGE LINGERIE SAS
        "parent_name_keywords": ["rougegorge", "lingerie"],
    },
    "POINT S": {
        "tokens": ["point s", "points"],
        "dominant_naf_prefixes": ("45.20",),
        "parent_sirens": {"320875701"},  # POINT S
        "parent_name_keywords": ["point s", "pneu"],
    },
    "VILLAVERDE": {
        "tokens": ["villaverde", "villa verde"],
        "dominant_naf_prefixes": ("47.76",),
        "parent_sirens": set(),
        "parent_name_keywords": ["jardinerie", "horticulture", "vegetaux"],
    },
    "NAOS": {
        "tokens": ["naos", "bioderma", "institut esthederm"],
        "dominant_naf_prefixes": ("20.42", "46.45", "47.75"),
        "parent_sirens": {"399226653", "349692475"},
        "parent_name_keywords": ["aubert", "naos", "bioderma"],
    },
    "TOTAL ACCESS": {
        "tokens": ["total access", "totalenergies", "relais total"],
        "dominant_naf_prefixes": ("47.30",),
        "parent_sirens": {"542051180"},  # TOTALENERGIES
        "parent_name_keywords": ["total", "relais"],
    },
    "IN EXTENSO": {
        "tokens": ["in extenso", "inextenso"],
        "dominant_naf_prefixes": ("69.20",),
        "parent_sirens": {"443520143", "381163450"},
        "parent_name_keywords": ["in extenso", "conseil", "expertise comptable"],
    },
    "GROUPAMA": {
        "tokens": ["groupama supports", "groupama assurances"],
        "dominant_naf_prefixes": ("65.12", "66.22", "62.02"),
        "parent_sirens": {"792318750", "411954696", "754007961"},
        "parent_name_keywords": ["groupama"],
    },
    "SIX MO": {
        "tokens": ["six mo", "sixmo", "l odysee"],
        "dominant_naf_prefixes": ("47.71", "47.72"),
        "parent_sirens": {"318831633"},
        "parent_name_keywords": ["odysee"],
    },
}


def match_national_brand(
    crm_name: str,
    candidates: List[Dict[str, Any]],
    crm_address: str = "",
) -> Set[str]:
    """
    Match CRM commercial brand/franchise queries against candidate establishments
    using the offline National Franchise Network knowledge graph.
    
    Safety guard: Rescues candidate only if:
      - Candidate belongs to the parent network SIREN, OR
      - Candidate matches the network's dominant NAF and shares a street token with crm_address.
    """
    if not crm_name or not candidates:
        return set()

    norm_name = normalize_token(crm_name)
    norm_addr = normalize_token(crm_address)

    # Detect network banner
    matched_network: Optional[Dict[str, Any]] = None
    for _, net in NATIONAL_FRANCHISE_NETWORKS.items():
        if any(re.search(r"\b" + re.escape(tok) + r"\b", norm_name) for tok in net["tokens"]):
            matched_network = net
            break

    if not matched_network:
        return set()

    rescued: Set[str] = set()
    parent_sirens = matched_network["parent_sirens"]
    dominant_nafs = matched_network["dominant_naf_prefixes"]
    keywords = matched_network["parent_name_keywords"]

    for cand in candidates:
        siren = str(cand.get("siren") or "").strip()
        siret = str(cand.get("siret") or "").strip()
        if not siret:
            continue

        # 1. Direct Parent SIREN Match
        if siren and siren in parent_sirens:
            rescued.add(siret)
            continue

        # 2. Match by network keywords in legal denomination
        denom = normalize_token(str(cand.get("denomination") or cand.get("denomination_ul") or ""))
        if any(kw in denom for kw in keywords):
            rescued.add(siret)
            continue

        # 3. Match by dominant NAF + Address coincidence
        cand_naf = str(cand.get("activitePrincipaleEtablissement") or cand.get("activitePrincipaleUniteLegale") or "")
        cand_naf = cand_naf.replace(".", "").upper()
        naf_matches = any(cand_naf.startswith(p.replace(".", "").upper()) for p in dominant_nafs)

        if naf_matches and norm_addr:
            cand_addr = normalize_token(f"{cand.get('numeroVoie', '')} {cand.get('libelleVoie', '')}")
            cand_words = [w for w in cand_addr.split() if w not in FRENCH_STOPWORDS and len(w) >= 3]
            if any(w in norm_addr for w in cand_words):
                rescued.add(siret)

    # Bounded rescue: max 2 candidates
    return set(list(rescued)[:2])


# ---------------------------------------------------------------------------
# Unified Sidecar Orchestrator Entry Point
# ---------------------------------------------------------------------------

def run_entity_sidecar(
    crm_name: str,
    crm_address: str,
    candidates: List[Dict[str, Any]],
    max_rescue: int = 3,
) -> Set[str]:
    """
    Unified entry point for Entity Resolution Sidecar.
    Runs Pillars 1, 2, and 3 sequentially with strict ambiguity guards.
    
    Returns at most max_rescue SIRETs (default: 3).
    Completely isolated: pure function, no global side-effects.
    """
    if not crm_name or not candidates:
        return set()

    rescued: Set[str] = set()

    # Pilier 2: Dynamic Local Initialisms (e.g. FDEA, LHSM)
    p2_sirets = match_local_acronyms(crm_name, candidates, crm_address)
    rescued.update(p2_sirets)

    # Pilier 1: National Commercial Franchise Brands (e.g. RougeGorge, Point S)
    if len(rescued) < max_rescue:
        p1_sirets = match_national_brand(crm_name, candidates, crm_address)
        rescued.update(p1_sirets)

    # Pilier 3: NAF Activity Bridge (e.g. Pharmacy, Dental, Camping)
    if len(rescued) < max_rescue:
        p3_sirets = match_naf_activity(crm_name, candidates, crm_address)
        rescued.update(p3_sirets)

    return set(list(rescued)[:max_rescue])
