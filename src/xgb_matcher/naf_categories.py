from typing import Optional

# Mapping of CRM name keywords to industry categories
CRM_KEYWORDS_TO_CATEGORY = {
    # Education
    "COLLEGE": "EDUCATION",
    "LYCEE": "EDUCATION",
    "ECOLE": "EDUCATION",
    "UNIVERSITE": "EDUCATION",
    "FACULTE": "EDUCATION",
    "ACADEMIE": "EDUCATION",
    "SCOLAIRE": "EDUCATION",
    
    # Healthcare / Care
    "EHPAD": "HEALTHCARE",
    "EPHAD": "HEALTHCARE",
    "HOPITAL": "HEALTHCARE",
    "CLINIQUE": "HEALTHCARE",
    "SANTE": "HEALTHCARE",
    "MEDECIN": "HEALTHCARE",
    "DISPENSAIRE": "HEALTHCARE",
    "CMP": "HEALTHCARE",
    "CH": "HEALTHCARE",
    "CHU": "HEALTHCARE",
    "CLIN": "HEALTHCARE",
    
    # Public Admin / Municipalities
    "MAIRIE": "PUBLIC_ADMIN",
    "PREFECTURE": "PUBLIC_ADMIN",
    "CONSEIL": "PUBLIC_ADMIN",
    "SDIS": "PUBLIC_ADMIN",
    "COMMUNE": "PUBLIC_ADMIN",
    "MINISTERE": "PUBLIC_ADMIN",
    "POLICE": "PUBLIC_ADMIN",
    "GENDARMERIE": "PUBLIC_ADMIN",
    "POMPIERS": "PUBLIC_ADMIN",
    
    # Real Estate / Social Housing
    "HABITAT": "REAL_ESTATE",
    "HLM": "REAL_ESTATE",
    "LOGEMENT": "REAL_ESTATE",
}

def get_expected_category(crm_name: str) -> Optional[str]:
    """Identify expected business category based on keywords in the CRM name."""
    if not crm_name:
        return None
    tokens = set(crm_name.upper().replace("-", " ").replace("'", " ").split())
    for kw, cat in CRM_KEYWORDS_TO_CATEGORY.items():
        if kw in tokens:
            return cat
    return None

def is_naf_aligned(expected_category: Optional[str], naf_code: Optional[str]) -> Optional[bool]:
    """Check if the candidate's NAF code aligns with the expected CRM category.
    
    Supports both NAF Rev 2 (2008+) and NAF 1993 (pre-2008) nomenclatures.
    
    Returns:
        True if aligned, False if misaligned, None if category or NAF is missing.
    """
    if not expected_category or not naf_code:
        return None
        
    # Standardize APE/NAF code (remove dots, strip spaces)
    naf_clean = str(naf_code).replace(".", "").strip().upper()
    if not naf_clean:
        return None
        
    # Detect nomenclature version based on cleaned length (excluding dots)
    # NAF 1993 (pre-2008): e.g. "851A" (length 4)
    # NAF Rev 2 (2008+):  e.g. "8610Z" (length 5)
    is_naf_1993 = (len(naf_clean) == 4)
    
    if is_naf_1993:
        # NAF 1993 category mappings:
        # - Division 85: Santé et action sociale (Healthcare)
        # - Division 80: Enseignement (Education)
        # - Division 75: Administration publique (Public Admin)
        # - Division 70: Activités immobilières (Real Estate)
        if expected_category == "HEALTHCARE":
            return naf_clean.startswith("85")
        elif expected_category == "EDUCATION":
            return naf_clean.startswith("80")
        elif expected_category == "PUBLIC_ADMIN":
            return naf_clean.startswith("75")
        elif expected_category == "REAL_ESTATE":
            return naf_clean.startswith("70")
    else:
        # NAF Rev 2 category mappings:
        # - Divisions 86, 87, 88: Santé et action sociale (Healthcare)
        # - Division 85: Enseignement (Education)
        # - Division 84: Administration publique (Public Admin)
        # - Division 68: Activités immobilières (Real Estate)
        if expected_category == "HEALTHCARE":
            return any(naf_clean.startswith(prefix) for prefix in {"86", "87", "88"})
        elif expected_category == "EDUCATION":
            return naf_clean.startswith("85")
        elif expected_category == "PUBLIC_ADMIN":
            return naf_clean.startswith("84")
        elif expected_category == "REAL_ESTATE":
            return naf_clean.startswith("68")
            
    return None
