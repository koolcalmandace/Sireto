import re

DEPARTMENTS = {
    "77": "SEINE ET MARNE",
    "44": "LOIRE ATLANTIQUE",
    "59": "NORD",
    "60": "OISE",
}

def apply_local_translations(text: str) -> str:
    """Translate common CRM abbreviations to align with official SIRENE names."""
    if not text:
        return ""
    text_upper = text.upper()
    
    # 1. Translate CG77, CG44, CG59 etc. (Conseil Général) to DEPARTEMENT [NAME]
    m = re.search(r"\bCG\s*([0-9]{2})\b", text_upper)
    if m:
        dept_num = m.group(1)
        dept_name = DEPARTMENTS.get(dept_num, "")
        if dept_name:
            text_upper = text_upper.replace(m.group(0), f"DEPARTEMENT DE {dept_name} CONSEIL GENERAL")
            
    # 2. Translate DSI [CITY] to COMMUNE DE [CITY]
    m_dsi = re.search(r"\bDSI\s+([A-Z]+)\b", text_upper)
    if m_dsi:
        city = m_dsi.group(1)
        text_upper = text_upper.replace(m_dsi.group(0), f"COMMUNE DE {city} MAIRIE")
        
    # 3. Translate COM DE COM to COMMUNAUTE DE COMMUNES
    text_upper = text_upper.replace("COM DE COM", "COMMUNAUTE DE COMMUNES CC")
        
    return text_upper

def clean_street_name(address: str) -> str:
    """Extract the core street name word for SQL LIKE matching (e.g. 'WATTEAU' from '20 rue Antoine Watteau')."""
    if not address:
        return ""
    addr_upper = str(address).upper()
    
    # Remove numbers
    addr_upper = re.sub(r"\b[0-9]+\b", "", addr_upper)
    
    # Remove common French street types and qualifiers
    for word in ["RUE", "AVENUE", "BOULEVARD", "ALLEE", "ROUTE", "CHEMIN", "PLACE", "SQUARE", "IMPASSE", "COURS", "AV", "BD", "R", "RTE", "PL", "ILOT", "BAT", "NUMERO", "NUM", "RUELE", "RUELLE"]:
        addr_upper = re.sub(r"\b" + word + r"\b", "", addr_upper)
        
    # Remove common French prepositions/articles
    for word in ["DE", "LA", "LE", "DU", "DES", "LES", "ET", "EN", "AU", "AUX", "SUR", "D", "L", "DESE", "DESE", "DES"]:
        addr_upper = re.sub(r"\b" + word + r"\b", "", addr_upper)
        
    # Retrieve first alphabetical token longer than 2 characters
    words = [w for w in re.sub(r"[^A-Z]", " ", addr_upper).split() if len(w) > 2]
    if words:
        return words[0]
    return ""
