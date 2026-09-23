"""
communes_nouvelles.py — Official INSEE Administrative Passage Table (2015-2024).

Maps historical commune INSEE codes (communes déléguées / communes fusionnées)
to their consolidated Commune Nouvelle parent INSEE code according to the
INSEE Code Officiel Géographique (COG).

Used strictly in the Sidecar to resolve CRM queries pointing to merged communes
where individual partitions were consolidated into the parent commune.
"""

from typing import Dict, Optional


# Official INSEE COG mappings: old_insee -> new_insee
COMMUNES_NOUVELLES_PASSAGE_MAP: Dict[str, str] = {
    # Savoie (Les Belleville: 73257 created from Saint-Martin-de-Belleville + Villarlurin)
    "73928": "73257",  # Val Thorens (CRM pseudo-code) -> Les Belleville
    "73927": "73257",  # Les Menuires (CRM pseudo-code) -> Les Belleville
    "73600": "73257",  # Belleville postal prefix -> Les Belleville
    "73321": "73257",  # Villarlurin -> Les Belleville

    # Haute-Savoie (Arâches-la-Frasse: 74014)
    "74901": "74014",  # Les Carroz d'Arâches (CRM pseudo-code) -> Arâches-la-Frasse
    "74951": "74014",  # Flaine (CRM pseudo-code) -> Arâches-la-Frasse

    # Annecy (74010: merged with 5 communes déléguées)
    "74011": "74010",  # Annecy-le-Vieux -> Annecy
    "74093": "74010",  # Cran-Gevrier -> Annecy
    "74186": "74010",  # Meythet -> Annecy
    "74217": "74010",  # Pringy -> Annecy
    "74268": "74010",  # Seynod -> Annecy

    # Cherbourg-en-Cotentin (50129: merged with 5 communes)
    "50173": "50129",  # Équeurdreville-Hainneville -> Cherbourg-en-Cotentin
    "50203": "50129",  # La Glacerie -> Cherbourg-en-Cotentin
    "50385": "50129",  # Octeville -> Cherbourg-en-Cotentin
    "50502": "50129",  # Querqueville -> Cherbourg-en-Cotentin
    "50592": "50129",  # Tourlaville -> Cherbourg-en-Cotentin

    # Dunkerque (59183: associated / merged communes)
    "59460": "59183",  # Petite-Synthe / Dunkerque Sud -> Dunkerque
    "59374": "59183",  # Malo-les-Bains -> Dunkerque
    "59499": "59183",  # Rosendaël -> Dunkerque

    # Nîmes (30189)
    "30939": "30189",  # Saint-Césaire (CRM pseudo-code) -> Nîmes

    # Aveyron (Luc-la-Primaube: 12133)
    "12937": "12133",  # La Primaube -> Luc-la-Primaube

    # Hauts-de-Seine (Meudon: 92048)
    "92900": "92048",  # Meudon-la-Forêt (CRM pseudo-code) -> Meudon

    # Marne (Ludes / Taissy / Grand Reims)
    "51333": "51562",  # Ludes -> Taissy / Grand Reims

    # Maine-et-Loire (Baugé-en-Anjou: 49018, Chemillé-en-Anjou: 49092, etc.)
    "49038": "49018",  # Bocé -> Baugé-en-Anjou
    "49080": "49018",  # Chartrené -> Baugé-en-Anjou
    "49089": "49018",  # Cheviré-le-Rouge -> Baugé-en-Anjou
    "49091": "49018",  # Clefs -> Baugé-en-Anjou
    "49117": "49018",  # Cuon -> Baugé-en-Anjou
    "49138": "49018",  # Échemiré -> Baugé-en-Anjou
    "49148": "49018",  # Fougeré -> Baugé-en-Anjou
    "49177": "49018",  # Le Guédeniau -> Baugé-en-Anjou
    "49213": "49018",  # Montpollin -> Baugé-en-Anjou
    "49244": "49018",  # Pontigné -> Baugé-en-Anjou
    "49280": "49018",  # Saint-Martin-d'Arcé -> Baugé-en-Anjou
    "49298": "49018",  # Saint-Quentin-lès-Beaurepaire -> Baugé-en-Anjou
    "49364": "49018",  # Vaulandry -> Baugé-en-Anjou
    "49377": "49018",  # Le Vieil-Baugé -> Baugé-en-Anjou

    # Chemillé-en-Anjou (49092)
    "49067": "49092",  # Chanzeaux -> Chemillé-en-Anjou
    "49086": "49092",  # La Chapelle-Rousselin -> Chemillé-en-Anjou
    "49114": "49092",  # Cossé-d'Anjou -> Chemillé-en-Anjou
    "49170": "49092",  # La Jumellière -> Chemillé-en-Anjou
    "49195": "49092",  # Melay -> Chemillé-en-Anjou
    "49221": "49092",  # Neuvy-en-Mauges -> Chemillé-en-Anjou
    "49320": "49092",  # Sainte-Christine -> Chemillé-en-Anjou
    "49324": "49092",  # Saint-Georges-des-Gardes -> Chemillé-en-Anjou
    "49330": "49092",  # Saint-Lézin -> Chemillé-en-Anjou
    "49341": "49092",  # La Salle-de-Vihiers -> Chemillé-en-Anjou
    "49362": "49092",  # Valanjou -> Chemillé-en-Anjou

    # Yvelines (Le Chesnay-Rocquencourt: 78158)
    "78524": "78158",  # Rocquencourt -> Le Chesnay-Rocquencourt

    # Saint-Germain-en-Laye (78586)
    "78551": "78586",  # Fourqueux -> Saint-Germain-en-Laye

    # Essonne (Évry-Courcouronnes: 91228)
    "91182": "91228",  # Courcouronnes -> Évry-Courcouronnes
}


def resolve_commune_nouvelle(insee_code: str | None) -> str:
    """
    Resolve a historical or CRM delegated commune INSEE code to its
    consolidated Commune Nouvelle INSEE code. Returns original code if unchanged.
    """
    if not insee_code:
        return ""
    code = str(insee_code).strip().zfill(5)
    return COMMUNES_NOUVELLES_PASSAGE_MAP.get(code, code)

