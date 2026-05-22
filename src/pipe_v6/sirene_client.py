"""SIRENE API client (task 2.2).

Implements fetching all establishments for a given commune and exposes
helper utilities used by the cache layer (task 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
import time
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from .config import PipelineConfig
from .commune_detection import CommuneKey


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filter:
    key: str
    value: str


def _mask_key(k: str) -> str:
    if not k:
        return "<empty>"
    if len(k) <= 6:
        return "***"
    return f"{k[:4]}…{k[-4:]}"


def arrondissement_codes_for_parent(insee_code: str) -> List[str]:
    """Return arrondissement INSEE codes for Paris, Lyon, Marseille parent codes.

    - Paris parent: 75056 -> 75101..75120
    - Lyon parent: 69123 -> 69381..69389
    - Marseille parent: 13055 -> 13201..13216
    Otherwise, returns [insee_code].
    """

    if insee_code == "75056":
        return [f"75{100 + i:03d}" for i in range(1, 21)]  # 75101..75120
    if insee_code == "69123":
        return [f"6938{i}" for i in range(1, 10)]  # 69381..69389
    if insee_code == "13055":
        return [f"132{str(i).zfill(2)}" for i in range(1, 17)]  # 13201..13216
    return [insee_code]


def resolve_filters_for_commune(commune: CommuneKey) -> List[Filter]:
    """Compute the list of filters to fetch for this commune.

    Priority:
    - If INSEE is available: codeCommuneEtablissement for that code, expanding
      parent-city codes to arrondissement codes for P/L/M.
    - Else if postcode is available: codePostalEtablissement only (no city).
    - Else: raise ValueError (city-only fetch is not supported by the current rules).
    """

    if commune.insee_code:
        codes = arrondissement_codes_for_parent(commune.insee_code)
        return [Filter("codeCommuneEtablissement", c) for c in codes]
    if commune.postcode:
        return [Filter("codePostalEtablissement", commune.postcode)]
    raise ValueError("Cannot resolve filter: INSEE code or postcode required")


def _http_get_json(url: str, headers: Dict[str, str], *, timeout: float = 15.0, max_retries: int = 3, logger: logging.Logger | None = None) -> Dict:
    """HTTP GET with retries and basic 429 handling, returning parsed JSON."""

    logger = logger or LOGGER
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except HTTPError as e:
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            if status in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2.0
                else:
                    delay = min(2.0 * attempt, 10.0) + (0.1 * (attempt - 1))
                logger.warning("HTTP %s on %s (attempt %s/%s) – sleeping %.1fs", status, url, attempt, max_retries, delay)
                time.sleep(delay)
                last_exc = e
                continue
            else:
                logger.error("HTTP %s on %s -> %s", status, url, body[:200])
                raise
        except URLError as e:
            delay = min(2.0 * attempt, 10.0)
            logger.warning("URLError on %s (attempt %s/%s) – sleeping %.1fs", url, attempt, max_retries, delay)
            time.sleep(delay)
            last_exc = e
            continue
    assert last_exc is not None
    raise last_exc


def _ensure_trailing_slash(base: str) -> str:
    return base if base.endswith('/') else base + '/'


def fetch_establishments_for_commune(
    commune: CommuneKey, config: PipelineConfig, logger: logging.Logger | None = None
) -> List[dict]:
    """Fetch all SIRENE establishments for a given commune.

    - Builds the appropriate filter(s) based on `CommuneKey`.
    - Paginates with `nombre=1000`, using `debut` offsets until all results are retrieved.
    - Returns a de-duplicated list of raw establishment dicts (complete objects).
    """

    logger = logger or LOGGER

    base = _ensure_trailing_slash(config.sirene_api_url)
    endpoint = urljoin(base, "siret")

    api_key = (getattr(config, "sirene_token", None) or getattr(config, "sirene_api_key", None) or "").strip()
    if not api_key:
        return _fetch_from_datagouv_fallback(commune, config, logger)

    headers = {
        "X-INSEE-Api-Key-Integration": api_key,
        "Accept": "application/json",
        "User-Agent": "Sireto-PipeV6/2.2",
    }

    filters = resolve_filters_for_commune(commune)
    logger.info(
        "SIRENE fetch: base=%s, key=%s, filters=%s",
        config.sirene_api_url,
        _mask_key(api_key),
        [f"{f.key}={f.value}" for f in filters],
    )

    all_records: List[dict] = []
    seen_siret: set[str] = set()
    per_page = 1000

    for flt in filters:
        params = {"q": f"{flt.key}:{flt.value}", "nombre": per_page}
        cursor = "*"
        fetched = 0
        total = None

        while True:
            if cursor is not None:
                params["curseur"] = cursor
            url = f"{endpoint}?{urlencode(params)}"
            payload = _http_get_json(url, headers, logger=logger)
            header = payload.get("header", {})
            if total is None:
                total = int(header.get("total", 0))
                logger.info(
                    "SIRENE fetch start: %s=%s -> total=%s",
                    flt.key,
                    flt.value,
                    total,
                )
            results = payload.get("etablissements", []) or payload.get("etablissement", []) or []
            logger.debug(
                "SIRENE page: %s=%s -> cursor=%s batch=%s",
                flt.key,
                flt.value,
                cursor,
                len(results),
            )
            for rec in results:
                siret = str(rec.get("siret", "")).strip()
                if siret and siret not in seen_siret:
                    all_records.append(rec)
                    seen_siret.add(siret)
            fetched += len(results)

            cursor_next = header.get("curseurSuivant")
            if not cursor_next or not results:
                break
            cursor = cursor_next

    logger.info(
        "SIRENE fetch done: commune=%s (insee=%s, cp=%s), records=%s",
        commune.city or "",
        commune.insee_code or "",
        commune.postcode or "",
        len(all_records),
    )

    return all_records


def _parse_address_string(addr_str: str | None) -> tuple[str | None, str | None, str | None]:
    """Parse a French address string into (street_number, street_type, street_name)."""
    if not addr_str:
        return None, None, None
    
    # Remove postcode and city at the end (e.g. 69150 DECINES-CHARPIEU or similar 5-digit code)
    cleaned = re.sub(r'\s+\d{5}\s+.*$', '', addr_str, flags=re.IGNORECASE).strip()
    
    # Try to match street number at start (e.g. 16, 16 BIS, 16A, etc.)
    m = re.match(r'^(\d+)(?:\s+(BIS|TER|A|B|C|QUATER))?\s+(.*)$', cleaned, re.IGNORECASE)
    if m:
        num = m.group(1)
        rep = m.group(2)
        street_part = m.group(3)
        street_number = f"{num} {rep}" if rep else num
    else:
        street_number = None
        street_part = cleaned
        
    # Common French street types
    street_types = [
        "RUE", "AVENUE", "BOULEVARD", "ALLÉE", "ALLEY", "CHEMIN", "PLACE", "ROUTE", "IMPASSE", 
        "SQUARE", "COURS", "QUAI", "PROMENADE", "ZONE", "ZA", "ZI", "ZAC", "AV", "BD", "PL", "RTE",
        "ALL", "CHE"
    ]
    words = street_part.split()
    if words and words[0].upper() in street_types:
        street_type = words[0]
        street_name = " ".join(words[1:])
    else:
        street_type = None
        street_name = street_part
        
    return street_number, street_type, street_name


def _map_datagouv_to_insee_format(result: dict) -> List[dict]:
    """Map a DataGouv company search result to a list of raw INSEE-style establishment dicts."""
    insee_records = []
    
    siren = result.get("siren")
    nom_complet = result.get("nom_complet") or result.get("nom_raison_sociale")
    nature_juridique = result.get("nature_juridique")
    activite_principale_ul = result.get("activite_principale")
    tranche_effectifs_ul = result.get("tranche_effectif_salarie")
    annee_effectifs_ul = result.get("annee_tranche_effectif_salarie")
    
    unite_legale = {
        "denominationUniteLegale": nom_complet,
        "categorieJuridiqueUniteLegale": nature_juridique,
        "activitePrincipaleUniteLegale": activite_principale_ul,
        "trancheEffectifsUniteLegale": tranche_effectifs_ul,
        "anneeEffectifsUniteLegale": annee_effectifs_ul,
        "dateCreationUniteLegale": result.get("date_creation"),
    }
    
    # Process both siege and matching establishments
    etabs_to_process = []
    
    siege = result.get("siege")
    if siege:
        etabs_to_process.append(siege)
        
    for etab in result.get("matching_etablissements") or []:
        if siege and etab.get("siret") == siege.get("siret"):
            continue
        etabs_to_process.append(etab)
        
    for etab in etabs_to_process:
        siret = etab.get("siret")
        if not siret:
            continue
            
        enseignes = etab.get("liste_enseignes") or []
        enseigne1 = enseignes[0] if len(enseignes) > 0 else None
        enseigne2 = enseignes[1] if len(enseignes) > 1 else None
        enseigne3 = enseignes[2] if len(enseignes) > 2 else None
        
        addr_str = etab.get("adresse")
        num_part, type_part, name_part = _parse_address_string(addr_str)
        
        insee_rec = {
            "siret": siret,
            "siren": siren,
            "nic": siret[-5:] if len(siret) == 14 else "",
            "etablissementSiege": etab.get("est_siege", False),
            "uniteLegale": unite_legale,
            "enseigne1Etablissement": enseigne1,
            "enseigne2Etablissement": enseigne2,
            "enseigne3Etablissement": enseigne3,
            "adresseEtablissement": {
                "numeroVoieEtablissement": num_part,
                "indiceRepetitionEtablissement": None,
                "typeVoieEtablissement": type_part,
                "libelleVoieEtablissement": name_part,
                "complementAdresseEtablissement": etab.get("complement_adresse"),
                "codePostalEtablissement": etab.get("code_postal"),
                "libelleCommuneEtablissement": etab.get("libelle_commune"),
                "codeCommuneEtablissement": etab.get("commune"),
            },
            "etatAdministratifEtablissement": etab.get("etat_administratif"),
            "dateCreationEtablissement": etab.get("date_creation"),
            "dateDebut": etab.get("date_debut_activite"),
            "dateDernierTraitementEtablissement": etab.get("date_mise_a_jour") or etab.get("date_mise_a_jour_insee"),
            "activitePrincipaleEtablissement": etab.get("activite_principale"),
            "trancheEffectifsEtablissement": etab.get("tranche_effectif_salarie"),
            "anneeEffectifsEtablissement": etab.get("annee_tranche_effectif_salarie"),
        }
        insee_records.append(insee_rec)
        
    return insee_records


def _fetch_from_datagouv_fallback(
    commune: CommuneKey, config: PipelineConfig, logger: logging.Logger
) -> List[dict]:
    """Fetch establishments using the public search API as a fallback when INSEE credentials are missing."""
    
    logger.info("Using keyless DataGouv API fallback for commune: insee=%s postcode=%s city=%s",
                commune.insee_code, commune.postcode, commune.city)
    
    base_url = f"{config.datagouv_api_url.rstrip('/')}/search"
    
    params: dict[str, Any] = {}
    if commune.insee_code:
        params["code_commune"] = commune.insee_code
    elif commune.postcode:
        params["code_postal"] = commune.postcode
    else:
        logger.warning("Neither INSEE code nor postcode available for commune: %s. Fallback cannot query.", commune)
        return []
        
    params["per_page"] = 25
    headers = {
        "Accept": "application/json",
        "User-Agent": "Sireto-PipeV6/2.2",
    }
    
    all_records: List[dict] = []
    seen_sirets = set()
    page = 1
    
    # Rate limiter: max 7 requests per second, so 0.15s delay between requests
    last_req_time = 0.0
    
    while True:
        params["page"] = page
        url = f"{base_url}?{urlencode(params)}"
        
        # Enforce rate limit
        now = time.time()
        elapsed = now - last_req_time
        if elapsed < 0.15:
            time.sleep(0.15 - elapsed)
        
        logger.info("Fetching DataGouv page %d for commune %s...", page, commune.city or commune.postcode)
        try:
            last_req_time = time.time()
            payload = _http_get_json(url, headers, timeout=20.0, logger=logger)
        except Exception as e:
            logger.error("Failed to fetch DataGouv page %d: %s", page, e)
            break
            
        results = payload.get("results") or []
        if not results:
            break
            
        for company in results:
            mapped_records = _map_datagouv_to_insee_format(company)
            for rec in mapped_records:
                siret = rec.get("siret")
                # Filter by geographic code to ensure we only load establishments matching the target commune/postcode
                addr_etab = rec.get("adresseEtablissement") or {}
                etab_insee = addr_etab.get("codeCommuneEtablissement")
                etab_postcode = addr_etab.get("codePostalEtablissement")
                
                geo_match = False
                if commune.insee_code and etab_insee == commune.insee_code:
                    geo_match = True
                elif not commune.insee_code and commune.postcode and etab_postcode == commune.postcode:
                    geo_match = True
                
                if geo_match and siret and siret not in seen_sirets:
                    all_records.append(rec)
                    seen_sirets.add(siret)
                    
        total_pages = payload.get("total_pages", 1)
        if page >= total_pages or len(results) < 25 or len(all_records) >= 10000:
            break
            
        page += 1
        
    logger.info("Fallback done: fetched %d establishments from DataGouv for %s", len(all_records), commune.city or commune.postcode)
    return all_records


__all__ = [
    "fetch_establishments_for_commune",
    "arrondissement_codes_for_parent",
    "resolve_filters_for_commune",
    "Filter",
]
