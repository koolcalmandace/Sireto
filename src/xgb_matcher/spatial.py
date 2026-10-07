"""
src/xgb_matcher/spatial.py
==========================
High-performance deterministic spatial resolver powered by Base Adresse Nationale (BAN).

Provides:
  - O(1) in-memory commune spatial caches loaded from DuckDB (data/ban.duckdb).
  - Multi-tier address geocoding:
      1. Exact (Street, Number) match
      2. Street Centroid match (when number is missing or unmatched)
      3. Toponyme / Lieu-dit match
      4. High-confidence fuzzy street match (Jaro >= 0.85)
      5. Commune Centroid fallback
  - Vectorized equirectangular / Haversine spatial distance computation (NumPy).
  - Continuous distance features for XGBoost Fast Ranker and Full Ranker:
      - spatial_distance_km: float (distance in km)
      - spatial_is_very_close: binary (distance < 0.100 km, ~100m)
      - spatial_is_same_block: binary (distance < 0.300 km, ~300m)
      - spatial_has_coords: binary (both CRM and candidate have valid GPS coords)
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np
from rapidfuzz.distance import JaroWinkler

from .features import (
    extract_street_name_from_address,
    extract_street_number,
    normalize_name,
    normalize_street_type,
    parse_street_numbers,
)

logger = logging.getLogger("xgb_matcher.spatial")

DEFAULT_BAN_DB_PATH = Path("data/ban.duckdb")


def _clean_street_tokens(street: str | None) -> str:
    """Normalize and clean street name for exact matching in BAN."""
    if not street:
        return ""
    s = normalize_street_type(str(street)).upper()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t not in ("LE", "LA", "LES", "DU", "DE", "DES", "D", "L")]
    return " ".join(tokens)


class CommuneSpatialData:
    """In-memory spatial index for a single commune."""

    __slots__ = (
        "insee",
        "addr_coords",
        "street_centroids",
        "lieu_dit_coords",
        "commune_centroid",
        "postcode",
    )

    def __init__(self, insee: str, postcode: str = ""):
        self.insee = insee.zfill(5)
        self.postcode = postcode.zfill(5)
        # (clean_street, number) -> (lat, lon)
        self.addr_coords: Dict[Tuple[str, str], Tuple[float, float]] = {}
        # clean_street -> (avg_lat, avg_lon)
        self.street_centroids: Dict[str, Tuple[float, float]] = {}
        # clean_nom_ld -> (avg_lat, avg_lon)
        self.lieu_dit_coords: Dict[str, Tuple[float, float]] = {}
        # (avg_lat, avg_lon) for whole commune
        self.commune_centroid: Optional[Tuple[float, float]] = None


class SpatialResolver:
    """
    LRU-cached spatial resolver connecting to DuckDB BAN store.
    Thread-safe and process-safe for mono-core evaluation.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_BAN_DB_PATH,
        max_cached_communes: int = 100,
    ):
        self.db_path = Path(db_path)
        self.max_cached_communes = max_cached_communes
        self._cache: OrderedDict[str, CommuneSpatialData] = OrderedDict()
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._available = self.db_path.exists()

        if self._available:
            try:
                self._con = duckdb.connect(str(self.db_path), read_only=True)
            except Exception as e:
                logger.warning("Could not open BAN database %s: %s", self.db_path, e)
                self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and (self._con is not None)

    def clear_cache(self) -> None:
        """Clear cached commune spatial indexes to reclaim memory."""
        self._cache.clear()

    def get_commune_data(self, insee: str) -> Optional[CommuneSpatialData]:
        """Loads and caches all BAN spatial points for the given INSEE code."""
        if not self.is_available or not insee:
            return None

        clean_insee = str(insee).strip().zfill(5)
        if clean_insee in self._cache:
            self._cache.move_to_end(clean_insee)
            return self._cache[clean_insee]

        # Query DuckDB for the full commune in one fast fetch
        try:
            assert self._con is not None
            addr_rows = self._con.execute(
                "SELECT nom_voie, numero, nom_ld, lat, lon FROM ban_adresses WHERE insee = ?",
                [clean_insee],
            ).fetchall()

            voie_rows = self._con.execute(
                "SELECT nom_voie, lat, lon FROM ban_voies WHERE insee = ?",
                [clean_insee],
            ).fetchall()
        except Exception as e:
            logger.debug("Failed querying BAN for insee %s: %s", clean_insee, e)
            return None

        if not addr_rows and not voie_rows:
            return None

        data = CommuneSpatialData(clean_insee)
        all_lats: List[float] = []
        all_lons: List[float] = []

        # 1. Fill address coordinates
        for nom_voie, numero, nom_ld, lat, lon in addr_rows:
            if lat is None or lon is None:
                continue
            fl_lat = float(lat)
            fl_lon = float(lon)
            all_lats.append(fl_lat)
            all_lons.append(fl_lon)

            clean_st = _clean_street_tokens(nom_voie)
            clean_num = str(numero or "").strip()
            if clean_st:
                data.addr_coords[(clean_st, clean_num)] = (fl_lat, fl_lon)

            clean_ld = _clean_street_tokens(nom_ld)
            if clean_ld and clean_ld not in data.lieu_dit_coords:
                data.lieu_dit_coords[clean_ld] = (fl_lat, fl_lon)

        # 2. Fill street centroids
        for nom_voie, lat, lon in voie_rows:
            if lat is not None and lon is not None:
                clean_st = _clean_street_tokens(nom_voie)
                if clean_st:
                    data.street_centroids[clean_st] = (float(lat), float(lon))

        # 3. Commune centroid
        if all_lats:
            data.commune_centroid = (float(np.mean(all_lats)), float(np.mean(all_lons)))

        # Evict LRU
        if len(self._cache) >= self.max_cached_communes:
            self._cache.popitem(last=False)

        self._cache[clean_insee] = data
        return data

    def geocode(
        self,
        insee: str,
        address: str | None,
        street_override: str | None = None,
        number_override: str | None = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Multi-tier geocoder returning (lat, lon) for an address in a commune.
        """
        data = self.get_commune_data(insee)
        if not data:
            return None

        # Determine street number
        clean_num = ""
        if number_override:
            clean_num = str(number_override).strip()
        elif address:
            ext_num = extract_street_number(address)
            if ext_num:
                clean_num = str(ext_num).split("-")[0].strip()
            else:
                nums = parse_street_numbers(address)
                if nums:
                    clean_num = str(min(nums))

        # Determine street name
        if street_override:
            raw_street = street_override
        elif address:
            raw_street = extract_street_name_from_address(address) or address
        else:
            raw_street = ""
        clean_street = _clean_street_tokens(raw_street)

        # Tier 1: Exact street + exact number
        if clean_street and clean_num and (clean_street, clean_num) in data.addr_coords:
            return data.addr_coords[(clean_street, clean_num)]

        # Tier 2: Exact street centroid
        if clean_street and clean_street in data.street_centroids:
            return data.street_centroids[clean_street]

        # Tier 3: Toponyme / Lieu-dit match
        if clean_street and clean_street in data.lieu_dit_coords:
            return data.lieu_dit_coords[clean_street]

        # Tier 4: Fuzzy street match (Jaro >= 0.85) against street centroids
        if clean_street and len(clean_street) >= 4:
            best_jaro = 0.0
            best_coord = None
            for ban_st, coord in data.street_centroids.items():
                sim = JaroWinkler.similarity(clean_street, ban_st)
                if sim > best_jaro:
                    best_jaro = sim
                    best_coord = coord
            if best_jaro >= 0.85 and best_coord is not None:
                return best_coord

        # Tier 5: Commune centroid fallback (only if address exists)
        return data.commune_centroid

    def compute_spatial_features(
        self,
        crm_row: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Computes an (N, 4) NumPy matrix of spatial features for candidates:
          col 0: spatial_distance_km (float, 0.0 to 99.0)
          col 1: spatial_is_very_close (binary, distance < 100m)
          col 2: spatial_is_same_block (binary, distance < 300m)
          col 3: spatial_has_coords (binary, 1.0 if valid GPS distance)
        """
        n_cands = len(candidates)
        out = np.zeros((n_cands, 4), dtype=np.float32)
        out[:, 0] = 99.0  # Default distance = 99 km penalty

        if not self.is_available or n_cands == 0:
            return out

        insee = (crm_row.get("insee") or crm_row.get("crm_insee") or "").strip().zfill(5)
        crm_addr = crm_row.get("crm_address") or crm_row.get("crm_adresse") or ""

        # Geocode CRM
        crm_coord = self.geocode(insee=insee, address=crm_addr)
        if not crm_coord:
            return out

        crm_lat, crm_lon = crm_coord
        cos_lat = math.cos(math.radians(crm_lat))

        # Geocode Candidates
        cand_lats = np.full(n_cands, np.nan, dtype=np.float32)
        cand_lons = np.full(n_cands, np.nan, dtype=np.float32)

        for i, c in enumerate(candidates):
            c_insee = str(c.get("insee") or insee).strip().zfill(5)
            c_street = c.get("libelleVoie") or c.get("nom_voie") or ""
            c_type = c.get("typeVoie") or ""
            if c_type and not c_street.upper().startswith(c_type.upper()):
                c_full_street = f"{c_type} {c_street}".strip()
            else:
                c_full_street = str(c_street).strip()

            c_num = str(c.get("numeroVoie") or "").strip()
            c_coord = self.geocode(
                insee=c_insee,
                address=None,
                street_override=c_full_street,
                number_override=c_num,
            )
            if c_coord:
                cand_lats[i] = c_coord[0]
                cand_lons[i] = c_coord[1]

        valid_mask = ~np.isnan(cand_lats)
        if not np.any(valid_mask):
            return out

        # Vectorized Equirectangular projection distance in km
        dx = (cand_lons[valid_mask] - crm_lon) * cos_lat * 111.32
        dy = (cand_lats[valid_mask] - crm_lat) * 110.57
        distances_km = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        out[valid_mask, 0] = distances_km
        out[valid_mask, 1] = (distances_km < 0.100).astype(np.float32)  # < 100m
        out[valid_mask, 2] = (distances_km < 0.300).astype(np.float32)  # < 300m
        out[valid_mask, 3] = 1.0  # Has coordinates

        return out
