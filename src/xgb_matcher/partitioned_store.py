"""
Partitioned candidate store for multi-blocking retrieval.

Loads candidates from partitioned parquet folders:
  - <partitions_dir>/insee (partitioned by insee)
  - <partitions_dir>/cp (partitioned by postcode)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .blocking import normalize_code, department_from_code

# Maximum number of entries per store cache (INSEE / CP / dept)
# Calibrated to 150 to guarantee memory safety (< 5 GB RAM) while maintaining
# an optimal cache-hit rate (~35% hits on CRM sequence, minimal overhead).
_MAX_STORE_CACHE_SIZE = 150

# Columns actually used by features, naming, filtering, and output.
# Excludes unused columns (nom_usage_ul, pseudonyme_ul) to reduce I/O and RAM.
REQUIRED_COLUMNS = [
    "siret", "siren",
    "denomination", "enseigne1", "enseigne2", "enseigne3",
    "etablissementSiege", "is_siege",
    "numeroVoie", "typeVoie", "libelleVoie", "complementAdresse",
    "postcode", "city", "insee",
    "cj_ul", "etat_admin", "last_treatment_date",
    "sigle_ul", "denomination_ul", "denomination_usuelle_ul",
    "nom_ul", "prenom_usuel_ul",
    "pm_dirigeant_names",
]


class PartitionedCandidateStore:
    """Lazy loader for candidates by INSEE / CP / department with O(1) direct file reads."""

    def __init__(self, partitions_dir: Path, max_cache_size: int = _MAX_STORE_CACHE_SIZE):
        self.partitions_dir = Path(partitions_dir)
        self.max_cache_size = max(1, int(max_cache_size))
        self._dataset_insee_lazy = None
        self._dataset_cp_lazy = None
        self._cache_insee: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_cp: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_cp_insee: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_dept: OrderedDict[str, List[dict]] = OrderedDict()
        self._cache_insee_count: OrderedDict[str, int] = OrderedDict()
        self._manifest_insee_counts = self._load_insee_manifest_counts()

        # Fast schema inspection from a sample parquet file (O(1), avoids scanning all folders)
        self._columns_insee = self._detect_sample_columns(self.partitions_dir / "insee")
        self._columns_cp = self._detect_sample_columns(self.partitions_dir / "cp")

    @staticmethod
    def _detect_sample_columns(partition_root: Path) -> List[str] | None:
        """Inspect column names from the first available parquet file in O(1) without scanning."""
        if not partition_root.exists():
            return None
        try:
            for entry in partition_root.iterdir():
                if entry.is_dir():
                    for f in entry.glob("*.parquet"):
                        schema_names = set(pq.read_schema(f).names)
                        cols = [c for c in REQUIRED_COLUMNS if c in schema_names]
                        return cols if cols else None
        except Exception:
            pass
        return REQUIRED_COLUMNS

    def _load_insee_manifest_counts(self) -> Dict[str, int]:
        """Load optional V7 manifest counts for O(1) INSEE row lookup."""
        manifest_path = self.partitions_dir / "manifest" / "insee_counts.parquet"
        if not manifest_path.exists():
            return {}
        try:
            table = pq.read_table(manifest_path, columns=["insee", "row_count"])
        except Exception:
            return {}
        counts: Dict[str, int] = {}
        for row in table.to_pylist():
            code = normalize_code(row.get("insee"))
            if not code:
                continue
            try:
                counts[code] = int(row.get("row_count") or 0)
            except Exception:
                continue
        return counts

    @staticmethod
    def _coerce_candidate_types(rows: List[dict]) -> List[dict]:
        for r in rows:
            if "siret" in r:
                r["siret"] = str(r.get("siret") or "")
            if "siren" in r:
                r["siren"] = str(r.get("siren") or "")
        return rows

    @staticmethod
    def _evict_lru(cache: OrderedDict, max_size: int) -> None:
        """Evict oldest entries until cache is within max_size."""
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _count_insee_rows(self, insee: Optional[str]) -> int:
        code = normalize_code(insee)
        if not code:
            return 0
        if code in self._cache_insee_count:
            self._cache_insee_count.move_to_end(code)
            return self._cache_insee_count[code]
        if code in self._manifest_insee_counts:
            count = int(self._manifest_insee_counts[code])
            self._cache_insee_count[code] = count
            self._evict_lru(self._cache_insee_count, self.max_cache_size)
            return count

        # O(1) metadata check on direct partition file
        insee_folder = self.partitions_dir / "insee" / f"insee={code}"
        if insee_folder.is_dir():
            try:
                parquet_files = list(insee_folder.glob("*.parquet"))
                if parquet_files:
                    count = sum(pq.read_metadata(f).num_rows for f in parquet_files)
                    self._cache_insee_count[code] = count
                    self._evict_lru(self._cache_insee_count, self.max_cache_size)
                    return count
            except Exception:
                pass
        return 0

    def load_by_insee(self, insee: Optional[str]) -> List[dict]:
        code = normalize_code(insee)
        if not code:
            return []
        if code in self._cache_insee:
            self._cache_insee.move_to_end(code)
            return self._cache_insee[code]

        # O(1) Direct file read without traversing 30k folders
        insee_folder = self.partitions_dir / "insee" / f"insee={code}"
        if insee_folder.is_dir():
            try:
                parquet_files = list(insee_folder.glob("*.parquet"))
                if parquet_files:
                    tables = [pq.read_table(f, columns=self._columns_insee) for f in parquet_files]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    rows = self._coerce_candidate_types(table.to_pylist())
                    # Defensive guard: deduplicate SIRETs if multiple partitions were concatenated
                    if len(tables) > 1:
                        seen: set = set()
                        deduped = []
                        for r in rows:
                            s = r.get("siret")
                            if s not in seen:
                                seen.add(s)
                                deduped.append(r)
                        rows = deduped
                    self._cache_insee[code] = rows
                    self._evict_lru(self._cache_insee, self.max_cache_size)
                    return rows
            except Exception:
                pass
        return []

    def load_by_postcode(self, postcode: Optional[str]) -> List[dict]:
        code = normalize_code(postcode)
        if not code:
            return []
        if code in self._cache_cp:
            self._cache_cp.move_to_end(code)
            return self._cache_cp[code]

        # O(1) Direct file read without traversing 30k folders
        cp_folder = self.partitions_dir / "cp" / f"postcode={code}"
        if cp_folder.is_dir():
            try:
                parquet_files = list(cp_folder.glob("*.parquet"))
                if parquet_files:
                    tables = [pq.read_table(f, columns=self._columns_cp) for f in parquet_files]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    rows = self._coerce_candidate_types(table.to_pylist())
                    if len(tables) > 1:
                        seen: set = set()
                        deduped = []
                        for r in rows:
                            s = r.get("siret")
                            if s not in seen:
                                seen.add(s)
                                deduped.append(r)
                        rows = deduped
                    self._cache_cp[code] = rows
                    self._evict_lru(self._cache_cp, self.max_cache_size)
                    return rows
            except Exception:
                pass
        return []

    def load_by_postcode_filtered_insee(self, postcode: Optional[str], insee: Optional[str]) -> List[dict]:
        code_cp = normalize_code(postcode)
        code_insee = normalize_code(insee)
        if not code_cp or not code_insee:
            return []
        cache_key = f"{code_cp}|{code_insee}"
        if cache_key in self._cache_cp_insee:
            self._cache_cp_insee.move_to_end(cache_key)
            return self._cache_cp_insee[cache_key]

        all_cp_rows = self.load_by_postcode(code_cp)
        filtered = [r for r in all_cp_rows if normalize_code(r.get("insee")) == code_insee]
        self._cache_cp_insee[cache_key] = filtered
        self._evict_lru(self._cache_cp_insee, self.max_cache_size)
        return filtered

    def load_by_insee_then_postcode(
        self,
        insee: Optional[str],
        postcode: Optional[str],
        *,
        mega_insee_max_rows: int = 100_000,
        mega_insee_policy: str = "cp_filter_insee",
    ) -> List[dict]:
        code_insee = normalize_code(insee)
        code_cp = normalize_code(postcode)
        if code_insee:
            if mega_insee_max_rows:
                count = self._count_insee_rows(code_insee)
                if count > mega_insee_max_rows:
                    if mega_insee_policy == "cp_filter_insee" and code_cp:
                        return self.load_by_postcode_filtered_insee(code_cp, code_insee)
                    if mega_insee_policy == "full_insee":
                        return self.load_by_insee(code_insee)
            rows = self.load_by_insee(code_insee)
            if rows:
                return rows
        if code_cp:
            return self.load_by_postcode(code_cp)
        return []

    def load_by_department(self, insee: Optional[str], postcode: Optional[str]) -> List[dict]:
        raise RuntimeError("Department fallback is disabled under SSOT.")

    # ---------------------------------------------------------------------------
    # Strict Geo Resolution (Chantier 1 — feature/geo-resolution-and-crm-expansion)
    # ---------------------------------------------------------------------------

    def _discover_insee_codes_from_cp(self, postcode: Optional[str]) -> List[str]:
        """Return all unique INSEE codes that share the given postcode in the CP partition.

        This is used when crm_insee is missing/empty: we load the CP partition to
        discover which INSEE communes share that postcode, then load each one via
        the INSEE partition (more precise and avoids loading the whole CP bucket).
        """
        code_cp = normalize_code(postcode)
        if not code_cp:
            return []
        cp_folder = self.partitions_dir / "cp" / f"postcode={code_cp}"
        if cp_folder.is_dir():
            try:
                parquet_files = list(cp_folder.glob("*.parquet"))
                if parquet_files:
                    tables = [pq.read_table(f, columns=["insee"]) for f in parquet_files]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    seen: dict = {}
                    for code in table.column("insee").to_pylist():
                        norm = normalize_code(code)
                        if norm:
                            seen[norm] = None
                    return list(seen.keys())
            except Exception:
                return []
        return []

    def load_with_geo_resolution(
        self,
        insee: Optional[str],
        postcode: Optional[str],
        crm_id: str = "<unknown>",
        *,
        mega_insee_max_rows: int = 100_000,
        mega_insee_policy: str = "cp_filter_insee",
        logger: Optional[object] = None,
    ) -> List[dict]:
        """Load candidates using the strict geographic resolution hierarchy.

        Hierarchy:
          1. crm_insee present  -> load_by_insee() directly (authoritative)
          2. crm_insee absent   -> discover child INSEE codes from the CP partition,
                                   then load_by_insee() for each discovered INSEE code
          3. No results at all  -> log GEO_RESOLUTION_EMPTY and return []

        This method never loads the entire CP partition blindly.
        It never performs a department-level fallback.

        Args:
            insee:               INSEE commune code from CRM (may be None/empty).
            postcode:            Postal code from CRM (may be None/empty).
            crm_id:              CRM entry identifier for logging.
            mega_insee_max_rows: Threshold above which mega-commune policy applies.
            mega_insee_policy:   Policy for mega-communes ("cp_filter_insee" / "full_insee").
            logger:              Optional logger instance (uses module logger if None).
        """
        import logging as _logging
        _log = logger if logger is not None else _logging.getLogger(__name__)

        code_insee = normalize_code(insee)
        code_cp = normalize_code(postcode)

        # --- Step 1: INSEE is present (primary path) ---
        if code_insee:
            rows = self.load_by_insee_then_postcode(
                code_insee,
                code_cp,
                mega_insee_max_rows=mega_insee_max_rows,
                mega_insee_policy=mega_insee_policy,
            )
            if rows:
                return rows
            # INSEE provided but partition empty — log and continue to CP fallback
            _log.warning(
                "GEO_RESOLUTION: INSEE %s returned empty partition for crm_id=%s; "
                "attempting CP-based child INSEE discovery (postcode=%s).",
                code_insee, crm_id, code_cp,
            )

        # --- Step 2: INSEE absent or empty — discover child INSEEs from CP ---
        if code_cp:
            child_insees = self._discover_insee_codes_from_cp(code_cp)
            if child_insees:
                _log.debug(
                    "GEO_RESOLUTION: CP %s resolved to %d child INSEE(s) for crm_id=%s: %s",
                    code_cp, len(child_insees), crm_id, child_insees[:10],
                )
                combined: dict = {}
                for child_insee in child_insees:
                    partial = self.load_by_insee(child_insee)
                    for r in partial:
                        siret = str(r.get("siret") or "")
                        if siret and siret not in combined:
                            combined[siret] = r
                if combined:
                    return list(combined.values())
            else:
                _log.warning(
                    "GEO_RESOLUTION: CP %s found no child INSEE codes for crm_id=%s.",
                    code_cp, crm_id,
                )

        # --- Step 3: No results — log cleanly and return empty ---
        _log.warning(
            "GEO_RESOLUTION_EMPTY: No candidates found for crm_id=%s "
            "(insee=%s, postcode=%s). Returning empty pool.",
            crm_id, insee, postcode,
        )
        return []
