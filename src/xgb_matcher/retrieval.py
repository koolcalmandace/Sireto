"""Unified retrieval module (Variant B / SSOT) — with hybrid dense+sparse support.

Single code-path for candidate pool construction (train + serve).
Strict insee_then_postcode, no department fallback.

P0: Persistent TF-IDF cache + timing instrumentation.
P1: Optional hybrid dense (FAISS) + sparse (TF-IDF) retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import hashlib
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .partitioned_store import PartitionedCandidateStore
from .retrieval_config import RetrievalConfigV1
from .candidates import compute_name_idf_map
from .blocking import (
    build_tfidf_index,
    build_address_tfidf_index,
    build_char_tfidf_index,
    prefilter_candidates_tfidf,
    prefilter_candidates_address_tfidf,
    dedupe_candidates,
    address_hash,
    extract_numeric_tokens,
    attach_address_density,
    build_address_hash_index,
    build_numeric_token_index,
)

if TYPE_CHECKING:
    from .tfidf_cache import TfidfPersistentCache
    from .dense_retrieval import PartitionEmbeddingStore
    from .timing import PipelineTimer

_logger = logging.getLogger(__name__)


@dataclass
class CandidatePoolResult:
    """Result of candidate pool construction with GT tracking."""

    candidates: List[dict]

    # Prefilter order (TF-IDF/union) for cheap sampling
    prefilter_sirets: List[str] = field(default_factory=list)

    # GT tracking (for train/eval)
    gt_in_base_pool: bool = False
    gt_in_filtered_pool: bool = False
    gt_in_tfidf_pool: bool = False
    gt_was_injected: bool = False

    # Pool sizes at each stage
    pool_sizes: Dict[str, int] = field(default_factory=dict)

    # IDF map (from full pool, pre-TF-IDF)
    idf_map: Dict[str, float] = field(default_factory=dict)
    default_idf: float = 0.0

    # Loss reason if GT not in final pool
    loss_reason: str = ""


def _check_siret_in_list(candidates: List[dict], siret: str | None) -> bool:
    if not siret:
        return False
    siret_norm = str(siret).zfill(14)
    return any(str(c.get("siret", "")).zfill(14) == siret_norm for c in candidates)


def _get_siret_set(candidates: List[dict]) -> Set[str]:
    return {str(c.get("siret", "")).zfill(14) for c in candidates if c.get("siret")}


def _apply_filters(
    candidates: List[dict],
    drop_unnamed: bool,
    include_closed: bool,
) -> List[dict]:
    out = []
    for c in candidates:
        if drop_unnamed:
            has_name = any([
                c.get("denomination"),
                c.get("denomination_usuelle_ul"),
                c.get("enseigne1"),
                c.get("enseigne2"),
                c.get("enseigne3"),
                c.get("denomination_ul"),
                c.get("sigle_ul"),
                c.get("nom_ul"),
                c.get("prenom_usuel_ul"),
            ])
            if not has_name:
                continue
        if not include_closed:
            if c.get("etat_admin") == "F":
                continue
        out.append(c)
    return out


def _stable_seed(value: str) -> int:
    base = value or ""
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# TF-IDF artifact building (extracted for caching)
# ---------------------------------------------------------------------------

def _build_tfidf_artifacts(
    candidates: List[dict],
    config: RetrievalConfigV1,
) -> tuple:
    """Build all TF-IDF indexes and return as a cacheable artifact bundle."""
    name_vec, name_mat, names = build_tfidf_index(
        candidates,
        name_mode=config.tfidf_name_mode,
        siren_siblings=config.siren_siblings,
    )
    char_vec, char_mat = (None, None)
    if names:
        char_vec, char_mat = build_char_tfidf_index(names)
    addr_vec, addr_mat = build_address_tfidf_index(candidates)
    return (name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat)


def _get_tfidf_artifacts(
    candidates: List[dict],
    config: RetrievalConfigV1,
    tfidf_cache: Dict[Tuple[str, str], tuple],
    cache_key: Tuple[str, str],
    persistent_cache: Optional["TfidfPersistentCache"] = None,
    partition_key: str = "",
    timer: Optional["PipelineTimer"] = None,
) -> tuple:
    """Get TF-IDF artifacts: in-memory cache -> persistent cache -> build."""
    # 1. In-memory cache (current run, shared within loc_key group)
    cached = tfidf_cache.get(cache_key)
    if cached and len(cached) >= 7:
        return cached

    # 2. Persistent cache (cross-run, on disk)
    if persistent_cache and partition_key:
        disk_cached = persistent_cache.get(partition_key)
        if disk_cached and len(disk_cached) >= 7:
            tfidf_cache[cache_key] = disk_cached
            return disk_cached

    # 3. Build from scratch
    if timer:
        with timer.stage("tfidf_fit"):
            artifacts = _build_tfidf_artifacts(candidates, config)
    else:
        artifacts = _build_tfidf_artifacts(candidates, config)

    # Populate both caches
    tfidf_cache[cache_key] = artifacts
    if persistent_cache and partition_key:
        persistent_cache.put(partition_key, artifacts)
    return artifacts


# ---------------------------------------------------------------------------
# Dense retrieval integration
# ---------------------------------------------------------------------------

def _dense_retrieval_indices(
    crm_name: str,
    candidates: List[dict],
    partition_key: str,
    dense_store: Optional["PartitionEmbeddingStore"],
    top_k: int,
    timer: Optional["PipelineTimer"] = None,
) -> List[int]:
    """Return candidate indices from dense (FAISS) retrieval. Empty if unavailable."""
    if dense_store is None:
        return []
    if not dense_store.has_embeddings(partition_key):
        return []

    try:
        from .dense_retrieval import encode_query
    except ImportError:
        return []

    if timer:
        with timer.stage("dense_encode"):
            query_vec = encode_query(crm_name)
    else:
        query_vec = encode_query(crm_name)

    if query_vec is None:
        return []

    if timer:
        with timer.stage("dense_search"):
            idx = dense_store.get_index(partition_key)
    else:
        idx = dense_store.get_index(partition_key)

    if idx is None:
        return []

    scores, indices = idx.search(query_vec, top_k)
    # Filter out -1 padding indices from FAISS
    return [int(i) for i in indices if i >= 0 and i < len(candidates)]


# ---------------------------------------------------------------------------
# EPCI Helpers (V4.0)
# ---------------------------------------------------------------------------

_EPCI_MAP: Dict[str, str] = {}
def get_epci_map() -> Dict[str, str]:
    global _EPCI_MAP
    if not _EPCI_MAP:
        import json
        from pathlib import Path
        epci_path = Path("C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/data/insee_to_epci.json")
        if epci_path.exists():
            try:
                with open(epci_path, "r", encoding="utf-8") as f:
                    _EPCI_MAP = json.load(f)
            except Exception:
                _EPCI_MAP = {}
    return _EPCI_MAP


_EPCI_TO_INSEE: Dict[str, List[str]] = {}
def get_epci_to_insee() -> Dict[str, List[str]]:
    global _EPCI_TO_INSEE
    if not _EPCI_TO_INSEE:
        epci_map = get_epci_map()
        for insee_code, epci_id in epci_map.items():
            _EPCI_TO_INSEE.setdefault(epci_id, []).append(insee_code)
    return _EPCI_TO_INSEE


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_candidate_pool(
    store: PartitionedCandidateStore,
    crm_row: dict,
    crm_pre: dict,
    config: RetrievalConfigV1,
    tfidf_cache: Dict[Tuple[str, str], tuple],
    gt_siret: str | None = None,
    *,
    persistent_cache: Optional["TfidfPersistentCache"] = None,
    dense_store: Optional["PartitionEmbeddingStore"] = None,
    timer: Optional["PipelineTimer"] = None,
    siren_global_index: Optional[Any] = None,
    siren_to_geo: Optional[Any] = None,
    partition_cache: Optional[Dict[Any, Any]] = None,
) -> CandidatePoolResult:
    """Build candidate pool with unified code-path for train and inference.

    New optional parameters (backward-compatible):
        persistent_cache: Disk-backed TF-IDF cache (P0a).
        dense_store: Pre-computed FAISS embeddings for hybrid retrieval (P1).
        timer: Pipeline timing instrumentation (P0c).
        siren_global_index: Route B SIREN global index (optional).
        siren_to_geo: Route B SIREN → geo mapping (optional).
    """
    result = CandidatePoolResult(candidates=[])

    insee = crm_row.get("insee") or crm_row.get("crm_insee")
    postcode = crm_row.get("postcode") or crm_row.get("crm_cp")
    crm_name = crm_row.get("crm_name", "")
    crm_address = crm_pre.get("crm_addr", "")
    partition_key = f"{insee or ''}_{postcode or ''}"

    gt_norm = str(gt_siret).zfill(14) if gt_siret else None

    # === ROUTE B: SIREN-first retrieval (only if global SIREN index present, expansion mode excluded) ===
    if siren_global_index is not None and not config.siren_expansion_enabled:
        crm_name_bag = crm_pre.get("crm_name", "")

        # Phase 1: Query SIREN global index
        top_sirens = siren_global_index.query(crm_name_bag, top_k=config.siren_top_k)

        # Phase 2: Retrieve SIRETs for top SIRENs, geo-aware
        candidates: List[Dict[str, Any]] = []
        seen_sirets: set[str] = set()

        for siren, _ in top_sirens:
            siren_locs = siren_to_geo.get_locations(siren)
            if not siren_locs:
                continue

            # Prioritize locations: INSEE match > postcode match > all others
            sorted_locs = sorted(
                siren_locs,
                key=lambda loc: (loc[0] != insee, loc[1] != postcode)
            )

            loaded_for_siren = 0
            for loc_insee, loc_postcode in sorted_locs:
                if loaded_for_siren >= config.max_sirets_per_siren:
                    break

                try:
                    partition = store.load_by_insee(loc_insee)
                except Exception:
                    continue

                for cand in partition:
                    if (str(cand.get("siren") or "") == siren and
                        str(cand.get("siret") or "") not in seen_sirets):

                        # Apply filters
                        if config.drop_unnamed and not any([
                            cand.get("denomination"),
                            cand.get("denomination_usuelle_ul"),
                            cand.get("enseigne1"),
                            cand.get("enseigne2"),
                            cand.get("enseigne3"),
                            cand.get("denomination_ul"),
                            cand.get("sigle_ul"),
                            cand.get("nom_ul"),
                            cand.get("prenom_usuel_ul"),
                        ]):
                            continue

                        if not config.include_closed and cand.get("etat_admin") == "F":
                            continue

                        candidates.append(cand)
                        seen_sirets.add(str(cand.get("siret") or ""))
                        loaded_for_siren += 1

        # Phase 3: Compute IDF map
        candidates_dict = {
            str(c.get("siret") or ""): c
            for c in candidates
            if c.get("siret")
        }
        idf_map, default_idf = compute_name_idf_map(candidates_dict)

        result.pool_sizes["base"] = len(candidates)
        result.pool_sizes["filtered"] = len(candidates)
        result.pool_sizes["prefilter"] = len(candidates)
        result.pool_sizes["tfidf"] = len(candidates)
        result.candidates = candidates
        result.idf_map = idf_map
        result.default_idf = float(default_idf)
        result.gt_in_base_pool = _check_siret_in_list(candidates, gt_norm)
        result.gt_in_filtered_pool = _check_siret_in_list(candidates, gt_norm)
        result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)

        return result

    # === V7: Standard geo-partitioned retrieval (default) ===
    # Check partition cache for pre-computed filtered/deduped pool and indexes
    cache_key = (partition_key, config.include_closed, config.drop_unnamed)
    cache_hit = False
    
    if partition_cache is not None and cache_key in partition_cache:
        # Move key to end for LRU cache behavior
        if hasattr(partition_cache, "move_to_end"):
            try:
                partition_cache.move_to_end(cache_key)
            except Exception:
                pass
        
        cached_data = partition_cache[cache_key]
        pool_list = cached_data["pool_list"]
        pool = cached_data.get("pool")
        if pool is None:
            pool = {str(c.get("siret") or "").zfill(14): c for c in pool_list if c.get("siret")}
        idf_map = cached_data["idf_map"]
        default_idf = cached_data["default_idf"]
        addr_index = cached_data["addr_index"]
        num_index = cached_data["num_index"]
        tfidf_artifacts = cached_data["tfidf_artifacts"]
        
        result.pool_sizes["base"] = cached_data["pool_sizes"]["base"]
        result.pool_sizes["filtered"] = cached_data["pool_sizes"]["filtered"]
        result.gt_in_base_pool = _check_siret_in_list(pool_list, gt_norm)
        result.gt_in_filtered_pool = gt_norm in _get_siret_set(pool_list) if gt_norm else False
        result.idf_map = idf_map
        result.default_idf = default_idf
        
        if result.gt_in_base_pool and not result.gt_in_filtered_pool:
            if not config.include_closed:
                result.loss_reason = "FILTERED_CLOSED"
            elif config.drop_unnamed:
                result.loss_reason = "FILTERED_UNNAMED"
            else:
                result.loss_reason = "FILTERED_OTHER"
                
        cache_hit = True
    else:
        # Step 1: Load base candidates (strict insee_then_postcode + mega policy)
        if timer:
            with timer.stage("partition_load"):
                base_candidates = store.load_by_insee_then_postcode(
                    insee,
                    postcode,
                    mega_insee_max_rows=config.mega_insee_max_rows,
                    mega_insee_policy=config.mega_insee_policy,
                )
        else:
            base_candidates = store.load_by_insee_then_postcode(
                insee,
                postcode,
                mega_insee_max_rows=config.mega_insee_max_rows,
                mega_insee_policy=config.mega_insee_policy,
            )

        result.pool_sizes["base"] = len(base_candidates)
        result.gt_in_base_pool = _check_siret_in_list(base_candidates, gt_norm)

        # Step 2: Apply filters + dedupe
        filtered = _apply_filters(base_candidates, config.drop_unnamed, config.include_closed)
        pool = dedupe_candidates(filtered)
        pool_list = list(pool.values())

        # Version 4.0 Cascade Gate: Strict Local Check + Bypass
        from .naming import build_candidate_names, normalize_name
        from .features import jaro_sim
        
        crm_name_norm = normalize_name(crm_name or "")
        bypass_found = False
        if pool_list and crm_name_norm:
            for cand in pool_list:
                cand_names = build_candidate_names(cand)
                best_sim = 0.0
                for nm in cand_names:
                    sim = jaro_sim(crm_name_norm, nm.text)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim >= 0.90:
                    bypass_found = True
                    break

        if not bypass_found and len(pool_list) < 100:
            # Bypass failed and local pool is scarce -> Run Stream 1 (EPCI Fallback) and Stream 2 (Departmental Acronym)
            epci_candidates = []
            epci_map = get_epci_map()
            if insee and insee in epci_map:
                epci_id = epci_map[insee]
                epci_to_insee = get_epci_to_insee()
                neighbors = epci_to_insee.get(epci_id, [])
                for nb in neighbors:
                    if nb != insee:
                        try:
                            nb_candidates = store.load_by_insee(nb)
                            nb_filtered = _apply_filters(nb_candidates, config.drop_unnamed, config.include_closed)
                            epci_candidates.extend(nb_filtered)
                        except Exception:
                            pass

            acronym_candidates = []
            dept_prefix = postcode[:2] if postcode else None
            if dept_prefix:
                acro = None
                if crm_name:
                    import re
                    clean_name = re.sub(r"[^\w\s]", " ", crm_name)
                    words = clean_name.split()
                    EXCLUDED_ACRONYMS = {
                        "SA", "SAS", "SARL", "SCI", "EURL", "SNC", "SELARL", "SCP", "ASS", "ASSOC", "ASSOCIATION", 
                        "ETS", "STE", "SOCIETE", "GROUP", "GROUPE", "COOP", "GIE", "GAEC", "EARL"
                    }
                    for w in words:
                        if len(w) >= 2 and len(w) <= 6 and w.isupper() and w.isalpha():
                            if w not in EXCLUDED_ACRONYMS:
                                acro = w
                                break
                if acro:
                    try:
                        import pyarrow.dataset as ds
                        from pathlib import Path
                        # Only query partitions within target department to avoid scanning the entire country
                        cp_dir = Path("C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/data/candidates_v7_all/cp")
                        dept_postcodes = []
                        if cp_dir.exists():
                            for p_dir in cp_dir.iterdir():
                                if p_dir.is_dir() and p_dir.name.startswith("postcode="):
                                    pc_val = p_dir.name.split("=", 1)[1]
                                    if pc_val.startswith(dept_prefix):
                                        dept_postcodes.append(pc_val)
                        
                        if dept_postcodes:
                            filt = (ds.field("sigle_ul") == acro) & (ds.field("postcode").isin(dept_postcodes))
                        else:
                            filt = (ds.field("sigle_ul") == acro)
                            
                        table = store._dataset_cp.to_table(
                            filter=filt,
                            columns=store._columns_cp
                        )
                        acro_rows = store._coerce_candidate_types(table.to_pylist())
                        for r in acro_rows:
                            cp = r.get("postcode")
                            if cp and str(cp).startswith(dept_prefix):
                                if config.drop_unnamed and not any([
                                    r.get("denomination"),
                                    r.get("denomination_usuelle_ul"),
                                    r.get("enseigne1"),
                                    r.get("enseigne2"),
                                    r.get("enseigne3"),
                                    r.get("denomination_ul"),
                                    r.get("sigle_ul"),
                                    r.get("nom_ul"),
                                    r.get("prenom_usuel_ul"),
                                ]):
                                    continue
                                if not config.include_closed and r.get("etat_admin") == "F":
                                    continue
                                acronym_candidates.append(r)
                    except Exception:
                        pass

            if epci_candidates or acronym_candidates:
                filtered = filtered + epci_candidates + acronym_candidates
                pool = dedupe_candidates(filtered)
                pool_list = list(pool.values())

        result.pool_sizes["filtered"] = len(pool_list)
        result.gt_in_filtered_pool = gt_norm in _get_siret_set(pool_list) if gt_norm else False

        candidates_dict = {str(c.get("siret") or ""): c for c in pool_list if c.get("siret")}
        idf_map = {}
        default_idf = 0.0
        if candidates_dict:
            idf_map_raw, default_idf_raw = compute_name_idf_map(candidates_dict)
            idf_map = idf_map_raw
            default_idf = float(default_idf_raw)
            result.idf_map = idf_map
            result.default_idf = default_idf

        if result.gt_in_base_pool and not result.gt_in_filtered_pool:
            if not config.include_closed:
                result.loss_reason = "FILTERED_CLOSED"
            elif config.drop_unnamed:
                result.loss_reason = "FILTERED_UNNAMED"
            else:
                result.loss_reason = "FILTERED_OTHER"

        # Step 3: Universal rescue whitelist (addr_hash + numeric tokens)
        addr_index = {}
        num_index = {}
        if pool_list:
            addr_index = build_address_hash_index(pool_list)
            num_index = build_numeric_token_index(pool_list)

        # Build TF-IDF artifacts to cache
        tfidf_artifacts = None
        if config.sparse_retrieval_enabled and pool_list:
            tfidf_artifacts = _get_tfidf_artifacts(
                pool_list, config, tfidf_cache, ("main", partition_key),
                persistent_cache=persistent_cache,
                partition_key=partition_key,
                timer=timer,
            )

        # Save to partition cache
        if partition_cache is not None:
            partition_cache[cache_key] = {
                "pool_list": pool_list,
                "pool": pool,
                "idf_map": idf_map,
                "default_idf": default_idf,
                "addr_index": addr_index,
                "num_index": num_index,
                "tfidf_artifacts": tfidf_artifacts,
                "pool_sizes": {
                    "base": result.pool_sizes.get("base", 0),
                    "filtered": result.pool_sizes.get("filtered", 0),
                }
            }
            # Limit cache size to 128 to prevent memory exhaustion (MemoryError) while maintaining sub-second query speed
            if len(partition_cache) > 128:
                try:
                    partition_cache.popitem(last=False)
                except TypeError:
                    try:
                        partition_cache.popitem()
                    except Exception:
                        first_key = next(iter(partition_cache))
                        partition_cache.pop(first_key, None)

    # Step 3: Universal rescue whitelist (addr_hash + numeric tokens) - query specific
    whitelisted_sirets: set[str] = set()
    if pool_list:
        addr_h = address_hash(
            crm_pre.get("crm_street_num"),
            crm_pre.get("crm_street_name"),
        )
        if config.rescue_addr_hash and addr_h and addr_index:
            for idx in addr_index.get(addr_h, []):
                if idx < len(pool_list):
                    siret = str(pool_list[idx].get("siret") or "")
                    if siret:
                        whitelisted_sirets.add(siret)

        numeric_tokens = extract_numeric_tokens(crm_name)
        if config.rescue_numeric_tokens and numeric_tokens and num_index:
            for token in numeric_tokens:
                for idx in num_index.get(token, []):
                    if idx < len(pool_list):
                        siret = str(pool_list[idx].get("siret") or "")
                        if siret:
                            whitelisted_sirets.add(siret)

    # Step 4: Prefilter — hybrid sparse (TF-IDF) + dense (FAISS) + rescue
    candidates = pool_list
    if config.prefilter_k and len(candidates) > config.prefilter_k:
        tfidf_cache_key = ("main", f"{insee}_{postcode}")

        # 4a. Sparse retrieval (TF-IDF) — with persistent cache
        sparse_idx: List[int] = []
        if config.sparse_retrieval_enabled:
            if tfidf_artifacts is None:
                tfidf_artifacts = _get_tfidf_artifacts(
                    candidates, config, tfidf_cache, tfidf_cache_key,
                    persistent_cache=persistent_cache,
                    partition_key=partition_key,
                    timer=timer,
                )
            name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat = tfidf_artifacts

            name_idx: List[int] = []
            if name_vec is not None and name_mat is not None:
                if timer:
                    with timer.stage("tfidf_query"):
                        name_idx = prefilter_candidates_tfidf(
                            crm_name, name_vec, name_mat, config.prefilter_k,
                            cand_names=names, char_top_k=config.char_top_k,
                            char_vectorizer=char_vec, char_matrix=char_mat,
                        )
                else:
                    name_idx = prefilter_candidates_tfidf(
                        crm_name, name_vec, name_mat, config.prefilter_k,
                        cand_names=names, char_top_k=config.char_top_k,
                        char_vectorizer=char_vec, char_matrix=char_mat,
                    )

            addr_idx: List[int] = []
            if addr_vec is not None and addr_mat is not None:
                addr_idx = prefilter_candidates_address_tfidf(
                    crm_address, addr_vec, addr_mat, config.prefilter_k,
                )

            sparse_idx = list(dict.fromkeys(name_idx + addr_idx))

        # 4b. Dense retrieval (FAISS) — if enabled and available
        dense_idx: List[int] = []
        if config.dense_retrieval_enabled and dense_store is not None:
            dense_idx = _dense_retrieval_indices(
                crm_name, candidates, partition_key,
                dense_store, config.dense_top_k, timer,
            )

        # 4c. Union: sparse + dense (deduped, order-preserved)
        combined_idx = list(dict.fromkeys(sparse_idx + dense_idx))
        if config.prefilter_union_cap:
            combined_idx = combined_idx[: config.prefilter_union_cap]

        if combined_idx:
            prefilter_cands = [candidates[i] for i in combined_idx if i < len(candidates)]
            prefilter_sirets = {str(c.get("siret")) for c in prefilter_cands if c.get("siret")}

            # Add rescue whitelist candidates not already in the pool
            final_cands = list(prefilter_cands)
            for cand in pool.values():
                siret = str(cand.get("siret") or "")
                if siret in whitelisted_sirets and siret not in prefilter_sirets:
                    final_cands.append(cand)

            if len(final_cands) >= config.min_candidates:
                candidates = final_cands
            else:
                final_sirets = {str(c.get("siret") or "") for c in final_cands if c.get("siret")}
                remaining = [c for c in candidates if str(c.get("siret") or "") not in final_sirets]
                needed = min(config.min_candidates, config.prefilter_k) - len(final_cands)
                if needed > 0 and remaining:
                    seed_val = str(crm_row.get("crm_id") or "") or crm_name
                    rng = random.Random(_stable_seed(seed_val))
                    random_extra = rng.sample(remaining, min(needed, len(remaining)))
                    candidates = final_cands + random_extra
                else:
                    candidates = final_cands

    result.pool_sizes["prefilter"] = len(candidates)
    # Backward compatibility for existing reports/scripts expecting "tfidf" key
    result.pool_sizes["tfidf"] = len(candidates)
    result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)

    if result.gt_in_filtered_pool and not result.gt_in_tfidf_pool and not result.loss_reason:
        result.loss_reason = "PRUNED_BY_TFIDF" if config.sparse_retrieval_enabled else "PRUNED_BY_PREFILTER"

    if not result.gt_in_base_pool and not result.loss_reason:
        result.loss_reason = "NOT_IN_PARTITION"

    # === Step 5: SIREN expansion (local + cross-partition) ===
    if config.siren_expansion_enabled and siren_to_geo is not None and candidates:
        pool_before = len(candidates)
        seed_sirens = sorted({c.get("siren") for c in candidates if c.get("siren")})
        seen_sirets = {c["siret"] for c in candidates if c.get("siret")}
        crm_insee = crm_pre.get("insee") or ""
        crm_cp = crm_pre.get("postcode") or ""

        exp_insee: list[dict] = []
        exp_cp: list[dict] = []
        exp_other: list[dict] = []
        loaded_partitions: dict[str, list] = {}  # cache local per INSEE

        def _load(insee_code: str) -> list:
            if insee_code not in loaded_partitions:
                loaded_partitions[insee_code] = store.load_by_insee(insee_code)
            return loaded_partitions[insee_code]

        for siren in seed_sirens:
            locs = siren_to_geo.get_locations(siren)
            
            epci_map = get_epci_map()
            crm_epci = epci_map.get(crm_insee) if crm_insee else None
            
            def geo_proximity_key(loc):
                loc_insee, loc_cp = loc
                if loc_insee == crm_insee:
                    return 0  # Commune
                if crm_epci and epci_map.get(loc_insee) == crm_epci:
                    return 1  # EPCI
                if loc_cp and crm_cp and str(loc_cp)[:2] == str(crm_cp)[:2]:
                    return 2  # Department
                return 3  # National (others)
                
            sorted_locs = sorted(locs, key=geo_proximity_key)
            added_siren = 0

            for insee_code, _ in sorted_locs:
                if added_siren >= config.max_sirets_per_siren:
                    break
                for c in _load(insee_code):
                    siret = c.get("siret")
                    if c.get("siren") != siren or siret in seen_sirets:
                        continue

                    # Apply business filters (consistent with Route B)
                    if config.drop_unnamed and not any([
                        c.get("denomination"),
                        c.get("denomination_usuelle_ul"),
                        c.get("enseigne1"),
                        c.get("enseigne2"),
                        c.get("enseigne3"),
                        c.get("denomination_ul"),
                        c.get("sigle_ul"),
                        c.get("nom_ul"),
                        c.get("prenom_usuel_ul"),
                    ]):
                        continue

                    if not config.include_closed and c.get("etat_admin") == "F":
                        continue

                    if insee_code == crm_insee:
                        exp_insee.append(c)
                    elif c.get("postcode") == crm_cp:
                        exp_cp.append(c)
                    else:
                        exp_other.append(c)
                    seen_sirets.add(siret)
                    added_siren += 1

        def _sort_key(c):
            return (c.get("etat_admin") == "F", not c.get("is_siege", False), c.get("siret", ""))

        expansion = (
            sorted(exp_insee, key=_sort_key)
            + sorted(exp_cp, key=_sort_key)
            + sorted(exp_other, key=_sort_key)
        )

        if expansion:
            candidates = candidates + expansion
            if len(candidates) > config.siren_expansion_pool_cap:
                candidates = candidates[:config.siren_expansion_pool_cap]

            # Recalculer IDF sur pool élargi — retourner dans CandidatePoolResult
            # NE PAS appeler set_global_name_idf_map() ici (le caller le fait)
            tmp = {c["siret"]: c for c in candidates if c.get("siret")}
            result.idf_map, result.default_idf = compute_name_idf_map(tmp)

        # Telémétrie
        result.pool_sizes.update({
            "prefilter_before_expansion": pool_before,
            "expansion_added_insee": len(exp_insee),
            "expansion_added_postcode": len(exp_cp),
            "expansion_added_cross": len(exp_other),
            "expanded_pool_final": len(candidates),
        })

        # Recalculate GT metrics after expansion
        result.gt_in_tfidf_pool = _check_siret_in_list(candidates, gt_norm)
        if result.gt_in_filtered_pool and not result.gt_in_tfidf_pool and result.loss_reason == "PRUNED_BY_TFIDF":
            # GT was recovered by expansion, clear loss reason
            result.loss_reason = None
    elif config.siren_expansion_enabled and siren_to_geo is None:
        logger = logging.getLogger(__name__)
        logger.warning("siren_expansion_enabled=True but siren_to_geo not loaded — skipping expansion")

    attach_address_density(candidates)
    result.prefilter_sirets = [str(c.get("siret") or "") for c in candidates if c.get("siret")]
    result.candidates = candidates
    return result


def inject_gt_if_missing(
    result: CandidatePoolResult,
    gt_candidate: dict | None,
) -> CandidatePoolResult:
    if gt_candidate is None:
        return result
    gt_siret = str(gt_candidate.get("siret", "")).zfill(14)
    if not gt_siret:
        return result
    existing_sirets = _get_siret_set(result.candidates)
    if gt_siret not in existing_sirets:
        result.candidates.append(gt_candidate)
        result.gt_was_injected = True
    return result


__all__ = [
    "CandidatePoolResult",
    "build_candidate_pool",
    "inject_gt_if_missing",
]
