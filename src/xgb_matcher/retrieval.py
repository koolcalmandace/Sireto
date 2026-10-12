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
import re
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
    normalize_code,
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
# Semantic vector prototypes for public sector / municipal / social classification
# ---------------------------------------------------------------------------

_PUBLIC_SECTOR_PROTOTYPES = [
    # 1. Enseignement & Ecoles
    "ecole elementaire primaire maternelle college lycee groupe scolaire enseignement classe clis ulis rased internat externat pedagogique medico pedagogique segpa erea cantine scolaire periscolaire",
    # 2. Collectivités locales & Communes
    "mairie commune hotel de ville services techniques batiment municipal municipale collectivite communale presbytere cimetiere centre technique municipal salle des fetes salle polyvalente foyer rural ateliers municipaux assainissement",
    # 3. Équipements sportifs & Loisirs communaux
    "gymnase stade piscine municipale complexe sportif salle des fetes dojo terrain sports gymnases hall des sports",
    # 4. Culture & Social communal
    "espace culturel mediatheque bibliotheque centre social ccas cias maison de quartier action sociale communal communale tiers lieu municipal centre socio culturel foyer rural",
    # 5. Petite enfance & Périscolaire
    "halte garderie creche centre aere loisirs enfance jeunesse cantine scolaire periscolaire relais petite enfance rpe",
    # 6. Secours, Sécurité & Hôpitaux publics
    "police municipale centre de secours pompiers collectivite territoriale sdis hopital centre hospitalier ch ehpad chu samu",
    # 7. Intercommunalités & Territoires (Grand Reims, communautés de communes, syndicats mixtes...)
    "communaute urbaine grand reims communaute de communes communaute d agglomeration metropole epci syndicat mixte syndicats mixtes sm e collectivites pays canton intercommunal territorial agglomeration sivom sivu petr poles territoriaux",
    # 8. Logement social & Bailleurs publics (OPH, HLM...)
    "hlm office public de l habitat oph bailleur social office public d amenagement logement social cite jardin residence sociale habitat public",
    # 9. Emploi & Organismes sociaux (Pôle Emploi / France Travail, CAF, CPAM...)
    "pole emploi france travail anpe agence pour l emploi demandeur d emploi caisse d allocations familiales caf cpam carsat urssaf secu securite sociale msa cnamts cnav",
    # 10. Départements & Régions (Conseils départementaux, régionaux, voirie...)
    "departement departements conseil departemental conseils departementaux conseil regional conseils regionaux region regions hotel du departement services departementaux college public voirie departementale circonspection d action sociale",
    # 11. Santé mentale ambulatoire, Handicap & Médico-social (CATTP, CMP, APAJH, ADAPA...)
    "cattp cmp cmpp hopital de jour psychiatrique pedopsychiatrie centre medico psychologique csapa caarud fam mas ehpad ime itep esat sessad foyer d accueil medicalise foyer de vie adapa apajh adapei iesda unapei codes medico pedagogique pedagogique",
    # 12. Enseignement Supérieur, Recherche & Grands corps (CNAM...)
    "cnam conservatoire national des arts et metiers universite faculte cfa epe rectorat academie inspe iut crous cnrs inserm inrae",
    # 13. Direction territoriale, Action éducative & Judiciaire (PJJ, DTPJJ, SDJES...)
    "direction territoriale protection judiciaire jeunesse pjj dtpjj sdjes dsden rectorat education nationale",
]
_PUBLIC_VECTORIZER = None
_PUBLIC_PROTO_MAT = None


_PRIVATE_BUSINESS_TERMS = {
    "sarl", "sas", "eurl", "snc", "sci", "garage", "boulangerie", "boucherie",
    "restaurant", "coiffure", "bar", "brasserie", "cafe", "boutique",
    "salon", "auto", "imprimerie", "carrosserie", "optique", "pharmacie", "press",
    "pressing", "commercial", "societe", "entreprise", "holding",
}

_PUBLIC_COMPOUNDS = {
    "hotel de ville", "hotel du departement", "hotel de region", "hotel de communaute",
    "syndicat mixte", "syndicats mixtes", "e collectivites", "e collectivite",
    "communaute de communes", "communaute d agglomeration", "communaute urbaine",
    "conseil departemental", "conseil regional", "pole metropolitain",
    "centre hospitalier", "finances publiques", "tresorerie", "action sociale",
    "centre socio culturel", "centre socioculturel", "centre technique municipal",
    "services techniques", "ateliers municipaux", "police municipale",
    "salle des fetes", "salle polyvalente", "foyer rural",
    "protection judiciaire de la jeunesse", "protection judiciaire jeunesse",
    "direction territoriale", "jeunesse et sports", "direction des services departementaux",
}


_PUBLIC_ACRONYMS = {
    "hlm", "oph", "cattp", "cmp", "cmpp", "ccas", "cias", "sdis", "sdiss",
    "caf", "cpam", "carsat", "urssaf", "cnam", "crous", "epci",
    "sm", "sivom", "sivu", "petr", "epe", "ogec", "cfa", "mfr", "csapa", "caarud",
    "adapa", "apajh", "adapei", "iesda", "unapei", "esat", "ime", "itep", "sessad",
    "mas", "fam", "pjj", "dtpjj", "sdjes", "dsden", "rectorat", "codes",
    "presbytere", "cimetiere", "internat", "externat",
}


def _is_public_sector_query(text: str) -> bool:
    """Classify via semantic vector prototype if query relates to public/municipal/social services."""
    global _PUBLIC_VECTORIZER, _PUBLIC_PROTO_MAT
    if not text:
        return False

    # Normalize punctuation into whitespace and de-concatenate alphanumeric tokens (e.g. ADAPA01 -> ADAPA 01)
    clean_text = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    clean_text = re.sub(r"([a-z]+)(\d+)", r"\1 \2", clean_text)
    clean_text = re.sub(r"(\d+)([a-z]+)", r"\1 \2", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    tokens = set(clean_text.split())

    # Protect compounds like 'hotel de ville' from being discarded by generic 'hotel'
    is_compound_public = any(comp in clean_text for comp in _PUBLIC_COMPOUNDS)
    if not is_compound_public and (tokens & _PRIVATE_BUSINESS_TERMS):
        return False

    if tokens & _PUBLIC_ACRONYMS:
        return True

    if _PUBLIC_VECTORIZER is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        stopwords = ["de", "du", "des", "la", "le", "les", "et", "en", "un", "une", "d", "l", "au", "aux"]
        _PUBLIC_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", stop_words=stopwords)
        _PUBLIC_PROTO_MAT = _PUBLIC_VECTORIZER.fit_transform(_PUBLIC_SECTOR_PROTOTYPES)
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        q_vec = _PUBLIC_VECTORIZER.transform([clean_text])
        sims = cosine_similarity(q_vec, _PUBLIC_PROTO_MAT)[0]
        return float(max(sims)) >= 0.12
    except Exception:
        return False



# ---------------------------------------------------------------------------
# TF-IDF artifact building (extracted for caching)
# ---------------------------------------------------------------------------

def _build_tfidf_artifacts(
    candidates: List[dict],
    config: RetrievalConfigV1,
) -> tuple:
    """Build all TF-IDF indexes and return as a cacheable artifact bundle.

    Returns (name_vec, name_mat, None, char_vec, char_mat, addr_vec, addr_mat).
    The raw candidate name strings are deliberately discarded (set to None) after
    fitting char_vec to free ~2.5-3.5 GB of RAM across cached communes.
    """
    name_vec, name_mat, names = build_tfidf_index(
        candidates,
        name_mode=config.tfidf_name_mode,
        siren_siblings=config.siren_siblings,
    )
    char_vec, char_mat = (None, None)
    if names:
        char_vec, char_mat = build_char_tfidf_index(names)
    addr_vec, addr_mat = build_address_tfidf_index(candidates)
    return (name_vec, name_mat, None, char_vec, char_mat, addr_vec, addr_mat)


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
        if hasattr(tfidf_cache, "move_to_end"):
            tfidf_cache.move_to_end(cache_key)
        return cached

    # 2. Persistent cache (cross-run, on disk)
    if persistent_cache and partition_key:
        disk_cached = persistent_cache.get(partition_key)
        if disk_cached and len(disk_cached) >= 7:
            tfidf_cache[cache_key] = disk_cached
            if hasattr(tfidf_cache, "move_to_end"):
                tfidf_cache.move_to_end(cache_key)
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
    compute_idf_map: bool = True,
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

    code_insee = normalize_code(insee)
    code_cp = normalize_code(postcode)
    if code_insee:
        partition_key = f"insee_{code_insee}"
    elif code_cp:
        partition_key = f"cp_{code_cp}"
    else:
        partition_key = "geo_unknown"

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

        # Phase 3: Compute IDF map (optional for recall evaluation)
        if compute_idf_map:
            candidates_dict = {
                str(c.get("siret") or ""): c
                for c in candidates
                if c.get("siret")
            }
            idf_map, default_idf = compute_name_idf_map(candidates_dict)
            result.idf_map = idf_map
            result.default_idf = float(default_idf)

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
    # Step 1: Load base candidates (strict geo resolution: INSEE -> CP child discovery -> empty)
    crm_id = str(crm_row.get("crm_id") or crm_row.get("ID") or crm_row.get("id") or "<unknown>")
    crm_street = (crm_pre.get("crm_street_name") if crm_pre else None) or crm_row.get("crm_address") or ""
    if timer:
        with timer.stage("partition_load"):
            base_candidates = store.load_with_geo_resolution(
                insee,
                postcode,
                crm_id=crm_id,
                mega_insee_max_rows=config.mega_insee_max_rows,
                mega_insee_policy=config.mega_insee_policy,
                street_name=crm_street,
                logger=_logger,
            )
    else:
        base_candidates = store.load_with_geo_resolution(
            insee,
            postcode,
            crm_id=crm_id,
            mega_insee_max_rows=config.mega_insee_max_rows,
            mega_insee_policy=config.mega_insee_policy,
            street_name=crm_street,
            logger=_logger,
        )

    result.pool_sizes["base"] = len(base_candidates)
    result.gt_in_base_pool = _check_siret_in_list(base_candidates, gt_norm)

    # Step 2: Apply filters + dedupe
    filtered = _apply_filters(base_candidates, config.drop_unnamed, config.include_closed)
    pool = dedupe_candidates(filtered)
    pool_list = list(pool.values())

    result.pool_sizes["filtered"] = len(pool_list)
    result.gt_in_filtered_pool = _check_siret_in_list(pool_list, gt_norm)

    if compute_idf_map:
        candidates_dict = {str(c.get("siret") or ""): c for c in pool_list if c.get("siret")}
        if candidates_dict:
            idf_map, default_idf = compute_name_idf_map(candidates_dict)
            result.idf_map = idf_map
            result.default_idf = float(default_idf)

    if result.gt_in_base_pool and not result.gt_in_filtered_pool:
        if not config.include_closed:
            result.loss_reason = "FILTERED_CLOSED"
        elif config.drop_unnamed:
            result.loss_reason = "FILTERED_UNNAMED"
        else:
            result.loss_reason = "FILTERED_OTHER"

    # Step 3: Universal rescue whitelist (addr_hash with density guard + numeric tokens + public expansion)
    whitelisted_sirets: set[str] = set()
    sidecar_rescued_sirets: set[str] = set()
    if pool_list:
        # 3a. Address rescue with density guard
        # Universal density cap: 10 if exact street number is present, 5 otherwise
        addr_index = build_address_hash_index(pool_list)
        num_index = build_numeric_token_index(pool_list)

        addr_h = address_hash(
            crm_pre.get("crm_street_num"),
            crm_pre.get("crm_street_name"),
        )
        if config.rescue_addr_hash and addr_h and addr_index:
            matching_indices = addr_index.get(addr_h, [])
            # Density Guard: Only rescue if co-tenants <= effective cap
            # Allows up to 10 when exact street number is provided (universal multi-tenant support)
            # Restricts to 5 when street number is missing (protects against entire street explosion)
            has_street_num = bool(crm_pre.get("crm_street_num"))
            eff_cap = config.address_density_cap if has_street_num else min(5, config.address_density_cap)
            if matching_indices and len(matching_indices) <= eff_cap:
                for idx in matching_indices:
                    if idx < len(pool_list):
                        siret = str(pool_list[idx].get("siret") or "")
                        if siret:
                            whitelisted_sirets.add(siret)

        # 3b. Numeric token rescue
        numeric_tokens = extract_numeric_tokens(crm_name)
        if config.rescue_numeric_tokens and numeric_tokens and num_index:
            for token in numeric_tokens:
                for idx in num_index.get(token, []):
                    if idx < len(pool_list):
                        siret = str(pool_list[idx].get("siret") or "")
                        if siret:
                            whitelisted_sirets.add(siret)

        # 3c. Public & social sector expansion (secondary establishments of local commune/public authorities/social landlords)
        if config.rescue_public_expansion and _is_public_sector_query(crm_name):
            # 1. Identify local Commune SIREN(s) in the locality pool
            commune_sirens: set[str] = set()
            for cand in pool_list:
                cj = str(cand.get("cj_ul") or "")
                denom = str(cand.get("denomination") or cand.get("denomination_ul") or "").upper()
                if cj == "7210" or denom.startswith(("COMMUNE DE ", "COMMUNE D ", "VILLE DE ", "MAIRIE DE ", "MAIRIE D ")):
                    s = str(cand.get("siren") or "")
                    if s:
                        commune_sirens.add(s)

            for cand in pool_list:
                cj = str(cand.get("cj_ul") or "")
                siren = str(cand.get("siren") or "")

                # A. Any secondary establishment of the local Commune (schools, gymnasiums, municipal annexes, presbytères):
                is_commune_child = (cj == "7210") or (bool(siren) and siren in commune_sirens)

                # B. Public sector, social security & social housing legal entities:
                # - cj_ul starting with '7' (Collectivités territoriales, EPA, EPS, EPLE, administrations)
                # - cj_ul starting with '41' (Établissements publics locaux à caractère industriel ou commercial, ex: 4140 OPH)
                # - cj_ul starting with '84' or '85' (Sécurité Sociale, CAF, CPAM, URSSAF, CARSAT, régimes sociaux)
                # - cj_ul in ('9210', '9220', '9230') (Associations reconnues d'utilité publique, médico-sociales, enseignement)
                is_public_org = (
                    cj.startswith("7")
                    or cj.startswith("41")
                    or cj.startswith("84")
                    or cj.startswith("85")
                    or cj in ("9210", "9220", "9230")
                )

                if is_commune_child or is_public_org:
                    siret = str(cand.get("siret") or "")
                    if siret:
                        whitelisted_sirets.add(siret)

        # 3d. Semantic Sidecar rescue (Brands & Acronyms)
        if getattr(config, "rescue_sidecar", False):
            try:
                from .sidecar import run_entity_sidecar
                sidecar_sirets = run_entity_sidecar(
                    crm_name,
                    crm_address,
                    pool_list,
                    crm_insee=insee,
                    store=store,
                )
                if sidecar_sirets:
                    whitelisted_sirets.update(sidecar_sirets)
                    sidecar_rescued_sirets.update(sidecar_sirets)
            except Exception as e:
                _logger.debug("Sidecar rescue skipped or error: %s", e)


    # Step 4: Prefilter — hybrid sparse (TF-IDF) + dense (FAISS) + rescue
    candidates = pool_list
    if config.prefilter_k and len(candidates) > config.prefilter_k:
        cache_key = ("main", partition_key)

        # 4a. Sparse retrieval (TF-IDF) — with persistent cache
        sparse_idx: List[int] = []
        if config.sparse_retrieval_enabled:
            artifacts = _get_tfidf_artifacts(
                candidates, config, tfidf_cache, cache_key,
                persistent_cache=persistent_cache,
                partition_key=partition_key,
                timer=timer,
            )
            name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat = artifacts

            name_idx: List[int] = []
            if name_vec is not None and name_mat is not None:
                if timer:
                    with timer.stage("tfidf_query"):
                        name_idx = prefilter_candidates_tfidf(
                            crm_name, name_vec, name_mat, config.prefilter_k,
                            cand_names=None, char_top_k=config.char_top_k,
                            char_vectorizer=char_vec, char_matrix=char_mat,
                        )
                else:
                    name_idx = prefilter_candidates_tfidf(
                        crm_name, name_vec, name_mat, config.prefilter_k,
                        cand_names=None, char_top_k=config.char_top_k,
                        char_vectorizer=char_vec, char_matrix=char_mat,
                    )

            addr_idx: List[int] = []
            if addr_vec is not None and addr_mat is not None and crm_address:
                addr_idx = prefilter_candidates_address_tfidf(
                    crm_address, addr_vec, addr_mat, 50,
                )

            # Strict inclusion: keep 100% of name candidates (up to prefilter_k)
            # and append top-50 candidates by physical address concordance
            sparse_idx = list(dict.fromkeys(name_idx + addr_idx[:50]))

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
            effective_cap = max(config.prefilter_union_cap, config.prefilter_k + 50)
            combined_idx = combined_idx[: effective_cap]

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
            sorted_locs = sorted(locs, key=lambda loc: (loc[0] != crm_insee, loc[1] != crm_cp))
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

    if sidecar_rescued_sirets:
        for c in candidates:
            s = str(c.get("siret") or "").strip()
            if s in sidecar_rescued_sirets:
                c["_is_sidecar_match"] = 1.0

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
    if not _check_siret_in_list(result.candidates, gt_siret):
        result.candidates.append(gt_candidate)
        result.gt_was_injected = True
    return result


__all__ = [
    "CandidatePoolResult",
    "build_candidate_pool",
    "inject_gt_if_missing",
]
