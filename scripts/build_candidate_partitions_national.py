#!/usr/bin/env python3
"""
scripts/build_candidate_partitions_national.py
==============================================
Génération haute performance de la base partitionnée SIRENE Nationale (France entière ~35 000 communes).

Sorties :
  data/candidates_v8_national/
    insee/insee=XXXXX/*.parquet   (Partitions communales strictes)
    cp/postcode=XXXXX/*.parquet   (Partitions codes postaux)

Caractéristiques :
- Traitement par département optimisé via DuckDB C++ native COPY PARTITION_BY
- Jointure multi-threadée Etablissements + Unités Légales (Snapshot Septembre 2026)
- Padding strict des codes INSEE (5 caractères) et Postcode (5 caractères)
- Schéma 100% conforme au standard Pipe V7 / V8b (23 colonnes)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import duckdb
from tqdm import tqdm

DEFAULT_ETAB_PARQUET = Path("data/source_sirene_20260901/StockEtablissement_utf8_20260901.parquet")
DEFAULT_UL_PARQUET = Path("data/source_sirene_20260901/StockUniteLegale_utf8_20260901.parquet")
DEFAULT_OUT_DIR = Path("data/candidates_v8_national")
CHECKPOINT_FILE = Path(".checkpoints/checkpoint_sirene_national.json")


def load_completed_departments() -> set[str]:
    """Charge la liste des départements déjà finalisés depuis le checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("completed_departments", []))
        except Exception:
            pass
    return set()


def save_completed_department(dpt: str) -> None:
    """Enregistre un département terminé dans le checkpoint atomique."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_departments()
    completed.add(dpt)
    tmp_file = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump({"completed_departments": sorted(list(completed)), "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=2)
    tmp_file.replace(CHECKPOINT_FILE)


def get_department_list() -> List[str]:
    """Retourne la liste complète des départements français (Métropole + Corse + DROM-COM)."""
    dpts = [f"{i:02d}" for i in range(1, 96) if i != 20]
    dpts.extend(["2A", "2B", "971", "972", "973", "974", "976", "984", "986", "987", "988"])
    return sorted(dpts)


def build_national_partitions(
    etab_path: Path,
    ul_path: Path,
    output_dir: Path,
    max_dpts: int | None = None,
    clean_target: bool = False,
    threads: int = 8,
) -> None:
    start_time = time.time()
    output_insee = output_dir / "insee"
    output_cp = output_dir / "cp"

    if clean_target:
        for p in [output_insee, output_cp]:
            if p.exists():
                print(f"🧹 Nettoyage du répertoire cible : {p}")
                shutil.rmtree(p)

    output_insee.mkdir(parents=True, exist_ok=True)
    output_cp.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 GÉNÉRATION DU STORE SIRENE NATIONAL PARTITIONNÉ — CANDIDATES V8")
    print("=" * 80)
    print(f"  Source Etablissements : {etab_path}")
    print(f"  Source Unités Légales : {ul_path}")
    print(f"  Répertoire de sortie  : {output_dir}")
    print(f"  Threads alloués       : {threads}")
    print("=" * 80)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads = {threads};")
    con.execute("PRAGMA preserve_insertion_order = false;")

    dpt_list = get_department_list()
    if max_dpts:
        dpt_list = dpt_list[:max_dpts]

    completed_dpts = load_completed_departments() if not clean_target else set()
    pending_dpts = [d for d in dpt_list if d not in completed_dpts]

    print(f"Total départements : {len(dpt_list)} | Déjà finalisés : {len(completed_dpts)} | À traiter : {len(pending_dpts)}\n")

    for dpt in tqdm(pending_dpts, desc="Départements traités", unit="dpt"):
        # Requête SQL départementale unifiée avec partitionnement direct
        query_insee = f"""
            COPY (
                WITH filtered_etab AS (
                    SELECT
                        e.siret,
                        e.siren,
                        e.denominationUsuelleEtablissement,
                        e.enseigne1Etablissement AS enseigne1,
                        e.enseigne2Etablissement AS enseigne2,
                        e.enseigne3Etablissement AS enseigne3,
                        CASE 
                            WHEN lower(cast(e.etablissementSiege as varchar)) in ('true', '1', 'vrai', 'oui') THEN true 
                            ELSE false 
                        END AS etablissementSiege,
                        e.numeroVoieEtablissement AS numeroVoie,
                        e.typeVoieEtablissement AS typeVoie,
                        e.libelleVoieEtablissement AS libelleVoie,
                        e.complementAdresseEtablissement AS complementAdresse,
                        LPAD(TRIM(e.codePostalEtablissement), 5, '0') AS postcode,
                        e.libelleCommuneEtablissement AS city,
                        CASE 
                            WHEN length(TRIM(e.codeCommuneEtablissement)) < 5 AND TRIM(e.codeCommuneEtablissement) NOT LIKE '2A%' AND TRIM(e.codeCommuneEtablissement) NOT LIKE '2B%'
                            THEN LPAD(TRIM(e.codeCommuneEtablissement), 5, '0')
                            ELSE TRIM(e.codeCommuneEtablissement)
                        END AS insee,
                        e.etatAdministratifEtablissement AS etat_admin,
                        e.dateDernierTraitementEtablissement AS last_treatment_date
                    FROM read_parquet('{etab_path.as_posix()}') e
                    WHERE (
                        e.codeCommuneEtablissement LIKE '{dpt}%' 
                        OR LPAD(TRIM(e.codeCommuneEtablissement), 5, '0') LIKE '{dpt}%'
                        OR e.codePostalEtablissement LIKE '{dpt}%'
                    )
                    AND (
                        e.etatAdministratifEtablissement IS NULL 
                        OR e.etatAdministratifEtablissement != 'F' 
                        OR (e.etatAdministratifEtablissement = 'F' AND TRY_CAST(e.dateDebut AS DATE) >= DATE '2016-01-01')
                    )
                    AND e.siret IS NOT NULL AND length(e.siret) = 14
                )
                SELECT 
                    f.siret,
                    f.siren,
                    COALESCE(f.denominationUsuelleEtablissement, ul.denominationUniteLegale) AS denomination,
                    f.enseigne1,
                    f.enseigne2,
                    f.enseigne3,
                    f.etablissementSiege,
                    f.etablissementSiege AS is_siege,
                    f.numeroVoie,
                    f.typeVoie,
                    f.libelleVoie,
                    f.complementAdresse,
                    f.postcode,
                    f.city,
                    f.insee,
                    ul.categorieJuridiqueUniteLegale AS cj_ul,
                    f.etat_admin,
                    f.last_treatment_date,
                    ul.sigleUniteLegale AS sigle_ul,
                    ul.denominationUniteLegale AS denomination_ul,
                    TRIM(CONCAT_WS(' ',
                        ul.denominationUsuelle1UniteLegale,
                        ul.denominationUsuelle2UniteLegale,
                        ul.denominationUsuelle3UniteLegale
                    )) AS denomination_usuelle_ul,
                    ul.nomUniteLegale AS nom_ul,
                    ul.prenomUsuelUniteLegale AS prenom_usuel_ul,
                    '' AS pm_dirigeant_names
                FROM filtered_etab f
                LEFT JOIN read_parquet('{ul_path.as_posix()}') ul ON f.siren = ul.siren
                WHERE f.insee IS NOT NULL AND length(f.insee) >= 4
            ) TO '{output_insee.as_posix()}' (FORMAT PARQUET, PARTITION_BY (insee), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1);
        """

        query_cp = f"""
            COPY (
                WITH filtered_etab AS (
                    SELECT
                        e.siret,
                        e.siren,
                        e.denominationUsuelleEtablissement,
                        e.enseigne1Etablissement AS enseigne1,
                        e.enseigne2Etablissement AS enseigne2,
                        e.enseigne3Etablissement AS enseigne3,
                        CASE 
                            WHEN lower(cast(e.etablissementSiege as varchar)) in ('true', '1', 'vrai', 'oui') THEN true 
                            ELSE false 
                        END AS etablissementSiege,
                        e.numeroVoieEtablissement AS numeroVoie,
                        e.typeVoieEtablissement AS typeVoie,
                        e.libelleVoieEtablissement AS libelleVoie,
                        e.complementAdresseEtablissement AS complementAdresse,
                        LPAD(TRIM(e.codePostalEtablissement), 5, '0') AS postcode,
                        e.libelleCommuneEtablissement AS city,
                        CASE 
                            WHEN length(TRIM(e.codeCommuneEtablissement)) < 5 AND TRIM(e.codeCommuneEtablissement) NOT LIKE '2A%' AND TRIM(e.codeCommuneEtablissement) NOT LIKE '2B%'
                            THEN LPAD(TRIM(e.codeCommuneEtablissement), 5, '0')
                            ELSE TRIM(e.codeCommuneEtablissement)
                        END AS insee,
                        e.etatAdministratifEtablissement AS etat_admin,
                        e.dateDernierTraitementEtablissement AS last_treatment_date
                    FROM read_parquet('{etab_path.as_posix()}') e
                    WHERE (
                        e.codePostalEtablissement LIKE '{dpt}%'
                        OR LPAD(TRIM(e.codePostalEtablissement), 5, '0') LIKE '{dpt}%'
                    )
                    AND (
                        e.etatAdministratifEtablissement IS NULL 
                        OR e.etatAdministratifEtablissement != 'F' 
                        OR (e.etatAdministratifEtablissement = 'F' AND TRY_CAST(e.dateDebut AS DATE) >= DATE '2016-01-01')
                    )
                    AND e.siret IS NOT NULL AND length(e.siret) = 14
                )
                SELECT 
                    f.siret,
                    f.siren,
                    COALESCE(f.denominationUsuelleEtablissement, ul.denominationUniteLegale) AS denomination,
                    f.enseigne1,
                    f.enseigne2,
                    f.enseigne3,
                    f.etablissementSiege,
                    f.etablissementSiege AS is_siege,
                    f.numeroVoie,
                    f.typeVoie,
                    f.libelleVoie,
                    f.complementAdresse,
                    f.postcode,
                    f.city,
                    f.insee,
                    ul.categorieJuridiqueUniteLegale AS cj_ul,
                    f.etat_admin,
                    f.last_treatment_date,
                    ul.sigleUniteLegale AS sigle_ul,
                    ul.denominationUniteLegale AS denomination_ul,
                    TRIM(CONCAT_WS(' ',
                        ul.denominationUsuelle1UniteLegale,
                        ul.denominationUsuelle2UniteLegale,
                        ul.denominationUsuelle3UniteLegale
                    )) AS denomination_usuelle_ul,
                    ul.nomUniteLegale AS nom_ul,
                    ul.prenomUsuelUniteLegale AS prenom_usuel_ul,
                    '' AS pm_dirigeant_names
                FROM filtered_etab f
                LEFT JOIN read_parquet('{ul_path.as_posix()}') ul ON f.siren = ul.siren
                WHERE f.postcode IS NOT NULL AND length(f.postcode) = 5
            ) TO '{output_cp.as_posix()}' (FORMAT PARQUET, PARTITION_BY (postcode), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1);
        """

        try:
            con.execute(query_insee)
            con.execute(query_cp)
            save_completed_department(dpt)
        except Exception as err:
            print(f"⚠️ Erreur lors du traitement du département {dpt}: {err}")
            continue

    elapsed = time.time() - start_time
    total_insee_dirs = len(list(output_insee.glob("insee=*")))
    total_cp_dirs = len(list(output_cp.glob("postcode=*")))

    print("\n" + "=" * 80)
    print("✅ GÉNÉRATION DU STORE NATIONAL V8 TERMINÉE AVEC SUCCÈS")
    print("=" * 80)
    print(f"  Temps total d'exécution : {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"  Communes partitionnées  : {total_insee_dirs:,} partitions insee")
    print(f"  Codes postaux générés   : {total_cp_dirs:,} partitions postcode")
    print(f"  Répertoire cible        : {output_dir}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Génération haute performance du store partitionné SIRENE National V8.")
    parser.add_argument("--etab-path", type=Path, default=DEFAULT_ETAB_PARQUET, help="Chemin du Parquet StockEtablissement")
    parser.add_argument("--ul-path", type=Path, default=DEFAULT_UL_PARQUET, help="Chemin du Parquet StockUniteLegale")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Dossier de sortie des partitions")
    parser.add_argument("--max-dpts", type=int, default=None, help="Limiter à N départements (pour test)")
    parser.add_argument("--clean", action="store_true", help="Nettoyer le dossier cible avant écriture")
    parser.add_argument("--threads", type=int, default=8, help="Nombre de threads DuckDB")
    args = parser.parse_args()

    build_national_partitions(
        etab_path=args.etab_path,
        ul_path=args.ul_path,
        output_dir=args.output_dir,
        max_dpts=args.max_dpts,
        clean_target=args.clean,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
