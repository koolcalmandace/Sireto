#!/usr/bin/env python3
"""
scripts/repair_missing_v8_departments.py
========================================
Targeted extraction and repair for the 12 missing/incomplete departments in data/candidates_v8_national:
  ['25', '26', '31', '33', '38', '39', '45', '60', '70', '71', '74', '83'] + Morlaix ('29151').

Uses DuckDB C++ partitioned copy into isolated staging, then moves complete insee=XXXXX and postcode=XXXXX
folders into data/candidates_v8_national, avoiding any risk of accidental overwrite.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import duckdb
from tqdm import tqdm

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_ETAB = Path("data/source_sirene_20260901/StockEtablissement_utf8_20260901.parquet")
DEFAULT_UL = Path("data/source_sirene_20260901/StockUniteLegale_utf8_20260901.parquet")
DEFAULT_OUT = Path("data/candidates_v8_national")

MISSING_DEPARTMENTS = ["25", "26", "31", "33", "38", "39", "45", "60", "70", "71", "74", "83"]


def _safe_move_dir(src: Path | str, dst: Path | str) -> None:
    src = Path(src)
    dst = Path(dst)
    for attempt in range(5):
        try:
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.move(str(src), str(dst))
            return
        except Exception:
            time.sleep(0.5 * (attempt + 1))
            try:
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                shutil.rmtree(str(src), ignore_errors=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5)



def repair_department(
    dpt: str,
    threads: int,
    etab_path: Path,
    ul_path: Path,
    target_insee: Path,
    target_cp: Path,
    staging_root: Path,
) -> tuple[int, int]:
    """Extract and populate insee and cp partitions for a single department."""
    staging_dpt = staging_root / f"stage_{dpt}"
    stage_insee = staging_dpt / "insee"
    stage_cp = staging_dpt / "cp"

    if staging_dpt.exists():
        shutil.rmtree(staging_dpt, ignore_errors=True)
    stage_insee.mkdir(parents=True, exist_ok=True)
    stage_cp.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    con = duckdb.connect()
    con.execute(f"PRAGMA threads = {threads};")
    con.execute("PRAGMA preserve_insertion_order = false;")

    # 1. Extraction INSEE
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
        ) TO '{stage_insee.as_posix()}' (FORMAT PARQUET, PARTITION_BY (insee), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1);
    """
    con.execute(query_insee)

    # 2. Extraction CP
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
        ) TO '{stage_cp.as_posix()}' (FORMAT PARQUET, PARTITION_BY (postcode), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1);
    """
    con.execute(query_cp)
    con.close()

    # 3. Déplacement atomique des dossiers générés vers data/candidates_v8_national
    insee_count = 0
    with os.scandir(stage_insee) as it:
        for entry in it:
            if entry.is_dir() and entry.name.startswith("insee="):
                dest = target_insee / entry.name
                _safe_move_dir(entry.path, dest)
                insee_count += 1

    cp_count = 0
    with os.scandir(stage_cp) as it:
        for entry in it:
            if entry.is_dir() and entry.name.startswith("postcode="):
                dest = target_cp / entry.name
                _safe_move_dir(entry.path, dest)
                cp_count += 1

    shutil.rmtree(staging_dpt, ignore_errors=True)
    t1 = time.perf_counter()
    print(f"[OK] Dept {dpt:>2} repare : {insee_count:>3} communes INSEE, {cp_count:>3} codes postaux en {t1 - t0:.1f}s")
    return insee_count, cp_count


def repair_morlaix(
    threads: int,
    etab_path: Path,
    ul_path: Path,
    target_insee: Path,
    staging_root: Path,
) -> None:
    """Spécifiquement pour Morlaix (29151) qui avait été écrasé."""
    staging = staging_root / "stage_morlaix"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads = {threads};")
    con.execute("PRAGMA preserve_insertion_order = false;")

    query = f"""
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
                    '29151' AS insee,
                    e.etatAdministratifEtablissement AS etat_admin,
                    e.dateDernierTraitementEtablissement AS last_treatment_date
                FROM read_parquet('{etab_path.as_posix()}') e
                WHERE (e.codeCommuneEtablissement = '29151' OR LPAD(TRIM(e.codeCommuneEtablissement), 5, '0') = '29151')
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
        ) TO '{staging.as_posix()}' (FORMAT PARQUET, PARTITION_BY (insee), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1);
    """
    con.execute(query)
    con.close()

    dest = target_insee / "insee=29151"
    src = staging / "insee=29151"
    if src.exists():
        _safe_move_dir(src, dest)
        print(f"[OK] Morlaix (29151) repare avec succes !")
    shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing departments in V8 national store")
    parser.add_argument("--etab", type=Path, default=DEFAULT_ETAB)
    parser.add_argument("--ul", type=Path, default=DEFAULT_UL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--departments",
        nargs="+",
        default=MISSING_DEPARTMENTS,
        help="List of department numbers to repair (default: all 12 missing)",
    )
    args = parser.parse_args()

    target_insee = args.out / "insee"
    target_cp = args.out / "cp"
    staging_root = Path("scratch/staging_repair_v8")
    staging_root.mkdir(parents=True, exist_ok=True)

    dpts_to_repair = args.departments

    print("=" * 80)
    print("[REPAIR] REPARATION CIBLEE DES DEPARTEMENTS DANS CANDIDATES V8")
    print("=" * 80)
    print(f"Departements cibles : {', '.join(dpts_to_repair)}")

    total_insee = 0
    total_cp = 0
    t_global = time.perf_counter()

    for dpt in dpts_to_repair:
        insee_cnt, cp_cnt = repair_department(
            dpt, args.threads, args.etab, args.ul, target_insee, target_cp, staging_root
        )
        total_insee += insee_cnt
        total_cp += cp_cnt

    if args.departments == MISSING_DEPARTMENTS or "29" in args.departments or "29151" in args.departments:
        print("\nReparation de Morlaix (29151)...")
        repair_morlaix(args.threads, args.etab, args.ul, target_insee, staging_root)

    shutil.rmtree(staging_root, ignore_errors=True)
    t_end = time.perf_counter()
    print("=" * 80)
    print(f"[DONE] REPARATION TERMINEE en {t_end - t_global:.1f}s !")
    print(f"   Total communes restaurees : {total_insee}")
    print(f"   Total codes postaux restaures : {total_cp}")
    print("=" * 80)


if __name__ == "__main__":
    main()
