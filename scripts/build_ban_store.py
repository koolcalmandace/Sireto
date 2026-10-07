"""
scripts/build_ban_store.py
==========================
Converts official Base Adresse Nationale (BAN) gzip CSV:
  data/adresses-france.csv.gz (~26M rows)
into a compact, indexed DuckDB database:
  data/ban.duckdb (~600-800 MB)

Extracts:
  - insee: 5-char INSEE code
  - postcode: 5-char postal code
  - nom_voie: normalized AFNOR street name
  - numero: street number
  - rep: repetition index (bis, ter, B, T...)
  - nom_ld: lieu-dit / toponyme
  - lat, lon: WGS84 coordinates (float32)

Builds covering indexes for sub-millisecond retrieval during inference.
"""

import os
import sys
import time
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_GZ_PATH = DATA_DIR / "adresses-france.csv.gz"
DB_PATH = DATA_DIR / "ban.duckdb"


def build_ban_store():
    print("=" * 75)
    print("CONVERSION & INDEXATION DE LA BASE ADRESSE NATIONALE (BAN)")
    print(f"Source CSV : {CSV_GZ_PATH}")
    print(f"Cible DB   : {DB_PATH}")
    print("=" * 75)

    if not CSV_GZ_PATH.exists():
        print(f"[ERREUR] Le fichier source {CSV_GZ_PATH} n'existe pas !")
        sys.exit(1)

    # Temporary database path to ensure atomic swap
    tmp_db_path = DATA_DIR / "ban.tmp.duckdb"
    if tmp_db_path.exists():
        tmp_db_path.unlink()

    t_start = time.perf_counter()
    con = duckdb.connect(str(tmp_db_path))

    # Configure DuckDB for high-throughput creation
    con.execute("PRAGMA threads = 4;")
    con.execute("PRAGMA memory_limit = '4GB';")

    print("[1/3] Création de la table 'ban_adresses' depuis le CSV gzip (~26M lignes)...")
    create_table_sql = f"""
    CREATE TABLE ban_adresses AS 
    SELECT 
        lpad(cast(code_insee as varchar), 5, '0') as insee,
        lpad(cast(code_postal as varchar), 5, '0') as postcode,
        upper(trim(coalesce(nom_afnor, nom_voie))) as nom_voie,
        cast(numero as varchar) as numero,
        upper(trim(coalesce(rep, ''))) as rep,
        upper(trim(coalesce(nom_ld, ''))) as nom_ld,
        cast(lat as float) as lat,
        cast(lon as float) as lon
    FROM read_csv(
        '{CSV_GZ_PATH.as_posix()}',
        delim=';',
        quote='"',
        header=true,
        compression='gzip',
        null_padding=true
    )
    WHERE lat IS NOT NULL AND lon IS NOT NULL;
    """
    t0 = time.perf_counter()
    con.execute(create_table_sql)
    t1 = time.perf_counter()
    count_rows = con.execute("SELECT count(*) FROM ban_adresses").fetchone()[0]
    print(f"  -> {count_rows:,} adresses géocodées insérées en {t1 - t0:.1f}s.")

    print("[2/3] Création de la table des centroïdes de voies et lieux-dits...")
    # Precompute street-level centroids for fallback when street number doesn't match
    create_streets_sql = """
    CREATE TABLE ban_voies AS 
    SELECT 
        insee,
        nom_voie,
        avg(lat)::float as lat,
        avg(lon)::float as lon,
        count(*) as nb_adresses
    FROM ban_adresses
    GROUP BY insee, nom_voie;
    """
    con.execute(create_streets_sql)
    nb_voies = con.execute("SELECT count(*) FROM ban_voies").fetchone()[0]
    print(f"  -> {nb_voies:,} voies et centroïdes pré-calculés.")

    print("[3/3] Création des index haute performance...")
    t2 = time.perf_counter()
    con.execute("CREATE INDEX idx_ban_insee_voie ON ban_adresses(insee, nom_voie);")
    con.execute("CREATE INDEX idx_ban_insee ON ban_adresses(insee);")
    con.execute("CREATE INDEX idx_voies_insee_nom ON ban_voies(insee, nom_voie);")
    t3 = time.perf_counter()
    print(f"  -> Index créés en {t3 - t2:.1f}s.")

    con.close()

    # Atomic rename
    if DB_PATH.exists():
        DB_PATH.unlink()
    tmp_db_path.rename(DB_PATH)

    total_time = time.perf_counter() - t_start
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print("=" * 75)
    print(f"[SUCCÈS] Base spatiale BAN prête : {DB_PATH}")
    print(f"  - Taille disque : {db_size_mb:.1f} Mo")
    print(f"  - Total adresses : {count_rows:,}")
    print(f"  - Total voies : {nb_voies:,}")
    print(f"  - Durée totale  : {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 75)


if __name__ == "__main__":
    build_ban_store()
