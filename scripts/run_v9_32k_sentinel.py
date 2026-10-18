"""Sentinel runner for Fast Ranker 32K (V9) with Windows Anti-Sleep."""
import ctypes
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "sentinel_v9_32k.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sentinel_v9_32k")

IS_WINDOWS = sys.platform.startswith("win")
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


def anti_sleep_worker(stop_event: threading.Event):
    """Background thread preventing Windows sleep/standby during execution."""
    logger.info("[ANTI-SLEEP] Sentinelle anti-veille activee.")
    while not stop_event.is_set():
        if IS_WINDOWS:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED | ES_DISPLAY_REQUIRED
                )
            except Exception as e:
                logger.debug("SetThreadExecutionState error: %s", e)
        time.sleep(15)
    logger.info("[ANTI-SLEEP] Sentinelle anti-veille desactivee.")


def main():
    stop_event = threading.Event()
    anti_sleep_thread = threading.Thread(target=anti_sleep_worker, args=(stop_event,), daemon=True)
    anti_sleep_thread.start()

    model_path = PROJECT_ROOT / "models" / "xgbranker_fast_v9.json"
    baseline_details = PROJECT_ROOT / "reports" / "fast_ranker_32k" / "fast_ranker_32k_v8_retention_v2_details.csv"
    log_file = PROJECT_ROOT / "logs" / "run_fast_ranker_32k_v9.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_fast_ranker_32k.py"),
        "--crm", str(PROJECT_ROOT / "data" / "crm_ok_gt_merged_v1.csv"),
        "--partitions-dir", str(PROJECT_ROOT / "data" / "candidates_v8_national"),
        "--output-dir", str(PROJECT_ROOT / "reports" / "fast_ranker_32k"),
        "--tag", "32k_v9_retention_v2",
        "--model-path", str(model_path),
        "--baseline-details", str(baseline_details),
        "--prefilter-k", "500",
        "--top-k", "50",
        "--cache-size", "40",
        "--max-ram-gb", "3.5",
        "--checkpoint-interval", "250",
        "--no-pause-on-lock",
        "--log-file", str(log_file),
    ]

    logger.info("=" * 80)
    logger.info("LANCEMENT OFFICIEL DU BENCHMARK NATIONAL FAST RANKER 32K (V9)")
    logger.info("Modele   : %s", model_path)
    logger.info("Baseline : %s", baseline_details)
    logger.info("Commande : %s", " ".join(cmd))
    logger.info("=" * 80)

    try:
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        logger.info("Benchmark national 32K termine avec le code de sortie: %s", ret.returncode)
        sys.exit(ret.returncode)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
