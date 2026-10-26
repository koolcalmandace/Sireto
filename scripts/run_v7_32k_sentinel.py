"""Sentinel runner for Full Pipe V7 32K (Stage 1 + Stage 2 + Stage 3) with Windows Anti-Sleep."""
import ctypes
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "sentinel_v7_32k.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sentinel_v7_32k")

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

    ranker_path = PROJECT_ROOT / "models" / "xgbranker_fast_v9.json"
    decider_path = PROJECT_ROOT / "models" / "xgb_decider_unified_20261025_082618.json"
    calibrator_path = PROJECT_ROOT / "models" / "xgb_decider_unified_calibrator_isotonic_20261025_082618.pkl"
    risk_model = PROJECT_ROOT / "models" / "routing_risk_model.pkl"
    risk_meta = PROJECT_ROOT / "models" / "routing_risk_meta.json"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_v7_full_pipeline_32k.py"),
        "--crm", str(PROJECT_ROOT / "data" / "crm_ok_gt_merged_v1.csv"),
        "--partitions-dir", str(PROJECT_ROOT / "data" / "candidates_v8_national"),
        "--output-dir", str(PROJECT_ROOT / "reports" / "full_pipeline_32k"),
        "--tag", "32k_v7_full_pipeline",
        "--ranker-model", str(ranker_path),
        "--decider-model", str(decider_path),
        "--calibrator-path", str(calibrator_path),
        "--risk-model", str(risk_model),
        "--risk-meta", str(risk_meta),
        "--prefilter-k", "500",
        "--top-k", "50",
        "--cache-size", "40",
        "--max-ram-gb", "3.5",
        "--checkpoint-interval", "250",
        "--no-pause-on-lock",
    ]

    logger.info("=" * 80)
    logger.info("LANCEMENT OFFICIEL DU BENCHMARK NATIONAL PIPE V7 32K (STAGE 1 + 2 + 3)")
    logger.info("Ranker (S1) : %s", ranker_path)
    logger.info("Decider (S2): %s", decider_path)
    logger.info("Risk (S3)   : %s", risk_model)
    logger.info("Commande    : %s", " ".join(cmd))
    logger.info("=" * 80)

    try:
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        logger.info("Benchmark national 32K termine avec le code de sortie: %s", ret.returncode)
        sys.exit(ret.returncode)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
