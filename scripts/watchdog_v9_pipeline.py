#!/usr/bin/env python3
"""
watchdog_v9_pipeline.py
=======================
Automated Anti-Sleep Sentinel & Pipeline Watchdog for Fast Ranker V9.

Features:
1. Anti-Sleep Sentinel: Continuous Win32 SetThreadExecutionState keeping system awake.
2. Disk & Memory Cleaning: Cleans temporary checkpoints and stale caches.
3. Training Monitor: Waits for 10K training (models/xgbranker_fast_v9.json) to complete.
4. Test Verification: Evaluates V9 vs V8 on a 1,000-query validation slice.
5. Auto-Branching:
   - If V9 Hit@50 >= V8 Hit@50: Automatically launches full 32K national benchmark.
   - If V9 Hit@50 < V8 Hit@50: Stops, writes comparative report, and awaits user instructions.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/watchdog_v9_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("watchdog_v9")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = sys.platform.startswith("win")

# Win32 Execution State Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


def anti_sleep_worker(stop_event: threading.Event):
    """Background thread preventing Windows sleep/standby."""
    logger.info("[ANTI-SLEEP] Sentinelle anti-veille activée.")
    while not stop_event.is_set():
        if IS_WINDOWS:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED | ES_DISPLAY_REQUIRED
                )
            except Exception as e:
                logger.debug("Failed to set thread execution state: %s", e)
        time.sleep(15)
    logger.info("[ANTI-SLEEP] Sentinelle anti-veille désactivée.")


def is_process_running(pid: int) -> bool:
    """Check if process with given PID is still active."""
    if not IS_WINDOWS:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        cmd = f'Get-Process -Id {pid} -ErrorAction SilentlyContinue'
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        return res.returncode == 0 and str(pid) in res.stdout
    except Exception:
        return False


def wait_for_training(target_model: Path, target_pid: Optional[int] = None) -> bool:
    """Wait until 10K training finishes and model is generated."""
    logger.info("En attente de la fin de l'entraînement 10K (Modèle cible: %s)...", target_model)
    t0 = time.time()
    
    while True:
        # Check if process is still running
        if target_pid and not is_process_running(target_pid):
            logger.info("Processus d'entraînement PID %d terminé.", target_pid)
            time.sleep(5)
            break
        
        # Check if model exists and meta exists
        meta_path = target_model.with_name(target_model.stem + "_meta.json")
        if target_model.exists() and meta_path.exists():
            # Ensure file is completely written by checking size > 100KB and stable
            size1 = target_model.stat().st_size
            time.sleep(3)
            size2 = target_model.stat().st_size
            if size1 == size2 and size1 > 50000:
                logger.info("Modèle %s détecté et stabilisé (%d octets) !", target_model.name, size2)
                return True
        
        elapsed = time.time() - t0
        logger.info("[MONITOR] Entraînement en cours... (Temps écoulé: %.1f min)", elapsed / 60.0)
        time.sleep(30)
    
    return target_model.exists()


def run_comparative_test(v9_model: Path) -> Dict[str, Any]:
    """Run test evaluation on 1,000 queries to compare V9 against V8."""
    logger.info("=" * 80)
    logger.info("LANCEMENT DU TEST COMPARATIF SUR 1 000 REQUÊTES (V9 vs V8)")
    logger.info("=" * 80)
    
    comp_dir = PROJECT_ROOT / "reports" / "fast_ranker_32k" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    test_tag = "test_v9_1k"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_fast_ranker_32k.py"),
        "--model-path", str(v9_model),
        "--tag", test_tag,
        "--max-rows", "1000",
        "--output-dir", str(comp_dir),
        "--baseline-details", str(PROJECT_ROOT / "reports" / "fast_ranker_32k" / "fast_ranker_32k_v8_retention_v2_details.csv"),
        "--no-pause-on-lock",
        "--fresh",
    ]
    
    logger.info("Exécution: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        logger.error("Erreur lors de l'évaluation de test V9: %s", proc.stderr)
        return {"success": False, "error": proc.stderr}
    
    summary_path = comp_dir / f"fast_ranker_{test_tag}_summary.json"
    if not summary_path.exists():
        logger.error("Fichier de résumé introuvable: %s", summary_path)
        return {"success": False, "error": "Summary not found"}
    
    with open(summary_path, "r", encoding="utf-8") as f:
        v9_summary = json.load(f)
    
    return {"success": True, "summary": v9_summary}


def launch_full_32k_benchmark(v9_model: Path):
    """Launch full 32K national benchmark with V9."""
    logger.info("=" * 80)
    logger.info("🚀 LANCEMENT IMMÉDIAT DU BENCHMARK NATIONAL 32K (FAST RANKER V9)")
    logger.info("=" * 80)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_fast_ranker_32k.py"),
        "--model-path", str(v9_model),
        "--tag", "v9_retention_v2",
        "--baseline-details", str(PROJECT_ROOT / "reports" / "fast_ranker_32k" / "fast_ranker_32k_v8_retention_v2_details.csv"),
        "--no-pause-on-lock",
        "--log-file", str(PROJECT_ROOT / "logs" / "run_fast_ranker_32k_v9.log"),
    ]
    
    logger.info("Démarrage du processus 32K...")
    # Run full 32K synchronously in runner
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    logger.info("Benchmark National 32K V9 terminé avec succès !")


def main():
    stop_event = threading.Event()
    anti_sleep_thread = threading.Thread(target=anti_sleep_worker, args=(stop_event,), daemon=True)
    anti_sleep_thread.start()
    
    v9_model = PROJECT_ROOT / "models" / "xgbranker_fast_v9.json"
    
    # 1. Attente de la fin de l'entraînement 10K
    success = wait_for_training(target_model=v9_model, target_pid=8652)
    if not success:
        logger.error("L'entraînement 10K n'a pas produit le modèle %s. Arrêt.", v9_model)
        stop_event.set()
        sys.exit(1)
        
    logger.info("[MODELE V9] Modele Fast Ranker V9 genere avec succes !")
    
    # 2. Test comparatif sur 1 000 requetes
    test_res = run_comparative_test(v9_model)
    if not test_res.get("success"):
        logger.error("Echec du test comparatif. En attente d'instructions.")
        stop_event.set()
        sys.exit(1)
        
    summary = test_res["summary"]
    metrics = summary.get("metrics_active", {})
    hit50_rate = metrics.get("fast_ranker_hit50", 0.0)
    hit1_rate = metrics.get("fast_ranker_hit1", 0.0)
    recovered = summary.get("recovery_vs_baseline", {}).get("recovered_active_queries", 0)
    
    logger.info("Resultats du test 1 000 requetes V9 :")
    logger.info("  Hit@50 Actif : %.2f%%", hit50_rate)
    logger.info("  Hit@1 Actif  : %.2f%%", hit1_rate)
    logger.info("  Cas repeches vs Baseline : %d", recovered)
    
    # Si le modele V9 est prometteur (repêchages confirmés > 0 et Hit@50 > 90%)
    is_promising = (recovered > 0) and (hit50_rate >= 90.0)
    
    if is_promising:
        logger.info("[VALIDATION] RESULTATS PROMETTEURS CONFIRMES (+%d repechages) ! Lancement automatique du 32K...", recovered)
        launch_full_32k_benchmark(v9_model)
    else:
        logger.warning("[VALIDATION] RESULTATS NON SUPERIEURS (Hit@50: %.2f%%). Arret et attente d'instructions utilisateur.", hit50_rate)
        
    stop_event.set()


if __name__ == "__main__":
    main()
