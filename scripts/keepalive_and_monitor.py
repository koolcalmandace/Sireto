"""
keepalive_and_monitor.py
========================
Keeps Windows awake (prevents standby/sleep/hibernate) using SetThreadExecutionState
while monitoring a specific PID (default: 19952 - Fast Ranker 10k training).
Temporarily disables Windows standby timeouts, and AUTOMATICALLY RESTORES
original settings (5 min on AC, 3 min on DC) when training is done.
"""

import atexit
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

_RESTORED = False

def restore_power_settings():
    global _RESTORED
    if _RESTORED:
        return
    _RESTORED = True
    print("\n[RESTAURATION] Réactivation automatique des paramètres de veille Windows normaux...", flush=True)
    try:
        # Reset thread execution state
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass

    try:
        # Restore AC to 5 minutes (original value: 300s)
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "5"], check=False)
        # Restore DC to 3 minutes (original value: 180s)
        subprocess.run(["powercfg", "/change", "standby-timeout-dc", "3"], check=False)
        print("[SUCCÈS RESTAURATION] Veille normale réactivée : 5 min (secteur) / 3 min (batterie).", flush=True)
    except Exception as e:
        print(f"[ERREUR RESTAURATION] Impossible de restaurer powercfg: {e}", flush=True)

# Register cleanup on normal or abnormal exit
atexit.register(restore_power_settings)

def handle_sig(signum, frame):
    restore_power_settings()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sig)
signal.signal(signal.SIGTERM, handle_sig)

def set_keepalive():
    try:
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        res = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        return res != 0
    except Exception as e:
        print(f"[KeepAlive Error] {e}", flush=True)
        return False

def get_process_stats(pid: int):
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    h_proc = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
    if not h_proc:
        return None
    
    exit_code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h_proc, ctypes.byref(exit_code))
    is_alive = (exit_code.value == 259) # STILL_ACTIVE = 259

    creation = ctypes.c_ulonglong()
    exit_t = ctypes.c_ulonglong()
    kernel_t = ctypes.c_ulonglong()
    user_t = ctypes.c_ulonglong()
    ctypes.windll.kernel32.GetProcessTimes(
        h_proc,
        ctypes.byref(creation),
        ctypes.byref(exit_t),
        ctypes.byref(kernel_t),
        ctypes.byref(user_t)
    )
    total_cpu_sec = (kernel_t.value + user_t.value) / 10_000_000.0

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ctypes.windll.psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(pmc), pmc.cb)
    rss_mb = pmc.WorkingSetSize / (1024 * 1024)

    ctypes.windll.kernel32.CloseHandle(h_proc)

    return {
        "alive": is_alive,
        "exit_code": exit_code.value,
        "cpu_seconds": total_cpu_sec,
        "rss_mb": rss_mb,
    }

def get_system_ram():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "free_gb": stat.ullAvailPhys / (1024**3),
        "total_gb": stat.ullTotalPhys / (1024**3),
        "load_pct": stat.dwMemoryLoad,
    }

def main():
    target_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 19952
    project_root = Path(__file__).resolve().parent.parent
    target_file = Path(sys.argv[2]) if len(sys.argv) > 2 else (project_root / "reports" / "fast_ranker_32k" / "fast_ranker_32k_v8_unified_ban_summary.json")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Démarrage Sentinelle KeepAlive pour PID {target_pid}", flush=True)
    print(f"  Fichier cible attendu : {target_file}", flush=True)
    
    # Disable standby timeouts temporarily while running
    subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], check=False)
    subprocess.run(["powercfg", "/change", "standby-timeout-dc", "0"], check=False)
    set_keepalive()
    print("[SUCCÈS] Mode anti-veille activé (powercfg timeout=0 + SetThreadExecutionState).", flush=True)

    ckpt_file = project_root / "reports" / "fast_ranker_32k" / ".checkpoints" / "checkpoint_32k_v8_unified_ban.json"
    last_keepalive_refresh = time.time()

    try:
        while True:
            stats = get_process_stats(target_pid)
            if stats is None or not stats["alive"]:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processus PID {target_pid} terminé.", flush=True)
                if target_file.exists():
                    print(f"[SUCCÈS MAJEUR] Rapport cible sauvegardé : {target_file} ({target_file.stat().st_size} octets) !", flush=True)
                else:
                    print(f"[INFO] Processus achevé. Vérifiez les logs : logs/run_fast_ranker_32k_unified_ban.log", flush=True)
                break

            if time.time() - last_keepalive_refresh > 120:
                set_keepalive()
                last_keepalive_refresh = time.time()

            sys_ram = get_system_ram()
            cpu_s = stats["cpu_seconds"]
            rss = stats["rss_mb"]

            # Try to read real progress from details CSV or checkpoint JSON
            prog_str = ""
            details_file = project_root / "reports" / "fast_ranker_32k" / "fast_ranker_32k_v8_unified_ban_details.csv"
            if details_file.exists():
                try:
                    with open(details_file, "r", encoding="utf-8", errors="replace") as f:
                        lines_cnt = max(0, sum(1 for _ in f) - 1)
                    if lines_cnt > 0:
                        pct = (lines_cnt / 32570.0) * 100.0
                        prog_str = f" | Progrès={lines_cnt}/32570 ({pct:.1f}%)"
                except Exception:
                    pass
            elif ckpt_file.exists():
                try:
                    with open(ckpt_file, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
                    total_q = c_data.get("total_queries", 0)
                    total_r = c_data.get("total_rows", 32570)
                    if total_r > 0:
                        pct = (total_q / total_r) * 100.0
                        prog_str = f" | Progrès={total_q}/{total_r} ({pct:.1f}%)"
                except Exception:
                    pass

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"PID {target_pid}: CPU={cpu_s/3600:.2f}h ({cpu_s:.0f}s) | "
                f"RAM={rss:.0f}Mo{prog_str} | "
                f"RAM Libre Sys={sys_ram['free_gb']:.2f}Go ({sys_ram['load_pct']}%)",
                flush=True
            )

            time.sleep(30)
    finally:
        restore_power_settings()

if __name__ == "__main__":
    main()
