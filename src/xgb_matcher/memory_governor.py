"""
memory_governor.py — Dynamic Memory Watchdog & Cache Governor.

Monitors process Resident Set Size (RSS) in real time and enforces a hard upper
memory ceiling (< 4.5 GB) during high-throughput evaluation and matching loops.

Multi-tier memory release strategy:
  1. Application tier: Flush in-memory candidate store and TF-IDF LRU caches.
  2. CPython GC tier: Force generational garbage collection (gc.collect(2)).
  3. OS working set tier: Release uncommitted/idle memory pages back to Windows OS
     via kernel32.SetProcessWorkingSetSize(-1, -1).
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try importing psutil for high-precision cross-platform memory telemetry
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False


class MemoryGovernor:
    """Monitors and governs memory usage to prevent RAM exhaustion."""

    def __init__(
        self,
        target_max_gb: float = 4.0,
        critical_max_gb: float = 4.5,
        check_interval: int = 10,
    ) -> None:
        """
        Args:
            target_max_gb: Soft ceiling (in GB) triggering preventative cache purge.
            critical_max_gb: Hard ceiling (in GB) triggering aggressive multi-tier release.
            check_interval: Check memory every N queries/iterations.
        """
        self.target_max_bytes = int(target_max_gb * 1024 * 1024 * 1024)
        self.critical_max_bytes = int(critical_max_gb * 1024 * 1024 * 1024)
        self.check_interval = max(1, int(check_interval))
        self._counter = 0
        self._process = psutil.Process() if _HAS_PSUTIL else None
        self._total_purges = 0

    def get_rss_bytes(self) -> int:
        """Return the current process RSS (Resident Set Size) in bytes."""
        if self._process is not None:
            try:
                return int(self._process.memory_info().rss)
            except Exception:
                pass

        # Windows ctypes fallback if psutil is not available
        if sys.platform == "win32":
            try:
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
                kernel32 = ctypes.windll.kernel32
                psapi = ctypes.windll.psapi
                kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                psapi.GetProcessMemoryInfo.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                    ctypes.c_ulong,
                ]
                psapi.GetProcessMemoryInfo.restype = ctypes.c_bool

                handle = kernel32.GetCurrentProcess()
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                    return int(pmc.WorkingSetSize)
            except Exception:
                pass

        return 0

    def get_rss_mb(self) -> float:
        """Return current RSS in Megabytes."""
        return self.get_rss_bytes() / (1024.0 * 1024.0)

    def get_rss_gb(self) -> float:
        """Return current RSS in Gigabytes."""
        return self.get_rss_bytes() / (1024.0 * 1024.0 * 1024.0)

    @staticmethod
    def trim_os_working_set() -> None:
        """Request the OS memory manager to trim uncommitted memory pages."""
        if sys.platform == "win32":
            try:
                # Passing (-1, -1) tells Windows kernel to empty process working set
                ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    ctypes.windll.kernel32.GetCurrentProcess(),
                    ctypes.c_size_t(-1),
                    ctypes.c_size_t(-1),
                )
            except Exception:
                pass

    def release_memory(
        self,
        store: Optional[Any] = None,
        tfidf_cache: Optional[Any] = None,
        reason: str = "MANUAL",
    ) -> tuple[float, float]:
        """Perform a multi-tier memory purge and return (rss_before_gb, rss_after_gb)."""
        rss_before_gb = self.get_rss_gb()

        # Tier 1: Application cache flush
        if store is not None and hasattr(store, "clear_all_caches"):
            store.clear_all_caches()

        if tfidf_cache is not None:
            try:
                tfidf_cache.clear()
            except Exception:
                pass

        # Tier 2: Generational Garbage Collection
        gc.collect(2)

        # Tier 3: OS Working Set Trim
        self.trim_os_working_set()

        rss_after_gb = self.get_rss_gb()
        self._total_purges += 1

        logger.info(
            "MEMORY_GOVERNOR: Purge triggered (%s) — RSS: %.2f GB -> %.2f GB (Freed: %.2f GB)",
            reason,
            rss_before_gb,
            rss_after_gb,
            max(0.0, rss_before_gb - rss_after_gb),
        )

        return (rss_before_gb, rss_after_gb)

    def check_and_release(
        self,
        store: Optional[Any] = None,
        tfidf_cache: Optional[Any] = None,
        force: bool = False,
    ) -> bool:
        """Check RSS against memory ceiling and release if exceeded or forced.

        Args:
            store: Optional PartitionedCandidateStore instance to flush.
            tfidf_cache: Optional OrderedDict / in-memory cache to clear.
            force: If True, executes purge unconditionally.

        Returns:
            bool: True if memory release was executed, False otherwise.
        """
        self._counter += 1
        if not force and (self._counter % self.check_interval != 0):
            return False

        rss = self.get_rss_bytes()

        if force:
            self.release_memory(store, tfidf_cache, reason="FORCED")
            return True

        if rss >= self.critical_max_bytes:
            self.release_memory(store, tfidf_cache, reason="CRITICAL_CAP_EXCEEDED")
            return True

        if rss >= self.target_max_bytes:
            self.release_memory(store, tfidf_cache, reason="TARGET_CAP_REACHED")
            return True

        return False
