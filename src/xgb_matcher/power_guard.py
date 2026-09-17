"""
power_guard.py — Windows Sleep, Lid-Close, Lock & Termination Power Monitor.

Monitors system power events on Windows (Modern Standby, Lid Close, Suspend,
Session Lock, System Shutdown) and signals compute loops to safely finish the active
query, persist checkpoints, and yield CPU load to 0% so Windows can enter true,
cool low-power sleep safely without thermal or power conflicts.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("xgb_matcher.power_guard")

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    DESKTOP_SWITCHDESKTOP = 0x0100

    # Win32 Execution State Flags
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.UINT),
            ("dwTime", wintypes.DWORD),
        ]


def _set_thread_execution_state(prevent_sleep: bool) -> None:
    """Set Win32 thread execution state to prevent unintended idle sleep while actively computing."""
    if not IS_WINDOWS:
        return
    try:
        if prevent_sleep:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception as e:
        logger.debug("Failed to set thread execution state: %s", e)


class PowerMonitor:
    """Monitors OS power state transitions, lid close, session lock, and user inactivity."""

    def __init__(
        self,
        pause_on_lock: bool = True,
        idle_timeout_seconds: float = 300.0,
        poll_interval: float = 1.0,
        on_pause_callback: Optional[Callable[[], None]] = None,
        on_resume_callback: Optional[Callable[[], None]] = None,
        on_exit_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.pause_on_lock = pause_on_lock
        self.idle_timeout_seconds = idle_timeout_seconds
        self.poll_interval = poll_interval
        self.on_pause_callback = on_pause_callback
        self.on_resume_callback = on_resume_callback
        self.on_exit_callback = on_exit_callback

        self._should_pause = False
        self._pause_reason = ""
        self._should_exit = False
        self._exit_reason = ""
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._ctrl_handler_ref = None

        # Thread synchronization events
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()  # Initially running

    @property
    def is_pause_requested(self) -> bool:
        """Check if a pause has been requested by a power/system event."""
        return self._should_pause

    def should_pause(self) -> bool:
        """Convenience method for worker loops to query pause status."""
        return self._should_pause

    @property
    def is_exit_requested(self) -> bool:
        """Check if a terminal exit (OS shutdown/window close) has been requested."""
        return self._should_exit

    def should_exit(self) -> bool:
        """Convenience method for worker loops to query terminal exit status."""
        return self._should_exit

    def trigger_pause(self, reason: str = "Manual / System Event") -> None:
        """Signal that computation must pause cleanly after current query."""
        if not self._should_pause and not self._should_exit:
            self._should_pause = True
            self._pause_reason = reason
            self._resume_event.clear()
            self._pause_event.set()
            _set_thread_execution_state(prevent_sleep=False)
            logger.warning("[POWER GUARD] Pause requested (%s). Finishing current query...", reason)
            if self.on_pause_callback:
                try:
                    self.on_pause_callback()
                except Exception as e:
                    logger.error("Error in on_pause_callback: %s", e)

    def trigger_exit(self, reason: str = "System Shutdown") -> None:
        """Signal that computation must immediately flush and exit (OS shutdown/terminal close)."""
        if not self._should_exit:
            self._should_exit = True
            self._exit_reason = reason
            # Cancel pause and unblock any waiting thread immediately so the process can exit cleanly
            self._should_pause = False
            self._pause_event.clear()
            self._resume_event.set()
            _set_thread_execution_state(prevent_sleep=False)
            logger.warning(
                "[POWER GUARD] Terminal exit requested (%s). Unblocking wait loop for immediate clean exit...",
                reason,
            )
            if self.on_exit_callback:
                try:
                    self.on_exit_callback()
                except Exception as e:
                    logger.error("Error in on_exit_callback: %s", e)

    def trigger_resume(self, reason: str = "System Resumed") -> None:
        """Signal that computation may resume."""
        if self._should_pause and not self._should_exit:
            logger.info("[POWER GUARD] Resume signal received (%s). Resuming computation...", reason)
            self._should_pause = False
            self._pause_reason = ""
            self._pause_event.clear()
            self._resume_event.set()
            _set_thread_execution_state(prevent_sleep=True)
            if self.on_resume_callback:
                try:
                    self.on_resume_callback()
                except Exception as e:
                    logger.error("Error in on_resume_callback: %s", e)

    def wait_for_resume(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks caller at 0% CPU until system resumes from sleep / lock / inactivity,
        or returns immediately if system is shutting down (should_exit).
        Returns True if resumed, False if timeout, stopped, or should exit.
        """
        if self._should_exit:
            return False

        logger.info(
            "[POWER GUARD] Computation safely paused for: '%s'. CPU load yielded to 0%%. Waiting for resume/wake...",
            self._pause_reason or "Sleep/Lid Event"
        )
        start_time = time.time()
        while self._should_pause and self._running and not self._should_exit:
            # Sleep in short intervals so thread responds to Ctrl+C, shutdown, or kill signals
            resumed = self._resume_event.wait(timeout=0.5)
            if self._should_exit:
                logger.warning("[POWER GUARD] Unblocked from pause: System shutdown requested!")
                return False
            if resumed and not self._should_pause:
                logger.info("[POWER GUARD] Woke up! Resuming execution seamlessly.")
                return True
            if timeout is not None and (time.time() - start_time) >= timeout:
                return not self._should_pause
        return not self._should_exit

    def start(self) -> None:
        """Start power monitoring thread and register OS hooks."""
        if self._running:
            return
        self._running = True
        _set_thread_execution_state(prevent_sleep=True)

        if IS_WINDOWS:
            self._register_windows_ctrl_handler()
            self._monitor_thread = threading.Thread(
                target=self._windows_monitor_loop,
                name="PowerGuardMonitor",
                daemon=True
            )
            self._monitor_thread.start()
        else:
            self._register_posix_signals()

    def stop(self) -> None:
        """Stop monitor and clean up OS hooks."""
        if not self._running:
            return
        self._running = False
        self.trigger_resume("Monitor stopped")
        _set_thread_execution_state(prevent_sleep=False)

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

    def __enter__(self) -> PowerMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # --- Windows Specific Implementation ---

    def _register_windows_ctrl_handler(self) -> None:
        """Register Console Control Handler to cleanly catch Ctrl+C, Terminal Close, and Shutdown."""
        HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def ctrl_handler(ctrl_type: int) -> bool:
            if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT):
                logger.info("[POWER GUARD] Interrupt signal (Ctrl+C) received. Triggering safe pause & save.")
                self.trigger_pause("User Ctrl+C Interrupt")
                return True  # Handled
            elif ctrl_type in (CTRL_CLOSE_EVENT, CTRL_SHUTDOWN_EVENT, CTRL_LOGOFF_EVENT):
                logger.warning("[POWER GUARD] System shutdown / console close detected! Triggering immediate clean exit.")
                self.trigger_exit("System Shutdown / Window Close")
                time.sleep(1.0)
                return True
            return False

        self._ctrl_handler_ref = HandlerRoutine(ctrl_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(self._ctrl_handler_ref, True)

    def _is_windows_session_locked(self) -> bool:
        """Check if Windows desktop is currently locked."""
        user32 = ctypes.windll.user32
        h_desk = user32.OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP)
        if not h_desk:
            return True
        user32.CloseDesktop(h_desk)
        return False

    def _get_windows_idle_seconds(self) -> float:
        """Returns seconds elapsed since last user keyboard or mouse input."""
        if not IS_WINDOWS:
            return 0.0
        try:
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                current_tick = ctypes.windll.kernel32.GetTickCount()
                elapsed_ms = (current_tick - lii.dwTime) & 0xFFFFFFFF
                return elapsed_ms / 1000.0
        except Exception:
            pass
        return 0.0

    def _windows_monitor_loop(self) -> None:
        """
        Daemon thread monitoring session lock state, user inactivity, and system time discontinuities
        (which occur when waking up from system sleep or hibernation).
        """
        last_tick = time.monotonic()

        while self._running:
            time.sleep(self.poll_interval)
            if not self._running:
                break

            now_tick = time.monotonic()
            elapsed_real = now_tick - last_tick
            last_tick = now_tick

            # 1. Detection of sleep wake: If monotonic time jumped significantly more than sleep duration
            if elapsed_real > (self.poll_interval * 3.0 + 2.0):
                logger.info(
                    "[POWER GUARD] Time discontinuity detected (jumped %.1fs). System woke from sleep!",
                    elapsed_real
                )
                if self._should_pause and any(k in self._pause_reason for k in ("Sleep", "Inactive", "Locked")):
                    self.trigger_resume("System Woke from Sleep")

            # 2. Check session lock if enabled
            if self.pause_on_lock:
                try:
                    is_locked = self._is_windows_session_locked()
                    if is_locked and not self._should_pause:
                        self.trigger_pause("Session Locked")
                    elif not is_locked and self._should_pause and self._pause_reason == "Session Locked":
                        self.trigger_resume("Session Unlocked")
                except Exception:
                    pass

            # 3. Check user inactivity timeout (default 5 min = 300s)
            if self.idle_timeout_seconds > 0:
                try:
                    idle_sec = self._get_windows_idle_seconds()
                    resume_threshold = min(5.0, self.idle_timeout_seconds * 0.5)
                    if idle_sec >= self.idle_timeout_seconds and not self._should_pause:
                        self.trigger_pause(
                            f"User Inactive ({idle_sec:.0f}s >= {self.idle_timeout_seconds:.0f}s)"
                        )
                    elif idle_sec < resume_threshold and self._should_pause and "User Inactive" in self._pause_reason:
                        self.trigger_resume("User Activity Detected")
                except Exception:
                    pass

    def _register_posix_signals(self) -> None:
        """Register POSIX signal handlers (SIGINT for pause, SIGTERM for clean exit) for Linux/Mac."""
        def sigint_handler(signum, frame):
            logger.info("[POWER GUARD] Signal SIGINT received. Triggering safe pause...")
            self.trigger_pause(f"POSIX Signal {signum}")

        def sigterm_handler(signum, frame):
            logger.warning("[POWER GUARD] Signal SIGTERM received. Triggering clean exit...")
            self.trigger_exit(f"POSIX Signal {signum}")

        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGTERM, sigterm_handler)

