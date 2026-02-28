# -*- coding: utf-8 -*-
"""
SEANO Vision package init.

Patch kecil supaya rclpy.shutdown() aman dipanggil lebih dari sekali.
Ini menghilangkan traceback 'rcl_shutdown already called' saat launch mengirim SIGINT
dan node juga memanggil shutdown lagi di finally.

Catatan:
- Ini tidak mengubah perilaku runtime control.
- Hanya membuat shutdown lebih bersih (no noisy traceback).
"""

from __future__ import annotations

import rclpy


def _patch_safe_shutdown() -> None:
    if getattr(rclpy, "_seano_safe_shutdown_patched", False):
        return

    _orig_shutdown = rclpy.shutdown

    def _safe_shutdown(*args, **kwargs):
        try:
            _orig_shutdown(*args, **kwargs)
        except Exception as e:
            # Abaikan error "shutdown sudah dipanggil" (biar log bersih)
            msg = str(e)
            if "rcl_shutdown already called" in msg:
                return
            raise

    rclpy.shutdown = _safe_shutdown  # type: ignore[attr-defined]
    rclpy._seano_safe_shutdown_patched = True  # type: ignore[attr-defined]


_patch_safe_shutdown()
