#!/usr/bin/env python3
"""Copy the root-only Compose secret, then run the OCR service as its own UID."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SECRET_SOURCE = Path("/run/secrets/postgres_password")
SECRET_DESTINATION = Path("/run/ocr-secrets/postgres_password")
OCR_UID = 10001
OCR_GID = 10001


def main() -> None:
    """Prepare a private secret copy and replace this process with the service."""
    if os.geteuid() != 0:
        raise RuntimeError("OCR entrypoint must begin as root to read the Compose secret")
    if not SECRET_SOURCE.is_file():
        raise FileNotFoundError(f"Missing PostgreSQL secret: {SECRET_SOURCE}")

    SECRET_DESTINATION.parent.mkdir(mode=0o755, exist_ok=True)
    shutil.copyfile(SECRET_SOURCE, SECRET_DESTINATION)
    os.chown(SECRET_DESTINATION, OCR_UID, OCR_GID)
    os.chmod(SECRET_DESTINATION, 0o400)

    os.setgroups([OCR_GID])
    os.setgid(OCR_GID)
    os.setuid(OCR_UID)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: ocr-entrypoint.py COMMAND [ARGUMENT ...]")
    main()
