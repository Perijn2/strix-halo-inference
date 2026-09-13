#!/usr/bin/env python3
"""Privileged handoff for the OCR ensemble service.

Author: Perijn
Summary: Copy the root-only Compose secret, then replace this process with the service running as the OCR account under that account's own home.
Usage: Container ENTRYPOINT only. It must begin as root; any other caller is rejected before the secret is read.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SECRET_SOURCE = Path("/run/secrets/postgres_password")
SECRET_DESTINATION = Path("/run/ocr-secrets/postgres_password")
OCR_UID = 10001
OCR_GID = 10001
# Must match the home of the account created in docker/ocr-ensemble/Dockerfile.
# setuid() changes who the process is but not where "~" resolves, so a dropped
# process keeps root's HOME. PP-OCRv5 failed exactly there: PaddleX builds its
# cache and temp root from HOME, and UID 10001 may not create anything inside
# root's 0700 home ("PermissionError: [Errno 13] Permission denied:
# '/root/.paddlex/temp'"). Carrying HOME across the drop is the fix.
OCR_HOME = "/home/ocr"


def main() -> None:
    """Prepare a private secret copy, then hand the process to the OCR account."""
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
    os.environ["HOME"] = OCR_HOME
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: ocr-entrypoint.py COMMAND [ARGUMENT ...]")
    main()
