"""Remote SSH configuration and command construction."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass


@dataclass
class RemoteConfig:
    host: str
    user: str
    port: int = 22
    key_path: str | None = None
    work_dir: str = ""

    @classmethod
    def from_env(cls) -> "RemoteConfig":
        host = os.getenv("BIO_REMOTE_HOST", "")
        user = os.getenv("BIO_REMOTE_USER", "")
        port = int(os.getenv("BIO_REMOTE_PORT", "22"))
        key_path = os.getenv("BIO_REMOTE_KEY") or None
        work_dir = os.getenv("BIO_REMOTE_WORK_DIR", "")
        return cls(host=host, user=user, port=port, key_path=key_path, work_dir=work_dir)

    def validate(self) -> str | None:
        if not self.host:
            return "BIO_REMOTE_HOST is not configured"
        if not self.user:
            return "BIO_REMOTE_USER is not configured"
        if not self.work_dir:
            return "BIO_REMOTE_WORK_DIR is not configured"
        return None


def build_ssh_command(config: RemoteConfig, remote_command: str) -> list[str]:
    target = f"{config.user}@{config.host}"
    cmd = [
        "ssh",
        "-p",
        str(config.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if config.key_path:
        cmd.extend(["-i", os.path.expanduser(config.key_path)])
    cmd.extend([target, remote_command])
    return cmd


def shell_cd_and_run(cwd: str, command: str) -> str:
    return f"cd {shlex.quote(cwd)} && {command}"
