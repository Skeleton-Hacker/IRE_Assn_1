from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ire_assn1.paths import ROOT, project_path
from ire_assn1.settings import load_config


def check_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_doctor(config: Path) -> None:
    root, _ = load_config(config)
    paths = [
        root.paths.data,
        root.paths.models,
        root.paths.logs,
        root.paths.output,
        root.paths.plots,
    ]
    for path in paths:
        project_path(path).mkdir(parents=True, exist_ok=True)
    git = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    usage = shutil.disk_usage(project_path(root.paths.data))
    result: dict[str, Any] = {
        "git_clean": not bool(git.stdout.strip()),
        "disk_free_bytes": usage.free,
        "imports": {
            name: check_import(name)
            for name in [
                "numpy",
                "polars",
                "pyarrow",
                "bm25s",
                "faiss",
                "torch",
                "sentence_transformers",
            ]
        },
    }
    if result["imports"]["torch"]:
        import importlib

        torch = importlib.import_module("torch")

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    print(json.dumps(result, indent=2, sort_keys=True))
