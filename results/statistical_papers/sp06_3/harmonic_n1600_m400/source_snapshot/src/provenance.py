"""Content-addressed experiment manifests and atomic, resumable checkpoints."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
CORE = ['src/toroidal.py','src/experimental_design.py','src/fit_model.py',
        'src/metro_circle.py','src/loading.py','src/metro_variants.py',
        'src/experiments_v2.py','src/provenance.py']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def prepare(config: dict) -> Path:
    manifest = dict(config=config, code={p:digest(ROOT/p) for p in CORE},
                    environment={'python':platform.python_version(),'numpy':np.__version__,
                                 'scipy':scipy.__version__, 'platform':platform.platform()})
    signature = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    directory = ROOT/'results/v2'/config['id']
    path = directory/'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['signature'] != signature:
            raise RuntimeError(f'{directory}: code/config/environment changed; create a new experiment id')
    else:
        manifest.update(signature=signature,created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(path,manifest)
        source=directory/'source_snapshot'
        for p in CORE:
            target=source/p;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/p).read_bytes())
    return directory


def stream(config: dict, replicate: int, role: str) -> np.random.Generator:
    # Role names decouple algorithms from their order in the configuration.
    key = int.from_bytes(hashlib.sha256(role.encode()).digest()[:4],'little')
    return np.random.default_rng(np.random.SeedSequence([config['seed'],replicate,key]))
