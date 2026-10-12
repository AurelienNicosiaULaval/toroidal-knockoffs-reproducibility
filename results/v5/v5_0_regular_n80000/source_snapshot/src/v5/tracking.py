"""Immutable configurations, source snapshots and timestamped v5 experiments."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from provenance import ROOT, atomic_json, digest, stream


def prepare(config: dict, sources: list[str]) -> Path:
    out = ROOT/'results/v5'/config['id']
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(config=config,code={f:digest(ROOT/f) for f in sources},
                   python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                   platform=platform.platform())
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['signature'] != signature:
            raise RuntimeError('Code or configuration changed: use a new experiment ID')
    else:
        atomic_json(manifest,dict(**payload,signature=signature,created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        for name in sources:
            target=out/'source_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    return out
