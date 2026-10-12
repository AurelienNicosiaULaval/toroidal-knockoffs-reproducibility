"""Versioned v4 runs; named tracking to avoid shadowing the retained provenance module."""
from __future__ import annotations
import hashlib,json,platform
from pathlib import Path
import numpy as np
import scipy
from provenance import ROOT,atomic_json,digest,stream


def prepare(config:dict,sources:list[str])->Path:
    out=ROOT/'results/v4'/config['id'];out.mkdir(parents=True,exist_ok=True)
    content=dict(config=config,code={f:digest(ROOT/f) for f in sources},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())
    signature=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    path=out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())['signature']!=signature:raise RuntimeError('Source/configuration changed; use a new experiment ID')
    else:
        atomic_json(path,dict(**content,signature=signature))
        for source in sources:
            target=out/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/source).read_bytes())
    return out
