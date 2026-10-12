"""Replay three complete v6 data sets without touching their retained records."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
from v6 import run_experiment
from v6.tracking import ROOT,atomic_json


def main() -> dict:
    checks=[]
    for label in ['small','large','comparable']:
        c=json.loads((ROOT/f'configs/v6/{label}.json').read_text());c['repetitions']=1
        before=json.loads((ROOT/f"results/v6/{c['id']}/records/000000.json").read_text())
        with tempfile.TemporaryDirectory(prefix='edge_v6_replay_') as folder:
            old=run_experiment.prepare;run_experiment.prepare=lambda *args:Path(folder)
            try:run_experiment.run(c)
            finally:run_experiment.prepare=old
            after=json.loads((Path(folder)/'records/000000.json').read_text())
        if after['status']!='ok':raise RuntimeError(after)
        error=0.
        for b,a in zip(before['methods'],after['methods']):
            np.testing.assert_allclose(b['scores'],a['scores'],rtol=1e-12,atol=1e-12)
            error=max(error,float(np.max(np.abs(np.array(b['scores'])-a['scores']))))
            for name in ['method','selected','fdp','power','empty']:
                if b[name]!=a[name]:raise AssertionError('Decision not reproduced')
        checks.append(dict(id=c['id'],rep=0,score_vectors=4,maximum_absolute_error=error,bitwise_scores=error==0))
    result=dict(checks=checks,score_vectors=12,scope='Three individual repetitions, not a second execution of 3000 data sets')
    atomic_json(ROOT/'results/v6/aggregate/replay.json',result);return result


if __name__=='__main__':print(json.dumps(main(),indent=2))
