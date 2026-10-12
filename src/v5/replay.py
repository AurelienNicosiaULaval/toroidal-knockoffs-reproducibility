"""Reproduce three v5 seeds without modifying their original records or timing logs."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import numpy as np
from v5 import run_experiment
from v5.tracking import ROOT,atomic_json


def replay() -> dict:
    checks=[]
    for name in ['harmonic_cost_n400','regular_n80000','regular_n320000']:
        c=json.loads((ROOT/f'configs/v5/{name}.json').read_text())
        original=json.loads((ROOT/f"results/v5/{c['id']}/records/000000.json").read_text())
        c['repetitions']=1
        with tempfile.TemporaryDirectory(prefix='edge_v5_replay_') as temporary:
            saved=run_experiment.prepare;run_experiment.prepare=lambda *args:Path(temporary)
            try:run_experiment.run(c)
            finally:run_experiment.prepare=saved
            repeated=json.loads((Path(temporary)/'records/000000.json').read_text())
        if repeated['status']!='ok':raise RuntimeError(repeated)
        error=0.;exact=True
        for before,after in zip(original['methods'],repeated['methods']):
            difference=np.max(np.abs(np.array(before['scores'])-after['scores']))
            error=max(error,float(difference));exact=exact and difference==0
            np.testing.assert_allclose(before['scores'],after['scores'],rtol=1e-12,atol=1e-12)
            for key in ['selected','fdp','power','empty']:
                if before[key]!=after[key]:raise AssertionError('Non-reproduced decision')
        checks.append(dict(id=c['id'],rep=0,score_vectors=len(original['methods']),maximum_absolute_error=error,bitwise_scores=bool(exact)))
    result=dict(checks=checks,score_vectors=sum(r['score_vectors'] for r in checks),
                scope='Three individual seeds, not a second execution of the main studies')
    atomic_json(ROOT/'results/v5/aggregate/replay.json',result)
    return result


if __name__=='__main__':print(json.dumps(replay(),indent=2))
