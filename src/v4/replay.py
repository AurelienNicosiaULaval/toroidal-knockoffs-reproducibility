"""Re-execute three preselected v4 replicates in temporary directories."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from v4 import run_local, run_counterexample
from v4.tracking import ROOT, atomic_json


def replay() -> dict:
    checks = []
    for file, module in [('local_star_g1p0_n1600.json',run_local),
                         ('counter_n2000.json',run_counterexample),
                         ('counter_n32000.json',run_counterexample)]:
        config = json.loads((ROOT/'configs/v4'/file).read_text())
        original = json.loads((ROOT/'results/v4'/config['id']/'records/000000.json').read_text())
        config['repetitions'] = 1
        with tempfile.TemporaryDirectory(prefix='edge_v4_replay_') as temporary:
            saved_prepare = module.prepare
            module.prepare = lambda *args: Path(temporary)
            try:
                module.run(config)
            finally:
                module.prepare = saved_prepare
            repeated = json.loads((Path(temporary)/'records/000000.json').read_text())
        if repeated['status'] != 'ok':
            raise RuntimeError(repeated)
        for old, new in zip(original['methods'], repeated['methods']):
            for key in ('scores','selected','fdp','power','empty'):
                if old[key] != new[key]:
                    raise AssertionError(f"Replay differs: {config['id']} {key}")
        checks.append(dict(id=config['id'],replicate=0,matching_score_vectors=len(original['methods'])))
    result=dict(checks=checks,total_matching_score_vectors=sum(c['matching_score_vectors'] for c in checks),
                scope='Three individual replicates, not a second execution of the 9000 data sets.')
    atomic_json(ROOT/'results/v4/aggregate/replay.json',result)
    return result


if __name__ == '__main__':
    print(json.dumps(replay(),indent=2))
