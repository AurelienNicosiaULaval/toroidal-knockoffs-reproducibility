"""Re-execute one saved seed per principal v3 configuration in a temporary root."""
from __future__ import annotations
import json,shutil,tempfile,time
from pathlib import Path
import numpy as np
from provenance import ROOT,atomic_json,digest
from v3 import run_learned


def main():
    rows=[];started=time.perf_counter()
    for name in json.loads((ROOT/'configs/v3/study_manifest.json').read_text())['experiments']:
        config=json.loads((ROOT/name).read_text());saved=ROOT/'results/v3'/config['id']
        manifest=json.loads((saved/'manifest.json').read_text())
        for source,sha in manifest['code'].items():
            if digest(ROOT/source)!=sha:raise RuntimeError('Use the recorded source version for replay')
        reference=json.loads((saved/'records/000000.json').read_text())
        with tempfile.TemporaryDirectory(prefix='toroidal_v3_replay_') as directory:
            temp=Path(directory)
            for f in run_learned.SOURCES:
                target=temp/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
            original=run_learned.ROOT
            try:
                run_learned.ROOT=temp;run_learned.run(config,limit=1)
            finally:run_learned.ROOT=original
            actual=json.loads((temp/'results/v3'/config['id']/'records/000000.json').read_text())
        if actual['status']!=reference['status'] or actual['status']!='ok':raise AssertionError('Replay failure')
        maximum=0.
        for a,b in zip(actual['methods'],reference['methods']):
            if (a['method'],a['response'],a['selected'])!=(b['method'],b['response'],b['selected']):raise AssertionError('Selection mismatch')
            maximum=max(maximum,float(np.max(np.abs(np.array(a['scores'])-b['scores']))))
            for key in ['max_field_mse','max_null_logratio']:
                if not np.isclose(a[key],b[key],atol=1e-12,rtol=1e-12):raise AssertionError(key+' mismatch')
        for label in actual['fits']:
            for key in ['h','M','edges']:
                np.testing.assert_allclose(actual['fits'][label][key],reference['fits'][label][key],atol=1e-12,rtol=1e-12)
        if maximum>1e-12:raise AssertionError('Score mismatch')
        rows.append(dict(experiment=config['id'],rep=0,method_response_scores=8,max_absolute_score_difference=maximum))
    atomic_json(ROOT/'results/v3/validation/replay.json',dict(rows=rows,seconds=time.perf_counter()-started))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
