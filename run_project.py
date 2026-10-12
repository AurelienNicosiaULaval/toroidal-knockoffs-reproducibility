"""Check, rebuild tables/figures, or replay one seed in the public SP14 package."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result)
    return result


def knockoff(w: np.ndarray) -> np.ndarray:
    for t in np.unique(np.abs(w[w != 0])):
        if (1+np.count_nonzero(w <= -t))/max(1, np.count_nonzero(w >= t)) <= .1:
            return w >= t
    return np.zeros(len(w), bool)


def by(pvalues: np.ndarray) -> np.ndarray:
    p = len(pvalues)
    passed = np.flatnonzero(np.sort(pvalues) <= .1*np.arange(1,p+1)/(p*sum(1/k for k in range(1,p+1))))
    return pvalues <= np.sort(pvalues)[passed[-1]] if len(passed) else np.zeros(p, bool)


def audit_scores(out: Path) -> dict:
    summary, pairs = [], []
    decisions = records = 0
    for folder in sorted((ROOT/'compact_scores').glob('*/*')):
        version, case = folder.parent.name, folder.name
        original = ROOT/'results/statistical_papers'/version/case
        retained = json.loads((original/'summary.json').read_bytes())
        provenance = json.loads((folder/'provenance.json').read_bytes())
        manifest = json.loads((original/'manifest.json').read_bytes())
        cfg = manifest['config']; base = cfg.get('source_config',cfg)
        names = [r['method'] for r in retained['summary']]
        metrics = {name:{key:[] for key in ['fdp','power','empty','marginal_null_rejection']} for name in names}
        with np.load(folder/'scores.npz', allow_pickle=False) as stored_arrays:
            # NPZ is lazy: cache each stacked array once, rather than decompress
            # all 1000 repetitions again for every individual decision.
            arrays = {key:stored_arrays[key] for key in stored_arrays.files}
            for index, name in enumerate(provenance['record_order']):
                record = json.loads((original/name).read_bytes())
                assert record['replicate'] == index
                assert record['arrays_sha256'] == provenance['original_npz_sha256'][name.replace('.json','.npz')]
                pnames = record.get('pvalue_order',['wald_pilot_by','wald_naive_by','wald_oracle_by'])
                active = np.array(record['active'],int)
                null = np.setdiff1d(np.arange(base['p']), active)
                for saved in record['methods']:
                    method = saved['method']; pvalue = None
                    if saved['failed']:
                        mask = np.zeros(base['p'],bool)
                    elif method in pnames:
                        pvalue = arrays['pvalues'][index,pnames.index(method)]
                        mask = by(pvalue)
                    else:
                        kind = 0 if method.endswith('pilot') else 1
                        key = ('h'+method[1]+'_knockoff_scores') if method.startswith('h') else 'knockoff_scores'
                        mask = knockoff(arrays[key][index,kind])
                    assert np.flatnonzero(mask).tolist() == saved['selected'], (version,case,index,method)
                    values = {'fdp':float(mask[null].sum()/max(1,mask.sum())),
                              'power':float(mask[active].sum()/max(1,len(active))), 'empty':float(not mask.any())}
                    if pvalue is not None:
                        values['marginal_null_rejection'] = float((pvalue[null] <= .05).mean())
                    for key, value in values.items():
                        assert abs(value-saved[key]) < 1e-14
                        metrics[method][key].append(value)
                    decisions += 1
                records += 1
        for method in names:
            row = {'version':version,'configuration':case,'method':method,
                   'n':base['n'],'m':base['m'],'p':base['p'],'repetitions':len(provenance['record_order'])}
            reference = next(r for r in retained['summary'] if r['method'] == method)
            for metric, values in metrics[method].items():
                if values:
                    a = np.asarray(values)
                    mean, se = float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a)))
                    assert max(abs(mean-reference[metric]),abs(se-reference[metric+'_mcse'])) < 5e-15
                    row.update({metric:mean,metric+'_mcse':se})
            summary.append(row)
        for left,right in itertools.combinations(names,2):
            for metric in ['fdp','power','empty']:
                a = np.asarray(metrics[left][metric])-np.asarray(metrics[right][metric])
                mean,se = float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a)))
                ref = next(r for r in retained['paired'] if r['left']==left and r['right']==right and r['metric']==metric)
                assert max(abs(mean-ref['difference_left_minus_right']),abs(se-ref['mcse'])) < 5e-15
                pairs.append({'version':version,'configuration':case,'left':left,'right':right,'metric':metric,
                              'difference_left_minus_right':mean,'mcse':se})
    for name, rows in [('summary_recomputed.csv',summary),('paired_recomputed.csv',pairs)]:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (out/name).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    return {'SP06_records':records,'SP06_selection_decisions_recomputed':decisions,
            'SP06_summary_rows':len(summary),'SP06_paired_rows':len(pairs),
            'scope':'Saved scores and p-values; fitted covariance arrays were not recomputed.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','rebuild','replay'])
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out = ROOT/'results/minimal_runs'/stamp
    out.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {'status':'running','command':args.command,'new_independent_datasets':0,'python':sys.version}
    print(out,flush=True)
    env = dict(os.environ,PYTHONPATH=str(ROOT/'src'))

    def run(command: list[str], name: str) -> str:
        value = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        (out/(name+'.log')).write_text(value.stdout+value.stderr)
        if value.returncode:
            raise RuntimeError('Failed '+name+'; inspect '+str(out/(name+'.log')))
        return value.stdout

    try:
        for name, expected in json.loads((ROOT/'MANIFEST_SHA256.json').read_bytes()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,name
        receipt['packaged_hashes_verified'] = True
        record_hashes = json.loads((ROOT/'provenance/individual_record_hashes.json').read_bytes())
        with tarfile.open(ROOT/'individual_records.tar.gz','r:gz') as tar:
            seen = set()
            for member in tar:
                path = Path(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                assert member.name in record_hashes and member.name not in seen
                seen.add(member.name)
                stream = tar.extractfile(member)
                assert stream is not None
                data = stream.read()
                assert hashlib.sha256(data).hexdigest() == record_hashes[member.name]
                target = ROOT/path
                if target.exists():
                    assert target.read_bytes() == data, 'Refusing to replace changed record: '+member.name
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(data)
            assert seen == set(record_hashes)
        receipt['individual_record_hashes_verified'] = len(record_hashes)
        receipt['scope'] = 'Public code/results package: numerical outputs; no manuscript compilation.'
        if args.command == 'replay':
            sys.path.insert(0,str(ROOT/'src'))
            sys.path.insert(0,str(ROOT/'scripts/statistical_papers'))
            producer = module('sp06_selection',ROOT/'scripts/statistical_papers/sp06_selection.py')
            case = ROOT/'results/statistical_papers/sp06_1/harmonic_n400_m400'
            cfg = json.loads((case/'manifest.json').read_bytes())['config']
            record, arrays = producer.repetition(cfg,0)
            original = json.loads((case/'rep_0000.json').read_bytes())
            assert not record['errors']
            with np.load(ROOT/'compact_scores/sp06_1/harmonic_n400_m400/scores.npz') as saved:
                for key in ['pvalues','knockoff_scores']:
                    np.testing.assert_allclose(arrays[key],saved[key][0],rtol=1e-10,atol=1e-12)
            assert [r['selected'] for r in record['methods']] == [r['selected'] for r in original['methods']]
            receipt['recorded_seeds_regenerated'] = 1
        else:
            run([sys.executable,'-m','pytest','-q','tests/test_sp06_wald.py',
                 'tests/test_sp06_star.py','tests/test_sp06_harmonics_hc.py','tests/v10'],'tests')
            score_out = out/'score_audit'
            score_out.mkdir()
            audited = audit_scores(score_out)
            receipt.update(audited)
            (score_out/'receipt.json').write_text(json.dumps(dict(status='complete',**audited),indent=2)+'\n')
            # The historical reaggregator writes files; isolate its working copy.
            work = out/'historical_reaggregation'
            for relative in ['configs/v5','configs/v6','results/v5','results/v6','results/v10']:
                shutil.copytree(ROOT/relative,work/relative)
            old = module('historical_reaggregator',ROOT/'scripts/v10/reaggregate.py')
            old.ROOT = work
            old.reaggregate()
            receipt['v5_v6_reaggregation'] = json.loads((work/'results/v10/aggregation.json').read_bytes())
            check = run([sys.executable,'scripts/statistical_papers/sp14_compact_check.py',
                '--result','results/statistical_papers/sp14_represented_20261010T193859_53f423c7',
                '--cache','results/statistical_papers/sp12_growing_20261010T180723_6739aaf9'],'SP14_checks')
            receipt['SP14_checker_folder'] = check.splitlines()[0]
            receipt['SP14_checked_saved_outputs'] = 9
            receipt['SP14_raw_design_QR_checks'] = 3
            if args.command == 'rebuild':
                run([sys.executable,'scripts/statistical_papers/plot_sp06_evaluation.py',str(score_out)],'figure')
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='incomplete',error=repr(error))
        raise
    finally:
        receipt['elapsed_seconds'] = time.perf_counter()-started
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
