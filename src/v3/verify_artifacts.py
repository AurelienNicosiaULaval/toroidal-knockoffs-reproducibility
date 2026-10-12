"""Check every numerical table row against source CSV, plus execution provenance."""
from __future__ import annotations
import json,re
import numpy as np
import pandas as pd
from provenance import ROOT,atomic_json,digest
from v3.report import ORDER,LABELS


def main():
    table=pd.read_csv(ROOT/'results/v3/aggregate/summary.csv')
    note=(ROOT/'manuscript/obstacles_v3.tex').read_text()
    lines=[line for line in note.splitlines() if re.match(r'^(200|800|3200) & ',line)]
    if len(lines)!=24:raise AssertionError('Expected 24 numerical rows')
    count=0
    for response in ['linear','nonlinear_omitted']:
        for n in [200,800,3200]:
            for method in ORDER:
                row=table[(table.response==response)&(table.n==n)&(table.method==method)].iloc[0]
                cells=[cell.strip() for cell in lines[count].removesuffix('\\\\').split('&')]
                expected=[str(n),LABELS[method],f"{row['fdp']:.4f} ({row['fdp_mcse']:.4f})",
                          f"{row['power']:.4f} ({row['power_mcse']:.4f})",f"{row['empty']:.3f}"]
                if cells!=expected:raise AssertionError(f'Table row {count} differs from CSV: {cells} vs {expected}')
                count+=1
    principal=json.loads((ROOT/'results/v3/aggregate/audit.json').read_text())
    if principal['successful_datasets']!=3000 or principal['method_response_evaluations']!=24000 or principal['failures']:
        raise AssertionError('Principal experiment counts')
    for name,sha in principal['individual_record_sha256'].items():
        if digest(ROOT/name)!=sha:raise AssertionError('Individual record changed')
    sign=ROOT/'results/v3/v3_1_sign_validation';manifest=json.loads((sign/'manifest.json').read_text())
    for source,sha in manifest['code'].items():
        if digest(sign/'source_snapshot'/source)!=sha:raise AssertionError('Sign-study source changed')
    records=[json.loads(f.read_text()) for f in (sign/'records').glob('*.json')]
    summary=json.loads((sign/'summary.json').read_text())
    if len(records)!=1000 or summary['completed']!=1000:raise AssertionError('Sign-study counts')
    for saved in summary['summary']:
        rows=[r for record in records for r in record['results'] if r['signals']==saved['signals']]
        if sum(r['nonempty'] for r in rows)!=saved['nonempty_count']:raise AssertionError('Selection count')
        if abs(np.mean([r['fdp'] for r in rows])-saved['fdr'])>1e-14:raise AssertionError('FDR mean')
    log=(ROOT/'manuscript/obstacles_v3.log').read_text(errors='replace')
    if re.search(r'Overfull|undefined|multiply defined|^!',log,re.M):raise AssertionError('Unresolved TeX diagnostic')
    from pypdf import PdfReader
    reader=PdfReader(ROOT/'manuscript/obstacles_v3.pdf')
    if not reader.pages or any(not (page.extract_text() or '').strip() for page in reader.pages):raise AssertionError('Empty PDF page')
    atomic_json(ROOT/'results/v3/validation/artifacts.json',dict(table_rows_checked=count,
        principal_records_checked=3000,sign_records_checked=1000,pdf_pages=len(reader.pages),
        pdf_sha256=digest(ROOT/'manuscript/obstacles_v3.pdf'),status='passed'))
    print('Artifact checks passed:',count,'table rows, 4000 individual datasets,',len(reader.pages),'PDF pages')


if __name__=='__main__':main()
