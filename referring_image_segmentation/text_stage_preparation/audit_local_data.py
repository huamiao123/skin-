"""Read-only inventory of checked local segmentation metadata; no text fabrication."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HOME_ROOT=ROOT.parent.parent
CSV_ROOTS=(HOME_ROOT/'rsi_data/ima/metadata',HOME_ROOT/'work/RSI_TailProbe_20261004/data_manifests')
TEXT_FIELDS={'text','expression','referring_expression','caption','sentence','description','prompt'}


def main():
    csvs=[]
    for folder in CSV_ROOTS:
        if not folder.exists():continue
        for p in sorted(folder.glob('*.csv')):
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            with p.open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames or [];n=sum(1 for _ in reader)
            csvs.append(dict(path=str(p),sha256=h,rows=n,columns=fields,
                candidate_text_columns=sorted(set(fields)&TEXT_FIELDS)))
    datasets=[]
    roots=[ROOT.parent/'data/isic2018',HOME_ROOT/'rsi_data/ima']+list((HOME_ROOT/'rsi_data/benchmarks').glob('*'))
    for folder in roots:
        if not folder.is_dir():continue
        images={}
        for split in ('train','val','test','external'):
            for kind in ('images','masks'):
                p=folder/split/kind
                if p.is_dir():images[f'{split}/{kind}']=sum(x.is_file() for x in p.iterdir())
        datasets.append(dict(path=str(folder),resolved_path=str(folder.resolve()),split_file_counts=images))
    # RefCOCO filename discovery is restricted to data locations, excluding source/vendor code.
    ref_candidates=[]
    for folder in (HOME_ROOT/'rsi_data',ROOT.parent/'data'):
        for pattern in ('*refcoco*','*refcocog*','refs(unc).p','refs(umd).p'):
            ref_candidates.extend(str(p) for p in folder.rglob(pattern))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),checked_csv_roots=[str(p) for p in CSV_ROOTS],
        csv_metadata=csvs,dataset_roots=datasets,refcoco_named_candidates=sorted(set(ref_candidates)),
        candidate_text_tables=[r['path'] for r in csvs if r['candidate_text_columns']],
        verified_image_expression_mask_manifest=None,
        status='No verified referring-expression manifest in inspected metadata; dataset direction pending.',
        caveats=['Inventory is scoped, not an exhaustive content search of all files.',
                 'reference_id in legacy masks is not verified natural language.',
                 'Existing benchmark paths and historic test labels do not establish an untouched test set.'])
    (ROOT/'local_data_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(tables=len(csvs),candidate_text_tables=report['candidate_text_tables'],refcoco_candidates=report['refcoco_named_candidates'],status=report['status']),ensure_ascii=False))


if __name__=='__main__':main()
