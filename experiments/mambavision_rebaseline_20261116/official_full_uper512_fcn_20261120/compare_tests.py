from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
records = []
for ds in ('ISIC2018', 'PH2', 'Waterloo', 'HAM10000'):
    folder = ROOT / ds
    results = []
    ids = None
    for label in ('epoch0_test', 'final_test'):
        record = json.loads((folder / (label + '.json')).read_text())
        path = folder / (label + '_per_image.csv')
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == record['per_image_sha256']
        with path.open() as f:
            rows = list(csv.DictReader(f))
        current = [r['image_id'] for r in rows]
        assert ids is None or current == ids
        ids = current
        assert len(ids) == len(set(ids)) == record['n']
        assert abs(statistics.mean(float(r['foreground_dice']) for r in rows) - record['test_macro_dice']) < 1e-12
        results.append(record)
    start, end = results
    assert start['split_sha256'] == end['split_sha256']
    assert start['model_sha256'] == json.loads((ROOT / 'initialization.json').read_text())['initial_sha256']
    fit = json.loads((folder / 'fit_complete.json').read_text())
    assert end['model_sha256'] == fit['sha256']
    with (folder / fit['best_checkpoint']).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == end['model_sha256']
    records.append([ds,len(ids),start['test_macro_dice']*100,end['test_macro_dice']*100,
                    (end['test_macro_dice']-start['test_macro_dice'])*100,fit['best_epoch'],fit['epochs']])
output = ROOT / 'four_dataset_test_comparison.csv'
tmp = output.with_suffix('.tmp')
with tmp.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['dataset','test_images','epoch0_test_dice_percent','fine_tuned_test_dice_percent',
                     'change_percentage_points','selected_epoch','epochs_trained'])
    writer.writerows(records)
tmp.replace(output)
print('All four paired test comparisons verified', flush=True)
