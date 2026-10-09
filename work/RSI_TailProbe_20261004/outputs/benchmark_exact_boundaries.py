"""CPU file-only exact-boundary verification; never performs model scoring."""
import csv
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rsi import metrics
from rsi.runtime import atomic_json, code_provenance, sha256_file
from tools.fast_exact_metrics import binary_mask_metrics as fast_metrics

report_path = PROJECT / 'outputs/exact_boundary_native_verification.json'
manifest = PROJECT / 'data_manifests/final_references.csv'
# These IDs were selected by image dimensions nearest 6.1MP / 29.9MP only,
# before running either metric. The first available unique reference is fixed.
fixed_ids = ['ISIC_0001208', 'ISIC_0011881']
with manifest.open() as stream:
    rows = list(csv.DictReader(stream))
selected = [next(row for row in rows if row['image_id'] == image_id and row['split'] == 'val' and row['subset'] == 'M')
            for image_id in fixed_ids]
before = code_provenance()
report = dict(status='running', GPU_compute=False, model_scoring=False,
              test_scoring_locked=True, prediction_source='fixed GT-only shift(11,-7)',
              selection='predeclared IDs selected solely by native image size nearest 6.1MP / 29.9MP',
              manifest_sha256=sha256_file(manifest), frozen_sources_before=before,
              benchmark_script_sha256=sha256_file(__file__),
              fast_tool_sha256=sha256_file(PROJECT/'tools/fast_exact_metrics.py'), cases=[])
atomic_json(report_path, report)
try:
    for row in selected:
        with Image.open(row['mask_path']) as image:
            gt = np.array(image.convert('L')) > 0
        for variant in ('shift',):
            if variant == 'shift':
                pred = np.roll(gt, (11, -7), axis=(0, 1))
            elif variant == 'erode':
                pred = binary_erosion(gt, iterations=3)
            else:
                pred = binary_dilation(gt, iterations=3)
            print(json.dumps(dict(active=row['image_id'], variant=variant, shape=gt.shape, phase='original_EDT')), flush=True)
            tick = time.perf_counter(); expected = metrics.binary_mask_metrics(pred, gt); original_seconds = time.perf_counter() - tick
            gc.collect()
            print(json.dumps(dict(active=row['image_id'], variant=variant, phase='exact_KD')), flush=True)
            tick = time.perf_counter(); actual = fast_metrics(pred, gt); fast_seconds = time.perf_counter() - tick
            mismatches = {name: [expected.get(name), actual.get(name)] for name in set(expected) | set(actual)
                          if expected.get(name) != actual.get(name)}
            case = dict(image_id=row['image_id'], reference_id=row['reference_id'],
                        mask_path=row['mask_path'], mask_sha256=sha256_file(row['mask_path']),
                        shape=list(gt.shape), megapixels=gt.size/1e6, variant=variant,
                        original_EDT_seconds=original_seconds, exact_KD_seconds=fast_seconds,
                        speedup=original_seconds/fast_seconds, every_field_exact_equal=not mismatches,
                        mismatches=mismatches, original_values=expected, exact_KD_values=actual,
                        process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            report['cases'].append(case); atomic_json(report_path, report)
            print(json.dumps({k:case[k] for k in ('image_id','variant','original_EDT_seconds','exact_KD_seconds','speedup','every_field_exact_equal')}), flush=True)
            if mismatches:
                raise AssertionError('Original EDT and exact KD differ')
            del pred; gc.collect()
        del gt; gc.collect()
    report['status'] = 'complete'
except BaseException as error:
    report.update(status='failed', error_type=type(error).__name__, error=str(error))
    raise
finally:
    report['frozen_sources_after'] = code_provenance()
    report['frozen_sources_unchanged'] = before['source_sha256'] == report['frozen_sources_after']['source_sha256']
    atomic_json(report_path, report)
