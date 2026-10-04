"""Hand-calculated CPU checks for the descriptive threshold sidecar."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('threshold_sidecar', Path(__file__).with_name('source_threshold_sensitivity.py'))
sidecar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidecar)


def reference(image, index, dice_gain, loss_gain, changed=True, tool='T1'):
    return dict(protocol_id='P2-IMA-M-v2', seed=17, run_id='B', checkpoint_hash='checkpoint',
                image_id=image, subset='M', reference_id=f'{image}-r{index}', action=1, alpha=1,
                **{'lambda': 0}, dice=.8+dice_gain, dice_anchor=.8, loss=.4-loss_gain,
                loss_anchor=.4, loss_view='canonical_letterbox_valid', changed_pixel_fraction=.01 if changed else 0,
                tool=tool, split='val', gain_dice=dice_gain, gain_loss=loss_gain)


def examples():
    # A has conflicting gains, B has smaller conflicting gains, C is unchanged.
    # Counts are 2/3/2 references, so flat-reference averaging is incorrect.
    return [reference('A', 0, -.02, .03), reference('A', 1, .01, -.02, tool='T3'),
            reference('B', 0, -.004, -.01), reference('B', 1, -.004, .01, tool='T2'),
            reference('B', 2, .002, 0, tool='T3'),
            reference('C', 0, 0, .01, False), reference('C', 1, 0, .01, False, tool='T3')]


class HandCalculatedSensitivity(unittest.TestCase):
    def test_macro_harm_gains_and_three_flip_denominators(self):
        result = sidecar.summarize(examples(), .005)
        self.assertAlmostEqual(result['G_plus'], 17/9000)
        self.assertAlmostEqual(result['H_minus'], 19/4500)
        self.assertAlmostEqual(result['H_epsilon'], 1/400)
        self.assertAlmostEqual(result['mean_gain_dice'], -7/3000)
        self.assertAlmostEqual(result['mean_gain_dice'], result['G_plus']-result['H_minus'])
        self.assertAlmostEqual(result['any_harm'], 1/3)
        self.assertEqual(result['sign_flip_count'], 1)
        self.assertEqual(result['sign_flip_all_image_denominator'], 3)
        self.assertAlmostEqual(result['sign_flip'], 1/3)
        self.assertEqual(result['sign_flip_changed_denominator'], 2)
        self.assertAlmostEqual(result['sign_flip_changed_rate'], 1/2)
        self.assertEqual(result['sign_flip_exceed_denominator'], 1)
        self.assertAlmostEqual(result['sign_flip_exceed_rate'], 1)
        self.assertAlmostEqual(result['canonical_rho'], 5/18)
        self.assertAlmostEqual(result['canonical_J_mean'], 1/225)
        self.assertAlmostEqual(result['canonical_kappa_pm'], 2/3)
        self.assertAlmostEqual(result['worst_reference_mean'], -.008)
        self.assertAlmostEqual(result['worst_tail_10pct'], -.02)
        self.assertEqual(result['worst_tail_n'], 1)

    def test_strict_predeclared_tolerances_and_disagreement(self):
        expected = {0: (2/3, 2/3, 19/4500), .002: (2/3, 1/3, 31/9000),
                    .005: (1/3, 1/3, 1/400), .01: (1/3, 0, 1/600)}
        for epsilon, (any_harm, flip, harm) in expected.items():
            result = sidecar.summarize(examples(), epsilon)
            self.assertAlmostEqual(result['any_harm'], any_harm)
            self.assertAlmostEqual(result['sign_flip'], flip)
            self.assertAlmostEqual(result['H_epsilon'], harm)
        result = sidecar.summarize(examples(), .005)
        self.assertEqual(result['soft_harm_reference_count'], 2)
        self.assertEqual(result['hard_harm_reference_count'], 1)
        self.assertEqual(result['both_harm_reference_count'], 0)
        self.assertEqual(result['soft_hard_harm_disagreement_reference_count'], 3)
        self.assertAlmostEqual(result['soft_hard_harm_disagreement_macro_rate'], 4/9)
        self.assertEqual(result['soft_hard_harm_disagreement_image_count'], 2)
        masses = [value for key, value in result.items() if key.endswith('_macro_mass')]
        self.assertAlmostEqual(sum(masses), 1)

    def test_reference_set_copy_preserves_macro_metrics(self):
        rows = examples()
        copies = [dict(row, reference_id=row['reference_id']+'-copy') for row in rows]
        before, after = sidecar.summarize(rows, .005), sidecar.summarize(rows+copies, .005)
        for key in ['G_plus', 'H_epsilon', 'mean_gain_dice', 'any_harm', 'sign_flip',
                    'soft_hard_harm_disagreement_macro_rate']:
            self.assertAlmostEqual(before[key], after[key])
        self.assertEqual(after['n_references'], 2*before['n_references'])

    def test_single_reference_tool_has_no_eligible_conflict_denominator(self):
        row = reference('one', 0, -.02, .03)
        result = sidecar.summarize([row], .005)
        self.assertEqual(result['sign_flip_all_image_rate'], 0)
        self.assertEqual(result['n_single_reference_images'], 1)
        self.assertIsNone(result['sign_flip'])
        self.assertIsNone(result['sign_flip_changed_rate'])
        self.assertIsNone(result['sign_flip_exceed_rate'])
        self.assertIsNone(result['weighted_reference_gain_pearson'])

    def test_gain_correlations_and_ties(self):
        x, y, weights = [-1, -1, 1], [2, 2, -2], [.25, .25, .5]
        self.assertAlmostEqual(sidecar.correlation(x, y, weights), -1)
        self.assertAlmostEqual(sidecar.correlation(x, y, weights, ranks=True), -1)
        self.assertIsNone(sidecar.correlation([1, 1], [0, 1]))

    def test_file_hash_binding_and_original_csv_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'manifest.csv'; manifest.write_bytes(b'frozen cohort')
            config = root/'config.json'; config.write_text(json.dumps({'threshold': .5, 'loss': {'epsilon_L': 0}, 'manifest': str(manifest)}))
            rows = examples()
            for row in rows:
                row['manifest_hash'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            source = root/'B.csv'
            with source.open('w', newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            original = source.read_bytes()
            result = sidecar.run([source], root/'analysis', config)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['inputs'][0]['sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(result['configuration']['epsilon_D'], [0, .002, .005, .01])
            self.assertEqual(result['configuration']['threshold'], .5)
            self.assertEqual(result['output_sha256'], hashlib.sha256(Path(result['output_csv']).read_bytes()).hexdigest())
            with Path(result['output_csv']).open() as handle:
                output = list(csv.DictReader(handle))
            self.assertEqual(len(output), 20)  # native, three tools, native fixed-two × four tolerances
            fixed = sidecar.fixed_two(rows)
            self.assertEqual(len(fixed), 6)
            self.assertEqual(sorted(r['reference_id'] for r in fixed),
                             sorted(r['reference_id'] for r in sidecar.fixed_two(list(reversed(rows)))))

    def test_reject_duplicate_noncanonical_and_test_exports(self):
        for modification in ['duplicate', 'loss_view', 'test_split']:
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory)/'invalid.csv'
                rows = examples()
                if modification == 'duplicate': rows.append(copy.copy(rows[0]))
                if modification == 'loss_view': rows[0]['loss_view'] = 'original'
                if modification == 'test_split': rows[0]['split'] = 'test'
                with source.open('w', newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                with self.assertRaises(ValueError): sidecar.read_exports([source])


if __name__ == '__main__':
    unittest.main()
