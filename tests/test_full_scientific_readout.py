"""Small CPU checks for the new readout's additional statistical contracts."""
from copy import deepcopy

import numpy as np
import pytest

from tools import full_scientific_readout as readout


def paired_rows():
    a, b = [], []
    for image, group, gains, boundary in (("a", "shared", (-.01, .01), .1),
                                         ("b", "shared", (-.01, -.01), .2),
                                         ("c", "other", (.02, .02), .3)):
        for index, gain in enumerate(gains):
            base = dict(image_id=image, reference_id=str(index), group_id=group,
                        dice_anchor=.5, dice=.5, **{field: 0. for field in readout.BOUNDARY})
            b.append(base)
            a.append(dict(base, dice=.5 + gain, **{field: boundary for field in readout.BOUNDARY}))
    return a, b


def test_incidence_and_boundary_resample_whole_group_with_image_denominator():
    a, b = paired_rows()
    actual = readout.sidecar_comparison(a, b, repeats=2, draws=np.array([[0, 0], [1, 1]]))
    assert actual["bootstrap_units"] == 2 and actual["n_images"] == 3
    assert actual["any_harm_rsi_count"] == 2 and actual["sign_flip_rsi_count"] == 1
    assert actual["any_harm_difference"] == pytest.approx(2/3)
    assert actual["sign_flip_difference"] == pytest.approx(1/3)
    assert actual["any_harm_ci_low"] == pytest.approx(.025)
    assert actual["any_harm_ci_high"] == pytest.approx(.975)
    assert actual["sign_flip_ci_high"] == pytest.approx(.4875)
    assert actual["bf1_difference"] == pytest.approx(.2)
    assert actual["bf1_ci_low"] == pytest.approx(.15375)
    assert actual["bf1_ci_high"] == pytest.approx(.29625)


def test_pairing_rejects_changed_reference_anchor_and_single_reference_cohort():
    a, b = paired_rows()
    broken = deepcopy(b); broken[0]["dice_anchor"] = .4
    with pytest.raises(ValueError, match="anchors differ"):
        readout.sidecar_comparison(a, broken)
    with pytest.raises(ValueError, match="multiple references"):
        readout.sidecar_comparison(a[:1], b[:1])


def test_applied_message_scale_preserves_input_raw_and_zero_endpoint():
    original = [dict(run_id=run, subset="M", image_id="a", message_rms=raw,
                     alpha=alpha, action=alpha) for run, raw, alpha in
                (("Shrink-D0-0.000000", .6, 0.), ("Shrink-D0-0.666667", .6, 2/3),
                 ("RSI-1", .5, 1.), ("Anchor", 0., 0.))]
    saved = deepcopy(original)
    derived = readout.applied_message_rows(original)
    assert original == saved
    assert [r["applied_message_rms"] for r in derived] == pytest.approx([0., .4, .5, 0.])
    assert derived[0]["raw_message_rms"] == .6


def test_validate_only_missing_matrix_creates_no_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(readout, "PROJECT", tmp_path)
    output = tmp_path / "outputs/seed17"
    result = readout.run(output, tmp_path / "runs", validate_only=True)
    assert result["status"] == "not_ready" and result["outputs_written"] is False
    assert not output.exists()


def test_undefined_statistics_are_null_without_relabeling_metadata_or_zero():
    row = {"image_id": "nan", "stratum": "", "sign_flip": "", "weighted_reference_gain_pearson": "nan",
           "sign_flip_count": "0", "any_harm": "0.0"}
    result = readout.csv_rows_for_json([row])[0]
    assert result["image_id"] == "nan" and result["stratum"] == ""
    assert result["sign_flip"] is None and result["weighted_reference_gain_pearson"] is None
    assert result["sign_flip_count"] == "0" and result["any_harm"] == "0.0"
    assert row["sign_flip"] == ""
