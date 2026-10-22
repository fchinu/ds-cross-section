"""
Tests of utils/sparse_reweighter.py
"""
import numpy as np
import pytest

from helpers import (AXES, AXES_CFG, PT_WEIGHTS, PTB_WEIGHTS, make_sparse, random_points, sparse_bins, weight,
                     write_weights)
from utils.sparse_reweighter import SparseReweighter


@pytest.fixture(name="reweighter")
def fixture_reweighter(tmp_path):
    """Reweighter for Ds with the PT_WEIGHTS (prompt) and PTB_WEIGHTS (non-prompt) histograms"""
    file_name = tmp_path / "ptweights.root"
    write_weights(file_name, {"weights/hist_prompt": PT_WEIGHTS, "weights/hist_nonprompt": PTB_WEIGHTS})
    cfg = {"file_name": str(file_name),
           "hist_names": {"Ds": {"Prompt": "weights/hist_prompt", "NonPrompt": "weights/hist_nonprompt"}}}
    return SparseReweighter(cfg, AXES_CFG)


@pytest.mark.parametrize("origin, kind", [("Prompt", "reco"), ("Prompt", "gen"),
                                          ("NonPrompt", "reco"), ("NonPrompt", "gen")])
def test_weights_follow_pt_or_ptb(reweighter, origin, kind):
    """Each bin is scaled by the weight at its D pT (prompt) or pT(B) (non-prompt), and its error alike"""
    axes = AXES[f"{kind}_{origin.lower()}"]
    sparse = make_sparse("sparse", axes, random_points(axes, 500, np.random.default_rng(1)))
    variable, spec = ("pt", PT_WEIGHTS) if origin == "Prompt" else ("ptb", PTB_WEIGHTS)
    i_axis = AXES_CFG[f"axis_{variable}_{kind}"]

    before = sparse_bins(sparse)
    after = sparse_bins(reweighter.reweight(sparse, "Ds", origin, kind == "gen"))

    assert after.keys() == before.keys()
    for coord, (content, _) in before.items():
        w = weight(spec, sparse.GetAxis(i_axis).GetBinCenter(int(coord[i_axis])))
        assert after[coord][0] == pytest.approx(content * w)
        assert after[coord][1] == pytest.approx(content * w**2)  # unit-weight fills: error^2 = content


def test_first_bin_is_reweighted(reweighter):
    """Linear bin 0 is reweighted too: with cppyy's overload choice it used to segfault"""
    sparse = make_sparse("sparse", AXES["reco_prompt"], [[1.995, 4.5, 50., 0.05, 0.5, 0.5, 50.]])
    reweighted = reweighter.reweight(sparse, "Ds", "Prompt", False)
    assert reweighted.GetNbins() == 1
    assert reweighted.GetBinContent(0, np.zeros(7, dtype=np.int32)) == pytest.approx(weight(PT_WEIGHTS, 4.5))


def test_weights_by_value_on_pt_bin_projection(reweighter):
    """On a pT-bin projection the pT bin index does not match the weight histogram: the weight follows the value"""
    axes = AXES["reco_prompt"]
    sparse = make_sparse("sparse", axes, random_points(axes, 500, np.random.default_rng(2)))
    sparse.GetAxis(AXES_CFG["axis_pt_reco"]).SetRangeUser(5., 8.)
    projection = sparse.Projection(len(axes), np.arange(len(axes), dtype=np.int32))
    pt_axis = projection.GetAxis(AXES_CFG["axis_pt_reco"])

    before = sparse_bins(projection)
    after = sparse_bins(reweighter.reweight(projection, "Ds", "Prompt", False))

    assert before
    for coord, (content, _) in before.items():
        pt = pt_axis.GetBinCenter(int(coord[AXES_CFG["axis_pt_reco"]]))
        assert 5. < pt < 8.
        assert after[coord][0] == pytest.approx(content * weight(PT_WEIGHTS, pt))


def test_input_sparse_unchanged(reweighter):
    """reweight returns a copy owned by Python and leaves the input sparse untouched"""
    axes = AXES["gen_nonprompt"]
    sparse = make_sparse("sparse", axes, random_points(axes, 200, np.random.default_rng(3)))
    before = sparse_bins(sparse)
    reweighted = reweighter.reweight(sparse, "Ds", "NonPrompt", True)
    assert sparse_bins(sparse) == before
    assert reweighted.__python_owns__


def test_missing_weight_histogram(tmp_path):
    """A weight histogram missing from the file raises a KeyError naming it"""
    file_name = tmp_path / "ptweights.root"
    write_weights(file_name, {"hist_prompt": PT_WEIGHTS})
    cfg = {"file_name": str(file_name), "hist_names": {"Ds": {"Prompt": "hist_prompt", "NonPrompt": "hist_missing"}}}
    with pytest.raises(KeyError, match="hist_missing"):
        SparseReweighter(cfg, AXES_CFG)


def test_missing_axis(reweighter):
    """Weighting vs pT(B) a sparse without the pT(B) axis raises a ValueError"""
    axes = AXES["reco_prompt"]  # no pT(B) axis
    sparse = make_sparse("sparse", axes, random_points(axes, 10, np.random.default_rng(4)))
    with pytest.raises(ValueError, match="axis_ptb_reco"):
        reweighter.reweight(sparse, "Ds", "NonPrompt", False)
