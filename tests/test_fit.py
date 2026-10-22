"""
Tests of fit/fit.py, on fake mass spectra with a Ds and a D+ signal peak
"""
import importlib.util

import numpy as np
import pytest
import ROOT
import uproot
import yaml
import zfit

from conftest import REPO_DIR

PTBINS = [2., 4., 6.]
MASS_RANGE = (1.65, 2.25)  # mass axis of the fake projections, 1 MeV/c^2 bins
N_BINS_MASS = 600
FIT_RANGE = [1.75, 2.1]
MU_DS, MU_DPLUS = 1.9683, 1.8696
SIGMA_DS, SIGMA_DPLUS = 0.011, 0.009
N_DS, N_DPLUS, N_BKG = 4000, 2500, 30000  # signal and background candidates in data
N_MC = 20000
CB_PARS = {"alpha": 1.3, "n": 3.}  # symmetric double Crystal Ball tails of the signals


def load_fit_module():
    """Import fit/fit.py, a script that is not in a package"""
    spec = importlib.util.spec_from_file_location("fit_script", REPO_DIR / "fit" / "fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit_script = load_fit_module()


def peak(rng, n_cands, mu, sigma, shape):
    """Fake mass values of a signal peak, gaussian or symmetric double Crystal Ball"""
    if shape == "gaussian":
        return rng.normal(mu, sigma, n_cands)
    obs = zfit.Space("fM", limits=MASS_RANGE)
    pdf = zfit.pdf.DoubleCB(mu, sigma, CB_PARS["alpha"], CB_PARS["n"], CB_PARS["alpha"], CB_PARS["n"], obs=obs)
    return pdf.sample(n_cands).to_numpy().flatten()


def combinatorial(rng, n_cands):
    """Fake mass values of an exponentially falling background"""
    values = FIT_RANGE[0] + rng.exponential(0.25, 3 * n_cands)
    return values[values < FIT_RANGE[1]][:n_cands]


def write_projection(file_name, spectra):
    """Write {pT bin: mass values} as hMass_<pt_min*10>_<pt_max*10> histograms, as done by project/project.py"""
    histos = {}
    for (pt_min, pt_max), values in spectra.items():
        name = f"hMass_{pt_min * 10:.0f}_{pt_max * 10:.0f}"
        histo = ROOT.TH1D(name, "", N_BINS_MASS, *MASS_RANGE)
        histo.SetDirectory(0)
        for value in values:
            histo.Fill(value)
        histos[name] = histo
    with ROOT.TFile.Open(str(file_name), "recreate") as out_file:
        for name, histo in histos.items():
            out_file.WriteTObject(histo, name)


def make_inputs(tmp_path, shape, corr_bkg_fraction=0.):
    """Write fake data and MC projections; the truth of the data is the same in every pT bin"""
    rng = np.random.default_rng(42)
    bins = list(zip(PTBINS[:-1], PTBINS[1:]))
    n_corr = int(corr_bkg_fraction * N_DPLUS)  # candidates of the correlated background in data
    # the correlated-background template is broader than the signals, and peaks below the D+ one
    corr = lambda n_cands: rng.normal(1.83, 0.03, n_cands)  # pylint: disable=unnecessary-lambda-assignment
    data = {pt_bin: np.concatenate([peak(rng, N_DS, MU_DS, SIGMA_DS, shape),
                                    peak(rng, N_DPLUS, MU_DPLUS, SIGMA_DPLUS, shape),
                                    combinatorial(rng, N_BKG), corr(n_corr)]) for pt_bin in bins}
    mc_ds = {pt_bin: peak(rng, N_MC, MU_DS, SIGMA_DS, shape) for pt_bin in bins}
    mc_dplus = {pt_bin: peak(rng, N_MC, MU_DPLUS, SIGMA_DPLUS, shape) for pt_bin in bins}
    mc_bkg = {pt_bin: corr(int(corr_bkg_fraction * N_MC)) for pt_bin in bins}
    files = {name: tmp_path / f"projections_{name}.root"
             for name in ("data", "ds_prompt", "dplus_prompt", "dplus_bkg")}
    for name, spectra in zip(files, (data, mc_ds, mc_dplus, mc_bkg)):
        write_projection(files[name], spectra)
    return files


def make_config(tmp_path, files, **fit_configs):
    """Fit configuration for the fake inputs; fit_configs override the defaults of fit_configs"""
    cutset = tmp_path / "cutset.yml"
    cutset.write_text(yaml.dump({"ptbins": PTBINS, "cuts": []}))
    out_dir = tmp_path / "fit_out"
    out_dir.mkdir()
    n_pt = len(PTBINS) - 1
    cfg = {
        "inputs": {"data": str(files["data"]),
                   "mc": [str(files["ds_prompt"]), str(files["dplus_prompt"])],
                   "mc_corr_bkg": str(files["dplus_bkg"])},
        "cutset_file_name": str(cutset),
        "fit_configs": {
            "extended": False,
            "signal_funcs": ["gaussian", "gaussian"],
            "fix_params_to_mc": [],
            "fix_ratio_dplus_ds_sigma": True,
            "bkg_funcs": ["expo"],
            "mass_limits": FIT_RANGE,
            "rebin": 4,
            "use_bkg_templ": False,
            "correlated_bkg": {"br_pdg": 1., "br_sim": 1.},
            "signal_br": {"pdg": 1., "sim": 1.},
            "fix_correlated_bkg_to_signal": True,
            **fit_configs,
        },
        "plot_style": {"n_bins": [150] * n_pt},
        "outputs": {"directory": str(out_dir), "suffix": "_test", "produce_sweights": False},
    }
    cfg_file = tmp_path / "config_fit.yml"
    cfg_file.write_text(yaml.dump(cfg))
    return cfg_file, out_dir


def read_hist(out_dir, name):
    """(values, uncertainties) of a histogram of the fit output, per pT bin"""
    with uproot.open(out_dir / "fit_mass_test.root") as out_file:
        histo = out_file[name]
        return histo.values(), np.sqrt(histo.variances())


def check_pulls(out_dir, expected, n_sigma=5):
    """The fitted values {histogram name: truth} agree with the truth within n_sigma times the fit uncertainty"""
    for name, truth in expected.items():
        values, uncs = read_hist(out_dir, name)
        assert np.all(uncs > 0), name
        assert np.all(np.abs(values - truth) < n_sigma * uncs), f"{name}: {values} +- {uncs}, expected {truth}"


@pytest.mark.parametrize("value, is_single, expected", [
    (False, lambda v: isinstance(v, bool), [False] * 3),
    ([True, False, True], lambda v: isinstance(v, bool), [True, False, True]),
    (["gaussian", "gaussian"], lambda v: all(isinstance(f, str) for f in v), [["gaussian", "gaussian"]] * 3),
    ([["a"], ["b"], ["c"]], lambda v: all(isinstance(f, str) for f in v), [["a"], ["b"], ["c"]]),
    ([1.75, 2.1], lambda v: not isinstance(v[0], (list, tuple)), [[1.75, 2.1]] * 3),
    ([[1.7, 2.1], [1.75, 2.1], [1.8, 2.1]], lambda v: not isinstance(v[0], (list, tuple)),
     [[1.7, 2.1], [1.75, 2.1], [1.8, 2.1]]),
    (4, lambda v: isinstance(v, int), [4] * 3),
])
def test_expand_per_pt(value, is_single, expected):
    """A single setting is repeated for each pT bin, per-bin settings are kept"""
    assert fit_script.expand_per_pt(value, 3, is_single) == expected


def test_expand_per_pt_independent_copies():
    """Each pT bin gets its own list: the fit modifies the background functions in place"""
    expanded = fit_script.expand_per_pt(["expo"], 3, lambda v: True)
    expanded[0].insert(0, "hist")
    assert expanded == [["hist", "expo"], ["expo"], ["expo"]]


def test_expand_per_pt_wrong_length():
    """A per-bin setting must have one entry for each pT bin"""
    with pytest.raises(ValueError, match="Expected 3 entries"):
        fit_script.expand_per_pt([True, False], 3, lambda v: isinstance(v, bool))


def test_fit_gaussian_signals(tmp_path):
    """Raw yields, masses and widths of the fake Ds and D+ peaks are recovered in each pT bin"""
    cfg_file, out_dir = make_config(tmp_path, make_inputs(tmp_path, "gaussian"))
    fit_script.fit(str(cfg_file))

    check_pulls(out_dir, {"h_rawyields": N_DS, "h_means": MU_DS, "h_sigmas": SIGMA_DS})
    # the MC fits give the peak positions and widths used as starting values
    expected_mc = {"h_means_mc_ds": MU_DS, "h_means_mc_dplus": MU_DPLUS,
                   "h_sigmas_mc_ds": SIGMA_DS, "h_sigmas_mc_dplus": SIGMA_DPLUS}
    for name, truth in expected_mc.items():
        values, _ = read_hist(out_dir, name)
        assert values.shape[0] == len(PTBINS) - 1
        assert np.allclose(values, truth, rtol=2e-2), name
    for name in ("fit_mass_pt20_40.pdf", "fit_mass_pt40_60.pdf", "fit_mass_pt20_40_mc.pdf"):
        assert (out_dir / name).is_file()


def test_fit_per_pt_settings(tmp_path):
    """Settings given for each pT bin (here the rebinning and the fit range) are used bin by bin"""
    cfg_file, out_dir = make_config(tmp_path, make_inputs(tmp_path, "gaussian"),
                                    rebin=[4, 2], mass_limits=[FIT_RANGE, [1.78, 2.08]])
    fit_script.fit(str(cfg_file))
    check_pulls(out_dir, {"h_rawyields": N_DS})


def test_fit_params_fixed_to_mc(tmp_path, monkeypatch):
    """alpha and n of the Crystal Ball tails are fixed to the MC fit values, with a correlated-background template"""
    fixed = []  # (pT bin of the call, signal index, parameter, value) of the parameters fixed in the fits
    original = fit_script.F2MassFitter.set_signal_initpar

    def spy(self, idx, par_name, init_value, **kwargs):
        if kwargs.get("fix"):
            fixed.append((idx, par_name, init_value))
        return original(self, idx, par_name, init_value, **kwargs)

    monkeypatch.setattr(fit_script.F2MassFitter, "set_signal_initpar", spy)
    cfg_file, out_dir = make_config(
        tmp_path, make_inputs(tmp_path, "doublecb", corr_bkg_fraction=0.3),
        signal_funcs=["doublecbsymm", "doublecbsymm"], fix_params_to_mc=["alpha", "n"],
        use_bkg_templ=True)
    fit_script.fit(str(cfg_file))

    check_pulls(out_dir, {"h_rawyields": N_DS, "h_means": MU_DS}, n_sigma=6)
    # 2 signals x 2 parameters in each of the 2 pT bins, fixed to the (well measured) generated tails of the MC
    assert sorted((idx, par_name) for idx, par_name, _ in fixed) == sorted(
        [(idx, par_name) for idx in (0, 1) for par_name in ("alpha", "n")] * (len(PTBINS) - 1))
    for _, par_name, value in fixed:
        assert value == pytest.approx(CB_PARS[par_name], rel=0.3)
