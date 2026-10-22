"""
Tests of cross_section/get_cross_section.py
"""
import subprocess
import sys
from array import array

import numpy as np
import pytest
import ROOT
import yaml

from conftest import REPO_DIR

PT_EDGES = [1., 2., 4., 8.]  # bins of different widths: the cross section is a density in pT
LUMI = 4.0e6  # mub^-1
BR = {"ds": 0.0225, "dplus": 0.00267}
RAW_YIELDS = {"ds": ([1000., 4000., 3000.], [40., 80., 60.]), "dplus": ([500., 2500., 900.], [30., 60., 40.])}
EFFICIENCY = {"Ds": ([0.01, 0.05, 0.10], [0.001, 0.002, 0.004]), "Dplus": ([0.02, 0.06, 0.12], [0.002, 0.003, 0.005])}


def make_histo(name, contents, uncs):
    """Histogram in the pT bins of the test"""
    histo = ROOT.TH1D(name, "", len(PT_EDGES) - 1, array("d", PT_EDGES))
    histo.SetDirectory(0)
    for i_bin, (content, unc) in enumerate(zip(contents, uncs), start=1):
        histo.SetBinContent(i_bin, content)
        histo.SetBinError(i_bin, unc)
    return histo


def write_histos(file_name, histos):
    """Write histograms to a new ROOT file"""
    with ROOT.TFile.Open(str(file_name), "recreate") as out_file:
        for histo in histos:
            out_file.WriteTObject(histo, histo.GetName())


def run_cross_section(tmp_path):
    """Write fake raw yields and efficiencies, run get_cross_section.py; return the output file name"""
    raw_yields = tmp_path / "raw_yields.root"
    efficiency = tmp_path / "efficiency.root"
    h_lumi = ROOT.TH1D("hLuminosity", "", 1, 0., 1.)  # the luminosity is in the output of the fit, with the yields
    h_lumi.SetDirectory(0)
    h_lumi.SetBinContent(1, LUMI)
    write_histos(raw_yields, [h_lumi] + [make_histo(f"h_rawyields_{particle}", *values)
                                         for particle, values in RAW_YIELDS.items()])
    effs = [make_histo(f"h_efficiency_{particle}_Prompt", *values) for particle, values in EFFICIENCY.items()]
    # the non-prompt efficiencies are not used for the prompt cross section
    effs += [make_histo(f"h_efficiency_{particle}_NonPrompt", [0.5] * 3, [0.01] * 3) for particle in EFFICIENCY]
    write_histos(efficiency, effs)

    cfg = {
        "inputs": {"raw_yields": str(raw_yields), "efficiency": str(efficiency)},
        "br": {"ds": {"value": BR["ds"]}, "dplus": {"value": BR["dplus"]}},
        "output_dir": str(tmp_path / "out"),
    }
    cfg_file = tmp_path / "config_cross_section.yml"
    cfg_file.write_text(yaml.dump(cfg))
    result = subprocess.run([sys.executable, str(REPO_DIR / "cross_section" / "get_cross_section.py"), str(cfg_file)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return tmp_path / "out" / "cross_section.root"


@pytest.mark.parametrize("particle, eff_key", [("ds", "Ds"), ("dplus", "Dplus")])
def test_cross_section(tmp_path, particle, eff_key):
    """dsigma/dpT = N / (Acc x eff x 2 x BR x L x width), with N and eff uncertainties added in quadrature"""
    out_file_name = run_cross_section(tmp_path)

    counts, count_uncs = (np.array(values) for values in RAW_YIELDS[particle])
    effs, eff_uncs = (np.array(values) for values in EFFICIENCY[eff_key])
    widths = np.diff(PT_EDGES)
    expected = counts / effs / (2. * BR[particle] * LUMI * widths)
    expected_unc = expected * np.sqrt((count_uncs / counts) ** 2 + (eff_uncs / effs) ** 2)

    with ROOT.TFile.Open(str(out_file_name)) as out_file:
        histo = out_file.Get(f"h_cross_section_{particle}")
        assert [histo.GetXaxis().GetBinLowEdge(i) for i in range(1, histo.GetNbinsX() + 2)] == PT_EDGES
        values = np.array([histo.GetBinContent(i) for i in range(1, histo.GetNbinsX() + 1)])
        uncs = np.array([histo.GetBinError(i) for i in range(1, histo.GetNbinsX() + 1)])
    assert values == pytest.approx(expected, rel=1e-9)
    assert uncs == pytest.approx(expected_unc, rel=1e-9)
