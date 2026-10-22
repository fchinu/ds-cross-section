"""
Convert HEPData ROOT files to a single ROOT file with histograms and graphs for Run 2 cross section results.
"""
from pathlib import Path
import ROOT

ROOT.TH1.AddDirectory(False)

def process_run2_results(particle):
    """Get the cross section results from Run 2"""
    folder = ""
    if particle == "ds":
        folder = "Figure 4 Ds pp 13 TeV"
    elif particle == "dplus":
        folder = "Figure 4 Dplus pp 13 TeV"

    with ROOT.TFile.Open(str(Path("data") / "hepdata" / f"{particle}_cross_section_13_tev_alice.root")) as in_file:
        h_hepd_central = in_file[f"{folder}/Hist1D_y1"]
        h_hepd_stat = in_file[f"{folder}/Hist1D_y1_e1"]
        h_hepd_syst_high = in_file[f"{folder}/Hist1D_y1_e2plus"]
        h_hepd_syst_low = in_file[f"{folder}/Hist1D_y1_e2minus"]
        h_hepd_syst_br = in_file[f"{folder}/Hist1D_y1_e3"]
        h_hepd_syst_lumi = in_file[f"{folder}/Hist1D_y1_e4"]

    h_stat = h_hepd_central.Clone(f"h_stat_{particle}")
    g_syst = ROOT.TGraphAsymmErrors(h_stat)
    g_syst_no_lumi = ROOT.TGraphAsymmErrors(h_stat)
    g_syst_no_br = ROOT.TGraphAsymmErrors(h_stat)
    g_syst_no_lumi_no_br = ROOT.TGraphAsymmErrors(h_stat)
    g_syst.SetName(f"g_syst_{particle}")
    g_syst_no_lumi.SetName(f"g_syst_no_lumi_{particle}")
    g_syst_no_br.SetName(f"g_syst_no_br_{particle}")
    g_syst_no_lumi_no_br.SetName(f"g_syst_no_lumi_no_br_{particle}")

    for i_bin in range(1, h_stat.GetNbinsX() + 1):
        stat = h_hepd_stat.GetBinContent(i_bin)
        syst_high = h_hepd_syst_high.GetBinContent(i_bin)
        syst_low = h_hepd_syst_low.GetBinContent(i_bin)
        lumi_unc = h_hepd_syst_lumi.GetBinContent(i_bin)
        br_unc = h_hepd_syst_br.GetBinContent(i_bin)
        h_stat.SetBinError(i_bin, stat)
        g_syst.SetPointEYhigh(i_bin - 1, (stat ** 2 + syst_high ** 2 + lumi_unc ** 2 + br_unc ** 2) ** 0.5)
        g_syst.SetPointEYlow(i_bin - 1, (stat ** 2 + syst_low ** 2 + lumi_unc ** 2 + br_unc ** 2) ** 0.5)
        g_syst_no_lumi.SetPointEYhigh(i_bin - 1, (stat ** 2 + syst_high ** 2 + br_unc ** 2) ** 0.5)
        g_syst_no_lumi.SetPointEYlow(i_bin - 1, (stat ** 2 + syst_low ** 2 + br_unc ** 2) ** 0.5)
        g_syst_no_br.SetPointEYhigh(i_bin - 1, (stat ** 2 + syst_high ** 2 + lumi_unc ** 2) ** 0.5)
        g_syst_no_br.SetPointEYlow(i_bin - 1, (stat ** 2 + syst_low ** 2 + lumi_unc ** 2) ** 0.5)
        g_syst_no_lumi_no_br.SetPointEYhigh(i_bin - 1, (stat ** 2 + syst_high ** 2) ** 0.5)
        g_syst_no_lumi_no_br.SetPointEYlow(i_bin - 1, (stat ** 2 + syst_low ** 2) ** 0.5)

    return h_stat, g_syst, g_syst_no_lumi, g_syst_no_br, g_syst_no_lumi_no_br

def get_run2_results():
    """Get the cross section results from Run 2"""

    outputs_ds = process_run2_results("ds")
    outputs_dplus = process_run2_results("dplus")

    with ROOT.TFile.Open(str(Path("data") / "run2_cross_section.root"), "recreate"):
        for hist in outputs_ds + outputs_dplus:
            hist.Write()

if __name__ == "__main__":
    get_run2_results()
