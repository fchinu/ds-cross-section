"""
Draw the Run 3 cross sections together with the Run 2 results, and their ratio.
"""
import argparse
from array import array
from pathlib import Path

import numpy as np
import ROOT

ROOT.TH1.AddDirectory(False)
REPO_DIR = Path(__file__).resolve().parents[1]
RUN3_COLOR, RUN2_COLOR = ROOT.kAzure + 4, ROOT.kRed + 1
PT_RANGE = (0., 24.)
RATIO_RANGE = (0.4, 1.6)
Y_TITLE = "d#sigma/d#it{p}_{T} (mb/(GeV/#it{c}))"


def get_run2_result(particle):
    """Statistical (histogram) and systematic (graph, without luminosity) uncertainties of the Run 2 result"""
    with ROOT.TFile.Open(str(REPO_DIR / "data" / "run2_cross_section.root")) as in_file:
        h_stat = in_file[f"h_stat_{particle}"]
        g_syst_no_lumi = in_file[f"g_syst_no_lumi_{particle}"]

    return h_stat, g_syst_no_lumi


def style(h, color):
    """Marker and line style of a histogram or graph"""
    h.SetLineColor(color)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(ROOT.kFullCircle)
    h.SetMarkerSize(1.2)


def get_bin_edges(histo):
    """Bin edges of a histogram"""
    axis = histo.GetXaxis()
    return [axis.GetBinLowEdge(i_bin) for i_bin in range(1, histo.GetNbinsX() + 2)]


def get_common_edges(histo_a, histo_b):
    """Bin edges present in both histograms: the binning on which they can be compared"""
    edges_a, edges_b = get_bin_edges(histo_a), get_bin_edges(histo_b)
    return [edge for edge in edges_a if np.isclose(edges_b, edge).any()]


def rebin_density(histo, edges):
    """
    Rebin a histogram of densities (e.g. d sigma / d pT) to the given edges, which must be edges of the histogram.
    The content of a merged bin is the average of the merged ones weighted by their widths, and their
    uncertainties, assumed uncorrelated, are added in quadrature with the same weights.
    """
    old_edges = get_bin_edges(histo)
    for edge in edges:
        if not np.isclose(old_edges, edge).any():
            raise ValueError(f"{edge} is not a bin edge of {histo.GetName()}")

    h_rebin = ROOT.TH1D(f"{histo.GetName()}_rebin", "", len(edges) - 1, array("d", edges))
    for i_new, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        content, variance = 0., 0.
        for i_old in range(1, histo.GetNbinsX() + 1):
            if low - 1e-6 <= old_edges[i_old - 1] and old_edges[i_old] <= high + 1e-6:
                width = histo.GetBinWidth(i_old)
                content += histo.GetBinContent(i_old) * width
                variance += (histo.GetBinError(i_old) * width) ** 2
        h_rebin.SetBinContent(i_new, content / (high - low))
        h_rebin.SetBinError(i_new, np.sqrt(variance) / (high - low))
    return h_rebin


def get_relative_syst(g_syst, pt_min, pt_max):
    """Systematic uncertainties of the points of a graph between pT min and max, relative to the points, around 1"""
    g_rel = ROOT.TGraphAsymmErrors()
    for i_point in range(g_syst.GetN()):
        x, y = g_syst.GetPointX(i_point), g_syst.GetPointY(i_point)
        if not pt_min < x < pt_max:
            continue
        i_rel = g_rel.GetN()
        g_rel.SetPoint(i_rel, x, 1.)
        g_rel.SetPointError(i_rel, g_syst.GetErrorXlow(i_point), g_syst.GetErrorXhigh(i_point),
                            g_syst.GetErrorYlow(i_point) / y, g_syst.GetErrorYhigh(i_point) / y)
    return g_rel


def style_axes(frame, label_size, title_size, y_offset):
    """Font sizes of the axes of a frame, scaled to the height of the pad"""
    for axis in (frame.GetXaxis(), frame.GetYaxis()):
        axis.SetLabelSize(label_size)
        axis.SetTitleSize(title_size)
        axis.SetLabelFont(42)
        axis.SetTitleFont(42)
    frame.GetYaxis().SetTitleOffset(y_offset)


def draw_particle(particle, label, h_run3, out_path):  # pylint: disable=too-many-statements
    """Draw the Run 3 and Run 2 cross sections of a particle with their ratio in a pad below"""
    h_stat, g_syst = get_run2_result(particle)
    style(h_run3, RUN3_COLOR)
    style(h_stat, RUN2_COLOR)
    style(g_syst, RUN2_COLOR)
    g_syst.SetFillStyle(0)

    # ratio on the binning common to Run 3 and Run 2
    edges = get_common_edges(h_run3, h_stat)
    h_ratio = ROOT.TH1D(f"h_ratio_{particle}", "", len(edges) - 1, array("d", edges))
    h_ratio.Divide(rebin_density(h_run3, edges), rebin_density(h_stat, edges))
    style(h_ratio, RUN3_COLOR)
    g_syst_ratio = get_relative_syst(g_syst, edges[0], edges[-1])
    style(g_syst_ratio, RUN2_COLOR)
    g_syst_ratio.SetFillStyle(0)

    canvas = ROOT.TCanvas(f"c_{particle}", f"c_{particle}", 800, 800)
    frac = 0.3  # fraction of the canvas height taken by the ratio pad
    pad_top = ROOT.TPad(f"pad_top_{particle}", "", 0., frac, 1., 1.)
    pad_ratio = ROOT.TPad(f"pad_ratio_{particle}", "", 0., 0., 1., frac)
    for pad in (pad_top, pad_ratio):
        pad.SetLeftMargin(0.15)
        pad.SetRightMargin(0.05)
        pad.SetTicks(1, 1)
        pad.Draw()
    pad_top.SetBottomMargin(0.)
    pad_ratio.SetTopMargin(0.)
    pad_ratio.SetBottomMargin(0.35)

    pad_top.cd()
    pad_top.SetLogy()
    frame_top = pad_top.DrawFrame(PT_RANGE[0], 0.6e-2, PT_RANGE[1], 4.e2, f";;{Y_TITLE}")
    style_axes(frame_top, 0.045, 0.05, 1.4)
    frame_top.GetXaxis().SetLabelSize(0)
    h_run3.Draw("same")
    h_stat.Draw("same")
    g_syst.Draw("2 same")

    legend = ROOT.TLegend(0.6, 0.55, 0.9, 0.75)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.SetHeader(label, "C")
    legend.AddEntry(h_run3, "Run 3", "lp")
    legend.AddEntry(h_stat, "Run 2", "lp")
    legend.Draw()

    pad_ratio.cd()
    scale = (1. - frac) / frac  # sizes in the ratio pad are relative to its smaller height
    frame_ratio = pad_ratio.DrawFrame(PT_RANGE[0], RATIO_RANGE[0], PT_RANGE[1], RATIO_RANGE[1],
                                      ";#it{p}_{T} (GeV/#it{c});Run 3 / Run 2")
    style_axes(frame_ratio, 0.045 * scale, 0.05 * scale, 1.4 / scale)
    frame_ratio.GetXaxis().SetTitleOffset(1.0)
    frame_ratio.GetYaxis().SetNdivisions(505)
    line = ROOT.TLine(PT_RANGE[0], 1., PT_RANGE[1], 1.)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()
    g_syst_ratio.Draw("2 same")
    h_ratio.Draw("same")

    canvas.SaveAs(str(out_path))


def draw(infile):
    """Draw the cross sections of the Ds and D+ in a file with the Run 2 results"""
    with ROOT.TFile.Open(infile) as in_file:
        h_cross_section_ds = in_file.Get("h_cross_section_ds")
        h_cross_section_dplus = in_file.Get("h_cross_section_dplus")

    out_dir = Path(__file__).parent
    draw_particle("ds", "D_{s}^{+}", h_cross_section_ds, out_dir / "cross_section_ds.pdf")
    draw_particle("dplus", "D^{+}", h_cross_section_dplus, out_dir / "cross_section_dplus.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw Run 2 cross section results.')
    parser.add_argument("cross_section_file", metavar="text", help="file with the cross section results")
    args = parser.parse_args()
    draw(args.cross_section_file)
