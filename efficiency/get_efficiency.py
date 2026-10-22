'''
Script to evaluate the efficiency from the preprocessed data.
'''
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import ROOT

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.utils import logger, preprocess_pt_bin_dir, get_object, get_cut_values  # pylint: disable=wrong-import-position, import-error

PARTICLE_CLASSES = [("Ds", "Prompt"), ("Ds", "NonPrompt"),
                    ("Dplus", "Prompt"), ("Dplus", "NonPrompt")]


def get_efficiency(cfg, particle, origin, out_file):
    """
    Get the efficiency of a particle class vs pT
    """
    pt_lims = np.array(cfg["ptbins"], dtype=np.float64)
    h_efficiency = ROOT.TH1D(f"h_efficiency_{particle}_{origin}", f"Efficiency for {particle} {origin}",
                             len(pt_lims) - 1, pt_lims)
    h_efficiency.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
    h_efficiency.GetYaxis().SetTitle("Acc.#times#varepsilon")
    for i_bin, (pt_min, pt_max) in enumerate(zip(pt_lims[:-1], pt_lims[1:])):
        bin_dir = preprocess_pt_bin_dir(cfg["indir"], pt_min, pt_max)
        with ROOT.TFile.Open(str(bin_dir / f"AnalysisResults_{bin_dir.name}.root")) as in_file:
            if not in_file:
                raise FileNotFoundError(f"Preprocessed file of pT bin {pt_min}-{pt_max} not found in {bin_dir}")

            sparse_reco = get_object(in_file, f"hf-task-ds/MC/{particle}/{origin}/hSparseMass")
            sparse_gen = get_object(in_file, f"hf-task-ds/MC/{particle}/{origin}/hSparseGen")
            for cut in cfg["cuts"]:
                cut_min, cut_max = get_cut_values(cut, i_bin, len(cfg["ptbins"]) - 1)
                sparse_reco.GetAxis(cut["axis"]).SetRangeUser(cut_min, cut_max)
            n_reco = sparse_reco.Integral(True)  # only the bins inside the cut ranges
            n_gen = sparse_gen.Integral(False)

        efficiency = n_reco / n_gen if n_gen > 0 else 0.
        h_efficiency.SetBinContent(i_bin + 1, efficiency)
        h_efficiency.SetBinError(i_bin + 1, np.sqrt(efficiency * (1 - efficiency) / n_gen) if n_gen > 0 else 0.)

    out_file.cd()
    h_efficiency.Write()

PLOTS = {
    "efficiency_ds": [(("Ds", "Prompt"), "Prompt D_{s}^{+}"), (("Ds", "NonPrompt"), "Non-Prompt D_{s}^{+}")],
    "efficiency_dplus": [(("Dplus", "Prompt"), "Prompt D^{+}"), (("Dplus", "NonPrompt"), "Non-Prompt D^{+}")],
    "efficiency_prompt": [(("Ds", "Prompt"), "Prompt D_{s}^{+}"), (("Dplus", "Prompt"), "Prompt D^{+}")],
    "efficiency_nonprompt": [(("Ds", "NonPrompt"), "Non-Prompt D_{s}^{+}"),
                             (("Dplus", "NonPrompt"), "Non-Prompt D^{+}")],
}
COLORS = {("Ds", "Prompt"): ROOT.kAzure + 4, ("Ds", "NonPrompt"): ROOT.kRed + 1,
          ("Dplus", "Prompt"): ROOT.kOrange - 3, ("Dplus", "NonPrompt"): ROOT.kTeal - 7}


def draw_efficiency_plots(out_file):
    """Draw the comparisons of the efficiencies"""
    out_file.cd()
    h_eff = {}
    for particle_class, color in COLORS.items():
        h_eff[particle_class] = out_file.Get(f"h_efficiency_{particle_class[0]}_{particle_class[1]}")
        h_eff[particle_class].SetLineColor(color)
        h_eff[particle_class].SetMarkerColor(color)
        h_eff[particle_class].SetMarkerStyle(ROOT.kFullCircle)
        h_eff[particle_class].SetMarkerSize(1.2)

    for name, entries in PLOTS.items():
        canvas = ROOT.TCanvas(f"c_{name}", "Efficiency", 800, 600)
        canvas.SetLogy()
        canvas.SetTicks(1, 1)
        canvas.DrawFrame(0., 5e-4, 24, 1.1, ";#it{p}_{T} (GeV/#it{c});Acc.#times#varepsilon")
        legend = ROOT.TLegend(0.6, 0.3, 0.9, 0.5)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.04)
        legend.SetTextFont(42)
        for particle_class, label in entries:
            h_eff[particle_class].Draw("][ same")
            legend.AddEntry(h_eff[particle_class], label, "lp")
        legend.Draw()
        canvas.SaveAs(str(Path(out_file.GetName()).parent / f"{name}.pdf"))


def main(config):
    """
    Project all the pT bins with the cutset of the configuration file.

    Args:
        config (str): path to the YAML configuration file
    """
    with open(config, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with open(cfg["cutset"], "r", encoding="utf-8") as cutset_file:
        cutset = yaml.safe_load(cutset_file)
    cfg["cuts"] = cutset["cuts"]
    cfg["ptbins"] = cutset["ptbins"]


    out_dir = Path(cfg["outdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with ROOT.TFile.Open(str(out_dir / "efficiency.root"), "recreate") as out_file:
        for particle, origin in PARTICLE_CLASSES:
            get_efficiency(cfg, particle, origin, out_file)
        draw_efficiency_plots(out_file)
    logger("Finished efficiency evaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate efficiencies for ML selection.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file.')
    args = parser.parse_args()

    main(args.config_file)
