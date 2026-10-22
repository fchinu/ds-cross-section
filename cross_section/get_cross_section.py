"""Compute the cross section from the raw yields and efficiencies."""
import argparse
from pathlib import Path
import ROOT
import yaml

ROOT.TH1.AddDirectory(False)

def main(config):
    """Compute the cross section from the raw yields and efficiencies."""
    with open(config, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with ROOT.TFile.Open(cfg["inputs"]["raw_yields"]) as in_file:
        h_ry_ds = in_file.Get("h_rawyields_ds")
        h_ry_dp = in_file.Get("h_rawyields_dplus")
        lumi = in_file.Get("hLuminosity").GetBinContent(1)

    with ROOT.TFile.Open(cfg["inputs"]["efficiency"]) as in_file:
        h_eff_ds_prompt = in_file.Get("h_efficiency_Ds_Prompt")
        h_eff_dp_prompt = in_file.Get("h_efficiency_Dplus_Prompt")

    h_cross_section_ds = h_ry_ds.Clone("h_cross_section_ds")
    h_cross_section_ds.Divide(h_eff_ds_prompt)
    h_cross_section_ds.Scale(1.0 / lumi / cfg["br"]["ds"]["value"] / 2.0, "width")
    h_cross_section_dp = h_ry_dp.Clone("h_cross_section_dplus")
    h_cross_section_dp.Divide(h_eff_dp_prompt)
    h_cross_section_dp.Scale(1.0 / lumi / cfg["br"]["dplus"]["value"] / 2.0, "width")

    if not Path(cfg["output_dir"]).exists():
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    with ROOT.TFile.Open(str(Path(cfg["output_dir"]) / "cross_section.root"), "recreate"):
        h_cross_section_ds.Write()
        h_cross_section_dp.Write()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate efficiencies for ML selection.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config_file)
