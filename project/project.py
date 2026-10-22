"""
Project the preprocessed sparses of each pT bin onto the mass axis, after applying a cutset.
"""
# pylint: disable=duplicate-code
import argparse
import sys
from pathlib import Path

import yaml
import ROOT

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.utils import logger, preprocess_pt_bin_dir, get_object, get_cut_values  # pylint: disable=wrong-import-position


def project_bin(cfg, i_bin, sparse_name, out_file):
    """
    Apply the cutset to the sparse of a pT bin and write the mass projection, with the event histograms, to out_file.
    """
    pt_min, pt_max = cfg["ptbins"][i_bin], cfg["ptbins"][i_bin + 1]
    bin_dir = preprocess_pt_bin_dir(cfg["indir"], pt_min, pt_max)
    in_file = ROOT.TFile.Open(str(bin_dir / f"AnalysisResults_{bin_dir.name}.root"))
    if not in_file:
        raise FileNotFoundError(f"Preprocessed file of pT bin {pt_min}-{pt_max} not found in {bin_dir}")

    sparse = get_object(in_file, sparse_name)
    for cut in cfg["cuts"]:
        cut_min, cut_max = get_cut_values(cut, i_bin, len(cfg["ptbins"]) - 1)
        sparse.GetAxis(cut["axis"]).SetRangeUser(cut_min, cut_max)
    h_mass = sparse.Projection(cfg["axis_mass"])
    suffix = bin_dir.name.removeprefix("pt_")
    h_mass.SetName(f"hMass_{suffix}")
    h_mass.SetTitle(f"{pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}")
    h_mass.SetDirectory(0)

    h_ev = get_object(in_file, cfg["event_histogram"]).Clone(f"hEvents_{suffix}")
    h_coll = get_object(in_file, cfg["collision_histogram"]).Clone(f"hCollisions_{suffix}")
    h_ev.SetDirectory(0)
    h_coll.SetDirectory(0)
    in_file.Close()

    out_file.cd()
    for obj in (h_mass, h_ev, h_coll):
        obj.Write()
    logger(f"pT {pt_min}-{pt_max}: {h_mass.GetEntries():.0f} candidates after cuts")


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
    for sparse_config in cfg["sparses"]:
        with ROOT.TFile.Open(str(out_dir / f"projections_{sparse_config["label"]}.root"), "recreate") as out_file:
            for i_bin in range(len(cfg["ptbins"]) - 1):
                project_bin(cfg, i_bin, sparse_config["name"], out_file)
    logger("Finished projecting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", help="cutset configuration file")
    args = parser.parse_args()

    main(args.config)
