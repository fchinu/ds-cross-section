"""
Preprocess the AnalysisResults.root files of a dataset into pT bins, writing one output file per bin.
"""
import argparse
import multiprocessing
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
import ROOT

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.utils import logger, preprocess_pt_bin_dir, get_object  # pylint: disable=wrong-import-position
from utils.sparse_reweighter import SparseReweighter  # pylint: disable=wrong-import-position


def get_or_make_dir(root_file, path):
    """Directory at path in a ROOT file, created if missing"""
    if not root_file.GetDirectory(path):
        root_file.mkdir(path)  # for nested paths ROOT 6.32 returns the top directory: look it up below
    return root_file.GetDirectory(path)


def particle_origin(path):
    """(particle, origin) of an MC sparse at hf-task-ds/MC/<particle>/<origin>/<name>, None otherwise"""
    parts = path.split("/")
    return (parts[2], parts[3]) if len(parts) == 5 and parts[1] == "MC" else None


def get_bin_by_label(axis, label):
    """Bin number of the bin of a labelled axis with the given label, raising if there is none"""
    for i_bin in range(1, axis.GetNbins() + 1):
        if axis.GetBinLabel(i_bin) == label:
            return i_bin
    raise KeyError(f"No bin labelled {label} in axis {axis.GetName()}")


def get_luminosity_histogram(in_file, lumi_cfg, h_coll):
    """
    One-bin histogram (hLuminosity) with the integrated luminosity, in 1/mub, of an input file. It is the luminosity
    of the RCT flag of the luminosity histogram, summed over the runs, scaled by the ratio of the MC and vdM visible
    cross sections and by the ratio of the collisions passing the PV z and the PV chi2 selections.
    Histograms of different files add up when merged with hadd.

    Args:
        in_file (TFile): input AnalysisResults.root
        lumi_cfg (dict): "lumi" section of the input configuration
        h_coll (TH1): collision selection histogram of the input file, with labelled bins
    """
    h_lumi = get_object(in_file, lumi_cfg["histogram"])
    i_flag = get_bin_by_label(h_lumi.GetYaxis(), lumi_cfg["rct_flag"])
    lumi = sum(h_lumi.GetBinContent(i_run, i_flag) for i_run in range(1, h_lumi.GetNbinsX() + 1))
    lumi *= lumi_cfg["mc_visible_cross_section"] / lumi_cfg["vdm_visible_cross_section"]

    selection = lumi_cfg["collision_selection"]
    n_num = h_coll.GetBinContent(get_bin_by_label(h_coll.GetXaxis(), "PV #it{z}"))
    n_den = h_coll.GetBinContent(get_bin_by_label(h_coll.GetXaxis(), "PV #it{#chi}^{2}"))
    if n_den <= 0:
        raise ValueError(f"No collisions in {selection['denominator']} of {h_coll.GetName()} in {in_file.GetName()}")
    lumi *= n_num / n_den

    h_luminosity = ROOT.TH1D("hLuminosity", "Luminosity (1/#mub)", 1, 0., 1.)
    h_luminosity.SetDirectory(0)
    h_luminosity.SetBinContent(1, lumi)
    return h_luminosity


def process_file(i_file, file_name, cfg, out_dir):
    """
    Split the configured sparses of one input file into pT bins, writing
    <out_dir>/pt_X_Y/jobs/AnalysisResults_<i_file>.root for each pT bin.

    Args:
        i_file (int): index of the input file, used to name the job files
        file_name (str): input AnalysisResults.root
        cfg (dict): full configuration
        out_dir (Path): output directory for this input (data or mc)
    """
    logger(f"Processing file {i_file}: {file_name}")
    input_cfg, axes = cfg["preprocess"]["input"], cfg["preprocess"]["axes"]
    pt_mins, pt_maxs = cfg["ptbins"][:-1], cfg["ptbins"][1:]
    reweighter = None
    if input_cfg["is_mc"] and cfg["preprocess"]["weights"]["pt"]["apply"]:
        reweighter = SparseReweighter(cfg["preprocess"]["weights"]["pt"], axes)

    in_file = ROOT.TFile.Open(file_name)
    h_coll = get_object(in_file, input_cfg["collision_histogram"])
    h_lumi = get_luminosity_histogram(in_file, input_cfg["lumi"], h_coll)
    logger(f"File {i_file}: luminosity {h_lumi.GetBinContent(1):.1f} 1/mub")

    out_files = []
    for pt_min, pt_max in zip(pt_mins, pt_maxs):
        jobs_dir = preprocess_pt_bin_dir(out_dir, pt_min, pt_max) / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        out_files.append(ROOT.TFile.Open(str(jobs_dir / f"AnalysisResults_{i_file}.root"), "recreate"))

    for sparse_cfg in input_cfg["sparses"]:
        sparse = get_object(in_file, sparse_cfg["path"])
        sparse_dir, sparse_name = sparse_cfg["path"].rsplit("/", 1)
        is_gen = sparse_cfg["type"] == "gen"
        mc_species = particle_origin(sparse_cfg["path"])
        dims = np.arange(sparse.GetNdimensions(), dtype=np.int32)

        for out_file, pt_min, pt_max, bkg_max in zip(out_files, pt_mins, pt_maxs, cfg["preprocess"]["bkg_cuts"]):
            sparse.GetAxis(axes[f"axis_pt_{sparse_cfg['type']}"]).SetRangeUser(pt_min, pt_max)
            if not is_gen:
                sparse.GetAxis(axes["axis_bkg_reco"]).SetRangeUser(0, bkg_max)
            sparse_pt = sparse.Projection(len(dims), dims)

            out_dir_root = get_or_make_dir(out_file, sparse_dir)
            out_dir_root.WriteTObject(sparse_pt, sparse_name)
            if reweighter and mc_species:
                sparse_rw = reweighter.reweight(sparse_pt, *mc_species, is_gen)
                out_dir_root.WriteTObject(sparse_rw, f"{sparse_name}_reweighted")
            for histo in (h_lumi, h_coll):
                out_file.WriteTObject(histo, histo.GetName().split("/")[-1], "Overwrite")
        logger(f"File {i_file}: {sparse_cfg['path']} split into {len(out_files)} pT bins")

    for out_file in out_files:
        out_file.Close()
    in_file.Close()


def merge(out_dir, cfg, n_files):
    """
    Merge the job files of each pT bin into <out_dir>/pt_X_Y/AnalysisResults_pt_X_Y.root with hadd,
    then delete them. If hadd fails, the job files of that bin are kept.
    """
    for pt_min, pt_max in zip(cfg["ptbins"][:-1], cfg["ptbins"][1:]):
        bin_dir = preprocess_pt_bin_dir(out_dir, pt_min, pt_max)
        jobs = [str(bin_dir / "jobs" / f"AnalysisResults_{i_file}.root") for i_file in range(n_files)]
        merged = bin_dir / f"AnalysisResults_{bin_dir.name}.root"
        logger(f"Merging {n_files} job files into {merged}")
        with open(bin_dir / "log_merge.txt", "w", encoding="utf-8") as log:
            subprocess.run(["hadd", "-f", str(merged), *jobs], stdout=log, check=True)
        shutil.rmtree(bin_dir / "jobs")


def main(config, workers):
    """
    Split the input files of the configuration into pT bins and merge the outputs of each bin.

    Args:
        config (str): path to the YAML configuration file
        workers (int): number of input files processed in parallel
    """
    with open(config, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    n_bins, n_cuts = len(cfg["ptbins"]) - 1, len(cfg["preprocess"]["bkg_cuts"])
    if n_cuts != n_bins:
        raise ValueError(f"{n_cuts} bkg_cuts for {n_bins} pT bins in {config}")
    input_cfg = cfg["preprocess"]["input"]
    files = input_cfg["files"]
    if not files:
        raise ValueError(f"No input files in {config}")
    out_dir = Path(cfg["outdir"]) / ("mc" if input_cfg["is_mc"] else "data")

    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(process_file, i_file, file_name, cfg, out_dir)
                   for i_file, file_name in enumerate(files)]
        for future in futures:
            future.result()

    merge(out_dir, cfg, len(files))
    logger("Finished processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", help="configuration file")
    parser.add_argument("--workers", "-w", type=int, default=1, help="number of input files processed in parallel")
    args = parser.parse_args()

    main(args.config, args.workers)
