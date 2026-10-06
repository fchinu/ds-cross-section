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
from utils.utils import logger  # pylint: disable=wrong-import-position
from utils.sparse_reweighter import SparseReweighter  # pylint: disable=wrong-import-position


def pt_bin_dir(out_dir, pt_min, pt_max):
    """Output directory of a pT bin, e.g. <out_dir>/pt_20_25 for 2 < pT < 2.5 GeV/c"""
    return Path(out_dir) / f"pt_{int(pt_min * 10)}_{int(pt_max * 10)}"


def get_object(in_file, path):
    """Get an object from a ROOT file, raising if it is missing"""
    obj = in_file.Get(path)
    if not obj:
        raise KeyError(f"{path} not found in {in_file.GetName()}")
    return obj


def particle_origin(path):
    """(particle, origin) of an MC sparse at hf-task-ds/MC/<particle>/<origin>/<name>, None otherwise"""
    parts = path.split("/")
    return (parts[2], parts[3]) if len(parts) == 5 and parts[1] == "MC" else None


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
    h_ev = get_object(in_file, input_cfg["event_histogram"])
    h_coll = get_object(in_file, input_cfg["collision_histogram"])

    out_files = []
    for pt_min, pt_max in zip(pt_mins, pt_maxs):
        jobs_dir = pt_bin_dir(out_dir, pt_min, pt_max) / "jobs"
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

            out_dir_root = out_file.mkdir(sparse_dir, "", True)
            out_dir_root.WriteTObject(sparse_pt, sparse_name)
            if reweighter and mc_species:
                sparse_rw = reweighter.reweight(sparse_pt, *mc_species, is_gen)
                out_dir_root.WriteTObject(sparse_rw, f"{sparse_name}_reweighted")
            # event histograms go next to each sparse; overwrite if two sparses share the directory.
            # O2 histogram names can include their sub-folder (luminosity/hCounterTVXafterBCcuts): drop it
            for histo in (h_ev, h_coll):
                out_dir_root.WriteTObject(histo, histo.GetName().split("/")[-1], "Overwrite")
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
        bin_dir = pt_bin_dir(out_dir, pt_min, pt_max)
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
