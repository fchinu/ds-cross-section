"""Main script to run the scripts to compute the feed-down fraction."""
import argparse
import copy
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_DIR))
from utils.utils import logger  # pylint: disable=wrong-import-position, import-error

def per_pt_bin(value, n_bins, label):
    """Expand a single value or a list with one value per pT bin to a list of n_bins values"""
    if isinstance(value, list):
        if len(value) != n_bins:
            raise ValueError(f"{label}: {len(value)} values for {n_bins} pT bins")
        return [float(v) for v in value]
    return [float(value)] * n_bins


def compact(values):
    """Write a single number when the value is the same in all pT bins, a list otherwise"""
    return values[0] if len(set(values)) == 1 else values


def produce_cutsets(cfg):
    """
    Produce one cutset file per step of the variation cut, in the same format as configs/cutset.yml.
    Each cutset contains the background cut chosen in the config and the variation cut at that step.

    Variation cut (cfg["cuts"]["variation"]): n_cuts thresholds equally spaced between min and max
    (single values or one value per pT bin). direction: upper keeps candidates above the threshold
    (cut range [threshold, limits[1]]), direction: lower keeps them below it ([limits[0], threshold]).

    Returns the list of cutset file names.
    """
    with open(cfg["inputs"]["central_fit"], "r", encoding="utf-8") as fit_cfg_file:
        fit_cfg = yaml.safe_load(fit_cfg_file)
    with open(cfg["inputs"]["central_efficiency"], "r", encoding="utf-8") as eff_cfg_file:
        eff_cfg = yaml.safe_load(eff_cfg_file)

    if fit_cfg["cutset_file_name"] != eff_cfg["cutset"]:
        raise ValueError("Cutset file names in central fit and efficiency configs do not match.")

    with open(fit_cfg["cutset_file_name"], "r", encoding="utf-8") as cutset_file:
        central_cutset = yaml.safe_load(cutset_file)
    ptbins = central_cutset["ptbins"]
    n_bins = len(ptbins) - 1

    var_cfg = cfg["cuts"]["variation"]
    bkg_cfg = cfg["cuts"]["bkg"]

    n_cuts = var_cfg["n_cuts"]
    direction = var_cfg["direction"]
    if direction not in ("upper", "lower"):
        raise ValueError(f"Variation cut: direction must be 'upper' or 'lower', got '{direction}'")
    limit_low, limit_high = 0., 1.

    var_min = per_pt_bin(var_cfg["min"], n_bins, "Variation cut min")
    var_max = per_pt_bin(var_cfg["max"], n_bins, "Variation cut max")

    bkg_cut = {
        "name": bkg_cfg["name"],
        "axis": bkg_cfg["axis"],
        "min": compact(per_pt_bin(bkg_cfg["min"], n_bins, "Background cut min")),
        "max": compact(per_pt_bin(bkg_cfg["max"], n_bins, "Background cut max")),
    }

    out_dir = Path(cfg["output"]["dir"]) / "cutsets"
    out_dir.mkdir(parents=True, exist_ok=True)

    cutset_files = []
    for i_cut in range(n_cuts):
        thresholds = [round(lo + i_cut * (hi - lo) / (n_cuts - 1), 6) for lo, hi in zip(var_min, var_max)]
        if direction == "upper":
            var_range = {"min": compact(thresholds), "max": limit_high}
        else:
            var_range = {"min": limit_low, "max": compact(thresholds)}
        var_cut = {"name": var_cfg["name"], "axis": var_cfg["axis"], **var_range}

        cutset = {"ptbins": ptbins, "cuts": [dict(bkg_cut), var_cut]}
        cutset_file_name = out_dir / f"cutset_{var_cfg['name']}_{i_cut:02d}.yml"
        with open(cutset_file_name, "w", encoding="utf-8") as f:
            yaml.dump(cutset, f, default_flow_style=None, sort_keys=False)
        cutset_files.append(cutset_file_name)

    return cutset_files

def produce_projection_configs(cfg, cutset_files):
    """
    Produce the projection config files (one per central projection config, e.g. data and MC) of each cutset,
    equal to the central ones except for the cutset file and the output directory.

    Returns, for each cutset, the list of config file names and a dict {central outdir: outdir}
    to find the projections of the cutset.
    """
    with open(cfg["inputs"]["central_fit"], "r", encoding="utf-8") as fit_cfg_file:
        central_cutset_file = yaml.safe_load(fit_cfg_file)["cutset_file_name"]

    central_cfgs = {}
    for central_file in cfg["inputs"]["central_projections"]:
        with open(central_file, "r", encoding="utf-8") as proj_cfg_file:
            central_cfgs[Path(central_file).stem.removeprefix("config_")] = yaml.safe_load(proj_cfg_file)
    for kind, central_cfg in central_cfgs.items():
        if central_cfg["cutset"] != central_cutset_file:
            raise ValueError(f"Cutset file names in central fit and {kind} projection configs do not match.")

    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)

    proj_config_files, proj_dirs = [], []
    for cutset_file in cutset_files:
        label = Path(cutset_file).stem.removeprefix("cutset_")
        proj_config_files.append([])
        proj_dirs.append({})
        for kind, central_cfg in central_cfgs.items():
            proj_cfg = copy.deepcopy(central_cfg)
            proj_cfg["cutset"] = str(cutset_file)
            proj_cfg["outdir"] = str(out_dir / "projections" / label / kind)
            proj_dirs[-1][Path(central_cfg["outdir"])] = Path(proj_cfg["outdir"])

            proj_config_file = out_dir / "configs" / f"config_project_{kind}_{label}.yml"
            with open(proj_config_file, "w", encoding="utf-8") as f:
                yaml.dump(proj_cfg, f, default_flow_style=None, sort_keys=False)
            proj_config_files[-1].append(proj_config_file)

    return proj_config_files, proj_dirs


def produce_efficiency_configs(cfg, cutset_files):
    """
    Produce one efficiency config file per cutset, equal to the central one except for the cutset file
    and the output directory.

    Returns the list of efficiency config file names.
    """
    with open(cfg["inputs"]["central_efficiency"], "r", encoding="utf-8") as eff_cfg_file:
        central_cfg = yaml.safe_load(eff_cfg_file)

    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)

    eff_config_files = []
    for cutset_file in cutset_files:
        label = Path(cutset_file).stem.removeprefix("cutset_")
        eff_cfg = copy.deepcopy(central_cfg)
        eff_cfg["cutset"] = str(cutset_file)
        eff_cfg["outdir"] = str(out_dir / "efficiencies" / label)

        eff_config_file = out_dir / "configs" / f"config_efficiency_{label}.yml"
        with open(eff_config_file, "w", encoding="utf-8") as f:
            yaml.dump(eff_cfg, f, default_flow_style=None, sort_keys=False)
        eff_config_files.append(eff_config_file)

    return eff_config_files


def produce_fit_configs(cfg, cutset_files, proj_dirs):
    """
    Produce one fit config file per cutset, equal to the central one except for the cutset file,
    the input projections (proj_dirs, from produce_projection_configs), the output directory
    and the parameters in cfg["fit"]["fix_params_to_central_case"], which are fixed to the values of the central fit.

    Returns the list of fit config file names.
    """
    def relocate(file_name, dirs):
        file_name = Path(file_name)
        if file_name.parent not in dirs:
            raise ValueError(f"Fit input {file_name} is not in the output directory of a central projection config.")
        return str(dirs[file_name.parent] / file_name.name)

    with open(cfg["inputs"]["central_fit"], "r", encoding="utf-8") as fit_cfg_file:
        central_cfg = yaml.safe_load(fit_cfg_file)
    central_outputs = central_cfg["outputs"]
    central_fit_file = Path(central_outputs["directory"]) / f"fit_mass{central_outputs['suffix']}.root"
    fixed_pars = list(cfg["fit"]["fix_params_to_central_case"])

    out_dir = Path(cfg["output"]["dir"])
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)

    fit_config_files = []
    for cutset_file, dirs in zip(cutset_files, proj_dirs):
        label = Path(cutset_file).stem.removeprefix("cutset_")
        fit_cfg = copy.deepcopy(central_cfg)
        fit_cfg["cutset_file_name"] = str(cutset_file)
        inputs = fit_cfg["inputs"]
        inputs["data"] = relocate(inputs["data"], dirs)
        inputs["mc"] = [relocate(file_name, dirs) for file_name in inputs["mc"]]
        # the normalisation follows the cuts, the shape template (mc_corr_bkg_shape) is the same for all cut sets
        inputs["mc_corr_bkg_norm"] = relocate(inputs["mc_corr_bkg_norm"], dirs)
        if inputs.get("mc_dplus_norm"):
            inputs["mc_dplus_norm"] = [relocate(file_name, dirs) for file_name in inputs["mc_dplus_norm"]]
        fit_cfg["outputs"]["directory"] = str(out_dir / "fits" / label)
        fit_cfg["fit_configs"]["fix_params_to_file"] = {
            "do": True, "file": str(central_fit_file), "params": fixed_pars}
        if "sigma" in fixed_pars:
            # the D+ sigma cannot be fixed if composed from the Ds one, fixing both keeps the central ratio
            fit_cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"] = False
        Path(fit_cfg["outputs"]["directory"]).mkdir(parents=True, exist_ok=True)

        fit_config_file = out_dir / "configs" / f"config_fit_{label}.yml"
        with open(fit_config_file, "w", encoding="utf-8") as f:
            yaml.dump(fit_cfg, f, default_flow_style=None, sort_keys=False)
        fit_config_files.append(fit_config_file)

    return fit_config_files


def run_jobs(script, config_files, n_jobs, log_dir):
    """
    Run a python script on each config file, n_jobs at a time, with the output of each job in
    <log_dir>/<config file name>.log

    Returns the list of config files whose job failed.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    def run_job(config_file):
        log_file = log_dir / f"{Path(config_file).stem}.log"
        logger(f"Running {script.name} on {config_file}")
        with open(log_file, "w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(script), str(config_file)],
                                    stdout=log, stderr=subprocess.STDOUT, check=False,
                                    env={**os.environ, "PYTHONFAULTHANDLER": "1"})
        if result.returncode != 0:
            logger(f"{script.name} failed on {config_file}, see {log_file}", "WARNING")
        return result.returncode == 0

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        succeeded = list(pool.map(run_job, config_files))
    return [config_file for config_file, ok in zip(config_files, succeeded) if not ok]



def run(config_file, projection=False, fit=False, efficiency=False):
    """
    Run the scripts to compute the feed-down fraction
    """
    config_file = Path(config_file)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not (projection or fit or efficiency):
        projection = fit = efficiency = True
        logger("No step specified, running all steps: projection, efficiency and fit")

    # produce config files
    cutset_files = produce_cutsets(cfg)
    first_cut, last_cut = cfg["cuts"]["variation"].get("run_range") or [0, len(cutset_files)]
    cutset_files = cutset_files[first_cut:last_cut]
    if projection or fit:
        proj_configs, proj_dirs = produce_projection_configs(cfg, cutset_files)
    if fit:
        fit_configs = produce_fit_configs(cfg, cutset_files, proj_dirs)  # pylint: disable=possibly-used-before-assignment
    if efficiency:
        eff_configs = produce_efficiency_configs(cfg, cutset_files)

    log_dir = Path(cfg["output"]["dir"]) / "logs"
    failed = []
    if projection:
        failed += run_jobs(REPO_DIR / "project" / "project.py",
                        [proj_config for proj_configs_cut in proj_configs for proj_config in proj_configs_cut],
                        cfg["n_jobs"]["projections"], log_dir)
    if fit:
        fit_configs_ok = [fit_config for fit_config, proj_configs_cut in zip(fit_configs, proj_configs)
                        if not set(proj_configs_cut) & set(failed)]
        for fit_config in set(fit_configs) - set(fit_configs_ok):
            logger(f"Skipping {fit_config}: its projections failed", "WARNING")
        failed += run_jobs(REPO_DIR / "fit" / "fit.py", fit_configs_ok, cfg["n_jobs"]["fit"], log_dir)
    if efficiency:
        failed += run_jobs(REPO_DIR / "efficiency" / "get_efficiency.py", eff_configs,
                        cfg["n_jobs"]["efficiency"], log_dir)

    if failed:
        raise RuntimeError(f"{len(failed)} jobs failed, see the logs in {log_dir}: {[str(f) for f in failed]}")
    logger("Finished running")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the feed-down fraction analysis.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file.')
    parser.add_argument('--projection', '-p', action='store_true', help='Run the projection step.')
    parser.add_argument('--fit', '-f', action='store_true', help='Run the fit step.')
    parser.add_argument('--efficiency', '-e', action='store_true', help='Run the efficiency step.')
    args = parser.parse_args()

    run(args.config_file, projection=args.projection, fit=args.fit, efficiency=args.efficiency)
