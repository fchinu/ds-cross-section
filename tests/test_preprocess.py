"""
Tests of preprocess/preprocess.py
"""
import copy
import shutil
import subprocess

import numpy as np
import pytest
import ROOT
import yaml

from helpers import (AXES, AXES_CFG, COLLISION_COUNTS, COLLISIONS_HISTOGRAM, LUMI_HISTOGRAM, N_RUNS,
                     PT_WEIGHTS, PTB_WEIGHTS, make_collisions_histogram,
                     make_sparse, random_points, sparse_sum, weight, write_analysis_results, write_weights)
import preprocess
from utils.utils import preprocess_pt_bin_dir

PTBINS = [0., 2., 4., 8., 24.]
BKG_CUTS = [0.1, 0.3, 0.5, 0.9]
MC_XSEC, VDM_XSEC = 59.5, 54.0  # visible cross sections (mb): used for the luminosity histogram, from vdM scans
# luminosity of the CBT_hadronPID flag (second row of the mock luminosity histogram: 200 * (run + 1)) summed over the
# runs, scaled by the cross sections and by the PV z over PV chi2 collisions
LUMI_CFG = {
    "histogram": LUMI_HISTOGRAM,
    "rct_flag": "CBT_hadronPID",
    "mc_visible_cross_section": MC_XSEC,
    "vdm_visible_cross_section": VDM_XSEC,
    "collision_selection": {"numerator": "PV #it{z}", "denominator": "PV #it{#chi}^{2}"},
}
LUMINOSITY = (sum(200. * (i_run + 1) for i_run in range(N_RUNS)) * MC_XSEC / VDM_XSEC
              * COLLISION_COUNTS["PV #it{z}"] / COLLISION_COUNTS["PV #it{#chi}^{2}"])
DATA_SPARSE = "hf-task-ds/Data/hSparseMass"
DATA_SPARSES = {DATA_SPARSE: ("reco", "data")}  # path: (sparse type, axes)
MC_SPARSES = {
    DATA_SPARSE: ("reco", "data"),
    "hf-task-ds/MC/Ds/Prompt/hSparseMass": ("reco", "reco_prompt"),
    "hf-task-ds/MC/Ds/NonPrompt/hSparseMass": ("reco", "reco_nonprompt"),
    "hf-task-ds/MC/Ds/Prompt/hSparseGen": ("gen", "gen_prompt"),
    "hf-task-ds/MC/Ds/NonPrompt/hSparseGen": ("gen", "gen_nonprompt"),
}

needs_hadd = pytest.mark.skipif(shutil.which("hadd") is None, reason="hadd not on PATH: load ROOT first")


def make_input(tmp_path, name, sparses, seed):
    """Write a mock AnalysisResults.root with the given sparses; return its path and the points filled"""
    rng = np.random.default_rng(seed)
    points = {path: random_points(AXES[axes], 2000, rng) for path, (_, axes) in sparses.items()}
    file_name = tmp_path / name
    write_analysis_results(
        file_name, {path: make_sparse(path.replace("/", "_"), AXES[axes], points[path])
                    for path, (_, axes) in sparses.items()})
    return file_name, points


def make_config(tmp_path, files, sparses, is_mc=False, pt_weights=None):
    """Preprocess configuration for the given input files and sparses"""
    cfg = {
        "outdir": str(tmp_path / "out"),
        "ptbins": PTBINS,
        "preprocess": {
            "bkg_cuts": BKG_CUTS,
            "axes": AXES_CFG,
            "input": {
                "files": [str(file_name) for file_name in files],
                "is_mc": is_mc,
                "sparses": [{"type": kind, "path": path} for path, (kind, _) in sparses.items()],
            },
        },
    }
    cfg["preprocess"]["input"].update(lumi=LUMI_CFG, collision_histogram=COLLISIONS_HISTOGRAM)
    if pt_weights is not None:
        cfg["preprocess"]["weights"] = {"pt": pt_weights}
    return cfg


def selected(points, kind, pt_min, pt_max, bkg_max):
    """Points passing the pT-bin selection and, for reco sparses, the background-score cut"""
    pt = points[:, AXES_CFG[f"axis_pt_{kind}"]]
    mask = (pt > pt_min) & (pt < pt_max)
    if kind == "reco":
        mask &= points[:, AXES_CFG["axis_bkg_reco"]] < bkg_max
    return points[mask]


def keys(root_file, directory):
    """Sorted 'name;cycle' of the keys in a directory of a ROOT file"""
    return sorted(f"{key.GetName()};{key.GetCycle()}" for key in root_file.Get(directory).GetListOfKeys())


def job_file(out_dir, pt_min, pt_max, i_file=0):
    """Path of the job file of an input file for a pT bin"""
    return preprocess_pt_bin_dir(out_dir, pt_min, pt_max) / "jobs" / f"AnalysisResults_{i_file}.root"


@pytest.mark.parametrize("path, expected", [
    ("hf-task-ds/MC/Ds/Prompt/hSparseMass", ("Ds", "Prompt")),
    ("hf-task-ds/MC/Dplus/NonPrompt/hSparseGen", ("Dplus", "NonPrompt")),
    ("hf-task-ds/Data/hSparseMass", None),
    ("hf-task-ds/MC/Bkg/hSparseMass", None),
])
def test_particle_origin(path, expected):
    """Only sparses at hf-task-ds/MC/<particle>/<origin>/ have a particle and origin to reweight"""
    assert preprocess.particle_origin(path) == expected


def test_get_or_make_dir(tmp_path):
    """Nested directories are returned at their full path, also when they or their parents already exist"""
    with ROOT.TFile.Open(str(tmp_path / "dirs.root"), "recreate") as root_file:
        for path in ("a/b", "a/b", "a/c/d"):
            assert preprocess.get_or_make_dir(root_file, path).GetPath().endswith(f":/{path}")


def test_process_file_data(tmp_path):
    """Each pT-bin job file holds the selected candidates, with the luminosity and collision histograms at top level"""
    file_name, points = make_input(tmp_path, "AnalysisResults.root", DATA_SPARSES, seed=1)
    out_dir = tmp_path / "out" / "data"
    preprocess.process_file(0, str(file_name), make_config(tmp_path, [file_name], DATA_SPARSES), out_dir)

    for pt_min, pt_max, bkg_max in zip(PTBINS[:-1], PTBINS[1:], BKG_CUTS):
        n_selected = len(selected(points[DATA_SPARSE], "reco", pt_min, pt_max, bkg_max))
        assert n_selected > 0
        with ROOT.TFile.Open(str(job_file(out_dir, pt_min, pt_max))) as job:
            assert keys(job, "hf-task-ds/Data") == ["hSparseMass;1"]
            assert sparse_sum(job.Get(DATA_SPARSE)) == n_selected
            assert job.Get("hLuminosity").GetBinContent(1) == pytest.approx(LUMINOSITY)
            h_coll = job.Get("hCollisions")
            counts = [h_coll.GetBinContent(i) for i in range(1, h_coll.GetNbinsX() + 1)]
            assert counts == list(COLLISION_COUNTS.values())


def test_get_luminosity_histogram_missing_label(tmp_path):
    """The RCT flag is found by its label, a missing one raises"""
    file_name, _ = make_input(tmp_path, "AnalysisResults.root", DATA_SPARSES, seed=5)
    lumi_cfg = copy.deepcopy(LUMI_CFG)
    lumi_cfg["rct_flag"] = "CBT_missing"
    with ROOT.TFile.Open(str(file_name)) as in_file:
        with pytest.raises(KeyError, match="CBT_missing"):
            preprocess.get_luminosity_histogram(in_file, lumi_cfg, in_file.Get(COLLISIONS_HISTOGRAM))


def test_get_luminosity_histogram_label_order(tmp_path):
    """Labels are matched to bins by bin number, not by the order of the label list of the axis"""
    file_name, _ = make_input(tmp_path, "AnalysisResults.root", DATA_SPARSES, seed=7)
    h_coll = make_collisions_histogram(reverse_label_order=True)
    labels = h_coll.GetXaxis().GetLabels()
    assert labels.At(0).GetName() == list(COLLISION_COUNTS)[-1]  # the label list starts with the last bin
    with ROOT.TFile.Open(str(file_name)) as in_file:
        h_luminosity = preprocess.get_luminosity_histogram(in_file, LUMI_CFG, h_coll)
    assert h_luminosity.GetBinContent(1) == pytest.approx(LUMINOSITY)


def test_get_luminosity_histogram_no_collisions(tmp_path):
    """The luminosity cannot be scaled by the collision selections if the denominator has no collisions"""
    file_name, _ = make_input(tmp_path, "AnalysisResults.root", DATA_SPARSES, seed=6)
    with ROOT.TFile.Open(str(file_name)) as in_file:
        h_coll = in_file.Get(COLLISIONS_HISTOGRAM)
        h_coll.SetBinContent(2, 0.)  # PV chi2
        with pytest.raises(ValueError, match="No collisions"):
            preprocess.get_luminosity_histogram(in_file, LUMI_CFG, h_coll)


@pytest.mark.parametrize("apply_weights", [True, False])
def test_process_file_mc(tmp_path, apply_weights):
    """MC sparses and luminosity are processed like data; with pT weights on, each MC sparse gets a reweighted copy"""
    file_name, points = make_input(tmp_path, "AnalysisResults.root", MC_SPARSES, seed=2)
    weights_file = tmp_path / "ptweights.root"
    write_weights(weights_file, {"hist_prompt": PT_WEIGHTS, "hist_nonprompt": PTB_WEIGHTS})
    pt_weights = {"apply": apply_weights, "file_name": str(weights_file),
                  "hist_names": {"Ds": {"Prompt": "hist_prompt", "NonPrompt": "hist_nonprompt"}}}
    out_dir = tmp_path / "out" / "mc"
    cfg = make_config(tmp_path, [file_name], MC_SPARSES, is_mc=True, pt_weights=pt_weights)
    preprocess.process_file(0, str(file_name), cfg, out_dir)

    for pt_min, pt_max, bkg_max in zip(PTBINS[:-1], PTBINS[1:], BKG_CUTS):
        with ROOT.TFile.Open(str(job_file(out_dir, pt_min, pt_max))) as job:
            assert job.Get("hLuminosity").GetBinContent(1) == pytest.approx(LUMINOSITY)
            for path, (kind, _) in MC_SPARSES.items():
                sel = selected(points[path], kind, pt_min, pt_max, bkg_max)
                directory, name = path.rsplit("/", 1)
                assert len(sel) > 0
                assert sparse_sum(job.Get(path)) == len(sel)
                assert all(key.endswith(";1") for key in keys(job, directory))
                species = preprocess.particle_origin(path)
                if apply_weights and species:
                    variable, spec = ("pt", PT_WEIGHTS) if species[1] == "Prompt" else ("ptb", PTB_WEIGHTS)
                    expected = sum(weight(spec, value) for value in sel[:, AXES_CFG[f"axis_{variable}_{kind}"]])
                    assert sparse_sum(job.Get(f"{path}_reweighted")) == pytest.approx(expected)
                else:
                    assert f"{name}_reweighted;1" not in keys(job, directory)


def test_process_file_missing_sparse(tmp_path):
    """A sparse missing from the input file raises a KeyError naming it"""
    file_name, _ = make_input(tmp_path, "AnalysisResults.root", DATA_SPARSES, seed=3)
    cfg = make_config(tmp_path, [file_name], {"hf-task-ds/Data/hSparseMissing": ("reco", "data")})
    with pytest.raises(KeyError, match="hSparseMissing"):
        preprocess.process_file(0, str(file_name), cfg, tmp_path / "out" / "data")


@pytest.mark.parametrize("change, message", [
    (lambda cfg: cfg["preprocess"].update(bkg_cuts=BKG_CUTS[:-1]), "bkg_cuts"),
    (lambda cfg: cfg["preprocess"]["input"].update(files=[]), "No input files"),
])
def test_main_rejects_invalid_config(tmp_path, change, message):
    """main stops before processing if the cuts do not match the pT bins or there are no input files"""
    cfg = make_config(tmp_path, [tmp_path / "AnalysisResults.root"], DATA_SPARSES)
    change(cfg)
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        preprocess.main(str(cfg_file), 1)


@needs_hadd
def test_main_merges_and_cleans_up(tmp_path):
    """main processes the input files in parallel, merges each pT bin with hadd and deletes the job files"""
    inputs = [make_input(tmp_path, f"AnalysisResults_{i}.root", DATA_SPARSES, seed=10 + i) for i in range(2)]
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(yaml.safe_dump(make_config(tmp_path, [f for f, _ in inputs], DATA_SPARSES)), encoding="utf-8")
    preprocess.main(str(cfg_file), workers=2)

    out_dir = tmp_path / "out" / "data"
    for pt_min, pt_max, bkg_max in zip(PTBINS[:-1], PTBINS[1:], BKG_CUTS):
        bin_dir = preprocess_pt_bin_dir(out_dir, pt_min, pt_max)
        assert not (bin_dir / "jobs").exists()
        assert (bin_dir / "log_merge.txt").exists()
        with ROOT.TFile.Open(str(bin_dir / f"AnalysisResults_{bin_dir.name}.root")) as merged:
            expected = sum(len(selected(points[DATA_SPARSE], "reco", pt_min, pt_max, bkg_max)) for _, points in inputs)
            assert sparse_sum(merged.Get(DATA_SPARSE)) == expected
            assert merged.Get("hLuminosity").GetBinContent(1) == pytest.approx(2 * LUMINOSITY)


@needs_hadd
def test_merge_keeps_jobs_when_hadd_fails(tmp_path):
    """If hadd fails, merge raises and keeps the job files of that pT bin"""
    file_name, _ = make_input(tmp_path, "AnalysisResults.root", DATA_SPARSES, seed=4)
    cfg = make_config(tmp_path, [file_name], DATA_SPARSES)
    out_dir = tmp_path / "out" / "data"
    preprocess.process_file(0, str(file_name), cfg, out_dir)
    with pytest.raises(subprocess.CalledProcessError):
        preprocess.merge(out_dir, cfg, n_files=2)  # the job file of a second input does not exist
    assert job_file(out_dir, PTBINS[0], PTBINS[1]).exists()
