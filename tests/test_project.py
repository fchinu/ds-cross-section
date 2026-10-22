"""
Tests of project/project.py
"""
import subprocess
import sys

import numpy as np
import ROOT
import yaml

from helpers import AXES, MASS, make_sparse, random_points, write_objects
from conftest import REPO_DIR

PTBINS = [2., 4., 6.]
SPARSE = "hf-task-ds/Data/hSparseMass"
LUMINOSITY = 1234.5
COLLISIONS = 800.
AXIS_PROMPT, AXIS_BKG = 4, 3  # score axes of the data sparse
# cuts as in configs/cutset.yml: single values or one value per pT bin
CUTS = [
    {"name": "score_prompt", "axis": AXIS_PROMPT, "min": 0.1, "max": 1.},
    {"name": "score_bkg", "axis": AXIS_BKG, "min": 0., "max": [0.5, 0.3]},
]


def write_preprocessed(indir, seed=0):
    """Write a mock preprocess output (one AnalysisResults file per pT bin); return the points of each bin"""
    rng = np.random.default_rng(seed)
    points = {}
    for pt_min, pt_max in zip(PTBINS[:-1], PTBINS[1:]):
        bin_dir = indir / f"pt_{int(pt_min * 10)}_{int(pt_max * 10)}"
        bin_dir.mkdir(parents=True)
        points[(pt_min, pt_max)] = random_points(AXES["data"], 3000, rng)
        h_lumi = ROOT.TH1D("hLuminosity", "", 1, 0., 1.)
        h_coll = ROOT.TH1D("hCollisions", "", 1, 0., 1.)
        for histo in (h_lumi, h_coll):
            histo.SetDirectory(0)
        h_lumi.SetBinContent(1, LUMINOSITY)
        h_coll.SetBinContent(1, COLLISIONS)
        sparse = make_sparse("hSparseMass", AXES["data"], points[(pt_min, pt_max)])
        with ROOT.TFile.Open(str(bin_dir / f"AnalysisResults_{bin_dir.name}.root"), "recreate") as out_file:
            write_objects(out_file, {"hLuminosity": h_lumi, "hCollisions": h_coll, SPARSE: sparse})
    return points


def write_configs(tmp_path, cuts=None):
    """Write the projection config and cutset; return the config path"""
    cutset = tmp_path / "cutset.yml"
    cutset.write_text(yaml.dump({"ptbins": PTBINS, "cuts": CUTS if cuts is None else cuts}))
    cfg = {
        "indir": str(tmp_path / "preprocessed"),
        "outdir": str(tmp_path / "projected"),
        "sparses": [{"label": "data", "name": SPARSE}],
        "luminosity_histogram": "hLuminosity",
        "collision_histogram": "hCollisions",
        "axis_mass": 0,
        "cutset": str(cutset),
    }
    cfg_file = tmp_path / "config_project.yml"
    cfg_file.write_text(yaml.dump(cfg))
    return cfg_file


def run_project(cfg_file):
    """Run project/project.py as a script, as done in the analysis"""
    return subprocess.run([sys.executable, str(REPO_DIR / "project" / "project.py"), str(cfg_file)],
                          capture_output=True, text=True, check=False)


def expected_mass(points, ibin):
    """Mass histogram contents of the candidates passing the cutset in a pT bin"""
    mask = np.ones(len(points), dtype=bool)
    for cut in CUTS:
        cut_max = cut["max"][ibin] if isinstance(cut["max"], list) else cut["max"]
        values = points[:, cut["axis"]]
        mask &= (values > cut["min"]) & (values < cut_max)
    return np.histogram(points[mask][:, 0], bins=MASS[0], range=MASS[1:])[0]


def test_project(tmp_path):
    """The mass projection of each pT bin holds the candidates passing the cuts, with per-pT-bin cut values"""
    points = write_preprocessed(tmp_path / "preprocessed")
    result = run_project(write_configs(tmp_path))
    assert result.returncode == 0, result.stderr

    with ROOT.TFile.Open(str(tmp_path / "projected" / "projections_data.root")) as out_file:
        for ibin, (pt_min, pt_max) in enumerate(zip(PTBINS[:-1], PTBINS[1:])):
            suffix = f"{int(pt_min * 10)}_{int(pt_max * 10)}"
            expected = expected_mass(points[(pt_min, pt_max)], ibin)
            assert expected.sum() > 0
            h_mass = out_file.Get(f"hMass_{suffix}")
            contents = np.array([h_mass.GetBinContent(i) for i in range(1, h_mass.GetNbinsX() + 1)])
            assert np.array_equal(contents, expected)
            assert h_mass.GetEntries() == expected.sum()
        # one luminosity and one collision histogram per file, not per pT bin
        keys = [key.GetName() for key in out_file.GetListOfKeys()]
        assert [key for key in keys if "Lumi" in key or "Coll" in key] == ["hLuminosity", "hCollisions"]
        assert out_file.Get("hLuminosity").GetBinContent(1) == LUMINOSITY
        assert out_file.Get("hCollisions").GetBinContent(1) == COLLISIONS


def test_project_cuts_depend_on_pt_bin(tmp_path):
    """A looser cut in one pT bin keeps more candidates there only"""
    points = write_preprocessed(tmp_path / "preprocessed", seed=1)
    result = run_project(write_configs(tmp_path))
    assert result.returncode == 0, result.stderr

    with ROOT.TFile.Open(str(tmp_path / "projected" / "projections_data.root")) as out_file:
        n_cands = [out_file.Get(f"hMass_{int(pt_min * 10)}_{int(pt_max * 10)}").GetEntries()
                   for pt_min, pt_max in zip(PTBINS[:-1], PTBINS[1:])]
    # the bkg-score max is 0.5 in the first bin and 0.3 in the second, with the same number of candidates generated
    assert n_cands[0] > n_cands[1]
    assert n_cands[0] < len(points[(PTBINS[0], PTBINS[1])])


def test_project_wrong_number_of_cut_values(tmp_path):
    """A cut given per pT bin must have one value for each pT bin"""
    write_preprocessed(tmp_path / "preprocessed")
    cuts = [{"name": "score_bkg", "axis": AXIS_BKG, "min": 0., "max": [0.5, 0.3, 0.2]}]
    result = run_project(write_configs(tmp_path, cuts))
    assert result.returncode != 0
    assert "score_bkg" in result.stderr and "3 values of max for 2 pT bins" in result.stderr


def test_project_missing_preprocessed_bin(tmp_path):
    """A missing preprocessed pT bin is reported instead of silently skipped"""
    write_preprocessed(tmp_path / "preprocessed")
    for path in (tmp_path / "preprocessed" / "pt_40_60").iterdir():
        path.unlink()
    result = run_project(write_configs(tmp_path))
    assert result.returncode != 0
    assert "pt_40_60" in result.stderr  # ROOT raises OSError or project.py FileNotFoundError, depending on the version
