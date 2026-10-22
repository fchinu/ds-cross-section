"""
Helpers to build small synthetic hf-task-ds outputs and pT-weight files for the tests
"""
import numpy as np
import ROOT

# (nbins, min, max) of the sparse axes, coarser than in the real task outputs
MASS, PT, CENT, SCORE = (12, 1.67, 2.27), (24, 0., 24.), (10, 0., 100.), (10, 0., 1.)
NPV, PTB, FLAG, Y = (10, 0., 100.), (10, 0., 100.), (5, 0., 5.), (10, -1., 1.)
# axes of each sparse type, in the order of the hf-task-ds outputs
AXES = {
    "data": [MASS, PT, CENT, SCORE, SCORE, SCORE],
    "reco_prompt": [MASS, PT, CENT, SCORE, SCORE, SCORE, NPV],
    "reco_nonprompt": [MASS, PT, CENT, SCORE, SCORE, SCORE, NPV, PTB, FLAG],
    "gen_prompt": [PT, Y, NPV, CENT],
    "gen_nonprompt": [PT, Y, NPV, CENT, PTB, FLAG],
}
# axis indices, as in the preprocess configs
AXES_CFG = {"axis_pt_reco": 1, "axis_bkg_reco": 3, "axis_ptb_reco": 7, "axis_pt_gen": 0, "axis_ptb_gen": 4}

EVENT_HISTOGRAM = "eventselection-run3/luminosity/hCounterTVXafterBCcuts"
COLLISION_HISTOGRAM = "hf-candidate-creator-3prong/hCollisionsCentOcc"

# pT-weight histograms as (nbins, min, max, offset, slope): bin i has content offset + slope * i
PT_WEIGHTS = (12, 0., 24., 1.0, 0.1)  # 2 GeV/c bins, coarser than the 1 GeV/c sparse pT axis
PTB_WEIGHTS = (5, 0., 100., 0.5, 0.1)


def random_points(axes, n_points, rng):
    """Random points at the bin centres of the given axes, as an (n_points, n_axes) array"""
    columns = []
    for nbins, xmin, xmax in axes:
        width = (xmax - xmin) / nbins
        columns.append(xmin + (rng.integers(0, nbins, n_points) + 0.5) * width)
    return np.column_stack(columns)


def make_sparse(name, axes, points):
    """THnSparseF with the given axes, filled once at each point (no stored errors, as in the task outputs)"""
    sparse = ROOT.THnSparseF(name, name, len(axes),
                             np.array([axis[0] for axis in axes], dtype=np.int32),
                             np.array([axis[1] for axis in axes], dtype=np.float64),
                             np.array([axis[2] for axis in axes], dtype=np.float64))
    for point in points:
        sparse.Fill(np.asarray(point, dtype=np.float64))
    return sparse


def sparse_bins(sparse):
    """{bin coordinates: (content, error^2)} of the filled bins of a sparse"""
    coord = np.zeros(sparse.GetNdimensions(), dtype=np.int32)
    bins = {}
    for i_bin in range(sparse.GetNbins()):
        content = sparse.GetBinContent(i_bin, coord)
        bins[tuple(coord)] = (content, sparse.GetBinError2(i_bin))
    return bins


def sparse_sum(sparse):
    """Sum of the contents of all filled bins of a sparse"""
    return sum(content for content, _ in sparse_bins(sparse).values())


def weight(spec, value):
    """Weight at value of a pT-weight histogram written by write_weights with the given spec"""
    nbins, xmin, xmax, offset, slope = spec
    i_bin = int((value - xmin) // ((xmax - xmin) / nbins)) + 1
    return offset + slope * i_bin


def write_objects(out_file, objects):
    """Write {path in file: object} to an open ROOT file, creating the directories"""
    for path, obj in objects.items():
        directory, _, name = path.rpartition("/")
        if directory and not out_file.GetDirectory(directory):
            out_file.mkdir(directory)  # for nested paths ROOT 6.32 returns the top directory: look it up below
        (out_file.GetDirectory(directory) if directory else out_file).WriteTObject(obj, name)


def write_weights(file_name, specs):
    """Write pT-weight histograms {path in file: spec} to a new ROOT file"""
    histos = {}
    for path, (nbins, xmin, xmax, offset, slope) in specs.items():
        histo = ROOT.TH1D(path.replace("/", "_"), "", nbins, xmin, xmax)
        histo.SetDirectory(0)
        for i_bin in range(1, nbins + 1):
            histo.SetBinContent(i_bin, offset + slope * i_bin)
        histos[path] = histo
    with ROOT.TFile.Open(str(file_name), "recreate") as out_file:
        write_objects(out_file, histos)


def write_analysis_results(file_name, sparses, n_events):
    """Write a mock AnalysisResults.root with the given {path: sparse} and the event histograms"""
    h_ev = ROOT.TH1D("luminosity/hCounterTVXafterBCcuts", "", 1, 0., 1.)  # O2 keeps the sub-folder in the name
    h_coll = ROOT.TH2D("hCollisionsCentOcc", "", 2, 0., 100., 2, 0., 1.)
    for histo in (h_ev, h_coll):
        histo.SetDirectory(0)
    h_ev.SetBinContent(1, n_events)
    h_coll.SetBinContent(1, 1, n_events)
    with ROOT.TFile.Open(str(file_name), "recreate") as out_file:
        write_objects(out_file, {EVENT_HISTOGRAM: h_ev, COLLISION_HISTOGRAM: h_coll, **sparses})
