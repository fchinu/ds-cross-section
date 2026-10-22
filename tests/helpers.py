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

# luminosity and collision selection histograms: the bin of run j and RCT flag k of the luminosity histogram
# has content 100 * (j + 1) * (k + 1), the collision selection histogram has labelled bins
LUMI_HISTOGRAM = "eventselection-run3/luminosity/hLumiTVXafterBCcutsRCT"
COLLISIONS_HISTOGRAM = "hf-candidate-creator-3prong/hCollisions"
RCT_FLAGS = ["CBT", "CBT_hadronPID", "CBT_calo"]
N_RUNS = 3
COLLISION_COUNTS = {"All": 1000., "PV #it{#chi}^{2}": 900., "PV #it{z}": 800.}

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


def make_lumi_histogram():
    """Mock luminosity histogram with a bin per run (x) and a labelled row per RCT flag (y)"""
    h_lumi = ROOT.TH2D("luminosity/hLumiTVXafterBCcutsRCT", "", N_RUNS, 0., N_RUNS, len(RCT_FLAGS), 0., len(RCT_FLAGS))
    h_lumi.SetDirectory(0)
    for i_flag, flag in enumerate(RCT_FLAGS):
        h_lumi.GetYaxis().SetBinLabel(i_flag + 1, flag)
        for i_run in range(N_RUNS):
            h_lumi.SetBinContent(i_run + 1, i_flag + 1, 100. * (i_run + 1) * (i_flag + 1))
    return h_lumi


def make_collisions_histogram(reverse_label_order=False):
    """
    Mock collision selection histogram with labelled bins. The order in which the labels are created, which is the
    order of the label list of the axis, can differ from the bin order, as in the O2 histograms.
    """
    h_coll = ROOT.TH1D("hCollisions", "", len(COLLISION_COUNTS), 0., len(COLLISION_COUNTS))
    h_coll.SetDirectory(0)
    bins = list(enumerate(COLLISION_COUNTS.items(), start=1))
    for i_bin, (label, count) in (reversed(bins) if reverse_label_order else bins):
        h_coll.GetXaxis().SetBinLabel(i_bin, label)
        h_coll.SetBinContent(i_bin, count)
    return h_coll


def write_analysis_results(file_name, sparses):
    """Write a mock AnalysisResults.root with the given {path: sparse} and the luminosity and collision histograms"""
    with ROOT.TFile.Open(str(file_name), "recreate") as out_file:
        write_objects(out_file, {LUMI_HISTOGRAM: make_lumi_histogram(),
                                 COLLISIONS_HISTOGRAM: make_collisions_histogram(), **sparses})
