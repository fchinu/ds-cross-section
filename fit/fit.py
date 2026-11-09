"""
Script for the raw-yield extraction of B mesons
"""

import argparse
import contextlib
import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import ROOT
import yaml
from hist import Hist
from matplotlib.offsetbox import AnchoredText
from pypdf import PdfWriter
from flarefly import F2MassFitter, DataHandler

SIGNAL_SUFFIXES = ("_ds", "_dplus")
OUTFILE_LOCK = None

def create_hist(pt_lims, contents, errors, label_pt=r"$p_\mathrm{T}~(\mathrm{GeV}/c)$"):
    """
    Helper method to create histogram

    Parameters
    ----------

    - pt_lims (list): list of pt limits
    - contents (list): histogram contents
    - errors (list): histogram errors
    - label_pt (str): label for x axis

    Returns
    ----------
    - histogram (hist.Hist)

    """

    pt_cent = [(pt_min+pt_max)/2 for pt_min, pt_max in zip(pt_lims[:-1], pt_lims[1:])]
    histo = Hist.new.Var(pt_lims, name="x", label=label_pt).Weight()
    histo.fill(pt_cent, weight=contents)
    histo.view(flow=False).variance = np.array(errors)**2

    return histo


# pylint: disable=too-many-arguments
def add_info_on_canvas(axs, loc, system, pt_min, pt_max, fitter=None):
    """
    Helper method to add text on flarefly mass fit plot

    Parameters
    ----------
    - axs: matplotlib.figure.Axis
        Axis instance of the mass fit figure

    - loc: str
        Location of the info on the figure

    - system: str
        System (pp, MC pp)

    - pt_min: float
        Minimum pT value in the pT range

    - pt_max: float
        Maximum pT value in the pT range

    - fitter: F2MassFitter
        Fitter instance allowing to access chi2 and ndf if wanted
    """
    xspace = " "
    text = xspace
    if fitter is not None:
        chi2 = fitter.get_chi2()
        ndf = fitter.get_ndf()
        text += fr"$\chi^2 / \mathrm{{ndf}} =${chi2:.2f} / {ndf} $\simeq$ {chi2/ndf:.2f}""\n"

    text += "\n\n"
    text += xspace + system + ", " + r"$\sqrt{s} = 13.6$ TeV" + "\n"
    text += xspace + fr"{pt_min:.1f} < $p_{{\mathrm{{T}}}}$ < {pt_max:.1f} GeV/$c$, $|y|$ < 0.8""\n"

    anchored_text = AnchoredText(text, loc=loc, frameon=False)
    axs.add_artist(anchored_text)


def set_comb_bkg_initpars(fitter, ibkg, bkg_func):
    """
    Set the initial parameters of the combinatorial background, only for the
    parameters defined for the chosen function (flarefly raises otherwise)

    Parameters
    ----------

    - fitter (F2MassFitter): fitter instance
    - ibkg (int): index of the combinatorial-background pdf
    - bkg_func (str): name of the combinatorial-background function
    """
    if bkg_func in ("expo", "expopow"):
        fitter.set_background_initpar(ibkg, "lam", -1.2, limits=[-10., 10.])
    elif bkg_func.startswith("chebpol"):
        order = int(bkg_func.replace("chebpol", ""))
        if order >= 1:
            fitter.set_background_initpar(ibkg, "c1", -0.05, limits=[-0.2, 0.2])
        if order >= 2:
            fitter.set_background_initpar(ibkg, "c2", 0.008, limits=[-0.03, 0.03])


def get_sweights_df(fitter, df, mass_limits):
    """
    Get the sweights of a fit as a DataFrame indexed like the fitted candidates

    Parameters
    ----------

    - fitter (F2MassFitter): fitter after the fit
    - df (pandas.DataFrame): candidates given to the fitter
    - mass_limits (list): mass limits of the fit, candidates outside are dropped by zfit
    """
    df_in_range = df.query(f"{mass_limits[0]} <= fM <= {mass_limits[1]}")
    sweights = pd.DataFrame({name: np.asarray(sw) for name, sw in fitter.get_sweights().items()})
    assert len(sweights) == len(df_in_range), \
        f"{len(sweights)} sweights for {len(df_in_range)} candidates in the fit range"
    sweights.index = df_in_range.index
    return sweights


def expand_per_pt(value, n_pt_bins, is_single):
    """
    Expand a single setting to one copy per pT bin.
    If the setting is already given per pT bin, it is returned unchanged.

    Parameters
    ----------

    - value: setting from the config
    - n_pt_bins (int): number of pT bins
    - is_single (callable): returns True if value is a single setting (to be replicated)

    Returns
    ----------

    - list with n_pt_bins elements (independent copies, since they can be modified in place)
    """
    if is_single(value):
        return [copy.deepcopy(value) for _ in range(n_pt_bins)]
    if len(value) != n_pt_bins:
        raise ValueError(f"Expected {n_pt_bins} entries (one per pT bin), got {len(value)}")
    return value


def get_corr_bkg_handler(cfg, ipt, file_name, pt_min, pt_max):
    """
    DataHandler of the correlated background template of a pT bin

    Parameters
    ----------

    - cfg (dict): fit config, with the per-pT-bin settings already expanded
    - ipt (int): pT bin index
    - file_name (str): file with the template
    - pt_min (float): lower edge of the pT bin
    - pt_max (float): upper edge of the pT bin
    - smooth (bool): smooth the template with TH1::Smooth
    """
    with ROOT.TFile.Open(file_name) as f_mc:
        h_bkg = f_mc[f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}"]
        h_bkg.SetDirectory(0)
    return DataHandler(h_bkg, var_name="fM",
                       limits=cfg["fit_configs"]["mass_limits"][ipt],
                       rebin=cfg["fit_configs"]["rebin"][ipt])


def init_worker(lock):
    """
    Initialise a worker process with the lock of the output file

    Parameters
    ----------

    - lock (multiprocessing.Lock): lock shared by the workers
    """
    global OUTFILE_LOCK  # pylint: disable=global-statement
    OUTFILE_LOCK = lock


def fit_pt_bin(cfg, ipt, pt_min, pt_max, outfile_name):  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """
    Fit MC and data in one pT bin

    Parameters
    ----------

    - cfg (dict): fit config, with the per-pT-bin settings already expanded
    - ipt (int): pT bin index
    - pt_min (float): lower edge of the pT bin
    - pt_max (float): upper edge of the pT bin
    - outfile_name (str): output ROOT file

    Returns
    ----------

    - dict with "mc" and "data": {(signal index, result name): (value, uncertainty)},
      empty if the corresponding fit did not converge
    """
    outdir = cfg["outputs"]["directory"]
    bin_results = {"mc": {}, "data": {}}
    use_corr_bkg_pt = cfg["fit_configs"]["use_bkg_templ"][ipt]

    # we first fit MC only
    data_hdl_mc = DataHandler(cfg["inputs"]["mc"][0], var_name="fM",
                              histoname=f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}",
                              limits=cfg["fit_configs"]["mass_limits"][ipt],
                              rebin=cfg["fit_configs"]["rebin"][ipt])
    data_hdl_mc.add_data(cfg["inputs"]["mc"][1], var_name="fM", histoname=f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}")
    fitter_mc_pt = F2MassFitter(data_hdl_mc,
                                cfg["fit_configs"]["signal_funcs"][ipt],
                                ["nobkg"],
                                name=f"fit_mc_pt{pt_min*10:.0f}_{pt_max*10:.0f}",
                                label_signal_pdf=[r"$\mathrm{D_s^+}$ signal", r"$\mathrm{D^+}$ signal"],
                                extended=cfg["fit_configs"]["extended"][ipt])
    fitter_mc_pt.set_signal_initpar(0, "sigma", 0.015, limits=[0.0, 0.10])
    fitter_mc_pt.set_particle_mass(0, pdg_id=431)
    fitter_mc_pt.set_signal_initpar(1, "sigma", 0.015, limits=[0.0, 0.10])
    fitter_mc_pt.set_particle_mass(1, pdg_id=421)
    fitter_mc_pt.set_signal_initpar(0, "alpha", 1.5, limits=[0.5, 10.])
    fitter_mc_pt.set_signal_initpar(0, "n", 5., limits=[0.5, 50.])
    fitter_mc_pt.set_signal_initpar(1, "alpha", 1.5, limits=[0.5, 10.])
    fitter_mc_pt.set_signal_initpar(1, "n", 5., limits=[0.5, 50.])

    result = fitter_mc_pt.mass_zfit()
    if not result.converged:
        raise RuntimeError(f"MC fit did not converge in pT bin {ipt} ({pt_min}-{pt_max} GeV/c)")

    fig, axs = fitter_mc_pt.plot_mass_fit(
        style="ATLAS",
        figsize=(8, 8),
        axis_title=r"$M(\mathrm{K^\pm K^\mp\pi^\pm})$ (GeV/$c^2$)"
    )
    add_info_on_canvas(axs, "upper left", "MC pp", pt_min, pt_max, fitter_mc_pt)

    fig_res, axs_res = fitter_mc_pt.plot_raw_residuals(
        style="ATLAS",
        figsize=(8, 8),
        axis_title=r"$M(\mathrm{K^\pm K^\mp\pi^\pm})$ (GeV/$c^2$)"
    )
    add_info_on_canvas(axs_res, "upper left", "MC pp", pt_min, pt_max, fitter_mc_pt)

    fig.savefig(os.path.join(outdir, f"fit_mass_pt{pt_min*10:.0f}_{pt_max*10:.0f}_mc.pdf"))
    fig_res.savefig(os.path.join(outdir, f"fit_massres_pt{pt_min*10:.0f}_{pt_max*10:.0f}_mc.pdf"))
    plt.close(fig)
    plt.close(fig_res)

    for i_sig in range(len(SIGNAL_SUFFIXES)):
        for name in ("mu", "sigma"):
            value, value_unc = fitter_mc_pt.get_signal_parameter(i_sig, name)
            bin_results["mc"][(i_sig, name)] = (float(value), float(value_unc))
    sigmas_mc = [bin_results["mc"][(i_sig, "sigma")][0] for i_sig in range(len(SIGNAL_SUFFIXES))]

    fixed_pars_mc = {
        (i_sig, par_name): fitter_mc_pt.get_signal_parameter(i_sig, par_name)[0]
        for i_sig in range(len(cfg["fit_configs"]["signal_funcs"][ipt]))
        for par_name in cfg["fit_configs"]["fix_params_to_mc"][ipt]
    }

    # then we fit data
    data_hdl = DataHandler(cfg["inputs"]["data"], var_name="fM", histoname=f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}",
                           limits=cfg["fit_configs"]["mass_limits"][ipt],
                           rebin=cfg["fit_configs"]["rebin"][ipt])

    bkg_funcs = list(cfg["fit_configs"]["bkg_funcs"][ipt])
    label_bkg_pdf = ["Comb. bkg"]
    if use_corr_bkg_pt:
        # D+ signal MC for the normalisation of the correlated background: prompt + non-prompt
        n_dplus_signal = sum(
            DataHandler(file_name, var_name="fM", histoname=f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}",
                        limits=cfg["fit_configs"]["mass_limits"][ipt],
                        rebin=cfg["fit_configs"]["rebin"][ipt]).get_norm()
            for file_name in cfg["inputs"].get("mc_dplus_norm") or [cfg["inputs"]["mc"][1]])
        data_hdl_bkg_norm = get_corr_bkg_handler(cfg, ipt, cfg["inputs"]["mc_corr_bkg_norm"], pt_min, pt_max)
        data_hdl_template = get_corr_bkg_handler(cfg, ipt, cfg["inputs"]["mc_corr_bkg_shape"], pt_min, pt_max)
        bkg_funcs.insert(0, "hist")
        label_bkg_pdf.insert(0, r'$\mathrm{D^\pm \rightarrow K^\pm K^\mp\pi^\pm}$')

    fitter_pt = F2MassFitter(data_hdl,
                             cfg["fit_configs"]["signal_funcs"][ipt],
                             bkg_funcs,
                             name=f"fit_data_pt{pt_min*10:.0f}_{pt_max*10:.0f}",
                             label_signal_pdf=[r"$\mathrm{D_s^+}$ signal", r"$\mathrm{D^+}$ signal"],
                             label_bkg_pdf=label_bkg_pdf,
                             extended=cfg["fit_configs"]["extended"][ipt]
                             )
    if use_corr_bkg_pt:
        fitter_pt.set_background_template(0, data_hdl_template)
        num = data_hdl_bkg_norm.get_norm() *  cfg["fit_configs"]["correlated_bkg"]["br_pdg"]
        num /= cfg["fit_configs"]["correlated_bkg"]["br_sim"]
        denom = n_dplus_signal * cfg["fit_configs"]["signal_br"]["pdg"]
        denom /= cfg["fit_configs"]["signal_br"]["sim"]
        if cfg["fit_configs"]["fix_correlated_bkg_to_signal"][ipt]:
            fitter_pt.background_pdfs[0]["frac"] = fitter_pt.signal_pdfs[1]["frac"] * num / denom

    fitter_pt.set_particle_mass(0, pdg_id=431, limits=[0., 10])
    fitter_pt.set_particle_mass(1, pdg_id=421, limits=[0., 10])
    fitter_pt.set_signal_initpar(0, "sigma", sigmas_mc[0], limits=[0.005, 0.1])
    fitter_pt.set_signal_initpar(1, "sigma", sigmas_mc[1], limits=[0.005, 0.1])
    if cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"][ipt]:
        if isinstance(cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"][ipt], bool):
            fitter_pt.signal_pdfs[1]["sigma"] = fitter_pt.signal_pdfs[0]["sigma"] * sigmas_mc[1] / sigmas_mc[0]
        else:
            fitter_pt.signal_pdfs[1]["sigma"] = fitter_pt.signal_pdfs[0]["sigma"] * cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"][ipt]  # pylint: disable=line-too-long
    fitter_pt.set_signal_initpar(0, "frac", 0.2, limits=[0., 1.])
    fitter_pt.set_signal_initpar(1, "frac", 0.2, limits=[0., 1.])
    for (i_sig, par_name), value in fixed_pars_mc.items():
        fitter_pt.set_signal_initpar(i_sig, par_name, value, fix=True)
    fix_file_cfg = cfg["fit_configs"]["fix_params_to_file"]
    if fix_file_cfg["do"][ipt]:
        with uproot.open(fix_file_cfg["file"][ipt]) as file_pars:
            for i_sig, suffix in enumerate(SIGNAL_SUFFIXES):
                for par_name in fix_file_cfg["params"][ipt]:
                    value = file_pars[f"h_{par_name}{suffix}"].values()[ipt]
                    fitter_pt.set_signal_initpar(i_sig, par_name, value, fix=True)
    if use_corr_bkg_pt and not cfg["fit_configs"]["fix_correlated_bkg_to_signal"][ipt]:
        fitter_pt.set_background_initpar(0, "frac", 0.5, limits=[0., 1.])
    icombbkg = 1 if use_corr_bkg_pt else 0
    set_comb_bkg_initpars(fitter_pt, icombbkg, bkg_funcs[icombbkg])
    result = fitter_pt.mass_zfit()
    if not result.converged:
        return bin_results

    fig, axs = fitter_pt.plot_mass_fit(
        style="ATLAS",
        figsize=(8, 8),
        axis_title=r"$M(\mathrm{K^\pm K^\mp\pi^\pm})$ (GeV/$c^2$)",
        show_extra_info=True,
        extra_info_loc=["lower right", "lower left"]
    )
    add_info_on_canvas(axs, "upper left", "pp", pt_min, pt_max)

    fig_res, axs_res = fitter_pt.plot_raw_residuals(
        style="ATLAS",
        figsize=(8, 8),
        axis_title=r"$M(\mathrm{K^\pm K^\mp\pi^\pm})$ (GeV/$c^2$)"
    )
    add_info_on_canvas(axs_res, "upper left", "pp", pt_min, pt_max)

    fig.savefig(os.path.join(outdir, f"fit_mass_pt{pt_min*10:.0f}_{pt_max*10:.0f}.pdf"))
    fig_res.savefig(os.path.join(outdir, f"fit_massres_pt{pt_min*10:.0f}_{pt_max*10:.0f}.pdf"))
    plt.close(fig)
    plt.close(fig_res)

    for i_sig in range(len(SIGNAL_SUFFIXES)):
        fit_results = {
            "rawyields": fitter_pt.get_raw_yield(i_sig),
            "significance": fitter_pt.get_significance(i_sig),
            "soverb": fitter_pt.get_signal_over_background(i_sig),
        }
        for name in fitter_pt.signal_pdfs[i_sig].parameters:
            fit_results[name] = fitter_pt.get_signal_parameter(i_sig, name)
        for name, (value, value_unc) in fit_results.items():
            bin_results["data"][(i_sig, name)] = (float(value), float(value_unc))

    with OUTFILE_LOCK or contextlib.nullcontext():
        fitter_pt.dump_to_root(outfile_name, option="update", suffix=f"_pt{pt_min*10:.0f}_{pt_max*10:.0f}")

    return bin_results


def merge_pdfs(outdir, pt_mins, pt_maxs, plot_suffix, merged_file_name):
    """
    Merge the per-pT-bin plots with the given suffix ("" for data, "_mc" for MC) into a single file:
    first the fits of all pT bins, then the residuals of all pT bins, with one bookmark per plot.
    Plots missing for a pT bin are skipped.
    """
    writer = PdfWriter()
    for plot, title in (("fit_mass", "Fits"), ("fit_massres", "Residuals")):
        parent = None
        for pt_min, pt_max in zip(pt_mins, pt_maxs):
            file_name = os.path.join(outdir, f"{plot}_pt{pt_min*10:.0f}_{pt_max*10:.0f}{plot_suffix}.pdf")
            if not os.path.isfile(file_name):
                continue
            page = len(writer.pages)
            writer.append(file_name)
            if parent is None:
                parent = writer.add_outline_item(title, page)
            writer.add_outline_item(f"{pt_min} < pT < {pt_max} GeV/c", page, parent=parent)
    with open(merged_file_name, "wb") as merged_file:
        writer.write(merged_file)


def fit(config_file): # pylint: disable=too-many-locals,too-many-statements, too-many-branches
    """
    Main function for fitting

    Parameters
    ----------

    - config_file (string): config file name
    """

    with open(config_file, "r") as yml_cfg:  # pylint: disable=unspecified-encoding
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    with open(cfg["cutset_file_name"], "r") as yml_cfg:  # pylint: disable=unspecified-encoding
        cut_set = yaml.load(yml_cfg, yaml.FullLoader)

    pt_lims = cut_set["ptbins"]
    pt_mins = pt_lims[:-1]
    pt_maxs = pt_lims[1:]

    fit_cfg = cfg["fit_configs"]
    n_pt = len(pt_mins)
    fit_cfg["extended"] = expand_per_pt(fit_cfg["extended"], n_pt, lambda v: isinstance(v, bool))
    for key in ("signal_funcs", "bkg_funcs"):
        fit_cfg[key] = expand_per_pt(fit_cfg[key], n_pt, lambda v: all(isinstance(f, str) for f in v))
    fit_cfg["fix_params_to_mc"] = expand_per_pt(
        fit_cfg.get("fix_params_to_mc") or [], n_pt, lambda v: all(isinstance(p, str) for p in v))
    fix_file_cfg = fit_cfg.setdefault("fix_params_to_file", {})
    fix_file_cfg["do"] = expand_per_pt(fix_file_cfg.get("do", False), n_pt, lambda v: isinstance(v, bool))
    fix_file_cfg["file"] = expand_per_pt(fix_file_cfg.get("file"), n_pt, lambda v: v is None or isinstance(v, str))
    fix_file_cfg["params"] = expand_per_pt(
        fix_file_cfg.get("params") or [], n_pt, lambda v: all(isinstance(p, str) for p in v))
    fit_cfg["fix_correlated_bkg_to_signal"] = expand_per_pt(
        fit_cfg["fix_correlated_bkg_to_signal"], n_pt, lambda v: isinstance(v, bool))
    fit_cfg["mass_limits"] = expand_per_pt(
        fit_cfg["mass_limits"], n_pt, lambda v: not isinstance(v[0], (list, tuple)))
    fit_cfg["rebin"] = expand_per_pt(fit_cfg["rebin"], n_pt, lambda v: isinstance(v, int))
    fit_cfg["use_bkg_templ"] = expand_per_pt(fit_cfg["use_bkg_templ"], n_pt, lambda v: isinstance(v, bool))
    fit_cfg["fix_ratio_dplus_ds_sigma"] = expand_per_pt(
        fit_cfg["fix_ratio_dplus_ds_sigma"], n_pt, lambda v: isinstance(v, bool))

    outdir = cfg["outputs"]["directory"]
    outfile_name = os.path.join(outdir,
                                f"fit_mass{cfg['outputs']['suffix']}.root")
    file_root = uproot.recreate(outfile_name)
    file_root.close()

    with open(os.path.join(outdir, f"config_fit{cfg['outputs']['suffix']}.yml"), "w") as f:  # pylint: disable=unspecified-encoding
        yaml.dump(cfg, f, sort_keys=False)
    with open(os.path.join(outdir, f"cutset{cfg['outputs']['suffix']}.yml"), "w") as f:  # pylint: disable=unspecified-encoding
        yaml.dump(cut_set, f, sort_keys=False)

    n_jobs = min(cfg.get("n_jobs", 1), n_pt)
    if n_jobs <= 1:
        bin_results = [fit_pt_bin(cfg, ipt, pt_min, pt_max, outfile_name)
                       for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs))]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx,
                                 initializer=init_worker, initargs=(ctx.Lock(),)) as pool:
            try:
                bin_results = list(pool.map(fit_pt_bin, [cfg] * n_pt, range(n_pt), pt_mins, pt_maxs,
                                            [outfile_name] * n_pt))
            except BrokenProcessPool as exc:
                exit_codes = [proc.exitcode for proc in pool._processes.values()]  # pylint: disable=protected-access
                raise RuntimeError(f"A fit worker died, exit codes of the workers: {exit_codes}") from exc

    file_root = uproot.update(outfile_name)
    for res_type, label in (("data", ""), ("mc", "_mc")):
        keys = list(dict.fromkeys(key for res in bin_results for key in res[res_type]))
        for i_sig, name in keys:
            values, uncs = zip(*[res[res_type].get((i_sig, name), (np.nan, np.nan)) for res in bin_results])
            file_root[f"h_{name}{label}{SIGNAL_SUFFIXES[i_sig]}"] = create_hist(pt_lims, values, uncs)

    with uproot.open(cfg["inputs"]["data"]) as data_file:
        for name in ("hLuminosity", "hCollisions"):
            file_root[name] = data_file[name]
    file_root.close()

    for plot_suffix, merged_name in (("", "all"), ("_mc", "all_mc")):
        merge_pdfs(outdir, pt_mins, pt_maxs, plot_suffix,
                   os.path.join(outdir, f"fit_mass_{merged_name}{cfg['outputs']['suffix']}.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", default="config_fit.yml",
                        help="yaml config file for fit")
    args = parser.parse_args()

    fit(args.config)
