"""
Script for the raw-yield extraction of B mesons
"""

import argparse
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position

import numpy as np
import pandas as pd
import uproot
import ROOT
import yaml
from hist import Hist
from matplotlib.offsetbox import AnchoredText
from flarefly import F2MassFitter, DataHandler

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
            fitter.set_background_initpar(ibkg, "c1", -0.05, limits=[-0.2, 0.])
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

    # zfit.run.set_cpus_explicit(
    #     intra=cfg['zfit_n_cpus']['intra'],
    #     inter=cfg['zfit_n_cpus']['inter'],
    # )

    pt_lims = cut_set["ptbins"]
    pt_mins = pt_lims[:-1]
    pt_maxs = pt_lims[1:]

    # settings given only once (bool / single list of functions) are used for all pT bins
    fit_cfg = cfg["fit_configs"]
    n_pt = len(pt_mins)
    fit_cfg["extended"] = expand_per_pt(fit_cfg["extended"], n_pt, lambda v: isinstance(v, bool))
    for key in ("signal_funcs", "bkg_funcs"):
        fit_cfg[key] = expand_per_pt(fit_cfg[key], n_pt, lambda v: all(isinstance(f, str) for f in v))
    fit_cfg["fix_params_to_mc"] = expand_per_pt(
        fit_cfg.get("fix_params_to_mc", []), n_pt, lambda v: all(isinstance(p, str) for p in v))
    fit_cfg["fix_correlated_bkg_to_signal"] = expand_per_pt(
        fit_cfg["fix_correlated_bkg_to_signal"], n_pt, lambda v: isinstance(v, bool))
    fit_cfg["mass_limits"] = expand_per_pt(
        fit_cfg["mass_limits"], n_pt, lambda v: not isinstance(v[0], (list, tuple)))
    fit_cfg["rebin"] = expand_per_pt(fit_cfg["rebin"], n_pt, lambda v: isinstance(v, int))
    fit_cfg["use_bkg_templ"] = expand_per_pt(fit_cfg["use_bkg_templ"], n_pt, lambda v: isinstance(v, bool))
    fit_cfg["fix_ratio_dplus_ds_sigma"] = expand_per_pt(
        fit_cfg["fix_ratio_dplus_ds_sigma"], n_pt, lambda v: isinstance(v, bool))

    # define output file
    outdir = cfg["outputs"]["directory"]
    outfile_name = os.path.join(outdir,
                                f"fit_mass{cfg['outputs']['suffix']}.root")
    file_root = uproot.recreate(outfile_name)
    file_root.close()

    # keep a copy of the configs next to the outputs for reproducibility
    with open(os.path.join(outdir, f"config_fit{cfg['outputs']['suffix']}.yml"), "w") as f:  # pylint: disable=unspecified-encoding
        yaml.dump(cfg, f, sort_keys=False)
    with open(os.path.join(outdir, f"cutset{cfg['outputs']['suffix']}.yml"), "w") as f:  # pylint: disable=unspecified-encoding
        yaml.dump(cut_set, f, sort_keys=False)

    signal_suffixes = ("_ds", "_dplus")
    result_names = ("rawyields", "significance", "soverb", "means", "sigmas")
    results = {(i_sig, name): ([], []) for i_sig in range(len(signal_suffixes)) for name in result_names}
    means_mc, means_mc_unc, sigmas_mc, sigmas_mc_unc = [], [], [], []
    fixed_pars_mc = []  # per pT bin: {(signal index, parameter name): MC value}

    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)): #pylint:disable=too-many-nested-blocks

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
        result = fitter_mc_pt.mass_zfit()
        if result.converged:
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

            mean_ds, mean_ds_unc = fitter_mc_pt.get_signal_parameter(0, "mu")
            sigma_ds, sigma_ds_unc = fitter_mc_pt.get_signal_parameter(0, "sigma")
            mean_dplus, mean_dplus_unc = fitter_mc_pt.get_signal_parameter(1, "mu")
            sigma_dplus, sigma_dplus_unc = fitter_mc_pt.get_signal_parameter(1, "sigma")

            means_mc.append((mean_ds, mean_dplus))
            means_mc_unc.append((mean_ds_unc, mean_dplus_unc))
            sigmas_mc.append((sigma_ds, sigma_dplus))
            sigmas_mc_unc.append((sigma_ds_unc, sigma_dplus_unc))

            fixed_pars_mc.append({
                (i_sig, par_name): fitter_mc_pt.get_signal_parameter(i_sig, par_name)[0]
                for i_sig in range(len(cfg["fit_configs"]["signal_funcs"][ipt]))
                for par_name in cfg["fit_configs"]["fix_params_to_mc"][ipt]
            })
        elif cfg["fit_configs"]["fix_params_to_mc"][ipt]:
            raise RuntimeError(f"MC fit did not converge in pT bin {ipt}, cannot fix parameters to MC")
        else:
            fixed_pars_mc.append({})

        # then we fit data
        data_hdl = DataHandler(cfg["inputs"]["data"], var_name="fM", histoname=f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}",
                               limits=cfg["fit_configs"]["mass_limits"][ipt],
                               rebin=cfg["fit_configs"]["rebin"][ipt])

        bkg_funcs = cfg["fit_configs"]["bkg_funcs"][ipt]
        label_bkg_pdf = ["Comb. bkg"]
        if use_corr_bkg_pt:
            with ROOT.TFile.Open(cfg["inputs"]["mc_corr_bkg"]) as f_mc:
                h_bkg = f_mc[f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}"]
                h_bkg.SetDirectory(0)
            h_bkg.Smooth(1000)
            data_hdl_dplus_signal = DataHandler(cfg["inputs"]["mc"][1], var_name="fM",
                                                histoname=f"hMass_{pt_min*10:.0f}_{pt_max*10:.0f}",
                                                limits=cfg["fit_configs"]["mass_limits"][ipt],
                                                rebin=cfg["fit_configs"]["rebin"][ipt])
            data_hdl_bkg = DataHandler(h_bkg, var_name="fM",
                                       limits=cfg["fit_configs"]["mass_limits"][ipt],
                                       rebin=cfg["fit_configs"]["rebin"][ipt])
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
            fitter_pt.set_background_template(0, data_hdl_bkg)
            num = data_hdl_bkg.get_norm() *  cfg["fit_configs"]["correlated_bkg"]["br_pdg"]
            num /= cfg["fit_configs"]["correlated_bkg"]["br_sim"]
            denom = data_hdl_dplus_signal.get_norm() * cfg["fit_configs"]["signal_br"]["pdg"]
            denom /= cfg["fit_configs"]["signal_br"]["sim"]
            if cfg["fit_configs"]["fix_correlated_bkg_to_signal"][ipt]:
                fitter_pt.background_pdfs[0]["frac"] = fitter_pt.signal_pdfs[1]["frac"] * num / denom

        fitter_pt.set_particle_mass(0, pdg_id=431, limits=[0., 10])
        fitter_pt.set_particle_mass(1, pdg_id=421, limits=[0., 10])
        fitter_pt.set_signal_initpar(0, "sigma", sigmas_mc[ipt][0], limits=[0.005, 0.1])
        fitter_pt.set_signal_initpar(1, "sigma", sigmas_mc[ipt][1], limits=[0.005, 0.1])
        if cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"][ipt]:
            if isinstance(cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"][ipt], bool):
                fitter_pt.signal_pdfs[1]["sigma"] = fitter_pt.signal_pdfs[0]["sigma"] * sigmas_mc[ipt][1] / sigmas_mc[ipt][0]  # pylint: disable=line-too-long
            else:
                fitter_pt.signal_pdfs[1]["sigma"] = fitter_pt.signal_pdfs[0]["sigma"] * cfg["fit_configs"]["fix_ratio_dplus_ds_sigma"][ipt]  # pylint: disable=line-too-long
        fitter_pt.set_signal_initpar(0, "frac", 0.2, limits=[0., 1.])
        fitter_pt.set_signal_initpar(1, "frac", 0.2, limits=[0., 1.])
        for (i_sig, par_name), value in fixed_pars_mc[ipt].items():
            fitter_pt.set_signal_initpar(i_sig, par_name, value, fix=True)
        if use_corr_bkg_pt and not cfg["fit_configs"]["fix_correlated_bkg_to_signal"][ipt]:
            fitter_pt.set_background_initpar(0, "frac", 0.5, limits=[0., 1.])
        icombbkg = 1 if use_corr_bkg_pt else 0
        set_comb_bkg_initpars(fitter_pt, icombbkg, bkg_funcs[icombbkg])
        result = fitter_pt.mass_zfit()
        if result.converged:
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

            for i_sig in range(len(signal_suffixes)):
                fit_results = {
                    "rawyields": fitter_pt.get_raw_yield(i_sig),
                    "significance": fitter_pt.get_significance(i_sig),
                    "soverb": fitter_pt.get_signal_over_background(i_sig),
                    "means": fitter_pt.get_signal_parameter(i_sig, "mu"),
                    "sigmas": fitter_pt.get_signal_parameter(i_sig, "sigma"),
                }
                for name, (value, value_unc) in fit_results.items():
                    results[(i_sig, name)][0].append(value)
                    results[(i_sig, name)][1].append(value_unc)

            fitter_pt.dump_to_root(
                outfile_name, option="update", suffix=f"_pt{pt_min*10:.0f}_{pt_max*10:.0f}")

    file_root = uproot.update(outfile_name)
    for (i_sig, name), (values, uncs) in results.items():
        file_root[f"h_{name}{signal_suffixes[i_sig]}"] = create_hist(pt_lims, values, uncs)
    for i_sig, label in enumerate(("ds", "dplus")):
        file_root[f"h_means_mc_{label}"] = create_hist(
            pt_lims, [mean[i_sig] for mean in means_mc], [unc[i_sig] for unc in means_mc_unc])
        file_root[f"h_sigmas_mc_{label}"] = create_hist(
            pt_lims, [sigma[i_sig] for sigma in sigmas_mc], [unc[i_sig] for unc in sigmas_mc_unc])

    with uproot.open(cfg["inputs"]["data"]) as data_file:
        for name in ("hLuminosity", "hCollisions"):
            file_root[name] = data_file[name]
    file_root.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", default="config_fit.yml",
                        help="yaml config file for fit")
    args = parser.parse_args()

    fit(args.config)
