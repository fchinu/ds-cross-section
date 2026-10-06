"""
Reweight MC THnSparses with pT weights
"""
import numpy as np
import ROOT


class SparseReweighter:  # pylint: disable=too-few-public-methods
    """
    Apply pT weights to MC THnSparses. Prompt candidates are weighted vs the D-meson pT,
    non-prompt candidates vs the pT of the B-hadron mother.
    """

    def __init__(self, pt_weights_cfg, axes):
        """
        Args:
            pt_weights_cfg (dict): pT-weights configuration, with a structure like:
                file_name: 'ptweights.root'
                hist_names:
                    Ds:
                        Prompt: 'hist_ds_weights'
                        NonPrompt: 'hist_bs_weights'
                    Dplus:
                        Prompt: 'hist_d_weights'
                        NonPrompt: 'hist_b_weights'
            axes (dict): sparse axis indices: axis_pt_reco, axis_ptb_reco, axis_pt_gen, axis_ptb_gen
        """
        self.axes = axes
        self.histos = {}
        with ROOT.TFile.Open(pt_weights_cfg["file_name"]) as weights_file:
            for particle, hist_names in pt_weights_cfg["hist_names"].items():
                for origin, hist_name in hist_names.items():
                    histo = weights_file.Get(hist_name)
                    if not histo:
                        raise KeyError(f"pT-weight histogram {hist_name} not found in {pt_weights_cfg['file_name']}")
                    histo.SetDirectory(0)
                    self.histos[(particle, origin)] = histo

    def reweight(self, sparse, particle, origin, is_gen):
        """
        Return a copy of the sparse with the content and error of each bin scaled by its pT weight.

        Args:
            sparse (THnSparse): sparse to reweight
            particle (str): 'Ds' or 'Dplus'
            origin (str): 'Prompt' or 'NonPrompt'
            is_gen (bool): True for generated-particle sparses, False for reconstructed candidates

        Returns:
            THnSparse: the reweighted sparse
        """
        histo = self.histos[(particle, origin)]
        variable = "pt" if origin == "Prompt" else "ptb"
        axis_key = f"axis_{variable}_{'gen' if is_gen else 'reco'}"
        i_axis = self.axes[axis_key]
        if i_axis >= sparse.GetNdimensions():
            raise ValueError(f"{axis_key} = {i_axis}, but sparse {sparse.GetName()} only has "
                             f"{sparse.GetNdimensions()} axes")

        axis = sparse.GetAxis(i_axis)
        bin_weights = [histo.GetBinContent(histo.FindBin(axis.GetBinCenter(i_bin)))
                       for i_bin in range(axis.GetNbins() + 2)]

        reweighted = sparse.Clone()
        ROOT.SetOwnership(reweighted, True)  # otherwise the clone is never freed
        # select the bin-index setters explicitly: otherwise cppyy can pass bin 0 as a null
        # pointer to the (const Int_t* coordinates, Double_t) overloads, which segfaults
        set_content = reweighted.SetBinContent.__overload__("Long64_t,Double_t")
        set_error = reweighted.SetBinError.__overload__("Long64_t,Double_t")
        coord = np.zeros(reweighted.GetNdimensions(), dtype=np.int32)
        for i_bin in range(reweighted.GetNbins()):
            content = reweighted.GetBinContent(i_bin, coord)
            error = np.sqrt(reweighted.GetBinError2(i_bin))
            weight = bin_weights[coord[i_axis]]
            set_content(i_bin, content * weight)
            set_error(i_bin, error * weight)

        return reweighted
