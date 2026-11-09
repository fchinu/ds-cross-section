"""Script to create templates from MC histograms."""
import argparse
import ROOT

def double_gaus(m, pars):
    """
    Double gaussian function for smoothing    
    """

    gaus_1 = ROOT.TMath.Gaus(m[0], pars[2], pars[3], True)
    gaus_2 = ROOT.TMath.Gaus(m[0], pars[4], pars[5], True)

    return pars[0] * (pars[1] * gaus_1 + (1-pars[1]) * gaus_2)

def make_templates(infile_name, out_file):
    """
    Main function to create templates
    """

    with ROOT.TFile.Open(infile_name) as infile:
        hist_orig, hist_templ, func_templ = [], [], []

        for key in infile.GetListOfKeys():
            if "hMass" in key.GetName():
                hist_name = key.GetName()
                hist_orig.append(infile.Get(hist_name))
                hist_orig[-1].SetDirectory(0)
                func_templ.append(
                    ROOT.TF1(hist_name.replace("hMass", "f_mass"), double_gaus, 1.95, 2.15, 6))
                func_templ[-1].SetParameters(hist_orig[-1].Integral(), 0.5, 2.05, 0.015, 2.14, 0.030)
                func_templ[-1].SetParLimits(0, 0., 1.e6) # this is an integral
                func_templ[-1].SetParLimits(1, 0.3, 0.7) # this is a fraction (we want more or less 50-50)
                func_templ[-1].SetParLimits(2, 1.95, 2.15) # the mean must be within the range
                func_templ[-1].SetParLimits(4, 1.95, 2.15) # the mean must be within the range
                hist_templ.append(hist_orig[-1].Clone(hist_name))
                hist_templ[-1].SetDirectory(0)
                hist_orig[-1].SetName(hist_name.replace("hMass", "hMass_orig"))
                if hist_templ[-1].GetEntries() > 30:
                    hist_orig[-1].Fit(func_templ[-1], "RL")
                    for ibin in range(1, hist_templ[-1].GetNbinsX()+1):
                        mass = hist_templ[-1].GetBinCenter(ibin)
                        hist_templ[-1].SetBinContent(ibin, func_templ[-1].Eval(mass))

    output = ROOT.TFile(out_file, "recreate")
    for hist in hist_orig:
        hist.Write()
    for hist in hist_templ:
        hist.Write()
    output.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get D+ mass templates from MC")
    parser.add_argument("input", type=str, help="Input ROOT file with histograms")
    parser.add_argument("output", type=str, help="Output ROOT file to save templates")
    args = parser.parse_args()

    make_templates(args.input, args.output)
