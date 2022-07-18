import numpy as np

def get_bin_ranges(ncells, bin_sz, overlap):
    bin_ranges = [[x, x+bin_sz] for x in range(0, ncells, bin_sz - overlap)]
    bin_ranges = [p for p in bin_ranges if p[1] <= ncells]
    bin_ranges[-1][1] = ncells
    return bin_ranges

def get_bins_from_intervals(srt, bin_ranges, bidx0, bidx1):
    pst_bins = [srt[brng[0]:brng[1]] for brng in bin_ranges[bidx0 : bidx1 + 1]]
    return [np.sort(x) for x in pst_bins]

def load_psts(pst_fname):
    """
    Given path to pseudotime file, assumes a header, and separated values "index pseudotime".
    Returns numpy 2d array of floats [[idx1, pst1], [idx2, pst2], ...]
    """
    return np.genfromtxt(pst_fname, skip_header=True, dtype='float')
    