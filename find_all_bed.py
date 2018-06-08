"""
supporting function that captures and merges bed files for any analysis that should be performed on several files
"""

import numpy as np
import os, errno
import glob
from pandas_plink import read_plink
import dask.array as da
import pandas as pd

def find_bed_files(filename):
    """Find a set of files that can be loaded with
    pandas_plink.read_plink. The normal case is that the (bim,fam,bed)
    files are specified without prefix, as they are loaded as one. But
    if a .bed suffix is included due to tab expansion, we strip it.
    Also, we allow path name expansion in order to be able to load
    several files at once, should they be split by chromosome.

    """

    if os.path.isfile(filename + ".bed"):
        return [filename]

    base = os.path.splitext(filename)[0]
    if os.path.isfile(filename):
        return [base]

    file_list = glob.glob(filename)
    if len(file_list) > 0:
        return unique_file_no_suffix(file_list)

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

def unique_file_no_suffix(file_list):
    base_list = [os.path.splitext(f)[0] for f in file_list]
    unique_dict = {}
    for f in base_list: unique_dict[f] = True
    return list(unique_dict.keys())

def read_bed_files(file_list):
    """Read one or a set of bed files and accopanying fam and bim files.
    The content is merged into one dask array for bed and one
    dataframe for bim, with correct chromosome ordering (provided that
    there is a strict chromosome ordering within and between files).

    """

    parts = [read_plink(f) for f in file_list]
    #parts = sort_bed_by_chromosome(parts)
    bims, fams, beds = zip(*parts)
    #assert all([len(fam) == len(fams[0]) for fam in fams])

    bim = pd.concat(bims)
    # TODO: Do we want to reindex like this? bim.i = np.arange(1, len(bim)+1)
    fam = fams[0]
    bed = da.concatenate(beds)
    return bim, fam, bed

def sort_bed_by_chromosome(parts):
    annotated = [(int(bim['chrom'].iloc[0]), (bim, fam, bed)) for bim, fam, bed in parts]
    ordered = sorted(annotated)
    return [e[1] for e in ordered]
