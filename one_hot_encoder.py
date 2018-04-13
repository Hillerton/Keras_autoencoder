"""
defines a small class for turning biallelic bed data to onehot encoding
using plink and numpy. Exectued from init_autoencoder to ensure data is one-hot encoded
designed for:
numpy 1.14.0
pandas_plink 1.2.25
"""

from pandas_plink import read_plink
import numpy as np
from find_all_bed import find_bed_files, read_bed_files

class biallelic_to_onehot:
    def onehot(file_prefix, cutoff=False):

        #(bim, fam, bed) = read_plink(data, verbose=False)
        files = find_bed_files(file_prefix+"/*")
        (bim, fam, bed) = read_bed_files(files)

        if cutoff != False:
            bed = bed[:int(cutoff),]

        bed=bed.astype('float32')
        flat_bed = np.reshape(bed, np.prod(bed.shape))
        flat_cls = np.zeros(flat_bed.shape+(3,), dtype=np.float32)
        row_ix = np.where(np.logical_not(np.isnan(flat_bed)))
        col_ix = flat_bed[row_ix].astype(np.int)
        flat_cls[row_ix, col_ix] = 1.0
        cls = np.reshape(flat_cls, bed.shape+(3,))
        return (cls, bim)
