"""
script to plot the variance found in SNV data after being transformed in to bed
uses the find all bed to capture all bedfiles in given directory
"""

import sys
from os import path
import numpy as np
from pandas_plink import read_plink
import dask.array as da
import matplotlib.pyplot as plt
from one_hot_encoder import biallelic_to_onehot
from find_all_bed import find_bed_files, read_bed_files
import matplotlib.mlab as mlab

in_file = "/home/hillerton/Data/intersect_1000g_cancer/212_cancer/bed_files"
histogram_file = "/home/hillerton/Data/intersect_1000g_cancer/212c_intersect_variance_hist"
cumulative_file = "/home/hillerton/Data/intersect_1000g_cancer/212c_intersect_variance_cumul"

files = find_bed_files(in_file+"/*")
(bim, fam, G) = read_bed_files(files)
import seaborn as sns

#G, bim =  biallelic_to_onehot.onehot(in_file)
samples = G.shape[1]

mjr = np.max(np.stack([np.sum(G == 0, axis=1),
                       np.sum(G == 1, axis=1),
                       np.sum(G == 2, axis=1)],
                      axis=1),
             axis=1) / samples
#hist = np.histogram(mjr, np.arange(0,1.01,.01))
#plt.plot(hist[1][1:], hist[0])

"""
plt.hist(mjr, bins=100, cumulative=False)
plt.savefig(histogram_file)
#plt.show()
"""

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, len(mjr), 5000))

plt.hist(mjr, bins=100, cumulative=False, facecolor='steelblue', edgecolor='w', linewidth=0.5)
plt.xlabel("Major genotype frequency")
plt.ylabel("Number of SNVs")
plt.grid(alpha=0.3, ls="--")

plt.savefig(cumulative_file)
#plt.show()
