"""
Script for picking out heigh weight SNPs and closes gene to the SNPs
"""

import keras
from keras.models import load_model
import numpy as np
from tree import NearestGene

def get_weights(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            weights = layer.get_weights()
    return weights

path = "/home/hillerton/results/2018-04-05_ae_test/"
model = path+"AE_model.h5"
dict_file = "/home/hillerton/Data/ref_genes/human_GCRh38_swiss_prot_ID.csv"
bim_in = "/home/hillerton/results/2018-04-09_ae_test/master_bim.bim"

ae_model = load_model(model)
weight = get_weights(ae_model, "dense_1")
np.savetxt(path+"weights.csv", weight[0], delimiter="\t")

gene_dict = {}
chromomosomes = list(map(str, range(1,23)))
chromomosomes.append("X")

with open(dict_file, 'r') as fil:
    for line in fil:
        line = line.strip("\n")
        split_line = line.split(",")
        if split_line[2] not in gene_dict.keys() and split_line[2] in chromomosomes:
            q = np.array((split_line[0], int(split_line[3]), int(split_line[4])))
            gene_dict[split_line[2]] = q
        elif split_line[2] in gene_dict.keys() and gene_dict[split_line[2]].ndim == 1:
            q = np.array((split_line[0],int(split_line[3]), int(split_line[4])))
            gene_dict[split_line[2]] = np.concatenate(([gene_dict[split_line[2]]], [q]), axis=0)
        elif split_line[2] in gene_dict.keys():
            q = np.array((split_line[0], int(split_line[3]), int(split_line[4])))
            q = np.reshape(q, (1,3))
            gene_dict[split_line[2]] = np.concatenate((gene_dict[split_line[2]], q), axis=0)

nearest_gene=[]


bim_file = bim_in
with open(bim_file, "r") as bim:
    header = bim.readline()
    for line in bim:
        for chrom in chromomosomes:

            line = line.strip("\n")
            split_line = line.split("\t")
            pos = int(split_line[4])
            if split_line[1] == chrom:
                for val in gene_dict[chrom]:
                    if pos in range(int(val[1]), int(val[2])):
                        gene = (val[0]+"\t"+val[1]+"\t"+val[2]+"\t"+line)
                        nearest_gene.append(gene)

#            gene = NearestGene(gene_dict).nearest(chrom, pos)
#            print (chrom, gene)


weight = np.reshape(weight[0], (5000, 3, 128))
weight = np.swapaxes(weight, 2, 0)

hw = {}
node_count = 0
for line in weight:
    node_mean = np.mean(line)
    std_err = np.std(line)
    up_cut = node_mean + 2 * std_err
    low_cut = node_mean - 2 * std_err

    for i,name in zip(line, nearest_gene):

        for z in i:

            if z > up_cut or z < low_cut:
                if node_count not in hw.keys():
                    hw[node_count] = [name, z]
                else :
                    hw[node_count].append([name, z])

    node_count += 1

for i in hw.keys():
    print (hw[i])
