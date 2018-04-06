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
bim_path = "/home/hillerton/Data/intersect_1000g_cancer/212_cancer/bed_files"

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
"""
nearest_gene=[]

for chrom in chromomosomes:
    print ("working on chromosome", chrom)
    bim_file = bim_path+"/chr"+chrom+".bim"
    with open(bim_file, "r") as bim:
        count = 0
        for line in bim:
            line = line.strip("\n")
            split_line = line.split("\t")
            pos = int(split_line[3])
            for val in gene_dict[chrom]:
                if pos in range(int(val[1]), int(val[2])):
                    gene = (str(count)+"\t"+val[0]+"\t"+val[1]+"\t"+val[2]+"\t"+line)
                    nearest_gene.append(gene)
            count+=1

for i in nearest_gene:
    print (i)

#            gene = NearestGene(gene_dict).nearest(chrom, pos)
#            print (chrom, gene)
"""

weight = np.reshape(weight[0], (5000, 3, 128))
weight = np.swapaxes(weight, 2, 0)

hw = {}
node_count = 0
for line in weight:
    node_mean = np.mean(line)
    std_err = np.std(line)
    up_cut = node_mean + 2 * std_err
    low_cut = node_mean - 2 * std_err

    for i in line:
        count = 0

        for z in i:
            count += 1

            if z > up_cut or z < low_cut:
                if node_count not in hw.keys():
                    hw[node_count] = [count, z]
                else :
                    hw[node_count].append([count, z])

    node_count += 1

print (hw)
