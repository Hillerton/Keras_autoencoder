"""
Script for picking out heigh weight SNPs and closes gene to the SNPs
"""

import keras
from keras.models import load_model
import numpy as np
from tree import NearestGene
import time
import os
import glob

def get_weights(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            weights = layer.get_weights()
    return weights

today = time.strftime("%Y-%m-%d")

path = "/home/hillerton/results/2018-04-18_ae_test/"
model = path+"AE_model.h5"
dict_file = "/home/hillerton/Data/ref_genes/human_GCRh38_all_genes_NCBI_ID.csv"
bim_in = "/home/hillerton/results/2018-04-18_ae_test/master_bim.bim"
out_path = "/home/hillerton/results/"

out_fil = out_path+today+"_high_weight_SNVs/"
if not glob.glob(out_fil):
    os.makedirs(out_fil)

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
    SNV_number=0
    header = bim.readline()
    for line in bim:
        SNV_number+=1
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



weight = np.reshape(weight[0], (SNV_number, 3, 128))
weight = np.swapaxes(weight, 2, 0)


hw = {}
node_count = 1
seen_gene = []
for line in weight:
    node_mean = np.mean(line)
    std_err = np.std(line)
    up_cut = node_mean + 3 * std_err
    low_cut = node_mean - 3 * std_err

    for i in line:

        for z,name in zip(i, nearest_gene):

            if z > up_cut or z < low_cut:
                if node_count not in hw.keys():
                    hw[node_count] = [name+"\t"+str(z)]
                    seen_gene.append(name+"\t"+str(z)+"\tnot_hw")
                else :
                    hw[node_count].append(name+"\t"+str(z))
                    seen_gene.append(name+"\t"+str(z)+"\tnot_hw")
            elif name in seen_gene:
                hw[node_count].append(name+"\t"+str(z))

    node_count += 1

node = ""

for i in hw.keys():
    fil = open(out_fil+"node"+str(i)+"_SNVs.csv", "w+")
    name_fil = open(out_fil+"node_gene_names.txt", "a+")
    print ("GeneID\tGene_Start\tGene_Stop\tchrom\tSNV_name\tposition (morgans)\tposition\tAllele1\tAllele2", file=fil)


    for j in hw[i]:
        split_line = j.split("\t")
        print (j, file=fil)

        node+=split_line[0]+"\t"+split_line[-1]

    print (node, file=name_fil)

    fil.close()
    name_fil.close()
