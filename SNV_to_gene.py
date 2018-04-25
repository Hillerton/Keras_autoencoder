import numpy as np
import os
import glob
from one_hot_encoder import biallelic_to_onehot

dict_file = "/home/hillerton/Data/ref_genes/human_GCRh38_all_genes_NCBI_ID.csv"
path = "/home/hillerton/Data/intersect_1000g_cancer/212_cancer/bed_files"
out_path = "/home/hillerton/Data/ref_genes"

data, bim = biallelic_to_onehot.onehot(path)

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

SNV_number=0

bim = bim.values.tolist()

for line in bim:

    SNV_number+=1

    for chrom in chromomosomes:

        #line = line.strip("\n")
        #split_line = line.split("\t")
        pos = int(line[3])

        if line[0] == chrom:

            for val in gene_dict[chrom]:

                if pos in range(int(val[1]), int(val[2])):
                    gene = (val[0]+"\t"+val[1]+"\t"+val[2]+"\t"+str(line[0])+"\t"+str(line[1])+str(line[2])+"\t"+str(line[3])+"\t"+str(line[5]))
                    nearest_gene.append(gene)

fil = open((out_path+"/genes_with_SNVs_cancer_proj.csv"), "a+")

for i in nearest_gene:
    print (i, file=fil)
fil.close()
