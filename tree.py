GENE_ID = 0
START = 1
STOP = 2

class NearestGene:
    def __init__(self, gene_locations):
        """Args:
            gene_locations (map): map int -> ndarray, where the key is chromosome number and the value is an ndarray with elements [gene_ix, start, stop].
        """

        self.gene_locations = gene_locations

    def nearest(self, chr, pos):
        """Args:
            chr (int): Chromosome
            pos (int): Base pair position withing the chromosome
        """

        chr_genes = self.gene_locations[chr]
        return self.find_nearest(chr_genes, 0, len(chr_genes), pos)

    def find_nearest(self, chr_genes, start, stop, pos):
        middle = int((start+stop)/2)
        # print(middle, start, stop, pos)
        if start == middle:
            return self.nearest_neighbour(chr_genes, pos, start)
        if chr_genes[middle, START] >= pos:
            return self.find_nearest(chr_genes, start, middle, pos)
        else:
            return self.find_nearest(chr_genes, middle, stop, pos)

    def nearest_neighbour(self, chr_genes, pos, start):
        if start+1 >= len(chr_genes) or pos - chr_genes[start, STOP] < chr_genes[start+1, START] - pos:
            return chr_genes[start, GENE_ID]
        else:
            return chr_genes[start+1, GENE_ID]
