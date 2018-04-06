GENE_ID = 0
START = 1
STOP = 2

class NearestGene:
    def __init__(self, gene_locations):
        """Args:
            gene_locations (map): map int -> ndarray, where the key is chromosome number and the value is an ndarray with elements [gene_ix, start, stop].
        """

        self.gene_locations = gene_locations

    def nearest(self, chrom, pos):
        """Args:
            chrom (int): chromomosome
            pos (int): Base pair position withing the chromomosome
        """

        chrom_genes = self.gene_locations[chrom]
        return self.find_nearest(chrom_genes, 0, len(chrom_genes), pos)

    def find_nearest(self, chrom_genes, start, stop, pos):
        middle = int((start+stop)/2)
        
        if start == middle:
            return self.nearest_neighbour(chrom_genes, pos, start)
        if chrom_genes[middle, START] >= pos:
            return self.find_nearest(chrom_genes, start, middle, pos)
        else:
            return self.find_nearest(chrom_genes, middle, stop, pos)

    def nearest_neighbour(self, chrom_genes, pos, start):
        if start+1 >= len(chrom_genes) or pos - chrom_genes[start, STOP] < chrom_genes[start+1, START] - pos:
            return chrom_genes[start, GENE_ID]
        else:
            return chrom_genes[start+1, GENE_ID]
