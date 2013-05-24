import random


def parse_genes(genome):
    genes = []
    for g in [genome[i:i+5] for i in range(0, len(genome), 5)]:
        genes.append(g)
    return genes


def random_gene():
    reg = random.choice(('+', '-'))
    s0, w0 = str(random.randrange(8)), str(random.randrange(10))
    s1, w1 = str(random.randrange(8)), str(random.randrange(10))
    return ''.join((reg, s0, w0, s1, w1))


def mutate_add_gene(genome):
    return genome + random_gene()


def mutate_del_gene(genome):
    genes = parse_genes(genome)
    if len(genes) == 1:
        return mutate_add_gene(genome)
    i = random.randrange(len(genes))
    del genes[i]
    return ''.join(genes)


def mutate_adddel_gene(genome):
    n = len(genome)//5
    k = random.expovariate(1/3.0)
    if k > n:
        return mutate_add_gene(genome)
    else:
        return mutate_del_gene(genome)


def mutate_flip_regulation(genome):
    genes = parse_genes(genome)
    i = random.randrange(len(genes))
    oldgene = genes[i]
    reg = '+' if oldgene[0] == '-' else '-'
    genes[i] = reg + oldgene[1:]
    return ''.join(genes)


def mutate_change_signal(genome):
    genes = parse_genes(genome)
    i = random.randrange(len(genes))
    oldgene = genes[i]
    idx = random.choice((1, 3))
    newsig = str(random.randrange(8))
    newgene = oldgene[:idx] + newsig + oldgene[idx+1:]
    genes[i] = newgene
    return ''.join(genes)


def mutate_change_weight(genome):
    genes = parse_genes(genome)
    i = random.randrange(len(genes))
    oldgene = genes[i]
    idx = random.choice((2, 4))
    newweight = oldweight = float(oldgene[idx])
    while newweight == oldweight:
        newweight = max(0, min(9, int(round(random.gauss(oldweight, 1)))))
    newgene = oldgene[:idx] + str(newweight) + oldgene[idx+1:]
    genes[i] = newgene
    return ''.join(genes)


mutators = ([mutate_adddel_gene] * 2 + [mutate_flip_regulation] +
            [mutate_change_signal] * 3 + [mutate_change_weight] * 4)


def crossover_concatenate(parents):
    """
    Concatenate genomes, then knock out half the difference in the number
    of genes between the two parents, so the child will have the mean number of
    genes.
    """
    genome = ''.join(parents)
    newlen = (len(parents[0]) + len(parents[1]))//5
    target = newlen // 2 + random.randint(0, 1)
    knockouts = newlen - target
    for i in range(knockouts):
        genome = mutate_del_gene(genome)
    return genome


def crossover_replace(parents):
    """
    Swap half the genes in the second parent with genes in the first parent
    to produce a child.
    """
    genes_a = parse_genes(parents[0])
    genes_b = parse_genes(parents[1])
    swaps = len(genes_b)//2
    for _ in range(swaps):
        i = random.randrange(len(genes_a))
        j = random.randrange(len(genes_b))
        genes_a[i] = genes_b[j]
        del genes_b[j]
    return ''.join(genes_a)


def crossover_pickone(parents):
    """Just go with one parent."""
    return random.choice(parents)


crossovers = [crossover_concatenate, crossover_replace, crossover_pickone]
