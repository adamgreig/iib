from iib.server import operators
import multiprocessing
import numpy as np
import random


def random_genome():
    """
    Make a random genome such that at least one diffusing and one non-diffusing
    signal are expressed.
    """
    genes = []
    while True:
        genes.append(operators.random_gene())
        s = {int(g[3])//4 for g in genes}
        if 0 in s and 1 in s:
            break
    return ''.join(genes)


def first_generation(n):
    """Create the initial generation of *n* individuals."""
    return [random_genome() for i in range(n)]


def select_parents(n, generation):
    """
    Use stochastic universal sampling to select *n* parents from *generation*.
    """
    generation = sorted(generation, key=lambda i: float(i[1]), reverse=True)
    scores = np.array([float(individual[1]) for individual in generation])
    scores /= np.sum(scores)
    scores = np.add.accumulate(scores)
    position = random.uniform(0, 1/n)
    parents = []
    for idx, score in enumerate(scores):
        if score > position:
            parents.append(generation[idx][0])
            position += 1/n
    return parents


def crossover(parents):
    """Create a new genome by crossing over the two *parents*."""
    return random.choice(operators.crossovers)(parents)


def mutate(child):
    """Modify the single *child* genome by mutating it."""
    return random.choice(operators.mutators)(child)


def new_child(parents):
    myparents = random.sample(parents, 2)
    return mutate(crossover(myparents))


def new_generation(old):
    """Create a new generation from *old* of the same size."""
    n_parents = len(old) // 10
    n_elites = len(old) // 100
    parents = select_parents(n_parents, old)
    elites = select_parents(n_elites, old)
    results = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for _ in range(len(old) - n_elites):
        results.append(pool.apply_async(new_child, [parents]))
    pool.close()
    pool.join()
    children = [result.get() for result in results]
    return elites + children


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    def fake_scores(generation):
        out = []
        for i in generation:
            out.append([i, random.randrange(100)])
        return out

    lengths = []
    t0 = time.time()
    cohort = fake_scores(first_generation(1000))
    iters = 100
    for i in range(iters):
        cohort = fake_scores(new_generation(cohort))
        lengths.append(sum(len(i[0])/5 for i in cohort) / len(cohort))
        t = time.time() - t0
        print("iter {0}, {1:.2f}, {2:.2f}s".format(i, lengths[-1], t))
    print("Average time/iter: {0:.2f}".format((time.time() - t0)/iters))
    plt.plot(lengths)
    plt.show()
