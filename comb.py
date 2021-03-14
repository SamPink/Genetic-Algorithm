import random
import numpy as np

# MINIMUM GLOBAL VARIABLES TO BE USED
POPULATION_SIZE = 500  # Change POPULATION_SIZE to obtain better fitness.

GENERATIONS = 10000  # Change GENERATIONS to obtain better fitness.
SOLUTION_FOUND = False

CROSSOVER_RATE = 0.5  # Change CROSSOVER_RATE  to obtain better fitness.
MUTATION_RATE = 0.05  # Change MUTATION_RATE to obtain better fitness.


class gene(object):
    def __init__(self, val=None):
        self.dna = val or [random_chr() for _ in range(len(problem.target))]
        self.fitness = -1

    def __str__(self) -> str:
        return f'gene: {self.dna} \nfitness: {self.fitness}\n'


class problems:
    def __init__(self, p=0) -> None:
        if p == 0:
            self.p = 0
            self.genes = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
            self.target = '''Test string to solve :)'''
        elif p == 1:
            self.p = 1
            self.genes = [0, 1]
            self.target = [random.randint(0, 1) for _ in range(50)]

    def problem(self, i):
        if self.p == 0:
            return sum(gs != gt for gs, gt in zip(i.dna, problem.target))
        if self.p == 1:
            return i.dna.count(1)


    def generate_population(self):
        if self.p == 0:
            return [gene() for _ in range(POPULATION_SIZE)]
        if self.p == 1:
            # random.choice(self.genes)
            return [gene() for _ in range(POPULATION_SIZE)]


def generate_pop():
    return problem.generate_population()


def compute_fitness(p):
    for i in p:
        i.fitness = problem.problem(i)
    return p


def selection(population):
    # sort the population in increasing order of fitness score
    pop_sort = sorted(population, key=lambda x: x.fitness)

    # crossover rate using rounding to get the values to nearest int
    survivors = int(POPULATION_SIZE * CROSSOVER_RATE)
    num_breeds = int(np.round(survivors, 0))
    crossover_size = int(POPULATION_SIZE * (1 - CROSSOVER_RATE))

    if survivors + crossover_size < POPULATION_SIZE:
        # this is to account for rouding error when spliting population
        crossover_size += 1

    num_survivors = int(np.round(crossover_size, 0))

    survivors = pop_sort[:num_survivors]

    children = []
    for _ in range(num_breeds):
        parent1 = random.choice(population[:50])
        parent2 = random.choice(population[:50])
        child = crossover(parent1.dna, parent2.dna)
        #child = crossover(parent1,parent2)
        children.append(child)

    pop_sort = survivors + children

    return pop_sort


def crossover(p1, p2):
    k = random.randint(0, len(problem.target) - 1)
    split = p1[:k] + p2[k:]

    mutate(split)

    return gene(split)


def mutate(split):
    for i in range(len(split)):
        prob = random.random()

        if prob < MUTATION_RATE:
            split[i] = random_chr()


def random_chr():
    return random.choice(problem.genes)


def print_pop(p, g):
    print(f"Generation: {g}\tString: {p[0].dna}\t Fitness: {p[0].fitness}")


def main():
    global POPULATION_SIZE
    global problem

    # current generation
    generation = 1

    found = False

    problem = problems(0)

    # create initial population
    population = generate_pop()

    print_pop(population, generation)

    while not found:

        population = compute_fitness(population)

        population = selection(population)

        # if the gene having lowest fitness score ie.
        # 0 then we know that we have reached to the target
        # and break the loop
        if population[0].fitness <= 0:
            found = True
            break

        generation += 1

    print_pop(population, generation)


if __name__ == '__main__':
    main()
