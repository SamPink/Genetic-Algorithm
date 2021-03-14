# -*- coding: utf-8 -*-
"""
Complete this code for continuous optimization  problem

Subtask 1.A. With such a definition in mind, complete the genetic algorithm code
ga-continuous-distrib.py for this continuous optimization problem.

"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
    return an array of instance's of the gene object 
'''


def generate_population():
    return [gene() for _ in range(t.pop)]


'''
    calculates fitness score for each instnce in population

    this is where paramter is changed for setting which problem to be solved
'''


def compute_fitness(p):
    for i in range(t.pop):
        # TODO change code to follow comb
        p[i].fitness = t.problem.problem(p[i].dna)
    return p


'''
    random selection with elitism is used
'''


def selection(population):
    # stort population by fitness descending
    pop_sort = sorted(
        population, key=lambda population: population.fitness, reverse=True)

    # this is the elitism section, take the CROSSOVER_RATE % of the population and move to next generation
    count_survivors = int(t.pop * t.xover)

    # remaining % of population go into random breeding
    children = breeding(pop_sort[count_survivors-1:])

    return pop_sort[:count_survivors] + children


'''
    randomly breed each instances in pool
'''


def breeding(parent_pool):
    children = []
    for _ in range(len(parent_pool)-1):
        parent1 = random.choice(parent_pool)
        parent2 = random.choice(parent_pool)

        # each child is taken into random crossover
        children.append(crossover(parent1.dna, parent2.dna))
    return children


'''
    take random selection of p1 and p2 dna split
'''


def crossover(p1, p2):
    split_point = random.randint(0, len(p1) - 1)
    child = (p1[:split_point] + p2[split_point:])

    # each child is subject to random mutation
    return gene(mutation(child))


'''
    flip random bits in gene dna based on mutation rate
'''


def mutation(gene):
    for i in range(len(gene)):
        # use binary so single bit can be flipped
        binary_gene = str(format(i, '#010b'))
        binary_gene = binary_gene[2:]
        for j in binary_gene:
            if random.uniform(0, 1) <= t.mutation:
                # flip bit
                j = str(int(not int(j)))
        i = int(binary_gene, 2)
    return gene


'''
    perfrom selection then compute fitness
'''


def next_generation(p):
    return compute_fitness(selection(p))


'''
    used to print each element of the population
'''


def print_pop(p):
    return [print(i) for i in p]


'''
    if last 5 values are the same return True
'''


def calc_average_fitness(pop):
    return (sum(p.fitness for p in pop)) / t.pop


def reached_max_fitness(f):
    if len(f) <= 5:
        return False

    lst = f[-5:]

    return all(ele == lst[0] for ele in lst)


"""
This class is being used to store each instance of the population
    each gene has DNA of the length of 20, each 'DNA' is a random number in the global range
    each gene also stores its fintess 
"""


class gene:
    def __init__(self, val=None):
        self.gene_size = 20
        self.dna = []
        self.fitness = -1
        if val:
            self.dna = val
        else:
            for _ in range(self.gene_size):
                self.dna.append(np.random.randint(
                    low=t.lower, high=t.upper))

    def __str__(self) -> str:
        return f'gene: {self.dna} \nfitness: {self.fitness}\n'


"""
Problems contains each of the continous distribution problems
"""


class problems:
    def __init__(self, i) -> None:
        if i == 0:
            self.problem_num = i
            self.name = 'sum of squares'
        elif i == 1:
            self.problem_num = i
            self.name = 'rastigrin'
        elif i == 2:
            self.problem_num = i
            self.name = 'griewank'
        elif i == 3:
            self.problem_num = i
            self.name = 'schafferf7'
        elif i == 4:
            self.problem_num = i
            self.name = 'quartic'
        elif i == 5:
            self.problem_num = i
            self.name = 'whitley'

    def problem(self, gene):
        if self.problem_num == 0:
            return sum(int((t.upper + 1) - abs(i)) for i in gene)
        elif self.problem_num == 1:
            fitness = 10 * len(gene)
            for i in range(len(gene)):
                fitness += gene[i] ** 2 - \
                    (10 * math.cos(2 * math.pi * gene[i]))
            return fitness
        elif self.problem_num == 2:
            part1 = 0
            for i in range(len(gene)):
                part1 += gene[i] ** 2
                part2 = 1
            for i in range(len(gene)):
                part2 *= math.cos(float(gene[i]) / math.sqrt(i + 1))
            return 1 + (float(part1) / 4000.0) - float(part2)
        elif self.problem_num == 3:
            fitness = 0
            normalizer = 1.0 / float(len(gene) - 1)
            for i in range(len(gene) - 1):
                si = math.sqrt(gene[i] ** 2 + gene[i + 1] ** 2)
                fitness += (normalizer * math.sqrt(si) *
                            (math.sin(50 * si ** 0.20) + 1)) ** 2
            return fitness
        elif self.problem_num == 4:
            total = 0.0
            for i in range(len(gene)):
                total += (i + 1.0) * gene[i] ** 4.0
            return total + random.random()
        elif self.problem_num == 5:
            fitness = 0
            limit = len(gene)
            for i in range(limit):
                for j in range(limit):
                    temp = 100*((gene[i]**2)-gene[j]) + \
                        (1-gene[j])**2
                    fitness += (float(temp**2)/4000.0) - math.cos(temp) + 1
            return fitness


class tests:
    def __init__(self, nubmer_of_runs, min_pop_size, max_pop_size, min_gens, max_gens, min_xover, max_xover, min_muts, max_muts, lower, upper, problem_number):
        self.num_of_tests = nubmer_of_runs
        self.pop_size = np.random.uniform(
            low=min_pop_size, high=max_pop_size, size=self.num_of_tests)
        self.gens = np.random.uniform(
            low=min_gens, high=max_gens, size=self.num_of_tests)
        self.xovers = np.random.uniform(
            low=min_xover, high=max_xover, size=self.num_of_tests)
        self.muts = np.random.uniform(
            low=min_muts, high=max_muts, size=self.num_of_tests)
        self.upper = upper
        self.lower = lower
        self.problem = problems(problem_number)

    def set_gen(self, i):
        self.max_fitness = False
        self.pop = int(self.pop_size[i])
        self.gen = int(self.gens[i])
        self.xover = self.xovers[i]
        self.mutation = self.muts[i]


# USE THIS MAIN FUNCTION TO COMPLETE YOUR CODE - MAKE SURE IT WILL RUN FROM COMOND LINE
def main():
    global t

    nubmer_of_runs = 10
    min_pop_size = 10
    max_pop_size = 1000
    min_gens = 100
    max_gens = 1000
    min_xover = 0.1
    max_xover = 0.7
    min_muts = 0.001
    max_muts = 0.3
    lower = 0
    upper = 500
    problem_number = 0

    for p in range(6):

        # instance of the test calss is used to store the paramaters for each run
        t = tests(nubmer_of_runs, min_pop_size, max_pop_size, min_gens, max_gens,
                min_xover, max_xover, min_muts, max_muts, lower, upper, p)

        # dataframe used to store results of each run
        df = pd.DataFrame(columns=['problem name', 'run number', 'population size', 'generations',
                                'crossover', 'mutations', 'fits', 'max fitness'])

        for i in range(t.num_of_tests):

            print(i)

            # set current test number
            t.set_gen(i)

            # used to store average fitness
            fits = []

            # step 1. generate initial population at random
            pop = generate_population()

            for _ in range(t.gen):

                # rank each instance of the population based on fitness (score)
                pop = compute_fitness(pop)

                # apply selection, crossover and mutation
                pop = next_generation(pop)

                # store the average fitness of this generation
                fits.append(calc_average_fitness(pop))

                # if no improvments have been made in 5 genrations
                if reached_max_fitness(fits):
                    t.max_fitness = True
                    break

            # store results for this test
            df.loc[i] = [t.problem.name, i, t.pop,
                        t.gen, t.xover, t.mutation, fits, t.max_fitness]

            #plt.plot(fits)
            #plt.title(t.problem.name)
        #plt.show()
        print(df)
        df.to_csv(f'{t.problem.name}.csv')


if __name__ == '__main__':
    main()
