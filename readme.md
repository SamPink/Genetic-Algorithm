# how to run cont.py 
file is run from main function 

paramaters

 - nubmer_of_runs = 10
 - min_pop_size = 10
 - max_pop_size = 1000
 - min_gens = 100
 - max_gens = 1000
 - min_xover = 0.1
 - max_xover = 0.7
 - min_muts = 0.001
 - max_muts = 0.3
 - lower = 0
 - upper = 500

each of these paramaters defines the limitation of the random values set for each run number

problem_number - this value sets the problem to be calculated

 1. problem_number = 0 = sum of squares
 2.  problem_number = 1 = rastigrin_function
 3.  problem_number = 2 = griewank_function
 4.  problem_number = 3 = schafferf7
 5.  problem_number = 4 = qudratic
 6.  problem_number = 5 = Whitley

# how to run comb.py

file is run from main 

problem = problems(0)

this value can be set to either 1 or 2. 1 for solving numbers and 2 for solving strings