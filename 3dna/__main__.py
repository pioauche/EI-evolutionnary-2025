from .RotTable import RotTable
from .Traj3D import Traj3D
from .genetic_optimizer import GeneticOptimizer
from .simulated_annealing_optimizer import SimulatedAnnealingOptimizer

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="input filename of DNA sequence")
    parser.add_argument("--genetic", action="store_true", help="Run genetic optimization")
    parser.add_argument("--annealing", action="store_true", help="Run annealing optimization")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations for optimization")
    parser.add_argument("--population", type=int, default=50, help="Population size for genetic algorithm")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for genetic algorithm")
    parser.add_argument("--crossover_type", type=int, default=1, help="Crossover type for genetic algorithm")
    args = parser.parse_args()

    # Read file
    with open(args.filename, 'r') as f:
        lineList = [line.rstrip('\n') for line in f]
    dna_sequence = ''.join(lineList[1:])

    if args.genetic:
        # Genetic optimization
        print("Running genetic optimization...")
        optimizer = GeneticOptimizer(population_size=args.population, generations=args.generations, mutation_rate=args.mutation_rate, crossover_type=args.crossover_type)
        optimized_table = optimizer.optimize(dna_sequence)
        optimizer.save_solution('optimized_table.json')

        # Use optimized table
        rot_table = RotTable('optimized_table.json') 
    else:
        rot_table = RotTable()

    if args.annealing:
        # Annealing optimization
        print("Running annealing optimization...")
        optimizer_a = SimulatedAnnealingOptimizer(population_size=args.population, generations=args.generations, mutation_rate=args.mutation_rate)
        optimized_table_a = optimizer_a.optimize(dna_sequence)
        optimizer_a.save_solution('optimized_table_a.json')
        
        # Use optimized table
        rot_table_a = RotTable('optimized_table_a.json')
        
    else:
        rot_table_a = RotTable()

    # Create trajectory
    traj = Traj3D('genetic') if args.genetic else Traj3D()
    traj.compute(dna_sequence, rot_table)
    traj.draw()
    traj.write(args.filename+"_genetic.png")

    traj_a = Traj3D('annealing') if args.annealing else Traj3D()
    traj_a.compute(dna_sequence, rot_table_a)
    traj_a.draw()
    traj_a.write(args.filename+"_annealing.png")


if __name__ == "__main__" :
    #Pour les utilisateurs Windows : python -m 3dna data/plasmid_8k.fasta --genetic --annealing --generations 200 --population 100 --mutation_rate 0.2
    #Pour les utilisateurs Mac : python3 -m 3dna data/plasmid_8k.fasta --genetic --annealing --generations 200 --population 100 --mutation_rate 0.2
    main()
