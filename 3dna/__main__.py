from .RotTable import RotTable
from .Traj3D import Traj3D
from .genetic_optimizer import GeneticOptimizer
from .simulated_annealing_optimizer import SimulatedAnnealingOptimizer

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="input filename of DNA sequence")
    parser.add_argument("--optimize", action="store_true", help="Run genetic optimization")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations for optimization")
    parser.add_argument("--population", type=int, default=50, help="Population size for genetic algorithm")
    args = parser.parse_args()

    # Read file
    with open(args.filename, 'r') as f:
        lineList = [line.rstrip('\n') for line in f]
    dna_sequence = ''.join(lineList[1:])
    if args.optimize:
        # # Genetic optimization
        print("Running genetic optimization...")
        optimizer = GeneticOptimizer(population_size=args.population)
        optimizer.load_table()
        optimized_table = optimizer.optimize(dna_sequence, generations=args.generations)
        optimizer.save_solution('optimized_table.json')
        
        # # Use optimized table
        rot_table = RotTable('optimized_table.json')

        #Annealing optimization
        """print("Running annealing optimization...")
        optimizer1 = SimulatedAnnealingOptimizer(population_size=args.population)
        optimizer1.load_table()
        optimized_table1 = optimizer1.optimize(dna_sequence, generations=args.generations)
        optimizer1.save_solution('optimized_table1.json')"""
        
        # Use optimized table
        rot_table1 = RotTable('optimized_table1.json')
        
    else:
        rot_table = RotTable()

    # # Create trajectory
    # traj = Traj3D()
    # traj.compute(dna_sequence, rot_table)
    # traj.draw()s
    # traj.write(args.filename+".png")

    traj1 = Traj3D()
    traj1.compute(dna_sequence, rot_table)
    traj1.draw()
    traj1.write("solution.png")


if __name__ == "__main__" :
    #python3 -m 3dna data/plasmid_8k.fasta --optimize --generations 200 --population 100
    main()
