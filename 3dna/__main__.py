from .RotTable import RotTable
from .Traj3D import Traj3D
from .genetic_optimizer import GeneticOptimizer

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
    seq = ''.join(lineList[1:])
    print (seq)
    if args.optimize:
        print("Running genetic optimization...")
        optimizer = GeneticOptimizer(population_size=args.population)
        optimizer.load_table()
        optimized_table = optimizer.optimize(seq, generations=args.generations)
        optimizer.save_solution('optimized_table.json')
        
        # Use optimized table
        rot_table = RotTable()
        rot_table.table = optimized_table
    else:
        rot_table = RotTable()

    # Create trajectory
    traj = Traj3D()
    traj.compute(seq, rot_table)
    traj.draw()
    traj.write(args.filename+".png")


if __name__ == "__main__" :
    main()
