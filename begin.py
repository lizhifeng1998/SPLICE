

import os
import platform
import argparse
from mpi4py import MPI

from utils3 import Param, tree2str, find1, find2, find3


def main_cpu(args):
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print(main_args)
    param = Param(args, rank=comm.Get_rank())
    if args.save_directory is None:
        sd = os.getcwd()
    else:
        sd = args.save_directory
    for seq in param['SequenceTreeList']:
        if args.find_protocol == 1:
            find1(comm, param, seq, sd=sd)
        elif args.find_protocol == 2:
            find2(comm, param, seq, sd=sd)
        elif args.find_protocol == 3:
            find3(comm, param, seq, sd=sd)


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser()
    mode_subparser = main_parser.add_subparsers(dest="mode", help="main arguments")
    mode_subparser.required = True
    cpu_parser = mode_subparser.add_parser("cpu", help="cpu arguments")
    cpu_parser.add_argument('-st', '--sequence_tree', nargs='*')
    cpu_parser.add_argument('-tn', '--turn_number', default=-1, type=int)
    cpu_parser.add_argument('-no', '--number_once', default=-2, type=int)
    cpu_parser.add_argument('-sd', '--save_directory')
    cpu_parser.add_argument('-fp', '--find_protocol', default=3, type=int)
    cpu_parser.add_argument('--user_int1', default=90, type=int)
    cpu_parser.add_argument('--user_int2', default=3333, type=int)
    cpu_parser.add_argument('-xs', '--xtb_solvation', default='')
    cpu_parser.add_argument('-xl', '--xtb_level', default='')
    cpu_parser.add_argument('-ds', '--dftb_solvation', default='')
    cpu_parser.add_argument('-of', '--openmm_ffxml', default='amber14-all.xml')
    cpu_parser.add_argument('-ot', '--openmm_tolerance', default=10, type=float)
    cpu_parser.add_argument('-os', '--openmm_solvation', default=None, choices=['gbn2', 'gbn', 'hct', 'obc1', 'obc2'])
    cpu_parser = mode_subparser.add_parser("tree2str", help="read split_file, then print tree string")
    main_args = main_parser.parse_args()
    if main_args.mode == 'cpu':
        try:
            main_cpu(main_args)
        except Exception as e:
            print(f"An unexpected error occurred: {e}, {type(e)}")
            print()
    elif main_args.mode == 'tree2str':
        print(tree2str())

