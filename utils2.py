# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 2024

@author: lzf
"""

import os
import sys
import time
import math
import glob
import shutil
import pickle
import platform
import textwrap
import numpy as np
from mpi4py import MPI
from copy import deepcopy
from functools import wraps
from tempfile import mkdtemp

import zmat2xyz
import template_generate
from xyz2pdb_graph2 import cut_conh
from utils1 import sidechain_dihedrals, ReadPdbAtom
from utils1 import GetDihs, xyz2pdb, openmm_opt, xyz_dftbin, openmm_opt_amber

ROOT_RANK = 0


def prothod_mquchong_tfd_initial_thresh(seq, mini_angle=25, tfd_value=5, 
                                        factor=0.5, threshod_max=180, 
                                        max_quanzhong=12, angle_initial=45, 
                                        angle_side_base=100, 
                                        ang_gap_min_2r_side_chain=50, ni=1):
    """
    Get the number of structures from Prothod_mquchong_tfd_initial_thresh.c

    Parameters
    ----------
    seq : str
        One-letter sequence.
    mini_angle : float, optional
        Max angle for two same structures. The default is 25.
    tfd_value : float, optional
        Max tfd for two same structures. The default is 5.
    factor : float, optional
        Side-chain weight decrease. The default is 0.5.
    threshod_max : float, optional
        Prothod_mquchong_tfd_initial_thresh.c parameter. The default is 180.
    max_quanzhong : float, optional
        Max main-chain weight. The default is 12.
    angle_initial : float, optional
        Prothod_mquchong_tfd_initial_thresh.c parameter. The default is 45.
    angle_side_base : float, optional
        Prothod_mquchong_tfd_initial_thresh.c parameter. The default is 100.
    ang_gap_min_2r_side_chain : float, optional
        Prothod_mquchong_tfd_initial_thresh.c parameter. The default is 50.
    ni : int, optional
        Rounds take. The default is 1.

    Returns
    -------
    n : int
        The number of unique structures.

    """
    dir0 = os.getcwd()
    if os.path.isdir('qc'):
        shutil.rmtree('qc')
    os.mkdir('qc')
    shutil.copy("dihedral_angles_list", "qc")
    shutil.copy("dihedral_angles_list", "qc/unique_list.txt")
    os.chdir("qc")
    args = '%s %f 2 %f l %f %f %f %f %f %f' % (
        seq, mini_angle, tfd_value, factor, threshod_max, max_quanzhong, 
        angle_initial, angle_side_base, ang_gap_min_2r_side_chain)
    for _ in range(ni):
        os.system(f'Prothod_mquchong_tfd_initial_thresh.c {args} >/dev/null') 
        shutil.copy("dihedral_angles_list2", "dihedral_angles_list")
        shutil.copy("dihedral_angles_list2", "unique_list.txt")
    os.chdir(dir0)
    with open('qc/dihedral_angles_list2', 'r') as f:
        n = len(f.readlines())
    shutil.rmtree("qc")
    return n


def get_tfd_max(seq, factor=0.8, max_quanzhong=4.):
    """
    Get the max possible tfd for a given sequence.

    Parameters
    ----------
    seq : str
        One-letter sequence.
    factor : float, optional
        Parameter in prothod_mquchong_tfd_initial_thresh. The default is 0.8.
    max_quanzhong : float, optional
        Parameter in prothod_mquchong_tfd_initial_thresh. The default is 4..

    Returns
    -------
    tfd_max : float
        Max possible tfd.

    """
    w = [min(i+1., max_quanzhong) for i in range(len(seq))]
    for i in range(len(seq), 0, -1):
        w.append(min(i+0., max_quanzhong))
    w.append(min(1., max_quanzhong))
    for i in range(len(seq), 0, -1):
        for j in range(len(sidechain_dihedrals[seq[-i]])):
            w.append(max(0., 1.-j*factor))
        if seq[-i] == 'P':
            w.append(w[2*i])
            w.append(w[2*i+1])
    tfd_max = sum([(180*w[i])**2 for i in range(len(w))])**0.5/len(w)
    return tfd_max


def backup_file(filename):
    """
    Backup file.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    None.

    """
    if not os.path.isfile(filename):
        return
    i = 0
    while os.path.isfile('%s.bak%d' % (filename, i)):
        i += 1
    os.rename(filename, '%s.bak%d' % (filename, i))


def end_file_open(filename, ):
    """open a read-only file whose final line should be END
    """
    while True:
        while not os.path.isfile(filename):
            time.sleep(1)
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0 and lines[-1].strip() == "END":
                break
        time.sleep(1)
    return open(filename, 'r')


def read_namelist(fn='uni_diha_list.txt'):
    """
    Get the first column in fn.

    Parameters
    ----------
    fn : str, optional
        Filename. The default is 'uni_diha_list.txt'.

    Returns
    -------
    namelist : list
        The first column.

    """
    namelist = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            namelist.append(line.strip().split()[0])
    return namelist


def write_txt(dih_dict, seq):
    """
    Write z-matrix file in txt format given template.

    Parameters
    ----------
    dih_dict : dict
        {name: {'dihl': []},}.
    seq : str
        One-letter sequence.

    Returns
    -------
    None.

    """
    try:
        with open('template', 'r') as f:
            lt = f.readlines()
    except FileNotFoundError:
        with open('../template', 'r') as f:
            lt = f.readlines()
    # M1, s11
    ms = [x.strip().split()[-1] for x in lt[5:] if 'M' in x or 's' in x]
    for k in dih_dict.keys():
        dihs = dih_dict[k]['dihl']
        with open('%s.txt' % k, 'w') as f:
            for x in lt:
                f.write(x)
            f.write('\n')
            o = 0 # the number of angle in dihs but not template
            for j in range(len(ms)):
                if ms[j][0] == 's':
                    side_i = int(ms[j][1:-1]) - 1 # @ in s@?
                else:
                    side_i = None
                # s?5 for K(LYN), 2 atoms center -> 1 atom
                if ms[j][0] == 's' and seq[side_i] == 'K' and ms[j][-1] == '5':
                    # -60 degree
                    if dihs[j+o] > -120:
                        f.write(f'{ms[j]} {dihs[j+o]-60:.2f}\n')
                    else:
                        f.write(f'{ms[j]} {dihs[j+o]+300:.2f}\n')
                else:
                    try:
                        f.write("%s %.2f\n" % (ms[j], dihs[j+o]))
                    except:
                        print(f'{seq}: {ms} {j}, {dihs} {j+o}')
                        raise
                # whether degenerate
                if ms[j][0] == 's':
                    sd = sidechain_dihedrals[seq[side_i]][-1]
                    last = len(sidechain_dihedrals[seq[side_i]]) - 1
                    if int(ms[j][-1]) == last and sd[0] == sd[3]:
                        o += 1
                if ms[j][0] == 'M' or seq[side_i] != 'P':
                    continue
                o += 1
                if side_i == 0 or seq[side_i-1] != 'P':
                    # be the first or not behind PRO
                    pass
                f.write("p%s1 %.2f\n" % (ms[j][1:-1], dihs[j+o]))
                o += 1
                if side_i + 1 != len(seq) and seq[side_i+1] != 'P':
                    # not the last nor in front of PRO
                    f.write("p%s2 %.2f\n" % (ms[j][1:-1], dihs[j+o]))
            f.write('\n')
    return


def calc_master(comm, job_list):
    size = comm.Get_size()
    status = MPI.Status()
    numjobs = len(job_list)
    rank_ndx = [-1] * size
    results = [''] * numjobs
    for rank in range(size):
        if rank == ROOT_RANK:
            continue
        i = rank if rank < ROOT_RANK else rank - 1
        if i == numjobs:
            break
        comm.send(job_list[i], rank, 0)
        rank_ndx[rank] = i
    recv_n, recv_n0 = 0, 0
    for i in range(size - 1, numjobs):
        e = comm.recv(status=status)
        if isinstance(e, Exception):
            raise
        recv_n += 1
        if recv_n - recv_n0 >= numjobs / 100:
            print('%d %%' % (100 * recv_n / numjobs), end='\r', flush=True)
            recv_n0 = recv_n
        rank = status.Get_source()
        results[rank_ndx[rank]] = e
        comm.send(job_list[i], rank, 0)
        rank_ndx[rank] = i
    for i in range(min(numjobs, size - 1)):
        e = comm.recv(status=status)
        if isinstance(e, Exception):
            raise
        recv_n += 1
        if recv_n - recv_n0 >= numjobs / 100:
            print('%d %%' % (100 * recv_n / numjobs), end='\r', flush=True)
            recv_n0 = recv_n
        rank = status.Get_source()
        results[rank_ndx[rank]] = e
    for rank in range(size):
        if rank == ROOT_RANK:
            continue
        comm.send('', rank, 1)
    print()
    return results


def calc_master_generator(comm, job_list):
    size = comm.Get_size()
    status = MPI.Status()
    numjobs = len(job_list)
    rank_ndx = [-1] * size
    for rank in range(size):
        if rank == ROOT_RANK:
            continue
        i = rank if rank < ROOT_RANK else rank - 1
        if i == numjobs:
            break
        comm.send(job_list[i], rank, 0)
        rank_ndx[rank] = i
    recv_n, recv_n0 = 0, 0
    for i in range(size - 1, numjobs):
        e = comm.recv(status=status)
        if isinstance(e, Exception):
            raise
        recv_n += 1
        if recv_n - recv_n0 >= numjobs / 100:
            print('%d %%' % (100 * recv_n / numjobs), end='\r', flush=True)
            recv_n0 = recv_n
        rank = status.Get_source()
        yield (rank_ndx[rank], e)
        comm.send(job_list[i], rank, 0)
        rank_ndx[rank] = i
    for i in range(min(numjobs, size - 1)):
        e = comm.recv(status=status)
        if isinstance(e, Exception):
            raise
        recv_n += 1
        if recv_n - recv_n0 >= numjobs / 100:
            print('%d %%' % (100 * recv_n / numjobs), end='\r', flush=True)
            recv_n0 = recv_n
        rank = status.Get_source()
        yield (rank_ndx[rank], e)
    for rank in range(size):
        if rank == ROOT_RANK:
            continue
        comm.send('', rank, 1)
    print()


def calc_slave(comm, calc_func, *args, **kwargs):
    status = MPI.Status()
    while True:
        job = comm.recv(source=ROOT_RANK, status=status)
        if status.Get_tag():
            break
        try:
            comm.send(calc_func(job, *args, **kwargs), ROOT_RANK, 0)
        except Exception as e:
            comm.send(e, ROOT_RANK, 0)
            raise
    return


class Timer:
    def __init__(self, key):
        self.key = key

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            mu = args[0]
            start = time.time()
            result = func(*args, **kwargs)
            last = time.time() - start
            if self.key in mu.time.keys():
                mu.time[self.key] += last
            else:
                mu.time[self.key] = last
            return result
        return wrapped_function


class DFTBhsd(dict):
    def init2(self, seq=''):
        dict.__init__(self)
        self['Geometry'] = 'GenFormat {\n\t<<<INPUT.txt\n}'
        # self['Driver'] = 'GeometryOptimization {}'
        self['Driver = ConjugateGradient'] = {
            'MovedAtoms': '1:-1', 'MaxForceComponent': '1e-4', 
            'MaxSteps': 2000, 'OutputPrefix': '"opt"'}
        self['Hamiltonian = DFTB'] = {
            'SCC': 'Yes',
            'MaxAngularMomentum': {'H': '"s"', 'C': '"p"', 
                                   'N': '"p"', 'O': '"p"'},
            'HubbardDerivs': {'C': '-0.1492', 'H': '-0.1857', 
                              'N': '-0.1535', 'O': '-0.1575'},
            'ThirdOrderFull': 'Yes',
            'HCorrection': 'Damping { Exponent = 4.0 }',
            'SlaterKosterFiles = Type2FileNames': {
                'Prefix': '"%s/dftb+1.2.2/slako/3ob-3-1/"' % (
                    os.path.expanduser('~')),
                'Separator': '"-"', 'Suffix': '".skf"'},
            'Dispersion = DFTD4': {'s6': '1.0', 's9': '0.0', 's8': '0.4727337',
                                   'a1': '0.5467502', 'a2': '4.4955068'}
        }

    def __init__(self, seq=''):
        dict.__init__(self)
        self['Geometry'] = 'GenFormat {\n\t<<<INPUT.txt\n}'
        self['Driver = ConjugateGradient'] = {
            'MovedAtoms': '1:-1', 'MaxForceComponent': '1e-4', 
            'MaxSteps': 2000, 'OutputPrefix': '"opt"'}
        self['Hamiltonian = DFTB'] = {
            'SCC': 'Yes', 'SCCTolerance': '1e-5', 'MaxSCCIterations': 200, 
            'ThirdOrderFull': 'Yes',
            'HubbardDerivs': {'C': '-0.1492', 'H': '-0.1857', 
                              'N': '-0.1535', 'O': '-0.1575'},
            'MaxAngularMomentum': {'H': '"s"', 'C': '"p"', 
                                   'N': '"p"', 'O': '"p"'},
            'Filling = Fermi': {'Temperature [Kelvin]': '0.0'}, 
            'Charge': '0.0',
            'SlaterKosterFiles = Type2FileNames': {
                'Prefix': '"%s/dftb+1.2.2/slako/3ob-3-1/"' % (
                    os.path.expanduser('~')),
                'Separator': '"-"', 'Suffix': '".skf"'},
            'Dispersion = DftD3': 
                {'Damping = BeckeJohnson': {'a1': '0.746', 'a2': '4.191'}, 
                 's6': '1.0', 's8': '3.209'}}
        if 'C' in seq or 'M' in seq:
            self['Hamiltonian = DFTB']['HubbardDerivs']['S'] = '-0.11'
            self['Hamiltonian = DFTB']['MaxAngularMomentum']['S'] = '"d"'

    def str(self, x=None, indent=0):
        if x is None:
            y = ''
            for k in self.keys():
                y += k
                if isinstance(self[k], dict):
                    y += self.str(self[k], indent=indent + 1)
                else:
                    y += ' = %s\n' % self[k]
                y += '\n'
        elif isinstance(x, dict):
            y = ' {\n'
            for k in x.keys():
                y += '\t' * indent
                y += k
                if isinstance(x[k], dict):
                    y += self.str(x[k], indent=indent + 1)
                else:
                    y += ' = %s\n' % x[k]
            y += '\t' * (indent - 1)
            y += '}\n'
        else:
            y = str(x)
        return y


def findk3(k, d_1, d_2, d_3, i1, i2, i3, dl1, dl2, dl3):
    if k in d_1.keys():
        if i1 >= len(dl1):
            print(f'i1={i1} >= len(dl1)={len(dl1)}, {seq1} {seq2} {seq3}')
        return dl1[i1][d_1[k]]
    elif k in d_2.keys():
        if i2 >= len(dl2):
            print(f'i2={i2} >= len(dl2)={len(dl2)}, {seq1} {seq2} {seq3}')
        return dl2[i2][d_2[k]]
    elif k in d_3.keys():
        if i3 >= len(dl3):
            print(f'i3={i3} >= len(dl3)={len(dl3)}, {seq1} {seq2} {seq3}')
        return dl3[i3][d_3[k]]
    else:
        print(f'{k}{d_1.keys()}{d_2.keys()}{d_3.keys()} {seq1} {seq2} {seq3}')
        assert False

def joint_p1(dl, e, d):
    dl1 = deepcopy(dl)
    e1 = deepcopy(e)
    d1 = deepcopy(d)
    if d1 is None:
        if e1 is None:
            e1 = np.array([])
        elif isinstance(e1, str) and os.path.isfile(e1):
            e1 = np.loadtxt(e1, usecols=5)
        if dl1 is None:
            dl1 = []
        elif isinstance(dl1, str) and os.path.isfile(dl1):
            with open(dl1, 'r') as f:
                _dl1 = []
                for x in f.readlines():
                    _dl1.append([float(y) for y in x.strip().split()[2:]])
            dl1 = np.array(_dl1)
    else:
        e1, dl1 = [], []
        for n in sorted(d1.items(), key=lambda x: x[1]['energy']):
            e1.append(n[1]['energy'])
            dl1.append(n[1]['dihl'])
        e1 = np.array(e1)
    return dl1, e1, d1

def p2_k_list(item_array_list_longest, domain_max, domain_list):
    # used in quchongp2
    k_list = [[] for _ in domain_list]
    for n, i in enumerate(item_array_list_longest, 1):
        if n == 1:
            me = i[1]['energy']
            e = 0
        else:
            e = i[1]['energy'] - me
        if e > domain_max:
            break
        if e <= domain_list[0]:
            k_list[0].append(i[0])
        for j in range(1, len(domain_list)):
            if domain_list[j-1] < e <= domain_list[j]:
                k_list[j].append(i[0])
    return k_list

def p2_tn(k_list):
    if len(k_list[0]) <= 200:
        tn = [len(k_list[0])]
    else:
        tn = [300 - 10000//(len(k_list[0])-100)]
    if len(k_list[1]) <= 100:
        tn.append(len(k_list[1]))
    else:
        tn.append(150 - 5000//(len(k_list[1])-50))
    if len(k_list[2]) <= 100:
        tn.append(len(k_list[2]))
    else:
        tn.append(150 - 5000//(len(k_list[2])-50))
    return tn

class MyUtils:
    """
    This class contains some functions used in searching ensemble of low enengy
    structures of natural peptides.
    This class is sequence-specific. It is necessary to instantiate every time 
    once there is any new peptide.

    functions:
        __init__
        end
        quchong
        create_dict_from_file
        quchongp
        quchongp2
        write_uni_dih_file
        rosetta_relax_from_pdb_xyz_dih_calc
        rosetta_relax_from_pdb_xyz_dih_master
        rosetta_relax_from_pdb_xyz_dih
        amber_opt_from_pdb_xyz_dih_calc
        amber_opt_from_pdb_xyz_dih_master
        amber_opt_from_pdb_xyz_dih
        openmm_opt_from_xyz_dih_pdb_calc
        openmm_opt_from_xyz_dih_pdb_master
        openmm_opt_from_xyz_dih_pdb
        write_pkl
        load_pkl
        xtb_sp_from_xyz_calc
        write_input_xyz
        xtb_opt_from_xyz_dih_calc
        xtb_opt_from_xyz_dih_master
        xtb_sp_from_xyz
        xtb_opt_from_xyz_dih
        dftb_opt_from_pdb_xyz_dih_calc
        dftb_opt_from_pdb_xyz_dih
        dftb_sp_from_pdb_xyz_calc
        dftb_sp_from_pdb_xyz
        combine_dict
        subdict_energy
        bond_rotate
        bond_rotate2
        read_joint
        systematics
        joint3_2level
        joint_3level
        qc_dih
    """
    def __init__(
            self, comm, seq, common_dir='.', nt="ace", ct="nme", dftb_sol=''):
        """
        Args:
            comm, mpi4py.MPI.Comm
            seq, one-letter representation, d for depronated D, 
                $ for protonated R
            common_dir, dircetory where results will be stored
            nt, N termination, default is capped, pro for protonation, 
                neu for neutral
            ct, C termination, default is capped, dep for deprotonation, 
                neu for neutral
        """
        if comm.Get_rank() > 0:
            comm.recv(source=comm.Get_rank()-1)
        print(f'rank {comm.Get_rank()} on {platform.node()} for {seq}', 
              flush=True, file=sys.stderr)
        if comm.Get_rank() < comm.Get_size() - 1:
            comm.send('', dest=comm.Get_rank()+1)
        self.seq = seq
        self.seq_cap = seq.upper().replace('4', 'r').replace('J', 'H')
        self.comm = comm
        self.rank = comm.Get_rank()
        self.common_dir = common_dir
        if os.path.isdir('/local'):
            self.local_dir = mkdtemp(dir='/local', suffix=str(self.rank))
        else:
            self.local_dir = mkdtemp(suffix=str(self.rank))
        self.get_dihs = GetDihs(seq=self.seq, nt=nt, ct=ct)
        if ct != 'neu':
            self.get_dihs_rosetta = GetDihs(seq=self.seq, nt=nt, ct=ct, 
                                            rosetta=True)
        self.time = {'total': time.time()}
        self.max_dftb_steps = 0
        self.nt = nt
        self.ct = ct
        self.xtb_sol = ''
        self.xtb_lvl = ''
        if len(dftb_sol) == 0:
            self.dftb_sol = None
        else:
            self.dftb_sol = dftb_sol
        self.openmm_sol = None
        self.ffxml = 'amber14-all.xml'
        self.charge = 0
        if nt == "pro":
            self.charge += 1
        if ct == "dep":
            self.charge -= 1
        for s in seq:
            if s in ['d', 'e', 'y']:
                self.charge -= 1
            elif s in ['4', 'k', '6']:
                self.charge += 1
        self.dftbplus = "dftb+22.2"
        self.xtb = "xtb"
        self.rosetta_relax = 'relax.static.linuxgccrelease'
        self.openmm_tol = -1
        self.br_count = 0
        self.dih_record = {}
        i = 0
        while not os.path.isdir(self.common_dir):
            if self.rank == ROOT_RANK:
                os.makedirs(self.common_dir)
            time.sleep(1)
            i += 1

    def end(self):
        """
        Remove temporal files and print time.
        """
        shutil.rmtree(self.local_dir)
        self.time['total'] = time.time() - self.time['total']
        if self.rank == ROOT_RANK:
            for k in self.time:
                print(k, self.time[k])
            print(self.seq, self.seq_cap)
            if self.xtb_sol is not None:
                print(self.xtb_sol)

    @Timer("quchong")
    def quchong(self, input_dict, mini_angle=45, tfd_value=15, factor=0.8, 
                threshod_max=180, max_quanzhong=4, angle_initial=90, 
                angle_side_base=120, ang_gap_min_2r_side_chain=120, 
                ni=1, from_file=False):
        """
        Prothod_mquchong_tfd_initial_thresh.c is used so necessary.

        Args:
            input_dict, dictionary of input structures whose dihedrals is 
            necessary
            mini_angle, tfd_value, factor, threshod_max, max_quanzhong, 
            angle_initial, angle_side_base, ang_gap_min_2r_side_chain: 
                parameters of Prothod_mquchong_tfd_initial_thresh.c
            ni, number of iterations to use 
            Prothod_mquchong_tfd_initial_thresh.c

        Return:
            dictionary of output structures of final 
            Prothod_mquchong_tfd_initial_thresh.c

        """
        if from_file and os.path.isfile(from_file):
            return self.load_pkl(from_file)
        elif from_file:
            return {}
        if self.rank != ROOT_RANK:
            return {}
        os.chdir(self.local_dir)
        if os.path.isdir('qc'):
            shutil.rmtree('qc')
        os.mkdir('qc')
        os.chdir('qc')
        with open('unique_list.txt', 'w') as fu, \
             open('dihedral_angles_list', 'w') as fd:
            for n, i in enumerate(sorted(
                    input_dict.items(), key=lambda x: x[1]['energy']), 1):
                fu.write(f'{i[0]}.xyz {i[1]["energy"]/627.51} {n} {n}\n')
                fd.write('%s.xyz normal' % (i[0]))
                new_dih = self.qc_dih(i[1]['dihl'])
                for dih in new_dih:
                    fd.write(f' {dih:.2f}')
                fd.write('\n')
                if n == 1:
                    print([f'{x:.2f}' for x in new_dih])
        # start
        lt = time.strftime("%d%H%M%S", time.localtime())
        args = '%s %f 2 %f l %f %f %f %f %f %f' % (
            self.seq_cap, mini_angle, tfd_value, factor, threshod_max, 
            max_quanzhong, angle_initial, angle_side_base, 
            ang_gap_min_2r_side_chain)
        tfd_str = f'tfd={tfd_value:.1f}'
        print(f'start qc {self.seq} {len(input_dict)} at {lt} {tfd_str}')
        for _ in range(ni):
            os.system(
                f'Prothod_mquchong_tfd_initial_thresh.c {args} > /dev/null')
            shutil.copy("dihedral_angles_list2", "dihedral_angles_list")
            shutil.copy("dihedral_angles_list2", "unique_list.txt")
        namelist = read_namelist()
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'complete quchong {len(namelist)} at {lt}', flush=True)
        # complete
        if len(namelist) == 0:
            print(args)
            for n, i in enumerate(sorted(
                    input_dict.items(), key=lambda x: x[1]['energy']), 1):
                print(f'{i[0]}.xyz {i[1]["energy"] / 627.51} {n} {n}')
                for j in range(len(i[1]['dihl'])):
                    print(' %.2f' % i[1]['dihl'][j], end='')
                print()
        output_dict = {n[:-4]: input_dict[n[:-4]] for n in namelist}
        for pkl_name in glob.glob(f'{self.local_dir}/openmm_opt_*.pkl'):
            with open(pkl_name, 'rb') as f:
                d = pickle.load(f)
            backup_file(pkl_name)
            for k in d.keys():
                if k in output_dict.keys():
                    output_dict[k]['xyzl'] = d[k]
        return {n[:-4]: input_dict[n[:-4]] for n in namelist}

    def create_dict_from_file(self, df=None, ef=None):
        if self.rank != ROOT_RANK:
            return {}
        print(df, ef)
        output_dict = {}
        with open(df, 'r') as f:
            for line in f.readlines():
                s = line.strip().split()
                output_dict[s[0]] = {'dihl': [float(x) for x in s[2:]], 
                                     'energy': 0.}
        with open(ef, 'r') as f:
            for line in f.readlines():
                s = line.strip().split()
                if len(s) == 6:
                    output_dict[s[0]]['energy'] = float(s[-1])
                else:
                    output_dict[s[0]]['energy'] = float(s[1]) * 627.51
        return output_dict

    @Timer("quchongp")
    def quchongp(self, input_dict, factor=0.8, max_quanzhong=4, tn=300, ni=3, 
                 max_tfd=180):
        """
        Find parameters for Prothod_mquchong_tfd_initial_thresh.c to get 
        specific number of structures.
    
        Args:
            input_dict:
                dictionary of input structures whose dihedrals is necessary
            factor, max_quanzhong: 
                parameters of Prothod_mquchong_tfd_initial_thresh.c
            tn, target number of structures
            ni, number of iterations to use 
            Prothod_mquchong_tfd_initial_thresh.c
            max_tfd, maximum angle for TFD
    
        Return:
            tuple, (tfd, angle)
    
        """
        if self.rank != ROOT_RANK:
            return -1, -1
        os.chdir(self.local_dir)
        if len(input_dict) <= tn:
            return -1, -1
        with open('unique_list.txt', 'w') as fu, \
             open('dihedral_angles_list', 'w') as fd:
            for n, i in enumerate(sorted(
                    input_dict.items(), key=lambda x: x[1]['energy']), 1):
                fu.write(f'{i[0]}.xyz {i[1]["energy"]/627.51} {n} {n}\n')
                fd.write('%s.xyz normal' % (i[0]))
                new_dih = self.qc_dih(i[1]['dihl'])
                for dih in new_dih:
                    fd.write(f' {dih:.2f}')
                fd.write('\n')
        tfd_max = get_tfd_max(
            self.seq_cap, factor=factor, max_quanzhong=max_quanzhong)
        tfd, angle = [0, tfd_max * max_tfd / 180], [0, 180]
        while tfd[1] - tfd[0] > 1:
            a = sum(angle) / 2
            t = sum(tfd) / 2
            n = prothod_mquchong_tfd_initial_thresh(
                self.seq_cap, mini_angle=a, tfd_value=t, angle_initial=a, 
                factor=factor, max_quanzhong=max_quanzhong, angle_side_base=a, 
                ang_gap_min_2r_side_chain=a, ni=ni)
            if n > tn:
                tfd[0] = t
                angle[0] = a
            else:
                tfd[1] = t
                angle[1] = a
        return sum(tfd) / 2, sum(angle) / 2

    @Timer("quchongp2")
    def quchongp2(self, input_dict, factor=0.8, max_quanzhong=9, tn=None, 
                  ni=3, domain=None, max_tfd=180, max_gap=9, split=False):
        """
        Get specific number of structures using 
        Prothod_mquchong_tfd_initial_thresh.c

        Args:
            input_dict: 
                dictionary of input structures whose dihedrals is necessary
            factor, max_quanzhong: 
                parameters of Prothod_mquchong_tfd_initial_thresh.c
            domain, energy ranges to use Prothod_mquchong_tfd_initial_thresh.c 
            seperately, default is 0~5, 5~8, 8~10
            tn, target number of structures for each energy range, default are 
                3 incremental numbers less than 300, 150, 150
            ni, number of iterations to use 
            Prothod_mquchong_tfd_initial_thresh.c
            max_tfd, maximum angle for TFD
            
        Return:
            dictionary of output structures of final 
            Prothod_mquchong_tfd_initial_thresh.c

        """
        if self.rank != ROOT_RANK:
            return [{}] if split else {}
        if domain is None:
            domain_list = (5, 8, 10)
        else:
            domain_list = domain
        os.chdir(self.local_dir)
        me = 0  # minimum energy
        domain_max = max(domain_list)
        item_list = sorted(input_dict.items(), key=lambda x: x[1]['energy'])
        _energy_list = []
        for i in range(1, len(item_list)):
            _energy_list.append(
                item_list[i][1]['energy'] - item_list[i-1][1]['energy'])
        energy_gap = np.array(_energy_list)
        item_array_list = np.split(
            item_list, np.where(energy_gap > max_gap)[0] + 1)
        longest = np.argmax([len(x) for x in item_array_list])
        k_list = p2_k_list(item_array_list[longest], domain_max, domain_list)
        if split:
            olist, olist_len = [], []
            for item_array in item_array_list:
                olist.append({i[0]: i[1] for i in item_array})
                olist_len.append(len(item_array))
            print('split: ', olist_len)
            return olist
        if tn is None:
            tn = p2_tn(k_list) # target number
        else:
            tn = tn
        assert len(tn) == len(domain_list)
        odict = {}
        for i in range(len(k_list)):
            if tn[i] >= len(k_list[i]):
                # already less than target
                for k in k_list[i]:
                    odict[k] = input_dict[k]
                continue
            idict = {k: input_dict[k] for k in k_list[i]}
            t, a = self.quchongp(input_dict=idict, tn=tn[i], max_tfd=max_tfd)
            ndict = self.quchong(
                input_dict=idict, mini_angle=a, tfd_value=t, factor=factor,
                max_quanzhong=max_quanzhong, angle_initial=a, 
                angle_side_base=a, ang_gap_min_2r_side_chain=a, ni=ni)
            if len(ndict) > tn[i] * 2:
                odict = self.combine_dict(odict, {k: ndict[k] for k in sorted(
                    ndict.keys(), key=lambda x:ndict[x]['energy'])[:tn[i]*2]})
            else:
                odict = self.combine_dict(odict, ndict)
        return odict

    @Timer("quchongp3")
    def quchongp3(self, input_dict, factor=0.8, max_quanzhong=9, tn=None, 
                  ni=3, domain=None, max_tfd=180, max_gap=9):
        if self.rank != ROOT_RANK:
            return {}
        dict_list = self.quchongp2(input_dict, split=True)
        mq = max_quanzhong
        if len(dict_list) == 1:
            qc_dict = self.quchongp2(input_dict, factor=0.8, max_quanzhong=mq,
                                   max_tfd=90, domain=domain, tn=tn,)
            return qc_dict
        opt_dict = self.quchongp2(dict_list[0], factor=0.8, domain=domain,
                               max_quanzhong=mq, max_tfd=90, tn=tn)
        sp_dict = self.quchongp2(dict_list[1], factor=0.8, domain=domain,
                               max_quanzhong=mq, max_tfd=90, tn=tn)
        q2_dict = self.combine_dict(sp_dict, opt_dict)
        return q2_dict

    def write_uni_dih_file(self, idict):
        if self.rank != ROOT_RANK:
            return
        kl = [a[0] for a in sorted(
            idict.items(), key=lambda x: x[1]['energy'])]
        if len(kl) == 0:
            return
        uni_name = '%s/uni_diha_list.txt' % self.common_dir
        dih_name = '%s/dihedral_angles_list2' % self.common_dir
        if os.path.isfile(uni_name):
            backup_file(uni_name)
        if os.path.isfile(dih_name):
            backup_file(dih_name)
        with open(uni_name, 'w') as fu, open(dih_name, 'w') as fd:
            for i, k in enumerate(kl, 1):
                fu.write('%s.xyz %f %d %d %d %f\n' % (
                    k, idict[k]['energy']/627.51, i, i, i, 
                    (idict[k]['energy']-idict[kl[0]]['energy'])))
                fd.write("%s.xyz normal" % k)
                if 'dihl' in idict[k].keys():
                    for x in idict[k]['dihl']:
                        fd.write(" %.2f" % x)
                fd.write('\n')

    def rosetta_relax_from_pdb_xyz_dih_calc(self, input_dict):
        t0 = time.time()
        energy, xyzl, pdb, dih = 0, [], None, []
        if os.path.isdir(f'{self.local_dir}/rosetta'):
            shutil.rmtree(f'{self.local_dir}/rosetta')
        os.mkdir(f'{self.local_dir}/rosetta')
        shutil.copy(f'{self.local_dir}/template', f'{self.local_dir}/rosetta')
        os.chdir(f'{self.local_dir}/rosetta')
        # naming conventions
        if 'pdb' in input_dict.keys():
            input_dict['pdb'].write_rosetta('rosetta_input.pdb')
        elif 'xyzl' not in input_dict.keys():
            write_txt({'all_atom': {'dihl': input_dict['dihl']}}, 
                      seq=self.seq)
            zmat2xyz.txt2xyz()
            os.remove('all_atom.txt')
            try:
                cut_conh('all_atom.xyz', verbose=False)
                raw_pdb = ReadPdbAtom('all_atom.pdb')
            except:
                return 1, energy, xyzl, dih, time.time() - t0
            if self.nt == 'pro':
                raw_pdb.nt_cap2pro()
            elif self.nt == 'neu':
                raw_pdb.nt_cap2neu()
            if self.ct == 'nh2':
                raw_pdb.ct_cap2nh2()
            elif self.ct == 'dep':
                raw_pdb.ct_cap2dep()
            elif self.ct == 'neu':
                raw_pdb.ct_cap2neu()
            raw_pdb.write_rosetta('rosetta_input.pdb')
        else:
            xyzl = input_dict['xyzl']
            with open('all_atom.xyz', 'w') as f:
                for line in input_dict['xyzl']:
                    f.write(line)
            try:
                cut_conh('all_atom.xyz', verbose=False)
                raw_pdb = ReadPdbAtom('all_atom.pdb')
            except:
                return 1, energy, xyzl, dih, time.time() - t0
            raw_pdb.write_rosetta('rosetta_input.pdb')
        with open('general_relax_flags', 'w') as f:
            f.write('-nstruct 1\n-relax:default_repeats 5\n-out:path:pdb .\
                    \n-out:path:score .')
        os.system(f'{self.rosetta_relax} -s rosetta_input.pdb \
                  -out:suffix _relaxed @general_relax_flags 1>stdout 2>stderr')
        with open('score_relaxed.sc', 'r') as f:
            lines = f.readlines()
            energy = float(lines[2].strip().split()[1])
        pdb = ReadPdbAtom('rosetta_input_relaxed_0001.pdb')
        dih = self.get_dihs_rosetta.calc_pdb(pdb)
        return 0, energy, xyzl, dih, time.time() - t0

    def rosetta_relax_from_pdb_xyz_dih_master(self, xyz_dict):
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'start rosetta relax {len(xyz_dict)} at {lt}', flush=True)
        key_list = [k for k in xyz_dict.keys()]
        results = calc_master(self.comm, [xyz_dict[k] for k in key_list])
        output = {}
        for i in range(len(results)):
            if results[i][0] != 0:
                continue
            output[key_list[i]] = {'energy': results[i][1], 
                                   'xyzl': results[i][2], 
                                   'dihl': results[i][3], 
                                   'time': results[i][-1]}
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'complete rosetta relax {len(output)} at {lt}', flush=True)
        return output

    @Timer("rosetta_relax")
    def rosetta_relax_from_pdb_xyz_dih(self, xyz_dict,):
        if self.rank != ROOT_RANK:
            os.chdir(self.local_dir)
            template_generate.wrt(self.seq, mz=True, mp=True)
            calc_slave(self.comm, self.rosetta_relax_from_pdb_xyz_dih_calc)
            return {}
        return self.rosetta_relax_from_pdb_xyz_dih_master(xyz_dict,)

    def amber_opt_from_pdb_xyz_dih_calc(self, input_dict, opt=True):
        t0 = time.time()
        status, energy, xyzl = 0, 0, []
        if os.path.isdir(f'{self.local_dir}/amber'):
            shutil.rmtree(f'{self.local_dir}/amber')
        os.mkdir(f'{self.local_dir}/amber')
        os.chdir(f'{self.local_dir}/amber')
        # write input pdb file
        if 'pdbl' in input_dict.keys() and False:
            with open('raw.pdb', 'w') as f:
                for line in input_dict['pdbl']:
                    f.write(line)
        elif 'xyzl' not in input_dict.keys():
            write_txt({'openmm_input': {'dihl': input_dict['dihl']}}, 
                      seq=self.seq)
            zmat2xyz.txt2xyz()
            os.remove('openmm_input.txt')
            xyz2pdb(self.seq, 'openmm_input.xyz', 'openmm_input.pdb')
            pdb = ReadPdbAtom('openmm_input.pdb')
            if os.path.exists('openmm_input.xyz'):
                os.remove('openmm_input.xyz')
            if self.nt == 'pro':
                pdb.nt_cap2pro()
            elif self.nt == 'neu':
                pdb.nt_cap2neu()
            if self.ct == 'nh2':
                pdb.ct_cap2nh2()
            elif self.ct == 'dep':
                pdb.ct_cap2dep()
            elif self.ct == 'neu':
                pdb.ct_cap2neu()
            pdb.write('raw.pdb')
        else:
            with open('raw.xyz', 'w') as f:
                for line in input_dict['xyzl']:
                    f.write(line)
            try:
                cut_conh('raw.xyz', verbose=False)
            except:
                return 1, energy, xyzl, [], time.time() - t0
        if not os.path.exists('raw.pdb'):
            return 1, energy, xyzl, [], time.time() - t0
        raw_pdb = ReadPdbAtom('raw.pdb')
        raw_pdb.write_amber('input.pdb')
        with open('tleap.in', 'w') as f:
            f.write("source leaprc.protein.ff14SBonlysc\n")
            f.write("tc5b = loadpdb input.pdb\nset default pbradii mbondi3\n")
            f.write("saveamberparm tc5b input.parm7 input.rst7\n")
        amberbin = "~/miniforge3/envs/AmberTools23/bin"
        os.system(f"{amberbin}/tleap -f tleap.in > tleap.out 2>/dev/null")
        if not os.path.exists('input.rst7'):
            with open('tleap.out', 'r') as f:
                xyzl = f.readlines()
            with open('input.pdb', 'r') as f:
                lines = f.readlines()
            with open('raw.pdb', 'r') as f:
                energy = f.readlines()
            return 2, energy, xyzl, lines, time.time() - t0
        """
        # input for amber sander
        with open('min.in', 'w') as f:
            f.write("energy minimization\n &cntrl\n  imin = 1, \n")
            f.write("  maxcyc=1000,  \n  ntx = 1, \n  ntwr = 10000, \n")
            f.write("  ntpr = 1000, \n  ioutfm=0, \n  ntxo=1, \n")
            f.write("  cut = 1000.0, \n  ntb=0, \n  igb = 8, \n  gbsa=1, \n")
            f.write("  surften=0.007, \n  saltcon = 0.0, \n &end\n")
        s = "-c input.rst7 -x min.crd -inf min.info -r min.rst7 -o min.out"
        os.system(f"{amberbin}/sander -O -i min.in -p input.parm7 {s}")
        if not os.path.exists('min.info') or not os.path.exists('min.rst7'):
            return 3, energy, xyzl, [], time.time() - t0
        with open('min.info', 'r') as f:
            lines = f.readlines()
        try:
            energy = float(lines[3].strip().split()[1])
        except:
            return 4, energy, xyzl, lines, time.time() - t0
        anames = []
        with open('input.parm7', 'r') as f:
            flag = False
            for line in f.readlines():
                if line.startswith('%FLAG ATOM_NAME'):
                    flag = True
                    continue
                if line.startswith('%FLAG CHARGE'):
                    break
                if line.startswith('%'):
                    continue
                if flag:
                    anames += textwrap.wrap(line.strip(), 4)
        with open('min.rst7', 'r') as f:
            lines = f.readlines()
        xyzs = []
        for line in lines[2:]:
            s = line.strip().split()
            xyzs.append([float(s[0]), float(s[1]), float(s[2])])
            if len(s) == 6:
                xyzs.append([float(s[3]), float(s[4]), float(s[5])])
        with open('output.xyz', 'w') as f:
            xyzl.append(f'{len(anames)}\n')
            f.write(xyzl[-1])
            xyzl.append('\n')
            f.write(xyzl[-1])
            for aname, xyz in zip(anames, xyzs):
                xyzl.append(f'{aname.strip()[0]} {xyz[0]} {xyz[1]} {xyz[2]}\n')
                f.write(xyzl[-1])
        try:
            cut_conh('output.xyz', verbose=False)
        except:
            return 5, energy, xyzl, [], time.time() - t0
        pdb = ReadPdbAtom('output.pdb')
        dih = self.get_dihs.calc_pdb(pdb)
        return 0, energy, xyzl, dih, time.time() - t0
        """
        flag = openmm_opt_amber(opt=opt)
        if flag < 0:
            shutil.copy('input.rst7', 
                        '%s/%d.rst7' % (self.common_dir, self.rank))
            return 3, energy, xyzl, time.time() - t0
        if not os.path.exists('output.xyz'):
            return 5, energy, xyzl, time.time() - t0
        with open('output.xyz', 'r') as f:
            xyzl = f.readlines()
        try:
            e = float(xyzl[1].strip().split()[-1])
        except IndexError:
            e = 9.9
        with open('output.pdb', 'r') as f:
            pdbl = f.readlines()
        pdb = ReadPdbAtom('output.pdb')
        dih = self.get_dihs.calc_pdb(pdb)
        return status, e, xyzl, dih, pdbl, flag, time.time() - t0

    def amber_opt_from_pdb_xyz_dih_master(self, xyz_dict, opt_sp='opt'):
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'start amber {opt_sp} {len(xyz_dict)} at {lt}', flush=True)
        key_list = [k for k in xyz_dict.keys()]
        results = calc_master(self.comm, [xyz_dict[k] for k in key_list])
        output = {}
        err_dict = {}
        flag2, flag4, flag5 = True, True, True
        time_count = 0
        for i in range(len(results)):
            if results[i][0] != 0:
                if results[i][0] in err_dict.keys():
                    err_dict[results[i][0]] += 1
                else:
                    err_dict[results[i][0]] = 1
                if results[i][0] == 2 and flag2:
                    for line in results[i][2]:
                        print(line.strip(), file=sys.stderr)
                    print(file=sys.stderr)
                    for line in results[i][3]:
                        print(line.strip(), file=sys.stderr)
                    print(file=sys.stderr)
                    for line in results[i][1]:
                        print(line.strip(), file=sys.stderr)
                    print(file=sys.stderr)
                    for line in xyz_dict[key_list[i]]['xyzl']:
                        print(line.strip(), file=sys.stderr)
                    flag2 = False
                if results[i][0] == 4 and flag4:
                    for line in results[i][3]:
                        print(line.strip(), file=sys.stderr)
                    flag4 = False
                if results[i][0] == 5 and flag5:
                    for line in results[i][2]:
                        print(line.strip(), file=sys.stderr)
                    flag5 = False
                continue
            time_count += results[i][5]
            output[key_list[i]] = {'energy': results[i][1], 
                                   'xyzl': results[i][2], 
                                   'dihl': results[i][3], 
                                   'time': results[i][-1]}
        for k in err_dict.keys():
            print(k, err_dict[k])
        print(time_count)
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'complete amber {len(output)} at {lt}', flush=True)
        return output

    @Timer("amber")
    def amber_opt_from_pdb_xyz_dih(self, xyz_dict, opt=True):
        if self.rank != ROOT_RANK:
            os.chdir(self.local_dir)
            template_generate.wrt(self.seq, mz=True, mp=True)
            calc_slave(self.comm, self.amber_opt_from_pdb_xyz_dih_calc, opt=opt)
            return {}
        opt_sp = 'opt' if opt else 'sp'
        return self.amber_opt_from_pdb_xyz_dih_master(xyz_dict, opt_sp=opt_sp)

    def openmm_opt_from_xyz_dih_pdb_calc(self, input_dict, noSol=False, tol=None):
        t0 = time.time()
        status, e, xyzl, pdbl = 0, 0, [], []
        os.chdir(self.local_dir)
        # write input pdb file
        if 'pdb' in input_dict.keys():
            input_dict['pdb'].write('openmm_input.pdb')
        elif 'pdbl' in input_dict.keys():
            with open('openmm_input.pdb', 'w') as f:
                for line in input_dict['pdbl']:
                    f.write(line)
        elif 'xyzl' not in input_dict.keys():
            write_txt({'openmm_input': {'dihl': input_dict['dihl']}}, 
                      seq=self.seq)
            zmat2xyz.txt2xyz()
            os.remove('openmm_input.txt')
            xyz2pdb(self.seq, 'openmm_input.xyz', 'openmm_input.pdb')
        else:
            with open('openmm_input.xyz', 'w') as f:
                for line in input_dict['xyzl']:
                    f.write(line)
            xyz2pdb(self.seq, 'openmm_input.xyz', 'openmm_input.pdb')
        # modify terminal
        pdb = ReadPdbAtom('openmm_input.pdb')
        if os.path.exists('openmm_input.xyz'):
            os.remove('openmm_input.xyz')
        if self.nt == 'pro':
            pdb.nt_cap2pro()
        elif self.nt == 'neu':
            pdb.nt_cap2neu()
        if self.ct == 'nh2':
            pdb.ct_cap2nh2()
        elif self.ct == 'dep':
            pdb.ct_cap2dep()
        elif self.ct == 'neu':
            pdb.ct_cap2neu()
        pdb.write('openmm_input.pdb')
        # opt
        if noSol:
            flag = openmm_opt('openmm_input.pdb', oname='openmm_output.xyz', 
                              tol=1000, ffxml=self.ffxml)
        elif self.openmm_tol > 0:
            flag = openmm_opt('openmm_input.pdb', oname='openmm_output.xyz', 
                              tol=self.openmm_tol, solvent=self.openmm_sol, 
                              ffxml=self.ffxml)
        else:
            flag = openmm_opt('openmm_input.pdb', oname='openmm_output.xyz',
                              solvent=self.openmm_sol, ffxml=self.ffxml)
        if flag < 0:
            shutil.copy('openmm_input.pdb', 
                        '%s/%d.pdb' % (self.common_dir, self.rank))
            return 1, e, xyzl, time.time() - t0
        with open('openmm_output.xyz', 'r') as f:
            xyzl = f.readlines()
        try:
            e = float(xyzl[1].strip().split()[-1])
        except IndexError:
            e = 9.9
        with open('openmm_output.pdb', 'r') as f:
            pdbl = f.readlines()
        pdb = ReadPdbAtom('openmm_output.pdb')
        dih = self.get_dihs.calc_pdb(pdb)
        os.remove('openmm_input.pdb')
        os.remove('openmm_output.xyz')
        os.remove('openmm_output.pdb')
        return status, e, xyzl, dih, pdbl, time.time() - t0

    def openmm_opt_from_xyz_dih_pdb_master(self, xyz_dict):
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'start openmm opt {len(xyz_dict)} at {lt}', flush=True)
        key_list = [k for k in xyz_dict.keys()]
        output = {}
        output_xyz = {}
        Npkl = 0
        err_dict = {}
        for result in calc_master_generator(self.comm, 
                                            [xyz_dict[k] for k in key_list]):
            if result[1][0] != 0:
                if result[1][0] in err_dict.keys():
                    err_dict[result[1][0]] += 1
                else:
                    err_dict[result[1][0]] = 1
                continue
            output[key_list[result[0]]] = {'energy': result[1][1], 
                                           'xyzl': result[1][2], 
                                           'dihl': result[1][3], 
                                           'pdbl': result[1][4], 
                                           'time': result[1][-1]}
            output_xyz[key_list[result[0]]] = result[1][2]
            if len(output_xyz) >= 5e5:
                self.write_pkl(
                    output_xyz, 
                    pkl_name=f'{self.local_dir}/openmm_opt_{Npkl}.pkl')
                Npkl += 1
                output_xyz = {}
        for k in err_dict.keys():
            print(k, err_dict[k])
        if len(output_xyz) > 0:
            self.write_pkl(output_xyz, 
                           pkl_name=f'{self.local_dir}/openmm_opt_{Npkl}.pkl')
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'complete openmm opt {len(output)} at {lt}', flush=True)
        return output

    @Timer("openmm_opt")
    def openmm_opt_from_xyz_dih_pdb(self, dih_dict, from_file=False, ns=False):
        if from_file:
            return {}
        if self.rank != ROOT_RANK:
            os.chdir(self.local_dir)
            template_generate.wrt(self.seq, mz=True, mp=True)
            calc_slave(self.comm, self.openmm_opt_from_xyz_dih_pdb_calc, ns)
            return {}
        return self.openmm_opt_from_xyz_dih_pdb_master(dih_dict)

    def print(self, *args, **kwargs):
        if self.rank == ROOT_RANK:
            print(*args, **kwargs)

    def write_pkl(self, wrt_data, pkl_name=None, prefix=''):
        if self.rank != ROOT_RANK:
            return
        if pkl_name is None:
            pkl_name = '%s/%s%s.pkl' % (self.common_dir, prefix, 
                time.strftime("%d%H%M", time.localtime()))
        if os.path.isfile(pkl_name):
            backup_file(pkl_name)
        try:
            with open(pkl_name, 'wb') as f:
                pickle.dump(wrt_data, f)
        except OSError as e:
            print(e)

    def load_pkl(self, pkl_name, key=None):
        if self.rank != ROOT_RANK:
            return {}
        with open(pkl_name, 'rb') as f:
            if key is None:
                d = pickle.load(f)
                new_key = next(iter(d.keys()))
                new_key2 = next(iter(d[new_key].keys()))
                if isinstance(d[new_key][new_key2], dict):
                    return d[new_key]
                return d
            return pickle.load(f)[key]

    def xtb_sp_from_xyz_calc(self, input_dict):
        t0 = time.time()
        e, xyzl = 0, []
        if os.path.isdir('%s/xtbsp' % self.local_dir):
            shutil.rmtree('%s/xtbsp' % self.local_dir)
        os.mkdir('%s/xtbsp' % self.local_dir)
        os.chdir('%s/xtbsp' % self.local_dir)
        xyzl = deepcopy(input_dict['xyzl'])
        with open('xtb_input.xyz', 'w') as f:
            f.write(f'{len(xyzl)-2}\n' + xyzl[1])
            for line in xyzl[2:]:
                f.write(line)
        try:
            if cut_conh('xtb_input.xyz'):
                return 2, e, xyzl, [], time.time() - t0
        except:
            return 3, e, xyzl, [], time.time() - t0
        with open('temp.sh', 'w') as f:
            f.write(f'OMP_NUM_THREADS=1,1 {self.xtb} xtb_input.xyz -c \
                    {self.charge} --dipole -P 1 \
                    {self.xtb_sol} 1>stdout.txt 2>stderr.txt\n')
        os.system('sh temp.sh 1>&2')
        with open('stdout.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('         :: -> Gsolv'):
                    se = float(line.strip().split()[3])*62.751
                if 'TOTAL ENERGY' in line:
                    e = float(line.strip().split()[3])*627.51
                    break
        flag = True
        with open('stderr.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('normal termination of xtb'):
                    flag = False
        if flag:
            return 1, e, xyzl, [], time.time() - t0
        else:
            return 0, e, xyzl, input_dict['dihl'], time.time() - t0

    def write_input_xyz(self, input_dict, name='xtb_input'):
        if 'xyzl' not in input_dict.keys():
            template_generate.wrt(self.seq, mz=True, mp=True)
            write_txt({name: {'dihl': input_dict['dihl']}}, 
                      seq=self.seq)
            zmat2xyz.txt2xyz()
            os.remove(f'{name}.txt')
            xyz2pdb(self.seq, f'{name}.xyz', f'{name}.pdb')
            pdb = ReadPdbAtom(f'{name}.pdb')
            if self.nt == 'pro':
                pdb.nt_cap2pro()
            elif self.nt == 'neu':
                pdb.nt_cap2neu()
            if self.ct == 'nh2':
                pdb.ct_cap2nh2()
            elif self.ct == 'dep':
                pdb.ct_cap2dep()
            elif self.ct == 'neu':
                pdb.ct_cap2neu()
            pdb.write_xyz(f'{name}.xyz')
            with open(f"{name}.xyz", 'r') as f:
                xyzl = f.readlines()
        else:
            xyzl = deepcopy(input_dict['xyzl'])
            with open(f'{name}.xyz', 'w') as f:
                f.write(f'{len(xyzl)-2}\n' + xyzl[1])
                for line in xyzl[2:]:
                    f.write(line)
        return xyzl

    def xtb_opt_from_xyz_dih_calc(self, input_dict):
        t0 = time.time()
        e, xyzl = 0, []
        if os.path.isdir('%s/xtbopt' % self.local_dir):
            shutil.rmtree('%s/xtbopt' % self.local_dir)
        os.mkdir('%s/xtbopt' % self.local_dir)
        os.chdir('%s/xtbopt' % self.local_dir)
        # write input
        self.write_input_xyz(input_dict)
        with open('temp.sh', 'w') as f:
            f.write(f'OMP_NUM_THREADS=1,1 {self.xtb} xtb_input.xyz -c \
                    {self.charge} {self.xtb_lvl} --opt --dipole -P 1 \
                        {self.xtb_sol} 1>stdout.txt 2>stderr.txt\n')
        os.system('sh temp.sh 1>&2')
        flag = True
        with open('stderr.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('normal termination of xtb'):
                    flag = False
        if flag:
            return 1, e, xyzl, [], time.time() - t0
        elif not os.path.isfile('xtbopt.xyz'):
            return 2, e, xyzl, [], time.time() - t0
        with open('xtbopt.xyz', 'r') as f:
            xyzl = f.readlines()
        try:
            if cut_conh('xtbopt.xyz'):
                return 3, e, xyzl, [], time.time() - t0
        except:
            return 4, e, xyzl, [], time.time() - t0
        pdb = ReadPdbAtom('xtbopt.pdb')
        # check N ends
        if pdb.seq3[0] == "ACE":
            nt = "ace"
        elif "H3" in pdb[0].keys():
            nt = "pro"
        else:
            nt = "neu"
        # check C ends
        if pdb.seq3[-1] == "NME":
            ct = "nme"
        elif pdb.seq3[-1] == "NHE":
            ct = "nh2"
        elif "HXT" in pdb[-1].keys():
            ct = "neu"
        else:
            ct = "dep"
        if ''.join(pdb.seq) != self.seq or nt != self.nt or ct != self.ct:
            return 5, e, xyzl, [], time.time() - t0
        dihs = self.get_dihs.calc_pdb(pdb)
        with open('stdout.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('         :: -> Gsolv'):
                se = float(line.strip().split()[3])*62.751
        os.chdir('..')
        e = float(xyzl[1].strip().split()[1])*627.51
        return 0, e, xyzl, dihs, time.time() - t0

    def xtb_opt_from_xyz_dih_master(self, xyz_dict, max_energy=-1, 
                                    name='xtb opt'):
        if max_energy < 0:
            key_list = [k for k in xyz_dict.keys()]
        elif len(xyz_dict) > 0:
            min_energy = min([x['energy'] for x in xyz_dict.values()])
            key_list = []
            for k in xyz_dict.keys():
                if xyz_dict[k]['energy'] - min_energy < max_energy:
                    key_list.append(k)
        else:
            return {}
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'start {name} {len(key_list)} at {lt}', flush=True)
        results = calc_master(self.comm, [xyz_dict[k] for k in key_list])
        output = {}
        err_dict = {}
        for i in range(len(results)):
            if results[i][0] != 0:
                if results[i][0] in err_dict.keys():
                    err_dict[results[i][0]] += 1
                else:
                    err_dict[results[i][0]] = 1
                continue
            output[key_list[i]] = {'energy': results[i][1], 
                                   'xyzl': results[i][2], 
                                   'dihl': results[i][3], 
                                   'time': results[i][-1]}
        for k in err_dict.keys():
            print(k, err_dict[k])
        lt = time.strftime("%d%H%M%S", time.localtime())
        print(f'complete {name} {len(output)} at {lt}', flush=True)
        return output

    @Timer("xtb_sp")
    def xtb_sp_from_xyz(self, xyz_dict, max_energy=-1):
        if self.rank != ROOT_RANK:
            calc_slave(self.comm, self.xtb_sp_from_xyz_calc)
            return {}
        return self.xtb_opt_from_xyz_dih_master(xyz_dict, 
                                                max_energy=max_energy,
                                                name='xtb sp')

    @Timer("xtb_opt")
    def xtb_opt_from_xyz_dih(self, xyz_dict, max_energy=-1):
        if self.rank != ROOT_RANK:
            calc_slave(self.comm, self.xtb_opt_from_xyz_dih_calc)
            return {}
        return self.xtb_opt_from_xyz_dih_master(xyz_dict, 
                                                max_energy=max_energy)
    
    def dftb_opt_from_pdb_xyz_dih_calc(self, input_dict):
        t0 = time.time()
        e, xyzl = 0, []
        if os.path.isdir('%s/dftbopt' % self.local_dir):
            shutil.rmtree('%s/dftbopt' % self.local_dir)
        os.mkdir('%s/dftbopt' % self.local_dir)
        os.chdir('%s/dftbopt' % self.local_dir)
        hsd = DFTBhsd(seq=self.seq)
        if self.max_dftb_steps:
            hsd['Driver = ConjugateGradient']['MaxSteps'] = self.max_dftb_steps
        if self.charge:
            hsd['Hamiltonian = DFTB']['Charge'] = "%d.0" % self.charge
        if self.dftb_sol is not None:
            hsd['Hamiltonian = DFTB']['Solvation'] = 'GeneralizedBorn \
                {ParamFile = "%s"}' % self.dftb_sol
        if 'pdb' in input_dict.keys():
            with open("dftb_input.xyz", 'r') as f:
                xyzl = f.readlines()
        elif 'xyzl' not in input_dict.keys():
            template_generate.wrt(self.seq, mz=True, mp=True)
            write_txt({'dftb_input': {'dihl': input_dict['dihl']}}, seq=self.seq)
            zmat2xyz.txt2xyz()
            os.remove('dftb_input.txt')
            with open("dftb_input.xyz", 'r') as f:
                xyzl = f.readlines()
        else:
            xyzl = deepcopy(input_dict['xyzl'])
        flag = True
        for n in range(4):
            os.makedirs('opt_%d' % n)
            os.chdir('opt_%d' % n)
            with open('INPUT.xyz', 'w') as f:
                f.write(xyzl[0] + xyzl[1])
                for line in xyzl[2:]:
                    f.write(line)
            with open('dftb_in.hsd', 'w') as f:
                f.write(hsd.str())
            xyz_dftbin('INPUT.xyz')
            with open('temp.sh', 'w') as f:
                f.write(f'OMP_NUM_THREADS=1 \
                        {self.dftbplus} 1>stdout.txt 2>stderr.txt\n')
            os.system('sh temp.sh 1>&2')
            if not os.path.isfile('detailed.out'):
                return 1, e, [], [], time.time() - t0
            flag_scc, flag_geo = False, False
            with open('detailed.out', 'r') as f:
                # energy
                while True:
                    line = f.readline()
                    if len(line) == 0:
                        break
                    if line.startswith('Total energy:'):
                        x = line.strip().split()
                        e = float(x[2]) * 627.51
                        break
                # converge
                while True:
                    line = f.readline()
                    if len(line) == 0:
                        break
                    if line.startswith('SCC converged'):
                        flag_scc = True
                    if line.startswith('Geometry converged'):
                        flag_geo = True
            if not os.path.isfile('opt.xyz'):
                return 2, e, xyzl, [], time.time() - t0
            with open('opt.xyz', 'r') as f:
                xyzl = f.readlines()
            if not flag_scc:
                return 3, e, xyzl, [], time.time() - t0
            elif not flag_geo and self.max_dftb_steps == 0:
                os.chdir('..')
            else:
                flag = False
                break
        if flag:
            return 4, e, xyzl, [], time.time() - t0
        try:
            if cut_conh('opt.xyz'):
                return 5, e, xyzl, [], time.time() - t0
        except:
            return 6, e, xyzl, [], time.time() - t0
        pdb = ReadPdbAtom('opt.pdb')
        # check N ends
        if pdb.seq3[0] == "ACE":
            nt = "ace"
        elif "H3" in pdb[0].keys():
            nt = "pro"
        else:
            nt = "neu"
        # check C ends
        if pdb.seq3[-1] == "NME":
            ct = "nme"
        elif pdb.seq3[-1] == "NHE":
            ct = "nh2"
        elif "HXT" in pdb[-1].keys():
            ct = "neu"
        else:
            ct = "dep"
        if ''.join(pdb.seq) != self.seq or nt != self.nt or ct != self.ct:
            return 7, e, xyzl, [], time.time() - t0
        dihs = self.get_dihs.calc_pdb(pdb)
        return 0, e, xyzl, dihs, time.time() - t0

    @Timer("dftb_opt")
    def dftb_opt_from_pdb_xyz_dih(self, input_dict, max_energy=-1, 
                                  max_dftb_steps=0):
        self.max_dftb_steps = max_dftb_steps
        if self.rank != ROOT_RANK:
            calc_slave(self.comm, self.dftb_opt_from_pdb_xyz_dih_calc)
            return {}
        return self.xtb_opt_from_xyz_dih_master(input_dict, name='dftb opt', 
                                                max_energy=max_energy)

    def dftb_sp_from_pdb_xyz_calc(self, input_dict):
        t0 = time.time()
        e, xyzl = 0, []
        if os.path.isdir('%s/dftbsp' % self.local_dir):
            shutil.rmtree('%s/dftbsp' % self.local_dir)
        os.mkdir('%s/dftbsp' % self.local_dir)
        os.chdir('%s/dftbsp' % self.local_dir)
        hsd = DFTBhsd(seq=self.seq)
        k2 = None
        for k in hsd.keys():
            if 'Driver' in k:
                k2 = k
        if k2 is not None:
            del hsd[k2]
        # del hsd['Driver = ConjugateGradient']
        if self.charge:
            hsd['Hamiltonian = DFTB']['Charge'] = "%d.0" % self.charge
        if self.dftb_sol is not None:
            hsd['Hamiltonian = DFTB']['Solvation'] =  'GeneralizedBorn \
                {ParamFile = "%s"}' % self.dftb_sol
        if 'pdb' in input_dict.keys():
            input_dict['pdb'].write_xyz('INPUT.xyz')
        else:
            with open('INPUT.xyz', 'w') as f:
                for line in input_dict['xyzl']:
                    f.write(line)
        with open("INPUT.xyz", 'r') as f:
            xyzl = f.readlines()
        with open('dftb_in.hsd', 'w') as f:
            f.write(hsd.str())
        xyz_dftbin('INPUT.xyz')
        with open('temp.sh', 'w') as f:
            f.write(f'OMP_NUM_THREADS=1 \
                    {self.dftbplus} 1>stdout.txt 2>stderr.txt\n')
        os.system('sh temp.sh 1>&2')
        if not os.path.isfile('detailed.out'):
            return 1, e, [], time.time() - t0
        flag_scc = False
        with open('detailed.out', 'r') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                if line.startswith('Total energy:'):
                    x = line.strip().split()
                    try:
                        e = float(x[2]) * 627.51
                    except ValueError:
                        return 3, e, [], time.time() - t0
                    break
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                if line.startswith('SCC converged'):
                    flag_scc = True
        flag_warn1 = False  # atom too close
        with open('stdout.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('-> Atoms'):
                    flag_warn1 = True
                    break
        if not flag_scc:
            return 2, e, xyzl, time.time() - t0
        if flag_warn1:
            if os.path.isfile('../../%d.out' % self.rank):
                backup_file('../../%d.out' % self.rank)
            shutil.copy('stdout.txt', '../../%d.out' % self.rank)
            return 4, e, xyzl, time.time() - t0
        else:
            return 0, e, xyzl, input_dict['dihl'], time.time() - t0

    @Timer("dftb_sp")
    def dftb_sp_from_pdb_xyz(self, xyz_dict, get_dih=False):
        if self.rank != ROOT_RANK:
            calc_slave(self.comm, self.dftb_sp_from_pdb_xyz_calc)
            return {}
        return self.xtb_opt_from_xyz_dih_master(xyz_dict, name='dftb sp', )

    def combine_dict(self, dict1, dict2):
        if self.rank != ROOT_RANK:
            return
        output_dict = {k: dict1[k] for k in dict1.keys()}
        for k in dict2.keys():
            output_dict[k] = dict2[k]
        return output_dict

    def subdict_energy(self, dict1, dict2):
        """substitute energy in dict1 with those in dict2"""
        if self.rank != ROOT_RANK:
            return
        output_dict = {}
        for k in dict2.keys():
            output_dict[k] = {}
            for kk in dict1[k].keys():
                if kk == 'energy':
                    continue
                output_dict[k][kk] = dict1[k][kk]
        for k in dict2.keys():
            output_dict[k]['energy'] = dict2[k]['energy']
        return output_dict

    def bond_rotate(self, input_dict, dihi, dihn=-1, dih=90):
        """
        dihi, int
        dihn, number of structures to be rotated
        dih, interval
        """
        if self.rank != ROOT_RANK:
            return
        ratio = math.ceil(360 / dih) - 1
        dih_num = int(dihn) * ratio
        new_dict = {}
        n = 0
        for i in sorted(input_dict.items(), key=lambda x: x[1]['energy']):
            if dih_num > 0 and n >= dih_num:
                break
            for j in np.arange(dih, 360, dih):
                if n == dih_num:
                    break
                k = 'r%d_%d%s' % (dihi, j, i[0])
                new_dict[k] = {}
                new_dict[k]['dihl'] = deepcopy(i[1]['dihl'])
                new_dict[k]['dihl'][dihi] += j
                new_dict[k]['dihl'][dihi] += 180
                new_dict[k]['dihl'][dihi] %= 360
                new_dict[k]['dihl'][dihi] -= 180
                n += 1
        return new_dict

    def bond_rotate2(self, input_dict, dihi_list, dihn=-1, dih=90):
        """
        dihn, number of structures to be rotated, minus means all
        dih, interval
        """
        if self.rank != ROOT_RANK:
            return
        ratio = math.ceil(360 / dih) - 1
        dih_num = int(dihn) * ratio
        new_dict = {}
        n, n2 = 0, 0
        for i in sorted(input_dict.items(), key=lambda x: x[1]['energy']):
            # for structures
            if dih_num > 0 and n >= dih_num:
                break
            if i[0].count('r') < self.br_count:
                continue
            for j in np.arange(dih, 360, dih):
                # for angle
                if dih_num > 0 and n >= dih_num:
                    break
                for dihi in dihi_list:
                    # for dihedral angle index
                    if dih_num > 0 and n >= dih_num:
                        break
                    k = 'r%d_%d%s' % (dihi, j, i[0])
                    new_dict[k] = {}
                    new_dict[k]['dihl'] = deepcopy(i[1]['dihl'])
                    new_dict[k]['dihl'][dihi] += j
                    new_dict[k]['dihl'][dihi] += 180
                    new_dict[k]['dihl'][dihi] %= 360
                    new_dict[k]['dihl'][dihi] -= 180
                    self.dih_record[k] = new_dict[k]['dihl']
                    n += 1
            if i[0] in self.dih_record.keys():
                n2 += 1
        self.br_count += 1
        print(f'already {len(self.dih_record)} recorded {n2}')
        return new_dict

    def read_joint(self, dih_file):
        if self.rank != ROOT_RANK:
            return
        with open(dih_file, 'r') as f:
            ls = f.readlines()
        lendih = 10 ** (len(str(len(ls))))
        output_dict = {}
        for i in range(len(ls)):
            output_dict['x%s' % (str(lendih + i + 1)[1:])] = {}
            output_dict['x%s' % (str(lendih + i + 1)[1:])]['dihl'] = [
                float(x) for x in ls[i].strip().split()]
        return output_dict

    def systematics(self, ):
        if self.rank != ROOT_RANK:
            n = self.comm.bcast(0, root=ROOT_RANK)
            return {i: i for i in range(n)}
        dihs_list = [[]]
        for dih in self.get_dihs.dihs_list:
            atom_list = self.get_dihs.dihs_dict[dih]
            resi = (int(dih[1:])-1)//2 if dih[0] == 'M' else int(dih[1:-1])-1
            name = self.seq[resi]
            # omega for final residue
            if dih == 'M0':
                if self.ct == 'neu':
                    dihs_list = [l+[n] for l in dihs_list for n in [0, 180]]
                else:
                    dihs_list = [l+[180] for l in dihs_list]
            # phi for PRO
            elif dih[0] == 'M' and name == 'P' and int(dih[1:])%2 == 1:
                # phi + chi = -180
                dihs_list = [
                    l+[n] for l in dihs_list for n in [-105, -60, -15]]
            # chi for PRO
            elif dih[0] == 's' and name == 'P' and int(dih[-1]) == 1:
                dihs_list = [l+[-180-l[2*resi]] for l in dihs_list]
            # omega before PRO
            elif dih[0] == 's' and name == 'P' and int(dih[-1]) == 2:
                dihs_list = [l+[n] for l in dihs_list for n in [0, 180]]
            # omega after PRO
            elif dih[0] == 's' and name == 'P' and int(dih[-1]) == 3:
                dihs_list = [l+[180] for l in dihs_list]
            # degenerate final chi
            elif atom_list[0] == atom_list[-1]:
                dihs_list = [l+[95] for l in dihs_list]
            else:
                dihs_list = [l+[n] for l in dihs_list for n in [0, 120, 240]]
        mi = min(len(dihs_list), 1000000)
        output_dict = {f'S{i}': 
                       {'dihl': dihs_list[i]} for i in range(mi)}
        self.comm.bcast(len(output_dict), root=ROOT_RANK)
        # print(output_dict['S0'])
        print(f'{self.seq} systematics {len(output_dict)}')
        return output_dict

    def joint3_2level(self, seq1='', seq2='', seq3='', l1=5, l2=8, d1=None, 
                      d2=None, d3=None, from_file=False):
        if from_file:
            return {}
        if self.rank != ROOT_RANK:
            n = self.comm.bcast(0, root=ROOT_RANK)
            return {i: i for i in range(n)}
        # e, energy array; dl, dihedral angle list list
        print(f'joint3 {seq1} {len(d1)}, {seq2} {len(d2)}, {seq3} {len(d3)}')
        e1, dl1 = [], []
        for n in sorted(d1.items(), key=lambda x: x[1]['energy']):
            e1.append(n[1]['energy'])
            dl1.append(n[1]['dihl'])
        e1 = np.array(e1) - min(e1)
        e2, dl2 = [], []
        for n in sorted(d2.items(), key=lambda x: x[1]['energy']):
            e2.append(n[1]['energy'])
            dl2.append(n[1]['dihl'])
        e2 = np.array(e2) - min(e2)
        e3, dl3 = [], []
        for n in sorted(d3.items(), key=lambda x: x[1]['energy']):
            e3.append(n[1]['energy'])
            dl3.append(n[1]['dihl'])
        e3 = np.array(e3) - min(e3)
        assert len(seq1) * len(seq2) * len(seq3) > 0
        assert len(dl1) * len(dl2) * len(dl3) * len(e1) * len(e2) * len(e3) > 0
        output_dict = {}
        i11, i12 = np.where(e1 <= l1)[0], np.where(l1 < e1[e1 <= l2])[0]
        i21, i22 = np.where(e2 <= l1)[0], np.where(l1 < e2[e2 <= l2])[0]
        i31, i32 = np.where(e3 <= l1)[0], np.where(l1 < e3[e3 <= l2])[0]
        gd1, gd2, gd3 = GetDihs(seq=seq1), GetDihs(seq=seq2), GetDihs(seq=seq3)
        d_1 = {'M%d' % (i + 1): i for i in range(2 * len(seq1))}
        d_2 = {'M%d' % (i + 1 + 2 * len(seq1)): 
               i for i in range(2 * len(seq2))}
        d_3 = {'M%d' % (i + 1 + 2 * len(seq1) + 2 * len(seq2)): 
               i for i in range(2 * len(seq3))}
        for i in range(2 * len(seq1) + 1, len(gd1.dihs_list)):
            d_1[gd1.dihs_list[i]] = i
        for i in range(2 * len(seq2) + 1, len(gd2.dihs_list)):
            j = int(gd2.dihs_list[i][1:-1]) + len(seq1)
            d_2['%s%d%s' % (gd2.dihs_list[i][0], j, gd2.dihs_list[i][-1])] = i
        for i in range(2 * len(seq3) + 1, len(gd3.dihs_list)):
            j = int(gd3.dihs_list[i][1:-1]) + len(seq1) + len(seq2)
            d_3['%s%d%s' % (gd3.dihs_list[i][0], j, gd3.dihs_list[i][-1])] = i
        d_3['M0'] = 2 * len(seq3)
        for i1 in i11:
            for i2 in i21:
                for i3 in i31:
                    output_dict[f'j{i1}_{i2}_{i3}'] = {'dihl': []}
                    for k in self.get_dihs.dihs_list:
                        x = findk3(k, d_1, d_2, d_3, i1, i2, i3, dl1, dl2, dl3)
                        output_dict[f'j{i1}_{i2}_{i3}']['dihl'].append(x)
        for i1 in i12:
            for i2 in i21:
                for i3 in i31:
                    output_dict[f'j{i1}_{i2}_{i3}'] = {'dihl': []}
                    for k in self.get_dihs.dihs_list:
                        x = findk3(k, d_1, d_2, d_3, i1, i2, i3, dl1, dl2, dl3)
                        output_dict[f'j{i1}_{i2}_{i3}']['dihl'].append(x)
        for i1 in i11:
            for i2 in i22:
                for i3 in i31:
                    output_dict[f'j{i1}_{i2}_{i3}'] = {'dihl': []}
                    for k in self.get_dihs.dihs_list:
                        x = findk3(k, d_1, d_2, d_3, i1, i2, i3, dl1, dl2, dl3)
                        output_dict[f'j{i1}_{i2}_{i3}']['dihl'].append(x)
        for i1 in i11:
            for i2 in i21:
                for i3 in i32:
                    output_dict[f'j{i1}_{i2}_{i3}'] = {'dihl': []}
                    for k in self.get_dihs.dihs_list:
                        x = findk3(k, d_1, d_2, d_3, i1, i2, i3, dl1, dl2, dl3)
                        output_dict[f'j{i1}_{i2}_{i3}']['dihl'].append(x)
        self.comm.bcast(len(output_dict), root=ROOT_RANK)
        return output_dict

    def joint_3level(self, seq1='', seq2='', dl1=None, dl2=None, e1=None, 
                     e2=None, l1=5, l2=8, l3=10, d1=None, d2=None, 
                     from_file=False):
        if from_file:
            return {}
        if self.rank != ROOT_RANK:
            n = self.comm.bcast(0, root=ROOT_RANK)
            return {i: i for i in range(n)}
        # e, energy array; dl, dihedral angle list list
        dl1, e1, d1 = joint_p1(dl1, e1, d1)
        dl2, e2, d2 = joint_p1(dl2, e2, d2)
        assert len(seq1) * len(seq2) > 0
        assert len(dl1) * len(dl2) * len(e1) * len(e2) > 0
        output_dict = {}
        e1 -= min(e1)
        e2 -= min(e2)
        i11, i12 = np.where(e1 <= l1)[0], np.where(l1 < e1[e1 <= l2])[0]
        i13 = np.where(l2 < e1[e1 <= l3])[0]
        i21, i22 = np.where(e2 <= l1)[0], np.where(e2 <= l2)[0]
        i23 = np.where(e2 <= l3)[0]
        gd1, gd2 = GetDihs(seq=seq1), GetDihs(seq=seq2)
        # {dihedral angle in seq: index in seq1}
        d_1 = {'M%d' % (i + 1): i for i in range(2 * len(seq1))}
        for i in range(2 * len(seq1) + 1, len(gd1.dihs_list)):
            d_1[gd1.dihs_list[i]] = i # side-chain
        # {dihedral angle in seq: index in seq2}
        d_2 = {f'M{i + 1 + 2 * len(seq1)}': i for i in range(2 * len(seq2))}
        d_2['M0'] = 2 * len(seq2)
        for i in range(2 * len(seq2) + 1, len(gd2.dihs_list)):
            j = int(gd2.dihs_list[i][1:-1]) + len(seq1)
            d_2['%s%d%s' % (gd2.dihs_list[i][0], j, gd2.dihs_list[i][-1])] = i
        for i in i11:
            for j in i23:
                n = f'j{i}_{j}'
                output_dict[n] ={'dihl': []}
                for k in self.get_dihs.dihs_list:
                    if k in d_1.keys():
                        output_dict[n]['dihl'].append(dl1[i][d_1[k]])
                    else:
                        output_dict[n]['dihl'].append(dl2[j][d_2[k]])
        for i in i13:
            for j in i21:
                n = f'j{i}_{j}'
                output_dict[n] = {'dihl': []}
                for k in self.get_dihs.dihs_list:
                    if k in d_1.keys():
                        output_dict[n]['dihl'].append(dl1[i][d_1[k]])
                    else:
                        output_dict[n]['dihl'].append(dl2[j][d_2[k]])
        for i in i12:
            for j in i22:
                n = f'j{i}_{j}'
                output_dict[n] = {'dihl': []}
                for k in self.get_dihs.dihs_list:
                    if k in d_1.keys():
                        output_dict[n]['dihl'].append(dl1[i][d_1[k]])
                    else:
                        output_dict[n]['dihl'].append(dl2[j][d_2[k]])
        self.comm.bcast(len(output_dict), root=ROOT_RANK)
        return output_dict

    def qc_dih(self, dihs):
        new_dihs = deepcopy(dihs)
        if self.ct == 'dep':
            new_dihs[2*len(self.seq)-1] %= 180
        n = 2 * len(self.seq)
        for i in range(len(self.seq)-1, -1, -1):
            if self.seq[i] == 'k':
                new_dihs[n+5] %= 120
            elif self.seq[i] == 'd':
                new_dihs[n+2] %= 180
            elif self.seq[i] == 'e':
                new_dihs[n+3] %= 180
            elif self.seq[i] == 'F':
                new_dihs[n+2] %= 180
            elif self.seq[i] == 'Y':
                new_dihs[n+2] %= 180
            elif self.seq[i] == 'y':
                new_dihs[n+2] %= 180
            n += len(sidechain_dihedrals[self.seq[i]])
        return new_dihs


