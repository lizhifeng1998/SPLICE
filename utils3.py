# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 2024

@author: lzf
"""

# str2tree with more than one line
# find() with less lines
# Pro rotating dihedrals

import os
import pickle
import tempfile
from contextlib import contextmanager

from utils2 import MyUtils, ROOT_RANK


def str2tree(_s):
    if _s is None:
        return []
    if '(' not in _s:
        return (_s,)
    s = _s.replace('\n','')
    a = s.split('(')[0]
    s1 = s[len(a)+1:-1]
    flag = 0
    for i in range(len(s1)):
        if s1[i] == '(':
            flag += 1
        elif s1[i] == ')':
            flag -= 1
        if flag < 0:
            break
    for j in range(i+1, len(s1)):
        if s1[j] == '(':
            flag += 1
        elif s1[j] == ')':
            flag -= 1
        if flag < 0:
            break
    s2 = s1[:i]
    if j+2 < len(s1):
        return a, str2tree(s2), str2tree(s1[i+2:j]), str2tree(s1[j+2:])
    else:
        return a, str2tree(s2), str2tree(s1[i+2:])


def list2tree(x):
    if len(x) == 1:
        return (x[0][0],)
    elif len(x) == 2:
        if x[1][0] == '0':
            return (x[0][0],)
        else:
            return (x[0][0], (x[1][0],), (x[1][1],))
    else:
        return (x[0][0], list2tree([x[i+1][:2**i] for i in range(len(x)-1)]), 
                list2tree([x[i+1][2**i:2**(i+1)] for i in range(len(x)-1)]))


def tree2str(fn="split_file", test_flag=False):
    with open(fn, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    if test_flag:
        print(lines)
    tree_list = list2tree(lines[:-1])
    o = str(tree_list)[1:-1].replace(',', '').replace(' ', '').replace("'", '')
    return o


def test_tree2str():
    tree_str = 'A\nA'
    fp = tempfile.NamedTemporaryFile('w')
    fp.write(tree_str)
    fp.seek(0)
    print(tree2str(fp.name, test_flag=True))
    fp.close()


def test_str2tree():
    s = 'ABC,pro,dep(A,pro)(B)(C,,dep)'
    print(str2tree(s))
    s = 'ABCDEF,nt=pro,ct=dep(AB(A)(B))(CD)(EF)'
    print(str2tree(s))


class Param(dict):
    def __init__(self, args, rank=ROOT_RANK):
        dict.__init__(self)
        if args.sequence_tree is None:
            with open("split_file", 'r') as f:
                lines = [line.strip().split() for line in f.readlines()]
                self['SequenceTreeList'] = [list2tree(lines[:-1])]
        else:
            self['SequenceTreeList'] = []
            for x in args.sequence_tree:
                self['SequenceTreeList'].append(str2tree(x))
        self['EnsembleDictionary'] = {}
        self['FirstCluster'] = {'factor': 0.8, 'max_quanzhong': 4, 
                                'max_tfd': 90, 'domain': None, 'tn': None}
        self['FinalCluster'] = {'factor': 0.8, 'max_quanzhong': 4, 
                                'max_tfd': 90, 'domain': (5, 8, 10, 99), 
                                'tn': (400, 300, 200, 100)}
        self['Joint'] = {'l1':5, 'l2':8, 'l3':10}
        self['BondRotation'] = {'NumberOnce': args.number_once, 
                                'TurnNumber': args.turn_number}
        self['XTBSolvationModel'] = args.xtb_solvation
        self['XTBLevel'] = args.xtb_level
        self['OPENMMSolvationModel'] = args.openmm_solvation
        self['OPENMMTolerance'] = args.openmm_tolerance
        self['OPENMMffxml'] = args.openmm_ffxml
        self['DFTBSolvationModel'] = args.dftb_solvation
        self['UserInteger1'] = args.user_int1

        if os.path.exists('command_file') and rank == ROOT_RANK:
            with open('command_file', 'r') as f:
                lines = f.readlines()
        else:
            lines = []
        for line in lines:
            s = line.strip().split()
            if line.startswith('lzf.protocol1.first_cluster.factor='):
                self['FirstCluster']['factor'] = float(s[0][35:])
            if line.startswith('lzf.protocol1.first_cluster.max_quanzhong='):
                self['FirstCluster']['max_quanzhong'] = float(s[0][42:])
            if line.startswith('lzf.protocol1.first_cluster.max_tfd='):
                self['FirstCluster']['max_tfd'] = float(s[0][36:])
            if line.startswith('lzf.protocol1.first_cluster.domain='):
                if len(s[0]) == 35:
                    self['FirstCluster']['domain'] = None
                else:
                    self['FirstCluster']['domain'] = [
                        float(x) for x in s[0][35:].split(',')]
            if line.startswith('lzf.protocol1.first_cluster.tn='):
                if len(s[0]) == 31:
                    self['FirstCluster']['tn'] = None
                else:
                    self['FirstCluster']['tn'] = [
                        int(x) for x in s[0][31:].split(',')]
            if line.startswith('lzf.protocol1.joint_3level.l1'):
                self['Joint']['l1'] = float(s[0][30:])
            if line.startswith('lzf.protocol1.joint_3level.l2'):
                self['Joint']['l2'] = float(s[0][30:])
            if line.startswith('lzf.protocol1.joint_3level.l3'):
                self['Joint']['l3'] = float(s[0][30:])
            if line.startswith('lzf.protocol1.bond_rotation.turn_number'):
                self['BondRotation']['TurnNumber'] = int(s[0][40:])
            if line.startswith('lzf.protocol1.bond_rotation.number_once'):
                self['BondRotation']['NumberOnce'] = int(s[0][40:])
            if line.startswith('lzf.protocol1.final_cluster.factor='):
                self['FinalCluster']['factor'] = float(s[0][35:])
            if line.startswith('lzf.protocol1.final_cluster.max_quanzhong='):
                self['FinalCluster']['max_quanzhong'] = float(s[0][42:])
            if line.startswith('lzf.protocol1.final_cluster.max_tfd='):
                self['FinalCluster']['max_tfd'] = float(s[0][36:])
            if line.startswith('lzf.protocol1.final_cluster.domain='):
                if len(s[0]) == 35:
                    self['FinalCluster']['domain'] = None
                else:
                    self['FinalCluster']['domain'] = [
                        float(x) for x in s[0][35:].split(',')]
            if line.startswith('lzf.protocol1.final_cluster.tn='):
                if len(s[0]) == 31:
                    self['FinalCluster']['tn'] = None
                else:
                    self['FinalCluster']['tn'] = [
                        int(x) for x in s[0][31:].split(',')]


@contextmanager
def my_utils(comm, seq, cd, **kwargs):
    mu = MyUtils(comm, seq, cd, **kwargs)
    try:
        yield mu
    finally:
        mu.end()


def get_dih_list(param, seq0):
    if param['BondRotation']['TurnNumber'] == 8 and len(seq0) > 4:
        n = len(seq0)
        _dihi_list = [0, 1, n-2, n-1, n, n+1, 2*n-2, 2*n-1]
    elif param['BondRotation']['TurnNumber'] == 8 and len(seq0) == 4:
        # _dihi_list = [2, 3, 4, 5, 0, 1, 6, 7]
        # _dihi_list = [0, 1, 6, 7, 2, 3, 4, 5]
        # _dihi_list = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
        # _dihi_list = [0, 1, 2, 3, 4, 5, 6, 7, 1, 0]
        _dihi_list = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        _dihi_list = [i for i in range(2*len(seq0))]
    # phi of P unrotatable
    indices_to_delete = [i*2 for i in range(len(seq0)) if seq0[i] == 'P']
    dihi_list = []
    for index, value in enumerate(_dihi_list):
        if index not in indices_to_delete:
            dihi_list.append(value)
    dihn_list = [param['BondRotation']['NumberOnce'] for _ in dihi_list]
    return dihi_list, dihn_list

def get_nt_ct_sol2(seq_tree):
    pl = seq_tree[0].split(',')
    par = {'nt': None, 'ct': None, 'xtb_sol': None, 'seq': pl[0],
           'openmm_sol': None, 'xtb_lvl': None}
    for p in pl[1:]:
        par[p.split('=')[0]] = p.split('=')[1]
    return par

def get_seq_dict_nofind(comm, seq_tree, param, wd, domain=None, tn=None, 
                        nt=None, ct=None):
    mypar = get_nt_ct_sol2(seq_tree)
    if mypar['xtb_sol'] is None:
        mypar['xtb_sol'] = param['XTBSolvationModel']
    if mypar['xtb_lvl'] is None:
        mypar['xtb_lvl'] = param['XTBLevel']
    if mypar['openmm_sol'] is None:
        mypar['openmm_sol'] = param['OPENMMSolvationModel']
    if mypar['nt'] is None:
        mypar['nt'] = 'ace' if nt is None else nt
    if mypar['ct'] is None:
        mypar['ct'] = 'nme' if ct is None else ct
    seq, nt, ct = mypar['seq'], mypar['nt'], mypar['ct']
    if os.path.exists(f"{wd}/{seq}_{nt}_{ct}_uni_diha_list") \
        and os.path.exists(f"{wd}/{seq}_{nt}_{ct}_dihedra_list"):
        with my_utils(comm, seq, f"{wd}/{seq}_{nt}_{ct}") as s0:
            d0 = s0.create_dict_from_file(
                df=f"{wd}/{seq}_{nt}_{ct}_dihedra_list", 
                ef=f"{wd}/{seq}_{nt}_{ct}_uni_diha_list")
            d0 = s0.quchong(input_dict=d0)
            return True, s0.quchongp2(
                d0, factor=param['FirstCluster']['factor'], 
                max_quanzhong=param['FirstCluster']['max_quanzhong'], 
                max_tfd=param['FirstCluster']['max_tfd'], )
    if os.path.exists(f"{wd}/{seq}_{nt}_{ct}.pkl"):
        with my_utils(comm, seq, f"{wd}/{seq}_{nt}_{ct}") as s1:
            with open(f"{wd}/{seq}_{nt}_{ct}.pkl", 'rb') as f:
                d0 = pickle.load(f)
                k0 = next(iter(d0.keys()))
                d1 = s1.quchong(input_dict=d0[k0])
            if domain is not None and tn is not None:
                d1 = s1.quchongp2(d1, factor=0.8, max_quanzhong=4, max_tfd=90, 
                                  domain=domain, tn=tn,)
        return True, d1
    if nt != 'ace' or ct != 'nme':
        return False, mypar
    if seq in param['EnsembleDictionary'].keys():
        return True, param['EnsembleDictionary'][seq]
    return False, mypar

def pj3(seq_tree, d0, findx, comm, param, sd, nt, ct, mu):
    seq1 = seq_tree[1][0].split(',')[0]
    seq2 = seq_tree[2][0].split(',')[0]
    seq3 = seq_tree[3][0].split(',')[0]
    if os.path.exists(f'{d0}/1of3.pkl'):
        d1 = mu.load_pkl(f'{d0}/1of3.pkl')
    else:
        d1 = findx(comm, param, seq_tree[1], sd, domain=(5, 8), 
                tn=(50, 30), nt=nt)
        mu.write_pkl(d1, prefix='1of3_')
    if os.path.exists(f'{d0}/2of3.pkl'):
        d2 = mu.load_pkl(f'{d0}/2of3.pkl')
    else:
        d2 = findx(comm, param, seq_tree[2], sd, domain=(5, 8), 
                tn=(50, 30))
        mu.write_pkl(d2, prefix='2of3_')
    if os.path.exists(f'{d0}/3of3.pkl'):
        d3 = mu.load_pkl(f'{d0}/3of3.pkl')
    else:
        d3 = findx(comm, param, seq_tree[3], sd, domain=(5, 8), 
                tn=(50, 30), ct=ct)
        mu.write_pkl(d3, prefix='3of3_')
    return mu.joint3_2level(seq1, seq2, seq3, d1=d1, d2=d2, d3=d3)

def pj(seq_tree, d0, findx, comm, param, sd, nt, ct, mu, pj_f):
    seq1 = seq_tree[1][0].split(',')[0]
    seq2 = seq_tree[2][0].split(',')[0]
    if len(seq_tree) == 4:
        # 3 to 1
        start_dict = pj3(seq_tree, d0, findx, comm, param, sd, nt, ct, mu)
    else:
        if os.path.exists(f'{d0}/1of2.pkl'):
            d1 = mu.load_pkl(f'{d0}/1of2.pkl')
        else:
            d1 = findx(comm, param, seq_tree[1], sd, nt=nt)
            mu.write_pkl(d1, prefix='1of2_')
        if os.path.exists(f'{d0}/2of2.pkl'):
            d2 = mu.load_pkl(f'{d0}/2of2.pkl')
        else:
            d2 = findx(comm, param, seq_tree[2], sd, ct=ct)
            mu.write_pkl(d2, prefix='2of2_')
        start_dict = mu.joint_3level(seq1=seq1, seq2=seq2, d1=d1, d2=d2)
    pinjie_dict = pj_f(mu, start_dict) # 1
    qc_dict = mu.quchong(input_dict=pinjie_dict)
    mu.write_pkl({'pinjie_qc_dict': qc_dict}, prefix='splice_')
    return qc_dict

def find0(comm, param, seq_tree, sd, pj_f, br_f, xt_f, final_f, findx, 
          domain=None, tn=None, nt=None, ct=None):
    flag, seq0 = get_seq_dict_nofind(comm, seq_tree, param, sd, domain=domain, 
                                     tn=tn, nt=nt, ct=ct)
    if flag:
        return seq0
    seq, xtb_sol, nt, ct = seq0['seq'], seq0['xtb_sol'], seq0['nt'], seq0['ct']
    osol, xl = seq0['openmm_sol'], seq0['xtb_lvl']
    ds, ot = param['DFTBSolvationModel'], param['OPENMMTolerance']
    of = param['OPENMMffxml']
    dihi_list, dihn_list = get_dih_list(param, seq)
    # start finding
    d0 = f'{sd}/{seq}_{nt}_{ct}'
    with my_utils(comm, seq, d0, nt=nt, ct=ct, dftb_sol=ds) as mu:
        mu.ffxml = of
        mu.xtb_lvl = xl
        mu.openmm_sol = osol
        mu.openmm_tol = ot
        mu.xtb_sol=xtb_sol
        if len(seq_tree) > 1:
            if os.path.exists(f'{d0}/splice.pkl'):
                qc_dict = mu.load_pkl(f'{d0}/splice.pkl', 'pinjie_qc_dict')
            else:
                qc_dict = pj(seq_tree, d0, findx, comm, 
                             param, sd, nt, ct, mu, pj_f) # 1
            for dihi, dihn in zip(dihi_list, dihn_list):
                mu.print(dihi, dihi_list)
                if dihn == -1:
                    q2_dict = mu.quchongp2(qc_dict, factor=0.8, 
                                           max_quanzhong=4, max_tfd=90,)
                    br_dict = mu.bond_rotate(q2_dict, dihi, dihn=dihn)
                elif dihn == -2:
                    q2_dict = mu.quchongp2(qc_dict, factor=0.8, 
                                           max_quanzhong=4, max_tfd=90,)
                    br_dict = mu.bond_rotate2(q2_dict, dihi_list, dihn=dihn)
                elif dihn == -3:
                    dict_list = mu.quchongp2(qc_dict, split=True)
                    opt_dict = mu.quchongp2(dict_list[0], factor=0.8, 
                                           max_quanzhong=4, max_tfd=90,)
                    sp_dict = mu.quchongp2(dict_list[-1], factor=0.8, 
                                           max_quanzhong=4, max_tfd=90,)
                    q2_dict = mu.combine_dict(sp_dict, opt_dict)
                    br_dict = mu.bond_rotate2(q2_dict, dihi_list, dihn=dihn)
                else:
                    br_dict = mu.bond_rotate(qc_dict, dihi, dihn=dihn)
                b2_dict = br_f(mu, br_dict) # 2
                qc_dict = mu.quchong(input_dict=mu.combine_dict(qc_dict, 
                                                                b2_dict))
                mu.write_pkl({'bond_rotate_qc_dict': qc_dict}, 
                             prefix=f'br_{dihi}_')
        else:
            start_dict = mu.systematics()
            sys_dict = xt_f(mu, start_dict) # 3
            qc_dict = mu.quchong(input_dict=sys_dict)
            mu.write_pkl({'sys_qc_dict': qc_dict})
        qc_dict = mu.quchongp2(qc_dict, factor=0.8, max_quanzhong=4, 
                               max_tfd=90,)
        qc_dict = final_f(mu, qc_dict) # 4
        mu.write_pkl({'final_dict': qc_dict}, prefix='final_')
        qc_dict = mu.quchong(input_dict=qc_dict)
        if domain is not None and tn is not None:
            qc_dict = mu.quchongp2(qc_dict, factor=0.8, max_quanzhong=4, 
                                   max_tfd=90, domain=domain, tn=tn,)
        mu.write_uni_dih_file(qc_dict)
        if nt == 'ace' and ct == 'nme':
            param['EnsembleDictionary'][seq_tree[0]] = qc_dict
        return qc_dict

# XTB only
def find1(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find1, domain=domain, tn=tn, nt=nt, ct=ct)

def find2(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        openmm_qc_dict = mu.quchong(input_dict=openmm_dict)
        return mu.xtb_sp_from_xyz(openmm_qc_dict)
    def my_F(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find2, domain=domain, tn=tn, nt=nt, ct=ct)

def find3(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        openmm_qc_dict = mu.quchong(input_dict=openmm_dict)
        return mu.rosetta_relax_from_pdb_xyz_dih(openmm_qc_dict)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find3, domain=domain, tn=tn, nt=nt, ct=ct)

def find4(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        return mu.openmm_opt_from_xyz_dih_pdb(d)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find4, domain=domain, tn=tn, nt=nt, ct=ct)

def find5(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        openmm_qc_dict = mu.quchong(input_dict=openmm_dict)
        return mu.dftb_sp_from_pdb_xyz(openmm_qc_dict)
    def my_F(mu, d):
        return mu.dftb_opt_from_pdb_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find5, domain=domain, tn=tn, nt=nt, ct=ct)

def find6(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d, ns=True)
        openmm_qc_dict = mu.quchong(input_dict=openmm_dict)
        return mu.openmm_opt_from_xyz_dih_pdb(openmm_qc_dict)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find6, domain=domain, tn=tn, nt=nt, ct=ct)

# ROSETTA only
def find7(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        return mu.rosetta_relax_from_pdb_xyz_dih(d)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find7, domain=domain, tn=tn, nt=nt, ct=ct)

def find8(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        return mu.rosetta_relax_from_pdb_xyz_dih(openmm_dict)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find8, domain=domain, tn=tn, nt=nt, ct=ct)

def find9(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        openmm_qc_dict = mu.quchong(input_dict=openmm_dict)
        return mu.amber_opt_from_pdb_xyz(openmm_qc_dict)
    my_F = lambda x, y: y
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find9, domain=domain, tn=tn, nt=nt, ct=ct)

def find10(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        amber_dict = mu.amber_opt_from_pdb_xyz(openmm_dict, opt=False)
        openmm_qc_dict = mu.quchong(input_dict=amber_dict)
        return mu.xtb_sp_from_xyz(openmm_qc_dict)
    def my_F(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find10, domain=domain, tn=tn, nt=nt, ct=ct)

def find11(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        amber_dict = mu.amber_opt_from_pdb_xyz(openmm_dict)
        openmm_qc_dict = mu.quchong(input_dict=amber_dict)
        return mu.xtb_sp_from_xyz(openmm_qc_dict)
    def my_F(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find11, domain=domain, tn=tn, nt=nt, ct=ct)

def find12(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        return mu.xtb_sp_from_xyz(openmm_dict)
    def my_F(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find12, domain=domain, tn=tn, nt=nt, ct=ct)

def find13(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        sp_dict = mu.xtb_sp_from_xyz(openmm_dict)
        qc_dict = mu.quchongp2(sp_dict)
        opt_dict = mu.xtb_opt_from_xyz_dih(qc_dict)
        return mu.combine_dict(sp_dict, opt_dict)
    def my_F(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find13, domain=domain, tn=tn, nt=nt, ct=ct)

def find14(comm, param, seq_tree, sd, domain=None, tn=None, nt=None, ct=None):
    def my_f(mu, d):
        openmm_dict = mu.openmm_opt_from_xyz_dih_pdb(d)
        openmm_qc_dict = mu.quchong(input_dict=openmm_dict)
        sp_dict = mu.xtb_sp_from_xyz(openmm_qc_dict)
        # sp_dict = mu.xtb_sp_from_xyz(openmm_dict)
        qc_dict = mu.quchongp2(sp_dict, max_tfd=param['UserInteger1'])
        opt_dict = mu.xtb_opt_from_xyz_dih(qc_dict)
        return mu.combine_dict(sp_dict, opt_dict)
    def my_F(mu, d):
        return mu.xtb_opt_from_xyz_dih(d)
    return find0(comm, param, seq_tree, sd, pj_f=my_f, br_f=my_f, xt_f=my_f, 
                 final_f=my_F, findx=find14, domain=domain, tn=tn, nt=nt, ct=ct)


if __name__ == '__main__':
    # test_tree2str()
    test_str2tree()

