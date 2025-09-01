# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:10:57 2024

@author: 11976
"""

import sys
import math
import glob
import time
import numpy as np
import networkx as nx
from copy import deepcopy
from networkx.algorithms import isomorphism

from utils1 import one2three, AtomName, Elements

Bonds = {
    'A': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10)],
    'C': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11)],
    'D': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (11, 13)],
    'd': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12)],
    'E': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (14, 16)],
    'e': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15)],
    'F': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 19), (11, 12), (11, 13), (13, 14), (13, 15), (15
, 16), (15, 17), (17, 18), (17, 19), (19, 20)],
    'G': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7)],
    '6': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 17), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15
, 17), (17, 18)],
    'h': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 16), (11, 12), (11, 13), (13, 14), (13, 15), (15
, 16), (16, 17)],
    'H': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 16), (11, 12), (12, 13), (12, 14), (14, 15), (14
, 16), (16, 17)],
    'I': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (10, 17), (10, 18), (10, 19), (11,
 14), (11, 15), (11, 16)],
    'k': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (11, 16), (14,
 17), (14, 18), (14, 19), (17, 20), (17, 21), (17, 22)],
    'K': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (11, 16), (14,
 17), (14, 18), (14, 19), (17, 20), (17, 21)],
    'L': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (11, 16), (13,
 17), (13, 18), (13, 19)],
    'M': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (14, 15), (14, 16), (14,
 17)],
    'N': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (11, 13), (11, 14)],
    'P': [(1, 2), (2, 8), (2, 13), (2, 14), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12)],
    'Q': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (14, 16), (14,
 17)],
    '4': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (11, 16), (14,
 17), (14, 24), (17, 18), (17, 19), (18, 20), (18, 21), (19, 22), (19, 23)],
    'R': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 
13), (11, 14), (11, 15), (11, 16), (14, 17), (14, 18), (17, 19), (17, 20), (19, 21), (20, 22), (20, 23)],
    'r': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (11, 14), (11, 15), (11, 16), (14,
 17), (17, 18), (17, 19), (18, 20), (18, 21), (19, 22), (19, 23)],
    'S': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11)],
    'T': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (10, 12), (10, 13), (10, 14)],
    'V': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 12), (8, 13), (10, 14), (10, 15), (10, 16)],
    'W': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 23), (11, 12), (11, 20), (12, 13), (12, 14), (14
, 15), (14, 16), (16, 17), (16, 18), (18, 19), (18, 20), (20, 21), (21, 22), (21, 23), (23, 24)],
    'Y': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 19), (11, 12), (11, 13), (13, 14), (13, 15), (15
, 16), (15, 17), (16, 21), (17, 18), (17, 19), (19, 20)],
    'y': [(1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 7), (6, 8), (6, 9), (6, 10), (8, 11), (8, 19), (11, 12), (11, 13), (13, 14), (13, 15), (15
, 16), (15, 17), (17, 18), (17, 19), (19, 20)],
    'ACE': [(1, 2), (1, 4), (1, 5), (1, 6), (2, 3)],
    'NME': [(1, 2), (1, 4), (1, 5), (1, 6), (2, 3)],
    'NHE': [(1, 2), (1, 3)],
}
other_Bonds = {
    'HOH': [(1, 2), (1, 3)],
}
other_Elements = {
    'HOH': ['O', 'H', 'H']
}
other_AtomName = {
    'HOH': ['O', 'H1', 'H2']
}

neu_n_Elements = {k: Elements[k] + ['H'] for k in Elements.keys() if len(k) == 1}
pro_n_Elements = {k: Elements[k] + ['H', 'H'] for k in Elements.keys() if len(k) == 1}
dep_c_Elements = {k: Elements[k] + ['O'] for k in Elements.keys() if len(k) == 1}
neu_c_Elements = {k: Elements[k] + ['O', 'H'] for k in Elements.keys() if len(k) == 1}
neu_n_dep_c_Elements = {k: neu_n_Elements[k] + ['O'] for k in neu_n_Elements.keys()}
neu_n_neu_c_Elements = {k: neu_n_Elements[k] + ['O', 'H'] for k in neu_n_Elements.keys()}
pro_n_dep_c_Elements = {k: pro_n_Elements[k] + ['O'] for k in pro_n_Elements.keys()}
pro_n_neu_c_Elements = {k: pro_n_Elements[k] + ['O', 'H'] for k in pro_n_Elements.keys()}

nAtomName = deepcopy(AtomName)
for k in nAtomName.keys():
    if len(k) == 1:
        nAtomName[k][1] = 'H1'
neu_n_AtomName = {k: nAtomName[k] + ['H2'] for k in nAtomName.keys() if len(k) == 1}
pro_n_AtomName = {k: nAtomName[k] + ['H2', 'H3'] for k in nAtomName.keys() if len(k) == 1}
dep_c_AtomName = {k: AtomName[k] + ['OXT'] for k in AtomName.keys() if len(k) == 1}
neu_c_AtomName = {k: AtomName[k] + ['OXT', 'HXT'] for k in AtomName.keys() if len(k) == 1}
neu_n_dep_c_AtomName = {k: neu_n_AtomName[k] + ['OXT'] for k in neu_n_AtomName.keys()}
neu_n_neu_c_AtomName = {k: neu_n_AtomName[k] + ['OXT', 'HXT'] for k in neu_n_AtomName.keys()}
pro_n_dep_c_AtomName = {k: pro_n_AtomName[k] + ['OXT'] for k in pro_n_AtomName.keys()}
pro_n_neu_c_AtomName = {k: pro_n_AtomName[k] + ['OXT', 'HXT'] for k in pro_n_AtomName.keys()}

neu_n_Bonds = {k: Bonds[k] + [(1, len(Elements[k]) + 1)] for k in Bonds.keys() if len(k) == 1}
pro_n_Bonds = {k: Bonds[k] + [(1, len(Elements[k]) + 1), (1, len(Elements[k]) + 2)] for k in Bonds.keys() if len(k) == 1}
dep_c_Bonds = {k: Bonds[k] + [(4, len(Elements[k]) + 1)] for k in Bonds.keys() if len(k) == 1}
neu_c_Bonds = {k: Bonds[k] + [(4, len(Elements[k]) + 1), (len(Elements[k]) + 1, len(Elements[k]) + 2)] for k in Bonds.keys() if len(k) == 1}
neu_n_dep_c_Bonds = {k: neu_n_Bonds[k] + [(4, len(neu_n_Elements[k]) + 1)] for k in neu_n_Bonds.keys()}
neu_n_neu_c_Bonds = {k: neu_n_Bonds[k] + [(4, len(neu_n_Elements[k]) + 1), (len(neu_n_Elements[k]) + 1, len(neu_n_Elements[k]) + 2)] for k in neu_n_Bonds.keys()}
pro_n_dep_c_Bonds = {k: pro_n_Bonds[k] + [(4, len(pro_n_Elements[k]) + 1)] for k in pro_n_Bonds.keys()}
pro_n_neu_c_Bonds = {k: pro_n_Bonds[k] + [(4, len(pro_n_Elements[k]) + 1), (len(pro_n_Elements[k]) + 1, len(pro_n_Elements[k]) + 2)] for k in pro_n_Bonds.keys()}

def res_graph(res, elements=Elements, bonds=Bonds):
    natoms = len(elements[res])
    G = nx.Graph()
    G.add_nodes_from([i + 1 for i in range(natoms)])
    element_dict = {i + 1: elements[res][i] for i in range(natoms)}
    for k in element_dict.keys():
        G.nodes[k]['element'] = element_dict[k]
    G.add_edges_from(bonds[res])
    return G

def xyz2graph(fn, test=False):
    atom_list = []
    G = nx.Graph()
    with open(fn, 'r') as f:
        i = 0
        for line in f.readlines()[2:]:
            s = line.strip().split()
            atom_list.append((s[0], float(s[1]), float(s[2]), float(s[3])))
            G.add_node(i)
            G.nodes[i]['element'] = s[0]
            G.nodes[i]['x'] = s[1]
            G.nodes[i]['y'] = s[2]
            G.nodes[i]['z'] = s[3]
            G.nodes[i]['env'] = 0
            G.nodes[i]['int'] = 1 if s[0] == 'H' else 10 if s[0] == 'C' else 100 if s[0] == 'N' else 1000 if s[0] == 'O' else 10000
            i += 1
    nbonds = [0 for _ in range(len(atom_list))]
    for i in range(len(atom_list)):
        for j in range(i+1, len(atom_list)):
            v = (atom_list[j][1] - atom_list[i][1], atom_list[j][2] - atom_list[i][2], atom_list[j][3] - atom_list[i][3])
            d = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            if atom_list[i][0] == 'H' and atom_list[j][0] == 'H':
                continue
            elif atom_list[i][0] == 'H' or atom_list[j][0] == 'H':
                # if atom_list[i][0] == 'C' or atom_list[j][0] == 'C':
                #     continue
                # elif atom_list[i][0] == 'N' or atom_list[j][0] == 'N':
                #     continue
                if 0.85 <= d <= 1.15:
                    G.add_edge(i, j)
                    nbonds[i] += 1
                    nbonds[j] += 1
                    G.nodes[i]['env'] += G.nodes[j]['int']
                    G.nodes[j]['env'] += G.nodes[i]['int']
            elif atom_list[i][0] == 'S' or atom_list[j][0] == 'S':
                if 1.7 <= d <= 2:
                    G.add_edge(i, j)
                    nbonds[i] += 1
                    nbonds[j] += 1
                    G.nodes[i]['env'] += G.nodes[j]['int']
                    G.nodes[j]['env'] += G.nodes[i]['int']
            else:
                if 1.1 <= d <= 1.62:
                    G.add_edge(i, j)
                    nbonds[i] += 1
                    nbonds[j] += 1
                    G.nodes[i]['env'] += G.nodes[j]['int']
                    G.nodes[j]['env'] += G.nodes[i]['int']
    if test:
        print('Graph:')
        for i, n in enumerate(nbonds):
            print(i, atom_list[i][0], n)
    return G

def find_res(G1, res, elements=Elements, bonds=Bonds):
    # find one subgraph
    Gres = res_graph(res, elements, bonds)
    GMres = isomorphism.GraphMatcher(G1, Gres)
    if GMres.subgraph_is_isomorphic():
        for m in GMres.subgraph_isomorphisms_iter():
            flag = False
            for n in m.items():
                if G1.nodes[n[0]]['element'] != Gres.nodes[n[1]]['element']:
                    flag = True
                    break
            if flag:
                continue
            return m
        
def find_res2(G1, res, elements=Elements, bonds=Bonds):
    # find one graph
    Gres = res_graph(res, elements, bonds)
    GMres = isomorphism.GraphMatcher(G1, Gres)
    if GMres.is_isomorphic():
        for m in GMres.isomorphisms_iter():
            flag = False
            for n in m.items():
                if G1.nodes[n[0]]['element'] != Gres.nodes[n[1]]['element']:
                    flag = True
                    break
            if flag:
                continue
            return m
        
def find_res_all(G1, res, elements=Elements, bonds=Bonds):
    # find all subgraph
    G2 = deepcopy(G1)
    ml = []
    while True:
        m = find_res(G2, res, elements, bonds)
        if m is None:
            break
        ml.append(deepcopy(m))
        for k in m.keys():
            G2.remove_node(k)
    return ml

def cut_conh(fn, verbose=False):
    G1 = xyz2graph(fn)
    G2 = deepcopy(G1)
    edges_to_remove = []
    ml, rl = [], [] # map and residue list
    xyz2pdb = {'index': [], 'prefix': [], 'suffix': []}
    ace_idx = None
    n_AtomName = None
    for node in G1.nodes:
        if G1.nodes[node]['element'] != 'C':
            continue
        elif G1.nodes[node]['env'] != 1110:
            continue
        for k in G1.neighbors(node):
            if G1.nodes[k]['element'] == 'C':
                n_c = k
            elif G1.nodes[k]['element'] == 'N':
                n_n = k
        if G1.nodes[n_c]['env'] == 22:
            continue # ASN or GLN
        edges_to_remove.append((node, n_n))
    for edge in edges_to_remove:
        G2.remove_edge(*edge)
    subgraph_list = [G2.subgraph(cc) for cc in nx.connected_components(G2)]
    subgraph_flag = [False for _ in subgraph_list]
    for k in pro_n_neu_c_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, pro_n_neu_c_Elements, pro_n_neu_c_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            n_AtomName = pro_n_neu_c_AtomName
            if verbose:
                print('N-term pro C-term neu')
            break # AA
    for k in pro_n_dep_c_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, pro_n_dep_c_Elements, pro_n_dep_c_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            n_AtomName = pro_n_dep_c_AtomName
            if verbose:
                print('N-term pro C-term dep')
            break # AA
    for k in pro_n_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, pro_n_Elements, pro_n_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            n_AtomName = pro_n_AtomName
            if verbose:
                print('N-term pro')
            break # N-term
    for k in neu_n_neu_c_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, neu_n_neu_c_Elements, neu_n_neu_c_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            n_AtomName = neu_n_neu_c_AtomName
            if verbose:
                print('N-term neu C-term neu')
            break # AA
    for k in neu_n_dep_c_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, neu_n_dep_c_Elements, neu_n_dep_c_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            n_AtomName = neu_n_dep_c_AtomName
            if verbose:
                print('N-term neu C-term dep')
            break # AA
    for k in neu_n_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, neu_n_Elements, neu_n_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            n_AtomName = neu_n_AtomName
            if verbose:
                print('N-term neu')
            break # N-term
    for k in neu_c_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, neu_c_Elements, neu_c_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            if verbose:
                print('C-term neu')
            break # C-term
    for k in dep_c_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, dep_c_Elements, dep_c_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            ml.append(m)
            rl.append(k)
            if verbose:
                print('C-term dep')
            break # C-term
    for k in Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k)
            if m is None:
                continue
            subgraph_flag[i]= True
            if k == 'ACE':
                ace_idx = len(ml)
            ml.append(m)
            rl.append(k)
    other_m = []
    other_r = []
    for k in other_Bonds.keys():
        for i in range(len(subgraph_list)):
            if subgraph_flag[i]:
                continue
            m = find_res2(subgraph_list[i], k, other_Elements, other_Bonds)
            if m is None:
                continue
            subgraph_flag[i]= True
            other_m.append(m)
            other_r.append(k)
    sl = []
    if ace_idx is not None:
        c0 = {i[1]: i[0] for i in ml[ace_idx].items()}[2] # C of ACE
        sl.append(ace_idx)
    else:
        try:
            c0 = {i[1]: i[0] for i in ml[0].items()}[4]
        except KeyError:
            if verbose:
                print(f'{fn} bad end')
            return 1
        sl.append(0) # N-term
    for _ in range(len(ml) - 1):
        try:
            n1 = {G1.nodes[k]['element']: k for k in G1.adj[c0].keys()}['N']
        except KeyError:
            if verbose:
                print(fn, G1.adj[c0])
            raise
        try:
            r1 = [i for i in range(len(ml)) if n1 in ml[i].keys()][0]
        except IndexError:
            if verbose:
                print(fn, [i for i in range(len(ml)) if n1 in ml[i].keys()], )
                print(ml, rl)
                print(G1.edges())
                print({k:G1.nodes[k]['element'] for k in G1.nodes()})
                print(edges_to_remove)
                print(subgraph_flag)
                print([cc for cc in nx.connected_components(G2)])
            raise
        sl.append(r1)
        if 4 in ml[r1].values():
            c1 = {i[1]: i[0] for i in ml[r1].items()}[4]
        else:
            break
        c0 = c1
    count_r, count_a = 0, 0
    for idx in sl:
        count_r += 1
        for i in ml[idx].items():
            count_a += 1
            atomname = n_AtomName if count_r == 1 and len(rl[idx]) == 1 else neu_c_AtomName if count_r == len(sl) and len(rl[idx]) == 1 else AtomName
            # print('ATOM  %5s %4s %s A%4s     %7.7s %7.7s %7.7s  1.00  0.00          %2s  ' % (
            #     count_a, atomname[rl[idx]][i[1] - 1], one2three[rl[idx]], count_r, G1.nodes[i[0]]['x'], G1.nodes[i[0]]['y'], G1.nodes[i[0]]['z'], G1.nodes[i[0]]['element']))
            try:
                xyz2pdb['prefix'].append('ATOM  %5s %4s %s A%4s    ' % (count_a, atomname[rl[idx]][i[1] - 1], one2three[rl[idx]], count_r))
            except TypeError:
                if verbose:
                    print(f'count_r={count_r}, len(rl[idx])={len(rl[idx])}, len(sl)={len(sl)}')
                raise
            xyz2pdb['index'].append(i[0])
            xyz2pdb['suffix'].append('  1.00  0.00          %2s  ' % G1.nodes[i[0]]['element'])
    for om, r in zip(other_m, other_r):
        count_r += 1
        for i in om.items():
            count_a += 1
            xyz2pdb['prefix'].append('HETATM%5s %4s %s A%4s    ' % (i[0]+1, other_AtomName[r][i[1] - 1], r, count_r))
            xyz2pdb['index'].append(i[0])
            xyz2pdb['suffix'].append('  1.00  0.00          %2s  ' % G1.nodes[i[0]]['element'])
    xyz = np.loadtxt(fn, skiprows=2, usecols=[1,2,3])
    if len(xyz) != len(xyz2pdb['prefix']):
        if verbose:
            print(f"{len(xyz)} {len(xyz2pdb['index'])} {len(xyz2pdb['prefix'])}")
            for i in range(len(xyz)):
                if i not in xyz2pdb['index']:
                    print(i)
            # print(xyz2pdb['prefix'])
        return 2
    with open(fn[:-4] + '.pdb', 'w') as f:
        for i, p, s in zip(xyz2pdb['index'], xyz2pdb['prefix'], xyz2pdb['suffix']):
            f.write('%s%8.3f%8.3f%8.3f%s\n' % (p, xyz[i][0], xyz[i][1], xyz[i][2], s))
    return 0

    
if __name__ =='__main__':
    # t = time.time()
    cut_conh(sys.argv[1], True)
    # print(time.time() - t)
    
