# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:14:27 2024

@author: lzf
"""

import sys
import time
import math
from copy import deepcopy
try:
    import numpy as np
    from openmm.app import PDBFile, Modeller, ForceField, Simulation, NoCutoff
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile, GBn2
    from openmm import LangevinMiddleIntegrator, OpenMMException
    from openmm.unit import Quantity, kelvin, picosecond, picoseconds
    from openmm.unit import kilojoule, nanometer, mole, kilocalorie, angstrom
    class MyForceField(ForceField):
        def __init__(self, *args):
            ForceField.__init__(self, *args)
except ModuleNotFoundError:
    pass


one2three = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASH', 'd': 'ASP',
    'E': 'GLH', 'e': 'GLU', 'F': 'PHE', 'G': 'GLY',
    'H': 'HIE', 'I': 'ILE', 'K': 'LYN', 'k': 'LYS',
    'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO',
    'Q': 'GLN', 'r': 'arg', '4': 'ARG', 'R': 'aRg',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP',
    'Y': 'TYR', 'ACE': 'ACE', 'NME': 'NME', 'NHE': 'NHE',
    'h': 'HID', '6': 'HIS', 'y': 'TyR', 'NH2': 'NH2'
}
three2one = {one2three[k]:k for k in one2three.keys()}
AtomName = {
    'A': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'HB1', 'HB2', 'HB3'],
    'C': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'SG', 'HB1', 'HB2', 
          'HG'],
    'D': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'OD1', 'OD2', 'HD1'],
    'd': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'OD1', 'OD2'],
    'E': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'OE2', 'OE1', 'HE2'],
    'e': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'OE2', 'OE1'],
    'F': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD1', 'HD1', 'CE1', 'HE1', 'CZ', 'HZ', 'CE2', 'HE2', 'CD2', 'HD2'],
    'G': ['N', 'H', 'CA', 'C', 'HA2', 'HA3', 'O'],
    '6': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2'],
    'h': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'ND1', 'HD1', 'CE1', 'HE1', 'NE2', 'CD2', 'HD2'],
    'H': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'ND1', 'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2'],
    'I': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG1', 'HB', 'CG2', 
          'CD1', 'HG11', 'HG12', 'HD11', 'HD12', 'HD13', 'HG21', 'HG22', 'HG23'
          ],
    'K': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'CE', 'HD1', 'HD2', 'NZ', 'HE1', 'HE2', 'HZ1', 
          'HZ2'],
    'k': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'CE', 'HD1', 'HD2', 'NZ', 'HE1', 'HE2', 'HZ1', 
          'HZ2', 'HZ3'],
    'L': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD1', 'HG', 'CD2', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23'],
    'M': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'SD', 'HG1', 'HG2', 'CE', 'HE1', 'HE2', 'HE3'],
    'N': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'ND2', 'OD1', 'HD21', 'HD22'],
    'P': ['N', 'CD', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'HG1', 'HG2', 'HD1', 'HD2'],
    'Q': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'NE2', 'OE1', 'HE21', 'HE22'],
    '4': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'NE', 'HD1', 'HD2', 'CZ', 'NH1', 'NH2', 'HH11', 
          'HH12', 'HH21', 'HH22', "HE"],
    'r': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'NE', 'HD1', 'HD2', 'CZ', 'NH1', 'NH2', 'HH11', 
          'HH12', 'HH21', 'HH22'],
    'R': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD', 'HG1', 'HG2', 'NE', 'HD1', 'HD2', 'CZ', 'HE', 'NH1', 'NH2', 
          'HH12', 'HH21', 'HH22'],
    'S': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'OG', 'HB1', 'HB2', 
          'HG'],
    'T': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'OG1', 'HB', 'CG2', 
          'HG1', 'HG21', 'HG22', 'HG23'],
    'V': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG1', 'HB', 'CG2', 
          'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23'],
    'W': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD2', 'CE3', 'HE3', 'CZ3', 'HZ3', 'CH2', 'HH2', 'CZ2', 'HZ2', 'CE2', 
          'NE1', 'HE1', 'CD1', 'HD1'],
    'Y': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD2', 'HD2', 'CE2', 'HE2', 'CZ', 'OH', 'CE1', 'HE1', 'CD1', 'HD1', 
          'HH'],
    'y': ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 'HB2', 
          'CD2', 'HD2', 'CE2', 'HE2', 'CZ', 'OH', 'CE1', 'HE1', 'CD1', 'HD1'], 
    'ACE': ['CH3', 'C', 'O', 'HH31', 'HH32', 'HH33'],
    'NME': ['CH3', 'N', 'H', 'HH31', 'HH32', 'HH33'],
    'NHE': ['N', 'HN1', 'HN2'],
    'NH2': ['N', 'HN1', 'HN2'],
}
AtomNameRosetta = {k: [x for x in AtomName[k]] for k in AtomName.keys()}
AtomNameRosetta['k'] = ['N', 'H', 'CA', 'C', 'HA', 'CB', 'O', 'CG', 'HB1', 
                        'HB2', 'CD', 'HG1', 'HG2', 'CE', 'HD1', 'HD2', 'NZ', 
                        'HE1', 'HE2', '1HZ', '2HZ', '3HZ']
Elements = {
    'A': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'H', 'H', 'H'],
    'C': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'S', 'H', 'H', 
          'H'],
    'D': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'O', 'O', 'H'],
    'd': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'O', 'O'],
    'E': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'O', 'O', 'H'],
    'e': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'O', 'O'],
    'F': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H'],
    'G': ['N', 'H', 'C', 'C', 'H', 'H', 'O'],
    '6': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'N', 'H', 'C', 'H', 'N', 'H', 'C', 'H'],
    'h': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'N', 'H', 'C', 'H', 'N', 'C', 'H'],
    'H': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'N', 'C', 'H', 'N', 'H', 'C', 'H'],
    'I': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'C', 
          'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
    'K': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'C', 'H', 'H', 'N', 'H', 'H', 'H', 
          'H'],
    'k': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'C', 'H', 'H', 'N', 'H', 'H', 'H', 
          'H', 'H'],
    'L': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    'M': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'S', 'H', 'H', 'C', 'H', 'H', 'H'],
    'N': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'N', 'O', 'H', 'H'],
    'P': ['N', 'C', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'H', 'H', 'H', 'H'],
    'Q': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'N', 'O', 'H', 'H'],
    '4': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'N', 'H', 'H', 'C', 'N', 'N', 'H', 
          'H', 'H', 'H', 'H'],
    'r': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'H', 'N', 'H', 'H', 'C', 'N', 'N', 'H', 
          'H', 'H', 'H'],
    'R': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H',
          'C', 'H', 'H', 'N', 'H', 'H', 'C', 'H', 'N', 'N', 
          'H', 'H', 'H'],
    'S': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'O', 'H', 'H', 
          'H'],
    'T': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'O', 'H', 'C', 
          'H', 'H', 'H', 'H'],
    'V': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'C', 
          'H', 'H', 'H', 'H', 'H', 'H'],
    'W': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'C', 'H', 'C', 'H', 'C', 'H', 'C', 'H', 'C', 
          'N', 'H', 'C', 'H'],
    'Y': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'C', 'H', 'C', 'O', 'C', 'H', 'C', 'H', 
          'H'],
    'y': ['N', 'H', 'C', 'C', 'H', 'C', 'O', 'C', 'H', 'H', 
          'C', 'H', 'C', 'H', 'C', 'O', 'C', 'H', 'C', 'H'], 
    'ACE': ['C', 'C', 'O', 'H', 'H', 'H'],
    'NME': ['C', 'N', 'H', 'H', 'H', 'H'],
    'NHE': ['N', 'H', 'H'],
    'NH2': ['N', 'H', 'H'],
}
# 2 atoms means center of those 2 atoms
sidechain_dihedrals = {
    'A': [],
    'C': [['SG', 'CB', 'CA', 'C'], ['HG', 'SG', 'CB', 'CA']],
    'D': [['CG', 'CB', 'CA', 'C'], ['OD1', 'CG', 'CB', 'CA'], 
          ['HD1', 'OD1', 'CG', 'OD2']],
    'd': [['CG', 'CB', 'CA', 'C'], ['OD1', 'CG', 'CB', 'CA'], 
          ['OD2', 'OD1', 'CG', 'OD2']],
    'E': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['OE2', 'CD', 'CG', 'CB'], ['HE2', 'OE2', 'CD', 'OE1']],
    'e': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['OE2', 'CD', 'CG', 'CB'], ['OE1', 'OE2', 'CD', 'OE1']],
    'F': [['CG', 'CB', 'CA', 'C'], ['CD1', 'CG', 'CB', 'CA']],
    'G': [],
    'H': [['CG', 'CB', 'CA', 'C'], ['ND1', 'CG', 'CB', 'CA']],
    'h': [['CG', 'CB', 'CA', 'C'], ['ND1', 'CG', 'CB', 'CA']],
    '6': [['CG', 'CB', 'CA', 'C'], ['ND1', 'CG', 'CB', 'CA']],
    'I': [['CG1', 'CB', 'CA', 'C'], ['CD1', 'CG1', 'CB', 'CA']],
    'K': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['CE', 'CD', 'CG', 'CB'], ['NZ', 'CE', 'CD', 'CG'],
          [['HZ1', 'HZ2'], 'NZ', 'CE', 'CD']],
    'k': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['CE', 'CD', 'CG', 'CB'], ['NZ', 'CE', 'CD', 'CG'],
          ['HZ1', 'NZ', 'CE', 'CD']],
    'L': [['CG', 'CB', 'CA', 'C'], ['CD1', 'CG', 'CB', 'CA']],
    'M': [['CG', 'CB', 'CA', 'C'], ['SD', 'CG', 'CB', 'CA'], 
          ['CE', 'SD', 'CG', 'CB']],
    'N': [['CG', 'CB', 'CA', 'C'], ['ND2', 'CG', 'CB', 'CA']],
    'P': [['CG', 'CB', 'CA', 'C']],
    'Q': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['NE2', 'CD', 'CG', 'CB']],
    'r': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['NE', 'CD', 'CG', 'CB'], ['CZ', 'NE', 'CD', 'CG']],
    'R': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['NE', 'CD', 'CG', 'CB'], ['CZ', 'NE', 'CD', 'CG'], 
          ['NH1', 'CZ', 'NE', 'CD']],
    '4': [['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
          ['NE', 'CD', 'CG', 'CB'], ['CZ', 'NE', 'CD', 'CG']],
    'S': [['OG', 'CB', 'CA', 'C'], ['HG', 'OG', 'CB', 'CA']],
    'T': [['OG1', 'CB', 'CA', 'C'], ['HG1', 'OG1', 'CB', 'CA']],
    'V': [['HB', 'CB', 'CA', 'C']],
    'W': [['CG', 'CB', 'CA', 'C'], ['CD2', 'CG', 'CB', 'CA']],
    'Y': [['CG', 'CB', 'CA', 'C'], ['CD2', 'CG', 'CB', 'CA'], 
          ['HH', 'OH', 'CB', 'CA']],
    'y': [['CG', 'CB', 'CA', 'C'], ['CD2', 'CG', 'CB', 'CA'], 
          ['CA', 'OH', 'CB', 'CA']],
}
sidechain_dihedrals_rosetta = {
    k: deepcopy(sidechain_dihedrals[k]) for k in sidechain_dihedrals.keys()}
sidechain_dihedrals_rosetta['k'] = [
    ['CG', 'CB', 'CA', 'C'], ['CD', 'CG', 'CB', 'CA'], 
    ['CE', 'CD', 'CG', 'CB'], ['NZ', 'CE', 'CD', 'CG'], 
    [['1HZ', '2HZ'], 'NZ', 'CE', 'CD']]


def rotate(r, n, phi):
    """Rotate of r about n by phi rad closewisely, n normalized"""
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)
    nr = n[0]*r[0] + n[1]*r[1] + n[2]*r[2]
    x = r[0]*cosphi + n[0]*nr*(1-cosphi) + (r[1]*n[2]-r[2]*n[1])*sinphi
    y = r[1]*cosphi + n[1]*nr*(1-cosphi) + (r[2]*n[0]-r[0]*n[2])*sinphi
    z = r[2]*cosphi + n[2]*nr*(1-cosphi) + (r[0]*n[1]-r[1]*n[0])*sinphi
    return [x, y, z]

    
def xyz2pdb(seq, xyzfile, pdbfile=None, nt="ace", ct="nme"):
    """Convert xyz file to pdb format
    
    Convert xyz file to pdb format. The order of atoms in the input xyz file is
    fixed as follows:
        First comes N of the 1st non-cap residue;
        second comes C of ACE or H2 of the 1st non-cap residue;
        third comes H1 or CD for PRO of the 1st non-cap residue.
        After the fisrt 3 atoms come main-chain atoms arranged as:
            CA of the (i)th residue,
            C of the (i)th residue,
            HA of the (i)th residue,
            CB or HA2 for GLY of the (i)th residue,
            N of the (i+1)th residue or OXT of the final residue,
            O of the (i)th residue,
            H or CD for PRO of the (i+1)th residue, if any.
        Then come side-chain atoms from the last residue to the first, arranged
        as in AtomName.
        Finally come atoms in C-cap and N-cap, arranged as:
            CH3, HH31, HH32, HH33 for NME;
            CH3, O, HH31, HH32, HH33 for ACE;
            H3 of the 1st protonated residue;
    """
    assert nt in ['ace', 'pro']
    assert ct in ['nme', 'dep']
    with open(xyzfile, 'r') as f:
        xyz = []
        for line in f.readlines()[2:]:
            xyz.append([float(x) for x in line.strip().split()[1:]])
    cap_na = 0 # the number of cap atoms
    nt_na = 0 # the number of N cap atoms
    if nt == "ace":
        cap_na += 5 # CH3, O, HH31, HH32, HH33
        nt_na += 5
    elif nt == "pro":
        cap_na += 1 # H3
        nt_na += 1
    if ct == "nme":
        cap_na += 4
    pdblines = []
    if nt == "ace":
        for i, j in enumerate([-5, 1, -4, -3, -2, -1], 0):
            # CH3, C, O, HH31, HH32, HH33
            pdblines.append(
                'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
                AtomName['ACE'][i], 'ACE', 0, xyz[j][0], xyz[j][1], xyz[j][2], 
                Elements['ACE'][i]
            ))
    elif nt == "pro":
        pdblines.append(
            'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
            "H2", one2three[seq[0]], 1, xyz[1][0], xyz[1][1], xyz[1][2], 'H'
        ))
        pdblines.append(
            'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
            "H3", one2three[seq[0]], 1, xyz[-1][0], xyz[-1][1], xyz[-1][2], 'H'
        ))
    elif nt == "neu":
        pdblines.append(
            'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
            "H2", one2three[seq[0]], 1, xyz[1][0], xyz[1][1], xyz[1][2], 'H'
        ))
    siden = 0 # the number of side-chain atoms
    for n in range(len(seq)):
        siden += len(AtomName[seq[n]]) - 7 # 7 main-chain atoms each residue
        mj = [x + 7 * n for x in [0, 2, 3, 4, 5, 6, 8]]
        sj = [] # index of side-chain atom in xyz
        for x in range(len(AtomName[seq[n]]) - 7):
            sj.append(len(xyz) - cap_na - siden + x)
        for i, j in enumerate(mj + sj, 0):
            # i: index in AtomName
            # j: index in xyz
            # xyz                     pdb
            # N () H C C H C N O H -> N H C C H C O
            # 0    2 3 4 5 6   8  
            #        C   H C              C   H C
            #        A   A B              A   A B
            if n == 0 and nt == "pro" and i == 1 and seq[0] != 'P':
                an = "H1"
            elif n == 0 and nt == "neu" and i == 1 and seq[0] != 'P':
                an = "H1"
            else:
                an = AtomName[seq[n]][i]
            pdblines.append(
                'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
                    an, one2three[seq[n]], n+1, xyz[j][0], xyz[j][1], 
                    xyz[j][2], Elements[seq[n]][i]
            ))
    if ct == "nme":
        nme_j = [
            -4-nt_na, 7*len(seq), 7*len(seq)+2, -1-nt_na, -2-nt_na, -3-nt_na]
        for i, j in enumerate(nme_j, 0):
            # CH3, N, H, HH31, HH32, HH33
            pdblines.append(
                'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
                    AtomName['NME'][i], 'NME', len(seq)+1, xyz[j][0], 
                    xyz[j][1], xyz[j][2], Elements['NME'][i]
            ))
    elif ct == "dep":
        pdblines.append(
            'ATOM      1%5s%4s A%4d    %8.3f%8.3f%8.3f  1.00  0.00%12s' % (
                "OXT", one2three[seq[-1]], len(seq), xyz[7*len(seq)][0], 
                xyz[7*len(seq)][1], xyz[7*len(seq)][2], 'O'
        ))
    if pdbfile is None:
        outfile = xyzfile[:-4]+'.pdb'
    else:
        outfile = pdbfile
    with open(outfile, 'w') as f:
        for line in pdblines:
            f.write('%s\n' % line)
    return


def xyz_dftbin(fn):
    with open(fn, 'r') as f:
        ilines = f.readlines()
    na = int(ilines[0].strip())
    olines = ['%d C' % na, 'H C N O']
    has_s = False
    for i in range(2, len(ilines)):
        line = ilines[i].strip().split()
        if line[0] == 'H':
            olines.append("%d 1 %.8f %.8f %.8f" % (
                i - 1, float(line[1]), float(line[2]), float(line[3])))
        elif line[0] == 'C':
            olines.append("%d 2 %.8f %.8f %.8f" % (
                i - 1, float(line[1]), float(line[2]), float(line[3])))
        elif line[0] == 'N':
            olines.append("%d 3 %.8f %.8f %.8f" % (
                i - 1, float(line[1]), float(line[2]), float(line[3])))
        elif line[0] == 'O':
            olines.append("%d 4 %.8f %.8f %.8f" % (
                i - 1, float(line[1]), float(line[2]), float(line[3])))
        elif line[0] == 'S':
            olines.append("%d 5 %.8f %.8f %.8f" % (
                i - 1, float(line[1]), float(line[2]), float(line[3])))
            has_s = True
        else:
            return 1
    if has_s:
        olines[1] += ' S'
    with open(fn[:-4]+'.txt', 'w') as f:
        for x in olines:
            f.write(x+'\n')
    return 0


class GetDihs:
    """
    GetDihs gets dihedral angles given xyz or pdb
    
    Attributes:
        dihs_list (list): ['M1', 'M2', ... 'M0', 's??', ...]
        
    Methods:
        calc_pdb(pdb): calculate dihedral angles of input pdb
    """
    def __init__(self, seq='', nt="ace", ct="nme", rosetta=False):
        self.seq = seq
        self.nt = nt
        self.ct = ct
        assert len(seq) > 0
        self.dihs_list = [] # M1 M2 ... M0 s...
        self.dihs_dict = {}
        if nt in ["neu", "pro"] or rosetta:
            has_nt = 0
        else:
            has_nt = 1
        self.h2 = '2H' if rosetta else 'H2'
        self.h1 = '1H' if rosetta else 'H1'
        for i in range(len(seq)):
            j = i + has_nt # index of residue include cap if any
            # PHI
            self.dihs_list.append('M%d' % (2*i+1))
            if i == 0 and nt == "neu":
                self.dihs_dict['M1'] = [
                    '0 C', '0 CA', '0 N', [f'0 {self.h1}', f'0 {self.h2}']]
            elif i == 0 and nt == "pro":
                self.dihs_dict['M1'] = [
                    '0 C', '0 CA', '0 N', f'0 {self.h2}']
            elif i == 0 and rosetta: # ACE C -> CO
                self.dihs_dict['M%d' % (2*i+1)] = [
                    '0 C', '0 CA', '0 N', '0 CO']
            else:
                self.dihs_dict['M%d' % (2*i+1)] = [
                    f'{j} C', f'{j} CA', f'{j} N', f'{j-1} C']
            # PSI
            self.dihs_list.append('M%d' % (2*i+2))
            if i == len(seq) - 1 and ct in ['dep', "neu"]:
                self.dihs_dict['M%d' % (2*i+2)] = [
                    f'{j} OXT', f'{j} C', f'{j} CA', f'{j} N']
            elif i == len(seq) - 1 and rosetta: # NME NHE
                self.dihs_dict['M%d' % (2*i+2)] = [
                    f'{i} {"NT" if ct == "nh2" else "NM"}', 
                    f'{j} C', f'{j} CA', f'{j} N']
            else:
                self.dihs_dict['M%d' % (2*i+2)] = [
                    f'{j+1} N', f'{j} C', f'{j} CA', f'{j} N']
        # M0
        self.dihs_list.append('M0')
        j = len(seq) + has_nt - 1
        if ct == 'nh2':
            self.dihs_dict['M0'] = [
                f'{j} O',f'{j} NT' if rosetta else f'{j+1} N', 
                f'{j} C', f'{j} O']
        elif ct == 'dep':
            self.dihs_dict['M0'] = [f'{j} O', f'{j} OXT', f'{j} C', f'{j} O']
        elif ct == 'neu':
            self.dihs_dict['M0'] = [f'{j} HXT', f'{j} OXT', f'{j} C', f'{j} O']
        elif rosetta: # NME N -> NM, CH3 -> CN
            self.dihs_dict['M0'] = [f'{j} CN', f'{j} NM', f'{j} C', f'{j} O']
        else:
            self.dihs_dict['M0'] = [f'{j+1} H', f'{j+1} N', f'{j} C', f'{j} O']
        # list of atoms
        # first three atoms
        if seq[0] == 'P':
            if nt == "ace" and rosetta:
                self.atoms_list = ['0 N', '0 CO', '0 CD']
            elif nt == 'ace':
                self.atoms_list = ['1 N', '0 C', '1 CD']
            elif nt in ["neu", "pro"]:
                self.atoms_list = ['0 N', f'0 {self.h2}', '0 CD']
        else:
            if nt == "ace" and rosetta:
                self.atoms_list = ['0 N', '0 CO', '0 H']
            elif nt == 'ace':
                self.atoms_list = ['1 N', '0 C', '1 H']
            elif nt in ["neu", "pro"]:
                self.atoms_list = ['0 N', f'0 {self.h2}', f'0 {self.h1}']
        # mainchain C C H C(H) N O H
        for _i in range(len(seq)):
            i = _i + has_nt # index of residue include cap if any
            # C C
            self.atoms_list.append(f'{i} CA')
            self.atoms_list.append(f'{i} C')
            # H C(H) 
            if seq[_i] == 'G':
                self.atoms_list.append(f'{i} HA2')
                self.atoms_list.append(f'{i} HA3')
            else:
                self.atoms_list.append(f'{i} HA')
                self.atoms_list.append(f'{i} CB')
            # N
            if _i == len(seq) - 1 and ct in ["dep", "neu"]:
                self.atoms_list.append(f'{i} OXT')
            elif rosetta and _i == len(seq) - 1:
                self.atoms_list.append(f'{i} {"NT" if ct == "nh2" else "NM"}')
            else:
                self.atoms_list.append(f'{i+1} N')
            # O
            self.atoms_list.append(f'{i} O')
            # H
            if _i == len(seq) - 1 and ct == "dep":
                pass
            elif _i == len(seq) - 1 and ct == "neu":
                self.atoms_list.append(f'{i} HXT')
            elif _i == len(seq) - 1 and ct == "nh2":
                self.atoms_list.append(f'{i} 1HN' if rosetta else f'{i+1} HN1')
            elif _i == len(seq) - 1 and rosetta:
                self.atoms_list.append(f'{i} HM')
            elif _i + 1 < len(seq) and seq[_i+1] == 'P':
                self.atoms_list.append(f'{i+1} CD')
            else:
                self.atoms_list.append(f'{i+1} H')
        # sidechain
        for _i in range(len(seq)-1, -1, -1):
            i = _i + has_nt # index of residue include cap if any
            newAtomName = AtomNameRosetta if rosetta else AtomName
            # list of atoms
            for a in newAtomName[seq[_i]][7:]:
                self.atoms_list.append(f'{i} {a}')
            if rosetta:
                new_sidechain_dihedrals = sidechain_dihedrals_rosetta
            else:
                new_sidechain_dihedrals = sidechain_dihedrals
            for j in range(len(new_sidechain_dihedrals[seq[_i]])):
                # count from 1, exclude cap if any
                self.dihs_list.append('s%d%d' % (_i+1, j+1))
                self.dihs_dict['s%d%d' % (_i+1, j+1)] = [
                    f'{i} {y}' if isinstance(y, str) else 
                    [f'{i} {z}' for z in y]
                    for y in new_sidechain_dihedrals[seq[_i]][j]]
            if seq[_i] == 'P':
                # omega before
                self.dihs_list.append('s%d2' % (_i+1))
                if _i == 0 and nt == 'pro':
                    # degenerate
                    self.dihs_dict[f's{_i+1}2'] = [
                        f'{i} CD' , f'{i} N', f'{i} {self.h2}', f'{i} CD']
                elif _i == 0 and rosetta and nt == 'ace':
                    self.dihs_dict[f's{_i+1}2'] = [
                        '0 CD', '0 N', '0 CO', '0 OP1']
                else:
                    self.dihs_dict['s%d2' % (_i+1)] = [
                        f'{i} CD', f'{i} N', f'{i-1} C', f'{i-1} O']
                # omega after
                self.dihs_list.append('s%d3' % (_i+1))
                if _i+1 < len(seq) and seq[_i+1] == 'P':
                    self.dihs_dict['s%d3' % (_i+1)] = [
                        f'{i+1} CD', f'{i+1} N', f'{i} C', f'{i} O']
                elif _i == len(seq) - 1 and ct == 'nh2':
                    # degenerate
                    self.dihs_dict['s%d3' % (_i+1)] = [
                        f'{i} O',f'{i} NT' if rosetta else 
                        f'{i+1} N', f'{i} C', f'{i} O']
                elif _i == len(seq) - 1 and ct == 'dep':
                    # degenerate
                    self.dihs_dict['s%d3' % (_i+1)] = [
                        f'{i} O', f'{i} OXT', f'{i} C', f'{i} O']
                elif _i == len(seq) - 1 and rosetta:
                    self.dihs_dict['s%d3' % (_i+1)] = [
                        f'{i} HM', f'{i} NM', f'{i} C', f'{i} O']
                else:
                    self.dihs_dict['s%d3' % (_i+1)] = [
                        f'{i+1} H', f'{i+1} N', f'{i} C', f'{i} O']
        if ct == "nme":
            self.atoms_list.append(
                f'{len(seq)-1} CN' if rosetta else '{len(seq)+has_nt} CH3')
            self.atoms_list.append(
                f'{len(seq)+has_nt-1} {"1HN" if rosetta else "H1"}')
            self.atoms_list.append(
                f'{len(seq)+has_nt-1} {"2HN" if rosetta else "H2"}')
            self.atoms_list.append(
                f'{len(seq)+has_nt-1} {"3HN" if rosetta else "H3"}')
        elif ct == "nh2":
            self.atoms_list.append(
                f'{len(seq)-1} 2HN' if rosetta else f'{len(seq)+has_nt} HN2')
        if nt == "ace":
            self.atoms_list.append(f'0 {"CP2" if rosetta else "CH3"}')
            self.atoms_list.append(f'0 {"OP1" if rosetta else "O"}')
            self.atoms_list.append(f'0 {"1HP2" if rosetta else "H1"}')
            self.atoms_list.append(f'0 {"2HP2" if rosetta else "H2"}')
            self.atoms_list.append(f'0 {"3HP2" if rosetta else "H3"}')
        elif nt == "pro":
            self.atoms_list.append('0 H3')
        # atoms' rank in list
        atoms_dict = {
            self.atoms_list[i]: i for i in range(len(self.atoms_list))}
        self.dih_atom_list = []
        for c in self.dihs_list:
            self.dih_atom_list.append([])
            for k in self.dihs_dict[c]:
                if isinstance(k, str):
                    if k not in atoms_dict.keys():
                        print(c, k, seq, atoms_dict.keys())
                        raise KeyError
                    self.dih_atom_list[-1].append(atoms_dict[k])
                else:
                    self.dih_atom_list[-1].append([atoms_dict[x] for x in k])
    
    def calc_pdb(self, pdb):
        """calculate dihedral angles of input pdb
        
        calculate dihedral angles of input pdb, an instance of ReadPdbAtom
        """
        dihs = []
        for dn in self.dihs_list:
            a, flag1, flag2 = [], False, False
            for i_an in self.dihs_dict[dn]:
                if isinstance(i_an, str):
                    i, an = i_an.split()
                    i = int(i)
                    try:
                        a.append([
                            pdb[i][an]['x'], pdb[i][an]['y'], pdb[i][an]['z']])
                    except KeyError:
                        print(dn, i, an, pdb[i].keys())
                        raise
                    except IndexError:
                        print(i, an, len(pdb), pdb[-1].keys())
                        raise
                else:
                    xyz = []
                    for _i_an in i_an:
                        i, an = _i_an.split()
                        i = int(i)
                        xyz.append([
                            pdb[i][an]['x'], pdb[i][an]['y'], pdb[i][an]['z']])
                    a.append(np.mean(xyz, axis=-2))
            a = np.array(a)
            b = [a[0]-a[1], a[2]-a[1], a[3]-a[2]]
            b[1] /= np.linalg.norm(b[1])
            v = b[0] - np.dot(b[0], b[1]) * b[1]
            w = b[2] - np.dot(b[2], b[1]) * b[1]
            if dn == 'M1' and self.nt == 'pro':
                flag2 = True
            elif dn == f'M{len(self.seq)*2}' and self.ct == 'dep':
                flag1 = True
            elif not dn.startswith('s'):
                pass
            elif self.seq[int(dn[1:-1])-1] == 'd' and int(dn[-1]) == 2:
                flag1 = True
            elif self.seq[int(dn[1:-1])-1] == 'e' and int(dn[-1]) == 3:
                flag1 = True
            elif self.seq[int(dn[1:-1])-1] == 'Y' and int(dn[-1]) == 2:
                flag1 = True
            elif self.seq[int(dn[1:-1])-1] == 'F' and int(dn[-1]) == 2:
                flag1 = True
            dihx = np.arctan2(np.dot(np.cross(b[1], v), w), np.dot(v, w)) * 180 / np.pi
            if flag1:
                if dihx < 0:
                    dihx += 180
            elif flag2:
                if dihx < -60:
                    dihx += 120
                elif dihx > 60:
                    dihx -= 120
            else:
                pass
            dihs.append(dihx)
        return dihs


class ReadPdbAtom(list):
    """
    https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
    """
    def __init__(self, filename):
        list.__init__(self)
        self.seq3, self.seq = [], []
        with open(filename, 'r') as f:
            pdb_lines = f.readlines()
        for lines in pdb_lines:
            line = lines.strip()
            if line[:6] != 'ATOM  ' and line[:6] != 'HETATM':
                continue
            serial = line[6:11]
            name = line[12:16]
            altloc = line[16]
            resname = line[17:20]
            chainid = line[21]
            resseq = line[22:26]
            icode = line[26]
            x = line[30:38]
            y = line[38:46]
            z = line[46:54]
            occupancy = line[54:60]
            tempfactor = line[60:66]
            element = line[76:78]
            charge = line[78:80]
            # new residue
            if self.__len__() == 0 or next(
                    iter(self.__getitem__(self.__len__() - 1).values())
                    )['resSeq'] != resseq:
                self.append({})
                self.seq3.append(resname.strip())
                if self.seq3[-1] not in ['ACE', 'NME', 'NHE', 'HOH', 'NH2']:
                    self.seq.append(three2one[self.seq3[-1]])
            if name.strip() in self.__getitem__(-1).keys():
                print('error: duplicate %s in residue %s %s %s' % (
                    name, resseq, resname, filename))
                raise BaseException
            self.__getitem__(-1)[name.strip()] = {
                'serial': serial,
                'name': name,
                'altLoc': altloc,
                'resName': resname,
                'chainID': chainid,
                'resSeq': resseq,
                'iCode': icode,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'occupancy': occupancy,
                'tempFactor': tempfactor,
                'element': element,
                'charge': charge
            }
        self.atom_list = None
    
    def rot_phi(self, i, degree):
        if self.__getitem__(i)['C']['resName'] == 'PRO':
            return
        caxyz = [self.__getitem__(i)['CA']['x'], 
                 self.__getitem__(i)['CA']['y'], 
                 self.__getitem__(i)['CA']['z']]
        nxyz = [self.__getitem__(i)['N']['x'], 
                 self.__getitem__(i)['N']['y'], 
                 self.__getitem__(i)['N']['z']]
        ca_n = math.sqrt((caxyz[0]-nxyz[0])**2
                + (caxyz[1]-nxyz[1])**2
                + (caxyz[2]-nxyz[2])**2)
        ca_nn = [(nxyz[j] - caxyz[j]) / ca_n for j in range(3)]
        for j in range(self.__len__()):
            if j >= i:
                continue
            residue = self.__getitem__(j)
            for k in residue.keys():
                x0 = self.__getitem__(j)[k]['x'] - caxyz[0]
                y0 = self.__getitem__(j)[k]['y'] - caxyz[1]
                z0 = self.__getitem__(j)[k]['z'] - caxyz[2]
                x, y, z = rotate([x0, y0, z0], ca_nn, degree * math.pi / 180)
                self.__getitem__(j)[k]['x'] = caxyz[0] + x
                self.__getitem__(j)[k]['y'] = caxyz[1] + y
                self.__getitem__(j)[k]['z'] = caxyz[2] + z
        mc = ['N', 'H', 'H1', 'H2', 'H3']
        for k in self.__getitem__(i).keys():
            if k not in mc:
                continue
            x0 = self.__getitem__(i)[k]['x'] - caxyz[0]
            y0 = self.__getitem__(i)[k]['y'] - caxyz[1]
            z0 = self.__getitem__(i)[k]['z'] - caxyz[2]
            x, y, z = rotate([x0, y0, z0], ca_nn, degree * math.pi / 180)
            self.__getitem__(i)[k]['x'] = caxyz[0] + x
            self.__getitem__(i)[k]['y'] = caxyz[1] + y
            self.__getitem__(i)[k]['z'] = caxyz[2] + z
    
    def rot_psi(self, i, degree):
        caxyz = [self.__getitem__(i)['CA']['x'], 
                 self.__getitem__(i)['CA']['y'], 
                 self.__getitem__(i)['CA']['z']]
        nxyz = [self.__getitem__(i)['N']['x'], 
                 self.__getitem__(i)['N']['y'], 
                 self.__getitem__(i)['N']['z']]
        ca_n = math.sqrt((caxyz[0]-nxyz[0])**2
                + (caxyz[1]-nxyz[1])**2
                + (caxyz[2]-nxyz[2])**2)
        ca_nn = [(nxyz[j] - caxyz[j]) / ca_n for j in range(3)]
        for j in range(self.__len__()):
            if j <= i:
                continue
            residue = self.__getitem__(j)
            for k in residue.keys():
                x0 = self.__getitem__(j)[k]['x'] - caxyz[0]
                y0 = self.__getitem__(j)[k]['y'] - caxyz[1]
                z0 = self.__getitem__(j)[k]['z'] - caxyz[2]
                x, y, z = rotate([x0, y0, z0], ca_nn, degree * math.pi / 180)
                self.__getitem__(j)[k]['x'] = caxyz[0] + x
                self.__getitem__(j)[k]['y'] = caxyz[1] + y
                self.__getitem__(j)[k]['z'] = caxyz[2] + z
        mc = ['C', 'O', 'OXT', 'HXT']
        for k in self.__getitem__(i).keys():
            if k not in mc:
                continue
            x0 = self.__getitem__(i)[k]['x'] - caxyz[0]
            y0 = self.__getitem__(i)[k]['y'] - caxyz[1]
            z0 = self.__getitem__(i)[k]['z'] - caxyz[2]
            x, y, z = rotate([x0, y0, z0], ca_nn, degree * math.pi / 180)
            self.__getitem__(i)[k]['x'] = caxyz[0] + x
            self.__getitem__(i)[k]['y'] = caxyz[1] + y
            self.__getitem__(i)[k]['z'] = caxyz[2] + z
    
    def rot_sc(self, i, degree):
        if self.__getitem__(i)['C']['resName'] == 'GLY':
            return
        if self.__getitem__(i)['C']['resName'] == 'PRO':
            return
        caxyz = [self.__getitem__(i)['CA']['x'], 
                 self.__getitem__(i)['CA']['y'], 
                 self.__getitem__(i)['CA']['z']]
        cbxyz = [self.__getitem__(i)['CB']['x'], 
                 self.__getitem__(i)['CB']['y'], 
                 self.__getitem__(i)['CB']['z']]
        ca_cb = math.sqrt((caxyz[0]-cbxyz[0])**2
                + (caxyz[1]-cbxyz[1])**2
                + (caxyz[2]-cbxyz[2])**2)
        ca_cbn = [(cbxyz[j] - caxyz[j]) / ca_cb for j in range(3)]
        mc = ['CA', 'N', 'H', 'HA', 'C', 'O', 'CB', 'H1', 'H2', 'H3', 'OXT', 'HXT']
        for k in self.__getitem__(i).keys():
            if k in mc:
                continue
            x0 = self.__getitem__(i)[k]['x'] - cbxyz[0]
            y0 = self.__getitem__(i)[k]['y'] - cbxyz[1]
            z0 = self.__getitem__(i)[k]['z'] - cbxyz[2]
            x, y, z = rotate([x0, y0, z0], ca_cbn, degree * math.pi / 180)
            self.__getitem__(i)[k]['x'] = cbxyz[0] + x
            self.__getitem__(i)[k]['y'] = cbxyz[1] + y
            self.__getitem__(i)[k]['z'] = cbxyz[2] + z
    
    def nt_pro2cap(self):
        if 'C' not in self.__getitem__(0).keys():
            return
        if self.__getitem__(0)['C']['resName'] == 'ACE':
            return
        nxyz = [self.__getitem__(0)['N']['x'], 
                self.__getitem__(0)['N']['y'], 
                self.__getitem__(0)['N']['z']]
        caxyz = [self.__getitem__(0)['CA']['x'], 
                 self.__getitem__(0)['CA']['y'], 
                 self.__getitem__(0)['CA']['z']]
        h1xyz = [self.__getitem__(0)['H1']['x'], 
                 self.__getitem__(0)['H1']['y'], 
                 self.__getitem__(0)['H1']['z']]
        self.__getitem__(0)['H1']['name'] = '   H'
        del self.__getitem__(0)['H2']
        del self.__getitem__(0)['H3']
        ca_n = math.sqrt((caxyz[0]-nxyz[0])**2
                + (caxyz[1]-nxyz[1])**2
                + (caxyz[2]-nxyz[2])**2)
        ca_nn = [(nxyz[i] - caxyz[i]) / ca_n for i in range(3)]
        h1_n = math.sqrt((h1xyz[0]-nxyz[0])**2
                + (h1xyz[1]-nxyz[1])**2
                + (h1xyz[2]-nxyz[2])**2)
        h1_nn = [(nxyz[i] - h1xyz[i]) / h1_n for i in range(3)]
        n_c = math.sqrt((caxyz[0]+h1xyz[0]-2*nxyz[0])**2
               + (caxyz[1]+h1xyz[1]-2*nxyz[1])**2
               + (caxyz[2]+h1xyz[2]-2*nxyz[2])**2)
        n_cn = [-(caxyz[i] + h1xyz[i] - 2*nxyz[i]) / n_c for i in range(3)]
        cxyz = [nxyz[i] + n_cn[i] * 1.34537 for i in range(3)]
        oxyz = [cxyz[i] + h1_nn[i] * 1.21961 for i in range(3)]
        ch3xyz = [cxyz[i] + ca_nn[i] * 1.52290 for i in range(3)]
        s6 = math.sqrt(6)
        ch3_hh31n = [h1_nn[i]*4*s6/9+ca_nn[i]*(2*s6+3)/9 for i in range(3)]
        hh31xyz = [ch3xyz[i] + ch3_hh31n[i] * 1.08195 for i in range(3)]
        ch3_hh32n = [x for x in rotate(ch3_hh31n, ca_nn, math.pi * 2 / 3)]
        ch3_hh33n = [x for x in rotate(ch3_hh31n, ca_nn, math.pi * 4 / 3)]
        hh32xyz = [ch3xyz[i] + ch3_hh32n[i] * 1.08195 for i in range(3)]
        hh33xyz = [ch3xyz[i] + ch3_hh33n[i] * 1.08195 for i in range(3)]
        ace = {'C':self.__getitem__(0)['C'].copy(),
               'O':self.__getitem__(0)['O'].copy(),
               'CH3':self.__getitem__(0)['C'].copy(),
               'HH31':self.__getitem__(0)['HA'].copy(),
               'HH32':self.__getitem__(0)['HA'].copy(),
               'HH33':self.__getitem__(0)['HA'].copy()}
        for k in ace.keys():
            ace[k]['resName'] = 'ACE'
            ace[k]['name'] = k.rjust(4)
        ace['C']['x'] = cxyz[0]
        ace['C']['y'] = cxyz[1]
        ace['C']['z'] = cxyz[2]
        ace['O']['x'] = oxyz[0]
        ace['O']['y'] = oxyz[1]
        ace['O']['z'] = oxyz[2]
        ace['CH3']['x'] = ch3xyz[0]
        ace['CH3']['y'] = ch3xyz[1]
        ace['CH3']['z'] = ch3xyz[2]
        ace['HH31']['x'] = hh31xyz[0]
        ace['HH31']['y'] = hh31xyz[1]
        ace['HH31']['z'] = hh31xyz[2]
        ace['HH32']['x'] = hh32xyz[0]
        ace['HH32']['y'] = hh32xyz[1]
        ace['HH32']['z'] = hh32xyz[2]
        ace['HH33']['x'] = hh33xyz[0]
        ace['HH33']['y'] = hh33xyz[1]
        ace['HH33']['z'] = hh33xyz[2]
        self.insert(0, ace)
   
    def ct_dep2cap(self):
        if 'N' not in self.__getitem__(-1).keys():
            return
        if self.__getitem__(-1)['N']['resName'] == 'NME':
            return
        cxyz = [self.__getitem__(-1)['C']['x'], 
                self.__getitem__(-1)['C']['y'], 
                self.__getitem__(-1)['C']['z']]
        caxyz = [self.__getitem__(-1)['CA']['x'], 
                 self.__getitem__(-1)['CA']['y'], 
                 self.__getitem__(-1)['CA']['z']]
        oxyz = [self.__getitem__(-1)['O']['x'], 
                self.__getitem__(-1)['O']['y'], 
                self.__getitem__(-1)['O']['z']]
        del self.__getitem__(-1)['OXT']
        ca_c = math.sqrt((caxyz[0]-cxyz[0])**2
                + (caxyz[1]-cxyz[1])**2
                + (caxyz[2]-cxyz[2])**2)
        ca_cn = [(cxyz[i] - caxyz[i]) / ca_c for i in range(3)]
        o_c = math.sqrt((oxyz[0]-cxyz[0])**2
                + (oxyz[1]-cxyz[1])**2
                + (oxyz[2]-cxyz[2])**2)
        o_cn = [(cxyz[i] - oxyz[i]) / o_c for i in range(3)]
        c_n = math.sqrt((caxyz[0]+oxyz[0]-2*cxyz[0])**2
               + (caxyz[1]+oxyz[1]-2*cxyz[1])**2
               + (caxyz[2]+oxyz[2]-2*cxyz[2])**2)
        c_nn = [-(caxyz[i] + oxyz[i] - 2*cxyz[i]) / c_n for i in range(3)]
        nxyz = [cxyz[i] + c_nn[i] * 1.34537 for i in range(3)]
        hxyz = [nxyz[i] + o_cn[i] * 1.00881 for i in range(3)]
        ch3xyz = [nxyz[i] + ca_cn[i] * 1.44577 for i in range(3)]
        s6 = math.sqrt(6)
        ch3_hh31n = [o_cn[i]*4*s6/9+ca_cn[i]*(2*s6+3)/9 for i in range(3)]
        hh31xyz = [ch3xyz[i] + ch3_hh31n[i] * 1.08195 for i in range(3)]
        ch3_hh32n = [x for x in rotate(ch3_hh31n, ca_cn, math.pi * 2 / 3)]
        ch3_hh33n = [x for x in rotate(ch3_hh31n, ca_cn, math.pi * 4 / 3)]
        hh32xyz = [ch3xyz[i] + ch3_hh32n[i] * 1.08195 for i in range(3)]
        hh33xyz = [ch3xyz[i] + ch3_hh33n[i] * 1.08195 for i in range(3)]
        nme = {'N':self.__getitem__(-1)['N'].copy(),
               'H':self.__getitem__(-1)['H'].copy(),
               'CH3':self.__getitem__(-1)['C'].copy(),
               'HH31':self.__getitem__(-1)['HA'].copy(),
               'HH32':self.__getitem__(-1)['HA'].copy(),
               'HH33':self.__getitem__(-1)['HA'].copy()}
        for k in nme.keys():
            nme[k]['resName'] = 'NME'
            nme[k]['name'] = k.rjust(4)
        nme['N']['x'] = nxyz[0]
        nme['N']['y'] = nxyz[1]
        nme['N']['z'] = nxyz[2]
        nme['H']['x'] = hxyz[0]
        nme['H']['y'] = hxyz[1]
        nme['H']['z'] = hxyz[2]
        nme['CH3']['x'] = ch3xyz[0]
        nme['CH3']['y'] = ch3xyz[1]
        nme['CH3']['z'] = ch3xyz[2]
        nme['HH31']['x'] = hh31xyz[0]
        nme['HH31']['y'] = hh31xyz[1]
        nme['HH31']['z'] = hh31xyz[2]
        nme['HH32']['x'] = hh32xyz[0]
        nme['HH32']['y'] = hh32xyz[1]
        nme['HH32']['z'] = hh32xyz[2]
        nme['HH33']['x'] = hh33xyz[0]
        nme['HH33']['y'] = hh33xyz[1]
        nme['HH33']['z'] = hh33xyz[2]
        self.append(nme)
   

    def nt_cap2pro(self):
        if 'CH3' not in self.__getitem__(0).keys():
            return
        if self.__getitem__(0)['CH3']['resName'] != 'ACE':
            return
        nxyz = [self.__getitem__(1)['N']['x'], 
                self.__getitem__(1)['N']['y'], 
                self.__getitem__(1)['N']['z']]
        caxyz = [self.__getitem__(1)['CA']['x'], 
                 self.__getitem__(1)['CA']['y'], 
                 self.__getitem__(1)['CA']['z']]
        if 'PRO' in self.__getitem__(1)['N']['resName']:
            hxyz = [self.__getitem__(1)['CD']['x'], 
                    self.__getitem__(1)['CD']['y'], 
                    self.__getitem__(1)['CD']['z']]
        else:
            hxyz = [self.__getitem__(1)['H']['x'], 
                    self.__getitem__(1)['H']['y'], 
                    self.__getitem__(1)['H']['z']]
        ca_n = ((caxyz[0]-nxyz[0])**2
                + (caxyz[1]-nxyz[1])**2
                + (caxyz[2]-nxyz[2])**2)**0.5
        ca_nn = [(nxyz[i] - caxyz[i]) / ca_n for i in range(3)]
        n_h = ((hxyz[0]-nxyz[0])**2
               + (hxyz[1]-nxyz[1])**2
               + (hxyz[2]-nxyz[2])**2)**0.5
        n_hn = [(hxyz[i] - nxyz[i]) / n_h for i in range(3)]
        n_h2v = [x * 1.008807 for x in rotate(n_hn, ca_nn, math.pi * 2 / 3)]
        n_h3v = [x * 1.008807 for x in rotate(n_hn, ca_nn, math.pi * 4 / 3)]
        h2xyz = [nxyz[i] + n_h2v[i] for i in range(3)]
        h3xyz = [nxyz[i] + n_h3v[i] for i in range(3)]
        # copy
        if 'PRO' in self.__getitem__(1)['N']['resName']:
            self.__getitem__(1)['H2'] = self.__getitem__(1)['CD'].copy()
            self.__getitem__(1)['H3'] = self.__getitem__(1)['CD'].copy()
            self.__getitem__(1)['H2']['element'] = ' H'
            self.__getitem__(1)['H3']['element'] = ' H'
        else:
            self.__getitem__(1)['H1'] = self.__getitem__(1)['H'].copy()
            self.__getitem__(1)['H1']['name'] = '  H1'
            self.__getitem__(1)['H2'] = self.__getitem__(1)['H'].copy()
            self.__getitem__(1)['H3'] = self.__getitem__(1)['H']
            del self.__getitem__(1)['H']
        self.__getitem__(1)['H2']['name'] = '  H2'
        self.__getitem__(1)['H2']['x'] = h2xyz[0]
        self.__getitem__(1)['H2']['y'] = h2xyz[1]
        self.__getitem__(1)['H2']['z'] = h2xyz[2]
        self.__getitem__(1)['H3']['name'] = '  H3'
        self.__getitem__(1)['H3']['x'] = h3xyz[0]
        self.__getitem__(1)['H3']['y'] = h3xyz[1]
        self.__getitem__(1)['H3']['z'] = h3xyz[2]
        self.pop(0) # remove original ACE
        return
    
    def nt_cap2neu(self):
        if 'CH3' not in self.__getitem__(0).keys():
            return
        if self.__getitem__(0)['CH3']['resName'] != 'ACE':
            return
        nxyz = [self.__getitem__(1)['N']['x'], 
                self.__getitem__(1)['N']['y'], 
                self.__getitem__(1)['N']['z']]
        caxyz = [self.__getitem__(1)['CA']['x'], 
                 self.__getitem__(1)['CA']['y'], 
                 self.__getitem__(1)['CA']['z']]
        if 'PRO' in self.__getitem__(1)['N']['resName']:
            hxyz = [self.__getitem__(1)['CD']['x'], 
                    self.__getitem__(1)['CD']['y'], 
                    self.__getitem__(1)['CD']['z']]
        else:
            hxyz = [self.__getitem__(1)['H']['x'], 
                    self.__getitem__(1)['H']['y'], 
                    self.__getitem__(1)['H']['z']]
        ca_n = ((caxyz[0]-nxyz[0])**2
                + (caxyz[1]-nxyz[1])**2
                + (caxyz[2]-nxyz[2])**2)**0.5
        ca_nn = [(nxyz[i] - caxyz[i]) / ca_n for i in range(3)]
        n_h = ((hxyz[0]-nxyz[0])**2
               + (hxyz[1]-nxyz[1])**2
               + (hxyz[2]-nxyz[2])**2)**0.5
        n_hn = [(hxyz[i] - nxyz[i]) / n_h for i in range(3)]
        n_h2v = [x * 1.008807 for x in rotate(n_hn, ca_nn, math.pi * 2 / 3)]
        h2xyz = [nxyz[i] + n_h2v[i] for i in range(3)]
        # copy
        if 'PRO' in self.__getitem__(1)['N']['resName']:
            self.__getitem__(1)['H2'] = self.__getitem__(1)['CD'].copy()
            self.__getitem__(1)['H2']['element'] = ' H'
        else:
            self.__getitem__(1)['H1'] = self.__getitem__(1)['H'].copy()
            self.__getitem__(1)['H1']['name'] = '  H1'
            self.__getitem__(1)['H2'] = self.__getitem__(1)['H']
            del self.__getitem__(1)['H']
        self.__getitem__(1)['H2']['name'] = '  H2'
        self.__getitem__(1)['H2']['x'] = h2xyz[0]
        self.__getitem__(1)['H2']['y'] = h2xyz[1]
        self.__getitem__(1)['H2']['z'] = h2xyz[2]
        self.pop(0) # remove original ACE
        return
    
    def ct_cap2neu(self):
        if 'CH3' not in self.__getitem__(-1).keys():
            return
        if self.__getitem__(-1)['CH3']['resName'] != 'NME':
            return
        self.__getitem__(-2)['OXT'] = self.__getitem__(-2)['O'].copy()
        self.__getitem__(-2)['OXT']['name'] = ' OXT'
        self.__getitem__(-2)['OXT']['x'] = self.__getitem__(-1)['N']['x']
        self.__getitem__(-2)['OXT']['y'] = self.__getitem__(-1)['N']['y']
        self.__getitem__(-2)['OXT']['z'] = self.__getitem__(-1)['N']['z']
        self.__getitem__(-2)['OXT']['element'] = ' O'
        self.__getitem__(-2)['HXT'] = self.__getitem__(-2)['O'].copy()
        self.__getitem__(-2)['HXT']['name'] = ' HXT'
        self.__getitem__(-2)['HXT']['x'] = self.__getitem__(-1)['H']['x']
        self.__getitem__(-2)['HXT']['y'] = self.__getitem__(-1)['H']['y']
        self.__getitem__(-2)['HXT']['z'] = self.__getitem__(-1)['H']['z']
        self.__getitem__(-2)['HXT']['element'] = ' H'
        self.pop(-1)
        return
    
    def ct_cap2dep(self):
        if 'CH3' not in self.__getitem__(-1).keys():
            return
        if self.__getitem__(-1)['CH3']['resName'] != 'NME':
            return
        self.__getitem__(-2)['OXT'] = self.__getitem__(-2)['O'].copy()
        self.__getitem__(-2)['OXT']['name'] = ' OXT'
        self.__getitem__(-2)['OXT']['x'] = self.__getitem__(-1)['N']['x']
        self.__getitem__(-2)['OXT']['y'] = self.__getitem__(-1)['N']['y']
        self.__getitem__(-2)['OXT']['z'] = self.__getitem__(-1)['N']['z']
        self.__getitem__(-2)['OXT']['element'] = ' O'
        self.pop(-1)
        return
    
    def ct_cap2nh2(self):
        if 'CH3' not in self.__getitem__(-1).keys():
            return
        if self.__getitem__(-1)['CH3']['resName'] != 'NME':
            return
        nxyz = [self.__getitem__(-1)['N']['x'], 
                self.__getitem__(-1)['N']['y'], 
                self.__getitem__(-1)['N']['z']]
        cxyz = [self.__getitem__(-1)['CH3']['x'], 
                self.__getitem__(-1)['CH3']['y'], 
                self.__getitem__(-1)['CH3']['z']]
        c_n = ((cxyz[0]-nxyz[0])**2
               + (cxyz[1]-nxyz[1])**2
               + (cxyz[2]-nxyz[2])**2)**0.5
        n_cn = [(cxyz[i] - nxyz[i]) / c_n for i in range(3)]
        n_h2v = [x * 1.008807 for x in n_cn]
        hxyz = [nxyz[i] + n_h2v[i] for i in range(3)]
        nh2 = {'N': self.__getitem__(-1)['N'], 
               'HN1': self.__getitem__(-1)['H']}
        nh2['N']['resName'] = 'NHE'
        nh2['HN1']['resName'] = 'NHE'
        nh2['HN1']['name'] = ' HN1'
        nh2['HN2'] = nh2['HN1'].copy()
        nh2['HN2']['name'] = ' HN2'
        nh2['HN2']['x'] = hxyz[0]
        nh2['HN2']['y'] = hxyz[1]
        nh2['HN2']['z'] = hxyz[2]
        self.append(nh2)
        self.pop(-2) # remove original NME
        return
    
    def write(self, filename):
        out_str = ''
        i = 0
        for residue in self.__iter__():
            i += 1
            for name in residue.keys():
                out_str += 'ATOM  %s %s%s%s %s%4d%s    %7.3f %7.3f %7.3f' % (
                    residue[name]['serial'],
                    residue[name]['name'],
                    residue[name]['altLoc'],
                    residue[name]['resName'],
                    residue[name]['chainID'],
                    i,
                    residue[name]['iCode'],
                    residue[name]['x'],
                    residue[name]['y'],
                    residue[name]['z'])
                out_str += '%s%s          %s%s\n' % (
                    residue[name]['occupancy'],
                    residue[name]['tempFactor'],
                    residue[name]['element'],
                    residue[name]['charge'])
        with open(filename, 'w') as f:
            f.write(out_str)
        return
    
    def write_amber(self, filename):
        out_str = ''
        i = 0
        for residue in self.__iter__():
            i += 1
            for name in residue.keys():
                rn = residue[name]['resName'].strip()
                if rn == 'HIS':
                    rname = 'HIP'
                else:
                    rname = residue[name]['resName']
                if rn == 'ACE' and name == 'HH31':
                    aname = '  H1'
                elif rn == 'ACE' and name == 'HH32':
                    aname = '  H2'
                elif rn == 'ACE' and name == 'HH33':
                    aname = '  H3'
                elif rn == 'NME' and name == 'CH3':
                    aname = '   C'
                elif rn == 'NME' and name == 'HH31':
                    aname = '  H1'
                elif rn == 'NME' and name == 'HH32':
                    aname = '  H2'
                elif rn == 'NME' and name == 'HH33':
                    aname = '  H3'
                elif rn == 'ILE' and name == 'HG11':
                    aname = 'HG13'
                elif rn == 'GLY' and name == 'H' and 'H2' in residue.keys():
                    aname = '  H1'
                elif rn == 'TYR' and name == 'HE1':
                    aname = residue[name]['name']
                elif rn == 'TYR' and name == 'HD1':
                    aname = residue[name]['name']
                elif rn == 'TRP' and name == 'HD1':
                    aname = residue[name]['name']
                elif rn == 'THR' and name == 'HG1':
                    aname = residue[name]['name']
                elif rn == 'HIS' and name == 'HD1':
                    aname = residue[name]['name']
                elif rn == 'HIS' and name == 'HE1':
                    aname = residue[name]['name']
                elif rn == 'HIE' and name == 'HE1':
                    aname = residue[name]['name']
                elif rn == 'PHE' and name == 'HD1':
                    aname = residue[name]['name']
                elif rn == 'PHE' and name == 'HE1':
                    aname = residue[name]['name']
                elif rn == 'NHE' and name == 'HN1':
                    aname = residue[name]['name']
                elif name[0] == 'H' and name[-1] == '1' and len(name) == 3:
                    if f'H{name[1:-1]}3' in residue.keys():
                        aname = residue[name]['name']
                    else:
                        aname = f'H{name[1:-1]}3'.rjust(4)
                else:
                    aname = residue[name]['name']
                out_str += 'ATOM  %s %s%s%s %s%4d%s    %7.3f %7.3f %7.3f' % (
                    residue[name]['serial'],
                    aname,
                    residue[name]['altLoc'],
                    # residue[name]['resName'],
                    rname,
                    residue[name]['chainID'],
                    i,
                    residue[name]['iCode'],
                    residue[name]['x'],
                    residue[name]['y'],
                    residue[name]['z'])
                out_str += '%s%s          %s%s\n' % (
                    residue[name]['occupancy'],
                    residue[name]['tempFactor'],
                    residue[name]['element'],
                    residue[name]['charge'])
        with open(filename, 'w') as f:
            f.write(out_str)
        return
    
    def write_rosetta(self, filename):
        """write pdb according to rosetta standard
        
        write pdb according to rosetta standard:
            no hydrogen;
            contract cap into first and last residue
            NME N -> NM
            NME CH3 -> CN
            ACE CH3 -> CP2
            ACE O -> OP1
            ACE C -> CO
            NHE N -> NT
        """
        out_str = ''
        for i in range(self.__len__()):
            residue = self.__getitem__(i)
            for name in residue.keys():
                if residue[name]['element'].strip() == 'H':
                    continue
                if residue[name]['resName'].strip() == 'NME':
                    if name == 'N':
                        atom_name = '  NM'
                    elif name == 'CH3':
                        atom_name = '  CN'
                elif residue[name]['resName'].strip() == 'ACE':
                    if name == 'CH3':
                        atom_name = ' CP2'
                    elif name == 'O':
                        atom_name = ' OP1'
                    elif name == 'C':
                        atom_name = '  CO'
                elif residue[name]['resName'].strip() == 'NHE' and name == 'N':
                    atom_name = '  NT'
                else:
                    atom_name = residue[name]['name']
                if residue[name]['resName'].strip() == 'NHE':
                    resname = next(
                        iter(self.__getitem__(i-1).values()))['resName']
                elif residue[name]['resName'].strip() == 'HIE':
                    resname = 'HIS'
                else:
                    resname = residue[name]['resName']
                if residue[name]['resName'].strip() == 'NHE':
                    resseq = next(
                        iter(self.__getitem__(i-1).values()))['resSeq']
                else:
                    resseq = residue[name]['resSeq']
                out_str += 'ATOM  %s %s%s%s %s%s%s    %7.3f %7.3f %7.3f' % (
                    residue[name]['serial'],
                    atom_name,
                    residue[name]['altLoc'],
                    resname,
                    residue[name]['chainID'],
                    resseq,
                    residue[name]['iCode'],
                    residue[name]['x'],
                    residue[name]['y'],
                    residue[name]['z'])
                out_str += '%s%s          %s%s\n' % (
                    residue[name]['occupancy'],
                    residue[name]['tempFactor'],
                    residue[name]['element'],
                    residue[name]['charge'])
        with open(filename, 'w') as f:
            f.write(out_str)
        return
    
    def write_heavy(self, filename, no_side=False, backbone=False, ca=False):
        """write only heavy atom according to order in AtomName"""
        if backbone:
            no_side = True
        out_str = ''
        for residue in self.__iter__():
            resname = three2one[residue[next(iter(residue.keys()))]['resName']]
            if no_side:
                name_list = AtomName[resname][:7]
            else:
                name_list = AtomName[resname]
            for name in name_list:
                if name not in residue.keys():
                    continue
                if residue[name]['element'].strip() == 'H':
                    continue
                if name != 'CA' and ca:
                    continue
                if name == 'CB' and no_side:
                    continue
                if name == 'O' and backbone:
                    continue
                if name == 'CH3' and backbone:
                    continue
                if name == 'CD' and backbone:
                    continue
                out_str += 'ATOM  %s %s%s%s %s%s%s    %7.3f %7.3f %7.3f' % (
                    residue[name]['serial'],
                    residue[name]['name'],
                    residue[name]['altLoc'],
                    residue[name]['resName'],
                    residue[name]['chainID'],
                    residue[name]['resSeq'],
                    residue[name]['iCode'],
                    residue[name]['x'],
                    residue[name]['y'],
                    residue[name]['z'])
                out_str += '%s%s          %s%s\n' % (
                    residue[name]['occupancy'],
                    residue[name]['tempFactor'],
                    residue[name]['element'],
                    residue[name]['charge'])
            if 'OXT' in residue.keys() and not ca:
                name = 'OXT'
                out_str += 'ATOM  %s %s%s%s %s%s%s    %7.3f %7.3f %7.3f' % (
                    residue[name]['serial'],
                    residue[name]['name'],
                    residue[name]['altLoc'],
                    residue[name]['resName'],
                    residue[name]['chainID'],
                    residue[name]['resSeq'],
                    residue[name]['iCode'],
                    residue[name]['x'],
                    residue[name]['y'],
                    residue[name]['z'])
                out_str += '%s%s          %s%s\n' % (
                    residue[name]['occupancy'],
                    residue[name]['tempFactor'],
                    residue[name]['element'],
                    residue[name]['charge'])
        with open(filename, 'w') as f:
            f.write(out_str)
    
    def write_xyz(self, filename=None):
        atoms = []
        for i in range(self.__len__()):
            residue = self.__getitem__(i)
            for name in residue.keys():
                atoms.append('%s %f %f %f\n' % (residue[name]['element'], 
                                                residue[name]['x'], 
                                                residue[name]['y'], 
                                                residue[name]['z']))
        if filename is None:
            return [f'{len(atoms)}\n', '\n'] + atoms
        with open(filename, 'w') as f:
            f.write('%d\n\n' % len(atoms))
            for a in atoms:
                f.write(a)

def openmm_amber_simulation(crd='input.rst7', top='input.parm7', useForce=-1):
    inpcrd = AmberInpcrdFile(crd)
    prmtop = AmberPrmtopFile(top)
    system = prmtop.createSystem(implicitSolvent=GBn2, nonbondedMethod=NoCutoff)
    if useForce >= 0:
        for i in range(10, -1, -1):
            if i == useForce:
                continue
            try:
                system.removeForce(i)
            except:
                pass
    simulation = Simulation(prmtop.topology, system,
                            LangevinMiddleIntegrator(300 * kelvin,
                                                     1 / picosecond,
                                                     0.004 * picoseconds))
    simulation.context.setPositions(inpcrd.positions)
    return simulation

def openmm_opt_simulation(simulation, oname='output.lzf', tol=10, opt=True):
    t0 = time.time()
    try:
        if opt:
            simulation.minimizeEnergy(
                tolerance=Quantity(value=tol, unit=kilojoule/(nanometer*mole)))
    except OpenMMException:
        return t0 - time.time()
    t0 = time.time() - t0
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    a = state.getPositions(asNumpy=True).value_in_unit(angstrom)
    b = min([(a[i][0]-a[j][0])**2+(a[i][1]-a[j][1])**2+(a[i][2]-a[j][2])**2 for i in range(len(a)) for j in range(i)])
    if b < 0.7:
        return t0
    opdb = oname[:-4] + '.pdb'
    oxyz = oname[:-4] + '.xyz'
    PDBFile.writeFile(
        simulation.topology, state.getPositions(), open(opdb, 'w'))
    xyz_np = state.getPositions(asNumpy=True).value_in_unit(angstrom)
    with open(oxyz, 'w') as F:
        F.write("%d\nGenerate by lzf.openmm_opt, energy in kcal/mol is %f\n" %
                (len(xyz_np), 
                 state.getPotentialEnergy().value_in_unit(kilocalorie/mole)))
        for s, xyz in zip(
                np.array(
                    [x.element.symbol for x in simulation.topology.atoms()]), 
                xyz_np):
            F.write("%s %f %f %f\n" % (s, xyz[0], xyz[1], xyz[2]))
    return t0

def openmm_opt_amber(tol=10, opt=True, useForce=-1):
    simulation = openmm_amber_simulation(useForce=useForce)
    t = openmm_opt_simulation(simulation, tol=tol, opt=opt)
    return t

def openmm_opt(pdb, oname, tol=10, solvent=None, ffxml='amber14-all.xml'):
    pdbfile = PDBFile(pdb)
    modeller = Modeller(pdbfile.topology, pdbfile.positions)
    if solvent == 'amoeba_gk':
        ff = MyForceField('amoeba2018.xml', 'amoeba2018_gk.xml')
        system = ff.createSystem(modeller.topology, 
                                 mutualInducedTargetEpsilon=0.0001)
    elif solvent is None:
        # ff = MyForceField('amber14-all.xml', 'amber14/tip3p.xml')
        ff = MyForceField(ffxml)
        try:
            system = ff.createSystem(modeller.topology, 
                                     nonbondedMethod=NoCutoff)
        except:
            with open(pdb, 'r') as f:
                for line in f.readlines():
                    print(line)
            for b in pdbfile.topology.bonds():
                print(b)
            raise
    else:
        ff = MyForceField(ffxml, 'amber14/tip3p.xml', 
                          'implicit/%s.xml' % solvent)
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                                 soluteDielectric=1.0, solventDielectric=80.0)
    simulation = Simulation(modeller.topology, system,
                            LangevinMiddleIntegrator(300 * kelvin,
                                                     1 / picosecond,
                                                     0.004 * picoseconds))
    simulation.context.setPositions(pdbfile.positions)  #
    t0 = time.time()
    try:
        simulation.minimizeEnergy(
            tolerance=Quantity(value=tol, unit=kilojoule/(nanometer*mole)))
    except OpenMMException:
        return t0 - time.time()
    t0 = time.time() - t0
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    a = state.getPositions(asNumpy=True).value_in_unit(angstrom)
    b = min([(a[i][0]-a[j][0])**2+(a[i][1]-a[j][1])**2+(a[i][2]-a[j][2])**2 for i in range(len(a)) for j in range(i)])
    if b < 0.7:
        return t0 - time.time()
    opdb = oname[:-4] + '.pdb'
    oxyz = oname[:-4] + '.xyz'
    PDBFile.writeFile(
        simulation.topology, state.getPositions(), open(opdb, 'w'))
    with open(oxyz, 'w') as F:
        F.write("%d\nGenerate by lzf.openmm_opt, energy in kcal/mol is %f\n" %
                (len(pdbfile.positions), 
                 state.getPotentialEnergy().value_in_unit(kilocalorie/mole)))
        for s, xyz in zip(
                np.array(
                    [x.element.symbol for x in simulation.topology.atoms()]), 
                state.getPositions(asNumpy=True).value_in_unit(angstrom)):
            F.write("%s %f %f %f\n" % (s, xyz[0], xyz[1], xyz[2]))
    return t0


if __name__ == '__main__':
    # pdb = ReadPdbAtom(sys.argv[1])
    # pdb.write_amber(sys.argv[1]+'.PDB')
    xyz2pdb(seq=sys.argv[1], xyzfile=sys.argv[2], nt=sys.argv[3], ct=sys.argv[4])
    # openmm_opt('template.pdb', 'y.pdb')
    
