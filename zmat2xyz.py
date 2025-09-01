#!/bin/env python3
import numpy as np
import os
import time


'''将当前目录由路径矩阵生成的txt文件转成xyz文件和inp文件，
inp文件用于ORCA做PM3计算，判断构型对错'''

def rotate(r, w, ang):
    # 返回绕轴心向量w旋转角度a后的向量
    # 利用Rodrigues旋转公式v = r·cos(a) + n×r·sin(a) + (n·r)*n*(1 - cos(a))
    #  n为w方向的单位向量
    n = w / np.linalg.norm(w)
    a = np.deg2rad(ang)
    return r * np.cos(a) + np.cross(n, r) * np.sin(a) + np.dot(n, r) * n * (1 - np.cos(a))

def zmat2xyz(zmatrix):
    N = len(zmatrix)
    atomlist = [line[0] for line in zmatrix]
    coords = np.zeros((N, 3))

    for i in range(1, N):
        line = zmatrix[i]
        if i == 1:
            dist = line[2]
            coords[i][0] = coords[0][0] + dist
        elif i == 2:
            a, b = line[1] - 1, line[3] - 1
            dist, ang = line[2], line[4]
            # coords[i][1] = coords[a][1] + dist * np.sin(np.deg2rad(ang))
            # coords[i][0] = coords[a][0] + dist * np.cos(np.deg2rad(ang))
            r = (coords[b] - coords[a]) / distance(coords[a], coords[b]) * dist
            w = np.array([0.0, 0.0, 1.0])
            coords[i] = rotate(r, w, ang) + coords[a]
        else:
            a, b, c = line[1] - 1, line[3] - 1, line[5] - 1
            dist, ang, dih = line[2], line[4], line[6]
            ab = coords[b] - coords[a]
            ac = coords[c] - coords[a]
            w = rotate(np.cross(ab, ac), ab, -dih)
            r = ab / np.linalg.norm(ab) * dist
            coords[i] = rotate(r, w, ang) + coords[a]
    coords -= np.average(coords, 0)
    return atomlist, coords


def readxyz(xyzfile):
    t = np.loadtxt(xyzfile, dtype=np.str, skiprows=2)
    atomlist = t[:,0]
    coords = np.array(t[:,1:], dtype=np.float64)
    return atomlist, coords

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def interangle(p1, p2, p3):
    k1, k2 = p1 - p2, p3 - p2
    ans = np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2))
    return np.rad2deg(np.arccos(ans))

def diheangle(a, b, c, d):
    ba, bc = a - b, c - b
    k1 = np.cross(ba, bc)
    cb, cd = b - c, d - c
    k2 = np.cross(cb, cd)
    flag = np.dot(np.cross(k1, k2), bc)
    ans = np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2))
    ang = np.rad2deg(np.arccos(ans))
    return ang if flag > 0 else -ang


def txt2zmat(txtfile):
    with open(txtfile) as f:
        tf = f.readlines()
    temp = [line.strip().split() for line in tf[5:]]
    ans = [] # zmatrix
    zmatend = False
    rotang = {}     # dict of setting diheangs   character: degrees
    changedrow = []
    for i, line in enumerate(temp):
        if not zmatend:
            if line:
                ans.append([x.strip() for x in line])
                if i >= 3 and line[6][0].isalpha():
                    changedrow.append(i)
            else:
                zmatend = True
        else:
            if line and len(line) == 2:
                ch, dang = line[0].strip(), float(line[1].strip())
                rotang[ch] = dang
    if len(rotang) != len(changedrow):
        print(txtfile + " file wrong")
    for p in changedrow:
        ch = ans[p][6]
        ans[p][6] = rotang[ch]

    for i, line in enumerate(ans):
        if i == 1:
            line[1], line[2] = int(line[1]), float(line[2])
        elif i == 2:
            line[1], line[3] = int(line[1]), int(line[3])
            line[2], line[4] = float(line[2]), float(line[4])
        elif i >= 3:
            line[1], line[3], line[5] = int(line[1]), int(line[3]), int(line[5])
            line[2], line[4], line[6] = float(line[2]), float(line[4]), float(line[6])

    return ans



def xyz2zmat(atomlist, coords, templatefile):
    # return zmatrix
    with open(templatefile) as f:
        tz = f.readlines()
    temp = tz[5:]
    N = len(atomlist)
    if len(temp) != N:
        return
    ans = []
    for i in range(N):
        line = temp[i].split()
        if i + 1 == 1:
            ans.append([atomlist[i]])
        elif i + 1 == 2:
            a, b = i, int(line[1]) - 1
            dist = distance(coords[a], coords[b])
            ans.append([atomlist[i], b + 1, dist])
        elif i + 1 == 3:
            a, b, c = i, int(line[1]) - 1, int(line[3]) - 1
            dist = distance(coords[a], coords[b])
            ang = interangle(coords[a], coords[b], coords[c])
            ans.append([atomlist[i], b + 1, dist, c + 1, ang])
        else:
            a, b, c, d = i, int(line[1]) - 1, int(line[3]) - 1, int(line[5]) - 1
            dist = distance(coords[a], coords[b])
            ang = interangle(coords[a], coords[b], coords[c])
            dih = diheangle(coords[a], coords[b], coords[c], coords[d])
            ans.append([atomlist[i], b + 1, dist, c + 1, ang, d + 1, dih])
    return ans

def writexyz(xyzfile, atomlist, coords):
    N = len(atomlist)
    with open(xyzfile, 'w') as f:
        f.write(str(N)+'\n')
        f.write('\n')
        for i in range(N):
            f.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(atomlist[i], *coords[i]))


def writeinp(inpfile, atomlist, coords):
    N = len(atomlist)
    with open(inpfile, 'w') as f:
        f.write("! PM3\n")
        f.write("* xyz 0 1\n")
        for i in range(N):
            f.write("{}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(atomlist[i], *coords[i]))
        f.write("*\n")

def writezmat(zmatfile, zmatrix):
    with open(zmatfile, 'w') as f:

        for i, line in enumerate(zmatrix):
            if i == 0:
                f.write(line + '\n')
            elif i == 1:
                f.write("%s\t%d\t%.5f\n" % (line[0], line[1], line[2]))
            elif i == 2:
                f.write("%s\t%d\t%.5f\t%d\t%.2f\n" % (line[0], line[1], line[2], line[3], line[4]))
            else:
                f.write("%s\t%d\t%.5f\t%d\t%.2f\t%d\t%.2f\n" % (line[0], line[1], line[2], line[3], line[4], line[5], line[6]))

import glob
import argparse
def writexyzinp(txtfile):
    zmatrix = txt2zmat(txtfile)
    atomlist, coords = zmat2xyz(zmatrix)
    inpfile = txtfile[:-4] + '.inp'
    xyzfile = txtfile[:-4] + '.xyz'
    writexyz(xyzfile, atomlist, coords)
    writeinp(inpfile, atomlist, coords)
def writexyzinplist(txtfilelist):
    for txtfile in txtfilelist:
        zmatrix = txt2zmat(txtfile)
        atomlist, coords = zmat2xyz(zmatrix)
        inpfile = txtfile[:-4] + '.inp'
        xyzfile = txtfile[:-4] + '.xyz'
        writexyz(xyzfile, atomlist, coords)
        writeinp(inpfile, atomlist, coords)
def txt2xyz(fn=None):
    if fn is None:
        txtlist = glob.glob("*.txt")
    else:
        txtlist = [fn]
    for txtfile in txtlist:
        try:
            zmatrix = txt2zmat(txtfile)
        except:
            print(txtfile)
            raise
        atomlist, coords = zmat2xyz(zmatrix)
        xyzfile = txtfile[:-4] + '.xyz'
        writexyz(xyzfile, atomlist, coords)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--template', type=str, default='template', help='default is ./template')
    parser.add_argument('-np', type=int, default=1, help='default is 1')
    parser.add_argument('-xyz', action='store_true', default=True, help='default is true')
    parser.add_argument('-inp', action='store_true', default=False, help='default is false')
    args = parser.parse_args()
    templatefile = args.template #'template'
    path = '.'
    flist = os.listdir(path)
    txtlist = [fname for fname in flist if fname[-4:] == '.txt']
    # print(txtlist)
    time1 = time.time()
    if args.np == 1:
        for i, txtfile in enumerate(txtlist, 1):
            if i%1000==0: print(i, txtfile)
            zmatrix = txt2zmat(path + '/' + txtfile)
            atomlist, coords = zmat2xyz(zmatrix)
            inpfile = path + '/' + txtfile[:-4] + '.inp'
            xyzfile = path + '/' + txtfile[:-4] + '.xyz'
            if args.xyz: writexyz(xyzfile, atomlist, coords)
            if args.inp: writeinp(inpfile, atomlist, coords)
    else:
        import multiprocessing
        from tqdm import trange
        pool = multiprocessing.Pool(args.np)
        I = [i for i in range(0,len(txtlist),1000)] + [len(txtlist)]
        p = trange(len(I)-1)
        results = []
        for i in range(len(I)-1):
            result = pool.apply_async(writexyzinplist,
                args=(txtlist[I[i]:I[i+1]],),
                callback=lambda _: p.update(1),
                error_callback=lambda _: p.update(1))
            results.append(result)
        [i.get() for i in results]
    print("ok")
    time2 = time.time()
    print("%.1fs" % (time2 - time1))
