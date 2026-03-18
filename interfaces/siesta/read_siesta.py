'''
Author: Changwei Zhang 
Date: 2023-05-18 17:07:53 
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-18 17:07:53 
'''

import numpy as np
from numba import njit
from copy import deepcopy
import json
import re
import sys
import os
from time import time
import multiprocessing
from scipy.sparse import csr_matrix as csr
# import matplotlib.pyplot as plt

# the hamilt matrix almost stored as follows, with some modification.
# shape: no_u * no_s
# csr((hamilt, listh, listhptr))
# hamilt:   (nh)
# numh:     (no_u)  nonzero element # in each row
# listhptr: (no_u)  row start index
# listh:    (nh)    col index
# indxuo:   (no_s)  sawtooth function mapping no_s to no_u

# Before using this script,
# Make sure all the atoms is in the cell.

au2ang = 0.5291772490000065
ry2ha  = 13.60580 / 27.21138506

def convInt(x):
  if type(x) == list:
    for idx in range(len(x)):
      x[idx] = convInt(x[idx])
  elif type(x) == np.ndarray:
    return x.tolist()
  elif type(x) == np.int32 or type(x) == np.int64:
    return int(x)
  return x
def convFloat(x):
  if type(x) == list:
    for idx in range(len(x)):
      x[idx] = convFloat(x[idx])
  elif type(x) == np.ndarray:
    return x.tolist()
  elif type(x) == np.float32 or type(x) == np.float64:
    return float(x)
  return x

def getCellShift(pos:np.ndarray, invcell:np.ndarray):
  # convert to fractional coordinate.
  direct = pos @ invcell
  return np.around(direct).astype(int)

class FDF:
  '''
  @property: z: ndarray. Z number of each atom. shape: (natoms)
  @property: cell
  @property: invcell
  @property: pos
  '''

  def __init__(self, file:str):
    self.fp = file

    with open(self.fp) as f:
      content = f.read()
    num = r'-?\d+\.?\d*'
    wht = r'\s+'
    pattern_lattconst = re.compile(r'LatticeConstant\s+' + f'({num})' + r'\s+([A-Za-z]+)', flags=re.I)
    pattern_latt = re.compile(r'%block LatticeVectors.*' + f'{wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num})' + r'\s+%endblock LatticeVectors', flags=re.I)
    pattern_unit = re.compile(r'AtomicCoordinatesFormat\s+([A-Za-z]+)', flags=re.I)
    pattern_cblk = re.compile(r'%block AtomicCoordinatesAndAtomicSpecies(.+)%endblock AtomicCoordinatesAndAtomicSpecies', flags=re.S)
    pattern_coor = re.compile(f'{wht}({num}){wht}({num}){wht}({num}){wht}(\d+)')
    pattern_sblk = re.compile(r'%block ChemicalSpeciesLabel(.+)%endblock ChemicalSpeciesLabel', flags=re.S)
    pattern_spec = re.compile(r'\s+(\d+)\s+(\d+)\s+(\w+)')

    # lattice, support ang / bohr
    lattconst, lattunit = pattern_lattconst.findall(content)[0]
    latt = pattern_latt.findall(content)[0]
    latt = np.array([float(var) for var in latt]).reshape(-1, 3) * float(lattconst)
    if lattunit.lower() == 'ang':
      latt /= au2ang

    # species
    sblk = pattern_sblk.findall(content)[0]
    species = {}
    z = [] # shape: (natoms,)
    for idx, iz, spec in pattern_spec.findall(sblk):
      species[int(idx)] = int(iz)

    # coordinates, only support ang
    unit = pattern_unit.findall(content)[0]
    cblk = pattern_cblk.findall(content)[0]
    coordinates = []
    for coor in pattern_coor.findall(cblk):
      coordinates.append([float(coor[0]), float(coor[1]), float(coor[2])])
      z.append(species[int(coor[3])])
    coordinates = np.array(coordinates, dtype=float)
    if unit.lower() == 'ang':
      coordinates /= au2ang

    self.pos = coordinates
    self.z = np.array(z, dtype=int)
    self.cell = latt
    self.invcell = np.linalg.inv(latt)


def mp_parse_sparse_matrix(fdf:FDF, hamilt:csr, Sover:csr, xijs:list[csr], no:np.ndarray, indo:np.ndarray,
                           no_u, no_s, na_u, ia, ispin=0):
  edge_idx_src = []
  edge_idx_dst = []
  Hon = []
  Hoff= []
  Son = []
  Soff= []
  cell_shift = []
  nbr_shift = []
  for jsuper in range(0, no_s, no_u):
    hamil:csr = hamilt[:,jsuper:jsuper+no_u]
    if ispin == 0:
      Sove:csr = Sover[:,jsuper:jsuper+no_u]
    for ja in range(na_u):
      ham:csr = hamil[:,indo[ja]:indo[ja]+no[ja]]
      if ham.getnnz() == 0:
        continue
      if ispin == 0:
        sr:csr = Sove[:,indo[ja]:indo[ja]+no[ja]] 
      indptr = np.array(ham.indptr, dtype=int)
      io = len(indptr[indptr==0])-1
      jo = ham.indices[0]+indo[ja]
      jos = jo + jsuper
      xij = np.array([xijs[0][io,jos], xijs[1][io,jos], xijs[2][io,jos]])
      cs = getCellShift(fdf.pos[ia]-fdf.pos[ja]+xij, fdf.invcell) # cell_shift

      if ia == ja and np.all(cs == 0):
        # onsite
        Hon.append(ham.toarray().flatten())
        if ispin == 0:
          Son.append(sr.toarray().flatten())
      else:
        # offsite
        Hoff.append(ham.toarray().flatten())
        if ispin == 0:
          Soff.append(sr.toarray().flatten())
        edge_idx_src.append(ia)
        edge_idx_dst.append(ja)
        cell_shift.append(cs)
        nbr_shift.append(cs @ fdf.cell)
  return edge_idx_src, edge_idx_dst, Hon, Hoff, Son, Soff, cell_shift, nbr_shift

class HSX:
  
  def __init__(s, file:str):
    s.fp = open(file, 'rb')
    tmp = np.fromfile(s.fp, dtype=np.int32, count=8)
    s.nspecies = tmp[0]
    s.na_u     = tmp[1]
    s.no_u     = tmp[2]
    s.no_s     = tmp[3]
    s.nspin    = tmp[4]
    s.nh       = tmp[5]
    s.gamma    = bool(tmp[6])
    s.has_xij  = bool(tmp[7])
    if s.gamma:
      pass
      # print('gamma')
    if not s.has_xij:
      print('has_xij eq False')
    if s.no_u != s.no_s:
      pass
      # print('no_u neq no_s')
    s.no = np.fromfile(s.fp, dtype=np.int32, count=s.nspecies)
    # s.nquant = np.fromfile(s.fp, dtype=np.int32, count=MAXNAO*s.nspecies). \
    #             reshape(MAXNAO, -1)
    # s.lquant = np.fromfile(s.fp, dtype=np.int32, count=MAXNAO*s.nspecies). \
    #             reshape(MAXNAO, -1)
    # s.zeta   = np.fromfile(s.fp, dtype=np.int32, count=MAXNAO*s.nspecies). \
    #             reshape(MAXNAO, -1)
    s.iaorb    = np.fromfile(s.fp, dtype=np.int32, count=s.no_u)
    s.iphorb   = np.fromfile(s.fp, dtype=np.int32, count=s.no_u)
    s.numh     = np.fromfile(s.fp, dtype=np.int32, count=s.no_u)
    s.listhptr = np.fromfile(s.fp, dtype=np.int32, count=s.no_u)
    s.listh    = np.fromfile(s.fp, dtype=np.int32, count=s.nh)
    s.indxuo   = np.fromfile(s.fp, dtype=np.int32, count=s.no_s)
    s.hamilt   = np.fromfile(s.fp, dtype=np.float32, count=s.nh*s.nspin). \
                reshape(-1, s.nh) * ry2ha
    s.Sover    = np.fromfile(s.fp, dtype=np.float32, count=s.nh)
    s.xij      = np.fromfile(s.fp, dtype=np.float32, count=3*s.nh). \
                reshape(-1, 3)
    # x = np.linalg.norm(s.xij, axis=1)
    # print(np.max(x))
    # exit()
    s.isa      = np.fromfile(s.fp, dtype=np.int32, count=s.na_u)
    s.zval     = np.fromfile(s.fp, dtype=np.float32, count=s.nspecies)
    s.fp.close()

  def getGraph2(s, fdf:FDF, graph:dict={}, skip=False, tojson=False):
    #######################################################
    @njit
    def constructInvEdges(noff, edge_idx_src, edge_idx_dst, cell_shift):
      ierr = False
      inv_edge_idx = -1 * np.ones(noff, dtype=np.int32)
      for idx in range(0, noff):
        if inv_edge_idx[idx] != -1:
          continue
        for jdx in range(0, noff):
          if (edge_idx_dst[jdx] == edge_idx_src[idx] and
              edge_idx_src[jdx] == edge_idx_dst[idx] and
              np.all(cell_shift[jdx] == -cell_shift[idx])):
            inv_edge_idx[idx] = jdx
            inv_edge_idx[jdx] = idx
            break
        else:
          ierr = True
      return ierr, inv_edge_idx
    #######################################################
    @njit
    def fillHoff(noff, cx, cy, cz, ia, ja,
                 edge_idx_src, edge_idx_dst, cell_shift):
      '''ierr, ioff'''
      for ioff, src, dst, cs in zip(np.arange(noff), edge_idx_src, edge_idx_dst, cell_shift):
        if (src == ia and dst == ja and cx == cs[0] and cy == cs[1] and cz == cs[2]):
          return False, ioff
      return True, 0
    #######################################################
    assert (not graph and not skip) or (graph and skip)
    edge_idx_src = []
    edge_idx_dst = []
    cell_shift   = []
    nbr_shift    = []
    Hon = [[]] if s.nspin == 1 else [[],[]] if s.nspin == 2 else [[],[],[],[]]
    Hoff= [[]] if s.nspin == 1 else [[],[]] if s.nspin == 2 else [[],[],[],[]]
    Son = []
    Soff= []

    if skip:
      graph_ = deepcopy(graph)
      s.noff = len(graph_['inv_edge_idx'])
      Hoff = graph_['Hoff']
      for ispin in range(s.nspin):
        for ioff in range(s.noff):
          Hoff[ispin][ioff] = np.zeros_like(Hoff[ispin][ioff], dtype=np.float32)

    no = []
    tmp = [0,0]
    for io in range(s.no_u):
      ia = s.iaorb[io]-1
      if tmp[0] == 0:
        tmp = [1,ia]
      elif tmp[1] == ia:
        tmp[0] += 1
      else:
        no.append(tmp[0])
        tmp = [1,ia]
    no.append(tmp[0])
    no = np.array(no, dtype=int) # shape: (natoms)
    indo = np.zeros_like(no, dtype=int) # shape: (natoms)
    indo[1:] = np.cumsum(no[:-1])

    s.listh -= 1
    s.listhptr = np.append(s.listhptr, s.nh) # listhptr start from 0 in fortran!
    hamilts = [] # shape (nspin, no_u, no_s)
    xijs = [] # shape(3, no_u, no_s)
    for ispin in range(0, s.nspin):
      hamilts.append(csr((s.hamilt[ispin], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))
    Sovers = csr((s.Sover, s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)) # shape (no_u, no_s)
    for tmp in range(0, 3):
      xijs.append(csr((s.xij[:,tmp], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))

    for ispin in range(0, s.nspin):
      for jsuper in range(0, s.no_s, s.no_u):
        hamilt:csr = hamilts[ispin][:,jsuper:jsuper+s.no_u]
        if ispin == 0 and not skip:
          Sover:csr = Sovers[:,jsuper:jsuper+s.no_u]
        for ia in range(s.na_u):
          hamil:csr = hamilt[indo[ia]:indo[ia]+no[ia],:]
          if ispin == 0 and not skip:
            Sove:csr = Sover[indo[ia]:indo[ia]+no[ia],:]
          for ja in range(s.na_u):
            ham:csr = hamil[:,indo[ja]:indo[ja]+no[ja]]
            if ham.getnnz() == 0:
              continue
            if ispin == 0 and not skip:
              sr:csr = Sove[:,indo[ja]:indo[ja]+no[ja]] 
            indptr = np.array(ham.indptr, dtype=int)
            io = len(indptr[indptr==0])-1+indo[ia]
            jo = ham.indices[0]+indo[ja]
            jos = jo + jsuper
            xij = np.array([xijs[0][io,jos], xijs[1][io,jos], xijs[2][io,jos]])
            cs = getCellShift(fdf.pos[ia]-fdf.pos[ja]+xij, fdf.invcell) # cell_shift

            # plt.scatter(np.ones_like(ham.toarray().flatten())*np.linalg.norm(xij), ham.toarray().flatten(), s=0.2, marker='.')
            if ia == ja and np.all(cs == 0):
              # onsite
              Hon[ispin].append(ham.toarray().flatten())
              if ispin == 0 and not skip:
                Son.append(sr.toarray().flatten())
            else:
              # offsite
              if not skip:
                Hoff[ispin].append(ham.toarray().flatten())
                if ispin == 0:
                  Soff.append(sr.toarray().flatten())
                  edge_idx_src.append(ia)
                  edge_idx_dst.append(ja)
                  cell_shift.append(cs)
                  nbr_shift.append(cs @ fdf.cell)
              else:
                ierr, ioff = fillHoff(s.noff, cs[0], cs[1], cs[2], ia, ja,
                                      np.array(graph['edge_index'][0]), 
                                      np.array(graph['edge_index'][1]), 
                                      np.array(graph['cell_shift']))
                Hoff[ispin][ioff] = ham.toarray().flatten()
    #######################################################
    if not skip:
      # construct the edges
      edge_index = [edge_idx_src, edge_idx_dst]
      s.noff = len(edge_idx_src)
      # inv_edge_idx
      ierr, inv_edge_idx = constructInvEdges(s.noff, 
                                             np.array(edge_idx_src),
                                             np.array(edge_idx_dst),
                                             np.array(cell_shift))
      if ierr:
        print('inv_edge_idx error')
        exit()
      # construct the graph
      graph_ = {}
      graph_['edge_index'] = convInt(edge_index) if tojson else np.array(edge_index)
      graph_['inv_edge_idx'] = convInt(inv_edge_idx) if tojson else np.array(inv_edge_idx)
      graph_['cell_shift'] = convInt(cell_shift) if tojson else np.array(cell_shift)
      graph_['nbr_shift'] = convFloat(nbr_shift) if tojson else np.array(nbr_shift)
      graph_['pos'] = convFloat(fdf.pos) if tojson else fdf.pos
      graph_['Son'] = convFloat(Son) if tojson else Son
      graph_['Soff'] = convFloat(Soff) if tojson else Soff
    graph_['Hon'] = convFloat(Hon) if tojson else Hon
    graph_['Hoff'] = convFloat(Hoff) if tojson else Hoff
    return graph_
  
  def getGraph3(s, fdf:FDF, graph:dict={}, ntask=1, tojson=False):
    #######################################################
    def constructInvEdges(noff, edge_idx_src, edge_idx_dst, cell_shift):
      ierr = False
      assert np.mod(noff, 2) == 0
      cast = np.hstack((np.arange(noff).reshape(-1,1), cell_shift, edge_idx_src.reshape(-1,1), edge_idx_dst.reshape(-1,1)))

      uporder0 = sorted(cast[cast[:,4] == cast[:,5]], key=lambda x: (x[1], x[2], x[3], x[4]))

      uporder1 = sorted(cast[cast[:,4] == cast[:,5]], key=lambda x: (x[1], x[2], x[3], -x[4]))
      
      uporder2 = sorted(cast[cast[:,4] < cast[:,5]],  key=lambda x: (x[1], x[2], x[3], x[4], x[5]))

      uporder3 = sorted(cast[cast[:,4] > cast[:,5]],  key=lambda x: (x[1], x[2], x[3], -x[5], -x[4]))
      
      if len(uporder0) != 0:
        leneq = len(uporder0) // 2
        upordereq = np.array(uporder0[:leneq] + uporder1[leneq:], dtype=np.int32)[:,0]
      else:
        upordereq = []
      uporderne = np.array(uporder2 + uporder3, dtype=np.int32)[:,0]
      downordereq = upordereq[::-1]
      downorderne = uporderne[::-1]
      cast0 = np.array([upordereq, downordereq])
      cast1 = np.array([uporderne, downorderne])
      cast = np.hstack((cast0, cast1)).T
      inv_edge_idx = sorted(cast, key=lambda x: x[0])
      inv_edge_idx = np.array(inv_edge_idx, dtype=np.int32)[:,1]
      return ierr, inv_edge_idx
    #######################################################
    edge_idx_src = []
    edge_idx_dst = []
    cell_shift   = []
    nbr_shift    = []
    Hon = [[]] if s.nspin == 1 else [[],[]] if s.nspin == 2 else [[],[],[],[]]
    Hoff= [[]] if s.nspin == 1 else [[],[]] if s.nspin == 2 else [[],[],[],[]]
    Son = []
    Soff= []

    # multiprocessing
    mp_results = []
    mp_nproc = min(multiprocessing.cpu_count(), ntask)
    mp_pool = multiprocessing.Pool(processes=mp_nproc)

    #################################################
    time1 = time()###################################
    #################################################

    no = []
    tmp = [0,0]
    for io in range(s.no_u):
      ia = s.iaorb[io]-1
      if tmp[0] == 0:
        tmp = [1,ia]
      elif tmp[1] == ia:
        tmp[0] += 1
      else:
        no.append(tmp[0])
        tmp = [1,ia]
    no.append(tmp[0])
    no = np.array(no, dtype=int) # shape: (natoms)
    indo = np.zeros_like(no, dtype=int) # shape: (natoms)
    indo[1:] = np.cumsum(no[:-1])

    s.listh -= 1
    s.listhptr = np.append(s.listhptr, s.nh) # listhptr start from 0 in fortran!
    hamilts = [] # shape (nspin, no_u, no_s)
    xijs = [] # shape(3, no_u, no_s)
    for ispin in range(0, s.nspin):
      hamilts.append(csr((s.hamilt[ispin], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))
    Sovers = csr((s.Sover, s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)) # shape (no_u, no_s)
    for tmp in range(0, 3):
      xijs.append(csr((s.xij[:,tmp], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))
    delattr(s, 'hamilt')
    delattr(s, 'Sover')
    delattr(s, 'xij')

    #################################################
    print('PART1 %f' % (time() - time1), flush=True)#
    time1 = time()###################################

    for ispin in range(0, s.nspin):
      for ia in range(s.na_u):
        hamilt:csr = hamilts[ispin][indo[ia]:indo[ia]+no[ia],:]
        if ispin == 0:
          Sover:csr = Sovers[indo[ia]:indo[ia]+no[ia],:]
        xij1 = xijs[0][indo[ia]:indo[ia]+no[ia],:]
        xij2 = xijs[1][indo[ia]:indo[ia]+no[ia],:]
        xij3 = xijs[2][indo[ia]:indo[ia]+no[ia],:]
        tmp = mp_pool.apply_async(mp_parse_sparse_matrix, (fdf, hamilt, Sover, [xij1, xij2, xij3],
                                                           no, indo, s.no_u, s.no_s, s.na_u, ia, ispin))
        mp_results.append(tmp)
      for mp_res in mp_results:
        tmp = mp_res.get()
        edge_idx_src.extend(tmp[0])
        edge_idx_dst.extend(tmp[1])
        Hon[ispin].extend(tmp[2])
        Hoff[ispin].extend(tmp[3])
        Son.extend(tmp[4])
        Soff.extend(tmp[5])
        cell_shift.extend(tmp[6])
        nbr_shift.extend(tmp[7])
      mp_results = []
    #################################################
    print('PART2 %f' % (time() - time1), flush=True)#
    time1 = time()###################################
    #######################################################
    # construct the edges
    edge_index = [edge_idx_src, edge_idx_dst]
    s.noff = len(edge_idx_src)
    # inv_edge_idx
    ierr, inv_edge_idx = constructInvEdges(s.noff, 
                                            np.array(edge_idx_src),
                                            np.array(edge_idx_dst),
                                            np.array(cell_shift))
    if ierr:
      print('inv_edge_idx error')
      exit()
    #####################################
    print('PART3 %f' % (time() - time1), flush=True)#
    #####################################
    mp_pool.close()
    mp_pool.join()
    # construct the graph
    graph_ = {}
    graph_['edge_index'] = convInt(edge_index) if tojson else np.array(edge_index)
    graph_['inv_edge_idx'] = convInt(inv_edge_idx) if tojson else np.array(inv_edge_idx)
    graph_['cell_shift'] = convInt(cell_shift) if tojson else np.array(cell_shift)
    graph_['nbr_shift'] = convFloat(nbr_shift) if tojson else np.array(nbr_shift)
    graph_['pos'] = convFloat(fdf.pos) if tojson else fdf.pos
    graph_['Son'] = convFloat(Son) if tojson else Son
    graph_['Soff'] = convFloat(Soff) if tojson else Soff
    graph_['Hon'] = convFloat(Hon) if tojson else Hon
    graph_['Hoff'] = convFloat(Hoff) if tojson else Hoff
    return graph_


  def getHK(s, fdf:FDF, k=np.array([0,0,0]), isSOC=False):
    assert(np.all(k == 0))
    dtype = np.float32 if not isSOC else np.complex64
    HK = np.zeros([s.nspin, s.no_u, s.no_u], dtype=np.float32)

    s.listh -= 1
    s.listhptr = np.append(s.listhptr, s.nh) # listhptr start from 0 in fortran!
    hamilts = [] # shape (nspin, no_u, no_s)
    # xijs = [] # shape(3, no_u, no_s)
    for ispin in range(0, s.nspin):
      hamilts.append(csr((s.hamilt[ispin], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))
    # for tmp in range(0, 3):
    #   xijs.append(csr((s.xij[:,tmp], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))

    for ispin in range(0, s.nspin):
      for jsuper in range(0, s.no_s, s.no_u):
        hamilt:csr = hamilts[ispin][:,jsuper:jsuper+s.no_u]
        HK[ispin] += hamilt
    return HK
  
  def getSK(s, fdf:FDF, k=np.array([0,0,0]), isSOC=False):
    assert(np.all(k == 0))
    SK = np.zeros([s.no_u, s.no_u], dtype=np.float32)

    s.listh -= 1
    s.listhptr = np.append(s.listhptr, s.nh) # listhptr start from 0 in fortran!
    # xijs = [] # shape(3, no_u, no_s)
    Sovers = csr((s.Sover, s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s))
    # for tmp in range(0, 3):
    #   xijs.append(csr((s.xij[:,tmp], s.listh, s.listhptr), dtype=np.float32, shape=(s.no_u,s.no_s)))

    for jsuper in range(0, s.no_s, s.no_u):
      Sover:csr = Sovers[:,jsuper:jsuper+s.no_u]
      SK += Sover
    return SK


if __name__ == '__main__':
  hsxfp = sys.argv[1]
  if not os.path.isfile(hsxfp):
    print('Please input HSX file path.')
    exit()
  fdffp = sys.argv[2]
  if not os.path.isfile(fdffp):
    print('Please input FDF file path.')
    exit()
  # hsxfp = r'C:\Users\zhang\UserSpace\Github\HamNetold\old\HSX'
  # fdffp = r'C:\Users\zhang\UserSpace\Github\HamNetold\old\Si_uc.fdf'

  fdf = FDF(file=fdffp)
  hsx = HSX(file=hsxfp)
  graph = hsx.getGraph2(fdf, tojson=True)
  # print(len(graph['inv_edge_idx']))
  # plt.yscale('log')
  # plt.show()

  with open('HS.json', 'w') as f:
    json.dump(graph, f, separators=[',',':'])
  with open('HS.json', 'r') as f:
    file = f.read()
    file = re.sub(r', *"', ',\n"', file)
  with open('HS.json', 'w') as f:
    f.write(file)
