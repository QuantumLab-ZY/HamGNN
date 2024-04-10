'''
Author: Changwei Zhang 
Date: 2023-05-20 15:23:31 
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-20 15:23:31 
'''

import numpy as np
from numba import njit
from copy import deepcopy
import json
import os
import re
import sys
from scipy.sparse import csr_matrix as csr
from pymatgen.core.periodic_table import Element

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
def convComplex(x):
  def convComplexRe(x):
    if type(x) == list:
      for idx in range(len(x)):
        x[idx] = convComplexRe(x[idx])
    elif type(x) == np.ndarray:
      return x.real.tolist()
    elif type(x) == np.complex64 or type(x) == np.complex128:
      return float(x.real)
    return x
  def convComplexIm(x):
    if type(x) == list:
      for idx in range(len(x)):
        x[idx] = convComplexIm(x[idx])
    elif type(x) == np.ndarray:
      return x.imag.tolist()
    elif type(x) == np.complex64 or type(x) == np.complex128:
      return float(x.imag)
    return x
  x_re = deepcopy(x)
  x_im = deepcopy(x)
  return convComplexRe(x_re), convComplexIm(x_im)
  

class STRU:
  '''
  @property: nspecies: int.
  @property: na_u: int. natoms in unit cell
  @property: species: List[str]
  @property: no: List. norbital per species. shape: (nspecies)
  @property: na_s: List. natoms per species. shape: (nspecies)
  @property: cell: ndarray
  @property: pos: ndarray
  @property: z: ndarray. Z number of each atom. shape: (natoms)
  '''

  def __init__(self, file:str) -> None:      
    self.fp = open(file)
    self.species = []
    self.no = [] # norbital per species. shape: (nspecies)
    self.na_s = [] # natoms per species.
    self.cell = []
    self.pos = []
    activeBlock = None
    while True:
      line = self.fp.readline().split('//')[0].split('#')[0]
      if not line:
        break
      elif 'ATOMIC_SPECIES' in line:
        activeBlock = 'ATOMIC_SPECIES'
        continue
      elif 'NUMERICAL_ORBITAL' in line:
        activeBlock = 'NUMERICAL_ORBITAL'
        continue
      elif 'LATTICE_CONSTANT' in line:
        activeBlock = 'LATTICE_CONSTANT'
        continue
      elif 'LATTICE_VECTORS' in line:
        activeBlock = 'LATTICE_VECTORS'
        continue
      elif 'ATOMIC_POSITIONS' in line:
        activeBlock = 'ATOMIC_POSITIONS'
        continue
      elif line.strip() == '': # blank lines
        continue

      if activeBlock == 'ATOMIC_SPECIES':
        self.species.append(line.split()[0])
      elif activeBlock == 'NUMERICAL_ORBITAL':
        tmp = line.split('.orb')[0].split('_')[-1]
        splitnorm = re.findall(r'\d', tmp)
        s = int(splitnorm[0])*1 if 's' in tmp else 0
        p = int(splitnorm[1])*3 if 'p' in tmp else 0
        d = int(splitnorm[2])*5 if 'd' in tmp else 0
        f = int(splitnorm[3])*7 if 'f' in tmp else 0
        self.no.append(s+p+d+f)
      elif activeBlock == 'LATTICE_CONSTANT':
        latconst = float(line)
      elif activeBlock == 'LATTICE_VECTORS':
        tmp = [float(i) for i in line.split()]
        self.cell.append(tmp)
      elif activeBlock == 'ATOMIC_POSITIONS':
        posType = line.strip().lower()
        for is_ in range(len(self.no)):
          # element
          while True:
            line = self.fp.readline().split('//')[0].split('#')[0]
            if line.strip() == '':
              continue
            break
          # mag
          while True:
            line = self.fp.readline().split('//')[0].split('#')[0]
            if line.strip() == '':
              continue
            break
          # num
          while True:
            line = self.fp.readline().split('//')[0].split('#')[0]
            if line.strip() == '':
              continue
            na = int(line)
            self.na_s.append(na)
            break
          # pos
          ia = 0
          while ia < na:
            line = self.fp.readline().split('//')[0].split('#')[0]
            if line.strip() == '':
              continue
            tmp = line.split()
            self.pos.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
            ia += 1

    self.nspecies = len(self.species)
    self.na_u = sum(self.na_s)
    self.cell = np.array(self.cell) * latconst
    if posType == 'cartesian':
      self.pos = np.array(self.pos) * latconst
    elif posType == 'direct':
      cart = np.zeros_like(self.pos)
      for i, pos in enumerate(self.pos):
        cart[i] = pos @ self.cell
      self.pos = cart
    else:
      raise NotImplementedError
    z = []
    for spec, na in zip(self.species, self.na_s):
      z += [Element(spec).Z] * na
    self.z = np.array(z, dtype=int)
    
    self.fp.close()

class ABACUSHS:

  def __init__(s, file:str) -> None:
    s.fp = open(file)
    line = s.fp.readline() # STEP: 0
    if 'STEP' in line:
      s.no_u = int(s.fp.readline().split()[-1])
    else:
      s.no_u = int(line.split()[-1]) # number of orbitals in the unit cell.
    s.ncell_shift = int(s.fp.readline().split()[-1])

  def getGraph(s, stru:STRU, graph:dict={}, skip=False, isH=False, isSOC=False, calcRcut=False, tojson=False):
    '''
    -> edge_index: ndarray
    -> inv_edge_idx: ndarray
    -> cell_shift: ndarray
    -> nbr_shift: ndarray
    -> pos: ndarray
    -> Hon: List[List[ndarray]] shape [nspin, natoms, nham]
    -> Hoff: List[List[ndarray]] shape [nspin, nedge, nham]
    '''
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
    dtype = np.float32 if not isSOC else np.complex64
    repeat = 1 if not isSOC else 2
    nspin = 1 if not isSOC else 4
    edge_idx_src = []
    edge_idx_dst = []
    cell_shift   = []
    nbr_shift    = []
    Hon = [[]] if not isSOC else [[],[],[],[]] # cannot written as [[]]*4
    Hoff= [[]] if not isSOC else [[],[],[],[]]

    if skip:
      graph_ = deepcopy(graph)
      s.noff = len(graph_['inv_edge_idx'])
      edge_idx_src = graph_['edge_index'][0]
      edge_idx_dst = graph_['edge_index'][1]
      cell_shift = graph_['cell_shift']
      Hoff = graph_['Hoff']
      for ispin in range(nspin):
        for ioff in range(s.noff):
          Hoff[ispin][ioff] = np.zeros_like(Hoff[ispin][ioff], dtype=dtype)

    no = []
    for is_ in range(stru.nspecies):
      no += [stru.no[is_]] * stru.na_s[is_]
    no = np.array(no, dtype=int) * repeat # shape: (natoms)
    indo = np.zeros_like(no, dtype=int) # shape: (natoms)
    indo[1:] = np.cumsum(no[:-1])
    if no.sum() != s.no_u:
      print('STRU parse error!')
      exit()

    while True:
      line = s.fp.readline()
      if not line:
        break
      tmp = line.split()
      cx, cy, cz = int(tmp[0]), int(tmp[1]), int(tmp[2])
      nh = int(tmp[3])
      if nh == 0:
        continue
      val = s.fp.readline()
      col = s.fp.readline().split()
      row = s.fp.readline().split()
      if not isSOC:
        val = list(map(float, val.split()))
      else:
        val_raw = re.findall('[\-\+\d\.eE]+', val)
        val_raw = np.asarray(val_raw, dtype=np.float32)
        val = np.zeros(len(val_raw)//2, dtype=np.complex64)
        val += val_raw[0::2] + 1j * val_raw[1::2]
      col = list(map(int, col))
      row = list(map(int, row))
      hamilt = csr((val, col, row), shape=[s.no_u, s.no_u], dtype=dtype) 
      if isH:
        hamilt *= ry2ha

      for ia in range(stru.na_u):
        for ja in range(stru.na_u):
          ham:csr = hamilt[indo[ia]:indo[ia]+no[ia], indo[ja]:indo[ja]+no[ja]]
          if ia == ja and cx == 0 and cy == 0 and cz == 0:
            # onsite
            if not isSOC:
              Hon[0].append(ham.toarray().flatten())
            else:
              Hon[0].append(ham[0::2,0::2].toarray().flatten()) # uu
              if isH:
                Hon[1].append(ham[0::2,1::2].toarray().flatten()) # ud
                Hon[2].append(ham[1::2,0::2].toarray().flatten()) # du
                Hon[3].append(ham[1::2,1::2].toarray().flatten()) # dd
          elif ham.getnnz() > 0:
            # offsite
            if not skip:
              if not isSOC:
                Hoff[0].append(ham.toarray().flatten())
              else:
                Hoff[0].append(ham[0::2,0::2].toarray().flatten()) # uu
                if isH:
                  Hoff[1].append(ham[0::2,1::2].toarray().flatten()) # ud
                  Hoff[2].append(ham[1::2,0::2].toarray().flatten()) # du
                  Hoff[3].append(ham[1::2,1::2].toarray().flatten()) # dd
              edge_idx_src.append(ia)
              edge_idx_dst.append(ja)
              cell_shift.append(np.array([cx,cy,cz], dtype=int))
              nbr_shift.append(np.array([cx,cy,cz]) @ stru.cell)
            else:
              ierr, ioff = fillHoff(s.noff, cx, cy, cz, ia, ja,
                                    edge_idx_src, 
                                    edge_idx_dst, 
                                    cell_shift)
              if ierr:
                continue 
                # if range(H0) < range(H), extra edges should be ignored.
                # if range(H0) >=range(H), ierr=True should never happen.
                print('Something went wrong!')
                exit()
              if not isSOC:
                Hoff[0][ioff] = ham.toarray().flatten()
              else:
                Hoff[0][ioff] = ham[0::2,0::2].toarray().flatten() # uu
                if isH:
                  Hoff[1][ioff] = ham[0::2,1::2].toarray().flatten() # ud
                  Hoff[2][ioff] = ham[1::2,0::2].toarray().flatten() # du
                  Hoff[3][ioff] = ham[1::2,1::2].toarray().flatten() # dd
    #######################################################
    if calcRcut:
      s.max_rcut = np.zeros((stru.nspecies, stru.nspecies))
      isa = np.zeros(stru.na_u, dtype=int) # shape (natoms,)
      num = 0
      for is_ in range(stru.nspecies):
        for ia in range(stru.na_s[is_]):
          isa[num] = is_
          num += 1
      for ia, ja, cs in zip(edge_idx_src, edge_idx_dst, cell_shift):
        # to speed the calculation, ignore all the non-diagnal terms.
        if isa[ia] != isa[ja]:
          continue
        distance = np.linalg.norm(stru.pos[ja] - stru.pos[ia] + (cs @ stru.cell.T))
        s.max_rcut[isa[ia], isa[ja]] = max(distance, s.max_rcut[isa[ia], isa[ja]])
        s.max_rcut[isa[ja], isa[ia]] = max(distance, s.max_rcut[isa[ja], isa[ia]])
      # np.savetxt('RCUT', s.max_rcut)
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
      graph_['edge_index'] = edge_index if tojson else np.array(edge_index)
      graph_['inv_edge_idx'] = convInt(inv_edge_idx) if tojson else inv_edge_idx
      graph_['cell_shift'] = convInt(cell_shift) if tojson else np.array(cell_shift)
      graph_['nbr_shift'] = convFloat(nbr_shift) if tojson else np.array(nbr_shift)
      graph_['pos'] = convFloat(stru.pos) if tojson else stru.pos
    if not tojson:
      graph_['Hon'] = Hon
      graph_['Hoff'] = Hoff
    else:
      if not isSOC:
        graph_['Hon'] = convFloat(Hon)
        graph_['Hoff'] = convFloat(Hoff)
      else:
        graph_['Hon'], graph_['iHon'] = convComplex(Hon)
        graph_['Hoff'], graph_['iHoff'] = convComplex(Hoff)
    return graph_

  def getHK(s, stru:STRU, k=np.array([0,0,0]), isH=False, isSOC=False):
    '''
    return HK shape: [no_u,no_u]
    '''
    assert(np.all(k == 0)) # only support gamma now!
    dtype = np.float32 if not isSOC else np.complex64
    HK = np.zeros([s.no_u, s.no_u], dtype=dtype)

    while True:
      line = s.fp.readline()
      if not line:
        break
      tmp = line.split()
      cx, cy, cz = int(tmp[0]), int(tmp[1]), int(tmp[2])
      nh = int(tmp[3])
      if nh == 0:
        continue
      val = s.fp.readline()
      col = s.fp.readline().split()
      row = s.fp.readline().split()
      if not isSOC:
        val = list(map(float, val.split()))
      else:
        val_raw = re.findall('[\-\+\d\.eE]+', val)
        val_raw = np.asarray(val_raw, dtype=np.float32)
        val = np.zeros(len(val_raw)//2, dtype=np.complex64)
        val += val_raw[0::2] + 1j * val_raw[1::2]
      col = list(map(int, col))
      row = list(map(int, row))
      hamilt = csr((val, col, row), shape=[s.no_u, s.no_u], dtype=dtype) 
      if isH:
        hamilt *= ry2ha

      HK += hamilt
    return HK

  def close(self):
    self.fp.close()


if __name__ == '__main__':
  poscar = STRU(os.path.join('../', r'STRU'))
  H = ABACUSHS(os.path.join('./', r'data-HR-sparse_SPIN0.csr'))
  graphH = H.getGraph(stru=poscar, graph={}, isH=True, tojson=True)
  S = ABACUSHS(os.path.join('./', r'data-SR-sparse_SPIN0.csr'))
  graphS = S.getGraph(stru=poscar, graph=graphH, skip=True, tojson=True)
  T = ABACUSHS(os.path.join('./', r'data-TR-sparse_SPIN0.csr'))
  graphT = T.getGraph(stru=poscar, graph=graphH, skip=True, tojson=True)
  H.close()
  S.close()
  T.close()
  for graph, fname in zip([graphH, graphT], ['HS.json', 'H0S.json']):
    g = deepcopy(graphH)
    g['Hon'] = [graph['Hon']]
    g['Hoff']= [graph['Hoff']]
    g['Son'] = graphS['Hon']
    g['Soff']= graphS['Hoff']

    with open(fname, 'w') as f:
      json.dump(g, f, separators=[',',':'])
    with open(fname, 'r') as f:
      file = f.read()
      file = re.sub(r', *"', ',\n"', file)
    with open(fname, 'w') as f:
      f.write(file)