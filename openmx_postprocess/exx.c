/*----------------------------------------------------------------------
  exx.c
  
  M. Toyoda, 10 Nov. 2009.
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "exx.h"
#include "exx_log.h"
#include "exx_vector.h"
#include "exx_index.h"


struct EXX_Struct {
  /* unit cell */
  int     natom;    /* number of atoms */
  double *atom_rc;  /* confinement length of atoms */
  double *atom_v;   /* atomic positions */
  int    *atom_nb;  /* number of basis */
  double  pvec[9];  /* primitive translational vectors */
  int     nbmax;    /* max number of basis */

  /* parameters */
  double w_scr;     /* screening parameter */
  double rc_cut;    /* truncation length */

  /* overlaping pairs */
  int    nshell_op; /* number of shells */
  int    nop;       /* number of pairs */
  int   *op_atom1;  /* arrays for atom1 */
  int   *op_atom2;  /* arrays for atom2 */
  int   *op_cell;   /* arrays for cells */

  /* exchanging pairs */
  int    nshell_ep; /* number of shells */
  int    nep;       /* number of pairs */
  int   *ep_atom1;  /* arrays for atom1 */
  int   *ep_atom2;  /* arrays for atom2 */
  int   *ep_cell;   /* arrays for cells */

  /* path */
  char path_cachedir[EXX_PATHLEN];
};  




int g_exx_skip1 = 0;
int g_exx_skip2 = 0;
char g_exx_cachedir[EXX_PATHLEN];
int g_exx_liberi_lmax = 20;
int g_exx_liberi_ngrid = 1024;
int g_exx_liberi_ngl = 100;
double g_exx_rc_cut = -1.0;
double g_exx_w_scr = -1.0;



/*----------------------------------------------------------------------
  EXX_New

  IN:
    natom   : number of atoms 
    atom_rc : confinement length of atoms 
    atom_v  : atomic positions 
    pvec    : primitive translational vectors 
    w_scr   : screening parameter 
    rc_cut  : truncation length 

  OUT:
    return  : pointer to EXX object
----------------------------------------------------------------------*/
EXX_t* EXX_New(
  int           natom,
  const double *atom_v,
  const int    *atom_sp,
  int           nspec,
  const double *spec_rc,
  const int    *spec_nb,
  const double *pvec,
  double        w_scr,
  double        rc_cut,
  int           system_type,
  const char   *cachedir
)
{
  int i, j, nshell_op, nshell_ep, nop, nep, nbmax;
  EXX_t *self;

  self = (EXX_t*)malloc(sizeof(EXX_t));
 
  /* max number of basis */ 
  nbmax = 0; 
  for (i=0; i<nspec; i++) { 
    if (spec_nb[i] > nbmax) { nbmax = spec_nb[i]; }
  }
  
  /* allocate memory */
  self->atom_rc  = (double*)malloc(sizeof(double)*natom);
  self->atom_v   = (double*)malloc(sizeof(double)*natom*3);
  self->atom_nb  = (int*)malloc(sizeof(int)*natom);

  /* copy data */
  self->natom = natom;
  self->nbmax = nbmax;
  for (i=0; i<natom; i++) {
    self->atom_v[3*i+0] = atom_v[3*i+0];
    self->atom_v[3*i+1] = atom_v[3*i+1];
    self->atom_v[3*i+2] = atom_v[3*i+2];
    j = atom_sp[i];
    self->atom_rc[i] = spec_rc[j];
    self->atom_nb[i] = spec_nb[j];
  }

  /* number of shells for OP and EP */
  switch (system_type) {
  case EXX_SYSTEM_PERIODIC:
    nshell_op = 
      EXX_Index_OP_NShell(natom, self->atom_rc, self->atom_v, pvec);
    nshell_ep = 
      EXX_Index_EP_NShell(natom, self->atom_rc, self->atom_v, pvec, rc_cut);
    break;
  case EXX_SYSTEM_CLUSTER:
    nshell_op = 0;
    nshell_ep = 0;
    break;
  default:
    EXX_ERROR( "undefined system_type" );
    break;
  }
    
  /* number of pais */
  nop = EXX_Index_OP(natom, nshell_op, self->atom_rc, self->atom_v, 
                     pvec, NULL, NULL, NULL); 
  nep = EXX_Index_EP(natom, nshell_ep, self->atom_rc, self->atom_v, 
                     pvec, rc_cut, NULL, NULL, NULL);

  /* allocate memory */ 
  self->op_atom1 = (int*)malloc(sizeof(int)*nop);
  self->op_atom2 = (int*)malloc(sizeof(int)*nop);
  self->op_cell  = (int*)malloc(sizeof(int)*nop);
  self->ep_atom1 = (int*)malloc(sizeof(int)*nep);
  self->ep_atom2 = (int*)malloc(sizeof(int)*nep);
  self->ep_cell  = (int*)malloc(sizeof(int)*nep);

  /* enumurate pairs */
  EXX_Index_OP(natom, nshell_op, self->atom_rc, self->atom_v, pvec, 
               self->op_atom1, self->op_atom2, self->op_cell);
  EXX_Index_EP(natom, nshell_ep, self->atom_rc, self->atom_v, pvec, rc_cut,
               self->ep_atom1, self->ep_atom2, self->ep_cell);

  /* copy data */
  for (i=0; i<9; i++) { self->pvec[i] = pvec[i]; }
  self->w_scr  = w_scr;
  self->rc_cut = rc_cut;
  self->nshell_op = nshell_op;
  self->nop       = nop;
  self->nshell_ep = nshell_ep;
  self->nep       = nep;

  strncpy(self->path_cachedir, cachedir, EXX_PATHLEN);

  return self;
}




void EXX_Free(EXX_t * self)
{
  if (self) {
    if (self->atom_rc)  { free(self->atom_rc);  }
    if (self->atom_v)   { free(self->atom_v);   }
    if (self->atom_nb)  { free(self->atom_nb);  }
    if (self->op_atom1) { free(self->op_atom1); }
    if (self->op_atom2) { free(self->op_atom2); }
    if (self->op_cell)  { free(self->op_cell);  }
    if (self->ep_atom1) { free(self->ep_atom1); }
    if (self->ep_atom2) { free(self->ep_atom2); }
    if (self->ep_cell)  { free(self->ep_cell);  }
    free(self);
  }
}



int EXX_natom(const EXX_t *self)
{ 
  return self->natom; 
}

const double* EXX_atom_rc(const EXX_t *self)
{ 
  return self->atom_rc; 
}

const double* EXX_atom_v(const EXX_t *self) 
{ 
  return self->atom_v; 
}

const int* EXX_atom_nb(const EXX_t *self) 
{ 
  return self->atom_nb; 
}

const double* EXX_pvec(const EXX_t *self) 
{ 
  return self->pvec; 
}

double EXX_w_scr(const EXX_t *self) 
{ 
  return self->w_scr; 
}

double EXX_rc_cut(const EXX_t *self) 
{ 
  return self->rc_cut; 
}

int EXX_nbmax(const EXX_t *self) 
{ 
  return self->nbmax; 
}

int EXX_Number_of_OP_Shells(const EXX_t *self) 
{ 
  return self->nshell_op; 
}

int EXX_Number_of_OP(const EXX_t *self) 
{ 
  return self->nop;
}

const int* EXX_Array_OP_Atom1(const EXX_t *self)
{
  return self->op_atom1;
}

const int* EXX_Array_OP_Atom2(const EXX_t *self)
{
  return self->op_atom2;
}

const int* EXX_Array_OP_Cell(const EXX_t *self)
{
  return self->op_cell;
}

int EXX_Number_of_EP_Shells(const EXX_t *self)
{
  return self->nshell_ep;
}

int EXX_Number_of_EP(const EXX_t *self)
{
  return self->nep;
}

const int* EXX_Array_EP_Atom1(const EXX_t *self)
{
  return self->ep_atom1;
}

const int* EXX_Array_EP_Atom2(const EXX_t *self)
{
  return self->ep_atom2;
}

const int* EXX_Array_EP_Cell(const EXX_t *self)
{
  return self->ep_cell; 
}

const char* EXX_CacheDir(const EXX_t *self)
{
  return self->path_cachedir;
}



/*----------------------------------------------------------------------
  EXX_Find_OP

  Find OP index whose iatom1, iatom2, and icell are equivalent with 
  those specified. If index is not found, this returns -1.
----------------------------------------------------------------------*/
int EXX_Find_OP(
  const EXX_t *self,
  int iatom1,
  int iatom2,
  int iR_x,
  int iR_y,
  int iR_z
)
{
  int i, nop, nshell, ncd, iR_max, icell;  
  const int *op_atom1, *op_atom2, *op_cell;

  nop = EXX_Number_of_OP(self);
  nshell = EXX_Number_of_OP_Shells(self);
  ncd = 2*nshell+1;

  op_atom1 = EXX_Array_OP_Atom1(self);
  op_atom2 = EXX_Array_OP_Atom2(self);
  op_cell  = EXX_Array_OP_Cell(self);

  iR_max = abs(iR_x);
  if (abs(iR_y)>iR_max) { iR_max = abs(iR_y); }
  if (abs(iR_z)>iR_max) { iR_max = abs(iR_z); }
  if (iR_max>nshell) { return -1; }

  icell = (iR_x+nshell) + (iR_y+nshell)*ncd + (iR_z+nshell)*ncd*ncd;

  for (i=0; i<nop; i++) {
    if (iatom1==op_atom1[i] && iatom2==op_atom2[i] && icell==op_cell[i]) {
      return i;
    }
  }
  
  return -1;
}


/*----------------------------------------------------------------------
  EXX_Find_EP

  Find EP index whose iatom1, iatom2, and icell are equivalent with 
  those specified. If index is not found, this returns -1.
----------------------------------------------------------------------*/
int EXX_Find_EP(
  const EXX_t *self,
  int iatom1,
  int iatom2,
  int iR_x,
  int iR_y,
  int iR_z
)
{
  int  i, nep, nshell, ncd, iR_max, icell; 
  const int *ep_atom1, *ep_atom2, *ep_cell;

  nep = EXX_Number_of_EP(self);
  nshell = EXX_Number_of_EP_Shells(self);
  ncd = 2*nshell+1;

  ep_atom1 = EXX_Array_EP_Atom1(self);
  ep_atom2 = EXX_Array_EP_Atom2(self);
  ep_cell  = EXX_Array_EP_Cell(self);

  iR_max = abs(iR_x);
  if (abs(iR_y)>iR_max) { iR_max = abs(iR_y); }
  if (abs(iR_z)>iR_max) { iR_max = abs(iR_z); }
  if (iR_max>nshell) { return -1; }

  icell = (iR_x+nshell) + (iR_y+nshell)*ncd + (iR_z+nshell)*ncd*ncd;

  for (i=0; i<nep; i++) {
    if (iatom1==ep_atom1[i] && iatom2==ep_atom2[i] && icell==ep_cell[i]) {
      return i;
    }
  }
  
  return -1;
}



#if 0
/*---*/


static void quad(
  const EXX_t *xfm,
  int iatom1,
  int iatom2,
  int iatom3,
  int iatom4,
  int icell1,
  int icell2,
  int icell3,
  int *w,
  int q_ep1[8],
  int q_ep2[8],
  int q_cell[8],
  int *mul
)
{
  int nshell_op, ncd_op, ncell_op, ic0_op;
  int nshell_ep, ncd_ep, ncell_ep, ic0_ep;
  int f1, f2, f3;
  int iR1, imR1, iR2, imR2, iRd, imRd, iRdmR1, iR1mRd;
  int iR2pRd, imR2mRd, iR2mR1pRd, iR1mR2mRd;
  int vR1[3], vR2[3], vRd[3], v[3];
  int n;

  nshell_op = EXX_Number_of_OP_Shells(xfm);
  nshell_ep = EXX_Number_of_EP_Shells(xfm);

  ncd_op   = 2*nshell_op+1;
  ncell_op = ncd_op*ncd_op*ncd_op;
  ic0_op   = (ncell_op-1)/2;
  
  ncd_ep   = 2*nshell_ep+1;
  ncell_ep = ncd_ep*ncd_ep*ncd_ep;
  ic0_ep   = (ncell_ep-1)/2;

  /* weight factor */ 
  {
    f1 = EXX_Index_Cmp_Sites(iatom1, iatom2, icell1, nshell_op); /* i=j ? */
    f2 = EXX_Index_Cmp_Sites(iatom3, iatom4, icell2, nshell_op); /* k=l ? */
    
    f3 = EXX_Index_Cmp_OP(iatom1, iatom2, iatom3, iatom4,
      icell1, icell2, icell3, nshell_op, nshell_ep);
    
    /* fool-proof test */
    assert( f1 <= 0 );
    assert( f2 <= 0 );
    assert( f3 <= 0 );
  
    if (0==f1 && 0==f2 && 0==f3) { *w = 8; } 
    else if (0==f1 && 0==f2 && -1==f3) { *w = 4; }
    else if (-1==f1 && -1==f2 && -1==f3) { *w = 1; } 
    else { *w = 2; }
  }

  /* cell vectors */
  {  
    /* IN : icell1, icell2, icell3, nshell_op, nshell_ep 
       OUT: iR1, imR1, iR2, imR2, iRd, imRd, iRdmR1, iR1mRd 
            iR2pRd, imR2mRd, iR2mR1pRd, iR1mR2mRd 
       TMP: vR1, vR2, vR3, v */

    EXX_Index_Cell2XYZ(icell1, nshell_op, vR1);
    EXX_Index_Cell2XYZ(icell2, nshell_op, vR2);
    EXX_Index_Cell2XYZ(icell3, nshell_ep, vRd);

    /* R1 */
    v[0] = vR1[0];
    v[1] = vR1[1];
    v[2] = vR1[2];
    iR1 = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* -R1 */
    v[0] = -vR1[0];
    v[1] = -vR1[1];
    v[2] = -vR1[2];
    imR1 = EXX_Index_XYZ2Cell(v, nshell_ep);
 
    /* R2 */
    v[0] = vR2[0];
    v[1] = vR2[1];
    v[2] = vR2[2];
    iR2 = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* -R2 */
    v[0] = -vR2[0];
    v[1] = -vR2[1];
    v[2] = -vR2[2];
    imR2 = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* Rd */
    v[0] = vRd[0];
    v[1] = vRd[1];
    v[2] = vRd[2];
    iRd = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* -Rd */
    v[0] = -vRd[0];
    v[1] = -vRd[1];
    v[2] = -vRd[2];
    imRd = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* Rd-R1 */
    v[0] = vRd[0]-vR1[0];
    v[1] = vRd[1]-vR1[1];
    v[2] = vRd[2]-vR1[2];
    iRdmR1 = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* R1-Rd */
    v[0] = vR1[0]-vRd[0];
    v[1] = vR1[1]-vRd[1];
    v[2] = vR1[2]-vRd[2];
    iR1mRd = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* R2+Rd */
    v[0] = vR2[0]+vRd[0];
    v[1] = vR2[1]+vRd[1];
    v[2] = vR2[2]+vRd[2];
    iR2pRd = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* -R2-Rd */
    v[0] = -vR2[0]-vRd[0];
    v[1] = -vR2[1]-vRd[1];
    v[2] = -vR2[2]-vRd[2];
    imR2mRd = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* R2-R1+Rd */
    v[0] = vR2[0]-vR1[0]+vRd[0];
    v[1] = vR2[1]-vR1[1]+vRd[1];
    v[2] = vR2[2]-vR1[2]+vRd[2];
    iR2mR1pRd = EXX_Index_XYZ2Cell(v, nshell_ep);

    /* R1-R2-Rd */
    v[0] = vR1[0]-vR2[0]-vRd[0];
    v[1] = vR1[1]-vR2[1]-vRd[1];
    v[2] = vR1[2]-vR2[2]-vRd[2];
    iR1mR2mRd = EXX_Index_XYZ2Cell(v, nshell_ep);
  }

  n = 0;

  /* x1=(i,k,Rd), x2=(j,l,R2-R1+Rd), Rm=R1 */
  if (iRd>=0 && iR2mR1pRd>=0) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom1, iatom3, iRd);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom2, iatom4, iR2mR1pRd);
    q_cell[n] = iR1;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }
      
  /* x1=(j,k,Rd-R1), x2=(i,l,R2+Rd), Rm=-R1 */
  if (iRdmR1>=0 && iR2pRd>=0) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom2, iatom3, iRdmR1);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom1, iatom4, iR2pRd);
    q_cell[n] = imR1;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }
            
  /* x1=(i,l,R2+Rd), x2=(j,k,Rd-R1), Rm=R1 */
  if (iR2pRd>=0 && iRdmR1>=0) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom1, iatom4, iR2pRd);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom2, iatom3, iRdmR1);
    q_cell[n] = iR1;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }

  /* x1=(j,l,R2-R1+Rd), x2=(i,k,Rd), Rm=-R1 */
  if (iR2mR1pRd>=0 && iRd>=0) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom2, iatom4, iR2mR1pRd);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom1, iatom3, iRd);
    q_cell[n] = imR1;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }
            
  /* x1=(k,i,-Rd), x2=(l,j,R1-R2-Rd), Rm=R2 */
  if (imR1>=0 && iR1mR2mRd>=0) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom3, iatom1, imR1);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom4, iatom2, iR1mR2mRd);
    q_cell[n] = iR2;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }

  /* x1=(k,j,R1-Rd), x2=(l,i,-R2-Rd), Rm=R2 */
  if (iR1mRd>=0 && imR2mRd>=0) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom3, iatom2, iR1mRd);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom4, iatom1, imR2mRd);
    q_cell[n] = iR2;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }

  /* x1=(l,i,-R2-Rd), x2=(k,j,R1-Rd), Rm=-R2 */
  if (imR2mRd>=0 && iR1mRd) {
    q_ep1[n]  = EXX_Find_EP(xfm, iatom4, iatom1, imR2mRd);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom3, iatom2, iR1mRd);
    q_cell[n] = imR2;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }

  /* x1=(l,j,R1-R2-Rd), x2=(k,i,-Rd), Rm=-R2 */
  if (iR1mR2mRd>=0 && imRd>=0) { 
    q_ep1[n]  = EXX_Find_EP(xfm, iatom4, iatom2, iR1mR2mRd);
    q_ep2[n]  = EXX_Find_EP(xfm, iatom3, iatom1, imRd);
    q_cell[n] = imR2;
    assert(q_ep1[n] >= 0);
    assert(q_ep2[n] >= 0);
    assert(q_cell[n] >= 0);
    n++;
  }

  *mul = n;
}



int EXX_Make_Quartets(
  const EXX_t *xfm,
  int *q_op1, /* [nq] */
  int *q_op2, /* [nq] */
  int *q_opd, /* [nq] */
  int *q_wf,  /* [nq] */
  int *q_ep1, /* [nq*8] */
  int *q_ep2, /* [nq*8] */
  int *q_mul  /* [nq] */
)
{
  int i, f;
  int nshell_op, nshell_ep, ncd_ep, ncell_ep, ic0_ep, nq, nop;
  int iop1, iop2, iatom1, iatom2, iatom3, iatom4;
  int icell1, icell2, icell3;
  double rc1, rc2, rc3, rc4, rc12, rc34, d12, d34, d, cx12, cx34;
  double c1[3], c2[3], c3[3], c4[3], c12[3], c34[3], c34_off[3];
  double cc[3], cc_frac[3];
  int cell_xyz[3];
  int w, iep1[8], iep2[8], iRm[8], nmul;

  const int *op_atom1, *op_atom2, *op_cell;
  const double *atom_rc, *atom_v, *pvec;
  double rc_cut, x, y, z;
  int natom;
  int ncd_op, ncell_op, ic0_op;

  natom = EXX_natom(xfm);
  nshell_op = EXX_Number_of_OP_Shells(xfm);
  nshell_ep = EXX_Number_of_EP_Shells(xfm);
  nop       = EXX_Number_of_OP(xfm);

  op_atom1 = EXX_Array_OP_Atom1(xfm);
  op_atom2 = EXX_Array_OP_Atom2(xfm);
  op_cell  = EXX_Array_OP_Cell(xfm);

  atom_rc  = EXX_atom_rc(xfm);
  atom_v   = EXX_atom_v(xfm);
  pvec     = EXX_pvec(xfm);
  rc_cut   = EXX_rc_cut(xfm);

  ncd_op = 2*nshell_op+1;
  ncell_op = ncd_op*ncd_op*ncd_op;
  ic0_op = (ncell_op-1)/2;

  ncd_ep = 2*nshell_ep+1;
  ncell_ep = ncd_ep*ncd_ep*ncd_ep;
  ic0_ep = (ncell_ep-1)/2;

  nq = 0;

  for (iop1=0; iop1<nop; iop1++) {
    iatom1 = op_atom1[iop1];
    iatom2 = op_atom2[iop1];
    icell1 = op_cell[iop1];

    EXX_Vector_F2C(c1, &atom_v[3*iatom1], pvec);
    EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatom2], pvec, icell1, nshell_op);

    rc1 = atom_rc[iatom1];
    rc2 = atom_rc[iatom2];
    d12 = EXX_Vector_Distance(c1, c2);
    EXX_Vector_PAO_Overlap(rc1, rc2, d12, &rc12, &cx12);

    c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
    c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
    c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];

    for (iop2=0; iop2<nop; iop2++) {
      iatom3 = op_atom1[iop2];
      iatom4 = op_atom2[iop2];
      icell2 = op_cell[iop2];

      EXX_Vector_F2C(c3, &atom_v[3*iatom3], pvec);
      EXX_Vector_F2C_Offsite(c4, &atom_v[3*iatom4], pvec, icell2, nshell_op);

      rc3 = atom_rc[iatom3];
      rc4 = atom_rc[iatom4];
      d34 = EXX_Vector_Distance(c3, c4);
      EXX_Vector_PAO_Overlap(rc3, rc4, d34, &rc34, &cx34);

      c34[0] = cx34*c3[0] + (1.0-cx34)*c4[0];
      c34[1] = cx34*c3[1] + (1.0-cx34)*c4[1];
      c34[2] = cx34*c3[2] + (1.0-cx34)*c4[2];

      for (icell3=ic0_ep; icell3<ncell_ep; icell3++) {
        f = EXX_Index_Cmp_OP(iatom1, iatom2, iatom3, iatom4,
                             icell1, icell2, icell3, 
                             nshell_op, nshell_ep);
        if (f>0) { continue; }
       
        EXX_Index_Cell2Cartesian(icell3, nshell_ep, pvec, cc);
        
        c34_off[0] = c34[0] + cc[0];
        c34_off[1] = c34[1] + cc[1];
        c34_off[2] = c34[2] + cc[2];

        d = EXX_Vector_Distance(c12, c34_off);
        if (d > rc_cut + rc12 + rc34) { continue; }

        quad(xfm, iatom1, iatom2, iatom3, iatom4, 
             icell1, icell2, icell3, 
             &w, iep1, iep2, iRm, &nmul);

        if (q_op1) { q_op1[nq] = iop1; }
        if (q_op2) { q_op2[nq] = iop2; }
        if (q_opd) { q_opd[nq] = icell3; }
        if (q_wf)   { q_wf[nq]   = w; }
        for (i=0; i<nmul; i++) {
          if (q_ep1)  { q_ep1[8*nq+i]  = iep1[i]; }
          if (q_ep2)  { q_ep2[8*nq+i]  = iep2[i]; }
        }
        if (q_mul) { q_mul[nq] = nmul; }
       
        nq++;
      } /* icell3 */
    } /* iop2 */
  } /* iop1 */
 
  return nq;
}
#endif

