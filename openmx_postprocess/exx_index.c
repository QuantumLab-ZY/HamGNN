/*----------------------------------------------------------------------
  exx_index.c

  Indexing 

  Coded by M. Toyoda, 10/NOV/2009
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "exx.h"
#include "exx_vector.h"
#include "exx_index.h"


/*----------------------------------------------------------------------
  EXX_Index_Cmp_Sites

  This defines relation between two off-site sites.
  SITE1 is given by atom-site {iatom1} at the central cell, and SITE2
  by atom-site {iatom2} at a cell {icell}.

  SITE1 is SMALLER than SITE2 if 
    ic0<icell OR (ic0==icell AND iatom1<iatom2)),
  SITE1 is LARGER than SITE2 if  
    ic0>icell OR (ic0==icell AND iatom1>iatom2)),
  and SITE1 is EQUAL to SITE2 if 
    ic0==icell AND iatom1==iatom2.

  This returns -1, +1, or 0 if SITE1 is smaller than, larger than, or 
  equal to SITE2, respectively.
----------------------------------------------------------------------*/
int EXX_Index_Cmp_Sites(
  int iatom1,
  int iatom2,
  int icell,
  int nshell
)
{
  int ncd, ncell, ic0;

  ncd   = 2*nshell+1;
  ncell = ncd*ncd*ncd;
  ic0   = (ncell-1)/2;

  if (icell>ic0) { 
    return -1;
  } else if (icell==ic0) {
    if (iatom1<iatom2) { 
      return -1;
    } else if (iatom1==iatom2) {
      return 0;
    }
  }

  return 1;
}



/*----------------------------------------------------------------------
  EXX_Index_Cmp_OP

  This defines relation between two OPs.
   
  SITE1 is given by atom-site {iatom1} at the central cell, and SITE2
  by atom-site {iatom2} at a cell {icell}.

  SITE1 is SMALLER than SITE2 if 
    ic0<icell OR (ic0==icell AND iatom1<iatom2)),
  SITE1 is LARGER than SITE2 if  
    ic0>icell OR (ic0==icell AND iatom1>iatom2)),
  and SITE1 is EQUAL to SITE2 if 
    ic0==icell AND iatom1==iatom2.

  This returns -1, +1, or 0 if SITE1 is smaller than, larger than, or 
  equal to SITE2, respectively.
----------------------------------------------------------------------*/
int EXX_Index_Cmp_OP(
  int iatom1,
  int iatom2,
  int iatom3,
  int iatom4,
  int icell1,
  int icell2,
  int icell3,
  int nshell_op,
  int nshell_ep
)
{
  int ncd_op, ncell_op, ic0_op;
  int ncd_ep, ncell_ep, ic0_ep;

  ncd_op   = 2*nshell_op+1;
  ncell_op = ncd_op*ncd_op*ncd_op;
  ic0_op   = (ncell_op-1)/2;
  
  ncd_ep   = 2*nshell_ep+1;
  ncell_ep = ncd_ep*ncd_ep*ncd_ep;
  ic0_ep   = (ncell_ep-1)/2;

  if (ic0_ep>icell3) { 
    return +1;
  } else if (ic0_ep<icell3) {
    return -1;
  } else {
    /* icell3==ic0_ep */
    if (iatom1>iatom3) { 
      return +1;
    } else if (iatom1<iatom3) {
      return -1;
    } else {
      /* iatom1==iatom3 */
      if (icell1>icell2) { 
        return +1;
      } else if (icell1<icell2) {
        return -1;
      } else {
        /* icell1==icell2 */
        if (iatom2>iatom4) {
          return +1; 
        } else if (iatom2<iatom4) {
          return -1;
        } else {
          /* iatom2==iatom4 */
          return 0; 
        }
      }
    }
  }
  fprintf(stderr, "***ERROR: %s (%d)\n", __FILE__, __LINE__);
  abort();
  return 0;
}
  



/*----------------------------------------------------------------------
  EXX_Index_Cell

  
----------------------------------------------------------------------*/
void EXX_Index_Cell2XYZ(
  int icell,
  int nshell,
  int c[3]
)
{
  int ncd;
  ncd = 2*nshell+1;
  c[0] = icell%ncd - nshell;
  c[1] = (icell/ncd)%ncd - nshell;
  c[2] = (icell/ncd/ncd)%ncd - nshell;
}


int EXX_Index_XYZ2Cell(
  int c[3],
  int nshell
)
{
  int ncd;
  ncd = 2*nshell+1;

  if (abs(c[0])>nshell || abs(c[1])>nshell || abs(c[2])>nshell) {
    return -1;
  }

  return (c[0]+nshell) + (c[1]+nshell)*ncd + (c[2]+nshell)*ncd*ncd;
}


void EXX_Index_Cell2Cartesian(
  int icell,
  int nshell,
  const double pvec[9],
  double c[3]
)
{
  int ncd;
  double cf[3];
  ncd = 2*nshell+1;
  cf[0] = (double)(icell%ncd - nshell);
  cf[1] = (double)((icell/ncd)%ncd - nshell);
  cf[2] = (double)((icell/ncd/ncd)%ncd - nshell);
  EXX_Vector_F2C(c, cf, pvec);
}


/*----------------------------------------------------------------------
  EXX_Index_OP_NShell

  Minimun number of shells for overlaping pairs (OPs).
  
  IN:
    natom   : number of atoms in the unit cell
    atom_rc : confinement length of atoms 
    atom_v  : position of atoms in fractional cooridnate
    pvec    : primitive translational vectors

  OUT:
    return  : minimum number of shells for OPs.
              If something wrong happens, return value is -1.

  Note:
    {atom_v} must be given in fractional coordinate. For example, if 
    the atom is at the body-center of the unit cell, then the {atom_v}
    for the atom should be {0.5, 0.5, 0.5}. 
----------------------------------------------------------------------*/
int EXX_Index_OP_NShell(
  int           natom,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec 
)
{
  int i, nop, nop0;

  /* number of OPs wthin the central cell */
  nop0 = EXX_Index_OP(natom, 0, atom_rc, atom_v, pvec, NULL, NULL, NULL);

  /* increase nshell untill the number of OPs converges */
  for (i=1; i<100; i++) {
    nop = EXX_Index_OP(natom, i, atom_rc, atom_v, pvec, NULL, NULL, NULL);
    if (nop0 == nop) { return i-1; } /* converged */
    nop0 = nop;
  }

  return -1; /* no convergence */
}




/*----------------------------------------------------------------------
  EXX_Index_EP_NShell

  Minimun number of shells for exchanging pairs (EPs).
  
  IN:
    natom   : number of atoms in the unit cell
    atom_rc : confinement length of atoms 
    atom_v  : position of atoms in fractional cooridnate
    pvec    : primitive translational vectors
    rc_cut  : truncation length

  OUT:
    return  : minimum number of shells for OPs.
              If something wrong happens, return value is -1.

  Note:
    {atom_v} must be given in fractional coordinate. For example, if 
    the atom is at the body-center of the unit cell, then the {atom_v}
    for the atom should be {0.5, 0.5, 0.5}. 
----------------------------------------------------------------------*/
int EXX_Index_EP_NShell(
  int           natom,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut
)
{
  int i, nep, nep0;

  /* number of OPs wthin the central cell */
  nep0 = EXX_Index_EP(natom, 0, atom_rc, atom_v, pvec, rc_cut,
                      NULL, NULL, NULL);

  /* increase nshell untill the number of OPs converges */
  for (i=1; i<100; i++) {
    nep = EXX_Index_EP(natom, i, atom_rc, atom_v, pvec, rc_cut,
                       NULL, NULL, NULL);
    if (nep0 == nep) { return i-1; } /* converged */
    nep0 = nep;
  }

  return -1; /* no convergence */
}




/*----------------------------------------------------------------------
  EXX_Index_OP
 
  List up OPs.
  Array for {op_atom1}, {op_atom2}, and {op_cell} must be of enough 
  length. One can know the required minimum length for the arrays by 
  calling this function with specifing NULL for {op_atom1}, {op_atom2},
  and {op_cell}.  

  A typical usage of this function is as follows:
  ---
    nshell = EXX_Index_OP_NShell(natom, atom_rc, atom_v, pvec);
    nop    = CCM_OP_Enum(natom, nshell, atom_rc, atom_v, pvec,
                         NULL, NULL, NULL);
    op_atom1 = (int*)malloc(sizeof(int)*nop);
    op_atom2 = (int*)malloc(sizeof(int)*nop);
    op_cell  = (int*)malloc(sizeof(int)*nop);
    CCM_OP_Enum(natom, nshell, atom_rc, atom_v, pvec,
                op_atom1, op_atom2, op_cell);
  ---

  IN:
    natom    : number of atoms in the unit cell.
    nshell   : number of shells to be considered.
    atom_rc  : confinement length of atoms.
    atom_v   : atomic positions in fractional coordinates.
    pvec     : primitive translational vectors.
  
  OUT:
    op_atom1 : array for atom1 of OPs (optional)
    op_atom2 : array for atom2 of OPs (optional)
    op_cell  : array for cells of OPs (optional)
    return   : number of OPs
----------------------------------------------------------------------*/
int EXX_Index_OP(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  int          *op_atm1, /* OUT, optional */
  int          *op_atm2, /* OUT, optional */
  int          *op_cell  /* OUT, optional */
)
{
  int iatm1, iatm2, icell, ncell, ncd, nop, ic0;
  double c1[3], c2[3], rc1, rc2, cc[3], x, y, z, d;
        
  /* number of cells to consider */
  ncd = 2*nshell+1;
  ncell = ncd*ncd*ncd;
  ic0 = (ncell-1)/2;
  nop = 0;

  for (iatm1=0; iatm1<natom; iatm1++) {
    /* atom 1 in the central cell */
    EXX_Vector_F2C(c1, &atom_v[3*iatm1], pvec);
    /* frac2cart(c1, &atom_v[3*iatm1], pvec, ic0, nshell); */
    rc1 = atom_rc[iatm1];

    for (icell=ic0; icell<ncell; icell++) {
      /* lattice cell */
  
      for (iatm2=0; iatm2<natom; iatm2++) {
        if (icell==ic0 && iatm2 < iatm1 ) { continue; }
        /* atom 2 in the lattice cell */
        EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatm2], pvec, icell, nshell);
        /* frac2cart(c2, &atom_v[3*iatm2], pvec, icell, nshell); */
        rc2 = atom_rc[iatm2];

        d = EXX_Vector_Distance(c1, c2);
        if (d > rc1+rc2) { continue; }
        
        if (op_atm1) { op_atm1[nop] = iatm1; }
        if (op_atm2) { op_atm2[nop] = iatm2; }
        if (op_cell) { op_cell[nop] = icell; }
        nop++;

      } /* loop of iatm2 */
    } /* loop of icell */
  } /* loop of iatm1 */
      
  return nop;
}





/*----------------------------------------------------------------------
  EXX_Index_EQ
 
  List up EQs.
  
  IN:
    natom    : number of atoms in the unit cell.
    nshell   : number of shells to be considered.
    atom_rc  : confinement length of atoms.
    atom_v   : atomic positions in fractional coordinates.
    pvec     : primitive translational vectors.
    rc_cut   : truncation length 
 
  OUT:
    eq_atom1 : array for atom1 of EQs (optional)
    eq_atom2 : array for atom2 of EQs (optional)
    eq_atom3 : array for atom3 of EQs (optional)
    eq_atom4 : array for atom4 of EQs (optional)
    eq_cell1 : array for cell1 of EQs (optional)
    eq_cell2 : array for cell2 of EQs (optional)
    eq_cell3 : array for cell3 of EQs (optional)
    return   : number of EQs
----------------------------------------------------------------------*/
int EXX_Index_EQ(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut,
  int          *eq_atom1,  /* OUT, optional */
  int          *eq_atom2,  /* OUT, optional */
  int          *eq_atom3,  /* OUT, optional */
  int          *eq_atom4,  /* OUT, optional */
  int          *eq_cell1,  /* OUT, optional */
  int          *eq_cell2,  /* OUT, optional */
  int          *eq_cell3   /* OUT, optional */
)
{
  int iatm1, iatm2, iatm3, iatm4, icell1, icell2, icell3;
  int neq, ncell, ncd, nshell_op, ncell_op, ncd_op, ic0_op;
  int iscr;
  double x, y, z, rc1, rc2, rc3, rc4, rc12, rc34, cx12, cx34;
  double c1[3], c2[3], c3[3], c4[3], c12[3], c34[3], c34_off[3];
  double cc1[3], cc2[3], cc3[3];
  double d12, d34, d;

  /* cells for OP */
  nshell_op = EXX_Index_OP_NShell(natom, atom_rc, atom_v, pvec);
  ncd_op    = 2*nshell_op+1; 
  ncell_op  = ncd_op*ncd_op*ncd_op;
  ic0_op = (ncell_op-1)/2;

  /* cells for EQ */
  ncd = 2*nshell+1;
  ncell = ncd*ncd*ncd;

  neq = 0;
 
  /* bare/screneed */
  iscr = (rc_cut>1e-10);

  for (iatm1=0; iatm1<natom; iatm1++) {
    /* atom 1 in the central cell */
    EXX_Vector_F2C(c1, &atom_v[3*iatm1], pvec);
    /* frac2cart(c1, &atom_v[3*iatm1], pvec, ic0_op, nshell_op); */
    rc1 = atom_rc[iatm1];

    for (icell1=0; icell1<ncell_op; icell1++) {
      /* lattice cell */
      
      for (iatm2=0; iatm2<natom; iatm2++) {
        /* atom 2 in the lattice cell */
        EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatm2], pvec, icell1, nshell_op);
        /* frac2cart(c2, &atom_v[3*iatm2], pvec, icell1, nshell_op); */
        rc2 = atom_rc[iatm2];

        /* d12 = distance(c1, c2); */
        d12 = EXX_Vector_Distance(c1, c2);
        if (d12 > rc1+rc2) { continue; }

        EXX_Vector_PAO_Overlap(rc1, rc2, d12, &rc12, &cx12);
        /* slo_pair(rc1, rc2, d12, &rc12, &cx12); */
        c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
        c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
        c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];

        for (iatm3=0; iatm3<natom; iatm3++) {
          /* atom 3 in the central cell */
          EXX_Vector_F2C(c3, &atom_v[3*iatm3], pvec);
          /* frac2cart(c3, &atom_v[3*iatm3], pvec, ic0_op, nshell_op); */
          rc3 = atom_rc[iatm3];

          for (icell2=0; icell2<ncell_op; icell2++) {
            /* lattice cell */
      
            for (iatm4=0; iatm4<natom; iatm4++) {
              /* atom 4 in the lattice cell */
              EXX_Vector_F2C_Offsite(c4, &atom_v[3*iatm4], pvec, 
                                     icell2, nshell_op);
              /* frac2cart(c4, &atom_v[3*iatm4], pvec, icell2, nshell_op); */
              rc4 = atom_rc[iatm4];

              /* d34 = distance(c3, c4); */
              d34 = EXX_Vector_Distance(c3, c4);
              if (d34 > rc3+rc4) { continue; }
        
              EXX_Vector_PAO_Overlap(rc3, rc4, d34, &rc34, &cx34);
              /* slo_pair(rc3, rc4, d34, &rc34, &cx34); */
              c34[0] = cx34*c3[0] + (1.0-cx34)*c4[0];
              c34[1] = cx34*c3[1] + (1.0-cx34)*c4[1];
              c34[2] = cx34*c3[2] + (1.0-cx34)*c4[2];

              for (icell3=0; icell3<ncell; icell3++) {
                /* lattice cell */
                x = (double)( icell3%ncd - nshell );
                y = (double)( (icell3/ncd)%ncd - nshell );
                z = (double)( (icell3/ncd/ncd)%ncd - nshell );
                cc3[0] = pvec[0]*x + pvec[3]*y + pvec[6]*z;
                cc3[1] = pvec[1]*x + pvec[4]*y + pvec[7]*z;
                cc3[2] = pvec[2]*x + pvec[5]*y + pvec[8]*z;
             
                c34_off[0] = c34[0] + cc3[0];
                c34_off[1] = c34[1] + cc3[1];
                c34_off[2] = c34[2] + cc3[2];

                if (iscr) { 
                  /* d = distance(c12, c34_off); */
                  d = EXX_Vector_Distance(c12, c34_off);
                  if (d > rc_cut + rc12 + rc34) { continue; }
                }
                
                if (eq_atom1) { eq_atom1[neq] = iatm1; }
                if (eq_atom2) { eq_atom2[neq] = iatm2; }
                if (eq_atom3) { eq_atom3[neq] = iatm3; }
                if (eq_atom4) { eq_atom4[neq] = iatm4; }
                if (eq_cell1) { eq_cell1[neq] = icell1; }
                if (eq_cell2) { eq_cell2[neq] = icell2; }
                if (eq_cell3) { eq_cell3[neq] = icell3; }
                
                neq++;
              }
            } /* iatm4 */
          } /* icell2 */
        } /* iatm3 */
      } /* iatm2 */
    } /* icell1 */
  } /* iatm1 */

  return neq;
}




/*----------------------------------------------------------------------
  EXX_Index_EP
 
  List up EQs.
  
  IN:
    natom    : number of atoms in the unit cell.
    nshell   : number of shells to be considered.
    atom_rc  : confinement length of atoms.
    atom_v   : atomic positions in fractional coordinates.
    pvec     : primitive translational vectors.
    rc_cut   : truncation length 

  OUT:
    ep_atom1 : array for atom1 of EPs (optional)
    ep_atom2 : array for atom2 of EPs (optional)
    ep_cell  : array for cells of EPs (optional)
    return   : number of EPs
----------------------------------------------------------------------*/
int EXX_Index_EP(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut,
  int          *ep_atom1, /* OUT, optional */
  int          *ep_atom2, /* OUT, optional */
  int          *ep_cell   /* OUT, optional */
)
{
  int iatm1, iatm2, iatm3, iatm4, icell1, icell2, icell3;
  int nshell_op, ncd_op, ncell_op, ic0_op, ncd, ncell, iflag, nep;
  int iscr;
  double x, y, z, c1[3], c2[3], c3[3], c4[3], cc3[3], c12[3], c34[3];
  double rc1, rc2, rc3, rc4, d12, d34, rc12, rc34, cx12, cx34;
  double c34_off[3], d;

  /* cells for OP */
  nshell_op = EXX_Index_OP_NShell(natom, atom_rc, atom_v, pvec);
  ncd_op    = 2*nshell_op+1; 
  ncell_op  = ncd_op*ncd_op*ncd_op;
  ic0_op = (ncell_op-1)/2;

  /* cells for EQ */
  ncd = 2*nshell+1;
  ncell = ncd*ncd*ncd;
  
  nep = 0;

  /* bare/screen */
  iscr = (rc_cut>1e-10);

  for (iatm1=0; iatm1<natom; iatm1++) {
    /* atom 1 in the central cell */
    EXX_Vector_F2C(c1, &atom_v[3*iatm1], pvec);
    /* frac2cart(c1, &atom_v[3*iatm1], pvec, ic0_op, nshell_op); */
    rc1 = atom_rc[iatm1];
        
    for (iatm3=0; iatm3<natom; iatm3++) {
      /* atom 3 in the central cell */
      EXX_Vector_F2C(c3, &atom_v[3*iatm3], pvec);
      /* frac2cart(c3, &atom_v[3*iatm3], pvec, ic0_op, nshell_op); */
      rc3 = atom_rc[iatm3];

      for (icell3=0; icell3<ncell; icell3++) {
        /* lattice cell */
        x = (double)( icell3%ncd - nshell );
        y = (double)( (icell3/ncd)%ncd - nshell );
        z = (double)( (icell3/ncd/ncd)%ncd - nshell );
        cc3[0] = pvec[0]*x + pvec[3]*y + pvec[6]*z;
        cc3[1] = pvec[1]*x + pvec[4]*y + pvec[7]*z;
        cc3[2] = pvec[2]*x + pvec[5]*y + pvec[8]*z;

        iflag = 0;

        for (icell1=0; icell1<ncell_op && 0==iflag; icell1++) {
          /* lattice cell */
      
          for (iatm2=0; iatm2<natom && 0==iflag; iatm2++) {
            /* atom 2 in the lattice cell */
            EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatm2], pvec, 
                                     icell1, nshell_op);
            /* frac2cart(c2, &atom_v[3*iatm2], pvec, icell1, nshell_op); */
            rc2 = atom_rc[iatm2];

            /* d12 = distance(c1, c2); */
            d12 = EXX_Vector_Distance(c1, c2);
            if (d12 > rc1+rc2) { continue; }

            EXX_Vector_PAO_Overlap(rc1, rc2, d12, &rc12, &cx12);
            /* slo_pair(rc1, rc2, d12, &rc12, &cx12); */
            c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
            c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
            c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];

            for (icell2=0; icell2<ncell_op && 0==iflag; icell2++) {
              /* lattice cell */
      
              for (iatm4=0; iatm4<natom && 0==iflag; iatm4++) {
                /* atom 4 in the lattice cell */
                EXX_Vector_F2C_Offsite(c4, &atom_v[3*iatm4], pvec, 
                                       icell2, nshell_op);
                /* frac2cart(c4, &atom_v[3*iatm4], pvec, icell2, nshell_op); */
                rc4 = atom_rc[iatm4];
  
                /* d34 = distance(c3, c4); */
                d34 = EXX_Vector_Distance(c3, c4);
                if (d34 > rc3+rc4) { continue; }
       
                if (iscr) { 
                  EXX_Vector_PAO_Overlap(rc3, rc4, d34, &rc34, &cx34);
                  /* slo_pair(rc3, rc4, d34, &rc34, &cx34); */
                  c34[0] = cx34*c3[0] + (1.0-cx34)*c4[0];
                  c34[1] = cx34*c3[1] + (1.0-cx34)*c4[1];
                  c34[2] = cx34*c3[2] + (1.0-cx34)*c4[2];
 
                  c34_off[0] = c34[0] + cc3[0];
                  c34_off[1] = c34[1] + cc3[1];
                  c34_off[2] = c34[2] + cc3[2];
 
                  /* d = distance(c12, c34_off); */
                  d = EXX_Vector_Distance(c12, c34_off);
                  if (d > rc_cut + rc12 + rc34) { continue; }
                }            
                iflag = 1;
              } /* iatm4 */
            } /* icell2 */
          } /* iatm2 */
        } /* icell1 */

        if (iflag) { 
          if (ep_atom1) { ep_atom1[nep] = iatm1; }
          if (ep_atom2) { ep_atom2[nep] = iatm3; }
          if (ep_cell)  { ep_cell[nep]  = icell3; }
          nep++; 
        }

      } /* icell3*/
    } /* iatm2 */
  } /* iatm1 */

  return nep; 
}



int EXX_Index_NQ_Full(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut
)
{
  int iatm1, iatm2, iatm3, iatm4, icell1, icell2, icell3, ic0;
  int nq, ncell, ncd, nshell_op, ncell_op, ncd_op, ic0_op;
  int iscr;
  double x, y, z, rc1, rc2, rc3, rc4, rc12, rc34, cx12, cx34;
  double c1[3], c2[3], c3[3], c4[3], c12[3], c34[3], c34_off[3];
  double cc1[3], cc2[3], cc3[3];
  double d12, d34, d;

  /* cells for OP */
  nshell_op = EXX_Index_OP_NShell(natom, atom_rc, atom_v, pvec);
  ncd_op    = 2*nshell_op+1; 
  ncell_op  = ncd_op*ncd_op*ncd_op;
  ic0_op = (ncell_op-1)/2;

  /* cells for EQ */
  ncd = 2*nshell+1;
  ncell = ncd*ncd*ncd;
  ic0   = (ncell-1)/2;

  nq = 0;

  /* bear/screened */
  iscr = (rc_cut>1e-10);

  for (iatm1=0; iatm1<natom; iatm1++) {
    /* atom 1 in the central cell */
    EXX_Vector_F2C(c1, &atom_v[3*iatm1], pvec);
    rc1 = atom_rc[iatm1];

    for (icell1=0; icell1<ncell_op; icell1++) {
      /* lattice cell */
      
      for (iatm2=0; iatm2<natom; iatm2++) {
        /* atom 2 in the lattice cell */

        EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatm2], pvec, icell1, nshell_op);
        rc2 = atom_rc[iatm2];

        /* d12 = distance(c1, c2); */
        d12 = EXX_Vector_Distance(c1, c2);
        if (d12 > rc1+rc2) { continue; }

        EXX_Vector_PAO_Overlap(rc1, rc2, d12, &rc12, &cx12);
        c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
        c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
        c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];

        for (iatm3=0; iatm3<natom; iatm3++) {
          /* atom 3 in the central cell */
          EXX_Vector_F2C(c3, &atom_v[3*iatm3], pvec);
          rc3 = atom_rc[iatm3];

          for (icell2=0; icell2<ncell_op; icell2++) {
            /* lattice cell */
      
            for (iatm4=0; iatm4<natom; iatm4++) {
         
              /* atom 4 in the lattice cell */
              EXX_Vector_F2C_Offsite(c4, &atom_v[3*iatm4], pvec, 
                                     icell2, nshell_op);
              rc4 = atom_rc[iatm4];

              /* d34 = distance(c3, c4); */
              d34 = EXX_Vector_Distance(c3, c4);
              if (d34 > rc3+rc4) { continue; }
        
              EXX_Vector_PAO_Overlap(rc3, rc4, d34, &rc34, &cx34);
              c34[0] = cx34*c3[0] + (1.0-cx34)*c4[0];
              c34[1] = cx34*c3[1] + (1.0-cx34)*c4[1];
              c34[2] = cx34*c3[2] + (1.0-cx34)*c4[2];

              for (icell3=0; icell3<ncell; icell3++) {
                /* lattice cell */
                if (iscr) { 
                  x = (double)( icell3%ncd - nshell );
                  y = (double)( (icell3/ncd)%ncd - nshell );
                  z = (double)( (icell3/ncd/ncd)%ncd - nshell );
                  cc3[0] = pvec[0]*x + pvec[3]*y + pvec[6]*z;
                  cc3[1] = pvec[1]*x + pvec[4]*y + pvec[7]*z;
                  cc3[2] = pvec[2]*x + pvec[5]*y + pvec[8]*z;
             
                  c34_off[0] = c34[0] + cc3[0];
                  c34_off[1] = c34[1] + cc3[1];
                  c34_off[2] = c34[2] + cc3[2];

                  d = EXX_Vector_Distance(c12, c34_off);
                  if (d > rc_cut + rc12 + rc34) { continue; }
                }
                
                nq++;
              }
            } /* iatm4 */
          } /* icell2 */
        } /* iatm3 */
      } /* iatm2 */
    } /* icell1 */
  } /* iatm1 */

  return nq;
}


 
int EXX_Index_NQ_Reduced(
  int           natom,
  int           nshell,
  const double *atom_rc,
  const double *atom_v,
  const double *pvec,
  double        rc_cut
)
{
  int iatm1, iatm2, iatm3, iatm4, icell1, icell2, icell3, ic0;
  int nq, ncell, ncd, nshell_op, ncell_op, ncd_op, ic0_op;
  int iscr;
  double x, y, z, rc1, rc2, rc3, rc4, rc12, rc34, cx12, cx34;
  double c1[3], c2[3], c3[3], c4[3], c12[3], c34[3], c34_off[3];
  double cc1[3], cc2[3], cc3[3];
  double d12, d34, d;
  int f;

  /* cells for OP */
  nshell_op = EXX_Index_OP_NShell(natom, atom_rc, atom_v, pvec);
  ncd_op    = 2*nshell_op+1; 
  ncell_op  = ncd_op*ncd_op*ncd_op;
  ic0_op = (ncell_op-1)/2;

  /* cells for EQ */
  ncd = 2*nshell+1;
  ncell = ncd*ncd*ncd;
  ic0   = (ncell-1)/2;

  nq = 0;

  /* bare/screened */
  iscr = (rc_cut>1e-10);
  
  for (iatm1=0; iatm1<natom; iatm1++) {
    /* atom 1 in the central cell */
    EXX_Vector_F2C(c1, &atom_v[3*iatm1], pvec);
    rc1 = atom_rc[iatm1];

    for (icell1=0; icell1<ncell_op; icell1++) {
      /* lattice cell */
      
      for (iatm2=0; iatm2<natom; iatm2++) {
        /* atom 2 in the lattice cell */
        f = EXX_Index_Cmp_Sites(iatm1, iatm2, icell1, nshell_op);
        if (f>0) { continue; }

        EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatm2], pvec, icell1, nshell_op);
        rc2 = atom_rc[iatm2];

        /* d12 = distance(c1, c2); */
        d12 = EXX_Vector_Distance(c1, c2);
        if (d12 > rc1+rc2) { continue; }

        EXX_Vector_PAO_Overlap(rc1, rc2, d12, &rc12, &cx12);
        c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
        c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
        c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];

        for (iatm3=0; iatm3<natom; iatm3++) {
          /* atom 3 in the central cell */
          EXX_Vector_F2C(c3, &atom_v[3*iatm3], pvec);
          rc3 = atom_rc[iatm3];

          for (icell2=0; icell2<ncell_op; icell2++) {
            /* lattice cell */
      
            for (iatm4=0; iatm4<natom; iatm4++) {
              f = EXX_Index_Cmp_Sites(iatm3, iatm4, icell2, nshell_op);
              if (f>0) { continue; }
         
              /* atom 4 in the lattice cell */
              EXX_Vector_F2C_Offsite(c4, &atom_v[3*iatm4], pvec, 
                                     icell2, nshell_op);
              rc4 = atom_rc[iatm4];

              /* d34 = distance(c3, c4); */
              d34 = EXX_Vector_Distance(c3, c4);
              if (d34 > rc3+rc4) { continue; }
        
              EXX_Vector_PAO_Overlap(rc3, rc4, d34, &rc34, &cx34);
              c34[0] = cx34*c3[0] + (1.0-cx34)*c4[0];
              c34[1] = cx34*c3[1] + (1.0-cx34)*c4[1];
              c34[2] = cx34*c3[2] + (1.0-cx34)*c4[2];

              for (icell3=0; icell3<ncell; icell3++) {
                f = EXX_Index_Cmp_OP(iatm1, iatm2, iatm3, iatm4,
                  icell1, icell2, icell3, nshell_op, nshell);
                if (f>0) { continue; }
               
                if (iscr) { 
                  /* lattice cell */
                  x = (double)( icell3%ncd - nshell );
                  y = (double)( (icell3/ncd)%ncd - nshell );
                  z = (double)( (icell3/ncd/ncd)%ncd - nshell );
                  cc3[0] = pvec[0]*x + pvec[3]*y + pvec[6]*z;
                  cc3[1] = pvec[1]*x + pvec[4]*y + pvec[7]*z;
                  cc3[2] = pvec[2]*x + pvec[5]*y + pvec[8]*z;
               
                  c34_off[0] = c34[0] + cc3[0];
                  c34_off[1] = c34[1] + cc3[1];
                  c34_off[2] = c34[2] + cc3[2];

                  d = EXX_Vector_Distance(c12, c34_off);
                  if (d > rc_cut + rc12 + rc34) { continue; }
                }
                
                nq++;
              }
            } /* iatm4 */
          } /* icell2 */
        } /* iatm3 */
      } /* iatm2 */
    } /* icell1 */
  } /* iatm1 */

  return nq;
}


 
