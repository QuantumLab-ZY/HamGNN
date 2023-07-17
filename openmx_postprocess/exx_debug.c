#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "exx.h"
#include "exx_debug.h"
#include "exx_log.h"


void EXX_Debug_Copy_DM(
  int MaxN,
  double *****CDM,
  EXX_t  *exx,
  dcomplex ****exx_CDM,
  int symbrk
)
{
  int spin, i, j, myrank, nproc, iproc, ib;
  int GA_AN, GB_AN, MA_AN, LB_AN;
  int iep, nep, iatm, nb, nb1, nb2, nbmax;
  const int *ep_atom1, *ep_atom2;
  double dm1, dm2, diff;
  double w, du, dd;

  int nbuf;
  double *buffer;
  MPI_Comm comm;

  nep  = EXX_Number_of_EP(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);

  /* MPI information */
  comm = g_exx_mpicomm;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nproc);

  /* max number of basis */
  nbmax = 0;
  for (i=1; i<=atomnum; i++) {
    nb = Spe_Total_CNO[WhatSpecies[i]];
    if (nb>nbmax) { nbmax=nb; }
  }

  /* allocate buffer */
  nbuf = nep*nbmax*nbmax;
  buffer = (double*)malloc(sizeof(double)*nbuf);

  for (spin=0; spin<=SpinP_switch; spin++){
 
    /* clear buffer */ 
    for (i=0; i<nbuf; i++) { buffer[i] = 0.0; }

    /* copy DM to buffer */
    for (iep=0; iep<nep; iep++) {
      GA_AN = ep_atom1[iep]+1;
      GB_AN = ep_atom2[iep]+1;

      nb1 = Spe_Total_CNO[WhatSpecies[GA_AN]];
      nb2 = Spe_Total_CNO[WhatSpecies[GB_AN]];

      /* find MA_AN */
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++) {
        if (GA_AN==M2G[MA_AN]) { break; }
      }
      if (GA_AN != M2G[MA_AN]) { continue; }

      /* find LB_AN */
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        if (GB_AN==natn[GA_AN][LB_AN]) { break; }
      }
      if (GB_AN!=natn[GA_AN][LB_AN]) { continue; }

      for (i=0; i<nb1; i++){
        for (j=0; j<nb2; j++){
          ib = iep*nbmax*nbmax + i*nbmax + j;
          buffer[ib] = CDM[spin][MA_AN][LB_AN][i][j];
        }
      }
    } /* loop of iep */

    /* all-to-all communication */
    MPI_Allreduce(buffer, buffer, nbuf, MPI_DOUBLE, MPI_SUM, comm);
   
    /* copy buffer to EXX-DM */
    for (iep=0; iep<nep; iep++) {
      GA_AN = ep_atom1[iep]+1;
      GB_AN = ep_atom2[iep]+1;
      nb1 = Spe_Total_CNO[WhatSpecies[GA_AN]];
      nb2 = Spe_Total_CNO[WhatSpecies[GB_AN]];
   
      /* symmetry breaking factor */
      if (SpinP_switch==1 && symbrk) {
        du = InitN_USpin[GA_AN]+ InitN_USpin[GB_AN];
        dd = InitN_DSpin[GA_AN]+ InitN_DSpin[GB_AN];
        w = (spin==0) ? (2.0*du/(dd + du)) : (2.0*dd/(dd+du));
      } else {
        w = 1.0;
      }

      for (i=0; i<nb1; i++){
        for (j=0; j<nb2; j++){
          ib = iep*nbmax*nbmax + i*nbmax + j;
          exx_CDM[spin][iep][i][j].r = w*buffer[ib];
          exx_CDM[spin][iep][i][j].i = 0.0;
        }
      }
    } /* loop of iep */
  } /* loop of spin */

  free(buffer);
}



void EXX_Initial_DM(
  EXX_t *exx,
  dcomplex ****exx_CDM
)
{
  int i, iep, nep, GA_AN, GB_AN, nb, spin;
  double den;
  const int *ep_atom1, *ep_atom2;

  nep = EXX_Number_of_EP(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);

  for (iep=0; iep<nep; iep++) {
    GA_AN = ep_atom1[iep]+1;
    GB_AN = ep_atom2[iep]+1;
    if (GA_AN != GB_AN) { continue; }
    nb = Spe_Total_CNO[WhatSpecies[GA_AN]];
    
    den = InitN_USpin[GA_AN]/nb;
    for (i=0; i<nb; i++){
      exx_CDM[spin][iep][i][i].r = den;
      exx_CDM[spin][iep][i][i].i = 0.0;
    }

    if (0<SpinP_switch) {
      den = InitN_DSpin[GA_AN]/nb;
      for (i=0; i<nb; i++){
        exx_CDM[spin][iep][i][i].r = den;
        exx_CDM[spin][iep][i][i].i = 0.0;
      }
    }
  } /* loop of iep */
}



#if 0
static find_ep(EXX_t *exx, int iatm1, int iatm2, int Rx, int Ry, int Rz)
{
  int nep, iep, nshell, ncd, ia1, ia2, icell, ix, iy, iz;
  const int *ep_atom1, *ep_atom2, *ep_cell;

  nep = EXX_Number_of_EP(exx);
  nshell = EXX_Number_of_EP_Shells(exx);
  ncd = 2*nshell+1;

  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
  ep_cell  = EXX_Array_EP_Cell(exx);

  for (iep=0; iep<nep; iep++) {
    ia1 = ep_atom1[iep];
    ia1 = ep_atom2[iep];
    icell = ep_cell[iep]; 
    ix = icell%ncd - nshell;
    iy = (icell/ncd)%ncd - nshell;
    iz = (icell/ncd/ncd)%ncd - nshell;
    if (ia1==iatm1 && ia2== iatm2 
        && ix==Rx && iy==Ry && iz==Rz) { return iep; }
  }

  return -1;
}
#endif


void EXX_Debug_Check_DM(
  EXX_t *exx, 
  dcomplex ****exx_DM, 
  double *****DM
)
{
  int i, j, spin, MA_AN, GA_AN, Anum, LB_AN, GB_AN;
  int nb1, nb2;
  int ia1, ia2, l1, l2, l3, iep, Rn;
  double err, maxerr, x, y;

  maxerr = 0.0;
 
  for (spin=0; spin<=SpinP_switch; spin++){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      nb1   = Spe_Total_CNO[WhatSpecies[GA_AN]];
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	nb2   = Spe_Total_CNO[WhatSpecies[GB_AN]];  
        Rn    = ncn[GA_AN][LB_AN];

        l1 = atv_ijk[Rn][1];
        l2 = atv_ijk[Rn][2];
        l3 = atv_ijk[Rn][3];

        /*iep = find_ep(exx, GA_AN-1, GB_AN-1, l1, l2, l3);*/
        iep = EXX_Find_EP(exx, GA_AN-1, GB_AN-1, l1, l2, l3);

        if (-1==iep) { EXX_ERROR("iep not found"); }
 
	for (i=0; i<nb1; i++){
	  for (j=0; j<nb2; j++){
            x = DM[spin][MA_AN][LB_AN][i][j];
            y = exx_DM[spin][iep][i][j].r;
            err = fabs(x-y);
            if (err>1e-8) { 
              EXX_WARN("err too large");
              fprintf(stderr, "   DM= %20.12f\n", 
                DM[spin][MA_AN][LB_AN][i][j]);
              fprintf(stderr, "  XDM= %20.12f %20.12f\n", 
                exx_DM[spin][iep][i][j].r, exx_DM[spin][iep][i][j].i);
            }
            if (err>maxerr) { maxerr = err; }
          }
        }
      } /* loop of LB_AN */
    } /* loop of MA_AN */
  } /* loop of spin */

  if (maxerr < 1e-10) {
    fprintf(stderr, "DM-Check passed (maxerr=%20.12f)\n", maxerr);
  } else {
    fprintf(stderr, "DM-Check failed (maxerr=%20.12f)\n", maxerr);
  }
}
