/*----------------------------------------------------------------------
  exx_interface_openmx.c:

  07/JAN/2010 Coded by M. Toyoda
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "exx.h"
#include "exx_log.h"
#include "exx_debug.h"
#include "exx_interface_openmx.h"
#include "exx_step1.h"
#include "exx_step2.h"
#include "exx_vector.h"
#include "exx_index.h"
#include "exx_file_eri.h"
#include "exx_rhox.h"

#define PATHLEN 256

static int g_step = 0;

MPI_Comm g_exx_mpicomm = MPI_COMM_NULL;

dcomplex *****g_exx_DM; 
double g_exx_U[2];
int g_exx_switch;
int g_exx_switch_output_DM;
int g_exx_switch_rhox;

static double time_step1;
static double time_step2;


/*----------------------------------------------------------------------
  Initialization of EXX for OpenMX
----------------------------------------------------------------------*/
void EXX_on_OpenMX_Init(MPI_Comm comm)
{
  int i, j, l, m, k, system_type, myrank, nproc;

  double pvec[9]; /* primitive translational vectors */
  double v[3];
  int natom, nspec;
  double *atom_v, *spec_rc;
  int *atom_sp, *spec_nb;
  double w_scr, rc_cut;

  int nbmax;
  int **spec_l, **spec_m, **spec_n, *spec_nmesh;
  double ****pao_fr, **pao_xr;
  
  int lmax, mul, nmesh, nb, nmul, nm;
  
  int iep, nep, Gc_AN, Cwan, tno0, Gh_AN, Hwan, tno1;
  const int *exx_atm1, *exx_atm2;

  double time0, time1;

  /* MPI information */
  g_exx_mpicomm = comm;
  MPI_Comm_size(g_exx_mpicomm, &nproc);
  MPI_Comm_rank(g_exx_mpicomm, &myrank);

  /* start logging */
  EXX_Log_Open();
  
#ifdef EXX_USE_MPI
  EXX_Log_Print("MPI MODE (NPROC= %d)\n", nproc);
#else
  EXX_Log_Print("SERIAL MODE\n");
#endif
  EXX_Log_Print("  MYRANK= %d\n", myrank);
  EXX_Log_Print("\n");

  /* solver type */ 
  switch (Solver) {
  case 2: /* cluster */
    EXX_Log_Print("SOLVER= CLUSTER\n");
    system_type = EXX_SYSTEM_CLUSTER;
    rc_cut = 0.0;
    w_scr  = 0.0;
    break;
  case 3: /* band */
    EXX_Log_Print("SOLVER= PERIODIC\n");
    system_type = EXX_SYSTEM_PERIODIC;
    rc_cut = 10.0 / BohrR; /* 10 Angstrom */
    w_scr  = 0.0;
    break;
  default:
    EXX_ERROR("EXX is available for 'CLUSTER' or 'BAND' solvers.");
  }

  /* user values */
  if (g_exx_rc_cut > 1e-10) { rc_cut = g_exx_rc_cut; }
  if (g_exx_w_scr > 1e-10) { w_scr = g_exx_w_scr; }

  if (EXX_SYSTEM_PERIODIC==system_type && rc_cut <1e-10) {
    EXX_ERROR("finite rc_cut is required for periodic mode.");
  } 

  /* swithces */
  EXX_Log_Print("SKIPPING STEP1 = %d\n", g_exx_skip1);
  EXX_Log_Print("SKIPPING STEP2 = %d\n", g_exx_skip2);
  EXX_Log_Print("CACHDIR= %s\n", g_exx_cachedir);
  EXX_Log_Print("LIBERI_LMAX= %d\n", g_exx_liberi_lmax);
  EXX_Log_Print("LIBERI_NGRID= %d\n", g_exx_liberi_ngrid);
  EXX_Log_Print("LIBERI_NGL= %d\n", g_exx_liberi_ngl);
  EXX_Log_Print("OUTPUT DM= %d\n", g_exx_switch_output_DM);
  EXX_Log_Print("\n");

  /* primitive translational vectors */
  pvec[0] = tv[1][1]; pvec[1] = tv[1][2]; pvec[2] = tv[1][3];
  pvec[3] = tv[2][1]; pvec[4] = tv[2][2]; pvec[5] = tv[2][3];
  pvec[6] = tv[3][1]; pvec[7] = tv[3][2]; pvec[8] = tv[3][3];

  natom = atomnum;
  nspec = SpeciesNum;

  /* allocation */ 
  atom_v = (double*)malloc(sizeof(double)*3*natom);
  atom_sp = (int*)malloc(sizeof(int)*natom);
  spec_rc = (double*)malloc(sizeof(double)*nspec);
  spec_nb = (int*)malloc(sizeof(int)*nspec);
 
  /* atomic positions */
  for (i=0; i<natom; i++) {
    v[0] = Gxyz[i+1][1];
    v[1] = Gxyz[i+1][2];
    v[2] = Gxyz[i+1][3];

    EXX_Vector_C2F(&atom_v[3*i], v, pvec);
  }

  /* species */
  for (i=0; i<natom; i++) { atom_sp[i] = WhatSpecies[i+1]; }

  /* cut-off radii */
  for (i=0; i<nspec; i++) { spec_rc[i] = Spe_Atom_Cut1[i]; }

  /* number of basis */
  nbmax = 0;
  for (i=0; i<nspec; i++) {
    spec_nb[i] = 0;
    for (l=0; l<=Spe_MaxL_Basis[i]; l++) {
      mul = Spe_Num_Basis[i][l];
      spec_nb[i] += mul * (2*l+1);
    }
    if (nbmax<spec_nb[i]) { nbmax = spec_nb[i]; }
  }
 
  /* PAO wavefunction */ 
  spec_l = (int**)malloc(sizeof(int*)*nspec);
  spec_m = (int**)malloc(sizeof(int*)*nspec);
  spec_n = (int**)malloc(sizeof(int*)*nspec);
  for (i=0; i<nspec; i++) {
    spec_l[i] = (int*)malloc(sizeof(int)*nbmax);
    spec_m[i] = (int*)malloc(sizeof(int)*nbmax);
    spec_n[i] = (int*)malloc(sizeof(int)*nbmax);
  }
  spec_nmesh = (int*)malloc(sizeof(int)*nspec);
  pao_fr = (double****)malloc(sizeof(double***)*nspec);
  pao_xr = (double**)malloc(sizeof(double*)*nspec);
  for (i=0; i<nspec; i++) { 
    lmax  = Spe_PAO_LMAX[i]+1;
    mul   = Spe_PAO_Mul[i];
    nmesh = Spe_Num_Mesh_PAO[i];
    pao_fr[i] = (double***)malloc(sizeof(double**)*lmax);
    for (l=0; l<lmax; l++) {
      nb = mul*(2*l+1);
      pao_fr[i][l] = (double**)malloc(sizeof(double*)*nb);
      for (m=0; m<nb; m++) {
        pao_fr[i][l][m] = (double*)malloc(sizeof(double)*nmesh);
      }
    }
    pao_xr[i] = (double*)malloc(sizeof(double)*nmesh);
  }

  /* rotating angular momentum quantum numbers */
  for (i=0; i<nspec; i++) {
    j = 0;
    for (l=0; l<=Spe_MaxL_Basis[i]; l++) {
      nmul = Spe_Num_Basis[i][l];
      nm = 2*l+1;
      for (mul=0; mul<nmul; mul++) {
        for (m=0; m<nm; m++) {
          spec_l[i][j] = l;
          if (l==1) {
            switch (m) {
            case 0: spec_m[i][j] = +1; break;
            case 1: spec_m[i][j] = -1; break;
            case 2: spec_m[i][j] = 0; break;
            default: 
              EXX_ERROR("");
            }
          } else if (l==2) {  
            switch (m) {
            case 0: spec_m[i][j] = 0; break;
            case 1: spec_m[i][j] = +2; break;
            case 2: spec_m[i][j] = -2; break;
            case 3: spec_m[i][j] = +1; break;
            case 4: spec_m[i][j] = -1; break;
            default: 
              EXX_ERROR("");
            }
          } else { 
            if (m==0) {
              spec_m[i][j] = 0;
            } else {
              if (0==m%2) {
                spec_m[i][j] = -(m+1)/2;
              } else {
                spec_m[i][j] = +(m+1)/2;
              }
            }
          }
          spec_n[i][j] = mul;
          j++;
        }
      }
    }
    spec_nmesh[i] = Spe_Num_Mesh_PAO[i];
    
    lmax  = Spe_PAO_LMAX[i]+1;
    mul   = Spe_PAO_Mul[i];
    nmesh = Spe_Num_Mesh_PAO[i];
 
    for (l=0; l<lmax; l++) {
      for (m=0; m<mul; m++) {
        for (k=0; k<nmesh; k++) {
          pao_fr[i][l][m][k] = Spe_PAO_RWF[i][l][m][k];
        }
      }
    }

    for (k=0; k<nmesh; k++) {
      pao_xr[i][k] = Spe_PAO_RV[i][k];
    } 
  }

  /* logging */
  EXX_Log_Print("Species:\n");
  for (i=0; i<nspec; i++) {
    for (j=0; j<spec_nb[i]; j++) {
      EXX_Log_Print("  %3d, %3d : %3d %3d %3d\n", 
        i, j, spec_l[i][j], spec_m[i][j], spec_n[i][j]);
    }
  }
  EXX_Log_Print("\n");

  MPI_Barrier(g_exx_mpicomm);

  g_exx = EXX_New(natom, atom_v, atom_sp, nspec, spec_rc, spec_nb, 
                pvec, w_scr, rc_cut, system_type, g_exx_cachedir); 

  if (Host_ID==myrank) { printf("<EXX_Init> EXX Initialized\n"); }
  
  EXX_Log_Print("EXX Initiatlized:\n");
  EXX_Log_Print("  natom= %d\n", EXX_natom(g_exx));
  EXX_Log_Print("  nbmax= %d\n", EXX_nbmax(g_exx));
  EXX_Log_Print("\n");

  EXX_Log_Print("Atoms:\n");
  EXX_Log_Print("   #  : X         Y         Z         RC        NB\n");
  for (i=0; i<EXX_natom(g_exx); i++) {
    EXX_Log_Print("  %3d : %8.4f  %8.4f  %8.4f  %8.4f  %2d\n",
      i, EXX_atom_v(g_exx)[3*i+0], EXX_atom_v(g_exx)[3*i+1], 
      EXX_atom_v(g_exx)[3*i+2], EXX_atom_rc(g_exx)[i], 
      EXX_atom_nb(g_exx)[i]);
  }
  EXX_Log_Print("\n");     

  EXX_Log_Print("Translational Vectors:\n");
  EXX_Log_Print("  a= %8.4f  %8.4f  %8.4f\n", 
    EXX_pvec(g_exx)[0], EXX_pvec(g_exx)[1], EXX_pvec(g_exx)[2]);
  EXX_Log_Print("  b= %8.4f  %8.4f  %8.4f\n",
    EXX_pvec(g_exx)[3], EXX_pvec(g_exx)[4], EXX_pvec(g_exx)[5]);
  EXX_Log_Print("  c= %8.4f  %8.4f  %8.4f\n", 
    EXX_pvec(g_exx)[6], EXX_pvec(g_exx)[7], EXX_pvec(g_exx)[8]);
  EXX_Log_Print("  w_scr= %8.4f\n", EXX_w_scr(g_exx));
  EXX_Log_Print("  rc_cut= %8.4f\n", EXX_rc_cut(g_exx));
  EXX_Log_Print("\n");

  EXX_Log_Print("Overlapping Pairs:\n");
  EXX_Log_Print("  nshell_op= %d\n", EXX_Number_of_OP_Shells(g_exx));
  EXX_Log_Print("  nop= %d\n", EXX_Number_of_OP(g_exx));
  EXX_Log_Print("   #  : A1  A2  CELL \n");
  for (i=0; i<EXX_Number_of_OP(g_exx); i++) {
    EXX_Log_Print("  %3d : %3d %3d %5d\n",
      i, EXX_Array_OP_Atom1(g_exx)[i], EXX_Array_OP_Atom2(g_exx)[i],
      EXX_Array_OP_Cell(g_exx)[i]);
  }
  EXX_Log_Print("\n");

  EXX_Log_Print("Exchanging Pairs:\n");
  EXX_Log_Print("  nshell_ep= %d\n", EXX_Number_of_EP_Shells(g_exx));
  EXX_Log_Print("  nep= %d\n", EXX_Number_of_EP(g_exx));
  EXX_Log_Print("   #  : A1  A2  CELL \n");
  for (i=0; i<EXX_Number_of_EP(g_exx); i++) {
    EXX_Log_Print("  %3d : %3d %3d %5d\n",
      i, EXX_Array_EP_Atom1(g_exx)[i], EXX_Array_EP_Atom2(g_exx)[i],
      EXX_Array_EP_Cell(g_exx)[i]);
  }
  EXX_Log_Print("\n");

  /* DM */
  exx_atm1 = EXX_Array_EP_Atom1(g_exx);
  exx_atm2 = EXX_Array_EP_Atom2(g_exx);
  nep      = EXX_Number_of_EP(g_exx);

  g_exx_DM = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[16]); 
  for (m=0; m<List_YOUSO[16]; m++){
    g_exx_DM[m] = (dcomplex****)malloc(sizeof(dcomplex***)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      g_exx_DM[m][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(nep));
      for (iep=0; iep<nep; iep++) {
        Gc_AN = exx_atm1[iep]+1;
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  

	Gh_AN = exx_atm2[iep]+1;
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_NO[Hwan];

        g_exx_DM[m][k][iep] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0); 
        for (i=0; i<tno0; i++){
	  g_exx_DM[m][k][iep][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno1); 
  	  for (j=0; j<tno1; j++) {
            g_exx_DM[m][k][iep][i][j].r = 0.0; 
            g_exx_DM[m][k][iep][i][j].i = 0.0; 
	  }
	}
      }
    }
  }

  /* STEP1: calculation of overlaps */ 

  dtime(&time0); 
  {
    EXX_Step1(g_exx, natom, atom_sp, nspec,
      spec_nb, spec_rc, spec_l, spec_m, spec_n,
      pao_fr, pao_xr, spec_nmesh);
  }
  dtime(&time1); 
  time_step1 = time1 - time0;

  if (Host_ID==myrank) { 
    printf("<EXX_Init> Overlaps (NOP=%5d)\n", EXX_Number_of_OP(g_exx));
  }

  /* STEP2: calculation of ERIs */

  dtime(&time0);
  {
    EXX_Step2(g_exx);
  }
  dtime(&time1);
  time_step2 = time1 - time0;

  if (Host_ID==myrank) { 
    printf("<EXX_Init> ERIs (NEP=%10d)\n", EXX_Number_of_EP(g_exx));
  }

  /* free memory */ 
  free(atom_v);
  free(atom_sp);
  free(spec_rc);
  free(spec_nb);
  
  for (i=0; i<nspec; i++) {
    free(spec_l[i]);
    free(spec_m[i]);
    free(spec_n[i]);
  }
  free(spec_l);
  free(spec_m);
  free(spec_n);
  free(spec_nmesh);
  
  for (i=0; i<nspec; i++) { 
    lmax  = Spe_PAO_LMAX[i]+1;
    mul   = Spe_PAO_Mul[i];
    for (l=0; l<lmax; l++) {
      for (m=0; m<mul; m++) {
        free(pao_fr[i][l][m]);
      }
      free(pao_fr[i][l]);
    }
    free(pao_fr[i]);
    free(pao_xr[i]);
  }
  free(pao_fr);
  free(pao_xr);
}


void EXX_Time_Report(void)
{
  printf("  EXX_Step1         = %10.5f\n",time_step1);
  printf("  EXX_Step2         = %10.5f\n",time_step2);
}


void EXX_on_OpenMX_Free(void)
{
  int m, k, iep, nep, i, Gc_AN, Cwan, tno0;
  const int *exx_atm1;
 
  exx_atm1 = EXX_Array_EP_Atom1(g_exx);
  nep      = EXX_Number_of_EP(g_exx);

  for (m=0; m<List_YOUSO[16]; m++){
    for (k=0; k<=SpinP_switch; k++){
      for (iep=0; iep<nep; iep++) {
        Gc_AN = exx_atm1[iep]+1;
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
        for (i=0; i<tno0; i++){
	  free(g_exx_DM[m][k][iep][i]);
	}
        free(g_exx_DM[m][k][iep]);
      }
      free(g_exx_DM[m][k]);
    }
    free(g_exx_DM[m]);
  }
  free(g_exx_DM);
  
  EXX_Free(g_exx);
  g_exx = NULL;

  EXX_Log_Close();
}


static int find_EP(
  EXX_t *exx,
  int ia1,
  int ia2,
  int r[3]
)
{
  int i, ic, nep, nshep;
  const int *ep_atom1, *ep_atom2, *ep_cell;

  nep      = EXX_Number_of_EP(exx);
  nshep    = EXX_Number_of_EP_Shells(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
  ep_cell  = EXX_Array_EP_Cell(exx);

  ic = EXX_Index_XYZ2Cell(r, nshep);

  for (i=0; i<nep; i++) {
    if (ia1==ep_atom1[i] && ia2==ep_atom2[i] && ic==ep_cell[i]) { break; }
  }

  if (i==nep) { 
    fprintf(stderr, "EP not found!\n");
    fprintf(stderr, "  ia1= %d\n", ia1);
    fprintf(stderr, "  ia2= %d\n", ia2);
    fprintf(stderr, "  ic = %d ( %3d %3d %3d)\n", ic, r[0], r[1], r[2]);
    abort();
  }

  return i;
}


static void EXX_OP2EP_Band(
  EXX_t *exx,
  int iop1, 
  int iop2,
  int icell,
  int *iep1, /* [8] */
  int *iep2, /* [8] */
  int *iRd,  /* [8] */
  int *mul
)
{
  int m, i, j, k, l, ic1, ic2;
  int nshop, nshep;
  int r1[3], r2[3], rd[3];
  int ep_r1[3], ep_r2[3], ep_rd[3];
  const int *op_atom1, *op_atom2, *op_cell;
  int Rd_is_0, R1_is_0, R2_is_0;

  nshop    = EXX_Number_of_OP_Shells(exx);
  nshep    = EXX_Number_of_EP_Shells(exx);

  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);
  op_cell  = EXX_Array_OP_Cell(exx);

  i = op_atom1[iop1]; 
  j = op_atom2[iop1]; 
  k = op_atom1[iop2]; 
  l = op_atom2[iop2]; 
  ic1 = op_cell[iop1];
  ic2 = op_cell[iop2];

  EXX_Index_Cell2XYZ(ic1, nshop, r1);
  EXX_Index_Cell2XYZ(ic2, nshop, r2);
  EXX_Index_Cell2XYZ(icell, nshep, rd);

  /* 0: (i,k,Rd), (j,l,R2+Rd-R1), R1 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = rd[m];
    ep_r2[m] = rd[m]+r2[m]-r1[m];
    ep_rd[m] = r1[m];
  }
  iep1[0] = find_EP(exx, i, k, ep_r1);
  iep2[0] = find_EP(exx, j, l, ep_r2);
  iRd[0]  = EXX_Index_XYZ2Cell(ep_rd, nshep);

  /* 1: (j,k,Rd-R1), (i,l,R2+Rd), -R1 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = rd[m]-r1[m];
    ep_r2[m] = rd[m]+r2[m];
    ep_rd[m] = -r1[m];
  }
  iep1[1] = find_EP(exx, j, k, ep_r1);
  iep2[1] = find_EP(exx, i, l, ep_r2);
  iRd[1]  = EXX_Index_XYZ2Cell(ep_rd, nshep);

  /* 2: (i,l,R2+Rd), (j,k,Rd-R1), R1 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = rd[m]+r2[m];
    ep_r2[m] = rd[m]-r1[m];
    ep_rd[m] = r1[m];
  }
  iep1[2] = find_EP(exx, i, l, ep_r1);
  iep2[2] = find_EP(exx, j, k, ep_r2);
  iRd[2]  = EXX_Index_XYZ2Cell(ep_rd, nshep);
  
  /* 3: (j,l,R2+Rd-R1), (i,k,Rd), -R1 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = rd[m]+r2[m]-r1[m];
    ep_r2[m] = rd[m];
    ep_rd[m] = -r1[m];
  }
  iep1[3] = find_EP(exx, j, l, ep_r1);
  iep2[3] = find_EP(exx, i, k, ep_r2);
  iRd[3]  = EXX_Index_XYZ2Cell(ep_rd, nshep);

  /* 4: (k,i,-Rd), (l,j,R1-R2-Rd), R2 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = -rd[m];
    ep_r2[m] = r1[m]-r2[m]-rd[m];
    ep_rd[m] = r2[m];
  }
  iep1[4] = find_EP(exx, k, i, ep_r1);
  iep2[4] = find_EP(exx, l, j, ep_r2);
  iRd[4]  = EXX_Index_XYZ2Cell(ep_rd, nshep);

  /* 5: (k,j,R1-Rd), (l,i,-R2-Rd), R2 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = r1[m]-rd[m];
    ep_r2[m] = -r2[m]-rd[m];
    ep_rd[m] = r2[m];
  }
  iep1[5] = find_EP(exx, k, j, ep_r1);
  iep2[5] = find_EP(exx, l, i, ep_r2);
  iRd[5]  = EXX_Index_XYZ2Cell(ep_rd, nshep);

  /* 6: (l,i,-R2-Rd), (k,j,R1-Rd), -R2 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = -r2[m]-rd[m];
    ep_r2[m] = r1[m]-rd[m];
    ep_rd[m] = -r2[m];
  }
  iep1[6] = find_EP(exx, l, i, ep_r1);
  iep2[6] = find_EP(exx, k, j, ep_r2);
  iRd[6]  = EXX_Index_XYZ2Cell(ep_rd, nshep);
  
  /* 7: (l,j,R1-R2-Rd), (k,i,-Rd), -R2 */
  for (m=0; m<3; m++) { 
    ep_r1[m] = r1[m]-r2[m]-rd[m];
    ep_r2[m] = -rd[m];
    ep_rd[m] = -r2[m];
  }
  iep1[7] = find_EP(exx, l, j, ep_r1);
  iep2[7] = find_EP(exx, k, i, ep_r2);
  iRd[7]  = EXX_Index_XYZ2Cell(ep_rd, nshep);

  /* multiplicity */
  Rd_is_0 = (0==rd[0]) && (0==rd[1]) && (0==rd[2]);
  R1_is_0 = (0==r1[0]) && (0==r1[1]) && (0==r1[2]);
  R2_is_0 = (0==r2[0]) && (0==r2[1]) && (0==r2[2]);
  if (Rd_is_0 && iop1==iop2) {
    if (R1_is_0 && i==j) { *mul = 8; } else { *mul = 2; }
  } else {
    if (R1_is_0 && i==j) { 
      if (R2_is_0 && k==l) { *mul = 4; } else { *mul = 2; }
    } else {
      if (R2_is_0 && k==l) { *mul = 2; } else { *mul = 1; }
    }
  }
}


static int find_EP_cluster(
  EXX_t *exx,
  int ia1,
  int ia2
)
{
  int i, nep;
  const int *ep_atom1, *ep_atom2;

  nep      = EXX_Number_of_EP(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);

  for (i=0; i<nep; i++) {
    if (ia1==ep_atom1[i] && ia2==ep_atom2[i]) { break; }
  }

  if (i==nep) { 
    fprintf(stderr, "***Error at %s (%d)\n", __FILE__, __LINE__);
    fprintf(stderr, "   EP not found!\n");
    abort();
  }

  return i;
}



static void EXX_OP2EP_Cluster(
  EXX_t *exx,
  int iop1, 
  int iop2,
  int *iep1, /* [8] */
  int *iep2, /* [8] */
  int *mul
)
{
  int i, j, k, l, nep;
  const int *op_atom1, *op_atom2;

  nep      = EXX_Number_of_OP(exx);
  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);

  i = op_atom1[iop1]; 
  j = op_atom2[iop1]; 
  k = op_atom1[iop2]; 
  l = op_atom2[iop2]; 

  /* 0: (i,k), (j,l) */
  iep1[0] = find_EP_cluster(exx, i, k);
  iep2[0] = find_EP_cluster(exx, j, l);

  /* 1: (j,k), (i,l) */
  iep1[1] = find_EP_cluster(exx, j, k);
  iep2[1] = find_EP_cluster(exx, i, l);

  /* 2: (i,l), (j,k) */
  iep1[2] = find_EP_cluster(exx, i, l);
  iep2[2] = find_EP_cluster(exx, j, k);
  
  /* 3: (j,l), (i,k) */
  iep1[3] = find_EP_cluster(exx, j, l);
  iep2[3] = find_EP_cluster(exx, i, k);

  /* 4: (k,i), (l,j) */
  iep1[4] = find_EP_cluster(exx, k, i);
  iep2[4] = find_EP_cluster(exx, l, j);

  /* 5: (k,j), (l,i) */
  iep1[5] = find_EP_cluster(exx, k, j);
  iep2[5] = find_EP_cluster(exx, l, i);

  /* 6: (l,i), (k,j) */
  iep1[6] = find_EP_cluster(exx, l, i);
  iep2[6] = find_EP_cluster(exx, k, j);
  
  /* 7: (l,j), (k,i) */
  iep1[7] = find_EP_cluster(exx, l, j);
  iep2[7] = find_EP_cluster(exx, k, i);

  /* multiplicity */
  if (iop1==iop2) {
    if (i==j) { *mul = 8; } else { *mul = 2; }
  } else {
    if (i==j) { 
      if (k==l) { *mul = 4;} else { *mul = 2; }
    } else {
      if (k==l) { *mul = 2;} else { *mul = 1; }
    }
  }
}


static void EXX_Basis_Index(
  int ib1,
  int ib2,
  int ib3,
  int ib4,
  int *ib1_op,
  int *ib2_op,
  int *ib3_op,
  int *ib4_op,
  int mul
)
{
  switch (mul) {
  case 0: *ib1_op = ib1; *ib2_op = ib3; *ib3_op = ib2; *ib4_op = ib4; break;
  case 1: *ib1_op = ib3; *ib2_op = ib1; *ib3_op = ib2; *ib4_op = ib4; break;
  case 2: *ib1_op = ib1; *ib2_op = ib3; *ib3_op = ib4; *ib4_op = ib2; break;
  case 3: *ib1_op = ib3; *ib2_op = ib1; *ib3_op = ib4; *ib4_op = ib2; break;
  case 4: *ib1_op = ib2; *ib2_op = ib4; *ib3_op = ib1; *ib4_op = ib3; break;
  case 5: *ib1_op = ib4; *ib2_op = ib2; *ib3_op = ib1; *ib4_op = ib3; break;
  case 6: *ib1_op = ib2; *ib2_op = ib4; *ib3_op = ib3; *ib4_op = ib1; break;
  case 7: *ib1_op = ib4; *ib2_op = ib2; *ib3_op = ib3; *ib4_op = ib1; break;
#if 0
  case 0: *ib1_op = ib1; *ib2_op = ib3; *ib3_op = ib2; *ib4_op = ib4; break;
  case 1: *ib1_op = ib2; *ib2_op = ib3; *ib3_op = ib1; *ib4_op = ib4; break;
  case 2: *ib1_op = ib1; *ib2_op = ib4; *ib3_op = ib2; *ib4_op = ib3; break;
  case 3: *ib1_op = ib2; *ib2_op = ib4; *ib3_op = ib1; *ib4_op = ib3; break;
  case 4: *ib1_op = ib3; *ib2_op = ib1; *ib3_op = ib4; *ib4_op = ib2; break;
  case 5: *ib1_op = ib3; *ib2_op = ib2; *ib3_op = ib4; *ib4_op = ib1; break;
  case 6: *ib1_op = ib4; *ib2_op = ib1; *ib3_op = ib3; *ib4_op = ib2; break;
  case 7: *ib1_op = ib4; *ib2_op = ib2; *ib3_op = ib3; *ib4_op = ib1; break;
#endif
  }
}


void EXX_Fock_Cluster(
  double **H,
  double *Ux,
  EXX_t *exx,
  dcomplex ***exx_CDM, 
  const int *MP
)
{
  double *local_H;
  /* double *buffer_H;*/
  int H_n;
  int i, j, n, ir, nr, irn, nep;

  int iop1, iop2, nb1, nb2, nb3, nb4, nrn;
  int ia1, ia2, ia3, ia4, ib1, ib2, ib3, ib4;
  int ib1_ep, ib2_ep, ib3_ep, ib4_ep;
  int nb1_ep, nb2_ep, nb3_ep, nb4_ep;
  const int *ep_atom1, *ep_atom2, *atom_nb;
  const int *op_atom1, *op_atom2;
 
  int iep1[8], iep2[8], mul;
  double w;

  int GA_AN, wanA, tnoA, Anum, GB_AN, wanB, tnoB, Bnum;
  
  double *eri;
  int     neri;
  double sum, den;
   
  int nproc, myid, iproc;
  MPI_Status stat;

  double local_Ux;
  /*double buffer_Ux;*/

  MPI_Comm_size(g_exx_mpicomm, &nproc);
  MPI_Comm_rank(g_exx_mpicomm, &myid);
 
  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
  atom_nb  = EXX_atom_nb(exx);

  /* matrix size */
  H_n = 0; 
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    H_n += Spe_Total_CNO[wanA];
  }

  /* allocation */
  local_H = (double*)malloc(sizeof(double)*H_n*H_n);
  /*buffer_H = (double*)malloc(sizeof(double)*H_n*H_n);*/
  for (i=0; i<H_n*H_n; i++) { local_H[i] = 0.0; } 

  nr = EXX_File_ERI_Read_NRecord(exx);
  local_Ux = 0.0;

  for (ir=0; ir<nr; ir++) {
    EXX_File_ERI_Read_Data_Head(exx, ir, &iop1, &iop2,
      &nb1, &nb2, &nb3, &nb4, &nrn);

    /*EXX_Log_Print("ir= %2d  iop1= %2d  iop2= %2d  ", ir, iop1, iop2);*/
    /*EXX_Log_Print("atom= %1d %1d %1d %1d\n", */
    /*  op_atom1[iop1], op_atom2[iop1], op_atom1[iop2], op_atom2[iop2]);*/

    if (nrn != 1) { 
      fprintf(stderr, "***Error at %s (%d)\n", __FILE__, __LINE__);
      fprintf(stderr, "   file is broken.\n");
      abort();
    }

    neri = nb1*nb2*nb3*nb4*nrn; 
    eri = (double*)malloc(sizeof(double)*neri);
    EXX_File_ERI_Read(exx, ir, eri, NULL, iop1, iop2, nrn, neri);

    
#if 0
    if (0==g_step) {
      EXX_Log_Print("step3: ir= %d\n", ir);
  
      for (ib1=0; ib1<nb1; ib1++) {
        for (ib2=0; ib2<nb2; ib2++) {
          for (ib3=0; ib3<nb3; ib3++) {
            for (ib4=0; ib4<nb4; ib4++) {
              for (irn=0; irn<nrn; irn++) {
                EXX_Log_Print("IB1= %2d  IB2= %2d  ", ib1, ib2);
                EXX_Log_Print("IB3= %2d  IB4= %2d  ", ib3, ib4);
                EXX_Log_Print("IRN= %3d  ", irn);
                i = (((ib1*nb2+ib2)*nb3+ib3)*nb4+ib4)*nrn+irn;
                EXX_Log_Print("  ERI= %10.6f\n", eri[i]);
              }
            }
          }
        }
      }
    }
#endif
 
    EXX_OP2EP_Cluster(exx, iop1, iop2, iep1, iep2, &mul);
    /*if (0==SpinP_switch) { mul *= 2; }*/
    w = 1.0/(double)mul; 

    for (j=0; j<8; j++) {
      ia1 = ep_atom1[iep1[j]]; /* i */
      ia2 = ep_atom2[iep1[j]]; /* j */
      ia3 = ep_atom1[iep2[j]]; /* k */
      ia4 = ep_atom2[iep2[j]]; /* l */
      /*EXX_Log_Print("  j=%1d  iep1= %2d  iep2= %2d  ", j, iep1[j], iep2[j]);*/
      /*EXX_Log_Print("atom= %1d %1d %1d %1d\n", ia1, ia2, ia3, ia4);*/
   
      nb1_ep = atom_nb[ia1];
      nb2_ep = atom_nb[ia2];
      nb3_ep = atom_nb[ia3];
      nb4_ep = atom_nb[ia4];
      
      /* EXX_Log_Print("  nb= %2d %2d %2d %2d\n", nb1_ep, nb2_ep, nb3_ep, nb4_ep);*/

      GA_AN = ia1+1;
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
        
      GB_AN = ia2+1;
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      for (ib1_ep=0; ib1_ep<nb1_ep; ib1_ep++) {
        for (ib2_ep=0; ib2_ep<nb2_ep; ib2_ep++) {
          for (ib3_ep=0; ib3_ep<nb3_ep; ib3_ep++) {
            for (ib4_ep=0; ib4_ep<nb4_ep; ib4_ep++) {
              EXX_Basis_Index(ib1_ep, ib2_ep, ib3_ep, ib4_ep,
                &ib1, &ib2, &ib3, &ib4, j);
              /*EXX_Log_Print("  basis(OP)= %2d %2d %2d %2d",*/ 
              /*  ib1, ib2, ib3, ib4);*/
              /*EXX_Log_Print("  basis(EP)= %2d %2d %2d %2d\n",*/ 
              /*  ib1_ep, ib2_ep, ib3_ep, ib4_ep);*/
              i = (((ib1*nb2+ib2)*nb3+ib3)*nb4+ib4)*nrn;
              den = exx_CDM[iep2[j]][ib3_ep][ib4_ep].r;
              /*if (den<0.0) { den = 0.0; }*/
              /*if (den>1.0) { den = 1.0; }*/
              sum = eri[i] * den;
              /*i = (Anum+ib1-1)*H_n + (Bnum+ib2-1);*/
              i = (Anum+ib1_ep-1)*H_n + (Bnum+ib2_ep-1);
              local_H[i] += -w*sum;
              den = exx_CDM[iep1[j]][ib1_ep][ib2_ep].r;
              /*if (den<0.0) { den = 0.0; }*/
              /*if (den>1.0) { den = 1.0; }*/
              local_Ux += -0.5*w*sum*den;
            }
          }
          /*local_H[i] += -0.5*w*sum;*/
          /*local_Ux += -0.5*w*sum*exx_CDM[iep1[j]][ib1_ep][ib2_ep];*/
	}
      }
    } /* loop of j */
 
    free(eri);
  }


#if 0
  /* gather data */
  if (EXX_ROOT_RANK==myid) {
    for (iproc=0; iproc<nproc; iproc++) {
      if (EXX_ROOT_RANK==iproc) { continue; }
      /* Fock matrix */
      MPI_Recv(buffer_H, H_n*H_n, MPI_DOUBLE, iproc, 0, g_exx_mpicomm, &stat);
      for (i=0; i<H_n*H_n; i++) { local_H[i] += buffer_H[i]; }
      /* energy */
      MPI_Recv(&buffer_Ux, 1, MPI_DOUBLE, iproc, 1, g_exx_mpicomm, &stat);
      local_Ux += buffer_Ux;
    } /* iproc */
  } else {
    MPI_Send(local_H, H_n*H_n, MPI_DOUBLE, EXX_ROOT_RANK, 0, g_exx_mpicomm);
    MPI_Send(&local_Ux, 1, MPI_DOUBLE, EXX_ROOT_RANK, 1, g_exx_mpicomm);
  }
#endif
  /* all-to-all communication */
  MPI_Allreduce(local_H, local_H, H_n*H_n, MPI_DOUBLE, 
    MPI_SUM, g_exx_mpicomm);
  MPI_Allreduce(&local_Ux, &local_Ux, 1, MPI_DOUBLE,
    MPI_SUM, g_exx_mpicomm);


#if 0
#if 0
  if (2==g_step || 3==g_step) { EXX_Log_Print("Local_H:\n"); }
#endif

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    for (GB_AN=1; GB_AN<=atomnum; GB_AN++) {
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];

      for (ib1=0; ib1<tnoA; ib1++) {
        for (ib2=0; ib2<tnoB; ib2++) {
          i = (Anum+ib1-1)*H_n + (Bnum+ib2-1);
          H[Anum+ib1][Bnum+ib2] += local_H[i];
#if 0
          if (2==g_step || 3==g_step) { 
            EXX_Log_Print("  %2d-%2d  %2d-%2d  %15.8f\n", 
              GA_AN, ib1, GB_AN, ib2, local_H[i]);
          }
#endif
        }
      }
    }
  }
#if 0
  EXX_Log_Print("\n");
#endif
#endif
  *Ux = local_Ux;

  g_step++;

  /* free*/
  free(local_H); 
}


void EXX_Simple_Mixing_DM(
  EXX_t *exx,
  double mix_wgt,
  dcomplex ***exx_CDM,
  dcomplex ***exx_PDM,
  dcomplex ***exx_P2DM
)
{
  int iep, nep, Gc_AN, Gh_AN, ian, jan, m, n;
  const int *ep_atom1, *ep_atom2;
  double w1, w2;

  nep = EXX_Number_of_EP(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
    
  w1 = mix_wgt;
  w2 = 1.0 - mix_wgt;

#if EXX_MIX_DM
  for (iep=0; iep<nep; iep++) {
    Gc_AN = ep_atom1[iep]+1;
    ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    Gh_AN = ep_atom2[iep]+1;
    jan = Spe_Total_CNO[WhatSpecies[Gh_AN]];
    for (m=0; m<ian; m++){
      for (n=0; n<jan; n++){
        exx_CDM[iep][m][n].r = w1 * exx_CDM[iep][m][n].r
 	                      + w2 * exx_PDM[iep][m][n].r;
        exx_CDM[iep][m][n].i = w1 * exx_CDM[iep][m][n].i
 	                      + w2 * exx_PDM[iep][m][n].i;
        exx_P2DM[iep][m][n].r = exx_PDM[iep][m][n].r;
        exx_P2DM[iep][m][n].i = exx_PDM[iep][m][n].i;
        exx_PDM[iep][m][n].r  = exx_CDM[iep][m][n].r;
        exx_PDM[iep][m][n].i  = exx_CDM[iep][m][n].i;
      }
    }
  } /* loop of iep */
#endif
}





void EXX_Fock_Band(
  dcomplex **H,
  EXX_t *exx,
  dcomplex ****exx_CDM, 
  int    *MP,
  double k1,  
  double k2,  
  double k3,
  int    spin
)
{
  double *local_H, *mpi_H;
  int H_n, nbuf;
  int i, j, n, ir, nr, irn, nep;

  int iop1, iop2, nb1, nb2, nb3, nb4, nrn;
  int ia1, ia2, ia3, ia4, ib1, ib2, ib3, ib4, icell1, icell2;
  int ib1_ep, ib2_ep, ib3_ep, ib4_ep;
  int nb1_ep, nb2_ep, nb3_ep, nb4_ep;
  const int *ep_atom1, *ep_atom2, *ep_cell, *atom_nb;
  const int *op_atom1, *op_atom2;
 
  int iep1[8], iep2[8], icell3[8], mul;
  double w;

  int GA_AN, Anum, GB_AN, Bnum, tnoA, tnoB;
  
  int     neri;
  double *eri_list;
  int *iRm;

  double eri;
  dcomplex den;
   
  int nproc, myid, iproc;
  MPI_Status stat;

  int ncd, nshell_ep;
  int iRn_x, iRn_y, iRn_z, iRmp_x, iRmp_y, iRmp_z;
  double kRn, kRmp, co1, si1, co2, si2;

  MPI_Comm comm;
  double *k1_list, *k2_list, *k3_list;
  int *spin_list;

  comm = g_exx_mpicomm;
  
  MPI_Comm_size(comm, &nproc);
  MPI_Comm_rank(comm, &myid);

  k1_list = (double*)malloc(sizeof(double)*nproc);
  k2_list = (double*)malloc(sizeof(double)*nproc);
  k3_list = (double*)malloc(sizeof(double)*nproc);
  spin_list = (int*)malloc(sizeof(int)*nproc);

  /* all-to-all */
  MPI_Allgather(&k1,   1, MPI_DOUBLE, k1_list,   1, MPI_DOUBLE, comm);
  MPI_Allgather(&k2,   1, MPI_DOUBLE, k2_list,   1, MPI_DOUBLE, comm);
  MPI_Allgather(&k3,   1, MPI_DOUBLE, k3_list,   1, MPI_DOUBLE, comm);
  MPI_Allgather(&spin, 1, MPI_INT,    spin_list, 1, MPI_INT,    comm);

  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
  ep_cell  = EXX_Array_EP_Cell(exx);
  atom_nb  = EXX_atom_nb(exx);
  nshell_ep = EXX_Number_of_EP_Shells(exx);

  ncd = 2*nshell_ep+1;

  /* matrix size */
  H_n = 0; 
  for (i=1; i<=atomnum; i++){ H_n += Spe_Total_CNO[WhatSpecies[i]]; }
 
  /* allocation */
  nbuf = H_n*H_n*2;
  local_H = (double*)malloc(sizeof(double)*nbuf);
  mpi_H   = (double*)malloc(sizeof(double)*nbuf);

  nr = EXX_File_ERI_Read_NRecord(exx);

  /* clear buffer */
  for (i=0; i<nbuf; i++) { local_H[i] = 0.0; }

  for (ir=0; ir<nr; ir++) {
    EXX_File_ERI_Read_Data_Head(exx, ir, &iop1, &iop2,
      &nb1, &nb2, &nb3, &nb4, &nrn);

    neri = nb1*nb2*nb3*nb4*nrn; 
    eri_list = (double*)malloc(sizeof(double)*neri);
    iRm = (int*)malloc(sizeof(int)*nrn); /* Rm */
    EXX_File_ERI_Read(exx, ir, eri_list, iRm, iop1, iop2, nrn, neri);
 
    for (iproc=0; iproc<nproc; iproc++) {
      /* clear buffer */
      for (i=0; i<nbuf; i++) { mpi_H[i] = 0.0; } 

      k1   = k1_list[iproc];
      k2   = k2_list[iproc];
      k3   = k3_list[iproc];
      spin = spin_list[iproc];
      
      for (irn=0; irn<nrn; irn++) {
        EXX_OP2EP_Band(exx, iop1, iop2, iRm[irn], 
          &iep1[0], &iep2[0], &icell3[0], &mul);
        w = 1.0/(double)mul; 

        for (j=0; j<8; j++) {
          ia1 = ep_atom1[iep1[j]]; /* i */
          ia2 = ep_atom2[iep1[j]]; /* j */
          ia3 = ep_atom1[iep2[j]]; /* k */
          ia4 = ep_atom2[iep2[j]]; /* l */
          icell1 = ep_cell[iep1[j]]; /* Rn */
          icell2 = ep_cell[iep2[j]]; /* Rm' */
 
          nb1_ep = atom_nb[ia1];
          nb2_ep = atom_nb[ia2];
          nb3_ep = atom_nb[ia3];
          nb4_ep = atom_nb[ia4];
      
          GA_AN = ia1+1;
          Anum = MP[GA_AN];
        
          GB_AN = ia2+1;
          Bnum = MP[GB_AN];

          /* phase */
          iRn_x = icell1%ncd - nshell_ep;
          iRn_y = (icell1/ncd)%ncd - nshell_ep;
          iRn_z = (icell1/ncd/ncd)%ncd - nshell_ep;
          kRn = k1*(double)iRn_x + k2*(double)iRn_y + k3*(double)iRn_z;
          si1 = sin(2.0*PI*kRn);
          co1 = cos(2.0*PI*kRn);

          iRmp_x = icell2%ncd - nshell_ep;
          iRmp_y = (icell2/ncd)%ncd - nshell_ep;
          iRmp_z = (icell2/ncd/ncd)%ncd - nshell_ep;

          kRmp = k1*(double)iRmp_x + k2*(double)iRmp_y + k3*(double)iRmp_z;
          si2 = sin(2.0*PI*kRmp);
          co2 = cos(2.0*PI*kRmp);
 
          for (ib1_ep=0; ib1_ep<nb1_ep; ib1_ep++) {
            for (ib2_ep=0; ib2_ep<nb2_ep; ib2_ep++) {
              for (ib3_ep=0; ib3_ep<nb3_ep; ib3_ep++) {
                for (ib4_ep=0; ib4_ep<nb4_ep; ib4_ep++) {
                  EXX_Basis_Index(ib1_ep, ib2_ep, ib3_ep, ib4_ep,
                    &ib1, &ib2, &ib3, &ib4, j);
                  i = (((ib1*nb2+ib2)*nb3+ib3)*nb4+ib4)*nrn+irn;
                  eri = eri_list[i];
                  den.r = exx_CDM[spin][iep2[j]][ib3_ep][ib4_ep].r;
                  den.i = exx_CDM[spin][iep2[j]][ib3_ep][ib4_ep].i;
                  i = (Anum+ib1_ep-1)*H_n + (Bnum+ib2_ep-1);
                  mpi_H[2*i+0] += -w*eri*(co1*den.r - si1*den.i); /* real */
                  mpi_H[2*i+1] += -w*eri*(co1*den.i + si1*den.r); /* imag */
                }
              }
  	    }
          }
        } /* loop of j */
      } /* loop of irn */ 

      /* reduce to the iproc-th proc */
      MPI_Reduce(mpi_H, mpi_H, nbuf, MPI_DOUBLE, MPI_SUM, iproc, comm);

      if (myid==iproc) {
        for (i=0; i<nbuf; i++) { local_H[i] = mpi_H[i]; }
      }
    } /* loop of iproc */

    free(eri_list);
    free(iRm);
  } /* loop of irn */


#if 1
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
    tnoA = Spe_Total_CNO[WhatSpecies[GA_AN]];
    Anum = MP[GA_AN];
    for (GB_AN=1; GB_AN<=atomnum; GB_AN++) {
      tnoB = Spe_Total_CNO[WhatSpecies[GB_AN]];
      Bnum = MP[GB_AN];
      for (ib1=0; ib1<tnoA; ib1++) {
        for (ib2=0; ib2<tnoB; ib2++) {
          i = (Anum+ib1-1)*H_n + (Bnum+ib2-1);
          H[Anum+ib1][Bnum+ib2].r += local_H[2*i+0];
          H[Anum+ib1][Bnum+ib2].i += local_H[2*i+1];
        }
      }
    }
  }
#endif

  /* free*/
  free(local_H); 
  free(mpi_H); 
}


/*----------------------------------------------------------------------
  EXX_Energy_Band
----------------------------------------------------------------------*/
void EXX_Energy_Band(
  double *Ux,
  EXX_t *exx,
  dcomplex ****exx_CDM, 
  int    *MP
)
{
  int myrank, nproc, iproc;
  int i, j, n, ir, nr, irn, nep, spin;

  int GA_AN,Anum, GB_AN, Bnum;
  int iop1, iop2, nb1, nb2, nb3, nb4, nrn;
  int iatm1, iatm2, iatm3, iatm4, ib1, ib2, ib3, ib4, icell1, icell2;
  int ib1_ep, ib2_ep, ib3_ep, ib4_ep;
  int nb1_ep, nb2_ep, nb3_ep, nb4_ep;
  const int *ep_atom1, *ep_atom2, *ep_cell, *atom_nb;
  const int *op_atom1, *op_atom2;
  int iep1[8], iep2[8], icell3[8], mul;
  double w;
  
  int neri;
  double *eri_list;
  double eri;
  int *iRm;
  dcomplex den1, den2;
   
  MPI_Status stat;

  double local_Ux[2];

  MPI_Comm comm;

  /* MPI information */
  comm = g_exx_mpicomm;
  MPI_Comm_size(comm, &nproc);
  MPI_Comm_rank(comm, &myrank);

  /* Index information */
  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
  ep_cell  = EXX_Array_EP_Cell(exx);
  atom_nb  = EXX_atom_nb(exx);

  /* saved datapath */
  nr = EXX_File_ERI_Read_NRecord(exx);

  local_Ux[0] = 0.0;
  local_Ux[1] = 0.0;

  for (ir=0; ir<nr; ir++) {
    /* read data */
    EXX_File_ERI_Read_Data_Head(exx, ir, &iop1, &iop2,
      &nb1, &nb2, &nb3, &nb4, &nrn);
    neri = nb1*nb2*nb3*nb4*nrn; 
    eri_list = (double*)malloc(sizeof(double)*neri);
    iRm = (int*)malloc(sizeof(int)*nrn); /* Rm */
    EXX_File_ERI_Read(exx, ir, eri_list, iRm, iop1, iop2, nrn, neri);
 
    for (irn=0; irn<nrn; irn++) {
      EXX_OP2EP_Band(exx, iop1, iop2, iRm[irn], 
        &iep1[0], &iep2[0], &icell3[0], &mul);
      w = 1.0/(double)mul; 

      for (j=0; j<8; j++) {
        iatm1  = ep_atom1[iep1[j]]; /* i */
        iatm2  = ep_atom2[iep1[j]]; /* j */
        iatm3  = ep_atom1[iep2[j]]; /* k */
        iatm4  = ep_atom2[iep2[j]]; /* l */
        icell1 = ep_cell[iep1[j]];  /* Rn */
        icell2 = ep_cell[iep2[j]];  /* Rm' */
 
        nb1_ep = atom_nb[iatm1];
        nb2_ep = atom_nb[iatm2];
        nb3_ep = atom_nb[iatm3];
        nb4_ep = atom_nb[iatm4];
      
        Anum = MP[iatm1+1];
        Bnum = MP[iatm2+1];

        for (ib1_ep=0; ib1_ep<nb1_ep; ib1_ep++) {
          for (ib2_ep=0; ib2_ep<nb2_ep; ib2_ep++) {
            for (ib3_ep=0; ib3_ep<nb3_ep; ib3_ep++) {
              for (ib4_ep=0; ib4_ep<nb4_ep; ib4_ep++) {
                /* rotate basis index */
                EXX_Basis_Index(ib1_ep, ib2_ep, ib3_ep, ib4_ep,
                  &ib1, &ib2, &ib3, &ib4, j);

                i = (((ib1*nb2+ib2)*nb3+ib3)*nb4+ib4)*nrn+irn;
                eri = eri_list[i];

                for (spin=0; spin<=SpinP_switch; spin++) {
                  den1.r = exx_CDM[spin][iep2[j]][ib3_ep][ib4_ep].r;
                  den1.i = exx_CDM[spin][iep2[j]][ib3_ep][ib4_ep].i;
                  den2.r = exx_CDM[spin][iep1[j]][ib1_ep][ib2_ep].r;
                  den2.i = exx_CDM[spin][iep1[j]][ib1_ep][ib2_ep].i;
                  local_Ux[spin] += 
                    -0.5*w*eri*(den1.r*den1.r - den2.i*den2.i);
                }
              }
            }
          }
        }

      } /* loop of j */
    } /* loop of irn */ 

    MPI_Allreduce(local_Ux, local_Ux, 2, MPI_DOUBLE, MPI_SUM, comm);

    free(eri_list);
    free(iRm);
  } /* loop of irn */

  Ux[0] = local_Ux[0];
  Ux[1] = local_Ux[1];
}




/*----------------------------------------------------------------------
  EXX_Reduce_DM
----------------------------------------------------------------------*/
void EXX_Reduce_DM(EXX_t *exx, dcomplex ***exx_DM)
{
  int i, j, iatm1, iatm2, iep, ib,  myrank, nproc;
  int nep, nb, nbmax, nbuf, nb1, nb2;
  const int *ep_atom1, *ep_atom2;
  double *buffer;
  MPI_Comm comm;

  comm = g_exx_mpicomm;

  nep = EXX_Number_of_EP(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);

  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nproc);

  /* max number of basis */
  nbmax = 0;
  for (i=1; i<=atomnum; i++) {
    nb = Spe_Total_CNO[WhatSpecies[i]];
    if (nb>nbmax) { nbmax=nb; }
  }

  /* 1d array */
  nbuf = nbmax*nbmax*2;
  buffer = (double*)malloc(sizeof(double)*nbuf);

  for (iep=0; iep<nep; iep++) {
    iatm1 = ep_atom1[iep]+1;
    iatm2 = ep_atom2[iep]+1;
    nb1 = Spe_Total_CNO[WhatSpecies[iatm1]];
    nb2 = Spe_Total_CNO[WhatSpecies[iatm2]];

    /* clear array */ 
    for (i=0; i<nbuf; i++) { buffer[i] = 0.0; }

    /* put DM on array */
    for (i=0; i<nb1; i++){
      for (j=0; j<nb2; j++){
        ib = i*nbmax + j;
        buffer[2*ib+0] = exx_DM[iep][i][j].r;
        buffer[2*ib+1] = exx_DM[iep][i][j].i;
      }
    }

    /* all-to-all communication */
    MPI_Allreduce(buffer, buffer, nbuf, MPI_DOUBLE, MPI_SUM, comm);
   
    /* get back from array */ 
    for (i=0; i<nb1; i++){
      for (j=0; j<nb2; j++){
        ib = i*nbmax + j;
        exx_DM[iep][i][j].r = buffer[2*ib+0];
        exx_DM[iep][i][j].i = buffer[2*ib+1];
      }
    }
  } /* loop of iep */

  free(buffer);
}



