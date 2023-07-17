/*----------------------------------------------------------------------
 exx_step1.c
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "exx.h"
#include "exx_vector.h"
#include "exx_log.h"
#include "exx_step1.h"
#include "exx_file_overlap.h"
#include "eri.h"
#include "eri_sf.h"
#include "eri_def.h"


int EXX_Step1(
  EXX_t  *exx,
  int     natom,
  int    *atom_sp,    /* [natom] */
  int    nspec, 
  int    *spec_nb,    /* [nspec] */
  double *spec_rc,    /* [nspec] */ 
  int    **spec_l,    /* [nspec][nbmax] */
  int    **spec_m,    /* [nspec][nbmax] */
  int    **spec_mul,  /* [nspec][nbmax] */
  double ****spec_fr, /* [nspec][lmax][mulmax][nmeshmax] */
  double **spec_xr,   /* [nspec][nmeshmax] */  
  int    *spec_nmesh   /* [nspec] */
)
{
  int i, j;
  int iatom1, iatom2, icell, ispec1, ispec2, ib1, ib2, iop;
  int nb1, nb2, nbmax, nshell;
  int nop, nop_n, nop_m, nop_local, iop0, npp_local;
  int lmax, lmax_gl, ngrid, ngl, ndglf, ndalp, jmax_gl;
  int l, m, mul;
  double x, y, z, c1[3], c2[3], rc1, rc2, d, rc12, cx12, c12[3];
  double ar1, at1, ap1, ar2, at2, ap2;

  const int *op_atom1, *op_atom2, *op_cell, *atom_nb;
  double *fk, *gam, *alp1, *alp2, *ovlp, *ovlF;
  double **glf; /* [nop_local][ndglf*nbmax*nbmax] */
  const double *fr, *xr, *pvec, *atom_v;

  ERI_Init_Misc info;
  ERI_t *solver;

  clock_t clk1, clk2;
  int nstep;
  double et;


  int nq_full, nq_sig;
 
  char path_cache[EXX_PATHLEN];

  int myrank, nproc;
 
#ifdef EXX_USE_MPI
  MPI_Comm comm;
  MPI_Status mpistat;
#endif

#ifdef EXX_USE_MPI
  comm = g_exx_mpicomm;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nproc);
#else
  myrank  = EXX_ROOT_RANK;
  nproc = 1;
#endif

  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);
  op_cell  = EXX_Array_OP_Cell(exx);
  atom_nb  = EXX_atom_nb(exx);
  pvec     = EXX_pvec(exx);
  atom_v   = EXX_atom_v(exx);
  nbmax    = EXX_nbmax(exx);

  nshell = EXX_Number_of_OP_Shells(exx);
  nop   = EXX_Number_of_OP(exx);
  nop_n = nop / nproc;
  nop_m = nop - nop_n*nproc; 

  EXX_LOG_TRACE_INTEGER("NOP", nop);
  EXX_LOG_TRACE_INTEGER("NOP_N", nop_n);
  EXX_LOG_TRACE_INTEGER("NOP_M", nop_m);

  sprintf(path_cache, "%s/exxovlp.dat", EXX_CacheDir(exx));

#ifdef EXX_USE_MPI
  MPI_Barrier(comm);
#endif

  nop_local = (myrank<nop_m) ? (nop_n+1) : (nop_n);
  iop0 = 0;
  for (i=0; i<myrank; i++) { 
    iop0 += (i<nop_m) ? (nop_n+1) : (nop_n); 
  }

  EXX_LOG_TRACE_INTEGER("IOP0"     , iop0);
  EXX_LOG_TRACE_INTEGER("NOP_N"    , nop_n);
  EXX_LOG_TRACE_INTEGER("NOP_M"    , nop_m);
  EXX_LOG_TRACE_INTEGER("NOP_LOCAL", nop_local);
  
  /*----- initialize LIBERI -----*/
  lmax    = g_exx_liberi_lmax;
  lmax_gl = (g_exx_liberi_lmax+1)/2;
  ngrid   = g_exx_liberi_ngrid;
  ngl     = g_exx_liberi_ngl;;

  info.sbttype = ERI_SBT_LINEAR;
  info.rmax = 80.0; 

  solver = ERI_Init(lmax, lmax_gl, ngrid, ngl, ERI_SH_REAL, &info);
  if (NULL==solver) { EXX_ERROR( "failed to initialize LIBERI" ); }

#if 1
  for (i=0; i<natom; i++) {
    EXX_Vector_F2C(c1, &atom_v[3*i], pvec);
    EXX_Log_Print("  atom %2d : ( %10.4f, %10.4f, %10.4f)\n",  
      i, c1[0], c1[1], c1[2]);
  }
  EXX_Log_Print("\n");
#endif

  jmax_gl = lmax_gl*lmax_gl;

  npp_local = 0;
  for (i=0; i<nop_local; i++) {
    iop = iop0 + i;
    nb1 = atom_nb[op_atom1[iop]];
    nb2 = atom_nb[op_atom2[iop]];
    npp_local += nb1*nb2;
  }

  ndglf = ERI_Size_of_GLF(solver)/sizeof(double);
  ndalp = ERI_Size_of_Alpha(solver)/sizeof(double);
  fk    = (double*)malloc( ERI_Size_of_Orbital(solver) );
  gam   = (double*)malloc( ERI_Size_of_Gamma(solver)   );
  alp1  = (double*)malloc( sizeof(double)*ndalp*nbmax );
  alp2  = (double*)malloc( sizeof(double)*ndalp*nbmax );
  ovlp  = (double*)malloc( ERI_Size_of_Overlap(solver) );
  ovlF  = (double*)malloc( ERI_Size_of_Overlap(solver) );
  glf  = (double**)malloc( sizeof(double*)*nop_local );
  for (i=0; i<nop_local; i++) {
    glf[i] = (double*)malloc( sizeof(double)*ndglf*nbmax*nbmax );
  }
 
  clk1 = clock();

  nstep = 0;

  for (i=0; i<nop_local; i++) {
    iop = iop0 + i;
    iatom1 = op_atom1[iop];
    iatom2 = op_atom2[iop];
    icell  = op_cell[iop];

    /* atom 1 */
    EXX_Vector_F2C(c1, &atom_v[3*iatom1], pvec);
    ispec1 = atom_sp[iatom1];
    rc1 = spec_rc[ispec1];
    nb1 = spec_nb[ispec1];

    /* atom 2 */ 
    EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatom2], pvec, icell, nshell);
    ispec2 = atom_sp[iatom2];
    rc2 = spec_rc[ispec2];
    nb2 = spec_nb[ispec2];

    d = EXX_Vector_Distance(c1, c2);
    EXX_Vector_PAO_Overlap(rc1, rc2, d, &rc12, &cx12);
    c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
    c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
    c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];
 
    /* translation */
    c1[0] -= c12[0]; c1[1] -= c12[1]; c1[2] -= c12[2]; 
    c2[0] -= c12[0]; c2[1] -= c12[1]; c2[2] -= c12[2]; 

    /* cartesian -> spherical */    
    EXX_Vector_C2S(c1, &ar1, &at1, &ap1);
    EXX_Vector_C2S(c2, &ar2, &at2, &ap2);

    /* pre-calculate alpha terms */
    for (ib1=0; ib1<nb1; ib1++) { 
      l     = spec_l[ispec1][ib1];
      mul   = spec_mul[ispec1][ib1];
      m     = spec_m[ispec1][ib1];
      fr    = spec_fr[ispec1][l][mul];
      xr    = spec_xr[ispec1];
      ngrid = spec_nmesh[ispec1];
   
      ERI_Transform_Orbital(solver, fk, fr, xr, ngrid, l);
      ERI_LL_Gamma(solver, gam, NULL, fk, ar1); 
      ERI_LL_Alpha(solver, &alp1[ndalp*ib1], gam, at1, ap1, l, m);
    }

    for (ib2=0; ib2<nb2; ib2++) { 
      l     = spec_l[ispec2][ib2];
      mul   = spec_mul[ispec2][ib2];
      m     = spec_m[ispec2][ib2];
      fr    = spec_fr[ispec2][l][mul];
      xr    = spec_xr[ispec2];
      ngrid = spec_nmesh[ispec2];

      ERI_Transform_Orbital(solver, fk, fr, xr, ngrid, l);
      ERI_LL_Gamma(solver, gam, NULL, fk, ar2); 
      ERI_LL_Alpha(solver, &alp2[ndalp*ib2], gam, at2, ap2, l, m);
    }
    
    for (ib1=0; ib1<nb1; ib1++) {
      for (ib2=0; ib2<nb2; ib2++) {
        ERI_LL_Overlap(solver, ovlp, &alp1[ndalp*ib1], &alp2[ndalp*ib2]);
        ERI_Transform_Overlap(solver, ovlF, ovlp);
        ERI_GL_Interpolate(solver, &glf[i][(ib1*nbmax+ib2)*ndglf], ovlF);
        nstep++;
      }
    }
  } /* loop of iop */

  clk2 = clock();

#ifdef EXX_USE_MPI
  MPI_Barrier(comm);
#endif

  /* save data */
  EXX_File_Overlap_Write(ndglf, nop_local, nbmax, jmax_gl, glf, path_cache);

#ifdef EXX_USE_MPI
  MPI_Barrier(comm);
#endif

  free(fk);
  free(gam);
  free(alp1);
  free(alp2);
  free(ovlp);
  free(ovlF);
  for (i=0; i<nop_local; i++) { free( glf[i]); }
  free(glf);

  ERI_Free(solver);

  et = (double)(clk2-clk1)/(double)CLOCKS_PER_SEC;

  EXX_LOG_MESSAGE("REPORT OF STEP1:");
  EXX_LOG_TRACE_INTEGER("NPROC", nproc);
  EXX_LOG_TRACE_DOUBLE("TIME ", et);
  EXX_LOG_TRACE_DOUBLE("AVG_TIME", et/(double)nstep);
  
  return 0;
}


