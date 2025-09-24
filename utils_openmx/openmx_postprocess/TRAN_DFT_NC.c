/**********************************************************************
  TRAN_DFT_NC.c:

  TRAN_DFT_NC.c is a subroutine to perform self-consistent calculations
  of a central region with left and right infinite leads based on
  a non-equilibrium Green's function method coupled with NC-DFT.

  Log of TRAN_DFT_NC.c:

     11/Dec/2005   Released by H.Kino
     24/July/2008  Modified by T.Ozaki
     27/June/2016  Modified by T.Ozaki

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#define MEASURE_TIME  0

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "tran_prototypes.h"
#include "tran_variables.h"
#include "openmx_common.h"

static void TRAN_Add_MAT(
    int mode,
    int NUM_c,
    dcomplex w_weight,
    dcomplex *v,  
    dcomplex *out
    );

static double TRAN_DFT_Original(
		/* input */
                MPI_Comm comm1,
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,    /* H */
		double *****ImNL,  /* SO coupling */
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int *G2ID,
                int *F_G2M, 
		int **atv_ijk,
		int *List_YOUSO,
		/* output */
		double *****CDM,  /* output, density matrix */
                double *****iCDM,
		double *****EDM,  /* not used */
                double ***TRAN_DecMulP, /* output, partial DecMulP */
		double Eele0[2], double Eele1[2], 
                double ChemP_e0[2]); 


static void TRAN_DFT_Kdependent(
			  /* input */
			  MPI_Comm comm1,
                          int parallel_mode,
                          int numprocs,
                          int myid,
			  int level_stdout,
			  int iter,
			  int SpinP_switch,
                          double k2,
                          double k3,
                          int k_op,
                          int *order_GA,
                          double **DM1,
                          double **DM2,
                          double **H1,
                          double **H2,
                          double *S1,
			  double *****nh,   /* H */
			  double *****ImNL, /* SO coupling */
			  double ****CntOLP, 
			  int atomnum,
			  int Matomnum,
			  int *WhatSpecies,
			  int *Spe_Total_CNO,
			  int *FNAN,
			  int **natn, 
			  int **ncn,
			  int *M2G, 
			  int *G2ID, 
			  int **atv_ijk,
			  int *List_YOUSO,
			  /* output */
			  double *****CDM,  /* output, charge density */
			  double *****EDM,  /* not used */
			  double Eele0[2], double Eele1[2]); /* not used */



double TRAN_DFT_NC(
		/* input */
                MPI_Comm comm1,
                int SucceedReadingDMfile, 
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,   /* H */
		double *****ImNL, /* SO coupling */
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int *G2ID,
                int *F_G2M, 
		int **atv_ijk,
		int *List_YOUSO,
                double *koS,
                dcomplex **S,
		/* output */
		double *****CDM,  /* output, density matrix */
                double *****iCDM,
		double *****EDM,  /* not used */
                double ***TRAN_DecMulP, /* output, partial DecMulP */
		double Eele0[2], double Eele1[2], 
                double ChemP_e0[2]) 
{
  double TStime,TEtime,time5;

  dtime(&TStime);

  /******************************************************
    if TRAN_Iter_Initial_Band<SCF_iter, employ TRAN_DFT
  ******************************************************/

  if (TRAN_SCF_Iter_Band<iter || SucceedReadingDMfile==1){ 

    TRAN_DFT_Original( comm1,
		       level_stdout,
		       iter, 
		       SpinP_switch,
		       nh,   /* H */
		       ImNL, /* SO coupling */
		       CntOLP, 
		       atomnum,
		       Matomnum,
		       WhatSpecies,
		       Spe_Total_CNO,
		       FNAN,
		       natn, 
		       ncn,
		       M2G, 
		       G2ID,
		       F_G2M, 
		       atv_ijk,
		       List_YOUSO,
		       /* output */
		       CDM,  /* output, density matrix */
                       iCDM,
		       EDM,  /* not used */
		       TRAN_DecMulP, /* output, partial DecMulP */
		       Eele0, Eele1, 
		       ChemP_e0); 
  }

  /*************************************************************
   if TRAN_SCF_Iter_Band<iter, employ the band diagonalization
  *************************************************************/

  else {

    double *Ss_Re,*Cs_Re,*Hs_Re;
    double *rHs11_Re,*rHs12_Re,*rHs22_Re,*iHs11_Re,*iHs12_Re,*iHs22_Re;
    double *CDM1,*EDM1,*PDM1,*Work1;
    dcomplex *EVec1_NonCol;
    double **EVec1_Re;
    double *H1,*S1;
    double ***EIGEN_Band;
    int *My_NZeros,*SP_NZeros,*SP_Atoms,*is2,*ie2,*MP,*order_GA;
    int n,n2,size_H1,myworld1,myworld2,T_knum;
    MPI_Comm *MPI_CommWD1,*MPI_CommWD2;
    int *Comm_World_StartID1,*Comm_World_StartID2;
    int Num_Comm_World1,Num_Comm_World2;
    int *NPROCS_ID1,*Comm_World1,*NPROCS_WD1;
    int *NPROCS_ID2,*Comm_World2,*NPROCS_WD2;
    int ***k_op,*T_k_op,**T_k_ID;
    dcomplex *rHs11_Cx,*rHs22_Cx,*rHs12_Cx;
    dcomplex *iHs11_Cx,*iHs22_Cx,*iHs12_Cx;
    dcomplex *Hs2_Cx,*Ss2_Cx,*Cs2_Cx;
    dcomplex *Hs_Cx,*Ss_Cx,*Cs_Cx;
    dcomplex **EVec1_Cx;
    double *T_KGrids1,*T_KGrids2,*T_KGrids3;
    double **ko_col,*ko_noncol,*ko,*koS;
    int ZERO=0, ONE=1,info,myid0,numprocs0,myid1,numprocs1,myid2,numprocs2;;
    int spinmax,i,j,k,ii,ij,ik,nblk_m,nblk_m2,wanA,spin,size_EVec1_Cx;
    double tmp,tmp1;
    int Kspace_grid1,Kspace_grid2,Kspace_grid3;

    /* MPI */ 
    MPI_Comm_size(comm1,&numprocs0);
    MPI_Comm_rank(comm1,&myid0);

    Kspace_grid1 = 1;
    Kspace_grid2 = TRAN_Kspace_grid2;
    Kspace_grid3 = TRAN_Kspace_grid3;

    /***********************************************
                  allocation of arrays 
    ***********************************************/

    spinmax = 1;
    Num_Comm_World1 = 1;

    n = 0;
    for (i=1; i<=atomnum; i++){
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = 2*n;

    is2 = (int*)malloc(sizeof(int)*numprocs0);
    ie2 = (int*)malloc(sizeof(int)*numprocs0);

    MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
    order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));
    My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
    SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
    SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

    ko = (double*)malloc(sizeof(double)*(n+1));
    koS = (double*)malloc(sizeof(double)*(n+1));
    ko_noncol = (double*)malloc(sizeof(double)*(n2+1));

    /* find size_H1 */

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    H1 = (double*)malloc(sizeof(double)*size_H1);
    S1 = (double*)malloc(sizeof(double)*size_H1);

    k_op = (int***)malloc(sizeof(int**)*Kspace_grid1);
    for (i=0;i<Kspace_grid1; i++) {
      k_op[i] = (int**)malloc(sizeof(int*)*Kspace_grid2);
      for (j=0;j<Kspace_grid2; j++) {
	k_op[i][j] = (int*)malloc(sizeof(int)*Kspace_grid3);
      }
    }

    for (i=0; i<Kspace_grid1; i++) {
      for (j=0; j<Kspace_grid2; j++) {
	for (k=0; k<Kspace_grid3; k++) {
	  k_op[i][j][k] = -999;
	}
      }
    }

    for (i=0; i<Kspace_grid1; i++) {
      for (j=0; j<Kspace_grid2; j++) {
	for (k=0; k<Kspace_grid3; k++) {

	  if ( k_op[i][j][k]==-999 ) {

	    k_inversion(i,j,k,Kspace_grid1,Kspace_grid2,Kspace_grid3,&ii,&ij,&ik);

	    if ( i==ii && j==ij && k==ik ) {
	      k_op[i][j][k]    = 1;
	    }
	    else {
	      k_op[i][j][k]    = 1;
	      k_op[ii][ij][ik] = 1;
	    }
	  }
	} /* k */
      } /* j */
    } /* i */
  
    /* find T_knum */

    T_knum = 0;
    for (i=0; i<Kspace_grid1; i++) {
      for (j=0; j<Kspace_grid2; j++) {
	for (k=0; k<Kspace_grid3; k++) {
	  if (0<k_op[i][j][k]){
	    T_knum++;
	  }
	}
      }
    }

    T_KGrids1 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
    T_k_op    = (int*)malloc(sizeof(int)*T_knum);

    T_k_ID    = (int**)malloc(sizeof(int*)*2);
    for (i=0; i<2; i++){
      T_k_ID[i] = (int*)malloc(sizeof(int)*T_knum);
    }

    EIGEN_Band = (double***)malloc(sizeof(double**)*spinmax);
    for (i=0; i<spinmax; i++){
      EIGEN_Band[i] = (double**)malloc(sizeof(double*)*T_knum);
      for (j=0; j<T_knum; j++){
	EIGEN_Band[i][j] = (double*)malloc(sizeof(double)*(n2+1));
	for (k=0; k<(n+1); k++) EIGEN_Band[i][j][k] = 1.0e+5;
      }
    }

    /***********************************************
      allocation of arrays for the first world 
      and 
      make the first level worlds 
    ***********************************************/
 
    NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0); 
    Comm_World1 = (int*)malloc(sizeof(int)*numprocs0); 
    NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
    MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

    Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

    /***********************************************
        allocation of arrays for the second world 
        and 
        make the second level worlds 
    ***********************************************/

    if (T_knum<=numprocs1) Num_Comm_World2 = T_knum;
    else                   Num_Comm_World2 = numprocs1;

    NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs1);
    Comm_World2 = (int*)malloc(sizeof(int)*numprocs1);
    NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
    Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
    MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);

    Make_Comm_Worlds(MPI_CommWD1[myworld1], myid1, numprocs1, Num_Comm_World2, 
		     &myworld2, MPI_CommWD2, NPROCS_ID2, Comm_World2, 
		     NPROCS_WD2, Comm_World_StartID2);

    MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

    double av_num;
    int ke,ks,n3;

    n3 = n2;

    av_num = (double)n3/(double)numprocs2;
    ke = (int)(av_num*(double)(myid2+1));
    ks = (int)(av_num*(double)myid2) + 1;
    if (myid1==0)             ks = 1;
    if (myid1==(numprocs2-1)) ke = n3;
    k = ke - ks + 2;

    EVec1_Cx = (dcomplex**)malloc(sizeof(dcomplex*)*spinmax);
    for (spin=0; spin<spinmax; spin++){
      EVec1_Cx[spin] = (dcomplex*)malloc(sizeof(dcomplex)*k*n3);
    }
    size_EVec1_Cx = spinmax*k*n3; 

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    np_cols = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols)==0) break;
      np_cols--;
    } while (np_cols>=2);
    np_rows = numprocs2/np_cols;

    nblk_m = NBLK;
    while((nblk_m*np_rows>n || nblk_m*np_cols>n) && (nblk_m > 1)){
      nblk_m /= 2;
    }
    if(nblk_m<1) nblk_m = 1;

    MPI_Allreduce(&nblk_m,&nblk,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow = myid2/np_cols;
    my_pcol = myid2%np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols );

    bhandle2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt2 = bhandle2;
    Cblacs_gridinit(&ictxt2, "Row", np_rows, np_cols);
	
    Ss_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    Hs_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);

    MPI_Allreduce(&na_rows,&na_rows_max,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols,&na_cols_max,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    Cs_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max*na_cols_max);

    rHs11_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    rHs12_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    rHs22_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    iHs11_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    iHs12_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
    iHs22_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);

    descinit_(descS, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descH, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descC, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);

    /* ***************************************************
          setting for BLACS in the matrix size of n2
    *************************************************** */

    np_cols2 = (int)(sqrt((float)numprocs2));
    do{
      if((numprocs2%np_cols2)==0) break;
      np_cols2--;
    } while (np_cols2>=2);
    np_rows2 = numprocs2/np_cols2;

    nblk_m2 = NBLK;
    while((nblk_m2*np_rows2>n2 || nblk_m2*np_cols2>n2) && (nblk_m2 > 1)){
      nblk_m2 /= 2;
    }
    if(nblk_m2<1) nblk_m2 = 1;

    MPI_Allreduce(&nblk_m2,&nblk2,1,MPI_INT,MPI_MIN,mpi_comm_level1);

    my_prow2 = myid2/np_cols2;
    my_pcol2 = myid2%np_cols2;

    na_rows2 = numroc_(&n2, &nblk2, &my_prow2, &ZERO, &np_rows2 );
    na_cols2 = numroc_(&n2, &nblk2, &my_pcol2, &ZERO, &np_cols2 );

    bhandle1_2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt1_2 = bhandle1_2;

    Cblacs_gridinit(&ictxt1_2, "Row", np_rows2, np_cols2);
    Hs2_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);
    Ss2_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows2*na_cols2);

    MPI_Allreduce(&na_rows2,&na_rows_max2,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols2,&na_cols_max2,1,MPI_INT,MPI_MAX,MPI_CommWD2[myworld2]);
    Cs2_Cx = (dcomplex*)malloc(sizeof(dcomplex)*na_rows_max2*na_cols_max2);

    descinit_(descH2, &n2,  &n2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
    descinit_(descC2, &n2,  &n2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);
    descinit_(descS2, &n2,  &n2,  &nblk2,  &nblk2,  &ZERO, &ZERO, &ictxt1_2, &na_rows2,  &info);

    /***********************************************
               calling Band_DFT_NonCol
    ***********************************************/

    time5 += Band_DFT_NonCol(1,
			     Kspace_grid1,Kspace_grid2,Kspace_grid3,
			     SpinP_switch,H,iHNL,OLP[0],DM[0],EDM,Eele0,Eele1, 
			     MP,order_GA,ko_noncol,koS,EIGEN_Band,
			     H1,S1,
			     rHs11_Cx,rHs22_Cx,rHs12_Cx,iHs11_Cx,iHs22_Cx,iHs12_Cx,
			     EVec1_Cx,
			     Ss_Cx,Cs_Cx,Hs_Cx,
			     Ss2_Cx,Cs2_Cx,Hs2_Cx,
			     k_op,T_k_op,T_k_ID,
			     T_KGrids1,T_KGrids2,T_KGrids3,
			     myworld1,
			     NPROCS_ID1,
			     Comm_World1,
			     NPROCS_WD1,
			     Comm_World_StartID1,
			     MPI_CommWD1,
			     myworld2,
			     NPROCS_ID2,
			     NPROCS_WD2,
			     Comm_World2,
			     Comm_World_StartID2,
			     MPI_CommWD2);


    Eele0[0] = 0.0;
    Eele0[1] = 0.0;
 
    Eele1[0] = 0.0;
    Eele1[1] = 0.0;

    /***********************************************
                  freeing of arrays
    ***********************************************/

    spinmax = 1;
    Num_Comm_World1 = 1;

    n = 0;
    for (i=1; i<=atomnum; i++){
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }

    free(is2);
    free(ie2);

    free(MP);
    free(order_GA);
    free(My_NZeros);
    free(SP_NZeros);
    free(SP_Atoms);

    free(ko);
    free(koS);
    free(ko_noncol);

    free(H1);
    free(S1);

    for (i=0;i<Kspace_grid1; i++) {
      for (j=0;j<Kspace_grid2; j++) {
	free(k_op[i][j]);
      }
      free(k_op[i]);
    }
    free(k_op);

    free(T_KGrids1);
    free(T_KGrids2);
    free(T_KGrids3);
    free(T_k_op);

    for (i=0; i<2; i++){
      free(T_k_ID[i]);
    }
    free(T_k_ID);

    for (i=0; i<spinmax; i++){
      for (j=0; j<T_knum; j++){
	free(EIGEN_Band[i][j]);
      }
      free(EIGEN_Band[i]);
    }
    free(EIGEN_Band);

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    free(Ss_Cx);
    free(Hs_Cx);
    free(Cs_Cx);
    free(rHs11_Cx);
    free(rHs12_Cx);
    free(rHs22_Cx);
    free(iHs11_Cx);
    free(iHs12_Cx);
    free(iHs22_Cx);

    Cfree_blacs_system_handle(bhandle2);
    Cblacs_gridexit(ictxt2);

    /* ***************************************************
          setting for BLACS in the matrix size of n2
    *************************************************** */

    free(Hs2_Cx);
    free(Ss2_Cx);
    free(Cs2_Cx);

    Cfree_blacs_system_handle(bhandle1_2);
    Cblacs_gridexit(ictxt1_2);

    /***********************************************
        freeing of arrays for the second world 
        and 
        make the second level worlds 
    ***********************************************/

    for (spin=0; spin<spinmax; spin++){
      free(EVec1_Cx[spin]);
    }
    free(EVec1_Cx);

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    if ( Num_Comm_World2<=numprocs1 ) MPI_Comm_free(&MPI_CommWD2[myworld2]);

    free(MPI_CommWD2);
    free(Comm_World_StartID2);
    free(NPROCS_WD2);
    free(Comm_World2);
    free(NPROCS_ID2);

    /***********************************************
      freeing of arrays for the first world 
      and 
      make the first level worlds 
    ***********************************************/

    if ( Num_Comm_World2<=numprocs1 ) MPI_Comm_free(&MPI_CommWD1[myworld1]);

    free(MPI_CommWD1);
    free(Comm_World_StartID1);
    free(NPROCS_WD1);
    free(Comm_World1);
    free(NPROCS_ID1);
  }     

  dtime(&TEtime);
  return TEtime - TStime;
}





double TRAN_DFT_Original(
		/* input */
                MPI_Comm comm1,
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,   /* H */
		double *****ImNL, /* SO coupling */
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int *G2ID,
                int *F_G2M, 
		int **atv_ijk,
		int *List_YOUSO,
		/* output */
		double *****CDM,  /* output, density matrix */
                double *****iCDM,
		double *****EDM,  /* not used */
                double ***TRAN_DecMulP, /* output, partial DecMulP */
		double Eele0[2], double Eele1[2], 
                double ChemP_e0[2]) 
{
  int numprocs0,myid0,ID;
  int i2,i3,k_op,l1,l2,l3,RnB;
  int k,E_knum,S_knum,T_knum,num_kloop0;
  int parallel_mode,kloop,kloop0;
  int i,j,spin,MA_AN,GA_AN,wanA,tnoA;
  int LB_AN,GB_AN,wanB,tnoB;
  int **op_flag,*T_op_flag,*T_k_ID;
  double *T_KGrids2,*T_KGrids3;
  double k2,k3,tmp;
  double TStime,TEtime;

  int *MP;
  int *order_GA;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;  

  int size_H1;
  int myworld1; 
  int numprocs1,myid1;
  int Num_Comm_World1;
  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;

  double **DM1,*TDM1,**DM2,*TDM2;
  double **H1,*S1;
  double **H2;

  MPI_Comm_size(comm1,&numprocs0);
  MPI_Comm_rank(comm1,&myid0);

  dtime(&TStime);

  if (myid0==Host_ID){
    printf("<TRAN_DFT>\n");
  }

  /***************************************
    ChemP_e0 will be used in outputfile1
  ***************************************/

  ChemP_e0[0] = ChemP_e[0];
  ChemP_e0[1] = ChemP_e[1];

  /***********************************
            initialize CDM
  ************************************/

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++) {
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    CDM[spin][MA_AN][LB_AN][i][j] = 0.0;
            if(spin<2) {
             iCDM[spin][MA_AN][LB_AN][i][j] = 0.0;
            }
	  }
	}
     } 
    }
  }

  /***********************************
        set up operation flag
  ************************************/

  op_flag = (int**)malloc(sizeof(int*)*TRAN_Kspace_grid2); 
  for (i2=0; i2<TRAN_Kspace_grid2; i2++){
    op_flag[i2] = (int*)malloc(sizeof(int)*TRAN_Kspace_grid3); 
    for (i3=0; i3<TRAN_Kspace_grid3; i3++){
        op_flag[i2][i3] = 1;
    }
  }

  /***********************************
       one-dimentionalize for MPI
  ************************************/

  T_knum = 0;
  for (i2=0; i2<TRAN_Kspace_grid2; i2++){
    for (i3=0; i3<TRAN_Kspace_grid3; i3++){
      if (0<op_flag[i2][i3]) T_knum++;  
    }
  }         

  T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
  T_op_flag = (int*)malloc(sizeof(int)*T_knum);
  T_k_ID = (int*)malloc(sizeof(int)*T_knum);

  T_knum = 0;

  for (i2=0; i2<TRAN_Kspace_grid2; i2++){

    k2 = -0.5 + (2.0*(double)i2+1.0)/(2.0*(double)TRAN_Kspace_grid2) + Shift_K_Point;

    for (i3=0; i3<TRAN_Kspace_grid3; i3++){

      k3 = -0.5 + (2.0*(double)i3+1.0)/(2.0*(double)TRAN_Kspace_grid3) - Shift_K_Point;

      if (0<op_flag[i2][i3]){  

        T_KGrids2[T_knum] = k2;
        T_KGrids3[T_knum] = k3;
        T_op_flag[T_knum] = op_flag[i2][i3];

        T_knum++;        
      }
    }
  }

  /***************************************
    print out     
  ***************************************/

  if (myid0==Host_ID){

    printf(" KGrids2: ");fflush(stdout);

    for (i=0;i<TRAN_Kspace_grid2;i++){
      if (TRAN_Kspace_grid2==1)  k2 = 0.0;
      else                       k2 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)TRAN_Kspace_grid2) + Shift_K_Point;
      printf("%9.5f ",k2);fflush(stdout);
    }
    printf("\n");fflush(stdout);

    printf(" KGrids3: ");fflush(stdout);
    for (i=0;i<TRAN_Kspace_grid3;i++){
      if (TRAN_Kspace_grid3==1)  k3 = 0.0;
      else                       k3 = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)TRAN_Kspace_grid3) - Shift_K_Point;
      printf("%9.5f ",k3);fflush(stdout);
    }
    printf("\n");fflush(stdout);
  }

  /***************************************************
   allocate calculations of k-points into processors 
  ***************************************************/

  if (numprocs0<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs0; ID++){
      tmp = (double)T_knum/(double)numprocs0;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs0-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the first level world */
        T_k_ID[k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs0; 
    S_knum = (int)((double)myid0*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid0+1)*(tmp+1.0e-12)) - 1;
    if (myid0==(numprocs0-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World1 = T_knum;

    NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0);
    Comm_World1 = (int*)malloc(sizeof(int)*numprocs0);
    NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

    Make_Comm_Worlds(comm1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

    S_knum = myworld1;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
      /* ID in the first level world */
      T_k_ID[k] = Comm_World_StartID1[k];
    }

  }

  /*************************************************************
   one-dimensitonalize H and S and store them in a compact form  
  *************************************************************/

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  order_GA = (int*)malloc(sizeof(int)*(atomnum+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  size_H1 = Get_OneD_HS_Col(0, nh[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  DM1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (k=0; k<(SpinP_switch+1); k++){
    DM1[k] = (double*)malloc(sizeof(double)*size_H1);
    for (i=0; i<size_H1; i++) DM1[k][i] = 0.0;
  }

  TDM1 = (double*)malloc(sizeof(double)*size_H1);

  DM2 = (double**)malloc(sizeof(double*)*(1+1));
  for (k=0; k<(1+1); k++){
    DM2[k] = (double*)malloc(sizeof(double)*size_H1);
    for (i=0; i<size_H1; i++) DM2[k][i] = 0.0;
  }

  TDM2 = (double*)malloc(sizeof(double)*size_H1);

  H1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (k=0; k<(SpinP_switch+1); k++){
    H1[k] = (double*)malloc(sizeof(double)*size_H1);
  }

            /* s-o coupling */
  H2 = (double**)malloc(sizeof(double*)*(2+1));
  for (k=0; k<(2+1); k++){
    H2[k] = (double*)malloc(sizeof(double)*size_H1);
  }

  S1 = (double*)malloc(sizeof(double)*size_H1);

  for (k=0; k<(SpinP_switch+1); k++){
    size_H1 = Get_OneD_HS_Col(1, nh[k], H1[k], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

            /* s-o coupling */
  for (k=0; k<(2+1); k++){
    size_H1 = Get_OneD_HS_Col(1, ImNL[k], H2[k], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /***********************************************************
   start "kloop0"
  ***********************************************************/

  for (kloop0=0; kloop0<num_kloop0; kloop0++){

    kloop = S_knum + kloop0;

    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];
    k_op = T_op_flag[kloop];

    if (parallel_mode){

      TRAN_DFT_Kdependent(MPI_CommWD1[myworld1],
			  parallel_mode, numprocs1, myid1,
			  level_stdout, iter, SpinP_switch, k2, k3, k_op, order_GA,
                          DM1,DM2,H1,H2,S1,
                          nh, ImNL, CntOLP,
			  atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN,
			  natn, ncn, M2G, G2ID, atv_ijk, List_YOUSO, CDM, EDM, Eele0, Eele1);
    }
    else{

      TRAN_DFT_Kdependent(comm1,
			  parallel_mode, 1, 0,
			  level_stdout, iter, SpinP_switch, k2, k3, k_op, order_GA,
                          DM1,DM2,H1,H2,S1,
                          nh, ImNL, CntOLP,
			  atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN,
			  natn, ncn, M2G, G2ID, atv_ijk, List_YOUSO, CDM, EDM, Eele0, Eele1);
    }

  } /* kloop0 */

  /* MPI communication of DM1 */

  tmp = 1.0/(double)(TRAN_Kspace_grid2*TRAN_Kspace_grid3);
  
  for (k=0; k<=SpinP_switch; k++) {

    int itot0,Anum,Bnum;  
    double co,si,kRn;  

    MPI_Allreduce( DM1[k], TDM1, size_H1, MPI_DOUBLE, MPI_SUM, comm1);

    if (k<2) {
      MPI_Allreduce( DM2[k], TDM2, size_H1, MPI_DOUBLE, MPI_SUM, comm1);
    } 

    itot0 = 0; 

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      MA_AN = F_G2M[GA_AN]; 

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	RnB = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];

	for (i=0;i<tnoA;i++) {
	  for (j=0;j<tnoB;j++) {

	    if (1<=MA_AN && MA_AN<=Matomnum){   
              CDM[k][MA_AN][LB_AN][i][j] = TDM1[itot0]*tmp;
              /* for iCDM */
              if (k<2) {
               iCDM[k][MA_AN][LB_AN][i][j] = TDM2[itot0]*tmp;
              }
              /* until here */
	    }

            itot0++;        

	  }
	}
      }
    }

  } /* k */

  /***********************************************************
           overwrite CMD with l1=-1 or l1=1 by zero
  ***********************************************************/

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++) {

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
        RnB = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
        l1 = atv_ijk[RnB][1];
        l2 = atv_ijk[RnB][2];
        l3 = atv_ijk[RnB][3];

        /* DM between C-L or C-R */

        if (l1==-1 || l1==1){
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
	      CDM[spin][MA_AN][LB_AN][i][j] = 0.0;  
              /* for iCDM */
              if (spin<2) {
              iCDM[spin][MA_AN][LB_AN][i][j] = 0.0;
              }
	    }
	  }
        }

      }
    }
  }

  /***********************************************************
     calculate (partial) decomposed Mulliken population by 
               density matrix between CL or CR

     This contribution will be added in Mulliken_Charge.c
  ***********************************************************/

  {
    int MA_AN,GA_AN,wanA,tnoA,GA_AN_e,LB_AN_e,iside;
    int direction,Rn_e,GB_AN_e;
    double sum;

    /* initialize */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (spin=0; spin<=SpinP_switch; spin++) {
	for (i=0; i<tnoA; i++){
          TRAN_DecMulP[spin][MA_AN][i] = 0.0;
	}
      }
    }

    /* Left lead */

    iside = 0;
    direction = -1;

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      if (TRAN_region[GA_AN]%10==2){

        GA_AN_e =  TRAN_Original_Id[GA_AN];

        for (spin=0; spin<=SpinP_switch; spin++) {
	  for (i=0; i<tnoA; i++){

	    sum = 0.0;

	    for (LB_AN_e=0; LB_AN_e<=FNAN_e[iside][GA_AN_e]; LB_AN_e++){

	      GB_AN_e = natn_e[iside][GA_AN_e][LB_AN_e];
	      Rn_e = ncn_e[iside][GA_AN_e][LB_AN_e];
	      wanB = WhatSpecies_e[iside][GB_AN_e];
	      tnoB = Spe_Total_CNO_e[iside][wanB];
	      l1 = atv_ijk_e[iside][Rn_e][1];

	      if (l1==direction) {
		for (j=0; j<tnoB; j++){
		  sum += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*
                         DM_e[iside][0][spin][GA_AN_e][LB_AN_e][i][j]; 
		}
	      }
	    }
            
            if(spin==3) {
	       TRAN_DecMulP[spin][MA_AN][i] = -sum;
            } else {
               TRAN_DecMulP[spin][MA_AN][i] = sum;
            }

	  } /* i */
	} /* spin */
      }

    } /* MA_AN */

    /* Right lead */

    iside = 1;
    direction = 1;

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      if (TRAN_region[GA_AN]%10==3){

        GA_AN_e = TRAN_Original_Id[GA_AN];

        for (spin=0; spin<=SpinP_switch; spin++) {
	  for (i=0; i<tnoA; i++){

	    sum = 0.0;

	    for (LB_AN_e=0; LB_AN_e<=FNAN_e[iside][GA_AN_e]; LB_AN_e++){

	      GB_AN_e = natn_e[iside][GA_AN_e][LB_AN_e];
	      Rn_e = ncn_e[iside][GA_AN_e][LB_AN_e];
	      wanB = WhatSpecies_e[iside][GB_AN_e];
	      tnoB = Spe_Total_CNO_e[iside][wanB];
	      l1 = atv_ijk_e[iside][Rn_e][1];

	      if (l1==direction) {
		for (j=0; j<tnoB; j++){
		  sum += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*
                         DM_e[iside][0][spin][GA_AN_e][LB_AN_e][i][j]; 
		}
	      }
	    }

            if(spin==3) {
               TRAN_DecMulP[spin][MA_AN][i] = -sum;
            } else {
               TRAN_DecMulP[spin][MA_AN][i] = sum;
            }

	  } /* i */
	} /* spin */
      } 
    } /* MA_AN */

  }

  /* set Eele as 0 */

  Eele0[0] = 0.0;
  Eele0[1] = 0.0;
  Eele1[0] = 0.0;
  Eele1[1] = 0.0;

  /* free arrays */

  for (i2=0; i2<TRAN_Kspace_grid2; i2++){
    free(op_flag[i2]);
  }
  free(op_flag);

  free(T_KGrids2);
  free(T_KGrids3);
  free(T_op_flag);
  free(T_k_ID);

  if (T_knum<=numprocs0){

    if (Num_Comm_World1<=numprocs0){
      MPI_Comm_free(&MPI_CommWD1[myworld1]);
    }

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);
  }

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (k=0; k<(SpinP_switch+1); k++){
    free(DM1[k]);
  }
  free(DM1);

  free(TDM1);
  
  for (k=0; k<(1+1); k++){
    free(DM2[k]);
  }
  free(DM2);

  free(TDM2);

  for (k=0; k<(SpinP_switch+1); k++){
    free(H1[k]);
  }
  free(H1);

  for (k=0; k<(2+1); k++){
    free(H2[k]);
  }
  free(H2);

  free(S1);

  /* for elapsed time */
  dtime(&TEtime);
  if (myid0==Host_ID){
    printf("<TRAN_DFT> time=%lf\n", TEtime-TStime);fflush(stdout);
  }

  return TEtime - TStime;
}




/*
 *   calculate CDM from nh, CntOLP, ... 
 *
 *       an alternative routine for Cluster_DFT or Band_DFT 
 *
 */




static void TRAN_DFT_Kdependent(
			  /* input */
			  MPI_Comm comm1,
                          int parallel_mode,
                          int numprocs,
                          int myid,
			  int level_stdout,
			  int iter,
			  int SpinP_switch,
                          double k2,
                          double k3,
                          int k_op,
                          int *order_GA,
                          double **DM1,
                          double **DM2,
                          double **H1,
                          double **H2,
                          double *S1,
			  double *****nh,    /* H */
			  double *****ImNL,  /* SO-coupling */
			  double ****CntOLP, 
			  int atomnum,
			  int Matomnum,
			  int *WhatSpecies,
			  int *Spe_Total_CNO,
			  int *FNAN,
			  int **natn, 
			  int **ncn,
			  int *M2G, 
			  int *G2ID, 
			  int **atv_ijk,
			  int *List_YOUSO,
			  /* output */
			  double *****CDM,  /* output, charge density */
			  double *****EDM,  /* not used */
			  double Eele0[2], double Eele1[2]) /* not used */
{
  int i,j,k,iside,spinsize; 
  int *MP;
  int  iw,iw_method;
  dcomplex w, w_weight;
  dcomplex *GC,*GRL,*GRR;
  dcomplex *SigmaL, *SigmaR; 
  dcomplex *work1,*work2,*Gless;
  dcomplex **v2,*v20;
  double dum;
  double TStime,TEtime;
  int MA_AN, GA_AN, wanA, tnoA, Anum;
  int LB_AN, GB_AN, wanB, tnoB, Bnum; 
  int NUM_c0;

  static int ID;
  int Miw,iw0; 
  double time_a0, time_a1, time_a2; 

  /* setup MP */

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  TRAN_Set_MP(1, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, MP);
  NUM_c0 = NUM_c;  
  NUM_c = 2*NUM_c; 
 
  /* initialize */
  TRAN_Set_Value_double(SCC_nc,NUM_c*NUM_c,    0.0,0.0);
  TRAN_Set_Value_double(SCL_nc,NUM_c*NUM_e[0], 0.0,0.0);
  TRAN_Set_Value_double(SCR_nc,NUM_c*NUM_e[1], 0.0,0.0);
  TRAN_Set_Value_double(HCC_nc,NUM_c*NUM_c,    0.0,0.0);
  TRAN_Set_Value_double(HCL_nc,NUM_c*NUM_e[0], 0.0,0.0);
  TRAN_Set_Value_double(HCR_nc,NUM_c*NUM_e[1], 0.0,0.0);

  /* set Hamiltonian and overlap matrices of left and right leads */
  TRAN_Set_SurfOverlap_NC(comm1,"left", k2, k3);
  
  TRAN_Set_SurfOverlap_NC(comm1,"right",k2, k3);

  /* set CC, CL and CR */

  TRAN_Set_CentOverlap_NC(   comm1,
                          3,
                          SpinP_switch, 
                          k2,
                          k3,
                          order_GA,
                          H1,
                          H2,
                          S1,
                          nh,      /* input */
                          CntOLP,  /* input */
                          atomnum,
			  Matomnum,
			  M2G,
			  G2ID,
                          WhatSpecies,
                          Spe_Total_CNO,
                          FNAN,
                          natn,
                          ncn,
                          atv_ijk);

  if (MEASURE_TIME){
    dtime(&time_a0);
  }
  
  /* allocate */

  v2 = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));

  for (k=0; k<=SpinP_switch; k++) {
    v2[k] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c0*NUM_c0);
    TRAN_Set_Value_double( v2[k], NUM_c0*NUM_c0, 0.0, 0.0);
  }
  
  /* added by Y. Xiao */
  v20 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  TRAN_Set_Value_double( v20, NUM_c*NUM_c, 0.0, 0.0);

  GC = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  GRL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0]* NUM_e[0]);
  GRR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1]* NUM_e[1]);

  SigmaL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  work1 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  work2 = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  Gless = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c); 

  if (2<=level_stdout){
    printf("NUM_c=%d, NUM_e= %d %d\n",NUM_c, NUM_e[0], NUM_e[1]);
    printf("# of freq. to calculate G =%d\n",tran_omega_n_scf);
  }

  if      (SpinP_switch==0) spinsize = 1;
  else if (SpinP_switch==1) spinsize = 2;
  else if (SpinP_switch==3) spinsize = 1; /* The three below lines are not used for NC cal. */

  /**************************************************************
             calculation of Green functions at k and iw
  **************************************************************/

  for ( Miw=myid; Miw<tran_omega_n_scf; Miw+=numprocs ) {

    iw = Miw; 
    w = tran_omega_scf[iw];
    w_weight  = tran_omega_weight_scf[iw];
    iw_method = tran_integ_method_scf[iw]; 
    
    /*
    printf("Miw=%3d iw=%d of %d w=%le %le weight=%le %le method=%d\n",
            Miw,iw,tran_omega_n_scf, w.r,w.i, w_weight.r, w_weight.i, iw_method);
    */

    /* calculation of surface Green's function and self energy from the LEFT lead */

    iside = 0;

    TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_nc_e[iside], H01_nc_e[iside],
			       S00_nc_e[iside], S01_nc_e[iside], tran_surfgreen_iteration_max,
			       tran_surfgreen_eps, GRL);

    TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL_nc, SCL_nc, SigmaL);

    /* calculation of surface Green's function and self energy from the RIGHT lead */

    iside = 1;

    TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_nc_e[iside], H01_nc_e[iside],
			       S00_nc_e[iside], S01_nc_e[iside], tran_surfgreen_iteration_max,
			       tran_surfgreen_eps, GRR);

    TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR_nc, SCR_nc, SigmaR);

    /* calculation of central retarded Green's function */

    TRAN_Calc_CentGreen(w, NUM_c, SigmaL, SigmaR, HCC_nc, SCC_nc, GC);

    /***********************************************
       The non-equilibrium part is calculated by 
        using the lesser Green's function.
    ***********************************************/

    /* calculation of central advanced Green's function */

    if (iw_method==2){

      /* calculation of central lesser Green's function */

      TRAN_Calc_CentGreenLesser( w, ChemP_e, NUM_c, 
				 Order_Lead_Side, 
				 SigmaL, 
				 SigmaR, 
				 GC, 
				 HCC_nc, SCC_nc,
				 work1, work2, Gless);

    } /* if (iw_method==2) */

    /***********************************************
        add it to construct the density matrix
    ***********************************************/

    if      (iw_method==0)
      TRAN_Add_MAT( 0, NUM_c, w_weight, GC,     v20);
    else if (iw_method==1)
      TRAN_Add_MAT( 1, NUM_c, w_weight, GC,     v20);
    else if (iw_method==2)
      TRAN_Add_MAT( 1, NUM_c, w_weight, Gless,  v20);

    /*
    if (Miw==0){

      for (i=0; i<NUM_c; i++){
        for (j=0; j<NUM_c; j++){
          printf("Miw=%3d iw_method=%2d i=%3d j=%3d re=%15.12f im=%15.12f\n",Miw,iw_method,i,j,v20[NUM_c*j+i].r,v20[NUM_c*j+i].i);
        }
      }

      MPI_Finalize();
      exit(0);
    }
    */


  } /* iw */

  if (MEASURE_TIME){
    dtime(&time_a1);
  }

  /* distribution of spin-spin  */
  for (i=1; i<=NUM_c0; i++) {
    for (j=1; j<=NUM_c0; j++) {
    
      /* alpha-alpha */ 
      v2[0][(j-1)*NUM_c0+(i)-1].r = v20[(j-1)*NUM_c+(i)-1].r; 
      v2[0][(j-1)*NUM_c0+(i)-1].i = v20[(j-1)*NUM_c+(i)-1].i;
      /* beta-beta */
      v2[1][(j-1)*NUM_c0+(i)-1].r = v20[(j+NUM_c0-1)*NUM_c+(i+NUM_c0)-1].r;
      v2[1][(j-1)*NUM_c0+(i)-1].i = v20[(j+NUM_c0-1)*NUM_c+(i+NUM_c0)-1].i;
      /* alpha-beta */

      v2[2][(j-1)*NUM_c0+(i)-1].r = v20[(j+NUM_c0-1)*NUM_c+(i)-1].r;
      v2[2][(j-1)*NUM_c0+(i)-1].i = v20[(j+NUM_c0-1)*NUM_c+(i)-1].i; 

      /* beta-alpha */

      v2[3][(j-1)*NUM_c0+(i)-1].r = v20[(j-1)*NUM_c+(i+NUM_c0)-1].r; 
      v2[3][(j-1)*NUM_c0+(i)-1].i = v20[(j-1)*NUM_c+(i+NUM_c0)-1].i; 

    
    }
  } 

  /***********************************************
          calculation of density matrix
  ***********************************************/

  {
    int l1,l2,l3,RnB;
    int size_v3,itot,itot0,size_temp;
    double kRn,si,co,re,im;
    double *my_v3;
    double *v3;
    dcomplex **DM1_temp;

    /* find the size of v3 */

    size_v3 = 0;
    size_temp = 0;

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	for (i=0; i<tnoA; i++) {
	  for (j=0; j<tnoB; j++) {
	    size_v3++;
	    size_v3++;
   
            size_temp++;
	  }
	}
      }
    }  

    /* allocate arrays */
  
    my_v3 = (double*)malloc(sizeof(double)*size_v3);
    v3 = (double*)malloc(sizeof(double)*size_v3);
    /* allocate a temporal array for DM1 */
    DM1_temp = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
    for (k=0; k<=SpinP_switch; k++){
    DM1_temp[k] = (dcomplex*)malloc(sizeof(dcomplex)*size_temp);
    TRAN_Set_Value_double( DM1_temp[k], size_temp, 0.0, 0.0);
    }

    /* set up v3 */

#define v_idx(i,j)  ( ((j)-1)*NUM_c0 + (i)-1 ) 

    for (k=0; k<=SpinP_switch; k++) {

      itot = 0;

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

        for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  for (i=0; i<tnoA; i++) {
	    for (j=0; j<tnoB; j++) {
	      my_v3[itot++] = v2[k][ v_idx( Anum+i, Bnum+j) ].r;
	      my_v3[itot++] = v2[k][ v_idx( Anum+i, Bnum+j) ].i;
	    }
	  }
	}
      }  

      if (parallel_mode){
        MPI_Allreduce( my_v3, v3, itot, MPI_DOUBLE, MPI_SUM, comm1);
      }
      else {
        for (i=0; i<itot; i++) {
          v3[i] = my_v3[i]; 
	}
      }

      /* v3 -> CDM */

      itot = 0;
      itot0 = 0; 

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	  GB_AN = natn[GA_AN][LB_AN];
	  RnB = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];
	  l1 = atv_ijk[RnB][1];
	  l2 = atv_ijk[RnB][2];
	  l3 = atv_ijk[RnB][3];

	  kRn = k2*(double)l2 + k3*(double)l3;
	  si = (double)k_op*sin(2.0*PI*kRn);
	  co = (double)k_op*cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++) {
	    for (j=0; j<tnoB; j++) {
	      re = v3[itot++];
	      im = v3[itot++];

              /* divided by numprocs due to the later MPI_Allreduce  */
              DM1_temp[k][itot0].r += (re*co + im*si)/(double)numprocs; 
              DM1_temp[k][itot0].i += (im*co - re*si)/(double)numprocs;
              itot0++;

	    }
	  }
	}
      }

    } /* k */

    /* transfer the value of DM1_temp to DM1 */
    for (i=0; i<size_temp; i++) {

       DM1[0][i] += DM1_temp[0][i].r;
       DM1[1][i] += DM1_temp[1][i].r;
       DM1[2][i] += DM1_temp[2][i].r;
       DM1[3][i] -= DM1_temp[2][i].i; 

       DM2[0][i] += DM1_temp[0][i].i;
       DM2[1][i] += DM1_temp[1][i].i;  

    }
    

  /* free arrays */

    free(my_v3);
    free(v3);
    
    for (i=0; i<=SpinP_switch; i++){
      free(DM1_temp[i]);
    }
    free(DM1_temp);
 
  }

  if (MEASURE_TIME){
    MPI_Barrier(comm1);
    dtime(&time_a2);
    printf("TRAN_DFT(%d)> calculaiton (%le)\n",myid, time_a1-time_a0 ); 
  }


  /* free arrays */

  free(Gless);
  free(work2);
  free(work1);
  free(SigmaR);
  free(SigmaL);
  free(GRR);
  free(GRL);
  free(GC);

  for (k=0; k<=SpinP_switch; k++) {
    free(v2[k]);
  }
  free(v2);
  free(v20);

  free(MP);
}



static void TRAN_Add_MAT(
    int mode,
    int NUM_c,
    dcomplex w_weight,
    dcomplex *v,
    dcomplex *out
)

#define v_ref(i,j)        v[NUM_c*(j)+(i)]
#define out_ref(i,j)      out[NUM_c*(j)+(i)]

{
  int i,j;
  double re,im; 

  switch (mode) {

  case 0:

    for (i=0; i<NUM_c; i++) {
      for (j=0; j<NUM_c; j++) {
        re = v_ref(i,j).r + v_ref(j,i).r; 
        im = v_ref(i,j).i - v_ref(j,i).i; 
        out_ref(i,j).r += re*w_weight.r - im*w_weight.i;
        out_ref(i,j).i += re*w_weight.i + im*w_weight.r;
      }
    }

    break;

  case 1:

    for (i=0; i<NUM_c*NUM_c; i++) {
      out[i].r += v[i].r*w_weight.r - v[i].i*w_weight.i;
      out[i].i += v[i].r*w_weight.i + v[i].i*w_weight.r;
    }

    break;

  }

}

