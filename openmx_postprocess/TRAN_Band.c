/**********************************************************************
  TRAN_Band_Col.c:

     TRAN_Band_Col.c is a subroutine to perform band calculation 
     in the initial stage of the NEGF calculation.

  Log of TRAN_Band_Col.c:

     23/Nov/2019  Released by T. Ozaki

***********************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
#include "tran_prototypes.h"


/* variables for band calculations  */
double *Ss_Re,*Cs_Re,*Hs_Re;
double *rHs11_Re,*rHs12_Re,*rHs22_Re,*iHs11_Re,*iHs12_Re,*iHs22_Re;
double *CDM1,*EDM1,*PDM1,*Work1;
dcomplex *Hs2_Cx,*Ss2_Cx,*Cs2_Cx;
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

void Allocate_Free_Band_Col(int todo_flag);
void Allocate_Free_Band_NonCol(int todo_flag);



double TRAN_Band()
{
  static int firsttime=1;
  double ECE[20],Eele0[2],Eele1[2];
  double pUele,ChemP_e0[2];
  double Norm1,Norm2,Norm3,Norm4,Norm5;
  double TotalE;
  double S_coordinate[3];
  double tmp,tmp0;
  int Cnt_kind,Calc_CntOrbital_ON,spin,spinmax,m;
  int SCF_iter,SCF_iter_shift,SCF_MAX;
  int i,j,k,fft_charge_flag,M1N;
  int orbitalOpt_iter,LSCF_iter,OrbOpt_end; 
  int wanA,ETemp_controller;
  double time0,time1,time2,time3,time4;
  double time5,time6,time7,time8,time9;
  double time10,time11,time12,time13;
  double time14,time15,etime;
  double x,y,z;
  int po3,po,TRAN_Poisson_flag2;
  int  SucceedReadingDMfile,My_SucceedReadingDMfile;
  char file_DFTSCF[YOUSO10] = ".DFTSCF";
  char file_OrbOpt[YOUSO10] = ".OrbOpt";
  char operate[200];
  char *s_vec[20];
  double TStime,TEtime;
  FILE *fp_DFTSCF;
  FILE *fp_OrbOpt;

  int numprocs0,myid0;
  int numprocs1,myid1;

  dtime(&TStime);

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  /************************************************************
                       allocation of arrays
  *************************************************************/

  if      (SpinP_switch<=1)  Allocate_Free_Band_Col(1);        
  else if (SpinP_switch==3)  Allocate_Free_Band_NonCol(1);        

  /****************************************************


  ****************************************************/


  if (SpinP_switch<=1){

    time5 += Band_DFT_Col(LSCF_iter,
			  Kspace_grid1,Kspace_grid2,Kspace_grid3,
			  SpinP_switch,H,iHNL,OLP[0],DM[0],EDM,Eele0,Eele1, 
			  MP,order_GA,ko,koS,EIGEN_Band,
			  H1,S1,
			  CDM1,EDM1,
			  EVec1_Cx,
			  Ss_Cx,
			  Cs_Cx, 
			  Hs_Cx,
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
  }
  else{

    time5 += Band_DFT_NonCol(LSCF_iter,
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
  }


  /*******************************************************************
     freeing of arrays for cluster and band calculations
  *******************************************************************/

  if      (SpinP_switch<=1)  Allocate_Free_Band_Col(2);        
  else if (SpinP_switch==3)  Allocate_Free_Band_NonCol(2);        

  return 0.0;
}




void Allocate_Free_Band_Col(int todo_flag)
{
  static int firsttime=1;
  int ZERO=0, ONE=1,info,myid0,numprocs0,myid1,numprocs1,myid2,numprocs2;;
  int spinmax,i,j,k,ii,ij,ik,nblk_m,nblk_m2,wanA,spin,size_EVec1_Cx;
  double tmp,tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  /********************************************
              allocation of arrays 
  ********************************************/

  if (todo_flag==1){

    spinmax = SpinP_switch + 1; 
    Num_Comm_World1 = SpinP_switch + 1; 

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

    /* find size_H1 */

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, &tmp1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    H1 = (double*)malloc(sizeof(double)*size_H1);
    S1 = (double*)malloc(sizeof(double)*size_H1);
    CDM1 = (double*)malloc(sizeof(double)*size_H1);
    EDM1 = (double*)malloc(sizeof(double)*size_H1);

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
	      k_op[i][j][k]    = 2;
	      k_op[ii][ij][ik] = 0;
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
	EIGEN_Band[i][j] = (double*)malloc(sizeof(double)*(n+1));
	for (k=0; k<(n+1); k++) EIGEN_Band[i][j][k] = 1.0e+5;
      }
    }

    if (CDDF_on==1){
      H_Band_Col = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
      for (j=0; j<n+1; j++){
	H_Band_Col[j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
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

    n3 = n;
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

    descinit_(descS, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descH, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descC, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);

    /* PrintMemory */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_Band_Col: is2",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: ie2",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: MP",sizeof(int)*List_YOUSO[1],NULL);
      PrintMemory("Allocate_Free_Band_Col: order_GA",sizeof(int)*(List_YOUSO[1]+1),NULL);
      PrintMemory("Allocate_Free_Band_Col: My_NZeros",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: SP_NZeros",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: SP_Atoms",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: ko",sizeof(double)*(n+1),NULL);
      PrintMemory("Allocate_Free_Band_Col: koS",sizeof(double)*(n+1),NULL);
      PrintMemory("Allocate_Free_Band_Col: H1",sizeof(double)*size_H1,NULL);
      PrintMemory("Allocate_Free_Band_Col: S1",sizeof(double)*size_H1,NULL);
      PrintMemory("Allocate_Free_Band_Col: CDM1",sizeof(double)*size_H1,NULL);
      PrintMemory("Allocate_Free_Band_Col: EDM1",sizeof(double)*size_H1,NULL);
      PrintMemory("Allocate_Free_Band_Col: ko_op",sizeof(double)*Kspace_grid1*Kspace_grid2*Kspace_grid3,NULL);
      PrintMemory("Allocate_Free_Band_Col: T_KGrids1",sizeof(double)*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_Col: T_KGrids2",sizeof(double)*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_Col: T_KGrids3",sizeof(double)*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_Col: T_k_ID",sizeof(int)*2*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_Col: EIGEN_Band",sizeof(double)*spinmax*T_knum*(n+1),NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_ID1",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World1",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_WD1",sizeof(int)*Num_Comm_World1,NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World_StartID1",sizeof(int)*Num_Comm_World1,NULL);
      PrintMemory("Allocate_Free_Band_Col: MPI_CommWD1",sizeof(MPI_Comm)*Num_Comm_World1,NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_ID2",sizeof(int)*numprocs1,NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World2",sizeof(int)*numprocs1,NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_WD2",sizeof(int)*Num_Comm_World2,NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World_StartID2",sizeof(int)*Num_Comm_World2,NULL);
      PrintMemory("Allocate_Free_Band_Col: MPI_CommWD2",sizeof(MPI_Comm)*Num_Comm_World2,NULL);
      PrintMemory("Allocate_Free_Band_Col: EVec1_Cx",sizeof(dcomplex)*size_EVec1_Cx,NULL);
      PrintMemory("Allocate_Free_Band_Col: Ss_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_Col: Hs_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_Col: Cs_Cx",sizeof(dcomplex)*na_rows_max*na_cols_max,NULL);
      if (CDDF_on==1){
        PrintMemory("Allocate_Free_Band_Col: H_Band_Col",sizeof(dcomplex)*(n+1)*(n+1),NULL);
      }
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays 
  ********************************************/

  if (todo_flag==2){

    n = 0;
    for (i=1; i<=atomnum; i++){
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = 2*n;

    spinmax = SpinP_switch + 1; 

    free(is2);
    free(ie2);

    free(MP);
    free(order_GA);
    free(My_NZeros);
    free(SP_NZeros);
    free(SP_Atoms);

    free(ko);
    free(koS);

    /* find size_H1 */

    free(H1);
    free(S1);
    free(CDM1);
    free(EDM1);

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

    if (CDDF_on==1){
      for (j=0; j<(n+1); j++){
	free(H_Band_Col[j]);
      }
      free(H_Band_Col);
    }

    /***********************************************
        allocation of arrays for the second world 
        and 
        make the second level worlds 
    ***********************************************/

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);

    if ( Num_Comm_World2<=numprocs1 ) MPI_Comm_free(&MPI_CommWD2[myworld2]);

    free(NPROCS_ID2);
    free(Comm_World2);
    free(NPROCS_WD2);
    free(Comm_World_StartID2);
    free(MPI_CommWD2);

    for (spin=0; spin<spinmax; spin++){
      free(EVec1_Cx[spin]);
    }
    free(EVec1_Cx);

    /***********************************************
      allocation of arrays for the first world 
      and 
      make the first level worlds 
    ***********************************************/

    if ( Num_Comm_World1<=numprocs0 ) MPI_Comm_free(&MPI_CommWD1[myworld1]);

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    free(Ss_Cx);
    free(Hs_Cx);
    free(Cs_Cx);

    Cfree_blacs_system_handle(bhandle2);
    Cblacs_gridexit(ictxt2);

  }

}



void Allocate_Free_Band_NonCol(int todo_flag)
{
  static int firsttime=1;
  int ZERO=0, ONE=1,info,myid0,numprocs0,myid1,numprocs1,myid2,numprocs2;;
  int spinmax,i,j,k,ii,ij,ik,nblk_m,nblk_m2,wanA,spin,size_EVec1_Cx;
  double tmp,tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  /********************************************
              allocation of arrays 
  ********************************************/

  if (todo_flag==1){

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

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, &tmp1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

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

    if (CDDF_on==1){
      S_Band = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
      for (i=0; i<n+1; i++){
	S_Band[i] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
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

    int nblk_m2;

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

    /* for generalized Bloch theorem */
    if (GB_switch){
      koSU = (double*)malloc(sizeof(double)*(n+1));
      koSL = (double*)malloc(sizeof(double)*(n+1));
      SU_Band = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
      for (i=0; i<n+1; i++){
        SU_Band[i] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
      }

      SL_Band = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
      for (i=0; i<n+1; i++){
        SL_Band[i] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
      }
    }
    /* -------------------------------------- */

    /* PrintMemory */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_Band_NonCol: is2",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ie2",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: MP",sizeof(int)*List_YOUSO[1],NULL);
      PrintMemory("Allocate_Free_Band_NonCol: order_GA",sizeof(int)*(List_YOUSO[1]+1),NULL);
      PrintMemory("Allocate_Free_Band_NonCol: My_NZeros",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: SP_NZeros",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: SP_Atoms",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ko",sizeof(double)*(n+1),NULL);
      PrintMemory("Allocate_Free_Band_NonCol: koS",sizeof(double)*(n+1),NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ko_noncol",sizeof(double)*(n2+1),NULL);
      PrintMemory("Allocate_Free_Band_NonCol: H1",sizeof(double)*size_H1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: S1",sizeof(double)*size_H1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ko_op",sizeof(double)*Kspace_grid1*Kspace_grid2*Kspace_grid3,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_KGrids1",sizeof(double)*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_KGrids2",sizeof(double)*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_KGrids3",sizeof(double)*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_k_ID",sizeof(int)*2*T_knum,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: EIGEN_Band",sizeof(double)*spinmax*T_knum*(n2+1),NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_ID1",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World1",sizeof(int)*numprocs0,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_WD1",sizeof(int)*Num_Comm_World1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World_StartID1",sizeof(int)*Num_Comm_World1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: MPI_CommWD1",sizeof(MPI_Comm)*Num_Comm_World1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_ID2",sizeof(int)*numprocs1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World2",sizeof(int)*numprocs1,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_WD2",sizeof(int)*Num_Comm_World2,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World_StartID2",sizeof(int)*Num_Comm_World2,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: MPI_CommWD2",sizeof(MPI_Comm)*Num_Comm_World2,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: EVec1_Cx",sizeof(dcomplex)*size_EVec1_Cx,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Ss_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Hs_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Cs_Cx",sizeof(dcomplex)*na_rows_max*na_cols_max,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: rHs11_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: rHs12_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: rHs22_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: iHs11_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: iHs12_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: iHs22_Cx",sizeof(dcomplex)*na_rows*na_cols,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Hs2_Cx",sizeof(dcomplex)*na_rows2*na_cols2,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Ss2_Cx",sizeof(dcomplex)*na_rows2*na_cols2,NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Cs2_Cx",sizeof(dcomplex)*na_rows_max2*na_cols_max2,NULL);
      if (GB_switch){
	PrintMemory("Allocate_Free_Band_NonCol: koSU",sizeof(double)*(n+1),NULL);
	PrintMemory("Allocate_Free_Band_NonCol: koSL",sizeof(double)*(n+1),NULL);
	PrintMemory("Allocate_Free_Band_NonCol: SU_Band",sizeof(double)*(n+1)*(n+1),NULL);
	PrintMemory("Allocate_Free_Band_NonCol: SL_Band",sizeof(double)*(n+1)*(n+1),NULL);
      }
    }
    firsttime = 0;

  } /* if (todo_flag==1) */

  /********************************************
               freeing of arrays 
  ********************************************/

  if (todo_flag==2){

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

    if (CDDF_on==1){
      for (i=0; i<n+1; i++){
	free(S_Band[i]);
      }
      free(S_Band);
    }

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

    /* for generalized Bloch theorem */
    if (GB_switch){
      free(koSU);
      free(koSL);
      for (i=0; i<n+1; i++){
        free(SU_Band[i]);
      }
      free(SU_Band);

      for (i=0; i<n+1; i++){
        free(SL_Band[i]);
      }
      free(SL_Band);
    }
  }
}






void Allocate_Free_GridData(int todo_flag)
{
  static int firsttime=1;
  int i,m,spinmax,spin;

  MPI_Barrier(mpi_comm_level1);

  /********************************************
              allocation of arrays 
  ********************************************/

  if (todo_flag==1){

    ReVk = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (i=0; i<My_Max_NumGridB; i++) ReVk[i] = 0.0;

    ImVk = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (i=0; i<My_Max_NumGridB; i++) ImVk[i] = 0.0;

    if ( Mixing_switch==3 || Mixing_switch==4 ){

      if      (SpinP_switch==0)  spinmax = 1;
      else if (SpinP_switch==1)  spinmax = 2;
      else if (SpinP_switch==3)  spinmax = 3;

      ReRhoAtomk = (double*)malloc(sizeof(double)*My_Max_NumGridB);
      for (i=0; i<My_Max_NumGridB; i++) ReRhoAtomk[i] = 0.0;

      ImRhoAtomk = (double*)malloc(sizeof(double)*My_Max_NumGridB);
      for (i=0; i<My_Max_NumGridB; i++) ImRhoAtomk[i] = 0.0;

      ReRhok = (double***)malloc(sizeof(double**)*List_YOUSO[38]); 
      for (m=0; m<List_YOUSO[38]; m++){
	ReRhok[m] = (double**)malloc(sizeof(double*)*spinmax); 
	for (spin=0; spin<spinmax; spin++){
	  ReRhok[m][spin] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
	  for (i=0; i<My_Max_NumGridB; i++) ReRhok[m][spin][i] = 0.0;
	}
      }

      ImRhok = (double***)malloc(sizeof(double**)*List_YOUSO[38]); 
      for (m=0; m<List_YOUSO[38]; m++){
	ImRhok[m] = (double**)malloc(sizeof(double*)*spinmax); 
	for (spin=0; spin<spinmax; spin++){
	  ImRhok[m][spin] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
	  for (i=0; i<My_Max_NumGridB; i++) ImRhok[m][spin][i] = 0.0;
	}
      }

      Residual_ReRhok = (double**)malloc(sizeof(double*)*spinmax); 
      for (spin=0; spin<spinmax; spin++){
	Residual_ReRhok[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB*List_YOUSO[38]); 
	for (i=0; i<My_NumGridB_CB*List_YOUSO[38]; i++) Residual_ReRhok[spin][i] = 0.0;
      }

      Residual_ImRhok = (double**)malloc(sizeof(double*)*spinmax); 
      for (spin=0; spin<spinmax; spin++){
	Residual_ImRhok[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB*List_YOUSO[38]); 
	for (i=0; i<My_NumGridB_CB*List_YOUSO[38]; i++) Residual_ImRhok[spin][i] = 0.0;
      }
    }

    /* PrintMemory */

    if (firsttime && memoryusage_fileout) {
      PrintMemory("Allocate_Free_GridData: ReVk",sizeof(double)*My_Max_NumGridB,NULL);
      PrintMemory("Allocate_Free_GridData: ImVk",sizeof(double)*My_Max_NumGridB,NULL);
      PrintMemory("Allocate_Free_GridData: ReRhoAtomk",sizeof(double)*My_Max_NumGridB,NULL);
      PrintMemory("Allocate_Free_GridData: ImRhoAtomk",sizeof(double)*My_Max_NumGridB,NULL);
      PrintMemory("Allocate_Free_GridData: ReRhok",sizeof(double)*List_YOUSO[38]*spinmax*My_Max_NumGridB,NULL);
      PrintMemory("Allocate_Free_GridData: ImRhok",sizeof(double)*List_YOUSO[38]*spinmax*My_Max_NumGridB,NULL);
      PrintMemory("Allocate_Free_GridData: Residual_ReRhok",sizeof(double)*List_YOUSO[38]*spinmax*My_NumGridB_CB,NULL);
      PrintMemory("Allocate_Free_GridData: Residual_ImRhok",sizeof(double)*List_YOUSO[38]*spinmax*My_NumGridB_CB,NULL);
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays 
  ********************************************/

  if (todo_flag==2){

    free(ReVk);
    free(ImVk);

    if ( Mixing_switch==3 || Mixing_switch==4 ){

      if      (SpinP_switch==0)  spinmax = 1;
      else if (SpinP_switch==1)  spinmax = 2;
      else if (SpinP_switch==3)  spinmax = 3;

      free(ReRhoAtomk);
      free(ImRhoAtomk);

      for (m=0; m<List_YOUSO[38]; m++){
	for (spin=0; spin<spinmax; spin++){
	  free(ReRhok[m][spin]);
	}
        free(ReRhok[m]);
      }
      free(ReRhok);

      for (m=0; m<List_YOUSO[38]; m++){
	for (spin=0; spin<spinmax; spin++){
	  free(ImRhok[m][spin]);
	}
        free(ImRhok[m]);
      }
      free(ImRhok);

      for (spin=0; spin<spinmax; spin++){
	free(Residual_ReRhok[spin]);
      }
      free(Residual_ReRhok);

      for (spin=0; spin<spinmax; spin++){
	free(Residual_ImRhok[spin]);
      }
      free(Residual_ImRhok);
    }
  }
}


