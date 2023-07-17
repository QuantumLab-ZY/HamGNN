/**********************************************************************
  truncation.c:

     truncation.c is a subrutine to divide a large system
     to small systems and set grid data.

  Log of truncation.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h" 
#include "mpi.h"
#include "omp.h"
#include "tran_prototypes.h"

#define  measure_time   0


static void Fixed_FNAN_SNAN();
int Set_Periodic(int CpyN, int Allocate_switch); //Yang Zhong
void Free_truncation(int CpyN, int TN, int Free_switch); //Yang Zhong
static void free_arrays_truncation0();
static void Trn_System(int MD_iter, int CpyCell, int TCpyCell);
static void Estimate_Trn_System(int CpyCell, int TCpyCell);
static void Check_System();
static void Set_RMI();
static void Output_Connectivity(FILE *fp);
static void UCell_Box(int MD_iter, int estimate_switch, int CpyCell);
static void Set_Inf_SndRcv();
static void Construct_MPI_Data_Structure_Grid();
static void Construct_ONAN();
static void Set_up_for_DCLNO();

int TFNAN,TFNAN2,TSNAN,TSNAN2;




double truncation(int MD_iter,int UCell_flag)
{ 
  static int firsttime=1;
  int i,j,k,m,ct_AN,h_AN,Gh_AN,Mc_AN,Gc_AN,s1,s2;
  int tno0,tno1,tno2,Cwan,Hwan,N,so,spin;
  int num,n2,wanA,wanB,Gi,NO1;
  int Anum,Bnum,fan,csize,NUM;
  int size_Orbs_Grid,size_COrbs_Grid;
  int size_Orbs_Grid_FNAN;
  int size_H0,size_CntH0,size_H,size_CntH;
  int size_HNL,size_HisH1,size_HisH2;
  int size_iHNL,size_iHNL0,size_iCntHNL;
  int size_DS_NL,size_CntDS_NL;
  int size_NumOLG,size_OLP,size_OLP_CH,size_CntOLP;
  int size_OLP_L,size_OLP_p;
  int size_HVNA,size_DS_VNA,size_CntDS_VNA;
  int size_HVNA2,size_CntHVNA2;
  int size_HVNA3,size_CntHVNA3;
  int size_H_Hub,size_DM_onsite,size_DM0;
  int size_v_eff,size_NC_OcpN,size_NC_v_eff;
  int size_S12,size_iDM;
  int size_ResidualDM,size_iResidualDM;
  int size_Krylov_U,size_EC_matrix;
  int size_H_Zeeman_NCO;
  int size_TRAN_DecMulP,size_DecEkin,size_SubSpace_EC;
  int My_Max_GridN_Atom,My_Max_OneD_Grids,My_Max_NumOLG;
  int numprocs,myid,ID;
  double r0,scale_rc0;
  int po,po0;
  double time0,time1,time2,time3,time4,time5,time6,time7;
  double time8,time9,time10,time11,time12,time13,time14;
  double time15,time16,time17,time18,time19,time20,time21;
  double stime,etime;

  double My_KryDH,My_KryDS;
  double KryDH,KryDS;

  char file_TRN[YOUSO10] = ".TRN";
  double TStime,TEtime;
  FILE *fp_TRN;
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  time0 = 0.0; time1 = 0.0; time2 = 0.0; time3 = 0.0; time4 = 0.0;
  time5 = 0.0; time6 = 0.0; time7 = 0.0; time8 = 0.0; time9 = 0.0;
  time10= 0.0; time11= 0.0; time12= 0.0; time13= 0.0; time14= 0.0;
  time15= 0.0; time16= 0.0; time17= 0.0; time18= 0.0; time19= 0.0;

  if (myid==Host_ID && 0<level_stdout){
    printf("\n*******************************************************\n"); 
    printf("        Analysis of neighbors and setting of grids        \n");
    printf("*******************************************************\n\n"); 
  }

  dtime(&TStime); 

  /****************************************************
     freeing of dynamically allocated arrays using
     FNAN and SNAN at the previous MD step
  ****************************************************/

  if (measure_time) dtime(&stime); 

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&time_per_atom[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  free_arrays_truncation0();

  if (measure_time){
    dtime(&etime); 
    time0 = etime - stime;
  }

  /****************************************************
            allocation of atoms to processors
  ****************************************************/

  if (measure_time) dtime(&stime); 

  if (2<=MD_iter){

    /*****************************
      last input
      0: atomnum, 
      1: elapsed time 
    *****************************/    

    if (Solver==5 || Solver==8 || Solver==11)  /* DC, Krylov, DC-LNO */
      Set_Allocate_Atom2CPU(MD_iter,0,1);  /* 1: elapsed time */
    else 
      Set_Allocate_Atom2CPU(MD_iter,0,0);  /* 0: atomnum */
  }

  if (measure_time){
    dtime(&etime); 
    time1 = etime - stime;
  }

  /****************************************************
                       Truncation
  ****************************************************/

  if (MD_iter==1){

    /****************************************************
     find:

       List_YOUSO[4] 

     Note: 

       In Set_Periodic(CpyCell,0)
         allocation of arrays:
            ratv           (global)
            atv            (global
            atv_ijk        (global)
            and
        Generation_ATV(CpyN);

       In Allocate_Arrays(3) 
         freeing and allocation of arrays:
            natn           (global)
            ncn            (global)
            Dis            (global)

       In Free_truncation(CpyCell,TCpyCell,0)
         freeing of arrays:
            ratv           (global)
            atv            (global)
            atv_ijk        (global)

    ****************************************************/

    CpyCell = 0;
    po = 0;
    TFNAN = 0;
    TSNAN = 0;

    do{

      CpyCell++;

      /**********************************************
           allocation of arrays listed above and 
                 Generation_ATV(CpyN);
      **********************************************/

      if (measure_time) dtime(&stime); 

      TCpyCell = Set_Periodic(CpyCell,0);

      if (measure_time){
	dtime(&etime); 
	time2 += etime - stime;
      }

      /**********************************************
        find Max_FSNAN by the physical truncation
          for allocation of natn, ncn, and Dis 
      **********************************************/

      if (measure_time) dtime(&stime); 

      Estimate_Trn_System(CpyCell,TCpyCell);

      if (measure_time){
	dtime(&etime); 
	time3 += etime - stime;
      }

      /**********************************************
              allocation of natn, ncn, and Dis 
      **********************************************/

      if (measure_time) dtime(&stime); 

      Allocate_Arrays(3);

      if (measure_time){
	dtime(&etime); 
	time4 += etime - stime;
      }

      /**********************
       find TFNAN and TSNAN
      **********************/

      TFNAN2 = TFNAN;
      TSNAN2 = TSNAN;

      if (measure_time) dtime(&stime); 

      Trn_System(MD_iter,CpyCell,TCpyCell);

      if (measure_time){
	dtime(&etime); 
	time5 += etime - stime;
      }

      if ( TFNAN==TFNAN2 && TSNAN==TSNAN2 && (Solver==5 || Solver==6 || Solver==8 || Solver==10 || Solver==11 || pop_anal_aow_flag==1)) po++;
      else if (TFNAN==TFNAN2 && (Solver==2 || Solver==3 || Solver==4 || Solver==7 || Solver==9 || Solver==12) ) po++;
      else if (CellNN_flag==1)                                                                    po++;

      /**********************************************
         freeing of arrays which are allocated in
                 Set_Periodic(CpyCell,0)
      **********************************************/

      if (measure_time) dtime(&stime); 

      Free_truncation(CpyCell,TCpyCell,0);

      if (measure_time){
	dtime(&etime); 
	time6 += etime - stime;
      }

    } while (po==0);

    List_YOUSO[4] = CpyCell;

    if (measure_time) dtime(&stime); 

    TCpyCell = Set_Periodic(CpyCell,0);
    my_CpyCell = CpyCell; // added by Yang Zhong

    if (measure_time){
      dtime(&etime); 
      time2 += etime - stime;
    }

    if (measure_time) dtime(&stime); 

    Estimate_Trn_System(CpyCell,TCpyCell);

    if (measure_time){
      dtime(&etime); 
      time3 += etime - stime;
    }

    if (measure_time) dtime(&stime); 

    Allocate_Arrays(3);

    if (measure_time){
      dtime(&etime); 
      time7 += etime - stime;
    }

    if (measure_time) dtime(&stime); 

    Trn_System(MD_iter,CpyCell,TCpyCell);

    if (measure_time){
      dtime(&etime); 
      time5 += etime - stime;
    }

    if (measure_time) dtime(&stime); 

    Set_Inf_SndRcv();

    if (measure_time){
      dtime(&etime); 
      time8 += etime - stime;
    }

    if (measure_time) dtime(&stime); 

    Set_RMI();

    if (measure_time){
      dtime(&etime); 
      time9 += etime - stime;
    }

  } /* if (MD_iter==1) */ 

  else{

    TCpyCell = Set_Periodic(CpyCell,0);
    Estimate_Trn_System(CpyCell,TCpyCell);
    Allocate_Arrays(3);
    Trn_System(MD_iter,CpyCell,TCpyCell);
    Set_Inf_SndRcv();
    Set_RMI();
  }

  if (2<=level_stdout){
    printf("List_YOUSO[4]=%2d\n",List_YOUSO[4]);
  } 

  /* for the EGAC method */
  if (Solver==10){
    Construct_ONAN(); 
  }

  /* for the DCLNO */
  if (Solver==11){
    Set_up_for_DCLNO(); 
  }

  /****************************************************
    allocation of arrays:

    NumOLG
  ****************************************************/

  if (measure_time) dtime(&stime); 

  FNAN[0] = 0;
  NumOLG = (int**)malloc(sizeof(int*)*(Matomnum+1));
  size_NumOLG = 0;
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0) Gc_AN = 0;
    else          Gc_AN = M2G[Mc_AN];
    NumOLG[Mc_AN] = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1));
    size_NumOLG += FNAN[Gc_AN] + 1;
  }
  alloc_first[5] = 0;
  
  /* PrintMemory */

  if (firsttime)
  PrintMemory("truncation: NumOLG", sizeof(int)*size_NumOLG, NULL);

  if (measure_time){
    dtime(&etime); 
    time10 += etime - stime;
  }

  /****************************************************
                  check the system type
  ****************************************************/

  if (measure_time) dtime(&stime); 

  if (MD_iter==1) Check_System();

  if (measure_time){
    dtime(&etime); 
    time11 += etime - stime;
  }

  /****************************************************
                       UCell_Box
  ****************************************************/

  if (MD_iter==1 && UCell_flag==1){

    /*************************************
      find 
            Max_GridN_Atom
            Max_OneD_Grids
    *************************************/

    Max_GridN_Atom = 0;
    Max_OneD_Grids = 0;
    if (2<=level_stdout) printf("\n***** UCell_Box(MD_iter,1,CpyCell) *****\n");

    if (measure_time) dtime(&stime); 
    UCell_Box(MD_iter,1,CpyCell);

    if (measure_time){
      dtime(&etime); 
      time12 += etime - stime;
    }

    if (measure_time) dtime(&stime); 

    My_Max_GridN_Atom = Max_GridN_Atom;

    MPI_Reduce(&My_Max_GridN_Atom, &Max_GridN_Atom, 1,
                MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Max_GridN_Atom, 1, MPI_INT, Host_ID, mpi_comm_level1);

    My_Max_OneD_Grids = Max_OneD_Grids;
    MPI_Reduce(&My_Max_OneD_Grids, &Max_OneD_Grids, 1,
                MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Max_OneD_Grids, 1, MPI_INT, Host_ID, mpi_comm_level1);
    List_YOUSO[11] = (int)(Max_GridN_Atom*ScaleSize) + 1;
    List_YOUSO[17] = (int)(Max_OneD_Grids*ScaleSize);

    if (2<=level_stdout){
      printf("Max_OneD_Grids=%2d\n",Max_OneD_Grids);
    }

    if (measure_time){
      dtime(&etime); 
      time13 += etime - stime;
    }

    /*************************************
      find 
            Max_NumOLG
    *************************************/

    Max_NumOLG = 0;
    if (2<=level_stdout) printf("\n***** UCell_Box(MD_iter,2,CpyCell) *****\n");

    if (measure_time) dtime(&stime); 
    UCell_Box(MD_iter,2,CpyCell);

    if (measure_time){
      dtime(&etime); 
      time14 += etime - stime;
    }

    My_Max_NumOLG = Max_NumOLG;
    MPI_Reduce(&My_Max_NumOLG, &Max_NumOLG, 1,
                MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Max_NumOLG, 1, MPI_INT, Host_ID, mpi_comm_level1);
    List_YOUSO[12] = (int)(Max_NumOLG*ScaleSize) + 1;

    if (2<=level_stdout){
      printf("YOUSO11=%i YOUSO12=%i YOUSO17=%i\n",
             List_YOUSO[11],List_YOUSO[12],List_YOUSO[17]);
    }

    /*************************************
      setting of
                 GListTAtoms1
                 GListTAtoms2
    *************************************/

    if (myid==Host_ID && 0<level_stdout)  printf("<UCell_Box> Info. of cutoff energy and num. of grids\n");

    if (measure_time) dtime(&stime); 
    UCell_Box(MD_iter,0,CpyCell);

    if (measure_time){
      dtime(&etime); 
      time15 += etime - stime;
    }

    My_Max_GridN_Atom = Max_GridN_Atom;
    MPI_Reduce(&My_Max_GridN_Atom, &Max_GridN_Atom, 1,
                MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Max_GridN_Atom, 1, MPI_INT, Host_ID, mpi_comm_level1);

    My_Max_OneD_Grids = Max_OneD_Grids;
    MPI_Reduce(&My_Max_OneD_Grids, &Max_OneD_Grids, 1,
                MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Max_OneD_Grids, 1, MPI_INT, Host_ID, mpi_comm_level1);
    List_YOUSO[11] = (int)(Max_GridN_Atom*ScaleSize) + 1;
    List_YOUSO[17] = (int)(Max_OneD_Grids*ScaleSize);
  }

  else if (UCell_flag==1) {

    if (myid==Host_ID && 0<level_stdout) printf("<UCell_Box> Info. of cutoff energy and num. of grids\n");

    UCell_Box(MD_iter,0,CpyCell);

    List_YOUSO[11] = (int)(Max_GridN_Atom*ScaleSize) + 1;
    List_YOUSO[17] = (int)(Max_OneD_Grids*ScaleSize);
  }

  /****************************************************
    allocation of arrays:

     H0
     CntH0
     OLP
     CntOLP
     H
     CntH
     iCntHNL
     DS_NL
     CntDS_NL
     DM
     ResidualDM
     EDM
     PDM
     double CntCoes[Matomnum+MatomnumF+1][YOUSO7][YOUSO24];
     HVNA
     DS_VNA
     CntDS_VNA
     HVNA2
     CntHVNA2
     H_Hub
     DM_onsite
     v_eff
     NC_OcpN
     NC_v_eff
  ****************************************************/

  if (measure_time) dtime(&stime); 

  /* H0 */  

  size_H0 = 0;
  H0 = (double*****)malloc(sizeof(double****)*4);
  for (k=0; k<4; k++){
    H0[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      H0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        H0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          H0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          size_H0 += tno1;
        }
      }
    }
  }

  /* CntH0 */  

  size_CntH0 = 0;

  if (Cnt_switch==1){

    CntH0 = (double*****)malloc(sizeof(double****)*4);
    for (k=0; k<4; k++){
      CntH0[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	CntH0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  CntH0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    CntH0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_CntH0 += tno1;
	  }
	}
      }
    }
  }

  /* HNL */  

  size_HNL = 0;
  HNL = (double*****)malloc(sizeof(double****)*List_YOUSO[5]);
  for (k=0; k<List_YOUSO[5]; k++){
    HNL[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      HNL[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        HNL[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          HNL[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          size_HNL += tno1;
	  for (j=0; j<tno1; j++) HNL[k][Mc_AN][h_AN][i][j] = 0.0;
        }
      }
    }
  }

  /* iHNL */  

  if ( SpinP_switch==3 ){

    size_iHNL = 0;
    iHNL = (double*****)malloc(sizeof(double****)*List_YOUSO[5]);
    for (k=0; k<List_YOUSO[5]; k++){
      iHNL[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	iHNL[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  iHNL[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iHNL[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	    for (j=0; j<tno1; j++) iHNL[k][Mc_AN][h_AN][i][j] = 0.0;
	  }
          size_iHNL += tno0*tno1;  
	}
      }
    }
  }

  /* iCntHNL */  

  if (SO_switch==1 && Cnt_switch==1){

    size_iCntHNL = 0;
    iCntHNL = (double*****)malloc(sizeof(double****)*List_YOUSO[5]);
    for (k=0; k<List_YOUSO[5]; k++){
      iCntHNL[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	iCntHNL[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  iCntHNL[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iCntHNL[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
            size_iCntHNL += tno1;  
	  }
	}
      }
    }
  }

  /* for core hole calculations */

  if (core_hole_state_flag==1){

    HCH = (double*****)malloc(sizeof(double****)*List_YOUSO[5]);
    for (k=0; k<List_YOUSO[5]; k++){
      HCH[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	HCH[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  HCH[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    HCH[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    for (j=0; j<tno1; j++) HCH[k][Mc_AN][h_AN][i][j] = 0.0;
	  }
	}
      }
    }

    /* iHCH */  

    if ( SpinP_switch==3 ){

      iHCH = (double*****)malloc(sizeof(double****)*List_YOUSO[5]);
      for (k=0; k<List_YOUSO[5]; k++){
	iHCH[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  iHCH[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    iHCH[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      iHCH[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	      for (j=0; j<tno1; j++) iHCH[k][Mc_AN][h_AN][i][j] = 0.0;
	    }
	  }
	}
      }
    }
  }

  /* H_Hub  --- added by MJ */  

  if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    size_H_Hub = 0;

    H_Hub = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      H_Hub[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	H_Hub[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  H_Hub[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    H_Hub[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	    for (j=0; j<tno1; j++) H_Hub[k][Mc_AN][h_AN][i][j] = 0.0;
	  }

          size_H_Hub += tno0*tno1;

	}
      }
    }
  }

  /* H_Zeeman_NCO */  

  if (Zeeman_NCO_switch==1){

    size_H_Zeeman_NCO = 0;

    H_Zeeman_NCO = (double***)malloc(sizeof(double**)*(Matomnum+1)); 
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_NO[Cwan];  
      }    

      H_Zeeman_NCO[Mc_AN] = (double**)malloc(sizeof(double*)*tno0); 
      for (i=0; i<tno0; i++){
        H_Zeeman_NCO[Mc_AN][i] = (double*)malloc(sizeof(double)*tno0); 
	for (j=0; j<tno0; j++) H_Zeeman_NCO[Mc_AN][i][j] = 0.0;
      }

      size_H_Zeeman_NCO += tno0*tno0;
    }
  }

  /* iHNL0 */

  size_iHNL0 = 0;

  if (SpinP_switch==3){

    iHNL0 = (double*****)malloc(sizeof(double****)*List_YOUSO[5]);
    for (k=0; k<List_YOUSO[5]; k++){
      iHNL0[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	iHNL0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  iHNL0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iHNL0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	    for (j=0; j<tno1; j++) iHNL0[k][Mc_AN][h_AN][i][j] = 0.0;
	  }
          size_iHNL0 += tno0*tno1;  
	}
      }
    }
  }

  /* OLP_L */  

  size_OLP_L = 0;

  OLP_L = (double*****)malloc(sizeof(double****)*3); 
  for (k=0; k<3; k++){

    OLP_L[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = F_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_NO[Cwan];  
      }    

      OLP_L[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_NO[Hwan];
	} 

	OLP_L[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	for (i=0; i<tno0; i++){
	  OLP_L[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	  size_OLP_L += tno1;
	}
      }
    }
  }

  /* OLP */  

  OLP = (double*****)malloc(sizeof(double****)*4);
  size_OLP = 0;
  for (k=0; k<4; k++){
    OLP[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+2)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<(Matomnum+MatomnumF+MatomnumS+2); Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
        fan = FNAN[Gc_AN]; 
      }
      else if (Mc_AN==(Matomnum+1)){
        tno0 = List_YOUSO[7];
        fan = List_YOUSO[8];
      }
      else if ( (Hub_U_switch==0 || Hub_U_occupation!=1 || core_hole_state_flag!=1) 
                && 0<k 
                && (Matomnum+1)<Mc_AN
                && Mc_AN<=(Matomnum+MatomnumF+MatomnumS)){

        Gc_AN = S_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = 1;
        fan = FNAN[Gc_AN]; 
      }
      else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){

        Gc_AN = S_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
        fan = FNAN[Gc_AN]; 
      }    
      else {
        tno0 = List_YOUSO[7];
        fan = List_YOUSO[8];
      }

      OLP[k][Mc_AN] = (double***)malloc(sizeof(double**)*(fan+1)); 
      for (h_AN=0; h_AN<=fan; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else if (Mc_AN==(Matomnum+1)){
          tno1 = List_YOUSO[7];
        }
        else if ( (Hub_U_switch==0 || Hub_U_occupation!=1 || core_hole_state_flag!=1) 
                  && 0<k 
                  && (Matomnum+1)<Mc_AN
                  && Mc_AN<=(Matomnum+MatomnumF+MatomnumS)){

          tno1 = 1;
	}
        else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){

          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
	}
        else{
          tno1 = List_YOUSO[7];
        } 

        OLP[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          OLP[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          size_OLP += tno1;
        }
      }
    }
  }

  /*** added by Ohwaki ***/

  /* OLP_p */

  if (0<=CLE_Type){

    OLP_p = (double*****)malloc(sizeof(double****)*4);
    size_OLP_p = 0;
    for (k=0; k<4; k++){
      OLP_p[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+2)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<(Matomnum+MatomnumF+MatomnumS+2); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	  fan = FNAN[Gc_AN]; 
	}
	else if (Mc_AN==(Matomnum+1)){
	  tno0 = List_YOUSO[7];
	  fan = List_YOUSO[8];
	}
	else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){

	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	  fan = FNAN[Gc_AN]; 
	}    
	else {
	  tno0 = List_YOUSO[7];
	  fan = List_YOUSO[8];
	}

	OLP_p[k][Mc_AN] = (double***)malloc(sizeof(double**)*(fan+1)); 
	for (h_AN=0; h_AN<=fan; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else if (Mc_AN==(Matomnum+1)){
	    tno1 = List_YOUSO[7];
	  }
	  else if ( (Hub_U_switch==0 || Hub_U_occupation!=1 || core_hole_state_flag!=1) 
		    && 0<k 
		    && (Matomnum+1)<Mc_AN
		    && Mc_AN<=(Matomnum+MatomnumF+MatomnumS)){

	    tno1 = 1;
	  }
	  else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  }
	  else{
	    tno1 = List_YOUSO[7];
	  } 

	  OLP_p[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    OLP_p[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_OLP_p += tno1;
	  }
	}
      }
    }
  }

  /*** added by Ohwaki (end) ***/

  /* OLP_CH */  

  if (core_hole_state_flag==1){

    OLP_CH = (double*****)malloc(sizeof(double****)*4);
    size_OLP_CH = 0;
    for (k=0; k<4; k++){
      OLP_CH[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+2)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<(Matomnum+MatomnumF+MatomnumS+2); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	  fan = FNAN[Gc_AN]; 
	}
	else if (Mc_AN==(Matomnum+1)){
	  tno0 = List_YOUSO[7];
	  fan = List_YOUSO[8];
	}
	else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){

	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	  fan = FNAN[Gc_AN]; 
	}    
	else {
	  tno0 = List_YOUSO[7];
	  fan = List_YOUSO[8];
	}

	OLP_CH[k][Mc_AN] = (double***)malloc(sizeof(double**)*(fan+1)); 
	for (h_AN=0; h_AN<=fan; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else if (Mc_AN==(Matomnum+1)){
	    tno1 = List_YOUSO[7];
	  }
	  else if ( (Hub_U_switch==0 || Hub_U_occupation!=1 || core_hole_state_flag!=1) 
		    && 0<k 
		    && (Matomnum+1)<Mc_AN
		    && Mc_AN<=(Matomnum+MatomnumF+MatomnumS)){

	    tno1 = 1;
	  }
	  else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  }
	  else{
	    tno1 = List_YOUSO[7];
	  } 

	  OLP_CH[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    OLP_CH[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_OLP_CH += tno1;
	  }
	}
      }
    }
  }

  /* CntOLP */  

  size_CntOLP = 0;

  if (Cnt_switch==1){
 
    CntOLP = (double*****)malloc(sizeof(double****)*4);
    for (k=0; k<4; k++){
      CntOLP[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	CntOLP[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  CntOLP[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    CntOLP[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_CntOLP += tno1;
	  }
	}
      }
    }
  }

  /* H */  

  size_H = 0;
  H = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    H[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = S_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      H[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        H[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          H[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          for (j=0; j<tno1; j++) H[k][Mc_AN][h_AN][i][j] = 0.0;
          size_H += tno1;  
        }
      }
    }
  }

  /* CntH */  

  size_CntH = 0;

  if (Cnt_switch==1){

    CntH = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      CntH[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}    

	CntH[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 

	  CntH[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    CntH[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_CntH += tno1;  
	  }
	}
      }
    }
  }

  /* DS_NL */  

  size_DS_NL = 0;
  DS_NL = (double******)malloc(sizeof(double*****)*(SO_switch+1));
  for (so=0; so<(SO_switch+1); so++){
    DS_NL[so] = (double*****)malloc(sizeof(double****)*4);
    for (k=0; k<4; k++){
      DS_NL[so][k] = (double****)malloc(sizeof(double***)*(Matomnum+2)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
          fan = FNAN[Gc_AN];
	}
	else if ( (Matomnum+1)<=Mc_AN ){
          fan = List_YOUSO[8];
          tno0 = List_YOUSO[7];
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
          fan = FNAN[Gc_AN];
	}    

	DS_NL[so][k][Mc_AN] = (double***)malloc(sizeof(double**)*(fan+1)); 
	for (h_AN=0; h_AN<(fan+1); h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
  	  else if ( (Matomnum+1)<=Mc_AN ){
	    tno1 = List_YOUSO[20];  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_VPS_Pro[Hwan] + 2;
	  } 

	  DS_NL[so][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    DS_NL[so][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    size_DS_NL += tno1;
	  }
	}
      }
    }
  }
 
  /* CntDS_NL */  

  size_CntDS_NL = 0;

  if (Cnt_switch==1){

    CntDS_NL = (double******)malloc(sizeof(double*****)*(SO_switch+1));
    for (so=0; so<(SO_switch+1); so++){
      CntDS_NL[so] = (double*****)malloc(sizeof(double****)*4); 
      for (k=0; k<4; k++){
	CntDS_NL[so][k] = (double****)malloc(sizeof(double***)*(Matomnum+2)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
            fan = FNAN[Gc_AN];
	  }
	  else if ( (Matomnum+1)<=Mc_AN ){
            fan = List_YOUSO[8];
            tno0 = List_YOUSO[7];
  	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
            fan = FNAN[Gc_AN];
	  }    

	  CntDS_NL[so][k][Mc_AN] = (double***)malloc(sizeof(double**)*(fan+1)); 
	  for (h_AN=0; h_AN<(fan+1); h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
  	    else if ( (Matomnum+1)<=Mc_AN ){
	      tno1 = List_YOUSO[20];  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_VPS_Pro[Hwan] + 2;
	    } 

	    CntDS_NL[so][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      CntDS_NL[so][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	      size_CntDS_NL += tno1;
	    }
	  }
	}
      }
    }
  }

  /* LNO_coes */  

  if (LNO_flag==1){

    LNO_coes = (double***)malloc(sizeof(double**)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      LNO_coes[k] = (double**)malloc(sizeof(double*)*(Matomnum+MatomnumF+MatomnumS+1)); 
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){
        LNO_coes[k][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
      }
    }

    LNO_pops = (double***)malloc(sizeof(double**)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      LNO_pops[k] = (double**)malloc(sizeof(double*)*(Matomnum+MatomnumF+MatomnumS+1)); 
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){
        LNO_pops[k][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }

    /* DM0 */
     
    size_DM0 = 0;
    DM0 = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      DM0[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	DM0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  DM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    DM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    for (j=0; j<tno1; j++) DM0[k][Mc_AN][h_AN][i][j] = 0.0;
	  }
	  size_DM0 += tno0*tno1;  
	}
      }
    }
  }

  /* for RMM-DIISH */

  size_HisH1 = 0;
  size_HisH2 = 0;

  if (Mixing_switch==5){

    /* HisH1 */

    HisH1 = (double******)malloc(sizeof(double*****)*List_YOUSO[39]); 
    for (m=0; m<List_YOUSO[39]; m++){
      HisH1[m] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
      for (k=0; k<=SpinP_switch; k++){
	HisH1[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  HisH1[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    HisH1[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      HisH1[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	      for (j=0; j<tno1; j++) HisH1[m][k][Mc_AN][h_AN][i][j] = 0.0;
	    }

            size_HisH1 += tno0*tno1;  

	  }
	}
      }
    }

    /* HisH2 */

    if (SpinP_switch==3){

      HisH2 = (double******)malloc(sizeof(double*****)*List_YOUSO[39]); 

      for (m=0; m<List_YOUSO[39]; m++){
	HisH2[m] = (double*****)malloc(sizeof(double****)*SpinP_switch); 
	for (k=0; k<SpinP_switch; k++){
	  HisH2[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    HisH2[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      HisH2[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	      for (i=0; i<tno0; i++){
		HisH2[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
		for (j=0; j<tno1; j++) HisH2[m][k][Mc_AN][h_AN][i][j] = 0.0;
	      }

	      size_HisH2 += tno0*tno1;  

	    }
	  }
	}
      }
    }

    /* ResidualH1 */

    ResidualH1 = (double******)malloc(sizeof(double*****)*List_YOUSO[39]); 
    for (m=0; m<List_YOUSO[39]; m++){
      ResidualH1[m] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
      for (k=0; k<=SpinP_switch; k++){
	ResidualH1[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  ResidualH1[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    ResidualH1[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      ResidualH1[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	      for (j=0; j<tno1; j++) ResidualH1[m][k][Mc_AN][h_AN][i][j] = 0.0;
	    }
	  }
	}
      }
    }

    /* ResidualH2 */

    if (SpinP_switch==3){

      ResidualH2 = (double******)malloc(sizeof(double*****)*List_YOUSO[39]); 

      for (m=0; m<List_YOUSO[39]; m++){
	ResidualH2[m] = (double*****)malloc(sizeof(double****)*SpinP_switch); 
	for (k=0; k<SpinP_switch; k++){
	  ResidualH2[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    ResidualH2[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      ResidualH2[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	      for (i=0; i<tno0; i++){
		ResidualH2[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
		for (j=0; j<tno1; j++) ResidualH2[m][k][Mc_AN][h_AN][i][j] = 0.0;
	      }
	    }
	  }
	}
      }
    }

  } /* if (Mixing_switch==5) */

  /* for RMM-DIIS and RMM-ADIIS */

  if (Mixing_switch==1 || Mixing_switch==6){

    size_HisH1 = 0;
    size_HisH2 = 0;

    /* HisH1 */

    HisH1 = (double******)malloc(sizeof(double*****)*List_YOUSO[39]); 
    for (m=0; m<List_YOUSO[39]; m++){
      HisH1[m] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
      for (k=0; k<=SpinP_switch; k++){
	HisH1[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  HisH1[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    HisH1[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      HisH1[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	      for (j=0; j<tno1; j++) HisH1[m][k][Mc_AN][h_AN][i][j] = 0.0;
	    }

            size_HisH1 += tno0*tno1;  

	  }
	}
      }
    }

    /* HisH2 */

    if (SpinP_switch==3){

      HisH2 = (double******)malloc(sizeof(double*****)*List_YOUSO[39]); 

      for (m=0; m<List_YOUSO[39]; m++){
	HisH2[m] = (double*****)malloc(sizeof(double****)*SpinP_switch); 
	for (k=0; k<SpinP_switch; k++){
	  HisH2[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    HisH2[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      HisH2[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	      for (i=0; i<tno0; i++){
		HisH2[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
		for (j=0; j<tno1; j++) HisH2[m][k][Mc_AN][h_AN][i][j] = 0.0;
	      }

	      size_HisH2 += tno0*tno1;  

	    }
	  }
	}
      }
    }

  } /* if (Mixing_switch==1 || Mixing_switch==6) */

  /* DM */

  DM = (double******)malloc(sizeof(double*****)*List_YOUSO[16]); 
  for (m=0; m<List_YOUSO[16]; m++){
    DM[m] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){
      DM[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
	}
	else{
          Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	DM[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  DM[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    DM[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	    for (j=0; j<tno1; j++) DM[m][k][Mc_AN][h_AN][i][j] = 0.0; 
	  }
	}
      }
    }
  }

  /* Partial_DM */

  if (cal_partial_charge==1){

    Partial_DM = (double*****)malloc(sizeof(double****)*2); 
    for (k=0; k<=1; k++){
      Partial_DM[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
	}
	else{
          Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	Partial_DM[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  Partial_DM[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    Partial_DM[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	    for (j=0; j<tno1; j++) Partial_DM[k][Mc_AN][h_AN][i][j] = 0.0; 
	  }
	}
      }
    }
  }

  /* DM_onsite   --- MJ */  

  if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
  
    size_DM_onsite = 0;

    DM_onsite = (double*****)malloc(sizeof(double****)*2); 

    for (m=0; m<2; m++){

      DM_onsite[m] = (double****)malloc(sizeof(double***)*(SpinP_switch+1)); 
      for (k=0; k<=SpinP_switch; k++){

	DM_onsite[m][k] = (double***)malloc(sizeof(double**)*(Matomnum+1)); 
	FNAN[0] = 0;

	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  
	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }  

	  h_AN = 0;
	      
	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  DM_onsite[m][k][Mc_AN] = (double**)malloc(sizeof(double*)*tno0); 

	  for (i=0; i<tno0; i++){
	    DM_onsite[m][k][Mc_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    for (j=0; j<tno1; j++) DM_onsite[m][k][Mc_AN][i][j] = 0.0;
	  }

	  size_DM_onsite += tno0*tno1;

	}
      }
    }
  }

  /*  v_eff  --- MJ */

  if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

    size_v_eff = 0;

    v_eff = (double****)malloc(sizeof(double***)*(SpinP_switch+1)); 
    for (k=0; k<=SpinP_switch; k++){

      v_eff[k] = (double***)malloc(sizeof(double**)*(Matomnum+MatomnumF+1)); 
      FNAN[0] = 0;

      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
	  
	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}  

	v_eff[k][Mc_AN] = (double**)malloc(sizeof(double*)*tno0); 

	for (i=0; i<tno0; i++){
	  v_eff[k][Mc_AN][i] = (double*)malloc(sizeof(double)*tno0); 
	  for (j=0; j<tno0; j++) v_eff[k][Mc_AN][i][j] = 0.0; 
	}

        size_v_eff += tno0*tno0;
      }
    }
  }

  /*  NC_OcpN */

  if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
       && SpinP_switch==3 ){

    size_NC_OcpN = 0;

    NC_OcpN = (dcomplex******)malloc(sizeof(dcomplex*****)*2); 

    for (m=0; m<2; m++){

      NC_OcpN[m] = (dcomplex*****)malloc(sizeof(dcomplex****)*2); 
      for (s1=0; s1<2; s1++){
	NC_OcpN[m][s1] = (dcomplex****)malloc(sizeof(dcomplex***)*2);
	for (s2=0; s2<2; s2++){
	  NC_OcpN[m][s1][s2] = (dcomplex***)malloc(sizeof(dcomplex**)*(Matomnum+1));
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }  

	    NC_OcpN[m][s1][s2][Mc_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0);

	    for (i=0; i<tno0; i++){
	      NC_OcpN[m][s1][s2][Mc_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno0);
	      for (j=0; j<tno0; j++)  NC_OcpN[m][s1][s2][Mc_AN][i][j] = Complex(0.0,0.0);
	    }

	    size_NC_OcpN += tno0*tno0;
	  }
	}
      }
    }           
  }

  /*  NC_v_eff */

  if ( (Hub_U_switch==1 
     || 1<=Constraint_NCS_switch
     || Zeeman_NCS_switch==1 
     || Zeeman_NCO_switch==1) && SpinP_switch==3 ){

    size_NC_v_eff = 0;

    NC_v_eff = (dcomplex*****)malloc(sizeof(dcomplex****)*2); 
    for (s1=0; s1<2; s1++){

      NC_v_eff[s1] = (dcomplex****)malloc(sizeof(dcomplex***)*2);

      for (s2=0; s2<2; s2++){

	NC_v_eff[s1][s2] = (dcomplex***)malloc(sizeof(dcomplex**)*(Matomnum+MatomnumF+1));

	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }  

	  NC_v_eff[s1][s2][Mc_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0);

	  for (i=0; i<tno0; i++){

	    NC_v_eff[s1][s2][Mc_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno0);
	    for (j=0; j<tno0; j++)  NC_v_eff[s1][s2][Mc_AN][i][j] = Complex(0.0,0.0);
	  }

	  size_NC_v_eff += tno0*tno0;
	}
      }
    }
  }

  /* ResidualDM */  

  size_ResidualDM = 0;
  if ( Mixing_switch==0 || Mixing_switch==1 || Mixing_switch==2 || Mixing_switch==6 ){

    ResidualDM = (double******)malloc(sizeof(double*****)*List_YOUSO[16]); 
    for (m=0; m<List_YOUSO[16]; m++){
      ResidualDM[m] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
      for (k=0; k<=SpinP_switch; k++){
	ResidualDM[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  ResidualDM[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    ResidualDM[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      ResidualDM[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	      for (j=0; j<tno1; j++) ResidualDM[m][k][Mc_AN][h_AN][i][j] = 0.0;
	    }

            size_ResidualDM += tno0*tno1; 
	  }
	}
      }
    }
  }

  /* iResidualDM */

  size_iResidualDM = 0;

  if ( (Mixing_switch==0 || Mixing_switch==1 || Mixing_switch==2 || Mixing_switch==6)
       && SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
        || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) ){

    iResidualDM = (double******)malloc(sizeof(double*****)*List_YOUSO[16]); 
    for (m=0; m<List_YOUSO[16]; m++){
      iResidualDM[m] = (double*****)malloc(sizeof(double****)*2); 
      for (k=0; k<2; k++){
	iResidualDM[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  iResidualDM[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    iResidualDM[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      iResidualDM[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
  	      for (j=0; j<tno1; j++) iResidualDM[m][k][Mc_AN][h_AN][i][j] = 0.0;
	    }

            size_iResidualDM += tno0*tno1; 
	  }
	}
      }
    }
  }

  else {
    iResidualDM = (double******)malloc(sizeof(double*****)*List_YOUSO[16]); 
    for (m=0; m<List_YOUSO[16]; m++){
      iResidualDM[m] = (double*****)malloc(sizeof(double****)*1); 
      iResidualDM[m][0] = (double****)malloc(sizeof(double***)*1); 
      iResidualDM[m][0][0] = (double***)malloc(sizeof(double**)*1); 
      iResidualDM[m][0][0][0] = (double**)malloc(sizeof(double*)*1); 
      iResidualDM[m][0][0][0][0] = (double*)malloc(sizeof(double)*1); 
    }   
  }

  /* EDM */  

  EDM = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    EDM[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      EDM[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        EDM[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          EDM[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
        }
      }
    }
  }

  /* PDM */  

  PDM = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    PDM[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      PDM[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        PDM[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          PDM[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
        }
      }
    }
  }

  /* iDM */  

  size_iDM = 0;
  iDM = (double******)malloc(sizeof(double*****)*List_YOUSO[16]);
  for (m=0; m<List_YOUSO[16]; m++){
    iDM[m] = (double*****)malloc(sizeof(double****)*2); 
    for (k=0; k<2; k++){
      iDM[m][k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	iDM[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  iDM[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iDM[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    for (j=0; j<tno1; j++)  iDM[m][k][Mc_AN][h_AN][i][j] = 0.0; 
	  }
	  size_iDM += tno0*tno1;
	}
      }
    }
  }

  /* S12 for recursion, or DC */  

  if (Solver==1 || Solver==5){

    size_S12 = 0;
    int *Msize,myid1;

    Msize = (int*)malloc(sizeof(int)*(Matomnum+2));

    if ( Solver==1 || Solver==5 ){

      S12 = (double***)malloc(sizeof(double**)*(Matomnum+1));

      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0) n2 = 1;
	else{
	  Gc_AN = M2G[Mc_AN];

	  num = 1;
	  for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	    Gi = natn[Gc_AN][i];
	    wanA = WhatSpecies[Gi];
	    num += Spe_Total_CNO[wanA];
	  }
	  n2 = num + 2;
	}

	S12[Mc_AN] = (double**)malloc(sizeof(double*)*n2);
	for (i=0; i<n2; i++){
	  S12[Mc_AN][i] = (double*)malloc(sizeof(double)*n2);
	}
	size_S12 += n2*n2;

	Msize[Mc_AN] = n2;
      }
  }

    free(Msize);
  }

  /* CntCoes */

  if (Cnt_switch==1){
    CntCoes = (double***)malloc(sizeof(double**)*(Matomnum+MatomnumF+1));
    for (i=0; i<=(Matomnum+MatomnumF); i++){
      CntCoes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	CntCoes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }
    }

    CntCoes_Species = (double***)malloc(sizeof(double**)*(SpeciesNum+1));
    for (i=0; i<(SpeciesNum+1); i++){
      CntCoes_Species[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
	CntCoes_Species[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }
    }
  }

  if (ProExpn_VNA==1){

    /* HVNA */  

    size_HVNA = 0;
    HVNA = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      HVNA[Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        HVNA[Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          HVNA[Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          size_HVNA += tno1;
        }
      }
    }

    /* DS_VNA */

    size_DS_VNA = 0;
    DS_VNA = (Type_DS_VNA*****)malloc(sizeof(Type_DS_VNA****)*4); 
    for (k=0; k<4; k++){

      DS_VNA[k] = (Type_DS_VNA****)malloc(sizeof(Type_DS_VNA***)*(Matomnum+2)); 

      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){
          
	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
          fan = FNAN[Gc_AN];
	}
	else if ( (Matomnum+1)<=Mc_AN ){
          fan = List_YOUSO[8];
          tno0 = List_YOUSO[7];
        }
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
          fan = FNAN[Gc_AN];
	}
          
	DS_VNA[k][Mc_AN] = (Type_DS_VNA***)malloc(sizeof(Type_DS_VNA**)*(fan+1)); 

	for (h_AN=0; h_AN<(fan+1); h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    tno1 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	  } 

	  DS_VNA[k][Mc_AN][h_AN] = (Type_DS_VNA**)malloc(sizeof(Type_DS_VNA*)*tno0); 
	  for (i=0; i<tno0; i++){
	    DS_VNA[k][Mc_AN][h_AN][i] = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*tno1);
	    size_DS_VNA += tno1;
	  }
	}
      }
    }

    /* CntDS_VNA */  

    if (Cnt_switch==1){

      size_CntDS_VNA = 0;
      CntDS_VNA = (Type_DS_VNA*****)malloc(sizeof(Type_DS_VNA****)*4); 
      for (k=0; k<4; k++){

	CntDS_VNA[k] = (Type_DS_VNA****)malloc(sizeof(Type_DS_VNA***)*(Matomnum+2));

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){
          
	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
            fan = FNAN[Gc_AN];
	  }
	  else if ( Mc_AN==(Matomnum+1) ){
            fan = List_YOUSO[8];
            tno0 = List_YOUSO[7];
          }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
            fan = FNAN[Gc_AN];
	  }
          
	  CntDS_VNA[k][Mc_AN] = (Type_DS_VNA***)malloc(sizeof(Type_DS_VNA**)*(fan+1)); 
	  for (h_AN=0; h_AN<(fan+1); h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      tno1 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	    } 

	    CntDS_VNA[k][Mc_AN][h_AN] = (Type_DS_VNA**)malloc(sizeof(Type_DS_VNA*)*tno0); 
	    for (i=0; i<tno0; i++){
	      CntDS_VNA[k][Mc_AN][h_AN][i] = (Type_DS_VNA*)malloc(sizeof(Type_DS_VNA)*tno1); 
	      size_CntDS_VNA += tno1;
	    }
	  }
	}
      }
    }

    /* HVNA2 */  

    size_HVNA2 = 0;
    HVNA2 = (double*****)malloc(sizeof(double****)*4);
    for (k=0; k<4; k++){
      HVNA2[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	HVNA2[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  HVNA2[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    HVNA2[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno0); 
	    size_HVNA2 += tno0;
	  }
	}
      }
    }

    /* HVNA3 */  

    size_HVNA3 = 0;
    HVNA3 = (double*****)malloc(sizeof(double****)*4);
    for (k=0; k<4; k++){
      HVNA3[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0)  Gc_AN = 0;
	else           Gc_AN = F_M2G[Mc_AN];

	HVNA3[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno0 = 1;
	  }
	  else{
            Gh_AN = natn[Gc_AN][h_AN];        
            Hwan = WhatSpecies[Gh_AN];
	    tno0 = Spe_Total_NO[Hwan];  
	  }    

	  HVNA3[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    HVNA3[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno0); 
	    size_HVNA3 += tno0;
	  }
	}
      }
    }

    /* CntHVNA2 */  

    if (Cnt_switch==1){

      size_CntHVNA2 = 0;
      CntHVNA2 = (double*****)malloc(sizeof(double****)*4);
      for (k=0; k<4; k++){
	CntHVNA2[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
	  }    

	  CntHVNA2[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    CntHVNA2[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      CntHVNA2[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno0); 
	      size_CntHVNA2 += tno0;
	    }
	  }
	}
      }
    }

    /* CntHVNA3 */  

    if (Cnt_switch==1){

      size_CntHVNA3 = 0;
      CntHVNA3 = (double*****)malloc(sizeof(double****)*4);
      for (k=0; k<4; k++){
	CntHVNA3[k] = (double****)malloc(sizeof(double***)*(Matomnum+1)); 
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0) Gc_AN = 0;
	  else          Gc_AN = F_M2G[Mc_AN];

	  CntHVNA3[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno0 = 1;
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno0 = Spe_Total_CNO[Hwan];  
	    }    

	    CntHVNA3[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	    for (i=0; i<tno0; i++){
	      CntHVNA3[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno0); 
	      size_CntHVNA3 += tno0;
	    }
	  }
	}
      }
    }

  }

  if (Solver==8) { /* Krylov subspace method */

    if (EKC_expand_core_flag==1)
      scale_rc0 = 1.10;
    else 
      scale_rc0 = 0.80;

    EKC_core_size = (int*)malloc(sizeof(int)*(Matomnum+1));
    scale_rc_EKC  = (double*)malloc(sizeof(double)*(Matomnum+1));

    for (i=1; i<=Matomnum; i++){

      scale_rc_EKC[i] = scale_rc0;

      ct_AN = M2G[i];
      wanA = WhatSpecies[ct_AN];

      /* find the nearest atom with distance of r0 */   

      r0 = 1.0e+10;
      for (h_AN=1; h_AN<=FNAN[ct_AN]; h_AN++){
        Gh_AN = natn[ct_AN][h_AN];
        wanB = WhatSpecies[Gh_AN];
        if (Dis[ct_AN][h_AN]<r0)  r0 = Dis[ct_AN][h_AN]; 
      }

      /* find atoms within scale_rc_EKC times r0 */

      po0 = 0;

      do {
      
        Anum = 0;
        for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
          Gh_AN = natn[ct_AN][h_AN];
          wanB = WhatSpecies[Gh_AN];
          if ( Dis[ct_AN][h_AN]<(scale_rc_EKC[i]*r0) ){
            Anum += Spe_Total_CNO[wanB];
	  }
        }

        if (EKC_expand_core_flag==1){
          if (30<Anum || 1<=po0) po0 = 3;
          else                   scale_rc_EKC[i] *= 1.2;  
	}
        else{
     	  po0 = 3;
	}

	po0++;

      } while(po0<2);

      EKC_core_size[i] = Anum;  
    }

    rlmax_EC  = (int*)malloc(sizeof(int)*(Matomnum+1));
    rlmax_EC2 = (int*)malloc(sizeof(int)*(Matomnum+1));

    size_Krylov_U = 0;

    Krylov_U = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (i=0; i<=SpinP_switch; i++){

      Krylov_U[i] = (double**)malloc(sizeof(double*)*(Matomnum+1));

      My_KryDH = 0.0;
      My_KryDS = 0.0;

      for (j=0; j<=Matomnum; j++){

	if (j==0){
	  Gc_AN = 0;
	  tno0 = 1;
          FNAN[0] = 1;
          rlmax_EC[0] = 1;
          Anum = 1; 
          Bnum = 0;
          EKC_core_size[0] = 1;
	}
	else{
	  Gc_AN = M2G[j];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  

          Anum = 0;  
          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];
            wanA = WhatSpecies[Gh_AN];
            Anum += Spe_Total_CNO[wanA];
	  }

          Bnum = 0;  
          for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){
            Gh_AN = natn[Gc_AN][h_AN];
            wanA = WhatSpecies[Gh_AN];
            Bnum += Spe_Total_CNO[wanA];
	  }

          if (KrylovH_order<Bnum){
            rlmax_EC[j] = (int)ceil((double)KrylovH_order/(double)EKC_core_size[j]);
	  }	
          else{
            rlmax_EC[j] = (int)ceil((double)Bnum/(double)EKC_core_size[j]);
          } 

          if (KrylovS_order<Bnum){
            rlmax_EC2[j] = (int)ceil((double)KrylovS_order/(double)EKC_core_size[j]);
	  }
          else{
            rlmax_EC2[j] = (int)ceil((double)Bnum/(double)EKC_core_size[j]);
          } 

          if (2<=level_stdout){
            printf("<Krylov parameters>  Gc_AN=%4d rlmax_EC=%3d rlmax_EC2=%3d EKC_core_size=%3d\n",
                    Gc_AN,rlmax_EC[j],rlmax_EC2[j],EKC_core_size[j]);
	  }

          My_KryDH += rlmax_EC[j]*EKC_core_size[j];
          My_KryDS += rlmax_EC2[j]*EKC_core_size[j];

	}    

	csize = Bnum + 2;
	
	Krylov_U[i][j] = (double*)malloc(sizeof(double)*rlmax_EC[j]*EKC_core_size[j]*csize);

        size_Krylov_U += rlmax_EC[j]*EKC_core_size[j]*csize;

      } /* j */

      if (i==0){
        MPI_Reduce(&My_KryDH, &KryDH, 1, MPI_DOUBLE, MPI_SUM, Host_ID, mpi_comm_level1);
        MPI_Reduce(&My_KryDS, &KryDS, 1, MPI_DOUBLE, MPI_SUM, Host_ID, mpi_comm_level1);

        if (myid==Host_ID && 2<=level_stdout){
	  printf("<Krylov parameters>  Av Krlov dimension H=%10.5f S=%10.5f\n",
                   KryDH/atomnum,KryDS/atomnum);
        }       
      }

    } /* i */

    size_EC_matrix = 0;

    EC_matrix = (double****)malloc(sizeof(double***)*(SpinP_switch+1));
    for (i=0; i<=SpinP_switch; i++){
      EC_matrix[i] = (double***)malloc(sizeof(double**)*(Matomnum+1));
      for (j=0; j<=Matomnum; j++){

	if (j==0){
	  tno0 = 1;
          EKC_core_size[0] = 1;
          rlmax_EC[0] = 1;
	}
	else{
	  Gc_AN = M2G[j];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

        EC_matrix[i][j] = (double**)malloc(sizeof(double*)*(rlmax_EC[j]*EKC_core_size[j]+1));

        for (k=0; k<(rlmax_EC[j]*EKC_core_size[j]+1); k++){
          EC_matrix[i][j][k] = (double*)malloc(sizeof(double)*(rlmax_EC[j]*EKC_core_size[j]+1));
	}

        size_EC_matrix += (rlmax_EC[j]*EKC_core_size[j]+1)*(rlmax_EC[j]*EKC_core_size[j]+1);
      }
    }

    /* find EKC_core_size_max */
        
    EKC_core_size_max = 0;
    for (j=1; j<=Matomnum; j++){
      if (EKC_core_size_max<EKC_core_size[j]) EKC_core_size_max = EKC_core_size[j];
    }

  } /* if (Solver==8) */

  /* NEGF */

  if (Solver==4){
 
    size_TRAN_DecMulP = (SpinP_switch+1)*(Matomnum+1)*List_YOUSO[7];

    TRAN_DecMulP = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin=0; spin<(SpinP_switch+1); spin++){
      TRAN_DecMulP[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	TRAN_DecMulP[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++) TRAN_DecMulP[spin][Mc_AN][i] = 0.0;
      }
    }
  }

  /* Energy decomposition */

  if (Energy_Decomposition_flag==1){

    size_DecEkin = 2*(Matomnum+1)*List_YOUSO[7];

    /* DecEkin */
    DecEkin = (double***)malloc(sizeof(double**)*2);
    for (spin=0; spin<2; spin++){
      DecEkin[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	DecEkin[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++) DecEkin[spin][Mc_AN][i] = 0.0;
      }
    }

    /* DecEv */
    DecEv = (double***)malloc(sizeof(double**)*2);
    for (spin=0; spin<2; spin++){
      DecEv[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	DecEv[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++) DecEv[spin][Mc_AN][i] = 0.0;
      }
    }

    /* DecEcon */
    DecEcon = (double***)malloc(sizeof(double**)*2);
    for (spin=0; spin<2; spin++){
      DecEcon[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	DecEcon[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++) DecEcon[spin][Mc_AN][i] = 0.0;
      }
    }

    /* DecEscc */
    DecEscc = (double***)malloc(sizeof(double**)*2);
    for (spin=0; spin<2; spin++){
      DecEscc[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	DecEscc[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++) DecEscc[spin][Mc_AN][i] = 0.0;
      }
    }

    /* DecEvdw */
    DecEvdw = (double***)malloc(sizeof(double**)*2);
    for (spin=0; spin<2; spin++){
      DecEvdw[spin] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	DecEvdw[spin][Mc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
	for (i=0; i<List_YOUSO[7]; i++) DecEvdw[spin][Mc_AN][i] = 0.0;
      }
    }

  } /* if (Energy_Decomposition_flag==1) */

  /* set zero */

  alloc_first[4] = 0;

  /* PrintMemory */

  if (firsttime){
  PrintMemory("truncation: H0",      sizeof(double)*size_H0,      NULL);
  PrintMemory("truncation: CntH0",   sizeof(double)*size_CntH0,   NULL);
  PrintMemory("truncation: HNL",     sizeof(double)*size_HNL,     NULL);
  PrintMemory("truncation: OLP",     sizeof(double)*size_OLP,     NULL);
  if (0<=CLE_Type){
  PrintMemory("truncation: OLP_p",   sizeof(double)*size_OLP_p,     NULL);
  }
  if (core_hole_state_flag==1){
  PrintMemory("truncation: OLP_CH",  sizeof(double)*size_OLP_CH,     NULL);
  }
  PrintMemory("truncation: CntOLP",  sizeof(double)*size_CntOLP,  NULL);
  PrintMemory("truncation: OLP_L",   sizeof(double)*size_OLP_L,   NULL);
  PrintMemory("truncation: H",       sizeof(double)*size_H,       NULL);
  PrintMemory("truncation: CntH",    sizeof(double)*size_CntH,    NULL);
  PrintMemory("truncation: DS_NL",   sizeof(double)*size_DS_NL,   NULL);
  PrintMemory("truncation: CntDS_NL",sizeof(double)*size_CntDS_NL,NULL);
  PrintMemory("truncation: HisH1",   sizeof(double)*size_HisH1,   NULL);
  PrintMemory("truncation: HisH2",   sizeof(double)*size_HisH2,   NULL);
  PrintMemory("truncation: ResidualH1",   sizeof(double)*size_HisH1,   NULL);
  PrintMemory("truncation: ResidualH2",   sizeof(double)*size_HisH2,   NULL);
  PrintMemory("truncation: DM",      sizeof(double)*List_YOUSO[16]*
                                                        size_H0,  NULL);
  PrintMemory("truncation: iDM",     sizeof(double)*size_iDM,     NULL);
  PrintMemory("truncation: ResidualDM",sizeof(double)*size_ResidualDM,  NULL);
  PrintMemory("truncation: EDM",     sizeof(double)*size_H0,      NULL);
  PrintMemory("truncation: PDM",     sizeof(double)*size_H0,      NULL);
  if (SO_switch==1 || SpinP_switch==3){
  PrintMemory("truncation: iResidualDM",sizeof(double)*size_iResidualDM,  NULL);
  PrintMemory("truncation: iHNL",    sizeof(double)*size_iHNL,    NULL);
  PrintMemory("truncation: iHNL0",   sizeof(double)*size_iHNL0,   NULL);
  }
  if (SO_switch==1 && Cnt_switch==1){
  PrintMemory("truncation: iCntHNL", sizeof(double)*size_iCntHNL,     NULL);
  }
  if (ProExpn_VNA==1){
  PrintMemory("truncation: DS_VNA",  sizeof(Type_DS_VNA)*size_DS_VNA,      NULL);
  PrintMemory("truncation: HVNA",    sizeof(double)*size_HVNA,        NULL);
  PrintMemory("truncation: HVNA2",   sizeof(double)*size_HVNA2,       NULL);
  PrintMemory("truncation: HVNA3",   sizeof(double)*size_HVNA3,       NULL);
  }
  if (Cnt_switch==1){
  PrintMemory("truncation: CntDS_VNA",sizeof(Type_DS_VNA)*size_CntDS_VNA,  NULL);
  PrintMemory("truncation: CntHVNA2", sizeof(double)*size_CntHVNA2,   NULL);
  PrintMemory("truncation: CntHVNA3", sizeof(double)*size_CntHVNA3,   NULL);
  }
  if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
  PrintMemory("truncation: H_Hub",     sizeof(double)*size_H_Hub,     NULL);
  PrintMemory("truncation: DM_onsite", sizeof(double)*size_DM_onsite, NULL);
  PrintMemory("truncation: v_eff",     sizeof(double)*size_v_eff,     NULL);
  }
  if (LNO_flag==1){
  PrintMemory("truncation: size_DM0",   sizeof(double)*size_DM0,     NULL);
  }
  if (Zeeman_NCO_switch==1){
  PrintMemory("truncation: size_H_Zeeman_NCO",   sizeof(double)*size_H_Zeeman_NCO,     NULL);
  }
  if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
        && SpinP_switch==3 ){
  PrintMemory("truncation: size_NC_OcpN",   sizeof(dcomplex)*size_NC_OcpN,     NULL);
  PrintMemory("truncation: size_NC_v_eff",  sizeof(dcomplex)*size_NC_v_eff,    NULL);
  }
  if (Solver==1 || Solver==5 || Solver==6){
  PrintMemory("truncation: S12",       sizeof(double)*size_S12,       NULL);
  }
  if (Solver==8){
  PrintMemory("truncation: Krylov_U",      sizeof(double)*size_Krylov_U,      NULL);
  PrintMemory("truncation: EC_matrix",     sizeof(double)*size_EC_matrix,     NULL);
  }
  if (Solver==4){
  PrintMemory("truncation: TRAN_DecMulP",  sizeof(double)*size_TRAN_DecMulP, NULL);
  }
  if (Energy_Decomposition_flag==1){
  PrintMemory("truncation: DecEkin",      sizeof(double)*size_DecEkin,        NULL);
  PrintMemory("truncation: DecEv",        sizeof(double)*size_DecEkin,        NULL);
  PrintMemory("truncation: DecEcon",      sizeof(double)*size_DecEkin,        NULL);
  PrintMemory("truncation: DecEscc",      sizeof(double)*size_DecEkin,        NULL);
  PrintMemory("truncation: DecEvdw",      sizeof(double)*size_DecEkin,        NULL);
  }
  if (core_hole_state_flag==1){
  PrintMemory("truncation: HCH",     sizeof(double)*size_HNL,     NULL);
  if ( SpinP_switch==3 ){
  PrintMemory("truncation: iHCH",    sizeof(double)*size_HNL,     NULL);
  }
  }

  }

  /****************************************************
     allocation of arrays:
  ****************************************************/

  if (UCell_flag==1){

    N = My_NumGridC;

    if (SpinP_switch==3){ /* spin non-collinear */
      Density_Grid = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Density_Grid[k] = (double*)malloc(sizeof(double)*N); 
        for (i=0; i<N; i++) Density_Grid[k][i] = 0.0;
      }
    }
    else{
      Density_Grid = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Density_Grid[k] = (double*)malloc(sizeof(double)*N); 
        for (i=0; i<N; i++) Density_Grid[k][i] = 0.0;
      }
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      Vxc_Grid = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Vxc_Grid[k] = (double*)malloc(sizeof(double)*N); 
      }
    }
    else{
      Vxc_Grid = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Vxc_Grid[k] = (double*)malloc(sizeof(double)*N); 
      }
    }

    RefVxc_Grid = (double*)malloc(sizeof(double)*N); 

    dVHart_Grid = (double*)malloc(sizeof(double)*N); 
    for (i=0; i<N; i++) dVHart_Grid[i] = 0.0;

    if (SpinP_switch==3){ /* spin non-collinear */
      Vpot_Grid = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Vpot_Grid[k] = (double*)malloc(sizeof(double)*N); 
      }
    }
    else{
      Vpot_Grid = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Vpot_Grid[k] = (double*)malloc(sizeof(double)*N); 
      }
    }

    /* arrays for the partitions B and C */

    if (SpinP_switch==3){ /* spin non-collinear */
      Density_Grid_B = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Density_Grid_B[k] = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
        for (i=0; i<My_NumGridB_AB; i++) Density_Grid_B[k][i] = 0.0;
      }
    }
    else{
      Density_Grid_B = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Density_Grid_B[k] = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
        for (i=0; i<My_NumGridB_AB; i++) Density_Grid_B[k][i] = 0.0;
      }
    }

    ADensity_Grid_B = (double*)malloc(sizeof(double)*My_NumGridB_AB); 

    PCCDensity_Grid_B = (double**)malloc(sizeof(double*)*2); 
    PCCDensity_Grid_B[0] = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
    PCCDensity_Grid_B[1] = (double*)malloc(sizeof(double)*My_NumGridB_AB); 

    for (i=0; i<My_NumGridB_AB; i++){
      PCCDensity_Grid_B[0][i] = 0.0;  
      PCCDensity_Grid_B[1][i] = 0.0;
    }

    dVHart_Grid_B = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
    for (i=0; i<My_Max_NumGridB; i++) dVHart_Grid_B[i] = 0.0;

    if ( (core_hole_state_flag==1 && Scf_RestartFromFile==1) || scf_coulomb_cutoff_CoreHole==1 ){
      dVHart_Periodic_Grid_B = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
      Density_Periodic_Grid_B = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

      for (i=0; i<My_Max_NumGridB; i++){
        dVHart_Periodic_Grid_B[i] = 0.0;
        Density_Periodic_Grid_B[i] = 0.0;
      }
    }

    RefVxc_Grid_B = (double*)malloc(sizeof(double)*My_NumGridB_AB); 

    if (SpinP_switch==3){ /* spin non-collinear */
      Vxc_Grid_B = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Vxc_Grid_B[k] = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
      }
    }
    else{
      Vxc_Grid_B = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Vxc_Grid_B[k] = (double*)malloc(sizeof(double)*My_NumGridB_AB);
      }
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      Vpot_Grid_B = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Vpot_Grid_B[k] = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
      }
    }
    else{
      Vpot_Grid_B = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Vpot_Grid_B[k] = (double*)malloc(sizeof(double)*My_NumGridB_AB);
      }
    }

    /* if ( Mixing_switch==7 ) */

    if ( Mixing_switch==7 ){

      int spinmax;

      if      (SpinP_switch==0)  spinmax = 1;
      else if (SpinP_switch==1)  spinmax = 2;
      else if (SpinP_switch==3)  spinmax = 3;

      ReVKSk = (double***)malloc(sizeof(double**)*List_YOUSO[38]); 
      for (m=0; m<List_YOUSO[38]; m++){
	ReVKSk[m] = (double**)malloc(sizeof(double*)*spinmax); 
	for (spin=0; spin<spinmax; spin++){
	  ReVKSk[m][spin] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
	  for (i=0; i<My_Max_NumGridB; i++) ReVKSk[m][spin][i] = 0.0;
	}
      }

      ImVKSk = (double***)malloc(sizeof(double**)*List_YOUSO[38]); 
      for (m=0; m<List_YOUSO[38]; m++){
	ImVKSk[m] = (double**)malloc(sizeof(double*)*spinmax); 
	for (spin=0; spin<spinmax; spin++){
	  ImVKSk[m][spin] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
	  for (i=0; i<My_Max_NumGridB; i++) ImVKSk[m][spin][i] = 0.0;
	}
      }

      Residual_ReVKSk = (double**)malloc(sizeof(double*)*spinmax); 
      for (spin=0; spin<spinmax; spin++){
	Residual_ReVKSk[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB*List_YOUSO[38]); 
	for (i=0; i<My_NumGridB_CB*List_YOUSO[38]; i++) Residual_ReVKSk[spin][i] = 0.0;
      }

      Residual_ImVKSk = (double**)malloc(sizeof(double*)*spinmax); 
      for (spin=0; spin<spinmax; spin++){
	Residual_ImVKSk[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB*List_YOUSO[38]); 
	for (i=0; i<My_NumGridB_CB*List_YOUSO[38]; i++) Residual_ImVKSk[spin][i] = 0.0;
      }
    }

    /* if (ProExpn_VNA==off) */
    if (ProExpn_VNA==0){
      VNA_Grid = (double*)malloc(sizeof(double)*N); 
      VNA_Grid_B = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
    }

    /* electric energy by electric field */
    if (E_Field_switch==1){
      VEF_Grid = (double*)malloc(sizeof(double)*N); 
      VEF_Grid_B = (double*)malloc(sizeof(double)*My_NumGridB_AB); 
    }

    /* arrays for the partitions D */

    PCCDensity_Grid_D = (double**)malloc(sizeof(double*)*2); 
    PCCDensity_Grid_D[0] = (double*)malloc(sizeof(double)*My_NumGridD); 
    PCCDensity_Grid_D[1] = (double*)malloc(sizeof(double)*My_NumGridD); 

    for (i=0; i<My_NumGridD; i++){
      PCCDensity_Grid_D[0][i] = 0.0; PCCDensity_Grid_D[1][i] = 0.0; 
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      Density_Grid_D = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Density_Grid_D[k] = (double*)malloc(sizeof(double)*My_NumGridD); 
        for (i=0; i<My_NumGridD; i++) Density_Grid_D[k][i] = 0.0;
      }
    }
    else{
      Density_Grid_D = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Density_Grid_D[k] = (double*)malloc(sizeof(double)*My_NumGridD); 
        for (i=0; i<My_NumGridD; i++) Density_Grid_D[k][i] = 0.0;
      }
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      Vxc_Grid_D = (double**)malloc(sizeof(double*)*4); 
      for (k=0; k<=3; k++){
        Vxc_Grid_D[k] = (double*)malloc(sizeof(double)*My_NumGridD); 
      }
    }
    else{
      Vxc_Grid_D = (double**)malloc(sizeof(double*)*2); 
      for (k=0; k<=1; k++){
        Vxc_Grid_D[k] = (double*)malloc(sizeof(double)*My_NumGridD);
      }
    }

    /* Orbs_Grid */
    size_Orbs_Grid = 0;
    Orbs_Grid = (Type_Orbs_Grid***)malloc(sizeof(Type_Orbs_Grid**)*(Matomnum+1)); 
    Orbs_Grid[0] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*1); 
    Orbs_Grid[0][0] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*1); 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      /* AITUNE */
      Orbs_Grid[Mc_AN] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*GridN_Atom[Gc_AN]); 
      int Nc;
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
        Orbs_Grid[Mc_AN][Nc] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*Spe_Total_NO[Cwan]); 
	size_Orbs_Grid += Spe_Total_NO[Cwan];
      }
      /* AITUNE */
    }

    /* COrbs_Grid */
    size_COrbs_Grid = 0;
    if (Cnt_switch!=0){
      COrbs_Grid = (Type_Orbs_Grid***)malloc(sizeof(Type_Orbs_Grid**)*(Matomnum+MatomnumF+1)); 
      COrbs_Grid[0] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*1); 
      COrbs_Grid[0][0] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*1); 
      for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
        Gc_AN = F_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        COrbs_Grid[Mc_AN] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*Spe_Total_CNO[Cwan]); 
        for (i=0; i<Spe_Total_CNO[Cwan]; i++){
          COrbs_Grid[Mc_AN][i] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*GridN_Atom[Gc_AN]); 
          size_COrbs_Grid += GridN_Atom[Gc_AN];
        }
      }
    }

    /* Orbs_Grid_FNAN */
    size_Orbs_Grid_FNAN = 0;
    Orbs_Grid_FNAN = (Type_Orbs_Grid****)malloc(sizeof(Type_Orbs_Grid***)*(Matomnum+1)); 
    Orbs_Grid_FNAN[0] = (Type_Orbs_Grid***)malloc(sizeof(Type_Orbs_Grid**)*1); 
    Orbs_Grid_FNAN[0][0] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*1); 
    Orbs_Grid_FNAN[0][0][0] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*1); 

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      Orbs_Grid_FNAN[Mc_AN] = (Type_Orbs_Grid***)malloc(sizeof(Type_Orbs_Grid**)*(FNAN[Gc_AN]+1)); 

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        Gh_AN = natn[Gc_AN][h_AN];

        if (G2ID[Gh_AN]!=myid){

	  Hwan = WhatSpecies[Gh_AN];
          NO1 = Spe_Total_NO[Hwan];

          /* AITUNE */
	  Orbs_Grid_FNAN[Mc_AN][h_AN] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*(NumOLG[Mc_AN][h_AN]+1)); 
	  int Nc;
	  for (Nc=0; Nc<(NumOLG[Mc_AN][h_AN]+1); Nc++){
	    Orbs_Grid_FNAN[Mc_AN][h_AN][Nc] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*NO1); 
	    size_Orbs_Grid_FNAN += NO1;
	  }
          /* AITUNE */
	}

        else {
          Orbs_Grid_FNAN[Mc_AN][h_AN] = (Type_Orbs_Grid**)malloc(sizeof(Type_Orbs_Grid*)*1); 
          Orbs_Grid_FNAN[Mc_AN][h_AN][0] = (Type_Orbs_Grid*)malloc(sizeof(Type_Orbs_Grid)*1); 
          size_Orbs_Grid_FNAN += 1;
        }
      }
    }

    alloc_first[3] = 0;

    /* PrintMemory */

    if (firsttime){

      int mul;
      if (SpinP_switch==3) mul = 4;
      else                 mul = 2;

      PrintMemory("truncation: Density_Grid",     sizeof(double)*N*mul,              NULL);
      PrintMemory("truncation: Vxc_Grid",         sizeof(double)*N*mul,              NULL);
      PrintMemory("truncation: RefVxc_Grid",      sizeof(double)*N,                  NULL);
      PrintMemory("truncation: Vpot_Grid",        sizeof(double)*N*mul,              NULL);
      PrintMemory("truncation: dVHart_Grid",      sizeof(double)*N,                  NULL);
      PrintMemory("truncation: Vpot_Grid_B",      sizeof(double)*My_NumGridB_AB*mul, NULL);
      PrintMemory("truncation: Density_Grid_B",   sizeof(double)*My_NumGridB_AB*mul, NULL);
      PrintMemory("truncation: ADensity_Grid_B",  sizeof(double)*My_NumGridB_AB,     NULL);
      PrintMemory("truncation: PCCDensity_Grid_B",sizeof(double)*My_NumGridB_AB*2,   NULL);
      PrintMemory("truncation: dVHart_Grid_B",    sizeof(double)*My_Max_NumGridB,    NULL);
      PrintMemory("truncation: RefVxc_Grid_B",    sizeof(double)*My_NumGridB_AB,     NULL);
      PrintMemory("truncation: Vxc_Grid_B",       sizeof(double)*My_NumGridB_AB*mul, NULL);
      PrintMemory("truncation: Density_Grid_D",   sizeof(double)*My_NumGridD*mul,    NULL);
      PrintMemory("truncation: Vxc_Grid_D",       sizeof(double)*My_NumGridD*mul,    NULL);
      PrintMemory("truncation: PCCDensity_Grid_D",sizeof(double)*My_NumGridD*2,      NULL);
      PrintMemory("truncation: Orbs_Grid",        sizeof(Type_Orbs_Grid)*size_Orbs_Grid,   NULL);
      PrintMemory("truncation: COrbs_Grid",       sizeof(Type_Orbs_Grid)*size_COrbs_Grid,  NULL);
      PrintMemory("truncation: Orbs_Grid_FNAN",   sizeof(Type_Orbs_Grid)*size_Orbs_Grid_FNAN,  NULL);

      if (ProExpn_VNA==0){
        PrintMemory("truncation: VNA_Grid",         sizeof(double)*N,                  NULL);
	PrintMemory("truncation: VNA_Grid_B",       sizeof(double)*My_NumGridB_AB,   NULL);
      }
      if (E_Field_switch==1){
	PrintMemory("truncation: VEF_Grid",         sizeof(double)*N,                NULL);
	PrintMemory("truncation: VEF_Grid_B",       sizeof(double)*My_NumGridB_AB,   NULL);
      }
    }
  
  } /* if (UCell_flag==1) */

  /****************************************************
      Output the truncation data to filename.TRN
  ****************************************************/

  if (2<=level_fileout && myid==Host_ID){
    fnjoint(filepath,filename,file_TRN);
    if ((fp_TRN = fopen(file_TRN,"w")) != NULL){

      setvbuf(fp_TRN,buf,_IOFBF,fp_bsize);  /* setvbuf */
      Output_Connectivity(fp_TRN);
      fclose(fp_TRN);
    }
    else
      printf("Failure of saving the TRN file.\n");

  }

  if (measure_time){
    dtime(&etime); 
    time16 += etime - stime;
  }

  if (measure_time){
    printf("myid=%5d time0 =%6.3f time1 =%6.3f time2 =%6.3f time3 =%6.3f time4 =%6.3f time5 =%6.3f\n",
            myid,time0,time1,time2,time3,time4,time5);
    printf("myid=%5d time6 =%6.3f time7 =%6.3f time8 =%6.3f time9 =%6.3f time10=%6.3f time11=%6.3f\n",
            myid,time6,time7,time8,time9,time10,time11);
    printf("myid=%5d time12=%6.3f time13=%6.3f time14=%6.3f time15=%6.3f time16=%6.3f\n",
            myid,time12,time13,time14,time15,time16);
  }

  /* for PrintMemory */
  firsttime = 0;
 
  /* for time */
  dtime(&TEtime);
  return (TEtime-TStime);
}







void Trn_System(int MD_iter, int CpyCell, int TCpyCell)
{
  int i,j,k,l,m,Rn,fan,san,tan,wanA,wanB,po0;
  int ct_AN,h_AN,m2,m3,size_RMI1,size_array;
  int My_TFNAN,My_TSNAN,Gh_AN,LT_switch,Nloop;
  double r,rcutA,rcutB,rcut,dx,dy,dz,rcut_max;
  double *fDis,*sDis;
  int *fnan2,*fncn;
  int *snan2,*sncn;
  int numprocs,myid,tag=999,ID;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************************
                      start calc.
  *******************************************************/

#pragma omp parallel shared(myid,ScaleSize,Max_FSNAN,level_stdout,BCR,atv,Gxyz,Dis,ncn,natn,TCpyCell,CpyCell,atomnum,FNAN,SNAN,Spe_Atom_Cut1,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,i,ct_AN,wanA,rcutA,j,wanB,rcutB,rcut,Rn,dx,dy,dz,r,l,k,fDis,fncn,fnan2,sDis,sncn,snan2,size_array,rcut_max)
  {

    /* allocation of arrays */
 
    size_array = (int)((Max_FSNAN+2)*ScaleSize);

    fDis = (double*)malloc(sizeof(double)*size_array); 
    sDis = (double*)malloc(sizeof(double)*size_array);

    fnan2 = (int*)malloc(sizeof(int)*size_array);
    fncn  = (int*)malloc(sizeof(int)*size_array); 
    snan2 = (int*)malloc(sizeof(int)*size_array); 
    sncn  = (int*)malloc(sizeof(int)*size_array); 

    /* get info. on OpenMP */ 
  
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (i=(OMPID+1); i<=Matomnum; i+=Nthrds){

      ct_AN = M2G[i];
      wanA = WhatSpecies[ct_AN];
      rcutA = Spe_Atom_Cut1[wanA];

      FNAN[ct_AN] = 0;
      SNAN[ct_AN] = 0;

      for (j=1; j<=atomnum; j++){

	wanB = WhatSpecies[j];
	rcutB = Spe_Atom_Cut1[wanB];
	rcut = rcutA + rcutB;

	if (rcut<BCR) rcut_max = BCR;
	else          rcut_max = rcut; 

	for (Rn=0; Rn<=TCpyCell; Rn++){

	  if ((ct_AN==j) && Rn==0){
	    natn[ct_AN][0] = ct_AN;
	    ncn[ct_AN][0]  = 0;
	    Dis[ct_AN][0]  = 0.0;
	  }
            
	  else{

	    dx = fabs(Gxyz[ct_AN][1] - Gxyz[j][1] - atv[Rn][1]);
	    dy = fabs(Gxyz[ct_AN][2] - Gxyz[j][2] - atv[Rn][2]);
	    dz = fabs(Gxyz[ct_AN][3] - Gxyz[j][3] - atv[Rn][3]);

	    if (dx<=rcut_max && dy<=rcut_max && dz<=rcut_max){

	      r = sqrt(dx*dx + dy*dy + dz*dz);

	      if (r<=rcut){
		FNAN[ct_AN]++;
		fnan2[FNAN[ct_AN]] = j;
		fncn[FNAN[ct_AN]] = Rn;
		fDis[FNAN[ct_AN]] = r;
	      }

	      else if (r<=BCR){
		SNAN[ct_AN]++;
		snan2[SNAN[ct_AN]] = j;
		sncn[SNAN[ct_AN]] = Rn;
		sDis[SNAN[ct_AN]] = r;
	      }

	    }

	  } /* else */

	} /* Rn */
      } /* j */

      for (k=1; k<=FNAN[ct_AN]; k++){
	natn[ct_AN][k] = fnan2[k];
	ncn[ct_AN][k] = fncn[k];
	Dis[ct_AN][k] = fDis[k];
      }
      for (k=1; k<=SNAN[ct_AN]; k++){
	l = FNAN[ct_AN] + k;
	natn[ct_AN][l] = snan2[k];
	ncn[ct_AN][l] = sncn[k];
	Dis[ct_AN][l] = sDis[k];
      }

      if (2<=level_stdout){
	printf("<truncation> myid=%4d CpyCell=%2d ct_AN=%4d FNAN SNAN %3d %3d\n",
	       myid,CpyCell,ct_AN,FNAN[ct_AN],SNAN[ct_AN]);fflush(stdout);
      }

    } /* i */  

    /* freeing of arrays */

    free(sncn);
    free(snan2);
    free(fncn);
    free(fnan2);
    free(sDis);
    free(fDis);

  } /* #pragma omp parallel */

  /**************************************************************
   if (orderN_FNAN_SNAN_flag==1)

   if one of order-N methods is used for geometry optimization,
   FNAN and SNAN should be fixed to avoid the energy jump.
  **************************************************************/

  if (orderN_FNAN_SNAN_flag==1 && MD_iter==1){ 

    Fixed_FNAN_SNAN();
  }

  /* DC-LNO */

  if (Solver==11){

    int Mc_AN,po,num;
    double rc,r0,sf;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      ct_AN = M2G[Mc_AN];

      qsort_double3B( SNAN[ct_AN],
                      &Dis[ct_AN][FNAN[ct_AN]+1],
		      &natn[ct_AN][FNAN[ct_AN]+1], 
		      &ncn[ct_AN][FNAN[ct_AN]+1]);

      r0 = 0.0;
      for (h_AN=1; h_AN<=FNAN[ct_AN]; h_AN++){
        if (r0<Dis[ct_AN][h_AN]) r0 = Dis[ct_AN][h_AN]; 
      } 

      sf = 0.1;
      po = 0;
      num = 0;

      do {

        rc = r0 + sf*(BCR-r0);
        for (h_AN=(FNAN[ct_AN]+1); h_AN<=(FNAN[ct_AN]+SNAN[ct_AN]); h_AN++){
  	  if (rc<Dis[ct_AN][h_AN]) break;
        }
        h_AN--;

        if ( (int)((double)SNAN[ct_AN]*orderN_LNO_Buffer) < (h_AN-FNAN[ct_AN]) ){
          po = 1;
	}
        else{
          sf = 1.05*sf;
	}

        num++; 

      }	while (po==0 && num<400);

      /*
      printf("ABC Mc_AN=%2d ct_AN=%2d h_AN=%2d FNAN+SNAN=%2d rc=%10.5f Dis=%10.5f BCR=%10.5f\n",
	     Mc_AN,ct_AN,h_AN,FNAN[ct_AN]+SNAN[ct_AN],rc,Dis[ct_AN][FNAN[ct_AN]],BCR);
      */

      FNAN_DCLNO[ct_AN] = h_AN;
      SNAN_DCLNO[ct_AN] = FNAN[ct_AN] + SNAN[ct_AN] - FNAN_DCLNO[ct_AN];

    }
  }

  /****************************************************
   MPI: 

         My_TFNAN -> TFNAN
         My_TSNAN -> TSNAN
         FNAN
         SNAN 
         natn
         ncn
         Dis
  ****************************************************/

  My_TFNAN = 0;
  My_TSNAN = 0;
  for (i=1; i<=Matomnum; i++){
    ct_AN = M2G[i];
    My_TFNAN = My_TFNAN + FNAN[ct_AN];
    My_TSNAN = My_TSNAN + SNAN[ct_AN];
  }

  MPI_Reduce(&My_TFNAN, &TFNAN, 1, MPI_INT, MPI_SUM, Host_ID, mpi_comm_level1);
  MPI_Bcast(&TFNAN, 1, MPI_INT, Host_ID, mpi_comm_level1);
  MPI_Reduce(&My_TSNAN, &TSNAN, 1, MPI_INT, MPI_SUM, Host_ID, mpi_comm_level1);
  MPI_Bcast(&TSNAN, 1, MPI_INT, Host_ID, mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout){
    printf("TFNAN=%8d   Average FNAN=%10.5f\n",TFNAN,(double)TFNAN/(double)atomnum);fflush(stdout);
    printf("TSNAN=%8d   Average SNAN=%10.5f\n",TSNAN,(double)TSNAN/(double)atomnum);fflush(stdout);
  }

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    ID = G2ID[ct_AN];
    MPI_Bcast(&FNAN[ct_AN], 1, MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&SNAN[ct_AN], 1, MPI_INT, ID, mpi_comm_level1);

    MPI_Bcast(&natn[ct_AN][0], FNAN[ct_AN]+SNAN[ct_AN]+1,
              MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&ncn[ct_AN][0], FNAN[ct_AN]+SNAN[ct_AN]+1,
              MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&Dis[ct_AN][0], FNAN[ct_AN]+SNAN[ct_AN]+1,
              MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (Solver==11){

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      ID = G2ID[ct_AN];
      MPI_Bcast(&FNAN_DCLNO[ct_AN], 1, MPI_INT, ID, mpi_comm_level1);
      MPI_Bcast(&SNAN_DCLNO[ct_AN], 1, MPI_INT, ID, mpi_comm_level1);
    }
  }

  if (myid==Host_ID && 0<level_stdout){
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      if (ct_AN<=20 && level_stdout<=1){
        printf("<truncation> CpyCell=%2d ct_AN=%4d FNAN SNAN %3d %3d\n",
                 CpyCell,ct_AN,FNAN[ct_AN],SNAN[ct_AN]);fflush(stdout);
      }
    }

    if (20<atomnum && level_stdout<=1){
      printf("     ..........\n");
      printf("     ......\n\n");
    }
  }
}



void Estimate_Trn_System(int CpyCell, int TCpyCell)
{
  /****************************************************
     FNAN, SNAN, Max_FNAN, Max_FSNAN are determined
               by the physical truncation
  ****************************************************/

  int i,j,ct_AN,Rn,wanA,wanB;
  double r,rcutA,rcutB,rcut;
  double dx,dy,dz,rcut_max;
  int numprocs,myid,tag=999,ID;
  int MFNAN,MFSNAN;
  int abnormal_bond,my_abnormal_bond;
  int spe1,spe2;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  abnormal_bond = 0;
  my_abnormal_bond = 0;

#pragma omp parallel shared(CpyCell,level_stdout,BCR,Spe_WhatAtom,atv,Gxyz,TCpyCell,SNAN,FNAN,Spe_Atom_Cut1,WhatSpecies,M2G,Matomnum,atomnum) private(i,ct_AN,wanA,rcutA,j,wanB,rcutB,rcut,Rn,dx,dy,dz,r,spe1,spe2,OMPID,Nthrds,Nprocs,rcut_max)

  {
    /* get info. on OpenMP */ 
  
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (i=(OMPID+1); i<=Matomnum; i+=Nthrds){
        
      ct_AN = M2G[i];
      wanA = WhatSpecies[ct_AN];
      rcutA = Spe_Atom_Cut1[wanA];

      FNAN[ct_AN] = 0;
      SNAN[ct_AN] = 0;

      for (j=1; j<=atomnum; j++){

	wanB = WhatSpecies[j];
	rcutB = Spe_Atom_Cut1[wanB];
	rcut = rcutA + rcutB;

	if (rcut<BCR) rcut_max = BCR;
	else          rcut_max = rcut; 

	for (Rn=0; Rn<=TCpyCell; Rn++){

	  if ((ct_AN==j) && Rn==0){
	    /* Nothing to be done */
	  }
 
	  else{

	    dx = fabs(Gxyz[ct_AN][1] - Gxyz[j][1] - atv[Rn][1]);
	    dy = fabs(Gxyz[ct_AN][2] - Gxyz[j][2] - atv[Rn][2]);
	    dz = fabs(Gxyz[ct_AN][3] - Gxyz[j][3] - atv[Rn][3]);

	    if (dx<=rcut_max && dy<=rcut_max && dz<=rcut_max){

	      r = sqrt(dx*dx + dy*dy + dz*dz);
	      if (r<=rcut)     FNAN[ct_AN]++;
	      else if (r<=BCR) SNAN[ct_AN]++;
	    }

	  }
	}
      } /* j */

      if (2<=level_stdout){
	printf("<truncation> CpyCell=%2d ct_AN=%2d FNAN SNAN %2d %2d\n",
	       CpyCell,i,FNAN[ct_AN],SNAN[ct_AN]);
      }
  
    } /* i */

  } /* #pragma omp parallel */

  Max_FNAN  = 0;
  Max_FSNAN = 0;

  for (i=1; i<=Matomnum; i++){
    ct_AN = M2G[i];

    if (Max_FNAN<FNAN[ct_AN])                Max_FNAN  = FNAN[ct_AN];
    if (Max_FSNAN<(FNAN[ct_AN]+SNAN[ct_AN])) Max_FSNAN = FNAN[ct_AN] + SNAN[ct_AN];
  }

  MFNAN  = Max_FNAN;
  MFSNAN = Max_FSNAN;

  /***************************************************
   MPI:
         MFNAN
         MFSNAN
  ***************************************************/ 

  /* printf("A myid=%i Max_FNAN=%i Max_FSNAN=%i\n",myid,Max_FNAN,Max_FSNAN); */

  MPI_Allreduce(&my_abnormal_bond, &abnormal_bond, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  /* check unphysical structure */

  if (abnormal_bond==1){
    if (myid==Host_ID && 0<level_stdout){
      printf("\nFound unphysical bond length: check your structure!\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /*  MFNAN */

  MPI_Reduce(&MFNAN, &Max_FNAN, 1, MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
  MPI_Bcast(&Max_FNAN, 1, MPI_INT, Host_ID, mpi_comm_level1);

  /*  MFSNAN */
  MPI_Reduce(&MFSNAN, &Max_FSNAN, 1, MPI_INT, MPI_MAX, Host_ID, mpi_comm_level1);
  MPI_Bcast(&Max_FSNAN, 1, MPI_INT, Host_ID, mpi_comm_level1);

  List_YOUSO[8] = Max_FNAN  + 7;
  if (Solver==1 || Solver==5 || Solver==6 || Solver==7 || Solver==8 || Solver==10 || Solver==11 || pop_anal_aow_flag==1 )
    List_YOUSO[2] = Max_FSNAN + 7;
  else
    List_YOUSO[2] = Max_FNAN  + 7;

}






void Check_System()
{

  char *s_vec[20];
  int i,ct_AN,h_AN,Rn,po[4],num;
  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  po[1] = 0;
  po[2] = 0;
  po[3] = 0;

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    for (h_AN=1; h_AN<=FNAN[ct_AN]; h_AN++){
      Rn = ncn[ct_AN][h_AN];
      if (Rn!=0){
        if (atv_ijk[Rn][1]!=0) po[1] = 1;
        if (atv_ijk[Rn][2]!=0) po[2] = 1;
        if (atv_ijk[Rn][3]!=0) po[3] = 1;
      }
    }
  }

  num = 0;  
  for (i=1; i<=3; i++){
    if (po[i]==1) num++;
  }

  s_vec[0] = "molecule.";
  s_vec[1] = "chain.";
  s_vec[2] = "slab.";
  s_vec[3] = "bulk.";

  specified_system = num;
  
  if (myid==Host_ID && 0<level_stdout) printf("<Check_System> The system is %s\n",s_vec[num]);

  /* in case of solver==cluster */
  if (specified_system!=0 && (Solver==2 || Solver==12) ) PeriodicGamma_flag = 1;
}




void Set_RMI()
{
  /****************************************************

    What is RMI?

    RMI[Mc_AN][i][j] is a array which specifies
    the position of arraies storing hopping and
    overlap integrals between atoms i and j.

    If ig = natn[Gc_AN][i]
       mi = F_G2M[ig] or S_G2M[ig]
       k = RMI[Mc_AN][i][j],

    then, we can find the hopping and overlap integrals
    between atoms i and j from h[mi][k] and OLP[mi][k],
    respectively.
  ****************************************************/

  static int firsttime=1;
  int i,j,k,Mc_AN,Gc_AN,size_RMI1;
  int fan,san,can,wan,ig,Rni,jg,Rnj;
  int l1,l2,l3,m1,m2,m3,Rn;
  int i_rlt,po;
  int numprocs,myid,tag=999,ID;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
   allocation of arrays:

      RMI1[Matomnum+1]
          [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
          [FNAN[Gc_AN]+SNAN[Gc_AN]+1]

      RMI2[Matomnum+1]
          [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
          [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
  ****************************************************/
  
  FNAN[0] = 0;
  SNAN[0] = 0;

  size_RMI1 = 0;
  RMI1 = (int***)malloc(sizeof(int**)*(Matomnum+1)); 
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0) Gc_AN = 0;
    else          Gc_AN = M2G[Mc_AN];
    RMI1[Mc_AN] = (int**)malloc(sizeof(int*)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1)); 
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      RMI1[Mc_AN][i] = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1)); 
      size_RMI1 += FNAN[Gc_AN]+SNAN[Gc_AN]+1;
    }      
  }
  
  RMI2 = (int***)malloc(sizeof(int**)*(Matomnum+1)); 
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0) Gc_AN = 0;
    else          Gc_AN = M2G[Mc_AN];
    RMI2[Mc_AN] = (int**)malloc(sizeof(int*)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1)); 
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      RMI2[Mc_AN][i] = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1)); 
    }      
  }

  alloc_first[6] = 0;
  
  if (firsttime){
  PrintMemory("truncation: RMI1", sizeof(int)*size_RMI1, NULL);
  PrintMemory("truncation: RMI2", sizeof(int)*size_RMI1, NULL);
  firsttime = 0; 
  }

  /****************************************************
                  setting of RMI matrix
  ****************************************************/

#pragma omp parallel shared(time_per_atom,RMI2,RMI1,ratv,CpyCell,atv_ijk,ncn,natn,WhatSpecies,SNAN,FNAN,M2G) private(OMPID,Nthrds,Nprocs,Mc_AN,Stime_atom,Gc_AN,fan,san,can,wan,i,ig,Rni,j,jg,Rnj,l1,l2,l3,m1,m2,m3,Rn,i_rlt,k,po,Etime_atom)
  {

    /* get info. on OpenMP */ 
  
    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();
  
    for (Mc_AN=(OMPID+1); Mc_AN<=Matomnum; Mc_AN+=Nthrds){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      fan = FNAN[Gc_AN];
      san = SNAN[Gc_AN];
      can = fan + san;
      wan = WhatSpecies[Gc_AN];

      for (i=0; i<=can; i++){

	ig = natn[Gc_AN][i];
	Rni = ncn[Gc_AN][i];

	for (j=0; j<=can; j++){

	  jg = natn[Gc_AN][j];
	  Rnj = ncn[Gc_AN][j];
	  l1 = atv_ijk[Rnj][1] - atv_ijk[Rni][1];
	  l2 = atv_ijk[Rnj][2] - atv_ijk[Rni][2];
	  l3 = atv_ijk[Rnj][3] - atv_ijk[Rni][3];
	  if (l1<0) m1=-l1; else m1 = l1;
	  if (l2<0) m2=-l2; else m2 = l2;
	  if (l3<0) m3=-l3; else m3 = l3;

	  if (m1<=CpyCell && m2<=CpyCell && m3<=CpyCell){

	    Rn = ratv[l1+CpyCell][l2+CpyCell][l3+CpyCell];

	    /* FNAN */

	    k = 0; po = 0;
            RMI1[Mc_AN][i][j] = -1;
	    do {
	      if (natn[ig][k]==jg && ncn[ig][k]==Rn){
		RMI1[Mc_AN][i][j] = k;
		po = 1;
	      }
	      k++;
	    } while (po==0 && k<=FNAN[ig]);

	    /* FNAN + SNAN */

	    k = 0; po = 0;
            RMI2[Mc_AN][i][j] = -1;
	    do {
	      if (natn[ig][k]==jg && ncn[ig][k]==Rn){
		RMI2[Mc_AN][i][j] = k;
		po = 1;
	      }
	      k++;
	    } while(po==0 && k<=(FNAN[ig]+SNAN[ig]));

	  }
	  else{
	    RMI1[Mc_AN][i][j] = -1;
	    RMI2[Mc_AN][i][j] = -1;
	  }
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */
  } /* #pragma omp parallel */
}




void Fixed_FNAN_SNAN()
{
  int i,j,Gc_AN;
  int hops;
  int numprocs,myid;
  double Stime_atom, Etime_atom;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (i=1; i<=Matomnum; i++){

    dtime(&Stime_atom);

    Gc_AN = M2G[i];

    if ( (orderN_FNAN[Gc_AN]+orderN_SNAN[Gc_AN])<=(FNAN[Gc_AN]+SNAN[Gc_AN]) ){

      qsort_double3B(SNAN[Gc_AN], &Dis[Gc_AN][FNAN[Gc_AN]+1],
                                  &natn[Gc_AN][FNAN[Gc_AN]+1], 
                                  &ncn[Gc_AN][FNAN[Gc_AN]+1]);

      SNAN[Gc_AN] = orderN_FNAN[Gc_AN] + orderN_SNAN[Gc_AN] - FNAN[Gc_AN];
      if (SNAN[Gc_AN]<0) SNAN[Gc_AN] = 0;
    }

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

}





void Output_Connectivity(FILE *fp)
{
  int ct_AN,h_AN,i,j,can;

  fprintf(fp,"\n");
  fprintf(fp,"***********************************************************\n");
  fprintf(fp,"***********************************************************\n");
  fprintf(fp,"                       Connectivity                        \n");
  fprintf(fp,"***********************************************************\n");
  fprintf(fp,"***********************************************************\n");

  fprintf(fp,"   FNAN SNAN\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fprintf(fp,"      %i   ",ct_AN);
    fprintf(fp,"%i %i\n",FNAN[ct_AN],SNAN[ct_AN]);
  }

  fprintf(fp,"   natn\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fprintf(fp,"      %i   ",ct_AN);
    for (h_AN=0; h_AN<=(FNAN[ct_AN]+SNAN[ct_AN]); h_AN++){
      fprintf(fp,"%i ",natn[ct_AN][h_AN]);
    }
    fprintf(fp,"\n");
  }

  fprintf(fp,"   ncn\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fprintf(fp,"      %i   ",ct_AN);
    for (h_AN=0; h_AN<=(FNAN[ct_AN]+SNAN[ct_AN]); h_AN++){
      fprintf(fp,"%i ",ncn[ct_AN][h_AN]);
    }
    fprintf(fp,"\n");
  }

  fprintf(fp,"   Dis\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fprintf(fp,"      %i   ",ct_AN);
    for (h_AN=0; h_AN<=(FNAN[ct_AN]+SNAN[ct_AN]); h_AN++){
      fprintf(fp,"%7.4f ",0.529177*Dis[ct_AN][h_AN]);
    }
    fprintf(fp,"\n");
  }

  /*
  fprintf(fp,"   RMI1\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fprintf(fp,"      %i\n",ct_AN);
    can = FNAN[ct_AN] + SNAN[ct_AN];
    for (i=0; i<=can; i++){
      for (j=0; j<=can; j++){
        if (j==0)
          fprintf(fp,"      %i ",RMI1[ct_AN][i][j]);
        else 
          fprintf(fp,"%i ",RMI1[ct_AN][i][j]);
      }
      fprintf(fp,"\n");
    }
  }

  fprintf(fp,"   RMI2\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fprintf(fp,"      %i\n",ct_AN);
    can = FNAN[ct_AN] + SNAN[ct_AN];
    for (i=0; i<=can; i++){
      for (j=0; j<=can; j++){
        if (j==0)
          fprintf(fp,"      %i ",RMI2[ct_AN][i][j]);
        else 
          fprintf(fp,"%i ",RMI2[ct_AN][i][j]);
      }
      fprintf(fp,"\n");
    }
  }
  */

}



  

void UCell_Box(int MD_iter, int estimate_switch, int CpyCell)
{
  static int firsttime=1;
  int size_GListTAtoms1;
  int po,N3[4],i,j,k;
  int size_GridListAtom,size_MGridListAtom;
  int NOC[4],nn1,nn2,nn3,l1,l2,l3,lmax;
  int ct_AN,N,n1,n2,n3,Cwan,Rn,Rn1,Ng1,Ng2,Ng3;
  int p,q,r,s,pmax,qmax,rmax,smax,popt,qopt,ropt,sopt;
  int Nct,MinN,Scale,ScaleA,ScaleB,ScaleC,Nm[4];
  int Mc_AN,Mc_AN0,Gc_AN,h_AN,h_AN0,Mh_AN,Gh_AN,Nh,Rh,Nog,GNh,GRh;
  int ll1,ll2,ll3,Nnb,My_Max_NumOLG;
  int lll1,lll2,lll3,GRh1,Nc,GNc,GRc,size_array;
  int *TAtoms0,*TCells0,*TAtoms1,*TAtoms2;
  int **Tmp_GridListAtom,**Tmp_CellListAtom;
  int nmin[4],nmax[4],Np;
  double LgTN,LgN,Lg2,Lg3,Lg5,Lg7,DouN[4],A1,A2,A3;
  double MinD,MinR,CutR2,r2,coef;
  double sa,sa_cri,tmp[4],Cxyz[4];
  double b[4],c[4],v[4],rcut;
  double xc,yc,zc,xm,ym,zm;
  double dx,dy,dz,sn1,sn2,sn3;
  double B2,C2,CellV;
  double S_Lng,L_Lng,LngA,LngB,LngC,x,y,z;
  double GVolume,buffer_scale,GridV;
  double stime,etime;
  double time0,time1,time2,time3,time4,time5,time6,time7;
  double time8,time9,time10,time11,time12,time13,time14;

  int *TempGrid,*TempCell;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  double Stime_atom, Etime_atom;
  char file_UCell[YOUSO10];
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  time0 = 0.0; time1 = 0.0; time2 = 0.0; time3 = 0.0; time4 = 0.0;
  time5 = 0.0; time6 = 0.0; time7 = 0.0; time8 = 0.0; time9 = 0.0;

  /****************************************************
                Reciprocal lattice vectors
  ****************************************************/

  if (measure_time) dtime(&stime); 

  if (estimate_switch<=1){
  
    Cross_Product(tv[2],tv[3],tmp);
    CellV = Dot_Product(tv[1],tmp); 
    Cell_Volume = fabs(CellV);
  
    Cross_Product(tv[2],tv[3],tmp);
    rtv[1][1] = 2.0*PI*tmp[1]/CellV;
    rtv[1][2] = 2.0*PI*tmp[2]/CellV;
    rtv[1][3] = 2.0*PI*tmp[3]/CellV;
  
    Cross_Product(tv[3],tv[1],tmp);
    rtv[2][1] = 2.0*PI*tmp[1]/CellV;
    rtv[2][2] = 2.0*PI*tmp[2]/CellV;
    rtv[2][3] = 2.0*PI*tmp[3]/CellV;
  
    Cross_Product(tv[1],tv[2],tmp);
    rtv[3][1] = 2.0*PI*tmp[1]/CellV;
    rtv[3][2] = 2.0*PI*tmp[2]/CellV;
    rtv[3][3] = 2.0*PI*tmp[3]/CellV;

    if (myid==Host_ID && 0<level_stdout){    

      printf("lattice vectors (bohr)\n");
      printf("A  = %15.12f, %15.12f, %15.12f\n",tv[1][1],tv[1][2],tv[1][3]);
      printf("B  = %15.12f, %15.12f, %15.12f\n",tv[2][1],tv[2][2],tv[2][3]);
      printf("C  = %15.12f, %15.12f, %15.12f\n",tv[3][1],tv[3][2],tv[3][3]);

      printf("reciprocal lattice vectors (bohr^-1)\n");
      printf("RA = %15.12f, %15.12f, %15.12f\n",rtv[1][1],rtv[1][2],rtv[1][3]);
      printf("RB = %15.12f, %15.12f, %15.12f\n",rtv[2][1],rtv[2][2],rtv[2][3]);
      printf("RC = %15.12f, %15.12f, %15.12f\n",rtv[3][1],rtv[3][2],rtv[3][3]);
    }  

    if (Ngrid_fixed_flag==0){

      /* find proper N1, N2, and N3 */ 

      Cross_Product(tv[2],tv[3],tmp);
      A1 = PI*PI*Dot_Product(tmp,tmp)/Cell_Volume/Cell_Volume; 
      
      Cross_Product(tv[3],tv[1],tmp);
      A2 = PI*PI*Dot_Product(tmp,tmp)/Cell_Volume/Cell_Volume; 

      Cross_Product(tv[1],tv[2],tmp);
      A3 = PI*PI*Dot_Product(tmp,tmp)/Cell_Volume/Cell_Volume; 

      DouN[1] = sqrt(Grid_Ecut/A1);
      DouN[2] = sqrt(Grid_Ecut/A2);
      DouN[3] = sqrt(Grid_Ecut/A3);

      Lg2 = log(2);
      Lg3 = log(3);
      Lg5 = log(5);
      Lg7 = log(7);

      for (i=1; i<=3; i++){

        LgN = log(DouN[i]);
        MinD = 10e+10;
        
        pmax = ceil(LgN/Lg2); 
        for (p=0; p<=pmax; p++){
          qmax = ceil((LgN-p*Lg2)/Lg3); 
          for (q=0; q<=qmax; q++){
            rmax = ceil((LgN-p*Lg2-q*Lg3)/Lg5); 
            for (r=0; r<=rmax; r++){
              smax = ceil((LgN-p*Lg2-q*Lg3-r*Lg5)/Lg7); 
              for (s=0; s<=smax; s++){
                
                LgTN = p*Lg2+q*Lg3+r*Lg5+s*Lg7;
                
                if (fabs(LgTN-LgN)<MinD){
                  MinD = fabs(LgTN-LgN);
                  popt = p;
                  qopt = q;
                  ropt = r;
                  sopt = s;
                }
	      }
	    }
	  }
        }
        
        k = 1;
        for (p=0; p<popt; p++) k *= 2;
        for (q=0; q<qopt; q++) k *= 3;
        for (r=0; r<ropt; r++) k *= 5;
        for (s=0; s<sopt; s++) k *= 7;

        if      (i==1) Ngrid1 = k;
        else if (i==2) Ngrid2 = k;
        else if (i==3) Ngrid3 = k;
      }

      /* adjust Ngrid for NEGF  */
      if (Solver==4) {
        TRAN_adjust_Ngrid(mpi_comm_level1, &Ngrid1, &Ngrid2, &Ngrid3);
      }

    } /* if (Ngrid_fixed_flag==0) */

    /* calculate gtv, rgtv, A2, B2, C2, and used cutoff energies */

    gtv[1][1] = tv[1][1]/(double)Ngrid1;
    gtv[1][2] = tv[1][2]/(double)Ngrid1;
    gtv[1][3] = tv[1][3]/(double)Ngrid1;
    
    gtv[2][1] = tv[2][1]/(double)Ngrid2;
    gtv[2][2] = tv[2][2]/(double)Ngrid2;
    gtv[2][3] = tv[2][3]/(double)Ngrid2;
    
    gtv[3][1] = tv[3][1]/(double)Ngrid3;
    gtv[3][2] = tv[3][2]/(double)Ngrid3;
    gtv[3][3] = tv[3][3]/(double)Ngrid3;

    Cross_Product(gtv[2],gtv[3],tmp);

    GridV = Dot_Product(gtv[1],tmp); 
    GVolume = fabs( GridV );

    Cross_Product(gtv[2],gtv[3],tmp);
    rgtv[1][1] = 2.0*PI*tmp[1]/GridV;
    rgtv[1][2] = 2.0*PI*tmp[2]/GridV;
    rgtv[1][3] = 2.0*PI*tmp[3]/GridV;

    Cross_Product(gtv[3],gtv[1],tmp);
    rgtv[2][1] = 2.0*PI*tmp[1]/GridV;
    rgtv[2][2] = 2.0*PI*tmp[2]/GridV;
    rgtv[2][3] = 2.0*PI*tmp[3]/GridV;
    
    Cross_Product(gtv[1],gtv[2],tmp);
    rgtv[3][1] = 2.0*PI*tmp[1]/GridV;
    rgtv[3][2] = 2.0*PI*tmp[2]/GridV;
    rgtv[3][3] = 2.0*PI*tmp[3]/GridV;

    A2 = rgtv[1][1]*rgtv[1][1] + rgtv[1][2]*rgtv[1][2] + rgtv[1][3]*rgtv[1][3];
    B2 = rgtv[2][1]*rgtv[2][1] + rgtv[2][2]*rgtv[2][2] + rgtv[2][3]*rgtv[2][3];
    C2 = rgtv[3][1]*rgtv[3][1] + rgtv[3][2]*rgtv[3][2] + rgtv[3][3]*rgtv[3][3];

    A2 = A2/4.0;  /* note: change the unit from Hatree to Rydberg by multiplying 1/2 */
    B2 = B2/4.0;
    C2 = C2/4.0;

    if (Ngrid_fixed_flag==1)  Grid_Ecut = (A2 + B2 + C2)/3.0;

    /* for calculation of the delta-factor */

    if (MD_switch==16) Ngrid_fixed_flag = 1;

    /* in case of the cell optimization */

    if (MD_cellopt_flag==1) Ngrid_fixed_flag = 1;

    /* print information to std output */

    if (estimate_switch==0 || 2<=level_stdout){
      if (myid==Host_ID && 0<level_stdout) {
        printf("Required cutoff energy (Ryd) for 3D-grids = %7.4f\n",Grid_Ecut);
        printf("    Used cutoff energy (Ryd) for 3D-grids = %7.4f, %7.4f, %7.4f\n",
                A2,B2,C2);
        printf("Num. of grids of a-, b-, and c-axes = %2d, %2d, %2d\n",
                Ngrid1,Ngrid2,Ngrid3);

        /***********************************************
            output informations on grids to a file
        ***********************************************/

        sprintf(file_UCell,"%s%s.UCell",filepath,filename);
        fp = fopen(file_UCell, "w");
        setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

        if (fp!=NULL && estimate_switch==0){
          fprintf(fp,"\n\n***********************************************************\n");
          fprintf(fp,"***********************************************************\n\n");
          fprintf(fp,"  Required cutoff energy (Ryd) for 3D-grids = %7.4f\n",Grid_Ecut);
          fprintf(fp,"      Used cutoff energy (Ryd) for 3D-grids = %7.4f, %7.4f, %7.4f\n",
                  A2,B2,C2);
          fprintf(fp,"  Num. of grids of a-, b-, and c-axes = %2d, %2d, %2d\n\n",
                  Ngrid1,Ngrid2,Ngrid3);

          fprintf(fp,"  Num.Grid1. %5d\n",Ngrid1);
          fprintf(fp,"  Num.Grid2. %5d\n",Ngrid2);
          fprintf(fp,"  Num.Grid3. %5d\n",Ngrid3);
          fprintf(fp,"\n\n");

          fclose(fp); 
        }
      }
    }

    Max_OneD_Grids = Ngrid1;
    if (Max_OneD_Grids<Ngrid2) Max_OneD_Grids = Ngrid2;
    if (Max_OneD_Grids<Ngrid3) Max_OneD_Grids = Ngrid3;

  } /* if (estimate_switch<=1) */

  if (measure_time){
    dtime(&etime); 
    time0 += etime - stime;
  }

  /****************************************************
       Setting the center of unit cell and grids
  ****************************************************/

  if (measure_time) dtime(&stime); 

  /* the center of the system */

  xc = 0.0;
  yc = 0.0;
  zc = 0.0;

 for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    xc += Gxyz[ct_AN][1];
    yc += Gxyz[ct_AN][2];
    zc += Gxyz[ct_AN][3];
  }

  xc = xc/(double)atomnum;
  yc = yc/(double)atomnum;
  zc = zc/(double)atomnum;

  /* added by T.Ohwaki */
  /* modified by AdvanceSoft */
  if(length_gtv[1] != 0.0 && ESM_switch!=0){

    double c_tmp;

    switch (ESM_direction){
    case 1:
      c_tmp = (int)(xc / length_gtv[1]);
      xc = ((double)c_tmp * length_gtv[1]);
      if(myid==Host_ID && 0<level_stdout){
	printf("<ESM> length_gtv[1],xc/length_gtv[1] = %12.9f,%12.9f \n",length_gtv[1],xc/length_gtv[1]);
      }
      break;
    case 2:
      c_tmp = (int)(yc / length_gtv[1]);
      yc = ((double)c_tmp * length_gtv[1]);
      if(myid==Host_ID && 0<level_stdout){
	printf("<ESM> length_gtv[1],yc/length_gtv[1] = %12.9f,%12.9f \n",length_gtv[1],yc/length_gtv[1]);
      }
      break;
    case 3:
      c_tmp = (int)(zc / length_gtv[1]);
      zc = ((double)c_tmp * length_gtv[1]);
      if(myid==Host_ID && 0<level_stdout){
	printf("<ESM> length_gtv[1],zc/length_gtv[1] = %12.9f,%12.9f \n",length_gtv[1],zc/length_gtv[1]);
      }
      break;
    }
  } /* added by T.Ohwaki */

  if (MD_iter==1 || Last_TNumGrid<Ngrid1*Ngrid2*Ngrid3){

    /************
     start calc.
    ************/

    /* gtv */

    gtv[1][1] = tv[1][1]/(double)Ngrid1;
    gtv[1][2] = tv[1][2]/(double)Ngrid1;
    gtv[1][3] = tv[1][3]/(double)Ngrid1;

    gtv[2][1] = tv[2][1]/(double)Ngrid2;
    gtv[2][2] = tv[2][2]/(double)Ngrid2;
    gtv[2][3] = tv[2][3]/(double)Ngrid2;

    gtv[3][1] = tv[3][1]/(double)Ngrid3;
    gtv[3][2] = tv[3][2]/(double)Ngrid3;
    gtv[3][3] = tv[3][3]/(double)Ngrid3;

    sn1 = 0.5*( (Ngrid1+1) % 2 ); 
    sn2 = 0.5*( (Ngrid2+1) % 2 ); 
    sn3 = 0.5*( (Ngrid3+1) % 2 ); 

    xm = ( (double)(Ngrid1/2) - sn1 )*gtv[1][1]
       + ( (double)(Ngrid2/2) - sn2 )*gtv[2][1]
       + ( (double)(Ngrid3/2) - sn3 )*gtv[3][1];

    ym = ( (double)(Ngrid1/2) - sn1 )*gtv[1][2]
       + ( (double)(Ngrid2/2) - sn2 )*gtv[2][2]
       + ( (double)(Ngrid3/2) - sn3 )*gtv[3][2];

    zm = ( (double)(Ngrid1/2) - sn1 )*gtv[1][3]
       + ( (double)(Ngrid2/2) - sn2 )*gtv[2][3]
       + ( (double)(Ngrid3/2) - sn3 )*gtv[3][3];

    if ( 1.0e+8<scf_fixed_origin[0] &&
         1.0e+8<scf_fixed_origin[1] &&
	 1.0e+8<scf_fixed_origin[2] ){

      Grid_Origin[1] = xc - xm;
      Grid_Origin[2] = yc - ym;
      Grid_Origin[3] = zc - zm;

      /* added by T.Ohwaki */
      if (myid==Host_ID && ESM_switch!=0 && 0<level_stdout){
        printf("xc=%15.12f yc=%15.12f zc=%15.12f\n",xc,yc,zc);
        printf("xm=%15.12f ym=%15.12f zm=%15.12f\n",xm,ym,zm);
      }

    }
    else{
      Grid_Origin[1] = scf_fixed_origin[0];
      Grid_Origin[2] = scf_fixed_origin[1];
      Grid_Origin[3] = scf_fixed_origin[2];
    }
    
    if (myid==Host_ID && 0<level_stdout){
      printf("Grid_Origin %15.12f %15.12f %15.12f\n",
              Grid_Origin[1],Grid_Origin[2],Grid_Origin[3]);
    }    

    TNumGrid = Ngrid1*Ngrid2*Ngrid3;
    Last_TNumGrid = (int)(ScaleSize*Ngrid1*Ngrid2*Ngrid3);

    LastBoxCenterX = xc;
    LastBoxCenterY = yc;
    LastBoxCenterZ = zc;

  }

  /****************************************************
            xyz-coordinate to cell-coordinate
  ****************************************************/

  if (estimate_switch<=1){

#pragma omp parallel shared(level_stdout,rtv,Cell_Gxyz,Grid_Origin,Gxyz,M2G,Matomnum) private(Mc_AN,Gc_AN,OMPID,Nthrds,Nprocs)
    {

      double Cxyz[4];

      /* get info. on OpenMP */ 
  
      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Mc_AN=(OMPID+1); Mc_AN<=Matomnum; Mc_AN+=Nthrds){

	Gc_AN = M2G[Mc_AN];
	Cxyz[1] = Gxyz[Gc_AN][1] - Grid_Origin[1];
	Cxyz[2] = Gxyz[Gc_AN][2] - Grid_Origin[2];
	Cxyz[3] = Gxyz[Gc_AN][3] - Grid_Origin[3];
	Cell_Gxyz[Gc_AN][1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	Cell_Gxyz[Gc_AN][2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	Cell_Gxyz[Gc_AN][3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

	if (2<=level_stdout){
	  printf("Cell_Gxyz %3d  %15.12f %15.12f %15.12f\n",
		 Gc_AN,
		 Cell_Gxyz[Gc_AN][1],
		 Cell_Gxyz[Gc_AN][2],
		 Cell_Gxyz[Gc_AN][3]); 
	}
      }

    } /* #pragma omp parallel */

    /****************
     MPI: 
         Cell_Gxyz
    *****************/

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      ID = G2ID[ct_AN];
      MPI_Bcast(&Cell_Gxyz[ct_AN][0], 4, MPI_DOUBLE, ID, mpi_comm_level1);
    }
  }

  /****************************************************
            Find grids overlaping to each atom
  ****************************************************/

  /* gtv */

  gtv[1][1] = tv[1][1]/(double)Ngrid1;
  gtv[1][2] = tv[1][2]/(double)Ngrid1;
  gtv[1][3] = tv[1][3]/(double)Ngrid1;

  gtv[2][1] = tv[2][1]/(double)Ngrid2;
  gtv[2][2] = tv[2][2]/(double)Ngrid2;
  gtv[2][3] = tv[2][3]/(double)Ngrid2;

  gtv[3][1] = tv[3][1]/(double)Ngrid3;
  gtv[3][2] = tv[3][2]/(double)Ngrid3;
  gtv[3][3] = tv[3][3]/(double)Ngrid3;

  Cross_Product(gtv[2],gtv[3],tmp);
  GridV = Dot_Product(gtv[1],tmp);
  GridVol = fabs( GridV );

  length_gtv[1] = sqrt( Dot_Product(gtv[1], gtv[1]) );
  length_gtv[2] = sqrt( Dot_Product(gtv[2], gtv[2]) );
  length_gtv[3] = sqrt( Dot_Product(gtv[3], gtv[3]) );

  /* rgtv */

  Cross_Product(gtv[2],gtv[3],tmp);
  rgtv[1][1] = 2.0*PI*tmp[1]/GridV;
  rgtv[1][2] = 2.0*PI*tmp[2]/GridV;
  rgtv[1][3] = 2.0*PI*tmp[3]/GridV;

  Cross_Product(gtv[3],gtv[1],tmp);
  rgtv[2][1] = 2.0*PI*tmp[1]/GridV;
  rgtv[2][2] = 2.0*PI*tmp[2]/GridV;
  rgtv[2][3] = 2.0*PI*tmp[3]/GridV;

  Cross_Product(gtv[1],gtv[2],tmp);
  rgtv[3][1] = 2.0*PI*tmp[1]/GridV;
  rgtv[3][2] = 2.0*PI*tmp[2]/GridV;
  rgtv[3][3] = 2.0*PI*tmp[3]/GridV;

  if (myid==Host_ID && 0<level_stdout){    
    printf("Cell_Volume = %19.12f (Bohr^3)\n",Cell_Volume); 
    printf("GridVol     = %19.12f (Bohr^3)\n",GridVol); 
  }

  if ( (estimate_switch==0 || 2<=level_stdout) && myid==Host_ID ){

    if (0<level_stdout){ 

      printf("Cell vectors (bohr) of the grid cell (gtv)\n");
      printf("  gtv_a = %15.12f, %15.12f, %15.12f\n",gtv[1][1],gtv[1][2],gtv[1][3]);
      printf("  gtv_b = %15.12f, %15.12f, %15.12f\n",gtv[2][1],gtv[2][2],gtv[2][3]);
      printf("  gtv_c = %15.12f, %15.12f, %15.12f\n",gtv[3][1],gtv[3][2],gtv[3][3]);
      printf("  |gtv_a| = %15.12f\n",length_gtv[1]);
      printf("  |gtv_b| = %15.12f\n",length_gtv[2]);
      printf("  |gtv_c| = %15.12f\n",length_gtv[3]);
    }

    /***********************************************
        output informations on grids to a file
    ***********************************************/

    sprintf(file_UCell,"%s%s.UCell",filepath,filename);
    fp = fopen(file_UCell, "a");
    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    if (fp!=NULL && estimate_switch==0){

      fprintf(fp,"  Cell_Volume = %19.12f (Bohr^3)\n",Cell_Volume); 
      fprintf(fp,"  GridVol     = %19.12f (Bohr^3)\n",GridVol); 

      fprintf(fp,"  Cell vectors (bohr) of the grid cell (gtv)\n");
      fprintf(fp,"    gtv_a = %15.12f, %15.12f, %15.12f\n",gtv[1][1],gtv[1][2],gtv[1][3]);
      fprintf(fp,"    gtv_b = %15.12f, %15.12f, %15.12f\n",gtv[2][1],gtv[2][2],gtv[2][3]);
      fprintf(fp,"    gtv_c = %15.12f, %15.12f, %15.12f\n",gtv[3][1],gtv[3][2],gtv[3][3]);
      fprintf(fp,"    |gtv_a| = %15.12f\n",length_gtv[1]);
      fprintf(fp,"    |gtv_b| = %15.12f\n",length_gtv[2]);
      fprintf(fp,"    |gtv_c| = %15.12f\n",length_gtv[3]);

      fprintf(fp,"\n***********************************************************\n");
      fprintf(fp,"***********************************************************\n\n");

      fclose(fp); 
    }
  }

  if (measure_time){
    dtime(&etime); 
    time1 += etime - stime;
  }

  /**********************************
    allocation of arrays: 

    Tmp_GridListAtom
    Tmp_CellListAtom
    MGridListAtom
  **********************************/

  Tmp_GridListAtom = (int**)malloc(sizeof(int*)*(Matomnum+MatomnumF+1));
  Tmp_CellListAtom = (int**)malloc(sizeof(int*)*(Matomnum+MatomnumF+1));
  MGridListAtom = (int**)malloc(sizeof(int*)*(Matomnum+1));
  Tmp_GridListAtom[0] = (int*)malloc(sizeof(int)*1);
  Tmp_CellListAtom[0] = (int*)malloc(sizeof(int)*1);
  MGridListAtom[0] = (int*)malloc(sizeof(int)*1);
  alloc_first[2] = 0;

  /****************************************************
   1) find a neighbouring point of the atom Mc_AN
   2) the ranges which deterinie a box on the atom Mc_AN 
   3) determine whether overlap exists or not
  ****************************************************/

  if (measure_time) dtime(&stime); 

  /* for allocation of arrays */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    rcut = Spe_Atom_Cut1[Cwan] + 0.5;

    for (k=1; k<=3; k++){

      if      (k==1){ i = 2; j = 3; }
      else if (k==2){ i = 3; j = 1; }
      else if (k==3){ i = 1; j = 2; }

      b[1] = tv[i][1];
      b[2] = tv[i][2];
      b[3] = tv[i][3];

      c[1] = tv[j][1];
      c[2] = tv[j][2];
      c[3] = tv[j][3];

      Cross_Product(b,c,v);
      coef = 1.0/sqrt(fabs( Dot_Product(v,v) ));

      v[1] = coef*v[1];
      v[2] = coef*v[2];
      v[3] = coef*v[3];

      Cxyz[1] = Gxyz[Gc_AN][1] + rcut*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[Gc_AN][2] + rcut*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[Gc_AN][3] + rcut*v[3] - Grid_Origin[3];

      /* find the maximum range of grids */
      nmax[k] = Dot_Product(Cxyz,rgtv[k])*0.5/PI;

      Cxyz[1] = Gxyz[Gc_AN][1] - rcut*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[Gc_AN][2] - rcut*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[Gc_AN][3] - rcut*v[3] - Grid_Origin[3];

      /* find the mimum range of grids */
      nmin[k] = Dot_Product(Cxyz,rgtv[k])*0.5/PI;

      if (nmax[k]<nmin[k]){
        i = nmin[k];
        j = nmax[k];
        nmin[k] = j;
        nmax[k] = i;
      } 
  
    } /* k */  

    /* allocation of arrays */ 

    Np = (nmax[1]-nmin[1]+1)*(nmax[2]-nmin[2]+1)*(nmax[3]-nmin[3]+1)*3/2;
    
    Tmp_GridListAtom[Mc_AN] = (int*)malloc(sizeof(int)*Np);
    Tmp_CellListAtom[Mc_AN] = (int*)malloc(sizeof(int)*Np);
    MGridListAtom[Mc_AN] = (int*)malloc(sizeof(int)*Np);

  } /* Mc_AN */

  /* store Tmp_GridListAtom and Tmp_CellListAtom */

  size_array = (int)(Max_GridN_Atom*ScaleSize);
  TempGrid = (int*)malloc(sizeof(int)*size_array); 
  TempCell = (int*)malloc(sizeof(int)*size_array); 

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    rcut = Spe_Atom_Cut1[Cwan] + 0.5;

    for (k=1; k<=3; k++){

      if      (k==1){ i = 2; j = 3; }
      else if (k==2){ i = 3; j = 1; }
      else if (k==3){ i = 1; j = 2; }

      b[1] = tv[i][1];
      b[2] = tv[i][2];
      b[3] = tv[i][3];

      c[1] = tv[j][1];
      c[2] = tv[j][2];
      c[3] = tv[j][3];

      Cross_Product(b,c,v);
      coef = 1.0/sqrt(fabs( Dot_Product(v,v) ));

      v[1] = coef*v[1];
      v[2] = coef*v[2];
      v[3] = coef*v[3];

      Cxyz[1] = Gxyz[Gc_AN][1] + rcut*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[Gc_AN][2] + rcut*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[Gc_AN][3] + rcut*v[3] - Grid_Origin[3];

      /* find the maximum range of grids */
      nmax[k] = Dot_Product(Cxyz,rgtv[k])*0.5/PI;

      Cxyz[1] = Gxyz[Gc_AN][1] - rcut*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[Gc_AN][2] - rcut*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[Gc_AN][3] - rcut*v[3] - Grid_Origin[3];

      /* find the mimum range of grids */
      nmin[k] = Dot_Product(Cxyz,rgtv[k])*0.5/PI;
  
      if (nmax[k]<nmin[k]){
        i = nmin[k];
        j = nmax[k];
        nmin[k] = j;
        nmax[k] = i;
      } 

    } /* k */  

    CutR2 = Spe_Atom_Cut1[Cwan]*Spe_Atom_Cut1[Cwan];

    Nct = 0;
    for (n1=nmin[1]; n1<=nmax[1]; n1++){
      for (n2=nmin[2]; n2<=nmax[2]; n2++){
	for (n3=nmin[3]; n3<=nmax[3]; n3++){

	  Find_CGrids(1,n1,n2,n3,Cxyz,NOC);
	  Rn = NOC[0];
	  l1 = NOC[1];
	  l2 = NOC[2];
	  l3 = NOC[3];
	  N = l1*Ngrid2*Ngrid3 + l2*Ngrid3 + l3;

	  dx = Cxyz[1] - Gxyz[Gc_AN][1];
	  dy = Cxyz[2] - Gxyz[Gc_AN][2];
	  dz = Cxyz[3] - Gxyz[Gc_AN][3];

	  r2 = dx*dx + dy*dy + dz*dz;
	  if (r2<=CutR2){
	    if (estimate_switch!=1){
	      TempGrid[Nct+1] = N;
	      TempCell[Nct+1] = Rn;
	    }             
	    Nct++;
	  }
	}
      }
    }

    Np = (nmax[1]-nmin[1]+1)*(nmax[2]-nmin[2]+1)*(nmax[3]-nmin[3]+1)*3/2;
    if (Np<Nct){
      printf("Invalid access in truncation.c\n"); 
      MPI_Finalize();
      exit(0); 
    }

    GridN_Atom[Gc_AN] = Nct;

    if (estimate_switch!=1){

      /* sorting */
      qsort_int((long)Nct,TempGrid,TempCell);

      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	Tmp_GridListAtom[Mc_AN][Nc] = TempGrid[Nc+1];
	Tmp_CellListAtom[Mc_AN][Nc] = TempCell[Nc+1];
      }
    }
  }

  free(TempCell);
  free(TempGrid);

  /* calculate size_GridListAtom */

  size_GridListAtom = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    size_GridListAtom += GridN_Atom[Gc_AN];
    if (Max_GridN_Atom<GridN_Atom[Gc_AN]) Max_GridN_Atom = GridN_Atom[Gc_AN];
  }

  if (measure_time){
    dtime(&etime); 
    time2 += etime - stime;
  }

  /****************************************************
   MPI: 

       GridN_Atom
  ****************************************************/

  if (measure_time) dtime(&stime); 

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    ID = G2ID[ct_AN];
    MPI_Bcast(&GridN_Atom[ct_AN], 1, MPI_INT, ID, mpi_comm_level1);
  }

  if (myid==Host_ID && estimate_switch==0 && 0<level_stdout){
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      if (ct_AN<=20 && level_stdout<=1){
         printf("Num. of grids overlapping with atom %4d = %4d\n",
                 ct_AN, GridN_Atom[ct_AN]);
      }
    }

    if (20<atomnum && level_stdout<=1){
      printf("     ..........\n");
      printf("     ......\n\n");
    }
  }

  /****************************************************
    allocation of arrays:

    Tmp_GridListAtom
    Tmp_CellListAtom
  ****************************************************/
  
  size_MGridListAtom = size_GridListAtom;

  for (Mc_AN=Matomnum+1; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    Tmp_GridListAtom[Mc_AN] = (int*)malloc(sizeof(int)*GridN_Atom[Gc_AN]);
    Tmp_CellListAtom[Mc_AN] = (int*)malloc(sizeof(int)*GridN_Atom[Gc_AN]);
    size_MGridListAtom += GridN_Atom[Gc_AN];
  }

  /* PrintMemory */
  if (firsttime){
  PrintMemory("truncation: Tmp_GridListAtom", sizeof(int)*size_MGridListAtom, NULL);
  PrintMemory("truncation: Tmp_CellListAtom", sizeof(int)*size_MGridListAtom, NULL);
  }

  if (measure_time){
    dtime(&etime); 
    time3 += etime - stime;
  }

  /****************************************************
   MPI: 

       Tmp_GridListAtom
       Tmp_CellListAtom
  ****************************************************/

  if (measure_time) dtime(&stime); 

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);

  /* Tmp_GridListAtom */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;
      /* Sending of data to IDS */
      if (F_Snd_Num[IDS]!=0){
        for (i=0; i<F_Snd_Num[IDS]; i++){
          Mc_AN = Snd_MAN[IDS][i];
          Gc_AN = Snd_GAN[IDS][i];
          MPI_Isend(&Tmp_GridListAtom[Mc_AN][0], GridN_Atom[Gc_AN], MPI_INT,
                    IDS, tag, mpi_comm_level1, &request);
	}
      }

      /* Receiving of data from IDR */
      if (F_Rcv_Num[IDR]!=0){
        tag = 999;
        for (i=0; i<F_Rcv_Num[IDR]; i++){
          Mc_AN = F_TopMAN[IDR] + i;
          Gc_AN = F_M2G[Mc_AN];
          MPI_Recv(&Tmp_GridListAtom[Mc_AN][0], GridN_Atom[Gc_AN], MPI_INT,
                   IDR, tag, mpi_comm_level1, &stat);
	}          
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
    }     
  }

  /* Tmp_CellListAtom */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;
      /* Sending of data to IDS */
      if (F_Snd_Num[IDS]!=0){
        for (i=0; i<F_Snd_Num[IDS]; i++){
          Mc_AN = Snd_MAN[IDS][i];
          Gc_AN = Snd_GAN[IDS][i];
          MPI_Isend(&Tmp_CellListAtom[Mc_AN][0], GridN_Atom[Gc_AN], MPI_INT,
                    IDS, tag, mpi_comm_level1, &request);
	}
      }

      /* Receiving of data from IDR */
      if (F_Rcv_Num[IDR]!=0){
        tag = 999;
        for (i=0; i<F_Rcv_Num[IDR]; i++){
          Mc_AN = F_TopMAN[IDR] + i;
          Gc_AN = F_M2G[Mc_AN];
          MPI_Recv(&Tmp_CellListAtom[Mc_AN][0], GridN_Atom[Gc_AN], MPI_INT,
                   IDR, tag, mpi_comm_level1, &stat);
	}          
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
    }     
  }

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);

  if (measure_time){
    dtime(&etime); 
    time4 += etime - stime;
  }

  if (measure_time) dtime(&stime); 

  if (estimate_switch!=1){

    /****************************************************
            Find overlap grids between two orbitals
    ****************************************************/
    
    if (estimate_switch==0){
      size_GListTAtoms1 = 0;

      GListTAtoms1 = (int***)malloc(sizeof(int**)*(Matomnum+1));
      GListTAtoms2 = (int***)malloc(sizeof(int**)*(Matomnum+1));
      alloc_first[0] = 0;
    }

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      if (Mc_AN==0) Gc_AN = 0;
      else          Gc_AN = M2G[Mc_AN];

      if (Mc_AN==0){

	FNAN[0] = 0; 

        if (estimate_switch==0){

          GListTAtoms1[0] = (int**)malloc(sizeof(int*)*1);
          GListTAtoms2[0] = (int**)malloc(sizeof(int*)*1);

          GListTAtoms1[0][0] = (int*)malloc(sizeof(int)*1);
          GListTAtoms2[0][0] = (int*)malloc(sizeof(int)*1);
        }
      }

      else{

        if (estimate_switch==0){

          GListTAtoms1[Mc_AN] = (int**)malloc(sizeof(int*)*(FNAN[Gc_AN]+1));
          GListTAtoms2[Mc_AN] = (int**)malloc(sizeof(int*)*(FNAN[Gc_AN]+1));
        }

        h_AN0 = 0;

#pragma omp parallel shared(List_YOUSO,GListTAtoms1,GListTAtoms2,ScaleSize,Max_NumOLG,size_GListTAtoms1,level_stdout,NumOLG,estimate_switch,CpyCell,Mc_AN,Tmp_CellListAtom,Tmp_GridListAtom,GridN_Atom,atv_ijk,ncn,F_G2M,natn,h_AN0,FNAN,Gc_AN) private(OMPID,Nthrds,Nprocs,h_AN,Gh_AN,Mh_AN,Rh,l1,l2,l3,Nog,Nc,Nh,GNh,GRh,ll1,ll2,ll3,lll1,lll2,lll3,GRh1,po,GNc,GRc,TAtoms0,TCells0,TAtoms1,TAtoms2,size_array,i)
	{        

	  /*******************************************************
            allocation of temporal arrays
	  *******************************************************/

	  size_array = (int)((Max_NumOLG+2)*ScaleSize);
	  TAtoms0 = (int*)malloc(sizeof(int)*size_array);
	  TCells0 = (int*)malloc(sizeof(int)*size_array);
	  TAtoms1 = (int*)malloc(sizeof(int)*size_array);
	  TAtoms2 = (int*)malloc(sizeof(int)*size_array);

	  /* get info. on OpenMP */ 
  
	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

          do {  

#pragma omp barrier
            h_AN = h_AN0 + OMPID;

            if (h_AN<=FNAN[Gc_AN]){

	      Gh_AN = natn[Gc_AN][h_AN];
	      Mh_AN = F_G2M[Gh_AN];
	      Rh = ncn[Gc_AN][h_AN];

	      l1 = atv_ijk[Rh][1];
	      l2 = atv_ijk[Rh][2];
	      l3 = atv_ijk[Rh][3];
  
	      Nog = -1;
	      Nc = 0;

	      for (Nh=0; Nh<GridN_Atom[Gh_AN]; Nh++){

		GNh = Tmp_GridListAtom[Mh_AN][Nh];
		GRh = Tmp_CellListAtom[Mh_AN][Nh];

		ll1 = atv_ijk[GRh][1];
		ll2 = atv_ijk[GRh][2];
		ll3 = atv_ijk[GRh][3];

		lll1 = l1 + ll1;
		lll2 = l2 + ll2;
		lll3 = l3 + ll3;

		if (Tmp_GridListAtom[Mc_AN][0]<=GNh) {

		  /* find the initial Nc */

		  if (GNh==0) {
		    Nc = 0;
		  }
		  else {
		    while ( GNh<=Tmp_GridListAtom[Mc_AN][Nc] && Nc!=0 ){
		      Nc = Nc - 10;
		      if (Nc<0) Nc = 0;
		    }
		  }

		  /*  find whether there is the overlapping or not. */

		  if (abs(lll1)<=CpyCell && abs(lll2)<=CpyCell && abs(lll3)<=CpyCell){

		    GRh1 = R_atv(CpyCell,lll1,lll2,lll3);

		    po = 0;

		    while (po==0 && Nc<GridN_Atom[Gc_AN]) {

		      GNc = Tmp_GridListAtom[Mc_AN][Nc];
		      GRc = Tmp_CellListAtom[Mc_AN][Nc];

		      if (GNc==GNh && GRc==GRh1){

			Nog++;

			if (estimate_switch==0){
			  TAtoms0[Nog] = GNc;
			  TCells0[Nog] = GRc;
			  TAtoms1[Nog] = Nc;
			  TAtoms2[Nog] = Nh;
			}

			po = 1;
		      }

		      else if (GNh<GNc){
			po = 1; 
		      } 

		      Nc++;

		    } /* while (...) */

		    /* for Nc==GridN_Atom[Gc_AN] */

		    Nc--;
		    if (Nc<0) Nc = 0;

		  } /* if (abs.... ) */
		} /* if (Tmp_GridListAtom[Mc_AN][0]<=GNh) */
	      } /* Nh */

	      NumOLG[Mc_AN][h_AN] = Nog + 1;

	      if (List_YOUSO[12]<(Nog+1) && estimate_switch==0){
		printf("YOUSO12<(Nog+1)\n");
                MPI_Finalize();
		exit(1);
	      }

	      if (2<=level_stdout){
		printf("Num. of grids overlapping between atoms %2d (G) and %2d (L) = %2d\n",
		       Gc_AN,h_AN,Nog+1);
	      }
	    
	    } /* if (h_AN<=FNAN[Gc_AN]) */

#pragma omp barrier
#pragma omp flush(NumOLG)

	    if (estimate_switch==0){

              /* allocation of arrays */

              if (OMPID==0){

		for (i=h_AN0; i<(h_AN0+Nthrds); i++){

                  if (i<=FNAN[Gc_AN]){

		    size_GListTAtoms1 += NumOLG[Mc_AN][i];

		    GListTAtoms1[Mc_AN][i] = (int*)malloc(sizeof(int)*NumOLG[Mc_AN][i]);
		    GListTAtoms2[Mc_AN][i] = (int*)malloc(sizeof(int)*NumOLG[Mc_AN][i]);

		  }
		}
	      }

#pragma omp barrier
#pragma omp flush(GListTAtoms1,GListTAtoms2)

              if (h_AN<=FNAN[Gc_AN]){

		for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

		  GListTAtoms1[Mc_AN][h_AN][Nog] = TAtoms1[Nog];
		  GListTAtoms2[Mc_AN][h_AN][Nog] = TAtoms2[Nog];
		}
	      }
	    }

            /* increament of h_AN0 */

            if (OMPID==0) h_AN0 += Nthrds;
#pragma omp barrier
#pragma omp flush(h_AN0)

	  } while (h_AN0<=FNAN[Gc_AN]);

          /* freeing of arrays */

	  free(TAtoms2);
	  free(TAtoms1);
	  free(TCells0);
	  free(TAtoms0);

	} /* #pragma omp parallel */

	dtime(&Etime_atom);
	time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
      }

    } /* Mc_AN */ 

    if (estimate_switch==0){

      if (firsttime){
      PrintMemory("truncation: GListTAtoms1", sizeof(int)*size_GListTAtoms1, NULL);
      PrintMemory("truncation: GListTAtoms2", sizeof(int)*size_GListTAtoms1, NULL);
      }
    }

    /* find Max_NumOLG */

    My_Max_NumOLG = 0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        if (My_Max_NumOLG<NumOLG[Mc_AN][h_AN]) My_Max_NumOLG = NumOLG[Mc_AN][h_AN];
      }
    }

    MPI_Allreduce(&My_Max_NumOLG, &Max_NumOLG, 1, MPI_INT, MPI_MAX, mpi_comm_level1);

  } /* if (estimate_switch!=1) */

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);

  if (measure_time){
    dtime(&etime); 
    time5 += etime - stime;
  }

  /****************************************************
       Tmp_GridListAtom -> GridListAtom
       Tmp_CellListAtom -> CellListAtom                     
  ****************************************************/

  if (measure_time) dtime(&stime); 

  size_GridListAtom = 0;

  GridListAtom = (int**)malloc(sizeof(int*)*(Matomnum+1));
  GridListAtom[0] = (int*)malloc(sizeof(int)*1);
  size_GridListAtom++;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    GridListAtom[Mc_AN] = (int*)malloc(sizeof(int)*GridN_Atom[Gc_AN]);
    size_GridListAtom += GridN_Atom[Gc_AN];
    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
      GridListAtom[Mc_AN][Nc] = Tmp_GridListAtom[Mc_AN][Nc];
    }
  }

  CellListAtom = (int**)malloc(sizeof(int*)*(Matomnum+1));
  CellListAtom[0] = (int*)malloc(sizeof(int)*1);
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    CellListAtom[Mc_AN] = (int*)malloc(sizeof(int)*GridN_Atom[Gc_AN]);
    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
      CellListAtom[Mc_AN][Nc] = Tmp_CellListAtom[Mc_AN][Nc];
    }
  }

  /* PrintMemory */
  if (firsttime){
  PrintMemory("truncation: GridListAtom", sizeof(int)*size_GridListAtom, NULL);
  PrintMemory("truncation: CellListAtom", sizeof(int)*size_GridListAtom, NULL);
  PrintMemory("truncation: MGridListAtom", sizeof(int)*size_GridListAtom, NULL);
  }

  /****************************************************
   construct the data structure for MPI communications
   for grid data
  ****************************************************/

  if (estimate_switch==0){

    Construct_MPI_Data_Structure_Grid();

    Ng1 = Max_Grid_Index[1] - Min_Grid_Index[1] + 1;
    Ng2 = Max_Grid_Index[2] - Min_Grid_Index[2] + 1;
    Ng3 = Max_Grid_Index[3] - Min_Grid_Index[3] + 1;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = F_M2G[Mc_AN];

#pragma omp parallel shared(Min_Grid_Index,atv_ijk,Ng1,Ng2,Ng3,Ngrid1,Ngrid2,Ngrid3,MGridListAtom,Mc_AN,Tmp_GridListAtom,Tmp_CellListAtom,GridN_Atom,Gc_AN) private(Nc,GNc,GRc,N3,n1,n2,n3,N,OMPID,Nthrds,Nprocs)
      {

	/* get info. on OpenMP */ 
  
	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (Nc=OMPID; Nc<GridN_Atom[Gc_AN]; Nc+=Nthrds){

	  GNc = Tmp_GridListAtom[Mc_AN][Nc];
          GRc = Tmp_CellListAtom[Mc_AN][Nc];

	  GN2N(GNc,N3);

	  n1 = N3[1] + Ngrid1*atv_ijk[GRc][1] - Min_Grid_Index[1];  
	  n2 = N3[2] + Ngrid2*atv_ijk[GRc][2] - Min_Grid_Index[2];  
	  n3 = N3[3] + Ngrid3*atv_ijk[GRc][3] - Min_Grid_Index[3];  

	  MGridListAtom[Mc_AN][Nc] = n1*Ng2*Ng3 + n2*Ng3 + n3;
	}

      } /* #pragma omp parallel */

      dtime(&Etime_atom); 
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

  }

  if (measure_time){
    dtime(&etime); 
    time6 += etime - stime;
  }

  /* for PrintMemory */
  firsttime = 0;

  /****************************************************
                          Free
  ****************************************************/

  if (measure_time) dtime(&stime); 

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);

  for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
    free(Tmp_CellListAtom[Mc_AN]);
    free(Tmp_GridListAtom[Mc_AN]);
  }
  free(Tmp_CellListAtom);
  free(Tmp_GridListAtom);

  if (alloc_first[2]==0 && estimate_switch!=0){

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(MGridListAtom[Mc_AN]);
    }
    free(MGridListAtom);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(GridListAtom[Mc_AN]);
    }
    free(GridListAtom);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(CellListAtom[Mc_AN]);
    }
    free(CellListAtom);
  }

  if (measure_time){
    dtime(&etime); 
    time7 += etime - stime;
  }

  if (measure_time){
    printf("UCell_Box myid=%5d time0=%6.3f time1=%6.3f time2=%6.3f time3=%6.3f time4=%6.3f time5=%6.3f time6=%6.3f time7=%6.3f\n",
            myid,time0,time1,time2,time3,time4,time5,time6,time7);
  }
}








int Set_Periodic(int CpyN, int Allocate_switch)
{
  static int firsttime=1;
  int i,j,n,n2,TN;
  long n3;

  TN = (2*CpyN+1)*(2*CpyN+1)*(2*CpyN+1) - 1;

  if (Allocate_switch==0){

    /****************************************************
                         Allocation
    ****************************************************/

    atv = (double**)malloc(sizeof(double*)*(TN+1));
    for (i=0; i<(TN+1); i++){
      atv[i] = (double*)malloc(sizeof(double)*4);
    }

    n = 2*CpyN + 4;
    ratv = (int***)malloc(sizeof(int**)*n);
    for (i=0; i<n; i++){
      ratv[i] = (int**)malloc(sizeof(int*)*n);
      for (j=0; j<n; j++){
	ratv[i][j] = (int*)malloc(sizeof(int)*n);
      }
    }
    if (firsttime) 
    PrintMemory("Set_Periodic: ratv",sizeof(int)*n*n*n,NULL);

    atv_ijk = (int**)malloc(sizeof(int*)*(TN+1));
    for (i=0; i<(TN+1); i++){
      atv_ijk[i] = (int*)malloc(sizeof(int)*4);
    }

    alloc_first[7] = 0;

    /* for PrintMemory */
    firsttime=0;

    /****************************************************
           setting of parameters of periodic cells
    ****************************************************/

    Generation_ATV(CpyN);
  }

  /* return */

  return TN;
}











void Free_truncation(int CpyN, int TN, int Free_switch)
{
  int i,j,n,Nc;

  if (Free_switch==0){

    if (alloc_first[7]==0){

      for (i=0; i<(TN+1); i++){
        free(atv[i]);
      }
      free(atv);

      n = 2*CpyN + 4;

      for (i=0; i<n; i++){
        for (j=0; j<n; j++){
	  free(ratv[i][j]);
        }
        free(ratv[i]);
      }
      free(ratv);

      for (i=0; i<(TN+1); i++){
        free(atv_ijk[i]);
      }
      free(atv_ijk);
    }
  }
} 








void free_arrays_truncation0()
{
  int i,j,k,m,ct_AN,h_AN,Gh_AN,Hwan;
  int tno0,tno1,tno2,tno,Cwan,so,s1,s2;
  int num,wan,n2,wanA,wanB,Gi,Nc;
  int Gc_AN,Mc_AN,spin,fan;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    freeing of arrays:

      H0
      CntH0
      HNL
      iCntHNL
      OLP
      CntOLP
      H
      CntH
      DS_NL
      CntDS_NL
      DM
      ResidualDM
      EDM
      PDM
      CntCoes
      HVNA
      DS_VNA
      HVNA2
      CntHVNA2
      DM_onsite
      v_eff
      NC_OcpN
      NC_v_eff
  ****************************************************/

  if (alloc_first[4]==0){

    /* H0 */

    for (k=0; k<4; k++){
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
          FNAN[0] = 0;
	}
	else{
          Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  for (i=0; i<tno0; i++){
	    free(H0[k][Mc_AN][h_AN][i]);
	  }
	  free(H0[k][Mc_AN][h_AN]);
	}
	free(H0[k][Mc_AN]);
      }
      free(H0[k]);
    }
    free(H0);

    /* CntH0 */

    if (Cnt_switch==1){
      for (k=0; k<4; k++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(CntH0[k][Mc_AN][h_AN][i]);
	    }
	    free(CntH0[k][Mc_AN][h_AN]);
	  }
	  free(CntH0[k][Mc_AN]);
	}
	free(CntH0[k]);
      }
      free(CntH0);
    }

    /* HNL */

    for (k=0; k<List_YOUSO[5]; k++){
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
          FNAN[0] = 0;
	}
	else{
          Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  for (i=0; i<tno0; i++){
	    free(HNL[k][Mc_AN][h_AN][i]);
	  }
	  free(HNL[k][Mc_AN][h_AN]);
	}
	free(HNL[k][Mc_AN]);
      }
      free(HNL[k]);
    }
    free(HNL);

    /* iHNL */

    if ( SpinP_switch==3 ){

      for (k=0; k<List_YOUSO[5]; k++){
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(iHNL[k][Mc_AN][h_AN][i]);
	    }
	    free(iHNL[k][Mc_AN][h_AN]);
	  }
	  free(iHNL[k][Mc_AN]);
	}
	free(iHNL[k]);
      }
      free(iHNL);
    }

    /* iCntHNL */

    if (SO_switch==1 && Cnt_switch==1){

      for (k=0; k<List_YOUSO[5]; k++){
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(iCntHNL[k][Mc_AN][h_AN][i]);
	    }
	    free(iCntHNL[k][Mc_AN][h_AN]);
	  }
	  free(iCntHNL[k][Mc_AN]);
	}
	free(iCntHNL[k]);
      }
      free(iCntHNL);
    }

    /* for core hole calculations */

    if (core_hole_state_flag==1){

      /* HCH */

      for (k=0; k<List_YOUSO[5]; k++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(HCH[k][Mc_AN][h_AN][i]);
	    }
	    free(HCH[k][Mc_AN][h_AN]);
	  }
	  free(HCH[k][Mc_AN]);
	}
	free(HCH[k]);
      }
      free(HCH);

      /* iHCH */

      if ( SpinP_switch==3 ){

	for (k=0; k<List_YOUSO[5]; k++){
	  for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	      FNAN[0] = 0;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      for (i=0; i<tno0; i++){
		free(iHCH[k][Mc_AN][h_AN][i]);
	      }
	      free(iHCH[k][Mc_AN][h_AN]);
	    }
	    free(iHCH[k][Mc_AN]);
	  }
	  free(iHCH[k]);
	}
	free(iHCH);
      }
    }

    /* H_Hub  --- added by MJ */  

    if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

      for (k=0; k<=SpinP_switch; k++){

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(H_Hub[k][Mc_AN][h_AN][i]);
	    }
            free(H_Hub[k][Mc_AN][h_AN]);
	  }
          free(H_Hub[k][Mc_AN]);
	}
        free(H_Hub[k]);
      }
      free(H_Hub);
    }

    /* H_Zeeman_NCO */  

    if (Zeeman_NCO_switch==1){

      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (i=0; i<tno0; i++){
	  free(H_Zeeman_NCO[Mc_AN][i]);
	}
        free(H_Zeeman_NCO[Mc_AN]);
      }
      free(H_Zeeman_NCO);
    }

    /* iHNL0 */

    if (SpinP_switch==3){

      for (k=0; k<List_YOUSO[5]; k++){

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(iHNL0[k][Mc_AN][h_AN][i]);
	    }
	    free(iHNL0[k][Mc_AN][h_AN]);
	  }
	  free(iHNL0[k][Mc_AN]);
	}
        free(iHNL0[k]);
      }
      free(iHNL0);
    }

    /* OLP_L */  

    for (k=0; k<3; k++){

      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = F_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(OLP_L[k][Mc_AN][h_AN][i]);
	  }
          free(OLP_L[k][Mc_AN][h_AN]);
	}
        free(OLP_L[k][Mc_AN]);
      }
      free(OLP_L[k]);
    }
    free(OLP_L);

    /* OLP */

    for (k=0; k<4; k++){
      for (Mc_AN=0; Mc_AN<(Matomnum+MatomnumF+MatomnumS+2); Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
          FNAN[0] = 0;
	  fan = FNAN[Gc_AN]; 
	}
        else if (Mc_AN==(Matomnum+1)){
          tno0 = List_YOUSO[7];
          fan = List_YOUSO[8];
        }
        else if ( (Hub_U_switch==0 || Hub_U_occupation!=1 || core_hole_state_flag!=1) 
                  && 0<k 
                  && (Matomnum+1)<Mc_AN
                  && Mc_AN<=(Matomnum+MatomnumF+MatomnumS)){

	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = 1;
	  fan = FNAN[Gc_AN]; 
	}
	else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	  fan = FNAN[Gc_AN]; 
	}    
	else {
	  tno0 = List_YOUSO[7];
	  fan = List_YOUSO[8];
	}

	for (h_AN=0; h_AN<=fan; h_AN++){
	  for (i=0; i<tno0; i++){
	    free(OLP[k][Mc_AN][h_AN][i]);
	  }
          free(OLP[k][Mc_AN][h_AN]);
	}
        free(OLP[k][Mc_AN]);
      }
      free(OLP[k]);
    }
    free(OLP);

    /*** added by Ohwaki ***/

    /* OLP_p */

    if (0<=CLE_Type){

      for (k=0; k<4; k++){
	for (Mc_AN=0; Mc_AN<(Matomnum+MatomnumF+MatomnumS+2); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	    fan = FNAN[Gc_AN]; 
	  }
	  else if (Mc_AN==(Matomnum+1)){
	    tno0 = List_YOUSO[7];
	    fan = List_YOUSO[8];
	  }
	  else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	    fan = FNAN[Gc_AN]; 
	  }    
	  else {
	    tno0 = List_YOUSO[7];
	    fan = List_YOUSO[8];
	  }

	  for (h_AN=0; h_AN<=fan; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(OLP_p[k][Mc_AN][h_AN][i]);
	    }
	    free(OLP_p[k][Mc_AN][h_AN]);
	  }
	  free(OLP_p[k][Mc_AN]);
	}
	free(OLP_p[k]);
      }
      free(OLP_p);
    }

    /*** added by Ohwaki (end) ***/

    /* OLP_CH */

    if (core_hole_state_flag==1){

      for (k=0; k<4; k++){
	for (Mc_AN=0; Mc_AN<(Matomnum+MatomnumF+MatomnumS+2); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	    fan = FNAN[Gc_AN]; 
	  }
	  else if (Mc_AN==(Matomnum+1)){
	    tno0 = List_YOUSO[7];
	    fan = List_YOUSO[8];
	  }
	  else if ( Mc_AN<=(Matomnum+MatomnumF+MatomnumS) ){
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	    fan = FNAN[Gc_AN]; 
	  }    
	  else {
	    tno0 = List_YOUSO[7];
	    fan = List_YOUSO[8];
	  }

	  for (h_AN=0; h_AN<=fan; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(OLP_CH[k][Mc_AN][h_AN][i]);
	    }
	    free(OLP_CH[k][Mc_AN][h_AN]);
	  }
	  free(OLP_CH[k][Mc_AN]);
	}
	free(OLP_CH[k]);
      }
      free(OLP_CH);
    }

    /* CntOLP */

    if (Cnt_switch==1){

      for (k=0; k<4; k++){
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    FNAN[0] = 0;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(CntOLP[k][Mc_AN][h_AN][i]);
	    }
	    free(CntOLP[k][Mc_AN][h_AN]);
	  }
	  free(CntOLP[k][Mc_AN]);
	}
	free(CntOLP[k]);
      }
      free(CntOLP);
    }

    /* H */

    for (k=0; k<=SpinP_switch; k++){
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
	}
	else{
          Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(H[k][Mc_AN][h_AN][i]);
	  }

	  free(H[k][Mc_AN][h_AN]);
	}
	free(H[k][Mc_AN]);
      }
      free(H[k]);
    }
    free(H);

    /* CntH */

    if (Cnt_switch==1){

      for (k=0; k<=SpinP_switch; k++){
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = S_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_CNO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(CntH[k][Mc_AN][h_AN][i]);
	    }

	    free(CntH[k][Mc_AN][h_AN]);
	  }
	  free(CntH[k][Mc_AN]);
	}
	free(CntH[k]);
      }
      free(CntH);
    }

    /* DS_NL */  

    for (so=0; so<(SO_switch+1); so++){
      for (k=0; k<4; k++){
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    fan = FNAN[Gc_AN];
	  }
	  else if ( (Matomnum+1)<=Mc_AN ){
	    fan = List_YOUSO[8];
	    tno0 = List_YOUSO[7];
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	    fan = FNAN[Gc_AN];
	  }    

	  for (h_AN=0; h_AN<(fan+1); h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else if ( (Matomnum+1)<=Mc_AN ){
	      tno1 = List_YOUSO[20];  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_VPS_Pro[Hwan] + 2;
	    } 

	    for (i=0; i<tno0; i++){
	      free(DS_NL[so][k][Mc_AN][h_AN][i]);
	    }
	    free(DS_NL[so][k][Mc_AN][h_AN]);
	  }
	  free(DS_NL[so][k][Mc_AN]);
	}
	free(DS_NL[so][k]);
      }
      free(DS_NL[so]);
    }
    free(DS_NL);

    /* CntDS_NL */  

    if (Cnt_switch==1){

      for (so=0; so<(SO_switch+1); so++){
	for (k=0; k<4; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	      fan = FNAN[Gc_AN];
	    }
	    else if ( (Matomnum+1)<=Mc_AN ){
	      fan = List_YOUSO[8];
	      tno0 = List_YOUSO[7];
	    }
	    else{
	      Gc_AN = F_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_CNO[Cwan];  
	      fan = FNAN[Gc_AN];
	    }    

	    for (h_AN=0; h_AN<(fan+1); h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else if ( (Matomnum+1)<=Mc_AN ){
		tno1 = List_YOUSO[20];  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_VPS_Pro[Hwan] + 2;
	      } 

	      for (i=0; i<tno0; i++){
		free(CntDS_NL[so][k][Mc_AN][h_AN][i]);
	      }
	      free(CntDS_NL[so][k][Mc_AN][h_AN]);
	    }
	    free(CntDS_NL[so][k][Mc_AN]);
	  }
	  free(CntDS_NL[so][k]);
	}
	free(CntDS_NL[so]);
      }
      free(CntDS_NL);
    }

    /* for LNO */  

    if (LNO_flag==1){

      for (k=0; k<=SpinP_switch; k++){
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){
	  free(LNO_coes[k][Mc_AN]);
	}
        free(LNO_coes[k]);
      }
      free(LNO_coes);

      for (k=0; k<=SpinP_switch; k++){
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF+MatomnumS); Mc_AN++){
	  free(LNO_pops[k][Mc_AN]);
	}
        free(LNO_pops[k]);
      }
      free(LNO_pops);

      for (k=0; k<=SpinP_switch; k++){

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(DM0[k][Mc_AN][h_AN][i]);
	    }
            free(DM0[k][Mc_AN][h_AN]);
	  }
          free(DM0[k][Mc_AN]);
	}
        free(DM0[k]);
      }
      free(DM0);

    }

    /* for RMM-DIISH */

    if (Mixing_switch==5){

      /* HisH1 */

      for (m=0; m<List_YOUSO[39]; m++){
	for (k=0; k<=SpinP_switch; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      for (i=0; i<tno0; i++){
		free(HisH1[m][k][Mc_AN][h_AN][i]);
	      }
              free(HisH1[m][k][Mc_AN][h_AN]);
	    }
            free(HisH1[m][k][Mc_AN]);
	  }
          free(HisH1[m][k]);
	}
        free(HisH1[m]);
      }
      free(HisH1);

      /* HisH2 */

      if (SpinP_switch==3){

	for (m=0; m<List_YOUSO[39]; m++){
	  for (k=0; k<SpinP_switch; k++){
	    FNAN[0] = 0;
	    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	      if (Mc_AN==0){
		Gc_AN = 0;
		tno0 = 1;
	      }
	      else{
		Gc_AN = S_M2G[Mc_AN];
		Cwan = WhatSpecies[Gc_AN];
		tno0 = Spe_Total_NO[Cwan];  
	      }    

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

		if (Mc_AN==0){
		  tno1 = 1;  
		}
		else{
		  Gh_AN = natn[Gc_AN][h_AN];
		  Hwan = WhatSpecies[Gh_AN];
		  tno1 = Spe_Total_NO[Hwan];
		} 

		for (i=0; i<tno0; i++){
		  free(HisH2[m][k][Mc_AN][h_AN][i]);
		}
                free(HisH2[m][k][Mc_AN][h_AN]);
	      }
              free(HisH2[m][k][Mc_AN]);
	    }
            free(HisH2[m][k]);
	  }
          free(HisH2[m]);
	}
        free(HisH2);

      } /* if (SpinP_switch==3) */

      /* ResidualH1 */

      for (m=0; m<List_YOUSO[39]; m++){
	for (k=0; k<=SpinP_switch; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      for (i=0; i<tno0; i++){
		free(ResidualH1[m][k][Mc_AN][h_AN][i]);
	      }
              free(ResidualH1[m][k][Mc_AN][h_AN]);
	    }
            free(ResidualH1[m][k][Mc_AN]);
	  }
          free(ResidualH1[m][k]);
	}
        free(ResidualH1[m]);
      }
      free(ResidualH1);

      /* ResidualH2 */

      if (SpinP_switch==3){

	for (m=0; m<List_YOUSO[39]; m++){
	  for (k=0; k<SpinP_switch; k++){
	    FNAN[0] = 0;
	    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	      if (Mc_AN==0){
		Gc_AN = 0;
		tno0 = 1;
	      }
	      else{
		Gc_AN = S_M2G[Mc_AN];
		Cwan = WhatSpecies[Gc_AN];
		tno0 = Spe_Total_NO[Cwan];  
	      }    

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

		if (Mc_AN==0){
		  tno1 = 1;  
		}
		else{
		  Gh_AN = natn[Gc_AN][h_AN];
		  Hwan = WhatSpecies[Gh_AN];
		  tno1 = Spe_Total_NO[Hwan];
		} 

		for (i=0; i<tno0; i++){
		  free(ResidualH2[m][k][Mc_AN][h_AN][i]);
		}
                free(ResidualH2[m][k][Mc_AN][h_AN]);
	      }
              free(ResidualH2[m][k][Mc_AN]);
	    }
            free(ResidualH2[m][k]);
	  }
          free(ResidualH2[m]);
	}
        free(ResidualH2);

      } /* if (SpinP_switch==3) */

    } /* if (Mixing_switch==5) */

    /* for RMM-DIIS */

    if (Mixing_switch==1 || Mixing_switch==6){

      /* HisH1 */

      for (m=0; m<List_YOUSO[39]; m++){
	for (k=0; k<=SpinP_switch; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = S_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      for (i=0; i<tno0; i++){
		free(HisH1[m][k][Mc_AN][h_AN][i]);
	      }
              free(HisH1[m][k][Mc_AN][h_AN]);
	    }
            free(HisH1[m][k][Mc_AN]);
	  }
          free(HisH1[m][k]);
	}
        free(HisH1[m]);
      }
      free(HisH1);

      /* HisH2 */

      if (SpinP_switch==3){

	for (m=0; m<List_YOUSO[39]; m++){
	  for (k=0; k<SpinP_switch; k++){
	    FNAN[0] = 0;
	    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	      if (Mc_AN==0){
		Gc_AN = 0;
		tno0 = 1;
	      }
	      else{
		Gc_AN = S_M2G[Mc_AN];
		Cwan = WhatSpecies[Gc_AN];
		tno0 = Spe_Total_NO[Cwan];  
	      }    

	      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

		if (Mc_AN==0){
		  tno1 = 1;  
		}
		else{
		  Gh_AN = natn[Gc_AN][h_AN];
		  Hwan = WhatSpecies[Gh_AN];
		  tno1 = Spe_Total_NO[Hwan];
		} 

		for (i=0; i<tno0; i++){
		  free(HisH2[m][k][Mc_AN][h_AN][i]);
		}
                free(HisH2[m][k][Mc_AN][h_AN]);
	      }
              free(HisH2[m][k][Mc_AN]);
	    }
            free(HisH2[m][k]);
	  }
          free(HisH2[m]);
	}
        free(HisH2);

      } /* if (SpinP_switch==3) */

    } /* if (Mixing_switch==1 || Mixing_switch==6) */

    /* DM */  

    for (m=0; m<List_YOUSO[16]; m++){
      for (k=0; k<=SpinP_switch; k++){
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
            Gc_AN = 0;
	    tno0 = 1;
            FNAN[0] = 0;
	  }
	  else{
            Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(DM[m][k][Mc_AN][h_AN][i]);
	    }
            free(DM[m][k][Mc_AN][h_AN]);
	  }
          free(DM[m][k][Mc_AN]);
	}
        free(DM[m][k]);
      }
      free(DM[m]);
    }
    free(DM);

    /* Partial_DM */

    if (cal_partial_charge==1){

      for (k=0; k<=1; k++){

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(Partial_DM[k][Mc_AN][h_AN][i]);
	    }
            free(Partial_DM[k][Mc_AN][h_AN]);
	  }
          free(Partial_DM[k][Mc_AN]);
	}
        free(Partial_DM[k]);
      }
      free(Partial_DM);
    }

    /* DM_onsite   --- MJ */  

    if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){

      for (m=0; m<2; m++){
	for (k=0; k<=SpinP_switch; k++){

	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  
	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }  

	    h_AN = 0;
	      
	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(DM_onsite[m][k][Mc_AN][i]);
	    }

            free(DM_onsite[m][k][Mc_AN]);
	  }
          free(DM_onsite[m][k]);
	}
        free(DM_onsite[m]);
      }
      free(DM_onsite);
    }

    /* v_eff   --- MJ */  

    if (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1){
   
      for (k=0; k<=SpinP_switch; k++){

	FNAN[0] = 0;

        for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }  

	  for (i=0; i<tno0; i++){
	    free(v_eff[k][Mc_AN][i]);
	  }
	  free(v_eff[k][Mc_AN]);

	}
	free(v_eff[k]);
      }
      free(v_eff);
    }

    /*  NC_OcpN */

    if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) 
         && SpinP_switch==3 ){

      for (m=0; m<2; m++){
	for (s1=0; s1<2; s1++){
	  for (s2=0; s2<2; s2++){
	    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	      if (Mc_AN==0){
		Gc_AN = 0;
		tno0 = 1;
	      }
	      else{
		Gc_AN = M2G[Mc_AN];
		Cwan = WhatSpecies[Gc_AN];
		tno0 = Spe_Total_NO[Cwan];  
	      }  

	      for (i=0; i<tno0; i++){
                free(NC_OcpN[m][s1][s2][Mc_AN][i]);
	      }
              free(NC_OcpN[m][s1][s2][Mc_AN]);
	    }
            free(NC_OcpN[m][s1][s2]);
	  }
          free(NC_OcpN[m][s1]);
	}
        free(NC_OcpN[m]);
      }
      free(NC_OcpN);
    }

    /*  NC_v_eff */

    if ( (Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
         && SpinP_switch==3 ){

      for (s1=0; s1<2; s1++){
	for (s2=0; s2<2; s2++){
	  for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = F_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }  

	    for (i=0; i<tno0; i++){
	      free(NC_v_eff[s1][s2][Mc_AN][i]);
	    }
            free(NC_v_eff[s1][s2][Mc_AN]);
	  }
          free(NC_v_eff[s1][s2]);
	}
        free(NC_v_eff[s1]);
      }           
      free(NC_v_eff);
    }

    /* ResidualDM */  

    if ( Mixing_switch==0 || Mixing_switch==1 || Mixing_switch==2 || Mixing_switch==6 ){

      for (m=0; m<List_YOUSO[16]; m++){
	for (k=0; k<=SpinP_switch; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      for (i=0; i<tno0; i++){
		free(ResidualDM[m][k][Mc_AN][h_AN][i]);
	      }
	      free(ResidualDM[m][k][Mc_AN][h_AN]);
	    }
	    free(ResidualDM[m][k][Mc_AN]);
	  }
	  free(ResidualDM[m][k]);
	}
	free(ResidualDM[m]);
      }
      free(ResidualDM);
    }

    /* iResidualDM */  

    if ( (Mixing_switch==0 || Mixing_switch==1 || Mixing_switch==2 || Mixing_switch==6)
	 && SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
         || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1) ){

      for (m=0; m<List_YOUSO[16]; m++){
        for (k=0; k<2; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_NO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];
		Hwan = WhatSpecies[Gh_AN];
		tno1 = Spe_Total_NO[Hwan];
	      } 

	      for (i=0; i<tno0; i++){
		free(iResidualDM[m][k][Mc_AN][h_AN][i]);
	      }
 	      free(iResidualDM[m][k][Mc_AN][h_AN]);
	    }
            free(iResidualDM[m][k][Mc_AN]);
	  }
          free(iResidualDM[m][k]);
	}
        free(iResidualDM[m]);
      }
      free(iResidualDM);
    }
    else{
      for (m=0; m<List_YOUSO[16]; m++){
        free(iResidualDM[m][0][0][0][0]);
        free(iResidualDM[m][0][0][0]);
        free(iResidualDM[m][0][0]);
        free(iResidualDM[m][0]);
        free(iResidualDM[m]);
      }   
      free(iResidualDM);
    }

    /* EDM */

    for (k=0; k<=SpinP_switch; k++){
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
	}
	else{
          Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(EDM[k][Mc_AN][h_AN][i]);
	  }

	  free(EDM[k][Mc_AN][h_AN]);
	}
	free(EDM[k][Mc_AN]);
      }
      free(EDM[k]);
    }
    free(EDM);

    /* PDM */

    for (k=0; k<=SpinP_switch; k++){
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
          Gc_AN = 0;
	  tno0 = 1;
	}
	else{
          Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(PDM[k][Mc_AN][h_AN][i]);
	  }

	  free(PDM[k][Mc_AN][h_AN]);
	}
	free(PDM[k][Mc_AN]);
      }
      free(PDM[k]);
    }
    free(PDM);

    /* iDM */

    for (m=0; m<List_YOUSO[16]; m++){
      for (k=0; k<2; k++){

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_NO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(iDM[m][k][Mc_AN][h_AN][i]);
	    }
	    free(iDM[m][k][Mc_AN][h_AN]);
	  }
	  free(iDM[m][k][Mc_AN]);
	}
	free(iDM[m][k]);
      }
      free(iDM[m]);
    }
    free(iDM);

    /* S12 for DC, recursion, or DCLNO */  

    if (Solver==1 || Solver==5 ){

      int *Msize,myid1;

      Msize = (int*)malloc(sizeof(int)*(Matomnum+2));

      if ( Solver==1 || Solver==5 ){

	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0) n2 = 1;
	  else{
	    Gc_AN = M2G[Mc_AN];

	    num = 1;
	    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
	      Gi = natn[Gc_AN][i];
	      wanA = WhatSpecies[Gi];
	      num += Spe_Total_CNO[wanA];
	    }
	    n2 = num + 2;
	  }

  	  Msize[Mc_AN] = n2;

	  for (i=0; i<n2; i++){
	    free(S12[Mc_AN][i]);
	  }
	  free(S12[Mc_AN]);
	}
	free(S12);
      }

      free(Msize);
    }

    /* CntCoes */

    if (Cnt_switch==1){
      for (i=0; i<=(Matomnum+MatomnumF); i++){
	for (j=0; j<List_YOUSO[7]; j++){
	  free(CntCoes[i][j]);
	}
	free(CntCoes[i]);
      }
      free(CntCoes);

      for (i=0; i<(SpeciesNum+1); i++){
	for (j=0; j<List_YOUSO[7]; j++){
	  free(CntCoes_Species[i][j]);
	}
	free(CntCoes_Species[i]);
      }
      free(CntCoes_Species);
    }

    if (ProExpn_VNA==1){

      /* HVNA */  

      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(HVNA[Mc_AN][h_AN][i]);
	  }
          free(HVNA[Mc_AN][h_AN]);
	}
        free(HVNA[Mc_AN]);
      }
      free(HVNA);

      /* DS_VNA */  

      for (k=0; k<4; k++){

	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){
          
	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	    fan = FNAN[Gc_AN];
	  }
  	  else if ( (Matomnum+1)<=Mc_AN ){
	    fan = List_YOUSO[8];
	    tno0 = List_YOUSO[7];
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	    fan = FNAN[Gc_AN];
	  }
          
	  for (h_AN=0; h_AN<(fan+1); h_AN++){

	    if (Mc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      tno1 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	    } 

	    for (i=0; i<tno0; i++){
	      free(DS_VNA[k][Mc_AN][h_AN][i]);
	    }
            free(DS_VNA[k][Mc_AN][h_AN]);
	  }
          free(DS_VNA[k][Mc_AN]);
	}
        free(DS_VNA[k]);
      }
      free(DS_VNA);

      /* CntDS_VNA */  

      if (Cnt_switch==1){

	for (k=0; k<4; k++){

	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<(Matomnum+2); Mc_AN++){
          
	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	      fan = FNAN[Gc_AN];
	    }
	    else if ( Mc_AN==(Matomnum+1) ){
	      fan = List_YOUSO[8];
	      tno0 = List_YOUSO[7];
	    }
	    else{
	      Gc_AN = F_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_CNO[Cwan];  
              fan = FNAN[Gc_AN];
	    }

	    for (h_AN=0; h_AN<(fan+1); h_AN++){

	      if (Mc_AN==0){
		tno1 = 1;  
	      }
	      else{
		tno1 = (List_YOUSO[35]+1)*(List_YOUSO[35]+1)*List_YOUSO[34];
	      } 

	      for (i=0; i<tno0; i++){
		free(CntDS_VNA[k][Mc_AN][h_AN][i]);
	      }
 	      free(CntDS_VNA[k][Mc_AN][h_AN]);
	    }
            free(CntDS_VNA[k][Mc_AN]);
	  }
          free(CntDS_VNA[k]);
	}
        free(CntDS_VNA);
      }

      /* HVNA2 */

      for (k=0; k<4; k++){
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Gc_AN = F_M2G[Mc_AN];
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_NO[Cwan];  
	  }    

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    for (i=0; i<tno0; i++){
	      free(HVNA2[k][Mc_AN][h_AN][i]);
	    }
	    free(HVNA2[k][Mc_AN][h_AN]);
	  }
	  free(HVNA2[k][Mc_AN]);
	}
	free(HVNA2[k]);
      }
      free(HVNA2);

      /* HVNA3 */

      for (k=0; k<4; k++){
	FNAN[0] = 0;
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	  if (Mc_AN==0)  Gc_AN = 0;
	  else           Gc_AN = F_M2G[Mc_AN];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Mc_AN==0){
	      tno0 = 1;
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno0 = Spe_Total_NO[Hwan];  
	    }    

	    for (i=0; i<tno0; i++){
	      free(HVNA3[k][Mc_AN][h_AN][i]);
	    }
	    free(HVNA3[k][Mc_AN][h_AN]);
	  }
	  free(HVNA3[k][Mc_AN]);
	}
	free(HVNA3[k]);
      }
      free(HVNA3);

      /* CntHVNA2 */

      if (Cnt_switch==1){

	for (k=0; k<4; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0){
	      Gc_AN = 0;
	      tno0 = 1;
	    }
	    else{
	      Gc_AN = F_M2G[Mc_AN];
	      Cwan = WhatSpecies[Gc_AN];
	      tno0 = Spe_Total_CNO[Cwan];  
	    }    

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      for (i=0; i<tno0; i++){
		free(CntHVNA2[k][Mc_AN][h_AN][i]);
	      }
	      free(CntHVNA2[k][Mc_AN][h_AN]);
	    }
	    free(CntHVNA2[k][Mc_AN]);
	  }
	  free(CntHVNA2[k]);
	}
	free(CntHVNA2);
      }

      /* CntHVNA3 */

      if (Cnt_switch==1){

	for (k=0; k<4; k++){
	  FNAN[0] = 0;
	  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	    if (Mc_AN==0) Gc_AN = 0;
	    else          Gc_AN = F_M2G[Mc_AN];

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	      if (Mc_AN==0){
		tno0 = 1;
	      }
	      else{
		Gh_AN = natn[Gc_AN][h_AN];        
		Hwan = WhatSpecies[Gh_AN];
		tno0 = Spe_Total_CNO[Hwan];  
	      }    

	      for (i=0; i<tno0; i++){
		free(CntHVNA3[k][Mc_AN][h_AN][i]);
	      }
	      free(CntHVNA3[k][Mc_AN][h_AN]);
	    }
	    free(CntHVNA3[k][Mc_AN]);
	  }
	  free(CntHVNA3[k]);
	}
	free(CntHVNA3);
      }
    }  

    if (Solver==8) { /* Krylov subspace method */

      for (i=0; i<=SpinP_switch; i++){
	for (j=0; j<=Matomnum; j++){
          free(Krylov_U[i][j]);
	}
        free(Krylov_U[i]);
      }
      free(Krylov_U);

      for (i=0; i<=SpinP_switch; i++){
	for (j=0; j<=Matomnum; j++){
	  for (k=0; k<(rlmax_EC[j]*EKC_core_size[j]+1); k++){
	    free(EC_matrix[i][j][k]);
	  }
          free(EC_matrix[i][j]);
	}
        free(EC_matrix[i]);
      }
      free(EC_matrix);

      free(rlmax_EC);
      free(rlmax_EC2);
      free(EKC_core_size);
      free(scale_rc_EKC);

    } /* if (Solver==8) */

    /* NEGF */

    if (Solver==4){
      for (spin=0; spin<(SpinP_switch+1); spin++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  free(TRAN_DecMulP[spin][Mc_AN]);
	}
	free(TRAN_DecMulP[spin]);
      }
      free(TRAN_DecMulP);
    }

    /* Energy decomposition */

    if (Energy_Decomposition_flag==1){

      /* DecEkin */
      for (spin=0; spin<2; spin++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  free(DecEkin[spin][Mc_AN]);
	}
        free(DecEkin[spin]);
      }
      free(DecEkin);

      /* DecEv */
      for (spin=0; spin<2; spin++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  free(DecEv[spin][Mc_AN]);
	}
        free(DecEv[spin]);
      }
      free(DecEv);

      /* DecEcon */
      for (spin=0; spin<2; spin++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  free(DecEcon[spin][Mc_AN]);
	}
        free(DecEcon[spin]);
      }
      free(DecEcon);

      /* DecEscc */
      for (spin=0; spin<2; spin++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  free(DecEscc[spin][Mc_AN]);
	}
        free(DecEscc[spin]);
      }
      free(DecEscc);

      /* DecEvdw */
      for (spin=0; spin<2; spin++){
	for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	  free(DecEvdw[spin][Mc_AN]);
	}
        free(DecEvdw[spin]);
      }
      free(DecEvdw);

    } /* if (Energy_Decomposition_flag==1) */

  } /*  if (alloc_first[4]==0){ */

  /****************************************************
    freeing of arrays:
  ****************************************************/

  if (alloc_first[0]==0){

    FNAN[0] = 0; 
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0) Gc_AN = 0;
      else          Gc_AN = M2G[Mc_AN];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        free(GListTAtoms2[Mc_AN][h_AN]);
        free(GListTAtoms1[Mc_AN][h_AN]);
      }
      free(GListTAtoms2[Mc_AN]);
      free(GListTAtoms1[Mc_AN]);
    }
    free(GListTAtoms2);
    free(GListTAtoms1);
  }

  if (alloc_first[3]==0){

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Density_Grid[k]);
      }
      free(Density_Grid);
    }
    else{ 
      for (k=0; k<=1; k++){
        free(Density_Grid[k]);
      }
      free(Density_Grid);
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Vxc_Grid[k]);
      }
      free(Vxc_Grid);
    }
    else{
      for (k=0; k<=1; k++){
        free(Vxc_Grid[k]);
      }
      free(Vxc_Grid);
    }

    free(RefVxc_Grid);
    free(dVHart_Grid);

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Vpot_Grid[k]);
      }
      free(Vpot_Grid);
    }
    else{
      for (k=0; k<=1; k++){
        free(Vpot_Grid[k]);
      }
      free(Vpot_Grid);
    }

    /* arrays for the partitions B and C */

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Density_Grid_B[k]);
      }
      free(Density_Grid_B);
    }
    else{
      for (k=0; k<=1; k++){
        free(Density_Grid_B[k]);
      }
      free(Density_Grid_B);
    }

    free(ADensity_Grid_B);
    free(PCCDensity_Grid_B[1]); free(PCCDensity_Grid_B[0]);
    free(PCCDensity_Grid_B);

    free(dVHart_Grid_B);

    if ( (core_hole_state_flag==1 && Scf_RestartFromFile==1) || scf_coulomb_cutoff_CoreHole==1 ){
      free(dVHart_Periodic_Grid_B);
      free(Density_Periodic_Grid_B);
    }

    free(RefVxc_Grid_B);

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Vxc_Grid_B[k]);
      }
      free(Vxc_Grid_B);
    }
    else{
      for (k=0; k<=1; k++){
        free(Vxc_Grid_B[k]);
      }
      free(Vxc_Grid_B);
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Vpot_Grid_B[k]);
      }
      free(Vpot_Grid_B);
    }
    else{
      for (k=0; k<=1; k++){
        free(Vpot_Grid_B[k]);
      }
      free(Vpot_Grid_B);
    }

    /* if ( Mixing_switch==7 ) */

    if ( Mixing_switch==7 ){

      int spinmax;

      if      (SpinP_switch==0)  spinmax = 1;
      else if (SpinP_switch==1)  spinmax = 2;
      else if (SpinP_switch==3)  spinmax = 3;

      for (m=0; m<List_YOUSO[38]; m++){
	for (spin=0; spin<spinmax; spin++){
	  free(ReVKSk[m][spin]);
	}
        free(ReVKSk[m]);
      }
      free(ReVKSk);

      for (m=0; m<List_YOUSO[38]; m++){
	for (spin=0; spin<spinmax; spin++){
	  free(ImVKSk[m][spin]);
	}
        free(ImVKSk[m]);
      }
      free(ImVKSk);

      for (spin=0; spin<spinmax; spin++){
	free(Residual_ReVKSk[spin]);
      }
      free(Residual_ReVKSk);

      for (spin=0; spin<spinmax; spin++){
	free(Residual_ImVKSk[spin]);
      }
      free(Residual_ImVKSk);
    }

    /* if (ProExpn_VNA==off) */
    if (ProExpn_VNA==0){
      free(VNA_Grid);
      free(VNA_Grid_B);
    }

    /* electric energy by electric field */
    if (E_Field_switch==1){
      free(VEF_Grid);
      free(VEF_Grid_B);
    }

    /* arrays for the partition D */

    free(PCCDensity_Grid_D[1]); free(PCCDensity_Grid_D[0]);
    free(PCCDensity_Grid_D); 

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Density_Grid_D[k]);
      }
      free(Density_Grid_D);
    }
    else{
      for (k=0; k<=1; k++){
        free(Density_Grid_D[k]);
      }
      free(Density_Grid_D);
    }

    if (SpinP_switch==3){ /* spin non-collinear */
      for (k=0; k<=3; k++){
        free(Vxc_Grid_D[k]);
      }
      free(Vxc_Grid_D);
    }
    else{
      for (k=0; k<=1; k++){
        free(Vxc_Grid_D[k]);
      }
      free(Vxc_Grid_D);
    }

    /* Orbs_Grid */

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        Gc_AN = 0;
        num = 1;
      }
      else{
        Gc_AN = F_M2G[Mc_AN];
        num = GridN_Atom[Gc_AN];
      }

      for (Nc=0; Nc<num; Nc++){
        free(Orbs_Grid[Mc_AN][Nc]);
      }
      free(Orbs_Grid[Mc_AN]); 
    }
    free(Orbs_Grid); 

    /* COrbs_Grid */

    if (Cnt_switch!=0){
      for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
        if (Mc_AN==0){
          tno = 1;
          Gc_AN = 0;
        }
        else{
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno = Spe_Total_CNO[Cwan];
        }

        for (i=0; i<tno; i++){
          free(COrbs_Grid[Mc_AN][i]); 
        }
        free(COrbs_Grid[Mc_AN]); 
      }
      free(COrbs_Grid);
    }

    /* Orbs_Grid_FNAN */

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        free(Orbs_Grid_FNAN[0][0][0]);
        free(Orbs_Grid_FNAN[0][0]);
      }
      else{

	Gc_AN = M2G[Mc_AN];    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];

	  if (G2ID[Gh_AN]!=myid){
            num = NumOLG[Mc_AN][h_AN] + 1;  
	  }
	  else {
            num = 1;
	  }

	  for (Nc=0; Nc<num; Nc++){
	    free(Orbs_Grid_FNAN[Mc_AN][h_AN][Nc]);
	  }
	  free(Orbs_Grid_FNAN[Mc_AN][h_AN]);

	} /* h_AN */
      } /* else */

      free(Orbs_Grid_FNAN[Mc_AN]);
    }
    free(Orbs_Grid_FNAN);

  }

  /****************************************************
    freeing of arrays:

     NumOLG
  ****************************************************/

  if (alloc_first[5]==0){

    /* NumOLG */
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(NumOLG[Mc_AN]);
    }
    free(NumOLG);

  } /* if (alloc_first[5]==0){ */

  /****************************************************
    freeing of arrays:

      RMI1
      RMI2
  ****************************************************/

  if (alloc_first[6]==0){

    FNAN[0] = 0;
    SNAN[0] = 0;

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0) Gc_AN = 0;
      else          Gc_AN = M2G[Mc_AN];
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
        free(RMI1[Mc_AN][i]);
      }      
      free(RMI1[Mc_AN]);
    }
    free(RMI1);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0) Gc_AN = 0;
      else          Gc_AN = M2G[Mc_AN];
      for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
        free(RMI2[Mc_AN][i]);
      }      
      free(RMI2[Mc_AN]);
    }
    free(RMI2);

  } /* if (alloc_first[6]==0){ */

  /****************************************************
    freeing of arrays:

      ratv
      atv
      atv_ijk
  ****************************************************/

  if (alloc_first[7]==0){

    for (i=0; i<(TCpyCell+1); i++){
      free(atv[i]);
    }
    free(atv);

    for (i=0; i<(2*CpyCell+4); i++){
      for (j=0; j<(2*CpyCell+4); j++){
	free(ratv[i][j]);
      }
      free(ratv[i]);
    }
    free(ratv);

    for (i=0; i<(TCpyCell+1); i++){
      free(atv_ijk[i]);
    }
    free(atv_ijk);

  } /* if (alloc_first[7]==0){ */

  /**********************************
    freeing of arrays: 
      GridListAtom
      CellListAtom
      MGridListAtom
  **********************************/

  if (alloc_first[2]==0){

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(MGridListAtom[Mc_AN]);
    }
    free(MGridListAtom);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(GridListAtom[Mc_AN]);
    }
    free(GridListAtom);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(CellListAtom[Mc_AN]);
    }
    free(CellListAtom);
  }

  /* for the EGAC method */

  if (alloc_first[34]==0){

    for (spin=0; spin<(SpinP_switch+1); spin++){
      for (Mc_AN=0; Mc_AN<Matomnum_EGAC; Mc_AN++){

        Gc_AN = M2G_EGAC[Mc_AN];
        Cwan = WhatSpecies[Gc_AN]; 
        tno1 = Spe_Total_CNO[Cwan];

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

          Gh_AN = natn[Gc_AN][h_AN];        
          Hwan = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[Hwan];
 
          for (i=0; i<tno1; i++){
            free(H_EGAC[spin][Mc_AN][h_AN][i]);
	  }
          free(H_EGAC[spin][Mc_AN][h_AN]);
	}
        free(H_EGAC[spin][Mc_AN]);
      }
      free(H_EGAC[spin]);
    }
    free(H_EGAC);

    for (Mc_AN=0; Mc_AN<Matomnum_EGAC; Mc_AN++){

      Gc_AN = M2G_EGAC[Mc_AN];
      Cwan = WhatSpecies[Gc_AN]; 
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];        
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];
 
	for (i=0; i<tno1; i++){
	  free(OLP_EGAC[Mc_AN][h_AN][i]);
	}
        free(OLP_EGAC[Mc_AN][h_AN]);
      }
      free(OLP_EGAC[Mc_AN]);
    }
    free(OLP_EGAC);

    int job_id,job_gid,N3[4];

    for (job_id=0; job_id<EGAC_Num; job_id++){

      job_gid = job_id + EGAC_Top[myid];
      GN2N_EGAC(job_gid,N3);
      Gc_AN = N3[1];

      for (i=0; i<(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1); i++){
	free(RMI1_EGAC[job_id][i]);
      }
      free(RMI1_EGAC[job_id]);
    }
    free(RMI1_EGAC);

    for (job_id=0; job_id<EGAC_Num; job_id++){

      job_gid = job_id + EGAC_Top[myid];
      GN2N_EGAC(job_gid,N3);
      Gc_AN = N3[1];

      for (i=0; i<(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1); i++){
	free(RMI2_EGAC[job_id][i]);
      }
      free(RMI2_EGAC[job_id]);
    }
    free(RMI2_EGAC);

    for (spin=0; spin<(SpinP_switch+1); spin++){
      for (Mc_AN=0; Mc_AN<Matomnum_DM_Snd_EGAC; Mc_AN++){

	Gc_AN = M2G_DM_Snd_EGAC[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_CNO[Cwan];

	for (h_AN=0; h_AN<(FNAN[Gc_AN]+1); h_AN++){
	  for (i=0; i<tno1; i++){
	    free(DM_Snd_EGAC[spin][Mc_AN][h_AN][i]);
	  }
          free(DM_Snd_EGAC[spin][Mc_AN][h_AN]);
	}
        free(DM_Snd_EGAC[spin][Mc_AN]);
      }
      free(DM_Snd_EGAC[spin]);
    }
    free(DM_Snd_EGAC);

    free(dim_GD_EGAC);
    free(dim_IA_EGAC);

    for (m=0; m<DIIS_History_EGAC; m++){
      for (job_id=0; job_id<EGAC_Num; job_id++){
	free(Sigma_EGAC[m][job_id]);
      }  
      free(Sigma_EGAC[m]);
    }
    free(Sigma_EGAC);

    for (job_id=0; job_id<(EGAC_Num+1); job_id++){
      free(fGD_EGAC[job_id]);
    }  
    free(fGD_EGAC);

    for (job_id=0; job_id<EGAC_Num; job_id++){ /* job_id: local job_id */

      job_gid = job_id + EGAC_Top[myid]; /* job_gid: global job_id */
      GN2N_EGAC(job_gid,N3);

      Gc_AN = N3[1];
      Cwan = WhatSpecies[Gc_AN]; 
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];        
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  free(GD_EGAC[job_id][h_AN][i]);
	}
	free(GD_EGAC[job_id][h_AN]);

      } 
      free(GD_EGAC[job_id]);
    }
    free(GD_EGAC);

    for (k=0; k<Num_GA_EGAC; k++){       /* k is the first index of GA_EGAC */

      job_gid = M2G_JOB_EGAC[k];
      GN2N_EGAC(job_gid,N3);

      Gc_AN = N3[1];
      Cwan = WhatSpecies[Gc_AN]; 
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];        
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];

	for (i=0; i<tno1; i++){
	  free(GA_EGAC[k][h_AN][i]);
	}
        free(GA_EGAC[k][h_AN]);
      } 
      free(GA_EGAC[k]);
    }
    free(GA_EGAC);
  }

}



void Set_Inf_SndRcv()
{ 
  int i,ID,IDS,IDR,Mc_AN,Gc_AN,Num,ID1,Lh_AN,Gh_AN;
  int myid,numprocs,tag=999;
  int *flag_DoubleCounting;
  int **Rcv_FGAN,**Rcv_SGAN;
  int **Snd_FGAN,**Snd_SGAN;
  int *Num_Pro_Snd;

  int Rn,Rn2,m1,m2,m3,n1,n2,n3,j,po,Gj_AN;
  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*********************************
   allocation of arrays:

   int flag_DoubleCounting[atomnum+1]
  *********************************/

  flag_DoubleCounting = (int*)malloc(sizeof(int)*(atomnum+1));

  /*********************************
   initialize

      F_Snd_Num
      F_Rcv_Num
      S_Snd_Num
      S_Rcv_Num
  *********************************/

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num[ID] = 0;
    F_Rcv_Num[ID] = 0;
    S_Snd_Num[ID] = 0;
    S_Rcv_Num[ID] = 0;
  }
    
  /************************************************
      find F_Rcv_Num and S_Rcv_Num
  *************************************************/

  /* initialize flag_DoubleCounting */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    flag_DoubleCounting[Gc_AN] = 0;
  }

  /* find F_Rcv_Num */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (Lh_AN=0; Lh_AN<=FNAN[Gc_AN]; Lh_AN++){
      Gh_AN = natn[Gc_AN][Lh_AN];
      ID1 = G2ID[Gh_AN];
      if (flag_DoubleCounting[Gh_AN]==0 && ID1!=myid){
        F_Rcv_Num[ID1]++;
        flag_DoubleCounting[Gh_AN] = 1;
      }
    }
  }

  /* find S_Rcv_Num */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (Lh_AN=(FNAN[Gc_AN]+1); Lh_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); Lh_AN++){
      Gh_AN = natn[Gc_AN][Lh_AN];
      ID1 = G2ID[Gh_AN];
      if (flag_DoubleCounting[Gh_AN]==0 && ID1!=myid){
        S_Rcv_Num[ID1]++;
        flag_DoubleCounting[Gh_AN] = 1;
      }
    }
  }

  /************************************************
   allocation of array:

   int Rcv_GAN[numprocs]
              [F_Rcv_Num[ID]+S_Rcv_Num[ID]]
  *************************************************/

  if (alloc_first[12]==0){
    for (ID=0; ID<numprocs; ID++){
      free(Rcv_GAN[ID]); 
    }
    free(Rcv_GAN); 
  }

  Rcv_GAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Rcv_GAN[ID] = (int*)malloc(sizeof(int)*(F_Rcv_Num[ID]+S_Rcv_Num[ID]));
  }

  alloc_first[12] = 0;

  Rcv_FGAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Rcv_FGAN[ID] = (int*)malloc(sizeof(int)*F_Rcv_Num[ID]);
  }

  Rcv_SGAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Rcv_SGAN[ID] = (int*)malloc(sizeof(int)*S_Rcv_Num[ID]);
  }

  /************************************************
             set Rcv_FGAN and Rcv_SGAN
  *************************************************/
  
  /* initialize F_Rcv_Num and S_Rcv_Num */

  for (ID=0; ID<numprocs; ID++){
    F_Rcv_Num[ID] = 0;
    S_Rcv_Num[ID] = 0;
  }

  /* initialized flag_DoubleCounting */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    flag_DoubleCounting[Gc_AN] = 0;
  }

  /* set Rcv_FGAN */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (Lh_AN=0; Lh_AN<=FNAN[Gc_AN]; Lh_AN++){
      Gh_AN = natn[Gc_AN][Lh_AN];
      ID1 = G2ID[Gh_AN];
      if (flag_DoubleCounting[Gh_AN]==0 && ID1!=myid){

        Rcv_FGAN[ID1][F_Rcv_Num[ID1]] = Gh_AN;
        F_Rcv_Num[ID1]++;
        flag_DoubleCounting[Gh_AN] = 1;
      }
    }
  }

  /* set Rcv_SGAN */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (Lh_AN=(FNAN[Gc_AN]+1); Lh_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); Lh_AN++){
      Gh_AN = natn[Gc_AN][Lh_AN];
      ID1 = G2ID[Gh_AN];
      if (flag_DoubleCounting[Gh_AN]==0 && ID1!=myid){

        Rcv_SGAN[ID1][S_Rcv_Num[ID1]] = Gh_AN;
        S_Rcv_Num[ID1]++;
        flag_DoubleCounting[Gh_AN] = 1;
      }
    }
  }

  /*****************************************
       MPI:  F_Rcv_Num
  *****************************************/

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);
  
  tag = 999;
  for (ID=0; ID<numprocs; ID++){
    if (ID!=0){
      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;
      MPI_Isend(&F_Rcv_Num[IDS],1,MPI_INT,IDS,tag,mpi_comm_level1,&request);
      MPI_Recv( &F_Snd_Num[IDR],1,MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
      MPI_Wait(&request,&stat);
    }
  }

  Snd_FGAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Snd_FGAN[ID] = (int*)malloc(sizeof(int)*F_Snd_Num[ID]);
  }

  for (ID=0; ID<numprocs; ID++){
    if (ID!=0){
      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;
      MPI_Isend(&Rcv_FGAN[IDS][0],F_Rcv_Num[IDS],MPI_INT,IDS,tag,mpi_comm_level1,&request);
      MPI_Recv( &Snd_FGAN[IDR][0],F_Snd_Num[IDR],MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
      MPI_Wait(&request,&stat);
    }
  }

  /*****************************************
       MPI:  S_Rcv_Num
  *****************************************/

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);
  
  tag = 999;
  for (ID=0; ID<numprocs; ID++){
    if (ID!=0){
      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;
      MPI_Isend(&S_Rcv_Num[IDS],1,MPI_INT,IDS,tag,mpi_comm_level1,&request);
      MPI_Recv( &S_Snd_Num[IDR],1,MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
      MPI_Wait(&request,&stat);
    }
  }

  Snd_SGAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Snd_SGAN[ID] = (int*)malloc(sizeof(int)*S_Snd_Num[ID]);
  }

  for (ID=0; ID<numprocs; ID++){
    if (ID!=0){
      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;
      MPI_Isend(&Rcv_SGAN[IDS][0],S_Rcv_Num[IDS],MPI_INT,IDS,tag,mpi_comm_level1,&request);
      MPI_Recv( &Snd_SGAN[IDR][0],S_Snd_Num[IDR],MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
      MPI_Wait(&request,&stat);
    }
  }

  /********************************************************
    allocation of arrays:
  
    int Snd_MAN[numprocs][F_Snd_Num[ID]+S_Snd_Num[ID]+1]
    int Snd_GAN[numprocs][F_Snd_Num[ID]+S_Snd_Num[ID]+1]
  *********************************************************/

  if (alloc_first[11]==0){
    for (ID=0; ID<numprocs; ID++){
      free(Snd_MAN[ID]); 
    }
    free(Snd_MAN);

    for (ID=0; ID<numprocs; ID++){
      free(Snd_GAN[ID]); 
    }
    free(Snd_GAN); 
  }  

  Snd_MAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Snd_MAN[ID] = (int*)malloc(sizeof(int)*(F_Snd_Num[ID]+S_Snd_Num[ID]+1));
  }

  Snd_GAN = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Snd_GAN[ID] = (int*)malloc(sizeof(int)*(F_Snd_Num[ID]+S_Snd_Num[ID]+1));
  }

  alloc_first[11] = 0;

  /************************************************
      find data structures to send informations
      related to FNAN from myid to the other IDs 
  *************************************************/

  for (ID=0; ID<numprocs; ID++){

    Num = 0;

    for (i=0; i<F_Snd_Num[ID]; i++){ 

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];

	if (Gc_AN==Snd_FGAN[ID][i]){

	  Snd_MAN[ID][Num] = Mc_AN;
	  Snd_GAN[ID][Num] = Gc_AN;
	  Num++;
	}      
      }
    }
  }

  /************************************************
      find data structures to send informations
      related to SNAN from myid to the other IDs 
  *************************************************/

  for (ID=0; ID<numprocs; ID++){

    Num = 0;

    for (i=0; i<S_Snd_Num[ID]; i++){ 

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];

	if (Gc_AN==Snd_SGAN[ID][i]){

	  Snd_MAN[ID][F_Snd_Num[ID]+Num] = Mc_AN;
	  Snd_GAN[ID][F_Snd_Num[ID]+Num] = Gc_AN;
	  Num++;
	}      
      }
    }
  }

  /************************************************
   MPI:

     Snd_GAN
     Rcv_GAN
  *************************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;
      MPI_Isend(&Snd_GAN[IDS][0], F_Snd_Num[IDS]+S_Snd_Num[IDS],
                MPI_INT,IDS, tag, mpi_comm_level1, &request);
      MPI_Recv(&Rcv_GAN[IDR][0],  F_Rcv_Num[IDR]+S_Rcv_Num[IDR],
                MPI_INT, IDR, tag, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
  }

  /************************************************
          setting of F_TopMAN and S_TopMAN

    F_TopMAN and S_TopMAN give the first intermediate
    atom number in atoms sent from ID in the size of
    F_Rcv_Num[ID] and F_Rcv_Num[ID] + S_Rcv_Num[ID],
    respectively.
  *************************************************/

  Num = Matomnum + 1;
  for (ID=0; ID<numprocs; ID++){
    if (F_Rcv_Num[ID]!=0 && ID!=myid){
      F_TopMAN[ID] = Num;
      Num = Num + F_Rcv_Num[ID];
    }
  }
  
  Num = Matomnum + 1;
  for (ID=0; ID<numprocs; ID++){
    if ((F_Rcv_Num[ID]!=0 || S_Rcv_Num[ID]!=0) && ID!=myid){
      S_TopMAN[ID] = Num;
      Num = Num + F_Rcv_Num[ID] + S_Rcv_Num[ID];
    }
  }

  /************************************************
       MatomnumF = the sum of F_Rcv_Num[ID]
       MatomnumS = the sum of S_Rcv_Num[ID]
  *************************************************/

  MatomnumF = 0;
  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid) MatomnumF += F_Rcv_Num[ID];
  }

  MatomnumS = 0;
  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid) MatomnumS += S_Rcv_Num[ID];
  }

  /************************************************
       allocation of arrays:
         
          F_M2G
          S_M2G
  *************************************************/

  if (alloc_first[13]==0){
    free(F_M2G);
    free(S_M2G);
  }

  F_M2G = (int*)malloc(sizeof(int)*(Matomnum+MatomnumF+1));
  S_M2G = (int*)malloc(sizeof(int)*(Matomnum+MatomnumF+MatomnumS+1));
  alloc_first[13] = 0;

  /************************************************
           setting of F_G2M and S_G2M

    F_G2M and S_G2M give a conversion from the
    global atom number to the intermediate atom number
    for atoms sent from ID in the size of
    F_Rcv_Num[ID] and F_Rcv_Num[ID] + S_Rcv_Num[ID],
    respectively. 
  *************************************************/
  
  /* initialization of F_G2M and S_G2M */
  
  for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
    F_G2M[Gc_AN] = -1;
    S_G2M[Gc_AN] = -1;
  } 
  
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    F_G2M[Gc_AN] = Mc_AN;
    S_G2M[Gc_AN] = Mc_AN;
    F_M2G[Mc_AN] = Gc_AN;
    S_M2G[Mc_AN] = Gc_AN;
  }

  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid && F_Rcv_Num[ID]!=0){
      for (Num=0; Num<F_Rcv_Num[ID]; Num++){
        Gc_AN = Rcv_GAN[ID][Num];
        F_G2M[Gc_AN] = F_TopMAN[ID] + Num;
        F_M2G[F_TopMAN[ID] + Num] = Gc_AN;  
      }
    }
  }

  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid && (F_Rcv_Num[ID]!=0 || S_Rcv_Num[ID]!=0)){
      for (Num=0; Num<(F_Rcv_Num[ID]+S_Rcv_Num[ID]); Num++){

        Gc_AN = Rcv_GAN[ID][Num];
        S_G2M[Gc_AN] = S_TopMAN[ID] + Num;
        S_M2G[S_TopMAN[ID] + Num] = Gc_AN;
      }
    }
  }

  /************************************************
      analysis of data structures  
      for MPI communication of DS_VNA
  *************************************************/

  Num_Pro_Snd = (int*)malloc(sizeof(int)*numprocs);

  for (ID=0; ID<numprocs; ID++)  Num_Pro_Snd[ID] = 0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (Lh_AN=0; Lh_AN<=FNAN[Gc_AN]; Lh_AN++){
      Gh_AN = natn[Gc_AN][Lh_AN];
      ID1 = G2ID[Gh_AN];
      Num_Pro_Snd[ID1]++;
    }
  }

  if (alloc_first[22]==0){

    for (ID=0; ID<numprocs; ID++){
      free(Pro_Snd_GAtom[ID]);
    }
    free(Pro_Snd_GAtom);

    for (ID=0; ID<numprocs; ID++){
      free(Pro_Snd_MAtom[ID]);
    }
    free(Pro_Snd_MAtom);

    for (ID=0; ID<numprocs; ID++){
      free(Pro_Snd_LAtom[ID]);
    }
    free(Pro_Snd_LAtom);

    for (ID=0; ID<numprocs; ID++){
      free(Pro_Snd_LAtom2[ID]);
    }
    free(Pro_Snd_LAtom2);
  }

  Pro_Snd_GAtom = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Pro_Snd_GAtom[ID] = (int*)malloc(sizeof(int)*(Num_Pro_Snd[ID]+1));
    for (i=0; i<(Num_Pro_Snd[ID]+1); i++){
      Pro_Snd_GAtom[ID][i] = 0;
    }   
  }

  Pro_Snd_MAtom = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Pro_Snd_MAtom[ID] = (int*)malloc(sizeof(int)*(Num_Pro_Snd[ID]+1));
    for (i=0; i<(Num_Pro_Snd[ID]+1); i++){
      Pro_Snd_MAtom[ID][i] = 0;
    }   
  }

  Pro_Snd_LAtom = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Pro_Snd_LAtom[ID] = (int*)malloc(sizeof(int)*(Num_Pro_Snd[ID]+1));
    for (i=0; i<(Num_Pro_Snd[ID]+1); i++){
      Pro_Snd_LAtom[ID][i] = 0;
    }   
  }

  Pro_Snd_LAtom2 = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Pro_Snd_LAtom2[ID] = (int*)malloc(sizeof(int)*(Num_Pro_Snd[ID]+1));
    for (i=0; i<(Num_Pro_Snd[ID]+1); i++){
      Pro_Snd_LAtom2[ID][i] = 0;
    }   
  }

  alloc_first[22] = 0;

  for (ID=0; ID<numprocs; ID++)  Num_Pro_Snd[ID] = 0;
  
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    for (Lh_AN=0; Lh_AN<=FNAN[Gc_AN]; Lh_AN++){
    
      Gh_AN = natn[Gc_AN][Lh_AN];
      ID1 = G2ID[Gh_AN];
    
      Pro_Snd_GAtom[ID1][Num_Pro_Snd[ID1]] = Gh_AN;
      Pro_Snd_MAtom[ID1][Num_Pro_Snd[ID1]] = Mc_AN;
      Pro_Snd_LAtom[ID1][Num_Pro_Snd[ID1]] = Lh_AN;
      
      Num_Pro_Snd[ID1]++;
    }
  }

  /* quick sorting of Pro_Snd_GAtom */ 

  for (ID=0; ID<numprocs; ID++){ 
    qsort_int3(Num_Pro_Snd[ID], Pro_Snd_GAtom[ID], Pro_Snd_MAtom[ID], Pro_Snd_LAtom[ID]);
  }

  /* setting of Pro_Snd_LAtom2 */

  for (ID=0; ID<numprocs; ID++){ 

    for (i=0; i<Num_Pro_Snd[ID]; i++){

       Gh_AN = Pro_Snd_GAtom[ID][i];
       Mc_AN = Pro_Snd_MAtom[ID][i];
       Lh_AN = Pro_Snd_LAtom[ID][i];

       Gc_AN = M2G[Mc_AN];
       Rn = ncn[Gc_AN][Lh_AN];

       m1 =-atv_ijk[Rn][1];
       m2 =-atv_ijk[Rn][2];
       m3 =-atv_ijk[Rn][3];

       j = 0;
       po = 0;

       do{ 

         Gj_AN = natn[Gh_AN][j];
         Rn2 = ncn[Gh_AN][j];

         n1 = atv_ijk[Rn2][1];
         n2 = atv_ijk[Rn2][2];
         n3 = atv_ijk[Rn2][3];
              
         if (m1==n1 && m2==n2 && m3==n3 &&  Gj_AN==Gc_AN){
           Pro_Snd_LAtom2[ID][i] = j;
	   po = 1;
         }

         j++;

       } while (po==0);

    }
  }

  /*********************************
   freeing of arrays:
  *********************************/

  free(Num_Pro_Snd);

  for (ID=0; ID<numprocs; ID++){
    free(Snd_SGAN[ID]);
  }
  free(Snd_SGAN);

  for (ID=0; ID<numprocs; ID++){
    free(Snd_FGAN[ID]);
  }
  free(Snd_FGAN);

  for (ID=0; ID<numprocs; ID++){
    free(Rcv_SGAN[ID]);
  }
  free(Rcv_SGAN);

  for (ID=0; ID<numprocs; ID++){
    free(Rcv_FGAN[ID]);
  }
  free(Rcv_FGAN);

  free(flag_DoubleCounting);
} 







#pragma optimization_level 1
void Construct_MPI_Data_Structure_Grid()
{
  static int firsttime=1;
  int i,j,k,Mc_AN,Gc_AN,wan,n1,n2,n3;
  int min_n1,max_n1,min_n2,max_n2,N3[4];
  unsigned long long int AN,BN,CN,DN;
  unsigned long long int B_AB2,Bs,Be;
/* modified by mari 05.12.2014 */
  unsigned long long int BN_AB,BN_CB,BN_CA,GN_B_AB,BN_C;
  unsigned long long int GN,GNs,GR,n2D,N2D;
/* modified by mari 05.12.2014 */
  unsigned long long int GN_AB,GN_CB,GN_CA,GN_C;
  int size_Index_Snd_Grid_A2B;
  int size_Index_Rcv_Grid_A2B;
  int size_Index_Snd_Grid_B2C;
  int size_Index_Rcv_Grid_B2C;
  int size_Index_Snd_Grid_B2D;
  int size_Index_Rcv_Grid_B2D;
  int size_Index_Snd_Grid_B_AB2CA;
  int size_Index_Rcv_Grid_B_AB2CA;
/* added by mari 05.12.2014 */
  int size_Index_Snd_Grid_B_AB2C;
  int size_Index_Rcv_Grid_B_AB2C;
  int size_Index_Snd_Grid_B_CA2CB;
  int size_Index_Rcv_Grid_B_CA2CB;
  int myid,numprocs,ID,IDS,IDR,tag=999;
  double Vec0,Vec1,coef,MinV,MaxV,rcut;
  double Cxyz[4],b[4],c[4],v[4];

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /******************************************************
    find the smallest parallelepipedon which contains 
    atoms allocated to my ID under consideration of 
    cutoff radii of basis functions. 
  ******************************************************/
 
  for (k=1; k<=3; k++){

    if      (k==1){ i = 2; j = 3; }
    else if (k==2){ i = 3; j = 1; }
    else if (k==3){ i = 1; j = 2; }

    b[1] = tv[i][1];
    b[2] = tv[i][2];
    b[3] = tv[i][3];

    c[1] = tv[j][1];
    c[2] = tv[j][2];
    c[3] = tv[j][3];

    Cross_Product(b,c,v);
    coef = 1.0/sqrt(fabs( Dot_Product(v,v) ));
    
    v[1] = coef*v[1];
    v[2] = coef*v[2];
    v[3] = coef*v[3];

    MinV =  1.0e+10;
    MaxV = -1.0e+10;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      wan  = WhatSpecies[Gc_AN];
      rcut = Spe_Atom_Cut1[wan];

      Cxyz[1] = Gxyz[Gc_AN][1] + rcut*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[Gc_AN][2] + rcut*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[Gc_AN][3] + rcut*v[3] - Grid_Origin[3];

      Vec0 = Dot_Product(Cxyz,rgtv[k])*0.5/PI;

      Cxyz[1] = Gxyz[Gc_AN][1] - rcut*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[Gc_AN][2] - rcut*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[Gc_AN][3] - rcut*v[3] - Grid_Origin[3];

      Vec1 = Dot_Product(Cxyz,rgtv[k])*0.5/PI;

      if (Vec0<MinV) MinV = Vec0;
      if (Vec1<MinV) MinV = Vec1;
      if (MaxV<Vec0) MaxV = Vec0;   
      if (MaxV<Vec1) MaxV = Vec1;
    }

    Min_Grid_Index[k] = (int)MinV;  /* buffer for GGA */
    Max_Grid_Index[k] = (int)MaxV;  /* buffer for GGA */

  } /* k */

  /******************************************************
    find the smallest parallelepipedon which contains 
    grids in the partition B_AB
    
    The parallelepipedon definess the partition D.
  ******************************************************/

  N2D = Ngrid1*Ngrid2;
  Bs = (myid*N2D+numprocs-1)/numprocs;
  Be = ((myid+1)*N2D+numprocs-1)/numprocs;
  
  min_n1 = 1000000;
  max_n1 =-1000000;
  min_n2 = 1000000;
  max_n2 =-1000000;

  for (B_AB2=Bs; B_AB2<Be; B_AB2++){

    n1 = B_AB2/Ngrid2;
    n2 = B_AB2 - n1*Ngrid2;

    if (n1<min_n1) min_n1 = n1;
    if (max_n1<n1) max_n1 = n1;
    if (n2<min_n2) min_n2 = n2;
    if (max_n2<n2) max_n2 = n2;
  }

  Min_Grid_Index_D[1] = min_n1 - 2;
  Max_Grid_Index_D[1] = max_n1 + 2;
  Min_Grid_Index_D[2] = min_n2 - 2;
  Max_Grid_Index_D[2] = max_n2 + 2;
  Min_Grid_Index_D[3] = -2;
  Max_Grid_Index_D[3] = (Ngrid3-1) + 2;

  /****************************************************************
      The partitions A to B

      construct the data structure for transfering rho_i from 
      the partitions A to B when rho is calculated 
      in the partition B using rho_i 
  ****************************************************************/

  /* find Num_Snd_Grid_A2B[ID] */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;

  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan  = WhatSpecies[Gc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];

      /* get process ID and increment Num_Snd_Grid_A2B */

      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = n2D*numprocs/N2D;
      Num_Snd_Grid_A2B[ID]++;
    }
  }    

  /* MPI: Num_Snd_Grid_A2B */  

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      tag = 999;
      MPI_Isend(&Num_Snd_Grid_A2B[IDS], 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      MPI_Recv(&Num_Rcv_Grid_A2B[IDR], 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv_Grid_A2B[IDR] = Num_Snd_Grid_A2B[IDS];
    }
  }

  /* allocation of arrays */  

  if (alloc_first[26]==0){

    for (ID=0; ID<numprocs; ID++){
      free(Index_Snd_Grid_A2B[ID]);
    }  
    free(Index_Snd_Grid_A2B);
  }

  size_Index_Snd_Grid_A2B = 0;
  Index_Snd_Grid_A2B = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Snd_Grid_A2B[ID] = (int*)malloc(sizeof(int)*3*Num_Snd_Grid_A2B[ID]);
    size_Index_Snd_Grid_A2B += 3*Num_Snd_Grid_A2B[ID];
  } 

  if (alloc_first[26]==0){

    for (ID=0; ID<numprocs; ID++){
      free(Index_Rcv_Grid_A2B[ID]);
    }  
    free(Index_Rcv_Grid_A2B);
  }

  size_Index_Rcv_Grid_A2B = 0; 
  Index_Rcv_Grid_A2B = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Rcv_Grid_A2B[ID] = (int*)malloc(sizeof(int)*3*Num_Rcv_Grid_A2B[ID]);
    size_Index_Rcv_Grid_A2B += 3*Num_Rcv_Grid_A2B[ID]; 
  }  

  alloc_first[26] = 0;

  /* construct Index_Snd_Grid_A2B */  

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;

  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan  = WhatSpecies[Gc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GR = CellListAtom[Mc_AN][AN];

      /* get process ID and grid index (BN) for the partition B */

      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = n2D*numprocs/N2D;
      GN_B_AB = N3[1]*Ngrid2*Ngrid3 + N3[2]*Ngrid3 + N3[3];
      BN = GN_B_AB - ((ID*N2D+numprocs-1)/numprocs)*Ngrid3;

      Index_Snd_Grid_A2B[ID][3*Num_Snd_Grid_A2B[ID]+0] = BN; 
      Index_Snd_Grid_A2B[ID][3*Num_Snd_Grid_A2B[ID]+1] = Gc_AN; 
      Index_Snd_Grid_A2B[ID][3*Num_Snd_Grid_A2B[ID]+2] = GR; 

      Num_Snd_Grid_A2B[ID]++;
    }
  }    

  /* MPI: Index_Snd_Grid_A2B */
      
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      tag = 999;
      if (Num_Snd_Grid_A2B[IDS]!=0){
	MPI_Isend( &Index_Snd_Grid_A2B[IDS][0], 3*Num_Snd_Grid_A2B[IDS], 
		   MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Rcv_Grid_A2B[IDR]!=0){
	MPI_Recv( &Index_Rcv_Grid_A2B[IDR][0], 3*Num_Rcv_Grid_A2B[IDR], 
		  MPI_INT, IDR, tag, mpi_comm_level1, &stat);
      }
      if (Num_Snd_Grid_A2B[IDS]!=0)  MPI_Wait(&request,&stat);
    }

    else{
      for (i=0; i<3*Num_Rcv_Grid_A2B[IDR]; i++){
        Index_Rcv_Grid_A2B[IDR][i] = Index_Snd_Grid_A2B[IDS][i]; 
      } 
    }
  }

  /* count NN_A2B_S and NN_A2B_R */
  
  NN_A2B_S = 0;
  NN_A2B_R = 0;

  for (ID=0; ID<numprocs; ID++){
    if (Num_Snd_Grid_A2B[ID]!=0) NN_A2B_S++;
    if (Num_Rcv_Grid_A2B[ID]!=0) NN_A2B_R++;
  }

  /****************************************************************
      The partitions B to C

      construct the data structure for transfering rho from 
      the partitions B to C when rho is constructed in the 
      partition C using rho stored in the partition B.
  ****************************************************************/

  /* find Num_Rcv_Grid_B2C[ID] */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_Grid_B2C[ID] = 0;

  N2D = Ngrid1*Ngrid2;

  for (n1=Min_Grid_Index[1]; n1<=Max_Grid_Index[1]; n1++){
    for (n2=Min_Grid_Index[2]; n2<=Max_Grid_Index[2]; n2++){
      for (n3=Min_Grid_Index[3]; n3<=Max_Grid_Index[3]; n3++){

         Find_CGrids(0,n1,n2,n3,Cxyz,N3);
         n2D = N3[1]*Ngrid2 + N3[2];
         ID = n2D*numprocs/N2D;
         Num_Rcv_Grid_B2C[ID]++;
      }
    }
  }

  /* MPI: Num_Rcv_Grid_B2C */  

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;
      MPI_Isend(&Num_Rcv_Grid_B2C[IDS], 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      MPI_Recv(&Num_Snd_Grid_B2C[IDR], 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Snd_Grid_B2C[IDR] = Num_Rcv_Grid_B2C[IDS];
    }
  }

  /* find Max_Num_Snd_Grid_B2C and Max_Num_Rcv_Grid_B2C */

  Max_Num_Snd_Grid_B2C = 0;
  Max_Num_Rcv_Grid_B2C = 0;
  for (ID=0; ID<numprocs; ID++){
    if ( Max_Num_Snd_Grid_B2C<Num_Snd_Grid_B2C[ID] ) Max_Num_Snd_Grid_B2C = Num_Snd_Grid_B2C[ID];
    if ( Max_Num_Rcv_Grid_B2C<Num_Rcv_Grid_B2C[ID] ) Max_Num_Rcv_Grid_B2C = Num_Rcv_Grid_B2C[ID];
  }

  /* allocation of arrays */  

  if (alloc_first[27]==0){
    for (ID=0; ID<numprocs; ID++){
      free(Index_Snd_Grid_B2C[ID]);
    }  
    free(Index_Snd_Grid_B2C);

    for (ID=0; ID<numprocs; ID++){
      free(Index_Rcv_Grid_B2C[ID]);
    }  
    free(Index_Rcv_Grid_B2C);
  }

  size_Index_Snd_Grid_B2C = 0;
  Index_Snd_Grid_B2C = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Snd_Grid_B2C[ID] = (int*)malloc(sizeof(int)*Num_Snd_Grid_B2C[ID]);
    size_Index_Snd_Grid_B2C += Num_Snd_Grid_B2C[ID];
  }  

  size_Index_Rcv_Grid_B2C = 0;  
  Index_Rcv_Grid_B2C = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Rcv_Grid_B2C[ID] = (int*)malloc(sizeof(int)*Num_Rcv_Grid_B2C[ID]);
    size_Index_Rcv_Grid_B2C += Num_Rcv_Grid_B2C[ID];
  }  

  /* construct Index_Rcv_Grid_B2C
     first BN is stored to Index_Rcv_Grid_B2C.  
     and after MPI communication, CN is stored 
     to Index_Rcv_Grid_B2C.  
     Also note that Index_Snd_Grid_B2C stores BN
  */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_Grid_B2C[ID] = 0;

  N2D = Ngrid1*Ngrid2;

  for (n1=Min_Grid_Index[1]; n1<=Max_Grid_Index[1]; n1++){
    for (n2=Min_Grid_Index[2]; n2<=Max_Grid_Index[2]; n2++){
      for (n3=Min_Grid_Index[3]; n3<=Max_Grid_Index[3]; n3++){

         Find_CGrids(0,n1,n2,n3,Cxyz,N3);
         n2D = N3[1]*Ngrid2 + N3[2];
         ID = n2D*numprocs/N2D;
         GN = N3[1]*Ngrid2*Ngrid3 + N3[2]*Ngrid3 + N3[3];
         BN = GN - ((ID*N2D+numprocs-1)/numprocs)*Ngrid3;
         Index_Rcv_Grid_B2C[ID][Num_Rcv_Grid_B2C[ID]] = BN; 
         Num_Rcv_Grid_B2C[ID]++;
      }
    }
  }

  /* MPI: Index_Rcv_Grid_B2C */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;
      if (Num_Rcv_Grid_B2C[IDS]!=0){
	MPI_Isend( &Index_Rcv_Grid_B2C[IDS][0], Num_Rcv_Grid_B2C[IDS],
		   MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Snd_Grid_B2C[IDR]!=0){ 
	MPI_Recv( &Index_Snd_Grid_B2C[IDR][0], Num_Snd_Grid_B2C[IDR], 
		  MPI_INT, IDR, tag, mpi_comm_level1, &stat);
      }
      if (Num_Rcv_Grid_B2C[IDS]!=0)  MPI_Wait(&request,&stat);
    }
    else{
      for (i=0; i<Num_Snd_Grid_B2C[IDR]; i++){
        Index_Snd_Grid_B2C[IDR][i] = Index_Rcv_Grid_B2C[IDS][i];
      }
    }
  }

  /* reconstruct Index_Rcv_Grid_B2C
     Index_Rcv_Grid_B2C stores CN.
  */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_Grid_B2C[ID] = 0;

  N2D = Ngrid1*Ngrid2;
  CN = 0;
  for (n1=Min_Grid_Index[1]; n1<=Max_Grid_Index[1]; n1++){
    for (n2=Min_Grid_Index[2]; n2<=Max_Grid_Index[2]; n2++){
      for (n3=Min_Grid_Index[3]; n3<=Max_Grid_Index[3]; n3++){

         Find_CGrids(0,n1,n2,n3,Cxyz,N3);
         n2D = N3[1]*Ngrid2 + N3[2];
         ID = n2D*numprocs/N2D;
         Index_Rcv_Grid_B2C[ID][Num_Rcv_Grid_B2C[ID]] = CN; 
         Num_Rcv_Grid_B2C[ID]++;

         CN++;
      }
    }
  }

  /* find NN_B2C_S and NN_B2C_R
     and set ID_NN_B2C_S
             ID_NN_B2C_R
             GP_B2C_S
             GP_B2C_R 
  */ 

  NN_B2C_S = 0;
  NN_B2C_R = 0;

  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (Num_Snd_Grid_B2C[IDS]!=0) NN_B2C_S++;
    if (Num_Rcv_Grid_B2C[IDR]!=0) NN_B2C_R++;
  }

  if (alloc_first[27]==0){
    free(ID_NN_B2C_S);
    free(ID_NN_B2C_R);
    free(GP_B2C_S);
    free(GP_B2C_R);
  }

  ID_NN_B2C_S = (int*)malloc(sizeof(int)*NN_B2C_S);
  ID_NN_B2C_R = (int*)malloc(sizeof(int)*NN_B2C_R);
  GP_B2C_S = (int*)malloc(sizeof(int)*(NN_B2C_S+1));
  GP_B2C_R = (int*)malloc(sizeof(int)*(NN_B2C_R+1));

  alloc_first[27] = 0;

  NN_B2C_S = 0;
  NN_B2C_R = 0;
  GP_B2C_S[0] = 0;
  GP_B2C_R[0] = 0;

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_B2C[IDS]!=0){
      ID_NN_B2C_S[NN_B2C_S] = IDS;
      NN_B2C_S++;
      GP_B2C_S[NN_B2C_S] = GP_B2C_S[NN_B2C_S-1] + Num_Snd_Grid_B2C[IDS];
    }

    if (Num_Rcv_Grid_B2C[IDR]!=0){
      ID_NN_B2C_R[NN_B2C_R] = IDR;
      NN_B2C_R++;
      GP_B2C_R[NN_B2C_R] = GP_B2C_R[NN_B2C_R-1] + Num_Rcv_Grid_B2C[IDR];
    }
  }

  /* set the number of grids allocated to the partitions B and C for myid */

  N2D = Ngrid1*Ngrid2;
  My_NumGridB_AB =  (((myid+1)*N2D+numprocs-1)/numprocs)*Ngrid3
                  - ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;
  My_NumGridC = CN;

  /****************************************************************
      The partitions B to D

      construct the data structure for transfering rho from 
      the partitions B to D for the case that rho is constructed
      in the partition D using rho stored in the partition B.
  ****************************************************************/

  /* find Num_Rcv_Grid_B2D[ID] */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_Grid_B2D[ID] = 0;

  N2D = Ngrid1*Ngrid2;

  for (n1=Min_Grid_Index_D[1]; n1<=Max_Grid_Index_D[1]; n1++){
    for (n2=Min_Grid_Index_D[2]; n2<=Max_Grid_Index_D[2]; n2++){
      for (n3=Min_Grid_Index_D[3]; n3<=Max_Grid_Index_D[3]; n3++){

         Find_CGrids(0,n1,n2,n3,Cxyz,N3);
         n2D = N3[1]*Ngrid2 + N3[2];
         ID = n2D*numprocs/N2D;
         Num_Rcv_Grid_B2D[ID]++;
      }
    }
  }

  /* MPI: Num_Rcv_Grid_B2D */  

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
    MPI_Isend(&Num_Rcv_Grid_B2D[IDS], 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    MPI_Recv(&Num_Snd_Grid_B2D[IDR], 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    MPI_Wait(&request,&stat);
  }

  /* find Max_Num_Snd_Grid_B2D and Max_Num_Rcv_Grid_B2D */

  Max_Num_Snd_Grid_B2D = 0;
  Max_Num_Rcv_Grid_B2D = 0;
  for (ID=0; ID<numprocs; ID++){
    if ( Max_Num_Snd_Grid_B2D<Num_Snd_Grid_B2D[ID] ) Max_Num_Snd_Grid_B2D = Num_Snd_Grid_B2D[ID];
    if ( Max_Num_Rcv_Grid_B2D<Num_Rcv_Grid_B2D[ID] ) Max_Num_Rcv_Grid_B2D = Num_Rcv_Grid_B2D[ID];
  }

  /* allocation of arrays */  

  if (alloc_first[30]==0){
    for (ID=0; ID<numprocs; ID++){
      free(Index_Snd_Grid_B2D[ID]);
    }  
    free(Index_Snd_Grid_B2D);

    for (ID=0; ID<numprocs; ID++){
      free(Index_Rcv_Grid_B2D[ID]);
    }  
    free(Index_Rcv_Grid_B2D);

    free(ID_NN_B2D_S);
    free(ID_NN_B2D_R);
    free(GP_B2D_S);
    free(GP_B2D_R);
  }

  size_Index_Snd_Grid_B2D = 0;
  Index_Snd_Grid_B2D = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Snd_Grid_B2D[ID] = (int*)malloc(sizeof(int)*Num_Snd_Grid_B2D[ID]);
    size_Index_Snd_Grid_B2D += Num_Snd_Grid_B2D[ID];
  }  

  size_Index_Rcv_Grid_B2D = 0;  
  Index_Rcv_Grid_B2D = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Rcv_Grid_B2D[ID] = (int*)malloc(sizeof(int)*Num_Rcv_Grid_B2D[ID]);
    size_Index_Rcv_Grid_B2D += Num_Rcv_Grid_B2D[ID];
  }  
    
  alloc_first[30] = 0;

  /* construct Index_Rcv_Grid_B2D
     first BN is stored to Index_Rcv_Grid_B2D.  
     and after MPI communication, DN is stored 
     to Index_Rcv_Grid_B2D.  
     Also note that Index_Snd_Grid_B2D stores BN
  */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_Grid_B2D[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;
  
  for (n1=Min_Grid_Index_D[1]; n1<=Max_Grid_Index_D[1]; n1++){
    for (n2=Min_Grid_Index_D[2]; n2<=Max_Grid_Index_D[2]; n2++){
      for (n3=Min_Grid_Index_D[3]; n3<=Max_Grid_Index_D[3]; n3++){
  
         Find_CGrids(0,n1,n2,n3,Cxyz,N3);
         n2D = N3[1]*Ngrid2 + N3[2];
         ID = n2D*numprocs/N2D;
         GN = N3[1]*Ngrid2*Ngrid3 + N3[2]*Ngrid3 + N3[3];
         BN = GN - ((ID*N2D+numprocs-1)/numprocs)*Ngrid3;
         Index_Rcv_Grid_B2D[ID][Num_Rcv_Grid_B2D[ID]] = BN; 
         Num_Rcv_Grid_B2D[ID]++;
      }
    }
  }

  /* MPI: Index_Rcv_Grid_B2D */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;

    if (Num_Rcv_Grid_B2D[IDS]!=0){
      MPI_Isend( &Index_Rcv_Grid_B2D[IDS][0], Num_Rcv_Grid_B2D[IDS],
                 MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }

    if (Num_Snd_Grid_B2D[IDR]!=0){ 
      MPI_Recv( &Index_Snd_Grid_B2D[IDR][0], Num_Snd_Grid_B2D[IDR], 
                MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    }

    if (Num_Rcv_Grid_B2D[IDS]!=0)  MPI_Wait(&request,&stat);
  }

  /* reconstruct Index_Rcv_Grid_B2D
     Index_Rcv_Grid_B2D stores DN.
  */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_Grid_B2D[ID] = 0;

  N2D = Ngrid1*Ngrid2;
  DN = 0;
  for (n1=Min_Grid_Index_D[1]; n1<=Max_Grid_Index_D[1]; n1++){
    for (n2=Min_Grid_Index_D[2]; n2<=Max_Grid_Index_D[2]; n2++){
      for (n3=Min_Grid_Index_D[3]; n3<=Max_Grid_Index_D[3]; n3++){

         Find_CGrids(0,n1,n2,n3,Cxyz,N3);
         n2D = N3[1]*Ngrid2 + N3[2];
         ID = n2D*numprocs/N2D;
         Index_Rcv_Grid_B2D[ID][Num_Rcv_Grid_B2D[ID]] = DN; 
         Num_Rcv_Grid_B2D[ID]++;

         DN++;
      }
    }
  }

  /* find NN_B2C_S and NN_B2C_R
     and set ID_NN_B2C_S
             ID_NN_B2C_R
             GP_B2C_S
             GP_B2C_R 
  */ 

  NN_B2D_S = 0;
  NN_B2D_R = 0;

  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (Num_Snd_Grid_B2D[IDS]!=0) NN_B2D_S++;
    if (Num_Rcv_Grid_B2D[IDR]!=0) NN_B2D_R++;
  }

  ID_NN_B2D_S = (int*)malloc(sizeof(int)*NN_B2D_S);
  ID_NN_B2D_R = (int*)malloc(sizeof(int)*NN_B2D_R);
  GP_B2D_S = (int*)malloc(sizeof(int)*(NN_B2D_S+1));
  GP_B2D_R = (int*)malloc(sizeof(int)*(NN_B2D_R+1));

  NN_B2D_S = 0;
  NN_B2D_R = 0;
  GP_B2D_S[0] = 0;
  GP_B2D_R[0] = 0;

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_B2D[IDS]!=0){
      ID_NN_B2D_S[NN_B2D_S] = IDS;
      NN_B2D_S++;
      GP_B2D_S[NN_B2D_S] = GP_B2D_S[NN_B2D_S-1] + Num_Snd_Grid_B2D[IDS];
    }

    if (Num_Rcv_Grid_B2D[IDR]!=0){
      ID_NN_B2D_R[NN_B2D_R] = IDR;
      NN_B2D_R++;
      GP_B2D_R[NN_B2D_R] = GP_B2D_R[NN_B2D_R-1] + Num_Rcv_Grid_B2D[IDR];
    }
  }

  /* set the number of grids allocated to the D for myid */

  My_NumGridD = DN;

  /****************************************************************
      AB to CA in the partition B for MPI communication in FFT 
  ****************************************************************/

  /* set GNs */

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  /* find Num_Snd_Grid_B_AB2CA */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_AB2CA[ID] = 0;

  N2D = Ngrid3*Ngrid1; /* for CA */

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN_AB = BN_AB + GNs;

    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
    n2D = n3*Ngrid1 + n1;
    ID = n2D*numprocs/N2D;
    Num_Snd_Grid_B_AB2CA[ID]++;
  }

  /* MPI: Num_Snd_Grid_B_AB2CA */  

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
    MPI_Isend(&Num_Snd_Grid_B_AB2CA[IDS], 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    MPI_Recv(&Num_Rcv_Grid_B_AB2CA[IDR], 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    MPI_Wait(&request,&stat);
  }

  /* allocation of arrays */  

  if (alloc_first[28]==0){

    for (ID=0; ID<numprocs; ID++){
      free(Index_Snd_Grid_B_AB2CA[ID]);
    }  
    free(Index_Snd_Grid_B_AB2CA);
  
    for (ID=0; ID<numprocs; ID++){
      free(Index_Rcv_Grid_B_AB2CA[ID]);
    }  
    free(Index_Rcv_Grid_B_AB2CA);

    free(ID_NN_B_AB2CA_S);
    free(ID_NN_B_AB2CA_R);
    free(GP_B_AB2CA_S);
    free(GP_B_AB2CA_R);
  }

  size_Index_Snd_Grid_B_AB2CA = 0;
  Index_Snd_Grid_B_AB2CA = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Snd_Grid_B_AB2CA[ID] = (int*)malloc(sizeof(int)*Num_Snd_Grid_B_AB2CA[ID]);
    size_Index_Snd_Grid_B_AB2CA += Num_Snd_Grid_B_AB2CA[ID];
  }  
  
  size_Index_Rcv_Grid_B_AB2CA = 0; 
  Index_Rcv_Grid_B_AB2CA = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Rcv_Grid_B_AB2CA[ID] = (int*)malloc(sizeof(int)*Num_Rcv_Grid_B_AB2CA[ID]);
    size_Index_Rcv_Grid_B_AB2CA += Num_Rcv_Grid_B_AB2CA[ID]; 
  }  

  alloc_first[28] = 0;

  /* construct Index_Snd_Grid_B_AB2CA */  

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_AB2CA[ID] = 0;

  N2D = Ngrid3*Ngrid1;  /* for CA */

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN_AB = BN_AB + GNs;

    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
    n2D = n3*Ngrid1 + n1;
    ID = n2D*numprocs/N2D;
    GN_CA = n3*Ngrid1*Ngrid2 + n1*Ngrid2 + n2;
    BN_CA = GN_CA - ((ID*N2D+numprocs-1)/numprocs)*Ngrid2;
    Index_Snd_Grid_B_AB2CA[ID][Num_Snd_Grid_B_AB2CA[ID]] = BN_CA;
    Num_Snd_Grid_B_AB2CA[ID]++;
  }

  /* MPI: Index_Snd_Grid_B_AB2CA */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
   
    if (Num_Snd_Grid_B_AB2CA[IDS]!=0){
      MPI_Isend( &Index_Snd_Grid_B_AB2CA[IDS][0], Num_Snd_Grid_B_AB2CA[IDS], 
		 MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }

    if (Num_Rcv_Grid_B_AB2CA[IDR]!=0){
      MPI_Recv( &Index_Rcv_Grid_B_AB2CA[IDR][0], Num_Rcv_Grid_B_AB2CA[IDR], 
		MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    }

    if (Num_Snd_Grid_B_AB2CA[IDS]!=0)  MPI_Wait(&request,&stat);
  }

  /* reset: Index_Snd_Grid_B_AB2CA

  Index_Snd_Grid_B_AB2CA:  BN_AB
  Index_Rcv_Grid_B_AB2CA:  BN_CA
  */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_AB2CA[ID] = 0;

  N2D = Ngrid3*Ngrid1; /* for CA */

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN_AB = BN_AB + GNs;
    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
    n2D = n3*Ngrid1 + n1;
    ID = n2D*numprocs/N2D;
    Index_Snd_Grid_B_AB2CA[ID][Num_Snd_Grid_B_AB2CA[ID]] = BN_AB;
    Num_Snd_Grid_B_AB2CA[ID]++;
  }

  /* find the maximum Num_Snd_Grid_B_AB2CA and Num_Rcv_Grid_B_AB2CA */

  Max_Num_Snd_Grid_B_AB2CA = 0;
  Max_Num_Rcv_Grid_B_AB2CA = 0;

  for (ID=0; ID<numprocs; ID++){
    if (Max_Num_Snd_Grid_B_AB2CA<Num_Snd_Grid_B_AB2CA[ID]){ 
      Max_Num_Snd_Grid_B_AB2CA = Num_Snd_Grid_B_AB2CA[ID];
    }

    if (Max_Num_Rcv_Grid_B_AB2CA<Num_Rcv_Grid_B_AB2CA[ID]){ 
      Max_Num_Rcv_Grid_B_AB2CA = Num_Rcv_Grid_B_AB2CA[ID];
    }
  }

  /* find NN_B_AB2CA_S and NN_B_AB2CA_R 
     and set ID_NN_B_AB2CA_S, 
             ID_NN_B_AB2CA_R,
             GP_B_AB2CA_S,
             GP_B_AB2CA_R
  */

  NN_B_AB2CA_S = 0;
  NN_B_AB2CA_R = 0;
 
  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (Num_Snd_Grid_B_AB2CA[IDS]!=0) NN_B_AB2CA_S++;
    if (Num_Rcv_Grid_B_AB2CA[IDR]!=0) NN_B_AB2CA_R++;
  }

  ID_NN_B_AB2CA_S = (int*)malloc(sizeof(int)*NN_B_AB2CA_S);
  ID_NN_B_AB2CA_R = (int*)malloc(sizeof(int)*NN_B_AB2CA_R);
  GP_B_AB2CA_S = (int*)malloc(sizeof(int)*(NN_B_AB2CA_S+1));
  GP_B_AB2CA_R = (int*)malloc(sizeof(int)*(NN_B_AB2CA_R+1));

  NN_B_AB2CA_S = 0;
  NN_B_AB2CA_R = 0;
  GP_B_AB2CA_S[0] = 0;
  GP_B_AB2CA_R[0] = 0;

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_B_AB2CA[IDS]!=0){
      ID_NN_B_AB2CA_S[NN_B_AB2CA_S] = IDS;
      NN_B_AB2CA_S++;
      GP_B_AB2CA_S[NN_B_AB2CA_S] = GP_B_AB2CA_S[NN_B_AB2CA_S-1] + Num_Snd_Grid_B_AB2CA[IDS];
    }

    if (Num_Rcv_Grid_B_AB2CA[IDR]!=0){
      ID_NN_B_AB2CA_R[NN_B_AB2CA_R] = IDR;
      NN_B_AB2CA_R++;
      GP_B_AB2CA_R[NN_B_AB2CA_R] = GP_B_AB2CA_R[NN_B_AB2CA_R-1] + Num_Rcv_Grid_B_AB2CA[IDR];
    }
  }

  /* set My_NumGridB_CA */

  N2D = Ngrid3*Ngrid1;
  My_NumGridB_CA = (((myid+1)*N2D+numprocs-1)/numprocs)*Ngrid2
                  - ((myid*N2D+numprocs-1)/numprocs)*Ngrid2;

  /* beginning added by mari 12.05.2014 */
  /****************************************************************
      AB to C in the partition B for MPI communication in FFT 
  ****************************************************************/

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_AB2C[ID] = 0;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    GN_AB = BN_AB + GNs;
    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
    ID = n3*numprocs/Ngrid3;
    Num_Snd_Grid_B_AB2C[ID]++;
  }

  /* MPI: Num_Snd_Grid_B_AB2C */  

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
    MPI_Isend(&Num_Snd_Grid_B_AB2C[IDS], 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    MPI_Recv(&Num_Rcv_Grid_B_AB2C[IDR], 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    MPI_Wait(&request,&stat);
  }

  /* allocation of arrays */  

  if (alloc_first[31]==0){

    for (ID=0; ID<numprocs; ID++){
      free(Index_Snd_Grid_B_AB2C[ID]);
    }  
    free(Index_Snd_Grid_B_AB2C);
  
    for (ID=0; ID<numprocs; ID++){
      free(Index_Rcv_Grid_B_AB2C[ID]);
    }  
    free(Index_Rcv_Grid_B_AB2C);
  }

  size_Index_Snd_Grid_B_AB2C = 0;
  Index_Snd_Grid_B_AB2C = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Snd_Grid_B_AB2C[ID] = (int*)malloc(sizeof(int)*Num_Snd_Grid_B_AB2C[ID]);
    size_Index_Snd_Grid_B_AB2C += Num_Snd_Grid_B_AB2C[ID];
  }  
  
  size_Index_Rcv_Grid_B_AB2C = 0; 
  Index_Rcv_Grid_B_AB2C = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Rcv_Grid_B_AB2C[ID] = (int*)malloc(sizeof(int)*Num_Rcv_Grid_B_AB2C[ID]);
    size_Index_Rcv_Grid_B_AB2C += Num_Rcv_Grid_B_AB2C[ID]; 
  }  

  alloc_first[31] = 0;

  /* construct Index_Snd_Grid_B_AB2C */  

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_AB2C[ID] = 0;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN_AB = BN_AB + GNs;

    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
    ID = n3*numprocs/Ngrid3;
    GN_C = n3*Ngrid1*Ngrid2 + n1*Ngrid2 + n2;
    BN_C = GN_C - ((ID*Ngrid3+numprocs-1)/numprocs)*Ngrid1*Ngrid2;
    Index_Snd_Grid_B_AB2C[ID][Num_Snd_Grid_B_AB2C[ID]] = BN_C;
    Num_Snd_Grid_B_AB2C[ID]++;
  }

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
   
    if (Num_Snd_Grid_B_AB2C[IDS]!=0){
      MPI_Isend( &Index_Snd_Grid_B_AB2C[IDS][0], Num_Snd_Grid_B_AB2C[IDS], 
		 MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }

    if (Num_Rcv_Grid_B_AB2C[IDR]!=0){
      MPI_Recv( &Index_Rcv_Grid_B_AB2C[IDR][0], Num_Rcv_Grid_B_AB2C[IDR], 
		MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    }

    if (Num_Snd_Grid_B_AB2C[IDS]!=0)  MPI_Wait(&request,&stat);
  }

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_AB2C[ID] = 0;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN_AB = BN_AB + GNs;
    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;
    ID = n3*numprocs/Ngrid3;
    Index_Snd_Grid_B_AB2C[ID][Num_Snd_Grid_B_AB2C[ID]] = BN_AB;
    Num_Snd_Grid_B_AB2C[ID]++;
  }

  /* find the maximum Num_Snd_Grid_B_AB2C and Num_Rcv_Grid_B_AB2C */

  Max_Num_Snd_Grid_B_AB2C = 0;
  Max_Num_Rcv_Grid_B_AB2C = 0;

  for (ID=0; ID<numprocs; ID++){
    if (Max_Num_Snd_Grid_B_AB2C<Num_Snd_Grid_B_AB2C[ID]){ 
      Max_Num_Snd_Grid_B_AB2C = Num_Snd_Grid_B_AB2C[ID];
    }

    if (Max_Num_Rcv_Grid_B_AB2C<Num_Rcv_Grid_B_AB2C[ID]){ 
      Max_Num_Rcv_Grid_B_AB2C = Num_Rcv_Grid_B_AB2C[ID];
    }
  }
  /* end added by mari 05.12.2014 */

  /****************************************************************
      CA to CB in the partition B for MPI communication in FFT 
  ****************************************************************/

  /* set GNs */

  N2D = Ngrid3*Ngrid1;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid2;

  /* find Num_Snd_Grid_B_CA2CB */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_CA2CB[ID] = 0;

  N2D = Ngrid3*Ngrid2; /* for CB */

  for (BN_CA=0; BN_CA<My_NumGridB_CA; BN_CA++){

    GN_CA = GNs + BN_CA;    
    n3 = GN_CA/(Ngrid1*Ngrid2);
    n1 = (GN_CA - n3*(Ngrid1*Ngrid2))/Ngrid2;
    n2 = GN_CA - n3*(Ngrid1*Ngrid2) - n1*Ngrid2;
    n2D = n3*Ngrid2 + n2;
    ID = n2D*numprocs/N2D;
    Num_Snd_Grid_B_CA2CB[ID]++;
  }

  /* MPI: Num_Snd_Grid_B_CA2CB */  

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
    MPI_Isend(&Num_Snd_Grid_B_CA2CB[IDS], 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    MPI_Recv(&Num_Rcv_Grid_B_CA2CB[IDR], 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    MPI_Wait(&request,&stat);
  }

  /* allocation of arrays */  

  if (alloc_first[29]==0){

    for (ID=0; ID<numprocs; ID++){
      free(Index_Snd_Grid_B_CA2CB[ID]);
    }  
    free(Index_Snd_Grid_B_CA2CB);
  
    for (ID=0; ID<numprocs; ID++){
      free(Index_Rcv_Grid_B_CA2CB[ID]);
    }  
    free(Index_Rcv_Grid_B_CA2CB);

    free(ID_NN_B_CA2CB_S);
    free(ID_NN_B_CA2CB_R);
    free(GP_B_CA2CB_S);
    free(GP_B_CA2CB_R);
  }

  size_Index_Snd_Grid_B_CA2CB = 0;
  Index_Snd_Grid_B_CA2CB = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Snd_Grid_B_CA2CB[ID] = (int*)malloc(sizeof(int)*Num_Snd_Grid_B_CA2CB[ID]);
    size_Index_Snd_Grid_B_CA2CB += Num_Snd_Grid_B_CA2CB[ID];
  }  
  
  size_Index_Rcv_Grid_B_CA2CB = 0; 
  Index_Rcv_Grid_B_CA2CB = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Index_Rcv_Grid_B_CA2CB[ID] = (int*)malloc(sizeof(int)*Num_Rcv_Grid_B_CA2CB[ID]);
    size_Index_Rcv_Grid_B_CA2CB += Num_Rcv_Grid_B_CA2CB[ID]; 
  }  

  alloc_first[29] = 0;

  /* construct Index_Snd_Grid_B_CA2CB */  

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_CA2CB[ID] = 0;

  N2D = Ngrid3*Ngrid2; /* for CB */

  for (BN_CA=0; BN_CA<My_NumGridB_CA; BN_CA++){

    GN_CA = GNs + BN_CA;    
    n3 = GN_CA/(Ngrid1*Ngrid2);
    n1 = (GN_CA - n3*(Ngrid1*Ngrid2))/Ngrid2;
    n2 = GN_CA - n3*(Ngrid1*Ngrid2) - n1*Ngrid2;
    n2D = n3*Ngrid2 + n2;
    ID = n2D*numprocs/N2D;
    GN_CB = n3*Ngrid2*Ngrid1 + n2*Ngrid1 + n1;
    BN_CB = GN_CB - ((ID*N2D+numprocs-1)/numprocs)*Ngrid1;
    Index_Snd_Grid_B_CA2CB[ID][Num_Snd_Grid_B_CA2CB[ID]] = BN_CB;
    Num_Snd_Grid_B_CA2CB[ID]++;
  }

  /* MPI: Index_Snd_Grid_B_CA2CB */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;
   
    if (Num_Snd_Grid_B_CA2CB[IDS]!=0){
      MPI_Isend( &Index_Snd_Grid_B_CA2CB[IDS][0], Num_Snd_Grid_B_CA2CB[IDS], 
                 MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }

    if (Num_Rcv_Grid_B_CA2CB[IDR]!=0){
      MPI_Recv( &Index_Rcv_Grid_B_CA2CB[IDR][0], Num_Rcv_Grid_B_CA2CB[IDR], 
                MPI_INT, IDR, tag, mpi_comm_level1, &stat);
    }

    if (Num_Snd_Grid_B_CA2CB[IDS]!=0)  MPI_Wait(&request,&stat);
  }

  /* reset: Index_Snd_Grid_B_CA2CB

     Index_Snd_Grid_B_CB2CA:  BN_CA
     Index_Rcv_Grid_B_CB2CA:  BN_CB
  */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_B_CA2CB[ID] = 0;

  N2D = Ngrid3*Ngrid2; /* for CB */

  for (BN_CA=0; BN_CA<My_NumGridB_CA; BN_CA++){

    GN_CA = GNs + BN_CA;    
    n3 = GN_CA/(Ngrid1*Ngrid2);
    n1 = (GN_CA - n3*(Ngrid1*Ngrid2))/Ngrid2;
    n2 = GN_CA - n3*(Ngrid1*Ngrid2) - n1*Ngrid2;
    n2D = n3*Ngrid2 + n2;
    ID = n2D*numprocs/N2D;
    Index_Snd_Grid_B_CA2CB[ID][Num_Snd_Grid_B_CA2CB[ID]] = BN_CA;
    Num_Snd_Grid_B_CA2CB[ID]++;
  }

  /* find the maximum Num_Snd_Grid_B_CA2CB and Num_Rcv_Grid_B_CA2CB */

  Max_Num_Snd_Grid_B_CA2CB = 0;
  Max_Num_Rcv_Grid_B_CA2CB = 0;

  for (ID=0; ID<numprocs; ID++){
    if (Max_Num_Snd_Grid_B_CA2CB<Num_Snd_Grid_B_CA2CB[ID]){ 
      Max_Num_Snd_Grid_B_CA2CB = Num_Snd_Grid_B_CA2CB[ID];
    }

    if (Max_Num_Rcv_Grid_B_CA2CB<Num_Rcv_Grid_B_CA2CB[ID]){ 
      Max_Num_Rcv_Grid_B_CA2CB = Num_Rcv_Grid_B_CA2CB[ID];
    }
  }

  /* find NN_B_CA2CB_S and NN_B_CA2CB_R 
     and set ID_NN_B_CA2CB_S, 
             ID_NN_B_CA2CB_R,
             GP_B_CA2CB_S,
             GP_B_CA2CB_R
  */

  NN_B_CA2CB_S = 0;
  NN_B_CA2CB_R = 0;

  for (ID=0; ID<numprocs; ID++){
    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;
    if (Num_Snd_Grid_B_CA2CB[IDS]!=0) NN_B_CA2CB_S++;
    if (Num_Rcv_Grid_B_CA2CB[IDR]!=0) NN_B_CA2CB_R++;
  }

  ID_NN_B_CA2CB_S = (int*)malloc(sizeof(int)*NN_B_CA2CB_S);
  ID_NN_B_CA2CB_R = (int*)malloc(sizeof(int)*NN_B_CA2CB_R);
  GP_B_CA2CB_S = (int*)malloc(sizeof(int)*(NN_B_CA2CB_S+1));
  GP_B_CA2CB_R = (int*)malloc(sizeof(int)*(NN_B_CA2CB_R+1));

  NN_B_CA2CB_S = 0;
  NN_B_CA2CB_R = 0;
  GP_B_CA2CB_S[0] = 0;
  GP_B_CA2CB_R[0] = 0;

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_B_CA2CB[IDS]!=0){
      ID_NN_B_CA2CB_S[NN_B_CA2CB_S] = IDS;
      NN_B_CA2CB_S++;
      GP_B_CA2CB_S[NN_B_CA2CB_S] = GP_B_CA2CB_S[NN_B_CA2CB_S-1] + Num_Snd_Grid_B_CA2CB[IDS];
    }

    if (Num_Rcv_Grid_B_CA2CB[IDR]!=0){
      ID_NN_B_CA2CB_R[NN_B_CA2CB_R] = IDR;
      NN_B_CA2CB_R++;
      GP_B_CA2CB_R[NN_B_CA2CB_R] = GP_B_CA2CB_R[NN_B_CA2CB_R-1] + Num_Rcv_Grid_B_CA2CB[IDR];
    }
  }

  /* set My_NumGridB_CB */

  N2D = Ngrid3*Ngrid2;
  My_NumGridB_CB = (((myid+1)*N2D+numprocs-1)/numprocs)*Ngrid1
                  - ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  /* find My_Max_NumGridB */

  My_Max_NumGridB = 0;
  if (My_Max_NumGridB<My_NumGridB_AB) My_Max_NumGridB = My_NumGridB_AB;
  if (My_Max_NumGridB<My_NumGridB_CA) My_Max_NumGridB = My_NumGridB_CA;
  if (My_Max_NumGridB<My_NumGridB_CB) My_Max_NumGridB = My_NumGridB_CB;

  /* PrintMemory */
  if (firsttime){
  PrintMemory("truncation: Index_Snd_Grid_A2B",     sizeof(int)*size_Index_Snd_Grid_A2B,  NULL);
  PrintMemory("truncation: Index_Rcv_Grid_A2B",     sizeof(int)*size_Index_Rcv_Grid_A2B,  NULL);
  PrintMemory("truncation: Index_Snd_Grid_B2C",     sizeof(int)*size_Index_Snd_Grid_B2C,  NULL);
  PrintMemory("truncation: Index_Rcv_Grid_B2C",     sizeof(int)*size_Index_Rcv_Grid_B2C,  NULL);
  PrintMemory("truncation: Index_Snd_Grid_B2D",     sizeof(int)*size_Index_Snd_Grid_B2D,  NULL);
  PrintMemory("truncation: Index_Rcv_Grid_B2D",     sizeof(int)*size_Index_Rcv_Grid_B2D,  NULL);
  PrintMemory("truncation: Index_Snd_Grid_B_AB2CA", sizeof(int)*size_Index_Snd_Grid_B_AB2CA, NULL);
  PrintMemory("truncation: Index_Rcv_Grid_B_AB2CA", sizeof(int)*size_Index_Rcv_Grid_B_AB2CA, NULL);
  PrintMemory("truncation: Index_Snd_Grid_B_CA2CB", sizeof(int)*size_Index_Snd_Grid_B_CA2CB, NULL);
  PrintMemory("truncation: Index_Rcv_Grid_B_CA2CB", sizeof(int)*size_Index_Rcv_Grid_B_CA2CB, NULL);
  PrintMemory("truncation: ID_NN_B_AB2CA_S", sizeof(int)*NN_B_AB2CA_S, NULL);
  PrintMemory("truncation: ID_NN_B_AB2CA_R", sizeof(int)*NN_B_AB2CA_R, NULL);
  PrintMemory("truncation: ID_NN_B_CA2CB_S", sizeof(int)*NN_B_CA2CB_S, NULL);
  PrintMemory("truncation: ID_NN_B_CA2CB_R", sizeof(int)*NN_B_CA2CB_R, NULL);
  PrintMemory("truncation: GP_B_AB2CA_S", sizeof(int)*(NN_B_AB2CA_S+1), NULL);
  PrintMemory("truncation: GP_B_AB2CA_R", sizeof(int)*(NN_B_AB2CA_R+1), NULL);
  PrintMemory("truncation: GP_B_CA2CB_S", sizeof(int)*(NN_B_CA2CB_S+1), NULL);
  PrintMemory("truncation: GP_B_CA2CB_R", sizeof(int)*(NN_B_CA2CB_R+1), NULL);
  firsttime = 0;
  }
}

#pragma optimization_level 1
void Construct_ONAN()
{
  /**********************************************
      construct information of ONAN for EGAC 
  **********************************************/

  int i,m,job_id,job_gid,ct_AN,ID;
  int Rh,Rq,Rp,Gc_AN,Gh_AN,Gq_AN,Gp_AN,h_AN,q_AN,p_AN,Mc_AN;
  int po,m1,m2,m3,l1,l2,l3,tmp_ONAN,onan_tmp;
  int *tmp_natn_onan,*tmp_ncn_onan1,*tmp_ncn_onan2,*tmp_ncn_onan3;
  double r2,dx,dy,dz;
  int Cwan,Hwan,tno1,tno2;
  int spin,pole,n,k,jobid,Gi;
  int ig,j,jg,Rni,Rnj,Rn,p1,p2,p3;
  int mi1,mi2,mi3,mj1,mj2,mj3,i0,j0;
  int Ntot,num1,num2,num3,Num_Procs2,N3[4];
  int *atom_required_flag;
  int **Indx_JOB,**Indx_h_AN;
  int size1,size2;
  int myid,numprocs,IDS,IDR,tag;
  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    /******************************************
                 the first search
    ******************************************/

    onan_tmp = 0;
    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Rh = ncn[Gc_AN][h_AN];

      l1 = atv_ijk[Rh][1];
      l2 = atv_ijk[Rh][2];
      l3 = atv_ijk[Rh][3];

      for (q_AN=0; q_AN<=FNAN[Gh_AN]; q_AN++){

	Gq_AN = natn[Gh_AN][q_AN];
	Rq = ncn[Gh_AN][q_AN];

	m1 = atv_ijk[Rq][1] + l1;
	m2 = atv_ijk[Rq][2] + l2;
	m3 = atv_ijk[Rq][3] + l3;

	/* check whether q_AN is in the FNAN+SNAN or not. */                    

	dx = Gxyz[Gc_AN][1] - (Gxyz[Gq_AN][1] + m1*tv[1][1] + m2*tv[2][1] + m3*tv[3][1]);
	dy = Gxyz[Gc_AN][2] - (Gxyz[Gq_AN][2] + m1*tv[1][2] + m2*tv[2][2] + m3*tv[3][2]);
	dz = Gxyz[Gc_AN][3] - (Gxyz[Gq_AN][3] + m1*tv[1][3] + m2*tv[2][3] + m3*tv[3][3]);

	r2 = dx*dx + dy*dy + dz*dz;

	if ((BCR*BCR)<r2){
	  onan_tmp++;
	}

      } /* q_AN */  
    } /* h_AN */

    /******************************************
                 the second search
    ******************************************/

    /* allocation of arrays */      

    tmp_ONAN = 0;
    tmp_natn_onan = (int*)malloc(sizeof(int)*(onan_tmp+1));
    tmp_ncn_onan1 = (int*)malloc(sizeof(int)*(onan_tmp+1));
    tmp_ncn_onan2 = (int*)malloc(sizeof(int)*(onan_tmp+1));
    tmp_ncn_onan3 = (int*)malloc(sizeof(int)*(onan_tmp+1));

    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Rh = ncn[Gc_AN][h_AN];

      l1 = atv_ijk[Rh][1];
      l2 = atv_ijk[Rh][2];
      l3 = atv_ijk[Rh][3];

      for (q_AN=0; q_AN<=FNAN[Gh_AN]; q_AN++){

	Gq_AN = natn[Gh_AN][q_AN];
	Rq = ncn[Gh_AN][q_AN];

	m1 = atv_ijk[Rq][1] + l1;
	m2 = atv_ijk[Rq][2] + l2;
	m3 = atv_ijk[Rq][3] + l3;

        po = 0;

	/* check whether q_AN exists in the FNAN+SNAN or not. */
        for (p_AN=0; p_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); p_AN++){

	  Gp_AN = natn[Gc_AN][p_AN];
	  Rp = ncn[Gc_AN][p_AN];
	  p1 = atv_ijk[Rp][1];
	  p2 = atv_ijk[Rp][2];
	  p3 = atv_ijk[Rp][3];

	  if ( Gp_AN==Gq_AN 
	       && p1==m1 
	       && p2==m2 
	       && p3==m3 ){
                
	    po = 1;
	    break;
	  }
	}

	/* check whether q_AN has been already registered or not. */
        if (po==0){
	  for (p_AN=0; p_AN<tmp_ONAN; p_AN++){

	    if (    tmp_natn_onan[p_AN]==Gq_AN 
		    && tmp_ncn_onan1[p_AN]==m1 
		    && tmp_ncn_onan2[p_AN]==m2 
		    && tmp_ncn_onan3[p_AN]==m3 ){
                
	      po = 1;
	      break;
	    }
	  }
	}

	/* if q_AN is in the outer region, then register it. */ 
        if (po==0){

	  tmp_natn_onan[tmp_ONAN] = Gq_AN;
	  tmp_ncn_onan1[tmp_ONAN] = m1;
	  tmp_ncn_onan2[tmp_ONAN] = m2;
	  tmp_ncn_onan3[tmp_ONAN] = m3;

	  tmp_ONAN++;
	}

      } /* q_AN */  
    } /* h_AN */

    /* store ONAN */

    ONAN[Gc_AN] = tmp_ONAN;

    /* freening of arrays */      

    free(tmp_natn_onan);
    free(tmp_ncn_onan1);
    free(tmp_ncn_onan2);
    free(tmp_ncn_onan3);

  } /* Mc_AN */

    /* MPI_Bcast of ONAN */

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    ID = G2ID[ct_AN];
    MPI_Bcast(&ONAN[ct_AN], 1, MPI_INT, ID, mpi_comm_level1);
  }

  /******************************************
       freeing and allocation of arrays
  ******************************************/

  if (alloc_first[33]==0){

    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(natn_onan[Gc_AN]);
      free(ncn_onan1[Gc_AN]);
      free(ncn_onan2[Gc_AN]);
      free(ncn_onan3[Gc_AN]);
    }
    
    free(natn_onan);
    free(ncn_onan1);
    free(ncn_onan2);
    free(ncn_onan3);
  }

  natn_onan = (int**)malloc(sizeof(int*)*(atomnum+1));
  ncn_onan1 = (int**)malloc(sizeof(int*)*(atomnum+1));
  ncn_onan2 = (int**)malloc(sizeof(int*)*(atomnum+1));
  ncn_onan3 = (int**)malloc(sizeof(int*)*(atomnum+1));

  ONAN[0] = 0;
  Max_ONAN = 0;
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    natn_onan[ct_AN] = (int*)malloc(sizeof(int)*(ONAN[ct_AN]+1));
    ncn_onan1[ct_AN] = (int*)malloc(sizeof(int)*(ONAN[ct_AN]+1));
    ncn_onan2[ct_AN] = (int*)malloc(sizeof(int)*(ONAN[ct_AN]+1));
    ncn_onan3[ct_AN] = (int*)malloc(sizeof(int)*(ONAN[ct_AN]+1));
    if (Max_ONAN<ONAN[ct_AN]) Max_ONAN = ONAN[ct_AN];
  }

  alloc_first[33] = 0;

  /******************************************
               the third search
  ******************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    /* allocation of arrays */      

    tmp_ONAN = 0;
    tmp_natn_onan = (int*)malloc(sizeof(int)*(Max_ONAN+1));
    tmp_ncn_onan1 = (int*)malloc(sizeof(int)*(Max_ONAN+1));
    tmp_ncn_onan2 = (int*)malloc(sizeof(int)*(Max_ONAN+1));
    tmp_ncn_onan3 = (int*)malloc(sizeof(int)*(Max_ONAN+1));

    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Rh = ncn[Gc_AN][h_AN];

      l1 = atv_ijk[Rh][1];
      l2 = atv_ijk[Rh][2];
      l3 = atv_ijk[Rh][3];

      for (q_AN=0; q_AN<=FNAN[Gh_AN]; q_AN++){

	Gq_AN = natn[Gh_AN][q_AN];
	Rq = ncn[Gh_AN][q_AN];

	m1 = atv_ijk[Rq][1] + l1;
	m2 = atv_ijk[Rq][2] + l2;
	m3 = atv_ijk[Rq][3] + l3;

	po = 0;

	/* check whether q_AN exists in the FNAN+SNAN or not. */
        for (p_AN=0; p_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); p_AN++){

	  Gp_AN = natn[Gc_AN][p_AN];
	  Rp = ncn[Gc_AN][p_AN];
	  p1 = atv_ijk[Rp][1];
	  p2 = atv_ijk[Rp][2];
	  p3 = atv_ijk[Rp][3];

	  if (    Gp_AN==Gq_AN 
		  && p1==m1 
		  && p2==m2 
		  && p3==m3 ){
                
	    po = 1;
	    break;
	  }
	}

	/* check whether q_AN has been already registered or not. */
        if (po==0){
	  for (p_AN=0; p_AN<tmp_ONAN; p_AN++){

	    if (    tmp_natn_onan[p_AN]==Gq_AN 
		    && tmp_ncn_onan1[p_AN]==m1 
		    && tmp_ncn_onan2[p_AN]==m2 
		    && tmp_ncn_onan3[p_AN]==m3 ){
                
	      po = 1;
	      break;
	    }
	  }
	}

	if (po==0){

	  tmp_natn_onan[tmp_ONAN] = Gq_AN;
	  tmp_ncn_onan1[tmp_ONAN] = m1;
	  tmp_ncn_onan2[tmp_ONAN] = m2;
	  tmp_ncn_onan3[tmp_ONAN] = m3;
	  tmp_ONAN++;
	}

      } /* q_AN */  
    } /* h_AN */

      /* store information */

    for (h_AN=0; h_AN<ONAN[Gc_AN]; h_AN++){
      natn_onan[Gc_AN][h_AN] = tmp_natn_onan[h_AN];
      ncn_onan1[Gc_AN][h_AN] = tmp_ncn_onan1[h_AN];
      ncn_onan2[Gc_AN][h_AN] = tmp_ncn_onan2[h_AN];
      ncn_onan3[Gc_AN][h_AN] = tmp_ncn_onan3[h_AN];
    }

    /* freening of arrays */      

    free(tmp_natn_onan);
    free(tmp_ncn_onan1);
    free(tmp_ncn_onan2);
    free(tmp_ncn_onan3);
  }

  /* MPI_Bcast of natn_onan, ncn_onan1, ncn_onan2, and ncn_onan3. */      

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    ID = G2ID[ct_AN];
    MPI_Bcast(&natn_onan[ct_AN][0], ONAN[ct_AN], MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&ncn_onan1[ct_AN][0], ONAN[ct_AN], MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&ncn_onan2[ct_AN][0], ONAN[ct_AN], MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&ncn_onan3[ct_AN][0], ONAN[ct_AN], MPI_INT, ID, mpi_comm_level1);
  }

  /*
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    for (h_AN=0; h_AN<ONAN[ct_AN]; h_AN++){
      printf("ABC ct_AN=%2d h_AN=%2d  natn=%2d  ncn=%2d %2d %2d\n",
	     ct_AN,h_AN,natn_onan[ct_AN][h_AN],
	     ncn_onan1[ct_AN][h_AN],ncn_onan2[ct_AN][h_AN],ncn_onan3[ct_AN][h_AN]);
    }
  }
  MPI_Finalize();   
  exit(0);
  */

  /*************************************************************************

   information for the MPI communication of H and S
   atoms, spin, poles

   Indx_Snd_HS_EGAC: local index of atom (Matomnum) starting from 1.
   Indx_Rcv_HS_EGAC: global index of atom
   Top_Index_HS_EGAC: local index of atom (Matomnum_EGAC) starting from 0.

   Num_Snd_HS_EGAC: the number of atoms whose information is sent.
   Num_Rcv_HS_EGAC: the number of atoms whose information is received. 

   Note that indexing of Matomnum_EGAC is totally different from Matomnum.   

  *************************************************************************/

  /* allocation of arrays */

  atom_required_flag = (int*)malloc(sizeof(int)*(atomnum+1));

  /* allocation of computational domain */

  Ntot = atomnum*(SpinP_switch+1)*EGAC_Npoles;

  for (i=0; i<numprocs; i++){
    EGAC_Top[i] = 0;
    EGAC_End[i] = 0;
  }

  if (Ntot<numprocs) Num_Procs2 = Ntot;
  else               Num_Procs2 = numprocs;

  num1 = Ntot/Num_Procs2;
  num2 = Ntot%Num_Procs2;

  for (i=0; i<Num_Procs2; i++){
    EGAC_Top[i] = num1*i;
    EGAC_End[i] = num1*(i + 1) - 1;
  }

  if (num2!=0){
    for (i=0; i<num2; i++){
      EGAC_Top[i] = EGAC_Top[i] + i;
      EGAC_End[i] = EGAC_End[i] + i + 1;
    }
    for (i=num2; i<Num_Procs2; i++){
      EGAC_Top[i] = EGAC_Top[i] + num2;
      EGAC_End[i] = EGAC_End[i] + num2;
    }
  }

  if (myid<Num_Procs2)
    EGAC_Num = EGAC_End[myid] - EGAC_Top[myid] + 1;
  else 
    EGAC_Num = 0;

  /*****************************************************************
                   for MPI communication of DM 
  *****************************************************************/

  for (i=1; i<=atomnum; i++){
    atom_required_flag[i] = 0; 
  }  

  for (job_id=0; job_id<EGAC_Num; job_id++){
    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];
    atom_required_flag[Gc_AN] = 1; 
  }

  for (i=0; i<numprocs; i++){
    Num_Snd_DM_EGAC[i] = 0;
  }

  for (i=1; i<=atomnum; i++){
    if (atom_required_flag[i]==1){
      ID = G2ID[i];
      Num_Snd_DM_EGAC[ID]++;
    }
  }  

  /* MPI: Num_Snd_DM_EGAC */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    MPI_Isend(&Num_Snd_DM_EGAC[IDS],1,MPI_INT,IDS,tag,mpi_comm_level1,&request);
    MPI_Recv( &Num_Rcv_DM_EGAC[IDR],1,MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
    MPI_Wait(&request,&stat);
  }  

  /* allocation of arrays: Indx_Rcv_DM_EGAC and Indx_Snd_DM_EGAC */

  if (alloc_first[34]==0){

    for (i=0; i<numprocs; i++){
      free(Indx_Rcv_DM_EGAC[i]);
    }
    free(Indx_Rcv_DM_EGAC);

    for (i=0; i<numprocs; i++){
      free(Indx_Snd_DM_EGAC[i]);
    }
    free(Indx_Snd_DM_EGAC);
  }

  Indx_Rcv_DM_EGAC = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_Rcv_DM_EGAC[i] = (int*)malloc(sizeof(int)*(Num_Rcv_DM_EGAC[i]+1));
  }

  Indx_Snd_DM_EGAC = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_Snd_DM_EGAC[i] = (int*)malloc(sizeof(int)*(Num_Snd_DM_EGAC[i]+1));
  }

  /* set Indx_Snd_DM_EGAC */

  for (ID=0; ID<numprocs; ID++) Num_Snd_DM_EGAC[ID] = 0;

  for (i=1; i<=atomnum; i++){
    if (atom_required_flag[i]==1){
      ID = G2ID[i];
      Indx_Snd_DM_EGAC[ID][Num_Snd_DM_EGAC[ID]] = i; 
      Num_Snd_DM_EGAC[ID]++;
    }
  }  

  /* MPI: Indx_Snd_DM_EGAC and Indx_Rcv_DM_EGAC */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_DM_EGAC[IDS]!=0){
      MPI_Isend(&Indx_Snd_DM_EGAC[IDS][0],Num_Snd_DM_EGAC[IDS],MPI_INT,IDS,tag,mpi_comm_level1,&request);
    }

    if (Num_Rcv_DM_EGAC[IDR]!=0){
      MPI_Recv( &Indx_Rcv_DM_EGAC[IDR][0],Num_Rcv_DM_EGAC[IDR],MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
    }

    MPI_Wait(&request,&stat);
  }  

  /* set M2G_DM_Snd_EGAC */

  if (alloc_first[34]==0){
    free(M2G_DM_Snd_EGAC);
  }

  num3 = 0;
  for (ID=0; ID<numprocs; ID++){
    num3 += Num_Snd_DM_EGAC[ID];
  }

  M2G_DM_Snd_EGAC = (int*)malloc(sizeof(int)*num3);

  Matomnum_DM_Snd_EGAC = num3;

  for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++) G2M_DM_Snd_EGAC[Gc_AN] = -1;

  k = 0;
  for (ID=0; ID<numprocs; ID++){
    for (i=0; i<Num_Snd_DM_EGAC[ID]; i++){

      Gc_AN = Indx_Snd_DM_EGAC[ID][i]; 
      M2G_DM_Snd_EGAC[k] = Gc_AN;
      G2M_DM_Snd_EGAC[Gc_AN] = k;

      k++; 
    }
  }

  /* allocation of DM_Snd_EGAC */
  /* note that freeing of DM_Snd_EGAC is done in free_arrays_truncation0() */

  DM_Snd_EGAC = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){

    DM_Snd_EGAC[spin] = (double****)malloc(sizeof(double***)*Matomnum_DM_Snd_EGAC);

    for (Mc_AN=0; Mc_AN<Matomnum_DM_Snd_EGAC; Mc_AN++){

      Gc_AN = M2G_DM_Snd_EGAC[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      DM_Snd_EGAC[spin][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<(FNAN[Gc_AN]+1); h_AN++){

        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno2 = Spe_Total_CNO[Hwan];

        DM_Snd_EGAC[spin][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno1);
	for (i=0; i<tno1; i++){
          DM_Snd_EGAC[spin][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno2);
	}
      }
    }
  }

  /* determine the data size in MPI of DM */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;

    /* find data size to send block data */

    if ( Num_Snd_DM_EGAC[IDS]!=0 ){

      size1 = 0;

      for (n=0; n<Num_Snd_DM_EGAC[IDS]; n++){

	Gc_AN = Indx_Snd_DM_EGAC[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_CNO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[Hwan];
	  size1 += tno1*tno2;
	}
      }

      Snd_DM_EGAC_Size[IDS] = size1*(SpinP_switch+1);

      if (ID!=0) MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }
    else{
      Snd_DM_EGAC_Size[IDS] = 0;
    }

    /* receiving of size of data */

    if ( Num_Rcv_DM_EGAC[IDR]!=0 ){

      if (ID!=0){ 
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_DM_EGAC_Size[IDR] = size2;
      }
      else{
        Rcv_DM_EGAC_Size[IDR] = Snd_DM_EGAC_Size[IDS];
      } 
    }
    else{
      Rcv_DM_EGAC_Size[IDR] = 0;
    }

    if ( Num_Snd_DM_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
  }

  /* find Max_Snd_OLP_EGAC_Size among Snd_DM_EGAC_Size and Snd_OLP_EGAC_Size
     find Max_Rcv_OLP_EGAC_Size among Rcv_DM_EGAC_Size and Rcv_OLP_EGAC_Size */

  Max_Snd_OLP_EGAC_Size = 0;
  Max_Rcv_OLP_EGAC_Size = 0;

  for (ID=0; ID<numprocs; ID++){
    if (Max_Snd_OLP_EGAC_Size<Snd_DM_EGAC_Size[ID]) Max_Snd_OLP_EGAC_Size = Snd_DM_EGAC_Size[ID];
    if (Max_Rcv_OLP_EGAC_Size<Rcv_DM_EGAC_Size[ID]) Max_Rcv_OLP_EGAC_Size = Rcv_DM_EGAC_Size[ID];
  }

  /*****************************************************************
                   for MPI communication of H and S 
  *****************************************************************/

  for (i=1; i<=atomnum; i++){
    atom_required_flag[i] = 0; 
  }  

  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];

    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      atom_required_flag[Gh_AN] = 1; 
    }

    for (h_AN=0; h_AN<ONAN[Gc_AN]; h_AN++){
      Gh_AN = natn_onan[Gc_AN][h_AN];
      atom_required_flag[Gh_AN] = 1; 
    }
  }

  for (i=0; i<numprocs; i++){
    Num_Rcv_HS_EGAC[i] = 0;
  }

  for (i=1; i<=atomnum; i++){
    if (atom_required_flag[i]==1){
      ID = G2ID[i];
      Num_Rcv_HS_EGAC[ID]++;
    }
  }  

  /* MPI: Num_Rcv_HS_EGAC */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    MPI_Isend(&Num_Rcv_HS_EGAC[IDS],1,MPI_INT,IDS,tag,mpi_comm_level1,&request);
    MPI_Recv( &Num_Snd_HS_EGAC[IDR],1,MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
    MPI_Wait(&request,&stat);
  }  

  /* allocation of arrays: Indx_Rcv_HS_EGAC and Indx_Snd_HS_EGAC */

  if (alloc_first[34]==0){

    for (i=0; i<numprocs; i++){
      free(Indx_Rcv_HS_EGAC[i]);
    }
    free(Indx_Rcv_HS_EGAC);

    for (i=0; i<numprocs; i++){
      free(Indx_Snd_HS_EGAC[i]);
    }
    free(Indx_Snd_HS_EGAC);
  }

  Indx_Rcv_HS_EGAC = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_Rcv_HS_EGAC[i] = (int*)malloc(sizeof(int)*(Num_Rcv_HS_EGAC[i]+1));
  }

  Indx_Snd_HS_EGAC = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_Snd_HS_EGAC[i] = (int*)malloc(sizeof(int)*(Num_Snd_HS_EGAC[i]+1));
  }

  /* set Indx_Rcv_HS_EGAC */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_HS_EGAC[ID] = 0;

  for (i=1; i<=atomnum; i++){
    if (atom_required_flag[i]==1){
      ID = G2ID[i];
      Indx_Rcv_HS_EGAC[ID][Num_Rcv_HS_EGAC[ID]] = i; 
      Num_Rcv_HS_EGAC[ID]++;
    }
  }  

  /* MPI: Indx_Rcv_HS_EGAC */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Rcv_HS_EGAC[IDS]!=0){
      MPI_Isend(&Indx_Rcv_HS_EGAC[IDS][0],Num_Rcv_HS_EGAC[IDS],MPI_INT,IDS,tag,mpi_comm_level1,&request);
    }

    if (Num_Snd_HS_EGAC[IDR]!=0){
      MPI_Recv( &Indx_Snd_HS_EGAC[IDR][0],Num_Snd_HS_EGAC[IDR],MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
    }

    MPI_Wait(&request,&stat);
  }  

  /*
  if (myid==0){
    for (i=0; i<Num_Rcv_HS_EGAC[1]; i++){ 
      printf("ABC myid=%2d Indx_Rcv=%2d\n",myid,Indx_Rcv_HS_EGAC[1][i]);
    } 
    for (i=0; i<Num_Snd_HS_EGAC[1]; i++){ 
      printf("ABC myid=%2d Indx_Snd=%2d\n",myid,Indx_Snd_HS_EGAC[1][i]);
    } 
  }  
  */

  /* set M2G_EGAC */

  if (alloc_first[34]==0){
    free(M2G_EGAC);
  }

  num3 = 0;
  for (ID=0; ID<numprocs; ID++){
    num3 += Num_Rcv_HS_EGAC[ID];
  }

  M2G_EGAC = (int*)malloc(sizeof(int)*num3);

  Matomnum_EGAC = num3;
  
  for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++) G2M_EGAC[Gc_AN] = -1;

  k = 0;
  for (ID=0; ID<numprocs; ID++){
    for (i=0; i<Num_Rcv_HS_EGAC[ID]; i++){

      if (i==0) Top_Index_HS_EGAC[ID] = k;
      Gc_AN = Indx_Rcv_HS_EGAC[ID][i]; 
      M2G_EGAC[k] = Gc_AN;
      G2M_EGAC[Gc_AN] = k;

      k++; 
    }
  }

  /* Indx_Snd_HS_EGAC: change global to local index */
  
  for (ID=0; ID<numprocs; ID++){
    for (i=0; i<Num_Snd_HS_EGAC[ID]; i++){ 

      k = Indx_Snd_HS_EGAC[ID][i]; /* k: global */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        if (M2G[Mc_AN]==k) break;    
      }

      Indx_Snd_HS_EGAC[ID][i] = Mc_AN; /* Mc_AN: local */
    }
  }

  /* allocation of H_EGAC */
  /* note that freeing of H_EGAC is done in free_arrays_truncation0() */

  H_EGAC = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    H_EGAC[spin] = (double****)malloc(sizeof(double***)*Matomnum_EGAC);
    for (Mc_AN=0; Mc_AN<Matomnum_EGAC; Mc_AN++){

      Gc_AN = M2G_EGAC[Mc_AN];
      Cwan = WhatSpecies[Gc_AN]; 
      tno1 = Spe_Total_CNO[Cwan];

      H_EGAC[spin][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];        
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_CNO[Hwan];
 
	H_EGAC[spin][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno1);
	for (i=0; i<tno1; i++){
	  H_EGAC[spin][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno2);
	}
      }
    }
  }

  /* allocation of OLP_EGAC */
  /* note that freeing of OLP_EGAC is done in free_arrays_truncation0() */

  OLP_EGAC = (double****)malloc(sizeof(double***)*Matomnum_EGAC);
  for (Mc_AN=0; Mc_AN<Matomnum_EGAC; Mc_AN++){

    Gc_AN = M2G_EGAC[Mc_AN];
    Cwan = WhatSpecies[Gc_AN]; 
    tno1 = Spe_Total_CNO[Cwan];

    OLP_EGAC[Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];        
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];
 
      OLP_EGAC[Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno1);
      for (i=0; i<tno1; i++){
	OLP_EGAC[Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno2);
      }
    }
  }

  /* determine the data size in MPI of OLP */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;

    /* find data size to send block data */

    if ( Num_Snd_HS_EGAC[IDS]!=0 ){

      size1 = 0;

      for (n=0; n<Num_Snd_HS_EGAC[IDS]; n++){

	Mc_AN = Indx_Snd_HS_EGAC[IDS][n];
	Gc_AN = M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_CNO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[Hwan];
	  size1 += tno1*tno2;
	}
      }

      Snd_OLP_EGAC_Size[IDS] = size1;

      if (ID!=0) MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }
    else{
      Snd_OLP_EGAC_Size[IDS] = 0;
    }

    /* receiving of size of data */

    if ( Num_Rcv_HS_EGAC[IDR]!=0 ){

      if (ID!=0){ 
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_OLP_EGAC_Size[IDR] = size2;
      }
      else{
        Rcv_OLP_EGAC_Size[IDR] = Snd_OLP_EGAC_Size[IDS];
      } 
    }
    else{
      Rcv_OLP_EGAC_Size[IDR] = 0;
    }

    if ( Num_Snd_HS_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
  }

  /* find Max_Snd_OLP_EGAC_Size among Snd_DM_EGAC_Size and Snd_OLP_EGAC_Size
     find Max_Rcv_OLP_EGAC_Size among Rcv_DM_EGAC_Size and Rcv_OLP_EGAC_Size */

  for (ID=0; ID<numprocs; ID++){
    if (Max_Snd_OLP_EGAC_Size<Snd_OLP_EGAC_Size[ID]) Max_Snd_OLP_EGAC_Size = Snd_OLP_EGAC_Size[ID];
    if (Max_Rcv_OLP_EGAC_Size<Rcv_OLP_EGAC_Size[ID]) Max_Rcv_OLP_EGAC_Size = Rcv_OLP_EGAC_Size[ID];
  }

  /*****************************************************************************
                    connectivity information to make matrices

     RMI1_EGAC: for construction of D, C, and B
     RMI2_EGAC: for construction of GA, GC, and GC

  *****************************************************************************/

  /* allocation of RMI1_EGAC and RMI2_EGAC */
  /* note that freeing of RMI1_EGAC and RMI2_EGAC is done in free_arrays_truncation0() */

  RMI1_EGAC = (int***)malloc(sizeof(int**)*EGAC_Num);
  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];

    RMI1_EGAC[job_id] = (int**)malloc(sizeof(int*)*(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1));
    for (i=0; i<(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1); i++){
      RMI1_EGAC[job_id][i] = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1));
    }
  }

  RMI2_EGAC = (int***)malloc(sizeof(int**)*EGAC_Num);
  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];

    RMI2_EGAC[job_id] = (int**)malloc(sizeof(int*)*(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1));
    for (i=0; i<(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1); i++){
      RMI2_EGAC[job_id][i] = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]+1));
    }
  }

  /* construction of RMI1_EGAC and RMI2_EGAC */

  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];

    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]); i++){

      if ( i<=(FNAN[Gc_AN]+SNAN[Gc_AN]) ){ 
        ig = natn[Gc_AN][i];
        Rni = ncn[Gc_AN][i];
        mi1 = atv_ijk[Rni][1]; 
        mi2 = atv_ijk[Rni][2]; 
        mi3 = atv_ijk[Rni][3]; 
      }
      else{
        i0 = i - (FNAN[Gc_AN]+SNAN[Gc_AN]+1);
        ig = natn_onan[Gc_AN][i0];
        mi1 = ncn_onan1[Gc_AN][i0];
        mi2 = ncn_onan2[Gc_AN][i0];
        mi3 = ncn_onan3[Gc_AN][i0];
      }

      for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]+ONAN[Gc_AN]); j++){

        if ( j<=(FNAN[Gc_AN]+SNAN[Gc_AN]) ){ 

          jg = natn[Gc_AN][j];
          Rnj = ncn[Gc_AN][j];
          mj1 = atv_ijk[Rnj][1]; 
          mj2 = atv_ijk[Rnj][2]; 
          mj3 = atv_ijk[Rnj][3];
	}
        else{

	  j0 = j - (FNAN[Gc_AN]+SNAN[Gc_AN]+1);
          jg = natn_onan[Gc_AN][j0];
          mj1 = ncn_onan1[Gc_AN][j0];
          mj2 = ncn_onan2[Gc_AN][j0];
          mj3 = ncn_onan3[Gc_AN][j0];
	} 

        l1 = mj1 - mi1;
        l2 = mj2 - mi2;
        l3 = mj3 - mi3;
        if (l1<0) m1=-l1; else m1 = l1;
        if (l2<0) m2=-l2; else m2 = l2;
        if (l3<0) m3=-l3; else m3 = l3;

        RMI1_EGAC[job_id][i][j] = -1;
        RMI2_EGAC[job_id][i][j] = -1;

        if (m1<=CpyCell && m2<=CpyCell && m3<=CpyCell){

          Rn = ratv[l1+CpyCell][l2+CpyCell][l3+CpyCell];

          /* RMI1_EGAC */

          k = 0; po = 0;
	  do {
	    if (natn[ig][k]==jg && ncn[ig][k]==Rn){
	      RMI1_EGAC[job_id][i][j] = k;
	      po = 1;
	    }
	    k++;
	  } while (po==0 && k<=FNAN[ig]);

          /* RMI2_EGAC */

          k = 0; po = 0;
	  do {
	    if (natn[ig][k]==jg && ncn[ig][k]==Rn){
	      RMI2_EGAC[job_id][i][j] = k;
	      po = 1;
	    }
	    k++;
	  } while (po==0 && k<=(FNAN[ig]+SNAN[ig]));
	}
      }
    }
  }

  /*****************************************************************************
    information for the MPI communication of GD (GA)
  *****************************************************************************/

  for (ID=0; ID<numprocs; ID++) Num_Rcv_GA_EGAC[ID] = 0;
  
  for (job_id=0; job_id<EGAC_Num; job_id++){
  
    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);

    Gc_AN = N3[1];
    spin  = N3[2];
    pole  = N3[3];

    for (h_AN=0; h_AN<ONAN[Gc_AN]; h_AN++){

      Gh_AN = natn_onan[Gc_AN][h_AN];

      /* search process ID where the data is stored. */

      for (n=pole; n<Ntot; n=n+EGAC_Npoles){

        GN2N_EGAC(n,N3);

        if ( N3[1]==Gh_AN && N3[2]==spin && N3[3]==pole){

          for (ID=0; ID<Num_Procs2; ID++){
            if (EGAC_Top[ID]<=n && n<=EGAC_End[ID]) break;
	  }

          Num_Rcv_GA_EGAC[ID]++;
	}
      }
    }
  }  

  /* MPI: Num_Rcv_GA_EGAC */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    MPI_Isend(&Num_Rcv_GA_EGAC[IDS],1,MPI_INT,IDS,tag,mpi_comm_level1,&request);
    MPI_Recv( &Num_Snd_GA_EGAC[IDR],1,MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
    MPI_Wait(&request,&stat);
  }  
  
  /* allocation of arrays: Indx_Rcv_GA_EGAC and Indx_Snd_GA_EGAC */

  if (alloc_first[34]==0){

    for (i=0; i<numprocs; i++){
      free(Indx_Rcv_GA_EGAC[i]);
    }
    free(Indx_Rcv_GA_EGAC);

    for (i=0; i<numprocs; i++){
      free(Indx_Snd_GA_EGAC[i]);
    }
    free(Indx_Snd_GA_EGAC);
  }  

  Indx_Rcv_GA_EGAC = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_Rcv_GA_EGAC[i] = (int*)malloc(sizeof(int)*(Num_Rcv_GA_EGAC[i]+1));
  }

  Indx_Snd_GA_EGAC = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_Snd_GA_EGAC[i] = (int*)malloc(sizeof(int)*(Num_Snd_GA_EGAC[i]+1));
  }

  Indx_JOB = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_JOB[i] = (int*)malloc(sizeof(int)*(Num_Rcv_GA_EGAC[i]+1));
  }

  Indx_h_AN = (int**)malloc(sizeof(int*)*numprocs);
  for (i=0; i<numprocs; i++){
    Indx_h_AN[i] = (int*)malloc(sizeof(int)*(Num_Rcv_GA_EGAC[i]+1));
  }

  /* set Indx_Rcv_GA_EGAC */

  for (ID=0; ID<numprocs; ID++) Num_Rcv_GA_EGAC[ID] = 0;

  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);

    Gc_AN = N3[1];
    spin  = N3[2];
    pole  = N3[3];

    for (h_AN=0; h_AN<ONAN[Gc_AN]; h_AN++){

      Gh_AN = natn_onan[Gc_AN][h_AN];

      /* search process ID where the data is stored. */

      for (n=pole; n<Ntot; n=n+EGAC_Npoles){ /* n: global index of job */

        GN2N_EGAC(n,N3);

        if ( N3[1]==Gh_AN && N3[2]==spin && N3[3]==pole){

          for (ID=0; ID<Num_Procs2; ID++){
            if (EGAC_Top[ID]<=n && n<=EGAC_End[ID]) break;
	  }

          Indx_Rcv_GA_EGAC[ID][Num_Rcv_GA_EGAC[ID]] = n; 
          Indx_JOB[ID][Num_Rcv_GA_EGAC[ID]] = job_id;  /* local job_id */
          Indx_h_AN[ID][Num_Rcv_GA_EGAC[ID]] = h_AN;

          Num_Rcv_GA_EGAC[ID]++;
	}
      }
    }
  }  

  /* MPI: Indx_Rcv_GA_EGAC */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Rcv_GA_EGAC[IDS]!=0){
      MPI_Isend(&Indx_Rcv_GA_EGAC[IDS][0],Num_Rcv_GA_EGAC[IDS],MPI_INT,IDS,tag,mpi_comm_level1,&request);
    }

    if (Num_Snd_GA_EGAC[IDR]!=0){
      MPI_Recv( &Indx_Snd_GA_EGAC[IDR][0],Num_Snd_GA_EGAC[IDR],MPI_INT,IDR,tag,mpi_comm_level1,&stat); 
    }

    MPI_Wait(&request,&stat);
  }  

  /*
  if (myid==0){

    printf("EGAC_Num=%2d\n",EGAC_Num);
    printf("Ntot=%2d\n",Ntot);

    for (i=0; i<Num_Rcv_GA_EGAC[0]; i++){ 
      printf("ABC myid=%2d i=%2d Indx_Rcv=%2d\n",myid,i,Indx_Rcv_GA_EGAC[0][i]);
    } 
    for (i=0; i<Num_Snd_GA_EGAC[0]; i++){ 
      printf("ABC myid=%2d i=%2d Indx_Snd=%2d\n",myid,i,Indx_Snd_GA_EGAC[0][i]);
    } 
  }  

  MPI_Finalize();   
  exit(0);
  */

  /* set M2G_JOB_EGAC */

  if (alloc_first[34]==0){

    free(M2G_JOB_EGAC);

    for (i=0; i<EGAC_Num; i++){
      free(L2L_ONAN[i]);
    }
    free(L2L_ONAN);
  }

  num3 = 0;
  for (ID=0; ID<numprocs; ID++){
    num3 += Num_Rcv_GA_EGAC[ID];
  }
  Num_GA_EGAC = num3; 

  M2G_JOB_EGAC = (int*)malloc(sizeof(int)*(num3+1));
  L2L_ONAN = (int**)malloc(sizeof(int*)*EGAC_Num);
  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];
    L2L_ONAN[job_id] = (int*)malloc(sizeof(int)*(ONAN[Gc_AN]+1));
  }

  k = 0;
  for (ID=0; ID<numprocs; ID++){
    for (i=0; i<Num_Rcv_GA_EGAC[ID]; i++){
  
      if (i==0) Top_Index_GA_EGAC[ID] = k;
      M2G_JOB_EGAC[k] = Indx_Rcv_GA_EGAC[ID][i]; /* global job_id */
      job_id = Indx_JOB[ID][i];                  /* local job_id */
      h_AN = Indx_h_AN[ID][i];

      /* h_AN is for ONAN[Gc_AN], and k is the first index of GA_EGAC */
      L2L_ONAN[job_id][h_AN] = k;
  
      k++; 
    }
  }

  /********************************************************
            determine the data size in MPI of GA 
  ********************************************************/

  /*
  for (ID=0; ID<numprocs; ID++){
    printf("ABC1 myid=%2d ID=%2d  %2d %2d\n",myid,ID,Num_Snd_GA_EGAC[ID],Num_Rcv_GA_EGAC[ID]);
  }
  */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    tag = 999;

    /* find data size to send block data */

    if ( Num_Snd_GA_EGAC[IDS]!=0 ){

      size1 = 0;

      for (m=0; m<Num_Snd_GA_EGAC[IDS]; m++){

        n = Indx_Snd_GA_EGAC[IDS][m]; /* n: job_id */
        GN2N_EGAC(n,N3);

        Gc_AN = N3[1];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_CNO[Cwan];

	for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_CNO[Hwan];
	  size1 += tno1*tno2;
	}
      }

      Snd_GA_EGAC_Size[IDS] = size1;

      if (ID!=0) MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
    }
    else{
      Snd_GA_EGAC_Size[IDS] = 0;
    }

    /* receiving of size of data */

    if ( Num_Rcv_GA_EGAC[IDR]!=0 ){

      if (ID!=0){ 
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_GA_EGAC_Size[IDR] = size2;
      }
      else{
        Rcv_GA_EGAC_Size[IDR] = Snd_GA_EGAC_Size[IDS];
      } 
    }
    else{
      Rcv_GA_EGAC_Size[IDR] = 0;
    }

    if ( Num_Snd_GA_EGAC[IDS]!=0 && ID!=0 ) MPI_Wait(&request,&stat);
  }

  /* find Max_Snd_GA_EGAC_Size among Snd_GA_EGAC_Size 
     find Max_Rcv_GA_EGAC_Size among Rcv_GA_EGAC_Size */

  Max_Snd_GA_EGAC_Size = 0;
  Max_Rcv_GA_EGAC_Size = 0;
  for (ID=0; ID<numprocs; ID++){
    if (Max_Snd_GA_EGAC_Size<Snd_GA_EGAC_Size[ID]) Max_Snd_GA_EGAC_Size = Snd_GA_EGAC_Size[ID];
    if (Max_Rcv_GA_EGAC_Size<Rcv_GA_EGAC_Size[ID]) Max_Rcv_GA_EGAC_Size = Rcv_GA_EGAC_Size[ID];
  }

  /* allocation of dim_IA_EGAC */
  /* note that freeing of dim_IA_EGAC is done in free_arrays_truncation0() */

  dim_IA_EGAC = (int*)malloc(sizeof(int)*EGAC_Num);

  /* find dim_IA_EGAC and Max_dim_GA_EGAC */

  Max_dim_GA_EGAC = 0;
  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];

    size1 = 0;
    for (i=0; i<ONAN[Gc_AN]; i++){
      ig = natn_onan[Gc_AN][i];
      Cwan = WhatSpecies[ig]; 
      size1 += Spe_Total_CNO[Cwan];
    }

    dim_IA_EGAC[job_id] = size1;

    if (Max_dim_GA_EGAC<size1) Max_dim_GA_EGAC = size1; 
  }

  /* allocation of dim_GD_EGAC */
  /* note that freeing of dim_GD_EGAC is done in free_arrays_truncation0() */

  dim_GD_EGAC = (int*)malloc(sizeof(int)*EGAC_Num);

  /* find dim_GD_EGAC and Max_dim_GD_EGAC */

  Max_dim_GD_EGAC = 0;
  for (job_id=0; job_id<EGAC_Num; job_id++){

    job_gid = job_id + EGAC_Top[myid];
    GN2N_EGAC(job_gid,N3);
    Gc_AN = N3[1];

    size1 = 0;
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      Gi = natn[Gc_AN][i];
      Cwan = WhatSpecies[Gi];
      size1 += Spe_Total_CNO[Cwan];
    }

    dim_GD_EGAC[job_id] = size1;

    if (Max_dim_GD_EGAC<size1) Max_dim_GD_EGAC = size1; 
  }

  /* allocation of Sigma_EGAC */
  /* note that freeing of Sigma_EGAC is done in free_arrays_truncation0() */

  Sigma_EGAC = (dcomplex***)malloc(sizeof(dcomplex**)*DIIS_History_EGAC);
  for (m=0; m<DIIS_History_EGAC; m++){
    Sigma_EGAC[m] = (dcomplex**)malloc(sizeof(dcomplex*)*EGAC_Num);
    for (job_id=0; job_id<EGAC_Num; job_id++){
      Sigma_EGAC[m][job_id] = (dcomplex*)malloc(sizeof(dcomplex)*dim_GD_EGAC[job_id]*dim_GD_EGAC[job_id]);
      for (i=0; i<dim_GD_EGAC[job_id]*dim_GD_EGAC[job_id]; i++){
	Sigma_EGAC[m][job_id][i].r = 0.0;
	Sigma_EGAC[m][job_id][i].i = 0.0;
      }
    }  
  }

  /* allocation of fGD_EGAC */
  /* note that freeing of fGD_EGAC is done in free_arrays_truncation0() */

  fGD_EGAC = (dcomplex**)malloc(sizeof(dcomplex*)*(EGAC_Num+1));
  for (job_id=0; job_id<EGAC_Num; job_id++){
    fGD_EGAC[job_id] = (dcomplex*)malloc(sizeof(dcomplex)*dim_GD_EGAC[job_id]*dim_GD_EGAC[job_id]);
    for (i=0; i<dim_GD_EGAC[job_id]*dim_GD_EGAC[job_id]; i++){
      fGD_EGAC[job_id][i].r = 0.0;
      fGD_EGAC[job_id][i].i = 0.0;
    }
  }  
  fGD_EGAC[EGAC_Num] = (dcomplex*)malloc(sizeof(dcomplex)*Max_dim_GD_EGAC*Max_dim_GD_EGAC);

  /* allocation of GD_EGAC */
  /* note that freeing of GD_EGAC is done in free_arrays_truncation0() */

  GD_EGAC = (dcomplex****)malloc(sizeof(dcomplex***)*EGAC_Num);
  for (job_id=0; job_id<EGAC_Num; job_id++){ /* job_id: local job_id */

    job_gid = job_id + EGAC_Top[myid]; /* job_gid: global job_id */
    GN2N_EGAC(job_gid,N3);

    Gc_AN = N3[1];
    Cwan = WhatSpecies[Gc_AN]; 
    tno1 = Spe_Total_CNO[Cwan];

    GD_EGAC[job_id] = (dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1));
    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];        
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];

      GD_EGAC[job_id][h_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno1);
      for (i=0; i<tno1; i++){
	GD_EGAC[job_id][h_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno2);
	for (j=0; j<tno2; j++){
	  GD_EGAC[job_id][h_AN][i][j].r = 0.0;
	  GD_EGAC[job_id][h_AN][i][j].i = 0.0;
	}
      }
    } 
  }

  /* allocation of GA_EGAC */
  /* note that freeing of GA_EGAC is done in free_arrays_truncation0() */

  GA_EGAC = (dcomplex****)malloc(sizeof(dcomplex***)*Num_GA_EGAC);
  for (k=0; k<Num_GA_EGAC; k++){       /* k is the first index of GA_EGAC */

    job_gid = M2G_JOB_EGAC[k];
    GN2N_EGAC(job_gid,N3);

    Gc_AN = N3[1];
    Cwan = WhatSpecies[Gc_AN]; 
    tno1 = Spe_Total_CNO[Cwan];

    GA_EGAC[k] = (dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1));
    for (h_AN=0; h_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];        
      Hwan = WhatSpecies[Gh_AN];
      tno2 = Spe_Total_CNO[Hwan];

      GA_EGAC[k][h_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno1);
      for (i=0; i<tno1; i++){
        GA_EGAC[k][h_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno2);
      }
    } 
  }

  /* set alloc_first[34] */

  alloc_first[34] = 0;

  /* freeing of arrays */

  free(atom_required_flag);

  for (i=0; i<numprocs; i++){
    free(Indx_JOB[i]);
  }
  free(Indx_JOB);

  for (i=0; i<numprocs; i++){
    free(Indx_h_AN[i]);
  }
  free(Indx_h_AN);

}



void Set_up_for_DCLNO()
{
  int numprocs0,myid0;
  int numprocs1,myid1;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  /* making new comm worlds. */

  if (atomnum<=numprocs0 && alloc_first[35]==1){

    Num_Comm_World1_DCLNO = atomnum;

    NPROCS_ID1_DCLNO = (int*)malloc(sizeof(int)*numprocs0); 
    Comm_World1_DCLNO = (int*)malloc(sizeof(int)*numprocs0); 
    NPROCS_WD1_DCLNO = (int*)malloc(sizeof(int)*Num_Comm_World1_DCLNO); 
    Comm_World_StartID1_DCLNO = (int*)malloc(sizeof(int)*Num_Comm_World1_DCLNO); 
    MPI_CommWD1_DCLNO = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1_DCLNO);

    Make_Comm_Worlds( mpi_comm_level1, myid0, numprocs0, Num_Comm_World1_DCLNO, &myworld1_DCLNO, MPI_CommWD1_DCLNO, 
                      NPROCS_ID1_DCLNO, Comm_World1_DCLNO, NPROCS_WD1_DCLNO, Comm_World_StartID1_DCLNO); 

    /* MPI_CommWD1_DCLNO[myworld1_DCLNO] is further divided depending on spin polarization. */

    MPI_Comm_size( MPI_CommWD1_DCLNO[myworld1_DCLNO], &numprocs1);
    MPI_Comm_rank( MPI_CommWD1_DCLNO[myworld1_DCLNO], &myid1);

    /* MPI_CommWD2_DCLNO */

    if      (SpinP_switch==0) Num_Comm_World2_DCLNO = 1;
    else if (SpinP_switch==1) Num_Comm_World2_DCLNO = 2;
    else if (SpinP_switch==3) Num_Comm_World2_DCLNO = 1;

    NPROCS_ID2_DCLNO = (int*)malloc(sizeof(int)*numprocs1); 
    Comm_World2_DCLNO = (int*)malloc(sizeof(int)*numprocs1); 
    NPROCS_WD2_DCLNO = (int*)malloc(sizeof(int)*Num_Comm_World2_DCLNO); 
    Comm_World_StartID2_DCLNO = (int*)malloc(sizeof(int)*Num_Comm_World2_DCLNO); 
    MPI_CommWD2_DCLNO = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2_DCLNO);
     
    Make_Comm_Worlds( MPI_CommWD1_DCLNO[myworld1_DCLNO], myid1, numprocs1, Num_Comm_World2_DCLNO, 
                      &myworld2_DCLNO, MPI_CommWD2_DCLNO, NPROCS_ID2_DCLNO, Comm_World2_DCLNO, 
                      NPROCS_WD2_DCLNO, Comm_World_StartID2_DCLNO);

    /* alloc_first[35] = 0 */
    alloc_first[35] = 0;
  }

  else if (alloc_first[35]==1){

    Num_Comm_World2_DCLNO = numprocs0;

    NPROCS_ID2_DCLNO = (int*)malloc(sizeof(int)*numprocs0); 
    Comm_World2_DCLNO = (int*)malloc(sizeof(int)*numprocs0); 
    NPROCS_WD2_DCLNO = (int*)malloc(sizeof(int)*Num_Comm_World2_DCLNO); 
    Comm_World_StartID2_DCLNO = (int*)malloc(sizeof(int)*Num_Comm_World2_DCLNO); 
    MPI_CommWD2_DCLNO = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2_DCLNO);

    Make_Comm_Worlds( mpi_comm_level1, myid0, numprocs0, Num_Comm_World2_DCLNO, 
                      &myworld2_DCLNO, MPI_CommWD2_DCLNO, NPROCS_ID2_DCLNO, Comm_World2_DCLNO, 
                      NPROCS_WD2_DCLNO, Comm_World_StartID2_DCLNO);

    /* alloc_first[35] = 0 */
    alloc_first[35] = 0;
  }

}
