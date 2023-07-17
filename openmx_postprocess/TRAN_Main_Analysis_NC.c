/**********************************************************************
  TRAN_Main_Analysis_NC.c:

  TRAN_Main_Analysis_NC.c is a subroutine to analyze transport properties
  such as electronic transmission, current, eigen channel, and current 
  distribution in real space based on the non-colliear density functional 
  theories and the non-equilibrium Green's function method.

  Log of TRAN_Main_Analysis_NC.c:

     15/Dec./2011  released by Y. Xiao
     2/June/2015  integrated in OpenMX by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "Inputtools.h"
#include "mpi.h"
#include "tran_prototypes.h"

#define eV2Hartree    27.2113845                
#define kB            0.00008617251324000000   /* eV/K  */          
#define PI            3.1415926535897932384626

#define Host_ID  0
#define PrintLevel  0

/*
#define SCC_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define SCL_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define SCR_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define S00l_ref(i,j) ( ((j)-1)*NUM_e[0]+(i)-1 )
#define S00r_ref(i,j) ( ((j)-1)*NUM_e[1]+(i)-1 )
*/

static int SpinP_switch,SpinP_switch2;
static int NUM_c, NUM_e[2];
static double E_Temp;

/* the center region */

static double ChemP;
static double *****OLP;
static double *****H;
static double *****iHNL;
static double *****H0;
static int atomnum;
static int SpeciesNum;
static int *WhatSpecies;
static int *Spe_Total_CNO;
static int *FNAN;
static int **natn;
static int **ncn;
static int **atv_ijk; 
static int Max_FSNAN;
static double ScaleSize;
static int TCpyCell;
static int *TRAN_region;
static int *TRAN_Original_Id;

/* the leads region */

static double ChemP_e[2];
static double *****OLP_e[2];
static double *****H_e[2];
static double *****iHNL_e[2]; 
static int atomnum_e[2];
static int SpeciesNum_e[2];
static int *WhatSpecies_e[2];
static int *Spe_Total_CNO_e[2];
static int *FNAN_e[2];
static int **natn_e[2];
static int **ncn_e[2];
static int **atv_ijk_e[2]; 
static int Max_FSNAN_e[2];
static double ScaleSize_e[2];
static int TCpyCell_e[2];

/* k-dependent matrices */

static dcomplex *S00_e[2];
static dcomplex *S01_e[2];
static dcomplex *H00_e[2];
static dcomplex *H01_e[2];

static dcomplex *SCC;
static dcomplex *SCL;
static dcomplex *SCR;
static dcomplex *HCC;
static dcomplex *HCL;
static dcomplex *HCR;

static dcomplex ****tran_transmission;
static dcomplex **tran_transmission_iv;

static int tran_surfgreen_iteration_max;
static double tran_surfgreen_eps;

static int Tran_current_num_step;
static double Tran_current_energy_step,Tran_current_cutoff;
static double Tran_current_lower_bound,Tran_current_im_energy;

static int SCF_tran_bias_apply;
static int tran_transmission_on;
static double tran_transmission_energyrange[3];
static int tran_transmission_energydiv;
static int tran_interpolate;
static char interpolate_filename1[YOUSO10];
static char interpolate_filename2[YOUSO10];
static double interpolate_c1,interpolate_c2;
static int TRAN_TKspace_grid2,TRAN_TKspace_grid3;
static double ***current;
static int Order_Lead_Side[2];
  
static char filepath[100];
static char filename[100];

/* S MitsuakiKAWAMURA */

/* Eigenchannel analysis */
static int TRAN_Channel;
static int TRAN_CurrentDensity;
static int TRAN_OffDiagonalCurrent;
static int TRAN_Channel_Num;
static int TRAN_Channel_Nkpoint;
static int TRAN_Channel_Nenergy;
static double **TRAN_Channel_kpoint;
static double *TRAN_Channel_energy;

/* Current Density */
static dcomplex *VCC;
static double ****JLocSym, ***JLocASym, ***RhoNL, ****Jmat;

#define BUFSIZE 200
/* E MitsuakiKAWAMURA */
void Make_Comm_Worlds(
   MPI_Comm MPI_Curret_Comm_WD,   
   int myid0,
   int numprocs0,
   int Num_Comm_World, 
   int *myworld1, 
   MPI_Comm *MPI_CommWD,     /* size: Num_Comm_World */
   int *NPROCS1_ID,          /* size: numprocs0 */
   int *Comm_World1,         /* size: numprocs0 */
   int *NPROCS1_WD,          /* size: Num_Comm_World */
   int *Comm_World_StartID   /* size: Num_Comm_World */
   );


static void MTRAN_Read_Tran_HS(MPI_Comm comm1, char *filepath, char *filename, char *ext ) ;

static void MTRAN_Transmission(
                        MPI_Comm comm1,
                        int numprocs,
                        int myid,
			int SpinP_switch,
                        double ChemP_e[2],
			int NUM_c,
			int NUM_e[2],
			dcomplex *H00_e[2],
			dcomplex *S00_e[2],
			dcomplex *H01_e[2],
			dcomplex *S01_e[2],
			dcomplex *HCC,
			dcomplex *HCL,
			dcomplex *HCR,
			dcomplex *SCC,
			dcomplex *SCL,
			dcomplex *SCR, 
			double tran_surfgreen_iteration_max,
			double tran_surfgreen_eps, 
			double tran_transmission_energyrange[3],
			int tran_transmission_energydiv, 
			dcomplex **tran_transmission);

/* S MitsuakiKAWAMURA*/

static void MTRAN_Current(
                   MPI_Comm comm1,
                   int numprocs,
                   int myid,
		   int SpinP_switch,
		   double ChemP_e[2],
                   double E_Temp,
		   int NUM_c,
		   int NUM_e[2],
		   dcomplex *H00_e[2],
		   dcomplex *S00_e[2],
		   dcomplex *H01_e[2],
		   dcomplex *S01_e[2],
		   dcomplex *HCC,
		   dcomplex *HCL,
		   dcomplex *HCR,
		   dcomplex *SCC,
		   dcomplex *SCL,
		   dcomplex *SCR, 
		   double tran_surfgreen_iteration_max,
		   double tran_surfgreen_eps, 
                   double *current,
  double k2,
  double k3);

void TRAN_Calc_Sinv(
  int NUM_c,
  dcomplex *SCC,
  dcomplex *Sinv);

void TRAN_Calc_CurrentDensity_NC(
  int NUM_c,
  dcomplex *GC,
  dcomplex *SigmaL,
  dcomplex *SigmaR,
  dcomplex *VCC,
  dcomplex *Sinv,
  double *kvec,
  double fL,
  double fR,
  double Tran_current_energy_step,
  double ****JLocSym,
  double ***JLocAsym,
  double ***RhoNLNL,
  double ****Jmat
  );

void TRAN_CDen_Main(
  int NUM_c,
  int *MP,
  double ****JLocSym,
  double ***JLocASym,
  double ***Rho,
  double ****Jmat,
  dcomplex *SCC,
  int TRAN_OffDiagonalCurrent);

void MTRAN_EigenChannel_NC(
        MPI_Comm comm1,
        int numprocs,
        int myid,
        int myid0,
        int SpinP_switch,
        double ChemP_e[2],
        int NUM_c,
        int NUM_e[2],
        dcomplex *H00_e[2],
        dcomplex *S00_e[2],
        dcomplex *H01_e[2],
        dcomplex *S01_e[2],
        dcomplex *HCC,
        dcomplex *HCL,
        dcomplex *HCR,
        dcomplex *SCC,
        dcomplex *SCL,
        dcomplex *SCR,
        double tran_surfgreen_iteration_max,
        double tran_surfgreen_eps,
        double tran_transmission_energyrange[3],
        int TRAN_Channel_Nenergy,
        double *TRAN_Channel_energy,
        int TRAN_Channel_Num,
        int kloop,
        double *TRAN_Channel_kpoint,
        dcomplex ***EChannel,
        double **eigentrans,
        double **eigentrans_sum
        ); /* void MTRAN_EigenChannel_NC */

void TRAN_Output_eigentrans_sum(
  int TRAN_Channel_Nkpoint,
  int TRAN_Channel_Nenergy,
  double ***eigentrans_sum);

void TRAN_Output_ChannelCube(
  int kloop,
  int iw,
  int ispin,
  int orbit,
  int NUM_c,
  double *TRAN_Channel_kpoint,
  dcomplex *EChannel,
  int *MP,
  double eigentrans,
  double TRAN_Channel_energy
  ); /* void TRAN_Output_ChannelCube */

static void MTRAN_Free_All_NC();
/* E MitsuakiKAWAMURA */

static void MTRAN_Output_Transmission(
        MPI_Comm comm1,
        int ID,
        char *fname,
        double k2,
        double k3,
        int SpinP_switch,
        int tran_transmission_energydiv,
        double tran_transmission_energyrange[3],
        dcomplex **tran_transmission
        );

static void MTRAN_Output_Current(
        MPI_Comm comm1,
        char *fname,
        int TRAN_TKspace_grid2,
        int TRAN_TKspace_grid3,
        int SpinP_switch,
        double ***current
        );

static void MTRAN_Output_Conductance(
        MPI_Comm comm1,
        char *fname,
        int *T_k_ID,
        int T_knum,
        int TRAN_TKspace_grid2,
        int TRAN_TKspace_grid3,
        int *T_IGrids2, 
        int *T_IGrids3,
        double *T_KGrids2,
        double *T_KGrids3,
        int SpinP_switch,
        int tran_transmission_energydiv,
        double tran_transmission_energyrange[3],
        dcomplex ****tran_transmission
        );

static void MTRAN_Input(
                 MPI_Comm comm1,
                 int argc,
                 char *fname,
                 double ChemP_e[2],
                 int *TRAN_TKspace_grid2,
                 int *TRAN_TKspace_grid3,
		 int *SpinP_switch,
		 double *E_Temp,
		 int *tran_surfgreen_iteration_max,
		 double *tran_surfgreen_eps,
		 int  *tran_transmission_on,
		 double tran_transmission_energyrange[3],
		 int *tran_transmission_energydiv,
		 dcomplex ****(*tran_transmission)
		 );


static void MTRAN_Input_Sys(
                     int argc,char *file, char *filepath, char *filename,
                     int *tran_interpolate, 
                     char *interpolate_filename1,
                     char *interpolate_filename2);

static void MTRAN_Set_MP(
        int job, 
        int anum, int *WhatSpecies, int *Spe_Total_CNO, 
        int *NUM,  /* output */
        int *MP    /* output */
	);

static void MTRAN_Set_SurfOverlap(
                           char *position, 
                           double k2,
                           double k3,
                           int SpinP_switch,
                           int atomnum_e[2],
                           double *****OLP_e[2],
                           double *****H_e[2],
                           double *****iHNL_e[2],
                           int SpeciesNum_e[2], 
                           int *WhatSpecies_e[2], 
                           int *Spe_Total_CNO_e[2], 
                           int *FNAN_e[2],
                           int **natn_e[2], 
                           int **ncn_e[2], 
                           int **atv_ijk_e[2],
			   dcomplex *S00_e[2],
			   dcomplex *S01_e[2],
                           dcomplex *H00_e[2],
			   dcomplex *H01_e[2]
                           );

static void MTRAN_Set_CentOverlap( 
			   int job, 
			   int SpinP_switch, 
			   double k2,
			   double k3,
			   int NUM_c,
			   int NUM_e[2],
			   double *****H, 
                           double *****iHNL, 
			   double *****OLP,
			   int atomnum,
			   int atomnum_e[2],
			   int *WhatSpecies,
			   int *WhatSpecies_e[2],
			   int *Spe_Total_CNO,
			   int *Spe_Total_CNO_e[2],
			   int *FNAN,
			   int **natn,
			   int **ncn, 
			   int **atv_ijk,
			   int *TRAN_region,
			   int *TRAN_Original_Id 
			   );

static void MTRAN_Allocate_HS(
                       int NUM_c,
                       int NUM_e[2],
                       int SpinP_switch);





void TRAN_Main_Analysis_NC( MPI_Comm comm1, 
                            int argc, char *argv[], 
                            int Matomnum, int *M2G, 
                            int *GridN_Atom, 
                            int **GridListAtom,
                            int **CellListAtom,
                            Type_Orbs_Grid ***Orbs_Grid,
                            int TNumGrid ) 
{
  int i,j,i2,i3,ii2,ii3,k,iw;
  int Gc_AN,h_AN,Gh_AN;
  int iside,tno0,tno1,Cwan,Hwan;
  int kloop0,kloop,S_knum,E_knum,k_op;
  /* S MitsuakiKAWAMURA */
  int myworld1, myworld2, parallel_mode, num_kloop0;
  int myid0, numprocs0, myid1, myid2, numprocs1, numprocs2;
  /* E MitsuakiKAWAMURA */
  int ID, myid_tmp, numprocs_tmp, T_knum;
  int **op_flag,*T_op_flag,*T_k_ID;
  double *T_KGrids2,*T_KGrids3;
  int *T_IGrids2,*T_IGrids3;
  /* S MitsuakiKAWAMURA */
  int Num_Comm_World1, Num_Comm_World2;
  int *NPROCS_ID1, *NPROCS_ID2;
  int *Comm_World1, *Comm_World2;
  int *NPROCS_WD1, *NPROCS_WD2;
  int *Comm_World_StartID1, *Comm_World_StartID2;
  MPI_Comm *MPI_CommWD1, *MPI_CommWD2;
  double ***eigentrans;
  double ***eigentrans_sum;
  dcomplex ****EChannel;
  int *MP;
  int NUM_cs;
  /* E MitsuakiKAWAMURA */
  double k2,k3,tmp;
  MPI_Comm comm_tmp;
  char fnameout[100];

  MPI_Comm_size(comm1,&numprocs0);
  MPI_Comm_rank(comm1,&myid0);

  /**********************************************
                 show something 
  **********************************************/

  if (myid0==Host_ID){  
    printf("\n*******************************************************\n"); 
    printf("*******************************************************\n"); 
    printf(" Welcome to TranMain_NC                                \n");
    printf(" This is a post-processing code of OpenMX to calculate \n");
    printf(" electronic transmission and current.                  \n");
    printf(" Copyright (C), 2002-2013, H.Kino and T.Ozaki          \n"); 
    printf(" TranMain_NC comes with ABSOLUTELY NO WARRANTY.         \n"); 
    printf(" This is free software, and you are welcome to         \n"); 
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("*******************************************************\n"); 
    printf("*******************************************************\n\n"); 
  } 

  /**********************************************
                   read system
  **********************************************/

  MTRAN_Input_Sys(argc,argv[1],filepath,filename,&tran_interpolate,interpolate_filename1,interpolate_filename2);

  /**********************************************
                 read tranb file
  **********************************************/

  MTRAN_Read_Tran_HS(comm1, filepath, filename, "tranb" );

  /**********************************************
                 read input file
  **********************************************/

  MTRAN_Input(comm1, 
              argc,
              argv[1],
              ChemP_e, 
              /* output */
              &TRAN_TKspace_grid2,
              &TRAN_TKspace_grid3,
	      &SpinP_switch2,
              &E_Temp, 
	      &tran_surfgreen_iteration_max,
	      &tran_surfgreen_eps,
	      &tran_transmission_on,
	      tran_transmission_energyrange,
	      &tran_transmission_energydiv,
	      &tran_transmission );

  if (SpinP_switch!=SpinP_switch2) {
     printf("SpinP_switch conflicts\n");fflush(stdout);
     printf("SpinP_switch=%d  SpinP_switch2=%d\n", SpinP_switch,SpinP_switch2);fflush(stdout);
     exit(0); 
  }

  MTRAN_Allocate_HS(NUM_c,NUM_e,SpinP_switch);

  /**********************************************
              calculate transmission
  **********************************************/

  if (tran_transmission_on) {

    /* allocation of arrays */

    current = (double***)malloc(sizeof(double**)*TRAN_TKspace_grid2); 
    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      current[i2] = (double**)malloc(sizeof(double*)*TRAN_TKspace_grid3); 
      for (i3=0; i3<TRAN_TKspace_grid3; i3++){
        current[i2][i3] = (double*)malloc(sizeof(double)*3);
      }
    }

    op_flag = (int**)malloc(sizeof(int*)*TRAN_TKspace_grid2); 
    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      op_flag[i2] = (int*)malloc(sizeof(int)*TRAN_TKspace_grid3); 
      for (i3=0; i3<TRAN_TKspace_grid3; i3++){
        op_flag[i2][i3] = 1;
      }
    }

    /***********************************
         one-dimentionalize for MPI
    ************************************/

    T_knum = 0;
    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      for (i3=0; i3<TRAN_TKspace_grid3; i3++){
	if (0<op_flag[i2][i3]) T_knum++;  
      }
    }         

    T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
    T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
    T_IGrids2 = (int*)malloc(sizeof(int)*T_knum);
    T_IGrids3 = (int*)malloc(sizeof(int)*T_knum);
    T_op_flag = (int*)malloc(sizeof(int)*T_knum);
    T_k_ID = (int*)malloc(sizeof(int)*T_knum);

    T_knum = 0;

    for (i2=0; i2<TRAN_TKspace_grid2; i2++){

      k2 = -0.5 + (2.0*(double)i2+1.0)/(2.0*(double)TRAN_TKspace_grid2) + Shift_K_Point;

      for (i3=0; i3<TRAN_TKspace_grid3; i3++){

	k3 = -0.5 + (2.0*(double)i3+1.0)/(2.0*(double)TRAN_TKspace_grid3) - Shift_K_Point;

	if (0<op_flag[i2][i3]){  

	  T_KGrids2[T_knum] = k2;
	  T_KGrids3[T_knum] = k3;
	  T_IGrids2[T_knum] = i2;
	  T_IGrids3[T_knum] = i3;
	  T_op_flag[T_knum] = op_flag[i2][i3];

	  T_knum++;        
	}
      }
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

    /***********************************************************
     start "kloop0"
    ***********************************************************/

    /* S MitsuakiKAWAMURA 2*/
    NUM_cs = NUM_c / 2;

    if (fabs(ChemP_e[0] - ChemP_e[1]) < 0.000001) TRAN_CurrentDensity = 0;
    
    JLocSym = (double****)malloc(sizeof(double***) * (SpinP_switch + 1));
    Jmat = (double****)malloc(sizeof(double***) * (SpinP_switch + 1));
    JLocASym = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    RhoNL = (double***)malloc(sizeof(double**) * (SpinP_switch + 1));
    for (k = 0; k < SpinP_switch + 1; k++) {
      JLocSym[k] = (double***)malloc(sizeof(double**) * 3);
      for (iside = 0; iside < 3; iside++) {
        JLocSym[k][iside] = (double**)malloc(sizeof(double*) * NUM_cs);
        for (i = 0; i < NUM_cs; i++) {
          JLocSym[k][iside][i] = (double*)malloc(sizeof(double) * NUM_cs);
          for (j = 0; j < NUM_cs; j++) JLocSym[k][iside][i][j] = 0.0;
        } /* for (i = 0; i < NUM_cs; i++) */
      } /* for (iside = 0; iside < 3; iside++) */

      Jmat[k] = (double***)malloc(sizeof(double**) * 2);
      for (iside = 0; iside < 2; iside++) {
        Jmat[k][iside] = (double**)malloc(sizeof(double*) * NUM_cs);
        for (i = 0; i < NUM_cs; i++) {
          Jmat[k][iside][i] = (double*)malloc(sizeof(double) * NUM_cs);
          for (j = 0; j < NUM_cs; j++) Jmat[k][iside][i][j] = 0.0;
        } /* for (i = 0; i < NUM_cs; i++) */
      } /* for (iside = 0; iside < 2; iside++) */

      JLocASym[k] = (double**)malloc(sizeof(double*) * NUM_cs);
      RhoNL[k] = (double**)malloc(sizeof(double*) * NUM_cs);
      for (i = 0; i < NUM_cs; i++) {
        JLocASym[k][i] = (double*)malloc(sizeof(double) * NUM_cs);
        RhoNL[k][i] = (double*)malloc(sizeof(double) * NUM_cs);
        for (j = 0; j < NUM_cs; j++) {
          JLocASym[k][i][j] = 0.0;
          RhoNL[k][i][j] = 0.0;
        } /* for (j = 0; j < NUM_cs; j++) */
      } /* for (i = 0; i < NUM_cs; i++) */
    } /* for (k = 0; k < SpinP_switch + 1; k++) */
      /* E MitsuakiKAWAMURA 2*/

    if (myid0==Host_ID) printf("\n  calculating...\n\n"); fflush(stdout);
    MPI_Barrier(comm1);

    for (kloop0=0; kloop0<num_kloop0; kloop0++){

      kloop = S_knum + kloop0;

      k2 = T_KGrids2[kloop];
      k3 = T_KGrids3[kloop];
      i2 = T_IGrids2[kloop];
      i3 = T_IGrids3[kloop];
      k_op = T_op_flag[kloop];

      printf("  myid0=%5d i2=%2d i3=%2d  k2=%8.4f k3=%8.4f\n",myid0,i2,i3,k2,k3); fflush(stdout);

      if (parallel_mode){
        comm_tmp = MPI_CommWD1[myworld1];
        numprocs_tmp = numprocs1;
        myid_tmp = myid1;
      }
      else{
        comm_tmp = comm1;
        numprocs_tmp = 1;
        myid_tmp = 0;
      }

      /* set Hamiltonian and overlap matrices of left and right leads */

      MTRAN_Set_SurfOverlap( "left", k2,k3,SpinP_switch,atomnum_e,
			     OLP_e,H_e,iHNL_e,SpeciesNum_e,WhatSpecies_e,Spe_Total_CNO_e,
			     FNAN_e,natn_e,ncn_e,atv_ijk_e,S00_e,S01_e,H00_e,H01_e );
  
      MTRAN_Set_SurfOverlap( "right", k2,k3,SpinP_switch,atomnum_e,
			     OLP_e,H_e,iHNL_e,SpeciesNum_e,WhatSpecies_e,Spe_Total_CNO_e,
			     FNAN_e,natn_e,ncn_e,atv_ijk_e,S00_e,S01_e,H00_e,H01_e );

      /* set CC, CL and CR */

      MTRAN_Set_CentOverlap(3,
			    SpinP_switch, 
			    k2,
			    k3,
			    NUM_c,
			    NUM_e,
			    H,
                            iHNL,
			    OLP,
			    atomnum,
			    atomnum_e,
			    WhatSpecies,
			    WhatSpecies_e,
			    Spe_Total_CNO,
			    Spe_Total_CNO_e,
			    FNAN,
			    natn,
			    ncn,
			    atv_ijk,
			    TRAN_region,
			    TRAN_Original_Id
			    );

      /* calculate transmission */

      MTRAN_Transmission(comm_tmp,
                         numprocs_tmp,
                         myid_tmp, 
			 SpinP_switch, 
			 ChemP_e,
			 NUM_c,
			 NUM_e,
			 H00_e,
			 S00_e,
			 H01_e,
			 S01_e,
			 HCC,
			 HCL,
			 HCR,
			 SCC,
			 SCL,
			 SCR,
			 tran_surfgreen_iteration_max,
			 tran_surfgreen_eps,
			 tran_transmission_energyrange,
			 tran_transmission_energydiv,
			 /* output */
			 tran_transmission[i2][i3]
			 );

      /* calculate current */

      MTRAN_Current(comm_tmp,
                    numprocs_tmp,
                    myid_tmp,
                    SpinP_switch,
                    ChemP_e,
                    E_Temp,
                    NUM_c,
                    NUM_e,
                    H00_e,
                    S00_e,
                    H01_e,
                    S01_e,
                    HCC,
                    HCL,
                    HCR,
                    SCC,
                    SCL,
                    SCR,
                    tran_surfgreen_iteration_max,
                    tran_surfgreen_eps,
                    current[i2][i3],
        k2, k3
        );


    } /* kloop0 */
  } /* if (tran_transmission_on) */

  /**********************************************
       MPI:  current 
  **********************************************/

  for (k=0; k<T_knum; k++){

    ID = T_k_ID[k];

    i2 = T_IGrids2[k];
    i3 = T_IGrids3[k];

    MPI_Bcast(current[i2][i3], 2, MPI_DOUBLE, ID, comm1);
    MPI_Barrier(comm1);
    
  }

  /**********************************************
                output transmission
  **********************************************/

  if (tran_transmission_on) {

    MPI_Barrier(comm1);
    if (myid0==Host_ID) printf("\nTransmission:  files\n\n");fflush(stdout);
    MPI_Barrier(comm1);

    for (k=0; k<T_knum; k++){

      ID = T_k_ID[k];

      k2 = T_KGrids2[k];
      k3 = T_KGrids3[k];
      i2 = T_IGrids2[k];
      i3 = T_IGrids3[k];

      sprintf(fnameout,"%s%s.tran%i_%i",filepath,filename,i2,i3);

      MTRAN_Output_Transmission(
				comm1,
                                ID,
				fnameout,
				k2,
				k3,
				SpinP_switch, 
				tran_transmission_energydiv,
				tran_transmission_energyrange,
				tran_transmission[i2][i3]
				);

    }
 
  }

  /**********************************************
                   output current
  **********************************************/

  if (tran_transmission_on) {

    MPI_Barrier(comm1);
    if (myid0==Host_ID) printf("\nCurrent:  file\n\n");fflush(stdout);
    MPI_Barrier(comm1);

    sprintf(fnameout,"%s%s.current",filepath,filename);

    MTRAN_Output_Current(
			 comm1,
			 fnameout,
                         TRAN_TKspace_grid2,
                         TRAN_TKspace_grid3,
			 SpinP_switch, 
			 current
			 );

    if (myid0==Host_ID) printf("\n");fflush(stdout);
  }

  /**********************************************
                output conductance
  **********************************************/

  if (tran_transmission_on) {

    MPI_Barrier(comm1);
    if (myid0==Host_ID) printf("\nConductance:  file\n\n");fflush(stdout);
    MPI_Barrier(comm1);

    sprintf(fnameout,"%s%s.conductance",filepath,filename);

    MTRAN_Output_Conductance(
			     comm1,
			     fnameout,
                             T_k_ID,
                             T_knum,
			     TRAN_TKspace_grid2,
			     TRAN_TKspace_grid3,
                             T_IGrids2,
                             T_IGrids3,
                             T_KGrids2,
                             T_KGrids3,
			     SpinP_switch, 
			     tran_transmission_energydiv,
			     tran_transmission_energyrange,
			     tran_transmission);

    if (myid0==Host_ID) printf("\n");fflush(stdout);
  }

  /*S MitsuakiKAWAMURA2*/
  /********************************************
  Compute & Output Current density
  *********************************************/

  if (TRAN_CurrentDensity == 1) {

    if (myid0 == Host_ID) printf("\nCurrentdensity:  file\n\n"); fflush(stdout);

    MP = (int*)malloc(sizeof(int)*(atomnum + 1));
    MTRAN_Set_MP(1, atomnum, WhatSpecies, Spe_Total_CNO, &i, MP);

    for (k = 0; k < SpinP_switch + 1; k++) {
      for (iside = 0; iside < 3; iside++) {
        for (i = 0; i < NUM_cs; i++) {
          for (j = 0; j < NUM_cs; j++)
            JLocSym[k][iside][i][j] = JLocSym[k][iside][i][j]
            / (double)(TRAN_TKspace_grid2 * TRAN_TKspace_grid3);
          MPI_Allreduce(MPI_IN_PLACE, JLocSym[k][iside][i], NUM_cs,
            MPI_DOUBLE_PRECISION, MPI_SUM, comm1);
        } /* for (i = 0; i < NUM_cs; i++) */
      } /* for (iside = 0; iside < 3; iside++) */

      for (iside = 0; iside < 2; iside++) {
        for (i = 0; i < NUM_cs; i++) {
          for (j = 0; j < NUM_cs; j++)
            Jmat[k][iside][i][j] = Jmat[k][iside][i][j]
            / (double)(TRAN_TKspace_grid2 * TRAN_TKspace_grid3);
          MPI_Allreduce(MPI_IN_PLACE, Jmat[k][iside][i], NUM_cs,
            MPI_DOUBLE_PRECISION, MPI_SUM, comm1);
        } /* for (i = 0; i < NUM_cs; i++) */
      } /* for (iside = 0; iside < 2; iside++) */

      for (i = 0; i < NUM_cs; i++) {
        for (j = 0; j < NUM_cs; j++) {
          JLocASym[k][i][j] = JLocASym[k][i][j]
            / (double)(TRAN_TKspace_grid2 * TRAN_TKspace_grid3);
          RhoNL[k][i][j] = RhoNL[k][i][j]
            / (double)(TRAN_TKspace_grid2 * TRAN_TKspace_grid3);
        } /* for (j = 0; j < NUM_cs; j++) */
        MPI_Allreduce(MPI_IN_PLACE, JLocASym[k][i], NUM_cs,
          MPI_DOUBLE_PRECISION, MPI_SUM, comm1);
        MPI_Allreduce(MPI_IN_PLACE, RhoNL[k][i], NUM_cs,
          MPI_DOUBLE_PRECISION, MPI_SUM, comm1);
      } /* for (i = 0; i < NUM_cs; i++) */
    } /* for (k = 0; k < SpinP_switch + 1; k++) */

    TRAN_CDen_Main(NUM_cs, MP, JLocSym, JLocASym, RhoNL, Jmat, SCC, TRAN_OffDiagonalCurrent);
    free(MP);
  } /*if (TRAN_CurrentDensity == 1)*/

  for (k = 0; k < SpinP_switch + 1; k++) {
    for (iside = 0; iside < 3; iside++) {
      for (i = 0; i < NUM_cs; i++) {
        free(JLocSym[k][iside][i]);
      } /* for (i = 0; i < NUM_cs; i++) */
      free(JLocSym[k][iside]);
    } /* for (iside = 0; iside < 3; iside++) */
    free(JLocSym[k]);

    for (iside = 0; iside < 2; iside++) {
      for (i = 0; i < NUM_cs; i++) {
        free(Jmat[k][iside][i]);
      } /* for (i = 0; i < NUM_cs; i++) */
      free(Jmat[k][iside]);
    } /* for (iside = 0; iside < 2; iside++) */
    free(Jmat[k]);

    for (i = 0; i < NUM_cs; i++) {
      free(JLocASym[k][i]);
      free(RhoNL[k][i]);
    } /* for (i = 0; i < NUM_cs; i++) */
    free(JLocASym[k]);
    free(RhoNL[k]);
  } /* for (k = 0; k < SpinP_switch + 1; k++) */
  free(JLocSym);
  free(Jmat);
  free(JLocASym);
  free(RhoNL);
  /*E MitsuakiKAWAMURA2*/

  /* S MitsuakiKAWAMURA*/
  /********************************************
  Start EigenChannel Analysis
  *********************************************/

  /***********************************************************
  start "kloop0"
  ***********************************************************/
  if (TRAN_Channel_Nenergy * TRAN_Channel_Nkpoint <= 0) TRAN_Channel == 0;

  if (TRAN_Channel == 1){

    if (myid0 == Host_ID){
      printf("\n**************************************************\n");
      printf(" Calculation of transmission eigenchannels starts\n");
      printf("**************************************************\n\n");
      printf("  File index : *.traneval#k_#E_#spin *.tranevec#k_#E_#spin \n\n");
    }

    fflush(stdout);
    MPI_Barrier(comm1);

    eigentrans = (double***)malloc(sizeof(double**) * TRAN_Channel_Nkpoint);
    eigentrans_sum = (double***)malloc(sizeof(double**) * TRAN_Channel_Nkpoint);
    EChannel = (dcomplex****)malloc(sizeof(dcomplex***) * TRAN_Channel_Nkpoint);
    for (kloop = 0; kloop < TRAN_Channel_Nkpoint; kloop++){
      eigentrans[kloop] = (double**)malloc(sizeof(double*) * TRAN_Channel_Nenergy);
      eigentrans_sum[kloop] = (double**)malloc(sizeof(double*) * TRAN_Channel_Nenergy);
      EChannel[kloop] = (dcomplex***)malloc(sizeof(dcomplex**) * TRAN_Channel_Nenergy);
      for (iw = 0; iw < TRAN_Channel_Nenergy; iw++){
        eigentrans[kloop][iw] = (double*)malloc(sizeof(double) * (TRAN_Channel_Num + 1));
        eigentrans_sum[kloop][iw] = (double*)malloc(sizeof(double) * 1);
        eigentrans_sum[kloop][iw][0] = 0.0;
        EChannel[kloop][iw] = (dcomplex**)malloc(sizeof(dcomplex*) * (TRAN_Channel_Num + 1));
        for (i = 0; i < TRAN_Channel_Num + 1; i++){
          eigentrans[kloop][iw][i] = 0.0;
          EChannel[kloop][iw][i] = (dcomplex*)malloc(sizeof(dcomplex) * NUM_c);
          for (j = 0; j < NUM_c; j++){
            EChannel[kloop][iw][i][j].r = 0.0;
            EChannel[kloop][iw][i][j].i = 0.0;
          }
        }
      }
    }

    /* Set k parallelization mode*/

    free(T_k_ID);
    T_k_ID = (int*)malloc(sizeof(int)*TRAN_Channel_Nkpoint);

    if (numprocs0<TRAN_Channel_Nkpoint){

      /* set parallel_mode */
      parallel_mode = 0;

      /* allocation of kloop to ID */

      for (ID = 0; ID<numprocs0; ID++){

        tmp = (double)TRAN_Channel_Nkpoint / (double)numprocs0;
        S_knum = (int)((double)ID*(tmp + 1.0e-12));
        E_knum = (int)((double)(ID + 1)*(tmp + 1.0e-12)) - 1;
        if (ID == (numprocs0 - 1)) E_knum = TRAN_Channel_Nkpoint - 1;
        if (E_knum<0)          E_knum = 0;

        for (k = S_knum; k <= E_knum; k++){
          /* ID in the first level world */
          T_k_ID[k] = ID;
        }
      }

      /* find own informations */

      tmp = (double)TRAN_Channel_Nkpoint / (double)numprocs0;
      S_knum = (int)((double)myid0*(tmp + 1.0e-12));
      E_knum = (int)((double)(myid0 + 1)*(tmp + 1.0e-12)) - 1;
      if (myid0 == (numprocs0 - 1)) E_knum = TRAN_Channel_Nkpoint - 1;
      if (E_knum<0)             E_knum = 0;

      num_kloop0 = E_knum - S_knum + 1;

    } /* if (numprocs0<TRAN_Channel_Nkpoint) */
    else {

      /* set parallel_mode */
      parallel_mode = 1;
      num_kloop0 = 1;

      Num_Comm_World2 = TRAN_Channel_Nkpoint;

      NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs0);
      Comm_World2 = (int*)malloc(sizeof(int)*numprocs0);
      NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
      Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2);
      MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);

      Make_Comm_Worlds(comm1, myid0, numprocs0, Num_Comm_World2, &myworld2, MPI_CommWD2,
        NPROCS_ID2, Comm_World2, NPROCS_WD2, Comm_World_StartID2);

      MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
      MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

      S_knum = myworld2;

      /* allocate k-points into processors */

      for (k = 0; k<TRAN_Channel_Nkpoint; k++){
        /* ID in the first level world */
        T_k_ID[k] = Comm_World_StartID2[k];
      }

    } /* else if (numprocs0>=TRAN_Channel_Nkpoint) */

    for (kloop0 = 0; kloop0 < num_kloop0; kloop0++){

      kloop = kloop0 + S_knum;

      k2 = TRAN_Channel_kpoint[kloop][0];
      k3 = TRAN_Channel_kpoint[kloop][1];

      if (parallel_mode){
        comm_tmp = MPI_CommWD2[myworld2];
        numprocs_tmp = numprocs2;
        myid_tmp = myid2;
      }
      else{
        comm_tmp = comm1;
        numprocs_tmp = 1;
        myid_tmp = 0;
      }

      /* set Hamiltonian and overlap matrices of left and right leads */

      MTRAN_Set_SurfOverlap("left", k2, k3, SpinP_switch, atomnum_e,
        OLP_e, H_e, iHNL_e, SpeciesNum_e, WhatSpecies_e, Spe_Total_CNO_e,
        FNAN_e, natn_e, ncn_e, atv_ijk_e, S00_e, S01_e, H00_e, H01_e);

      MTRAN_Set_SurfOverlap("right", k2, k3, SpinP_switch, atomnum_e,
        OLP_e, H_e, iHNL_e, SpeciesNum_e, WhatSpecies_e, Spe_Total_CNO_e,
        FNAN_e, natn_e, ncn_e, atv_ijk_e, S00_e, S01_e, H00_e, H01_e);

      /* set CC, CL and CR */

      MTRAN_Set_CentOverlap(3,
        SpinP_switch,
        k2,
        k3,
        NUM_c,
        NUM_e,
        H,
        iHNL,
        OLP,
        atomnum,
        atomnum_e,
        WhatSpecies,
        WhatSpecies_e,
        Spe_Total_CNO,
        Spe_Total_CNO_e,
        FNAN,
        natn,
        ncn,
        atv_ijk,
        TRAN_region,
        TRAN_Original_Id
        );

      /* calculate transmission */

      MTRAN_EigenChannel_NC(
        comm_tmp,
        numprocs_tmp,
        myid_tmp,
        myid0,
        SpinP_switch,
        ChemP_e,
        NUM_c,
        NUM_e,
        H00_e,
        S00_e,
        H01_e,
        S01_e,
        HCC,
        HCL,
        HCR,
        SCC,
        SCL,
        SCR,
        tran_surfgreen_iteration_max,
        tran_surfgreen_eps,
        tran_transmission_energyrange,
        TRAN_Channel_Nenergy,
        TRAN_Channel_energy,
        TRAN_Channel_Num,
        kloop,
        TRAN_Channel_kpoint[kloop],
        EChannel[kloop],
        eigentrans[kloop],
        eigentrans_sum[kloop]
        );

    } /* for (kloop0 = 0; kloop0 < num_kloop0; kloop0++) */

    fflush(stdout);

    MP = (int*)malloc(sizeof(int)*(atomnum + 1));
    MTRAN_Set_MP(1, atomnum, WhatSpecies, Spe_Total_CNO, &i, MP);

    for (kloop = 0; kloop < TRAN_Channel_Nkpoint; kloop++){
      for (iw = 0; iw < TRAN_Channel_Nenergy; iw++){
        MPI_Allreduce(MPI_IN_PLACE, eigentrans[kloop][iw], TRAN_Channel_Num,
          MPI_DOUBLE_PRECISION, MPI_SUM, comm1);
        MPI_Allreduce(MPI_IN_PLACE, eigentrans_sum[kloop][iw], 1,
          MPI_DOUBLE_PRECISION, MPI_SUM, comm1);
        for (i = 0; i < TRAN_Channel_Num; i++){
          MPI_Allreduce(MPI_IN_PLACE, EChannel[kloop][iw][i], NUM_c,
            MPI_DOUBLE_COMPLEX, MPI_SUM, comm1);
        }
      }
    }

    TRAN_Output_eigentrans_sum(TRAN_Channel_Nkpoint, TRAN_Channel_Nenergy, eigentrans_sum);

    if (myid0 == Host_ID) {
      printf("\n  Eigenchannel calculation finished \n");
      printf("  They are written in plottable files. \n");
      printf("  File index : *.tranec#k_#E_#spin_#branch_r.xsf  \n");
      printf("               *.tranec#k_#E_#spin_#branch_i.xsf  \n\n");
    }

    for (kloop = 0; kloop < TRAN_Channel_Nkpoint; kloop++){
      for (iw = 0; iw < TRAN_Channel_Nenergy; iw++){
        for (i = 0; i < TRAN_Channel_Num; i++){
          for (k = 0; k < 2; k++){
            TRAN_Output_ChannelCube(
              kloop,
              iw,
              k,
              i,
              NUM_c / 2,
              TRAN_Channel_kpoint[kloop],
              &EChannel[kloop][iw][i][k * NUM_c / 2],
              MP,
              eigentrans[kloop][iw][i],
              TRAN_Channel_energy[iw]);
          }
        }
      }

    }

    free(MP);
    for (kloop = 0; kloop < TRAN_Channel_Nkpoint; kloop++){
      for (iw = 0; iw < TRAN_Channel_Nenergy; iw++){
        for (i = 0; i < TRAN_Channel_Num + 1; i++){
          free(EChannel[kloop][iw][i]);
        }
        free(eigentrans[kloop][iw]);
        free(EChannel[kloop][iw]);
      }
      free(eigentrans[kloop]);
      free(EChannel[kloop]);
    }
    free(eigentrans);
    free(EChannel);

    free(NPROCS_ID2);
    free(Comm_World2);
    free(NPROCS_WD2);
    free(Comm_World_StartID2);
    free(MPI_CommWD2);
  } /* if (TRAN_Channel == 1) */

  /********************************************
  Finish EigenChannel Analysis
  *********************************************/
  /* E MitsuakiKAWAMURA*/

  /**********************************************
                freeing of arrays
  **********************************************/

  if (tran_transmission_on) {
   
    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      for (i3=0; i3<TRAN_TKspace_grid3; i3++){
        free(current[i2][i3]);
      }
      free(current[i2]);
    }
    free(current);

    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      free(op_flag[i2]);
    }
    free(op_flag);

    free(T_KGrids2);
    free(T_KGrids3);
    free(T_IGrids2);
    free(T_IGrids3);
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
  }

  /* MPI_Finalize() and exit(0) should not be done here.
  MitsuakiKawamura

  MPI_Finalize();
  exit(0);
  */
}







void MTRAN_Read_Tran_HS(MPI_Comm comm1, char *filepath, char *filename, char *ext ) 
{
  FILE *fp;
  int iv[20];
  double v[20];
  double vtmp[50];
  int i,j,k,id;
  int Gc_AN,Cwan,tno0,tno1;
  int h_AN,Gh_AN,Hwan;
  int iside,spin;
  int size1;
  int *ia_vec;
  char fname[300];
  int inconsistency;
  int numprocs0,myid0;

  MPI_Comm_size(comm1,&numprocs0);
  MPI_Comm_rank(comm1,&myid0);

  if (tran_interpolate==0){
    sprintf(fname,"%s%s.%s",filepath,filename,ext);
  }
  else if (tran_interpolate==1){
    sprintf(fname,"%s%s",filepath,interpolate_filename1);
  }

  /**********************************************
       read the first Hamiltonian regardless
         of performing the interpolation 
  **********************************************/

  if (  (fp=fopen(fname,"r"))==NULL ) {
    printf("cannot open %s\n",fname);
    printf("in MTRAN_Read_Tran_HS\n");
    exit(0);
  }

  /* SpinP_switch, NUM_c, and NUM_e */

  i=0;
  fread(iv, sizeof(int),4,fp);
  SpinP_switch = iv[i++];
  NUM_c    = iv[i++];
  NUM_e[0] = iv[i++];
  NUM_e[1] = iv[i++];

  if (PrintLevel){
    printf("spin=%d NUM_c=%d NUM_e=%d %d\n", SpinP_switch, NUM_c, NUM_e[0], NUM_e[1]);
  }

  /* chemical potential */

  i=0;
  fread(v,sizeof(double),3,fp);
  ChemP     = v[i++]; 
  ChemP_e[0]= v[i++];
  ChemP_e[1]= v[i++];

  if (myid0==0 && tran_interpolate==0){
    printf("\n");fflush(stdout);
    printf("Chemical potentials used in the SCF calculation\n");  fflush(stdout);
    printf("  Left lead:  %15.12f (eV)\n",ChemP_e[0]*eV2Hartree);fflush(stdout);
    printf("  Right lead: %15.12f (eV)\n",ChemP_e[1]*eV2Hartree);fflush(stdout);
  }

  /* SCF_tran_bias_apply */

  fread(iv, sizeof(int),1,fp);
  SCF_tran_bias_apply = iv[0];

  /* the number of atoms */

  i=0;
  fread(iv, sizeof(int),3,fp);
  atomnum      = iv[i++];
  atomnum_e[0] = iv[i++];
  atomnum_e[1] = iv[i++];

  if (PrintLevel){
    printf("atomnum=%d atomnum_e=%d %d\n", atomnum, atomnum_e[0], atomnum_e[1]);
  }

  /* the number of species */

  i=0;
  fread(iv, sizeof(int),3,fp);
  SpeciesNum      = iv[i++];
  SpeciesNum_e[0] = iv[i++];
  SpeciesNum_e[1] = iv[i++];

  if (PrintLevel){
    printf("SpeciesNum=%d SpeciesNum_e=%d %d\n",SpeciesNum, SpeciesNum_e[0], SpeciesNum_e[1]);
  }

  /* TCpyCell */

  i=0;
  fread(iv, sizeof(int),3,fp);
  TCpyCell      = iv[i++];
  TCpyCell_e[0] = iv[i++];
  TCpyCell_e[1] = iv[i++];

  if (PrintLevel){
    printf("TCpyCell=%d TCpyCell_e=%d %d\n",TCpyCell, TCpyCell_e[0], TCpyCell_e[1]);
  }

  /* TRAN_region */

  TRAN_region = (int*)malloc(sizeof(int)*(atomnum+1));
  fread(TRAN_region,sizeof(int),atomnum+1,fp);

  /* TRAN_Original_Id */

  TRAN_Original_Id = (int*)malloc(sizeof(int)*(atomnum+1));
  fread(TRAN_Original_Id,sizeof(int),atomnum+1,fp);

  /**********************************************
       informations of the central region
  **********************************************/

  WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));
  fread(WhatSpecies,   sizeof(int), atomnum+1,  fp);

  if (PrintLevel){
    for (i=0; i<=atomnum; i++) {
      printf("i=%2d WhatSpecies=%2d\n",i,WhatSpecies[i]);
    }
  }

  Spe_Total_CNO = (int*)malloc(sizeof(int)*(SpeciesNum));
  fread(Spe_Total_CNO, sizeof(int), SpeciesNum, fp);

  FNAN = (int*)malloc(sizeof(int)*(atomnum+1));
  fread(FNAN,sizeof(int), atomnum+1,fp);

  if (PrintLevel){
    for (i=0; i<=atomnum; i++) {
      printf("i=%2d FNAN=%2d\n",i,FNAN[i]);
    }   
  }
  
  fread(&Max_FSNAN,sizeof(int),1,fp);
  fread(&ScaleSize,sizeof(double),1,fp);

  if (PrintLevel){
    printf("Max_FSNAN=%2d ScaleSize=%10.5f\n",Max_FSNAN,ScaleSize);
  }

  size1=(int)(Max_FSNAN*ScaleSize) + 1;

  natn = (int**)malloc(sizeof(int*)*(atomnum+1));
  for (i=0; i<=(atomnum); i++) {
    natn[i] = (int*)malloc(sizeof(int)*size1);
  }
  for (i=0; i<=(atomnum); i++) {
    fread(natn[i],sizeof(int),size1,fp);
  }

  if (PrintLevel){
    for (i=0; i<=(atomnum); i++) {
      for (j=0; j<size1; j++) {
	printf("i=%3d j=%3d  natn=%2d\n",i,j,natn[i][j]);   
      }
    }
  }

  ncn = (int**)malloc(sizeof(int*)*(atomnum+1));
  for (i=0; i<=(atomnum); i++) {
    ncn[i] = (int*)malloc(sizeof(int)*size1);
  }
  for (i=0; i<=(atomnum); i++) {
    fread(ncn[i],sizeof(int),size1,fp);
  }

  if (PrintLevel){
    for (i=0; i<=(atomnum); i++) {
      for (j=0; j<size1; j++) {
	printf("i=%3d j=%3d  ncn=%2d\n",i,j,natn[i][j]);   
      }
    }
  }

  size1=(TCpyCell+1)*4;
  ia_vec=(int*)malloc(sizeof(int)*size1);
  fread(ia_vec,sizeof(int),size1,fp);

  atv_ijk = (int**)malloc(sizeof(int*)*(TCpyCell+1));
  for (i=0; i<(TCpyCell+1); i++) {
    atv_ijk[i] = (int*)malloc(sizeof(int)*4);
  }

  id=0;
  for (i=0; i<(TCpyCell+1); i++) {
    for (j=0; j<=3; j++) {
      atv_ijk[i][j] = ia_vec[id++];
    }

    if (PrintLevel){
      printf("atv_ijk %3d   %2d %2d %2d\n",i,atv_ijk[i][1],atv_ijk[i][2],atv_ijk[i][3]);
    }

  }
  free(ia_vec);

  /* OLP */

  OLP = (double*****)malloc(sizeof(double****)*4);
  for (k=0; k<4; k++) {

    OLP[k] = (double****)malloc(sizeof(double***)*(atomnum+1));

    FNAN[0] = 0;
    for (Gc_AN=0; Gc_AN<=(atomnum); Gc_AN++){

      if (Gc_AN==0){
	tno0 = 1;
      }
      else{
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      OLP[k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Gc_AN==0){
	  tno1 = 1;
	}
	else {
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	OLP[k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	for (i=0; i<tno0; i++){
	  OLP[k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);

	  if (Gc_AN!=0){
	    fread(OLP[k][Gc_AN][h_AN][i],sizeof(double), tno1, fp); 

	    /*
	     */

	  }
	}
      }
    }
  }

  /* H */

  H = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
  for (k=0; k<(SpinP_switch+1); k++) {

    H[k] = (double****)malloc(sizeof(double***)*(atomnum+1));

    FNAN[0] = 0;
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

      if (Gc_AN==0){
	tno0 = 1;
      }
      else{
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      H[k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Gc_AN==0){
	  tno1 = 1;
	}
	else {
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	H[k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	for (i=0; i<tno0; i++){
	  H[k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	  if (Gc_AN!=0){
	    fread(H[k][Gc_AN][h_AN][i],sizeof(double),tno1,fp); 
	  }
	}
      }
    }
  }

  /* revised by Y. Xiao for Noncollinear NEGF calculations */
  if(SpinP_switch == 3) {
    iHNL = (double*****)malloc(sizeof(double****)*(2+1));
    for (k=0; k<(2+1); k++) {

      iHNL[k] = (double****)malloc(sizeof(double***)*(atomnum+1));

      FNAN[0] = 0;
      for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

	if (Gc_AN==0){
	  tno0 = 1;
	}
	else{
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];
	}

	iHNL[k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Gc_AN==0){
	    tno1 = 1;
	  }
	  else {
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  }

	  iHNL[k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	  for (i=0; i<tno0; i++){
	    iHNL[k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	    if (Gc_AN!=0){
	      fread(iHNL[k][Gc_AN][h_AN][i],sizeof(double),tno1,fp);
	    }
	  }
	}
      }
    }
  } /* if (SpinP_switch == 3) */

  /* H0 (matrix elements for kinetic energy) */

  H0 = (double*****)malloc(sizeof(double****)*4);
  for (k=0; k<4; k++) {

    H0[k] = (double****)malloc(sizeof(double***)*(atomnum+1));

    FNAN[0] = 0;
    for (Gc_AN=0; Gc_AN<=(atomnum); Gc_AN++){

      if (Gc_AN==0){
	tno0 = 1;
      }
      else{
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      H0[k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Gc_AN==0){
	  tno1 = 1;
	}
	else {
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	H0[k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	for (i=0; i<tno0; i++){
	  H0[k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);

	  if (Gc_AN!=0){
	    fread(H0[k][Gc_AN][h_AN][i],sizeof(double), tno1, fp); 

	    /*
	    for (j=0; j<tno1; j++){
	      printf("k=%2d Gc_AN=%2d h_AN=%2d i=%2d j=%2d H0=%15.12f\n",
		     k,Gc_AN,h_AN,i,j,H0[k][Gc_AN][h_AN][i][j]);
	    } 
	    */       

	  }
	}
      }
    }
  }

  /**********************************************
              informations of leads
  **********************************************/

  for (iside=0; iside<=1; iside++) {

    WhatSpecies_e[iside] = (int*)malloc(sizeof(int)*(atomnum_e[iside]+1));
    fread(WhatSpecies_e[iside],   sizeof(int), atomnum_e[iside]+1,  fp);

    if (PrintLevel){
      for (i=0; i<=atomnum_e[iside]; i++) {
	printf("iside=%2d i=%2d WhatSpecies_e=%2d\n",iside,i,WhatSpecies_e[iside][i]);
      }
    }

    Spe_Total_CNO_e[iside] = (int*)malloc(sizeof(int)*SpeciesNum_e[iside]);
    fread(Spe_Total_CNO_e[iside], sizeof(int), SpeciesNum_e[iside], fp);

    FNAN_e[iside] = (int*)malloc(sizeof(int)*(atomnum_e[iside]+1));
    fread(FNAN_e[iside],          sizeof(int), atomnum_e[iside]+1,  fp);

    fread(&Max_FSNAN_e[iside],    sizeof(int), 1,                   fp);
    fread(&ScaleSize_e[iside],    sizeof(double),1,                 fp);

    size1=(int)(Max_FSNAN_e[iside]*ScaleSize_e[iside]) + 1;

    natn_e[iside] = (int**)malloc(sizeof(int*)*(atomnum_e[iside]+1));
    for (i=0; i<=atomnum_e[iside]; i++) {
      natn_e[iside][i] = (int*)malloc(sizeof(int)*size1);
    }
    for (i=0; i<=atomnum_e[iside]; i++) {
      fread(natn_e[iside][i],sizeof(int),size1,fp);
    }

    ncn_e[iside] = (int**)malloc(sizeof(int*)*(atomnum_e[iside]+1));
    for (i=0; i<=atomnum_e[iside]; i++) {
      ncn_e[iside][i] = (int*)malloc(sizeof(int)*size1);
    }
    for (i=0; i<=atomnum_e[iside]; i++) {
      fread(ncn_e[iside][i],sizeof(int),size1,fp);
    }

    size1=(TCpyCell_e[iside]+1)*4;
    ia_vec=(int*)malloc(sizeof(int)*size1);
    fread(ia_vec,sizeof(int),size1,fp);

    atv_ijk_e[iside] = (int**)malloc(sizeof(int*)*(TCpyCell_e[iside]+1));
    for (i=0; i<(TCpyCell_e[iside]+1); i++) {
      atv_ijk_e[iside][i] = (int*)malloc(sizeof(int)*4);
    }

    id=0;
    for (i=0; i<(TCpyCell_e[iside]+1); i++) {
      for (j=0; j<=3; j++) {
	atv_ijk_e[iside][i][j] = ia_vec[id++];
      }
    }
    free(ia_vec);

    /* overlap matrix */

    OLP_e[iside] = (double*****)malloc(sizeof(double****)*4);
    for (k=0; k<4; k++) {

      OLP_e[iside][k] = (double****)malloc(sizeof(double***)*(atomnum_e[iside]+1));

      FNAN_e[iside][0] = 0;
      for (Gc_AN=0; Gc_AN<=atomnum_e[iside]; Gc_AN++){

	if (Gc_AN==0){
	  tno0 = 1;
	}
	else{
	  Cwan = WhatSpecies_e[iside][Gc_AN];
	  Cwan = WhatSpecies_e[iside][Gc_AN];
	  tno0 = Spe_Total_CNO_e[iside][Cwan];
	}

	OLP_e[iside][k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN_e[iside][Gc_AN]+1));

	for (h_AN=0; h_AN<=FNAN_e[iside][Gc_AN]; h_AN++){

	  if (Gc_AN==0){
	    tno1 = 1;
	  }
	  else{
	    Gh_AN = natn_e[iside][Gc_AN][h_AN];
	    Hwan = WhatSpecies_e[iside][Gh_AN];
	    tno1 = Spe_Total_CNO_e[iside][Hwan];
	  }

	  OLP_e[iside][k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	  for (i=0; i<tno0; i++){
	    OLP_e[iside][k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	    if (Gc_AN!=0){
	      fread(OLP_e[iside][k][Gc_AN][h_AN][i],sizeof(double),tno1,fp); 
	    }
	  }
	}
      }
    }

    /* Hamiltonian matrix */

    H_e[iside] = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
    for (k=0; k<(SpinP_switch+1); k++) {

      H_e[iside][k] = (double****)malloc(sizeof(double***)*(atomnum_e[iside]+1));

      FNAN_e[iside][0] = 0;
      for (Gc_AN=0; Gc_AN<=atomnum_e[iside]; Gc_AN++){

	if (Gc_AN==0){
	  tno0 = 1;
	}
	else{
	  Cwan = WhatSpecies_e[iside][Gc_AN];
	  tno0 = Spe_Total_CNO_e[iside][Cwan];
	}

	H_e[iside][k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN_e[iside][Gc_AN]+1));

	for (h_AN=0; h_AN<=FNAN_e[iside][Gc_AN]; h_AN++){

	  if (Gc_AN==0){
	    tno1 = 1;
	  }
	  else{ 
	    Gh_AN = natn_e[iside][Gc_AN][h_AN];
	    Hwan = WhatSpecies_e[iside][Gh_AN];
	    tno1 = Spe_Total_CNO_e[iside][Hwan];
	  }

	  H_e[iside][k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	  for (i=0; i<tno0; i++){
	    H_e[iside][k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	    if (Gc_AN!=0){
	      fread(H_e[iside][k][Gc_AN][h_AN][i],sizeof(double),tno1,fp); 
	    }
	  }
	}
      }
    }

    /* revised by Y. Xiao for Noncollinear NEGF calculations */
    if (SpinP_switch == 3) {
      iHNL_e[iside] = (double*****)malloc(sizeof(double****)*(2+1));
      for (k=0; k<(2+1); k++) {

	iHNL_e[iside][k] = (double****)malloc(sizeof(double***)*(atomnum_e[iside]+1));

	FNAN_e[iside][0] = 0;
	for (Gc_AN=0; Gc_AN<=atomnum_e[iside]; Gc_AN++){

	  if (Gc_AN==0){
	    tno0 = 1;
	  }
	  else{
	    Cwan = WhatSpecies_e[iside][Gc_AN];
	    tno0 = Spe_Total_CNO_e[iside][Cwan];
	  }

	  iHNL_e[iside][k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN_e[iside][Gc_AN]+1));

	  for (h_AN=0; h_AN<=FNAN_e[iside][Gc_AN]; h_AN++){

	    if (Gc_AN==0){
	      tno1 = 1;
	    }
	    else{
	      Gh_AN = natn_e[iside][Gc_AN][h_AN];
	      Hwan = WhatSpecies_e[iside][Gh_AN];
	      tno1 = Spe_Total_CNO_e[iside][Hwan];
	    }

	    iHNL_e[iside][k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

	    for (i=0; i<tno0; i++){
	      iHNL_e[iside][k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	      if (Gc_AN!=0){
		fread(iHNL_e[iside][k][Gc_AN][h_AN][i],sizeof(double),tno1,fp);
	      }
	    }
	  }
	}
      }
    }

  }

  /**********************************************
              close the file pointer
  **********************************************/

  fclose(fp);

  /************************************************************
            interpolation of the bias voltage effect
  ************************************************************/

  if (tran_interpolate==1){

    inconsistency = 0; 

    sprintf(fname,"%s%s",filepath,interpolate_filename2);

    if (  (fp=fopen(fname,"r"))==NULL ) {
      printf("cannot open %s\n",fname);
      printf("in MTRAN_Read_Tran_HS\n");
      exit(0);
    }

    /* SpinP_switch, NUM_c, and NUM_e */

    i=0;
    fread(iv, sizeof(int),4,fp);

    if (iv[i++]!=SpinP_switch) inconsistency++; 
    if (iv[i++]!=NUM_c)        inconsistency++; 
    if (iv[i++]!=NUM_e[0])     inconsistency++; 
    if (iv[i++]!=NUM_e[1])     inconsistency++; 

    /* chemical potential */

    i=0;
    fread(v,sizeof(double),3,fp);
     
    if (myid0==0){
      printf("The interpolation method is used to take account of the bias voltage.\n");
      printf("  Coefficients for the interpolation %10.6f %10.6f\n\n",interpolate_c1,interpolate_c2);  fflush(stdout);

      printf("               First SCF calc.   Second SCF calc.  Interporated\n");fflush(stdout);
      printf("  Left lead:  %15.12f   %15.12f   %15.12f (eV)\n",  
             ChemP_e[0]*eV2Hartree,
             v[1]*eV2Hartree,
	     (interpolate_c1*ChemP_e[0] + interpolate_c2*v[1])*eV2Hartree);fflush(stdout);

      printf("  Right lead: %15.12f   %15.12f   %15.12f (eV)\n",
             ChemP_e[1]*eV2Hartree,
             v[2]*eV2Hartree,
             (interpolate_c1*ChemP_e[1] + interpolate_c2*v[2])*eV2Hartree);fflush(stdout);

    }     

    ChemP = interpolate_c1*ChemP + interpolate_c2*v[i++];
    ChemP_e[0] = interpolate_c1*ChemP_e[0] + interpolate_c2*v[i++];
    ChemP_e[1] = interpolate_c1*ChemP_e[1] + interpolate_c2*v[i++];

    /* SCF_tran_bias_apply */

    fread(iv, sizeof(int),1,fp);
    SCF_tran_bias_apply = iv[0];

    /* the number of atoms */

    i=0;
    fread(iv, sizeof(int),3,fp);

    if (iv[i++]!=atomnum)      inconsistency++; 
    if (iv[i++]!=atomnum_e[0]) inconsistency++; 
    if (iv[i++]!=atomnum_e[1]) inconsistency++; 

    /* the number of species */

    i=0;
    fread(iv, sizeof(int),3,fp);

    if (iv[i++]!=SpeciesNum)        inconsistency++; 
    if (iv[i++]!=SpeciesNum_e[0])   inconsistency++; 
    if (iv[i++]!=SpeciesNum_e[1])   inconsistency++; 

    /* TCpyCell */

    i=0;
    fread(iv, sizeof(int),3,fp);

    if (iv[i++]!=TCpyCell)        inconsistency++; 
    if (iv[i++]!=TCpyCell_e[0])   inconsistency++; 
    if (iv[i++]!=TCpyCell_e[1])   inconsistency++; 

    /* TRAN_region */

    TRAN_region = (int*)malloc(sizeof(int)*(atomnum+1));
    fread(TRAN_region,sizeof(int),atomnum+1,fp);

    /* TRAN_Original_Id */

    TRAN_Original_Id = (int*)malloc(sizeof(int)*(atomnum+1));
    fread(TRAN_Original_Id,sizeof(int),atomnum+1,fp);

    /**********************************************
       informations of the central region
    **********************************************/

    WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));
    fread(WhatSpecies,   sizeof(int), atomnum+1,  fp);

    Spe_Total_CNO = (int*)malloc(sizeof(int)*(SpeciesNum));
    fread(Spe_Total_CNO, sizeof(int), SpeciesNum, fp);

    FNAN = (int*)malloc(sizeof(int)*(atomnum+1));
    fread(FNAN,sizeof(int), atomnum+1,fp);
  
    fread(&Max_FSNAN,sizeof(int),1,fp);
    fread(&ScaleSize,sizeof(double),1,fp);

    size1=(int)(Max_FSNAN*ScaleSize) + 1;

    natn = (int**)malloc(sizeof(int*)*(atomnum+1));
    for (i=0; i<=(atomnum); i++) {
      natn[i] = (int*)malloc(sizeof(int)*size1);
    }
    for (i=0; i<=(atomnum); i++) {
      fread(natn[i],sizeof(int),size1,fp);
    }

    ncn = (int**)malloc(sizeof(int*)*(atomnum+1));
    for (i=0; i<=(atomnum); i++) {
      ncn[i] = (int*)malloc(sizeof(int)*size1);
    }
    for (i=0; i<=(atomnum); i++) {
      fread(ncn[i],sizeof(int),size1,fp);
    }

    size1=(TCpyCell+1)*4;
    ia_vec=(int*)malloc(sizeof(int)*size1);
    fread(ia_vec,sizeof(int),size1,fp);

    atv_ijk = (int**)malloc(sizeof(int*)*(TCpyCell+1));
    for (i=0; i<(TCpyCell+1); i++) {
      atv_ijk[i] = (int*)malloc(sizeof(int)*4);
    }

    id=0;
    for (i=0; i<(TCpyCell+1); i++) {
      for (j=0; j<=3; j++) {
	atv_ijk[i][j] = ia_vec[id++];
      }
    }
    free(ia_vec);

    /* OLP */

    for (k=0; k<4; k++) {

      for (Gc_AN=1; Gc_AN<=(atomnum); Gc_AN++){

        Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];

	  for (i=0; i<tno0; i++){
            fread(OLP[k][Gc_AN][h_AN][i],sizeof(double), tno1, fp); 
	  }
	}
      }
    }

    /* H */

    for (k=0; k<(SpinP_switch+1); k++) {

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];

	  for (i=0; i<tno0; i++){

	    fread(&vtmp[0],sizeof(double),tno1,fp); 

            for (j=0; j<tno1; j++){
              H[k][Gc_AN][h_AN][i][j] = interpolate_c1*H[k][Gc_AN][h_AN][i][j] + interpolate_c2*vtmp[j];
            }
	  }
	}
      }
    }

    /* iHNL */
    /*revised by Y. Xiao for Noncollinear NEGF calculations */
    if( SpinP_switch == 3) {
      for (k=0; k<(2+1); k++) {

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];

	    for (i=0; i<tno0; i++){

	      fread(&vtmp[0],sizeof(double),tno1,fp);

	      for (j=0; j<tno1; j++){
		iHNL[k][Gc_AN][h_AN][i][j] = interpolate_c1*iHNL[k][Gc_AN][h_AN][i][j] + interpolate_c2*vtmp[j];
	      }
	    }
	  }
	}
      }
    } /* if( SpinP_switch == 3) */
    /* until here by Y. Xiao for Noncollinear NEGF calculations */

    /**********************************************
              informations of leads
    **********************************************/

    for (iside=0; iside<=1; iside++) {

      WhatSpecies_e[iside] = (int*)malloc(sizeof(int)*(atomnum_e[iside]+1));
      fread(WhatSpecies_e[iside],   sizeof(int), atomnum_e[iside]+1,  fp);

      Spe_Total_CNO_e[iside] = (int*)malloc(sizeof(int)*SpeciesNum_e[iside]);
      fread(Spe_Total_CNO_e[iside], sizeof(int), SpeciesNum_e[iside], fp);

      FNAN_e[iside] = (int*)malloc(sizeof(int)*(atomnum_e[iside]+1));
      fread(FNAN_e[iside],          sizeof(int), atomnum_e[iside]+1,  fp);

      fread(&Max_FSNAN_e[iside],    sizeof(int), 1,                   fp);
      fread(&ScaleSize_e[iside],    sizeof(double),1,                 fp);

      size1=(int)(Max_FSNAN_e[iside]*ScaleSize_e[iside]) + 1;

      natn_e[iside] = (int**)malloc(sizeof(int*)*(atomnum_e[iside]+1));
      for (i=0; i<=atomnum_e[iside]; i++) {
	natn_e[iside][i] = (int*)malloc(sizeof(int)*size1);
      }
      for (i=0; i<=atomnum_e[iside]; i++) {
	fread(natn_e[iside][i],sizeof(int),size1,fp);
      }

      ncn_e[iside] = (int**)malloc(sizeof(int*)*(atomnum_e[iside]+1));
      for (i=0; i<=atomnum_e[iside]; i++) {
	ncn_e[iside][i] = (int*)malloc(sizeof(int)*size1);
      }
      for (i=0; i<=atomnum_e[iside]; i++) {
	fread(ncn_e[iside][i],sizeof(int),size1,fp);
      }

      size1=(TCpyCell_e[iside]+1)*4;
      ia_vec=(int*)malloc(sizeof(int)*size1);
      fread(ia_vec,sizeof(int),size1,fp);

      atv_ijk_e[iside] = (int**)malloc(sizeof(int*)*(TCpyCell_e[iside]+1));
      for (i=0; i<(TCpyCell_e[iside]+1); i++) {
	atv_ijk_e[iside][i] = (int*)malloc(sizeof(int)*4);
      }

      id=0;
      for (i=0; i<(TCpyCell_e[iside]+1); i++) {
	for (j=0; j<=3; j++) {
	  atv_ijk_e[iside][i][j] = ia_vec[id++];
	}
      }
      free(ia_vec);

      /* overlap matrix */

      for (k=0; k<4; k++) {
	for (Gc_AN=1; Gc_AN<=atomnum_e[iside]; Gc_AN++){

	  Cwan = WhatSpecies_e[iside][Gc_AN];
	  Cwan = WhatSpecies_e[iside][Gc_AN];
	  tno0 = Spe_Total_CNO_e[iside][Cwan];

	  for (h_AN=0; h_AN<=FNAN_e[iside][Gc_AN]; h_AN++){

	    Gh_AN = natn_e[iside][Gc_AN][h_AN];
	    Hwan = WhatSpecies_e[iside][Gh_AN];
	    tno1 = Spe_Total_CNO_e[iside][Hwan];

	    for (i=0; i<tno0; i++){
	      fread(OLP_e[iside][k][Gc_AN][h_AN][i],sizeof(double),tno1,fp); 
	    }
	  }
	}
      }

      /* Hamiltonian matrix */

      for (k=0; k<(SpinP_switch+1); k++) {

	for (Gc_AN=1; Gc_AN<=atomnum_e[iside]; Gc_AN++){

	  Cwan = WhatSpecies_e[iside][Gc_AN];
	  tno0 = Spe_Total_CNO_e[iside][Cwan];

	  for (h_AN=0; h_AN<=FNAN_e[iside][Gc_AN]; h_AN++){

	    Gh_AN = natn_e[iside][Gc_AN][h_AN];
	    Hwan = WhatSpecies_e[iside][Gh_AN];
	    tno1 = Spe_Total_CNO_e[iside][Hwan];

	    for (i=0; i<tno0; i++){

	      fread(&vtmp[0],sizeof(double),tno1,fp); 

              for (j=0; j<tno1; j++){
                H_e[iside][k][Gc_AN][h_AN][i][j] = interpolate_c1*H_e[iside][k][Gc_AN][h_AN][i][j] + interpolate_c2*vtmp[j];
 	      }
	    }
	  }
	}
      }

      /* iHNL */
      /* revised by Y. Xiao for Noncollinear NEGF calculations */
      if ( SpinP_switch == 3 ) {
	for (k=0; k<(2+1); k++) {

	  for (Gc_AN=1; Gc_AN<=atomnum_e[iside]; Gc_AN++){

	    Cwan = WhatSpecies_e[iside][Gc_AN];
	    tno0 = Spe_Total_CNO_e[iside][Cwan];

	    for (h_AN=0; h_AN<=FNAN_e[iside][Gc_AN]; h_AN++){

	      Gh_AN = natn_e[iside][Gc_AN][h_AN];
	      Hwan = WhatSpecies_e[iside][Gh_AN];
	      tno1 = Spe_Total_CNO_e[iside][Hwan];

	      for (i=0; i<tno0; i++){

		fread(&vtmp[0],sizeof(double),tno1,fp);

		for (j=0; j<tno1; j++){
		  iHNL_e[iside][k][Gc_AN][h_AN][i][j] = interpolate_c1*iHNL_e[iside][k][Gc_AN][h_AN][i][j] + interpolate_c2*vtmp[j];
		}
	      }
	    }
	  }
	}
      } /* if ( SpinP_switch == 3 ) */
      /* until here by Y. Xiao for Noncollinear NEGF calculations */

    } /* iside */


    if (inconsistency!=0){
      printf("found some inconsistency in two files %s %s\n",interpolate_filename1,interpolate_filename2);
      MPI_Finalize();
      exit(0);
    }

  } /* if (tran_interpolate==1) */


}














void MTRAN_Transmission(MPI_Comm comm1,
                        int numprocs,
                        int myid,
			int SpinP_switch,
                        double ChemP_e[2],
			int NUM_c,
			int NUM_e[2],
			dcomplex *H00_e[2],
			dcomplex *S00_e[2],
			dcomplex *H01_e[2],
			dcomplex *S01_e[2],
			dcomplex *HCC,
			dcomplex *HCL,
			dcomplex *HCR,
			dcomplex *SCC,
			dcomplex *SCL,
			dcomplex *SCR, 
			double tran_surfgreen_iteration_max,
			double tran_surfgreen_eps, 
			double tran_transmission_energyrange[3],
			int tran_transmission_energydiv, 
			dcomplex **tran_transmission)
{
  dcomplex w;
  dcomplex *GRL,*GRR;
  dcomplex *GC_R,*GC_A;
  dcomplex *SigmaL_R,*SigmaL_A;
  dcomplex *SigmaR_R,*SigmaR_A;
  dcomplex *v1,*v2;
  dcomplex value;

  int iw,k,iside;
  int ID;
  int tag=99;

  int **iwIdx;
  int Miw,Miwmax ;
  int i,j;
  MPI_Status status;

  v1 = (dcomplex*) malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  v2 = (dcomplex*) malloc(sizeof(dcomplex)*NUM_c*NUM_c);

  /* allocate */
  GRL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0]* NUM_e[0]);
  GRR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1]* NUM_e[1]);

  GC_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GC_A = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaL_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaL_A = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR_A = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  /* initialize */

  for (k=0; k<=1; k++) {
    for (iw=0; iw<tran_transmission_energydiv ; iw++) {
      tran_transmission[k][iw].r = 0.0;
      tran_transmission[k][iw].i = 0.0;
    }
  }

  /*parallel setup*/

  iwIdx=(int**)malloc(sizeof(int*)*numprocs);
  Miwmax = (tran_transmission_energydiv)/numprocs+1;
  for (i=0;i<numprocs;i++) {
    iwIdx[i]=(int*)malloc(sizeof(int)*Miwmax);
  }
  TRAN_Distribute_Node_Idx(0, tran_transmission_energydiv-1, numprocs, Miwmax,
                           iwIdx); /* output */

  /* parallel global iw 0:tran_transmission_energydiv-1 */
  /* parallel local  Miw 0:Miwmax-1                     */
  /* parallel variable iw=iwIdx[myid][Miw]              */

  for (Miw=0; Miw<Miwmax ; Miw++) {

    iw = iwIdx[myid][Miw];

    if ( iw>=0 ) {

      w.r = tran_transmission_energyrange[0] + ChemP_e[0]
            +
  	    (tran_transmission_energyrange[1]-tran_transmission_energyrange[0])*
	    (double)iw/(tran_transmission_energydiv-1);

      w.i = tran_transmission_energyrange[2];

      /*
      printf("iw=%d of %d  w= % 9.6e % 9.6e \n" ,iw, tran_transmission_energydiv,  w.r,w.i);
      */

      /* for (k=0; k<=SpinP_switch; k++) { */
      for (k=0; k<=0; k++) {

        /*****************************************************************
         Note that retarded and advanced Green functions and self energies
         are not conjugate comlex in case of the k-dependent case. 
        **************************************************************/ 

        /* in case of retarded ones */ 

	iside=0;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRL);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL, SCL, SigmaL_R);
        
	iside=1;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRR);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR, SCR, SigmaR_R);

	TRAN_Calc_CentGreen(w, NUM_c, SigmaL_R, SigmaR_R, HCC, SCC, GC_R);

        /* in case of advanced ones */ 

        w.i = -w.i;

	iside=0;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRL);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL, SCL, SigmaL_A);
        
	iside=1;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRR);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR, SCR, SigmaR_A);

	TRAN_Calc_CentGreen(w, NUM_c, SigmaL_A, SigmaR_A, HCC, SCC, GC_A);

        w.i = -w.i;

        /* calculation of transmission  */ 

	TRAN_Calc_OneTransmission(NUM_c, SigmaL_R, SigmaL_A, SigmaR_R, SigmaR_A, GC_R, GC_A, v1, v2 ,&value);

	tran_transmission[k][iw].r = value.r; 
	tran_transmission[k][iw].i = value.i;

        if (PrintLevel){
          printf("k=%2d w.r=%6.3f w.i=%6.3f value.r=%15.12f value.i=%15.12f\n",k,w.r,w.i,value.r,value.i);
	}

        if (SpinP_switch==3){
  	  tran_transmission[1][iw].r = value.r; 
	  tran_transmission[1][iw].i = value.i;
        } 

      } /* for k */
    } /* if ( iw>=0 ) */
  } /* iw */

  /* MPI communication */

  for (k=0; k<=1; k++) {

    for (ID=0; ID<numprocs; ID++) {
      for (Miw=0; Miw<Miwmax ; Miw++) {

	double v[2];

	iw = iwIdx[ID][Miw];

        if (2<=numprocs) MPI_Barrier(comm1);

	v[0] = tran_transmission[k][iw].r;
	v[1] = tran_transmission[k][iw].i;

	if (iw>0 && 2<=numprocs) {
          MPI_Bcast(v, 2, MPI_DOUBLE, ID, comm1);
	}

	tran_transmission[k][iw].r = v[0];
	tran_transmission[k][iw].i = v[1];

      }
    }
  }

  /* freeing of arrays */

  free(GC_R);
  free(GC_A);
  free(SigmaL_R);
  free(SigmaL_A);
  free(SigmaR_R);
  free(SigmaR_A);

  free(GRR);
  free(GRL);
  free(v2);
  free(v1);

  for (i=0;i<numprocs;i++) {
    free(iwIdx[i]);
  }
  free(iwIdx);
}









void MTRAN_Current(MPI_Comm comm1,
                   int numprocs,
                   int myid,
		   int SpinP_switch,
		   double ChemP_e[2],
                   double E_Temp,
		   int NUM_c,
		   int NUM_e[2],
		   dcomplex *H00_e[2],
		   dcomplex *S00_e[2],
		   dcomplex *H01_e[2],
		   dcomplex *S01_e[2],
		   dcomplex *HCC,
		   dcomplex *HCL,
		   dcomplex *HCR,
		   dcomplex *SCC,
		   dcomplex *SCL,
		   dcomplex *SCR, 
		   double tran_surfgreen_iteration_max,
		   double tran_surfgreen_eps, 
                   double *current,
                   double k2, double k3)
{
  dcomplex w;
  dcomplex *GRL,*GRR;
  dcomplex *GC_R,*GC_A;
  dcomplex *SigmaL_R,*SigmaL_A;
  dcomplex *SigmaR_R,*SigmaR_A;
  /*S MitsuakiKAWAMURA2*/
  dcomplex *v1,*v2, *Sinv;
  double kvec[2];
  /*E MitsuakiKAWAMURA2*/
  dcomplex value;

  double Beta,xL,xR,fL,fR;
  double my_current[4];
  int iw,k,iside;
  int ID;
  int tag=99;

  int **iwIdx;
  int Miw,Miwmax ;
  int i,j;
  MPI_Status status;

  Beta = 1.0/kB/E_Temp;

  /* allocation of arrays */

  v1 = (dcomplex*) malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  v2 = (dcomplex*) malloc(sizeof(dcomplex)*NUM_c*NUM_c);

  GRL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0]*NUM_e[0]);
  GRR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1]*NUM_e[1]);

  GC_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  GC_A = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  SigmaL_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  SigmaL_A = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  SigmaR_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  SigmaR_A = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);

  /*S MitsuakiKAWAMURA2*/
  Sinv = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
  TRAN_Calc_Sinv(NUM_c, SCC, Sinv);
  kvec[0] = k2;
  kvec[1] = k3;
  /*E MitsuakiKAWAMURA2*/

  /* parallel setup */

  iwIdx=(int**)malloc(sizeof(int*)*numprocs);
  Miwmax = (Tran_current_num_step)/numprocs+1;

  for (i=0; i<numprocs; i++) {
    iwIdx[i]=(int*)malloc(sizeof(int)*Miwmax);
  }

  TRAN_Distribute_Node_Idx(0, Tran_current_num_step-1, numprocs, Miwmax,
                           iwIdx); /* output */

  /* parallel global iw 0:tran_transmission_energydiv-1 */
  /* parallel local  Miw 0:Miwmax-1 */
  /* parallel variable iw=iwIdx[myid][Miw] */

  for (k=0; k<=1; k++) my_current[k] = 0.0;

  for (Miw=0; Miw<Miwmax ; Miw++) {

    iw = iwIdx[myid][Miw];

    if ( iw>=0 ) {

      w.r = Tran_current_lower_bound + (double)iw*Tran_current_energy_step;
      w.i = Tran_current_im_energy;

        /*****************************************************************
         Note that retarded and advanced Green functions and self energies
         are not conjugate comlex in case of the k-dependent case. 
        **************************************************************/ 

        /* in case of retarded ones */ 

	iside=0;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRL);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL, SCL, SigmaL_R);
        
	iside=1;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRR);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR, SCR, SigmaR_R);

	TRAN_Calc_CentGreen(w, NUM_c, SigmaL_R, SigmaR_R, HCC, SCC, GC_R);

        /* in case of advanced ones */ 

        w.i = -w.i;

	iside=0;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRL);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL, SCL, SigmaL_A);
        
	iside=1;
	TRAN_Calc_SurfGreen_direct(w,NUM_e[iside], H00_e[iside],H01_e[iside],
				   S00_e[iside], S01_e[iside],
				   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRR);

	TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR, SCR, SigmaR_A);

	TRAN_Calc_CentGreen(w, NUM_c, SigmaL_A, SigmaR_A, HCC, SCC, GC_A);

        w.i = -w.i;

        /* calculation of transmission  */ 

	TRAN_Calc_OneTransmission(NUM_c, SigmaL_R, SigmaL_A, SigmaR_R, SigmaR_A, GC_R, GC_A, v1, v2 ,&value);

        /* add the contribution */ 

        xL = (w.r - ChemP_e[0])*Beta;
        fL = 1.0/(1.0 + exp(xL));

        xR = (w.r - ChemP_e[1])*Beta;
        fR = 1.0/(1.0 + exp(xR));

        my_current[0] -= (fL-fR)*value.r*Tran_current_energy_step/(2.0*PI);  /* in atomic unit */
        my_current[1] -= (fL-fR)*value.r*Tran_current_energy_step/(2.0*PI); 

        /*S MitsuakiKAWAMURA2*/
        if (TRAN_CurrentDensity == 1)
          TRAN_Calc_CurrentDensity_NC(NUM_c, GC_R, SigmaL_R, SigmaR_R, VCC, Sinv, kvec, fL, fR,
            Tran_current_energy_step, JLocSym, JLocASym, RhoNL, Jmat);
        /*E MitsuakiKAWAMURA2*/
    } /* if ( iw>=0 ) */
  } /* iw */


  /* parallel communication */

  for (k=0; k<=1; k++) {

    if (2<=numprocs){
      MPI_Allreduce(&my_current[k], &current[k], 1, MPI_DOUBLE, MPI_SUM, comm1);
    }
    else{
      current[k] = my_current[k];
    }

    /**********************************************************

     Current:

       convert the unit from a.u to ampere

       The unit of current is given by eEh/bar{h} in
       atomic unit, where e is the elementary charge,
       Eh Hartree, and bar{h} h/{2pi} given by

        e = 1.60217653 * 10^{-19} C
        Eh = 4.35974417 * 10^{-18} J
        bar{h} = 1.05457168 * 10^{-34} J s

      Therefore, 

      1 a.u.
         = 1.60217653 * 10^{-19} C * 4.35974417 * 10^{-18} J
           / (1.05457168 * 10^{-34} J s )
         = 6.6236178 * 10^{-3} [Cs^{-1}=ampere] 

     Electric potential:

       convert the unit from a.u to volt

       The unit of electric potential is given by Eh/e in
       atomic unit.

      Therefore, 

      1 a.u.
         = 4.35974417 * 10^{-18} J/ (1.60217653 * 10^{-19} C) 
         = 27.21138 [JC^{-1}=volt] 

      This allows us to consider that the difference of 1 eV in
      the chemical potential of leads can be regarded as 1 volt.
    *********************************************************/

    current[k] *= 0.0066236178;

  }


  /* freeing of arrays */

  free(GC_R);
  free(GC_A);
  free(SigmaL_R);
  free(SigmaL_A);
  free(SigmaR_R);
  free(SigmaR_A);

  free(GRR);
  free(GRL);
  free(v2);
  free(v1);

  for (i=0;i<numprocs;i++) {
    free(iwIdx[i]);
  }
  free(iwIdx);
}





















void MTRAN_Output_Transmission(
        MPI_Comm comm1,
        int ID,
        char *fname,
        double k2,
        double k3,
        int SpinP_switch,
        int tran_transmission_energydiv,
        double tran_transmission_energyrange[3],
        dcomplex **tran_transmission
        )
{
  int iw,k;
  dcomplex w;
  int myid;
  FILE *fp;

  MPI_Comm_rank(comm1,&myid);
  if (myid!=ID) { return; }

  printf("  %s\n",fname);fflush(stdout);

  if ( ( fp =fopen(fname,"w") )== NULL ) {
    printf("\ncannot open file to write transmission\n");fflush(stdout);
    exit(0);
  }

  fprintf(fp,"#/************************************************************/\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#  Current:\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    convert the unit from a.u to ampere\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    The unit of current is given by eEh/bar{h} in\n");
  fprintf(fp,"#    atomic unit, where e is the elementary charge,\n");
  fprintf(fp,"#    Eh Hartree, and bar{h} h/{2pi} given by\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#    e = 1.60217653 * 10^{-19} C\n");
  fprintf(fp,"#    Eh = 4.35974417 * 10^{-18} J\n");
  fprintf(fp,"#    bar{h} = 1.05457168 * 10^{-34} J s\n");    
  fprintf(fp,"#\n");
  fprintf(fp,"#    Therefore,\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#    1 a.u.\n");
  fprintf(fp,"#    = 1.60217653 * 10^{-19} C * 4.35974417 * 10^{-18} J\n");
  fprintf(fp,"#    / (1.05457168 * 10^{-34} J s )\n");
  fprintf(fp,"#    = 6.6236178 * 10^{-3} [Cs^{-1}=ampere]\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#  Electric potential:\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    convert the unit from a.u to volt\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    The unit of electric potential is given by Eh/e in\n");
  fprintf(fp,"#    atomic unit.\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    Therefore,\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    1 a.u.\n");
  fprintf(fp,"#    = 4.35974417 * 10^{-18} J/ (1.60217653 * 10^{-19} C)\n"); 
  fprintf(fp,"#    = 27.21138 [JC^{-1}=volt]\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#    This allows us to consider that the difference of 1 eV in\n");
  fprintf(fp,"#    the chemical potential of leads can be regarded as 1 volt.\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#***********************************************************/\n");
  fprintf(fp,"#\n");
  fprintf(fp,"# k2= %8.4f  k3= %8.4f\n",k2,k3);
  fprintf(fp,"# SpinP_switch= %d\n",SpinP_switch);
  fprintf(fp,"# Chemical potential (eV)      Left, Right Leads=% 9.6e % 9.6e\n",
            ChemP_e[0]*eV2Hartree,ChemP_e[1]*eV2Hartree);
  fprintf(fp,"# Chemical potential (Hartree) Left, Right Leads=% 9.6e % 9.6e\n",
            ChemP_e[0],ChemP_e[1]);
  fprintf(fp,"# diff Chemical potential (eV)     =% 9.6e\n",
            (ChemP_e[0]-ChemP_e[1])*eV2Hartree);
  fprintf(fp,"# diff Chemical potential (Hartree)=% 9.6e\n",
            ChemP_e[0]-ChemP_e[1]);
  fprintf(fp,"# tran_transmission_energydiv= %d\n", tran_transmission_energydiv);
  fprintf(fp,"#\n");
  fprintf(fp,"# iw w.real(au) w.imag(au) w.real(eV) w.imag(eV) trans.real trans.imag\n");
  fprintf(fp,"#\n");

  for (iw=0; iw<tran_transmission_energydiv; iw++){

    /* The zero energy is the chemical potential of the left lead. */

    w.r = tran_transmission_energyrange[0] + ChemP_e[0] - ChemP_e[0]
      + 
      (tran_transmission_energyrange[1]-tran_transmission_energyrange[0])*
      (double)iw/(tran_transmission_energydiv-1);

    w.i = tran_transmission_energyrange[2];

    fprintf(fp,"%3d % 9.6e % 9.6e % 9.6e % 9.6e % 9.6e % 9.6e\n",
	    iw,
	    w.r,
	    w.i,
	    w.r*eV2Hartree, w.i*eV2Hartree, 
	    tran_transmission[0][iw].r,
	    tran_transmission[0][iw].i);

  } /* iw */

  fclose(fp);
}


void MTRAN_Output_Current(
        MPI_Comm comm1,
        char *fname,
        int TRAN_TKspace_grid2,
        int TRAN_TKspace_grid3,
        int SpinP_switch,
        double ***current
        )
{
  int iw,i2,i3;
  double k2,k3;
  double crt0,crt1;
  dcomplex w;
  int myid;
  FILE *fp;

  MPI_Comm_rank(comm1,&myid);
  if (myid!=Host_ID) { return; }

  printf("  %s\n",fname);

  if ( ( fp =fopen(fname,"w") )== NULL ) {
    printf("\ncannot open file to write current\n");
    printf("write current to stdout\n");
    exit(0);
  }

  /* total current */

  crt0 = 0.0;
  crt1 = 0.0;
  for (i2=0; i2<TRAN_TKspace_grid2; i2++){
    for (i3=0; i3<TRAN_TKspace_grid3; i3++){
      crt0 += current[i2][i3][0];
      crt1 += current[i2][i3][1];
    }
  }

  crt0 /= (double)(TRAN_TKspace_grid2*TRAN_TKspace_grid3);
  crt1 /= (double)(TRAN_TKspace_grid2*TRAN_TKspace_grid3);

  fprintf(fp,"#/************************************************************/\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#  Current:\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    convert the unit from a.u to ampere\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    The unit of current is given by eEh/bar{h} in\n");
  fprintf(fp,"#    atomic unit, where e is the elementary charge,\n");
  fprintf(fp,"#    Eh Hartree, and bar{h} h/{2pi} given by\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#    e = 1.60217653 * 10^{-19} C\n");
  fprintf(fp,"#    Eh = 4.35974417 * 10^{-18} J\n");
  fprintf(fp,"#    bar{h} = 1.05457168 * 10^{-34} J s\n");    
  fprintf(fp,"#\n");
  fprintf(fp,"#    Therefore,\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#    1 a.u.\n");
  fprintf(fp,"#    = 1.60217653 * 10^{-19} C * 4.35974417 * 10^{-18} J\n");
  fprintf(fp,"#    / (1.05457168 * 10^{-34} J s )\n");
  fprintf(fp,"#    = 6.6236178 * 10^{-3} [Cs^{-1}=ampere]\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#  Electric potential:\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    convert the unit from a.u to volt\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    The unit of electric potential is given by Eh/e in\n");
  fprintf(fp,"#    atomic unit.\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    Therefore,\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#    1 a.u.\n");
  fprintf(fp,"#    = 4.35974417 * 10^{-18} J/ (1.60217653 * 10^{-19} C)\n"); 
  fprintf(fp,"#    = 27.21138 [JC^{-1}=volt]\n"); 
  fprintf(fp,"#\n");
  fprintf(fp,"#    This allows us to consider that the difference of 1 eV in\n");
  fprintf(fp,"#    the chemical potential of leads can be regarded as 1 volt.\n");
  fprintf(fp,"#\n");
  fprintf(fp,"#***********************************************************/\n");
  fprintf(fp,"#\n");
  fprintf(fp,"# SpinP_switch= %d\n",SpinP_switch);
  fprintf(fp,"# Chemical potential (eV)      Left, Right Leads=% 9.6e % 9.6e\n",
            ChemP_e[0]*eV2Hartree,ChemP_e[1]*eV2Hartree);
  fprintf(fp,"# Chemical potential (Hartree) Left, Right Leads=% 9.6e % 9.6e\n",
            ChemP_e[0],ChemP_e[1]);
  fprintf(fp,"# diff Chemical potential (eV)     =% 9.6e\n",
            (ChemP_e[0]-ChemP_e[1])*eV2Hartree);
  fprintf(fp,"# diff Chemical potential (Hartree)=% 9.6e\n",
            ChemP_e[0]-ChemP_e[1]);
  fprintf(fp,"# Corresponding bias voltage (Volt)=% 9.6e\n",
            (ChemP_e[0]-ChemP_e[1])*27.21138 );
  fprintf(fp,"# average current (ampere)=% 9.6e \n",crt0);
  fprintf(fp,"#\n");
  fprintf(fp,"# i2 i3 k2 k3  current (ampere) \n");
  fprintf(fp,"#\n");

  for (i2=0; i2<TRAN_TKspace_grid2; i2++){
    k2 = -0.5 + (2.0*(double)i2+1.0)/(2.0*(double)TRAN_TKspace_grid2);
    for (i3=0; i3<TRAN_TKspace_grid3; i3++){
      k3 = -0.5 + (2.0*(double)i3+1.0)/(2.0*(double)TRAN_TKspace_grid3);

      fprintf(fp,"%3d %3d %8.4f %8.4f % 9.6e \n", 
              i2,i3,k2,k3,current[i2][i3][0]);
    }
  }

  fprintf(fp,"\n\n");

  fclose(fp);
}



void MTRAN_Output_Conductance(
        MPI_Comm comm1,
        char *fname,
        int *T_k_ID,
        int T_knum,
        int TRAN_TKspace_grid2,
        int TRAN_TKspace_grid3,
        int *T_IGrids2, 
        int *T_IGrids3,
        double *T_KGrids2,
        double *T_KGrids3,
        int SpinP_switch,
        int tran_transmission_energydiv,
        double tran_transmission_energyrange[3],
        dcomplex ****tran_transmission
        )
{
  int spin,k,i2,i3,iw,iw0,iw1;
  int po,ii2,ii3,ID;
  double k2,k3,Av_ChemP;
  double ***conductance;
  double w0,w1=0.0,e0,e1;
  double wm,wn,cond[2];
  int myid;
  FILE *fp;

  MPI_Comm_rank(comm1,&myid);
  if (myid==Host_ID)  printf("  %s\n",fname);

  /***********************************************
                allocation of arrays
  ***********************************************/

  conductance = (double***)malloc(sizeof(double**)*TRAN_TKspace_grid2);
  for (i2=0; i2<TRAN_TKspace_grid2; i2++){
    conductance[i2] = (double**)malloc(sizeof(double*)*TRAN_TKspace_grid3);
    for (i3=0; i3<TRAN_TKspace_grid3; i3++){
      conductance[i2][i3] = (double*)malloc(sizeof(double)*2);
      for (spin=0; spin<2; spin++){
        conductance[i2][i3][spin] = 0.0;
      }
    }
  }    

  /***********************************************
   find the conductance at each spin and k-point
  ***********************************************/

  Av_ChemP = 0.50*(ChemP_e[0] + ChemP_e[1]);

  po = 0; 

  for (iw=0; iw<tran_transmission_energydiv; iw++){

    /* The zero energy is the chemical potential of the left lead. */

    w0 = w1;
    w1 = tran_transmission_energyrange[0] + ChemP_e[0] - ChemP_e[0]
      + 
      (tran_transmission_energyrange[1]-tran_transmission_energyrange[0])*
      (double)iw/(tran_transmission_energydiv-1);

    /* find iw0 which is the index near the average chemical potential */

    if (Av_ChemP<(w1+ChemP_e[0]) && po==0){

      po  = 1;
      iw0 = iw - 1;
      iw1 = iw;
      e0  = w0 + ChemP_e[0];
      e1  = w1 + ChemP_e[0];
    }         
  }

  wm = (Av_ChemP - e0)/(e1-e0);
  wn = (e1 - Av_ChemP)/(e1-e0);

  /* calculate the conductance by a linear interpolation */

  for (spin=0; spin<=1; spin++){
    for (k=0; k<T_knum; k++){

      ID = T_k_ID[k];

      k2 = T_KGrids2[k];
      k3 = T_KGrids3[k];
      i2 = T_IGrids2[k];
      i3 = T_IGrids3[k];

      if (myid==ID){

	conductance[i2][i3][spin]   = tran_transmission[i2][i3][spin][iw0].r*wn 
                                    + tran_transmission[i2][i3][spin][iw1].r*wm;

      }

      MPI_Bcast(conductance[i2][i3],   2, MPI_DOUBLE, ID, comm1);

    }  
  }

  /***********************************************
            calculate average conductance
  ***********************************************/

  for (spin=0; spin<=1; spin++){

    cond[spin] = 0.0;
    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      for (i3=0; i3<TRAN_TKspace_grid3; i3++){
        cond[spin] += conductance[i2][i3][spin];
      }
    }

    cond[spin] = cond[spin]/(double)(TRAN_TKspace_grid2*TRAN_TKspace_grid3); 
  }

  /***********************************************
                  save conductance
  ***********************************************/

  if (myid==Host_ID){

    if ( ( fp =fopen(fname,"w") )== NULL ) {
      printf("\ncannot open file to write conductance\n");fflush(stdout);
      exit(0);
    }

    fprintf(fp,"#/************************************************************/\n");
    fprintf(fp,"#\n");
    fprintf(fp,"#  Conductance:\n");
    fprintf(fp,"#\n");
    fprintf(fp,"#    G0 = e^2/h\n");
    fprintf(fp,"#       = (1.60217653 * 10^{-19})^2/(6.626076 * 10^{-34})\n");
    fprintf(fp,"#       = 3.87404194169 * 10^{-5} [C^2 J^{-1} s^{-1}]\n");
    fprintf(fp,"#    note that\n");
    fprintf(fp,"#    e = 1.60217653 * 10^{-19} C\n");
    fprintf(fp,"#    h = 6.626076 * 10^{-34} J s\n");    
    fprintf(fp,"#***********************************************************/\n");
    fprintf(fp,"#\n");

    fprintf(fp,"# SpinP_switch= %d\n",SpinP_switch);
    fprintf(fp,"# Chemical potential (eV)      Left, Right Leads=% 9.6e % 9.6e\n",
            ChemP_e[0]*eV2Hartree,ChemP_e[1]*eV2Hartree);
    fprintf(fp,"# Chemical potential (Hartree) Left, Right Leads=% 9.6e % 9.6e\n",
            ChemP_e[0],ChemP_e[1]);
    fprintf(fp,"# diff Chemical potential (eV)     =% 9.6e\n",
            (ChemP_e[0]-ChemP_e[1])*eV2Hartree);
    fprintf(fp,"# diff Chemical potential (Hartree)=% 9.6e\n",
            ChemP_e[0]-ChemP_e[1]);
    fprintf(fp,"# Corresponding bias voltage (Volt)=% 9.6e\n",
            (ChemP_e[0]-ChemP_e[1])*27.21138 );
    fprintf(fp,"# average conductance (G0)=% 9.6e \n",cond[0]);
    fprintf(fp,"#\n");
    fprintf(fp,"# i2 i3 k2 k3  conductance (G0)\n");
    fprintf(fp,"#\n");

    for (i2=0; i2<TRAN_TKspace_grid2; i2++){
      k2 = -0.5 + (2.0*(double)i2+1.0)/(2.0*(double)TRAN_TKspace_grid2);
      for (i3=0; i3<TRAN_TKspace_grid3; i3++){
	k3 = -0.5 + (2.0*(double)i3+1.0)/(2.0*(double)TRAN_TKspace_grid3);

	fprintf(fp,"%3d %3d %8.4f %8.4f % 9.6e \n", 
		i2,i3,k2,k3,conductance[i2][i3][0]);
      }
    }

    fclose(fp);
  }

  /***********************************************
                freeing of arrays
  ***********************************************/

  for (i2=0; i2<TRAN_TKspace_grid2; i2++){
    for (i3=0; i3<TRAN_TKspace_grid3; i3++){
      free(conductance[i2][i3]);
    }
    free(conductance[i2]);
  }    
  free(conductance);
}











void MTRAN_Input(MPI_Comm comm1,
                 int argc,
                 char *fname,
                 double ChemP_e[2],
                 int *TRAN_TKspace_grid2,
                 int *TRAN_TKspace_grid3,
		 int *SpinP_switch,
		 double *E_Temp,
		 int *tran_surfgreen_iteration_max,
		 double *tran_surfgreen_eps,
		 int  *tran_transmission_on,
		 double tran_transmission_energyrange[3],
		 int *tran_transmission_energydiv,
		 dcomplex ****(*tran_transmission)
		 )
{
  int i,i2,i3,k,po;
  int side0,side1;
  double r_vec[20];
  int i_vec[20];
  int i_vec2[20];
  char *s_vec[20];
  double Beta,f0,f1;
  double x,x0,x1,tmpx0,tmpx1;
  double Av_ChemP;
  int myid;
  /* S MitsuakiKAWAMURA */
  char buf[BUFSIZE];
  FILE *fp;
  /* E MitsuakiKAWAMURA*/

  MPI_Comm_rank(comm1,&myid);

  /* open the input file */

  if (input_open(fname)==0){
    MPI_Finalize(); 
    exit(0);
  }

  s_vec[0]="Off"; s_vec[1]="On"; s_vec[2]="NC";
  i_vec[0]=0    ; i_vec[1]=1   ; i_vec[2]=3;
  input_string2int("scf.SpinPolarization", SpinP_switch, 3, s_vec,i_vec);
  input_double("scf.ElectronicTemperature", E_Temp, 300);
  /* chage the unit from K to a.u. */
  *E_Temp = *E_Temp/eV2Hartree;

  Beta = 1.0/kB/(*E_Temp);

  input_int(   "NEGF.Surfgreen.iterationmax", tran_surfgreen_iteration_max, 600);
  input_double("NEGF.Surfgreen.convergeeps", tran_surfgreen_eps, 1.0e-12);

  /****  k-points parallel to the layer, which are used for the transmission calc. ****/

  i_vec2[0]=1;
  i_vec2[1]=1;
  input_intv("NEGF.tran.Kgrid",2,i_vec,i_vec2);
  *TRAN_TKspace_grid2 = i_vec[0];
  *TRAN_TKspace_grid3 = i_vec[1];

  if (*TRAN_TKspace_grid2<=0){
    printf("NEGF.tran.Kgrid should be over 1\n");
    MPI_Finalize();
    exit(1);
  } 

  if (*TRAN_TKspace_grid3<=0){
    printf("NEGF.tran.Kgrid should be over 1\n");
    MPI_Finalize();
    exit(1);
  } 

  input_logical("NEGF.tran.on",tran_transmission_on,1);  /* default=on */

  if (tran_transmission_on) {
    i=0;
    r_vec[i++] = -10.0;
    r_vec[i++] =  10.0;
    r_vec[i++] = 1.0e-3;
    input_doublev("NEGF.tran.energyrange",i, tran_transmission_energyrange, r_vec); /* in eV */
    /* change the unit from eV to Hartree */
    tran_transmission_energyrange[0] /= eV2Hartree;
    tran_transmission_energyrange[1] /= eV2Hartree;
    tran_transmission_energyrange[2] /= eV2Hartree;

    input_int("NEGF.tran.energydiv",tran_transmission_energydiv,200);

    *tran_transmission = (dcomplex****)malloc(sizeof(dcomplex***)*(*TRAN_TKspace_grid2));
    for (i2=0; i2<(*TRAN_TKspace_grid2); i2++){
      (*tran_transmission)[i2] = (dcomplex***)malloc(sizeof(dcomplex**)*(*TRAN_TKspace_grid3));
      for (i3=0; i3<(*TRAN_TKspace_grid3); i3++){
	(*tran_transmission)[i2][i3] = (dcomplex**)malloc(sizeof(dcomplex*)*3);
	for (i=0; i<3; i++) {
	  (*tran_transmission)[i2][i3][i] = (dcomplex*)malloc(sizeof(dcomplex)*(*tran_transmission_energydiv));
	}
      }
    }
  }
  else {
    tran_transmission_energydiv=0;
    *tran_transmission=NULL;
  }

  /* set Order_Lead_Side */

  if (ChemP_e[0]<ChemP_e[1]){
    Order_Lead_Side[0] = 0;
    Order_Lead_Side[1] = 1;
  }
  else {
    Order_Lead_Side[0] = 1;
    Order_Lead_Side[1] = 0;
  }

  input_double("NEGF.current.energy.step", &Tran_current_energy_step, 0.01);  /* in eV */
  if (Tran_current_energy_step<0.0) {
    printf("NEGF.bias.neq.energy.step should be positive.\n");
    MPI_Finalize();
    exit(1);
  }

  /* change the unit from eV to Hartree */
  Tran_current_energy_step /= eV2Hartree; 

  input_double("NEGF.current.im.energy", &Tran_current_im_energy, tran_transmission_energyrange[2]*eV2Hartree); /* in eV */

  /* change the unit from eV to Hartree */
  Tran_current_im_energy /= eV2Hartree;  

  input_double("NEGF.current.cutoff", &Tran_current_cutoff, 1.0e-8);  /* dimensionless */

  side0 = Order_Lead_Side[0];
  side1 = Order_Lead_Side[1];

  Av_ChemP = 0.5*(ChemP_e[side0] + ChemP_e[side1]);
  x = Av_ChemP;   

  po = 0;
  do {

    f0 = 1.0/(1.0+exp((x-ChemP_e[side0])*Beta));
    f1 = 1.0/(1.0+exp((x-ChemP_e[side1])*Beta));

    if ( fabs(f1-f0)<Tran_current_cutoff ){

      po = 1;
      x1 = x; 
    };

    x += Tran_current_energy_step*0.1;

  } while (po==0);

  x0 = Av_ChemP - (x1 - Av_ChemP);   
  Tran_current_lower_bound = x0;    
  Tran_current_num_step = (int)((x1-x0)/(double)Tran_current_energy_step);

  if (Tran_current_num_step<10 && myid==0){
    printf("NEGF.current.energy.step %8.4e seems to be large for the calculation of current in the bias voltage %8.4e\n",
	   Tran_current_energy_step*eV2Hartree,(ChemP_e[side1]-ChemP_e[side0])*eV2Hartree);
    printf("The recommended Tran.current.energy.step is %8.4e (eV).\n",
	   (x1-x0)/50.0*eV2Hartree);
  } 

  /* S MitsuakiKAWAMURA */

  /********************************************************
  Parameters for the Transmission EigenChannel
  *********************************************************/

  /* Check Whether eigenchannel is computed or not */

  input_logical("NEGF.tran.channel", &TRAN_Channel, 1);

  if (TRAN_Channel == 1){

    /* The number of k points where eigchannels are computed */

    input_int("NEGF.Channel.Nkpoint", &TRAN_Channel_Nkpoint, 1);
    if (TRAN_Channel_Nkpoint < 1){
      printf("NEGF.Channel.Nkpoint must be >= 1 \n");
      MPI_Finalize();
      exit(1);
    }

    /* Allocate and read k points where eigenchannel are computed */

    TRAN_Channel_kpoint = (double**)malloc(sizeof(double*)*TRAN_Channel_Nkpoint);
    for (i = 0; i < TRAN_Channel_Nkpoint; i++){
      TRAN_Channel_kpoint[i] = (double*)malloc(sizeof(double) * 2);
    } /* for (i = 0; i<(TRAN_Channel_Nkpoint + 1); i++) */

    if (fp = input_find("<NEGF.Channel.kpoint")) {
      for (i = 0; i < TRAN_Channel_Nkpoint - 1; i++){
        fgets(buf, BUFSIZE, fp);
        sscanf(buf, "%lf %lf",
          &TRAN_Channel_kpoint[i][0], &TRAN_Channel_kpoint[i][1]);
      } /* for (i = 0; i<TRAN_Channel_Nkpoint; i++) */
      fscanf(fp, "%lf %lf",
        &TRAN_Channel_kpoint[i][0], &TRAN_Channel_kpoint[i][1]);

      if (!input_last("NEGF.Channel.kpoint>")) {
        /* format error */
        printf("Format error for NEGF.Channel.kpoint\n");
        MPI_Finalize();
        exit(1);
      } /* if (!input_last("NEGE.Channel.kpoint>")) */

      /* k points are read as the fractional coordinate */
    } /* if (fp = input_find("<NEGF.Channel.kpoint")) */
    else if (TRAN_Channel_Nkpoint == 1){
      /* In default, the first k point is 0.0 0.0 */
      TRAN_Channel_kpoint[0][0] = 0.0;
      TRAN_Channel_kpoint[0][1] = 0.0;
    } /* (TRAN_Channel_Nkpoint == 1) */
    else if (TRAN_Channel_Nkpoint > 1){
      printf("NEGF.Channel.Nkpoint > 1 and NEGF.Channel.kpoint is NOT found \n");
      MPI_Finalize();
      exit(1);
    } /* else if (TRAN_Channel_Nkpoint > 1) */

    if (myid == Host_ID) for (i = 0; i < TRAN_Channel_Nkpoint; i++)
      printf("  TRAN_Channel_kpoint %2d  %10.6f  %10.6f\n",
      i, TRAN_Channel_kpoint[i][0], TRAN_Channel_kpoint[i][1]);

    /* The number of energy where eigenchannels are computed */

    input_int("NEGF.Channel.Nenergy", &TRAN_Channel_Nenergy, 1);
    if (TRAN_Channel_Nenergy < 1){
      printf("NEGF.Channel.Nenergy must be >= 1 \n");
      MPI_Finalize();
      exit(1);
    }

    /* Allocate and read energies where eigenchannels are computed */

    TRAN_Channel_energy = (double*)malloc(sizeof(double)*(TRAN_Channel_Nenergy));

    if (fp = input_find("<NEGF.Channel.Energy")){
      for (i = 0; i<TRAN_Channel_Nenergy - 1; i++){
        fgets(buf, BUFSIZE, fp);
        sscanf(buf, "%lf", &TRAN_Channel_energy[i]);
      } /* for (i = 0; i<TRAN_Channel_Nenergy; i++) */
      fscanf(fp, "%lf", &TRAN_Channel_energy[i]);

      if (!input_last("NEGF.Channel.Energy>")){
        /* format error */
        printf("Format error for NEGF.Channel.energy\n");
        MPI_Finalize();
        exit(1);
      } /* if (!input_last("NEGE.Channel.energy>")) */

    } /* if (fp = input_find("<NEGF.Channel.Energy")) */
    else if (TRAN_Channel_Nenergy == 1){
      /* In default, the first energy is 0.0 (E_F of the left lead) */
      TRAN_Channel_energy[0] = 0.0;
    } /* else if (TRAN_Channel_Nenergy == 1) */
    else if (TRAN_Channel_Nenergy > 1){
      printf("NEGF.Channel.Nenergy > 1 and NEGF.Channel.energy is NOT found \n");
      MPI_Finalize();
      exit(1);
    } /* else if (TRAN_Channel_Nenergy > 1) */

    for (i = 0; i < TRAN_Channel_Nenergy; i++){
      if (myid == Host_ID)
        printf("  TRAN_Channel_energy %2d  %10.6f eV\n", i, TRAN_Channel_energy[i]);
      TRAN_Channel_energy[i] /= eV2Hartree;
    }

    /* The number of eigenchannel-cube file */

    input_int("NEGF.Channel.Num", &TRAN_Channel_Num, 10);
    if (myid == Host_ID)
      printf("  TRAN_Channel_Num %d \n", TRAN_Channel_Num);
    if (TRAN_Channel_Num < 0){
      printf("NEGF.Channel.Num must be >= 0 \n");
      MPI_Finalize();
      exit(1);
    }
  }
  /***********************************************************************
  Current density calculation flag
  ***********************************************************************/

  input_logical("NEGF.tran.CurrentDensity", &TRAN_CurrentDensity, 1);
  input_logical("NEGF.OffDiagonalCurrent", &TRAN_OffDiagonalCurrent, 0);
  if (TRAN_OffDiagonalCurrent == 1 && TRAN_CurrentDensity != 1) {
    TRAN_CurrentDensity = 1;
    if (myid == Host_ID)
      printf("Since NEGF.OffDiagonalCurrent is on, NEGF.tran.CurrentDensity is automatically set to on.\n");
  }

  /* E MitsuakiKAWAMURA */
  /* print information */

  if (myid==0){
    printf("\n");
    printf("Parameters for the calculation of the current\n");
    printf("  lower bound:    %15.12f (eV)\n",x0*eV2Hartree);
    printf("  upper bound:    %15.12f (eV)\n",x1*eV2Hartree);
    printf("  energy step:    %15.12f (eV)\n",Tran_current_energy_step*eV2Hartree);
    printf("  imginary energy %15.12f (eV)\n",Tran_current_im_energy*eV2Hartree);
    printf("  number of steps: %3d     \n",Tran_current_num_step);
  }            

  input_close();

}


void MTRAN_Input_Sys(int argc,char *file, char *filepath, char *filename,
                     int *tran_interpolate, 
                     char *interpolate_filename1,
                     char *interpolate_filename2)
{
  double r_vec[20];
  double r_vec1[20];

  if (input_open(file)==0){
    MPI_Finalize(); 
    exit(0);
  }

  input_string("System.CurrrentDirectory",filepath,"./");
  input_string("System.Name",filename,"default");

  /* check whether the interpolation is made or not */
  input_logical("NEGF.tran.interpolate",tran_interpolate,0);  /* default=off */
  input_string("NEGF.tran.interpolate.file1",interpolate_filename1,"file1.tranb1");
  input_string("NEGF.tran.interpolate.file2",interpolate_filename2,"file2.tranb2");

  r_vec[0] = 1.0; r_vec[1] = 0.0;
  input_doublev("NEGF.tran.interpolate.coes",2, r_vec1, r_vec);
  interpolate_c1 = r_vec1[0];
  interpolate_c2 = r_vec1[1];

  if (1.0e-9<fabs(r_vec1[0]+r_vec1[1]-1.0)){
    printf("The sum of coefficients for the interpolation should be unity.\n");
    MPI_Finalize();
    exit(0);
  }

  input_close();
}


void MTRAN_Set_MP(
        int job, 
        int anum, int *WhatSpecies, int *Spe_Total_CNO, 
        int *NUM,  /* output */
        int *MP    /* output */
        )
{
  int Anum, i, wanA, tnoA;

 /* setup MP */
  Anum = 1;
  for (i=1; i<=anum; i++){
    if (job) MP[i]=Anum; 
    wanA = WhatSpecies[i];
    tnoA = Spe_Total_CNO[wanA];
    Anum += tnoA;
  }

  *NUM=Anum-1;
} 






void MTRAN_Set_SurfOverlap(char *position, 
                           double k2,
                           double k3,
                           int SpinP_switch,
                           int atomnum_e[2],
                           double *****OLP_e[2],
                           double *****H_e[2],
                           double *****iHNL_e[2],
                           int SpeciesNum_e[2], 
                           int *WhatSpecies_e[2], 
                           int *Spe_Total_CNO_e[2], 
                           int *FNAN_e[2],
                           int **natn_e[2], 
                           int **ncn_e[2], 
                           int **atv_ijk_e[2],
			   dcomplex *S00_e[2],
			   dcomplex *S01_e[2],
                           dcomplex *H00_e[2],
			   dcomplex *H01_e[2]
                           )
/* #define S00_ref(i,j) ( ((j)-1)*NUM+(i)-1 ) */
#define S00_ref(i,j) ( ((j)-1)*NUM+(i)-1 ) 
#define S01_ref(i,j) ( ((j+NUM0)-1)*NUM+(i)-1 )
#define S10_ref(i,j) ( ((j)-1)*NUM+(i+NUM0)-1 )
#define S11_ref(i,j) ( ((j+NUM0)-1)*NUM+(i+NUM0)-1 ) 
{
  int NUM,n2;
  int Anum, Bnum, wanA, tnoA;
  int i,j,k;
  int GA_AN;
  int GB_AN, LB_AN,wanB, tnoB,Rn;
  int l1,l2,l3; 
  int spin,MN;
  int SpinP_switch_e[2]; 
  int direction,iside;
  double si,co,kRn;
  double s,h[10];
  static double epscutoff=1.0e-12;
  double epscutoff2;
  int *MP;
  double a00,a11,a01r,a01i,b00,b11,b01;
  int NUM0;
  /*debug */
  char msg[100];
  /*end debug */

  SpinP_switch_e[0] = SpinP_switch;
  SpinP_switch_e[1] = SpinP_switch;

  /* position -> direction */

  if      ( strcasecmp(position,"left")==0) {
    direction=-1;
    iside=0;
  }
  else if ( strcasecmp(position,"right")==0) {
    direction= 1;
    iside=1;
  } 

  /* set MP */
 /* MTRAN_Set_MP(0, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &NUM, &i);*/
  MP = (int*)malloc(sizeof(int)*(atomnum_e[iside]+1));
  MTRAN_Set_MP(1, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &NUM, MP);
  NUM0=NUM;
  NUM=2*NUM;

  n2 = NUM ;   

  for (i=0; i<n2*n2; i++){
    S00_e[iside][i].r = 0.0;
    S00_e[iside][i].i = 0.0;
    S01_e[iside][i].r = 0.0;
    S01_e[iside][i].i = 0.0;
  }

    for (i=0; i<n2*n2; i++){
      H00_e[iside][i].r = 0.0;
      H00_e[iside][i].i = 0.0;
      H01_e[iside][i].r = 0.0;
      H01_e[iside][i].i = 0.0;
    }

  for (GA_AN=1; GA_AN<=atomnum_e[iside]; GA_AN++){

    wanA = WhatSpecies_e[iside][GA_AN];
    tnoA = Spe_Total_CNO_e[iside][wanA];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN_e[iside][GA_AN]; LB_AN++){

      GB_AN = natn_e[iside][GA_AN][LB_AN];
      Rn = ncn_e[iside][GA_AN][LB_AN];
      wanB = WhatSpecies_e[iside][GB_AN];
      tnoB = Spe_Total_CNO_e[iside][wanB];
      Bnum = MP[GB_AN];

      l1 = atv_ijk_e[iside][Rn][1];
      l2 = atv_ijk_e[iside][Rn][2];
      l3 = atv_ijk_e[iside][Rn][3];

      kRn = k2*(double)l2 + k3*(double)l3;
      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);

      if (l1==0) {

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            /* S_alpha-alpha */
            S00_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S00_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            /* S_beta-beta */
            S00_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S00_e[iside][S11_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];

            a00=H_e[iside][0][GA_AN][LB_AN][i][j];
            a11=H_e[iside][1][GA_AN][LB_AN][i][j];
            a01r=H_e[iside][2][GA_AN][LB_AN][i][j];
            a01i=H_e[iside][3][GA_AN][LB_AN][i][j];

            b00=iHNL_e[iside][0][GA_AN][LB_AN][i][j];
            b11=iHNL_e[iside][1][GA_AN][LB_AN][i][j];
            b01=iHNL_e[iside][2][GA_AN][LB_AN][i][j];
            /* H_alpha-alpha */
            H00_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*a00 - si*b00;
            H00_e[iside][S00_ref(Anum+i,Bnum+j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            H00_e[iside][S01_ref(Anum+i,Bnum+j)].r += co*a01r - si*(a01i+b01);
            H00_e[iside][S01_ref(Anum+i,Bnum+j)].i += si*a01r + co*(a01i+b01);
            /* H_beta-beta */
            H00_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*a11 - si*b11;
            H00_e[iside][S11_ref(Anum+i,Bnum+j)].i += co*b11 + si*a11;

	  }
	}
      }

      if (l1==direction) {

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
            /* S_alpha-alpha */
            S01_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S01_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            /* S_beta-beta */
            S01_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S01_e[iside][S11_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];

            a00=H_e[iside][0][GA_AN][LB_AN][i][j];
            a11=H_e[iside][1][GA_AN][LB_AN][i][j];
            a01r=H_e[iside][2][GA_AN][LB_AN][i][j];
            a01i=H_e[iside][3][GA_AN][LB_AN][i][j];

            b00=iHNL_e[iside][0][GA_AN][LB_AN][i][j];
            b11=iHNL_e[iside][1][GA_AN][LB_AN][i][j];
            b01=iHNL_e[iside][2][GA_AN][LB_AN][i][j];
            /* H_alpha-alpha */
            H01_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*a00 - si*b00;
            H01_e[iside][S00_ref(Anum+i,Bnum+j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            H01_e[iside][S01_ref(Anum+i,Bnum+j)].r += co*a01r - si*(a01i+b01);
            H01_e[iside][S01_ref(Anum+i,Bnum+j)].i += si*a01r + co*(a01i+b01);
            /* H_beta-beta */
            H01_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*a11 - si*b11;
            H01_e[iside][S11_ref(Anum+i,Bnum+j)].i += co*b11 + si*a11;
            /*
	    S01_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
	    S01_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];

	    for (k=0; k<=SpinP_switch_e[iside]; k++ ){
	      H01_e[iside][k][S00_ref(Anum+i,Bnum+j)].r += co*H_e[iside][k][GA_AN][LB_AN][i][j];
	      H01_e[iside][k][S00_ref(Anum+i,Bnum+j)].i += si*H_e[iside][k][GA_AN][LB_AN][i][j];
	    }
            */
	  }
	}
      }
    }
  }  /* GA_AN */

   for (i=1; i<=NUM0; i++) {
     for (j=1; j<=NUM0; j++) {

       H00_e[iside][(j-1)*NUM+i+NUM0-1].r =  H00_e[iside][(i+NUM0-1)*NUM+j-1].r;
       H00_e[iside][(j-1)*NUM+i+NUM0-1].i = -H00_e[iside][(i+NUM0-1)*NUM+j-1].i;

       H01_e[iside][(j-1)*NUM+i+NUM0-1].r =  H01_e[iside][(i+NUM0-1)*NUM+j-1].r;
       H01_e[iside][(j-1)*NUM+i+NUM0-1].i = -H01_e[iside][(i+NUM0-1)*NUM+j-1].i;

      }
    }


 /* to set some very small value to zero to avoid numerical instability*/
/*
  for (i=0; i<n2*n2; i++){

    if ( (fabs(S00_e[iside][i].r)+fabs(S00_e[iside][i].r)) < epscutoff ) {
      S00_e[iside][i].r = 0.0;
      S00_e[iside][i].i = 0.0;
    }

    if ( (fabs(S01_e[iside][i].r)+fabs(S01_e[iside][i].r)) < epscutoff ) {
      S01_e[iside][i].r = 0.0;
      S01_e[iside][i].i = 0.0;
    }

    if ( (fabs(H00_e[iside][i].r)+fabs(H00_e[iside][i].r)) < epscutoff ) {
      H00_e[iside][i].r = 0.0;
      H00_e[iside][i].i = 0.0;
    }

    if ( (fabs(H01_e[iside][i].r)+fabs(H01_e[iside][i].r)) < epscutoff ) {
      H01_e[iside][i].r = 0.0;
      H01_e[iside][i].i = 0.0;
    }

  }
*/

  /* free arrays */

  free(MP);
} 




void MTRAN_Set_CentOverlap( 
			   int job, 
			   int SpinP_switch, 
			   double k2,
			   double k3,
			   int NUM_c,
			   int NUM_e[2],
			   double *****H, 
                           double *****iHNL, 
			   double *****OLP,
			   int atomnum,
			   int atomnum_e[2],
			   int *WhatSpecies,
			   int *WhatSpecies_e[2],
			   int *Spe_Total_CNO,
			   int *Spe_Total_CNO_e[2],
			   int *FNAN,
			   int **natn,
			   int **ncn, 
			   int **atv_ijk,
			   int *TRAN_region,
			   int *TRAN_Original_Id 
			   )

#define SCC00_ref(i,j) ( ((j)-1)*nc+(i)-1 )
#define SCC01_ref(i,j) ( ((j+nc0)-1)*nc+(i)-1 )
#define SCC10_ref(i,j) ( ((j)-1)*nc+(i+nc0)-1 )
#define SCC11_ref(i,j) ( ((j+nc0)-1)*nc+(i+nc0)-1 )

#define SCL00_ref(i,j) ( ((j)-1)*nc+(i)-1 )
#define SCL01_ref(i,j) ( ((j+na0)-1)*nc+(i)-1 )
#define SCL10_ref(i,j) ( ((j)-1)*nc+(i+nc0)-1 )
#define SCL11_ref(i,j) ( ((j+na0)-1)*nc+(i+nc0)-1 )

#define SCR00_ref(i,j) ( ((j)-1)*nc+(i)-1 )
#define SCR01_ref(i,j) ( ((j+nb0)-1)*nc+(i)-1 )
#define SCR10_ref(i,j) ( ((j)-1)*nc+(i+nc0)-1 )
#define SCR11_ref(i,j) ( ((j+nb0)-1)*nc+(i+nc0)-1 )

#define S00l00_ref(i,j) ( ((j)-1)*na+(i)-1 )
#define S00l01_ref(i,j) ( ((j+na0)-1)*na+(i)-1 )
#define S00l10_ref(i,j) ( ((j)-1)*na+(i+na0)-1 )
#define S00l11_ref(i,j) ( ((j+na0)-1)*na+(i+na0)-1 )

#define S00r00_ref(i,j) ( ((j)-1)*nb+(i)-1 )
#define S00r01_ref(i,j) ( ((j+nb0)-1)*nb+(i)-1 )
#define S00r10_ref(i,j) ( ((j)-1)*nb+(i+nb0)-1 )
#define S00r11_ref(i,j) ( ((j+nb0)-1)*nb+(i+nb0)-1 )

{
  int *MP, *MP_e[2];
  int i;
  int nc0,nc,na0,na,nb0,nb;

  /* setup MP */
  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  MTRAN_Set_MP( 1,  atomnum, WhatSpecies, Spe_Total_CNO, &i, MP);
  nc0=i;
  nc=2*i;

  MP_e[0] = (int*)malloc(sizeof(int)*(atomnum_e[0]+1));
  MTRAN_Set_MP( 1,  atomnum_e[0], WhatSpecies_e[0], Spe_Total_CNO_e[0], &i, MP_e[0]);
  na0=i;
  na=2*i;

  MP_e[1] = (int*)malloc(sizeof(int)*(atomnum_e[1]+1));
  MTRAN_Set_MP( 1,  atomnum_e[1], WhatSpecies_e[1], Spe_Total_CNO_e[1], &i, MP_e[1]);
  nb0=i;
  nb=2*i;


  for (i=0; i<NUM_c*NUM_c; i++) {
    SCC[i].r = 0.0;
    SCC[i].i = 0.0;
  }
  for (i=0; i<NUM_c*NUM_e[0]; i++) {
    SCL[i].r = 0.0;
    SCL[i].i = 0.0;
  }
  for (i=0; i<NUM_c*NUM_e[1]; i++) {
    SCR[i].r = 0.0;
    SCR[i].i = 0.0;
  }

  for (i=0; i<NUM_c*NUM_c; i++) {
    HCC[i].r = 0.0;
    HCC[i].i = 0.0;
    /* S MitsuakiKAWAMURA */
    VCC[i].r = 0.0;
    VCC[i].i = 0.0;
    /* E MitsuakiKAWAMURA */
  }

  for (i=0; i<NUM_c*NUM_e[0]; i++) {
    HCL[i].r = 0.0;
    HCL[i].i = 0.0;
  }

  for (i=0; i<NUM_c*NUM_e[1]; i++) {
    HCR[i].r = 0.0;
    HCR[i].i = 0.0;
  }


  if ((job&1)==1) {

    int MA_AN,GA_AN, wanA, tnoA, Anum;
    int LB_AN, GB_AN, wanB, tnoB, l1,l2,l3, Bnum;
    int i,j,k,q,AN;
    int Rn;
    double kRn,si,co;
    double tmp,s0,h0;
    double a00,a11,a01r,a01i,b00,b11,b01;

    /* make Overlap ,  HCC, SCC               */
    /* parallel global GA_AN 1:atomnum        */

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

        GB_AN = natn[GA_AN][LB_AN];
        Rn = ncn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        Bnum = MP[GB_AN];

        l1 = atv_ijk[Rn][1];
        l2 = atv_ijk[Rn][2];
        l3 = atv_ijk[Rn][3];

        kRn = k2*(double)l2 + k3*(double)l3;
        si = sin(2.0*PI*kRn);
        co = cos(2.0*PI*kRn);

        for (i=0; i<tnoA; i++){
          for (j=0; j<tnoB; j++){

            /* l1 is the direction to the electrode */

            if (l1==0){

            /* S_alpha-alpha */
            SCC[SCC00_ref(Anum+i,Bnum+j)].r += co*OLP[0][GA_AN][LB_AN][i][j];
            SCC[SCC00_ref(Anum+i,Bnum+j)].i += si*OLP[0][GA_AN][LB_AN][i][j];
            /* S_beta-beta */
            SCC[SCC11_ref(Anum+i,Bnum+j)].r += co*OLP[0][GA_AN][LB_AN][i][j];
            SCC[SCC11_ref(Anum+i,Bnum+j)].i += si*OLP[0][GA_AN][LB_AN][i][j];

            a00=H[0][GA_AN][LB_AN][i][j];
            a11=H[1][GA_AN][LB_AN][i][j];
            a01r=H[2][GA_AN][LB_AN][i][j];
            a01i=H[3][GA_AN][LB_AN][i][j];

            b00=iHNL[0][GA_AN][LB_AN][i][j];
            b11=iHNL[1][GA_AN][LB_AN][i][j];
            b01=iHNL[2][GA_AN][LB_AN][i][j];

            /* H_alpha-alpha */
            HCC[SCC00_ref(Anum+i,Bnum+j)].r += co*a00 - si*b00;
            HCC[SCC00_ref(Anum+i,Bnum+j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            HCC[SCC01_ref(Anum+i,Bnum+j)].r += co*a01r - si*(a01i+b01);
            HCC[SCC01_ref(Anum+i,Bnum+j)].i += si*a01r + co*(a01i+b01);
            /* H_beta-beta */
            HCC[SCC11_ref(Anum+i,Bnum+j)].r += co*a11 - si*b11;
            HCC[SCC11_ref(Anum+i,Bnum+j)].i += co*b11 + si*a11;

            a00 = (H[0][GA_AN][LB_AN][i][j] - H0[0][GA_AN][LB_AN][i][j]);
            a11 = (H[1][GA_AN][LB_AN][i][j] - H0[0][GA_AN][LB_AN][i][j]);

            /* H_alpha-alpha */
            VCC[SCC00_ref(Anum + i, Bnum + j)].r += co*a00 - si*b00;
            VCC[SCC00_ref(Anum + i, Bnum + j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            VCC[SCC01_ref(Anum + i, Bnum + j)].r += co*a01r - si*(a01i + b01);
            VCC[SCC01_ref(Anum + i, Bnum + j)].i += si*a01r + co*(a01i + b01);
            /* H_beta-beta */
            VCC[SCC11_ref(Anum + i, Bnum + j)].r += co*a11 - si*b11;
            VCC[SCC11_ref(Anum + i, Bnum + j)].i += co*b11 + si*a11;

            }

          }
        }

      } /* LB_AN */
    }   /* MA_AN */

    for (i=1; i<=nc0; i++) {
     for (j=1; j<=nc0; j++) {

       HCC[(j - 1)*nc + i + nc0 - 1].r =  HCC[(i + nc0 - 1)*nc + j - 1].r;
       HCC[(j - 1)*nc + i + nc0 - 1].i = -HCC[(i + nc0 - 1)*nc + j - 1].i;

       VCC[(j - 1)*nc + i + nc0 - 1].r =  VCC[(i + nc0 - 1)*nc + j - 1].r;
       VCC[(j - 1)*nc + i + nc0 - 1].i = -VCC[(i + nc0 - 1)*nc + j - 1].i;

     }
    }


  }     /* job&1 */


  if ( (job&2) == 2 ) {

    {
      int MA_AN, GA_AN, wanA, tnoA, Anum;
      int GA_AN_e, Anum_e;
      int GB_AN, wanB, tnoB, Bnum;
      int GB_AN_e, Bnum_e;
      int i,j,k;
      int iside;

      /* overwrite CL1 region */

      iside = 0;

      /* parallel global GA_AN 1:atomnum */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];

        for (GB_AN=1; GB_AN<=atomnum; GB_AN++){

          if ( TRAN_region[GA_AN]==12 && TRAN_region[GB_AN]==12 ) {

            GA_AN_e = TRAN_Original_Id[GA_AN];
            Anum_e = MP_e[iside][GA_AN_e];

            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            Bnum = MP[GB_AN];

            GB_AN_e = TRAN_Original_Id[GB_AN];
            Bnum_e = MP_e[iside][GB_AN_e];

            for (i=0; i<tnoA; i++){
              for (j=0; j<tnoB; j++){

                /* S_alpha-alpha */
                SCC[SCC00_ref(Anum + i, Bnum + j)].r = S00_e[iside][S00l00_ref(Anum_e + i, Bnum_e + j)].r;
                /* S_beta-beta */
                SCC[SCC11_ref(Anum + i, Bnum + j)].r = S00_e[iside][S00l11_ref(Anum_e + i, Bnum_e + j)].r;

                /* V_alpha-alpha */
                VCC[SCC00_ref(Anum + i, Bnum + j)].r = VCC[SCC00_ref(Anum + i, Bnum + j)].r
                  + H00_e[iside][S00l00_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC00_ref(Anum + i, Bnum + j)].r;
                /* V_alpha-beta */
                VCC[SCC01_ref(Anum + i, Bnum + j)].r = VCC[SCC01_ref(Anum + i, Bnum + j)].r
                  + H00_e[iside][S00l01_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC01_ref(Anum + i, Bnum + j)].r;
                /* V_beta-alpha */
                VCC[SCC10_ref(Anum + i, Bnum + j)].r = VCC[SCC10_ref(Anum + i, Bnum + j)].r
                  + H00_e[iside][S00l10_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC10_ref(Anum + i, Bnum + j)].r;
                /* V_beta-beta */
                VCC[SCC11_ref(Anum + i, Bnum + j)].r = VCC[SCC11_ref(Anum + i, Bnum + j)].r
                  + H00_e[iside][S00l11_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC11_ref(Anum + i, Bnum + j)].r;

                /* H_alpha-alpha */
                HCC[SCC00_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00l00_ref(Anum_e + i, Bnum_e + j)].r;
                /* H_alpha-beta */
                HCC[SCC01_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00l01_ref(Anum_e + i, Bnum_e + j)].r;
                /* H_beta-alpha */
                HCC[SCC10_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00l10_ref(Anum_e + i, Bnum_e + j)].r;
                /* H_beta-beta */
                HCC[SCC11_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00l11_ref(Anum_e + i, Bnum_e + j)].r;

                /* S_alpha-alpha */
                SCC[SCC00_ref(Anum + i, Bnum + j)].i = S00_e[iside][S00l00_ref(Anum_e + i, Bnum_e + j)].i;
                /* S_beta-beta */
                SCC[SCC11_ref(Anum + i, Bnum + j)].i = S00_e[iside][S00l11_ref(Anum_e + i, Bnum_e + j)].i;

                /* V_alpha-alpha */
                VCC[SCC00_ref(Anum + i, Bnum + j)].i = VCC[SCC00_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00l00_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC00_ref(Anum + i, Bnum + j)].i;
                /* V_alpha-beta */
                VCC[SCC01_ref(Anum + i, Bnum + j)].i = VCC[SCC01_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00l01_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC01_ref(Anum + i, Bnum + j)].i;
                /* V_beta-alpha */
                VCC[SCC10_ref(Anum + i, Bnum + j)].i = VCC[SCC10_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00l10_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC10_ref(Anum + i, Bnum + j)].i;
                /* V_beta-beta */
                VCC[SCC11_ref(Anum + i, Bnum + j)].i = VCC[SCC11_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00l11_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC11_ref(Anum + i, Bnum + j)].i;

                /* H_alpha-alpha */
                HCC[SCC00_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00l00_ref(Anum_e + i, Bnum_e + j)].i;
                /* H_alpha-beta */
                HCC[SCC01_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00l01_ref(Anum_e + i, Bnum_e + j)].i;
                /* H_beta-alpha */
                HCC[SCC10_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00l10_ref(Anum_e + i, Bnum_e + j)].i;
                /* H_beta-beta */
                HCC[SCC11_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00l11_ref(Anum_e + i, Bnum_e + j)].i;
                /*
                SCC[SCC_ref(Anum+i,Bnum+j)] = S00_e[iside][S00l_ref(Anum_e+i,Bnum_e+j)];

                for (k=0; k<=SpinP_switch; k++) {
                  HCC[k][SCC_ref(Anum+i,Bnum+j)] = H00_e[iside][k][S00l_ref(Anum_e+i,Bnum_e+j)];
                }  */
              }
            }

          }
        }
      }
    }

    {
      int MA_AN, GA_AN, wanA, tnoA, Anum;
      int GA_AN_e, Anum_e;
      int GB_AN, wanB, tnoB, Bnum;
      int GB_AN_e, Bnum_e;
      int i,j,k;
      int iside;

      /* overwrite CR1 region */

      iside = 1;

      /*parallel global GA_AN  1:atomnum */
      /*parallel local  MA_AN  1:Matomnum */
      /*parallel variable GA_AN = M2G[MA_AN] */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];

        for (GB_AN=1; GB_AN<=atomnum; GB_AN++){

          if ( TRAN_region[GA_AN]==13 && TRAN_region[GB_AN]==13 )  {

            GA_AN_e = TRAN_Original_Id[GA_AN];
            Anum_e = MP_e[iside][GA_AN_e]; /* = Anum */
            wanB = WhatSpecies[GB_AN];
            tnoB = Spe_Total_CNO[wanB];
            Bnum = MP[GB_AN];

            GB_AN_e = TRAN_Original_Id[GB_AN];
            Bnum_e = MP_e[iside][GB_AN_e];

            for (i=0; i<tnoA; i++){
              for (j=0; j<tnoB; j++){

                /* S_alpha-alpha */
                SCC[SCC00_ref(Anum + i, Bnum + j)].r = S00_e[iside][S00r00_ref(Anum_e + i, Bnum_e + j)].r;
                /* S_beta-beta */
                SCC[SCC11_ref(Anum + i, Bnum + j)].r = S00_e[iside][S00r11_ref(Anum_e + i, Bnum_e + j)].r;

                /* H_alpha-alpha */
                VCC[SCC00_ref(Anum + i, Bnum + j)].r = VCC[SCC00_ref(Anum + i, Bnum + j)].r
                  +H00_e[iside][S00r00_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC00_ref(Anum + i, Bnum + j)].r;
                /* H_alpha-beta */
                VCC[SCC01_ref(Anum + i, Bnum + j)].r = VCC[SCC01_ref(Anum + i, Bnum + j)].r
                  +H00_e[iside][S00r01_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC01_ref(Anum + i, Bnum + j)].r;
                /* H_beta-alpha */
                VCC[SCC10_ref(Anum + i, Bnum + j)].r = VCC[SCC10_ref(Anum + i, Bnum + j)].r
                  +H00_e[iside][S00r10_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC10_ref(Anum + i, Bnum + j)].r;
                /* H_beta-beta */
                VCC[SCC11_ref(Anum + i, Bnum + j)].r = VCC[SCC11_ref(Anum + i, Bnum + j)].r
                  +H00_e[iside][S00r11_ref(Anum_e + i, Bnum_e + j)].r - HCC[SCC11_ref(Anum + i, Bnum + j)].r;

                /* H_alpha-alpha */
                HCC[SCC00_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00r00_ref(Anum_e + i, Bnum_e + j)].r;
                /* H_alpha-beta */
                HCC[SCC01_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00r01_ref(Anum_e + i, Bnum_e + j)].r;
                /* H_beta-alpha */
                HCC[SCC10_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00r10_ref(Anum_e + i, Bnum_e + j)].r;
                /* H_beta-beta */
                HCC[SCC11_ref(Anum + i, Bnum + j)].r = H00_e[iside][S00r11_ref(Anum_e + i, Bnum_e + j)].r;

                /* S_alpha-alpha */
                SCC[SCC00_ref(Anum + i, Bnum + j)].i = S00_e[iside][S00r00_ref(Anum_e + i, Bnum_e + j)].i;
                /* S_beta-beta */
                SCC[SCC11_ref(Anum + i, Bnum + j)].i = S00_e[iside][S00r11_ref(Anum_e + i, Bnum_e + j)].i;

                /* H_alpha-alpha */
                VCC[SCC00_ref(Anum + i, Bnum + j)].i = VCC[SCC00_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00r00_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC00_ref(Anum + i, Bnum + j)].i;
                /* H_alpha-beta */
                VCC[SCC01_ref(Anum + i, Bnum + j)].i = VCC[SCC01_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00r01_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC01_ref(Anum + i, Bnum + j)].i;
                /* H_beta-alpha */
                VCC[SCC10_ref(Anum + i, Bnum + j)].i = VCC[SCC10_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00r10_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC10_ref(Anum + i, Bnum + j)].i;
                /* H_beta-beta */
                VCC[SCC11_ref(Anum + i, Bnum + j)].i = VCC[SCC11_ref(Anum + i, Bnum + j)].i
                  + H00_e[iside][S00r11_ref(Anum_e + i, Bnum_e + j)].i - HCC[SCC11_ref(Anum + i, Bnum + j)].i;

                /* H_alpha-alpha */
                HCC[SCC00_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00r00_ref(Anum_e + i, Bnum_e + j)].i;
                /* H_alpha-beta */
                HCC[SCC01_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00r01_ref(Anum_e + i, Bnum_e + j)].i;
                /* H_beta-alpha */
                HCC[SCC10_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00r10_ref(Anum_e + i, Bnum_e + j)].i;
                /* H_beta-beta */
                HCC[SCC11_ref(Anum + i, Bnum + j)].i = H00_e[iside][S00r11_ref(Anum_e + i, Bnum_e + j)].i;
                /*
                SCC[SCC_ref(Anum+i,Bnum+j)] = S00_e[iside][S00r_ref(Anum_e+i,Bnum_e+j)];

                for (k=0; k<=SpinP_switch; k++) {
                  HCC[k][SCC_ref(Anum+i,Bnum+j)] = H00_e[iside][k][S00r_ref(Anum_e+i,Bnum_e+j)];
                } */
              }
            }

          }

        }
      }
    }

    {
      int iside;
      int MA_AN, GA_AN, wanA, tnoA, Anum, GA_AN_e, Anum_e;
      int GB_AN_e, wanB_e, tnoB_e, Bnum_e;
      int i,j,k;

      /* make Overlap,  HCL, SCL from OLP_e, and H_e*/

      iside = 0;

      /*parallel global GA_AN  1:atomnum */
      /*parallel local  MA_AN  1:Matomnum */
      /*parallel variable GA_AN = M2G[MA_AN] */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

        if (TRAN_region[GA_AN]%10==2  || TRAN_region[GA_AN]==6) {

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];  /* GA_AN is in C */

        GA_AN_e =  TRAN_Original_Id[GA_AN];
        Anum_e = MP_e[iside][GA_AN_e];

        for (GB_AN_e=1; GB_AN_e<=atomnum_e[iside]; GB_AN_e++) {

          wanB_e = WhatSpecies_e[iside][GB_AN_e];
          tnoB_e = Spe_Total_CNO_e[iside][wanB_e];
          Bnum_e = MP_e[iside][GB_AN_e];

          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB_e; j++){

            /* S_alpha-alpha */
            SCL[SCL00_ref(Anum+i,Bnum_e+j)].r = S01_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_beta-beta */
            SCL[SCL11_ref(Anum+i,Bnum_e+j)].r = S01_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].r;

            /* H_alpha-alpha */
            HCL[SCL00_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_alpha-beta */
            HCL[SCL01_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00l01_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_beta-alpha */
            HCL[SCL10_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00l10_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_beta-beta */
            HCL[SCL11_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_alpha-alpha */
            SCL[SCL00_ref(Anum+i,Bnum_e+j)].i = S01_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].i;
            /* S_beta-beta */
            SCL[SCL11_ref(Anum+i,Bnum_e+j)].i = S01_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].i;

            /* H_alpha-alpha */
            HCL[SCL00_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_alpha-beta */
            HCL[SCL01_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00l01_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_beta-alpha */
            HCL[SCL10_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00l10_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_beta-beta */
            HCL[SCL11_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].i;
       /*
              SCL[SCL_ref(Anum+i,Bnum_e+j)] = S01_e[iside][ S00l_ref(Anum_e+i, Bnum_e+j)];

              for (k=0; k<=SpinP_switch; k++) {
                HCL[k][SCL_ref(Anum+i,Bnum_e+j)] = H01_e[iside][k][ S00l_ref(Anum_e+i, Bnum_e+j)];
              } */
            }
          }
         }
        }
      }
    }

    {
      int iside;
      int MA_AN, GA_AN, wanA, tnoA, Anum, GA_AN_e, Anum_e;
      int GB_AN_e, wanB_e, tnoB_e, Bnum_e;
      int i,j,k;

      /* make Overlap ,  HCR, SCR from OLP_e, and H_e*/

      iside = 1;

/*      for (i=0; i<NUM_c*NUM_e[iside]; i++) {
      SCR_nc[i].r = 0.0;
      SCR_nc[i].i = 0.0;
      HCR_nc[i].r = 0.0;
      HCR_nc[i].i = 0.0;
    } */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

        if (TRAN_region[GA_AN]%10==3  || TRAN_region[GA_AN]==6 ) {

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];  /* GA_AN is in C */
        GA_AN_e =  TRAN_Original_Id[GA_AN];
        Anum_e = MP_e[iside][GA_AN_e];

        for (GB_AN_e=1; GB_AN_e<=atomnum_e[iside];GB_AN_e++) {
          wanB_e = WhatSpecies_e[iside][GB_AN_e];
          tnoB_e = Spe_Total_CNO_e[iside][wanB_e];
          Bnum_e = MP_e[iside][GB_AN_e];
          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB_e; j++){

            /* S_alpha-alpha */
            SCR[SCR00_ref(Anum+i,Bnum_e+j)].r = S01_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_beta-beta */
            SCR[SCR11_ref(Anum+i,Bnum_e+j)].r = S01_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].r;

            /* H_alpha-alpha */
            HCR[SCR00_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_alpha-beta */
            HCR[SCR01_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00r01_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_beta-alpha */
            HCR[SCR10_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00r10_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_beta-beta */
            HCR[SCR11_ref(Anum+i,Bnum_e+j)].r = H01_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].r;

            /* S_alpha-alpha */
            SCR[SCR00_ref(Anum+i,Bnum_e+j)].i = S01_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].i;
            /* S_beta-beta */
            SCR[SCR11_ref(Anum+i,Bnum_e+j)].i = S01_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].i;

            /* H_alpha-alpha */
            HCR[SCR00_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_alpha-beta */
            HCR[SCR01_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00r01_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_beta-alpha */
            HCR[SCR10_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00r10_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_beta-beta */
            HCR[SCR11_ref(Anum+i,Bnum_e+j)].i = H01_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].i;
         /*
              SCR[SCR_ref(Anum+i,Bnum_e+j)] = S01_e[iside][ S00r_ref(Anum_e+i, Bnum_e+j)];

              for (k=0; k<=SpinP_switch; k++) {
                HCR[k][SCR_ref(Anum+i,Bnum_e+j)] = H01_e[iside][k][ S00r_ref(Anum_e+i, Bnum_e+j)];
              } */
            }
          }
         }
        }
      }
    }

  } /* job&2 */


  free(MP);
  free(MP_e[1]);
  free(MP_e[0]);
}



void MTRAN_Allocate_HS(int NUM_c,
                       int NUM_e[2],
                       int SpinP_switch)
{
  int i,side;

  for (side=0; side<=1; side++) {

    S00_e[side] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[side]*NUM_e[side] );
    S01_e[side] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[side]*NUM_e[side] );
    for (i=0; i<(NUM_e[side]*NUM_e[side]); i++) {
      S00_e[side][i].r = 0.0;
      S00_e[side][i].i = 0.0;
      S01_e[side][i].r = 0.0;
      S01_e[side][i].i = 0.0;
    }

      H00_e[side] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[side]*NUM_e[side] );
      H01_e[side] = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[side]*NUM_e[side] );
      for (i=0; i<(NUM_e[side]*NUM_e[side]); i++) {
        H00_e[side][i].r = 0.0;
        H00_e[side][i].i = 0.0;
        H01_e[side][i].r = 0.0;
        H01_e[side][i].i = 0.0;
    }
  }

  SCC = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c ); 
  SCL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0] ); 
  SCR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1] ); 

  for (i=0; i<NUM_c*NUM_c; i++) {
    SCC[i].r = 0.0;
    SCC[i].i = 0.0;
  }
  for (i=0; i<NUM_c*NUM_e[0]; i++) {
    SCL[i].r = 0.0;
    SCL[i].i = 0.0;
  }
  for (i=0; i<NUM_c*NUM_e[1]; i++) {
    SCR[i].r = 0.0;
    SCR[i].i = 0.0;
  }

  /* S MitsuakiKAWAMURA*/
    HCC = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c); 
    VCC = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_c);
    for (i = 0; i<NUM_c*NUM_c; i++) {
      HCC[i].r = 0.0;
      HCC[i].i = 0.0;
      VCC[i].r = 0.0;
      VCC[i].i = 0.0;
    }
    /* S MitsuakiKAWAMURA*/

    HCL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]); 
    for (i=0; i<NUM_c*NUM_e[0]; i++) {
      HCL[i].r = 0.0;
      HCL[i].i = 0.0;
    }

    HCR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]); 
    for (i=0; i<NUM_c*NUM_e[1]; i++) {
      HCR[i].r = 0.0;
      HCR[i].i = 0.0;
    }

}

/*S MitsuakiKawaMura*/
static void MTRAN_Free_All_NC()
{
  int i, i2, i3, k, iside;
  int Gc_AN, Cwan, tno0;
  int h_AN;

  for (iside = 0; iside <= 1; iside++) {
    free(S00_e[iside]);
    free(S01_e[iside]);
    free(H00_e[iside]);
    free(H01_e[iside]);
  }

  free(SCC);
  free(HCC);
  free(VCC);
  free(SCL);
  free(SCR);
  free(HCL);
  free(HCR);
  /*
  Malloc in MTRAN_Input
  */
  if (tran_transmission_on) {

    for (i2 = 0; i2 < TRAN_TKspace_grid2; i2++) {
      for (i3 = 0; i3 < TRAN_TKspace_grid3; i3++) {
        for (i = 0; i<3; i++) {
          free(tran_transmission[i2][i3][i]);
        }
        free(tran_transmission[i2][i3]);
      }
      free(tran_transmission[i2]);
    }
    free(tran_transmission);
  }

  if (TRAN_Channel == 1) {

    for (i = 0; i < TRAN_Channel_Nkpoint; i++) {
      free(TRAN_Channel_kpoint[i]);
    } /* for (i = 0; i<(TRAN_Channel_Nkpoint + 1); i++) */
    free(TRAN_Channel_kpoint);

    free(TRAN_Channel_energy);
  }
  /*
  Malloc in MTRAN_Read_Tran_HS
  */
  free(TRAN_region);
  free(TRAN_Original_Id);
  free(WhatSpecies);
  free(Spe_Total_CNO);
  free(FNAN);

  for (i = 0; i <= (atomnum); i++) {
    free(natn[i]);
    free(ncn[i]);
    free(atv_ijk[i]);
  }
  free(natn);
  free(ncn);
  free(atv_ijk);

  for (k = 0; k<4; k++) {

    FNAN[0] = 0;
    for (Gc_AN = 0; Gc_AN <= (atomnum); Gc_AN++) {

      if (Gc_AN == 0) {
        tno0 = 1;
      }
      else {
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_CNO[Cwan];
      }

      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++) {

        for (i = 0; i < tno0; i++) {
          free(OLP[k][Gc_AN][h_AN][i]);
          free(H[k][Gc_AN][h_AN][i]);
          free(H0[k][Gc_AN][h_AN][i]);
          free(iHNL[k][Gc_AN][h_AN][i]);
        }
        free(OLP[k][Gc_AN][h_AN]);
        free(H[k][Gc_AN][h_AN]);
        free(H0[k][Gc_AN][h_AN]);
        free(iHNL[k][Gc_AN][h_AN]);
      }
      free(OLP[k][Gc_AN]);
      free(H[k][Gc_AN]);
      free(H0[k][Gc_AN]);
      free(iHNL[k][Gc_AN]);
    }
    free(OLP[k]);
    free(H[k]);
    free(H0[k]);
    free(iHNL[k]);
  }
  free(OLP);
  free(H);
  free(H0);
  free(iHNL);

  /**********************************************
  informations of leads
  **********************************************/

  for (iside = 0; iside <= 1; iside++) {
 
    for (k = 0; k<4; k++) {

      FNAN_e[iside][0] = 0;
      for (Gc_AN = 0; Gc_AN <= atomnum_e[iside]; Gc_AN++) {

        if (Gc_AN == 0) {
          tno0 = 1;
        }
        else {
          Cwan = WhatSpecies_e[iside][Gc_AN];
          Cwan = WhatSpecies_e[iside][Gc_AN];
          tno0 = Spe_Total_CNO_e[iside][Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN_e[iside][Gc_AN]; h_AN++) {

          for (i = 0; i < tno0; i++) {
            free(OLP_e[iside][k][Gc_AN][h_AN][i]);
            free(H_e[iside][k][Gc_AN][h_AN][i]);
            free(iHNL_e[iside][k][Gc_AN][h_AN][i]);
          }
          free(OLP_e[iside][k][Gc_AN][h_AN]);
          free(H_e[iside][k][Gc_AN][h_AN]);
          free(iHNL_e[iside][k][Gc_AN][h_AN]);
        }
        free(OLP_e[iside][k][Gc_AN]);
        free(H_e[iside][k][Gc_AN]);
        free(iHNL_e[iside][k][Gc_AN]);
      }
      free(OLP_e[iside][k]);
      free(H_e[iside][k]);
      free(iHNL_e[iside][k]);
    }
    free(OLP_e[iside]);
    free(H_e[iside]);
    free(iHNL_e[iside]);
  }
  
  for (iside = 0; iside <= 1; iside++) {

    free(WhatSpecies_e[iside]);
    free(Spe_Total_CNO_e[iside]);
    free(FNAN_e[iside]);
    for (i = 0; i <= atomnum_e[iside]; i++) {
      free(natn_e[iside][i]);
      free(ncn_e[iside][i]);
    }
    free(natn_e[iside]);
    free(ncn_e[iside]);

    for (i=0; i<(TCpyCell+1); i++) {
      free(atv_ijk_e[iside][i]);
    }
    free(atv_ijk_e[iside]);
  }

}
/*E MitsuakiKawamura*/







