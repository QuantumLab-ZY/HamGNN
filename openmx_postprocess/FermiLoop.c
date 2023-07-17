#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef nompi
#include "mimic_mpi.h"
#else
#include "mpi.h"
#endif

#include "Tools_BandCalc.h"
#include "Inputtools.h"
#include "read_scfout.h"
#include "EigenValue_Problem.h"

/* Added by N. Yamaguchi ***/
#define PRINTFF(...)\
  do{\
    printf(__VA_ARGS__);\
    fflush(stdout);\
  }while(0)
#include "GetOrbital.h"
/* ***/

void func_Newton(double *k1, double E1, double *k2, double E2, double *k3, double *E3, double EF, int l, int loopMAX);
int func_periodicity(int *tail_Cc, int count_Pair3M, int *trace_Cc, int **k_index_hitD, int **pair_hit3M, int k1_domain, int k2_domain);

/* Added by N. Yamaguchi ***/
double scaleVector(double *v, double *v1, double *v2);
void vectorize(double scale, double *v1, double *v2, double *v, int n);
void func_Brent(double *k1, double E1, double *k2, double E2, double *k3, double *E3, double EF, int l, int loopMAX);
#ifdef SIGMAEK
int FermiLoop()
#else
#pragma intel optimization_level 1
  /* ***/

int main(int argc, char *argv[])

  /* Added by N. Yamaguchi ***/
#endif
  /* ***/

{

  FILE *fp, *fp1, *fp2;
  int i,j,k,l,m,n,n2, i1,i2, j1,j2, l2;           // loop variable

  /* Disabled by N. Yamaguchi
   * int l_max, l_min, l_cal;                        // band index
   */

  int *S_knum, *E_knum, T_knum, size_Tknum;       // MPI variable (k-point divide)
  int *S2_knum, *E2_knum, T2_knum;                // MPI variable (k-point divide)
  int count_Pair3M, i_Cc;                         // 3-mash method variable
  int trial_Newton = 5;
  int namelen, num_procs, myrank;                 // MPI_variable

  char fname_FS[256], fname_MP[256], fname_Spin[256];
  char Pdata_s[256];
  //  char processor_name[MPI_MAX_PROCESSOR_NAME];

  double k1, k2, k3;                              // k-point variable
  double d0, d1, d2, d3;
  double **k_xyz, *EIGEN_MP;                      // Eigen solve array
  double E1, E2, EF;                              // Eigen value
  double data1[5], data2[5], data3[5];
  double **k_xyz_Cc,*EIG_Cc, **MulP_Cc;
  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2

  double TStime, TEtime, Stime, Etime;            // Time variable
  double Time_EIG, *Time_Contour, *Time_MulP;     // Time measurement variable

  int fo_inp = 0;
  int fo_wf  = 0;
  int i_vec[20],i_vec2[20];                       // input variable
  char *s_vec[20];                                // input variable
  double r_vec[20];                               // input variable

  // ### (2-step Contour )    ###
  int count_hitD, i_hitD, **k_index_hitD;
  double **k_xyz_domain, *EIGEN_domain;

  int ***frame_hit3M, *count_hit3M, count_T_hit3M;
  double ***EIG_frame3M, ****k_xyz_frame3M;       //

  int **pair_hit3M;
  int hit_Cc;
  int head_Cc[8], tail_Cc[8], *trace_Cc;
  int hit_Total[2];

  // ### Orbital Data    ###
  int TNO_MAX ,ClaOrb_MAX[2] ,**ClaOrb;              // Orbital
  //  char
  char OrbSym[5][2], **OrbName, **An2Spe;
  double ***OrbMulP_Cc, ***Orb2MulP_Cc;

  // ### k_height    ###
  int i_height, k3_height;
  double kRange_height;

  /* Added by N. Yamaguchi ***/
  int cnt=0;
  /* ***/

  /* Added by N. Yamaguchi ***/
  int calcBandbyband, calcBandMax, calcBandMin;
  int *numK, *recvCount, *displs;
  double **MulP_CcDivision;
  double ***OrbMulP_CcDivision;
  double ***Orb2MulP_CcDivision;
  int sizeMulP=Spin_Dege ? 8 : 4;
#ifndef DEBUG_SIGMAEK_OLDER_20181126
  mode=(int(*)())FermiLoop;
#endif
  /* ***/

  // ### MPI_Init ############################################

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  /* ***/

  MPI_Init(&argc, &argv);

  /* Added by N. Yamaguchi ***/
#endif
  /* ***/

  MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //  MPI_Get_processor_name(processor_name, &namelen);
  //  printf("myrank:%d \n",myrank);
  dtime(&TStime);

  // ### INPUT_FILE ##########################################

  /* Disabled by N. Yamaguchi
   * if (myrank ==0) printf("\n");
   */

  /* Added by N. Yamaguchi ***/
  if (myrank ==0) PRINTFF("\n");
  /* ***/

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  /* ***/

  sprintf(fname,"%s",argv[1]);

  if((fp = fopen(fname,"r")) != NULL){
    fo_inp = 1;

    /* Disabled by N. Yamaguchi
     * if (myrank ==0) printf("open \"%s\" file... \n" ,fname);
     */

    /* Added by N. Yamaguchi ***/
    if (myrank ==0) PRINTFF("open \"%s\" file... \n" ,fname);
    /* ***/

    fclose(fp);
  }else{
    fo_inp = 0;

    /* Disabled by N. Yamaguchi
     * if (myrank ==0) printf("Cannot open \"%s\" file.\n" ,fname);
     */

    /* Added by N. Yamaguchi ***/
    if (myrank ==0) PRINTFF("Cannot open \"%s\" file.\n" ,fname);
    /* ***/

  }

  /* Added by N. Yamaguchi ***/
#else
  fo_inp=1;
#endif
  /* ***/

  if (fo_inp == 1){
    input_open(fname);

    /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
    /* ***/

    input_string("Filename.scfout",fname_wf,"default");
    input_string("Filename.outdata",fname_out,"default");

    /* Added by N. Yamaguchi ***/
#endif
    /* ***/

    /* Disabled by N. Yamaguchi ***
       r_vec[0]=-0.2; r_vec[1]=0.1;
     * ***/

    /* Added by N. Yamaguchi ***/
    r_vec[0]=-0.5; r_vec[1]=0.5;
    /* ***/

    input_doublev("Energy.Range",2,E_Range,r_vec);
    if (E_Range[0]>E_Range[1]){
      d0 = E_Range[0];
      E_Range[0] = E_Range[1];
      E_Range[1] = d0;
    }//if
    r_vec[0]=0.0; r_vec[1]=0.0; r_vec[2]=0.0;
    input_doublev("Search.kCentral",3,k_CNT,r_vec);

    /* Disabled by N. Yamaguchi
     * input_int("Eigen.Newton",&switch_Eigen_Newton,0);
     * if (switch_Eigen_Newton != 0) switch_Eigen_Newton = 1;
     */

    input_int("Calc.Type.3mesh",&plane_3mesh,1);
    i_vec2[0]=2;  i_vec2[1]=2;
    input_intv("k-plane.1stStep",2,i_vec,i_vec2);

    /* Disabled by N. Yamaguchi
     * k1_3mesh = i_vec[0];    k2_3mesh = i_vec[1];
     */

    /* Added by N. Yamaguchi ***/
    k1_3mesh = i_vec[0]-1;    k2_3mesh = i_vec[1]-1;
    /* ***/

    /* Disabled by N. Yamaguchi
     * r_vec[0]=0.0; r_vec[1]=0.0;
     */

    /* Added by N. Yamaguchi ***/
    r_vec[0]=0.5; r_vec[1]=0.5;
    /* ***/

    input_doublev("kRange.3mesh",2,kRange_3mesh,r_vec);

    input_int("k-plane.height",&k3_height,1);

    /* Added by N. Yamaguchi ***/
    --k3_height;
    /* ***/

    input_double("kRange.height", &kRange_height, 0.5);

    r_vec[0]=1.0; r_vec[1]=1.0; r_vec[2]=1.0;
    input_doublev("MulP.Vec.Scale",3, MulP_VecScale, r_vec);

    /* Disabled by N. Yamaguchi ***
       input_int("Trial.Newton",&trial_Newton,5);
       if (trial_Newton<2) trial_Newton = 2;
     * ***/

    i_vec2[0]=3;  i_vec2[1]=3;
    input_intv("k-plane.2ndStep",2,i_vec,i_vec2);

    /* Disabled by N. Yamaguchi
     * k1_domain = i_vec[0];    k2_domain = i_vec[1]; 
     */

    /* Added by N. Yamaguchi ***/
    k1_domain = i_vec[0]-1;    k2_domain = i_vec[1]-1; 
    /* ***/

    /* Disabled by N. Yamaguchi
     * input_int("Spin.Degenerate",&Spin_Dege,0);
     * if((Spin_Dege<0) || (Spin_Dege>1)) Spin_Dege = 0;
     */

    /* Added by N. Yamaguchi ***/
    input_logical("Eigen.Brent",&switch_Eigen_Newton,1);
    input_int("Trial.Brent",&trial_Newton,5);
    if (trial_Newton<2) trial_Newton = 2;
    input_logical("Spin.Degenerate",&Spin_Dege,0);
    input_logical("Calc.Bandbyband", &calcBandbyband, 0);
    input_int("Calc.Band.Max",&calcBandMax,0);
    input_int("Calc.Band.Min",&calcBandMin,0);
    if (calcBandbyband) {
      if (calcBandMin==0 || calcBandMax==0) {
	PRINTFF("Error: Check the keywords 'Calc.Band.Max' and 'Calc.Band.Min'.\n");
#ifndef SIGMAEK
	MPI_Finalize();
#endif
	return 0;
      } else if (calcBandMax<calcBandMin) {
	int tmp=calcBandMin;
	calcBandMin=calcBandMax;
	calcBandMax=tmp;
      }
    }
    /* ***/

    input_close();
  }//fo_inp

  // ### READ_SCFout_FILE ####################################

  /* Added by N. Yamaguchi ***/
  int dir= plane_3mesh==1 ? 2 : (plane_3mesh==2 ? 3 : 1);
  int caseAcute= rtv[plane_3mesh][1]*rtv[dir][1]+rtv[plane_3mesh][2]*rtv[dir][2]+rtv[plane_3mesh][3]*rtv[dir][3]>0 ? 1 : 0;
#ifndef SIGMAEK
  /* ***/

  if (fo_inp == 1){
    if((fp = fopen(fname_wf,"r")) != NULL){
      fo_wf = 1;

      /* Disabled by N. Yamaguchi
       * if (myrank ==0) printf("\nInput filename is \"%s\"  \n\n", fname_wf);
       */

      /* Added by N. Yamaguchi ***/
      if (myrank ==0) PRINTFF("\nInput filename is \"%s\"  \n\n", fname_wf);
      /* ***/

      fclose(fp);
    }else{
      fo_wf = 0;

      /* Disabled by N. Yamaguchi
       * if (myrank ==0) printf("Cannot open *.scfout File. \"%s\" is not found.\n" ,fname_wf);
       */

      /* Added by N. Yamaguchi ***/
      if (myrank ==0) PRINTFF("Cannot open *.scfout File. \"%s\" is not found.\n" ,fname_wf);
      /* ***/

    }
  }//if(fo_inp)

  /* Added by N. Yamaguchi ***/
#else
  fo_wf=1;
#endif
  /* ***/

  if ((fo_inp == 1) && (fo_wf == 1)){
    // ### Get Calculation Data ##############################

    /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
    /* ***/

    // ### wave functions　###
    if (myrank == 0){
      read_scfout(fname_wf, 1);
    }else {
      read_scfout(fname_wf, 0);
    }

    /* Added by N. Yamaguchi ***/
#endif
    if (SpinP_switch!=3){
      PRINTFF("Error: \"FermiLoop\" is available only for non-collinear cases.\n");
#ifndef SIGMAEK
      MPI_Finalize();
#endif
      return 0;
    }
    /* ***/

    // ### Central k-point ###
    k_CNT1[0] = rtv[1][1]*k_CNT[0] +rtv[2][1]*k_CNT[1] +rtv[3][1]*k_CNT[2];
    k_CNT1[1] = rtv[1][2]*k_CNT[0] +rtv[2][2]*k_CNT[1] +rtv[3][2]*k_CNT[2];
    k_CNT1[2] = rtv[1][3]*k_CNT[0] +rtv[2][3]*k_CNT[1] +rtv[3][3]*k_CNT[2];

    // ### Total Num Orbs  ###
    TNO_MAX = 0;
    for(i=1;i<=atomnum;i++){
      if(TNO_MAX < Total_NumOrbs[i]) TNO_MAX = Total_NumOrbs[i];
    }
    // ### Classify Orbs   ###
    An2Spe = (char**)malloc(sizeof(char*)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      An2Spe[i] = (char*)malloc(sizeof(char)*(asize10));
    }
    //kotaka
    ClaOrb = (int**)malloc(sizeof(int*)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      ClaOrb[i] = (int*)malloc(sizeof(int)*(TNO_MAX+1));
      for (j=0; j<=TNO_MAX; j++) ClaOrb[i][j]=0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0){
      Classify_OrbNum(ClaOrb, An2Spe, 1);
    }else{
      Classify_OrbNum(ClaOrb, An2Spe, 0);
    }
    ClaOrb_MAX[1] = 0;
    for(i=1;i<=atomnum;i++){
      for (j=0; j<=TNO_MAX; j++){
	if(ClaOrb_MAX[1] < ClaOrb[i][j]) ClaOrb_MAX[1] = ClaOrb[i][j];
      }
    }
    ClaOrb_MAX[0] = 0;
    OrbSym[0][0] = 's';  OrbSym[0][1] = '\0';
    OrbName = (char**)malloc(sizeof(char*)*(ClaOrb_MAX[1]+1));
    for (i=0; i<=ClaOrb_MAX[1]; i++){
      OrbName[i] = (char*)malloc(sizeof(char)*3);
      if (i == 0){// 0
	OrbName[i][0]='s';  OrbName[i][1]=(char)(1+48);      OrbName[i][2]='\0';
      }else if(i > 0 && i < 4){// 1,2,3
	if (ClaOrb_MAX[0] < 1){ ClaOrb_MAX[0] = 1; OrbSym[1][0] = 'p';   OrbSym[1][1] = '\0';}
	OrbName[i][0]='p';  OrbName[i][1]=(char)((i+0)+48);  OrbName[i][2]='\0';
      }else if(i > 3 && i < 9){// 4,5,6,7,8
	if (ClaOrb_MAX[0] < 2){ ClaOrb_MAX[0] = 2; OrbSym[2][0] = 'd';   OrbSym[2][1] = '\0';}
	OrbName[i][0]='d';  OrbName[i][1]=(char)((i-3)+48);  OrbName[i][2]='\0';
      }else if(i >8 && i <16){// 9,10,11,12,13,14,15
	if (ClaOrb_MAX[0] < 3){ ClaOrb_MAX[0] = 3; OrbSym[3][0] = 'f';   OrbSym[3][1] = '\0';}
	OrbName[i][0]='f';  OrbName[i][1]=(char)((i-8)+48);  OrbName[i][2]='\0';
      }else{//16
	OrbName[i][0]=(char)((i-15)/10+48);
	OrbName[i][1]=(char)((i-15)%10+48);  OrbName[i][2]='\0';
      }
      //      if (myrank == 0) printf("OrbName:%s\n",OrbName[i]);
    }//i
    //    if (myrank == 0) for (i=1; i<=atomnum; i++){  printf("%4d %s\n", i, An2Spe[i]);  }
    // ### Band Total (n2) ###
    k = 1;
    for (i=1; i<=atomnum; i++){ k+= Total_NumOrbs[i]; }
    n = k - 1;    n2 = 2*k + 2;

    if (myrank == 0){

      /* Disabled by N. Yamaguchi
       * printf("########### ORBITAL DATA ##################\n");
       * //       for(i=1;i<=atomnum;i++) printf("%4d:%4d\n", i, Total_NumOrbs[i]);
       * //       printf("  MAX:%4d\n",TNO_MAX);
       * printf("ClaOrb_MAX[0]:%4d\n",ClaOrb_MAX[0]);
       * printf("ClaOrb_MAX[1]:%4d\n",ClaOrb_MAX[1]);
       * printf("Total Band (2*n):%4d\n",n2-4);
       * printf("Central (%10.6lf %10.6lf %10.6lf)\n",k_CNT1[0],k_CNT1[1],k_CNT1[2]);
       * printf("###########################################\n\n");
       */

      /* Added by N. Yamaguchi ***/
      PRINTFF("########### ORBITAL DATA ##################\n");
      PRINTFF("ClaOrb_MAX[0]:%4d\n",ClaOrb_MAX[0]);
      PRINTFF("ClaOrb_MAX[1]:%4d\n",ClaOrb_MAX[1]);
      PRINTFF("Total Band (2*n):%4d\n",n2-4);
      PRINTFF("Central (%10.6lf %10.6lf %10.6lf)\n",k_CNT1[0],k_CNT1[1],k_CNT1[2]);
      PRINTFF("###########################################\n\n");
      /* ***/

    }
    // ### MALLOC ARRAY ######################################
    // ### (MPI Calculation)    ###

    size_Tknum = (k1_3mesh+1)*(k2_3mesh+1)*(k3_height+1);

    //    size_Tknum = (k1_3mesh+1)*(k2_3mesh+1);
    S_knum = (int*)malloc(sizeof(int)*num_procs);
    E_knum = (int*)malloc(sizeof(int)*num_procs);
    S2_knum = (int*)malloc(sizeof(int)*num_procs);
    E2_knum = (int*)malloc(sizeof(int)*num_procs);

    // ### (EigenValue Problem) ###
    k_xyz = (double**)malloc(sizeof(double*)*3);
    for(i=0; i<3; i++){
      k_xyz[i] = (double*)malloc(sizeof(double)*(size_Tknum+1));
    }
    EIGEN = (double*)malloc(sizeof(double)*n2);
    for (i = 0; i < n2; i++) EIGEN[i] = 0.0;
    EIGEN_MP = (double*)malloc(sizeof(double)*((size_Tknum+1)*n2));
    for (i = 0; i < ((size_Tknum+1)*n2); i++) EIGEN_MP[i] = 0.0;

    /* Disabled by N. Yamaguchi
     * // ### (MulP Calculation)   ###
     * Data_MulP = (double****)malloc(sizeof(double***)*4);
     * for (i=0; i<4; i++){
     * Data_MulP[i] = (double***)malloc(sizeof(double**)*n2);
     * for (l=0; l<n2; l++){
     * Data_MulP[i][l] = (double**)malloc(sizeof(double*)*(atomnum+1));
     * for (j=0; j<=atomnum; j++){
     * Data_MulP[i][l][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
     * for (k=0; k<=TNO_MAX; k++) Data_MulP[i][l][j][k] = 0.0;
     * }
     * }
     * }
     * if (myrank == 0)  printf("########### EIGEN VALUE ###################\n");
     */

    /* Added by N. Yamaguchi ***/
    if (myrank == 0)  PRINTFF("########### EIGEN VALUE ###################\n");
    /* ***/

    dtime(&Stime);

    // ### k-GRID & EIGEN(intersection) ######################
    for(i=0; i<=k1_3mesh; i++){

      /* Disabled by N. Yamaguchi
       * if(k1_3mesh==1)\{ k1 = 0.0; \}else\{
       */

      /* Added by N. Yamaguchi ***/
      if(k1_3mesh==0){ k1 = -kRange_3mesh[0]; }else{
	/* ***/

	//        k1 = -kRange_3mesh[0] + 2.0*kRange_3mesh[0]*(2.0*(double)i+1.0)/(2.0*(double)k1_3mesh); \}
	k1 = -kRange_3mesh[0] + 2.0*kRange_3mesh[0]*(2.0*(double)i)/(2.0*(double)k1_3mesh); }
      for(j=0; j<=k2_3mesh; j++){

	/* Disabled by N. Yamaguchi
	 * if(k2_3mesh==1)\{ k2 = 0.0; \}else\{
	 */

	/* Added by N. Yamaguchi ***/
	if(k2_3mesh==0){ k2 = -kRange_3mesh[1]; }else{
	  /* ***/

	  //          k2 = -kRange_3mesh[1] + 2.0*kRange_3mesh[1]*(2.0*(double)j+1.0)/(2.0*(double)k2_3mesh); \}
	  k2 = -kRange_3mesh[1] + 2.0*kRange_3mesh[1]*(2.0*(double)j)/(2.0*(double)k2_3mesh); }
	for(k=0; k<=k3_height; k++){

	  /* Disabled by N. Yamaguchi
	   * if(k3_height==1)\{ k3 = 0.0; \}else\{
	   */

	  /* Added by N. Yamaguchi ***/
	  if(k3_height==0){ k3 = 0.0; }else{
	    /* ***/

	    //            k3 = -kRange_height + 2.0*kRange_height*(2.0*(double)k+1.0)/(2.0*(double)k3_height); \}
	    k3 = -kRange_height + 2.0*kRange_height*(2.0*(double)k)/(2.0*(double)k3_height); }
	  if (plane_3mesh == 1){
	    k_xyz[0][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k1 + k_CNT[0] + Shift_K_Point;
	    k_xyz[1][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k2 + k_CNT[1] - Shift_K_Point;
	    k_xyz[2][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k3 + k_CNT[2] + 2.0*Shift_K_Point;
	  }else if(plane_3mesh == 2){
	    k_xyz[0][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k3 + k_CNT[0] + Shift_K_Point;
	    k_xyz[1][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k1 + k_CNT[1] - Shift_K_Point;
	    k_xyz[2][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k2 + k_CNT[2] + 2.0*Shift_K_Point;
	  }else if(plane_3mesh == 3){
	    k_xyz[0][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k2 + k_CNT[0] + Shift_K_Point;
	    k_xyz[1][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k3 + k_CNT[1] - Shift_K_Point;
	    k_xyz[2][i*(k2_3mesh+1)+j +k*(k1_3mesh+1)*(k2_3mesh+1)] = k1 + k_CNT[2] + 2.0*Shift_K_Point;
	  }//if
	}//k
      }//j
    }//i

    /* Added by N. Yamaguchi ***/
    if (size_Tknum>1){
      /* ***/

      // ### Division CALC_PART ################################
      T_knum = size_Tknum;
      for(i=0; i<num_procs; i++){
	if (T_knum <= i){
	  S_knum[i] = -10;   E_knum[i] = -100;

	  /* Added by N. Yamaguchi ***/
#if 0
	  /* ***/

	} else if (T_knum < num_procs) {

	  /* Added by N. Yamaguchi ***/
#endif
	} else if (T_knum<=num_procs){
	  /* ***/

	  S_knum[i] = i;     E_knum[i] = i;
	} else {
	  d0 = (double)T_knum/(double)num_procs;
	  S_knum[i] = (int)((double)i*(d0+0.0001));
	  E_knum[i] = (int)((double)(i+1)*(d0+0.0001)) - 1;
	  if (i==(num_procs-1)) E_knum[i] = T_knum - 1;
	  if (E_knum[i]<0)      E_knum[i] = 0;
	}
      }
      // ### EIGEN_VALUE_PROBLEM ###############################
      for (i = S_knum[myrank]; i <= E_knum[myrank]; i++){
	EigenValue_Problem(k_xyz[0][i], k_xyz[1][i], k_xyz[2][i], 0);
	for(l=1; l<=2*n; l++){ EIGEN_MP[i*n2+l] = EIGEN[l]; }//l
      }//i
      // ### MPI part ##########################################
      for (i=0; i<num_procs; i++){

	/* Disabled by N. Yamaguchi
	 * k = S_knum[i]*n2;
	 * l = abs(E_knum[i]-S_knum[i]+1)*n2;
	 * MPI_Bcast(&EIGEN_MP[k], l, MPI_DOUBLE, i, MPI_COMM_WORLD);
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	/* Added by N. Yamaguchi ***/
	if (S_knum[i]>=0){
	  cnt=S_knum[i]*n2;
	  l=(E_knum[i]-S_knum[i]+1)*n2;
	} else {
	  cnt=0;
	  l=0;
	}
	MPI_Bcast(EIGEN_MP+cnt, l, MPI_DOUBLE, i, MPI_COMM_WORLD);
	/* ***/

      }
      dtime(&Etime);
      Time_EIG = Etime-Stime;
      // ### SELECT CALC_BAND ##################################
      l_min = 0;  l_max = 0;
      EF = (E_Range[1]+E_Range[0])/2;
      l_cal = 0;
      for(l=1; l<=2*n; l++){
	E1 = EIGEN_MP[0+l];  E2 = EIGEN_MP[0+l];
	for (i = 0; i < size_Tknum; i++){
	  if (E1 > EIGEN_MP[i*n2+l])  E1 = EIGEN_MP[i*n2+l];  //min
	  if (E2 < EIGEN_MP[i*n2+l])  E2 = EIGEN_MP[i*n2+l];  //max
	}//i
	if ((E1-EF)*(E2-EF) <=0){
	  if (l_cal == 0){
	    l_cal = 1;
	    l_min = l;  l_max = l;
	  }else if(l_cal > 0){
	    l_max = l;
	  }
	}//if
      }//l
      if (l_cal > 0) {
	l_cal = (l_max-l_min+1);

	/* Disabled by N. Yamaguchi
	 * \} else \{
	 * puts("Error: Bands are not found.");
	 */

      } else {
	/* Added by N. Yamaguchi ***/
	PRINTFF("Error: No Bands are found.\n");
#ifndef SIGMAEK
	/* ***/

	MPI_Finalize();

	/* Added by N. Yamaguchi ***/
#endif
	/* ***/

	return 0;
      }

      /* Added by N. Yamaguchi ***/
      if (calcBandbyband) {

	/*
	   if (calcBandMax<l_min || calcBandMin>l_max) {
	   PRINTFF("Error: No bands are found in the range of %d-%d.\n", calcBandMin, calcBandMax);
	   MPI_Finalize();
	   return 0;
	   \} else \{
	   */

	l_max= l_max<calcBandMax ? l_max: calcBandMax;
	l_min= l_min>calcBandMin ? l_min: calcBandMin;
	l_cal=l_max-l_min+1;

	/*
	   }
	   */
      }
      /* ***/

      /* Disabled by N. Yamaguchi
       * if (myrank==0) printf("The number of BANDs %4d (%4d->%4d)\n", l_cal, l_min, l_max);
       */

      /* Added by N. Yamaguchi ***/
      if (myrank==0) PRINTFF("The number of BANDs %4d (%4d->%4d)\n", l_cal, l_min, l_max);
      /* ***/

      /* Added by N. Yamaguchi ***/
    } else if (size_Tknum==1){
      if (calcBandbyband){
	l_max=calcBandMax;
	l_min=calcBandMin;
	EF=(E_Range[1]+E_Range[0])*0.5;
      } else {
	PRINTFF("Error: Bands to calculate should be specified if the first step is omitted.\n");
#ifndef SIGMAEK
	MPI_Finalize();
#endif
	return 0;
      }
      Time_EIG=0.0;
    }
    /* ***/

    // ### MALLOC k-GRID & EIGEN(frame) ######################

    // ### (2-step Contour )    ###
    k_index_hitD = (int**)malloc(sizeof(int*)*2);
    for(i=0; i<2; i++){
      k_index_hitD[i] = (int*)malloc(sizeof(int)*((k1_3mesh+1)*(k2_3mesh+1)));
      for(j=0; j< (k1_3mesh+1)*(k2_3mesh+1); j++) k_index_hitD[i][j] = 0;
    }
    k_xyz_domain = (double**)malloc(sizeof(double*)*3);
    for(i=0; i<3; i++){
      k_xyz_domain[i] = (double*)malloc(sizeof(double)*((k1_domain+1)*(k2_domain+1)));
    }
    EIGEN_domain = (double*)malloc(sizeof(double)*((k1_domain+1)*(k2_domain+1)));

    /* Disabled by N. Yamaguchi
     * kRange_domain[0] = 2.0*kRange_3mesh[0]/((double)k1_3mesh);
     * kRange_domain[1] = 2.0*kRange_3mesh[1]/((double)k2_3mesh);
     */

    /* Added by N. Yamaguchi ***/
    kRange_domain[0] = 2.0*kRange_3mesh[0]/((double)(k1_3mesh+1));
    kRange_domain[1] = 2.0*kRange_3mesh[1]/((double)(k2_3mesh+1));
    /* ***/

    // ### (MALLOC TIME)        ###

    /* Disabled by N. Yamaguchi
     * l = abs(l_max-l_min);
     * if (l<1) l = 1;
     */

    /* Added by N. Yamaguchi ***/
    l = l_max-l_min;
    if (l<1) l = 0;
    /* ***/

    Time_Contour = (double*)malloc(sizeof(double)*(l+1));
    Time_MulP    = (double*)malloc(sizeof(double)*(l+1));
    for (i=0; i<=l; i++){
      Time_Contour[i] = 0;
      Time_MulP[i] = 0;
    }
    // #######################################################

    if (myrank == 0){

      /* Added by N. Yamaguchi ***/
#if 0
      /* ***/

      // ### SCRIPT FILE ###  kotaka
      fp1 = fopen("SampleScript.csh","w");

      /* Modified by N. Yamaguchi ***/
      fprintf(fp1, "#!/bin/csh -f\n\n");
      /* ***/

      /* Disabled by N. Yamaguchi ***
	 fprintf(fp1, "set InpName = \"%s\"\n",argv[1]);
       * ***/

      /* Added by N. Yamaguchi ***/
      fprintf(fp1, "set InpName = \"%s\"\n", fname);
      /* ***/

      fprintf(fp1, "set ExeFile = \"Calc_MulP3\"\n");
      fprintf(fp1, "make $ExeFile\n\n");
      fprintf(fp1, "set n=%d\n",l_min);
      fprintf(fp1, "while ( $n <= %d )\n",l_max);
      for(i=1;i<=atomnum;i++){
	fprintf(fp1, "sed -e \"s/INP_MULP_NAME/%s.AtomMulP/g\" $InpName > STRAGE1.inp\n",fname_out);
	fprintf(fp1, "sed -e \"s/OUT_MULP_NAME/%s_%d%s_$n/g\" STRAGE1.inp > STRAGE2.inp\n",fname_out,i,An2Spe[i]);
	fprintf(fp1, "sed -e \"s/ATOM_NUM/%d/g\" STRAGE2.inp > STRAGE1.inp\n",i);
	fprintf(fp1, "sed -e \"s/BAND_NUM/$n/g\" STRAGE1.inp > %2s_%d_$n.inp\n",An2Spe[i],i);
	//      fprintf(fp1, "rm STRAGE1.inp\n");
	//      fprintf(fp1, "rm STRAGE2.inp\n");
	fprintf(fp1, "./$ExeFile %2s_%d_$n.inp > %2s_%d_$n.std_MulP\n",An2Spe[i],i,An2Spe[i],i);
	for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  fprintf(fp1, "sed -e \"s/INP_MULP_NAME/%s.MulP_%s/g\" $InpName > STRAGE1.inp\n",fname_out,OrbSym[i1]);
	  fprintf(fp1, "sed -e \"s/OUT_MULP_NAME/%s_%d%s_%s_$n/g\" STRAGE1.inp > STRAGE2.inp\n",fname_out,i,An2Spe[i],OrbSym[i1]);
	  fprintf(fp1, "sed -e \"s/ATOM_NUM/%d/g\" STRAGE2.inp > STRAGE1.inp\n",i);
	  fprintf(fp1, "sed -e \"s/BAND_NUM/$n/g\" STRAGE1.inp > %2s_%d_$n.inp\n",An2Spe[i],i);
	  fprintf(fp1, "./$ExeFile %2s_%d_$n.inp >> %2s_%d_$n.std_MulP\n",An2Spe[i],i,An2Spe[i],i);
	}//i1
	for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	  fprintf(fp1, "sed -e \"s/INP_MULP_NAME/%s.MulP_%s/g\" $InpName > STRAGE1.inp\n",fname_out,OrbName[i1]);
	  fprintf(fp1, "sed -e \"s/OUT_MULP_NAME/%s_%d%s_%s_$n/g\" STRAGE1.inp > STRAGE2.inp\n",fname_out,i,An2Spe[i],OrbName[i1]);
	  fprintf(fp1, "sed -e \"s/ATOM_NUM/%d/g\" STRAGE2.inp > STRAGE1.inp\n",i);
	  fprintf(fp1, "sed -e \"s/BAND_NUM/$n/g\" STRAGE1.inp > %2s_%d_$n.inp\n",An2Spe[i],i);
	  fprintf(fp1, "./$ExeFile %2s_%d_$n.inp >> %2s_%d_$n.std_MulP\n",An2Spe[i],i,An2Spe[i],i);
	}//i1
      }//i
      fprintf(fp1, "echo $n\n");
      fprintf(fp1, "@ n = $n + 1\n");
      fprintf(fp1, "end\n\n");
      fprintf(fp1, "rm STRAGE1.inp\n");
      fprintf(fp1, "rm STRAGE2.inp\n");
      fclose(fp1);

      /* Added by N. Yamaguchi ***/
#endif
      char fname_gp[256];
      strcpy(fname_gp,fname_out);
      strcat(fname_gp,".plotexample");
      fp1=fopen(fname_gp, "w");
      fputs("Bohr2Ang=1./0.529177249\n", fp1);
      if (plane_3mesh==1){
	fprintf(fp1, "xc=%f*Bohr2Ang\n", k_CNT1[0]);
	fprintf(fp1, "yc=%f*Bohr2Ang\n", k_CNT1[1]);
      } else if (plane_3mesh==2){
	fprintf(fp1, "xc=%f*Bohr2Ang\n", k_CNT1[1]);
	fprintf(fp1, "yc=%f*Bohr2Ang\n", k_CNT1[2]);
      } else if (plane_3mesh==3){
	fprintf(fp1, "xc=%f*Bohr2Ang\n", k_CNT1[2]);
	fprintf(fp1, "yc=%f*Bohr2Ang\n", k_CNT1[0]);
      }
      fputs("tics=0.2\n", fp1);
      fputs("#reduction=1\n", fp1);
      fputs("reduction=6\n", fp1);
      fputs("linewidth=3\n", fp1);
      fputs("set size ratio -1\n", fp1);
      fputs("set encoding iso\n", fp1);
      fputs("set xlabel 'k_x (/\\305)'\n", fp1);
      fputs("set ylabel 'k_y (/\\305)'\n", fp1);
      fputs("set xtics tics\n", fp1);
      fputs("set ytics tics\n", fp1);
      fputs("set label 1 center at first xc, yc '+' front\n", fp1);
      fputs("#set label 2 center at first xc, yc-0.1 '{/Symbol G}' front\n", fp1);
      for (l=l_min; l<=l_max; l++){
	if (l>l_min){
	  fputs("re", fp1);
	}
	if (plane_3mesh==1){
	  fprintf(fp1, "plot '%s.FermiSurf_%d' using ($1*Bohr2Ang):($2*Bohr2Ang) with lines notitle linewidth linewidth\n", fname_out, l);
	  fprintf(fp1, "replot '%s.Pxyz_%d' every reduction using ($1*Bohr2Ang):($2*Bohr2Ang):4:5 with vectors notitle linewidth linewidth\n", fname_out, l);
	} else if (plane_3mesh==2){
	  fprintf(fp1, "plot '%s.FermiSurf_%d' using ($2*Bohr2Ang):($3*Bohr2Ang) with lines notitle linewidth linewidth\n", fname_out, l);
	  fprintf(fp1, "replot '%s.Pxyz_%d' every reduction using ($2*Bohr2Ang):($3*Bohr2Ang):5:6 with vectors notitle linewidth linewidth\n", fname_out, l);
	} else if (plane_3mesh==3){
	  fprintf(fp1, "plot '%s.FermiSurf_%d' using ($3*Bohr2Ang):($1*Bohr2Ang) with lines notitle linewidth linewidth\n", fname_out, l);
	  fprintf(fp1, "replot '%s.Pxyz_%d' every reduction using ($3*Bohr2Ang):($1*Bohr2Ang):6:4 with vectors notitle linewidth linewidth\n", fname_out, l);
	}
      }
      fputs("pause -1\n", fp1);
      fclose(fp1);
      /* ***/

      /* Disabled by N. Yamaguchi
       * printf("########### CONTOUR CALC ##################\n");
       */

      /* Added by N. Yamaguchi ***/
      PRINTFF("########### CONTOUR CALC ##################\n");
      /* ***/

      for(l=l_min; l<=l_max; l++){
	strcpy(fname_FS,fname_out);
	name_Nband(fname_FS,".FermiSurf_",l);
	fp1 = fopen(fname_FS,"w");
	fclose(fp1);
      }//l

      for (j2=0; j2<=Spin_Dege; j2++){
	strcpy(fname_MP,fname_out);    strcat(fname_MP,".AtomMulP");
	if (j2 == 1) strcat(fname_MP,"_Dege");
	fp1= fopen(fname_MP,"w");
	fprintf(fp1,"                                      \n");
	fclose(fp1);
	for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  strcpy(fname_MP,fname_out);     strcat(fname_MP,".MulP_");      strcat(fname_MP,OrbSym[i1]);
	  if (j2 == 1) strcat(fname_MP,"_Dege");
	  fp1= fopen(fname_MP,"w");
	  fprintf(fp1,"                                    \n");
	  fclose(fp1);
	}//i1
	for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	  strcpy(fname_MP,fname_out);     strcat(fname_MP,".MulP_");     strcat(fname_MP,OrbName[i1]);
	  if (j2 == 1) strcat(fname_MP,"_Dege");
	  fp1= fopen(fname_MP,"w");
	  fprintf(fp1,"                                    \n");
	  fclose(fp1);
	}//i1
      }//j2

      for(l=l_min; l<=l_max; l++){
	strcpy(fname_Spin,fname_out);
	name_Nband(fname_Spin,".Pxyz_",l);
	fp1 = fopen(fname_Spin,"w");
	fclose(fp1);
      }//l
    }//if(myrank)

    hit_Total[0] = 0;
    hit_Total[1] = 0;

    /* Disabled by N. Yamaguchi
     * for(i_height = 0; i_height< k3_height; i_height++)\{
     */

    /* Added by N. Yamaguchi ***/
    for(i_height = 0; i_height<= k3_height; i_height++){
      /* ***/

      /* Disabled by N. Yamaguchi
       * if (k3_height==1){ k3 = 0.0; }else{
       * k3 = -kRange_height + 2.0*kRange_height*(2.0*(double)i_height+1.0)/(2.0*(double)k3_height); }
       * if (myrank == 0) printf("  k-height :%4d %10.6lf\n", i_height, k3);
       */

      /* Added by N. Yamaguchi ***/
      if (k3_height==0){
	k3 = 0.0;
      }else{
	k3 = -kRange_height + 2.0*kRange_height*(2.0*(double)i_height)/(2.0*(double)k3_height);
      }
      if (myrank == 0) PRINTFF("  k-height :%4d %10.6lf\n", i_height, k3);
      /* ***/

      /* Disabled by N. Yamaguchi
       * l_min = 0;  l_max = 0;
       * l_cal = 0;
       * j =i_height*(k1_3mesh+1)*(k2_3mesh+1);
       * for(l=1; l<=2*n; l++){
       * E1 = EIGEN_MP[(j+ 0)*n2+l];  E2 = EIGEN_MP[(j+ 0)*n2+l];
       * for (i = 0; i < ((k1_3mesh+1)*(k2_3mesh+1)); i++){
       * if (E1 > EIGEN_MP[(j +i)*n2+l])  E1 = EIGEN_MP[(j+ i)*n2+l];  //min
       * if (E2 < EIGEN_MP[(j +i)*n2+l])  E2 = EIGEN_MP[(j+ i)*n2+l];  //max
       * }//i
       * if ((E1-EF)*(E2-EF) <=0){
       * if (l_cal == 0){
       * l_cal = 1;
       * l_min = l;  l_max = l;
       * }else if(l_cal > 0){
       * l_max = l;
       * }
       * }//if
       * }//l
       * if (l_cal > 0) l_cal = (l_max-l_min+1);
       */

      /* Added by N. Yamaguchi ***/
      if (calcBandbyband) {
	if (calcBandMax<l_min || calcBandMin>l_max) {
	  PRINTFF("Error: No bands are found in the range of %d-%d.\n", calcBandMin, calcBandMax);
#ifndef SIGMAEK
	  MPI_Finalize();
#endif
	  return 0;
	} else {
	  l_max=l_max<calcBandMax ? l_max: calcBandMax;
	  l_min=l_min>calcBandMin ? l_min: calcBandMin;
	  l_cal=l_max-l_min+1;
	}
      } else {
	l_min = 0;  l_max = 0;
	l_cal = 0;
	j =i_height*(k1_3mesh+1)*(k2_3mesh+1);
	for(l=1; l<=2*n; l++){
	  E1 = EIGEN_MP[(j+ 0)*n2+l];  E2 = EIGEN_MP[(j+ 0)*n2+l];
	  for (i = 0; i < ((k1_3mesh+1)*(k2_3mesh+1)); i++){
	    if (E1 > EIGEN_MP[(j +i)*n2+l])  E1 = EIGEN_MP[(j+ i)*n2+l];  //min
	    if (E2 < EIGEN_MP[(j +i)*n2+l])  E2 = EIGEN_MP[(j+ i)*n2+l];  //max
	  }//i
	  if ((E1-EF)*(E2-EF) <=0){
	    if (l_cal == 0){
	      l_cal = 1;
	      l_min = l;  l_max = l;
	    }else if(l_cal > 0){
	      l_max = l;
	    }
	  }//if
	}//l
	if (l_cal > 0) l_cal = (l_max-l_min+1);
      }
      /* ***/

      /* Disabled by N. Yamaguchi
       * if (myrank==0) printf("The number of BANDs %4d (%4d->%4d)\n", l_cal, l_min, l_max);
       */

      /* Added by N. Yamaguchi ***/
      if (myrank==0) PRINTFF("The number of BANDs %4d (%4d->%4d)\n", l_cal, l_min, l_max);
      /* ***/

      /* Added by N. Yamaguchi ***/
      // ### (MulP Calculation)   ###
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
      Data_MulP = (double****)malloc(sizeof(double***)*4);
      for (i=0; i<4; i++){
#ifdef DEBUG_SIGMAEK_OLDER_20181126
	Data_MulP[i] = (double***)malloc(sizeof(double**)*l_cal);
	for (l=0; l<l_cal; l++){
	  Data_MulP[i][l] = (double**)malloc(sizeof(double*)*(atomnum+1));
	  for (j=0; j<=atomnum; j++){
	    Data_MulP[i][l][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
	  }
	}
#else
	Data_MulP[i] = (double***)malloc(sizeof(double**)*1);
	Data_MulP[i][0] = (double**)malloc(sizeof(double*)*(atomnum+1));
	for (j=0; j<=atomnum; j++){
	  Data_MulP[i][0][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
	}
      }
#endif
#endif
      /* ***/

      //      for(l=l_min; (l<=l_max && l_cal>0); l++)\{
      for(l=l_min; l<=l_max; l++){

	dtime(&Stime);

	// #####################################################

	count_hitD = 0;
	k =i_height*(k1_3mesh+1)*(k2_3mesh+1);

	/* Disabled by N. Yamaguchi
	 * for(i=0; i< k1_3mesh; i++){
	 * for(j=0; j< k2_3mesh; j++){
	 * d0 = EIGEN_MP[(i*(k2_3mesh+1)+j +k)*n2+l]-EF;
	 * d1 = EIGEN_MP[((i+1)*(k2_3mesh+1)+j +k)*n2+l]-EF;
	 * d2 = EIGEN_MP[(i*(k2_3mesh+1)+(j+1) +k)*n2+l]-EF;
	 * d3 = EIGEN_MP[((i+1)*(k2_3mesh+1)+(j+1) +k)*n2+l]-EF;
	 * if ((d0*d1<0) || (d0*d2<0) || (d1*d3<0) || (d2*d3<0)){
	 * k_index_hitD[0][count_hitD] = i;
	 * k_index_hitD[1][count_hitD] = j;
	 * count_hitD++;
	 * }//if
	 * }//j
	 * }//i
	 */

	/* Added by N. Yamaguchi ***/
	if (size_Tknum==1){
	  k_index_hitD[0][count_hitD] = 0;
	  k_index_hitD[1][count_hitD] = 0;
	  count_hitD=1;
	} else {
	  for(i=0; i< k1_3mesh; i++){
	    for(j=0; j< k2_3mesh; j++){
	      d0 = EIGEN_MP[(i*(k2_3mesh+1)+j +k)*n2+l]-EF;
	      d1 = EIGEN_MP[((i+1)*(k2_3mesh+1)+j +k)*n2+l]-EF;
	      d2 = EIGEN_MP[(i*(k2_3mesh+1)+(j+1) +k)*n2+l]-EF;
	      d3 = EIGEN_MP[((i+1)*(k2_3mesh+1)+(j+1) +k)*n2+l]-EF;
	      if ((d0*d1<0) || (d0*d2<0) || (d1*d3<0) || (d2*d3<0)){
		k_index_hitD[0][count_hitD] = i;
		k_index_hitD[1][count_hitD] = j;
		count_hitD++;
	      }//if
	    }//j
	  }//i
	}
	//        if (myrank == 0) printf("count_hitD:%4d\n",count_hitD);
	// ### MALLOC ##########################################
	frame_hit3M = (int***)malloc(sizeof(int**)*(4));
	for(i=0; i<4; i++){
	  frame_hit3M[i] = (int**)malloc(sizeof(int*)*(count_hitD+1));
	  for(j=0; j<=count_hitD; j++){
	    k = (k1_domain+1)*(k2_domain+1);
	    frame_hit3M[i][j] = (int*)malloc(sizeof(int)*(k+1));
	    for(i2=0; i2<=k; i2++)  frame_hit3M[i][j][i2] = 0;
	  }//j
	}//i
	k_xyz_frame3M = (double****)malloc(sizeof(double***)*(4));
	for(i=0; i<4; i++){
	  k_xyz_frame3M[i] = (double***)malloc(sizeof(double**)*(3));
	  for(j=0; j<3; j++){
	    k_xyz_frame3M[i][j] = (double**)malloc(sizeof(double*)*(count_hitD+1));
	    for(i2=0; i2<=count_hitD; i2++){
	      k = (k1_domain+1)*(k2_domain+1);
	      k_xyz_frame3M[i][j][i2] = (double*)malloc(sizeof(double)*(k+1));
	      for(j2=0; j2<=k; j2++) k_xyz_frame3M[i][j][i2][j2] = 0.0;
	    }//i2
	  }//j
	}//i
	EIG_frame3M = (double***)malloc(sizeof(double**)*(4));
	for(i=0; i<4; i++){
	  EIG_frame3M[i] = (double**)malloc(sizeof(double*)*(count_hitD+1));
	  for(j=0; j<=count_hitD; j++){
	    k = (k1_domain+1)*(k2_domain+1);
	    EIG_frame3M[i][j] = (double*)malloc(sizeof(double)*(k+1));
	    for(i2=0; i2<=k; i2++) EIG_frame3M[i][j][i2] = 0.0;
	  }//j
	}//i
	count_hit3M = (int*)malloc(sizeof(int**)*(count_hitD+1));
	for(i=0; i<=count_hitD; i++) count_hit3M[i] = 0;
	count_T_hit3M = 0;
	for(i_hitD=0; i_hitD< count_hitD; i_hitD++){
	  // ###################################################
	  for(i=0; i<=k1_domain; i++){

	    /* Disabled by N. Yamaguchi
	     * if(k1_domain==1)\{
	     */

	    /* Added by N. Yamaguchi ***/
	    if(k1_domain==0){
	      /* ***/
	      k1 = 0;
	    }else{
	      //              k1 = kRange_domain[0]*(2.0*(double)i+1.0)/(2.0*(double)k1_domain);
	      k1 = kRange_domain[0]*((double)i)/((double)k1_domain);
	    }
	    for(j=0; j<=k2_domain; j++){

	      /* Disabled by N. Yamaguchi
	       * if(k2_domain==1)\{
	       */

	      /* Added by N. Yamaguchi ***/
	      if(k2_domain==0){
		k2 = 0;
		/* ***/
	      }else{
		//                k2 = kRange_domain[1]*(2.0*(double)j+1.0)/(2.0*(double)k2_domain);
		k2 = kRange_domain[1]*((double)j)/((double)k2_domain); }
	      i2 = k_index_hitD[0][i_hitD]*(k2_3mesh+1) + k_index_hitD[1][i_hitD];
	      if (plane_3mesh == 1){
		k_xyz_domain[0][i*(k2_domain+1)+j] = k1 + k_xyz[0][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] + Shift_K_Point;      //k_CNT[0]
		k_xyz_domain[1][i*(k2_domain+1)+j] = k2 + k_xyz[1][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] - Shift_K_Point;      //k_CNT[1]
		k_xyz_domain[2][i*(k2_domain+1)+j] =      k_xyz[2][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] + 2.0*Shift_K_Point;  //k_CNT[2]
	      }else if(plane_3mesh == 2){
		k_xyz_domain[0][i*(k2_domain+1)+j] =      k_xyz[0][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] + Shift_K_Point;      //k_CNT[0]
		k_xyz_domain[1][i*(k2_domain+1)+j] = k1 + k_xyz[1][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] - Shift_K_Point;      //k_CNT[1]
		k_xyz_domain[2][i*(k2_domain+1)+j] = k2 + k_xyz[2][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] + 2.0*Shift_K_Point;  //k_CNT[2]
	      }else if(plane_3mesh == 3){
		k_xyz_domain[0][i*(k2_domain+1)+j] = k2 + k_xyz[0][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] + Shift_K_Point;      //k_CNT[0]
		k_xyz_domain[1][i*(k2_domain+1)+j] =      k_xyz[1][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] - Shift_K_Point;      //k_CNT[1]
		k_xyz_domain[2][i*(k2_domain+1)+j] = k1 + k_xyz[2][i2 +i_height*(k1_3mesh+1)*(k2_3mesh+1)] + 2.0*Shift_K_Point;  //k_CNT[2]
	      }//if
	    }//j
	  }//i
	  // ### Division CALC_PART ############################
	  T_knum = (k1_domain+1)*(k2_domain+1);
	  for(i=0; i<num_procs; i++){
	    if (T_knum <= i){
	      S_knum[i] = -10;   E_knum[i] = -100;

	      /* Added by N. Yamaguchi ***/
#if 0
	      /* ***/

	    } else if (T_knum < num_procs) {

	      /* Added by N. Yamaguchi ***/
#endif
	    } else if (T_knum<=num_procs){
	      /* ***/


	      S_knum[i] = i;     E_knum[i] = i;
	    } else {
	      d0 = (double)T_knum/(double)num_procs;
	      S_knum[i] = (int)((double)i*(d0+0.0001));
	      E_knum[i] = (int)((double)(i+1)*(d0+0.0001)) - 1;
	      if (i==(num_procs-1)) E_knum[i] = T_knum - 1;
	      if (E_knum[i]<0)      E_knum[i] = 0;
	    }
	  }//i
	  // ### EIGEN_VALUE_PROBLEM ###########################
	  for (i = S_knum[myrank]; i <= E_knum[myrank]; i++){
	    EigenValue_Problem(k_xyz_domain[0][i], k_xyz_domain[1][i], k_xyz_domain[2][i], 0);
	    EIGEN_domain[i] = EIGEN[l];
	  }//i
	  // MPI part

	  for (i=0; i<num_procs; i++){

	    /* Disabled by N. Yamaguchi
	     * k = S_knum[i];
	     * m = abs(E_knum[i]-S_knum[i]+1);
	     * MPI_Barrier(MPI_COMM_WORLD);
	     * MPI_Bcast(&EIGEN_domain[k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     */

	    /* Added by N. Yamaguchi ***/
	    if (S_knum[i]>=0){
	      cnt=S_knum[i];
	      m=E_knum[i]-S_knum[i]+1;
	    } else {
	      cnt=0;
	      m=0;
	    }

	    MPI_Bcast(EIGEN_domain+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    /* ***/

	  }//i

	  // ###################################################
	  /*
	     if (myrank == 0){
	     for(i=0; i<=k1_domain; i++){
	     for(j=0; j<=k2_domain; j++){
	     for(i2=0; i2<3; i2++){
	     printf("%10.6lf ",rtv[1][i2+1]*k_xyz_domain[0][i*(k2_domain+1)+j] + rtv[2][i2+1]*k_xyz_domain[1][i*(k2_domain+1)+j] + rtv[3][i2+1]*k_xyz_domain[2][i*(k2_domain+1)+j]);
	     } printf("%4d %10.6lf\n",l,EIGEN_domain[i*(k2_domain+1)+j]);
	     }//j
	     }//i
	     }//if
	     */
	  // ###################################################

	  /* Disabled by N. Yamaguchi ***
	  // ### frame2 ((k1_mesh)-direction) ##################
	   * ***/

	  /* Added by N. Yamaguchi ***
	  // ### frame2 ((k2_mesh)-direction) ##################
	   * ***/

	  for(i=0; i<=k1_domain; i++){
	    for(j=0; j<=k2_domain; j++) frame_hit3M[2][i_hitD][i*(k2_domain+1)+j] = 0;
	  }
	  for (m = S_knum[myrank]; m <= E_knum[myrank]; m++){
	    i = m/(k2_domain+1);  // for(i=0; i<=k1_domain; i++)
	    j = m%(k2_domain+1);  // for(j=0; j< k2_domain; j++)
	    if (j<k2_domain){

	      /* Added by N. Yamaguchi ***/
#if 0
	      * ***/

		if ((EIGEN_domain[i*(k2_domain+1)+j]-EF)*(EIGEN_domain[i*(k2_domain+1)+j+1]-EF) <=0){

		  /* Added by N. Yamaguchi ***/
#endif
		  if (EIGEN_domain[i*(k2_domain+1)+j]<=EF && EIGEN_domain[i*(k2_domain+1)+j+1]>=EF || EIGEN_domain[i*(k2_domain+1)+j]>=EF && EIGEN_domain[i*(k2_domain+1)+j+1]<=EF){
		    /* ***/

		    for(k=0; k<3; k++){
		      data1[k] = k_xyz_domain[k][i*(k2_domain+1)+j];
		      data2[k] = k_xyz_domain[k][i*(k2_domain+1)+j+1];
		    }//k

		    /* Diabled by N. Yamaguchi ***
		       func_Newton(data1, EIGEN_domain[i*(k2_domain+1)+j], data2, EIGEN_domain[i*(k2_domain+1)+j+1], data3, &E1,  EF, l, trial_Newton);
		     * ***/

		    /* Added by N. Yamaguchi ***/
		    func_Brent(data1, EIGEN_domain[i*(k2_domain+1)+j], data2, EIGEN_domain[i*(k2_domain+1)+j+1], data3, &E1,  EF, l, trial_Newton);
		    /* ***/

		    frame_hit3M[2][i_hitD][i*(k2_domain+1)+j] = 1;
		    for(k=0; k<3; k++){
		      k_xyz_frame3M[2][k][i_hitD][i*(k2_domain+1)+j] = data3[k];
		    }EIG_frame3M[2][i_hitD][i*(k2_domain+1)+j] = E1;

		    /* Added by N. Yamaguchi ***/
		  }
#if 0
		  /* ***/

		}//if

	      /* Added by N. Yamaguchi ***/
#endif
	      /* ***/

	    }//if
	  }//m
	  // ###################################################

	  /* Disabled by N. Yamaguchi ***
	  // ### frame1 ((k2_mesh)-direction) ##################
	   * ***/

	  /* Added by N. Yamaguchi ***/
	  // ### frame1 ((k1_mesh)-direction) ##################
	  /* ***/

	  for(i=0; i<=k1_domain; i++){
	    for(j=0; j<=k2_domain; j++) frame_hit3M[1][i_hitD][i*(k2_domain+1)+j] = 0;
	  }
	  for (m = S_knum[myrank]; m <= E_knum[myrank]; m++){
	    i = m/(k2_domain+1);  // for(i=0; i< k1_domain; i++)
	    j = m%(k2_domain+1);  // for(j=0; j<=k2_domain; j++)
	    if (i<k1_domain){

	      /* Added by N. Yamaguchi ***/
#if 0
	      * ***/

		if ((EIGEN_domain[i*(k2_domain+1)+j]-EF)*(EIGEN_domain[(i+1)*(k2_domain+1)+j]-EF) <=0){

		  /* Added by N. Yamaguchi ***/
#endif
		  if (EIGEN_domain[i*(k2_domain+1)+j]<=EF && EIGEN_domain[(i+1)*(k2_domain+1)+j]>=EF || EIGEN_domain[i*(k2_domain+1)+j]>=EF && EIGEN_domain[(i+1)*(k2_domain+1)+j]<=EF){
		    /* ***/

		    for(k=0; k<3; k++){
		      data1[k] = k_xyz_domain[k][i*(k2_domain+1)+j];
		      data2[k] = k_xyz_domain[k][(i+1)*(k2_domain+1)+j];
		    }

		    /* Disabled by N. Yamaguchi ***
		       func_Newton(data1, EIGEN_domain[i*(k2_domain+1)+j], data2, EIGEN_domain[(i+1)*(k2_domain+1)+j], data3, &E1,  EF, l, trial_Newton);
		     * ***/

		    /* Added by N. Yamaguchi ***/
		    func_Brent(data1, EIGEN_domain[i*(k2_domain+1)+j], data2, EIGEN_domain[(i+1)*(k2_domain+1)+j], data3, &E1,  EF, l, trial_Newton);
		    /* ***/

		    frame_hit3M[1][i_hitD][i*(k2_domain+1)+j] = 1;
		    for(k=0; k<3; k++){
		      k_xyz_frame3M[1][k][i_hitD][i*(k2_domain+1)+j] = data3[k];
		    }EIG_frame3M[1][i_hitD][i*(k2_domain+1)+j] = E1;

		    /* Added by N. Yamaguchi ***/
		  }
#if 0
		  /* ***/

		}//if

	      /* Added by N. Yamaguchi ***/
#endif
	      /* ***/

	    }//if
	  }//m
	  // #####################################################
	  // ### frame3 ((NANAME)-direction) #####################
	  for(i=0; i<=k1_domain; i++){
	    for(j=0; j<=k2_domain; j++) frame_hit3M[3][i_hitD][i*(k2_domain+1)+j] = 0;
	  }
	  for (m = S_knum[myrank]; m <= E_knum[myrank]; m++){
	    i = m/(k2_domain+1);  // for(i=0; i< k1_domain; i++)
	    j = m%(k2_domain+1);  // for(j=0; j< k2_domain; j++)
	    if ((i<k1_domain) && (j<k2_domain)){

	      /* Added by N. Yamaguchi ***/
	      if (caseAcute){
		/* ***/

		i1 = i+1; j1 = j;   // i1 = i;   j1 = j;
		i2 = i;   j2 = j+1; // i2 = i+1; j2 = j+1;

		/* Added by N. Yamaguchi ***/
	      } else {
		i1=i; j1=j;
		i2=i+1; j2=j+1;
	      }
	      /* ***/

	      /* Added by N. Yamaguchi ***/
#if 0
	      * ***/

		if ((EIGEN_domain[i1*(k2_domain+1)+j1]-EF)*(EIGEN_domain[i2*(k2_domain+1)+j2]-EF) <=0){

		  /* Added by N. Yamaguchi ***/
#endif
		  if (EIGEN_domain[i1*(k2_domain+1)+j1]<=EF && EIGEN_domain[i2*(k2_domain+1)+j2]>=EF || EIGEN_domain[i1*(k2_domain+1)+j1]>=EF && EIGEN_domain[i2*(k2_domain+1)+j2]<=EF){
		    /* ***/

		    for(k=0; k<3; k++){
		      data1[k] = k_xyz_domain[k][i1*(k2_domain+1)+j1];
		      data2[k] = k_xyz_domain[k][i2*(k2_domain+1)+j2];
		    }

		    /* Disabled by N. Yamaguchi ***
		       func_Newton(data1, EIGEN_domain[i1*(k2_domain+1)+j1], data2, EIGEN_domain[i2*(k2_domain+1)+j2], data3, &E1,  EF, l, trial_Newton);
		     * ***/

		    /* Added by N. Yamaguchi ***/
		    func_Brent(data1, EIGEN_domain[i1*(k2_domain+1)+j1], data2, EIGEN_domain[i2*(k2_domain+1)+j2], data3, &E1,  EF, l, trial_Newton);
		    /* ***/

		    frame_hit3M[3][i_hitD][i*(k2_domain+1)+j] = 1;
		    for(k=0; k<3; k++){
		      k_xyz_frame3M[3][k][i_hitD][i*(k2_domain+1)+j] = data3[k];
		    }EIG_frame3M[3][i_hitD][i*(k2_domain+1)+j] = E1;

		    /* Added by N. Yamaguchi ***/
		  }
#if 0
		  /* ***/

		}//if

	      /* Added by N. Yamaguchi ***/
#endif
	      /* ***/

	    }//if
	  }//m
	  // #####################################################
	  // ### MPI part ########################################
	  for (i=0; i<num_procs; i++){

	    /* Disabled by N. Yamaguchi
	     * k = S_knum[i];
	     * m = abs(E_knum[i]-S_knum[i]+1);
	     * MPI_Barrier(MPI_COMM_WORLD);
	     * MPI_Bcast(&EIG_frame3M[1][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&EIG_frame3M[2][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&EIG_frame3M[3][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[2][0][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[2][1][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[2][2][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&frame_hit3M[2][i_hitD][k], m, MPI_INT, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[1][0][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[1][1][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[1][2][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&frame_hit3M[1][i_hitD][k], m, MPI_INT, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[3][0][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[3][1][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&k_xyz_frame3M[3][2][i_hitD][k], m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	     * MPI_Bcast(&frame_hit3M[3][i_hitD][k], m, MPI_INT, i, MPI_COMM_WORLD);
	     */

	    /* Added by N. Yamaguchi ***/
	    if (S_knum[i]>=0){
	      cnt=S_knum[i];
	      m=E_knum[i]-S_knum[i]+1;
	    } else {
	      cnt=0;
	      m=0;
	    }

	    MPI_Bcast(EIG_frame3M[1][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(EIG_frame3M[2][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(EIG_frame3M[3][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[2][0][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[2][1][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[2][2][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(frame_hit3M[2][i_hitD]+cnt, m, MPI_INT, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[1][0][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[1][1][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[1][2][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(frame_hit3M[1][i_hitD]+cnt, m, MPI_INT, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[3][0][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[3][1][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(k_xyz_frame3M[3][2][i_hitD]+cnt, m, MPI_DOUBLE, i, MPI_COMM_WORLD);
	    MPI_Bcast(frame_hit3M[3][i_hitD]+cnt, m, MPI_INT, i, MPI_COMM_WORLD);
	    /* ***/

	  }
	  // #####################################################
	  // ### Data Count ######################################
	  count_hit3M[i_hitD] = 0;
	  for(j=0; j<=k2_domain; j++){
	    for(i=0; i<k1_domain; i++){
	      if (frame_hit3M[1][i_hitD][i*(k2_domain+1)+j] == 1){
		count_hit3M[i_hitD]++;
		count_T_hit3M++;
	      }//if
	    }//i
	  }//j
	  for(i=0; i<=k1_domain; i++){
	    for(j=0; j<k2_domain; j++){
	      if (frame_hit3M[2][i_hitD][i*(k2_domain+1)+j] == 1){
		count_hit3M[i_hitD]++;
		count_T_hit3M++;
	      }//if
	    }//j
	  }//i
	  for(i=0; i<k1_domain; i++){
	    for(j=0; j<k2_domain; j++){
	      if (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j] == 1){
		count_hit3M[i_hitD]++;
		count_T_hit3M++;
	      }//if
	    }//j
	  }//i
	  //          if (myrank == 0) printf("Hit-Count Domain-Calculation (%4d): %4d\n" ,i_hitD+1 ,count_hit3M[i_hitD]);
	  // #####################################################
	}//i_hitD
	//
	//        if (myrank == 0) printf("Hit-Count Domain-Calculation (Total): %4d\n",count_T_hit3M);

	pair_hit3M = (int**)malloc(sizeof(int*)*(count_T_hit3M+1));
	for(i=0; i<=count_T_hit3M; i++){
	  pair_hit3M[i] = (int*)malloc(sizeof(int)*(8));
	  for(j=0; j<6; j++){ pair_hit3M[i][j] = 0;}
	}//i
	// #####################################################
	// pair_hit3M is indicator of the data
	// pair_hit3M[i][0] = frame number of 1st Data
	// pair_hit3M[i][1] = count_hitD number of 1st Data
	// pair_hit3M[i][2] = k1_domain direction of 1st Data
	// pair_hit3M[i][3] = k2_domain direction of 1st Data
	// pair_hit3M[i][4] = frame number of 2nd Data
	// pair_hit3M[i][5] = count_hitD number of 2nd Data
	// pair_hit3M[i][6] = k1_domain direction of 2nd Data
	// pair_hit3M[i][7] = k2_domain direction of 2nd Data
	// ### MAKE pair_hit3M ###################################
	count_Pair3M = 0;
	for(i_hitD=0; i_hitD< count_hitD; i_hitD++){
	  for(i=0; i<k1_domain; i++){
	    for(j=0; j<k2_domain; j++){
	      i1 = i+1; j1 = j;   // i1 = i;   j1 = j;
	      i2 = i;   j2 = j+1; // i2 = i+1; j2 = j+1;

	      /* Added by N. Yamaguchi ***/
	      if (caseAcute){
		/* ***/

		if ((frame_hit3M[1][i_hitD][i*(k2_domain+1)+j1]== 1) &&
		    (frame_hit3M[2][i_hitD][i2*(k2_domain+1)+j]== 1) &&
		    (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]== 0)){    //1&2
		  pair_hit3M[count_Pair3M][0] = 1;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;  pair_hit3M[count_Pair3M][3] = j1;
		  pair_hit3M[count_Pair3M][4] = 2;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i2;  pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		}else if((frame_hit3M[1][i_hitD][i*(k2_domain+1)+j1]== 0) &&
		    (frame_hit3M[2][i_hitD][i2*(k2_domain+1)+j]== 1) &&
		    (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]== 1)){    //2&3
		  pair_hit3M[count_Pair3M][0] = 2;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i2;  pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 3;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		}else if((frame_hit3M[1][i_hitD][i*(k2_domain+1)+j1]== 1) &&
		    (frame_hit3M[2][i_hitD][i2*(k2_domain+1)+j]== 0) &&
		    (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]== 1)){    //3&1
		  pair_hit3M[count_Pair3M][0] = 3;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;   pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 1;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j1;
		  count_Pair3M++;
		}else {}

		if((frame_hit3M[1][i_hitD][i*(k2_domain+1)+j2]== 1) && 
		    (frame_hit3M[2][i_hitD][i1*(k2_domain+1)+j]== 1) && 
		    (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]== 0)){    //1&2
		  pair_hit3M[count_Pair3M][0] = 1;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;   pair_hit3M[count_Pair3M][3] = j2;
		  pair_hit3M[count_Pair3M][4] = 2;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i1;  pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		}else if((frame_hit3M[1][i_hitD][i*(k2_domain+1)+j2]== 0) &&
		    (frame_hit3M[2][i_hitD][i1*(k2_domain+1)+j]== 1) &&
		    (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]== 1)){    //2&3
		  pair_hit3M[count_Pair3M][0] = 2;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i1;  pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 3;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		}else if((frame_hit3M[1][i_hitD][i*(k2_domain+1)+j2]== 1) &&
		    (frame_hit3M[2][i_hitD][i1*(k2_domain+1)+j]== 0) &&
		    (frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]== 1)){    //3&1
		  pair_hit3M[count_Pair3M][0] = 3;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;   pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 1;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j2;
		  count_Pair3M++;
		}else{}

		/* Added by N. Yamaguchi ***/
	      } else {
		if (frame_hit3M[1][i_hitD][i*(k2_domain+1)+j1] && frame_hit3M[2][i_hitD][i1*(k2_domain+1)+j] && !frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]){
		  pair_hit3M[count_Pair3M][0] = 1;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;  pair_hit3M[count_Pair3M][3] = j1;
		  pair_hit3M[count_Pair3M][4] = 2;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i1;  pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		} else if (!frame_hit3M[1][i_hitD][i*(k2_domain+1)+j1] && frame_hit3M[2][i_hitD][i1*(k2_domain+1)+j] && frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]){
		  pair_hit3M[count_Pair3M][0] = 2;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i1;  pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 3;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		} else if (frame_hit3M[1][i_hitD][i*(k2_domain+1)+j1] && !frame_hit3M[2][i_hitD][i1*(k2_domain+1)+j] && frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]){
		  pair_hit3M[count_Pair3M][0] = 3;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;   pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 1;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j1;
		  count_Pair3M++;
		}

		if (frame_hit3M[1][i_hitD][i*(k2_domain+1)+j2] && frame_hit3M[2][i_hitD][i2*(k2_domain+1)+j] && !frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]){
		  pair_hit3M[count_Pair3M][0] = 1;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;   pair_hit3M[count_Pair3M][3] = j2;
		  pair_hit3M[count_Pair3M][4] = 2;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i2;  pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		} else if (!frame_hit3M[1][i_hitD][i*(k2_domain+1)+j2] && frame_hit3M[2][i_hitD][i2*(k2_domain+1)+j] && frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]){
		  pair_hit3M[count_Pair3M][0] = 2;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i2;  pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 3;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j;
		  count_Pair3M++;
		} else if (frame_hit3M[1][i_hitD][i*(k2_domain+1)+j2] && !frame_hit3M[2][i_hitD][i2*(k2_domain+1)+j] && frame_hit3M[3][i_hitD][i*(k2_domain+1)+j]){
		  pair_hit3M[count_Pair3M][0] = 3;  pair_hit3M[count_Pair3M][1] = i_hitD;
		  pair_hit3M[count_Pair3M][2] = i;   pair_hit3M[count_Pair3M][3] = j;
		  pair_hit3M[count_Pair3M][4] = 1;  pair_hit3M[count_Pair3M][5] = i_hitD;
		  pair_hit3M[count_Pair3M][6] = i;   pair_hit3M[count_Pair3M][7] = j2;
		  count_Pair3M++;
		}
	      }
	      /* ***/

	    }//j
	  }//i
	}//i_hitD
	// #####################################################
	// ### SEARCH CLOSED CIRCLES ###########################
	// MALLOC
	trace_Cc = (int*)malloc(sizeof(int)*(count_Pair3M*2+1));
	for(i=0; i<=count_Pair3M*2; i++)  trace_Cc[i] = 0;

	k_xyz_Cc = (double**)malloc(sizeof(double*)*3);
	for(i=0; i<3; i++){
	  k_xyz_Cc[i] = (double*)malloc(sizeof(double)*(count_Pair3M*2+1));
	}
	EIG_Cc = (double*)malloc(sizeof(double)*(count_Pair3M*2+1));

	/* Disabled by N. Yamaguchi ***
	   MulP_Cc = (double**)malloc(sizeof(double*)*8);
	   for (i=0; i<8; i++){
	   MulP_Cc[i] = (double*)malloc(sizeof(double)*((atomnum+1)*(count_Pair3M*2+1)));
	   for (k=0; k<((atomnum+1)*(count_Pair3M*2+1)); k++) MulP_Cc[i][k] = 0.0;
	   }
	   OrbMulP_Cc = (double***)malloc(sizeof(double**)*8);
	   for (i=0; i<8; i++){
	   OrbMulP_Cc[i] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	   for (j=0; j<=ClaOrb_MAX[0]; j++){
	   OrbMulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(count_Pair3M*2+1)));
	   for (k=0; k<((atomnum+1)*(count_Pair3M*2+1)); k++) OrbMulP_Cc[i][j][k] = 0.0;
	   }//j
	   }//i
	   Orb2MulP_Cc = (double***)malloc(sizeof(double**)*8);
	   for (i=0; i<8; i++){
	   Orb2MulP_Cc[i] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	   for (j=0; j<=ClaOrb_MAX[1]; j++){
	   Orb2MulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(count_Pair3M*2+1)));
	   for (k=0; k<((atomnum+1)*(count_Pair3M*2+1)); k++) Orb2MulP_Cc[i][j][k] = 0.0;
	   }//j
	   }//i
	// kotaka
	 * ***/

	/* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
	MulP_Cc=(double**)malloc(sizeof(double*)*sizeMulP);
	if (myrank==0){
	  for (i=0; i<sizeMulP; i++){
	    MulP_Cc[i]=(double*)malloc(sizeof(double)*(atomnum+1)*(count_Pair3M*2+1));
	    for (k=0; k<(atomnum+1)*(count_Pair3M*2+1); k++){
	      MulP_Cc[i][k]=0.0;
	    }
	  }
	}
	OrbMulP_Cc=(double***)malloc(sizeof(double**)*sizeMulP);
	for (i=0; i<sizeMulP; i++){
	  OrbMulP_Cc[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	  if (myrank==0){
	    for (j=0; j<=ClaOrb_MAX[0]; j++){
	      OrbMulP_Cc[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*(count_Pair3M*2+1));
	      for (k=0; k<(atomnum+1)*(count_Pair3M*2+1); k++){
		OrbMulP_Cc[i][j][k]=0.0;
	      }
	    }
	  }
	}
	Orb2MulP_Cc=(double***)malloc(sizeof(double**)*sizeMulP);
	for (i=0; i<sizeMulP; i++){
	  Orb2MulP_Cc[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	  if (myrank==0){
	    for (j=0; j<=ClaOrb_MAX[1]; j++){
	      Orb2MulP_Cc[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*(count_Pair3M*2+1));
	      for (k=0; k<(atomnum+1)*(count_Pair3M*2+1); k++){
		Orb2MulP_Cc[i][j][k]=0.0;
	      }
	    }
	  }
	}
#endif
	/* ***/

	strcpy(fname_FS,fname_out);
	name_Nband(fname_FS,".FermiSurf_",l);
	fp1 = fopen(fname_FS,"a");

	head_Cc[0] = 0;    head_Cc[1] = 0;    head_Cc[2] = 0;    head_Cc[3] = 0;
	tail_Cc[0] = 0;    tail_Cc[1] = 0;    tail_Cc[2] = 0;    tail_Cc[3] = 0;
	i_Cc = 0;
	for(i=0; i<count_Pair3M; i++){
	  if (trace_Cc[i]==0){
	    for(j=0; j<2; j++){
	      i1 = pair_hit3M[i][4*j+2]*(k2_domain+1)+pair_hit3M[i][4*j+3];
	      k_xyz_Cc[0][i_Cc] = k_xyz_frame3M[pair_hit3M[i][4*j+0]][0][pair_hit3M[i][4*j+1]][i1];
	      k_xyz_Cc[1][i_Cc] = k_xyz_frame3M[pair_hit3M[i][4*j+0]][1][pair_hit3M[i][4*j+1]][i1];
	      k_xyz_Cc[2][i_Cc] = k_xyz_frame3M[pair_hit3M[i][4*j+0]][2][pair_hit3M[i][4*j+1]][i1];
	      EIG_Cc[i_Cc] = EIG_frame3M[pair_hit3M[i][4*j+0]][pair_hit3M[i][4*j+1]][i1];
	      if (myrank == 0){
		Print_kxyzEig(Pdata_s, k_xyz_Cc[0][i_Cc], k_xyz_Cc[1][i_Cc], k_xyz_Cc[2][i_Cc], l, EIG_Cc[i_Cc]);
		fprintf(fp1, "%s ", Pdata_s);

		/* Disabled by N. Yamaguchi ***
		   fprintf(fp1, " (%4d %4d %4d %4d)\n", pair_hit3M[i][4*j+0], pair_hit3M[i][4*j+1], pair_hit3M[i][4*j+2], pair_hit3M[i][4*j+3]);
		 * ***/

		/* Added by N. Yamaguchi ***/
		fputs("\n", fp1);
		/* ***/

	      }
	      i_Cc++;
	    }//j
	    head_Cc[0] = pair_hit3M[i][0];    head_Cc[1] = pair_hit3M[i][1];
	    head_Cc[2] = pair_hit3M[i][2];    head_Cc[3] = pair_hit3M[i][3];
	    tail_Cc[0] = pair_hit3M[i][4];    tail_Cc[1] = pair_hit3M[i][5];
	    tail_Cc[2] = pair_hit3M[i][6];    tail_Cc[3] = pair_hit3M[i][7];
	    trace_Cc[i] = 1;

	    while(1){//while
	      for(hit_Cc=0,i2=0; i2<count_Pair3M; i2++){
		if (trace_Cc[i2] == 0){
		  for(j=0; j<2; j++){
		    if(    (pair_hit3M[i2][4*j+0] == tail_Cc[0]) && (pair_hit3M[i2][4*j+1] == tail_Cc[1])
			&& (pair_hit3M[i2][4*j+2] == tail_Cc[2]) && (pair_hit3M[i2][4*j+3] == tail_Cc[3])){

		      tail_Cc[0] = pair_hit3M[i2][4*(abs(j-1))+0];    tail_Cc[1] = pair_hit3M[i2][4*(abs(j-1))+1];
		      tail_Cc[2] = pair_hit3M[i2][4*(abs(j-1))+2];    tail_Cc[3] = pair_hit3M[i2][4*(abs(j-1))+3];
		      k_xyz_Cc[0][i_Cc] = k_xyz_frame3M[tail_Cc[0]][0][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		      k_xyz_Cc[1][i_Cc] = k_xyz_frame3M[tail_Cc[0]][1][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		      k_xyz_Cc[2][i_Cc] = k_xyz_frame3M[tail_Cc[0]][2][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		      EIG_Cc[i_Cc] = EIG_frame3M[tail_Cc[0]][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		      if (myrank == 0){
			Print_kxyzEig(Pdata_s, k_xyz_Cc[0][i_Cc], k_xyz_Cc[1][i_Cc], k_xyz_Cc[2][i_Cc], l, EIG_Cc[i_Cc]);
			fprintf(fp1, "%s ", Pdata_s);

			/* Disabled by N. Yamaguchi ***
			   fprintf(fp1, " (%4d %4d %4d %4d)\n", tail_Cc[0], tail_Cc[1], tail_Cc[2], tail_Cc[3]);
			 * ***/

			/* Added by N. Yamaguchi ***/
			fputs("\n", fp1);
			/* ***/

		      }
		      hit_Cc = 1;    i_Cc++;    trace_Cc[i2] = 1;    break;
		    }//if
		  }//j
		}//if (trace_Cc[i2] == 0)
		if (hit_Cc == 1) break;
	      }//i2
	      //kotaka
	      if (hit_Cc == 0){
		if(func_periodicity(tail_Cc, count_Pair3M, trace_Cc, k_index_hitD, pair_hit3M, k1_domain, k2_domain) >0){

		  k_xyz_Cc[0][i_Cc] = k_xyz_frame3M[tail_Cc[0]][0][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		  k_xyz_Cc[1][i_Cc] = k_xyz_frame3M[tail_Cc[0]][1][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		  k_xyz_Cc[2][i_Cc] = k_xyz_frame3M[tail_Cc[0]][2][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		  EIG_Cc[i_Cc] = EIG_frame3M[tail_Cc[0]][tail_Cc[1]][tail_Cc[2]*(k2_domain+1)+tail_Cc[3]];
		  if (myrank == 0){
		    Print_kxyzEig(Pdata_s, k_xyz_Cc[0][i_Cc], k_xyz_Cc[1][i_Cc], k_xyz_Cc[2][i_Cc], l, EIG_Cc[i_Cc]);
		    fprintf(fp1, "%s ", Pdata_s);

		    /* Disabled by N. Yamaguchi ***
		       fprintf(fp1, " (%4d %4d %4d %4d)\n", tail_Cc[0], tail_Cc[1], tail_Cc[2], tail_Cc[3]);
		     * ***/

		    /* Added by N. Yamaguchi ***/
		    fputs("\n", fp1);
		    /* ***/

		  }
		  hit_Cc = 1;  i_Cc++;
		}//if
	      }//if
	      if((hit_Cc == 0) ||
		  (    (tail_Cc[0] == head_Cc[0]) && (tail_Cc[1] == head_Cc[1])
		       && (tail_Cc[2] == head_Cc[2]) && (tail_Cc[3] == head_Cc[3]))){ break; }
	    }//while
	    if (myrank == 0) fprintf(fp1, "\n\n");
	  }//if
	}//i
	if (myrank == 0){
	  fclose(fp1);

	  /* Disabled by N. Yamaguchi
	   * printf(" l=%4d,   k_points:%4d (array:%4d)", l, i_Cc, 2*count_Pair3M+1);
	   */

	  /* Added by N. Yamaguchi ***/
	  PRINTFF(" l=%4d,   k_points:%4d (array:%4d)", l, i_Cc, 2*count_Pair3M+1);
	  /* ***/

	  //          printf(" (hitPair: %4d) (hit3M: %4d)",count_Pair3M+1,count_T_hit3M);

	  /* Disabled by N. Yamaguchi
	   * printf("\n");
	   */

	  /* Added by N. Yamaguchi ***/
	  PRINTFF("\n");
	  /* ***/

	}
	// ### Time ############################################
	dtime(&Etime);
	Time_Contour[l-l_min]+= Etime-Stime;
	dtime(&Stime);
	// #####################################################
	if (i_Cc > 0){
	  // ### MulP_Calculation ################################
	  T2_knum = i_Cc;  //i_Cc
	  // Division CALC_PART
	  for(i=0; i<num_procs; i++){
	    if (T2_knum <= i){
	      S2_knum[i] = -10;   E2_knum[i] = -100;

	      /* Added by N. Yamaguchi ***/
#if 0
	      /* ***/

	    } else if (T2_knum < num_procs) {

	      /* Added by N. Yamaguchi ***/
#endif
	    } else if (T2_knum<=num_procs){
	      /* ***/

	      S2_knum[i] = i;     E2_knum[i] = i;
	    } else {
	      d0 = (double)T2_knum/(double)num_procs;
	      S2_knum[i] = (int)((double)i*(d0+0.0001));
	      E2_knum[i] = (int)((double)(i+1)*(d0+0.0001)) - 1;
	      if (i==(num_procs-1)) E2_knum[i] = T2_knum - 1;
	      if (E2_knum[i]<0)      E2_knum[i] = 0;
	    }
	  }

	  /* Added by N. Yamaguchi ***/
	  numK=(int*)malloc(sizeof(int)*num_procs);
	  for (j=0; j<num_procs; j++){
	    numK[j]=E2_knum[j]-S2_knum[j]+1;
	    if (numK[j]<=0){
	      numK[j]=0;
	    }
	  }
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
	  MulP_CcDivision=(double**)malloc(sizeof(double*)*sizeMulP);
	  for (i=0; i<sizeMulP; i++){
	    MulP_CcDivision[i]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	    for (j=0; j<(atomnum+1)*numK[myrank]; j++) MulP_CcDivision[i][j]=0.0;
	  }
#endif
#ifdef DEBUG_SIGMAEK_OLDER_20181126
	  OrbMulP_CcDivision=(double***)malloc(sizeof(double**)*sizeMulP);
	  for (i=0; i<sizeMulP; i++){
	    OrbMulP_CcDivision[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	    for (j=0; j<=ClaOrb_MAX[0]; j++){
	      OrbMulP_CcDivision[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	      for (k=0; k<(atomnum+1)*numK[myrank]; k++) OrbMulP_CcDivision[i][j][k]=0.0;
	    }
	  }
#endif
	  Orb2MulP_CcDivision=(double***)malloc(sizeof(double**)*sizeMulP);
	  for (i=0; i<sizeMulP; i++){
	    Orb2MulP_CcDivision[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	    for (j=0; j<=ClaOrb_MAX[1]; j++){
	      Orb2MulP_CcDivision[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	      for (k=0; k<(atomnum+1)*numK[myrank]; k++) Orb2MulP_CcDivision[i][j][k]=0.0;
	    }
	  }
	  /* ***/

	  Nband[0] = l;
	  if ((l-l_min)%2==0){Nband[1]=l+1;}
	  else if((l-l_min)%2==1){Nband[1]=l-1;}
	  else{Nband[1]=0;}

	  /* Disabled by N. Yamaguchi ***
	  //for (j = 0; j < i_Cc; j++)
	  for (j = S2_knum[myrank]; j <= E2_knum[myrank]; j++){
	  for (i=0; i < atomnum; i++){
	  for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  for (j1=0; j1<8; j1++){ OrbMulP_Cc[j1][i1][j*(atomnum+1)+i] = 0;  }//j1
	  }//for(i1)
	  for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
	  for (j1=0; j1<8; j1++){ Orb2MulP_Cc[j1][i1][j*(atomnum+1)+i] = 0; }//j1
	  }//for(i1)
	  }//for(i)
	  }//for(j)
	   * ***/

	  for (j = S2_knum[myrank]; j <= E2_knum[myrank]; j++){

	    /* Added by N. Yamaguchi ***/
#if !defined DEBUG_SIGMAEK_OLDER_20181126 && !defined DEBUG_SIGMAEK_OLDER_20190114
	    Data_MulP = (double****)malloc(sizeof(double***)*4);
	    for (i=0; i<4; i++){
	      Data_MulP[i] = (double***)malloc(sizeof(double**)*1);
	      Data_MulP[i][0] = (double**)malloc(sizeof(double*)*(atomnum+1));
	      for (j1=0; j1<=atomnum; j1++){
		Data_MulP[i][0][j1] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
	      }
	    }
#endif
	    /* ***/

	    EigenValue_Problem(k_xyz_Cc[0][j] ,k_xyz_Cc[1][j] ,k_xyz_Cc[2][j],1);
	    for (i=0; i < atomnum; i++){

	      /* Added by N. Yamaguchi ***/
#if 0
	      /* ***/

	      for (j2=0; j2<2; j2++){

		/* Added by N. Yamaguchi ***/
#endif
		for (j2=0; j2<Spin_Dege+1; j2++){
		  /* ***/

		  for (j1=0; j1<4; j1++){

		    /* Disabled by N. Yamaguchi ***
		       MulP_Cc[j1+j2*4][j*(atomnum+1)+i] = 0.0;
		     * ***/

		    for (i1=0; i1 < Total_NumOrbs[i+1]; i1++){
		      if(Nband[j2]==l){

			/* Disabled by N. Yamaguchi
			 * Orb2MulP_Cc[j1+j2*4][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]][i+1][i1];
			 * MulP_Cc[j1+j2*4][j*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]][i+1][i1];
			 */

			/* Added by N. Yamaguchi ***/
#ifdef DEBUG_SIGMAEK_OLDER_20181126
			//Orb2MulP_Cc[j1+j2*4][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
			//MulP_Cc[j1+j2*4][j*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
			Orb2MulP_CcDivision[j1+j2*4][ClaOrb[i+1][i1]][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
#else
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
			MulP_CcDivision[j1+j2*4][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][0][i+1][i1];
#endif
			Orb2MulP_CcDivision[j1+j2*4][ClaOrb[i+1][i1]][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][0][i+1][i1];
#endif
#ifdef DEBUG_SIGMAEK_OLDER_20181126
			if (ClaOrb[i+1][i1]==0){
			  OrbMulP_CcDivision[j1+j2*4][0][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
			} else if (ClaOrb[i+1][i1]>=1 && ClaOrb[i+1][i1]<=3){
			  OrbMulP_CcDivision[j1+j2*4][1][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
			} else if (ClaOrb[i+1][i1]>=4 && ClaOrb[i+1][i1]<=8){
			  OrbMulP_CcDivision[j1+j2*4][2][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
			} else if (ClaOrb[i+1][i1]>=9 && ClaOrb[i+1][i1]<=15){
			  OrbMulP_CcDivision[j1+j2*4][3][(j-S2_knum[myrank])*(atomnum+1)+i]+= Data_MulP[j1][Nband[j2]-l_min][i+1][i1];
			}
#endif
			/* ***/

		      }//if
		    }//for(i1)
		  }//for(j1)

		  /* Added by N. Yamaguchi ***/
		}
#if 0
		/* ***/

	      }//for(j2)

	      /* Added by N. Yamaguchi ***/
#endif
	      /* ***/

	    }//for(i)

	    /* Added by N. Yamaguchi ***/
#if !defined DEBUG_SIGMAEK_OLDER_20181126 && !defined DEBUG_SIGMAEK_OLDER_20190114
	    for (i=0; i<4; i++){
	      for (j1=0; j1<=atomnum; j1++){
		free(Data_MulP[i][0][j1]);
	      } free(Data_MulP[i][0]);
	      free(Data_MulP[i]);
	    } free(Data_MulP);
#endif
	    /* ***/

	  }//for(j)
	  MPI_Barrier(MPI_COMM_WORLD);

	  /* Added by N. Yamaguchi ***/
#if !defined DEBUG_SIGMAEK_OLDER_20181126 && !defined DEBUG_SIGMAEK_OLDER_20190114
	  Orb2MulP_Cc=(double***)malloc(sizeof(double**)*sizeMulP);
	  for (i=0; i<sizeMulP; i++){
	    Orb2MulP_Cc[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	    if (myrank==0){
	      for (j=0; j<=ClaOrb_MAX[1]; j++){
		Orb2MulP_Cc[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*(count_Pair3M*2+1));
		for (k=0; k<(atomnum+1)*(count_Pair3M*2+1); k++){
		  Orb2MulP_Cc[i][j][k]=0.0;
		}
	      }
	    }
	  }
#endif
	  /* ***/

	  /* Disabled by N. Yamaguchi
	   * MPI_Barrier(MPI_COMM_WORLD);
	   */

	  /* Disabled by N. Yamaguchi
	   * for (i=0; i<num_procs; i++)\{
	   * k = S2_knum[i]*(atomnum+1);
	   * i2 = abs(E2_knum[i]-S2_knum[i]+1)*(atomnum+1);
	   * MPI_Barrier(MPI_COMM_WORLD);
	   * for (j1=0; j1<8; j1++){
	   * MPI_Bcast(&MulP_Cc[j1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
	   * for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
	   * MPI_Bcast(&Orb2MulP_Cc[j1][i1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
	   * }//for(i1)
	   * }//for(j1)
	   */

	  /* Added by N. Yamaguchi ***/
#if 0
	  for (i=0; i<num_procs; i++){
	    if (S2_knum[i]>0){
	      k = S2_knum[i]*(atomnum+1);
	      i2 = abs(E2_knum[i]-S2_knum[i]+1)*(atomnum+1);
	      for (j1=0; j1<4*(Spin_Dege+1); j1++){
		MPI_Bcast(&MulP_Cc[j1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
		for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
		  MPI_Bcast(&Orb2MulP_Cc[j1][i1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
		}//for(i1)
	      }//for(j1)
	    }
	  }//i
#endif
	  recvCount=(int*)malloc(sizeof(int)*num_procs);
	  displs=(int*)malloc(sizeof(int)*num_procs);
	  i2=(atomnum+1)*numK[myrank];
	  for (j=0; j<num_procs; j++){
	    recvCount[j]=(atomnum+1)*numK[j];
	    displs[j]=S2_knum[j]*(atomnum+1);
	  }
	  free(numK);
	  for (j1=0; j1<sizeMulP; j1++){
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
	    MPI_Gatherv(MulP_CcDivision[j1], i2, MPI_DOUBLE, MulP_Cc[j1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
#ifdef DEBUG_SIGMAEK_OLDER_20181126
	    for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	      MPI_Gatherv(OrbMulP_CcDivision[j1][i1], i2, MPI_DOUBLE, OrbMulP_Cc[j1][i1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    }
#endif
	    for (i1=0; i1<=ClaOrb_MAX[1]; i1++){
	      MPI_Gatherv(Orb2MulP_CcDivision[j1][i1], i2, MPI_DOUBLE, Orb2MulP_Cc[j1][i1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    }
	  }
	  free(recvCount);
	  free(displs);
	  /* ***/


	  /* Disabled by N. Yamaguchi ***
	     for(j=0; j<i_Cc; j++){
	     for (i=0; i<atomnum; i++){
	     for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	     for (i2=0; i2 <=ClaOrb_MAX[1]; i2++){
	     if (OrbSym[i1][0] == OrbName[i2][0]){
	     for (j1=0; j1<8; j1++){
	     OrbMulP_Cc[j1][i1][j*(atomnum+1)+i]+= Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
	     }//j1
	     }//if
	     }//i2
	     }//i1
	     }//j
	     }//i
	  //kotaka
	   * ***/

	  /* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
	  if (myrank==0){
	    for(j=0; j<i_Cc; j++){
	      for (i=0; i<atomnum; i++){
		for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
		  for (i2=0; i2<=ClaOrb_MAX[1]; i2++){
		    if (OrbSym[i1][0]==OrbName[i2][0]){
		      for (j1=0; j1<sizeMulP; j1++){
			OrbMulP_Cc[j1][i1][j*(atomnum+1)+i]+=Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
		      }
		    }
		  }
		}
	      }
	    }
	  }
#endif
	  /* ***/

	  hit_Total[0]+= i_Cc;
	  if (myrank == 0){
	    // ###################################################
	    strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP");
	    fp1= fopen(fname_MP,"a");
	    for(j=0; j<i_Cc; j++){
	      Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l, EIG_Cc[j]);
	      fprintf(fp1, "%s", Pdata_s);
	      for (i=0; i<atomnum; i++){

		/* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
		/* ***/

		fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[0][j*(atomnum+1)+i], MulP_Cc[1][j*(atomnum+1)+i]);
		fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[2][j*(atomnum+1)+i], MulP_Cc[3][j*(atomnum+1)+i]);

		/* Added by N. Yamaguchi ***/
#else
		for (j1=0; j1<4; j1++){
		  double tempMulP_Cc=0.0;
		  for (i2=0; i2<=ClaOrb_MAX[1]; i2++){
		    tempMulP_Cc+=Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
		  }
		  fprintf(fp1,"%10.6lf ", tempMulP_Cc);
		}
#endif
		/* ***/

	      } fprintf(fp1,"\n");
	    }//j
	    fclose(fp1);
	    // ###################################################
	    for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	      strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbSym[i1]);
	      fp1= fopen(fname_MP,"a");
	      for(j=0; j<i_Cc; j++){
		Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l, EIG_Cc[j]);
		fprintf(fp1, "%s", Pdata_s);
		for (i=0; i<atomnum; i++){

		  /* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
		  /* ***/

		  fprintf(fp1,"%10.6lf %10.6lf ", OrbMulP_Cc[0][i1][j*(atomnum+1)+i], OrbMulP_Cc[1][i1][j*(atomnum+1)+i]);
		  fprintf(fp1,"%10.6lf %10.6lf ", OrbMulP_Cc[2][i1][j*(atomnum+1)+i], OrbMulP_Cc[3][i1][j*(atomnum+1)+i]);

		  /* Added by N. Yamaguchi ***/
#else
		  for (j1=0; j1<4; j1++){
		    double tempMulP_Cc=0.0;
		    for (i2=0; i2<=ClaOrb_MAX[1]; i2++){
		      if (OrbSym[i1][0]==OrbName[i2][0]){
			tempMulP_Cc+=Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
		      }
		    }
		    fprintf(fp1,"%10.6lf ", tempMulP_Cc);
		  }
#endif
		  /* ***/

		} fprintf(fp1,"\n");
	      }//j
	      fclose(fp1);
	    }//i1
	    // ###################################################
	    for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	      strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbName[i1]);
	      fp1= fopen(fname_MP,"a");
	      for(j=0; j<i_Cc; j++){
		Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l, EIG_Cc[j]);
		fprintf(fp1, "%s", Pdata_s);
		for (i=0; i<atomnum; i++){
		  fprintf(fp1,"%10.6lf %10.6lf ", Orb2MulP_Cc[0][i1][j*(atomnum+1)+i], Orb2MulP_Cc[1][i1][j*(atomnum+1)+i]);
		  fprintf(fp1,"%10.6lf %10.6lf ", Orb2MulP_Cc[2][i1][j*(atomnum+1)+i], Orb2MulP_Cc[3][i1][j*(atomnum+1)+i]);
		} fprintf(fp1,"\n");
	      }//j
	      fclose(fp1);
	    }//i1
	    // ###################################################
	  }//if(myrank)

	  // ### Spin Degenerate #################################
	  if ((Spin_Dege == 1) && (Nband[0] < Nband[1])){
	    hit_Total[1]+= i_Cc*2;
	    if (myrank == 0){
	      // ###################################################
	      strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP");  strcat(fname_MP,"_Dege");
	      fp1= fopen(fname_MP,"a");
	      for(j=0; j<i_Cc; j++){
		Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l, EIG_Cc[j]);
		fprintf(fp1, "%s", Pdata_s);
		for (i=0; i<atomnum; i++){

		  /* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
		  /* ***/

		  fprintf(fp1,"%10.6lf ", (MulP_Cc[0][j*(atomnum+1)+i]+ MulP_Cc[4][j*(atomnum+1)+i]) );
		  fprintf(fp1,"%10.6lf ", (MulP_Cc[1][j*(atomnum+1)+i]+ MulP_Cc[5][j*(atomnum+1)+i]) );
		  fprintf(fp1,"%10.6lf ", (MulP_Cc[2][j*(atomnum+1)+i]+ MulP_Cc[6][j*(atomnum+1)+i]) );
		  fprintf(fp1,"%10.6lf ", (MulP_Cc[3][j*(atomnum+1)+i]+ MulP_Cc[7][j*(atomnum+1)+i]) );

		  /* Added by N. Yamaguchi ***/
#else
		  for (j1=0; j1<8; j1++){
		    double tempMulP_Cc=0.0;
		    for (i2=0; i2<=ClaOrb_MAX[1]; i2++){
		      tempMulP_Cc+=Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
		    }
		    fprintf(fp1,"%10.6lf ", tempMulP_Cc);
		  }
#endif
		  /* ***/

		} fprintf(fp1,"\n");
	      }//j

	      /* Added by N. Yamaguchi ***/
	      fclose(fp1);
	      /* ***/

	      // #################################################
	      for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
		strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbSym[i1]);
		strcat(fname_MP,"_Dege");
		fp1= fopen(fname_MP,"a");
		for(j=0; j<i_Cc; j++){
		  Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l, EIG_Cc[j]);
		  fprintf(fp1, "%s", Pdata_s);
		  for (i=0; i<atomnum; i++){

		    /* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
		    /* ***/

		    fprintf(fp1,"%10.6lf ", (OrbMulP_Cc[0][i1][j*(atomnum+1)+i]+ OrbMulP_Cc[4][i1][j*(atomnum+1)+i]) );
		    fprintf(fp1,"%10.6lf ", (OrbMulP_Cc[1][i1][j*(atomnum+1)+i]+ OrbMulP_Cc[5][i1][j*(atomnum+1)+i]) );
		    fprintf(fp1,"%10.6lf ", (OrbMulP_Cc[2][i1][j*(atomnum+1)+i]+ OrbMulP_Cc[6][i1][j*(atomnum+1)+i]) );
		    fprintf(fp1,"%10.6lf ", (OrbMulP_Cc[3][i1][j*(atomnum+1)+i]+ OrbMulP_Cc[7][i1][j*(atomnum+1)+i]) );

		  /* Added by N. Yamaguchi ***/
#else
		  for (j1=0; j1<8; j1++){
		    double tempMulP_Cc=0.0;
		    for (i2=0; i2<=ClaOrb_MAX[1]; i2++){
		      if (OrbSym[i1][0]==OrbName[i2][0]){
			tempMulP_Cc+=Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
		      }
		    }
		    fprintf(fp1,"%10.6lf ", tempMulP_Cc);
		  }
#endif
		  /* ***/

		  } fprintf(fp1,"\n");
		}//j
		fclose(fp1);
	      }//i1
	      // #################################################
	      for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
		strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbName[i1]);
		strcat(fname_MP,"_Dege");
		fp1= fopen(fname_MP,"a");
		for(j=0; j<i_Cc; j++){
		  Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l, EIG_Cc[j]);
		  fprintf(fp1, "%s", Pdata_s);
		  for (i=0; i<atomnum; i++){
		    fprintf(fp1,"%10.6lf ", (Orb2MulP_Cc[0][i1][j*(atomnum+1)+i]+ Orb2MulP_Cc[4][i1][j*(atomnum+1)+i]) );
		    fprintf(fp1,"%10.6lf ", (Orb2MulP_Cc[1][i1][j*(atomnum+1)+i]+ Orb2MulP_Cc[5][i1][j*(atomnum+1)+i]) );
		    fprintf(fp1,"%10.6lf ", (Orb2MulP_Cc[2][i1][j*(atomnum+1)+i]+ Orb2MulP_Cc[6][i1][j*(atomnum+1)+i]) );
		    fprintf(fp1,"%10.6lf ", (Orb2MulP_Cc[3][i1][j*(atomnum+1)+i]+ Orb2MulP_Cc[7][i1][j*(atomnum+1)+i]) );
		  } fprintf(fp1,"\n");
		}//j
		fclose(fp1);
	      }//i1
	      // #################################################
	    }//if(myrank)
	  }//if

	  if (myrank == 0){
	    // ###################################################
	    strcpy(fname_Spin,fname_out);
	    name_Nband(fname_Spin,".Pxyz_",l);
	    fp1 = fopen(fname_Spin,"a");
	    for(j=0; j<i_Cc; j++){
	      for(i=0; i<3; i++){
		fprintf(fp1,"%10.6lf ",rtv[1][i+1]*k_xyz_Cc[0][j]+ rtv[2][i+1]*k_xyz_Cc[1][j]+ rtv[3][i+1]*k_xyz_Cc[2][j]);
	      }
	      Re11 = 0;      Re22 = 0;      Re12 = 0;      Im12 = 0;
	      for (i=0; i<atomnum; i++){

		/* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
		/* ***/

		Re11+= MulP_Cc[0][j*(atomnum+1)+i];        Re22+= MulP_Cc[1][j*(atomnum+1)+i];
		Re12+= MulP_Cc[2][j*(atomnum+1)+i];        Im12+= MulP_Cc[3][j*(atomnum+1)+i];

		/* Added by N. Yamaguchi ***/
#else
		double tempMulP_Cc[4]={0.0};
		for (j1=0; j1<4; j1++){
		  for (i2=0; i2<=ClaOrb_MAX[1]; i2++){
		    tempMulP_Cc[j1]+=Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
		  }
		}
		Re11+=tempMulP_Cc[0];
		Re22+=tempMulP_Cc[1];
		Re12+=tempMulP_Cc[2];
		Im12+=tempMulP_Cc[3];
#endif
		/* ***/

	      } EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndw, Ntheta, Nphi );

	      /* Added by N. Yamaguchi ***/
#ifdef DEBUG
	      fprintf(fp1,"%10.6lf  ",Re11);
	      fprintf(fp1,"%10.6lf  ",Re12);
	      fprintf(fp1,"%10.6lf  ",Re22);
	      fprintf(fp1,"%10.6lf  ",Im12);
	      fprintf(fp1,"%10.6lf  ",Nup[0]);
	      fprintf(fp1,"%10.6lf  ",Ndw[0]);
#endif
	      /* ***/

	      fprintf(fp1,"%10.6lf  ",MulP_VecScale[0]* (Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]));
	      fprintf(fp1,"%10.6lf  ",MulP_VecScale[1]* (Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]));
	      fprintf(fp1,"%10.6lf\n",MulP_VecScale[2]* (Nup[0] -Ndw[0]) *cos(Ntheta[0]));
	    }//j
	    fclose(fp1);
	  }//if(myrank)
	  // ### MALLOC FREE #####################################
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
	  for (i=0; i<sizeMulP; i++){
	    free(MulP_CcDivision[i]);
	  } free(MulP_CcDivision);
#endif
#ifdef DEBUG_SIGMAEK_OLDER_20181126
	  for (i=0; i<sizeMulP; i++){
	    for (j=0; j<=ClaOrb_MAX[0]; j++){
	      free(OrbMulP_CcDivision[i][j]);
	    } free(OrbMulP_CcDivision[i]);
	  } free(OrbMulP_CcDivision);
#endif
	  for (i=0; i<sizeMulP; i++){
	    for (j=0; j<=ClaOrb_MAX[1]; j++){
	      free(Orb2MulP_CcDivision[i][j]);
	    } free(Orb2MulP_CcDivision[i]);
	  } free(Orb2MulP_CcDivision);
#if !defined DEBUG_SIGMAEK_OLDER_20181126 && !defined DEBUG_SIGMAEK_OLDER_20190114
	  for (i=0; i<sizeMulP; i++){
	    for (j=0; j<=ClaOrb_MAX[1]; j++){
	      if (myrank==0){
		free(Orb2MulP_Cc[i][j]);
	      }
	    } free(Orb2MulP_Cc[i]);
	  } free(Orb2MulP_Cc);
#endif
	}//if(i_Cc>0)
	free(trace_Cc);

	for(i=0; i<3; i++){
	  free(k_xyz_Cc[i]);
	} free(k_xyz_Cc);
	free(EIG_Cc);

	/* Disabled by N. Yamaguchi ***
	   for (i=0; i<8; i++){
	   free(MulP_Cc[i]);
	   } free(MulP_Cc);
	   for (i=0; i<8; i++){
	   for (j=0; j<=ClaOrb_MAX[0]; j++){
	   free(OrbMulP_Cc[i][j]);
	   } free(OrbMulP_Cc[i]);
	   } free(OrbMulP_Cc);
	   for (i=0; i<8; i++){
	   for (j=0; j<=ClaOrb_MAX[1]; j++){
	   free(Orb2MulP_Cc[i][j]);
	   } free(Orb2MulP_Cc[i]);
	   } free(Orb2MulP_Cc);
	 * ***/

	/* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
	for (i=0; i<sizeMulP; i++){
	  if (myrank==0){
	    free(MulP_Cc[i]);
	  }
	} free(MulP_Cc);
	for (i=0; i<sizeMulP; i++){
	  for (j=0; j<=ClaOrb_MAX[0]; j++){
	    if (myrank==0){
	      free(OrbMulP_Cc[i][j]);
	    }
	  } free(OrbMulP_Cc[i]);
	} free(OrbMulP_Cc);
	for (i=0; i<sizeMulP; i++){
	  for (j=0; j<=ClaOrb_MAX[1]; j++){
	    if (myrank==0){
	      free(Orb2MulP_Cc[i][j]);
	    }
	  } free(Orb2MulP_Cc[i]);
	} free(Orb2MulP_Cc);
#endif
	/* ***/

	for(i=0; i<4; i++){
	  for(j=0; j<=count_hitD; j++){
	    free(frame_hit3M[i][j]);
	  } free(frame_hit3M[i]);
	} free(frame_hit3M);
	for(i=0; i<4; i++){
	  for(j=0; j<3; j++){
	    for(i2=0; i2<=count_hitD; i2++){
	      free(k_xyz_frame3M[i][j][i2]);
	    }free(k_xyz_frame3M[i][j]);
	  } free(k_xyz_frame3M[i]);
	} free(k_xyz_frame3M);
	for(i=0; i<4; i++){
	  for(j=0; j<=count_hitD; j++){
	    free(EIG_frame3M[i][j]);
	  } free(EIG_frame3M[i]);
	} free(EIG_frame3M);
	free(count_hit3M);
	for(i=0; i<=count_T_hit3M; i++){
	  free(pair_hit3M[i]);
	}free(pair_hit3M);

	// ## Time #############################################
	dtime(&Etime);
	Time_MulP[l-l_min]+= Etime-Stime;
	// #####################################################
      }//l

      /* Added by N. Yamaguchi ***/
#if defined DEBUG_SIGMAEK_OLDER_20181126 || defined DEBUG_SIGMAEK_OLDER_20190114
      for (i=0; i<4; i++){
#ifdef DEBUG_SIGMAEK_OLDER_20181126
	for (l=0; l<l_cal; l++){
	  for (j=0; j<=atomnum; j++){
	    free(Data_MulP[i][l][j]);
	  } free(Data_MulP[i][l]);
	} free(Data_MulP[i]);
#else
	for (j=0; j<=atomnum; j++){
	  free(Data_MulP[i][0][j]);
	} free(Data_MulP[i][0]);
	free(Data_MulP[i]);
#endif
      } free(Data_MulP);
#endif
      /* ***/

    }//i_height

    if (myrank == 0){
      //      fclose(fp);

      /* Disabled by N. Yamaguchi
       * printf("Total MulP data:%4d\n" ,hit_Total[0]); 
       * printf("###########################################\n\n");
       */

      /* Added by N. Yamaguchi ***/
      PRINTFF("Total MulP data:%4d\n" ,hit_Total[0]);
      PRINTFF("###########################################\n\n");
      /* ***/

      for (j2=0; j2<=Spin_Dege; j2++){
	//### atomnum & data_size ###
	strcpy(fname_MP,fname_out);     strcat(fname_MP,".AtomMulP");
	if (j2 == 1) strcat(fname_MP,"_Dege");
	fp1= fopen(fname_MP,"r+");
	fseek(fp1, 0L, SEEK_SET);
	fprintf(fp1,"%6d %4d", hit_Total[j2], atomnum);
	fclose(fp1);
	for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  strcpy(fname_MP,fname_out);    strcat(fname_MP,".MulP_");    strcat(fname_MP,OrbSym[i1]);
	  if (j2 == 1) strcat(fname_MP,"_Dege");
	  fp1= fopen(fname_MP,"r+");
	  fseek(fp1, 0L, SEEK_SET);
	  fprintf(fp1,"%6d %4d", hit_Total[j2], atomnum);
	  fclose(fp1);
	}//i1
	for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	  strcpy(fname_MP,fname_out);    strcat(fname_MP,".MulP_");    strcat(fname_MP,OrbName[i1]);
	  if (j2 == 1) strcat(fname_MP,"_Dege");
	  fp1= fopen(fname_MP,"r+");
	  fseek(fp1, 0L, SEEK_SET);
	  fprintf(fp1,"%6d %4d", hit_Total[j2], atomnum);
	  fclose(fp1);
	}//i1
      }//j2
    }//if(myrank)

    // ### MALLOC FREE #######################################

    for (i=0; i<=ClaOrb_MAX[1]; i++){
      free(OrbName[i]);
    } free(OrbName);

    // ### (EigenValue Problem) ###
    for(i=0; i<3; i++){
      free(k_xyz[i]);
    } free(k_xyz);
    free(EIGEN);
    free(EIGEN_MP);
    for(i=0; i<2; i++){
      free(k_index_hitD[i]);
    } free(k_index_hitD);
    for(i=0; i<3; i++){
      free(k_xyz_domain[i]);
    } free(k_xyz_domain);
    free(EIGEN_domain);
    if (myrank==0) PRINTFF("###########################################\n\n");

    // ### (MulP Calculation)   ###
    for (i=0; i<=atomnum; i++){
      free(An2Spe[i]);
    } free(An2Spe);
    for (i=0; i<=atomnum; i++){
      free(ClaOrb[i]);
    } free(ClaOrb);

    /* Disabled by N. Yamaguchi
     * for (i=0; i<4; i++){
     * for (l=0; l<n2; l++){
     * for (j=0; j<=atomnum; j++){
     * free(Data_MulP[i][l][j]);
     * } free(Data_MulP[i][l]);
     * } free(Data_MulP[i]);
     * } free(Data_MulP);
     */

    // ### (MPI Calculation)    ###
    free(S_knum);
    free(E_knum);
    free(S2_knum);
    free(E2_knum);

    // #######################################################

    /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
    free_scfout();
#endif
    /* ***/

  }//if(fo_wf)

  // ### Time Measurement ####################################
  if (myrank ==0){

    /* Disabled by N. Yamaguchi
     * printf("############ CALC TIME ####################\n");
     * dtime(&TEtime);
     * printf("  Total Calculation Time:%10.6lf (s)\n",TEtime-TStime);
     * if ((fo_inp == 1) && (fo_wf == 1)){
     * printf("        Eigen Value Calc:%10.6lf (s)\n",Time_EIG);
     * for(l=l_min; l<=l_max; l++){
     * printf(" l=%4d:    Contour Calc:%10.6lf (s)\n",l,Time_Contour[l-l_min]);
     * printf("               MulP Calc:%10.6lf (s)\n",Time_MulP[l-l_min]);
     * }//l
     * }//if(fo_wf)
     * printf("###########################################\n");
     */

    /* Added by N. Yamaguchi ***/
    PRINTFF("############ CALC TIME ####################\n");
    dtime(&TEtime);
    PRINTFF("  Total Calculation Time:%10.6lf (s)\n",TEtime-TStime);
    if ((fo_inp == 1) && (fo_wf == 1)){
      PRINTFF("        Eigen Value Calc:%10.6lf (s)\n",Time_EIG);
      for(l=l_min; l<=l_max; l++){
	PRINTFF(" l=%4d:    Contour Calc:%10.6lf (s)\n",l,Time_Contour[l-l_min]);
	PRINTFF("               MulP Calc:%10.6lf (s)\n",Time_MulP[l-l_min]);
      }//l
    }//if(fo_wf)
    PRINTFF("###########################################\n");
    /* ***/

  }
  // ### (MALLOC TIME)        ###
  if ((fo_inp == 1) && (fo_wf == 1)){
    free(Time_Contour);
    free(Time_MulP);
  }//if(fo_wf)

  // ### MPI_Finalize ########################################

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  MPI_Finalize();
#endif
  /* ***/

  return 0;
}


// ###########################################################
// ### SUB ROUTINEs ##########################################
// ###########################################################

/* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
/* ***/

void func_Newton(double *k1, double E1, double *k2, double E2, double *k3, double *E3, double EF, int l, int loopMAX){
  int i,j,k;
  double d0,d1,d2,d3;

  if (loopMAX <1) loopMAX = 1;

  for (i=0; i<loopMAX; i++){
    d0 = (E2*k1[0] - E1*k2[0])/(E2 - E1);
    if ((d0<k1[0] && d0<k2[0]) || (d0>k1[0] && d0>k2[0])) d0 = (k1[0]+k2[0])*0.5;
    d1 = (E2*k1[1] - E1*k2[1])/(E2 - E1);
    if ((d1<k1[1] && d1<k2[1]) || (d1>k1[1] && d1>k2[1])) d1 = (k1[1]+k2[1])*0.5;
    d2 = (E2*k1[2] - E1*k2[2])/(E2 - E1);
    if ((d2<k1[2] && d2<k2[2]) || (d2>k1[2] && d2>k2[2])) d2 = (k1[2]+k2[2])*0.5;

    if(switch_Eigen_Newton==1){

      /* Disabled by N. Yamaguchi ***
	 d3 = (E2*E2 - E1*E1)/(E2 - E1);
       * ***/

      /* Added by N. Yamaguchi ***/
      d3=E1+E2;
      /* ***/

      if ((d3<E1 && d3<E2) || (d3>E1 && d3>E2)) d3 = (E1+E2)*0.5;
    }else if(switch_Eigen_Newton==0){
      EigenValue_Problem(d0, d1, d2, 0);
      d3 = EIGEN[l];
    }//

    if ((d3-EF)*(E1-EF)<0){
      k2[0] = d0;    k2[1] = d1;    k2[2] = d2;    E2 = d3;
    }else if((d3-EF)*(E2-EF)<0){
      k1[0] = d0;    k1[1] = d1;    k1[2] = d2;    E1 = d3;
    }
    if(fabs(d3-EF)<=1.0e-5){  break;  }
  }//i
  k3[0] = d0;    k3[1] = d1;    k3[2] = d2;    E3[0] = d3;

}

/* Added by N. Yamaguchi ***/
#endif
double scaleVector(double *v, double *v1, double *v2){
  return (*v-*v1)/(*v2-*v1);
}
void vectorize(double scale, double *v1, double *v2, double *v, int n){
  int i;
  for (i=0; i<n; i++){
    v[i]=(1.0-scale)*v1[i]+scale*v2[i];
  }
}
void func_Brent(double *k1, double E1, double *k2, double E2, double *k3, double *E3, double EF, int l, int loopMAX){
  double a=0.0;
  double b=1.0;
  if (fabs(E1-EF)<fabs(E2-EF)){
    double tmp=a;
    a=b;
    b=tmp;
    tmp=E1;
    E1=E2;
    E2=tmp;
  }
  double c=a;
  double E0=E1;
  int m=1;
  double delta=1.0e-5;
  double d;
  double E;
  if (loopMAX<1){
    loopMAX=1;
  }
  int i;
  for (i=0; i<loopMAX; i++){
    double s=E1!=E0 && E2!=E0 ? a*(E2-EF)*(E0-EF)/((E1-E2)*(E1-E0))+b*(E1-EF)*(E0-EF)/((E2-E1)*(E2-E0))+c*(E1-EF)*(E2-EF)/((E0-E1)*(E0-E2)) : b-(E2-EF)*(b-a)/(E2-E1);
    if (!switch_Eigen_Newton){
      vectorize(s, k1, k2, k3, 3);
      *E3=EF;
      break;
    }
    double t=0.25*(3.0*a+b);
    double u;
    if (s>=t && s>=b || s<=t && s<=b || m && fabs(s-b)>=0.5*(u=fabs(b-c)) || !m && fabs(s-b)>=0.5*(u=fabs(c-d)) || u<delta){
      s=0.5*(a+b);
      m=1;
    } else {
      m=0;
    }
    vectorize(s, k1, k2, k3, 3);
    EigenValue_Problem(k3[0], k3[1], k3[2], 0);
    *E3=EIGEN[l];
    d=c;
    E=E0;
    c=b;
    E0=E2;
    if (E1>EF && *E3<EF || E1<EF && *E3>EF){
      b=s;
      E2=*E3;
    } else {
      a=s;
      E1=*E3;
    }
    if (fabs(E1-EF)<fabs(E2-EF)){
      double tmp=a;
      a=b;
      b=tmp;
      tmp=E1;
      E1=E2;
      E2=tmp;
    }
    if (fabs(*E3-EF)<=delta){
      break;
    }
  }
}
/* ***/


int func_periodicity(int *tail_Cc, int count_Pair3M, int *trace_Cc, int **k_index_hitD, int **pair_hit3M, int k1_domain, int k2_domain){
  int i,j,k;
  int hit_Periodicity = 0;

  for(i=0; i<count_Pair3M; i++){
    for(j=0; j<2; j++){
      if (      (trace_Cc[i] == 0)             &&
	  ((    (k_index_hitD[0][tail_Cc[1]]+1 == k_index_hitD[0][pair_hit3M[i][4*j+1]]  )
		&& (k_index_hitD[1][tail_Cc[1]]   == k_index_hitD[1][pair_hit3M[i][4*j+1]]  )
		&& (tail_Cc[0] == 2)              && (pair_hit3M[i][4*j+0] == 2)
		&& (tail_Cc[2] == k1_domain)      && (pair_hit3M[i][4*j+2] == 0)
		&& (tail_Cc[3] == pair_hit3M[i][4*j+3])                                      )
	   ||
	   (    (k_index_hitD[0][tail_Cc[1]]   == k_index_hitD[0][pair_hit3M[i][4*j+1]]+1)
		&& (k_index_hitD[1][tail_Cc[1]]   == k_index_hitD[1][pair_hit3M[i][4*j+1]]  )
		&& (tail_Cc[0] == 2)              && (pair_hit3M[i][4*j+0] == 2)
		&& (tail_Cc[2] == 0)              && (pair_hit3M[i][4*j+2] == k1_domain)
		&& (tail_Cc[3] == pair_hit3M[i][4*j+3])                                      )
	   ||
	   (   (k_index_hitD[0][tail_Cc[1]]    == k_index_hitD[0][pair_hit3M[i][4*j+1]]  )
	       && (k_index_hitD[1][tail_Cc[1]]+1 == k_index_hitD[1][pair_hit3M[i][4*j+1]]  )
	       && (tail_Cc[0] == 1)              && (pair_hit3M[i][4*j+0] == 1)
	       && (tail_Cc[3] == k2_domain)      && (pair_hit3M[i][4*j+3] == 0)
	       && (tail_Cc[2] == pair_hit3M[i][4*j+2])                                      )
	   ||
	   (    (k_index_hitD[0][tail_Cc[1]]   == k_index_hitD[0][pair_hit3M[i][4*j+1]]  )
		&& (k_index_hitD[1][tail_Cc[1]]   == k_index_hitD[1][pair_hit3M[i][4*j+1]]+1)
		&& (tail_Cc[0] == 1)              && (pair_hit3M[i][4*j+0] == 1)
		&& (tail_Cc[3] == 0)              && (pair_hit3M[i][4*j+3] == k2_domain)
		&& (tail_Cc[2] == pair_hit3M[i][4*j+2])                                      ))){
		  tail_Cc[0] = pair_hit3M[i][4*abs(j-1)+0];    tail_Cc[1] = pair_hit3M[i][4*abs(j-1)+1];
		  tail_Cc[2] = pair_hit3M[i][4*abs(j-1)+2];    tail_Cc[3] = pair_hit3M[i][4*abs(j-1)+3];
		  //        tail_Cc[4] = pair_hit3M[i][4*j+0];           tail_Cc[5] = pair_hit3M[i][4*j+1];
		  //        tail_Cc[6] = pair_hit3M[i][4*j+2];           tail_Cc[7] = pair_hit3M[i][4*j+3];
		  trace_Cc[i] = 1;
		  hit_Periodicity++;
		  break;
		}//if
    }//j
    if (hit_Periodicity > 0) break;
  }//i
  return hit_Periodicity;
};
