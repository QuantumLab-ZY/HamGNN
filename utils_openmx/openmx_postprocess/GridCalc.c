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
#include "GetOrbital.h"

/* Added by N. Yamaguchi ***/
#define PRINTFF(...)\
  do{\
    printf(__VA_ARGS__);\
    fflush(stdout);\
  }while(0)
/* ***/

/* Added by N. Yamaguchi ***/
#ifdef SIGMAEK
int GridCalc()
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
  double **k_xyz_Cc,*EIG_Cc, ***MulP_Cc;
  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2

  double TStime,TEtime, Stime,Etime, Time_EIG;    // Time measurement variable

  int fo_inp = 0;
  int fo_wf  = 0;
  int i_vec[20],i_vec2[20];                       // input variable
  char *s_vec[20];                                // input variable
  double r_vec[20];                               // input variable

  int data_Count;   
  int hit_Cc;
  int head_Cc[8], tail_Cc[8], *trace_Cc;
  int hit_Total[2];

  // ### Orbital Data    ###
  int TNO_MAX ,ClaOrb_MAX[2] ,**ClaOrb;              // Orbital  

  char OrbSym[5][2], **OrbName, **An2Spe;

  /* Disabled by N. Yamaguchi ***
     double ***OrbMulP_Cc, ***Orb2MulP_Cc;
   * ***/

  /* Added by N. Yamaguchi ***/
  double ****OrbMulP_Cc, ****Orb2MulP_Cc;
  /* ***/

  // ### k_height    ###
  int i_height, k3_height;
  double kRange_height;

  /* Added by N. Yamaguchi ***/
  int calcBandbyband, calcBandMax, calcBandMin;
  int *numK, *recvCount, *displs;
  double ***MulP_CcDivision;
  double ****OrbMulP_CcDivision;
  double ****Orb2MulP_CcDivision;
#ifndef DEBUG_SIGMAEK_OLDER_20181126
  mode=(int(*)())GridCalc;
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
  if (myrank ==0) printf("\n");fflush(stdout);

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  /* ***/

  sprintf(fname,"%s",argv[1]);

  if((fp = fopen(fname,"r")) != NULL){
    fo_inp = 1;
    if (myrank ==0) printf("open \"%s\" file... \n" ,fname);fflush(stdout);
    fclose(fp);
  }else{
    fo_inp = 0;
    if (myrank ==0) printf("Cannot open \"%s\" file.\n" ,fname);fflush(stdout);
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

    /* Disabled by N. Yamaguchi ***
       input_int("Eigen.Newton",&switch_Eigen_Newton,0);
       if (switch_Eigen_Newton != 0) switch_Eigen_Newton = 1;
     * ***/

    input_int("Calc.Type.3mesh",&plane_3mesh,1);

    i_vec2[0]=2;  i_vec2[1]=2;
    input_intv("k-plane.1stStep",2,i_vec,i_vec2);

    /* Disabled by N. Yamaguchi
     * k1_3mesh = i_vec[0];    k2_3mesh = i_vec[1];
     */

    /* Added by N. Yamaguchi ***/
    k1_3mesh = i_vec[0]-1;    k2_3mesh = i_vec[1]-1;
    /* ***/

    r_vec[0]=0.5; r_vec[1]=0.5;
    input_doublev("kRange.3mesh",2,kRange_3mesh,r_vec);

    input_int("k-plane.height",&k3_height,1);
    input_double("kRange.height", &kRange_height, 0.0);

    r_vec[0]=1.0; r_vec[1]=1.0; r_vec[2]=1.0;
    input_doublev("MulP.Vec.Scale",3, MulP_VecScale, r_vec);

    /* Disabled by N. Yamaguchi ***
    input_int("Trial.Newton",&trial_Newton,5);
    if (trial_Newton<2) trial_Newton = 2;
    * ***/

    /* Disabled by N. Yamaguchi ***
       input_int("Spin.Degenerate",&Spin_Dege,0);
       if((Spin_Dege<0) || (Spin_Dege>1)) Spin_Dege = 0;
     * ***/

    /* Added by N. Yamaguchi ***/
    input_logical("Spin.Degenerate", &Spin_Dege, 0);
    input_logical("Calc.Bandbyband", &calcBandbyband, 0);
    input_int("Calc.Band.Max", &calcBandMax, 0);
    input_int("Calc.Band.Min", &calcBandMin, 0);
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
#ifndef SIGMAEK
  /* ***/

  if (fo_inp == 1){
    if((fp = fopen(fname_wf,"r")) != NULL){
      fo_wf = 1;
      if (myrank ==0) printf("\nInput filename is \"%s\"  \n\n", fname_wf);fflush(stdout);
      fclose(fp);
    }else{
      fo_wf = 0;
      if (myrank ==0) printf("Cannot open *.scfout File. \"%s\" is not found.\n" ,fname_wf);fflush(stdout);
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
      PRINTFF("Error: \"GridCalc\" is available only for non-collinear cases.\n");
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
      printf("########### ORBITAL DATA ##################\n");fflush(stdout);
      //       for(i=1;i<=atomnum;i++) printf("%4d:%4d\n", i, Total_NumOrbs[i]);
      //       printf("  MAX:%4d\n",TNO_MAX);
      printf("ClaOrb_MAX[0]:%4d\n",ClaOrb_MAX[0]);fflush(stdout);
      printf("ClaOrb_MAX[1]:%4d\n",ClaOrb_MAX[1]);fflush(stdout);
      printf("Total Band (2*n):%4d\n",n2-4);fflush(stdout);
      printf("Central (%10.6lf %10.6lf %10.6lf)\n",k_CNT1[0],k_CNT1[1],k_CNT1[2]);fflush(stdout);
      printf("###########################################\n\n");fflush(stdout);
    }
    // #######################################################
    if (myrank == 0){

      /* Disabled by N. Yamaguchi ***
	 data_Count = 0;
       * ***/

      strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP");
      fp1= fopen(fname_MP,"w");
      fprintf(fp1,"                    \n");
      fclose(fp1);

      /* Added by N. Yamaguchi ***/
      for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	strcpy(fname_MP, fname_out);
	strcat(fname_MP, ".MulP_");
	strcat(fname_MP, OrbSym[i1]);
	fp1=fopen(fname_MP, "w");
	fputs("                    \n", fp1);
	fclose(fp1);
      }
      for (i1=1; i1<=ClaOrb_MAX[1]; i1++){
	strcpy(fname_MP, fname_out);
	strcat(fname_MP, ".MulP_");
	strcat(fname_MP, OrbName[i1]);
	fp1=fopen(fname_MP, "w");
	fputs("                    \n", fp1);
	fclose(fp1);
      }
      /* ***/

    }
    // ### MALLOC ARRAY ###################################### 
    // ### (MPI Calculation)    ###
    size_Tknum = (k1_3mesh+1)*(k2_3mesh+1)*(k3_height);
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
    for (i = 0; i < ((size_Tknum+1)*n2+1); i++) EIGEN_MP[i] = 0.0;

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
     * MulP_Cc = (double***)malloc(sizeof(double**)*4);
     * for (i=0; i<4; i++){
     * MulP_Cc[i] = (double**)malloc(sizeof(double*)*n2);
     * for (j=0; j<n2; j++){
     * MulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(size_Tknum+1)));
     * for (k=0; k<((atomnum+1)*(size_Tknum+1)); k++) MulP_Cc[i][j][k] = 0.0;
     * }
     * }
     */
    if (myrank == 0)  printf("########### EIGEN VALUE ###################\n");fflush(stdout);
    dtime(&Stime);

    // ### k-GRID & EIGEN(intersection) ######################
    for(i=0; i<=k1_3mesh; i++){
      if(k1_3mesh==1){ k1 = 0.0; }else{
	//        k1 = -kRange_3mesh[0] + 2.0*kRange_3mesh[0]*(2.0*(double)i+1.0)/(2.0*(double)k1_3mesh); \}
	k1 = -kRange_3mesh[0] + 2.0*kRange_3mesh[0]*(2.0*(double)i)/(2.0*(double)k1_3mesh); }
      for(j=0; j<=k2_3mesh; j++){
	if(k2_3mesh==1){ k2 = 0.0; }else{
	  //          k2 = -kRange_3mesh[1] + 2.0*kRange_3mesh[1]*(2.0*(double)j+1.0)/(2.0*(double)k2_3mesh); \}
	  k2 = -kRange_3mesh[1] + 2.0*kRange_3mesh[1]*(2.0*(double)j)/(2.0*(double)k2_3mesh); }
	for(k=0; k<k3_height; k++){
	  if(k3_height==1){ k3 = 0.0; }else{
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

    /* Added by N. Yamaguchi ***/
    numK=(int*)malloc(sizeof(int)*num_procs);
    for (j=0; j<num_procs; j++){
      numK[j]=E_knum[j]-S_knum[j]+1;
      if (numK[j]<=0){
	numK[j]=0;
      }
    }
    /* ***/

    // ### EIGEN_VALUE_PROBLEM ###############################

    /* Disabled by N. Yamaguchi
     * for (j = S_knum[myrank]; j <= E_knum[myrank]; j++){
     * EigenValue_Problem(k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], 1);
     * for(l=1; l<=2*n; l++){
     * EIGEN_MP[j*n2+l] = EIGEN[l];
     *
     * for (i=0; i < atomnum; i++){
     * for (j1=0; j1<4; j1++){
     * MulP_Cc[j1][l][j*(atomnum+1)+i] = 0.0;
     * for (i1=0; i1 < Total_NumOrbs[i+1]; i1++){
     * //              if(Nband[j2]==l){
     * //              Orb2MulP_Cc[j1][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+= Data_MulP[j1][l][i+1][i1];
     * MulP_Cc[j1][l][j*(atomnum+1)+i]+= Data_MulP[j1][l][i+1][i1];
     * //              }//if
     * }//for(i1)
     * }//for(j1)
     * }//for(i)
     * }//l
     * }//j
     */

    /* Added by N. Yamaguchi ***/
    for (j = S_knum[myrank]; j <= E_knum[myrank]; j++){
      EigenValue_Problem(k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], 0);
      for(l=1; l<=2*n; l++){
	EIGEN_MP[j*n2+l] = EIGEN[l];
      }//l
    }//j
    /* ***/

    // ### MPI part ##########################################
    for (i=0; i<num_procs; i++){

      /* Disabled by N. Yamaguchi
       * k = S_knum[i]*n2;
       * l = abs(E_knum[i]-S_knum[i]+1)*n2;
       * MPI_Barrier(MPI_COMM_WORLD);
       * MPI_Bcast(&EIGEN_MP[k], l, MPI_DOUBLE, i, MPI_COMM_WORLD);
       */

      /* Added by N. Yamaguchi ***/
      if (S_knum[i]>=0){
	k=S_knum[i]*n2;
	l=numK[i]*n2;
      } else {
	k=0;
	l=0;
      }
      MPI_Bcast(EIGEN_MP+k, l, MPI_DOUBLE, i, MPI_COMM_WORLD);
      /* ***/

    }

    /* Added by N. Yamaguchi ***/
    if (!calcBandbyband){
      /* ***/

      // ### SELECT CALC_BAND ###
      l_min = 0;  l_max = 0;
      /*
       * EF = (E_Range[1]+E_Range[0])/2;
       */
      l_cal = 0;
      for(l=1; l<=2*n; l++){
	E1 = EIGEN_MP[0+l];  E2 = EIGEN_MP[0+l];
	for (i = 0; i < size_Tknum; i++){
	  if (E1 > EIGEN_MP[i*n2+l])  E1 = EIGEN_MP[i*n2+l];  //min
	  if (E2 < EIGEN_MP[i*n2+l])  E2 = EIGEN_MP[i*n2+l];  //max
	}//i
	/*
	 * if ((E1-EF)*(E2-EF) <=0){
	 * if (l_cal == 0){
	 * l_cal = 1;
	 * l_min = l;  l_max = l;
	 * }else if(l_cal > 0){
	 * l_max = l;
	 * }
	 * }//if
	 */
	if ( ((E_Range[0]-E1)*(E_Range[1]-E1) <=0)
	    || ((E_Range[0]-E2)*(E_Range[1]-E2) <=0) ){
	  if (l_cal == 0){
	    l_cal = 1;
	    l_min = l;  l_max = l;
	  }else if(l_cal > 0){
	    l_max = l;
	  }
	}//if
      }//l
      if (l_cal > 0) l_cal = (l_max-l_min+1);

      /* Added by N. Yamaguchi ***/
      if (l_cal==0){
	PRINTFF("Error: No bands are found.\n");
#ifndef SIGMAEK
	MPI_Finalize();
#endif
	return 0;
      }
    }
    if (calcBandbyband) {
#if 0
      if (calcBandMax<l_min || calcBandMin>l_max) {
	PRINTFF("Error: No bands are found in the range of %d-%d.\n", l_min, l_max);
#ifndef SIGMAEK
	MPI_Finalize();
#endif
	return 0;
      } else {
#endif
	l_max=calcBandMax;
	l_min=calcBandMin;
	l_cal=l_max-l_min+1;
#if 0
      }
#endif
    }
    /* ***/

    if (myrank==0) printf("The number of BANDs %4d (%4d->%4d)\n", l_cal, l_min, l_max);fflush(stdout);

    /* Added by N. Yamaguchi ***/
    // ### (MulP Calculation)   ###
    Data_MulP = (double****)malloc(sizeof(double***)*4);
    for (i=0; i<4; i++){
      Data_MulP[i] = (double***)malloc(sizeof(double**)*l_cal);
      for (l=0; l<l_cal; l++){
	Data_MulP[i][l] = (double**)malloc(sizeof(double*)*(atomnum+1));
	for (j=0; j<=atomnum; j++){
	  Data_MulP[i][l][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
	  for (k=0; k<=TNO_MAX; k++) Data_MulP[i][l][j][k] = 0.0;
	}
      }
    }
    MulP_Cc = (double***)malloc(sizeof(double**)*4);
    for (i=0; i<4; i++){
      MulP_Cc[i] = (double**)malloc(sizeof(double*)*l_cal);
      if (myrank==0) {
	for (j=0; j<l_cal; j++){
	  MulP_Cc[i][j] = (double*)malloc(sizeof(double)*(atomnum+1)*(size_Tknum+1));
	  for (k=0; k<(atomnum+1)*(size_Tknum+1); k++) MulP_Cc[i][j][k] = 0.0;
	}
      }
    }
    MulP_CcDivision = (double***)malloc(sizeof(double**)*4);
    for (i=0; i<4; i++){
      MulP_CcDivision[i] = (double**)malloc(sizeof(double*)*l_cal);
      for (j=0; j<l_cal; j++){
	MulP_CcDivision[i][j] = (double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	for (k=0; k<(atomnum+1)*numK[myrank]; k++) MulP_CcDivision[i][j][k] = 0.0;
      }
    }
    OrbMulP_Cc=(double****)malloc(sizeof(double***)*4);
    for (i=0; i<4; i++){
      OrbMulP_Cc[i]=(double***)malloc(sizeof(double**)*(ClaOrb_MAX[0]+1));
      for (j=0; j<=ClaOrb_MAX[0]; j++){
	OrbMulP_Cc[i][j]=(double**)malloc(sizeof(double*)*l_cal);
	if (myrank==0) {
	  for (k=0; k<l_cal; k++){
	    OrbMulP_Cc[i][j][k]=(double*)malloc(sizeof(double)*(atomnum+1)*(size_Tknum+1));
	    for (l=0; l<(atomnum+1)*(size_Tknum+1); l++) OrbMulP_Cc[i][j][k][l]=0.0;
	  }
	}
      }
    }
    Orb2MulP_Cc=(double****)malloc(sizeof(double***)*4);
    for (i=0; i<4; i++){
      Orb2MulP_Cc[i]=(double***)malloc(sizeof(double**)*(ClaOrb_MAX[1]+1));
      for (j=0; j<=ClaOrb_MAX[1]; j++){
	Orb2MulP_Cc[i][j]=(double**)malloc(sizeof(double*)*l_cal);
	if (myrank==0) {
	  for (k=0; k<l_cal; k++){
	    Orb2MulP_Cc[i][j][k]=(double*)malloc(sizeof(double)*(atomnum+1)*(size_Tknum+1));
	    for (l=0; l<(atomnum+1)*(size_Tknum+1); l++) Orb2MulP_Cc[i][j][k][l]=0.0;
	  }
	}
      }
    }
    OrbMulP_CcDivision=(double****)malloc(sizeof(double***)*4);
    for (i=0; i<4; i++){
      OrbMulP_CcDivision[i]=(double***)malloc(sizeof(double**)*(ClaOrb_MAX[0]+1));
      for (j=0; j<=ClaOrb_MAX[0]; j++){
	OrbMulP_CcDivision[i][j]=(double**)malloc(sizeof(double*)*l_cal);
	for (k=0; k<l_cal; k++){
	  OrbMulP_CcDivision[i][j][k]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	  for (l=0; l<(atomnum+1)*numK[myrank]; l++) OrbMulP_CcDivision[i][j][k][l]=0.0;
	}
      }
    }
    Orb2MulP_CcDivision=(double****)malloc(sizeof(double***)*4);
    for (i=0; i<4; i++){
      Orb2MulP_CcDivision[i]=(double***)malloc(sizeof(double**)*(ClaOrb_MAX[1]+1));
      for (j=0; j<=ClaOrb_MAX[1]; j++){
	Orb2MulP_CcDivision[i][j]=(double**)malloc(sizeof(double*)*l_cal);
	for (k=0; k<l_cal; k++){
	  Orb2MulP_CcDivision[i][j][k]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	  for (l=0; l<(atomnum+1)*numK[myrank]; l++) Orb2MulP_CcDivision[i][j][k][l]=0.0;
	}
      }
    }
    /* ***/

    // ### EIGEN_VALUE_PROBLEM ###############################
    for (j = S_knum[myrank]; j <= E_knum[myrank]; j++){
      EigenValue_Problem(k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], 1);
      for(l=l_min; l<=l_max; l++){

	for (i=0; i < atomnum; i++){
	  for (j1=0; j1<4; j1++){

	    /* Disabled by N. Yamaguchi
	     * MulP_Cc[j1][l-l_min][j*(atomnum+1)+i] = 0.0;
	     */

	    for (i1=0; i1 < Total_NumOrbs[i+1]; i1++){

	      /* Disabled by N. Yamaguchi
	       * MulP_Cc[j1][l-l_min][j*(atomnum+1)+i]+= Data_MulP[j1][l-l_min][i+1][i1];
	       */

	      /* Added by N. Yamaguchi ***/
	      MulP_CcDivision[j1][l-l_min][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l-l_min][i+1][i1];
	      Orb2MulP_CcDivision[j1][ClaOrb[i+1][i1]][l-l_min][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l-l_min][i+1][i1];
	      if (ClaOrb[i+1][i1]==0){
		OrbMulP_CcDivision[j1][0][l-l_min][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l-l_min][i+1][i1];
	      } else if (ClaOrb[i+1][i1]>=1 && ClaOrb[i+1][i1]<=3){
		OrbMulP_CcDivision[j1][1][l-l_min][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l-l_min][i+1][i1];
	      } else if (ClaOrb[i+1][i1]>=4 && ClaOrb[i+1][i1]<=8){
		OrbMulP_CcDivision[j1][2][l-l_min][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l-l_min][i+1][i1];
	      } else if (ClaOrb[i+1][i1]>=9 && ClaOrb[i+1][i1]<=15){
		OrbMulP_CcDivision[j1][3][l-l_min][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l-l_min][i+1][i1];
	      }
	      /* ***/

	    }//for(i1)
	  }//for(j1)
	}//for(i)
      }//l
    }//j
    /* ***/

    // ### MPI part ##########################################

    /* Disabled by N. Yamaguchi
     * for (i=0; i<num_procs; i++){
     * k = S_knum[i]*(atomnum+1);
     * i2 = abs(E_knum[i]-S_knum[i]+1)*(atomnum+1);
     * for (j1=0; j1<4; j1++){
     * for(l=l_min;l<=l_max;l++){
     * MPI_Barrier(MPI_COMM_WORLD);
     * MPI_Bcast(&MulP_Cc[j1][l-l_min][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
     * }//for(l)
     * }//for(j1)
     * }//i
     */

    /* Added by N. Yamaguchi ***/
    recvCount=(int*)malloc(sizeof(int)*num_procs);
    displs=(int*)malloc(sizeof(int)*num_procs);
    i2=(atomnum+1)*numK[myrank];
    for (j=0; j<num_procs; j++){
      recvCount[j]=(atomnum+1)*numK[j];
      displs[j]=S_knum[j]*(atomnum+1);
    }
    for (j1=0; j1<4; j1++){
      for(l=l_min;l<=l_max;l++){
	MPI_Gatherv(MulP_CcDivision[j1][l-l_min], i2, MPI_DOUBLE, MulP_Cc[j1][l-l_min], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
      for (i=0; i<=ClaOrb_MAX[0]; i++){
	for(l=l_min;l<=l_max;l++){
	  MPI_Gatherv(OrbMulP_CcDivision[j1][i][l-l_min], i2, MPI_DOUBLE, OrbMulP_Cc[j1][i][l-l_min], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
      }
      for (i=0; i<=ClaOrb_MAX[1]; i++){
	for(l=l_min;l<=l_max;l++){
	  MPI_Gatherv(Orb2MulP_CcDivision[j1][i][l-l_min], i2, MPI_DOUBLE, Orb2MulP_Cc[j1][i][l-l_min], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
      }
    }
    /* ***/

    MPI_Barrier(MPI_COMM_WORLD);
    dtime(&Etime);
    Time_EIG = Etime-Stime;

    //kotaka
    if (myrank == 0){
      // ###################################################

      /* Added by N. Yamaguchi ***/
      data_Count=0;
      /* ***/

      strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP");
      fp1= fopen(fname_MP,"a");
      for(l=l_min;l<=l_max;l++){
	for(j=0; j<(k3_height)*(k1_3mesh+1)*(k2_3mesh+1); j++){
	  data_Count++;
	  Print_kxyzEig(Pdata_s, k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], l, EIGEN_MP[j*n2+l]);
	  fprintf(fp1, "%s", Pdata_s);
	  for (i=0; i<atomnum; i++){
	    fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[0][l-l_min][j*(atomnum+1)+i], MulP_Cc[1][l-l_min][j*(atomnum+1)+i]);
	    fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[2][l-l_min][j*(atomnum+1)+i], MulP_Cc[3][l-l_min][j*(atomnum+1)+i]);
	  } fprintf(fp1,"\n");
	}//j
      }//l
      fclose(fp1);

      /* Added by N. Yamaguchi ***/
      for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	strcpy(fname_MP, fname_out);
	strcat(fname_MP, ".MulP_");
	strcat(fname_MP, OrbSym[i1]);
	fp1=fopen(fname_MP, "a");
	for(l=l_min; l<=l_max; l++){
	  for(j=0; j<k3_height*(k1_3mesh+1)*(k2_3mesh+1); j++){
	    Print_kxyzEig(Pdata_s, k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], l, EIGEN_MP[j*n2+l]);
	    fprintf(fp1, "%s", Pdata_s);
	    for (i=0; i<atomnum; i++){
	      fprintf(fp1, "%10.6lf %10.6lf ", OrbMulP_Cc[0][i1][l-l_min][j*(atomnum+1)+i], OrbMulP_Cc[1][i1][l-l_min][j*(atomnum+1)+i]);
	      fprintf(fp1, "%10.6lf %10.6lf ", OrbMulP_Cc[2][i1][l-l_min][j*(atomnum+1)+i], OrbMulP_Cc[3][i1][l-l_min][j*(atomnum+1)+i]);
	    }
	    fprintf(fp1, "\n");
	  }
	}
	fclose(fp1);
      }
      for (i1=1; i1<=ClaOrb_MAX[1]; i1++){
	strcpy(fname_MP, fname_out);
	strcat(fname_MP, ".MulP_");
	strcat(fname_MP, OrbName[i1]);
	fp1=fopen(fname_MP, "a");
	for(l=l_min; l<=l_max; l++){
	  for(j=0; j<k3_height*(k1_3mesh+1)*(k2_3mesh+1); j++){
	    Print_kxyzEig(Pdata_s, k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], l, EIGEN_MP[j*n2+l]);
	    fprintf(fp1, "%s", Pdata_s);
	    for (i=0; i<atomnum; i++){
	      fprintf(fp1, "%10.6lf %10.6lf ", Orb2MulP_Cc[0][i1][l-l_min][j*(atomnum+1)+i], Orb2MulP_Cc[1][i1][l-l_min][j*(atomnum+1)+i]);
	      fprintf(fp1, "%10.6lf %10.6lf ", Orb2MulP_Cc[2][i1][l-l_min][j*(atomnum+1)+i], Orb2MulP_Cc[3][i1][l-l_min][j*(atomnum+1)+i]);
	    }
	    fprintf(fp1, "\n");
	  }
	}
	fclose(fp1);
      }
      /* ***/

      // ###################################################
      for(l=l_min;l<=l_max;l++){
	strcpy(fname_Spin,fname_out);
	name_Nband(fname_Spin,".Pxyz_",l);
	fp1 = fopen(fname_Spin,"w");
	for(j=0; j<(k3_height)*(k1_3mesh+1)*(k2_3mesh+1); j++){
	  for(i=0; i<3; i++){
	    fprintf(fp1,"%10.6lf ",rtv[1][i+1]*k_xyz[0][j]+ rtv[2][i+1]*k_xyz[1][j]+ rtv[3][i+1]*k_xyz[2][j]);
	  }
	  Re11 = 0;      Re22 = 0;      Re12 = 0;      Im12 = 0;
	  for (i=0; i<atomnum; i++){
	    Re11+= MulP_Cc[0][l-l_min][j*(atomnum+1)+i];        Re22+= MulP_Cc[1][l-l_min][j*(atomnum+1)+i];
	    Re12+= MulP_Cc[2][l-l_min][j*(atomnum+1)+i];        Im12+= MulP_Cc[3][l-l_min][j*(atomnum+1)+i];
	  } EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndw, Ntheta, Nphi );
	  fprintf(fp1,"%10.6lf  ",MulP_VecScale[0]* (Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]));
	  fprintf(fp1,"%10.6lf  ",MulP_VecScale[1]* (Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]));
	  fprintf(fp1,"%10.6lf\n",MulP_VecScale[2]* (Nup[0] -Ndw[0]) *cos(Ntheta[0]));
	}//j      
	fclose(fp1);
      }//l

      /* Added by N. Yamaguchi ***/
      for(l=l_min;l<=l_max;l++){
	char fname_EV[256];
	strcpy(fname_EV,fname_out);
	name_Nband(fname_EV,".EigenMap_",l);
	fp1 = fopen(fname_EV,"w");
	int count=0;
	for(k=0; k<=k1_3mesh; k++){
	  for(j=0; j<=k2_3mesh; j++){
	    for(i=0; i<3; i++){
	      fprintf(fp1,"%10.6lf ", rtv[1][i+1]*k_xyz[0][count]+ rtv[2][i+1]*k_xyz[1][count]+ rtv[3][i+1]*k_xyz[2][count]);
	    }
	    fprintf(fp1, "%10.6lf\n", EIGEN_MP[count*n2+l]);
	    ++count;
	  }
	  fputs("\n", fp1);
	}
	fclose(fp1);
	char fname_gp[256];
	strcpy(fname_gp,fname_out);
	name_Nband(fname_gp,".plotexample_",l);
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
	fputs("reduction=1\n", fp1);
	fputs("#reduction=20\n", fp1);
	fputs("linewidth=3\n", fp1);
	fputs("set size ratio -1\n", fp1);
	fputs("set encoding iso\n", fp1);
	fputs("set xlabel 'k_x (/\\305)'\n", fp1);
	fputs("set ylabel 'k_y (/\\305)'\n", fp1);
	fputs("set cblabel 'eV'\n", fp1);
	fputs("set pm3d map\n", fp1);
	fputs("set palette defined (-2 'blue', -1 'white', 0 'red')\n", fp1);
	fputs("set pm3d interpolate 5, 5\n", fp1);
	fputs("#set xtics tics\n", fp1);
	fputs("#set ytics tics\n", fp1);
	fputs("set label 1 center at first xc, yc '+' front\n", fp1);
	fputs("#set label 2 center at first xc, yc-0.1 '{/Symbol G}' front\n", fp1);
	if (plane_3mesh==1){
	  fprintf(fp1, "splot '%s.EigenMap_%d' using ($1*Bohr2Ang):($2*Bohr2Ang):4 notitle\n", fname_out, l);
	  fprintf(fp1, "replot '%s.Pxyz_%d' every reduction using ($1*Bohr2Ang):($2*Bohr2Ang):($3*Bohr2Ang):4:5:6 with vectors notitle linetype -1 linewidth linewidth\n", fname_out, l);
	} else if (plane_3mesh==2){
	  fprintf(fp1, "splot '%s.EigenMap_%d' using ($2*Bohr2Ang):($3*Bohr2Ang):4 notitle\n", fname_out, l);
	  fprintf(fp1, "replot '%s.Pxyz_%d' every reduction using ($2*Bohr2Ang):($3*Bohr2Ang):($1*Bohr2Ang):5:6:4 with vectors notitle linetype -1 linewidth linewidth\n", fname_out, l);
	} else if (plane_3mesh==3){
	  fprintf(fp1, "splot '%s.EigenMap_%d' using ($3*Bohr2Ang):($1*Bohr2Ang):4 notitle\n", fname_out, l);
	  fprintf(fp1, "replot '%s.Pxyz_%d' every reduction using ($3*Bohr2Ang):($1*Bohr2Ang):($2*Bohr2Ang):6:4:5 with vectors notitle linetype -1 linewidth linewidth\n", fname_out, l);
	}
	fputs("pause -1\n", fp1);
	fclose(fp1);
      }
      /* ***/

      // ###################################################
    }//if(myrank)
    // ### MALLOC FREE #####################################

    /* Disabled by N. Yamaguchi
     * for (i=0; i<4; i++){
     * for (j=0; j<l_cal; j++){
     * free(MulP_Cc[i][j]);
     * } free(MulP_Cc[i]);
     * } free(MulP_Cc);
     */

    /* Added by N. yamaguchi ***/
    for (i=0; i<4; i++){
      if (myrank==0) {
	for (j=0; j<l_cal; j++){
	  free(MulP_Cc[i][j]);
	}
      }
      free(MulP_Cc[i]);
    } free(MulP_Cc);
    free(numK);
    free(recvCount);
    free(displs);
    for (i=0; i<4; i++){
      for (j=0; j<l_cal; j++){
	free(MulP_CcDivision[i][j]);
      } free(MulP_CcDivision[i]);
    } free(MulP_CcDivision);
    for (i=0; i<4; i++){
      for (j=0; j<=ClaOrb_MAX[0]; j++){
	for (k=0; k<l_cal; k++){
	  free(OrbMulP_CcDivision[i][j][k]);
	} free(OrbMulP_CcDivision[i][j]);
      } free(OrbMulP_CcDivision[i]);
    } free(OrbMulP_CcDivision);
    for (i=0; i<4; i++){
      for (j=0; j<=ClaOrb_MAX[1]; j++){
	for (k=0; k<l_cal; k++){
	  free(Orb2MulP_CcDivision[i][j][k]);
	} free(Orb2MulP_CcDivision[i][j]);
      } free(Orb2MulP_CcDivision[i]);
    } free(Orb2MulP_CcDivision);
    /* ***/

    if (myrank == 0){
      printf("Total MulP data:%4d\n" , data_Count);fflush(stdout);  //hit_Total[0]);
      printf("###########################################\n\n");fflush(stdout);
      //### atomnum & data_size ###
      strcpy(fname_MP,fname_out);     strcat(fname_MP,".AtomMulP");
      fp1= fopen(fname_MP,"r+");
      fseek(fp1, 0L, SEEK_SET);
      fprintf(fp1,"%6d %4d", data_Count, atomnum);
      fclose(fp1);
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

    // ### (MulP Calculation)   ###
    for (i=0; i<=atomnum; i++){
      free(An2Spe[i]);
    } free(An2Spe);
    for (i=0; i<=atomnum; i++){
      free(ClaOrb[i]);
    } free(ClaOrb);
    for (i=0; i<4; i++){
      for (l=0; l<l_cal; l++){
	for (j=0; j<=atomnum; j++){
	  free(Data_MulP[i][l][j]);
	} free(Data_MulP[i][l]);
      } free(Data_MulP[i]);
    } free(Data_MulP);

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
    printf("############ CALC TIME ####################\n");fflush(stdout);
    dtime(&TEtime);
    printf("  Total Calculation Time:%10.6lf (s)\n",TEtime-TStime);fflush(stdout);
    if ((fo_inp == 1) && (fo_wf == 1)){
      printf("        Eigen Value Calc:%10.6lf (s)\n",Time_EIG);fflush(stdout);
    }//if(fo_wf)
    printf("###########################################\n");fflush(stdout);
  }

  // ### MPI_Finalize ########################################

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  MPI_Finalize();
#endif
  /* ***/

  return 0;
}






