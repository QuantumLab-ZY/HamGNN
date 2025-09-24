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

#ifdef SIGMAEK
int MulPOnly()
#else
  /* ***/

int main(int argc, char *argv[])

  /* Added by N. Yamaguchi ***/
#endif
  /* ***/

{

  FILE *fp, *fp1, *fp2;
  int i,j,k,l,m,n,n2, i1,i2, j1,j2, l2;           // loop variable

  /* Disabled by N. Yamaguchi ***
     int l_max, l_min, l_cal;                        // band index
   * ***/

  int *S_knum, *E_knum, T_knum, size_Tknum;       // MPI variable (k-point divide)
  int namelen, num_procs, myrank;                 // MPI_variable
  int Data_size, *NBand_Cc;

  char c;
  char fname_kpoint[256], fname_MP[256], fname_Spin[256];
  char Pdata_s[256];
  //  char processor_name[MPI_MAX_PROCESSOR_NAME];

  double k1, k2, k3;                              // k-point variable
  double d0, d1, d2, d3;
  double E1, E2, EF;                              // Eigen value
  double **k_xyz_Cc,*EIGEN_MP, **MulP_Cc;
  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2

  double TStime, TEtime, Stime, Etime;            // Time variable

  int fo_inp = 0;
  int fo_wf  = 0;
  int i_vec[20],i_vec2[20];                       // input variable
  char *s_vec[20];                                // input variable
  double r_vec[20];                               // input variable

  // ### Orbital Data    ###
  int TNO_MAX ,ClaOrb_MAX[2] ,**ClaOrb;              // Orbital
  //  char
  char OrbSym[5][2], **OrbName, **An2Spe;
  double ***OrbMulP_Cc, ***Orb2MulP_Cc;

  /* Added by N. Yamaguchi ***/
  int sizeMulP= Spin_Dege ? 8 : 4;
  int l1;
  int *numK, *recvCount, *displs;
  double **MulP_CcDivision;
  double ***OrbMulP_CcDivision;
  double ***Orb2MulP_CcDivision;
#ifndef DEBUG_SIGMAEK_OLDER_20181126
  mode=(int(*)())MulPOnly;
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
  if (myrank ==0) printf("\n");

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  /* ***/

  sprintf(fname,"%s",argv[1]);

  if((fp = fopen(fname,"r")) != NULL){
    fo_inp = 1;
    if (myrank ==0) printf("open \"%s\" file... \n" ,fname);
    fclose(fp);
  }else{
    fo_inp = 0;
    if (myrank ==0) printf("Cannot open \"%s\" file.\n" ,fname);
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

    input_string("Filename.kpointdata",fname_kpoint,"default");

    r_vec[0]=1.0; r_vec[1]=1.0; r_vec[2]=1.0;
    input_doublev("MulP.Vec.Scale",3, MulP_VecScale, r_vec);

    /* Disabled by N. Yamaguchi ***
       input_int("Spin.Degenerate",&Spin_Dege,0);
       if((Spin_Dege<0) || (Spin_Dege>1)) Spin_Dege = 0;
     * ***/

    /* Added by N. Yamaguchi ***/
    input_logical("Spin.Degenerate", &Spin_Dege, 0);
    l_cal=Spin_Dege+1;
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
      if (myrank ==0) printf("\nInput filename is \"%s\"  \n\n", fname_wf);
      fclose(fp);
    }else{
      fo_wf = 0;
      if (myrank ==0) printf("Cannot open *.scfout File. \"%s\" is not found.\n" ,fname_wf);
    }
  }//if(fo_inp)

  /* Added by N. Yamaguchi ***/
#else
  fo_wf=1;
#endif
  /* ***/

  if ((fo_inp == 1) && (fo_wf == 1)){
    // ### Get Calculation Data ##############################
    // ### wave functions　###

    /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
    /* ***/

    if (myrank == 0){      read_scfout(fname_wf, 1);
    }else {                read_scfout(fname_wf, 0);    }

    /* Added by N. Yamaguchi ***/
#endif
    if (SpinP_switch!=3){
      PRINTFF("Error: \"MulPOnly\" is available only for non-collinear cases.\n");
#ifndef SIGMAEK
      MPI_Finalize();
#endif
      return 0;
    }
    /* ***/

    S_knum = (int*)malloc(sizeof(int)*num_procs);
    E_knum = (int*)malloc(sizeof(int)*num_procs);

    // ### Total Num Orbs  ###
    TNO_MAX = 0;
    for(i=1;i<=atomnum;i++){  if(TNO_MAX < Total_NumOrbs[i]) TNO_MAX = Total_NumOrbs[i];   }
    // ### Classify Orbs   ###
    An2Spe = (char**)malloc(sizeof(char*)*(atomnum+1));
    for (i=0; i<=atomnum; i++){      An2Spe[i] = (char*)malloc(sizeof(char)*(asize10));    }
    ClaOrb = (int**)malloc(sizeof(int*)*(atomnum+1));
    for (i=0; i<=atomnum; i++){
      ClaOrb[i] = (int*)malloc(sizeof(int)*(TNO_MAX+1));
      for (j=0; j<=TNO_MAX; j++) ClaOrb[i][j]=0;
    } 

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0){       Classify_OrbNum(ClaOrb, An2Spe, 1);
    }else{                  Classify_OrbNum(ClaOrb, An2Spe, 0);    }
    ClaOrb_MAX[1] = 0;
    for(i=1;i<=atomnum;i++){
      for (j=0; j<=TNO_MAX; j++){  if(ClaOrb_MAX[1] < ClaOrb[i][j]) ClaOrb_MAX[1] = ClaOrb[i][j];  }
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
    EIGEN = (double*)malloc(sizeof(double)*n2);
    for (i = 0; i < n2; i++) EIGEN[i] = 0.0;

    if (myrank == 0){
      printf("########### ORBITAL DATA ##################\n");
      //       for(i=1;i<=atomnum;i++) printf("%4d:%4d\n", i, Total_NumOrbs[i]);
      //       printf("  MAX:%4d\n",TNO_MAX);
      printf("ClaOrb_MAX[0]:%4d\n",ClaOrb_MAX[0]);
      printf("ClaOrb_MAX[1]:%4d\n",ClaOrb_MAX[1]);
      printf("Total Band (2*n):%4d\n",n2-4);
      printf("###########################################\n\n");
    }
    // ### (MulP Calculation)   ###
    Data_MulP = (double****)malloc(sizeof(double***)*4);
    for (i=0; i<4; i++){

      /* Disabled by N. Yamaguchi ***
	 Data_MulP[i] = (double***)malloc(sizeof(double**)*n2);
       * ***/
      /* Added by N. Yamaguchi ***/
      Data_MulP[i] = (double***)malloc(sizeof(double**)*l_cal);
      /* ***/

      /* Added by N. Yamaguchi ***/
#if 0
      /* ***/

      for (l=0; l<n2; l++){

	/* Added by N. Yamaguchi ***/
#endif
	for (l=0; l<l_cal; l++){
	  /* ***/

	  Data_MulP[i][l] = (double**)malloc(sizeof(double*)*(atomnum+1));
	  for (j=0; j<=atomnum; j++){
	    Data_MulP[i][l][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
	    for (k=0; k<=TNO_MAX; k++) Data_MulP[i][l][j][k] = 0.0;
	  }

	  /* Added by N. Yamaguchi ***/
	}
#if 0
	/* ***/

      }

      /* Added by N. Yamaguchi ***/
#endif
      /* ***/

    }

    // #######################################################
    if((fp = fopen(fname_kpoint,"r")) != NULL){
      fscanf(fp,"%d", &Data_size);

      /* Disabled by N. Yamaguchi ***
	 fscanf(fp,"%d", &i);
       * ***/

      k_xyz_Cc = (double**)malloc(sizeof(double*)*3);
      for(i=0; i<3; i++){
	k_xyz_Cc[i] = (double*)malloc(sizeof(double)*(Data_size+1));
      } NBand_Cc = (int*)malloc(sizeof(int)*(Data_size+1));
      for(j=0; j<Data_size; j++){
	fscanf(fp,"%lf",&k1);        fscanf(fp,"%lf",&k2);        fscanf(fp,"%lf",&k3);
	k_xyz_Cc[0][j] = ( tv[1][1]*k1 +tv[1][2]*k2 +tv[1][3]*k3 )/2/PI;
	k_xyz_Cc[1][j] = ( tv[2][1]*k1 +tv[2][2]*k2 +tv[2][3]*k3 )/2/PI;
	k_xyz_Cc[2][j] = ( tv[3][1]*k1 +tv[3][2]*k2 +tv[3][3]*k3 )/2/PI;
	fscanf(fp,"%d",&NBand_Cc[j]);
	//        if (myrank==0) printf("%10.6lf  %10.6lf  %10.6lf  %4d\n", k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], NBand_Cc[j]);
      } fclose(fp);
      // #######################################################
      EIGEN_MP = (double*)malloc(sizeof(double)*((Data_size+1)*n2));
      for (i = 0; i < ((Data_size+1)*n2); i++) EIGEN_MP[i] = 0.0;
      //    MULP malloc

      /* Disabled by N. Yamaguchi ***
	 MulP_Cc = (double**)malloc(sizeof(double*)*8);
	 for (i=0; i<8; i++){
	 MulP_Cc[i] = (double*)malloc(sizeof(double)*((atomnum+1)*(Data_size+1)));
	 for (k=0; k<((atomnum+1)*(Data_size+1)); k++) MulP_Cc[i][k] = 0.0;
	 }
	 OrbMulP_Cc = (double***)malloc(sizeof(double**)*8);
	 for (i=0; i<8; i++){
	 OrbMulP_Cc[i] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	 for (j=0; j<=ClaOrb_MAX[0]; j++){
	 OrbMulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(Data_size+1)));
	 for (k=0; k<((atomnum+1)*(Data_size+1)); k++) OrbMulP_Cc[i][j][k] = 0.0;
	 }//j
	 }//i
	 Orb2MulP_Cc = (double***)malloc(sizeof(double**)*8);
	 for (i=0; i<8; i++){
	 Orb2MulP_Cc[i] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	 for (j=0; j<=ClaOrb_MAX[1]; j++){
	 Orb2MulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(Data_size+1)));
	 for (k=0; k<((atomnum+1)*(Data_size+1)); k++) Orb2MulP_Cc[i][j][k] = 0.0;
	 }//j
	 }//i
       * ***/

      /* Added by N. Yamaguchi ***/
      MulP_Cc=(double**)malloc(sizeof(double*)*sizeMulP);
      for (i=0; i<sizeMulP; i++){
	if (myrank==0){
	  MulP_Cc[i]=(double*)malloc(sizeof(double)*(atomnum+1)*(Data_size+1));
	  for (k=0; k<(atomnum+1)*(Data_size+1); k++){
	    MulP_Cc[i][k]=0.0;
	  }
	}
      }
      OrbMulP_Cc=(double***)malloc(sizeof(double**)*sizeMulP);
      for (i=0; i<sizeMulP; i++){
	OrbMulP_Cc[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	for (j=0; j<=ClaOrb_MAX[0]; j++){
	  if (myrank==0){
	    OrbMulP_Cc[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*(Data_size+1));
	    for (k=0; k<(atomnum+1)*(Data_size+1); k++){
	      OrbMulP_Cc[i][j][k]=0.0;
	    }
	  }
	}
      }
      Orb2MulP_Cc=(double***)malloc(sizeof(double**)*sizeMulP);
      for (i=0; i<sizeMulP; i++){
	Orb2MulP_Cc[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	for (j=0; j<=ClaOrb_MAX[1]; j++){
	  if (myrank==0){
	    Orb2MulP_Cc[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*(Data_size+1));
	    for (k=0; k<((atomnum+1)*(Data_size+1)); k++){
	      Orb2MulP_Cc[i][j][k]=0.0;
	    }
	  }
	}
      }
      /* ***/

      // ### MulP_Calculation ################################
      T_knum = Data_size;  //
      // Division CALC_PART
      for(i=0; i<num_procs; i++){
	if (T_knum <= i){                   S_knum[i] = -10;   E_knum[i] = -100;

	  /* Added by N. Yamaguchi ***/
#if 0
	  /* ***/

	} else if (T_knum < num_procs) {   S_knum[i] = i;     E_knum[i] = i;

	  /* Added by N. Yamaguchi ***/
#endif
	} else if (T_knum<=num_procs){
	  S_knum[i]=i;     E_knum[i]=i;
	  /* ***/

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
      MulP_CcDivision=(double**)malloc(sizeof(double*)*sizeMulP);
      for (i=0; i<sizeMulP; i++){
	MulP_CcDivision[i]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	for (j=0; j<(atomnum+1)*numK[myrank]; j++) MulP_CcDivision[i][j]=0.0;
      }
      OrbMulP_CcDivision=(double***)malloc(sizeof(double**)*sizeMulP);
      for (i=0; i<sizeMulP; i++){
	OrbMulP_CcDivision[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	for (j=0; j<=ClaOrb_MAX[0]; j++){
	  OrbMulP_CcDivision[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	  for (k=0; k<(atomnum+1)*numK[myrank]; k++) OrbMulP_CcDivision[i][j][k]=0.0;
	}
      }
      Orb2MulP_CcDivision=(double***)malloc(sizeof(double**)*sizeMulP);
      for (i=0; i<sizeMulP; i++){
	Orb2MulP_CcDivision[i]=(double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	for (j=0; j<=ClaOrb_MAX[1]; j++){
	  Orb2MulP_CcDivision[i][j]=(double*)malloc(sizeof(double)*(atomnum+1)*numK[myrank]);
	  for (k=0; k<(atomnum+1)*numK[myrank]; k++) Orb2MulP_CcDivision[i][j][k]=0.0;
	}
      }
      /* ***/

      // #######################################################
      for (j = S_knum[myrank]; j <= E_knum[myrank]; j++){

	//        k1 = (tv[1][1]*k_xyz_Cc[0][j] +tv[1][2]*k_xyz_Cc[1][j] +tv[1][3]*k_xyz_Cc[2][j] )/2/PI;
	//        k2 = (tv[2][1]*k_xyz_Cc[0][j] +tv[2][2]*k_xyz_Cc[1][j] +tv[2][3]*k_xyz_Cc[2][j] )/2/PI;
	//        k3 = (tv[3][1]*k_xyz_Cc[0][j] +tv[3][2]*k_xyz_Cc[1][j] +tv[3][3]*k_xyz_Cc[2][j] )/2/PI;
	//        EigenValue_Problem(k1 ,k2 ,k3 ,1);

	/* Added by N. Yamaguchi ***/
	if (!Spin_Dege){
	  l_min=l_max=NBand_Cc[j];
	} else {
	  l_min=NBand_Cc[j]%2 ? NBand_Cc[j] : NBand_Cc[j]-1;
	  l_max=NBand_Cc[j]%2 ? NBand_Cc[j]+1 : NBand_Cc[j];
	}
	/* ***/

	EigenValue_Problem( k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], 1);
	//      kotaka
	for(l=1; l<=2*n; l++){ EIGEN_MP[j*n2+l] = EIGEN[l]; }//l

	/* Disabled by N. Yamaguchi ***
	   if     (NBand_Cc[j]%2==1) {l2 = NBand_Cc[j] +1;}
	   else if(NBand_Cc[j]%2==0) {l2 = NBand_Cc[j] -1;}
	   else {}
	 * ***/

	/* Added by N. Yamaguchi ***/
	if (!Spin_Dege){
	  l1=0;
	} else {
	  l1=NBand_Cc[j]%2 ? 0 : 1;
	  l2=NBand_Cc[j]%2 ? 1 : 0;
	}

	for (i=0; i < atomnum; i++){
	  for (j1=0; j1<4; j1++){

	    /* Disabled by N. Yamaguchi ***
	       MulP_Cc[j1][j*(atomnum+1)+i] = 0.0;            MulP_Cc[j1+4][j*(atomnum+1)+i] = 0.0;
	     * ***/

	    for (i1=0; i1 < Total_NumOrbs[i+1]; i1++){

	      /* Disabled by N. Yamaguchi ***
		 Orb2MulP_Cc[j1][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+= Data_MulP[j1][NBand_Cc[j]][i+1][i1];
		 MulP_Cc[j1][j*(atomnum+1)+i]+= Data_MulP[j1][NBand_Cc[j]][i+1][i1];
	       * ***/

	      /* Added by N. Yamaguchi ***/
	      Orb2MulP_CcDivision[j1][ClaOrb[i+1][i1]][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l1][i+1][i1];
	      MulP_CcDivision[j1][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l1][i+1][i1];
	      if (ClaOrb[i+1][i1]==0){
		OrbMulP_CcDivision[j1][0][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l1][i+1][i1];
	      } else if (ClaOrb[i+1][i1]>=1 && ClaOrb[i+1][i1]<=3){
		OrbMulP_CcDivision[j1][1][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l1][i+1][i1];
	      } else if (ClaOrb[i+1][i1]>=4 && ClaOrb[i+1][i1]<=8){
		OrbMulP_CcDivision[j1][2][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l1][i+1][i1];
	      } else if (ClaOrb[i+1][i1]>=9 && ClaOrb[i+1][i1]<=15){
		OrbMulP_CcDivision[j1][3][(j-S_knum[myrank])*(atomnum+1)+i]+=Data_MulP[j1][l1][i+1][i1];
	      }
	      /* ***/

	      /* Added by N. Yamaguchi ***/
	      if (Spin_Dege){
		/* ***/

		//Spin_Dege

		/* Disabled by N. Yamaguchi ***
		   Orb2MulP_Cc[j1+4][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+= Data_MulP[j1][l2][i+1][i1];
		   MulP_Cc[j1+4][j*(atomnum+1)+i]+= Data_MulP[j1][l2][i+1][i1];
		 * ***/

		/* Added by N. Yamaguchi ***/
		Orb2MulP_Cc[j1+4][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+=Data_MulP[j1][l2][i+1][i1];
		MulP_Cc[j1+4][j*(atomnum+1)+i]+=Data_MulP[j1][l2][i+1][i1];
		/* ***/

		/* Added by N. Yamaguchi ***/
	      }
	      /* ***/

	    }//for(i1)
	  }//for(j1)
	}//for(i)
      }//for(j)
      // ### MPI part ##########################################

      /* Disabled by N. Yamaguchi ***
	 for (i=0; i<num_procs; i++){
	 k = S_knum[i]*n2;
	 l = abs(E_knum[i]-S_knum[i]+1)*n2;
	 MPI_Barrier(MPI_COMM_WORLD);
	 MPI_Bcast(&EIGEN_MP[k], l, MPI_DOUBLE, i, MPI_COMM_WORLD);
	 }
	 MPI_Barrier(MPI_COMM_WORLD);
	 for (i=0; i<num_procs; i++){
	 k = S_knum[i]*(atomnum+1);
	 i2 = abs(E_knum[i]-S_knum[i]+1)*(atomnum+1);
	 MPI_Barrier(MPI_COMM_WORLD);
	 for (j1=0; j1<8; j1++){
	 MPI_Bcast(&MulP_Cc[j1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
	 for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
	 MPI_Bcast(&Orb2MulP_Cc[j1][i1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
	 }//for(i1)
	 }//for(j1)
	 }//i
	 for(j=0; j<Data_size; j++){
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
       * ***/

      /* Added by N. Yamaguchi ***/
      if (S_knum[i]>=0){
	k=S_knum[i]*n2;
	l=numK[i]*n2;
      } else {
	k=0;
	l=0;
      }
      MPI_Bcast(EIGEN_MP+k, l, MPI_DOUBLE, i, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      recvCount=(int*)malloc(sizeof(int)*num_procs);
      displs=(int*)malloc(sizeof(int)*num_procs);
      i2=(atomnum+1)*numK[myrank];
      for (j=0; j<num_procs; j++){
	recvCount[j]=(atomnum+1)*numK[j];
	displs[j]=S_knum[j]*(atomnum+1);
      }
      for (j1=0; j1<sizeMulP; j1++){
	MPI_Gatherv(MulP_CcDivision[j1], i2, MPI_DOUBLE, MulP_Cc[j1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	  MPI_Gatherv(OrbMulP_CcDivision[j1][i1], i2, MPI_DOUBLE, OrbMulP_Cc[j1][i1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	for (i1=0; i1<=ClaOrb_MAX[1]; i1++){
	  MPI_Gatherv(Orb2MulP_CcDivision[j1][i1], i2, MPI_DOUBLE, Orb2MulP_Cc[j1][i1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
      }
      return 0;
      /* ***/

      // #######################################################
      if (myrank == 0){
	strcpy(fname_MP,fname_out);    strcat(fname_MP,".AtomMulP");
	fp1= fopen(fname_MP,"w");

	/* Disabled by N. Yamaguchi ***
	   fprintf(fp1,"                                      \n");
	 * ***/

	/* Added by N. Yamaguchi ***/
	fputs("                                    \n", fp1);
	/* ***/

	fclose(fp1);

	/* Added by N. Yamaguchi ***/
	if (Spin_Dege){
	  /* ***/

	  strcpy(fname_MP,fname_out);    strcat(fname_MP,".AtomMulP_Dege");
	  fp1= fopen(fname_MP,"w");

	  /* Disabled by N. Yamaguchi ***
	     fprintf(fp1,"                                      \n");
	   * ***/

	  /* Added by N. Yamaguchi ***/
	  fputs("                                    \n", fp1);
	  /* ***/

	  fclose(fp1);

	  /* Added by N. Yamaguchi ***/
	}
	/* ***/

	// #############
	for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  strcpy(fname_MP,fname_out);     strcat(fname_MP,".MulP_");      strcat(fname_MP,OrbSym[i1]);
	  fp1= fopen(fname_MP,"w");
	  fprintf(fp1,"                                    \n");
	  fclose(fp1);
	}//i1
	for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	  strcpy(fname_MP,fname_out);     strcat(fname_MP,".MulP_");     strcat(fname_MP,OrbName[i1]);
	  fp1= fopen(fname_MP,"w");
	  fprintf(fp1,"                                    \n");
	  fclose(fp1);
	}//i1
	// ###################################################
	strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP");
	fp1= fopen(fname_MP,"a");
	for(j=0; j<Data_size; j++){
	  Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], NBand_Cc[j], EIGEN_MP[j*n2+NBand_Cc[j]]);
	  fprintf(fp1, "%s", Pdata_s);
	  for (i=0; i<atomnum; i++){
	    fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[0][j*(atomnum+1)+i], MulP_Cc[1][j*(atomnum+1)+i]);
	    fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[2][j*(atomnum+1)+i], MulP_Cc[3][j*(atomnum+1)+i]);
	  } fprintf(fp1,"\n");
	}//j
	fclose(fp1);
	// ###################################################

	/* Added by N. Yamaguchi ***/
	if (Spin_Dege){
	  strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP_Dege");
	  fp1= fopen(fname_MP,"a");
	  for(j=0; j<Data_size; j++){
	    if     (NBand_Cc[j]%2==1) {l2 = NBand_Cc[j] +1;}
	    else if(NBand_Cc[j]%2==0) {l2 = NBand_Cc[j] -1;}
	    else {}
	    Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], l2, EIGEN_MP[j*n2+l2] );
	    fprintf(fp1, "%s", Pdata_s);
	    for (i=0; i<atomnum; i++){
	      fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[4][j*(atomnum+1)+i], MulP_Cc[5][j*(atomnum+1)+i]);
	      fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[6][j*(atomnum+1)+i], MulP_Cc[7][j*(atomnum+1)+i]);
	    } fprintf(fp1,"\n");
	  }//j
	  fclose(fp1);

	  /* Added by N. Yamaguchi ***/
	}
	/* ***/

	// ###################################################
	for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbSym[i1]);
	  fp1= fopen(fname_MP,"a");
	  for(j=0; j<Data_size; j++){
	    Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], NBand_Cc[j], EIGEN_MP[j*n2+NBand_Cc[j]]);
	    fprintf(fp1, "%s", Pdata_s);
	    for (i=0; i<atomnum; i++){
	      fprintf(fp1,"%10.6lf %10.6lf ", OrbMulP_Cc[0][i1][j*(atomnum+1)+i], OrbMulP_Cc[1][i1][j*(atomnum+1)+i]);
	      fprintf(fp1,"%10.6lf %10.6lf ", OrbMulP_Cc[2][i1][j*(atomnum+1)+i], OrbMulP_Cc[3][i1][j*(atomnum+1)+i]);
	    } fprintf(fp1,"\n");
	  }//j
	  fclose(fp1);
	}//i1
	// ###################################################
	for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	  strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbName[i1]);
	  fp1= fopen(fname_MP,"a");
	  for(j=0; j<Data_size; j++){
	    Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], NBand_Cc[j], EIGEN_MP[j*n2+NBand_Cc[j]]);
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
      // #######################################################
      //      if (myrank == 0){      
      //        for(j=0; j<Data_size; j++){
      //          Print_kxyzEig(Pdata_s, k_xyz_Cc[0][j], k_xyz_Cc[1][j], k_xyz_Cc[2][j], NBand_Cc[j], EIGEN_MP[j*n2 + NBand_Cc[j]]);
      //          printf("%s\n", Pdata_s);
      //        }
      //      }//if (myrank)
      // #####################################################
      if (myrank == 0){      
	//### atomnum & data_size ###
	strcpy(fname_MP,fname_out);     strcat(fname_MP,".AtomMulP");
	fp1= fopen(fname_MP,"r+");
	fseek(fp1, 0L, SEEK_SET);
	fprintf(fp1,"%6d %4d", Data_size, atomnum);
	fclose(fp1);

	/* Added by N. Yamaguchi ***/
	if (Spin_Dege){
	  /* ***/

	  strcpy(fname_MP,fname_out);     strcat(fname_MP,".AtomMulP_Dege");
	  fp1= fopen(fname_MP,"r+");
	  fseek(fp1, 0L, SEEK_SET);
	  fprintf(fp1,"%6d %4d", Data_size, atomnum);
	  fclose(fp1);

	  /* Added by N. Yamaguchi ***/
	}
	/* ***/

	//###########################
	for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	  strcpy(fname_MP,fname_out);    strcat(fname_MP,".MulP_");    strcat(fname_MP,OrbSym[i1]);
	  fp1= fopen(fname_MP,"r+");
	  fseek(fp1, 0L, SEEK_SET);
	  fprintf(fp1,"%6d %4d", Data_size, atomnum);
	  fclose(fp1);
	}//i1
	for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
	  strcpy(fname_MP,fname_out);    strcat(fname_MP,".MulP_");    strcat(fname_MP,OrbName[i1]);
	  fp1= fopen(fname_MP,"r+");
	  fseek(fp1, 0L, SEEK_SET);
	  fprintf(fp1,"%6d %4d", Data_size, atomnum);
	  fclose(fp1);
	}//i1
      }//if(myrank)
      // #######################################################
      for(i=0; i<3; i++){
	free(k_xyz_Cc[i]);
      } free(k_xyz_Cc);
      free(NBand_Cc);
      free(EIGEN_MP);

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
      for (i=0; i<sizeMulP; i++){
	if (numK[myrank]>0){
	  free(MulP_CcDivision[i]);
	}
      } free(MulP_CcDivision);
      for (i=0; i<sizeMulP; i++){
	for (j=0; j<=ClaOrb_MAX[0]; j++){
	  if (numK[myrank]>0){
	    free(OrbMulP_CcDivision[i][j]);
	  }
	} free(OrbMulP_CcDivision[i]);
      } free(OrbMulP_CcDivision);
      for (i=0; i<sizeMulP; i++){
	for (j=0; j<=ClaOrb_MAX[1]; j++){
	  if (numK[myrank]>0){
	    free(Orb2MulP_CcDivision[i][j]);
	  }
	} free(Orb2MulP_CcDivision[i]);
      } free(Orb2MulP_CcDivision);
      /* ***/

    }//if()
    // ### MALLOC FREE #######################################

    for (i=0; i<=ClaOrb_MAX[1]; i++){
      free(OrbName[i]);
    } free(OrbName);

    // ### (MulP Calculation)   ###
    for (i=0; i<=atomnum; i++){
      free(An2Spe[i]);
    } free(An2Spe);
    for (i=0; i<=atomnum; i++){
      free(ClaOrb[i]);
    } free(ClaOrb);
    free(EIGEN);

    /* Disabled by N. Yamaguchi ***
       for (i=0; i<4; i++){
       for (l=0; l<n2; l++){
       for (j=0; j<=atomnum; j++){
       free(Data_MulP[i][l][j]);
       } free(Data_MulP[i][l]);
       } free(Data_MulP[i]);
       } free(Data_MulP);
     * ***/
    for (i=0; i<4; i++){
      for (l=0; l<l_cal; l++){
	for (j=0; j<=atomnum; j++){
	  free(Data_MulP[i][l][j]);
	} free(Data_MulP[i][l]);
      } free(Data_MulP[i]);
    } free(Data_MulP);

    free(S_knum);
    free(E_knum);
    // #######################################################

    /* Added by N> Yamaguchi ***/
#ifndef SIGMAEK
    /* ***/

    free_scfout();

    /* Added by N. Yamaguchi ***/
#endif
    /* ***/

  }//if(fo_wf)

  // ### Time Measurement ####################################
  if (myrank ==0){
    printf("############ CALC TIME ####################\n");
    dtime(&TEtime);
    printf("  Total Calculation Time:%10.6lf (s)\n",TEtime-TStime);
    printf("###########################################\n");
  }
  // ### MPI_Finalize ########################################

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  MPI_Finalize();
#endif
  /* ***/

  return 0;
}
