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
/* ***/

/* Disabled by N. Yamaguchi ***
   int Band_Dispersion()
 * ***/

/* Added by N. Yamaguchi ***/
int BandDispersion()
#else
#pragma intel optimization_level 1
int main(int argc, char *argv[])
#endif
  /* ***/

{

  FILE *fp, *fp1, *fp2;
  char c;
  char fname_BandAll[256], fname_Band[256], fname_MP[256];
  char Pdata_s[256];

  int i,j,k,l,m,n,n2, i1,i2, j1,j2, l2;           // loop variable
  int id;
  int Count_data;
  int *S_knum, *E_knum, T_knum, size_Tknum;       // MPI variable (k-point divide)
  int namelen, num_procs, myrank;                 // MPI_variable

  double k1, k2, k3;                              // k-point variable                         
  double **k_xyz, *EIGEN_Band;                      // Eigen solve array

  double d0, d1, d2, d3;

  double TStime, TEtime, Stime, Etime;            // Time variable
  int *S2_knum, *E2_knum, T2_knum;                // MPI variable (k-point divide)

  double line_k, line_k_old, *line_Nk;

  // ### Orbital Data    ###
  double **MulP_Band, ***OrbMulP_Band, ***Orb2MulP_Band;

  /* Added by N. Yamaguchi ***/
  double **MulP_BandDivision;
  double ***OrbMulP_BandDivision;
  double ***Orb2MulP_BandDivision;
#ifndef DEBUG_SIGMAEK_OLDER_20181126
  mode=(int(*)())BandDispersion;
#endif
  /* ***/

  int i_vec[20],i_vec2[20];                       // input variable
  char *s_vec[20];                                // input variable
  double r_vec[20];                               // input variable

  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2

  // ### MPI_Init ############################################
  //  MPI_Init(&argc, &argv);

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  MPI_Init(&argc, &argv);
#endif
  /* ***/

  MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  //  MPI_Get_processor_name(processor_name, &namelen);
  //  printf("myrank:%d \n",myrank);

  //  dtime(&Stime);

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  dtime(&TStime);
#endif
  /* ***/

  // ### INPUT_FILE ##########################################

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  sprintf(fname,"%s",argv[1]);
  if((fp = fopen(fname,"r")) != NULL){
    if (myrank ==0) printf("open \"%s\" file... \n" ,fname);fflush(stdout);
    fclose(fp);
  }else{
    if (myrank ==0) printf("Cannot open \"%s\" file.\n" ,fname);fflush(stdout);
    MPI_Finalize();
    return 0;
  }
#endif
  /* ***/

  input_open(fname);

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  input_string("Filename.scfout",fname_wf,"default");
  input_string("Filename.outdata",fname_out,"default");
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

  //BAND
  input_int("Band.Nkpath",&Band_Nkpath,0);
  if (Band_Nkpath>0) {

    Band_N_perpath=(int*)malloc(sizeof(int)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++) Band_N_perpath[i] = 0;

    Band_kpath = (double***)malloc(sizeof(double**)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++){
      Band_kpath[i] = (double**)malloc(sizeof(double*)*3);
      for (j=0; j<3; j++){
	Band_kpath[i][j] = (double*)malloc(sizeof(double)*4);
	for (k=0; k<4; k++) Band_kpath[i][j][k] = 0.0;
      }//for(i)
    }//for(j)
    Band_kname = (char***)malloc(sizeof(char**)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++){
      Band_kname[i] = (char**)malloc(sizeof(char*)*3);
      for (j=0; j<3; j++){
	Band_kname[i][j] = (char*)malloc(sizeof(char)*16);
      }//for(i)
    }//for(j)
    if( (fp=input_find("<Band.kpath"))  != NULL) {
      for (i=1; i<=Band_Nkpath; i++) {
	fscanf(fp,"%d %lf %lf %lf %lf %lf %lf %s %s",
	    &Band_N_perpath[i]  ,
	    &Band_kpath[i][1][1], &Band_kpath[i][1][2], &Band_kpath[i][1][3],
	    &Band_kpath[i][2][1], &Band_kpath[i][2][2], &Band_kpath[i][2][3],
	    Band_kname[i][1], Band_kname[i][2]);
      }//for(Band_Nkpath)
      if ( ! input_last("Band.kpath>") ) { // format error
	if (myrank ==0) PRINTFF("Format error near Band.kpath>\n");
	return -1;
      }
    }//else
    else { // format error
      if (myrank ==0) PRINTFF("<Band.kpath is necessary.\n");
      return -1;
    }//else
  }//if(Nkpath>0)
  input_close();

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  if (myrank==0){
    read_scfout(fname_wf, 1);
  } else {
    read_scfout(fname_wf, 0);
  }
#endif
  if (SpinP_switch!=3){
    PRINTFF("Error: \"BandDispersion\" is available only for non-collinear cases.\n");
#ifndef SIGMAEK
    MPI_Finalize();
#endif
    return 0;
  }
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
  } ClaOrb_MAX[0] = 0;
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
    //    if (myrank == 0) printf("OrbName:%s\n",OrbName[i]);
  }//i
  /* ***/

  line_Nk=(double*)malloc(sizeof(double)*(Band_Nkpath+1));
  line_Nk[0] = 0;
  for (i=1; i<(Band_Nkpath+1); i++){
    line_Nk[i] = kgrid_dist(Band_kpath[i][1][1], Band_kpath[i][2][1], Band_kpath[i][1][2], Band_kpath[i][2][2], Band_kpath[i][1][3], Band_kpath[i][2][3]) +line_Nk[i-1]; 

    /* Disabled by N. Yamaguchi ***
       if (myrank ==0) printf("line_Nk[%d]:%10.6lf\n",i,line_Nk[i]);
       if (myrank ==0) printf("%10.6lf-%10.6lf,%10.6lf-%10.6lf,%10.6lf-%10.6lf\n",Band_kpath[i][1][1], Band_kpath[i][2][1], Band_kpath[i][1][2], Band_kpath[i][2][2], Band_kpath[i][1][3], Band_kpath[i][2][3]);
     * ***/

    /* Added by N. Yamaguchi ***/
    if (myrank==0) PRINTFF("line_Nk[%d]:%10.6f\n",i,line_Nk[i]);
    if (myrank==0) PRINTFF("(%9.6f, %9.6f, %9.6f ) -> (%9.6f, %9.6f, %9.6f )\n", Band_kpath[i][1][1], Band_kpath[i][1][2], Band_kpath[i][1][3], Band_kpath[i][2][1], Band_kpath[i][2][2], Band_kpath[i][2][3]);
    /* ***/

  }

  // ### Band Total (n2) ###
  k = 1;
  for (i=1; i<=atomnum; i++){ k+= Total_NumOrbs[i]; }
  n = k - 1;    n2 = 2*k + 2;

  if (myrank == 0){
    PRINTFF("########### ORBITAL DATA ##################\n");
    //      for(i=1;i<=atomnum;i++) printf("%4d:%4d\n", i, Total_NumOrbs[i]);
    //      printf("  MAX:%4d\n",TNO_MAX);
    PRINTFF("ClaOrb_MAX[0]:%4d\n",ClaOrb_MAX[0]);
    PRINTFF("ClaOrb_MAX[1]:%4d\n",ClaOrb_MAX[1]);
    PRINTFF("Total Band (2*n):%4d\n",n2-4);
    //      printf("Central (%10.6lf %10.6lf %10.6lf)\n",k_CNT1[0],k_CNT1[1],k_CNT1[2]);
    PRINTFF("###########################################\n");
  }

  // ######################################################
  S_knum = (int*)malloc(sizeof(int)*num_procs);
  E_knum = (int*)malloc(sizeof(int)*num_procs);

  // ### (EigenValue Problem) ###
  EIGEN = (double*)malloc(sizeof(double)*n2);
  for (i = 0; i < n2; i++) EIGEN[i] = 0.0;

  // ### (EigenValue Problem) ###
  if (Band_Nkpath>0) {
    // ################################################## 
    if (myrank ==0) {
      PRINTFF("Band.Nkpath: %d\n",Band_Nkpath);
      for (i=1; i<=Band_Nkpath; i++) {

	/* Disabled by N. Yamaguchi ***
	   printf("%d (%lf %lf %lf) >>> (%lf %lf %lf) %s %s\n",
	   Band_N_perpath[i] ,
	   Band_kpath[i][1][1], Band_kpath[i][1][2],Band_kpath[i][1][3],
	   Band_kpath[i][2][1], Band_kpath[i][2][2],Band_kpath[i][2][3],
	   Band_kname[i][1],Band_kname[i][2]);
	 * ***/

	/* Added by N. Yamaguchi ***/
	PRINTFF("%d (%9.6f, %9.6f, %9.6f ) >>> (%9.6f, %9.6f, %9.6f ) %s %s\n",
	    Band_N_perpath[i],
	    Band_kpath[i][1][1], Band_kpath[i][1][2],Band_kpath[i][1][3],
	    Band_kpath[i][2][1], Band_kpath[i][2][2],Band_kpath[i][2][3],
	    Band_kname[i][1], Band_kname[i][2]);
	/* ***/

      }
      //### MULP OUTPUT ###
      strcpy(fname_MP, fname_out);
      strcat(fname_MP, ".AMulPBand");
      fp1 = fopen(fname_MP, "w");

      /* Disabled by N. Yamaguchi ***
	 fprintf(fp1,"                                        ");
	 fprintf(fp1,"                                        ");
	 fprintf(fp1,"                                        ");
       * ***/

      /* Added by N. Yamaguchi ***/
      fputs("                                                                ", fp1);
      fputs("                                                                ", fp1);
      fputs("                                                            ", fp1);
      /* ***/

      fclose(fp1);

      /* Added by N. Yamaguchi ***/
      for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	strcpy(fname_MP, fname_out);
	strcat(fname_MP, ".AMulPBand_");
	strcat(fname_MP, OrbSym[i1]);
	fp1=fopen(fname_MP, "w");
	fputs("                                                                ", fp1);
	fputs("                                                                ", fp1);
	fputs("                                                            ", fp1);
	fclose(fp1);
      }
      for (i1=1; i1<=ClaOrb_MAX[1]; i1++){
	strcpy(fname_MP, fname_out);
	strcat(fname_MP, ".AMulPBand_");
	strcat(fname_MP, OrbName[i1]);
	fp1= fopen(fname_MP,"w");
	fputs("                                                                ", fp1);
	fputs("                                                                ", fp1);
	fputs("                                                            ", fp1);
	fclose(fp1);
      }
      char fname_gp[256];
      strcpy(fname_gp, fname_out);
      strcat(fname_gp, ".plotexample");
      fp1=fopen(fname_gp, "w");
      fputs("reduction=1\n", fp1);
      fputs("#reduction=20\n", fp1);
      fputs("linewidth=3\n", fp1);
      fputs("set nokey\n", fp1);
      fputs("set zeroaxis\n", fp1);
      fputs("set mytics 5\n", fp1);
      fputs("set grid\n", fp1);
      fputs("set ylabel 'E-E_F (eV)'\n", fp1);
      fprintf(fp1, "set xrange [%f:%f]\n", line_Nk[0], line_Nk[Band_Nkpath]);
      fprintf(fp1, "set yrange [%f:%f]\n", E_Range[0], E_Range[1]);
      for (i=1; i<=Band_Nkpath; i++) {
	if (i==1){
	  fprintf(fp1, "set xtics ('%s' %f", Band_kname[i][1], line_Nk[i-1]);
	}
	if (i==Band_Nkpath){
	  fprintf(fp1, ", '%s' %f)\n", Band_kname[i][2], line_Nk[i]);
	} else if (!strcmp(Band_kname[i][2], Band_kname[i+1][1])){
	  fprintf(fp1, ", '%s' %f", Band_kname[i+1][1], line_Nk[i]);
	} else {
	  fprintf(fp1, ", '%s, %s' %f", Band_kname[i][2], Band_kname[i+1][1], line_Nk[i]);
	}
      }
      fprintf(fp1, "plot '%s.BAND' with lines notitle linewidth linewidth\n", fname_out);
      fclose(fp1);
      /* ***/

    }
    // ################################################## 


    Count_data = 0;

    for (i=1; i<=Band_Nkpath; i++) {
      // ### MALLOC ARRAY ############################### 
      size_Tknum = (Band_N_perpath[i] + 1);

      k_xyz = (double**)malloc(sizeof(double*)*3);
      for(j=0; j<3; j++) k_xyz[j] = (double*)malloc(sizeof(double)*(size_Tknum+1));
      EIGEN_Band = (double*)malloc(sizeof(double)*((size_Tknum+1)*n2));
      for (j = 0; j < ((size_Tknum+1)*n2); j++) EIGEN_Band[j] = 0.0;

      // ### Division CALC_PART #########################
      T_knum = size_Tknum;
      for(j=0; j<num_procs; j++){
	if (T_knum <= j){
	  S_knum[j] = -10;   E_knum[j] = -100;

	  /* Added by N. Yamaguchi ***/
#if 0
	  /* ***/

	} else if (T_knum < num_procs) {

	  /* Added by N. Yamaguchi ***/
#endif
	} else if (T_knum<=num_procs){
	  /* ***/

	  /* Disabled by N. Yamaguchi
	   * S_knum[j] = j;     E_knum[j] = i;
	   */

	  S_knum[j] = j;     E_knum[j] = j;
	} else {
	  d0 = (double)T_knum/(double)num_procs;
	  S_knum[j] = (int)((double)j*(d0+0.0001));
	  E_knum[j] = (int)((double)(j+1)*(d0+0.0001)) - 1;
	  if (j==(num_procs-1)) E_knum[j] = T_knum - 1;
	  if (E_knum[j]<0)      E_knum[j] = 0;
	}
      }

      int *numK=(int*)malloc(sizeof(int)*num_procs);
      for (j=0; j<num_procs; j++){
	numK[j]=E_knum[j]-S_knum[j]+1;
	if (numK[j]<=0){
	  numK[j]=0;
	}
      }
      // ################################################
      for (j = S_knum[myrank]; j <= E_knum[myrank]; j++){
	id=1; 
	k1 = Band_kpath[i][1][id]+(Band_kpath[i][2][id]-Band_kpath[i][1][id])*j/Band_N_perpath[i] + Shift_K_Point;
	id=2;
	k2 = Band_kpath[i][1][id]+(Band_kpath[i][2][id]-Band_kpath[i][1][id])*j/Band_N_perpath[i] - Shift_K_Point;
	id=3;
	k3 = Band_kpath[i][1][id]+(Band_kpath[i][2][id]-Band_kpath[i][1][id])*j/Band_N_perpath[i] + 2.0*Shift_K_Point;
	k_xyz[0][j] = k1;      k_xyz[1][j] = k2;      k_xyz[2][j] = k3;
	EigenValue_Problem(k1, k2, k3, 0);
	for(l=1; l<=2*n; l++){  EIGEN_Band[j*n2+l] = EIGEN[l];  }//l
      }//j
      // ### MPI part ###################################
      for (j=0; j<num_procs; j++){

	/* Disabled by N. Yamaguchi
	 * k = S_knum[j];          l = abs(E_knum[j]-S_knum[j]+1);
	 */

	/* Added by N. Yamaguchi ***/
	if (S_knum[j]>=0){
	  k=S_knum[j];
	  l=numK[j];
	} else {
	  k=0;
	  l=0;
	}
	/* ***/

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&k_xyz[0][k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&k_xyz[1][k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&k_xyz[2][k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * k = S_knum[j]*n2;       l = abs(E_knum[j]-S_knum[j]+1)*n2;
	 */

	/* Added by N. Yamaguchi ***/
	if (S_knum[j]>=0){
	  k=S_knum[j]*n2;
	  l=numK[j]*n2;
	} else {
	  k=0;
	  l=0;
	}
	MPI_Bcast(EIGEN_Band+k, l, MPI_DOUBLE, j, MPI_COMM_WORLD);
	/* ***/

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 * MPI_Bcast(&EIGEN_Band[k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

      }//j
      // ################################################
      // # SELECT CALC_BAND #############################
      id = 0;

      /* Disabled by N. Yamaguchi ***
	 for(l=1; l<=2*n; l++){
	 if ((E_Range[1]-EIGEN_Band[l])*(E_Range[0]-EIGEN_Band[l])<=0){
	 if (id == 0){ l_min=l; l_max=l; id=1;}
	 else if(id == 1){ l_max=l;}
	 }
	 }
	 for(l=1; l<=2*n; l++){
	 for(j=0; j<size_Tknum; j++){
	 if ((E_Range[1]-EIGEN_Band[j*n2+l])*(E_Range[0]-EIGEN_Band[j*n2+l])<=0){
	 if (l_min>l){ l_min=l; }
	 else if(l_max<l){ l_max=l;}
	 }
	 }
	 }
       * ***/

      /* Added by N. Yamaguchi ***/
      for(l=1; l<=2*n; l++){
	for(j=0; j<size_Tknum; j++){
	  if ((E_Range[1]-EIGEN_Band[j*n2+l])*(E_Range[0]-EIGEN_Band[j*n2+l])<=0){
	    if (id == 0){ l_min=l; l_max=l; id=1;}
	    else if(id == 1){ l_max=l;}
	  }
	}
      }
      //if (id==0) MPI_Abort(MPI_COMM_WORLD, 1);
      if (id==0){
	PRINTFF("Error: No Bands are found.\n");
#ifndef SIGMAEK
	MPI_Finalize();
#endif
	return 0;
      }
      /* ***/

      l_cal = l_max-l_min +1;
      if(l_cal<1) l_cal=1;
      if (myrank ==0) PRINTFF("l_min:%4d   l_max:%4d   l_cal:%4d \n" ,l_min ,l_max , l_cal);

      // # Print Band ALL ###############################
      if (myrank == 0){
	strcpy(fname_BandAll, fname_out);  strcat(fname_BandAll, ".BAND");

	/* Added by N. Yamaguchi ***/
	if (i>1){
	  /* ***/
	  fp1 = fopen(fname_BandAll, "a");
	  /* Added by N. Yamaguchi ***/
	} else {
	  fp1 = fopen(fname_BandAll, "w");
	}
	/* ***/
	for(l=1; l<=2*n; l++){
	  for(j=0; j<size_Tknum; j++){
	    line_k=line_Nk[i-1]+kgrid_dist(Band_kpath[i][1][1],k_xyz[0][j],Band_kpath[i][1][2],k_xyz[1][j],Band_kpath[i][1][3],k_xyz[2][j]);
	    fprintf(fp1, "%10.6lf %10.6lf\n", line_k, EIGEN_Band[j*n2+l]);
	  } fprintf(fp1, "\n\n");
	} fclose(fp1);
	// # Print Band (division) ######################
	for(l=l_min; l<=l_max; l++){

	  /* Disabled by N. Yamaguchi ***
	  strcpy(fname_Band,fname_out);  name_Nband(fname_Band,"_",i);  name_Nband(fname_Band,".Band",l);
	  * ***/

	  /* Added by N. Yamaguchi ***/
	  strcpy(fname_Band,fname_out);
	  name_Nband(fname_Band,".Band",l);
	  name_Nband(fname_Band,"_",i);
	  PRINTFF("%s\n", fname_Band);
	  /* ***/

	  fp1= fopen(fname_Band,"w");
	  for(j=0; j<size_Tknum; j++){
	    line_k=line_Nk[i-1]+kgrid_dist(Band_kpath[i][1][1],k_xyz[0][j],Band_kpath[i][1][2],k_xyz[1][j],Band_kpath[i][1][3],k_xyz[2][j]);
	    if ((E_Range[1]-EIGEN_Band[j*n2+l])*(E_Range[0]-EIGEN_Band[j*n2+l])<=0){
	      fprintf(fp1, "%10.6lf %10.6lf\n", line_k, EIGEN_Band[j*n2+l]);
	    }
	  } fclose(fp1);
	}//l

	/* Added by N. Yamaguchi ***/
	char fname_gp[256];
	strcpy(fname_gp, fname_out);
	strcat(fname_gp, ".plotexample");
	fp1=fopen(fname_gp, "a");
	for (l=l_min; l<=l_max; l++){

	  /* Disabled by N. Yamaguchi ***
	  fprintf(fp1, "#replot '%s_%d.Band%d' with lines notitle linewidth linewidth\n", fname_out, i, l);
	  * ***/

	  /* Added by N. Yamaguchi ***/
	  fprintf(fp1, "#replot '%s.Band%d_%d' with lines notitle linewidth linewidth\n", fname_out, l, i);
	  /* ***/

	}
	fclose(fp1);
	/* ***/

      }//if(myrank)
      // ################################################

      // ### (MulP Calculation)   ###
      Data_MulP = (double****)malloc(sizeof(double***)*4);
      for (i1=0; i1<4; i1++){
	Data_MulP[i1] = (double***)malloc(sizeof(double**)*l_cal);
	for (l=0; l<l_cal; l++){
	  Data_MulP[i1][l] = (double**)malloc(sizeof(double*)*(atomnum+1));
	  for (j=0; j<=atomnum; j++){
	    Data_MulP[i1][l][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
	    for (k=0; k<=TNO_MAX; k++) Data_MulP[i1][l][j][k] = 0.0;
	  }
	}
      }
      MulP_Band = (double**)malloc(sizeof(double*)*4);

      /* Added by N. Yamaguchi ***/
      if (myrank==0){
	/* ***/

	for (j=0; j<4; j++){
	  MulP_Band[j] = (double*)malloc(sizeof(double)*((atomnum+1)*(size_Tknum+1)*l_cal));
	  for (k=0; k<((atomnum+1)*(size_Tknum+1)*l_cal); k++) MulP_Band[j][k] = 0.0;
	}

	/* Added by N. Yamaguchi ***/
      }
      /* ***/

      OrbMulP_Band = (double***)malloc(sizeof(double**)*4);
      for (j=0; j<4; j++){
	OrbMulP_Band[j] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));

	/* Added by N. Yamaguchi ***/
	if (myrank==0){
	  /* ***/

	  for (j1=0; j1<=ClaOrb_MAX[0]; j1++){
	    OrbMulP_Band[j][j1] = (double*)malloc(sizeof(double)*((atomnum+1)*(size_Tknum+1)*l_cal));
	    for (k=0; k<((atomnum+1)*(size_Tknum+1)*l_cal); k++) OrbMulP_Band[j][j1][k] = 0.0;
	  }//j1

	  /* Added by N. Yamaguchi ***/
	}
	/* ***/

      }//j
      Orb2MulP_Band = (double***)malloc(sizeof(double**)*4);
      for (j=0; j<4; j++){
	Orb2MulP_Band[j] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));

	/* Added by N. Yamaguchi ***/
	if (myrank==0){
	  /* ***/

	  for (j1=0; j1<=ClaOrb_MAX[1]; j1++){
	    Orb2MulP_Band[j][j1] = (double*)malloc(sizeof(double)*((atomnum+1)*(size_Tknum+1)*l_cal));
	    for (k=0; k<((atomnum+1)*(size_Tknum+1)*l_cal); k++) Orb2MulP_Band[j][j1][k] = 0.0;
	  }//j1

	  /* Added by N. Yamaguchi ***/
	}
	/* ***/

      }//j
      /* Added by N. Yamaguchi ***/
      MulP_BandDivision = (double**)malloc(sizeof(double*)*4);
      for (j=0; j<4; j++){
	MulP_BandDivision[j] = (double*)malloc(sizeof(double)*((atomnum+1)*numK[myrank]*l_cal));
	for (k=0; k<((atomnum+1)*numK[myrank]*l_cal); k++) MulP_BandDivision[j][k] = 0.0;
      }
      OrbMulP_BandDivision = (double***)malloc(sizeof(double**)*4);
      for (j=0; j<4; j++){
	OrbMulP_BandDivision[j] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
	for (j1=0; j1<=ClaOrb_MAX[0]; j1++){
	  OrbMulP_BandDivision[j][j1] = (double*)malloc(sizeof(double)*((atomnum+1)*numK[myrank]*l_cal));
	  for (k=0; k<((atomnum+1)*numK[myrank]*l_cal); k++) OrbMulP_BandDivision[j][j1][k] = 0.0;
	}
      }
      Orb2MulP_BandDivision = (double***)malloc(sizeof(double**)*4);
      for (j=0; j<4; j++){
	Orb2MulP_BandDivision[j] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
	for (j1=0; j1<=ClaOrb_MAX[1]; j1++){
	  Orb2MulP_BandDivision[j][j1] = (double*)malloc(sizeof(double)*((atomnum+1)*numK[myrank]*l_cal));
	  for (k=0; k<((atomnum+1)*numK[myrank]*l_cal); k++) Orb2MulP_BandDivision[j][j1][k] = 0.0;
	}
      }
      /* ***/

      for (j = S_knum[myrank]; j <= E_knum[myrank]; j++){
	EigenValue_Problem(k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], 1);
	for(l=l_min; l<=l_max; l++){
	  for (i2=0; i2 < atomnum; i2++){
	    for (j1=0; j1<4; j1++){
	      for (i1=0; i1 < Total_NumOrbs[i2+1]; i1++){

		/* Disabled by N. Yamaguchi
		 * Orb2MulP_Band[j1][ClaOrb[i2+1][i1]][j*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+= Data_MulP[j1][l-l_min][i2+1][i1];
		 * MulP_Band[j1][j*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+= Data_MulP[j1][l-l_min][i2+1][i1];
		 */

		/* Added by N. Yamaguchi ***/
		MulP_BandDivision[j1][(j-S_knum[myrank])*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+=Data_MulP[j1][l-l_min][i2+1][i1];
		Orb2MulP_BandDivision[j1][ClaOrb[i2+1][i1]][(j-S_knum[myrank])*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+=Data_MulP[j1][l-l_min][i2+1][i1];
		if (ClaOrb[i2+1][i1]==0){
		  OrbMulP_BandDivision[j1][0][(j-S_knum[myrank])*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+=Data_MulP[j1][l-l_min][i2+1][i1];
		} else if (ClaOrb[i2+1][i1]>=1 && ClaOrb[i2+1][i1]<=3){
		  OrbMulP_BandDivision[j1][1][(j-S_knum[myrank])*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+=Data_MulP[j1][l-l_min][i2+1][i1];
		} else if (ClaOrb[i2+1][i1]>=4 && ClaOrb[i2+1][i1]<=8){
		  OrbMulP_BandDivision[j1][2][(j-S_knum[myrank])*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+=Data_MulP[j1][l-l_min][i2+1][i1];
		} else if (ClaOrb[i2+1][i1]>=9 && ClaOrb[i2+1][i1]<=15){
		  OrbMulP_BandDivision[j1][3][(j-S_knum[myrank])*(atomnum+1)*l_cal +i2*l_cal +(l-l_min)]+=Data_MulP[j1][l-l_min][i2+1][i1];
		}
		/* ***/

	      }//for(i1)
	    }//for(j1)
	  }//for(i2)
	}//l

      }//j
      // ### MPI part ###################################

      /* Disabled by N. Yamaguchi
       * for (j=0; j<num_procs; j++){
       * k = S_knum[j]*(atomnum+1)*l_cal;
       * i2 = abs(E_knum[j]-S_knum[j]+1)*(atomnum+1)*l_cal;
       * for (j1=0; j1<4; j1++){
       * MPI_Barrier(MPI_COMM_WORLD);
       * MPI_Bcast(&MulP_Band[j1][k], i2, MPI_DOUBLE, j, MPI_COMM_WORLD);
       * for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
       * MPI_Bcast(&Orb2MulP_Band[j1][i1][k], i2, MPI_DOUBLE, j, MPI_COMM_WORLD);
       * }//for(i1)
       * }//for(j1)
       * }//for(j)
       */

      for (j=0; j<num_procs; j++){

	/* Disabled by N. Yamaguchi
	 * k = S_knum[j];          l = abs(E_knum[j]-S_knum[j]+1);
	 */

	/* Added by N. Yamaguchi ***/
	l = E_knum[j]-S_knum[j]+1;
	if (l<0) {
	  l=0;
	  k=0;
	} else {
	  k=S_knum[j];
	}
	/* ***/

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&k_xyz[0][k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&k_xyz[1][k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&k_xyz[2][k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * k = S_knum[j]*n2;       l = abs(E_knum[j]-S_knum[j]+1)*n2;
	 */

	/* Added by N. Yamaguchi ***/
	l=(E_knum[j]-S_knum[j]+1)*n2;
	if (l<0) {
	  l=0;
	  k=0;
	} else {
	  k=S_knum[j]*n2;
	}
	/* ***/

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	MPI_Bcast(&EIGEN_Band[k], l, MPI_DOUBLE, j, MPI_COMM_WORLD);

	/* Disabled by N. Yamaguchi
	 * MPI_Barrier(MPI_COMM_WORLD);
	 */

	/*
	   k = S_knum[j]*(atomnum+1)*n2;
	   i2 = abs(E_knum[j]-S_knum[j]+1)*(atomnum+1)*n2;
	   for (j1=0; j1<4; j1++){
	   MPI_Barrier(MPI_COMM_WORLD);
	   MPI_Bcast(&MulP_Band[j1][k], i2, MPI_DOUBLE, j, MPI_COMM_WORLD);
	   for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
	   MPI_Bcast(&Orb2MulP_Band[j1][i1][k], i2, MPI_DOUBLE, j, MPI_COMM_WORLD);
	   }//for(i1)
	   }//for(j1)
	 */
      }//j

      /* Added by N. Yamaguchi ***/
      int *recvCount=(int*)malloc(sizeof(int)*num_procs);
      int *displs=(int*)malloc(sizeof(int)*num_procs);
      i2=(atomnum+1)*numK[myrank]*l_cal;
      for (j=0; j<num_procs; j++){
	recvCount[j]=(atomnum+1)*numK[j]*l_cal;
	displs[j]=S_knum[j]*(atomnum+1)*l_cal;
      }
      for (j1=0; j1<4; j1++){
	MPI_Gatherv(MulP_BandDivision[j1], i2, MPI_DOUBLE, MulP_Band[j1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	  MPI_Gatherv(OrbMulP_BandDivision[j1][i1], i2, MPI_DOUBLE, OrbMulP_Band[j1][i1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	for (i1=0; i1<=ClaOrb_MAX[1]; i1++){
	  MPI_Gatherv(Orb2MulP_BandDivision[j1][i1], i2, MPI_DOUBLE, Orb2MulP_Band[j1][i1], recvCount, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
      }
      /* ***/

      // ################################################

      /* Disabled by N. Yamaguchi ***
	 if (myrank==0){
	 for(l=l_min; l<=l_max; l++){
	 for(j=0; j<size_Tknum; j++){
	 for (id=0; id<atomnum; id++){
	 for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
	 for (i2=0; i2 <=ClaOrb_MAX[1]; i2++){
	 if (OrbSym[i1][0] == OrbName[i2][0]){
	 for (j1=0; j1<4; j1++){
	 OrbMulP_Band[j1][i1][j*(atomnum+1)*l_cal +id*l_cal +(l-l_min)]+= Orb2MulP_Band[j1][i2][j*(atomnum+1)*l_cal +id*l_cal +(l-l_min)];
	 }//j1
	 }//if
	 }//i2
	 }//i1
	 }//id
	 }//j
	 }//l
	 }
       * ***/

      // ##############################################
      if (myrank == 0){
	strcpy(fname_MP,fname_out);  strcat(fname_MP,".AMulPBand");
	fp1= fopen(fname_MP,"a");

	/* Added by N. Yamaguchi ***/
#if 0
	/* ***/

	for(l=1; l<=2*n; l++){

	  /* Added by N. Yamaguchi ***/
#endif
	  for (l=l_min; l<=l_max; l++){
	    /* ***/

	    for(j=0; j<size_Tknum; j++){


	      /* Added by N. Yamaguchi ***/
#if 0
	      /* ***/

	      if ((E_Range[1]-EIGEN_Band[j*n2+l])*(E_Range[0]-EIGEN_Band[j*n2+l])<=0){

		/* Added by N. Yamaguchi ***/
#endif
		/* ***/

		Print_kxyzEig(Pdata_s, k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], l, EIGEN_Band[j*n2+l]);
		fprintf(fp1, "%s", Pdata_s);
		for (k=0; k<atomnum; k++){
		  fprintf(fp1,"%10.6lf %10.6lf "
		      , MulP_Band[0][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)], MulP_Band[1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)]);
		  fprintf(fp1,"%10.6lf %10.6lf "
		      , MulP_Band[2][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)], MulP_Band[3][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)]);
		}
		Count_data++;

		// BAND ONLY
		line_k=line_Nk[i-1]+kgrid_dist(Band_kpath[i][1][1],k_xyz[0][j],Band_kpath[i][1][2],k_xyz[1][j],Band_kpath[i][1][3],k_xyz[2][j]);
		fprintf(fp1, "              %10.6lf\n", line_k);

		/* Added by N. Yamaguchi ***/
#if 0
		/* ***/

	      }//if

	      /* Added by N. Yamaguchi ***/
#endif
	      /* ***/

	    }//j

	    /* Added by N. Yamaguchi ***/
	  }
#if 0
	  /* ***/

	}//l

	/* Added by N. Yamaguchi ***/
#endif
	/* ***/
	fclose(fp1);

	/* Added by N. Yamaguchi ***/
	for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
	  strcpy(fname_MP, fname_out);
	  strcat(fname_MP, ".AMulPBand_");
	  strcat(fname_MP, OrbSym[i1]);
	  fp1= fopen(fname_MP,"a");
	  for (l=l_min; l<=l_max; l++){
	    for(j=0; j<size_Tknum; j++){
	      Print_kxyzEig(Pdata_s, k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], l, EIGEN_Band[j*n2+l]);
	      fprintf(fp1, "%s", Pdata_s);
	      for (k=0; k<atomnum; k++){
		fprintf(fp1, "%10.6lf %10.6lf "
		    , OrbMulP_Band[0][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)], OrbMulP_Band[1][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)]);
		fprintf(fp1, "%10.6lf %10.6lf "
		    , OrbMulP_Band[2][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)], OrbMulP_Band[3][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)]);
	      }
	      line_k=line_Nk[i-1]+kgrid_dist(Band_kpath[i][1][1],k_xyz[0][j],Band_kpath[i][1][2],k_xyz[1][j],Band_kpath[i][1][3],k_xyz[2][j]);
	      fprintf(fp1, "              %10.6lf\n", line_k);
	    }
	  }
	  fclose(fp1);
	}
	for (i1=1; i1<=ClaOrb_MAX[1]; i1++){
	  strcpy(fname_MP, fname_out);
	  strcat(fname_MP, ".AMulPBand_");
	  strcat(fname_MP, OrbName[i1]);
	  fp1= fopen(fname_MP,"a");
	  for (l=l_min; l<=l_max; l++){
	    for(j=0; j<size_Tknum; j++){
	      Print_kxyzEig(Pdata_s, k_xyz[0][j], k_xyz[1][j], k_xyz[2][j], l, EIGEN_Band[j*n2+l]);
	      fprintf(fp1, "%s", Pdata_s);
	      for (k=0; k<atomnum; k++){
		fprintf(fp1, "%10.6lf %10.6lf "
		    , Orb2MulP_Band[0][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)], Orb2MulP_Band[1][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)]);
		fprintf(fp1, "%10.6lf %10.6lf "
		    , Orb2MulP_Band[2][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)], Orb2MulP_Band[3][i1][j*(atomnum+1)*l_cal +k*l_cal +(l-l_min)]);
	      }
	      line_k=line_Nk[i-1]+kgrid_dist(Band_kpath[i][1][1],k_xyz[0][j],Band_kpath[i][1][2],k_xyz[1][j],Band_kpath[i][1][3],k_xyz[2][j]);
	      fprintf(fp1, "              %10.6lf\n", line_k);
	    }
	  }
	  fclose(fp1);
	}
      }//if(myrank)

      // ################################################

      // ### free ###
      for(j=0; j<3; j++) free(k_xyz[j]);
      free(k_xyz);
      free(EIGEN_Band);

      /* Added by N. Yamaguchi ***/
      if (myrank==0){
	for (j=0; j<4; j++){
	  free(MulP_Band[j]);
	} free(MulP_Band);
	for (j=0; j<4; j++){
	  for (j1=0; j1<=ClaOrb_MAX[0]; j1++){
	    free(OrbMulP_Band[j][j1]);
	  } free(OrbMulP_Band[j]);
	} free(OrbMulP_Band);
	for (j=0; j<4; j++){
	  for (j1=0; j1<=ClaOrb_MAX[1]; j1++){
	    free(Orb2MulP_Band[j][j1]);
	  } free(Orb2MulP_Band[j]);
	} free(Orb2MulP_Band);
      }
      free(numK);
      for (j=0; j<4; j++){
	free(MulP_BandDivision[j]);
      } free(MulP_BandDivision);
      for (j=0; j<4; j++){
	for (j1=0; j1<=ClaOrb_MAX[0]; j1++){
	  free(OrbMulP_BandDivision[j][j1]);
	} free(OrbMulP_BandDivision[j]);
      } free(OrbMulP_BandDivision);
      for (j=0; j<4; j++){
	for (j1=0; j1<=ClaOrb_MAX[1]; j1++){
	  free(Orb2MulP_BandDivision[j][j1]);
	} free(Orb2MulP_BandDivision[j]);
      } free(Orb2MulP_BandDivision);
      /* ***/

      for (i1=0; i1<4; i1++){
	for (l=0; l<l_cal; l++){
	  for (j=0; j<=atomnum; j++){
	    free(Data_MulP[i1][l][j]);
	  } free(Data_MulP[i1][l]);
	} free(Data_MulP[i1]);
      } free(Data_MulP);

      /* Disabled by N. Yamaguchi ***
       * for (j=0; j<4; j++){
       * free(MulP_Band[j]);
       * } free(MulP_Band);
       * for (j=0; j<4; j++){
       * for (j1=0; j1<=ClaOrb_MAX[0]; j1++){
       * free(OrbMulP_Band[j][j1]);
       * } free(OrbMulP_Band[j]);
       * } free(OrbMulP_Band);
       * for (j=0; j<4; j++){
       * for (j1=0; j1<=ClaOrb_MAX[1]; j1++){
       * free(Orb2MulP_Band[j][j1]);
       * } free(Orb2MulP_Band[j]);
       * } free(Orb2MulP_Band);
       * ***/

      // ################################################

    }//i(Band_Nkpath)
  }//if(Band_Nkpath>0)

  if (myrank == 0){
    //### atomnum & data_size ###
    PRINTFF("###########################################\n");
    PRINTFF("Total MulP data:%4d\n", Count_data); 

    strcpy(fname_MP,fname_out);      strcat(fname_MP,".AMulPBand");
    fp1= fopen(fname_MP,"r+");
    fseek(fp1, 0L, SEEK_SET);

    /* Disabled by N. Yamaguchi ***
       fprintf(fp1,"%6d %4d   \n %10.6lf %10.6lf %10.6lf \n %10.6lf %10.6lf %10.6lf \n %10.6lf %10.6lf %10.6lf \n",  Count_data, atomnum, rtv[1][1], rtv[2][1], rtv[3][1], rtv[1][2], rtv[2][2], rtv[3][2], rtv[1][3], rtv[2][3], rtv[3][3] );
     * ***/

    /* Added by N. Yamaguchi ***/
    fprintf(fp1,"%6d %4d\n %18.14f %18.14f %18.14f \n %18.14f %18.14f %18.14f \n %18.14f %18.14f %18.14f\n", Count_data, atomnum, rtv[1][1], rtv[2][1], rtv[3][1], rtv[1][2], rtv[2][2], rtv[3][2], rtv[1][3], rtv[2][3], rtv[3][3]);
    fclose(fp1);
    for (i1=0; i1<=ClaOrb_MAX[0]; i1++){
      strcpy(fname_MP, fname_out);
      strcat(fname_MP, ".AMulPBand_");
      strcat(fname_MP, OrbSym[i1]);
      fp1= fopen(fname_MP, "r+");
      fseek(fp1, 0L, SEEK_SET);
      fprintf(fp1,"%6d %4d\n %18.14f %18.14f %18.14f \n %18.14f %18.14f %18.14f \n %18.14f %18.14f %18.14f\n", Count_data, atomnum, rtv[1][1], rtv[2][1], rtv[3][1], rtv[1][2], rtv[2][2], rtv[3][2], rtv[1][3], rtv[2][3], rtv[3][3]);
      fclose(fp1);
    }
    for (i1=1; i1<=ClaOrb_MAX[1]; i1++){
      strcpy(fname_MP, fname_out);
      strcat(fname_MP, ".AMulPBand_");
      strcat(fname_MP, OrbName[i1]);
      fp1= fopen(fname_MP, "r+");
      fseek(fp1, 0L, SEEK_SET);
      fprintf(fp1,"%6d %4d\n %18.14f %18.14f %18.14f \n %18.14f %18.14f %18.14f \n %18.14f %18.14f %18.14f\n", Count_data, atomnum, rtv[1][1], rtv[2][1], rtv[3][1], rtv[1][2], rtv[2][2], rtv[3][2], rtv[1][3], rtv[2][3], rtv[3][3]);
      fclose(fp1);
    }
    char fname_gp[256];
    strcpy(fname_gp, fname_out);
    strcat(fname_gp, ".plotexample");
    fp1=fopen(fname_gp, "a");
    fputs("pause -1\n", fp1);
    fclose(fp1);
    /* ***/

  }//if(myrank==0)

  // ### (MPI Calculation)    ###
  free(S_knum);
  free(E_knum);

  // ### (EIGEN Calclation)   ### 
  free(EIGEN);

  // ### (Band Calculation)   ###
  free(line_Nk);
  free(Band_N_perpath);
  if (Band_Nkpath>0) {
    for (i=0; i<(Band_Nkpath+1); i++){
      for (j=0; j<3; j++){
	free(Band_kpath[i][j]);
      }free(Band_kpath[i]);
    }free(Band_kpath);
    for (i=0; i<(Band_Nkpath+1); i++){
      for (j=0; j<3; j++){
	free(Band_kname[i][j]);
      }free(Band_kname[i]);
    }free(Band_kname);
  }//if(Band_Nkpath>0)

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  free_scfout();
#endif
  /* ***/

  // #######################################################

  /* Added by N. Yamaguchi ***/
#ifndef SIGMAEK
  if (myrank ==0){
    printf("############ CALC TIME ####################\n");
    dtime(&TEtime);
    printf("  Total Calculation Time:%10.6lf (s)\n",TEtime-TStime);
    printf("###########################################\n");
  }
  MPI_Finalize();
#endif
  /* ***/

  return 0;

}
