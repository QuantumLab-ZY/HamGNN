/**********************************************************************
  TRAN_RestartFile.c:

  TRAN_RestartFile.c is a subroutine to save and load data to
  restart without input file from RestartFileDFT.c.

  Log of RestartFile.c:

     11/Dec/2005   Released by H. Kino

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "tran_prototypes.h" 
#include "tran_variables.h"

#define print_stdout 0

/****************************************************************

"read" mode allocates 

int *WhatSpecies_l;
int *Spe_Total_CNO_l;
int *Spe_Total_NO_l;
int *FNAN_l;
int **natn_l;
int **ncn_l;
int **atv_ijk_l;

double *****OLP_l;
double *****H_l;
double ******DM_l;

or 

int *WhatSpecies_r;
int *Spe_Total_CNO_r;
int *Spe_Total_NO_r;
int *FNAN_r;
int **natn_r;
int **ncn_r;
int **atv_ijk_r;

double *****OLP_r;
double *****H_r;
double ******DM_r;



*****************************************************************/
 /*
  e.g. Overlap_Band, which gives hints of data to be saved
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
  GA_AN = M2G[MA_AN];  
  wanA = WhatSpecies[GA_AN]; int* WhatSpecies[atomnum+1]
  tnoA = Spe_Total_CNO[wanA]; int* Spe_Total_CNO[SpeciesNum]
  Anum = MP[GA_AN];   int *MP, neglect!

  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
  GB_AN = natn[GA_AN][LB_AN]; int** natn[atomnum+1][Max_FSNAN*ScaleSize+1]
  Rn = ncn[GA_AN][LB_AN];  int** ncn[atomnum+1][Max_FSNAN*ScaleSize+1]
  wanB = WhatSpecies[GB_AN]; 
  tnoB = Spe_Total_CNO[wanB];

  l1 = atv_ijk[Rn][1];  int** atv_ijk[TCpyCell+1][4];
  l2 = atv_ijk[Rn][2];
  l3 = atv_ijk[Rn][3];
  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

  si = sin(2.0*PI*kRn);
  co = cos(2.0*PI*kRn);
  Bnum = MP[GB_AN];
  for (i=0; i<tnoA; i++){
  for (j=0; j<tnoB; j++){
  s = OLP[MA_AN][LB_AN][i][j]; double****
  size: OLP[4]
  [Matomnum+MatomnumF+MatomnumS+1]
  [FNAN[Gc_AN]+1]
  [Spe_Total_NO[Cwan]]
  [Spe_Total_NO[Hwan]]

  int *Spe_Total_NO  Spe_Total_NO[SpeciesNum]
*/

/***************************************************************************/

static double ScaleSize_t;
static int  SpinP_switch_t, atomnum_t, SpeciesNum_t, Max_FSNAN_t, TCpyCell_t, Matomnum_t, MatomnumF_t, MatomnumS_t;
/* revised by Y. Xiao for Noncollinear NEGF calculations */
static double *****iHNL_t;
/* until here by Y. Xiao for Noncollinear NEGF calculations */
static int *WhatSpecies_t;
static int *Spe_Total_CNO_t;
static int *Spe_Total_NO_t;
static int *FNAN_t;
static int **natn_t;
static int **ncn_t;
static int **atv_ijk_t;
static double Grid_Origin_t[4];
static double **Gxyz_t;

static double *****OLP_t;
static double *****H_t;
static double ******DM_t;

static double *dDen_Grid_t;
static double *dVHart_Grid_t;

static int Ngrid1_t, Ngrid2_t, Ngrid3_t;
static int  Num_Cells0_t; 
static double tv_t[4][4];
static double gtv_t[4][4];
static double ScaleSize_t;
static int TCpyCell_t;

static double ChemP_t; 




int TRAN_Input_HKS( MPI_Comm comm1, char *fileHKS)
{
  FILE *fp;
  int i_vec[100],i2_vec[2];
  double *d_vec;
  int i,id,j;
  int size1,size;
  int *ia_vec;
  int Gc_AN, k,Mc_AN, tno0, Cwan, h_AN, tno1, Gh_AN, Hwan,m,N;
  int numprocs,myid,ID;
 
  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  if (myid==Host_ID)  printf("<TRAN_Input_HKS>\n");fflush(stdout);
  /* make a filename */ 
  if ( (fp=fopen(fileHKS,"r"))==NULL) {
    printf("can not open %s\n",fileHKS);
    exit(0);
  }

  if (print_stdout){
    printf("file=%s\n",fileHKS);
  }

  /* save data to the file (*fp) */

  /* parameter to allocate memory */

  fread(i_vec,sizeof(int),2,fp);
  /*i_vec[i++]=1;  major version */
  /*i_vec[i++]=0;  minor version*/

  fread(i_vec,sizeof(int),12,fp); 

  i=0;
  SpinP_switch_t = i_vec[i++];
  atomnum_t      = i_vec[i++];
  SpeciesNum_t   = i_vec[i++];
  Max_FSNAN_t    = i_vec[i++];
  TCpyCell_t     = i_vec[i++];
  Matomnum_t     = i_vec[i++];
  MatomnumF_t    = i_vec[i++];
  MatomnumS_t    = i_vec[i++];
  Ngrid1_t       = i_vec[i++];
  Ngrid2_t       = i_vec[i++];
  Ngrid3_t       = i_vec[i++];
  Num_Cells0_t   = i_vec[i++];

#if 0

  if (print_stdout){
    printf("%d %d %d %d %d %d\n",SpinP_switch_t,atomnum_t,SpeciesNum_t,
	   Max_FSNAN_t, TCpyCell_t, Matomnum_t );
    printf("%d %d %d %d %d %d\n",MatomnumF_t,MatomnumS_t,Ngrid1_t,
	   Ngrid2_t,Ngrid3_t,Num_Cells0_t);
  }

#endif

  /* allocation of arrays */

  d_vec = (double*)malloc(sizeof(double)*(3*(atomnum_t+1)+100));
  
  Gxyz_t = (double**)malloc(sizeof(double*)*(atomnum_t+1));
  for (k=0; k<(atomnum_t+1); k++){
    Gxyz_t[k] = (double*)malloc(sizeof(double)*4);
  }

  /* read d_vec */

  fread(d_vec,sizeof(double),19+3+3*atomnum_t+1,fp);

  i=0;
  ScaleSize_t=d_vec[i++];
  for (j=1;j<=3;j++)  {
    for (k=1;k<=3;k++) {
      tv_t[j][k]=d_vec[i++];
    }
  }
  for (j=1;j<=3;j++)  {
    for (k=1;k<=3;k++) {
      gtv_t[j][k]=d_vec[i++];
    }
  }
  for (j=1;j<=3;j++) {
    Grid_Origin_t[j]= d_vec[i++];
  }

  for (k=1; k<=atomnum_t; k++) {
    for (j=1; j<=3; j++) {
      Gxyz_t[k][j] = d_vec[i++];
    }
  }

  ChemP_t = d_vec[i++];

  /* freeing of d_vec */
  free(d_vec);

  if (print_stdout){
    printf("Grid_Origin=%lf %lf %lf\n",Grid_Origin_t[1],Grid_Origin_t[2],Grid_Origin_t[3]);
    printf("Gxyz=%lf %lf %lf\n",Gxyz_t[1][1],Gxyz_t[1][2],Gxyz_t[1][3]);
    printf("ChemP=%le\n",ChemP_t);
  }

#if 0
  if (print_stdout){
    printf("Scale =%lf\n",ScaleSize_t);
  }
#endif


  /*  data in arrays */

  WhatSpecies_t = (int*)malloc(sizeof(int)*(atomnum_t+1));
  fread(WhatSpecies_t, sizeof(int), atomnum_t+1, fp);

  Spe_Total_CNO_t = (int*)malloc(sizeof(int)*SpeciesNum_t);
  fread(Spe_Total_CNO_t, sizeof(int), SpeciesNum_t, fp);

  Spe_Total_NO_t = (int*)malloc(sizeof(int)*SpeciesNum_t);
  fread(Spe_Total_NO_t,sizeof(int),SpeciesNum_t,fp);

  FNAN_t = (int*)malloc(sizeof(int)*(atomnum_t+1));
  fread(FNAN_t,sizeof(int),atomnum_t+1,fp);

  size1 = (int)Max_FSNAN_t*ScaleSize_t+1;
  natn_t = (int**)malloc(sizeof(int*)*(atomnum_t+1) );
  for (i=0; i<= atomnum_t; i++) {
    natn_t[i] = (int*)malloc(sizeof(int)*size1);
    fread(natn_t[i],sizeof(int),size1,fp);
  }

  ncn_t = (int**)malloc(sizeof(int*)*(atomnum_t+1) );
  for (i=0; i<= atomnum_t; i++) {
    ncn_t[i] = (int*)malloc(sizeof(int)*size1);
    fread(ncn_t[i],sizeof(int),size1,fp);
  }

#if 0
  if (print_stdout){
    printf("ncn\n");
    for (i=0;i<=atomnum_t;i++) {
      for (j=0;j<size1;j++) {
	printf("%d ",ncn_t[i][j]);
      }
      printf("\n");
    }
  }
#endif
 
  /*  printf("atv_ijk\n"); */

  size1 = (TCpyCell_t+1)*4;
  ia_vec = (int*)malloc(sizeof(int)*size1);
  id=0;
  fread(ia_vec,sizeof(int),size1,fp);

  atv_ijk_t = (int**)malloc(sizeof(int*)*(TCpyCell_t+1));
  for (i=0; i<TCpyCell_t+1; i++) {
    atv_ijk_t[i] = (int*)malloc(sizeof(int)*4);
    for (j=0; j<=3; j++) {

      atv_ijk_t[i][j] = ia_vec[id++];
      /*      printf("%d ",atv_ijk_t[i][j]); */
    }

    /*    printf("\n"); */
  }
  free(ia_vec);

  /* OLP,  this is complex */

  OLP_t = (double*****)malloc(sizeof(double****)*4);
  for (k=0; k<4 ; k++){
    OLP_t[k] = (double****)malloc(sizeof(double***)*(atomnum_t+1)); 
    for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = Mc_AN;
        Cwan = WhatSpecies_t[Gc_AN];
        tno0 = Spe_Total_NO_t[Cwan];  
      }    

      OLP_t[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN_t[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn_t[Gc_AN][h_AN];
          Hwan = WhatSpecies_t[Gh_AN];
          tno1 = Spe_Total_NO_t[Hwan];
        } 

        OLP_t[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          OLP_t[k][Mc_AN][h_AN][i]=(double*)malloc(sizeof(double)*tno1);
          fread(OLP_t[k][Mc_AN][h_AN][i],sizeof(double),tno1,fp);
	  /*  printf("%le ",OLP_t[k][Mc_AN][h_AN][i][0]);  */

        }
      }
    }
  }

 
#if 0
  /*debug*/
  int GA_AN, wanA, tnoA, Anum; 
  int LB_AN, GB_AN, wanB, tnoB; 
  int *MP,num_e; 
  TRAN_set_MP(0, atomnum_t, WhatSpecies_t, Spe_Total_CNO_t, &num_e, MP);
  MP = (int*)malloc(sizeof(int)*(num_e+1));
  TRAN_set_MP(1, atomnum_t, WhatSpecies_t, Spe_Total_CNO_t, &num_e, MP);

  printf("OLP_e\n");
  for (GA_AN=1;GA_AN<=atomnum_t; GA_AN++) {
    wanA = WhatSpecies_t[GA_AN];
    tnoA = Spe_Total_CNO_t[wanA];
    Anum = MP[GA_AN];
    for (LB_AN=0; LB_AN<=FNAN_t[GA_AN]; LB_AN++){
      GB_AN = natn_t[GA_AN][LB_AN];
      wanB = WhatSpecies_t[GB_AN];
      tnoB = Spe_Total_CNO_t[wanB];
      printf("GA_AN=%d LB_AN=%d GB_AN=%d tnoA=%d tnoB=%d S=%lf\n",
	     GA_AN,LB_AN, GB_AN,tnoA,tnoB, OLP_t[0][GA_AN][LB_AN][0][0]);
    }
  }
  free(MP);
#endif


  /* H */
  H_t = (double*****)malloc(sizeof(double****)*(SpinP_switch_t+1)); 
  for (k=0; k<=SpinP_switch_t; k++){
    H_t[k] = (double****)malloc(sizeof(double***)*(atomnum_t+1)); 
    for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = Mc_AN;
        Cwan = WhatSpecies_t[Gc_AN];
        tno0 = Spe_Total_NO_t[Cwan];  
      }    

      H_t[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN_t[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn_t[Gc_AN][h_AN];
          Hwan = WhatSpecies_t[Gh_AN];
          tno1 = Spe_Total_NO_t[Hwan];
        } 

        H_t[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          H_t[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
          fread(H_t[k][Mc_AN][h_AN][i],sizeof(double),tno1,fp);
        }
      }
    }
  }

/* revised by Y. Xiao for Noncollinear NEGF calculations */
  /* iHNL */
 if(SpinP_switch_t==3) {
  
  iHNL_t = (double*****)malloc(sizeof(double****)*(2+1));
  for (k=0; k<=2; k++){
    iHNL_t[k] = (double****)malloc(sizeof(double***)*(atomnum_t+1));
    for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = Mc_AN;
        Cwan = WhatSpecies_t[Gc_AN];
        tno0 = Spe_Total_NO_t[Cwan];
      }

      iHNL_t[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN_t[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;
        }
        else{
          Gh_AN = natn_t[Gc_AN][h_AN];
          Hwan = WhatSpecies_t[Gh_AN];
          tno1 = Spe_Total_NO_t[Hwan];
        }

        iHNL_t[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
        for (i=0; i<tno0; i++){
          iHNL_t[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
          fread(iHNL_t[k][Mc_AN][h_AN][i],sizeof(double),tno1,fp);
        }
      }
    }
  }
} /* if (SpinP_switch==3) */
/* until here by Y. Xiao for Noncollinear NEGF calculations */

  /* DM */
  DM_t = (double******)malloc(sizeof(double*****)*1);
  for (m=0; m<1; m++){
    DM_t[m] = (double*****)malloc(sizeof(double****)*(SpinP_switch_t+1)); 
    for (k=0; k<=SpinP_switch_t; k++){
      DM_t[m][k] = (double****)malloc(sizeof(double***)*(atomnum_t+1)); 
      FNAN_t[0] = 0;
      for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){

        if (Mc_AN==0){
          Gc_AN = 0;
          tno0 = 1;
        }
        else{
          Gc_AN = Mc_AN;
          Cwan = WhatSpecies_t[Gc_AN];
          tno0 = Spe_Total_NO_t[Cwan];  
        }    

        DM_t[m][k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN_t[Gc_AN]+1));
        for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){

          if (Mc_AN==0){
            tno1 = 1;  
          }
          else{
            Gh_AN = natn_t[Gc_AN][h_AN];
            Hwan = WhatSpecies_t[Gh_AN];
            tno1 = Spe_Total_NO_t[Hwan];
          } 

          DM_t[m][k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
          for (i=0; i<tno0; i++){
            DM_t[m][k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
            fread(DM_t[m][k][Mc_AN][h_AN][i],sizeof(double),tno1,fp);
          }
        }
      }
    }
  }

  N = Ngrid1_t*Ngrid2_t*Ngrid3_t;
  dDen_Grid_t=(double*)malloc(sizeof(double)*N);
  fread(dDen_Grid_t, sizeof(double), N,fp);

  /* dVHart_Grid */

  dVHart_Grid_t=(double*)malloc(sizeof(double)*N);
  fread(dVHart_Grid_t,sizeof(double), N,fp);

  /* debug */
  if (print_stdout){
    printf("the last of dVHart_Grid=%20.10le\n",dVHart_Grid_t[N-1]);
  }

#ifdef DEBUG 
  { double R[4];
  R[1]=0.0; R[2]=0.0; R[3]=0.0;
  TRAN_Print_Grid_z("dVHart_e2", Grid_Origin_t, gtv_t,
		    Ngrid1_t,Ngrid2_t,0, Ngrid3_t-1,R,
		    dVHart_Grid_t);
  }
#endif


  fclose(fp);

  return 1;
}







int TRAN_RestartFile(MPI_Comm comm1, char *mode, char *position,char *filepath, char *filename)
{
  int i,side,j,k;
  char fileHKS[YOUSO10];
  int numprocs,myid,ID;
 
  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  if (myid==Host_ID){
    printf("<TRAN_RestartFile called, mode=%s pos=%s>\n", mode,position);fflush(stdout);
  }

  if (strcasecmp(position,"left")==0) {
    side=0;
  } else if (strcasecmp(position,"right")==0) {
    side=1;
  } else {
    printf("mode=%s position=%s, not supported\n",mode, position);
    return 1;
  }

  if (strcasecmp(position,"left")==0) {
    sprintf(fileHKS,"%s%s",filepath,filename);

  }
  else if (strcasecmp(position,"right")==0) {
    sprintf(fileHKS,"%s%s",filepath,filename);
  }
  else {
    printf("***ERROR***\n\nmode=%s position=%s, not supported\n\n\n",mode, position);
    return 1;
  }

  /*******************
          write 
  *******************/

  if (strcasecmp(mode,"write")==0) {

    TRAN_Output_HKS(fileHKS);
  }

  /*******************
          read
  *******************/

  else if (strcasecmp(mode,"read")==0) {

    TRAN_Input_HKS(comm1, fileHKS);

#if 0
    if (strcasecmp(position,"left")==0) {
      side=0;
    } else if (strcasecmp(position,"right")==0) {
      side=1;
    } else {
      printf("mode=%s position=%s, not supported\n",mode, position);
      return 1;
    }
#endif

    ScaleSize_e[side]=ScaleSize_t;
    SpinP_switch_e[side]=SpinP_switch_t;
    atomnum_e[side]=atomnum_t;
    SpeciesNum_e[side]=SpeciesNum_t;
    Max_FSNAN_e[side]=Max_FSNAN_t;
    TCpyCell_e[side]=TCpyCell_t;
    Matomnum_e[side]=Matomnum_t;
    MatomnumF_e[side]=MatomnumF_t;
    MatomnumS_e[side]=MatomnumS_t;
    WhatSpecies_e[side]=WhatSpecies_t;
    Spe_Total_CNO_e[side]=Spe_Total_CNO_t;
    Spe_Total_NO_e[side]=Spe_Total_NO_t;
    FNAN_e[side]=FNAN_t;
    natn_e[side]=natn_t;
    ncn_e[side]=ncn_t;
    atv_ijk_e[side]=atv_ijk_t;
    OLP_e[side]=OLP_t;
    H_e[side]=H_t;
    DM_e[side]=DM_t;
    dDen_Grid_e[side]=dDen_Grid_t;
    dVHart_Grid_e[side] = dVHart_Grid_t;
/* revised by Y. Xiao for Noncollinear NEGF calculations */
    iHNL_e[side]=iHNL_t;
/* until here by Y. Xiao for Noncollinear NEGF calculations */
    Ngrid1_e[side]=Ngrid1_t;
    Ngrid2_e[side]=Ngrid2_t;
    Ngrid3_e[side]=Ngrid3_t;
    Num_Cells0_e[side]=Num_Cells0_t; 

    for (j=1;j<=3;j++)  {
      for (k=1;k<=3;k++) {
	tv_e[side][j][k]= tv_t[j][k];
      }
    }
    for (j=1;j<=3;j++)  {
      for (k=1;k<=3;k++) {
	gtv_e[side][j][k]=gtv_t[j][k];
      }
    }

    for (i=1;i<=3;i++)  {
      Grid_Origin_e[side][i]=Grid_Origin_t[i];
    }

    Gxyz_e[side] = Gxyz_t;
    ChemP_e[side] = ChemP_t; 

    /*
    for (j=0; j<Ngrid1_e[side]*Ngrid2_e[side]*Ngrid3_e[side]; j++){
      printf("side=%2d i=%2d dDen=%15.12f\n",side,j,dDen_Grid_e[side][j]);fflush(stdout);
    }
    */

  }

  return 0;
}
	  
  
