/**********************************************************************
  Population_Analysis_Wannier.c:

  Population analysis based on atomic orbitals resembling Wannier functions

  Log of Population_Analysis_Wannier.c:

     03/June/2017 Started development by Taisuke Ozaki
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
 
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
#include <omp.h>

static void Set_Inf_SndRcv_for_PopAnal();
static void Copy_OLP_DM();
static void Opt_Object_Function( int Nto, int *atom_targeted_orbital, int *orbital_targeted_orbital, 
                          int **spe_targeted_orbital, int *Num_vOrbs);
static void Calc_Gradient( int step, int dim, int Nto, int ****RMI3,
                           double ****acoes, double ****gra, double **occupation, 
                           double **similarity, double ****overlap_wanniers, double ****b,
                           int *MP, int *atom_targeted_orbital, int *orbital_targeted_orbital, 
			   int **spe_targeted_orbital, int *Num_vOrbs);


int *dim_SRS;
double *****RhoPA;
double ****SPA;
double ***SRS;

#pragma optimization_level 1
void Population_Analysis_Wannier2(char *argv[])
{
  int spe,n,l,i,j,max_occupied_N,Number_VPS;
  int mul,ind,k,m,num,Nto,dim,ita,ocupied_flag;
  int tno0,tno1,h_AN,Hwan,Gh_AN,Cwan,Gc_AN,wanA;
  int Bnum,Mc_AN,Gi;
  int NVPS[20],LVPS[20];
  int **Num_L_channels,*Num_vOrbs;
  int *atom_targeted_orbital;
  int *orbital_targeted_orbital;
  int **spe_targeted_orbital;
  double OcpN[7][8];
  double dum1,dum2;
  char ExtVPS[YOUSO10] = ".vps";
  char DirVPS[YOUSO10];
  char FN_VPS[YOUSO10];
  FILE *fp;

  /***************************************************
               allocation of arrays
  ***************************************************/

  Num_L_channels = (int**)malloc(sizeof(int*)*real_SpeciesNum);
  for (spe=0; spe<real_SpeciesNum; spe++){
    Num_L_channels[spe] = (int*)malloc(sizeof(int)*10);
    for (l=0; l<10; l++) Num_L_channels[spe][l] = 0;
  }

  Num_vOrbs = (int*)malloc(sizeof(int)*real_SpeciesNum);

  /***************************************************
                   analysis of species 
  ***************************************************/

  sprintf(DirVPS,"%s/VPS/",DFT_DATA_PATH);

  for (spe=0; spe<real_SpeciesNum; spe++){

    for (n=1; n<=6; n++){
      for (l=0; l<n; l++){
	OcpN[n][l] = 0.0;
      }
    }

    fnjoint2(DirVPS,SpeVPS[spe],ExtVPS,FN_VPS);    

    /* open a vps file */     
    input_open(FN_VPS);

    input_int("max.occupied.N",&max_occupied_N,0);
    ocupied_flag = 0;
  
    if (max_occupied_N==0){
      input_int("max.ocupied.N",&max_occupied_N,0);
      ocupied_flag = 1;
    }

    if (ocupied_flag==0){

      if (fp=input_find("<occupied.electrons") ) {

	for (n=1; n<=max_occupied_N; n++){
	  fscanf(fp,"%d",&i);
	  if (i!=n){
	    printf("\n**** Error message in Population_Analysis_Wannier ****\n");
	    printf("!! Format error in occupied.electrons !!\n");
	    MPI_Finalize();
	    exit(0); 
	  }
	  for (l=0; l<n; l++){
	    fscanf(fp,"%lf",&OcpN[n][l]);
	  }
	}

	if (!input_last("occupied.electrons>")) {
	  /* format error */
	  printf("\n**** Error message in Population_Analysis_Wannier ****\n");
	  printf("!! Format error for occupied.electrons !!\n");
	  MPI_Finalize();
	  exit(0); 
	}
      }
    }

    else {

      if (fp=input_find("<ocupied.electrons") ) {

	for (n=1; n<=max_occupied_N; n++){
	  fscanf(fp,"%d",&i);
	  if (i!=n){
	    printf("\n**** Error message in Population_Analysis_Wannier ****\n");
	    printf("!! Format error in occupied.electrons !!\n");
	    MPI_Finalize();
	    exit(0); 
	  }

	  for (l=0; l<n; l++){
	    fscanf(fp,"%lf",&OcpN[n][l]);
	  }
	}

	if (!input_last("ocupied.electrons>")) {
	  /* format error */
	  printf("\n**** Error message in Population_Analysis_Wannier ****\n");
	  printf("!! Format error for occupied.electrons !!\n");
	  MPI_Finalize();
	  exit(0); 
	}
      }
    }

    input_int("number.vps",&Number_VPS,0);

    if (fp=input_find("<pseudo.NandL") ) {
      for (i=0; i<Number_VPS; i++){
	fscanf(fp,"%d %d %d %lf %lf",&j,&NVPS[i],&LVPS[i],&dum1,&dum2);
      }
    }

    for (i=0; i<Number_VPS; i++){
      n = NVPS[i];
      l = LVPS[i];
      if (0.0<OcpN[n][l]) Num_L_channels[spe][l]++;
    }    

    num = 0;
    for (l=0; l<=5; l++){
      num += Num_L_channels[spe][l]*(2*l+1);
    }

    Num_vOrbs[spe] = num;

    /* close a vps file */     
    input_close();

  } /* spe */

  spe_targeted_orbital = (int**)malloc(sizeof(int*)*real_SpeciesNum);
  for (spe=0; spe<real_SpeciesNum; spe++){
    spe_targeted_orbital[spe] = (int*)malloc(sizeof(int)*Num_vOrbs[spe]);
  }

  for (spe=0; spe<real_SpeciesNum; spe++){

    ind = 0;
    ita = 0;
    for (l=0; l<=Supported_MaxL; l++){

      j = 0;

      for (k=0; k<Num_L_channels[spe][l]; k++){
        for (m=0; m<(2*l+1); m++){
          spe_targeted_orbital[spe][ita] = ind + j; 
          ita++;
          j++;
	}
      }

      for (mul=0; mul<Spe_Num_CBasis[spe][l]; mul++){
        for (m=0; m<(2*l+1); m++){
          ind++; 
	}
      }
    }
  }

  /***************************************************
               set up targeted orbitals
  ***************************************************/

  dim = 0;
  for (i=1; i<=atomnum; i++){
    spe  = WhatSpecies[i];
    dim += Spe_Total_CNO[spe];
  }

  num = 0;
  for (i=1; i<=atomnum; i++){
    spe = WhatSpecies[i];
    num += Num_vOrbs[spe]; 
  }

  Nto = num; 

  atom_targeted_orbital = (int*)malloc(sizeof(int)*dim);
  orbital_targeted_orbital = (int*)malloc(sizeof(int)*dim);

  /* targeted orbitals */

  ita = 0;
  for (i=1; i<=atomnum; i++){
    spe = WhatSpecies[i];

    ind = 0;

    for (l=0; l<=Supported_MaxL; l++){

      if (Spe_Num_CBasis[spe][l]<Num_L_channels[spe][l]){
	printf("\n**** Error message in Population_Analysis_Wannier ****\n");
	printf("  Spe_Num_CBasis[spe][l] is smaller than Num_L_channels[spe][l].\n");
        MPI_Finalize();
        exit(0); 
      }

      j = 0;

      for (k=0; k<Num_L_channels[spe][l]; k++){
        for (m=0; m<(2*l+1); m++){
          atom_targeted_orbital[ita] = i;
          orbital_targeted_orbital[ita] = ind + j; 
          ita++;
          j++;
        }
      }

      for (mul=0; mul<Spe_Num_CBasis[spe][l]; mul++){
        for (m=0; m<(2*l+1); m++){
          ind++; 
	}
      }
    }
  }

  /* other orbitals */

  for (i=1; i<=atomnum; i++){
    spe = WhatSpecies[i];

    ind = 0;
    for (l=0; l<=Supported_MaxL; l++){

      if (Spe_Num_CBasis[spe][l]<Num_L_channels[spe][l]){
	printf("\n**** Error message in Population_Analysis_Wannier ****\n");
	printf("  Spe_Num_CBasis[spe][l] is smaller than Num_L_channels[spe][l].\n");
        MPI_Finalize();
        exit(0); 
      }

      for (mul=0; mul<Num_L_channels[spe][l]; mul++){
        for (m=0; m<(2*l+1); m++){
          ind++; 
	}
      }

      for (mul=Num_L_channels[spe][l]; mul<Spe_Num_CBasis[spe][l]; mul++){
        for (m=0; m<(2*l+1); m++){
          atom_targeted_orbital[ita] = i;
          orbital_targeted_orbital[ita] = ind;
          ita++;
          ind++;
        }
      }

    }
  }

  /***************************************************
                     Copy_OLP_DM
  ***************************************************/

  Copy_OLP_DM();

  /***************************************************
          optimization of the object function
  ***************************************************/

  Opt_Object_Function(Nto, atom_targeted_orbital,orbital_targeted_orbital,
                      spe_targeted_orbital,Num_vOrbs);

  /***************************************************
                     freeing of arrays
  ***************************************************/

  for (spe=0; spe<real_SpeciesNum; spe++){
    free(Num_L_channels[spe]);
  }
  free(Num_L_channels);

  free(Num_vOrbs);

  free(atom_targeted_orbital);
  free(orbital_targeted_orbital);

  for (spe=0; spe<real_SpeciesNum; spe++){
    free(spe_targeted_orbital[spe]);
  }
  free(spe_targeted_orbital);

  free(dim_SRS);

  for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    if (Mc_AN==0){
      Bnum = 1;
    }
    else{
      Bnum = 0;
      for (i=0; i<=FNAN[Gc_AN]; i++){
        Gi = natn[Gc_AN][i];
        wanA = WhatSpecies[Gi];
        Bnum += Spe_Total_NO[wanA];
      }
    }

    for (i=0; i<Bnum; i++){
      free(SRS[Mc_AN][i]);
    }
    free(SRS[Mc_AN]);
  }
  free(SRS);

  for (k=0; k<=SpinP_switch; k++){
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
          free(RhoPA[k][Mc_AN][h_AN][i]);
        }
        free(RhoPA[k][Mc_AN][h_AN]);
      }
      free(RhoPA[k][Mc_AN]);
    }
    free(RhoPA[k]);
  }
  free(RhoPA);

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
	free(SPA[Mc_AN][h_AN][i]);
      }
      free(SPA[Mc_AN][h_AN]);
    }
    free(SPA[Mc_AN]);
  }
  free(SPA);

}



void Copy_OLP_DM()
{
  int spin,Gh_AN,h_AN,tno0,tno1,tno2,Cwan,Gc_AN;
  int n,num,size1,size2,i,j,k,Hwan,Mc_AN,Anum,NUM;
  int Bnum,Gi,wanA,jan,jg,m,ian,n2,ig,ih,kl,wan;
  int *MP;
  int *Snd_H_Size,*Rcv_H_Size;
  int *Snd_S_Size,*Rcv_S_Size;
  double *tmp_array;
  double *tmp_array2;
  double sum;
  double **S_DC,**Rho_DC,**C;
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  Snd_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_H_Size = (int*)malloc(sizeof(int)*numprocs);
  Snd_S_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_S_Size = (int*)malloc(sizeof(int)*numprocs);

  /*******************************
    allocation of RhoPA and SPA
  ********************************/

  /* RhoPA */

  RhoPA = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 
  for (k=0; k<=SpinP_switch; k++){
    RhoPA[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
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

      RhoPA[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        RhoPA[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          RhoPA[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
          for (j=0; j<tno1; j++)  RhoPA[k][Mc_AN][h_AN][i][j] = 0.0;
        }
      }
    }
  }

  /* SPA */

  SPA = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+MatomnumS+1)); 
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

    SPA[Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      if (Mc_AN==0){
	tno1 = 1;  
      }
      else{
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_NO[Hwan];
      } 

      SPA[Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
      for (i=0; i<tno0; i++){
	SPA[Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	for (j=0; j<tno1; j++) SPA[Mc_AN][h_AN][i][j] = 0.0;
      }
    }
  }

  /* initialize RhoPA and SPA */

  for (k=0; k<=SpinP_switch; k++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = S_M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];  
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno1 = Spe_Total_NO[Hwan];

        for (i=0; i<tno0; i++){
          for (j=0; j<tno1; j++){
            RhoPA[k][Mc_AN][h_AN][i][j] = DM[0][k][Mc_AN][h_AN][i][j];
            SPA[Mc_AN][h_AN][i][j] = OLP[0][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }
  }

  /****************************************************
   MPI: RhoPA
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              size1 += tno1*tno2;
	    }
          }
	}

        Snd_H_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_H_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_H_Size[IDR] = size2;
      }
      else{
        Rcv_H_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_H_Size[IDS] = 0;
      Rcv_H_Size[IDR] = 0;
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

        size1 = Snd_H_Size[IDS];

        /* allocation of array */

        tmp_array = (double*)malloc(sizeof(double)*size1);

        /* multidimentional array to vector array */

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  tmp_array[num] = RhoPA[spin][Mc_AN][h_AN][i][j];
                  num++;
                } 
              } 
	    }
          }
	}

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);

      }

      /*****************************
         receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){

        size2 = Rcv_H_Size[IDR];
        
        /* allocation of array */
        tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (spin=0; spin<=SpinP_switch; spin++){
          Mc_AN = S_TopMAN[IDR] - 1;  /* S_TopMAN should be used. */
          for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN]; 
            tno1 = Spe_Total_CNO[Cwan];

            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
              Gh_AN = natn[Gc_AN][h_AN];        
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_CNO[Hwan];
              for (i=0; i<tno1; i++){
                for (j=0; j<tno2; j++){
                  RhoPA[spin][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
		}
	      }
	    }
	  }        
	}

        /* freeing of array */
        free(tmp_array2);
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
        MPI_Wait(&request,&stat);
        free(tmp_array); /* freeing of array */
      }
    }
  }

  /****************************************************
   MPI

   OLP0
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	size1 = 0;
	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		size1++; 
	      } 
	    } 
	  }
	}

	Snd_S_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_S_Size[IDS] = 0;
      }

      /* receiving of size of data */
 
      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_S_Size[IDR] = size2;
      }
      else{
	Rcv_S_Size[IDR] = 0;
      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0) MPI_Wait(&request,&stat);
    }
    else{
      Snd_S_Size[IDS] = 0;
      Rcv_S_Size[IDR] = 0;
    }
  }

  /***********************************
               data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
                sending of data 
      *****************************/

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){

	size1 = Snd_S_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;

	for (n=0; n<(F_Snd_Num[IDS]+S_Snd_Num[IDS]); n++){
	  Mc_AN = Snd_MAN[IDS][n];
	  Gc_AN = Snd_GAN[IDS][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		tmp_array[num] = SPA[Mc_AN][h_AN][i][j];
		num++;
	      } 
	    } 
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
           receiving of block data
      *****************************/

      if ((F_Rcv_Num[IDR]+S_Rcv_Num[IDR])!=0){
          
	size2 = Rcv_S_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
         
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	num = 0;
	Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
	for (n=0; n<(F_Rcv_Num[IDR]+S_Rcv_Num[IDR]); n++){
	  Mc_AN++;
	  Gc_AN = Rcv_GAN[IDR][n];
	  Cwan = WhatSpecies[Gc_AN]; 
	  tno1 = Spe_Total_CNO[Cwan];

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];        
	    Hwan = WhatSpecies[Gh_AN];
	    tno2 = Spe_Total_CNO[Hwan];
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
		SPA[Mc_AN][h_AN][i][j] = tmp_array2[num];
		num++;
	      }
	    }
	  }
	}        

	/* freeing of array */
	free(tmp_array2);

      }

      if ((F_Snd_Num[IDS]+S_Snd_Num[IDS])!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array); /* freeing of array */
      }
    }
  }

  /****************************************************
              calculation of S*Rho*S                 
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[2]);
  dim_SRS = (int*)malloc(sizeof(int)*(Matomnum+1));

  SRS = (double***)malloc(sizeof(double**)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    if (Mc_AN==0){
      Bnum = 1;
    }
    else{
      Bnum = 0;
      for (i=0; i<=FNAN[Gc_AN]; i++){
        Gi = natn[Gc_AN][i];
        wanA = WhatSpecies[Gi];
        Bnum += Spe_Total_NO[wanA];
      }
    }

    SRS[Mc_AN] = (double**)malloc(sizeof(double*)*Bnum);
    for (i=0; i<Bnum; i++){
      SRS[Mc_AN][i] = (double*)malloc(sizeof(double)*Bnum);
    }

    dim_SRS[Mc_AN] = Bnum;
  }

  /* loop for Mc_AN */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];

    Anum = 1;
    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      MP[i] = Anum;
      Gi = natn[Gc_AN][i];
      wanA = WhatSpecies[Gi];
      Anum += Spe_Total_CNO[wanA];
    }
    NUM = Anum - 1;
    n2 = NUM + 3;

    S_DC = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      S_DC[i] = (double*)malloc(sizeof(double)*n2);
    }

    Rho_DC = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      Rho_DC[i] = (double*)malloc(sizeof(double)*n2);
    }

    C = (double**)malloc(sizeof(double*)*n2);
    for (i=0; i<n2; i++){
      C[i] = (double*)malloc(sizeof(double)*n2);
    }

    /***********************************************
     construct cluster full matrices of rho
           and overlap for the atom Mc_AN             
    ***********************************************/

    for (i=0; i<=(FNAN[Gc_AN]+SNAN[Gc_AN]); i++){
      ig = natn[Gc_AN][i];
      ian = Spe_Total_CNO[WhatSpecies[ig]];
      Anum = MP[i];
      ih = S_G2M[ig];

      for (j=0; j<=(FNAN[Gc_AN]+SNAN[Gc_AN]); j++){

	kl = RMI1[Mc_AN][i][j];
	jg = natn[Gc_AN][j];
	jan = Spe_Total_CNO[WhatSpecies[jg]];
	Bnum = MP[j];

	if (0<=kl){

	  if (SpinP_switch==0){
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		S_DC[Anum+m][Bnum+n] = SPA[ih][kl][m][n];
		Rho_DC[Bnum+n][Anum+m] = 2.0*RhoPA[0][ih][kl][m][n]; /* change [][] */
	      }
	    }
	  }
          else{
	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){
		S_DC[Anum+m][Bnum+n] = SPA[ih][kl][m][n];
		Rho_DC[Bnum+n][Anum+m] = RhoPA[0][ih][kl][m][n] + RhoPA[1][ih][kl][m][n]; /* change [][] */
	      }
	    }
	  }
	}

	else{

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = 0.0;
	      Rho_DC[Bnum+n][Anum+m] = 0.0; /* change [][] */
	    }
	  }
	}
      }
    }

    /***********************************************
     calculation of S*Rho*S
    ***********************************************/

    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){

        sum = 0.0;
        for (k=1; k<=NUM; k++){
          sum += S_DC[i][k]*Rho_DC[j][k]; 
	}

	C[i][j] = sum;
      }
    }

    for (i=1; i<=dim_SRS[Mc_AN]; i++){
      for (j=1; j<=dim_SRS[Mc_AN]; j++){

        sum = 0.0;
        for (k=1; k<=NUM; k++){
          sum += C[i][k]*S_DC[j][k]; 
	}

	SRS[Mc_AN][i-1][j-1] = sum;
      }
    }

    /*
    printf("SRS Mc_AN=%2d dim=%2d\n",Mc_AN,dim_SRS[Mc_AN]);
    for (i=0; i<dim_SRS[Mc_AN]; i++){
      for (j=0; j<dim_SRS[Mc_AN]; j++){
        printf("%10.5f ",SRS[Mc_AN][i][j]);
      }
      printf("\n");
    }
    */

    /* freeing of arrays */

    for (i=0; i<n2; i++){
      free(S_DC[i]);
    }
    free(S_DC);

    for (i=0; i<n2; i++){
      free(Rho_DC[i]);
    }

    for (i=0; i<n2; i++){
      free(C[i]);
    }
    free(C);
  }

  /****************************************************
                   freeing of arrays:
  ****************************************************/

  free(Snd_H_Size);
  free(Rcv_H_Size);
  free(Snd_S_Size);
  free(Rcv_S_Size);
}




#pragma optimization_level 1
void Opt_Object_Function( int Nto, int *atom_targeted_orbital, int *orbital_targeted_orbital, 
                          int **spe_targeted_orbital, int *Num_vOrbs)
{
  int dim,wanA,i,j,k;
  int Gc_AN,gamma,step,ian,ih,kl;
  int tno0,tno1,Cwan,Hwan,Gh_AN,h_AN,Mc_AN,spe;
  int lA1,lA2,lA3,RA,GA_AN,GB_AN,hA_AN;
  int lD1,lD2,lD3,RD,MC_AN,lC1,lC2,lC3,RC,GC_AN;
  int GD_AN,hD_AN,hC_AN,po,lB1,lB2,lB3,hB_AN,RB;
  int *MP,****RMI3;
  double ****acoes,****gra;
  double ****overlap_wanniers;
  double **occupation;
  double **similarity;
  double ****bb;
  double norm,sum;

  /***********************************************
      allocation of arrays for the first world 
  ***********************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);

  dim = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = dim;
    wanA  = WhatSpecies[i];
    dim += Spe_Total_CNO[wanA];
  }
  dim = dim - 1; 

  acoes = (double****)malloc(sizeof(double***)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
    }

    acoes[Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      acoes[Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

      if (Mc_AN==0){
	tno1 = 1;  
      }
      else{
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_NO[Hwan];
      } 
      
      for (i=0; i<tno0; i++){
        acoes[Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
        for (j=0; j<tno1; j++) acoes[Mc_AN][h_AN][i][j] = 0.0;
      }
    }
  }

  gra = (double****)malloc(sizeof(double***)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
    }

    gra[Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      gra[Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);

      if (Mc_AN==0){
	tno1 = 1;  
      }
      else{
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_NO[Hwan];
      } 
      
      for (i=0; i<tno0; i++){
        gra[Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
        for (j=0; j<tno1; j++) gra[Mc_AN][h_AN][i][j] = 0.0;
      }
    }
  }

  occupation = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
    }

    occupation[Mc_AN] = (double*)malloc(sizeof(double)*tno0);
    for (i=0; i<tno0; i++) occupation[Mc_AN][i] = 0.0;
  }

  similarity = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
    }

    similarity[Mc_AN] = (double*)malloc(sizeof(double)*tno0);
    for (i=0; i<tno0; i++) similarity[Mc_AN][i] = 0.0;
  }

  overlap_wanniers = (double****)malloc(sizeof(double***)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
      SNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
    }

    overlap_wanniers[Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+SNAN[Gc_AN]+1));

    for (h_AN=0; h_AN<(FNAN[Gc_AN]+SNAN[Gc_AN]+1); h_AN++){

      if (Mc_AN==0){
        tno1 = 1;
      }
      else{
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno1 = Num_vOrbs[Hwan];
      }

      overlap_wanniers[Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
      for (i=0; i<tno0; i++){
        overlap_wanniers[Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
      }
    }
  }

  bb = (double****)malloc(sizeof(double***)*List_YOUSO[8]);
  for (h_AN=0; h_AN<List_YOUSO[8]; h_AN++){
    bb[h_AN] = (double***)malloc(sizeof(double**)*List_YOUSO[7]);
    for (i=0; i<List_YOUSO[7]; i++){
      bb[h_AN][i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
      for (j=0; j<List_YOUSO[7]; j++){
        bb[h_AN][i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }
    }
  }

  /***************************************************
      check the connectivity from hA_AN to hC_AN 
      and construct RMI3
  ***************************************************/

  RMI3 = (int****)malloc(sizeof(int***)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
    RMI3[Mc_AN] = (int***)malloc(sizeof(int**)*List_YOUSO[8]);
    for (i=0; i<List_YOUSO[8]; i++){
      RMI3[Mc_AN][i] = (int**)malloc(sizeof(int*)*List_YOUSO[2]);
      for (j=0; j<List_YOUSO[2]; j++){
        RMI3[Mc_AN][i][j] = (int*)malloc(sizeof(int)*List_YOUSO[8]);
        for (k=0; k<List_YOUSO[8]; k++) RMI3[Mc_AN][i][j][k] = -1;
      }
    }
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    Gc_AN = M2G[Mc_AN];

    for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){ /* neighbors of the center 1 */

      GA_AN = natn[Gc_AN][hA_AN];
      RA = ncn[Gc_AN][hA_AN];
      lA1 = atv_ijk[RA][1];
      lA2 = atv_ijk[RA][2];
      lA3 = atv_ijk[RA][3];

      for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

	GB_AN = natn[Gc_AN][hB_AN];
	RB = ncn[Gc_AN][hB_AN];
	lB1 = atv_ijk[RB][1];
	lB2 = atv_ijk[RB][2];
	lB3 = atv_ijk[RB][3];

	for (hC_AN=0; hC_AN<=FNAN[GB_AN]; hC_AN++){ /* neighbors of the center 2 */

	  GC_AN = natn[GB_AN][hC_AN];
	  RC = ncn[GB_AN][hC_AN];
	  MC_AN = S_G2M[GC_AN];
	  lC1 = lB1 + atv_ijk[RC][1];
	  lC2 = lB2 + atv_ijk[RC][2];
	  lC3 = lB3 + atv_ijk[RC][3];

	  if (MC_AN!=-1){
            
	    po = 0; 
	    for (hD_AN=0; hD_AN<=FNAN[GA_AN]; hD_AN++){ /* neighbors of the h_AN */

	      GD_AN = natn[GA_AN][hD_AN];
	      RD = ncn[GA_AN][hD_AN];
	      lD1 = lA1 + atv_ijk[RD][1];
	      lD2 = lA2 + atv_ijk[RD][2];
	      lD3 = lA3 + atv_ijk[RD][3];

	      if (lC1==lD1 && lC2==lD2 && lC3==lD3 && GC_AN==GD_AN){
		RMI3[Mc_AN][hA_AN][hB_AN][hC_AN] = hD_AN;
		po = 1; 
	      }
	      if (po==1) break;
	    }
	  }
	}
      }
    }
  }

  /*
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    for (hB_AN=0; hB_AN<=1; hB_AN++){
      printf("ZZZ Mc_AN=%2d hB_AN=%2d\n",Mc_AN,hB_AN);
      for (hA_AN=0; hA_AN<=1; hA_AN++){
	for (hC_AN=0; hC_AN<=1; hC_AN++){
	  printf("%2d ",RMI3[Mc_AN][hA_AN][hB_AN][hC_AN]);
	}
	printf("\n");
      }
    }
  }

  MPI_Finalize();
  exit(0);
  */

  /***************************************************
                initialize coefficients
  ***************************************************/
  
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Num_vOrbs[Cwan];

    for (i=0; i<tno0; i++){
      j = spe_targeted_orbital[Cwan][i];
      acoes[Mc_AN][0][i][j] = 1.0;
    }
  }

  /***********************************************
                   optimization 
  ***********************************************/

  for (step=1; step<2000; step++){

    Calc_Gradient( step,dim,Nto,RMI3,
                   acoes,gra,occupation,similarity,overlap_wanniers, bb,
                   MP,atom_targeted_orbital,orbital_targeted_orbital,spe_targeted_orbital,Num_vOrbs);

    /* update acoes by a steepest decent method */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_NO[Hwan];
	for (i=0; i<tno0; i++){
	  for (j=0; j<tno1; j++){

            if (step<100)
  	      acoes[Mc_AN][h_AN][i][j] = acoes[Mc_AN][h_AN][i][j] - 0.0004*gra[Mc_AN][h_AN][i][j];
            else 
  	      acoes[Mc_AN][h_AN][i][j] = acoes[Mc_AN][h_AN][i][j] - 0.0010*gra[Mc_AN][h_AN][i][j];

	  }
	}
      }
    }
  }

  sum = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (i=0; i<Num_vOrbs[Cwan]; i++){ 

      printf("Mc_AN=%2d i=%2d occupation=%15.12f similarity=%15.12f\n",
              Mc_AN,i,occupation[Mc_AN][i],similarity[Mc_AN][i]);

      sum += occupation[Mc_AN][i];
    }
  }

  printf("Sum of occupations = %15.12f\n",sum);


  printf("overlap_wanniers\n");
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    int mu,nu,wanB,tnoB;

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */

      for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

	GB_AN = natn[Gc_AN][hB_AN];
	wanB  = WhatSpecies[GB_AN];
	tnoB  = Spe_Total_NO[wanB];

	for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */
          printf("%6.4f ",overlap_wanniers[Mc_AN][hB_AN][mu][nu]);
	}
      }

      printf("\n");
    }
  }


  /*
  {
  int gamma,i;

  for (gamma=0; gamma<dim; gamma++){
    for (i=0; i<dim; i++){
      printf("%6.3f ",acoes[gamma][i]); 
    }
    printf("\n");
  }
  }

  sum = 0.0;
  for (i=0; i<dim; i++){
    sum += occupation[i]; 
    printf("i=%2d occupation=%15.12f similarity=%15.12f\n",i,occupation[i],similarity[i]);
  } 
  printf("sum of occups = %15.12f\n",sum);

  {
    int mu,nu;

  printf("overlap_wanniers\n");
  for (mu=0; mu<Nto; mu++){
    for (nu=0; nu<Nto; nu++){
      printf("%15.12f ",overlap_wanniers[mu][nu]); 
    }
    printf("\n");
  }
  }
  */



  /***********************************************
                 freeing of arrays
  ***********************************************/

  for (Mc_AN=0; Mc_AN<(Matomnum+1); Mc_AN++){
    for (i=0; i<List_YOUSO[8]; i++){
      for (j=0; j<List_YOUSO[2]; j++){
        free(RMI3[Mc_AN][i][j]);
      }
      free(RMI3[Mc_AN][i]);
    }
    free(RMI3[Mc_AN]);
  }
  free(RMI3);

  free(MP);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
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
        free(acoes[Mc_AN][h_AN][i]);
      }
      free(acoes[Mc_AN][h_AN]);
    }
    free(acoes[Mc_AN]);
  }
  free(acoes);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
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
        free(gra[Mc_AN][h_AN][i]);
      }
      free(gra[Mc_AN][h_AN]);
    }
    free(gra[Mc_AN]);
  }
  free(gra);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(occupation[Mc_AN]);
  }
  free(occupation);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(similarity[Mc_AN]);
  }
  free(similarity);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
      FNAN[0] = 0;
      SNAN[0] = 0;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Num_vOrbs[Cwan];
    }

    for (h_AN=0; h_AN<(FNAN[Gc_AN]+SNAN[Gc_AN]+1); h_AN++){
      for (i=0; i<tno0; i++){
        free(overlap_wanniers[Mc_AN][h_AN][i]);
      }
      free(overlap_wanniers[Mc_AN][h_AN]);
    }
    free(overlap_wanniers[Mc_AN]);
  }
  free(overlap_wanniers);

  for (h_AN=0; h_AN<List_YOUSO[8]; h_AN++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (j=0; j<List_YOUSO[7]; j++){
        free(bb[h_AN][i][j]);
      }
      free(bb[h_AN][i]);
    }
    free(bb[h_AN]);
  }
  free(bb);

}

#pragma optimization_level 1
void Calc_Gradient( int step, int dim, int Nto, int ****RMI3,
                    double ****acoes, double ****gra, double **occupation, 
                    double **similarity, double ****overlap_wanniers, double ****bb,
                    int *MP, int *atom_targeted_orbital, int *orbital_targeted_orbital, 
                    int **spe_targeted_orbital, int *Num_vOrbs)
{
  int i,j,k,l,m,Gc_AN;
  int ta,to,gamma,mu,nu,TGc_AN,TOrb;
  int wanA,wanB,GA_AN,GB_AN,h_AN,MA_AN;
  int tnoA,tnoB,iA,jB,num,orb,orbi,Gh_AN,tno1;
  int Hwan,Cwan,tno0,spe;
  int Mc_AN,hA_AN,hB_AN;
  int MC_AN,wanC,tnoC,kAC,kBC;
  int orb1,orb2,MB_AN,RB,hC_AN,GC_AN; 
  double sum,sum1,sum2,sum3;
  double F1,F2,F3,norm,tmp;
  double factor1,factor2,factor3;
  int *LMP;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  factor1 = 0.2;
  factor2 = 0.2;
  factor3 = 100.0;

  /*
  factor1 = 0.0;
  factor2 = 0.0;
  factor3 = 1.0;
  */

  /* initialize arrays */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Num_vOrbs[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_NO[Hwan];
      for (i=0; i<tno0; i++){
        for (j=0; j<tno1; j++){
          gra[Mc_AN][h_AN][i][j] = 0.0;
	}
      }
    }
  }

  /***************************************************************
     1st gradients for occupied Wannier functions
     constratint on occupation  
  ***************************************************************/

  /* calculation of the 1st part of the object function */

  LMP = (int*)malloc(sizeof(int)*List_YOUSO[2]);

  F1 = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    num = 0;
    for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){
      GA_AN = natn[Gc_AN][hA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_NO[wanA];
      LMP[hA_AN] = num;
      num += tnoA;
    }

    for (orb=0; orb<Num_vOrbs[Cwan]; orb++){ /* Mc_AN+orb -> mu */

      sum1 = 0.0;

      for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){
	GA_AN = natn[Gc_AN][hA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_NO[wanA];
        iA = LMP[hA_AN];

        for (i=0; i<tnoA; i++){

	  sum2 = 0.0;  

	  for (hB_AN=0; hB_AN<=FNAN[Gc_AN]; hB_AN++){
	    GB_AN = natn[Gc_AN][hB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_NO[wanB];
	    jB = LMP[hB_AN];

            for (j=0; j<tnoB; j++){
	      sum1 += acoes[Mc_AN][hA_AN][orb][i]*acoes[Mc_AN][hB_AN][orb][j]*SRS[Mc_AN][iA+i][jB+j];
	      sum2 += acoes[Mc_AN][hB_AN][orb][j]*SRS[Mc_AN][iA+i][jB+j];
  	    }
	  }

          gra[Mc_AN][hA_AN][orb][i] = -factor1*sum2;  

	}
      }

      occupation[Mc_AN][orb] = sum1;

      F1 -= factor1*sum1;
    }
  }


  norm = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Num_vOrbs[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_NO[Hwan];
      for (i=0; i<tno0; i++){
        for (j=0; j<tno1; j++){

          norm += gra[Mc_AN][h_AN][i][j]*gra[Mc_AN][h_AN][i][j];
	  /*
          printf("WWW1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d gra=%15.12f\n",Mc_AN,h_AN,i,j,gra[Mc_AN][h_AN][i][j]);
	  */

	}
      }
    }
  }

  /*
  printf("ABC2 F1=%15.12f norm=%15.12f\n",F1,norm);
  MPI_Finalize();
  exit(0);
  */

  /*********************************************************
     2nd gradients for occupied Wannier functions
     constratint on similarity for atomic orbitals
  *********************************************************/

  F2 = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (orb=0; orb<Num_vOrbs[Cwan]; orb++){ /* Mc_AN+orb -> mu */

      orbi = spe_targeted_orbital[Cwan][orb];       
      sum1 = 0.0;

      for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){
	GA_AN = natn[Gc_AN][hA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_NO[wanA];

        for (i=0; i<tnoA; i++){

	  sum2 = 0.0;  

	  for (hB_AN=0; hB_AN<=FNAN[Gc_AN]; hB_AN++){
	    GB_AN = natn[Gc_AN][hB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_NO[wanB];

            for (j=0; j<tnoB; j++){

	      sum1 +=  acoes[Mc_AN][hA_AN][orb ][i]* acoes[Mc_AN][hB_AN][orb ][j]
                     *OLP[0][Mc_AN][hA_AN][orbi][i]*OLP[0][Mc_AN][hB_AN][orbi][j];

	      sum2 +=  acoes[Mc_AN][hB_AN][orb ][j]
                     *OLP[0][Mc_AN][hA_AN][orbi][i]*OLP[0][Mc_AN][hB_AN][orbi][j];

  	    }
	  }

          gra[Mc_AN][hA_AN][orb][i] -= factor2*sum2;

	}
      }

      similarity[Mc_AN][orb] = sum1;
      F2 -= factor2*sum1;

    }
  }

  norm = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Num_vOrbs[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_NO[Hwan];
      for (i=0; i<tno0; i++){
        for (j=0; j<tno1; j++){

          norm += gra[Mc_AN][h_AN][i][j]*gra[Mc_AN][h_AN][i][j];
	  /*
          printf("WWW1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d gra=%15.12f\n",Mc_AN,h_AN,i,j,gra[Mc_AN][h_AN][i][j]);
	  */

	}
      }
    }
  }

  /*
  printf("ABC2 F2=%18.15f norm=%18.15f\n",F2,norm);
  MPI_Finalize();
  exit(0);
  */

  /***************************************************************
     3rd gradients for occupied Wannier functions
     constratint on orthonormalization
  ***************************************************************/

  /* initialize overlap_wannier */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

      GB_AN = natn[Gc_AN][hB_AN];
      wanB  = WhatSpecies[GB_AN];
      tnoB  = Spe_Total_NO[wanB];

      for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */
	for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */

	  overlap_wanniers[Mc_AN][hB_AN][mu][nu] = 0.0;
	}
      }
    }
  }

  /* calculations of <mu/nu> */

  if (0){

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */

      for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){ /* neighbors of the center 1 */

	GA_AN = natn[Gc_AN][hA_AN];
        MA_AN = S_G2M[GA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_NO[wanA];

        for (i=0; i<tnoA; i++){ /* basis functions on the center 1 */

	  for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

	    RB = ncn[Gc_AN][hB_AN];
	    GB_AN = natn[Gc_AN][hB_AN];
	    MB_AN = S_G2M[GB_AN];
	    wanB  = WhatSpecies[GB_AN];
	    tnoB  = Spe_Total_NO[wanB];

	    for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */

	      for (hC_AN=0; hC_AN<=FNAN[GB_AN]; hC_AN++){ /* neighbors of the center 2 */

		GC_AN = natn[GB_AN][hC_AN];
		MC_AN = S_G2M[GC_AN];

		if (MC_AN!=-1){

		  wanC  = WhatSpecies[GC_AN];
		  tnoC  = Spe_Total_NO[wanC];

		  kAC = RMI3[Mc_AN][hA_AN][hB_AN][hC_AN];

		  if (0<=kAC){

		    for (j=0; j<tnoC; j++){

		      overlap_wanniers[Mc_AN][hB_AN][mu][nu] += acoes[Mc_AN][hA_AN][mu][i]
                                                               *acoes[MB_AN][hC_AN][nu][j]
                                                                 *OLP[0][MA_AN][kAC][i][j];
		    }
		  }
		}
	      } /* hC_AN */

	    } /* nu */
	  } /* hB_AN */
	} /* i */
      } /* hA_AN */
    } /* mu */
  } /* Mc_AN */

  /* gradients contributed from F3 */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */

      for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){ /* neighbors of the center 1 */

	GA_AN = natn[Gc_AN][hA_AN];
        MA_AN = S_G2M[GA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_NO[wanA];

        for (i=0; i<tnoA; i++){ /* basis functions on the center 1 */

          sum1 = 0.0;

	  for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

	    RB = ncn[Gc_AN][hB_AN];
	    GB_AN = natn[Gc_AN][hB_AN];
	    MB_AN = S_G2M[GB_AN];
	    wanB  = WhatSpecies[GB_AN];
	    tnoB  = Spe_Total_NO[wanB];

	    for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */

	      sum2 = 0.0;

	      for (hC_AN=0; hC_AN<=FNAN[GB_AN]; hC_AN++){ /* neighbors of the center 2 */

		GC_AN = natn[GB_AN][hC_AN];
		MC_AN = S_G2M[GC_AN];

		if (MC_AN!=-1){

		  wanC  = WhatSpecies[GC_AN];
		  tnoC  = Spe_Total_NO[wanC];

		  kAC = RMI3[Mc_AN][hA_AN][hB_AN][hC_AN];

		  if (0<=kAC){

		    for (j=0; j<tnoC; j++){
		      sum2 += acoes[MB_AN][hC_AN][nu][j]*OLP[0][MA_AN][kAC][i][j];
		    }
		  }
		}
	      }

	      if (hB_AN==0 && mu==nu){
		sum1 += 2.0*(1.0 - overlap_wanniers[Mc_AN][hB_AN][mu][nu])*(-sum2);
	      }
	      else{
		sum1 += 2.0*(0.0 - overlap_wanniers[Mc_AN][hB_AN][mu][nu])*(-sum2);
	      }

	    } /* nu */
	  } /* hB_AN */

          gra[Mc_AN][hA_AN][mu][i] += factor3*sum1;

	} /* i */
      } /* hA_AN */
    } /* mu */
  } /* Mc_AN */

  }

  if (1){

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    /* loop for hB_AN */

    for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

      RB = ncn[Gc_AN][hB_AN];
      GB_AN = natn[Gc_AN][hB_AN];
      MB_AN = S_G2M[GB_AN];
      wanB  = WhatSpecies[GB_AN];

      /* initialize bb */

      for (h_AN=0; h_AN<List_YOUSO[8]; h_AN++){
	for (i=0; i<List_YOUSO[7]; i++){
	  for (j=0; j<List_YOUSO[7]; j++){
	    for (k=0; k<List_YOUSO[7]; k++){
	      bb[h_AN][i][j][k] = 0.0;
	    }
	  }
	}
      }

      for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){ /* neighbors of the center 1 */

	GA_AN = natn[Gc_AN][hA_AN];
	MA_AN = S_G2M[GA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_NO[wanA];

	for (hC_AN=0; hC_AN<=FNAN[GB_AN]; hC_AN++){ /* neighbors of the center 2 */

	  GC_AN = natn[GB_AN][hC_AN];
	  MC_AN = S_G2M[GC_AN];

	  if (MC_AN!=-1){

	    wanC  = WhatSpecies[GC_AN];
	    tnoC  = Spe_Total_NO[wanC];

	    kAC = RMI3[Mc_AN][hA_AN][hB_AN][hC_AN];

	    if (0<=kAC){

	      for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */
		for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */

		  for (i=0; i<tnoA; i++){ /* basis functions on the center 1 */
		    for (j=0; j<tnoC; j++){

                      tmp = acoes[MB_AN][hC_AN][nu][j]*OLP[0][MA_AN][kAC][i][j];  
		      overlap_wanniers[Mc_AN][hB_AN][mu][nu] += acoes[Mc_AN][hA_AN][mu][i]*tmp;
                      bb[hA_AN][mu][nu][i] += tmp;
		    }
		  }
		}
	      }
	    } 
	  }
 
	} /* hC_AN */
      } /* hA_AN */

      /* calculation of gradients from F3 */

      for (hA_AN=0; hA_AN<=FNAN[Gc_AN]; hA_AN++){ /* neighbors of the center 1 */

	GA_AN = natn[Gc_AN][hA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_NO[wanA];

	for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */
	  for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */
            tmp = 2.0*factor3*overlap_wanniers[Mc_AN][hB_AN][mu][nu];
	    for (i=0; i<tnoA; i++){ /* basis functions on the center 1 */
	      gra[Mc_AN][hA_AN][mu][i] += tmp*bb[hA_AN][mu][nu][i]; 
	    }  
	  }
	}

        if (hB_AN==0){
	  for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */
            tmp = 2.0*factor3; 
	    for (i=0; i<tnoA; i++){ /* basis functions on the center 1 */
	      gra[Mc_AN][hA_AN][mu][i] -= tmp*bb[hA_AN][mu][mu][i]; 
	    }
	  }
	}
      }

    } /* hB_AN */ 
  } /* Mc_AN */ 

  }

  /* calculation of F3 */

  F3 = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){ /* center 1 */

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (hB_AN=0; hB_AN<=(FNAN[Gc_AN]+SNAN[Gc_AN]); hB_AN++){ /* center 2 */

      GB_AN = natn[Gc_AN][hB_AN];
      wanB  = WhatSpecies[GB_AN];
      tnoB  = Spe_Total_NO[wanB];

      for (mu=0; mu<Num_vOrbs[Cwan]; mu++){ /* Wannier functions on the center 1 */
	for (nu=0; nu<Num_vOrbs[wanB]; nu++){ /* Wannier functions on the center 2 */

          if (hB_AN==0 && mu==nu){

            tmp = overlap_wanniers[Mc_AN][hB_AN][mu][nu];
            F3 += factor3*(1.0 - tmp)*(1.0 - tmp);
	  }
          else{

            tmp = overlap_wanniers[Mc_AN][hB_AN][mu][nu];
            F3 += factor3*tmp*tmp;
	  }
	}
      }
    }
  }


  norm = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Num_vOrbs[Cwan];

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_NO[Hwan];
      for (i=0; i<tno0; i++){
        for (j=0; j<tno1; j++){

          norm += gra[Mc_AN][h_AN][i][j]*gra[Mc_AN][h_AN][i][j];
	  /*
          printf("WWW1 Mc_AN=%2d h_AN=%2d i=%2d j=%2d gra=%15.12f\n",Mc_AN,h_AN,i,j,gra[Mc_AN][h_AN][i][j]);
	  */

	}
      }
    }
  }

  /*
  printf("ABC3 F3=%18.15f norm=%18.15f\n",F3,norm);
  MPI_Finalize();
  exit(0);
  */

  printf("step=%4d F1=%16.12f F2=%16.12f F3=%16.12f F=%16.12f norm=%16.12f\n",
          step,F1,F2,F3,F1+F2+F3,norm); fflush(stdout);

  /*
  MPI_Finalize();
  exit(0);
  */

  /* freeing of arrays */

  free(LMP);

}
