/**********************************************************************
  Population_Analysis_Wannier.c:

  Population analysis based on atomic orbitals resembling Wannier functions

  Log of Population_Analysis_Wannier.c:

     03/June/2017  Developed by Taisuke Ozaki
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
static void Opt_Object_Function(int Nto, int *atom_targeted_orbital, int *orbital_targeted_orbital);
static void Calc_Gradient( int step, int dim, int Nto, 
                    double **acoes, double *gra, double *occupation, 
                    double *similarity, double **overlap_wanniers, double **SDS,
                    int *MP, int *atom_targeted_orbital, int *orbital_targeted_orbital);

int *dim_SRS;
double *****RhoPA;
double ****SPA;
double ***SRS;

void Population_Analysis_Wannier(char *argv[])
{
  int spe,n,l,i,j,max_occupied_N,Number_VPS;
  int mul,ind,k,m,num,Nto,dim,ita,ocupied_flag;
  int NVPS[20],LVPS[20];
  int **Num_L_channels,*Num_vOrbs;
  int *atom_targeted_orbital;
  int *orbital_targeted_orbital;
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
      for(i=0; i<Number_VPS; i++){
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
              Set_Inf_SndRcv_for_PopAnal
  ***************************************************/

  Copy_OLP_DM();

  /***************************************************
          optimization of the object function
  ***************************************************/

  Opt_Object_Function(Nto, atom_targeted_orbital,orbital_targeted_orbital);

  MPI_Finalize();
  exit(0);

  /***************************************************
                     free arrays
  ***************************************************/

  for (spe=0; spe<real_SpeciesNum; spe++){
    free(Num_L_channels[spe]);
  }
  free(Num_L_channels);

  free(Num_vOrbs);

  free(atom_targeted_orbital);
  free(orbital_targeted_orbital);
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
        Bnum += Spe_Total_CNO[wanA];
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

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = SPA[ih][kl][m][n];
	      Rho_DC[Anum+m][Bnum+n] = RhoPA[0][ih][kl][m][n];
	    }
	  }
	}

	else{

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
	      S_DC[Anum+m][Bnum+n] = 0.0;
	      Rho_DC[Anum+m][Bnum+n] = 0.0;
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
          sum += S_DC[i][k]*Rho_DC[k][j]; 
	}

	C[i][j] = sum;
      }
    }

    for (i=1; i<=dim_SRS[Mc_AN]; i++){
      for (j=1; j<=dim_SRS[Mc_AN]; j++){

        sum = 0.0;
        for (k=1; k<=NUM; k++){
          sum += C[i][k]*S_DC[k][j]; 
	}

	SRS[Mc_AN][i-1][j-1] = sum;
      }
    }

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





void Opt_Object_Function(int Nto, int *atom_targeted_orbital, int *orbital_targeted_orbital)
{
  int dim,wanA,i,j,k;
  int Gc_AN,gamma,step,ian,ih,kl;
  int *MP;
  double **acoes,*gra;
  double **overlap_wanniers;
  double *occupation;
  double *similarity;
  double **Sv,**Dv,**SDS;
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

  acoes = (double**)malloc(sizeof(double*)*(dim+2));
  for (i=0; i<(dim+2); i++){
    acoes[i] = (double*)malloc(sizeof(double)*(dim+2));
    for (j=0; j<(dim+2); j++){
      acoes[i][j] = 0.0;
    }
  }

  overlap_wanniers = (double**)malloc(sizeof(double*)*(dim+2));
  for (i=0; i<(dim+2); i++){
    overlap_wanniers[i] = (double*)malloc(sizeof(double)*(dim+2));
    for (j=0; j<(dim+2); j++){
      overlap_wanniers[i][j] = 0.0;
    }
  }

  gra = (double*)malloc(sizeof(double)*(dim+2)*(dim+2));
  for (i=0; i<(dim+2)*(dim+2); i++){
    gra[i] = 0.0;
  }

  occupation = (double*)malloc(sizeof(double)*(dim+2));
  similarity = (double*)malloc(sizeof(double)*(dim+2));

  for (i=0; i<(dim+2); i++) occupation[i] = 0.0;
  for (i=0; i<(dim+2); i++) similarity[i] = 0.0;

  Sv = (double**)malloc(sizeof(double*)*(dim+2));
  for (i=0; i<(dim+2); i++){
    Sv[i] = (double*)malloc(sizeof(double)*(dim+2));
    for (j=0; j<(dim+2); j++){
      Sv[i][j] = 0.0;
    }
  }

  Dv = (double**)malloc(sizeof(double*)*(dim+2));
  for (i=0; i<(dim+2); i++){
    Dv[i] = (double*)malloc(sizeof(double)*(dim+2));
    for (j=0; j<(dim+2); j++){
      Dv[i][j] = 0.0;
    }
  }

  SDS = (double**)malloc(sizeof(double*)*(dim+2));
  for (i=0; i<(dim+2); i++){
    SDS[i] = (double*)malloc(sizeof(double)*(dim+2));
    for (j=0; j<(dim+2); j++){
      SDS[i][j] = 0.0;
    }
  }

  /***************************************************
                 set up S and rho matrices
  ***************************************************/

  Overlap_Cluster(OLP[0],Sv,MP);
  Overlap_Cluster(DM[0][0],Dv,MP);

  /*
  printf("Sv\n");
  for (i=1; i<=dim; i++){
    for (j=1; j<=dim; j++){
      printf("%10.5f ",Sv[i][j]);
    }
    printf("\n");
  }

  printf("Dv\n");
  for (i=1; i<=dim; i++){
    for (j=1; j<=dim; j++){
      printf("%10.5f ",Dv[i][j]);
    }
    printf("\n");
  }

  printf("D12\n");
  for (i=0; i<4; i++){
    for (j=0; j<4; j++){
      printf("%10.5f ",DM[0][0][1][1][i][j]);
    }
    printf("\n");
  }
  */

  for (i=1; i<=dim; i++){
    for (j=1; j<=dim; j++){

      sum = 0.0;
      for (k=1; k<=dim; k++){
        sum += Sv[i][k]*Dv[k][j];
      }     
      SDS[i][j] = sum; 
    }
  }

  for (i=1; i<=dim; i++){
    for (j=1; j<=dim; j++){

      sum = 0.0;
      for (k=1; k<=dim; k++){
        sum += SDS[i][k]*Sv[k][j];
      }     
      Dv[i][j] = sum; 
    }
  }

  for (i=1; i<=dim; i++){
    for (j=1; j<=dim; j++){
      SDS[i-1][j-1] = Dv[i][j];
    }
  }

  /***************************************************
                 initialize coefficients
  ***************************************************/
  
  for (i=0; i<dim; i++){

    Gc_AN = atom_targeted_orbital[i];
    k = orbital_targeted_orbital[i];
    j = MP[Gc_AN] - 1 + k;

    acoes[i][j] = 1.0;
  }

  /*
  acoes[0][3] = acoes[0][3] + 0.0001;
  */

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

  MPI_Finalize();
  exit(0); 
  */

  /***********************************************
                   optimization 
  ***********************************************/

  for (step=1; step<2000; step++){

    Calc_Gradient( step,dim,Nto,
                   acoes,gra,occupation,similarity,overlap_wanniers,SDS,
                   MP,atom_targeted_orbital,orbital_targeted_orbital);

    for (gamma=0; gamma<dim; gamma++){
      for (i=0; i<dim; i++){

       if (step<100)
          acoes[gamma][i] = acoes[gamma][i] - 0.0004*gra[gamma*dim+i]; 
       else 
          acoes[gamma][i] = acoes[gamma][i] - 0.0010*gra[gamma*dim+i]; 

      }
    }
  }


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

  /***********************************************
                 freeing of arrays
  ***********************************************/

  free(MP);

  for (i=0; i<(dim+2); i++){
    free(acoes[i]);
  }
  free(acoes);

  for (i=0; i<(dim+2); i++){
    free(overlap_wanniers[i]);
  }
  free(overlap_wanniers);

  free(gra);
  free(occupation);
  free(similarity);

  for (i=0; i<(dim+2); i++){
    free(Sv[i]);
  }
  free(Sv);

  for (i=0; i<(dim+2); i++){
    free(Dv[i]);
  }
  free(Dv);

  for (i=0; i<(dim+2); i++){
    free(SDS[i]);
  }
  free(SDS);

}


void Calc_Gradient( int step, int dim, int Nto, 
                    double **acoes, double *gra, double *occupation, 
                    double *similarity, double **overlap_wanniers, double **SDS,
                    int *MP, int *atom_targeted_orbital, int *orbital_targeted_orbital)
{
  int i,j,k,l,m,Gc_AN;
  int ta,to,gamma,mu,nu,TGc_AN,TOrb;
  int wanA,wanB,GA_AN,GB_AN,h_AN,MA_AN;
  int Mc_AN,hA_AN,hB_AN;
  double sum1,sum2,sum3;
  double F1,F2,F3,norm;
  double factor1,factor2,factor3;
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

  /* initialize gra */
  for (i=0; i<(dim+2)*(dim+2); i++){
    gra[i] = 0.0;
  }

  /***************************************************************
     1st gradients for occupied and unoccupied Wannier functions
     constratint on occupation  
  ***************************************************************/

  /* calculation of the 1st part of the object function */

  F1 = 0.0;

  for (mu=0; mu<Nto; mu++){ /* !! */
 
    sum1 = 0.0; 

    for (i=0; i<dim; i++){

      sum2 = 0.0;

      for (j=0; j<dim; j++){
        sum1 += acoes[mu][i]*acoes[mu][j]*SDS[i][j];
        sum2 += acoes[mu][j]*SDS[i][j];
      }

      if (mu<Nto)  
	gra[mu*dim+i] = -factor1*sum2;
      else 
	gra[mu*dim+i] = factor1*sum2;

    }

    if (mu<Nto){
      F1 -= factor1*sum1;
    }
    else{
      F1 += factor1*sum1;
    }

    occupation[mu] = sum1;
  }

  norm = 0.0;
  for (gamma=0; gamma<dim; gamma++){
    for (i=0; i<dim; i++){

      norm += gra[gamma*dim+i]*gra[gamma*dim+i]; 
      /*
      printf("WWW1 gamma=%2d i=%2d gra=%15.12f\n",gamma,i,gra[gamma*dim+i]);
      */

    }
  }

  /*
  printf("ABC1 F1=%15.12f norm=%15.12f\n",F1,norm);
  MPI_Finalize();
  exit(0);
  */

  /*********************************************************
     2nd gradients for occupied Wannier functions
     constratint on similarity for atomic orbitals
  *********************************************************/

  F2 = 0.0;

  for (gamma=0; gamma<Nto; gamma++){ /* !! */

    TGc_AN = atom_targeted_orbital[gamma];
    TOrb = orbital_targeted_orbital[gamma];

    ID = G2ID[TGc_AN];
    Mc_AN = F_G2M[TGc_AN];

    sum1 = 0.0;

    if (ID==myid){

      for (hA_AN=0; hA_AN<=FNAN[TGc_AN]; hA_AN++){

        GA_AN = natn[TGc_AN][hA_AN];
        wanA  = WhatSpecies[GA_AN];

        for (i=0; i<Spe_Total_CNO[wanA]; i++){

          sum2 = 0.0;

	  for (hB_AN=0; hB_AN<=FNAN[TGc_AN]; hB_AN++){

	    GB_AN = natn[TGc_AN][hB_AN];
	    wanB  = WhatSpecies[GB_AN];

	    for (j=0; j<Spe_Total_CNO[wanB]; j++){

              l = MP[GA_AN] - 1 + i;
              m = MP[GB_AN] - 1 + j;

              sum1 += acoes[gamma][l]*acoes[gamma][m]*OLP[0][Mc_AN][hA_AN][TOrb][i]*OLP[0][Mc_AN][hB_AN][TOrb][j];
              sum2 += acoes[gamma][m]*OLP[0][Mc_AN][hA_AN][TOrb][i]*OLP[0][Mc_AN][hB_AN][TOrb][j];
	    }
	  }

          l = MP[GA_AN] - 1 + i;
          gra[gamma*dim+l] -= factor2*sum2;

	} /* i */
      } /* hA_AN */ 
    } /* if (ID==myid) */

    similarity[gamma] = sum1;
    F2 -= factor2*sum1;

  } /* gamma */


  norm = 0.0;
  for (gamma=0; gamma<dim; gamma++){
    for (i=0; i<dim; i++){

      norm += gra[gamma*dim+i]*gra[gamma*dim+i]; 
      /*
      printf("WWW1 gamma=%2d i=%2d gra=%15.12f\n",gamma,i,gra[gamma*dim+i]);
      */

    }
  }

  /*
  printf("ABC1 F2=%18.15f norm=%18.15f\n",F2,norm);
  MPI_Finalize();
  exit(0);
  */

  /***************************************************************
     3rd gradients for occupied and unoccupied Wannier functions
     constratint on orthonormalization
  ***************************************************************/

  /* calculations of <mu/nu> */

  F3 = 0.0;

  for (mu=0; mu<Nto; mu++){ /* !! */
    for (nu=0; nu<Nto; nu++){ /* !! */

      sum1 = 0.0; 
   
      /*  !!!!  this should be parallelized */ 
 
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

	GA_AN = M2G[MA_AN];
	wanA  = WhatSpecies[GA_AN];

	for (hB_AN=0; hB_AN<=FNAN[GA_AN]; hB_AN++){
             
	  GB_AN = natn[GA_AN][hB_AN];
	  wanB  = WhatSpecies[GB_AN];

	  for (i=0; i<Spe_Total_CNO[wanA]; i++){
	    for (j=0; j<Spe_Total_CNO[wanB]; j++){

              m = MP[GA_AN] - 1 + i;
              l = MP[GB_AN] - 1 + j;

              sum1 += acoes[mu][m]*acoes[nu][l]*OLP[0][MA_AN][hB_AN][i][j];
	    }           
          }
	} 
      }

      if (mu==nu){
	F3 += factor3*(1.0 - sum1)*(1.0 - sum1);
      }
      else{
	F3 += factor3*(0.0 - sum1)*(0.0 - sum1);
      }

      overlap_wanniers[mu][nu] = sum1;        

    } /* nu */
  } /* mu */

  // printf("F3=%18.15f\n",F3); 

  /* calculation of gradients for the 3rd part */

  for (gamma=0; gamma<Nto; gamma++){ /* !! */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA  = WhatSpecies[GA_AN];

      for (i=0; i<Spe_Total_CNO[wanA]; i++){
        
        sum3 = 0.0;

        for (nu=0; nu<Nto; nu++){ /* !! */

          sum2 = 0.0; 

          for (hB_AN=0; hB_AN<=FNAN[GA_AN]; hB_AN++){
             
	    GB_AN = natn[GA_AN][hB_AN];
	    wanB  = WhatSpecies[GB_AN];

	    for (j=0; j<Spe_Total_CNO[wanB]; j++){

              m = MP[GA_AN] - 1 + i;
              l = MP[GB_AN] - 1 + j;

              sum2 += acoes[nu][l]*OLP[0][MA_AN][hB_AN][i][j];
	    }           
          }

          if (gamma==nu){
            sum3 += 2.0*(1.0 - overlap_wanniers[gamma][nu])*(-sum2);
	  }
          else{
            sum3 += 2.0*(0.0 - overlap_wanniers[gamma][nu])*(-sum2);
	  }

	} /* nu */

        l = MP[GA_AN] - 1 + i;
        gra[gamma*dim+l] += factor3*sum3;

      } /* i */
    } /* MA_AN */
  } /* gamma */


  norm = 0.0;
  for (gamma=0; gamma<dim; gamma++){
    for (i=0; i<dim; i++){

      norm += gra[gamma*dim+i]*gra[gamma*dim+i]; 
      /*
      printf("WWW1 gamma=%2d i=%2d gra=%15.12f\n",gamma,i,gra[gamma*dim+i]);
      */

    }
  }

  /*
  printf("F1=%18.15f F2=%18.15f F3=%18.15f F=%18.15f norm=%18.15f\n",F1,F2,F3,F1+F2+F3,norm); 
  for (gamma=0; gamma<dim; gamma++){
    for (i=0; i<dim; i++){
      printf("%16.12f ",gra[gamma*dim+i]); 
    }
    printf("\n");
  }
  MPI_Finalize();
  exit(0);

  */

  norm = 0.0;
  for (gamma=0; gamma<dim; gamma++){
    for (i=0; i<dim; i++){
      norm += gra[gamma*dim+i]*gra[gamma*dim+i];
    }
  }

  printf("step=%4d F1=%16.12f F2=%16.12f F3=%16.12f F=%16.12f norm=%16.12f\n",
          step,F1,F2,F3,F1+F2+F3,norm); 

  /*
  MPI_Finalize();
  exit(0);
  */

}


