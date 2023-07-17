/**********************************************************************
  RestartFileDFT.c:

   RestartFileDFT.c is a subroutine to make a set of restart files whose 
   file name is 'filename', and to read a set of restart files whose file 
   name is 'restart_filename'.

  Log of RestartFileDFT.c:

   22/Nov/2001  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <complex.h> /*pohao*/
/*  end stat section */
#include "openmx_common.h"
#include "mpi.h"


static int  Input_HKS( int MD_iter, double *Uele, double *****CH );
static void Output_HKS(int MD_iter, double *Uele, double *****CH );
static void Output_Charge_Density(int MD_iter);
static int Input_Charge_Density(int MD_iter, double *extpln_coes);
static void Inverse(int n, double **a, double **ia);
static void Extp_Charge(int MD_iter, double *extpln_coes);


int RestartFileDFT(char *mode, int MD_iter, double *Uele, double *****H, double *****CntH, double *etime)
{
  int ret,ret1;
  int numprocs,myid;
  double *extpln_coes;
  double Stime,Etime;

  *etime = 0.0; 
  dtime(&Stime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (ML_flag) return 0;

  ret = 1;

  /* write */
  if ( strcasecmp(mode,"write")==0 ) {

    MPI_Barrier(mpi_comm_level1);

    if      (Cnt_switch==0)  Output_HKS(MD_iter, Uele, H);
    else if (Cnt_switch==1)  Output_HKS(MD_iter, Uele, CntH);

    MPI_Barrier(mpi_comm_level1);

    Output_Charge_Density(MD_iter);
  }

  /* read */
  else if (strcasecmp(mode,"read")==0) {
    
    /* find coefficients for extrapolation of charge density or H */

    /* allocation */

    extpln_coes = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+2)); 

    Extp_Charge(MD_iter,extpln_coes);

    /* read H */

    if (Use_of_Collinear_Restart==0){

      if      (Cnt_switch==0)  ret = Input_HKS(MD_iter, Uele, H);
      else if (Cnt_switch==1)  ret = Input_HKS(MD_iter, Uele, CntH);

    }

    /* read charge density */

    ret1 = Input_Charge_Density(MD_iter,extpln_coes);

    /* pohao 2017. 8. 25 */
    if (Use_of_Collinear_Restart==0){
      ret *= ret1; 
    }
    else if (Use_of_Collinear_Restart==1){
      ret = ret1;
    }

    /* free */

    free(extpln_coes);
  }

  dtime(&Etime);
  *etime = Etime - Stime;

  return ret;
}

  

 
void Extp_Charge(int MD_iter, double *extpln_coes)
{
  int i,j,k;
  int flag_nan,flag_improper;
  int NumHis;
  double sum;
  double **A,**IA,*B;
  char nanchar[300];
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* initialization of extpln_coes */

  for (i=0; i<Extrapolated_Charge_History; i++){
    extpln_coes[i] = 0.0;
  }
  extpln_coes[0] = 1.0;

  if (Cnt_switch==1) return;

  if (Extrapolated_Charge_History<MD_iter && 1<Extrapolated_Charge_History){

    /* allocation */

    A = (double**)malloc(sizeof(double*)*(Extrapolated_Charge_History+2));
    for (i=0; i<(Extrapolated_Charge_History+2); i++){
      A[i] = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+2));
    }

    IA = (double**)malloc(sizeof(double*)*(Extrapolated_Charge_History+2));
    for (i=0; i<(Extrapolated_Charge_History+2); i++){
      IA[i] = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+2));
    }
 
    B = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+2));

    /* find the current number of history */

    NumHis = MD_iter - 1;
    
    if (Extrapolated_Charge_History<NumHis){
      NumHis = Extrapolated_Charge_History;
    }

    /* make the matrix A */

    for (i=0; i<NumHis; i++){
      for (j=i; j<NumHis; j++){

        sum = rnd(1.0e-16);
        for (k=0; k<3*atomnum; k++){
          sum += His_Gxyz[i][k]*His_Gxyz[j][k];       
	}

        A[i][j] = sum;
        A[j][i] = sum;
      }
    }

    /* make the vector B */

    for (i=0; i<NumHis; i++){

      sum = 0.0;
      k = 0;

      for (j=1; j<=atomnum; j++){
	sum += His_Gxyz[i][k]*Gxyz[j][1]; k++;       
	sum += His_Gxyz[i][k]*Gxyz[j][2]; k++;       
	sum += His_Gxyz[i][k]*Gxyz[j][3]; k++;       
      }

      B[i] = sum; 
    }
  
    /* calculate the inverse of A */

    Inverse(NumHis-1,A,IA);

    /* calculate the coefficients */

    for (i=0; i<NumHis; i++){

      sum = 0.0;
      for (j=0; j<NumHis; j++){
        sum += IA[i][j]*B[j]; 
      }

      extpln_coes[i] = sum;
    }

    /*************************************
      check "nan", "NaN", "inf" or "Inf"
    *************************************/

    flag_nan = 0;
    for (i=0; i<NumHis; i++){
      sprintf(nanchar,"%8.4f",extpln_coes[i]);
      if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
       || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

        flag_nan = 1;
      }
    }

    if (flag_nan==1){
      for (i=0; i<Extrapolated_Charge_History; i++){
        extpln_coes[i] = 0.0;
      }
      extpln_coes[0] = 1.0;
    }

    /*******************************************************
       if a set of extpln_coes is improper, correct them
    *******************************************************/

    flag_improper = 0;

    for (i=0; i<NumHis; i++){
      if ( 10.0<fabs(extpln_coes[i]) ) flag_improper = 1;
    }

    sum = 0.0;
    for (i=0; i<NumHis; i++){
      sum += extpln_coes[i];
    }
    if (0.001<fabs(sum-1.0)) flag_improper = 1;

    if (flag_improper==1){
      for (i=0; i<Extrapolated_Charge_History; i++){
        extpln_coes[i] = 0.0;
      }
      extpln_coes[0] = 1.0;
    }

    /********************************
                   free 
    ********************************/

    for (i=0; i<(Extrapolated_Charge_History+2); i++){
      free(A[i]);
    }
    free(A);
 
    for (i=0; i<(Extrapolated_Charge_History+2); i++){
      free(IA[i]);
    }
    free(IA);

    free(B);

  } /* if (Extrapolated_Charge_History<MD_iter && 1<Extrapolated_Charge_History) */   

  /* shift His_Gxyz */

  for (i=(Extrapolated_Charge_History-2); 0<=i; i--){
    for (k=0; k<3*atomnum; k++){
      His_Gxyz[i+1][k] = His_Gxyz[i][k];
    }
  }

  k = 0;
  for (i=1; i<=atomnum; i++){
    His_Gxyz[0][k] = Gxyz[i][1]; k++;       
    His_Gxyz[0][k] = Gxyz[i][2]; k++;       
    His_Gxyz[0][k] = Gxyz[i][3]; k++;       
  }
}



void Inverse(int n, double **a, double **ia)
{
  int i,j,k;
  double sum;
  double **a0,*ko,*iko;

  /***************************************************
    allocation of arrays: 
  ***************************************************/

  a0 = (double**)malloc(sizeof(double*)*(Extrapolated_Charge_History+3));
  for (i=0; i<(Extrapolated_Charge_History+3); i++){
    a0[i] = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+3));
  }

  ko = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+3));
  iko = (double*)malloc(sizeof(double)*(Extrapolated_Charge_History+3));

  /***************************************************
    calculate the inverse
  ***************************************************/

  for (i=0; i<=n; i++){
    for (j=0; j<=n; j++){
      a0[i+1][j+1] = a[i][j];
    }  
  }  

  Eigen_lapack(a0,ko,n+1,n+1);

  for (i=1; i<=(n+1); i++){
    if (fabs(ko[i])<1.0e-15) 
      iko[i] = 0.0;
    else    
      iko[i] = 1.0/ko[i];
  }

  for (i=1; i<=(n+1); i++){
    for (j=1; j<=(n+1); j++){

      sum = 0.0;

      for (k=1; k<=(n+1); k++){
        sum += a0[i][k]*iko[k]*a0[j][k]; 
      }
      ia[i-1][j-1] = sum;
    }
  }

  /***************************************************
    freeing of arrays: 
  ***************************************************/

  for (i=0; i<(Extrapolated_Charge_History+3); i++){
    free(a0[i]);
  }
  free(a0);

  free(ko);
  free(iko);
}




int Input_HKS( int MD_iter, double *Uele, double *****CH )
{
  int Mc_AN,Gc_AN,h_AN,i,j,can;
  int Gh_AN,pSCF,spin,Rn;
  int wan1,wan2,TNO1,TNO2;
  int h_AN0,Gh_AN0,Rn0,wan20,TNO20;
  int i_vec[20],*p_vec,po;
  int my_check,exit_flag;
  int pFNAN;
  int l1,l2,l3,l10,l20,l30;
  int numprocs,myid;
  char fileHKS[YOUSO10];
  FILE *fp;
  double *tmpvec;  
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of array */ 
  tmpvec = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  /****************************************************
   List_YOUSO[23] spin poralized     1
                  non spin poralized 0
   List_YOUSO[1]  atomnum
   List_YOUSO[8]  max # of atoms in a rcut-off cluster
   List_YOUSO[7]  max # of orbitals in an atom
  ****************************************************/

  my_check = 1;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    TNO1 = Spe_Total_CNO[wan1];

    sprintf(fileHKS,"%s%s_rst/%s.rst%i",filepath,restart_filename,restart_filename,Gc_AN);

    if ((fp = fopen(fileHKS,"rb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      /****************************************************
       List_YOUSO[23] 0:  non spin poralized
                      1:  spin poralized
                      3:  spin non-collinear
       List_YOUSO[1]  atomnum
       List_YOUSO[8]  max # of atoms in a rcut-off cluster
       List_YOUSO[7]  max # of orbitals in an atom
      ****************************************************/

      fread(i_vec,sizeof(int),10,fp);

      pFNAN = i_vec[8];

      if ( i_vec[0]!=SpinP_switch   ||
           i_vec[1]!=List_YOUSO[23] || 
           i_vec[2]!=List_YOUSO[1]  ||
           i_vec[4]!=List_YOUSO[7]  ||
           i_vec[5]!=atomnum        ||
           i_vec[6]!=wan1           ||
           i_vec[7]!=TNO1           
         )
      {

        if ( core_hole_state_flag==0 ){

	  printf("Failed (1) in reading the restart file %s\n",fileHKS); fflush(stdout);     
	  my_check = 0;
	}
        else if ( core_hole_state_flag==1  && 
                  i_vec[0]==SpinP_switch   &&
                  i_vec[1]==List_YOUSO[23] && 
                  i_vec[2]==List_YOUSO[1]  &&
                  i_vec[4]==List_YOUSO[7]  &&
                  i_vec[5]==atomnum)        
        {
          /* nothing is done. */
	}
        else {

	  printf("Failed (1) in reading the restart file %s\n",fileHKS); fflush(stdout);     
	  my_check = 0;
	}
      }
      
      if (my_check!=0){

        /****************************************************
              read Gh_AN0, l10, l20, l30, wan20, TNO20
        ****************************************************/

        p_vec = (int*)malloc(sizeof(int)*(pFNAN+1)*6);
        fread(p_vec, sizeof(int), (pFNAN+1)*6, fp);
        fread(Uele,sizeof(double),1,fp);

        /****************************************************
          store Hamiltonian to appropriate position while 
          comparing Gh_AN, l1, l2, l3, wan2, TNO2
        ****************************************************/

        for (spin=0; spin<=SpinP_switch; spin++){ 
	  for (h_AN0=0; h_AN0<=pFNAN; h_AN0++){
	    Gh_AN0 = p_vec[              h_AN0];
	    l10    = p_vec[(pFNAN+1)*1 + h_AN0];
	    l20    = p_vec[(pFNAN+1)*2 + h_AN0];
	    l30    = p_vec[(pFNAN+1)*3 + h_AN0];
	    wan20  = p_vec[(pFNAN+1)*4 + h_AN0];
	    TNO20  = p_vec[(pFNAN+1)*5 + h_AN0]; 
 
	    exit_flag = 0;
	    h_AN = 0;

	    do {

	      Gh_AN = natn[Gc_AN][h_AN];
	      Rn    = ncn[Gc_AN][h_AN];
              l1 = atv_ijk[Rn][1];
              l2 = atv_ijk[Rn][2];
              l3 = atv_ijk[Rn][3];
	      wan2  = WhatSpecies[Gh_AN];
	      TNO2  = Spe_Total_CNO[wan2];

	      if ( Gh_AN==Gh_AN0 &&
		   l1==l10 &&
		   l2==l20 &&
		   l3==l30 &&
		   wan2==wan20   &&
		   TNO2==TNO20 )
		{

		  for (i=0; i<TNO1; i++){
		    fread(&CH[spin][Mc_AN][h_AN][i][0],sizeof(double),TNO2,fp);
		  }

		  /* add contribution of SO coupling */
		  if (i_vec[9]==0 && SO_switch==1 && spin==2){

		    for (i=0; i<TNO1; i++){
		      for (j=0; j<TNO2; j++){
			CH[2][Mc_AN][h_AN][i][j] += HNL[2][Mc_AN][h_AN][i][j];
		      }
		    }
		  }

		  exit_flag = 1;
		}

	      h_AN++;

	    } while (h_AN<=FNAN[Gc_AN] && exit_flag==0);

            /* In case appropriate one is not found, just read the Hamiltonian */ 

            if (exit_flag==0){
	      for (i=0; i<TNO1; i++){
	        fread(&tmpvec[0],sizeof(double),TNO20,fp);
	      }
            }

	  } /* h_AN */
	} /* spin */

        /****************************************************
          store iHNL to appropriate position while 
          comparing Gh_AN, Rn, wan2, TNO2
        ****************************************************/

        if (SpinP_switch==3){

	  for (spin=0; spin<SpinP_switch; spin++){
	    for (h_AN0=0; h_AN0<=pFNAN; h_AN0++){

	      Gh_AN0 = p_vec[              h_AN0];
	      l10    = p_vec[(pFNAN+1)*1 + h_AN0];
	      l20    = p_vec[(pFNAN+1)*2 + h_AN0];
	      l30    = p_vec[(pFNAN+1)*3 + h_AN0];
	      wan20  = p_vec[(pFNAN+1)*4 + h_AN0];
	      TNO20  = p_vec[(pFNAN+1)*5 + h_AN0]; 
 
	      exit_flag = 0;
	      h_AN = 0;

	      do {

		Gh_AN = natn[Gc_AN][h_AN];
		Rn    = ncn[Gc_AN][h_AN];
                l1 = atv_ijk[Rn][1];
                l2 = atv_ijk[Rn][2];
                l3 = atv_ijk[Rn][3];
		wan2  = WhatSpecies[Gh_AN];
		TNO2  = Spe_Total_CNO[wan2];

		if ( Gh_AN==Gh_AN0 &&
		     l1==l10 &&
		     l2==l20 &&
		     l3==l30 &&
		     wan2==wan20   &&
		     TNO2==TNO20 )
		  {

		    for (i=0; i<TNO1; i++){
		      fread(&iHNL[spin][Mc_AN][h_AN][i][0],sizeof(double),TNO2,fp);
		    }

                    /* add contribution of SO coupling */
                    if (i_vec[9]==0 && SO_switch==1){

  	  	      for (i=0; i<TNO1; i++){
    	    	        for (j=0; j<TNO2; j++){
                          iHNL[spin][Mc_AN][h_AN][i][j] += iHNL0[spin][Mc_AN][h_AN][i][j];
			}
		      }
                    }

		    exit_flag = 1;
		  }

		h_AN++;

	      } while (h_AN<=FNAN[Gc_AN] && exit_flag==0);

	      /* In case appropriate one is not found, just read the iHNL */ 

	      if (exit_flag==0){
		for (i=0; i<TNO1; i++){
		  fread(&tmpvec[0],sizeof(double),TNO20,fp);
		}
	      }

	    } /* h_AN */
	  } /* spin */

	} /* if (SpinP_switch==3) */

        /****************************************************
          store DM[0] to appropriate position while 
          comparing Gh_AN, l1, l2, l3, wan2, TNO2
        ****************************************************/

        for (spin=0; spin<=SpinP_switch; spin++){ 
	  for (h_AN0=0; h_AN0<=pFNAN; h_AN0++){
	    Gh_AN0 = p_vec[              h_AN0];
	    l10    = p_vec[(pFNAN+1)*1 + h_AN0];
	    l20    = p_vec[(pFNAN+1)*2 + h_AN0];
	    l30    = p_vec[(pFNAN+1)*3 + h_AN0];
	    wan20  = p_vec[(pFNAN+1)*4 + h_AN0];
	    TNO20  = p_vec[(pFNAN+1)*5 + h_AN0]; 
 
	    exit_flag = 0;
	    h_AN = 0;

	    do {

	      Gh_AN = natn[Gc_AN][h_AN];
	      Rn    = ncn[Gc_AN][h_AN];
              l1 = atv_ijk[Rn][1];
              l2 = atv_ijk[Rn][2];
              l3 = atv_ijk[Rn][3];
	      wan2  = WhatSpecies[Gh_AN];
	      TNO2  = Spe_Total_CNO[wan2];

	      if ( Gh_AN==Gh_AN0 &&
		   l1==l10 &&
		   l2==l20 &&
		   l3==l30 &&
		   wan2==wan20   &&
		   TNO2==TNO20 )
		{

		  for (i=0; i<TNO1; i++){
		    fread(&DM[0][spin][Mc_AN][h_AN][i][0],sizeof(double),TNO2,fp);
		  }

		  exit_flag = 1;
		}

	      h_AN++;

	    } while (h_AN<=FNAN[Gc_AN] && exit_flag==0);

            /* In case appropriate one is not found, just read the DM */ 

            if (exit_flag==0){
	      for (i=0; i<TNO1; i++){
	        fread(&tmpvec[0],sizeof(double),TNO20,fp);
	      }
            }

	  } /* h_AN */
	} /* spin */

        /* freeing of array */ 
        free(p_vec);
      }

      /* close the file */

      fclose(fp);
    }
    else{
      printf("Failed (2) in reading the restart file %s\n",fileHKS); fflush(stdout);     
      my_check = 0;
    }
  }

  /* freeing of array */ 
  free(tmpvec);

  return my_check;
}





void Output_HKS(int MD_iter, double *Uele, double *****CH )
{
  int Mc_AN,Gc_AN,h_AN,i,j,can,Gh_AN;
  int wan1,wan2,TNO1,TNO2,spin,Rn,po;
  int i_vec[20],*p_vec;
  int numprocs,myid;
  char operate[1000];
  char fileHKS[YOUSO10];
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* delete if there are */

  if (myid==Host_ID){

    if (MD_iter==1){
      sprintf(operate,"%s%s_rst",filepath,filename);
      mkdir(operate,0775); 
    }

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      sprintf(operate,"%s%s_rst/%s.rst%i",filepath,filename,filename,Gc_AN);
      remove(operate);
    }
  }

  MPI_Barrier(mpi_comm_level1);

  /* write */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    TNO1 = Spe_Total_CNO[wan1];

    sprintf(fileHKS,"%s%s_rst/%s.rst%i",filepath,filename,filename,Gc_AN);

    po = 0;

    do {

      if ((fp = fopen(fileHKS,"wb")) != NULL){

         setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	/****************************************************
         List_YOUSO[23] 0:  non spin poralized
                        1:  spin poralized
                        3:  spin non-collinear
         List_YOUSO[1]  atomnum
         List_YOUSO[8]  max # of atoms in a rcut-off cluster
         List_YOUSO[7]  max # of orbitals including an atom
	****************************************************/

	i_vec[0] = SpinP_switch;
	i_vec[1] = List_YOUSO[23];
	i_vec[2] = List_YOUSO[1];
	i_vec[3] = List_YOUSO[8];
	i_vec[4] = List_YOUSO[7];
	i_vec[5] = atomnum;
	i_vec[6] = wan1;
	i_vec[7] = TNO1;
	i_vec[8] = FNAN[Gc_AN];
	i_vec[9] = SO_switch;

	fwrite(i_vec,sizeof(int),10,fp);

	/****************************************************
                  # of orbitals in each FNAN atom
	****************************************************/

	p_vec = (int*)malloc(sizeof(int)*(FNAN[Gc_AN]+1)*6);
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Rn = ncn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];
	  p_vec[                    h_AN] = Gh_AN;
	  p_vec[(FNAN[Gc_AN]+1)*1 + h_AN] = atv_ijk[Rn][1];
	  p_vec[(FNAN[Gc_AN]+1)*2 + h_AN] = atv_ijk[Rn][2];
	  p_vec[(FNAN[Gc_AN]+1)*3 + h_AN] = atv_ijk[Rn][3];
	  p_vec[(FNAN[Gc_AN]+1)*4 + h_AN] = wan2;
	  p_vec[(FNAN[Gc_AN]+1)*5 + h_AN] = TNO2;
	}
	fwrite(p_vec,sizeof(int), (FNAN[Gc_AN]+1)*6, fp);
	free(p_vec);

	/****************************************************
                               Uele
	****************************************************/

	fwrite(Uele,sizeof(double),1,fp);

	/****************************************************
                         Kohn-Sham Hamiltonian
	****************************************************/

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];
	    TNO2 = Spe_Total_CNO[wan2];
	    for (i=0; i<TNO1; i++){
	      fwrite(CH[spin][Mc_AN][h_AN][i],sizeof(double),TNO2,fp);
	    }
	  }
	}

        if (SpinP_switch==3){
	  for (spin=0; spin<SpinP_switch; spin++){
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      wan2 = WhatSpecies[Gh_AN];
	      TNO2 = Spe_Total_CNO[wan2];
	      for (i=0; i<TNO1; i++){
	        fwrite(iHNL[spin][Mc_AN][h_AN][i],sizeof(double),TNO2,fp);
	      }
	    }
  	  }
        }

	/****************************************************
                                  DM
	****************************************************/

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];
	    TNO2 = Spe_Total_CNO[wan2];
	    for (i=0; i<TNO1; i++){
	      fwrite(DM[0][spin][Mc_AN][h_AN][i],sizeof(double),TNO2,fp);
	    }
	  }
	}

        /* fclose(fp) */

	fclose(fp);

	po = 1;

      }
      else{

	printf("Failed in saving the restart file %s\n",fileHKS);      
	/* openmx gives up to save the restart file. */
	po = 1;
      }

    } while (po==0);
  }

  /**************************************************
               save core charge in *.cc       
  **************************************************/

  if (myid==Host_ID){

    int Cwan;
    double *d_vec,Zc;
    d_vec = (double*)malloc(sizeof(double)*(atomnum+1));

    sprintf(fileHKS,"%s%s_rst/%s.cc",filepath,filename,filename);

    if ((fp = fopen(fileHKS,"wb")) != NULL){

      d_vec[0] = system_charge;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
        Cwan = WhatSpecies[Gc_AN];
        Zc = Spe_Core_Charge[Cwan];
	d_vec[Gc_AN] = Zc;
      }

      fwrite(d_vec,sizeof(double),atomnum+1,fp);
      fclose(fp);
    }
    else{
      printf("Failed in saving the cc file %s\n",fileHKS);      
    }

    free(d_vec);
  }

}





void Output_Charge_Density(int MD_iter)
{
  int i,spin,BN;
  int numprocs,myid;
  double *TmpRho;
  FILE *fp;
  char file_check[YOUSO10];
  char fileCD0[YOUSO10];
  char fileCD1[YOUSO10];
  char fileCD2[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of array */
  TmpRho = (double*)malloc(sizeof(double)*My_NumGridB_AB); 

  /* save numprocs, Ngrid1, Ngrid2, and Ngrid3 */

  if (myid==Host_ID){

    sprintf(file_check,"%s%s_rst/%s.crst_check",filepath,filename,filename);

    if ((fp = fopen(file_check,"w")) != NULL){
      fprintf(fp,"%d %d %d %d %d",numprocs,Ngrid1,Ngrid2,Ngrid3,SpinP_switch);
      fclose(fp);
    }
    else{
      printf("Failure of saving %s\n",file_check);
    }
  }

  /****************************************************
     output data for  
     Density_Grid_B - ADensity_Grid_B
     or 
     Density_Grid_B
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    /* store data to TmpRho */

    if (spin<=1){
      for (BN=0; BN<My_NumGridB_AB; BN++){
        TmpRho[BN] = Density_Grid_B[spin][BN] - ADensity_Grid_B[BN];
      }
    }
    else{
      for (BN=0; BN<My_NumGridB_AB; BN++){
        TmpRho[BN] = Density_Grid_B[spin][BN];
      }
    } 

    /* shift the index of stored data */

    for (i=(Extrapolated_Charge_History-2); 0<=i; i--){
 
      sprintf(fileCD1,"%s%s_rst/%s.crst%i_%i_%i",filepath,filename,filename,spin,myid,i);
      sprintf(fileCD2,"%s%s_rst/%s.crst%i_%i_%i",filepath,filename,filename,spin,myid,i+1);

      if ((fp = fopen(fileCD1,"rb")) != NULL){
	fclose(fp);
	rename(fileCD1,fileCD2); 
      }
    } 

    /* save current data */

    sprintf(fileCD0,"%s%s_rst/%s.crst%i_%i_0",filepath,filename,filename,spin,myid);

    if ((fp = fopen(fileCD0,"wb")) != NULL){

      fwrite(TmpRho,sizeof(double),My_NumGridB_AB,fp);
      fclose(fp);
    }
    else{
      printf("Could not open a file %s\n",fileCD0);
    }

  } /* spin */

  /****************************************************
            output data for ADensity_Grid_B 
  ****************************************************/
 
  sprintf(fileCD0,"%s%s_rst/%s.adrst%i",filepath,filename,filename,myid);

  if ((fp = fopen(fileCD0,"wb")) != NULL){

    fwrite(ADensity_Grid_B,sizeof(double),My_NumGridB_AB,fp);
    fclose(fp);
  }
  else{
    printf("Could not open a file %s\n",fileCD0);
  }
	
  /* freeing of array */
  free(TmpRho);
}








int Input_Charge_Density(int MD_iter, double *extpln_coes)
{
  double phi[2],theta[2],si_sq,co_sq,sc, Qx; /*pohao*/
  double Re11,Re22,Re12,Im12;    /*pohao*/
  int j,k, l, rest_col;    /*pohao*/
  double complex z;    /*pohao*/
  double complex N_spin_rot[2][2];    /*pohao*/
  double complex U_nQx[2][2];    /*pohao*/
  double complex U_nQx_h[2][2];  /*pohao*/
  double complex N_spin[2][2];   /*pohao*/
  double n_r[3];   /*pohao*/           
  double Nu,Nd;    /*pohao*/
  double t,p;

  int spin,i,BN,ret;
  int numprocs,myid;
  FILE *fp;
  char fileCD[YOUSO10];
  char file_check[YOUSO10];
  double *tmp_array;
  int i_vec[10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* check consistency in the number of grids */

  sprintf(file_check,"%s%s_rst/%s.crst_check",filepath,restart_filename,restart_filename);

  i_vec[1] = 0; i_vec[2] = 0; i_vec[3] = 0;       
  if ((fp = fopen(file_check,"r")) != NULL){
    fscanf(fp,"%d %d %d %d %d",&i_vec[0],&i_vec[1],&i_vec[2],&i_vec[3],&i_vec[4]);
    fclose(fp);
  }

  /* read files */

  if ( i_vec[1]==Ngrid1 && i_vec[2]==Ngrid2 && i_vec[3]==Ngrid3 ){

    /* allocation of array */

    tmp_array = (double*)malloc(sizeof(double)*My_NumGridB_AB);

    /* read data and extrapolate data of crst files */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<Extrapolated_Charge_History; i++){

	sprintf(fileCD,"%s%s_rst/%s.crst%i_%i_%i",filepath,restart_filename,restart_filename,spin,myid,i);

	if ((fp = fopen(fileCD,"rb")) != NULL){

	  fread(tmp_array,sizeof(double),My_NumGridB_AB,fp);
	  fclose(fp);

	  if (i==0){

	    if (spin<=1){
	      for (BN=0; BN<My_NumGridB_AB; BN++){
		Density_Grid_B[spin][BN] = ADensity_Grid_B[BN] + extpln_coes[i]*tmp_array[BN];
	      }
	    }
	    else{
	      for (BN=0; BN<My_NumGridB_AB; BN++){
		Density_Grid_B[spin][BN] = extpln_coes[i]*tmp_array[BN];
	      }
	    }

	  }

	  else{
	    for (BN=0; BN<My_NumGridB_AB; BN++){
	      Density_Grid_B[spin][BN] += extpln_coes[i]*tmp_array[BN];
	    }
	  }

	}
	else{
	  /* printf("Could not open a file %s\n",fileCD); */
	}
      }
    }

    /* A restart file generated by a collinear calculation is used for the non-collinear calculation. */

    if (Use_of_Collinear_Restart==1){

      if (myid==0) printf("The collinear charge density is read for the non-collinear calculation.\n"); fflush(stdout);     

      /* copy rho11 to rho22 */

      if (SpinP_switch_RestartFiles==0){
        for (BN=0; BN<My_NumGridB_AB; BN++){
	  Density_Grid_B[1][BN] = Density_Grid_B[0][BN];
	}
      }

      /****************************************************************************
                               rotation of spin angle 
      ******************************************************************************/
      
      t = Restart_Spin_Angle_Theta/180.0*PI*0.5; 
      p = Restart_Spin_Angle_Phi/180.0*PI*0.5; 
      
      U_nQx[0][0]  = cos(p)*cos(t)+I*sin(p)*cos(t);
      U_nQx[1][0]  =-cos(p)*sin(t)-I*sin(p)*sin(t);
      U_nQx[0][1]  = cos(p)*sin(t)-I*sin(p)*sin(t);
      U_nQx[1][1]  = cos(p)*cos(t)-I*sin(p)*cos(t);

      U_nQx_h[0][0]= cos(p)*cos(t)-I*sin(p)*cos(t);
      U_nQx_h[1][0]= cos(p)*sin(t)+I*sin(p)*sin(t);
      U_nQx_h[0][1]=-cos(p)*sin(t)+I*sin(p)*sin(t); 
      U_nQx_h[1][1]= cos(p)*cos(t)+I*sin(p)*cos(t);

      for (BN=0; BN<My_NumGridB_AB; BN++){

	N_spin[0][0] = Density_Grid_B[0][BN]; 
	N_spin[1][0] = 0.0; 
	N_spin[0][1] = 0.0; 
	N_spin[1][1] = Density_Grid_B[1][BN];

	for (k=0; k <2; k++) {
	  for (j=0; j <2; j++) {
	    N_spin_rot[k][j] = 0.0;
	  }
	}

	/*  U(q)*N_spinU_dagger*(q)  */

	for (j=0; j<2; j++) {
	  for (k=0; k<2; k++) {
	    for (l=0; l<2; l++) {
	      N_spin_rot[j][l] += U_nQx_h[j][k] * N_spin[k][k] * U_nQx[k][l];
	    }    
	  }
	} 

	Re11 = creal(N_spin_rot[0][0]); 
	Re22 = creal(N_spin_rot[1][1]);
	Re12 = creal(N_spin_rot[0][1]);
	Im12 = cimag(N_spin_rot[0][1]);

	Density_Grid_B[0][BN] = Re11;  
	Density_Grid_B[1][BN] = Re22;
	Density_Grid_B[2][BN] = Re12;
	Density_Grid_B[3][BN] = Im12;

      }

    } /* if (Use_of_Collinear_Restart==1) */

    /* MPI: from the partitions B to D */

    Density_Grid_Copy_B2D(Density_Grid_B);

    /* copy Density_Grid_B in a core hole calculation to Density_Periodic_Grid_B */

    if (core_hole_state_flag || scf_coulomb_cutoff_CoreHole==1){

      if (SpinP_switch==0){
	for (BN=0; BN<My_NumGridB_AB; BN++){
	  Density_Periodic_Grid_B[BN] = 2.0*Density_Grid_B[0][BN];
	}  
      }
      else{
	for (BN=0; BN<My_NumGridB_AB; BN++){
	  Density_Periodic_Grid_B[BN] = Density_Grid_B[0][BN] + Density_Grid_B[1][BN];
	}  
      }
    }

    /* free */
    free(tmp_array);

    /* ret1 */
    ret = 1;
  }

  else{
    if (myid==0) printf("Failed (3) in reading the restart files\n"); fflush(stdout);     
    ret = 0;
  }

  return ret;  
}

