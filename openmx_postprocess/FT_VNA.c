/**********************************************************************
  FT_VNA.c:

     FT_VNA.c is a subroutine to Fourier transform VNA potentials

  Log of FT_VNA.c:

     18/May/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*  end stat section */
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>





void FT_VNA()
{
  int numprocs,myid,ID,tag=999;
  int count,NumSpe;
  int L,i,j;
  int Lspe,spe,GL,Mul;
  int RestartRead_Succeed;
  double Sr,Dr,Sk,Dk,kmin,kmax;
  double norm_k,h,dum0;
  double xmin,xmax,x,r,sum;
  double sj;
  double tmp0,tmp1;
  double *tmp_SphB,*tmp_SphBp;
  double TStime, TEtime;
  /* for MPI */
  MPI_Status stat;
  MPI_Request request;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  char fileFT[YOUSO10];
  char operate[300];
  FILE *fp;
  size_t size; 

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout) printf("<FT_VNA>          Fourier transform of VNA potentials\n");

  RestartRead_Succeed = 0;

  /***********************************************************
   In case of Scf_RestartFromFile==1, read Spe_CrudeVNA_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /****************************************************
         regenerate radial grids in the k-space
         for the MPI calculation
    ****************************************************/

    for (j=0; j<GL_Mesh; j++){
      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;
      norm_k = 0.50*(Dk*GL_Abscissae[j] + Sk);
      GL_NormK[j] = norm_k;
    }

    /***********************************************************
                        read Spe_VNA_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftCvna",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){

	size = fread(&Spe_CrudeVNA_Bessel[spe][0],sizeof(double),GL_Mesh,fp);
	if (size!=GL_Mesh) RestartRead_Succeed = 0;
      }

      fclose(fp);
    }
    else{
      printf("Error1: Could not open a file %s in FT_VNA\n",fileFT);
    }
  }

  /***********************************************************
   if (RestartRead_Succeed==0), calculate Spe_CrudeVNA_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    /* loop for Lspe */

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      /****************************************************
                   \int jL(k*r)RL r^2 dr 
      ****************************************************/

      /* tabulation on Gauss-Legendre radial grid */

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;
      xmin = sqrt(Spe_PAO_RV[spe][0]);
      xmax = sqrt(Spe_Atom_Cut1[spe] + 0.5);
      h = (xmax - xmin)/(double)OneD_Grid;

      /* loop for j */

#pragma omp parallel shared(spe,Dk,Sk,GL_Abscissae,xmin,xmax,h,OneD_Grid,Spe_CrudeVNA_Bessel)  private(OMPID,Nthrds,Nprocs,j,norm_k,sum,x,r,i,tmp_SphB,tmp_SphBp,sj)
      {

	/* allocate arrays */

	tmp_SphB  = (double*)malloc(sizeof(double)*3);
	tmp_SphBp = (double*)malloc(sizeof(double)*3);

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for ( j=OMPID; j<GL_Mesh; j+=Nthrds ){

	  norm_k = 0.50*(Dk*GL_Abscissae[j] + Sk);

	  /**************************
           trapezoidal rule

            grid: r = x^2
                  dr = 2*x*dx 
	  ***************************/

	  sum = 0.0;

	  for (i=0; i<=OneD_Grid; i++){
	    x = xmin + (double)i*h;
	    r = x*x; 

	    Spherical_Bessel(norm_k*r,0,tmp_SphB,tmp_SphBp);
	    sj = tmp_SphB[0];

	    if (i==0 || i==OneD_Grid)
	      sum += r*r*x*sj*VNAF(spe,r);
	    else 
	      sum += 2.0*r*r*x*sj*VNAF(spe,r);
	  }
	  sum = sum*h;

	  Spe_CrudeVNA_Bessel[spe][j] = sum;

	} /* j */

	/* free arrays */

	free(tmp_SphB);
	free(tmp_SphBp);

#pragma omp flush(Spe_CrudeVNA_Bessel)

      } /* #pragma omp parallel */
    } /* Lspe */

    /****************************************************
         regenerate radial grids in the k-space
         for the MPI calculation
    ****************************************************/

    for (j=0; j<GL_Mesh; j++){
      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;
      norm_k = 0.50*(Dk*GL_Abscissae[j] + Sk);
      GL_NormK[j] = norm_k;
    }

    /***********************************************************
      sending and receiving of Spe_CrudeVNA_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	MPI_Bcast(&Spe_CrudeVNA_Bessel[spe][0],
		  GL_Mesh,MPI_DOUBLE,ID,mpi_comm_level1);
      }
    }

    /***********************************************************
                        save Spe_CrudeVNA_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftCvna",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  fwrite(&Spe_CrudeVNA_Bessel[spe][0],sizeof(double),GL_Mesh,fp);
	}

	fclose(fp);
      }
      else{
	printf("Error2: Could not open a file %s in FT_VNA\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

  /*
  printf("myid=%2d Elapsed Time (s) = %15.12f\n",myid,TEtime-TStime);
  MPI_Finalize();
  exit(0);
  */
}




