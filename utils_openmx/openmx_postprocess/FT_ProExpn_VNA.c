/**********************************************************************
  FT_ProExpn_VNA.c:

     FT_ProExpn_VNA.c is a subroutine to Fourier transform
     VNA separable projectors.

  Log of FT_ProExpn_VNA.c:

     7/Apr/2004  Released by T.Ozaki

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



void FT_ProExpn_VNA()
{
  int numprocs,myid,ID,tag=999;
  int count,NumSpe;
  int L,i,kj;
  int Lspe,spe,GL,Mul;
  int RestartRead_Succeed;
  double Sr,Dr;
  double norm_k,h,dum0;
  double rmin,rmax,r,sum;
  double kmin,kmax,Sk,Dk;
  double RGL[GL_Mesh + 2];
  double *SumTmp;
  double tmp0,tmp1;
  double **SphB;
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

  if (myid==Host_ID && 0<level_stdout) printf("<FT_ProExpn_VNA>  Fourier transform of VNA separable projectors\n");

  RestartRead_Succeed = 0;

  /***********************************************************
     In case of Scf_RestartFromFile==1, read Spe_VNA_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /****************************************************
         regenerate radial grids in the k-space
         for the MPI calculation
    ****************************************************/

    for (kj=0; kj<GL_Mesh; kj++){
      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;
      norm_k = 0.50*(Dk*GL_Abscissae[kj] + Sk);
      GL_NormK[kj] = norm_k;
    }

    /***********************************************************
                        read Spe_VNA_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftPEvna",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
        for (L=0; L<=List_YOUSO[35]; L++){
	  for (Mul=0; Mul<List_YOUSO[34]; Mul++){

	    size = fread(&Spe_VNA_Bessel[spe][L][Mul][0],sizeof(double),GL_Mesh,fp);
	    if (size!=GL_Mesh) RestartRead_Succeed = 0;
	  }
        }
      }

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_ProExpn_VNA\n",fileFT);
    }
  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_VNA_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      /* initalize */
      /* tabulation on Gauss-Legendre radial grid */

      rmin = Spe_VPS_RV[spe][0];
      rmax = Spe_Atom_Cut1[spe] + 0.5;
      Sr = rmax + rmin;
      Dr = rmax - rmin;
      for (i=0; i<GL_Mesh; i++){
	RGL[i] = 0.50*(Dr*GL_Abscissae[i] + Sr);
      }

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      /* loop for kj */

#pragma omp parallel shared(spe,List_YOUSO,GL_Weight,GL_Abscissae,Dr,Dk,Sk,RGL,Projector_VNA,Spe_VPS_RV,Spe_Num_Mesh_VPS,Spe_VNA_Bessel)  private(SumTmp,SphB,tmp_SphB,tmp_SphBp,OMPID,Nthrds,Nprocs,kj,norm_k,i,r,L,Mul,tmp0,dum0)

      {

	/* allocate arrays */

	SumTmp = (double*)malloc(sizeof(double)*List_YOUSO[34]);

	SphB = (double**)malloc(sizeof(double*)*(List_YOUSO[35]+3));
	for(L=0; L<(List_YOUSO[35]+3); L++){ 
	  SphB[L] = (double*)malloc(sizeof(double)*GL_Mesh);
	}

	tmp_SphB  = (double*)malloc(sizeof(double)*(List_YOUSO[35]+3));
	tmp_SphBp = (double*)malloc(sizeof(double)*(List_YOUSO[35]+3));

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for ( kj=OMPID; kj<GL_Mesh; kj+=Nthrds ){

	  norm_k = 0.50*(Dk*GL_Abscissae[kj] + Sk);

	  /* calculate SphB */

	  for (i=0; i<GL_Mesh; i++){

	    r = RGL[i];
	    Spherical_Bessel(norm_k*r,List_YOUSO[35],tmp_SphB,tmp_SphBp);

	    for(L=0; L<=List_YOUSO[35]; L++){ 
	      SphB[L][i]  =  tmp_SphB[L]; 
	    }
	  }

	  /* loop for L */
 
	  for (L=0; L<=List_YOUSO[35]; L++){

	    /****************************************************
                      \int jL(k*r)RL r^2 dr 
	    ****************************************************/

	    for (Mul=0; Mul<List_YOUSO[34]; Mul++) SumTmp[Mul] = 0.0;

	    /* Gauss-Legendre quadrature */

	    for (i=0; i<GL_Mesh; i++){

	      r = RGL[i];

	      tmp0 = r*r*GL_Weight[i]*SphB[L][i];
	      for (Mul=0; Mul<List_YOUSO[34]; Mul++){
		dum0 = PhiF(r, Projector_VNA[spe][L][Mul], Spe_VPS_RV[spe], Spe_Num_Mesh_VPS[spe]);   
		SumTmp[Mul] += dum0*tmp0;
	      }
	    }

	    for (Mul=0; Mul<List_YOUSO[34]; Mul++){
	      Spe_VNA_Bessel[spe][L][Mul][kj] = 0.5*Dr*SumTmp[Mul];
	    }

	  } /* L */
	} /* kj */ 

	/* free arrays */

	free(SumTmp);

	for(L=0; L<(List_YOUSO[35]+3); L++){ 
	  free(SphB[L]);
	}
	free(SphB);

	free(tmp_SphB);
	free(tmp_SphBp);

#pragma omp flush(Spe_VNA_Bessel)

      } /* #pragma omp parallel */
    } /* Lspe */

    /****************************************************
         regenerate radial grids in the k-space
         for the MPI calculation
    ****************************************************/

    for (kj=0; kj<GL_Mesh; kj++){
      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;
      norm_k = 0.50*(Dk*GL_Abscissae[kj] + Sk);
      GL_NormK[kj] = norm_k;
    }

    /***********************************************************
        sending and receiving of Spe_VNA_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	for (L=0; L<=List_YOUSO[35]; L++){
	  for (Mul=0; Mul<List_YOUSO[34]; Mul++){
	    MPI_Bcast(&Spe_VNA_Bessel[spe][L][Mul][0],
		      GL_Mesh,MPI_DOUBLE,ID,mpi_comm_level1);
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_VNA_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftPEvna",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  for (L=0; L<=List_YOUSO[35]; L++){
	    for (Mul=0; Mul<List_YOUSO[34]; Mul++){
	      fwrite(&Spe_VNA_Bessel[spe][L][Mul][0],sizeof(double),GL_Mesh,fp);
	    }
	  }
	}

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_ProExpn_VNA\n",fileFT);
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




