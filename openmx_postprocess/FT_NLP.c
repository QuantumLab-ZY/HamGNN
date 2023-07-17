/**********************************************************************
  FT_NLP.c:

     FT_NLP.c is a subroutine to Fourier transform projectors
     of nonlocal potentials.

  Log of FT_NLP.c:

     15/Sep/2002  Released by T.Ozaki

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



void FT_NLP()
{
  int numprocs,myid,ID,tag=999;
  int count,NumSpe;
  int i,kj,num_k,so;
  int Lspe,spe,L,GL,MaxGL;
  int RestartRead_Succeed;
  double dk,norm_k;
  double rmin,rmax,r,r2,h,sum[2];
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
 
  if (myid==Host_ID && 0<level_stdout) printf("<FT_NLP>          Fourier transform of non-local projectors\n");

  RestartRead_Succeed = 0;

  /***********************************************************
     In case of Scf_RestartFromFile==1, read Spe_NLRF_Bessel
  ***********************************************************/

  if (Scf_RestartFromFile==1){

    /****************************************************
          generate radial grids in the k-space
    ****************************************************/

    dk = PAO_Nkmax/(double)Ngrid_NormK;
    for (i=0; i<Ngrid_NormK; i++){
      NormK[i] = (double)i*dk;
    }

    /***********************************************************
                      read Spe_NLRF_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftnlp",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
	for (so=0; so<=VPS_j_dependency[spe]; so++){
	  for (L=1; L<=Spe_Num_RVPS[spe]; L++){

	    size = fread(&Spe_NLRF_Bessel[so][spe][L][0],sizeof(double),List_YOUSO[15],fp);
	    if (size!=List_YOUSO[15]) RestartRead_Succeed = 0;
	  }
	}
      }

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_NLP\n",fileFT);
    }
  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_NLRF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      num_k = Ngrid_NormK;
      dk = PAO_Nkmax/(double)num_k;
      rmin = Spe_VPS_RV[spe][0];
      rmax = Spe_Atom_Cut1[spe] + 0.5;
      h = (rmax - rmin)/(double)OneD_Grid;

      /* kj loop */

#pragma omp parallel shared(Spe_VPS_List,spe,Spe_Num_RVPS,num_k,dk,OneD_Grid,rmin,h,VPS_j_dependency,Spe_NLRF_Bessel)  private(MaxGL,L,GL,SphB,tmp_SphB,tmp_SphBp,OMPID,Nthrds,Nprocs,norm_k,i,r,r2,sum,so,kj)
      {

	/* allocate SphB */

	MaxGL = -1;
	for (L=1; L<=Spe_Num_RVPS[spe]; L++){
	  GL = Spe_VPS_List[spe][L];
	  if (MaxGL<GL) MaxGL = GL;
	}      

	SphB = (double**)malloc(sizeof(double*)*(MaxGL+3));
	for(GL=0; GL<(MaxGL+3); GL++){ 
	  SphB[GL] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
	}

	tmp_SphB  = (double*)malloc(sizeof(double)*(MaxGL+3));
	tmp_SphBp = (double*)malloc(sizeof(double)*(MaxGL+3));

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for ( kj=OMPID; kj<num_k; kj+=Nthrds ){

	  norm_k = (double)kj*dk;

	  /* calculate SphB */

	  for (i=0; i<=OneD_Grid; i++){

	    r = rmin + (double)i*h;

	    Spherical_Bessel(norm_k*r,MaxGL,tmp_SphB,tmp_SphBp);

	    r2 = r*r;
	    for(GL=0; GL<=MaxGL; GL++){ 
	      SphB[GL][i] = tmp_SphB[GL]*r2; 
	    }
	  }

	  for(GL=0; GL<=MaxGL; GL++){ 
	    SphB[GL][0] *= 0.5;
	    SphB[GL][OneD_Grid] *= 0.5;
	  }

	  /* loof for L */

	  for (L=1; L<=Spe_Num_RVPS[spe]; L++){

	    GL = Spe_VPS_List[spe][L];

	    /****************************************************
                      \int jL(k*r)*RL*r^2 dr 
	    ****************************************************/

	    sum[0] = 0.0;
	    sum[1] = 0.0;

	    for (i=0; i<=OneD_Grid; i++){
	      r = rmin + (double)i*h;
	      for (so=0; so<=VPS_j_dependency[spe]; so++){
		sum[so] += Nonlocal_RadialF(spe,L-1,so,r)*SphB[GL][i];
	      }
	    }

	    for (so=0; so<=VPS_j_dependency[spe]; so++){
	      Spe_NLRF_Bessel[so][spe][L][kj] = sum[so]*h;
	    }

	  } /* L */
	} /* kj */

	/* free arrays */

	for(GL=0; GL<(MaxGL+3); GL++){ 
	  free(SphB[GL]);
	}
	free(SphB);

	free(tmp_SphB);
	free(tmp_SphBp);

#pragma omp flush(Spe_NLRF_Bessel)

      } /* #pragma omp parallel */

    } /* Lspe */

    /****************************************************
     Remedy for MSpeciesNum==0
     generate radial grids in the k-space
    ****************************************************/

    dk = PAO_Nkmax/(double)Ngrid_NormK;
    for (i=0; i<Ngrid_NormK; i++){
      NormK[i] = (double)i*dk;
    }

    /***********************************************************
        sending and receiving of Spe_RF_Bessel by MPI
    ***********************************************************/

    for (ID=0; ID<Num_Procs2; ID++){
      NumSpe = Species_End[ID] - Species_Top[ID] + 1;
      for (Lspe=0; Lspe<NumSpe; Lspe++){
	spe = Species_Top[ID] + Lspe;
	for (so=0; so<=VPS_j_dependency[spe]; so++){
	  for (L=1; L<=Spe_Num_RVPS[spe]; L++){
	    MPI_Bcast(&Spe_NLRF_Bessel[so][spe][L][0],
		      List_YOUSO[15],MPI_DOUBLE,ID,mpi_comm_level1);
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_NLRF_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(fileFT,"%s%s_rst/%s.ftnlp",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
  	  for (so=0; so<=VPS_j_dependency[spe]; so++){
	    for (L=1; L<=Spe_Num_RVPS[spe]; L++){
	      fwrite(&Spe_NLRF_Bessel[so][spe][L][0],sizeof(double),List_YOUSO[15],fp);
	    }
	  }
	}  

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_NLP\n",fileFT);
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
