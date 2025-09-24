/**********************************************************************
  FT_PAO.c:

     FT_PAO.c is a subroutine to Fourier transform pseudo atomic 
     orbitals.

  Log of FT_PAO.c:

     15/Sep/2002  Released by T. Ozaki

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



void FT_PAO()
{
  int numprocs,myid,ID,tag=999;
  int count,NumSpe;
  int i,kj,num_k;
  int Lspe,spe,GL,Mul,LB;
  int RestartRead_Succeed;
  double dk,norm_k,h;
  double rmin,rmax,r,r2,r3,sum;
  double sy,sjp,syp;
  double Sr,Dr,dum0;
  double **SphB,**SphB2;
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

  if (myid==Host_ID && 0<level_stdout) printf("<FT_PAO>          Fourier transform of pseudo atomic orbitals\n");

  RestartRead_Succeed = 0;

  /***********************************************************
      In case of Scf_RestartFromFile==1, read Spe_RF_Bessel
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
                      read Spe_RF_Bessel
    ***********************************************************/

    sprintf(fileFT,"%s%s_rst/%s.ftpao",filepath,restart_filename,restart_filename);

    if ((fp = fopen(fileFT,"rb")) != NULL){

      RestartRead_Succeed = 1;

      for (spe=0; spe<SpeciesNum; spe++){
	for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	  for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){

	    size = fread(&Spe_RF_Bessel[spe][GL][Mul][0],sizeof(double),List_YOUSO[15],fp);
            if (size!=List_YOUSO[15]) RestartRead_Succeed = 0;
	  }
	}
      }  

      fclose(fp);
    }
    else{
      printf("Could not open a file %s in FT_PAO\n",fileFT);
    }

  }

  /***********************************************************
     if (RestartRead_Succeed==0), calculate Spe_RF_Bessel
  ***********************************************************/

  if (RestartRead_Succeed==0){

    for (Lspe=0; Lspe<MSpeciesNum; Lspe++){

      spe = Species_Top[myid] + Lspe;

      num_k = Ngrid_NormK;
      dk = PAO_Nkmax/(double)num_k;
      rmin = Spe_PAO_RV[spe][0];
      rmax = Spe_Atom_Cut1[spe] + 0.5;
      h = (rmax - rmin)/(double)OneD_Grid;

      /* kj loop */

#pragma omp parallel shared(List_YOUSO,num_k,dk,spe,rmin,rmax,h,Spe_PAO_RV,Spe_Atom_Cut1,OneD_Grid,Spe_MaxL_Basis,Spe_Num_Basis,Spe_RF_Bessel)  private(OMPID,Nthrds,Nprocs,kj,norm_k,i,r,r2,tmp_SphB,SphB2,tmp_SphBp,GL,SphB,Mul,sum,LB)
      {

	/* allocate SphB */

	SphB = (double**)malloc(sizeof(double*)*(Spe_MaxL_Basis[spe]+3));
	for(GL=0; GL<(Spe_MaxL_Basis[spe]+3); GL++){ 
	  SphB[GL] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
	}

        SphB2 = (double**)malloc(sizeof(double*)*(List_YOUSO[25]*2+3));
        for(LB=0; LB<(List_YOUSO[25]*2+3); LB++){
          SphB2[LB] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
        }

	tmp_SphB  = (double*)malloc(sizeof(double)*(List_YOUSO[25]*2+3));
	tmp_SphBp = (double*)malloc(sizeof(double)*(List_YOUSO[25]*2+3));

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for ( kj=OMPID; kj<num_k; kj+=Nthrds ){

	  norm_k = (double)kj*dk;

	  /* calculate SphB */

	  for (i=0; i<=OneD_Grid; i++){

	    r = rmin + (double)i*h;
	    Spherical_Bessel(norm_k*r,Spe_MaxL_Basis[spe],tmp_SphB,tmp_SphBp);

	    r2 = r*r;
	    for(GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){ 
	      SphB[GL][i] = tmp_SphB[GL]*r2; 
	    }
	  }

	  for(GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){ 
	    SphB[GL][0] *= 0.5;
	    SphB[GL][OneD_Grid] *= 0.5;
	  }

	  /* loof for GL and Mul */

	  for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	    for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){

	      /****************************************************
                        \int jL(k*r)RL r^2 dr 
	      ****************************************************/

	      /* trapezoidal rule */

	      sum = 0.0;
	      for (i=0; i<=OneD_Grid; i++){
		r = rmin + (double)i*h;
	        sum += RadialF(spe,GL,Mul,r)*SphB[GL][i];
	      } 
	      sum = sum*h;
	      Spe_RF_Bessel[spe][GL][Mul][kj] = sum;

	    } /* Mul */
	  } /* GL */

	} /* kj */

	/* free SphB */

	for(GL=0; GL<(Spe_MaxL_Basis[spe]+3); GL++){ 
	  free(SphB[GL]);
	}
	free(SphB);

	free(tmp_SphB);
	free(tmp_SphBp);

        /*** added by Ohwaki ***/

        for(LB=0; LB<(List_YOUSO[25]*2+3); LB++){
          free(SphB2[LB]);
        }
        free(SphB2);

      } /* #pragma omp parallel */

    } /* Lspe */

    /****************************************************
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
	for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	  for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){

	    MPI_Bcast(&Spe_RF_Bessel[spe][GL][Mul][0],
		      List_YOUSO[15],MPI_DOUBLE,ID, mpi_comm_level1);
            MPI_Barrier(mpi_comm_level1);
	  }
	}
      }
    }

    /***********************************************************
                      save Spe_RF_Bessel
    ***********************************************************/

    if (myid==Host_ID){

      sprintf(operate,"%s%s_rst",filepath,filename);
      mkdir(operate,0775); 
   
      sprintf(fileFT,"%s%s_rst/%s.ftpao",filepath,filename,filename);

      if ((fp = fopen(fileFT,"wb")) != NULL){

	for (spe=0; spe<SpeciesNum; spe++){
	  for (GL=0; GL<=Spe_MaxL_Basis[spe]; GL++){
	    for (Mul=0; Mul<Spe_Num_Basis[spe][GL]; Mul++){
	      fwrite(&Spe_RF_Bessel[spe][GL][Mul][0],sizeof(double),List_YOUSO[15],fp);
	    }
	  }
	}  

	fclose(fp);
      }
      else{
	printf("Could not open a file %s in FT_PAO\n",fileFT);
      }
    }

  } /* if (RestartRead_Succeed==0) */

  /***********************************************************
                         elapsed time
  ***********************************************************/

  dtime(&TEtime);

}



