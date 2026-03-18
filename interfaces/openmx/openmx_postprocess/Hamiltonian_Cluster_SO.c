/**********************************************************************
  Hamiltonian_Cluster_SO.c:

     Hamiltonian_Cluster_SO.c is a subroutine to make a Hamiltonian
     matrix in spin-collinear with spin-orbit coupling for cluster
     or molecular systems.

  Log of Hamiltonian_Cluster_SO.c:

     8/Jan/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


void Hamiltonian_Cluster_SO(double ****RH, double ****IH, dcomplex **H, int *MP)
{
  static int firsttime=1;
  int i,j,n2;
  int MA_AN,GA_AN,LB_AN,GB_AN;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int ID,myid,numprocs,tag=999;
  double **H1,**H2;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* set MP */
  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum = Anum + Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;
  H[0][0].r = NUM;

  /****************************************************
    allocation of arrays
  ****************************************************/

  n2 = NUM + 2;

  H1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H1[i] = (double*)malloc(sizeof(double)*n2);
  }

  H2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H2[i] = (double*)malloc(sizeof(double)*n2);
  }

  for (i=0; i<=NUM; i++){
    for (j=0; j<=NUM; j++){
      H1[i][j] = 0.0;
      H2[i][j] = 0.0;
    }
  }

  /* for PrintMemory */
  if (firsttime) {
    PrintMemory("Hamiltonian_Cluster_SO: H1",sizeof(double)*n2*n2,NULL);
    PrintMemory("Hamiltonian_Cluster_SO: H2",sizeof(double)*n2*n2,NULL);
  }
  firsttime=0;


  /* set full H */
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  H1[Anum+i][Bnum+j] = RH[MA_AN][LB_AN][i][j];
	  H2[Anum+i][Bnum+j] = IH[MA_AN][LB_AN][i][j];
	}
      }
    }
  }

  /******************************************************
    MPI:
  ******************************************************/

  /* H1 */
  if (myid!=Host_ID){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      tag = 999;
      for (i=0; i<tnoA; i++){
        MPI_Isend(&H1[Anum+i][0], NUM+1, MPI_DOUBLE, Host_ID,
                   tag, mpi_comm_level1, &request);
        MPI_Wait(&request,&stat);
      }
    }
  }
  else{
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      ID = G2ID[GA_AN];
      if (ID!=Host_ID){
        tag = 999;
        for (i=0; i<tnoA; i++){
          MPI_Recv(&H1[Anum+i][0], NUM+1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
	}
      }    
    }
  }  

  /* H2 */
  if (myid!=Host_ID){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      tag = 999;
      for (i=0; i<tnoA; i++){
        MPI_Isend(&H2[Anum+i][0], NUM+1, MPI_DOUBLE, Host_ID,
                   tag, mpi_comm_level1, &request);
        MPI_Wait(&request,&stat);
      }
    }
  }
  else{
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      ID = G2ID[GA_AN];
      if (ID!=Host_ID){
        tag = 999;
        for (i=0; i<tnoA; i++){
          MPI_Recv(&H2[Anum+i][0], NUM+1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
	}
      }    
    }
  }  

  /****************************************************
    putting to H1 and H2 to H
  ****************************************************/

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H[i][j].r = H1[i][j];
      H[i][j].i = H2[i][j];
    }
  }

  /****************************************************
    freeing of arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(H1[i]);
  }
  free(H1);

  for (i=0; i<n2; i++){
    free(H2[i]);
  }
  free(H2);

}


