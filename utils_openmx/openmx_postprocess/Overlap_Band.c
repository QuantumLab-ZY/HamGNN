/**********************************************************************
  Overlap_Band.c:

     Overlap_Band.c is a subroutine to make an overlap matrix for
     periodic boundary systems using the Bloch theorem.

  Log of Overlap_Band.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"


void Overlap_Band(int Host_ID1,
                  double ****OLP,
                  dcomplex **S, int *MP,
                  double k1, double k2, double k3)
{
  static int firsttime=1;
  int i,j,k,wanA,wanB,tnoA,tnoB,Anum,Bnum;
  int NUM,MA_AN,GA_AN,LB_AN,GB_AN;
  int l1,l2,l3,Rn,n2;
  double *tmp_array1,*tmp_array2;
  double kRn,si,co,s;
  int ID,myid,numprocs,tag=999;

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
    tnoA = Spe_Total_CNO[wanA];
    Anum = Anum + tnoA;
  }
  NUM = Anum - 1;
  n2 = NUM + 1;

  /****************************************************
                       Allocation
  ****************************************************/

  tmp_array1 = (double*)malloc(sizeof(double)*2*n2*List_YOUSO[7]);
  tmp_array2 = (double*)malloc(sizeof(double)*2*n2*List_YOUSO[7]);

  /****************************************************
                       set overlap
  ****************************************************/

  S[0][0].r = NUM;
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S[i][j].r = 0.0;
      S[i][j].i = 0.0;
    }
  }

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  s = OLP[MA_AN][LB_AN][i][j];
	  S[Anum+i][Bnum+j].r += s*co;
	  S[Anum+i][Bnum+j].i += s*si;
	}
      }
    }
  }

  /******************************************************
    MPI: S.r and S.i
  ******************************************************/

  if (myid!=Host_ID1){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      k = 0;  
      for (i=0; i<tnoA; i++){
        for (j=0; j<n2; j++){
          tmp_array1[k] = S[Anum+i][j].r;          
	  k++; 
	}
      }
      for (i=0; i<tnoA; i++){
        for (j=0; j<n2; j++){
          tmp_array1[k] = S[Anum+i][j].i;          
	  k++; 
	}
      }

      tag = 999;
      MPI_Isend(&tmp_array1[0], 2*tnoA*n2, MPI_DOUBLE, Host_ID1,
                tag, mpi_comm_level1, &request);
      MPI_Wait(&request,&stat);
    }
  }
  else{
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      ID = G2ID[GA_AN];
      if (ID!=Host_ID1){

        tag = 999;
        MPI_Recv(&tmp_array2[0], 2*tnoA*n2, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);

	k = 0;
        for (i=0; i<tnoA; i++){
          for (j=0; j<n2; j++){
            S[Anum+i][j].r = tmp_array2[k];
	    k++;
	  }
	}
        for (i=0; i<tnoA; i++){
          for (j=0; j<n2; j++){
            S[Anum+i][j].i = tmp_array2[k];
	    k++;
	  }
	}

      }    
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  free(tmp_array1);
  free(tmp_array2);
}
