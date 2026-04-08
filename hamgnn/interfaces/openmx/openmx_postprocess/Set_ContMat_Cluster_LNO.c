/**********************************************************************
  Set_ContMat_Cluster_LNO.c:

   Set_ContMat_Cluster_LNO.c is a subroutine to make 
   a contracted matrix represented by LNOs for isolated systems.
   The rountine is used for both overlap and Hamiltonian matrices.

  Log of Set_ContMat_Cluster_LNO.c:

     07/March/2018  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"


void Set_ContMat_Cluster_LNO(double ****OLP0, double *****nh, double ***S, double ***H, int *MP)
{
  int i,j,k,p,p_shift;
  int MA_AN,MB_AN,GA_AN,LB_AN,GB_AN,AN,spin;
  int wanA,wanB,tnoA,tnoA0,tnoB,tnoB0,Anum,Bnum,NUM;
  int num,tnum,num_orbitals;
  int ID,ID_prev,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double *S1,**Stmp,**Htmp,sumS,sumH;
  double *Sc,*Hc; 

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  Stmp = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    Stmp[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  Htmp = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    Htmp[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  Sc = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);
  Hc = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[7]);

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

    GA_AN = M2G[MA_AN];
    tnoA = LNO_Num[GA_AN];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

      GB_AN = natn[GA_AN][LB_AN];
      tnoB = LNO_Num[GB_AN];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num*(SpinP_switch+1)*2;
  }

  MPI_Barrier(mpi_comm_level1);
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  MPI_Barrier(mpi_comm_level1);
  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  MPI_Barrier(mpi_comm_level1);
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += LNO_Num[i];
  }
  NUM = Anum - 1;

  /* set S1
     The way of storing data: 
     S(spin=0), S(spin=1), H(spin=0), H(spin=1), S(spin=0), S(spin=1), H(spin=0), H(spin=1)....
  */

  S1 = (double*)malloc(sizeof(double)*(tnum+5));

  p_shift = My_NZeros[myid]/2;
  p = is1[myid];

  for (spin=0; spin<=SpinP_switch; spin++){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA0 = Spe_Total_CNO[wanA];
      tnoA = LNO_Num[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	MB_AN = S_G2M[GB_AN];

	wanB = WhatSpecies[GB_AN];
	tnoB0 = Spe_Total_CNO[wanB];
	tnoB = LNO_Num[GB_AN];

	/* transformation of representation for OLP0 and nh */

        for (i=0; i<tnoA0; i++){
	  for (j=0; j<tnoB0; j++){
            Sc[j*tnoA0+i] = OLP0[MA_AN][LB_AN][i][j];   
            Hc[j*tnoA0+i] = nh[spin][MA_AN][LB_AN][i][j];   
	  }
	}

        for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB0; j++){

	    sumS = 0.0;
	    sumH = 0.0;

	    for (k=0; k<tnoA0; k++){
	      sumS += LNO_coes[spin][MA_AN][tnoA0*i+k]*Sc[tnoA0*j+k];
	      sumH += LNO_coes[spin][MA_AN][tnoA0*i+k]*Hc[tnoA0*j+k];
	    }

	    Stmp[i][j] = sumS;
	    Htmp[i][j] = sumH;
	  }
	}

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    sumS = 0.0;
	    sumH = 0.0;

	    for (k=0; k<tnoB0; k++){
	      sumS += Stmp[i][k]*LNO_coes[spin][MB_AN][tnoB0*j+k];
	      sumH += Htmp[i][k]*LNO_coes[spin][MB_AN][tnoB0*j+k];
	    }

	    S1[p]         = sumS;
	    S1[p_shift+p] = sumH;

	    p++;
	  }
	}

      }
    }
  } /* spin */

  /* MPI S1 */

  MPI_Barrier(mpi_comm_level1);

  for (ID=0; ID<numprocs; ID++){
    p = is1[ID];
    MPI_Bcast(&S1[p], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* S1 -> S and H */
  /* S1:
     The way of storing data: 
     S(spin=0), S(spin=1), H(spin=0), H(spin=1), S(spin=0), S(spin=1), H(spin=0), H(spin=1).... */

  for (spin=0; spin<=SpinP_switch; spin++){
    S[spin][0][0] = NUM;
    H[spin][0][0] = NUM;
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
	S[spin][i][j] = 0.0;
	H[spin][i][j] = 0.0;
      }
    }
  }

  ID_prev = -1;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (AN=1; AN<=atomnum; AN++){

      GA_AN = order_GA[AN];
      tnoA = LNO_Num[GA_AN];
      Anum = MP[GA_AN];
      ID = G2ID[GA_AN];

      if (ID!=ID_prev){

        if      (spin==0) p = is1[ID];
        else if (spin==1) p = is1[ID] + My_NZeros[ID]/4;

        p_shift = My_NZeros[ID]/2;
        ID_prev = ID;
      }

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	tnoB = LNO_Num[GB_AN];
	Bnum = MP[GB_AN];

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    S[spin][Anum+i][Bnum+j] += S1[p];
	    H[spin][Anum+i][Bnum+j] += S1[p_shift+p];

	    p++;
	  }
	}
      }
    }
  }

  /*
  if (myid==1){
  printf("FFF1 S0\n"); fflush(stdout);
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      printf("%10.5f ",S[0][i][j]);fflush(stdout);
    }
    printf("\n"); fflush(stdout);
  }

  printf("FFF1 H0\n"); fflush(stdout);
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      printf("%10.5f ",H[0][i][j]);fflush(stdout);
    }
    printf("\n"); fflush(stdout);
  }

  printf("FFF1 S1\n"); fflush(stdout);
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      printf("%10.5f ",S[1][i][j]);fflush(stdout);
    }
    printf("\n"); fflush(stdout);
  }

  printf("FFF1 H1\n"); fflush(stdout);
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      printf("%10.5f ",H[1][i][j]);fflush(stdout);
    }
    printf("\n"); fflush(stdout);
  }
  }
  */

  /* freeing of arrays */

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);
  free(S1);

  for (i=0; i<List_YOUSO[7]; i++){
    free(Stmp[i]);
  }
  free(Stmp);

  for (i=0; i<List_YOUSO[7]; i++){
    free(Htmp[i]);
  }
  free(Htmp);

  free(Sc);
  free(Hc);

}
