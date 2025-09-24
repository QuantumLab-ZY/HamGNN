/**********************************************************************
  BroadCast_ComplexMatrix.c:

     BroadCast_ComplexMatrix.c is a subroutine to broadcast a matrix "Mat"
     which is distributed by row in each processor.

  Log of BroadCast_ComplexMatrix.c:

     14/Dec/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h> 
#include <math.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "mpi.h"


void BroadCast_ComplexMatrix(MPI_Comm MPI_Curret_Comm_WD, 
                             dcomplex **Mat, int n, int *is1, int *ie1, int myid, int numprocs, 
                             MPI_Status *stat_send,
                             MPI_Request *request_send,
                             MPI_Request *request_recv)
{
  int tag=999;
  long long int i,j,ID,N;
  long long int k,k0,k1,num0,num1;
  double *Mat1;

  N = n;

  /*********************************************
     Elemements are stored from 1 to n in Mat. 
  **********************************************/

  if (numprocs!=1){

    Mat1 = (double*)malloc(sizeof(double)*(N+1)*(N+1));

    /********************************
           Real part of Mat 
    ********************************/

    for (i=is1[myid]; i<=ie1[myid]; i++){
      for (j=1; j<=N; j++){
	k = (i-1)*N + j - 1;
	Mat1[k] = Mat[i][j].r;
      }
    }

    /* receiving */

    for (ID=0; ID<numprocs; ID++){
      k1 = (is1[ID]-1)*N;
      if (k1<0) k1 = 0;  
      num1 = (ie1[ID] - is1[ID] + 1)*N;
      if (num1<0 || ID==myid) num1 = 0;
      MPI_Irecv(&Mat1[k1], num1, MPI_DOUBLE, ID, tag, MPI_Curret_Comm_WD, &request_recv[ID]);
    }

    /* sending */

    k0 = (is1[myid]-1)*N;
    if (k0<0) k0 = 0;  
    num0 = (ie1[myid] - is1[myid] + 1)*N;
    if (num0<0) num0 = 0;

    for (ID=0; ID<numprocs; ID++){
      if (ID!=myid)
        MPI_Isend(&Mat1[k0], num0, MPI_DOUBLE, ID, tag, MPI_Curret_Comm_WD, &request_send[ID]);
      else 
        MPI_Isend(&Mat1[k0], 0,    MPI_DOUBLE, ID, tag, MPI_Curret_Comm_WD, &request_send[ID]);
    }

    /* waitall */

    MPI_Waitall(numprocs,request_recv,stat_send);
    MPI_Waitall(numprocs,request_send,stat_send);

    for (ID=0; ID<numprocs; ID++){
      for (i=is1[ID]; i<=ie1[ID]; i++){
	for (j=1; j<=N; j++){
	  k = (i-1)*N + j - 1;
	  Mat[i][j].r = Mat1[k];
	}
      }
    }

    /********************************
          Imaginary part of Mat 
    ********************************/

    for (i=is1[myid]; i<=ie1[myid]; i++){
      for (j=1; j<=N; j++){
	k = (i-1)*N + j - 1;
	Mat1[k] = Mat[i][j].i;
      }
    }

    /* receiving */

    for (ID=0; ID<numprocs; ID++){
      k1 = (is1[ID]-1)*N;
      if (k1<0) k1 = 0;  
      num1 = (ie1[ID] - is1[ID] + 1)*N;
      if (num1<0 || ID==myid) num1 = 0;
      MPI_Irecv(&Mat1[k1], num1, MPI_DOUBLE, ID, tag, MPI_Curret_Comm_WD, &request_recv[ID]);
    }

    /* sending */

    k0 = (is1[myid]-1)*N;
    if (k0<0) k0 = 0;  
    num0 = (ie1[myid] - is1[myid] + 1)*N;
    if (num0<0) num0 = 0;

    for (ID=0; ID<numprocs; ID++){
      if (ID!=myid)
        MPI_Isend(&Mat1[k0], num0, MPI_DOUBLE, ID, tag, MPI_Curret_Comm_WD, &request_send[ID]);
      else 
        MPI_Isend(&Mat1[k0], 0,    MPI_DOUBLE, ID, tag, MPI_Curret_Comm_WD, &request_send[ID]);
    }

    /* waitall */

    MPI_Waitall(numprocs,request_recv,stat_send);
    MPI_Waitall(numprocs,request_send,stat_send);

    for (ID=0; ID<numprocs; ID++){
      for (i=is1[ID]; i<=ie1[ID]; i++){
	for (j=1; j<=N; j++){
	  k = (i-1)*N + j - 1;
	  Mat[i][j].i = Mat1[k];
	}
      }
    }

    free(Mat1);
  }
}


