#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include "mpi.h"



static void dtime(double *t);
static int rnd(int numprocs);


int main(int argc, char *argv[]) 
{
  int numprocs,myid,ID;
  int i,NN; 
  int *Num_Snd,*Num_Rcv;

  double **SndA;
  double **RcvA;
  double TStime,TEtime;
  double sum;

  MPI_Status stat;
  MPI_Request request;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  NN = 3;

  /* allocation of arrays */
  
  Num_Snd = (int*)malloc(sizeof(int)*numprocs);
  Num_Rcv = (int*)malloc(sizeof(int)*numprocs);
  
  /* set data */

  for (ID=0; ID<numprocs; ID++){
    Num_Snd[ID] = 0; 
    Num_Rcv[ID] = 0; 
  }

  for (i=0; i<NN; i++){
    printf("%3d\n",rnd(numprocs));  
  }  



  
  /* 
    First case:
    MPI_Isend and MPI_Recv 
  */






  /* 
    Second case:
    MPI_Isend and MPI_Irecv 
  */


  /*
  SndA = (double**)malloc(sizeof(double*)*numprocs);
  for (i=0; i<numprocs; i++){
    SndA[i] = (double*)malloc(sizeof(double)*asize1);
  }

  RcvA = (double**)malloc(sizeof(double*)*numprocs);
  for (i=0; i<numprocs; i++){
    RcvA[i] = (double*)malloc(sizeof(double)*asize1);
  }


  printf("numprocs=%2d myid=%2d\n",numprocs,myid);

  for (i=0; i<asize1; i++){
    SndA[myid][i] = cos((double)myid*(double)i);
  }

  tag = 999;
 
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);
  */

  /*
  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid){
      MPI_Isend(&SndA[myid][0], asize1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &request);
      MPI_Recv( &RcvA[ID][0], asize1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
  }
  */

  /*
  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid){
      MPI_Isend(&SndA[myid][0], asize1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &request);
    }
  }
  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid){
      MPI_Recv( &RcvA[ID][0], asize1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
    }
  }
  MPI_Wait(&request,&stat);


  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);

  sum = 0.0;
  for (i=0; i<asize1; i++){
    sum += RcvA[1][i];
  }
  printf("myid=%2d elapsed time = %15.10f sum=%15.12f\n",myid,TEtime-TStime,sum);
  */

  MPI_Finalize();
}


void dtime(double *t)
{

  // real time
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;

}


int rnd(int numprocs)
{
  double result;
  result = rand();
  while (2.0<result){
    result = result/2.0;
  }

  return (int)result;
}

