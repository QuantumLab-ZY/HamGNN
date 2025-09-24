#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include "mpi.h"
#include "openmx_common.h"

#define asize1  819200  // 819200

static void dtime(double *t);

int main(int argc, char *argv[]) 
{

  int numprocs,myid,ID,count,tag,i,j,k;

  double *SndA;
  double *RcvA;
  double TStime,TEtime;

  MPI_Status stat;
  MPI_Request request;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  SndA = (double*)malloc(sizeof(double)*asize1);
  RcvA = (double*)malloc(sizeof(double)*asize1);

  printf("numprocs=%2d myid=%2d\n",numprocs,myid);

  tag = 999;
 
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  for (ID=0; ID<numprocs; ID++){
    if (ID!=myid){
      MPI_Isend(&SndA[0], asize1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &request);
      MPI_Recv( &RcvA[0], asize1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      MPI_Wait(&request,&stat);
    }
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);

  printf("elapsed time = %15.10f\n",TEtime-TStime);

  MPI_Finalize();
}


void dtime(double *t)
{

  // real time
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;

}


