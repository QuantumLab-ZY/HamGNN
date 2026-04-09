#include <stdio.h>
#include <stdlib.h>  
#include <math.h>
#include <string.h>
#include <time.h>
#include "mpi.h"
#include "openmx_common.h"

#define asize1   10
#define asize2    8
#define asize3    4


int main(int argc, char *argv[]) 
{
  int numprocs,myid,ID,count,tag,i,j,k;
  int numprocs0,myid0;
  int ID0,vsize0;
  double *v0; 
     

  MPI_Status stat;
  MPI_Request request;

  int Pnum[10];

  Pnum[0] = 4;
  Pnum[1] = 8;
  Pnum[2] = 3;
  Pnum[3] = 2;
  Pnum[4] = 7;
  Pnum[5] = 3;
  



  mpi_comm_level1 = MPI_COMM_WORLD; 
  MPI_Init(&argc,&argv);
  MPI_Comm_size(mpi_comm_level1,&numprocs0);
  MPI_Comm_rank(mpi_comm_level1,&myid0);

  printf("numprocs0=%2d myid0=%2d\n",numprocs0,myid0);

  for (k=0; k<6; k++){

    /* make a new group */

    if (Pnum[k]<numprocs0){

       int *new_ranks; 
       MPI_Group  new_group,old_group; 

       new_ranks = (int*)malloc(sizeof(int)*Pnum[k]);
       for (i=0; i<Pnum[k]; i++) {
         new_ranks[i]=i; /* a new group is made of original rank=0:Pnum[k]-1 */
       }

       MPI_Comm_group(MPI_COMM_WORLD, &old_group);

       /* define a new group */
       MPI_Group_incl(old_group,Pnum[k],new_ranks,&new_group);
       MPI_Comm_create(MPI_COMM_WORLD,new_group,&mpi_comm_level1);

       free(new_ranks); /* never forget cleaning! */
    } 

    if (myid0<Pnum[k]){

      MPI_Comm_size(mpi_comm_level1,&numprocs);
      MPI_Comm_rank(mpi_comm_level1,&myid);
      printf("B k=%3d  numprocs=%3d myid=%3d\n",k,numprocs,myid);
    }

    printf("A k=%3d  numprocs=%3d myid=%3d\n",k,numprocs0,myid0);

    /* return original */

    {
    int *new_ranks; 
    MPI_Group  new_group,old_group; 

    new_ranks = (int*)malloc(sizeof(int)*numprocs0);
    for (i=0; i<numprocs0; i++) {
      new_ranks[i]=i; /* a new group is made of original rank=0:Pnum[k]-1 */
    }

    MPI_Comm_group(MPI_COMM_WORLD, &old_group);

    /* define a new group */
    MPI_Group_incl(old_group,numprocs0,new_ranks,&new_group);
    MPI_Comm_create(MPI_COMM_WORLD,new_group,&mpi_comm_level1);

    free(new_ranks); /* never forget cleaning! */

    }

  }

  MPI_Finalize();
  exit(0);



  return 3;


}




