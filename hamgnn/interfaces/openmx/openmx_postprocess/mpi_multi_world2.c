#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include "mpi.h"

void Make_Comm_Worlds(
   MPI_Comm MPI_Curret_Comm_WD,   
   int myid0,
   int numprocs0,
   int Num_Comm_World, 
   int *myworld1, 
   MPI_Comm *MPI_CommWD,     /* size: Num_Comm_World */
   int *NPROCS1_ID,          /* size: numprocs0 */
   int *Comm_World1,         /* size: numprocs0 */
   int *NPROCS1_WD,          /* size: Num_Comm_World */
   int *Comm_World_StartID   /* size: Num_Comm_World */
   );



int main(int argc, char *argv[]) 
{
  int i,j;
  int numprocs0,myid0,ID0;
  int numprocs1,myid1;
  int num;
  int Num_Comm_World1;
  int myworld0,myworld1;
  int *NPROCS1_ID,*NPROCS1_WD;
  int *Comm_World1;
  int *Comm_World_StartID;
  MPI_Comm *MPI_CommWD;
  MPI_Comm comm;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs0);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid0);
  myworld0 = 0;

  Num_Comm_World1 = 2;

  /* allocation of arrays */

  NPROCS1_ID = (int*)malloc(sizeof(int)*numprocs0); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs0); 
  NPROCS1_WD = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  Comm_World_StartID = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1); 

  /* Make_Comm_Worlds */

  Make_Comm_Worlds(MPI_COMM_WORLD, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD, 
                   NPROCS1_ID, Comm_World1, NPROCS1_WD, Comm_World_StartID);

  /* check the result */

  MPI_Comm_size(MPI_CommWD[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD[myworld1],&myid1);

  printf("numprocs0=%2d myid0=%2d myworld1=%2d numprocs1=%2d myid1=%2d\n",
         numprocs0,myid0,myworld1,numprocs1,myid1);




  MPI_Comm_rank(MPI_CommWD[myworld1],&myid1);

  MPI_Comm_free(&MPI_CommWD[myworld1]);


  /*
  comm = MPI_CommWD[0];
  MPI_Comm_rank(MPI_CommWD[0],&myid1);
  */

  /*
  MPI_Comm_dup( MPI_CommWD[0], &comm );
  MPI_Comm_free(&comm);
  */


  /*
  for (i=0; i<Num_Comm_World1; i++){
    MPI_Comm_free(&MPI_CommWD[i]);
  }
  */

  /* freeing of arrays */
  free(NPROCS1_ID);
  free(Comm_World1);
  free(NPROCS1_WD);
  free(Comm_World_StartID);
  free(MPI_CommWD);

  /* MPI_Finalize() */

  MPI_Finalize();

}




void Make_Comm_Worlds(
   MPI_Comm MPI_Curret_Comm_WD,   
   int myid0,
   int numprocs0,
   int Num_Comm_World, 
   int *myworld1, 
   MPI_Comm *MPI_CommWD,     /* size: Num_Comm_World */
   int *NPROCS1_ID,          /* size: numprocs0 */
   int *Comm_World1,         /* size: numprocs0 */
   int *NPROCS1_WD,          /* size: Num_Comm_World */
   int *Comm_World_StartID   /* size: Num_Comm_World */
   )
{
  int i,j;
  int ID0;
  int numprocs1,myid1;
  int num;
  double avnum;
  int *new_ranks; 
  MPI_Group new_group,old_group; 
  MPI_Comm new_comm;

  /******************************************
     Set up informations to construct the 
     (numprocs0/Num_Comm_World)-th worlds 
  ******************************************/

  if (Num_Comm_World<=numprocs0){

    avnum = (double)numprocs0/(double)Num_Comm_World;
    for (i=0; i<Num_Comm_World; i++){
      if ( (int)((double)(i)*avnum+1.0e-12)<=myid0
         && myid0<(int)((double)(i+1)*avnum+1.0e-12)){
        numprocs1 = (int)((double)(i+1)*avnum+1.0e-12) - (int)((double)(i)*avnum+1.0e-12);
        *myworld1 = i;
      }
    }

    for (i=0; i<Num_Comm_World; i++){
      num = 0;
      for (ID0=0; ID0<numprocs0; ID0++){
        if ( (int)((double)(i)*avnum+1.0e-12)<=ID0
           && ID0<(int)((double)(i+1)*avnum+1.0e-12)){
          NPROCS1_ID[ID0] = (int)((double)(i+1)*avnum+1.0e-12) - (int)((double)(i)*avnum+1.0e-12);
          Comm_World1[ID0] = i;
          if (num==0) Comm_World_StartID[i] = ID0; 
          num++;
        }
      }
 
      NPROCS1_WD[i] = num;
    }
  }

  else {
    printf("Error in Make_Comm_Worlds\n");
    MPI_Finalize();
    exit(0);
  }

  /**************************
   make a set of MPI_CommWD
  **************************/

  for (i=0; i<Num_Comm_World; i++){

    new_ranks = (int*)malloc(sizeof(int)*NPROCS1_WD[i]);

    for (j=0; j<NPROCS1_WD[i]; j++) {
      new_ranks[j] = Comm_World_StartID[i] + j; 
    }

    MPI_Comm_group(MPI_Curret_Comm_WD, &old_group);

    /* define a new group */

    for (j=0; j<NPROCS1_WD[i]; j++) {
      printf("myid0=%2d i=%2d j=%2d new_ranks[j]=%2d\n",myid0,i,j,new_ranks[j]);
    }

    MPI_Group_incl(old_group,NPROCS1_WD[i],new_ranks,&new_group);

    MPI_Comm_create(MPI_Curret_Comm_WD,new_group,&MPI_CommWD[i]);

    /*
    MPI_Comm_create(MPI_Curret_Comm_WD,new_group,&new_comm);
    MPI_Comm_free(&new_comm);

    MPI_Finalize();
    exit(0);
    */

    /*
    MPI_Comm_free(&MPI_CommWD[i]);
    */

    /*
    MPI_Group_free(&new_group);
    */

    free(new_ranks); /* never forget cleaning! */
  }

}





