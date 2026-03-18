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
  int *NPROCS_ID1,*NPROCS_WD1;
  int *Comm_World1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs0);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid0);
  myworld0 = 0;

  Num_Comm_World1 = 1;

  /* allocation of arrays */

  NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs0); 
  NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1); 

  /* Make_Comm_Worlds */

  Make_Comm_Worlds(MPI_COMM_WORLD, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
                   NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

  /* check the result */

  MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

  printf("numprocs0=%2d myid0=%2d myworld1=%2d numprocs1=%2d myid1=%2d\n",
         numprocs0,myid0,myworld1,numprocs1,myid1);

  {

  int numprocs2,myid2,myworld2;
  int *NPROCS_ID2,*NPROCS_WD2;
  int *Comm_World2;
  int *Comm_World_StartID2;
  MPI_Comm *MPI_CommWD2;
  int Num_Comm_World2;

  Num_Comm_World2 = 2;

  /* allocation of arrays */

  NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs1); 
  Comm_World2 = (int*)malloc(sizeof(int)*numprocs1); 
  NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
  Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
  MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2); 

  /* Make_Comm_Worlds */

  Make_Comm_Worlds(MPI_CommWD1[myworld1], myid1, numprocs1, Num_Comm_World2, &myworld2,
                   MPI_CommWD2, NPROCS_ID2, Comm_World2, NPROCS_WD2, Comm_World_StartID2);

  /* check the result */

  MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
  MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);

  printf("numprocs0=%2d myid0=%2d myworld1=%2d numprocs1=%2d myid1=%2d myworld2=%2d numprocs2=%2d myid2=%2d \n",
         numprocs0,myid0,myworld1,numprocs1,myid1,myworld2,numprocs2,myid2);

  /* freeing of arrays */

  free(NPROCS_ID2);
  free(Comm_World2);
  free(NPROCS_WD2);
  free(Comm_World_StartID2);
  free(MPI_CommWD2);


  }


  /* freeing of arrays */

  free(NPROCS_ID1);
  free(Comm_World1);
  free(NPROCS_WD1);
  free(Comm_World_StartID1);
  free(MPI_CommWD1);

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
      MPI_Group_incl(old_group,NPROCS1_WD[i],new_ranks,&new_group);
      MPI_Comm_create(MPI_Curret_Comm_WD,new_group,&MPI_CommWD[i]);

      free(new_ranks); /* never forget cleaning! */
    }
  }

  else {

    numprocs1 = numprocs0;
    *myworld1 = 0;
    for (ID0=0; ID0<numprocs0; ID0++){
      NPROCS1_ID[ID0] = numprocs0;
      Comm_World1[ID0] = 0;

    }
    for (i=0; i<Num_Comm_World; i++){
      Comm_World_StartID[i] = 0;
      NPROCS1_WD[i] = numprocs0;
    }

    for (i=0; i<Num_Comm_World; i++){
      MPI_CommWD[i] = MPI_Curret_Comm_WD;
    }
  }
}





