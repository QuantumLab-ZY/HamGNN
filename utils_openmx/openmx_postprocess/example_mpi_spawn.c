/**********************************************************************

  example_mpi_spawn.c:

  An example of calling OpenMX as library by MPI_Comm_spawn, where 
  a given MPI processes are grouped to three new MPI communication 
  groups and OpenMX runs with three input files: 'Methane.dat', 'C60.dat', 
  and 'Fe2.dat' in each MPI group. 

  Log of example_mpi_spawn.c:

     25/Sep./2019  Released by Taisuke Ozaki

***********************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

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
  int myworld1;
  int *NPROCS1_ID,*NPROCS1_WD;
  int *Comm_World1;
  int *Comm_World_StartID;
  MPI_Comm *MPI_CommWD;
  MPI_Comm comm;
  MPI_Comm *intercomm;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs0);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid0);

  /* set Num_Comm_World1 */

  Num_Comm_World1 = 3;

  /* allocation of arrays */

  NPROCS1_ID = (int*)malloc(sizeof(int)*numprocs0); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs0); 
  NPROCS1_WD = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  Comm_World_StartID = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1); 
  intercomm = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1); 

  /* Make_Comm_Worlds */

  Make_Comm_Worlds(MPI_COMM_WORLD, myid0, numprocs0, Num_Comm_World1, 
                   &myworld1, MPI_CommWD, 
                   NPROCS1_ID, Comm_World1, NPROCS1_WD, Comm_World_StartID);

  /* get numprocs1 and myid1 */

  MPI_Comm_size(MPI_CommWD[myworld1],&numprocs1);
  MPI_Comm_rank(MPI_CommWD[myworld1],&myid1);

  /*
  printf("numprocs0=%2d myid0=%2d myworld1=%2d numprocs1=%2d myid1=%2d\n",
          numprocs0,myid0,myworld1,numprocs1,myid1);
  */

  /* MPI_Comm_spawn */

  char command[] = "./openmx";
  char **argvin; 
  char *inputfiles[] = { "Methane.dat", "C60.dat", "Fe2.dat" }; 

  argvin=(char **)malloc(2 * sizeof(char *)); 
  argvin[0] = inputfiles[myworld1];
  argvin[1] = NULL;

  MPI_Comm_spawn( command, argvin, numprocs1, MPI_INFO_NULL, 0,
                  MPI_CommWD[myworld1], &intercomm[myworld1], MPI_ERRCODES_IGNORE );

  /* MPI_Barrier */

  MPI_Barrier(MPI_COMM_WORLD);

  /* freeing of arrays */

  free(NPROCS1_ID);
  free(Comm_World1);
  free(NPROCS1_WD);
  free(Comm_World_StartID);
  free(MPI_CommWD);
  free(intercomm);

  /* MPI_Finalize() */

  fflush(stdout);
  MPI_Finalize();

  return 0;
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
  int i,j,is,ie;
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

    for (i=0; i<Num_Comm_World; i++){

      is = (int)( (long int)((long int)numprocs0*(long int)(i  ))/(long int)Num_Comm_World);      
      ie = (int)( (long int)((long int)numprocs0*(long int)(i+1))/(long int)Num_Comm_World);      

      if ( is<=myid0 && myid0<ie ){
        numprocs1 = ie - is;
        *myworld1 = i;
      }
    }

    for (i=0; i<Num_Comm_World; i++){

      is = (int)( (long int)((long int)numprocs0*(long int)(i  ))/(long int)Num_Comm_World);      
      ie = (int)( (long int)((long int)numprocs0*(long int)(i+1))/(long int)Num_Comm_World);      

      num = 0;

      for (ID0=0; ID0<numprocs0; ID0++){

        if ( is<=ID0 && ID0<ie ){

          NPROCS1_ID[ID0] = ie - is;
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

      MPI_Group_free(&new_group);
      MPI_Group_free(&old_group);

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
