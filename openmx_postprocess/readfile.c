/**********************************************************************
  readfile.c:

     readfile.c is a subrutine to read a input file or restart file.

  Log of readfile.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include "tran_prototypes.h"


double readfile(char *argv[])
{ 
  double time0;
  double TStime,TEtime;
  FILE *fp;
  int numprocs,myid; 
  char fileMemory[YOUSO10]; 
  char buf[fp_bsize];          /* setvbuf */

  dtime(&TStime);

  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  /****************************************************
           Read a input file or restart file.
  ****************************************************/

  if ((fp = fopen(argv[1],"r")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
    Input_std(argv[1]);
    fclose(fp);

  }
  else{
    printf("Failure of reading the input file.\n");
    exit(0);
  }

  Allocate_Arrays(2);
  Set_Allocate_Atom2CPU(0,1,0); /* for species */
  SetPara_DFT();

  if ( TRAN_Check_Region_Lead(atomnum, WhatSpecies, Spe_Atom_Cut1, Gxyz, tv)==0 ) {
    printf("\n\nERROR: PAOs of lead atoms can overlap only to the next nearest region.\n\n");
    MPI_Finalize();
    exit(1);
  }

  if ( Solver==4 ) {
    if ( TRAN_Check_Region(atomnum, WhatSpecies, Spe_Atom_Cut1, Gxyz)==0 ) {
      printf("\n\nERROR: PAOs of atoms of L|C|R can overlap only to the next nearest region.\n\n");
      MPI_Finalize();
      exit(1);
    }
  }

  /*****************************
    last input
    0: atomnum, 
    1: elapsed time 
  *****************************/    

  Set_Allocate_Atom2CPU(1,0,0); 

  /***************************************************************
   NEGF:
   check the consistency between the current and previous inputs 
  ***************************************************************/

  TRAN_Check_Input(MPI_COMM_WORLD1, Solver);

  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}




