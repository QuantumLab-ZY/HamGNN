/**********************************************************************
  neb_run.c:

     neb_run.c is a code which mediates between neb.c and openmx.

  Log of neb_run.c:

     13th/April/2011,  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>



void neb_run(char *argv[], MPI_Comm mpi_commWD, int index_images, double ***neb_atom_coordinates,
             int *WhatSpecies_NEB, int *Spe_WhatAtom_NEB, char **SpeName_NEB) 
{
  int i,j,k,MD_iter; 
  int numprocs,myid;
  double TStime,TEtime;
  static char fileMemory[YOUSO10]; 

  mpi_comm_level1 = mpi_commWD;
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  NUMPROCS_MPI_COMM_WORLD = numprocs;
  MYID_MPI_COMM_WORLD = myid;
  Num_Procs = numprocs;

  /* for measuring elapsed time */

  dtime(&TStime);

  /* allocation of CompTime */

  CompTime = (double**)malloc(sizeof(double*)*numprocs); 
  for (i=0; i<numprocs; i++){
    CompTime[i] = (double*)malloc(sizeof(double)*20); 
    for (j=0; j<20; j++) CompTime[i][j] = 0.0;
  }

  Init_List_YOUSO();
  remake_headfile = 0;
  ScaleSize = 1.2; 
  
  /****************************************************
                    Read the input file
  ****************************************************/

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  init_alloc_first();

  CompTime[myid][1] = readfile(argv);

  MPI_Barrier(mpi_comm_level1);

  /* initialize PrintMemory routine */

  sprintf(fileMemory,"%s%s.memory%i",filepath,filename,myid);
  PrintMemory(fileMemory,0,"init"); 
  PrintMemory_Fix();

  /* initialize */

  init();

  /* for DFTD-vdW by okuno */
  /* for version_dftD by Ellner*/
  if(dftD_switch==1 && version_dftD==2) DFTDvdW_init();
  if(dftD_switch==1 && version_dftD==3) DFTD3vdW_init();

  /****************************************************
                     SCF-DFT calculations
  ****************************************************/

  MD_iter = 1;

  CompTime[myid][2] += truncation(MD_iter,1);
  CompTime[myid][3] += DFT(MD_iter,(MD_iter-1)%orbitalOpt_per_MDIter+1);

  /****************************************************
   store the total energy, coordinates, and gradients
  ****************************************************/

  /* total energy */
  neb_atom_coordinates[index_images][0][0] = Utot;     

  /* atomic coordinates */

  for (i=1; i<=atomnum; i++){
    neb_atom_coordinates[index_images][i][1] = Gxyz[i][1];
    neb_atom_coordinates[index_images][i][2] = Gxyz[i][2];
    neb_atom_coordinates[index_images][i][3] = Gxyz[i][3];
  }
 
  /* gradients on atoms */

  for (i=1; i<=atomnum; i++){
    neb_atom_coordinates[index_images][i][17] = Gxyz[i][17];
    neb_atom_coordinates[index_images][i][18] = Gxyz[i][18];
    neb_atom_coordinates[index_images][i][19] = Gxyz[i][19];
  }

  /* charge, magnetic moment (muB), and angles of spin */

  for (k=1; k<=atomnum; k++){

    double angle0,angle1;

    i = WhatSpecies[k];
    j = Spe_WhatAtom[i];
    /* Net charge, electron charge is defined to be negative. */
    neb_atom_coordinates[index_images][k][7] = Spe_Core_Charge[i]-(InitN_USpin[k]+InitN_DSpin[k]);
    /* magnetic moment (muB) */
    neb_atom_coordinates[index_images][k][8] = InitN_USpin[k]-InitN_DSpin[k];
    /* angles of spin */ 

    if (SpinP_switch==3){
      angle0 = Angle0_Spin[k]/PI*180.0;
      angle1 = Angle1_Spin[k]/PI*180.0;
    }
    else {
      angle0 = 0.0;
      angle1 = 0.0;
    }

    neb_atom_coordinates[index_images][k][ 9] = angle0;
    neb_atom_coordinates[index_images][k][10] = angle1;
  }

  /****************************************************
    store WhatSpecies_NEB, Spe_WhatAtom_NEB, 
    and SpeName_NEB 
  ****************************************************/

  for (i=1; i<=atomnum; i++){
    WhatSpecies_NEB[i] = WhatSpecies[i];
  }

  for (i=0; i<SpeciesNum; i++){
    Spe_WhatAtom_NEB[i] = Spe_WhatAtom[i];
  }

  for (i=0; i<SpeciesNum; i++){
    sprintf(SpeName_NEB[i],"%s",SpeName[i]);
  }

  /****************************************************
               finalize the calculation
  ****************************************************/

  /* elapsed time */

  dtime(&TEtime);
  CompTime[myid][0] = TEtime - TStime;
  Output_CompTime();
  for (i=0; i<numprocs; i++){
    free(CompTime[i]);
  }
  free(CompTime);

  /* merge log files */

  Merge_LogFile(argv[1]);

  /* print memory */

  PrintMemory("total",0,"sum");

  /* free arrays */

  Free_Arrays(0);
}
