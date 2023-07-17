/**********************************************************************
  Make_FracCoord.c:

     Make_FracCoord.c is a subrutine to generate a file including 
     the fractional coordinates of the system.

  Log of Make_FracCoord.c:

     22/Nov/2007  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*  end stat section */
#include "openmx_common.h"
#include "mpi.h"


void Make_FracCoord(char *file)
{
  int i,k,Mc_AN,Gc_AN,ct_AN;
  int itmp;
  int numprocs,myid,ID,tag=999;
  double Cxyz[4];
  char fname1[YOUSO10];
  FILE *fp;

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){

    sprintf(fname1,"%s%s.frac",filepath,filename);

    if ((fp = fopen(fname1,"w")) != NULL){

      fprintf(fp,"\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"       Fractional coordinates of the final structure       \n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n\n");

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

        /* The zero is taken as the origin of the unit cell. */

	Cxyz[1] = Gxyz[Gc_AN][1];
	Cxyz[2] = Gxyz[Gc_AN][2];
	Cxyz[3] = Gxyz[Gc_AN][3];

	Cell_Gxyz[Gc_AN][1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	Cell_Gxyz[Gc_AN][2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	Cell_Gxyz[Gc_AN][3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

        /* The fractional coordinates are kept within 0 to 1. */

        for (i=1; i<=3; i++){

          itmp = (int)Cell_Gxyz[Gc_AN][i]; 

	  if (1.0<Cell_Gxyz[Gc_AN][i]){
	    Cell_Gxyz[Gc_AN][i] = fabs(Cell_Gxyz[Gc_AN][i] - (double)itmp);
	  }
	  else if (Cell_Gxyz[Gc_AN][i]<-1.0e-13){
	    Cell_Gxyz[Gc_AN][i] = fabs(Cell_Gxyz[Gc_AN][i] + (double)(abs(itmp)+1));
	  }
	}

        k = WhatSpecies[Gc_AN];

        fprintf(fp,"%6d   %4s   %18.14f %18.14f %18.14f\n",
                Gc_AN,SpeName[k],
                Cell_Gxyz[Gc_AN][1],
                Cell_Gxyz[Gc_AN][2],
                Cell_Gxyz[Gc_AN][3]);
      }

      fclose(fp);
    }
  }

}
