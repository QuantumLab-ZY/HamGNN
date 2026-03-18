/**********************************************************************
  Voronoi_Charge.c:

     Voronoi_Charge.c is a subroutine to calculate Voronoi charges

  Log of Voronoi_Charge.c:

     2/Feb/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>


void Voronoi_Charge()
{
  double time0;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int Cwan,GNc,GRc,Nog,Nh,MN,spin;
  double x,y,z,dx,dy,dz,fw;
  double Cxyz[4];
  double FuzzyW,sum0,sum1;
  double magx,magy,magz;
  double tmagx,tmagy,tmagz;
  double tden,tmag,theta,phi,rho,mag;
  double den0,den1,vol;
  double VC_S,T_VC0,T_VC1;
  double **VC,*Voronoi_Vol;
  double TStime,TEtime;
  double S_coordinate[3];
  int numprocs,myid,tag=999,ID;
  FILE *fp_VC;
  char file_VC[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);
  if (myid==Host_ID) printf("\n<Voronoi_Charge>  calculate Voronoi charges\n");fflush(stdout);

  /*****************************************************
    allocation of array
  *****************************************************/

  VC = (double**)malloc(sizeof(double*)*4);
  for (spin=0; spin<4; spin++){
    VC[spin] = (double*)malloc(sizeof(double)*(atomnum+1));
  }

  Voronoi_Vol = (double*)malloc(sizeof(double)*(atomnum+1));

  /*****************************************************
            calculation of Voronoi charge
  *****************************************************/

#pragma omp parallel shared(S_coordinate,GridVol,VC,Voronoi_Vol,Density_Grid,SpinP_switch,MGridListAtom,atv,CellListAtom,GridListAtom,NumOLG,WhatSpecies,M2G,Matomnum) private(OMPID,Nthrds,Nprocs,Mc_AN,Gc_AN,Cwan,sum0,sum1,vol,tden,tmagx,tmagy,tmagz,Nog,GNc,GRc,Cxyz,x,y,z,FuzzyW,MN,den0,den1,theta,phi,rho,mag,magx,magy,magz,tmag)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=1+OMPID; Mc_AN<=Matomnum; Mc_AN+=Nthrds){

      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];

      sum0 = 0.0;
      sum1 = 0.0;
      vol  = 0.0;

      tden  = 0.0;
      tmagx = 0.0;
      tmagy = 0.0;
      tmagz = 0.0;

      for (Nog=0; Nog<NumOLG[Mc_AN][0]; Nog++){

	/* calculate fuzzy weight */

	GNc = GridListAtom[Mc_AN][Nog];
	GRc = CellListAtom[Mc_AN][Nog];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1];
	y = Cxyz[2] + atv[GRc][2]; 
	z = Cxyz[3] + atv[GRc][3];
	FuzzyW = Fuzzy_Weight(Gc_AN,Mc_AN,0,x,y,z);

	/* find charge */

	MN = MGridListAtom[Mc_AN][Nog];

	if (SpinP_switch<=1){

	  den0  = Density_Grid[0][MN];
	  den1  = Density_Grid[1][MN];

	  /* sum density */
	  sum0 += den0*FuzzyW; 
	  sum1 += den1*FuzzyW; 

	  /* sum volume */
          vol += FuzzyW;

	}

	else{

	  den0  = Density_Grid[0][MN];
	  den1  = Density_Grid[1][MN];
	  theta = Density_Grid[2][MN];
	  phi   = Density_Grid[3][MN];

	  rho = den0 + den1;
	  mag = den0 - den1;
	  magx = mag*sin(theta)*cos(phi);
	  magy = mag*sin(theta)*sin(phi);
	  magz = mag*cos(theta);

	  /* sum density */
 
	  tden  +=  rho*FuzzyW; 
	  tmagx += magx*FuzzyW; 
	  tmagy += magy*FuzzyW; 
	  tmagz += magz*FuzzyW; 

	  /* sum volume */
          vol += FuzzyW;
	}

      }

      if (SpinP_switch<=1){
	VC[0][Gc_AN] = sum0*GridVol; 
	VC[1][Gc_AN] = sum1*GridVol;
      }

      else {

	tmag = sqrt(tmagx*tmagx + tmagy*tmagy + tmagz*tmagz); 
	sum0 = 0.5*(tden + tmag);
	sum1 = 0.5*(tden - tmag);

	xyz2spherical( tmagx,tmagy,tmagz, 0.0,0.0,0.0, S_coordinate ); 

	VC[0][Gc_AN] = sum0*GridVol; 
	VC[1][Gc_AN] = sum1*GridVol;
	VC[2][Gc_AN] = S_coordinate[1];
	VC[3][Gc_AN] = S_coordinate[2];
      }

      Voronoi_Vol[Gc_AN] = vol*GridVol*BohrR*BohrR*BohrR;

    } /* Mc_AN */

  } /* #pragma omp parallel */

  /*****************************************************
    MPI VC
  *****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&VC[0][Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&VC[1][Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (SpinP_switch==3){

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      ID = G2ID[Gc_AN];
      MPI_Bcast(&VC[2][Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    }

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      ID = G2ID[Gc_AN];
      MPI_Bcast(&VC[3][Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    }
  }

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Voronoi_Vol[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  VC_S = 0.0;
  T_VC0 = 0.0;
  T_VC1 = 0.0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    VC_S += VC[0][Gc_AN] - VC[1][Gc_AN];  
    T_VC0 += VC[0][Gc_AN];
    T_VC1 += VC[1][Gc_AN];
  }

  /****************************************
   file, *.VC
  ****************************************/

  if ( myid==Host_ID ){

    sprintf(file_VC,"%s%s.VC",filepath,filename);

    if ((fp_VC = fopen(file_VC,"w")) != NULL){

#ifdef xt3
      setvbuf(fp_VC,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      fprintf(fp_VC,"\n");
      fprintf(fp_VC,"***********************************************************\n");
      fprintf(fp_VC,"***********************************************************\n");
      fprintf(fp_VC,"                     Voronoi charges                       \n");
      fprintf(fp_VC,"***********************************************************\n");
      fprintf(fp_VC,"***********************************************************\n\n");

      fprintf(fp_VC,"  Sum of Voronoi charges for up    = %15.12f\n", T_VC0);
      fprintf(fp_VC,"  Sum of Voronoi charges for down  = %15.12f\n", T_VC1);
      fprintf(fp_VC,"  Sum of Voronoi charges for total = %15.12f\n\n",
              T_VC0+T_VC1);

      fprintf(fp_VC,"  Total spin magnetic moment (muB) by Voronoi charges  = %15.12f\n\n",VC_S);

      if (SpinP_switch<=1){

	fprintf(fp_VC,"                     Up spin      Down spin     Sum           Diff       Voronoi Volume (Ang.^3)\n");
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  fprintf(fp_VC,"       Atom=%4d  %12.9f %12.9f  %12.9f  %12.9f  %12.9f\n",
		  Gc_AN, VC[0][Gc_AN], VC[1][Gc_AN],
		  VC[0][Gc_AN] + VC[1][Gc_AN],
		  VC[0][Gc_AN] - VC[1][Gc_AN],
                  Voronoi_Vol[Gc_AN]);
	}
      }

      else{
	fprintf(fp_VC,"                     Up spin      Down spin     Sum           Diff        Theta(Deg)   Phi(Deg)   Voronoi Volume (Ang.^3)\n");
	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	  fprintf(fp_VC,"       Atom=%4d  %12.9f %12.9f  %12.9f  %12.9f  %8.4f    %8.4f   %12.9f\n",
		  Gc_AN, VC[0][Gc_AN], VC[1][Gc_AN],
		  VC[0][Gc_AN] + VC[1][Gc_AN],
		  VC[0][Gc_AN] - VC[1][Gc_AN],
                  VC[2][Gc_AN]/PI*180.0,VC[3][Gc_AN]/PI*180.0,
                  Voronoi_Vol[Gc_AN]);
	}
      }

      fclose(fp_VC);
    }
    else{
      printf("Failure of saving the VC file.\n");
    }

  }

  /*****************************************************
    freeing of array
  *****************************************************/

  for (spin=0; spin<4; spin++){
    free(VC[spin]);
  }
  free(VC);

  free(Voronoi_Vol);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

}

