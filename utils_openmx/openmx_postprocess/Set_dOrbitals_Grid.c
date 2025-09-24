/**********************************************************************
  Set_dOrbitals_Grid.c:

   Set_dOrbitals_Grid.c is a subroutine to calculate the value of basis
   functions on each grid point.

  Log of Set_dOrbitals_Grid_ByFukuda.c:

     30/Dec/2014  Released by M.Fukuda

     29/May/2018  Modified by YT Lee
                    for calculating momentum matrix elements

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

double Set_dOrbitals_Grid(int Cnt_kind)
{
  int i,j,n,Mc_AN,Gc_AN,Cwan,NO0,GNc,GRc;
  int Gh_AN,Mh_AN,Rnh,Hwan,NO1,Nog,h_AN;
  long int k,Nc;
  double time0;
  double x,y,z;
  double TStime,TEtime;
  double Cxyz[4];
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom,Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /*****************************************************
                Calculate orbitals on grids
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    if (Cnt_kind==0)  NO0 = Spe_Total_NO[Cwan];
    else              NO0 = Spe_Total_CNO[Cwan]; 

#pragma omp parallel shared(List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,i,j)
    {
      double **dChi;
      double Cxyz0[4]; 
      int i,j;

      /* allocation of array */

      dChi = (double**)malloc(sizeof(double*)*4);
      for (j=0; j<4; j++){
         dChi[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

	GNc = GridListAtom[Mc_AN][Nc]; 
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
	y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
	z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	if (Cnt_kind==0){
          Get_dOrbitals2(Cwan,x,y,z,dChi);
	}
	else{
          Get_Cnt_dOrbitals2(Mc_AN,x,y,z,dChi);
	}

	for (i=0; i<NO0; i++){
	   Orbs_Grid[Mc_AN][Nc][i]    = (Type_Orbs_Grid)dChi[0][i];/* AITUNE */
//	  dOrbs_Grid[0][Mc_AN][Nc][i] = (Type_Orbs_Grid)dChi[1][i];/* AITUNE */
//	  dOrbs_Grid[1][Mc_AN][Nc][i] = (Type_Orbs_Grid)dChi[2][i];/* AITUNE */
//	  dOrbs_Grid[2][Mc_AN][Nc][i] = (Type_Orbs_Grid)dChi[3][i];/* AITUNE */
	}

      } /* Nc */

      /* freeing of array */

      for (j=0; j<4; j++){
         free(dChi[j]);
      }
      free(dChi);

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /****************************************************
     Calculate Orbs_Grid_FNAN
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];

      if (G2ID[Gh_AN]!=myid){

        Mh_AN = F_G2M[Gh_AN];
        Rnh = ncn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];

        if (Cnt_kind==0)  NO1 = Spe_Total_NO[Hwan];
        else              NO1 = Spe_Total_CNO[Hwan];

#pragma omp parallel shared(List_YOUSO,Orbs_Grid_FNAN,NO1,Mh_AN,Hwan,Cnt_kind,Rnh,Gh_AN,Gxyz,atv,NumOLG,Mc_AN,h_AN,GListTAtoms1,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nog,Nc,GNc,GRc,x,y,z,j)
        {

     double **dChi;
	  double Cxyz0[4]; 

          /* allocation of arrays */

      dChi = (double**)malloc(sizeof(double*)*4);
      for (j=0; j<4; j++){
         dChi[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Nog=OMPID*NumOLG[Mc_AN][h_AN]/Nthrds; Nog<(OMPID+1)*NumOLG[Mc_AN][h_AN]/Nthrds; Nog++){

	    Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	    GNc = GridListAtom[Mc_AN][Nc];
	    GRc = CellListAtom[Mc_AN][Nc]; 

	    Get_Grid_XYZ(GNc,Cxyz0);

	    x = Cxyz0[1] + atv[GRc][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
	    y = Cxyz0[2] + atv[GRc][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
	    z = Cxyz0[3] + atv[GRc][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

	    if (Cnt_kind==0){
              Get_dOrbitals(Hwan,x,y,z,dChi);
	    } 
	    else{
              Get_Cnt_dOrbitals(Mh_AN,x,y,z,dChi);
	    }

	    for (j=0; j<NO1; j++){
	       Orbs_Grid_FNAN   [Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)dChi[0][j];/* AITUNE */
//	      dOrbs_Grid_FNAN[0][Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)dChi[1][j];/* AITUNE */
//	      dOrbs_Grid_FNAN[1][Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)dChi[2][j];/* AITUNE */
//	      dOrbs_Grid_FNAN[2][Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)dChi[3][j];/* AITUNE */
	    }

	  } /* Nog */

          /* freeing of arrays */
          for (j=0; j<4; j++){
             free(dChi[j]);
          }
          free(dChi);
        } 
      }
    }
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}

// added by YTLee
double Set_dOrbitals_Grid_xyz(int Cnt_kind,int xyz) // type = 0,1=x,2=y,3=z
{
  int i,j,n,Mc_AN,Gc_AN,Cwan,NO0,GNc,GRc;
  int Gh_AN,Mh_AN,Rnh,Hwan,NO1,Nog,h_AN;
  long int k,Nc;
  double time0;
  double x,y,z;
  double TStime,TEtime;
  double Cxyz[4];
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom,Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /*****************************************************
                Calculate orbitals on grids
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    if (Cnt_kind==0)  NO0 = Spe_Total_NO[Cwan];
    else              NO0 = Spe_Total_CNO[Cwan]; 

#pragma omp parallel shared(List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,i,j)
    {
      double **dChi;
      double Cxyz0[4]; 
      int i,j;

      /* allocation of array */

      dChi = (double**)malloc(sizeof(double*)*4);
      for (j=0; j<4; j++){
         dChi[j] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      }

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

  GNc = GridListAtom[Mc_AN][Nc]; 
  GRc = CellListAtom[Mc_AN][Nc];

  Get_Grid_XYZ(GNc,Cxyz);
  x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
  y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
  z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

  if (Cnt_kind==0){
          Get_dOrbitals2(Cwan,x,y,z,dChi);
  }
  else{
          Get_Cnt_dOrbitals2(Mc_AN,x,y,z,dChi);
  }

  for (i=0; i<NO0; i++){
    Orbs_Grid[Mc_AN][Nc][i]    = (Type_Orbs_Grid)dChi[xyz][i];/* AITUNE */
    //Orbs_Grid[Mc_AN][Nc][i]    = (Type_Orbs_Grid)dChi[0][i];/* AITUNE */
    //dOrbs_Grid[0][Mc_AN][Nc][i] = (Type_Orbs_Grid)dChi[1][i];/* AITUNE */
    //dOrbs_Grid[1][Mc_AN][Nc][i] = (Type_Orbs_Grid)dChi[2][i];/* AITUNE */
    //dOrbs_Grid[2][Mc_AN][Nc][i] = (Type_Orbs_Grid)dChi[3][i];/* AITUNE */
  }

      } /* Nc */

      /* freeing of array */

      for (j=0; j<4; j++){
         free(dChi[j]);
      }
      free(dChi);

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}
