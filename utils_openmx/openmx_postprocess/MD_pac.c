/**********************************************************************
  MD_pac.c:

     MD_pac.c is a subroutine to perform molecular dynamics
     simulations and geometry optimization.

  Log of MD_pac.c:

     22/Nov/2001  Released by T. Ozaki
     15/Dec/2003  DIISs are added by H. Kino
     14/May/2004  NVT_VS is added by M. Ohfuti
     25/May/2004  Modified by T. Ozaki
     14/Jul/2007  RF added by H.M. Weng
     08/Jan/2010  NVT_VS2 added by T. Ohwaki 
     23/Dec/2012  RestartFiles4GeoOpt added by T. Ozaki
     08/Oct/2019  NPT-MD added by MIZUHO, T. Iitaka, and T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"


#define Criterion_Max_Step            0.20

 
static void NoMD(int iter);
static void VerletXYZ(int iter);
static void NVT_VS(int iter);  /* added by mari */
static void NVT_VS2(int iter); /* added by Ohwaki */
static void NVT_VS4(int iter); /* added by Ohwaki */
static void NVT_NH(int iter); 
static void NVT_Langevin(int iter); /* added by Ohwaki */
static void Steepest_Descent(int iter, int SD_scaling_flag);
static void GDIIS(int iter, int iter0);
static void GDIIS_BFGS(int iter, int iter0);
static void GDIIS_EF(int iter, int iter0);
static void Geometry_Opt_DIIS(int iter);
static void Geometry_Opt_DIIS_BFGS(int iter);
static void Geometry_Opt_DIIS_EF(int iter);
static void Correct_Position_In_First_Cell();
static void Geometry_Opt_RF(int iter);
static void Estimate_Initial_Hessian(int diis_iter, int CellOpt_flag, double itv[4][4]);
static void RF(int iter, int iter0);
static void RFC5(int iter, int iter0, double dE_da[4][4], double itv[4][4]);
static void EvsLC(int iter);
static void Delta_Factor(int iter);
static void Correct_Force();
static void Correct_Velocity();
static int RestartFiles4GeoOpt(char *mode);
static void Output_abc_file(int iter);
static void Cell_Opt_SD(int iter, int SD_scaling_flag);
static void Cell_Opt_RF(int iter);
static void NPT_VS_PR(int iter); /* added by MIZUHO for NPT-MD */
static void NPT_VS_WV(int iter); /* added by MIZUHO for NPT-MD */
static void NPT_NH_PR(int iter); /* added by MIZUHO for NPT-MD */
static void NPT_NH_WV(int iter); /* added by MIZUHO for NPT-MD */

int SuccessReadingfiles;




double MD_pac(int iter, char *fname_input)
{
  double time0;
  double TStime,TEtime;
  int numprocs,myid;
  int i,j,k;

  dtime(&TStime);
 
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("\n*******************************************************\n"); 
    printf("             MD or geometry opt. at MD =%2d              \n",iter);
    printf("*******************************************************\n\n"); 
  }

  if (myid==Host_ID && MD_OutABC==1) Output_abc_file(iter);

  /*********************************************** 
    read restart files for geometry optimization
    MD_switch==3: Steepest_Descent
    MD_switch==4: Geometry_Opt_DIIS_EF
    MD_switch==5: Geometry_Opt_DIIS_BFGS
    MD_switch==6: Geometry_Opt_RF
    MD_switch==7: Geometry_Opt_DIIS
  ***********************************************/

  if (iter==1 && GeoOpt_RestartFromFile){
    if ( MD_switch==3 || MD_switch==4 || MD_switch==5 || MD_switch==6 || MD_switch==7 ){
      if (RestartFiles4GeoOpt("read")){
        SuccessReadingfiles = 1;
        if (myid==Host_ID){
          printf("Found restart files for geometry optimization\n");
        }
      }
    }
  }
  else {
    SuccessReadingfiles = 0;
  }

  /* correct forces for MD simulations so that the sum of forces can be zero. */

  if (  MD_switch==1 || MD_switch==2 || MD_switch==9 
     || MD_switch==11 || MD_switch==14 || MD_switch==15 ) Correct_Force();

  /* Call a subroutine based on MD_switch */

  switch (MD_switch) {
    case  0: NoMD(iter+MD_Current_Iter);                    break;
    case  1: VerletXYZ(iter+MD_Current_Iter);               break;
    case  2: NVT_VS(iter+MD_Current_Iter);                  break;  /* added by mari */
    case  3: Steepest_Descent(iter+MD_Current_Iter,1);      break;
    case  4: Geometry_Opt_DIIS_EF(iter+MD_Current_Iter);    break;
    case  5: Geometry_Opt_DIIS_BFGS(iter+MD_Current_Iter);  break;
    case  6: Geometry_Opt_RF(iter+MD_Current_Iter);         break;  /* added by hmweng */
    case  7: Geometry_Opt_DIIS(iter+MD_Current_Iter);       break;
    case  8:                                                break;  /* not used */
    case  9: NVT_NH(iter+MD_Current_Iter);                  break;
    case 10:                                                break;  /* not used */
    case 11: NVT_VS2(iter+MD_Current_Iter);                 break;  /* added by Ohwaki */
    case 12: EvsLC(iter);                                   break;
    case 14: NVT_VS4(iter+MD_Current_Iter);                 break;  /* added by Ohwaki */
    case 15: NVT_Langevin(iter+MD_Current_Iter);            break;  /* added by Ohwaki */
    case 16: Delta_Factor(iter);                            break;  /* delta-factor */
    case 17: Cell_Opt_SD(iter+MD_Current_Iter,1);           break;
    case 18: Cell_Opt_RF(iter+MD_Current_Iter);             break;
    /* added by MIZUHO for NPT-MD */
    case 27: NPT_VS_PR(iter+MD_Current_Iter);               break;
    case 28: NPT_VS_WV(iter+MD_Current_Iter);               break;
    case 29: NPT_NH_PR(iter+MD_Current_Iter);               break;
    case 30: NPT_NH_WV(iter+MD_Current_Iter);               break;
  }

  /***************************************************************
    correct atoms which are out of the first unit cell during 
    molecular dynamics simulations. The correction is not applied 
    for geometry optimization.
  ***************************************************************/

  if (   MD_switch==1  || // NVE
         MD_switch==2  || // NVT_VS
         MD_switch==9  || // NVT_NH
         MD_switch==11 || // NVT_VS2
         MD_switch==14 || // NVT_VS4
         MD_switch==15 || // NVT_Langevin
         /* added by MIZUHO for NPT-MD */
         MD_switch==27 || // NPT_VS_PR
         MD_switch==28 || // NPT_VS_WV
         MD_switch==29 || // NPT_NH_PR
         MD_switch==30 )  // NPT_NH_WV
  {

    Correct_Position_In_First_Cell();
  }

  /* making of an input file with the final structure */

  if (Runtest_flag==0){

    Make_InputFile_with_FinalCoord(fname_input,iter+MD_Current_Iter);
  }

  /*********************************************** 
    save restart files for geometry optimization
    MD_switch==4: Geometry_Opt_DIIS_EF
    MD_switch==5: Geometry_Opt_DIIS_BFGS
    MD_switch==6: Geometry_Opt_RF
    MD_switch==7: Geometry_Opt_DIIS
  ***********************************************/

  if ( MD_switch==3 || MD_switch==4 || MD_switch==5 || MD_switch==6 || MD_switch==7 ){
    RestartFiles4GeoOpt("write");
  }

  MPI_Bcast(&MD_Opt_OK, 1, MPI_INT, Host_ID, mpi_comm_level1);

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}


void NoMD(int iter)
{
  char fileCoord[YOUSO10];
  FILE *fp_crd,*fp_SD;
  int i,j,k;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 1;

  if (myid==Host_ID){ 

    if (MD_Opt_OK==1 || iter==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

}

    


void Correct_Position_In_First_Cell()
{
  int i,Mc_AN,Gc_AN,ct_AN,k;
  int itmp,My_Correct_Position_flag;
  int numprocs,myid,ID,tag=999;
  double Cxyz[4],Frac[4];

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* My_Correct_Position_flag  */

  My_Correct_Position_flag = 0; 

  /* loop for Mc_AN */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cxyz[1] = Gxyz[Gc_AN][1] - Grid_Origin[1];
    Cxyz[2] = Gxyz[Gc_AN][2] - Grid_Origin[2];
    Cxyz[3] = Gxyz[Gc_AN][3] - Grid_Origin[3];
    Cell_Gxyz[Gc_AN][1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
    Cell_Gxyz[Gc_AN][2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
    Cell_Gxyz[Gc_AN][3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

    /* The fractional coordinates are kept within 0 to 1. */

    for (i=1; i<=3; i++){

      itmp = (int)Cell_Gxyz[Gc_AN][i]; 

      if (1.0<Cell_Gxyz[Gc_AN][i]){

        My_Correct_Position_flag = 1; 

	Cell_Gxyz[Gc_AN][i] = Cell_Gxyz[Gc_AN][i] - (double)itmp;

        if (0<level_stdout){ 
  	  if (i==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",Gc_AN);
	  if (i==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",Gc_AN);
	  if (i==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",Gc_AN);
	}

        /* update His_Gxyz */ 

        for (k=0; k<Extrapolated_Charge_History; k++){

	  Cxyz[1] = His_Gxyz[k][(Gc_AN-1)*3+0] - Grid_Origin[1];
	  Cxyz[2] = His_Gxyz[k][(Gc_AN-1)*3+1] - Grid_Origin[2];
	  Cxyz[3] = His_Gxyz[k][(Gc_AN-1)*3+2] - Grid_Origin[3];

	  Frac[1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	  Frac[2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	  Frac[3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

          Frac[i] = Frac[i] - (double)itmp;
          
          His_Gxyz[k][(Gc_AN-1)*3+0] = 
                      Frac[1]*tv[1][1]
                    + Frac[2]*tv[2][1]
                    + Frac[3]*tv[3][1] + Grid_Origin[1];

          His_Gxyz[k][(Gc_AN-1)*3+1] = 
                      Frac[1]*tv[1][2]
                    + Frac[2]*tv[2][2]
                    + Frac[3]*tv[3][2] + Grid_Origin[2];

          His_Gxyz[k][(Gc_AN-1)*3+2] = 
                      Frac[1]*tv[1][3]
                    + Frac[2]*tv[2][3]
                    + Frac[3]*tv[3][3] + Grid_Origin[3];
	}

        /* update GxyzHistoryIn */ 

        for (k=0; k<(M_GDIIS_HISTORY+1); k++){

	  Cxyz[1] = GxyzHistoryIn[k][Gc_AN][1] - Grid_Origin[1];
	  Cxyz[2] = GxyzHistoryIn[k][Gc_AN][2] - Grid_Origin[2];
	  Cxyz[3] = GxyzHistoryIn[k][Gc_AN][3] - Grid_Origin[3];

	  Frac[1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	  Frac[2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	  Frac[3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

          Frac[i] = Frac[i] - (double)itmp;

          GxyzHistoryIn[k][Gc_AN][1] = 
                      Frac[1]*tv[1][1]
                    + Frac[2]*tv[2][1]
                    + Frac[3]*tv[3][1] + Grid_Origin[1];

          GxyzHistoryIn[k][Gc_AN][2] = 
                      Frac[1]*tv[1][2]
                    + Frac[2]*tv[2][2]
                    + Frac[3]*tv[3][2] + Grid_Origin[2];

          GxyzHistoryIn[k][Gc_AN][3] = 
                      Frac[1]*tv[1][3]
                    + Frac[2]*tv[2][3]
                    + Frac[3]*tv[3][3] + Grid_Origin[3];
	}

      }
      else if (Cell_Gxyz[Gc_AN][i]<0.0){

        My_Correct_Position_flag = 1; 

	Cell_Gxyz[Gc_AN][i] = Cell_Gxyz[Gc_AN][i] + (double)(abs(itmp)+1);

        if (0<level_stdout){ 
          if (i==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",Gc_AN);
	  if (i==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",Gc_AN);
	  if (i==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",Gc_AN);
	}

        /* update His_Gxyz */ 

        for (k=0; k<Extrapolated_Charge_History; k++){

	  Cxyz[1] = His_Gxyz[k][(Gc_AN-1)*3+0] - Grid_Origin[1];
	  Cxyz[2] = His_Gxyz[k][(Gc_AN-1)*3+1] - Grid_Origin[2];
	  Cxyz[3] = His_Gxyz[k][(Gc_AN-1)*3+2] - Grid_Origin[3];

	  Frac[1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	  Frac[2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	  Frac[3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

          Frac[i] = Frac[i] + (double)(abs(itmp)+1);
          
          His_Gxyz[k][(Gc_AN-1)*3+0] = 
                      Frac[1]*tv[1][1]
                    + Frac[2]*tv[2][1]
                    + Frac[3]*tv[3][1] + Grid_Origin[1];

          His_Gxyz[k][(Gc_AN-1)*3+1] = 
                      Frac[1]*tv[1][2]
                    + Frac[2]*tv[2][2]
                    + Frac[3]*tv[3][2] + Grid_Origin[2];

          His_Gxyz[k][(Gc_AN-1)*3+2] = 
                      Frac[1]*tv[1][3]
                    + Frac[2]*tv[2][3]
                    + Frac[3]*tv[3][3] + Grid_Origin[3];
	}

        /* update GxyzHistoryIn */ 

        for (k=0; k<(M_GDIIS_HISTORY+1); k++){

	  Cxyz[1] = GxyzHistoryIn[k][Gc_AN][1] - Grid_Origin[1];
	  Cxyz[2] = GxyzHistoryIn[k][Gc_AN][2] - Grid_Origin[2];
	  Cxyz[3] = GxyzHistoryIn[k][Gc_AN][3] - Grid_Origin[3];

	  Frac[1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	  Frac[2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	  Frac[3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

          Frac[i] = Frac[i] + (double)(abs(itmp)+1);

          GxyzHistoryIn[k][Gc_AN][1] = 
                      Frac[1]*tv[1][1]
                    + Frac[2]*tv[2][1]
                    + Frac[3]*tv[3][1] + Grid_Origin[1];

          GxyzHistoryIn[k][Gc_AN][2] = 
                      Frac[1]*tv[1][2]
                    + Frac[2]*tv[2][2]
                    + Frac[3]*tv[3][2] + Grid_Origin[2];

          GxyzHistoryIn[k][Gc_AN][3] = 
                      Frac[1]*tv[1][3]
                    + Frac[2]*tv[2][3]
                    + Frac[3]*tv[3][3] + Grid_Origin[3];
	}

      }
    }

    Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

    Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

    Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
  }

  /****************
    MPI:  Gxyz
  *****************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    ID = G2ID[ct_AN];

    MPI_Bcast(&Gxyz[ct_AN][0], 4, MPI_DOUBLE, ID, mpi_comm_level1);

    for (k=0; k<Extrapolated_Charge_History; k++){
      MPI_Bcast(&His_Gxyz[k][(ct_AN-1)*3], 3, MPI_DOUBLE, ID, mpi_comm_level1);
    }

    for (k=0; k<(M_GDIIS_HISTORY+1); k++){
      MPI_Bcast(&GxyzHistoryIn[k][ct_AN][0], 4, MPI_DOUBLE, ID, mpi_comm_level1);
    }

  }

  /*
  for (k=0; k<(M_GDIIS_HISTORY+1); k++){
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      printf("myid=%2d k=%2d ct_AN=%2d %15.12f %15.12f %15.12f\n",
             myid,k,ct_AN,
             GxyzHistoryIn[k][ct_AN][1],
             GxyzHistoryIn[k][ct_AN][2],
             GxyzHistoryIn[k][ct_AN][3]);fflush(stdout);
    }
  }
  */

  /* MPI:  My_Correct_Position_flag = 0; */

  MPI_Allreduce(&My_Correct_Position_flag, &Correct_Position_flag, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
}


void VerletXYZ(int iter)
{
  /***********************************************************
   NVE molecular dynamics with velocity-Verlet integrator
  ***********************************************************/
  /*********************************************************** 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ***********************************************************/
  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step

    Gxyz[][27] = x-component of velocity at t+dt/2
    Gxyz[][28] = y-component of velocity at t+dt/2
    Gxyz[][29] = z-component of velocity at t+dt/2

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  char fileE[YOUSO10];
  double dt,dt2,back,sum,My_Ukc;
  double Wscale,scaled_force;
  int Mc_AN,Gc_AN,j,k,l;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /****************************************************
                 velocity-Verlet algorithm
  ****************************************************/

  if (iter==1){

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    My_Ukc = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      sum = 0.0;
      for (j=1; j<=3; j++){
        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	}
      }
      My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    }

    /****************************************************
     MPI, Ukc
    ****************************************************/

    MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /* calculation of temperature (K) */
    Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  
      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(iter,MD_TimeStep*(iter-1),fileE);
    }

    /****************************************************
      first step in velocity Verlet 
    ****************************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){

          scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  

          /* v( t+0.5*dt ) */
          Gxyz[Gc_AN][26+j] = Gxyz[Gc_AN][23+j] + scaled_force*0.5*dt;
          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j];

          /* r( t+dt ) */
          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
 	  Gxyz[Gc_AN][j] =  Gxyz[Gc_AN][j] + Gxyz[Gc_AN][26+j]*dt;

	}
      }
    }
  }
  else{

    /****************************************************
      second step in velocity Verlet 
    ****************************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  
          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j] + scaled_force*0.5*dt;
	}
      }
    }

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    My_Ukc = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      sum = 0.0;
      for (j=1; j<=3; j++){
        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	}
      }
      My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    }

    /****************************************************
     MPI, Ukc
    ****************************************************/

    MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /* calculation of temperature (K) */
    Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(iter,MD_TimeStep*(iter-1),fileE);
    } 

    /****************************************************
      first step in velocity Verlet 
    ****************************************************/

    if (iter!=MD_IterNumber){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	for (j=1; j<=3; j++){

	  if (atom_Fixed_XYZ[Gc_AN][j]==0){

	    scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  
	    /* v( t+0.5*dt ) */
	    Gxyz[Gc_AN][26+j] = Gxyz[Gc_AN][23+j] + scaled_force*0.5*dt;
	    Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j];

	    /* r( t+dt ) */
	    Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
	    Gxyz[Gc_AN][j] =  Gxyz[Gc_AN][j] + Gxyz[Gc_AN][26+j]*dt;
	  }

	}

      }
    }
  }

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17],13, MPI_DOUBLE, ID, mpi_comm_level1);
  }
}




void Steepest_Descent(int iter, int SD_scaling_flag)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */
  int i,j,k,l,Mc_AN,Gc_AN;
  double dt,SD_max,SD_min,SD_init,Atom_W,tmp0,scale;
  double My_Max_Force,Wscale;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  int numprocs,myid,ID;
  double tmp1,MaxStep;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  char file_name[YOUSO10];
  FILE *fp;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /******************************************
              read *.rst4gopt.SD1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(&Correct_Position_flag, sizeof(int), 1, fp);
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /****************************************************
   find the maximum value of force 
  ****************************************************/

  My_Max_Force = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    tmp0 = 0.0;
    for (j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
        tmp0 += Gxyz[Gc_AN][16+j]*Gxyz[Gc_AN][16+j];
      }
    }
    tmp0 = sqrt(tmp0); 
    if (My_Max_Force<tmp0) My_Max_Force = tmp0;
  }

  /****************************************************
   MPI, Max_Force
  ****************************************************/

  MPI_Allreduce(&My_Max_Force, &Max_Force, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_level1);
  if (Max_Force<MD_Opt_criterion) MD_Opt_OK = 1;

  /****************************************************
   set up SD_scaling
  ****************************************************/

  dt = 41.3411*2.0;
  SD_init = dt*dt/Wscale;
  SD_max = SD_init*15.0;   /* default 15   */
  SD_min = SD_init*0.04;   /* default 0.02 */
  Atom_W = 12.0;

  if (iter==1 || SD_scaling_flag==0){

    SD_scaling_user = Max_Force/BohrR/1.5;
    SD_scaling = SD_scaling_user/(Max_Force+1.0e-10);

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;
  }

  else{

    if (Past_Utot[1]<Utot){ 
      SD_scaling = SD_scaling/2.0;
    }
    else if (Past_Utot[1]<Past_Utot[2] && Utot<Past_Utot[1] && iter%4==1){
      SD_scaling = SD_scaling*2.5;
    }

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;

    Past_Utot[5] = Past_Utot[4];
    Past_Utot[4] = Past_Utot[3];
    Past_Utot[3] = Past_Utot[2];
    Past_Utot[2] = Past_Utot[1];
    Past_Utot[1] = Utot;
  }
  
  if (myid==Host_ID && 0<level_stdout) printf("<Steepest_Descent>  SD_scaling=%15.12f\n",SD_scaling);

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    move atoms
  ****************************************************/

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* avoid too large movement */
    if ( Criterion_Max_Step<(Max_Force*SD_scaling) )
      scale = Criterion_Max_Step/(Max_Force*SD_scaling);
    else 
      scale = 1.0; 

    /* update coordinates */
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){
	if (atom_Fixed_XYZ[Gc_AN][j]==0){
	  Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] - scale*SD_scaling*Gxyz[Gc_AN][16+j];
	}
      }
    }
  }

  /****************************************************
   MPI, Gxyz
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<Steepest_Descent>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
             Max_Force);
      printf("<Steepest_Descent>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     MD_Opt_criterion);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("  atom=%3d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);
    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if ( Criterion_Max_Step<(Max_Force*SD_scaling) )
	MaxStep = Criterion_Max_Step;
      else 
	MaxStep = SD_scaling*Max_Force;

      if (iter==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");

        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }
      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter,SD_scaling,Max_Force,MaxStep*BohrR, Utot);
      fclose(fp_SD);
    }
    else
      printf("Error(7) in MD_pac.c\n");

    if (MD_Opt_OK==1 || iter==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /******************************************
              save *.rst4gopt.SD1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      fwrite(&Correct_Position_flag, sizeof(int), 1, fp);
      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}



void Cell_Opt_SD(int iter, int SD_scaling_flag)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int i,j,k;
  double itv[4][4],dE_da[4][4];
  double sum,detA,Max_Gradient;
  double My_Max_Force,Max_Force;

  int l,Mc_AN,Gc_AN;
  double dt,SD_max,SD_min,SD_init,Atom_W,tmp0,scale;
  double Wscale;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  int numprocs,myid,ID;
  double tmp1,MaxStep;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  char file_name[YOUSO10];
  FILE *fp;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /******************************************
              read *.rst4gopt.SD1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(&Correct_Position_flag, sizeof(int), 1, fp);
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /****************************************************
     calculate the inverse of a matrix consisiting 
     of the cell vectors   
  ****************************************************/

  detA = tv[1][1]*tv[2][2]*tv[3][3]+tv[2][1]*tv[3][2]*tv[1][3]+tv[3][1]*tv[1][2]*tv[2][3]
        -tv[1][1]*tv[3][2]*tv[2][3]-tv[3][1]*tv[2][2]*tv[1][3]-tv[2][1]*tv[1][2]*tv[3][3];

  itv[1][1] = (tv[2][2]*tv[3][3] - tv[2][3]*tv[3][2])/detA; 
  itv[1][2] = (tv[1][3]*tv[3][2] - tv[1][2]*tv[3][3])/detA;
  itv[1][3] = (tv[1][2]*tv[2][3] - tv[1][3]*tv[2][2])/detA;

  itv[2][1] = (tv[2][3]*tv[3][1] - tv[2][1]*tv[3][3])/detA; 
  itv[2][2] = (tv[1][1]*tv[3][3] - tv[1][3]*tv[3][1])/detA;
  itv[2][3] = (tv[1][3]*tv[2][1] - tv[1][1]*tv[2][3])/detA;

  itv[3][1] = (tv[2][1]*tv[3][2] - tv[2][2]*tv[3][1])/detA; 
  itv[3][2] = (tv[1][2]*tv[3][1] - tv[1][1]*tv[3][2])/detA;
  itv[3][3] = (tv[1][1]*tv[2][2] - tv[1][2]*tv[2][1])/detA;

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){

      sum = 0.0;
      for (k=1; k<=3; k++){
        sum += itv[k][i]*Stress_Tensor[3*(k-1)+(j-1)];
      }

      dE_da[i][j] = sum;
      if (Cell_Fixed_XYZ[i][j]==1) dE_da[i][j] = 0.0;
    }
  }

  /****************************************************
          find the maximum gradient dE/daij
  ****************************************************/

  Max_Gradient = 0.0;

  /* no constraint for cell vectors */
  /* while keeping the fractional coordinates */

  if (cellopt_swtich==1){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){

	if (Max_Gradient<fabs(dE_da[i][j]) && Cell_Fixed_XYZ[i][j]==0){
	  Max_Gradient = fabs(dE_da[i][j]); 
	}
      }
    }
  }

  /* angles fixed for cell vectors */
  /* while keeping the fractional coordinates */

  else if (cellopt_swtich==2){

    for (i=1; i<=3; i++){

      if (   Cell_Fixed_XYZ[i][1]==0
          && Cell_Fixed_XYZ[i][2]==0 
	  && Cell_Fixed_XYZ[i][3]==0 ){ 

	sum = 0.0;
	for (j=1; j<=3; j++){
	  sum += tv[i][j]*dE_da[i][j]; 
	}

        sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

        if (Max_Gradient<fabs(sum)) Max_Gradient = fabs(sum);

      }
    }
  }

  /* angles fixed and |a1|=|a2|=|a3| for cell vectors */
  /* while keeping the fractional coordinates */

  else if (cellopt_swtich==3){

    for (i=1; i<=3; i++){

      sum = 0.0;
      for (j=1; j<=3; j++){
	sum += tv[i][j]*dE_da[i][j]; 
      }

      sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

      if (Max_Gradient<fabs(sum)) Max_Gradient = fabs(sum);
    }
  }

  /* angles fixed and |a1|=|a2|!=|a3| for cell vectors */
  /* while keeping the fractional coordinates */

  else if (cellopt_swtich==4){

    for (i=1; i<=3; i++){

      sum = 0.0;
      for (j=1; j<=3; j++){
	sum += tv[i][j]*dE_da[i][j]; 
      }

      sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

      if (Max_Gradient<fabs(sum)) Max_Gradient = fabs(sum);
    }
  }

  /* no constraint for atomic coordinates */

  else if (cellopt_swtich==5 || cellopt_swtich==6 || cellopt_swtich==7){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){

	if (Max_Gradient<fabs(dE_da[i][j]) && Cell_Fixed_XYZ[i][j]==0){
	  Max_Gradient = fabs(dE_da[i][j]); 
	}
      }
    }

    My_Max_Force = 0.0;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];

      tmp0 = 0.0;
      for (j=1; j<=3; j++){
	if (atom_Fixed_XYZ[Gc_AN][j]==0){
	  tmp0 += Gxyz[Gc_AN][16+j]*Gxyz[Gc_AN][16+j];
	}
      }

      tmp0 = sqrt(tmp0); 
      if (My_Max_Force<tmp0) My_Max_Force = tmp0;
    }

    MPI_Allreduce(&My_Max_Force, &Max_Force, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_level1);

    if (Max_Gradient<Max_Force) Max_Gradient = Max_Force;
  }

  /****************************************************
   MD_Opt_OK
  ****************************************************/

  if (Max_Gradient<MD_Opt_criterion) MD_Opt_OK = 1;

  /****************************************************
   set up SD_scaling
  ****************************************************/

  dt = 41.3411*2.0;
  SD_init = dt*dt/Wscale;
  SD_max = SD_init*15.0;   /* default 15   */
  SD_min = SD_init*0.04;   /* default 0.02 */
  Atom_W = 12.0;

  if (iter==1 || SD_scaling_flag==0){

    SD_scaling_user = Max_Gradient/BohrR/1.5;
    SD_scaling = SD_scaling_user/(Max_Gradient+1.0e-10);

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;
  }

  else{
    if (Past_Utot[1]<(Utot+UpV) && iter%3==1){ 
      SD_scaling = SD_scaling/5.0;
    }
    else if (Past_Utot[1]<Past_Utot[2] && (Utot+UpV)<Past_Utot[1] && iter%4==1){
      SD_scaling = SD_scaling*2.5;
    }

    if (SD_max<SD_scaling) SD_scaling = SD_max;
    if (SD_scaling<SD_min) SD_scaling = SD_min;

    Past_Utot[5] = Past_Utot[4];
    Past_Utot[4] = Past_Utot[3];
    Past_Utot[3] = Past_Utot[2];
    Past_Utot[2] = Past_Utot[1];
    Past_Utot[1] = Utot + UpV;
  }
  
  if (myid==Host_ID && 0<level_stdout) printf("<Steepest_Descent>  SD_scaling=%15.12f\n",SD_scaling);

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    update lattice vectors and cartesian coordinates
  ****************************************************/

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* avoid too large movement */
    if ( Criterion_Max_Step<(Max_Gradient*SD_scaling) )
      scale = Criterion_Max_Step/(Max_Gradient*SD_scaling);
    else 
      scale = 1.0; 

    /***************************************************
                    update lattice vectors
    ***************************************************/

    /* no constraint for cell vectors */
    /* while keeping the fractional coordinates */

    if (cellopt_swtich==1){
    
      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
	  if (Cell_Fixed_XYZ[i][j]==0){
	    tv[i][j] = tv[i][j] - scale*SD_scaling*dE_da[i][j];
	  }
	}
      }
    }

    /* angles fixed for cell vectors */
    /* while keeping the fractional coordinates */

    else if (cellopt_swtich==2){

      for (i=1; i<=3; i++){

        if (   Cell_Fixed_XYZ[i][1]==0
            && Cell_Fixed_XYZ[i][2]==0 
            && Cell_Fixed_XYZ[i][3]==0 ){ 

	  sum = 0.0;
	  for (j=1; j<=3; j++){
            sum += tv[i][j]*dE_da[i][j]; 
	  }

          sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

          tv[i][1] = tv[i][1] - scale*SD_scaling*sum*tv[i][1];
          tv[i][2] = tv[i][2] - scale*SD_scaling*sum*tv[i][2];
          tv[i][3] = tv[i][3] - scale*SD_scaling*sum*tv[i][3];
	}
      }
    }

    /* angles fixed and |a1|=|a2|=|a3| for cell vectors */
    /* while keeping the fractional coordinates */

    else if (cellopt_swtich==3){

      double length_a[4],av_a;

      for (i=1; i<=3; i++){

        sum = 0.0;  
	for (j=1; j<=3; j++){
	  sum += tv[i][j]*dE_da[i][j]; 
	}
	sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

        tv[i][1] = tv[i][1] - scale*SD_scaling*sum*tv[i][1];
        tv[i][2] = tv[i][2] - scale*SD_scaling*sum*tv[i][2];
        tv[i][3] = tv[i][3] - scale*SD_scaling*sum*tv[i][3];
      }

      for (i=1; i<=3; i++){
        length_a[i] = 0.0;  
	for (j=1; j<=3; j++){
	  length_a[i] += tv[i][j]*tv[i][j];
	}
        length_a[i] = sqrt(length_a[i]);
      }

      av_a = (length_a[1]+length_a[2]+length_a[3])/3.0;      

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
          tv[i][j] = tv[i][j]/length_a[i]*av_a;
	}
      }
    }

    /* angles fixed and |a1|=|a2|!=|a3| for cell vectors */
    /* while keeping the fractional coordinates */

    else if (cellopt_swtich==4){

      double length_a[4],av_a;

      for (i=1; i<=3; i++){

        sum = 0.0;  
	for (j=1; j<=3; j++){
	  sum += tv[i][j]*dE_da[i][j]; 
	}
	sum = sum/sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);

        tv[i][1] = tv[i][1] - scale*SD_scaling*sum*tv[i][1];
        tv[i][2] = tv[i][2] - scale*SD_scaling*sum*tv[i][2];
        tv[i][3] = tv[i][3] - scale*SD_scaling*sum*tv[i][3];
      }

      for (i=1; i<=3; i++){
        length_a[i] = 0.0;  
	for (j=1; j<=3; j++){
	  length_a[i] += tv[i][j]*tv[i][j];
	}
        length_a[i] = sqrt(length_a[i]);
      }

      av_a = (length_a[1]+length_a[2])/2.0;      

      for (i=1; i<=2; i++){
	for (j=1; j<=3; j++){
          tv[i][j] = tv[i][j]/length_a[i]*av_a;
	}
      }
    }

    /* no constraint for atomic coordinates */

    else if (cellopt_swtich==5 || cellopt_swtich==6 || cellopt_swtich==7){

      /* calculate dE_dq */

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
       for (i=1; i<=3; i++){
          sum = 0.0;
          for (j=1; j<=3; j++){
            sum += Gxyz[Gc_AN][16+j]*tv[i][j];
	  }
          Gxyz[Gc_AN][13+i] = sum;
	}
      }

      /* update tv */

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
	  if (Cell_Fixed_XYZ[i][j]==0){
	    tv[i][j] = tv[i][j] - scale*SD_scaling*dE_da[i][j];
	  }
	}
      }
    }

    /***************************************************
                update cartesian coordinates
    ***************************************************/

    /* update coordinates while keeping the fractional coordinates */

    if (  cellopt_swtich==1 
       || cellopt_swtich==2
       || cellopt_swtich==3
       || cellopt_swtich==4
       ){

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	Gxyz[Gc_AN][1] = Cell_Gxyz[Gc_AN][1]*tv[1][1]
                       + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                       + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

	Gxyz[Gc_AN][2] = Cell_Gxyz[Gc_AN][1]*tv[1][2]
                       + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                       + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

	Gxyz[Gc_AN][3] = Cell_Gxyz[Gc_AN][1]*tv[1][3]
                       + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                       + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
      }
    }

    /* no constraint for atomic coordinates */

    else if (cellopt_swtich==5 || cellopt_swtich==6 || cellopt_swtich==7){

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	if (1){

	  Gxyz[Gc_AN][1] = Cell_Gxyz[Gc_AN][1]*tv[1][1]
                         + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                         + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

	  Gxyz[Gc_AN][2] = Cell_Gxyz[Gc_AN][1]*tv[1][2]
                         + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                         + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

	  Gxyz[Gc_AN][3] = Cell_Gxyz[Gc_AN][1]*tv[1][3]
                         + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                         + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];

	  for (j=1; j<=3; j++){
	    if (atom_Fixed_XYZ[Gc_AN][j]==0){
	      Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] - 0.5*scale*SD_scaling*Gxyz[Gc_AN][16+j];
	    }
	  }

	}

	else {

	  /* update Cell_Gxyz */

	  double len[4];

	  len[1] = sqrt( Dot_Product(tv[1], tv[1]) ); 
	  len[2] = sqrt( Dot_Product(tv[2], tv[2]) ); 
	  len[3] = sqrt( Dot_Product(tv[3], tv[3]) ); 

	  for (j=1; j<=3; j++){
	    if (atom_Fixed_XYZ[Gc_AN][j]==0){
	      Cell_Gxyz[Gc_AN][j] = Cell_Gxyz[Gc_AN][j] - 0.01*scale*SD_scaling*Gxyz[Gc_AN][13+j]/len[j];
	    }
	  }

	  /* update Gxyz */

	  Gxyz[Gc_AN][1] = Cell_Gxyz[Gc_AN][1]*tv[1][1]
                         + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                         + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

	  Gxyz[Gc_AN][2] = Cell_Gxyz[Gc_AN][1]*tv[1][2]
                         + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                         + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

	  Gxyz[Gc_AN][3] = Cell_Gxyz[Gc_AN][1]*tv[1][3]
                          + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                          + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
	}


      }
    }

  }

  /****************************************************
                show and save information
  ****************************************************/

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<Steepest_Descent>  |Maximum dE/da| (Hartree/Bohr) =%15.12f\n",
             Max_Gradient);
      printf("<Steepest_Descent>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     MD_Opt_criterion);

      printf("\n");
      printf(" Cell vectors and derivatives of total energy with respect to them\n");

      printf(" a1(Ang.) =%10.5f %10.5f %10.5f   dE/da1(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
      printf(" a2(Ang.) =%10.5f %10.5f %10.5f   dE/da2(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
      printf(" a3(Ang.) =%10.5f %10.5f %10.5f   dE/da3(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);


      if (1<level_stdout){ 

	printf("\n");
	for (i=1; i<=atomnum; i++){
	  printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
		 i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
		 Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
	}   
      }
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);
    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if ( Criterion_Max_Step<(Max_Gradient*SD_scaling) )
	MaxStep = Criterion_Max_Step;
      else 
	MaxStep = SD_scaling*Max_Gradient;

      if (iter==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"                History of cell optimization               \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");

        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot             Enpy           Volume\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)        (Hartree)        (Ang^3)\n\n");
      }
      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter,SD_scaling,Max_Gradient,MaxStep*BohrR, Utot, Utot+UpV,Cell_Volume*0.14818474347690476628);
      fclose(fp_SD);
    }
    else
      printf("Error(7) in MD_pac.c\n");

    if (MD_Opt_OK==1 || iter==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"    Cell vectors (Ang.) and derivatives of total energy    \n");
        fprintf(fp_crd,"             with respect to them (Hartree/Bohr)           \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd," a1 =%10.5f %10.5f %10.5f   dE/da1 =%10.5f %10.5f %10.5f\n",
                tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
        fprintf(fp_crd," a2 =%10.5f %10.5f %10.5f   dE/da2 =%10.5f %10.5f %10.5f\n",
               tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
        fprintf(fp_crd," a3 =%10.5f %10.5f %10.5f   dE/da3 =%10.5f %10.5f %10.5f\n",
               tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang.) and forces (Hartree/Bohr)    \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /******************************************
              save *.rst4gopt.SD1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.SD1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      fwrite(&Correct_Position_flag, sizeof(int), 1, fp);
      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}


void Cell_Opt_RF(int iter)
{
  int i,j,iatom,icell,k,diis_iter;
  double dE_da[4][4],detA,itv[4][4],sum;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.RF1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /****************************************************
     calculate the inverse of a matrix consisiting 
     of the cell vectors
     and
     calculate dE_da   
  ****************************************************/

  detA = tv[1][1]*tv[2][2]*tv[3][3]+tv[2][1]*tv[3][2]*tv[1][3]+tv[3][1]*tv[1][2]*tv[2][3]
        -tv[1][1]*tv[3][2]*tv[2][3]-tv[3][1]*tv[2][2]*tv[1][3]-tv[2][1]*tv[1][2]*tv[3][3];

  itv[1][1] = (tv[2][2]*tv[3][3] - tv[2][3]*tv[3][2])/detA; 
  itv[1][2] = (tv[1][3]*tv[3][2] - tv[1][2]*tv[3][3])/detA;
  itv[1][3] = (tv[1][2]*tv[2][3] - tv[1][3]*tv[2][2])/detA;

  itv[2][1] = (tv[2][3]*tv[3][1] - tv[2][1]*tv[3][3])/detA; 
  itv[2][2] = (tv[1][1]*tv[3][3] - tv[1][3]*tv[3][1])/detA;
  itv[2][3] = (tv[1][3]*tv[2][1] - tv[1][1]*tv[2][3])/detA;

  itv[3][1] = (tv[2][1]*tv[3][2] - tv[2][2]*tv[3][1])/detA; 
  itv[3][2] = (tv[1][2]*tv[3][1] - tv[1][1]*tv[3][2])/detA;
  itv[3][3] = (tv[1][1]*tv[2][2] - tv[1][2]*tv[2][1])/detA;

  /* calculate dE_d(a1,a2,a3) */

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){

      sum = 0.0;
      for (k=1; k<=3; k++){
        sum += itv[k][i]*Stress_Tensor[3*(k-1)+(j-1)];
      }

      dE_da[i][j] = sum;
      if (Cell_Fixed_XYZ[i][j]==1) dE_da[i][j] = 0.0;
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Cell_Opt_SD(iter,0);
    else 
      Cell_Opt_SD(iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {

      /* cartesian coordinates */

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }

      /* cell vectors */
 
      for (icell=1; icell<=3; icell++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][atomnum+icell][k] = GxyzHistoryIn[i][atomnum+icell][k];
	  GxyzHistoryR[i+1][atomnum+icell][k]  = GxyzHistoryR[i][atomnum+icell][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {

      /* calculate dE_dq */

      for (i=1; i<=3; i++){
	sum = 0.0;
	for (j=1; j<=3; j++){
	  sum += Gxyz[iatom][16+j]*tv[i][j];
	}
	Gxyz[iatom][13+i] = sum;
      }

      /* add Cell_Gxyz and dE_dq */

      for (k=1; k<=3; k++) {

        if (atom_Fixed_XYZ[iatom][k]==1){
	  Gxyz[iatom][13+k] = 0.0;
	}

        GxyzHistoryIn[0][iatom][k] = Cell_Gxyz[iatom][k];
        GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][13+k];
      }
    }

    /* add information on the cell vectors */

    for (icell=1; icell<=3; icell++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[0][atomnum+icell][k] = tv[icell][k];
	GxyzHistoryR[0][atomnum+icell][k]  = dE_da[icell][k];
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* RFC */

  else {
 
    if (cellopt_swtich==5 || cellopt_swtich==6 || cellopt_swtich==7) RFC5(local_iter,iter,dE_da,itv);

    else {

      if (myid==Host_ID){
        printf("The optimizer is not supported.\n");
      }

      MPI_Finalize();
      exit(1);
    }

    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.RF1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}




/*BEGIN hmweng*/
void Geometry_Opt_RF(int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.RF1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(iter,0);
    else 
      Steepest_Descent(iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {

    RF(local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.RF1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.RF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}
/*END hmweng */


void Geometry_Opt_DIIS(int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.DIIS1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.DIIS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(iter,0);
    else 
      Steepest_Descent(iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {

    GDIIS(local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.DIIS1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.DIIS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}




void Geometry_Opt_DIIS_BFGS(int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.BFGS1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.BFGS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(iter,0);
    else 
      Steepest_Descent(iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {
    GDIIS_BFGS(local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.BFGS1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.BFGS1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }
}


void Geometry_Opt_DIIS_EF(int iter)
{
  int i,iatom,k,diis_iter;
  double sMD_TimeStep;
  static int local_iter=1;
  static int SD_iter=0,GDIIS_iter=0;
  static int flag=0;
  static int Every_iter;
  char file_name[YOUSO10];
  FILE *fp;
  int tmp_array[10];
  int everyiter,buf_iter;
  int numprocs,myid,ID;  

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.EF1
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(int), 6, fp);

      local_iter            = tmp_array[0];
      SD_iter               = tmp_array[1];
      GDIIS_iter            = tmp_array[2];
      flag                  = tmp_array[3];
      Every_iter            = tmp_array[4];
      Correct_Position_flag = tmp_array[5];
 
      fread(&SD_scaling,      sizeof(double), 1, fp);
      fread(&SD_scaling_user, sizeof(double), 1, fp);
      fread(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* set parameters */

  Every_iter = OptEveryDIIS;

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* increament of iter */

  if (iter<OptStartDIIS){
    flag = 0;     
  }

  else if (iter==OptStartDIIS){
    flag = 1;     
    GDIIS_iter++;
  }

  else if (flag==0){
    SD_iter++; 
  }  
  else if (flag==1){
    GDIIS_iter++;
  }

  /* SD */

  if (flag==0){

    if (SD_iter==1)
      Steepest_Descent(iter,0);
    else 
      Steepest_Descent(iter,1);

    /* shift one */

    for (i=(diis_iter-2); 0<=i; i--) {
      for (iatom=1; iatom<=atomnum; iatom++) {
	for (k=1; k<=3; k++) {
	  GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	  GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
	}
      }
    }

    /* add GxyzHistoryIn and GxyzHisotryR */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
        if (atom_Fixed_XYZ[iatom][k]==0){
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
	}
        else{
	  GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	  GxyzHistoryR[0][iatom][k]  = 0.0;
        }
      }
    }

    /* initialize local_iter */

    local_iter = 1;

  }

  /* GDIIS */

  else {

    GDIIS_EF(local_iter,iter);
    local_iter++;
  }

  /* check the number of iterations */

  if (Every_iter<=SD_iter){
    flag = 1;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  else if (Every_iter<=GDIIS_iter){
    flag = 0;
    SD_iter = 0;
    GDIIS_iter = 0;
  }

  /******************************************
              save *.rst4gopt.EF1 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF1",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = local_iter;
      tmp_array[1] = SD_iter;
      tmp_array[2] = GDIIS_iter;
      tmp_array[3] = flag;
      tmp_array[4] = Every_iter;
      tmp_array[5] = Correct_Position_flag;
 
      fwrite(tmp_array, sizeof(int), 6, fp);

      fwrite(&SD_scaling,      sizeof(double), 1, fp);
      fwrite(&SD_scaling_user, sizeof(double), 1, fp);
      fwrite(&Past_Utot,       sizeof(double), 10, fp);

      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }
  
}



void GDIIS(int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  char *func_name="DIIS";
  char *JOBB="L";
  double *A,*B,sumB,max_A, RR,dRi[4],dRj[4];
  double *work;
  double mixing,force_Max;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx,diff,Max_Step;
  int *ipiv;
  INTEGER i,j,k,iatom,N,LDA,LWORK,info;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];

  /* variables for MPI */
  int Gc_AN;
  int numprocs,myid,ID;  

  /* MPI myid */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* share Gxyz */
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  if (myid!=Host_ID)   goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  A = (double*)malloc(sizeof(double)*(diis_iter+1)*(diis_iter+1));
  for (i=0; i<(diis_iter+1)*(diis_iter+1); i++) A[i] = 0.0;
  B = (double*)malloc(sizeof(double)*(diis_iter+1));

  /* Max of force */

  force_Max=0.0;
  for (iatom=1;iatom<=atomnum;iatom++)   {
    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }
    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {

      RR = 0.0;

      for (iatom=1; iatom<=atomnum; iatom++)   {
	for (k=1; k<=3; k++) {
	  dRi[k] = GxyzHistoryR[i][iatom][k];
	  dRj[k] = GxyzHistoryR[j][iatom][k];
	}

	RR += dRi[1]*dRj[1] + dRi[2]*dRj[2] + dRi[3]*dRj[3];
      }

      A[ i*(diis_iter+1)+j ]= RR;
    }
  }

  /* find max of A */

  max_A = 0.0;

  for (i=0;i<diis_iter;i++) {
    for (j=0;j<diis_iter;j++) {
      RR = fabs(A[i*(diis_iter+1)+j]) ;
      if (max_A< RR ) max_A = RR;
    }
  }

  max_A = 1.0/max_A;

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {
      A[ i*(diis_iter+1)+j ] *= max_A;
    }
  }

  for (i=0; i<diis_iter; i++) {
    A[ i*(diis_iter+1)+diis_iter ] = 1.0;
    A[ diis_iter*(diis_iter+1)+i ] = 1.0;
  }

  A[diis_iter*(diis_iter+1)+diis_iter] = 0.0;

  for (i=0; i<diis_iter; i++) B[i] = 0.0;
  B[diis_iter] = 1.0;

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  N=diis_iter+1;
  LDA=diis_iter+1;
  LWORK=diis_iter+1;
  work=(double*)malloc(sizeof(double)*LWORK);
  ipiv = (int*)malloc(sizeof(int)*(diis_iter+1));

  i = 1; 

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  F77_NAME(dsysv,DSYSV)( JOBB, &N, &i, A, &LDA,  ipiv, B, &LDA, work, &LWORK, &info);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion )  MD_Opt_OK = 1;

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j] ;
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    if      (1.0e-2<sumB)  mixing =-0.2;
    else if (1.0e-3<sumB)  mixing =-0.3;
    else if (1.0e-4<sumB)  mixing =-0.4;
    else if (1.0e-5<sumB)  mixing =-0.5;
    else if (1.0e-6<sumB)  mixing =-0.6;
    else                   mixing =-0.7;

    if (2<=level_stdout){
      printf("<%s> mixing=%15.12f\n",func_name,mixing);
    }

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1;j<=3;j++) {
	Gxyz[iatom][j] *= mixing;
      }
    }

    /* tilde{x} */

    for (iatom=1;iatom<=atomnum;iatom++) {
      for (j=1;j<=3;j++) {
	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}
      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(Utot-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  Past_Utot[1]=Utot;

  Max_Force = force_Max;
  SD_scaling_user = SD_scaling*Max_Force*0.2;

  /* free arrays */

  free(A);
  free(B);
  free(ipiv);
  free(work);

  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    /* ID = G2ID[Gc_AN]; */
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    /*    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, Host_ID, mpi_comm_level1); */
  }


  if (myid==Host_ID){ 

    if (0<level_stdout){
      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);
    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (iter0==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,Utot);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}




/*hmweng start RF method*/
void RF(int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int k1,k2,m1,m2;
  char *func_name="RF";
  char *JOBB="L";
  double dt,diff,Max_Step;
  double *A,*B,sumB,max_A, RR,dRi[4],dRj[4];
  double *work, *work2,**ahes;
/*hmweng c0, c1 are not used and lamda is newly defined */  
/*  double sum1,tmp1,tmp2,c0,c1; */
  double sum1,tmp1,tmp2,lamda;
  double mixing,force_Max;
  double itv[4][4];
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx;
  int *ipiv;
  INTEGER i,j,k,iatom,N,LDA,LWORK,info;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];

  /* variables for MPI */
  int Gc_AN;
  int numprocs,myid,ID;  

  /* MPI myid */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* share Gxyz */
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* set diis_iter */

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] =  Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  =  Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  /* set the initial approximate Hessian */

  if (iter%30==1){

    if (iter0<M_GDIIS_HISTORY)
      Estimate_Initial_Hessian(iter0-1,0,itv);
    else   
      Estimate_Initial_Hessian(M_GDIIS_HISTORY-1,0,itv);
  }

  if (myid!=Host_ID) goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  A = (double*)malloc(sizeof(double)*(diis_iter+2)*(diis_iter+2));
  for (i=0; i<(diis_iter+2)*(diis_iter+2); i++) A[i] = 0.0;
  B = (double*)malloc(sizeof(double)*(diis_iter+2));

  /* Max of force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++)   {
    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }
    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {

      RR = 0.0;

      for (iatom=1; iatom<=atomnum; iatom++)   {
	for (k=1; k<=3; k++) {
	  dRi[k] = GxyzHistoryR[i][iatom][k];
	  dRj[k] = GxyzHistoryR[j][iatom][k];
	}

	RR += dRi[1]*dRj[1] + dRi[2]*dRj[2] + dRi[3]*dRj[3];
      }

      A[ i*(diis_iter+1)+j ]= RR;
    }
  }

  /* find max of A */

  max_A = 0.0;

  for (i=0;i<diis_iter;i++) {
    for (j=0;j<diis_iter;j++) {
      RR = fabs(A[i*(diis_iter+1)+j]) ;
      if (max_A< RR ) max_A = RR;
    }
  }

  max_A = 1.0/max_A;

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {
      A[ i*(diis_iter+1)+j ] *= max_A;
    }
  }

  for (i=0; i<diis_iter; i++) {
    A[ i*(diis_iter+1)+diis_iter ] = 1.0;
    A[ diis_iter*(diis_iter+1)+i ] = 1.0;
  }

  A[diis_iter*(diis_iter+1)+diis_iter] = 0.0;

  for (i=0; i<diis_iter; i++) B[i] = 0.0;
  B[diis_iter] = 1.0;

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  N = diis_iter+1;
  LDA = diis_iter+1;
  LWORK = diis_iter+1;
  work = (double*)malloc(sizeof(double)*LWORK);
  ipiv = (int*)malloc(sizeof(int)*(diis_iter+1));

  i = 1; 

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  F77_NAME(dsysv,DSYSV)( JOBB, &N, &i, A, &LDA,  ipiv, B, &LDA, work, &LWORK, &info);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion ){
    MD_Opt_OK = 1;
    Max_Force = force_Max;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j];
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    /*********************************************************************************
     update an approximate Hessian matrix 
   
     H_k = H_{k-1} + y*y^t/(y^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
     y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
     s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    *********************************************************************************/

    if (iter!=1){
   
      /* H*s  */

      for (i=1; i<=3*atomnum; i++){     

	sum1 = 0.0;  
	for (k=1; k<=atomnum; k++){     

	  sum1 += ( Hessian[i][3*k-2]*(GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])
		   +Hessian[i][3*k-1]*(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])
		   +Hessian[i][3*k  ]*(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3]));
	}
 
	/* store H*s */

	Hessian[0][i] = sum1;
      }        

      /* tmp1 = y^t*s, tmp2 = s^t*H*s */ 
    
      tmp1 = 0.0;
      tmp2 = 0.0;
    
      for (k=1; k<=atomnum; k++){     

	tmp1 += ((GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*(GxyzHistoryR[0][k][1] - GxyzHistoryR[1][k][1])
	        +(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*(GxyzHistoryR[0][k][2] - GxyzHistoryR[1][k][2])
		+(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*(GxyzHistoryR[0][k][3] - GxyzHistoryR[1][k][3]));

	tmp2 += ((GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*Hessian[0][3*k-2]
		+(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*Hessian[0][3*k-1] 
		+(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*Hessian[0][3*k  ]); 
      }

      /* update the approximate Hessian by the BFGS method */

      if (tmp1<1.0e-10) tmp1 = 1.0e-10; 
      if (tmp2<1.0e-10) tmp2 = 1.0e-10; 

      m1 = 0;
      for (i=1; i<=atomnum; i++){   
	for (k1=1; k1<=3; k1++){     

	  m1++;

	  m2 = 0;
	  for (j=1; j<=atomnum; j++){     
	    for (k2=1; k2<=3; k2++){     

	      m2++;

	      Hessian[m1][m2] += 

		1.0/tmp1*(GxyzHistoryR[0][i][k1]-GxyzHistoryR[1][i][k1])*(GxyzHistoryR[0][j][k2]-GxyzHistoryR[1][j][k2])
		- 1.0/tmp2*Hessian[0][m1]*Hessian[0][m2];

	    }
	  }
	}
      }
    }

    /*
    if (myid==Host_ID){  
      printf("Hessian\n");
      for (i=1; i<=3*atomnum; i++){     
	for (j=1; j<=3*atomnum; j++){     
	  printf("%8.4f ",Hessian[i][j]);
	}
	printf("\n");
      }
    }
    */

    /************************************************************
            Construct augmented Hessian matrix  
           | H    g | ( s )         ( s ) 
           |        |       = lamda             
           | g    0 | ( 1 )         ( 1 )
    ************************************************************/
  
    Hessian[3*atomnum+1][3*atomnum+1]=0.0;
  
    m2 = 0;
    for (i=1; i<=atomnum; i++){   
      for (k=1; k<=3; k++){   
	m2++;
	Hessian[3*atomnum+1][m2]=Gxyz[i][k];
	Hessian[m2][3*atomnum+1]=Gxyz[i][k];  
      }	
    }
  
    /************************************************************
     find the lowest eigenvalue and corresponding eigenvector
               of the augmented Hessian matrix
    ************************************************************/  

    work2 = (double*)malloc(sizeof(double)*(3*atomnum+3));

    ahes = (double**)malloc(sizeof(double*)*(3*atomnum+3));
    for (i=0; i<3*atomnum+3; i++){
      ahes[i] = (double*)malloc(sizeof(double)*(3*atomnum+3));
    }

    for(i=0; i<3*atomnum+2; i++){
      for(j=0;j<3*atomnum+2; j++){
	ahes[i][j] = Hessian[i][j];
      }
    }
    Eigen_lapack(ahes, work2, 3*atomnum+1, 1); 

    /*
    for(i=1; i<=1; i++){
      printf("Eigenvalue=%15.8f\n",work2[i]);
    }
    printf("EigenVector is\n");
    */

    for(i=1;i<=3*atomnum+1; i++){
      Hessian[0][i]=ahes[i][1]/ahes[3*atomnum+1][1];
    }
 
    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1;j<=3;j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* calculate the DIIS coordinates tilde{x} */

    m1 = 0;
    for (iatom=1;iatom<=atomnum;iatom++) {
      for (j=1;j<=3;j++) {

	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}

	/* a quasi Newton method */ 

	m1++;
	Gxyz[iatom][j] += Hessian[0][m1]; 

      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                   show cooridinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(Utot-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

    Past_Utot[1]=Utot;

    Max_Force = force_Max;
    SD_scaling_user = SD_scaling*Max_Force*0.2;

    /* free arrays */

    free(A);
    free(B);
    free(ipiv);
    free(work);
    free(work2);

    for (i=0; i<(3*atomnum+3); i++){
      free(ahes[i]);
    }
    free(ahes);

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    /* ID = G2ID[Gc_AN]; */
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    /*    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, Host_ID, mpi_comm_level1); */
  }

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (iter0==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,Utot);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}



void RFC5(int iter, int iter0, double dE_da[4][4], double itv[4][4])
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int k1,k2,m1,m2;
  char *func_name="RF";
  char *JOBB="L";
  double dt,diff,Max_Step,sum;
  double *A,*B,sumB,max_A, RR,dRi[4],dRj[4];
  double *work, *work2,**ahes;
  double sum1,tmp1,tmp2,lamda;
  double mixing,force_Max,Max_Gradient;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx;
  int *ipiv;
  INTEGER i,j,k,iatom,icell,N,LDA,LWORK,info;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  int Gc_AN,nvecs;
  int numprocs,myid,ID;  

  /* MPI myid */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* share Gxyz */
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* set the number of vectors to be changed */

  if      (cellopt_swtich==5) nvecs = 3;
  else if (cellopt_swtich==6) nvecs = 2;
  else if (cellopt_swtich==7) nvecs = 1;

  /* set diis_iter */

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {

    /* fractional coordinates and gradients of the total energy w.r.t. the coordinates */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }

    /* cell vectors and gradients of the total energy w.r.t. the cell vectors */
 
    for (icell=1; icell<=nvecs; icell++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][atomnum+icell][k] = GxyzHistoryIn[i][atomnum+icell][k];
	GxyzHistoryR[i+1][atomnum+icell][k]  = GxyzHistoryR[i][atomnum+icell][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    /* calculate dE_dq */

    for (i=1; i<=3; i++){
      sum = 0.0;
      for (j=1; j<=3; j++){
	sum += Gxyz[iatom][16+j]*tv[i][j];
      }
      Gxyz[iatom][13+i] = sum;
    }

    for (k=1; k<=3; k++) {
      GxyzHistoryIn[0][iatom][k] = Cell_Gxyz[iatom][k];
      GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][13+k];
    }
  }

  /* add information on the cell vectors */

  for (icell=1; icell<=nvecs; icell++) {
    for (k=1; k<=3; k++) {
      GxyzHistoryIn[0][atomnum+icell][k] = tv[icell][k];
      GxyzHistoryR[0][atomnum+icell][k]  = dE_da[icell][k];
    }
  }

  /* set the initial approximate Hessian */

  if (iter==1){

    if (iter0<M_GDIIS_HISTORY)
      Estimate_Initial_Hessian(iter0-1,1,itv);
    else   
      Estimate_Initial_Hessian(M_GDIIS_HISTORY-1,1,itv);
  }
  
  if (myid!=Host_ID) goto Last_Bcast; 
  
  /*********************** for myid==Host_ID **************************/
  
  /* allocation of arrays */
  
  A = (double*)malloc(sizeof(double)*(diis_iter+2)*(diis_iter+2));
  for (i=0; i<(diis_iter+2)*(diis_iter+2); i++) A[i] = 0.0;
  B = (double*)malloc(sizeof(double)*(diis_iter+2));
  
  /* find the maximum gradient dE_da */
  
  Max_Gradient = 0.0;
  
  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      if (Max_Gradient<fabs(dE_da[i][j])) Max_Gradient = fabs(dE_da[i][j]); 
    }
  }

  /* find the maximum force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++){
    for (k=1; k<=3; k++) {
      if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
    }
  }
  
  if (Max_Gradient<force_Max) Max_Gradient = force_Max;
  
  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum gradient| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,Max_Gradient, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {

      RR = 0.0;

      /* dE/dr and dE/da */

      for (iatom=1; iatom<=(atomnum+nvecs); iatom++)   {
	for (k=1; k<=3; k++) {
	  dRi[k] = GxyzHistoryR[i][iatom][k];
	  dRj[k] = GxyzHistoryR[j][iatom][k];
	}

	RR += dRi[1]*dRj[1] + dRi[2]*dRj[2] + dRi[3]*dRj[3];
      }

      A[ i*(diis_iter+1)+j ]= RR;
    }
  }
  
  /* find the maximum value in elements of the matrix A */

  max_A = 0.0;

  for (i=0;i<diis_iter;i++) {
    for (j=0;j<diis_iter;j++) {
      RR = fabs(A[i*(diis_iter+1)+j]) ;
      if (max_A< RR ) max_A = RR;
    }
  }

  max_A = 1.0/max_A;

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {
      A[ i*(diis_iter+1)+j ] *= max_A;
    }
  }

  for (i=0; i<diis_iter; i++) {
    A[ i*(diis_iter+1)+diis_iter ] = 1.0;
    A[ diis_iter*(diis_iter+1)+i ] = 1.0;
  }

  A[diis_iter*(diis_iter+1)+diis_iter] = 0.0;

  for (i=0; i<diis_iter; i++) B[i] = 0.0;
  B[diis_iter] = 1.0;

  if (2<=level_stdout && myid==0){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  N = diis_iter+1;
  LDA = diis_iter+1;
  LWORK = diis_iter+1;
  work = (double*)malloc(sizeof(double)*LWORK);
  ipiv = (int*)malloc(sizeof(int)*(diis_iter+1));

  i = 1; 

  if (2<=level_stdout){
    printf("<%s> M_GDIIS_HISTORY=%2d diis_iter=%2d\n",func_name,M_GDIIS_HISTORY,diis_iter);
  }

  F77_NAME(dsysv,DSYSV)( JOBB, &N, &i, A, &LDA,  ipiv, B, &LDA, work, &LWORK, &info);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (Max_Gradient<MD_Opt_criterion ){
    MD_Opt_OK = 1;
    Max_Force = Max_Gradient;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize Gxyz */

    for (iatom=1; iatom<=(atomnum+nvecs); iatom++) {
      for (j=1; j<=3; j++) {

        /* store coordinates at the previous step */

        if (iatom<=atomnum) 
          Gxyz[iatom][20+j] = Gxyz[iatom][j]; 
        else 
          Gxyz[iatom][20+j] = tv[iatom-atomnum][j]; 

        /* initialize */
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} for Gxyz */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	}
      }
    }

    /* add tilde{R} for tv */

    for (icell=1; icell<=nvecs; icell++) {
      for (j=1; j<=3; j++) {
	for (i=0; i<diis_iter; i++) {
	  Gxyz[atomnum+icell][j] += GxyzHistoryR[i][atomnum+icell][j]*B[i];
	}
      }
    }

    /* calculate the norm of tilde{R} */

    sumB = 0.0;

    for (iatom=1; iatom<=(atomnum+nvecs); iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j];
      }
    }

    sumB = sqrt(sumB)/(double)(atomnum+nvecs);

    /*********************************************************************************
     update an approximate Hessian matrix 
   
     H_k = H_{k-1} + y*y^t/(y^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
     y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
     s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    *********************************************************************************/

    if (iter!=1){
   
      /* H*s  */

      for (i=1; i<=(3*atomnum+3*nvecs); i++){

	sum1 = 0.0;  
	for (k=1; k<=(atomnum+nvecs); k++){     

	  sum1 += ( Hessian[i][3*k-2]*(GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])
		   +Hessian[i][3*k-1]*(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])
		   +Hessian[i][3*k  ]*(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3]));
	}
 
	/* store H*s */

	Hessian[0][i] = sum1;
      }

      /* tmp1 = y^t*s, tmp2 = s^t*H*s */ 
    
      tmp1 = 0.0;
      tmp2 = 0.0;
    
      for (k=1; k<=(atomnum+nvecs); k++){     

	tmp1 += ((GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*(GxyzHistoryR[0][k][1] - GxyzHistoryR[1][k][1])
	        +(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*(GxyzHistoryR[0][k][2] - GxyzHistoryR[1][k][2])
		+(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*(GxyzHistoryR[0][k][3] - GxyzHistoryR[1][k][3]));

	tmp2 += ((GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*Hessian[0][3*k-2]
		+(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*Hessian[0][3*k-1] 
		+(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*Hessian[0][3*k  ]); 
      }

      /* update the approximate Hessian by the BFGS method */

      if (tmp1<1.0e-10) tmp1 = 1.0e-10; 
      if (tmp2<1.0e-10) tmp2 = 1.0e-10; 

      m1 = 0;
      for (i=1; i<=(atomnum+nvecs); i++){   
	for (k1=1; k1<=3; k1++){     

	  m1++;

	  m2 = 0;
	  for (j=1; j<=(atomnum+nvecs); j++){     
	    for (k2=1; k2<=3; k2++){     

	      m2++;

	      Hessian[m1][m2] += 

		1.0/tmp1*(GxyzHistoryR[0][i][k1]-GxyzHistoryR[1][i][k1])*(GxyzHistoryR[0][j][k2]-GxyzHistoryR[1][j][k2])
               -1.0/tmp2*Hessian[0][m1]*Hessian[0][m2];

	    }
	  }
	}
      }

    }

    /*
    if (myid==Host_ID){  
      printf("Hessian\n");
      for (i=1; i<=(3*atomnum+nvecs*3); i++){     
	for (j=1; j<=(3*atomnum+nvecs*3); j++){     
	  printf("%6.3f ",Hessian[i][j]);
	}
	printf("\n");
      }
    }
    */

    /************************************************************
            Construct augmented Hessian matrix  
           | H    g | ( s )         ( s ) 
           |        |       = lamda             
           | g    0 | ( 1 )         ( 1 )
    ************************************************************/
  
    Hessian[3*atomnum+nvecs*3+1][3*atomnum+nvecs*3+1] = 0.0;
    
    m2 = 0;
    for (i=1; i<=(atomnum+nvecs); i++){   
      for (k=1; k<=3; k++){   
	m2++;
	Hessian[3*atomnum+nvecs*3+1][m2] = Gxyz[i][k];
	Hessian[m2][3*atomnum+nvecs*3+1] = Gxyz[i][k];  
      }	
    }
  
    /************************************************************
     find the lowest eigenvalue and corresponding eigenvector
               of the augmented Hessian matrix
    ************************************************************/  

    work2 = (double*)malloc(sizeof(double)*(3*atomnum+12));

    ahes = (double**)malloc(sizeof(double*)*(3*atomnum+12));
    for (i=0; i<3*atomnum+12; i++){
      ahes[i] = (double*)malloc(sizeof(double)*(3*atomnum+12));
    }

    for(i=0; i<=(3*atomnum+nvecs*3+1); i++){
	for(j=0; j<=(3*atomnum+nvecs*3+1); j++){
	ahes[i][j] = Hessian[i][j];
      }
    }

    Eigen_lapack(ahes, work2, 3*atomnum+nvecs*3+1, 1); 

    for (i=1; i<=(3*atomnum+nvecs*3+1); i++){
      Hessian[0][i]=ahes[i][1]/ahes[3*atomnum+nvecs*3+1][1];
    }
 
    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /**************************************************
      update fractional coordinates and cell vectors 
    **************************************************/

    /* initialize */

    for (iatom=1; iatom<=(atomnum+nvecs); iatom++) {
      for (j=1; j<=3; j++) {
        Cell_Gxyz[iatom][j] = 0.0;
      }
    }

    /* calculate the DIIS coordinates tilde{cell_x} */

    m1 = 0;
    for (iatom=1; iatom<=(atomnum+nvecs); iatom++) {
      for (j=1; j<=3; j++) {

	/* DIIS mixing */
        
	for (i=0; i<diis_iter; i++) {
	  Cell_Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}
        
	/* a quasi Newton method */ 
        
	m1++;
        
	Cell_Gxyz[iatom][j] += Hessian[0][m1];
      }
    }

    /************************************
                    set tv 
    ************************************/

    for (i=1; i<=nvecs; i++){
      for (j=1; j<=3; j++){
        tv[i][j] = Cell_Gxyz[atomnum+i][j];
      }
    }

    /*********************************
              calculate Gxyz 
    *********************************/

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      Gxyz[Gc_AN][1] = Cell_Gxyz[Gc_AN][1]*tv[1][1]
                     + Cell_Gxyz[Gc_AN][2]*tv[2][1]
	             + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

      Gxyz[Gc_AN][2] = Cell_Gxyz[Gc_AN][1]*tv[1][2]
	             + Cell_Gxyz[Gc_AN][2]*tv[2][2]
	             + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

      Gxyz[Gc_AN][3] = Cell_Gxyz[Gc_AN][1]*tv[1][3]
	             + Cell_Gxyz[Gc_AN][2]*tv[2][3]
	             + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
    }

    /************************************************************
       In case of a too large updating, do a modest updating
       find Max_Step:
    ************************************************************/  

    Max_Step = 0.0;

    /* atomic coordinates */
     
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
    
        diff = fabs(Gxyz[iatom][j] - Gxyz[iatom][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }
     
    /* cell vectors */
    
    for (i=1; i<=nvecs; i++) {
      for (j=1; j<=3; j++) {
    
        diff = fabs(tv[i][j] - Gxyz[atomnum+i][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }
    
    /******************************************************
     if (Criterion_Max_Step<Max_Step), correct the update
    ******************************************************/

    if (Criterion_Max_Step<Max_Step){

      /* atomic coordinates */

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - Gxyz[iatom][20+j];
	  Gxyz[iatom][j] = Gxyz[iatom][20+j] + diff/Max_Step*Criterion_Max_Step;
	}
      }

      /* cell vectors */

      for (i=1; i<=nvecs; i++) {
	for (j=1; j<=3; j++) {

	  diff = tv[i][j] - Gxyz[atomnum+i][20+j];
	  tv[i][j] = Gxyz[atomnum+i][20+j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /*************************
          find Max_Step 
    *************************/

    Max_Step = 0.0;

    /* atomic coordinates */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - Gxyz[iatom][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /* cell vectors */

    for (i=1; i<=nvecs; i++) {
      for (j=1; j<=3; j++) {

        diff = fabs(tv[i][j] - Gxyz[atomnum+i][20+j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                   show cooridinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(Utot-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

    Past_Utot[1]=Utot;

    Max_Force = Max_Gradient;
    SD_scaling_user = SD_scaling*Max_Force*0.2;

    /* free arrays */

    free(A);
    free(B);
    free(ipiv);
    free(work);
    free(work2);

    for (i=0; i<(3*atomnum+3); i++){
      free(ahes[i]);
    }
    free(ahes);

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  for (Gc_AN=1; Gc_AN<=(atomnum+3); Gc_AN++){
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
  }

  /* copy tv[i] */ 

  for (i=1; i<=3; i++){
    MPI_Bcast(&tv[i][1],  3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
  }

  /* print information */ 

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      printf(" Cell vectors and derivatives of total energy with respect to them\n");

      printf(" a1(Ang.) =%10.5f %10.5f %10.5f   dE/da1(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
      printf(" a2(Ang.) =%10.5f %10.5f %10.5f   dE/da2(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
      printf(" a3(Ang.) =%10.5f %10.5f %10.5f   dE/da3(a.u.) =%10.5f %10.5f %10.5f\n",
             tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);

      if (1<level_stdout){ 

	printf("\n");
	for (i=1; i<=atomnum; i++){
	  printf("     atom=%4d, XYZ(Ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
		 i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
		 Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
	}   
      }
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);

    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (iter0==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"                History of cell optimization               \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot             Enpy           Volume\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)        (Hartree)        (Ang^3)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0, SD_scaling, Max_Force, Max_Step*BohrR, Utot, Utot+UpV,Cell_Volume*0.14818474347690476628);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"    Cell vectors (Ang.) and derivatives of total energy    \n");
        fprintf(fp_crd,"             with respect to them (Hartree/Bohr)           \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd," a1 =%10.5f %10.5f %10.5f   dE/da1 =%10.5f %10.5f %10.5f\n",
                tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR,dE_da[1][1],dE_da[1][2],dE_da[1][3]);
        fprintf(fp_crd," a2 =%10.5f %10.5f %10.5f   dE/da2 =%10.5f %10.5f %10.5f\n",
               tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR,dE_da[2][1],dE_da[2][2],dE_da[2][3]);
        fprintf(fp_crd," a3 =%10.5f %10.5f %10.5f   dE/da3 =%10.5f %10.5f %10.5f\n",
               tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR,dE_da[3][1],dE_da[3][2],dE_da[3][3]);

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}






void GDIIS_BFGS(int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int k1,k2,m1,m2;
  char *func_name="BFGS";
  char *JOBB="L";
  double dt;
  double *A,*B,sumB,max_A, RR,dRi[4],dRj[4];
  double *work;
  double sum1,tmp1,tmp2,c0,c1;
  double mixing,force_Max;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx,diff,Max_Step;
  int *ipiv;
  INTEGER i,j,k,iatom,N,LDA,LWORK,info;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];

  /* variables for MPI */
  int Gc_AN;
  int numprocs,myid,ID;  

  /* MPI myid */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* share Gxyz */
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* Max of force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++)   {
    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }
    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout && myid==Host_ID){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1;k<=3;k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  if (myid!=Host_ID)  goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  A = (double*)malloc(sizeof(double)*(diis_iter+1)*(diis_iter+1));
  for (i=0; i<(diis_iter+1)*(diis_iter+1); i++) A[i] = 0.0;
  B = (double*)malloc(sizeof(double)*(diis_iter+1));

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {

      RR = 0.0;

      for (iatom=1; iatom<=atomnum; iatom++)   {
	for (k=1; k<=3; k++) {
	  dRi[k] = GxyzHistoryR[i][iatom][k];
	  dRj[k] = GxyzHistoryR[j][iatom][k];
	}

	RR += dRi[1]*dRj[1] + dRi[2]*dRj[2] + dRi[3]*dRj[3];
      }

      A[ i*(diis_iter+1)+j ]= RR;
    }
  }

  /* find max of A */

  max_A = 0.0;

  for (i=0;i<diis_iter;i++) {
    for (j=0;j<diis_iter;j++) {
      RR = fabs(A[i*(diis_iter+1)+j]) ;
      if (max_A< RR ) max_A = RR;
    }
  }

  max_A = 1.0/max_A;

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {
      A[ i*(diis_iter+1)+j ] *= max_A;
    }
  }

  for (i=0; i<diis_iter; i++) {
    A[ i*(diis_iter+1)+diis_iter ] = 1.0;
    A[ diis_iter*(diis_iter+1)+i ] = 1.0;
  }

  A[diis_iter*(diis_iter+1)+diis_iter] = 0.0;

  for (i=0; i<diis_iter; i++) B[i] = 0.0;
  B[diis_iter] = 1.0;

  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%10.5f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  N=diis_iter+1;
  LDA=diis_iter+1;
  LWORK=diis_iter+1;
  work=(double*)malloc(sizeof(double)*LWORK);
  ipiv = (int*)malloc(sizeof(int)*(diis_iter+1));

  i = 1; 

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  F77_NAME(dsysv,DSYSV)( JOBB, &N, &i, A, &LDA,  ipiv, B, &LDA, work, &LWORK, &info);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%10.5f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion )  MD_Opt_OK = 1;

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j] ;
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    /************************************************************
     update an approximate inverse of Hessian matrix 

     y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
     s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    ************************************************************/

    if (iter==1){
      for (i=1; i<=3*atomnum; i++){     
	for (j=1; j<=3*atomnum; j++){     
	  InvHessian[i][j] = 0.0;
	}
	InvHessian[i][i] = 1.0;
      }
    }

    else {
   
      /* invH*y  */

      for (i=1; i<=3*atomnum; i++){     

	sum1 = 0.0;  
	for (k=1; k<=atomnum; k++){     

	  sum1 += ( InvHessian[i][3*k-2]*(GxyzHistoryR[0][k][1] - GxyzHistoryR[1][k][1])
		   +InvHessian[i][3*k-1]*(GxyzHistoryR[0][k][2] - GxyzHistoryR[1][k][2])
		   +InvHessian[i][3*k  ]*(GxyzHistoryR[0][k][3] - GxyzHistoryR[1][k][3]));
	}
 
	/* store invH*y */

	InvHessian[0][i] = sum1;
      }        

      /* tmp1 = s^t*y, tmp2 = y^t*H*y */ 

      tmp1 = 0.0;
      tmp2 = 0.0;

      for (k=1; k<=atomnum; k++){     

	tmp1 += ( (GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*(GxyzHistoryR[0][k][1] - GxyzHistoryR[1][k][1])
		 +(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*(GxyzHistoryR[0][k][2] - GxyzHistoryR[1][k][2])
		 +(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*(GxyzHistoryR[0][k][3] - GxyzHistoryR[1][k][3]));

	tmp2 += ( (GxyzHistoryR[0][k][1] - GxyzHistoryR[1][k][1])*InvHessian[0][3*k-2]
		 +(GxyzHistoryR[0][k][2] - GxyzHistoryR[1][k][2])*InvHessian[0][3*k-1] 
		 +(GxyzHistoryR[0][k][3] - GxyzHistoryR[1][k][3])*InvHessian[0][3*k  ]); 
      }

      /* c0=(tmp1+tmp2)/(tmp1*tmp1), c1=-1.0/tmp1 */

      c0 = (tmp1 + tmp2)/(tmp1*tmp1);
      c1 =-1.0/tmp1;

      /* update the approximate Hessian by the BFGS method */

      m1 = 0;
      for (i=1; i<=atomnum; i++){   
	for (k1=1; k1<=3; k1++){     

	  m1++;

	  m2 = 0;
	  for (j=1; j<=atomnum; j++){     
	    for (k2=1; k2<=3; k2++){     

	      m2++;

	      InvHessian[m1][m2] += 

		c0*(GxyzHistoryIn[0][i][k1]-GxyzHistoryIn[1][i][k1])*(GxyzHistoryIn[0][j][k2]-GxyzHistoryIn[1][j][k2])
		+ c1*InvHessian[0][m1]*(GxyzHistoryIn[0][j][k2]-GxyzHistoryIn[1][j][k2])
		+ c1*(GxyzHistoryIn[0][i][k1]-GxyzHistoryIn[1][i][k1])*InvHessian[0][m2];

	    }
	  }
	}
      }
    }

    /************************************************************
            perform a quasi Newton-Raphson method  
    ************************************************************/

    /* H^-1*g */

    m1 = 0;
    for (i=1; i<=atomnum; i++){   
      for (k1=1; k1<=3; k1++){     

	m1++;
  
	m2 = 0;
	sum1 = 0.0;
	for (j=1; j<=atomnum; j++){     
	  for (k2=1; k2<=3; k2++){     

	    m2++;

	    sum1 += InvHessian[m1][m2]*Gxyz[j][k2];
	  }
	}

        InvHessian[0][m1] = sum1;

      }
    }

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* calculate the DIIS coordinates tilde{x} */

    m1 = 0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}

	/* a quasi Newton method */ 

	m1++;
	Gxyz[iatom][j] -= InvHessian[0][m1]; 

      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                   show cooridinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(Utot-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(Ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  Past_Utot[1]=Utot;

  Max_Force = force_Max;
  SD_scaling_user = SD_scaling*Max_Force*0.2;

  /* free arrays */

  free(A);
  free(B);
  free(ipiv);
  free(work);

  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    /* ID = G2ID[Gc_AN]; */
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    /*    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, Host_ID, mpi_comm_level1); */
  }

  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);
    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (iter0==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,Utot);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

}









void GDIIS_EF(int iter, int iter0)
{
  /* 1au=2.4189*10^-2 fs, 1fs=41.341105 au */

  int k1,k2,m1,m2,isp,wan,wsp;
  static double Utot0,scaling_factor;
  char *func_name="EF";
  char *JOBB="L";
  double dt,MinKo;
  double itv[4][4];
  double *A,*B,sumB,max_A, RR,dRi[4],dRj[4];
  double *ko,**U;
  double *work;
/*hmweng c0, c1 are not used and lamda is newly defined */  
/*  double sum1,tmp1,tmp2,c0,c1; */
  double sum,sum1,tmp1,tmp2,lamda,c0,c1;
  double mixing,force_Max;
  static double sMD_TimeStep,dx_max=0.05; 
  double diff_dx,diff,Max_Step;
  int *ipiv;
  INTEGER i,j,k,iatom,N,LDA,LWORK,info;
  int diis_iter;
  char fileCoord[YOUSO10];
  char fileSD[YOUSO10];
  FILE *fp_crd,*fp_SD;
  char buf[fp_bsize];          /* setvbuf */
  char fileE[YOUSO10];
  double tmp_array[10];
  char file_name[YOUSO10];
  FILE *fp;

  /* variables for MPI */
  int Gc_AN;
  int numprocs,myid,ID;  

  /* MPI myid */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /******************************************
              read *.rst4gopt.EF2 
  ******************************************/

  if (SuccessReadingfiles){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF2",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      fread(tmp_array, sizeof(double), 2, fp);

      Utot0          = tmp_array[0];
      scaling_factor = tmp_array[1];
 
      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

  /* share Gxyz */
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /* set diis_iter */

  if (iter<M_GDIIS_HISTORY)
    diis_iter = iter;
  else   
    diis_iter = M_GDIIS_HISTORY;

  /* shift one */

  for (i=(diis_iter-2); 0<=i; i--) {
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (k=1; k<=3; k++) {
	GxyzHistoryIn[i+1][iatom][k] = GxyzHistoryIn[i][iatom][k];
	GxyzHistoryR[i+1][iatom][k]  = GxyzHistoryR[i][iatom][k];
      }
    }
  }

  /* add GxyzHistoryIn and GxyzHisotryR */

  for (iatom=1; iatom<=atomnum; iatom++)   {

    for (k=1; k<=3; k++) {
      if (atom_Fixed_XYZ[iatom][k]==0){
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = Gxyz[iatom][16+k];
      }
      else{
	GxyzHistoryIn[0][iatom][k] = Gxyz[iatom][k];
	GxyzHistoryR[0][iatom][k]  = 0.0;
      }
    }
  }

  /* set the initial approximate Hessian */

  if (iter==1){
    scaling_factor = 2.0;
    Estimate_Initial_Hessian(diis_iter,0,itv);
  }
  
  if (myid!=Host_ID)   goto Last_Bcast; 

  /*********************** for myid==Host_ID **************************/

  /* allocation of arrays */

  A = (double*)malloc(sizeof(double)*(diis_iter+1)*(diis_iter+1));
  for (i=0; i<(diis_iter+1)*(diis_iter+1); i++) A[i] = 0.0;
  B = (double*)malloc(sizeof(double)*(diis_iter+1));
  ko = (double*)malloc(sizeof(double)*(3*atomnum+2));
  U = (double**)malloc(sizeof(double*)*(3*atomnum+2));
  for (i=0; i<(3*atomnum+2); i++){
    U[i] = (double*)malloc(sizeof(double)*(3*atomnum+2));
  }

  /* Max of force */

  force_Max=0.0;
  for (iatom=1; iatom<=atomnum; iatom++)   {

    wsp = WhatSpecies[iatom];
    wan = Spe_WhatAtom[wsp];

    for (k=1;k<=3;k++) {

      if (atom_Fixed_XYZ[iatom][k]==0 && wan!=0){

	if (force_Max< fabs(Gxyz[iatom][16+k]) ) force_Max = fabs(Gxyz[iatom][16+k]);
      }

    }
  }

  sMD_TimeStep = 0.05/(0.01*41.341105);

  if (2<=level_stdout){
    printf("<%s>  |Maximum force| (Hartree/Bohr) = %15.12f tuned_dt= %f\n",func_name,force_Max, sMD_TimeStep);
    printf("<%s>  Criterion      (Hartree/Bohr) = %15.12f\n", func_name, MD_Opt_criterion);
  }

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {

      RR = 0.0;

      for (iatom=1; iatom<=atomnum; iatom++)   {
	for (k=1; k<=3; k++) {
	  dRi[k] = GxyzHistoryR[i][iatom][k];
	  dRj[k] = GxyzHistoryR[j][iatom][k];
	}

	RR += dRi[1]*dRj[1] + dRi[2]*dRj[2] + dRi[3]*dRj[3];
      }

      A[ i*(diis_iter+1)+j ]= RR;
    }
  }

  /* find max of A */

  max_A = 0.0;

  for (i=0;i<diis_iter;i++) {
    for (j=0;j<diis_iter;j++) {
      RR = fabs(A[i*(diis_iter+1)+j]) ;
      if (max_A< RR ) max_A = RR;
    }
  }

  max_A = 1.0/max_A;

  for (i=0; i<diis_iter; i++) {
    for (j=0; j<diis_iter; j++) {
      A[ i*(diis_iter+1)+j ] *= max_A;
    }
  }

  for (i=0; i<diis_iter; i++) {
    A[ i*(diis_iter+1)+diis_iter ] = 1.0;
    A[ diis_iter*(diis_iter+1)+i ] = 1.0;
  }

  A[diis_iter*(diis_iter+1)+diis_iter] = 0.0;

  for (i=0; i<diis_iter; i++) B[i] = 0.0;
  B[diis_iter] = 1.0;
  
  if (2<=level_stdout){
    printf("<%s>  DIIS matrix\n",func_name);
    for (i=0; i<(diis_iter+1); i++) {
      printf("<%s> ",func_name);
      for (j=0; j<(diis_iter+1); j++) {
        printf("%6.3f ",A[i*(diis_iter+1)+j]);
      }
      printf("\n");
    }
  }

  /* lapack routine */

  N=diis_iter+1;
  LDA=diis_iter+1;
  LWORK=diis_iter+1;
  work=(double*)malloc(sizeof(double)*LWORK);
  ipiv = (int*)malloc(sizeof(int)*(diis_iter+1));

  i = 1; 

  if (2<=level_stdout){
    printf("M_GDIIS_HISTORY=%2d diis_iter=%2d\n",M_GDIIS_HISTORY,diis_iter);
  }

  F77_NAME(dsysv,DSYSV)( JOBB, &N, &i, A, &LDA,  ipiv, B, &LDA, work, &LWORK, &info);

  if (info!=0) {
    printf("<%s> dsysv_, info=%d\n",func_name,info);
    printf("<%s> \n",func_name);
    printf("<%s> ERROR, aborting\n",func_name);
    printf("<%s> \n",func_name);

    MD_Opt_OK =1; 
    /* no change */

    goto Last_Bcast ;
  }

  if (2<=level_stdout){
    printf("<%s> diis alpha=",func_name);
    sumB = 0;
    for (i=0; i<diis_iter; i++) {
      printf("%f ",B[i]);
      sumB += B[i];
    }
    printf("%lf\n",B[diis_iter]);
  }

  if (force_Max<MD_Opt_criterion )  MD_Opt_OK = 1;
    
  /****************************************************
   write informatins to *.ene
  ****************************************************/
    
  if (myid==Host_ID){  
    
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  } 

  if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    /* add tilde{R} */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	if (atom_Fixed_XYZ[iatom][j]==0){
	  for (i=0; i<diis_iter; i++) {
	    Gxyz[iatom][j] += GxyzHistoryR[i][iatom][j]*B[i];
	  }
	}
      }
    }

    /* store tilde{R} into Hessian[][0] */

    m1 = 0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	m1++;
	Hessian[m1][0] = Gxyz[iatom][j];
      }
    }

    sumB = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {
	sumB += Gxyz[iatom][j]*Gxyz[iatom][j] ;
      }
    }

    sumB = sqrt(sumB)/(double)atomnum;

    if (2<=level_stdout){
      printf("<%s> |tilde{R}|=%E\n",func_name, sumB);
    }

    /***********************************************************************************
      update an approximate Hessian matrix 
   
      H_k = H_{k-1} + y*y^t/(y^t*y) - H_{k-1}*s * s^t * H_{k-1}^t /(s^t * H_{k-1} * s)
      y = GxyzHistoryR[0][iatom][k]  - GxyzHistoryR[1][iatom][k]
      s = GxyzHistoryIn[0][iatom][k] - GxyzHistoryIn[1][iatom][k]
    ***********************************************************************************/

    if (iter!=1){
   
      /* H*s  */

      for (i=1; i<=3*atomnum; i++){     

	sum1 = 0.0;  
	for (k=1; k<=atomnum; k++){     

	  sum1 += (Hessian[i][3*k-2]*(GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])
		  +Hessian[i][3*k-1]*(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])
		  +Hessian[i][3*k  ]*(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3]));
	}
 
	/* store H*s */

	Hessian[0][i] = sum1;
      }        

      /* tmp1 = y^t*s, tmp2 = s^t*H*s */ 
    
      tmp1 = 0.0;
      tmp2 = 0.0;
    
      for (k=1; k<=atomnum; k++){

	tmp1 += ((GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*(GxyzHistoryR[0][k][1] - GxyzHistoryR[1][k][1])
		 +(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*(GxyzHistoryR[0][k][2] - GxyzHistoryR[1][k][2])
		 +(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*(GxyzHistoryR[0][k][3] - GxyzHistoryR[1][k][3]));

	tmp2 += ((GxyzHistoryIn[0][k][1] - GxyzHistoryIn[1][k][1])*Hessian[0][3*k-2]
	        +(GxyzHistoryIn[0][k][2] - GxyzHistoryIn[1][k][2])*Hessian[0][3*k-1] 
		+(GxyzHistoryIn[0][k][3] - GxyzHistoryIn[1][k][3])*Hessian[0][3*k  ]); 
      }

      /* c0=1.0/tmp1, c1=1.0/tmp2 */

      /*
	if (myid==Host_ID){
	printf("tmp1=%15.12f tmp2=%15.12f\n",1.0/tmp1,1.0/tmp2);
	}
      */

      c0 = 1.0/tmp1;
      c1 = 1.0/tmp2;

      /* update the approximate Hessian by the BFGS method if 0.0<c0 */
    
      if (0.0<c0){
	m1 = 0;
	for (i=1; i<=atomnum; i++){   
	  for (k1=1; k1<=3; k1++){     

	    m1++;

	    m2 = 0;
	    for (j=1; j<=atomnum; j++){     
	      for (k2=1; k2<=3; k2++){     

		m2++;

		Hessian[m1][m2] += 

		  c0*(GxyzHistoryR[0][i][k1]-GxyzHistoryR[1][i][k1])*(GxyzHistoryR[0][j][k2]-GxyzHistoryR[1][j][k2])
		  -c1*Hessian[0][m1]*Hessian[0][m2];

	      }
	    }
	  }
	}

      }
    } 

    /************************************************************
             diagonalize the approximate Hessian
    ************************************************************/

    for (i=1; i<=3*atomnum; i++){
      for (j=1; j<=3*atomnum; j++){
	U[i][j] = Hessian[i][j];
      }
    }

    Eigen_lapack(U,ko,3*atomnum,3*atomnum);


    /*
      if (myid==Host_ID){
      for (i=1; i<=3*atomnum; i++){
      printf("i=%3d ko=%15.12f\n",i,ko[i]);
      }

      printf("EV\n"); 
      for (i=1; i<=3*atomnum; i++){
      for (j=1; j<=3*atomnum; j++){
      printf("%8.4f ",U[i][j]); 
      }
      printf("\n");
      }
      }
    */


    isp = 0;

    /*
      for (i=1; i<=3*atomnum; i++){
      if (ko[i]<1.0e-1) isp = i;
      }
    */

    if (atomnum<=4) MinKo = 0.10;
    else            MinKo = 0.005;

    for (i=1; i<=3*atomnum; i++){
      if (ko[i]<MinKo) ko[i] = MinKo;
    }

    if (isp!=0 && myid==Host_ID && 0<level_stdout){
      printf("Hessian is ill-conditioned.\n");
    } 

    /************************************************************
             U*lambda^{-1} U^{dag} * g
    ************************************************************/

    for (i=(isp+1); i<=3*atomnum; i++){
      sum = 0.0;
      for (j=1; j<=3*atomnum; j++){
	sum += U[j][i]*Hessian[j][0];
      }
      U[i][0] = sum;
    }  
  
    for (i=1; i<=3*atomnum; i++){
      sum = 0.0;
      for (j=(isp+1); j<=3*atomnum; j++){
	sum += U[i][j]*U[j][0]/ko[j];
      }
      U[0][i] = sum;
    }

    /************************************************************
            calculate the DIIS coordinates tilde{x}
              and perform a quasi Newton method
    ************************************************************/

    /* update scaling_factor */

    if (Utot0<Utot && iter!=1) scaling_factor = 0.95*scaling_factor;

    /* initialize */

    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1;j<=3;j++) {
	Gxyz[iatom][j] = 0.0;
      }
    }

    m1 = 0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

	/* DIIS coordinate */

	for (i=0; i<diis_iter; i++) {
	  Gxyz[iatom][j] += GxyzHistoryIn[i][iatom][j]*B[i]; 
	}


	/* with a quasi Newton method */ 

	m1++;

	Gxyz[iatom][j] -= scaling_factor*U[0][m1];
      }
    }

    /************************************************************
        In case of a too large updating, do a modest updating
    ************************************************************/  

    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    if (Criterion_Max_Step<Max_Step){

      for (iatom=1; iatom<=atomnum; iatom++) {
	for (j=1; j<=3; j++) {

	  diff = Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j];
	  Gxyz[iatom][j] = GxyzHistoryIn[0][iatom][j] + diff/Max_Step*Criterion_Max_Step;
	}
      }
    }

    /* find Max_Step */
    Max_Step = 0.0;
    for (iatom=1; iatom<=atomnum; iatom++) {
      for (j=1; j<=3; j++) {

        diff = fabs(Gxyz[iatom][j] - GxyzHistoryIn[0][iatom][j]);
        if (Max_Step<diff) Max_Step = diff;
      }
    }

    /************************************************************
                    show coordinates and gradients
    ************************************************************/  

    if (2<=level_stdout){

      printf("<%s> diff_x= %f , dE= %f\n",func_name, Max_Step, fabs(Utot-Past_Utot[1]) );

      /* print atomic positions */
      printf("<%s> atomnum= %d\n",func_name,atomnum);
      for (i=1; i<=atomnum; i++){
	j = Spe_WhatAtom[WhatSpecies[i]];
	printf("  %3d %s XYZ(ang) Fxyz(a.u.)= %9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,Atom_Symbol[j],
	       BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); 
      }   
    }

  } /* if (MD_Opt_OK!=1 && iter!=MD_IterNumber) */

  Past_Utot[1]=Utot;

  Max_Force = force_Max;
  SD_scaling_user = SD_scaling*Max_Force*0.2;

  /* save Utot */

  Utot0 = Utot;

  /* free arrays */

  free(A);
  free(B);
  free(ko);
  for (i=0; i<(3*atomnum+2); i++){
    free(U[i]);
  }
  free(U);

  free(ipiv);
  free(work);

  /*********************** end of "myid==Host_ID" **************************/

 Last_Bcast: 

  MPI_Bcast(&MD_Opt_OK,1,MPI_INT, Host_ID, mpi_comm_level1);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    /* ID = G2ID[Gc_AN]; */
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    /*    MPI_Bcast(&Gxyz[Gc_AN][17], 3, MPI_DOUBLE, Host_ID, mpi_comm_level1); */
  }


  if (myid==Host_ID){ 

    if (0<level_stdout){ 

      printf("<%s>  |Maximum force| (Hartree/Bohr) =%15.12f\n",
	     func_name,Max_Force);fflush(stdout);
      printf("<%s>  Criterion       (Hartree/Bohr) =%15.12f\n",
	     func_name,MD_Opt_criterion);fflush(stdout);

      printf("\n");
      for (i=1; i<=atomnum; i++){
	printf("     atom=%4d, XYZ(ang) Fxyz(a.u.)=%9.4f %9.4f %9.4f  %9.4f %9.4f %9.4f\n",
	       i,BohrR*Gxyz[i][1],BohrR*Gxyz[i][2],BohrR*Gxyz[i][3],
	       Gxyz[i][17],Gxyz[i][18],Gxyz[i][19] ); fflush(stdout);
      }   
    }

    strcpy(fileSD,".SD");
    fnjoint(filepath,filename,fileSD);
    if ((fp_SD = fopen(fileSD,"a")) != NULL){

      setvbuf(fp_SD,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (iter0==1){

        fprintf(fp_SD,"\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"              History of geometry optimization             \n");
        fprintf(fp_SD,"***********************************************************\n");
        fprintf(fp_SD,"***********************************************************\n\n");


        fprintf(fp_SD,"  MD_iter   SD_scaling     |Maximum force|   Maximum step        Utot\n");
        fprintf(fp_SD,"                           (Hartree/Bohr)        (Ang)         (Hartree)\n\n");
      }

      fprintf(fp_SD,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
              iter0,SD_scaling,Max_Force,Max_Step*BohrR,Utot);
      fclose(fp_SD);
    }
    else{
      printf("Could not open a file in MD_pac.!\n");
    }

    if (MD_Opt_OK==1 || iter0==MD_IterNumber){

      strcpy(fileCoord,".crd");
      fnjoint(filepath,filename,fileCoord);
      if ((fp_crd = fopen(fileCoord,"w")) != NULL){

        setvbuf(fp_crd,buf,_IOFBF,fp_bsize);  /* setvbuf */

        fprintf(fp_crd,"\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"       xyz-coordinates (Ang) and forces (Hartree/Bohr)  \n");
        fprintf(fp_crd,"***********************************************************\n");
        fprintf(fp_crd,"***********************************************************\n\n");

        fprintf(fp_crd,"<coordinates.forces\n");
        fprintf(fp_crd,"  %i\n",atomnum);
        for (k=1; k<=atomnum; k++){
          i = WhatSpecies[k];
          j = Spe_WhatAtom[i];
          fprintf(fp_crd," %4d  %4s   %9.5f %9.5f %9.5f  %15.12f %15.12f %15.12f\n",
                  k,Atom_Symbol[j],
	          Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	    	  -Gxyz[k][17],-Gxyz[k][18],-Gxyz[k][19]);
        }
        fprintf(fp_crd,"coordinates.forces>\n");
        fclose(fp_crd);
      }
      else
        printf("error(1) in MD_pac.c\n");
    }

  } /* if (myid==Host_ID) */

  /******************************************
              save *.rst4gopt.EF2 
  ******************************************/

  if (myid==Host_ID){

    sprintf(file_name,"%s%s_rst/%s.rst4gopt.EF2",filepath,filename,filename);

    if ((fp = fopen(file_name,"wb")) != NULL){

      tmp_array[0] = Utot0;
      tmp_array[1] = scaling_factor;
 
      fwrite(tmp_array, sizeof(double), 2, fp);
      fclose(fp);
    }    
    else{
      printf("Failure of saving %s\n",file_name);
    }
  }

}



















void NVT_VS(int iter)
{
  /* added by mari */
  /********************************************************
   This routine is added by Mari Ohfuti (May 20004).                

   a constant temperature molecular dynamics by a velocity
   scaling method with velocity-Verlet integrator
  ********************************************************/
  /******************************************************* 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step

    Gxyz[][27] = x-component of velocity at t+dt/2
    Gxyz[][28] = y-component of velocity at t+dt/2
    Gxyz[][29] = z-component of velocity at t+dt/2

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  double dt,dt2,sum,My_Ukc,x,t,xyz0[4],xyz0_l[4];
  double Wscale;
  int Mc_AN,Gc_AN,i,j,k,l;
  int numprocs,myid,ID;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /****************************************************
                    update velocity 
  ****************************************************/

  if (iter==1){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){

	  Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]+dt*Gxyz[Gc_AN][23+j]
                          -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];
 	  Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;
	}
      }
    }
  }
  else{
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){
 	  Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;
	}
      }
    }
  }

  /****************************************************
   correct so that the sum of velocities can be zero.                              
  ****************************************************/

  Correct_Velocity();  

  /****************************************************
                     Kinetic Energy 
  ****************************************************/

  Ukc=0.0;
  My_Ukc = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    sum = 0.0;
    for (j=1; j<=3; j++){
      sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
    }
    My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  /****************************************************
   MPI: Ukc 
  ****************************************************/
  
  MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  
  /* calculation of temperature (K) */
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;
  
  /* calculation of a given temperature (K) */
  for (i=1; i<=TempNum; i++) {
    if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {
      GivenTemp = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
          ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
    }
  }
  
  /****************************************************
   write informatins to *.ene
  ****************************************************/
  
  if (myid==Host_ID){  
  
    sprintf(fileE,"%s%s.ene",filepath,filename);

    /* corrected on 2016/1/19 by ADO */
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }
  
  /****************************************************
                    velocity scaling 
  ****************************************************/
  
  if(iter!=1) {

    x = 1.0;
    for (i=1; i<=TempNum; i++) {

      if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {

        /**************************************************
         find a scaling parameter, x, when MD step matches
         at the step where the temperature scaling is made.
         Otherwise, x = 1.0.
        **************************************************/

        if((iter-NumScale[i-1])%IntScale[i]==0) {

          GivenTemp = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
               ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
 
          x = GivenTemp + (Temp-GivenTemp)*RatScale[i];
          x = sqrt(1.5*kB*x/(Ukc*eV2Hartree)*(double)atomnum);
        }
      }
    }

    /* do scaling */

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	for (j=1; j<=3; j++){

	  if (atom_Fixed_XYZ[Gc_AN][j]==0){

	    Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]*x;
	    Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]+dt*Gxyz[Gc_AN][23+j]
	                    -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;
	    Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];
	  }
	}
      }

    }

  }

  /****************************************************
   MPI: Gxyz
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][14], 3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][24], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }
}




void NVT_Langevin(int iter)
{
  /* added by T.Ohwaki */
  /********************************************************
   This routine is added by Tsuruku Ohwaki (May 2012).

   a constant temperature molecular dynamics by Langevin
   heat-bath method with velocity-Verlet integrator
  ********************************************************/
  /*******************************************************
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u.
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at t
    Gxyz[][25] = y-component of velocity at t
    Gxyz[][26] = z-component of velocity at t

    Gxyz[][27] = x-component of tilde velocity at t+dt
    Gxyz[][28] = y-component of tilde velocity at t+dt
    Gxyz[][29] = z-component of tilde velocity at t+dt

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  double dt,dt2,sum,My_Ukc,x,t;
  double Wscale,rt,rt_mdt,rt_pdt,vt,vt_mdt,vt_pdt;
  double rtmp1,rtmp2,tmp,Lang_sig,RandomF;
  double vtt,ft_mdt,ftt,ft,vtt_pdt;
  double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
  int Mc_AN,Gc_AN,i,j,k,l;
  int numprocs,myid,ID;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /* calculation of a given temperature (K) */

  i = 1;
  do {
    if ( TempPara[i][1]<=iter && iter<TempPara[i+1][1] ){

      GivenTemp = TempPara[i][2] + (TempPara[i+1][2] - TempPara[i][2])*
                 ((double)iter - (double)TempPara[i][1])
                /((double)TempPara[i+1][1] - (double)TempPara[i][1]);

    }
    i++;
  } while (i<=(TempNum-1));

  /****************************************************
   add random forces on Gxyz[17-19]
  ****************************************************/

  /*
  srand((unsigned)time(NULL));
  */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Lang_sig = sqrt(2.0*Gxyz[Gc_AN][20]*Wscale*FricFac*kB*GivenTemp/dt/eV2Hartree);

    for (j=1; j<=3; j++){

      if (atom_Fixed_XYZ[Gc_AN][j]==0){

	rtmp1 = (double)rand()/RAND_MAX;
	rtmp2 = (double)rand()/RAND_MAX;

	RandomF = Lang_sig * sqrt(-2.0*log(rtmp1)) * cos(2.0*PI*rtmp2);
	Gxyz[Gc_AN][16+j] -= RandomF;

      }
    }
  }

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  if (iter==1){

    /****************************************************
      first step in velocity Verlet 
    ****************************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){

          /* update coordinates */
          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];

          vt = Gxyz[Gc_AN][23+j];
          vt_mdt = Gxyz[Gc_AN][26+j];

          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] + dt*vt
                        -0.5*dt2*(Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)+FricFac*vt);

          /* store current gradient */
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];

          /* update velocity */
 	  Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;
        }
      }
    }
  }

  else{

    /****************************************************
      for the second step and then onward 
      in velocity Verlet 
    ****************************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];

      tmp = 1.0/(1.0 + 0.50*dt*FricFac);
      tmp2 = 1.0 - 0.50*dt*FricFac;

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){

          /* modified Euler */

	  if (0){
          rt = Gxyz[Gc_AN][j];
          vtt = Gxyz[Gc_AN][26+j];
          vt_mdt = Gxyz[Gc_AN][23+j];
          ft_mdt = Gxyz[Gc_AN][13+j];

          ftt = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale) - FricFac*vtt;
          vt = vt_mdt + 0.5*dt*(ft_mdt + ftt);

          ft = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale) - FricFac*vt;

          vtt_pdt = vt + dt*ft;
          rt_pdt = rt + 0.5*dt*(vt + vtt_pdt); 

          Gxyz[Gc_AN][j] = rt_pdt;
          Gxyz[Gc_AN][26+j] = vtt_pdt;
          Gxyz[Gc_AN][23+j] = vt;
          Gxyz[Gc_AN][13+j] = ft;
	  }

          /* Taylor expansion */

          if (0){

          vt = Gxyz[Gc_AN][23+j];
          vt_mdt = Gxyz[Gc_AN][26+j];

          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] + dt*vt
                        -0.5*dt2*(Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)+FricFac*vt);

          Gxyz[Gc_AN][23+j] = vt_mdt - 2.0*dt*Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale) - 2.0*dt*FricFac*vt;
          Gxyz[Gc_AN][26+j] = vt;

	  }

          /* simple Taylor expansion */
	  if (0){
          vt = Gxyz[Gc_AN][23+j];

          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] + dt*vt
                        -0.5*dt2*(Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)+FricFac*vt);

          Gxyz[Gc_AN][23+j] = vt - dt*Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale) - dt*FricFac*vt;

	  }


          /* Brunger-Brooks-Karplus integrator */ 
	  if (0){

          /* update coordinates */

          rt_mdt = Gxyz[Gc_AN][20+j];
          rt = Gxyz[Gc_AN][j];

          rt_pdt = 2.0*rt - tmp2*rt_mdt - dt2/(Gxyz[Gc_AN][20]*Wscale)*Gxyz[Gc_AN][16+j]; 
          rt_pdt *= tmp; 

          /* update velocity */

          vt = 0.5*(rt_pdt - rt_mdt)/dt;

          /* store data */

          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];  /* store coordinate at t    */
          Gxyz[Gc_AN][j] = rt_pdt;             /* store coordinate at t+dt */
          Gxyz[Gc_AN][23+j] = vt;              /* store velocity at t      */
	  }


          /* Velocity Verlet formulation of the BBK integrator */
	  if (0){

          /* velocity at the current step */
          Gxyz[Gc_AN][23+j] = tmp2*Gxyz[Gc_AN][23+j] 
                            -(Gxyz[Gc_AN][16+j] + Gxyz[Gc_AN][13+j])
                            /(Gxyz[Gc_AN][20]*Wscale)*dt*0.50;

          Gxyz[Gc_AN][23+j] *= tmp;

          /* update coordinates */
          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] + dt*Gxyz[Gc_AN][23+j]*tmp2
                          -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;

          /* store current gradient */
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];
	  }


          /* Velocity Verlet formulation of the BBK integrator with velocity correction by Dr. Ohwaki */
	  if (0){

          /* velocity at the current step */

          tmp1 = Gxyz[Gc_AN][23+j];

          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j] 
                            -(Gxyz[Gc_AN][16+j] + Gxyz[Gc_AN][13+j])
                            /(Gxyz[Gc_AN][20]*Wscale)*dt*0.50;

          Gxyz[Gc_AN][23+j] *= tmp;

          /* add contribution of friction force */
          Gxyz[Gc_AN][16+j] += Gxyz[Gc_AN][20]*Wscale*FricFac*Gxyz[Gc_AN][23+j];


          /* update coordinates */
          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j] + dt*Gxyz[Gc_AN][23+j]
                          -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;

          /* store current gradient */
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j]
                             -Gxyz[Gc_AN][20]*Wscale*FricFac*Gxyz[Gc_AN][23+j]
                             +Gxyz[Gc_AN][20]*Wscale*FricFac*tmp1;
	  }

          /* Velocity Verlet formulation by Ozaki's integrator */

	  if (1){

          /* velocity at the current step */
          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j] 
                            -(Gxyz[Gc_AN][16+j] + Gxyz[Gc_AN][13+j])/(Gxyz[Gc_AN][20]*Wscale)*dt*0.50 
                            -FricFac*(Gxyz[Gc_AN][j] - Gxyz[Gc_AN][20+j]);

          /* update coordinates */
          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]
                        + tmp*(dt*Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50);

          /* store current gradient */
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];
	  }


        }
      }
    }
  }

  /****************************************************
   MPI: Gxyz
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][0], 40, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /****************************************************
                       Kinetic Energy 
  ****************************************************/

  My_Ukc = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    sum = 0.0;
    for (j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
	sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
      }
    }
    My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  /****************************************************
     MPI, Ukc
  ****************************************************/

  MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* calculation of temperature (K) */
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

  /****************************************************
     write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }
}




void NVT_VS2(int iter)
{
  /* added by T.Ohwaki */
  /********************************************************
   This routine is added by T.Ohwaki (May 20004).                

   a constant temperature molecular dynamics by a velocity
   scaling method for each atom with velocity-Verlet integrator
  ********************************************************/
  /******************************************************* 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step
  ****************************************************/

  double dt,dt2,sum,My_Ukc,x,t,xyz0[4],xyz0_l[4];
  double Wscale;

  double *Atomic_Temp,*Atomic_Ukc,*Atomic_Scale;

  int Mc_AN,Gc_AN,i,j,k,l;
  int numprocs,myid,ID;

  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  Atomic_Temp  = (double*)malloc(sizeof(double)*(atomnum+1));
  Atomic_Ukc   = (double*)malloc(sizeof(double)*(atomnum+1));
  Atomic_Scale = (double*)malloc(sizeof(double)*(atomnum+1));

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  if (iter==1){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){

	  Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]+dt*Gxyz[Gc_AN][23+j]
                          -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];
 	  Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;
	}
      }
    }
  }
  else{
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){
 	  Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;
	}
      }
    }
  }

  /****************************************************
      Kinetic Energy & Temperature (for each atom)
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    sum = 0.0;
    for (j=1; j<=3; j++){
      sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
    }

    Atomic_Ukc[Gc_AN]  = 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    Atomic_Temp[Gc_AN] = Atomic_Ukc[Gc_AN]/(1.5*kB)*eV2Hartree;
  }

  /****************************************************
   correct so that the sum of velocities can be zero.                              
  ****************************************************/

  Correct_Velocity();  

  /****************************************************
            Kinetic Energy (for whole system) 
  ****************************************************/

  Ukc=0.0;
  My_Ukc = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    sum = 0.0;
    for (j=1; j<=3; j++){
      sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
    }
    My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  /****************************************************
   MPI: Ukc 
  ****************************************************/

  MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* calculation of temperature (K) */
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  

    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
   velocity scaling for each atom
  ****************************************************/

  if(iter!=1) {

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Atomic_Scale[Gc_AN] = 1.0;
    }

    for (i=1; i<=TempNum; i++) {

      if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {

        /*************************************************************
         find a scaling parameter, Atomic_Scale, when MD step matches
         at the step where the temperature scaling is made.
         Otherwise, Atomic_Scale = 1.0.
        *************************************************************/

        if((iter-NumScale[i-1])%IntScale[i]==0) {

          GivenTemp = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
               ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
 
          for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
            Gc_AN = M2G[Mc_AN];

            Atomic_Scale[Gc_AN] = sqrt((GivenTemp + (Atomic_Temp[Gc_AN]-GivenTemp)*RatScale[i])/Atomic_Temp[Gc_AN]);

          }

        }
      }
    }

    /* do scaling */

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	for (j=1; j<=3; j++){

	  if (atom_Fixed_XYZ[Gc_AN][j]==0){

	    Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]*Atomic_Scale[Gc_AN];
	    Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]+dt*Gxyz[Gc_AN][23+j]
	                    -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;
	    Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];

	  }
	}
      }

    }

  }

  /****************************************************
   MPI: Gxyz
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][14], 3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][24], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  free(Atomic_Temp);
  free(Atomic_Ukc);
  free(Atomic_Scale);

}


void NVT_VS4(int iter)
{
  /* added by T. Ohwaki */
  /********************************************************
   This routine is added by T. Ohwaki (2011/11/11).

   a constant temperature molecular dynamics
   by a velocity scaling method for each atom group
   with velocity-Verlet integrator
  ********************************************************/
  /*******************************************************
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u.
  ********************************************************/

  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step
  ****************************************************/

  double dt,dt2,sum,My_Ukc,x,t;
  double Wscale;

  double *Atomic_Temp,*Atomic_Ukc,*Atomic_Scale;

  int Mc_AN,Gc_AN,i,j,k,l;
  int numprocs,myid,ID;

  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  double sum_AtGr[num_AtGr+1],My_Ukc_AtGr[num_AtGr+1],Ukc_AtGr[num_AtGr+1];
  double AtomGr_Scale[num_AtGr+1];

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  if (iter==1){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){

          Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]+dt*Gxyz[Gc_AN][23+j]
                          -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;
          Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];
 	  Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;
        }
      }
    }
  }
  else{
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]-Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt;

        }
      }
    }
  }

  /****************************************************
   correct so that the sum of velocities can be zero.                              
  ****************************************************/

  Correct_Velocity();  

  /******************************************************
      Kinetic Energy & Temperature (for each atom group)
  ******************************************************/

  for (k=1; k<=num_AtGr; k++){ 
    sum_AtGr[k] = 0.0; 
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    k = AtomGr[Gc_AN];

    for (j=1; j<=3; j++){
      sum_AtGr[k] += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][20];
    }
  }

  for (k=1; k<=num_AtGr; k++){
    My_Ukc_AtGr[k]  = 0.5*Wscale*sum_AtGr[k];
  }

  /****************************************************
            Kinetic Energy (for whole system)
  ****************************************************/

  Ukc=0.0;
  My_Ukc = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];

    sum = 0.0;
    for (j=1; j<=3; j++){
      sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
    }
    My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  /****************************************************
   MPI: Ukc
  ****************************************************/

  MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* calculation of temperature (K) */
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

  /****************************************************
   MPI: Ukc_AtGr
  ****************************************************/

  for (k=1; k<=num_AtGr; k++){
    MPI_Allreduce(&My_Ukc_AtGr[k], &Ukc_AtGr[k], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /* calculation of temperature (K) */
    Temp_AtGr[k] = Ukc_AtGr[k]/(1.5*kB*(double)atnum_AtGr[k])*eV2Hartree;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
   velocity scaling for each atom group
  ****************************************************/

  if(iter!=1) {

  for (k=1; k<=num_AtGr; k++){
      AtomGr_Scale[k] = 1.0;
  }

    for (i=1; i<=TempNum; i++) {

      if( (iter>NumScale[i-1]) && (iter<=NumScale[i]) ) {

        /*************************************************************
         find a scaling parameter, Atomic_Scale, when MD step matches
         at the step where the temperature scaling is made.
         Otherwise, Atomic_Scale = 1.0.
        *************************************************************/

        if((iter-NumScale[i-1])%IntScale[i]==0) {

          GivenTemp = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
                     ((double)iter - (double)TempPara[i-1][1])
                    /((double)TempPara[i][1] - (double)TempPara[i-1][1]);

          for (k=1; k<=num_AtGr; k++){

            AtomGr_Scale[k] = sqrt((GivenTemp + (Temp_AtGr[k]-GivenTemp)*RatScale[i])/Temp_AtGr[k]);

          }

        }
      }
    }

    /* do scaling */

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        k = AtomGr[Gc_AN];
        for (j=1; j<=3; j++){

          if (atom_Fixed_XYZ[Gc_AN][j]==0){

            Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][23+j]*AtomGr_Scale[k];
            Gxyz[Gc_AN][j] = Gxyz[Gc_AN][j]+dt*Gxyz[Gc_AN][23+j]
                            -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale)*dt2*0.50;
            Gxyz[Gc_AN][13+j] = Gxyz[Gc_AN][16+j];

          }
        }
      }

    }

  }

  /****************************************************
   MPI: Gxyz
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],  3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][14], 3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][24], 3, MPI_DOUBLE, ID, mpi_comm_level1);
  }

}




void NVT_NH(int iter)
{
  /***********************************************************
   a constant temperature molecular dynamics by a Nose-Hoover
   method with velocity-Verlet integrator
  ***********************************************************/
  /*********************************************************** 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ***********************************************************/
  /****************************************************
    Gxyz[][1] = x-coordinate at current step
    Gxyz[][2] = y-coordinate at current step
    Gxyz[][3] = z-coordinate at current step

    Gxyz[][14] = dEtot/dx at previous step
    Gxyz[][15] = dEtot/dy at previous step
    Gxyz[][16] = dEtot/dz at previous step

    Gxyz[][17] = dEtot/dx at current step
    Gxyz[][18] = dEtot/dy at current step
    Gxyz[][19] = dEtot/dz at current step

    Gxyz[][20] = atomic mass

    Gxyz[][21] = x-coordinate at previous step
    Gxyz[][22] = y-coordinate at previous step
    Gxyz[][23] = z-coordinate at previous step

    Gxyz[][24] = x-component of velocity at current step
    Gxyz[][25] = y-component of velocity at current step
    Gxyz[][26] = z-component of velocity at current step

    Gxyz[][27] = x-component of velocity at t+dt/2
    Gxyz[][28] = y-component of velocity at t+dt/2
    Gxyz[][29] = z-component of velocity at t+dt/2

    Gxyz[][30] = hx
    Gxyz[][31] = hy
    Gxyz[][32] = hz

  ****************************************************/

  int Mc_AN,Gc_AN,i,j,k,l,po,num,NH_switch;
  int numprocs,myid,ID;

  double dt,dt2,sum,My_sum,My_Ukc,x,t,xyz0[4],xyz0_l[4];
  double scaled_force,Wscale,back;
  double dzeta,dv,h_zeta;
  double My_sum1,sum1,My_sum2,sum2;
  char fileE[YOUSO10];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  MD_Opt_OK = 0;
  dt = 41.3411*MD_TimeStep;
  dt2 = dt*dt;
  Wscale = unified_atomic_mass_unit/electron_mass;

  /* find a given temperature by a linear interpolation */

  NH_switch = 0;
  i = 1;
  do {

    if ( TempPara[i][1]<=iter && iter<TempPara[i+1][1] ){

      GivenTemp = TempPara[i][2] + (TempPara[i+1][2] - TempPara[i][2])*
            ((double)iter-(double)TempPara[i][1])/((double)TempPara[i+1][1]-(double)TempPara[i][1]);

      NH_switch = 1; 
    }

    i++;
  } while (NH_switch==0 && i<=(TempNum-1));  

  /****************************************************
                Velocity Verlet algorithm
  ****************************************************/

  if (iter==1){

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    My_Ukc = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      sum = 0.0;
      for (j=1; j<=3; j++){
        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	}
      }
      My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    }

    /****************************************************
     MPI, Ukc
    ****************************************************/

    MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /* calculation of temperature (K) */
    Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ****************************************************/

    NH_Ham = Utot + Ukc + 0.5*NH_czeta*NH_czeta*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*GivenTemp*NH_R/eV2Hartree; 

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(iter,MD_TimeStep*(iter-1),fileE);
    }

    /****************************************************
      first step in velocity Verlet 
    ****************************************************/

    NH_czeta = 0.0;
    NH_R = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){
        
          scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  

          /* v( r+0.5*dt ) */
          Gxyz[Gc_AN][26+j] = Gxyz[Gc_AN][23+j] + (scaled_force - NH_czeta*Gxyz[Gc_AN][23+j])*0.5*dt;
          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j];

          /* r( r+dt ) */
          Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
 	  Gxyz[Gc_AN][j] =  Gxyz[Gc_AN][j] + Gxyz[Gc_AN][26+j]*dt;
	}

      }
    }

    /* zeta( t+0.5*dt ) */

    NH_nzeta = NH_czeta + (Ukc - 1.5*kB*(double)atomnum*GivenTemp/eV2Hartree)*dt/(TempQ*Wscale);
    NH_czeta = NH_nzeta;

    /* R( r+dt ) */
    NH_R = NH_R + NH_nzeta*dt;
  }

  else if (NH_switch==1) {

    /*****************************************************
     second step:

     refinement of v and zeta by a Newton-Raphson method
    *****************************************************/

    po = 0;
    num = 0;

    do {

      /* Ukc */

      Ukc = 0.0;
      My_Ukc = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];

        sum = 0.0;
        for (j=1; j<=3; j++){
          if (atom_Fixed_XYZ[Gc_AN][j]==0){
            sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	  }
        }
        My_Ukc += 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
      }

      /* MPI: Ukc */

      MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      /* calculation of h */ 

      h_zeta = NH_nzeta
              + (Ukc - 1.5*kB*(double)atomnum*GivenTemp/eV2Hartree)*dt/(TempQ*Wscale) - NH_czeta;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        for (j=1; j<=3; j++){

          if (atom_Fixed_XYZ[Gc_AN][j]==0){

            scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  
            Gxyz[Gc_AN][j+29] = Gxyz[Gc_AN][26+j] + (scaled_force - NH_czeta*Gxyz[Gc_AN][23+j])*0.5*dt
                                -Gxyz[Gc_AN][23+j];
	  }
	}  
      }

      /* sum1 */
     
      sum1=0.0;
      My_sum1 = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];

        sum = 0.0;
        for (j=1; j<=3; j++){
          if (atom_Fixed_XYZ[Gc_AN][j]==0){
            sum += Gxyz[Gc_AN][j+29]*Gxyz[Gc_AN][j+23];
	  }
        }
        My_sum1 += Gxyz[Gc_AN][20]*Wscale*sum*dt/(TempQ*Wscale);
      }

      /* MPI: sum1 */

      MPI_Allreduce(&My_sum1, &sum1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      /* sum2 */
     
      sum2=0.0;
      My_sum2 = 0.0;

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];

        sum = 0.0;
        for (j=1; j<=3; j++){
          if (atom_Fixed_XYZ[Gc_AN][j]==0){
            sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	  }
        }
        My_sum2 -= 0.5*Gxyz[Gc_AN][20]*Wscale*sum*dt*dt/(TempQ*Wscale);
      }

      /* MPI: sum2 */

      MPI_Allreduce(&My_sum2, &sum2, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      /* new NH_czeta and new v */

      dzeta = (-h_zeta*(NH_czeta*0.5*dt+1.0)-sum1)/(-(NH_czeta*0.5*dt+1.0)+sum2);

      My_sum = 0.0;
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
        for (j=1; j<=3; j++){

          if (atom_Fixed_XYZ[Gc_AN][j]==0){
            dv = (Gxyz[Gc_AN][j+29] - 0.5*Gxyz[Gc_AN][j+23]*dt*dzeta)/(NH_czeta*0.5*dt + 1.0); 
            Gxyz[Gc_AN][j+23] += dv;
            My_sum += dv*dv; 
	  }
        }
      }

      NH_czeta += dzeta; 

      MPI_Allreduce(&My_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      sum += dzeta*dzeta;

      if (sum<1.0e-12) po = 1;

      num++; 

      if (20<num) po = 1;

    } while(po==0);

    /****************************************************
     correct so that the sum of velocities can be zero.                              
    ****************************************************/

    Correct_Velocity();  

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    Ukc = 0.0;
    My_Ukc = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      sum = 0.0;
      for (j=1; j<=3; j++){
        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	}
      }
      My_Ukc += 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    }

    /****************************************************
     MPI: Ukc 
    ****************************************************/

    MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /* calculation of temperature (K) */

    Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ****************************************************/

    NH_Ham = Utot + Ukc + 0.5*NH_czeta*NH_czeta*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*GivenTemp*NH_R/eV2Hartree; 

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(iter,MD_TimeStep*(iter-1),fileE);
    }

    /*****************************************************
     first step:

       v(t)    ->  v(t+0.5*dt) 
       r(t)    ->  r(t+dt) 
       zeta(t) ->  zeta(t+0.5*dt)
       R(t)    ->  R(t+dt) 
    *****************************************************/

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	for (j=1; j<=3; j++){

	  if (atom_Fixed_XYZ[Gc_AN][j]==0){
        
	    scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  

	    /* v( r+0.5*dt ) */
	    Gxyz[Gc_AN][26+j] = Gxyz[Gc_AN][23+j] + (scaled_force - NH_czeta*Gxyz[Gc_AN][23+j])*0.5*dt;
	    Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j];

	    /* r( r+dt ) */
	    Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
	    Gxyz[Gc_AN][j] =  Gxyz[Gc_AN][j] + Gxyz[Gc_AN][26+j]*dt;
	  }
	}
      }

      /* zeta( t+0.5*dt ) */

      NH_nzeta = NH_czeta + (Ukc - 1.5*kB*(double)atomnum*GivenTemp/eV2Hartree)*dt/(TempQ*Wscale);
      NH_czeta = NH_nzeta;

      /* R( r+dt ) */
      NH_R = NH_R + NH_nzeta*dt;

    }

  }

  else {

    /****************************************************
      second step in velocity Verlet 
    ****************************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      for (j=1; j<=3; j++){

        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  
          Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j] + scaled_force*0.5*dt;
	}
      }
    }

    /****************************************************
     correct so that the sum of velocities can be zero.                              
    ****************************************************/

    Correct_Velocity();  

    /****************************************************
                       Kinetic Energy 
    ****************************************************/

    My_Ukc = 0.0;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];

      sum = 0.0;
      for (j=1; j<=3; j++){
        if (atom_Fixed_XYZ[Gc_AN][j]==0){
          sum = sum + Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
	}
      }
      My_Ukc = My_Ukc + 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
    }

    /****************************************************
     MPI: Ukc 
    ****************************************************/

    MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /* calculation of temperature (K) */
    Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;

    /****************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ****************************************************/

    NH_Ham = Utot + Ukc + 0.5*NH_czeta*NH_czeta*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*GivenTemp*NH_R/eV2Hartree; 

    /****************************************************
     write informatins to *.ene
    ****************************************************/

    if (myid==Host_ID){  

      sprintf(fileE,"%s%s.ene",filepath,filename);
      iterout_md(iter,MD_TimeStep*(iter-1),fileE);
    }

    /****************************************************
      first step in velocity Verlet 
    ****************************************************/

    if (MD_Opt_OK!=1 && iter!=MD_IterNumber){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	for (j=1; j<=3; j++){

	  if (atom_Fixed_XYZ[Gc_AN][j]==0){

	    scaled_force = -Gxyz[Gc_AN][16+j]/(Gxyz[Gc_AN][20]*Wscale);  
	    /* v( r+0.5*dt ) */
	    Gxyz[Gc_AN][26+j] = Gxyz[Gc_AN][23+j] + scaled_force*0.5*dt;
	    Gxyz[Gc_AN][23+j] = Gxyz[Gc_AN][26+j];

	    /* r( r+dt ) */
	    Gxyz[Gc_AN][20+j] = Gxyz[Gc_AN][j];
	    Gxyz[Gc_AN][j] =  Gxyz[Gc_AN][j] + Gxyz[Gc_AN][26+j]*dt;
	  }
	}
      }
    }

    /******************************************************
     Nose-Hoover Hamiltonian which is a conserved quantity
    ******************************************************/

    NH_Ham = Utot + Ukc + 0.5*NH_czeta*NH_czeta*TempQ*Wscale
                        + 3.0*kB*(double)atomnum*GivenTemp*NH_R/eV2Hartree; 

  }

  /****************************************************
   MPI: Gxyz
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];
    MPI_Bcast(&Gxyz[Gc_AN][1],   3, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Gxyz[Gc_AN][14], 19, MPI_DOUBLE, ID, mpi_comm_level1);
  }
}


static void Correct_Force()
{
  int Mc_AN,Gc_AN,nx,ny,nz;
  double my_sumx,my_sumy,my_sumz;
  double sumx,sumy,sumz;

  my_sumx = 0.0;
  my_sumy = 0.0;
  my_sumz = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    if (atom_Fixed_XYZ[Gc_AN][1]==0) my_sumx += Gxyz[Gc_AN][17];  
    if (atom_Fixed_XYZ[Gc_AN][2]==0) my_sumy += Gxyz[Gc_AN][18];  
    if (atom_Fixed_XYZ[Gc_AN][3]==0) my_sumz += Gxyz[Gc_AN][19];
  }  

  MPI_Allreduce(&my_sumx, &sumx, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_sumy, &sumy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_sumz, &sumz, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  nx = 0; ny = 0; nz = 0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    if (atom_Fixed_XYZ[Gc_AN][1]==0) nx++;
    if (atom_Fixed_XYZ[Gc_AN][2]==0) ny++;
    if (atom_Fixed_XYZ[Gc_AN][3]==0) nz++;
  }

  if (nx!=0){

    sumx /= (double)nx;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      if (atom_Fixed_XYZ[Gc_AN][1]==0) Gxyz[Gc_AN][17] -= sumx;  
    }
  }

  if (ny!=0){

    sumy /= (double)ny;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      if (atom_Fixed_XYZ[Gc_AN][2]==0) Gxyz[Gc_AN][18] -= sumy;  
    }
  }

  if (nz!=0){

    sumz /= (double)nz;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      if (atom_Fixed_XYZ[Gc_AN][3]==0) Gxyz[Gc_AN][19] -= sumz;  
    }
  }

}


static void Correct_Velocity()
{
  int Mc_AN,Gc_AN,nx,ny,nz;
  double my_sumx,my_sumy,my_sumz;
  double sumx,sumy,sumz;

  my_sumx = 0.0;
  my_sumy = 0.0;
  my_sumz = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    if (atom_Fixed_XYZ[Gc_AN][1]==0) my_sumx += Gxyz[Gc_AN][24];  
    if (atom_Fixed_XYZ[Gc_AN][2]==0) my_sumy += Gxyz[Gc_AN][25];  
    if (atom_Fixed_XYZ[Gc_AN][3]==0) my_sumz += Gxyz[Gc_AN][26];
  }  

  MPI_Allreduce(&my_sumx, &sumx, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_sumy, &sumy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_sumz, &sumz, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  nx = 0; ny = 0; nz = 0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    if (atom_Fixed_XYZ[Gc_AN][1]==0) nx++;
    if (atom_Fixed_XYZ[Gc_AN][2]==0) ny++;
    if (atom_Fixed_XYZ[Gc_AN][3]==0) nz++;
  }

  if (nx!=0){

    sumx /= (double)nx;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      if (atom_Fixed_XYZ[Gc_AN][1]==0) Gxyz[Gc_AN][24] -= sumx;  
    }
  }

  if (ny!=0){

    sumy /= (double)ny;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      if (atom_Fixed_XYZ[Gc_AN][2]==0) Gxyz[Gc_AN][25] -= sumy;  
    }
  }

  if (nz!=0){

    sumz /= (double)nz;
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      if (atom_Fixed_XYZ[Gc_AN][3]==0) Gxyz[Gc_AN][26] -= sumz;  
    }
  }
}





static void Delta_Factor(int iter)
{
  int i,j,Gc_AN,myid;
  char fileE[YOUSO10];
  double scaling_factor;
  static double tv0[4][4];
  static double Cell_Volume0;
  static int firsttime=1;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
   store tv into tv0 
  ****************************************************/

  if (firsttime){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	tv0[i][j] = tv[i][j];
      }
    }

    Cell_Volume0 = Cell_Volume;

    firsttime = 0;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    scale tv 
  ****************************************************/

  if      (iter==1) scaling_factor = pow(0.940,1.0/3.0);
  else if (iter==2) scaling_factor = pow(0.960,1.0/3.0);
  else if (iter==3) scaling_factor = pow(0.980,1.0/3.0);
  else if (iter==4) scaling_factor = pow(1.020,1.0/3.0);
  else if (iter==5) scaling_factor = pow(1.040,1.0/3.0);
  else if (iter==6) scaling_factor = pow(1.060,1.0/3.0);

  if (scaling_factor<7){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	tv[i][j] = tv0[i][j]*scaling_factor;
      }
    }

    /****************************************************
                update cartesian coordinates
    ****************************************************/

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

      Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

      Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
    }
  }

}




static void EvsLC(int iter)
{
  int i,j,Gc_AN,myid;
  char fileE[YOUSO10];
  static double tv0[4][4];
  static int firsttime=1;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
   store tv into tv0 
  ****************************************************/

  if (firsttime){

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
	tv0[i][j] = tv[i][j];
      }
    }

    firsttime = 0;
  }

  /****************************************************
   write informatins to *.ene
  ****************************************************/

  if (myid==Host_ID){  
    sprintf(fileE,"%s%s.ene",filepath,filename);
    iterout_md(iter,MD_TimeStep*(iter-1),fileE);
  }

  /****************************************************
    scale tv 
  ****************************************************/

  for (i=1; i<=3; i++){
    if (MD_EvsLattice_flag[i-1]==1){
      for (j=1; j<=3; j++){
        tv[i][j] += tv0[i][j]*MD_EvsLattice_Step/100.0;
      }
    }
    else if (MD_EvsLattice_flag[i-1]==-1){
      for (j=1; j<=3; j++){
        tv[i][j] -= tv0[i][j]*MD_EvsLattice_Step/100.0;
      }
    }
  }

  /****************************************************
    update cartesian coordinates
  ****************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

    Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

    Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
  }
}



void Calc_Temp_Atoms(int iter)
{
  int i,Mc_AN,Gc_AN,j,My_Nr,Nr;
  double My_Ukc,sum,Wscale,dt,v;

  /* calculation of temperature (K) of the atomic system */

  dt = 41.3411*MD_TimeStep;
  Wscale = unified_atomic_mass_unit/electron_mass;
  Ukc = 0.0;
  My_Ukc = 0.0;
  My_Nr = 0;
  Nr = 0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    sum = 0.0;
    for (j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
	sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
        My_Nr++;
      }
    }
    My_Ukc += 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  MPI_Allreduce(&My_Ukc, &Ukc, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Nr, &Nr, 1, MPI_INT, MPI_SUM, mpi_comm_level1);

  Temp = Ukc/(0.5*kB*(double)Nr)*eV2Hartree;

  /* for old version */
  /*
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;
  */

  /* calculation of given temperature (K) of the atomic system */

  for (i=1; i<=TempNum; i++) {
    if( (iter>TempPara[i-1][1]) && (iter<=TempPara[i][1]) ) {
      GivenTemp = TempPara[i-1][2] + (TempPara[i][2] - TempPara[i-1][2])*
          ((double)iter-(double)TempPara[i-1][1])/((double)TempPara[i][1]-(double)TempPara[i-1][1]);
    }
  }
}




int RestartFiles4GeoOpt(char *mode)
{
  int i,j,num;
  int success_flag;
  int numprocs,myid;
  char file_name[YOUSO10];
  FILE *fp;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
   
  /* write */
  if ( strcasecmp(mode,"write")==0) {

    sprintf(file_name,"%s%s_rst/%s.rst4gopt",filepath,filename,filename);

    if (myid==Host_ID){

      if ( (fp = fopen(file_name,"wb")) != NULL ){

	num = M_GDIIS_HISTORY + 1;
      
	for(i=0; i<num; i++) {
	  for(j=0; j<(atomnum+1); j++) {
	    fwrite(GxyzHistoryIn[i][j], sizeof(double), 4, fp);
	  }
	}

	for(i=0; i<num; i++) {
	  for(j=0; j<(atomnum+1); j++) {
	    fwrite(GxyzHistoryR[i][j], sizeof(double), 4, fp);
	  }
	}

	for(i=0; i<Extrapolated_Charge_History; i++) {
	  fwrite(His_Gxyz[i], sizeof(double), atomnum*3, fp);
	}

	/* Geometry_Opt_DIIS_EF */
	if (MD_switch==4){
	  for (i=0; i<(3*atomnum+2); i++){
	    fwrite(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	  }
	}

	/* BFGS */
	if (MD_switch==5){
	  for (i=0; i<(3*atomnum+2); i++){
	    fwrite(InvHessian[i], sizeof(double), 3*atomnum+2, fp);
	  }
	}

	/* RF */
	if (MD_switch==6){
	  for (i=0; i<(3*atomnum+2); i++){
	    fwrite(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	  }
	}

	/* RFC */
	if (MD_switch==18){
	  for (i=0; i<(3*atomnum+10); i++){
	    fwrite(Hessian[i], sizeof(double), 3*atomnum+10, fp);
	  }
	}

	success_flag = 1;
	fclose(fp);
      }
      else{
        printf("Failure of saving %s\n",file_name);
        success_flag = 0;
      }
    }

    else{
      success_flag = 1;
    }

    /* MPI_Bcast */    
    MPI_Bcast(&success_flag, 1, MPI_INT, Host_ID, mpi_comm_level1);
  }

  /* read */
  else if (strcasecmp(mode,"read")==0) {

    sprintf(file_name,"%s%s_rst/%s.rst4gopt",filepath,filename,filename);

    if ((fp = fopen(file_name,"rb")) != NULL){

      num = M_GDIIS_HISTORY + 1;
      
      for(i=0; i<num; i++) {
	for(j=0; j<(atomnum+1); j++) {
	  fread(GxyzHistoryIn[i][j], sizeof(double), 4, fp);
	}
      }

      for(i=0; i<num; i++) {
	for(j=0; j<(atomnum+1); j++) {
	  fread(GxyzHistoryR[i][j], sizeof(double), 4, fp);
	}
      }

      for(i=0; i<Extrapolated_Charge_History; i++) {
	fread(His_Gxyz[i], sizeof(double), atomnum*3, fp);
      }

      /* Geometry_Opt_DIIS_EF */
      if (MD_switch==4){
	for (i=0; i<(3*atomnum+2); i++){
	  fread(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	}
      }

      /* BFGS */
      if (MD_switch==5){
	for (i=0; i<(3*atomnum+2); i++){
	  fread(InvHessian[i], sizeof(double), 3*atomnum+2, fp);
	}
      }

      /* RF */
      if (MD_switch==6){
	for (i=0; i<(3*atomnum+2); i++){
	  fread(Hessian[i], sizeof(double), 3*atomnum+2, fp);
	}
      }

      /* RFC */
      if (MD_switch==18){
	for (i=0; i<(3*atomnum+10); i++){
	  fread(Hessian[i], sizeof(double), 3*atomnum+10, fp);
	}
      }

      success_flag = 1;
      fclose(fp);

    }
    else{
      printf("Failure of reading %s\n",file_name);
      success_flag = 0;
    }
  }

  return success_flag;
}


void Output_abc_file(int iter)
{
  int i,j,k;
  FILE *fp_abc;
  char buf[fp_bsize];          /* setvbuf */
  char fileabc[YOUSO10]="",dotabc[YOUSO10],iter_s[YOUSO10];

  printf("\nOutputting to abc file.\n");
  snprintf(iter_s,sizeof(iter_s),"%d",iter);
  strcpy(dotabc,".abc");
  fnjoint(iter_s,dotabc,fileabc);
  fnjoint(filepath,filename,fileabc);

  if ((fp_abc = fopen(fileabc,"w")) != NULL){

    setvbuf(fp_abc,buf,_IOFBF,fp_bsize);  /* setvbuf */

    fprintf(fp_abc,"# Total Energy (Hartree), abc-coordinates and Lattice Parameters (Ang) #\n");
    fprintf(fp_abc,"TotalEnergy   %18.15f\n",Utot);
    fprintf(fp_abc,"Atoms.Number  %i\n",atomnum);
    fprintf(fp_abc,"<Atoms.SpeciesAndCoordinates\n");

    /* atomic coordinates in case of colliear non-spin polarization */

    if (SpinP_switch==0){

      for (k=1; k<=atomnum; k++){
	i = WhatSpecies[k];
	j = Spe_WhatAtom[i];
	fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f\n",
		k,Atom_Symbol[j],
		Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR);
      }
    }

    /* atomic coordinates in case of colliear spin polarization */

    else if (SpinP_switch==1){

      for (k=1; k<=atomnum; k++){
	i = WhatSpecies[k];
	j = Spe_WhatAtom[i];

        if ( 0.0<=(InitN_USpin[k] - InitN_DSpin[k])){
  	   fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f %18.15f %18.15f %18.14f %18.14f\n",
		   k,Atom_Symbol[j],
		   Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
                   InitN_USpin[k]+InitN_DSpin[k],
                   fabs(InitN_USpin[k]-InitN_DSpin[k]),
                   0.0,0.0);
	}
        else {
  	   fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f %18.15f %18.15f %18.14f %18.14f\n",
		   k,Atom_Symbol[j],
		   Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
                   InitN_USpin[k]+InitN_DSpin[k],
                   fabs(InitN_USpin[k]-InitN_DSpin[k]),
                   180.0,0.0);
        }

      }
    }

    /* atomic coordinates in case of non-colliear spin polarization */

    else if (SpinP_switch==3){

      for (k=1; k<=atomnum; k++){
	i = WhatSpecies[k];
	j = Spe_WhatAtom[i];

	fprintf(fp_abc," %4d  %4s   %18.15f %18.15f %18.15f %18.15f %18.15f %18.14f %18.14f\n",
		k,Atom_Symbol[j],
		Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
		InitN_USpin[k]+InitN_DSpin[k],
		InitN_USpin[k]-InitN_DSpin[k],
		Angle0_Spin[k]/PI*180.0, Angle1_Spin[k]/PI*180.0);

        /* check negativity of magnetic moment */

        if ( (InitN_USpin[k]-InitN_DSpin[k])<0.0 ){

          printf("found an error (negativity) in writing the abc file.\n");
          MPI_Finalize();
          exit(1);
        }

      }

      


    }

    /* cell vectors */

    fprintf(fp_abc,"Atoms.SpeciesAndCoordinates>\n");
    fprintf(fp_abc,"<Atoms.UnitVectors\n");
    fprintf(fp_abc,"%18.15f %18.15f %18.15f\n",tv[1][1]*BohrR,tv[1][2]*BohrR,tv[1][3]*BohrR);
    fprintf(fp_abc,"%18.15f %18.15f %18.15f\n",tv[2][1]*BohrR,tv[2][2]*BohrR,tv[2][3]*BohrR);
    fprintf(fp_abc,"%18.15f %18.15f %18.15f\n",tv[3][1]*BohrR,tv[3][2]*BohrR,tv[3][3]*BohrR);
    fprintf(fp_abc,"Atoms.UnitVectors>\n");
    fprintf(fp_abc,"# End #\n");
    fclose(fp_abc);
  }
  else
    printf("error(1) in MD_pac.c\n");

}


void Estimate_Initial_Hessian(int diis_iter, int CellOpt_flag, double itv[4][4])
{
  int numprocs,myid,ID;

  /* MPI communication */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************
               the identity matrix 
  *******************************************/

  if (Initial_Hessian_flag==0){

    int i,j;

    if (CellOpt_flag==1){

      for (i=1; i<=(3*atomnum+9); i++){     
	for (j=1; j<=(3*atomnum+9); j++){     
	  Hessian[i][j] = 0.0;
	}
	Hessian[i][i] = 1.0;
      }
    }

    else{

      for (i=1; i<=3*atomnum; i++){     
	for (j=1; j<=3*atomnum; j++){     
	  Hessian[i][j] = 0.0;
	}
	Hessian[i][i] = 1.0;
      }
    }
  }

  /**************************************************************
      A model Hessian proposed by H.B. Schlegel

      Refs.
      H.B. Schlegel, Theo. Chim. Acta (Berl.) 66, 333 (1984).
      J.M.Wittbrodt and H.B. Schlegel, 
      J. Mol. Str. (Theochem) 398-399, 55 (1997).
  **************************************************************/

  else if (Initial_Hessian_flag==1){

    int i,j,k,I,J;
    int Mc_AN,Gc_AN,h_AN,Gh_AN,Rn;
    int wsp1,wsp2,m1,m2,n1,n2;
    double r,g[4],gr,d;
    double *Hess_tmp;
    double B[8][8];

    B[0][0] =  0.5000; B[0][1] =  0.5000; B[0][2] =  0.5000; B[0][3] =  0.5000; B[0][4] =  0.5000; B[0][5] =  0.5000; B[0][6] =  0.5000; B[0][7] =  0.5000;
    B[1][0] =  0.5000; B[1][1] = -0.2573; B[1][2] =  0.3401; B[1][3] =  0.6937; B[1][4] =  0.7126; B[1][5] =  0.8335; B[1][6] =  0.9491; B[1][7] =  1.0000;
    B[2][0] =  0.5000; B[2][1] =  0.3401; B[2][2] =  0.9652; B[2][3] =  1.2843; B[2][4] =  1.4625; B[2][5] =  1.6549; B[2][6] =  1.7190; B[2][7] =  2.0000;
    B[3][0] =  0.5000; B[3][1] =  0.6937; B[3][2] =  1.2843; B[3][3] =  1.6925; B[3][4] =  1.8238; B[3][5] =  2.1164; B[3][6] =  2.3185; B[3][7] =  2.5000;
    B[4][0] =  0.5000; B[4][1] =  0.7126; B[4][2] =  1.4625; B[4][3] =  1.8238; B[4][4] =  2.0203; B[4][5] =  2.2137; B[4][6] =  2.5206; B[4][7] =  2.7000;
    B[5][0] =  0.5000; B[5][1] =  0.8335; B[5][2] =  1.6549; B[5][3] =  2.1164; B[5][4] =  2.2137; B[5][5] =  2.3718; B[5][6] =  2.5110; B[5][7] =  2.7000;
    B[6][0] =  0.5000; B[6][1] =  0.9491; B[6][2] =  1.7190; B[6][3] =  2.3185; B[6][4] =  2.5206; B[6][5] =  2.5110; B[6][6] =  2.5200; B[6][7] =  2.7000;
    B[7][0] =  0.5000; B[7][1] =  1.0000; B[7][2] =  2.0000; B[7][3] =  2.5000; B[7][4] =  2.7000; B[7][5] =  2.7000; B[7][6] =  2.7000; B[7][7] =  2.9000;

    /* initialize Hessian */

    if (CellOpt_flag==1){

      for (i=1; i<=(3*atomnum+9); i++){     
	for (j=1; j<=(3*atomnum+9); j++){     
	  Hessian[i][j] = 0.0;
	}
      }

      for (i=1; i<=3*atomnum; i++){     
        Hessian[i][i] = 0.1;
      }

      for (i=3*atomnum+1; i<=(3*atomnum+9); i++){     
        Hessian[i][i] = 0.05;
      }
    }

    else{

      for (i=1; i<=3*atomnum; i++){     
	for (j=1; j<=3*atomnum; j++){     
	  Hessian[i][j] = 0.0;
	}
	Hessian[i][i] = 0.1;
      }
    }

    /* calculate the approximate Hessian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    

      wsp1 = WhatSpecies[Gc_AN];
      m1 = Spe_WhatAtom[wsp1];
      if      (m1==1)   n1 = 1;
      else if (m1<=10)  n1 = 2;
      else if (m1<=18)  n1 = 3;
      else if (m1<=36)  n1 = 4;
      else if (m1<=54)  n1 = 5;
      else if (m1<=86)  n1 = 6;
      else if (m1<=103) n1 = 7;
      if (m1==2 || m1==10 || m1==18 || m1==36 || m1==54 || m1==86) n1 = 0;

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	wsp2 = WhatSpecies[Gh_AN];
	m2 = Spe_WhatAtom[wsp2];

	if      (m2==1)   n2 = 1;
	else if (m2<=10)  n2 = 2;
	else if (m2<=18)  n2 = 3;
	else if (m2<=36)  n2 = 4;
	else if (m2<=54)  n2 = 5;
	else if (m2<=86)  n2 = 6;
	else if (m2<=103) n2 = 7;
	if (m2==2 || m2==10 || m2==18 || m2==36 || m2==54 || m2==86) n2 = 0;

	Rn = ncn[Gc_AN][h_AN];
	r = Dis[Gc_AN][h_AN];

	g[1] = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	g[2] = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	g[3] = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;

	d = r - B[n1][n2];
	gr = 1.734/d/d/d;

	/* diagonal terms */
   
	for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    Hessian[(Gc_AN-1)*3+i][(Gc_AN-1)*3+j] += g[i]*g[j]*gr;
	  }
	}
      
	/* off-diagonal terms */

	for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    Hessian[(Gc_AN-1)*3+i][(Gh_AN-1)*3+j] += -g[i]*g[j]*gr;
	  }
	}

      } /* h_AN */
    } /* Mc_AN */

    /* MPI communication: Hessian */

    Hess_tmp = (double*)malloc(sizeof(double)*(atomnum+1)*9);

    for (i=1; i<=atomnum; i++){     

      ID = G2ID[i];

      if (myid==ID){
	for (k=0; k<3; k++){
	  for (j=1; j<=3*atomnum; j++){
	    Hess_tmp[3*k*atomnum+j-1] = Hessian[(i-1)*3+k+1][j];
	  }
	}
      }

      MPI_Bcast(&Hess_tmp[0], atomnum*9, MPI_DOUBLE, ID, mpi_comm_level1);

      if (myid!=ID){
	for (k=0; k<3; k++){
	  for (j=1; j<=3*atomnum; j++){
	    Hessian[(i-1)*3+k+1][j] = Hess_tmp[3*k*atomnum+j-1];
	  }
	}
      }
    }

    /**************************************************************
                        if (CellOpt_flag==1)    
    **************************************************************/

    if (CellOpt_flag==1){

      int I,J,ki,kj;
      double sum,HesF[4][4];

      /* converting Hessian w.r.t cartesian coordinate to that w.r.t. fractional coordinate */

      for (I=1; I<=atomnum; I++){     
        for (J=1; J<=atomnum; J++){     
	  for (i=1; i<=3; i++){     
	    for (j=1; j<=3; j++){     

              sum = 0.0;

	      for (ki=1; ki<=3; ki++){     
		for (kj=1; kj<=3; kj++){     
                  sum += Hessian[(I-1)*3+ki][(J-1)*3+kj]*tv[i][ki]*tv[j][kj];
		}
	      }

              HesF[i][j] = sum;
	    }
	  }

	  for (i=1; i<=3; i++){     
	    for (j=1; j<=3; j++){     
              Hessian[(I-1)*3+i][(J-1)*3+j] = HesF[i][j];
	    }
	  }

        }
      }

      /* calculate d2E_de2 */

      int k1,k2,l1,l2,q1,q2;
      double d2E_de2[4][4][4][4];
      double d2E_da2[4][4][4][4];

      for (k1=1; k1<=3; k1++){
        for (l1=1; l1<=3; l1++){
	  for (k2=1; k2<=3; k2++){
	    for (l2=1; l2<=3; l2++){
              d2E_de2[k1][l1][k2][l2] = 0.0;
	    }
	  }
        }
      }

      for (k1=1; k1<=3; k1++){
        for (l1=1; l1<=3; l1++){
	  for (k2=1; k2<=3; k2++){
	    for (l2=1; l2<=3; l2++){

              for (I=1; I<=atomnum; I++){     

		wsp1 = WhatSpecies[I];
		m1 = Spe_WhatAtom[wsp1];
		if      (m1==1)   n1 = 1;
		else if (m1<=10)  n1 = 2;
		else if (m1<=18)  n1 = 3;
		else if (m1<=36)  n1 = 4;
		else if (m1<=54)  n1 = 5;
		else if (m1<=86)  n1 = 6;
		else if (m1<=103) n1 = 7;
		if (m1==2 || m1==10 || m1==18 || m1==36 || m1==54 || m1==86) n1 = 0;


                for (h_AN=1; h_AN<=FNAN[I]; h_AN++){

		  J = natn[I][h_AN];
		  wsp2 = WhatSpecies[J];
		  m2 = Spe_WhatAtom[wsp2];

		  if      (m2==1)   n2 = 1;
		  else if (m2<=10)  n2 = 2;
		  else if (m2<=18)  n2 = 3;
		  else if (m2<=36)  n2 = 4;
		  else if (m2<=54)  n2 = 5;
		  else if (m2<=86)  n2 = 6;
		  else if (m2<=103) n2 = 7;
		  if (m2==2 || m2==10 || m2==18 || m2==36 || m2==54 || m2==86) n2 = 0;

		  Rn = ncn[I][h_AN];
		  r = Dis[I][h_AN];

		  g[1] = (Gxyz[I][1] - Gxyz[J][1] - atv[Rn][1])/r;
		  g[2] = (Gxyz[I][2] - Gxyz[J][2] - atv[Rn][2])/r;
		  g[3] = (Gxyz[I][3] - Gxyz[J][3] - atv[Rn][3])/r;

		  d = r - B[n1][n2];
		  gr = 1.734/d/d/d;


                  d2E_de2[k1][l1][k2][l2] += 0.5*gr*g[k1]*g[k2]*g[l1]*g[l2]*r*r;

		}
	      }
	    }
	  }
        }
      }

      /* calculate d2E_da2 */

      for (k1=1; k1<=3; k1++){
        for (l1=1; l1<=3; l1++){
	  for (k2=1; k2<=3; k2++){
	    for (l2=1; l2<=3; l2++){
              d2E_da2[k1][l1][k2][l2] = 0.0;
	    }
	  }
        }
      }

      for (k1=1; k1<=3; k1++){
        for (l1=1; l1<=3; l1++){
	  for (k2=1; k2<=3; k2++){
	    for (l2=1; l2<=3; l2++){

              for (m1=1; m1<=3; m1++){
		for (m2=1; m2<=3; m2++){

		  d2E_da2[k1][l1][k2][l2] += itv[m1][k1]*itv[m2][k2]*d2E_de2[m1][l1][m2][l2];

		}
	      }
	    }
	  }
	}
      }

      /* Set d2E_da2 to Hessian */

      q1 = 0;
      for (k1=1; k1<=3; k1++){
        for (l1=1; l1<=3; l1++){

          q1++;

          q2 = 0;
	  for (k2=1; k2<=3; k2++){
	    for (l2=1; l2<=3; l2++){

              q2++;

	      Hessian[3*atomnum+q1][3*atomnum+q2] += d2E_da2[k1][l1][k2][l2];
	    }
	  }
	}
      }

      /* calculate d2E_dqde */

      for (I=1; I<=atomnum; I++){     
        for (i=1; i<=3; i++){

	  wsp1 = WhatSpecies[I];
	  m1 = Spe_WhatAtom[wsp1];
	  if      (m1==1)   n1 = 1;
	  else if (m1<=10)  n1 = 2;
	  else if (m1<=18)  n1 = 3;
	  else if (m1<=36)  n1 = 4;
	  else if (m1<=54)  n1 = 5;
	  else if (m1<=86)  n1 = 6;
	  else if (m1<=103) n1 = 7;
	  if (m1==2 || m1==10 || m1==18 || m1==36 || m1==54 || m1==86) n1 = 0;

	  for (k1=1; k1<=3; k1++){
	    for (l1=1; l1<=3; l1++){

	      for (h_AN=1; h_AN<=FNAN[I]; h_AN++){

		J = natn[I][h_AN];
		wsp2 = WhatSpecies[J];
		m2 = Spe_WhatAtom[wsp2];

		if      (m2==1)   n2 = 1;
		else if (m2<=10)  n2 = 2;
		else if (m2<=18)  n2 = 3;
		else if (m2<=36)  n2 = 4;
		else if (m2<=54)  n2 = 5;
		else if (m2<=86)  n2 = 6;
		else if (m2<=103) n2 = 7;
		if (m2==2 || m2==10 || m2==18 || m2==36 || m2==54 || m2==86) n2 = 0;

		Rn = ncn[I][h_AN];
		r = Dis[I][h_AN];

		g[1] = (Gxyz[I][1] - Gxyz[J][1] - atv[Rn][1])/r;
		g[2] = (Gxyz[I][2] - Gxyz[J][2] - atv[Rn][2])/r;
		g[3] = (Gxyz[I][3] - Gxyz[J][3] - atv[Rn][3])/r;

		d = r - B[n1][n2];
		gr = 1.734/d/d/d;

                for (k=1; k<=3; k++){
                  Hessian[3*(I-1)+i][atomnum*3+(k1-1)*3+l1] += 0.5*gr*g[k]*g[k1]*tv[i][k]*g[l1]*r;
		}

	      } /* h_AN */

	    }
	  }
        }
      }

      /* convert d2E_dqde to d2E_dqda */
       
      double tmpM[4][4][4];
   
      for (I=1; I<=atomnum; I++){     
        for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    for (l1=1; l1<=3; l1++){

              sum = 0.0;
	      for (k1=1; k1<=3; k1++){
                sum += itv[k1][j]*Hessian[3*(I-1)+i][atomnum*3+(k1-1)*3+l1];
	      }

              tmpM[i][j][l1] = sum;
	    }
	  }
	}

        for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    for (l1=1; l1<=3; l1++){
              Hessian[3*(I-1)+i][atomnum*3+(j-1)*3+l1] = tmpM[i][j][l1]; 
              Hessian[atomnum*3+(j-1)*3+l1][3*(I-1)+i] = tmpM[i][j][l1]; 
	    }
	  }
	}
      }

      /*
      if (myid==Host_ID){

      for (i=1; i<=(3*atomnum+9); i++){     
	for (j=1; j<=(3*atomnum+9); j++){     
	  printf("%7.4f ",Hessian[i][j]);
	}
	printf("\n");
      }
      }

      MPI_Finalize();
      exit(0);
      */

    }

    /* check constraint and correct Hessian */

    for (I=1; I<=atomnum; I++){     
      for (i=1; i<=3; i++){     

        if (atom_Fixed_XYZ[I][i]==1){

	  for (J=1; J<=atomnum; J++){     
	    for (j=1; j<=3; j++){     
	      Hessian[(I-1)*3+i][(J-1)*3+j] = 0.0;
	      Hessian[(J-1)*3+j][(I-1)*3+i] = 0.0;
	    }
	  }

          Hessian[(I-1)*3+i][(I-1)*3+i] = 1.0;
	}
      }
    }

    /* freeing of Hess_tmp */

    free(Hess_tmp);
  }

  /*******************************************
    an approximate Hessian by force fitting
  *******************************************/

  else if (Initial_Hessian_flag==2){

    int Gc_AN,Mc_AN,i,j,p,Rn;
    int wsp1,wsp2,m1,m2;
    int h_AN,Gh_AN,po,loopN;
    double **ep,**sig;
    double **dQ_dep,**dQ_dsig;
    double ***GLJ,ELJ[20],my_ELJ[20],r,r2,r3,g[4];
    double norm_sig,norm_ep,my_norm_sig,my_norm_ep;
    double sr,sr2,sr4,sr5,sr6,sr11,sr12;
    double df_dr,d2f_dr2,d2f_depdr,d2f_dsigdr;
    double dgx,dgy,dgz,coe_ep,coe_sig;
    double Q=0.0,Qold,My_Q,dg_dep[4],dg_dsig[4];
    double d2r_dx1dx2[4][4],d[4];
    double ep0[104],sig0[104];

    /* set sig0 in a.u. */

    sig0[  0] = 0.8908987/BohrR*0.75;
    sig0[  1] = 0.8908987/BohrR*0.75;
    sig0[  2] = 0.8908987/BohrR*0.64;
    sig0[  3] = 0.8908987/BohrR*2.68;  
    sig0[  4] = 0.8908987/BohrR*1.80;  
    sig0[  5] = 0.8908987/BohrR*1.64; 
    sig0[  6] = 0.8908987/BohrR*1.54; 
    sig0[  7] = 0.8908987/BohrR*1.50; 
    sig0[  8] = 0.8908987/BohrR*1.46; 
    sig0[  9] = 0.8908987/BohrR*1.42; 
    sig0[ 10] = 0.8908987/BohrR*1.38; 
    sig0[ 11] = 0.8908987/BohrR*3.08; 
    sig0[ 12] = 0.8908987/BohrR*2.60; 
    sig0[ 13] = 0.8908987/BohrR*2.36; 
    sig0[ 14] = 0.8908987/BohrR*2.22; 
    sig0[ 15] = 0.8908987/BohrR*2.12; 
    sig0[ 16] = 0.8908987/BohrR*2.04;
    sig0[ 17] = 0.8908987/BohrR*1.98; 

    sig0[ 18] = 0.8908987/BohrR*2.68; 
    sig0[ 19] = 0.8908987/BohrR*2.68; 
    sig0[ 20] = 0.8908987/BohrR*2.68; 
    sig0[ 21] = 0.8908987/BohrR*2.68; 
    sig0[ 22] = 0.8908987/BohrR*2.68; 
    sig0[ 23] = 0.8908987/BohrR*2.68; 
    sig0[ 24] = 0.8908987/BohrR*2.68; 
    sig0[ 25] = 0.8908987/BohrR*2.68; 
    sig0[ 26] = 0.8908987/BohrR*2.68; 
    sig0[ 27] = 0.8908987/BohrR*2.68; 
    sig0[ 28] = 0.8908987/BohrR*2.68; 
    sig0[ 29] = 0.8908987/BohrR*2.68; 
    sig0[ 30] = 0.8908987/BohrR*2.68; 
    sig0[ 31] = 0.8908987/BohrR*2.68; 
    sig0[ 32] = 0.8908987/BohrR*2.68; 
    sig0[ 33] = 0.8908987/BohrR*2.68; 
    sig0[ 34] = 0.8908987/BohrR*2.68; 
    sig0[ 35] = 0.8908987/BohrR*2.68; 
    sig0[ 36] = 0.8908987/BohrR*2.68; 
    sig0[ 37] = 0.8908987/BohrR*2.68; 
    sig0[ 38] = 0.8908987/BohrR*2.68; 
    sig0[ 39] = 0.8908987/BohrR*2.68; 
    sig0[ 40] = 0.8908987/BohrR*2.68; 
    sig0[ 41] = 0.8908987/BohrR*2.68; 
    sig0[ 42] = 0.8908987/BohrR*2.68; 
    sig0[ 43] = 0.8908987/BohrR*2.68; 
    sig0[ 44] = 0.8908987/BohrR*2.68; 
    sig0[ 45] = 0.8908987/BohrR*2.68; 
    sig0[ 46] = 0.8908987/BohrR*2.68; 

    sig0[ 47] = 0.8908987/BohrR*3.06; 

    sig0[ 48] = 0.8908987/BohrR*2.68; 
    sig0[ 49] = 0.8908987/BohrR*2.68; 
    sig0[ 50] = 0.8908987/BohrR*2.68; 
    sig0[ 51] = 0.8908987/BohrR*2.68; 
    sig0[ 52] = 0.8908987/BohrR*2.68; 
    sig0[ 53] = 0.8908987/BohrR*2.68; 
    sig0[ 54] = 0.8908987/BohrR*2.68; 
    sig0[ 55] = 0.8908987/BohrR*2.68; 
    sig0[ 56] = 0.8908987/BohrR*2.68; 
    sig0[ 57] = 0.8908987/BohrR*2.68; 
    sig0[ 58] = 0.8908987/BohrR*2.68; 
    sig0[ 59] = 0.8908987/BohrR*2.68; 
    sig0[ 60] = 0.8908987/BohrR*2.68; 
    sig0[ 61] = 0.8908987/BohrR*2.68; 
    sig0[ 62] = 0.8908987/BohrR*2.68; 
    sig0[ 63] = 0.8908987/BohrR*2.68; 
    sig0[ 64] = 0.8908987/BohrR*2.68; 
    sig0[ 65] = 0.8908987/BohrR*2.68; 
    sig0[ 66] = 0.8908987/BohrR*2.68; 
    sig0[ 67] = 0.8908987/BohrR*2.68; 
    sig0[ 68] = 0.8908987/BohrR*2.68; 
    sig0[ 69] = 0.8908987/BohrR*2.68; 
    sig0[ 70] = 0.8908987/BohrR*2.68; 
    sig0[ 71] = 0.8908987/BohrR*2.68; 
    sig0[ 72] = 0.8908987/BohrR*2.68; 
    sig0[ 73] = 0.8908987/BohrR*2.68; 
    sig0[ 74] = 0.8908987/BohrR*2.68; 
    sig0[ 75] = 0.8908987/BohrR*2.68; 
    sig0[ 76] = 0.8908987/BohrR*2.68; 
    sig0[ 77] = 0.8908987/BohrR*2.68; 
    sig0[ 78] = 0.8908987/BohrR*2.68; 
    sig0[ 79] = 0.8908987/BohrR*2.68; 
    sig0[ 80] = 0.8908987/BohrR*2.68; 
    sig0[ 81] = 0.8908987/BohrR*2.68; 
    sig0[ 82] = 0.8908987/BohrR*2.68; 
    sig0[ 83] = 0.8908987/BohrR*2.68; 
    sig0[ 84] = 0.8908987/BohrR*2.68; 
    sig0[ 85] = 0.8908987/BohrR*2.68; 
    sig0[ 86] = 0.8908987/BohrR*2.68; 
    sig0[ 87] = 0.8908987/BohrR*2.68; 
    sig0[ 88] = 0.8908987/BohrR*2.68; 
    sig0[ 89] = 0.8908987/BohrR*2.68; 
    sig0[ 90] = 0.8908987/BohrR*2.68; 
    sig0[ 91] = 0.8908987/BohrR*2.68; 
    sig0[ 92] = 0.8908987/BohrR*2.68; 
    sig0[ 93] = 0.8908987/BohrR*2.68; 
    sig0[ 94] = 0.8908987/BohrR*2.68; 
    sig0[ 95] = 0.8908987/BohrR*2.68; 
    sig0[ 96] = 0.8908987/BohrR*2.68; 
    sig0[ 97] = 0.8908987/BohrR*2.68; 
    sig0[ 98] = 0.8908987/BohrR*2.68; 
    sig0[ 99] = 0.8908987/BohrR*2.68; 
    sig0[100] = 0.8908987/BohrR*2.68; 
    sig0[101] = 0.8908987/BohrR*2.68; 
    sig0[102] = 0.8908987/BohrR*2.68; 
    sig0[103] = 0.8908987/BohrR*2.68; 

    /* set ep0 */

    ep0[  0] = 0.01/eV2Hartree;
    ep0[  1] = 0.27/eV2Hartree;
    ep0[  2] = 0.01/eV2Hartree;  
    ep0[  3] = 0.19/eV2Hartree;
    ep0[  4] = 0.39/eV2Hartree;
    ep0[  5] = 0.69/eV2Hartree;
    ep0[  6] = 0.87/eV2Hartree;
    ep0[  7] = 0.58/eV2Hartree;
    ep0[  8] = 0.31/eV2Hartree;
    ep0[  9] = 0.10/eV2Hartree;
    ep0[ 10] = 0.01/eV2Hartree;
    ep0[ 11] = 0.13/eV2Hartree;
    ep0[ 12] = 0.18/eV2Hartree;
    ep0[ 13] = 0.40/eV2Hartree;
    ep0[ 14] = 0.55/eV2Hartree;
    ep0[ 15] = 0.41/eV2Hartree;
    ep0[ 16] = 0.34/eV2Hartree;
    ep0[ 17] = 0.17/eV2Hartree;
    ep0[ 18] = 0.01/eV2Hartree;
    ep0[ 19] = 0.11/eV2Hartree;
    ep0[ 20] = 0.22/eV2Hartree;
    ep0[ 21] = 0.46/eV2Hartree;

    ep0[ 22] = 0.01/eV2Hartree;
    ep0[ 23] = 0.01/eV2Hartree;
    ep0[ 24] = 0.01/eV2Hartree;
    ep0[ 25] = 0.01/eV2Hartree;
    ep0[ 26] = 0.01/eV2Hartree;
    ep0[ 27] = 0.01/eV2Hartree;
    ep0[ 28] = 0.01/eV2Hartree;
    ep0[ 29] = 0.01/eV2Hartree;
    ep0[ 30] = 0.01/eV2Hartree;
    ep0[ 31] = 0.01/eV2Hartree;
    ep0[ 32] = 0.01/eV2Hartree;
    ep0[ 33] = 0.01/eV2Hartree;
    ep0[ 34] = 0.01/eV2Hartree;
    ep0[ 35] = 0.01/eV2Hartree;
    ep0[ 36] = 0.01/eV2Hartree;
    ep0[ 37] = 0.01/eV2Hartree;
    ep0[ 38] = 0.01/eV2Hartree;
    ep0[ 39] = 0.01/eV2Hartree;
    ep0[ 40] = 0.01/eV2Hartree;
    ep0[ 41] = 0.01/eV2Hartree;
    ep0[ 42] = 0.01/eV2Hartree;
    ep0[ 43] = 0.01/eV2Hartree;
    ep0[ 44] = 0.01/eV2Hartree;
    ep0[ 45] = 0.01/eV2Hartree;
    ep0[ 46] = 0.01/eV2Hartree;
    ep0[ 47] = 0.35/eV2Hartree;
    ep0[ 48] = 0.01/eV2Hartree; 
    ep0[ 49] = 0.01/eV2Hartree;
    ep0[ 50] = 0.01/eV2Hartree;
    ep0[ 51] = 0.01/eV2Hartree;
    ep0[ 52] = 0.01/eV2Hartree;
    ep0[ 53] = 0.01/eV2Hartree;
    ep0[ 54] = 0.01/eV2Hartree;
    ep0[ 55] = 0.01/eV2Hartree;
    ep0[ 56] = 0.01/eV2Hartree;
    ep0[ 57] = 0.01/eV2Hartree;
    ep0[ 58] = 0.01/eV2Hartree;
    ep0[ 59] = 0.01/eV2Hartree;
    ep0[ 60] = 0.01/eV2Hartree;
    ep0[ 61] = 0.01/eV2Hartree;
    ep0[ 62] = 0.01/eV2Hartree;
    ep0[ 63] = 0.01/eV2Hartree;
    ep0[ 64] = 0.01/eV2Hartree;
    ep0[ 65] = 0.01/eV2Hartree;
    ep0[ 66] = 0.01/eV2Hartree;
    ep0[ 67] = 0.01/eV2Hartree;
    ep0[ 68] = 0.01/eV2Hartree;
    ep0[ 69] = 0.01/eV2Hartree;
    ep0[ 70] = 0.01/eV2Hartree;
    ep0[ 71] = 0.01/eV2Hartree;
    ep0[ 72] = 0.01/eV2Hartree;
    ep0[ 73] = 0.01/eV2Hartree;
    ep0[ 74] = 0.01/eV2Hartree;
    ep0[ 75] = 0.01/eV2Hartree;
    ep0[ 76] = 0.01/eV2Hartree;
    ep0[ 77] = 0.01/eV2Hartree;
    ep0[ 78] = 0.01/eV2Hartree;
    ep0[ 79] = 0.01/eV2Hartree;
    ep0[ 80] = 0.01/eV2Hartree;
    ep0[ 81] = 0.01/eV2Hartree;
    ep0[ 82] = 0.01/eV2Hartree;
    ep0[ 83] = 0.01/eV2Hartree;
    ep0[ 84] = 0.01/eV2Hartree;
    ep0[ 85] = 0.01/eV2Hartree;
    ep0[ 86] = 0.01/eV2Hartree;
    ep0[ 87] = 0.01/eV2Hartree;
    ep0[ 88] = 0.01/eV2Hartree;
    ep0[ 89] = 0.01/eV2Hartree;
    ep0[ 90] = 0.01/eV2Hartree;
    ep0[ 91] = 0.01/eV2Hartree;
    ep0[ 92] = 0.01/eV2Hartree;
    ep0[ 93] = 0.01/eV2Hartree;
    ep0[ 94] = 0.01/eV2Hartree;
    ep0[ 95] = 0.01/eV2Hartree;
    ep0[ 96] = 0.01/eV2Hartree;
    ep0[ 97] = 0.01/eV2Hartree;
    ep0[ 98] = 0.01/eV2Hartree;
    ep0[ 99] = 0.01/eV2Hartree;
    ep0[100] = 0.01/eV2Hartree;
    ep0[101] = 0.01/eV2Hartree;
    ep0[102] = 0.01/eV2Hartree;
    ep0[103] = 0.01/eV2Hartree;

    /* allocation of arrays */

    GLJ = (double***)malloc(sizeof(double**)*(M_GDIIS_HISTORY+1));
    for (p=0; p<(M_GDIIS_HISTORY+1); p++){
      GLJ[p] = (double**)malloc(sizeof(double*)*(Matomnum+1));
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	GLJ[p][Mc_AN] = (double*)malloc(sizeof(double)*4);
	for (i=0; i<4; i++) GLJ[p][Mc_AN][i] = 0.0;
      }
    }

    ep = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        ep[Mc_AN] = (double*)malloc(sizeof(double)*1);
        ep[Mc_AN][0] = 0.0;
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        wsp1 = WhatSpecies[Gc_AN];
        m1 = Spe_WhatAtom[wsp1];

        ep[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
        for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
          wsp2 = WhatSpecies[Gh_AN];
          m2 = Spe_WhatAtom[wsp2];
          ep[Mc_AN][h_AN] = sqrt(ep0[m1]*ep0[m2]);  /* set the initial value */
	}
      }
    }

    sig = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        sig[Mc_AN] = (double*)malloc(sizeof(double)*1);
        sig[Mc_AN][0] = 0.0;
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        wsp1 = WhatSpecies[Gc_AN];
        m1 = Spe_WhatAtom[wsp1];

        sig[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
        for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
          wsp2 = WhatSpecies[Gh_AN];
          m2 = Spe_WhatAtom[wsp2];
          sig[Mc_AN][h_AN] = 0.5*(sig0[m1]+sig0[m2]); /* set the initial value */
	}
      }
    }

    dQ_dep = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        dQ_dep[Mc_AN] = (double*)malloc(sizeof(double)*1);
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        dQ_dep[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      }
    }

    dQ_dsig = (double**)malloc(sizeof(double*)*(Matomnum+1));
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      if (Mc_AN==0){
        dQ_dsig[Mc_AN] = (double*)malloc(sizeof(double)*1);
      }
      else {
        Gc_AN = M2G[Mc_AN];    
        dQ_dsig[Mc_AN] = (double*)malloc(sizeof(double)*(FNAN[Gc_AN]+1));
      }
    }

    /*************************************************************
      optimization of the obeject function for gradient fitting 
    *************************************************************/

    po = 0;
    loopN = 1;

    do {

      My_Q = 0.0;

      for (p=0; p<diis_iter; p++) {

	/* calculate the LJ total energy, the LJ gradients, and the object function Q */

	my_ELJ[p] = 0.0;

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    

	  GLJ[p][Mc_AN][1] = 0.0;
	  GLJ[p][Mc_AN][2] = 0.0;
	  GLJ[p][Mc_AN][3] = 0.0;

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Rn = ncn[Gc_AN][h_AN];

            d[1] = GxyzHistoryIn[p][Gc_AN][1] - GxyzHistoryIn[p][Gh_AN][1] - atv[Rn][1];
            d[2] = GxyzHistoryIn[p][Gc_AN][2] - GxyzHistoryIn[p][Gh_AN][2] - atv[Rn][2];
            d[3] = GxyzHistoryIn[p][Gc_AN][3] - GxyzHistoryIn[p][Gh_AN][3] - atv[Rn][3];
	    r = sqrt(d[1]*d[1]+d[2]*d[2]+d[3]*d[3]);

	    g[1] = d[1]/r;
	    g[2] = d[2]/r;
	    g[3] = d[3]/r;

	    sr = sig[Mc_AN][h_AN]/r;
	    sr2 = sr*sr;
	    sr4 = sr2*sr2;
	    sr6 = sr2*sr4;
	    sr12 = sr6*sr6;

	    my_ELJ[p] += 2.0*ep[Mc_AN][h_AN]*(sr12 - sr6); /* The factor of 0.5 is considered. */
         
	    d2f_depdr = 24.0*(sr6 - 2.0*sr12)/r;
	    df_dr = d2f_depdr*ep[Mc_AN][h_AN];

	    GLJ[p][Mc_AN][1] += g[1]*df_dr;
	    GLJ[p][Mc_AN][2] += g[2]*df_dr;
	    GLJ[p][Mc_AN][3] += g[3]*df_dr;
	  }

	  dgx = GLJ[p][Mc_AN][1] - GxyzHistoryR[p][Gc_AN][1];
	  dgy = GLJ[p][Mc_AN][2] - GxyzHistoryR[p][Gc_AN][2];
	  dgz = GLJ[p][Mc_AN][3] - GxyzHistoryR[p][Gc_AN][3];
      
	  My_Q += 0.5*(dgx*dgx+dgy*dgy+dgz*dgz);  

	} /* Mc_AN */
      } /* p */

      /* MPI of Q */ 

      Qold = Q;
      MPI_Allreduce(&My_Q, &Q, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

      for (p=0; p<diis_iter; p++) {
        MPI_Allreduce(&my_ELJ[p], &ELJ[p], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      }

      /*
      for (p=0; p<diis_iter; p++) {
        printf("ELJ[%d]=%18.15f\n",p,ELJ[p]);
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  printf("Mc_AN=%2d p=%2d GLJ=%18.15f %18.15f %18.15f\n",
                  Gc_AN,p,GLJ[p][Mc_AN][1],GLJ[p][Mc_AN][2],GLJ[p][Mc_AN][3]); 
	}    
      }
      */

      /* calculate the gradients of Q with respect to ep and sig */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  dQ_dep[Mc_AN][h_AN]  = 0.0;
	  dQ_dsig[Mc_AN][h_AN] = 0.0;
	}
      }

      for (p=0; p<diis_iter; p++) {
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    

	  for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	    Gh_AN = natn[Gc_AN][h_AN];
	    Rn = ncn[Gc_AN][h_AN];

            d[1] = GxyzHistoryIn[p][Gc_AN][1] - GxyzHistoryIn[p][Gh_AN][1] - atv[Rn][1];
            d[2] = GxyzHistoryIn[p][Gc_AN][2] - GxyzHistoryIn[p][Gh_AN][2] - atv[Rn][2];
            d[3] = GxyzHistoryIn[p][Gc_AN][3] - GxyzHistoryIn[p][Gh_AN][3] - atv[Rn][3];
	    r = sqrt(d[1]*d[1]+d[2]*d[2]+d[3]*d[3]);

	    g[1] = d[1]/r;
	    g[2] = d[2]/r;
	    g[3] = d[3]/r;

	    sr = sig[Mc_AN][h_AN]/r;
	    sr2 = sr*sr;
	    sr4 = sr2*sr2;
	    sr5 = sr*sr4;
	    sr6 = sr2*sr4;
	    sr11 = sr5*sr6;
	    sr12 = sr6*sr6;

	    d2f_depdr = 24.0*(sr6 - 2.0*sr12)/r;
	    df_dr = d2f_depdr*ep[Mc_AN][h_AN];
	    d2f_dsigdr = 144.0*ep[Mc_AN][h_AN]*(sr5 - 4.0*sr11)/r/r;

	    dg_dep[1] = g[1]*d2f_depdr;
	    dg_dep[2] = g[2]*d2f_depdr;
	    dg_dep[3] = g[3]*d2f_depdr;

	    dg_dsig[1] = g[1]*d2f_dsigdr;
	    dg_dsig[2] = g[2]*d2f_dsigdr;
	    dg_dsig[3] = g[3]*d2f_dsigdr;

  	    dgx = GLJ[p][Mc_AN][1] - GxyzHistoryR[p][Gc_AN][1];
	    dgy = GLJ[p][Mc_AN][2] - GxyzHistoryR[p][Gc_AN][2];
	    dgz = GLJ[p][Mc_AN][3] - GxyzHistoryR[p][Gc_AN][3];

	    dQ_dep[Mc_AN][h_AN]  += dgx*dg_dep[1] + dgy*dg_dep[2] + dgz*dg_dep[3];
	    dQ_dsig[Mc_AN][h_AN] += dgx*dg_dsig[1] + dgy*dg_dsig[2] + dgz*dg_dsig[3];

	    /*
	      printf("Gc_AN=%2d h_AN=%2d dQ_dep=%15.12f dQ_dsig=%15.12f\n",
	      Gc_AN,h_AN,dQ_dep[Mc_AN][h_AN],dQ_dsig[Mc_AN][h_AN]);
	    */

	  } /* h_AN */

	  /*
          printf("DDD1 p=%2d Gc_AN=%2d %15.12f %15.12f\n",
                       p,Gc_AN,GLJ[p][Mc_AN][1],GxyzHistoryR[p][Gc_AN][1]);
	  */

	} /* Mc_AN */
      } /* p */

      /* calculate the norm of gradients */

      my_norm_ep  = 0.0; 
      my_norm_sig = 0.0; 
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	  my_norm_ep  +=  dQ_dep[Mc_AN][h_AN]*dQ_dep[Mc_AN][h_AN];
	  my_norm_sig += dQ_dsig[Mc_AN][h_AN]*dQ_dsig[Mc_AN][h_AN];
	}
      }

      /* MPI of Q */ 

      if (loopN==1){

        MPI_Allreduce(&my_norm_ep,  &norm_ep,  1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        MPI_Allreduce(&my_norm_sig, &norm_sig, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

	norm_ep  = norm_ep/(double)atomnum;
	norm_sig = norm_sig/(double)atomnum;

        coe_ep  = 0.001/norm_ep; 
        if (0.1<coe_ep) coe_ep = 0.1;

        coe_sig = 0.001/norm_sig; 
        if (0.1<coe_sig) coe_sig = 0.1;

      }
      else{

        if (Q<Qold){
          coe_ep  = 1.5*coe_ep;
          coe_sig = 1.5*coe_sig;
	}
        else{
          coe_ep  = coe_ep/3.0;
          coe_sig = coe_sig/3.0;
        }
      }

      /*
      printf("loopN=%4d Q=%18.15f coe_ep=%15.12f coe_sig=%15.12f\n",loopN,Q,coe_ep,coe_sig);
      */

      /* update ep and sig */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	Gc_AN = M2G[Mc_AN];    

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

          ep[Mc_AN][h_AN]  -= coe_ep*dQ_dep[Mc_AN][h_AN]; 
          sig[Mc_AN][h_AN] -= coe_sig*dQ_dsig[Mc_AN][h_AN]; 
	}
      }

      if (Q<0.000000001) po = 1;

      loopN++;

    } while(po==0 && loopN<200);

    /* initialize Hessian */

    for (i=1; i<=3*atomnum; i++){     
      for (j=1; j<=3*atomnum; j++){     
	Hessian[i][j] = 0.0;
      }
      Hessian[i][i] = 0.1;
    }

    /* construct the Hessian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    

      for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Rn = ncn[Gc_AN][h_AN];
	r = Dis[Gc_AN][h_AN];
        r2 = r*r;
        r3 = r2*r; 

	g[1] = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	g[2] = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	g[3] = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;

	sr = sig[Mc_AN][h_AN]/r;
	sr2 = sr*sr;
	sr4 = sr2*sr2;
	sr5 = sr*sr4;
	sr6 = sr2*sr4;
	sr11 = sr5*sr6;
	sr12 = sr6*sr6;

	d2f_depdr = 24.0*(sr6 - 2.0*sr12)/r;
	df_dr = d2f_depdr*ep[Mc_AN][h_AN];
        d2f_dr2 = -ep[Mc_AN][h_AN]*(168.0*sr6-624.0*sr12)/r2;

        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
            d2r_dx1dx2[i][j] = -g[i]*g[j]/r;
	  }
          d2r_dx1dx2[i][i] += 1.0/r;
        }
        
        /* diagonal block */

        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
	    Hessian[(Gc_AN-1)*3+i][(Gc_AN-1)*3+j] += d2r_dx1dx2[i][j]*df_dr + g[i]*g[j]*d2f_dr2;      
	  }
	}
  
        /* off-diagonal block */

        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
	    Hessian[(Gc_AN-1)*3+i][(Gh_AN-1)*3+j] += -d2r_dx1dx2[i][j]*df_dr - g[i]*g[j]*d2f_dr2;      
	  }
	}

      } /* h_AN */
    } /* Mc_AN */       

    /* MPI Hessian */
   
    for (i=1; i<=atomnum; i++){     
      ID = G2ID[i];
      MPI_Bcast(&Hessian[(i-1)*3+1][1], 3*atomnum, MPI_DOUBLE, ID, mpi_comm_level1);
      MPI_Bcast(&Hessian[(i-1)*3+2][1], 3*atomnum, MPI_DOUBLE, ID, mpi_comm_level1);
      MPI_Bcast(&Hessian[(i-1)*3+3][1], 3*atomnum, MPI_DOUBLE, ID, mpi_comm_level1);
    }    

    /* freeing of arrays */

    for (p=0; p<(M_GDIIS_HISTORY+1); p++){
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
	free(GLJ[p][Mc_AN]);
      }
      free(GLJ[p]);
    }
    free(GLJ);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(ep[Mc_AN]);
    }
    free(ep);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(sig[Mc_AN]);
    }
    free(sig);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(dQ_dep[Mc_AN]);
    }
    free(dQ_dep);

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(dQ_dsig[Mc_AN]);
    }
    free(dQ_dsig);

  }

  else{

    int i,j;

    for (i=1; i<=3*atomnum; i++){     
      for (j=1; j<=3*atomnum; j++){     
	Hessian[i][j] = 0.0;
      }
      Hessian[i][i] = 1.0;
    }
  }


  /*
  for (i=1; i<=3*atomnum; i++){     
    for (j=1; j<=3*atomnum; j++){     
      printf("%8.5f ",Hessian[i][j]);
    }
    printf("\n");
  }

  MPI_Finalize();
  exit(0);
  */

}



/* added by MIZUHO for NPT-MD */
// macros for Gxyz array
#define POSITION(Gxyz)    (&Gxyz[ 0])
#define POSITION_X(Gxyz)  (Gxyz[ 1])
#define POSITION_Y(Gxyz)  (Gxyz[ 2])
#define POSITION_Z(Gxyz)  (Gxyz[ 3])

#define VELOCITY(Gxyz)    (&Gxyz[23])
#define VELOCITY_X(Gxyz)  (Gxyz[24])
#define VELOCITY_Y(Gxyz)  (Gxyz[25])
#define VELOCITY_Z(Gxyz)  (Gxyz[26])

#define GRADIENT(Gxyz)    (&Gxyz[16])
#define GRADIENT_X(Gxyz)  (Gxyz[17])
#define GRADIENT_Y(Gxyz)  (Gxyz[18])
#define GRADIENT_Z(Gxyz)  (Gxyz[19])

#define ATOMIC_MASS(Gxyz) (Gxyz[20]*(unified_atomic_mass_unit/electron_mass))

#define FIXED(atom_Fixed_XYZ)   (&atom_Fixed_XYZ[0])
#define FIXED_X(atom_Fixed_XYZ) (atom_Fixed_XYZ[1])
#define FIXED_Y(atom_Fixed_XYZ) (atom_Fixed_XYZ[2])
#define FIXED_Z(atom_Fixed_XYZ) (atom_Fixed_XYZ[3])

double getStress( int al, int be, double Volume )
{
  return (-1.0)*Stress_Tensor[3*(al-1)+(be-1)]/Volume;
}

double defaultMassBarostat( void )
{
  double MassAtom = 0.0;
  int ai, a;
  int numprocs, myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  for( ai=1; ai<=Matomnum; ai++ ){
    a = M2G[ai];
    MassAtom += ATOMIC_MASS(Gxyz[a]);

    //printf("ABC1 a=%2d mass=%15.12f\n",a,ATOMIC_MASS(Gxyz[a]));
  }
  MPI_Allreduce( MPI_IN_PLACE, &MassAtom, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  double MassBarostat;

  switch (MD_switch) {
  case 27: // NPT_VS_PR
  case 29: // NPT_NH_PR
    MassBarostat = 0.75*MassAtom/(M_PI*M_PI);
    break;
  case 28: // NPT_VS_WV
  case 30: // NPT_NH_WV
    MassBarostat = 0.75*MassAtom/(M_PI*M_PI)/pow(Cell_Volume,2.0/3.0)*0.01;
    break;
  default:
    printf("default MassBarostat is not supported for MD_switch %d.\n", MD_switch );
    MPI_Finalize();
    exit(1);
  }

  if (myid==Host_ID){
    printf("A default NPT.Mass.Barostat is set to %15.12f (atomic mass unit).\n",
            MassBarostat/(unified_atomic_mass_unit/electron_mass));
  }

  return MassBarostat;
}



// Velocity-Scaling and Parrinello-Rahman method
void NPT_VS_PR(int iter)
{

} // NPT_VS_PR





// Velocity-Scaling and Wentzcovitch method
void NPT_VS_WV(int iter)
{

} // NPT_VS_WV



// Nose-Hoover and Parrinello-Rahman method
void NPT_NH_PR( int iter )
{

} // NPT_NH_PR




// Nose-Hoover and Wentzcovitch method
void NPT_NH_WV( int iter )
{

} // NPT_NH_WV
