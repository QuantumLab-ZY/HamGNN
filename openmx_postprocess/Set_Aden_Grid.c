/**********************************************************************
  Set_Aden_Grid.c:

     Set_Aden_Grid.c is a subroutine to calculate a charge density 
     superposed atomic densities on grid.

  Log of Set_Aden_Grid.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>


double Set_Aden_Grid()
{
  /****************************************************
          Densities by the atomic superposition
                   densities on grids
  ****************************************************/

  static int firsttime=1;
  int i,j,k,Gc_AN,Mc_AN,NN_S,NN_R;
  int Cwan,Nc,GNc,GRc,mul,gp,spe;
  int N3[4],AN,BN,CN,DN,LN;
  unsigned long long int GN,n2D,N2D;
  int size_AtomDen_Grid;
  int size_AtomDen_Snd_Grid_A2B;
  int size_AtomDen_Rcv_Grid_A2B;
  double DenA,x,dx,dy,dz,r;
  double Nele,Nu,Nd,M,ocupcy_u,ocupcy_d;
  double rho,mag,magx,magy,magz,theta,phi;
  double time0,TStime,TEtime;
  double Stime_atom, Etime_atom;
  double Cxyz[4],Nup,Ndown;
  double **AtomDen_Snd_Grid_A2B;
  double **AtomDen_Rcv_Grid_A2B;
  double **PCCDen_Snd_Grid_A2B;
  double **PCCDen_Rcv_Grid_A2B;
  double **AtomDen_Grid;
  double **PCCDen_Grid;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  double *Work_Array_Snd_Grid_B2D;
  double *Work_Array_Rcv_Grid_B2D;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  size_AtomDen_Snd_Grid_A2B = 0; 
  AtomDen_Snd_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    AtomDen_Snd_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Snd_Grid_A2B[ID]);
    size_AtomDen_Snd_Grid_A2B += Num_Snd_Grid_A2B[ID];
  }  

  size_AtomDen_Rcv_Grid_A2B = 0;   
  AtomDen_Rcv_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    AtomDen_Rcv_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Rcv_Grid_A2B[ID]);
    size_AtomDen_Rcv_Grid_A2B += Num_Rcv_Grid_A2B[ID];   
  }

  if (PCC_switch==1){

    PCCDen_Snd_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
    for (ID=0; ID<numprocs; ID++){
      PCCDen_Snd_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Snd_Grid_A2B[ID]);
    }  
  
    PCCDen_Rcv_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
    for (ID=0; ID<numprocs; ID++){
      PCCDen_Rcv_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Rcv_Grid_A2B[ID]);
    }
  }

  size_AtomDen_Grid = 0;
  AtomDen_Grid = (double**)malloc(sizeof(double*)*(Matomnum+1)); 
  AtomDen_Grid[0] = (double*)malloc(sizeof(double)*1); 
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    AtomDen_Grid[Mc_AN] = (double*)malloc(sizeof(double)*GridN_Atom[Gc_AN]);
    size_AtomDen_Grid += GridN_Atom[Gc_AN];
  }

  if (PCC_switch==1){
    PCCDen_Grid = (double**)malloc(sizeof(double*)*(Matomnum+1)); 
    PCCDen_Grid[0] = (double*)malloc(sizeof(double)*1); 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      PCCDen_Grid[Mc_AN] = (double*)malloc(sizeof(double)*GridN_Atom[Gc_AN]); 
    }
  }

  if      (SpinP_switch<=1) mul = 2;
  else if (SpinP_switch==3) mul = 4;

  Work_Array_Snd_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_S[NN_B2C_S]*mul); 
  Work_Array_Rcv_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_R[NN_B2C_R]*mul); 
  Work_Array_Snd_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_S[NN_B2D_S]*(mul+2)); 
  Work_Array_Rcv_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_R[NN_B2D_R]*(mul+2)); 

  /* PrintMemory */

  if (firsttime==1){
    PrintMemory("Set_Aden_Grid: AtomDen_Grid", sizeof(double)*size_AtomDen_Grid, NULL);
    if (PCC_switch==1){
    PrintMemory("Set_Aden_Grid: PCCDen_Grid",  sizeof(double)*size_AtomDen_Grid, NULL);
    }
    PrintMemory("Set_Aden_Grid: AtomDen_Snd_Grid_A2B", sizeof(double)*size_AtomDen_Snd_Grid_A2B, NULL);
    PrintMemory("Set_Aden_Grid: AtomDen_Rcv_Grid_A2B", sizeof(double)*size_AtomDen_Rcv_Grid_A2B, NULL);
    if (PCC_switch==1){
    PrintMemory("Set_Aden_Grid: PCCDen_Snd_Grid_A2B", sizeof(double)*size_AtomDen_Snd_Grid_A2B, NULL);
    PrintMemory("Set_Aden_Grid: PCCDen_Rcv_Grid_A2B", sizeof(double)*size_AtomDen_Rcv_Grid_A2B, NULL);
    }

    PrintMemory("Set_Aden_Grid: Work_Array_Snd_Grid_B2C",   sizeof(double)*GP_B2C_S[NN_B2C_S]*mul, NULL);
    PrintMemory("Set_Aden_Grid: Work_Array_Rcv_Grid_B2C",   sizeof(double)*GP_B2C_R[NN_B2C_R]*mul, NULL);
    PrintMemory("Set_Aden_Grid: Work_Array_Snd_Grid_B2D",   sizeof(double)*GP_B2D_S[NN_B2D_S]*(mul+2), NULL);
    PrintMemory("Set_Aden_Grid: Work_Array_Rcv_Grid_B2D",   sizeof(double)*GP_B2D_R[NN_B2D_R]*(mul+2), NULL);
    firsttime = 0;
  }

  /******************************************************
        calculation of AtomDen_Grid and PCCDen_Grid
  ******************************************************/
 
  /* calculate AtomDen_Grid and PCCDen_Grid */

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
 
#pragma omp parallel shared(Spe_Atomic_PCC,Spe_VPS_RV,Spe_VPS_XV,Spe_Num_Mesh_VPS,Spe_PAO_RV,Spe_Atomic_Den,Spe_PAO_XV,Spe_Num_Mesh_PAO,PCCDen_Grid,PCC_switch,AtomDen_Grid,Cwan,Gxyz,atv,Mc_AN,CellListAtom,GridListAtom,GridN_Atom,Gc_AN) private(OMPID,Nthrds,Nc,GNc,GRc,Cxyz,dx,dy,dz,r,x)
    {

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();

      for (Nc=OMPID; Nc<GridN_Atom[Gc_AN]; Nc+=Nthrds){

	GNc = GridListAtom[Mc_AN][Nc];
	GRc = CellListAtom[Mc_AN][Nc];
      
	Get_Grid_XYZ(GNc,Cxyz);
	dx = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1];
	dy = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2];
	dz = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];
      
        x = 0.5*log(dx*dx + dy*dy + dz*dz);

	/* atomic density */

	AtomDen_Grid[Mc_AN][Nc] = KumoF( Spe_Num_Mesh_PAO[Cwan], x, 
 		                         Spe_PAO_XV[Cwan], Spe_PAO_RV[Cwan], Spe_Atomic_Den[Cwan]);

	/*  partial core correction */

	if (PCC_switch==1) {
	  PCCDen_Grid[Mc_AN][Nc] =  KumoF( Spe_Num_Mesh_VPS[Cwan], x, 
 			                   Spe_VPS_XV[Cwan], Spe_VPS_RV[Cwan], Spe_Atomic_PCC[Cwan]);
	}
      }

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /******************************************************
      MPI communication from the partitions A to B 
  ******************************************************/
  
  /* copy AtomDen_Grid to AtomDen_Snd_Grid_A2B */
  /* copy PCCDen_Grid to PCCDen_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = (int)(n2D*(unsigned long long int)numprocs/N2D);
      AtomDen_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = AtomDen_Grid[Mc_AN][AN];

      if (PCC_switch==1){
        PCCDen_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = PCCDen_Grid[Mc_AN][AN];
      }

      Num_Snd_Grid_A2B[ID]++;
    }
  }    

  /* MPI: A to B */  

  request_send = malloc(sizeof(MPI_Request)*NN_A2B_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_A2B_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_A2B_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_A2B_R);

  NN_S = 0;
  NN_R = 0;

  tag = 999;
  for (ID=1; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_A2B[IDS]!=0){
      MPI_Isend( &AtomDen_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS],
	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      MPI_Irecv( &AtomDen_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR],
  	         MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  /* for myid */
  for (i=0; i<Num_Rcv_Grid_A2B[myid]; i++){
    AtomDen_Rcv_Grid_A2B[myid][i] = AtomDen_Snd_Grid_A2B[myid][i];
  }
 
  /* PCC */

  if (PCC_switch==1){

    NN_S = 0;
    NN_R = 0;

    tag = 999;
    for (ID=1; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (Num_Snd_Grid_A2B[IDS]!=0){
	MPI_Isend( &PCCDen_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS],
		   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
	NN_S++;
      }

      if (Num_Rcv_Grid_A2B[IDR]!=0){
	MPI_Irecv( &PCCDen_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR],
		   MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
	NN_R++;
      }
    }

    if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
    if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

    /* for myid */
    for (i=0; i<Num_Rcv_Grid_A2B[myid]; i++){
      PCCDen_Rcv_Grid_A2B[myid][i] = PCCDen_Snd_Grid_A2B[myid][i];
    }
  }

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /******************************************************
   superposition of atomic densities and PCC densities
   in the partition B.

   for spin non-collinear
   1. set rho, mx, my, mz
   2. later calculate theta and phi 
   3. n_up = (rho+m)/2 
      n_dn = (rho-m)/2 
  ******************************************************/

  /* initialize arrays */

  if (SpinP_switch==3){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B[0][BN] = 0.0;
      Density_Grid_B[1][BN] = 0.0;
      Density_Grid_B[2][BN] = 0.0;
      Density_Grid_B[3][BN] = 0.0;
    }
  }
  else{ 
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B[0][BN] = 0.0;
      Density_Grid_B[1][BN] = 0.0;
    }
  }  

  if (PCC_switch==1){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      PCCDensity_Grid_B[0][BN] = 0.0;
      PCCDensity_Grid_B[1][BN] = 0.0;
    }
  }

  /* superposition of densities */

  for (ID=0; ID<numprocs; ID++){

    for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

      BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];      
      Gc_AN = Index_Rcv_Grid_A2B[ID][3*LN+1];        
      GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2]; 
      spe = WhatSpecies[Gc_AN];
      Nele  = InitN_USpin[Gc_AN] + InitN_DSpin[Gc_AN];
      Nu    = InitN_USpin[Gc_AN];
      Nd    = InitN_DSpin[Gc_AN];

      DenA = AtomDen_Rcv_Grid_A2B[ID][LN];

      if (1.0e-15<Nele){
	ocupcy_u = Nu/Nele;
	ocupcy_d = Nd/Nele;
      }
      else{
	ocupcy_u = 0.0;
	ocupcy_d = 0.0;
      }

      if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){

	/* spin collinear non-polarization */
	if ( SpinP_switch==0 ){
	  Density_Grid_B[0][BN] += 0.5*DenA;
	  Density_Grid_B[1][BN] += 0.5*DenA;
	}

	/* spin collinear polarization */
	else if ( SpinP_switch==1 ){
	  Density_Grid_B[0][BN] += ocupcy_u*DenA;
	  Density_Grid_B[1][BN] += ocupcy_d*DenA;
	} 

	/* spin non-collinear */
	else if ( SpinP_switch==3 ){

	  theta = Angle0_Spin[Gc_AN];
	  phi   = Angle1_Spin[Gc_AN];

	  rho = DenA;
	  mag = (ocupcy_u - ocupcy_d)*DenA;
	  magx = mag*sin(theta)*cos(phi);
	  magy = mag*sin(theta)*sin(phi);
	  magz = mag*cos(theta);

	  Density_Grid_B[0][BN] += rho;
	  Density_Grid_B[1][BN] += magx;
	  Density_Grid_B[2][BN] += magy;
	  Density_Grid_B[3][BN] += magz;
	} 

	/* partial core correction       */
	/* later add this in Set_XC_Grid */

	if (PCC_switch==1){
  	  if ( SpinP_switch==0 ){
 	    PCCDensity_Grid_B[0][BN] += 0.5*PCCDen_Rcv_Grid_A2B[ID][LN];
	    PCCDensity_Grid_B[1][BN] += 0.5*PCCDen_Rcv_Grid_A2B[ID][LN];
	  }
          else {

            if (Spe_OpenCore_flag[spe]==0){
 	      PCCDensity_Grid_B[0][BN] += 0.5*PCCDen_Rcv_Grid_A2B[ID][LN];
	      PCCDensity_Grid_B[1][BN] += 0.5*PCCDen_Rcv_Grid_A2B[ID][LN];
	    }
            else if (Spe_OpenCore_flag[spe]==1){
 	      PCCDensity_Grid_B[0][BN] += PCCDen_Rcv_Grid_A2B[ID][LN];
	    }
            else if (Spe_OpenCore_flag[spe]==-1){
 	      PCCDensity_Grid_B[1][BN] += PCCDen_Rcv_Grid_A2B[ID][LN];
            }
          }
	}

      } /* if (Solve!=4.....) */           

    } /* AN */ 
  } /* ID */  

  /****************************************************
     initialize diagonal and off-diagonal densities
           in case of spin non-collinear DFT
  ****************************************************/

  if (SpinP_switch==3){
    for (BN=0; BN<My_NumGridB_AB; BN++){

      rho  = Density_Grid_B[0][BN];
      magx = Density_Grid_B[1][BN];
      magy = Density_Grid_B[2][BN];
      magz = Density_Grid_B[3][BN];

      Density_Grid_B[0][BN] = 0.5*(rho + magz);
      Density_Grid_B[1][BN] = 0.5*(rho - magz);
      Density_Grid_B[2][BN] = 0.5*magx;
      Density_Grid_B[3][BN] =-0.5*magy;
    }
  }

  /******************************************************
            Density_Grid_B to ADensity_Grid_B
  ******************************************************/

  if ( SpinP_switch==0 ){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      ADensity_Grid_B[BN] = Density_Grid_B[0][BN];
    }
  } 
  else if ( SpinP_switch==1 || SpinP_switch==3 ){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      ADensity_Grid_B[BN] = 0.5*(Density_Grid_B[0][BN] + Density_Grid_B[1][BN]);
    }
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2C_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2C_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2C_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2C_R);

  NN_S = 0;
  NN_R = 0;

  if      (SpinP_switch<=1) mul = 2;
  else if (SpinP_switch==3) mul = 4;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[mul*gp], Num_Rcv_Grid_B2C[IDR]*mul,
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2C_S; ID++){

    IDS = ID_NN_B2C_S[ID];
    gp = GP_B2C_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2C */

    for (LN=0; LN<Num_Snd_Grid_B2C[IDS]; LN++){

      BN = Index_Snd_Grid_B2C[IDS][LN];

      if (SpinP_switch<=1){
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+0] = Density_Grid_B[0][BN];
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+1] = Density_Grid_B[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+0] = Density_Grid_B[0][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+1] = Density_Grid_B[1][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+2] = Density_Grid_B[2][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+3] = Density_Grid_B[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[mul*gp], Num_Snd_Grid_B2C[IDS]*mul,
	          MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Snd_Grid_B2C to Density_Grid */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){

      gp = GP_B2C_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[myid]; LN++){

	CN = Index_Rcv_Grid_B2C[myid][LN];

	if (SpinP_switch<=1){
	  Density_Grid[0][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+0];
	  Density_Grid[1][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid[0][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+0];
	  Density_Grid[1][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+1];
	  Density_Grid[2][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+2];
	  Density_Grid[3][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+3];
	}
      } /* LN */   
    }

    else{

      gp = GP_B2C_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){

	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch<=1){
	  Density_Grid[0][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+0];
	  Density_Grid[1][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid[0][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+0];
	  Density_Grid[1][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+1];
	  Density_Grid[2][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+2];
	  Density_Grid[3][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+3];
	}
      }
    }
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  request_send = malloc(sizeof(MPI_Request)*NN_B2D_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B2D_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B2D_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B2D_R);

  NN_S = 0;
  NN_R = 0;

  if      (SpinP_switch<=1) mul = 4;
  else if (SpinP_switch==3) mul = 6;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];
    gp = GP_B2D_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2D[mul*gp], Num_Rcv_Grid_B2D[IDR]*mul,
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2D_S; ID++){

    IDS = ID_NN_B2D_S[ID];
    gp = GP_B2D_S[ID];

    /* copy Density_Grid_B to Work_Array_Snd_Grid_B2D */

    for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){

      BN = Index_Snd_Grid_B2D[IDS][LN];

      if (SpinP_switch<=1){
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+0] = Density_Grid_B[0][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+1] = Density_Grid_B[1][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+2] = PCCDensity_Grid_B[0][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+3] = PCCDensity_Grid_B[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2D[6*gp+6*LN+0] = Density_Grid_B[0][BN];
        Work_Array_Snd_Grid_B2D[6*gp+6*LN+1] = Density_Grid_B[1][BN];
        Work_Array_Snd_Grid_B2D[6*gp+6*LN+2] = Density_Grid_B[2][BN];
        Work_Array_Snd_Grid_B2D[6*gp+6*LN+3] = Density_Grid_B[3][BN];
        Work_Array_Snd_Grid_B2D[6*gp+6*LN+4] = PCCDensity_Grid_B[0][BN];
        Work_Array_Snd_Grid_B2D[6*gp+6*LN+5] = PCCDensity_Grid_B[1][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2D[mul*gp], Num_Snd_Grid_B2D[IDS]*mul,
	          MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Snd_Grid_B2D to Density_Grid */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];

    if (IDR==myid){

      gp = GP_B2D_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[myid]; LN++){

	DN = Index_Rcv_Grid_B2D[myid][LN];

	if (SpinP_switch<=1){
	  Density_Grid_D[0][DN]    = Work_Array_Snd_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN]    = Work_Array_Snd_Grid_B2D[4*gp+4*LN+1];
	  PCCDensity_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+2];
	  PCCDensity_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+3];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN]    = Work_Array_Snd_Grid_B2D[6*gp+6*LN+0];
	  Density_Grid_D[1][DN]    = Work_Array_Snd_Grid_B2D[6*gp+6*LN+1];
	  Density_Grid_D[2][DN]    = Work_Array_Snd_Grid_B2D[6*gp+6*LN+2];
	  Density_Grid_D[3][DN]    = Work_Array_Snd_Grid_B2D[6*gp+6*LN+3];
	  PCCDensity_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[6*gp+6*LN+4];
	  PCCDensity_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[6*gp+6*LN+5];
	}
      }
    }

    else {

      gp = GP_B2D_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch<=1){
	  Density_Grid_D[0][DN]    = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN]    = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+1];
	  PCCDensity_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+2];
	  PCCDensity_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+3];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN]    = Work_Array_Rcv_Grid_B2D[6*gp+6*LN+0];
	  Density_Grid_D[1][DN]    = Work_Array_Rcv_Grid_B2D[6*gp+6*LN+1];
	  Density_Grid_D[2][DN]    = Work_Array_Rcv_Grid_B2D[6*gp+6*LN+2];
	  Density_Grid_D[3][DN]    = Work_Array_Rcv_Grid_B2D[6*gp+6*LN+3];
	  PCCDensity_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[6*gp+6*LN+4];
	  PCCDensity_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[6*gp+6*LN+5];
	}
      }
    }
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  for (ID=0; ID<numprocs; ID++){
    free(AtomDen_Snd_Grid_A2B[ID]);
  }  
  free(AtomDen_Snd_Grid_A2B);

  for (ID=0; ID<numprocs; ID++){
    free(AtomDen_Rcv_Grid_A2B[ID]);
  }
  free(AtomDen_Rcv_Grid_A2B);

  if (PCC_switch==1){

    for (ID=0; ID<numprocs; ID++){
      free(PCCDen_Snd_Grid_A2B[ID]);
    }  
    free(PCCDen_Snd_Grid_A2B);
  
    for (ID=0; ID<numprocs; ID++){
      free(PCCDen_Rcv_Grid_A2B[ID]);
    }
    free(PCCDen_Rcv_Grid_A2B);
  }

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(AtomDen_Grid[Mc_AN]);
  }
  free(AtomDen_Grid);

  if (PCC_switch==1){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(PCCDen_Grid[Mc_AN]);
    }
    free(PCCDen_Grid);
  }

  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
  free(Work_Array_Snd_Grid_B2D);
  free(Work_Array_Rcv_Grid_B2D);

  /* elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}
