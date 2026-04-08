/**********************************************************************
  Set_Vpot.c:

    Set_Vpot.c is a subroutine to calculate the value of local potential
    on each grid point.

  Log of Set_Vpot.c:

     22/Nov/2001  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>


static void Make_VNA_Grid();


void Set_Vpot(int MD_iter,
              int SCF_iter, 
              int SCF_iter0,
              int TRAN_Poisson_flag2,
              int XC_P_switch)
{
  /****************************************************
        XC_P_switch:
            0  \epsilon_XC (XC energy density)  
            1  \mu_XC      (XC potential)  
            2  \epsilon_XC - \mu_XC
  ****************************************************/

  int i,j,k,n,nmax,ri,Mc_AN,Gc_AN,Rn,GNc,GRc;
  int Nc,Nd,n1,n2,n3,Cwan,spin,MN,BN,DN,ct_AN;
  int N2D,GNs,BN_AB,GN_AB;
  int h_AN,Gh_AN,Hwan,Rnh,N3[4];
  int hNgrid1,hNgrid2,hNgrid3,Ng1,Ng2,Ng3;
  double Gx,Gy,Gz,sum,x,y,z;
  double Cxyz[4],dx,dy,dz,r,xc,yc,zc;
  int numprocs,myid;
  double Stime_atom,Etime_atom;
  double time15,time16; 

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
                       Vxc on grid
  ****************************************************/

  Set_XC_Grid(SCF_iter,XC_P_switch,XC_switch,
              Density_Grid_D[0],Density_Grid_D[1],
              Density_Grid_D[2],Density_Grid_D[3],
              Vxc_Grid_D[0], Vxc_Grid_D[1],
	      Vxc_Grid_D[2], Vxc_Grid_D[3],
              NULL,NULL);

  /****************************************************
             copy Vxc_Grid_D to Vxc_Grid_B
  ****************************************************/

  Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
  Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
  Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

  for (n=0; n<Num_Rcv_Grid_B2D[myid]; n++){
    DN = Index_Rcv_Grid_B2D[myid][n];
    BN = Index_Snd_Grid_B2D[myid][n];

    i = DN/(Ng2*Ng3);
    j = (DN-i*Ng2*Ng3)/Ng3;
    k = DN - i*Ng2*Ng3 - j*Ng3; 

    if ( !(i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k)){
      for (spin=0; spin<=SpinP_switch; spin++){
        Vxc_Grid_B[spin][BN] = Vxc_Grid_D[spin][DN];
      }
    }
  }

  /****************************************************
          The neutral atom potential on grids
  ****************************************************/

  if (SCF_iter<=2 && ProExpn_VNA==0) Make_VNA_Grid();

  /****************************************************
                external electric field
  ****************************************************/

  if ( SCF_iter<=2 && E_Field_switch==1 ){

    /* the center of the system */

    xc = 0.0;
    yc = 0.0;
    zc = 0.0;

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      xc += Gxyz[ct_AN][1];
      yc += Gxyz[ct_AN][2];
      zc += Gxyz[ct_AN][3];
    }

    xc = xc/(double)atomnum;
    yc = yc/(double)atomnum;
    zc = zc/(double)atomnum;

    hNgrid1 = Ngrid1/2;
    hNgrid2 = Ngrid2/2;
    hNgrid3 = Ngrid3/2;

    /* calculate VEF_Grid in the partition B */

    N2D = Ngrid1*Ngrid2;
    GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
       
      GN_AB = BN_AB + GNs;
      i = GN_AB/(Ngrid2*Ngrid3);
      j = (GN_AB - i*(Ngrid2*Ngrid3))/Ngrid3;
      k = GN_AB - i*(Ngrid2*Ngrid3) - j*Ngrid3;

      Find_CGrids(1,i,j,k,Cxyz,N3);
      i = N3[1];
      j = N3[2];
      k = N3[3];

      dx = (double)i*length_gtv[1] + Grid_Origin[1] - xc;
      dy = (double)j*length_gtv[2] + Grid_Origin[2] - yc;
      dz = (double)k*length_gtv[3] + Grid_Origin[3] - zc;
      VEF_Grid_B[BN_AB] = dx*E_Field[0] + dy*E_Field[1] + dz*E_Field[2];
    } 

    /* MPI: from the partitions B to C */
    Data_Grid_Copy_B2C_1( VEF_Grid_B, VEF_Grid );
 
  } /* if ( SCF_iter<=2 && E_Field_switch==1 ) */

  /****************************************************
                         Sum
  ****************************************************/

  /* spin non-collinear */
  if (SpinP_switch==3){  

    /******************************
     diagonal part 
     spin=0:  11
     spin=1:  22
    ******************************/

    if ( E_Field_switch==1 ){

      if (ProExpn_VNA==0){
        for (spin=0; spin<=1; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN]
                                  + F_VNA_flag*VNA_Grid_B[MN]
                                  + F_VEF_flag*VEF_Grid_B[MN];
          }
        }
      }
      else{
        for (spin=0; spin<=1; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN]
                                  + F_VEF_flag*VEF_Grid_B[MN];
          }
        }
      }

    }

    else{  

      if (ProExpn_VNA==0){
        for (spin=0; spin<=1; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN] 
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN]
                                  + F_VNA_flag*VNA_Grid_B[MN];
          }
        }
      }
      else{

        for (spin=0; spin<=1; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN];
          }
        }
      }
    }

    /******************************
     off-diagonal part 
     spin=2:  real 12
     spin=3:  imaginary 12
    ******************************/

    for (spin=2; spin<=3; spin++){
      for (MN=0; MN<My_NumGridB_AB; MN++){
        Vpot_Grid_B[spin][MN] = F_Vxc_flag*Vxc_Grid_B[spin][MN];
      }
    }
  }

  /* spin collinear */
  else{

    if ( E_Field_switch==1 ){

      if (ProExpn_VNA==0){
        for (spin=0; spin<=SpinP_switch; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN]
                                  + F_VNA_flag*VNA_Grid_B[MN]
                                  + F_VEF_flag*VEF_Grid_B[MN];
          }
        }
      }
      else{
        for (spin=0; spin<=SpinP_switch; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN]
                                  + F_VEF_flag*VEF_Grid_B[MN];
          }
        }
      }

    }
    else{  
     
      if (ProExpn_VNA==0){
        for (spin=0; spin<=SpinP_switch; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN]
                                  + F_VNA_flag*VNA_Grid_B[MN];
          }
        }
      }
      else{
        for (spin=0; spin<=SpinP_switch; spin++){
          for (MN=0; MN<My_NumGridB_AB; MN++){
            Vpot_Grid_B[spin][MN] = F_dVHart_flag*dVHart_Grid_B[MN]
                                  + F_Vxc_flag*Vxc_Grid_B[spin][MN];
          }
        }
      }
    }
  }

  /*****************************************************
    RMM-DIISV:
    if (Mixing_switch==7), 
    mixing of potential is performed here.
  *****************************************************/

  if (Mixing_switch==7){

    time15 = 0.0;

    /********************************************* 
     FFT of VKS, where FFT_Density is used for FFT 
    *********************************************/

    /* non-spin polarization */
    if (SpinP_switch==0){
      time15 += FFT_Density(6,ReVKSk[0][0],ImVKSk[0][0]);
    }

    /* collinear spin polarization */
    else if (SpinP_switch==1) {
      time15 += FFT_Density(6,ReVKSk[0][0],ImVKSk[0][0]);
      time15 += FFT_Density(7,ReVKSk[0][1],ImVKSk[0][1]);
    }

    /* non-collinear spin polarization */
    else if (SpinP_switch==3) {
      time15 += FFT_Density(6,ReVKSk[0][0],ImVKSk[0][0]);
      time15 += FFT_Density(7,ReVKSk[0][1],ImVKSk[0][1]);
      time15 += FFT_Density(8,ReVKSk[0][2],ImVKSk[0][2]);
    }

    /********************************************* 
                     call Mixing_V
    *********************************************/

    time16 = Mixing_V(MD_iter,SCF_iter,SCF_iter0);
  }

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  Data_Grid_Copy_B2C_2( Vpot_Grid_B, Vpot_Grid );
}



void Make_VNA_Grid()
{
  static int firsttime=1;
  unsigned long long int n2D,N2D,GNc,GN;
  int i,Mc_AN,Gc_AN,BN,CN,LN,GRc,N3[4];
  int AN,Nc,MN,Cwan,NN_S,NN_R;
  int size_AtomVNA_Grid;
  int size_AtomVNA_Snd_Grid_A2B;
  int size_AtomVNA_Rcv_Grid_A2B;
  double Cxyz[4];
  double r,dx,dy,dz;
  double **AtomVNA_Grid;
  double **AtomVNA_Snd_Grid_A2B;
  double **AtomVNA_Rcv_Grid_A2B;
  double Stime_atom, Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  int OMPID,Nthrds,Nprocs;
  
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  /* allocation of arrays */
  
  size_AtomVNA_Grid = 1;
  AtomVNA_Grid = (double**)malloc(sizeof(double*)*(Matomnum+1)); 
  AtomVNA_Grid[0] = (double*)malloc(sizeof(double)*1); 
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    AtomVNA_Grid[Mc_AN] = (double*)malloc(sizeof(double)*GridN_Atom[Gc_AN]);
    size_AtomVNA_Grid += GridN_Atom[Gc_AN];
  }
  
  size_AtomVNA_Snd_Grid_A2B = 0; 
  AtomVNA_Snd_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    AtomVNA_Snd_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Snd_Grid_A2B[ID]);
    size_AtomVNA_Snd_Grid_A2B += Num_Snd_Grid_A2B[ID];
  }  
  
  size_AtomVNA_Rcv_Grid_A2B = 0;   
  AtomVNA_Rcv_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    AtomVNA_Rcv_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Rcv_Grid_A2B[ID]);
    size_AtomVNA_Rcv_Grid_A2B += Num_Rcv_Grid_A2B[ID];   
  }

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_Vpot: AtomVNA_Grid",sizeof(double)*size_AtomVNA_Grid,NULL);
    PrintMemory("Set_Vpot: AtomVNA_Snd_Grid_A2B",sizeof(double)*size_AtomVNA_Snd_Grid_A2B,NULL);
    PrintMemory("Set_Vpot: AtomVNA_Rcv_Grid_A2B",sizeof(double)*size_AtomVNA_Rcv_Grid_A2B,NULL);
  }

  /* calculation of AtomVNA_Grid */

  for (MN=0; MN<My_NumGridC; MN++) VNA_Grid[MN] = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

#pragma omp parallel shared(AtomVNA_Grid,GridN_Atom,atv,Gxyz,Gc_AN,Cwan,Mc_AN,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,dx,dy,dz,r)
    {

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

	GNc = GridListAtom[Mc_AN][Nc];
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	dx = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1];
	dy = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2];
	dz = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	r = sqrt(dx*dx + dy*dy + dz*dz);
	AtomVNA_Grid[Mc_AN][Nc] = VNAF(Cwan,r);
      }

#pragma omp flush(AtomVNA_Grid)

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

  } /* Mc_AN */

  /******************************************************
    MPI communication from the partitions A to B 
  ******************************************************/
  
  /* copy AtomVNA_Grid to AtomVNA_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = (int)(n2D*(unsigned long long int)numprocs/N2D);
      AtomVNA_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = AtomVNA_Grid[Mc_AN][AN];

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
      MPI_Isend(&AtomVNA_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS], MPI_DOUBLE,
		IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      MPI_Irecv( &AtomVNA_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR],
  	         MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* for myid */
  for (i=0; i<Num_Rcv_Grid_A2B[myid]; i++){
    AtomVNA_Rcv_Grid_A2B[myid][i] = AtomVNA_Snd_Grid_A2B[myid][i];
  }

  /******************************************************
           superposition of VNA in the partition B
  ******************************************************/

  /* initialize VNA_Grid_B */

  for (BN=0; BN<My_NumGridB_AB; BN++) VNA_Grid_B[BN] = 0.0;

  /* superposition of VNA */

  for (ID=0; ID<numprocs; ID++){
    for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

      BN = Index_Rcv_Grid_A2B[ID][3*LN+0];      
      VNA_Grid_B[BN] += AtomVNA_Rcv_Grid_A2B[ID][LN];

    } /* LN */ 
  } /* ID */  

  /******************************************************
           MPI: from the partitions B to C
  ******************************************************/

  Data_Grid_Copy_B2C_1( VNA_Grid_B, VNA_Grid );

  /* freeing of arrays */

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    free(AtomVNA_Grid[Mc_AN]);
  }
  free(AtomVNA_Grid);

  for (ID=0; ID<numprocs; ID++){
    free(AtomVNA_Snd_Grid_A2B[ID]);
  }  
  free(AtomVNA_Snd_Grid_A2B);

  for (ID=0; ID<numprocs; ID++){
    free(AtomVNA_Rcv_Grid_A2B[ID]);
  }
  free(AtomVNA_Rcv_Grid_A2B);
}
