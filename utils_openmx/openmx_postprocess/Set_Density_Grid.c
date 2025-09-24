/**********************************************************************
  Set_Density_Grid.c:

     Set_Density_Grid.c is a subroutine to calculate a charge density 
     on grid by one-particle wave functions.

  Log of Set_Density_Grid.c:

     22/Nov/2001  Released by T.Ozaki
     19/Apr/2013  Modified by A.M.Ito     

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0



double Set_Density_Grid(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM, double **Density_Grid_B0)
{
  static int firsttime=1;
  int al,L0,Mul0,M0,p,size1,size2;
  int Gc_AN,Mc_AN,Mh_AN,LN,AN,BN,CN;
  int n1,n2,n3,k1,k2,k3,N3[4];
  int Cwan,NO0,NO1,Rn,N,Hwan,i,j,k,n;
  int NN_S,NN_R;
  unsigned long long int N2D,n2D,GN; 
  int Max_Size,My_Max;
  int size_Tmp_Den_Grid;
  int size_Den_Snd_Grid_A2B;
  int size_Den_Rcv_Grid_A2B;
  int h_AN,Gh_AN,Rnh,spin,Nc,GRc,Nh,Nog;
  int Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3;

  double threshold;
  double tmp0,tmp1,sk1,sk2,sk3,tot_den,sum;
  double tmp0_0,tmp0_1,tmp0_2,tmp0_3;
  double sum_0,sum_1,sum_2,sum_3;
  double d1,d2,d3,cop,sip,sit,cot;
  double x,y,z,Cxyz[4];
  double TStime,TEtime;
  double ***Tmp_Den_Grid;
  double **Den_Snd_Grid_A2B;
  double **Den_Rcv_Grid_A2B;
  double *tmp_array;
  double *tmp_array2;
  double *orbs0,*orbs1;
  double *orbs0_0,*orbs0_1,*orbs0_2,*orbs0_3;
  double *orbs1_0,*orbs1_1,*orbs1_2,*orbs1_3;
  double ***tmp_CDM;
  int *Snd_Size,*Rcv_Size;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;
  double time0,time1,time2;

  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* for OpenMP */
  int OMPID,Nthrds;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /* allocation of arrays */

  size_Tmp_Den_Grid = 0;
  Tmp_Den_Grid = (double***)malloc(sizeof(double**)*(SpinP_switch+1)); 
  for (i=0; i<(SpinP_switch+1); i++){
    Tmp_Den_Grid[i] = (double**)malloc(sizeof(double*)*(Matomnum+1)); 
    Tmp_Den_Grid[i][0] = (double*)malloc(sizeof(double)*1); 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      Tmp_Den_Grid[i][Mc_AN] = (double*)malloc(sizeof(double)*GridN_Atom[Gc_AN]);
	  
      /* AITUNE */
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	Tmp_Den_Grid[i][Mc_AN][Nc] = 0.0;
      }
      
      size_Tmp_Den_Grid += GridN_Atom[Gc_AN];
    }
  }

  size_Den_Snd_Grid_A2B = 0; 
  Den_Snd_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Den_Snd_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Snd_Grid_A2B[ID]*(SpinP_switch+1));
    size_Den_Snd_Grid_A2B += Num_Snd_Grid_A2B[ID]*(SpinP_switch+1);
  }  

  size_Den_Rcv_Grid_A2B = 0;   
  Den_Rcv_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Den_Rcv_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Rcv_Grid_A2B[ID]*(SpinP_switch+1));
    size_Den_Rcv_Grid_A2B += Num_Rcv_Grid_A2B[ID]*(SpinP_switch+1);   
  }

  /* PrintMemory */

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: AtomDen_Grid",    sizeof(double)*size_Tmp_Den_Grid, NULL);
    PrintMemory("Set_Density_Grid: Den_Snd_Grid_A2B",sizeof(double)*size_Den_Snd_Grid_A2B, NULL);
    PrintMemory("Set_Density_Grid: Den_Rcv_Grid_A2B",sizeof(double)*size_Den_Rcv_Grid_A2B, NULL);
    firsttime = 0;
  }

  /****************************************************
                when orbital optimization
  ****************************************************/

  if (Calc_CntOrbital_ON==1 && Cnt_kind==0 && Cnt_switch==1){
      
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
       
      dtime(&Stime_atom);
      
      /* COrbs_Grid */
 
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan]; 
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        al = -1;
	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_CBasis[Cwan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){

	      al++;
	      tmp0 = 0.0;

	      for (p=0; p<Spe_Specified_Num[Cwan][al]; p++){
	        j = Spe_Trans_Orbital[Cwan][al][p];  
	        tmp0 += CntCoes[Mc_AN][al][p]*Orbs_Grid[Mc_AN][Nc][j];/* AITUNE */
	      }

	      COrbs_Grid[Mc_AN][al][Nc] = (Type_Orbs_Grid)tmp0;
	    }
	  }
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }

    /**********************************************
     MPI:

     COrbs_Grid    
    ***********************************************/

    /* allocation of arrays  */
    Snd_Size = (int*)malloc(sizeof(int)*numprocs); 
    Rcv_Size = (int*)malloc(sizeof(int)*numprocs); 

    /* find data size for sending and receiving */

    My_Max = -10000;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){
        /*  sending size */
        if (F_Snd_Num[IDS]!=0){
          /* find data size */ 
          size1 = 0; 
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            size1 += GridN_Atom[Gc_AN]*Spe_Total_CNO[Cwan];
          }

          Snd_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else{
          Snd_Size[IDS] = 0;
        }

        /* receiving size */
        if (F_Rcv_Num[IDR]!=0){
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_Size[IDR] = size2;
        }
        else{
          Rcv_Size[IDR] = 0;
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      } 
      else{
        Snd_Size[IDS] = 0;
        Rcv_Size[IDR] = 0;
      }

      if (My_Max<Snd_Size[IDS]) My_Max = Snd_Size[IDS];
      if (My_Max<Rcv_Size[IDR]) My_Max = Rcv_Size[IDR];

    }  

    MPI_Allreduce(&My_Max, &Max_Size, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    /* allocation of arrays */ 
    tmp_array  = (double*)malloc(sizeof(double)*Max_Size);
    tmp_array2 = (double*)malloc(sizeof(double)*Max_Size);

    /* send and receive COrbs_Grid */

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID!=0){

        /* sending of data */ 

        if (F_Snd_Num[IDS]!=0){

          /* find data size */
          size1 = Snd_Size[IDS];

          /* multidimentional array to vector array */
          k = 0; 
          for (n=0; n<F_Snd_Num[IDS]; n++){
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan]; 
            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                tmp_array[k] = COrbs_Grid[Mc_AN][i][Nc];
                k++;
              }          
            }
          } 

          /* MPI_Isend */
          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS,
                    tag, mpi_comm_level1, &request);
        }

        /* receiving of block data */

        if (F_Rcv_Num[IDR]!=0){

          /* find data size */
          size2 = Rcv_Size[IDR]; 

          /* MPI_Recv */
          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          k = 0;
          Mc_AN = F_TopMAN[IDR] - 1;
          for (n=0; n<F_Rcv_Num[IDR]; n++){
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            NO0 = Spe_Total_CNO[Cwan]; 

            for (i=0; i<NO0; i++){
              for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
                COrbs_Grid[Mc_AN][i][Nc] = tmp_array2[k];
                k++;
              }          
            }
          }
        }
        if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);
      } 
    }  

    /* freeing of arrays  */
    free(tmp_array);
    free(tmp_array2);
    free(Snd_Size);
    free(Rcv_Size);
  }

  /**********************************************
              calculate Tmp_Den_Grid
  ***********************************************/
    
  dtime(&time1);
  
  
  /* AITUNE ========================== */ 
  int OneD_Nloop = 0;
  int ai_MaxNc = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    int Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
      if(ai_MaxNc < GridN_Atom[Gc_AN]) {ai_MaxNc = GridN_Atom[Gc_AN];}
    }
  }  
  /* ai_MaxNc is maximum of GridN_Atom[] */

  int gNthrds;
#pragma omp parallel
  {
    gNthrds = omp_get_num_threads();
  }

  double*** ai_tmpDG_all = (double***)malloc(sizeof(double**)*gNthrds);
	
  /* ========================== AITUNE */ 

#pragma omp parallel shared(myid,G2ID,Orbs_Grid_FNAN,List_YOUSO,time_per_atom,Tmp_Den_Grid,Orbs_Grid,COrbs_Grid,Cnt_switch,Cnt_kind,GListTAtoms2,GListTAtoms1,NumOLG,CDM,SpinP_switch,WhatSpecies,ncn,F_G2M,natn,Spe_Total_CNO,M2G) private(OMPID,Nthrds,Mc_AN,h_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,NO0,Gh_AN,Mh_AN,Rnh,Hwan,NO1,spin,i,j,tmp_CDM,Nog,Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3,orbs0_0,orbs0_1,orbs0_2,orbs0_3,orbs1_0,orbs1_1,orbs1_2,orbs1_3,sum_0,sum_1,sum_2,sum_3,tmp0_0,tmp0_1,tmp0_2,tmp0_3,Nc,Nh,orbs0,orbs1,sum,tmp0)
  {

    orbs0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    orbs0_0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs0_1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs0_2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs0_3 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_3 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    tmp_CDM = (double***)malloc(sizeof(double**)*(SpinP_switch+1)); 
    for (i=0; i<(SpinP_switch+1); i++){
      tmp_CDM[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
      for (j=0; j<List_YOUSO[7]; j++){
	tmp_CDM[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]); 
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

	
    /* AITUNE ========================== */  


    double *ai_tmpDGs[4];
    {
      int spin;
      for (spin=0; spin<=SpinP_switch; spin++){
	ai_tmpDGs[spin] = (double*)malloc(sizeof(double)* ai_MaxNc);
      }
    }
    ai_tmpDG_all[OMPID] = ai_tmpDGs;
    /* ==================================== AITUNE */


    /* for (Mc_AN=(OMPID+1); Mc_AN<=Matomnum; Mc_AN+=Nthrds){ AITUNE */
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan]; 
	  
      int spin;
      for (spin=0; spin<=SpinP_switch; spin++){
	int Nc;
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	  ai_tmpDGs[spin][Nc] = 0.0;
	}
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	/* set data on h_AN */
    
	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* store CDM into tmp_CDM */

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=0; i<NO0; i++){
	    for (j=0; j<NO1; j++){
	      tmp_CDM[spin][i][j] = CDM[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	/* summation of non-zero elements */
	/* for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){ */
#pragma omp for
	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]-3; Nog+=4){

	  Nc_0 = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nc_1 = GListTAtoms1[Mc_AN][h_AN][Nog+1];
	  Nc_2 = GListTAtoms1[Mc_AN][h_AN][Nog+2];
	  Nc_3 = GListTAtoms1[Mc_AN][h_AN][Nog+3];
	  
	  Nh_0 = GListTAtoms2[Mc_AN][h_AN][Nog];
	  Nh_1 = GListTAtoms2[Mc_AN][h_AN][Nog+1];
	  Nh_2 = GListTAtoms2[Mc_AN][h_AN][Nog+2];
	  Nh_3 = GListTAtoms2[Mc_AN][h_AN][Nog+3];
	  
	  /* Now under the orbital optimization */
	  if (Cnt_kind==0 && Cnt_switch==1){
	    for (i=0; i<NO0; i++){
	      orbs0_0[i] = COrbs_Grid[Mc_AN][i][Nc_0];
	      orbs0_1[i] = COrbs_Grid[Mc_AN][i][Nc_1];
	      orbs0_2[i] = COrbs_Grid[Mc_AN][i][Nc_2];
	      orbs0_3[i] = COrbs_Grid[Mc_AN][i][Nc_3];
	    }
	    for (j=0; j<NO1; j++){
	      orbs1_0[j] = COrbs_Grid[Mh_AN][j][Nh_0];
	      orbs1_1[j] = COrbs_Grid[Mh_AN][j][Nh_1];
	      orbs1_2[j] = COrbs_Grid[Mh_AN][j][Nh_2];
	      orbs1_3[j] = COrbs_Grid[Mh_AN][j][Nh_3];
	    }
	  }
	  /* else if ! "now under the orbital optimization" */
	  else{
	    for (i=0; i<NO0; i++){
	      orbs0_0[i] = Orbs_Grid[Mc_AN][Nc_0][i];
	      orbs0_1[i] = Orbs_Grid[Mc_AN][Nc_1][i];
	      orbs0_2[i] = Orbs_Grid[Mc_AN][Nc_2][i];
	      orbs0_3[i] = Orbs_Grid[Mc_AN][Nc_3][i]; 
	    }

            if (G2ID[Gh_AN]==myid){
	      for (j=0; j<NO1; j++){
		orbs1_0[j] = Orbs_Grid[Mh_AN][Nh_0][j];
		orbs1_1[j] = Orbs_Grid[Mh_AN][Nh_1][j];
		orbs1_2[j] = Orbs_Grid[Mh_AN][Nh_2][j];
		orbs1_3[j] = Orbs_Grid[Mh_AN][Nh_3][j]; 
	      }
	    }
            else{
	      for (j=0; j<NO1; j++){
		orbs1_0[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog  ][j];
		orbs1_1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+1][j];
		orbs1_2[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+2][j];
		orbs1_3[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+3][j]; 
	      }
	    }
	  }
	  
	  for (spin=0; spin<=SpinP_switch; spin++){

	    /* Tmp_Den_Grid */

	    sum_0 = 0.0;
	    sum_1 = 0.0;
	    sum_2 = 0.0;
	    sum_3 = 0.0;

	    for (i=0; i<NO0; i++){

	      tmp0_0 = 0.0;
	      tmp0_1 = 0.0;
	      tmp0_2 = 0.0;
	      tmp0_3 = 0.0;

	      for (j=0; j<NO1; j++){
		tmp0_0 += orbs1_0[j]*tmp_CDM[spin][i][j];
		tmp0_1 += orbs1_1[j]*tmp_CDM[spin][i][j];
		tmp0_2 += orbs1_2[j]*tmp_CDM[spin][i][j];
		tmp0_3 += orbs1_3[j]*tmp_CDM[spin][i][j];
	      }

	      sum_0 += orbs0_0[i]*tmp0_0;
	      sum_1 += orbs0_1[i]*tmp0_1;
	      sum_2 += orbs0_2[i]*tmp0_2;
	      sum_3 += orbs0_3[i]*tmp0_3;
	    }
		
	    ai_tmpDGs[spin][Nc_0] += sum_0;
	    ai_tmpDGs[spin][Nc_1] += sum_1;
	    ai_tmpDGs[spin][Nc_2] += sum_2;
	    ai_tmpDGs[spin][Nc_3] += sum_3;

	    /*
	    Tmp_Den_Grid[spin][Mc_AN][Nc_0] += sum_0;
	    Tmp_Den_Grid[spin][Mc_AN][Nc_1] += sum_1;
	    Tmp_Den_Grid[spin][Mc_AN][Nc_2] += sum_2;
	    Tmp_Den_Grid[spin][Mc_AN][Nc_3] += sum_3;
	    */

	  } /* spin */
	} /* Nog */

#pragma omp for
	for (Nog = NumOLG[Mc_AN][h_AN] - (NumOLG[Mc_AN][h_AN] % 4); Nog<NumOLG[Mc_AN][h_AN]; Nog++){
	  /*for (; Nog<NumOLG[Mc_AN][h_AN]; Nog++){*/
	
	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog]; 
 

	  if (Cnt_kind==0 && Cnt_switch==1){
	    for (i=0; i<NO0; i++){
	      orbs0[i] = COrbs_Grid[Mc_AN][i][Nc];
	    }
	    for (j=0; j<NO1; j++){
	      orbs1[j] = COrbs_Grid[Mh_AN][j][Nh];
	    }
	  }
	  else{
	    for (i=0; i<NO0; i++){
	      orbs0[i] = Orbs_Grid[Mc_AN][Nc][i];
	    }

	    if (G2ID[Gh_AN]==myid){
	      for (j=0; j<NO1; j++){
		orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];
	      }
	    }
	    else{
	      for (j=0; j<NO1; j++){
		orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];
	      }
	    }
	  }

	  for (spin=0; spin<=SpinP_switch; spin++){
 
 
	    sum = 0.0;
	    for (i=0; i<NO0; i++){
	      tmp0 = 0.0;
	      for (j=0; j<NO1; j++){
		tmp0 += orbs1[j]*tmp_CDM[spin][i][j];
	      }
	      sum += orbs0[i]*tmp0;
	    }
 
	    ai_tmpDGs[spin][Nc] += sum;
	    /*Tmp_Den_Grid[spin][Mc_AN][Nc] += sum;*/
	  }

	} /* Nog */
	
      } /* h_AN */

      /* AITUNE   merge temporary buffer for all omp threads */	
      for (spin=0; spin<=SpinP_switch; spin++){
	int Nc;
#pragma omp for
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	  double sum = 0.0;
	  int th;
	  for(th = 0; th < Nthrds; th++){
	    sum += ai_tmpDG_all[th][spin][Nc];
	  }
	  Tmp_Den_Grid[spin][Mc_AN][Nc] += sum;
	}
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */ 

    free(orbs0);
    free(orbs1);

    free(orbs0_0);
    free(orbs0_1);
    free(orbs0_2);
    free(orbs0_3);
    free(orbs1_0);
    free(orbs1_1);
    free(orbs1_2);
    free(orbs1_3);

    for (i=0; i<(SpinP_switch+1); i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(tmp_CDM[i][j]);
      }
      free(tmp_CDM[i]);
	
      free(ai_tmpDGs[i]); /* AITUNE */
	
    }
    free(tmp_CDM);

#pragma omp flush(Tmp_Den_Grid)

  } /* #pragma omp parallel */
  
  free(ai_tmpDG_all);

  dtime(&time2);
  if(myid==0 && measure_time){
    printf("Time for Part1=%18.5f\n",(time2-time1));fflush(stdout);
  }

  /******************************************************
      MPI communication from the partitions A to B 
  ******************************************************/
  
  /* copy Tmp_Den_Grid to Den_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = (int)(n2D*(unsigned long long int)numprocs/N2D);

      if (SpinP_switch==0){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = Tmp_Den_Grid[0][Mc_AN][AN];
      }
      else if (SpinP_switch==1){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+1] = Tmp_Den_Grid[1][Mc_AN][AN];
      }
      else if (SpinP_switch==3){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+1] = Tmp_Den_Grid[1][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+2] = Tmp_Den_Grid[2][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*4+3] = Tmp_Den_Grid[3][Mc_AN][AN];
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
      MPI_Isend( &Den_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS]*(SpinP_switch+1), 
	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      MPI_Irecv( &Den_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR]*(SpinP_switch+1), 
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
  for (i=0; i<Num_Rcv_Grid_A2B[myid]*(SpinP_switch+1); i++){
    Den_Rcv_Grid_A2B[myid][i] = Den_Snd_Grid_A2B[myid][i];
  }

  /******************************************************
   superposition of rho_i to calculate charge density 
   in the partition B.
  ******************************************************/

  /* initialize arrays */
  
  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[spin][BN] = 0.0;
    }
  }
  
  /* superposition of densities rho_i */

  for (ID=0; ID<numprocs; ID++){

    for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

      BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];      
      Gc_AN = Index_Rcv_Grid_A2B[ID][3*LN+1];        
      GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2]; 

      if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){

	/* spin collinear non-polarization */
	if ( SpinP_switch==0 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN];
	}

	/* spin collinear polarization */
	else if ( SpinP_switch==1 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN*2  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][LN*2+1];
	} 

	/* spin non-collinear */
	else if ( SpinP_switch==3 ){
	  Density_Grid_B0[0][BN] += Den_Rcv_Grid_A2B[ID][LN*4  ];
	  Density_Grid_B0[1][BN] += Den_Rcv_Grid_A2B[ID][LN*4+1];
	  Density_Grid_B0[2][BN] += Den_Rcv_Grid_A2B[ID][LN*4+2];
	  Density_Grid_B0[3][BN] += Den_Rcv_Grid_A2B[ID][LN*4+3];
	} 

      } /* if (Solve!=4.....) */           

    } /* AN */ 
  } /* ID */  

  /****************************************************
   Conjugate complex of Density_Grid[3][MN] due to
   difference in the definition between density matrix
   and charge density
  ****************************************************/

  if (SpinP_switch==3){

    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[3][BN] = -Density_Grid_B0[3][BN]; 
    }
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B0);

  /* freeing of arrays */

  for (i=0; i<(SpinP_switch+1); i++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(Tmp_Den_Grid[i][Mc_AN]);
    }
    free(Tmp_Den_Grid[i]);
  }
  free(Tmp_Den_Grid);

  for (ID=0; ID<numprocs; ID++){
    free(Den_Snd_Grid_A2B[ID]);
  }  
  free(Den_Snd_Grid_A2B);

  for (ID=0; ID<numprocs; ID++){
    free(Den_Rcv_Grid_A2B[ID]);
  }
  free(Den_Rcv_Grid_A2B);

  /* elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  if(myid==0 && measure_time) printf("time0=%18.5f\n",time0);

  return time0;
}



void Data_Grid_Copy_B2C_2(double **data_B, double **data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_S[NN_B2C_S]*(SpinP_switch+1)); 
  Work_Array_Rcv_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_R[NN_B2C_R]*(SpinP_switch+1)); 

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Snd_Grid_B2C",
		sizeof(double)*GP_B2C_S[NN_B2C_S]*(SpinP_switch+1), NULL);
    PrintMemory("Data_Grid_Copy_B2C_2: Work_Array_Rcv_Grid_B2C",
		sizeof(double)*GP_B2C_R[NN_B2C_R]*(SpinP_switch+1), NULL);
    firsttime = 0;
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

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[(SpinP_switch+1)*gp], Num_Rcv_Grid_B2C[IDR]*(SpinP_switch+1),
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

      if (SpinP_switch==0){
        Work_Array_Snd_Grid_B2C[gp+LN]       = data_B[0][BN];
      }
      else if (SpinP_switch==1){
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+0] = data_B[0][BN];
        Work_Array_Snd_Grid_B2C[2*gp+2*LN+1] = data_B[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+0] = data_B[0][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+1] = data_B[1][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+2] = data_B[2][BN];
        Work_Array_Snd_Grid_B2C[4*gp+4*LN+3] = data_B[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[(SpinP_switch+1)*gp], Num_Snd_Grid_B2C[IDS]*(SpinP_switch+1), 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){

      gp = GP_B2C_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){

	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[gp+LN];
	}     
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Snd_Grid_B2C[4*gp+4*LN+3];
	}
      } /* LN */   

    }
    else {

      gp = GP_B2C_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];

	if (SpinP_switch==0){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
	}
	else if (SpinP_switch==1){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  data_C[0][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+0];
	  data_C[1][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+1];
	  data_C[2][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+2];
	  data_C[3][CN] = Work_Array_Rcv_Grid_B2C[4*gp+4*LN+3];
	}
      }
    }
  }

  /* if (SpinP_switch==0), 
     copy data_B[0] to data_B[1]
     copy data_C[0] to data_C[1]
  */

  if (SpinP_switch==0){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      data_B[1][BN] = data_B[0][BN]; 
    }

    for (CN=0; CN<My_NumGridC; CN++){
      data_C[1][CN] = data_C[0][CN]; 
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}



void Data_Grid_Copy_B2C_1(double *data_B, double *data_C)
{
  static int firsttime=1;
  int CN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2C;
  double *Work_Array_Rcv_Grid_B2C;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_S[NN_B2C_S]); 
  Work_Array_Rcv_Grid_B2C = (double*)malloc(sizeof(double)*GP_B2C_R[NN_B2C_R]); 

  if (firsttime==1){
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Snd_Grid_B2C",
		sizeof(double)*GP_B2C_S[NN_B2C_S], NULL);
    PrintMemory("Data_Grid_Copy_B2C_1: Work_Array_Rcv_Grid_B2C",
		sizeof(double)*GP_B2C_R[NN_B2C_R], NULL);
    firsttime = 0;
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

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];
    gp = GP_B2C_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2C[gp], Num_Rcv_Grid_B2C[IDR],
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
      Work_Array_Snd_Grid_B2C[gp+LN] = data_B[BN];
    } 

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2C[gp], Num_Snd_Grid_B2C[IDS], 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2C to data_C */

  for (ID=0; ID<NN_B2C_R; ID++){

    IDR = ID_NN_B2C_R[ID];

    if (IDR==myid){
      gp = GP_B2C_S[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Snd_Grid_B2C[gp+LN];
      } 
    }
    else{

      gp = GP_B2C_R[ID];
      for (LN=0; LN<Num_Rcv_Grid_B2C[IDR]; LN++){
	CN = Index_Rcv_Grid_B2C[IDR][LN];
	data_C[CN] = Work_Array_Rcv_Grid_B2C[gp+LN];
      }
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2C);
  free(Work_Array_Rcv_Grid_B2C);
}





void Density_Grid_Copy_B2D(double **Density_Grid_B0)
{
  static int firsttime=1;
  int DN,BN,LN,spin,i,gp,NN_S,NN_R;
  double *Work_Array_Snd_Grid_B2D;
  double *Work_Array_Rcv_Grid_B2D;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */
  
  Work_Array_Snd_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_S[NN_B2D_S]*(SpinP_switch+1)); 
  Work_Array_Rcv_Grid_B2D = (double*)malloc(sizeof(double)*GP_B2D_R[NN_B2D_R]*(SpinP_switch+1)); 

  if (firsttime==1){
    PrintMemory("Set_Density_Grid: Work_Array_Snd_Grid_B2D",
		sizeof(double)*GP_B2D_S[NN_B2D_S]*(SpinP_switch+1), NULL);
    PrintMemory("Set_Density_Grid: Work_Array_Rcv_Grid_B2D",
		sizeof(double)*GP_B2D_R[NN_B2D_R]*(SpinP_switch+1), NULL);
    firsttime = 0;
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

  /* MPI_Irecv */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];
    gp = GP_B2D_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &Work_Array_Rcv_Grid_B2D[(SpinP_switch+1)*gp], Num_Rcv_Grid_B2D[IDR]*(SpinP_switch+1),
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B2D_S; ID++){

    IDS = ID_NN_B2D_S[ID];
    gp = GP_B2D_S[ID];

    /* copy Density_Grid_B0 to Work_Array_Snd_Grid_B2D */

    for (LN=0; LN<Num_Snd_Grid_B2D[IDS]; LN++){

      BN = Index_Snd_Grid_B2D[IDS][LN];

      if (SpinP_switch==0){
        Work_Array_Snd_Grid_B2D[gp+LN]       = Density_Grid_B0[0][BN];
      }
      else if (SpinP_switch==1){
        Work_Array_Snd_Grid_B2D[2*gp+2*LN+0] = Density_Grid_B0[0][BN];
        Work_Array_Snd_Grid_B2D[2*gp+2*LN+1] = Density_Grid_B0[1][BN];
      }
      else if (SpinP_switch==3){
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+0] = Density_Grid_B0[0][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+1] = Density_Grid_B0[1][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+2] = Density_Grid_B0[2][BN];
        Work_Array_Snd_Grid_B2D[4*gp+4*LN+3] = Density_Grid_B0[3][BN];
      }
    } /* LN */        

    if (IDS!=myid){
      MPI_Isend( &Work_Array_Snd_Grid_B2D[(SpinP_switch+1)*gp], Num_Snd_Grid_B2D[IDS]*(SpinP_switch+1), 
		 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  /* copy Work_Array_Rcv_Grid_B2D to Density_Grid_D */

  for (ID=0; ID<NN_B2D_R; ID++){

    IDR = ID_NN_B2D_R[ID];

    if (IDR==myid){

      gp = GP_B2D_S[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[gp+LN];
	}     
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Snd_Grid_B2D[4*gp+4*LN+3];
	}
      } /* LN */   

    }

    else{

      gp = GP_B2D_R[ID];

      for (LN=0; LN<Num_Rcv_Grid_B2D[IDR]; LN++){

	DN = Index_Rcv_Grid_B2D[IDR][LN];

	if (SpinP_switch==0){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[gp+LN];
	}     
	else if (SpinP_switch==1){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[2*gp+2*LN+1];
	}     
	else if (SpinP_switch==3){
	  Density_Grid_D[0][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+0];
	  Density_Grid_D[1][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+1];
	  Density_Grid_D[2][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+2];
	  Density_Grid_D[3][DN] = Work_Array_Rcv_Grid_B2D[4*gp+4*LN+3];
	}
      }

    }
  }

  /* if (SpinP_switch==0), copy Density_Grid to Density_Grid? */

  if (SpinP_switch==0){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B0[1][BN] = Density_Grid_B0[0][BN]; 
    }

    for (DN=0; DN<My_NumGridD; DN++){
      Density_Grid_D[1][DN] = Density_Grid_D[0][DN]; 
    }
  }

  /* freeing of arrays */
  free(Work_Array_Snd_Grid_B2D);
  free(Work_Array_Rcv_Grid_B2D);
}


void diagonalize_nc_density(double **Density_Grid_B0)
{
  int BN,DN,Mc_AN,Gc_AN,Nog,GRc;
  double Re11,Re22,Re12,Im12;
  double phi[2],theta[2],sit,cot,sip,cop;
  double d1,d2,d3,x,y,z,Cxyz[4];
  double Nup[2],Ndown[2];
  /* for OpenMP */
  int OMPID,Nthrds;

  /************************************
     Density_Grid in the partition B
  ************************************/

#pragma omp parallel shared(Density_Grid_B0,My_NumGridB_AB) private(OMPID,Nthrds,BN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (BN=OMPID; BN<My_NumGridB_AB; BN+=Nthrds){

      Re11 = Density_Grid_B0[0][BN];
      Re22 = Density_Grid_B0[1][BN];
      Re12 = Density_Grid_B0[2][BN];
      Im12 = Density_Grid_B0[3][BN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      /*
      if (    1.0e-7<fabs(Re11-Nup[0]) 
	   || 1.0e-7<fabs(Re22-Ndown[0])
	   || 1.0e-7<fabs(Re12-theta[0])
	   || 1.0e-7<fabs(Im12-phi[0]) ){

        printf("ZZZ1 BN=%2d Re11=%15.12f Re22=%15.12f Re12=%15.12f Im12=%15.12f\n",BN,Re11,Re22,Re12,Im12);
        printf("ZZZ2 BN=%2d Nup =%15.12f Ndn =%15.12f thet=%15.12f phi =%15.12f\n",BN,Nup[0],Ndown[0],theta[0],phi[0]);
      }
      */

      Density_Grid_B0[0][BN] = Nup[0];
      Density_Grid_B0[1][BN] = Ndown[0];
      Density_Grid_B0[2][BN] = theta[0];
      Density_Grid_B0[3][BN] = phi[0];
    }

#pragma omp flush(Density_Grid_B)

  } /* #pragma omp parallel */

  /************************************
     Density_Grid in the partition D
  ************************************/

#pragma omp parallel shared(Density_Grid_D,My_NumGridD) private(OMPID,Nthrds,DN,Re11,Re22,Re12,Im12,Nup,Ndown,theta,phi) default(none)
  {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (DN=OMPID; DN<My_NumGridD; DN+=Nthrds){

      Re11 = Density_Grid_D[0][DN];
      Re22 = Density_Grid_D[1][DN];
      Re12 = Density_Grid_D[2][DN];
      Im12 = Density_Grid_D[3][DN];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      Density_Grid_D[0][DN] = Nup[0];
      Density_Grid_D[1][DN] = Ndown[0];
      Density_Grid_D[2][DN] = theta[0];
      Density_Grid_D[3][DN] = phi[0];
    }

#pragma omp flush(Density_Grid_D)

  } /* #pragma omp parallel */

}
