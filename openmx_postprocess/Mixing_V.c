#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "openmx_common.h"
#include "tran_variables.h"
#include "mpi.h"

#define  maxima_step  1000000.0
#define  measure_time   0

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) ((a)<(b))?  (a):(b)

static void Kerker_Mixing_VKSk( int Change_switch, double Mix_wgt );
static void DIIS_Mixing_VKSk(int SCF_iter, double Mix_wgt );
static void Inverse(int n, double **a, double **ia);
static void Complex_Inverse(int n, double **a, double **ia, double **b, double **ib);

double Mixing_V( int MD_iter,
		 int SCF_iter,
		 int SCF_iter0 )
{
  int pSCF_iter,NumMix,NumSlide;
  int spin,ct_AN,wan1,TNO1,h_AN,Gh_AN,wan2;
  int TNO2,i,j,ian,jan,m,n;
  int spinmax,k1,k2,k3;
  int Mc_AN,Gc_AN;
  double time0;
  double TStime,TEtime;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /*********************************************************
   RMM-DIIS mixing with Kerker's weighting factor in k-space
   for the Kohn-Sham potential
  *********************************************************/

  if (MD_iter==1 && SCF_iter==1){
    Kerker_Mixing_VKSk(0,Mixing_weight);
  }

  else if (SCF_iter<Pulay_SCF){
    Kerker_Mixing_VKSk(1,Mixing_weight);
  }

  else{

    if (6<SCF_iter){
      DIIS_Mixing_VKSk(SCF_iter-(Pulay_SCF-3),Mixing_weight);
    }
    else{
      DIIS_Mixing_VKSk(SCF_iter,Mixing_weight);
    }
  }

  /* if SCF_iter0==1, then NormRD[0]=1 */

  if (SCF_iter0==1) NormRD[0] = 1.0;

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 


void DIIS_Mixing_VKSk(int SCF_iter, double Mix_wgt)
{
  static int firsttime=1;
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan,k1,k2,k3,kk2; 
  int pSCF_iter,size_Re_OptVKSk,size_Kerker_weight;
  int spinmax,MN,flag_nan,refiter,po3,po4,refiter2;
  int N2D,GN,GNs,N3[4],BN_AB,p0,p1,M,N,K;
  double time1,time2,time3,time4,time5; 
  double Stime1,Etime1,scale;
  double *alden,*alden0,*gradF;
  double **A,**IA;
  double *My_NMat,*NMat;
  double alpha,beta;
  double tmp0,itmp0,tmp1;
  double im1,im2,re1,re2; 
  double OptRNorm,My_OptRNorm,sum0,sum1;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptVKS;
  double Gx,Gy,Gz,G2,G12,G22,G32,F,F0;
  double Min_Weight;
  double Max_Weight;
  double G0,G02,G02p,weight,wgt0,wgt1;
  double sk1,sk2,sk3;
  double **Re_OptVKSk;
  double **Im_OptVKSk;
  double *Kerker_weight;
  int numprocs,myid;
  char nanchar[300];
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* find an optimum G0 */

  G12 = rtv[1][1]*rtv[1][1] + rtv[1][2]*rtv[1][2] + rtv[1][3]*rtv[1][3]; 
  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G12<G22) G0 = G12;
  else         G0 = G22;
  if (G32<G0)  G0 = G32;

  G0 = sqrt(G0);
  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.1*Kerker_factor*G0)*(0.1*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /****************************************************
    allocation of arrays:
  ****************************************************/
  
  alden = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  alden0 = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  gradF = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  
  My_NMat = (double*)malloc(sizeof(double)*List_YOUSO[38]*List_YOUSO[38]);
  NMat = (double*)malloc(sizeof(double)*List_YOUSO[38]*List_YOUSO[38]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  }
  
  IA = (double**)malloc(sizeof(double*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  }

  /* Re_OptVKSk and Im_OptVKSk */

  size_Re_OptVKSk = spinmax*My_NumGridB_CB;

  Re_OptVKSk = (double**)malloc(sizeof(double*)*spinmax); 
  for (spin=0; spin<spinmax; spin++){
    /* We allocate with My_Max_NumGridB since it is used
       as temporal array in "find VKS in real space" */
    Re_OptVKSk[spin] = (double*)malloc(sizeof(double)*My_Max_NumGridB);  
  }

  Im_OptVKSk = (double**)malloc(sizeof(double*)*spinmax); 
  for (spin=0; spin<spinmax; spin++){
    /* We allocate with My_Max_NumGridB since it is used
       as temporal array in "find VKS in real space" */
    Im_OptVKSk[spin] = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  }

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = (double*)malloc(sizeof(double)*My_NumGridB_CB); 

  /***********************************
            set Kerker_weight 
  ************************************/

  if (measure_time==1) dtime(&Stime1);

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (k=0; k<My_NumGridB_CB; k++){

    /* get k3, k2, and k1 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

    if (k1<Ngrid1/2) sk1 = (double)k1;
    else             sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2) sk2 = (double)k2;
    else             sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    if (k1==0 && k2==0 && k3==0)  weight = 1.0;
    else                          weight = (G2 + G02)/(G2 + G02p);

    Kerker_weight[k] = sqrt(weight);

  } /* k */

  if (measure_time==1){ 
    dtime(&Etime1);
    time1 = Etime1 - Stime1;      
  }

  /****************************************************
     start calc.
  ****************************************************/
 
  if (SCF_iter==1){
    /* memory calc. done, only 1st iteration */
    if (firsttime) {
      PrintMemory("DIIS_Mixing_VKSk: Re_OptVKSk",sizeof(double)*size_Re_OptVKSk,NULL);
      PrintMemory("DIIS_Mixing_VKSk: Im_OptVKSk",sizeof(double)*size_Re_OptVKSk,NULL);
      PrintMemory("DIIS_Mixing_VKSk: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
      firsttime=0;
    }

    Kerker_Mixing_VKSk(0,1.0);
  } 

  else if (SCF_iter==2){

    if (measure_time==1) dtime(&Stime1);

    /* construct residual rho */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReVKSk[spin][2*My_NumGridB_CB+k] = (ReVKSk[0][spin][k] - ReVKSk[1][spin][k])*Kerker_weight[k];
	Residual_ImVKSk[spin][2*My_NumGridB_CB+k] = (ImVKSk[0][spin][k] - ImVKSk[1][spin][k])*Kerker_weight[k];

      } /* k */
    } /* spin */

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 = Etime1 - Stime1;      
    }

    /* rho1 to rho2 */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReVKSk[2][spin][k] = ReVKSk[1][spin][k];
	ImVKSk[2][spin][k] = ImVKSk[1][spin][k];
      }
    }

    Kerker_Mixing_VKSk(0,Mixing_weight);
  } 

  else {

    /****************************************************
             construct residual KS potential 
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReVKSk[spin][My_NumGridB_CB+k] = (ReVKSk[0][spin][k] - ReVKSk[1][spin][k])*Kerker_weight[k];
	Residual_ImVKSk[spin][My_NumGridB_CB+k] = (ImVKSk[0][spin][k] - ImVKSk[1][spin][k])*Kerker_weight[k];
      } /* k */
    } /* spin */

    if (measure_time==1){ 
      dtime(&Etime1);
      time3 = Etime1 - Stime1;      
    }

    if ((SCF_iter-1)<Num_Mixing_pDM){
      NumMix   = SCF_iter - 1;
      NumSlide = NumMix + 1;
    }
    else{
      NumMix   = Num_Mixing_pDM;
      NumSlide = NumMix;
    }

    /****************************************************
              alpha from residual KS potentials
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    /* calculation of the norm matrix for the residual vectors */

    for (i=0; i<List_YOUSO[38]*List_YOUSO[38]; i++) My_NMat[i] = 0.0;

    for (spin=0; spin<spinmax; spin++){

      M = NumMix + 1;
      N = NumMix + 1;
      K = My_NumGridB_CB;   
      alpha = 1.0;
      beta = 1.0;

      F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K, 
                             &alpha, 
                             Residual_ReVKSk[spin], &K, 
                             Residual_ReVKSk[spin], &K, 
                             &beta, 
                             My_NMat,
                             &M);

      F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K, 
                             &alpha, 
                             Residual_ImVKSk[spin], &K, 
                             Residual_ImVKSk[spin], &K, 
                             &beta, 
                             My_NMat,
                             &M);
    }      

    MPI_Allreduce(My_NMat, NMat, (NumMix+1)*(NumMix+1),
                  MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    for (i=0; i<NumMix; i++){
      for (j=0; j<NumMix; j++){
        A[i][j] = NMat[(i+1)*(NumMix+1)+(j+1)];
      }
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 = Etime1 - Stime1;      
    }

    /* store NormRD */

    for (i=4; 1<=i; i--){
      NormRD[i] = NormRD[i-1];
      History_Uele[i] = History_Uele[i-1];
    }
    NormRD[0] = A[0][0]/(double)(1.0e+5*Ngrid1*Ngrid2*Ngrid3*spinmax);
    History_Uele[0] = Uele;

    /* set matrix elements on lammda */

    for (SCFi=1; SCFi<=NumMix; SCFi++){
       A[SCFi-1][NumMix] = -1.0;
       A[NumMix][SCFi-1] = -1.0;
    }
    A[NumMix][NumMix] = 0.0;

    /* solve the linear equation */

    Inverse(NumMix,A,IA);

    for (SCFi=1; SCFi<=NumMix; SCFi++){
       alden[SCFi] = -IA[SCFi-1][NumMix];
    }

    /*************************************
      check "nan", "NaN", "inf" or "Inf"
    *************************************/

    flag_nan = 0;
    for (SCFi=1; SCFi<=NumMix; SCFi++){

      sprintf(nanchar,"%8.4f",alden[SCFi]);
      if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
       || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

        flag_nan = 1;
      }
    }

    if (flag_nan==1){
      for (SCFi=1; SCFi<=NumMix; SCFi++){
        alden[SCFi] = 0.0;
      }
      alden[1] = 0.1;
      alden[2] = 0.9;
    }

    /****************************************************
              calculate optimized residual rho
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
        Re_OptVKSk[spin][k] = 0.0;
        Im_OptVKSk[spin][k] = 0.0;
      }
    }   

    /* Pulay mixing for residual VKSk */

    for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

      tmp0 = alden[pSCF_iter]; 
      p0 = pSCF_iter*My_NumGridB_CB;

#pragma omp parallel shared(spinmax,My_NumGridB_CB,Re_OptVKSk,Im_OptVKSk,tmp0,p0,Residual_ReVKSk,Residual_ImVKSk)
      {

        int OMPID,Nthrds,k; 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();

	if (spinmax==1){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	    Re_OptVKSk[0][k] += tmp0*Residual_ReVKSk[0][p0+k];
	    Im_OptVKSk[0][k] += tmp0*Residual_ImVKSk[0][p0+k];
	  } /* k */
	}

	else if (spinmax==2){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	    Re_OptVKSk[0][k] += tmp0*Residual_ReVKSk[0][p0+k];
	    Im_OptVKSk[0][k] += tmp0*Residual_ImVKSk[0][p0+k];
	    Re_OptVKSk[1][k] += tmp0*Residual_ReVKSk[1][p0+k];
	    Im_OptVKSk[1][k] += tmp0*Residual_ImVKSk[1][p0+k];
	  } /* k */
	}

	else if (spinmax==3){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	    Re_OptVKSk[0][k] += tmp0*Residual_ReVKSk[0][p0+k];
	    Im_OptVKSk[0][k] += tmp0*Residual_ImVKSk[0][p0+k];
	    Re_OptVKSk[1][k] += tmp0*Residual_ReVKSk[1][p0+k];
	    Im_OptVKSk[1][k] += tmp0*Residual_ImVKSk[1][p0+k];
	    Re_OptVKSk[2][k] += tmp0*Residual_ReVKSk[2][p0+k];
	    Im_OptVKSk[2][k] += tmp0*Residual_ImVKSk[2][p0+k];
	  } /* k */
	}

      } /* #pragma omp parallel */

    } /* pSCF_iter */

    if      (1.0e-2<=NormRD[0])                        coef_OptVKS = 0.5;
    else if (1.0e-4<=NormRD[0]  && NormRD[0]<1.0e-2)   coef_OptVKS = 0.6;
    else if (1.0e-6<=NormRD[0]  && NormRD[0]<1.0e-4)   coef_OptVKS = 0.7;
    else if (1.0e-8<=NormRD[0]  && NormRD[0]<1.0e-6)   coef_OptVKS = 0.8;
    else if (1.0e-10<=NormRD[0] && NormRD[0]<1.0e-8)   coef_OptVKS = 0.9;
    else                                               coef_OptVKS = 1.0;

    /****************************************************
                       mixing of VKS
    ****************************************************/

    /****************************************************
       Pulay mixing
    ****************************************************/

    if ( SCF_iter%EveryPulay_SCF==0 ) {

      /* reduce Mixing_weight so that charge sloshing can be avoided at the next SCF */

      Mixing_weight = 0.1*Max_Mixing_weight;
      if (Mixing_weight<Min_Mixing_weight) Mixing_weight = Min_Mixing_weight;
      
      /* initial ReVKSk and ImVKSk */
      
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReVKSk[0][spin][k] = 0.0;
	  ImVKSk[0][spin][k] = 0.0;
	}
      }

      /* Pulay mixing */

      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

	tmp0 =  alden[pSCF_iter];

#pragma omp parallel shared(tmp0,pSCF_iter,spinmax,My_NumGridB_CB,ReVKSk,ImVKSk) 
	{

          int OMPID,Nthrds,k; 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();

	  if (spinmax==1){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReVKSk[0][0][k] += tmp0*ReVKSk[pSCF_iter][0][k];
	      ImVKSk[0][0][k] += tmp0*ImVKSk[pSCF_iter][0][k];
	    } /* k */
	  }

	  else if (spinmax==2){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReVKSk[0][0][k] += tmp0*ReVKSk[pSCF_iter][0][k];
	      ImVKSk[0][0][k] += tmp0*ImVKSk[pSCF_iter][0][k];
	      ReVKSk[0][1][k] += tmp0*ReVKSk[pSCF_iter][1][k];
	      ImVKSk[0][1][k] += tmp0*ImVKSk[pSCF_iter][1][k];
	    } /* k */
	  }

	  else if (spinmax==3){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReVKSk[0][0][k] += tmp0*ReVKSk[pSCF_iter][0][k];
	      ImVKSk[0][0][k] += tmp0*ImVKSk[pSCF_iter][0][k];
	      ReVKSk[0][1][k] += tmp0*ReVKSk[pSCF_iter][1][k];
	      ImVKSk[0][1][k] += tmp0*ImVKSk[pSCF_iter][1][k];
	      ReVKSk[0][2][k] += tmp0*ReVKSk[pSCF_iter][2][k];
	      ImVKSk[0][2][k] += tmp0*ImVKSk[pSCF_iter][2][k];
	    } /* k */
	  }

	} /* #pragma omp parallel */

      } /* pSCF_iter */

      /* Correction by optimized residual VKS */

      for (spin=0; spin<spinmax; spin++){

        for (k=0; k<My_NumGridB_CB; k++){

 	  if (1.0<NormRD[0])
  	    weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
          else 
            weight = 1.0/Kerker_weight[k];

	  ReVKSk[0][spin][k] += coef_OptVKS*Re_OptVKSk[spin][k]*weight;
	  ImVKSk[0][spin][k] += coef_OptVKS*Im_OptVKSk[spin][k]*weight;

	  /* correction to large changing components */

          tmp0 = ReVKSk[0][spin][k] - ReVKSk[1][spin][k];  
          tmp1 = ImVKSk[0][spin][k] - ImVKSk[1][spin][k];  

          if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
            ReVKSk[0][spin][k] = sgn(tmp0)*maxima_step + ReVKSk[1][spin][k]; 
            ImVKSk[0][spin][k] = sgn(tmp1)*maxima_step + ImVKSk[1][spin][k];
          }

	} /* k */
      } /* spin */

      if (measure_time==1){ 
        dtime(&Etime1);
        time5 = Etime1 - Stime1;      
      }

    } /* end of if ( SCF_iter%EveryPulay_SCF==0 ) */

    /****************************************************
       Kerker mixing
    ****************************************************/

    else {

      /* find an optimum mixing weight */

      Min_Weight = Min_Mixing_weight;
      Max_Weight = Max_Mixing_weight;

      if ((int)sgn(History_Uele[0]-History_Uele[1])
	  ==(int)sgn(History_Uele[1]-History_Uele[2])
	  && NormRD[0]<NormRD[1]){

	/* tmp0 = 1.6*Mixing_weight; */

	tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0){
	    Mixing_weight = tmp0;
	  }
	  else{ 
	    Mixing_weight = Min_Weight;
	  }
	}
	else{ 
	  Mixing_weight = Max_Weight;
	  SCF_RENZOKU++;  
	}
      }
   
      else if ((int)sgn(History_Uele[0]-History_Uele[1])
	       ==(int)sgn(History_Uele[1]-History_Uele[2])
	       && NormRD[1]<NormRD[0]){

	tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

	/* tmp0 = Mixing_weight/1.6; */

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0)
	    Mixing_weight = tmp0;
	  else 
	    Mixing_weight = Min_Weight;
	}
	else 
	  Mixing_weight = Max_Weight;

	SCF_RENZOKU = -1;  
      }

      else if ((int)sgn(History_Uele[0]-History_Uele[1])
	       !=(int)sgn(History_Uele[1]-History_Uele[2])
	       && NormRD[0]<NormRD[1]){

	/* tmp0 = Mixing_weight*1.2; */

	tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0)
	    Mixing_weight = tmp0;
	  else 
	    Mixing_weight = Min_Weight;
	}
	else{ 
	  Mixing_weight = Max_Weight;
	  SCF_RENZOKU++;
	}
      }

      else if ((int)sgn(History_Uele[0]-History_Uele[1])
	       !=(int)sgn(History_Uele[1]-History_Uele[2])
	       && NormRD[1]<NormRD[0]){

	/* tmp0 = Mixing_weight/2.0; */

	tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

	if (tmp0<Max_Weight){
	  if (Min_Weight<tmp0)
	    Mixing_weight = tmp0;
	  else 
	    Mixing_weight = Min_Weight;
	}
	else 
	  Mixing_weight = Max_Weight;

	SCF_RENZOKU = -1;
      }

      Mix_wgt = Mixing_weight;

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
         
	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
	  wgt0  = Mix_wgt*weight;
	  wgt1 =  1.0 - wgt0;
        
	  ReVKSk[0][spin][k] = wgt0*ReVKSk[0][spin][k] + wgt1*ReVKSk[1][spin][k];
	  ImVKSk[0][spin][k] = wgt0*ImVKSk[0][spin][k] + wgt1*ImVKSk[1][spin][k];
        
	} /* k */
      } /* spin */
    } /* else */

    /****************************************************
                         shift of VKS
    ****************************************************/

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
	  ReVKSk[pSCF_iter][spin][k] = ReVKSk[pSCF_iter-1][spin][k]; 
	  ImVKSk[pSCF_iter][spin][k] = ImVKSk[pSCF_iter-1][spin][k]; 
	}
      }
    }

    /****************************************************
                    shift of residual rho
    ****************************************************/

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

      p0 = pSCF_iter*My_NumGridB_CB;
      p1 = (pSCF_iter-1)*My_NumGridB_CB; 

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          Residual_ReVKSk[spin][p0+k] = Residual_ReVKSk[spin][p1+k];
          Residual_ImVKSk[spin][p0+k] = Residual_ImVKSk[spin][p1+k]; 
	}
      }
    }

  } /* else */

  /****************************************************
                 find VKS in real space 
  ****************************************************/

  tmp0 = 1.0/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3;

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      Re_OptVKSk[0][k] = ReVKSk[0][spin][k];
      Im_OptVKSk[0][k] = ImVKSk[0][spin][k];
    }

    if (spin==0 || spin==1){

      Get_Value_inReal(0,Vpot_Grid_B[spin],Vpot_Grid_B[spin],Re_OptVKSk[0],Im_OptVKSk[0]);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[spin][BN_AB] = Vpot_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Vpot_Grid_B[2],Vpot_Grid_B[3],Re_OptVKSk[0],Im_OptVKSk[0]);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[2][BN_AB] = Vpot_Grid_B[2][BN_AB]*tmp0;
	Vpot_Grid_B[3][BN_AB] = Vpot_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  if (measure_time==1){ 
    printf("time1=%12.6f time2=%12.6f time3=%12.6f time4=%12.6f time5=%12.6f\n",
            time1,time2,time3,time4,time5);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/
  
  free(alden);
  free(alden0);
  free(gradF);

  free(My_NMat);
  free(NMat);

  for (i=0; i<List_YOUSO[38]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[38]; i++){
    free(IA[i]);
  }
  free(IA);

  for (spin=0; spin<spinmax; spin++){
    free(Re_OptVKSk[spin]);
  }
  free(Re_OptVKSk);

  for (spin=0; spin<spinmax; spin++){
    free(Im_OptVKSk[spin]);
  }
  free(Im_OptVKSk);

  free(Kerker_weight);
}




void Kerker_Mixing_VKSk( int Change_switch, double Mix_wgt )
{
  static int firsttime=1;
  int ian,jan,Mc_AN,Gc_AN,spin,spinmax;
  int h_AN,Gh_AN,m,n,i,j,k,k1,k2,k3;
  int MN,pSCF_iter,p0,p1;
  int GN,GNs,N2D,BN_AB,N3[4];
  double Mix_wgt2,Norm,My_Norm,tmp0,tmp1;
  double Min_Weight,Max_Weight,wgt0,wgt1;
  double Gx,Gy,Gz,G2,size_Kerker_weight;
  double sk1,sk2,sk3,G12,G22,G32;
  double G0,G02,G02p,weight;
  int numprocs,myid,ID;
  double *Kerker_weight;
  double *ReVk,*ImVk;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocate arrays */

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = (double*)malloc(sizeof(double)*My_NumGridB_CB); 
  ReVk = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImVk = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

  if (firsttime)
  PrintMemory("Kerker_Mixing_VKSk: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
  firsttime=0;

  /* find an optimum G0 */

  G12 = rtv[1][1]*rtv[1][1] + rtv[1][2]*rtv[1][2] + rtv[1][3]*rtv[1][3]; 
  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G12<G22) G0 = G12;
  else         G0 = G22;
  if (G32<G0)  G0 = G32;

  if (Change_switch==0) G0 = 0.0;
  else                  G0 = sqrt(G0);

  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.1*Kerker_factor*G0)*(0.1*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /***********************************
            set Kerker_weight 
  ************************************/

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (k=0; k<My_NumGridB_CB; k++){

    /* get k3, k2, and k1 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

    if (k1<Ngrid1/2)  sk1 = (double)k1;
    else              sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2)  sk2 = (double)k2;
    else              sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2)  sk3 = (double)k3;
    else              sk3 = (double)(k3 - Ngrid3);

    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    if (k1==0 && k2==0 && k3==0)  weight = 1.0;
    else                          weight = (G2 + G02)/(G2+G02p);

    Kerker_weight[k] = sqrt(weight);

  } /* k */

  /* start... */

  Min_Weight = Min_Mixing_weight;
  if (SCF_RENZOKU==-1){
    Max_Weight = Max_Mixing_weight;
    Max_Mixing_weight2 = Max_Mixing_weight;
  }
  else if (SCF_RENZOKU==1000){  /* past 3 */
    Max_Mixing_weight2 = 2.0*Max_Mixing_weight2;
    if (0.7<Max_Mixing_weight2) Max_Mixing_weight2 = 0.7;
    Max_Weight = Max_Mixing_weight2;
    SCF_RENZOKU = 0;
  }
  else{
    Max_Weight = Max_Mixing_weight2;
  }

  /****************************************************
       norm of residual KS potential in k-space
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  My_Norm = 0.0;
  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = Kerker_weight[k];
      tmp0 = (ReVKSk[0][spin][k] - ReVKSk[1][spin][k])*weight;
      tmp1 = (ImVKSk[0][spin][k] - ImVKSk[1][spin][k])*weight;
      Residual_ReVKSk[spin][My_NumGridB_CB+k] = tmp0;
      Residual_ImVKSk[spin][My_NumGridB_CB+k] = tmp1;

      My_Norm += tmp0*tmp0 + tmp1*tmp1;

    } /* k */
  } /* spin */

  /****************************************************
    MPI: 

    My_Norm
  ****************************************************/

  MPI_Allreduce(&My_Norm, &Norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
    find an optimum mixing weight
  ****************************************************/

  for (i=4; 1<=i; i--){
    NormRD[i] = NormRD[i-1];
    History_Uele[i] = History_Uele[i-1];
  }
  NormRD[0] = Norm/(double)(1.0e+5*Ngrid1*Ngrid2*Ngrid3*spinmax);
  History_Uele[0] = Uele;

  if (Change_switch==1){

    if ((int)sgn(History_Uele[0]-History_Uele[1])
	  ==(int)sgn(History_Uele[1]-History_Uele[2])
       && NormRD[0]<NormRD[1]){

      /* tmp0 = 1.6*Mixing_weight; */

      tmp0 = 1.6*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0){
          Mixing_weight = tmp0;
	}
        else{ 
          Mixing_weight = Min_Weight;
	}
      }
      else{ 
        Mixing_weight = Max_Weight;
        SCF_RENZOKU++;  
      }
    }
   
    else if ((int)sgn(History_Uele[0]-History_Uele[1])
             ==(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[1]<NormRD[0]){

      /* tmp0 = Mixing_weight/1.6; */
 
      tmp0 = Mixing_weight/1.6;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else 
        Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;  
    }

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
	     !=(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[0]<NormRD[1]){

      /* tmp0 = Mixing_weight*1.2; */

      tmp0 = 1.2*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else{ 
        Mixing_weight = Max_Weight;
        SCF_RENZOKU++;
      }
    }

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
	     !=(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[1]<NormRD[0]){

      /* tmp0 = Mixing_weight/2.0; */

      tmp0 = Mixing_weight/2.0;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else 
        Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;
    }

    Mix_wgt = Mixing_weight;
  }

  /****************************************************
                        Mixing
  ****************************************************/

  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = 1.0/Kerker_weight[k];

      if (Change_switch==0){
        wgt0 = 1.0;
        wgt1 = 0.0;
      }
      else{
        wgt0  = Mix_wgt*weight;
        wgt1 =  1.0 - wgt0;
      }

      ReVKSk[0][spin][k] = wgt0*ReVKSk[0][spin][k] + wgt1*ReVKSk[1][spin][k];
      ImVKSk[0][spin][k] = wgt0*ImVKSk[0][spin][k] + wgt1*ImVKSk[1][spin][k];

      /* correction to largely changing components */

      tmp0 = ReVKSk[0][spin][k] - ReVKSk[1][spin][k];  
      tmp1 = ImVKSk[0][spin][k] - ImVKSk[1][spin][k];  

      if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
	ReVKSk[0][spin][k] = sgn(tmp0)*maxima_step + ReVKSk[1][spin][k]; 
	ImVKSk[0][spin][k] = sgn(tmp1)*maxima_step + ImVKSk[1][spin][k]; 
      }

    } /* k */
  } /* spin */

  /****************************************************
                  shift of KS potentials
  ****************************************************/

  for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){
    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReVKSk[pSCF_iter][spin][k] = ReVKSk[pSCF_iter-1][spin][k]; 
	ImVKSk[pSCF_iter][spin][k] = ImVKSk[pSCF_iter-1][spin][k]; 
      }
    }
  }

  /****************************************************
              shift of residual KS potentials
  ****************************************************/

  for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

    p0 = pSCF_iter*My_NumGridB_CB;
    p1 = (pSCF_iter-1)*My_NumGridB_CB; 

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	Residual_ReVKSk[spin][p0+k] = Residual_ReVKSk[spin][p1+k];
	Residual_ImVKSk[spin][p0+k] = Residual_ImVKSk[spin][p1+k]; 
      }
    }
  }

  /************************************************************
    find the KS potentials for the partition B in real space 
  ************************************************************/

  tmp0 = 1.0/(double)(Ngrid1*Ngrid2*Ngrid3);

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReVKSk[0][spin][k];
      ImVk[k] = ImVKSk[0][spin][k];
    }

    if (spin==0 || spin==1){

      Get_Value_inReal(0,Vpot_Grid_B[spin],Vpot_Grid_B[spin],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[spin][BN_AB] = Vpot_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Vpot_Grid_B[2],Vpot_Grid_B[3],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Vpot_Grid_B[2][BN_AB] = Vpot_Grid_B[2][BN_AB]*tmp0;
	Vpot_Grid_B[3][BN_AB] = Vpot_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  /* freeing of arrays */

  free(ImVk);
  free(ReVk);
  free(Kerker_weight);
}





void Inverse(int n, double **a, double **ia)
{
  int method_flag=1;

  if (method_flag==0){

  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w;
  double *x,*y;
  double **da;

  /***************************************************
    allocation of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = (double**)malloc(sizeof(double*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[38]);
    for (j=0; j<List_YOUSO[38]; j++){
      da[i][j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[38]; i++){
      for (j=0; j<List_YOUSO[38]; j++){
	a[i][j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/a[k][k];
      for (i=k+1; i<=n; i++){
	a[i][k] = w*a[i][k];
	for (j=k+1; j<=n; j++){
	  a[i][j] = a[i][j] - a[i][k]*a[k][j];
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - a[i][j]*y[j];
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - a[i][j]*x[j];
	}
	x[i] = x[i]/a[i][i];
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = da[i][j];
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[38]; i++){
    free(da[i]);
  }
  free(da);

  }
  
  else if (method_flag==1){

    int i,j,M,N,LDA,INFO;
    int *IPIV,LWORK;
    double *A,*WORK;

    A = (double*)malloc(sizeof(double)*(n+2)*(n+2));
    WORK = (double*)malloc(sizeof(double)*(n+2));
    IPIV = (int*)malloc(sizeof(int)*(n+2));

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        A[i*(n+1)+j] = a[i][j];
      }
    }

    M = n + 1;
    N = M;
    LDA = M;
    LWORK = M;

    F77_NAME(dgetrf,DGETRF)( &M, &N, A, &LDA, IPIV, &INFO);
    F77_NAME(dgetri,DGETRI)( &N, A, &LDA, IPIV, WORK, &LWORK, &INFO);

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        ia[i][j] = A[i*(n+1)+j];
      }
    }

    free(A);
    free(WORK);
    free(IPIV);
  }

  else if (method_flag==2){

    int N,i,j,k;
    double *A,*B,*ko;
    double sum;

    N = n + 1;

    A = (double*)malloc(sizeof(double)*(N+2)*(N+2));
    B = (double*)malloc(sizeof(double)*(N+2)*(N+2));
    ko = (double*)malloc(sizeof(double)*(N+2));

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        A[j*N+i] = a[i][j];
      }
    }

    Eigen_lapack3(A, ko, N, N); 

    for (i=0; i<N; i++){
      ko[i] = 1.0/(ko[i]+1.0e-13);
    } 

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        B[i*N+j] = A[i*N+j]*ko[i];
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        ia[i][j] = 0.0;
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        sum = 0.0;
	for (k=0; k<N; k++){
	  sum += A[k*N+i]*B[k*N+j];
	}
        ia[i][j] = sum;
      }
    }

    free(A);
    free(B);
    free(ko);
  }
}

void Complex_Inverse(int n, double **a, double **ia, double **b, double **ib)
{
  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w2,re,im;
  dcomplex w;
  dcomplex *x,*y;
  dcomplex **da;

  /***************************************************
    allocation of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/

  x = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[38]);
  y = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[38]);

  da = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    da[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[38]);
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[38]; i++){
      for (j=0; j<List_YOUSO[38]; j++){
 	 a[i][j] = 0.0;
	ia[i][j] = 0.0;
      }
    }
  }

  else {

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j].r =  a[i][j];
	da[i][j].i = ia[i][j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){

      w.r =  a[k][k];
      w.i = ia[k][k];
      w2 = 1.0/(w.r*w.r + w.i*w.i); 

      for (i=k+1; i<=n; i++){

	re = w2*(w.r* a[i][k] + w.i*ia[i][k]);
        im = w2*(w.r*ia[i][k] - w.i* a[i][k]);
	a[i][k] = re;
       ia[i][k] = im;

	for (j=k+1; j<=n; j++){
 	   re =  a[i][j] - (a[i][k]* a[k][j] - ia[i][k]*ia[k][j]);
	   im = ia[i][j] - (a[i][k]*ia[k][j] + ia[i][k]* a[k][j]);
 	   a[i][j] = re;
	  ia[i][j] = im;
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k){
	  y[i].r = 1.0;
	  y[i].i = 0.0;
	}
	else{
	  y[i].r = 0.0;
	  y[i].i = 0.0;
	}

	for (j=0; j<=i-1; j++){
   	  y[i].r = y[i].r - (a[i][j]*y[j].r - ia[i][j]*y[j].i);
	  y[i].i = y[i].i - (a[i][j]*y[j].i + ia[i][j]*y[j].r);
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i].r = x[i].r - (a[i][j]*x[j].r - ia[i][j]*x[j].i);
	  x[i].i = x[i].i - (a[i][j]*x[j].i + ia[i][j]*x[j].r);
	}

        w.r =  a[i][i];
        w.i = ia[i][i];
        w2 = 1.0/(w.r*w.r + w.i*w.i); 

	re = w2*(w.r*x[i].r + w.i*x[i].i);
        im = w2*(w.r*x[i].i - w.i*x[i].r);
	x[i].r = re;
	x[i].i = im;

         b[i][k] = x[i].r;
        ib[i][k] = x[i].i;
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	 a[i][j] = da[i][j].r;
	ia[i][j] = da[i][j].i;
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[38]]
     y[List_YOUSO[38]]
     da[List_YOUSO[38]][List_YOUSO[38]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[38]; i++){
    free(da[i]);
  }
  free(da);
}
