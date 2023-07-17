/**********************************************************************
  DIIS_Mixing_Rhok.c:

     DIIS_Mixing_Rhok.c is a subroutine to achieve self-consistent field
     using the direct inversion in the iterative subspace in k-space.

  Log of DIIS_Mixing_Rhok.c:

     3/Jan/2005  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "openmx_common.h"
#include "tran_prototypes.h"
#include "tran_variables.h"
#include "lapack_prototypes.h"
#include "mpi.h"
#include <omp.h>


#define  measure_time   0
#define  maxima_step  10000000.0

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) ((a)<(b))?  (a):(b)


static void Inverse(int n, double **a, double **ia);
static void Complex_Inverse(int n, double **a, double **ia, double **b, double **ib);


static void DIIS_Mixing_Rhok_Normal(int SCF_iter,
			      double Mix_wgt,
			      double ***ReRhok,
			      double ***ImRhok,
			      double **Residual_ReRhok,
			      double **Residual_ImRhok,
			      double *ReVk,
			      double *ImVk,
			      double *ReRhoAtomk,
			      double *ImRhoAtomk);


static void DIIS_Mixing_Rhok_NEGF(int SCF_iter,
				  double Mix_wgt,
				  double ***ReRhok,
				  double ***ImRhok,
				  double **Residual_ReRhok,
				  double **Residual_ImRhok,
				  double *ReVk,
				  double *ImVk,
				  double *ReRhoAtomk,
				  double *ImRhoAtomk);
	
void DIIS_Mixing_Rhok(int SCF_iter,
                      double Mix_wgt,
                      double ***ReRhok,
                      double ***ImRhok,
                      double **Residual_ReRhok,
                      double **Residual_ImRhok,
                      double *ReVk,
                      double *ImVk,
                      double *ReRhoAtomk,
                      double *ImRhoAtomk)
{

  if (Solver!=4 || TRAN_Poisson_flag==2){

    DIIS_Mixing_Rhok_Normal(
			    SCF_iter,
			    Mix_wgt,
			    ReRhok,
			    ImRhok,
			    Residual_ReRhok,
			    Residual_ImRhok,
			    ReVk,
			    ImVk,
			    ReRhoAtomk,
			    ImRhoAtomk);
  }

  else if (Solver==4){

    DIIS_Mixing_Rhok_NEGF(
			  SCF_iter,
			  Mix_wgt,
			  ReRhok,
			  ImRhok,
			  Residual_ReRhok,
			  Residual_ImRhok,
			  ReVk,
			  ImVk,
			  ReRhoAtomk,
			  ImRhoAtomk);
  }

}




void DIIS_Mixing_Rhok_Normal(int SCF_iter,
			     double Mix_wgt,
			     double ***ReRhok,
			     double ***ImRhok,
			     double **Residual_ReRhok,
			     double **Residual_ImRhok,
			     double *ReVk,
			     double *ImVk,
			     double *ReRhoAtomk,
			     double *ImRhoAtomk)
{
  static int firsttime=1;
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan,k1,k2,k3,kk2; 
  int pSCF_iter,size_Re_OptRhok,size_Kerker_weight;
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
  double dum1,dum2,bunsi,bunbo,sum,coef_OptRho;
  double Gx,Gy,Gz,G2,G12,G22,G32,F,F0;
  double Min_Weight;
  double Max_Weight;
  double G0,G02,G02p,weight,wgt0,wgt1;
  double sk1,sk2,sk3;
  double **Re_OptRhok;
  double **Im_OptRhok;
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

  /* Re_OptRhok and Im_OptRhok */

  size_Re_OptRhok = spinmax*My_NumGridB_CB;

  Re_OptRhok = (double**)malloc(sizeof(double*)*spinmax); 
  for (spin=0; spin<spinmax; spin++){
    Re_OptRhok[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB); 
  }

  Im_OptRhok = (double**)malloc(sizeof(double*)*spinmax); 
  for (spin=0; spin<spinmax; spin++){
    Im_OptRhok[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB); 
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

    if (k1==0 && k2==0 && k3==0)  weight = 0.5;
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
      PrintMemory("DIIS_Mixing_Rhok: Re_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Im_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
      firsttime=0;
    }

    Kerker_Mixing_Rhok(0,1.00,ReRhok,ImRhok,
                       Residual_ReRhok,Residual_ImRhok,
                       ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
  } 

  else if (SCF_iter==2){

    if (measure_time==1) dtime(&Stime1);

    /* construct residual rho */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][2*My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][2*My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];

      } /* k */
    } /* spin */

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 = Etime1 - Stime1;      
    }

    /* rho1 to rho2 */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReRhok[2][spin][k] = ReRhok[1][spin][k];
	ImRhok[2][spin][k] = ImRhok[1][spin][k];
      }
    }

    Kerker_Mixing_Rhok(0,Mixing_weight,ReRhok,ImRhok,
                       Residual_ReRhok,Residual_ImRhok,
                       ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
  } 

  else {

    /****************************************************
                   construct residual rho1
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];

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
                    alpha from residual rho
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
                             Residual_ReRhok[spin], &K, 
                             Residual_ReRhok[spin], &K, 
                             &beta, 
                             My_NMat,
                             &M);

      F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K, 
                             &alpha, 
                             Residual_ImRhok[spin], &K, 
                             Residual_ImRhok[spin], &K, 
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
    NormRD[0] = A[0][0]/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3/(double)spinmax;
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
        Re_OptRhok[spin][k] = 0.0;
        Im_OptRhok[spin][k] = 0.0;
      }
    }   

    /* Pulay mixing for residual Rhok */

    for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

      tmp0 = alden[pSCF_iter]; 
      p0 = pSCF_iter*My_NumGridB_CB;

#pragma omp parallel shared(spinmax,My_NumGridB_CB,Re_OptRhok,Im_OptRhok,tmp0,p0,Residual_ReRhok,Residual_ImRhok)
      {

        int OMPID,Nthrds,k; 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();

	if (spinmax==1){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	    Re_OptRhok[0][k] += tmp0*Residual_ReRhok[0][p0+k];
	    Im_OptRhok[0][k] += tmp0*Residual_ImRhok[0][p0+k];
	  } /* k */
	}

	else if (spinmax==2){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	    Re_OptRhok[0][k] += tmp0*Residual_ReRhok[0][p0+k];
	    Im_OptRhok[0][k] += tmp0*Residual_ImRhok[0][p0+k];
	    Re_OptRhok[1][k] += tmp0*Residual_ReRhok[1][p0+k];
	    Im_OptRhok[1][k] += tmp0*Residual_ImRhok[1][p0+k];
	  } /* k */
	}

	else if (spinmax==3){
	  for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	    Re_OptRhok[0][k] += tmp0*Residual_ReRhok[0][p0+k];
	    Im_OptRhok[0][k] += tmp0*Residual_ImRhok[0][p0+k];
	    Re_OptRhok[1][k] += tmp0*Residual_ReRhok[1][p0+k];
	    Im_OptRhok[1][k] += tmp0*Residual_ImRhok[1][p0+k];
	    Re_OptRhok[2][k] += tmp0*Residual_ReRhok[2][p0+k];
	    Im_OptRhok[2][k] += tmp0*Residual_ImRhok[2][p0+k];
	  } /* k */
	}

      } /* #pragma omp parallel */

    } /* pSCF_iter */

    if      (1.0e-2<=NormRD[0])                        coef_OptRho = 0.5;
    else if (1.0e-4<=NormRD[0]  && NormRD[0]<1.0e-2)   coef_OptRho = 0.6;
    else if (1.0e-6<=NormRD[0]  && NormRD[0]<1.0e-4)   coef_OptRho = 0.7;
    else if (1.0e-8<=NormRD[0]  && NormRD[0]<1.0e-6)   coef_OptRho = 0.8;
    else if (1.0e-10<=NormRD[0] && NormRD[0]<1.0e-8)   coef_OptRho = 0.9;
    else                                               coef_OptRho = 1.0;

    /****************************************************
                        mixing of rho 
    ****************************************************/

    /****************************************************
       Pulay mixing
    ****************************************************/

    if ( SCF_iter%EveryPulay_SCF==0 ) {

      /* reduce Mixing_weight so that charge sloshing can be avoided at the next SCF */

      Mixing_weight = 0.1*Max_Mixing_weight;
      if (Mixing_weight<Min_Mixing_weight) Mixing_weight = Min_Mixing_weight;
      
      /* initial ReRhok and ImRhok */
      
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReRhok[0][spin][k] = 0.0;
	  ImRhok[0][spin][k] = 0.0;
	}
      }

      /* Pulay mixing */

      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

	tmp0 =  alden[pSCF_iter];

#pragma omp parallel shared(tmp0,pSCF_iter,spinmax,My_NumGridB_CB,ReRhok,ImRhok) 
	{

          int OMPID,Nthrds,k; 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();

	  if (spinmax==1){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReRhok[0][0][k] += tmp0*ReRhok[pSCF_iter][0][k];
	      ImRhok[0][0][k] += tmp0*ImRhok[pSCF_iter][0][k];
	    } /* k */
	  }

	  else if (spinmax==2){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReRhok[0][0][k] += tmp0*ReRhok[pSCF_iter][0][k];
	      ImRhok[0][0][k] += tmp0*ImRhok[pSCF_iter][0][k];
	      ReRhok[0][1][k] += tmp0*ReRhok[pSCF_iter][1][k];
	      ImRhok[0][1][k] += tmp0*ImRhok[pSCF_iter][1][k];
	    } /* k */
	  }

	  else if (spinmax==3){
  	    for (k=My_NumGridB_CB*OMPID/Nthrds; k<My_NumGridB_CB*(OMPID+1)/Nthrds; k++){
	      ReRhok[0][0][k] += tmp0*ReRhok[pSCF_iter][0][k];
	      ImRhok[0][0][k] += tmp0*ImRhok[pSCF_iter][0][k];
	      ReRhok[0][1][k] += tmp0*ReRhok[pSCF_iter][1][k];
	      ImRhok[0][1][k] += tmp0*ImRhok[pSCF_iter][1][k];
	      ReRhok[0][2][k] += tmp0*ReRhok[pSCF_iter][2][k];
	      ImRhok[0][2][k] += tmp0*ImRhok[pSCF_iter][2][k];
	    } /* k */
	  }

	} /* #pragma omp parallel */

      } /* pSCF_iter */

      /* Correction by optimized residual rho */

      for (spin=0; spin<spinmax; spin++){

        for (k=0; k<My_NumGridB_CB; k++){

 	  if (1.0<NormRD[0])
  	    weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
          else 
            weight = 1.0/Kerker_weight[k];

	  ReRhok[0][spin][k] += coef_OptRho*Re_OptRhok[spin][k]*weight;
	  ImRhok[0][spin][k] += coef_OptRho*Im_OptRhok[spin][k]*weight;

	  /* correction to large changing components */

          tmp0 = ReRhok[0][spin][k] - ReRhok[1][spin][k];  
          tmp1 = ImRhok[0][spin][k] - ImRhok[1][spin][k];  

          if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
            ReRhok[0][spin][k] = sgn(tmp0)*maxima_step + ReRhok[1][spin][k]; 
            ImRhok[0][spin][k] = sgn(tmp1)*maxima_step + ImRhok[1][spin][k];
          }

	} /* k */
      } /* spin */

      if (measure_time==1){ 
        dtime(&Etime1);
        time5 = Etime1 - Stime1;      
      }

    }

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
        
	  ReRhok[0][spin][k] = wgt0*ReRhok[0][spin][k] + wgt1*ReRhok[1][spin][k];
	  ImRhok[0][spin][k] = wgt0*ImRhok[0][spin][k] + wgt1*ImRhok[1][spin][k];
        
	} /* k */
      } /* spin */
    } /* else */

    /****************************************************
                         shift of rho
    ****************************************************/

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
	  ReRhok[pSCF_iter][spin][k] = ReRhok[pSCF_iter-1][spin][k]; 
	  ImRhok[pSCF_iter][spin][k] = ImRhok[pSCF_iter-1][spin][k]; 
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
          Residual_ReRhok[spin][p0+k] = Residual_ReRhok[spin][p1+k];
          Residual_ImRhok[spin][p0+k] = Residual_ImRhok[spin][p1+k]; 
	}
      }
    }
  } /* else */

  /****************************************************
        find the charge density in real space 
  ****************************************************/

  tmp0 = 1.0/(double)Ngrid1/(double)Ngrid2/(double)Ngrid3;

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][spin][k];
      ImVk[k] = ImRhok[0][spin][k];
    }

    if (spin==0 || spin==1){
      Get_Value_inReal(0,Density_Grid_B[spin],Density_Grid_B[spin],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Density_Grid_B[spin][BN_AB] = Density_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Density_Grid_B[2],Density_Grid_B[3],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Density_Grid_B[2][BN_AB] = Density_Grid_B[2][BN_AB]*tmp0;
	Density_Grid_B[3][BN_AB] = Density_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B);

  /****************************************************
      set ReV2 and ImV2 which are used in Poisson.c
  ****************************************************/

  if (SpinP_switch==0){
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = 2.0*ReRhok[0][0][k] - ReRhoAtomk[k];
      ImVk[k] = 2.0*ImRhok[0][0][k] - ImRhoAtomk[k];
    }
  }
  
  else {
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][0][k] + ReRhok[0][1][k] - ReRhoAtomk[k];
      ImVk[k] = ImRhok[0][0][k] + ImRhok[0][1][k] - ImRhoAtomk[k];
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
    free(Re_OptRhok[spin]);
  }
  free(Re_OptRhok);

  for (spin=0; spin<spinmax; spin++){
    free(Im_OptRhok[spin]);
  }
  free(Im_OptRhok);

  free(Kerker_weight);
}



void DIIS_Mixing_Rhok_NEGF(int SCF_iter,
			   double Mix_wgt,
			   double ***ReRhok,
			   double ***ImRhok,
			   double **Residual_ReRhok,
			   double **Residual_ImRhok,
			   double *ReVk,
			   double *ImVk,
			   double *ReRhoAtomk,
			   double *ImRhoAtomk)
{
  static int firsttime=1;
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan,n1,k1,k2,k3,kk2; 
  int pSCF_iter,size_Re_OptRhok,size_Kerker_weight;
  int spinmax,MN,flag_nan,M,N,K;
  int N2D,GN,GNs,N3[4],BN_AB,p0,p1;
  double time1,time2,time3,time4,time5; 
  double Stime1,Etime1;
  double *alden,alpha,beta;
  double **A,**My_A,**IA,*My_A_thrds;
  double *My_NMat,*NMat;
  double tmp0,itmp0,tmp1,im1,im2,re1,re2;
  double OptRNorm,My_OptRNorm,x,wt;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptRho;
  double Gx,Gy,Gz,G2,G12,G22,G32;
  double Min_Weight,Q,sumL,sumR;
  double Max_Weight;
  double G0,G02,G02p,weight,wgt0,wgt1;
  double sk1,sk2,sk3;
  double **Re_OptRhok;
  double **Im_OptRhok;
  double *Kerker_weight;
  int numprocs,myid;
  char nanchar[300];
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* find an optimum G0 */

  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G22<G32) G0 = G22;
  else         G0 = G32;

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

  My_NMat = (double*)malloc(sizeof(double)*List_YOUSO[38]*List_YOUSO[38]);
  NMat = (double*)malloc(sizeof(double)*List_YOUSO[38]*List_YOUSO[38]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  }

  My_A = (double**)malloc(sizeof(double*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    My_A[i] = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[38]);
  for (i=0; i<List_YOUSO[38]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[38]);
  }

  /* Re_OptRhok and Im_OptRhok */

  size_Re_OptRhok = spinmax*My_NumGridB_CB;

  Re_OptRhok = (double**)malloc(sizeof(double*)*spinmax); 
  for (spin=0; spin<spinmax; spin++){
    Re_OptRhok[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB); 
    for (k=0; k<My_NumGridB_CB; k++) Re_OptRhok[spin][k] = 0.0;
  }

  Im_OptRhok = (double**)malloc(sizeof(double*)*spinmax); 
  for (spin=0; spin<spinmax; spin++){
    Im_OptRhok[spin] = (double*)malloc(sizeof(double)*My_NumGridB_CB); 
    for (k=0; k<My_NumGridB_CB; k++) Im_OptRhok[spin][k] = 0.0;
  }

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = (double*)malloc(sizeof(double)*My_NumGridB_CB); 

  /***********************************
            set Kerker_weight 
  ************************************/

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  if (measure_time==1) dtime(&Stime1);

  for (k=0; k<My_NumGridB_CB; k+=Ngrid1){

    /* get k3 and k2 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;

    if (k2<Ngrid2/2)  sk2 = (double)k2;
    else              sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2)  sk3 = (double)k3;
    else              sk3 = (double)(k3 - Ngrid3);

    Gx = sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    for (k1=0; k1<Ngrid1; k1++){

      if (k2==0 && k3==0){

	sumL = 0.0;
	sumR = 0.0; 

	if (SpinP_switch==0){

	  for (n1=0; n1<k1; n1++){
	    sumL += 2.0*ReRhok[0][0][k+n1];
	  }

	  for (n1=k1+1; n1<Ngrid1; n1++){
	    sumR += 2.0*ReRhok[0][0][k+n1];
	  }
	}
        
	else if (SpinP_switch==1 || SpinP_switch==3){

	  for (n1=0; n1<k1; n1++){
	    sumL += ReRhok[0][0][k+n1] + ReRhok[0][1][k+n1];
	  }

	  for (n1=k1+1; n1<Ngrid1; n1++){
	    sumR += ReRhok[0][0][k+n1] + ReRhok[0][1][k+n1];
	  }
	}

	Q = 4.0*fabs(sumL - sumR)*GridVol + 1.0;

      }    
      else {
	Q = 1.0;
      }

      weight = (G2 + G02)/(G2 + G02p)*Q;
      Kerker_weight[k+k1] = sqrt(weight);

    } /* k1 */
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
      PrintMemory("DIIS_Mixing_Rhok: Re_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Im_OptRhok",sizeof(double)*size_Re_OptRhok,NULL);
      PrintMemory("DIIS_Mixing_Rhok: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
      firsttime=0;
    }

    Kerker_Mixing_Rhok(0, 1.0, ReRhok, ImRhok,
                       Residual_ReRhok, Residual_ImRhok,
                       ReVk,ImVk, ReRhoAtomk, ImRhoAtomk);
  } 

  else if (SCF_iter==2){

    if (measure_time==1) dtime(&Stime1);

    /* construct residual rho */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][2*My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][2*My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];

      } /* k */
    }

    if (measure_time==1){ 
      dtime(&Etime1);
      time2 = Etime1 - Stime1;      
    }

    /* rho1 to rho2 */

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReRhok[2][spin][k] = ReRhok[1][spin][k];
	ImRhok[2][spin][k] = ImRhok[1][spin][k];
      }
    }

    Kerker_Mixing_Rhok(0, Mixing_weight, ReRhok, ImRhok,
                       Residual_ReRhok, Residual_ImRhok,
                       ReVk,ImVk, ReRhoAtomk, ImRhoAtomk);
  }

  else {

    /****************************************************
                   construct residual rho1
    ****************************************************/

    if (measure_time==1) dtime(&Stime1);

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){

	Residual_ReRhok[spin][My_NumGridB_CB+k] = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*Kerker_weight[k];
	Residual_ImRhok[spin][My_NumGridB_CB+k] = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*Kerker_weight[k];
      }
    }

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
                    alpha from residual rho
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
                             Residual_ReRhok[spin], &K, 
                             Residual_ReRhok[spin], &K, 
                             &beta, 
                             My_NMat,
                             &M);

      F77_NAME(dgemm,DGEMM)( "T","N", &M, &N, &K, 
                             &alpha, 
                             Residual_ImRhok[spin], &K, 
                             Residual_ImRhok[spin], &K, 
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

    /**********************************
              original code
    **********************************/

    /*
    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=SCFi; SCFj<=NumMix; SCFj++){ 

        tmp0 = 0.0; 
	for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){

	    re1 = Residual_ReRhok[spin][SCFi*My_NumGridB_CB+k];
	    im1 = Residual_ImRhok[spin][SCFi*My_NumGridB_CB+k];

	    re2 = Residual_ReRhok[spin][SCFj*My_NumGridB_CB+k];
	    im2 = Residual_ImRhok[spin][SCFj*My_NumGridB_CB+k];

	    tmp0 += re1*re2 + im1*im2;
	  }
	}

	My_A[SCFi][SCFj] = tmp0;

      }
    }

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=SCFi; SCFj<=NumMix; SCFj++){
        MPI_Allreduce(&My_A[SCFi][SCFj], &A[SCFi-1][SCFj-1],
                      1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
        A[SCFj-1][SCFi-1] = A[SCFi-1][SCFj-1];
      }
    }
    */

    if (measure_time==1){ 
      dtime(&Etime1);
      time4 = Etime1 - Stime1;      
    }

    /* store NormRD */

    for (i=4; 1<=i; i--){
      NormRD[i] = NormRD[i-1];
      History_Uele[i] = History_Uele[i-1];
    }
    NormRD[0] = A[0][0];
    History_Uele[0] = Uele;

    /* solve the linear equation */

    for (SCFi=1; SCFi<=NumMix; SCFi++){
       A[SCFi-1][NumMix] = -1.0;
       A[NumMix][SCFi-1] = -1.0;
    }
    A[NumMix][NumMix] = 0.0;

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

    for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

      tmp0 =  alden[pSCF_iter]; 
      p0 = pSCF_iter*My_NumGridB_CB;

      for (spin=0; spin<spinmax; spin++){
	for (k=0; k<My_NumGridB_CB; k++){

	  /* Pulay mixing */

	  Re_OptRhok[spin][k] += tmp0*Residual_ReRhok[spin][p0+k];
	  Im_OptRhok[spin][k] += tmp0*Residual_ImRhok[spin][p0+k];

	} /* k */
      } /* spin */

    } /* pSCF_iter */

    if      (1.0e-2<=NormRD[0])                        coef_OptRho = 0.5;
    else if (1.0e-4<=NormRD[0]  && NormRD[0]<1.0e-2)   coef_OptRho = 0.6;
    else if (1.0e-6<=NormRD[0]  && NormRD[0]<1.0e-4)   coef_OptRho = 0.7;
    else if (1.0e-8<=NormRD[0]  && NormRD[0]<1.0e-6)   coef_OptRho = 0.8;
    else if (1.0e-10<=NormRD[0] && NormRD[0]<1.0e-8)   coef_OptRho = 0.9;
    else                                               coef_OptRho = 1.0;

    /****************************************************
                        mixing of rho 
    ****************************************************/

    /****************************************************
       Pulay mixing
    ****************************************************/

    if ( SCF_iter%EveryPulay_SCF==0 ) {

      /* reduce Mixing_weight so that charge sloshing can be avoided at the next SCF */

      Mixing_weight = 0.1*Max_Mixing_weight;
      if (Mixing_weight<Min_Mixing_weight) Mixing_weight = Min_Mixing_weight;

      /* initial ReRhok and ImRhok */

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
          ReRhok[0][spin][k] = 0.0;
          ImRhok[0][spin][k] = 0.0;
	}
      }

      /* Pulay mixing */

      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){

	tmp0 =  alden[pSCF_iter];

        for (spin=0; spin<spinmax; spin++){
          for (k=0; k<My_NumGridB_CB; k++){

	    ReRhok[0][spin][k] += tmp0*ReRhok[pSCF_iter][spin][k];
	    ImRhok[0][spin][k] += tmp0*ImRhok[pSCF_iter][spin][k];

	  } /* k */
	} /* spin */
      } /* pSCF_iter */

      /* Correction by optimized residual rho */

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
 
	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);

	  ReRhok[0][spin][k] += coef_OptRho*Re_OptRhok[spin][k]*weight;
	  ImRhok[0][spin][k] += coef_OptRho*Im_OptRhok[spin][k]*weight;

	  /* correction to large changing components */

          tmp0 = ReRhok[0][spin][k] - ReRhok[1][spin][k];  
          tmp1 = ImRhok[0][spin][k] - ImRhok[1][spin][k];  

          if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
            ReRhok[0][spin][k] = sgn(tmp0)*maxima_step + ReRhok[1][spin][k]; 
            ImRhok[0][spin][k] = sgn(tmp1)*maxima_step + ImRhok[1][spin][k]; 
          }

	} /* k */
      } /* spin */

      if (measure_time==1){ 
        dtime(&Etime1);
        time5 = Etime1 - Stime1;      
      }

    }

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

      /* Kerer mixing */

      Mix_wgt = Mixing_weight;

      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){

	  weight = 1.0/(Kerker_weight[k]*Kerker_weight[k]);
	  wgt0  = Mix_wgt*weight;
	  wgt1 =  1.0 - wgt0;

	  ReRhok[0][spin][k] = wgt0*ReRhok[0][spin][k] + wgt1*ReRhok[1][spin][k];
	  ImRhok[0][spin][k] = wgt0*ImRhok[0][spin][k] + wgt1*ImRhok[1][spin][k];

	} /* k */
      } /* spin */
    } /* else */

    /****************************************************
                        shift of rho
    ****************************************************/

    for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<spinmax; spin++){
        for (k=0; k<My_NumGridB_CB; k++){
	  ReRhok[pSCF_iter][spin][k] = ReRhok[pSCF_iter-1][spin][k]; 
	  ImRhok[pSCF_iter][spin][k] = ImRhok[pSCF_iter-1][spin][k]; 
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
	  Residual_ReRhok[spin][p0+k] = Residual_ReRhok[spin][p1+k];
	  Residual_ImRhok[spin][p0+k] = Residual_ImRhok[spin][p1+k]; 
	}
      }
    }

  } /* else */

  /****************************************************
        find the charge density in real space 
  ****************************************************/

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][spin][k];
      ImVk[k] = ImRhok[0][spin][k];
    }
/* revised by Y. Xiao for Noncollinear NEGF calculations */
  if (spin==0 || spin==1) {
    Get_Value_inReal2D(0, Density_Grid_B[spin], NULL, ReVk, ImVk);
  } else {
    Get_Value_inReal2D(1, Density_Grid_B[2], Density_Grid_B[3], ReVk, ImVk);
  }
/* until here by Y. Xiao for Noncollinear NEGF calculations */

/*    Get_Value_inReal2D(0, Density_Grid_B[spin], NULL, ReVk, ImVk); */
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B);

  if (measure_time==1){ 
    printf("time1=%12.6f time2=%12.6f time3=%12.6f time4=%12.6f time5=%12.6f\n",
            time1,time2,time3,time4,time5);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/
  
  free(alden);

  for (i=0; i<List_YOUSO[38]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[38]; i++){
    free(My_A[i]);
  }
  free(My_A);

  for (i=0; i<List_YOUSO[38]; i++){
    free(IA[i]);
  }
  free(IA);

  free(My_NMat);
  free(NMat);

  for (spin=0; spin<spinmax; spin++){
    free(Re_OptRhok[spin]);
  }
  free(Re_OptRhok);

  for (spin=0; spin<spinmax; spin++){
    free(Im_OptRhok[spin]);
  }
  free(Im_OptRhok);

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

    /*
    printf("N=%2d\n",N);

    N = 3;

    a[0][0] =-1.2; a[0][1] =-1.0; a[0][2] = -0.0;
    a[1][0] =-1.0; a[1][1] =-1.2; a[1][2] = -1.0;
    a[2][0] =-0.0; a[2][1] =-1.0; a[2][2] = -1.2;
    */

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        A[j*N+i] = a[i][j];
      }
    }

    Eigen_lapack3(A, ko, N, N); 

    /*
    printf("ko\n");
    for (i=0; i<N; i++){
      printf("ko i=%2d %15.12f\n",i,ko[i]);
    } 

    printf("A\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%10.5f ",A[i*N+j]); 
      }
      printf("\n");
    }
    */

    for (i=0; i<N; i++){
      /*
      printf("i=%2d ko=%18.15f iko=%18.15f\n",i,ko[i],1.0/ko[i]);
      */

      ko[i] = 1.0/(ko[i]+1.0e-13);
    } 

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        B[i*N+j] = A[i*N+j]*ko[i];
      }
    }

    /*
    printf("A*ko\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%10.5f ",B[i*N+j]); 
      }
      printf("\n");
    }
    */

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

    /*	
    printf("ia\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%10.5f ",ia[i][j]);
      }
      printf("\n");
    }

    MPI_Finalize();
    exit(0);
    */

    free(A);
    free(B);
    free(ko);
  }


}


void Inverse2(int n, double **a, double **ia)
{
  int N,i,j,k;
  double *A,*B,*ko;
  double sum;

  N = n + 1;

  A = (double*)malloc(sizeof(double)*(N+2)*(N+2));
  B = (double*)malloc(sizeof(double)*(N+2)*(N+2));
  ko = (double*)malloc(sizeof(double)*(N+2));

  /*
    printf("N=%2d\n",N);

    N = 3;

    a[0][0] =-1.2; a[0][1] =-1.0; a[0][2] = -0.0;
    a[1][0] =-1.0; a[1][1] =-1.2; a[1][2] = -1.0;
    a[2][0] =-0.0; a[2][1] =-1.0; a[2][2] = -1.2;
  */

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      A[j*N+i] = a[i][j];
    }
  }

  Eigen_lapack3(A, ko, N, N); 

  /*
    printf("ko\n");
    for (i=0; i<N; i++){
    printf("ko i=%2d %15.12f\n",i,ko[i]);
    } 

    printf("A\n");
    for (i=0; i<N; i++){
    for (j=0; j<N; j++){
    printf("%10.5f ",A[i*N+j]); 
    }
    printf("\n");
    }
  */

  for (i=0; i<N; i++){

    printf("i=%2d ko=%18.15f iko=%18.15f\n",i,ko[i],1.0/ko[i]);

    ko[i] = 1.0/ko[i];
  } 
 
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      B[i*N+j] = A[i*N+j]*ko[i];
    }
  }

  /*
    printf("A*ko\n");
    for (i=0; i<N; i++){
    for (j=0; j<N; j++){
    printf("%10.5f ",B[i*N+j]); 
    }
    printf("\n");
    }
  */

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

  /*	
    printf("ia\n");
    for (i=0; i<N; i++){
    for (j=0; j<N; j++){
    printf("%10.5f ",ia[i][j]);
    }
    printf("\n");
    }

    MPI_Finalize();
    exit(0);
  */

  free(A);
  free(B);
  free(ko);

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
