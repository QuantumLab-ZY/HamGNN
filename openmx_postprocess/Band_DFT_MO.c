/**********************************************************************
  Band_DFT_MO.c:

     Band_DFT_MO.c is a subroutine to calculate wave functions
     at given k-points for the file output.

  Log of Band_DFT_MO.c:

     15/May/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>


static void Band_DFT_MO_Col(
                      int nkpoint, double **kpoint,
                      int SpinP_switch, 
                      double *****nh,
                      double ****CntOLP);

static void Band_DFT_MO_NonCol(
                      int nkpoint, double **kpoint,
                      int SpinP_switch, 
                      double *****nh,
                      double *****ImNL,
                      double ****CntOLP);


void Band_DFT_MO( int nkpoint, double **kpoint,
                  int SpinP_switch, 
                  double *****nh,
                  double *****ImNL,
                  double ****CntOLP)
{
  if (SpinP_switch==0 || SpinP_switch==1){
    Band_DFT_MO_Col( nkpoint, kpoint, SpinP_switch, nh, CntOLP);
  }
  else if (SpinP_switch==3){
    Band_DFT_MO_NonCol( nkpoint, kpoint, SpinP_switch, nh, ImNL, CntOLP);
  }
}



static void Band_DFT_MO_Col(
                      int nkpoint, double **kpoint,
                      int SpinP_switch, 
                      double *****nh,
                      double ****CntOLP)
{
  int i,j,k,l,n,wan;
  int *MP,*order_GA,*My_NZeros,*SP_NZeros,*SP_Atoms;
  int i1,j1,po,spin,n1,size_H1;
  int num2,RnB,l1,l2,l3,kloop;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,nhomos,nlumos;
  int ii,ij,ik,Rn,AN;
  int num0,num1,mul,m,wan1,Gc_AN;
  int LB_AN,GB_AN,Bnum;
  double time0,tmp,tmp1,av_num;
  double snum_i,snum_j,snum_k,k1,k2,k3,sum,sumi,Num_State,FermiF;
  double x,Dnum,Dnum2,AcP,EV_cut0;
  double **ko,*M1,***EIGEN;
  double *koS;
  double *S1,**H1,*B;
  dcomplex ***H,**S,***C;
  dcomplex Ctmp1,Ctmp2;
  double u2,v2,uv,vu;
  double dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2,Imdum;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double FermiEps = 1.0e-14;
  double x_cut = 30.0;
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;
  int *is1,*ie1;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;
  char operate[300];
  FILE *fp1;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout) printf("\nBand_DFT_MO start\n");fflush(stdout);

  dtime(&TStime);

  /* open a file pointer to store LCAO coefficients */

  if (myid==Host_ID){
    sprintf(operate,"%s%s.coes",filepath,filename);
    fp1 = fopen(operate, "ab");

    if (fp1!=NULL){
      remove(operate); 
      fclose(fp1); 
      fp1 = fopen(operate, "ab");
    }
  }

  /****************************************************
                  allocation of arrays
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));
  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);
  
  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }

  ko = (double**)malloc(sizeof(double*)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    ko[i] = (double*)malloc(sizeof(double)*(n+1));
  }

  koS = (double*)malloc(sizeof(double)*(n+1));

  EIGEN = (double***)malloc(sizeof(double**)*List_YOUSO[33]);
  for (i=0; i<List_YOUSO[33]; i++){
    EIGEN[i] = (double**)malloc(sizeof(double*)*List_YOUSO[23]);
    for (j=0; j<List_YOUSO[23]; j++){
      EIGEN[i][j] = (double*)malloc(sizeof(double)*(n+1));
    }
  }

  H = (dcomplex***)malloc(sizeof(dcomplex**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    H[i] = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
    for (j=0; j<n+1; j++){
      H[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
    }
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
  for (i=0; i<n+1; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
  }

  M1 = (double*)malloc(sizeof(double)*(n+1));

  C = (dcomplex***)malloc(sizeof(dcomplex**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    C[i] = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
    for (j=0; j<n+1; j++){
      C[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
    }
  }

  B = (double*)malloc(sizeof(double)*(n+1)*(n+1)*2); 

  /*****************************************************
        allocation of arrays for parallelization 
  *****************************************************/

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /* find size_H1 */
  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /* allocation of S1 and H1 */
  S1 = (double*)malloc(sizeof(double)*size_H1);
  H1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    H1[spin] = (double*)malloc(sizeof(double)*size_H1);
  }

  /* Get S1 */
  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SpinP_switch==0){ 
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1[0], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1[0], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, nh[1], H1[1], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  dtime(&SiloopTime);

  /*****************************************************
         Solve eigenvalue problem at each k-point
  *****************************************************/

  for (kloop=0; kloop<nkpoint; kloop++){

    if (myid==Host_ID && 0<level_stdout) printf("kpoint=%i /%i\n",kloop+1,nkpoint);

    k1 = kpoint[kloop][1];
    k2 = kpoint[kloop][2];
    k3 = kpoint[kloop][3];

    /* make S */

    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	S[i1][j1] = Complex(0.0,0.0);
      } 
    } 

    k = 0;
    for (AN=1; AN<=atomnum; AN++){
      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    S[Anum+i][Bnum+j].r += S1[k]*co;
	    S[Anum+i][Bnum+j].i += S1[k]*si;

	    k++;
	  }
	}
      }
    }

    /* diagonalization of S */
    Eigen_PHH(mpi_comm_level1,S,koS,n,n,1);

    if (3<=level_stdout){
      printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",kloop,k1,k2,k3);
      for (i1=1; i1<=n; i1++){
	printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,koS[i1]);
      }
    }

    /* minus eigenvalues to 1.0e-14 */

    for (l=1; l<=n; l++){
      if (koS[l]<0.0) koS[l] = 1.0e-14;
    }

    /* calculate S*1/sqrt(koS) */

    for (l=1; l<=n; l++) M1[l] = 1.0/sqrt(koS[l]);

    /* S * M1  */

    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	S[i1][j1].r = S[i1][j1].r*M1[j1];
	S[i1][j1].i = S[i1][j1].i*M1[j1];
      } 
    } 

    /* loop for spin */

    for (spin=0; spin<=SpinP_switch; spin++){

      /* make H */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  H[spin][i1][j1] = Complex(0.0,0.0);
	} 
      } 

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){

	    for (j=0; j<tnoB; j++){

	      H[spin][Anum+i][Bnum+j].r += H1[spin][k]*co;
	      H[spin][Anum+i][Bnum+j].i += H1[spin][k]*si;

	      k++;

	    }
	  }
	}
      }

      /* first transpose of S */

      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /****************************************************
                      M1 * U^t * H * U * M1
      ****************************************************/

      /* H * U * M1 */

#pragma omp parallel for shared(spin,n,myid,is1,ie1,S,H,C) private(i1,j1,l) 

      for (j1=is1[myid]; j1<=ie1[myid]; j1++){

	for (i1=1; i1<=(n-1); i1+=2){

	  double sum0  = 0.0, sum1  = 0.0;
	  double sumi0 = 0.0, sumi1 = 0.0;

	  for (l=1; l<=n; l++){
	    sum0  += H[spin][i1+0][l].r*S[j1][l].r - H[spin][i1+0][l].i*S[j1][l].i;
	    sum1  += H[spin][i1+1][l].r*S[j1][l].r - H[spin][i1+1][l].i*S[j1][l].i;

	    sumi0 += H[spin][i1+0][l].r*S[j1][l].i + H[spin][i1+0][l].i*S[j1][l].r;
	    sumi1 += H[spin][i1+1][l].r*S[j1][l].i + H[spin][i1+1][l].i*S[j1][l].r;
	  }

	  C[spin][j1][i1+0].r = sum0;
	  C[spin][j1][i1+1].r = sum1;

	  C[spin][j1][i1+0].i = sumi0;
	  C[spin][j1][i1+1].i = sumi1;
	}

	for (; i1<=n; i1++){

	  double sum  = 0.0;
	  double sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  += H[spin][i1][l].r*S[j1][l].r - H[spin][i1][l].i*S[j1][l].i;
	    sumi += H[spin][i1][l].r*S[j1][l].i + H[spin][i1][l].i*S[j1][l].r;
	  }

	  C[spin][j1][i1].r = sum;
	  C[spin][j1][i1].i = sumi;
	}

      } /* i1 */ 

      /* M1 * U^+ H * U * M1 */

#pragma omp parallel for shared(spin,n,is1,ie1,myid,S,H,C) private(i1,j1,l)  

      for (i1=1; i1<=n; i1++){
        for (j1=is1[myid]; j1<=ie1[myid]; j1++){
  
	  double sum  = 0.0;
	  double sumi = 0.0;

	  for (l=1; l<=n; l++){
	    sum  +=  S[i1][l].r*C[spin][j1][l].r + S[i1][l].i*C[spin][j1][l].i;
	    sumi +=  S[i1][l].r*C[spin][j1][l].i - S[i1][l].i*C[spin][j1][l].r;
	  }

	  H[spin][j1][i1].r = sum;
	  H[spin][j1][i1].i = sumi;

	}
      } 

      /* broadcast H */

      BroadCast_ComplexMatrix(mpi_comm_level1,H[spin],n,is1,ie1,myid,numprocs,
                              stat_send,request_send,request_recv);

      /* H to C */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  C[spin][j1][i1] = H[spin][i1][j1];
	}
      }

      /* penalty for ill-conditioning states */

      EV_cut0 = 1.0e-9;

      for (i1=1; i1<=n; i1++){

	if (koS[i1]<EV_cut0){
	  C[spin][i1][i1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	}
 
	/* cutoff the interaction between the ill-conditioned state */
 
	if (1.0e+3<C[spin][i1][i1].r){
	  for (j1=1; j1<=n; j1++){
	    C[spin][i1][j1] = Complex(0.0,0.0);
	    C[spin][j1][i1] = Complex(0.0,0.0);
	  }
	  C[spin][i1][i1].r = 1.0e+4;
	}
      }

      /* diagonalization of C */
      Eigen_PHH(mpi_comm_level1,C[spin],ko[spin],n,n,0);

      if (3<=level_stdout){
	printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",kloop,k1,k2,k3);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of Hks  %2d  %15.12f\n",i1,ko[spin][i1]);
	}
      }

      for (i1=1; i1<=n; i1++){
        EIGEN[kloop][spin][i1] = ko[spin][i1];
      }

      /****************************************************
          transformation to the original eigenvectors.
                 NOTE JRCAT-244p and JAIST-2122p 
      ****************************************************/

      /*  The H matrix is distributed by row */

      for (i1=1; i1<=n; i1++){
	for (j1=is1[myid]; j1<=ie1[myid]; j1++){
	  H[spin][j1][i1] = C[spin][i1][j1];
	}
      }

      /* second transpose of S */

      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /* C is distributed by row in each processor */

#pragma omp parallel for shared(spin,n,is1,ie1,myid,S,H,C) private(i1,j1,l,sum,sumi)  

      for (j1=is1[myid]; j1<=ie1[myid]; j1++){
        for (i1=1; i1<=n; i1++){

	  sum  = 0.0;
	  sumi = 0.0;
	  for (l=1; l<=n; l++){
	    sum  += S[i1][l].r*H[spin][j1][l].r - S[i1][l].i*H[spin][j1][l].i;
	    sumi += S[i1][l].r*H[spin][j1][l].i + S[i1][l].i*H[spin][j1][l].r;
	  }

	  C[spin][j1][i1].r = sum;
	  C[spin][j1][i1].i = sumi;
	}
      }

      /* broadcast C:
       C is distributed by row in each processor
      */

      BroadCast_ComplexMatrix(mpi_comm_level1,C[spin],n,is1,ie1,myid,numprocs,
			      stat_send,request_send,request_recv);

      /* find HOMO from eigenvalues */

      Bulk_HOMO[kloop][spin] = 0; 

      for (i1=1; i1<=n; i1++){
	x = (ko[spin][i1] - ChemP)*Beta;
	if (x<=-x_cut) x = -x_cut;
	if (x_cut<=x)  x = x_cut;

	FermiF = 1.0/(1.0 + exp(x));
	if (0.5<FermiF) Bulk_HOMO[kloop][spin] = i1;
      }      

    } /* spin */

    if (myid==Host_ID && SpinP_switch==0 && 2<=level_stdout){
      printf("k1=%7.3f k2=%7.3f k3=%7.3f  HOMO = %2d\n",
	     k1,k2,k3,Bulk_HOMO[kloop][0]);
    }
    else if (myid==Host_ID && SpinP_switch==1 && 2<=level_stdout){
      printf("k1=%7.3f k2=%7.3f k3=%7.3f  HOMO for up-spin   = %2d\n",
	     k1,k2,k3,Bulk_HOMO[kloop][0]);
      printf("k1=%7.3f k2=%7.3f k3=%7.3f  HOMO for down-spin = %2d\n",
	     k1,k2,k3,Bulk_HOMO[kloop][1]);
    }

    /****************************************************
        LCAO coefficients are stored for calculating
                 values of MOs on grids
    ****************************************************/

    nhomos = num_HOMOs;
    nlumos = num_LUMOs;

    if (SpinP_switch==0){
      if ( (Bulk_HOMO[kloop][0]-nhomos+1)<1 ) nhomos = Bulk_HOMO[kloop][0];
      if ( (Bulk_HOMO[kloop][0]+nlumos)>n )   nlumos = n - Bulk_HOMO[kloop][0];
    }
    else if (SpinP_switch==1){
      if ( (Bulk_HOMO[kloop][0]-nhomos+1)<1 ) nhomos = Bulk_HOMO[kloop][0];
      if ( (Bulk_HOMO[kloop][1]-nhomos+1)<1 ) nhomos = Bulk_HOMO[kloop][1];
      if ( (Bulk_HOMO[kloop][0]+nlumos)>n )   nlumos = n - Bulk_HOMO[kloop][0];
      if ( (Bulk_HOMO[kloop][1]+nlumos)>n )   nlumos = n - Bulk_HOMO[kloop][1];
    }

    /* HOMOs */
    for (spin=0; spin<=SpinP_switch; spin++){
      for (j=0; j<nhomos; j++){

        j1 = Bulk_HOMO[kloop][spin] - j;

        /* store eigenvalues */
        HOMOs_Coef[kloop][spin][j][0][0].r = EIGEN[kloop][spin][j1];

        /* store eigenvector */
        for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];
          for (i=0; i<tnoA; i++){
            HOMOs_Coef[kloop][spin][j][GA_AN][i].r = C[spin][j1][Anum+i].r;
            HOMOs_Coef[kloop][spin][j][GA_AN][i].i = C[spin][j1][Anum+i].i;
          }
        }
      }        
    }

    /* LUMOs */
    for (spin=0; spin<=SpinP_switch; spin++){
      for (j=0; j<nlumos; j++){

        j1 = Bulk_HOMO[kloop][spin] + 1 + j;

        /* store eigenvalue */
        LUMOs_Coef[kloop][spin][j][0][0].r = EIGEN[kloop][spin][j1];

        /* store eigenvector */
        for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];
          for (i=0; i<tnoA; i++){
            LUMOs_Coef[kloop][spin][j][GA_AN][i].r = C[spin][j1][Anum+i].r;
            LUMOs_Coef[kloop][spin][j][GA_AN][i].i = C[spin][j1][Anum+i].i;
          }
        }
      }
    }

    Bulk_Num_HOMOs[kloop] = nhomos;
    Bulk_Num_LUMOs[kloop] = nlumos;

    /****************************************************
                          Output
    ****************************************************/

    if (myid==Host_ID){

      strcpy(file_EV,".EV");
      fnjoint(filepath,filename,file_EV);

      if ((fp_EV = fopen(file_EV,"a")) != NULL){

#ifdef xt3
        setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

        if (kloop==0){

  	  fprintf(fp_EV,"\n");
	  fprintf(fp_EV,"***********************************************************\n");
	  fprintf(fp_EV,"***********************************************************\n");
	  fprintf(fp_EV,"        Eigenvalues (Hartree) and LCAO coefficients        \n");
	  fprintf(fp_EV,"        at the k-points specified in the input file.       \n");
	  fprintf(fp_EV,"***********************************************************\n");
	  fprintf(fp_EV,"***********************************************************\n");
	}

	k1 = kpoint[kloop][1];
	k2 = kpoint[kloop][2];
	k3 = kpoint[kloop][3];

	fprintf(fp_EV,"\n\n");
	fprintf(fp_EV,"   # of k-point = %i\n",kloop+1);
	fprintf(fp_EV,"   k1=%10.5f k2=%10.5f k3=%10.5f\n\n",k1,k2,k3);
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);

	if (SpinP_switch==0){
	  fprintf(fp_EV,"   HOMO = %i\n\n",Bulk_HOMO[kloop][0]);
	}
	else if (SpinP_switch==1){
	  fprintf(fp_EV,"   HOMO for up-spin   = %i\n",  Bulk_HOMO[kloop][0]);
	  fprintf(fp_EV,"   HOMO for down-spin = %i\n\n",Bulk_HOMO[kloop][1]);
	}

	fprintf(fp_EV,"   Real (Re) and imaginary (Im) parts of LCAO coefficients\n\n");

	num0 = 4;
	num1 = n/num0 + 1*(n%num0!=0);

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=1; i<=num1; i++){

	    fprintf(fp_EV,"\n");

	    for (i1=-2; i1<=0; i1++){

	      fprintf(fp_EV,"                     ");

	      for (j=1; j<=num0; j++){

		j1 = num0*(i-1) + j;

		if (j1<=n){ 

		  if (i1==-2){
		    if (spin==0){
		      fprintf(fp_EV,"  %4d (U)",j1);
		      fprintf(fp_EV,"          ");
		    }
		    else if (spin==1){
		      fprintf(fp_EV,"  %4d (D)",j1);
		      fprintf(fp_EV,"          ");
		    }
		  }

		  else if (i1==-1){
		    fprintf(fp_EV,"  %8.4f",EIGEN[kloop][spin][j1]);
		    fprintf(fp_EV,"          ");
		  }

		  else if (i1==0){
		    fprintf(fp_EV,"     Re   ");
		    fprintf(fp_EV,"     Im   ");
		  }
		}
	      }
	      fprintf(fp_EV,"\n");
	      if (i1==-1)  fprintf(fp_EV,"\n");
	      if (i1==0)   fprintf(fp_EV,"\n");
	    }

	    Name_Angular[0][0] = "s          ";
	    Name_Angular[1][0] = "px         ";
	    Name_Angular[1][1] = "py         ";
	    Name_Angular[1][2] = "pz         ";
	    Name_Angular[2][0] = "d3z^2-r^2  ";
	    Name_Angular[2][1] = "dx^2-y^2   ";
	    Name_Angular[2][2] = "dxy        ";
	    Name_Angular[2][3] = "dxz        ";
	    Name_Angular[2][4] = "dyz        ";
	    Name_Angular[3][0] = "f5z^2-3r^2 ";
	    Name_Angular[3][1] = "f5xz^2-xr^2";
	    Name_Angular[3][2] = "f5yz^2-yr^2";
	    Name_Angular[3][3] = "fzx^2-zy^2 ";
	    Name_Angular[3][4] = "fxyz       ";
	    Name_Angular[3][5] = "fx^3-3*xy^2";
	    Name_Angular[3][6] = "f3yx^2-y^3 ";
	    Name_Angular[4][0] = "g1         ";
	    Name_Angular[4][1] = "g2         ";
	    Name_Angular[4][2] = "g3         ";
	    Name_Angular[4][3] = "g4         ";
	    Name_Angular[4][4] = "g5         ";
	    Name_Angular[4][5] = "g6         ";
	    Name_Angular[4][6] = "g7         ";
	    Name_Angular[4][7] = "g8         ";
	    Name_Angular[4][8] = "g9         ";

	    Name_Multiple[0] = "0";
	    Name_Multiple[1] = "1";
	    Name_Multiple[2] = "2";
	    Name_Multiple[3] = "3";
	    Name_Multiple[4] = "4";
	    Name_Multiple[5] = "5";

	    i1 = 1; 

	    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	      wan1 = WhatSpecies[Gc_AN];
            
	      for (l=0; l<=Supported_MaxL; l++){
		for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		  for (m=0; m<(2*l+1); m++){

		    if (l==0 && mul==0 && m==0)
		      fprintf(fp_EV,"%4d %3s %s %s", 
			      Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
		    else
		      fprintf(fp_EV,"         %s %s", 
			      Name_Multiple[mul],Name_Angular[l][m]);

		    for (j=1; j<=num0; j++){

		      j1 = num0*(i-1) + j;

		      if (0<i1 && j1<=n){
			fprintf(fp_EV,"  %8.5f",C[spin][j1][i1].r);
			fprintf(fp_EV,"  %8.5f",C[spin][j1][i1].i);
		      }
		    }
		    fprintf(fp_EV,"\n");

		    i1++;
		  }
		}
	      }
	    }

	  }
	}

        /* close the file */ 
	fclose(fp_EV);
      }
      else{
	printf("Failure of saving the EV file.\n");
	fclose(fp_EV);
      }

    } /* if (myid==Host_ID) */

    /****************************************************
                 storing LCAO coefficients
    ****************************************************/

    if (myid==Host_ID){

      double *FermiF_array;

      FermiF_array  = (double*)malloc(sizeof(double)*(n+1));

      for (spin=0; spin<=SpinP_switch; spin++){

        /* k-point */

        fwrite(&k1, sizeof(double), 1, fp1);
        fwrite(&k2, sizeof(double), 1, fp1);
        fwrite(&k3, sizeof(double), 1, fp1);

        /* total energy */

        fwrite(&Utot, sizeof(double), 1, fp1);

        /* FermiF_array */

        for (i1=1; i1<=n; i1++){
  	  x = (ko[spin][i1] - ChemP)*Beta;
          if (x<=-x_cut) x = -x_cut;
          if (x_cut<=x)  x = x_cut;
	  FermiF_array[i1] = FermiFunc(x,spin,i1,&ii,&tmp);
	}

        fwrite(FermiF_array, sizeof(double), (n+1), fp1);

        /* B */

        k = 0; 
        for (i1=0; i1<=n; i1++){
	  for (j1=0; j1<=n; j1++){
	    B[k] = C[spin][i1][j1].r; k++;
	    B[k] = C[spin][i1][j1].i; k++;
	  }
	}

	/*
        for (i1=1; i1<=n; i1++){
	  for (j1=1; j1<=n; j1++){
            printf("%10.5f ",B[i1*2*(n+1)+2*j1]);  
	  }
          printf("\n");
	}
	*/

        fwrite(B, sizeof(double), (n+1)*(n+1)*2, fp1);
      }

      free(FermiF_array);
    }

  }  /* kloop */

  /* fclose the file pointer to store LCAO coefficients */

  if (myid==Host_ID){
    fclose(fp1);
  }

  /****************************************************
                       free arrays
  ****************************************************/

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);

  free(MP);
  free(order_GA);
  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (i=0; i<List_YOUSO[23]; i++){
    free(ko[i]);
  }
  free(ko);

  free(koS);

  for (i=0; i<List_YOUSO[33]; i++){
    for (j=0; j<List_YOUSO[23]; j++){
      free(EIGEN[i][j]);
    }
    free(EIGEN[i]);
  }
  free(EIGEN);  

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n+1; j++){
      free(H[i][j]);
    }
    free(H[i]);
  }
  free(H);  

  for (i=0; i<n+1; i++){
    free(S[i]);
  }
  free(S);

  free(M1);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n+1; j++){
      free(C[i][j]);
    }
    free(C[i]);
  }
  free(C);

  free(B);

  free(S1);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(H1[spin]);
  }
  free(H1);

  dtime(&TEtime);
}




static void Band_DFT_MO_NonCol(
                      int nkpoint, double **kpoint,
                      int SpinP_switch, 
                      double *****nh,
                      double *****ImNL,
                      double ****CntOLP)
{
  int i,j,k,l,n,wan,m,ii1,jj1,jj2,n2;
  int *MP;
  int *order_GA;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  int i1,j1,po,spin,n1,size_H1;
  int num2,RnB,l1,l2,l3,kloop,AN,Rn;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum,nhomos,nlumos;
  int ii,ij,ik,MaxN;
  int wan1,mul,Gc_AN,num0,num1;
  int LB_AN,GB_AN,Bnum;

  double time0,tmp,tmp1,av_num;
  double snum_i,snum_j,snum_k,k1,k2,k3,sum,sumi,Num_State,FermiF;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  double *S1;
  double *RH0;
  double *RH1;
  double *RH2;
  double *RH3;
  double *IH0;
  double *IH1;
  double *IH2;
  double *ko,*M1,**EIGEN;
  double *koS;
  double EV_cut0;
  dcomplex **H,**S,**C;
  dcomplex Ctmp1,Ctmp2;
  double u2,v2,uv,vu;
  double dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2,Imdum;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double FermiEps = 1.0e-14;
  double x_cut = 30.0;
  double OLP_eigen_cut=Threshold_OLP_Eigen;

  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10];
  FILE *fp_EV;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;
  int OMPID,Nthrds,Nprocs;
  int *is1,*ie1;
  int *is2,*ie2;
  int *is12,*ie12;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout) printf("\nBand_DFT_MO start\n");fflush(stdout);

  dtime(&TStime);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n + 2;

  /****************************************************
   Allocation
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);

  ko = (double*)malloc(sizeof(double)*n2);
  koS = (double*)malloc(sizeof(double)*(n+1));

  EIGEN = (double**)malloc(sizeof(double*)*List_YOUSO[33]);
  for (i=0; i<List_YOUSO[33]; i++){
    EIGEN[i] = (double*)malloc(sizeof(double)*n2);
  }

  H = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    H[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  M1 = (double*)malloc(sizeof(double)*n2);

  C = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    C[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  /*****************************************************
        allocation of arrays for parallelization 
  *****************************************************/

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  is12 = (int*)malloc(sizeof(int)*numprocs);
  ie12 = (int*)malloc(sizeof(int)*numprocs);

  is2 = (int*)malloc(sizeof(int)*numprocs);
  ie2 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  for (ID=0; ID<numprocs; ID++){
    is12[ID] = 2*is1[ID] - 1;
    ie12[ID] = 2*ie1[ID];
  }

  /* make is2 and ie2 */ 

  MaxN = 2*n;

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is2[ID] =  1;
      ie2[ID] = -2;
    }
  }

  /* find size_H1 */
  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /* allocation of arrays */
  S1  = (double*)malloc(sizeof(double)*size_H1);
  RH0 = (double*)malloc(sizeof(double)*size_H1);
  RH1 = (double*)malloc(sizeof(double)*size_H1);
  RH2 = (double*)malloc(sizeof(double)*size_H1);
  RH3 = (double*)malloc(sizeof(double)*size_H1);
  IH0 = (double*)malloc(sizeof(double)*size_H1);
  IH1 = (double*)malloc(sizeof(double)*size_H1);
  IH2 = (double*)malloc(sizeof(double)*size_H1);

  /* set S1, RH0, RH1, RH2, RH3, IH0, IH1, IH2 */

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1,  MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[0],  RH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[1],  RH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[2],  RH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[3],  RH3, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
       && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  
    
    /* nothing is done. */
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, ImNL[0], IH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[1], IH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[2], IH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  dtime(&SiloopTime);

  /*****************************************************
         Solve eigenvalue problem at each k-point
  *****************************************************/

  for (kloop=0; kloop<nkpoint; kloop++){

    if (myid==Host_ID && 0<level_stdout) printf("kpoint=%i /%i\n",kloop+1,nkpoint);

    k1 = kpoint[kloop][1];
    k2 = kpoint[kloop][2];
    k3 = kpoint[kloop][3];

    /* make S and H */

    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
	S[i][j] = Complex(0.0,0.0);
      } 
    } 

    for (i=1; i<=2*n; i++){
      for (j=1; j<=2*n; j++){
	H[i][j] = Complex(0.0,0.0);
      } 
    } 

    /* non-spin-orbit coupling and non-LDA+U */
    if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	&& Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      H[Anum+i  ][Bnum+j  ].r += co*RH0[k];
	      H[Anum+i  ][Bnum+j  ].i += si*RH0[k];

	      H[Anum+i+n][Bnum+j+n].r += co*RH1[k];
	      H[Anum+i+n][Bnum+j+n].i += si*RH1[k];
            
	      H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*RH3[k];
	      H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*RH3[k];

	      S[Anum+i  ][Bnum+j  ].r += co*S1[k];
	      S[Anum+i  ][Bnum+j  ].i += si*S1[k];

	      k++;
	    }
	  }
	}
      }
    }

    /* spin-orbit coupling or LDA+U */
    else {  

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      H[Anum+i  ][Bnum+j  ].r += co*RH0[k] - si*IH0[k];
	      H[Anum+i  ][Bnum+j  ].i += si*RH0[k] + co*IH0[k];

	      H[Anum+i+n][Bnum+j+n].r += co*RH1[k] - si*IH1[k];
	      H[Anum+i+n][Bnum+j+n].i += si*RH1[k] + co*IH1[k];
            
	      H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*(RH3[k]+IH2[k]);
	      H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*(RH3[k]+IH2[k]);

	      S[Anum+i  ][Bnum+j  ].r += co*S1[k];
              S[Anum+i  ][Bnum+j  ].i += si*S1[k];

	      k++;
	    }
	  }
	}
      }
    }

    /* set off-diagonal part */

    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
	H[j+n][i].r = H[i][j+n].r;
	H[j+n][i].i =-H[i][j+n].i;
      } 
    } 

    /* diagonalization of S */
    Eigen_PHH(mpi_comm_level1,S,koS,n,n,1);

    if (2<=level_stdout){
      printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",
	     kloop,k1,k2,k3);
      for (i1=1; i1<=n; i1++){
	printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,koS[i1]);
      }
    }

    /* minus eigenvalues to 1.0e-10 */

    for (l=1; l<=n; l++){
      if (koS[l]<1.0e-10) koS[l] = 1.0e-10;
    }

    /* calculate S*1/sqrt(koS) */

    for (l=1; l<=n; l++) koS[l] = 1.0/sqrt(koS[l]);

    /* S * 1.0/sqrt(koS[l]) */

#pragma omp parallel for shared(n,S,koS) private(i1,j1) 

    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=n; j1++){
	S[i1][j1].r = S[i1][j1].r*koS[j1];
	S[i1][j1].i = S[i1][j1].i*koS[j1];
      } 
    } 

    /****************************************************
                  set H' and diagonalize it
    ****************************************************/

    /* U'^+ * H * U * M1 */

    /* transpose S */

    for (i1=1; i1<=n; i1++){
      for (j1=i1+1; j1<=n; j1++){
	Ctmp1 = S[i1][j1];
	Ctmp2 = S[j1][i1];
	S[i1][j1] = Ctmp2;
	S[j1][i1] = Ctmp1;
      }
    }

    /* H * U' */
    /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){
	for (j1=is1[myid]; j1<=ie1[myid]; j1++){

	  double sum_r0 = 0.0;
	  double sum_i0 = 0.0;

	  double sum_r1 = 0.0;
	  double sum_i1 = 0.0;

	  for (l=1; l<=n; l++){
	    sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
	    sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;

	    sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
	    sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
	  }

	  C[2*j1-1][i1].r = sum_r0;
	  C[2*j1-1][i1].i = sum_i0;

	  C[2*j1  ][i1].r = sum_r1;
	  C[2*j1  ][i1].i = sum_i1;
	}
      } 

    } /* #pragma omp parallel */

    /* U'^+ H * U' */
    /* H is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l,jj1,jj2)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=is1[myid]+OMPID; j1<=ie1[myid]; j1+=Nthrds){
	for (i1=1; i1<=n; i1++){

	  double sum_r00 = 0.0;
	  double sum_i00 = 0.0;

	  double sum_r01 = 0.0;
	  double sum_i01 = 0.0;

	  double sum_r10 = 0.0;
	  double sum_i10 = 0.0;

	  double sum_r11 = 0.0;
	  double sum_i11 = 0.0;

	  jj1 = 2*j1 - 1;
	  jj2 = 2*j1;

	  for (l=1; l<=n; l++){

	    sum_r00 += S[i1][l].r*C[jj1][l  ].r + S[i1][l].i*C[jj1][l  ].i;
	    sum_i00 += S[i1][l].r*C[jj1][l  ].i - S[i1][l].i*C[jj1][l  ].r;

	    sum_r01 += S[i1][l].r*C[jj1][l+n].r + S[i1][l].i*C[jj1][l+n].i;
	    sum_i01 += S[i1][l].r*C[jj1][l+n].i - S[i1][l].i*C[jj1][l+n].r;

	    sum_r10 += S[i1][l].r*C[jj2][l  ].r + S[i1][l].i*C[jj2][l  ].i;
	    sum_i10 += S[i1][l].r*C[jj2][l  ].i - S[i1][l].i*C[jj2][l  ].r;

	    sum_r11 += S[i1][l].r*C[jj2][l+n].r + S[i1][l].i*C[jj2][l+n].i;
	    sum_i11 += S[i1][l].r*C[jj2][l+n].i - S[i1][l].i*C[jj2][l+n].r;
	  }

	  H[jj1][2*i1-1].r = sum_r00;
	  H[jj1][2*i1-1].i = sum_i00;

	  H[jj1][2*i1  ].r = sum_r01;
	  H[jj1][2*i1  ].i = sum_i01;

	  H[jj2][2*i1-1].r = sum_r10;
	  H[jj2][2*i1-1].i = sum_i10;

	  H[jj2][2*i1  ].r = sum_r11;
	  H[jj2][2*i1  ].i = sum_i11;

	}
      }

    } /* #pragma omp parallel */

    /* broadcast H */

    BroadCast_ComplexMatrix(mpi_comm_level1,H,2*n,is12,ie12,myid,numprocs,
			    stat_send,request_send,request_recv);

    /* H to C (transposition) */

#pragma omp parallel for shared(n,C,H)  

    for (i1=1; i1<=2*n; i1++){
      for (j1=1; j1<=2*n; j1++){
	C[j1][i1].r = H[i1][j1].r;
	C[j1][i1].i = H[i1][j1].i;
      }
    }

    /* solve the standard eigenvalue problem */
    /*  The output C matrix is distributed by column. */

    Eigen_PHH(mpi_comm_level1,C,ko,2*n,MaxN,0);

    for (i1=1; i1<=MaxN; i1++){
      EIGEN[kloop][i1] = ko[i1];
    }

    /* calculation of wave functions */

    /*  The H matrix is distributed by row */

    for (i1=1; i1<=2*n; i1++){
      for (j1=is2[myid]; j1<=ie2[myid]; j1++){
	H[j1][i1] = C[i1][j1];
      }
    }

    /* transpose */

    for (i1=1; i1<=n; i1++){
      for (j1=i1+1; j1<=n; j1++){
	Ctmp1 = S[i1][j1];
	Ctmp2 = S[j1][i1];
	S[i1][j1] = Ctmp2;
	S[j1][i1] = Ctmp1;
      }
    }

    /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is2,ie2,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l,l1)
    { 

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (j1=is2[myid]+OMPID; j1<=ie2[myid]; j1+=Nthrds){
	for (i1=1; i1<=n; i1++){

	  double sum_r0 = 0.0; 
	  double sum_i0 = 0.0;

	  double sum_r1 = 0.0; 
	  double sum_i1 = 0.0;

	  l1 = 0; 

	  for (l=1; l<=n; l++){

	    l1++; 

	    sum_r0 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
	    sum_i0 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;

	    l1++; 

	    sum_r1 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
	    sum_i1 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;
	  } 

	  C[j1][i1  ].r = sum_r0;
	  C[j1][i1  ].i = sum_i0;

	  C[j1][i1+n].r = sum_r1;
	  C[j1][i1+n].i = sum_i1;

	}
      }

    } /* #pragma omp parallel */

    /* broadcast C: C is distributed by row in each processor */

    BroadCast_ComplexMatrix(mpi_comm_level1,C,2*n,is2,ie2,myid,numprocs,
			    stat_send,request_send,request_recv);

    /* C to H (transposition)
       H consists of column vectors
    */ 

    for (i1=1; i1<=MaxN; i1++){
      for (j1=1; j1<=2*n; j1++){
	H[j1][i1] = C[i1][j1];
      }
    }

    /* find HOMO from eigenvalues */

    Bulk_HOMO[kloop][0] = 0;

    for (i1=1; i1<=MaxN; i1++){
      x = (ko[i1] - ChemP)*Beta;
      if (x<=-x_cut) x = -x_cut;
      if (x_cut<=x)  x = x_cut;
      FermiF = 1.0/(1.0 + exp(x));
      if (0.5<FermiF) Bulk_HOMO[kloop][0] = i1;
    }      

    if (2<=level_stdout){
      printf("k1=%7.3f k2=%7.3f k3=%7.3f  HOMO = %2d\n",
	     k1,k2,k3,Bulk_HOMO[kloop][0]);
    }

    /****************************************************
        LCAO coefficients are stored for calculating
                 values of MOs on grids
    ****************************************************/

    nhomos = num_HOMOs;
    nlumos = num_LUMOs;

    if ( (Bulk_HOMO[kloop][0]-nhomos+1)<1 )  nhomos = Bulk_HOMO[kloop][0];
    if ( (Bulk_HOMO[kloop][0]+nlumos)>MaxN)  nlumos = MaxN - Bulk_HOMO[kloop][0];

    /* HOMOs */

    for (j=0; j<nhomos; j++){

      j1 = Bulk_HOMO[kloop][0] - j;

      /* store eigenvalue */
      HOMOs_Coef[kloop][0][j][0][0].r = EIGEN[kloop][j1];
      HOMOs_Coef[kloop][1][j][0][0].r = EIGEN[kloop][j1];

      /* store eigenvector */
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];
        for (i=0; i<tnoA; i++){
          HOMOs_Coef[kloop][0][j][GA_AN][i].r = H[Anum+i][j1].r;
          HOMOs_Coef[kloop][0][j][GA_AN][i].i = H[Anum+i][j1].i;
          HOMOs_Coef[kloop][1][j][GA_AN][i].r = H[Anum+i+n][j1].r;
          HOMOs_Coef[kloop][1][j][GA_AN][i].i = H[Anum+i+n][j1].i;
        }
      }
    }        

    /* LUMOs */
    for (j=0; j<nlumos; j++){

      j1 = Bulk_HOMO[kloop][0] + 1 + j;

      /* store eigenvalue */
      LUMOs_Coef[kloop][0][j][0][0].r = EIGEN[kloop][j1];
      LUMOs_Coef[kloop][1][j][0][0].r = EIGEN[kloop][j1];

      /* store eigenvector */
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];
        for (i=0; i<tnoA; i++){
          LUMOs_Coef[kloop][0][j][GA_AN][i].r = H[Anum+i][j1].r;
          LUMOs_Coef[kloop][0][j][GA_AN][i].i = H[Anum+i][j1].i;
          LUMOs_Coef[kloop][1][j][GA_AN][i].r = H[Anum+i+n][j1].r;
          LUMOs_Coef[kloop][1][j][GA_AN][i].i = H[Anum+i+n][j1].i;
        }
      }
    }

    Bulk_Num_HOMOs[kloop] = nhomos;
    Bulk_Num_LUMOs[kloop] = nlumos;

    /****************************************************
                        Output
    ****************************************************/

    if (myid==Host_ID){

      strcpy(file_EV,".EV");
      fnjoint(filepath,filename,file_EV);

      if ((fp_EV = fopen(file_EV,"a")) != NULL){

#ifdef xt3
	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

        if (kloop==0){

  	  fprintf(fp_EV,"\n");
	  fprintf(fp_EV,"***********************************************************\n");
	  fprintf(fp_EV,"***********************************************************\n");
	  fprintf(fp_EV,"        Eigenvalues (Hartree) and LCAO coefficients        \n");
	  fprintf(fp_EV,"        at the k-points specified in the input file.       \n");
	  fprintf(fp_EV,"***********************************************************\n");
	  fprintf(fp_EV,"***********************************************************\n");
	}

	k1 = kpoint[kloop][1];
	k2 = kpoint[kloop][2];
	k3 = kpoint[kloop][3];

	fprintf(fp_EV,"\n\n");
	fprintf(fp_EV,"   # of k-point = %i\n",kloop+1);
	fprintf(fp_EV,"   k1=%10.5f k2=%10.5f k3=%10.5f\n\n",k1,k2,k3);
	fprintf(fp_EV,"   Chemical Potential (Hartree) = %18.14f\n",ChemP);
	fprintf(fp_EV,"   HOMO = %i\n\n",Bulk_HOMO[kloop][0]);

	fprintf(fp_EV,"   Real (Re) and imaginary (Im) parts of LCAO coefficients\n\n");

	num0 = 2;
	num1 = 2*n/num0 + 1*((2*n)%num0!=0);
  
	for (i=1; i<=num1; i++){

	  fprintf(fp_EV,"\n");

	  for (i1=-2; i1<=0; i1++){

	    fprintf(fp_EV,"                     ");

	    for (j=1; j<=num0; j++){

	      j1 = num0*(i-1) + j;

	      if (j1<=2*n){ 

		if (i1==-2){
		  fprintf(fp_EV," %4d",j1);
		  fprintf(fp_EV,"                                   ");
		}

		else if (i1==-1){
		  fprintf(fp_EV,"   %8.5f",EIGEN[kloop][j1]);
		  fprintf(fp_EV,"                             ");
		}

		else if (i1==0){
		  fprintf(fp_EV,"     Re(U)");
		  fprintf(fp_EV,"     Im(U)");
		  fprintf(fp_EV,"     Re(D)");
		  fprintf(fp_EV,"     Im(D)");
		}
	      }
	    }
	    fprintf(fp_EV,"\n");
	    if (i1==-1)  fprintf(fp_EV,"\n");
	    if (i1==0)   fprintf(fp_EV,"\n");
	  }

	  Name_Angular[0][0] = "s          ";
	  Name_Angular[1][0] = "px         ";
	  Name_Angular[1][1] = "py         ";
	  Name_Angular[1][2] = "pz         ";
	  Name_Angular[2][0] = "d3z^2-r^2  ";
	  Name_Angular[2][1] = "dx^2-y^2   ";
	  Name_Angular[2][2] = "dxy        ";
	  Name_Angular[2][3] = "dxz        ";
	  Name_Angular[2][4] = "dyz        ";
	  Name_Angular[3][0] = "f5z^2-3r^2 ";
	  Name_Angular[3][1] = "f5xz^2-xr^2";
	  Name_Angular[3][2] = "f5yz^2-yr^2";
	  Name_Angular[3][3] = "fzx^2-zy^2 ";
	  Name_Angular[3][4] = "fxyz       ";
	  Name_Angular[3][5] = "fx^3-3*xy^2";
	  Name_Angular[3][6] = "f3yx^2-y^3 ";
	  Name_Angular[4][0] = "g1         ";
	  Name_Angular[4][1] = "g2         ";
	  Name_Angular[4][2] = "g3         ";
	  Name_Angular[4][3] = "g4         ";
	  Name_Angular[4][4] = "g5         ";
	  Name_Angular[4][5] = "g6         ";
	  Name_Angular[4][6] = "g7         ";
	  Name_Angular[4][7] = "g8         ";
	  Name_Angular[4][8] = "g9         ";

	  Name_Multiple[0] = "0";
	  Name_Multiple[1] = "1";
	  Name_Multiple[2] = "2";
	  Name_Multiple[3] = "3";
	  Name_Multiple[4] = "4";
	  Name_Multiple[5] = "5";

	  i1 = 1; 

	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	    wan1 = WhatSpecies[Gc_AN];
            
	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		for (m=0; m<(2*l+1); m++){

		  if (l==0 && mul==0 && m==0)
		    fprintf(fp_EV,"%4d %3s %s %s", 
			    Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
		  else
		    fprintf(fp_EV,"         %s %s", 
			    Name_Multiple[mul],Name_Angular[l][m]);

		  for (j=1; j<=num0; j++){

		    j1 = num0*(i-1) + j;

		    if (0<i1 && j1<=2*n){
		      fprintf(fp_EV,"  %8.5f",H[i1][j1].r);
		      fprintf(fp_EV,"  %8.5f",H[i1][j1].i);
		      fprintf(fp_EV,"  %8.5f",H[i1+n][j1].r);
		      fprintf(fp_EV,"  %8.5f",H[i1+n][j1].i);
		    }
		  }

		  fprintf(fp_EV,"\n");
		  if (i1==-1)  fprintf(fp_EV,"\n");
		  if (i1==0)   fprintf(fp_EV,"\n");

		  i1++;
		}
	      }
	    }
	  }

	}

	fclose(fp_EV);
      }
      else{
	printf("Failure of saving the EV file.\n");
	fclose(fp_EV);
      }
    } /* if (myid==Host_ID) */

    MPI_Barrier(mpi_comm_level1);

  }  /* kloop */

  /****************************************************
                       free arrays
  ****************************************************/

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);
  free(is2);
  free(ie2);
  free(is12);
  free(ie12);

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(ko);
  free(koS);

  free(S1);
  free(RH0);
  free(RH1);
  free(RH2);
  free(RH3);
  free(IH0);
  free(IH1);
  free(IH2);

  for (i=0; i<List_YOUSO[33]; i++){
    free(EIGEN[i]);
  }
  free(EIGEN);

  for (j=0; j<n2; j++){
    free(H[j]);
  }
  free(H);

  for (i=0; i<n2; i++){
    free(S[i]);
  }
  free(S);

  free(M1);

  for (j=0; j<n2; j++){
    free(C[j]);
  }
  free(C);

  dtime(&TEtime);
}
