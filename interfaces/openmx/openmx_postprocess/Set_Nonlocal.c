/**********************************************************************
  Set_Nonlocal.c:

     Set_Nonlocal.c is a subroutine to calculate matrix elements
     and the derivatives of nonlocal potentials in the momentum space.

  Log of Set_Nonlocal.c:

     15/Sep/2002  Released by T.Ozaki

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#ifdef kcomp
static double NLRF_BesselF(int Gensi, int L, int so, double R);
static void Nonlocal0(double *****HNL, double ******DS_NL);
static void Multiply_DS_NL(int Mc_AN, int Mj_AN, int k, int kl, 
                           int Cwan, int Hwan, int wakg, dcomplex ***NLH);
#else
inline double NLRF_BesselF(int Gensi, int L, int so, double R);
inline void Nonlocal0(double *****HNL, double ******DS_NL);
inline void Multiply_DS_NL(int Mc_AN, int Mj_AN, int k, int kl, 
                           int Cwan, int Hwan, int wakg, dcomplex ***NLH);
#endif


double Set_Nonlocal(double *****HNL, double ******DS_NL)
{
  double TStime,TEtime;

  dtime(&TStime);

  Nonlocal0(HNL, DS_NL);

  /* for time */
  dtime(&TEtime);
  return TEtime - TStime;
}



void Nonlocal0(double *****HNL, double ******DS_NL)
{
  /****************************************************
   Evaluate matrix elements of nonlocal potentials
   in the KB or Blochl separable form in momentum space.
  ****************************************************/
  static int firsttime=1;
  int i,j,kl,n,m;
  int Mc_AN,Gc_AN,h_AN,k,Cwan,Gh_AN,Hwan,so;
  int tno0,tno1,tno2,i1,j1,p,ct_AN,spin;
  int fan,jg,kg,wakg,jg0,Mj_AN0,j0;
  int size_NLH,size_SumNL0,size_TmpNL;
  int Mj_AN,num,size1,size2;
  int *Snd_DS_NL_Size,*Rcv_DS_NL_Size;
  int Original_Mc_AN,po; 
  double rcutA,rcutB,rcut,dmp;
  double time1,time2,time3;
  double stime,etime;
  dcomplex ***NLH;
  double *tmp_array;
  double *tmp_array2;
  double TStime,TEtime;
  double Stime_atom,Etime_atom;
  int numprocs,myid,tag=999,ID,IDS,IDR;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OneD_Nloop,*OneD2Mc_AN,*OneD2h_AN;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
             calculate the size of arrays:
  ****************************************************/

  /*
  Spe_Num_RVPS[Hwan]     ->  YOUSO35*YOUSO34 
  Spe_VPS_List[Hwan][L]  ->  0,0,0,.., 1,1,1,.., 2,2,2,..,
  */

  size_SumNL0 = (List_YOUSO[25]+1)*List_YOUSO[24]*(List_YOUSO[19]+1);

  size_TmpNL = (List_YOUSO[25]+1)*List_YOUSO[24]*
               (2*(List_YOUSO[25]+1)+1)*List_YOUSO[19]*(2*List_YOUSO[30]+1);

  /* PrintMemory */
  if (firsttime) {
    PrintMemory("Set_Nonlocal: SumNL0",sizeof(double)*size_SumNL0,NULL);
    PrintMemory("Set_Nonlocal: SumNLr0",sizeof(double)*size_SumNL0,NULL);
    PrintMemory("Set_Nonlocal: TmpNL", sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_Nonlocal: TmpNLr",sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_Nonlocal: TmpNLt",sizeof(dcomplex)*size_TmpNL,NULL);
    PrintMemory("Set_Nonlocal: TmpNLp",sizeof(dcomplex)*size_TmpNL,NULL);
    firsttime=0;
  }

  /* one-dimensionalize the Mc_AN and h_AN loops */

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD_Nloop++;
    }
  }  

  OneD2Mc_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));
  OneD2h_AN = (int*)malloc(sizeof(int)*(OneD_Nloop+1));

  OneD_Nloop = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      OneD2Mc_AN[OneD_Nloop] = Mc_AN; 
      OneD2h_AN[OneD_Nloop] = h_AN; 
      OneD_Nloop++;
    }
  }

  /****************************************************
                    calculate DS_NL
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

#pragma omp parallel shared(List_YOUSO,time_per_atom,DS_NL,Comp2Real,OneD_Grid,Spe_Num_RVPS,Spe_VPS_List,Spe_Num_Basis,Spe_MaxL_Basis,PAO_Nkmax,VPS_j_dependency,atv,Gxyz,WhatSpecies,ncn,natn,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,Ngrid_NormK,NormK,Spe_NLRF_Bessel) 
  {

    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,h_AN,Gc_AN,Cwan,Gh_AN;
    int Rnh,Hwan,so,i,j,k,l,m;
    int LL,L0,L1,L,Mul0,M0,M1,num0,num1;
    int Lmax,Lmax_Four_Int,Ls;

    double dx,dy,dz,r;
    double S_coordinate[3];
    double siT,coT,siP,coP;
    double gant,theta,phi;
    double SH[2],dSHt[2],dSHp[2];
    double Stime_atom, Etime_atom;
    double Normk,kmin,kmax;
    double Bessel_Pro0,Bessel_Pro1;
    double h,coe0,sj,sjp;
    double tmp0,tmp1,tmp2,tmp3,tmp4;
    double **SphB,**SphBp;
    double *tmp_SphB,*tmp_SphBp;
    double ***SumNL0;
    double ***SumNLr0;

    dcomplex CsumNL0,CsumNLr,CsumNLt,CsumNLp;
    dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
    dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
    dcomplex *****TmpNL;
    dcomplex *****TmpNLr;
    dcomplex *****TmpNLt;
    dcomplex *****TmpNLp;
    dcomplex **CmatNL0;
    dcomplex **CmatNLr;
    dcomplex **CmatNLt;
    dcomplex **CmatNLp;

    /* allocation of arrays */

    TmpNL = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNL[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNL[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNL[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[19]);
	  for (l=0; l<List_YOUSO[19]; l++){
	    TmpNL[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
	  }
	}
      }
    }

    TmpNLr = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLr[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLr[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLr[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[19]);
	  for (l=0; l<List_YOUSO[19]; l++){
	    TmpNLr[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
	  }
	}
      }
    }

    TmpNLt = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLt[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLt[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLt[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[19]);
	  for (l=0; l<List_YOUSO[19]; l++){
	    TmpNLt[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
	  }
	}
      }
    }

    TmpNLp = (dcomplex*****)malloc(sizeof(dcomplex****)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      TmpNLp[i] = (dcomplex****)malloc(sizeof(dcomplex***)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	TmpNLp[i][j] = (dcomplex***)malloc(sizeof(dcomplex**)*(2*(List_YOUSO[25]+1)+1));
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  TmpNLp[i][j][k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[19]);
	  for (l=0; l<List_YOUSO[19]; l++){
	    TmpNLp[i][j][k][l] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
	  }
	}
      }
    }

    SumNL0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNL0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNL0[i][j] = (double*)malloc(sizeof(double)*(List_YOUSO[19]+1));
      }
    }

    SumNLr0 = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      SumNLr0[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
	SumNLr0[i][j] = (double*)malloc(sizeof(double)*(List_YOUSO[19]+1));
      }
    }

    CmatNL0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNL0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
    }

    CmatNLr = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLr[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
    }

    CmatNLt = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLt[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
    }

    CmatNLp = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      CmatNLp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*List_YOUSO[30]+1));
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and h_AN */

      Mc_AN = OneD2Mc_AN[Nloop];
      h_AN  = OneD2h_AN[Nloop];

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      /* set data on h_AN */

      Gh_AN = natn[Gc_AN][h_AN];        
      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Gxyz[Gc_AN][1]; 
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Gxyz[Gc_AN][2]; 
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Gxyz[Gc_AN][3];

      xyz2spherical(dx,dy,dz,0.0,0.0,0.0,S_coordinate); 
      r     = S_coordinate[0];
      theta = S_coordinate[1];
      phi   = S_coordinate[2];

      /* for empty atoms or finite elemens basis */

      if (r<1.0e-10) r = 1.0e-10;

      /* precalculation of sin and cos */

      siT = sin(theta);
      coT = cos(theta);
      siP = sin(phi);
      coP = cos(phi);

      for (so=0; so<=VPS_j_dependency[Hwan]; so++){

	/****************************************************
           Evaluate ovelap integrals <chi0|P> between PAOs
           and progectors of nonlocal potentials. 
	****************************************************/
	/****************************************************
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3 
	****************************************************/

	kmin = Radial_kmin;
	kmax = PAO_Nkmax;

	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	    for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
	      L1 = Spe_VPS_List[Hwan][L];
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
		  TmpNL[L0][Mul0][L0+M0][L][L1+M1]  = Complex(0.0,0.0); 
		  TmpNLr[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
		  TmpNLt[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
		  TmpNLp[L0][Mul0][L0+M0][L][L1+M1] = Complex(0.0,0.0); 
		}
	      }
	    } 
	  }
	}

	Lmax = -10;
	for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
	  if (Lmax<Spe_VPS_List[Hwan][L]) Lmax = Spe_VPS_List[Hwan][L];
	}
	if (Spe_MaxL_Basis[Cwan]<Lmax)
	  Lmax_Four_Int = 2*Lmax;
	else 
	  Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

	/* allocate SphB and SphBp */

	SphB = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
	for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	  SphB[LL] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
	}

	SphBp = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
	for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	  SphBp[LL] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
	}
      
	tmp_SphB  = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));
	tmp_SphBp = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));

	/* calculate SphB and SphBp */

	h = (kmax - kmin)/(double)OneD_Grid;

	for (i=0; i<=OneD_Grid; i++){
	  Normk = kmin + (double)i*h;
	  Spherical_Bessel(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
	  for(LL=0; LL<=Lmax_Four_Int; LL++){ 
	    SphB[LL][i]  = tmp_SphB[LL]; 
	    SphBp[LL][i] = tmp_SphBp[LL];
	  }
	}

	free(tmp_SphBp);
	free(tmp_SphB);

	/* LL loop */

	for(LL=0; LL<=Lmax_Four_Int; LL++){ 

	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
	      for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
		SumNL0[L0][Mul0][L] = 0.0;
		SumNLr0[L0][Mul0][L] = 0.0;
	      }
	    }
	  }

	  h = (kmax - kmin)/(double)OneD_Grid;
	  for (i=0; i<=OneD_Grid; i++){

	    if (i==0 || i==OneD_Grid) coe0 = 0.50;
	    else                      coe0 = 1.00;

	    Normk = kmin + (double)i*h;

	    sj  =  SphB[LL][i];
	    sjp = SphBp[LL][i];

	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

		Bessel_Pro0 = RF_BesselF(Cwan,L0,Mul0,Normk);
		tmp0 = coe0*h*Normk*Normk*Bessel_Pro0;
		tmp1 = tmp0*sj;
		tmp2 = tmp0*Normk*sjp;

		for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){

		  Bessel_Pro1 = NLRF_BesselF(Hwan,L,so,Normk);  /* j-dependent */

		  tmp3 = tmp1*Bessel_Pro1;
		  tmp4 = tmp2*Bessel_Pro1;
		  SumNL0[L0][Mul0][L]  += tmp3;
		  SumNLr0[L0][Mul0][L] += tmp4;
		}
	      }
	    }
	  }

	  /* derivatives of "on site" */
	  if (h_AN==0){ 
	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
		for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
		  SumNLr0[L0][Mul0][L] = 0.0;
		}
	      }
	    }
	  }

	  /****************************************************
            For "overlap",
            sum_m 8*(-i)^{-L0+L1+l}*
                  C_{L0,-M0,L1,M1,l,m}*
                  \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
	  ****************************************************/

	  for(m=-LL; m<=LL; m++){ 

	    ComplexSH(LL,m,theta,phi,SH,dSHt,dSHp);
	    SH[1]   = -SH[1];
	    dSHt[1] = -dSHt[1];
	    dSHp[1] = -dSHp[1];

	    CY  = Complex(SH[0],SH[1]);
	    CYt = Complex(dSHt[0],dSHt[1]);
	    CYp = Complex(dSHp[0],dSHp[1]);

	    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
		for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){

		  L1 = Spe_VPS_List[Hwan][L];
		  Ls = -L0 + L1 + LL;  

		  if (abs(L1-LL)<=L0 && L0<=(L1+LL) ){

		    Cpow = Im_pow(-1,Ls);
		    CY1  = Cmul(Cpow,CY);
		    CYt1 = Cmul(Cpow,CYt);
		    CYp1 = Cmul(Cpow,CYp);

		    for (M0=-L0; M0<=L0; M0++){

		      M1 = M0 - m;

		      if (abs(M1)<=L1){

			gant = Gaunt(L0,M0,L1,M1,LL,m);

			/* S */ 

			tmp0 = gant*SumNL0[L0][Mul0][L];
			Ctmp2 = CRmul(CY1,tmp0);
			TmpNL[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNL[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

			/* dS/dr */ 

			tmp0 = gant*SumNLr0[L0][Mul0][L];
			Ctmp2 = CRmul(CY1,tmp0); 
			TmpNLr[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNLr[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

			/* dS/dt */ 

			tmp0 = gant*SumNL0[L0][Mul0][L];
			Ctmp2 = CRmul(CYt1,tmp0); 
			TmpNLt[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNLt[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

			/* dS/dp */ 

			tmp0 = gant*SumNL0[L0][Mul0][L];
			Ctmp2 = CRmul(CYp1,tmp0); 
			TmpNLp[L0][Mul0][L0+M0][L][L1+M1] =
			  Cadd(TmpNLp[L0][Mul0][L0+M0][L][L1+M1],Ctmp2);

		      }
		    }
		  }

		}
	      }
	    }
	  }
	} /* LL */

	/* free SphB and SphBp */
      
	for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	  free(SphB[LL]);
	}
	free(SphB);

	for(LL=0; LL<(Lmax_Four_Int+3); LL++){ 
	  free(SphBp[LL]);
	}
	free(SphBp);

	/****************************************************
                           complex to real
	****************************************************/

	num0 = 0;
	for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	    num1 = 0;
	    for (L=1; L<=Spe_Num_RVPS[Hwan]; L++){
	      L1 = Spe_VPS_List[Hwan][L];

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumNL0 = Complex(0.0,0.0);
		  CsumNLr = Complex(0.0,0.0);
		  CsumNLt = Complex(0.0,0.0);
		  CsumNLp = Complex(0.0,0.0);

		  for (k=-L0; k<=L0; k++){

		    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

		    /* S */

		    Ctmp0 = TmpNL[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNL0 = Cadd(CsumNL0,Ctmp2);

		    /* dS/dr */

		    Ctmp0 = TmpNLr[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLr = Cadd(CsumNLr,Ctmp2);

		    /* dS/dt */

		    Ctmp0 = TmpNLt[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLt = Cadd(CsumNLt,Ctmp2);

		    /* dS/dp */

		    Ctmp0 = TmpNLp[L0][Mul0][L0+k][L][L1+M1];
		    Ctmp2 = Cmul(Ctmp1,Ctmp0);
		    CsumNLp = Cadd(CsumNLp,Ctmp2);

		  }

		  CmatNL0[L0+M0][L1+M1] = CsumNL0;
		  CmatNLr[L0+M0][L1+M1] = CsumNLr;
		  CmatNLt[L0+M0][L1+M1] = CsumNLt;
		  CmatNLp[L0+M0][L1+M1] = CsumNLp;
		}
	      }

	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

		  CsumNL0 = Complex(0.0,0.0);
		  CsumNLr = Complex(0.0,0.0);
		  CsumNLt = Complex(0.0,0.0);
		  CsumNLp = Complex(0.0,0.0);

		  for (k=-L1; k<=L1; k++){

		    /* S */ 

		    Ctmp1 = Cmul(CmatNL0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNL0 = Cadd(CsumNL0,Ctmp1);

		    /* dS/dr */ 

		    Ctmp1 = Cmul(CmatNLr[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLr = Cadd(CsumNLr,Ctmp1);

		    /* dS/dt */ 

		    Ctmp1 = Cmul(CmatNLt[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLt = Cadd(CsumNLt,Ctmp1);

		    /* dS/dp */ 

		    Ctmp1 = Cmul(CmatNLp[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
		    CsumNLp = Cadd(CsumNLp,Ctmp1);

		  }

		  DS_NL[so][0][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumNL0.r;

		  if (h_AN!=0){

		    if (fabs(siT)<10e-14){

		      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r);

		      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r);

		      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);

		    }

		    else{

		      DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*coP*CsumNLr.r + coT*coP/r*CsumNLt.r
			     - siP/siT/r*CsumNLp.r);

		      DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(siT*siP*CsumNLr.r + coT*siP/r*CsumNLt.r
			     + coP/siT/r*CsumNLp.r);

		      DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] =
			-8.0*(coT*CsumNLr.r - siT/r*CsumNLt.r);

		    }
		  }

		  else{
		    DS_NL[so][1][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    DS_NL[so][2][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		    DS_NL[so][3][Mc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 0.0;
		  } 

		}
	      }

	      num1 = num1 + 2*L1 + 1; 
	    }

	    num0 = num0 + 2*L0 + 1; 
	  }
	}
      } /* so */


      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLp[i]);
    }
    free(CmatNLp);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLt[i]);
    }
    free(CmatNLt);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNLr[i]);
    }
    free(CmatNLr);

    for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
      free(CmatNL0[i]);
    }
    free(CmatNL0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNLr0[i][j]);
      }
      free(SumNLr0[i]);
    }
    free(SumNLr0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(SumNL0[i][j]);
      }
      free(SumNL0[i]);
    }
    free(SumNL0);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNLp[i][j][k][l]);
	  }
	  free(TmpNLp[i][j][k]);
	}
	free(TmpNLp[i][j]);
      }
      free(TmpNLp[i]);
    }
    free(TmpNLp);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNLt[i][j][k][l]);
	  }
	  free(TmpNLt[i][j][k]);
	}
	free(TmpNLt[i][j]);
      }
      free(TmpNLt[i]);
    }
    free(TmpNLt);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNLr[i][j][k][l]);
	  }
	  free(TmpNLr[i][j][k]);
	}
	free(TmpNLr[i][j]);
      }
      free(TmpNLr[i]);
    }
    free(TmpNLr);

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
	  for (l=0; l<List_YOUSO[19]; l++){
	    free(TmpNL[i][j][k][l]);
	  }
	  free(TmpNL[i][j][k]);
	}
	free(TmpNL[i][j]);
      }
      free(TmpNL[i]);
    }
    free(TmpNL);

#pragma omp flush(DS_NL)

  } /* #pragma omp parallel */

  if (measure_time){
    dtime(&etime);
    time1 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITH COMMUNICATION

     MPI: communicate only for k=0
     DS_NL
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* allocation of arrays */

  NLH = (dcomplex***)malloc(sizeof(dcomplex**)*3); 
  for (k=0; k<3; k++){
    NLH[k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]); 
    for (i=0; i<List_YOUSO[7]; i++){
      NLH[k][i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]); 
    }
  }

  Snd_DS_NL_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DS_NL_Size = (int*)malloc(sizeof(int)*numprocs);

  for (ID=0; ID<numprocs; ID++){
    F_Snd_Num_WK[ID] = 0;
    F_Rcv_Num_WK[ID] = 0;
  }

  do {

    /***********************************                                                            
          set the size of data                                                                      
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /* find the data size to send the block data */

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

        size1 = 0;
        n = F_Snd_Num_WK[IDS];

        Mc_AN = Snd_MAN[IDS][n];
        Gc_AN = Snd_GAN[IDS][n];
        Cwan = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_NO[Cwan];

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];
          size1 += (VPS_j_dependency[Hwan]+1)*tno1*tno2;
        }

        Snd_DS_NL_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
        Snd_DS_NL_Size[IDS] = 0;
      }

      /* receiving of the size of the data */

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_DS_NL_Size[IDR] = size2;
      }
      else{
        Rcv_DS_NL_Size[IDR] = 0;
      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) )  MPI_Wait(&request,&stat);

    } /* ID */

    /***********************************
               data transfer
    ************************************/

    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      /******************************
         sending of the data 
      ******************************/

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ){

	size1 = Snd_DS_NL_Size[IDS];

	/* allocation of the array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to the vector array */

	num = 0;
	n = F_Snd_Num_WK[IDS];

	Mc_AN = Snd_MAN[IDS][n];
	Gc_AN = Snd_GAN[IDS][n];
	Cwan = WhatSpecies[Gc_AN]; 
	tno1 = Spe_Total_NO[Cwan];

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];        
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];

          for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
	        tmp_array[num] = DS_NL[so][0][Mc_AN][h_AN][i][j];
	        num++;
	      } 
	    } 
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /******************************
        receiving of the block data
      ******************************/

      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ){
        
	size2 = Rcv_DS_NL_Size[IDR];
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

	/* store */

	num = 0;
	n = F_Rcv_Num_WK[IDR];
	Original_Mc_AN = F_TopMAN[IDR] + n;

	Gc_AN = Rcv_GAN[IDR][n];
	Cwan = WhatSpecies[Gc_AN];
	tno1 = Spe_Total_NO[Cwan];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno2 = Spe_Total_VPS_Pro[Hwan];

          for (so=0; so<=VPS_j_dependency[Hwan]; so++){
	    for (i=0; i<tno1; i++){
	      for (j=0; j<tno2; j++){
	        DS_NL[so][0][Matomnum+1][h_AN][i][j] = tmp_array2[num];
	        num++;
	      }
	    }
	  }
	}

	/* free tmp_array2 */
	free(tmp_array2);

	/*****************************************************************
                           multiplying overlap integrals
	*****************************************************************/

        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	  dtime(&Stime_atom);

          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          fan = FNAN[Gc_AN];
          rcutA = Spe_Atom_Cut1[Cwan];

          n = F_Rcv_Num_WK[IDR];
          jg = Rcv_GAN[IDR][n];

          for (j0=0; j0<=fan; j0++){

            jg0 = natn[Gc_AN][j0];
            Mj_AN0 = F_G2M[jg0];

            po = 0;
            if (Original_Mc_AN==Mj_AN0){
              po = 1;
              j = j0;
            }

            if (po==1){

	      Hwan = WhatSpecies[jg];
              rcutB = Spe_Atom_Cut1[Hwan];
              rcut = rcutA + rcutB;

	      for (m=0; m<Spe_Total_NO[Cwan]; m++){
		for (n=0; n<Spe_Total_NO[Hwan]; n++){
		  NLH[0][m][n].r = 0.0;     /* <up|VNL|up> */
		  NLH[1][m][n].r = 0.0;     /* <dn|VNL|dn> */
		  NLH[2][m][n].r = 0.0;     /* <up|VNL|dn> */
		  NLH[0][m][n].i = 0.0;
		  NLH[1][m][n].i = 0.0;
		  NLH[2][m][n].i = 0.0;
		}
	      }

              for (k=0; k<=fan; k++){

                kg = natn[Gc_AN][k];
                wakg = WhatSpecies[kg];
                kl = RMI1[Mc_AN][j][k];

                if (0<=kl){

                  Multiply_DS_NL(Mc_AN, Matomnum+1, k, kl, Cwan, Hwan, wakg, NLH);

		} /* if (0<=kl) */

	      } /* k */

              /****************************************************
                      adding NLH to HNL
  
                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	      ****************************************************/

              dmp = dampingF(rcut,Dis[Gc_AN][j]);

	      for (p=0; p<List_YOUSO[5]; p++){
		for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
		  for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){

		    HNL[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].r*F_NL_flag;

		    if (SO_switch==1){
		      iHNL[ p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].i*F_NL_flag;
		      iHNL0[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].i*F_NL_flag;
		    }
		  }
		}
	      }

	    } /* if (po==1) */
	  } /* j0 */

          dtime(&Etime_atom);
          time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

	} /* Mc_AN */

        /********************************************                                               
            increment of F_Rcv_Num_WK[IDR]                                                          
	********************************************/

        F_Rcv_Num_WK[IDR]++;

      }

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) {

        MPI_Wait(&request,&stat);
        free(tmp_array);  /* freeing of array */

        /********************************************                                               
             increment of F_Snd_Num_WK[IDS]                                                         
	********************************************/

        F_Snd_Num_WK[IDS]++;
      }

    } /* ID */

    /*****************************************************                                          
      check whether all the communications have finished                                            
    *****************************************************/

    po = 0;
    for (ID=0; ID<numprocs; ID++){

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if ( 0<(F_Snd_Num[IDS]-F_Snd_Num_WK[IDS]) ) po += F_Snd_Num[IDS]-F_Snd_Num_WK[IDS];
      if ( 0<(F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR]) ) po += F_Rcv_Num[IDR]-F_Rcv_Num_WK[IDR];
    }

  } while (po!=0);

  /* freeing of array */

  free(Rcv_DS_NL_Size);
  free(Snd_DS_NL_Size);

  for (k=0; k<3; k++){
    for (i=0; i<List_YOUSO[7]; i++){
      free(NLH[k][i]);
    }
    free(NLH[k]);
  }
  free(NLH);

  if (measure_time){
    dtime(&etime);
    time2 = etime - stime;
  }

  /*******************************************************
   *******************************************************
     multiplying overlap integrals WITHOUT COMMUNICATION
  *******************************************************
  *******************************************************/

  MPI_Barrier(mpi_comm_level1);
  if (measure_time) dtime(&stime);

  /* allocation of arrays */

#pragma omp parallel shared(Matomnum,time_per_atom,HNL,iHNL,iHNL0,F_NL_flag,List_YOUSO,Dis,SpinP_switch,Spe_Total_NO,DS_NL,Spe_VPS_List,Spe_VNLE,Spe_Num_RVPS,VPS_j_dependency,Spe_Total_VPS_Pro,RMI1,F_G2M,natn,Spe_Atom_Cut1,FNAN,WhatSpecies,M2G,OneD2h_AN,OneD2Mc_AN,OneD_Nloop,SO_switch) 
  {
    int OMPID,Nthrds,Nprocs,Nloop;
    int Mc_AN,j,Gc_AN,Cwan,fan,jg,i1,j1,i;
    int Mj_AN,Hwan,k,kg,wakg,kl;
    int p,m,n,L,L1,L2,L3;
    double rcutA,rcutB,rcut,sum,ene;
    double Stime_atom, Etime_atom;
    double ene_m,ene_p,dmp;
    double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
    double PFp,PFm;
    dcomplex ***NLH;
    dcomplex sum0,sum1,sum2; 

    /* allocation of arrays */

    NLH = (dcomplex***)malloc(sizeof(dcomplex**)*3); 
    for (k=0; k<3; k++){
      NLH[k] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]); 
      for (i=0; i<List_YOUSO[7]; i++){
	NLH[k][i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[7]); 
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    /* one-dimensionalized loop */

    for (Nloop=OMPID*OneD_Nloop/Nthrds; Nloop<(OMPID+1)*OneD_Nloop/Nthrds; Nloop++){

      dtime(&Stime_atom);

      /* get Mc_AN and j */

      Mc_AN = OneD2Mc_AN[Nloop];
      j     = OneD2h_AN[Nloop];

      /* set data on Mc_AN */
    
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      rcutA = Spe_Atom_Cut1[Cwan];

      /* set data on j */
  
      jg = natn[Gc_AN][j];
      Mj_AN = F_G2M[jg];

      if (Mj_AN<=Matomnum){

	Hwan = WhatSpecies[jg];
	rcutB = Spe_Atom_Cut1[Hwan];
	rcut = rcutA + rcutB;

	for (m=0; m<Spe_Total_NO[Cwan]; m++){
	  for (n=0; n<Spe_Total_NO[Hwan]; n++){
	    NLH[0][m][n].r = 0.0;     /* <up|VNL|up> */
	    NLH[1][m][n].r = 0.0;     /* <dn|VNL|dn> */
	    NLH[2][m][n].r = 0.0;     /* <up|VNL|dn> */
	    NLH[0][m][n].i = 0.0;
	    NLH[1][m][n].i = 0.0;
	    NLH[2][m][n].i = 0.0;
	  }
	}

	for (k=0; k<=fan; k++){

	  kg = natn[Gc_AN][k];
	  wakg = WhatSpecies[kg];
	  kl = RMI1[Mc_AN][j][k];
        
	  if (0<=kl){

	    Multiply_DS_NL(Mc_AN, Mj_AN, k, kl, Cwan, Hwan, wakg, NLH);

	  } /* if (0<=kl) */
	} /* k */

	/****************************************************
                       adding NLH to HNL
  
                 HNL[0] and iHNL[0] for up-up
                 HNL[1] and iHNL[1] for dn-dn
                 HNL[2] and iHNL[2] for up-dn
	****************************************************/

	dmp = dampingF(rcut,Dis[Gc_AN][j]);

	for (p=0; p<List_YOUSO[5]; p++){
	  for (i1=0; i1<Spe_Total_NO[Cwan]; i1++){
	    for (j1=0; j1<Spe_Total_NO[Hwan]; j1++){

	      HNL[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].r*F_NL_flag;

	      if (SO_switch==1){
		iHNL[p][Mc_AN][j][i1][j1]  = dmp*NLH[p][i1][j1].i*F_NL_flag;
		iHNL0[p][Mc_AN][j][i1][j1] = dmp*NLH[p][i1][j1].i*F_NL_flag;
	      }
	    }
	  }
	}

      } /* if (Mj_AN<=Matomnum) */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Nloop */

    /* freeing of array */

    for (k=0; k<3; k++){
      for (i=0; i<List_YOUSO[7]; i++){
	free(NLH[k][i]);
      }
      free(NLH[k]);
    }
    free(NLH);

#pragma omp barrier
#pragma omp flush(HNL,iHNL,iHNL0)

  } /* #pragma omp parallel */

  /****************************************************
  freeing of arrays:
  ****************************************************/

  free(OneD2Mc_AN);
  free(OneD2h_AN);

  if (measure_time){
    dtime(&etime);
    time3 = etime - stime;
  }

  if (measure_time){
    printf("Set_Nonlocal: myid=%2d time1=%10.5f time2=%10.5f time3=%10.5f\n",
           myid,time1,time2,time3);
  }

  /****************************************************
   MPI_Barrier
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);
} 


void Multiply_DS_NL(int Mc_AN, int Mj_AN, int k, int kl, 
                    int Cwan, int Hwan, int wakg, dcomplex ***NLH)
{
  int m,n,L,L1,L2,L3,spin,target_spin;
  int GA_AN,GB_AN;
  double ene_core_hole[2][20];
  double sum,sumMul[2],ene,PFp,PFm,ene_p,ene_m; 
  double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
  dcomplex sum0,sum1,sum2; 

  /****************************************************
              l-dependent non-local part
  ****************************************************/
        
  if (VPS_j_dependency[wakg]==0){

    for (m=0; m<Spe_Total_NO[Cwan]; m++){
      for (n=0; n<Spe_Total_NO[Hwan]; n++){

	sum = 0.0;
	L = 0;

	for (L1=1; L1<=Spe_Num_RVPS[wakg]; L1++){

	  ene = Spe_VNLE[0][wakg][L1-1];
	  L2 = 2*Spe_VPS_List[wakg][L1];

	  for (L3=0; L3<=L2; L3++){

	    sum += ene*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];
	    L++;
	  }

	} /* L1 */

	NLH[0][m][n].r += sum;    /* <up|VNL|up> */
	NLH[1][m][n].r += sum;    /* <dn|VNL|dn> */

      }
    }

  } /* if */

  /****************************************************
               j-dependent non-local part
  ****************************************************/

  else if (VPS_j_dependency[wakg]==1){

    for (m=0; m<Spe_Total_NO[Cwan]; m++){
      for (n=0; n<Spe_Total_NO[Hwan]; n++){

	sum0 = Complex(0.0,0.0);
	sum1 = Complex(0.0,0.0);
	sum2 = Complex(0.0,0.0);

	L = 0;
	for (L1=1; L1<=Spe_Num_RVPS[wakg]; L1++){

	  ene_p = Spe_VNLE[0][wakg][L1-1];
	  ene_m = Spe_VNLE[1][wakg][L1-1];

	  if      (Spe_VPS_List[wakg][L1]==0) { L2=0; PFp=1.0;     PFm=0.0;     }  
	  else if (Spe_VPS_List[wakg][L1]==1) { L2=2; PFp=2.0/3.0; PFm=1.0/3.0; }
	  else if (Spe_VPS_List[wakg][L1]==2) { L2=4; PFp=3.0/5.0; PFm=2.0/5.0; }
	  else if (Spe_VPS_List[wakg][L1]==3) { L2=6; PFp=4.0/7.0; PFm=3.0/7.0; }

	  /****************************************************
                  off-diagonal contribution on up-dn
                  for spin non-collinear
	  ****************************************************/

	  if (SpinP_switch==3){

	    /***************
                   p
	    ***************/ 

	    if (L2==2){

	      /* real contribution of l+1/2 to off-diagonal up-down matrix */ 
	      sum2.r += 
		 ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L  ]; 

	      /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */ 
	      sum2.i +=
		-ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		+ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]; 

	      /* real contribution of l-1/2 for to off-diagonal up-down matrix */ 
	      sum2.r -=
		 ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L  ]; 

	      /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */ 
	      sum2.i -=
		-ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		+ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]; 
	    }

	    /***************
                   d
	    ***************/ 

	    else if (L2==4){

	      /* real contribution of l+1/2 to off diagonal up-down matrix */ 
	      tmp0 = sqrt(3.0);
	      tmp1 = ene_p/5.0; 
	      tmp2 = tmp0*tmp1;

	      sum2.r +=
		-tmp2*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp2*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+2];

	      /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */ 

	      sum2.i +=
		 tmp2*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp2*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp1*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp1*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+2];

	      /* real contribution of l-1/2 for to off-diagonal up-down matrix */ 

	      tmp1 = ene_m/5.0; 
	      tmp2 = tmp0*tmp1;

	      sum2.r -=
		-tmp2*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp2*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+2];

	      /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */ 

	      sum2.i -=
		 tmp2*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp2*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp1*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp1*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+2];

	    }

	    /***************
                   f
	    ***************/ 

	    else if (L2==6){

	      /* real contribution of l+1/2 to off diagonal up-down matrix */ 

	      tmp0 = sqrt(6.0);
	      tmp1 = sqrt(3.0/2.0);
	      tmp2 = sqrt(5.0/2.0);

	      tmp3 = ene_p/7.0; 
	      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
	      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
	      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

	      sum2.r += 
                -tmp6*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		+tmp6*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+6]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+4];

	      /* imaginary contribution of l+1/2 to off diagonal up-down matrix */ 

	      sum2.i +=
                 tmp6*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		-tmp6*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L  ]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+4]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+2]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+6]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+4];

	      /* real contribution of l-1/2 for to diagonal up-down matrix */ 

	      tmp3 = ene_m/7.0; 
	      tmp4 = tmp1*tmp3; /* sqrt(3.0/2.0) */
	      tmp5 = tmp2*tmp3; /* sqrt(5.0/2.0) */
	      tmp6 = tmp0*tmp3; /* sqrt(6.0)     */

	      sum2.r -=
                -tmp6*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		+tmp6*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+6]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+4];

	      /* imaginary contribution of l-1/2 to off diagonal up-down matrix */ 

	      sum2.i -= 
                 tmp6*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		-tmp6*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L  ]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+4]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+1]
		-tmp5*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		+tmp5*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+2]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+6]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+3]
		-tmp4*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+5]
		+tmp4*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+4];

	    }

	  }

	  /****************************************************
             off-diagonal contribution on up-up and dn-dn
	  ****************************************************/

	  /* p */ 

	  if (L2==2){

	    tmp0 =
	       ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L  ]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      -ene_p/3.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L  ]; 

	    /* contribution of l+1/2 for up spin */ 
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */ 
	    sum1.i += tmp0;

	    tmp0 = 
	       ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L  ]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      -ene_m/3.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L  ];

	    /* contribution of l-1/2 for up spin */
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */ 
	    sum1.i += -tmp0;
	  }

	  /* d */ 

	  else if (L2==4){

	    tmp0 =
	       ene_p*2.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
	      -ene_p*2.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      +ene_p*1.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+4]
	      -ene_p*1.0/5.0*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+3]; 

	    /* contribution of l+1/2 for up spin */ 
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */ 
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m*2.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
	      -ene_m*2.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      +ene_m*1.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+4]
	      -ene_m*1.0/5.0*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+3]; 

	    /* contribution of l-1/2 for up spin */ 
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */ 
	    sum1.i += -tmp0;

	  }

	  /* f */ 

	  else if (L2==6){

	    tmp0 =
	       ene_p*1.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+1]*DS_NL[0][0][Mj_AN][kl][n][L+2]
	      -ene_p*1.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+2]*DS_NL[0][0][Mj_AN][kl][n][L+1]
	      +ene_p*2.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+3]*DS_NL[0][0][Mj_AN][kl][n][L+4]
	      -ene_p*2.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+4]*DS_NL[0][0][Mj_AN][kl][n][L+3]
	      +ene_p*3.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+5]*DS_NL[0][0][Mj_AN][kl][n][L+6]
	      -ene_p*3.0/7.0*DS_NL[0][0][Mc_AN][k][m][L+6]*DS_NL[0][0][Mj_AN][kl][n][L+5];

	    /* contribution of l+1/2 for up spin */ 
	    sum0.i += -tmp0;

	    /* contribution of l+1/2 for down spin */ 
	    sum1.i += tmp0;

	    tmp0 =
	       ene_m*1.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+1]*DS_NL[1][0][Mj_AN][kl][n][L+2]
	      -ene_m*1.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+2]*DS_NL[1][0][Mj_AN][kl][n][L+1]
	      +ene_m*2.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+3]*DS_NL[1][0][Mj_AN][kl][n][L+4]
	      -ene_m*2.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+4]*DS_NL[1][0][Mj_AN][kl][n][L+3]
	      +ene_m*3.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+5]*DS_NL[1][0][Mj_AN][kl][n][L+6]
	      -ene_m*3.0/7.0*DS_NL[1][0][Mc_AN][k][m][L+6]*DS_NL[1][0][Mj_AN][kl][n][L+5];

	    /* contribution of l-1/2 for up spin */ 
	    sum0.i += tmp0;

	    /* contribution of l-1/2 for down spin */ 
	    sum1.i += -tmp0;
	  }

	  /****************************************************
                 diagonal contribution on up-up and dn-dn
	  ****************************************************/

	  for (L3=0; L3<=L2; L3++){

	    /* VNL for j=l+1/2 */
	    sum0.r += PFp*ene_p*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];
	    sum1.r += PFp*ene_p*DS_NL[0][0][Mc_AN][k][m][L]*DS_NL[0][0][Mj_AN][kl][n][L];

	    /* VNL for j=l-1/2 */
	    sum0.r += PFm*ene_m*DS_NL[1][0][Mc_AN][k][m][L]*DS_NL[1][0][Mj_AN][kl][n][L];
	    sum1.r += PFm*ene_m*DS_NL[1][0][Mc_AN][k][m][L]*DS_NL[1][0][Mj_AN][kl][n][L];

	    L++;
	  }

	}

	NLH[0][m][n].r += sum0.r;    /* <up|VNL|up> */
	NLH[1][m][n].r += sum1.r;    /* <dn|VNL|dn> */
	NLH[2][m][n].r += sum2.r;    /* <up|VNL|dn> */

	NLH[0][m][n].i += sum0.i;    /* <up|VNL|up> */
	NLH[1][m][n].i += sum1.i;    /* <dn|VNL|dn> */
	NLH[2][m][n].i += sum2.i;    /* <up|VNL|dn> */

      } /* n */
    } /* m */

  } /* else if */

}





double NLRF_BesselF(int Gensi, int L, int so, double R)
{
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,f;
  double result;

  mp_min = 0;
  mp_max = Ngrid_NormK - 1;
  po = 0;

  if (R<NormK[0]){
    m = 1;
  }
  else if (NormK[mp_max]<R){
    result = 0.0;
    po = 1;
  }
  else{
    do{
      m = (mp_min + mp_max)/2;
      if (NormK[m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;
  }

  /****************************************************
                 Spline like interpolation
  ****************************************************/

  if (po==0){

    if (m==1){
      h2 = NormK[m]   - NormK[m-1];
      h3 = NormK[m+1] - NormK[m];

      f2 = Spe_NLRF_Bessel[so][Gensi][L][m-1];
      f3 = Spe_NLRF_Bessel[so][Gensi][L][m];
      f4 = Spe_NLRF_Bessel[so][Gensi][L][m+1];

      h1 = -(h2+h3);
      f1 = f4;
    }
    else if (m==(Ngrid_NormK-1)){
      h1 = NormK[m-1] - NormK[m-2];
      h2 = NormK[m]   - NormK[m-1];

      f1 = Spe_NLRF_Bessel[so][Gensi][L][m-2];
      f2 = Spe_NLRF_Bessel[so][Gensi][L][m-1];
      f3 = Spe_NLRF_Bessel[so][Gensi][L][m];

      h3 = -(h1+h2);
      f4 = f1;
    }
    else{
      h1 = NormK[m-1] - NormK[m-2];
      h2 = NormK[m]   - NormK[m-1];
      h3 = NormK[m+1] - NormK[m];

      f1 = Spe_NLRF_Bessel[so][Gensi][L][m-2];
      f2 = Spe_NLRF_Bessel[so][Gensi][L][m-1];
      f3 = Spe_NLRF_Bessel[so][Gensi][L][m];
      f4 = Spe_NLRF_Bessel[so][Gensi][L][m+1];
    }

    /****************************************************
                Calculate the value at R
    ****************************************************/

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = R - NormK[m-1];
    x2 = R - NormK[m];
    y1 = x1/h2;
    y2 = x2/h2;

    f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

    result = f;
  }
  
  return result;
}




