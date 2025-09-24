/**********************************************************************
  Generate_Wannier.c:

  code for calculating the maximally localized Wannier Function.

  Log of Generete_Wannier.c:

     24/Mar/2008   Started by Hongming Weng
     06/Jun/2008   Start the coding for disentangling of mixed bands
     23/Sep/2015   Interface with wannier90 is added by Fumiyuki Ishii
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 

#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"
#include <omp.h>
 

#define printout  0    /* 0:off, 1:on */
#define measure_time   0
#define dste_flag      2
#define AU2Debye   2.54174776  
#define AU2Mucm    5721.52891433 /* 100 e/bohr/bohr */
#define BohrR_Wannier    0.529177249              /* Angstrom */
#define eV2Hartree 27.2113845
#define smallvalue   1.0e-6 /*smallvalue close to zero*/
#define BUFFSIZE    2048

#define debug1  0  /* Interpolating */
#define debug2  0  /* k grids */
#define debug3  0  /* initial guess */
#define debug4  0  /* for Mmnk_zero */
#define debug5  0  /* for updating Ukmatrix */
#define debug6  0  /* for wannier center */
#define debug7  0  /* for b vectors */
#define debugcg 1
#define debugdis 0 /* for disentangling */
#define debugdis_z 0 /* for disentangling */
#define MIN(a,b) ((a)<(b))?  (a):(b)


struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};
int output_opt_mmn;
double r_latt[3][3];
double ***sheet;
dcomplex ***csheet;
double **rguide;
dcomplex *****Wannier_Coef; 

int *Total_NumOrbs;
int *FNAN_WP;
int **natn_WP;
int **ncn_WP;
double ****OLP_WP;
dcomplex *****OLPe;

double ****fOLP;
double *****fHks;
double *****fiHks;




/* hmweng */
/**************************************** For bvectors ******************************************/                                  
int Cal_Weight_of_Shell(double **klatt, int *M_s, int **bvector, int *num_shell, 
			double *wb, int *Reject_Shell, int *searched_shell);
                         
void Shell_Structure(double **klatt, int *M_s, int **bvector, int *shell_num, int MAXSHELL);

static void Ascend_Ordering(double *xyz_value, int *ordering, int tot_atom);

/**************************************** For bvectors ******************************************/                                  

void EigenState_k(double k1, double k2, double k3, int *MP, int spinsize, 
                 int SpinP_switch, int fsize, int fsize2, int fsize3,  
                 dcomplex ***Wk1, double **EigenVal1, 
      	         double ****OLP, double *****Hks, double *****iHks);
                           
void Wannier_Center(double **wann_center, double *wann_r2, dcomplex ****Mmnkb,int wan_num, double **bvector, double *wb, int kpt_num, int tot_bvector);
void Output_WF_Spread(char *mode, double **wann_center, double *wann_r2, double omega_I, double omega_OD, double omega_D, int WANNUM);
void Projection_Amatrix(dcomplex ****Amnk, double **kg, int spinsize, int fsize, int SpinP_switch, int kpt_num, int band_num,  int wan_num, dcomplex ****Wkall, int *MP, int ***Nk);
void Getting_Uk(dcomplex ****Amnk, int spinsize, int kpt_num, int band_num, int wan_num, dcomplex **** Uk, int ***Nk);
void Udis_projector_frozen(dcomplex ****Uk,int ***Nk,int ***innerNk,int kpt_num, int spinsize, int band_num, int wan_num);
void Disentangling_Bands(dcomplex ****Uk, dcomplex *****Mmnkb_zero, int spinsize, int kpt_num, int band_num, int wan_num, int ***Nk, int ***innerNk, double **kg, double **bvector, double *wb, int **kplusb, int tot_bvector, double ***eigen);
void Getting_Zmatrix(dcomplex **Zmat,int presentk, double *wb,int tot_bvector,dcomplex ****Mmnkb, dcomplex ***Udis, int **kplusb, int band_num, int wan_num, int Mk, int **nbandwin, int **nbandfroz);
void Initial_Guess_Mmnkb(dcomplex ****Uk, int spinsize, int kpt_num, int band_num, int wan_num, dcomplex *****Mmnkb, dcomplex *****Mmnkb_zero, double **kg, double **bvector, int tot_bvector, int **kplusb, int ***Nk);
void Getting_Utilde(dcomplex ****Amnk, int spinsize, int kpt_num, int band_num, int wan_num, dcomplex ****Uk);
void Updating_Mmnkb(dcomplex ***Uk, int kpt_num, int band_num, int wan_num, dcomplex ****Mmnkb, dcomplex ****Mmnkb_zero, double **kg, double **bvector, int tot_bvector, int **kplusb);
void Cal_Omega(double *omega_I, double *omega_OD, double *omega_D, dcomplex ****Mmnkb,
	       double **wann_center, int wan_num, double **bvector, double *wb, int kpt_num, int tot_bvector);
void Cal_Gradient(dcomplex ***Gmnk,dcomplex ****Mmnkb, double **wann_center, double **bvector, double *wb, int wan_num, int kpt_num, int tot_bvector, int *zero_Gk);
void Cal_Ukmatrix(dcomplex ***deltaW, dcomplex ***Ukmatrix, int kpt_num, int wan_num);
void Calc_rT_dot_d(dcomplex *rmnk, dcomplex *dmnk, int n, double *amod);
void Gradient_at_next_x(dcomplex ***g, double wbtot, double alpha, dcomplex ***Uk, dcomplex ****m, int kpt_num, int WANNUM, double **kg, double **bvector, double **frac_bv, double *wb, int tot_bvector, int **kplusb, dcomplex *rimnk, double *omega_I, double *omega_OD, double *omega_D);

void new_Gradient(dcomplex ****m, int kpt_num, int WANNUM, double **bvector, double *wb, int tot_bvector, dcomplex *rimnk, double *omega_I, double *omega_OD, double *omega_D, double **wann_center, double *wann_r2);

void Taking_one_step(dcomplex ***Gmnk, double wbtot, double alpha, dcomplex ***Ukmatrix, dcomplex ****Mmnkb,
                     dcomplex ****Mmnkb_zero, int kpt_num, int WANNUM, double **kg, double **frac_bv,
                     int tot_bvector, int **kplusb);
void Wannier_Interpolation(dcomplex ****Uk, double ***eigen, int spinsize, int SpinP_switch, int kpt_num, int band_num, int wan_num, int ***Nk, double rtv[4][4], double **kg, double ChemP, int r_num, int **rvect, int *ndegen);
void MLWF_Coef(dcomplex ****Uk,dcomplex ****Wkall,int ***Nk, int *MP, int kpt_num, int fsize3, int fsize, int spinsize,int SpinP_switch, int band_num,int wan_num,double **kg, int r_num, int **rvect, dcomplex *****Wannier_Coef);
void Wigner_Seitz_Vectors(double metric[3][3], double rtv[4][4], int knum_i, int knum_j, int knum_k, int *r_num, int **rvect, int *ndegen);
void Center_Guide(dcomplex ****Mmnkb, int WANNUM, int *bdirection, int nbdir, int kpt_num, double **bvector, int tot_bvector);
double Invert3_mat(double smat[3][3], double **sinv);

void OutData_WF(char *inputfile, int rnum, int **rvect);
void Calc_SpinDirection_WF(int rnum, int **rvect);

void MPI_comm_OLP_Hks_iHks(double ****OLP, double *****Hks, double *****iHks);

void Calc_Mmnkb(int k, int kk, double k1[4], double k2[4], double b[4], 
                int bindx, int SpinP_switch, int *MP, double Sop[2], 
		dcomplex *Wk1, dcomplex *Wk2, int fsize, int mm, int nn);

void Overlap_Band_Wannier(double ****OLP,
			  dcomplex **S,int *MP,
			  double k1, double k2, double k3);

void Hamiltonian_Band_Wannier(double ****RH, dcomplex **H, int *MP,
			      double k1, double k2, double k3);

void Hamiltonian_Band_NC_Wannier(double *****RH, double *****IH,
		 		 dcomplex **H, int *MP,
				 double k1, double k2, double k3);

void Calc_OLP_exp();
void Find_NN_Projectors_Basis();
void Set_OLP_Projector_Basis(double ****OLP_WP);


void Wannier(int Solver, 
	     int atomnum, 
	     double ChemP, 
	     double E_Temp, 
	     int Valence_Electrons,
	     double Total_SpinS, 
	     int SpinP_switch, 
	     int *Total_NumOrbs, 
	     double tv[4][4],
	     double rtv[4][4], 
	     double **Gxyz,
	     int *FNAN,
	     int **natn,
	     int **ncn,
	     double **atv,
	     int **atv_ijk,
	     double *****Hks,
	     double *****iHks,
	     double ****OLP,
             int WANNUM,
             double oemin,
             double oemax,
             double iemin,
             double iemax,
             int lprojection,
             int knum_i, int knum_j, int knum_k,
             int MAXSHELL,
             char **Wannier_ProSpeName,
             char **Wannier_ProName,
             double **Wannier_Pos,
             double **Wannier_Z_Direction, 
             double **Wannier_X_Direction, 
             int max_steps,
             char *inputfile);





#pragma optimization_level 1
void Generate_Wannier(char *inputfile)
{
  int ct_AN,wan1,TNO1;
  int p,num,L,h_AN,Gh_AN,wan,tno,i;
  double ****tmpOLP;
  int tnum,num2;

  /* allocations of arrays */

  Total_NumOrbs = (int*)malloc(sizeof(int)*(atomnum+1));
  FNAN_WP = (int*)malloc(sizeof(int)*Wannier_Num_Kinds_Projectors);

  /* set of Total_NumOrbs */

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    wan1 = WhatSpecies[ct_AN];
    TNO1 = Spe_Total_CNO[wan1];
    Total_NumOrbs[ct_AN] = TNO1;
  }

  /* analyze data structure of overlap between projectors and basis functions */

    Find_NN_Projectors_Basis();

    /* allocate array */

    OLP_WP = (double****)malloc(sizeof(double***)*Wannier_Num_Kinds_Projectors);
    tmpOLP = (double****)malloc(sizeof(double***)*Wannier_Num_Kinds_Projectors);

    for (p=0; p<Wannier_Num_Kinds_Projectors; p++){

      num = 0;
      for (L=0; L<=3; L++){
        num += (2*L+1)*Wannier_NumL_Pro[p][L];
      }

      OLP_WP[p] = (double***)malloc(sizeof(double**)*FNAN_WP[p]);

      tmpOLP[p] = (double***)malloc(sizeof(double**)*FNAN_WP[p]);
      tnum  = WannierPro2SpeciesNum[p]; 
      num2= Spe_Total_CNO[tnum];
   
      for (h_AN=0; h_AN<FNAN_WP[p]; h_AN++){

        Gh_AN = natn_WP[p][h_AN];
        wan = WhatSpecies[Gh_AN];
        tno = Spe_Total_CNO[wan];  

        OLP_WP[p][h_AN] = (double**)malloc(sizeof(double*)*num);

        tmpOLP[p][h_AN] = (double**)malloc(sizeof(double*)*num2);

        for (i=0; i<num; i++){
          OLP_WP[p][h_AN][i] = (double*)malloc(sizeof(double)*tno);
        }

        for (i=0; i<num2; i++){
          tmpOLP[p][h_AN][i] = (double*)malloc(sizeof(double)*tno);
        }
  
      }
    }

    /* calculate overlap matrix between projectors and basis functions */

    Set_OLP_Projector_Basis(tmpOLP);

    for (p=0; p<Wannier_Num_Kinds_Projectors; p++){

      num = 0;
      for (L=0; L<=3; L++){
        num += (2*L+1)*Wannier_NumL_Pro[p][L];
      }
      tnum  = WannierPro2SpeciesNum[p];
      num2= Spe_Total_CNO[tnum];
      for(h_AN=0; h_AN<FNAN_WP[p]; h_AN++){
        Gh_AN = natn_WP[p][h_AN];
        wan = WhatSpecies[Gh_AN];
        tno = Spe_Total_CNO[wan];
        i=0; tnum=0; 
        for (L=0; L<=3; L++){
          if(Wannier_NumL_Pro[p][L]!=0){
            for(wan=0;wan<2*L+1;wan++){
              for(Gh_AN=0;Gh_AN<tno;Gh_AN++){
                OLP_WP[p][h_AN][i][Gh_AN] = tmpOLP[p][h_AN][tnum][Gh_AN];
              }
              i++;
              tnum++;
            }
          }else{
            tnum+=2*L+1;
          }
        }
      }
    }
    for (p=0; p<Wannier_Num_Kinds_Projectors; p++){
      tnum  = WannierPro2SpeciesNum[p];
      num2= Spe_Total_CNO[tnum];
      for(h_AN=0; h_AN<FNAN_WP[p]; h_AN++){
        Gh_AN = natn_WP[p][h_AN];
        wan = WhatSpecies[Gh_AN];
        tno = Spe_Total_CNO[wan];
        for (i=0; i<num2; i++){
          free(tmpOLP[p][h_AN][i]);
        }
        free(tmpOLP[p][h_AN]);
      }
      free(tmpOLP[p]);
    }
    free(tmpOLP);
  
  /* calculate Wannier functions */

  Wannier(Solver,
          atomnum,
          ChemP,
          E_Temp, 
	  Total_Num_Electrons,
	  Total_SpinS, 
	  SpinP_switch, 
	  Total_NumOrbs, 
	  tv,
	  rtv,
	  Gxyz,
	  FNAN,
	  natn,
	  ncn,
	  atv,
	  atv_ijk,
	  H,
	  iHNL,
	  OLP[0],
          Wannier_Func_Num,
          Wannier_Outer_Window_Bottom,
          Wannier_Outer_Window_Top,
          Wannier_Inner_Window_Bottom,
          Wannier_Inner_Window_Top,
          Wannier_Initial_Guess,
          Wannier_grid1, Wannier_grid2, Wannier_grid3, 
          Wannier_MaxShells,
          Wannier_ProSpeName,
          Wannier_ProName,
          Wannier_Pos,
          Wannier_Z_Direction,
          Wannier_X_Direction,
          Wannier_Minimizing_Max_Steps, inputfile);

  free(Total_NumOrbs);

  if(Wannier_Output_Projection_Matrix==1){ /* hmweng */ 
    for (p=0; p<Wannier_Num_Kinds_Projectors; p++){
      free(natn_WP[p]);
    }
    free(natn_WP);

    for (p=0; p<Wannier_Num_Kinds_Projectors; p++){
      free(ncn_WP[p]);
    }
    free(ncn_WP);

    for (p=0; p<Wannier_Num_Kinds_Projectors; p++){

      num = 0;
      for (L=0; L<=3; L++){
	num += (2*L+1)*Wannier_NumL_Pro[p][L];
      }
   
      for (h_AN=0; h_AN<FNAN_WP[p]; h_AN++){
	for (i=0; i<num; i++){
	  free(OLP_WP[p][h_AN][i]);
	}
	free(OLP_WP[p][h_AN]);
      }
      free(OLP_WP[p]);
    }
    free(OLP_WP);
  } /*hmweng */
  free(FNAN_WP);
}






#pragma optimization_level 1
void Set_OLP_Projector_Basis(double ****OLP_WP)
{
  /****************************************************
          Evaluate overlap and kinetic integrals
                 in the momentum space
  ****************************************************/
  static int firsttime=1;
  int l,m,L0,L1,Ls,M0,M1,Mul0,Mul1,k,p;
  int Mc_AN,Gc_AN,h_AN,Gh_AN,Rnh,i,j,Cwan,Hwan,num0,num1; 
  int Lmax_Four_Int,size_SumS0,size_TmpOLP;
  double dx,dy,dz,Normk,Normk2,kmin,kmax,Sk,Dk;
  double gant,r,theta,phi,SH[2],dSHt[2],dSHp[2];
  double Bessel_Pro0,Bessel_Pro1;
  double tmp0,tmp1,tmp2,tmp3,tmp4;
  double ReC0,ImC0,ReC1,ImC1,sj,sy,sjp,syp;
  double ReC2,ImC2,ReC3,ImC3,coe0,coe1,h,time0;
  double ****SumS0;
  double **SphB,**SphBp;
  double *tmp_SphB,*tmp_SphBp;
  double S_coordinate[3],siT,coT,siP,coP;
  dcomplex ******TmpOLP;

  dcomplex CsumS0,CsumSr,CsumSt,CsumSp;
  dcomplex Ctmp0,Ctmp1,Ctmp2,Cpow;
  dcomplex CY,CYt,CYp,CY1,CYt1,CYp1;
  dcomplex **CmatS0;
  double TStime,TEtime;
  int numprocs,myid;
  double Stime_atom, Etime_atom;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
   MPI_Barrier
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  /****************************************************************
  allocation of arrays:
  ****************************************************************/

  SumS0 = (double****)malloc(sizeof(double***)*(List_YOUSO[25]+1));
  for (i=0; i<(List_YOUSO[25]+1); i++){
    SumS0[i] = (double***)malloc(sizeof(double**)*List_YOUSO[24]);
    for (j=0; j<List_YOUSO[24]; j++){
      SumS0[i][j] = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (k=0; k<(List_YOUSO[25]+1); k++){
        SumS0[i][j][k] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
        for (m=0; m<List_YOUSO[24]; m++) SumS0[i][j][k][m] = 0.0;       
      }
    }
  }

  TmpOLP = (dcomplex******)malloc(sizeof(dcomplex*****)*(List_YOUSO[25]+1));
  for (i=0; i<(List_YOUSO[25]+1); i++){
    TmpOLP[i] = (dcomplex*****)malloc(sizeof(dcomplex****)*List_YOUSO[24]);
    for (j=0; j<List_YOUSO[24]; j++){
      TmpOLP[i][j] = (dcomplex****)malloc(sizeof(dcomplex***)*(2*(List_YOUSO[25]+1)+1));
      for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
        TmpOLP[i][j][k] = (dcomplex***)malloc(sizeof(dcomplex**)*(List_YOUSO[25]+1));
        for (l=0; l<(List_YOUSO[25]+1); l++){
          TmpOLP[i][j][k][l] = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[24]);
          for (m=0; m<List_YOUSO[24]; m++){
            TmpOLP[i][j][k][l][m] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
            for (p=0; p<(2*(List_YOUSO[25]+1)+1); p++) TmpOLP[i][j][k][l][m][p] = Complex(0.0,0.0);
	  }
	}
      }
    }
  }

  CmatS0 = (dcomplex**)malloc(sizeof(dcomplex*)*(2*(List_YOUSO[25]+1)+1));
  for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
    CmatS0[i] = (dcomplex*)malloc(sizeof(dcomplex)*(2*(List_YOUSO[25]+1)+1));
    for (p=0; p<(2*(List_YOUSO[25]+1)+1); p++) CmatS0[i][p] = Complex(0.0,0.0);
  }

  for (Mc_AN=0; Mc_AN<Wannier_Num_Kinds_Projectors; Mc_AN++){

    Gc_AN = Mc_AN;
    Cwan  = WannierPro2SpeciesNum[Gc_AN];

    for (h_AN=0; h_AN<FNAN_WP[Gc_AN]; h_AN++){

      Gh_AN = natn_WP[Gc_AN][h_AN];
      Rnh = ncn_WP[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];

      dx = Gxyz[Gh_AN][1] + atv[Rnh][1] - Wannier_Pos[Gc_AN][1]; 
      dy = Gxyz[Gh_AN][2] + atv[Rnh][2] - Wannier_Pos[Gc_AN][2]; 
      dz = Gxyz[Gh_AN][3] + atv[Rnh][3] - Wannier_Pos[Gc_AN][3];

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

      /****************************************************
          overlap integral
              \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
      ****************************************************/

      kmin = Radial_kmin;
      kmax = PAO_Nkmax;
      Sk = kmax + kmin;
      Dk = kmax - kmin;

      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	  for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){
                  TmpOLP[L0][Mul0][L0+M0][L1][Mul1][L1+M1]  = Complex(0.0,0.0);
		}
	      }
	    }
	  }
	}
      }

      if (Spe_MaxL_Basis[Cwan]<Spe_MaxL_Basis[Hwan])
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Hwan];
      else 
        Lmax_Four_Int = 2*Spe_MaxL_Basis[Cwan];

      /* allocate SphB and SphBp */

      SphB = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(l=0; l<(Lmax_Four_Int+3); l++){ 
        SphB[l] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
      }

      SphBp = (double**)malloc(sizeof(double*)*(Lmax_Four_Int+3));
      for(l=0; l<(Lmax_Four_Int+3); l++){ 
        SphBp[l] = (double*)malloc(sizeof(double)*(OneD_Grid+1));
      }
      
      tmp_SphB  = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));
      tmp_SphBp = (double*)malloc(sizeof(double)*(Lmax_Four_Int+3));

      /* calculate SphB and SphBp */

      h = (kmax - kmin)/(double)OneD_Grid;

      for (i=0; i<=OneD_Grid; i++){
        Normk = kmin + (double)i*h;
        Spherical_Bessel(Normk*r,Lmax_Four_Int,tmp_SphB,tmp_SphBp);
        for(l=0; l<=Lmax_Four_Int; l++){ 
          SphB[l][i]  = tmp_SphB[l]; 
          SphBp[l][i] = tmp_SphBp[l]; 
	}
      }

      free(tmp_SphB);
      free(tmp_SphBp);

      /* l loop */

      for(l=0; l<=Lmax_Four_Int; l++){

        for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
            for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	      for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
                SumS0[L0][Mul0][L1][Mul1] = 0.0;
	      }
	    }
	  }
	}

        h = (kmax - kmin)/(double)OneD_Grid;
	for (i=0; i<=OneD_Grid; i++){

          if (i==0 || i==OneD_Grid) coe0 = 0.50;
          else                      coe0 = 1.00;

	  Normk = kmin + (double)i*h;
          Normk2 = Normk*Normk;

          sj  =  SphB[l][i];
          sjp = SphBp[l][i];

	  for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){

	      Bessel_Pro0 = RF_BesselF(Cwan,L0,Mul0,Normk);

              tmp0 = coe0*h*Normk2*Bessel_Pro0;
              tmp1 = tmp0*sj;
              tmp2 = tmp0*Normk*sjp;

	      for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
		for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){

                  Bessel_Pro1 = RF_BesselF(Hwan,L1,Mul1,Normk);

                  tmp3 = tmp1*Bessel_Pro1;
                  tmp4 = tmp2*Bessel_Pro1;

                  SumS0[L0][Mul0][L1][Mul1] += tmp3;
		}
	      }

	    }
	  }
	}

        /****************************************************
          For overlap
          sum_m 8*(-i)^{-L0+L1+1}*
                C_{L0,-M0,L1,M1,l,m}*Y_{lm}
                \int RL(k)*RL'(k)*jl(k*R) k^2 dk^3,
        ****************************************************/

        for(m=-l; m<=l; m++){ 

          ComplexSH(l,m,theta,phi,SH,dSHt,dSHp);
          SH[1]   = -SH[1];
          dSHt[1] = -dSHt[1];
          dSHp[1] = -dSHp[1];

          CY  = Complex(SH[0],SH[1]);
          CYt = Complex(dSHt[0],dSHt[1]);
          CYp = Complex(dSHp[0],dSHp[1]);

          for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
            for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
              for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
                for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){

                  Ls = -L0 + L1 + l;  

                  if (abs(L1-l)<=L0 && L0<=(L1+l) ){

		    Cpow = Im_pow(-1,Ls);
		    CY1  = Cmul(Cpow,CY);
		    CYt1 = Cmul(Cpow,CYt);
		    CYp1 = Cmul(Cpow,CYp);

		    for (M0=-L0; M0<=L0; M0++){

		      M1 = M0 - m;

                      if (abs(M1)<=L1){

                        gant = Gaunt(L0,M0,L1,M1,l,m);

                        /* S */ 

                        tmp0 = gant*SumS0[L0][Mul0][L1][Mul1];
                        Ctmp2 = CRmul(CY1,tmp0);
                        TmpOLP[L0][Mul0][L0+M0][L1][Mul1][L1+M1] =
			  Cadd(TmpOLP[L0][Mul0][L0+M0][L1][Mul1][L1+M1],Ctmp2);
		      }
		    }
		  }
		}
	      }

	    }
	  }
        } 
      } /* l */

      /* free SphB and SphBp */

      for(l=0; l<(Lmax_Four_Int+3); l++){ 
        free(SphB[l]);
      }
      free(SphB);

      for(l=0; l<(Lmax_Four_Int+3); l++){ 
        free(SphBp[l]);
      }
      free(SphBp);

      /****************************************************
                         Complex to Real
      ****************************************************/

      num0 = 0;
      for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_Basis[Cwan][L0]; Mul0++){
          
          num1 = 0;
          for (L1=0; L1<=Spe_MaxL_Basis[Hwan]; L1++){
	    for (Mul1=0; Mul1<Spe_Num_Basis[Hwan][L1]; Mul1++){
              
    	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

                  CsumS0 = Complex(0.0,0.0);

     	          for (k=-L0; k<=L0; k++){

                    Ctmp1 = Conjg(Comp2Real[L0][L0+M0][L0+k]);

                    /* S */

                    Ctmp0 = TmpOLP[L0][Mul0][L0+k][L1][Mul1][L1+M1];
                    Ctmp2 = Cmul(Ctmp1,Ctmp0);
                    CsumS0 = Cadd(CsumS0,Ctmp2);
		  }

                  CmatS0[L0+M0][L1+M1] = CsumS0;
		}
	      }

    	      for (M0=-L0; M0<=L0; M0++){
		for (M1=-L1; M1<=L1; M1++){

                  CsumS0 = Complex(0.0,0.0);

     	          for (k=-L1; k<=L1; k++){

                    /* S */ 

                    Ctmp1 = Cmul(CmatS0[L0+M0][L1+k],Comp2Real[L1][L1+M1][L1+k]);
                    CsumS0 = Cadd(CsumS0,Ctmp1);
		  }

                  /* add a small vaule for stabilization of eigenvalue routine */
		  /* printf("Project kind %i, basis num0+L0+M0=%i\n",Gc_AN,num0+L0+M0 );fflush(0); */
                  OLP_WP[Gc_AN][h_AN][num0+L0+M0][num1+L1+M1] = 8.0*CsumS0.r;

		}
	      }

              num1 = num1 + 2*L1 + 1; 
	    }
	  }

          num0 = num0 + 2*L0 + 1; 
	}
      }
    }
  }

  /****************************************************
               freeing of arrays:
  ****************************************************/

  for (i=0; i<(List_YOUSO[25]+1); i++){
    for (j=0; j<List_YOUSO[24]; j++){
      for (k=0; k<(List_YOUSO[25]+1); k++){
        free(SumS0[i][j][k]);
      }
      free(SumS0[i][j]);
    }
    free(SumS0[i]);
  }
  free(SumS0);

  for (i=0; i<(List_YOUSO[25]+1); i++){
    for (j=0; j<List_YOUSO[24]; j++){
      for (k=0; k<(2*(List_YOUSO[25]+1)+1); k++){
        for (l=0; l<(List_YOUSO[25]+1); l++){
          for (m=0; m<List_YOUSO[24]; m++){
            free(TmpOLP[i][j][k][l][m]);
	  }
          free(TmpOLP[i][j][k][l]);
	}
        free(TmpOLP[i][j][k]);
      }
      free(TmpOLP[i][j]);
    }
    free(TmpOLP[i]);
  }
  free(TmpOLP);

  for (i=0; i<(2*(List_YOUSO[25]+1)+1); i++){
    free(CmatS0[i]);
  }
  free(CmatS0);
}


#pragma optimization_level 1
void Find_NN_Projectors_Basis()
{
  int p,Rn,i,wan,k;
  double r1,r2,dx,dy,dz,r;  

  /* find FNAN_WP */
   
  for (p=0; p<Wannier_Num_Kinds_Projectors; p++){

    FNAN_WP[p] = 0;

    k = WannierPro2SpeciesNum[p];
    r1 = Spe_Atom_Cut1[k];

    for (Rn=0; Rn<=TCpyCell; Rn++){

      for (i=1; i<=atomnum; i++){

	wan = WhatSpecies[i];
	r2 = Spe_Atom_Cut1[wan];

	dx = Wannier_Pos[p][1] - Gxyz[i][1] - atv[Rn][1];
	dy = Wannier_Pos[p][2] - Gxyz[i][2] - atv[Rn][2];
	dz = Wannier_Pos[p][3] - Gxyz[i][3] - atv[Rn][3];

        r = sqrt( fabs(dx*dx + dy*dy + dz*dz) );

        if (r<=(r1+r2)){
          FNAN_WP[p]++;
        }
      }
    }
  }

  /* allocation of arrays */

  natn_WP = (int**)malloc(sizeof(int*)*Wannier_Num_Kinds_Projectors);
  for (p=0; p<Wannier_Num_Kinds_Projectors; p++){
    natn_WP[p] = (int*)malloc(sizeof(int)*FNAN_WP[p]);
  }

  ncn_WP = (int**)malloc(sizeof(int*)*Wannier_Num_Kinds_Projectors);
  for (p=0; p<Wannier_Num_Kinds_Projectors; p++){
    ncn_WP[p] = (int*)malloc(sizeof(int)*FNAN_WP[p]);
  }

  /* find natn_WP and ncn_WP */

  for (p=0; p<Wannier_Num_Kinds_Projectors; p++){

    FNAN_WP[p] = 0;

    k = WannierPro2SpeciesNum[p];
    r1 = Spe_Atom_Cut1[k];

    for (Rn=0; Rn<=TCpyCell; Rn++){

      for (i=1; i<=atomnum; i++){

	wan = WhatSpecies[i];
	r2 = Spe_Atom_Cut1[wan];

	dx = Wannier_Pos[p][1] - Gxyz[i][1] - atv[Rn][1];
	dy = Wannier_Pos[p][2] - Gxyz[i][2] - atv[Rn][2];
	dz = Wannier_Pos[p][3] - Gxyz[i][3] - atv[Rn][3];

        r = sqrt( fabs(dx*dx + dy*dy + dz*dz) );

        if (r<=(r1+r2)){
          natn_WP[p][FNAN_WP[p]] = i;
          ncn_WP[p][FNAN_WP[p]]  = Rn;
          FNAN_WP[p]++;
        }
      }
    }

  }

}



#pragma optimization_level 1
void Calc_OLP_exp(int tot_bvector, double **bvector)
{
  int Mc_AN,Gc_AN,tno0,tno1;
  int Cwan,h_AN,Gh_AN,Hwan;
  int b,i,j,num0,num;
  int numprocs,myid,ID;
  int succeed_open;
  int NO0,NO1,Nc,GNc,GRc;
  int Mh_AN,Rnh,Nog,Nh;
  double x,y,z;
  double Cxyz[4];
  double mbdotr,co,si;
  double *vec0;
  dcomplex **ChiV1;
  dcomplex *****OLP_exp;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of OLP_exp */

  OLP_exp = (dcomplex*****)malloc(sizeof(dcomplex****)*tot_bvector);
  for (b=0; b<tot_bvector; b++){

    OLP_exp[b] = (dcomplex****)malloc(sizeof(dcomplex***)*(Matomnum+1)); 

    FNAN[0] = 0;

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = S_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  
      }   

      OLP_exp[b][Mc_AN] = (dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[Gc_AN]+1)); 

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	} 
        
	OLP_exp[b][Mc_AN][h_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0);
	for (i=0; i<tno0; i++){
	  OLP_exp[b][Mc_AN][h_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno1); 
          for (j=0; j<tno1; j++){
            OLP_exp[b][Mc_AN][h_AN][i][j].r = 0.0;
            OLP_exp[b][Mc_AN][h_AN][i][j].i = 0.0;
	  }
	}
      }
    }
  }

  /****************************************************
    allocation of arrays:
  ****************************************************/

  ChiV1 = (dcomplex**)malloc(sizeof(dcomplex*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiV1[i] = (dcomplex*)malloc(sizeof(dcomplex)*List_YOUSO[11]);
  }

  /*****************************************************
      calculation of matrix elements for OLP_exp
  *****************************************************/

  for (b=0; b<tot_bvector; b++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan];
    
      for (i=0; i<NO0; i++){
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	  GNc = GridListAtom[Mc_AN][Nc];
	  GRc = CellListAtom[Mc_AN][Nc];

	  Get_Grid_XYZ(GNc,Cxyz);
	  x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
	  y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
	  z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

          mbdotr = -(bvector[b][0]*x + bvector[b][1]*y + bvector[b][2]*z);
          co = cos(mbdotr);
          si = sin(mbdotr);

	  ChiV1[i][Nc].r = co*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	  ChiV1[i][Nc].i = si*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */

	}
      }
    
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];

	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* integration */

	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
          
          if (G2ID[Gh_AN]==myid){
	    for (i=0; i<NO0; i++){
	      for (j=0; j<NO1; j++){
		OLP_exp[b][Mc_AN][h_AN][i][j].r += ChiV1[i][Nc].r*Orbs_Grid[Mh_AN][Nh][j]*GridVol;/*AITUNE*/
		OLP_exp[b][Mc_AN][h_AN][i][j].i += ChiV1[i][Nc].i*Orbs_Grid[Mh_AN][Nh][j]*GridVol;/*AITUNE*/
	      }
	    }
	  }
          else{
	    for (i=0; i<NO0; i++){
	      for (j=0; j<NO1; j++){
		OLP_exp[b][Mc_AN][h_AN][i][j].r += ChiV1[i][Nc].r*Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j]*GridVol;/*AITUNE*/
		OLP_exp[b][Mc_AN][h_AN][i][j].i += ChiV1[i][Nc].i*Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j]*GridVol;/*AITUNE*/
	      }
	    }
          } 

	}

      }
    }
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiV1[i]);
  }
  free(ChiV1);
 
  /****************************************************
    allocation of OLPe
  ****************************************************/

  OLPe = (dcomplex*****)malloc(sizeof(dcomplex****)*tot_bvector);
  for (b=0; b<tot_bvector; b++){

    OLPe[b] = (dcomplex****)malloc(sizeof(dcomplex***)*(atomnum+1)); 

    FNAN[0] = 0;

    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

      if (Gc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  
      }   

      OLPe[b][Gc_AN] = (dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[Gc_AN]+1)); 

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Gc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	} 
        
	OLPe[b][Gc_AN][h_AN] = (dcomplex**)malloc(sizeof(dcomplex*)*tno0);
	for (i=0; i<tno0; i++){
	  OLPe[b][Gc_AN][h_AN][i] = (dcomplex*)malloc(sizeof(dcomplex)*tno1); 
          for (j=0; j<tno1; j++){
            OLPe[b][Gc_AN][h_AN][i][j].r = 0.0;
            OLPe[b][Gc_AN][h_AN][i][j].i = 0.0;
	  }
	}
      }
    }
  }

  /****************************************************
    MPI commucation of OLPe
  ****************************************************/

  num0 = 2*tot_bvector*List_YOUSO[2]*List_YOUSO[7]*List_YOUSO[7];
  vec0 = (double*)malloc(sizeof(double)*num0);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Spe_Total_CNO[Cwan];  

    if (myid==ID){

      Mc_AN = F_G2M[Gc_AN];

      num = 0;
      for (b=0; b<tot_bvector; b++){
	for (i=0; i<tno0; i++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];

	    for (j=0; j<tno1; j++){
	      vec0[num] = OLP_exp[b][Mc_AN][h_AN][i][j].r;  num++;
	      vec0[num] = OLP_exp[b][Mc_AN][h_AN][i][j].i;  num++;
	    }
	  }
	}
      }    
    }

    MPI_Bcast(&num, 1, MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&vec0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0;
    for (b=0; b<tot_bvector; b++){
      for (i=0; i<tno0; i++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];

	  for (j=0; j<tno1; j++){
            OLPe[b][Gc_AN][h_AN][i][j].r = vec0[num];  num++;
            OLPe[b][Gc_AN][h_AN][i][j].i = vec0[num];  num++;
	  }
	}
      }
    }    
     
  } /* Gc_AN */   

  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(vec0);

  for (b=0; b<tot_bvector; b++){

    FNAN[0] = 0;

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = S_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  
      }   

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	} 
        
	for (i=0; i<tno0; i++){
	  free(OLP_exp[b][Mc_AN][h_AN][i]);
	}
        free(OLP_exp[b][Mc_AN][h_AN]);
      }
      free(OLP_exp[b][Mc_AN]);
    }
    free(OLP_exp[b]);
  }
  free(OLP_exp);

}









#pragma optimization_level 1
void Wannier(int Solver, 
	     int atomnum, 
	     double ChemP, 
	     double E_Temp, 
	     int Valence_Electrons,
	     double Total_SpinS, 
	     int SpinP_switch, 
	     int *Total_NumOrbs, 
	     double tv[4][4],
	     double rtv[4][4], 
	     double **Gxyz,
	     int *FNAN,
	     int **natn,
	     int **ncn,
	     double **atv,
	     int **atv_ijk,
	     double *****Hks,
	     double *****iHks,
	     double ****OLP,
             int WANNUM,
             double oemin,
             double oemax,
             double iemin,
             double iemax,
             int lprojection,
             int knum_i, int knum_j, int knum_k,
             int MAXSHELL,
             char **Wannier_ProSpeName,
             char **Wannier_ProName,
             double **Wannier_Pos,
             double **Wannier_Z_Direction, 
             double **Wannier_X_Direction, 
             int max_steps,
             char *inputfile)
{
  int fsize,fsize2,fsize3,fsize4;
  int spin,spinsize,diag_flag;
  int i,j,k,l,hos,po,wan,hog2,valnonmag;
  int n1,n2,n3,i1,i2,i3,itmp;
  int itmp1,itmp2,itmp3,itmp4,m,n,m1;
  int kloop[4][4];
  int pflag[4];
  int hog[1];
  int Gc_AN,tno0,Cwan,tno1;
  int Gh_AN,Hwan;
  int odloop,odloop_num;
  double b[4],ktp[4];
  double pol_abc[4];
  double Edpx,Edpy,Edpz;
  double Cdpx,Cdpy,Cdpz,Cdpi[4];
  double Tdpx,Tdpy,Tdpz;
  double Bdpx,Bdpy,Bdpz;
  double Ptx,Pty,Ptz;
  double Pex,Pey,Pez;
  double Pcx,Pcy,Pcz;
  double Pbx,Pby,Pbz;
  double AbsD,Pion0;
  double Gabs[4],Parb[4];
  double Phidden[4], Phidden0;
  double tmpr,tmpi,sumr,sumi;
  double mulr[2],muli[2]; 
  double **kg;
  double sum,d;
  double norm;
  double pele,piony,pall;
  double gdd;
  double CellV;
  double Cell_Volume;
  double psi,sumpsi[2];
  double detr;
  double TZ;
  double Cxyz[4];
  int ct_AN,h_AN;
  char *s_vec[20];
  int *MP;  

  dcomplex ****Wkall;
/* this is the original overlap matrix with random phase factor */ 
  dcomplex *****Mmnkb_zero; 
/* This is the optimized overlap matrix after initial guess */
  dcomplex *****Mmnkb_dis; 

  dcomplex ****M_zero,****Mmnkb; 
  dcomplex ****Amnk, ****Uk, ****Utilde;
  int lwritek,lreadMmnk;
  dcomplex ***Smmk;
  double ***EigenValall;
 /* for gradient of spread function */
  dcomplex ***Gmnk, *rmnk, *dmnk, *rimnk;
  double delta_new, delta_0, delta_old, delta_mid, delta_d;
  double yita,  yita_prev,  beta;
  double epcg, epsec, sigma_0; 

  /* For optimization */
  int searching_scheme; /* 0 -- fixed finite step using Steepest Decent
                           1 -- CG method with secant linear searching scheme 
                           2 -- Hybrid scheme, firstly SD then CG */
 /* used in fixed finite step scheme */
  double alpha,conv_total_omega;
 /* whether gradients is zero or not */
  int zero_Gk;
  dcomplex ***deltaW,***Ukmatrix, **tmpM;
 /* the maximum steps to be taken */
  int step;
  int sec_step, sec_max ;

  /* for file operation*/
  FILE *fp,*fpkpt,*fpkpt2,*fpwn90, *fptmp,*fptmp2;
  char  *dumy;
  char fname[300];
  char fname2[300];
  char fname3[300];
  char fname4[300];

  int BANDNUM, MkNUM;
  int kpt_num, ki, kj, kk, fband_num;
  double *wb, *tmp_wb, wbtot;
  double **bvector, dkx, dky, dkz, **klatt;
  double **frac_bv;
  int **kplusb, *M_s, shell_num, tot_bvector, *tmp_M_s, **tmp_bvector, *Reject_Shell;

/* bdirection[nbdir] gives the total number of bvectors without inversion symmetry
                     and gives the vector index in frac_bv matrix */
  int *bdirection, nbdir; 
  int bindx, nindx, startbv,find_w,tmp_shellnum,searched_shell;
   
  /* Wannier */
 /* Wannier Function center, r_bar_n, defined in 
    Eq. 31. wann_center[spinsize][nindx][0:2] */
  double **wann_center;
  double *wann_r2;
  double sumr2;
 /* spread of wannier fucntion */
  double omega, omega_I, omega_OD, omega_D;
  double omega_prev, omega_I_prev, omega_OD_prev, omega_D_prev;
  double omega_next, omega_I_next, omega_OD_next, omega_D_next;

  /* for disentangling of mixed bands */    
 /* the outer window boundaries at each k point*/
  int ***Nk; 
 /* lower and upper boundary of inner window */
  int ***innerNk;
  int ***excludeBand;
  int have_frozen; 
  double ***eigen;
  /* MPI_Comm comm1; */
  int numprocs,myid,ID,ID1;
  double TStime,TEtime;

  /* for MPI*/

  int AB_knum,S_knum,E_knum,num_ABloop0;
  int ik1,ik2,ik3,ABloop,ABloop0,abcount;
  double tmp4;
  double *psiAB;

  int r_num, *ndegen;
  int **rvect;
  double metric[3][3];

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI initialize */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("\n*****************************************************************************\n");
    printf("*****************************************************************************\n"); 
    printf(" Wannier:\n"); 
    printf(" code for calculating the maximally localized wannier function\n"); 
    printf(" Copyright (C), 2006-2019, Hongming Weng, Taisuke Ozaki and Kiyoyuki Terakura\n"); 
    printf(" This is free software, and you are welcome to         \n"); 
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("******************************************************************************\n"); 
    printf("******************************************************************************\n"); 
  }

  s_vec[0]="Recursion";     s_vec[1]="Cluster"; s_vec[2]="Band";
  s_vec[3]="NEGF";          s_vec[4]="DC";      s_vec[5]="GDC";
  s_vec[6]="Cluster-DIIS";  s_vec[7]="Krylov";  s_vec[8]="Cluster2";
  s_vec[9]="EGAC";  

  if (myid==Host_ID){
    printf(" Previous eigenvalue solver = %s\n",s_vec[Solver-1]);
    printf(" atomnum                    = %i\n",atomnum);
    printf(" ChemP                      = %15.12f (Hartree)\n",ChemP);
    printf(" Total_SpinS                = %15.12f (K)\n",Total_SpinS);
/*
    if(BohrR_Wannier == 0.529177249){
      printf("All the unit related to WF will be in Angstrom.\n",BohrR_Wannier);
    }else{
      printf("All the unit related to WF will be in Bohr.\n",BohrR_Wannier);
    } 
*/
  }
  s_vec[0]="collinear spin-unpolarized";
  s_vec[1]="collinear spin-polarized";
  s_vec[3]="non-collinear";
  if (myid==Host_ID){
    printf(" Spin treatment             = %s\n",s_vec[SpinP_switch]);
  }

  /******************************************
      find the size of the full matrix 
  ******************************************/

  /* MP:
     a pointer which shows the starting number
     of basis orbitals associated with atom i
     in the full matrix 
  */

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  
  fsize = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = fsize;
    fsize += Total_NumOrbs[i];
  }
  fsize--;
 
  if      (SpinP_switch==0){ spinsize=1; fsize2=fsize;  fsize3=fsize+2; fsize4=Valence_Electrons/2;}
  else if (SpinP_switch==1){ spinsize=2; fsize2=fsize;  fsize3=fsize+2; 
    fsize4=Valence_Electrons/2+abs(floor(Total_SpinS))*2+1;}
  else if (SpinP_switch==3){ spinsize=1; fsize2=2*fsize;fsize3=2*fsize+2; fsize4=Valence_Electrons;}

  /******************************************
              the standard output
  ******************************************/
 
  if (myid==Host_ID){
    printf("\n r-space primitive vector (Bohr)\n");
    printf("  tv1=%10.6f %10.6f %10.6f\n",tv[1][1], tv[1][2], tv[1][3]);
    printf("  tv2=%10.6f %10.6f %10.6f\n",tv[2][1], tv[2][2], tv[2][3]);
    printf("  tv3=%10.6f %10.6f %10.6f\n",tv[3][1], tv[3][2], tv[3][3]);
    printf(" k-space primitive vector (Bohr^-1)\n");
    printf("  rtv1=%10.6f %10.6f %10.6f\n",rtv[1][1], rtv[1][2], rtv[1][3]);
    printf("  rtv2=%10.6f %10.6f %10.6f\n",rtv[2][1], rtv[2][2], rtv[2][3]);
    printf("  rtv3=%10.6f %10.6f %10.6f\n\n",rtv[3][1], rtv[3][2], rtv[3][3]);
    rtv[0][0]=sqrt(rtv[1][1]*rtv[1][1]+rtv[1][2]*rtv[1][2]+rtv[1][3]*rtv[1][3]);
    rtv[0][1]=sqrt(rtv[2][1]*rtv[2][1]+rtv[2][2]*rtv[2][2]+rtv[2][3]*rtv[2][3]);
    rtv[0][2]=sqrt(rtv[3][1]*rtv[3][1]+rtv[3][2]*rtv[3][2]+rtv[3][3]*rtv[3][3]);
  }
  
  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      r_latt[i-1][j-1] = tv[i][j]; 
    }
  }

  po = 0;
/* for Wannier90, F.Ishii*/ 
  if(Wannier90_fileout){
  Wannier_Dis_SCF_Max_Steps=0;
  Wannier_Minimizing_Max_Steps=0;
  }


  /* show and set parameters */

  if (myid==Host_ID){

    printf("Wannier Function number %2d\n",WANNUM);
    printf("K grids %ix%ix%i\n",knum_i,knum_j,knum_k);      

    printf("Iteration criteria for disentangling %e\n",Wannier_Dis_Conv_Criterion);
    printf("Maximum number of steps for disentangling bands is %i.\n",Wannier_Dis_SCF_Max_Steps);
    printf("Mixing parameter for disentangling iteration %10.5f.\n",Wannier_Dis_Mixing_Para); 

    if(Wannier_Min_Scheme==0){
      printf("Using Steepest Decent (SD) method for Minimization.\n");
      printf("In minimization step length is %10.5f\n",Wannier_Min_StepLength); 
    }else if(Wannier_Min_Scheme==1){
      printf("Using Conjugate Gradient (CG) method for Minimization.\n");
    }else if(Wannier_Min_Scheme==2){
      printf("Using Hybrid Steepest Decent (SD) and Conjugate Gradient (CG) methods for Minimization.\n");
    }
    if(Wannier_Min_Scheme==1 || Wannier_Min_Scheme==2){    
      printf("In CG: intial search step length for Secant method %10.5f.\n",fabs(Wannier_Min_Secant_StepLength));
      printf("In CG: Number of steps for Secant method %i.\n",Wannier_Min_Secant_Steps);  
    }
    printf("Criteria for minimization %e\n",Wannier_Min_Conv_Criterion);
    printf("Maximum number of minimization steps %i.\n",max_steps);
  }
/* not unsed presently 
    lwritek = Wannier_Output_kmesh;
    printf("Write k point information into file? %2d\n",lwritek);
*/
    lwritek = 0;
 
  lreadMmnk = Wannier_Readin_Overlap_Matrix;   

  if (myid==Host_ID){ 
    printf("Out-put original overlap matrix into %s%s.mmn file? %2d\n",filepath,filename,Wannier_Output_Overlap_Matrix);
    printf("Out-put initial guess projectors into %s%s.amn file? %2d\n",filepath,filename,Wannier_Output_Projection_Matrix);
    printf("Read in Overlap Matrix from file? %2d\n",lreadMmnk);
    printf("Outer window bottom(in eV, relative to Fermi level) is %10.5f\n",oemin);
    printf("Outer window top(in eV, relative to Fermi level) is %10.5f\n",oemax);
    printf("Inner window bottom(in eV, relative to Fermi level) is %10.5f\n",iemin);
    printf("Inner window top(in eV, relative to Fermi level) is %10.5f\n",iemax);
  }
  if(oemax-oemin<=0.0){
    if (myid==Host_ID){
       printf("Error:WF For OUTER window, its top (%10.5f) should be higher than bottom (%10.5f).\n",oemin,oemax); 
    }
    MPI_Finalize();
    exit(0); 
  }
  if(iemax-iemin<0.0){
    if (myid==Host_ID){
       printf("Error:WF For INNER window, its top (%10.5f) should be higher than bottom (%10.5f).\n",iemin,iemax);
    }
    MPI_Finalize();
    exit(0);
  }
  if(iemin<oemin || iemax>oemax){
    if (myid==Host_ID){
       printf("Error:WF INNER window (%10.5f,%10.5f) must be inside of OUTER window (%10.5f,%10.5f).\n",iemin,iemax,oemin,oemax);
    }
    MPI_Finalize();
    exit(0);
  }

  have_frozen=1;

  if(fabs(iemin-iemax)<smallvalue){
    iemax=oemin-999999.0;
    iemin=iemax;
    have_frozen=0; 
  }

  if (myid==Host_ID){

    printf("Using initial guess for optimization? %2d\n",lprojection);
     
    if(Wannier_Draw_MLWF){
      printf("Maximally Localized Wannier Function will be plotted.\n");
      printf("WF will be ploted in supercell size: %ix%ix%i.\n",2*Wannier_Plot_SuperCells[0]+1,
                                                                2*Wannier_Plot_SuperCells[1]+1,
                                                                2*Wannier_Plot_SuperCells[2]+1);
    }
    if(Wannier_Draw_Int_Bands){
      printf("Interpolated bands will be plotted.\n");
    }
  }  

  /* before calculation, check Wannier.Kgrid
     Sometime some settings of Wannier.Kgrid will cause failure in finding
     proper Wigner-Seitz supercell for making interpolation. Avoid this if
     interpolation is needed. 
  */
  /* metric is a matrix as following:
     a \dot a      a \dot b       a \dot c
     b \dot a      b \dot b       b \dot c
     c \dot a      c \dot b       c \dot c
  */
  for(j=0;j<3;j++){
    for(i=0;i<=j;i++){
      sumr=0.0;
      for(l=0;l<3;l++){
	sumr=sumr+tv[i+1][l+1]*tv[j+1][l+1];
      }
      metric[i][j]=sumr;
      if(i<j){
	metric[j][i]=metric[i][j];
      }
    }
  }
  r_num=1;
  rvect=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rvect[i]=(int*)malloc(sizeof(int)*r_num);
  }
  ndegen=(int*)malloc(sizeof(int)*r_num);

  r_num=-1;
  Wigner_Seitz_Vectors(metric, rtv, knum_i, knum_j, knum_k, &r_num, rvect, ndegen);

  free(ndegen);
  for(i=0;i<3;i++){
    free(rvect[i]);
  }
  free(rvect);

  rvect=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rvect[i]=(int*)malloc(sizeof(int)*r_num);
  }

  ndegen=(int*)malloc(sizeof(int)*r_num);

  r_num=0;
  Wigner_Seitz_Vectors(metric, rtv,  knum_i, knum_j, knum_k, &r_num, rvect, ndegen);

  /* Taking regular mesh */
  kpt_num=knum_i*knum_j*knum_k;

  kg = (double**)malloc(sizeof(double*)*kpt_num);
  for (i=0; i<kpt_num; i++){
    kg[i] = (double*)malloc(sizeof(double)*3);
  }
  
  /* MPI_Barrier(comm1); */

  sum=0.0;
  
  kpt_num=0;

  for(i=knum_i;i>=1;i--){
    for(j=knum_j;j>=1;j--){
      for(k=knum_k;k>=1;k--){
	kg[kpt_num][0]=(double)(knum_i-i)/(double)knum_i+1E-12;
	kg[kpt_num][1]=(double)(knum_j-j)/(double)knum_j+1E-12;
	kg[kpt_num][2]=(double)(knum_k-k)/(double)knum_k+1E-12;
	kpt_num++;
      }
    }
  } 




  /* Find the bvectors */

  tmp_M_s=(int*)malloc(sizeof(int)*MAXSHELL);
  for(i=0;i<MAXSHELL;i++){
    tmp_M_s[i]=0;
  }
    	
  tmp_bvector=(int**)malloc(sizeof(int*)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
    tmp_bvector[i]=(int*)malloc(sizeof(int)*3);
    for(j=0;j<3;j++){
      tmp_bvector[i][j]=0;
    }
  }
  klatt=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    klatt[i]=(double*)malloc(sizeof(double)*3);
  }
  klatt[0][0]=rtv[1][1]/(double)knum_i;
  klatt[0][1]=rtv[1][2]/(double)knum_i;
  klatt[0][2]=rtv[1][3]/(double)knum_i;
  klatt[1][0]=rtv[2][1]/(double)knum_j;
  klatt[1][1]=rtv[2][2]/(double)knum_j;
  klatt[1][2]=rtv[2][3]/(double)knum_j;
  klatt[2][0]=rtv[3][1]/(double)knum_k;
  klatt[2][1]=rtv[3][2]/(double)knum_k;
  klatt[2][2]=rtv[3][3]/(double)knum_k;

  /* Find the shell structure of k points */
  Shell_Structure(klatt, tmp_M_s, tmp_bvector, &shell_num, MAXSHELL);

  if(shell_num==0){
    if (myid==Host_ID){
        printf("******************************Error********************************\n");
        printf("*    Can not find proper b vectors, please increase parameter     *\n");
        printf("*    MAXSHELL OR change Wannier.Kgrids.                           *\n");
        printf("******************************Error********************************\n");
        printf("***********************************INFO**************************************\n");
        printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",rtv[0][0],rtv[0][1],rtv[0][2]);
        printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
        printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
        printf("************************************INFO*************************************\n");
    }
    for(i=0;i<3;i++){
      free(klatt[i]);
    }
    free(klatt);
    for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
      free(tmp_bvector[i]);
    }
    free(tmp_bvector);
    free(tmp_M_s);
    MPI_Finalize();
    exit(0);
  }

  tot_bvector=0;
  for(i=0;i<shell_num;i++){
    tot_bvector=tot_bvector+tmp_M_s[i];
  }
    	
  Reject_Shell=(int*)malloc(sizeof(int)*shell_num);
  for(i=0;i<shell_num;i++){
    Reject_Shell[i]=0;
  }
    	
  tmp_wb=(double*)malloc(sizeof(double)*shell_num);
  tmp_shellnum=shell_num;
  searched_shell=0;
  find_w=0;
  find_w=Cal_Weight_of_Shell(klatt, tmp_M_s, tmp_bvector, &shell_num, tmp_wb, Reject_Shell, &searched_shell) ;
  if(find_w==0){/* wb is not found within these shells */
    if (myid==Host_ID){
       printf("*************************** Error ****************************\n");
       printf("*    Weights for b vectors (totally %3d) are not found.      *\n",tmp_shellnum);
       printf("*    Please increase MAXSHELL (presently it is %3d) OR       *\n",MAXSHELL);
       printf("*    change Wannier.Kgrids and try again                     *\n");
       printf("*************************** Error ****************************\n");
       printf("***********************************INFO**************************************\n");
       printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",rtv[0][0],rtv[0][1],rtv[0][2]);            printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
       printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
       printf("************************************INFO*************************************\n");
    }
    free(tmp_wb);
    free(Reject_Shell);
    for(i=0;i<3;i++){
      free(klatt[i]);
    }
    free(klatt);
    for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
      free(tmp_bvector[i]);
    }
    free(tmp_bvector);
    free(tmp_M_s);
    MPI_Finalize();
    exit(0);
  }else{/*success in finding wb */
    /* Rearrange M_s and bvector arrays to remove other unnecessary shells */
    tot_bvector=0;
    M_s=(int*)malloc(sizeof(int)*shell_num);
    j=0;
    for(i=0;i<searched_shell;i++){
      if(Reject_Shell[i]==0){
	M_s[j]=tmp_M_s[i];
	tot_bvector=tot_bvector+M_s[j];
	j++;
      }
    }

    wb=(double*)malloc(sizeof(double)*tot_bvector);
    k=0;
    for(i=0;i<shell_num;i++){
      for(j=0;j<M_s[i];j++){
	wb[k]=tmp_wb[i];
	k++;
      }
    }

    bvector=(double**)malloc(sizeof(double*)*tot_bvector);
    for(i=0;i<tot_bvector;i++){
      bvector[i]=(double*)malloc(sizeof(double)*3);
      for(j=0;j<3;j++){
	bvector[i][j]=0.0;
      }
    }

    k=0; 

    for(i=0;i<searched_shell;i++){

      if(Reject_Shell[i]==0){

	startbv=0;

	if(i==0){
	  startbv=0;
	}
        else{
	  for(j=0; j<i; j++){
	    startbv=startbv+tmp_M_s[j];
	  }
	}

	for(j=startbv;j<startbv+tmp_M_s[i];j++){
	  bvector[k][0]=(double)tmp_bvector[j][0]/(double)knum_i;
	  bvector[k][1]=(double)tmp_bvector[j][1]/(double)knum_j;
	  bvector[k][2]=(double)tmp_bvector[j][2]/(double)knum_k;
	  k++;
	}

      }/* not rejected shell */    	
    }
                
    /* Now free tmp_M_s, tmp_bvector, tmp_wb, Reject_shell */ 

    free(tmp_wb);
    free(Reject_Shell);
    for(i=0;i<3;i++){
      free(klatt[i]);
    }
    free(klatt);

    for(i=0;i<(8*MAXSHELL*MAXSHELL*MAXSHELL);i++){
      free(tmp_bvector[i]);
    }
    free(tmp_bvector);
    free(tmp_M_s);

    if (myid==Host_ID){
       printf("There are %2d shells and total number of b vectors is %3d\n",shell_num,tot_bvector);
       fflush(0);
    }
    frac_bv=(double**)malloc(sizeof(double*)*tot_bvector);
    for(i=0;i<tot_bvector;i++){
      frac_bv[i]=(double*)malloc(sizeof(double)*3);
    }
   
    tot_bvector=0;
    wbtot=0.0;
    for(i=0;i<shell_num;i++){
      if (myid==Host_ID){
         printf("Shell %2d has %2d b vectors:\n",i+1,M_s[i]);
         if(BohrR_Wannier == 1.0){ 
           printf("No.|      Fractional Coordinate     || Cartesian Coordinate (Bohr^-1)||Weight_b(Bohr^2)||\n");     
         }else{
           printf("No.|      Fractional Coordinate     || Cartesian Coordinate (Angs^-1)||Weight_b(Angs^2)||\n");     
         }
      } 
      for(j=0;j<M_s[i];j++){
	frac_bv[tot_bvector][0]=bvector[tot_bvector][0];
	frac_bv[tot_bvector][1]=bvector[tot_bvector][1];
	frac_bv[tot_bvector][2]=bvector[tot_bvector][2];
	dkx = bvector[tot_bvector][0]*rtv[1][1] + bvector[tot_bvector][1]*rtv[2][1] + bvector[tot_bvector][2]*rtv[3][1];
	dky = bvector[tot_bvector][0]*rtv[1][2] + bvector[tot_bvector][1]*rtv[2][2] + bvector[tot_bvector][2]*rtv[3][2];
	dkz = bvector[tot_bvector][0]*rtv[1][3] + bvector[tot_bvector][1]*rtv[2][3] + bvector[tot_bvector][2]*rtv[3][3];
        if (myid==Host_ID){
           printf(" %2d|  (%8.5f,%8.5f,%8.5f)  ||  (%8.5f,%8.5f,%8.5f) || %13.5f  ||\n",
                    j+1,   bvector[tot_bvector][0],bvector[tot_bvector][1],bvector[tot_bvector][2],
                    dkx/BohrR_Wannier,dky/BohrR_Wannier,dkz/BohrR_Wannier,
                    wb[tot_bvector]*BohrR_Wannier*BohrR_Wannier); 
        }
	bvector[tot_bvector][0]=dkx;
	bvector[tot_bvector][1]=dky;
	bvector[tot_bvector][2]=dkz;
	wbtot=wbtot+wb[tot_bvector];
	tot_bvector++;
      }
    }
  }
  /* determine the b vectors list without inversion symmetry */
  bdirection=(int*)malloc(sizeof(int)*tot_bvector);
  for(i=0;i<tot_bvector;i++){
    bdirection[i]=-1;
  }
  nbdir=0;

  for(bindx=0;bindx<tot_bvector;bindx++){
    k=0;
    if(nbdir==0){ /* if the list still empty, put the first b into it */
      bdirection[nbdir]=bindx;
      nbdir++;
    }else{ /* compare the existing b in the list with all b vectors, to find new one */
      for(i=0;i<nbdir;i++){
        j=bdirection[i];
        if((frac_bv[bindx][0]+frac_bv[j][0])*(frac_bv[bindx][0]+frac_bv[j][0])+
           (frac_bv[bindx][1]+frac_bv[j][1])*(frac_bv[bindx][1]+frac_bv[j][1])+
           (frac_bv[bindx][2]+frac_bv[j][2])*(frac_bv[bindx][2]+frac_bv[j][2])<1.0e-8){
	  k=1;
        }
      }
      if(k==0){
        bdirection[nbdir]=bindx;
        nbdir++;
      }
    }
  }    
/* this is not necessary to show */
  if (myid==Host_ID && 0==1){
    printf("There are %i b vectors after deleting those having inversion symmetry.\n",nbdir);
    if(BohrR_Wannier == 1.0){
      printf("No.|      Fractional Coordinate     || Cartesian Coordinate (Bohr^-1)||Weight_b(Bohr^2)||\n");
    }else{
      printf("No.|      Fractional Coordinate     || Cartesian Coordinate (Angs^-1)||Weight_b(Angs^2)||\n");
    }
  }  
  for(i=0;i<nbdir;i++){
    k=bdirection[i];
    if (myid==Host_ID && 0==1){
      printf(" %2d|  (%8.5f,%8.5f,%8.5f)  ||  (%8.5f,%8.5f,%8.5f) || %10.5f  ||\n",i+1,frac_bv[k][0],frac_bv[k][1],frac_bv[k][2],bvector[k][0],bvector[k][1],bvector[k][2],wb[k]);
      fflush(0);
    }
  }

  if(lwritek){
    if (myid==Host_ID){
      sprintf(fname,"%s%s.kpt",filepath,filename);
      if((fpkpt=fopen(fname,"wt"))==NULL){
        printf("Error in opening %s for writing k point information.\n",fname);
        exit(0);
      }
      fprintf(fpkpt,"%i  %i\n",kpt_num,tot_bvector);
      i=0;
      for(k=0;k<tot_bvector;k++){
        fprintf(fpkpt, "%10.8f %10.8f  %10.8f  %10.8f\n",wb[k], bvector[k][0],bvector[k][1],bvector[k][2]);
      }
    }
  } 
 
  kplusb=(int**)malloc(sizeof(int*)*kpt_num);
  for(k=0;k<kpt_num;k++){
    kplusb[k]=(int*)malloc(sizeof(int)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      /* k+b */
      ktp[1]=kg[k][0]+frac_bv[bindx][0];  
      ktp[2]=kg[k][1]+frac_bv[bindx][1];
      ktp[3]=kg[k][2]+frac_bv[bindx][2];
      for(i=1;i<=3;i++){
	b[i]=ktp[i];
	if(ktp[i]>=1.0){
	  b[i]=ktp[i]-1.0;
	}
	if(ktp[i]<0.0){
	  b[i]=ktp[i]+1.0;
	}
      }
      dkx=b[1]; dky=b[2]; dkz=b[3];
      kj=0;kk=-1;
      for(ki=0;ki<kpt_num;ki++){
	if(fabs(dkx-kg[ki][0])<smallvalue && fabs(dky-kg[ki][1])<smallvalue && fabs(dkz-kg[ki][2])<smallvalue){
	  kj=1;
	  kk=ki;
	  break;
	}
      }
      if(kj==0){
        if (myid==Host_ID){
   	  printf("***************************** Error *********************************\n");
	  printf("* Check Wannier.Kgrids, the equivalent k points for k+b not found.  *\n");
	  printf("***************************** Error *********************************\n");
          printf("***********************************INFO**************************************\n");
          printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",rtv[0][0],rtv[0][1],rtv[0][2]);            printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
          printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
          printf("************************************INFO*************************************\n");
        }
        MPI_Finalize();
        exit(0);	
      }
      kplusb[k][bindx]=kk;
      if(lwritek){ /* write the nnkpts block */
       if (myid==Host_ID){
  	 fprintf(fpkpt,"%5d%5d%5d%5d%5d\n",k+1,kk+1,(int)(ktp[1]-b[1]),(int)(ktp[2]-b[2]),(int)(ktp[3]-b[3]));
       }
      }
    }
  }
  if(lwritek){/* write nnkpts block */
    if (myid==Host_ID){
      for(k=0;k<kpt_num;k++){
        fprintf(fpkpt,"%12.8f%12.8f%12.8f%12.8f\n",kg[k][0],kg[k][1],kg[k][2],1.0/(double)(kpt_num));
      }
      fclose(fpkpt);
    }
  }

  /* k grids ok */

  /* for entangled bands */
  Nk = (int***)malloc(sizeof(int**)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    Nk[spin]=(int**)malloc(sizeof(int*)*kpt_num);
    for(k=0;k<kpt_num;k++){
      Nk[spin][k]=(int*)malloc(sizeof(int)*2);
    } 
  }
  innerNk = (int***)malloc(sizeof(int**)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    innerNk[spin]=(int**)malloc(sizeof(int*)*kpt_num);
    for(k=0;k<kpt_num;k++){
      innerNk[spin][k]=(int*)malloc(sizeof(int)*2);
    }
  }
/* not used at present */
  excludeBand = (int***)malloc(sizeof(int**)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    excludeBand[spin]=(int**)malloc(sizeof(int*)*kpt_num);
    for(k=0;k<kpt_num;k++){
      excludeBand[spin][k]=(int*)malloc(sizeof(int)*2);
    }
  }
  excludeBand[0][0][0]=1;
  excludeBand[0][0][1]=2; 

  /* Wkall: 
     wave functions at all k points 
     Wave function Wkall just stores the c coefficient Cn,i,alpha (k)
     Dimension analysis: 1. Spin size;
     2. band index m,n, basis size;
     3. linear combination coefficient of each basis, basis size;
     4. k dependence is also included.
  */

  Wkall = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num); 
  for(k=0;k<kpt_num;k++){
    Wkall[k] = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize); 
    for (spin=0; spin<spinsize; spin++){
      Wkall[k][spin] = (dcomplex**)malloc(sizeof(dcomplex*)*(fsize3)); 
      for (i=0; i<fsize3; i++){
	Wkall[k][spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
	for (j=0; j<fsize3; j++){ Wkall[k][spin][i][j].r=0.0; Wkall[k][spin][i][j].i=0.0;}
      } 
    }
  }
 
  /* EigenValall:
     eigenvalues at all k points 
  */

  EigenValall = (double***)malloc(sizeof(double**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    EigenValall[k] = (double**)malloc(sizeof(double*)*spinsize);
    for (spin=0; spin<spinsize; spin++){
      EigenValall[k][spin] = (double*)malloc(sizeof(double)*(fsize3));
      for (j=0; j<fsize3; j++) EigenValall[k][spin][j] = 1.0e9;
    }
  }

  /*************************************************
       calculate eigenvalue and wave-function
  *************************************************/

  if (lreadMmnk==0){   

    /* calculation of <i\alpha|exp(-ib\dot r)|j\beta> */

    Calc_OLP_exp( tot_bvector, bvector );
    if (myid==Host_ID){
      printf("Calculating the eigenvalues and eigenvectors......\n"); fflush(0);
    }
    /* MPI commutation of OLP, Hks, iHks */

    MPI_comm_OLP_Hks_iHks( OLP, Hks, iHks );

    /* For each k point: find its eigenvalue and wavefunction */

#pragma omp parallel shared(myid,kpt_num,numprocs,kg,MP,spinsize,SpinP_switch,fsize,fsize2,fsize3,Wkall,EigenValall,OLP,Hks,iHks) private(k,OMPID,Nthrds,Nprocs)
    {
    
    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for( k=myid+numprocs*OMPID; k<kpt_num; k+=numprocs*Nthrds ){

      EigenState_k(kg[k][0], kg[k][1], kg[k][2], MP, spinsize, SpinP_switch, 
		   fsize, fsize2, fsize3, Wkall[k], EigenValall[k], OLP, Hks, iHks );

    } /* For each k point: find its eigenvalue and wavefunction */
    } /* #pragma omp parallel */

    /* MPI communicatio of Wkall */

    { 
      int num0,num;
      double *vec0;
   
      num0 = 2*spinsize*fsize3*fsize3;
      vec0 = (double*)malloc(sizeof(double)*num0);

      for( k=0; k<kpt_num; k++ ){
      
	ID = k % numprocs;

	num = 0;
	for (spin=0; spin<spinsize; spin++){
	  for (i=0; i<fsize3; i++){
	    for (j=0; j<fsize3; j++){ 
	      vec0[num] = Wkall[k][spin][i][j].r; num++;
	      vec0[num] = Wkall[k][spin][i][j].i; num++;
	    }
	  }
	}

	MPI_Bcast(&vec0[0], num0, MPI_DOUBLE, ID, mpi_comm_level1);

	num = 0;
	for (spin=0; spin<spinsize; spin++){
	  for (i=0; i<fsize3; i++){
	    for (j=0; j<fsize3; j++){ 
	      Wkall[k][spin][i][j].r = vec0[num]; num++;
	      Wkall[k][spin][i][j].i = vec0[num]; num++;
	    }
	  }
	}

      }

      free(vec0);
    }

    /* MPI communicatio of EigenValall */

    {
      int num0,num;
      double *vec0;
   
      num0 = spinsize*fsize3;
      vec0 = (double*)malloc(sizeof(double)*num0);

      for( k=0; k<kpt_num; k++ ){
      
	ID = k % numprocs;

	num = 0;
	for (spin=0; spin<spinsize; spin++){
	  for (i=0; i<fsize3; i++){
	    vec0[num] = EigenValall[k][spin][i]; num++;
	  }
	}

	MPI_Bcast(&vec0[0], num0, MPI_DOUBLE, ID, mpi_comm_level1);

	num = 0;
	for (spin=0; spin<spinsize; spin++){
	  for (i=0; i<fsize3; i++){
	    EigenValall[k][spin][i] = vec0[num]; num++;
	  }
	}

      }

      free(vec0);
    }

    /* freeing of fOLP */

    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

      if (Gc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  
      }   

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Gc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	} 

	for (i=0; i<tno0; i++){
	  free(fOLP[Gc_AN][h_AN][i]);
	}
        free(fOLP[Gc_AN][h_AN]);
      }
      free(fOLP[Gc_AN]);
    }
    free(fOLP);

    /* freeing of fHks */

    for (k=0; k<=SpinP_switch; k++){

      for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

	if (Gc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}   

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Gc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 
        
	  for (i=0; i<tno0; i++){
	    free(fHks[k][Gc_AN][h_AN][i]);
	  }
	  free(fHks[k][Gc_AN][h_AN]);

	}
	free(fHks[k][Gc_AN]);
      }
      free(fHks[k]);
    }
    free(fHks);

    /* freeing of fiHks */

    if (SpinP_switch==3){

      for (k=0; k<List_YOUSO[5]; k++){

	for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

	  if (Gc_AN==0){
	    Gc_AN = 0;
	    tno0 = 1;
	  }
	  else{
	    Cwan = WhatSpecies[Gc_AN];
	    tno0 = Spe_Total_CNO[Cwan];  
	  }   

	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	    if (Gc_AN==0){
	      tno1 = 1;  
	    }
	    else{
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_CNO[Hwan];
	    } 

	    for (i=0; i<tno0; i++){
	      free(fiHks[k][Gc_AN][h_AN][i]);
	    }
            free(fiHks[k][Gc_AN][h_AN]);
	  }
          free(fiHks[k][Gc_AN]);
	}
        free(fiHks[k]);
      }
      free(fiHks);
    }
  } /* if (lreadMmnk==0) */

  /*************************************************
     read eigenvalue and wave-function from file
  *************************************************/

  else { 

    sprintf(fname,"%s%s.eigen",filepath,filename);
    dumy=(char*)malloc(sizeof(char)*BUFFSIZE);

    if((fptmp=fopen(fname,"rt"))==NULL){
      printf("***************************Error**********************************\n");
      printf("*  Error in opening file for reading eigen-values and states.   *\n");
      printf("***************************Error**********************************\n");
      MPI_Finalize();
      exit(0);
    }

    else{
      if (myid==Host_ID){
        printf("\nReading eigenvalues and eigenvectors from file.\n");fflush(0);
      }
      fread(dumy,sizeof(char),11,fptmp);
      fscanf(fptmp,"%lf",&ChemP);
      fgets(dumy, BUFFSIZE, fptmp);     
      if (myid==Host_ID){
        printf("Efermi is %lf\n",ChemP);
      }
      fread(dumy,sizeof(char),15,fptmp);
      fscanf(fptmp,"%d",&BANDNUM); 
      if (myid==Host_ID){
        printf("Totally there are %i bands for each k point.\n",BANDNUM);fflush(0);
      }
      fgets(dumy, BUFFSIZE, fptmp);     

      for(spin=0;spin<spinsize;spin++){
	for(k=0;k<kpt_num;k++){
	  /* printf("kpt %i bands:\n",k,Nk[spin][k][0],Nk[spin][k][1]);  */
	  for(i=0;i<BANDNUM;i++){
	    fscanf(fptmp,"%d%d%lf",&ki,&kj,&EigenValall[k][spin][i+1]);
	    /*	    printf("%i %10.5f\n",i+1,EigenValall[k][spin][i+1]);  */
	  }
	}
      }
      fgets(dumy, BUFFSIZE, fptmp);    
      for(spin=0;spin<spinsize;spin++){
        for(k=0;k<kpt_num;k++){
          fgets(dumy, BUFFSIZE, fptmp);    
	  /* printf("%s\n",dumy); */
	  /* fprintf(fptmp, "WF kpt %i (%10.8f,%10.8f,%10.8f)\n",k+1,kg[k][0],kg[k][1],kg[k][2]); */
          for(i=1; i<BANDNUM+1; i++){
            for(j=1;j<fsize3; j++){
              fscanf(fptmp,"%d%d%lf%lf", 
                            &ki, &kj, 
                            &Wkall[k][spin][i][j].r,
                            &Wkall[k][spin][i][j].i);
	      /*  printf("%i %i %14.10f %14.10f\n",i, j,Wkall[k][spin][i][j].r,Wkall[k][spin][i][j].i); */
            }
          }
          fgets(dumy, BUFFSIZE, fptmp);    
        }
      }
      fclose(fptmp); 
    } 
    free(dumy);
  } 

  /* now determine the bands within the outer and inner energy window */

  oemin=oemin/eV2Hartree+ChemP;
  oemax=oemax/eV2Hartree+ChemP;
  iemin=iemin/eV2Hartree+ChemP;
  iemax=iemax/eV2Hartree+ChemP;

 /* the biggest number of bands within the energy windown among all the k points */ 
  BANDNUM = 0; 
  MkNUM=0;
 
  if (myid==Host_ID){
    printf("Selected bands within the Outer and Inner Windows at each k point:\n");
    printf("spin| kpt | Outer Window (Nk) |  Inner Window (Mk)  |\n"); 
  }

  for(k=0; k<kpt_num; k++){

    for(spin=0; spin<spinsize; spin++){
      if (myid==Host_ID){
        printf(" %1d  |%5d|",spin,k+1);
      }
      Nk[spin][k][0]      = 0;
      Nk[spin][k][1]      = 0;
      innerNk[spin][k][0] = 0;
      innerNk[spin][k][1] = 0;

      for(i=1; i<fsize3-1; i++){

	if (EigenValall[k][spin][i]<oemin){
	  Nk[spin][k][0] = i;
	}
	if (EigenValall[k][spin][i]<oemax){
	  Nk[spin][k][1] = i;
	}
	if (EigenValall[k][spin][i]<iemin){
	  innerNk[spin][k][0] = i;
	}
	if (EigenValall[k][spin][i]<iemax){
	  innerNk[spin][k][1] = i;
	} 
      } 

      if( BANDNUM<(Nk[spin][k][1]-Nk[spin][k][0]) ){
	BANDNUM = Nk[spin][k][1] - Nk[spin][k][0];
      }

      if( MkNUM<(innerNk[spin][k][1]-innerNk[spin][k][0]) ){
	MkNUM = innerNk[spin][k][1] - innerNk[spin][k][0];
      }
      if (myid==Host_ID){
        printf("  (%3d ,%3d]  %3d  |  (%3d ,%3d]   %3d   |\n",
             Nk[spin][k][0],Nk[spin][k][1],
             Nk[spin][k][1]-Nk[spin][k][0],
             innerNk[spin][k][0],
             innerNk[spin][k][1],
             innerNk[spin][k][1]-innerNk[spin][k][0]);
        fflush(0);
      }
      if (WANNUM>(Nk[spin][k][1]-Nk[spin][k][0])){  
        if (myid==Host_ID){
  	  printf("**************************ERROR**************************\n");
  	  printf("* Bands number within OUTER window [%10.5f, %10.5f]     *\n",oemin,oemax);
          printf("* is less than Wannier Function number %i.              *\n",WANNUM);
	  printf("* Please check them and try again.                      *\n");
	  printf("**************************ERROR**************************\n");      
        }
        MPI_Finalize();
	exit(0); 
      }

      if (WANNUM<(innerNk[spin][k][1]-innerNk[spin][k][0])){
        if (myid==Host_ID){
   	   printf("**************************ERROR**************************\n");
    	   printf("* Bands number within INNER window [%10.5f, %10.5f]     *\n",iemin,iemax);
	   printf("* is larger than wannier function number %i.            *\n",WANNUM);
	   printf("* Please check them and try again.\n");
	   printf("**************************ERROR**************************\n");
        }
        MPI_Finalize();
	exit(0);
      }
    } /* spin */
  }/* kpt */

  if(have_frozen==1 &&MkNUM==0){
    have_frozen=0;
  }

  if (myid==Host_ID)  printf("Totally, %i bands are included in calculation.\n",BANDNUM);

  eigen = (double***)malloc(sizeof(double**)*spinsize);

  for (spin=0;spin<spinsize;spin++){

    eigen[spin] = (double**)malloc(sizeof(double*)*kpt_num);

    for (k=0; k<kpt_num; k++){
      eigen[spin][k] = (double*)malloc(sizeof(double)*BANDNUM);
      for (j=0; j<BANDNUM; j++){ 
	eigen[spin][k][j] = 0.0;
      }
    }
  }

  for (spin=0;spin<spinsize;spin++){
    for (k=0;k<kpt_num;k++){
      for (i=1; i<BANDNUM+1; i++){
        eigen[spin][k][i-1] = EigenValall[k][spin][i+Nk[spin][k][0]];
      }
    }
  }

  /***************************************************
              save EigenValall and Wkall
              "*.eig" is added for wannier90 by F. Ishii
  ***************************************************/
  
  if (lreadMmnk==0 && myid==Host_ID){

    sprintf(fname,"%s%s.eigen",filepath,filename);
    sprintf(fname2,"%s%s.eig",filepath,filename);

    if((fptmp=fopen(fname,"wt"))==NULL){
      printf("Error in opening %s for writing eigenvalues.\n",fname);
      exit(0);
    }
    if((fptmp2=fopen(fname2,"wt"))==NULL){
      printf("Error in opening %s for writing eigenvalues.\n",fname2);
      exit(0);
    }

    fprintf(fptmp,"Fermi level %lf\n",ChemP);
    //fprintf(fptmp2,"Fermi level %lf\n",ChemP*eV2Hartree);
    fprintf(fptmp,"Number of bands %i\n",BANDNUM);
    //fprintf(fptmp2,"Number of bands %i\n",BANDNUM);

    for(spin=0; spin<spinsize; spin++){
      for(k=0; k<kpt_num; k++){

        for(i=1; i<BANDNUM+1; i++){

          if ( (i+Nk[spin][k][0])<fsize3 && (i+Nk[spin][k][0])<=Nk[spin][k][1]){
            fprintf(fptmp,"%5d%5d%18.12f\n",i,k+1,EigenValall[k][spin][i+Nk[spin][k][0]]);
            fprintf(fptmp2,"%5d%5d%18.12f\n",i,k+1,(EigenValall[k][spin][i+Nk[spin][k][0]]-ChemP)*eV2Hartree);
	  }
          else {
            double dtmp;
            dtmp = 10000.0;  
            fprintf(fptmp,"%5d%5d  %18.10f\n",i,k+1,dtmp);
            fprintf(fptmp2,"%5d%5d  %18.10f\n",i,k+1,dtmp);
          }
        }
      }
    }

    for(spin=0; spin<spinsize; spin++){
      for(k=0; k<kpt_num; k++){

        fprintf(fptmp, "WF kpt %i (%10.8f,%10.8f,%10.8f)\n",
                        k+1,kg[k][0],kg[k][1],kg[k][2]);
        //fprintf(fptmp2, "WF kpt %i (%10.8f,%10.8f,%10.8f)\n",
        //                k+1,kg[k][0],kg[k][1],kg[k][2]);

        for(i=1; i<BANDNUM+1; i++){
          for(j=1; j<fsize3; j++){

            if ( (i+Nk[spin][k][0])<fsize3 && (i+Nk[spin][k][0])<=Nk[spin][k][1] ){
 
              fprintf(fptmp,"%i %i %14.10f %14.10f\n",
                             i,j,Wkall[k][spin][i+Nk[spin][k][0]][j].r,
                                 Wkall[k][spin][i+Nk[spin][k][0]][j].i); 
          //    fprintf(fptmp2,"%i %i %14.10f %14.10f\n",
           //                  i,j,Wkall[k][spin][i+Nk[spin][k][0]][j].r,
           //                      Wkall[k][spin][i+Nk[spin][k][0]][j].i); 
	    }
            else {

              double dtmp1,dtmp2;
              dtmp1 = 0.0;
              dtmp2 = 0.0;

              fprintf(fptmp,"%i %i %14.10f %14.10f\n",i,j,dtmp1,dtmp2);
            //  fprintf(fptmp2,"%i %i %14.10f %14.10f\n",i,j,dtmp1,dtmp2);
            }
	  }
        }
      }
    } 

    fclose(fptmp);
    fclose(fptmp2);
  }

/* fpwn90 is added for wannier90 input *.win file by  F. Ishii*/
    if (myid==Host_ID){
      //sprintf(fname3,"%s%s.kptwin",filepath,filename);
      sprintf(fname4,"%s%s.win",filepath,filename);
      //if((fpkpt2=fopen(fname3,"wt"))==NULL){
      //  printf("Error in opening %s for writing k point for wannier90.\n",fname3);
      //  exit(0);
      //}
      if((fpwn90=fopen(fname4,"wt"))==NULL){
        printf("Error in opening %s for input file for wannier90.\n",fname4);
        exit(0);
      }
     // i=0;
      fprintf(fpwn90, "num_bands %d\n", BANDNUM);
      fprintf(fpwn90, "num_wann %d\n", Wannier_Func_Num);
      fprintf(fpwn90, "num_iter 15000\n");
      fprintf(fpwn90, "begin unit_cell_cart\n");
      fprintf(fpwn90, "Ang\n");
      for (i=1; i<=3; i++){
        fprintf(fpwn90,"%lf %lf %lf\n",tv[i][1]*BohrR_Wannier,tv[i][2]*BohrR_Wannier,tv[i][3]*BohrR_Wannier);
     }
      fprintf(fpwn90, "end unit_cell_cart\n");
      fprintf(fpwn90, "begin atoms_frac\n");
      
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

        /* The zero is taken as the origin of the unit cell. */

	Cxyz[1] = Gxyz[Gc_AN][1];
	Cxyz[2] = Gxyz[Gc_AN][2];
	Cxyz[3] = Gxyz[Gc_AN][3];

	Cell_Gxyz[Gc_AN][1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
	Cell_Gxyz[Gc_AN][2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
	Cell_Gxyz[Gc_AN][3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

        /* The fractional coordinates are kept within 0 to 1. */

        for (i=1; i<=3; i++){

          itmp = (int)Cell_Gxyz[Gc_AN][i]; 

	  if (1.0<Cell_Gxyz[Gc_AN][i]){
	    Cell_Gxyz[Gc_AN][i] = fabs(Cell_Gxyz[Gc_AN][i] - (double)itmp);
	  }
	  else if (Cell_Gxyz[Gc_AN][i]<-1.0e-13){
	    Cell_Gxyz[Gc_AN][i] = fabs(Cell_Gxyz[Gc_AN][i] + (double)(abs(itmp)+1));
	  }
	}

        k = WhatSpecies[Gc_AN];

//        fprintf(fp,"%4s   %18.14f %18.14f %18.14f\n",
        fprintf(fpwn90,"%4s %18.14f %18.14f %18.14f\n",
                SpeName[k],
               Cell_Gxyz[Gc_AN][1],
               Cell_Gxyz[Gc_AN][2],
               Cell_Gxyz[Gc_AN][3]);
      }



      fprintf(fpwn90, "end atoms_frac\n");
      fprintf(fpwn90, "dis_win_min =%18.14f\n",Wannier_Outer_Window_Bottom); 
      fprintf(fpwn90, "dis_win_max =%18.14f\n",Wannier_Outer_Window_Top); 
      fprintf(fpwn90, "dis_froz_min =%18.14f\n",Wannier_Inner_Window_Bottom); 
      fprintf(fpwn90, "dis_froz_max =%18.14f\n",Wannier_Inner_Window_Top); 
      fprintf(fpwn90, "mp_grid %d %d %d\n",knum_i, knum_j, knum_k);
      fprintf(fpwn90, "begin_kpoints\n");
      for(k=0;k<kpt_num;k++){
        //fprintf(fpkpt2, "%12.8f%12.8f%12.8f\n",kg[k][0],kg[k][1],kg[k][2]);
        fprintf(fpwn90, "%12.8f%12.8f%12.8f\n",kg[k][0],kg[k][1],kg[k][2]);
      }
      fprintf(fpwn90, "end_kpoints\n");
      fprintf(fpwn90, "!spinors T\n");
      fprintf(fpwn90, "bands_plot T\n");
      fprintf(fpwn90, "begin kpoint_path\n");
       for (i=1;i<=Band_Nkpath;i++) {
         fprintf(fpwn90,"%s %lf %lf %lf %s %lf %lf %lf\n",
                 Band_kname[i][1],
                 Band_kpath[i][1][1], Band_kpath[i][1][2], Band_kpath[i][1][3],
                 Band_kname[i][2],
                 Band_kpath[i][2][1], Band_kpath[i][2][2], Band_kpath[i][2][3]);
       }
      fprintf(fpwn90, "end kpoint_path\n");
      fprintf(fpwn90, "bands_plot_dim 2\n");
      fprintf(fpwn90, "bands_num_points 100\n");
      fprintf(fpwn90, "!fermi_energy_min =-3.0\n");
      fprintf(fpwn90, "!fermi_energy_max =3.0\n");
      fprintf(fpwn90, "!fermi_energy_step = 0.1\n");
      fprintf(fpwn90, "fermi_energy = 0.0\n");
      fprintf(fpwn90, "berry T\n");
      fprintf(fpwn90, "!berry_task ahc\n");
      fprintf(fpwn90, "berry_task kubo\n");
      fprintf(fpwn90, "kubo_freq_max = 7.0\n");
      fprintf(fpwn90, "kmesh 100\n");
      fprintf(fpwn90, "berry_curv_adpt_kmesh 3\n");

    }


  /* Now calculate the overlap matrix Mmn(k,b) for each k point */
  /* Mmnkb:
     overlap matrix between one-particle wave functions
     calculated at two k-points, k and k+b, Mmn(k,b). 
     Dimension analysis:   
     1. k point index;
     2. b vector index. 
     3. Spin-polarization or non-spin-polarization. Size is spinsize;
     4. m,n band index. Size is hamtonian matrix size, fsize3 by fsize3;
  */

  /* This is the overlap matrix before initial guess */

  Mmnkb_zero = (dcomplex*****)malloc(sizeof(dcomplex****)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Mmnkb_zero[k] = (dcomplex****)malloc(sizeof(dcomplex***)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      Mmnkb_zero[k][bindx] = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
      for (spin=0; spin<spinsize; spin++){
        Mmnkb_zero[k][bindx][spin] = (dcomplex**)malloc(sizeof(dcomplex*)*(BANDNUM+1));
        for (i=0; i<BANDNUM+1; i++){
          Mmnkb_zero[k][bindx][spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*(BANDNUM+1));
	  for (j=0; j<BANDNUM+1; j++){ 
	    Mmnkb_zero[k][bindx][spin][i][j].r=0.0; 
	    Mmnkb_zero[k][bindx][spin][i][j].i=0.0;
	  }
	} /* i, j band num */
      }/* spin */
    }/* b vector */
  }/* k point*/

  /**********************************************************
    If Mmnkb file doesn't exist, calculate it and open 
    a file for writing Mmn_zero(k,b) 
  **********************************************************/

  if (lreadMmnk==0){

    double bk[4],k1[4],k2[4],Sop[2];

    if (myid==Host_ID){
      printf("\nComing to the overlap matrix calculating......\n");fflush(0);
    }

    /*******************************
         calculation of Mmnkb
    ********************************/

    /* a parallelized loop for spinsize*kpt_num*tot_bvector*BANDNUM*BANDNUM */
    /*                           spin    k       b          m      n      */

    odloop_num = spinsize*kpt_num*tot_bvector*BANDNUM*BANDNUM;

#pragma omp parallel shared(odloop_num,myid,numprocs,kpt_num,tot_bvector,BANDNUM,kplusb,Nk,fsize3,kg,frac_bv,SpinP_switch,MP,Wkall,fsize,Mmnkb_zero) private(odloop,spin,itmp1,itmp2,itmp3,k,bindx,m,n,kk,m1,n1,k1,k2,bk,Sop,norm,OMPID,Nthrds,Nprocs)
    {

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for( odloop=myid+numprocs*OMPID; odloop<odloop_num; odloop+=numprocs*Nthrds ){

      /* get spin, k, bindx, m, and n */

      spin = odloop/(kpt_num*tot_bvector*BANDNUM*BANDNUM);
      itmp1 = odloop - spin*(kpt_num*tot_bvector*BANDNUM*BANDNUM);
      k = itmp1/(tot_bvector*BANDNUM*BANDNUM);
      itmp2 = itmp1 - k*(tot_bvector*BANDNUM*BANDNUM);
      bindx = itmp2/(BANDNUM*BANDNUM);
      itmp3 = itmp2 - bindx*(BANDNUM*BANDNUM);
      m = itmp3/BANDNUM;
      n = itmp3 - m*BANDNUM;

      /* kk is the index for k+b */

      kk = kplusb[k][bindx];

      /* m and n must be added to the offset. */

      m1 = m + 1 + Nk[spin][k][0];      
      n1 = n + 1 + Nk[spin][kk][0];      

      if (m1<(fsize3-1) && n1<(fsize3-1)){

	/* get k-value */

	k1[1] = kg[k][0];
	k1[2] = kg[k][1];
	k1[3] = kg[k][2];

	/* get k+b */

	k2[1] = k1[1] + frac_bv[bindx][0];  /* Equivalent k point of k+b */
	k2[2] = k1[2] + frac_bv[bindx][1];
	k2[3] = k1[3] + frac_bv[bindx][2];

	/* set the value of b-vector */

	bk[1] = frac_bv[bindx][0]; 
	bk[2] = frac_bv[bindx][1];
	bk[3] = frac_bv[bindx][2];

	/* calc Mmnkb */

	/*
        if (k==99 && bindx==9 && m==1 && n==0){

          printf("EEE kk=%2d spin=%2d m1=%2d n1=%2d\n",kk,spin,m1,n1);
          
          for (i=1; i<=fsize2; i++){
            printf("GGG i=%3d W1.r=%15.12f W1.i=%15.12f W2.r=%15.12f W2.i=%15.12f\n", 
		   i,
                   Wkall[k][spin][m1][i].r, Wkall[k][spin][m1][i].i,
                   Wkall[kk][spin][n1][i].r,Wkall[kk][spin][n1][i].i
                   );
          }
	}
	*/


	Calc_Mmnkb(k, kk, k1, k2,  bk, bindx, SpinP_switch, MP, Sop, 
		   Wkall[k][spin][m1], Wkall[kk][spin][n1], fsize, m,n);


	/*
	printf("k=%2d bindx=%2d m=%2d n=%2d O.r=%15.12f O.i=%15.12f\n",
	k,bindx,m,n,Sop[0],Sop[1]);

        if (k==0 && bindx==0 && m==0 && n==0){

          MPI_Finalize();
          exit(0); 

	}
	*/

	Mmnkb_zero[k][bindx][spin][m+1][n+1].r = Sop[0];
	Mmnkb_zero[k][bindx][spin][m+1][n+1].i = Sop[1];

        norm = sqrt( fabs(Sop[0]*Sop[0] + Sop[1]*Sop[1]) );

	if (norm>1.0){
	  printf("**********************WARNNING**********************\n");
	  printf("Attention! |Mmnkb=%10.5f|>1.0 at k=%i,b=%i,i=%i,j=%i\n",norm,k,bindx,m+1,n+1);
	  printf("**********************WARNNING**********************\n");
	}


	/*
        if (k==99 && bindx==9 && m==1 && n==0){
          MPI_Finalize();
          exit(0);
	}
	*/

      }

      /* if (!(m1<fsize3 && n1<fsize3))  */

      else {
        Mmnkb_zero[k][bindx][spin][m+1][n+1].r = -99999.0;
        Mmnkb_zero[k][bindx][spin][m+1][n+1].i = -99999.0;
      }

    } /* odloop */ 
    } /* #pragma omp parallel */

    /*******************************
       MPI communication of Mmnkb
    ********************************/

    for( odloop=0; odloop<odloop_num; odloop++ ){

      /* find ID */

      ID = odloop % numprocs;

      /* get spin, k, bindx, m, and n */

      spin = odloop/(kpt_num*tot_bvector*BANDNUM*BANDNUM);
      itmp1 = odloop - spin*(kpt_num*tot_bvector*BANDNUM*BANDNUM);
      k = itmp1/(tot_bvector*BANDNUM*BANDNUM);
      itmp2 = itmp1 - k*(tot_bvector*BANDNUM*BANDNUM);
      bindx = itmp2/(BANDNUM*BANDNUM);
      itmp3 = itmp2 - bindx*(BANDNUM*BANDNUM);
      m = itmp3/BANDNUM;
      n = itmp3 - m*BANDNUM;

      /* bcast of Mmnkb_zero */

      MPI_Bcast( &Mmnkb_zero[k][bindx][spin][m+1][n+1].r, 1, 
                 MPI_DOUBLE, ID, mpi_comm_level1);
      MPI_Bcast( &Mmnkb_zero[k][bindx][spin][m+1][n+1].i, 1, 
                 MPI_DOUBLE, ID, mpi_comm_level1);
    }

    /*************************************
      save the overlap matrix Mmnkb_zero
    *************************************/

    if ( Wannier_Output_Overlap_Matrix && myid==Host_ID ){

      sprintf(fname,"%s%s.mmn",filepath,filename);
      if((fp=fopen(fname,"wt"))==NULL){

	printf("******************************************************************\n");
	printf("* Error in opening file %s for writing Mmn(k,b).\n",fname);
	printf("******************************************************************\n");
	exit(0);     

      }

      else{
	printf(" ... ... Writting Mmn_zero(k,b) matrix into file.\n\n");
	fprintf(fp,"Mmn_zero(k,b). band_num, kpt_num, bvector num, spinsize\n");
	fprintf(fp,"%13d%13d%13d%13d\n",BANDNUM,kpt_num,tot_bvector,spinsize);    
      }

      for(spin=0;spin<spinsize; spin++){
	for(k=0;k<kpt_num;k++){

          /* for each b vectors within the shells around k point */

	  for(bindx=0;bindx<tot_bvector;bindx++){ 

	    kk=kplusb[k][bindx];
	    ktp[1]=kg[k][0]+frac_bv[bindx][0];  /* Equivalent k point of k+b */
	    ktp[2]=kg[k][1]+frac_bv[bindx][1];
	    ktp[3]=kg[k][2]+frac_bv[bindx][2];

	    for(i=1;i<=3;i++){
	      b[i]=ktp[i];
	      if(ktp[i]>=1.0){
		b[i]=ktp[i]-1.0;
	      }
	      if(ktp[i]<0.0){
		b[i]=ktp[i]+1.0;
	      }
	    }

	    fprintf(fp,"%5d%5d%5d%5d%5d\n",
                     k+1,kk+1,
                     (int)(ktp[1]-b[1]),
                     (int)(ktp[2]-b[2]),
                     (int)(ktp[3]-b[3])); 

	    for(i=1;i<BANDNUM+1;i++){
	      for(j=1;j<BANDNUM+1;j++){

		fprintf(fp,"%18.12f%18.12f\n",
                        Mmnkb_zero[k][bindx][spin][j][i].r,
                        Mmnkb_zero[k][bindx][spin][j][i].i);

	      } /* j */
	    }/* i */
	  } /* bv indx */
	}/* k point */ 
      } /* spin */
      fclose(fp); /* End writting case.mmn matrix */

    } /* if ( Wannier_Output_Overlap_Matrix && myid==Host_ID ) */

  } /* if (lreadMmnk==0) */  

  /**********************************************************
        If Mmnkb exists, read a file for Mmn_zero(k,b). 
  **********************************************************/

  if (lreadMmnk){ 

    sprintf(fname,"%s%s.mmn",filepath,filename);
    dumy=(char*)malloc(sizeof(char)*BUFFSIZE);

    if((fp=fopen(fname,"rt"))==NULL){
      if (myid==Host_ID){
        printf("*****************************************************************\n");
        printf(" Error in opening file %s for reading Mmn(k,b). Please check it!\n",fname);
        printf("*****************************************************************\n");
      }
      MPI_Finalize();
      exit(0);
    }

    else{

      if (myid==Host_ID){
        printf("\n Reading Mmn(k,b) matrix from file.\n");
      }

      fgets(dumy, BUFFSIZE, fp);
      fscanf(fp,"%d%d%d%d",&fband_num,&k,&j,&spin);

      if(spin!=spinsize ||fband_num<BANDNUM || k!=kpt_num || j!=tot_bvector){

        if (myid==Host_ID){
  	  printf("******************************* ERROR ********************************************\n");
	}

        if(spin!=spinsize && myid==Host_ID){
          printf("* Spin size in Mmn(k,b) file is not consistent with present calculation.*\n* Please Check them.\n");
        }

        if(fband_num<BANDNUM && myid==Host_ID){
          printf("* Outer window size in Mmn(k,b) file is smaller than that in present calculation.*\n* Please Check them.\n");
        }

        if(k!=kpt_num && myid==Host_ID){
          printf("* k point number in Mmn(k,b) file is not the same as that in present calculation.*\n* Please Check them.\n");
        }

        if(j!=tot_bvector && myid==Host_ID){
          printf("* b vectors in Mmn(k,b) file are not the same as those in present calculation.*\n* Please Check them.\n");
        }

        if (myid==Host_ID){
	  printf("******************************* ERROR ***************************************\n");
	}
 
        MPI_Finalize();
	exit(0);
      }
    } 

    fgets(dumy, BUFFSIZE, fp);   

    for(spin=0;spin<spinsize;spin++){
      for(k=0;k<kpt_num;k++){
	for(bindx=0;bindx<tot_bvector;bindx++){
	  fgets(dumy, BUFFSIZE, fp);

	  /* ignore that in old outer window */

          kk = kplusb[k][bindx];

	  for(i=1;i<fband_num+1;i++){
	    for(j=1;j<fband_num+1;j++){

              if(   i>Nk[spin][kk][0] && i<BANDNUM+1+Nk[spin][kk][0] 
                 && j>Nk[spin][k][0]  && j<BANDNUM+1+Nk[spin][k][0]){

                fscanf(fp,"%lf%lf",
                         &Mmnkb_zero[k][bindx][spin][j-Nk[spin][k][0]][i-Nk[spin][kk][0]].r,
                         &Mmnkb_zero[k][bindx][spin][j-Nk[spin][k][0]][i-Nk[spin][kk][0]].i);
              }

              else{
                fscanf(fp,"%lf%lf",&tmpr,&tmpi);
              }
            }
	  }

	  fgets(dumy, BUFFSIZE, fp);   

	} /* bindx */
      } /* k */
    } /* spin */

    fclose(fp);
    free(dumy);

  }/* Read Mmnkb */ 

  /* Release wave function and eigenvalue arrays, which will not be used anymore */

  for(k=0; k<kpt_num; k++){
    for (spin=0; spin<spinsize; spin++){
      free(EigenValall[k][spin] );
    }
    free(EigenValall[k]);
  }
  free(EigenValall);


  /*
  spin = 0;
  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){ 
      for(i=1;i<BANDNUM+1;i++){
	for(j=1;j<BANDNUM+1;j++){
          printf("ABC k=%2d bindx=%2d i=%2d j=%2d M.r=%15.12f M.i=%15.12f\n",
		 k,bindx,i,j,
                 Mmnkb_zero[k][bindx][spin][i][j].r,
                 Mmnkb_zero[k][bindx][spin][i][j].i);fflush(0);
	}
      }
    }
  }
  */


  /* start Wannier Function part */

  if (myid==Host_ID){
    printf("Coming to the initial guess of WF......\n");fflush(0);
  }

  /* AS^(-1/2), Uk, matrix using method in PRB65,035109 (2001) 
     This Uk matrix is also used for disentangling the attached bands. */
  Uk=(dcomplex****)malloc(sizeof(dcomplex***)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    Uk[spin]=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
    for(k=0;k<kpt_num;k++){
      Uk[spin][k]=(dcomplex**)malloc(sizeof(dcomplex*)*BANDNUM);
      for(i=0;i<BANDNUM;i++){
	Uk[spin][k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
	for(j=0;j<WANNUM;j++){
	  Uk[spin][k][i][j].r=0.0;Uk[spin][k][i][j].i=0.0;
	  if(i==j){
	    Uk[spin][k][i][j].r=1.0;Uk[spin][k][i][j].i=0.0;
	  }
	}/* wannier */
      }/* Windown */
    } /*kpt */
  } /* spin */

  /* for updating the Mmnkb matrix during the optimization of omega_tilde part*/
  Utilde=(dcomplex****)malloc(sizeof(dcomplex***)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    Utilde[spin]=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
    for(k=0;k<kpt_num;k++){
      Utilde[spin][k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
      for(i=0;i<WANNUM;i++){
	Utilde[spin][k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
	for(j=0;j<WANNUM;j++){
	  Utilde[spin][k][i][j].r=0.0;Utilde[spin][k][i][j].i=0.0;
	  if(j==i){
	    Utilde[spin][k][i][j].r=1.0;Utilde[spin][k][i][j].i=0.0;
	  }
	}/* wannier */
      }/* Windown */
    } /*kpt */
  } /* spin */

  tmpM=(dcomplex**)malloc(sizeof(dcomplex*)*BANDNUM);
  for(i=0;i<BANDNUM;i++){
    tmpM[i]=(dcomplex*)malloc(sizeof(dcomplex)*BANDNUM);
  }/* j */

  for(i=0;i<BANDNUM;i++){
    for(j=0;j<BANDNUM;j++){  
      tmpM[i][j].r=0.0;
      tmpM[i][j].i=0.0;
    }
  }

  /* overlap matrix Mmn(k,b) for each k point after disentangling and initial guess*/
  Mmnkb_dis = (dcomplex*****)malloc(sizeof(dcomplex****)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Mmnkb_dis[k] = (dcomplex****)malloc(sizeof(dcomplex***)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      Mmnkb_dis[k][bindx] = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
      for(spin=0; spin<spinsize; spin++){
        Mmnkb_dis[k][bindx][spin] = (dcomplex**)malloc(sizeof(dcomplex*)*(WANNUM+1));
        for (i=0; i<WANNUM+1; i++){
          Mmnkb_dis[k][bindx][spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*(WANNUM+1));
          for (j=0; j<WANNUM+1; j++){
            Mmnkb_dis[k][bindx][spin][i][j].r=0.0;
            Mmnkb_dis[k][bindx][spin][i][j].i=0.0;
          }
        } /* i, j band num */
      }/* spin */
    }/* b vector */
  }/* k point*/

  /* Amatrix Amn(k) 
     Projecting random phase Bloch wave functions on to the trial Wannier Functions
  */    
  Amnk=(dcomplex****)malloc(sizeof(dcomplex***)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    Amnk[spin]=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
    for(k=0;k<kpt_num;k++){
      Amnk[spin][k]=(dcomplex**)malloc(sizeof(dcomplex*)*BANDNUM);
      for(i=0;i<BANDNUM;i++){
        Amnk[spin][k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
        for(j=0;j<WANNUM;j++){
          Amnk[spin][k][i][j].r=0.0;Amnk[spin][k][i][j].i=0.0;
          if(j==i){
            Amnk[spin][k][i][j].r=1.0;Amnk[spin][k][i][j].i=0.0;
          }
        }/* wannier */
      }/* Windown */
    } /*kpt */
  } /* spin */
 
  if (lprojection){

    if(Wannier_Readin_Projection_Matrix){

      if (myid==Host_ID){
        printf("Read projection matrix\n");fflush(0);
      }

      dumy=(char*)malloc(sizeof(char)*BUFFSIZE);
      sprintf(fname,"%s%s.amn",filepath,filename);

      if((fp=fopen(fname,"rt"))==NULL){

        if (myid==Host_ID){
	  printf("******************************************************************\n");
	  printf("* Error in opening file for reading projection matrix. Please check it!*\n");
	  printf("******************************************************************\n");
	}

        MPI_Finalize();
	exit(0);
      }

      else{

        if (myid==Host_ID){
  	  printf("\n Reading initial guess projection matrix from file.\n");
	}

	fgets(dumy, BUFFSIZE, fp);
	fgets(dumy, BUFFSIZE, fp);

	for(spin=0;spin<spinsize;spin++){
	  for(k=0;k<kpt_num;k++){
	    for(i=0;i<WANNUM;i++){
	      for(j=0;j<BANDNUM;j++){/* band index in enegy window */
		fscanf(fp,"%d%d%d%lf%lf",&ki,&kj,&ki,&Amnk[spin][k][j][i].r,&Amnk[spin][k][j][i].i);
	      } /* band */        
	    } /* wf */
	  }
	}
      } 

      free(dumy);
    }

    else{
      Projection_Amatrix(Amnk, kg, spinsize, fsize, SpinP_switch, kpt_num, BANDNUM,  WANNUM, Wkall, MP, Nk);
    }
  }

  /* This Uk is that for disentangling attached bands with initial guess */

  Getting_Uk(Amnk, spinsize, kpt_num, BANDNUM, WANNUM, Uk, Nk); 

  /* If this is a mixed bands case, we need to disentangle the mixed band and 
     iteratively find out the proper Uk which minimize gauge invariant part of spread function. */

  if(BANDNUM>WANNUM){

    Disentangling_Bands(Uk, Mmnkb_zero, spinsize, kpt_num, 
                        BANDNUM, WANNUM, Nk, innerNk, kg, 
                        frac_bv, wb, kplusb, tot_bvector, eigen);

    /* after disentangling, we get Uk which will disentangle the original overlap matrix. 
       M_dis(N x N)= Uk^dagger(N x Nk) * M_zero (Nk x Nk)* Uk(k+b) (Nk x N)*/  

    Initial_Guess_Mmnkb(Uk, spinsize, kpt_num, BANDNUM, WANNUM, Mmnkb_dis, 
                        Mmnkb_zero, kg, frac_bv, tot_bvector, kplusb, Nk);

    /* Still we need to update our initial guess of U matrix since now we are 
       in the optimized subspace.
       Amnk_tilde (now is N x N )= Uk^dagger( N x Nk) * Amnk (Nk x N) */

    for(spin=0;spin<spinsize;spin++){
      for(k=0;k<kpt_num;k++){
        for(i=0;i<WANNUM;i++){
	  for(j=0;j<WANNUM;j++){
	    tmpr=0.0;tmpi=0.0;
	    for(l=0;l<Nk[spin][k][1]-Nk[spin][k][0];l++){

	      tmpr=tmpr + Uk[spin][k][l][i].r*Amnk[spin][k][l][j].r 
                        + Uk[spin][k][l][i].i*Amnk[spin][k][l][j].i;
	      tmpi=tmpi+Uk[spin][k][l][i].r*Amnk[spin][k][l][j].i-Uk[spin][k][l][i].i*Amnk[spin][k][l][j].r;
	    }
	    tmpM[i][j].r=tmpr;
	    tmpM[i][j].i=tmpi;
	  }
        }
        for(i=0;i<WANNUM;i++){
	  for(j=0;j<WANNUM;j++){
	    Amnk[spin][k][i][j].r=tmpM[i][j].r;
	    Amnk[spin][k][i][j].i=tmpM[i][j].i;
	  }
        }
      }/* kpt */
    }/* spin */
    /* Then we can find the initial guess for Utilde matrix which is usded for optimizing the omega_Tilde part */
    Getting_Utilde(Amnk, spinsize, kpt_num, WANNUM, WANNUM, Utilde); 
    /* and update Mmnk matrix to that with intial guess M_opt= Utilde^dagger * M_zero * Utilde(k+b) */

    Initial_Guess_Mmnkb(Utilde, spinsize, kpt_num, WANNUM, WANNUM, Mmnkb_zero, Mmnkb_dis, kg, frac_bv, tot_bvector, kplusb, Nk);
    /* Mmnkb_zero now is the disentangled WANNUMxWANNUM overlap matrix */ 

  }/* Disentangle */ 

  else{ /* NOT disentangle */ 

    /* update Mmnk matrix to that with intial guess M_opt= Utilde^dagger * M_zero * Utilde(k+b) */
    Initial_Guess_Mmnkb(Uk, spinsize, kpt_num, BANDNUM, WANNUM, Mmnkb_dis, Mmnkb_zero, kg, frac_bv, tot_bvector, kplusb, Nk); 
    /* Mmnkb_zero now is the disentangled WANNUMxWANNUM overlap matrix */ 
    for(k=0;k<kpt_num;k++){
      for(bindx=0;bindx<tot_bvector;bindx++){
	for (spin=0; spin<spinsize; spin++){
	  for (i=0; i<WANNUM+1; i++){
	    for (j=0; j<WANNUM+1; j++){
	      Mmnkb_zero[k][bindx][spin][i][j].r=Mmnkb_dis[k][bindx][spin][i][j].r;
	      Mmnkb_zero[k][bindx][spin][i][j].i=Mmnkb_dis[k][bindx][spin][i][j].i;
	    }
	  } /* i, j band num */
	}/* spin */
      }/* b vector */
    }/* k point*/

    /* for non-disentangle case, this Utilde should be identity matrix for latter interpolation */
    for(spin=0;spin<spinsize;spin++){
      for(k=0;k<kpt_num;k++){
	for(i=0;i<WANNUM;i++){
	  for(j=0;j<WANNUM;j++){
	    Utilde[spin][k][i][j].r=0.0;
	    Utilde[spin][k][i][j].i=0.0;
	    if(j==i){
	      Utilde[spin][k][i][j].r=1.0;
              Utilde[spin][k][i][j].i=0.0;
	    }      
	  }/* wannier */
	}/* Windown */
      } /*kpt */
    } /* spin */
  }

  /* Multiply Uk and Utilde and store in Uk. Therefore, Uk now is Udis * Uinitial */
  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      for(i=0;i<Nk[spin][k][1]-Nk[spin][k][0];i++){
	for(j=0;j<WANNUM;j++){
	  tmpr=0.0; tmpi=0.0;
	  for(l=0;l<WANNUM;l++){
	    tmpr=tmpr+Uk[spin][k][i][l].r*Utilde[spin][k][l][j].r-Uk[spin][k][i][l].i*Utilde[spin][k][l][j].i;
	    tmpi=tmpi+Uk[spin][k][i][l].r*Utilde[spin][k][l][j].i+Uk[spin][k][i][l].i*Utilde[spin][k][l][j].r;
	  }
	  Amnk[spin][k][i][j].r=tmpr;
	  Amnk[spin][k][i][j].i=tmpi;
	}
      }
      for(i=0;i<Nk[spin][k][1]-Nk[spin][k][0];i++){
	for(j=0;j<WANNUM;j++){
	  Uk[spin][k][i][j].r=Amnk[spin][k][i][j].r;
	  Uk[spin][k][i][j].i=Amnk[spin][k][i][j].i;
	}
      }
    }
  }

  /* release those not used any more */
  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      for(i=0;i<BANDNUM;i++){
        free(Amnk[spin][k][i]);
      }/* Windown */
      free(Amnk[spin][k]);
    } /*kpt */
    free(Amnk[spin]);
  } /* spin */
  free(Amnk);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for (spin=0; spin<spinsize; spin++){
        for (i=0; i<WANNUM+1; i++){
          free(Mmnkb_dis[k][bindx][spin][i]); 
        } /* i, j band num */
        free(Mmnkb_dis[k][bindx][spin]);
      }/* spin */
      free(Mmnkb_dis[k][bindx]); 
    }/* b vector */
    free(Mmnkb_dis[k]);    
  }/* k point*/
  free(Mmnkb_dis);

  /* From here seperate spin */
  sheet=(double***)malloc(sizeof(double**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    sheet[k]=(double**)malloc(sizeof(double*)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      sheet[k][bindx] = (double*)malloc(sizeof(double)*WANNUM);
      for(i=0;i<WANNUM;i++){
        sheet[k][bindx][i] = 0.0;
      } 
    } 
  }
  csheet=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    csheet[k]=(dcomplex**)malloc(sizeof(dcomplex*)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      csheet[k][bindx] = (dcomplex*)malloc(sizeof(dcomplex)*WANNUM);     
      for(i=0;i<WANNUM;i++){
        csheet[k][bindx][i].r = 1.0;
        csheet[k][bindx][i].i = 0.0;
      } 
    } 
  }
  rguide=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    rguide[i]=(double*)malloc(sizeof(double)*WANNUM);
    for(j=0;j<WANNUM;j++){
      rguide[i][j]=Wannier_Guide[i][j];
    }
  }

  Mmnkb = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);   
  for(k=0;k<kpt_num;k++){
    Mmnkb[k] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector); 
    for(bindx=0;bindx<tot_bvector;bindx++){
      Mmnkb[k][bindx] = (dcomplex**)malloc(sizeof(dcomplex*)*WANNUM); 
      for(i=0; i<WANNUM; i++){
	Mmnkb[k][bindx][i] = (dcomplex*)malloc(sizeof(dcomplex)*WANNUM); 
	for(j=0; j<WANNUM; j++){ 
	  Mmnkb[k][bindx][i][j].r=0.0; 
	  Mmnkb[k][bindx][i][j].i=0.0;
	}
      } /* i, j */
    }/* b vector */
  }/* k point*/

  M_zero = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);  
  for(k=0;k<kpt_num;k++){
    M_zero[k] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);    
    for(bindx=0;bindx<tot_bvector;bindx++){
      M_zero[k][bindx] = (dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);            
      for(i=0; i<WANNUM; i++){
	M_zero[k][bindx][i] = (dcomplex*)malloc(sizeof(dcomplex)*WANNUM);     
	for(j=0; j<WANNUM; j++){      
	  M_zero[k][bindx][i][j].r=0.0;     
	  M_zero[k][bindx][i][j].i=0.0;
	}
      } /* i, j */
    }/* b vector */
  }/* k point*/

  /* matrix for rotating Bloch bands */

  Ukmatrix=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Ukmatrix[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      Ukmatrix[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
        Ukmatrix[k][i][j].r=0.0;
        Ukmatrix[k][i][j].i=0.0;
        if(i==j){
          Ukmatrix[k][i][j].r=1.0;
          Ukmatrix[k][i][j].i=0.0;
        }
      }/* j */
    }/* i */
  }/* kpt */

  /* Gradients of spread function */

  Gmnk =(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Gmnk[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      Gmnk[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	Gmnk[k][i][j].r=0.0;
	Gmnk[k][i][j].i=0.0;
      }/* j */
    }/* i */
  }/* kpt */

  /* r and d vector */
  rmnk=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM*WANNUM*kpt_num);
  dmnk=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM*WANNUM*kpt_num);
  rimnk=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM*WANNUM*kpt_num);

  /* step taken in each trial step */
  deltaW=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    deltaW[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      deltaW[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	deltaW[k][i][j].r=0.0;
	deltaW[k][i][j].i=0.0;
      }/* j */
    }/* i */
  }/* kpt */

  /* centers for each wannier function r_bar_n 
     r_bar_n is defined in Eq. (31)
     array: wann_center[spinsize][nindx][0:2]
  */
  wann_center=(double**)malloc(sizeof(double*)*WANNUM);
  for(nindx=0;nindx<WANNUM;nindx++){
    wann_center[nindx]=(double*)malloc(sizeof(double)*3);
  }

  wann_r2=(double*)malloc(sizeof(double)*WANNUM);
  for(nindx=0;nindx<WANNUM;nindx++){
    wann_r2[nindx]=0.0;
  }

  /* here we start optimizing omega_tilde part */  

  if (myid==Host_ID){
    printf("\nStarting minimization of OMEGA_D and OMEGA_OD ......\n");fflush(0);
  }

  for(spin=0;spin<spinsize;spin++){

    if (myid==Host_ID){
      printf("For spin component %i: \n",spin);
    }

    /* M_zero */
    for(k=0;k<kpt_num;k++){
      for(bindx=0;bindx<tot_bvector;bindx++){
	for(i=0; i<WANNUM; i++){
	  for(j=0; j<WANNUM; j++){
	    M_zero[k][bindx][i][j].r=Mmnkb_zero[k][bindx][spin][i+1][j+1].r;
	    M_zero[k][bindx][i][j].i=Mmnkb_zero[k][bindx][spin][i+1][j+1].i;
	    Mmnkb[k][bindx][i][j].r=Mmnkb_zero[k][bindx][spin][i+1][j+1].r;
	    Mmnkb[k][bindx][i][j].i=Mmnkb_zero[k][bindx][spin][i+1][j+1].i;
	  }
	} /* i, j */
      }/* b vector */
    }/* k point*/
 
    /* contributions to spread function */
    omega=0.0;
    omega_I=0.0;
    omega_OD=0.0;
    omega_D=0.0;

    omega_prev=0.0;
    omega_I_prev=0.0;
    omega_OD_prev=0.0;
    omega_D_prev=0.0;

    delta_new=0.0;
    delta_0=0.0;
    delta_old=0.0;
    delta_mid=0.0;
    delta_d=0.0;
    yita=0.0;
    yita_prev=0.0;
    beta=0.0;
    /* step taken in each trial step */
    for(k=0;k<kpt_num;k++){
      for(i=0;i<WANNUM;i++){
	for(j=0;j<WANNUM;j++){
	  deltaW[k][i][j].r=0.0;
	  deltaW[k][i][j].i=0.0;
	}/* j */
      }/* i */
    }/* kpt */

    /* for initial guess of the Mmnkb matrix */
    for(k=0;k<kpt_num;k++){
      for(i=0;i<WANNUM;i++){
	for(j=0;j<WANNUM;j++){
	  Ukmatrix[k][i][j].r=0.0;
	  Ukmatrix[k][i][j].i=0.0;
	  if(i==j){
	    Ukmatrix[k][i][j].r=1.0;
	    Ukmatrix[k][i][j].i=0.0;
	  }
	}/* j */
      }/* i */
    }/* kpt */
    /* centers for each wannier function r_bar_n 
       r_bar_n is defined in Eq. (31)
       array: wann_center[nindx][0:2]
    */
    for(nindx=0;nindx<WANNUM;nindx++){
      for(i=0;i<3;i++){
	wann_center[nindx][i]=0.0;
      }
    } 
    for(nindx=0;nindx<WANNUM;nindx++){
      wann_r2[nindx]=0.0;
    }

    zero_Gk=0;
    step = 0;
    sec_step=0;
    sec_max=Wannier_Min_Secant_Steps;
    alpha=Wannier_Min_StepLength;
    epcg=0.001;
    searching_scheme=Wannier_Min_Scheme;
    delta_0=1.0;
    conv_total_omega=Wannier_Min_Conv_Criterion;
    sigma_0=-fabs(Wannier_Min_Secant_StepLength);
    /* determine the sheet selection */
    Center_Guide(Mmnkb, WANNUM, bdirection, nbdir, kpt_num, bvector, tot_bvector); 

    Wannier_Center(wann_center, wann_r2, Mmnkb, WANNUM, bvector, wb, kpt_num, tot_bvector);

    Cal_Omega(&omega_I, &omega_OD, &omega_D, Mmnkb, wann_center, WANNUM, bvector,
	      wb, kpt_num, tot_bvector);

    if (myid==Host_ID){
      printf("Initialized Wannier Function before optimization:\n");
      Output_WF_Spread("std", wann_center, wann_r2, omega_I, omega_OD, omega_D, WANNUM);
    }
    omega=omega_I + omega_OD+omega_D;
   
    if(searching_scheme==0 || searching_scheme==2){

      if (myid==Host_ID){
        printf("Using steepest decent (SD) scheme with fixed step length %10.5f:\n",alpha);fflush(0);

        if(BohrR_Wannier == 1.0){
           printf("|Opt Step |Mode of Gradient|d_Omega_in_steps|     d_Omega   | (in Bohr^2) ---> CONV \n");
           printf("|Opt Step |     Omega_I    |     Omega_D    |     Omega_OD  |    Tot_Omega  | (in Bohr^2) ---> SPRD \n");
        }else{
           printf("|Opt Step |Mode of Gradient|d_Omega_in_steps|     d_Omega   | (in Angs^2) ---> CONV \n");
           printf("|Opt Step |     Omega_I    |     Omega_D    |     Omega_OD  |    Tot_Omega  | (in Angs^2) ---> SPRD \n");
        }
      }

      while(step<max_steps&&fabs(delta_0)>conv_total_omega){

	omega_prev=omega;
	omega_I_prev=omega_I;
	omega_OD_prev=omega_OD;
	omega_D_prev=omega_D;
   
        /* calculation of gradient */

	Cal_Gradient(Gmnk,Mmnkb,wann_center,bvector,wb, WANNUM, kpt_num, tot_bvector, &zero_Gk); 

	/* get r0 and d0 */

	i=0;
	for(k=0;k<kpt_num;k++){
	  for(kj=0;kj<WANNUM;kj++){
	    for(ki=0;ki<WANNUM;ki++){
	      dmnk[i].r=Gmnk[k][ki][kj].r;
	      dmnk[i].i=Gmnk[k][ki][kj].i;
	      rmnk[i].r=Gmnk[k][ki][kj].r;
	      rmnk[i].i=Gmnk[k][ki][kj].i;
	      i++;
	    }
	  }
	} /* k */

	Calc_rT_dot_d(rmnk, dmnk, kpt_num*WANNUM*WANNUM, &delta_0);

	/* taking one step alpha*d */
	for(k=0;k<kpt_num;k++){
	  for(i=0;i<WANNUM;i++){
	    for(j=0;j<WANNUM;j++){
	      deltaW[k][i][j].r=Gmnk[k][i][j].r/wbtot/4.0*alpha;
	      deltaW[k][i][j].i=Gmnk[k][i][j].i/wbtot/4.0*alpha;
	    }/* j */
	  }/* i */
	} /* kpt */

	Cal_Ukmatrix(deltaW,Ukmatrix,kpt_num, WANNUM);

	/* update Mmnk matrix by M_opt= Uk * M_zero * U(k+b) */

	Updating_Mmnkb( Ukmatrix, kpt_num, WANNUM, WANNUM, Mmnkb, M_zero, 
                        kg, frac_bv, tot_bvector,kplusb);

	Wannier_Center(wann_center, wann_r2, Mmnkb, WANNUM, bvector, wb, kpt_num, tot_bvector);

	Cal_Omega(&omega_I, &omega_OD, &omega_D, Mmnkb, wann_center, WANNUM, bvector,
                  wb, kpt_num, tot_bvector);

	omega=omega_I+omega_OD+omega_D;

        if (myid==Host_ID){
	  printf("\n SD %5d ------------------------------------------------------------------------> CENT\n",step+1);
 	  Output_WF_Spread("std", wann_center, wann_r2, omega_I, omega_OD, omega_D, WANNUM);
	  printf("| SD%5d | %11.8E | %11.8E |%11.8E|  ---> CONV \n",
                  step+1,
                  delta_0,
                  alpha/4.0/wbtot*delta_0*BohrR_Wannier*BohrR_Wannier,
                  (omega-omega_prev)*BohrR_Wannier*BohrR_Wannier);
	  printf("| SD%5d |%14.8f  |%14.8f  |%14.8f |%14.8f |  ---> SPRD \n",
                  step+1,
                  omega_I*BohrR_Wannier*BohrR_Wannier, 
                  omega_D*BohrR_Wannier*BohrR_Wannier, 
                  omega_OD*BohrR_Wannier*BohrR_Wannier,
                  (omega_I+omega_D+omega_OD)*BohrR_Wannier*BohrR_Wannier);
	  fflush(0);
	}

	delta_0=omega-omega_prev;
	step++;
      }/* while loop for fixed optimizing step */
      if(step<max_steps&&fabs(delta_0)<conv_total_omega){
	searching_scheme=0; /* if already converged, not necessary to do CG any more*/
      }
    }/* searching_scheme ==0 || 2 with fixed step SD*/
    if(searching_scheme==1 || searching_scheme==2){/* CG method */
      step=0;
      delta_0=1.0;
      /* sigma_0=-2.0; */
      step=0;
      Cal_Gradient(Gmnk,Mmnkb,wann_center,bvector,wb, WANNUM, kpt_num, tot_bvector, &zero_Gk); /* r0 */
      /* get r0 and d0 */
      i=0;
      for(k=0;k<kpt_num;k++){
	for(kj=0;kj<WANNUM;kj++){
	  for(ki=0;ki<WANNUM;ki++){
	    dmnk[i].r=Gmnk[k][ki][kj].r;
	    dmnk[i].i=Gmnk[k][ki][kj].i;
	    rmnk[i].r=Gmnk[k][ki][kj].r;
	    rmnk[i].i=Gmnk[k][ki][kj].i;
	    i++;
	  }
	}
      } /* k */
      Calc_rT_dot_d(rmnk, dmnk, kpt_num*WANNUM*WANNUM, &delta_0);
      delta_new=delta_0;

      if (myid==Host_ID){
        printf("Using CG method for optimization Spread function. Max_step is %i:\n",max_steps);
        if(BohrR_Wannier == 1.0){
          printf("|Opt Step |Mode of Gradient|     d_Omega    | (Bohr^2) ---> CONV \n");
          printf("|Opt Step |     Omega_I    |     Omega_D    |     Omega_OD  |    Tot_Omega  | (Bohr^2) ---> SPRD \n");
        }else{
          printf("|Opt Step |Mode of Gradient|     d_Omega    | (Angs^2) ---> CONV \n");          
          printf("|Opt Step |     Omega_I    |     Omega_D    |     Omega_OD  |    Tot_Omega  | (Angs^2) ---> SPRD \n");
        }
      }

      while(step < max_steps && fabs(delta_0) > conv_total_omega ){

	omega_prev=omega;
	omega_I_prev=omega_I;
	omega_OD_prev=omega_OD;
	omega_D_prev=omega_D;

	sec_step=0;
	Calc_rT_dot_d(dmnk, dmnk, kpt_num*WANNUM*WANNUM, &delta_d);
	alpha=-sigma_0;
	/* for yita_prev. one trial step along search direction Gmnk(same as dmnk) and 
	   return rimnk as the gradient at that step. Keep Ukmatrix, Mmnkb^zero and WF not updated. 
	   f'(x+sigma*d) 
	*/
	Gradient_at_next_x(Gmnk, wbtot, sigma_0, Ukmatrix, M_zero, kpt_num, WANNUM, kg, bvector, frac_bv, wb, tot_bvector, kplusb, rimnk, &omega_I_next, &omega_OD_next, &omega_D_next);
	Calc_rT_dot_d(rimnk, dmnk, kpt_num*WANNUM*WANNUM, &yita_prev);
	if(debugcg){

          if (myid==Host_ID){
	    printf("      secant prev sec_step %i, sigma=%10.5f\n",sec_step, sigma_0);
	    printf("      secant Initial error mode is %10.5f \n",yita_prev);
	    fflush(0);
	  }

	}
	do{
	  /* return the gradient rimnk at this Ukmatrix (Mmnkb). WF and spread are recalculated. f'(x) */

	  new_Gradient(Mmnkb, kpt_num, WANNUM, bvector, wb, tot_bvector, rimnk, &omega_I, &omega_OD, &omega_D, wann_center, wann_r2);
	  Calc_rT_dot_d(rimnk,dmnk, kpt_num*WANNUM*WANNUM, &yita);

	  if(debugcg){

            if (myid==Host_ID){
  	      printf("      secant step %i:\n step size=%10.5f\n",sec_step,alpha);

	      Output_WF_Spread("std", wann_center, wann_r2, omega_I, omega_OD, omega_D, WANNUM);

	      printf("      secant previous error mode is %10.5f, present error mode is %10.5f \n",yita_prev,yita);
	      fflush(0);
	    }

	  }
	  alpha=alpha*yita/(yita_prev-yita);/* (omega_prev-omega)/fabs(omega_prev-omega); */
	  yita_prev=yita;
	  /* update Ukmatrix and Mmnkb by taking one step along the search direction Gmnk*/
	  Taking_one_step(Gmnk, wbtot, alpha, Ukmatrix, Mmnkb, M_zero, kpt_num, WANNUM, kg, frac_bv, tot_bvector,kplusb);
	  sec_step++;
	}while(sec_step < sec_max && alpha*alpha*delta_d > conv_total_omega );

	if(debugcg && myid==Host_ID){
	  printf("      Out of secant searching:");
	  if(sec_step < sec_max){
	    printf(" Secant searching converged.\n");
	  }else{
	    printf(" Secant searching not converged. Maybe no problem.\n");
	  }
	  fflush(0);
	}
	new_Gradient(Mmnkb, kpt_num, WANNUM, bvector, wb, tot_bvector, rimnk, &omega_I, &omega_OD, &omega_D, wann_center, wann_r2);
	delta_old=delta_new;
	Calc_rT_dot_d(rimnk,rmnk, kpt_num*WANNUM*WANNUM, &delta_mid);
	Calc_rT_dot_d(rimnk,rimnk,kpt_num*WANNUM*WANNUM, &delta_new);

	omega=omega_I+omega_D+omega_OD;

        if (myid==Host_ID){
  	  printf("\n CG %5d ------------------------------------------------------------------------> CENT\n",step+1);
	  Output_WF_Spread("std", wann_center, wann_r2, omega_I, omega_OD, omega_D, WANNUM);
        
  	  printf("| CG%5d | %11.8E | %11.8E|  ---> CONV \n",
                 step+1,delta_new, 
                 (omega-omega_prev)*BohrR_Wannier*BohrR_Wannier);
	  printf("| CG%5d | %14.8f | %14.8f |%14.8f |%14.8f |  ---> SPRD \n",
                 step+1,omega_I*BohrR_Wannier*BohrR_Wannier, 
                 omega_D*BohrR_Wannier*BohrR_Wannier, 
                 omega_OD*BohrR_Wannier*BohrR_Wannier,
                 (omega_I+omega_D+omega_OD)*BohrR_Wannier*BohrR_Wannier);
	}

	delta_0=omega-omega_prev;

	beta=(delta_new-delta_mid)/delta_old;

	if(debugcg && myid==Host_ID){
	  printf("beta=%10.5f for constructing next searching direction.\n", beta);
	}

	step++;
	/* spin problem beta][0] */
	if(step == kpt_num*WANNUM*WANNUM || beta <= 0.0){
	  /* re-set CG */

          if (myid==Host_ID){
	    printf("@@@@@@@@@@@@@@@@@@@@@@@@@@ Resetting CG! @@@@@@@@@@@@@@@@@@@@@@ ---> CONV\n");
	    printf("@@@@@@@@@@@@@@@@@@@@@@@@@@ Resetting CG! @@@@@@@@@@@@@@@@@@@@@@ ---> SPRD\n");
	  }

	  beta=0.0;
	  step = 0;
	}
	/* new searching direction */
	for(i=0;i<kpt_num*WANNUM*WANNUM;i++){
	  dmnk[i].r=rimnk[i].r+beta*dmnk[i].r;
	  dmnk[i].i=rimnk[i].i+beta*dmnk[i].i;
	  rmnk[i].r=rimnk[i].r;
	  rmnk[i].i=rimnk[i].i;
	}

	/* Gradient projected onto the the new searching direction */
	Calc_rT_dot_d(rimnk,dmnk, kpt_num*WANNUM*WANNUM, &delta_mid);

	if(delta_mid<0.0){ /* up-hill */

          if (myid==Host_ID){
	    printf("@@@@@@@@@@@@@@@ Search direction uphill. Reset CG. @@@@@@@@@@@@@@@ --->CONV\n");
	    printf("@@@@@@@@@@@@@@@ Search direction uphill. Reset CG. @@@@@@@@@@@@@@@ --->SPRD\n");
	  }

	  for(i=0;i<kpt_num*WANNUM*WANNUM;i++){
	    dmnk[i].r=rimnk[i].r;
	    dmnk[i].i=rimnk[i].i;
	    rmnk[i].r=rimnk[i].r;
	    rmnk[i].i=rimnk[i].i;
	  }
	  step=0;
	  Calc_rT_dot_d(rimnk,dmnk, kpt_num*WANNUM*WANNUM, &delta_mid);

	  if(delta_mid<0.0){ /* still uphill*/

            if (myid==Host_ID){
	      printf("@@@@@@@@@@@@@@@ Still uphill. Reverse direction. @@@@@@@@@@@@@@@ --->CONV\n");
	      printf("@@@@@@@@@@@@@@@ Still uphill. Reverse direction. @@@@@@@@@@@@@@@ --->SPRD\n");
	    }

	    for(i=0;i<kpt_num*WANNUM*WANNUM;i++){
	      dmnk[i].r=-rimnk[i].r;
	      dmnk[i].i=-rimnk[i].i;
	      rmnk[i].r=-rimnk[i].r;
	      rmnk[i].i=-rimnk[i].i;
	    }
	  }
	}
	/* new search direction */
	i=0;
	for(k=0;k<kpt_num;k++){
	  for(kj=0;kj<WANNUM;kj++){
	    for(ki=0;ki<WANNUM;ki++){
	      Gmnk[k][ki][kj].r=dmnk[i].r;
	      Gmnk[k][ki][kj].i=dmnk[i].i;
	      i++;
	    }
	  }
	} /* k */
      }/* end loop for CG */
    }/* searching_scheme==1 ||2 using CG method */

    if(step<max_steps && fabs(delta_0)<conv_total_omega){ /* convergence achieved */

      if (myid==Host_ID){
        printf("************************************************************* ---> CONV\n");
        printf("                  CONVERGENCE ACHIEVED !                      ---> CONV\n");
        printf("************************************************************* ---> CONV\n");
        printf("************************************************************* ---> SPRD\n");
        printf("                  CONVERGENCE ACHIEVED !                      ---> SPRD\n");
        printf("************************************************************* ---> SPRD\n");

        /* make a file *.wfinfo */

        Output_WF_Spread("file", wann_center, wann_r2, omega_I, omega_OD, omega_D, WANNUM);
      }

      /* store the rotation matrix and the converged overlapmatrix */

      for(k=0;k<kpt_num;k++){
	for(i=0;i<WANNUM;i++){
	  for(j=0;j<WANNUM;j++){
	    Utilde[spin][k][i][j].r=Ukmatrix[k][i][j].r;
	    Utilde[spin][k][i][j].i=Ukmatrix[k][i][j].i;
	  }
	}
	for(bindx=0;bindx<tot_bvector;bindx++){
	  for(i=0;i<WANNUM;i++){
	    for(j=0;j<WANNUM;j++){
	      Mmnkb_zero[k][bindx][spin][i][j].r=Mmnkb[k][bindx][i][j].r;
	      Mmnkb_zero[k][bindx][spin][i][j].i=Mmnkb[k][bindx][i][j].i;
	    }
	  }
	}/* bindx*/      
      }
    }

    else{

      if (myid==Host_ID){
        printf("**************************** WARNING ***************************** ---> CONV\n");
        printf("          Optimization NOT converged, please try again.            ---> CONV\n");
        printf("**************************** WARNING ***************************** ---> CONV\n");
        printf("**************************** WARNING ***************************** ---> SPRD\n");
        printf("          Optimization NOT converged, please try again.            ---> SPRD\n");
        printf("**************************** WARNING ***************************** ---> SPRD\n");
      }

      MPI_Finalize(); 
      exit(0);
    }

  }/* spin component */

  /* Multipy Uk(contains Udis*Uinitial) and Utilde(now it is Urot) to get Udis*Uinitial*Urot for wannier interpolation. Finall resutls are stored in Uk matrix */

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      for(i=0;i<Nk[spin][k][1]-Nk[spin][k][0];i++){
	for(j=0;j<WANNUM;j++){
	  tmpr=0.0; tmpi=0.0;
	  for(l=0;l<WANNUM;l++){
	    tmpr = tmpr+Uk[spin][k][i][l].r*Utilde[spin][k][l][j].r-Uk[spin][k][i][l].i*Utilde[spin][k][l][j].i;
	    tmpi = tmpi+Uk[spin][k][i][l].r*Utilde[spin][k][l][j].i+Uk[spin][k][i][l].i*Utilde[spin][k][l][j].r;
        
	  }
	  tmpM[i][j].r=tmpr;
	  tmpM[i][j].i=tmpi;
	}
      }
      for(i=0;i<Nk[spin][k][1]-Nk[spin][k][0];i++){
	for(j=0;j<WANNUM;j++){
	  Uk[spin][k][i][j].r=tmpM[i][j].r;
	  Uk[spin][k][i][j].i=tmpM[i][j].i;
	}    
      }
    }/* kpt */
  }/* spin */

  /* After convergence, we can release many things */
  /* Ukmatrix, all Mmnkb, all for gradients, matrix for rotating Bloch bands */

  free(wann_r2);
  for(nindx=0;nindx<WANNUM;nindx++){
    free(wann_center[nindx]);
  }
  free(wann_center);

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(deltaW[k][i]);
    }/* i */
    free(deltaW[k]);
  }/* kpt */
  free(deltaW);

  free(rimnk);
  free(dmnk);
  free(rmnk);

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(Gmnk[k][i]);
    }/* i */
    free(Gmnk[k]);
  }/* kpt */
  free(Gmnk); 

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(Ukmatrix[k][i]);
    }/* i */
    free(Ukmatrix[k]);
  }/* kpt */
  free(Ukmatrix);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(i=0; i<WANNUM; i++){
        free(M_zero[k][bindx][i]);
      } /* i, j */
      free(M_zero[k][bindx]);
    }/* b vector */
    free(M_zero[k]); 
  }/* k point*/
  free(M_zero);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(i=0; i<WANNUM; i++){
        free(Mmnkb[k][bindx][i]); 
      } /* i, j */
      free(Mmnkb[k][bindx]);
    }/* b vector */
    free(Mmnkb[k]);
  }/* k point*/
  free(Mmnkb); 


  for(i=0;i<3;i++){
    free(rguide[i]);
  }
  free(rguide);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      free(csheet[k][bindx]);
    }/* b vector */
    free(csheet[k]);
  }/* k point*/
  free(csheet);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      free(sheet[k][bindx]);
    }/* b vector */
    free(sheet[k]);
  }/* k point*/
  free(sheet);

  for(i=0;i<BANDNUM;i++){
    free(tmpM[i]);
  }/* j */
  free(tmpM);

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      for(i=0;i<WANNUM;i++){
        free(Utilde[spin][k][i]);
      }/* Windown */
      free(Utilde[spin][k]);
    } /*kpt */
    free(Utilde[spin]);
  } /* spin */
  free(Utilde);

  Wannier_Interpolation(Uk, eigen, spinsize, SpinP_switch, kpt_num, BANDNUM, WANNUM, Nk, rtv, kg, ChemP,r_num,rvect,ndegen);

  if(Wannier_Draw_MLWF){
    Wannier_Coef=(dcomplex*****)malloc(sizeof(dcomplex****)*r_num);
    if(SpinP_switch ==1 || SpinP_switch == 3){ /* in non-collinear case */
      ki=2;      
    }else{
      ki=1;
    }
    for(i=0;i<r_num;i++){
      Wannier_Coef[i]=(dcomplex****)malloc(sizeof(dcomplex***)*ki);
      for(spin=0;spin<ki;spin++){
        Wannier_Coef[i][spin]=(dcomplex***)malloc(sizeof(dcomplex**)*WANNUM);
        for(nindx=0;nindx<WANNUM;nindx++){
  	  Wannier_Coef[i][spin][nindx]=(dcomplex**)malloc(sizeof(dcomplex*)*(atomnum+1));
	  for(j=1;j<=atomnum;j++){
	    Wannier_Coef[i][spin][nindx][j]=(dcomplex*)malloc(sizeof(dcomplex)*Total_NumOrbs[j]);
	  }
        }
      }
    } 

  /* calculate Wannier_Coef */

    MLWF_Coef(Uk,Wkall,Nk,MP,kpt_num,fsize3,fsize, ki, SpinP_switch, 
	    BANDNUM,WANNUM, kg, r_num, rvect,Wannier_Coef);

  /* output of WFs to files in the cube format */

    OutData_WF(inputfile,r_num,rvect);fflush(0);

    /*
    if (SpinP_switch==3) Calc_SpinDirection_WF(r_num, rvect);
    */

  /* free arrays */
    if(SpinP_switch ==1 || SpinP_switch == 3){
      ki=2;
    }else{
      ki=1;
    }

    for(i=0;i<r_num;i++){
      for(spin=0;spin<ki;spin++){

        for(nindx=0;nindx<WANNUM;nindx++){
  	  for(j=1;j<=atomnum;j++){
	    free(Wannier_Coef[i][spin][nindx][j]);
	  }
	  free(Wannier_Coef[i][spin][nindx]);
        }
        free(Wannier_Coef[i][spin]);
      }
      free(Wannier_Coef[i]);
    } 
    free(Wannier_Coef);
  }/* plotting MLWFs */
  for(k=0;k<kpt_num;k++){
    for (spin=0; spin<spinsize; spin++){
      for (i=0; i<fsize3; i++){
	free(Wkall[k][spin][i]);
      }
      free(Wkall[k][spin]);
    }
    free(Wkall[k]);
  }
  free(Wkall);

  /********************************************
           Free  Arrays 
  ********************************************/

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      for(i=0;i<BANDNUM;i++){
        free(Uk[spin][k][i]);
      }/* Windown */
      free(Uk[spin][k]);
    } /*kpt */
    free(Uk[spin]);
  } /* spin */
  free(Uk);






  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(spin=0; spin<spinsize; spin++){
        for (i=0; i<BANDNUM+1; i++){
          free(Mmnkb_zero[k][bindx][spin][i] );
        } /* i, j band num */
        free(Mmnkb_zero[k][bindx][spin]);
      }/* spin */
      free(Mmnkb_zero[k][bindx]);
    }/* b vector */
    free(Mmnkb_zero[k]);
  }/* k point*/
  free(Mmnkb_zero);

  for(spin=0;spin<spinsize;spin++){
    for (k=0; k<kpt_num; k++){
      free(eigen[spin][k]);
    }
    free(eigen[spin]); 
  }
  free(eigen);

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      free(excludeBand[spin][k]);
    }
    free(excludeBand[spin]);
  }
  free(excludeBand);

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      free(innerNk[spin][k]);
    }
    free(innerNk[spin]);
  }
  free(innerNk);

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      free(Nk[spin][k]);
    }
    free(Nk[spin]);  
  }
  free(Nk);

  for(k=0;k<kpt_num;k++){
    free(kplusb[k]);
  }
  free(kplusb);
  
  free(bdirection);

  for(i=0;i<tot_bvector;i++){
    free(frac_bv[i]);
  }
  free(frac_bv);

  for(i=0;i<tot_bvector;i++){
    free(bvector[i]);
  }
  free(bvector);

  free(wb);
  free(M_s);
  for (i=0; i<kpt_num; i++){
    free(kg[i]);
  }
  free(kg);

  free(ndegen);
  for(i=0;i<3;i++){
    free(rvect[i]);
  }
  free(rvect);

  free(MP);

}/* End of main */


#pragma optimization_level 1
void MLWF_Coef(dcomplex ****Uk,dcomplex ****Wkall,int ***Nk, int *MP, int kpt_num, 
               int fsize3, int fsize, int spinsize, int SpinP_switch, int band_num,
               int wan_num, double **kg, int r_num, int **rvect, dcomplex *****Wannier_Coef)
{
  int i,j,k,spin, rindx, nindx,aindx,oindx, ki;
  double sumr, sumi, phr, phi, kRn, sumr2, sumi2;
  int ct_AN, Gc_AN, ialpha, h_AN, Rnh,l1,l2,l3,Gh_AN, tnoB, Bnum, Anum;
  dcomplex tmpWk[band_num][fsize3]; /* tmpWk2[band_num][fsize3]; */
  dcomplex tmp2Wk[band_num][fsize3];
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if(SpinP_switch!=3){

    for(rindx=0;rindx<r_num;rindx++){

      for(spin=0;spin<spinsize;spin++){

	for(nindx=0;nindx<wan_num;nindx++){

	  for(ct_AN=1;ct_AN<=atomnum;ct_AN++){

	    Gc_AN=ct_AN;
	    Anum=MP[ct_AN];

	    for(ialpha=0; ialpha<Total_NumOrbs[ct_AN]; ialpha++){

	      Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].r=0.0;  
	      Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].i=0.0;  

	      for(k=0; k<kpt_num; k++){ /* kpt */

		sumr=0.0;sumi=0.0;

		for(j=0; j<Nk[spin][k][1]-Nk[spin][k][0];j++){ /* bands */

		  tmpWk[j][ialpha].r=0.0;
		  tmpWk[j][ialpha].i=0.0;

		  l1 = rvect[0][rindx];
		  l2 = rvect[1][rindx];
		  l3 = rvect[2][rindx];

		  /* exp(i*k*(Rn+R)) */
		  kRn = 2.0*PI*(kg[k][0]*(double)l1 + kg[k][1]*(double)l2 + kg[k][2]*(double)l3); 

		  phi = sin(kRn);  /* Imaginary part */
		  phr = cos(kRn);  /* Real part      */

		  tmpWk[j][ialpha].r += phr*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].r
		    -phi*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].i;
		  tmpWk[j][ialpha].i += phr*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].i
		    +phi*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].r;

		  sumr+=Uk[spin][k][j][nindx].r*tmpWk[j][ialpha].r-Uk[spin][k][j][nindx].i*tmpWk[j][ialpha].i;        
		  sumi+=Uk[spin][k][j][nindx].r*tmpWk[j][ialpha].i+Uk[spin][k][j][nindx].i*tmpWk[j][ialpha].r;        

		} /* included bands sum up by rotation, disentangling */

		Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].r +=sumr;
		Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].i +=sumi;

	      } /* kpt sum up */

	      Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].r= Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].r/(double)(kpt_num);
	      Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].i= Wannier_Coef[rindx][spin][nindx][Gc_AN][ialpha].i/(double)(kpt_num);
	    }/* orbital */
	  }/* atom */
	} /* wannier */
      }/* spin */ 
    }/* R vector */
  }

  else if(SpinP_switch==3){
    spin=0;
    for(rindx=0;rindx<r_num;rindx++){
      for(nindx=0;nindx<wan_num;nindx++){
	for(ct_AN=1;ct_AN<=atomnum;ct_AN++){
	  Gc_AN=ct_AN;
	  Anum=MP[ct_AN];
	  for(ialpha=0;ialpha<Total_NumOrbs[ct_AN];ialpha++){
	    Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].r=0.0;  
	    Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].i=0.0;  
	    Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].r=0.0;    
	    Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].i=0.0;  
	    for(k=0;k<kpt_num;k++){ /* kpt */
	      sumr=0.0;sumi=0.0;
	      sumr2=0.0;sumi2=0.0;
	      for(j=0;j<Nk[spin][k][1]-Nk[spin][k][0];j++){ /* bands */
		tmpWk[j][ialpha].r=0.0;
		tmpWk[j][ialpha].i=0.0;
		tmp2Wk[j][ialpha].r=0.0;
		tmp2Wk[j][ialpha].i=0.0;
 	            l1 = rvect[0][rindx];
		    l2 = rvect[1][rindx];
		    l3 = rvect[2][rindx];
		    kRn = 2.0*PI*(kg[k][0]*(double)l1 + kg[k][1]*(double)l2 + kg[k][2]*(double)l3); /* exp(i*k*(Rn+R)) */
		    phi = sin(kRn);  /* Imaginary part */
		    phr = cos(kRn);  /* Real part      */
		    tmpWk[j][ialpha].r += phr*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].r-phi*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].i;
		    tmpWk[j][ialpha].i += phr*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].i+phi*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+ialpha].r;
		    tmp2Wk[j][ialpha].r+= phr*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+fsize+ialpha].r-phi*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+fsize+ialpha].i;
		    tmp2Wk[j][ialpha].i+= phr*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+fsize+ialpha].i+phi*Wkall[k][spin][j+1+Nk[spin][k][0]][Anum+fsize+ialpha].r;
		sumr+=Uk[spin][k][j][nindx].r*tmpWk[j][ialpha].r-Uk[spin][k][j][nindx].i*tmpWk[j][ialpha].i;
		sumi+=Uk[spin][k][j][nindx].r*tmpWk[j][ialpha].i+Uk[spin][k][j][nindx].i*tmpWk[j][ialpha].r;
		sumr2+=Uk[spin][k][j][nindx].r*tmp2Wk[j][ialpha].r-Uk[spin][k][j][nindx].i*tmp2Wk[j][ialpha].i;
		sumi2+=Uk[spin][k][j][nindx].r*tmp2Wk[j][ialpha].i+Uk[spin][k][j][nindx].i*tmp2Wk[j][ialpha].r;
	      }/* included bands sum up by rotation, disentangling */
	      Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].r +=sumr;
	      Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].i +=sumi;
	      Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].r +=sumr2;
	      Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].i +=sumi2;
	    } /* kpt sum up */
	    Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].r= Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].r/(double)(kpt_num);
	    Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].i= Wannier_Coef[rindx][0][nindx][Gc_AN][ialpha].i/(double)(kpt_num);
	    Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].r= Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].r/(double)(kpt_num);
	    Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].i= Wannier_Coef[rindx][1][nindx][Gc_AN][ialpha].i/(double)(kpt_num);
	  }/* orbital */
	}/* atom */
      } /* wannier */
    }/* R vector */
  }

  if(myid==Host_ID&&0==1){/* out put the components of MLWF at R(0,0,0) */
    oindx=-1;
    for(rindx=0;rindx<r_num;rindx++){
      if( rvect[0][rindx]==0 && rvect[1][rindx]==0 && rvect[2][rindx]==0){
         oindx=rindx;
         break; 
      }
    }
    if(oindx<0){
      printf("no R(0,0,0) found.\n");
      oindx=0;
    }
    printf("LCAO coefficient of MLWF with R(0,0,0) at %i.\n",oindx);
    if(SpinP_switch!=3){
      for(spin=0;spin<spinsize;spin++){
	printf("Spin componet %i:\n",spin);
	for(nindx=0;nindx<wan_num;nindx++){
	  printf(" WF%i:\n",nindx+1);
	  l1=0;
	  for(ct_AN=1;ct_AN<=atomnum;ct_AN++){
	    Anum=MP[ct_AN];
	    for(ialpha=0;ialpha<Total_NumOrbs[ct_AN];ialpha++){
	      printf(" Atom%iOrb%i  ",ct_AN,ialpha+1);
	      l1++;
	      if(l1%6==0){
		printf("\n");
		for(l2=0;l2<6;l2++){
		  printf("%10.5f  ",Wannier_Coef[oindx][spin][nindx][ct_AN][ialpha+1-6+l2].r);
		}
		printf("\n");
	      }
	      if(ct_AN==atomnum&&ialpha==Total_NumOrbs[ct_AN]-1&&l1%6!=0){ /* some left orbitals */
		printf("\n");
		for(l2=0;l2<l1%6;l2++){
		  printf("%10.5f  ",Wannier_Coef[oindx][spin][nindx][ct_AN][ialpha+1-(l1%6)+l2].r);
		}
		printf("\n");
	      }
	    } /* orbital */
	  } /* atom */
	} /* WF */
      }/* spin */ 
    }else{/* non-collinear case */
      printf("Non-Collinear case.\n");
      for(spin=0;spin<2;spin++){
	printf("Spin componet %i:\n",spin);
	for(nindx=0;nindx<wan_num;nindx++){
	  printf(" WF%i:\n",nindx+1);
	  l1=0;
	  for(ct_AN=1;ct_AN<=atomnum;ct_AN++){
	    Anum=MP[ct_AN];
	    for(ialpha=0;ialpha<Total_NumOrbs[ct_AN];ialpha++){
	      printf(" Atom%iOrb%i                 ",ct_AN,ialpha+1);
	      l1++;
	      if(l1%6==0){
		printf("\n");
		for(l2=0;l2<6;l2++){
		  printf("(%10.5f,%10.5f) ",Wannier_Coef[oindx][spin][nindx][ct_AN][ialpha+1-6+l2].r,Wannier_Coef[oindx][spin][nindx][ct_AN][ialpha+1-6+l2].i);
		}
		printf("\n");
	      }
	      if(ct_AN==atomnum&&ialpha==Total_NumOrbs[ct_AN]-1&&l1%6!=0){ /* some left orbitals */
		printf("\n");
		for(l2=0;l2<l1%6;l2++){
		  printf("(%10.5f,%10.5f) ",Wannier_Coef[oindx][spin][nindx][ct_AN][ialpha+1-(l1%6)+l2].r,Wannier_Coef[oindx][spin][nindx][ct_AN][ialpha+1-(l1%6)+l2].i);
		}
		printf("\n");
	      }
	    } /* orbital */
	  } /* atom */
	} /* WF */
      }/* spin */ 
    }/* non-collinear case */
  }

}/* MLWF_Coef */



#pragma optimization_level 1
void MPI_comm_OLP_Hks_iHks(double ****OLP, double *****Hks, double *****iHks)
{
  int Gc_AN,tno0,tno1,Cwan,h_AN,Hwan,Gh_AN;
  int i,j,k,num0,num,ID,Mc_AN;
  int numprocs,myid;
  double *vec0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
         allocation of fOLP, fHks, and fiHks
  ****************************************************/

  fOLP = (double****)malloc(sizeof(double***)*(atomnum+1)); 

  for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

    if (Gc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
    }
    else{
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_CNO[Cwan];  
    }   

    fOLP[Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      if (Gc_AN==0){
	tno1 = 1;  
      }
      else{
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_CNO[Hwan];
      } 
        
      fOLP[Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
      for (i=0; i<tno0; i++){
	fOLP[Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	for (j=0; j<tno1; j++){
	  fOLP[Gc_AN][h_AN][i][j] = 0.0;
	}
      }
    }
  }

  /****************************************************
         allocation of fHks
  ****************************************************/

  fHks = (double*****)malloc(sizeof(double****)*(SpinP_switch+1)); 

  for (k=0; k<=SpinP_switch; k++){

    fHks[k] = (double****)malloc(sizeof(double***)*(atomnum+1)); 

    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

      if (Gc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];  
      }   

      fHks[k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Gc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	} 
        
	fHks[k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
	for (i=0; i<tno0; i++){
	  fHks[k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	  for (j=0; j<tno1; j++){
	    fHks[k][Gc_AN][h_AN][i][j] = 0.0;
	  }
	}
      }
    }
  }

  /****************************************************
         allocation of fiHks
  ****************************************************/

  if (SpinP_switch==3){

    fiHks= (double*****)malloc(sizeof(double****)*List_YOUSO[5]); 
    for (k=0; k<List_YOUSO[5]; k++){

      fiHks[k] = (double****)malloc(sizeof(double***)*(atomnum+1)); 

      for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

	if (Gc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];  
	}   

	fiHks[k][Gc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Gc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  } 
        
	  fiHks[k][Gc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
	  for (i=0; i<tno0; i++){
	    fiHks[k][Gc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	    for (j=0; j<tno1; j++){
	      fiHks[k][Gc_AN][h_AN][i][j] = 0.0;
	    }
	  }
	}
      }
    }
  }

  /****************************************************
    MPI communication of OLP
  ****************************************************/

  num0 = List_YOUSO[2]*List_YOUSO[7]*List_YOUSO[7];
  vec0 = (double*)malloc(sizeof(double)*num0);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Spe_Total_CNO[Cwan];  

    if (myid==ID){

      Mc_AN = F_G2M[Gc_AN];

      num = 0;
      for (i=0; i<tno0; i++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];

	  for (j=0; j<tno1; j++){
	    vec0[num] = OLP[Mc_AN][h_AN][i][j];  num++;
	  }
	}
      }
    }

    MPI_Bcast(&num, 1, MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&vec0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0;
    for (i=0; i<tno0; i++){
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	tno1 = Spe_Total_CNO[Hwan];

	for (j=0; j<tno1; j++){
	  fOLP[Gc_AN][h_AN][i][j] = vec0[num];  num++;
	}
      }
    }

  } /* Gc_AN */   

  free(vec0);

  /****************************************************
    MPI communication of Hks
  ****************************************************/

  num0 = (SpinP_switch+1)*List_YOUSO[2]*List_YOUSO[7]*List_YOUSO[7];
  vec0 = (double*)malloc(sizeof(double)*num0);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Spe_Total_CNO[Cwan];  

    if (myid==ID){

      Mc_AN = F_G2M[Gc_AN];

      num = 0;
      for (k=0; k<=SpinP_switch; k++){
	for (i=0; i<tno0; i++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];

	    for (j=0; j<tno1; j++){
	      vec0[num] = Hks[k][Mc_AN][h_AN][i][j];  num++;
	    }
	  }
	}
      }
    }

    MPI_Bcast(&num, 1, MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&vec0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0;

    for (k=0; k<=SpinP_switch; k++){
      for (i=0; i<tno0; i++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];

	  for (j=0; j<tno1; j++){
	    fHks[k][Gc_AN][h_AN][i][j] = vec0[num];  num++;
	  }
	}
      }
    }
     
  } /* Gc_AN */   

  free(vec0);

  /****************************************************
    MPI communication of iHks
  ****************************************************/

  if (SpinP_switch==3){

    num0 = List_YOUSO[5]*List_YOUSO[2]*List_YOUSO[7]*List_YOUSO[7];
    vec0 = (double*)malloc(sizeof(double)*num0);

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID   = G2ID[Gc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_CNO[Cwan]; 

      if (myid==ID){

	Mc_AN = F_G2M[Gc_AN];

	num = 0;
	for (k=0; k<List_YOUSO[5]; k++){
	  for (i=0; i<tno0; i++){
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      tno1 = Spe_Total_CNO[Hwan];

	      for (j=0; j<tno1; j++){
		vec0[num] = iHks[k][Mc_AN][h_AN][i][j];  num++;
	      }
	    }
	  }
	}
      }

      MPI_Bcast(&num, 1, MPI_INT, ID, mpi_comm_level1);
      MPI_Bcast(&vec0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

      num = 0;
      for (k=0; k<List_YOUSO[5]; k++){
	for (i=0; i<tno0; i++){
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];

	    for (j=0; j<tno1; j++){
	      fiHks[k][Gc_AN][h_AN][i][j] = vec0[num];  num++;
	    }
	  }
	}
      }
     
    } /* Gc_AN */   

    free(vec0);
  }

}




#pragma optimization_level 1
void Projection_Amatrix(dcomplex ****Amnk, double **kg, int spinsize, 
                        int fsize, int SpinP_switch, 
                        int kpt_num, int band_num,  int wan_num, 
                        dcomplex ****Wkall, int *MP, int ***Nk)
{
  /* calculate the A matrix  */   
  int ct_AN, h_AN, Gh_AN, i,j,k, TNO1, TNO2;
  int spin, Rn, disentangle, BAND;
  int nindx, windx, proj_kind, tot_loc_basis;
  int mu1, L, tnoB, Anum, Bnum, Rnh, l1, l2, l3, i1,p;
  double co, si, kRn, sumr, sumi, tmpr, tmpi;
  double sr, pxr, pxi, pyr, pyi, pzr, pzi;
  dcomplex **tmpAmn; /* temporary Amn matrix*/ 
  dcomplex *tmpResult;
  FILE *fp;
  char fname[300];
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);
  

  /* count totally how many local orbitals are included */
  tot_loc_basis=0;
  for(i=0;i<Wannier_Num_Kinds_Projectors;i++){
    for(L=0;L<=3;L++){
      tot_loc_basis+=Wannier_NumL_Pro[i][L]*(2*L+1);
    }
  } 

  if(tot_loc_basis >=(2*3+1)){
    if(tot_loc_basis>=wan_num){
      tmpResult=(dcomplex*)malloc(sizeof(dcomplex)*tot_loc_basis);
    }
    else{
      tmpResult=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
    }
  }else{
    if(wan_num<2*3+1){
      tmpResult=(dcomplex*)malloc(sizeof(dcomplex)*(2*3+1));
    }else{
      tmpResult=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
    }
  }

  tmpAmn=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0;i<band_num;i++){
    tmpAmn[i]=(dcomplex*)malloc(sizeof(dcomplex)*tot_loc_basis);
  }

  BAND=band_num;
  if(band_num>wan_num){
    disentangle=1;
  }else{
    disentangle=0;
  }
  /*
    for(ct_AN=1;ct_AN<=atomnum;ct_AN++){
    printf("atom %i is (%10.5f,%10.5f,%10.5f)\n",ct_AN,Gxyz[ct_AN][1],Gxyz[ct_AN][2],Gxyz[ct_AN][3]);
    }
    for(i=0;i<wan_num;i++){
    printf("Projector Center: (%10.5f, %10.5f, %10.5f)\n",Wannier_Pos[i][1],Wannier_Pos[i][2],Wannier_Pos[i][3]);
    }
  */
  /* open file for writing A matrix */

  if(Wannier_Output_Projection_Matrix && myid==Host_ID){
    sprintf(fname,"%s%s.amn",filepath,filename);
    if((fp=fopen(fname,"wt"))==NULL){
      printf("******************************************************************\n");
      printf("* Error in opening file for Amn(k).\n");
      printf("******************************************************************\n");
    }else{
      printf(" ... ... Writting Amn(k) matrix into file.\n\n");
      fprintf(fp,"Amn. Fist line BANDNUM, KPTNUM, WANNUM, spinsize. Next is m n k and elements.Spin is the most outer loop.\n");
      fprintf(fp,"%13d%13d%13d%13d\n",band_num,kpt_num,wan_num,spinsize);
    }
  }

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      if(disentangle){
	band_num=Nk[spin][k][1]-Nk[spin][k][0];
      }else{
	band_num=wan_num;
      }
      for(mu1=0;mu1<band_num;mu1++){/* band index in enegy window */
	/* for(mu1=0;mu1<Nk[spin][k][1]-Nk[spin][k][0];mu1++){ */
	/*     for(nindx=0;nindx<wan_num;nindx++){ */
	windx=0;
	for(proj_kind=0;proj_kind<Wannier_Num_Kinds_Projectors; proj_kind++){
          Anum = 0;
          for (L=0; L<=3; L++){
            Anum += (2*L+1)*Wannier_NumL_Pro[proj_kind][L]; /* total number of basis for this kind of projector */
          }
	  /*  for(nindx=0;nindx<Wannier_Num_Pro[proj_kind];nindx++){ */
	  /*       printf("proj_kind =%i, projNum=%i Position index in tmpAmn is ",proj_kind, Anum);fflush(0); */
          for(nindx=0;nindx<Anum;nindx++){
	    /* 1. centeral atom  j */   
	    sumr=0.0;sumi=0.0; 
	    /* 3. neighboring site contributes to overlap i */

	    for(h_AN=0;h_AN<FNAN_WP[proj_kind];h_AN++){ /* neighboring atoms */
	      Rnh=ncn_WP[proj_kind][h_AN];
	      Gh_AN=natn_WP[proj_kind][h_AN];
	      tnoB=Total_NumOrbs[Gh_AN];
	      Bnum=MP[Gh_AN]; 
	      l1=atv_ijk[Rnh][1];
	      l2=atv_ijk[Rnh][2];
	      l3=atv_ijk[Rnh][3];
	      kRn=-2.0*PI*(kg[k][0]*(double)l1+kg[k][1]*(double)l2+kg[k][2]*(double)l3);
	      co=cos(kRn);
	      si=sin(kRn);
	      if(debug3){
		/*	printf("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n",ct_AN,h_AN,Gh_AN,Rnh); */
		/*		printf("overlap matrix OLP:\n"); */
	      }
	      for(i1=0;i1<tnoB;i1++){ /* basis on neighboring atom  alpha */
                if(SpinP_switch!=3){
                  tmpr= co*Wkall[k][spin][Nk[spin][k][0]+mu1+1][Bnum+i1].r+si*Wkall[k][spin][Nk[spin][k][0]+mu1+1][Bnum+i1].i;
                  tmpi=-Wkall[k][spin][Nk[spin][k][0]+mu1+1][Bnum+i1].i*co+si*Wkall[k][spin][Nk[spin][k][0]+mu1+1][Bnum+i1].r;
                }else{/* for non-collinear case */
                  if(windx<tot_loc_basis/2){ /* First half is for alpha spin state */
                    tmpr= co*Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+i1].r+si*Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+i1].i;
                    tmpi=-Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+i1].i*co+si*Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+i1].r;
		    /*
		      if(mu1==0&&k==0){
                      printf("windx=%i. NO fsize.\n",windx);
		      }
		    */
                  }else{/* Second half is for beta spin state */
                    tmpr= co*Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+fsize+i1].r+si*Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+fsize+i1].i;
                    tmpi=-Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+fsize+i1].i*co+si*Wkall[k][0][Nk[0][k][0]+mu1+1][Bnum+fsize+i1].r;
		    /*
		      if(mu1==0&&k==0){
                      printf("windx=%i. fsize added.\n",windx);fflush(0);
		      }
		    */
                  }
                }
		tmpr=tmpr*OLP_WP[proj_kind][h_AN][nindx][i1]; /* sqrt((double)(FNAN[ct_AN]+1)); */
		tmpi=tmpi*OLP_WP[proj_kind][h_AN][nindx][i1]; /* sqrt((double)(FNAN[ct_AN]+1)); */

		sumr=sumr+tmpr;
		sumi=sumi+tmpi;
		if(debug3){
		  printf("%i %i %10.7f \n",nindx,i1,OLP_WP[proj_kind][h_AN][nindx][i1]);
		}
	      } /*basis on neighboring atom */
	    }/* neighboring atom */
	    tmpAmn[mu1][windx].r=sumr/sqrt(fabs((double)FNAN_WP[proj_kind])); 
	    tmpAmn[mu1][windx].i=sumi/sqrt(fabs((double)FNAN_WP[proj_kind]));
	    /*            printf("%i ", windx);fflush(0); */
	    windx++;
	  }/*  nindx */
	  /*            printf("\n");fflush(0); */
	}/* proj_kind */
      }/* band in energy window */    
      /* Now we make rotation firstly, then selection and at last hybridize */   
      /*      printf("For Rotation:\n");fflush(0); */
      for(mu1=0;mu1<band_num;mu1++){
        windx=0;
        for(proj_kind=0;proj_kind<Wannier_Num_Kinds_Projectors; proj_kind++){
          for (L=0; L<=3; L++){
            if(Wannier_NumL_Pro[proj_kind][L]!=0 && L!=0){ /* if thie L is included and L!=0, rotate it*/
	      /*    printf("proj_kind=%i, rotate L=%i from %i to %i\n",proj_kind,L,windx,windx+2*L+1);fflush(0); */ 
              for(i=0;i<2*L+1;i++){
                sumr=0.0;sumi=0.0;
                for(j=0;j<2*L+1;j++){
		  /*                  printf("i=%i, j=%i \n",i,j);fflush(0); */
                  sumr=sumr+Wannier_RotMat_for_Real_Func[proj_kind][L][i][j]*tmpAmn[mu1][windx+j].r;
                  sumi=sumi+Wannier_RotMat_for_Real_Func[proj_kind][L][i][j]*tmpAmn[mu1][windx+j].i;
		  /*                  printf("sumr and sumi ok\n");fflush(0); */
                }
                tmpResult[i].r=sumr; 
                tmpResult[i].i=sumi;
              }
	      /*              printf("windx=%i L=%i.\n",windx,L);fflush(0); */
              for(i=0;i<2*L+1;i++){
		/*                printf("i=%i \n",i);fflush(0);  */
                tmpAmn[mu1][windx+i].r=tmpResult[i].r; 
                tmpAmn[mu1][windx+i].i=tmpResult[i].i; 
		/*                printf("i=%i \n",i);fflush(0); */
              }
              windx+=2*L+1; /* go over this L to the next */
	      /*              printf("windx=%i L=%i.\n",windx,L);fflush(0); */
            }else if(Wannier_NumL_Pro[proj_kind][L]!=0 && L==0){
	      /* printf("proj_kind=%i, skip L=%i from %i to %i\n",proj_kind,L,windx,windx+2*L+1);fflush(0);  */
              windx++; /* skip s orbital */
            }
          } /* Rotation done for each L */
        }
      }/* band index mu1 */   
      /* Make selection */
      /*      printf("For Selection:\n");fflush(0); */
      for(mu1=0;mu1<band_num;mu1++){
        windx=0; 
        nindx=0;
        for(proj_kind=0;proj_kind<Wannier_Num_Kinds_Projectors; proj_kind++){
          for(i=0;i<Wannier_Num_Pro[proj_kind];i++){ /* Wannier_Num_Pro[proj_kind] number projectors */
            Amnk[spin][k][mu1][nindx]=tmpAmn[mu1][Wannier_Select_Matrix[proj_kind][i]+windx];
	    /*       printf("proj_kind=%i, chose %i from tmpAmn to Amnk[%i]\n",
                     proj_kind, Wannier_Select_Matrix[proj_kind][i]+windx,nindx);
	    */		     
            nindx++;
          }
          for (L=0; L<=3; L++){
            windx += (2*L+1)*Wannier_NumL_Pro[proj_kind][L]; /* total number of basis for this kind of projector */
          }
        }
      }/* band index mu1 */  
      /* Make hybridization */ 
      /*      printf("For Hybridization:\n");fflush(0);  */
      for(mu1=0;mu1<band_num;mu1++){
        nindx=0;
        for(proj_kind=0;proj_kind<Wannier_Num_Kinds_Projectors; proj_kind++){
          for(i=0;i<Wannier_Num_Pro[proj_kind];i++){ /* Wannier_Num_Pro[proj_kind] number projectors */
            sumr=0.0;sumi=0.0;
            for(j=0;j<Wannier_Num_Pro[proj_kind];j++){ 
              sumr=sumr+Wannier_Projector_Hybridize_Matrix[proj_kind][i][j]*Amnk[spin][k][mu1][nindx+j].r;
              sumi=sumi+Wannier_Projector_Hybridize_Matrix[proj_kind][i][j]*Amnk[spin][k][mu1][nindx+j].i;
            }
            tmpResult[i].r=sumr;
            tmpResult[i].i=sumi; 
          }
          for(i=0;i<Wannier_Num_Pro[proj_kind];i++){ /* Wannier_Num_Pro[proj_kind] number projectors */
            Amnk[spin][k][mu1][nindx+i].r=tmpResult[i].r;
            Amnk[spin][k][mu1][nindx+i].i=tmpResult[i].i;
          }
	  /* 
	     printf("proj_kind=%i, from %i to %i hybridized.\n",proj_kind,nindx,nindx+Wannier_Num_Pro[proj_kind]);
	     for(i=0;i<Wannier_Num_Pro[proj_kind];i++){ 
	     for(j=0;j<Wannier_Num_Pro[proj_kind];j++){
	     printf("%8.5f  ",Wannier_Projector_Hybridize_Matrix[proj_kind][i][j]);
	     }
	     printf("\n");
	     }
	  */
          nindx+=Wannier_Num_Pro[proj_kind];
        }           
      }/* band index mu1 */  
      /* just for Benzene MOs */
      /*    printf("start Benzene!");fflush(0); */
      if(0){
	for(mu1=0;mu1<band_num;mu1++){
	  for(nindx=0;nindx<6*1;nindx++){
	    tmpResult[nindx].r=Amnk[spin][k][mu1][nindx].r;
	    tmpResult[nindx].i=Amnk[spin][k][mu1][nindx].i;
	  }
	  for(nindx=0;nindx<1;nindx++){
	    Amnk[spin][k][mu1][0+6*nindx].r=(tmpResult[0+6*nindx].r+tmpResult[1+6*nindx].r+tmpResult[2+6*nindx].r+tmpResult[3+6*nindx].r+tmpResult[4+6*nindx].r+tmpResult[5+6*nindx].r)/sqrt(6.0);
	    Amnk[spin][k][mu1][0+6*nindx].i=(tmpResult[0+6*nindx].i+tmpResult[1+6*nindx].i+tmpResult[2+6*nindx].i+tmpResult[3+6*nindx].i+tmpResult[4+6*nindx].i+tmpResult[5+6*nindx].i)/sqrt(6.0);

	    Amnk[spin][k][mu1][1+6*nindx].r=(tmpResult[1+6*nindx].r+tmpResult[2+6*nindx].r-tmpResult[4+6*nindx].r-tmpResult[5+6*nindx].r)/sqrt(4.0);
	    Amnk[spin][k][mu1][1+6*nindx].i=(tmpResult[1+6*nindx].i+tmpResult[2+6*nindx].i-tmpResult[4+6*nindx].i-tmpResult[5+6*nindx].i)/sqrt(4.0);

	    Amnk[spin][k][mu1][2+6*nindx].r=(2.0*tmpResult[0+6*nindx].r+tmpResult[1+6*nindx].r-tmpResult[2+6*nindx].r-2.0*tmpResult[3+6*nindx].r-tmpResult[4+6*nindx].r+tmpResult[5+6*nindx].r)/sqrt(12.0);
	    Amnk[spin][k][mu1][2+6*nindx].i=(2.0*tmpResult[0+6*nindx].i+tmpResult[1+6*nindx].i-tmpResult[2+6*nindx].i-2.0*tmpResult[3+6*nindx].i-tmpResult[4+6*nindx].i+tmpResult[5+6*nindx].i)/sqrt(12.0);

	    Amnk[spin][k][mu1][3+6*nindx].r=(tmpResult[1+6*nindx].r-tmpResult[2+6*nindx].r+tmpResult[4+6*nindx].r-tmpResult[5+6*nindx].r)/sqrt(4.0);
	    Amnk[spin][k][mu1][3+6*nindx].i=(tmpResult[1+6*nindx].i-tmpResult[2+6*nindx].i+tmpResult[4+6*nindx].i-tmpResult[5+6*nindx].i)/sqrt(4.0);

	    Amnk[spin][k][mu1][4+6*nindx].r=(2.0*tmpResult[0+6*nindx].r-tmpResult[1+6*nindx].r-tmpResult[2+6*nindx].r+2.0*tmpResult[3+6*nindx].r-tmpResult[4+6*nindx].r-tmpResult[5+6*nindx].r)/sqrt(12.0);
	    Amnk[spin][k][mu1][4+6*nindx].i=(2.0*tmpResult[0+6*nindx].i-tmpResult[1+6*nindx].i-tmpResult[2+6*nindx].i+2.0*tmpResult[3+6*nindx].i-tmpResult[4+6*nindx].i-tmpResult[5+6*nindx].i)/sqrt(12.0);

	    Amnk[spin][k][mu1][5+6*nindx].r=(tmpResult[0+6*nindx].r-tmpResult[1+6*nindx].r+tmpResult[2+6*nindx].r-tmpResult[3+6*nindx].r+tmpResult[4+6*nindx].r-tmpResult[5+6*nindx].r)/sqrt(6.0);
	    Amnk[spin][k][mu1][5+6*nindx].i=(tmpResult[0+6*nindx].i-tmpResult[1+6*nindx].i+tmpResult[2+6*nindx].i-tmpResult[3+6*nindx].i+tmpResult[4+6*nindx].i-tmpResult[5+6*nindx].i)/sqrt(6.0);
	  }
	}
      }
      /*     printf("end Benzene!kpt=%i\n",k);fflush(0); */
      if(Wannier_Output_Projection_Matrix && myid==Host_ID){ 
        for(nindx=0;nindx<wan_num;nindx++){
          for(mu1=0;mu1<BAND;mu1++){/* band index in enegy window */
            fprintf(fp,"%5d%5d%5d%18.12f%18.12f\n",mu1+1,nindx+1,k+1,Amnk[spin][k][mu1][nindx].r,Amnk[spin][k][mu1][nindx].i);
          } /* band */
        } /* wf */
      }
      /*    printf("next kpt=%i\n",k+1);fflush(0); */
    }/* kpt */
  }/* spin */  
  if(Wannier_Output_Projection_Matrix && myid==Host_ID){
    fclose(fp);
  }

  /**********************************************************
           Free Arrays 
  ***********************************************************/
  for(i=0;i<band_num;i++){
    free(tmpAmn[i]); 
  }
  free(tmpAmn);
  free(tmpResult);

}/* end of Projection_Amatrix */



#pragma optimization_level 1
void Getting_Uk(dcomplex ****Amnk, int spinsize, int kpt_num, int band_num, int wan_num, dcomplex ****Uk, int ***Nk){
  int spin, mu1, nindx, k, ia, info, i,j,lwork,ldu,lda;
  dcomplex **Amatrix;/* [band_num][wan_num]; */ /* for Amatrix[BANDNUM][WANNUM] */ 
  dcomplex *da; 
  dcomplex **Zmatrix;/* [band_num][band_num];*/ /* BANDNUMxBANDNUM matrix for left singular vector of Amatrix[BANDNUM][WANNUM] */
  dcomplex *dz;
  dcomplex **Dmatrix;/*[band_num][wan_num];  The same dimention as that of Amatrix, BANDNUMxWANNUM.
                       Its first WANNUM x WANNUM are diagonal with singular value on the diagonal, all 
                       other values are zero */
  double *dsing; /* store the singular values, which are the diagonal part of Dmatrix. */
  dcomplex **VTmatrix; /* a WANNUMxWANNUM matrix for right singular vector */
  dcomplex *dvt;
  char jobu, jobvt;
  double sumr, sumi;
  dcomplex *work;
  double *rwork;
  int debug31;
  jobu='A';
  jobvt='A';
  lwork=3*band_num;

  /* allocation of arrays */

  VTmatrix=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for(i=0; i<wan_num;i++){
    VTmatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  work=(dcomplex*)malloc(sizeof(dcomplex)*3*band_num);
  rwork=(double*)malloc(sizeof(double)*5*wan_num);

  /* start the calc. */

  debug31=0;
  for(spin=0;spin<spinsize;spin++){ /* spin */
    for(k=0;k<kpt_num;k++){/* kpt */

      /* Amatrix should be factorised using a singular value decomposition to get U, D, VT matrix */
      /* To use zgesvd, we need to do some conversion of arrays */

      band_num=Nk[spin][k][1]-Nk[spin][k][0];

      da=(dcomplex*)malloc(sizeof(dcomplex)*band_num*wan_num);
      dz=(dcomplex*)malloc(sizeof(dcomplex)*band_num*band_num);
      dvt=(dcomplex*)malloc(sizeof(dcomplex)*wan_num*wan_num);
      dsing=(double*)malloc(sizeof(double)*wan_num);

      for(ia=0;ia<band_num*wan_num;ia++){
	da[ia].r=0.0;da[ia].i=0.0;
      }
      for(ia=0;ia<band_num*band_num;ia++){
	dz[ia].r=0.0;dz[ia].i=0.0;
      }
      for(ia=0;ia<wan_num*wan_num;ia++){
	dvt[ia].r=0.0;dvt[ia].i=0.0;
      }

      Amatrix=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
      for(i=0;i<band_num;i++){
	Amatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      }

      Zmatrix=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
      for(i=0;i<band_num;i++){
	Zmatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
      }

      Dmatrix=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
      for(i=0;i<band_num;i++){
	Dmatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      }
      for(i=0;i<band_num;i++){
	for(j=0;j<wan_num;j++){
	  Dmatrix[i][j].r=0.0;       Dmatrix[i][j].i=0.0;
	}
      }

      ia=0;
      for(nindx=0;nindx<wan_num;nindx++){/*wan_num */
	for(mu1=0;mu1<band_num;mu1++){ /* band index */
	  Uk[spin][k][mu1][nindx].r=0.0;Uk[spin][k][mu1][nindx].i=0.0;
	  /* convert from 2-D arrays to 1-D arrays */
	  da[ia].r=Amnk[spin][k][mu1][nindx].r;
	  da[ia].i=Amnk[spin][k][mu1][nindx].i;
	  ia++;
	}/* band index */
      }/* wan num */ 
      if(debug31){
	printf("A matrix at k %2d, band num %i\n",k,band_num);
	for(mu1=0;mu1<band_num;mu1++){
	  for(nindx=0;nindx<wan_num;nindx++){
	    printf("(%10.5f, %10.5f) ",Amnk[spin][k][mu1][nindx].r,Amnk[spin][k][mu1][nindx].i);
	  }
	  printf("\n");
	}
      }  
      /* Singular Value Decompositon 
         Ozaki-san, would you please modify this calling of zgesvd?*/
      /*      zgesvd(jobu, jobvt, band_num, wan_num, da, band_num, dsing, dz, band_num, dvt, wan_num, &info); */
      /* clapack on Macbook Air */
      /*      zgesvd_(jobu, jobvt, band_num, wan_num, da, band_num, dsing, dz, band_num, dvt, 
              wan_num, work,2*wan_num+band_num, rwork, &info); */

      lda=band_num;
      ldu=band_num; 
  
      F77_NAME(zgesvd, ZGESVD)(&jobu,&jobvt, &band_num,&wan_num,da,&lda,dsing,
                               dz,&ldu,dvt,&wan_num,work,&lwork,rwork,&info);

      if(info==0){
	for(i=0;i<band_num;i++){
          for(j=0;j<wan_num;j++){
	    Dmatrix[i][j].r=0.0;           
            Dmatrix[i][j].i=0.0;
          }
	}
	for(i=0;i<wan_num;i++){
	  Dmatrix[i][i].r=dsing[i];
	}
	/*         if(debug3){
		   printf("Factorising OK and Singular value are:\n");
		   for(i=0;i<wan_num;i++){
		   printf("%10.5f\n",dsing[i]);
		   }
		   }
	*/

      }
 
      else if(info<0){
	printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
	printf("! In subroutine Getting_Uk:                !\n");
	printf("! When calling zgesvd, the %2dth argument  !\n",-info);
	printf("! had an illegal value.                    !\n");
	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for(i=0;i<band_num;i++){
          free(Dmatrix[i]);
        }
        free(Dmatrix);
      
        for(i=0;i<band_num;i++){
          free(Zmatrix[i]);
        }
        free(Zmatrix);
        
        for(i=0;i<band_num;i++){
          free(Amatrix[i]);
        }
        free(Amatrix);

	free(dsing);
	free(dvt);
	free(dz);
	free(da);
	exit(0);
      }else{
	printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
	printf("! In subroutine Getting_Uk:                !\n");
	printf("! Calling zgesvd failed in convergence.    !\n");
	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for(i=0;i<band_num;i++){
          free(Dmatrix[i]);
        }
        free(Dmatrix);

        for(i=0;i<band_num;i++){
          free(Zmatrix[i]);
        }
        free(Zmatrix);

        for(i=0;i<band_num;i++){
          free(Amatrix[i]);
        }
        free(Amatrix);

	free(dsing);
	free(dvt);
	free(dz);
	free(da);
	exit(0);
      }
      ia=0;
      for(j=0;j<band_num;j++){
	for(i=0;i<band_num;i++){
	  Zmatrix[i][j].r=dz[ia].r;
	  Zmatrix[i][j].i=dz[ia].i;
	  ia++;
	}
      }
      ia=0;
      for(j=0;j<wan_num;j++){
	for(i=0;i<wan_num;i++){
	  VTmatrix[i][j].r=dvt[ia].r;
	  VTmatrix[i][j].i=dvt[ia].i;
	  ia++;
	}
      }

      if(debug31){
	printf("Zmatrix is \n");
	for(j=0;j<band_num;j++){
          for(i=0;i<band_num;i++){
            printf("(%10.5f, %10.5f) ", Zmatrix[j][i].r, Zmatrix[j][i].i);
          }
          printf("\n");
	}
	printf("VTmatrix is \n");
	for(j=0;j<wan_num;j++){
          for(i=0;i<wan_num;i++){
            printf("(%10.5f, %10.5f) ", VTmatrix[j][i].r, VTmatrix[j][i].i);
          }
          printf("\n");
	}
	printf("Z*D is:\n");
	for(mu1=0;mu1<band_num;mu1++){
	  for(j=0;j<wan_num;j++){
	    sumr=0.0;sumi=0.0;
	    for(i=0;i<band_num;i++) {
	      sumr=sumr+Zmatrix[mu1][i].r*Dmatrix[i][j].r-Zmatrix[mu1][i].i*Dmatrix[i][j].i;
	      sumi=sumi+Zmatrix[mu1][i].r*Dmatrix[i][j].i+Zmatrix[mu1][i].i*Dmatrix[i][j].r;
	    }
	    Amatrix[mu1][j].r=sumr;  Amatrix[mu1][j].i=sumi;
	    printf("(%10.5f, %10.5f) ", Amatrix[mu1][j].r, Amatrix[mu1][j].i);
	  } 
	  printf("\n");
	}
	printf("Z*D*VT is \n");
	for(mu1=0;mu1<band_num;mu1++){
	  for(j=0;j<wan_num;j++){
	    sumr=0.0;sumi=0.0;
	    for(i=0;i<wan_num;i++){
	      sumr=sumr+Amatrix[mu1][i].r*VTmatrix[i][j].r-Amatrix[mu1][i].i*VTmatrix[i][j].i;
	      sumi=sumi+Amatrix[mu1][i].r*VTmatrix[i][j].i+Amatrix[mu1][i].i*VTmatrix[i][j].r;
	    }
	    Dmatrix[mu1][j].r=sumr;  Dmatrix[mu1][j].i=sumi;
	    printf("(%10.5f, %10.5f) ", Dmatrix[mu1][j].r, Dmatrix[mu1][j].i);
	  }
	  printf("\n");
	}
	printf("Z*I is\n"); 
      }
      for(i=0;i<band_num;i++){
	for(j=0;j<wan_num;j++){
	  Dmatrix[i][j].r=0.0; Dmatrix[i][j].i=0.0;
	}
      }
      for(i=0;i<wan_num;i++){
	Dmatrix[i][i].r=1.0;
      }
      for(mu1=0;mu1<band_num;mu1++){
	for(j=0;j<wan_num;j++){
	  sumr=0.0;sumi=0.0;
	  for(i=0;i<band_num;i++) {
	    sumr=sumr+Zmatrix[mu1][i].r*Dmatrix[i][j].r-Zmatrix[mu1][i].i*Dmatrix[i][j].i;
	    sumi=sumi+Zmatrix[mu1][i].r*Dmatrix[i][j].i+Zmatrix[mu1][i].i*Dmatrix[i][j].r;
	  }
	  Amatrix[mu1][j].r=sumr;  Amatrix[mu1][j].i=sumi;
	  if(debug31){
	    printf("(%10.5f, %10.5f) ", Amatrix[mu1][j].r, Amatrix[mu1][j].i); 
	  }
	}
	if(debug31){
	  printf("\n");
	} 
      }
      if(debug31){
	printf("At k point %i, Uk is\n",k);
      } 
      for(mu1=0;mu1<band_num;mu1++){ /* band index */
	for(nindx=0;nindx<wan_num;nindx++){ /*wan num*/
	  sumr=0.0; sumi=0.0;
	  for(i=0;i<wan_num;i++){
	    sumr=sumr+Amatrix[mu1][i].r*VTmatrix[i][nindx].r-Amatrix[mu1][i].i*VTmatrix[i][nindx].i;
	    sumi=sumi+Amatrix[mu1][i].r*VTmatrix[i][nindx].i+Amatrix[mu1][i].i*VTmatrix[i][nindx].r;
	  }
	  Uk[spin][k][mu1][nindx].r=sumr;
	  Uk[spin][k][mu1][nindx].i=sumi;
	  if(debug31){
	    printf("%18.12f%18.12f\n", Uk[spin][k][mu1][nindx].r, Uk[spin][k][mu1][nindx].i);
	  }
	}/* wan num */ 
	if(debug31){
	  printf("\n");fflush(0);
	}
      }/* band index */
      /*******************************************************
            free arrays 
      *******************************************************/
      for(i=0;i<band_num;i++){
	free(Dmatrix[i]);
      }
      free(Dmatrix);

      for(i=0;i<band_num;i++){
	free(Zmatrix[i]);
      }
      free(Zmatrix);

      for(i=0;i<band_num;i++){
	free(Amatrix[i]);
      }
      free(Amatrix);

      free(dsing);
      free(dvt);
      free(dz);
      free(da);
    }/* kpt */
  }/* spin */

  /* freeing of arrays */

  for(i=0; i<wan_num;i++){
    free(VTmatrix[i]);
  }
  free(VTmatrix);

  free(work);
  free(rwork);

}/* Getting_Uk  */


#pragma optimization_level 1
void Getting_Utilde(dcomplex ****Amnk, int spinsize, int kpt_num, 
                    int band_num, int wan_num, dcomplex ****Uk)
{
  int spin, mu1, nindx, k, ia, info, i,j,lwork;
  dcomplex **Amatrix;/* [band_num][wan_num];*/ /* for Amatrix[BANDNUM][WANNUM] */ 
  dcomplex *da; 
  dcomplex **Zmatrix;/* [band_num][band_num];*/ /* BANDNUMxBANDNUM matrix for left singular vector of Amatrix[BANDNUM][WANNUM] */
  dcomplex *dz;
  dcomplex **Dmatrix;/*[band_num][wan_num];  The same dimention as that of Amatrix, BANDNUMxWANNUM.
                       Its first WANNUM x WANNUM are diagonal with singular value on the diagonal, all 
                       other values are zero */
  double *dsing; /* store the singular values, which are the diagonal part of Dmatrix. */
  dcomplex **VTmatrix; /* a WANNUMxWANNUM matrix for right singular vector */
  dcomplex *dvt;
  char jobu, jobvt;
  double sumr, sumi;
  dcomplex *work;
  double *rwork;
  jobu='A';
  jobvt='A';
  lwork=3*band_num;

  /* allocation of arrays */

  da=(dcomplex*)malloc(sizeof(dcomplex)*band_num*wan_num);
  dz=(dcomplex*)malloc(sizeof(dcomplex)*band_num*band_num);
  dvt=(dcomplex*)malloc(sizeof(dcomplex)*wan_num*wan_num);
  dsing=(double*)malloc(sizeof(double)*wan_num);

  for(ia=0;ia<band_num*wan_num;ia++){
    da[ia].r=0.0;da[ia].i=0.0;
  }
  for(ia=0;ia<band_num*band_num;ia++){
    dz[ia].r=0.0;dz[ia].i=0.0;
  }
  for(ia=0;ia<wan_num*wan_num;ia++){
    dvt[ia].r=0.0;dvt[ia].i=0.0;
  }

  Amatrix=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0;i<band_num;i++){
    Amatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  Zmatrix=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0;i<band_num;i++){
    Zmatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  Dmatrix=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0;i<band_num;i++){
    Dmatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }
  for(i=0;i<band_num;i++){
    for(j=0;j<wan_num;j++){
      Dmatrix[i][j].r=0.0;       
      Dmatrix[i][j].i=0.0;
    }
  }

  VTmatrix=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for(i=0;i<wan_num;i++){
    VTmatrix[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  work=(dcomplex*)malloc(sizeof(dcomplex)*3*band_num);
  rwork=(double*)malloc(sizeof(double)*5*wan_num);

  /* start calc. */

  for(spin=0;spin<spinsize;spin++){ /* spin */
    for(k=0;k<kpt_num;k++){/* kpt */
      /*Amatrix should be factorised using a singular value decomposition to get U, D, VT matrix */
      /* To use zgesvd, we need to do some conversion of arrays */
      ia=0;
      for(nindx=0;nindx<wan_num;nindx++){/*wan_num */
	for(mu1=0;mu1<band_num;mu1++){ /* band index */
	  Uk[spin][k][mu1][nindx].r=0.0;Uk[spin][k][mu1][nindx].i=0.0;
	  /* convert from 2-D arrays to 1-D arrays */
	  da[ia].r=Amnk[spin][k][mu1][nindx].r;
	  da[ia].i=Amnk[spin][k][mu1][nindx].i;
	  ia++;
	}/* band index */
      }/* wan num */ 
      if(debug3){
	printf("A matrix at k %2d, band num %i\n",k,band_num);
	for(mu1=0;mu1<band_num;mu1++){
	  for(nindx=0;nindx<wan_num;nindx++){
	    printf("(%10.5f, %10.5f) ",Amnk[spin][k][mu1][nindx].r,Amnk[spin][k][mu1][nindx].i);
	  }
	  printf("\n");
	}
      }  
      /* Singular Value Decompositon 
         Ozaki-san, would you please modify this calling of zgesvd?*/
      /*      zgesvd(jobu, jobvt, band_num, wan_num, da, band_num, dsing, dz, band_num, dvt, wan_num, &info); */
      /* clapack on Macbook Air */
      /*      zgesvd_(jobu, jobvt, band_num, wan_num, da, band_num, dsing, dz, band_num, dvt, wan_num, work,2*wan_num+band_num, rwork, &info); */
      F77_NAME(zgesvd, ZGESVD)(&jobu,&jobvt, &band_num,&wan_num,da,&band_num,dsing,dz,&band_num,dvt,&wan_num,work,&lwork,rwork,&info);

      if(info==0){
	for(i=0;i<band_num;i++){
          for(j=0;j<wan_num;j++){
	    Dmatrix[i][j].r=0.0;           Dmatrix[i][j].i=0.0;
          }
	}
	for(i=0;i<wan_num;i++){
	  Dmatrix[i][i].r=dsing[i];
	}
	/*         if(debug3){
		   printf("Factorising OK and Singular value are:\n");
		   for(i=0;i<wan_num;i++){
		   printf("%10.5f\n",dsing[i]);
		   }
		   }
	*/
      }else if(info<0){
	printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
	printf("! In subroutine Getting_Utilde:                !\n");
	printf("! When calling zgesvd, the %2dth argument  !\n",-info);
	printf("! had an illegal value.                    !\n");
	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for(i=0;i<band_num;i++){
          free(Dmatrix[i]);
        }
        free(Dmatrix);
        
        for(i=0;i<band_num;i++){
          free(Zmatrix[i]);
        }
        free(Zmatrix);
        
        for(i=0;i<band_num;i++){
          free(Amatrix[i]);
        }
        free(Amatrix);

	free(dsing);
	free(dvt);
	free(dz);
	free(da);
	exit(0);
      }else{
	printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
	printf("! In subroutine Getting_Utilde:                !\n");
	printf("! Calling zgesvd failed in convergence.    !\n");
	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        for(i=0;i<band_num;i++){
          free(Dmatrix[i]);
        }
        free(Dmatrix);

        for(i=0;i<band_num;i++){
          free(Zmatrix[i]);
        }
        free(Zmatrix);

        for(i=0;i<band_num;i++){
          free(Amatrix[i]);
        }
        free(Amatrix);

	free(dsing);
	free(dvt);
	free(dz);
	free(da);
	exit(0);
      }
      ia=0;
      for(j=0;j<band_num;j++){
	for(i=0;i<band_num;i++){
	  Zmatrix[i][j].r=dz[ia].r;
	  Zmatrix[i][j].i=dz[ia].i;
	  ia++;
	}
      }
      ia=0;
      for(j=0;j<wan_num;j++){
	for(i=0;i<wan_num;i++){
	  VTmatrix[i][j].r=dvt[ia].r;
	  VTmatrix[i][j].i=dvt[ia].i;
	  ia++;
	}
      }

      if(debug3){
	printf("Zmatrix is \n");
	for(j=0;j<band_num;j++){
          for(i=0;i<band_num;i++){
            printf("(%10.5f, %10.5f) ", Zmatrix[j][i].r, Zmatrix[j][i].i);
          }
          printf("\n");
	}
	printf("VTmatrix is \n");
	for(j=0;j<wan_num;j++){
          for(i=0;i<wan_num;i++){
            printf("(%10.5f, %10.5f) ", VTmatrix[j][i].r, VTmatrix[j][i].i);
          }
          printf("\n");
	}
	printf("Z*D is:\n");
	for(mu1=0;mu1<band_num;mu1++){
	  for(j=0;j<wan_num;j++){
	    sumr=0.0;sumi=0.0;
	    for(i=0;i<band_num;i++) {
	      sumr=sumr+Zmatrix[mu1][i].r*Dmatrix[i][j].r-Zmatrix[mu1][i].i*Dmatrix[i][j].i;
	      sumi=sumi+Zmatrix[mu1][i].r*Dmatrix[i][j].i+Zmatrix[mu1][i].i*Dmatrix[i][j].r;
	    }
	    Amatrix[mu1][j].r=sumr;  Amatrix[mu1][j].i=sumi;
	    printf("(%10.5f, %10.5f) ", Amatrix[mu1][j].r, Amatrix[mu1][j].i);
	  } 
	  printf("\n");
	}
	printf("Z*D*VT is \n");
	for(mu1=0;mu1<band_num;mu1++){
	  for(j=0;j<wan_num;j++){
	    sumr=0.0;sumi=0.0;
	    for(i=0;i<wan_num;i++){
	      sumr=sumr+Amatrix[mu1][i].r*VTmatrix[i][j].r-Amatrix[mu1][i].i*VTmatrix[i][j].i;
	      sumi=sumi+Amatrix[mu1][i].r*VTmatrix[i][j].i+Amatrix[mu1][i].i*VTmatrix[i][j].r;
	    }
	    Dmatrix[mu1][j].r=sumr;  Dmatrix[mu1][j].i=sumi;
	    printf("(%10.5f, %10.5f) ", Dmatrix[mu1][j].r, Dmatrix[mu1][j].i);
	  }
	  printf("\n");
	}
	printf("Z*I is\n"); 
      }
      for(i=0;i<band_num;i++){
	for(j=0;j<wan_num;j++){
	  Dmatrix[i][j].r=0.0; Dmatrix[i][j].i=0.0;
	}
      }
      for(i=0;i<wan_num;i++){
	Dmatrix[i][i].r=1.0;
      }
      for(mu1=0;mu1<band_num;mu1++){
	for(j=0;j<wan_num;j++){
	  sumr=0.0;sumi=0.0;
	  for(i=0;i<band_num;i++) {
	    sumr=sumr+Zmatrix[mu1][i].r*Dmatrix[i][j].r-Zmatrix[mu1][i].i*Dmatrix[i][j].i;
	    sumi=sumi+Zmatrix[mu1][i].r*Dmatrix[i][j].i+Zmatrix[mu1][i].i*Dmatrix[i][j].r;
	  }
	  Amatrix[mu1][j].r=sumr;  Amatrix[mu1][j].i=sumi;
	  if(debug3){
	    printf("(%10.5f, %10.5f) ", Amatrix[mu1][j].r, Amatrix[mu1][j].i); 
	  }
	}
	if(debug3){
	  printf("\n");
	} 
      }
      if(debug3){
	printf("At k point %i, Uk is\n",k);
      } 
      for(mu1=0;mu1<band_num;mu1++){ /* band index */
	for(nindx=0;nindx<wan_num;nindx++){ /*wan num*/
	  sumr=0.0; sumi=0.0;
	  for(i=0;i<wan_num;i++){
	    sumr=sumr+Amatrix[mu1][i].r*VTmatrix[i][nindx].r-Amatrix[mu1][i].i*VTmatrix[i][nindx].i;
	    sumi=sumi+Amatrix[mu1][i].r*VTmatrix[i][nindx].i+Amatrix[mu1][i].i*VTmatrix[i][nindx].r;
	  }
	  Uk[spin][k][mu1][nindx].r=sumr;
	  Uk[spin][k][mu1][nindx].i=sumi;
	  if(debug3){
	    printf("(%10.5f, %10.5f) ", Uk[spin][k][mu1][nindx].r, Uk[spin][k][mu1][nindx].i);
	  }
	}/* wan num */ 
	if(debug3){
	  printf("\n");fflush(0);
	}
      }/* band index */
    }/* kpt */
  }/* spin */

  /*******************************************************
            free arrays 
  *******************************************************/

  free(rwork);
  free(work);

  for(i=0;i<wan_num;i++){
    free(VTmatrix[i]);
  }
  free(VTmatrix);

  for(i=0;i<band_num;i++){
    free(Dmatrix[i]);
  }
  free(Dmatrix);

  for(i=0;i<band_num;i++){
    free(Zmatrix[i]);
  }
  free(Zmatrix);

  for(i=0;i<band_num;i++){
    free(Amatrix[i]);
  }
  free(Amatrix);

  free(dsing);
  free(dvt);
  free(dz);
  free(da);

}/* Getting_Utilde  */



#pragma optimization_level 1
void Initial_Guess_Mmnkb(dcomplex ****Uk, int spinsize, int kpt_num, int band_num, 
                         int wan_num, dcomplex *****Mmnkb, dcomplex *****Mmnkb_zero, 
                         double **kg, double **bvector, int tot_bvector, int **kplusb, int ***Nk)
{
  int bindx, nindx, spin, k, i, j, kpb_band, disentangle;
  int ki,kj,kk;
  double kpb[3];
  dcomplex **M_zero;/* [band_num][band_num]; */
  dcomplex **Uk_hermitian;/* [wan_num][band_num]; */
  dcomplex **UkM;/* [wan_num][band_num]; */
  dcomplex **Ukb;/* [band_num][wan_num]; */
  double sumr, sumi;
  int mu1;
  FILE *fp;

  if(band_num>wan_num){
    disentangle=1;
  }else{
    disentangle=0;
  }
  
  for(spin=0;spin<spinsize; spin++){/* spin */
    for(k=0;k<kpt_num;k++){ /* kpt */
      if(disentangle){
        band_num=Nk[spin][k][1]-Nk[spin][k][0];
      }else{
        band_num=wan_num;
      }
      Uk_hermitian=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for(i=0;i<wan_num;i++){
        Uk_hermitian[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
      } 

      for(mu1=0;mu1<band_num;mu1++){/*band */
        for(nindx=0;nindx<wan_num;nindx++){ /* wan num */
          Uk_hermitian[nindx][mu1].r=Uk[spin][k][mu1][nindx].r;
          Uk_hermitian[nindx][mu1].i=-Uk[spin][k][mu1][nindx].i;
        }/* wan num */
      }/* band */

      for(bindx=0;bindx<tot_bvector;bindx++){ /* for b */
        kk=kplusb[k][bindx];
        if(disentangle){
          kpb_band=Nk[spin][kk][1]-Nk[spin][kk][0];
        }else{
          kpb_band=wan_num;
        }
        UkM=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
        for(i=0;i<wan_num;i++){
          UkM[i]=(dcomplex*)malloc(sizeof(dcomplex)*kpb_band);
        }

        Ukb=(dcomplex**)malloc(sizeof(dcomplex*)*kpb_band);
        for(i=0;i<kpb_band;i++){
          Ukb[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
        }
        M_zero=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
        for(i=0;i<band_num;i++){
          M_zero[i]=(dcomplex*)malloc(sizeof(dcomplex)*kpb_band);
        }
        for(mu1=0;mu1<kpb_band;mu1++){/*band */
          for(nindx=0;nindx<wan_num;nindx++){ /* wan num */
            Ukb[mu1][nindx].r=Uk[spin][kk][mu1][nindx].r;
            Ukb[mu1][nindx].i=Uk[spin][kk][mu1][nindx].i;
          }
        }
        for(mu1=0;mu1<band_num;mu1++){
          for(i=0;i<kpb_band;i++){
            M_zero[mu1][i].r=Mmnkb_zero[k][bindx][spin][mu1+1][i+1].r;
            M_zero[mu1][i].i=Mmnkb_zero[k][bindx][spin][mu1+1][i+1].i;
          }
        }
        /* updating by Uk^{+}*Mmnkb_zero*Uk+b */     
        for(i=0;i<wan_num;i++){
          for(mu1=0;mu1<kpb_band;mu1++){
            sumr=0.0;sumi=0.0;
            for(j=0;j<band_num;j++){
              sumr=sumr+Uk_hermitian[i][j].r*M_zero[j][mu1].r-Uk_hermitian[i][j].i*M_zero[j][mu1].i;
              sumi=sumi+Uk_hermitian[i][j].r*M_zero[j][mu1].i+Uk_hermitian[i][j].i*M_zero[j][mu1].r;
            }
            UkM[i][mu1].r=sumr;  UkM[i][mu1].i=sumi; 
          }
        }     
        for(i=0;i<wan_num;i++){
          for(j=0;j<wan_num;j++){
            sumr=0.0;sumi=0.0;
            for(mu1=0;mu1<kpb_band;mu1++){
              sumr=sumr+UkM[i][mu1].r*Ukb[mu1][j].r-UkM[i][mu1].i*Ukb[mu1][j].i;
              sumi=sumi+UkM[i][mu1].r*Ukb[mu1][j].i+UkM[i][mu1].i*Ukb[mu1][j].r;
            } 
            Mmnkb[k][bindx][spin][i+1][j+1].r=sumr; 
            Mmnkb[k][bindx][spin][i+1][j+1].i=sumi;
          }
        }      
        for(i=0;i<band_num;i++){
          free(M_zero[i]);
        }
        free(M_zero);

        for(i=0;i<kpb_band;i++){
	  free(Ukb[i]);
        }
        free(Ukb);

        for(i=0;i<wan_num;i++){
          free(UkM[i]);
        }
        free(UkM);
      }/* for b */
      for(i=0;i<wan_num;i++){
        free(Uk_hermitian[i]);
      }
      free(Uk_hermitian);
    }/* kpt */
  }/* spin */
}/* Initial_Guess_Mmnkb  */



#pragma optimization_level 1
void Updating_Mmnkb(dcomplex ***Uk, int kpt_num, int band_num, int wan_num, dcomplex ****Mmnkb, dcomplex ****Mmnkb_zero, double **kg, double **bvector, int tot_bvector, int **kplusb){
  int bindx, nindx, spin, k, i, j;
  int ki,kj,kk;
  double kpb[3];
  dcomplex **M_zero;
  dcomplex **Uk_hermitian,**UkM;
  dcomplex **Ukb;
  double sumr, sumi;
  int mu1;
  FILE *fp;

  /* allocation of arrays */

  M_zero=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0; i<band_num; i++){
    M_zero[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  Uk_hermitian=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for(i=0; i<wan_num; i++){
    Uk_hermitian[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  UkM=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for(i=0; i<wan_num; i++){
    UkM[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  Ukb=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0; i<band_num; i++){
    Ukb[i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  /* start calc. */

  for(k=0;k<kpt_num;k++){ /* kpt */
    for(bindx=0;bindx<tot_bvector;bindx++){ /* for b */
      kk=kplusb[k][bindx]; 
      for(mu1=0;mu1<band_num;mu1++){/*band */
	for(nindx=0;nindx<wan_num;nindx++){ /* wan num */
	  Uk_hermitian[nindx][mu1].r=Uk[k][mu1][nindx].r;
	  Uk_hermitian[nindx][mu1].i=-Uk[k][mu1][nindx].i;
	  Ukb[mu1][nindx].r=Uk[kk][mu1][nindx].r;
	  Ukb[mu1][nindx].i=Uk[kk][mu1][nindx].i;
	}/* wan num */
      }/* band */
      for(mu1=0;mu1<band_num;mu1++){
	for(i=0;i<band_num;i++){
	  M_zero[mu1][i].r=Mmnkb_zero[k][bindx][mu1][i].r;
	  M_zero[mu1][i].i=Mmnkb_zero[k][bindx][mu1][i].i;
	}
      }
      /* updating by Uk^{+}*Mmnkb_zero*Uk+b */
      for(i=0;i<wan_num;i++){
	for(mu1=0;mu1<band_num;mu1++){
	  sumr=0.0;sumi=0.0;
	  for(j=0;j<band_num;j++){
	    sumr=sumr+Uk_hermitian[i][j].r*M_zero[j][mu1].r-Uk_hermitian[i][j].i*M_zero[j][mu1].i;
	    sumi=sumi+Uk_hermitian[i][j].r*M_zero[j][mu1].i+Uk_hermitian[i][j].i*M_zero[j][mu1].r;
	  }
	  UkM[i][mu1].r=sumr;  UkM[i][mu1].i=sumi;
	}
      }
      for(i=0;i<wan_num;i++){
	for(j=0;j<wan_num;j++){
	  sumr=0.0;sumi=0.0;
	  for(mu1=0;mu1<band_num;mu1++){
	    sumr=sumr+UkM[i][mu1].r*Ukb[mu1][j].r-UkM[i][mu1].i*Ukb[mu1][j].i;
	    sumi=sumi+UkM[i][mu1].r*Ukb[mu1][j].i+UkM[i][mu1].i*Ukb[mu1][j].r;
	  }
	  Mmnkb[k][bindx][i][j].r=sumr; 
	  Mmnkb[k][bindx][i][j].i=sumi;
	}
      }
    }/* for b */
  }/* kpt */

  /* freeing of arrays */

  for(i=0; i<band_num; i++){
    free(M_zero[i]);
  }
  free(M_zero);

  for(i=0; i<wan_num; i++){
    free(Uk_hermitian[i]);
  }
  free(Uk_hermitian);

  for(i=0; i<wan_num; i++){
    free(UkM[i]);
  }
  free(UkM);

  for(i=0; i<band_num; i++){
    free(Ukb[i]);
  }
  free(Ukb);

}/* Updating_Mmnkb  */



#pragma optimization_level 1
void Wannier_Center(double **wann_center, double *wann_r2, dcomplex ****Mmnkb,int wan_num,
                    double **bvector, double *wb, int kpt_num, int tot_bvector){
  int i,j,k;
  int bindx, nindx;
  double norm,sum, nnr,nni;
  double ***lnMnnkb;
  double ***Mnnkb2;
  double tmp;

  /* allocation of arrays */

  lnMnnkb = (double***)malloc(sizeof(double**)*kpt_num);
  for (i=0; i<kpt_num; i++){
    lnMnnkb[i] = (double**)malloc(sizeof(double*)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      lnMnnkb[i][j] = (double*)malloc(sizeof(double)*wan_num);
    }
  }

  Mnnkb2 = (double***)malloc(sizeof(double**)*kpt_num);
  for (i=0; i<kpt_num; i++){
    Mnnkb2[i] = (double**)malloc(sizeof(double*)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      Mnnkb2[i][j] = (double*)malloc(sizeof(double)*wan_num);
    }
  }

  /* start calc. */

  for(nindx=0;nindx<wan_num;nindx++){
    wann_r2[nindx]=0.0;
    for(i=0;i<3;i++){
      wann_center[nindx][i]=0.0;
    }
  }   
    
  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(nindx=0;nindx<wan_num;nindx++){

	nnr=Mmnkb[k][bindx][nindx][nindx].r;
	nni=Mmnkb[k][bindx][nindx][nindx].i;
        norm=nnr*nnr+nni*nni;
        Mnnkb2[k][bindx][nindx]=norm; 

        nnr = Mmnkb[k][bindx][nindx][nindx].r*csheet[k][bindx][nindx].r
             -Mmnkb[k][bindx][nindx][nindx].i*csheet[k][bindx][nindx].i;
        nni = Mmnkb[k][bindx][nindx][nindx].r*csheet[k][bindx][nindx].i
             +Mmnkb[k][bindx][nindx][nindx].i*csheet[k][bindx][nindx].r;

        norm=fabs(nnr*nnr+nni*nni);
	norm=sqrt(norm);

        if(norm<1.0e-8){
          lnMnnkb[k][bindx][nindx]=0.0-sheet[k][bindx][nindx]; 
        }else{
  	  if(nni>=0.0){

            tmp = nnr/norm;
            tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
   	    lnMnnkb[k][bindx][nindx]= acos(tmp)-sheet[k][bindx][nindx];

      	  }else{

            tmp = nnr/norm;
            tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
	    lnMnnkb[k][bindx][nindx]=-acos(tmp)-sheet[k][bindx][nindx];
          }
        }

      } /* Wannier Function */
    }/*b vector */
  }/* k point */    

  for(nindx=0;nindx<wan_num;nindx++){ /* Wannier */
    for(i=0;i<3;i++){ /* cartesian */
      sum=0.0;
      for(k=0;k<kpt_num;k++){ /* k point */
        for(bindx=0;bindx<tot_bvector;bindx++){ /* b vectors */

	  sum=sum+wb[bindx]*bvector[bindx][i]*lnMnnkb[k][bindx][nindx];

        }/* b vector */
      }/* k point */

      wann_center[nindx][i]=-sum/(double)kpt_num;

    }/* cartesian */
  }/* wannier */

  for(nindx=0;nindx<wan_num;nindx++){ /* Wannier */

    sum=0.0;

    for(k=0;k<kpt_num;k++){ /* k point */
      for(bindx=0;bindx<tot_bvector;bindx++){ /* b vectors */

	sum=sum+wb[bindx]*(lnMnnkb[k][bindx][nindx]*lnMnnkb[k][bindx][nindx]+1.0-Mnnkb2[k][bindx][nindx]);
	/* printf("kpt %2d and bindx= %2d \n",k,bindx); fflush(0); */
      }/* b vectors */
    }/* k point */

    wann_r2[nindx]=sum/(double)kpt_num;

  }/* wannier */

  /* freeing of arrays */

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      free(lnMnnkb[i][j]);
    }
    free(lnMnnkb[i]);
  }
  free(lnMnnkb);

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      free(Mnnkb2[i][j]);
    }
    free(Mnnkb2[i]);
  }
  free(Mnnkb2);

}/* End of Wannier_Center */



#pragma optimization_level 1
void Cal_Omega(double *omega_I, double *omega_OD, double *omega_D, dcomplex ****Mmnkb, 
	       double **wann_center, int wan_num, double **bvector, double *wb, int kpt_num, int tot_bvector){
  int i,j,k,m,n;
  int bindx, nindx;
  double norm,sum, i_sum, od_sum, d_sum, omg_i, omg_od, omg_d, nnr, nni;
  double lnMnnkb;   /* wannierise.F90  1305L ln_tmp ImlnMnn kb */
  double tmp;

  *omega_I  = 0.0;
  *omega_OD = 0.0;
  *omega_D  = 0.0;  

  omg_i  = 0.0;
  omg_od = 0.0;
  omg_d  = 0.0;

  for(k=0;k<kpt_num;k++){
    for(bindx=0; bindx<tot_bvector; bindx++){ /* shell */

      i_sum  = 0.0;
      od_sum = 0.0;
      d_sum  = 0.0;

      for(m=0;m<wan_num;m++){
	for(n=0;n<wan_num;n++){

	  norm =  Mmnkb[k][bindx][m][n].r*Mmnkb[k][bindx][m][n].r
                + Mmnkb[k][bindx][m][n].i*Mmnkb[k][bindx][m][n].i;

	  i_sum = i_sum + norm;

	  if(sqrt(fabs(norm))>1.0){
  	    printf("ATTENTION! at k=%i, b=%i, m=%i, n=%i, Mmnkb model %10.8f>1.0\n",
                                   k,bindx,m,n,norm);
	  }

	  if(m!=n){

	    od_sum = od_sum + norm;

	  }

          else{ 

            nnr=Mmnkb[k][bindx][n][n].r*csheet[k][bindx][n].r-Mmnkb[k][bindx][n][n].i*csheet[k][bindx][n].i;
            nni=Mmnkb[k][bindx][n][n].r*csheet[k][bindx][n].i+Mmnkb[k][bindx][n][n].i*csheet[k][bindx][n].r;
            norm=sqrt(fabs(nnr*nnr+nni*nni));

            if(norm<1.0e-8){
              lnMnnkb =0.0-sheet[k][bindx][n];
            }else{
	      if(nni>=0.0){

                tmp = nnr/norm;
                tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
		lnMnnkb = acos(tmp)-sheet[k][bindx][n];
	      }else{

                tmp = nnr/norm;
                tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
		lnMnnkb = -acos(tmp)-sheet[k][bindx][n];
	      }
            }

	    d_sum = d_sum + (-lnMnnkb-bvector[bindx][0]*wann_center[n][0]-bvector[bindx][1]*wann_center[n][1]-bvector[bindx][2]*wann_center[n][2])*(-lnMnnkb-bvector[bindx][0]*wann_center[n][0]-bvector[bindx][1]*wann_center[n][1]-bvector[bindx][2]*wann_center[n][2]);
	  } /* m==n */
	} /* n Wannier Function */
      } /* m Wannier Function */

      omg_i=omg_i+wb[bindx]*(wan_num-i_sum);
      omg_od=omg_od+wb[bindx]*od_sum; 
      omg_d=omg_d+wb[bindx]*d_sum; 
    } /* b vectors */
  }/* k point */

  *omega_I=omg_i/(double)kpt_num;
  *omega_OD=omg_od/(double)kpt_num;
  *omega_D=omg_d/(double)kpt_num;
  
}/* Cal_Omega */



#pragma optimization_level 1
void  Cal_Gradient(dcomplex ***Gmnk,dcomplex ****Mmnkb, double **wann_center, 
                   double **bvector, double *wb, int wan_num, int kpt_num, 
                   int tot_bvector, int *zero_Gk)
{
  int i,j,k,m,n, bindx ;
  dcomplex ****Rmnkb;
  dcomplex ****R_tilde;
  dcomplex ****Tmnkb;
  double ***qnkb;
  double ***lnMnnkb;
  double nnr,nni,mnr,mni,sr,si;
  double tmp;

  /* allocation of arrays */

  Rmnkb = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);
  for (i=0; i<kpt_num; i++){
    Rmnkb[i] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      Rmnkb[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for (k=0; k<wan_num; k++){
        Rmnkb[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      }
    }
  }

  R_tilde = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);
  for (i=0; i<kpt_num; i++){
    R_tilde[i] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      R_tilde[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for (k=0; k<wan_num; k++){
        R_tilde[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      }
    }
  }

  Tmnkb = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);
  for (i=0; i<kpt_num; i++){
    Tmnkb[i] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      Tmnkb[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for (k=0; k<wan_num; k++){
        Tmnkb[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      }
    }
  }

  qnkb = (double***)malloc(sizeof(double**)*kpt_num);
  for (i=0; i<kpt_num; i++){
    qnkb[i] = (double**)malloc(sizeof(double*)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      qnkb[i][j] = (double*)malloc(sizeof(double)*wan_num);
    }
  }

  lnMnnkb = (double***)malloc(sizeof(double**)*kpt_num);
  for (i=0; i<kpt_num; i++){
    lnMnnkb[i] = (double**)malloc(sizeof(double*)*tot_bvector);
    for (j=0; j<tot_bvector; j++){
      lnMnnkb[i][j] = (double*)malloc(sizeof(double)*wan_num);
    }
  }

  /* start calc. */

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(m=0;m<wan_num;m++){
	qnkb[k][bindx][m]=0.0;
	lnMnnkb[k][bindx][m]=0.0; 
	for(n=0;n<wan_num;n++){
	  Rmnkb[k][bindx][m][n].r=0.0; 
	  Rmnkb[k][bindx][m][n].i=0.0; 
	  R_tilde[k][bindx][m][n].r=0.0;
	  R_tilde[k][bindx][m][n].i=0.0;
	  Tmnkb[k][bindx][m][n].r=0.0; 
	  Tmnkb[k][bindx][m][n].i=0.0; 
	}/* n */
      }/* m */
    }/* b vector  */
  } /* kpt */
  for(k=0;k<kpt_num;k++){
    for(i=0;i<wan_num;i++){      
      for(j=0;j<wan_num;j++){      
	Gmnk[k][i][j].r=0.0; 
	Gmnk[k][i][j].i=0.0;
      }/* j */
    }/* i */
  }/* kpt */  
  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(m=0;m<wan_num;m++){
	for(n=0;n<wan_num;n++){
	  nnr=Mmnkb[k][bindx][n][n].r;
	  nni=Mmnkb[k][bindx][n][n].i;
	  mnr=Mmnkb[k][bindx][m][n].r;
	  mni=Mmnkb[k][bindx][m][n].i;
	  Rmnkb[k][bindx][m][n].r=mnr*nnr+mni*nni;
	  Rmnkb[k][bindx][m][n].i=-mnr*nni+mni*nnr;
	  R_tilde[k][bindx][m][n].r=(mnr*nnr+mni*nni)/(nnr*nnr+nni*nni);
	  R_tilde[k][bindx][m][n].i=(-mnr*nni+mni*nnr)/(nnr*nnr+nni*nni);
          nnr=Mmnkb[k][bindx][n][n].r*csheet[k][bindx][n].r-Mmnkb[k][bindx][n][n].i*csheet[k][bindx][n].i;
          nni=Mmnkb[k][bindx][n][n].r*csheet[k][bindx][n].i+Mmnkb[k][bindx][n][n].i*csheet[k][bindx][n].r;

          if(sqrt(fabs(nnr*nnr+nni*nni))<1.0e-8){

            lnMnnkb[k][bindx][n]=0.0-sheet[k][bindx][n];

          }else{
	    if(nni>=0.0){

              tmp = nnr/sqrt(fabs(nnr*nnr+nni*nni));
              tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
  	      lnMnnkb[k][bindx][n] = acos(tmp)-sheet[k][bindx][n];
	    }else{

              tmp = nnr/sqrt(fabs(nnr*nnr+nni*nni));
              tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
	      lnMnnkb[k][bindx][n] =-acos(tmp)-sheet[k][bindx][n];
	    }
          }
	  qnkb[k][bindx][n] = bvector[bindx][0]*wann_center[n][0]+bvector[bindx][1]*wann_center[n][1]+bvector[bindx][2]*wann_center[n][2];
	  Tmnkb[k][bindx][m][n].r=R_tilde[k][bindx][m][n].r*(lnMnnkb[k][bindx][n]+qnkb[k][bindx][n]);
	  Tmnkb[k][bindx][m][n].i=R_tilde[k][bindx][m][n].i*(lnMnnkb[k][bindx][n]+qnkb[k][bindx][n]);
	}/* n */
      }/* m */
    } /* bindx */
  } /* kpt */

  /* 1 k point case check ok 
     printf("Rmnkb:\n");
     for(spin=0;spin<spinsize; spin++){
     for(k=0;k<kpt_num;k++){
     for(bindx=0;bindx<tot_bvector;bindx++){
     for(m=0;m<wan_num;m++){
     for(n=0;n<wan_num;n++){
     printf("Rmnkb k=%i b=%i m=%i n=%i %18.12f%18.12f\n",k,bindx,m+1,n+1,Rmnkb[spin][k][bindx][m][n].r, Rmnkb[spin][k][bindx][m][n].i);
     }
     printf("\n");
     }
     } 
     }
     }
     printf("R_tilde_mnkb:\n");
     for(spin=0;spin<spinsize; spin++){
     for(k=0;k<kpt_num;k++){
     for(bindx=0;bindx<tot_bvector;bindx++){
     for(m=0;m<wan_num;m++){
     for(n=0;n<wan_num;n++){
     printf("R_ti k=%i b=%i m=%i n=%i %18.12f%18.12f\n ",k,bindx,m+1,n+1, R_tilde[spin][k][bindx][m][n].r,R_tilde[spin][k][bindx][m][n].i);
     }
     printf("\n");
     }
     }
     }
     }
     printf("Tmnkb:\n");
     for(spin=0;spin<spinsize; spin++){
     for(k=0;k<kpt_num;k++){
     for(bindx=0;bindx<tot_bvector;bindx++){
     for(m=0;m<wan_num;m++){
     for(n=0;n<wan_num;n++){
     printf("Tmn k=%i b=%i m=%i n=%i %18.12f%18.12f\n ",k,bindx,m+1,n+1, Tmnkb[spin][k][bindx][m][n].r,Tmnkb[spin][k][bindx][m][n].i);
     }
     printf("\n");
     }
     }
     }
     }

     printf("qnkb:\n");
     for(spin=0;spin<spinsize; spin++){
     for(k=0;k<kpt_num;k++){
     for(bindx=0;bindx<tot_bvector;bindx++){
     for(n=0;n<wan_num;n++){
     printf("qnkb k=%i b=%i n=%i %18.12f\n",k,bindx,n+1, qnkb[spin][k][bindx][n]);
     }
     printf("\n");
     }
     }
     }
     printf("lnMnnkb:\n");
     for(spin=0;spin<spinsize; spin++){
     for(k=0;k<kpt_num;k++){
     for(bindx=0;bindx<tot_bvector;bindx++){
     for(n=0;n<wan_num;n++){
     printf("lnM k=%i b=%i n=%i %18.12f\n",k,bindx,n+1,lnMnnkb[spin][k][bindx][n]);
     }
     printf("\n");
     }
     }
     }
  */
  for(k=0;k<kpt_num;k++){
    for(m=0;m<wan_num;m++){
      for(n=0;n<wan_num;n++){
	nnr=0.0;
	nni=0.0;
	mnr=0.0;
	mni=0.0;
	sr=0.0; 
	si=0.0; 
	for(bindx=0;bindx<tot_bvector;bindx++){
	  nnr=nnr+wb[bindx]*(Rmnkb[k][bindx][m][n].r-Rmnkb[k][bindx][n][m].r)/2.0;  
	  nni=nni+wb[bindx]*(Rmnkb[k][bindx][m][n].i+Rmnkb[k][bindx][n][m].i)/2.0;    
	  mnr=mnr+wb[bindx]*(Tmnkb[k][bindx][m][n].i-Tmnkb[k][bindx][n][m].i)/2.0;  
	  mni=mni-wb[bindx]*(Tmnkb[k][bindx][m][n].r+Tmnkb[k][bindx][n][m].r)/2.0;  
	  /*
	    mnr=mnr+wb[bindx]*(R_tilde[k][bindx][m][n].i*lnMnnkb[k][bindx][n]-
	    R_tilde[k][bindx][n][m].i*lnMnnkb[k][bindx][m])/2.0;
	    mni=mni-wb[bindx]*(R_tilde[k][bindx][m][n].r*lnMnnkb[k][bindx][n]+
	    R_tilde[k][bindx][n][m].r*lnMnnkb[k][bindx][m])/2.0;
	    sr=sr+wb[bindx]*(R_tilde[k][bindx][m][n].i*qnkb[k][bindx][n]-
	    R_tilde[k][bindx][n][m].i*qnkb[k][bindx][m])/2.0;
	    si=si-wb[bindx]*(R_tilde[k][bindx][m][n].r*qnkb[k][bindx][n]+
	    R_tilde[k][bindx][n][m].r*qnkb[k][bindx][m])/2.0;
	  */
	} /* b vectors */
	Gmnk[k][m][n].r=4.0*(nnr-mnr)/(double)kpt_num;
	Gmnk[k][m][n].i=4.0*(nni-mni)/(double)kpt_num;
	/*         Gmnk[spin][k][m][n].r=4.0*(nnr-mnr-sr)/(double)kpt_num; */
	/*         Gmnk[spin][k][m][n].i=4.0*(nni-mni-si)/(double)kpt_num; */

      }/* n */
    }/* m */
  }/* kpt */ 

  /* freeing of arrays */

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      for (k=0; k<wan_num; k++){
        free(Rmnkb[i][j][k]);
      }
      free(Rmnkb[i][j]);
    }
    free(Rmnkb[i]);
  }
  free(Rmnkb);

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      for (k=0; k<wan_num; k++){
        free(R_tilde[i][j][k]);
      }
      free(R_tilde[i][j]);
    }
    free(R_tilde[i]);
  }
  free(R_tilde);

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      for (k=0; k<wan_num; k++){
        free(Tmnkb[i][j][k]);
      }
      free(Tmnkb[i][j]);
    }
    free(Tmnkb[i]);
  }
  free(Tmnkb);

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      free(qnkb[i][j]);
    }
    free(qnkb[i]);
  }
  free(qnkb);

  for (i=0; i<kpt_num; i++){
    for (j=0; j<tot_bvector; j++){
      free(lnMnnkb[i][j]);
    }
    free(lnMnnkb[i]);
  }
  free(lnMnnkb);

}/* Cal_Gradient */



#pragma optimization_level 1
void Cal_Ukmatrix(dcomplex ***deltaW, dcomplex ***Ukmatrix, int kpt_num, int wan_num)
{
  int i, j, k,l, m, n;
  dcomplex **Uk;
  dcomplex **expW;
  dcomplex *za,**zz,**H;
  char jobz, uplo;
  double *dw;
  int info, lwork;
  dcomplex *work;
  double *rwork;
  double sumr, sumi;

  /* allocation of arrays */

  Uk = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for (i=0; i<wan_num; i++){
    Uk[i] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  expW = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for (i=0; i<wan_num; i++){
    expW[i] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  za = (dcomplex*)malloc(sizeof(dcomplex)*wan_num*wan_num);

  zz = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for (i=0; i<wan_num; i++){
    zz[i] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  H = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for (i=0; i<wan_num; i++){
    H[i] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  dw = (double*)malloc(sizeof(double)*wan_num);

  work = (dcomplex*)malloc(sizeof(dcomplex)*3*wan_num);
  rwork = (double*)malloc(sizeof(double)*3*wan_num);

  /* start calc. */

  lwork=3*wan_num;
  jobz='V';
  uplo='U'; 

  for(k=0;k<kpt_num;k++){
    i=0;
    for(m=0;m<wan_num;m++){
      for(n=0;n<wan_num;n++){
	za[i].r=-deltaW[k][n][m].i; 
	za[i].i=deltaW[k][n][m].r;
	i++;
	Uk[m][n].r=Ukmatrix[k][m][n].r;
	Uk[m][n].i=Ukmatrix[k][m][n].i;
	H[m][n].r=-deltaW[k][m][n].i;
	H[m][n].i=deltaW[k][m][n].r;
      }/* n */ 
    }/* m */

    if(debug5){
      printf("k point %i, H is \n",k);
      for(m=0;m<wan_num;m++){
	for(n=0;n<wan_num;n++){      
	  printf("( %10.8f %10.8f )",H[m][n].r,H[m][n].i); 
	}
	printf("\n");
      }
    } 

    /*    zheev(jobz, uplo, wan_num, za, wan_num, dw, &info); */
    /* clapack  Macbook Air */ 
    /*   int zheev_(char *jobz, char *uplo, integer *n, doublecomplex
     *za, integer *lda, doublereal *dw, doublecomplex *work, integer *lwork,
     doublereal *rwork, integer *info)
     zheev_(jobz, uplo, wan_num, za, wan_num, dw, work, 2*wan_num-1, rwork, 3*wan_num-2, &info);
    */  
    F77_NAME(zheev, ZHEEV)(&jobz,&uplo,&wan_num,za,&wan_num,dw,work,&lwork,rwork,&info);

    if(info<0){
      printf("*********************** Error **********************\n");
      printf("Error in subroutine Cal_Ukmatrix when calling zheev\n");
      printf("with an illegal value for the %ith argument.\n", -info);
      printf("*********************** Error **********************\n");
      MPI_Finalize();
      exit(0); 
    }else if(info>0){
      printf("*********************** Error **********************\n");
      printf("Error in subroutine Cal_Ukmatrix when calling zheev\n");
      printf("which can not get convergence.\n");
      printf("*********************** Error **********************\n");
      MPI_Finalize();
      exit(0);
    }else{
      /*     printf("zheev success!\n"); */
    }
    /* getting the eigen vectors into zz matrix */
    i=0;
    for(m=0;m<wan_num;m++){
      for(n=0;n<wan_num;n++){
	zz[n][m].r=za[i].r;
	zz[n][m].i=za[i].i;
	i++;
	expW[n][m].r=0.0;
	expW[n][m].i=0.0;
	if(m==n){
	  expW[n][m].r=cos(dw[m]);
	  expW[n][m].i=-sin(dw[m]);
	}
      } /* n */
    }/* m */

    if(debug5){
      printf("Eigen value and vectors are:\n");
      for(m=0;m<wan_num;m++){
	printf("eigenvalue%i = %10.8f :",m+1,dw[m]);
	for(n=0;n<wan_num;n++){
	  printf("( %10.8f  %10.8f ), ",zz[n][m].r,zz[n][m].i);
	}
	printf("\n");
      }
      printf("checking HV=eV\n");
      for(l=0;l<wan_num;l++){
	printf("Eigen%i:\n",l);
	for(m=0;m<wan_num;m++){
	  sumr=0.0;sumi=0.0; 
	  for(n=0;n<wan_num;n++){
	    sumr=sumr+H[m][n].r*zz[n][l].r-H[m][n].i*zz[n][l].i;
	    sumi=sumi+H[m][n].r*zz[n][l].i+H[m][n].i*zz[n][l].r;
	  }
	  printf("( %10.8f %10.8f )",sumr/dw[l],sumi/dw[l]);
	}/* m */
	printf("\n");
      }/* l */ 
      printf("checking eigenvalue \n");
      for(l=0;l<wan_num;l++){
	sumr=0.0;sumi=0.0;
	for(m=0;m<wan_num;m++){
	  for(n=0;n<wan_num;n++){
	    sumr=sumr+(zz[m][l].r*H[m][n].r+zz[m][l].i*H[m][n].i)*zz[n][l].r-(zz[m][l].r*H[m][n].i-zz[m][l].i*H[m][n].r)*zz[n][l].i;
	    sumi=sumi+(zz[m][l].r*H[m][n].r+zz[m][l].i*H[m][n].i)*zz[n][l].i+(zz[m][l].r*H[m][n].i-zz[m][l].i*H[m][n].r)*zz[n][l].r;
	    /*              sumr=sumr+(zz[l][m].r*H[m][n].r-zz[l][m].i*H[m][n].i)*zz[n][l].r+(zz[l][m].r*H[m][n].i+zz[l][m].i*H[m][n].r)*zz[n][l].i; */
	    /*              sumi=sumi-(zz[l][m].r*H[m][n].r-zz[l][m].i*H[m][n].i)*zz[n][l].i+(zz[l][m].r*H[m][n].i+zz[l][m].i*H[m][n].r)*zz[n][l].r; */
	  } 
	}
	printf("eigenvalue %i=%10.8f %10.8f\n",l,sumr,sumi);
      }
      printf("checking Hamiltonian\n");
      for(l=0;l<wan_num;l++){
	for(m=0;m<wan_num;m++){
	  sumr=0.0;sumi=0.0;
	  for(n=0;n<wan_num;n++){
	    sumr=sumr+zz[l][n].r*dw[n]*zz[m][n].r+zz[l][n].i*dw[n]*zz[m][n].i;
	    sumi=sumi-zz[l][n].r*dw[n]*zz[m][n].i+zz[l][n].i*dw[n]*zz[m][n].r;
	    /*             sumr=sumr+(zz[l][m].r*H[m][n].r-zz[l][m].i*H[m][n].i)*zz[n][l].r+(zz[l][m].r*H[m][n].i+zz[l][m].i*H[m][n].r)*zz[n][l].i; */
	    /*             sumi=sumi-(zz[l][m].r*H[m][n].r-zz[l][m].i*H[m][n].i)*zz[n][l].i+(zz[l][m].r*H[m][n].i+zz[l][m].i*H[m][n].r)*zz[n][l].r; */
	  }
	  printf("(%10.8f %10.8f)",sumr,sumi);
	}
	printf("\n");
      }
    } /* debug5 */

    /*      printf("k=%i expWk=\n",k);  */
    for(l=0;l<wan_num;l++){
      for(m=0;m<wan_num;m++){
	sumr=0.0;sumi=0.0;
	for(n=0;n<wan_num;n++){
	  /*           sumr=sumr+(zz[l][n].r*cos(dw[n])+zz[l][n].i*sin(dw[n]))*zz[m][n].r+(-zz[l][n].r*sin(dw[n])+zz[l][n].i*cos(dw[n]))*zz[m][n].i; */
	  /*           sumi=sumi+(zz[l][n].r*cos(dw[n])+zz[l][n].i*sin(dw[n]))*(-zz[m][n].i)+(-zz[l][n].r*sin(dw[n])+zz[l][n].i*cos(dw[n]))*zz[m][n].r; */
	  sumr=sumr+zz[l][n].r*expW[n][m].r-zz[l][n].i*expW[n][m].i;
	  sumi=sumi+zz[l][n].r*expW[n][m].i+zz[l][n].i*expW[n][m].r;
	} /* n */
	H[l][m].r=sumr;
	H[l][m].i=sumi;
	/*         printf("(%10.8f %10.8f)",sumr,sumi); */
      }/* m */
	/*      printf("\n"); */
    }/* l */
    for(l=0;l<wan_num;l++){
      for(m=0;m<wan_num;m++){
	sumr=0.0;sumi=0.0;
	for(n=0;n<wan_num;n++){
	  /*           sumr=sumr+(zz[l][n].r*cos(dw[n])+zz[l][n].i*sin(dw[n]))*zz[m][n].r+(-zz[l][n].r*sin(dw[n])+zz[l][n].i*cos(dw[n]))*zz[m][n].i; */
	  /*           sumi=sumi+(zz[l][n].r*cos(dw[n])+zz[l][n].i*sin(dw[n]))*(-zz[m][n].i)+(-zz[l][n].r*sin(dw[n])+zz[l][n].i*cos(dw[n]))*zz[m][n].r; */
	  sumr=sumr+H[l][n].r*zz[m][n].r+H[l][n].i*zz[m][n].i;
	  sumi=sumi-H[l][n].r*zz[m][n].i+H[l][n].i*zz[m][n].r;
	} /* n */
	expW[l][m].r=sumr;
	expW[l][m].i=sumi;
	/*         printf(" %20.12f %20.12f \n ",sumr,sumi); */
      }/* m */
      /*       printf("\n"); */
    }/* l */

    /* updating Ukmatrix by Uk*expW */  
    /*     printf("k=%i Uk=\n",k); */
    for(l=0;l<wan_num;l++){
      for(m=0;m<wan_num;m++){
	sumr=0.0; sumi=0.0; 
	for(n=0;n<wan_num;n++){
	  sumr=sumr+Uk[l][n].r*expW[n][m].r-Uk[l][n].i*expW[n][m].i;
	  sumi=sumi+Uk[l][n].r*expW[n][m].i+Uk[l][n].i*expW[n][m].r;  
	}/* n */
	Ukmatrix[k][l][m].r=sumr;
	Ukmatrix[k][l][m].i=sumi;     
	/*         Ukmatrix[spin][k][l][m].r=expW[l][m].r; */
	/*         Ukmatrix[spin][k][l][m].i=expW[l][m].i; */
	/*         printf("(%10.8f %10.8f)",sumr,sumi); */
      }/* m */
      /*       printf("\n"); */
    }/* l */ 
  }/* kpt */

  /* freeing of arrays */

  for (i=0; i<wan_num; i++){
    free(Uk[i]);
  }
  free(Uk);

  for (i=0; i<wan_num; i++){
    free(expW[i]);
  }
  free(expW);

  free(za);

  for (i=0; i<wan_num; i++){
    free(zz[i]);
  }
  free(zz);

  for (i=0; i<wan_num; i++){
    free(H[i]);
  }
  free(H);

  free(dw);

  free(work);
  free(rwork);

} /* Cal_Ukmatrix */



#pragma optimization_level 1
void EigenState_k(double k1, double k2, double k3, int *MP, int spinsize, int SpinP_switch, 
                  int fsize, int fsize2, int fsize3,  dcomplex ***Wk1, double **EigenVal1, 
    	          double ****OLP, double *****Hks, double *****iHks)
{
  /*
    int spinsize (input variable)

    collinear spin-unpolarized:    spinsize = 1
    collinear spin-polarized:      spinsize = 2
    non-collinear spin-polarized:  spinsize = 1

    int fsize (input variable) 

    collinear spin-unpolarized:    fsize = sum of basis orbitals
    collinear spin-polarized:      fsize = sum of basis orbitals
    non-collinear spin-polarized:  fsize = 2 * sum of basis orbitals

    int fsize2 (input variable) 

    collinear spin-unpolarized:    fsize2 = fsize
    collinear spin-polarized:      fsize2 = fsize
    non-collinear spin-polarized:  fsize2 = 2*fsize

    int fsize3 (input variable) 

    collinear spin-unpolarized:    fsize2 = fsize + 2
    collinear spin-polarized:      fsize2 = fsize + 2
    non-collinear spin-polarized:  fsize2 = 2*fsize + 2

    int *MP (input variable)

    a pointer which shows the starting number
    of basis orbitals associated with atom i
    in the full matrix 

    dcomplex ***Wk1 (input/output variable)

    one-particle wave functions at the k1-point. Wk1[spin][i][m] 
    is the component m of the state i and spin index 'spin'.
    If diag_flag=2, then the input of Wk1 is used for calculation
    of the overlap matrix. The variables i and m run from 1 to fsize2.
    The corresponding eigenvalues are found in EigenVal1. 

    double **EigenVal1 (input/output variable)

    one-particle eigenenegies at the k1-point which are stored in 
    ascending order. EigenVal1[spin][i] is the eigenenegy of the
    state i and spin index 'spin', where the variable i runs from
    1 to fsize2. If diag_flag=2, then the input of EigenVal1 is
    not overwritten.
  */

  int spin;
  int ik,i1,j1,i,j,l,k;
  int ct_AN,h_AN,mu1,mu2;
  int Anum,Bnum,tnoA,tnoB;
  int mn,jj1,ii1,m;
  int Rnh,Gh_AN,l1,l2,l3;
  double kx,ky,kz;
  
  double sumr,sumi;
  double tmp1r,tmp1i;
  double tmp2r,tmp2i;
  double tmp3r,tmp3i;
  double si,co,kRn,k1da,k1db,k1dc;

  double *ko,*M1;
  dcomplex **S,**H,**C; 
  
  double OLP_eigen_cut = 1.0e-12;
  dcomplex Ctmp1,Ctmp2;  
  int numprocs,myid,ID,ID1;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  /****************************************************
                 allocation of arrays:
  ****************************************************/

  ko = (double*)malloc(sizeof(double)*fsize3);
  M1 = (double*)malloc(sizeof(double)*fsize3);

  S = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
  for (i=0; i<fsize3; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
    for (j=0; j<fsize3; j++){ S[i][j].r = 0.0; S[i][j].i = 0.0; }
  }

  H = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
  for (i=0; i<fsize3; i++){
    H[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
    for (j=0; j<fsize3; j++){ H[i][j].r = 0.0; H[i][j].i = 0.0; }
  }

  C = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
  for (i=0; i<fsize3; i++){
    C[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
    for (j=0; j<fsize3; j++){ C[i][j].r = 0.0; C[i][j].i = 0.0; }
  }
  
  /* 
     set kpoints, k1, k2 and k3 are fractional coordinates
     while kx, ky and kz are Cartesian coordinates
  */

  /*  printf("k1=%10.5f,k2=%10.5f,k3=%10.5f\n",k1,k2,k3); */

  kx = k1*rtv[1][1] + k2*rtv[2][1] + k3*rtv[3][1];
  ky = k1*rtv[1][2] + k2*rtv[2][2] + k3*rtv[3][2];
  kz = k1*rtv[1][3] + k2*rtv[2][3] + k3*rtv[3][3];

  Overlap_Band_Wannier(fOLP,S,MP,k1,k2,k3);
  EigenBand_lapack(S,ko,fsize,fsize,1);

  for (l=1; l<=fsize; l++){
    if (ko[l]<OLP_eigen_cut){
      printf("found an overcomplete basis set\n");

      MPI_Finalize(); 
      exit(0); 
    }
  } 

  for (l=1; l<=fsize; l++) M1[l] = 1.0/sqrt(ko[l]);

  /*
    for (i1=1; i1<=fsize; i1++){
    printf("ik=%2d i1=%3d ko[i1]=%15.12f\n",ik,i1,ko[i1]);
    }
  */

  /* S * M1  */

  for (i1=1; i1<=fsize; i1++){
    for (j1=1; j1<=fsize; j1++){
      S[i1][j1].r = S[i1][j1].r*M1[j1];
      S[i1][j1].i = S[i1][j1].i*M1[j1];
    } 
  } 

  for (spin=0; spin<spinsize; spin++){

    /* transpose S */

    for (i1=1; i1<=fsize; i1++){
      for (j1=i1+1; j1<=fsize; j1++){
	Ctmp1 = S[i1][j1];
	Ctmp2 = S[j1][i1];
	S[i1][j1] = Ctmp2;
	S[j1][i1] = Ctmp1;
      }
    }

    /****************************************************
                          collinear 
    ****************************************************/

    if (SpinP_switch==0 || SpinP_switch==1){

      Hamiltonian_Band_Wannier(fHks[spin],H,MP,k1,k2,k3);

      /****************************************************
                        M1 * U^t * H * U * M1
      ****************************************************/

      /* H * U * M1 */

      for (j1=1; j1<=fsize; j1++){
	for (i1=1; i1<=fsize; i1++){

	  sumr = 0.0;
	  sumi = 0.0;

	  for (l=1; l<=fsize; l++){
	    sumr += H[i1][l].r*S[j1][l].r - H[i1][l].i*S[j1][l].i;
	    sumi += H[i1][l].r*S[j1][l].i + H[i1][l].i*S[j1][l].r;
	  }

	  C[j1][i1].r = sumr;
	  C[j1][i1].i = sumi;
	}
      }     

      /* M1 * U^+ H * U * M1 */

      for (i1=1; i1<=fsize; i1++){
	for (j1=1; j1<=fsize; j1++){
	  sumr = 0.0;
	  sumi = 0.0;
	  for (l=1; l<=fsize; l++){
	    sumr +=  S[i1][l].r*C[j1][l].r + S[i1][l].i*C[j1][l].i;
	    sumi +=  S[i1][l].r*C[j1][l].i - S[i1][l].i*C[j1][l].r;
	  }
	  H[i1][j1].r = sumr;
	  H[i1][j1].i = sumi;
	  /*            printf("%10.5f ",sumr); */
	}
	/*              printf("\n"); */
      } 

      /* H to C */

      for (i1=1; i1<=fsize; i1++){
	for (j1=1; j1<=fsize; j1++){
	  C[i1][j1] = H[i1][j1];
	}
      }

      /* solve eigenvalue problem */

      EigenBand_lapack(C,EigenVal1[spin],fsize,fsize,1);

      /****************************************************
	     transformation to the original eigenvectors.
	     NOTE JRCAT-244p and JAIST-2122p 
      ****************************************************/

      /* transpose */
      for (i1=1; i1<=fsize; i1++){
	for (j1=i1+1; j1<=fsize; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /* transpose */
      for (i1=1; i1<=fsize; i1++){
	for (j1=i1+1; j1<=fsize; j1++){
	  Ctmp1 = C[i1][j1];
	  Ctmp2 = C[j1][i1];
	  C[i1][j1] = Ctmp2;
	  C[j1][i1] = Ctmp1;
	}
      }

      /* calculate wave functions */

      for (i1=1; i1<=fsize; i1++){
	for (j1=1; j1<=fsize; j1++){
	  sumr = 0.0;
	  sumi = 0.0;
	  for (l=1; l<=fsize; l++){
	    sumr +=  S[i1][l].r*C[j1][l].r - S[i1][l].i*C[j1][l].i;
	    sumi +=  S[i1][l].r*C[j1][l].i + S[i1][l].i*C[j1][l].r;
	  }

	  Wk1[spin][j1][i1].r = sumr;
	  Wk1[spin][j1][i1].i = sumi;
		
	}
      }
    }

    /****************************************************
                        non-collinear 
    ****************************************************/
 
    else if (SpinP_switch==3){

      Hamiltonian_Band_NC_Wannier(fHks,fiHks,H,MP,k1,k2,k3);

      /* H * U * M1 */

      for (j1=1; j1<=fsize; j1++){
	for (i1=1; i1<=fsize2; i1++){
	  for (m=0; m<=1; m++){

	    sumr = 0.0;
	    sumi = 0.0;
	    mn = m*fsize;
	    for (l=1; l<=fsize; l++){
	      sumr += H[i1][l+mn].r*S[j1][l].r - H[i1][l+mn].i*S[j1][l].i;
	      sumi += H[i1][l+mn].r*S[j1][l].i + H[i1][l+mn].i*S[j1][l].r;
	    }

	    jj1 = 2*j1 - 1 + m;

	    C[jj1][i1].r = sumr;
	    C[jj1][i1].i = sumi;
	  }
	}
      }     

      /* M1 * U^+ H * U * M1 */

      for (i1=1; i1<=fsize; i1++){

	for (m=0; m<=1; m++){

	  ii1 = 2*i1 - 1 + m;

	  for (j1=1; j1<=fsize2; j1++){
	    sumr = 0.0;
	    sumi = 0.0;
	    mn = m*fsize;
	    for (l=1; l<=fsize; l++){
	      sumr +=  S[i1][l].r*C[j1][l+mn].r + S[i1][l].i*C[j1][l+mn].i;
	      sumi +=  S[i1][l].r*C[j1][l+mn].i - S[i1][l].i*C[j1][l+mn].r;
	    }
	    H[ii1][j1].r = sumr;
	    H[ii1][j1].i = sumi;
	  }
	}
      }

      /* solve eigenvalue problem */

      EigenBand_lapack(H,EigenVal1[0],fsize2,fsize2,1);

      /****************************************************
             transformation to the original eigenvectors
                    NOTE JRCAT-244p and JAIST-2122p 
                       C = U * lambda^{-1/2} * D
      ****************************************************/

      /* transpose */

      for (i1=1; i1<=fsize; i1++){
	for (j1=i1+1; j1<=fsize; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      for (i1=1; i1<=fsize2; i1++){
	for (j1=1; j1<=fsize2; j1++){
	  C[i1][j1].r = 0.0;
	  C[i1][j1].i = 0.0;
	}
      }

      for (m=0; m<=1; m++){
	for (i1=1; i1<=fsize; i1++){
	  for (j1=1; j1<=fsize2; j1++){

	    sumr = 0.0; 
	    sumi = 0.0;

	    for (l=1; l<=fsize; l++){
	      sumr +=  S[i1][l].r*H[2*l-1+m][j1].r - S[i1][l].i*H[2*l-1+m][j1].i;
	      sumi +=  S[i1][l].r*H[2*l-1+m][j1].i + S[i1][l].i*H[2*l-1+m][j1].r;
	    } 

	    Wk1[0][j1][i1+m*fsize].r = sumr;
	    Wk1[0][j1][i1+m*fsize].i = sumi;

	  }
	}
      }

    }
  } /* spin */

  /* free of array */
  for (i=0; i<fsize3; i++){
    free(C[i]);
  }
  free(C);
 
  for (i=0; i<fsize3; i++){
    free(H[i]); 
  }
  free(H);
 
  for (i=0; i<fsize3; i++){
    free(S[i]); 
  }    
  free(S);

  free(M1); 
  free(ko);

} /* End of EigenState_k */
    






#pragma optimization_level 1
void Calc_Mmnkb(int k, int kk, double k1[4], double k2[4], double b[4], 
                int bindx, int SpinP_switch, int *MP, double Sop[2], 
                dcomplex *Wk1, dcomplex *Wk2, int fsize, int mm, int nn)
{
  int ct_AN,tnoA,Anum,h_AN;
  int i1,j1,l1,l2,l3,Rnh,Gh_AN,tnoB,Bnum;
  double si1,co1,si2,co2,kRn,dkx,dky,dkz;
  double tmp1r,tmp1i,tmp2r1,tmp2i1,tmp2r2,tmp2i2;
  double tmp3r,tmp3i,sumr,sumi;

  /****************************************************
                 allocation of arrays:
  ****************************************************/

  /* b in Cartesian coordinate */

  dkx = b[1]*rtv[1][1] + b[2]*rtv[2][1] + b[3]*rtv[3][1];
  dky = b[1]*rtv[1][2] + b[2]*rtv[2][2] + b[3]*rtv[3][2];
  dkz = b[1]*rtv[1][3] + b[2]*rtv[2][3] + b[3]*rtv[3][3];

  /****************************************************
  calculate an overlap matrix between one-particle
  wave functions  calculated at two k-points, k1 and k2
  ****************************************************/

  /* collinear  */

  if (SpinP_switch==0 || SpinP_switch==1){

    sumr = 0.0; 
    sumi = 0.0; 

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){ 

      tnoA = Total_NumOrbs[ct_AN];
      Anum = MP[ct_AN];

      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

	Rnh = ncn[ct_AN][h_AN];
	Gh_AN = natn[ct_AN][h_AN];
	tnoB = Total_NumOrbs[Gh_AN];
	Bnum = MP[Gh_AN];

	l1 = atv_ijk[Rnh][1];
	l2 = atv_ijk[Rnh][2];
	l3 = atv_ijk[Rnh][3];
	      
	/*  exp[i(k2 dot Rn - dk dot taui ) ]  */

	kRn = 2.0*PI*(k2[1]*(double)l1 + k2[2]*(double)l2 + k2[3]*(double)l3);
	kRn -= (dkx*Gxyz[ct_AN][1] + dky*Gxyz[ct_AN][2] + dkz*Gxyz[ct_AN][3]);

	si1 = sin(kRn);  /* Imaginary part */
	co1 = cos(kRn);  /* Real part      */
  	      
	/*  exp[i(-k1 dot Rn - dk dot taui ) ]  */

	kRn = -2.0*PI*(k1[1]*(double)l1 + k1[2]*(double)l2 + k1[3]*(double)l3);
	kRn -= (dkx*Gxyz[ct_AN][1] + dky*Gxyz[ct_AN][2] + dkz*Gxyz[ct_AN][3]);

	si2 = sin(kRn);  /* Imaginary part */
	co2 = cos(kRn);  /* Real part      */

	for (i1=0; i1<tnoA; i1++){
	  for (j1=0; j1<tnoB; j1++){

	    tmp1r = Wk1[Anum+i1].r*Wk2[Bnum+j1].r
	          + Wk1[Anum+i1].i*Wk2[Bnum+j1].i;
	    tmp1i = Wk1[Anum+i1].r*Wk2[Bnum+j1].i
	          - Wk1[Anum+i1].i*Wk2[Bnum+j1].r;

	    tmp2r1 = co1*tmp1r - si1*tmp1i;  
	    tmp2i1 = co1*tmp1i + si1*tmp1r; 

	    tmp1r = Wk1[Bnum+j1].r*Wk2[Anum+i1].r
	          + Wk1[Bnum+j1].i*Wk2[Anum+i1].i;

	    tmp1i = Wk1[Bnum+j1].r*Wk2[Anum+i1].i
	          - Wk1[Bnum+j1].i*Wk2[Anum+i1].r;

            tmp2r2 = co2*tmp1r - si2*tmp1i;
            tmp2i2 = co2*tmp1i + si2*tmp1r;

            tmp3r = OLPe[bindx][ct_AN][h_AN][i1][j1].r;
            tmp3i = OLPe[bindx][ct_AN][h_AN][i1][j1].i;

	    /* symmetrizing */

	    sumr += 0.5*(tmp2r1+tmp2r2)*tmp3r - 0.5*(tmp2i1+tmp2i2)*tmp3i;
	    sumi += 0.5*(tmp2r1+tmp2r2)*tmp3i + 0.5*(tmp2i1+tmp2i2)*tmp3r;

	    /* no symmetrizing */

	    /*
	    sumr += tmp2r1*tmp3r - tmp2i1*tmp3i;
	    sumi += tmp2r1*tmp3i + tmp2i1*tmp3r;
	    */

	  }/* j1*/
	}/* i1 */
      }   
    }       

    Sop[0] = sumr;
    Sop[1] = sumi;

  } /* if (SpinP_switch==0 || SpinP_switch==1) */

  /* non-collinear  */

  else if (SpinP_switch==3){

    sumr = 0.0; 
    sumi = 0.0; 

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){ 

      tnoA = Total_NumOrbs[ct_AN];
      Anum = MP[ct_AN];

      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){

	Rnh = ncn[ct_AN][h_AN];
	Gh_AN = natn[ct_AN][h_AN];
	tnoB = Total_NumOrbs[Gh_AN];
	Bnum = MP[Gh_AN];

	l1 = atv_ijk[Rnh][1];
	l2 = atv_ijk[Rnh][2];
	l3 = atv_ijk[Rnh][3];

	/*  exp[i(k2 dot Rn - dk dot taui ) ]  */

	kRn = 2.0*PI*(k2[1]*(double)l1 + k2[2]*(double)l2 + k2[3]*(double)l3);
	kRn -= (dkx*Gxyz[ct_AN][1] + dky*Gxyz[ct_AN][2] + dkz*Gxyz[ct_AN][3]);

	si1 = sin(kRn);  /* Imaginary part */
	co1 = cos(kRn);  /* Real part      */

	/*  exp[i(-k1 dot Rn - dk dot taui ) ]  */

	kRn = -2.0*PI*(k1[1]*(double)l1 + k1[2]*(double)l2 + k1[3]*(double)l3);
	kRn -= (dkx*Gxyz[ct_AN][1] + dky*Gxyz[ct_AN][2] + dkz*Gxyz[ct_AN][3]);

	si2 = sin(kRn);  /* Imaginary part */
	co2 = cos(kRn);  /* Real part      */
    
	for (i1=0; i1<tnoA; i1++){
	  for (j1=0; j1<tnoB; j1++){

	    tmp1r = Wk1[Anum      +i1].r*Wk2[Bnum      +j1].r
 	          + Wk1[Anum      +i1].i*Wk2[Bnum      +j1].i
	          + Wk1[Anum+fsize+i1].r*Wk2[Bnum+fsize+j1].r
	          + Wk1[Anum+fsize+i1].i*Wk2[Bnum+fsize+j1].i;

	    tmp1i = Wk1[Anum      +i1].r*Wk2[Bnum      +j1].i
	          - Wk1[Anum      +i1].i*Wk2[Bnum      +j1].r
	          + Wk1[Anum+fsize+i1].r*Wk2[Bnum+fsize+j1].i
	          - Wk1[Anum+fsize+i1].i*Wk2[Bnum+fsize+j1].r;

	    tmp2r1 = co1*tmp1r - si1*tmp1i;
	    tmp2i1 = co1*tmp1i + si1*tmp1r;

	    tmp1r = Wk1[Bnum      +j1].r*Wk2[Anum      +i1].r
 	          + Wk1[Bnum      +j1].i*Wk2[Anum      +i1].i
	          + Wk1[Bnum+fsize+j1].r*Wk2[Anum+fsize+i1].r
	          + Wk1[Bnum+fsize+j1].i*Wk2[Anum+fsize+i1].i;

	    tmp1i = Wk1[Bnum      +j1].r*Wk2[Anum      +i1].i
 	          - Wk1[Bnum      +j1].i*Wk2[Anum      +i1].r
	          + Wk1[Bnum+fsize+j1].r*Wk2[Anum+fsize+i1].i
	          - Wk1[Bnum+fsize+j1].i*Wk2[Anum+fsize+i1].r;

	    tmp2r2 = co2*tmp1r - si2*tmp1i;
	    tmp2i2 = co2*tmp1i + si2*tmp1r;

	    tmp3r = OLPe[bindx][ct_AN][h_AN][i1][j1].r;
	    tmp3i = OLPe[bindx][ct_AN][h_AN][i1][j1].i;

	    /* symmetrizing */

	    sumr += 0.5*(tmp2r1+tmp2r2)*tmp3r - 0.5*(tmp2i1+tmp2i2)*tmp3i;
	    sumi += 0.5*(tmp2r1+tmp2r2)*tmp3i + 0.5*(tmp2i1+tmp2i2)*tmp3r;

	    /* no symmetrizing */

	    /*
	    sumr += tmp2r1*tmp3r - tmp2i1*tmp3i;
	    sumi += tmp2r1*tmp3i + tmp2i1*tmp3r;
	    */


	    /*
        if (k==99 && bindx==9 && mm==1 && nn==0){

    printf("bindx=%2d ct_AN=%2d h_AN=%2d i1=%2d j1=%2d tmp2r1=%15.12f tmp2i1=%15.12f tmp3r=%15.12f tmp3i=%15.12f sumr=%15.12f sumi=%15.12f\n",
	   bindx,ct_AN,h_AN,i1,j1,tmp2r1,tmp2i1,tmp3r,tmp3i,sumr,sumi); 
	}
	    */

	  }
	}
      }   
    }       

    Sop[0] = sumr;
    Sop[1] = sumi;

  } /* else if (SpinP_switch==3) */

}  





#pragma optimization_level 1
void Shell_Structure(double **klatt, int *M_s, int **bvector, int *shell_num, int MAXSHELL){
  /*
    This subroutine will analyse the shell structure of given k space defined by klatt[3][3].
    It will return
    int *M_s;               b vectors number in each shell. 
    M_s[0] store the vector b number for first nearest neighbor  shell,
    M_s[1] store the vector b number for second nearest neighbor shell,
    and so on.
    int **bvector;      Store the b vectors in an ascending shell radius 
    b[vector_indx][xyz]
    The first M_s[0] elements are b vectors from first shell;
    The next M_s[1] elements are those from 2nd nearest-neighbor shell,
    and so on.  
    int *shell_num;     The total number of shells determined in this subroutine.
  */
  int i1,i2,i3;/* combination in 3 k-basis vectors */                 
  int current_shell; /* current_shell number */
  double *distance, dx,dy,dz, *wb;
  int kindx, *ordering,i,j,k, tot_kpt, tot_shell, tot_vectors;
  int **combination, **ordered_com;
  char c;
    
  current_shell=0;
  kindx=0;
    
  combination=(int**)malloc(sizeof(int*)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  ordered_com=(int**)malloc(sizeof(int*)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  distance=(double*)malloc(sizeof(double)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  ordering=(int*)malloc(sizeof(int)*(8*MAXSHELL*MAXSHELL*MAXSHELL));
  for(i1=0;i1<(8*MAXSHELL*MAXSHELL*MAXSHELL);i1++){
    combination[i1]=(int*)malloc(sizeof(int)*3);
    ordered_com[i1]=(int*)malloc(sizeof(int)*3);
    distance[i1]=0.0;
  }
    
  for(i1=-MAXSHELL+1;i1<MAXSHELL;i1++){
    for(i2=-MAXSHELL+1;i2<MAXSHELL;i2++){
      for(i3=-MAXSHELL+1;i3<MAXSHELL;i3++){
	/*generate one k point */
        /* Its distance to Zero point. Here Zero point is also shifted by shift[3].
           And the k lattice basis is klatt, which has been divided by knum_i,knum_j and knum_k.
        */
	dx=(double)i1*klatt[0][0]+(double)i2*klatt[1][0]+(double)i3*klatt[2][0];
	dy=(double)i1*klatt[0][1]+(double)i2*klatt[1][1]+(double)i3*klatt[2][1];
	dz=(double)i1*klatt[0][2]+(double)i2*klatt[1][2]+(double)i3*klatt[2][2];
	distance[kindx]=sqrt(fabs(dx*dx+dy*dy+dz*dz));
	combination[kindx][0]=i1;
	combination[kindx][1]=i2;
	combination[kindx][2]=i3;
	ordering[kindx]=kindx;
	kindx++;
      }/*end i3*/      	
    }/*end i2*/      	
  }/*end i1*/      	
  /* Odering these k points by the ascending order of distance and classify them into shells */  
  tot_kpt=kindx;
  if(debug6==1){
    printf("total kpoint number is %2d\n",tot_kpt);
    printf("before ordering, distance is\n");
    for(kindx=0;kindx<tot_kpt;kindx++){
      i1=combination[kindx][0];
      i2=combination[kindx][1];
      i3=combination[kindx][2];
      dx=(double)i1*klatt[0][0]+(double)i2*klatt[1][0]+(double)i3*klatt[2][0];
      dy=(double)i1*klatt[0][1]+(double)i2*klatt[1][1]+(double)i3*klatt[2][1];
      dz=(double)i1*klatt[0][2]+(double)i2*klatt[1][2]+(double)i3*klatt[2][2];
      printf("(%2d,%2d,%2d)(%10.5f%10.5f%10.5f)%10.5f\n",combination[kindx][0],combination[kindx][1],combination[kindx][2],dx/BohrR_Wannier,dy/BohrR_Wannier,dz/BohrR_Wannier,distance[kindx]/BohrR_Wannier);
    }
  }
  Ascend_Ordering(distance, ordering, tot_kpt);
	  
  for(kindx=0;kindx<tot_kpt;kindx++){
    ordered_com[kindx][0]=combination[ordering[kindx]][0];
    ordered_com[kindx][1]=combination[ordering[kindx]][1];
    ordered_com[kindx][2]=combination[ordering[kindx]][2];
  }
  if(debug6==1){
    printf("After ordering, distance is\n");
    for(kindx=0;kindx<tot_kpt;kindx++){
      i1=ordered_com[kindx][0];
      i2=ordered_com[kindx][1];
      i3=ordered_com[kindx][2];
      dx=(double)i1*klatt[0][0]+(double)i2*klatt[1][0]+(double)i3*klatt[2][0];
      dy=(double)i1*klatt[0][1]+(double)i2*klatt[1][1]+(double)i3*klatt[2][1];
      dz=(double)i1*klatt[0][2]+(double)i2*klatt[1][2]+(double)i3*klatt[2][2];
      printf("(%2d,%2d,%2d) %10.5f%10.5f%10.5f %10.5f\n",ordered_com[kindx][0],ordered_com[kindx][1],ordered_com[kindx][2],dx/BohrR_Wannier,dy/BohrR_Wannier,dz/BohrR_Wannier,distance[kindx]/BohrR_Wannier);
    }
    /*    	c=getchar(); */
  }
    
  /* Now, classifying them into shells. Be careful, the first shell determined here 
     is k point itself, with radius=0.0. */
  i=0;dx=distance[0]; /*mark the first shell */
  current_shell=0;
  for(kindx=0;kindx<tot_kpt;kindx++){/*find the total shell number */
    if(fabs(dx-distance[kindx])>smallvalue){ /* Come to the next shell */
      ordering[current_shell]=i; /*Memorize the number of b vectors in last shell */
      i=1; /* re-count the b vectors in next shell */
      current_shell++; /* next shell index */
      dx=distance[kindx]; /* next shell radius */
    }else{/* still in the same shell */
      i++;/*count the number of b vectors in this shell */
    }
  }
  if(debug6==1){
    printf("totally, there are %2d shells.\n",current_shell-1);
    for(i=1;i<current_shell;i++){
      printf("%2d shell has %3d k points.\n",i,ordering[i]);
    }
  }
  if((current_shell-1)>MAXSHELL){
    tot_shell=MAXSHELL;/*some of the last shell maynot be fully included, discard them */
  }else{
    /*    	printf("Please increase the MAXSHELL parameter. Presently it is %3d\n",MAXSHELL); */
    tot_shell=0;
  }

  /* Now we can asign the values for M_s and bvector */
  tot_vectors=0;
  for(i=1;i<=tot_shell;i++){/* Get the total number of b vectors */
    M_s[i-1]=ordering[i];
    tot_vectors=tot_vectors+M_s[i-1];
  }
   
  for(i=0;i<tot_vectors;i++){
    for(j=0;j<3;j++){
      bvector[i][j]=ordered_com[i+1][j];
    }
  }
  if(debug6==1){
    kindx=0;
    for(i=0;i<tot_shell;i++){
      printf("In shell %2d, there are %2d b vectors.\n",i+1,M_s[i]);
      for(j=0;j<M_s[i];j++){
	printf("(%2d,%2d,%2d) %10.5f\n",bvector[kindx][0],bvector[kindx][1],bvector[kindx][2],distance[kindx+1]);
	kindx++;
      }
    }
  }
  *shell_num=tot_shell;
  for(i1=0;i1<(8*MAXSHELL*MAXSHELL*MAXSHELL);i1++){
    free(ordered_com[i1]);
    free(combination[i1]);
  }
  free(ordering);
  free(distance);
  free(ordered_com);
  free(combination);
}/*End of Shell_Structure subroutine*/
                         
                         
#pragma optimization_level 1
int Cal_Weight_of_Shell(double **klatt, int *M_s, int **bvector, int *num_shell, 
			double *wb, int *Reject_Shell, int *searched_shell){
  /*
    This subroutine will find the weight of shells for given k space defined by klatt[3][3].
    The weight and b vectors are defined in Eqn. 26 in arXiV:0708.0650.
    INPUT
    int *M_s;        b vectors number in each shell. 
    M_s[0] store the vector b number for first nearest neighbor  shell,
    M_s[1] store the vector b number for second nearest neighbor shell,
    and so on

    int **bvector;   b vectors found in each shell. Shells are ordered with ascending shell radius.
    int shell_num;   total shell number.                 
                     
    Return
    double *wb;      weight obtained for shell_num. wb[shell_num].               
    find_w           0 -> not found; 1->founded
  */

  int find_w; /* whether w is found or not? To control the while loop */    
  
  double *qvector; /* qvector is a vector of length six. Corresponding to the combination 
                      of Cartesian indices of two b vectors.
		      qvector[0] --> xx
		      qvector[1] --> yx
		      qvector[2] --> zx
		      qvector[3] --> yy 
		      qvector[4] --> zy
		      qvector[5] --> zz 
                   */      
  double **Amatrix; /* Amatrix[6][shell_num] is the A matrix. 
                       First dimension correspondes to qvector[6];
                       Second correspondes to number of shells. 
                    */
  double **copy_Amatrix;                  
  double **Umatrix; /* a 6x6 matrix for left singular vector of Amatrix[6][shell_num] */
  double **Dmatrix; /* The same dimention as that of Amatrix, 6 x (Shell number).
                       It first min(6, Shell number) x min(6, Shell number) are diagonal
                       with singular value on the diagonal, all other values are zero
                    */                    
  double *dsing; /* store the singular values, which are the diagonal part of Dmatrix. In our
                    program, we need to check these value to see whether one or more of the
                    singular values be close to zero. If it is, reject the new shell and turn
                    to the next one untill Eqn.26 is satisfied. */
  double **VTmatrix; /* a (Shell number)x(Shell number) matrix for right singular vector */
  char jobu, jobvt;
  
  int info;
  /*The following are parameters temporarily used for calculation. */  
  double *da, *du, *dvt;
   
  double sum;
  int i,j,k, reject_indx,flag;
  int current_shell, bvindx, shellindx, startbv, endbv, realshellindx;
  int shell_size, old_shell_size; /* This array stores the possible linear dependent shells. 0, no rejection; 1-rejected  */
  double bx,by,bz,*work;
  char c;
  int shell_num, ROW, COL, lda, lwork;
  
  shell_num=*num_shell;

  qvector=(double*)malloc(sizeof(double)*6);
  for(i=0; i<6; i++){
    qvector[i]=0.0;
  }
  qvector[0]=1.0;/*xx*/
  qvector[3]=1.0;/*yy*/
  qvector[5]=1.0;/*zz*/
  for(i=0;i<shell_num;i++){
    Reject_Shell[i]=0;
  }
  jobu='A';/* all the right singular vectors should be calculated and stored in Umatrix */
  jobvt='A';/* all the left singular vectors should be calculated and stored in Umatrix */
  
  find_w=0;
  current_shell=1;  
  shell_size=1;
  
  if(debug6==1){
    printf("Shell structure:\n");
    for(shellindx=0;shellindx<shell_num;shellindx++){
      printf("In shell %2d, there are %2d vectors.\n",shellindx+1,M_s[shellindx]);
    }
  }
  while(find_w==0&&current_shell<=shell_num){
    old_shell_size=shell_size;
    if(debug6==1){
      /*printf("Start shell_size=%2d, old_shell_size =%2d\n",shell_size,old_shell_size);*/
    }
    Amatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      Amatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    copy_Amatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      copy_Amatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    Dmatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      Dmatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    Umatrix=(double**)malloc(sizeof(double*)*6);
    for(i=0;i<6;i++){
      Umatrix[i]=(double*)malloc(sizeof(double)*6);
    }
    
    VTmatrix=(double**)malloc(sizeof(double*)*shell_size);
    for(i=0;i<shell_size;i++){
      VTmatrix[i]=(double*)malloc(sizeof(double)*shell_size);
    }
    
    if(shell_size>6){
      dsing=(double*)malloc(sizeof(double)*6);
      for(i=0;i<6;i++){
	dsing[i]=0.0;
      }
    }else{
      dsing=(double*)malloc(sizeof(double)*shell_size);
      for(i=0;i<shell_size;i++){
	dsing[i]=0.0;
      }
    }
    
    for(i=0;i<shell_size;i++){
      wb[i]=0.0;
    }
    realshellindx=-1;
    for(shellindx=0;shellindx<current_shell;shellindx++){/* initialize A matrix*/
      if(Reject_Shell[shellindx]==1){
	printf("Shell %d is rejected.\n",shellindx+1);
	continue; /*Ignore this shell if it is linear dependent with existing shells */
      }
      /*The real index for effective shells in Amatrix. Since some shellindx will be discarded */
      realshellindx++; 
      if(debug6==1){
	/*printf("shellindx=%2d and realshellindx=%2d\n",shellindx,realshellindx);*/
      }
      for(i=0;i<6;i++){
	Amatrix[i][realshellindx]=0.0;
      }
      startbv=0;
      if(shellindx==0){
	startbv=0;
      }else{
	for(i=0;i<shellindx;i++){/* find the start index of b vectors in this shell */
	  startbv=startbv+M_s[i];
	}
      }
      if(debug6==1){
	/*printf("startbv=%2d, and M_s[%2d] is %2d. endbv is %2d.\n",startbv,shellindx,M_s[shellindx],M_s[shellindx]+startbv);*/
	printf("In shell %2d, there are %2d b vectors:\n",shellindx+1,M_s[shellindx]);
      }
      for(bvindx=startbv;bvindx<M_s[shellindx]+startbv;bvindx++){
	/* convert the b vectors into Cartesian coordinates */
	bx=(double)bvector[bvindx][0]*klatt[0][0]+(double)bvector[bvindx][1]*klatt[1][0]+(double)bvector[bvindx][2]*klatt[2][0];
	by=(double)bvector[bvindx][0]*klatt[0][1]+(double)bvector[bvindx][1]*klatt[1][1]+(double)bvector[bvindx][2]*klatt[2][1];
	bz=(double)bvector[bvindx][0]*klatt[0][2]+(double)bvector[bvindx][1]*klatt[1][2]+(double)bvector[bvindx][2]*klatt[2][2];
	Amatrix[0][realshellindx]=Amatrix[0][realshellindx]+bx*bx;/*xx*/
	Amatrix[1][realshellindx]=Amatrix[1][realshellindx]+by*bx;/*yx*/
	Amatrix[2][realshellindx]=Amatrix[2][realshellindx]+bz*bx;/*zx*/
	Amatrix[3][realshellindx]=Amatrix[3][realshellindx]+by*by;/*yy*/
	Amatrix[4][realshellindx]=Amatrix[4][realshellindx]+bz*by;/*zy*/
	Amatrix[5][realshellindx]=Amatrix[5][realshellindx]+bz*bz;/*zz*/
	if(debug6==1){
	  /*printf("b*b=%10.5f\n",bx*bx+by*by+bz*bz);*/
	  printf("bvector%2d (%2d,%2d,%2d) Cart(%10.5f,%10.5f,%10.5f) |b|=%10.5f\n",bvindx,bvector[bvindx][0],bvector[bvindx][1],bvector[bvindx][2],bx,by,bz,sqrt(fabs(bx*bx+by*by+bz*bz)));
	}
      }/*all the b vectors within each shell*/
      copy_Amatrix[0][realshellindx]=Amatrix[0][realshellindx];
      copy_Amatrix[1][realshellindx]=Amatrix[1][realshellindx];
      copy_Amatrix[2][realshellindx]=Amatrix[2][realshellindx];
      copy_Amatrix[3][realshellindx]=Amatrix[3][realshellindx];
      copy_Amatrix[4][realshellindx]=Amatrix[4][realshellindx];
      copy_Amatrix[5][realshellindx]=Amatrix[5][realshellindx];
    }/*Sum within each shell. A matrix is OK now*/
    if(debug6==1){
      printf("A matrix are:\n");
      for(i=0;i<6;i++){
	printf("%2d ",i+1);
	for(j=0;j<realshellindx+1;j++){
	  printf("%10.5f ",Amatrix[i][j]);
	}
	printf("\n");
      }
    }
    /*Amatrix should be factorised using a singular value decomposition to get U, D, VT matrix */
    /* To use dgesvd, we need to do some conversion of arrays */
    da=(double*)malloc(sizeof(double)*6*shell_size);
    du=(double*)malloc(sizeof(double)*6*6);
    dvt=(double*)malloc(sizeof(double)*shell_size*shell_size);
    lwork=shell_size*shell_size;
    if(6>shell_size){
      lwork=6*6;
    }
    work=(double*)malloc(sizeof(double)*lwork);
    /* convert from 2-D arrays to 1-D arrays */
    k=0;
    for(j=0;j<shell_size;j++){
      for(i=0;i<6;i++){
	da[k]=Amatrix[i][j];
	k++;
      }
    }
    k=0;
    for(j=0;j<6;j++){
      for(i=0;i<6;i++){
	du[k]=0.0;
	k++;
      }
    }
    k=0;
    for(j=0;j<shell_size;j++){
      for(i=0;i<shell_size;i++){
	dvt[k]=0.0;
	k++;
      }
    }
    /* Singular Value Decompositon 
       Ozaki-san, would you please modify this calling of dgesvd?*/
    /*    dgesvd(jobu, jobvt, 6, shell_size, da, 6, dsing, du, 6, dvt, shell_size, &info); */
    ROW=6;
    COL=shell_size;  	
    lda=6;
    F77_NAME(dgesvd,DGESVD)(&jobu, &jobvt, &ROW, &COL, da, &lda, dsing, du, &lda, dvt, &COL, work, &lwork, &info);

    if(info==0){
      /*printf("Factorising OK\n");*/
    }else if(info<0){
      printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
      printf("! In subroutine Cal_Weight_of_Shell:       !\n");
      printf("! When calling dgesvd, the %2dth argument  !\n",-info);
      printf("! had an illegal value.                    !\n");
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      return 0;
    }else{
      printf("!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!\n");
      printf("! In subroutine Cal_Weight_of_Shell:       !\n");
      printf("! Calling Dgesvd failed in convergence.    !\n");
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      return 0;
    }
    if(debug6==1){
      printf("Singular value are:\n");
      if(shell_size<6){
	for(i=0;i<shell_size;i++){
	  printf("%10.5f\n",dsing[i]);
	}
      }else{
	for(i=0;i<6;i++){
	  printf("%10.5f\n",dsing[i]);
	}
      }
    }
    /* checking the Singular value. If one or more of them are close or equal to 0, 
       reject current shell and go to next shell.*/
    if(shell_size<6){
      for(i=0;i<shell_size;i++){
	if(fabs(dsing[i])<smallvalue){
	  Reject_Shell[current_shell-1]=1;
          if(debug6==1){
	    printf("current shell %d is to be rejected.\n",current_shell);
          }
	  break;
	}
      }
    }else{
      for(i=0;i<6;i++){
	if(fabs(dsing[i])<smallvalue){
	  Reject_Shell[current_shell-1]=1;
	  if(debug6==1){
	    printf("current shell %d is to be rejected.\n",current_shell);
          }
	  break;
	}
      }
    }
    if(Reject_Shell[current_shell-1]==0){/* if this shell is needed */
      /* asign values to U, VT and D and try to calculate w for these shells*/	
      for(j=0;j<shell_size;j++){
	for(i=0;i<6;i++){
	  Dmatrix[i][j]=0.0;
	}
      }
      if(shell_size<6){
	for(i=0;i<shell_size;i++){
	  Dmatrix[i][i]=1.0/dsing[i];
	}
      }else{
	for(i=0;i<6;i++){
	  Dmatrix[i][i]=1.0/dsing[i];
	}
      }
      k=0;
      for(j=0;j<6;j++){
	for(i=0;i<6;i++){
	  Umatrix[i][j]=du[k];
	  k++;
	}
      }
      if(debug6==1){
	printf("U matrix is:\n");
	for(i=0;i<6;i++){
	  printf("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n",Umatrix[i][0],Umatrix[i][1],Umatrix[i][2],Umatrix[i][3],Umatrix[i][4],Umatrix[i][5]);
	}
      }
      k=0;
      for(j=0;j<shell_size;j++){
	for(i=0;i<shell_size;i++){
	  VTmatrix[i][j]=dvt[k];
	  k++;
	}
      }
      if(debug6==1){
	printf("VT matrix is:\n");
	for(i=0;i<shell_size;i++){
	  for(j=0;j<shell_size;j++){
	    printf("%10.5f ",VTmatrix[i][j]);
	  }
	  printf("\n");
	}
      }
      /*Now we have U, VT and D.*/
      /*Calculate VxD^{-1} (NxN)x(NxM) and store the result into Amatrix*/
      for(i=0;i<shell_size;i++){
	for(j=0;j<6;j++){
	  sum=0.0;
	  for(k=0;k<shell_size;k++){
	    sum=sum+VTmatrix[k][i]*Dmatrix[j][k];
	  }
	  Amatrix[j][i]=sum;
	}
      }
      if(debug6==1){/* out put VxD^{-1} */
	printf("VxD^{-1} (shell_sizexshell_size)x(shell_sizex6) is:\n");
	for(i=0;i<shell_size;i++){
	  for(j=0;j<6;j++){
	    printf("%10.5f ",Amatrix[j][i]);
	  }
	  printf("\n");
	}
      }
      /* Calculate VxD^{-1}*UT (NxM)x(MxM) and store the result into Dmatrix */
      for(i=0;i<shell_size;i++){
	for(j=0;j<6;j++){
	  sum=0.0;
	  for(k=0;k<6;k++){
	    sum=sum+Amatrix[k][i]*Umatrix[j][k];
	  }
	  Dmatrix[j][i]=sum;
	}
      }
      if(debug6==1){/* out put VxD^{-1}*UT */
	printf("VxD^{-1}*UT (shell_sizex6)x(6x6) is:\n");
	for(i=0;i<shell_size;i++){
	  for(j=0;j<6;j++){
	    printf("%10.5f ",Dmatrix[j][i]);
	  }
	  printf("\n");
	}
      }
      /* Calculate VxD^{-1}*UT*q (NxM)x(Mx1) and store the result into wb */
      for(i=0;i<shell_size;i++){
	sum=0.0;
	for(k=0;k<6;k++){
	  sum=sum+Dmatrix[k][i]*qvector[k];
	}
	wb[i]=sum;
	if(debug6==1){
	  printf("wb %2d is %10.5f\n",i+1,wb[i]);
	}
      }

      /* After getting wb, Eqn. 26 should be checked to see whether these wb is ok */
      flag=0;
      for(j=0;j<6;j++){
	sum=0.0;
	for(i=0;i<shell_size;i++){
	  sum=sum+wb[i]*copy_Amatrix[j][i];
	  /*printf("amatrix[%2d][%2d]=%10.5f wb[%2d]=%10.5f",j,i,copy_Amatrix[j][i],i,wb[i]);*/
      	}
      	if(fabs(sum-qvector[j])>smallvalue){
	  flag=1;
	  if(debug6==1){
	    /*printf("sum=%10.5f, qvector=%10.5f ",sum,qvector[j]);*/
	    printf("Sorry, this weight is not acceptable.\n");
	  }
      	}
      }
      if(flag==0){/* wb is founded. */
	find_w=1;/*mark it and stop searching */
	*num_shell=shell_size;/* The real number of shells used */
	*searched_shell=current_shell;
	if(debug6==1){
	  printf("Haha! I find them.\n");
	  for(i=0;i<shell_size;i++){
	    printf("wb[%2d]=%10.5f\n",i+1,wb[i]);
	  }
	}
      }else{/* wb not found */
	shell_size++; /* No wb was found, adding one more shell */
      }
    }/* if current shell is not rejected, calculating w */
    if(debug6==1){
      /*printf("End shell_size=%2d, old_shell_size =%2d\n",shell_size,old_shell_size);fflush(0);*/
    }
    current_shell++; /* Going to the next shell */
    if(debug6==1){
      printf("Current Shell=%2d.\n",current_shell);
    }
    free(work);
    free(dvt);
    free(du);
    free(da);
    free(dsing);
    
    for(i=0;i<old_shell_size;i++){
      free(VTmatrix[i]);
    }
    free(VTmatrix);
    
    for(i=0;i<6;i++){
      free(Umatrix[i]);
    }
    free(Umatrix);
    
    for(i=0;i<6;i++){
      free(Dmatrix[i]);
    }
    free(Dmatrix);
       
    for(i=0;i<6;i++){
      free(copy_Amatrix[i]);
    }
    free(copy_Amatrix);    
    
    for(i=0;i<6;i++){
      free(Amatrix[i]);
    }
    free(Amatrix);    
  }/*While not found*/
  free(qvector);
  return find_w;
}/*Cal_Weight_of_Shell */


#pragma optimization_level 1
static void Ascend_Ordering(double *xyz_value, int *ordering, int tot_atom){
  int i,j,k, tmp_order;
  double tmp_xyz;
	
  for(i=1;i<tot_atom; i++){/* taking one value */
    for(j=i;j>0;j--){ /* compare with all the other lower index value */
      if(xyz_value[j]<xyz_value[j-1]){/* if it is smaller than lower index value, exchange */
	tmp_xyz=xyz_value[j];
	xyz_value[j]=xyz_value[j-1];
	xyz_value[j-1]=tmp_xyz;
	tmp_order = ordering[j];
	ordering[j]=ordering[j-1];
	ordering[j-1]=tmp_order;
      }
    }
  }
  return;
}


#pragma optimization_level 1
void Calc_rT_dot_d(dcomplex *rmnk, dcomplex *dmnk, int n, double *amod){
  int i;
  double sum;
  *amod=0.0;
  sum=0.0;
  for(i=0;i<n;i++){
    sum=sum+(rmnk[i].r*dmnk[i].r+rmnk[i].i*dmnk[i].i);
    /* printf("rT %i %20.12f  %20.12f, d %20.12f  %20.12f rT*d %20.12f\n",i,rmnk[i].r, rmnk[i].i,dmnk[i].r,dmnk[i].i, rmnk[i].r*dmnk[i].r+rmnk[i].i*dmnk[i].i);*/
  }
  *amod=sum;
  /* printf("error mod %10.5f\n",*amod); */
}/* end of Calc_rT_dot_d */ 

/*void Gradient_at_next_x(dcomplex ****Gmnk, double wbtot, double alpha, dcomplex ****Uk, dcomplex *****m, int spinsize, int kpt_num, int WANNUM, double **kg, double **bvector, double **frac_bv, double *wb, int tot_bvector, int *M_s, int shell_num, dcomplex **rimnk,*/
/*double *omega_I, double *omega_OD, double *omega_D, double ***wann_center, double **wann_r2){ */


#pragma optimization_level 1
void Gradient_at_next_x(dcomplex ***Gmnk, double wbtot, double alpha, dcomplex ***Ukmatrix, dcomplex ****Mmnkb_zero, int kpt_num, int WANNUM, double **kg, double **bvector, double **frac_bv, double *wb, int tot_bvector, int **kplusb, dcomplex *rimnk, double *omega_I, double *omega_OD, double *omega_D){
  /* double ***wann_center, double **wann_r2){ */
  /* for yita_prev. Gmnk and Ukmatrix Mmnkb will not be changed during this subroutine */
  /*   taking one step alpha*d */
  dcomplex ***g, ***Uk, ****m; 
  /*  double *oi, *ood, *od, ***wr, **wr2; */
  dcomplex ***deltaW, ****m_zero; 
  int i,j, k, spin, bindx, nindx;
  double **wann_center, *wann_r2, *wk;

  m = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);
  for(k=0;k<kpt_num;k++){
    m[k] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      m[k][bindx] = (dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
      for (i=0; i<WANNUM; i++){
	m[k][bindx][i] = (dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
	for(j=0; j<WANNUM; j++){
	  m[k][bindx][i][j].r=Mmnkb_zero[k][bindx][i][j].r;
	  m[k][bindx][i][j].i=Mmnkb_zero[k][bindx][i][j].i;
	}
      }
    }
  }

  m_zero = (dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);
  for(k=0;k<kpt_num;k++){
    m_zero[k] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      m_zero[k][bindx] = (dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
      for (i=0; i<WANNUM; i++){
	m_zero[k][bindx][i] = (dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
	for(j=0; j<WANNUM; j++){ 
	  m_zero[k][bindx][i][j].r=m[k][bindx][i][j].r; 
	  m_zero[k][bindx][i][j].i=m[k][bindx][i][j].i;
	}
      } 
    }
  }
  /* Gradients of spread function */

  g = (dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    g[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      g[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	g[k][i][j].r=Gmnk[k][i][j].r;
	g[k][i][j].i=Gmnk[k][i][j].i;
      }
    }
  }

  Uk=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Uk[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      Uk[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	Uk[k][i][j].r=Ukmatrix[k][i][j].r;
	Uk[k][i][j].i=Ukmatrix[k][i][j].i;
      }
    }
  }

  wann_center=(double**)malloc(sizeof(double*)*WANNUM);
  for(nindx=0;nindx<WANNUM;nindx++){
    wann_center[nindx]=(double*)malloc(sizeof(double)*3);
  }

  wann_r2=(double*)malloc(sizeof(double)*WANNUM);
  for(nindx=0;nindx<WANNUM;nindx++){
    wann_r2[nindx]=0.0;
  }

  deltaW=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    deltaW[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      deltaW[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	deltaW[k][i][j].r=0.0;
	deltaW[k][i][j].i=0.0;
      }/* j */
    }/* i */
  }/* kpt */

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      for(j=0;j<WANNUM;j++){
	deltaW[k][i][j].r=Gmnk[k][i][j].r/wbtot/4.0*alpha;
	deltaW[k][i][j].i=Gmnk[k][i][j].i/wbtot/4.0*alpha;         
      }/* j */
    }/* i */
  }/* kpt */

  Cal_Ukmatrix(deltaW, Uk, kpt_num, WANNUM);

  /* update Mmnk matrix by M_opt= Uk * M_zero * U(k+b) */
  Updating_Mmnkb(Uk, kpt_num, WANNUM, WANNUM, m, m_zero, kg, frac_bv, tot_bvector, kplusb);

  Wannier_Center(wann_center, wann_r2, m, WANNUM, bvector, wb, kpt_num, tot_bvector);
  
  Cal_Omega(omega_I, omega_OD, omega_D, m, wann_center, WANNUM, bvector, wb, kpt_num, tot_bvector);
  /*  Cal_Omega(omega_I, omega_OD, omega_D, m, wann_center, WANNUM, bvector, */
  /*                  wb, kpt_num, tot_bvector); */
   
  /* Gradients of spread function */
  Cal_Gradient(g, m, wann_center, bvector, wb, WANNUM, kpt_num, tot_bvector, &i);

  bindx=0;
  for(k=0;k<kpt_num;k++){
    for(j=0;j<WANNUM;j++){
      for(i=0;i<WANNUM;i++){
        rimnk[bindx].r=g[k][i][j].r;
        rimnk[bindx].i=g[k][i][j].i;
        bindx++;
      }/* j */
    }/* i */
  }/* kpt */

  /* ========= free ============= */
  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(deltaW[k][i]);
    }/* i */
    free(deltaW[k]);
  }/* kpt */
  free(deltaW);

  free(wann_r2);

  for(nindx=0;nindx<WANNUM;nindx++){
    free(wann_center[nindx]);
  }
  free(wann_center);

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(Uk[k][i]);
    }
    free(Uk[k]);
  }
  free(Uk);

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(g[k][i]);
    }
    free(g[k]);
  }
  free(g);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for (i=0; i<WANNUM; i++){
        free(m_zero[k][bindx][i]);
      }
      free(m_zero[k][bindx]);
    }
    free(m_zero[k]);
  }
  free(m_zero);

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for (i=0; i<WANNUM; i++){
	free(m[k][bindx][i]);
      } 
      free(m[k][bindx]);
    }
    free(m[k]);
  }
  free(m);
}/* end of Gradient_at_next_x */



#pragma optimization_level 1
void Output_WF_Spread(char *mode, 
                      double **wann_center, double *wann_r2, 
                      double omega_I, double omega_OD, double omega_D, int WANNUM)
{
  double Ptx, Pty, Ptz, omega,sumr2;
  int nindx;
  FILE *fp;
  char fname[300];

  /* open fp */

  sprintf(fname,"%s%s.wfinfo",filepath,filename);
  if((fp=fopen(fname,"w"))==NULL){
     printf("Error in opening file for wrinfo: %s\n",fname);
  }

  if (strcasecmp(mode,"std")==0)
    printf("         Center of Wannier Function (Angs)   |   Spread (Angs^2) \n");
  else if (strcasecmp(mode,"file")==0){

    fprintf(fp,"\n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"                   Wannier functions                       \n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"***********************************************************\n\n");

    fprintf(fp,"          Center of Wannier Function (Angs)   |   Spread (Angs^2) \n");
  }

  Ptx = 0.0; 
  Pty = 0.0;
  Ptz = 0.0;
  omega = 0.0;
  
  for(nindx=0;nindx<WANNUM;nindx++){

    Ptx = Ptx + wann_center[nindx][0]*BohrR_Wannier;
    Pty = Pty + wann_center[nindx][1]*BohrR_Wannier;
    Ptz = Ptz + wann_center[nindx][2]*BohrR_Wannier;

    sumr2 = wann_center[nindx][0]*wann_center[nindx][0]
           +wann_center[nindx][1]*wann_center[nindx][1]
           +wann_center[nindx][2]*wann_center[nindx][2];

    sumr2 = (wann_r2[nindx]-sumr2)*BohrR_Wannier*BohrR_Wannier;

    omega = omega + sumr2;

    if (strcasecmp(mode,"std")==0){

      printf("WF %3d (%11.8f,%11.8f,%11.8f) | %12.8f  --->CENT\n",
             nindx+1,
             wann_center[nindx][0]*BohrR_Wannier,
             wann_center[nindx][1]*BohrR_Wannier,
             wann_center[nindx][2]*BohrR_Wannier,
             sumr2);
    }
    else if (strcasecmp(mode,"file")==0){

      fprintf(fp," WF %3d (%11.8f,%11.8f,%11.8f) | %12.8f\n",
             nindx+1,wann_center[nindx][0]*BohrR_Wannier,
             wann_center[nindx][1]*BohrR_Wannier,wann_center[nindx][2]*BohrR_Wannier,sumr2);
    }

  } /* WANNUM */

  if (strcasecmp(mode,"std")==0){
    printf("Total Center (%11.8f,%11.8f,%11.8f) sum_spread %12.8f --->CENT\n",Ptx,Pty,Ptz,omega);
    printf("The component of spread function is:\n");
    printf("Omega_I=%15.12f Omega_D=%15.12f Omega_OD=%15.12f Total_Omega=%20.18f \n",
            omega_I*BohrR_Wannier*BohrR_Wannier, 
            omega_D*BohrR_Wannier*BohrR_Wannier, 
            omega_OD*BohrR_Wannier*BohrR_Wannier,
            (omega_I+omega_OD+omega_D)*BohrR_Wannier*BohrR_Wannier );
  }

  else if (strcasecmp(mode,"file")==0){

    fprintf(fp,"\n");
    fprintf(fp," Total Center (%11.8f,%11.8f,%11.8f)\n",Ptx,Pty,Ptz);
    fprintf(fp," Sum.of.Spreads. %18.15f\n\n",omega);

    fprintf(fp,"The component of spread function is:\n");

    fprintf(fp," Omega I     = %15.12f\n", omega_I*BohrR_Wannier*BohrR_Wannier);
    fprintf(fp," Omega D     = %15.12f\n", omega_D*BohrR_Wannier*BohrR_Wannier);
    fprintf(fp," Omega OD    = %15.12f\n", omega_OD*BohrR_Wannier*BohrR_Wannier);
    fprintf(fp," Total.Omega.= %18.15f\n", (omega_I+omega_OD+omega_D)*BohrR_Wannier*BohrR_Wannier);
  }

  /* close fp */
  fclose(fp);

}/* Output_WF_Spread */ 



#pragma optimization_level 1
void new_Gradient(dcomplex ****m, int kpt_num, int WANNUM, double **bvector, double *wb, int tot_bvector, dcomplex *rimnk, double *omega_I, double *omega_OD, double *omega_D, double **wann_center, double *wann_r2){
  int i,j,k, bindx;
  double *wk;
  dcomplex ***g;         

  g=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    g[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      g[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	g[k][i][j].r=0.0;
	g[k][i][j].i=0.0;
      }
    }
  }

  Wannier_Center(wann_center, wann_r2, m, WANNUM, bvector, wb, kpt_num, tot_bvector);

  Cal_Omega(omega_I, omega_OD, omega_D, m, wann_center, WANNUM, bvector, wb, kpt_num, tot_bvector);

  /* Gradients of spread function */
  Cal_Gradient(g, m, wann_center, bvector, wb, WANNUM, kpt_num, tot_bvector, &i);

  bindx=0;
  for(k=0;k<kpt_num;k++){
    for(j=0;j<WANNUM;j++){
      for(i=0;i<WANNUM;i++){
	rimnk[bindx].r=g[k][i][j].r;
	rimnk[bindx].i=g[k][i][j].i;
	bindx++;
      }/* j */
    }/* i */
  }/* kpt */

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(g[k][i]);
    }   
    free(g[k]);
  }  
  free(g);

}/* new_Gradient */


#pragma optimization_level 1
void Taking_one_step(dcomplex ***Gmnk, double wbtot, double alpha, dcomplex ***Ukmatrix, dcomplex ****Mmnkb, dcomplex ****Mmnkb_zero, int kpt_num, int WANNUM, double **kg, double **frac_bv, int tot_bvector, int **kplusb){

  dcomplex ***deltaW;
  int i,j,k, bindx;

  deltaW=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    deltaW[k]=(dcomplex**)malloc(sizeof(dcomplex*)*WANNUM);
    for(i=0;i<WANNUM;i++){
      deltaW[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*WANNUM);
      for(j=0;j<WANNUM;j++){
	deltaW[k][i][j].r=0.0;
	deltaW[k][i][j].i=0.0;
      }/* j */
    }/* i */
  }/* kpt */

  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      for(j=0;j<WANNUM;j++){
	deltaW[k][i][j].r=Gmnk[k][i][j].r/wbtot/4.0*alpha;
	deltaW[k][i][j].i=Gmnk[k][i][j].i/wbtot/4.0*alpha;
      }/* j */
    }/* i */
  }/* kpt */
  /*
    m_zero = (dcomplex*****)malloc(sizeof(dcomplex****)*kpt_num);
    for(k=0;k<kpt_num;k++){
    m_zero[k] = (dcomplex****)malloc(sizeof(dcomplex***)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
    m_zero[k][bindx] = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize);
    for(spin=0; spin<spinsize; spin++){
    m_zero[k][bindx][spin] = (dcomplex**)malloc(sizeof(dcomplex*)*(WANNUM+1));
    for (i=0; i<WANNUM+1; i++){
    m_zero[k][bindx][spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*(WANNUM+1));
    for(j=0; j<WANNUM+1; j++){
    m_zero[k][bindx][spin][i][j].r=Mmnkb[k][bindx][spin][i][j].r;
    m_zero[k][bindx][spin][i][j].i=Mmnkb[k][bindx][spin][i][j].i;
    }
    }
    }
    }
    }
  */
  Cal_Ukmatrix(deltaW, Ukmatrix, kpt_num, WANNUM);
  /* update Mmnk matrix by M_opt= Uk * M_zero * U(k+b) */
  Updating_Mmnkb(Ukmatrix, kpt_num, WANNUM, WANNUM, Mmnkb, Mmnkb_zero, kg, frac_bv,tot_bvector,kplusb);
  /*
    for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
    for(spin=0; spin<spinsize; spin++){
    for (i=0; i<WANNUM+1; i++){
    free(m_zero[k][bindx][spin][i]);
    } 
    free(m_zero[k][bindx][spin]);
    }
    free(m_zero[k][bindx]);
    }
    free(m_zero[k]);
    }
    free(m_zero);
  */
  for(k=0;k<kpt_num;k++){
    for(i=0;i<WANNUM;i++){
      free(deltaW[k][i]);
    }/* i */
    free(deltaW[k]);
  }/* kpt */
  free(deltaW);
}/* Taking_one_step */




#pragma optimization_level 1
void Disentangling_Bands(dcomplex ****Uk, dcomplex *****Mmnkb_zero, int spinsize, int kpt_num,
                         int band_num, int wan_num, int ***Nk, int ***innerNk, double **kg, 
                         double **bvector, double *wb, int **kplusb, int tot_bvector, 
                         double ***eigen)
{
  int spin, i,j,k,l,m, bindx, nindx,iter, BAND;
  dcomplex **Zmat,***Zmat_prev,***Zmat_i,**Hrot;
  double omega_I, omega_I_prev, delta_OI,dis_conv_tol,dis_alpha_mix; 
  double wbtot, sumr, sumi;
  dcomplex ****Mmnkb, ***Udis;
  int **nbandwin, **nbandfroz, Mk, kpb, kpb_band, lwork, lda, info;
  dcomplex **Utmp;
  char jobz, uplo; 
  dcomplex *work;
  double *rwork;
  dcomplex *za; /*[band_num*band_num]; */
  double *dw; /*[band_num]; */
  int have_frozen, dis_max_iter;
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("Disentangling the attached bands ......\n");fflush(0);
  }

  /* allocation of arrays */

  Zmat_prev = (dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for (i=0; i<kpt_num; i++){
    Zmat_prev[i] = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
    for (j=0; j<band_num; j++){
      Zmat_prev[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
    }
  }

  Zmat_i = (dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for (i=0; i<kpt_num; i++){
    Zmat_i[i] = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
    for (j=0; j<band_num; j++){
      Zmat_i[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
    }
  }

  Hrot = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
  for (i=0; i<wan_num; i++){
    Hrot[i] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
  }

  work = (dcomplex*)malloc(sizeof(dcomplex)*3*band_num);
  rwork = (double*)malloc(sizeof(double)*3*band_num);

  /* start calc. */

  have_frozen=0;
  lwork=3*band_num;

  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      if((innerNk[spin][k][1]-innerNk[spin][k][0])>0 &&(innerNk[spin][k][1]-innerNk[spin][k][0])<=wan_num){
	have_frozen=1;
      }
    }
  }
  if(have_frozen){
    Udis_projector_frozen(Uk,Nk,innerNk,kpt_num,spinsize,band_num,wan_num);
  }

  dis_max_iter=Wannier_Dis_SCF_Max_Steps;
  dis_conv_tol=Wannier_Dis_Conv_Criterion;
  dis_alpha_mix=Wannier_Dis_Mixing_Para; 

  jobz='V';
  uplo='U';

  BAND=band_num;

  nbandfroz=(int**)malloc(sizeof(int*)*kpt_num);
  for(k=0;k<kpt_num;k++){
    nbandfroz[k]=(int*)malloc(sizeof(int)*2);
  }
 
  nbandwin=(int**)malloc(sizeof(int*)*kpt_num);
  for(k=0;k<kpt_num;k++){
    nbandwin[k]=(int*)malloc(sizeof(int)*2);
  }

  Utmp=(dcomplex**)malloc(sizeof(dcomplex*)*BAND);
  for(i=0;i<BAND;i++){
    Utmp[i]=(dcomplex*)malloc(sizeof(dcomplex)*BAND);
  }

  Mmnkb=(dcomplex****)malloc(sizeof(dcomplex***)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Mmnkb[k] = (dcomplex***)malloc(sizeof(dcomplex**)*tot_bvector);
    for(bindx=0;bindx<tot_bvector;bindx++){
      Mmnkb[k][bindx] = (dcomplex**)malloc(sizeof(dcomplex*)*BAND);
      for (i=0; i<BAND; i++){
        Mmnkb[k][bindx][i] = (dcomplex*)malloc(sizeof(dcomplex)*BAND);
        for (j=0; j<BAND; j++){
          Mmnkb[k][bindx][i][j].r=0.0;
          Mmnkb[k][bindx][i][j].i=0.0;
        }
      } /* i, j band num */
    }/* b vector */
  }
  Udis=(dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
  for(k=0;k<kpt_num;k++){
    Udis[k]=(dcomplex**)malloc(sizeof(dcomplex*)*BAND);
    for(i=0;i<BAND;i++){
      Udis[k][i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      for(j=0;j<wan_num;j++){
        Udis[k][i][j].r=0.0;
        Udis[k][i][j].i=0.0;
      }/* wannier */
    }/* Windown */
  } /*kpt */

  wbtot=0.0;
  for(i=0;i<tot_bvector;i++){
    wbtot=wbtot+wb[i];
  }  

  for(spin=0;spin<spinsize;spin++){
    if (myid==Host_ID){
      printf("Disentangling spin component %i.\n",spin);
    }
    /* firstly, seperate spin */
    for(k=0;k<kpt_num;k++){
      for(bindx=0;bindx<tot_bvector;bindx++){
	for (i=0; i<BAND; i++){
	  for (j=0; j<BAND; j++){
	    Mmnkb[k][bindx][i][j].r=Mmnkb_zero[k][bindx][spin][i+1][j+1].r;
	    Mmnkb[k][bindx][i][j].i=Mmnkb_zero[k][bindx][spin][i+1][j+1].i;
	  } 
	} /* i, j band num */
      }/* b vector */
      nbandwin[k][0]=Nk[spin][k][0];
      nbandwin[k][1]=Nk[spin][k][1];
      nbandfroz[k][0]=innerNk[spin][k][0];
      nbandfroz[k][1]=innerNk[spin][k][1];
      if(debugdis){
	printf("kpt %i, bottom=%i, top=%i\n",k,nbandwin[k][0],nbandwin[k][1]);fflush(0);
      }
    }/* kpt*/
    for(k=0;k<kpt_num;k++){
      for (i=0; i<BAND; i++){
	for (j=0; j<wan_num; j++){
	  Udis[k][i][j].r=Uk[spin][k][i][j].r;
	  Udis[k][i][j].i=Uk[spin][k][i][j].i;
	}
      } /* i, j band num */
    }/* kpt*/

    iter=0;
    omega_I=0.0;
    omega_I_prev=0.0;
    delta_OI=1.0;
    if (myid==Host_ID){
       printf("****************************************************\n");
       printf("  Iteration(s) to minimize OMEGA_I ......\n");
       printf("****************************************************\n");
       if(BohrR_Wannier == 1.0){ 
         printf("|  Iter  | Omega_I (Bohr^2) | Delta_I (Bohr^2) |  ---> DISE\n");
       }else{
         printf("|  Iter  | Omega_I (Angs^2) | Delta_I (Angs^2) |  ---> DISE\n");
       } 
       fflush(0);
    }
    /* start iteration */

    while(iter<dis_max_iter && fabs(delta_OI)>dis_conv_tol){

      omega_I_prev=omega_I;
      omega_I=0.0;
      /* omega_I contributed by frozen states
	 Eq. (12) of paper of SMV */
      for(k=0;k<kpt_num;k++){
	band_num=nbandwin[k][1]-nbandwin[k][0];
	Mk=nbandfroz[k][1]-nbandfroz[k][0];
	if(Mk>0){ 
	  for(bindx=0;bindx<tot_bvector;bindx++){
	    kpb=kplusb[k][bindx];
	    kpb_band=nbandwin[kpb][1]-nbandwin[kpb][0];
	    /* U(k)^dagger * M(k,b) (Mk1 x Nk1) x (Nk1 x Nk2) */
	    for(i=0;i<Mk;i++){  /* frozen states are putted in the lowest coloumns */
	      for(j=0;j<kpb_band;j++){
		sumr=0.0; sumi=0.0;
		for(l=0;l<band_num;l++){
		  sumr=sumr+Udis[k][l][i].r*Mmnkb[k][bindx][l][j].r+Udis[k][l][i].i*Mmnkb[k][bindx][l][j].i;
		  sumi=sumi+Udis[k][l][i].r*Mmnkb[k][bindx][l][j].i-Udis[k][l][i].i*Mmnkb[k][bindx][l][j].r;
		}
		Utmp[i][j].r=sumr;
		Utmp[i][j].i=sumi;
	      } 
	    }
	    /* U(k)^dagger * M(k,b) * U(k+b)  ( Mk1 x Nk2 ) x (Nk2 x WAN_NUM) */
	    for(i=0;i<Mk;i++){
	      for(j=0;j<wan_num;j++){
		sumr=0.0; sumi=0.0;
		for(l=0;l<kpb_band;l++){
		  sumr=sumr+Utmp[i][l].r*Udis[kpb][l][j].r-Utmp[i][l].i*Udis[kpb][l][j].i;
		  sumi=sumi+Utmp[i][l].r*Udis[kpb][l][j].i+Utmp[i][l].i*Udis[kpb][l][j].r;
		}
		Hrot[i][j].r=sumr;
		Hrot[i][j].i=sumi;
	      } 
	    }
	    /* Now Hrot contains the rotated overlap matrix (Mk1 x wan_num) */
	    sumr=0.0;
	    for(i=0;i<Mk;i++){
	      for(j=0;j<wan_num;j++){
		sumr=sumr+Hrot[i][j].r*Hrot[i][j].r+Hrot[i][j].i*Hrot[i][j].i;
	      }
	    }   
	    omega_I=omega_I+wb[bindx]*sumr;
	  }/*  bindx */ 
	}/* if there is frozen state */  
      }/* kpt */  
      for(k=0;k<kpt_num;k++){
	band_num=nbandwin[k][1]-nbandwin[k][0];
	Mk=nbandfroz[k][1]-nbandfroz[k][0];
	if(band_num<wan_num||Mk>wan_num){
         if (myid==Host_ID){
            printf("*********************** Error **********************\n");	    
	    printf("* At kpt %i, there are %i bands and %i of them      \n",k,band_num,Mk); 
            printf("* are frozen. While the WF number is %i.            \n",wan_num);
            printf("*********************** Error **********************\n");	    
         }
         MPI_Finalize();
	 exit(0);
	}
	if(Mk<wan_num){
	  Getting_Zmatrix(Utmp,k,wb,tot_bvector,Mmnkb,Udis,kplusb,band_num,wan_num,Mk,nbandwin,nbandfroz);
	  for(i=0;i<band_num-Mk;i++){
	    for(j=0;j<band_num-Mk;j++){
	      Zmat_i[k][i][j].r=Utmp[i][j].r;
	      Zmat_i[k][i][j].i=Utmp[i][j].i;
	    }
	  }
	}else{
	  continue; /* ignore those k points with Mk equal to wan_num */
	}
      }/* kpt*/

      for(k=0;k<kpt_num;k++){
	band_num=nbandwin[k][1]-nbandwin[k][0];
	Mk=nbandfroz[k][1]-nbandfroz[k][0];
	if(debugdis){
	  printf("kpt %i has %i bands and %i are frozen:\n",k,band_num,Mk);fflush(0);
	}
	if(band_num<wan_num||Mk>wan_num){
          if (myid==Host_ID){
            printf("*********************** Error **********************\n");
            printf("* At kpt %i, there are %i bands and %i of them      \n",k,band_num,Mk);
            printf("* are frozen. While the WF number is %i.            \n",wan_num);
            printf("*********************** Error **********************\n");
          }
          MPI_Finalize();
	  exit(0);
	}
	if(Mk==wan_num){
	  /* all the bands are frozen, which contribution is considered when calculating the frozen part*/
	  continue; 
	}
	Zmat=(dcomplex**)malloc(sizeof(dcomplex*)*(band_num-Mk));
	for(i=0;i<band_num-Mk;i++){
	  Zmat[i]=(dcomplex*)malloc(sizeof(dcomplex)*(band_num-Mk));
	  for(j=0;j<band_num-Mk;j++){
	    Zmat[i][j].r=0.0;
	    Zmat[i][j].i=0.0;
	  }
	}
	za=(dcomplex*)malloc(sizeof(dcomplex)*(band_num-Mk)*(band_num-Mk));
	dw=(double*)malloc(sizeof(double)*(band_num-Mk));
	if(iter>0){
	  for(i=0;i<band_num-Mk;i++){
	    for(j=0;j<band_num-Mk;j++){
	      Zmat[i][j].r=dis_alpha_mix*Zmat_i[k][i][j].r+(1.0-dis_alpha_mix)*Zmat_prev[k][i][j].r;
	      Zmat[i][j].i=dis_alpha_mix*Zmat_i[k][i][j].i+(1.0-dis_alpha_mix)*Zmat_prev[k][i][j].i;
	    }
	  }
	}/* iter>0 mixing */
	else{
	  for(i=0;i<band_num-Mk;i++){
	    for(j=0;j<band_num-Mk;j++){
	      Zmat[i][j].r=Zmat_i[k][i][j].r;
	      Zmat[i][j].i=Zmat_i[k][i][j].i;
	    }
	  }
	}/* NO mixing for first iteration */


	/* diagonize Zmat and select wan_num-Mk largest eigenvalue and corresponding eigenvectors
	   out of band_num-Mk  */
	bindx=0;
	for(i=0;i<band_num-Mk;i++){
	  for(j=0;j<band_num-Mk;j++){
	    za[bindx].r=Zmat[j][i].r;
	    za[bindx].i=Zmat[j][i].i;
	    bindx++;
	  }
	}
	/*       zheev(jobz, uplo, band_num-Mk, za, band_num-Mk, dw, &info); */
	/* clapack  Macbook Air */
	/*   int zheev_(char *jobz, char *uplo, integer *n, doublecomplex
	 *za, integer *lda, doublereal *dw, doublecomplex *work, integer *lwork,
	 doublereal *rwork, integer *info)

	 zheev_(jobz, uplo, wan_num, za, wan_num, dw, work, 2*wan_num-1, rwork, 3*wan_num-2, &info);
	*/
	lda=band_num-Mk;
	F77_NAME(zheev, ZHEEV)(&jobz,&uplo,&lda,za,&lda,dw,work,&lwork,rwork,&info);

	if(info<0){
          if (myid==Host_ID){
	    printf("*************************** Error *************************\n");
	    printf("  Error in disentangling mixed bands when calling zheev \n");
	    printf("  with an illegal value for the %ith argument.      \n", -info);
	    printf("  Stop at k %i with band_num=%i, frozen number=%i.\n",k+1,band_num, Mk);
	    printf("*************************** Error *************************\n");
            fflush(0);
          }
          MPI_Finalize();
	  exit(0);
	}else if(info>0){
          if (myid==Host_ID){
	    printf("*************************** Error *************************\n");
	    printf("  Error in disentangling mixed bands when calling zheev\n");
	    printf("  which can not get convergence.\n");
	    printf("  Stop at k %i with band_num=%i, frozen number=%i.\n",k+1,band_num, Mk);
	    printf("*************************** Error *************************\n");
            fflush(0);
          }
          MPI_Finalize();
	  exit(0);
	}

        else{
	  for(i=0;i<band_num-Mk;i++){
	    if(dw[i]>(wbtot+smallvalue) || dw[i]<-smallvalue){
              if (myid==Host_ID){
                 printf("**************************** WARNNING ****************************\n");
                 printf(" At %ith k point, the %ith eigenvalue %10.5f is out of [0.0, %10.5f].\n",k+1,i+1,dw[i],wbtot);
                 printf(" Please check and try again.\n");
                 printf("**************************** WARNNING ****************************\n");fflush(0);
              }

	      /*
              MPI_Finalize();
              exit(0);
	      */
	    }
	  }
	}
	/* ignore the first band_num-wan_num eigen value and states */
	bindx=0;
	for(i=0;i<(band_num-Mk)-(wan_num-Mk);i++){
	  for(j=0;j<band_num-Mk;j++){
	    bindx++;
	  }
	}
	l=0;
	for(i=(band_num-Mk)-(wan_num-Mk);i<band_num-Mk;i++){

	  omega_I=omega_I+dw[i];

	  for(j=0;j<band_num;j++){
	    Udis[k][j][l+Mk].r=0.0; /* the lowest Mk coloums contain those frozen states, so start from Mk coloumns */
	    Udis[k][j][l+Mk].i=0.0;
	  }
	  for(j=0;j<band_num-Mk;j++){
	    if(Mk>0){
	      if(j+1+nbandwin[k][0]<nbandfroz[k][0]+1){
                info=j;
	      }else{
                info=j+Mk;
	      }
	    }else{
	      info=j;
	    }
	    Udis[k][info][l+Mk].r=za[bindx].r;
	    Udis[k][info][l+Mk].i=za[bindx].i;
	    bindx++;
	  }
	  l++;
	}/* i coloumn */
	for(i=0;i<band_num-Mk;i++){
	  for(j=0;j<band_num-Mk;j++){
	    Zmat_prev[k][i][j].r=Zmat[i][j].r;
	    Zmat_prev[k][i][j].i=Zmat[i][j].i;
	  }
	}
	free(dw);
	free(za);
	for(i=0;i<band_num-Mk;i++){
	  free(Zmat[i]);
	}
	free(Zmat);
      }/*  kpt_num */

      omega_I=(double)wan_num*wbtot-omega_I/(double)kpt_num;
      omega_I=omega_I*BohrR_Wannier*BohrR_Wannier;
      delta_OI=omega_I-omega_I_prev;
      if (myid==Host_ID){
         printf("|  %4d  |%18.12f|%18.12f|  ---> DISE\n",iter+1,omega_I,omega_I-omega_I_prev);
         fflush(0);
      }
      iter++;

      /*     if(iter>3000){
	     dis_alpha_mix=0.5*(1.0+(double)iter/(double)dis_max_iter-0.5); 
	     printf("alpha=%10.6f\n",dis_alpha_mix);
	     }
      */
    }/* iteration */
    if(iter>=dis_max_iter && fabs(delta_OI)>dis_conv_tol){
      if(Wannier90_fileout==0){
      if (myid==Host_ID){
        printf("**************************** WARNNING ****************************\n");
        printf(" Iteration for minimizing Omega_I is not converged. Please change \n");
        printf(" the parameters and try again.\n");
        printf("**************************** WARNNING ****************************\n");
      }
      } else {
      if (myid==Host_ID){
         printf("\nThe input files for Wannier90,\n"); 
         printf("\nSystem.Name.amn\n"); 
         printf("System.Name.mmn\n"); 
         printf("System.Name.eig\n"); 
         printf("System.Name.win\n"); 
         printf("\nare successfully generated.\n"); 
          }
      }
      MPI_Finalize();
      exit(0);
    }
    /* haveing got the optimized subspace, diagonalize system's original Hamiltonian inside 
       this subspace to obtain N Bloch-like eigenfunctions psi^tilde. See Sec.III.E of SMV's paper 
       Eq. (24) */
    za=(dcomplex*)malloc(sizeof(dcomplex)*wan_num*wan_num);
    dw=(double*)malloc(sizeof(double)*wan_num);
    for(k=0;k<kpt_num;k++){
      band_num=nbandwin[k][1]-nbandwin[k][0];
      /* now Zmat is the Hamiltonian in origianl Nk dimensional full space, which is 
	 diagnonal with eigen values at the diagonal site. */
      Zmat=(dcomplex**)malloc(sizeof(dcomplex*)*band_num);
      for(i=0;i<band_num;i++){
	Zmat[i]=(dcomplex*)malloc(sizeof(dcomplex)*band_num);
      }
      for(i=0;i<band_num;i++){
	for(j=0;j<band_num;j++){
	  Zmat[i][j].r=0.0;
	  Zmat[i][j].i=0.0;
	  if(i==j){
	    Zmat[i][j].r=eigen[spin][k][i];
	  }
	}
      }
      /* construct the new Hamiltionian in the optimized subspace */
      for(i=0;i<wan_num;i++){
        for(l=0;l<band_num;l++){
	  sumr=0.0; sumi=0.0;
	  for(j=0;j<band_num;j++){
	    sumr=sumr+Udis[k][j][i].r*Zmat[j][l].r+Udis[k][j][i].i*Zmat[j][l].i;
	    sumi=sumi+Udis[k][j][i].r*Zmat[j][l].i-Udis[k][j][i].i*Zmat[j][l].r;
	  }
	  Utmp[i][l].r=sumr;
	  Utmp[i][l].i=sumi;
        }
      }
      for(i=0;i<wan_num;i++){
        for(l=0;l<wan_num;l++){
	  sumr=0.0; sumi=0.0;
	  for(j=0;j<band_num;j++){
	    sumr=sumr+Utmp[i][j].r*Udis[k][j][l].r-Utmp[i][j].i*Udis[k][j][l].i;
	    sumi=sumi+Utmp[i][j].r*Udis[k][j][l].i+Utmp[i][j].i*Udis[k][j][l].r;
	  }
	  Hrot[i][l].r=sumr;
	  Hrot[i][l].i=sumi;
        }
      }
      bindx=0;
      for(i=0;i<wan_num;i++){
	for(j=0;j<wan_num;j++){
	  za[bindx].r=Hrot[j][i].r;
	  za[bindx].i=Hrot[j][i].i;
	  bindx++;
	}
      }
      /*  zheev(jobz, uplo, wan_num, za, wan_num, dw, &info); */
      /* clapack  Macbook Air */
      /*   int zheev_(char *jobz, char *uplo, integer *n, doublecomplex
       *za, integer *lda, doublereal *dw, doublecomplex *work, integer *lwork,
       doublereal *rwork, integer *info)

       zheev_(jobz, uplo, wan_num, za, wan_num, dw, work, 2*wan_num-1, rwork, 3*wan_num-2, &info);
      */
      lda=wan_num;
      F77_NAME(zheev, ZHEEV)(&jobz,&uplo,&lda,za,&lda,dw,work,&lwork,rwork,&info);
      if(info<0){
        if (myid==Host_ID){
  	  printf("*********************** Error **********************\n");
	  printf("Error in calling zheev to diagonalize H within optimized\n");
	  printf("subspace with an illegal value for the %ith argument.\n", -info);
	  printf("*********************** Error **********************\n");
        }
        MPI_Finalize();
	exit(0);
      }else if(info>0){
        if (myid==Host_ID){
	  printf("*********************** Error **********************\n");
	  printf("Error in calling zheev to diagonalize H within optimized\n");
	  printf("subspace which can not get convergence.\n");
	  printf("*********************** Error **********************\n");
        }
        MPI_Finalize();
	exit(0);
      }else{
	/*       printf("zheev success!\n"); */
        if(myid==Host_ID){
          printf("At k %i [%6.4f,%6.4f,%6.4f] eigenvalues (in Ha) in \n  optimized subspace, original space and differences are:\n",k+1,kg[k][0],kg[k][1],kg[k][2]);
          for(i=0;i<wan_num;i++){
            printf("%i  %14.10f   %14.10f    %15.13f\n",i+1, dw[i], eigen[spin][k][i], dw[i]-eigen[spin][k][i]);
          }
        }
      }
      bindx=0;
      for(i=0;i<wan_num;i++){
        for(j=0;j<wan_num;j++){
          Hrot[j][i].r=za[bindx].r;
          Hrot[j][i].i=za[bindx].i;
          bindx++;
        } /* j row */
      }/* i coloumn */

      for(i=0;i<band_num;i++){
	for(j=0;j<wan_num;j++){
	  sumr=0.0; sumi=0.0;
	  for(l=0;l<wan_num;l++){
	    sumr=sumr+Udis[k][i][l].r*Hrot[l][j].r-Udis[k][i][l].i*Hrot[l][j].i;
	    sumi=sumi+Udis[k][i][l].r*Hrot[l][j].i+Udis[k][i][l].i*Hrot[l][j].r;
	  }
	  Utmp[i][j].r=sumr;
	  Utmp[i][j].i=sumi;
	}
      }
      /* updating Uk matrix. This is the final matrix for disentangling the attached bands. */
      for(i=0;i<BAND;i++){
        for(j=0;j<wan_num;j++){
          Uk[spin][k][i][j].r=Utmp[i][j].r;
          Uk[spin][k][i][j].i=Utmp[i][j].i;
        }
      }

      for(i=0;i<band_num;i++){
        free(Zmat[i]);
      }
      free(Zmat);
    }/* kpt */
    free(dw);
    free(za);
  }/* spin */

  /*******************************************
         free arrays 
  ********************************************/

  for (i=0; i<kpt_num; i++){
    for (j=0; j<band_num; j++){
      free(Zmat_prev[i][j]);
    }
    free(Zmat_prev[i]);
  }
  free(Zmat_prev);

  for (i=0; i<kpt_num; i++){
    for (j=0; j<band_num; j++){
      free(Zmat_i[i][j]);
    }
    free(Zmat_i[i]);
  }
  free(Zmat_i);

  for (i=0; i<wan_num; i++){
    free(Hrot[i]);
  }
  free(Hrot);

  free(work);
  free(rwork);

  for(k=0;k<kpt_num;k++){
    for(i=0;i<BAND;i++){
      free(Udis[k][i]);
    }/* Windown */
    free(Udis[k]);
  } /*kpt */
  free(Udis);
  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for (i=0; i<BAND; i++){
	free(Mmnkb[k][bindx][i]);
      } /* i, j band num */
      free(Mmnkb[k][bindx]);
    }/* b vector */
    free(Mmnkb[k]);
  }
  free(Mmnkb);

  for(i=0;i<BAND;i++){
    free(Utmp[i]);
  }
  free(Utmp);

  for(k=0;k<kpt_num;k++){
    free(nbandwin[k]);
  }
  free(nbandwin);
  for(k=0;k<kpt_num;k++){
    free(nbandfroz[k]);
  }
  free(nbandfroz);

  if (myid==Host_ID){
     printf("Leaving disentangling.\n"); fflush(0);
  }
}/* Disentangliing_Bands */



#pragma optimization_level 1
void Getting_Zmatrix(dcomplex **Zmat, int presentk, double *wb, int tot_bvector, 
                     dcomplex ****Mmnkb, dcomplex ***Udis, int **kplusb, int band_num, 
                     int wan_num, int Mk, int **nbandwin, int **nbandfroz)
{
  int i,j,l,kpb, kpb_band, i1,j1,id,jd;
  int bindx, info;
  double sumr, sumi;
  dcomplex **MmnUdis, **tmpZ;

  /* allocation of arrays */

  MmnUdis = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0; i<band_num; i++){
    MmnUdis[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }
 
  tmpZ = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for(i=0; i<band_num; i++){
    tmpZ[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  /* start calc. */

  for(i=0;i<band_num-Mk;i++){
    for(j=0;j<band_num-Mk;j++){
      Zmat[i][j].r=0.0;
      Zmat[i][j].i=0.0;
    }
  }

  for(bindx=0;bindx<tot_bvector;bindx++){
    kpb=kplusb[presentk][bindx];
    kpb_band=nbandwin[kpb][1]-nbandwin[kpb][0];

    for(i=0;i<band_num;i++){
      for(j=0;j<wan_num;j++){
	sumr=0.0;sumi=0.0;
	for(l=0;l<kpb_band;l++){ 

	  sumr=sumr + Mmnkb[presentk][bindx][i][l].r*Udis[kpb][l][j].r
                    - Mmnkb[presentk][bindx][i][l].i*Udis[kpb][l][j].i;
	  sumi=sumi + Mmnkb[presentk][bindx][i][l].r*Udis[kpb][l][j].i
                    + Mmnkb[presentk][bindx][i][l].i*Udis[kpb][l][j].r;

	}

	MmnUdis[i][j].r=sumr;
	MmnUdis[i][j].i=sumi;

      }
    }
    for(i=0;i<band_num-Mk;i++){ /* we must ignore the Mk number of frozen states */
      if(Mk>0){
	if((i+1+nbandwin[presentk][0])<(nbandfroz[presentk][0]+1)){/* if the i state is between two bottoms of outer and inner window */
	  i1=i;/*just take i value since all the states inside outer window is indexed from 0*/
	}else{/* if(i+1+nbandwin[k][0]>nbandfroz[k][1]) */ /* if i state is between tow tops of outter and inner windwo */
	  i1=i+Mk;/* jump Mk states that locate inside the inner window. */
	}
      }else{ /* No frozen case */
	i1=i;
      }
      for(j=0;j<band_num-Mk;j++){
	if(Mk>0){
	  if(j+1+nbandwin[presentk][0]<nbandfroz[presentk][0]+1){/* if the j state is between two bottoms of outer and inner window */
	    j1=j;/*just take j value since all the states inside outer window is indexed from 0*/
	  }else{/* if(j+1+nbandwin[k][0]>nbandfroz[k][1]) */ /* if i state is between tow tops of outter and inner windwo */
	    j1=j+Mk;/* jump Mk states that locate inside the inner window. */
	  }
	}else{
	  j1=j;
	}
	sumr=0.0;sumi=0.0;
	for(l=0;l<wan_num;l++){
	  sumr=sumr+MmnUdis[i1][l].r*MmnUdis[j1][l].r+MmnUdis[i1][l].i*MmnUdis[j1][l].i;
	  sumi=sumi-MmnUdis[i1][l].r*MmnUdis[j1][l].i+MmnUdis[i1][l].i*MmnUdis[j1][l].r;
	}
	tmpZ[i][j].r=sumr;
	tmpZ[i][j].i=sumi;
      }
    }     

    for(i=0;i<band_num-Mk;i++){
      for(j=0;j<band_num-Mk;j++){
	Zmat[i][j].r=Zmat[i][j].r+wb[bindx]*tmpZ[i][j].r;
	Zmat[i][j].i=Zmat[i][j].i+wb[bindx]*tmpZ[i][j].i;
      }
    }    
  }/* b vectors */

  /* freeing of arrays */

  for(i=0; i<band_num; i++){
    free(MmnUdis[i]);
  }
  free(MmnUdis);
 
  for(i=0; i<band_num; i++){
    free(tmpZ[i]);
  }
  free(tmpZ);

}/* Getting_Zmatrix */




#pragma optimization_level 1
void Wannier_Interpolation(dcomplex ****Uk, double ***eigen, int spinsize, int SpinP_switch, int kpt_num, int band_num, int wan_num, int ***Nk, double rtv[4][4], double **kg, double ChemP, int r_num, int**rvect, int *ndegen){
  int i,j,k,l, spin, rpt, num_kpath, npath;
  int BAND;
  double **kpath;
  dcomplex ****HWk;/* H(w)(k) Eq.(21)-paper of Mostofi */
  dcomplex ****HR;/*[spin][rvect][wan_num][wan_num];*/ /* Hamiltonian in real space Eq. (22)-paper of Mostofi or Eq.(13)-paper of Yates et al.*/
  dcomplex **tmpH, **tmpZ;
  double sumr, sumi,rdotk;
  dcomplex ****Hkpath;
  char jobz, uplo; 
  dcomplex *work;
  double *rwork;
  dcomplex *za;
  double *dw;
  int info,lwork,lda;
  FILE *fpBand;
  char fname[300];
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
     printf(" Wannier Interpolation ... ...\n");fflush(0);
  }

  /* allocation of arrays */

  HWk = (dcomplex****)malloc(sizeof(dcomplex***)*spinsize);
  for (i=0; i<spinsize; i++){
    HWk[i] = (dcomplex***)malloc(sizeof(dcomplex**)*kpt_num);
    for (j=0; j<kpt_num; j++){
      HWk[i][j] = (dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for (k=0; k<wan_num; k++){
        HWk[i][j][k] = (dcomplex*)malloc(sizeof(dcomplex)*wan_num);
      }
    }
  }

  tmpH = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for (i=0; i<band_num; i++){
    tmpH[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  tmpZ = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for (i=0; i<band_num; i++){
    tmpZ[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  work = (dcomplex*)malloc(sizeof(dcomplex)*3*wan_num);
  rwork = (double*)malloc(sizeof(double)*3*wan_num);
  za = (dcomplex*)malloc(sizeof(dcomplex)*wan_num*wan_num);
  dw = (double*)malloc(sizeof(double)*wan_num);

  /* start calc. */

  BAND=band_num;
  jobz='V';
  uplo='U';
  lwork=3*wan_num;
  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<kpt_num;k++){
      band_num=Nk[spin][k][1]-Nk[spin][k][0];
      /* tmpH is the Hamiltonian in origianl Nk dimensional full space, which is 
	 diagnonal with eigen values at the diagonal site. */
      for(i=0;i<band_num;i++){
        for(j=0;j<band_num;j++){
          tmpH[i][j].r=0.0;
          tmpH[i][j].i=0.0;
          if(i==j){
            tmpH[i][j].r=eigen[spin][k][i];
          }
        }
      }

      /* Uk^dagger * H(k)    Dimension: (N x Nk) x (Nk x Nk) = N x Nk */
      for(i=0;i<wan_num;i++){
        for(j=0;j<band_num;j++){
          sumr=0.0; sumi=0.0;
          for(l=0;l<band_num;l++){
            sumr=sumr+Uk[spin][k][l][i].r*tmpH[l][j].r+Uk[spin][k][l][i].i*tmpH[l][j].i;
            sumi=sumi+Uk[spin][k][l][i].r*tmpH[l][j].i-Uk[spin][k][l][i].i*tmpH[l][j].r;
          }
          tmpZ[i][j].r=sumr;
          tmpZ[i][j].i=sumi; 
        } 
      }

      /* (Uk^dagger * H(k)) * Uk   Dimesion: (N x Nk ) x (Nk x N) = N x N */
      for(i=0;i<wan_num;i++){
        for(j=0;j<wan_num;j++){
          sumr=0.0; sumi=0.0;
          for(l=0;l<band_num;l++){
            sumr=sumr+tmpZ[i][l].r*Uk[spin][k][l][j].r-tmpZ[i][l].i*Uk[spin][k][l][j].i;
            sumi=sumi+tmpZ[i][l].r*Uk[spin][k][l][j].i+tmpZ[i][l].i*Uk[spin][k][l][j].r;
          }
          HWk[spin][k][i][j].r=sumr;
          HWk[spin][k][i][j].i=sumi; 
        }
      }
      /* upto here, we get the Hamiltonian in Wannier gauge within reciprocal k-space
	 see Eq. (8) of PRB75,195121(2007), where q is used to express the k vectors before interpolation 
      */
    }/* kpt */ 
  }/* spin */ 

  /* out put the Hamiltonian in Wannier gauge at k=0.0*/
  if(debug1){
     printf("Hamiltonian in Wannier Gauge with k=0:\n");
     for(spin=0;spin<spinsize;spin++){
       for(i=0;i<wan_num;i++){
         for(j=0;j<wan_num;j++){
           printf("%10.5f  ",HWk[spin][0][i][j].r); 
         }
         printf("\n");
       }
     }
     /* Hamiltonian in Wannier gauge at k=0.0 should be real?*/
     printf("Check reality of Hamiltonian in Wannier Gauge with k=0.\n The following is its imaginary part\n");
     for(spin=0;spin<spinsize;spin++){
       for(i=0;i<wan_num;i++){
         for(j=0;j<wan_num;j++){
           printf("%10.5f  ",HWk[spin][0][i][j].i);
         }
         printf("\n");
       }
     }
  } 

  HR=(dcomplex****)malloc(sizeof(dcomplex***)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    HR[spin]=(dcomplex***)malloc(sizeof(dcomplex**)*r_num);
    for(rpt=0;rpt<r_num;rpt++){
      HR[spin][rpt]=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for(i=0;i<wan_num;i++){
        HR[spin][rpt][i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
        for(j=0;j<wan_num;j++){
          HR[spin][rpt][i][j].r=0.0;
          HR[spin][rpt][i][j].i=0.0;
        }
      }
    }
  }
  /* Now we can transfer HWk to HR */ 
  for(spin=0;spin<spinsize;spin++){
    for(rpt=0;rpt<r_num;rpt++){
      for(k=0;k<kpt_num;k++){
        rdotk=2.0*PI*(kg[k][0]*(double)rvect[0][rpt]+kg[k][1]*(double)rvect[1][rpt]+kg[k][2]*(double)rvect[2][rpt]); 
        for(i=0;i<wan_num;i++){
          for(j=0;j<wan_num;j++){
            HR[spin][rpt][i][j].r= HR[spin][rpt][i][j].r+(cos(rdotk)*HWk[spin][k][i][j].r+sin(rdotk)*HWk[spin][k][i][j].i)/(double)kpt_num;
            HR[spin][rpt][i][j].i= HR[spin][rpt][i][j].i+(-sin(rdotk)*HWk[spin][k][i][j].r+cos(rdotk)*HWk[spin][k][i][j].i)/(double)kpt_num; 
          }
        } /* i, j */ 
      } /* kpt */
    } /* rpt */
  }/* spin */

  /* out put the Hamiltonian in Wannier gauge in Real-space*/
  /* output real-space Hamiltonian in Wannier Gauge into file case.HWR*/
  if (myid==Host_ID){
    sprintf(fname,"%s%s.HWR",filepath,filename);
    printf("Real-space Hamiltonian Data file %s\n",fname);
    if((fpBand=fopen(fname,"wt"))==NULL){
      printf("Error in opening file for HWR: %s\n",fname);
    }else{
      fprintf(fpBand,"Real-space Hamiltonian in Wannier Gauge on Wigner-Seitz supercell.\n");
      fprintf(fpBand,"Number of Wannier Function %i\n",wan_num);
      fprintf(fpBand,"Number of Wigner-Seitz supercell %i\n",r_num);
      fprintf(fpBand,"Lattice vector (in Bohr)\n");
      for (i=0;i<3;i++){
        for (j=0;j<3;j++) {
          fprintf(fpBand,"%10.5f ", r_latt[i][j]);
        }
        fprintf(fpBand,"\n");
      }
      if(SpinP_switch==3){
        fprintf(fpBand,"non-collinear calculation spinsize %1d\nFermi level %lf\n",1,ChemP);  /* set SpinP_switch==0 */
      }else{
        fprintf(fpBand,"collinear calculation spinsize %1d\nFermi level %lf\n",spinsize,ChemP);
      }
      for(spin=0; spin<spinsize;spin++){
        for(rpt=0;rpt<r_num;rpt++){
          fprintf(fpBand,"R ( %4d %4d %4d ) %4d\n", rvect[0][rpt],rvect[1][rpt],rvect[2][rpt],ndegen[rpt]);
          for (i=0;i<wan_num;i++){
            for (j=0;j<wan_num;j++) {
              fprintf(fpBand,"%4d  %4d  %18.12f%18.12f\n",i+1,j+1,HR[spin][rpt][i][j].r,HR[spin][rpt][i][j].i);
            }
          } 
        } /* rpt */
      }/* spin */
    }/* end file output*/
  } 
  if(debug1){
    printf("Hamiltonian in Wannier Gauge in real space:\n");
    for(rpt=0;rpt<r_num;rpt++){
      if((rvect[0][rpt]==0 && rvect[1][rpt]==0 && rvect[2][rpt]==0)|| (rvect[0][rpt]==0 && rvect[1][rpt]==0 && rvect[2][rpt]==1) || (rvect[0][rpt]==0 && rvect[1][rpt]==1 && rvect[2][rpt]==0) || (rvect[0][rpt]==0 && rvect[1][rpt]==1 && rvect[2][rpt]==1) || (rvect[0][rpt]==1 && rvect[1][rpt]==1 && rvect[2][rpt]==0) ||(rvect[0][rpt]==1 && rvect[1][rpt]==1 && rvect[2][rpt]==1) || (rvect[0][rpt]==1 && rvect[1][rpt]==0 && rvect[2][rpt]==1)||(rvect[0][rpt]==0 && rvect[1][rpt]==0 && rvect[2][rpt]==2) ||(rvect[0][rpt]==0 && rvect[1][rpt]==2 && rvect[2][rpt]==0)) {
	/* 000, 001, 010, 011, 110, 111, 101, 002, 020 */
        printf("R vector: (%i,%i,%i)\n",rvect[0][rpt],rvect[1][rpt],rvect[2][rpt]);
        for(spin=0;spin<spinsize;spin++){
          printf("spin %i\n",spin);
          for(i=0;i<wan_num;i++){
            for(j=0;j<wan_num;j++){
              printf("%10.5f  ",HR[spin][rpt][i][j].r);
            }
            printf("\n");
          }
        } 
      }   
    }
    /* Hamiltonian in Wannier gauge at k=0.0 should be real?*/
    printf("Check reality of Hamiltonian in Wannier Gauge in real space.\n The following is its imaginary part\n");
    for(rpt=0;rpt<r_num;rpt++){
      if((rvect[0][rpt]==0 && rvect[1][rpt]==0 && rvect[2][rpt]==0)|| (rvect[0][rpt]==0 && rvect[1][rpt]==0 && rvect[2] [rpt]==1) || (rvect[0][rpt]==0 && rvect[1][rpt]==1 && rvect[2][rpt]==0) || (rvect[0][rpt]==0 && rvect[1][rpt]==1 && rvect[2][rpt]==1) || (rvect[0][rpt]==1 && rvect[1][rpt]==1 && rvect[2][rpt]==0) ||(rvect[0][rpt]==1 && rvect[1][rpt]== 1 && rvect[2][rpt]==1) || (rvect[0][rpt]==1 && rvect[1][rpt]==0 && rvect[2][rpt]==1)||(rvect[0][rpt]==0 && rvect[1][rpt]==0 && rvect[2][rpt]==2) ||(rvect[0][rpt]==0 && rvect[1][rpt]==2 && rvect[2][rpt]==0)) {
        printf("R vector: (%i,%i,%i)\n",rvect[0][rpt],rvect[1][rpt],rvect[2][rpt]);
        for(spin=0;spin<spinsize;spin++){
          printf("spin %i\n",spin);
          for(i=0;i<wan_num;i++){
            for(j=0;j<wan_num;j++){
              printf("%10.5f  ",HR[spin][rpt][i][j].i);
            }
            printf("\n");
          }
        }
      }
    }
  }
  /* make interpolation for band structure */
 if(Wannier_Draw_Int_Bands){
  /* 1. define the k path */
  num_kpath=0;
  for(i=1;i<=Band_Nkpath; i++){
    num_kpath+=Band_N_perpath[i];
  }
  kpath=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    kpath[i]=(double*)malloc(sizeof(double)*num_kpath);
  }
  Hkpath=(dcomplex****)malloc(sizeof(dcomplex***)*spinsize);
  for(spin=0;spin<spinsize;spin++){
    Hkpath[spin]=(dcomplex***)malloc(sizeof(dcomplex**)*num_kpath);
    for(k=0;k<num_kpath;k++){
      Hkpath[spin][k]=(dcomplex**)malloc(sizeof(dcomplex*)*wan_num);
      for(i=0;i<wan_num;i++){ 
        Hkpath[spin][k][i]=(dcomplex*)malloc(sizeof(dcomplex)*wan_num);
        for(j=0;j<wan_num;j++){
          Hkpath[spin][k][i][j].r=0.0;
          Hkpath[spin][k][i][j].i=0.0;
        }
      }
    }
  }
  k=0;
  for(i=1;i<=Band_Nkpath; i++){
    for(j=1;j<=Band_N_perpath[i];j++){
      kpath[0][k]=Band_kpath[i][1][1]+(Band_kpath[i][2][1]-Band_kpath[i][1][1])*(j-1)/(Band_N_perpath[i]-1)+1E-12;
      kpath[1][k]=Band_kpath[i][1][2]+(Band_kpath[i][2][2]-Band_kpath[i][1][2])*(j-1)/(Band_N_perpath[i]-1)+1E-12;
      kpath[2][k]=Band_kpath[i][1][3]+(Band_kpath[i][2][3]-Band_kpath[i][1][3])*(j-1)/(Band_N_perpath[i]-1)+1E-12;
      k++;
    }
  }

  num_kpath=k;
  /* 2. Interpolation along these kpt */  
  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<num_kpath;k++){
      for(rpt=0;rpt<r_num;rpt++){
        rdotk=2.0*PI*(kpath[0][k]*(double)rvect[0][rpt]+kpath[1][k]*(double)rvect[1][rpt]+kpath[2][k]*(double)rvect[2][rpt]); 
        for(i=0;i<wan_num;i++){
          for(j=0;j<wan_num;j++){
            Hkpath[spin][k][i][j].r=Hkpath[spin][k][i][j].r+(cos(rdotk)*HR[spin][rpt][i][j].r-sin(rdotk)*HR[spin][rpt][i][j].i)/(double)ndegen[rpt];
            Hkpath[spin][k][i][j].i=Hkpath[spin][k][i][j].i+(sin(rdotk)*HR[spin][rpt][i][j].r+cos(rdotk)*HR[spin][rpt][i][j].i)/(double)ndegen[rpt];      
	    /*            Hkpath[spin][k][i][j].r=Hkpath[spin][k][i][j].r+(cos(rdotk)*HR[spin][rpt][i][j].r-sin(rdotk)*HR[spin][rpt][i][j].i)/(double)r_num; */
	    /*            Hkpath[spin][k][i][j].i=Hkpath[spin][k][i][j].i+(sin(rdotk)*HR[spin][rpt][i][j].r+cos(rdotk)*HR[spin][rpt][i][j].i)/(double)r_num; */
          } 
        }
      }
    }
  } 
  if (myid==Host_ID){
    sprintf(fname,"%s%s.Wannier_Band",filepath,filename);
    printf("Band data file %s\n",fname);
     if((fpBand=fopen(fname,"wt"))==NULL){
       printf("Error in opening file for Bands: %s\n",fname);
     }else{
       if(SpinP_switch==3){
         fprintf(fpBand," %d  %d  %lf\n",wan_num,0,ChemP);  /* set SpinP_switch==0 */
       }else{
         fprintf(fpBand," %d  %d  %lf\n",wan_num,SpinP_switch,ChemP);  
       }
       for (i=1;i<=3;i++)
       for (j=1;j<=3;j++) {
         fprintf(fpBand,"%lf ", rtv[i][j]);
       }
       fprintf(fpBand,"\n");
       fprintf(fpBand,"%d\n",Band_Nkpath);
       for (i=1;i<=Band_Nkpath;i++) {
         fprintf(fpBand,"%d %lf %lf %lf  %lf %lf %lf  %s %s\n",
                 Band_N_perpath[i],
                 Band_kpath[i][1][1], Band_kpath[i][1][2], Band_kpath[i][1][3],
                 Band_kpath[i][2][1], Band_kpath[i][2][2], Band_kpath[i][2][3],
                 Band_kname[i][1],Band_kname[i][2]);
       }
    /* Get the eigen value */
       for(k=0;k<num_kpath;k++){
         for(spin=0;spin<spinsize;spin++){
      	   l=0;
 	   for(i=0;i<wan_num;i++){
	     for(j=0;j<wan_num;j++){
	       za[l].r=Hkpath[spin][k][j][i].r;
	       za[l].i=Hkpath[spin][k][j][i].i;
	       l++;
	     }
	   }
	/*      zheev(jobz, uplo, wan_num, za, wan_num, dw, &info); */
	/* clapack  Macbook Air */
	/*   int zheev_(char *jobz, char *uplo, integer *n, doublecomplex
	 *za, integer *lda, doublereal *dw, doublecomplex *work, integer *lwork,
	 doublereal *rwork, integer *info)

	 zheev_(jobz, uplo, wan_num, za, wan_num, dw, work, 2*wan_num-1, rwork, 3*wan_num-2, &info);
	*/
	   lda=wan_num;
  	   F77_NAME(zheev, ZHEEV)(&jobz,&uplo,&lda,za,&lda,dw,work,&lwork,rwork,&info);
 	   if(info<0){
	     printf("*********************** Error **********************\n");
	     printf("Error in calling zheev to diagonalize H within optimized\n");
	     printf("subspace with an illegal value for the %ith argument.\n", -info);
	     printf("*********************** Error **********************\n");
	     exit(0);
	   }else if(info>0){
	     printf("*********************** Error **********************\n");
	     printf("Error in calling zheev to diagonalize H within optimized\n");
	     printf("subspace which can not get convergence.\n");
	     printf("*********************** Error **********************\n");
	     exit(0);
	   }else{
	     /*       printf("zheev success!\n"); */
	     fprintf(fpBand,"%d %lf %lf %lf\n", wan_num,kpath[0][k], kpath[1][k], kpath[2][k]);
	     for (info=0; info<wan_num; info++) {
	       fprintf(fpBand,"%lf ",dw[info]);
	     }
	     fprintf(fpBand  ,"\n");
	   }
         } /* spin */
       } /* kpt */
       fclose(fpBand);
     }
  }
  for(spin=0;spin<spinsize;spin++){
    for(k=0;k<num_kpath;k++){
      for(i=0;i<wan_num;i++){
        free(Hkpath[spin][k][i]);
      }
      free(Hkpath[spin][k]);
    }
    free(Hkpath[spin]);
  }
  free(Hkpath);

  for(i=0;i<3;i++){
    free(kpath[i]);
  }
  free(kpath);
 }/* band interpolation */
  for(spin=0;spin<spinsize;spin++){
    for(rpt=0;rpt<r_num;rpt++){
      for(i=0;i<wan_num;i++){
        free(HR[spin][rpt][i]);
      }
      free(HR[spin][rpt]);
    }
    free(HR[spin]);
  }
  free(HR);

  /* freeing of arrays */

  for (i=0; i<spinsize; i++){
    for (j=0; j<kpt_num; j++){
      for (k=0; k<wan_num; k++){
        free(HWk[i][j][k]);
      }
      free(HWk[i][j]);
    }
    free(HWk[i]);
  }
  free(HWk);

  for (i=0; i<band_num; i++){
    free(tmpH[i]);
  }
  free(tmpH);

  for (i=0; i<band_num; i++){
    free(tmpZ[i]);
  }
  free(tmpZ);

  free(work);
  free(rwork);
  free(za);
  free(dw);

}/* Wannier_Interpolation */


#pragma optimization_level 1
void Wigner_Seitz_Vectors(double metric[3][3], double rtv[4][4], int knum_i, 
                         int knum_j, int knum_k, int *r_num, 
                         int **rvect, int *ndegen)

{
  int i1,i2,i3, n1, n2, n3, icnt, nd[3][125], i,j,rnum;
  double dist[125], dist_min, rdotk;
  int nosym;
 
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  nosym=0;

  if(nosym==1 && *r_num==-1){
    *r_num=(knum_i)*(knum_j)*(knum_k);
    return;
  }

  if(nosym==1 && *r_num!=-1){
    rnum=0;
    for(n1=0;n1<knum_i;n1++){
      for(n2=0;n2<knum_j;n2++){
	for(n3=0;n3<knum_k;n3++){
	  rvect[0][rnum]=n1;
	  rvect[1][rnum]=n2;
	  rvect[2][rnum]=n3;
	  ndegen[rnum]=1;
	  rnum++;
	}
      }
    } 
    *r_num=rnum;
    return;
  } 
  rnum=0;
  for(n1=-knum_i;n1<knum_i+1;n1++){
    for(n2=-knum_j;n2<knum_j+1;n2++){
      for(n3=-knum_k;n3<knum_k+1;n3++){
	icnt = 0;
	for(i1=-2;i1<3;i1++){
	  for(i2=-2;i2<3;i2++){
	    for(i3=-2;i3<3;i3++){
	      nd[0][icnt]=n1-i1*knum_i;        
	      nd[1][icnt]=n2-i2*knum_j;        
	      nd[2][icnt]=n3-i3*knum_k;       
	      dist[icnt]=0.0;
	      for(i=0;i<3;i++){
		for(j=0;j<3;j++){
		  dist[icnt]=dist[icnt]+(double)(nd[i][icnt])*metric[i][j]*(double)(nd[j][icnt]);
		}
	      } 
	      icnt++;
	    }/* i3 */       
	  }/* i2 */
	} /* i1 */ 
	dist_min=dist[0];
	for(i=0;i<icnt;i++){
	  if(dist_min>dist[i]){
	    dist_min=dist[i];
	  }
	}
	if(fabs(dist[62]-dist_min)<0.0000001){
	  if(*r_num!=-1){
            ndegen[rnum]=0;
	    for(i=0;i<125;i++){
	      if(fabs(dist[i]-dist_min)<0.0000001){
	        ndegen[rnum]++;
		/*
		  if(rnum<1){
                  rvect[0][rnum]=nd[0][i];
                  rvect[1][rnum]=nd[1][i];
                  rvect[2][rnum]=nd[2][i];
                  ndegen[rnum]=1;
                  rnum++;
		  }else{
                  icnt=0;
                  for(j=0;j<rnum;j++){
		  if(rvect[0][j]==nd[0][i] && rvect[1][j]==nd[1][i] && rvect[2][j]==nd[2][i]){
		  icnt=1;
		  break;
		  }
                  }
                  if(icnt==0){
		  rvect[0][rnum]=nd[0][i];
		  rvect[1][rnum]=nd[1][i];
		  rvect[2][rnum]=nd[2][i];
		  ndegen[rnum]=1;   
		  rnum++;
                  }
		  }
		*/
	      }
	    }
            rvect[0][rnum]=n1;
            rvect[1][rnum]=n2;
            rvect[2][rnum]=n3;
	  }
          rnum++;
	} 
      } /* n3 */
    }/* n2 */
  }/* n1 */
  if(*r_num==-1){
    *r_num=rnum;
    return;
  }
  *r_num=rnum; 
  if (myid==Host_ID){
     printf("There are %i lattice points found in Wigner-Seitz supercell.\n",rnum);
  }
/*
  if(rnum>0){
    for(i=0;i<rnum;i++){
      printf("vector (%i,%i,%i) degeneracy: %i\n",rvect[0][i],rvect[1][i],rvect[2][i],ndegen[i]); 
    }
  }
*/
  /* check sum rule  */
  dist_min=0.0;
  for(i=0;i<rnum;i++){
    dist_min=dist_min+1.0/(double)ndegen[i];
  }
  if(fabs(dist_min-(double)knum_i*knum_j*knum_k)>smallvalue){
    if (myid==Host_ID){
      printf("**************************** Error **********************\n");
      printf("*   In Wigner_Seitz_Vectors subroutine, error happens.  *\n");
      printf("*   Please change setting of Wannier.Kgrid.             *\n");
      printf("**************************** Error **********************\n");
      printf("***********************************INFO**************************************\n");
      printf("Reciprocal Lattices lengths are:%10.6f %10.6f %10.6f\n",rtv[0][0],rtv[0][1],rtv[0][2]);            printf("The ratio among them are: b1:b2=%10.6f b1:b3=%10.6f b2:b3=%10.6f\n",rtv[0][0]/rtv[0][1],rtv[0][0]/rtv[0][2],rtv[0][1]/rtv[0][2]);
      printf("Message: Please try to set Wannier.Kgrid has the similar ratio as above.\n");
      printf("************************************INFO*************************************\n");
    }
    MPI_Finalize();
    exit(0);
  }
}/* Wigner_Seitz_Vectors */



#pragma optimization_level 1
void Udis_projector_frozen(dcomplex ****Uk,int ***Nk,int ***innerNk,int kpt_num, int spinsize, int band_num, int wan_num){
  int i,j,k,l,spin, Nkp, Mk;
  double sumr, sumi;
  dcomplex **Qinner;
  dcomplex **Pgk, **PQ;

  int info,lwork,lda;
  char jobz, uplo;
  dcomplex *work;
  double *rwork;
  dcomplex *za;/* [band_num*band_num]; */
  double *dw;/* [band_num]; */

  /* allocation of arrays */

  Qinner = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for (i=0; i<band_num; i++){
    Qinner[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  Pgk = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for (i=0; i<band_num; i++){
    Pgk[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  PQ = (dcomplex**)malloc(sizeof(dcomplex*)*band_num);
  for (i=0; i<band_num; i++){
    PQ[i] = (dcomplex*)malloc(sizeof(dcomplex)*band_num);
  }

  work = (dcomplex*)malloc(sizeof(dcomplex)*3*band_num);
  rwork = (double*)malloc(sizeof(double)*3*band_num);

  /* start calc. */

  jobz='V';
  uplo='U';
  lwork=3*band_num;

/*  printf("Inner window is specified and start initial guess precessing with frozen states ......\n");fflush(0);
*/
  for(spin=0;spin<spinsize; spin++){
    for(k=0;k<kpt_num;k++){
      Nkp=Nk[spin][k][1]-Nk[spin][k][0];
      Mk=innerNk[spin][k][1]-innerNk[spin][k][0];
      for(i=0;i<Nkp;i++){
	for(j=0;j<Nkp;j++){
	  Qinner[i][j].r=0.0;
	  Qinner[i][j].i=0.0;
	}
      }
      for(i=0;i<Nkp;i++){/* i+1+Nk[spin][k][0] is the actual position of the bands. innerNk[spin][k][0]+1 marks the actual position of first frozen bands */
	if(i+Nk[spin][k][0]+1<innerNk[spin][k][0]+1 || i+Nk[spin][k][0]+1>innerNk[spin][k][1]){
          Qinner[i][i].r=1.0;
          Qinner[i][i].i=0.0;
	}
      }
      /* Pgk matrix is Uk x Uk^dagger (Nk x N) x (N x Nk)*/
      for(i=0;i<Nkp;i++){
	for(j=0;j<Nkp;j++){
	  sumr=0.0;sumi=0.0;
	  for(l=0;l<wan_num;l++){
	    sumr=sumr+Uk[spin][k][i][l].r*Uk[spin][k][j][l].r+Uk[spin][k][i][l].i*Uk[spin][k][j][l].i; 
	    sumi=sumi-Uk[spin][k][i][l].r*Uk[spin][k][j][l].i+Uk[spin][k][i][l].i*Uk[spin][k][j][l].r; 
	  }
	  Pgk[i][j].r=sumr;
	  Pgk[i][j].i=sumi;
	}
      }     
      /* PQ is Pgk*Qinner */
      for(i=0;i<Nkp;i++){
	for(j=0;j<Nkp;j++){
	  sumr=0.0;sumi=0.0;
	  for(l=0;l<Nkp;l++){
	    sumr=sumr+Pgk[i][l].r*Qinner[l][j].r-Pgk[i][l].i*Qinner[l][j].i;  
	    sumi=sumi+Pgk[i][l].r*Qinner[l][j].i+Pgk[i][l].i*Qinner[l][j].r;  
	  }
	  PQ[i][j].r=sumr; 
	  PQ[i][j].i=sumi;
	}
      }     
      /* Becareful: Pgk now is Qinner*Pgk*Qinner */
      for(i=0;i<Nkp;i++){
	for(j=0;j<Nkp;j++){
	  sumr=0.0;sumi=0.0;
	  for(l=0;l<Nkp;l++){
	    sumr=sumr+Qinner[i][l].r*PQ[l][j].r-Qinner[i][l].i*PQ[l][j].i;
	    sumi=sumi+Qinner[i][l].r*PQ[l][j].i+Qinner[i][l].i*PQ[l][j].r;
	  }
	  Pgk[i][j].r=sumr;
	  Pgk[i][j].i=sumi;
	}
      }

      za=(dcomplex*)malloc(sizeof(dcomplex)*Nkp*Nkp);
      dw=(double*)malloc(sizeof(double)*Nkp);
      l=0;
      for(i=0;i<Nkp;i++){
	for(j=0;j<Nkp;j++){
	  za[l].r=Pgk[j][i].r;
	  za[l].i=Pgk[j][i].i;
	  l++;
	}
      }    
      /*     zheev(jobz, uplo, Nkp, za, Nkp, dw, &info); */
      /* clapack  Macbook Air */
      /*   int zheev_(char *jobz, char *uplo, integer *n, doublecomplex
       *za, integer *lda, doublereal *dw, doublecomplex *work, integer *lwork,
       doublereal *rwork, integer *info)
       
       zheev_(jobz, uplo, wan_num, za, wan_num, dw, work, 2*wan_num-1, rwork, 3*wan_num-2, &info);      
      */     
      lda=Nkp;
      F77_NAME(zheev, ZHEEV)(&jobz,&uplo,&lda,za,&lda,dw,work,&lwork,rwork,&info);
      if(info<0){
	printf("*********************** Error **********************\n");
	printf("Error in Udis_projector_frozen when calling zheev\n");
	printf("with an illegal value for the %ith argument.\n", -info);
	printf("*********************** Error **********************\n");
	exit(0);
      }else if(info>0){
	printf("*********************** Error **********************\n");
	printf("Error in Udis_projector_frozen when calling zheev\n");
	printf("which can not get convergence.\n");
	printf("*********************** Error **********************\n");
	exit(0);
      }else{
	for(i=0;i<Nkp;i++){
	  if(dw[i]<-smallvalue || dw[i]>(1.0+smallvalue)){
	    printf("**************************** ERROR ****************************\n");
	    printf("At %ith k point, the %ith eigenvalue %20.15f is out of [0.0,1.0].\n",k+1,i+1,dw[i]);
	    printf("Please check and try again.\n");
	    printf("**************************** ERROR ****************************\n");
	    fflush(0);
	    exit(0);
	  }
	}
      }
      /* take the largest wan_num-Mk eigen values and states */
      l=0;
      for(i=0;i<Nkp-(wan_num-Mk);i++){
	for(j=0;j<Nkp;j++){
	  l++;
	}
      }
      for(i=Nkp-(wan_num-Mk);i<Nkp;i++){
	for(j=0;j<Nkp;j++){
	  Uk[spin][k][j][i-Nkp+wan_num].r=za[l].r;/* i-(Nkp-(wan_num-Mk))+Mk */
	  Uk[spin][k][j][i-Nkp+wan_num].i=za[l].i;
	  l++;
	} /* j row */
      }/* i column */
      /* put the frozen part into the lowest columns */
      for(i=0;i<Mk;i++){
	for(j=0;j<Nkp;j++){
	  Uk[spin][k][j][i].r=0.0;
	  Uk[spin][k][j][i].i=0.0;
	}
	Uk[spin][k][i+innerNk[spin][k][0]-Nk[spin][k][0]][i].r=1.0;
      }
      free(dw);free(za);     
    }/* kpt */
  }/* spin */

  /* freeing of arrays */

  for (i=0; i<band_num; i++){
    free(Qinner[i]);
  }
  free(Qinner);

  for (i=0; i<band_num; i++){
    free(Pgk[i]);
  }
  free(Pgk);

  for (i=0; i<band_num; i++){
    free(PQ[i]);
  }
  free(PQ);

  free(work);
  free(rwork);

}/* Udis_projector_frozen */



#pragma optimization_level 1
void Center_Guide( dcomplex ****Mmnkb, int WANNUM, int *bdirection, 
                   int nbdir, int kpt_num, double **bvector, int tot_bvector)
{
  /* This subroutine will follow wannier90 to get the optimal cut of sheet in ln() function */
    
  int i,j,k,nindx,bindx;
  dcomplex *csum,csumt;
  double *xx,smat[3][3],svec[3],**sinv; 
  double xx0,det,brn,tmp;  
  int myid;
  
  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  csum = (dcomplex*)malloc(sizeof(dcomplex)*nbdir);
  xx = (double*)malloc(sizeof(double)*nbdir);

  /* start calc. */
  
  if (myid==Host_ID){
    printf("Using guide for WF center.\n"); 
/*  no necessary to show
    for(j=0;j<WANNUM;j++){
       printf("wn %i  (%10.5f, %10.5f, %10.5f)\n",j,rguide[0][j],rguide[1][j],rguide[2][j]);
    }
*/
  }

  sinv=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    sinv[i]=(double*)malloc(sizeof(double)*3);
  }

  for(i=0;i<nbdir;i++){
    csum[i].r=0.0;
    csum[i].i=0.0;
    xx[i]=0.0;
  }

  for(i=0;i<3;i++){
    svec[i]=0.0;
    for(j=0;j<3;j++){
      smat[i][j]=0.0;
      sinv[i][j]=0.0;
    }
  }

  for(nindx=0;nindx<WANNUM;nindx++){
    for(bindx=0;bindx<nbdir;bindx++){

      csum[bindx].r=0.0;
      csum[bindx].i=0.0;

      for(k=0;k<kpt_num;k++){

        csum[bindx].r=csum[bindx].r+Mmnkb[k][bdirection[bindx]][nindx][nindx].r;
        csum[bindx].i=csum[bindx].i+Mmnkb[k][bdirection[bindx]][nindx][nindx].i;
      }
    }

    /* determine rguide */

    for(i=0;i<3;i++){
      svec[i]=0.0;
      for(j=0;j<3;j++){
        smat[i][j]=0.0;
      }
    }

    for(bindx=0; bindx<nbdir; bindx++){

      if(bindx<3){

        if (sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i) )<1.0e-8){
          xx[bindx]=0.0;
        }

        else{

	  if(csum[bindx].i<0.0){

	    /*
        printf("Y1 nindx=%2d bindx=%2d   %18.15f %18.15f %18.15f %18.15f\n",
	       nindx,bindx,csum[bindx].r,csum[bindx].i,
               csum[bindx].r/sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i)),
               acos(csum[bindx].r/sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i))));
	    */

            
   	    tmp = csum[bindx].r/sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i));
            tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
	    xx[bindx] = acos(tmp);

	    /*
        printf("Y2 nindx=%2d bindx=%2d   %18.15f\n",nindx,bindx,xx[bindx]);
	    */

	  }
          else{

	    /*
        printf("Y3 nindx=%2d bindx=%2d   %18.15f %18.15f %18.15f %18.15f\n",
	       nindx,bindx,csum[bindx].r,csum[bindx].i,
               csum[bindx].r/sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i)),
               acos(csum[bindx].r/sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i))) );
	    */

  	    tmp = csum[bindx].r/sqrt( fabs(csum[bindx].r*csum[bindx].r+csum[bindx].i*csum[bindx].i));
            tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
	    xx[bindx] = -acos(tmp);

	    /*
        printf("Y4 nindx=%2d bindx=%2d   %18.15f\n",nindx,bindx,xx[bindx]);
	    */

	  }
        }
      }

      else{

        xx0 = 0.0;
        for(j=0;j<3;j++){
          xx0 = xx0 + bvector[bdirection[bindx]][j]*rguide[j][nindx];
        }

        csumt.r = cos(xx0);
        csumt.i = sin(xx0); 

        det = csum[bindx].r*csumt.r - csum[bindx].i*csumt.i;
        brn = csum[bindx].r*csumt.i + csum[bindx].i*csumt.r;    

        if(sqrt( fabs(det*det+brn*brn) )<1.0e-8){
	  xx[bindx] = xx0 + 0.0;
        }

        else{
	  if(brn<0.0){

	    /*
        printf("Y5 nindx=%2d bindx=%2d   %18.15f %18.15f %18.15f %18.15f\n",
	       nindx,bindx,det,brn,
               det/sqrt( fabs(det*det+brn*brn)),
               acos(det/sqrt( fabs(det*det+brn*brn))));
	    */

  	    tmp = det/sqrt( fabs(det*det+brn*brn));
            tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
	    xx[bindx] = xx0 + acos(tmp);

	    /*
        printf("Y6 nindx=%2d bindx=%2d   %18.15f\n",nindx,bindx,xx[bindx]);
	    */


	  }else{


	    /*
        printf("Y7 nindx=%2d bindx=%2d   %18.15f %18.15f %18.15f %18.15f\n",
	       nindx,bindx,det,brn,
               det/sqrt( fabs(det*det+brn*brn)),
               acos(det/sqrt( fabs(det*det+brn*brn))));
	    */

	    tmp = det/sqrt( fabs(det*det+brn*brn));
            tmp = sgn(tmp)*MIN(fabs(tmp),0.99999999999999999);
	    xx[bindx] = xx0 - acos(tmp);

	    /*
        printf("Y8 nindx=%2d bindx=%2d   %18.15f\n",nindx,bindx,xx[bindx]);
	    */


	  }
        }
      }

      for(j=0;j<3;j++){
        for(i=0;i<3;i++){
          smat[j][i] = smat[j][i] + bvector[bdirection[bindx]][j]*bvector[bdirection[bindx]][i];
        }

	/*
        printf("A1 j=%2d bindx=%2d bdirection=%2d svec=%15.12f xx=%15.12f\n",
                   j,bindx,bdirection[bindx],svec[j],xx[bindx] );
	*/

         
        svec[j] = svec[j] + bvector[bdirection[bindx]][j]*xx[bindx];

	/*
        printf("A2 j=%2d bindx=%2d bdirection=%2d svec=%15.12f xx=%15.12f\n",
                   j,bindx,bdirection[bindx],svec[j],xx[bindx] );

	*/

      }

      if(bindx>=2){

        det = Invert3_mat(smat,sinv);    

        if(fabs(det)>1.0e-6){
          for(j=0; j<3; j++){

            rguide[j][nindx]=0.0;

            for(i=0;i<3;i++){

	      /*
              printf("B1 j=%2d i=%2d rguide=%15.12f svec=%15.12f det=%15.12f\n",
                      j,i,rguide[j][nindx],svec[i],det);
	      */

              rguide[j][nindx] = rguide[j][nindx]+sinv[j][i]*svec[i]/det;

	      /*
              printf("B2 j=%2d i=%2d rguide=%15.12f svec=%15.12f det=%15.12f\n",
                      j,i,rguide[j][nindx],svec[i],det);
	      */


            }
          }
        } 
      }

    }/* bindx */ 
/*
    if (myid==Host_ID){
      printf("rguide: ");
      for(j=0;j<3;j++){
        printf(" %10.5f  ",rguide[j][nindx]);
      }
      printf("\n");
    }
*/
  }/* nindx WANNUM */

  /* determine branch cut choice guided by rguide */

  for(k=0;k<kpt_num;k++){
    for(bindx=0;bindx<tot_bvector;bindx++){
      for(nindx=0;nindx<WANNUM;nindx++){
        sheet[k][bindx][nindx]=0.0;
        for(j=0;j<3;j++){
          sheet[k][bindx][nindx]=sheet[k][bindx][nindx]+bvector[bindx][j]*rguide[j][nindx];
        }
        csheet[k][bindx][nindx].r=cos(sheet[k][bindx][nindx]);
        csheet[k][bindx][nindx].i=sin(sheet[k][bindx][nindx]);
      }/* nindx */ 
    }/* b */
  }/* kpt */
  /* check */

  /* allocation of arrays */

  for(i=0;i<3;i++){
    free(sinv[i]);
  }
  free(sinv); 

  free(xx);
  free(csum);

}/* Center_Guide */


#pragma optimization_level 1
double Invert3_mat(double smat[3][3], double **sinv)
{
  double work[6][6],det,sum;
  int i,j,k,l,ll,kk;  

  for(i=0;i<2;i++){
    for(j=0;j<2;j++){
      for(k=0;k<3;k++){
	for(l=0;l<3;l++){
	  kk=3*i+k;
	  ll=3*j+l;  
	  work[kk][ll]=smat[k][l];
	}
      }
    }
  }
  det=0.0;
  for(i=0;i<3;i++){
    det=det+work[0][i]*work[1][i+1]*work[2][i+2];
  }
  for(i=3;i<6;i++){
    det=det-work[0][i]*work[1][i-1]*work[2][i-2];
  }
  for(j=0;j<3;j++){
    for(i=0;i<3;i++){
      sinv[j][i]= work[i+1][j+1]*work[i+2][j+2]-work[i+1][j+2]*work[i+2][j+1];
    }
  }

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      sum=0.0;
      for(l=0;l<3;l++){
	sum+=smat[i][l]*sinv[l][j];
      }
    }
  }

  return det;
}/* Invert3_mat */ 


#pragma optimization_level 1
void OutData_WF(char *inputfile, int rnum, int **rvect)
{
  int Mc_AN,Gc_AN,Cwan,NO0,spin,Nc;
  int kloop,Mh_AN,h_AN,Gh_AN,Rnh,Hwan;
  int NO1,l1,l2,l3,Nog,RnG,Nh,Rn;
  int orbit,GN,spe,i,j,k,i1,i2,i3,spinmax;
  int ncell,ncpy1,ncpy2,ncpy3,rvec1,rvec2,rvec3,GN1;
  int N3[4];
  double co,si,k1,k2,k3,kRn,ReCoef,ImCoef;
  double *RMO_Grid;
  double *IMO_Grid;
  double *RMO_Grid_tmp;
  double *IMO_Grid_tmp;
  dcomplex *MO_Grid;
  char file1[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;
  double wr,wi,wmod, tmpmod;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("outputting Wannier functions in cube files.\n");fflush(stdout);
  }

  /****************************************************
    allocation of arrays:

    dcomplex MO_Grid[TNumGrid*ncell];
    double RMO_Grid[TNumGrid*ncell];
    double IMO_Grid[TNumGrid*ncell];
    double RMO_Grid_tmp[TNumGrid*ncell];
    double IMO_Grid_tmp[TNumGrid*ncell];
  ****************************************************/

  ncpy1 = Wannier_Plot_SuperCells[0];
  ncpy2 = Wannier_Plot_SuperCells[1];
  ncpy3 = Wannier_Plot_SuperCells[2];

  ncell = (2*ncpy1+1)*(2*ncpy2+1)*(2*ncpy3+1);

  MO_Grid = (dcomplex*)malloc(sizeof(dcomplex)*TNumGrid*ncell);
  RMO_Grid = (double*)malloc(sizeof(double)*TNumGrid*ncell);
  IMO_Grid = (double*)malloc(sizeof(double)*TNumGrid*ncell);
  RMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid*ncell);
  IMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid*ncell);

  if      (SpinP_switch==0) spinmax = 0;
  else if (SpinP_switch==1) spinmax = 1;
  else if (SpinP_switch==3) spinmax = 1;

  /**************************************
        output of Wannier functions
  **************************************/

  for (spin=0; spin<=spinmax; spin++){

    for (orbit=0; orbit<Wannier_Func_Num; orbit++){ 

      for (GN=0; GN<TNumGrid*ncell; GN++){
	RMO_Grid_tmp[GN] = 0.0;
	IMO_Grid_tmp[GN] = 0.0;
      }

      for (kloop=0; kloop<rnum; kloop++){

	rvec1 = rvect[0][kloop];
	rvec2 = rvect[1][kloop];
	rvec3 = rvect[2][kloop];

	if ( abs(rvec1)<=ncpy1 && abs(rvec2)<=ncpy2 && abs(rvec3)<=ncpy3){

	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    NO0 = Spe_Total_CNO[Cwan];

	    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	      GN = GridListAtom[Mc_AN][Nc];
	      Rn = CellListAtom[Mc_AN][Nc];
              GN2N(GN, N3);

	      l1 = atv_ijk[Rn][1];
	      l2 = atv_ijk[Rn][2];
	      l3 = atv_ijk[Rn][3];

              if (   abs(rvec1+l1)<=ncpy1 
                  && abs(rvec2+l2)<=ncpy2 
                  && abs(rvec3+l3)<=ncpy3 ){

                i1 = N3[1] + (rvec1+l1+ncpy1)*Ngrid1;
                i2 = N3[2] + (rvec2+l2+ncpy2)*Ngrid2;
                i3 = N3[3] + (rvec3+l3+ncpy3)*Ngrid3;
                 
		GN1 = i1*Ngrid2*(2*ncpy2+1)*Ngrid3*(2*ncpy3+1) + i2*Ngrid3*(2*ncpy3+1) + i3;

		for (i=0; i<NO0; i++){

		  ReCoef = Wannier_Coef[kloop][spin][orbit][Gc_AN][i].r; 
		  ImCoef = Wannier_Coef[kloop][spin][orbit][Gc_AN][i].i;

		  RMO_Grid_tmp[GN1] += ReCoef*Orbs_Grid[Mc_AN][Nc][i];/*AITUNE*/
		  IMO_Grid_tmp[GN1] += ImCoef*Orbs_Grid[Mc_AN][Nc][i];/*AITUNE*/

		}
	      }

	    } /* Nc */
	  } /* Mc_AN */

	  MPI_Reduce(&RMO_Grid_tmp[0], &RMO_Grid[0], TNumGrid*ncell, MPI_DOUBLE,
		     MPI_SUM, Host_ID, mpi_comm_level1);

	  MPI_Reduce(&IMO_Grid_tmp[0], &IMO_Grid[0], TNumGrid*ncell, MPI_DOUBLE,
		     MPI_SUM, Host_ID, mpi_comm_level1);
	}
      } /* kloop */

      /* fix the global phase by setting the wannier to be real at the point where
	 it has max. moduls */

      if (myid==Host_ID){ 

	wr   = -99999.0;
	wi   = -99999.0;
	wmod = -99999.0;

	for (GN=0; GN<TNumGrid*ncell; GN++){

	  MO_Grid[GN].r = RMO_Grid[GN];
	  MO_Grid[GN].i = IMO_Grid[GN];

	  tmpmod=sqrt( fabs(MO_Grid[GN].r*MO_Grid[GN].r+MO_Grid[GN].i*MO_Grid[GN].i)); 

	  if(wmod<tmpmod){
	    wmod = tmpmod;
	    wr = MO_Grid[GN].r;
	    wi = MO_Grid[GN].i;
	  }
	}

	if (fabs(wmod)<1e-50){
	  printf("The mode of Wannier Function is nearly zero. Please check the calculation.\n");

	  MPI_Finalize(); 
	  exit(0);
	}

	wr=wr/wmod;
	wi=wi/wmod;

	for (GN=0; GN<TNumGrid*ncell; GN++){
	  ReCoef =  MO_Grid[GN].r*wr+MO_Grid[GN].i*wi;
	  ImCoef = -MO_Grid[GN].r*wi+MO_Grid[GN].i*wr;
	  MO_Grid[GN].r = ReCoef/(wr*wr+wi*wi);
	  MO_Grid[GN].i = ImCoef/(wr*wr+wi*wi);
	}

	/* check the ratio between Im and Re part of wannier function */

	wr=0.0;

	for (GN=0; GN<TNumGrid*ncell; GN++){

	  if(fabs(MO_Grid[GN].r)>0.001){

	    wi=fabs(MO_Grid[GN].i/MO_Grid[GN].r);

	    if(wr<wi){
	      wr=wi;
	    }
	  }
	}

	/* printf("Wannier Function %i maximum Im/Re Ratio = %10.5f\n",orbit+1,wr); */            

	/* output the real part of Wannier functions */

	sprintf(file1,"%s%s.mlwf%i_%i_r.cube",
		filepath,filename,spin,orbit);
  
	if ((fp = fopen(file1,"w")) != NULL){

#ifdef xt3
	  setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	  {
	    int ct_AN;
	    int spe; 

	    fprintf(fp," SYS1\n SYS1\n");

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    atomnum*ncell,
		    Grid_Origin[1] - ncpy1*tv[1][1] - ncpy2*tv[2][1] - ncpy3*tv[3][1],
		    Grid_Origin[2] - ncpy1*tv[1][2] - ncpy2*tv[2][2] - ncpy3*tv[3][2],
		    Grid_Origin[3] - ncpy1*tv[1][3] - ncpy2*tv[2][3] - ncpy3*tv[3][3]);

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    Ngrid1*(2*ncpy1+1),
		    gtv[1][1],gtv[1][2],gtv[1][3]);

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    Ngrid2*(2*ncpy2+1),
                    gtv[2][1],gtv[2][2],gtv[2][3]);

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    Ngrid3*(2*ncpy3+1),
                    gtv[3][1],gtv[3][2],gtv[3][3]);

	    for (i=-ncpy1; i<=ncpy1; i++){
	      for (j=-ncpy2; j<=ncpy2; j++){
		for (k=-ncpy3; k<=ncpy3; k++){

		  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

		    spe = WhatSpecies[ct_AN];

		    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf%12.6lf\n",
			    Spe_WhatAtom[spe],
			    Spe_Core_Charge[spe]-InitN_USpin[ct_AN]-InitN_DSpin[ct_AN],
			    Gxyz[ct_AN][1]
                            + (double)i*tv[1][1] + (double)j*tv[2][1] + (double)k*tv[3][1],
			    Gxyz[ct_AN][2]
                            + (double)i*tv[1][2] + (double)j*tv[2][2] + (double)k*tv[3][2],
			    Gxyz[ct_AN][3]
                            + (double)i*tv[1][3] + (double)j*tv[2][3] + (double)k*tv[3][3]);
		  }
		}
	      }
	    }

	  }


	  {
	    int i1,i2,i3;
	    int GN;
	    int cmd;

	    for (i1=0; i1<Ngrid1*(2*ncpy1+1); i1++){
	      for (i2=0; i2<Ngrid2*(2*ncpy2+1); i2++){
		for (i3=0; i3<Ngrid3*(2*ncpy3+1); i3++){

		  GN = i1*Ngrid2*(2*ncpy2+1)*Ngrid3*(2*ncpy3+1) + i2*Ngrid3*(2*ncpy3+1) + i3;

		  fprintf(fp,"%13.3E",MO_Grid[GN].r);

		  if ((i3+1)%6==0) { fprintf(fp,"\n"); }

		}

		/* avoid double \n\n when Ngrid3%6 == 0  */
		if ( (Ngrid3*(2*ncpy3+1))%6!=0) fprintf(fp,"\n");

	      }
	    }
	  }              

	  fclose(fp);

	}
	else{
	  printf("Failure of saving MOs\n");
	}

	/* output the imaginary part of Wannier functions */

	sprintf(file1,"%s%s.mlwf%i_%i_i.cube", 
		filepath,filename,spin,orbit);

	if ((fp = fopen(file1,"w")) != NULL){

#ifdef xt3
	  setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	  {
	    int ct_AN;
	    int spe; 

	    fprintf(fp," SYS1\n SYS1\n");

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    atomnum*ncell,
		    Grid_Origin[1] - ncpy1*tv[1][1] - ncpy2*tv[2][1] - ncpy3*tv[3][1],
		    Grid_Origin[2] - ncpy1*tv[1][2] - ncpy2*tv[2][2] - ncpy3*tv[3][2],
		    Grid_Origin[3] - ncpy1*tv[1][3] - ncpy2*tv[2][3] - ncpy3*tv[3][3]);

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    Ngrid1*(2*ncpy1+1),
		    gtv[1][1],gtv[1][2],gtv[1][3]);

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    Ngrid2*(2*ncpy2+1),
                    gtv[2][1],gtv[2][2],gtv[2][3]);

	    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf\n",
		    Ngrid3*(2*ncpy3+1),
                    gtv[3][1],gtv[3][2],gtv[3][3]);

	    for (i=-ncpy1; i<=ncpy1; i++){
	      for (j=-ncpy2; j<=ncpy2; j++){
		for (k=-ncpy3; k<=ncpy3; k++){

		  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

		    spe = WhatSpecies[ct_AN];

		    fprintf(fp,"%5d%12.6lf%12.6lf%12.6lf%12.6lf\n",
			    Spe_WhatAtom[spe],
			    Spe_Core_Charge[spe]-InitN_USpin[ct_AN]-InitN_DSpin[ct_AN],
			    Gxyz[ct_AN][1]
                            + (double)i*tv[1][1] + (double)j*tv[2][1] + (double)k*tv[3][1],
			    Gxyz[ct_AN][2]
                            + (double)i*tv[1][2] + (double)j*tv[2][2] + (double)k*tv[3][2],
			    Gxyz[ct_AN][3]
                            + (double)i*tv[1][3] + (double)j*tv[2][3] + (double)k*tv[3][3]);
		  }
		}
	      }
	    }

	  }


	  {
	    int i1,i2,i3;
	    int GN;
	    int cmd;

	    for (i1=0; i1<Ngrid1*(2*ncpy1+1); i1++){
	      for (i2=0; i2<Ngrid2*(2*ncpy2+1); i2++){
		for (i3=0; i3<Ngrid3*(2*ncpy3+1); i3++){

		  GN = i1*Ngrid2*(2*ncpy2+1)*Ngrid3*(2*ncpy3+1) + i2*Ngrid3*(2*ncpy3+1) + i3;

		  fprintf(fp,"%13.3E",MO_Grid[GN].i);

		  if ((i3+1)%6==0) { fprintf(fp,"\n"); }

		}

		/* avoid double \n\n when Ngrid3%6 == 0  */
		if ( (Ngrid3*(2*ncpy3+1))%6!=0) fprintf(fp,"\n");

	      }
	    }
	  }              

	  fclose(fp);
	}
	else{
	  printf("Failure of saving MOs\n");
	}

      } /* if (myid==Host_ID) */

    } /* orbit */
  } /* spin */ 

  /****************************************************
    freeing of arrays:

    dcomplex MO_Grid[TNumGrid];
    double RMO_Grid[TNumGrid];
    double IMO_Grid[TNumGrid];
    double RMO_Grid_tmp[TNumGrid];
    double IMO_Grid_tmp[TNumGrid];
  ****************************************************/

  free(MO_Grid);
  free(RMO_Grid);
  free(IMO_Grid);
  free(RMO_Grid_tmp);
  free(IMO_Grid_tmp);
}



#pragma optimization_level 1
void Calc_SpinDirection_WF(int rnum, int **rvect)
{
  int Mc_AN,Gc_AN,Cwan,NO0,spin,Nc;
  int kloop,Mh_AN,h_AN,Gh_AN,Rnh,Hwan;
  int NO1,l1,l2,l3,Nog,RnG,Nh,Rn;
  int orbit,GN,spe,i,j,k,i1,i2,i3,spinmax;
  int ncpy1,ncpy2,ncpy3,rvec1,rvec2,rvec3,GN1;
  int N3[4],n1,n2,n3;
  double co,si,k1,k2,k3,kRn,ReCoef,ImCoef;
  double *RMO_Grid0;
  double *IMO_Grid0;
  double *RMO_Grid1;
  double *IMO_Grid1;
  double *RMO_Grid0_tmp;
  double *IMO_Grid0_tmp;
  double *RMO_Grid1_tmp;
  double *IMO_Grid1_tmp;
  double Re_rho00,Re_rho01,Re_rho10,Re_rho11;
  double Im_rho00,Im_rho01,Im_rho10,Im_rho11;
  double sx,sy,sz;
  int numprocs,myid,ID;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("Calculate spin direction of WF.\n");fflush(stdout);
  }

  /****************************************************
    allocation of arrays:
  ****************************************************/

  ncpy1 = 3;
  ncpy2 = 3;
  ncpy3 = 3;

  RMO_Grid0 = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid0 = (double*)malloc(sizeof(double)*TNumGrid);
  RMO_Grid1 = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid1 = (double*)malloc(sizeof(double)*TNumGrid);

  RMO_Grid0_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid0_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  RMO_Grid1_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid1_tmp = (double*)malloc(sizeof(double)*TNumGrid);

  /**************************************
        output of Wannier functions
  **************************************/

  for (orbit=0; orbit<Wannier_Func_Num; orbit++){ 

    Re_rho00 = 0.0;
    Im_rho00 = 0.0;
    Re_rho01 = 0.0;
    Im_rho01 = 0.0;
    Re_rho10 = 0.0;
    Im_rho10 = 0.0;
    Re_rho11 = 0.0;
    Im_rho11 = 0.0;

    for (n1=-ncpy1; n1<=ncpy1; n1++){
      for (n2=-ncpy2; n2<=ncpy2; n2++){
	for (n3=-ncpy3; n3<=ncpy3; n3++){

	  for (GN=0; GN<TNumGrid; GN++){
	    RMO_Grid0_tmp[GN] = 0.0;
	    IMO_Grid0_tmp[GN] = 0.0;
	    RMO_Grid1_tmp[GN] = 0.0;
	    IMO_Grid1_tmp[GN] = 0.0;
	  }

	  for (kloop=0; kloop<rnum; kloop++){

	    rvec1 = rvect[0][kloop];
	    rvec2 = rvect[1][kloop];
	    rvec3 = rvect[2][kloop];

	    if ( abs(rvec1)<=ncpy1 && abs(rvec2)<=ncpy2 && abs(rvec3)<=ncpy3){

	      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

		Gc_AN = M2G[Mc_AN];    
		Cwan = WhatSpecies[Gc_AN];
		NO0 = Spe_Total_CNO[Cwan];

		for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

		  GN = GridListAtom[Mc_AN][Nc];
		  Rn = CellListAtom[Mc_AN][Nc];
		  GN2N(GN, N3);

		  l1 = atv_ijk[Rn][1];
		  l2 = atv_ijk[Rn][2];
		  l3 = atv_ijk[Rn][3];

		  if (   n1==(rvec1+l1)
		      && n2==(rvec2+l2)
		      && n3==(rvec3+l3) ){

		    i1 = N3[1];
		    i2 = N3[2];
		    i3 = N3[3];

		    GN1 = i1*Ngrid2*Ngrid3 + i2*Ngrid3 + i3;

		    for (i=0; i<NO0; i++){

                      spin = 0; 
		      ReCoef = Wannier_Coef[kloop][spin][orbit][Gc_AN][i].r; 
		      ImCoef = Wannier_Coef[kloop][spin][orbit][Gc_AN][i].i;

		      RMO_Grid0_tmp[GN1] += ReCoef*Orbs_Grid[Mc_AN][Nc][i];/*AITUNE*/
		      IMO_Grid0_tmp[GN1] += ImCoef*Orbs_Grid[Mc_AN][Nc][i];/*AITUNE*/

                      spin = 1; 
		      ReCoef = Wannier_Coef[kloop][spin][orbit][Gc_AN][i].r; 
		      ImCoef = Wannier_Coef[kloop][spin][orbit][Gc_AN][i].i;

		      RMO_Grid1_tmp[GN1] += ReCoef*Orbs_Grid[Mc_AN][Nc][i];/*AITUNE*/
		      IMO_Grid1_tmp[GN1] += ImCoef*Orbs_Grid[Mc_AN][Nc][i];/*AITUNE*/
		    }
		  }

		} /* Nc */
	      } /* Mc_AN */

	    }
	  } /* kloop */

	  MPI_Reduce(&RMO_Grid0_tmp[0], &RMO_Grid0[0], TNumGrid, MPI_DOUBLE,
		         MPI_SUM, Host_ID, mpi_comm_level1);

	  MPI_Reduce(&IMO_Grid0_tmp[0], &IMO_Grid0[0], TNumGrid, MPI_DOUBLE,
		         MPI_SUM, Host_ID, mpi_comm_level1);

	  MPI_Reduce(&RMO_Grid1_tmp[0], &RMO_Grid1[0], TNumGrid, MPI_DOUBLE,
		         MPI_SUM, Host_ID, mpi_comm_level1);

	  MPI_Reduce(&IMO_Grid1_tmp[0], &IMO_Grid1[0], TNumGrid, MPI_DOUBLE,
		         MPI_SUM, Host_ID, mpi_comm_level1);

          /* calculate a part of the density matrix */

	  if (myid==Host_ID){
	    for (GN=0; GN<TNumGrid; GN++){

	      Re_rho00 += RMO_Grid0[GN]*RMO_Grid0[GN] + IMO_Grid0[GN]*IMO_Grid0[GN];
	      Im_rho01 += IMO_Grid0[GN]*RMO_Grid1[GN] - RMO_Grid0[GN]*IMO_Grid1[GN];
	      Re_rho11 += RMO_Grid1[GN]*RMO_Grid1[GN] + IMO_Grid1[GN]*IMO_Grid1[GN];

	      /*
	      Re_rho01 += RMO_Grid0[GN]*RMO_Grid1[GN] + IMO_Grid0[GN]*IMO_Grid1[GN];
	      Re_rho10 += RMO_Grid1[GN]*RMO_Grid0[GN] + IMO_Grid1[GN]*IMO_Grid0[GN];
	      Im_rho10 += IMO_Grid1[GN]*RMO_Grid0[GN] - RMO_Grid1[GN]*IMO_Grid0[GN];
	      */

	    }
	  }

	} /* n1 */
      } /* n2 */
    } /* n3 */

    if (myid==Host_ID){   

      Re_rho00 *= GridVol;
      Im_rho01 *= GridVol;
      Re_rho11 *= GridVol;

      printf("WF%3d Re_rho00=%15.12f Re_rho11=%15.12f sum=%15.12f\n",
              orbit,Re_rho00,Re_rho11,Re_rho00+Re_rho11);

      sx = 2.0*Re_rho01;
      sy = 2.0*Im_rho01;
      sz = Re_rho00 - Re_rho11;
      printf("WF%3d  sx=%15.12f %15.12f %15.12f\n",orbit,sx,sy,sz);fflush(stdout);
    }

  } /* orbit */

  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(RMO_Grid0);
  free(IMO_Grid0);
  free(RMO_Grid1);
  free(IMO_Grid1);

  free(RMO_Grid0_tmp);
  free(IMO_Grid0_tmp);
  free(RMO_Grid1_tmp);
  free(IMO_Grid1_tmp);
}





#pragma optimization_level 1
void Overlap_Band_Wannier(double ****fOLP,
			  dcomplex **S,int *MP,
			  double k1, double k2, double k3)
{
  int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  int l1,l2,l3,Rn,n2;
  double **S1,**S2;
  double kRn,si,co,s;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
                       Allocation
  ****************************************************/

  n2 = NUM + 2;

  S1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    S1[i] = (double*)malloc(sizeof(double)*n2);
  }

  S2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    S2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                       set overlap
  ****************************************************/

  S[0][0].r = NUM;

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S1[i][j] = 0.0;
      S2[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  s = fOLP[GA_AN][LB_AN][i][j];
	  S1[Anum+i][Bnum+j] += s*co;
	  S2[Anum+i][Bnum+j] += s*si;
	}
      }
    }
  }

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S[i][j].r =  S1[i][j];
      S[i][j].i =  S2[i][j];
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(S1[i]);
    free(S2[i]);
  }
  free(S1);
  free(S2);

}



#pragma optimization_level 1
void Hamiltonian_Band_Wannier(double ****RH, dcomplex **H, int *MP,
			      double k1, double k2, double k3)
{
  int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  int l1,l2,l3,Rn,n2;
  double **H1,**H2;
  double kRn,si,co,h;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
                       Allocation
  ****************************************************/

  n2 = NUM + 2;

  H1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H1[i] = (double*)malloc(sizeof(double)*n2);
  }

  H2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                    set Hamiltonian
  ****************************************************/

  H[0][0].r = 2.0*NUM;
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H1[i][j] = 0.0;
      H2[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  h = RH[GA_AN][LB_AN][i][j];
	  H1[Anum+i][Bnum+j] += h*co;
	  H2[Anum+i][Bnum+j] += h*si;
	}
      }
    }
  }

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H[i][j].r = H1[i][j];
      H[i][j].i = H2[i][j];
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(H1[i]);
    free(H2[i]);
  }
  free(H1);
  free(H2);
}



#pragma optimization_level 1
void Hamiltonian_Band_NC_Wannier(double *****RH, double *****IH,
				 dcomplex **H, int *MP,
				 double k1, double k2, double k3)
{
  int i,j,k,wanA,wanB,tnoA,tnoB,Anum,Bnum;
  int NUM,GA_AN,LB_AN,GB_AN;
  int l1,l2,l3,Rn,n2;
  double **H11r,**H11i;
  double **H22r,**H22i;
  double **H12r,**H12i;
  double kRn,si,co,h;

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;
  n2 = NUM + 2;

  /*******************************************
   allocation of H11r, H11i,
                 H22r, H22i,
                 H12r, H12i
  *******************************************/

  H11r = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H11r[i] = (double*)malloc(sizeof(double)*n2);
  }

  H11i = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H11i[i] = (double*)malloc(sizeof(double)*n2);
  }

  H22r = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H22r[i] = (double*)malloc(sizeof(double)*n2);
  }

  H22i = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H22i[i] = (double*)malloc(sizeof(double)*n2);
  }

  H12r = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H12r[i] = (double*)malloc(sizeof(double)*n2);
  }

  H12i = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H12i[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                    set Hamiltonian
  ****************************************************/

  H[0][0].r = 2.0*NUM;
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H11r[i][j] = 0.0;
      H11i[i][j] = 0.0;
      H22r[i][j] = 0.0;
      H22i[i][j] = 0.0;
      H12r[i][j] = 0.0;
      H12i[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){

	  H11r[Anum+i][Bnum+j] += co*RH[0][GA_AN][LB_AN][i][j] -  si*IH[0][GA_AN][LB_AN][i][j];
	  H11i[Anum+i][Bnum+j] += si*RH[0][GA_AN][LB_AN][i][j] +  co*IH[0][GA_AN][LB_AN][i][j];
	  H22r[Anum+i][Bnum+j] += co*RH[1][GA_AN][LB_AN][i][j] -  si*IH[1][GA_AN][LB_AN][i][j];
	  H22i[Anum+i][Bnum+j] += si*RH[1][GA_AN][LB_AN][i][j] +  co*IH[1][GA_AN][LB_AN][i][j];
	  H12r[Anum+i][Bnum+j] += co*RH[2][GA_AN][LB_AN][i][j] - si*(RH[3][GA_AN][LB_AN][i][j]
								     + IH[2][GA_AN][LB_AN][i][j]);
	  H12i[Anum+i][Bnum+j] += si*RH[2][GA_AN][LB_AN][i][j] + co*(RH[3][GA_AN][LB_AN][i][j]
								     + IH[2][GA_AN][LB_AN][i][j]);
        }
      }

    }
  }

  /******************************************************
    the full complex matrix of H
  ******************************************************/

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H[i    ][j    ].r =  H11r[i][j];
      H[i    ][j    ].i =  H11i[i][j];
      H[i+NUM][j+NUM].r =  H22r[i][j];
      H[i+NUM][j+NUM].i =  H22i[i][j];
      H[i    ][j+NUM].r =  H12r[i][j];
      H[i    ][j+NUM].i =  H12i[i][j];
      H[j+NUM][i    ].r =  H[i][j+NUM].r;
      H[j+NUM][i    ].i = -H[i][j+NUM].i;
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(H11r[i]);
  }
  free(H11r);

  for (i=0; i<n2; i++){
    free(H11i[i]);
  }
  free(H11i);

  for (i=0; i<n2; i++){
    free(H22r[i]);
  }
  free(H22r);

  for (i=0; i<n2; i++){
    free(H22i[i]);
  }
  free(H22i);

  for (i=0; i<n2; i++){
    free(H12r[i]);
  }
  free(H12r);

  for (i=0; i<n2; i++){
    free(H12i[i]);
  }
  free(H12i);

}
