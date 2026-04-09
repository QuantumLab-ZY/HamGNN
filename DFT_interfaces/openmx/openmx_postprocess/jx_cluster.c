/**********************************************************************

jx.c:

This program code calculates spin-spin interaction
coupling constant J between the selected two atoms

Log of jx.c:
   30/Aug/2003  Released by Myung Joon Han (supervised by Prof. J. Yu)
    7/Dec/2003  Modified by Taisuke Ozaki
   03/Mar/2011  Modified by Fumiyuki Ishii for MPI
   02/May/2018  Modified by Asako Terasawa for blas
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <string.h>
// #include <malloc/malloc.h>
// #include <assert.h>

#include "read_scfout.h"
#include "lapack_prototypes.h"
#include "f77func.h"
#include "mpi.h"

#include "jx.h"
#include "jx_tools.h"
#include "jx_config.h"
#include "jx_total_mem.h"
//#include "jx_param_species_atoms.h"
//#include "jx_minimal_basis.h"

#define Host_ID       0         /* ID of the host CPU in MPI */

void Jij_cluster_partial_mat_calc(
  int F_TNumOrbs0, int NumOrbs_1st, int NumOrbs_2nd,
  double ***Hks_1st, double ***Hks_2nd,
  double ***Ci, double ***Cj, double **ko,
  double *J_ij);

#pragma optimization_level 0
void Jij_cluster(int argc, char *argv[], MPI_Comm comm1)
{

  static int i,j,l,n;

  static int ct_AN,h_AN,Gh_AN;
  static int spin,Rn;

  int II,JJ ;        /* dummy variables for loop    */
  int Anum;          /* temporary variable          */
  int T_NumOrbs;
  int optEV ;        /* variable to be used for option of eigenvector-printing */
  int *MP ;         /* Full system version of MP */
  int end_switch;    /* switch in the evalusion loop of J */
  double ***FullHks; /* full Hamiltonian */
  double **FullOLP;  /* full overlap     */

  int id_ij;

  static int NumOrbs_1st,NumOrbs_2nd;
  static int MP_1st,MP_2nd;
  static double ***Hks_1st,***Hks_2nd;
  static double ***Coes_1st,***Coes_2nd;

  /* variable & arrays for PART-2; same with that of Cluster_DFT.c */
  static double **ko, *M1;
  static double **B, ***C, **D;
//  static double sum,sum1;
  static double J_ij;
  double sum;

//  MPI_Comm comm1;
  int myid,numprocs;

  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  if (1<numprocs){
    if (myid==Host_ID){
      printf("\n MPI parallelization is not supported for the cluster mode.\n");
      printf(" Please try it again using a single core.\n\n");

    }
    MPI_Finalize();
    exit(0);
  }

  /*************************************************************************
     Calculation flow in the evaluation of J:

     PART-1 : set full Hamiltonian and overlap
     PART-2 : the generalized eigenvalue problem HC = ESC
     PART-3 : Calculation of J
  **************************************************************************/

  /*************************************************************************
     PART-1 :  set full Hamiltonian and overlap
  **************************************************************************/

  printf(" \nEvaluation of J based on cluster calculation\n");

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  T_NumOrbs = 0;
  for (i=0; i<atomnum; i++){
    ct_AN = i+1;
    MP[i+1] = T_NumOrbs+1;
    T_NumOrbs = T_NumOrbs + Total_NumOrbs[ct_AN];
  }

  FullHks = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    FullHks[spin] = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
    for (i=0; i<=T_NumOrbs; i++){
      FullHks[spin][i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
    }
  }

  FullOLP = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    FullOLP[i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<=T_NumOrbs; i++){
      for (j=0; j<=T_NumOrbs; j++){
        FullHks[spin][i][j] = 0.0;
      }
    }
  }

  for (i=0; i<=T_NumOrbs; i++){
    for (j=0; j<=T_NumOrbs; j++){
      FullOLP[i][j] = 0.0;
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    for (II=0; II<atomnum; II++){
      for (JJ=0; JJ<atomnum; JJ++){
        ct_AN = II+1 ;
        for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
          Gh_AN = natn[ct_AN][h_AN];
          if (Gh_AN == JJ+1){
            for (i=0; i<Total_NumOrbs[ct_AN]; i++){
              for (j=0; j<Total_NumOrbs[Gh_AN]; j++){
                FullHks[spin][i+MP[II+1]][j+MP[JJ+1]]
                  = Hks[spin][ct_AN][h_AN][i][j];
                FullOLP[i+MP[II+1]][j+MP[JJ+1]]
                  = OLP[ct_AN][h_AN][i][j];
              }
            }
          }
        }
      }
    }
  }

  /*************************************************************************
     PART-2 : solve the generalized eigenvalue problem HC = ESC
  **************************************************************************/

  /*******************************************
   allocation of arrays:

   double ko[SpinP_switch+1][(T_NumOrbs+1)];
   double M1[(T_NumOrbs+1)];
   double B[(T_NumOrbs+1)][(T_NumOrbs+1)];
   double C[SpinP_switch+1][(T_NumOrbs+1)][(T_NumOrbs+1)];
   double D[(T_NumOrbs+1)][(T_NumOrbs+1)];
  ********************************************/

  ko = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    ko[spin] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
  }
  M1 = (double*)malloc(sizeof(double)*(T_NumOrbs+1));

  B = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    B[i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
  }

  C = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    C[spin] = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
    for (i=0; i<=T_NumOrbs; i++){
      C[spin][i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
    }
  }

  D = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    D[i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
  }

  /*******************************************
   diagonalize the overlap matrix

   first
   FullOLP -> OLP matrix
   after call Eigen_lapack
   FullOLP -> eigenvectors of OLP matrix
  ********************************************/

  /* printf(" Diagonalize the overlap matrix\n"); */

  n = T_NumOrbs;

  Eigen_lapack(FullOLP,ko[0],n);

  for (l=1; l<=T_NumOrbs; l++) M1[l] = 1.0/sqrt(ko[0][l]);

  /****************************************************
   Calculations of eigenvalues for up and down spins
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    /* printf(" Diagonalize the Hamiltonian for spin=%2d\n",spin); */

    matmul_double_lapack("N","N",n,n,n,FullHks[spin],FullOLP,D);
    matmul_double_lapack("T","N",n,n,n,FullOLP,D,B);
    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
        D[i][j] = M1[i]*B[i][j]*M1[j];
      }
    }

    Eigen_lapack(D,ko[spin],n);

    /****************************************************
        Transformation to the original eigen vectors.
                          NOTE 244P
    ****************************************************/

    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
        sum = 0.0;
        for (l=1; l<=n; l++){
          sum += FullOLP[i][l]*M1[l]*D[l][j];
        }
        C[spin][i][j] = sum;
      }
    }
  }

  /*************************************************************************
     PART-3 : Calculation of J
     1) V_i,alpha,beta = < i,alpha | V_i | j,beta >
                       = 0.5 * ( H_i,alpha,i,beta(0) - H_i,alpha,i,beta(1) )
     2) V'_i = SUM_alpha,beta{ C_i,alpha * V_alpha,beta * C_i,beta }
     3) calculation of J from V'
  **************************************************************************/

  if (myid==Host_ID){
    printf("\n");
    printf("    i     j            J [meV]            J [mRy]\n");
    printf("-------------------------------------------------\n");
  }

  for (id_ij=0; id_ij<num_ij_total; id_ij++){

    NumOrbs_1st=Total_NumOrbs[id_atom[id_ij][0]];
    NumOrbs_2nd=Total_NumOrbs[id_atom[id_ij][1]];

    Hks_1st = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      Hks_1st[spin] = (double**)malloc(sizeof(double*)*(NumOrbs_1st+1));
        for (i=0; i<=NumOrbs_1st; i++){
        Hks_1st[spin][i] = (double*)malloc(sizeof(double)*(NumOrbs_1st+1));
      }
    }

    Coes_1st = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      Coes_1st[spin] = (double**)malloc(sizeof(double*)*(NumOrbs_1st+1));
      for (i=0; i<=NumOrbs_1st; i++){
        Coes_1st[spin][i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
      }
    }

    Hks_2nd = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      Hks_2nd[spin] = (double**)malloc(sizeof(double*)*(NumOrbs_2nd+1));
      for (i=0; i<=NumOrbs_2nd; i++){
        Hks_2nd[spin][i] = (double*)malloc(sizeof(double)*(NumOrbs_2nd+1));
      }
    }

    Coes_2nd = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      Coes_2nd[spin] = (double**)malloc(sizeof(double*)*(NumOrbs_2nd+1));
      for (i=0; i<=NumOrbs_2nd; i++){
        Coes_2nd[spin][i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
      }
    }

    ct_AN=id_atom[id_ij][0];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      Rn = ncn[ct_AN][h_AN];
      if (Gh_AN == id_atom[id_ij][0] && Rn==0) break;
    }

    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<NumOrbs_1st; i++){
        for (j=0; j<NumOrbs_1st; j++){
          Hks_1st[spin][i][j] = Hks[spin][ct_AN][h_AN][i][j];
        }
      }
    }

    ct_AN=id_atom[id_ij][1];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      Rn = ncn[ct_AN][h_AN];
      if (Gh_AN == id_atom[id_ij][1] && Rn==0) break;
    }
    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<NumOrbs_2nd; i++){
        for (j=0; j<NumOrbs_2nd; j++){
          Hks_2nd[spin][i][j] = Hks[spin][ct_AN][h_AN][i][j];
        }
      }
    }

    MP_1st = MP[id_atom[id_ij][0]];
    MP_2nd = MP[id_atom[id_ij][1]];
    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<NumOrbs_1st; i++){
        for (j=0; j<T_NumOrbs; j++){
          Coes_1st[spin][1+i][1+j] = C[spin][MP_1st+i][1+j];
        }
      }
      for (i=0; i<NumOrbs_2nd; i++){
        for (j=0; j<T_NumOrbs; j++){
          Coes_2nd[spin][1+i][1+j] = C[spin][MP_2nd+i][1+j];
        }
      }
    }

    Jij_cluster_partial_mat_calc(
      T_NumOrbs, NumOrbs_1st, NumOrbs_2nd,
      Hks_1st, Hks_2nd, Coes_1st, Coes_2nd, ko, &J_ij);

    printf(
      "%5i %5i %19.12f %19.12f\n",
      id_atom[id_ij][0],id_atom[id_ij][1],J_ij*27.211386*1000,J_ij*2*1000);

    /*
  MPI_Finalize();
  exit(0);
    */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (j=0; j<NumOrbs_1st+1; j++){
        free(Hks_1st[spin][j]);
      }
      free(Hks_1st[spin]);
    }
    free(Hks_1st);
    for (spin=0; spin<=SpinP_switch; spin++){
      for (j=0; j<NumOrbs_2nd+1; j++){
        free(Hks_2nd[spin][j]);
      }
      free(Hks_2nd[spin]);
    }
    free(Hks_2nd);

    for (spin=0; spin<=SpinP_switch; spin++){
      for (j=0; j<=NumOrbs_1st; j++){
        free(Coes_1st[spin][j]);
      }
      free(Coes_1st[spin]);
    }
    free(Coes_1st);

    for (spin=0; spin<=SpinP_switch; spin++){
      for (j=0; j<=NumOrbs_2nd; j++){
        free(Coes_2nd[spin][j]);
      }
      free(Coes_2nd[spin]);
    }
    free(Coes_2nd);
  }

  free(MP);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<=T_NumOrbs; i++){
      free(FullHks[spin][i]);
    }
    free(FullHks[spin]);
  }
  free(FullHks);
  for (i=0; i<=T_NumOrbs; i++){
    free(FullOLP[i]);
  }
  free(FullOLP);
  for (spin=0; spin<=SpinP_switch; spin++){
    free(ko[spin]);
  }
  free(ko);
  free(M1);
  for (i=0; i<=T_NumOrbs; i++){
    free(B[i]);
  }
  free(B);
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<=T_NumOrbs; i++){
      free(C[spin][i]);
    }
    free(C[spin]);
  }
  free(C);
  for (i=0; i<=T_NumOrbs; i++){
    free(D[i]);
  }
  free(D);
}

void Jij_cluster_partial_mat_calc(
  int T_NumOrbs, int NumOrbs_1st, int NumOrbs_2nd,
  double ***Hks_1st, double ***Hks_2nd,
  double ***Ci, double ***Cj, double **ko, double *J_ij)
{

  static int spin,spin1,spin2;   /* dummy variables for loop                               */
  static int i,j;

  static double **Fftn;        /* Fermi function */
  static double kB=0.000003166813628;  /* Boltzman constant (Hatree/K) */
  static double dFftn;
  static double dko;

  static double **mat_temp;
  static double **Vi,**Vj;         /* array for V_i and V_j */
  static double ****VVi,****VVj;         /* array for V'_i and V'_j */

  /*********************************************
        calculation of V from H(0)-H(1)
  *********************************************/
  /* memory allocation of Vi and Vj  */
  Vi = (double**)malloc(sizeof(double*)*(NumOrbs_1st+1));
  for (i=0; i<=NumOrbs_1st; i++){
    Vi[i] = (double*)malloc(sizeof(double)*(NumOrbs_1st+1));
  }

  Vj = (double**)malloc(sizeof(double*)*(NumOrbs_2nd+1));
  for (i=0; i<=NumOrbs_2nd; i++){
    Vj[i] = (double*)malloc(sizeof(double)*(NumOrbs_2nd+1));
  }

  for (i=1; i<=NumOrbs_1st; i++){
    for (j=1; j<=NumOrbs_1st; j++){
      Vi[i][j] = 0.5 * ( Hks_1st[0][i-1][j-1] - Hks_1st[1][i-1][j-1]);
    }
  }

  for (i=1; i<=NumOrbs_2nd; i++){
    for (j=1; j<=NumOrbs_2nd; j++){
      Vj[i][j] = 0.5*(Hks_2nd[0][i-1][j-1] - Hks_2nd[1][i-1][j-1]);
    }
  }

  /*********************************************
           calculation of VVi and VVj
  *********************************************/
  VVi = (double****)malloc(sizeof(double***)*(SpinP_switch+1));
  for (spin1=0; spin1<=SpinP_switch; spin1++){
    VVi[spin1] = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin2=0; spin2<=1; spin2++){
      VVi[spin1][spin2] = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
      for (i=0; i<=T_NumOrbs; i++){
        VVi[spin1][spin2][i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
      }
    }
  }
  VVj = (double****)malloc(sizeof(double***)*(SpinP_switch+1));
  for (spin1=0; spin1<=SpinP_switch; spin1++){
    VVj[spin1] = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    for (spin2=0; spin2<=SpinP_switch; spin2++){
      VVj[spin1][spin2] = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
      for (i=0; i<=T_NumOrbs; i++){
        VVj[spin1][spin2][i] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
      }
    }
  }

  /* calculation VVi */
  mat_temp = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    mat_temp[i] = (double*)malloc(sizeof(double)*(NumOrbs_1st+1));
  }

  for(spin1=0; spin1<=SpinP_switch; spin1++){
    for (spin2=0; spin2<=SpinP_switch; spin2++){

      matmul_double_lapack(
        "T","N",T_NumOrbs,NumOrbs_1st,NumOrbs_1st,
        Ci[spin1],Vi,mat_temp);
      matmul_double_lapack(
        "N","N",T_NumOrbs,T_NumOrbs,NumOrbs_1st,
        mat_temp,Ci[spin2],VVi[spin1][spin2]);

    }
  }

  for (i=0; i<=T_NumOrbs; i++){
    free(mat_temp[i]);
  }
  free(mat_temp);

  /* calculation VVj */
  mat_temp = (double**)malloc(sizeof(double*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    mat_temp[i] = (double*)malloc(sizeof(double)*(NumOrbs_2nd+1));
  }

  for(spin1=0; spin1<=SpinP_switch; spin1++){
    for (spin2=0; spin2<=SpinP_switch; spin2++){
      matmul_double_lapack(
        "C","N",T_NumOrbs,NumOrbs_2nd,NumOrbs_2nd,
        Cj[spin1],Vj,mat_temp);
      matmul_double_lapack(
        "N","N",T_NumOrbs,T_NumOrbs,NumOrbs_2nd,
        mat_temp,Cj[spin2],VVj[spin1][spin2]);

      }
    }

    for (i=0; i<=T_NumOrbs; i++){
      free(mat_temp[i]);
    }
    free(mat_temp);

  /*********************************************
              calculation of J_ij
  *********************************************/

  /* allocation of memory */
  Fftn = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Fftn[spin] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
  }

  /* the Fftn-array for Fermi function */
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=T_NumOrbs; i++){
      Fftn[spin][i] = 1.0/( exp( (ko[spin][i]- ChemP)/(kB*E_Temp) ) + 1.0 )  ;
    }
  }

  /* J_ij calculation from VVi & VVj */
  *J_ij=0.0;
  for (i=1; i<=T_NumOrbs; i++){
    for (j=1; j<=T_NumOrbs; j++){
      dFftn = Fftn[0][i] - Fftn[1][j] ;
      dko = ko[1][j] - ko[0][i] ;
      *J_ij = *J_ij + 0.5*dFftn * VVj[0][1][i][j] * VVi[1][0][j][i] / dko ;
    }
  }

  /*********************************************
    freeing of arrays:

    double **Vi;
    double **Vj;
    double ****VVi;
    double ****VVj;
    double **Fftn;
  *********************************************/

  for (i=0; i<=NumOrbs_1st; i++){
    free(Vi[i]);
  }
  free(Vi);
  for (i=0; i<=NumOrbs_2nd; i++){
    free(Vj[i]);
  }
  free(Vj);
  for (spin1=0; spin1<=SpinP_switch; spin1++){
      for (spin2=0; spin2<=1; spin2++){
        for (i=0; i<=T_NumOrbs; i++){
          free(VVi[spin1][spin2][i]);
          free(VVj[spin1][spin2][i]);
        }
        free(VVi[spin1][spin2]);
        free(VVj[spin1][spin2]);
      }
      free(VVi[spin1]);
      free(VVj[spin1]);
    }
    free(VVi);
    free(VVj);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Fftn[spin]);
  }
  free(Fftn);

}
