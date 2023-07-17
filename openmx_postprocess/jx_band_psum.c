/*
Calculation of exchange coupling constant J_{ij} between periodic images of atoms i and j.
Refer to arXiv:1907.08341 for details.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <string.h>

//#include <malloc/malloc.h>
//#include <assert.h>

#include "read_scfout.h"
#include "lapack_prototypes.h"
#include "f77func.h"
#include "mpi.h"

#include "jx.h"
#include "jx_tools.h"
#include "jx_config.h"
//#include "jx_param_species_atoms.h"
//#include "jx_minimal_basis.h"
#include "jx_LNO.h"

void Jij_band_psum_partial_mat_calc(
  int F_TNumOrbs0, int NumOrbs_1st, int NumOrbs_2nd,
  dcomplex **dH_1st, dcomplex **dH_2nd,
  dcomplex ***Ci, dcomplex ***Cj, double **ko,
  double *J_ij_r, double *J_ij_i);

#pragma optimization_level 2
void Jij_band_psum( MPI_Comm comm1 ){

  int id_ij_each, id_bunch, id_ij_total;

  /* dummy variables for loop    */
  static int i,j,k,l,n;

  static int ct_AN,h_AN,Gh_AN,TNO1,TNO2;
  static int spin,Rn;

  int Anum;          /* temporary variable          */
  int T_NumOrbs;
  int *MP;                 /* Full system version of MP */

//  int ii,jj,ll;
//  int *T_k_op;
//  static int ***k_op;
//  static int ierr;

  int numprocs,myid,ID,ID1;
  int kprocs,kloop;
  double **KGrids;
  double *JrTk;
  double *JiTk;


  static int ij,ik;
  static int knum_switch;
  static double k1,k2,k3;

  static dcomplex **H,**S,***vec_eig;

  static double OLP_eigen_cut = 1.0e-10;

  static int MP_1st,MP_2nd;

  static int id_1st,id_2nd;
  static int *NumOrbs_1st, *NumOrbs_2nd;
  static dcomplex ***dH_1st,***dH_2nd;

  static double ***val_eig;
  static dcomplex *****vec_eig_1st,*****vec_eig_2nd;

  static double Jr_sum,Ji_sum;
  static double Jr_node,Ji_node;
  static double Jk_i,Jk_r;
  static double *J_ij;

  static double time_start,time_eig,time_Jij;
  static double *dt_eig,*dt_Jij;

  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  NumOrbs_1st = (int*)malloc(sizeof(int)*num_ij_bunch);
  NumOrbs_2nd = (int*)malloc(sizeof(int)*num_ij_bunch);
  dH_1st = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  dH_2nd = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  vec_eig_1st = (dcomplex*****)malloc(sizeof(dcomplex****)*num_ij_bunch);
  vec_eig_2nd = (dcomplex*****)malloc(sizeof(dcomplex****)*num_ij_bunch);
  J_ij = (double*)malloc(sizeof(double)*num_ij_bunch);
  dt_eig = (double*)malloc(sizeof(double)*num_ij_bunch);
  dt_Jij = (double*)malloc(sizeof(double)*num_ij_bunch);

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  T_NumOrbs = 0;
  for (i=0; i<atomnum; i++){
    ct_AN=i+1;
    MP[i+1] = T_NumOrbs+1;
    T_NumOrbs = T_NumOrbs + Total_NumOrbs[ct_AN];
  }

  KGrids = (double**)malloc(sizeof(double*)*(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1));

  for (kprocs=0; kprocs<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1; kprocs++){
    KGrids[kprocs] = (double*)malloc(sizeof(double)*3);
  }
  kloop = -1;
  for (i=0; i<num_Kgrid[0]; i++){
    for (j=0; j<num_Kgrid[1]; j++){
      for (k=0; k<num_Kgrid[2]; k++){
        kloop++;
        kprocs=kloop/numprocs;
        if ( kloop % numprocs != myid ){
          continue;
        }
        KGrids[kprocs][0] = -0.5 + (2.0*(double)i+1.0)/(2.0*(double)num_Kgrid[0]);
        KGrids[kprocs][1] = -0.5 + (2.0*(double)j+1.0)/(2.0*(double)num_Kgrid[1]);
        KGrids[kprocs][2] = -0.5 + (2.0*(double)k+1.0)/(2.0*(double)num_Kgrid[2]);
      }
    }
  }

  H = (dcomplex**)malloc(sizeof(dcomplex*)*(T_NumOrbs+1));
  for (j=0; j<=T_NumOrbs; j++){
    H[j] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
  }

  val_eig = (double***)malloc(sizeof(double**)*(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1));
  for (kprocs=0; kprocs<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1; kprocs++){
    val_eig[kprocs] = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
    for (spin=0; spin<=SpinP_switch; spin++){
      val_eig[kprocs][spin] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
    }
  }

  if (myid==Host_ID && Flag_interactive == 0){
    printf("\n");
    printf("    i     j             J [meV]         J [mRy]   time_eig [s]   time_Jij [s]\n");
    printf("-----------------------------------------------------------------------------\n");
  }

  for (id_bunch=0; id_bunch<num_ij_total/(1.0*num_ij_bunch)-1.0e-10; id_bunch++){

    dtime(&time_start);
    MPI_Barrier(comm1);

    for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
       id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
      if (id_ij_total >= num_ij_total) {
          break;
      }

      id_1st=id_atom[id_ij_total][0];
      id_2nd=id_atom[id_ij_total][1];
      NumOrbs_1st[id_ij_each]=Total_NumOrbs[id_1st];
      NumOrbs_2nd[id_ij_each]=Total_NumOrbs[id_2nd];

      dH_1st[id_ij_each] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
      for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
          dH_1st[id_ij_each][i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_1st[id_ij_each]+1));
      }
      dH_2nd[id_ij_each] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_2nd[id_ij_each]+1));
      for (i=0; i<=NumOrbs_2nd[id_ij_each]; i++){
        dH_2nd[id_ij_each][i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd[id_ij_each]+1));
      }

      vec_eig_1st[id_ij_each] = (dcomplex****)malloc(sizeof(dcomplex***)*(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1));
      vec_eig_2nd[id_ij_each] = (dcomplex****)malloc(sizeof(dcomplex***)*(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1));
      for (kprocs=0; kprocs<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1; kprocs++){
        vec_eig_1st[id_ij_each][kprocs] = (dcomplex***)malloc(sizeof(dcomplex**)*(SpinP_switch+1));
        vec_eig_2nd[id_ij_each][kprocs] = (dcomplex***)malloc(sizeof(dcomplex**)*(SpinP_switch+1));
        for (spin=0; spin<=SpinP_switch; spin++){
          vec_eig_1st[id_ij_each][kprocs][spin] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
            for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
              vec_eig_1st[id_ij_each][kprocs][spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
            }
          vec_eig_2nd[id_ij_each][kprocs][spin] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_2nd[id_ij_each]+1));
          for (i=0; i<=NumOrbs_2nd[id_ij_each]; i++){
            vec_eig_2nd[id_ij_each][kprocs][spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
          }
        }
      }

      ct_AN=id_atom[id_ij_total][0];
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        if (Gh_AN == id_atom[id_ij_total][0] && Rn==0) break;
      }

      dH_1st[id_ij_each][0][0].r=0.0;
      dH_1st[id_ij_each][0][0].i=0.0;
      for (i=0; i<NumOrbs_1st[id_ij_each]; i++){
        for (j=0; j<NumOrbs_1st[id_ij_each]; j++){
          dH_1st[id_ij_each][i+1][j+1].r
          = Hks[0][ct_AN][h_AN][i][j]-Hks[1][ct_AN][h_AN][i][j];
          dH_1st[id_ij_each][i+1][j+1].i=0.0;
        }
      }

      ct_AN=id_atom[id_ij_total][1];
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        Gh_AN = natn[ct_AN][h_AN];
        Rn = ncn[ct_AN][h_AN];
        if (Gh_AN == id_atom[id_ij_total][1] && Rn==0) break;
      }
      dH_2nd[id_ij_each][0][0].r=0.0;
      dH_2nd[id_ij_each][0][0].i=0.0;
      for (i=0; i<NumOrbs_2nd[id_ij_each]; i++){
        for (j=0; j<NumOrbs_2nd[id_ij_each]; j++){
          dH_2nd[id_ij_each][i+1][j+1].r
          = Hks[0][ct_AN][h_AN][i][j]-Hks[1][ct_AN][h_AN][i][j];
          dH_2nd[id_ij_each][i+1][j+1].i=0.0;
        }
      }

      if ( flag_LNO==1 ){
        dcplx_basis_transformation(
          NumOrbs_1st[id_ij_each], LNO_mat[0][id_1st], dH_1st[id_ij_each], LNO_mat[1][id_1st]);
        dcplx_basis_transformation(
          NumOrbs_2nd[id_ij_each], LNO_mat[1][id_2nd], dH_2nd[id_ij_each], LNO_mat[0][id_2nd]);
      }

    }

    for (kloop=0; kloop<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]; kloop++){
      kprocs=kloop/numprocs;
      if ( kloop % numprocs != myid ){
        continue;
      }

      k1 = KGrids[kprocs][0];
      k2 = KGrids[kprocs][1];
      k3 = KGrids[kprocs][2];

      Overlap_Band(OLP,S,MP,k1,k2,k3);
      for (spin=0; spin<=SpinP_switch; spin++){
        Hamiltonian_Band(Hks[spin], H, MP, k1, k2, k3);
        gen_eigval_herm(H, S, val_eig[kprocs][spin], T_NumOrbs);
        for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
          id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
         if (id_ij_total >= num_ij_total) {
             break;
         }
          MP_1st = MP[id_atom[id_ij_total][0]];
          MP_2nd = MP[id_atom[id_ij_total][1]];
          for (i=0; i<NumOrbs_1st[id_ij_each]; i++){
            for (j=0; j<T_NumOrbs; j++){
              vec_eig_1st[id_ij_each][kprocs][spin][i+1][j+1].r=H[i+MP_1st][j+1].r;
              vec_eig_1st[id_ij_each][kprocs][spin][i+1][j+1].i=H[i+MP_1st][j+1].i;
            }
          }
          for (i=0; i<NumOrbs_2nd[id_ij_each]; i++){
            for (j=0; j<T_NumOrbs; j++){
              vec_eig_2nd[id_ij_each][kprocs][spin][i+1][j+1].r=H[i+MP_2nd][j+1].r;
              vec_eig_2nd[id_ij_each][kprocs][spin][i+1][j+1].i=H[i+MP_2nd][j+1].i;
            }
          }
        }
      }

    }

    MPI_Barrier(comm1);

    dtime(&time_eig);
    dt_eig[0]=time_eig-time_start;
    for (id_ij_each=1; id_ij_each<num_ij_bunch; id_ij_each++){
      dt_eig[id_ij_each]=0.0;
    }

  for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
     id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
    if (id_ij_total >= num_ij_total) {
        break;
    }


      dtime(&time_start);

      Jr_node=0.0;
      Ji_node=0.0;

      for (kloop=0; kloop<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]; kloop++){
        kprocs=kloop/numprocs;
        if ( kloop % numprocs != myid ){
          continue;
        }

        k1 = KGrids[kprocs][0];
        k2 = KGrids[kprocs][1];
        k3 = KGrids[kprocs][2];

        Jij_band_psum_partial_mat_calc(
          T_NumOrbs, NumOrbs_1st[id_ij_each], NumOrbs_2nd[id_ij_each],
          dH_1st[id_ij_each], dH_2nd[id_ij_each],
          vec_eig_1st[id_ij_each][kprocs], vec_eig_2nd[id_ij_each][kprocs], val_eig[kprocs],
          &Jk_r, &Jk_i);

        Jr_node += Jk_r;
        Ji_node += Jk_i;

      }

      Jr_sum=0.0;
      Ji_sum=0.0;

      MPI_Allreduce(&Jr_node,&Jr_sum,1,MPI_DOUBLE,MPI_SUM,comm1);
      MPI_Allreduce(&Ji_node,&Ji_sum,1,MPI_DOUBLE,MPI_SUM,comm1);

      dtime(&time_Jij);

      dt_Jij[id_ij_each]=time_Jij-time_start;
      J_ij[id_ij_each] = Jr_sum/(double)(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]);

    }

    for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
      id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
      if (id_ij_total >= num_ij_total) {
        break;
      }
      if (myid==Host_ID){
        printf(
          "%5i %5i %19.12f %19.12f %14.5f %14.5f\n",
          id_atom[id_ij_total][0],id_atom[id_ij_total][1],J_ij[id_ij_each]*27.211386*1000,J_ij[id_ij_each]*2*1000,
          dt_eig[id_ij_each],dt_Jij[id_ij_each]);
      }
    }

    MPI_Barrier(comm1);

    for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
      id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
      if (id_ij_total >= num_ij_total) {
         break;
      }

      for (j=0; j<NumOrbs_1st[id_ij_each]+1; j++){
        free(dH_1st[id_ij_each][j]);
      }
      free(dH_1st[id_ij_each]);
      for (j=0; j<NumOrbs_2nd[id_ij_each]+1; j++){
        free(dH_2nd[id_ij_each][j]);
      }
      free(dH_2nd[id_ij_each]);

      for (kprocs=0; kprocs<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1; kprocs++){
        for (spin=0; spin<=SpinP_switch; spin++){
          for (j=0; j<=NumOrbs_1st[id_ij_each]; j++){
            free(vec_eig_1st[id_ij_each][kprocs][spin][j]);
          }
          for (j=0; j<=NumOrbs_2nd[id_ij_each]; j++){
            free(vec_eig_2nd[id_ij_each][kprocs][spin][j]);
          }

          free(vec_eig_1st[id_ij_each][kprocs][spin]);
          free(vec_eig_2nd[id_ij_each][kprocs][spin]);
        }
        free(vec_eig_1st[id_ij_each][kprocs]);
        free(vec_eig_2nd[id_ij_each][kprocs]);
      }
      free(vec_eig_1st[id_ij_each]);
      free(vec_eig_2nd[id_ij_each]);
    }

  }

  for (kprocs=0; kprocs<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1; kprocs++){
    for (spin=0; spin<=SpinP_switch; spin++){
      free(val_eig[kprocs][spin]);
    }
    free(val_eig[kprocs]);
  }
  free(val_eig);

  for (kprocs=0; kprocs<num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1; kprocs++){
    free(KGrids[kprocs]);
  }
  free(KGrids);
  free(MP);
  for (j=0; j<=T_NumOrbs; j++){
    free(H[j]);
  }
  free(H);
  for (i=0; i<=T_NumOrbs; i++){
    free(S[i]);
  }
  free(S);

  free(NumOrbs_1st);
  free(NumOrbs_2nd);
  free(dH_1st);
  free(dH_2nd);
  free(vec_eig_1st);
  free(vec_eig_2nd);
  free(J_ij);
  free(dt_eig);
  free(dt_Jij);

}

void Jij_band_psum_partial_mat_calc(
  int T_NumOrbs, int NumOrbs_1st, int NumOrbs_2nd,
  dcomplex **dH_1st, dcomplex **dH_2nd,
  dcomplex ***Ci, dcomplex ***Cj, double **val_eig,
  double *Jk_r, double *Jk_i
){

  static int spin,spin1,spin2;   /* dummy variables for loop                               */
  static int i,j;

  static double **Fftn;        /* Fermi function */
  static dcomplex J_ij;         /* final exchange-interaction value  */
  static double kB=0.000003166813628;  /* Boltzman constant (Hatree/K) */
  static double dFftn;
  static double dval_eig;

  static dcomplex **mat_temp;
  static dcomplex ****VVi,****VVj;         /* array for V'_i and V'_j */

  VVi = (dcomplex****)malloc(sizeof(dcomplex***)*(SpinP_switch+1));
  for (spin1=0; spin1<=SpinP_switch; spin1++){
    VVi[spin1] = (dcomplex***)malloc(sizeof(dcomplex**)*(SpinP_switch+1));
    for (spin2=0; spin2<=1; spin2++){
      VVi[spin1][spin2] = (dcomplex**)malloc(sizeof(dcomplex*)*(T_NumOrbs+1));
      for (i=0; i<=T_NumOrbs; i++){
        VVi[spin1][spin2][i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
      }
    }
  }
  VVj = (dcomplex****)malloc(sizeof(dcomplex***)*(SpinP_switch+1));
  for (spin1=0; spin1<=SpinP_switch; spin1++){
    VVj[spin1] = (dcomplex***)malloc(sizeof(dcomplex**)*(SpinP_switch+1));
    for (spin2=0; spin2<=SpinP_switch; spin2++){
      VVj[spin1][spin2] = (dcomplex**)malloc(sizeof(dcomplex*)*(T_NumOrbs+1));
      for (i=0; i<=T_NumOrbs; i++){
        VVj[spin1][spin2][i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
      }
    }
  }

  mat_temp = (dcomplex**)malloc(sizeof(dcomplex*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    mat_temp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_1st+1));
  }

  for(spin1=0; spin1<=SpinP_switch; spin1++){
    for (spin2=0; spin2<=SpinP_switch; spin2++){

      matmul_dcomplex_lapack(
        "C","N",T_NumOrbs,NumOrbs_1st,NumOrbs_1st,
        Ci[spin1],dH_1st,mat_temp);
      matmul_dcomplex_lapack(
        "N","N",T_NumOrbs,T_NumOrbs,NumOrbs_1st,
        mat_temp,Ci[spin2],VVi[spin1][spin2]);

    }
  }

  for (i=0; i<=T_NumOrbs; i++){
    free(mat_temp[i]);
  }
  free(mat_temp);

  mat_temp = (dcomplex**)malloc(sizeof(dcomplex*)*(T_NumOrbs+1));
  for (i=0; i<=T_NumOrbs; i++){
    mat_temp[i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd+1));
  }

  for(spin1=0; spin1<=SpinP_switch; spin1++){
    for (spin2=0; spin2<=SpinP_switch; spin2++){
      matmul_dcomplex_lapack(
        "C","N",T_NumOrbs,NumOrbs_2nd,NumOrbs_2nd,
        Cj[spin1],dH_2nd,mat_temp);
      matmul_dcomplex_lapack(
        "N","N",T_NumOrbs,T_NumOrbs,NumOrbs_2nd,
        mat_temp,Cj[spin2],VVj[spin1][spin2]);

      }
    }

    for (i=0; i<=T_NumOrbs; i++){
      free(mat_temp[i]);
    }
    free(mat_temp);

  Fftn = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Fftn[spin] = (double*)malloc(sizeof(double)*(T_NumOrbs+1));
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=T_NumOrbs; i++){
      Fftn[spin][i] = 1.0/( exp( (val_eig[spin][i]- ChemP)/(kB*E_Temp) ) + 1.0 )  ;
    }
  }

  *Jk_r = 0.0;
  *Jk_i = 0.0;
  for (i=1; i<=T_NumOrbs; i++){
    for (j=1; j<=T_NumOrbs; j++){
      dFftn = Fftn[0][i] - Fftn[1][j] ;
      dval_eig = val_eig[1][j] - val_eig[0][i] ;
      *Jk_r = *Jk_r
        + 0.25*(dFftn/dval_eig)*(
          VVj[0][1][i][j].r * VVi[1][0][j][i].r
	        - VVj[0][1][i][j].i * VVi[1][0][j][i].i ) ;

      *Jk_i = *Jk_i
        + 0.25*(dFftn/dval_eig)*(
          VVj[0][1][i][j].r * VVi[1][0][j][i].i
	        + VVj[0][1][i][j].i * VVi[1][0][j][i].r ) ;
    }
  }

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
