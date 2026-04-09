/*
Calculation of exchange coupling constant J_{i0,jR} between atom i at cell 0 and atom j at cell R.
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
#include "jx_LNO.h"

//#include <malloc/malloc.h>
//#include <assert.h>

//#pragma optimization_level 2
void Jij_band_indiv(MPI_Comm comm1){

  static int i,j,k,l,n;
  static int ij,ik;
  static int j1,j2;

  static int ct_AN,h_AN,Gh_AN,TNO1,TNO2;
  static int spin,Rn;

  int numprocs,myid,ID;

  int T_NumOrbs,NumOrbs_max;
  int *MP;

  int id_ij_each, id_ij_total, id_bunch;

  static double *Jij, *Jij_sum;

  int kloop, kprocs;
  double **KGrids;
  static int knum_switch;
  static double k1,k2,k3;
  static double k_R;

  static dcomplex **H,**S;

  static int id_1st,id_2nd;
  static int *NumOrbs_1st,*NumOrbs_2nd;
  static int MP_1st,MP_2nd;
  static double E_r,E_i,G_r,G_i;

  static dcomplex ***dH_1st,***dH_2nd,***S_1st,***S_2nd;

  static dcomplex *Z_Fermi, *Res_Fermi;
  static double inv_beta;

  static int Eloop;
  static dcomplex Ener;
  static dcomplex ***Green;
  static dcomplex ***Green_up,***Green_down;
  static dcomplex ***Green_up_sum,***Green_down_sum;
  static dcomplex *****vec_eig_1st,*****vec_eig_2nd;
  static double ***val_eig;

  static double eta=1.0e-10;
  static double kB=0.000003166813628;  /* Boltzman constant (Hatree/K) */

  dcomplex **mat_1,**mat_2;

  static double time_eig_start,time_eig_end,time_Jij_start,time_Jij_end;
  static double *dt_eig,*dt_Jij;

  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  MPI_Barrier(comm1);

  NumOrbs_1st = (int*)malloc(sizeof(int)*num_ij_bunch);
  NumOrbs_2nd = (int*)malloc(sizeof(int)*num_ij_bunch);

  Green_up = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  Green_down = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  Green_up_sum = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  Green_down_sum = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  vec_eig_1st = (dcomplex*****)malloc(sizeof(dcomplex****)*num_ij_bunch);
  vec_eig_2nd = (dcomplex*****)malloc(sizeof(dcomplex****)*num_ij_bunch);
  dH_1st = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  dH_2nd = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  S_1st = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);
  S_2nd = (dcomplex***)malloc(sizeof(dcomplex**)*num_ij_bunch);

  Jij = (double*)malloc(sizeof(double)*num_ij_bunch);
  Jij_sum = (double*)malloc(sizeof(double)*num_ij_bunch);
  dt_eig = (double*)malloc(sizeof(double)*num_ij_bunch);
  dt_Jij = (double*)malloc(sizeof(double)*num_ij_bunch);

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  T_NumOrbs = 0;
  NumOrbs_max = 0;
  for (i=0; i<atomnum; i++){
    ct_AN=i+1;
    MP[i+1] = T_NumOrbs+1;
    T_NumOrbs = T_NumOrbs + Total_NumOrbs[ct_AN];
    if (NumOrbs_max < Total_NumOrbs[ct_AN]);{
      NumOrbs_max=Total_NumOrbs[ct_AN];
    }
  }

  inv_beta=kB*E_Temp;
  Z_Fermi = (dcomplex*)malloc(sizeof(dcomplex)*num_poles);
  Res_Fermi = (dcomplex*)malloc(sizeof(dcomplex)*num_poles);
  zero_cfrac(num_poles,Z_Fermi,Res_Fermi);

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

  total_mem+=2.0*sizeof(dcomplex)*num_poles;
  total_mem+=1.0*sizeof(double)*(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1)*3;
  total_mem+=7.0*sizeof(int)*num_ij_bunch;
  total_mem+=4.0*sizeof(double)*num_ij_bunch;
  total_mem+=2.0*sizeof(dcomplex)*(T_NumOrbs+1)*(T_NumOrbs+1);
  total_mem+=1.0*sizeof(double)*(
    num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1)*(SpinP_switch+1)*(T_NumOrbs+1);
  total_mem+=6.0*sizeof(dcomplex)*(NumOrbs_max+1)*(NumOrbs_max+1);
  total_mem+=2.0*sizeof(dcomplex)*(
    num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]/numprocs+1)*(SpinP_switch+1)*(T_NumOrbs+1)*(NumOrbs_max+1);
  total_mem+=1.0*sizeof(dcomplex)*(NumOrbs_max+1)*(T_NumOrbs+1);
  total_mem+=1.0*sizeof(dcomplex)*(NumOrbs_max+1)*(NumOrbs_max+1);
  total_mem+=1.0*sizeof(dcomplex)*(NumOrbs_max+1)*(T_NumOrbs+1);
  total_mem=total_mem*numprocs;

/*  if (myid==Host_ID){
    printf("\n");
    printf(" number of processes: %10i\n",numprocs);
    printf(" total number of k-points: %10i\n",num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]);
    if (numprocs>num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]){
      printf(" WARNING: inefficient numprocs specification\n");
    }
    printf(" approx. memory usage of stored allays: ");
    if ( total_mem/8.0 < pow(1024.0,2.0) ){
      printf("%10.2f kB\n",total_mem/(8.0*1024) );
    }
    else if (total_mem/8.0 < pow(1024.0,3.0)){
      printf("%10.2f MB\n",total_mem/(8.0*pow(1024.0,2.0)) );
    }
    else if (total_mem/8.0 < pow(1024.0,4.0) ){
      printf("%10.2f GB\n",total_mem/(8.0*pow(1024.0,3.0) ) );
    }
  } */

//  printf("212, myid=%i\n",myid);
  MPI_Barrier(comm1);

  if (myid==Host_ID && Flag_interactive == 0){
    printf("\n");
    printf("    i     j    c1    c2    c3            J [meV]            J [mRy]   time_eig [s]   time_Jij [s]\n");
    printf("-------------------------------------------------------------------------------------------------\n");
  }

  for (id_bunch=0; id_bunch<num_ij_total/(1.0*num_ij_bunch)-1.0e-10; id_bunch++){

    for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
       id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
      if (id_ij_total >= num_ij_total) {
          continue;
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

      Green_up[id_ij_each] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_2nd[id_ij_each]+1));
      for (i=0; i<=NumOrbs_2nd[id_ij_each]; i++){
        Green_up[id_ij_each][i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_1st[id_ij_each]+1));
      }
      Green_down[id_ij_each] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
      for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
        Green_down[id_ij_each][i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd[id_ij_each]+1));
      }

      Green_up_sum[id_ij_each] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_2nd[id_ij_each]+1));
      for (i=0; i<=NumOrbs_2nd[id_ij_each]; i++){
        Green_up_sum[id_ij_each][i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_1st[id_ij_each]+1));
      }
      Green_down_sum[id_ij_each] = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
      for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
        Green_down_sum[id_ij_each][i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd[id_ij_each]+1));
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

    dtime(&time_eig_start);

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
            continue;
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
    dtime(&time_eig_end);

    dt_eig[0]=time_eig_end-time_eig_start;
    for (id_ij_each=1; id_ij_each<num_ij_bunch; id_ij_each++){
      dt_eig[id_ij_each]=0.0;
    }

    for (id_ij_each=0; id_ij_each<num_ij_bunch; id_ij_each++){
      id_ij_total=num_ij_bunch*id_bunch+id_ij_each;
      if ( id_ij_total >= num_ij_total) {
          continue;
      }

      dtime(&time_Jij_start);

      Jij[id_ij_each]=0.0;

      for (Eloop=0; Eloop<num_poles; Eloop++){
        Ener.r=ChemP;
        Ener.i=inv_beta*Z_Fermi[Eloop].i;
        for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
          for (j=0; j<=NumOrbs_2nd[id_ij_each]; j++){
            Green_up[id_ij_each][j][i].r=0.0;
            Green_up[id_ij_each][j][i].i=0.0;
            Green_down[id_ij_each][i][j].r=0.0;
            Green_down[id_ij_each][i][j].i=0.0;
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

          k_R = KGrids[kprocs][0]*(double)id_cell[id_ij_total][0]
              + KGrids[kprocs][1]*(double)id_cell[id_ij_total][1]
              + KGrids[kprocs][2]*(double)id_cell[id_ij_total][2];
          k_R=2.0*PI*k_R;

          mat_1 = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_2nd[id_ij_each]+1));
          mat_2 = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_2nd[id_ij_each]+1));
          for (i=0; i<=NumOrbs_2nd[id_ij_each]; i++){
            mat_1[i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
            mat_2[i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_1st[id_ij_each]+1));
          }

          for (i=1; i<=NumOrbs_2nd[id_ij_each]; i++){
            for (n=1; n<=T_NumOrbs; n++){
              E_r=Ener.r-val_eig[kprocs][0][n];
              E_i=Ener.i+eta;
              G_r=E_r/(E_r*E_r+E_i*E_i);
              G_i=-E_i/(E_r*E_r+E_i*E_i);
              mat_1[i][n].r=vec_eig_2nd[id_ij_each][kprocs][0][i][n].r*G_r
                            -vec_eig_2nd[id_ij_each][kprocs][0][i][n].i*G_i;
              mat_1[i][n].i=vec_eig_2nd[id_ij_each][kprocs][0][i][n].r*G_i
                            +vec_eig_2nd[id_ij_each][kprocs][0][i][n].i*G_r;
            }
          }

          matmul_dcomplex_lapack(
            "N","C",NumOrbs_2nd[id_ij_each],NumOrbs_1st[id_ij_each],T_NumOrbs,
            mat_1,vec_eig_1st[id_ij_each][kprocs][0],mat_2);

          for (i=1; i<=NumOrbs_2nd[id_ij_each]; i++){
            for (j=1; j<=NumOrbs_1st[id_ij_each]; j++){
              Green_up[id_ij_each][i][j].r
                += mat_2[i][j].r*cos(k_R)-mat_2[i][j].i*sin(k_R);
              Green_up[id_ij_each][i][j].i
                += mat_2[i][j].r*sin(k_R)+mat_2[i][j].i*cos(k_R);
            }
          }

          for (i=0; i<=NumOrbs_2nd[id_ij_each]; i++){
            free(mat_1[i]);
            free(mat_2[i]);
          }
          free(mat_1);
          free(mat_2);

          mat_1 = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
          mat_2 = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
          for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
            mat_1[i] = (dcomplex*)malloc(sizeof(dcomplex)*(T_NumOrbs+1));
            mat_2[i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd[id_ij_each]+1));
          }

          for (n=1; n<=T_NumOrbs; n++){
            E_r=Ener.r-val_eig[kprocs][1][n];
            E_i=Ener.i+eta;
            G_r=E_r/(E_r*E_r+E_i*E_i);
            G_i=-E_i/(E_r*E_r+E_i*E_i);
            for (i=1; i<=NumOrbs_1st[id_ij_each]; i++){
              mat_1[i][n].r=vec_eig_1st[id_ij_each][kprocs][1][i][n].r*G_r
                            -vec_eig_1st[id_ij_each][kprocs][1][i][n].i*G_i;
              mat_1[i][n].i=vec_eig_1st[id_ij_each][kprocs][1][i][n].r*G_i
                            +vec_eig_1st[id_ij_each][kprocs][1][i][n].i*G_r;
            }
          }

          matmul_dcomplex_lapack(
            "N","C",NumOrbs_1st[id_ij_each],NumOrbs_2nd[id_ij_each],T_NumOrbs,
            mat_1,vec_eig_2nd[id_ij_each][kprocs][1],mat_2);

          for (i=1; i<=NumOrbs_1st[id_ij_each]; i++){
            for (j=1; j<=NumOrbs_2nd[id_ij_each]; j++){
            Green_down[id_ij_each][i][j].r
                += mat_2[i][j].r*cos(-k_R)-mat_2[i][j].i*sin(-k_R);
              Green_down[id_ij_each][i][j].i
                += mat_2[i][j].r*sin(-k_R)+mat_2[i][j].i*cos(-k_R);
            }
          }

          for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
            free(mat_1[i]);
            free(mat_2[i]);
          }
          free(mat_1);
          free(mat_2);

        }

    MPI_Barrier(comm1);

        for (i=1; i<=NumOrbs_2nd[id_ij_each]; i++){
          for (j=1; j<=NumOrbs_1st[id_ij_each]; j++){
            MPI_Allreduce(
              &Green_up[id_ij_each][i][j].r,&Green_up_sum[id_ij_each][i][j].r,
              1,MPI_DOUBLE,MPI_SUM,comm1);
            MPI_Allreduce(
              &Green_up[id_ij_each][i][j].i,&Green_up_sum[id_ij_each][i][j].i,
              1,MPI_DOUBLE,MPI_SUM,comm1);
            MPI_Allreduce(
              &Green_down[id_ij_each][j][i].r,&Green_down_sum[id_ij_each][j][i].r,
              1,MPI_DOUBLE,MPI_SUM,comm1);
            MPI_Allreduce(
              &Green_down[id_ij_each][j][i].i,&Green_down_sum[id_ij_each][j][i].i,
              1,MPI_DOUBLE,MPI_SUM,comm1);
          }
        }

    MPI_Barrier(comm1);
//    printf("560\n");

        mat_1 = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
        mat_2 = (dcomplex**)malloc(sizeof(dcomplex*)*(NumOrbs_1st[id_ij_each]+1));
        for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
          mat_1[i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd[id_ij_each]+1));
          mat_2[i] = (dcomplex*)malloc(sizeof(dcomplex)*(NumOrbs_2nd[id_ij_each]+1));
        }

        matmul_dcomplex_lapack(
          "N","N",NumOrbs_1st[id_ij_each],NumOrbs_2nd[id_ij_each],NumOrbs_1st[id_ij_each],
          dH_1st[id_ij_each],Green_down_sum[id_ij_each],mat_1);
        matmul_dcomplex_lapack(
          "N","N",NumOrbs_1st[id_ij_each],NumOrbs_2nd[id_ij_each],NumOrbs_2nd[id_ij_each],
          mat_1,dH_2nd[id_ij_each],mat_2);
        for (i=1; i<=NumOrbs_1st[id_ij_each]; i++){
          for (j=1; j<=NumOrbs_2nd[id_ij_each]; j++){
            Jij[id_ij_each]
             += Res_Fermi[Eloop].r*
                  ( mat_2[i][j].r*Green_up_sum[id_ij_each][j][i].r
                   -mat_2[i][j].i*Green_up_sum[id_ij_each][j][i].i )
              - Res_Fermi[Eloop].i*
                  ( mat_2[i][j].r*Green_up_sum[id_ij_each][j][i].i
                   +mat_2[i][j].i*Green_up_sum[id_ij_each][j][i].r );
          }
        }

        for (i=0; i<=NumOrbs_1st[id_ij_each]; i++){
          free(mat_1[i]);
          free(mat_2[i]);
        }
        free(mat_1);
        free(mat_2);

      } /* Eloop */

      dtime(&time_Jij_end);

      dt_Jij[id_ij_each]=time_Jij_end-time_Jij_start;

      Jij[id_ij_each]
        = 0.5*Jij[id_ij_each]*inv_beta
          /(double)((num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2])*(num_Kgrid[0]*num_Kgrid[1]*num_Kgrid[2]));

      if (myid==Host_ID){
        printf(
          "%5i %5i %5i %5i %5i %18.12f %18.12f %14.5f %14.5f\n",
          id_atom[id_ij_total][0],id_atom[id_ij_total][1],
          id_cell[id_ij_total][0],id_cell[id_ij_total][1],id_cell[id_ij_total][2],
          Jij[id_ij_each]*27.211386*1000,Jij[id_ij_each]*2*1000,
          dt_eig[id_ij_each],dt_Jij[id_ij_each]);
      }

      for (j=0; j<NumOrbs_1st[id_ij_each]+1; j++){
        free(dH_1st[id_ij_each][j]);
      }
      free(dH_1st[id_ij_each]);
      for (j=0; j<NumOrbs_2nd[id_ij_each]+1; j++){
        free(dH_2nd[id_ij_each][j]);
      }
      free(dH_2nd[id_ij_each]);

      for (j=0; j<NumOrbs_2nd[id_ij_each]+1; j++){
        free(Green_up[id_ij_each][j]);
      }
      free(Green_up[id_ij_each]);

      for (j=0; j<NumOrbs_1st[id_ij_each]+1; j++){
        free(Green_down[id_ij_each][j]);
      }
      free(Green_down[id_ij_each]);

      for (j=0; j<NumOrbs_2nd[id_ij_each]+1; j++){
        free(Green_up_sum[id_ij_each][j]);
      }
      free(Green_up_sum[id_ij_each]);
      for (j=0; j<NumOrbs_1st[id_ij_each]+1; j++){
        free(Green_down_sum[id_ij_each][j]);
      }
      free(Green_down_sum[id_ij_each]);

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

    MPI_Barrier(comm1);

  }

  free(NumOrbs_1st);
  free(NumOrbs_2nd);
  free(vec_eig_1st);
  free(vec_eig_2nd);
  free(dH_1st);
  free(dH_2nd);
  free(S_1st);
  free(S_2nd);
  free(Jij);
  free(Jij_sum);
  free(dt_eig);
  free(dt_Jij);

  free(Green_up);
  free(Green_down);
  free(Green_up_sum);
  free(Green_down_sum);

  free(Z_Fermi);
  free(Res_Fermi);

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

}
