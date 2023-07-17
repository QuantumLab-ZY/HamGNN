/**********************************************************************
  LNO.c:

   LNO.c is a subroutine to calculate strictly localized non-orthogonal
   natural orbitals.

  Log of LNO.c:

     06/March/2018  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "read_scfout.h"
#include "lapack_prototypes.h"
#include "f77func.h"
#include "mpi.h"
#include <omp.h>

#include "jx_config.h"
#include "jx_LNO.h"
#include "jx_quicksort.h"
//#include "jx_param_species_atoms.h"
#include "jx_tools.h"

//#include <malloc/malloc.h>
//#include <assert.h>

#define  measure_time   0

void LNO_alloc(MPI_Comm comm1){

  int spin,i,j,k,l,n,tno1;
  int ct_AN,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int T_NumOrbs,NumOrbs_max;

  T_NumOrbs = 0;
  NumOrbs_max = 0;
  for (i=0; i<atomnum; i++){
    ct_AN=i+1;
    T_NumOrbs = T_NumOrbs + Total_NumOrbs[ct_AN];
    if (NumOrbs_max < Total_NumOrbs[ct_AN]);{
      NumOrbs_max=Total_NumOrbs[ct_AN];
    }
  }

  LNO_coes = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (k=0; k<=SpinP_switch; k++){
    LNO_coes[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      LNO_coes[k][Gc_AN] = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max);
    }
  }

  LNO_pops = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (k=0; k<=SpinP_switch; k++){
    LNO_pops[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      LNO_pops[k][Gc_AN] = (double*)malloc(sizeof(double)*NumOrbs_max);
    }
  }

  LNO_Num = (int*)malloc(sizeof(int)*(atomnum+1));

  LNO_mat = (dcomplex****)malloc( sizeof(dcomplex***)*(SpinP_switch+1) ) ;
  for (spin=0; spin<=SpinP_switch; spin++){
    LNO_mat[spin] = (dcomplex***)malloc( sizeof(dcomplex**)*(atomnum+1) ) ;
  }
  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){

      tno1=Total_NumOrbs[Gc_AN];
      LNO_mat[spin][Gc_AN] = (dcomplex**)malloc( sizeof(dcomplex*)*(tno1+1) ) ;
      for (i=0; i<=tno1; i++){
          LNO_mat[spin][Gc_AN][i] = (dcomplex*)malloc( sizeof(dcomplex)*(tno1+1) );
      }
      for (i=0; i<=tno1; i++){
        for (j=0; j<=tno1; j++){
          LNO_mat[spin][Gc_AN][i][j].r = 0.0;
          LNO_mat[spin][Gc_AN][i][j].i = 0.0;
        }
      }
    }
  }

}

void LNO_free(MPI_Comm comm1){

  int spin,i,j,k,l,n,tno1;
  int ct_AN,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int T_NumOrbs,NumOrbs_max;

  for (k=0; k<=SpinP_switch; k++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(LNO_coes[k][Gc_AN]);
    }
    free(LNO_coes[k]);
  }
  free(LNO_coes);

  for (k=0; k<=SpinP_switch; k++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(LNO_pops[k][Gc_AN]);
    }
    free(LNO_pops[k]);
  }
  free(LNO_pops);

  free(LNO_Num);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      for (i=0; i<=Total_NumOrbs[Gc_AN]; i++){
        free( LNO_mat[spin][Gc_AN][i] );
      }
      free( LNO_mat[spin][Gc_AN] ) ;
    }
    free( LNO_mat[spin] ) ;
  }
  free( LNO_mat ) ;

}

double LNO_Col_Diag(MPI_Comm comm1)
//			   , char *mode, int SCF_iter,double ****OLP, double *****DM)
{
  int i,j,k,l,n;
	int ct_AN,Mc_AN,Gc_AN,h_AN,Mh_AN,Gh_AN;
  int Cwan,num,wan1,wan2,tno0,tno1,tno2,spin;
  int Nloop,po,p,q;
  char *JOBVL,*JOBVR;
  int N,A,LDA,LDVL,LDVR,SDIM,LWORK,INFO,*IWORK;
  double ***DMS,*WR,*WI,*VL,*VR,*WORK,RCONDE,RCONDV;
  double *B,*C,*IC,sum,sum0,F;
  double TStime,TEtime;

  int myid,numprocs,tag=999;
  MPI_Status stat;
  MPI_Request request;

	int T_NumOrbs,NumOrbs_max;

  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

	T_NumOrbs = 0;
  NumOrbs_max = 0;
  for (i=0; i<atomnum; i++){
    ct_AN=i+1;
    T_NumOrbs = T_NumOrbs + Total_NumOrbs[ct_AN];
    if (NumOrbs_max < Total_NumOrbs[ct_AN]);{
      NumOrbs_max=Total_NumOrbs[ct_AN];
    }
  }

  /********************************************
             allocation of arrays
  ********************************************/

/*  LNO_coes = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (k=0; k<=SpinP_switch; k++){
    LNO_coes[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      LNO_coes[k][Gc_AN] = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max);
    }
  }

  LNO_pops = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (k=0; k<=SpinP_switch; k++){
    LNO_pops[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      LNO_pops[k][Gc_AN] = (double*)malloc(sizeof(double)*NumOrbs_max);
    }
  }

  LNO_Num = (int*)malloc(sizeof(int)*(atomnum+1)); */

  DMS = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    DMS[spin] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<(atomnum+1); Gc_AN++){
      DMS[spin][Gc_AN] = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max);
      for (i=0; i<NumOrbs_max*NumOrbs_max; i++){
        DMS[spin][Gc_AN][i] = 0.0;
      }
    }
  }

  WR = (double*)malloc(sizeof(double)*NumOrbs_max);
  WI = (double*)malloc(sizeof(double)*NumOrbs_max);
  VL = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max*2);
  VR = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max*2);

  WORK = (double*)malloc(sizeof(double)*NumOrbs_max*10);
  IWORK = (int*)malloc(sizeof(int)*NumOrbs_max);

  B = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max);
  C = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max);
  IC = (double*)malloc(sizeof(double)*NumOrbs_max*NumOrbs_max);

  /********************************************
        calculation of DMS defined by DM*S
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      tno1=Total_NumOrbs[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        tno2=Total_NumOrbs[Gh_AN];
        for (i=0; i<tno1; i++){
	        for (j=0; j<tno1; j++){
            sum = 0.0;
            for (k=0; k<tno2; k++){
              sum += DM[spin][Gc_AN][h_AN][i][k]*OLP[Gc_AN][h_AN][j][k];
            }
            DMS[spin][Gc_AN][tno1*j+i] += sum;
	        }
	      }
      }
    } /* Gc_AN */
  } /* spin */

  /********************************************
            diagonalization of DMS
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      tno1=Total_NumOrbs[Gc_AN];
      JOBVL = "V";
      JOBVR = "V";
      N = tno1;
      LDA = tno1;
      LDVL = tno1*2;
      LDVR = tno1*2;
      LWORK = tno1*10;
      for (i=0; i<tno1*tno1; i++) B[i] = DMS[spin][Gc_AN][i];
      F77_NAME(dgeev,DGEEV)( JOBVL, JOBVR, &N, B, &LDA, WR, WI, VL, &LDVL, VR, &LDVR,
                             WORK, &LWORK, &INFO );
      if (INFO!=0){
        printf("warning: INFO=%2d in calling dgeev in a function 'LNO'\n",INFO);
      }
      /* ordering the eigenvalues and the orthogonal matrix */
      for (i=0; i<tno1; i++) IWORK[i] = i;
      qsort_double_int2(tno1,WR,IWORK);

      /* calculations of Frobenius norm */
      if (0 && myid==0){
        for (i=0; i<tno1; i++){
	        for (j=0; j<tno1; j++){
	          B[j*tno1+i] = 0.0;
	        }
	      }
        for (k=0; k<tno1; k++){
          l = IWORK[k];
          for (i=0; i<tno1; i++){
            for (j=0; j<tno1; j++){
              B[j*tno1+i] += VR[LDVR*l+i]*WR[k]*VL[LDVL*l+j];
						}
					}
				}
        printf("DMS\n");
				for (i=0; i<tno1; i++){
					for (j=0; j<tno1; j++){
            printf("%10.6f ",DMS[spin][Gc_AN][tno1*j+i]);
					}
          printf("\n");
	      }
        printf("B\n");
        for (i=0; i<tno1; i++){
          for (j=0; j<tno1; j++){
            printf("%10.6f ",B[tno1*j+i]);
					}
          printf("\n");
			  }
      }

      /* copy VR to LNO_coes, where vectors in LNO_coes are stored in column */
      for (j=0; j<tno1; j++){
        k = IWORK[j];
				for (i=0; i<tno1; i++){
					LNO_coes[spin][Gc_AN][tno1*j+i] = VR[LDVR*k+i];
				}
        sum = 0.0;
				for (i=0; i<tno1; i++){
					sum += LNO_coes[spin][Gc_AN][tno1*j+i]*LNO_coes[spin][Gc_AN][tno1*j+i];
				}
        sum = 1.0/sqrt(sum);
				for (i=0; i<tno1; i++){
					LNO_coes[spin][Gc_AN][tno1*j+i] *= sum;
				}
      }

      /* store the eigenvalues */
      for (i=0; i<tno1; i++){
//        printf("%i  %i  %i ",spin,Gc_AN,i);
//        printf("%f\n",WR[i]);
        LNO_pops[spin][Gc_AN][i] = WR[i];
      }

      /********************************************
       determine the number of LNOs which will be
       included in proceeding calculations.
      ********************************************/
      if (spin==0) LNO_Num[Gc_AN] = 0;
      for (i=0; i<tno1; i++){
        if (LNO_pops[spin][Gc_AN][i]<LNO_Occ_Cutoff && LNO_Num[Gc_AN]<=i){
          LNO_Num[Gc_AN] = i;
          break;
        }
      }
      if (i==tno1){
        LNO_Num[Gc_AN] = tno1;
      }

    } /* Gc_AN */
  } /* spin */

  /********************************************
             freeing of arrays
  ********************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=0; Gc_AN<(atomnum+1); Gc_AN++){
      free(DMS[spin][Gc_AN]);
    }
    free(DMS[spin]);
  }
  free(DMS);

  free(WR);
  free(WI);
  free(VL);
  free(VR);
  free(WORK);
  free(IWORK);
  free(B);
  free(C);
  free(IC);
//  assert( malloc_zone_check(NULL) );

  /* for time */
  dtime(&TEtime);

  return (TEtime-TStime);
}

void LNO_occ_trns_mat(MPI_Comm comm1){

  int i,j,k;
	int Gc_AN;
  int tno1,spin;
  double **Vmat, **inv_Vmat;

/*  LNO_mat = (dcomplex****)malloc( sizeof(dcomplex***)*(SpinP_switch+1) ) ;
  for (spin=0; spin<=SpinP_switch; spin++){
    LNO_mat[spin] = (dcomplex***)malloc( sizeof(dcomplex**)*(atomnum+1) ) ;
  } */

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      tno1=Total_NumOrbs[Gc_AN];
/*      LNO_mat[spin][Gc_AN] = (dcomplex**)malloc( sizeof(dcomplex*)*(tno1+1) ) ;
      for (i=0; i<=tno1; i++){
          LNO_mat[spin][Gc_AN][i] = (dcomplex*)malloc( sizeof(dcomplex)*(tno1+1) );
      } */
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          LNO_mat[spin][Gc_AN][i+1][j+1].r = 0.0;
          LNO_mat[spin][Gc_AN][i+1][j+1].i = 0.0;
        }
      }

      Vmat = (double**)malloc( sizeof(double*)*(tno1+1) ) ;
      inv_Vmat = (double**)malloc( sizeof(double*)*(tno1+1) ) ;
      for (i=0; i<=tno1; i++){
        Vmat[i] = (double*)malloc( sizeof(double)*(tno1+1) );
        inv_Vmat[i] = (double*)malloc( sizeof(double)*(tno1+1) );
      }

      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          Vmat[i+1][j+1]=LNO_coes[spin][Gc_AN][tno1*j+i];
          inv_Vmat[i+1][j+1]=Vmat[i+1][j+1];
        }
      }

/*      printf("Vmat\n");
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          printf("%i %i %i %i ", spin,Gc_AN,i,j);
          printf("%f \n",Vmat[i+1][j+1]);
        }
      } */

      matinv_double_lapack(inv_Vmat, tno1);

/*      printf("inv_Vmat\n");
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          printf("%i %i %i %i ", spin,Gc_AN,i,j);
          printf("%f \n",inv_Vmat[i+1][j+1]);
        }
      } */

      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          for (k=0; k<tno1; k++){
            if (LNO_pops[spin][Gc_AN][k]>LNO_Occ_Cutoff){
              LNO_mat[spin][Gc_AN][i+1][j+1].r += Vmat[i+1][k+1]*inv_Vmat[k+1][j+1];
            }
          }
        }
      }

/*      printf("LNO_pops\n");
      for (i=0; i<tno1; i++){
          printf("%i %i %i ", spin,Gc_AN,i);
          printf("%f %f \n",LNO_pops[spin][Gc_AN][i],LNO_Occ_Cutoff);
      }
      printf("LNO_mat\n");
      for (i=0; i<tno1; i++){
        for (j=0; j<tno1; j++){
          printf("%i %i %i %i ", spin,Gc_AN,i,j);
          printf("%f \n",LNO_mat[spin][Gc_AN][i+1][j+1].r);
        }
      } */

      for (i=0; i<=tno1; i++){
        free( Vmat[i] );
        free( inv_Vmat[i] );
      }
      free(Vmat) ;
      free(inv_Vmat) ;

    }
  }

}
