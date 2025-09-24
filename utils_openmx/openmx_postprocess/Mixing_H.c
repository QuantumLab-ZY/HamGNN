#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "openmx_common.h"
#include "mpi.h"


static void Simple_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Pulay_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Pulay_Mixing_H_MultiSecant(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Pulay_Mixing_H_with_One_Shot_Hessian(int MD_iter, int SCF_iter, int SCF_iter0 );
static void Inverse(int n, double **a, double **ia);


double Mixing_H( int MD_iter, int SCF_iter, int SCF_iter0 )
{
  double time0;
  double TStime,TEtime;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /*******************************************************
    Simple mixing 
  *******************************************************/

  if ( SCF_iter<=(Pulay_SCF-1) ){
    Simple_Mixing_H( MD_iter, SCF_iter, SCF_iter0 );
  }

  /*******************************************************
    Pulay's method:
    Residual Minimazation Method (RMM) using
    Direct Inversion in the Iterative Subspace (DIIS)
  *******************************************************/

  else{

    Pulay_Mixing_H( MD_iter, SCF_iter, SCF_iter0 );

    /*
    Pulay_Mixing_H_with_One_Shot_Hessian( MD_iter, SCF_iter, SCF_iter0 );
    */

    /*
    Pulay_Mixing_H_MultiSecant( MD_iter, SCF_iter, SCF_iter0 );
    */
  }

  /* if SCF_iter0==1, then NormRD[0]=1 */
  if (SCF_iter0==1) NormRD[0] = 1.0;

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 





void Pulay_Mixing_H_MultiSecant(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan;
  double sum,my_sum,tmp1,tmp2,alpha;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double my_sy,my_yy,sy,yy,norm,s,y,or,al,be;
  double **A,**IA,*coes,*coes2,*ror;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  coes2 = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  ror = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      A[n][m] = A[m][n];

    } /* n */
  } /* m */

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim+1] and ResidualH2[dim+1] */

	    ResidualH1[dim+1][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim+1][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim+1][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim+1][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim+1][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim+1][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim+1][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim+1] */

              ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /******************************************************
   calculations of inner products of <s0|y0> and <y0|y0>
   in order to estimate the parameter "al".
  ******************************************************/

  my_sy = 0.0;
  my_yy = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    tmp1 = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      tmp2 = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      my_sy += tmp1*tmp2; 
	      my_yy += tmp2*tmp2; 
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sy, &sy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_yy, &yy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* al < sy/yy */

  al = sy/yy - 0.2;

  /****************************************************
         calculations of inner products of <r|y> 
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

                /* m */
		s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][0][Mc_AN][h_AN][i][j] - ResidualH1[n+1][0][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][1][Mc_AN][h_AN][i][j] - ResidualH1[n+1][1][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][2][Mc_AN][h_AN][i][j] - ResidualH1[n+1][2][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH1[n][3][Mc_AN][h_AN][i][j] - ResidualH1[n+1][3][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][0][Mc_AN][h_AN][i][j] - ResidualH2[n+1][0][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][1][Mc_AN][h_AN][i][j] - ResidualH2[n+1][1][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

                /* m */
		s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
		y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
                r = s - al*y;                                                                    /* r */
                /* n */
		y = ResidualH2[n][2][Mc_AN][h_AN][i][j] - ResidualH2[n+1][2][Mc_AN][h_AN][i][j]; /* y */
		my_sum += r*y; 

	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){

		  /* m */
		  s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           /* s */
		  y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; /* y */
		  r = s - al*y;                                                                          /* r */
		  /* n */
		  y = ResidualH1[n][spin][Mc_AN][h_AN][i][j] - ResidualH1[n+1][spin][Mc_AN][h_AN][i][j]; /* y */
		  my_sum += r*y; 

		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    } /* n */
  } /* m */    

  Inverse(dim-1,A,IA);

  /****************************************************
    calculations of inner products of <r|OptResidualH> 
  ****************************************************/

  for (m=0; m<dim; m++){

    my_sum = 0.0;

    if (SpinP_switch==3){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH1[dim+1][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	      s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
              or = ResidualH2[dim+1][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	      my_sum += r*or; 

	    }
	  }
	}
      }

    } /* if (SpinP_switch==3 */

    else{

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           
		y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; 
		r = s - al*y;                                                                          
		or = ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j];                                   
		my_sum += r*or; 
	      }
	    }
	  }
	}
      }

    } /* else */

    MPI_Allreduce(&my_sum, &ror[m], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  } /* m */    

  /****************************************************
     calculation of \sum_j b_{ij} * <r_j|OptResidualH> 
  ****************************************************/
    
  for (m=0; m<dim; m++){
    sum = 0.0;  
    for (n=0; n<dim; n++){
      sum += IA[m][n]*ror[n];  
    }

    coes2[m] = sum;
  }

  /****************************************************
                 mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];

	      s = HisH1[m][0][Mc_AN][h_AN][i][j] - HisH1[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][0][Mc_AN][h_AN][i][j] - ResidualH1[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h10 -= r*coes2[m];

	      s = HisH1[m][1][Mc_AN][h_AN][i][j] - HisH1[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][1][Mc_AN][h_AN][i][j] - ResidualH1[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h11 -= r*coes2[m];

	      s = HisH1[m][2][Mc_AN][h_AN][i][j] - HisH1[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][2][Mc_AN][h_AN][i][j] - ResidualH1[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h12 -= r*coes2[m];

	      s = HisH1[m][3][Mc_AN][h_AN][i][j] - HisH1[m+1][3][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[m][3][Mc_AN][h_AN][i][j] - ResidualH1[m+1][3][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h13 -= r*coes2[m];

	      s = HisH2[m][0][Mc_AN][h_AN][i][j] - HisH2[m+1][0][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][0][Mc_AN][h_AN][i][j] - ResidualH2[m+1][0][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h20 -= r*coes2[m];

	      s = HisH2[m][1][Mc_AN][h_AN][i][j] - HisH2[m+1][1][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][1][Mc_AN][h_AN][i][j] - ResidualH2[m+1][1][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h21 -= r*coes2[m];

	      s = HisH2[m][2][Mc_AN][h_AN][i][j] - HisH2[m+1][2][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH2[m][2][Mc_AN][h_AN][i][j] - ResidualH2[m+1][2][Mc_AN][h_AN][i][j]; /* y */
	      r = s - al*y;                                                                    /* r */
	      h22 -= r*coes2[m];
	    }

	    H[0][Mc_AN][h_AN][i][j]    = h10 - al*ResidualH1[dim+1][0][Mc_AN][h_AN][i][j];
	    H[1][Mc_AN][h_AN][i][j]    = h11 - al*ResidualH1[dim+1][1][Mc_AN][h_AN][i][j];
	    H[2][Mc_AN][h_AN][i][j]    = h12 - al*ResidualH1[dim+1][2][Mc_AN][h_AN][i][j];
	    H[3][Mc_AN][h_AN][i][j]    = h13 - al*ResidualH1[dim+1][3][Mc_AN][h_AN][i][j];

            iHNL[0][Mc_AN][h_AN][i][j] = h20 - al*ResidualH2[dim+1][0][Mc_AN][h_AN][i][j];
            iHNL[1][Mc_AN][h_AN][i][j] = h21 - al*ResidualH2[dim+1][1][Mc_AN][h_AN][i][j];
            iHNL[2][Mc_AN][h_AN][i][j] = h22 - al*ResidualH2[dim+1][2][Mc_AN][h_AN][i][j];

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      h = 0.0;
	      for (m=0; m<dim; m++){

		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
		s = HisH1[m][spin][Mc_AN][h_AN][i][j] - HisH1[m+1][spin][Mc_AN][h_AN][i][j];           
		y = ResidualH1[m][spin][Mc_AN][h_AN][i][j] - ResidualH1[m+1][spin][Mc_AN][h_AN][i][j]; 
		r = s - al*y;                                                                          
		h -= r*coes2[m];
	      }

	      H[spin][Mc_AN][h_AN][i][j] = h - al*ResidualH1[dim+1][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
                   freeing of arrays 
  ****************************************************/

  free(coes);
  free(coes2);
  free(ror);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);
}










void Pulay_Mixing_H_with_One_Shot_Hessian(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan;
  double my_sum,tmp1,tmp2,alpha;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double my_sy,my_yy,sy,yy,norm,s,y,or,al,be;
  double **A,**IA,*coes;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      A[n][m] = A[m][n];

    } /* n */
  } /* m */

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim] and ResidualH2[dim] */

	    ResidualH1[dim][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim] */

              ResidualH1[dim][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /****************************************************
           innner products of <s|y> and <y|y>
  ****************************************************/

  my_sy = 0.0;
  my_yy = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    tmp1 = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 

	    tmp1 = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    tmp2 = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
	    my_sy += tmp1*tmp2; 
	    my_yy += tmp2*tmp2; 
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      tmp1 = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      tmp2 = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      my_sy += tmp1*tmp2; 
	      my_yy += tmp2*tmp2; 
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sy, &sy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&my_yy, &yy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* al < sy/yy */

  al = sy/yy - 0.2;

  /* be = 1/(<s|y>-al*<y|y>) */

  be = 1.0/(sy-al*yy);

  /****************************************************
      inner product of (<s|-al<y|)|OptResidualH>
  ****************************************************/
 
  my_sum = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    s = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;

	    s = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    my_sum += (s-al*y)*or;
	  }
	}
      }
    }

  } /* if (SpinP_switch==3 */

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      s = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
	      or = ResidualH1[dim][spin][Mc_AN][h_AN][i][j];                                       /* OptResidualH */
	      my_sum += (s-al*y)*or;
	    }
	  }
	}
      }
    }

  } /* else */

  MPI_Allreduce(&my_sum, &norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  be = norm*be;

  /****************************************************
                 mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

	    s = HisH1[0][0][Mc_AN][h_AN][i][j] - HisH1[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][0][Mc_AN][h_AN][i][j] - ResidualH1[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[0][Mc_AN][h_AN][i][j] = h10 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][1][Mc_AN][h_AN][i][j] - HisH1[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][1][Mc_AN][h_AN][i][j] - ResidualH1[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[1][Mc_AN][h_AN][i][j] = h11 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][2][Mc_AN][h_AN][i][j] - HisH1[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][2][Mc_AN][h_AN][i][j] - ResidualH1[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[2][Mc_AN][h_AN][i][j] = h12 - alpha*(al*or + (s-al*y)*be);

	    s = HisH1[0][3][Mc_AN][h_AN][i][j] - HisH1[1][3][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH1[0][3][Mc_AN][h_AN][i][j] - ResidualH1[1][3][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH1[dim][3][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    H[3][Mc_AN][h_AN][i][j] = h13 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][0][Mc_AN][h_AN][i][j] - HisH2[1][0][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][0][Mc_AN][h_AN][i][j] - ResidualH2[1][0][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][0][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[0][Mc_AN][h_AN][i][j] = h20 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][1][Mc_AN][h_AN][i][j] - HisH2[1][1][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][1][Mc_AN][h_AN][i][j] - ResidualH2[1][1][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][1][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[1][Mc_AN][h_AN][i][j] = h21 - alpha*(al*or + (s-al*y)*be);

	    s = HisH2[0][2][Mc_AN][h_AN][i][j] - HisH2[1][2][Mc_AN][h_AN][i][j];           /* s */
	    y = ResidualH2[0][2][Mc_AN][h_AN][i][j] - ResidualH2[1][2][Mc_AN][h_AN][i][j]; /* y */
            or = ResidualH2[dim][2][Mc_AN][h_AN][i][j];                                    /* OptResidualH */
	    iHNL[2][Mc_AN][h_AN][i][j] = h22 - alpha*(al*or + (s-al*y)*be);
	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      h = 0.0;
	      for (m=0; m<dim; m++){
		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

	      s = HisH1[0][spin][Mc_AN][h_AN][i][j] - HisH1[1][spin][Mc_AN][h_AN][i][j];           /* s */
	      y = ResidualH1[0][spin][Mc_AN][h_AN][i][j] - ResidualH1[1][spin][Mc_AN][h_AN][i][j]; /* y */
              or = ResidualH1[dim][spin][Mc_AN][h_AN][i][j];                                       /* OptResidualH */
	      H[spin][Mc_AN][h_AN][i][j] = h - alpha*(al*or + (s-al*y)*be);

	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
                   freeing of arrays 
  ****************************************************/

  free(coes);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);
}





void Pulay_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int Mc_AN,Gc_AN,Cwan,Hwan,h_AN,Gh_AN,i,j,spin;
  int dim,m,n,flag_nan,tno;
  double my_sum,tmp1,tmp2,alpha,max_diff,d;
  double r,r10,r11,r12,r13,r20,r21,r22;
  double h,h10,h11,h12,h13,h20,h21,h22;
  double **A,**IA,*coes,**metric;
  char nanchar[300];

  /****************************************************
       determination of dimension of the subspace
  ****************************************************/

  if (SCF_iter<=Num_Mixing_pDM) dim = SCF_iter-1;
  else                          dim = Num_Mixing_pDM;

  /****************************************************
                allocation of arrays 
  ****************************************************/

  coes = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  metric = (double**)malloc(sizeof(double*)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      tno = 1; 
    }
    else{
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    metric[Mc_AN] = (double*)malloc(sizeof(double)*tno);
  }  

  /****************************************************
     determine metric used for calculations of norm
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
    for (i=0; i<Spe_Total_NO[Cwan]; i++){
      d = fabs(HisH1[0][0][Mc_AN][0][i][i]-ChemP);
      metric[Mc_AN][i] = 5.0/(d*d+5.0);       
    }
  }

  /****************************************************
                 shift the residual H
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
	      ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

	      ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
	      ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
	      ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	    ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	    ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	    ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	    ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	    ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	    ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else{

    /* shift the residual Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* calculate the current residual Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
          calculation of the residual matrix
  ****************************************************/

  for (m=0; m<dim; m++){
    for (n=0; n<dim; n++){

      my_sum = 0.0;

      if (SpinP_switch==3){

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		tmp1 = ResidualH1[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][0][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH1[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][1][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH1[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][2][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH1[m][3][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH1[n][3][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH2[m][0][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][0][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH2[m][1][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][1][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 

		tmp1 = ResidualH2[m][2][Mc_AN][h_AN][i][j];
		tmp2 = ResidualH2[n][2][Mc_AN][h_AN][i][j];
                my_sum += metric[Mc_AN][i]*tmp1*tmp2; 
	      }
	    }
	  }
	}

      } /* if (SpinP_switch==3 */

      else{

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  tmp1 = ResidualH1[m][spin][Mc_AN][h_AN][i][j];
		  tmp2 = ResidualH1[n][spin][Mc_AN][h_AN][i][j];
                  my_sum += metric[Mc_AN][i]*tmp1*tmp2; 
		}
	      }
	    }
	  }
	}

      } /* else */

      MPI_Allreduce(&my_sum, &A[m][n], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
      A[n][m] = A[m][n];

    } /* n */
  } /* m */

  NormRD[0] = A[0][0]/(double)atomnum;

  for (m=1; m<=dim; m++){
    A[m-1][dim] = -1.0;
    A[dim][m-1] = -1.0;
  }
  A[dim][dim] = 0.0;

  Inverse(dim,A,IA);

  for (m=1; m<=dim; m++){
    coes[m] = -IA[m-1][dim];
  }

  /****************************************************
            check "nan", "NaN", "inf" or "Inf"
  ****************************************************/

  flag_nan = 0;
  for (m=1; m<=dim; m++){

    sprintf(nanchar,"%8.4f",coes[m]);
    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	|| strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;
    }
  }

  if (flag_nan==1){
    for (m=1; m<=dim; m++){
      coes[m] = 0.0;
    }
    coes[1] = 0.05;
    coes[2] = 0.95;
  }

  /****************************************************
      calculation of optimum residual Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            r10 = 0.0; 
            r11 = 0.0;
            r12 = 0.0;
            r13 = 0.0;

            r20 = 0.0; 
            r21 = 0.0;
            r22 = 0.0;

	    for (m=0; m<dim; m++){

	      r10 += ResidualH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r11 += ResidualH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r12 += ResidualH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      r13 += ResidualH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      r20 += ResidualH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      r21 += ResidualH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      r22 += ResidualH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

            /* optimum Residual H is stored in ResidualH1[dim] and ResidualH2[dim] */

	    ResidualH1[dim][0][Mc_AN][h_AN][i][j] = r10;
	    ResidualH1[dim][1][Mc_AN][h_AN][i][j] = r11;
	    ResidualH1[dim][2][Mc_AN][h_AN][i][j] = r12;
	    ResidualH1[dim][3][Mc_AN][h_AN][i][j] = r13;

	    ResidualH2[dim][0][Mc_AN][h_AN][i][j] = r20;
	    ResidualH2[dim][1][Mc_AN][h_AN][i][j] = r21;
	    ResidualH2[dim][2][Mc_AN][h_AN][i][j] = r22;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

              /* optimum Residual H is stored in ResidualH1[dim] */

              ResidualH1[dim][spin][Mc_AN][h_AN][i][j] = r;

	    }
	  }
	}
      }
    }
  }

  /****************************************************
                   mixing of Hamiltonian
  ****************************************************/

  if (1.0e-1<=NormRD[0])
    alpha = 0.5;
  else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
    alpha = 0.6;
  else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
    alpha = 0.7;
  else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
    alpha = 0.8;
  else
    alpha = 1.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

            h10 = 0.0; 
            h11 = 0.0;
            h12 = 0.0;
            h13 = 0.0;

            h20 = 0.0; 
            h21 = 0.0;
            h22 = 0.0;

	    for (m=0; m<dim; m++){

	      h10 += HisH1[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h11 += HisH1[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h12 += HisH1[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	      h13 += HisH1[m][3][Mc_AN][h_AN][i][j]*coes[m+1];

	      h20 += HisH2[m][0][Mc_AN][h_AN][i][j]*coes[m+1];
	      h21 += HisH2[m][1][Mc_AN][h_AN][i][j]*coes[m+1];
	      h22 += HisH2[m][2][Mc_AN][h_AN][i][j]*coes[m+1];
	    }

	    H[0][Mc_AN][h_AN][i][j] = h10 + alpha*ResidualH1[dim][0][Mc_AN][h_AN][i][j];
	    H[1][Mc_AN][h_AN][i][j] = h11 + alpha*ResidualH1[dim][1][Mc_AN][h_AN][i][j];
	    H[2][Mc_AN][h_AN][i][j] = h12 + alpha*ResidualH1[dim][2][Mc_AN][h_AN][i][j];
	    H[3][Mc_AN][h_AN][i][j] = h13 + alpha*ResidualH1[dim][3][Mc_AN][h_AN][i][j];
             
	    iHNL[0][Mc_AN][h_AN][i][j] = h20 + alpha*ResidualH2[dim][0][Mc_AN][h_AN][i][j];
	    iHNL[1][Mc_AN][h_AN][i][j] = h21 + alpha*ResidualH2[dim][1][Mc_AN][h_AN][i][j];
	    iHNL[2][Mc_AN][h_AN][i][j] = h22 + alpha*ResidualH2[dim][2][Mc_AN][h_AN][i][j];

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      r = 0.0; 
	      h = 0.0;

	      for (m=0; m<dim; m++){
		r += ResidualH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
		h += HisH1[m][spin][Mc_AN][h_AN][i][j]*coes[m+1];
	      }

	      H[spin][Mc_AN][h_AN][i][j] = h + alpha*r;
	    }
	  }
	}
      }
    }
  }

  /****************************************************
                  shifting of Hamiltonian
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* save the current Hamiltonian */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /****************************************************
                   freeing of arrays 
  ****************************************************/

  free(coes);

  for (i=0; i<List_YOUSO[39]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[39]; i++){
    free(IA[i]);
  }
  free(IA);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
    if (Mc_AN==0){
      tno = 1; 
    }
    else{
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      tno = Spe_Total_NO[Cwan];
    }
    free(metric[Mc_AN]);
  }  
  free(metric);

}



void Simple_Mixing_H(int MD_iter, int SCF_iter, int SCF_iter0 )
{
  int m,Mc_AN,Gc_AN,Cwan,spin,dim;
  int i,j,h_AN,Gh_AN,Hwan,ian,jan,n;
  double w1,w2,My_Norm,Norm;
  double d0,d1,d2,d3,d,tmp0;
  double Mix_wgt,Min_Weight,Max_Weight;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

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

  /* determination of dim */

  if (SCF_iter<Num_Mixing_pDM) dim = SCF_iter;
  else                         dim = Num_Mixing_pDM;

  /****************************************************
                  shift the residual H
  ****************************************************/

  if (Mixing_switch!=1 && Mixing_switch!=6){

    if (SpinP_switch==3){

      /* shift the residual Hamiltonian */

      for (m=(dim-1); 0<m; m--){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){

		ResidualH1[m][0][Mc_AN][h_AN][i][j] = ResidualH1[m-1][0][Mc_AN][h_AN][i][j];
		ResidualH1[m][1][Mc_AN][h_AN][i][j] = ResidualH1[m-1][1][Mc_AN][h_AN][i][j];
		ResidualH1[m][2][Mc_AN][h_AN][i][j] = ResidualH1[m-1][2][Mc_AN][h_AN][i][j];
		ResidualH1[m][3][Mc_AN][h_AN][i][j] = ResidualH1[m-1][3][Mc_AN][h_AN][i][j];

		ResidualH2[m][0][Mc_AN][h_AN][i][j] = ResidualH2[m-1][0][Mc_AN][h_AN][i][j];
		ResidualH2[m][1][Mc_AN][h_AN][i][j] = ResidualH2[m-1][1][Mc_AN][h_AN][i][j];
		ResidualH2[m][2][Mc_AN][h_AN][i][j] = ResidualH2[m-1][2][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }

      /* calculate the current residual Hamiltonian */

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      ResidualH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j] - HisH1[0][0][Mc_AN][h_AN][i][j];
	      ResidualH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j] - HisH1[0][1][Mc_AN][h_AN][i][j];
	      ResidualH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j] - HisH1[0][2][Mc_AN][h_AN][i][j];
	      ResidualH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j] - HisH1[0][3][Mc_AN][h_AN][i][j];

	      ResidualH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j] - HisH2[0][0][Mc_AN][h_AN][i][j];
	      ResidualH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j] - HisH2[0][1][Mc_AN][h_AN][i][j];
	      ResidualH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j] - HisH2[0][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }

    }

    else{

      /* shift the residual Hamiltonian */

      for (m=(dim-1); 0<m; m--){
	for (spin=0; spin<=SpinP_switch; spin++){
	  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	    Gc_AN = M2G[Mc_AN];    
	    Cwan = WhatSpecies[Gc_AN];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      Hwan = WhatSpecies[Gh_AN];
	      for (i=0; i<Spe_Total_NO[Cwan]; i++){
		for (j=0; j<Spe_Total_NO[Hwan]; j++){
		  ResidualH1[m][spin][Mc_AN][h_AN][i][j] = ResidualH1[m-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }
	  }
	}
      }

      /* calculate the current residual Hamiltonian */

      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		ResidualH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j] - HisH1[0][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }

    } /* else */

  } /* if (Mixing_switch!=1 && Mixing_switch!=6) */

  /****************************************************
              calculation of NormRH
  ****************************************************/

  My_Norm = 0.0;

  if (SpinP_switch==3){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    d0 = HisH1[0][0][Mc_AN][h_AN][i][j] - H[0][Mc_AN][h_AN][i][j];  
	    d1 = HisH1[0][1][Mc_AN][h_AN][i][j] - H[1][Mc_AN][h_AN][i][j];  
	    d2 = HisH1[0][2][Mc_AN][h_AN][i][j] - H[2][Mc_AN][h_AN][i][j];  
	    d3 = HisH1[0][3][Mc_AN][h_AN][i][j] - H[3][Mc_AN][h_AN][i][j];  
            My_Norm += d0*d0 + d1*d1 + d2*d2 + d3*d3;

            d0 = HisH2[0][0][Mc_AN][h_AN][i][j] - iHNL[0][Mc_AN][h_AN][i][j];
            d1 = HisH2[0][1][Mc_AN][h_AN][i][j] - iHNL[1][Mc_AN][h_AN][i][j];
            d2 = HisH2[0][2][Mc_AN][h_AN][i][j] - iHNL[2][Mc_AN][h_AN][i][j];
            My_Norm += d0*d0 + d1*d1 + d2*d2;

	  }
	}
      }
    }

  }

  else{

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
  	      d = HisH1[0][spin][Mc_AN][h_AN][i][j] - H[spin][Mc_AN][h_AN][i][j];  
              My_Norm += d*d;
	    }
	  }
	}
      }
    }
  }

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
  NormRD[0] = Norm/(double)atomnum;;
  History_Uele[0] = Uele;

  if (1<SCF_iter){

    if ( (int)sgn(History_Uele[0]-History_Uele[1])
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
   
    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
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
      else{
	Mixing_weight = Max_Weight;
      }

      SCF_RENZOKU = -1;  
    }

    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
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

    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
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
  }

  Mix_wgt = Mixing_weight;

  /****************************************************
            finding a proper mixing weight
  ****************************************************/

  if (SCF_iter==1){
    w1 = 1.0;
    w2 = 1.0 - w1;
  }
  else{
    w1 = Mix_wgt;
    w2 = 1.0 - w1;
  }

  /****************************************************
       performing the simple mixing for Hamiltonian 
  ****************************************************/

  if (SpinP_switch==3){

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){

	      HisH1[m][0][Mc_AN][h_AN][i][j] = HisH1[m-1][0][Mc_AN][h_AN][i][j];
	      HisH1[m][1][Mc_AN][h_AN][i][j] = HisH1[m-1][1][Mc_AN][h_AN][i][j];
	      HisH1[m][2][Mc_AN][h_AN][i][j] = HisH1[m-1][2][Mc_AN][h_AN][i][j];
	      HisH1[m][3][Mc_AN][h_AN][i][j] = HisH1[m-1][3][Mc_AN][h_AN][i][j];

	      HisH2[m][0][Mc_AN][h_AN][i][j] = HisH2[m-1][0][Mc_AN][h_AN][i][j];
	      HisH2[m][1][Mc_AN][h_AN][i][j] = HisH2[m-1][1][Mc_AN][h_AN][i][j];
	      HisH2[m][2][Mc_AN][h_AN][i][j] = HisH2[m-1][2][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

    /* mix the current Hamiltonian and the last one */

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];    
      Cwan = WhatSpecies[Gc_AN];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	for (i=0; i<Spe_Total_NO[Cwan]; i++){
	  for (j=0; j<Spe_Total_NO[Hwan]; j++){

	    H[0][Mc_AN][h_AN][i][j] = w2*HisH1[1][0][Mc_AN][h_AN][i][j] + w1*H[0][Mc_AN][h_AN][i][j];  
	    H[1][Mc_AN][h_AN][i][j] = w2*HisH1[1][1][Mc_AN][h_AN][i][j] + w1*H[1][Mc_AN][h_AN][i][j];  
	    H[2][Mc_AN][h_AN][i][j] = w2*HisH1[1][2][Mc_AN][h_AN][i][j] + w1*H[2][Mc_AN][h_AN][i][j];  
	    H[3][Mc_AN][h_AN][i][j] = w2*HisH1[1][3][Mc_AN][h_AN][i][j] + w1*H[3][Mc_AN][h_AN][i][j];  

	    HisH1[0][0][Mc_AN][h_AN][i][j] = H[0][Mc_AN][h_AN][i][j];
	    HisH1[0][1][Mc_AN][h_AN][i][j] = H[1][Mc_AN][h_AN][i][j];
	    HisH1[0][2][Mc_AN][h_AN][i][j] = H[2][Mc_AN][h_AN][i][j];
	    HisH1[0][3][Mc_AN][h_AN][i][j] = H[3][Mc_AN][h_AN][i][j];

	    iHNL[0][Mc_AN][h_AN][i][j] = w2*HisH2[1][0][Mc_AN][h_AN][i][j] + w1*iHNL[0][Mc_AN][h_AN][i][j];
	    iHNL[1][Mc_AN][h_AN][i][j] = w2*HisH2[1][1][Mc_AN][h_AN][i][j] + w1*iHNL[1][Mc_AN][h_AN][i][j];
	    iHNL[2][Mc_AN][h_AN][i][j] = w2*HisH2[1][2][Mc_AN][h_AN][i][j] + w1*iHNL[2][Mc_AN][h_AN][i][j];

	    HisH2[0][0][Mc_AN][h_AN][i][j] = iHNL[0][Mc_AN][h_AN][i][j];
	    HisH2[0][1][Mc_AN][h_AN][i][j] = iHNL[1][Mc_AN][h_AN][i][j];
	    HisH2[0][2][Mc_AN][h_AN][i][j] = iHNL[2][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }

  }

  else {

    /* shift the current Hamiltonian */

    for (m=dim; 0<m; m--){
      for (spin=0; spin<=SpinP_switch; spin++){
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    for (i=0; i<Spe_Total_NO[Cwan]; i++){
	      for (j=0; j<Spe_Total_NO[Hwan]; j++){
		HisH1[m][spin][Mc_AN][h_AN][i][j] = HisH1[m-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    /* mix the current Hamiltonian and the last one */

    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  for (i=0; i<Spe_Total_NO[Cwan]; i++){
	    for (j=0; j<Spe_Total_NO[Hwan]; j++){
	      H[spin][Mc_AN][h_AN][i][j] = w2*HisH1[1][spin][Mc_AN][h_AN][i][j] + w1*H[spin][Mc_AN][h_AN][i][j];
	      HisH1[0][spin][Mc_AN][h_AN][i][j] = H[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}
      }
    }

  } /* else */

  /* In case of RMM-DIIS */

  if (Mixing_switch==1 || Mixing_switch==6){

    for (spin=0; spin<=SpinP_switch; spin++){

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];
	ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];      
	  jan = Spe_Total_CNO[WhatSpecies[Gh_AN]];

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){
  
              ResidualDM[2][spin][Mc_AN][h_AN][m][n] = DM[0][spin][Mc_AN][h_AN][m][n]
                                                      -DM[1][spin][Mc_AN][h_AN][m][n];

	      DM[2][spin][Mc_AN][h_AN][m][n] = DM[1][spin][Mc_AN][h_AN][m][n];
	      DM[1][spin][Mc_AN][h_AN][m][n] = DM[0][spin][Mc_AN][h_AN][m][n];
	    }
	  }

	  if ( (SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
             || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )) && spin<=1 ){ 

	    for (m=0; m<ian; m++){
	      for (n=0; n<jan; n++){

	        iResidualDM[2][spin][Mc_AN][h_AN][m][n] = iDM[0][spin][Mc_AN][h_AN][m][n]
                                                         -iDM[1][spin][Mc_AN][h_AN][m][n];

		iDM[2][spin][Mc_AN][h_AN][m][n] = iDM[1][spin][Mc_AN][h_AN][m][n];
		iDM[1][spin][Mc_AN][h_AN][m][n] = iDM[0][spin][Mc_AN][h_AN][m][n];
	      }
	    }
	  }

	}
      }

    } /* spin */

  } /* if (Mixing_switch==1 || Mixing_switch==6) */

}





void Inverse(int n, double **a, double **ia)
{
  int method_flag=2;

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

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
    for (j=0; j<List_YOUSO[39]; j++){
      da[i][j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
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

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[39]; i++){
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


void Inverse0(int n, double **a, double **ia)
{
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

    x[List_YOUSO[39]]
    y[List_YOUSO[39]]
    da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[39]);

  da = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
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

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[39]; i++){
    free(da[i]);
  }
  free(da);
}
