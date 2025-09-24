/**********************************************************************
  DIIS_Mixing_DM.c:

     DIIS_Mixing_DM.c is a subroutine to achieve self-consistent field
     using the direct inversion in the iterative subspace.

  Log of DIIS_Mixing_DM.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "openmx_common.h"
#include "mpi.h"


static void Inverse(int n, double **a, double **ia);




void DIIS_Mixing_DM(int SCF_iter, double ******ResidualDM, double ******iResidualDM)
{
  static int firsttime=1;
  int spin,Mc_AN,Gc_AN,wan1,TNO1,h_AN,Gh_AN;
  int wan2,TNO2,i,j,k,NumMix,NumSlide;
  int SCFi,SCFj,tno1,tno0,Cwan,Hwan; 
  int pSCF_iter,size_OptRDM,size_iOptRDM;
  int SpinP_num,flag_nan;
  double *alden;
  double **A,My_A;
  double **IA;
  double Av_dia,IAv_dia,tmp0;
  double OptNorm_RDM,My_OptNorm_RDM;
  double dum1,dum2,bunsi,bunbo,sum,coef_OptRDM;
  double *****OptRDM;
  double *****iOptRDM;
  int numprocs,myid;
  char nanchar[300];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  SpinP_num = SpinP_switch;

  /****************************************************
    allocation of arrays:

    double alden[List_YOUSO[16]];
    double A[List_YOUSO[16]][List_YOUSO[16]];
    double IA[List_YOUSO[16]][List_YOUSO[16]];
    double OptRDM[SpinP_num+1]
                        [Matomnum+1]
                        [FNAN[Gc_AN]+1]
                        [Spe_Total_NO[Cwan]]
                        [Spe_Total_NO[Hwan]];
  ****************************************************/

  alden = (double*)malloc(sizeof(double)*List_YOUSO[16]);

  A = (double**)malloc(sizeof(double*)*List_YOUSO[16]);
  for (i=0; i<List_YOUSO[16]; i++){
    A[i] = (double*)malloc(sizeof(double)*List_YOUSO[16]);
  }

  IA = (double**)malloc(sizeof(double*)*List_YOUSO[16]);
  for (i=0; i<List_YOUSO[16]; i++){
    IA[i] = (double*)malloc(sizeof(double)*List_YOUSO[16]);
  }

  /* OptRDM */

  size_OptRDM = 0;
  OptRDM = (double*****)malloc(sizeof(double****)*(SpinP_num+1));
  for (k=0; k<=SpinP_num; k++){
    OptRDM[k] = (double****)malloc(sizeof(double***)*(atomnum+1)); 
    FNAN[0] = 0;

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      OptRDM[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        OptRDM[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
        for (i=0; i<tno0; i++){
          OptRDM[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
        }
        size_OptRDM += tno0*tno1;
      }
    }
  }

  /* iOptRDM */

  if (SpinP_switch==3 && ( SO_switch==1
                        || Hub_U_switch==1
                        || 1<=Constraint_NCS_switch
                        || Zeeman_NCS_switch==1
                        || Zeeman_NCO_switch==1 )){ 

    size_iOptRDM = 0;
    iOptRDM = (double*****)malloc(sizeof(double****)*2);
    for (k=0; k<2; k++){
      iOptRDM[k] = (double****)malloc(sizeof(double***)*(atomnum+1)); 
      FNAN[0] = 0;

      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	iOptRDM[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  iOptRDM[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	  for (i=0; i<tno0; i++){
	    iOptRDM[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	  }
	  size_iOptRDM += tno0*tno1;
	}
      }
    }
  }

  /****************************************************
     start calc.
  ****************************************************/
 
  if (SCF_iter==1){
    /* memory calc. done, only 1st iteration */
    if (firsttime) {
      PrintMemory("DIIS_Mixing_DM: OptRDM",sizeof(double)*size_OptRDM,NULL);
      if (SpinP_switch==3 && ( SO_switch==1 
                            || Hub_U_switch==1
                            || 1<=Constraint_NCS_switch
                            || Zeeman_NCS_switch==1
                            || Zeeman_NCO_switch==1 )){ 
        PrintMemory("DIIS_Mixing_DM: iOptRDM",sizeof(double)*size_iOptRDM,NULL);
      }
      firsttime=0;
    }

    Simple_Mixing_DM(0,1.00,DM[0],DM[1],DM[2],iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);
  } 

  else if (SCF_iter==2){

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      ResidualDM[2][spin][Mc_AN][h_AN][i][j] =
 		      DM[0][spin][Mc_AN][h_AN][i][j]
                     -DM[1][spin][Mc_AN][h_AN][i][j];
	    }
	  }

          if (SpinP_switch==3
              &&
             ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )
              && spin<2){ 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
	        iResidualDM[2][spin][Mc_AN][h_AN][i][j] =
 		        iDM[0][spin][Mc_AN][h_AN][i][j]
                       -iDM[1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }        

	}
      }
    }

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      DM[2][spin][Mc_AN][h_AN][i][j] = DM[1][spin][Mc_AN][h_AN][i][j];
	    }
	  }

          if (SpinP_switch==3
            && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
            && spin<2){ 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
	        iDM[2][spin][Mc_AN][h_AN][i][j] = iDM[1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
    }

    Simple_Mixing_DM(0,Mixing_weight,DM[0],DM[1],DM[2],
		     iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);

  } 
  else {

    /****************************************************
                        Calc of RDM1
    ****************************************************/

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){
	      ResidualDM[1][spin][Mc_AN][h_AN][i][j] =
		      DM[0][spin][Mc_AN][h_AN][i][j] - DM[1][spin][Mc_AN][h_AN][i][j];
	    }
	  }

          if (SpinP_switch==3
             && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
             && spin<2){ 
	
            for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
	        iResidualDM[1][spin][Mc_AN][h_AN][i][j] =
		        iDM[0][spin][Mc_AN][h_AN][i][j] - iDM[1][spin][Mc_AN][h_AN][i][j];
	      }
	    }
	  }
	}
      }
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
                        alpha from RDM
    ****************************************************/

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      for (SCFj=SCFi; SCFj<=NumMix; SCFj++){

        My_A = 0.0;

#pragma omp parallel for reduction(+:My_A) private(spin, Mc_AN, Gc_AN, wan1, TNO1, h_AN, Gh_AN, wan2, TNO2, i, j, dum1, dum2) if(SpinP_num > 1) schedule(static,1)
        for (spin=0; spin<=SpinP_num; spin++){
          for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
            Gc_AN = M2G[Mc_AN];
	    wan1 = WhatSpecies[Gc_AN];
	    TNO1 = Spe_Total_CNO[wan1];
            for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];
	      wan2 = WhatSpecies[Gh_AN];
	      TNO2 = Spe_Total_CNO[wan2];

              for (i=0; i<TNO1; i++){
                for (j=0; j<TNO2; j++){
                  dum1 = ResidualDM[SCFi][spin][Mc_AN][h_AN][i][j];
                  dum2 = ResidualDM[SCFj][spin][Mc_AN][h_AN][i][j];
                  My_A += dum1*dum2;
                }
              }

              if (SpinP_switch==3
                  &&
                 ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
                  && spin<2){
 
                for (i=0; i<TNO1; i++){
                  for (j=0; j<TNO2; j++){
                    dum1 = iResidualDM[SCFi][spin][Mc_AN][h_AN][i][j];
                    dum2 = iResidualDM[SCFj][spin][Mc_AN][h_AN][i][j];
                    My_A += dum1*dum2;
                  }
                }
	      }

            }
          }
	}

        /* MPI My_A */

        MPI_Allreduce(&My_A, &A[SCFi-1][SCFj-1], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

        A[SCFj-1][SCFi-1] = A[SCFi-1][SCFj-1];
      }
    }

    Av_dia = A[0][0];
    NormRD[0] = A[0][0];

    for (SCFi=1; SCFi<=NumMix; SCFi++){
      A[SCFi-1][NumMix] = -1.0;
      A[NumMix][SCFi-1] = -1.0;
    }
    A[NumMix][NumMix] = 0.0;

#pragma forceinline
    Inverse(NumMix,A,IA);
    for (SCFi=1; SCFi<=NumMix; SCFi++){
      alden[SCFi] = -IA[SCFi-1][NumMix];

      /*
      printf("AP Alpha_%i=%15.12f\n",SCFi,alden[SCFi]);
      */
    }

    /*
    printf("Lambda=%15.12f\n",-IA[NumMix][NumMix]);
    */

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
              Calculate optimized residual DM 
    ****************************************************/

    My_OptNorm_RDM = 0.0;

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){

              /* Pulay mixing */

              sum = 0.0;
	      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		sum += alden[pSCF_iter]*ResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
	      }

              OptRDM[spin][Mc_AN][h_AN][i][j] = sum;
              My_OptNorm_RDM += sum*sum;

	    }
	  }

          if (SpinP_switch==3
             &&
            ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
             &&
             spin<2){
 
	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){

		/* Pulay mixing */

		sum = 0.0;
		for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		  sum += alden[pSCF_iter]*iResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
		}

		iOptRDM[spin][Mc_AN][h_AN][i][j] = sum;
		My_OptNorm_RDM += sum*sum;

	      }
	    }
	  }

	}
      }
    }

    /* MPI My_OptNorm_RDM */
    MPI_Allreduce(&My_OptNorm_RDM, &OptNorm_RDM, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    /*
    printf("OptNorm_RDM=%18.12f\n",OptNorm_RDM);
    */

    if (1.0e-1<=NormRD[0])
      coef_OptRDM = 0.5;
    else if (1.0e-2<=NormRD[0] && NormRD[0]<1.0e-1)
      coef_OptRDM = 0.6;
    else if (1.0e-3<=NormRD[0] && NormRD[0]<1.0e-2)
      coef_OptRDM = 0.7;
    else if (1.0e-4<=NormRD[0] && NormRD[0]<1.0e-3)
      coef_OptRDM = 0.8;
    else
      coef_OptRDM = 1.0;

    /****************************************************
                        Mixing of DM 
    ****************************************************/

    for (spin=0; spin<=SpinP_num; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        Gc_AN = M2G[Mc_AN];
	wan1 = WhatSpecies[Gc_AN];
	TNO1 = Spe_Total_CNO[wan1];
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = natn[Gc_AN][h_AN];
	  wan2 = WhatSpecies[Gh_AN];
	  TNO2 = Spe_Total_CNO[wan2];

	  for (i=0; i<TNO1; i++){
	    for (j=0; j<TNO2; j++){

              /* Pulay mixing */

              sum = 0.0;
	      for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		sum += alden[pSCF_iter]*DM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
	      }

              /* Correction by optimized RDM */

              DM[0][spin][Mc_AN][h_AN][i][j] = sum + coef_OptRDM*OptRDM[spin][Mc_AN][h_AN][i][j]; 
	    }
	  }

          if (SpinP_switch==3
            &&
           ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
            && spin<2){ 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){

		/* Pulay mixing */

		sum = 0.0;
		for (pSCF_iter=1; pSCF_iter<=NumMix; pSCF_iter++){
		  sum += alden[pSCF_iter]*iDM[pSCF_iter][spin][Mc_AN][h_AN][i][j];
		}

		/* Correction by optimized RDM */

		iDM[0][spin][Mc_AN][h_AN][i][j] = sum + coef_OptRDM*iOptRDM[spin][Mc_AN][h_AN][i][j]; 
	      }
	    }
	  }

	}
      }
    }

    /****************************************************
                         Shift of DM
    ****************************************************/

    for (pSCF_iter=NumSlide; 0<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<=SpinP_num; spin++){
        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
          Gc_AN = M2G[Mc_AN];
	  wan1 = WhatSpecies[Gc_AN]; 
	  TNO1 = Spe_Total_CNO[wan1];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];          
	    TNO2 = Spe_Total_CNO[wan2]; 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
		DM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
                =
                DM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }

            if (SpinP_switch==3
                &&
               (SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
                && spin<2){
 
	      for (i=0; i<TNO1; i++){
		for (j=0; j<TNO2; j++){
		  iDM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
		  =
		  iDM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }

	  }
	}
      }
    }

    /****************************************************
                         Shift of RDM
    ****************************************************/

    for (pSCF_iter=NumSlide; 1<pSCF_iter; pSCF_iter--){
      for (spin=0; spin<=SpinP_num; spin++){
        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
          Gc_AN = M2G[Mc_AN];
	  wan1 = WhatSpecies[Gc_AN]; 
	  TNO1 = Spe_Total_CNO[wan1];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];          
	    TNO2 = Spe_Total_CNO[wan2]; 

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
		ResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
                =
                ResidualDM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
	      }
	    }

            if (SpinP_switch==3
               &&
               ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)
               && spin<2){ 
	      for (i=0; i<TNO1; i++){
		for (j=0; j<TNO2; j++){
		  iResidualDM[pSCF_iter][spin][Mc_AN][h_AN][i][j]
		  =
		  iResidualDM[pSCF_iter-1][spin][Mc_AN][h_AN][i][j];
		}
	      }
	    }
	  }
	}
      }
    }
  }

  /****************************************************
    freeing of arrays:

    double alden[List_YOUSO[16]];
    double A[List_YOUSO[16]][List_YOUSO[16]];
    double IA[List_YOUSO[16]][List_YOUSO[16]];
    double OptRDM[SpinP_num+1]
                        [Matomnum+1]
                        [FNAN[Gc_AN]+1]
                        [Spe_Total_NO[Cwan]]
                        [Spe_Total_NO[Hwan]];
  ****************************************************/

  free(alden);

  for (i=0; i<List_YOUSO[16]; i++){
    free(A[i]);
  }
  free(A);

  for (i=0; i<List_YOUSO[16]; i++){
    free(IA[i]);
  }
  free(IA);

  /* OptRDM */

  for (k=0; k<=SpinP_num; k++){
    FNAN[0] = 0;

    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
        Gc_AN = 0;
        tno0 = 1;
      }
      else{
        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];  
      }    

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

        if (Mc_AN==0){
          tno1 = 1;  
        }
        else{
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        } 

        for (i=0; i<tno0; i++){
          free(OptRDM[k][Mc_AN][h_AN][i]);
        }
        free(OptRDM[k][Mc_AN][h_AN]);
      }
      free(OptRDM[k][Mc_AN]);
    }
    free(OptRDM[k]);
  }
  free(OptRDM);

  /* iOptRDM */

  if (SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
      || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1)){ 

    for (k=0; k<2; k++){
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_NO[Cwan];  
	}    

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_NO[Hwan];
	  } 

	  for (i=0; i<tno0; i++){
	    free(iOptRDM[k][Mc_AN][h_AN][i]);
	  }
          free(iOptRDM[k][Mc_AN][h_AN]);
	}
        free(iOptRDM[k][Mc_AN]);
      }
      free(iOptRDM[k]);
    }
    free(iOptRDM);
  }

}




void Inverse(int n, double **a, double **ia)
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

     x[List_YOUSO[16]]
     y[List_YOUSO[16]]
     da[List_YOUSO[16]][List_YOUSO[16]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[16]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[16]);

  da = (double**)malloc(sizeof(double*)*List_YOUSO[16]);
  for (i=0; i<List_YOUSO[16]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[16]);
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[16]; i++){
      for (j=0; j<List_YOUSO[16]; j++){
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

     x[List_YOUSO[16]]
     y[List_YOUSO[16]]
     da[List_YOUSO[16]][List_YOUSO[16]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[16]; i++){
    free(da[i]);
  }
  free(da);

}
