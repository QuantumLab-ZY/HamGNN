/**********************************************************************
  TRAN_Apply_Bias2e.c:

  TRAN_Apply_Bias2e.c is a subroutine to apply source-drain voltage 
  to the electrodes. 

  Log of TRAN_Apply_Bias2e.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"



void  TRAN_Apply_Bias2e(
       MPI_Comm comm1,
       int side,

       double voltage, /* applied bias */

       double TRAN_eV2Hartree,
       int SpinP_switch,
       int atomnum,

       int *WhatSpecies,
       int *Spe_Total_CNO,
       int *FNAN,
       int **natn,
       
       int Ngrid1,
       int Ngrid2,
       int Ngrid3,
       double ****OLP,

       /* output: overwritten */
       double *ChemP, 
       double *****H, 
       double *dVHart_Grid
)
{
  int myid;

  MPI_Comm_rank(comm1,&myid);

  {
    int GA_AN,wanA,tnoA;
    int LB_AN,GB_AN,wanB,tnoB; 
    int i,j,k; 
    int myid;

    MPI_Comm_rank(comm1,&myid);

    /* ChemP */
    (*ChemP) += voltage; 

    /*   <i|H+V|j> = <i|H|j> + V<i|j>  */
    
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];

        for (i=0; i<tnoA; i++){
          for (j=0; j<tnoB; j++){
/* revised by Y. Xiao for Noncollinear NEGF calculations */
           if (SpinP_switch<2) {
            for (k=0; k<=SpinP_switch; k++) {
	      H[k][GA_AN][LB_AN][i][j] += voltage*OLP[GA_AN][LB_AN][i][j];
            }
           } else {
            H[0][GA_AN][LB_AN][i][j] += voltage*OLP[GA_AN][LB_AN][i][j];
            H[1][GA_AN][LB_AN][i][j] += voltage*OLP[GA_AN][LB_AN][i][j];
           }
/* until here by Y. Xiao for Noncollinear NEGF calculations */
          }  /* j */
        } /* i */
      } /* LB_AN */
    }   /* GA_AN */
  }

  { 
    int i;
    
    /* grid_value += voltage */
    
    for (i=0; i<Ngrid1*Ngrid2*Ngrid3; i++) {
      dVHart_Grid[i] += voltage ; 
    }
  }

  {
    char *s_vec[20];
    s_vec[0] = "left";
    s_vec[1] = "right";

    if (myid==Host_ID){
      printf("  add voltage =%8.4f (eV) to the %5s lead: new ChemP (eV): %8.4f\n",
                voltage*TRAN_eV2Hartree,s_vec[side],(*ChemP)*TRAN_eV2Hartree);
    }
  }

}



