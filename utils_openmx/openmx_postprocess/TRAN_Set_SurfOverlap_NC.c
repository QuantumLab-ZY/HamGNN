/**********************************************************************
  TRAN_Set_SurfOverlap_NC.c:

  TRAN_Set_SurfOverlap_NC.c is a subroutine to construct H00_e, H01_e, 
  S00_e, and S01_e to make surface Green's functions 

  Log of TRAN_Set_SurfOverlap_NC.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"

/***************************************** 

explanation:

atv_ijk[Rn][1..3] has index to the supercell


Rn==0, atv_ijk[Rn][1..3]=0 => means the same cell   => S00

atv_ijk[Rn][1]=1 or -1 means the overlap to the nearest neighbor
cell in the direction of the electrode 
1 or -1 => S01 

assuming atv_ijk[Rn][1]=-1,0 or 1 

---------------------------------------------------------
output 

double S00_nc_e, S01_nc_e, H00_nc_e, H01_nc_e, 

***********************************************/

#define S00_ref(i,j) ( ((j)-1)*NUM+(i)-1 ) 
#define S01_ref(i,j) ( ((j+NUM0)-1)*NUM+(i)-1 )
#define S10_ref(i,j) ( ((j)-1)*NUM+(i+NUM0)-1 )
#define S11_ref(i,j) ( ((j+NUM0)-1)*NUM+(i+NUM0)-1 )

void TRAN_Set_SurfOverlap_NC( MPI_Comm comm1,
                           char *position, 
                           double k2,
                           double k3)  
{
  int NUM,NUM0,n2;
  int Anum, Bnum, wanA, tnoA;
  int i,j,k;
  int GA_AN;
  int GB_AN, LB_AN,wanB, tnoB,Rn;
  int l1,l2,l3; 
  int direction,iside;
  double si,co,kRn;
  double s,h[10];
  double a00,a11,a01r,a01i,b00,b11,b01;
  static double epscutoff=1.0e-8;
  int *MP;
  /*debug */
  char msg[100];
  /*end debug */

  /*
  printf("<TRAN_Set_SurfOverlap, direction=%s>\n",position);
  */

  /* position -> direction */
  if      ( strcasecmp(position,"left")==0) {
    direction = -1;
    iside = 0;
  }
  else if ( strcasecmp(position,"right")==0) {
    direction = 1;
    iside = 1;
  } 

  /* set MP */
/*  TRAN_Set_MP(0, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &NUM, &i); */
  MP = (int*)malloc(sizeof(int)*(atomnum_e[iside]+1));
  TRAN_Set_MP(1, atomnum_e[iside], WhatSpecies_e[iside], Spe_Total_CNO_e[iside], &NUM, MP);
  NUM0=NUM;
  NUM=2*NUM;

  n2 = NUM + 1;   

  for (i=0; i<n2*n2; i++){
    S00_nc_e[iside][i].r = 0.0;
    S00_nc_e[iside][i].i = 0.0;
    S01_nc_e[iside][i].r = 0.0;
    S01_nc_e[iside][i].i = 0.0;
  }
  
  for (i=0; i<n2*n2; i++){
    H00_nc_e[iside][i].r = 0.0;
    H00_nc_e[iside][i].i = 0.0;
    H01_nc_e[iside][i].r = 0.0;
    H01_nc_e[iside][i].i = 0.0;
  }

  for (GA_AN=1; GA_AN<=atomnum_e[iside]; GA_AN++){

    wanA = WhatSpecies_e[iside][GA_AN];
    tnoA = Spe_Total_CNO_e[iside][wanA];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN_e[iside][GA_AN]; LB_AN++){

      GB_AN = natn_e[iside][GA_AN][LB_AN];
      Rn = ncn_e[iside][GA_AN][LB_AN];
      wanB = WhatSpecies_e[iside][GB_AN];
      tnoB = Spe_Total_CNO_e[iside][wanB];
      Bnum = MP[GB_AN];

      l1 = atv_ijk_e[iside][Rn][1];
      l2 = atv_ijk_e[iside][Rn][2];
      l3 = atv_ijk_e[iside][Rn][3];

      kRn = k2*(double)l2 + k3*(double)l3;
      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);

      if (l1==0) {
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            /* S_alpha-alpha */
	    S00_nc_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
	    S00_nc_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            /* S_beta-beta */
            S00_nc_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S00_nc_e[iside][S11_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            
            a00=H_e[iside][0][GA_AN][LB_AN][i][j];
            a11=H_e[iside][1][GA_AN][LB_AN][i][j];
            a01r=H_e[iside][2][GA_AN][LB_AN][i][j];
            a01i=H_e[iside][3][GA_AN][LB_AN][i][j];

            b00=iHNL_e[iside][0][GA_AN][LB_AN][i][j];
            b11=iHNL_e[iside][1][GA_AN][LB_AN][i][j];
            b01=iHNL_e[iside][2][GA_AN][LB_AN][i][j];
            /* H_alpha-alpha */
            H00_nc_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*a00 - si*b00;
            H00_nc_e[iside][S00_ref(Anum+i,Bnum+j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            H00_nc_e[iside][S01_ref(Anum+i,Bnum+j)].r += co*a01r - si*(a01i+b01);
            H00_nc_e[iside][S01_ref(Anum+i,Bnum+j)].i += si*a01r + co*(a01i+b01);
            /* H_beta-beta */
            H00_nc_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*a11 - si*b11;
            H00_nc_e[iside][S11_ref(Anum+i,Bnum+j)].i += co*b11 + si*a11;
           
            /*
            S00_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S00_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];

            for (k=0; k<=SpinP_switch_e[iside]; k++ ){
              H00_e[iside][k][S00_ref(Anum+i,Bnum+j)].r += co*H_e[iside][k][GA_AN][LB_AN][i][j];
              H00_e[iside][k][S00_ref(Anum+i,Bnum+j)].i += si*H_e[iside][k][GA_AN][LB_AN][i][j];
            } 
            */

	  }
	}
      }

      if (l1==direction) {

	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            /* S_alpha-alpha */
            S01_nc_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S01_nc_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            /* S_beta-beta */
            S01_nc_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
            S01_nc_e[iside][S11_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];

            a00=H_e[iside][0][GA_AN][LB_AN][i][j];
            a11=H_e[iside][1][GA_AN][LB_AN][i][j];
            a01r=H_e[iside][2][GA_AN][LB_AN][i][j];
            a01i=H_e[iside][3][GA_AN][LB_AN][i][j];
            
            b00=iHNL_e[iside][0][GA_AN][LB_AN][i][j];
            b11=iHNL_e[iside][1][GA_AN][LB_AN][i][j];
            b01=iHNL_e[iside][2][GA_AN][LB_AN][i][j];
            /* H_alpha-alpha */
            H01_nc_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*a00 - si*b00;
            H01_nc_e[iside][S00_ref(Anum+i,Bnum+j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            H01_nc_e[iside][S01_ref(Anum+i,Bnum+j)].r += co*a01r - si*(a01i+b01);
            H01_nc_e[iside][S01_ref(Anum+i,Bnum+j)].i += si*a01r + co*(a01i+b01);
            /* H_beta-beta */
            H01_nc_e[iside][S11_ref(Anum+i,Bnum+j)].r += co*a11 - si*b11;
            H01_nc_e[iside][S11_ref(Anum+i,Bnum+j)].i += co*b11 + si*a11;

            /*
	    S01_e[iside][S00_ref(Anum+i,Bnum+j)].r += co*OLP_e[iside][0][GA_AN][LB_AN][i][j];
	    S01_e[iside][S00_ref(Anum+i,Bnum+j)].i += si*OLP_e[iside][0][GA_AN][LB_AN][i][j];

	    for (k=0; k<=SpinP_switch_e[iside]; k++ ){
	      H01_e[iside][k][S00_ref(Anum+i,Bnum+j)].r += co*H_e[iside][k][GA_AN][LB_AN][i][j];
	      H01_e[iside][k][S00_ref(Anum+i,Bnum+j)].i += si*H_e[iside][k][GA_AN][LB_AN][i][j];
	    }
            */
	  }
	}
      }

    }
  }  /* GA_AN */
    
   for (i=1; i<=NUM0; i++) {
     for (j=1; j<=NUM0; j++) {

       H00_nc_e[iside][(j-1)*NUM+i+NUM0-1].r =  H00_nc_e[iside][(i+NUM0-1)*NUM+j-1].r;
       H00_nc_e[iside][(j-1)*NUM+i+NUM0-1].i = -H00_nc_e[iside][(i+NUM0-1)*NUM+j-1].i;

       H01_nc_e[iside][(j-1)*NUM+i+NUM0-1].r =  H01_nc_e[iside][(i+NUM0-1)*NUM+j-1].r;
       H01_nc_e[iside][(j-1)*NUM+i+NUM0-1].i = -H01_nc_e[iside][(i+NUM0-1)*NUM+j-1].i;

      }
    }

  /* free arrays */

  free(MP);
} 
