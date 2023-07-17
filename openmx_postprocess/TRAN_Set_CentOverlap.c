/**********************************************************************
  TRAN_Set_CentOverlap.c:

  TRAN_Set_CentOverlap.c is a subroutine to set Hamiltonian and 
  overlap matrices of the central region.

  Log of TRAN_Set_CentOverlap.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"

#define MEASURE_TIME 0



void dtime(double *);

/************************************************* 
 input 
                          H, OLP 
 implicit input
                          double **S00_e, S01_e, H00_e, H01_e 

 output
                          double *SCC, 
                          double *SCL,
                          double *SCR, 

                          double **HCC, 
                          double **HCL,
                          double **HCR,

*************************************************/

#define SCC_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define SCL_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define SCR_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )

#define S00l_ref(i,j) ( ((j)-1)*NUM_e[0]+(i)-1 )
#define S00r_ref(i,j) ( ((j)-1)*NUM_e[1]+(i)-1 )



/*
 *  job&1==1   : Central region
 *  job&2==2   : CL, CR region 
 */


void TRAN_Set_CentOverlap(
			  MPI_Comm comm1,
			  int job, 
			  int SpinP_switch, 
                          double k2,
                          double k3, 
                          int *order_GA,
                          double **H1,
                          double *S1,
			  double *****H,  /* input */
			  double ****OLP, /* input */
			  int atomnum,
			  int Matomnum,
			  int *M2G,
			  int *G2ID, 
			  int *WhatSpecies,
			  int *Spe_Total_CNO,
			  int *FNAN,
			  int **natn,
			  int **ncn, 
			  int **atv_ijk
			  /*    int *WhichRegion */
			  )
{ 
  int *MP, *MP_e[2];
  int myid,numprocs;
  double time_a0, time_a1, time_a2; 
  
  MPI_Comm_rank(comm1,&myid);
  MPI_Comm_size(comm1,&numprocs);
  
  if (MEASURE_TIME){
    dtime(&time_a0);
  }
  
  
  { 
    int i;
    /* setup MP */
    MP = (int*)malloc(sizeof(int)*(NUM_c+1));
    TRAN_Set_MP( 1,  atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, MP);
  
    MP_e[0] = (int*)malloc(sizeof(int)*(NUM_e[0]+1));
    TRAN_Set_MP( 1,  atomnum_e[0], WhatSpecies_e[0], Spe_Total_CNO_e[0], &i, MP_e[0]);
  
    MP_e[1] = (int*)malloc(sizeof(int)*(NUM_e[1]+1));
    TRAN_Set_MP( 1,  atomnum_e[1], WhatSpecies_e[1], Spe_Total_CNO_e[1], &i, MP_e[1]);
  }
  
  if ((job&1)==1) {

    int MA_AN,GA_AN, wanA, tnoA, Anum;
    int LB_AN, GB_AN, wanB, tnoB, l1,l2,l3, Bnum;
    int i,j,k,q,AN;
    int Rn;
    double kRn,si,co;
    double tmp,s0,h0;

    for (i=0; i<NUM_c*NUM_c; i++) {
      SCC[i].r = 0.0;
      SCC[i].i = 0.0;
      for (k=0; k<=SpinP_switch; k++) {
	HCC[k][i].r = 0.0;
	HCC[k][i].i = 0.0;
      }
    }

    /* make Overlap ,  HCC, SCC               */
    /* parallel global GA_AN 1:atomnum        */

    q = 0;

    for (AN=1; AN<=atomnum; AN++){

      GA_AN = order_GA[AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){

	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];

        kRn = k2*(double)l2 + k3*(double)l3;
        si = sin(2.0*PI*kRn);
        co = cos(2.0*PI*kRn);
        
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

            /* l1 is the direction to the electrode */

            if (l1==0){

	      SCC[SCC_ref(Anum+i,Bnum+j)].r += co*S1[q];
	      SCC[SCC_ref(Anum+i,Bnum+j)].i += si*S1[q];

	      for (k=0; k<=SpinP_switch; k++) {
		HCC[k][SCC_ref(Anum+i,Bnum+j)].r += co*H1[k][q];
		HCC[k][SCC_ref(Anum+i,Bnum+j)].i += si*H1[k][q];
	      }
	    }

            q++;

	  }
	}

      } /* LB_AN */
    }   /* MA_AN */

  }     /* job&1 */


  if ( (job&2) == 2 ) {

    {
      int MA_AN, GA_AN, wanA, tnoA, Anum;
      int GA_AN_e, Anum_e; 
      int GB_AN, wanB, tnoB, Bnum;
      int GB_AN_e, Bnum_e; 
      int i,j,k;
      int iside;

      /* overwrite CL1 region */

      iside = 0;

      /* parallel global GA_AN 1:atomnum */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (GB_AN=1; GB_AN<=atomnum; GB_AN++){

	  if ( TRAN_region[GA_AN]==12 && TRAN_region[GB_AN]==12 ) {

	    GA_AN_e = TRAN_Original_Id[GA_AN];
  	    Anum_e = MP_e[iside][GA_AN_e];

	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    GB_AN_e = TRAN_Original_Id[GB_AN];
	    Bnum_e = MP_e[iside][GB_AN_e];

	    for (i=0; i<tnoA; i++){
	      for (j=0; j<tnoB; j++){

	        SCC[SCC_ref(Anum+i,Bnum+j)] = S00_e[iside][S00l_ref(Anum_e+i,Bnum_e+j)];

                for (k=0; k<=SpinP_switch; k++) {
                  HCC[k][SCC_ref(Anum+i,Bnum+j)] = H00_e[iside][k][S00l_ref(Anum_e+i,Bnum_e+j)];
		}
	      }
	    }

	  }
	}
      }
    } 

    {
      int MA_AN, GA_AN, wanA, tnoA, Anum;
      int GA_AN_e, Anum_e;
      int GB_AN, wanB, tnoB, Bnum;
      int GB_AN_e, Bnum_e;
      int i,j,k;
      int iside;

      /* overwrite CR1 region */

      iside = 1;

      /*parallel global GA_AN  1:atomnum */
      /*parallel local  MA_AN  1:Matomnum */
      /*parallel variable GA_AN = M2G[MA_AN] */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (GB_AN=1; GB_AN<=atomnum; GB_AN++){

	  if ( TRAN_region[GA_AN]==13 && TRAN_region[GB_AN]==13 )  {

	    GA_AN_e = TRAN_Original_Id[GA_AN];
	    Anum_e = MP_e[iside][GA_AN_e]; /* = Anum */

	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    GB_AN_e = TRAN_Original_Id[GB_AN];
	    Bnum_e = MP_e[iside][GB_AN_e]; 

	    for (i=0; i<tnoA; i++){
	      for (j=0; j<tnoB; j++){
		SCC[SCC_ref(Anum+i,Bnum+j)] = S00_e[iside][S00r_ref(Anum_e+i,Bnum_e+j)];

		for (k=0; k<=SpinP_switch; k++) {
		  HCC[k][SCC_ref(Anum+i,Bnum+j)] = H00_e[iside][k][S00r_ref(Anum_e+i,Bnum_e+j)];
		}
	      }
	    }

	  }

	}
      }
    }

    {
      int iside;
      int MA_AN, GA_AN, wanA, tnoA, Anum, GA_AN_e, Anum_e;
      int GB_AN_e, wanB_e, tnoB_e, Bnum_e;
      int i,j,k;

      /* make Overlap,  HCL, SCL from OLP_e, and H_e*/

      iside = 0;

      /*parallel global GA_AN  1:atomnum */
      /*parallel local  MA_AN  1:Matomnum */
      /*parallel variable GA_AN = M2G[MA_AN] */

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

	if (TRAN_region[GA_AN]%10!=2) continue;

	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];  /* GA_AN is in C */

	GA_AN_e =  TRAN_Original_Id[GA_AN];
	Anum_e = MP_e[iside][GA_AN_e]; 

	for (GB_AN_e=1; GB_AN_e<=atomnum_e[iside]; GB_AN_e++) {

	  wanB_e = WhatSpecies_e[iside][GB_AN_e];
	  tnoB_e = Spe_Total_CNO_e[iside][wanB_e];
          Bnum_e = MP_e[iside][GB_AN_e];

          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB_e; j++){

              SCL[SCL_ref(Anum+i,Bnum_e+j)] = S01_e[iside][ S00l_ref(Anum_e+i, Bnum_e+j)];

              for (k=0; k<=SpinP_switch; k++) {
                HCL[k][SCL_ref(Anum+i,Bnum_e+j)] = H01_e[iside][k][ S00l_ref(Anum_e+i, Bnum_e+j)];
              }
            }
          }
	}
      }
    }

    {
      int iside;
      int MA_AN, GA_AN, wanA, tnoA, Anum, GA_AN_e, Anum_e;
      int GB_AN_e, wanB_e, tnoB_e, Bnum_e;
      int i,j,k;

      /* make Overlap ,  HCR, SCR from OLP_e, and H_e*/

      iside = 1;

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

        if (TRAN_region[GA_AN]%10!=3) continue;

        wanA = WhatSpecies[GA_AN];
        tnoA = Spe_Total_CNO[wanA];
        Anum = MP[GA_AN];  /* GA_AN is in C */

        GA_AN_e =  TRAN_Original_Id[GA_AN];
        Anum_e = MP_e[iside][GA_AN_e];

        for (GB_AN_e=1; GB_AN_e<=atomnum_e[iside];GB_AN_e++) {
          wanB_e = WhatSpecies_e[iside][GB_AN_e];
          tnoB_e = Spe_Total_CNO_e[iside][wanB_e];
          Bnum_e = MP_e[iside][GB_AN_e];
          for (i=0; i<tnoA; i++){
            for (j=0; j<tnoB_e; j++){

              SCR[SCR_ref(Anum+i,Bnum_e+j)] = S01_e[iside][ S00r_ref(Anum_e+i, Bnum_e+j)];

              for (k=0; k<=SpinP_switch; k++) {
                HCR[k][SCR_ref(Anum+i,Bnum_e+j)] = H01_e[iside][k][ S00r_ref(Anum_e+i, Bnum_e+j)];
              }
            }
          }
        }
      }
    }

  } /* job&2 */

  if (MEASURE_TIME){
    dtime(&time_a1);
  }

  if (MEASURE_TIME){
    dtime(&time_a2);
    printf(" TRAN_Set_CentOverlap (%d)  calculation (%le)\n",myid,(time_a1-time_a0));
  }

#if 0

  TRAN_FPrint2_double("zSCC",NUM_c, NUM_c, SCC);
  TRAN_FPrint2_double("zSCL",NUM_c, NUM_e[0], SCL);
  TRAN_FPrint2_double("zSCR",NUM_c, NUM_e[1], SCR);

  TRAN_FPrint2_double("zHCC0",NUM_c, NUM_c, HCC[0]);
  TRAN_FPrint2_double("zHCL0",NUM_c, NUM_e[0], HCL[0]);
  TRAN_FPrint2_double("zHCR0",NUM_c, NUM_e[1], HCR[0]);

#endif

  /* post-process */
  free(MP);
  free(MP_e[1]);
  free(MP_e[0]);
}
