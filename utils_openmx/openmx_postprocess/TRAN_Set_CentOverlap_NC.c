/**********************************************************************
  TRAN_Set_CentOverlap_NC.c:

  TRAN_Set_CentOverlap_NC.c is a subroutine to set Hamiltonian and 
  overlap matrices of the central region.

  Log of TRAN_Set_CentOverlap_NC.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

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
#define SCC00_ref(i,j) ( ((j)-1)*nc+(i)-1 )
#define SCC01_ref(i,j) ( ((j+nc0)-1)*nc+(i)-1 )
#define SCC10_ref(i,j) ( ((j)-1)*nc+(i+nc0)-1 )
#define SCC11_ref(i,j) ( ((j+nc0)-1)*nc+(i+nc0)-1 )

#define SCL00_ref(i,j) ( ((j)-1)*nc+(i)-1 )
#define SCL01_ref(i,j) ( ((j+na0)-1)*nc+(i)-1 )
#define SCL10_ref(i,j) ( ((j)-1)*nc+(i+nc0)-1 )
#define SCL11_ref(i,j) ( ((j+na0)-1)*nc+(i+nc0)-1 )

#define SCR00_ref(i,j) ( ((j)-1)*nc+(i)-1 )
#define SCR01_ref(i,j) ( ((j+nb0)-1)*nc+(i)-1 )
#define SCR10_ref(i,j) ( ((j)-1)*nc+(i+nc0)-1 )
#define SCR11_ref(i,j) ( ((j+nb0)-1)*nc+(i+nc0)-1 )

#define S00l00_ref(i,j) ( ((j)-1)*na+(i)-1 )
#define S00l01_ref(i,j) ( ((j+na0)-1)*na+(i)-1 )
#define S00l10_ref(i,j) ( ((j)-1)*na+(i+na0)-1 )
#define S00l11_ref(i,j) ( ((j+na0)-1)*na+(i+na0)-1 )

#define S00r00_ref(i,j) ( ((j)-1)*nb+(i)-1 )
#define S00r01_ref(i,j) ( ((j+nb0)-1)*nb+(i)-1 )
#define S00r10_ref(i,j) ( ((j)-1)*nb+(i+nb0)-1 )
#define S00r11_ref(i,j) ( ((j+nb0)-1)*nb+(i+nb0)-1 )

/*
#define SCC_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define SCL_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )
#define SCR_ref(i,j) ( ((j)-1)*NUM_c + (i)-1 )

#define S00l_ref(i,j) ( ((j)-1)*NUM_e[0]+(i)-1 )
#define S00r_ref(i,j) ( ((j)-1)*NUM_e[1]+(i)-1 )
*/

/*
 *  job&1==1   : Central region
 *  job&2==2   : CL, CR region 
 */


void TRAN_Set_CentOverlap_NC(
			  MPI_Comm comm1,
			  int job, 
			  int SpinP_switch,
                          double k2,
                          double k3, 
                          int *order_GA,
                          double **H1,
                          double **H2,
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
  int nc0,nc,na0,na,nb0,nb;
  int k; 
 
  MPI_Comm_rank(comm1,&myid);
  MPI_Comm_size(comm1,&numprocs);
  
  if (MEASURE_TIME){
    dtime(&time_a0);
  }
  
  
  { 
    int i;
    /* setup MP */
    MP = (int*)malloc(sizeof(int)*(atomnum+1));
    TRAN_Set_MP( 1,  atomnum, WhatSpecies, Spe_Total_CNO, &i, MP);
    nc0=i;
    nc=2*i;
  
    MP_e[0] = (int*)malloc(sizeof(int)*(atomnum_e[0]+1));
    TRAN_Set_MP( 1,  atomnum_e[0], WhatSpecies_e[0], Spe_Total_CNO_e[0], &i, MP_e[0]);
    na0=i;
    na=2*i;

    MP_e[1] = (int*)malloc(sizeof(int)*(atomnum_e[1]+1));
    TRAN_Set_MP( 1,  atomnum_e[1], WhatSpecies_e[1], Spe_Total_CNO_e[1], &i, MP_e[1]);
    nb0=i;
    nb=2*i;
  }
  
  if ((job&1)==1) {

    int MA_AN,GA_AN, wanA, tnoA, Anum;
    int LB_AN, GB_AN, wanB, tnoB, l1,l2,l3, Bnum;
    int i,j,k,q,AN;
    int Rn;
    double kRn,si,co;
    double tmp,s0,h0;
    double a00,a11,a01r,a01i,b00,b11,b01;

/*    for (i=0; i<NUM_c*NUM_c; i++) {
      SCC_nc[i].r = 0.0;
      SCC_nc[i].i = 0.0;
      HCC_nc[i].r = 0.0;
      HCC_nc[i].i = 0.0;
    } */

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

            /* S_alpha-alpha */
            SCC_nc[SCC00_ref(Anum+i,Bnum+j)].r += co*S1[q];
            SCC_nc[SCC00_ref(Anum+i,Bnum+j)].i += si*S1[q];
            /* S_beta-beta */
            SCC_nc[SCC11_ref(Anum+i,Bnum+j)].r += co*S1[q];
            SCC_nc[SCC11_ref(Anum+i,Bnum+j)].i += si*S1[q];

            a00=H1[0][q];
            a11=H1[1][q];
            a01r=H1[2][q];
            a01i=H1[3][q];

            b00=H2[0][q];
            b11=H2[1][q];
            b01=H2[2][q];

            /* H_alpha-alpha */
            HCC_nc[SCC00_ref(Anum+i,Bnum+j)].r += co*a00 - si*b00;
            HCC_nc[SCC00_ref(Anum+i,Bnum+j)].i += co*b00 + si*a00;
            /* H_alpha-beta */
            HCC_nc[SCC01_ref(Anum+i,Bnum+j)].r += co*a01r - si*(a01i+b01);
            HCC_nc[SCC01_ref(Anum+i,Bnum+j)].i += si*a01r + co*(a01i+b01);
            /* H_beta-beta */
            HCC_nc[SCC11_ref(Anum+i,Bnum+j)].r += co*a11 - si*b11;
            HCC_nc[SCC11_ref(Anum+i,Bnum+j)].i += co*b11 + si*a11;

            /*
	      SCC[SCC_ref(Anum+i,Bnum+j)].r += co*S1[q];
	      SCC[SCC_ref(Anum+i,Bnum+j)].i += si*S1[q];

		HCC[k][SCC_ref(Anum+i,Bnum+j)].r += co*H1[k][q];
		HCC[k][SCC_ref(Anum+i,Bnum+j)].i += si*H1[k][q];   
            */
	    }

            q++;

	  }
	}

      } /* LB_AN */
    }   /* MA_AN */

    for (i=1; i<=nc0; i++) {
     for (j=1; j<=nc0; j++) {

       HCC_nc[(j-1)*nc+i+nc0-1].r =  HCC_nc[(i+nc0-1)*nc+j-1].r;
       HCC_nc[(j-1)*nc+i+nc0-1].i = -HCC_nc[(i+nc0-1)*nc+j-1].i;

      }
    }


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

            /* S_alpha-alpha */
            SCC_nc[SCC00_ref(Anum+i,Bnum+j)].r = S00_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_beta-beta */
            SCC_nc[SCC11_ref(Anum+i,Bnum+j)].r = S00_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].r;

            /* H_alpha-alpha */
            HCC_nc[SCC00_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_alpha-beta */
            HCC_nc[SCC01_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00l01_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-alpha */
            HCC_nc[SCC10_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00l10_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-beta */
            HCC_nc[SCC11_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].r;


            /* S_alpha-alpha */
            SCC_nc[SCC00_ref(Anum+i,Bnum+j)].i = S00_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].i;
            /* S_beta-beta */
            SCC_nc[SCC11_ref(Anum+i,Bnum+j)].i = S00_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].i;

            /* H_alpha-alpha */
            HCC_nc[SCC00_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_alpha-beta */
            HCC_nc[SCC01_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00l01_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-alpha */
            HCC_nc[SCC10_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00l10_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-beta */
            HCC_nc[SCC11_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].i;
          /*
	        SCC[SCC_ref(Anum+i,Bnum+j)] = S00_e[iside][S00l_ref(Anum_e+i,Bnum_e+j)];

                for (k=0; k<=SpinP_switch; k++) {
                  HCC[k][SCC_ref(Anum+i,Bnum+j)] = H00_e[iside][k][S00l_ref(Anum_e+i,Bnum_e+j)];
		}  */
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

            /* S_alpha-alpha */
            SCC_nc[SCC00_ref(Anum+i,Bnum+j)].r = S00_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_beta-beta */
            SCC_nc[SCC11_ref(Anum+i,Bnum+j)].r = S00_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].r;

            /* H_alpha-alpha */
            HCC_nc[SCC00_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_alpha-beta */
            HCC_nc[SCC01_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00r01_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-alpha */
            HCC_nc[SCC10_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00r10_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-beta */
            HCC_nc[SCC11_ref(Anum+i,Bnum+j)].r = H00_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].r;

            /* S_alpha-alpha */
            SCC_nc[SCC00_ref(Anum+i,Bnum+j)].i = S00_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].i;
            /* S_beta-beta */
            SCC_nc[SCC11_ref(Anum+i,Bnum+j)].i = S00_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].i;

            /* H_alpha-alpha */
            HCC_nc[SCC00_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_alpha-beta */
            HCC_nc[SCC01_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00r01_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-alpha */
            HCC_nc[SCC10_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00r10_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-beta */
            HCC_nc[SCC11_ref(Anum+i,Bnum+j)].i = H00_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].i;
        /*
		SCC[SCC_ref(Anum+i,Bnum+j)] = S00_e[iside][S00r_ref(Anum_e+i,Bnum_e+j)];

		for (k=0; k<=SpinP_switch; k++) {
		  HCC[k][SCC_ref(Anum+i,Bnum+j)] = H00_e[iside][k][S00r_ref(Anum_e+i,Bnum_e+j)];
		} */
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

/*      for (i=0; i<NUM_c*NUM_e[iside]; i++) {
      SCL_nc[i].r = 0.0;
      SCL_nc[i].i = 0.0;
      HCL_nc[i].r = 0.0;
      HCL_nc[i].i = 0.0;
    } */

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

            /* S_alpha-alpha */
            SCL_nc[SCL00_ref(Anum+i,Bnum_e+j)].r = S01_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_beta-beta */
            SCL_nc[SCL11_ref(Anum+i,Bnum_e+j)].r = S01_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].r;

            /* H_alpha-alpha */
            HCL_nc[SCL00_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_alpha-beta */
            HCL_nc[SCL01_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00l01_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-alpha */
            HCL_nc[SCL10_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00l10_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-beta */
            HCL_nc[SCL11_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].r;

            /* S_alpha-alpha */
            SCL_nc[SCL00_ref(Anum+i,Bnum_e+j)].i = S01_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].i;
            /* S_beta-beta */
            SCL_nc[SCL11_ref(Anum+i,Bnum_e+j)].i = S01_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].i;

            /* H_alpha-alpha */
            HCL_nc[SCL00_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00l00_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_alpha-beta */
            HCL_nc[SCL01_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00l01_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-alpha */
            HCL_nc[SCL10_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00l10_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-beta */
            HCL_nc[SCL11_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00l11_ref(Anum_e+i,Bnum_e+j)].i;
       /*
              SCL[SCL_ref(Anum+i,Bnum_e+j)] = S01_e[iside][ S00l_ref(Anum_e+i, Bnum_e+j)];

              for (k=0; k<=SpinP_switch; k++) {
                HCL[k][SCL_ref(Anum+i,Bnum_e+j)] = H01_e[iside][k][ S00l_ref(Anum_e+i, Bnum_e+j)];
              } */
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

/*      for (i=0; i<NUM_c*NUM_e[iside]; i++) {
      SCR_nc[i].r = 0.0;
      SCR_nc[i].i = 0.0;
      HCR_nc[i].r = 0.0;
      HCR_nc[i].i = 0.0;
    } */

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

            /* S_alpha-alpha */
            SCR_nc[SCR00_ref(Anum+i,Bnum_e+j)].r = S01_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].r;
            /* S_beta-beta */
            SCR_nc[SCR11_ref(Anum+i,Bnum_e+j)].r = S01_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].r;

            /* H_alpha-alpha */
            HCR_nc[SCR00_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].r;
            /* H_alpha-beta */
            HCR_nc[SCR01_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00r01_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-alpha */
            HCR_nc[SCR10_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00r10_ref(Anum_e+i,Bnum_e+j)].r; 
            /* H_beta-beta */
            HCR_nc[SCR11_ref(Anum+i,Bnum_e+j)].r = H01_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].r;

            /* S_alpha-alpha */
            SCR_nc[SCR00_ref(Anum+i,Bnum_e+j)].i = S01_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].i;
            /* S_beta-beta */
            SCR_nc[SCR11_ref(Anum+i,Bnum_e+j)].i = S01_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].i;

            /* H_alpha-alpha */
            HCR_nc[SCR00_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00r00_ref(Anum_e+i,Bnum_e+j)].i;
            /* H_alpha-beta */
            HCR_nc[SCR01_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00r01_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-alpha */
            HCR_nc[SCR10_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00r10_ref(Anum_e+i,Bnum_e+j)].i; 
            /* H_beta-beta */
            HCR_nc[SCR11_ref(Anum+i,Bnum_e+j)].i = H01_nc_e[iside][S00r11_ref(Anum_e+i,Bnum_e+j)].i;
         /*
              SCR[SCR_ref(Anum+i,Bnum_e+j)] = S01_e[iside][ S00r_ref(Anum_e+i, Bnum_e+j)];

              for (k=0; k<=SpinP_switch; k++) {
                HCR[k][SCR_ref(Anum+i,Bnum_e+j)] = H01_e[iside][k][ S00r_ref(Anum_e+i, Bnum_e+j)];
              } */
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


