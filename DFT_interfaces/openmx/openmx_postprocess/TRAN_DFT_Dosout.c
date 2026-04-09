/**********************************************************************
  TRAN_DFT_Dosout.c:

  TRAN_DFT_Dosout.c is a subroutine to calculate density of states 
  of a central region with left and right infinite leads based on
  a non-equilibrium Green's function method. 

  Log of TRAN_DFT_Dosout.c:

     24/July/2008  Released by T.Ozaki

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#define MEASURE_TIME  0

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"

 
void dtime(double *);

int Get_OneD_HS_Col(int set_flag, double ****RH, double *H1, int *MP, 
                    int *order_GA, int *My_NZeros, int *is1, int *is2);

void Make_Comm_Worlds(
   MPI_Comm MPI_Curret_Comm_WD,   
   int myid0,
   int numprocs0,
   int Num_Comm_World, 
   int *myworld1, 
   MPI_Comm *MPI_CommWD,     /* size: Num_Comm_World */
   int *NPROCS1_ID,          /* size: numprocs0 */
   int *Comm_World1,         /* size: numprocs0 */
   int *NPROCS1_WD,          /* size: Num_Comm_World */
   int *Comm_World_StartID   /* size: Num_Comm_World */
   );




static void TRAN_DFT_Kdependent_NC(
			  /* input */
			  MPI_Comm comm1,
                          int parallel_mode,
                          int numprocs,
                          int myid,
			  int level_stdout,
			  int iter,
			  int SpinP_switch,
                          double k2,
                          double k3,
                          int k_op,
                          int *order_GA,
                          double **H1,
                          double **H2,
                          double *S1,
			  double *****nh,  /* H */
			  double *****ImNL, /* not used, s-o coupling */
			  double ****CntOLP, 
			  int atomnum,
			  int Matomnum,
			  int *WhatSpecies,
			  int *Spe_Total_CNO,
			  int *FNAN,
			  int **natn, 
			  int **ncn,
			  int *M2G, 
			  int *G2ID, 
			  int **atv_ijk,
			  int *List_YOUSO,
			  /* output */
			  float ***Dos,    /* output, DOS */
			  double *****EDM,  /* not used */
			  double Eele0[2], double Eele1[2]) /* not used */


{
  int i,j,k,q,AN,iside; 
  int *MP,*MP_e[2];
  int iw,iw_method;
  dcomplex w, w_weight;
  dcomplex *GCR,*GCA;
  dcomplex *GRL,*GRR,*SigmaL, *SigmaR; 
  dcomplex *GCL_R,*GCR_R,*GCL_A,*GCR_A;
  dcomplex *v1;
  double dum,sum,tmpr,tmpi;
  double co,si,kRn;
  double TStime,TEtime;
  int MA_AN, GA_AN, wanA, tnoA, Anum;
  int LB_AN, GB_AN, wanB, tnoB, Bnum; 
  int l1,l2,l3,Rn,direction;
  int LB_AN_e,GB_AN_e,GA_AN_e,Rn_e;

  double sum1,sum2;

  int ID;
  int **iwIdx, Miwmax, Miw,iw0; 
  double *r_energy,de;
  
  /* parallel setup */

  Miwmax = (tran_dos_energydiv)/numprocs+1;

  iwIdx=(int**)malloc(sizeof(int*)*numprocs);
  for (i=0;i<numprocs;i++) {
    iwIdx[i]=(int*)malloc(sizeof(int)*Miwmax);
  }

  TRAN_Distribute_Node_Idx(0, tran_dos_energydiv-1, numprocs, Miwmax,
                           iwIdx); /* output */

  /* set up energies where DOS is calculated */
  r_energy = (double*)malloc(sizeof(double)*tran_dos_energydiv);

  de = (tran_dos_energyrange[1]-tran_dos_energyrange[0])/(double)tran_dos_energydiv;
  for (i=0; i<tran_dos_energydiv; i++) {
    r_energy[i] = tran_dos_energyrange[0] + de*(double)i + ChemP_e[0];
  }

  /* setup MP */
  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  TRAN_Set_MP(1, atomnum, WhatSpecies, Spe_Total_CNO, &i, MP);
  NUM_c  = i*2; 

  MP_e[0] = (int*)malloc(sizeof(int)*(atomnum_e[0]+1));
  TRAN_Set_MP( 1,  atomnum_e[0], WhatSpecies_e[0], Spe_Total_CNO_e[0], &i, MP_e[0]);
  NUM_e[0]  = i*2;

  MP_e[1] = (int*)malloc(sizeof(int)*(atomnum_e[1]+1));
  TRAN_Set_MP( 1,  atomnum_e[1], WhatSpecies_e[1], Spe_Total_CNO_e[1], &i, MP_e[1]);
  NUM_e[1]  = i*2;
  
  /* initialize */
  TRAN_Set_Value_double(SCC_nc,NUM_c*NUM_c,    0.0,0.0);
  TRAN_Set_Value_double(SCL_nc,NUM_c*NUM_e[0], 0.0,0.0);
  TRAN_Set_Value_double(SCR_nc,NUM_c*NUM_e[1], 0.0,0.0);
  
  TRAN_Set_Value_double(HCC_nc,NUM_c*NUM_c,    0.0,0.0);
  TRAN_Set_Value_double(HCL_nc,NUM_c*NUM_e[0], 0.0,0.0);
  TRAN_Set_Value_double(HCR_nc,NUM_c*NUM_e[1], 0.0,0.0);
  

  /* set Hamiltonian and overlap matrices of left and right leads */

  TRAN_Set_SurfOverlap_NC(comm1,"left", k2, k3);
  TRAN_Set_SurfOverlap_NC(comm1,"right",k2, k3);

  /* set CC, CL and CR */

  TRAN_Set_CentOverlap_NC(comm1,
                          3,
                          SpinP_switch, 
                          k2,
                          k3,
                          order_GA,
                          H1,
                          H2,
                          S1,
                          nh, /* input */
                          CntOLP, /* input */
                          atomnum,
			  Matomnum,
			  M2G,
			  G2ID,
                          WhatSpecies,
                          Spe_Total_CNO,
                          FNAN,
                          natn,
                          ncn,
                          atv_ijk);

  GCR    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GCA    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GRL    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0]* NUM_e[0]);
  GRR    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1]* NUM_e[1]);
  SigmaL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  v1     = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GCL_R  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  GCR_R  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  GCL_A  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  GCR_A  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  
  /* parallel global iw 0: tran_dos_energydiv-1 */
  /* parallel local  Miw 0:Miwmax-1             */
  /* parllel variable iw=iwIdx[myid][Miw]       */

  for (Miw=0; Miw<Miwmax; Miw++) {

    iw = iwIdx[myid][Miw];

      if (iw>=0) {

        /**************************************
                    w = w.r + i w.i 
        **************************************/
        
        w.r = r_energy[iw];
        w.i = tran_dos_energyrange[2];
        
        iside = 0;

        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_nc_e[iside], H01_nc_e[iside],
				   S00_nc_e[iside], S01_nc_e[iside], tran_surfgreen_iteration_max,
                                   tran_surfgreen_eps, GRL);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL_nc, SCL_nc, SigmaL);

        iside = 1;

        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_nc_e[iside],H01_nc_e[iside],
				   S00_nc_e[iside], S01_nc_e[iside], tran_surfgreen_iteration_max,
                                   tran_surfgreen_eps, GRR);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR_nc, SCR_nc, SigmaR);

        TRAN_Calc_CentGreen(w, NUM_c, SigmaL,SigmaR, HCC_nc, SCC_nc, GCR);

        /* GCL_R and GCR_R */
 
        iside = 0;
        TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRL, GCR, HCL_nc, SCL_nc, GCL_R);

        iside = 1;
        TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRR, GCR, HCR_nc, SCR_nc, GCR_R);

        /**************************************
                    w = w.r - i w.i 
        **************************************/

          w.r = r_energy[iw];
          w.i =-tran_dos_energyrange[2];

	  iside=0;

	  TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_nc_e[iside],H01_nc_e[iside],
				     S00_nc_e[iside], S01_nc_e[iside], tran_surfgreen_iteration_max,
				     tran_surfgreen_eps, GRL);

	  TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL_nc, SCL_nc, SigmaL);

	  iside=1;

	  TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_nc_e[iside],H01_nc_e[iside],
				     S00_nc_e[iside], S01_nc_e[iside], tran_surfgreen_iteration_max,
				     tran_surfgreen_eps, GRR);

	  TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR_nc, SCR_nc, SigmaR);

	  TRAN_Calc_CentGreen(w, NUM_c, SigmaL,SigmaR, HCC_nc, SCC_nc, GCA);

          /* GCL_R and GCR_R */
 
          iside = 0;
          TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRL, GCA, HCL_nc, SCL_nc, GCL_A);

          iside = 1;
          TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRR, GCA, HCR_nc, SCR_nc, GCR_A);

        /***********************************************
          calculate density of states from the center G
        ***********************************************/

        q = 0;

	for (AN=1; AN<=atomnum; AN++) {

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

	    kRn = -(k2*(double)l2 + k3*(double)l3);
	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

	    /* note that SCC includes information on the phase factor of translational cells */

	      for (i=0; i<tnoA; i++) {

  	        sum1 = 0.0;
                sum2 = 0.0;
		for (j=0; j<tnoB; j++) {
                  /*
                  tmpr =-0.5*(GCR[ v_idx( Anum+i, Bnum+j ) ].i - GCA[ v_idx( Anum+i, Bnum+j ) ].i);
                  tmpi = 0.5*(GCR[ v_idx( Anum+i, Bnum+j ) ].r - GCA[ v_idx( Anum+i, Bnum+j ) ].r);
                  sum += (double)(l1==0)*S1[q]*(tmpr*co - tmpi*si);
                  */
                             /* up-up spin contribution */
		  tmpr =-0.5*(GCR[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].i - GCA[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].i); 
		  tmpi = 0.5*(GCR[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].r - GCA[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].r); 
		  sum1 += (double)(l1==0)*S1[q]*(tmpr*co - tmpi*si);
                             /* down-down spin contribution */
                  tmpr =-0.5*(GCR[ (Bnum+j-1+NUM_c/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].i - GCA[ (Bnum+j-1+NUM_c/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].i);
                  tmpi = 0.5*(GCR[ (Bnum+j-1+NUM_c/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].r - GCA[ (Bnum+j-1+NUM_c/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].r);
                  sum2 += (double)(l1==0)*S1[q]*(tmpr*co - tmpi*si);

		  q++;
		}

 	        Dos[iw][0][Anum+i-1] += (float)k_op*sum1/PI;
                Dos[iw][1][Anum+i-1] += (float)k_op*sum2/PI;

	      }
 
	  } /* LB_AN */
	} /* AN */ 

        /*******************************************************
         calculate density of states contributed from the 
         off-diagonal G between the Central and Left regions
        *******************************************************/

        iside = 0;
        direction = -1;

        for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          if (TRAN_region[GA_AN]%10==2){

	    GA_AN_e =  TRAN_Original_Id[GA_AN];

	    for (i=0; i<tnoA; i++){

	      sum1 = 0.0;
              sum2 = 0.0;

	      for (LB_AN_e=0; LB_AN_e<=FNAN_e[iside][GA_AN_e]; LB_AN_e++){

		GB_AN_e = natn_e[iside][GA_AN_e][LB_AN_e];
		Rn_e = ncn_e[iside][GA_AN_e][LB_AN_e];
		wanB = WhatSpecies_e[iside][GB_AN_e];
		tnoB = Spe_Total_CNO_e[iside][wanB];
		Bnum = MP_e[iside][GB_AN_e];

		l1 = atv_ijk_e[iside][Rn_e][1];
		l2 = atv_ijk_e[iside][Rn_e][2];
		l3 = atv_ijk_e[iside][Rn_e][3];

  	        kRn = -(k2*(double)l2 + k3*(double)l3);
		si = sin(2.0*PI*kRn);
		co = cos(2.0*PI*kRn);

		if (l1==direction) {
		  for (j=0; j<tnoB; j++){

		      tmpr =-0.5*(GCL_R[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].i - GCL_A[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].i); 
		      tmpi = 0.5*(GCL_R[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].r - GCL_A[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].r); 
		      sum1 += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(tmpr*co - tmpi*si); 

                      tmpr =-0.5*(GCL_R[ (Bnum+j-1+NUM_e[0]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].i - GCL_A[ (Bnum+j-1+NUM_e[0]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].i);
                      tmpi = 0.5*(GCL_R[ (Bnum+j-1+NUM_e[0]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].r - GCL_A[ (Bnum+j-1+NUM_e[0]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].r);
                      sum2 += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(tmpr*co - tmpi*si);

		  } /* j<tnoB */
		}
	      } /* LB_AN */

	      Dos[iw][0][Anum+i-1] += (float)k_op*sum1/PI;
              Dos[iw][1][Anum+i-1] += (float)k_op*sum2/PI;

	    } /* i */
	  }
	}

        /*******************************************************
         calculate density of states contributed from the 
         off-diagonal G between the Central and Right regions
        *******************************************************/

        iside = 1;
        direction = 1;

        for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          if (TRAN_region[GA_AN]%10==3){

	    GA_AN_e =  TRAN_Original_Id[GA_AN];

	    for (i=0; i<tnoA; i++){

	      sum1 = 0.0;
              sum2 = 0.0;

	      for (LB_AN_e=0; LB_AN_e<=FNAN_e[iside][GA_AN_e]; LB_AN_e++){

		GB_AN_e = natn_e[iside][GA_AN_e][LB_AN_e];
		Rn_e = ncn_e[iside][GA_AN_e][LB_AN_e];
		wanB = WhatSpecies_e[iside][GB_AN_e];
		tnoB = Spe_Total_CNO_e[iside][wanB];
		Bnum = MP_e[iside][GB_AN_e];

		l1 = atv_ijk_e[iside][Rn_e][1];
		l2 = atv_ijk_e[iside][Rn_e][2];
		l3 = atv_ijk_e[iside][Rn_e][3];

  	        kRn = -(k2*(double)l2 + k3*(double)l3);
		si = sin(2.0*PI*kRn);
		co = cos(2.0*PI*kRn);

		if (l1==direction) {
		  for (j=0; j<tnoB; j++){
                      /*
                      tmpr =-0.5*(GCR_R[ v_idx( Anum+i, Bnum+j) ].i - GCR_A[ v_idx( Anum+i, Bnum+j) ].i);
                      tmpi = 0.5*(GCR_R[ v_idx( Anum+i, Bnum+j) ].r - GCR_A[ v_idx( Anum+i, Bnum+j) ].r);
                      */

		      tmpr =-0.5*(GCR_R[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].i - GCR_A[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].i); 
		      tmpi = 0.5*(GCR_R[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].r - GCR_A[ (Bnum+j-1)*NUM_c+(Anum+i)-1 ].r); 
		      sum1 += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(tmpr*co - tmpi*si);

                      tmpr =-0.5*(GCR_R[ (Bnum+j-1+NUM_e[1]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].i - GCR_A[ (Bnum+j-1+NUM_e[1]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].i);
                      tmpi = 0.5*(GCR_R[ (Bnum+j-1+NUM_e[1]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].r - GCR_A[ (Bnum+j-1+NUM_e[1]/2)*NUM_c+(Anum+i+NUM_c/2)-1 ].r);
                      sum2 += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(tmpr*co - tmpi*si); 

		  } /* j<tnoB */
		}
	      } /* LB_AN */

	      Dos[iw][0][Anum+i-1] += (float)k_op*sum1/PI;
              Dos[iw][1][Anum+i-1] += (float)k_op*sum2/PI;

	    } /* i */
	  }
	}

      } /* iw>=0 */
  }     /* Miw   */

  /* free arrays */

  free(GCR_A);
  free(GCL_A);
  free(GCR_R);
  free(GCL_R);
  free(v1);
  free(SigmaR);
  free(SigmaL);
  free(GRR);
  free(GRL);
  free(GCR);
  free(GCA);
  free(MP_e[1]);
  free(MP_e[0]);
  free(MP);

  for (i=0;i<numprocs;i++) {
    free(iwIdx[i]);
  }
  free(iwIdx);

  free(r_energy);
}










static double TRAN_DFT_Dosout_NC(
		/* input */
                MPI_Comm comm1,
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,   /* H */
		double *****ImNL, /* not used, s-o coupling */
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int *G2ID, 
		int **atv_ijk,
		int *List_YOUSO,
                int **Spe_Num_CBasis,
                int SpeciesNum,
                char *filename,
                char *filepath,
		/* output */
		double *****CDM,  /* not used */
		double *****EDM,  /* not used */
		double Eele0[2], double Eele1[2]) /* not used */
{
  int numprocs0,numprocs1,myid0,myid1,ID;
  int myworld1,i2,i3,k_op,ik;
  int T_knum,kloop0,kloop,Anum;
  int i,j,spin,MA_AN,GA_AN,wanA,tnoA;
  int LB_AN,GB_AN,wanB,tnoB,k;
  int E_knum,S_knum,num_kloop0,parallel_mode;
  int **op_flag,*T_op_flag,*T_k_ID;
  double *T_KGrids2,*T_KGrids3;
  double k2,k3,tmp;
  double TStime,TEtime;
  float ***Dos;

  int *MP;
  int *order_GA;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  int size_H1;
  double **H1,*S1;
  double **H2;

  int Num_Comm_World1;
  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;

  MPI_Comm_size(comm1,&numprocs0);
  MPI_Comm_rank(comm1,&myid0);

  dtime(&TStime);

  if (myid0==Host_ID){
    printf("<TRAN_DFT_Dosout>\n"); fflush(stdout);
  }

  /* allocate Dos */

  TRAN_Set_MP(0, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, &i);

  Dos = (float***)malloc(sizeof(float**)*tran_dos_energydiv);
  for (ik=0; ik<tran_dos_energydiv; ik++) {
    Dos[ik] = (float**)malloc(sizeof(float*)*(1+1) );
    for (spin=0; spin<=1; spin++) {
      Dos[ik][spin] = (float*)malloc(sizeof(float)*NUM_c);
      for (i=0; i<NUM_c; i++)  Dos[ik][spin][i] = 0.0;
    }
  }

  /***********************************
        set up operation flag
  ************************************/

  op_flag = (int**)malloc(sizeof(int*)*TRAN_dos_Kspace_grid2); 
  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    op_flag[i2] = (int*)malloc(sizeof(int)*TRAN_dos_Kspace_grid3); 
    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){
      op_flag[i2][i3] = 1;
    }
  }

  /***********************************
       one-dimentionalize for MPI
  ************************************/

  T_knum = 0;
  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){
      if (0<op_flag[i2][i3]) T_knum++;  
    }
  }         

  T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
  T_op_flag = (int*)malloc(sizeof(int)*T_knum);
  T_k_ID = (int*)malloc(sizeof(int)*T_knum);

  T_knum = 0;

  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){

    k2 = -0.5 + (2.0*(double)i2+1.0)/(2.0*(double)TRAN_dos_Kspace_grid2) + Shift_K_Point;

    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){

      k3 = -0.5 + (2.0*(double)i3+1.0)/(2.0*(double)TRAN_dos_Kspace_grid3) - Shift_K_Point;

      if (0<op_flag[i2][i3]){  

        T_KGrids2[T_knum] = k2;
        T_KGrids3[T_knum] = k3;
        T_op_flag[T_knum] = op_flag[i2][i3];

        T_knum++;        
      }
    }
  }

  /***************************************************
   allocate calculations of k-points into processors 
  ***************************************************/

  if (numprocs0<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs0; ID++){

      tmp = (double)T_knum/(double)numprocs0;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs0-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the first level world */
        T_k_ID[k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs0; 
    S_knum = (int)((double)myid0*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid0+1)*(tmp+1.0e-12)) - 1;
    if (myid0==(numprocs0-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World1 = T_knum;

    NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0);
    Comm_World1 = (int*)malloc(sizeof(int)*numprocs0);
    NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

    Make_Comm_Worlds(comm1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

    S_knum = myworld1;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
      /* ID in the first level world */
      T_k_ID[k] = Comm_World_StartID1[k];
    }

  }
  
  /*************************************************************
   one-dimensitonalize H and S and store them in a compact form  
  *************************************************************/

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  order_GA = (int*)malloc(sizeof(int)*(atomnum+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  size_H1 = Get_OneD_HS_Col(0, nh[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  H1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    H1[spin] = (double*)malloc(sizeof(double)*size_H1);
  }

  H2 = (double**)malloc(sizeof(double*)*(2+1));
  for (k=0; k<(2+1); k++){
    H2[k] = (double*)malloc(sizeof(double)*size_H1);
  }

  S1 = (double*)malloc(sizeof(double)*size_H1);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    size_H1 = Get_OneD_HS_Col(1, nh[spin], H1[spin], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  for (k=0; k<(2+1); k++){
    size_H1 = Get_OneD_HS_Col(1, ImNL[k], H2[k], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /***********************************************************
   start "kloop0"
  ***********************************************************/

  for (kloop0=0; kloop0<num_kloop0; kloop0++){

    kloop = S_knum + kloop0;

    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];
    k_op = T_op_flag[kloop];

    if (parallel_mode){

        TRAN_DFT_Kdependent_NC(MPI_CommWD1[myworld1],
  			    parallel_mode, numprocs1, myid1,
                            level_stdout, iter, SpinP_switch, k2, k3, k_op, order_GA, 
                            H1, H2, S1,
                            nh, ImNL, CntOLP,
	  	  	    atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN,
			    natn, ncn, M2G, G2ID, atv_ijk, List_YOUSO, Dos, EDM, Eele0, Eele1);
    }

    else{

        TRAN_DFT_Kdependent_NC(comm1, 
 			    parallel_mode, 1, 0,
                            level_stdout, iter, SpinP_switch, k2, k3, k_op, order_GA, 
                            H1, H2, S1,
                            nh, ImNL, CntOLP,
	  	  	    atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN,
			    natn, ncn, M2G, G2ID, atv_ijk, List_YOUSO, Dos, EDM, Eele0, Eele1);
    }
  }

  /*******************************************************
          summ up Dos by MPI and send it Host_ID 
  *******************************************************/

  MPI_Barrier(comm1);

  /* in TRAN_DFT_Kdependent(), NUM_c is doubled. So, recalculate it */ 
  TRAN_Set_MP(0, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, &i);

  {
    float *Dos0;

    Dos0 = (float*)malloc(sizeof(float)*NUM_c);

    tmp = 1.0/(double)(TRAN_dos_Kspace_grid2*TRAN_dos_Kspace_grid3);

    for (ik=0; ik<tran_dos_energydiv; ik++) {
      for (spin=0; spin<=1; spin++) {

	MPI_Reduce(&Dos[ik][spin][0], &Dos0[0], NUM_c, MPI_FLOAT, MPI_SUM, Host_ID, comm1);
        MPI_Barrier(comm1);
        for (i=0; i<NUM_c; i++)  Dos[ik][spin][i] = Dos0[i]*tmp;
      }
    }  

    free(Dos0);
  }

  /**********************************************************
                     save Dos to a file
  **********************************************************/

  if (myid0==Host_ID){

    FILE *fp_eig, *fp_ev;
    char file_eig[YOUSO10],file_ev[YOUSO10];
    int l,MaxL;
    double *r_energy,de;
    int i_vec[10];

    r_energy = (double*)malloc(sizeof(double)*tran_dos_energydiv);

    de = (tran_dos_energyrange[1]-tran_dos_energyrange[0])/(double)tran_dos_energydiv;
    for (i=0; i<tran_dos_energydiv; i++) {
      r_energy[i] = tran_dos_energyrange[0] + de*(double)i + ChemP_e[0];
    }

    /* write *.Dos.val */

    sprintf(file_eig,"%s%s.Dos.val",filepath,filename);

    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("can not open a file %s\n",file_eig);
    }
    else {

      printf("  write eigenvalues\n");

      fprintf(fp_eig,"mode        6\n");
      fprintf(fp_eig,"NonCol      0\n");
      fprintf(fp_eig,"N           %d\n",NUM_c);
      fprintf(fp_eig,"Nspin       %d\n",1); /* SpinP_switch has been changed to 1 */
      fprintf(fp_eig,"Erange      %lf %lf\n",tran_dos_energyrange[0],tran_dos_energyrange[1]);
      fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
      fprintf(fp_eig,"atomnum     %d\n",atomnum);
      fprintf(fp_eig,"<WhatSpecies\n");
      for (i=1; i<=atomnum; i++) {
        fprintf(fp_eig,"%d ",WhatSpecies[i]);
      }
      fprintf(fp_eig,"\nWhatSpecies>\n");
      fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
      fprintf(fp_eig,"<Spe_Total_CNO\n");
      for (i=0;i<SpeciesNum;i++) {
        fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
      }
      fprintf(fp_eig,"\nSpe_Total_CNO>\n");
      MaxL=4;
      fprintf(fp_eig,"MaxL           %d\n",4);
      fprintf(fp_eig,"<Spe_Num_CBasis\n");
      for (i=0;i<SpeciesNum;i++) {
        for (l=0;l<=MaxL;l++) {
	  fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
        }
        fprintf(fp_eig,"\n");
      }
      fprintf(fp_eig,"Spe_Num_CBasis>\n");
      fprintf(fp_eig,"ChemP       %lf\n",ChemP_e[0]);

      fprintf(fp_eig,"irange      %d %d\n",0,tran_dos_energydiv-1);
      fprintf(fp_eig,"<Eigenvalues\n");
      for (spin=0; spin<=1; spin++) {
        fprintf(fp_eig,"%d %d %d ",0,0,0);
        for (ik=0; ik<tran_dos_energydiv; ik++) {
          fprintf(fp_eig,"%lf ",r_energy[ik]);
	}
        fprintf(fp_eig,"\n");
      }  
      fprintf(fp_eig,"Eigenvalues>\n");

      fclose(fp_eig);
    }

    /* write *.Dos.vec */

    printf("  write eigenvectors\n");

    sprintf(file_ev,"%s%s.Dos.vec",filepath,filename);

    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
      printf("can not open a file %s\n",file_ev);
    }
    else {

      for (spin=0; spin<=1; spin++) {
        for (ik=0; ik<tran_dos_energydiv; ik++) {

          i_vec[0]=i_vec[1]=i_vec[2]=0;
          if (myid0==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

          for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
            Anum = MP[GA_AN];
            fwrite(&Dos[ik][spin][Anum-1],sizeof(float),tnoA,fp_ev);
	  }
	}
      }

      fclose(fp_ev);
    }

    /* free arrays */

    free(r_energy);
  }

  /* free arrays */

  for (ik=0; ik<tran_dos_energydiv; ik++) {
    for (spin=0; spin<=1; spin++) {
      free(Dos[ik][spin]);
    }
    free(Dos[ik]);
  }
  free(Dos);

  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    free(op_flag[i2]);
  }
  free(op_flag);

  free(T_KGrids2);
  free(T_KGrids3);
  free(T_op_flag);
  free(T_k_ID);

  if (T_knum<=numprocs0){

    if (Num_Comm_World1<=numprocs0){
      MPI_Comm_free(&MPI_CommWD1[myworld1]);
    }

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);
  }

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(H1[spin]);
  }
  free(H1);

  for (spin=0; spin<(2+1); spin++){
    free(H2[spin]);
  }
  free(H2);
 
  free(S1);

  /* for elapsed time */
  dtime(&TEtime);

  /*
  if (myid==Host_ID){
    printf("TRAN_DFT_Dosout time=%12.7f\n",TEtime - TStime);
  }
  */

  return TEtime - TStime;
}




static void TRAN_DFT_Kdependent_Col(
			  /* input */
			  MPI_Comm comm1,
                          int parallel_mode,
                          int numprocs,
                          int myid,
			  int level_stdout,
			  int iter,
			  int SpinP_switch,
                          double k2,
                          double k3,
                          int k_op,
                          int *order_GA,
                          double **H1,
                          double *S1,
			  double *****nh,  /* H */
			  double *****ImNL, /* not used, s-o coupling */
			  double ****CntOLP, 
			  int atomnum,
			  int Matomnum,
			  int *WhatSpecies,
			  int *Spe_Total_CNO,
			  int *FNAN,
			  int **natn, 
			  int **ncn,
			  int *M2G, 
			  int *G2ID, 
			  int **atv_ijk,
			  int *List_YOUSO,
			  /* output */
			  float ***Dos,    /* output, DOS */
			  double *****EDM,  /* not used */
			  double Eele0[2], double Eele1[2]) /* not used */

#define GC_ref(i,j) GC[ NUM_c*((j)-1) + (i)-1 ] 
#define v_idx(i,j)   ( ((j)-1)*NUM_c + (i)-1 ) 

{
  int i,j,k,q,AN,iside; 
  int *MP,*MP_e[2];
  int iw,iw_method;
  dcomplex w, w_weight;
  dcomplex *GCR,*GCA;
  dcomplex *GRL,*GRR,*SigmaL, *SigmaR; 
  dcomplex *GCL_R,*GCR_R,*GCL_A,*GCR_A;
  dcomplex *v1;
  double dum,sum,tmpr,tmpi;
  double co,si,kRn;
  double TStime,TEtime;
  int MA_AN, GA_AN, wanA, tnoA, Anum;
  int LB_AN, GB_AN, wanB, tnoB, Bnum; 
  int l1,l2,l3,Rn,direction;
  int LB_AN_e,GB_AN_e,GA_AN_e,Rn_e;

  int ID;
  int **iwIdx, Miwmax, Miw,iw0; 
  double *r_energy,de;
  
  /* parallel setup */

  Miwmax = (tran_dos_energydiv)/numprocs+1;

  iwIdx=(int**)malloc(sizeof(int*)*numprocs);
  for (i=0;i<numprocs;i++) {
    iwIdx[i]=(int*)malloc(sizeof(int)*Miwmax);
  }

  TRAN_Distribute_Node_Idx(0, tran_dos_energydiv-1, numprocs, Miwmax,
                           iwIdx); /* output */

  /* set up energies where DOS is calculated */
  r_energy = (double*)malloc(sizeof(double)*tran_dos_energydiv);

  de = (tran_dos_energyrange[1]-tran_dos_energyrange[0])/(double)tran_dos_energydiv;
  for (i=0; i<tran_dos_energydiv; i++) {
    r_energy[i] = tran_dos_energyrange[0] + de*(double)i + ChemP_e[0];
  }

  /* setup MP */
  TRAN_Set_MP(0, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, &i);
  MP = (int*)malloc(sizeof(int)*(NUM_c+1));
  TRAN_Set_MP(1, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, MP);

  MP_e[0] = (int*)malloc(sizeof(int)*(NUM_e[0]+1));
  TRAN_Set_MP( 1,  atomnum_e[0], WhatSpecies_e[0], Spe_Total_CNO_e[0], &i, MP_e[0]);

  MP_e[1] = (int*)malloc(sizeof(int)*(NUM_e[1]+1));
  TRAN_Set_MP( 1,  atomnum_e[1], WhatSpecies_e[1], Spe_Total_CNO_e[1], &i, MP_e[1]);
  
  /* initialize */
  TRAN_Set_Value_double(SCC,NUM_c*NUM_c,    0.0,0.0);
  TRAN_Set_Value_double(SCL,NUM_c*NUM_e[0], 0.0,0.0);
  TRAN_Set_Value_double(SCR,NUM_c*NUM_e[1], 0.0,0.0);
  for (k=0; k<=SpinP_switch; k++) {
    TRAN_Set_Value_double(HCC[k],NUM_c*NUM_c,    0.0,0.0);
    TRAN_Set_Value_double(HCL[k],NUM_c*NUM_e[0], 0.0,0.0);
    TRAN_Set_Value_double(HCR[k],NUM_c*NUM_e[1], 0.0,0.0);
  }

  /* set Hamiltonian and overlap matrices of left and right leads */

  TRAN_Set_SurfOverlap(comm1,"left", k2, k3);
  TRAN_Set_SurfOverlap(comm1,"right",k2, k3);

  /* set CC, CL and CR */

  TRAN_Set_CentOverlap(   comm1,
                          3,
                          SpinP_switch, 
                          k2,
                          k3,
                          order_GA,
                          H1,
                          S1,
                          nh, /* input */
                          CntOLP, /* input */
                          atomnum,
			  Matomnum,
			  M2G,
			  G2ID,
                          WhatSpecies,
                          Spe_Total_CNO,
                          FNAN,
                          natn,
                          ncn,
                          atv_ijk);

  GCR    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GCA    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GRL    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0]* NUM_e[0]);
  GRR    = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1]* NUM_e[1]);
  SigmaL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  v1     = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  GCL_R  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  GCR_R  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  GCL_A  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[0]);
  GCR_A  = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c*NUM_e[1]);
  
  /* parallel global iw 0: tran_dos_energydiv-1 */
  /* parallel local  Miw 0:Miwmax-1             */
  /* parllel variable iw=iwIdx[myid][Miw]       */

  for (Miw=0; Miw<Miwmax; Miw++) {

    iw = iwIdx[myid][Miw];

    for (k=0; k<=SpinP_switch; k++) {

      if (iw>=0) {

        /**************************************
                    w = w.r + i w.i 
        **************************************/
        
        w.r = r_energy[iw];
        w.i = tran_dos_energyrange[2];
        
        iside = 0;

        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside][k], H01_e[iside][k],
				   S00_e[iside], S01_e[iside], tran_surfgreen_iteration_max,
                                   tran_surfgreen_eps, GRL);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL[k], SCL, SigmaL);

        iside = 1;

        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside][k],H01_e[iside][k],
				   S00_e[iside], S01_e[iside], tran_surfgreen_iteration_max,
                                   tran_surfgreen_eps, GRR);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR[k], SCR, SigmaR);

        TRAN_Calc_CentGreen(w, NUM_c, SigmaL,SigmaR, HCC[k], SCC, GCR);

        /* GCL_R and GCR_R */
 
        iside = 0;
        TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRL, GCR, HCL[k], SCL, GCL_R);

        iside = 1;
        TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRR, GCR, HCR[k], SCR, GCR_R);

        /**************************************
                    w = w.r - i w.i 
        **************************************/

        if (TRAN_dos_Kspace_grid2!=1 || TRAN_dos_Kspace_grid3!=1){

          w.r = r_energy[iw];
          w.i =-tran_dos_energyrange[2];

	  iside=0;

	  TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside][k],H01_e[iside][k],
				     S00_e[iside], S01_e[iside], tran_surfgreen_iteration_max,
				     tran_surfgreen_eps, GRL);

	  TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL[k], SCL, SigmaL);

	  iside=1;

	  TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside][k],H01_e[iside][k],
				     S00_e[iside], S01_e[iside], tran_surfgreen_iteration_max,
				     tran_surfgreen_eps, GRR);

	  TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR[k], SCR, SigmaR);

	  TRAN_Calc_CentGreen(w, NUM_c, SigmaL,SigmaR, HCC[k], SCC, GCA);

          /* GCL_R and GCR_R */
 
          iside = 0;
          TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRL, GCA, HCL[k], SCL, GCL_A);

          iside = 1;
          TRAN_Calc_Hopping_G(w, NUM_e[iside], NUM_c, GRR, GCA, HCR[k], SCR, GCR_A);
	}

        /***********************************************
          calculate density of states from the center G
        ***********************************************/

        q = 0;

	for (AN=1; AN<=atomnum; AN++) {

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

	    kRn = -(k2*(double)l2 + k3*(double)l3);
	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

	    /* note that SCC includes information on the phase factor of translational cells */

	    if (TRAN_dos_Kspace_grid2==1 && TRAN_dos_Kspace_grid3==1){

	      for (i=0; i<tnoA; i++) {

  	        sum = 0.0;
		for (j=0; j<tnoB; j++) {

		  sum += (double)(l1==0)*S1[q]*(-GCR[ v_idx( Anum+i, Bnum+j) ].i);

		  q++;
		}

 	        Dos[iw][k][Anum+i-1] += (float)k_op*sum/PI;

	      }
	    }
               
	    else {

	      for (i=0; i<tnoA; i++) {

  	        sum = 0.0;
		for (j=0; j<tnoB; j++) {

		  tmpr =-0.5*(GCR[ v_idx( Anum+i, Bnum+j ) ].i - GCA[ v_idx( Anum+i, Bnum+j ) ].i); 
		  tmpi = 0.5*(GCR[ v_idx( Anum+i, Bnum+j ) ].r - GCA[ v_idx( Anum+i, Bnum+j ) ].r); 
		  sum += (double)(l1==0)*S1[q]*(tmpr*co - tmpi*si);
		  q++;
		}

 	        Dos[iw][k][Anum+i-1] += (float)k_op*sum/PI;

	      }

	    } 
	  }


	} /* AN */ 

        /*******************************************************
         calculate density of states contributed from the 
         off-diagonal G between the Central and Left regions
        *******************************************************/

        iside = 0;
        direction = -1;

        for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          if (TRAN_region[GA_AN]%10==2){

	    GA_AN_e =  TRAN_Original_Id[GA_AN];

	    for (i=0; i<tnoA; i++){

	      sum = 0.0;

	      for (LB_AN_e=0; LB_AN_e<=FNAN_e[iside][GA_AN_e]; LB_AN_e++){

		GB_AN_e = natn_e[iside][GA_AN_e][LB_AN_e];
		Rn_e = ncn_e[iside][GA_AN_e][LB_AN_e];
		wanB = WhatSpecies_e[iside][GB_AN_e];
		tnoB = Spe_Total_CNO_e[iside][wanB];
		Bnum = MP_e[iside][GB_AN_e];

		l1 = atv_ijk_e[iside][Rn_e][1];
		l2 = atv_ijk_e[iside][Rn_e][2];
		l3 = atv_ijk_e[iside][Rn_e][3];

  	        kRn = -(k2*(double)l2 + k3*(double)l3);
		si = sin(2.0*PI*kRn);
		co = cos(2.0*PI*kRn);

		if (l1==direction) {
		  for (j=0; j<tnoB; j++){

		    if (TRAN_dos_Kspace_grid2==1 && TRAN_dos_Kspace_grid3==1){
		      sum += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(-GCL_R[ v_idx( Anum+i, Bnum+j) ].i);
		    }
		    else {
		      tmpr =-0.5*(GCL_R[ v_idx( Anum+i, Bnum+j) ].i - GCL_A[ v_idx( Anum+i, Bnum+j) ].i); 
		      tmpi = 0.5*(GCL_R[ v_idx( Anum+i, Bnum+j) ].r - GCL_A[ v_idx( Anum+i, Bnum+j) ].r); 
		      sum += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(tmpr*co - tmpi*si); 
		    }
		  }
		}
	      } /* LB_AN */

	      Dos[iw][k][Anum+i-1] += (float)k_op*sum/PI;

	    } /* i */
	  }
	}

        /*******************************************************
         calculate density of states contributed from the 
         off-diagonal G between the Central and Right regions
        *******************************************************/

        iside = 1;
        direction = 1;

        for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

          wanA = WhatSpecies[GA_AN];
          tnoA = Spe_Total_CNO[wanA];
          Anum = MP[GA_AN];

          if (TRAN_region[GA_AN]%10==3){

	    GA_AN_e =  TRAN_Original_Id[GA_AN];

	    for (i=0; i<tnoA; i++){

	      sum = 0.0;

	      for (LB_AN_e=0; LB_AN_e<=FNAN_e[iside][GA_AN_e]; LB_AN_e++){

		GB_AN_e = natn_e[iside][GA_AN_e][LB_AN_e];
		Rn_e = ncn_e[iside][GA_AN_e][LB_AN_e];
		wanB = WhatSpecies_e[iside][GB_AN_e];
		tnoB = Spe_Total_CNO_e[iside][wanB];
		Bnum = MP_e[iside][GB_AN_e];

		l1 = atv_ijk_e[iside][Rn_e][1];
		l2 = atv_ijk_e[iside][Rn_e][2];
		l3 = atv_ijk_e[iside][Rn_e][3];

  	        kRn = -(k2*(double)l2 + k3*(double)l3);
		si = sin(2.0*PI*kRn);
		co = cos(2.0*PI*kRn);

		if (l1==direction) {
		  for (j=0; j<tnoB; j++){

		    if (TRAN_dos_Kspace_grid2==1 && TRAN_dos_Kspace_grid3==1){
		      sum += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(-GCR_R[ v_idx( Anum+i, Bnum+j) ].i);
		    }
		    else {
		      tmpr =-0.5*(GCR_R[ v_idx( Anum+i, Bnum+j) ].i - GCR_A[ v_idx( Anum+i, Bnum+j) ].i); 
		      tmpi = 0.5*(GCR_R[ v_idx( Anum+i, Bnum+j) ].r - GCR_A[ v_idx( Anum+i, Bnum+j) ].r); 
		      sum += OLP_e[iside][0][GA_AN_e][LB_AN_e][i][j]*(tmpr*co - tmpi*si); 
		    }
		  }
		}
	      } /* LB_AN */

	      Dos[iw][k][Anum+i-1] += (float)k_op*sum/PI;

	    } /* i */
	  }
	}

      } /* iw>=0 */
    }   /* for k */
  }     /* Miw   */

  /* free arrays */

  free(GCR_A);
  free(GCL_A);
  free(GCR_R);
  free(GCL_R);
  free(v1);
  free(SigmaR);
  free(SigmaL);
  free(GRR);
  free(GRL);
  free(GCR);
  free(GCA);
  free(MP_e[1]);
  free(MP_e[0]);
  free(MP);

  for (i=0;i<numprocs;i++) {
    free(iwIdx[i]);
  }
  free(iwIdx);

  free(r_energy);
}










static double TRAN_DFT_Dosout_Col(
		/* input */
                MPI_Comm comm1,
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,   /* H */
		double *****ImNL, /* not used, s-o coupling */
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int *G2ID, 
		int **atv_ijk,
		int *List_YOUSO,
                int **Spe_Num_CBasis,
                int SpeciesNum,
                char *filename,
                char *filepath,
		/* output */
		double *****CDM,  /* not used */
		double *****EDM,  /* not used */
		double Eele0[2], double Eele1[2]) /* not used */
{
  int numprocs0,numprocs1,myid0,myid1,ID;
  int myworld1,i2,i3,k_op,ik;
  int T_knum,kloop0,kloop,Anum;
  int i,j,spin,MA_AN,GA_AN,wanA,tnoA;
  int LB_AN,GB_AN,wanB,tnoB,k;
  int E_knum,S_knum,num_kloop0,parallel_mode;
  int **op_flag,*T_op_flag,*T_k_ID;
  double *T_KGrids2,*T_KGrids3;
  double k2,k3,tmp;
  double TStime,TEtime;
  float ***Dos;

  int *MP;
  int *order_GA;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  int size_H1;
  double **H1,*S1;

  int Num_Comm_World1;
  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;

  MPI_Comm_size(comm1,&numprocs0);
  MPI_Comm_rank(comm1,&myid0);

  dtime(&TStime);

  if (myid0==Host_ID){
    printf("<TRAN_DFT_Dosout>\n"); fflush(stdout);
  }

  /* allocate Dos */

  TRAN_Set_MP(0, atomnum, WhatSpecies, Spe_Total_CNO, &NUM_c, &i);

  Dos = (float***)malloc(sizeof(float**)*tran_dos_energydiv);
  for (ik=0; ik<tran_dos_energydiv; ik++) {
    Dos[ik] = (float**)malloc(sizeof(float*)*(SpinP_switch+1) );
    for (spin=0; spin<=SpinP_switch; spin++) {
      Dos[ik][spin] = (float*)malloc(sizeof(float)*NUM_c);
      for (i=0; i<NUM_c; i++)  Dos[ik][spin][i] = 0.0;
    }
  }

  /***********************************
        set up operation flag
  ************************************/

  op_flag = (int**)malloc(sizeof(int*)*TRAN_dos_Kspace_grid2); 
  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    op_flag[i2] = (int*)malloc(sizeof(int)*TRAN_dos_Kspace_grid3); 
    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){
      op_flag[i2][i3] = -999;
    }
  }

  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){

      if (op_flag[i2][i3]<0){ 

	if ( (TRAN_dos_Kspace_grid2-1-i2)==i2 && (TRAN_dos_Kspace_grid3-1-i3)==i3 ){
	  op_flag[i2][i3] = 1;
	}
	else{
	  op_flag[i2][i3] = 2;
	  op_flag[TRAN_dos_Kspace_grid2-1-i2][TRAN_dos_Kspace_grid3-1-i3] = 0;
	}
      }

    }
  }

  /***********************************
       one-dimentionalize for MPI
  ************************************/

  T_knum = 0;
  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){
      if (0<op_flag[i2][i3]) T_knum++;  
    }
  }         

  T_KGrids2 = (double*)malloc(sizeof(double)*T_knum);
  T_KGrids3 = (double*)malloc(sizeof(double)*T_knum);
  T_op_flag = (int*)malloc(sizeof(int)*T_knum);
  T_k_ID = (int*)malloc(sizeof(int)*T_knum);

  T_knum = 0;

  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){

    k2 = -0.5 + (2.0*(double)i2+1.0)/(2.0*(double)TRAN_dos_Kspace_grid2) + Shift_K_Point;

    for (i3=0; i3<TRAN_dos_Kspace_grid3; i3++){

      k3 = -0.5 + (2.0*(double)i3+1.0)/(2.0*(double)TRAN_dos_Kspace_grid3) - Shift_K_Point;

      if (0<op_flag[i2][i3]){  

        T_KGrids2[T_knum] = k2;
        T_KGrids3[T_knum] = k3;
        T_op_flag[T_knum] = op_flag[i2][i3];

        T_knum++;        
      }
    }
  }

  /***************************************************
   allocate calculations of k-points into processors 
  ***************************************************/

  if (numprocs0<T_knum){

    /* set parallel_mode */
    parallel_mode = 0;

    /* allocation of kloop to ID */     

    for (ID=0; ID<numprocs0; ID++){

      tmp = (double)T_knum/(double)numprocs0;
      S_knum = (int)((double)ID*(tmp+1.0e-12)); 
      E_knum = (int)((double)(ID+1)*(tmp+1.0e-12)) - 1;
      if (ID==(numprocs0-1)) E_knum = T_knum - 1;
      if (E_knum<0)          E_knum = 0;

      for (k=S_knum; k<=E_knum; k++){
        /* ID in the first level world */
        T_k_ID[k] = ID;
      }
    }

    /* find own informations */

    tmp = (double)T_knum/(double)numprocs0; 
    S_knum = (int)((double)myid0*(tmp+1.0e-12)); 
    E_knum = (int)((double)(myid0+1)*(tmp+1.0e-12)) - 1;
    if (myid0==(numprocs0-1)) E_knum = T_knum - 1;
    if (E_knum<0)             E_knum = 0;

    num_kloop0 = E_knum - S_knum + 1;

  }

  else {

    /* set parallel_mode */
    parallel_mode = 1;
    num_kloop0 = 1;

    Num_Comm_World1 = T_knum;

    NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs0);
    Comm_World1 = (int*)malloc(sizeof(int)*numprocs0);
    NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

    Make_Comm_Worlds(comm1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);

    S_knum = myworld1;

    /* allocate k-points into processors */
    
    for (k=0; k<T_knum; k++){
      /* ID in the first level world */
      T_k_ID[k] = Comm_World_StartID1[k];
    }

  }
  
  /*************************************************************
   one-dimensitonalize H and S and store them in a compact form  
  *************************************************************/

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
  order_GA = (int*)malloc(sizeof(int)*(atomnum+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs0);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs0);

  size_H1 = Get_OneD_HS_Col(0, nh[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  H1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    H1[spin] = (double*)malloc(sizeof(double)*size_H1);
  }

  S1 = (double*)malloc(sizeof(double)*size_H1);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    size_H1 = Get_OneD_HS_Col(1, nh[spin], H1[spin], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /***********************************************************
   start "kloop0"
  ***********************************************************/

  for (kloop0=0; kloop0<num_kloop0; kloop0++){

    kloop = S_knum + kloop0;

    k2 = T_KGrids2[kloop];
    k3 = T_KGrids3[kloop];
    k_op = T_op_flag[kloop];

    if (parallel_mode){

        TRAN_DFT_Kdependent_Col(MPI_CommWD1[myworld1],
  			    parallel_mode, numprocs1, myid1,
                            level_stdout, iter, SpinP_switch, k2, k3, k_op, order_GA, 
                            H1, S1,
                            nh, ImNL, CntOLP,
	  	  	    atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN,
			    natn, ncn, M2G, G2ID, atv_ijk, List_YOUSO, Dos, EDM, Eele0, Eele1);
    }

    else{

        TRAN_DFT_Kdependent_Col(comm1, 
 			    parallel_mode, 1, 0,
                            level_stdout, iter, SpinP_switch, k2, k3, k_op, order_GA, 
                            H1, S1,
                            nh, ImNL, CntOLP,
	  	  	    atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN,
			    natn, ncn, M2G, G2ID, atv_ijk, List_YOUSO, Dos, EDM, Eele0, Eele1);
    }
  }

  /*******************************************************
          summ up Dos by MPI and send it Host_ID 
  *******************************************************/

  MPI_Barrier(comm1);

  {
    float *Dos0;

    Dos0 = (float*)malloc(sizeof(float)*NUM_c);

    tmp = 1.0/(double)(TRAN_dos_Kspace_grid2*TRAN_dos_Kspace_grid3);

    for (ik=0; ik<tran_dos_energydiv; ik++) {
      for (spin=0; spin<=SpinP_switch; spin++) {

	MPI_Reduce(&Dos[ik][spin][0], &Dos0[0], NUM_c, MPI_FLOAT, MPI_SUM, Host_ID, comm1);
        MPI_Barrier(comm1);
        for (i=0; i<NUM_c; i++)  Dos[ik][spin][i] = Dos0[i]*tmp;
      }
    }  

    free(Dos0);
  }

  /**********************************************************
                     save Dos to a file
  **********************************************************/

  if (myid0==Host_ID){

    FILE *fp_eig, *fp_ev;
    char file_eig[YOUSO10],file_ev[YOUSO10];
    int l,MaxL;
    double *r_energy,de;
    int i_vec[10];

    r_energy = (double*)malloc(sizeof(double)*tran_dos_energydiv);

    de = (tran_dos_energyrange[1]-tran_dos_energyrange[0])/(double)tran_dos_energydiv;
    for (i=0; i<tran_dos_energydiv; i++) {
      r_energy[i] = tran_dos_energyrange[0] + de*(double)i + ChemP_e[0];
    }

    /* write *.Dos.val */

    sprintf(file_eig,"%s%s.Dos.val",filepath,filename);

    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {
      printf("can not open a file %s\n",file_eig);
    }
    else {

      printf("  write eigenvalues\n");

      fprintf(fp_eig,"mode        6\n");
      fprintf(fp_eig,"NonCol      0\n");
      fprintf(fp_eig,"N           %d\n",NUM_c);
      fprintf(fp_eig,"Nspin       %d\n",SpinP_switch);
      fprintf(fp_eig,"Erange      %lf %lf\n",tran_dos_energyrange[0],tran_dos_energyrange[1]);
      fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
      fprintf(fp_eig,"atomnum     %d\n",atomnum);
      fprintf(fp_eig,"<WhatSpecies\n");
      for (i=1; i<=atomnum; i++) {
        fprintf(fp_eig,"%d ",WhatSpecies[i]);
      }
      fprintf(fp_eig,"\nWhatSpecies>\n");
      fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
      fprintf(fp_eig,"<Spe_Total_CNO\n");
      for (i=0;i<SpeciesNum;i++) {
        fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
      }
      fprintf(fp_eig,"\nSpe_Total_CNO>\n");
      MaxL=4;
      fprintf(fp_eig,"MaxL           %d\n",4);
      fprintf(fp_eig,"<Spe_Num_CBasis\n");
      for (i=0;i<SpeciesNum;i++) {
        for (l=0;l<=MaxL;l++) {
	  fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
        }
        fprintf(fp_eig,"\n");
      }
      fprintf(fp_eig,"Spe_Num_CBasis>\n");
      fprintf(fp_eig,"ChemP       %lf\n",ChemP_e[0]);

      fprintf(fp_eig,"irange      %d %d\n",0,tran_dos_energydiv-1);
      fprintf(fp_eig,"<Eigenvalues\n");
      for (spin=0; spin<=SpinP_switch; spin++) {
        fprintf(fp_eig,"%d %d %d ",0,0,0);
        for (ik=0; ik<tran_dos_energydiv; ik++) {
          fprintf(fp_eig,"%lf ",r_energy[ik]);
	}
        fprintf(fp_eig,"\n");
      }  
      fprintf(fp_eig,"Eigenvalues>\n");

      fclose(fp_eig);
    }

    /* write *.Dos.vec */

    printf("  write eigenvectors\n");

    sprintf(file_ev,"%s%s.Dos.vec",filepath,filename);

    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {
      printf("can not open a file %s\n",file_ev);
    }
    else {

      for (spin=0; spin<=SpinP_switch; spin++) {
        for (ik=0; ik<tran_dos_energydiv; ik++) {

          i_vec[0]=i_vec[1]=i_vec[2]=0;
          if (myid0==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

          for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];
            Anum = MP[GA_AN];
            fwrite(&Dos[ik][spin][Anum-1],sizeof(float),tnoA,fp_ev);
	  }
	}
      }

      fclose(fp_ev);
    }

    /* free arrays */

    free(r_energy);
  }

  /* free arrays */

  for (ik=0; ik<tran_dos_energydiv; ik++) {
    for (spin=0; spin<=SpinP_switch; spin++) {
      free(Dos[ik][spin]);
    }
    free(Dos[ik]);
  }
  free(Dos);

  for (i2=0; i2<TRAN_dos_Kspace_grid2; i2++){
    free(op_flag[i2]);
  }
  free(op_flag);

  free(T_KGrids2);
  free(T_KGrids3);
  free(T_op_flag);
  free(T_k_ID);

  if (T_knum<=numprocs0){

    if (Num_Comm_World1<=numprocs0){
      MPI_Comm_free(&MPI_CommWD1[myworld1]);
    }

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);
  }

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(H1[spin]);
  }
  free(H1);
 
  free(S1);

  /* for elapsed time */
  dtime(&TEtime);

  /*
  if (myid==Host_ID){
    printf("TRAN_DFT_Dosout time=%12.7f\n",TEtime - TStime);
  }
  */

  return TEtime - TStime;
}


/* revised by Y. Xiao for Noncollinear NEGF calculations */
double TRAN_DFT_Dosout(
                /* input */
                MPI_Comm comm1,
                int level_stdout,
                int iter,
                int SpinP_switch,
                double *****nh,   /* H */
                double *****ImNL, /* not used, s-o coupling */
                double ****CntOLP,
                int atomnum,
                int Matomnum,
                int *WhatSpecies,
                int *Spe_Total_CNO,
                int *FNAN,
                int **natn,
                int **ncn,
                int *M2G,
                int *G2ID,
                int **atv_ijk,
                int *List_YOUSO,
                int **Spe_Num_CBasis,
                int SpeciesNum,
                char *filename,
                char *filepath,
                /* output */
                double *****CDM,  /* not used */
                double *****EDM,  /* not used */
                double Eele0[2], double Eele1[2]) /* not used */
{
  double TStime,TEtime,time5;

  dtime(&TStime);

     if ( SpinP_switch < 2 ) {
                TRAN_DFT_Dosout_Col( comm1, level_stdout, iter, SpinP_switch, nh, ImNL, CntOLP,
                atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn, M2G, G2ID, atv_ijk,
                List_YOUSO, Spe_Num_CBasis, SpeciesNum, filename, filepath, CDM, EDM, Eele0, Eele1); 
     } else {

                TRAN_DFT_Dosout_NC( comm1, level_stdout, iter, SpinP_switch, nh, ImNL, CntOLP,
                atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn, M2G, G2ID, atv_ijk,
                List_YOUSO, Spe_Num_CBasis, SpeciesNum, filename, filepath, CDM, EDM, Eele0, Eele1);

     }

  dtime(&TEtime);
  return TEtime - TStime;
}

