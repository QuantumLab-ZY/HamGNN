/**********************************************************************
  MTRAN_EigenChannel.c:

  TRAN_Calc_EChannel.c is a subroutine to calculate the Eigen Channel
   See PRB 76, 115117 (2007).
  It is made by modifing TRAN_Calc_OneTransmission.c.

   input: SigmaL, SigmaR, G_CC^R(w)
   assuming z= w+i delta 

   Gamma(z) = i (Sigma_R(z) - Sigma_R^+(z))
 
   A(z) = G_CC_R(z) Gamma(z) G_CC_R^+(z)

  Log of MTRAN_EigenChannel.c:
     
     xx/Xxx/2015 Released by M. Kawamura

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"

int TRAN_Calc_OrtSpace(
  int NUM_c, 
  dcomplex *SCC, 
  dcomplex *rtS, 
  dcomplex *rtSinv
  ); /* int TRAN_Calc_OrtSpace */

void TRAN_Calc_Linewidth(
  int NUM_c, 
  dcomplex *SigmaL_R, 
  dcomplex *SigmaR_R
  ); /* void TRAN_Calc_Linewidth */

/* G Gamma_L G^+ -> Sigma_L */

void TRAN_Calc_MatTrans(
  int NUM_c, 
  dcomplex *SigmaL_R, 
  dcomplex *GC_R, 
  char *trans1, 
  char *trans2
  ); /* void TRAN_Calc_MatTrans */

/* L\"owdin orthogonalization */

void TRAN_Calc_LowdinOrt(
  int NUM_c, 
  dcomplex *SigmaL_R, 
  dcomplex *SigmaR_R, 
  int NUM_cs, 
  dcomplex *rtS, 
  dcomplex *rtSinv, 
  dcomplex *ALbar, 
  dcomplex *GamRbar
  ); /* void TRAN_Calc_LowdinOrt */

/* Diagonalize ALbar -> its eigenvector */
/* Then scale it : Albar_{ml} -> ALbar * sqrt(eig_l)  */

void TRAN_Calc_Diagonalize(
  int NUM_cs, 
  dcomplex *ALbar, 
  double *eval, 
  int lscale
  ); /* void TRAN_Calc_Diagonalize */

/* Transform eigenchannel into non-orthogonal basis space */

void TRAN_Calc_ChannelLCAO(
  int NUM_c, 
  int NUM_cs, 
  dcomplex *rtSinv, 
  dcomplex *ALbar,
  dcomplex *GamRbar, 
  double *eval, 
  dcomplex *GC_R, 
  int TRAN_Channel_Num, 
  dcomplex **EChannel, 
  double *eigentrans
  ); /* void TRAN_Calc_ChannelLCAO */

/* Output EigenChannel in the basis space */

void TRAN_Output_ChannelLCAO(
  int myid0,
  int kloop, 
  int iw, 
  int ispin, 
  int NUM_c,
  double *TRAN_Channel_kpoint, 
  double TRAN_Channel_energy, 
  double *eval, 
  dcomplex *GC_R, 
  double **eigentrans_sum
  ); /* void TRAN_Output_ChannelLCAO */

void MTRAN_EigenChannel(
  MPI_Comm comm1,
  int numprocs,
  int myid,
  int myid0,
  int SpinP_switch,
  double ChemP_e[2],
  int NUM_c,
  int NUM_e[2],
  dcomplex **H00_e[2],
  dcomplex *S00_e[2],
  dcomplex **H01_e[2],
  dcomplex *S01_e[2],
  dcomplex **HCC,
  dcomplex **HCL,
  dcomplex **HCR,
  dcomplex *SCC,
  dcomplex *SCL,
  dcomplex *SCR,
  double tran_surfgreen_iteration_max,
  double tran_surfgreen_eps,
  double tran_transmission_energyrange[3],
  int TRAN_Channel_Nenergy,
  double *TRAN_Channel_energy,
  int TRAN_Channel_Num,
  int kloop,
  double *TRAN_Channel_kpoint,
  dcomplex ****EChannel,
  double ***eigentrans,
  double **eigentrans_sum)
{
  dcomplex w;
  dcomplex *GRL, *GRR, *GC_R;
  dcomplex *SigmaL_R, *SigmaR_R;
  dcomplex *rtS, *rtSinv;
  double *eval;
  dcomplex *ALbar, *GamRbar;

  int iw, ispin, iside;
  int **iwIdx;
  int Miw, Miwmax;
  int NUM_cs;
        
  /* allocate */
  GRL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0] * NUM_e[0]);
  GRR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1] * NUM_e[1]);

  GC_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaL_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  rtS = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  rtSinv = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  /*parallel setup*/

  iwIdx = (int**)malloc(sizeof(int*)*numprocs);
  Miwmax = (TRAN_Channel_Nenergy) / numprocs + 1;
  for (iw = 0; iw<numprocs; iw++) {
    iwIdx[iw] = (int*)malloc(sizeof(int)*Miwmax);
  }
  TRAN_Distribute_Node_Idx(0, TRAN_Channel_Nenergy - 1, numprocs, Miwmax,
                           iwIdx); /* output */

  /* L\owdin orthogonalization : obtain S^{1/2} & S^{-1/2} */

  NUM_cs = TRAN_Calc_OrtSpace(NUM_c, SCC, rtS, rtSinv);
  printf("  myid0 = %3d, #k : %4d, N_{ort} / N_{nonort} : %d / %d \n", 
    myid0, kloop, NUM_cs, NUM_c);

  ALbar = (dcomplex*)malloc(sizeof(dcomplex)*NUM_cs* NUM_cs);
  GamRbar = (dcomplex*)malloc(sizeof(dcomplex)*NUM_cs* NUM_cs);
  eval = (double*)malloc(sizeof(double)*NUM_c);

  /* parallel global iw 0:tran_transmission_energydiv-1 */
  /* parallel local  Miw 0:Miwmax-1                     */
  /* parallel variable iw=iwIdx[myid][Miw]              */

  for (Miw = 0; Miw<Miwmax; Miw++) {

    iw = iwIdx[myid][Miw];

    if (iw >= 0) {

      w.r = TRAN_Channel_energy[iw] + ChemP_e[0];
      w.i = tran_transmission_energyrange[2];

      /*
        printf("iw=%d of %d  w= % 9.6e % 9.6e \n" ,iw, tran_transmission_energydiv,  w.r,w.i);
      */

      for (ispin = 0; ispin <= SpinP_switch; ispin++) {

        /*****************************************************************
        Note that retarded and advanced Green functions and self energies
        are not conjugate comlex in case of the k-dependent case.
        **************************************************************/

        /* in case of retarded ones */

        iside = 0;
        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside][ispin], H01_e[iside][ispin],
                                   S00_e[iside], S01_e[iside],
                                   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRL);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL[ispin], SCL, SigmaL_R);

        iside = 1;
        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside][ispin], H01_e[iside][ispin],
                                   S00_e[iside], S01_e[iside],
                                   tran_surfgreen_iteration_max, tran_surfgreen_eps, GRR);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR[ispin], SCR, SigmaR_R);

        TRAN_Calc_CentGreen(w, NUM_c, SigmaL_R, SigmaR_R, HCC[ispin], SCC, GC_R);

        /* in case of advanced ones */
        /* Advanced Green's function is Hermit conjugate of the retare one */

        /* Gamma = i(Sigma - Sigma^+) -> Sigma */

        TRAN_Calc_Linewidth(NUM_c, SigmaL_R, SigmaR_R);

        /* G Gamma_L G^+ -> Sigma_L */

        TRAN_Calc_MatTrans(NUM_c, SigmaL_R, GC_R, "N", "C");

        /* L\"owdin orthogonalization */
        
        TRAN_Calc_LowdinOrt(NUM_c, SigmaL_R, SigmaR_R, NUM_cs, rtS, rtSinv, ALbar, GamRbar);

        /* Diagonalize ALbar -> its eigenvector */
        /* Then scale it : Albar_{ml} -> ALbar * sqrt(eig_l)  */

        TRAN_Calc_Diagonalize(NUM_cs, ALbar, eval, 1);

        /* Albar^+ Gambar Albar -> Gambar  */

        TRAN_Calc_MatTrans(NUM_cs, GamRbar, ALbar, "C", "N");

        /* Diagonalize transformed GamRbar */

        TRAN_Calc_Diagonalize(NUM_cs, GamRbar, eval, 0);

        /* Transform eigenchannel into non-orthogonal basis space */

       TRAN_Calc_ChannelLCAO(NUM_c, NUM_cs, rtSinv, ALbar, GamRbar, eval, GC_R, TRAN_Channel_Num,
          EChannel[iw][ispin],eigentrans[iw][ispin]);

       /* Output EigenChannel in the basis space */
        
        TRAN_Output_ChannelLCAO(myid0,kloop, iw, ispin, NUM_c,
          TRAN_Channel_kpoint, TRAN_Channel_energy[iw], eval, GC_R, eigentrans_sum);

      } /* for ispin */
    } /* if ( iw>=0 ) */
  } /* iw */

  /* freeing of arrays */
  free(GC_R);
  free(SigmaR_R);
  free(SigmaL_R);

  free(GRR);
  free(GRL);

  free(rtS);
  free(rtSinv);
  free(eval);

  free(ALbar);
  free(GamRbar);

  for (iw = 0; iw<numprocs; iw++) {
    free(iwIdx[iw]);
  }
  free(iwIdx);
} /* void MTRAN_EigenChannel */

void MTRAN_EigenChannel_NC(
  MPI_Comm comm1,
  int numprocs,
  int myid,
  int myid0,
  int SpinP_switch,
  double ChemP_e[2],
  int NUM_c,
  int NUM_e[2],
  dcomplex *H00_e[2],
  dcomplex *S00_e[2],
  dcomplex *H01_e[2],
  dcomplex *S01_e[2],
  dcomplex *HCC,
  dcomplex *HCL,
  dcomplex *HCR,
  dcomplex *SCC,
  dcomplex *SCL,
  dcomplex *SCR,
  double tran_surfgreen_iteration_max,
  double tran_surfgreen_eps,
  double tran_transmission_energyrange[3],
  int TRAN_Channel_Nenergy,
  double *TRAN_Channel_energy,
  int TRAN_Channel_Num,
  int kloop,
  double *TRAN_Channel_kpoint,
  dcomplex ***EChannel,
  double **eigentrans,
  double **eigentrans_sum)
{
  dcomplex w;
  dcomplex *GRL, *GRR, *GC_R;
  dcomplex *SigmaL_R, *SigmaR_R;
  dcomplex *rtS, *rtSinv;
  double *eval;
  dcomplex *ALbar, *GamRbar;

  int iw, iside;
  int **iwIdx;
  int Miw, Miwmax;
  int NUM_cs;

  /* allocate */
  GRL = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[0] * NUM_e[0]);
  GRR = (dcomplex*)malloc(sizeof(dcomplex)*NUM_e[1] * NUM_e[1]);

  GC_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaL_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  SigmaR_R = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  rtS = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);
  rtSinv = (dcomplex*)malloc(sizeof(dcomplex)*NUM_c* NUM_c);

  /*parallel setup*/

  iwIdx = (int**)malloc(sizeof(int*)*numprocs);
  Miwmax = (TRAN_Channel_Nenergy) / numprocs + 1;
  for (iw = 0; iw<numprocs; iw++) {
    iwIdx[iw] = (int*)malloc(sizeof(int)*Miwmax);
  }
  TRAN_Distribute_Node_Idx(0, TRAN_Channel_Nenergy - 1, numprocs, Miwmax,
    iwIdx); /* output */

  /* L\owdin orthogonalization : obtain S^{1/2} & S^{-1/2} */

  NUM_cs = TRAN_Calc_OrtSpace(NUM_c, SCC, rtS, rtSinv);
  printf("  myid0 = %3d, #k : %4d, N_{ort} / N_{nonort} : %d / %d \n",
    myid0, kloop, NUM_cs, NUM_c);

  ALbar = (dcomplex*)malloc(sizeof(dcomplex)*NUM_cs* NUM_cs);
  GamRbar = (dcomplex*)malloc(sizeof(dcomplex)*NUM_cs* NUM_cs);
  eval = (double*)malloc(sizeof(double)*NUM_c);

  /* parallel global iw 0:tran_transmission_energydiv-1 */
  /* parallel local  Miw 0:Miwmax-1                     */
  /* parallel variable iw=iwIdx[myid][Miw]              */

  for (Miw = 0; Miw<Miwmax; Miw++) {

    iw = iwIdx[myid][Miw];

    if (iw >= 0) {

      w.r = TRAN_Channel_energy[iw] + ChemP_e[0];
      w.i = tran_transmission_energyrange[2];

      /*
      printf("iw=%d of %d  w= % 9.6e % 9.6e \n" ,iw, tran_transmission_energydiv,  w.r,w.i);
      */

        /*****************************************************************
        Note that retarded and advanced Green functions and self energies
        are not conjugate comlex in case of the k-dependent case.
        **************************************************************/

        /* in case of retarded ones */

        iside = 0;
        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside], H01_e[iside],
          S00_e[iside], S01_e[iside],
          tran_surfgreen_iteration_max, tran_surfgreen_eps, GRL);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRL, NUM_c, HCL, SCL, SigmaL_R);

        iside = 1;
        TRAN_Calc_SurfGreen_direct(w, NUM_e[iside], H00_e[iside], H01_e[iside],
          S00_e[iside], S01_e[iside],
          tran_surfgreen_iteration_max, tran_surfgreen_eps, GRR);

        TRAN_Calc_SelfEnergy(w, NUM_e[iside], GRR, NUM_c, HCR, SCR, SigmaR_R);

        TRAN_Calc_CentGreen(w, NUM_c, SigmaL_R, SigmaR_R, HCC, SCC, GC_R);

        /* in case of advanced ones */
        /* Advanced Green's function is Hermit conjugate of the retare one */

        /* Gamma = i(Sigma - Sigma^+) -> Sigma */

        TRAN_Calc_Linewidth(NUM_c, SigmaL_R, SigmaR_R);

        /* G Gamma_L G^+ -> Sigma_L */

        TRAN_Calc_MatTrans(NUM_c, SigmaL_R, GC_R, "N", "C");

        /* L\"owdin orthogonalization */

        TRAN_Calc_LowdinOrt(NUM_c, SigmaL_R, SigmaR_R, NUM_cs, rtS, rtSinv, ALbar, GamRbar);

        /* Diagonalize ALbar -> its eigenvector */
        /* Then scale it : Albar_{ml} -> ALbar * sqrt(eig_l)  */

        TRAN_Calc_Diagonalize(NUM_cs, ALbar, eval, 1);

        /* Albar^+ Gambar Albar -> Gambar  */

        TRAN_Calc_MatTrans(NUM_cs, GamRbar, ALbar, "C", "N");

        /* Diagonalize transformed GamRbar */

        TRAN_Calc_Diagonalize(NUM_cs, GamRbar, eval, 0);

        /* Transform eigenchannel into non-orthogonal basis space */

        TRAN_Calc_ChannelLCAO(NUM_c, NUM_cs, rtSinv, ALbar, GamRbar, eval, GC_R, TRAN_Channel_Num,
          EChannel[iw],eigentrans[iw]);

        /* Output EigenChannel in the basis space */

        TRAN_Output_ChannelLCAO(myid0, kloop, iw, 0, NUM_c,
          TRAN_Channel_kpoint, TRAN_Channel_energy[iw], eval, GC_R, eigentrans_sum);

    } /* if ( iw>=0 ) */
  } /* iw */

  /* freeing of arrays */
  free(GC_R);
  free(SigmaR_R);
  free(SigmaL_R);

  free(GRR);
  free(GRL);

  free(rtS);
  free(rtSinv);
  free(eval);

  free(ALbar);
  free(GamRbar);

  for (iw = 0; iw<numprocs; iw++) {
    free(iwIdx[iw]);
  }
  free(iwIdx);
} /* void MTRAN_EigenChannel_NC */

