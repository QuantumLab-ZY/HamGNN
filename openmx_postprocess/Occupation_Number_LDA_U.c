/**********************************************************************
  Occupation_Number_LDA_U.c:

    Occupation_Number_LDA_U.c is a subrutine to calculate occupation
    number for LDA+U method.

  Log of Occupation_Number_LDA_U.c:

     14/April/2004   -- Released by M.J.Han (MJ)
     29/Nov  /2004   -- Modified by T.Ozaki (AIST)
     16/Feb  /2006   -- a constraint DFT for spin orientation
                        was added by T.Ozaki (AIST)
     01/March/2017   -- General LDA+U scheme was added by S.Ryee
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

#ifdef c_complex
#include <complex.h>
#endif

#define SCF_Enhance_OP 9
#define quickcalc_flag 0

static void occupation_onsite();
static void occupation_full();
static void occupation_dual();
static void mixing_occupation(int SCF_iter);
static void Induce_Orbital_Polarization(int Mc_AN);
static void Induce_Orbital_Polarization_Together(int Mc_AN);
static void Induce_NC_Orbital_Polarization(int Mc_AN);
static void make_v_eff(int SCF_iter, int SucceedReadingDMfile, double dUele);
static void make_NC_v_eff(int SCF_iter, int SucceedReadingDMfile, double dUele, double ECE[]);
static void Output_Collinear_OcpN();
static void Output_NonCollinear_OcpN();
static void Calc_dTN(int constraint_flag,
                     dcomplex TN[2][2],
                     dcomplex dTN[2][2][2][2],
                     dcomplex U[2][2],
                     double theta[2], double phi[2]);

void Calc_dSxyz(dcomplex TN[2][2],
                dcomplex dSx[2][2],
                dcomplex dSy[2][2],
                dcomplex dSz[2][2],
                double Nup[2], double Ndn[2],
                double theta[2], double phi[2]);

void Occupation_Number_LDA_U(int SCF_iter, int SucceedReadingDMfile, double dUele, double ECE[], char *mode)
{
  int l1, l2, mul1, mul2, mul3, mul4, m1, m2, to1, to2, to3, to4;
  int Mc_AN, Gc_AN, Cwan, num, l, m, mul, n, k, tno1, tno0;
  int wan1, wan2, i, j, spin, size1, size2;
  double sden, tmp0, sum, Uvalue;
  double Stime_atom, Etime_atom;
  int numprocs, myid, ID, tag = 999;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /****************************************************
                      find DM_onsite
  ****************************************************/

  /* on site */
  if (Hub_U_occupation == 0)
  {
    occupation_onsite();
  }

  /* full */
  else if (Hub_U_occupation == 1)
  {
    occupation_full();
  }

  /* dual */
  else if (Hub_U_occupation == 2)
  {
    occupation_dual();
  }

  /****************************************************
        induce orbital polarization if necessary
  ****************************************************/

  if (SCF_iter < SCF_Enhance_OP && SucceedReadingDMfile == 0 && Hub_U_switch == 1)
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      if (SpinP_switch == 0)
        Induce_Orbital_Polarization(Mc_AN);
      else if (SpinP_switch == 1)
        Induce_Orbital_Polarization_Together(Mc_AN);
      else if (SpinP_switch == 3)
        Induce_NC_Orbital_Polarization(Mc_AN);
    }
  }

  /****************************************************
            mixing of occupation number matrix
  ****************************************************/

  /*
  if      (SpinP_switch==0) mixing_occupation(SCF_iter);
  else if (SpinP_switch==1) mixing_occupation(SCF_iter);
  */

  /****************************************************
                       make v_eff
  ****************************************************/

  if (SpinP_switch != 3)
    make_v_eff(SCF_iter, SucceedReadingDMfile, dUele);
  else
    make_NC_v_eff(SCF_iter, SucceedReadingDMfile, dUele, ECE);

  /****************************************************
   write a file, *.DM_onsite
  ****************************************************/

  if (strcasecmp(mode, "write") == 0)
  {
    if (SpinP_switch != 3)
      Output_Collinear_OcpN();
    else
      Output_NonCollinear_OcpN();
  }
}

void mixing_occupation(int SCF_iter)
{
  int Mc_AN, Gc_AN, wan1, i, j, spin;
  double mixw0, mixw1, tmp;

  if (SCF_iter == 1)
  {
    mixw0 = 1.0;
    mixw1 = 0.0;
  }
  else
  {
    mixw0 = 1.0;
    mixw1 = 1.0 - mixw0;
  }

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
  {

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

    /* collinear */

    if (SpinP_switch != 3)
    {

      if (Cnt_switch == 0)
      {

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_NO[wan1]; j++)
            {
              tmp = mixw0 * DM_onsite[0][spin][Mc_AN][i][j] + mixw1 * DM_onsite[1][spin][Mc_AN][i][j];
              DM_onsite[0][spin][Mc_AN][i][j] = tmp;
              DM_onsite[1][spin][Mc_AN][i][j] = tmp;
            }
          }
        }
      }

      else
      {

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (i = 0; i < Spe_Total_CNO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_CNO[wan1]; j++)
            {
              tmp = mixw0 * DM_onsite[0][spin][Mc_AN][i][j] + mixw1 * DM_onsite[1][spin][Mc_AN][i][j];
              DM_onsite[0][spin][Mc_AN][i][j] = tmp;
              DM_onsite[1][spin][Mc_AN][i][j] = tmp;
            }
          }
        }
      }
    }

    /* non-collinear */

    else
    {
    }
  }
}

void occupation_onsite()
{
  int l1, l2, mul1, mul2, mul3, mul4, m1, m2, to1, to2, to3, to4;
  int Mc_AN, Gc_AN, Cwan, num, l, m, mul;
  int wan1, wan2, i, j, spin;
  double Re11, Re22, Re12, Im12, d1, d2, d3;
  double theta, phi, sit, cot, sip, cop;
  double Stime_atom, Etime_atom;
  double sden, tmp0, sum;
  int ***Cnt_index;

  /****************************************************
  allocation of arrays:

  int Cnt_index[List_YOUSO[25]+1]
                      [List_YOUSO[24]]
                      [2*(List_YOUSO[25]+1)+1];
  ****************************************************/

  Cnt_index = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    Cnt_index[i] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      Cnt_index[i][j] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
  {

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

    /****************************************************
      if (SpinP_switch!=3)

      collinear case
    ****************************************************/

    if (SpinP_switch != 3)
    {

      if (Cnt_switch == 0)
      {

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_NO[wan1]; j++)
            {

              /* store the DM_onsite --- MJ */

              DM_onsite[0][spin][Mc_AN][i][j] = DM[0][spin][Mc_AN][0][i][j];
            }
          }
        } /* spin */
      }

      /***********************************
      Important note:

      In case of orbital optimization,
      the U potential is applied to
      the primitive orbital.
      ***********************************/

      else
      {

        to3 = 0;
        for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
        {
          for (mul3 = 0; mul3 < Spe_Num_CBasis[wan1][l1]; mul3++)
          {
            for (m1 = 0; m1 < (2 * l1 + 1); m1++)
            {
              Cnt_index[l1][mul3][m1] = to3;
              to3++;
            }
          }
        }

        for (spin = 0; spin <= SpinP_switch; spin++)
        {

          to1 = 0;
          for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
          {
            for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
            {
              for (m1 = 0; m1 < (2 * l1 + 1); m1++)
              {

                to2 = 0;
                for (l2 = 0; l2 <= Spe_MaxL_Basis[wan1]; l2++)
                {
                  for (mul2 = 0; mul2 < Spe_Num_Basis[wan1][l2]; mul2++)
                  {
                    for (m2 = 0; m2 < (2 * l2 + 1); m2++)
                    {

                      sum = 0.0;
                      for (mul3 = 0; mul3 < Spe_Num_CBasis[wan1][l1]; mul3++)
                      {
                        for (mul4 = 0; mul4 < Spe_Num_CBasis[wan1][l2]; mul4++)
                        {

                          to3 = Cnt_index[l1][mul3][m1];
                          to4 = Cnt_index[l2][mul4][m2];

                          sum += CntCoes[Mc_AN][to3][mul1] * CntCoes[Mc_AN][to4][mul2] * DM[0][spin][Mc_AN][0][to3][to4];
                        }
                      }

                      /* store the DM_onsite --- MJ */
                      DM_onsite[0][spin][Mc_AN][to1][to2] = sum;

                      to2++;
                    }
                  }
                }

                to1++;
              }
            }
          }
        }

      } /* else */

    } /* if (SpinP_switch!=3) */

    /****************************************************
      if (SpinP_switch==3)

      spin non-collinear
    ****************************************************/

    else
    {

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {

          /* store NC_OcpN */

          NC_OcpN[0][0][0][Mc_AN][i][j].r = DM[0][0][Mc_AN][0][i][j];
          NC_OcpN[0][1][1][Mc_AN][i][j].r = DM[0][1][Mc_AN][0][i][j];
          NC_OcpN[0][0][1][Mc_AN][i][j].r = DM[0][2][Mc_AN][0][i][j];
          NC_OcpN[0][1][0][Mc_AN][i][j].r = DM[0][2][Mc_AN][0][j][i];

          NC_OcpN[0][0][0][Mc_AN][i][j].i = iDM[0][0][Mc_AN][0][i][j];
          NC_OcpN[0][1][1][Mc_AN][i][j].i = iDM[0][1][Mc_AN][0][i][j];
          NC_OcpN[0][0][1][Mc_AN][i][j].i = DM[0][3][Mc_AN][0][i][j];
          NC_OcpN[0][1][0][Mc_AN][i][j].i = -DM[0][3][Mc_AN][0][j][i];
        }
      }

      /*
      printf("Re 00 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][0][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 11 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][1][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 01 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][1][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 10 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][0][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Im 00 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][0][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 11 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][1][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 01 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][1][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 10 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][0][Mc_AN][i][j].i);
  }
        printf("\n");
      }
      */
    }

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

  } /* Mc_AN */

  /****************************************************
  freeing of arrays:

  int Cnt_index[List_YOUSO[25]+1]
                      [List_YOUSO[24]]
                      [2*(List_YOUSO[25]+1)+1];
  ****************************************************/

  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      free(Cnt_index[i][j]);
    }
    free(Cnt_index[i]);
  }
  free(Cnt_index);
}

void occupation_full()
{
  int l1, l2, mul1, mul2, mul3, mul4, m1, m2, to1, to2, to3, to4;
  int Mc_AN, Gc_AN, Cwan, num, l, m, n, mul, kl, hL_AN, hR_AN;
  int MR_AN, ML_AN, GR_AN, GL_AN, Rwan, Lwan;
  int wan1, wan2, i, j, spin, size1, size2;
  int tno0, tno1, tno2, Hwan, h_AN, Gh_AN, k, p, p0;
  double Re11, Re22, Re12, Im12, d1, d2, d3;
  double theta, phi, sit, cot, sip, cop;
  double *Lsum, *Rsum;
  double Stime_atom, Etime_atom;
  double sden, tmp0, sum, ocn;
  double ReOcn00, ReOcn11, ReOcn01;
  double ImOcn00, ImOcn11, ImOcn01;
  double *****DM0;
  double *****iDM0;
  double *tmp_array;
  double *tmp_array2;
  int ***Cnt_index, *Snd_DM0_Size, *Rcv_DM0_Size;
  int numprocs, myid, ID, IDS, IDR, tag = 999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /****************************************************
  allocation of arrays:

  int Cnt_index[List_YOUSO[25]+1]
                      [List_YOUSO[24]]
                      [2*(List_YOUSO[25]+1)+1];

  double DM0[SpinP_switch+1]
                   [Matomnum+MatomnumF+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_NO[Cwan]]
                   [Spe_Total_NO[Hwan]];

  int Snd_DM0_Size[numprocs];
  int Rcv_DM0_Size[numprocs];

  double Lsum[List_YOUSO[7]];
  double Rsum[List_YOUSO[7]];
  ****************************************************/

  /* Cnt_index */

  Cnt_index = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    Cnt_index[i] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      Cnt_index[i][j] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  /* DM0 */

  DM0 = (double *****)malloc(sizeof(double ****) * (SpinP_switch + 1));
  for (k = 0; k <= SpinP_switch; k++)
  {
    DM0[k] = (double ****)malloc(sizeof(double ***) * (Matomnum + MatomnumF + 1));
    FNAN[0] = 0;
    for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
    {

      if (Mc_AN == 0)
      {
        Gc_AN = 0;
        tno0 = 1;
      }
      else
      {
        Gc_AN = F_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];
      }

      DM0[k][Mc_AN] = (double ***)malloc(sizeof(double **) * (FNAN[Gc_AN] + 1));
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {

        if (Mc_AN == 0)
        {
          tno1 = 1;
        }
        else
        {
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        }

        DM0[k][Mc_AN][h_AN] = (double **)malloc(sizeof(double *) * tno0);
        for (i = 0; i < tno0; i++)
        {
          DM0[k][Mc_AN][h_AN][i] = (double *)malloc(sizeof(double) * tno1);
        }
      }
    }
  }

  /* Snd_DM0_Size and Rcv_DM0_Size */

  Snd_DM0_Size = (int *)malloc(sizeof(int) * numprocs);
  Rcv_DM0_Size = (int *)malloc(sizeof(int) * numprocs);

  /* Lsum and Rsum */

  Lsum = (double *)malloc(sizeof(double) * List_YOUSO[7]);
  Rsum = (double *)malloc(sizeof(double) * List_YOUSO[7]);

  /****************************************************
    DM[k][Matomnum] -> DM0
  ****************************************************/

  for (k = 0; k <= SpinP_switch; k++)
  {
    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_NO[wan1];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {
        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno2 = Spe_Total_NO[Hwan];
        for (i = 0; i < tno1; i++)
        {
          for (j = 0; j < tno2; j++)
          {
            DM0[k][Mc_AN][h_AN][i][j] = DM[0][k][Mc_AN][h_AN][i][j];
          }
        }
      }
    }
  }

  /****************************************************
    MPI: DM0
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID = 0; ID < numprocs; ID++)
  {

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID != 0)
    {
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS] != 0)
      {

        size1 = 0;
        for (k = 0; k <= SpinP_switch; k++)
        {
          for (n = 0; n < F_Snd_Num[IDS]; n++)
          {
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_NO[Cwan];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
            {
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_NO[Hwan];
              size1 += tno1 * tno2;
            }
          }
        }

        Snd_DM0_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else
      {
        Snd_DM0_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR] != 0)
      {
        tag = 999;
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_DM0_Size[IDR] = size2;
      }
      else
      {
        Rcv_DM0_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS] != 0)
        MPI_Wait(&request, &stat);
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID = 0; ID < numprocs; ID++)
  {

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID != 0)
    {

      /*****************************
              sending of data
      *****************************/

      if (F_Snd_Num[IDS] != 0)
      {

        size1 = Snd_DM0_Size[IDS];

        /* allocation of array */

        tmp_array = (double *)malloc(sizeof(double) * size1);

        /* multidimentional array to vector array */

        num = 0;
        for (k = 0; k <= SpinP_switch; k++)
        {
          for (n = 0; n < F_Snd_Num[IDS]; n++)
          {
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_NO[Cwan];
            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
            {
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_NO[Hwan];
              for (i = 0; i < tno1; i++)
              {
                for (j = 0; j < tno2; j++)
                {
                  tmp_array[num] = DM0[k][Mc_AN][h_AN][i][j];
                  num++;
                }
              }
            }
          }
        }

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR] != 0)
      {

        size2 = Rcv_DM0_Size[IDR];

        /* allocation of array */
        tmp_array2 = (double *)malloc(sizeof(double) * size2);

        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (k = 0; k <= SpinP_switch; k++)
        {
          Mc_AN = F_TopMAN[IDR] - 1;
          for (n = 0; n < F_Rcv_Num[IDR]; n++)
          {
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_NO[Cwan];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
            {
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno2 = Spe_Total_NO[Hwan];
              for (i = 0; i < tno1; i++)
              {
                for (j = 0; j < tno2; j++)
                {
                  DM0[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
                }
              }
            }
          }
        }

        /* freeing of array */
        free(tmp_array2);
      }

      if (F_Snd_Num[IDS] != 0)
      {
        MPI_Wait(&request, &stat);
        free(tmp_array); /* freeing of array */
      }
    }
  }

  /****************************************************
                     collinear case
  ****************************************************/

  if (SpinP_switch != 3)
  {

    /****************************************************
                    calculate DM_full
    ****************************************************/

    if (Cnt_switch == 0)
    {

      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];

        for (spin = 0; spin <= SpinP_switch; spin++)
        {

          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_NO[wan1]; j++)
            {

              /* store the DM_full */

              ocn = 0.0;
              for (hL_AN = 0; hL_AN <= FNAN[Gc_AN]; hL_AN++)
              {
                GL_AN = natn[Gc_AN][hL_AN];
                ML_AN = F_G2M[GL_AN];
                Lwan = WhatSpecies[GL_AN];

                for (hR_AN = 0; hR_AN <= FNAN[Gc_AN]; hR_AN++)
                {
                  GR_AN = natn[Gc_AN][hR_AN];
                  MR_AN = F_G2M[GR_AN];
                  Rwan = WhatSpecies[GR_AN];
                  kl = RMI1[Mc_AN][hL_AN][hR_AN];

                  if (0 <= kl)
                  {
                    for (m = 0; m < Spe_Total_CNO[Lwan]; m++)
                    {
                      for (n = 0; n < Spe_Total_CNO[Rwan]; n++)
                      {
                        ocn += DM0[spin][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];
                      }
                    }
                  }
                }
              }

              DM_onsite[0][spin][Mc_AN][i][j] = ocn;
            }
          }

        } /* spin  */

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Mc_AN */
    }

    /***********************************
    In case of orbital optimization

    Important note:
    In case of orbital optimization,
    the U potential is applied to
    the primitive orbital.
    ***********************************/

    else
    {

      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_NO[wan1]; j++)
            {

              /* store the DM_onsite --- MJ */

              ocn = 0.0;
              for (hL_AN = 0; hL_AN <= FNAN[Gc_AN]; hL_AN++)
              {
                GL_AN = natn[Gc_AN][hL_AN];
                ML_AN = F_G2M[GL_AN];
                Lwan = WhatSpecies[GL_AN];

                for (hR_AN = 0; hR_AN <= FNAN[Gc_AN]; hR_AN++)
                {
                  GR_AN = natn[Gc_AN][hR_AN];
                  MR_AN = F_G2M[GR_AN];
                  Rwan = WhatSpecies[GR_AN];
                  kl = RMI1[Mc_AN][hL_AN][hR_AN];

                  if (0 <= kl)
                  {

                    for (m = 0; m < Spe_Total_CNO[Lwan]; m++)
                    {
                      Lsum[m] = 0.0;
                      for (p = 0; p < Spe_Specified_Num[Lwan][m]; p++)
                      {
                        p0 = Spe_Trans_Orbital[Lwan][m][p];
                        Lsum[m] += CntCoes[ML_AN][m][p] * OLP[0][Mc_AN][hL_AN][i][p0];
                      }
                    }

                    for (n = 0; n < Spe_Total_CNO[Rwan]; n++)
                    {
                      Rsum[n] = 0.0;
                      for (p = 0; p < Spe_Specified_Num[Rwan][n]; p++)
                      {
                        p0 = Spe_Trans_Orbital[Rwan][n][p];
                        Rsum[n] += CntCoes[MR_AN][n][p] * OLP[0][Mc_AN][hR_AN][j][p0];
                      }
                    }

                    for (m = 0; m < Spe_Total_CNO[Lwan]; m++)
                    {
                      for (n = 0; n < Spe_Total_CNO[Rwan]; n++)
                      {
                        ocn += DM0[spin][ML_AN][kl][m][n] * Lsum[m] * Rsum[n];
                      }
                    }
                  }
                }
              }

              DM_onsite[0][spin][Mc_AN][i][j] = ocn;
            }
          }
        } /* spin  */

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Mc_AN */
    }   /* else */

  } /* if (SpinP_switch!=3) */

  /****************************************************
                   non-collinear case
  ****************************************************/

  else
  {

    /* allocation of iDM0 */

    iDM0 = (double *****)malloc(sizeof(double ****) * 2);
    for (k = 0; k < 2; k++)
    {
      iDM0[k] = (double ****)malloc(sizeof(double ***) * (Matomnum + MatomnumF + 1));
      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        iDM0[k][Mc_AN] = (double ***)malloc(sizeof(double **) * (FNAN[Gc_AN] + 1));
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          if (Mc_AN == 0)
          {
            tno1 = 1;
          }
          else
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          iDM0[k][Mc_AN][h_AN] = (double **)malloc(sizeof(double *) * tno0);
          for (i = 0; i < tno0; i++)
          {
            iDM0[k][Mc_AN][h_AN][i] = (double *)malloc(sizeof(double) * tno1);
          }
        }
      }
    }

    /****************************************************
      iDM[0][k][Matomnum] -> iDM0
    ****************************************************/

    for (k = 0; k < 2; k++)
    {
      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {
        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_NO[wan1];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_NO[Hwan];
          for (i = 0; i < tno1; i++)
          {
            for (j = 0; j < tno2; j++)
            {
              iDM0[k][Mc_AN][h_AN][i][j] = iDM[0][k][Mc_AN][h_AN][i][j];
            }
          }
        }
      }
    }

    /****************************************************
    MPI: iDM0
    ****************************************************/

    /***********************************
             set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {
        tag = 999;

        /* find data size to send block data */
        if (F_Snd_Num[IDS] != 0)
        {

          size1 = 0;
          for (k = 0; k < 2; k++)
          {
            for (n = 0; n < F_Snd_Num[IDS]; n++)
            {
              Mc_AN = Snd_MAN[IDS][n];
              Gc_AN = Snd_GAN[IDS][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                size1 += tno1 * tno2;
              }
            }
          }

          Snd_DM0_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else
        {
          Snd_DM0_Size[IDS] = 0;
        }

        /* receiving of size of data */

        if (F_Rcv_Num[IDR] != 0)
        {
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_DM0_Size[IDR] = size2;
        }
        else
        {
          Rcv_DM0_Size[IDR] = 0;
        }

        if (F_Snd_Num[IDS] != 0)
          MPI_Wait(&request, &stat);
      }
    }

    /***********************************
             data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {

        /*****************************
                    sending of data
        *****************************/

        if (F_Snd_Num[IDS] != 0)
        {

          size1 = Snd_DM0_Size[IDS];

          /* allocation of array */

          tmp_array = (double *)malloc(sizeof(double) * size1);

          /* multidimentional array to vector array */

          num = 0;
          for (k = 0; k < 2; k++)
          {
            for (n = 0; n < F_Snd_Num[IDS]; n++)
            {
              Mc_AN = Snd_MAN[IDS][n];
              Gc_AN = Snd_GAN[IDS][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                for (i = 0; i < tno1; i++)
                {
                  for (j = 0; j < tno2; j++)
                  {
                    tmp_array[num] = iDM0[k][Mc_AN][h_AN][i][j];
                    num++;
                  }
                }
              }
            }
          }

          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
        }

        /*****************************
               receiving of block data
        *****************************/

        if (F_Rcv_Num[IDR] != 0)
        {

          size2 = Rcv_DM0_Size[IDR];

          /* allocation of array */
          tmp_array2 = (double *)malloc(sizeof(double) * size2);

          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          for (k = 0; k < 2; k++)
          {
            Mc_AN = F_TopMAN[IDR] - 1;
            for (n = 0; n < F_Rcv_Num[IDR]; n++)
            {
              Mc_AN++;
              Gc_AN = Rcv_GAN[IDR][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];

              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                for (i = 0; i < tno1; i++)
                {
                  for (j = 0; j < tno2; j++)
                  {
                    iDM0[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
                    num++;
                  }
                }
              }
            }
          }

          /* freeing of array */
          free(tmp_array2);
        }

        if (F_Snd_Num[IDS] != 0)
        {
          MPI_Wait(&request, &stat);
          free(tmp_array); /* freeing of array */
        }
      }
    }

    /****************************************************
                    calculate NC_OcpN
    ****************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {

          /* store NC_OcpN */

          ReOcn00 = 0.0;
          ReOcn11 = 0.0;
          ReOcn01 = 0.0;

          ImOcn00 = 0.0;
          ImOcn11 = 0.0;
          ImOcn01 = 0.0;

          for (hL_AN = 0; hL_AN <= FNAN[Gc_AN]; hL_AN++)
          {
            GL_AN = natn[Gc_AN][hL_AN];
            ML_AN = F_G2M[GL_AN];
            Lwan = WhatSpecies[GL_AN];

            for (hR_AN = 0; hR_AN <= FNAN[Gc_AN]; hR_AN++)
            {
              GR_AN = natn[Gc_AN][hR_AN];
              MR_AN = F_G2M[GR_AN];
              Rwan = WhatSpecies[GR_AN];
              kl = RMI1[Mc_AN][hL_AN][hR_AN];

              if (0 <= kl)
              {
                for (m = 0; m < Spe_Total_CNO[Lwan]; m++)
                {
                  for (n = 0; n < Spe_Total_CNO[Rwan]; n++)
                  {

                    ReOcn00 += DM0[0][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];

                    ReOcn11 += DM0[1][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];

                    ReOcn01 += DM0[2][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];

                    ImOcn00 += iDM0[0][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];

                    ImOcn11 += iDM0[1][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];

                    ImOcn01 += DM0[3][ML_AN][kl][m][n] *
                               OLP[0][Mc_AN][hL_AN][i][m] *
                               OLP[0][Mc_AN][hR_AN][j][n];
                  }
                }
              }
            }
          }

          NC_OcpN[0][0][0][Mc_AN][i][j].r = ReOcn00;
          NC_OcpN[0][1][1][Mc_AN][i][j].r = ReOcn11;
          NC_OcpN[0][0][1][Mc_AN][i][j].r = ReOcn01;
          NC_OcpN[0][1][0][Mc_AN][j][i].r = ReOcn01;

          NC_OcpN[0][0][0][Mc_AN][i][j].i = ImOcn00;
          NC_OcpN[0][1][1][Mc_AN][i][j].i = ImOcn11;
          NC_OcpN[0][0][1][Mc_AN][i][j].i = ImOcn01;
          NC_OcpN[0][1][0][Mc_AN][j][i].i = -ImOcn01;
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      /*
      printf("Re 00 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][0][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 11 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][1][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 01 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][1][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 10 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][0][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Im 00 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][0][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 11 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][1][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 01 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][1][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 10 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][0][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      */

    } /* Mc_AN */
  }

  /****************************************************
  freeing of arrays:

  int Cnt_index[List_YOUSO[25]+1]
                      [List_YOUSO[24]]
                      [2*(List_YOUSO[25]+1)+1];

  double DM0[SpinP_switch+1]
                   [Matomnum+MatomnumF+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_NO[Cwan]]
                   [Spe_Total_NO[Hwan]]

  int Snd_DM0_Size[numprocs];
  int Rcv_DM0_Size[numprocs];
  double Lsum[List_YOUSO[7]];
  double Rsum[List_YOUSO[7]];
  ****************************************************/

  /* Cnt_index */

  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      free(Cnt_index[i][j]);
    }
    free(Cnt_index[i]);
  }
  free(Cnt_index);

  /* DM0 */

  for (k = 0; k <= SpinP_switch; k++)
  {
    FNAN[0] = 0;
    for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
    {

      if (Mc_AN == 0)
      {
        Gc_AN = 0;
        tno0 = 1;
      }
      else
      {
        Gc_AN = F_M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];
      }

      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {

        if (Mc_AN == 0)
        {
          tno1 = 1;
        }
        else
        {
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno1 = Spe_Total_NO[Hwan];
        }

        for (i = 0; i < tno0; i++)
        {
          free(DM0[k][Mc_AN][h_AN][i]);
        }
        free(DM0[k][Mc_AN][h_AN]);
      }
      free(DM0[k][Mc_AN]);
    }
    free(DM0[k]);
  }
  free(DM0);

  /* Snd_DM0_Size and Rcv_DM0_Size */

  free(Snd_DM0_Size);
  free(Rcv_DM0_Size);

  /* Lsum and Rsum */

  free(Lsum);
  free(Rsum);

  /* freeing of iDM0 */

  if (SpinP_switch == 3)
  {

    for (k = 0; k < 2; k++)
    {

      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          if (Mc_AN == 0)
          {
            tno1 = 1;
          }
          else
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          for (i = 0; i < tno0; i++)
          {
            free(iDM0[k][Mc_AN][h_AN][i]);
          }
          free(iDM0[k][Mc_AN][h_AN]);
        }
        free(iDM0[k][Mc_AN]);
      }
      free(iDM0[k]);
    }
    free(iDM0);
  }
}

void occupation_dual()
{
  int l1, l2, mul1, mul2, mul3, mul4, m1, m2, to1, to2, to3, to4;
  int Mc_AN, Gc_AN, Cwan, num, l, m, n, mul, kl, hL_AN, hR_AN;
  int MR_AN, ML_AN, GR_AN, GL_AN, Rwan, Lwan, Mh_AN;
  int wan1, wan2, i, j, spin, size1, size2;
  int tno0, tno1, tno2, Hwan, h_AN, Gh_AN, k, p, p0;
  double Re11, Re22, Re12, Im12, d1, d2, d3;
  double theta, phi, sit, cot, sip, cop;
  double ReOcn00, ReOcn11, ReOcn01;
  double ImOcn00, ImOcn11, ImOcn01;
  double **DecMulP, ***Primitive_DM;
  double *****DM0;
  double *****iDM0;
  int *Snd_DM0_Size, *Rcv_DM0_Size;
  double *tmp_array;
  double *tmp_array2;
  double Stime_atom, Etime_atom;
  double sden, tmp0, sum, ocn;
  int ***Cnt_index1, ***Cnt_index2;
  int numprocs, myid, ID, IDS, IDR, tag = 999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /****************************************************
  allocation of arrays:

  int Cnt_index1[List_YOUSO[25]+1]
                       [List_YOUSO[24]]
                       [2*(List_YOUSO[25]+1)+1];

  int Cnt_index2[List_YOUSO[25]+1]
                       [List_YOUSO[24]]
                       [2*(List_YOUSO[25]+1)+1];

  double DecMulP[SpinP_switch+1][List_YOUSO[7]];
  double Primitive_DM[SpinP_switch+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[7]];
  ****************************************************/

  /* Cnt_index1 and Cnt_index2 */

  Cnt_index1 = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    Cnt_index1[i] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      Cnt_index1[i][j] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  Cnt_index2 = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    Cnt_index2[i] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      Cnt_index2[i][j] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  /* DecMulP */

  DecMulP = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
  for (spin = 0; spin <= SpinP_switch; spin++)
  {
    DecMulP[spin] = (double *)malloc(sizeof(double) * List_YOUSO[7]);
  }

  /* Primitive_DM */

  Primitive_DM = (double ***)malloc(sizeof(double **) * (SpinP_switch + 1));
  for (spin = 0; spin <= SpinP_switch; spin++)
  {
    Primitive_DM[spin] = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
    for (i = 0; i < List_YOUSO[7]; i++)
    {
      Primitive_DM[spin][i] = (double *)malloc(sizeof(double) * List_YOUSO[7]);
    }
  }

  /****************************************************
                     collinear case
  ****************************************************/

  if (SpinP_switch != 3)
  {

    /****************************************************
                     calculate DM_onsite
     ****************************************************/

    if (Cnt_switch == 0)
    {

      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];

        for (spin = 0; spin <= SpinP_switch; spin++)
        {

          for (m = 0; m < Spe_Total_NO[wan1]; m++)
          {
            for (n = 0; n < Spe_Total_NO[wan1]; n++)
            {

              tmp0 = 0.0;
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                wan2 = WhatSpecies[Gh_AN];
                for (k = 0; k < Spe_Total_NO[wan2]; k++)
                {
                  tmp0 += 0.5 * (DM[0][spin][Mc_AN][h_AN][n][k] * OLP[0][Mc_AN][h_AN][m][k] + DM[0][spin][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k]);
                }
              }

              DM_onsite[0][spin][Mc_AN][m][n] = tmp0;
            }
          }
        } /* spin */

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Mc_AN */
    }

    /***********************************
    In case of orbital optimization

    Important note:
    In case of orbital optimization,
    the U potential is applied to
    the primitive orbital.
    ***********************************/

    else
    {

      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {

        dtime(&Stime_atom);

        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];

        to3 = 0;
        for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
        {
          for (mul3 = 0; mul3 < Spe_Num_CBasis[wan1][l1]; mul3++)
          {
            for (m1 = 0; m1 < (2 * l1 + 1); m1++)
            {
              Cnt_index1[l1][mul3][m1] = to3;
              to3++;
            }
          }
        }

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_NO[wan1]; j++)
            {
              DM_onsite[0][spin][Mc_AN][i][j] = 0.0;
            }
          }
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          Gh_AN = natn[Gc_AN][h_AN];
          Mh_AN = F_G2M[Gh_AN];
          wan2 = WhatSpecies[Gh_AN];

          to3 = 0;
          for (l1 = 0; l1 <= Spe_MaxL_Basis[wan2]; l1++)
          {
            for (mul3 = 0; mul3 < Spe_Num_CBasis[wan2][l1]; mul3++)
            {
              for (m1 = 0; m1 < (2 * l1 + 1); m1++)
              {
                Cnt_index2[l1][mul3][m1] = to3;
                to3++;
              }
            }
          }

          /* transform DM of contracted to that of primitive orbitals */

          for (spin = 0; spin <= SpinP_switch; spin++)
          {

            to1 = 0;
            for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
            {
              for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
              {
                for (m1 = 0; m1 < (2 * l1 + 1); m1++)
                {

                  to2 = 0;
                  for (l2 = 0; l2 <= Spe_MaxL_Basis[wan2]; l2++)
                  {
                    for (mul2 = 0; mul2 < Spe_Num_Basis[wan2][l2]; mul2++)
                    {
                      for (m2 = 0; m2 < (2 * l2 + 1); m2++)
                      {

                        sum = 0.0;
                        for (mul3 = 0; mul3 < Spe_Num_CBasis[wan1][l1]; mul3++)
                        {
                          for (mul4 = 0; mul4 < Spe_Num_CBasis[wan2][l2]; mul4++)
                          {

                            to3 = Cnt_index1[l1][mul3][m1];
                            to4 = Cnt_index2[l2][mul4][m2];

                            sum += CntCoes[Mc_AN][to3][mul1] * CntCoes[Mh_AN][to4][mul2] * DM[0][spin][Mc_AN][h_AN][to3][to4];
                          }
                        }

                        Primitive_DM[spin][to1][to2] = sum;

                        to2++;
                      }
                    }
                  }

                  to1++;
                }
              }
            }
          } /* spin */

          /* calculate DM_onsite with respect to primitive orbitals */

          for (spin = 0; spin <= SpinP_switch; spin++)
          {
            for (m = 0; m < Spe_Total_NO[wan1]; m++)
            {
              for (n = 0; n < Spe_Total_NO[wan1]; n++)
              {

                tmp0 = 0.0;
                for (k = 0; k < Spe_Total_NO[wan2]; k++)
                {
                  tmp0 += 0.5 * (Primitive_DM[spin][n][k] * OLP[0][Mc_AN][h_AN][m][k] + Primitive_DM[spin][m][k] * OLP[0][Mc_AN][h_AN][n][k]);
                }

                DM_onsite[0][spin][Mc_AN][m][n] += tmp0;
              }
            }
          }

        } /* h_AN */

        dtime(&Etime_atom);
        time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      } /* Mc_AN */
    }   /* else */

    /*
    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
        printf("DM_onsite spin=%2d Mc_AN=%2d \n",spin,Mc_AN);

  Gc_AN = M2G[Mc_AN];
  wan1 = WhatSpecies[Gc_AN];

        for (m=0; m<Spe_Total_NO[wan1]; m++){
    for (n=0; n<Spe_Total_NO[wan1]; n++){
            printf("%8.4f ",DM_onsite[0][spin][Mc_AN][m][n]);
    }
          printf("\n");
  }
      }
    }
    */

    /*
    for (spin=0; spin<=SpinP_switch; spin++){
      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
  Gc_AN = M2G[Mc_AN];
  wan1 = WhatSpecies[Gc_AN];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          printf("DM spin=%2d Mc_AN=%2d h_AN=%2d\n",spin,Mc_AN,h_AN);
    for (m=0; m<Spe_Total_NO[wan1]; m++){
      for (n=0; n<Spe_Total_NO[wan1]; n++){
        printf("%8.4f ",DM[0][spin][Mc_AN][h_AN][m][n]);
      }
      printf("\n");
    }
  }
      }
    }
    */
  }

  /****************************************************
                   non-collinear case
  ****************************************************/

  else
  {

    /* DM0 */

    DM0 = (double *****)malloc(sizeof(double ****) * (SpinP_switch + 1));
    for (k = 0; k <= SpinP_switch; k++)
    {
      DM0[k] = (double ****)malloc(sizeof(double ***) * (Matomnum + MatomnumF + 1));
      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        DM0[k][Mc_AN] = (double ***)malloc(sizeof(double **) * (FNAN[Gc_AN] + 1));
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          if (Mc_AN == 0)
          {
            tno1 = 1;
          }
          else
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          DM0[k][Mc_AN][h_AN] = (double **)malloc(sizeof(double *) * tno0);
          for (i = 0; i < tno0; i++)
          {
            DM0[k][Mc_AN][h_AN][i] = (double *)malloc(sizeof(double) * tno1);
          }
        }
      }
    }

    /* Snd_DM0_Size and Rcv_DM0_Size */

    Snd_DM0_Size = (int *)malloc(sizeof(int) * numprocs);
    Rcv_DM0_Size = (int *)malloc(sizeof(int) * numprocs);

    /****************************************************
    DM[k][Matomnum] -> DM0
    ****************************************************/

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {
        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_NO[wan1];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_NO[Hwan];
          for (i = 0; i < tno1; i++)
          {
            for (j = 0; j < tno2; j++)
            {
              DM0[k][Mc_AN][h_AN][i][j] = DM[0][k][Mc_AN][h_AN][i][j];
            }
          }
        }
      }
    }

    /****************************************************
    MPI: DM0
    ****************************************************/

    /***********************************
             set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {
        tag = 999;

        /* find data size to send block data */
        if (F_Snd_Num[IDS] != 0)
        {

          size1 = 0;
          for (k = 0; k <= SpinP_switch; k++)
          {
            for (n = 0; n < F_Snd_Num[IDS]; n++)
            {
              Mc_AN = Snd_MAN[IDS][n];
              Gc_AN = Snd_GAN[IDS][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                size1 += tno1 * tno2;
              }
            }
          }

          Snd_DM0_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else
        {
          Snd_DM0_Size[IDS] = 0;
        }

        /* receiving of size of data */

        if (F_Rcv_Num[IDR] != 0)
        {
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_DM0_Size[IDR] = size2;
        }
        else
        {
          Rcv_DM0_Size[IDR] = 0;
        }

        if (F_Snd_Num[IDS] != 0)
          MPI_Wait(&request, &stat);
      }
    }

    /***********************************
             data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {

        /*****************************
                    sending of data
        *****************************/

        if (F_Snd_Num[IDS] != 0)
        {

          size1 = Snd_DM0_Size[IDS];

          /* allocation of array */

          tmp_array = (double *)malloc(sizeof(double) * size1);

          /* multidimentional array to vector array */

          num = 0;
          for (k = 0; k <= SpinP_switch; k++)
          {
            for (n = 0; n < F_Snd_Num[IDS]; n++)
            {
              Mc_AN = Snd_MAN[IDS][n];
              Gc_AN = Snd_GAN[IDS][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                for (i = 0; i < tno1; i++)
                {
                  for (j = 0; j < tno2; j++)
                  {
                    tmp_array[num] = DM0[k][Mc_AN][h_AN][i][j];
                    num++;
                  }
                }
              }
            }
          }

          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
        }

        /*****************************
               receiving of block data
        *****************************/

        if (F_Rcv_Num[IDR] != 0)
        {

          size2 = Rcv_DM0_Size[IDR];

          /* allocation of array */
          tmp_array2 = (double *)malloc(sizeof(double) * size2);

          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          for (k = 0; k <= SpinP_switch; k++)
          {
            Mc_AN = F_TopMAN[IDR] - 1;
            for (n = 0; n < F_Rcv_Num[IDR]; n++)
            {
              Mc_AN++;
              Gc_AN = Rcv_GAN[IDR][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];

              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                for (i = 0; i < tno1; i++)
                {
                  for (j = 0; j < tno2; j++)
                  {
                    DM0[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
                    num++;
                  }
                }
              }
            }
          }

          /* freeing of array */
          free(tmp_array2);
        }

        if (F_Snd_Num[IDS] != 0)
        {
          MPI_Wait(&request, &stat);
          free(tmp_array); /* freeing of array */
        }
      }
    }

    /* allocation of iDM0 */

    iDM0 = (double *****)malloc(sizeof(double ****) * 2);
    for (k = 0; k < 2; k++)
    {
      iDM0[k] = (double ****)malloc(sizeof(double ***) * (Matomnum + MatomnumF + 1));
      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        iDM0[k][Mc_AN] = (double ***)malloc(sizeof(double **) * (FNAN[Gc_AN] + 1));
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          if (Mc_AN == 0)
          {
            tno1 = 1;
          }
          else
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          iDM0[k][Mc_AN][h_AN] = (double **)malloc(sizeof(double *) * tno0);
          for (i = 0; i < tno0; i++)
          {
            iDM0[k][Mc_AN][h_AN][i] = (double *)malloc(sizeof(double) * tno1);
          }
        }
      }
    }

    /****************************************************
      iDM[0][k][Matomnum] -> iDM0
    ****************************************************/

    for (k = 0; k < 2; k++)
    {
      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {
        Gc_AN = M2G[Mc_AN];
        wan1 = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_NO[wan1];
        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {
          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_NO[Hwan];
          for (i = 0; i < tno1; i++)
          {
            for (j = 0; j < tno2; j++)
            {
              iDM0[k][Mc_AN][h_AN][i][j] = iDM[0][k][Mc_AN][h_AN][i][j];
            }
          }
        }
      }
    }

    /****************************************************
    MPI: iDM0
    ****************************************************/

    /***********************************
             set data size
    ************************************/

    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {
        tag = 999;

        /* find data size to send block data */
        if (F_Snd_Num[IDS] != 0)
        {

          size1 = 0;
          for (k = 0; k < 2; k++)
          {
            for (n = 0; n < F_Snd_Num[IDS]; n++)
            {
              Mc_AN = Snd_MAN[IDS][n];
              Gc_AN = Snd_GAN[IDS][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                size1 += tno1 * tno2;
              }
            }
          }

          Snd_DM0_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else
        {
          Snd_DM0_Size[IDS] = 0;
        }

        /* receiving of size of data */

        if (F_Rcv_Num[IDR] != 0)
        {
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_DM0_Size[IDR] = size2;
        }
        else
        {
          Rcv_DM0_Size[IDR] = 0;
        }

        if (F_Snd_Num[IDS] != 0)
          MPI_Wait(&request, &stat);
      }
    }

    /***********************************
             data transfer
    ************************************/

    tag = 999;
    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {

        /*****************************
                    sending of data
        *****************************/

        if (F_Snd_Num[IDS] != 0)
        {

          size1 = Snd_DM0_Size[IDS];

          /* allocation of array */

          tmp_array = (double *)malloc(sizeof(double) * size1);

          /* multidimentional array to vector array */

          num = 0;
          for (k = 0; k < 2; k++)
          {
            for (n = 0; n < F_Snd_Num[IDS]; n++)
            {
              Mc_AN = Snd_MAN[IDS][n];
              Gc_AN = Snd_GAN[IDS][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];
              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                for (i = 0; i < tno1; i++)
                {
                  for (j = 0; j < tno2; j++)
                  {
                    tmp_array[num] = iDM0[k][Mc_AN][h_AN][i][j];
                    num++;
                  }
                }
              }
            }
          }

          MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
        }

        /*****************************
               receiving of block data
        *****************************/

        if (F_Rcv_Num[IDR] != 0)
        {

          size2 = Rcv_DM0_Size[IDR];

          /* allocation of array */
          tmp_array2 = (double *)malloc(sizeof(double) * size2);

          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          for (k = 0; k < 2; k++)
          {
            Mc_AN = F_TopMAN[IDR] - 1;
            for (n = 0; n < F_Rcv_Num[IDR]; n++)
            {
              Mc_AN++;
              Gc_AN = Rcv_GAN[IDR][n];
              Cwan = WhatSpecies[Gc_AN];
              tno1 = Spe_Total_NO[Cwan];

              for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno2 = Spe_Total_NO[Hwan];
                for (i = 0; i < tno1; i++)
                {
                  for (j = 0; j < tno2; j++)
                  {
                    iDM0[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
                    num++;
                  }
                }
              }
            }
          }

          /* freeing of array */
          free(tmp_array2);
        }

        if (F_Snd_Num[IDS] != 0)
        {
          MPI_Wait(&request, &stat);
          free(tmp_array); /* freeing of array */
        }
      }
    }

    /****************************************************
                     calculate NC_OcpN
    ****************************************************/

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      for (m = 0; m < Spe_Total_NO[wan1]; m++)
      {
        for (n = 0; n < Spe_Total_NO[wan1]; n++)
        {

          ReOcn00 = 0.0;
          ReOcn11 = 0.0;
          ReOcn01 = 0.0;

          ImOcn00 = 0.0;
          ImOcn11 = 0.0;
          ImOcn01 = 0.0;

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Mh_AN = F_G2M[Gh_AN];
            wan2 = WhatSpecies[Gh_AN];
            kl = RMI1[Mc_AN][h_AN][0];

            for (k = 0; k < Spe_Total_NO[wan2]; k++)
            {

              ReOcn00 += 0.5 * (DM0[0][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k] + DM0[0][Mh_AN][kl][k][n] * OLP[0][Mc_AN][h_AN][m][k]);

              ReOcn11 += 0.5 * (DM0[1][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k] + DM0[1][Mh_AN][kl][k][n] * OLP[0][Mc_AN][h_AN][m][k]);

              ReOcn01 += 0.5 * (DM0[2][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k] + DM0[2][Mh_AN][kl][k][n] * OLP[0][Mc_AN][h_AN][m][k]);

              ImOcn00 += 0.5 * (iDM0[0][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k] + iDM0[0][Mh_AN][kl][k][n] * OLP[0][Mc_AN][h_AN][m][k]);

              ImOcn11 += 0.5 * (iDM0[1][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k] + iDM0[1][Mh_AN][kl][k][n] * OLP[0][Mc_AN][h_AN][m][k]);

              ImOcn01 += 0.5 * (DM0[3][Mc_AN][h_AN][m][k] * OLP[0][Mc_AN][h_AN][n][k] + DM0[3][Mh_AN][kl][k][n] * OLP[0][Mc_AN][h_AN][m][k]);
            }
          }

          NC_OcpN[0][0][0][Mc_AN][m][n].r = ReOcn00;
          NC_OcpN[0][1][1][Mc_AN][m][n].r = ReOcn11;
          NC_OcpN[0][0][1][Mc_AN][m][n].r = ReOcn01;
          NC_OcpN[0][1][0][Mc_AN][n][m].r = ReOcn01;

          NC_OcpN[0][0][0][Mc_AN][m][n].i = ImOcn00;
          NC_OcpN[0][1][1][Mc_AN][m][n].i = ImOcn11;
          NC_OcpN[0][0][1][Mc_AN][m][n].i = ImOcn01;
          NC_OcpN[0][1][0][Mc_AN][n][m].i = -ImOcn01;
        }
      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      /*
      printf("Re 00 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][0][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 11 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][1][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 01 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][1][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Re 10 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][0][Mc_AN][i][j].r);
  }
        printf("\n");
      }

      printf("Im 00 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][0][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 11 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][1][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 01 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][0][1][Mc_AN][i][j].i);
  }
        printf("\n");
      }

      printf("Im 10 Gc_AN=%2d\n",Gc_AN);
      for (i=0; i<Spe_Total_NO[wan1]; i++){
  for (j=0; j<Spe_Total_NO[wan1]; j++){
          printf("%8.4f ",NC_OcpN[0][1][0][Mc_AN][i][j].i);
  }
        printf("\n");
      }
      */

      /*
      for (spin=0; spin<=SpinP_switch; spin++){
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      printf("DM spin=%2d Mc_AN=%2d h_AN=%2d\n",spin,Mc_AN,h_AN);
      for (m=0; m<Spe_Total_NO[wan1]; m++){
        for (n=0; n<Spe_Total_NO[wan1]; n++){
    printf("%8.4f ",DM[0][spin][Mc_AN][h_AN][m][n]);
        }
        printf("\n");
      }
    }
  }
      }
      */

    } /* Mc_AN */

    /****************************************************
                      freeing of arrays:
    ****************************************************/

    /* DM0 */

    for (k = 0; k <= SpinP_switch; k++)
    {
      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          if (Mc_AN == 0)
          {
            tno1 = 1;
          }
          else
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          for (i = 0; i < tno0; i++)
          {
            free(DM0[k][Mc_AN][h_AN][i]);
          }
          free(DM0[k][Mc_AN][h_AN]);
        }
        free(DM0[k][Mc_AN]);
      }
      free(DM0[k]);
    }
    free(DM0);

    /* Snd_DM0_Size and Rcv_DM0_Size */

    free(Snd_DM0_Size);
    free(Rcv_DM0_Size);

    /* iDM0 */

    for (k = 0; k < 2; k++)
    {

      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          if (Mc_AN == 0)
          {
            tno1 = 1;
          }
          else
          {
            Gh_AN = natn[Gc_AN][h_AN];
            Hwan = WhatSpecies[Gh_AN];
            tno1 = Spe_Total_NO[Hwan];
          }

          for (i = 0; i < tno0; i++)
          {
            free(iDM0[k][Mc_AN][h_AN][i]);
          }
          free(iDM0[k][Mc_AN][h_AN]);
        }
        free(iDM0[k][Mc_AN]);
      }
      free(iDM0[k]);
    }
    free(iDM0);
  }

  /****************************************************
  freeing of arrays:

  int Cnt_index1[List_YOUSO[25]+1]
                       [List_YOUSO[24]]
                       [2*(List_YOUSO[25]+1)+1];

  int Cnt_index2[List_YOUSO[25]+1]
                       [List_YOUSO[24]]
                       [2*(List_YOUSO[25]+1)+1];

  double DecMulP[SpinP_switch+1][List_YOUSO[7]];
  double Primitive_DM[SpinP_switch+1]
                            [List_YOUSO[7]]
                            [List_YOUSO[7]];
  ****************************************************/

  /* Cnt_index1 and Cnt_index2 */

  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      free(Cnt_index1[i][j]);
    }
    free(Cnt_index1[i]);
  }
  free(Cnt_index1);

  for (i = 0; i < (List_YOUSO[25] + 1); i++)
  {
    for (j = 0; j < List_YOUSO[24]; j++)
    {
      free(Cnt_index2[i][j]);
    }
    free(Cnt_index2[i]);
  }
  free(Cnt_index2);

  /* DecMulP */

  for (spin = 0; spin <= SpinP_switch; spin++)
  {
    free(DecMulP[spin]);
  }
  free(DecMulP);

  /* Primitive_DM */

  for (spin = 0; spin <= SpinP_switch; spin++)
  {
    for (i = 0; i < List_YOUSO[7]; i++)
    {
      free(Primitive_DM[spin][i]);
    }
    free(Primitive_DM[spin]);
  }
  free(Primitive_DM);
}

void Induce_Orbital_Polarization(int Mc_AN)
{
  int wan1, Gc_AN, spin, spinmax;
  int i, j, to1, to2, l1, mul1, m0, m1, m2, m3;
  int ***trans_index;
  double **a, *ko;
  double sum1, tmp1, tmp2;
  double toccpn[5];
  double Ncut = 0.3;
  int Ns;

  Gc_AN = M2G[Mc_AN];
  wan1 = WhatSpecies[Gc_AN];

  if (SpinP_switch == 0)
    spinmax = 0;
  else
    spinmax = 1;

  /* allocation of arrays */

  trans_index = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (l1 = 0; l1 < (List_YOUSO[25] + 1); l1++)
  {
    trans_index[l1] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (mul1 = 0; mul1 < List_YOUSO[24]; mul1++)
    {
      trans_index[l1][mul1] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  Ns = 20;

  a = (double **)malloc(sizeof(double *) * Ns);
  for (i = 0; i < Ns; i++)
  {
    a[i] = (double *)malloc(sizeof(double) * Ns);
  }

  ko = (double *)malloc(sizeof(double) * Ns);

  /* set trans_index */

  to1 = 0;
  for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
  {
    for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
    {
      for (m1 = 0; m1 < (2 * l1 + 1); m1++)
      {
        trans_index[l1][mul1][m1] = to1;
        to1++;
      }
    }
  }

  /*****************************************
          induce orbital polaization
  *****************************************/

  if (OrbPol_flag[Gc_AN] != 0)
  {

    mul1 = 0;

    for (l1 = 2; l1 <= Spe_MaxL_Basis[wan1]; l1++)
    {
      for (spin = 0; spin <= spinmax; spin++)
      {

        for (m1 = 0; m1 < (2 * l1 + 1); m1++)
        {
          for (m2 = 0; m2 < (2 * l1 + 1); m2++)
          {
            to1 = trans_index[l1][mul1][m1];
            to2 = trans_index[l1][mul1][m2];
            a[m1 + 1][m2 + 1] = DM[0][spin][Mc_AN][0][to1][to2];
          }
        }

        sum1 = 0.0;
        for (m1 = 0; m1 < (2 * l1 + 1); m1++)
        {
          sum1 += a[m1 + 1][m1 + 1];
        }

        if (Ncut < sum1)
        {

          Eigen_lapack(a, ko, 2 * l1 + 1, 2 * l1 + 1);

          toccpn[spin] = 0.0;

          for (m1 = 0; m1 < (2 * l1 + 1); m1++)
          {
            toccpn[spin] += ko[m1 + 1];
          }

          for (m1 = 0; m1 < (2 * l1 + 1); m1++)
          {
            ko[m1 + 1] = 0.0;
          }

          /* normal orbital polarization */

          if (OrbPol_flag[Gc_AN] == 1)
          {

            m0 = 2 * l1 + 1 - (int)toccpn[spin];
            if (m0 < 0)
              m0 = 0;

            for (m1 = 2 * l1; m0 <= m1; m1--)
            {
              ko[m1 + 1] = 1.0;
            }

            if (0 <= (2 * l1 - (int)toccpn[spin]))
            {
              ko[2 * l1 - (int)toccpn[spin] + 1] = (double)(toccpn[spin] - (int)toccpn[spin]);
            }
          }

          /* orbital polarization for the first exited state */

          else if (OrbPol_flag[Gc_AN] == 2)
          {

            m0 = 2 * l1 + 1 - (int)toccpn[spin];
            if (m0 < 0)
              m0 = 0;

            for (m1 = 2 * l1; m0 <= m1; m1--)
            {
              ko[m1 + 1] = 1.0;
            }

            if (0 <= (2 * l1 - (int)toccpn[spin]))
            {
              ko[2 * l1 - (int)toccpn[spin] + 1] = (double)(toccpn[spin] - (int)toccpn[spin]);
            }

            m1 = 2 * l1 - (int)toccpn[spin] + 1;
            m2 = 2 * l1 - (int)toccpn[spin] + 2;

            if (1 <= m1 && m1 <= (2 * l1 + 1) && 1 <= m2 && m2 <= (2 * l1 + 1))
            {
              tmp1 = ko[m1];
              tmp2 = ko[m2];
              ko[m1] = tmp2;
              ko[m2] = tmp1;
            }
          }

          /*
          for (m1=0; m1<(2*l1+1); m1++){
            printf("Y1 Gc_AN=%2d spin=%2d %15.12f %15.12f\n",Gc_AN,spin,toccpn[spin],ko[m1+1]);
                }
          */

          /* a * ko * a^+ */

          for (m1 = 0; m1 < (2 * l1 + 1); m1++)
          {
            for (m2 = 0; m2 < (2 * l1 + 1); m2++)
            {

              sum1 = 0.0;
              for (m3 = 0; m3 < (2 * l1 + 1); m3++)
              {
                sum1 += a[m1 + 1][m3 + 1] * ko[m3 + 1] * a[m2 + 1][m3 + 1];
              }

              to1 = trans_index[l1][mul1][m1];
              to2 = trans_index[l1][mul1][m2];

              DM_onsite[0][spin][Mc_AN][to1][to2] = sum1;
            }
          }
        }
      }
    }
  }

  /* freeing of arrays */

  for (l1 = 0; l1 < (List_YOUSO[25] + 1); l1++)
  {
    for (mul1 = 0; mul1 < List_YOUSO[24]; mul1++)
    {
      free(trans_index[l1][mul1]);
    }
    free(trans_index[l1]);
  }
  free(trans_index);

  for (i = 0; i < Ns; i++)
  {
    free(a[i]);
  }
  free(a);

  free(ko);
}

void Induce_Orbital_Polarization_Together(int Mc_AN)
{
  int wan1, Gc_AN, spin, spinmax;
  int i, j, k, to1, to2, l1, mul1, m0, m1, m2, m3;
  int ***trans_index;
  double **a, *ko;
  double sum1, tmp1, tmp2;
  double toccpn;
  double Ncut = 0.3;
  int Ns;

  Gc_AN = M2G[Mc_AN];
  wan1 = WhatSpecies[Gc_AN];

  if (SpinP_switch == 0)
    spinmax = 0;
  else
    spinmax = 1;

  /* allocation of arrays */

  trans_index = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (l1 = 0; l1 < (List_YOUSO[25] + 1); l1++)
  {
    trans_index[l1] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (mul1 = 0; mul1 < List_YOUSO[24]; mul1++)
    {
      trans_index[l1][mul1] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  Ns = 4 * 4 * 2 + 1;

  a = (double **)malloc(sizeof(double *) * Ns);
  for (i = 0; i < Ns; i++)
  {
    a[i] = (double *)malloc(sizeof(double) * Ns);
    for (j = 0; j < Ns; j++)
      a[i][j] = 0.0;
  }

  ko = (double *)malloc(sizeof(double) * Ns);

  /* set trans_index */

  to1 = 0;
  for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
  {
    for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
    {
      for (m1 = 0; m1 < (2 * l1 + 1); m1++)
      {
        trans_index[l1][mul1][m1] = to1;
        to1++;
      }
    }
  }

  /*****************************************
          induce orbital polaization
  *****************************************/

  if (OrbPol_flag[Gc_AN] != 0)
  {

    mul1 = 0;

    for (l1 = 2; l1 <= Spe_MaxL_Basis[wan1]; l1++)
    {

      k = 2 * l1 + 1;

      for (m1 = 0; m1 < (2 * l1 + 1); m1++)
      {
        for (m2 = 0; m2 < (2 * l1 + 1); m2++)
        {
          to1 = trans_index[l1][mul1][m1];
          to2 = trans_index[l1][mul1][m2];

          /* rnd(1.0e-13) is a prescription to stabilize the lapack routines */
          a[m1 + 1][m2 + 1] = DM[0][0][Mc_AN][0][to1][to2] + rnd(1.0e-13);
          a[m1 + k + 1][m2 + k + 1] = DM[0][1][Mc_AN][0][to1][to2] + rnd(1.0e-13);
          a[m1 + 1][m2 + k + 1] = rnd(1.0e-13);
          a[m1 + k + 1][m2 + 1] = rnd(1.0e-13);
        }
      }

      sum1 = 0.0;
      for (m1 = 0; m1 < 2 * k; m1++)
      {
        sum1 += a[m1 + 1][m1 + 1];
      }

      if (Ncut < sum1)
      {

        Eigen_lapack(a, ko, 2 * k, 2 * k);

        /*
              printf("Col Gc_AN=%2d\n",Gc_AN);fflush(stdout);
              for (m1=0; m1<2*k; m1++){
          printf("Col m1=%2d %15.12f\n",m1,ko[m1+1]);fflush(stdout);
        }
        */

        toccpn = 0.0;
        for (m1 = 0; m1 < 2 * k; m1++)
        {
          toccpn += ko[m1 + 1];
        }

        for (m1 = 0; m1 < 2 * k; m1++)
        {
          ko[m1 + 1] = 0.0;
        }

        /* normal orbital polarization */

        if (OrbPol_flag[Gc_AN] == 1)
        {

          m0 = 4 * l1 + 2 - (int)toccpn;
          if (m0 < 0)
            m0 = 0;

          for (m1 = (4 * l1 + 1); m0 <= m1; m1--)
          {
            ko[m1 + 1] = 1.0;
          }

          if (0 <= (4 * l1 + 1 - (int)toccpn))
          {
            ko[4 * l1 - (int)toccpn + 1] = (double)(toccpn - (int)toccpn);
          }
        }

        /* orbital polarization for the first exited state */

        else if (OrbPol_flag[Gc_AN] == 2)
        {
          /* not supported */
        }

        /* a * ko * a^+ */

        for (m1 = 0; m1 < 2 * k; m1++)
        {
          for (m2 = 0; m2 < 2 * k; m2++)
          {

            sum1 = 0.0;
            for (m3 = 0; m3 < 2 * k; m3++)
            {
              sum1 += a[m1 + 1][m3 + 1] * ko[m3 + 1] * a[m2 + 1][m3 + 1];
            }

            to1 = trans_index[l1][mul1][m1 % k];
            to2 = trans_index[l1][mul1][m2 % k];

            if ((m1 / k) == 0 && (m2 / k) == 0)
            {
              DM_onsite[0][0][Mc_AN][to1][to2] = sum1;
            }
            else if ((m1 / k) == 1 && (m2 / k) == 1)
            {
              DM_onsite[0][1][Mc_AN][to1][to2] = sum1;
            }
          }
        }
      }
    }
  }

  /* freeing of arrays */

  for (l1 = 0; l1 < (List_YOUSO[25] + 1); l1++)
  {
    for (mul1 = 0; mul1 < List_YOUSO[24]; mul1++)
    {
      free(trans_index[l1][mul1]);
    }
    free(trans_index[l1]);
  }
  free(trans_index);

  for (i = 0; i < Ns; i++)
  {
    free(a[i]);
  }
  free(a);

  free(ko);
}

void Induce_NC_Orbital_Polarization(int Mc_AN)
{
  int wan1, Gc_AN;
  int i, j, k, to1, to2, l1, mul1, m0, m1, m2, m3;
  int ***trans_index;
  double *ko;
  dcomplex **a, sum1;
  double tmp1, tmp2;
  double toccpn;
  double Ncut = 0.3;
  int Ns;

  Gc_AN = M2G[Mc_AN];
  wan1 = WhatSpecies[Gc_AN];

  /* allocation of arrays */

  trans_index = (int ***)malloc(sizeof(int **) * (List_YOUSO[25] + 1));
  for (l1 = 0; l1 < (List_YOUSO[25] + 1); l1++)
  {
    trans_index[l1] = (int **)malloc(sizeof(int *) * List_YOUSO[24]);
    for (mul1 = 0; mul1 < List_YOUSO[24]; mul1++)
    {
      trans_index[l1][mul1] = (int *)malloc(sizeof(int) * (2 * (List_YOUSO[25] + 1) + 1));
    }
  }

  Ns = 4 * 4 * 2 + 1;

  a = (dcomplex **)malloc(sizeof(dcomplex *) * Ns);
  for (i = 0; i < Ns; i++)
  {
    a[i] = (dcomplex *)malloc(sizeof(dcomplex) * Ns);
  }

  ko = (double *)malloc(sizeof(double) * Ns);

  /* set trans_index */

  to1 = 0;
  for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
  {
    for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
    {
      for (m1 = 0; m1 < (2 * l1 + 1); m1++)
      {
        trans_index[l1][mul1][m1] = to1;
        to1++;
      }
    }
  }

  /*****************************************
          induce orbital polaization
  *****************************************/

  if (OrbPol_flag[Gc_AN] != 0)
  {

    mul1 = 0;

    for (l1 = 2; l1 <= Spe_MaxL_Basis[wan1]; l1++)
    {

      k = 2 * l1 + 1;

      for (m1 = 0; m1 < k; m1++)
      {
        for (m2 = 0; m2 < k; m2++)
        {
          to1 = trans_index[l1][mul1][m1];
          to2 = trans_index[l1][mul1][m2];

          /* rnd(1.0e-12) is a prescription to stabilize the lapack routines */
          a[m1 + 1][m2 + 1].r = DM[0][0][Mc_AN][0][to1][to2] + rnd(1.0e-12);
          a[m1 + k + 1][m2 + k + 1].r = DM[0][1][Mc_AN][0][to1][to2] + rnd(1.0e-12);
          a[m1 + 1][m2 + k + 1].r = DM[0][2][Mc_AN][0][to1][to2] + rnd(1.0e-12);
          a[m1 + k + 1][m2 + 1].r = DM[0][2][Mc_AN][0][to2][to1] + rnd(1.0e-12);

          a[m1 + 1][m2 + 1].i = iDM[0][0][Mc_AN][0][to1][to2] + rnd(1.0e-12);
          a[m1 + k + 1][m2 + k + 1].i = iDM[0][1][Mc_AN][0][to1][to2] + rnd(1.0e-12);
          a[m1 + 1][m2 + k + 1].i = DM[0][3][Mc_AN][0][to1][to2] + rnd(1.0e-12);
          a[m1 + k + 1][m2 + 1].i = -DM[0][3][Mc_AN][0][to2][to1] + rnd(1.0e-12);
        }
      }

      tmp1 = 0.0;
      for (m1 = 0; m1 < 2 * k; m1++)
      {
        tmp1 += a[m1 + 1][m1 + 1].r;
      }

      if (Ncut < tmp1)
      {

        EigenBand_lapack(a, ko, 2 * k, 2 * k, 1);

        /*
              printf("NCl Gc_AN=%2d\n",Gc_AN);fflush(stdout);
        for (m1=0; m1<2*k; m1++){
          printf("NCl m1=%2d %15.12f\n",m1,ko[m1+1]);fflush(stdout);
        }
        */

        toccpn = 0.0;
        for (m1 = 0; m1 < 2 * k; m1++)
        {
          toccpn += ko[m1 + 1];
        }

        for (m1 = 0; m1 < 2 * k; m1++)
        {
          ko[m1 + 1] = 0.0;
        }

        /* normal orbital polarization */

        if (OrbPol_flag[Gc_AN] == 1)
        {

          m0 = 4 * l1 + 2 - (int)toccpn;
          if (m0 < 0)
            m0 = 0;

          for (m1 = (4 * l1 + 1); m0 <= m1; m1--)
          {
            ko[m1 + 1] = 1.0;
          }

          if (0 <= (4 * l1 + 1 - (int)toccpn))
          {
            ko[4 * l1 + 1 - (int)toccpn + 1] = (double)(toccpn - (int)toccpn);
          }
        }

        /* orbital polarization for the first exited state */

        else if (OrbPol_flag[Gc_AN] == 2)
        {
          /* not supported */
        }

        /* a * ko * a^+ */

        for (m1 = 0; m1 < 2 * k; m1++)
        {
          for (m2 = 0; m2 < 2 * k; m2++)
          {

            sum1.r = 0.0;
            sum1.i = 0.0;

            for (m3 = 0; m3 < 2 * k; m3++)
            {
              sum1.r += (a[m1 + 1][m3 + 1].r * a[m2 + 1][m3 + 1].r + a[m1 + 1][m3 + 1].i * a[m2 + 1][m3 + 1].i) * ko[m3 + 1];
              sum1.i += (-a[m1 + 1][m3 + 1].r * a[m2 + 1][m3 + 1].i + a[m1 + 1][m3 + 1].i * a[m2 + 1][m3 + 1].r) * ko[m3 + 1];
            }

            to1 = trans_index[l1][mul1][m1 % k];
            to2 = trans_index[l1][mul1][m2 % k];

            NC_OcpN[0][m1 / k][m2 / k][Mc_AN][to2][to1].r = sum1.r;
            NC_OcpN[0][m1 / k][m2 / k][Mc_AN][to2][to1].i = sum1.i;
          }
        }
      }
    }
  }

  /* freeing of arrays */

  for (l1 = 0; l1 < (List_YOUSO[25] + 1); l1++)
  {
    for (mul1 = 0; mul1 < List_YOUSO[24]; mul1++)
    {
      free(trans_index[l1][mul1]);
    }
    free(trans_index[l1]);
  }
  free(trans_index);

  for (i = 0; i < Ns; i++)
  {
    free(a[i]);
  }
  free(a);

  free(ko);
}

void make_v_eff(int SCF_iter, int SucceedReadingDMfile, double dUele)
{
  int i, j, k, tno0, tno1, tno2, Cwan, Gc_AN, spin;
  int Mc_AN, num, n, size1, size2, to1, to2;
  int l1, l2, m1, m2, mul1, mul2, wan1;
  int *Snd_Size, *Rcv_Size;
  int numprocs, myid, ID, IDS, IDR, tag = 999;
  double *tmp_array;
  double *tmp_array2;
  double *tmp_array0;
  double Stime_atom, Etime_atom;
  double Uvalue;

  /* added by S.Ryee */
  double Jvalue, trace_spin, trace_opp_spin;
  int on_off_dc, on_off, mul_index1, mul_index2, tmp_l, to_start, ii, jj, kk, ll, dd;

  int NZUJ;
  /*******************/

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* allocations of arrays */

  Snd_Size = (int *)malloc(sizeof(int) * numprocs);
  Rcv_Size = (int *)malloc(sizeof(int) * numprocs);

  /****************************************************

                       make v_eff

      Important note:

      In case of orbital optimization,
      the U potential is applied to
      the primitive orbital.
  ****************************************************/

  switch (Hub_Type)
  {
  case 1: /* Dudarev form */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      for (spin = 0; spin <= SpinP_switch; spin++)
      {

        /* store the v_eff --- MJ */

        to1 = 0;
        for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
        {
          for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
          {
            for (m1 = 0; m1 < (2 * l1 + 1); m1++)
            {

              to2 = 0;
              for (l2 = 0; l2 <= Spe_MaxL_Basis[wan1]; l2++)
              {
                for (mul2 = 0; mul2 < Spe_Num_Basis[wan1][l2]; mul2++)
                {
                  for (m2 = 0; m2 < (2 * l2 + 1); m2++)
                  {

                    if (l1 == l2 && mul1 == mul2)
                      Uvalue = Hub_U_Basis[wan1][l1][mul1];
                    else
                      Uvalue = 0.0;

                    if (to1 == to2)
                    {
                      v_eff[spin][Mc_AN][to1][to2] = Uvalue * (0.5 - DM_onsite[0][spin][Mc_AN][to1][to2]);
                    }
                    else
                    {
                      v_eff[spin][Mc_AN][to1][to2] = Uvalue * (0.0 - DM_onsite[0][spin][Mc_AN][to1][to2]);
                    }

                    to2++;

                  } /* mul2 */
                }   /* m2 */
              }     /* l2 */

              to1++;

            } /* mul1 */
          }   /* m1 */
        }     /* l1 */
      }       /* spin */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */
    break;

  case 2: /* General form by S.Ryee */

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      /* store the v_eff */

      to1 = 0;
      on_off_dc = 1;
      mul_index1 = 0;

      for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
      {
        for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
        {
          Uvalue = Hub_U_Basis[wan1][l1][mul1];
          Jvalue = Hund_J_Basis[wan1][l1][mul1];
          NZUJ = Nonzero_UJ[wan1][l1][mul1];
          for (m1 = 0; m1 < (2 * l1 + 1); m1++)
          {
            to2 = 0;
            mul_index2 = 0;
            for (l2 = 0; l2 <= Spe_MaxL_Basis[wan1]; l2++)
            {
              for (mul2 = 0; mul2 < Spe_Num_Basis[wan1][l2]; mul2++)
              {
                for (m2 = 0; m2 < (2 * l2 + 1); m2++)
                {

                  if ((mul_index1 == mul_index2) && (NZUJ > 0))
                  {
                    to_start = 0;

                    switch (mul1)
                    {
                    case 0: /* mul1 = 0 */
                      if (l1 > 0)
                      {
                        for (tmp_l = 0; tmp_l < l1; tmp_l++)
                        {
                          to_start += (2 * tmp_l + 1) * Spe_Num_Basis[wan1][tmp_l];
                        }
                      }
                      else
                      { /* l1 = 0 */
                        to_start = 0;
                      }
                      break;
                    case 1: /* mul1 = 1 */
                      if (l1 > 0)
                      {
                        for (tmp_l = 0; tmp_l < l1; tmp_l++)
                        {
                          to_start += (2 * tmp_l + 1) * Spe_Num_Basis[wan1][tmp_l];
                        }
                        to_start += (2 * l1 + 1) * mul1;
                      }
                      else
                      { /* l1 = 0 */
                        to_start = (2 * l1 + 1) * mul1;
                      }
                      break;
                    } /* switch (mul1) */

                    ii = to1 - to_start;
                    kk = to2 - to_start;
                    v_eff[0][Mc_AN][to1][to2] = 0.0;
                    v_eff[1][Mc_AN][to1][to2] = 0.0;

                    if (to1 == to2)
                    { /* double counting correction */
                      trace_spin = 0.0;
                      trace_opp_spin = 0.0;
                      for (dd = 0; dd < (2 * l1 + 1); dd++)
                      {
                        trace_spin += DM_onsite[0][0][Mc_AN][to_start + dd][to_start + dd];
                        trace_opp_spin += DM_onsite[0][1][Mc_AN][to_start + dd][to_start + dd];
                      }

                      switch (dc_Type)
                      {
                      case 1: /* sFLL */
                        v_eff[0][Mc_AN][to1][to2] -= Uvalue * (trace_spin + trace_opp_spin - 0.5) - Jvalue * (trace_spin - 0.5);
                        v_eff[1][Mc_AN][to1][to2] -= Uvalue * (trace_spin + trace_opp_spin - 0.5) - Jvalue * (trace_opp_spin - 0.5);
                        break;

                      case 2: /* sAMF */
                        if (on_off_dc == 1)
                        {
                          for (jj = 0; jj < (2 * l1 + 1); jj++)
                          {
                            for (ll = 0; ll < (2 * l1 + 1); ll++)
                            {
                              if (jj == ll)
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll] - trace_spin / ((double)(2 * l1 + 1));
                                AMF_Array[NZUJ][1][0][jj][ll] = DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll] - trace_opp_spin / ((double)(2 * l1 + 1));
                              }
                              else
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll];
                                AMF_Array[NZUJ][1][0][jj][ll] = DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll];
                              }
                            }
                          }
                          on_off_dc = 0;
                        }
                        break;

                      case 3: /* cFLL */
                        v_eff[0][Mc_AN][to1][to2] -= Uvalue * (trace_spin + trace_opp_spin - 0.5) - Jvalue * 0.5 * (trace_spin + trace_opp_spin - 1.0);
                        v_eff[1][Mc_AN][to1][to2] -= Uvalue * (trace_spin + trace_opp_spin - 0.5) - Jvalue * 0.5 * (trace_spin + trace_opp_spin - 1.0);
                        break;

                      case 4: /* cAMF */
                        if (on_off_dc == 1)
                        {
                          for (jj = 0; jj < (2 * l1 + 1); jj++)
                          {
                            for (ll = 0; ll < (2 * l1 + 1); ll++)
                            {
                              if (jj == ll)
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll] - (trace_spin + trace_opp_spin) / (2.0 * ((double)(2 * l1 + 1)));
                                AMF_Array[NZUJ][1][0][jj][ll] = DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll] - (trace_spin + trace_opp_spin) / (2.0 * ((double)(2 * l1 + 1)));
                              }
                              else
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll];
                                AMF_Array[NZUJ][1][0][jj][ll] = DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll];
                              }
                            }
                          }
                          on_off_dc = 0;
                        }
                        break;
                      }

                    } /* double counting correction */

                    /* loop start for Hartree-Fock-like interaction potential */
                    for (jj = 0; jj < (2 * l1 + 1); jj++)
                    {
                      for (ll = 0; ll < (2 * l1 + 1); ll++)
                      {
                        switch (dc_Type)
                        {
                        case 1: /* sFLL */
                          v_eff[0][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll] + DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll];
                          v_eff[1][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll] + DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll];
                          break;

                        case 2: /* sAMF */
                          v_eff[0][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (AMF_Array[NZUJ][1][0][jj][ll] + AMF_Array[NZUJ][0][0][jj][ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           AMF_Array[NZUJ][0][0][jj][ll];
                          v_eff[1][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (AMF_Array[NZUJ][0][0][jj][ll] + AMF_Array[NZUJ][1][0][jj][ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           AMF_Array[NZUJ][1][0][jj][ll];
                          break;

                        case 3: /* cFLL */
                          v_eff[0][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll] + DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll];
                          v_eff[1][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (DM_onsite[0][0][Mc_AN][to_start + jj][to_start + ll] + DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           DM_onsite[0][1][Mc_AN][to_start + jj][to_start + ll];
                          break;

                        case 4: /* cAMF */
                          v_eff[0][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (AMF_Array[NZUJ][1][0][jj][ll] + AMF_Array[NZUJ][0][0][jj][ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           AMF_Array[NZUJ][0][0][jj][ll];
                          v_eff[1][Mc_AN][to1][to2] += Coulomb_Array[NZUJ][ii][jj][kk][ll] *
                                                           (AMF_Array[NZUJ][0][0][jj][ll] + AMF_Array[NZUJ][1][0][jj][ll]) -
                                                       Coulomb_Array[NZUJ][ii][jj][ll][kk] *
                                                           AMF_Array[NZUJ][1][0][jj][ll];
                          break;
                        } /* dc_Type switch */

                      } /* ll */
                    }   /* jj */

                  } /* mul_index1 == mul_index2 */
                  else
                  {
                    v_eff[0][Mc_AN][to1][to2] = 0.0;
                    v_eff[1][Mc_AN][to1][to2] = 0.0;
                  }

                  to2++;

                } /* m2 */
                mul_index2++;
              } /* mul2 */
            }   /* l2 */

            to1++;

          } /* m1 */
          mul_index1++;
          on_off_dc = 1;
        } /* mul1 */
      }   /* l1 */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

      /*   printf("%f\n", dc_alpha[count]);*/
    } /* Mc_AN */
    break;
  } /* switch (Hub_Type) */

  /****************************************************
    MPI: v_eff
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID = 0; ID < numprocs; ID++)
  {

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID != 0)
    {
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS] != 0)
      {

        size1 = 0;
        for (k = 0; k <= SpinP_switch; k++)
        {
          for (n = 0; n < F_Snd_Num[IDS]; n++)
          {
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_NO[Cwan];
            size1 += tno1 * tno1;
          }
        }

        Snd_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else
      {
        Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR] != 0)
      {
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_Size[IDR] = size2;
      }
      else
      {
        Rcv_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS] != 0)
        MPI_Wait(&request, &stat);
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID = 0; ID < numprocs; ID++)
  {

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID != 0)
    {

      /*****************************
              sending of data
      *****************************/

      if (F_Snd_Num[IDS] != 0)
      {

        size1 = Snd_Size[IDS];

        /* allocation of array */

        tmp_array = (double *)malloc(sizeof(double) * size1);

        /* multidimentional array to vector array */

        num = 0;
        for (k = 0; k <= SpinP_switch; k++)
        {
          for (n = 0; n < F_Snd_Num[IDS]; n++)
          {
            Mc_AN = Snd_MAN[IDS][n];
            Gc_AN = Snd_GAN[IDS][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_NO[Cwan];
            for (i = 0; i < tno1; i++)
            {
              for (j = 0; j < tno1; j++)
              {
                tmp_array[num] = v_eff[k][Mc_AN][i][j];
                num++;
              }
            }
          }
        }

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR] != 0)
      {

        size2 = Rcv_Size[IDR];

        /* allocation of array */
        tmp_array2 = (double *)malloc(sizeof(double) * size2);

        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        for (k = 0; k <= SpinP_switch; k++)
        {
          Mc_AN = F_TopMAN[IDR] - 1;
          for (n = 0; n < F_Rcv_Num[IDR]; n++)
          {
            Mc_AN++;
            Gc_AN = Rcv_GAN[IDR][n];
            Cwan = WhatSpecies[Gc_AN];
            tno1 = Spe_Total_NO[Cwan];
            for (i = 0; i < tno1; i++)
            {
              for (j = 0; j < tno1; j++)
              {
                v_eff[k][Mc_AN][i][j] = tmp_array2[num];
                num++;
              }
            }
          }
        }

        /* freeing of array */
        free(tmp_array2);
      }

      if (F_Snd_Num[IDS] != 0)
      {
        MPI_Wait(&request, &stat);
        free(tmp_array); /* freeing of array */
      }
    }
  }

  /* freeing of Snd_Size and Rcv_Size */

  free(Snd_Size);
  free(Rcv_Size);
}

void make_NC_v_eff(int SCF_iter, int SucceedReadingDMfile, double dUele, double ECE[])
{
  int i, j, k, tno0, tno1, tno2, Cwan, Gc_AN, spin;
  int Mc_AN, num, n, size1, size2, to1, to2;
  int l1, l2, m1, m2, mul1, mul2, wan1, s1, s2, s3, s4;
  int *Snd_Size, *Rcv_Size;
  int numprocs, myid, ID, IDS, IDR, tag = 999;
  double *tmp_array;
  double *tmp_array2;
  double *tmp_array0;
  double Stime_atom, Etime_atom;
  double Uvalue, Nup[2], Ndn[2], theta[2], phi[2];
  double Nup0, Ndn0;
  double A, B, C, L, Lx, Ly, Lz, tmp, tmp1, tmp2;
  double sit, cot, sip, cop, Bx, By, Bz, lx, ly, lz, sx, sy, sz;
  double My_Ucs, My_Uzs, My_Uzo;
  double theta0, phi0;
  dcomplex TN[2][2], dTN[2][2][2][2], TN0[2][2], U[2][2];
  dcomplex dSx[2][2], dSy[2][2], dSz[2][2];
  dcomplex csum1, csum2, ctmp1, ctmp2;

  /* added by S.Ryee */
  double Jvalue, dum_alpha, dum_alpha2;
  int on_off_dc, mul_index1, mul_index2, tmp_l, to_start, ii, jj, kk, ll, dd;
  dcomplex trace_N00, trace_N11, trace_N01, trace_N10;

  int NZUJ;
  /*******************/

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* allocations of arrays */

  Snd_Size = (int *)malloc(sizeof(int) * numprocs);
  Rcv_Size = (int *)malloc(sizeof(int) * numprocs);

  /****************************************************
                    make NC_v_eff
  ****************************************************/

  My_Ucs = 0.0;
  My_Uzs = 0.0;
  My_Uzo = 0.0;

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
  {
    switch (Hub_Type)
    {
    case 1: /* Dudarev form */
      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      /************************************************************
       ***********************************************************
       ***********************************************************
       ***********************************************************

                     calculate the v_eff for LDA+U

       ***********************************************************
       ***********************************************************
       ***********************************************************
      ************************************************************/

      to1 = 0;
      for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
      {
        for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
        {
          for (m1 = 0; m1 < (2 * l1 + 1); m1++)
          {

            to2 = 0;
            for (l2 = 0; l2 <= Spe_MaxL_Basis[wan1]; l2++)
            {
              for (mul2 = 0; mul2 < Spe_Num_Basis[wan1][l2]; mul2++)
              {
                for (m2 = 0; m2 < (2 * l2 + 1); m2++)
                {

                  if (Hub_U_switch == 1)
                  {

                    if (l1 == l2 && mul1 == mul2)
                      Uvalue = Hub_U_Basis[wan1][l1][mul1];
                    else
                      Uvalue = 0.0;

                    if (to1 == to2)
                    {
                      NC_v_eff[0][0][Mc_AN][to1][to2].r = Uvalue * (0.5 - NC_OcpN[0][0][0][Mc_AN][to2][to1].r);
                      NC_v_eff[0][0][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][0][0][Mc_AN][to2][to1].i);
                      NC_v_eff[1][1][Mc_AN][to1][to2].r = Uvalue * (0.5 - NC_OcpN[0][1][1][Mc_AN][to2][to1].r);
                      NC_v_eff[1][1][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][1][1][Mc_AN][to2][to1].i);
                      NC_v_eff[0][1][Mc_AN][to1][to2].r = Uvalue * (0.0 - NC_OcpN[0][1][0][Mc_AN][to2][to1].r);
                      NC_v_eff[0][1][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][1][0][Mc_AN][to2][to1].i);
                      NC_v_eff[1][0][Mc_AN][to1][to2].r = Uvalue * (0.0 - NC_OcpN[0][0][1][Mc_AN][to2][to1].r);
                      NC_v_eff[1][0][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][0][1][Mc_AN][to2][to1].i);
                    }
                    else
                    {
                      NC_v_eff[0][0][Mc_AN][to1][to2].r = Uvalue * (0.0 - NC_OcpN[0][0][0][Mc_AN][to2][to1].r);
                      NC_v_eff[0][0][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][0][0][Mc_AN][to2][to1].i);
                      NC_v_eff[1][1][Mc_AN][to1][to2].r = Uvalue * (0.0 - NC_OcpN[0][1][1][Mc_AN][to2][to1].r);
                      NC_v_eff[1][1][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][1][1][Mc_AN][to2][to1].i);
                      NC_v_eff[0][1][Mc_AN][to1][to2].r = Uvalue * (0.0 - NC_OcpN[0][1][0][Mc_AN][to2][to1].r);
                      NC_v_eff[0][1][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][1][0][Mc_AN][to2][to1].i);
                      NC_v_eff[1][0][Mc_AN][to1][to2].r = Uvalue * (0.0 - NC_OcpN[0][0][1][Mc_AN][to2][to1].r);
                      NC_v_eff[1][0][Mc_AN][to1][to2].i = Uvalue * (0.0 - NC_OcpN[0][0][1][Mc_AN][to2][to1].i);
                    }

                  } /* if (Hub_U_switch==1) */

                  /* initialize NC_v_eff */

                  else
                  {
                    NC_v_eff[0][0][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[0][0][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[1][1][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[1][1][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[0][1][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[0][1][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[1][0][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[1][0][Mc_AN][to1][to2].i = 0.0;
                  }

                  to2++;

                } /* mul2 */
              }   /* m2 */
            }     /* l2 */

            to1++;

          } /* mul1 */
        }   /* m1 */
      }     /* l1 */
      break;

    case 2: /* General form by S.Ryee */
      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];

      to1 = 0;
      on_off_dc = 1;
      mul_index1 = 0;
      for (l1 = 0; l1 <= Spe_MaxL_Basis[wan1]; l1++)
      {
        for (mul1 = 0; mul1 < Spe_Num_Basis[wan1][l1]; mul1++)
        {
          Uvalue = Hub_U_Basis[wan1][l1][mul1];
          Jvalue = Hund_J_Basis[wan1][l1][mul1];
          NZUJ = Nonzero_UJ[wan1][l1][mul1];
          for (m1 = 0; m1 < (2 * l1 + 1); m1++)
          {
            to2 = 0;
            mul_index2 = 0;
            for (l2 = 0; l2 <= Spe_MaxL_Basis[wan1]; l2++)
            {
              for (mul2 = 0; mul2 < Spe_Num_Basis[wan1][l2]; mul2++)
              {
                for (m2 = 0; m2 < (2 * l2 + 1); m2++)
                {
                  if ((mul_index1 == mul_index2) && (NZUJ > 0))
                  {
                    to_start = 0;
                    switch (mul1)
                    {
                    case 0: /* mul1 = 0 */
                      if (l1 > 0)
                      {
                        for (tmp_l = 0; tmp_l < l1; tmp_l++)
                        {
                          to_start += (2 * tmp_l + 1) * Spe_Num_Basis[wan1][tmp_l];
                        }
                      }
                      else
                      { /* l1 = 0 */
                        to_start = 0;
                      }
                      break;

                    case 1: /* mul1 = 1 */
                      if (l1 > 0)
                      {
                        for (tmp_l = 0; tmp_l < l1; tmp_l++)
                        {
                          to_start += (2 * tmp_l + 1) * Spe_Num_Basis[wan1][tmp_l];
                        }
                        to_start += (2 * l1 + 1) * mul1;
                      }
                      else
                      { /* l1 = 0 */
                        to_start = (2 * l1 + 1) * mul1;
                      }
                      break;
                    } /* switch (mul1) */

                    ii = to1 - to_start;
                    kk = to2 - to_start;
                    NC_v_eff[0][0][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[0][0][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[1][1][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[1][1][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[0][1][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[0][1][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[1][0][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[1][0][Mc_AN][to1][to2].i = 0.0;

                    if (to1 == to2)
                    { /* double counting correction */
                      trace_N00.r = 0.0;
                      trace_N00.i = 0.0;
                      trace_N11.r = 0.0;
                      trace_N11.i = 0.0;

                      trace_N01.r = 0.0;
                      trace_N01.i = 0.0;
                      trace_N10.r = 0.0;
                      trace_N10.i = 0.0;
                      for (dd = 0; dd < (2 * l1 + 1); dd++)
                      {
                        trace_N00.r += NC_OcpN[0][0][0][Mc_AN][to_start + dd][to_start + dd].r;
                        trace_N00.i += NC_OcpN[0][0][0][Mc_AN][to_start + dd][to_start + dd].i;
                        trace_N11.r += NC_OcpN[0][1][1][Mc_AN][to_start + dd][to_start + dd].r;
                        trace_N11.i += NC_OcpN[0][1][1][Mc_AN][to_start + dd][to_start + dd].i;

                        trace_N01.r += NC_OcpN[0][0][1][Mc_AN][to_start + dd][to_start + dd].r;
                        trace_N01.i += NC_OcpN[0][0][1][Mc_AN][to_start + dd][to_start + dd].i;
                        trace_N10.r += NC_OcpN[0][1][0][Mc_AN][to_start + dd][to_start + dd].r;
                        trace_N10.i += NC_OcpN[0][1][0][Mc_AN][to_start + dd][to_start + dd].i;
                      }

                      /* Double counting term */
                      switch (dc_Type)
                      {
                      case 1: /* sFLL */
                        NC_v_eff[0][0][Mc_AN][to1][to2].r -= (Uvalue) * (trace_N00.r + trace_N11.r - 0.5) - (Jvalue) * (trace_N00.r - 0.5);
                        NC_v_eff[0][0][Mc_AN][to1][to2].i -= (Uvalue) * (trace_N00.i + trace_N11.i) - (Jvalue) * (trace_N00.i);
                        NC_v_eff[1][1][Mc_AN][to1][to2].r -= (Uvalue) * (trace_N00.r + trace_N11.r - 0.5) - (Jvalue) * (trace_N11.r - 0.5);
                        NC_v_eff[1][1][Mc_AN][to1][to2].i -= (Uvalue) * (trace_N00.i + trace_N11.i) - (Jvalue) * (trace_N11.i);

                        NC_v_eff[0][1][Mc_AN][to1][to2].r -= -(Jvalue)*trace_N10.r;
                        NC_v_eff[0][1][Mc_AN][to1][to2].i -= -(Jvalue)*trace_N10.i;
                        NC_v_eff[1][0][Mc_AN][to1][to2].r -= -(Jvalue)*trace_N01.r;
                        NC_v_eff[1][0][Mc_AN][to1][to2].i -= -(Jvalue)*trace_N01.i;
                        break;

                      case 2: /* sAMF */
                        if (on_off_dc == 1)
                        {
                          for (jj = 0; jj < (2 * l1 + 1); jj++)
                          {
                            for (ll = 0; ll < (2 * l1 + 1); ll++)
                            {
                              if (jj == ll)
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r - trace_N00.r / ((double)(2 * l1 + 1));
                                AMF_Array[NZUJ][0][1][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i - trace_N00.i / ((double)(2 * l1 + 1));
                                AMF_Array[NZUJ][1][0][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r - trace_N11.r / ((double)(2 * l1 + 1));
                                AMF_Array[NZUJ][1][1][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i - trace_N11.i / ((double)(2 * l1 + 1));
                                AMF_Array[NZUJ][2][0][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].r - (trace_N01.r + trace_N10.r) / ((double)(2 * (2 * l1 + 1)));
                                AMF_Array[NZUJ][2][1][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].i - (trace_N01.i - trace_N10.i) / ((double)(2 * (2 * l1 + 1)));
                                AMF_Array[NZUJ][3][0][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].r - (trace_N01.r + trace_N10.r) / ((double)(2 * (2 * l1 + 1)));
                                AMF_Array[NZUJ][3][1][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].i + (trace_N01.i - trace_N10.i) / ((double)(2 * (2 * l1 + 1)));
                              }
                              else
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][0][1][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][1][0][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][1][1][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][2][0][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][2][1][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][3][0][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][3][1][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].i;
                              }
                            }
                          }
                          on_off_dc = 0;
                        }
                        break;

                      case 3: /* cFLL */
                        NC_v_eff[0][0][Mc_AN][to1][to2].r -= (Uvalue) * (trace_N00.r + trace_N11.r - 0.5) - (Jvalue) * ((trace_N00.r + trace_N11.r) / 2.0 - 0.5);
                        NC_v_eff[0][0][Mc_AN][to1][to2].i -= (Uvalue) * (trace_N00.i + trace_N11.i) - (Jvalue) * (trace_N00.i + trace_N11.i) / 2.0;
                        NC_v_eff[1][1][Mc_AN][to1][to2].r -= (Uvalue) * (trace_N00.r + trace_N11.r - 0.5) - (Jvalue) * ((trace_N00.r + trace_N11.r) / 2.0 - 0.5);
                        NC_v_eff[1][1][Mc_AN][to1][to2].i -= (Uvalue) * (trace_N00.i + trace_N11.i) - (Jvalue) * (trace_N00.i + trace_N11.i) / 2.0;
                        break;

                      case 4: /* cAMF */
                        if (on_off_dc == 1)
                        {
                          for (jj = 0; jj < (2 * l1 + 1); jj++)
                          {
                            for (ll = 0; ll < (2 * l1 + 1); ll++)
                            {
                              if (jj == ll)
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r - (trace_N00.r + trace_N11.r) / (2.0 * ((double)(2 * l1 + 1)));
                                AMF_Array[NZUJ][0][1][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i - (trace_N00.i + trace_N11.i) / (2.0 * ((double)(2 * l1 + 1)));
                                AMF_Array[NZUJ][1][0][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r - (trace_N00.r + trace_N11.r) / (2.0 * ((double)(2 * l1 + 1)));
                                AMF_Array[NZUJ][1][1][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i - (trace_N00.i + trace_N11.i) / (2.0 * ((double)(2 * l1 + 1)));
                                AMF_Array[NZUJ][2][0][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][2][1][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][3][0][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][3][1][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].i;
                              }
                              else
                              {
                                AMF_Array[NZUJ][0][0][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][0][1][jj][ll] = NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][1][0][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][1][1][jj][ll] = NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][2][0][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][2][1][jj][ll] = NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].i;
                                AMF_Array[NZUJ][3][0][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].r;
                                AMF_Array[NZUJ][3][1][jj][ll] = NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].i;
                              }
                            }
                          }
                          on_off_dc = 0;
                        }
                        break;
                      }

                      /*f(dc_Type==3){
                         dum_alpha=0.0;
                         for(jj=0; jj<(2*l1+1); jj++){
                           for(ll=0; ll<(2*l1+1); ll++){
                             dum_alpha += AMF_Array[count][0][0][jj][ll]*AMF_Array[count][0][0][jj][ll]
                                         +AMF_Array[count][0][1][jj][ll]*AMF_Array[count][0][1][jj][ll]
                                         +AMF_Array[count][1][0][jj][ll]*AMF_Array[count][1][0][jj][ll]
                                         +AMF_Array[count][1][1][jj][ll]*AMF_Array[count][1][1][jj][ll]
                                         +AMF_Array[count][2][0][jj][ll]*AMF_Array[count][2][0][jj][ll]
                                         +AMF_Array[count][2][1][jj][ll]*AMF_Array[count][2][1][jj][ll]
                                         +AMF_Array[count][3][0][jj][ll]*AMF_Array[count][3][0][jj][ll]
                                         +AMF_Array[count][3][1][jj][ll]*AMF_Array[count][3][1][jj][ll];
                           }
                         }
                         dum_alpha2 = ((double)(2*(2*l1+1)))*(trace_N00.r+trace_N11.r)
                                     -(trace_N00.r+trace_N11.r)*(trace_N00.r+trace_N11.r)
                                     -(M_z*M_z + M_x*M_x + M_y*M_y);
                         if(dum_alpha2>1.0e-14 || dum_alpha2<-1.0e-14){
                           dc_alpha[count] = ((double)(2*(2*l1+1))*dum_alpha)/dum_alpha2;
                         }
                         else{
                           dc_alpha[count] = 0.0;
                         }
                       } */

                    } /* double counting correction */

                    /* loop start for interaction potential */
                    for (jj = 0; jj < (2 * l1 + 1); jj++)
                    {
                      for (ll = 0; ll < (2 * l1 + 1); ll++)
                      {
                        switch (dc_Type)
                        {
                        case 1: /* sFLL */
                          NC_v_eff[0][0][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r + NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r);
                          NC_v_eff[0][0][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i + NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i);
                          NC_v_eff[1][1][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r + NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r);
                          NC_v_eff[1][1][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i + NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i);
                          NC_v_eff[0][1][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].r;
                          NC_v_eff[0][1][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].i;
                          NC_v_eff[1][0][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].r;
                          NC_v_eff[1][0][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].i;
                          break;

                        case 2: /* sAMF */
                          NC_v_eff[0][0][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][0][0][jj][ll] + AMF_Array[NZUJ][1][0][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][0][0][jj][ll]);
                          NC_v_eff[0][0][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][0][1][jj][ll] + AMF_Array[NZUJ][1][1][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][0][1][jj][ll]);
                          NC_v_eff[1][1][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][1][0][jj][ll] + AMF_Array[NZUJ][0][0][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][1][0][jj][ll]);
                          NC_v_eff[1][1][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][1][1][jj][ll] + AMF_Array[NZUJ][0][1][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][1][1][jj][ll]);
                          NC_v_eff[0][1][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][3][0][jj][ll];
                          NC_v_eff[0][1][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][3][1][jj][ll];
                          NC_v_eff[1][0][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][2][0][jj][ll];
                          NC_v_eff[1][0][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][2][1][jj][ll];
                          break;

                        case 3: /* cFLL */
                          NC_v_eff[0][0][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r + NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r);
                          NC_v_eff[0][0][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i + NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i);
                          NC_v_eff[1][1][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r + NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].r) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].r);
                          NC_v_eff[1][1][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i + NC_OcpN[0][0][0][Mc_AN][to_start + jj][to_start + ll].i) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (NC_OcpN[0][1][1][Mc_AN][to_start + jj][to_start + ll].i);
                          NC_v_eff[0][1][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].r;
                          NC_v_eff[0][1][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][1][0][Mc_AN][to_start + jj][to_start + ll].i;
                          NC_v_eff[1][0][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].r;
                          NC_v_eff[1][0][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * NC_OcpN[0][0][1][Mc_AN][to_start + jj][to_start + ll].i;
                          break;

                        case 4: /* cAMF */
                          NC_v_eff[0][0][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][0][0][jj][ll] + AMF_Array[NZUJ][1][0][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][0][0][jj][ll]);
                          NC_v_eff[0][0][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][0][1][jj][ll] + AMF_Array[NZUJ][1][1][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][0][1][jj][ll]);
                          NC_v_eff[1][1][Mc_AN][to1][to2].r += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][1][0][jj][ll] + AMF_Array[NZUJ][0][0][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][1][0][jj][ll]);
                          NC_v_eff[1][1][Mc_AN][to1][to2].i += Coulomb_Array[NZUJ][ii][jj][kk][ll] * (AMF_Array[NZUJ][1][1][jj][ll] + AMF_Array[NZUJ][0][1][jj][ll]) - Coulomb_Array[NZUJ][ii][jj][ll][kk] * (AMF_Array[NZUJ][1][1][jj][ll]);
                          NC_v_eff[0][1][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][3][0][jj][ll];
                          NC_v_eff[0][1][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][3][1][jj][ll];
                          NC_v_eff[1][0][Mc_AN][to1][to2].r += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][2][0][jj][ll];
                          NC_v_eff[1][0][Mc_AN][to1][to2].i += -Coulomb_Array[NZUJ][ii][jj][ll][kk] * AMF_Array[NZUJ][2][1][jj][ll];
                          break;
                          /*  case 3:
            NC_v_eff[0][0][Mc_AN][to1][to2].r += Coulomb_Array[count][ii][jj][kk][ll]*dc_alpha[count]*(NC_OcpN[0][0][0][Mc_AN][to_start+jj][to_start+ll].r
                                                                                                                        +NC_OcpN[0][1][1][Mc_AN][to_start+jj][to_start+ll].r)
                                  -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*(NC_OcpN[0][0][0][Mc_AN][to_start+jj][to_start+ll].r);
            NC_v_eff[0][0][Mc_AN][to1][to2].i += Coulomb_Array[count][ii][jj][kk][ll]*dc_alpha[count]*(NC_OcpN[0][0][0][Mc_AN][to_start+jj][to_start+ll].i
                                                                                                                        +NC_OcpN[0][1][1][Mc_AN][to_start+jj][to_start+ll].i)
                        -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*(NC_OcpN[0][0][0][Mc_AN][to_start+jj][to_start+ll].i);
            NC_v_eff[1][1][Mc_AN][to1][to2].r += Coulomb_Array[count][ii][jj][kk][ll]*dc_alpha[count]*(NC_OcpN[0][1][1][Mc_AN][to_start+jj][to_start+ll].r
                                                                                                                        +NC_OcpN[0][0][0][Mc_AN][to_start+jj][to_start+ll].r)
                        -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*(NC_OcpN[0][1][1][Mc_AN][to_start+jj][to_start+ll].r);
            NC_v_eff[1][1][Mc_AN][to1][to2].i += Coulomb_Array[count][ii][jj][kk][ll]*dc_alpha[count]*(NC_OcpN[0][1][1][Mc_AN][to_start+jj][to_start+ll].i
                                                                                                                        +NC_OcpN[0][0][0][Mc_AN][to_start+jj][to_start+ll].i)
                        -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*(NC_OcpN[0][1][1][Mc_AN][to_start+jj][to_start+ll].i);
            NC_v_eff[0][1][Mc_AN][to1][to2].r += -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*NC_OcpN[0][1][0][Mc_AN][to_start+jj][to_start+ll].r;
            NC_v_eff[0][1][Mc_AN][to1][to2].i += -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*NC_OcpN[0][1][0][Mc_AN][to_start+jj][to_start+ll].i;
            NC_v_eff[1][0][Mc_AN][to1][to2].r += -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*NC_OcpN[0][0][1][Mc_AN][to_start+jj][to_start+ll].r;
            NC_v_eff[1][0][Mc_AN][to1][to2].i += -Coulomb_Array[count][ii][jj][ll][kk]*dc_alpha[count]*NC_OcpN[0][0][1][Mc_AN][to_start+jj][to_start+ll].i;

            NC_v_eff[0][0][Mc_AN][to1][to2].r += Coulomb_Array[count][ii][jj][kk][ll]*(1.0-dc_alpha[count])*(AMF_Array[count][0][0][jj][ll]
                                                                                                                              +AMF_Array[count][1][0][jj][ll])
                                  -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*(AMF_Array[count][0][0][jj][ll]);
            NC_v_eff[0][0][Mc_AN][to1][to2].i += Coulomb_Array[count][ii][jj][kk][ll]*(1.0-dc_alpha[count])*(AMF_Array[count][0][1][jj][ll]
                                                                                                                              +AMF_Array[count][1][1][jj][ll])
                        -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*(AMF_Array[count][0][1][jj][ll]);
            NC_v_eff[1][1][Mc_AN][to1][to2].r += Coulomb_Array[count][ii][jj][kk][ll]*(1.0-dc_alpha[count])*(AMF_Array[count][1][0][jj][ll]
                                                                                                                              +AMF_Array[count][0][0][jj][ll])
                        -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*(AMF_Array[count][1][0][jj][ll]);
            NC_v_eff[1][1][Mc_AN][to1][to2].i += Coulomb_Array[count][ii][jj][kk][ll]*(1.0-dc_alpha[count])*(AMF_Array[count][1][1][jj][ll]
                                                                                                                              +AMF_Array[count][0][1][jj][ll])
                        -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*(AMF_Array[count][1][1][jj][ll]);
            NC_v_eff[0][1][Mc_AN][to1][to2].r += -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*AMF_Array[count][3][0][jj][ll];
            NC_v_eff[0][1][Mc_AN][to1][to2].i += -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*AMF_Array[count][3][1][jj][ll];
            NC_v_eff[1][0][Mc_AN][to1][to2].r += -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*AMF_Array[count][2][0][jj][ll];
            NC_v_eff[1][0][Mc_AN][to1][to2].i += -Coulomb_Array[count][ii][jj][ll][kk]*(1.0-dc_alpha[count])*AMF_Array[count][2][1][jj][ll];
                            break; */
                        }
                      }
                    }

                  } /* U || J != 0.0 */
                  else
                  {
                    NC_v_eff[0][0][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[0][0][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[1][1][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[1][1][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[0][1][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[0][1][Mc_AN][to1][to2].i = 0.0;
                    NC_v_eff[1][0][Mc_AN][to1][to2].r = 0.0;
                    NC_v_eff[1][0][Mc_AN][to1][to2].i = 0.0;
                  }

                  to2++;

                } /* m2 */
                mul_index2++;
              } /* mul2 */
            }   /* l2*/

            to1++;
          } /* m1 */
          mul_index1++;
          on_off_dc = 1;
        } /* mul1 */
      }   /* l1 */

      break;
    }

    /************************************************************
     ***********************************************************
     ***********************************************************
     ***********************************************************

       calculate veff by the constraint DFT which controls
       the spin direction but not the magnitude.

     ***********************************************************
     ***********************************************************
     ***********************************************************
    ************************************************************/

    if (Constraint_NCS_switch == 1 && Constraint_SpinAngle[Gc_AN] == 1)
    {

      /* calculate TN */

      TN[0][0].r = 0.0;
      TN[0][1].r = 0.0;
      TN[1][0].r = 0.0;
      TN[1][1].r = 0.0;
      TN[0][0].i = 0.0;
      TN[0][1].i = 0.0;
      TN[1][0].i = 0.0;
      TN[1][1].i = 0.0;

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {

        TN[0][0].r += NC_OcpN[0][0][0][Mc_AN][i][i].r;
        TN[0][1].r += NC_OcpN[0][0][1][Mc_AN][i][i].r;
        TN[1][0].r += NC_OcpN[0][1][0][Mc_AN][i][i].r;
        TN[1][1].r += NC_OcpN[0][1][1][Mc_AN][i][i].r;

        /*
        conjugate complex of TN due to difference
        in the definition between density matrix
        and charge density
        */

        TN[0][0].i -= NC_OcpN[0][0][0][Mc_AN][i][i].i;
        TN[0][1].i -= NC_OcpN[0][0][1][Mc_AN][i][i].i;
        TN[1][0].i -= NC_OcpN[0][1][0][Mc_AN][i][i].i;
        TN[1][1].i -= NC_OcpN[0][1][1][Mc_AN][i][i].i;
      }

      /*
      printf("TN.r Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN[i][j].r);
        }
        printf("\n");
      }

      printf("TN.i Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN[i][j].i);
        }
        printf("\n");
      }
      */

      EulerAngle_Spin(1,
                      TN[0][0].r, TN[1][1].r,
                      TN[0][1].r, TN[0][1].i,
                      TN[1][0].r, TN[1][0].i,
                      Nup, Ndn, theta, phi);

      /* calculate TN0 */

      /*
      printf("Nup=%15.12f Ndn=%15.12f theta=%15.12f phi=%15.12f\n",Nup[0],Ndn[0],theta[0],phi[0]);
      printf("theta =%15.12f phi =%15.12f\n",theta[0]/PI*180.0,phi[0]/PI*180.0);

      printf("theta0=%15.12f phi0=%15.12f\n",InitAngle0_Spin[Gc_AN]/PI*180.0,
                                             InitAngle1_Spin[Gc_AN]/PI*180.0);
      */

      sit = sin(0.5 * InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5 * InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5 * InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5 * InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop * cot;
      U[0][0].i = sip * cot;
      U[0][1].r = cop * sit;
      U[0][1].i = -sip * sit;
      U[1][0].r = -cop * sit;
      U[1][0].i = -sip * sit;
      U[1][1].r = cop * cot;
      U[1][1].i = -sip * cot;

      TN0[0][0].r = Nup[0] * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn[0] * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

      TN0[0][0].i = 0.0;

      TN0[0][1].r = Nup[0] * (U[0][0].r * U[0][1].r + U[0][0].i * U[0][1].i) + Ndn[0] * (U[1][0].r * U[1][1].r + U[1][0].i * U[1][1].i);

      TN0[0][1].i = Nup[0] * (-U[0][0].i * U[0][1].r + U[0][0].r * U[0][1].i) + Ndn[0] * (-U[1][0].i * U[1][1].r + U[1][0].r * U[1][1].i);

      TN0[1][0].r = Nup[0] * (U[0][1].r * U[0][0].r + U[0][1].i * U[0][0].i) + Ndn[0] * (U[1][1].r * U[1][0].r + U[1][1].i * U[1][0].i);

      TN0[1][0].i = Nup[0] * (-U[0][1].i * U[0][0].r + U[0][1].r * U[0][0].i) + Ndn[0] * (-U[1][1].i * U[1][0].r + U[1][1].r * U[1][0].i);

      TN0[1][1].r = Nup[0] * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn[0] * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

      TN0[1][1].i = 0.0;

      /* calculate dTN */

      Calc_dTN(Constraint_NCS_switch, TN, dTN, U, theta, phi);

      /*

      {


  dcomplex TNA[10][10];
  dcomplex TNB[10][10];
  dcomplex TNC[10][10];

        dcomplex ctmp1,ctmp2;



      l1 = 1;
      l2 = 0;
      tmp1 = 0.0001;
      tmp2 = 0.01;


      EulerAngle_Spin( 0,
                       TN[0][0].r, TN[1][1].r,
                       TN[0][1].r, TN[0][1].i,
                       TN[1][0].r, TN[1][0].i,
                       Nup, Ndn, theta, phi );

      printf("V0 Nup.r=%15.12f Nup.i=%15.12f\n",Nup[0],Nup[1]);
      printf("V0 Ndn.r=%15.12f Ndn.i=%15.12f\n",Ndn[0],Ndn[1]);


      sit = sin(0.5*InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5*InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5*InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5*InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop*cot;  U[0][0].i = sip*cot;
      U[0][1].r = cop*sit;  U[0][1].i =-sip*sit;
      U[1][0].r =-cop*sit;  U[1][0].i =-sip*sit;
      U[1][1].r = cop*cot;  U[1][1].i =-sip*cot;

      TN0[0][0].r =    Nup[0]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                     + Ndn[0]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      TN0[0][0].i =    Nup[1]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                     + Ndn[1]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      ctmp1.r = U[0][0].r*U[0][1].r + U[0][0].i*U[0][1].i;
      ctmp1.i =-U[0][0].i*U[0][1].r + U[0][0].r*U[0][1].i;
      ctmp2.r = U[1][0].r*U[1][1].r + U[1][0].i*U[1][1].i;
      ctmp2.i =-U[1][0].i*U[1][1].r + U[1][0].r*U[1][1].i;

      TN0[0][1].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[0][1].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      ctmp1.r =  U[0][1].r*U[0][0].r + U[0][1].i*U[0][0].i;
      ctmp1.i = -U[0][1].i*U[0][0].r + U[0][1].r*U[0][0].i;
      ctmp2.r =  U[1][1].r*U[1][0].r + U[1][1].i*U[1][0].i;
      ctmp2.i = -U[1][1].i*U[1][0].r + U[1][1].r*U[1][0].i;

      TN0[1][0].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[1][0].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      TN0[1][1].r =  Nup[0]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[0]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );

      TN0[1][1].i =  Nup[1]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[1]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );


      Calc_dTN( Constraint_NCS_switch, TN, dTN, U, theta, phi );

      printf("1 TN0.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].r);

          TNA[i][j] = TN0[i][j];
        }
        printf("\n");
      }

      printf("1 TN0.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].i);
        }
        printf("\n");
      }




      TN[l1][l2].r += tmp1;
      TN[l1][l2].i += tmp2;

      EulerAngle_Spin( 0,
                       TN[0][0].r, TN[1][1].r,
                       TN[0][1].r, TN[0][1].i,
                       TN[1][0].r, TN[1][0].i,
                       Nup, Ndn, theta, phi );

      printf("V1 Nup.r=%15.12f Nup.i=%15.12f\n",Nup[0],Nup[1]);
      printf("V1 Ndn.r=%15.12f Ndn.i=%15.12f\n",Ndn[0],Ndn[1]);


      sit = sin(0.5*InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5*InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5*InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5*InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop*cot;  U[0][0].i = sip*cot;
      U[0][1].r = cop*sit;  U[0][1].i =-sip*sit;
      U[1][0].r =-cop*sit;  U[1][0].i =-sip*sit;
      U[1][1].r = cop*cot;  U[1][1].i =-sip*cot;



      TN0[0][0].r =    Nup[0]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[0]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      TN0[0][0].i =    Nup[1]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[1]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      ctmp1.r = U[0][0].r*U[0][1].r + U[0][0].i*U[0][1].i;
      ctmp1.i =-U[0][0].i*U[0][1].r + U[0][0].r*U[0][1].i;
      ctmp2.r = U[1][0].r*U[1][1].r + U[1][0].i*U[1][1].i;
      ctmp2.i =-U[1][0].i*U[1][1].r + U[1][0].r*U[1][1].i;

      TN0[0][1].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[0][1].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      ctmp1.r =  U[0][1].r*U[0][0].r + U[0][1].i*U[0][0].i;
      ctmp1.i = -U[0][1].i*U[0][0].r + U[0][1].r*U[0][0].i;
      ctmp2.r =  U[1][1].r*U[1][0].r + U[1][1].i*U[1][0].i;
      ctmp2.i = -U[1][1].i*U[1][0].r + U[1][1].r*U[1][0].i;

      TN0[1][0].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[1][0].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      TN0[1][1].r =    Nup[0]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[0]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );

      TN0[1][1].i =    Nup[1]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[1]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );







      Calc_dTN( Constraint_NCS_switch, TN, dTN, U, theta, phi );

      printf("\nanalytical dTN.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",dTN[l1][l2][i][j].r);
        }
        printf("\n");
      }

      printf("analytical dTN.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",dTN[l1][l2][i][j].i);
        }
        printf("\n");
      }






      TN[l1][l2].r += tmp1;
      TN[l1][l2].i += tmp2;

      EulerAngle_Spin( 0,
                       TN[0][0].r, TN[1][1].r,
                       TN[0][1].r, TN[0][1].i,
                       TN[1][0].r, TN[1][0].i,
                       Nup, Ndn, theta, phi );

      printf("V2 Nup.r=%15.12f Nup.i=%15.12f\n",Nup[0],Nup[1]);
      printf("V2 Ndn.r=%15.12f Ndn.i=%15.12f\n",Ndn[0],Ndn[1]);


      sit = sin(0.5*InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5*InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5*InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5*InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop*cot;  U[0][0].i = sip*cot;
      U[0][1].r = cop*sit;  U[0][1].i =-sip*sit;
      U[1][0].r =-cop*sit;  U[1][0].i =-sip*sit;
      U[1][1].r = cop*cot;  U[1][1].i =-sip*cot;




      TN0[0][0].r =    Nup[0]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[0]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      TN0[0][0].i =    Nup[1]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[1]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      ctmp1.r = U[0][0].r*U[0][1].r + U[0][0].i*U[0][1].i;
      ctmp1.i =-U[0][0].i*U[0][1].r + U[0][0].r*U[0][1].i;
      ctmp2.r = U[1][0].r*U[1][1].r + U[1][0].i*U[1][1].i;
      ctmp2.i =-U[1][0].i*U[1][1].r + U[1][0].r*U[1][1].i;

      TN0[0][1].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[0][1].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      ctmp1.r =  U[0][1].r*U[0][0].r + U[0][1].i*U[0][0].i;
      ctmp1.i = -U[0][1].i*U[0][0].r + U[0][1].r*U[0][0].i;
      ctmp2.r =  U[1][1].r*U[1][0].r + U[1][1].i*U[1][0].i;
      ctmp2.i = -U[1][1].i*U[1][0].r + U[1][1].r*U[1][0].i;

      TN0[1][0].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[1][0].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      TN0[1][1].r =    Nup[0]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[0]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );

      TN0[1][1].i =    Nup[1]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[1]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );






      Calc_dTN( Constraint_NCS_switch, TN, dTN, U, theta, phi );




      printf("2 TN0.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].r);
          TNB[i][j] = TN0[i][j];

        }
        printf("\n");
      }

      printf("2 TN0.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].i);
        }
        printf("\n");
      }



      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          TNC[i][j].r = 0.5*(TNB[i][j].r - TNA[i][j].r)/( tmp1*tmp1 + tmp2*tmp2 );
          TNC[i][j].i = 0.5*(TNB[i][j].i - TNA[i][j].i)/( tmp1*tmp1 + tmp2*tmp2 );
        }
      }

      printf("\nnumerical dTN0.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ", tmp1*TNC[i][j].r + tmp2*TNC[i][j].i );
        }
        printf("\n");
      }

      printf("numerical dTN0.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ", tmp1*TNC[i][j].i - tmp2*TNC[i][j].r );
        }
        printf("\n");
      }


      }

      MPI_Finalize();
      exit(0);
      */

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {

          if (i == j)
          {

            for (s1 = 0; s1 < 2; s1++)
            {
              for (s2 = 0; s2 < 2; s2++)
              {

                csum1 = Complex(0.0, 0.0);

                for (s3 = 0; s3 < 2; s3++)
                {
                  for (s4 = 0; s4 < 2; s4++)
                  {

                    if (s1 == s3 && s2 == s4)
                    {

                      ctmp1.r = TN[s3][s4].r - TN0[s3][s4].r;
                      ctmp1.i = TN[s3][s4].i - TN0[s3][s4].i;
                      ctmp2.r = 1.0 - dTN[s1][s2][s4][s3].r;
                      ctmp2.i = -dTN[s1][s2][s4][s3].i;

                      csum1.r += ctmp1.r * ctmp2.r - ctmp1.i * ctmp2.i;
                      csum1.i += ctmp1.r * ctmp2.i + ctmp1.i * ctmp2.r;
                    }
                    else
                    {

                      ctmp1.r = TN[s3][s4].r - TN0[s3][s4].r;
                      ctmp1.i = TN[s3][s4].i - TN0[s3][s4].i;
                      ctmp2.r = -dTN[s1][s2][s4][s3].r;
                      ctmp2.i = -dTN[s1][s2][s4][s3].i;

                      csum1.r += ctmp1.r * ctmp2.r - ctmp1.i * ctmp2.i;
                      csum1.i += ctmp1.r * ctmp2.i + ctmp1.i * ctmp2.r;
                    }

                  } /* s4 */
                }   /* s3 */

                NC_v_eff[s1][s2][Mc_AN][i][j].r += 2.0 * Constraint_NCS_V * csum1.r;
                NC_v_eff[s1][s2][Mc_AN][i][j].i += 2.0 * Constraint_NCS_V * csum1.i;
              }
            }
          }
        }
      }

      /* calculate the penalty functional, Ucs */

      tmp1 = 0.0;

      for (s1 = 0; s1 < 2; s1++)
      {
        for (s2 = 0; s2 < 2; s2++)
        {

          ctmp1.r = TN[s1][s2].r - TN0[s1][s2].r;
          ctmp1.i = TN[s1][s2].i - TN0[s1][s2].i;
          tmp1 += ctmp1.r * ctmp1.r + ctmp1.i * ctmp1.i;
        }
      }

      My_Ucs += Constraint_NCS_V * tmp1;

    } /* if (Constraint_NCS_switch==1 && Constraint_SpinAngle[Gc_AN]==1 ) */

    /************************************************************
     ***********************************************************
     ***********************************************************
     ***********************************************************

      calculate veff by the constraint DFT which controls
      both the direction and the magnitude of spin.

     ***********************************************************
     ***********************************************************
     ***********************************************************
    ************************************************************/

    if (Constraint_NCS_switch == 2 && Constraint_SpinAngle[Gc_AN] == 1)
    {

      /* calculate TN */

      TN[0][0].r = 0.0;
      TN[0][1].r = 0.0;
      TN[1][0].r = 0.0;
      TN[1][1].r = 0.0;
      TN[0][0].i = 0.0;
      TN[0][1].i = 0.0;
      TN[1][0].i = 0.0;
      TN[1][1].i = 0.0;

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {

        TN[0][0].r += NC_OcpN[0][0][0][Mc_AN][i][i].r;
        TN[0][1].r += NC_OcpN[0][0][1][Mc_AN][i][i].r;
        TN[1][0].r += NC_OcpN[0][1][0][Mc_AN][i][i].r;
        TN[1][1].r += NC_OcpN[0][1][1][Mc_AN][i][i].r;

        /*
        conjugate complex of TN due to difference
        in the definition between density matrix
        and charge density
        */

        TN[0][0].i -= NC_OcpN[0][0][0][Mc_AN][i][i].i;
        TN[0][1].i -= NC_OcpN[0][0][1][Mc_AN][i][i].i;
        TN[1][0].i -= NC_OcpN[0][1][0][Mc_AN][i][i].i;
        TN[1][1].i -= NC_OcpN[0][1][1][Mc_AN][i][i].i;
      }

      /*
      printf("TN.r Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN[i][j].r);
        }
        printf("\n");
      }

      printf("TN.i Mc_AN=%2d\n",Mc_AN);
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN[i][j].i);
        }
        printf("\n");
      }
      */

      EulerAngle_Spin(1,
                      TN[0][0].r, TN[1][1].r,
                      TN[0][1].r, TN[0][1].i,
                      TN[1][0].r, TN[1][0].i,
                      Nup, Ndn, theta, phi);

      /**********************
           calculate TN0
      **********************/

      /*
      printf("Nup=%15.12f Ndn=%15.12f theta=%15.12f phi=%15.12f\n",Nup[0],Ndn[0],theta[0],phi[0]);
      printf("theta =%15.12f phi =%15.12f\n",theta[0]/PI*180.0,phi[0]/PI*180.0);

      printf("theta0=%15.12f phi0=%15.12f\n",InitAngle0_Spin[Gc_AN]/PI*180.0,
                                             InitAngle1_Spin[Gc_AN]/PI*180.0);
      */

      /* constraint which trys to keep the initial magnetic moment */

      Nup0 = Nup[0];
      Ndn0 = Ndn[0];

      Nup[0] = 0.5 * (Nup0 + Ndn0 + InitMagneticMoment[Gc_AN]);
      Ndn[0] = 0.5 * (Nup0 + Ndn0 - InitMagneticMoment[Gc_AN]);

      /* calculaion of TN0 */

      sit = sin(0.5 * InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5 * InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5 * InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5 * InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop * cot;
      U[0][0].i = sip * cot;
      U[0][1].r = cop * sit;
      U[0][1].i = -sip * sit;
      U[1][0].r = -cop * sit;
      U[1][0].i = -sip * sit;
      U[1][1].r = cop * cot;
      U[1][1].i = -sip * cot;

      TN0[0][0].r = Nup[0] * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn[0] * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

      TN0[0][0].i = 0.0;

      TN0[0][1].r = Nup[0] * (U[0][0].r * U[0][1].r + U[0][0].i * U[0][1].i) + Ndn[0] * (U[1][0].r * U[1][1].r + U[1][0].i * U[1][1].i);

      TN0[0][1].i = Nup[0] * (-U[0][0].i * U[0][1].r + U[0][0].r * U[0][1].i) + Ndn[0] * (-U[1][0].i * U[1][1].r + U[1][0].r * U[1][1].i);

      TN0[1][0].r = Nup[0] * (U[0][1].r * U[0][0].r + U[0][1].i * U[0][0].i) + Ndn[0] * (U[1][1].r * U[1][0].r + U[1][1].i * U[1][0].i);

      TN0[1][0].i = Nup[0] * (-U[0][1].i * U[0][0].r + U[0][1].r * U[0][0].i) + Ndn[0] * (-U[1][1].i * U[1][0].r + U[1][1].r * U[1][0].i);

      TN0[1][1].r = Nup[0] * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn[0] * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

      TN0[1][1].i = 0.0;

      /* calculate dTN */

      Calc_dTN(Constraint_NCS_switch, TN, dTN, U, theta, phi);

      /*

      {


  dcomplex TNA[10][10];
  dcomplex TNB[10][10];
  dcomplex TNC[10][10];

        dcomplex ctmp1,ctmp2;



      l1 = 1;
      l2 = 0;
      tmp1 = 0.0001;
      tmp2 = 0.01;


      EulerAngle_Spin( 0,
                       TN[0][0].r, TN[1][1].r,
                       TN[0][1].r, TN[0][1].i,
                       TN[1][0].r, TN[1][0].i,
                       Nup, Ndn, theta, phi );

      printf("V0 Nup.r=%15.12f Nup.i=%15.12f\n",Nup[0],Nup[1]);
      printf("V0 Ndn.r=%15.12f Ndn.i=%15.12f\n",Ndn[0],Ndn[1]);


      sit = sin(0.5*InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5*InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5*InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5*InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop*cot;  U[0][0].i = sip*cot;
      U[0][1].r = cop*sit;  U[0][1].i =-sip*sit;
      U[1][0].r =-cop*sit;  U[1][0].i =-sip*sit;
      U[1][1].r = cop*cot;  U[1][1].i =-sip*cot;

      TN0[0][0].r =    Nup[0]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                     + Ndn[0]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      TN0[0][0].i =    Nup[1]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                     + Ndn[1]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      ctmp1.r = U[0][0].r*U[0][1].r + U[0][0].i*U[0][1].i;
      ctmp1.i =-U[0][0].i*U[0][1].r + U[0][0].r*U[0][1].i;
      ctmp2.r = U[1][0].r*U[1][1].r + U[1][0].i*U[1][1].i;
      ctmp2.i =-U[1][0].i*U[1][1].r + U[1][0].r*U[1][1].i;

      TN0[0][1].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[0][1].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      ctmp1.r =  U[0][1].r*U[0][0].r + U[0][1].i*U[0][0].i;
      ctmp1.i = -U[0][1].i*U[0][0].r + U[0][1].r*U[0][0].i;
      ctmp2.r =  U[1][1].r*U[1][0].r + U[1][1].i*U[1][0].i;
      ctmp2.i = -U[1][1].i*U[1][0].r + U[1][1].r*U[1][0].i;

      TN0[1][0].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[1][0].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      TN0[1][1].r =  Nup[0]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[0]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );

      TN0[1][1].i =  Nup[1]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[1]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );


      Calc_dTN( Constraint_NCS_switch, TN, dTN, U, theta, phi );

      printf("1 TN0.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].r);

          TNA[i][j] = TN0[i][j];
        }
        printf("\n");
      }

      printf("1 TN0.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].i);
        }
        printf("\n");
      }




      TN[l1][l2].r += tmp1;
      TN[l1][l2].i += tmp2;

      EulerAngle_Spin( 0,
                       TN[0][0].r, TN[1][1].r,
                       TN[0][1].r, TN[0][1].i,
                       TN[1][0].r, TN[1][0].i,
                       Nup, Ndn, theta, phi );

      printf("V1 Nup.r=%15.12f Nup.i=%15.12f\n",Nup[0],Nup[1]);
      printf("V1 Ndn.r=%15.12f Ndn.i=%15.12f\n",Ndn[0],Ndn[1]);


      sit = sin(0.5*InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5*InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5*InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5*InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop*cot;  U[0][0].i = sip*cot;
      U[0][1].r = cop*sit;  U[0][1].i =-sip*sit;
      U[1][0].r =-cop*sit;  U[1][0].i =-sip*sit;
      U[1][1].r = cop*cot;  U[1][1].i =-sip*cot;



      TN0[0][0].r =    Nup[0]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[0]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      TN0[0][0].i =    Nup[1]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[1]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      ctmp1.r = U[0][0].r*U[0][1].r + U[0][0].i*U[0][1].i;
      ctmp1.i =-U[0][0].i*U[0][1].r + U[0][0].r*U[0][1].i;
      ctmp2.r = U[1][0].r*U[1][1].r + U[1][0].i*U[1][1].i;
      ctmp2.i =-U[1][0].i*U[1][1].r + U[1][0].r*U[1][1].i;

      TN0[0][1].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[0][1].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      ctmp1.r =  U[0][1].r*U[0][0].r + U[0][1].i*U[0][0].i;
      ctmp1.i = -U[0][1].i*U[0][0].r + U[0][1].r*U[0][0].i;
      ctmp2.r =  U[1][1].r*U[1][0].r + U[1][1].i*U[1][0].i;
      ctmp2.i = -U[1][1].i*U[1][0].r + U[1][1].r*U[1][0].i;

      TN0[1][0].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[1][0].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      TN0[1][1].r =    Nup[0]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[0]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );

      TN0[1][1].i =    Nup[1]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[1]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );







      Calc_dTN( Constraint_NCS_switch, TN, dTN, U, theta, phi );

      printf("\nanalytical dTN.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",dTN[l1][l2][i][j].r);
        }
        printf("\n");
      }

      printf("analytical dTN.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",dTN[l1][l2][i][j].i);
        }
        printf("\n");
      }






      TN[l1][l2].r += tmp1;
      TN[l1][l2].i += tmp2;

      EulerAngle_Spin( 0,
                       TN[0][0].r, TN[1][1].r,
                       TN[0][1].r, TN[0][1].i,
                       TN[1][0].r, TN[1][0].i,
                       Nup, Ndn, theta, phi );

      printf("V2 Nup.r=%15.12f Nup.i=%15.12f\n",Nup[0],Nup[1]);
      printf("V2 Ndn.r=%15.12f Ndn.i=%15.12f\n",Ndn[0],Ndn[1]);


      sit = sin(0.5*InitAngle0_Spin[Gc_AN]);
      cot = cos(0.5*InitAngle0_Spin[Gc_AN]);
      sip = sin(0.5*InitAngle1_Spin[Gc_AN]);
      cop = cos(0.5*InitAngle1_Spin[Gc_AN]);

      U[0][0].r = cop*cot;  U[0][0].i = sip*cot;
      U[0][1].r = cop*sit;  U[0][1].i =-sip*sit;
      U[1][0].r =-cop*sit;  U[1][0].i =-sip*sit;
      U[1][1].r = cop*cot;  U[1][1].i =-sip*cot;




      TN0[0][0].r =    Nup[0]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[0]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      TN0[0][0].i =    Nup[1]*( U[0][0].r*U[0][0].r + U[0][0].i*U[0][0].i )
                   + Ndn[1]*( U[1][0].r*U[1][0].r + U[1][0].i*U[1][0].i );

      ctmp1.r = U[0][0].r*U[0][1].r + U[0][0].i*U[0][1].i;
      ctmp1.i =-U[0][0].i*U[0][1].r + U[0][0].r*U[0][1].i;
      ctmp2.r = U[1][0].r*U[1][1].r + U[1][0].i*U[1][1].i;
      ctmp2.i =-U[1][0].i*U[1][1].r + U[1][0].r*U[1][1].i;

      TN0[0][1].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[0][1].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      ctmp1.r =  U[0][1].r*U[0][0].r + U[0][1].i*U[0][0].i;
      ctmp1.i = -U[0][1].i*U[0][0].r + U[0][1].r*U[0][0].i;
      ctmp2.r =  U[1][1].r*U[1][0].r + U[1][1].i*U[1][0].i;
      ctmp2.i = -U[1][1].i*U[1][0].r + U[1][1].r*U[1][0].i;

      TN0[1][0].r = Nup[0]*ctmp1.r - Nup[1]*ctmp1.i
                  + Ndn[0]*ctmp2.r - Ndn[1]*ctmp2.i;

      TN0[1][0].i = Nup[0]*ctmp1.i + Nup[1]*ctmp1.r
                  + Ndn[0]*ctmp2.i + Ndn[1]*ctmp2.r;

      TN0[1][1].r =    Nup[0]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[0]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );

      TN0[1][1].i =    Nup[1]*( U[0][1].r*U[0][1].r + U[0][1].i*U[0][1].i )
                   + Ndn[1]*( U[1][1].r*U[1][1].r + U[1][1].i*U[1][1].i );






      Calc_dTN( Constraint_NCS_switch, TN, dTN, U, theta, phi );




      printf("2 TN0.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].r);
          TNB[i][j] = TN0[i][j];

        }
        printf("\n");
      }

      printf("2 TN0.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ",TN0[i][j].i);
        }
        printf("\n");
      }



      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          TNC[i][j].r = 0.5*(TNB[i][j].r - TNA[i][j].r)/( tmp1*tmp1 + tmp2*tmp2 );
          TNC[i][j].i = 0.5*(TNB[i][j].i - TNA[i][j].i)/( tmp1*tmp1 + tmp2*tmp2 );
        }
      }

      printf("\nnumerical dTN0.r\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ", tmp1*TNC[i][j].r + tmp2*TNC[i][j].i );
        }
        printf("\n");
      }

      printf("numerical dTN0.i\n");
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          printf("%15.12f ", tmp1*TNC[i][j].i - tmp2*TNC[i][j].r );
        }
        printf("\n");
      }


      }

      MPI_Finalize();
      exit(0);
      */

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {

          if (i == j)
          {

            for (s1 = 0; s1 < 2; s1++)
            {
              for (s2 = 0; s2 < 2; s2++)
              {

                csum1 = Complex(0.0, 0.0);

                for (s3 = 0; s3 < 2; s3++)
                {
                  for (s4 = 0; s4 < 2; s4++)
                  {

                    if (s1 == s3 && s2 == s4)
                    {

                      ctmp1.r = TN[s3][s4].r - TN0[s3][s4].r;
                      ctmp1.i = TN[s3][s4].i - TN0[s3][s4].i;
                      ctmp2.r = 1.0 - dTN[s1][s2][s4][s3].r;
                      ctmp2.i = -dTN[s1][s2][s4][s3].i;

                      csum1.r += ctmp1.r * ctmp2.r - ctmp1.i * ctmp2.i;
                      csum1.i += ctmp1.r * ctmp2.i + ctmp1.i * ctmp2.r;
                    }
                    else
                    {

                      ctmp1.r = TN[s3][s4].r - TN0[s3][s4].r;
                      ctmp1.i = TN[s3][s4].i - TN0[s3][s4].i;
                      ctmp2.r = -dTN[s1][s2][s4][s3].r;
                      ctmp2.i = -dTN[s1][s2][s4][s3].i;

                      csum1.r += ctmp1.r * ctmp2.r - ctmp1.i * ctmp2.i;
                      csum1.i += ctmp1.r * ctmp2.i + ctmp1.i * ctmp2.r;
                    }

                  } /* s4 */
                }   /* s3 */

                NC_v_eff[s1][s2][Mc_AN][i][j].r += 2.0 * Constraint_NCS_V * csum1.r;
                NC_v_eff[s1][s2][Mc_AN][i][j].i += 2.0 * Constraint_NCS_V * csum1.i;
              }
            }
          }
        }
      }

      /* calculate the penalty functional, Ucs */

      tmp1 = 0.0;

      for (s1 = 0; s1 < 2; s1++)
      {
        for (s2 = 0; s2 < 2; s2++)
        {

          ctmp1.r = TN[s1][s2].r - TN0[s1][s2].r;
          ctmp1.i = TN[s1][s2].i - TN0[s1][s2].i;
          tmp1 += ctmp1.r * ctmp1.r + ctmp1.i * ctmp1.i;
        }
      }

      My_Ucs += Constraint_NCS_V * tmp1;

    } /* if (Constraint_NCS_switch==2 && Constraint_SpinAngle[Gc_AN]==1 ) */

    /************************************************************
     ***********************************************************
     ***********************************************************
     ***********************************************************

           calculate the v_eff for Zeeman term for spin

     ***********************************************************
     ***********************************************************
     ***********************************************************
    ************************************************************/

    else if (Zeeman_NCS_switch == 1 && Constraint_SpinAngle[Gc_AN] == 1)
    {

      theta0 = InitAngle0_Spin[Gc_AN];
      phi0 = InitAngle1_Spin[Gc_AN];

      /* calculate TN */

      TN[0][0].r = 0.0;
      TN[0][1].r = 0.0;
      TN[1][0].r = 0.0;
      TN[1][1].r = 0.0;
      TN[0][0].i = 0.0;
      TN[0][1].i = 0.0;
      TN[1][0].i = 0.0;
      TN[1][1].i = 0.0;

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {

        TN[0][0].r += NC_OcpN[0][0][0][Mc_AN][i][i].r;
        TN[0][1].r += NC_OcpN[0][0][1][Mc_AN][i][i].r;
        TN[1][0].r += NC_OcpN[0][1][0][Mc_AN][i][i].r;
        TN[1][1].r += NC_OcpN[0][1][1][Mc_AN][i][i].r;

        /*
        conjugate complex of TN due to difference
        in the definition between density matrix
        and charge density
        */

        TN[0][0].i -= NC_OcpN[0][0][0][Mc_AN][i][i].i;
        TN[0][1].i -= NC_OcpN[0][0][1][Mc_AN][i][i].i;
        TN[1][0].i -= NC_OcpN[0][1][0][Mc_AN][i][i].i;
        TN[1][1].i -= NC_OcpN[0][1][1][Mc_AN][i][i].i;
      }

      EulerAngle_Spin(1,
                      TN[0][0].r, TN[1][1].r,
                      TN[0][1].r, TN[0][1].i,
                      TN[1][0].r, TN[1][0].i,
                      Nup, Ndn, theta, phi);

      /*
      printf("Nup=   %18.15f\n",Nup[0],Nup[1]);
      printf("Ndn=   %18.15f\n",Ndn[0],Ndn[1]);
      printf("theta= %18.15f\n",theta[0],theta[1]);
      printf("phi=   %18.15f\n",phi[0],phi[1]);
      */

      /* calculate dSx, dSy, dSz */

      dSx[0][0].r = 0.0;
      dSx[0][1].r = 0.5;
      dSx[1][0].r = 0.5;
      dSx[1][1].r = 0.0;

      dSx[0][0].i = 0.0;
      dSx[0][1].i = 0.0;
      dSx[1][0].i = 0.0;
      dSx[1][1].i = 0.0;

      dSy[0][0].r = 0.0;
      dSy[0][1].r = 0.0;
      dSy[1][0].r = 0.0;
      dSy[1][1].r = 0.0;

      dSy[0][0].i = 0.0;
      dSy[0][1].i = -0.5; /* causion for the sign */
      dSy[1][0].i = 0.5;  /* causion for the sign */
      dSy[1][1].i = 0.0;

      dSz[0][0].r = 0.5;
      dSz[0][1].r = 0.0;
      dSz[1][0].r = 0.0;
      dSz[1][1].r = -0.5;

      dSz[0][0].i = 0.0;
      dSz[0][1].i = 0.0;
      dSz[1][0].i = 0.0;
      dSz[1][1].i = 0.0;

      /*
      Calc_dSxyz( TN, dSx, dSy, dSz, Nup, Ndn, theta, phi );

      {

        double Sx,Sy,Sz;

        Sx = 0.5*(Nup[0] - Ndn[0])*sin(theta[0])*cos(phi[0]);
        Sy = 0.5*(Nup[0] - Ndn[0])*sin(theta[0])*sin(phi[0]);
        Sz = 0.5*(Nup[0] - Ndn[0])*cos(theta[0]);

        printf("Sx=%18.15f\n",Sx);
        printf("Sy=%18.15f\n",Sy);
        printf("Sz=%18.15f\n",Sz);

        printf("Re dSx11=%18.15f\n",dSx[0][0].r);
        printf("Re dSx12=%18.15f\n",dSx[0][1].r);
        printf("Re dSx21=%18.15f\n",dSx[1][0].r);
        printf("Re dSx22=%18.15f\n",dSx[1][1].r);

        printf("Im dSx11=%18.15f\n",dSx[0][0].i);
        printf("Im dSx12=%18.15f\n",dSx[0][1].i);
        printf("Im dSx21=%18.15f\n",dSx[1][0].i);
        printf("Im dSx22=%18.15f\n\n",dSx[1][1].i);


        printf("Re dSy11=%18.15f\n",dSy[0][0].r);
        printf("Re dSy12=%18.15f\n",dSy[0][1].r);
        printf("Re dSy21=%18.15f\n",dSy[1][0].r);
        printf("Re dSy22=%18.15f\n",dSy[1][1].r);

        printf("Im dSy11=%18.15f\n",dSy[0][0].i);
        printf("Im dSy12=%18.15f\n",dSy[0][1].i);
        printf("Im dSy21=%18.15f\n",dSy[1][0].i);
        printf("Im dSy22=%18.15f\n\n",dSy[1][1].i);

        printf("Re dSz11=%18.15f\n",dSz[0][0].r);
        printf("Re dSz12=%18.15f\n",dSz[0][1].r);
        printf("Re dSz21=%18.15f\n",dSz[1][0].r);
        printf("Re dSz22=%18.15f\n",dSz[1][1].r);

        printf("Im dSz11=%18.15f\n",dSz[0][0].i);
        printf("Im dSz12=%18.15f\n",dSz[0][1].i);
        printf("Im dSz21=%18.15f\n",dSz[1][0].i);
        printf("Im dSz22=%18.15f\n\n",dSz[1][1].i);

      }


      MPI_Finalize();
      exit(0);
      */

      /* calculate the energy for the Zeeman term for spin, Uzs */

      lx = sin(theta0) * cos(phi0);
      ly = sin(theta0) * sin(phi0);
      lz = cos(theta0);

      Bx = -Mag_Field_Spin * lx;
      By = -Mag_Field_Spin * ly;
      Bz = -Mag_Field_Spin * lz;

      sx = 0.5 * (TN[0][1].r + TN[1][0].r);
      sy = -0.5 * (TN[0][1].i - TN[1][0].i);
      sz = 0.5 * (TN[0][0].r - TN[1][1].r);

      My_Uzs += sx * Bx + sy * By + sz * Bz;

      /*
      printf("Uzs=%15.12f\n",sx*Bx + sy*By + sz*Bz);

      printf("|s|=%18.15f\n",sqrt(sx*sx+sy*sy+sz*sz));
      printf("theta=%18.15f %18.15f\n",theta0/PI*180.0,theta[0]/PI*180.0);
      printf("phi  =%18.15f %18.15f\n",phi0/PI*180.0,phi[0]/PI*180.0);

      printf("sx=%18.15f %18.15f\n",sx,0.5*(Nup[0] - Ndn[0])*sin(theta[0])*cos(phi[0]));
      printf("sy=%18.15f %18.15f\n",sy,0.5*(Nup[0] - Ndn[0])*sin(theta[0])*sin(phi[0]));
      printf("sz=%18.15f %18.15f\n",sz,0.5*(Nup[0] - Ndn[0])*cos(theta[0]));

      printf("Bx=%18.15f\n",Bx);
      printf("By=%18.15f\n",By);
      printf("Bz=%18.15f\n",Bz);
      */

      /* calculate veff by the Zeeman term for spin magnetic moment */

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {

          if (i == j)
          {
            for (s1 = 0; s1 < 2; s1++)
            {
              for (s2 = 0; s2 < 2; s2++)
              {

                NC_v_eff[s1][s2][Mc_AN][i][j].r += Bx * dSx[s1][s2].r + By * dSy[s1][s2].r + Bz * dSz[s1][s2].r;
                NC_v_eff[s1][s2][Mc_AN][i][j].i += Bx * dSx[s1][s2].i + By * dSy[s1][s2].i + Bz * dSz[s1][s2].i;

                /*
                            printf("spin i=%2d j=%2d s1=%2d s2=%2d re=%15.12f im=%15.12f\n",
                                   i,j,s1,s2,
                                   Bx*dSx[s1][s2].r + By*dSy[s1][s2].r + Bz*dSz[s1][s2].r,
                                   Bx*dSx[s1][s2].i + By*dSy[s1][s2].i + Bz*dSz[s1][s2].i );
                */
              }
            }
          }
        }
      }

    } /* else if (Zeeman_NCS_switch==1 && Constraint_SpinAngle[Gc_AN]==1 ) */

    /************************************************************
     ***********************************************************
     ***********************************************************
     ***********************************************************

          calculate the v_eff for Zeeman term for orbital

     ***********************************************************
     ***********************************************************
     ***********************************************************
    ************************************************************/

    if (Zeeman_NCO_switch == 1 && Constraint_SpinAngle[Gc_AN] == 1)
    {

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      theta0 = InitAngle0_Orbital[Gc_AN];
      phi0 = InitAngle1_Orbital[Gc_AN];

      lx = sin(theta0) * cos(phi0);
      ly = sin(theta0) * sin(phi0);
      lz = cos(theta0);

      Lx = Orbital_Moment_XYZ[Gc_AN][0];
      Ly = Orbital_Moment_XYZ[Gc_AN][1];
      Lz = Orbital_Moment_XYZ[Gc_AN][2];

      L = sqrt(Lx * Lx + Ly * Ly + Lz * Lz);

      A = -0.5 * Mag_Field_Orbital * lx;
      B = -0.5 * Mag_Field_Orbital * ly;
      C = -0.5 * Mag_Field_Orbital * lz;

      for (i = 0; i < tno0; i++)
      {
        for (j = 0; j < tno0; j++)
        {

          tmp = A * OLP_L[0][Mc_AN][0][i][j] + B * OLP_L[1][Mc_AN][0][i][j] + C * OLP_L[2][Mc_AN][0][i][j];

          NC_v_eff[0][0][Mc_AN][i][j].i += tmp;
          NC_v_eff[1][1][Mc_AN][i][j].i += tmp;
        }
      }

      My_Uzo += A * Lx + B * Ly + C * Lz;

    } /* if (Zeeman_NCO_switch==1 && Constraint_SpinAngle[Gc_AN]==1 ) */

    /* measure elapsed time */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

  } /* Mc_AN */

  /****************************************************
    MPI: energy contributions
  ****************************************************/

  MPI_Allreduce(&My_Ucs, &tmp1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  ECE[9] = tmp1;

  MPI_Allreduce(&My_Uzs, &tmp1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  ECE[10] = tmp1;

  MPI_Allreduce(&My_Uzo, &tmp1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  ECE[11] = tmp1;

  /****************************************************
    MPI: NC_v_eff
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID = 0; ID < numprocs; ID++)
  {

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID != 0)
    {
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS] != 0)
      {

        size1 = 0;
        for (n = 0; n < F_Snd_Num[IDS]; n++)
        {
          Mc_AN = Snd_MAN[IDS][n];
          Gc_AN = Snd_GAN[IDS][n];
          Cwan = WhatSpecies[Gc_AN];
          tno1 = Spe_Total_NO[Cwan];
          size1 += tno1 * tno1;
        }
        size1 = 8 * size1;
        Snd_Size[IDS] = size1;
        MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else
      {
        Snd_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR] != 0)
      {
        MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
        Rcv_Size[IDR] = size2;
      }
      else
      {
        Rcv_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS] != 0)
        MPI_Wait(&request, &stat);
    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID = 0; ID < numprocs; ID++)
  {

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID != 0)
    {

      /*****************************
              sending of data
      *****************************/

      if (F_Snd_Num[IDS] != 0)
      {

        size1 = Snd_Size[IDS];

        /* allocation of array */

        tmp_array = (double *)malloc(sizeof(double) * size1);

        /* multidimentional array to vector array */

        num = 0;

        for (n = 0; n < F_Snd_Num[IDS]; n++)
        {
          Mc_AN = Snd_MAN[IDS][n];
          Gc_AN = Snd_GAN[IDS][n];
          Cwan = WhatSpecies[Gc_AN];
          tno1 = Spe_Total_NO[Cwan];
          for (i = 0; i < tno1; i++)
          {
            for (j = 0; j < tno1; j++)
            {
              tmp_array[num] = NC_v_eff[0][0][Mc_AN][i][j].r;
              num++;
              tmp_array[num] = NC_v_eff[0][0][Mc_AN][i][j].i;
              num++;
              tmp_array[num] = NC_v_eff[1][1][Mc_AN][i][j].r;
              num++;
              tmp_array[num] = NC_v_eff[1][1][Mc_AN][i][j].i;
              num++;
              tmp_array[num] = NC_v_eff[0][1][Mc_AN][i][j].r;
              num++;
              tmp_array[num] = NC_v_eff[0][1][Mc_AN][i][j].i;
              num++;
              tmp_array[num] = NC_v_eff[1][0][Mc_AN][i][j].r;
              num++;
              tmp_array[num] = NC_v_eff[1][0][Mc_AN][i][j].i;
              num++;
            }
          }
        }

        MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR] != 0)
      {

        size2 = Rcv_Size[IDR];

        /* allocation of array */
        tmp_array2 = (double *)malloc(sizeof(double) * size2);

        MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

        num = 0;
        Mc_AN = F_TopMAN[IDR] - 1;
        for (n = 0; n < F_Rcv_Num[IDR]; n++)
        {
          Mc_AN++;
          Gc_AN = Rcv_GAN[IDR][n];
          Cwan = WhatSpecies[Gc_AN];
          tno1 = Spe_Total_NO[Cwan];
          for (i = 0; i < tno1; i++)
          {
            for (j = 0; j < tno1; j++)
            {
              NC_v_eff[0][0][Mc_AN][i][j].r = tmp_array2[num];
              num++;
              NC_v_eff[0][0][Mc_AN][i][j].i = tmp_array2[num];
              num++;
              NC_v_eff[1][1][Mc_AN][i][j].r = tmp_array2[num];
              num++;
              NC_v_eff[1][1][Mc_AN][i][j].i = tmp_array2[num];
              num++;
              NC_v_eff[0][1][Mc_AN][i][j].r = tmp_array2[num];
              num++;
              NC_v_eff[0][1][Mc_AN][i][j].i = tmp_array2[num];
              num++;
              NC_v_eff[1][0][Mc_AN][i][j].r = tmp_array2[num];
              num++;
              NC_v_eff[1][0][Mc_AN][i][j].i = tmp_array2[num];
              num++;
            }
          }
        }

        /* freeing of array */
        free(tmp_array2);
      }

      if (F_Snd_Num[IDS] != 0)
      {
        MPI_Wait(&request, &stat);
        free(tmp_array); /* freeing of array */
      }
    }
  }

  /* freeing of Snd_Size and Rcv_Size */

  free(Snd_Size);
  free(Rcv_Size);
}

void Output_Collinear_OcpN()
{
  int Gc_AN, l, m, mul, spin, wan1, i, j, k, base;
  int tno0, Mc_AN, num, l1, mul1, m1, to1, Ns;
  int numprocs, myid, ID, tag = 999;
  double *tmp_vec;
  double sum, ele_max;
  FILE *fp_DM_onsite;
  char *Name_Angular[20][10];
  char *Name_Multiple[20];
  char file_DM_onsite[YOUSO10];
  double **a, *ko;
  char buf[fp_bsize]; /* setvbuf */

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  Ns = List_YOUSO[7] + 2;
  a = (double **)malloc(sizeof(double *) * Ns);
  for (i = 0; i < Ns; i++)
  {
    a[i] = (double *)malloc(sizeof(double) * Ns);
  }

  ko = (double *)malloc(sizeof(double) * Ns);

  tmp_vec = (double *)malloc(sizeof(double) * Ns * Ns * (SpinP_switch + 1));

  if (myid == Host_ID)
  {

    sprintf(file_DM_onsite, "%s%s.DM_onsite", filepath, filename);

    if ((fp_DM_onsite = fopen(file_DM_onsite, "w")) != NULL)
    {

      setvbuf(fp_DM_onsite, buf, _IOFBF, fp_bsize); /* setvbuf */

      fprintf(fp_DM_onsite, "\n\n\n\n");
      fprintf(fp_DM_onsite, "***********************************************************\n");
      fprintf(fp_DM_onsite, "***********************************************************\n");
      fprintf(fp_DM_onsite, "       Occupation Number in LDA+U and Constraint DFT       \n");
      fprintf(fp_DM_onsite, "                                                           \n");
      fprintf(fp_DM_onsite, "    Eigenvalues and eigenvectors for a matrix consisting   \n");
      fprintf(fp_DM_onsite, "           of occupation numbers on each site              \n");
      fprintf(fp_DM_onsite, "***********************************************************\n");
      fprintf(fp_DM_onsite, "***********************************************************\n\n");

      /* decomposed Mulliken charge */
      Name_Angular[0][0] = "s          ";
      Name_Angular[1][0] = "px         ";
      Name_Angular[1][1] = "py         ";
      Name_Angular[1][2] = "pz         ";
      Name_Angular[2][0] = "d3z^2-r^2  ";
      Name_Angular[2][1] = "dx^2-y^2   ";
      Name_Angular[2][2] = "dxy        ";
      Name_Angular[2][3] = "dxz        ";
      Name_Angular[2][4] = "dyz        ";
      Name_Angular[3][0] = "f5z^2-3r^2 ";
      Name_Angular[3][1] = "f5xz^2-xr^2";
      Name_Angular[3][2] = "f5yz^2-yr^2";
      Name_Angular[3][3] = "fzx^2-zy^2 ";
      Name_Angular[3][4] = "fxyz       ";
      Name_Angular[3][5] = "fx^3-3*xy^2";
      Name_Angular[3][6] = "f3yx^2-y^3 ";
      Name_Angular[4][0] = "g1         ";
      Name_Angular[4][1] = "g2         ";
      Name_Angular[4][2] = "g3         ";
      Name_Angular[4][3] = "g4         ";
      Name_Angular[4][4] = "g5         ";
      Name_Angular[4][5] = "g6         ";
      Name_Angular[4][6] = "g7         ";
      Name_Angular[4][7] = "g8         ";
      Name_Angular[4][8] = "g9         ";

      Name_Multiple[0] = " 0";
      Name_Multiple[1] = " 1";
      Name_Multiple[2] = " 2";
      Name_Multiple[3] = " 3";
      Name_Multiple[4] = " 4";
      Name_Multiple[5] = " 5";
    }
  }

  for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++)
  {
    wan1 = WhatSpecies[Gc_AN];
    ID = G2ID[Gc_AN];
    Mc_AN = F_G2M[Gc_AN];

    if (myid == ID)
    {

      num = 0;
      for (spin = 0; spin <= SpinP_switch; spin++)
      {
        for (i = 0; i < Spe_Total_NO[wan1]; i++)
        {
          for (j = 0; j < Spe_Total_NO[wan1]; j++)
          {
            tmp_vec[num] = DM_onsite[0][spin][Mc_AN][i][j];
            num++;
          }
        }
      }

      if (myid != Host_ID)
      {
        MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
        MPI_Isend(&tmp_vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
      }
    }

    else if (myid == Host_ID)
    {
      MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
      MPI_Recv(&tmp_vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
    }

    if (myid == Host_ID)
    {

      fprintf(fp_DM_onsite, "\n %4d %4s\n", Gc_AN, SpeName[wan1]);

      num = 0;
      for (spin = 0; spin <= SpinP_switch; spin++)
      {

        fprintf(fp_DM_onsite, "\n     spin=%2d\n\n", spin);

        ele_max = 0.0;

        for (i = 0; i < Spe_Total_NO[wan1]; i++)
        {
          for (j = 0; j < Spe_Total_NO[wan1]; j++)
          {
            a[i + 1][j + 1] = tmp_vec[num];
            num++;
            if (ele_max < fabs(a[i + 1][j + 1]))
              ele_max = fabs(a[i + 1][j + 1]);
          }
        }

        if (1.0e-13 < ele_max)
        {
          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            for (j = 0; j < Spe_Total_NO[wan1]; j++)
            {
              a[i + 1][j + 1] = a[i + 1][j + 1] / ele_max;
            }
          }
        }

        Eigen_lapack(a, ko, Spe_Total_NO[wan1], Spe_Total_NO[wan1]);

        if (1.0e-13 < ele_max)
        {
          for (i = 0; i < Spe_Total_NO[wan1]; i++)
          {
            ko[i + 1] = ko[i + 1] * ele_max;
          }
        }

        sum = 0.0;
        for (i = 0; i < Spe_Total_NO[wan1]; i++)
        {
          sum += ko[i + 1];
        }
        fprintf(fp_DM_onsite, "  Sum = %15.12f\n", sum);

        base = 8;

        for (k = 0; k < ((Spe_Total_NO[wan1] - 1) / base + 1); k++)
        {

          fprintf(fp_DM_onsite, "\n");
          fprintf(fp_DM_onsite, "                 ");
          for (i = k * base; i < (k * base + base); i++)
          {
            if (i < Spe_Total_NO[wan1])
            {
              fprintf(fp_DM_onsite, " %3d    ", i + 1);
            }
          }
          fprintf(fp_DM_onsite, "\n");

          fprintf(fp_DM_onsite, "  Individual     ");

          for (i = k * base; i < (k * base + base); i++)
          {
            if (i < Spe_Total_NO[wan1])
            {
              fprintf(fp_DM_onsite, "%7.4f ", ko[i + 1]);
            }
          }

          fprintf(fp_DM_onsite, "\n\n");

          i = 0;
          for (l = 0; l <= Supported_MaxL; l++)
          {
            for (mul = 0; mul < Spe_Num_Basis[wan1][l]; mul++)
            {
              for (m = 0; m < (2 * l + 1); m++)
              {
                fprintf(fp_DM_onsite, "  %s%s  ", Name_Angular[l][m], Name_Multiple[mul]);

                for (j = k * base; j < (k * base + base); j++)
                {
                  if (j < Spe_Total_NO[wan1])
                  {
                    fprintf(fp_DM_onsite, "%7.4f ", a[i + 1][j + 1]);
                  }
                }

                fprintf(fp_DM_onsite, "\n");
                i++;
              }
            }
          }
        }
      }
    }
  }

  if (myid == Host_ID)
  {
    fclose(fp_DM_onsite);
  }

  /* freeing of arrays */

  for (i = 0; i < Ns; i++)
  {
    free(a[i]);
  }
  free(a);

  free(ko);
  free(tmp_vec);
}

void Output_NonCollinear_OcpN()
{
  int Gc_AN, l, m, mul, spin, wan1, i, j, k;
  int tno0, Mc_AN, num, l1, mul1, m1, to1, Ns;
  int i1, j1, num0, num1;
  int numprocs, myid, ID, tag = 999;
  double *tmp_vec;
  double sum;
  FILE *fp_DM_onsite;
  char *Name_Angular[Supported_MaxL + 1][2 * (Supported_MaxL + 1) + 1];
  char *Name_Multiple[20];
  char file_DM_onsite[YOUSO10];
  dcomplex **a;
  double *ko;
  char buf[fp_bsize]; /* setvbuf */

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* allocation of arrays */

  Ns = 2 * List_YOUSO[7] + 2;
  a = (dcomplex **)malloc(sizeof(dcomplex *) * Ns);
  for (i = 0; i < Ns; i++)
  {
    a[i] = (dcomplex *)malloc(sizeof(dcomplex) * Ns);
  }

  ko = (double *)malloc(sizeof(double) * Ns);

  tmp_vec = (double *)malloc(sizeof(double) * Ns * Ns * 8);

  if (myid == Host_ID)
  {

    sprintf(file_DM_onsite, "%s%s.DM_onsite", filepath, filename);

    if ((fp_DM_onsite = fopen(file_DM_onsite, "w")) != NULL)
    {

      setvbuf(fp_DM_onsite, buf, _IOFBF, fp_bsize); /* setvbuf */

      fprintf(fp_DM_onsite, "\n\n\n\n");
      fprintf(fp_DM_onsite, "***********************************************************\n");
      fprintf(fp_DM_onsite, "***********************************************************\n");
      fprintf(fp_DM_onsite, "       Occupation Number in LDA+U and Constraint DFT       \n");
      fprintf(fp_DM_onsite, "                                                           \n");
      fprintf(fp_DM_onsite, "    Eigenvalues and eigenvectors for a matrix consisting   \n");
      fprintf(fp_DM_onsite, "           of occupation numbers on each site              \n");
      fprintf(fp_DM_onsite, "***********************************************************\n");
      fprintf(fp_DM_onsite, "***********************************************************\n\n");

      Name_Angular[0][0] = "s          ";
      Name_Angular[1][0] = "px         ";
      Name_Angular[1][1] = "py         ";
      Name_Angular[1][2] = "pz         ";
      Name_Angular[2][0] = "d3z^2-r^2  ";
      Name_Angular[2][1] = "dx^2-y^2   ";
      Name_Angular[2][2] = "dxy        ";
      Name_Angular[2][3] = "dxz        ";
      Name_Angular[2][4] = "dyz        ";
      Name_Angular[3][0] = "f5z^2-3r^2 ";
      Name_Angular[3][1] = "f5xz^2-xr^2";
      Name_Angular[3][2] = "f5yz^2-yr^2";
      Name_Angular[3][3] = "fzx^2-zy^2 ";
      Name_Angular[3][4] = "fxyz       ";
      Name_Angular[3][5] = "fx^3-3*xy^2";
      Name_Angular[3][6] = "f3yx^2-y^3 ";
      Name_Angular[4][0] = "g1         ";
      Name_Angular[4][1] = "g2         ";
      Name_Angular[4][2] = "g3         ";
      Name_Angular[4][3] = "g4         ";
      Name_Angular[4][4] = "g5         ";
      Name_Angular[4][5] = "g6         ";
      Name_Angular[4][6] = "g7         ";
      Name_Angular[4][7] = "g8         ";
      Name_Angular[4][8] = "g9         ";

      Name_Multiple[0] = " 0";
      Name_Multiple[1] = " 1";
      Name_Multiple[2] = " 2";
      Name_Multiple[3] = " 3";
      Name_Multiple[4] = " 4";
      Name_Multiple[5] = " 5";
    }
  }

  for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++)
  {
    wan1 = WhatSpecies[Gc_AN];
    ID = G2ID[Gc_AN];
    Mc_AN = F_G2M[Gc_AN];

    if (myid == ID)
    {

      num = 0;
      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {
          tmp_vec[num] = NC_OcpN[0][0][0][Mc_AN][i][j].r;
          num++;
          tmp_vec[num] = NC_OcpN[0][0][0][Mc_AN][i][j].i;
          num++;
          tmp_vec[num] = NC_OcpN[0][1][1][Mc_AN][i][j].r;
          num++;
          tmp_vec[num] = NC_OcpN[0][1][1][Mc_AN][i][j].i;
          num++;
          tmp_vec[num] = NC_OcpN[0][0][1][Mc_AN][i][j].r;
          num++;
          tmp_vec[num] = NC_OcpN[0][0][1][Mc_AN][i][j].i;
          num++;
          tmp_vec[num] = NC_OcpN[0][1][0][Mc_AN][i][j].r;
          num++;
          tmp_vec[num] = NC_OcpN[0][1][0][Mc_AN][i][j].i;
          num++;
        }
      }

      if (myid != Host_ID)
      {
        MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
        MPI_Isend(&tmp_vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
      }
    }

    else if (myid == Host_ID)
    {
      MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
      MPI_Recv(&tmp_vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
    }

    if (myid == Host_ID)
    {

      fprintf(fp_DM_onsite, "\n%4d %4s\n", Gc_AN, SpeName[wan1]);

      k = Spe_Total_NO[wan1];
      num = 0;

      for (i = 0; i < Spe_Total_NO[wan1]; i++)
      {
        for (j = 0; j < Spe_Total_NO[wan1]; j++)
        {
          /* rnd(1.0e-14) is a prescription to stabilize the lapack routines */
          a[i + 1][j + 1].r = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + 1][j + 1].i = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + k + 1][j + k + 1].r = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + k + 1][j + k + 1].i = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + 1][j + k + 1].r = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + 1][j + k + 1].i = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + k + 1][j + 1].r = tmp_vec[num] + rnd(1.0e-14);
          num++;
          a[i + k + 1][j + 1].i = tmp_vec[num] + rnd(1.0e-14);
          num++;
        }
      }

      EigenBand_lapack(a, ko, 2 * k, 2 * k, 1);

      sum = 0.0;
      for (i = 0; i < 2 * k; i++)
      {
        sum += ko[i + 1];
      }
      fprintf(fp_DM_onsite, "    Sum of occupancy numbers = %15.12f\n", sum);

      num0 = 2;
      num1 = 2 * k / num0 + 1 * ((2 * k) % num0 != 0);

      for (i = 1; i <= num1; i++)
      {
        fprintf(fp_DM_onsite, "\n");

        for (i1 = -2; i1 <= 0; i1++)
        {

          fprintf(fp_DM_onsite, "                     ");

          for (j = 1; j <= num0; j++)
          {
            j1 = num0 * (i - 1) + j;

            if (j1 <= 2 * k)
            {
              if (i1 == -2)
              {
                fprintf(fp_DM_onsite, " %4d", j1);
                fprintf(fp_DM_onsite, "                                   ");
              }
              else if (i1 == -1)
              {
                fprintf(fp_DM_onsite, "   %8.5f", ko[j1]);
                fprintf(fp_DM_onsite, "                             ");
              }

              else if (i1 == 0)
              {
                fprintf(fp_DM_onsite, "     Re(U)");
                fprintf(fp_DM_onsite, "     Im(U)");
                fprintf(fp_DM_onsite, "     Re(D)");
                fprintf(fp_DM_onsite, "     Im(D)");
              }
            }
          }
          fprintf(fp_DM_onsite, "\n");
          if (i1 == -1)
            fprintf(fp_DM_onsite, "\n");
          if (i1 == 0)
            fprintf(fp_DM_onsite, "\n");
        }

        for (l = 0; l <= Supported_MaxL; l++)
        {
          for (mul = 0; mul < Spe_Num_Basis[wan1][l]; mul++)
          {
            for (m = 0; m < (2 * l + 1); m++)
            {

              if (l == 0 && mul == 0 && m == 0)
                fprintf(fp_DM_onsite, "%4d %3s %s %s",
                        Gc_AN, SpeName[wan1], Name_Multiple[mul], Name_Angular[l][m]);
              else
                fprintf(fp_DM_onsite, "         %s %s",
                        Name_Multiple[mul], Name_Angular[l][m]);

              for (j = 1; j <= num0; j++)
              {

                j1 = num0 * (i - 1) + j;

                if (0 < i1 && j1 <= 2 * k)
                {
                  fprintf(fp_DM_onsite, "  %8.5f", a[i1][j1].r);
                  fprintf(fp_DM_onsite, "  %8.5f", a[i1][j1].i);
                  fprintf(fp_DM_onsite, "  %8.5f", a[i1 + k][j1].r);
                  fprintf(fp_DM_onsite, "  %8.5f", a[i1 + k][j1].i);
                }
              }

              fprintf(fp_DM_onsite, "\n");
              if (i1 == -1)
                fprintf(fp_DM_onsite, "\n");
              if (i1 == 0)
                fprintf(fp_DM_onsite, "\n");

              i1++;
            }
          }
        }
      }
    }
  }

  if (myid == Host_ID)
  {
    fclose(fp_DM_onsite);
  }

  /* freeing of arrays */

  for (i = 0; i < Ns; i++)
  {
    free(a[i]);
  }
  free(a);

  free(ko);
  free(tmp_vec);
}

void Calc_dTN(int constraint_flag,
              dcomplex TN[2][2],
              dcomplex dTN[2][2][2][2],
              dcomplex U[2][2],
              double theta[2], double phi[2])
{
  double dphi0, dtheta0;
  dcomplex tmp1, tmp2, tmp3;
  dcomplex Nup, Ndn;
  dcomplex I0, I1;

#ifdef c_complex
  double complex d11, d12, d21, d22;
  double complex dphi11, dphi12, dphi21, dphi22;
  double complex ctmp1, ctmp2, ctmp3;
  double complex cphi, ctheta;
  double complex cot, sit, cop, sip;
  double complex dtheta11, dtheta12, dtheta21, dtheta22;
  double complex dNup11, dNup12, dNup21, dNup22;
  double complex dNdn11, dNdn12, dNdn21, dNdn22;
#else
  dcomplex d11, d12, d21, d22;
  dcomplex dphi11, dphi12, dphi21, dphi22;
  dcomplex ctmp1, ctmp2, ctmp3;
  dcomplex cphi, ctheta;
  dcomplex cot, sit, cop, sip;
  dcomplex dtheta11, dtheta12, dtheta21, dtheta22;
  dcomplex dNup11, dNup12, dNup21, dNup22;
  dcomplex dNdn11, dNdn12, dNdn21, dNdn22;
#endif

  dcomplex ctmp4, ctmp5, ctmp6, ctmp7;
  dcomplex coe0;
  dcomplex ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8;

  I0 = Complex(0.0, 1.0);
  I1 = Complex(0.0, -1.0);

#ifdef c_complex
  d11 = TN[0][0].r + TN[0][0].i * I;
  d12 = TN[0][1].r + TN[0][1].i * I;
  d21 = TN[1][0].r + TN[1][0].i * I;
  d22 = TN[1][1].r + TN[1][1].i * I;
  cphi = phi[0] + phi[1] * I;
  ctheta = theta[0] + theta[1] * I;
  cot = ccos(ctheta);
  sit = csin(ctheta);
  cop = ccos(cphi);
  sip = csin(cphi);
#else
  d11 = TN[0][0];
  d12 = TN[0][1];
  d21 = TN[1][0];
  d22 = TN[1][1];
  cphi = Complex(phi[0], phi[1]);
  ctheta = Complex(theta[0], theta[1]);
  cot = Ccos(ctheta);
  sit = Csin(ctheta);
  cop = Ccos(cphi);
  sip = Csin(cphi);
#endif

  /* calculate dphi */

#ifdef c_complex

  dphi12 = 0.5 * d21 / (d12 * d21) * I;
  dphi21 = -0.5 * d12 / (d12 * d21) * I;

#else

  dphi11 = Complex(0.0, 0.0);
  dphi22 = Complex(0.0, 0.0);

  ctmp1 = Complex(0.0, 0.5);
  ctmp2 = Complex(0.0, -0.5);
  ctmp3 = Cmul(d12, d21);

  ct1 = Cdiv(d21, ctmp3);
  ct2 = Cdiv(d12, ctmp3);
  dphi12 = Cmul(ctmp1, ct1);
  dphi21 = Cmul(ctmp2, ct2);

#endif

  /*
  dphi12 = Cmul( ctmp1, Cdiv(d21,ctmp3) );
  dphi21 = Cmul( ctmp2, Cdiv(d12,ctmp3) );
  */

  /* calculate dtheta */

#ifdef c_complex

  ctmp1 = (d11 - d22) * (d11 - d22);
  ctmp2 = d12 * cexp(I * cphi) + d21 * cexp(-I * cphi);
  ctmp3 = ctmp1 / (ctmp1 + ctmp2 * ctmp2);

  dtheta11 = -ctmp3 * ctmp2 / ctmp1;
  dtheta22 = ctmp3 * ctmp2 / ctmp1;
  dtheta12 = ctmp3 * (cexp(I * cphi) + I * d12 * dphi12 * cexp(I * cphi) - I * d21 * dphi12 * cexp(-I * cphi)) / (d11 - d22);
  dtheta21 = ctmp3 * (cexp(-I * cphi) + I * d12 * dphi21 * cexp(I * cphi) - I * d21 * dphi21 * cexp(-I * cphi)) / (d11 - d22);

#else

  ct1 = Csub(d11, d22);
  ct2 = Csub(d11, d22);
  ctmp1 = Cmul(ct1, ct2);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp2 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp2, ctmp2);
  ct2 = Cadd(ctmp1, ct1);
  ctmp3 = Cdiv(ctmp1, ct2);

  ct1 = Cmul(ctmp3, ctmp2);
  ct2 = Cdiv(ct1, ctmp1);
  dtheta11 = RCmul(-1.0, ct1);

  ct1 = Cmul(ctmp3, ctmp2);
  dtheta22 = Cdiv(ct1, ctmp1);

  ct1 = Cmul(I0, cphi);
  ctmp4 = Cexp(ct1);

  ct1 = Cmul(I0, d12);
  ct2 = Cmul(ct1, dphi12);
  ct3 = Cmul(I0, cphi);
  ct4 = Cexp(ct3);
  ctmp5 = Cmul(ct2, ct3);

  ct1 = Cmul(I0, d21);
  ct2 = Cmul(ct1, dphi12);
  ct3 = Cmul(I1, cphi);
  ct4 = Cexp(ct3);
  ctmp6 = Cmul(ct2, ct4);

  ct1 = Csub(d11, d22);
  ctmp7 = Cdiv(ctmp3, ct1);

  ct1 = Cadd(ctmp4, ctmp5);
  ct2 = Csub(ct1, ctmp6);
  dtheta12 = Cmul(ctmp7, ct2);

  ct1 = Cmul(I1, cphi);
  ctmp4 = Cexp(ct1);

  ct1 = Cmul(I0, d12);
  ct2 = Cmul(ct1, dphi21);
  ct3 = Cmul(I0, cphi);
  ct4 = Cexp(ct3);
  ctmp5 = Cmul(ct2, ct4);

  ct1 = Cmul(I0, d21);
  ct2 = Cmul(ct1, dphi21);
  ct3 = Cmul(I1, cphi);
  ct4 = Cexp(ct3);
  ctmp6 = Cmul(ct2, ct4);

  ct1 = Csub(d11, d22);
  ctmp7 = Cdiv(ctmp3, ct1);

  ct1 = Cadd(ctmp4, ctmp5);
  ct2 = Csub(ct1, ctmp6);
  dtheta21 = Cmul(ctmp7, ct2);

#endif

  /*
  ctmp1 = Cmul( Csub(d11, d22), Csub(d11, d22) );
  ctmp2 = Cadd( Cmul(d12, Cexp(Cmul(I0,cphi))), Cmul(d21, Cexp(Cmul(I1,cphi))) );
  ctmp3 = Cdiv( ctmp1, Cadd(ctmp1, Cmul(ctmp2,ctmp2)) );

  dtheta11 = RCmul(-1.0, Cdiv( Cmul(ctmp3,ctmp2), ctmp1));
  dtheta22 = Cdiv( Cmul(ctmp3,ctmp2), ctmp1);

  ctmp4 = Cexp(Cmul(I0,cphi));
  ctmp5 = Cmul(Cmul(Cmul(I0,d12),dphi12),Cexp(Cmul(I0,cphi)));
  ctmp6 = Cmul(Cmul(Cmul(I0,d21),dphi12),Cexp(Cmul(I1,cphi)));
  ctmp7 = Cdiv(ctmp3, Csub(d11,d22));
  dtheta12 = Cmul(ctmp7, Csub(Cadd(ctmp4,ctmp5),ctmp6));

  ctmp4 = Cexp(Cmul(I1,cphi));
  ctmp5 = Cmul(Cmul(Cmul(I0,d12),dphi21),Cexp(Cmul(I0,cphi)));
  ctmp6 = Cmul(Cmul(Cmul(I0,d21),dphi21),Cexp(Cmul(I1,cphi)));
  ctmp7 = Cdiv(ctmp3, Csub(d11,d22));
  dtheta21 = Cmul(ctmp7, Csub(Cadd(ctmp4,ctmp5),ctmp6));
  */

  /* calculate dNup */

#ifdef c_complex

  dNup11 = 0.5 * ((1.0 + 0.0 * I) + cot - (d11 - d22) * sit * dtheta11 + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta11);

  dNup22 = 0.5 * ((1.0 + 0.0 * I) - cot - (d11 - d22) * sit * dtheta22 + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta22);

  dNup12 = 0.5 * (cexp(I * cphi) * sit - (d11 - d22) * sit * dtheta12 + I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi12 * sit + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta12 * cot);

  dNup21 = 0.5 * (cexp(-I * cphi) * sit - (d11 - d22) * sit * dtheta21 + I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi21 * sit + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta21 * cot);

#else

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Cadd(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta11);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta11);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  dNup11 = RCmul(0.5, ct2);

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Csub(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta22);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta22);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  dNup22 = RCmul(0.5, ct2);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp5 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta12);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  ct3 = Cadd(ct2, ctmp6);
  dNup12 = RCmul(0.5, ct3);

  ct1 = Cmul(I1, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d21, ct2);
  ct4 = Cmul(I0, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d12, ct5);
  ctmp5 = Cadd(ct5, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta21);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  ct3 = Cadd(ct2, ctmp6);
  dNup21 = RCmul(0.5, ct3);

#endif

  /*
  ctmp1 = Cadd(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta11);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta11);
  dNup11 = RCmul(0.5,Cadd(Csub(ctmp1, ctmp2),ctmp4));

  ctmp1 = Csub(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta22);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta22);
  dNup22 = RCmul(0.5,Cadd(Csub(ctmp1, ctmp2),ctmp4));

  ctmp1 = Cmul(Cexp(Cmul(I0,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta12);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi12);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta12);
  dNup12 = RCmul(0.5,Cadd(Cadd(Csub(ctmp1,ctmp2),ctmp4),ctmp6));

  ctmp1 = Cmul(Cexp(Cmul(I1,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta21);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi21);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta21);
  dNup21 = RCmul(0.5,Cadd(Cadd(Csub(ctmp1,ctmp2),ctmp4),ctmp6));
  */

  /* calculate dNdn */

#ifdef c_complex

  dNdn11 = 0.5 * ((1.0 + 0.0 * I) - cot + (d11 - d22) * sit * dtheta11 - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta11);

  dNdn22 = 0.5 * ((1.0 + 0.0 * I) + cot + (d11 - d22) * sit * dtheta22 - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta22);

  dNdn12 = 0.5 * (-cexp(I * cphi) * sit + (d11 - d22) * sit * dtheta12 - I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi12 * sit - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta12 * cot);

  dNdn21 = 0.5 * (-cexp(-I * cphi) * sit + (d11 - d22) * sit * dtheta21 - I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi21 * sit - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta21 * cot);

#else

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Csub(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta11);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta11);

  ct1 = Cadd(ctmp1, ctmp2);
  ct2 = Csub(ct1, ctmp4);
  dNdn11 = RCmul(0.5, ct2);

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Cadd(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta22);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta22);

  ct1 = Cadd(ctmp1, ctmp2);
  ct2 = Csub(ct1, ctmp4);
  dNdn22 = RCmul(0.5, ct2);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp5 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta12);

  ct1 = Csub(ctmp2, ctmp1);
  ct2 = Csub(ct1, ctmp4);
  ct3 = Csub(ct2, ctmp6);
  dNdn12 = RCmul(0.5, ct3);

  ct1 = Cmul(I1, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp5 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta21);

  ct1 = Csub(ctmp2, ctmp1);
  ct2 = Csub(ct1, ctmp4);
  ct3 = Csub(ct2, ctmp6);
  dNdn21 = RCmul(0.5, ct3);

#endif

  /*
  ctmp1 = Csub(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta11);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta11);
  dNdn11 = RCmul(0.5,Csub(Cadd(ctmp1, ctmp2),ctmp4));

  ctmp1 = Cadd(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta22);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta22);
  dNdn22 = RCmul(0.5,Csub(Cadd(ctmp1, ctmp2),ctmp4));

  ctmp1 = Cmul(Cexp(Cmul(I0,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta12);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi12);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta12);
  dNdn12 = RCmul(0.5,Csub(Csub(Csub(ctmp2,ctmp1),ctmp4),ctmp6));

  ctmp1 = Cmul(Cexp(Cmul(I1,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta21);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi21);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta21);
  dNdn21 = RCmul(0.5,Csub(Csub(Csub(ctmp2,ctmp1),ctmp4),ctmp6));
  */

  /*
  printf("dNdn11.r=%15.12f dNdn11.i=%15.12f\n",creal(dNdn11),cimag(dNdn11));
  printf("dNdn22.r=%15.12f dNdn22.i=%15.12f\n",creal(dNdn22),cimag(dNdn22));
  printf("dNdn12.r=%15.12f dNdn12.i=%15.12f\n",creal(dNdn12),cimag(dNdn12));
  printf("dNdn21.r=%15.12f dNdn21.i=%15.12f\n",creal(dNdn21),cimag(dNdn21));
  */

  /* calculate dTN11 */

#ifdef c_complex

  if (constraint_flag == 1)
  {
    Nup.r = creal(dNup11);
    Nup.i = cimag(dNup11);
    Ndn.r = creal(dNdn11);
    Ndn.i = cimag(dNdn11);
  }
  else if (constraint_flag == 2)
  {
    Nup.r = creal(0.5 * (dNup11 + dNdn11));
    Nup.i = cimag(0.5 * (dNup11 + dNdn11));
    Ndn.r = creal(0.5 * (dNup11 + dNdn11));
    Ndn.i = cimag(0.5 * (dNup11 + dNdn11));
  }

#else

  if (constraint_flag == 1)
  {
    Nup = dNup11;
    Ndn = dNdn11;
  }
  else if (constraint_flag == 2)
  {
    Nup.r = 0.5 * (dNup11.r + dNdn11.r);
    Nup.i = 0.5 * (dNup11.i + dNdn11.i);
    Ndn.r = 0.5 * (dNup11.r + dNdn11.r);
    Ndn.i = 0.5 * (dNup11.i + dNdn11.i);
  }

#endif

  dTN[0][0][0][0].r = Nup.r * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn.r * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

  dTN[0][0][0][0].i = 0.0;

  dTN[0][0][0][1].r = Nup.r * (U[0][0].r * U[0][1].r + U[0][0].i * U[0][1].i) + Ndn.r * (U[1][0].r * U[1][1].r + U[1][0].i * U[1][1].i);

  dTN[0][0][0][1].i = Nup.r * (-U[0][0].i * U[0][1].r + U[0][0].r * U[0][1].i) + Ndn.r * (-U[1][0].i * U[1][1].r + U[1][0].r * U[1][1].i);

  dTN[0][0][1][0].r = Nup.r * (U[0][1].r * U[0][0].r + U[0][1].i * U[0][0].i) + Ndn.r * (U[1][1].r * U[1][0].r + U[1][1].i * U[1][0].i);

  dTN[0][0][1][0].i = Nup.r * (-U[0][1].i * U[0][0].r + U[0][1].r * U[0][0].i) + Ndn.r * (-U[1][1].i * U[1][0].r + U[1][1].r * U[1][0].i);

  dTN[0][0][1][1].r = Nup.r * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn.r * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

  dTN[0][0][1][1].i = 0.0;

  /* calculate dTN22 */

#ifdef c_complex

  if (constraint_flag == 1)
  {
    Nup.r = creal(dNup22);
    Nup.i = cimag(dNup22);
    Ndn.r = creal(dNdn22);
    Ndn.i = cimag(dNdn22);
  }
  else if (constraint_flag == 2)
  {
    Nup.r = creal(0.5 * (dNup22 + dNdn22));
    Nup.i = cimag(0.5 * (dNup22 + dNdn22));
    Ndn.r = creal(0.5 * (dNup22 + dNdn22));
    Ndn.i = cimag(0.5 * (dNup22 + dNdn22));
  }

#else

  if (constraint_flag == 1)
  {
    Nup = dNup22;
    Ndn = dNdn22;
  }
  else if (constraint_flag == 2)
  {
    Nup.r = 0.5 * (dNup22.r + dNdn22.r);
    Nup.i = 0.5 * (dNup22.i + dNdn22.i);
    Ndn.r = 0.5 * (dNup22.r + dNdn22.r);
    Ndn.i = 0.5 * (dNup22.i + dNdn22.i);
  }

#endif

  dTN[1][1][0][0].r = Nup.r * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn.r * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

  dTN[1][1][0][0].i = 0.0;

  dTN[1][1][0][1].r = Nup.r * (U[0][0].r * U[0][1].r + U[0][0].i * U[0][1].i) + Ndn.r * (U[1][0].r * U[1][1].r + U[1][0].i * U[1][1].i);

  dTN[1][1][0][1].i = Nup.r * (-U[0][0].i * U[0][1].r + U[0][0].r * U[0][1].i) + Ndn.r * (-U[1][0].i * U[1][1].r + U[1][0].r * U[1][1].i);

  dTN[1][1][1][0].r = Nup.r * (U[0][1].r * U[0][0].r + U[0][1].i * U[0][0].i) + Ndn.r * (U[1][1].r * U[1][0].r + U[1][1].i * U[1][0].i);

  dTN[1][1][1][0].i = Nup.r * (-U[0][1].i * U[0][0].r + U[0][1].r * U[0][0].i) + Ndn.r * (-U[1][1].i * U[1][0].r + U[1][1].r * U[1][0].i);

  dTN[1][1][1][1].r = Nup.r * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn.r * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

  dTN[1][1][1][1].i = 0.0;

  /* calculate dTN12 */

#ifdef c_complex

  if (constraint_flag == 1)
  {
    Nup.r = creal(dNup12);
    Nup.i = cimag(dNup12);
    Ndn.r = creal(dNdn12);
    Ndn.i = cimag(dNdn12);
  }
  else if (constraint_flag == 2)
  {
    Nup.r = creal(0.5 * (dNup12 + dNdn12));
    Nup.i = cimag(0.5 * (dNup12 + dNdn12));
    Ndn.r = creal(0.5 * (dNup12 + dNdn12));
    Ndn.i = cimag(0.5 * (dNup12 + dNdn12));
  }

#else

  if (constraint_flag == 1)
  {
    Nup = dNup12;
    Ndn = dNdn12;
  }
  else if (constraint_flag == 2)
  {
    Nup.r = 0.5 * (dNup12.r + dNdn12.r);
    Nup.i = 0.5 * (dNup12.i + dNdn12.i);
    Ndn.r = 0.5 * (dNup12.r + dNdn12.r);
    Ndn.i = 0.5 * (dNup12.i + dNdn12.i);
  }

#endif

  dTN[0][1][0][0].r = Nup.r * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn.r * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

  dTN[0][1][0][0].i = Nup.i * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn.i * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

  tmp1.r = U[0][0].r * U[0][1].r + U[0][0].i * U[0][1].i;
  tmp1.i = -U[0][0].i * U[0][1].r + U[0][0].r * U[0][1].i;
  tmp2.r = U[1][0].r * U[1][1].r + U[1][0].i * U[1][1].i;
  tmp2.i = -U[1][0].i * U[1][1].r + U[1][0].r * U[1][1].i;

  dTN[0][1][0][1].r = Nup.r * tmp1.r - Nup.i * tmp1.i + Ndn.r * tmp2.r - Ndn.i * tmp2.i;

  dTN[0][1][0][1].i = Nup.r * tmp1.i + Nup.i * tmp1.r + Ndn.r * tmp2.i + Ndn.i * tmp2.r;

  tmp1.r = U[0][1].r * U[0][0].r + U[0][1].i * U[0][0].i;
  tmp1.i = -U[0][1].i * U[0][0].r + U[0][1].r * U[0][0].i;
  tmp2.r = U[1][1].r * U[1][0].r + U[1][1].i * U[1][0].i;
  tmp2.i = -U[1][1].i * U[1][0].r + U[1][1].r * U[1][0].i;

  dTN[0][1][1][0].r = Nup.r * tmp1.r - Nup.i * tmp1.i + Ndn.r * tmp2.r - Ndn.i * tmp2.i;

  dTN[0][1][1][0].i = Nup.r * tmp1.i + Nup.i * tmp1.r + Ndn.r * tmp2.i + Ndn.i * tmp2.r;

  dTN[0][1][1][1].r = Nup.r * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn.r * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

  dTN[0][1][1][1].i = Nup.i * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn.i * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

  /* calculate dTN21 */

#ifdef c_complex

  if (constraint_flag == 1)
  {
    Nup.r = creal(dNup21);
    Nup.i = cimag(dNup21);
    Ndn.r = creal(dNdn21);
    Ndn.i = cimag(dNdn21);
  }
  else if (constraint_flag == 2)
  {
    Nup.r = creal(0.5 * (dNup21 + dNdn21));
    Nup.i = cimag(0.5 * (dNup21 + dNdn21));
    Ndn.r = creal(0.5 * (dNup21 + dNdn21));
    Ndn.i = cimag(0.5 * (dNup21 + dNdn21));
  }

#else
  if (constraint_flag == 1)
  {
    Nup = dNup21;
    Ndn = dNdn21;
  }
  else if (constraint_flag == 2)
  {
    Nup.r = 0.5 * (dNup21.r + dNdn21.r);
    Nup.i = 0.5 * (dNup21.i + dNdn21.i);
    Ndn.r = 0.5 * (dNup21.r + dNdn21.r);
    Ndn.i = 0.5 * (dNup21.i + dNdn21.i);
  }

#endif

  dTN[1][0][0][0].r = Nup.r * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn.r * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

  dTN[1][0][0][0].i = Nup.i * (U[0][0].r * U[0][0].r + U[0][0].i * U[0][0].i) + Ndn.i * (U[1][0].r * U[1][0].r + U[1][0].i * U[1][0].i);

  tmp1.r = U[0][0].r * U[0][1].r + U[0][0].i * U[0][1].i;
  tmp1.i = -U[0][0].i * U[0][1].r + U[0][0].r * U[0][1].i;
  tmp2.r = U[1][0].r * U[1][1].r + U[1][0].i * U[1][1].i;
  tmp2.i = -U[1][0].i * U[1][1].r + U[1][0].r * U[1][1].i;

  dTN[1][0][0][1].r = Nup.r * tmp1.r - Nup.i * tmp1.i + Ndn.r * tmp2.r - Ndn.i * tmp2.i;

  dTN[1][0][0][1].i = Nup.r * tmp1.i + Nup.i * tmp1.r + Ndn.r * tmp2.i + Ndn.i * tmp2.r;

  tmp1.r = U[0][1].r * U[0][0].r + U[0][1].i * U[0][0].i;
  tmp1.i = -U[0][1].i * U[0][0].r + U[0][1].r * U[0][0].i;
  tmp2.r = U[1][1].r * U[1][0].r + U[1][1].i * U[1][0].i;
  tmp2.i = -U[1][1].i * U[1][0].r + U[1][1].r * U[1][0].i;

  dTN[1][0][1][0].r = Nup.r * tmp1.r - Nup.i * tmp1.i + Ndn.r * tmp2.r - Ndn.i * tmp2.i;

  dTN[1][0][1][0].i = Nup.r * tmp1.i + Nup.i * tmp1.r + Ndn.r * tmp2.i + Ndn.i * tmp2.r;

  dTN[1][0][1][1].r = Nup.r * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn.r * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);

  dTN[1][0][1][1].i = Nup.i * (U[0][1].r * U[0][1].r + U[0][1].i * U[0][1].i) + Ndn.i * (U[1][1].r * U[1][1].r + U[1][1].i * U[1][1].i);
}

void Calc_dSxyz(dcomplex TN[2][2],
                dcomplex dSx[2][2],
                dcomplex dSy[2][2],
                dcomplex dSz[2][2],
                double Nup[2], double Ndn[2],
                double theta[2], double phi[2])
{
  double dphi0, dtheta0;
  dcomplex tmp1, tmp2, tmp3;
  dcomplex I0, I1;

#ifdef c_complex
  double complex d11, d12, d21, d22;
  double complex dphi11, dphi12, dphi21, dphi22;
  double complex ctmp1, ctmp2, ctmp3;
  double complex cphi, ctheta;
  double complex cot, sit, cop, sip;
  double complex dtheta11, dtheta12, dtheta21, dtheta22;
  double complex dNup11, dNup12, dNup21, dNup22;
  double complex dNdn11, dNdn12, dNdn21, dNdn22;
  double complex dS_dNup, dS_dNdn, dS_dt, dS_dp;
#else
  dcomplex d11, d12, d21, d22;
  dcomplex dphi11, dphi12, dphi21, dphi22;
  dcomplex ctmp1, ctmp2, ctmp3;
  dcomplex cphi, ctheta;
  dcomplex cot, sit, cop, sip;
  dcomplex dtheta11, dtheta12, dtheta21, dtheta22;
  dcomplex dNup11, dNup12, dNup21, dNup22;
  dcomplex dNdn11, dNdn12, dNdn21, dNdn22;
  dcomplex dS_dNup, dS_dNdn, dS_dt, dS_dp;
#endif

  dcomplex ctmp4, ctmp5, ctmp6, ctmp7;
  dcomplex coe0;
  dcomplex ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8;

  I0 = Complex(0.0, 1.0);
  I1 = Complex(0.0, -1.0);

#ifdef c_complex
  d11 = TN[0][0].r + TN[0][0].i * I;
  d12 = TN[0][1].r + TN[0][1].i * I;
  d21 = TN[1][0].r + TN[1][0].i * I;
  d22 = TN[1][1].r + TN[1][1].i * I;
  cphi = phi[0] + phi[1] * I;
  ctheta = theta[0] + theta[1] * I;
  cot = ccos(ctheta);
  sit = csin(ctheta);
  cop = ccos(cphi);
  sip = csin(cphi);
#else
  d11 = TN[0][0];
  d12 = TN[0][1];
  d21 = TN[1][0];
  d22 = TN[1][1];
  cphi = Complex(phi[0], phi[1]);
  ctheta = Complex(theta[0], theta[1]);
  cot = Ccos(ctheta);
  sit = Csin(ctheta);
  cop = Ccos(cphi);
  sip = Csin(cphi);
#endif

  /* calculate dphi */

#ifdef c_complex

  dphi12 = 0.5 * d21 / (d12 * d21) * I;
  dphi21 = -0.5 * d12 / (d12 * d21) * I;

#else

  dphi11 = Complex(0.0, 0.0);
  dphi22 = Complex(0.0, 0.0);

  ctmp1 = Complex(0.0, 0.5);
  ctmp2 = Complex(0.0, -0.5);
  ctmp3 = Cmul(d12, d21);

  ct1 = Cdiv(d21, ctmp3);
  ct2 = Cdiv(d12, ctmp3);
  dphi12 = Cmul(ctmp1, ct1);
  dphi21 = Cmul(ctmp2, ct2);

#endif

  /*
  dphi12 = Cmul( ctmp1, Cdiv(d21,ctmp3) );
  dphi21 = Cmul( ctmp2, Cdiv(d12,ctmp3) );
  */

  /* calculate dtheta */

#ifdef c_complex

  ctmp1 = (d11 - d22) * (d11 - d22);
  ctmp2 = d12 * cexp(I * cphi) + d21 * cexp(-I * cphi);
  ctmp3 = ctmp1 / (ctmp1 + ctmp2 * ctmp2);

  dtheta11 = -ctmp3 * ctmp2 / ctmp1;
  dtheta22 = ctmp3 * ctmp2 / ctmp1;
  dtheta12 = ctmp3 * (cexp(I * cphi) + I * d12 * dphi12 * cexp(I * cphi) - I * d21 * dphi12 * cexp(-I * cphi)) / (d11 - d22);
  dtheta21 = ctmp3 * (cexp(-I * cphi) + I * d12 * dphi21 * cexp(I * cphi) - I * d21 * dphi21 * cexp(-I * cphi)) / (d11 - d22);

#else

  ct1 = Csub(d11, d22);
  ct2 = Csub(d11, d22);
  ctmp1 = Cmul(ct1, ct2);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp2 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp2, ctmp2);
  ct2 = Cadd(ctmp1, ct1);
  ctmp3 = Cdiv(ctmp1, ct2);

  ct1 = Cmul(ctmp3, ctmp2);
  ct2 = Cdiv(ct1, ctmp1);
  dtheta11 = RCmul(-1.0, ct1);

  ct1 = Cmul(ctmp3, ctmp2);
  dtheta22 = Cdiv(ct1, ctmp1);

  ct1 = Cmul(I0, cphi);
  ctmp4 = Cexp(ct1);

  ct1 = Cmul(I0, d12);
  ct2 = Cmul(ct1, dphi12);
  ct3 = Cmul(I0, cphi);
  ct4 = Cexp(ct3);
  ctmp5 = Cmul(ct2, ct3);

  ct1 = Cmul(I0, d21);
  ct2 = Cmul(ct1, dphi12);
  ct3 = Cmul(I1, cphi);
  ct4 = Cexp(ct3);
  ctmp6 = Cmul(ct2, ct4);

  ct1 = Csub(d11, d22);
  ctmp7 = Cdiv(ctmp3, ct1);

  ct1 = Cadd(ctmp4, ctmp5);
  ct2 = Csub(ct1, ctmp6);
  dtheta12 = Cmul(ctmp7, ct2);

  ct1 = Cmul(I1, cphi);
  ctmp4 = Cexp(ct1);

  ct1 = Cmul(I0, d12);
  ct2 = Cmul(ct1, dphi21);
  ct3 = Cmul(I0, cphi);
  ct4 = Cexp(ct3);
  ctmp5 = Cmul(ct2, ct4);

  ct1 = Cmul(I0, d21);
  ct2 = Cmul(ct1, dphi21);
  ct3 = Cmul(I1, cphi);
  ct4 = Cexp(ct3);
  ctmp6 = Cmul(ct2, ct4);

  ct1 = Csub(d11, d22);
  ctmp7 = Cdiv(ctmp3, ct1);

  ct1 = Cadd(ctmp4, ctmp5);
  ct2 = Csub(ct1, ctmp6);
  dtheta21 = Cmul(ctmp7, ct2);

#endif

  /*
  ctmp1 = Cmul( Csub(d11, d22), Csub(d11, d22) );
  ctmp2 = Cadd( Cmul(d12, Cexp(Cmul(I0,cphi))), Cmul(d21, Cexp(Cmul(I1,cphi))) );
  ctmp3 = Cdiv( ctmp1, Cadd(ctmp1, Cmul(ctmp2,ctmp2)) );

  dtheta11 = RCmul(-1.0, Cdiv( Cmul(ctmp3,ctmp2), ctmp1));
  dtheta22 = Cdiv( Cmul(ctmp3,ctmp2), ctmp1);

  ctmp4 = Cexp(Cmul(I0,cphi));
  ctmp5 = Cmul(Cmul(Cmul(I0,d12),dphi12),Cexp(Cmul(I0,cphi)));
  ctmp6 = Cmul(Cmul(Cmul(I0,d21),dphi12),Cexp(Cmul(I1,cphi)));
  ctmp7 = Cdiv(ctmp3, Csub(d11,d22));
  dtheta12 = Cmul(ctmp7, Csub(Cadd(ctmp4,ctmp5),ctmp6));

  ctmp4 = Cexp(Cmul(I1,cphi));
  ctmp5 = Cmul(Cmul(Cmul(I0,d12),dphi21),Cexp(Cmul(I0,cphi)));
  ctmp6 = Cmul(Cmul(Cmul(I0,d21),dphi21),Cexp(Cmul(I1,cphi)));
  ctmp7 = Cdiv(ctmp3, Csub(d11,d22));
  dtheta21 = Cmul(ctmp7, Csub(Cadd(ctmp4,ctmp5),ctmp6));
  */

  /* calculate dNup */

#ifdef c_complex

  dNup11 = 0.5 * ((1.0 + 0.0 * I) + cot - (d11 - d22) * sit * dtheta11 + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta11);

  dNup22 = 0.5 * ((1.0 + 0.0 * I) - cot - (d11 - d22) * sit * dtheta22 + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta22);

  dNup12 = 0.5 * (cexp(I * cphi) * sit - (d11 - d22) * sit * dtheta12 + I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi12 * sit + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta12 * cot);

  dNup21 = 0.5 * (cexp(-I * cphi) * sit - (d11 - d22) * sit * dtheta21 + I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi21 * sit + (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta21 * cot);

#else

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Cadd(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta11);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta11);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  dNup11 = RCmul(0.5, ct2);

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Csub(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta22);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta22);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  dNup22 = RCmul(0.5, ct2);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp5 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta12);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  ct3 = Cadd(ct2, ctmp6);
  dNup12 = RCmul(0.5, ct3);

  ct1 = Cmul(I1, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d21, ct2);
  ct4 = Cmul(I0, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d12, ct5);
  ctmp5 = Cadd(ct5, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta21);

  ct1 = Csub(ctmp1, ctmp2);
  ct2 = Cadd(ct1, ctmp4);
  ct3 = Cadd(ct2, ctmp6);
  dNup21 = RCmul(0.5, ct3);

#endif

  /*
  ctmp1 = Cadd(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta11);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta11);
  dNup11 = RCmul(0.5,Cadd(Csub(ctmp1, ctmp2),ctmp4));

  ctmp1 = Csub(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta22);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta22);
  dNup22 = RCmul(0.5,Cadd(Csub(ctmp1, ctmp2),ctmp4));

  ctmp1 = Cmul(Cexp(Cmul(I0,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta12);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi12);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta12);
  dNup12 = RCmul(0.5,Cadd(Cadd(Csub(ctmp1,ctmp2),ctmp4),ctmp6));

  ctmp1 = Cmul(Cexp(Cmul(I1,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta21);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi21);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta21);
  dNup21 = RCmul(0.5,Cadd(Cadd(Csub(ctmp1,ctmp2),ctmp4),ctmp6));
  */

  /* calculate dNdn */

#ifdef c_complex

  dNdn11 = 0.5 * ((1.0 + 0.0 * I) - cot + (d11 - d22) * sit * dtheta11 - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta11);

  dNdn22 = 0.5 * ((1.0 + 0.0 * I) + cot + (d11 - d22) * sit * dtheta22 - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * cot * dtheta22);

  dNdn12 = 0.5 * (-cexp(I * cphi) * sit + (d11 - d22) * sit * dtheta12 - I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi12 * sit - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta12 * cot);

  dNdn21 = 0.5 * (-cexp(-I * cphi) * sit + (d11 - d22) * sit * dtheta21 - I * (d12 * cexp(I * cphi) - d21 * cexp(-I * cphi)) * dphi21 * sit - (d12 * cexp(I * cphi) + d21 * cexp(-I * cphi)) * dtheta21 * cot);

#else

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Csub(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta11);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta11);

  ct1 = Cadd(ctmp1, ctmp2);
  ct2 = Csub(ct1, ctmp4);
  dNdn11 = RCmul(0.5, ct2);

  ct1 = Complex(1.0, 0.0);
  ctmp1 = Cadd(ct1, cot);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta22);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp3 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp3, cot);
  ctmp4 = Cmul(ct1, dtheta22);

  ct1 = Cadd(ctmp1, ctmp2);
  ct2 = Csub(ct1, ctmp4);
  dNdn22 = RCmul(0.5, ct2);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi12);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp5 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta12);

  ct1 = Csub(ctmp2, ctmp1);
  ct2 = Csub(ct1, ctmp4);
  ct3 = Csub(ct2, ctmp6);
  dNdn12 = RCmul(0.5, ct3);

  ct1 = Cmul(I1, cphi);
  ct2 = Cexp(ct1);
  ctmp1 = Cmul(ct2, sit);

  ct1 = Csub(d11, d22);
  ct2 = Cmul(ct1, sit);
  ctmp2 = Cmul(ct2, dtheta21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ct7 = Csub(ct3, ct6);
  ctmp3 = Cmul(I0, ct7);

  ct1 = Cmul(ctmp3, sit);
  ctmp4 = Cmul(ct1, dphi21);

  ct1 = Cmul(I0, cphi);
  ct2 = Cexp(ct1);
  ct3 = Cmul(d12, ct2);
  ct4 = Cmul(I1, cphi);
  ct5 = Cexp(ct4);
  ct6 = Cmul(d21, ct5);
  ctmp5 = Cadd(ct3, ct6);

  ct1 = Cmul(ctmp5, cot);
  ctmp6 = Cmul(ct1, dtheta21);

  ct1 = Csub(ctmp2, ctmp1);
  ct2 = Csub(ct1, ctmp4);
  ct3 = Csub(ct2, ctmp6);
  dNdn21 = RCmul(0.5, ct3);

#endif

  /*
  ctmp1 = Csub(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta11);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta11);
  dNdn11 = RCmul(0.5,Csub(Cadd(ctmp1, ctmp2),ctmp4));

  ctmp1 = Cadd(Complex(1.0, 0.0), cot);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta22);
  ctmp3 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp4 = Cmul(Cmul(ctmp3,cot),dtheta22);
  dNdn22 = RCmul(0.5,Csub(Cadd(ctmp1, ctmp2),ctmp4));

  ctmp1 = Cmul(Cexp(Cmul(I0,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta12);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi12);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta12);
  dNdn12 = RCmul(0.5,Csub(Csub(Csub(ctmp2,ctmp1),ctmp4),ctmp6));

  ctmp1 = Cmul(Cexp(Cmul(I1,cphi)),sit);
  ctmp2 = Cmul(Cmul(Csub(d11,d22), sit), dtheta21);
  ctmp3 = Cmul(I0,Csub(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi)))));
  ctmp4 = Cmul(Cmul(ctmp3,sit),dphi21);
  ctmp5 = Cadd(Cmul(d12,Cexp(Cmul(I0,cphi))),Cmul(d21,Cexp(Cmul(I1,cphi))));
  ctmp6 = Cmul(Cmul(ctmp5,cot),dtheta21);
  dNdn21 = RCmul(0.5,Csub(Csub(Csub(ctmp2,ctmp1),ctmp4),ctmp6));
  */

  /*
  printf("dNdn11.r=%15.12f dNdn11.i=%15.12f\n",creal(dNdn11),cimag(dNdn11));
  printf("dNdn22.r=%15.12f dNdn22.i=%15.12f\n",creal(dNdn22),cimag(dNdn22));
  printf("dNdn12.r=%15.12f dNdn12.i=%15.12f\n",creal(dNdn12),cimag(dNdn12));
  printf("dNdn21.r=%15.12f dNdn21.i=%15.12f\n",creal(dNdn21),cimag(dNdn21));
  */

  /*******************************
           calculate dSx
  *******************************/

#ifdef c_complex

  dS_dNup = 0.5 * sit * cop;
  dS_dNdn = -0.5 * sit * cop;

  ctmp1 = Nup[0] + Nup[1] * I;
  ctmp2 = Ndn[0] + Ndn[1] * I;
  dS_dt = 0.5 * (ctmp1 - ctmp2) * cot * cop;
  dS_dp = -0.5 * (ctmp1 - ctmp2) * sit * sip;

#else

  ct1 = Complex(0.5, 0.0);
  ct2 = Cmul(ct1, sit);
  dS_dNup = Cmul(ct2, cop);

  ct1 = Complex(-0.5, 0.0);
  ct2 = Cmul(ct1, sit);
  dS_dNdn = Cmul(ct2, cop);

  ct1 = Complex(0.5, 0.0);
  ct2 = Complex(Nup[0], 0.0);
  ct3 = Complex(Ndn[0], 0.0);
  ct4 = Csub(ct2, ct3);
  ct5 = Cmul(ct1, ct4);
  ct6 = Cmul(ct5, cot);
  ct7 = Cmul(ct6, cop);
  dS_dt = ct7;

  ct1 = Complex(-0.5, 0.0);
  ct2 = Complex(Nup[0], 0.0);
  ct3 = Complex(Ndn[0], 0.0);
  ct4 = Csub(ct2, ct3);
  ct5 = Cmul(ct1, ct4);
  ct6 = Cmul(ct5, sit);
  ct7 = Cmul(ct6, sip);
  dS_dp = ct7;

#endif

#ifdef c_complex

  /* dSx11 */

  ctmp1 = dS_dNup * dNup11 + dS_dNdn * dNdn11 + dS_dt * dtheta11 + dS_dp * dphi11;
  dSx[0][0].r = creal(ctmp1);
  dSx[0][0].i = cimag(ctmp1);

  /* dSx12 */

  ctmp1 = dS_dNup * dNup12 + dS_dNdn * dNdn12 + dS_dt * dtheta12 + dS_dp * dphi12;
  dSx[0][1].r = creal(ctmp1);
  dSx[0][1].i = cimag(ctmp1);

  /* dSx21 */

  ctmp1 = dS_dNup * dNup21 + dS_dNdn * dNdn21 + dS_dt * dtheta21 + dS_dp * dphi21;
  dSx[1][0].r = creal(ctmp1);
  dSx[1][0].i = cimag(ctmp1);

  /* dSx22 */

  ctmp1 = dS_dNup * dNup22 + dS_dNdn * dNdn22 + dS_dt * dtheta22 + dS_dp * dphi22;
  dSx[1][1].r = creal(ctmp1);
  dSx[1][1].i = cimag(ctmp1);

#else

  /* dSx11 */

  ct1 = Cmul(dS_dNup, dNup11);
  ct2 = Cmul(dS_dNdn, dNdn11);
  ct3 = Cmul(dS_dt, dtheta11);
  ct4 = Cmul(dS_dp, dphi11);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSx[0][0] = ct7;

  /* dSx12 */

  ct1 = Cmul(dS_dNup, dNup12);
  ct2 = Cmul(dS_dNdn, dNdn12);
  ct3 = Cmul(dS_dt, dtheta12);
  ct4 = Cmul(dS_dp, dphi12);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSx[0][1] = ct7;

  /* dSx21 */

  ct1 = Cmul(dS_dNup, dNup21);
  ct2 = Cmul(dS_dNdn, dNdn21);
  ct3 = Cmul(dS_dt, dtheta21);
  ct4 = Cmul(dS_dp, dphi21);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSx[1][0] = ct7;

  /* dSx22 */

  ct1 = Cmul(dS_dNup, dNup22);
  ct2 = Cmul(dS_dNdn, dNdn22);
  ct3 = Cmul(dS_dt, dtheta22);
  ct4 = Cmul(dS_dp, dphi22);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSx[1][1] = ct7;

#endif

  /*******************************
           calculate dSy
  *******************************/

#ifdef c_complex

  dS_dNup = 0.5 * sit * sip;
  dS_dNdn = -0.5 * sit * sip;

  ctmp1 = Nup[0] + Nup[1] * I;
  ctmp2 = Ndn[0] + Ndn[1] * I;
  dS_dt = 0.5 * (ctmp1 - ctmp2) * cot * sip;
  dS_dp = 0.5 * (ctmp1 - ctmp2) * sit * cop;

#else

  ct1 = Complex(0.5, 0.0);
  ct2 = Cmul(ct1, sit);
  dS_dNup = Cmul(ct2, sip);

  ct1 = Complex(-0.5, 0.0);
  ct2 = Cmul(ct1, sit);
  dS_dNdn = Cmul(ct2, sip);

  ct1 = Complex(0.5, 0.0);
  ct2 = Complex(Nup[0], 0.0);
  ct3 = Complex(Ndn[0], 0.0);
  ct4 = Csub(ct2, ct3);
  ct5 = Cmul(ct1, ct4);
  ct6 = Cmul(ct5, cot);
  ct7 = Cmul(ct6, sip);
  dS_dt = ct7;

  ct1 = Complex(0.5, 0.0);
  ct2 = Complex(Nup[0], 0.0);
  ct3 = Complex(Ndn[0], 0.0);
  ct4 = Csub(ct2, ct3);
  ct5 = Cmul(ct1, ct4);
  ct6 = Cmul(ct5, sit);
  ct7 = Cmul(ct6, cop);
  dS_dp = ct7;

#endif

#ifdef c_complex

  /* dSy11 */

  ctmp1 = dS_dNup * dNup11 + dS_dNdn * dNdn11 + dS_dt * dtheta11 + dS_dp * dphi11;
  dSy[0][0].r = creal(ctmp1);
  dSy[0][0].i = cimag(ctmp1);

  /* dSy12 */

  ctmp1 = dS_dNup * dNup12 + dS_dNdn * dNdn12 + dS_dt * dtheta12 + dS_dp * dphi12;
  dSy[0][1].r = creal(ctmp1);
  dSy[0][1].i = cimag(ctmp1);

  /* dSy21 */

  ctmp1 = dS_dNup * dNup21 + dS_dNdn * dNdn21 + dS_dt * dtheta21 + dS_dp * dphi21;
  dSy[1][0].r = creal(ctmp1);
  dSy[1][0].i = cimag(ctmp1);

  /* dSy22 */

  ctmp1 = dS_dNup * dNup22 + dS_dNdn * dNdn22 + dS_dt * dtheta22 + dS_dp * dphi22;
  dSy[1][1].r = creal(ctmp1);
  dSy[1][1].i = cimag(ctmp1);

#else

  /* dSy11 */

  ct1 = Cmul(dS_dNup, dNup11);
  ct2 = Cmul(dS_dNdn, dNdn11);
  ct3 = Cmul(dS_dt, dtheta11);
  ct4 = Cmul(dS_dp, dphi11);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSy[0][0] = ct7;

  /* dSy12 */

  ct1 = Cmul(dS_dNup, dNup12);
  ct2 = Cmul(dS_dNdn, dNdn12);
  ct3 = Cmul(dS_dt, dtheta12);
  ct4 = Cmul(dS_dp, dphi12);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSy[0][1] = ct7;

  /* dSy21 */

  ct1 = Cmul(dS_dNup, dNup21);
  ct2 = Cmul(dS_dNdn, dNdn21);
  ct3 = Cmul(dS_dt, dtheta21);
  ct4 = Cmul(dS_dp, dphi21);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSy[1][0] = ct7;

  /* dSy22 */

  ct1 = Cmul(dS_dNup, dNup22);
  ct2 = Cmul(dS_dNdn, dNdn22);
  ct3 = Cmul(dS_dt, dtheta22);
  ct4 = Cmul(dS_dp, dphi22);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSy[1][1] = ct7;

#endif

  /*******************************
           calculate dSz
  *******************************/

#ifdef c_complex

  dS_dNup = 0.5 * cot;
  dS_dNdn = -0.5 * cot;

  ctmp1 = Nup[0] + Nup[1] * I;
  ctmp2 = Ndn[0] + Ndn[1] * I;
  dS_dt = -0.5 * (ctmp1 - ctmp2) * sit;
  dS_dp = 0.0;

#else

  ct1 = Complex(0.5, 0.0);
  ct2 = Cmul(ct1, cot);
  dS_dNup = ct2;

  ct1 = Complex(-0.5, 0.0);
  ct2 = Cmul(ct1, cot);
  dS_dNdn = ct2;

  ct1 = Complex(-0.5, 0.0);
  ct2 = Complex(Nup[0], 0.0);
  ct3 = Complex(Ndn[0], 0.0);
  ct4 = Csub(ct2, ct3);
  ct5 = Cmul(ct1, ct4);
  ct6 = Cmul(ct5, sit);
  dS_dt = ct6;

  ct1 = Complex(0.0, 0.0);
  dS_dp = ct1;

#endif

#ifdef c_complex

  /* dSz11 */

  ctmp1 = dS_dNup * dNup11 + dS_dNdn * dNdn11 + dS_dt * dtheta11 + dS_dp * dphi11;
  dSz[0][0].r = creal(ctmp1);
  dSz[0][0].i = cimag(ctmp1);

  /* dSz12 */

  ctmp1 = dS_dNup * dNup12 + dS_dNdn * dNdn12 + dS_dt * dtheta12 + dS_dp * dphi12;
  dSz[0][1].r = creal(ctmp1);
  dSz[0][1].i = cimag(ctmp1);

  /* dSz21 */

  ctmp1 = dS_dNup * dNup21 + dS_dNdn * dNdn21 + dS_dt * dtheta21 + dS_dp * dphi21;
  dSz[1][0].r = creal(ctmp1);
  dSz[1][0].i = cimag(ctmp1);

  /* dSz22 */

  ctmp1 = dS_dNup * dNup22 + dS_dNdn * dNdn22 + dS_dt * dtheta22 + dS_dp * dphi22;
  dSz[1][1].r = creal(ctmp1);
  dSz[1][1].i = cimag(ctmp1);

#else

  /* dSz11 */

  ct1 = Cmul(dS_dNup, dNup11);
  ct2 = Cmul(dS_dNdn, dNdn11);
  ct3 = Cmul(dS_dt, dtheta11);
  ct4 = Cmul(dS_dp, dphi11);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSz[0][0] = ct7;

  /* dSz12 */

  ct1 = Cmul(dS_dNup, dNup12);
  ct2 = Cmul(dS_dNdn, dNdn12);
  ct3 = Cmul(dS_dt, dtheta12);
  ct4 = Cmul(dS_dp, dphi12);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSz[0][1] = ct7;

  /* dSz21 */

  ct1 = Cmul(dS_dNup, dNup21);
  ct2 = Cmul(dS_dNdn, dNdn21);
  ct3 = Cmul(dS_dt, dtheta21);
  ct4 = Cmul(dS_dp, dphi21);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSz[1][0] = ct7;

  /* dSz22 */

  ct1 = Cmul(dS_dNup, dNup22);
  ct2 = Cmul(dS_dNdn, dNdn22);
  ct3 = Cmul(dS_dt, dtheta22);
  ct4 = Cmul(dS_dp, dphi22);
  ct5 = Cadd(ct1, ct2);
  ct6 = Cadd(ct5, ct3);
  ct7 = Cadd(ct6, ct4);
  dSz[1][1] = ct7;

#endif
}
