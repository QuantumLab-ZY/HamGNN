/**********************************************************************
  Eff_Hub_Pot.c:

     Eff_Hub_Pot.c is a subroutine to construct effective potential
     for obtaining the effective Hubbard Hamiltonial in LDA+U calculation.

  Log of Eff_Hub_Pot.c:

      4/Aug/2004  Released by M.J.Han (--- MJ)
     29/Nov/2004  Modified by T.Ozaki (AIST)
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

static void H_U_onsite();
static void H_U_full(int SCF_iter, double ****OLP0);
static void H_U_dual(int SCF_iter, double ****OLP0);

void Eff_Hub_Pot(int SCF_iter, double ****OLP0)
{
  /****************************************************
    Important note:

    In case of orbital optimization, the U potential
    is applied to the primitive orbital.
  ****************************************************/
  int spin, a, b;
  int l, fan, j, k, i, jg, kg, ig, wakg, waig, kl, il1, il2, m, n;
  int Cwan, Hwan, Mul1, tno0, tno1, tno2;
  int Mc_AN, Gc_AN, Mj_AN, Mi_AN, Mk_AN;
  double sum;

  /* in case of correction */

  if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
  {

    /* on site */
    if (Hub_U_occupation == 0)
    {
      H_U_onsite();
    }

    /* full */
    else if (Hub_U_occupation == 1)
    {
      H_U_full(SCF_iter, OLP0);
    }

    /* dual */
    else if (Hub_U_occupation == 2)
    {
      H_U_dual(SCF_iter, OLP0);
    }

  } /* if */

  /* in case of no correction */

  else
  {

    /* According to the report 44, iHNL0 is copied to iHNL. */

    if (SpinP_switch == 3 && SO_switch == 1)
    {

      for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
      {

        Gc_AN = M2G[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        fan = FNAN[Gc_AN];
        tno0 = Spe_Total_NO[Cwan];

        for (j = 0; j <= fan; j++)
        {
          jg = natn[Gc_AN][j];
          Mj_AN = F_G2M[jg];
          Hwan = WhatSpecies[jg];
          tno1 = Spe_Total_NO[Hwan];

          for (m = 0; m < tno0; m++)
          {
            for (n = 0; n < tno1; n++)
            {

              /* imaginary part */

              iHNL[0][Mc_AN][j][m][n] = iHNL0[0][Mc_AN][j][m][n];
              iHNL[1][Mc_AN][j][m][n] = iHNL0[1][Mc_AN][j][m][n];
              iHNL[2][Mc_AN][j][m][n] = iHNL0[2][Mc_AN][j][m][n];
            }
          }

          /* in case of the core hole calculation */

          if (core_hole_state_flag == 1)
          {

            for (m = 0; m < tno0; m++)
            {
              for (n = 0; n < tno1; n++)
              {

                /* imaginary part */

                iHNL[0][Mc_AN][j][m][n] += iHCH[0][Mc_AN][j][m][n];
                iHNL[1][Mc_AN][j][m][n] += iHCH[1][Mc_AN][j][m][n];
                iHNL[2][Mc_AN][j][m][n] += iHCH[2][Mc_AN][j][m][n];
              }
            }
          }
        }
      }
    }

  } /* else */
}

void H_U_onsite()
{
  int spin, a, b;
  int l, fan, j, k, i, jg, kg, ig, wakg, waig, kl, il1, il2, m, n;
  int Cwan, Hwan, Mul1, tno0, tno1, tno2;
  int Mc_AN, Gc_AN, Mj_AN, Mi_AN, Mk_AN;

  /****************************************************
    if (SpinP_switch!=3)

    collinear case
  ****************************************************/

  if (SpinP_switch != 3)
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      for (j = 0; j <= fan; j++)
      {
        jg = natn[Gc_AN][j];
        Mj_AN = F_G2M[jg];
        Hwan = WhatSpecies[jg];
        tno1 = Spe_Total_NO[Hwan];

        if (j == 0)
        {
          for (spin = 0; spin <= SpinP_switch; spin++)
          {
            for (m = 0; m < tno0; m++)
            {
              for (n = 0; n < tno1; n++)
              {
                H_Hub[spin][Mc_AN][j][m][n] = v_eff[spin][Mc_AN][m][n];
              }
            }
          }
        }

        else
        {
          for (spin = 0; spin <= SpinP_switch; spin++)
          {
            for (m = 0; m < tno0; m++)
            {
              for (n = 0; n < tno1; n++)
              {
                H_Hub[spin][Mc_AN][j][m][n] = 0.0;
              }
            }
          }
        }
      }
    }
  }

  /****************************************************
    if (SpinP_switch==3)

    spin non-collinear
  ****************************************************/

  else
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      for (j = 0; j <= fan; j++)
      {
        jg = natn[Gc_AN][j];
        Mj_AN = F_G2M[jg];
        Hwan = WhatSpecies[jg];
        tno1 = Spe_Total_NO[Hwan];

        if (j == 0)
        {
          for (m = 0; m < tno0; m++)
          {
            for (n = 0; n < tno1; n++)
            {

              /* real part */

              H_Hub[0][Mc_AN][j][m][n] = NC_v_eff[0][0][Mc_AN][m][n].r;
              H_Hub[1][Mc_AN][j][m][n] = NC_v_eff[1][1][Mc_AN][m][n].r;
              H_Hub[2][Mc_AN][j][m][n] = NC_v_eff[0][1][Mc_AN][m][n].r;

              /* imaginary part */

              iHNL[0][Mc_AN][j][m][n] = iHNL0[0][Mc_AN][j][m][n] + F_U_flag * NC_v_eff[0][0][Mc_AN][m][n].i;
              iHNL[1][Mc_AN][j][m][n] = iHNL0[1][Mc_AN][j][m][n] + F_U_flag * NC_v_eff[1][1][Mc_AN][m][n].i;
              iHNL[2][Mc_AN][j][m][n] = iHNL0[2][Mc_AN][j][m][n] + F_U_flag * NC_v_eff[0][1][Mc_AN][m][n].i;
            }
          }
        }

        else
        {
          for (m = 0; m < tno0; m++)
          {
            for (n = 0; n < tno1; n++)
            {

              /* real part */

              H_Hub[0][Mc_AN][j][m][n] = 0.0;
              H_Hub[1][Mc_AN][j][m][n] = 0.0;
              H_Hub[2][Mc_AN][j][m][n] = 0.0;
            }
          }
        }
      }
    }
  }
}

void H_U_full(int SCF_iter, double ****OLP0)
{
  int spin, a, b;
  int l, fan, j, k, i, jg, kg, ig, wakg, waig, kl, il1, il2, m, n;
  int Cwan, Hwan, Mul1, tno0, tno1, tno2;
  int Mc_AN, Gc_AN, Mj_AN, Mi_AN, Mk_AN;
  int num, h_AN, Gh_AN, size1, size2;
  double *tmp_array;
  double *tmp_array2;
  int *Snd_S_Size, *Rcv_S_Size;
  double sum, sum0;
  double Resum00, Resum11, Resum01;
  double Imsum00, Imsum11, Imsum01;
  int numprocs, myid, ID, IDS, IDR, tag = 999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  Snd_S_Size = (int *)malloc(sizeof(int) * numprocs);
  Rcv_S_Size = (int *)malloc(sizeof(int) * numprocs);

  /****************************************************
   MPI

   OLP0
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  if (SCF_iter <= 2)
  {

    for (ID = 0; ID < numprocs; ID++)
    {

      IDS = (myid + ID) % numprocs;
      IDR = (myid - ID + numprocs) % numprocs;

      if (ID != 0)
      {

        tag = 999;

        /* find data size to send block data */
        if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
        {
          size1 = 0;
          for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++)
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

          Snd_S_Size[IDS] = size1;
          MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
        }
        else
        {
          Snd_S_Size[IDS] = 0;
        }

        /* receiving of size of data */

        if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0)
        {
          MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
          Rcv_S_Size[IDR] = size2;
        }
        else
        {
          Rcv_S_Size[IDR] = 0;
        }

        if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
          MPI_Wait(&request, &stat);
      }
      else
      {
        Snd_S_Size[IDS] = 0;
        Rcv_S_Size[IDR] = 0;
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

        if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
        {

          size1 = Snd_S_Size[IDS];

          /* allocation of array */

          tmp_array = (double *)malloc(sizeof(double) * size1);

          /* multidimentional array to vector array */

          num = 0;

          for (n = 0; n < (F_Snd_Num[IDS] + S_Snd_Num[IDS]); n++)
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
                  tmp_array[num] = OLP0[Mc_AN][h_AN][i][j];
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

        if ((F_Rcv_Num[IDR] + S_Rcv_Num[IDR]) != 0)
        {

          size2 = Rcv_S_Size[IDR];

          /* allocation of array */
          tmp_array2 = (double *)malloc(sizeof(double) * size2);

          MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);

          num = 0;
          Mc_AN = S_TopMAN[IDR] - 1; /* S_TopMAN should be used. */
          for (n = 0; n < (F_Rcv_Num[IDR] + S_Rcv_Num[IDR]); n++)
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
                  OLP0[Mc_AN][h_AN][i][j] = tmp_array2[num];
                  num++;
                }
              }
            }
          }

          /* freeing of array */
          free(tmp_array2);
        }

        if ((F_Snd_Num[IDS] + S_Snd_Num[IDS]) != 0)
        {
          MPI_Wait(&request, &stat);
          free(tmp_array); /* freeing of array */
        }
      }
    }
  }

  /****************************************************
                     make H_Hub
  ****************************************************/

  /****************************************************
    if (SpinP_switch!=3)

    collinear case
  ****************************************************/

  if (SpinP_switch != 3)
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      for (j = 0; j <= fan; j++)
      {
        jg = natn[Gc_AN][j];
        Mj_AN = S_G2M[jg]; /* S_G2G should be used due to consistency with DC method */
        Hwan = WhatSpecies[jg];
        tno1 = Spe_Total_NO[Hwan];

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (m = 0; m < tno0; m++)
          {
            for (n = 0; n < tno1; n++)
            {
              H_Hub[spin][Mc_AN][j][m][n] = 0.0;
            }
          }
        }

        for (k = 0; k <= fan; k++)
        {
          kg = natn[Gc_AN][k];
          Mk_AN = F_G2M[kg]; /* F_G2G should be used */
          wakg = WhatSpecies[kg];
          kl = RMI1[Mc_AN][j][k];
          tno2 = Spe_Total_NO[wakg];

          if (0 <= kl)
          {

            for (m = 0; m < tno0; m++)
            {
              for (n = 0; n < tno1; n++)
              {
                for (spin = 0; spin <= SpinP_switch; spin++)
                {

                  sum = 0.0;
                  for (a = 0; a < tno2; a++)
                  {
                    for (b = 0; b < tno2; b++)
                    {
                      sum += OLP0[Mc_AN][k][m][a] * v_eff[spin][Mk_AN][a][b] * OLP0[Mj_AN][kl][n][b];
                    }
                  }

                  if (3 <= spin)
                    H_Hub[spin][Mc_AN][j][m][n] = 0.0;
                  else
                    H_Hub[spin][Mc_AN][j][m][n] += sum;
                }
              }
            }
          }
        }
      }
    }
  }

  /****************************************************
    if (SpinP_switch==3)

    spin non-collinear
  ****************************************************/

  else
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      for (j = 0; j <= fan; j++)
      {
        jg = natn[Gc_AN][j];
        Mj_AN = S_G2M[jg]; /* S_G2G should be used due to consistency with DC method */
        Hwan = WhatSpecies[jg];
        tno1 = Spe_Total_NO[Hwan];

        for (m = 0; m < tno0; m++)
        {
          for (n = 0; n < tno1; n++)
          {

            /* real part */

            H_Hub[0][Mc_AN][j][m][n] = 0.0;
            H_Hub[1][Mc_AN][j][m][n] = 0.0;
            H_Hub[2][Mc_AN][j][m][n] = 0.0;

            /* imaginary part */

            iHNL[0][Mc_AN][j][m][n] = iHNL0[0][Mc_AN][j][m][n];
            iHNL[1][Mc_AN][j][m][n] = iHNL0[1][Mc_AN][j][m][n];
            iHNL[2][Mc_AN][j][m][n] = iHNL0[2][Mc_AN][j][m][n];
          }
        }

        for (k = 0; k <= fan; k++)
        {
          kg = natn[Gc_AN][k];
          Mk_AN = F_G2M[kg]; /* F_G2G should be used */
          wakg = WhatSpecies[kg];
          kl = RMI1[Mc_AN][j][k];
          tno2 = Spe_Total_NO[wakg];

          if (0 <= kl)
          {

            for (m = 0; m < tno0; m++)
            {
              for (n = 0; n < tno1; n++)
              {

                Resum00 = 0.0;
                Resum11 = 0.0;
                Resum01 = 0.0;

                Imsum00 = 0.0;
                Imsum11 = 0.0;
                Imsum01 = 0.0;

                for (a = 0; a < tno2; a++)
                {
                  for (b = 0; b < tno2; b++)
                  {

                    Resum00 += OLP0[Mc_AN][k][m][a] * NC_v_eff[0][0][Mk_AN][a][b].r * OLP0[Mj_AN][kl][n][b];
                    Resum11 += OLP0[Mc_AN][k][m][a] * NC_v_eff[1][1][Mk_AN][a][b].r * OLP0[Mj_AN][kl][n][b];
                    Resum01 += OLP0[Mc_AN][k][m][a] * NC_v_eff[0][1][Mk_AN][a][b].r * OLP0[Mj_AN][kl][n][b];

                    Imsum00 += OLP0[Mc_AN][k][m][a] * NC_v_eff[0][0][Mk_AN][a][b].i * OLP0[Mj_AN][kl][n][b];
                    Imsum11 += OLP0[Mc_AN][k][m][a] * NC_v_eff[1][1][Mk_AN][a][b].i * OLP0[Mj_AN][kl][n][b];
                    Imsum01 += OLP0[Mc_AN][k][m][a] * NC_v_eff[0][1][Mk_AN][a][b].i * OLP0[Mj_AN][kl][n][b];
                  }
                }

                /* real part */

                H_Hub[0][Mc_AN][j][m][n] += Resum00;
                H_Hub[1][Mc_AN][j][m][n] += Resum11;
                H_Hub[2][Mc_AN][j][m][n] += Resum01;

                /* imaginary part */

                iHNL[0][Mc_AN][j][m][n] += F_U_flag * Imsum00;
                iHNL[1][Mc_AN][j][m][n] += F_U_flag * Imsum11;
                iHNL[2][Mc_AN][j][m][n] += F_U_flag * Imsum01;
              }
            }
          }
        }
      }
    }
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  free(Snd_S_Size);
  free(Rcv_S_Size);
}

void H_U_dual(int SCF_iter, double ****OLP0)
{
  int spin, a, b;
  int l, fan, j, k, i, jg, kg, ig, wakg, waig, kl, il1, il2, m, n;
  int Cwan, Hwan, Mul1, tno0, tno1, tno2;
  int Mc_AN, Gc_AN, Mj_AN, Mi_AN, Mk_AN;
  double tmp0;
  double Resum00, Resum11, Resum01;
  double Imsum00, Imsum11, Imsum01;

  /****************************************************
    if (SpinP_switch!=3)

    collinear case
  ****************************************************/

  if (SpinP_switch != 3)
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      for (j = 0; j <= fan; j++)
      {
        jg = natn[Gc_AN][j];
        Mj_AN = F_G2M[jg];
        Hwan = WhatSpecies[jg];
        tno1 = Spe_Total_NO[Hwan];

        for (spin = 0; spin <= SpinP_switch; spin++)
        {
          for (m = 0; m < tno0; m++)
          {
            for (n = 0; n < tno1; n++)
            {

              tmp0 = 0.0;

              for (k = 0; k < tno0; k++)
              {
                tmp0 += v_eff[spin][Mc_AN][m][k] * OLP0[Mc_AN][j][k][n];
              }

              for (k = 0; k < tno1; k++)
              {
                tmp0 += v_eff[spin][Mj_AN][k][n] * OLP0[Mc_AN][j][m][k];
              }

              H_Hub[spin][Mc_AN][j][m][n] = 0.5 * tmp0;
            }
          }
        }
      }
    }
  }

  /****************************************************
    if (SpinP_switch==3)

    spin non-collinear
  ****************************************************/

  else
  {

    for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++)
    {

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      fan = FNAN[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      for (j = 0; j <= fan; j++)
      {
        jg = natn[Gc_AN][j];
        Mj_AN = F_G2M[jg];
        Hwan = WhatSpecies[jg];
        tno1 = Spe_Total_NO[Hwan];

        for (m = 0; m < tno0; m++)
        {
          for (n = 0; n < tno1; n++)
          {

            Resum00 = 0.0;
            Resum11 = 0.0;
            Resum01 = 0.0;

            Imsum00 = 0.0;
            Imsum11 = 0.0;
            Imsum01 = 0.0;

            for (k = 0; k < tno0; k++)
            {

              Resum00 += NC_v_eff[0][0][Mc_AN][m][k].r * OLP0[Mc_AN][j][k][n];
              Resum11 += NC_v_eff[1][1][Mc_AN][m][k].r * OLP0[Mc_AN][j][k][n];
              Resum01 += NC_v_eff[0][1][Mc_AN][m][k].r * OLP0[Mc_AN][j][k][n];

              Imsum00 += NC_v_eff[0][0][Mc_AN][m][k].i * OLP0[Mc_AN][j][k][n];
              Imsum11 += NC_v_eff[1][1][Mc_AN][m][k].i * OLP0[Mc_AN][j][k][n];
              Imsum01 += NC_v_eff[0][1][Mc_AN][m][k].i * OLP0[Mc_AN][j][k][n];
            }

            for (k = 0; k < tno1; k++)
            {

              Resum00 += NC_v_eff[0][0][Mj_AN][k][n].r * OLP0[Mc_AN][j][m][k];
              Resum11 += NC_v_eff[1][1][Mj_AN][k][n].r * OLP0[Mc_AN][j][m][k];
              Resum01 += NC_v_eff[0][1][Mj_AN][k][n].r * OLP0[Mc_AN][j][m][k];

              Imsum00 += NC_v_eff[0][0][Mj_AN][k][n].i * OLP0[Mc_AN][j][m][k];
              Imsum11 += NC_v_eff[1][1][Mj_AN][k][n].i * OLP0[Mc_AN][j][m][k];
              Imsum01 += NC_v_eff[0][1][Mj_AN][k][n].i * OLP0[Mc_AN][j][m][k];
            }

            /* real part */

            H_Hub[0][Mc_AN][j][m][n] = 0.5 * Resum00;
            H_Hub[1][Mc_AN][j][m][n] = 0.5 * Resum11;
            H_Hub[2][Mc_AN][j][m][n] = 0.5 * Resum01;

            /* imaginary part */

            iHNL[0][Mc_AN][j][m][n] = iHNL0[0][Mc_AN][j][m][n] + 0.5 * F_U_flag * Imsum00;
            iHNL[1][Mc_AN][j][m][n] = iHNL0[1][Mc_AN][j][m][n] + 0.5 * F_U_flag * Imsum11;
            iHNL[2][Mc_AN][j][m][n] = iHNL0[2][Mc_AN][j][m][n] + 0.5 * F_U_flag * Imsum01;
          }
        }
      }
    }
  }
}
