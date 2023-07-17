#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "mpi.h"

void array0();
void array1();
void array2();
void array3();

void Free_Arrays(int wherefrom)
{

  if (wherefrom == 0)
    array0();
  else if (wherefrom == 1)
    array1();
}

void array0()
{
  int i, j, k, ii, l, L, n, ct_AN, h_AN, wan, al, tno, Cwan, Nc;
  int tno0, tno1, tno2, Mc_AN, Gc_AN, Gh_AN, Hwan, m, so, s1, s2;
  int q_AN, Gq_AN, Qwan, Lmax, spe, ns, nc, spin, fan;
  int num, n2, wanA, wanB, Gi, MAnum, Mh_AN, NO1;
  int Anum, p, vsize, NUM;
  int numprocs, myid, ID;
  int a, b, c; /* S.Ryee */

  /* MPI */

  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  /* call from openmx.c */

  /* allocate in truncation.c */

  if (alloc_first[0] == 0)
  {

    FNAN[0] = 0;
    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {

      if (Mc_AN == 0)
        Gc_AN = 0;
      else
        Gc_AN = M2G[Mc_AN];

      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {
        free(GListTAtoms2[Mc_AN][h_AN]);
        free(GListTAtoms1[Mc_AN][h_AN]);
      }
      free(GListTAtoms2[Mc_AN]);
      free(GListTAtoms1[Mc_AN]);
    }
    free(GListTAtoms2);
    free(GListTAtoms1);
  }

  if (alloc_first[1] == 0)
  {
  }

  /* Allocation in UCell_Box() of truncation.c */

  if (alloc_first[2] == 0)
  {

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      free(MGridListAtom[Mc_AN]);
    }
    free(MGridListAtom);

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      free(GridListAtom[Mc_AN]);
    }
    free(GridListAtom);

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      free(CellListAtom[Mc_AN]);
    }
    free(CellListAtom);
  }

  /* Allocation in truncation.c */

  if (alloc_first[3] == 0)
  {

    if (SpinP_switch == 3)
    { /* spin non-collinear */

      for (k = 0; k <= 3; k++)
      {
        free(Density_Grid[k]);
      }
      free(Density_Grid);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Density_Grid[k]);
      }
      free(Density_Grid);
    }

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Vxc_Grid[k]);
      }
      free(Vxc_Grid);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Vxc_Grid[k]);
      }
      free(Vxc_Grid);
    }

    free(RefVxc_Grid);
    free(dVHart_Grid);

    if ((core_hole_state_flag == 1 && Scf_RestartFromFile == 1) || scf_coulomb_cutoff_CoreHole == 1)
    {
      free(dVHart_Periodic_Grid_B);
      free(Density_Periodic_Grid_B);
    }

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Vpot_Grid[k]);
      }
      free(Vpot_Grid);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Vpot_Grid[k]);
      }
      free(Vpot_Grid);
    }

    /* arrays for the partition B */

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Density_Grid_B[k]);
      }
      free(Density_Grid_B);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Density_Grid_B[k]);
      }
      free(Density_Grid_B);
    }

    free(ADensity_Grid_B);
    free(PCCDensity_Grid_B[1]);
    free(PCCDensity_Grid_B[0]);
    free(dVHart_Grid_B);
    free(RefVxc_Grid_B);

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Vxc_Grid_B[k]);
      }
      free(Vxc_Grid_B);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Vxc_Grid_B[k]);
      }
      free(Vxc_Grid_B);
    }

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Vpot_Grid_B[k]);
      }
      free(Vpot_Grid_B);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Vpot_Grid_B[k]);
      }
      free(Vpot_Grid_B);
    }

    /* if ( Mixing_switch==7 ) */

    if (Mixing_switch == 7)
    {

      int spinmax;

      if (SpinP_switch == 0)
        spinmax = 1;
      else if (SpinP_switch == 1)
        spinmax = 2;
      else if (SpinP_switch == 3)
        spinmax = 3;

      for (m = 0; m < List_YOUSO[38]; m++)
      {
        for (spin = 0; spin < spinmax; spin++)
        {
          free(ReVKSk[m][spin]);
        }
        free(ReVKSk[m]);
      }
      free(ReVKSk);

      for (m = 0; m < List_YOUSO[38]; m++)
      {
        for (spin = 0; spin < spinmax; spin++)
        {
          free(ImVKSk[m][spin]);
        }
        free(ImVKSk[m]);
      }
      free(ImVKSk);

      for (spin = 0; spin < spinmax; spin++)
      {
        free(Residual_ReVKSk[spin]);
      }
      free(Residual_ReVKSk);

      for (spin = 0; spin < spinmax; spin++)
      {
        free(Residual_ImVKSk[spin]);
      }
      free(Residual_ImVKSk);
    }

    /* if (ProExpn_VNA==off) */
    if (ProExpn_VNA == 0)
    {
      free(VNA_Grid);
      free(VNA_Grid_B);
    }

    /* electric energy by electric field */
    if (E_Field_switch == 1)
    {
      free(VEF_Grid);
      free(VEF_Grid_B);
    }

    /* arrays for the partition D */

    free(PCCDensity_Grid_D[1]);
    free(PCCDensity_Grid_D[0]);

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Density_Grid_D[k]);
      }
      free(Density_Grid_D);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Density_Grid_D[k]);
      }
      free(Density_Grid_D);
    }

    if (SpinP_switch == 3)
    { /* spin non-collinear */
      for (k = 0; k <= 3; k++)
      {
        free(Vxc_Grid_D[k]);
      }
      free(Vxc_Grid_D);
    }
    else
    {
      for (k = 0; k <= 1; k++)
      {
        free(Vxc_Grid_D[k]);
      }
      free(Vxc_Grid_D);
    }

    /* AITUNE */
    /* Orbs_Grid */
    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      if (Mc_AN == 0)
      {
        Gc_AN = 0;
        num = 1;
      }
      else
      {
        Gc_AN = F_M2G[Mc_AN];
        num = GridN_Atom[Gc_AN];
      }

      for (Nc = 0; Nc < num; Nc++)
      {
        free(Orbs_Grid[Mc_AN][Nc]);
      }
      free(Orbs_Grid[Mc_AN]);
    }
    free(Orbs_Grid);

    /* COrbs_Grid */
    if (Cnt_switch != 0)
    {
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF); Mc_AN++)
      {
        if (Mc_AN == 0)
        {
          tno = 1;
          Gc_AN = 0;
        }
        else
        {
          Gc_AN = F_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno = Spe_Total_CNO[Cwan];
        }

        for (i = 0; i < tno; i++)
        {
          free(COrbs_Grid[Mc_AN][i]);
        }
        free(COrbs_Grid[Mc_AN]);
      }
      free(COrbs_Grid);
    }

    /* Orbs_Grid_FNAN */

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {

      if (Mc_AN == 0)
      {
        free(Orbs_Grid_FNAN[0][0][0]);
        free(Orbs_Grid_FNAN[0][0]);
      }
      else
      {

        Gc_AN = M2G[Mc_AN];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          Gh_AN = natn[Gc_AN][h_AN];

          if (G2ID[Gh_AN] != myid)
          {

            Mh_AN = F_G2M[Gh_AN];
            Hwan = WhatSpecies[Gh_AN];
            NO1 = Spe_Total_NO[Hwan];
            num = NumOLG[Mc_AN][h_AN];
          }
          else
          {
            num = 1;
          }

          if (0 < NumOLG[Mc_AN][h_AN])
          {
            for (Nc = 0; Nc < num; Nc++)
            {
              free(Orbs_Grid_FNAN[Mc_AN][h_AN][Nc]);
            }
            free(Orbs_Grid_FNAN[Mc_AN][h_AN]);
          }

        } /* h_AN */
      }   /* else */

      free(Orbs_Grid_FNAN[Mc_AN]);
    }
    free(Orbs_Grid_FNAN);
  }

  /****************************************************
    Allocation in truncation.c

    freeing of arrays:

      H0
      CntH0
      HNL
      CntHNL
      OLP
      CntOLP
      H
      CntH
      DS_NL
      CntDS_NL
      DM
      DM_onsite
      NC_OcpN
      NC_v_eff
      ResidualDM
      EDM
      PDM
      iHNL
      iCntHNL
      H_Hub
      Coulomb_Array (if Hub_Type=2, by S.Ryee)
      AMF_Array (if Hub_Type=2 && dc_Type=2 or 4), by S.Ryee)
  ****************************************************/

  if (alloc_first[4] == 0)
  {

    /* H0 */

    for (k = 0; k < 4; k++)
    {
      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {
          for (i = 0; i < tno0; i++)
          {
            free(H0[k][Mc_AN][h_AN][i]);
          }
          free(H0[k][Mc_AN][h_AN]);
        }
        free(H0[k][Mc_AN]);
      }
      free(H0[k]);
    }
    free(H0);

    /* CntH0 */

    if (Cnt_switch == 1)
    {
      for (k = 0; k < 4; k++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_CNO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(CntH0[k][Mc_AN][h_AN][i]);
            }
            free(CntH0[k][Mc_AN][h_AN]);
          }
          free(CntH0[k][Mc_AN]);
        }
        free(CntH0[k]);
      }
      free(CntH0);
    }

    /* HNL */

    for (k = 0; k < List_YOUSO[5]; k++)
    {
      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
          FNAN[0] = 0;
        }
        else
        {
          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {
          for (i = 0; i < tno0; i++)
          {
            free(HNL[k][Mc_AN][h_AN][i]);
          }
          free(HNL[k][Mc_AN][h_AN]);
        }
        free(HNL[k][Mc_AN]);
      }
      free(HNL[k]);
    }
    free(HNL);

    /* iHNL */

    if (SpinP_switch == 3)
    {

      for (k = 0; k < List_YOUSO[5]; k++)
      {
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
          }
          else
          {
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(iHNL[k][Mc_AN][h_AN][i]);
            }
            free(iHNL[k][Mc_AN][h_AN]);
          }
          free(iHNL[k][Mc_AN]);
        }
        free(iHNL[k]);
      }
      free(iHNL);
    }

    /* iCntHNL */

    if (SO_switch == 1 && Cnt_switch == 1)
    {

      for (k = 0; k < List_YOUSO[5]; k++)
      {
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
          }
          else
          {
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_CNO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(iCntHNL[k][Mc_AN][h_AN][i]);
            }
            free(iCntHNL[k][Mc_AN][h_AN]);
          }
          free(iCntHNL[k][Mc_AN]);
        }
        free(iCntHNL[k]);
      }
      free(iCntHNL);
    }

    /* for core hole calculations */

    if (core_hole_state_flag == 1)
    {

      /* HCH */

      for (k = 0; k < List_YOUSO[5]; k++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(HCH[k][Mc_AN][h_AN][i]);
            }
            free(HCH[k][Mc_AN][h_AN]);
          }
          free(HCH[k][Mc_AN]);
        }
        free(HCH[k]);
      }
      free(HCH);

      /* iHCH */

      if (SpinP_switch == 3)
      {

        for (k = 0; k < List_YOUSO[5]; k++)
        {
          for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
              FNAN[0] = 0;
            }
            else
            {
              Gc_AN = S_M2G[Mc_AN];
              Cwan = WhatSpecies[Gc_AN];
              tno0 = Spe_Total_NO[Cwan];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
            {
              for (i = 0; i < tno0; i++)
              {
                free(iHCH[k][Mc_AN][h_AN][i]);
              }
              free(iHCH[k][Mc_AN][h_AN]);
            }
            free(iHCH[k][Mc_AN]);
          }
          free(iHCH[k]);
        }
        free(iHCH);
      }
    }

    /* H_Hub  --- added by MJ */

    if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
    {

      for (k = 0; k <= SpinP_switch; k++)
      {

        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(H_Hub[k][Mc_AN][h_AN][i]);
            }
            free(H_Hub[k][Mc_AN][h_AN]);
          } /* h_AN */
          free(H_Hub[k][Mc_AN]);
        } /* Mc_AN */
        free(H_Hub[k]);
      } /* k */
      free(H_Hub);
    }

    /* Coulomb_Array by S.Ryee */

    if (Hub_U_switch == 1 && Hub_Type == 2)
    {
      for (i = 0; i <= Nmul; i++)
      {
        for (a = 0; a < 7; a++)
        {
          for (b = 0; b < 7; b++)
          {
            for (c = 0; c < 7; c++)
            {
              free(Coulomb_Array[i][a][b][c]);
            }
            free(Coulomb_Array[i][a][b]);
          }
          free(Coulomb_Array[i][a]);
        }
        free(Coulomb_Array[i]);
      }
      free(Coulomb_Array);
    }

    /* AMF_Array by S.Ryee */

    if (Hub_U_switch == 1 && Hub_Type == 2 && (dc_Type == 2 || dc_Type == 4))
    {
      for (i = 0; i <= Nmul; i++)
      {
        for (a = 0; a < 4; a++)
        {
          for (b = 0; b < 2; b++)
          {
            for (c = 0; c < 7; c++)
            {
              free(AMF_Array[i][a][b][c]);
            }
            free(AMF_Array[i][a][b]);
          }
          free(AMF_Array[i][a]);
        }
        free(AMF_Array[i]);
      }
      free(AMF_Array);
    }

    /* H_Zeeman_NCO */

    if (Zeeman_NCO_switch == 1)
    {

      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (i = 0; i < tno0; i++)
        {
          free(H_Zeeman_NCO[Mc_AN][i]);
        }
        free(H_Zeeman_NCO[Mc_AN]);
      }
      free(H_Zeeman_NCO);
    }

    /* iHNL0 */

    if (SpinP_switch == 3)
    {

      for (k = 0; k < List_YOUSO[5]; k++)
      {

        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(iHNL0[k][Mc_AN][h_AN][i]);
            }
            free(iHNL0[k][Mc_AN][h_AN]);
          }
          free(iHNL0[k][Mc_AN]);
        }
        free(iHNL0[k]);
      }
      free(iHNL0);
    }

    /* OLP_L */

    for (k = 0; k < 3; k++)
    {

      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
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
          for (i = 0; i < tno0; i++)
          {
            free(OLP_L[k][Mc_AN][h_AN][i]);
          }
          free(OLP_L[k][Mc_AN][h_AN]);
        }
        free(OLP_L[k][Mc_AN]);
      }
      free(OLP_L[k]);
    }
    free(OLP_L);

    /* OLP */

    for (k = 0; k < 4; k++)
    {
      for (Mc_AN = 0; Mc_AN < (Matomnum + MatomnumF + MatomnumS + 2); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
          FNAN[0] = 0;
          fan = FNAN[Gc_AN];
        }
        else if (Mc_AN == (Matomnum + 1))
        {
          tno0 = List_YOUSO[7];
          fan = List_YOUSO[8];
        }
        else if ((Hub_U_switch == 0 || Hub_U_occupation != 1 || core_hole_state_flag != 1) && 0 < k && (Matomnum + 1) < Mc_AN && Mc_AN <= (Matomnum + MatomnumF + MatomnumS))
        {

          Gc_AN = S_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = 1;
          fan = FNAN[Gc_AN];
        }
        else if (Mc_AN <= (Matomnum + MatomnumF + MatomnumS))
        {
          Gc_AN = S_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
          fan = FNAN[Gc_AN];
        }
        else
        {
          tno0 = List_YOUSO[7];
          fan = List_YOUSO[8];
        }

        for (h_AN = 0; h_AN <= fan; h_AN++)
        {
          for (i = 0; i < tno0; i++)
          {
            free(OLP[k][Mc_AN][h_AN][i]);
          }
          free(OLP[k][Mc_AN][h_AN]);
        }
        free(OLP[k][Mc_AN]);
      }
      free(OLP[k]);
    }
    free(OLP);

    /*** added by Ohwaki ***/

    /* OLP_p */

    if (0 <= CLE_Type)
    {

      for (k = 0; k < 4; k++)
      {
        for (Mc_AN = 0; Mc_AN < (Matomnum + MatomnumF + MatomnumS + 2); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
            fan = FNAN[Gc_AN];
          }
          else if (Mc_AN == (Matomnum + 1))
          {
            tno0 = List_YOUSO[7];
            fan = List_YOUSO[8];
          }
          else if (Mc_AN <= (Matomnum + MatomnumF + MatomnumS))
          {
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
            fan = FNAN[Gc_AN];
          }
          else
          {
            tno0 = List_YOUSO[7];
            fan = List_YOUSO[8];
          }

          for (h_AN = 0; h_AN <= fan; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(OLP_p[k][Mc_AN][h_AN][i]);
            }
            free(OLP_p[k][Mc_AN][h_AN]);
          }
          free(OLP_p[k][Mc_AN]);
        }
        free(OLP_p[k]);
      }
      free(OLP_p);
    }

    /*** added by Ohwaki (end) ***/

    /* OLP_CH */

    if (core_hole_state_flag == 1)
    {

      for (k = 0; k < 4; k++)
      {
        for (Mc_AN = 0; Mc_AN < (Matomnum + MatomnumF + MatomnumS + 2); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
            fan = FNAN[Gc_AN];
          }
          else if (Mc_AN == (Matomnum + 1))
          {
            tno0 = List_YOUSO[7];
            fan = List_YOUSO[8];
          }
          else if (Mc_AN <= (Matomnum + MatomnumF + MatomnumS))
          {
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
            fan = FNAN[Gc_AN];
          }
          else
          {
            tno0 = List_YOUSO[7];
            fan = List_YOUSO[8];
          }

          for (h_AN = 0; h_AN <= fan; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(OLP_CH[k][Mc_AN][h_AN][i]);
            }
            free(OLP_CH[k][Mc_AN][h_AN]);
          }
          free(OLP_CH[k][Mc_AN]);
        }
        free(OLP_CH[k]);
      }
      free(OLP_CH);
    }

    /* CntOLP */

    if (Cnt_switch == 1)
    {

      for (k = 0; k < 4; k++)
      {
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            FNAN[0] = 0;
          }
          else
          {
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_CNO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(CntOLP[k][Mc_AN][h_AN][i]);
            }
            free(CntOLP[k][Mc_AN][h_AN]);
          }
          free(CntOLP[k][Mc_AN]);
        }
        free(CntOLP[k]);
      }
      free(CntOLP);
    }

    /* H */

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = S_M2G[Mc_AN];
          Cwan = WhatSpecies[Gc_AN];
          tno0 = Spe_Total_NO[Cwan];
        }

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          for (i = 0; i < tno0; i++)
          {
            free(H[k][Mc_AN][h_AN][i]);
          }
          free(H[k][Mc_AN][h_AN]);
        }
        free(H[k][Mc_AN]);
      }
      free(H[k]);
    }
    free(H);

    /* CntH */

    if (Cnt_switch == 1)
    {

      for (k = 0; k <= SpinP_switch; k++)
      {
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
          }
          else
          {
            Gc_AN = S_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_CNO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(CntH[k][Mc_AN][h_AN][i]);
            }
            free(CntH[k][Mc_AN][h_AN]);
          }
          free(CntH[k][Mc_AN]);
        }
        free(CntH[k]);
      }
      free(CntH);
    }

    /* DS_NL */

    for (so = 0; so < (SO_switch + 1); so++)
    {
      for (k = 0; k < 4; k++)
      {
        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN < (Matomnum + 2); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            fan = FNAN[Gc_AN];
          }
          else if ((Matomnum + 1) <= Mc_AN)
          {
            fan = List_YOUSO[8];
            tno0 = List_YOUSO[7];
          }
          else
          {
            Gc_AN = F_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
            fan = FNAN[Gc_AN];
          }

          for (h_AN = 0; h_AN < (fan + 1); h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(DS_NL[so][k][Mc_AN][h_AN][i]);
            }
            free(DS_NL[so][k][Mc_AN][h_AN]);
          }
          free(DS_NL[so][k][Mc_AN]);
        }
        free(DS_NL[so][k]);
      }
      free(DS_NL[so]);
    }
    free(DS_NL);

    /* CntDS_NL */

    if (Cnt_switch == 1)
    {

      for (so = 0; so < (SO_switch + 1); so++)
      {
        for (k = 0; k < 4; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN < (Matomnum + 2); Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
              fan = FNAN[Gc_AN];
            }
            else if ((Matomnum + 1) <= Mc_AN)
            {
              fan = List_YOUSO[8];
              tno0 = List_YOUSO[7];
            }
            else
            {
              Gc_AN = F_M2G[Mc_AN];
              Cwan = WhatSpecies[Gc_AN];
              tno0 = Spe_Total_CNO[Cwan];
              fan = FNAN[Gc_AN];
            }

            for (h_AN = 0; h_AN < (fan + 1); h_AN++)
            {
              for (i = 0; i < tno0; i++)
              {
                free(CntDS_NL[so][k][Mc_AN][h_AN][i]);
              }
              free(CntDS_NL[so][k][Mc_AN][h_AN]);
            }
            free(CntDS_NL[so][k][Mc_AN]);
          }
          free(CntDS_NL[so][k]);
        }
        free(CntDS_NL[so]);
      }
      free(CntDS_NL);
    }

    /* LNO_coes */

    if (LNO_flag == 1)
    {

      for (k = 0; k <= SpinP_switch; k++)
      {
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
        {
          free(LNO_coes[k][Mc_AN]);
        }
        free(LNO_coes[k]);
      }
      free(LNO_coes);

      for (k = 0; k <= SpinP_switch; k++)
      {
        for (Mc_AN = 0; Mc_AN <= (Matomnum + MatomnumF + MatomnumS); Mc_AN++)
        {
          free(LNO_pops[k][Mc_AN]);
        }
        free(LNO_pops[k]);
      }
      free(LNO_pops);

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
    }

    /* for RMM-DIISH */

    if (Mixing_switch == 5)
    {

      /* HisH1 */

      for (m = 0; m < List_YOUSO[39]; m++)
      {
        for (k = 0; k <= SpinP_switch; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
            }
            else
            {
              Gc_AN = S_M2G[Mc_AN];
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
                free(HisH1[m][k][Mc_AN][h_AN][i]);
              }
              free(HisH1[m][k][Mc_AN][h_AN]);
            }
            free(HisH1[m][k][Mc_AN]);
          }
          free(HisH1[m][k]);
        }
        free(HisH1[m]);
      }
      free(HisH1);

      /* HisH2 */

      if (SpinP_switch == 3)
      {

        for (m = 0; m < List_YOUSO[39]; m++)
        {
          for (k = 0; k < SpinP_switch; k++)
          {
            FNAN[0] = 0;
            for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
            {

              if (Mc_AN == 0)
              {
                Gc_AN = 0;
                tno0 = 1;
              }
              else
              {
                Gc_AN = S_M2G[Mc_AN];
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
                  free(HisH2[m][k][Mc_AN][h_AN][i]);
                }
                free(HisH2[m][k][Mc_AN][h_AN]);
              }
              free(HisH2[m][k][Mc_AN]);
            }
            free(HisH2[m][k]);
          }
          free(HisH2[m]);
        }
        free(HisH2);

      } /* if (SpinP_switch==3) */

      /* ResidualH1 */

      for (m = 0; m < List_YOUSO[39]; m++)
      {
        for (k = 0; k <= SpinP_switch; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
            }
            else
            {
              Gc_AN = S_M2G[Mc_AN];
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
                free(ResidualH1[m][k][Mc_AN][h_AN][i]);
              }
              free(ResidualH1[m][k][Mc_AN][h_AN]);
            }
            free(ResidualH1[m][k][Mc_AN]);
          }
          free(ResidualH1[m][k]);
        }
        free(ResidualH1[m]);
      }
      free(ResidualH1);

      /* ResidualH2 */

      if (SpinP_switch == 3)
      {

        for (m = 0; m < List_YOUSO[39]; m++)
        {
          for (k = 0; k < SpinP_switch; k++)
          {
            FNAN[0] = 0;
            for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
            {

              if (Mc_AN == 0)
              {
                Gc_AN = 0;
                tno0 = 1;
              }
              else
              {
                Gc_AN = S_M2G[Mc_AN];
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
                  free(ResidualH2[m][k][Mc_AN][h_AN][i]);
                }
                free(ResidualH2[m][k][Mc_AN][h_AN]);
              }
              free(ResidualH2[m][k][Mc_AN]);
            }
            free(ResidualH2[m][k]);
          }
          free(ResidualH2[m]);
        }
        free(ResidualH2);

      } /* if (SpinP_switch==3) */

    } /* if (Mixing_switch==5) */

    /* for RMM-DIIS */

    if (Mixing_switch == 1 || Mixing_switch == 6)
    {

      /* HisH1 */

      for (m = 0; m < List_YOUSO[39]; m++)
      {
        for (k = 0; k <= SpinP_switch; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
            }
            else
            {
              Gc_AN = S_M2G[Mc_AN];
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
                free(HisH1[m][k][Mc_AN][h_AN][i]);
              }
              free(HisH1[m][k][Mc_AN][h_AN]);
            }
            free(HisH1[m][k][Mc_AN]);
          }
          free(HisH1[m][k]);
        }
        free(HisH1[m]);
      }
      free(HisH1);

      /* HisH2 */

      if (SpinP_switch == 3)
      {

        for (m = 0; m < List_YOUSO[39]; m++)
        {
          for (k = 0; k < SpinP_switch; k++)
          {
            FNAN[0] = 0;
            for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
            {

              if (Mc_AN == 0)
              {
                Gc_AN = 0;
                tno0 = 1;
              }
              else
              {
                Gc_AN = S_M2G[Mc_AN];
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
                  free(HisH2[m][k][Mc_AN][h_AN][i]);
                }
                free(HisH2[m][k][Mc_AN][h_AN]);
              }
              free(HisH2[m][k][Mc_AN]);
            }
            free(HisH2[m][k]);
          }
          free(HisH2[m]);
        }
        free(HisH2);

      } /* if (SpinP_switch==3) */

    } /* if (Mixing_switch==1 || Mixing_switch==6) */

    /* DM */

    for (m = 0; m < List_YOUSO[16]; m++)
    {
      for (k = 0; k <= SpinP_switch; k++)
      {
        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(DM[m][k][Mc_AN][h_AN][i]);
            }
            free(DM[m][k][Mc_AN][h_AN]);
          }
          free(DM[m][k][Mc_AN]);
        }
        free(DM[m][k]);
      }
      free(DM[m]);
    }
    free(DM);

    /* Partial_DM */

    if (cal_partial_charge == 1)
    {

      for (k = 0; k <= 1; k++)
      {

        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {
            for (i = 0; i < tno0; i++)
            {
              free(Partial_DM[k][Mc_AN][h_AN][i]);
            }
            free(Partial_DM[k][Mc_AN][h_AN]);
          }
          free(Partial_DM[k][Mc_AN]);
        }
        free(Partial_DM[k]);
      }
      free(Partial_DM);
    }

    /* DM_onsite added by MJ */

    if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
    {

      for (m = 0; m < 2; m++)
      {
        for (k = 0; k <= SpinP_switch; k++)
        {

          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
            }
            else
            {
              Gc_AN = M2G[Mc_AN];
              Cwan = WhatSpecies[Gc_AN];
              tno0 = Spe_Total_NO[Cwan];
            }

            for (i = 0; i < tno0; i++)
            {
              free(DM_onsite[m][k][Mc_AN][i]);
            }
            free(DM_onsite[m][k][Mc_AN]);
          }
          free(DM_onsite[m][k]);
        }
        free(DM_onsite[m]);
      }
      free(DM_onsite);
    }

    /* v_eff added by MJ */

    if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
    {

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

          for (i = 0; i < tno0; i++)
          {
            free(v_eff[k][Mc_AN][i]);
          }
          free(v_eff[k][Mc_AN]);

        } /* for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++) */
        free(v_eff[k]);

      } /* for (k=0; k<=SpinP_switch; k++) */
      free(v_eff);
    }

    /*  NC_OcpN */

    if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) && SpinP_switch == 3)
    {

      for (m = 0; m < 2; m++)
      {
        for (s1 = 0; s1 < 2; s1++)
        {
          for (s2 = 0; s2 < 2; s2++)
          {
            for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
            {

              if (Mc_AN == 0)
              {
                Gc_AN = 0;
                tno0 = 1;
              }
              else
              {
                Gc_AN = M2G[Mc_AN];
                Cwan = WhatSpecies[Gc_AN];
                tno0 = Spe_Total_NO[Cwan];
              }

              for (i = 0; i < tno0; i++)
              {
                free(NC_OcpN[m][s1][s2][Mc_AN][i]);
              }
              free(NC_OcpN[m][s1][s2][Mc_AN]);
            }
            free(NC_OcpN[m][s1][s2]);
          }
          free(NC_OcpN[m][s1]);
        }
        free(NC_OcpN[m]);
      }
      free(NC_OcpN);
    }

    /*  NC_v_eff */

    if ((Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1) && SpinP_switch == 3)
    {

      for (s1 = 0; s1 < 2; s1++)
      {
        for (s2 = 0; s2 < 2; s2++)
        {
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

            for (i = 0; i < tno0; i++)
            {
              free(NC_v_eff[s1][s2][Mc_AN][i]);
            }
            free(NC_v_eff[s1][s2][Mc_AN]);
          }
          free(NC_v_eff[s1][s2]);
        }
        free(NC_v_eff[s1]);
      }
      free(NC_v_eff);
    }

    /* ResidualDM */

    if (Mixing_switch == 0 || Mixing_switch == 1 || Mixing_switch == 2 || Mixing_switch == 6)
    {

      for (m = 0; m < List_YOUSO[16]; m++)
      {
        for (k = 0; k <= SpinP_switch; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
            }
            else
            {
              Gc_AN = M2G[Mc_AN];
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
                free(ResidualDM[m][k][Mc_AN][h_AN][i]);
              }
              free(ResidualDM[m][k][Mc_AN][h_AN]);
            }
            free(ResidualDM[m][k][Mc_AN]);
          }
          free(ResidualDM[m][k]);
        }
        free(ResidualDM[m]);
      }
      free(ResidualDM);
    }

    /* iResidualDM */

    if ((Mixing_switch == 0 || Mixing_switch == 1 || Mixing_switch == 2 || Mixing_switch == 6) && SpinP_switch == 3 && (SO_switch == 1 || Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1))
    {

      for (m = 0; m < List_YOUSO[16]; m++)
      {
        for (k = 0; k < 2; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
            }
            else
            {
              Gc_AN = M2G[Mc_AN];
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
                free(iResidualDM[m][k][Mc_AN][h_AN][i]);
              }
              free(iResidualDM[m][k][Mc_AN][h_AN]);
            }
            free(iResidualDM[m][k][Mc_AN]);
          }
          free(iResidualDM[m][k]);
        }
        free(iResidualDM[m]);
      }
      free(iResidualDM);
    }
    else
    {
      for (m = 0; m < List_YOUSO[16]; m++)
      {
        free(iResidualDM[m][0][0][0][0]);
        free(iResidualDM[m][0][0][0]);
        free(iResidualDM[m][0][0]);
        free(iResidualDM[m][0]);
        free(iResidualDM[m]);
      }
      free(iResidualDM);
    }

    /* EDM */

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = M2G[Mc_AN];
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
            free(EDM[k][Mc_AN][h_AN][i]);
          }

          free(EDM[k][Mc_AN][h_AN]);
        }
        free(EDM[k][Mc_AN]);
      }
      free(EDM[k]);
    }
    free(EDM);

    /* PDM */

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = M2G[Mc_AN];
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
            free(PDM[k][Mc_AN][h_AN][i]);
          }

          free(PDM[k][Mc_AN][h_AN]);
        }
        free(PDM[k][Mc_AN]);
      }
      free(PDM[k]);
    }
    free(PDM);

    /* iDM */

    for (m = 0; m < List_YOUSO[16]; m++)
    {
      for (k = 0; k < 2; k++)
      {

        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[Mc_AN];
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
              free(iDM[m][k][Mc_AN][h_AN][i]);
            }
            free(iDM[m][k][Mc_AN][h_AN]);
          }
          free(iDM[m][k][Mc_AN]);
        }
        free(iDM[m][k]);
      }
      free(iDM[m]);
    }
    free(iDM);

    /* S12 */

    if (Solver == 1 || Solver == 5)
    {

      int *Msize, myid1;

      Msize = (int *)malloc(sizeof(int) * (Matomnum + 2));

      if (Solver == 1 || Solver == 5)
      {

        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
            n2 = 1;
          else
          {
            Gc_AN = M2G[Mc_AN];

            num = 1;
            for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++)
            {
              Gi = natn[Gc_AN][i];
              wanA = WhatSpecies[Gi];
              num += Spe_Total_CNO[wanA];
            }
            n2 = num + 2;
          }

          Msize[Mc_AN] = n2;

          for (i = 0; i < n2; i++)
          {
            free(S12[Mc_AN][i]);
          }
          free(S12[Mc_AN]);
        }
        free(S12);
      }

      free(Msize);
    }

    if (Cnt_switch == 1)
    {
      for (i = 0; i <= (Matomnum + MatomnumF); i++)
      {
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          free(CntCoes[i][j]);
        }
        free(CntCoes[i]);
      }
      free(CntCoes);

      for (i = 0; i < (SpeciesNum + 1); i++)
      {
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          free(CntCoes_Species[i][j]);
        }
        free(CntCoes_Species[i]);
      }
      free(CntCoes_Species);
    }

    if (ProExpn_VNA == 1)
    {

      /* HVNA */

      FNAN[0] = 0;
      for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
      {

        if (Mc_AN == 0)
        {
          Gc_AN = 0;
          tno0 = 1;
        }
        else
        {
          Gc_AN = M2G[Mc_AN];
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
            free(HVNA[Mc_AN][h_AN][i]);
          }
          free(HVNA[Mc_AN][h_AN]);
        }
        free(HVNA[Mc_AN]);
      }
      free(HVNA);

      /* DS_VNA */

      for (k = 0; k < 4; k++)
      {

        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN < (Matomnum + 2); Mc_AN++)
        {

          if (Mc_AN == 0)
          {
            Gc_AN = 0;
            tno0 = 1;
            fan = FNAN[Gc_AN];
          }
          else if ((Matomnum + 1) <= Mc_AN)
          {
            fan = List_YOUSO[8];
            tno0 = List_YOUSO[7];
          }
          else
          {
            Gc_AN = F_M2G[Mc_AN];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
            fan = FNAN[Gc_AN];
          }

          for (h_AN = 0; h_AN < (fan + 1); h_AN++)
          {

            if (Mc_AN == 0)
            {
              tno1 = 1;
            }
            else
            {
              tno1 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
            }

            for (i = 0; i < tno0; i++)
            {
              free(DS_VNA[k][Mc_AN][h_AN][i]);
            }
            free(DS_VNA[k][Mc_AN][h_AN]);
          }
          free(DS_VNA[k][Mc_AN]);
        }
        free(DS_VNA[k]);
      }
      free(DS_VNA);

      /* CntDS_VNA */

      if (Cnt_switch == 1)
      {

        for (k = 0; k < 4; k++)
        {

          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN < (Matomnum + 2); Mc_AN++)
          {

            if (Mc_AN == 0)
            {
              Gc_AN = 0;
              tno0 = 1;
              fan = FNAN[Gc_AN];
            }
            else if (Mc_AN == (Matomnum + 1))
            {
              fan = List_YOUSO[8];
              tno0 = List_YOUSO[7];
            }
            else
            {
              Gc_AN = F_M2G[Mc_AN];
              Cwan = WhatSpecies[Gc_AN];
              tno0 = Spe_Total_CNO[Cwan];
              fan = FNAN[Gc_AN];
            }

            for (h_AN = 0; h_AN < (fan + 1); h_AN++)
            {

              if (Mc_AN == 0)
              {
                tno1 = 1;
              }
              else
              {
                tno1 = (List_YOUSO[35] + 1) * (List_YOUSO[35] + 1) * List_YOUSO[34];
              }

              for (i = 0; i < tno0; i++)
              {
                free(CntDS_VNA[k][Mc_AN][h_AN][i]);
              }
              free(CntDS_VNA[k][Mc_AN][h_AN]);
            }
            free(CntDS_VNA[k][Mc_AN]);
          }
          free(CntDS_VNA[k]);
        }
        free(CntDS_VNA);
      }

      /* HVNA2 */

      for (k = 0; k < 4; k++)
      {
        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
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
            for (i = 0; i < tno0; i++)
            {
              free(HVNA2[k][Mc_AN][h_AN][i]);
            }
            free(HVNA2[k][Mc_AN][h_AN]);
          }
          free(HVNA2[k][Mc_AN]);
        }
        free(HVNA2[k]);
      }
      free(HVNA2);

      /* HVNA3 */

      for (k = 0; k < 4; k++)
      {
        FNAN[0] = 0;
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {

          if (Mc_AN == 0)
            Gc_AN = 0;
          else
            Gc_AN = F_M2G[Mc_AN];

          for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
          {

            if (Mc_AN == 0)
            {
              tno0 = 1;
            }
            else
            {
              Gh_AN = natn[Gc_AN][h_AN];
              Hwan = WhatSpecies[Gh_AN];
              tno0 = Spe_Total_NO[Hwan];
            }

            for (i = 0; i < tno0; i++)
            {
              free(HVNA3[k][Mc_AN][h_AN][i]);
            }
            free(HVNA3[k][Mc_AN][h_AN]);
          }
          free(HVNA3[k][Mc_AN]);
        }
        free(HVNA3[k]);
      }
      free(HVNA3);

      /* CntHVNA2 */

      if (Cnt_switch == 1)
      {

        for (k = 0; k < 4; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
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
              tno0 = Spe_Total_CNO[Cwan];
            }

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
            {
              for (i = 0; i < tno0; i++)
              {
                free(CntHVNA2[k][Mc_AN][h_AN][i]);
              }
              free(CntHVNA2[k][Mc_AN][h_AN]);
            }
            free(CntHVNA2[k][Mc_AN]);
          }
          free(CntHVNA2[k]);
        }
        free(CntHVNA2);
      }

      /* CntHVNA3 */

      if (Cnt_switch == 1)
      {

        for (k = 0; k < 4; k++)
        {
          FNAN[0] = 0;
          for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
          {

            if (Mc_AN == 0)
              Gc_AN = 0;
            else
              Gc_AN = F_M2G[Mc_AN];

            for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
            {

              if (Mc_AN == 0)
              {
                tno0 = 1;
              }
              else
              {
                Gh_AN = natn[Gc_AN][h_AN];
                Hwan = WhatSpecies[Gh_AN];
                tno0 = Spe_Total_CNO[Hwan];
              }

              for (i = 0; i < tno0; i++)
              {
                free(CntHVNA3[k][Mc_AN][h_AN][i]);
              }
              free(CntHVNA3[k][Mc_AN][h_AN]);
            }
            free(CntHVNA3[k][Mc_AN]);
          }
          free(CntHVNA3[k]);
        }
        free(CntHVNA3);
      }
    }

    if (Solver == 8)
    { /* O(N) Krylov subspace method */

      for (i = 0; i <= SpinP_switch; i++)
      {
        for (j = 0; j <= Matomnum; j++)
        {

          if (j == 0)
          {
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[j];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          free(Krylov_U[i][j]);
        }
        free(Krylov_U[i]);
      }
      free(Krylov_U);

      for (i = 0; i <= SpinP_switch; i++)
      {
        for (j = 0; j <= Matomnum; j++)
        {

          if (j == 0)
          {
            tno0 = 1;
          }
          else
          {
            Gc_AN = M2G[j];
            Cwan = WhatSpecies[Gc_AN];
            tno0 = Spe_Total_NO[Cwan];
          }

          for (k = 0; k < (rlmax_EC[j] * EKC_core_size[j] + 1); k++)
          {
            free(EC_matrix[i][j][k]);
          }
          free(EC_matrix[i][j]);
        }
        free(EC_matrix[i]);
      }
      free(EC_matrix);

      free(rlmax_EC);
      free(rlmax_EC2);
      free(EKC_core_size);
      free(scale_rc_EKC);
    }

    /* NEGF */

    if (Solver == 4)
    {
      for (spin = 0; spin < (SpinP_switch + 1); spin++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {
          free(TRAN_DecMulP[spin][Mc_AN]);
        }
        free(TRAN_DecMulP[spin]);
      }
      free(TRAN_DecMulP);
    }

    /* Energy decomposition */

    if (Energy_Decomposition_flag == 1)
    {

      /* DecEkin */
      for (spin = 0; spin < 2; spin++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {
          free(DecEkin[spin][Mc_AN]);
        }
        free(DecEkin[spin]);
      }
      free(DecEkin);

      /* DecEv */
      for (spin = 0; spin < 2; spin++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {
          free(DecEv[spin][Mc_AN]);
        }
        free(DecEv[spin]);
      }
      free(DecEv);

      /* DecEcon */
      for (spin = 0; spin < 2; spin++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {
          free(DecEcon[spin][Mc_AN]);
        }
        free(DecEcon[spin]);
      }
      free(DecEcon);

      /* DecEscc */
      for (spin = 0; spin < 2; spin++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {
          free(DecEscc[spin][Mc_AN]);
        }
        free(DecEscc[spin]);
      }
      free(DecEscc);

      /* DecEvdw */
      for (spin = 0; spin < 2; spin++)
      {
        for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
        {
          free(DecEvdw[spin][Mc_AN]);
        }
        free(DecEvdw[spin]);
      }
      free(DecEvdw);

    } /* if (Energy_Decomposition_flag==1) */

  } /*  if (alloc_first[4]==0){ */

  if (alloc_first[5] == 0)
  {

    /* NumOLG */
    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      free(NumOLG[Mc_AN]);
    }
    free(NumOLG);

  } /*  if (alloc_first[5]==0){ */

  if (alloc_first[6] == 0)
  {

    FNAN[0] = 0;
    SNAN[0] = 0;

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      if (Mc_AN == 0)
        Gc_AN = 0;
      else
        Gc_AN = M2G[Mc_AN];
      for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++)
      {
        free(RMI1[Mc_AN][i]);
      }
      free(RMI1[Mc_AN]);
    }
    free(RMI1);

    for (Mc_AN = 0; Mc_AN <= Matomnum; Mc_AN++)
    {
      if (Mc_AN == 0)
        Gc_AN = 0;
      else
        Gc_AN = M2G[Mc_AN];
      for (i = 0; i <= (FNAN[Gc_AN] + SNAN[Gc_AN]); i++)
      {
        free(RMI2[Mc_AN][i]);
      }
      free(RMI2[Mc_AN]);
    }
    free(RMI2);

  } /* if (alloc_first[6]==0){ */

  if (alloc_first[9] == 0 && Solver == 2)
  {

    free(EV_S);
    free(IEV_S);
  }

  if (alloc_first[10] == 0)
  {
    free(M2G);
  }

  if (alloc_first[11] == 0)
  {
    for (ID = 0; ID < numprocs; ID++)
    {
      free(Snd_MAN[ID]);
    }
    free(Snd_MAN);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Snd_GAN[ID]);
    }
    free(Snd_GAN);
  }

  if (alloc_first[12] == 0)
  {

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Rcv_GAN[ID]);
    }
    free(Rcv_GAN);
  }

  if (alloc_first[13] == 0)
  {
    free(F_M2G);
    free(S_M2G);
  }

  if (alloc_first[22] == 0)
  {

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Pro_Snd_GAtom[ID]);
    }
    free(Pro_Snd_GAtom);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Pro_Snd_MAtom[ID]);
    }
    free(Pro_Snd_MAtom);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Pro_Snd_LAtom[ID]);
    }
    free(Pro_Snd_LAtom);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Pro_Snd_LAtom2[ID]);
    }
    free(Pro_Snd_LAtom2);
  }

  /* allocation in Set_BasisPara() of SetPara_DFT.c */

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    for (j = 0; j < Spe_Total_NO[i]; j++)
    {
      free(Spe_Trans_Orbital[i][j]);
    }
    free(Spe_Trans_Orbital[i]);
    free(Spe_Specified_Num[i]);
  }
  free(Spe_Trans_Orbital);
  free(Spe_Specified_Num);

  /* Spe_ProductRF_Bessel by Allocate_Arrays(7) in SetPara_DFT.c */

  if (ProExpn_VNA == 1)
  {

    for (i = 0; i < List_YOUSO[18]; i++)
    {
      for (j = 0; j < (Spe_MaxL_Basis[i] + 1); j++)
      {
        for (k = 0; k < Spe_Num_Basis[i][j]; k++)
        {
          for (l = 0; l < (Spe_MaxL_Basis[i] + 1); l++)
          {

            if (j <= l)
            {
              Lmax = 2 * l;
            }
            else
            {
              Lmax = 1;
            }

            for (m = 0; m < Spe_Num_Basis[i][l]; m++)
            {
              for (n = 0; n <= Lmax; n++)
              {
                free(Spe_ProductRF_Bessel[i][j][k][l][m][n]);
              }
              free(Spe_ProductRF_Bessel[i][j][k][l][m]);
            }
            free(Spe_ProductRF_Bessel[i][j][k][l]);
          }
          free(Spe_ProductRF_Bessel[i][j][k]);
        }
        free(Spe_ProductRF_Bessel[i][j]);
      }
      free(Spe_ProductRF_Bessel[i]);
    }
    free(Spe_ProductRF_Bessel);
  }

  if (alloc_first[23] == 0)
  {
    free(NE_T_k_op);
    free(NE_KGrids3);
    free(NE_KGrids2);
    free(NE_KGrids1);
  }

  /* Allocation_Arrays(0) */

  /* arrays for LDA+U added by MJ */
  if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
  {
    for (i = 0; i < SpeciesNum; i++)
    {
      for (l = 0; l < (Spe_MaxL_Basis[i] + 1); l++)
      {
        free(Hub_U_Basis[i][l]);
      }
      free(Hub_U_Basis[i]);
    }
    free(Hub_U_Basis);

    free(OrbPol_flag);
  }

  /* arrays for general LDA+U by S.Ryee */
  if (Hub_U_switch == 1 && Hub_Type == 2)
  {
    for (i = 0; i < SpeciesNum; i++)
    {
      for (l = 0; l < (Spe_MaxL_Basis[i] + 1); l++)
      {
        free(Hund_J_Basis[i][l]);
        free(Nonzero_UJ[i][l]);
      }
      free(Hund_J_Basis[i]);
      free(Nonzero_UJ[i]);
    }
    free(Hund_J_Basis);
    free(Nonzero_UJ);
  }

  /* arrays for DFTD-vdW okuno */
  if (dftD_switch == 1 && version_dftD == 2)
  {

    for (i = 0; i < SpeciesNum; i++)
    {
      free(C6ij_dftD[i]);
    }
    free(C6ij_dftD);

    for (i = 0; i < SpeciesNum; i++)
    {
      free(Rsum_dftD[i]);
    }
    free(Rsum_dftD);
  }
  /* okuno */

  /* arrays for DFTD3_vdW Ellner */

  if (dftD_switch == 1 && version_dftD == 3)
  {

    free(maxcn_dftD);
    for (i = 0; i < SpeciesNum; i++)
    {
      free(r0ab_dftD[i]);
    }
    free(r0ab_dftD);
    for (i = 0; i < SpeciesNum; i++)
    {
      free(r2r4ab_dftD[i]);
    }
    free(r2r4ab_dftD);
    for (i = 0; i < SpeciesNum; i++)
    {
      free(rcovab_dftD[i]);
    }
    free(rcovab_dftD);
    for (i = 0; i < SpeciesNum; i++)
    {
      for (j = 0; j < SpeciesNum; j++)
      {
        for (k = 0; k < 5; k++)
        {
          for (l = 0; l < 5; l++)
          {
            free(C6ab_dftD[i][j][k][l]);
          }
          free(C6ab_dftD[i][j][k]);
        }
        free(C6ab_dftD[i][j]);
      }
      free(C6ab_dftD[i]);
    }
    free(C6ab_dftD);
  }
  /* Ellner */

  /* Allocation_Arrays(4) */

  if (remake_headfile == 0)
  {

    if (alloc_first[32] == 0)
    {

      for (i = 0; i < SpeciesNum; i++)
      {
        free(GridX_EH0[i]);
      }
      free(GridX_EH0);

      for (i = 0; i < SpeciesNum; i++)
      {
        free(GridY_EH0[i]);
      }
      free(GridY_EH0);

      for (i = 0; i < SpeciesNum; i++)
      {
        free(GridZ_EH0[i]);
      }
      free(GridZ_EH0);

      for (i = 0; i < SpeciesNum; i++)
      {
        free(Arho_EH0[i]);
      }
      free(Arho_EH0);

      for (i = 0; i < SpeciesNum; i++)
      {
        free(Wt_EH0[i]);
      }
      free(Wt_EH0);

      if (Energy_Decomposition_flag == 1)
      {

        for (i = 0; i < SpeciesNum; i++)
        {
          for (k = 0; k < Max_TGN_EH0; k++)
          {
            for (l = 0; l < (Spe_MaxL_Basis[i] + 1); l++)
            {
              free(Arho_EH0_Orb[i][k][l]);
            }
            free(Arho_EH0_Orb[i][k]);
          }
          free(Arho_EH0_Orb[i]);
        }
        free(Arho_EH0_Orb);
      }
    }
  }

  for (i = 0; i < SpeciesNum; i++)
  {
    free(SpeName[i]);
  }
  free(SpeName);

  for (i = 0; i < SpeciesNum; i++)
  {
    free(SpeBasis[i]);
  }
  free(SpeBasis);

  for (i = 0; i < SpeciesNum; i++)
  {
    free(SpeBasisName[i]);
  }
  free(SpeBasisName);

  for (i = 0; i < SpeciesNum; i++)
  {
    free(SpeVPS[i]);
  }
  free(SpeVPS);

  free(Spe_AtomicMass);

  free(Spe_MaxL_Basis);

  for (i = 0; i < SpeciesNum; i++)
  {
    free(Spe_Num_Basis[i]);
  }
  free(Spe_Num_Basis);

  for (i = 0; i < SpeciesNum; i++)
  {
    free(Spe_Num_CBasis[i]);
  }
  free(Spe_Num_CBasis);

  free(Spe_OpenCore_flag);

  free(Spe_Spe2Ban);
  free(Species_Top);
  free(Species_End);
  free(F_Snd_Num);
  free(S_Snd_Num);
  free(F_Rcv_Num);
  free(S_Rcv_Num);
  free(F_Snd_Num_WK);
  free(F_Rcv_Num_WK);
  free(F_TopMAN);
  free(S_TopMAN);
  free(Snd_DS_NL_Size);
  free(Rcv_DS_NL_Size);
  free(Snd_HFS_Size);
  free(Rcv_HFS_Size);
  free(VPS_j_dependency);
  free(Num_Snd_Grid_A2B);
  free(Num_Rcv_Grid_A2B);
  free(Num_Snd_Grid_B2C);
  free(Num_Rcv_Grid_B2C);
  free(Num_Snd_Grid_B2D);
  free(Num_Rcv_Grid_B2D);
  free(Num_Snd_Grid_B_AB2CA);
  free(Num_Rcv_Grid_B_AB2CA);
  free(Num_Snd_Grid_B_CA2CB);
  free(Num_Rcv_Grid_B_CA2CB);
  /* added by mari 05.12.2014 */
  free(Num_Snd_Grid_B_AB2C);
  free(Num_Rcv_Grid_B_AB2C);

  if (alloc_first[26] == 0)
  {

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Snd_Grid_A2B[ID]);
    }
    free(Index_Snd_Grid_A2B);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Rcv_Grid_A2B[ID]);
    }
    free(Index_Rcv_Grid_A2B);
  }

  if (alloc_first[27] == 0)
  {
    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Snd_Grid_B2C[ID]);
    }
    free(Index_Snd_Grid_B2C);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Rcv_Grid_B2C[ID]);
    }
    free(Index_Rcv_Grid_B2C);

    free(ID_NN_B2C_S);
    free(ID_NN_B2C_R);
    free(GP_B2C_S);
    free(GP_B2C_R);
  }

  if (alloc_first[28] == 0)
  {

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Snd_Grid_B_AB2CA[ID]);
    }
    free(Index_Snd_Grid_B_AB2CA);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Rcv_Grid_B_AB2CA[ID]);
    }
    free(Index_Rcv_Grid_B_AB2CA);

    free(ID_NN_B_AB2CA_S);
    free(ID_NN_B_AB2CA_R);
    free(GP_B_AB2CA_S);
    free(GP_B_AB2CA_R);
  }

  /* added by mari 05.12.2014 */
  if (alloc_first[31] == 0)
  {

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Snd_Grid_B_AB2C[ID]);
    }
    free(Index_Snd_Grid_B_AB2C);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Rcv_Grid_B_AB2C[ID]);
    }
    free(Index_Rcv_Grid_B_AB2C);
  }

  if (alloc_first[29] == 0)
  {

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Snd_Grid_B_CA2CB[ID]);
    }
    free(Index_Snd_Grid_B_CA2CB);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Rcv_Grid_B_CA2CB[ID]);
    }
    free(Index_Rcv_Grid_B_CA2CB);

    free(ID_NN_B_CA2CB_S);
    free(ID_NN_B_CA2CB_R);
    free(GP_B_CA2CB_S);
    free(GP_B_CA2CB_R);
  }

  if (alloc_first[30] == 0)
  {
    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Snd_Grid_B2D[ID]);
    }
    free(Index_Snd_Grid_B2D);

    for (ID = 0; ID < numprocs; ID++)
    {
      free(Index_Rcv_Grid_B2D[ID]);
    }
    free(Index_Rcv_Grid_B2D);

    free(ID_NN_B2D_S);
    free(ID_NN_B2D_R);
    free(GP_B2D_S);
    free(GP_B2D_R);
  }

  for (i = 0; i < SpeciesNum; i++)
  {
    free(EH0_scaling[i]);
  }
  free(EH0_scaling);

  for (i = 0; i < SpeciesNum; i++)
  {
    free(SO_factor[i]);
  }
  free(SO_factor);

  /* Allocation_Arrays(1) */

  for (i = 0; i < (atomnum + 4); i++)
  {
    free(Gxyz[i]);
  }
  free(Gxyz);

  num = M_GDIIS_HISTORY + 1;

  for (i = 0; i < num; i++)
  {
    for (j = 0; j < (atomnum + 4); j++)
    {
      free(GxyzHistoryIn[i][j]);
    }
    free(GxyzHistoryIn[i]);
  }
  free(GxyzHistoryIn);

  for (i = 0; i < num; i++)
  {
    for (j = 0; j < (atomnum + 4); j++)
    {
      free(GxyzHistoryR[i][j]);
    }
    free(GxyzHistoryR[i]);
  }
  free(GxyzHistoryR);

  num = Extrapolated_Charge_History;

  for (i = 0; i < num; i++)
  {
    free(His_Gxyz[i]);
  }
  free(His_Gxyz);

  for (i = 0; i <= atomnum; i++)
  {
    free(atom_Fixed_XYZ[i]);
  }
  free(atom_Fixed_XYZ);

  for (i = 0; i < (atomnum + 4); i++)
  {
    free(Cell_Gxyz[i]);
  }
  free(Cell_Gxyz);

  if (Solver == 1 || Solver == 5 || Solver == 6 || Solver == 8 || Solver == 11)
  {
    free(orderN_FNAN);
    free(orderN_SNAN);
  }

  /* EF */

  if (MD_switch == 4)
  {
    for (i = 0; i < (3 * atomnum + 2); i++)
    {
      free(Hessian[i]);
    }
    free(Hessian);
  }

  /* BFGS */

  if (MD_switch == 5)
  {

    for (i = 0; i < (3 * atomnum + 2); i++)
    {
      free(InvHessian[i]);
    }
    free(InvHessian);
  }

  /* RF by hmweng */

  if (MD_switch == 6)
  {

    for (i = 0; i < (3 * atomnum + 2); i++)
    {
      free(Hessian[i]);
    }
    free(Hessian);
  }

  /* RFC */

  if (MD_switch == 18)
  {

    for (i = 0; i < (3 * atomnum + 11); i++)
    {
      free(Hessian[i]);
    }
    free(Hessian);
  }

  /* Constraint_NCS_switch==2 */
  if (Constraint_NCS_switch == 2)
  {
    free(InitMagneticMoment);
  }

  if (LNO_flag == 1)
  {
    free(LNO_Num);
  }

  /* for MD_VS4 */
  if (MD_switch == 14)
  {

    free(AtomGr);
    free(atnum_AtGr);
    free(Temp_AtGr);
  }

  /* spin non-collinear */
  if (SpinP_switch == 3)
  {
    free(Angle0_Spin);
    free(Angle1_Spin);
    free(InitAngle0_Spin);
    free(InitAngle1_Spin);
    free(Angle0_Orbital);
    free(Angle1_Orbital);
    free(OrbitalMoment);
    free(Constraint_SpinAngle);

    free(InitAngle0_Orbital);
    free(InitAngle1_Orbital);
    for (i = 0; i < (atomnum + 1); i++)
    {
      free(Orbital_Moment_XYZ[i]);
    }
    free(Orbital_Moment_XYZ);
    free(Constraint_OrbitalAngle);
  }

  /* Cluster2 */
  if (Solver == 9)
  {

    free(ON2_zp);
    free(ON2_Rp);
    free(ON2_method);

    free(ON2_zp_f);
    free(ON2_Rp_f);
    free(ON2_method_f);
  }

  /* EGAC */
  if (Solver == 10)
  {

    for (Gc_AN = 0; Gc_AN <= atomnum; Gc_AN++)
    {
      free(natn_onan[Gc_AN]);
      free(ncn_onan1[Gc_AN]);
      free(ncn_onan2[Gc_AN]);
      free(ncn_onan3[Gc_AN]);
    }
    free(natn_onan);
    free(ncn_onan1);
    free(ncn_onan2);
    free(ncn_onan3);

    for (i = 0; i < numprocs; i++)
    {
      free(Indx_Rcv_HS_EGAC[i]);
    }
    free(Indx_Rcv_HS_EGAC);

    for (i = 0; i < numprocs; i++)
    {
      free(Indx_Snd_HS_EGAC[i]);
    }
    free(Indx_Snd_HS_EGAC);

    for (i = 0; i < numprocs; i++)
    {
      free(Indx_Rcv_GA_EGAC[i]);
    }
    free(Indx_Rcv_GA_EGAC);

    for (i = 0; i < numprocs; i++)
    {
      free(Indx_Snd_GA_EGAC[i]);
    }
    free(Indx_Snd_GA_EGAC);

    for (i = 0; i < EGAC_Num; i++)
    {
      free(L2L_ONAN[i]);
    }
    free(L2L_ONAN);

    for (i = 0; i < numprocs; i++)
    {
      free(Indx_Rcv_DM_EGAC[i]);
    }
    free(Indx_Rcv_DM_EGAC);

    for (i = 0; i < numprocs; i++)
    {
      free(Indx_Snd_DM_EGAC[i]);
    }
    free(Indx_Snd_DM_EGAC);

    for (spin = 0; spin < (SpinP_switch + 1); spin++)
    {
      for (Mc_AN = 0; Mc_AN < Matomnum_EGAC; Mc_AN++)
      {

        Gc_AN = M2G_EGAC[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[Cwan];

        for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
        {

          Gh_AN = natn[Gc_AN][h_AN];
          Hwan = WhatSpecies[Gh_AN];
          tno2 = Spe_Total_CNO[Hwan];

          for (i = 0; i < tno1; i++)
          {
            free(H_EGAC[spin][Mc_AN][h_AN][i]);
          }
          free(H_EGAC[spin][Mc_AN][h_AN]);
        }
        free(H_EGAC[spin][Mc_AN]);
      }
      free(H_EGAC[spin]);
    }
    free(H_EGAC);

    for (Mc_AN = 0; Mc_AN < Matomnum_EGAC; Mc_AN++)
    {

      Gc_AN = M2G_EGAC[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {

        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno2 = Spe_Total_CNO[Hwan];

        for (i = 0; i < tno1; i++)
        {
          free(OLP_EGAC[Mc_AN][h_AN][i]);
        }
        free(OLP_EGAC[Mc_AN][h_AN]);
      }
      free(OLP_EGAC[Mc_AN]);
    }
    free(OLP_EGAC);

    int job_id, job_gid, N3[4];

    for (job_id = 0; job_id < EGAC_Num; job_id++)
    {

      job_gid = job_id + EGAC_Top[myid];
      GN2N_EGAC(job_gid, N3);
      Gc_AN = N3[1];

      for (i = 0; i < (FNAN[Gc_AN] + SNAN[Gc_AN] + ONAN[Gc_AN] + 1); i++)
      {
        free(RMI1_EGAC[job_id][i]);
      }
      free(RMI1_EGAC[job_id]);
    }
    free(RMI1_EGAC);

    for (job_id = 0; job_id < EGAC_Num; job_id++)
    {

      job_gid = job_id + EGAC_Top[myid];
      GN2N_EGAC(job_gid, N3);
      Gc_AN = N3[1];

      for (i = 0; i < (FNAN[Gc_AN] + SNAN[Gc_AN] + ONAN[Gc_AN] + 1); i++)
      {
        free(RMI2_EGAC[job_id][i]);
      }
      free(RMI2_EGAC[job_id]);
    }
    free(RMI2_EGAC);

    for (spin = 0; spin < (SpinP_switch + 1); spin++)
    {
      for (Mc_AN = 0; Mc_AN < Matomnum_DM_Snd_EGAC; Mc_AN++)
      {

        Gc_AN = M2G_DM_Snd_EGAC[Mc_AN];
        Cwan = WhatSpecies[Gc_AN];
        tno1 = Spe_Total_CNO[Cwan];

        for (h_AN = 0; h_AN < (FNAN[Gc_AN] + 1); h_AN++)
        {
          for (i = 0; i < tno1; i++)
          {
            free(DM_Snd_EGAC[spin][Mc_AN][h_AN][i]);
          }
          free(DM_Snd_EGAC[spin][Mc_AN][h_AN]);
        }
        free(DM_Snd_EGAC[spin][Mc_AN]);
      }
      free(DM_Snd_EGAC[spin]);
    }
    free(DM_Snd_EGAC);

    free(dim_GD_EGAC);
    free(dim_IA_EGAC);

    for (m = 0; m < DIIS_History_EGAC; m++)
    {
      for (job_id = 0; job_id < EGAC_Num; job_id++)
      {
        free(Sigma_EGAC[m][job_id]);
      }
      free(Sigma_EGAC[m]);
    }
    free(Sigma_EGAC);

    for (job_id = 0; job_id < (EGAC_Num + 1); job_id++)
    {
      free(fGD_EGAC[job_id]);
    }
    free(fGD_EGAC);

    for (job_id = 0; job_id < EGAC_Num; job_id++)
    { /* job_id: local job_id */

      job_gid = job_id + EGAC_Top[myid]; /* job_gid: global job_id */
      GN2N_EGAC(job_gid, N3);

      Gc_AN = N3[1];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN = 0; h_AN <= (FNAN[Gc_AN] + SNAN[Gc_AN]); h_AN++)
      {

        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno2 = Spe_Total_CNO[Hwan];

        for (i = 0; i < tno1; i++)
        {
          free(GD_EGAC[job_id][h_AN][i]);
        }
        free(GD_EGAC[job_id][h_AN]);
      }
      free(GD_EGAC[job_id]);
    }
    free(GD_EGAC);

    for (k = 0; k < Num_GA_EGAC; k++)
    { /* k is the first index of GA_EGAC */

      job_gid = M2G_JOB_EGAC[k];
      GN2N_EGAC(job_gid, N3);

      Gc_AN = N3[1];
      Cwan = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_CNO[Cwan];

      for (h_AN = 0; h_AN <= (FNAN[Gc_AN] + SNAN[Gc_AN]); h_AN++)
      {

        Gh_AN = natn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];
        tno2 = Spe_Total_CNO[Hwan];

        for (i = 0; i < tno1; i++)
        {
          free(GA_EGAC[k][h_AN][i]);
        }
        free(GA_EGAC[k][h_AN]);
      }
      free(GA_EGAC[k]);
    }
    free(GA_EGAC);

    free(EGAC_Top);
    free(EGAC_End);
    free(Num_Rcv_HS_EGAC);
    free(Num_Snd_HS_EGAC);
    free(Top_Index_HS_EGAC);
    free(Num_Rcv_GA_EGAC);
    free(Num_Snd_GA_EGAC);
    free(Top_Index_GA_EGAC);
    free(Snd_GA_EGAC_Size);
    free(Rcv_GA_EGAC_Size);
    free(Snd_OLP_EGAC_Size);
    free(Rcv_OLP_EGAC_Size);
    free(G2M_EGAC);
    free(M2G_DM_Snd_EGAC);
    free(G2M_DM_Snd_EGAC);
    free(Snd_DM_EGAC_Size);
    free(Rcv_DM_EGAC_Size);
    free(M2G_JOB_EGAC);

    free(EGAC_zp);
    free(EGAC_Rp);
    free(EGAC_method);
    free(EGAC_zp_f);
    free(EGAC_Rp_f);
    free(EGAC_method_f);
    free(M2G_EGAC);
  }

  /* Allocation_Arrays(5) */

  for (i = 0; i < (MO_Nkpoint + 1); i++)
  {
    free(MO_kpoint[i]);
  }
  free(MO_kpoint);

  /* Set_Periodic() in truncation.c */

  n = 2 * CpyCell + 4;
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      free(ratv[i][j]);
    }
    free(ratv[i]);
  }
  free(ratv);

  for (i = 0; i < (TCpyCell + 1); i++)
  {
    free(atv[i]);
  }
  free(atv);

  for (i = 0; i < (TCpyCell + 1); i++)
  {
    free(atv_ijk[i]);
  }
  free(atv_ijk);

  /* Allocate_Arrays(6) in SetPara_DFT.c */

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_PAO_XV[i]);
  }
  free(Spe_PAO_XV);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_PAO_RV[i]);
  }
  free(Spe_PAO_RV);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_Atomic_Den[i]);
  }
  free(Spe_Atomic_Den);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_Atomic_Den2[i]);
  }
  free(Spe_Atomic_Den2);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    for (j = 0; j <= List_YOUSO[25]; j++)
    {
      for (k = 0; k < List_YOUSO[24]; k++)
      {
        free(Spe_PAO_RWF[i][j][k]);
      }
      free(Spe_PAO_RWF[i][j]);
    }
    free(Spe_PAO_RWF[i]);
  }
  free(Spe_PAO_RWF);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    for (j = 0; j <= List_YOUSO[25]; j++)
    {
      for (k = 0; k < List_YOUSO[24]; k++)
      {
        free(Spe_RF_Bessel[i][j][k]);
      }
      free(Spe_RF_Bessel[i][j]);
    }
    free(Spe_RF_Bessel[i]);
  }
  free(Spe_RF_Bessel);

  /* Allocate_Arrays(7) in SetPara_DFT.c */

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_VPS_XV[i]);
  }
  free(Spe_VPS_XV);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_VPS_RV[i]);
  }
  free(Spe_VPS_RV);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_Vna[i]);
  }
  free(Spe_Vna);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_VH_Atom[i]);
  }
  free(Spe_VH_Atom);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_Atomic_PCC[i]);
  }
  free(Spe_Atomic_PCC);

  for (so = 0; so < (SO_switch + 1); so++)
  {
    for (i = 0; i < List_YOUSO[18]; i++)
    {
      for (j = 0; j < List_YOUSO[19]; j++)
      {
        free(Spe_VNL[so][i][j]);
      }
      free(Spe_VNL[so][i]);
    }
    free(Spe_VNL[so]);
  }
  free(Spe_VNL);

  for (so = 0; so < (SO_switch + 1); so++)
  {
    for (i = 0; i < List_YOUSO[18]; i++)
    {
      free(Spe_VNLE[so][i]);
    }
    free(Spe_VNLE[so]);
  }
  free(Spe_VNLE);

  for (i = 0; i < List_YOUSO[18]; i++)
  {
    free(Spe_VPS_List[i]);
  }
  free(Spe_VPS_List);

  for (so = 0; so < (SO_switch + 1); so++)
  {
    for (i = 0; i < List_YOUSO[18]; i++)
    {
      for (j = 0; j < (List_YOUSO[19] + 2); j++)
      {
        free(Spe_NLRF_Bessel[so][i][j]);
      }
      free(Spe_NLRF_Bessel[so][i]);
    }
    free(Spe_NLRF_Bessel[so]);
  }
  free(Spe_NLRF_Bessel);

  if (ProExpn_VNA == 1)
  {

    for (i = 0; i < List_YOUSO[18]; i++)
    {
      for (L = 0; L < (List_YOUSO[35] + 1); L++)
      {
        for (j = 0; j < List_YOUSO[34]; j++)
        {
          free(Projector_VNA[i][L][j]);
        }
        free(Projector_VNA[i][L]);
      }
      free(Projector_VNA[i]);
    }
    free(Projector_VNA);

    for (i = 0; i < List_YOUSO[18]; i++)
    {
      for (L = 0; L < (List_YOUSO[35] + 1); L++)
      {
        free(VNA_proj_ene[i][L]);
      }
      free(VNA_proj_ene[i]);
    }
    free(VNA_proj_ene);

    for (i = 0; i < List_YOUSO[18]; i++)
    {
      for (L = 0; L < (List_YOUSO[35] + 1); L++)
      {
        for (j = 0; j < List_YOUSO[34]; j++)
        {
          free(Spe_VNA_Bessel[i][L][j]);
        }
        free(Spe_VNA_Bessel[i][L]);
      }
      free(Spe_VNA_Bessel[i]);
    }
    free(Spe_VNA_Bessel);

    for (i = 0; i < List_YOUSO[18]; i++)
    {
      free(Spe_CrudeVNA_Bessel[i]);
    }
    free(Spe_CrudeVNA_Bessel);
  }

  if ((Solver == 2 || Solver == 3 || Solver == 7) && MO_fileout == 1)
  {

    for (i = 0; i < List_YOUSO[33]; i++)
    {
      for (j = 0; j < 2; j++)
      {
        for (k = 0; k < List_YOUSO[31]; k++)
        {
          for (l = 0; l < List_YOUSO[1]; l++)
          {
            free(HOMOs_Coef[i][j][k][l]);
          }
          free(HOMOs_Coef[i][j][k]);
        }
        free(HOMOs_Coef[i][j]);
      }
      free(HOMOs_Coef[i]);
    }
    free(HOMOs_Coef);

    for (i = 0; i < List_YOUSO[33]; i++)
    {
      for (j = 0; j < 2; j++)
      {
        for (k = 0; k < List_YOUSO[32]; k++)
        {
          for (l = 0; l < List_YOUSO[1]; l++)
          {
            free(LUMOs_Coef[i][j][k][l]);
          }
          free(LUMOs_Coef[i][j][k]);
        }
        free(LUMOs_Coef[i][j]);
      }
      free(LUMOs_Coef[i]);
    }
    free(LUMOs_Coef);

    free(Bulk_Num_HOMOs);
    free(Bulk_Num_LUMOs);
    for (i = 0; i < List_YOUSO[33]; i++)
    {
      free(Bulk_HOMO[i]);
    }
    free(Bulk_HOMO);
  }

  if (CntOrb_fileout == 1)
  {
    free(CntOrb_Atoms);
  }

  /* allocated in Input_std.c */

  if (empty_occupation_flag == 1)
  {
    free(empty_occupation_spin);
    free(empty_occupation_orbital);
  }

  /* allocated in Input_std.c */

  if (Band_Nkpath > 0)
  {

    free(Band_N_perpath);

    for (i = 0; i < (Band_Nkpath + 1); i++)
    {
      for (j = 0; j < 3; j++)
      {
        free(Band_kpath[i][j]);
      }
      free(Band_kpath[i]);
    }
    free(Band_kpath);

    for (i = 0; i < (Band_Nkpath + 1); i++)
    {
      for (j = 0; j < 3; j++)
      {
        free(Band_kname[i][j]);
      }
      free(Band_kname[i]);
    }
    free(Band_kname);
  }

  /* hmweng */
  /* Those arrays were allocated in Allocate_Array(9). */

  if (alloc_first[25] == 0)
  {

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_Select_Matrix[i]);
    }
    free(Wannier_Select_Matrix);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      for (j = 0; j < Wannier_Num_Pro[i]; j++)
      {
        free(Wannier_Projector_Hybridize_Matrix[i][j]);
      }
      free(Wannier_Projector_Hybridize_Matrix[i]);
    }
    free(Wannier_Projector_Hybridize_Matrix);
  }

  /* hmweng */
  /* Those arrays were allocated in Allocate_Array(8). */

  if (alloc_first[24] == 0)
  {

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_ProSpeName[i]);
    }
    free(Wannier_ProSpeName);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_ProName[i]);
    }
    free(Wannier_ProName);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_Pos[i]);
    }
    free(Wannier_Pos);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_X_Direction[i]);
    }
    free(Wannier_X_Direction);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_Z_Direction[i]);
    }
    free(Wannier_Z_Direction);

    free(Wannier_Num_Pro);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_NumL_Pro[i]);
    }
    free(Wannier_NumL_Pro);

    free(WannierPro2SpeciesNum);

    for (i = 0; i < 3; i++)
    {
      free(Wannier_Guide[i]);
    }
    free(Wannier_Guide);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      free(Wannier_Euler_Rotation_Angle[i]);
    }
    free(Wannier_Euler_Rotation_Angle);

    for (i = 0; i < Wannier_Num_Kinds_Projectors; i++)
    {
      for (j = 0; j < 4; j++)
      {
        for (l = 0; l < 2 * j + 1; l++)
        {
          free(Wannier_RotMat_for_Real_Func[i][j][l]);
        }
        free(Wannier_RotMat_for_Real_Func[i][j]);
      }
      free(Wannier_RotMat_for_Real_Func[i]);
    }
    free(Wannier_RotMat_for_Real_Func);

    free(Wannier_ProName2Num);
  }

  /* Allocation_Arrays(11) */ /* NBO by T.Ohwaki */

  if (NBO_switch != 0)
  {

    free(NBO_FCenter);
    free(Num_NHOs);

    for (i = 0; i < (NAO_Nkpoint + 1); i++)
    {
      free(NAO_kpoint[i]);
    }
    free(NAO_kpoint);

    free(rlmax_EC_NAO);
    free(rlmax_EC2_NAO);
    free(EKC_core_size_NAO);

    free(F_Snd_Num_NAO);
    free(S_Snd_Num_NAO);
    free(F_Rcv_Num_NAO);
    free(S_Rcv_Num_NAO);

    free(F_TopMAN_NAO);
    free(S_TopMAN_NAO);

    free(F_G2M_NAO);
    free(S_G2M_NAO);

    free(Snd_HFS_Size_NAO);
    free(Rcv_HFS_Size_NAO);

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (i = 0; i < Size_Total_Matrix; i++)
      {
        for (j = 0; j <= atomnum; j++)
        {
          free(NHOs_Coef[k][i][j]);
        }
        free(NHOs_Coef[k][i]);
      }
      free(NHOs_Coef[k]);
    }
    free(NHOs_Coef);

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (i = 0; i < Size_Total_Matrix; i++)
      {
        for (j = 0; j <= atomnum; j++)
        {
          free(NBOs_Coef_b[k][i][j]);
        }
        free(NBOs_Coef_b[k][i]);
      }
      free(NBOs_Coef_b[k]);
    }
    free(NBOs_Coef_b);

    for (k = 0; k <= SpinP_switch; k++)
    {
      for (i = 0; i < Size_Total_Matrix; i++)
      {
        for (j = 0; j <= atomnum; j++)
        {
          free(NBOs_Coef_a[k][i][j]);
        }
        free(NBOs_Coef_a[k][i]);
      }
      free(NBOs_Coef_a[k]);
    }
    free(NBOs_Coef_a);
  }

  /* Allocation_Arrays(3) */

  if (alloc_first[8] == 0)
  {

    for (ct_AN = 0; ct_AN <= atomnum; ct_AN++)
    {
      free(natn[ct_AN]);
    }
    free(natn);

    for (ct_AN = 0; ct_AN <= atomnum; ct_AN++)
    {
      free(ncn[ct_AN]);
    }
    free(ncn);

    for (ct_AN = 0; ct_AN <= atomnum; ct_AN++)
    {
      free(Dis[ct_AN]);
    }
    free(Dis);
  }

  /* Allocation_Arrays(2) */

  free(NormK);
  free(Spe_Atom_Cut1);
  free(Spe_Core_Charge);
  free(TGN_EH0);
  free(dv_EH0);
  free(Spe_Num_Mesh_VPS);
  free(Spe_Num_Mesh_PAO);
  free(Spe_Total_VPS_Pro);
  free(Spe_Num_RVPS);
  free(Spe_PAO_LMAX);
  free(Spe_PAO_Mul);
  free(Spe_WhatAtom);
  free(Spe_Total_NO);
  free(Spe_Total_CNO);
  free(FNAN);
  free(SNAN);
  free(ONAN);
  free(zp);
  free(Ep);
  free(Rp);

  if (Solver == 11)
  {
    free(FNAN_DCLNO);
    free(SNAN_DCLNO);
  }

  free(InitN_USpin);
  free(InitN_DSpin);
  free(WhatSpecies);
  free(GridN_Atom);
  free(RNUM);
  free(RNUM2);
  free(G2ID);
  free(F_G2M);
  free(S_G2M);
  free(time_per_atom);

  /* DCLNO or Krylov */

  if (atomnum <= numprocs && Solver == 11)
  {

    int numprocs0, numprocs1;

    MPI_Comm_size(mpi_comm_level1, &numprocs0);
    MPI_Comm_size(MPI_CommWD1_DCLNO[myworld1_DCLNO], &numprocs1);

    if (Num_Comm_World2_DCLNO <= numprocs1)
      MPI_Comm_free(&MPI_CommWD2_DCLNO[myworld2_DCLNO]);
    if (Num_Comm_World1_DCLNO <= numprocs0)
      MPI_Comm_free(&MPI_CommWD1_DCLNO[myworld1_DCLNO]);

    free(NPROCS_ID1_DCLNO);
    free(Comm_World1_DCLNO);
    free(NPROCS_WD1_DCLNO);
    free(Comm_World_StartID1_DCLNO);
    free(MPI_CommWD1_DCLNO);

    free(NPROCS_ID2_DCLNO);
    free(Comm_World2_DCLNO);
    free(NPROCS_WD2_DCLNO);
    free(Comm_World_StartID2_DCLNO);
    free(MPI_CommWD2_DCLNO);

    alloc_first[35] = 1;
  }

  else if (Solver == 11)
  {

    int numprocs1;

    // MPI_Comm_size( MPI_CommWD1_DCLNO[myworld1_DCLNO], &numprocs1);
    // if (Num_Comm_World2_DCLNO<=numprocs1) MPI_Comm_free(&MPI_CommWD2_DCLNO[myworld2_DCLNO]);

    MPI_Comm_free(&MPI_CommWD2_DCLNO[myworld2_DCLNO]);
    free(NPROCS_ID2_DCLNO);
    free(Comm_World2_DCLNO);
    free(NPROCS_WD2_DCLNO);
    free(Comm_World_StartID2_DCLNO);
    free(MPI_CommWD2_DCLNO);

    alloc_first[35] = 1;
  }
}

void array1()
{
}
