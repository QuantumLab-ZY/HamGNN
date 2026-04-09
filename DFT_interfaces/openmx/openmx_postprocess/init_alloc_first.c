/**********************************************************************
  init_alloc_first.c:

     init_alloc_first.c is a subroutine to initialize an array 
     alloc_first[];

  Log of init_alloc_first.c:

     24/May/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

void init_alloc_first()
{
  
  /***********************************************
   truncation.c

    GListTAtoms1
    GListTAtoms2
  ***********************************************/
  alloc_first[0] = 1;

  /***********************************************
   truncation.c

  ***********************************************/
  alloc_first[1] = 1;

  /***********************************************
   truncation.c

    GridListAtom
    CellListAtom
  ***********************************************/
  alloc_first[2] = 1;

  /***********************************************
   truncation.c

    Density_Grid
    ADensity_Grid
    PCCDensity_Grid
    Vxc_Grid
    VNA_Grid
    dVHart_Grid
    Vpot_Grid
    Orbs_Grid
    COrbs_Grid
    dOrbs_Grid
    dCOrbs_Grid
  ***********************************************/
  alloc_first[3] = 1;

  /***********************************************
   truncation.c

     H0
     CntH0
     OLP
     CntOLP
     H
     CntH
     DS_NL
     CntDS_NL
     DM
     ResidualDM
     EDM
     PDM
     CntCoes
  ***********************************************/
  alloc_first[4] = 1;

  /***********************************************
   truncation.c

     NumOLG
  ***********************************************/
  alloc_first[5] = 1;

  /***********************************************
   truncation.c

      RMI1
      RMI2
  ***********************************************/
  alloc_first[6] = 1;

  /***********************************************
   truncation.c

      ratv
      atv
      atv_ijk
  ***********************************************/
  alloc_first[7] = 1;

  /***********************************************
   Allocation_Arrays.c

      natn
      ncn
      Dis
  ***********************************************/
  alloc_first[8] = 1;

  /***********************************************
   Allocation_Arrays.c

   double *EV_S;
   double *IEV_S;
  ***********************************************/
  alloc_first[9] = 1;

  /***********************************************
   Set_Allocate_Atom2CPU.c

      M2G
  ***********************************************/
  alloc_first[10] = 1;

  /***********************************************
   in Set_Inf_SndRcv() of truncation.c

   Snd_MAN[numprocs][FS_Snd_Num[ID1]]
   Snd_GAN[numprocs][FS_Snd_Num[ID1]]
  ***********************************************/
  alloc_first[11] = 1;

  /***********************************************
   in Set_Inf_SndRcv() of truncation.c

   int Rcv_GAN[numprocs]
              [F_Rcv_Num[ID]+S_Rcv_Num[ID]]
  ***********************************************/
  alloc_first[12] = 1;

  /***********************************************
   Set_Allocate_Atom2CPU.c

      F_M2G
      S_M2G
  ***********************************************/
  alloc_first[13] = 1;

  /***********************************************
    none
  ***********************************************/
  alloc_first[14] = 1;

  /***********************************************
    none
  ***********************************************/
  alloc_first[15] = 1;

  /***********************************************
    none
  ***********************************************/
  alloc_first[16] = 1;

  /***********************************************
    none
  ***********************************************/
  alloc_first[17] = 1;

  /***********************************************
    none
  ***********************************************/
  alloc_first[18] = 1;

  /***********************************************
   GDC_Allocation() of Set_Allocate_Atom2CPU.c.

  ***********************************************/
  alloc_first[19] = 1;

  /***********************************************
   GDC_Allocation() of Set_Allocate_Atom2CPU.c.

  ***********************************************/
  alloc_first[20] = 1;

  /***********************************************
   Setup_EC of truncation.c.

   NAtom_EC 
   MAtom_EC
   LAtom_EC
  ***********************************************/
  alloc_first[21] = 1;

  /***********************************************
   Set_Inf_SndRcv of truncation.c.

   Pro_Snd_GAtom
   Pro_Snd_MAtom
   Pro_Snd_LAtom
  ***********************************************/
  alloc_first[22] = 1;

  /***********************************************
   Generating_MP_Special_Kpt.c.

   NE_T_k_op
   NE_KGrids1
   NE_KGrids2
   NE_KGrids3
  ***********************************************/
  alloc_first[23] = 1;

  /***********************************************
   Generating_MP_Special_Kpt.c.

   Wannier_ProSpeName
   Wannier_ProName
   Wannier_Pos
   Wannier_X_Direction
   Wannier_Z_Direction
  ***********************************************/
  alloc_first[24] = 1;

  /***********************************************
   Wannier_Select_Matrix
   Wannier_Projector_Hybridize_Matrix
  ***********************************************/
  alloc_first[25] = 1;

  /***********************************************
   Index_Snd_Grid_A2B
   Index_Rcv_Grid_A2B
  ***********************************************/
  alloc_first[26] = 1;

  /***********************************************
   Index_Snd_Grid_B2C
   Index_Rcv_Grid_B2C
  ***********************************************/
  alloc_first[27] = 1;

  /***********************************************
   Index_Snd_Grid_B_AB2CA
   Index_Rcv_Grid_B_AB2CA
  ***********************************************/
  alloc_first[28] = 1;

  /***********************************************
   Index_Snd_Grid_B_CA2CB
   Index_Rcv_Grid_B_CA2CB
  ***********************************************/
  alloc_first[29] = 1;

  /***********************************************
   Index_Snd_Grid_B2D
   Index_Rcv_Grid_B2D
  ***********************************************/
  alloc_first[30] = 1;

  /***********************************************
   Index_Snd_Grid_B_AB2C
   Index_Rcv_Grid_B_AB2C
  ***********************************************/
  /* added by mari 05.12.2014 */
  alloc_first[31] = 1;

  /***********************************************
   Allocate_Arrays(4);
   GridX_EH0
   GridY_EH0
   GridZ_EH0
   Arho_EH0
   Wt_EH0  
  ***********************************************/
  alloc_first[32] = 1;

  /***********************************************
   truncation.c;
   natn_onan
   ncn_onan1
   ncn_onan2
   ncn_onan3
  ***********************************************/
  alloc_first[33] = 1;

  /***********************************************
   truncation.c;
   Indx_Rcv_HS_EGC
   Indx_Snd_HS_EGC
   M2G_EGAC
  ***********************************************/
  alloc_first[34] = 1;

  /***********************************************
   truncation.c;
   NPROCS_ID1_DCLNO;
   Comm_World1_DCLNO;
   NPROCS_WD1_DCLNO;
   MPI_CommWD1_DCLNO;

   NPROCS_ID2_DCLNO;
   Comm_World2_DCLNO;
   NPROCS_WD2_DCLNO;
   MPI_CommWD2_DCLNO;
  ***********************************************/
  alloc_first[35] = 1;

}
