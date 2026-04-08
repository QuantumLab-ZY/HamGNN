/**********************************************************************
  DFT.c:

     DFT.c is a subroutine to perform self-consistent calculations
     within LDA or GGA.

  Log of DFT.c:

     22/Nov/2001  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
#include "tran_prototypes.h"

/* variables for cluster and band calculations  */
double *Ss_Re, *Cs_Re, *Hs_Re;
double *rHs11_Re, *rHs12_Re, *rHs22_Re, *iHs11_Re, *iHs12_Re, *iHs22_Re;
double *CDM1, *EDM1, *PDM1, *Work1;
dcomplex *EVec1_NonCol;
double **EVec1_Re;
double *H1, *S1;
double ***EIGEN_Band;
int *My_NZeros, *SP_NZeros, *SP_Atoms, *is2, *ie2, *MP, *order_GA;
int n, n2, size_H1, myworld1, myworld2, T_knum;
MPI_Comm *MPI_CommWD1, *MPI_CommWD2;
int *Comm_World_StartID1, *Comm_World_StartID2;
int Num_Comm_World1, Num_Comm_World2;
int *NPROCS_ID1, *Comm_World1, *NPROCS_WD1;
int *NPROCS_ID2, *Comm_World2, *NPROCS_WD2;
int ***k_op, *T_k_op, **T_k_ID;
dcomplex *rHs11_Cx, *rHs22_Cx, *rHs12_Cx;
dcomplex *iHs11_Cx, *iHs22_Cx, *iHs12_Cx;
dcomplex *Hs2_Cx, *Ss2_Cx, *Cs2_Cx;
dcomplex *Hs_Cx, *Ss_Cx, *Cs_Cx;
dcomplex **EVec1_Cx;
double *T_KGrids1, *T_KGrids2, *T_KGrids3;
double **ko_col, *ko_noncol, *ko, *koS;

/* Calc_optical */
dcomplex **H_Band_Col, **S_Band;

/* grid data */
double *ReVk, *ImVk, *ReRhoAtomk, *ImRhoAtomk;
double ***ReRhok, ***ImRhok, **Residual_ReRhok, **Residual_ImRhok;

/* for orbital optimization */
double ****His_CntCoes, ****His_D_CntCoes;
double ****His_CntCoes_Species, ****His_D_CntCoes_Species;
double **OrbOpt_Hessian, *His_OrbOpt_Etot;

/* for Cluster_DFT_LNO, Cluster_ReCoes is not allocated. */
double ***Cluster_ReCoes;

/* for Band_DFT_NonCol_GB (added by T. B. Prayitno and supervised by Prof. F. Ishii)*/
double *koSU, *koSL;
dcomplex **SU_Band, **SL_Band;

static double Calc_optical_conductivities_dielectric_functions(
    int myid0,
    double **ko_col,
    double Eele0[2],
    double Eele1[2],
    int *MP,
    int *order_GA,
    double *ko,
    double *koS,
    double ***EIGEN_Band,
    double *S1,
    double *H1,
    double *CDM1,
    dcomplex **H_Band_Col,
    dcomplex **S_Band,
    double *EDM1,
    int myworld1,
    int myworld2,
    int *NPROCS_ID1,
    int *Comm_World1,
    int *NPROCS_WD1,
    int *Comm_World_StartID1,
    MPI_Comm *MPI_CommWD1,
    int *NPROCS_ID2,
    int *NPROCS_WD2,
    int *Comm_World2,
    int *Comm_World_StartID2,
    MPI_Comm *MPI_CommWD2,
    int *is2,
    int *ie2,
    int size_H1,
    int *SP_NZeros,
    int *SP_Atoms,
    double **EVec1_Re,
    double *Work1,
    double *Ss_Re,
    double *Cs_Re,
    double *Hs_Re,
    dcomplex *Ss_Cx,
    dcomplex *Cs_Cx,
    dcomplex *Hs_Cx);

static void Output_Energies_Forces(FILE *fp);
static double dUele;
static void Read_SCF_keywords();

void Allocate_Free_Cluster_Col(int todo_flag);
void Allocate_Free_Cluster_NonCol(int todo_flag);
void Allocate_Free_Band_Col(int todo_flag);
void Allocate_Free_Band_NonCol(int todo_flag);
void Allocate_Free_NEGF_NonCol(int todo_flag);
void Allocate_Free_GridData(int todo_flag);
void Allocate_Free_OrbOpt(int todo_flag);

double DFT(int MD_iter, int Cnt_Now)
{
  static int firsttime = 1;
  double ECE[20], Eele0[2], Eele1[2];
  double pUele, ChemP_e0[2];
  double Norm1, Norm2, Norm3, Norm4, Norm5;
  double TotalE;
  double S_coordinate[3];
  double tmp, tmp0;
  int Cnt_kind, Calc_CntOrbital_ON, spin, spinmax, m;
  int SCF_iter, SCF_iter_shift, SCF_MAX;
  int i, j, k, fft_charge_flag, M1N;
  int orbitalOpt_iter, LSCF_iter, OrbOpt_end;
  int wanA, ETemp_controller;
  double time0, time1, time2, time3, time4;
  double time5, time6, time7, time8, time9;
  double time10, time11, time12, time13;
  double time14, time15, etime;
  double x, y, z;
  int po3, po, TRAN_Poisson_flag2;
  int SucceedReadingDMfile, My_SucceedReadingDMfile;
  char file_DFTSCF[YOUSO10] = ".DFTSCF";
  char file_OrbOpt[YOUSO10] = ".OrbOpt";
  char operate[200];
  char *s_vec[20];
  double TStime, TEtime;
  FILE *fp_DFTSCF;
  FILE *fp_OrbOpt;

  int numprocs0, myid0;
  int numprocs1, myid1;

  dtime(&TStime);

  /* MPI */

  MPI_Comm_size(mpi_comm_level1, &numprocs0);
  MPI_Comm_rank(mpi_comm_level1, &myid0);

  /************************************************************
                       allocation of arrays
  *************************************************************/

  if (Solver == 2 && SpinP_switch <= 1)
    Allocate_Free_Cluster_Col(1);
  else if (Solver == 2 && SpinP_switch == 3)
    Allocate_Free_Cluster_NonCol(1);
  else if (Solver == 3 && SpinP_switch <= 1)
    Allocate_Free_Band_Col(1);
  else if (Solver == 3 && SpinP_switch == 3)
    Allocate_Free_Band_NonCol(1);
  else if (Solver == 4 && SpinP_switch == 3)
    Allocate_Free_NEGF_NonCol(1);

  Allocate_Free_GridData(1);
  if (Cnt_switch == 1)
    Allocate_Free_OrbOpt(1);

  /****************************************************
    print some informations to the standard output
    and initialize times
  ****************************************************/

  if (Cnt_switch == 1 && Cnt_Now == 1 && MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {
    printf("\n*******************************************************\n");
    fflush(stdout);
    printf("             Orbital optimization                      \n");
    fflush(stdout);
    printf("             SCF calculation at MD =%2d                \n", MD_iter);
    fflush(stdout);
    printf("*******************************************************\n\n");
    fflush(stdout);
  }
  else if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {
    printf("\n*******************************************************\n");
    fflush(stdout);
    printf("             SCF calculation at MD =%2d                \n", MD_iter);
    fflush(stdout);
    printf("*******************************************************\n\n");
    fflush(stdout);
  }

  fnjoint(filepath, filename, file_DFTSCF);
  fnjoint(filepath, filename, file_OrbOpt);

  /* initialize */

  time3 = 0.0;
  time4 = 0.0;
  time5 = 0.0;
  time6 = 0.0;
  time7 = 0.0;
  time8 = 0.0;
  time10 = 0.0;
  time11 = 0.0;
  time12 = 0.0;
  time13 = 0.0;
  time14 = 0.0;
  time15 = 0.0;

  for (i = 0; i < 20; i++)
    ECE[i] = 0.0;

  fft_charge_flag = 0;
  TRAN_Poisson_flag2 = 0;

  /****************************************************
         Calculations of overlap and Hamiltonian
         matrices for DFTSCF
  ****************************************************/

  if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {
    printf("<MD=%2d>  Calculation of the overlap matrix\n", MD_iter);
    fflush(stdout);
  }

  time1 = Set_OLP_Kin(OLP, H0);

  if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {
    printf("<MD=%2d>  Calculation of the nonlocal matrix\n", MD_iter);
    fflush(stdout);
  }

  time2 = Set_Nonlocal(HNL, DS_NL);

  if (core_hole_state_flag == 1)
  {
    if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
    {
      printf("<MD=%2d>  Calculation of the core hole matrix\n", MD_iter);
      fflush(stdout);
    }
    time2 += Set_CoreHoleMatrix(HCH);
  }

  if (ProExpn_VNA == 1)
  {
    if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
    {
      printf("<MD=%2d>  Calculation of the VNA projector matrix\n", MD_iter);
      fflush(stdout);
    }
    time12 = Set_ProExpn_VNA(HVNA, HVNA2, DS_VNA);
  }

  return 0; // added by Yang Zhong
  
  /* SCF loop */

  if (Cnt_switch == 1 && Cnt_Now == 1)
  {
    SCF_MAX = orbitalOpt_SCF * (orbitalOpt_MD + 1) - 1;
    Oopt_NormD[1] = 1.0e+5;
  }
  else
  {
    SCF_MAX = DFTSCF_loop;
  }

  orbitalOpt_iter = 1;
  OrbOpt_end = 0;
  SCF_iter = 0;
  LSCF_iter = 0;
  ETemp_controller = 0;
  SCF_iter_shift = 0;
  NormRD[0] = 100.0;

  SCF_RENZOKU = -1;
  po = 0;
  pUele = 100.0;
  Norm1 = 100.0;
  Norm2 = 100.0;
  Norm3 = 100.0;
  Norm4 = 100.0;

  /*****************************************************
              read contraction coefficients
  *****************************************************/

  if (Cnt_switch == 1 && MD_iter != 1)
    File_CntCoes("read");

  /****************************************************
            Start self consistent calculation
  ****************************************************/

  do
  {

    SCF_iter++;
    LSCF_iter++;

    /*****************************************************
                         print stdout
    *****************************************************/

    if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
    {
      if (Cnt_switch == 1 && Cnt_Now == 1)
      {
        printf("\n***************** Orbital optimization **************\n");
        fflush(stdout);
        printf("    MD=%2d  orbitalOpt_iter=%2d  G-SCF=%2d  L-SCF=%2d  \n",
               MD_iter, orbitalOpt_iter, SCF_iter, LSCF_iter);
        fflush(stdout);
      }
      else
      {
        printf("\n******************* MD=%2d  SCF=%2d *******************\n",
               MD_iter, SCF_iter);
        fflush(stdout);
      }
      fflush(stdout);
    }

    /*****************************************************
     setting of densities, potentials and KS Hamiltonian
    *****************************************************/
    /****************************************************
     if (SCF==1)
    ****************************************************/

    if (SCF_iter == 1)
    {

      if (MD_iter != 1)
        Mixing_weight = Max_Mixing_weight;

      if (Cnt_switch == 0 || (Cnt_switch == 1 && Cnt_Now == 1))
        Cnt_kind = 0;
      else if (Cnt_switch == 1)
        Cnt_kind = 1;
      else
        Cnt_kind = 0;

      time9 = Set_Aden_Grid();

      if (Solver == 4)
      {

        /*****************************************************
                          set grid for NEGF
  *****************************************************/

        TRAN_Set_Electrode_Grid(mpi_comm_level1,
                                &TRAN_Poisson_flag2,
                                Grid_Origin, tv, Left_tv, Right_tv, gtv,
                                Ngrid1, Ngrid2, Ngrid3);

        /* revised by Y. Xiao for Noncollinear NEGF calculations */
        if (SpinP_switch < 2)
        {
          TRAN_Allocate_Lead_Region(mpi_comm_level1);
          TRAN_Allocate_Cregion(mpi_comm_level1, SpinP_switch, atomnum, WhatSpecies, Spe_Total_CNO);
        }
        else
        {
          TRAN_Allocate_Lead_Region_NC(mpi_comm_level1);
          TRAN_Allocate_Cregion_NC(mpi_comm_level1, SpinP_switch, atomnum, WhatSpecies, Spe_Total_CNO);
        } /* until here by Y. Xiao for Noncollinear NEGF calculations*/

        /*****************************************************
                         add density from Leads
  *****************************************************/

        TRAN_Add_Density_Lead(mpi_comm_level1,
                              SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                              My_NumGridB_AB, Density_Grid_B);

        TRAN_Add_ADensity_Lead(mpi_comm_level1,
                               SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                               My_NumGridB_AB, ADensity_Grid_B);

      } /* end of if (Solver==4) */

      time10 += Set_Orbitals_Grid(Cnt_kind);

      /* YTL-start */
      if (CDDF_on == 1)
      {
        time10 += Set_dOrbitals_Grid(Cnt_kind);
        Global_Cnt_kind = Cnt_kind;
      }
      /* YTL-end */

      /******************************************************
                        read restart files
      ******************************************************/

      SucceedReadingDMfile = 0;
      if (Scf_RestartFromFile == 1 && ((Cnt_switch == 1 && Cnt_Now != 1) || Cnt_switch == 0))
      {

        My_SucceedReadingDMfile = RestartFileDFT("read", MD_iter, &Uele, H, CntH, &etime);

        /* Calling RestartFileDFT overwrites HNL.
        Thus, we call Set_Nonlocal again, since HNL depends on basis functions. */
        if (core_hole_state_flag == 1)
          time2 = Set_Nonlocal(HNL, DS_NL);

        time13 += etime;
        MPI_Barrier(mpi_comm_level1);
        MPI_Allreduce(&My_SucceedReadingDMfile, &SucceedReadingDMfile,
                      1, MPI_INT, MPI_PROD, mpi_comm_level1);

        if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
        {
          if (SucceedReadingDMfile == 1)
          {
            printf("<Restart>  Found restart files\n");
            fflush(stdout);
          }
          else
          {
            printf("<Restart>  Could not find restart files\n");
            fflush(stdout);
          }
        }

        /***********************************************************************
         If reading the restart files is terminated, the densities on grid are
         partially overwritten. So, Set_Aden_Grid() is called once more.
        ***********************************************************************/

        if (SucceedReadingDMfile == 0)
        {

          time9 += Set_Aden_Grid();

          if (Solver == 4)
          {
            TRAN_Add_ADensity_Lead(mpi_comm_level1,
                                   SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                                   My_NumGridB_AB, ADensity_Grid_B);
          }
        }
      }

      /*****************************************************
       FFT of the initial density for k-space charge mixing
      *****************************************************/

      if ((Mixing_switch == 3 || Mixing_switch == 4) && SCF_iter == 1)
      {

        /* non-spin polarization */
        if (SpinP_switch == 0)
        {
          if (Solver != 4 || TRAN_Poisson_flag2 == 2)
          {
            time15 += FFT_Density(1, ReRhok[1][0], ImRhok[1][0]);
          }
          else
          {
            time15 += FFT2D_Density(1, ReRhok[1][0], ImRhok[1][0]);
          }
        }

        /* collinear spin polarization */
        else if (SpinP_switch == 1)
        {
          if (Solver != 4 || TRAN_Poisson_flag2 == 2)
          {
            time15 += FFT_Density(1, ReRhok[1][0], ImRhok[1][0]);
            time15 += FFT_Density(2, ReRhok[1][1], ImRhok[1][1]);
          }
          else
          {
            time15 += FFT2D_Density(1, ReRhok[1][0], ImRhok[1][0]);
            time15 += FFT2D_Density(2, ReRhok[1][1], ImRhok[1][1]);
          }
        }

        /* non-collinear spin polarization */
        else if (SpinP_switch == 3)
        {
          if (Solver != 4 || TRAN_Poisson_flag2 == 2)
          {
            time15 += FFT_Density(1, ReRhok[1][0], ImRhok[1][0]);
            time15 += FFT_Density(2, ReRhok[1][1], ImRhok[1][1]);
            time15 += FFT_Density(4, ReRhok[1][2], ImRhok[1][2]);
          }
          else
          {
            time15 += FFT2D_Density(1, ReRhok[1][0], ImRhok[1][0]);
            time15 += FFT2D_Density(2, ReRhok[1][1], ImRhok[1][1]);
            time15 += FFT2D_Density(4, ReRhok[1][2], ImRhok[1][2]);
          }
        }
      }

      if (SpinP_switch == 3)
        diagonalize_nc_density(Density_Grid_B);

      /* In case the restart file is found */

      if (SucceedReadingDMfile && Cnt_switch == 0)
      {

        if (core_hole_state_flag && Solver != 4 && ESM_switch == 0)
          time4 += Poisson(2, ReVk, ImVk);
        else if (Solver != 4 && ESM_switch == 0)
          time4 += Poisson(1, ReVk, ImVk);
        else if (Solver != 4 && ESM_switch != 0)
          time4 += Poisson_ESM(1, ReVk, ImVk); /* added by Ohwaki */
        else
          time4 += TRAN_Poisson(ReVk, ImVk);

        /*
          Although the matrix of Hamiltonian is read from files, it would be better to
          reconstruct the matrix using charge density read from files in order to avoid
          abrupt change of NormRD at the second SCF step.
          Also, if (Correct_Position_flag), then the reconstruction has to be done
          regardless of the above reason.
        */

        if (Correct_Position_flag || Use_of_Collinear_Restart == 1 ||
            (SO_switch == 0 && Hub_U_switch == 0 && Constraint_NCS_switch == 0 && Zeeman_NCS_switch == 0 && Zeeman_NCO_switch == 0) /* non-spin-orbit coupling and non-LDA+U */
        )
        {

          time3 += Set_Hamiltonian("nostdout",
                                   MD_iter,
                                   LSCF_iter - SCF_iter_shift,
                                   SCF_iter - SCF_iter_shift,
                                   TRAN_Poisson_flag2,
                                   SucceedReadingDMfile, Cnt_kind, H0, HNL, DM[0], H);
        }
      }

      /* failure of reading restart files */
      else
      {

        if (Solver == 4)
          time4 += TRAN_Poisson(ReVk, ImVk);

        time3 += Set_Hamiltonian("nostdout",
                                 MD_iter,
                                 LSCF_iter - SCF_iter_shift,
                                 SCF_iter - SCF_iter_shift,
                                 TRAN_Poisson_flag2,
                                 SucceedReadingDMfile, Cnt_kind, H0, HNL, DM[0], H);
      }

      if (Cnt_switch == 1 && Cnt_Now == 1)
      {
        if (MD_iter == 1)
          Initial_CntCoes2(H, OLP);

        Contract_Hamiltonian(H, CntH, OLP, CntOLP);
        if (SO_switch == 1)
          Contract_iHNL(iHNL, iCntHNL);
      }

      /* switch a restart flag on for proceeding MD steps */
      if (MD_switch != 0 && Scf_RestartFromFile != -1)
      {
        Scf_RestartFromFile = 1;
        Use_of_Collinear_Restart = 0;
      }

    } /* end of if (SCF_iter==1) */

    /****************************************************
     if (SCF!=1)
    ****************************************************/

    else
    {

      if (Cnt_switch == 0 || (Cnt_switch == 1 && Cnt_Now == 1))
        Cnt_kind = 0;
      else if (Cnt_switch == 1)
        Cnt_kind = 1;
      else
        Cnt_kind = 0;

      if (Solver != 4 && ESM_switch == 0)
        time4 += Poisson(fft_charge_flag, ReVk, ImVk);
      else if (Solver != 4 && ESM_switch != 0)
        time4 += Poisson_ESM(fft_charge_flag, ReVk, ImVk); /* added by Ohwaki */
      else
        time4 += TRAN_Poisson(ReVk, ImVk);

      /* construct matrix elements for LDA+U or Zeeman term, added by MJ */
      Eff_Hub_Pot(SCF_iter, OLP[0]);

      time3 += Set_Hamiltonian("stdout",
                               MD_iter,
                               LSCF_iter - SCF_iter_shift,
                               SCF_iter - SCF_iter_shift,
                               TRAN_Poisson_flag2,
                               SucceedReadingDMfile, Cnt_kind, H0, HNL, DM[0], H);

      if (Cnt_switch == 1 && Cnt_Now == 1)
      {
        Contract_Hamiltonian(H, CntH, OLP, CntOLP);
        if (SO_switch == 1)
          Contract_iHNL(iHNL, iCntHNL);
      }
    }

    /****************************************************
     In case of RMM-DIISH (Mixing_switch==5), the mixing
     of Hamiltonian matrix elements is performed before
     the eigenvalue problem.

     Even for (Mixing_switch==1 || Mixing_switch==6),
     during (LSCF_iter-SCF_iter_shift)<=(Pulay_SCF/2)
     Mixing_H is used instead of Mixing_DM.
     In Mixing_DM the function call is imediately returned
     during (LSCF_iter-SCF_iter_shift)<=(Pulay_SCF/2).
    ****************************************************/

    if (Mixing_switch == 5 || (Cnt_switch != 1 && SucceedReadingDMfile != 1 && (Mixing_switch == 1 || Mixing_switch == 6) && (LSCF_iter - SCF_iter_shift) <= (Pulay_SCF / 2)) && Solver != 4)
    {

      time6 += Mixing_H(MD_iter, LSCF_iter - SCF_iter_shift, SCF_iter - SCF_iter_shift);
    }

    /****************************************************
                Solve the eigenvalue problem
    ****************************************************/

    s_vec[0] = "Recursion";
    s_vec[1] = "Cluster";
    s_vec[2] = "Band";
    s_vec[3] = "NEGF";
    s_vec[4] = "DC";
    s_vec[5] = "GDC";
    s_vec[6] = "Cluster-DIIS";
    s_vec[7] = "Krylov";
    s_vec[8] = "Cluster2";
    s_vec[9] = "EGAC";
    s_vec[10] = "DC-LNO";
    s_vec[11] = "Cluster-LNO";

    if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
    {
      printf("<%s>  Solving the eigenvalue problem...\n", s_vec[Solver - 1]);
      fflush(stdout);
    }

    if (Cnt_switch == 0)
    {

      switch (Solver)
      {

      case 1:
        /* not supported */
        break;

      case 2:

        if (SpinP_switch <= 1)
        {

          time5 += Cluster_DFT_Col("scf", LSCF_iter, SpinP_switch,
                                   ko_col, H, OLP[0], DM[0], EDM,
                                   Eele0, Eele1,
                                   myworld1, NPROCS_ID1, Comm_World1, NPROCS_WD1,
                                   Comm_World_StartID1, MPI_CommWD1, MP, is2, ie2,
                                   Ss_Re, Cs_Re, Hs_Re,
                                   CDM1, EDM1, PDM1, size_H1, SP_NZeros, SP_Atoms, EVec1_Re, Work1);
        }

        else if (SpinP_switch == 3)
        {

          time5 += Cluster_DFT_NonCol("scf", LSCF_iter, SpinP_switch,
                                      ko_noncol, H, iHNL, OLP[0], DM[0], EDM,
                                      Eele0, Eele1,
                                      MP, is2, ie2,
                                      Ss_Re, Cs_Re,
                                      rHs11_Re, rHs12_Re, rHs22_Re, iHs11_Re, iHs12_Re, iHs22_Re,
                                      Ss2_Cx, Hs2_Cx, Cs2_Cx,
                                      CDM1, size_H1, EVec1_NonCol, Work1);
        }

        break;

      case 3:

        if (SpinP_switch <= 1)
        {

          time5 += Band_DFT_Col(LSCF_iter,
                                Kspace_grid1, Kspace_grid2, Kspace_grid3,
                                SpinP_switch, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1,
                                MP, order_GA, ko, koS, EIGEN_Band,
                                H1, S1,
                                CDM1, EDM1,
                                EVec1_Cx,
                                Ss_Cx,
                                Cs_Cx,
                                Hs_Cx,
                                k_op, T_k_op, T_k_ID,
                                T_KGrids1, T_KGrids2, T_KGrids3,
                                myworld1,
                                NPROCS_ID1,
                                Comm_World1,
                                NPROCS_WD1,
                                Comm_World_StartID1,
                                MPI_CommWD1,
                                myworld2,
                                NPROCS_ID2,
                                NPROCS_WD2,
                                Comm_World2,
                                Comm_World_StartID2,
                                MPI_CommWD2);
        }
        else
        {

          if (GB_switch == 0)
          {

            time5 += Band_DFT_NonCol(LSCF_iter,
                                     Kspace_grid1, Kspace_grid2, Kspace_grid3,
                                     SpinP_switch, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1,
                                     MP, order_GA, ko_noncol, koS, EIGEN_Band,
                                     H1, S1,
                                     rHs11_Cx, rHs22_Cx, rHs12_Cx, iHs11_Cx, iHs22_Cx, iHs12_Cx,
                                     EVec1_Cx,
                                     Ss_Cx, Cs_Cx, Hs_Cx,
                                     Ss2_Cx, Cs2_Cx, Hs2_Cx,
                                     k_op, T_k_op, T_k_ID,
                                     T_KGrids1, T_KGrids2, T_KGrids3,
                                     myworld1,
                                     NPROCS_ID1,
                                     Comm_World1,
                                     NPROCS_WD1,
                                     Comm_World_StartID1,
                                     MPI_CommWD1,
                                     myworld2,
                                     NPROCS_ID2,
                                     NPROCS_WD2,
                                     Comm_World2,
                                     Comm_World_StartID2,
                                     MPI_CommWD2);
          }
          else if (GB_switch == 1)
          {
            /* added by T. B. Prayitno and supervised by Prof. F. Ishii*/
            time5 += Band_DFT_NonCol_GB(LSCF_iter,
                                        koSU, koSL, SU_Band, SL_Band,
                                        Kspace_grid1, Kspace_grid2, Kspace_grid3,
                                        SpinP_switch, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
          }
        }

        break;

      case 4:
        /* revised by Y. Xiao for Noncollinear NEGF calculations*/
        if (SpinP_switch < 2)
        {
          time5 += TRAN_DFT(mpi_comm_level1, SucceedReadingDMfile,
                            level_stdout, LSCF_iter, SpinP_switch, H, iHNL, OLP[0],
                            atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn,
                            M2G, G2ID, F_G2M, atv_ijk, List_YOUSO,
                            DM[0], EDM, TRAN_DecMulP, Eele0, Eele1, ChemP_e0);
        }
        else
        {
          time5 += TRAN_DFT_NC(mpi_comm_level1, SucceedReadingDMfile,
                               level_stdout, LSCF_iter, SpinP_switch, H, iHNL, OLP[0],
                               atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn,
                               M2G, G2ID, F_G2M, atv_ijk, List_YOUSO, koS, S_Band,
                               DM[0], iDM[0], EDM, TRAN_DecMulP, Eele0, Eele1, ChemP_e0);
        } /* until here by Y. Xiao for Noncollinear NEGF calculations*/

        break;

      case 5:
        time5 += Divide_Conquer("scf", LSCF_iter, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 6:
        /* not supported */
        break;

      case 7:
        break;

      case 8:
        time5 += Krylov("scf", LSCF_iter, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 9:
        time5 += Cluster_DFT_ON2("scf", LSCF_iter, SpinP_switch,
                                 H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 10:
        time5 += EGAC_DFT("scf", LSCF_iter, SpinP_switch, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 11:
        time5 += Divide_Conquer_LNO("scf", MD_iter, LSCF_iter, SucceedReadingDMfile, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 12:
        time5 += Cluster_DFT_LNO("scf", LSCF_iter, SpinP_switch,
                                 Cluster_ReCoes, ko_col, H, iHNL, OLP[0], DM[0], EDM,
                                 NULL, NULL, NULL, Eele0, Eele1);
        break;
      }
    }

    else
    {

      switch (Solver)
      {

      case 1:
        /* not supported */
        break;

      case 2:

        if (SpinP_switch <= 1)
        {

          time5 += Cluster_DFT_Col("scf", LSCF_iter, SpinP_switch,
                                   ko_col, CntH, CntOLP[0], DM[0], EDM,
                                   Eele0, Eele1,
                                   myworld1, NPROCS_ID1, Comm_World1, NPROCS_WD1,
                                   Comm_World_StartID1, MPI_CommWD1, MP, is2, ie2,
                                   Ss_Re, Cs_Re, Hs_Re,
                                   CDM1, EDM1, PDM1, size_H1, SP_NZeros, SP_Atoms, EVec1_Re, Work1);
        }

        else if (SpinP_switch == 3)
        {
          /* not supported */
        }

        break;

      case 3:

        if (SpinP_switch <= 1)
        {

          time5 += Band_DFT_Col(LSCF_iter,
                                Kspace_grid1, Kspace_grid2, Kspace_grid3,
                                SpinP_switch, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1,
                                MP, order_GA, ko, koS, EIGEN_Band,
                                H1, S1,
                                CDM1, EDM1,
                                EVec1_Cx,
                                Ss_Cx,
                                Cs_Cx,
                                Hs_Cx,
                                k_op, T_k_op, T_k_ID,
                                T_KGrids1, T_KGrids2, T_KGrids3,
                                myworld1,
                                NPROCS_ID1,
                                Comm_World1,
                                NPROCS_WD1,
                                Comm_World_StartID1,
                                MPI_CommWD1,
                                myworld2,
                                NPROCS_ID2,
                                NPROCS_WD2,
                                Comm_World2,
                                Comm_World_StartID2,
                                MPI_CommWD2);
        }

        else
        {

          if (GB_switch == 0)
          {

            time5 += Band_DFT_NonCol(LSCF_iter,
                                     Kspace_grid1, Kspace_grid2, Kspace_grid3,
                                     SpinP_switch, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1,
                                     MP, order_GA, ko_noncol, koS, EIGEN_Band,
                                     H1, S1,
                                     rHs11_Cx, rHs22_Cx, rHs12_Cx, iHs11_Cx, iHs22_Cx, iHs12_Cx,
                                     EVec1_Cx,
                                     Ss_Cx, Cs_Cx, Hs_Cx,
                                     Ss2_Cx, Cs2_Cx, Hs2_Cx,
                                     k_op, T_k_op, T_k_ID,
                                     T_KGrids1, T_KGrids2, T_KGrids3,
                                     myworld1,
                                     NPROCS_ID1,
                                     Comm_World1,
                                     NPROCS_WD1,
                                     Comm_World_StartID1,
                                     MPI_CommWD1,
                                     myworld2,
                                     NPROCS_ID2,
                                     NPROCS_WD2,
                                     Comm_World2,
                                     Comm_World_StartID2,
                                     MPI_CommWD2);
          }
          else if (GB_switch == 1)
          {
            /* added by T. B. Prayitno and supervised by Prof. F. Ishii*/
            time5 += Band_DFT_NonCol_GB(LSCF_iter,
                                        koSU, koSL, SU_Band, SL_Band,
                                        Kspace_grid1, Kspace_grid2, Kspace_grid3,
                                        SpinP_switch, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
          }
        }

        break;

      case 4:
        /* revised by Y. Xiao for Noncollinear NEGF calculations*/
        if (SpinP_switch < 2)
        {
          time5 += TRAN_DFT(mpi_comm_level1, SucceedReadingDMfile,
                            level_stdout, LSCF_iter, SpinP_switch, H, iHNL, OLP[0],
                            atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn,
                            M2G, G2ID, F_G2M, atv_ijk, List_YOUSO,
                            DM[0], EDM, TRAN_DecMulP, Eele0, Eele1, ChemP_e0);
        }
        else
        {
          time5 += TRAN_DFT_NC(mpi_comm_level1, SucceedReadingDMfile,
                               level_stdout, LSCF_iter, SpinP_switch, H, iHNL, OLP[0],
                               atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn,
                               M2G, G2ID, F_G2M, atv_ijk, List_YOUSO, koS, S_Band,
                               DM[0], iDM[0], EDM, TRAN_DecMulP, Eele0, Eele1, ChemP_e0);
        } /* until here by Y. Xiao for Noncollinear NEGF calculations*/

        break;

      case 5:
        time5 += Divide_Conquer("scf", LSCF_iter, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 6:
        /* not supported */
        break;

      case 7:
        break;

      case 8:
        time5 += Krylov("scf", LSCF_iter, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 9:
        time5 += Cluster_DFT_ON2("scf", LSCF_iter, SpinP_switch,
                                 CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
        break;

      case 10:
        /* not supported */
        break;

      case 11:
        time5 += Divide_Conquer_LNO("scf", MD_iter, LSCF_iter, SucceedReadingDMfile, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
        break;
      }
    }

    Uele_OS0 = Eele0[0];
    Uele_OS1 = Eele0[1];
    Uele_IS0 = Eele1[0];
    Uele_IS1 = Eele1[1];
    Uele = Uele_IS0 + Uele_IS1;

    /*****************************************************
                   Orbital magnetic moment
    *****************************************************/

    if (SpinP_switch == 3)
      Orbital_Moment("non");

    /*****************************************************
                      Mulliken charge
    *****************************************************/

    time14 += Mulliken_Charge("stdout");

    /*****************************************************
                    check SCF-convergence
    *****************************************************/

    if (Solver == 4)
    {
      dUele = 1.0; /* do not calculate Uele */
    }
    else
    {

      if (SCF_iter == 1)
        dUele = 1.0;
      else
        dUele = fabs(Uele - pUele);
    }

    if (
        2 <= SCF_iter &&
        ((dUele < SCF_Criterion && NormRD[0] < (10 * SCF_Criterion) && Cnt_switch == 0) ||
         (dUele < SCF_Criterion && Cnt_switch == 1 && Cnt_Now != 1) ||
         (dUele < SCF_Criterion && Cnt_switch == 1 && Cnt_Now == 1 && OrbOpt_end == 1)))
      po = 1;

    /*****************************************************
                     orbital optimization
    *****************************************************/

    if (Cnt_switch == 1 && Cnt_Now == 1 && OrbOpt_end != 1 &&
        (LSCF_iter % orbitalOpt_SCF == 0 ||
         (dUele < SCF_Criterion && NormRD[0] < 1.0e-7)))
    {

      /* calculate the total energy */

      /*
  time10 += Set_Orbitals_Grid(1);
  time11 += Set_Density_Grid(1, 0, DM[0], Density_Grid_B);
  time7 += Force(H0,DS_NL,OLP,DM[0],EDM);
  time8 += Total_Energy(MD_iter,DM[0],ECE);
  time10 += Set_Orbitals_Grid(0);

  TotalE = 0.0;
  for (i=0; i<=12; i++){
  TotalE += ECE[i];
  }
  printf("orbitalOpt_iter=%3d Etot=%15.12f\n",orbitalOpt_iter,TotalE);
      */

      /* optimization of contraction coefficients */

      if (Opt_Contraction(orbitalOpt_iter,
                          0.0, /* <- TotalE  */
                          H, OLP, DM[0], EDM,
                          His_CntCoes, His_D_CntCoes,
                          His_CntCoes_Species, His_D_CntCoes_Species,
                          His_OrbOpt_Etot, OrbOpt_Hessian) < orbitalOpt_criterion)
      {

        SCF_MAX = SCF_iter + orbitalOpt_SCF - 1;
        OrbOpt_end = 1;
      }

      /* save the information at iteration */
      outputfile1(3, MD_iter, orbitalOpt_iter, 0, SCF_iter, file_OrbOpt, ChemP_e0);

      orbitalOpt_iter++;
      LSCF_iter = 0;
      SCF_iter_shift = 0;
      ETemp_controller = 0;

      if ((orbitalOpt_SCF * (orbitalOpt_MD + 1) - 1) < SCF_iter)
        po = 1;
      if (orbitalOpt_MD < orbitalOpt_iter)
      {
        SCF_MAX = SCF_iter + orbitalOpt_SCF - 1;
        OrbOpt_end = 1;
      }
    }

    /*****************************************************
                   mixing of density matrices
    *****************************************************/

    if (SCF_iter == 1 || LSCF_iter == 0)
      Calc_CntOrbital_ON = 1;
    else
      Calc_CntOrbital_ON = 0;

    /********************************************************
      control the electric temperature
      for accelerating SCF convergence
    ********************************************************/

    if (Solver != 4 && SCF_Control_Temp == 1)
    {

      Norm5 = Norm4;
      Norm4 = Norm3;
      Norm3 = Norm2;
      Norm2 = Norm1;
      Norm1 = NormRD[0];
      tmp = (Norm1 + Norm2 + Norm3 + Norm4 + Norm5) / 5.0;
      tmp0 = sqrt(fabs(NormRD[0]));

      if (0.01 < tmp0 && tmp < Norm1 && 6 < LSCF_iter && ETemp_controller == 0)
      {
        E_Temp = 10.0 * Original_E_Temp;
        n = LSCF_iter - Pulay_SCF + 2;
        if (0 <= n && n < LSCF_iter)
          SCF_iter_shift = n;
        ETemp_controller = 1;
      }
      else if (tmp0 <= 0.01 && ETemp_controller <= 1)
      {
        E_Temp = 1.0 * Original_E_Temp;
        n = LSCF_iter - Pulay_SCF + 2;
        if (0 <= n && n < LSCF_iter)
          SCF_iter_shift = n;
        ETemp_controller = 2;
      }

      /* update Beta */
      Beta = 1.0 / kB / E_Temp;
    }

    /********************************************************
     simple, RMM-DIIS, or GR-Pulay mixing for density matrix
    ********************************************************/

    if (Mixing_switch == 0 || Mixing_switch == 1 || Mixing_switch == 2 || Mixing_switch == 6)
    {

      time6 += Mixing_DM(MD_iter,
                         LSCF_iter - SCF_iter_shift,
                         SCF_iter - SCF_iter_shift,
                         SucceedReadingDMfile,
                         ReRhok, ImRhok,
                         Residual_ReRhok, Residual_ImRhok,
                         ReVk, ImVk, ReRhoAtomk, ImRhoAtomk);

      time11 += Set_Density_Grid(Cnt_kind, Calc_CntOrbital_ON, DM[0], Density_Grid_B);

      if (Solver == 4)
      {
        TRAN_Add_Density_Lead(mpi_comm_level1,
                              SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                              My_NumGridB_AB, Density_Grid_B);
      }

      if (SpinP_switch == 3)
        diagonalize_nc_density(Density_Grid_B);

      fft_charge_flag = 1;
    }

    /********************************************************
        Kerker or RMM-DIISK mixing for density in k-space
    ********************************************************/

    else if (Mixing_switch == 3 || Mixing_switch == 4)
    {

      time11 += Set_Density_Grid(Cnt_kind, Calc_CntOrbital_ON, DM[0], Density_Grid_B);

      if (Solver == 4)
      {
        TRAN_Add_Density_Lead(mpi_comm_level1,
                              SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                              My_NumGridB_AB, Density_Grid_B);
      }

      /* non-spin polarization */
      if (SpinP_switch == 0)
      {
        if (Solver != 4 || TRAN_Poisson_flag2 == 2)
          time15 += FFT_Density(1, ReRhok[0][0], ImRhok[0][0]);
        else
          time15 += FFT2D_Density(1, ReRhok[0][0], ImRhok[0][0]);
      }

      /* collinear spin polarization */
      else if (SpinP_switch == 1)
      {
        if (Solver != 4 || TRAN_Poisson_flag2 == 2)
        {
          time15 += FFT_Density(1, ReRhok[0][0], ImRhok[0][0]);
          time15 += FFT_Density(2, ReRhok[0][1], ImRhok[0][1]);
        }
        else
        {
          time15 += FFT2D_Density(1, ReRhok[0][0], ImRhok[0][0]);
          time15 += FFT2D_Density(2, ReRhok[0][1], ImRhok[0][1]);
        }
      }

      /* non-collinear spin polarization */
      else if (SpinP_switch == 3)
      {
        if (Solver != 4 || TRAN_Poisson_flag2 == 2)
        {
          time15 += FFT_Density(1, ReRhok[0][0], ImRhok[0][0]);
          time15 += FFT_Density(2, ReRhok[0][1], ImRhok[0][1]);
          time15 += FFT_Density(4, ReRhok[0][2], ImRhok[0][2]);
        }
        else
        {
          time15 += FFT2D_Density(1, ReRhok[0][0], ImRhok[0][0]);
          time15 += FFT2D_Density(2, ReRhok[0][1], ImRhok[0][1]);
          time15 += FFT2D_Density(4, ReRhok[0][2], ImRhok[0][2]);
        }
      }

      if (SCF_iter == 1)
      {
        if (Solver != 4 || TRAN_Poisson_flag2 == 2)
          time15 += FFT_Density(3, ReRhoAtomk, ImRhoAtomk);
        else
          time15 += FFT2D_Density(3, ReRhoAtomk, ImRhoAtomk);
      }

      time6 += Mixing_DM(MD_iter,
                         LSCF_iter - SCF_iter_shift,
                         SCF_iter - SCF_iter_shift,
                         SucceedReadingDMfile,
                         ReRhok, ImRhok,
                         Residual_ReRhok, Residual_ImRhok,
                         ReVk, ImVk, ReRhoAtomk, ImRhoAtomk);

      if (SpinP_switch == 3)
        diagonalize_nc_density(Density_Grid_B);

      fft_charge_flag = 0;
    }

    /**********************************************************
     RMM-DIISH which is RMM-DIIS mixing method for Hamiltonian
     or
     RMM-DIISV mixing for Kohn-Sham potential in k-space
    **********************************************************/

    else if (Mixing_switch == 5 || Mixing_switch == 7)
    {

      time11 += Set_Density_Grid(Cnt_kind, Calc_CntOrbital_ON, DM[0], Density_Grid_B);

      if (Solver == 4)
      {
        TRAN_Add_Density_Lead(mpi_comm_level1,
                              SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                              My_NumGridB_AB, Density_Grid_B);
      }

      if (SpinP_switch == 3)
        diagonalize_nc_density(Density_Grid_B);

      fft_charge_flag = 1;
    }

    else
    {
      printf("unknown scf.Mixing.Type\n");
      fflush(stdout);
      MPI_Finalize();
      exit(0);
    }

    /*****************************************************
        calculate occupation number for LDA+U
        and calculate the effective potential
        for LDA+U, constraint for spin, Zeeman term for
        spin, Zeeman term for orbital
    *****************************************************/

    if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
    {

      Occupation_Number_LDA_U(SCF_iter, SucceedReadingDMfile, dUele, ECE, "stdout");
    }

    /*****************************************************
                       print informations
    *****************************************************/

    if (MYID_MPI_COMM_WORLD == Host_ID)
    {

      /* spin non-collinear */
      if (SpinP_switch == 3)
      {

        if (0 < level_stdout)
        {
          printf("<DFT>  Total Spin Moment    (muB) %12.9f   Angles %14.9f  %14.9f\n",
                 2.0 * Total_SpinS, Total_SpinAngle0 / PI * 180.0,
                 Total_SpinAngle1 / PI * 180.0);
          fflush(stdout);
          printf("<DFT>  Total Orbital Moment (muB) %12.9f   Angles %14.9f  %14.9f\n",
                 Total_OrbitalMoment,
                 Total_OrbitalMomentAngle0 / PI * 180.0,
                 Total_OrbitalMomentAngle1 / PI * 180.0);
          fflush(stdout);
        }

        x = 2.0 * Total_SpinS * sin(Total_SpinAngle0) * cos(Total_SpinAngle1);
        y = 2.0 * Total_SpinS * sin(Total_SpinAngle0) * sin(Total_SpinAngle1);
        z = 2.0 * Total_SpinS * cos(Total_SpinAngle0);

        x += Total_OrbitalMoment * sin(Total_OrbitalMomentAngle0) * cos(Total_OrbitalMomentAngle1);
        y += Total_OrbitalMoment * sin(Total_OrbitalMomentAngle0) * sin(Total_OrbitalMomentAngle1);
        z += Total_OrbitalMoment * cos(Total_OrbitalMomentAngle0);

        xyz2spherical(x, y, z, 0.0, 0.0, 0.0, S_coordinate);

        if (0 < level_stdout)
        {
          printf("<DFT>  Total Moment         (muB) %12.9f   Angles %14.9f  %14.9f\n",
                 S_coordinate[0], S_coordinate[1] / PI * 180.0, S_coordinate[2] / PI * 180.0);
          fflush(stdout);
        }
      }
      /* spin collinear */
      else
      {
        if (0 < level_stdout)
        {
          printf("<DFT>  Total Spin Moment (muB) = %15.12f\n", 2.0 * Total_SpinS);
          fflush(stdout);
        }
      }

      if (Mixing_switch == 0 && 0 < level_stdout)
      {
        printf("<DFT>  Mixing_weight=%15.12f SCF_RENZOKU=%2d\n",
               Mixing_weight, SCF_RENZOKU);
        fflush(stdout);
      }
      else
      {
        if (0 < level_stdout)
        {
          printf("<DFT>  Mixing_weight=%15.12f\n", Mixing_weight);
          fflush(stdout);
        }
      }

      if (0 < level_stdout)
      {
        printf("<DFT>  Uele   =%18.12f  dUele     =%17.12f\n", Uele, dUele);
        fflush(stdout);
        printf("<DFT>  NormRD =%18.12f  Criterion =%17.12f\n",
               sqrt(fabs(NormRD[0])), SCF_Criterion);
        fflush(stdout);
      }
      else if (0 == level_stdout)
      {
        printf("<DFT> MD=%4d SCF=%4d   NormRD =%18.12f  Criterion =%17.12f\n",
               MD_iter, SCF_iter, sqrt(fabs(NormRD[0])), SCF_Criterion);
        fflush(stdout);
      }
    }

    if (Solver == 4)
    {

      /* check whether the SCF converges or not */

      if (sqrt(fabs(NormRD[0])) < SCF_Criterion)
      {

        po = 1;

        /* printf("TRAN  NormRD < SCF_Criterion, SCF is satisfied \n");	*/
      }
    }

    outputfile1(1, MD_iter, orbitalOpt_iter, Cnt_Now, SCF_iter, file_DFTSCF, ChemP_e0);

    /*****************************************************
                         Uele -> pUele
    *****************************************************/

    pUele = Uele;

    /*****************************************************
            on-the-fly adjustment of Pulay_SCF
    *****************************************************/

    if (1 < MD_iter && SucceedReadingDMfile == 1 && Scf_RestartFromFile != -1 && Pulay_SCF == Pulay_SCF_original && SCF_iter < Pulay_SCF_original && NormRD[0] < 0.01 && Pulay_SCF_original < SCF_MAX)
    {

      Pulay_SCF = SCF_iter + 1;
    }

    /*************************************************************
     If there is a proper file, change parameters controlling the
     SCF iteration, based on keywords described in the file.
    **************************************************************/

    Read_SCF_keywords();
    if (Cnt_switch == 0)
      SCF_MAX = DFTSCF_loop;

    /************************************************************************
                              end of SCF calculation
    ************************************************************************/

  } while (po == 0 && SCF_iter < SCF_MAX);

  /*******************************************************************
    if Solver==2 (Cluster) && SpinP_switch==3, calculate EDM and PDM.
  *******************************************************************/

  if (Solver == 2 && SpinP_switch == 3)
  {

    /* EDM */

    time5 += Calc_DM_Cluster_non_collinear_ScaLAPACK(7, myid0, numprocs0, size_H1, is2, ie2, MP, n, n * 2,
                                                     DM[0], iDM[0], EDM, ko_noncol, CDM1, Work1, EVec1_NonCol);

    time5 += Calc_DM_Cluster_non_collinear_ScaLAPACK(8, myid0, numprocs0, size_H1, is2, ie2, MP, n, n * 2,
                                                     DM[0], iDM[0], EDM, ko_noncol, CDM1, Work1, EVec1_NonCol);

    time5 += Calc_DM_Cluster_non_collinear_ScaLAPACK(9, myid0, numprocs0, size_H1, is2, ie2, MP, n, n * 2,
                                                     DM[0], iDM[0], EDM, ko_noncol, CDM1, Work1, EVec1_NonCol);

    time5 += Calc_DM_Cluster_non_collinear_ScaLAPACK(10, myid0, numprocs0, size_H1, is2, ie2, MP, n, n * 2,
                                                     DM[0], iDM[0], EDM, ko_noncol, CDM1, Work1, EVec1_NonCol);

    /* Partial_DM */

    if (cal_partial_charge)
    {

      time5 += Calc_DM_Cluster_non_collinear_ScaLAPACK(11, myid0, numprocs0, size_H1, is2, ie2, MP, n, n * 2,
                                                       DM[0], iDM[0], EDM, ko_noncol, CDM1, Work1, EVec1_NonCol);

      time5 += Calc_DM_Cluster_non_collinear_ScaLAPACK(12, myid0, numprocs0, size_H1, is2, ie2, MP, n, n * 2,
                                                       DM[0], iDM[0], EDM, ko_noncol, CDM1, Work1, EVec1_NonCol);
    }
  }

  /*******************************************************************
    calculations of optical conductivities and dielectric functions,
    developed by YTL (2018/09/13)
  *******************************************************************/

  if (CDDF_on == 1)
  {

    time5 += Calc_optical_conductivities_dielectric_functions(
        myid0,
        ko_col, Eele0, Eele1,
        MP, order_GA, ko, koS, EIGEN_Band, S1, H1,
        CDM1, H_Band_Col, S_Band, EDM1,
        myworld1, myworld2, NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1, MPI_CommWD1,
        NPROCS_ID2, NPROCS_WD2, Comm_World2, Comm_World_StartID2, MPI_CommWD2,
        is2, ie2, size_H1, SP_NZeros, SP_Atoms, EVec1_Re, Work1,
        Ss_Re, Cs_Re, Hs_Re, Ss_Cx, Cs_Cx, Hs_Cx);
  }

  /*******************************************************************
   if (Solver==10), recalculate the self-energy and
   Green's funciton using the last Hamitonian
  *******************************************************************/

  /*
  if (Solver==10 && scf_GF_EGAC==0){
    time5 += EGAC_DFT("scf",2,SpinP_switch,H,iHNL,OLP[0],DM[0],EDM,Eele0,Eele1);
  }
  */

  /*******************************************************************
          making of the input data for TranMain
  *******************************************************************/

  /* revised by Y. Xiao for Noncollinear NEGF calculations */ /* iHNL is outputed */
  TRAN_Output_Trans_HS(mpi_comm_level1, Solver, SpinP_switch, ChemP, H, iHNL, OLP, H0,
                       atomnum, SpeciesNum, WhatSpecies,
                       Spe_Total_CNO, FNAN, natn, ncn, G2ID, atv_ijk,
                       Max_FSNAN, ScaleSize, F_G2M, TCpyCell, List_YOUSO,
                       filepath, filename, "tranb");

  /*******************************************************************
              save contraction coefficients
  *******************************************************************/

  if (Cnt_switch == 1)
    File_CntCoes("write");

  /*******************************************************************
              save orbital magnetic moment
  *******************************************************************/

  if (SpinP_switch == 3)
    Orbital_Moment("write");

  /*******************************************************************
              save Mulliken charge
  *******************************************************************/

  time14 += Mulliken_Charge("write");

  /*******************************************************************
            save occupation number in LDA+U
  *******************************************************************/

  /* ---- added by MJ */
  if (Hub_U_switch == 1 || 1 <= Constraint_NCS_switch || Zeeman_NCS_switch == 1 || Zeeman_NCO_switch == 1)
  {

    Occupation_Number_LDA_U(SCF_iter, SucceedReadingDMfile, dUele, ECE, "write");
  }

  /*****************************************************
            Natural Bond Orbital (NBO) Analysis

    NBO_switch = 1 : for Cluster
    NBO_switch = 2 : for Krylov
    NBO_switch = 3 : for Band
    NBO_switch = 4 : for Band (at designated k-points)
  *****************************************************/

  if (NBO_switch == 1)
  {
    if (Solver == 2)
    {
      Calc_NAO_Cluster(DM[0]);
    }
    else
    {
      if (MYID_MPI_COMM_WORLD == Host_ID)
      {
        printf("# NBO ERROR: Please check combination of calc. types of SCF and NAO\n");
      }
    }
  }

  if (NBO_switch == 2)
  {
    if (Solver == 8)
    {
      Calc_NAO_Krylov(H, OLP[0], DM[0]);
    }
    else
    {
      if (MYID_MPI_COMM_WORLD == Host_ID)
      {
        printf("# NBO ERROR: Please check combination of calc. types of SCF and NAO\n");
      }
    }
  }

  /*
  if (NBO_switch == 3 || NBO_switch == 4){

    int Nkpoint_NAO;
    double **kpoint_NAO;

    if      (NBO_switch == 3) Nkpoint_NAO = T_knum;
    else if (NBO_switch == 4) Nkpoint_NAO = NAO_Nkpoint;

    kpoint_NAO = (double**)malloc(sizeof(double*)*(Nkpoint_NAO+1));
    for (i=0; i<(Nkpoint_NAO+1); i++){
      kpoint_NAO[i] = (double*)malloc(sizeof(double)*4);
    }

    if (NBO_switch == 3){
      for (i=0; i<Nkpoint_NAO; i++){
        kpoint_NAO[i][1] = T_KGrids1[i];
        kpoint_NAO[i][2] = T_KGrids2[i];
        kpoint_NAO[i][3] = T_KGrids3[i];
      }
    }
    else if (NBO_switch == 4){
      for (i=0; i<Nkpoint_NAO; i++){
  kpoint_NAO[i][1] = NAO_kpoint[i][1];
  kpoint_NAO[i][2] = NAO_kpoint[i][2];
  kpoint_NAO[i][3] = NAO_kpoint[i][3];
      }
    }

    Calc_NAO_Band(Nkpoint_NAO, kpoint_NAO, SpinP_switch, H, OLP[0]);

    for (i=0; i<(Nkpoint_NAO+1); i++){
      free(kpoint_NAO[i]);
    }
    free(kpoint_NAO);

  }
  */

  /*****************************************************
                      Band dispersion
  *****************************************************/

  /* For the LNO method */

  if (Band_disp_switch == 1 && Band_Nkpath > 0 && LNO_flag == 1)
  {

    LNO("scf", SCF_iter, OLP[0], H, DM[0]);
    Band_DFT_kpath_LNO(Band_Nkpath, Band_N_perpath,
                       Band_kpath, Band_kname,
                       SpinP_switch, H, iHNL, OLP[0]);
  }

  /* For the conventional method */

  else if (Band_disp_switch == 1 && Band_Nkpath > 0)
  {
    if (Cnt_switch == 0)
    {

      Band_DFT_kpath(Band_Nkpath, Band_N_perpath,
                     Band_kpath, Band_kname,
                     SpinP_switch, H, iHNL, OLP[0]);
    }
    else
    {

      Band_DFT_kpath(Band_Nkpath, Band_N_perpath,
                     Band_kpath, Band_kname,
                     SpinP_switch, CntH, iCntHNL, CntOLP[0]);
    }
  }

  /******************************************************
          calculation of density of states (DOS)
  ******************************************************/

  if (Dos_fileout || DosGauss_fileout)
  {

    if (Solver == 2)
    { /* cluster */

      if (Cnt_switch == 0)
      {

        if (SpinP_switch <= 1)
        {

          time5 += Cluster_DFT_Col("dos", LSCF_iter, SpinP_switch,
                                   ko_col, H, OLP[0], DM[0], EDM,
                                   Eele0, Eele1,
                                   myworld1, NPROCS_ID1, Comm_World1, NPROCS_WD1,
                                   Comm_World_StartID1, MPI_CommWD1, MP, is2, ie2,
                                   Ss_Re, Cs_Re, Hs_Re,
                                   CDM1, EDM1, PDM1, size_H1, SP_NZeros, SP_Atoms, EVec1_Re, Work1);
        }
        else if (SpinP_switch == 3)
        {

          time5 += Cluster_DFT_NonCol("dos", LSCF_iter, SpinP_switch,
                                      ko_noncol, H, iHNL, OLP[0], DM[0], EDM,
                                      Eele0, Eele1,
                                      MP, is2, ie2,
                                      Ss_Re, Cs_Re,
                                      rHs11_Re, rHs12_Re, rHs22_Re, iHs11_Re, iHs12_Re, iHs22_Re,
                                      Ss2_Cx, Hs2_Cx, Cs2_Cx,
                                      CDM1, size_H1, EVec1_NonCol, Work1);
        }
      }
      else
      {

        if (SpinP_switch <= 1)
        {

          time5 += Cluster_DFT_Col("dos", LSCF_iter, SpinP_switch,
                                   ko_col, CntH, CntOLP[0], DM[0], EDM,
                                   Eele0, Eele1,
                                   myworld1, NPROCS_ID1, Comm_World1, NPROCS_WD1,
                                   Comm_World_StartID1, MPI_CommWD1, MP, is2, ie2,
                                   Ss_Re, Cs_Re, Hs_Re,
                                   CDM1, EDM1, PDM1, size_H1, SP_NZeros, SP_Atoms, EVec1_Re, Work1);
        }
        else if (SpinP_switch == 3)
        {
          /* not supported */
        }
      }
    }

    else if (Solver == 3)
    { /* band */

      if (Cnt_switch == 0)
      {

        Band_DFT_Dosout(Dos_Kgrid[0], Dos_Kgrid[1], Dos_Kgrid[2],
                        SpinP_switch, H, iHNL, OLP[0]);
      }
      else
      {
        Band_DFT_Dosout(Dos_Kgrid[0], Dos_Kgrid[1], Dos_Kgrid[2],
                        SpinP_switch, CntH, iCntHNL, CntOLP[0]);
      }
    }

    else if (Solver == 4)
    { /* NEGF */

      TRAN_DFT_Dosout(mpi_comm_level1, level_stdout, LSCF_iter, SpinP_switch, H, iHNL, OLP[0],
                      atomnum, Matomnum, WhatSpecies, Spe_Total_CNO, FNAN, natn, ncn,
                      M2G, G2ID, atv_ijk, List_YOUSO, Spe_Num_CBasis, SpeciesNum, filename, filepath,
                      DM[0], EDM, Eele0, Eele1);
    }

    else if (Solver == 5)
    { /* divide-conquer */
      if (Cnt_switch == 0)
      {
        time5 += Divide_Conquer("dos", LSCF_iter, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
      }
      else
      {
        time5 += Divide_Conquer("dos", LSCF_iter, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
      }
    }

    else if (Solver == 8)
    { /* Krylov subspace method */
      if (Cnt_switch == 0)
      {
        time5 += Krylov("dos", LSCF_iter, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
      }
      else
      {
        time5 += Krylov("dos", LSCF_iter, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
      }
    }

    else if (Solver == 11)
    { /* DC-LNO */
      if (Cnt_switch == 0)
      {
        time5 += Divide_Conquer_LNO("dos", MD_iter, LSCF_iter, SucceedReadingDMfile, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
      }
      else
      {
        time5 += Divide_Conquer_LNO("dos", MD_iter, LSCF_iter, SucceedReadingDMfile, CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
      }
    }
  }

  /*****************************************************
       write a restart file if SpinP_switch!=3
  *****************************************************/

  if (SpinP_switch != 3)
  {
    RestartFileDFT("write", MD_iter, &Uele, H, CntH, &etime);
    time13 += etime;
    MPI_Barrier(mpi_comm_level1);
  }

  /****************************************************
      Note on the force calculation.
      If you first call Total_Energy.c before
      Force.c, then the force calculation will fail.
  ****************************************************/

  if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {
    printf("<MD=%2d>  Force calculation\n", MD_iter);
    fflush(stdout);
  }

  if (Cnt_switch == 1 && Cnt_Now == 1)
    time10 += Set_Orbitals_Grid(1);
  time11 += Set_Density_Grid(1, 0, DM[0], Density_Grid_B);

  if (Solver == 4)
  {
    TRAN_Add_Density_Lead(mpi_comm_level1,
                          SpinP_switch, Ngrid1, Ngrid2, Ngrid3,
                          My_NumGridB_AB, Density_Grid_B);
  }

  /* to save mixing densities between the previous and current ones
     do charge mixing */

  if (Mixing_switch == 3 || Mixing_switch == 4)
  {

    /* non-spin polarization */
    if (SpinP_switch == 0)
    {
      if (Solver != 4 || TRAN_Poisson_flag2 == 2)
        time15 += FFT_Density(1, ReRhok[0][0], ImRhok[0][0]);
      else
        time15 += FFT2D_Density(1, ReRhok[0][0], ImRhok[0][0]);
    }

    /* collinear spin polarization */
    else if (SpinP_switch == 1)
    {
      if (Solver != 4 || TRAN_Poisson_flag2 == 2)
      {
        time15 += FFT_Density(1, ReRhok[0][0], ImRhok[0][0]);
        time15 += FFT_Density(2, ReRhok[0][1], ImRhok[0][1]);
      }
      else
      {
        time15 += FFT2D_Density(1, ReRhok[0][0], ImRhok[0][0]);
        time15 += FFT2D_Density(2, ReRhok[0][1], ImRhok[0][1]);
      }
    }
    /* non-collinear spin polarization */
    else if (SpinP_switch == 3)
    {
      if (Solver != 4 || TRAN_Poisson_flag2 == 2)
      {
        time15 += FFT_Density(1, ReRhok[0][0], ImRhok[0][0]);
        time15 += FFT_Density(2, ReRhok[0][1], ImRhok[0][1]);
        time15 += FFT_Density(4, ReRhok[0][2], ImRhok[0][2]);
      }
      else
      {
        time15 += FFT2D_Density(1, ReRhok[0][0], ImRhok[0][0]);
        time15 += FFT2D_Density(2, ReRhok[0][1], ImRhok[0][1]);
        time15 += FFT2D_Density(4, ReRhok[0][2], ImRhok[0][2]);
      }
    }

    time6 += Mixing_DM(1,
                       LSCF_iter - SCF_iter_shift,
                       SCF_iter - SCF_iter_shift,
                       SucceedReadingDMfile,
                       ReRhok, ImRhok,
                       Residual_ReRhok, Residual_ImRhok,
                       ReVk, ImVk, ReRhoAtomk, ImRhoAtomk);
  }

  /* write a restart file, it is important to save
     charge density before calling diagonalize_nc_density */

  if (SpinP_switch == 3)
  {
    RestartFileDFT("write", MD_iter, &Uele, H, CntH, &etime);
    time13 += etime;
    MPI_Barrier(mpi_comm_level1);
  }

  if (SpinP_switch == 3)
    diagonalize_nc_density(Density_Grid_B);

  /****************************************************
   if (Solver==9), calculate the energy density matrix
  ****************************************************/

  if (Solver == 9)
  {

    if (Cnt_switch == 0)
    {
      time5 += Cluster_DFT_ON2("force", LSCF_iter, SpinP_switch,
                               H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
    }
    else
    {

      time5 += Cluster_DFT_ON2("force", LSCF_iter, SpinP_switch,
                               CntH, iCntHNL, CntOLP[0], DM[0], EDM, Eele0, Eele1);
    }
  }

  /****************************************************
               calculation of forces
  ****************************************************/

  if (!orbitalOpt_Force_Skip)
    time7 += Force(H0, DS_NL, OLP, DM[0], EDM);

  if (scf_stress_flag)
  {
    Stress(H0, DS_NL, OLP, DM[0], EDM);
  }

  /*
  if (SpinP_switch==3) {
    Stress_NC(H0,DS_NL,OLP,DM[0],EDM);
  }
  else{
    Stress(H0,DS_NL,OLP,DM[0],EDM);
  }
  */

  /****************************************************
     set Pulay_SCF = Pulay_SCF_original in anycase
  ****************************************************/

  Pulay_SCF = Pulay_SCF_original;

  /****************************************************
     if the SCF iteration did not converge,
     set Scf_RestartFromFile = 0,
  ****************************************************/

  if (po == 0 && Scf_RestartFromFile == 1 && 0.3 < NormRD[0])
  {
    Scf_RestartFromFile = 0;
  }

  /****************************************************
               calculate the total energy
  ****************************************************/

  if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {
    printf("<MD=%2d>  Total Energy\n", MD_iter);
    fflush(stdout);
  }

  if (!orbitalOpt_Force_Skip)
    time8 += Total_Energy(MD_iter, DM[0], ECE);

  Uatom = 0.0;
  Ucore = ECE[0];
  UH0 = ECE[1];
  Ukin = ECE[2];
  Una = ECE[3];
  Unl = ECE[4];
  UH1 = ECE[5];
  Uxc0 = ECE[6];
  Uxc1 = ECE[7];
  Uhub = ECE[8];
  Ucs = ECE[9];
  Uzs = ECE[10];
  Uzo = ECE[11];
  Uef = ECE[12];
  UvdW = ECE[13];
  Uch = ECE[14];

  Utot = Ucore + UH0 + Ukin + Una + Unl + UH1 + Uxc0 + Uxc1 + Uhub + Ucs + Uzs + Uzo + Uef + UvdW + Uch;

  /* elapsed time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  if (MYID_MPI_COMM_WORLD == Host_ID && 0 < level_stdout)
  {

    printf("\n*******************************************************\n");
    printf("                Total Energy (Hartree) at MD =%2d        \n", MD_iter);
    printf("*******************************************************\n\n");

    printf("  Uele  = %20.12f\n\n", Uele);
    printf("  Ukin  = %20.12f\n", Ukin);
    printf("  UH0   = %20.12f\n", UH0);
    printf("  UH1   = %20.12f\n", UH1);
    printf("  Una   = %20.12f\n", Una);
    printf("  Unl   = %20.12f\n", Unl);
    printf("  Uxc0  = %20.12f\n", Uxc0);
    printf("  Uxc1  = %20.12f\n", Uxc1);
    printf("  Ucore = %20.12f\n", Ucore);
    printf("  Uhub  = %20.12f\n", Uhub);
    printf("  Ucs   = %20.12f\n", Ucs);
    printf("  Uzs   = %20.12f\n", Uzs);
    printf("  Uzo   = %20.12f\n", Uzo);
    printf("  Uef   = %20.12f\n", Uef);
    printf("  UvdW  = %20.12f\n", UvdW);
    printf("  Uch   = %20.12f\n", Uch);
    printf("  Utot  = %20.12f\n\n", Utot);
    printf("  UpV   = %20.12f\n", UpV);
    printf("  Enpy  = %20.12f\n", Utot + UpV);

    printf("  Note:\n\n");
    printf("  Uele:   band energy\n");
    printf("  Ukin:   kinetic energy\n");
    printf("  UH0:    electric part of screened Coulomb energy\n");
    printf("  UH1:    difference electron-electron Coulomb energy\n");
    printf("  Una:    neutral atom potential energy\n");
    printf("  Unl:    non-local potential energy\n");
    printf("  Uxc0:   exchange-correlation energy for alpha spin\n");
    printf("  Uxc1:   exchange-correlation energy for beta spin\n");
    printf("  Ucore:  core-core Coulomb energy\n");
    printf("  Uhub:   LDA+U energy\n");
    printf("  Ucs:    constraint energy for spin orientation\n");
    printf("  Uzs:    Zeeman term for spin magnetic moment\n");
    printf("  Uzo:    Zeeman term for orbital magnetic moment\n");
    printf("  Uef:    electric energy by electric field\n");
    printf("  UvdW:   semi-empirical vdW energy \n"); /* okuno */
    printf("  Uch:    penalty term to create a core hole\n");
    printf("  UpV:    pressure times volume\n");
    printf("  Enpy:   Enthalpy = Utot + UpV\n");
    printf("  (see also PRB 72, 045121(2005) for the energy contributions)\n\n");

    printf("\n*******************************************************\n");
    printf("           Computational times (s) at MD =%2d            \n", MD_iter);
    printf("*******************************************************\n\n");

    printf("  DFT in total      = %10.5f\n\n", time0);
    printf("  Set_OLP_Kin       = %10.5f\n", time1);
    printf("  Set_Nonlocal      = %10.5f\n", time2);
    printf("  Set_ProExpn_VNA   = %10.5f\n", time12);
    printf("  Set_Hamiltonian   = %10.5f\n", time3);
    printf("  Poisson           = %10.5f\n", time4);
    printf("  diagonalization   = %10.5f\n", time5);
    printf("  Mixing_DM         = %10.5f\n", time6);
    printf("  Force             = %10.5f\n", time7);
    printf("  Total_Energy      = %10.5f\n", time8);
    printf("  Set_Aden_Grid     = %10.5f\n", time9);
    printf("  Set_Orbitals_Grid = %10.5f\n", time10);
    printf("  Set_Density_Grid  = %10.5f\n", time11);
    printf("  RestartFileDFT    = %10.5f\n", time13);
    printf("  Mulliken_Charge   = %10.5f\n", time14);
    printf("  FFT(2D)_Density   = %10.5f\n", time15);
  }

  outputfile1(2, MD_iter, 0, 0, SCF_iter, file_DFTSCF, ChemP_e0);

  /*****************************************************
         Calc. of Bloch waves at given k-points
  *****************************************************/

  if (MO_fileout == 1 && MO_Nkpoint > 0 && (Solver == 3 || PeriodicGamma_flag == 1))
  {
    if (Cnt_switch == 0)
    {
      Band_DFT_MO(MO_Nkpoint, MO_kpoint, SpinP_switch, H, iHNL, OLP[0]);
    }
    else
    {
      Band_DFT_MO(MO_Nkpoint, MO_kpoint, SpinP_switch, CntH, iCntHNL, CntOLP[0]);
    }
  }

  /****************************************************
                     updating CompTime
  ****************************************************/

  /* The Sum of Atomic Energy */
  /* if (iter==1) Atomic_Energy();  */
  /* Output_Energies_Forces(fp_DFTSCF); */

  CompTime[myid0][5] += time1;   /* Set_OLP_Kin       */
  CompTime[myid0][6] += time2;   /* Set_Nonlocal      */
  CompTime[myid0][7] += time3;   /* Set_Hamiltonian   */
  CompTime[myid0][8] += time4;   /* Poisson           */
  CompTime[myid0][9] += time5;   /* diagonalization   */
  CompTime[myid0][10] += time6;  /* Mixing_DM         */
  CompTime[myid0][11] += time7;  /* Force             */
  CompTime[myid0][12] += time8;  /* Total_Energy      */
  CompTime[myid0][13] += time9;  /* Set_Aden_Grid     */
  CompTime[myid0][14] += time10; /* Set_Orbitals_Grid */
  CompTime[myid0][15] += time11; /* Set_Density_Grid  */
  CompTime[myid0][16] += time12; /* Set_ProExpn_VNA   */
  CompTime[myid0][17] += time13; /* RestartFileDFT    */
  CompTime[myid0][18] += time14; /* Mulliken_Charge   */
  CompTime[myid0][19] += time15; /* FFT(2D)_Density   */

  /* added by Chi-Cheng for unfolding */
  /*****************************************************
         Calc. of unfolded weight at given k-points
  *****************************************************/

  if (unfold_electronic_band == 1 && unfold_Nkpoint > 0 && (Solver == 2 || Solver == 3))
  {
    if (Cnt_switch == 0)
    {
      Unfolding_Bands(unfold_Nkpoint, unfold_kpoint, SpinP_switch, H, iHNL, OLP[0]);
    }
    else
    {
      Unfolding_Bands(unfold_Nkpoint, unfold_kpoint, SpinP_switch, CntH, iCntHNL, CntOLP[0]);
    }
  }
  /* end for unfolding */

  /*******************************************************************
     freeing of arrays for cluster and band calculations
  *******************************************************************/

  if (Solver == 2 && SpinP_switch <= 1)
    Allocate_Free_Cluster_Col(2);
  else if (Solver == 2 && SpinP_switch == 3)
    Allocate_Free_Cluster_NonCol(2);
  else if (Solver == 3 && SpinP_switch <= 1)
    Allocate_Free_Band_Col(2);
  else if (Solver == 3 && SpinP_switch == 3)
    Allocate_Free_Band_NonCol(2);
  else if (Solver == 4 && SpinP_switch == 3)
    Allocate_Free_NEGF_NonCol(2);

  /*********************************************************
                      freeing of arrays
  *********************************************************/

  Allocate_Free_GridData(2);
  if (Cnt_switch == 1)
    Allocate_Free_OrbOpt(2);

  if (Solver == 4)
  {

    if (SpinP_switch <= 1)
    {
      TRAN_Deallocate_Lead_Region();
      TRAN_Deallocate_Cregion(SpinP_switch);
    }
    else
    {
      TRAN_Deallocate_Lead_Region_NC();
      TRAN_Deallocate_Cregion_NC(SpinP_switch);
    }

    TRAN_Deallocate_Electrode_Grid(Ngrid2);
  }

  MPI_Barrier(mpi_comm_level1);

  return time0;
}

void Read_SCF_keywords()
{
  int po, myid;
  char fname[YOUSO10];
  FILE *fp;

  MPI_Comm_rank(mpi_comm_level1, &myid);

  sprintf(fname, "%s%s_SCF_keywords", filepath, filename);

  if (myid == Host_ID)
  {

    if ((fp = fopen(fname, "r")) != NULL)
      po = 1;
    else
      po = 0;

    if (po == 1)
    {

      printf("\n The keywords for SCF iteration are renewed by %s.\n", fname);
      fflush(stdout);

      /* open the file */

      input_open(fname);

      /* scf.maxIter */

      input_int("scf.maxIter", &DFTSCF_loop, DFTSCF_loop);

      /* scf.Min.Mixing.Weight */

      input_double("scf.Min.Mixing.Weight", &Min_Mixing_weight, Min_Mixing_weight);

      /* scf.Max.Mixing.Weight */

      input_double("scf.Max.Mixing.Weight", &Max_Mixing_weight, Max_Mixing_weight);

      /* scf.Kerker.factor */

      input_double("scf.Kerker.factor", &Kerker_factor, Kerker_factor);

      /* scf.Mixing.StartPulay */

      input_int("scf.Mixing.StartPulay", &Pulay_SCF, Pulay_SCF);

      /* scf.criterion */

      input_double("scf.criterion", &SCF_Criterion, SCF_Criterion);

      /* close the file */

      input_close();
    }

  } /* if (myid==Host_ID) */

  /* MPI_Bcast from Host_ID */

  MPI_Bcast(&po, 1, MPI_INT, Host_ID, mpi_comm_level1);

  if (po == 1)
  {
    MPI_Bcast(&DFTSCF_loop, 1, MPI_INT, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Min_Mixing_weight, 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Max_Mixing_weight, 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Kerker_factor, 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    MPI_Bcast(&Pulay_SCF, 1, MPI_INT, Host_ID, mpi_comm_level1);
    MPI_Bcast(&SCF_Criterion, 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
  }
}

void Output_Energies_Forces(FILE *fp)
{
  int ct_AN;
  double sumx, sumy, sumz;
  double AEx, AEy, AEz;

  fprintf(fp, "\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "                     Energies (Hartree)                    \n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "    Uatom                         = %20.14f\n", Uatom);
  fprintf(fp, "    Uele for   up-spin (OS)       = %20.14f\n", Uele_OS0);
  fprintf(fp, "    Uele for   up-spin (IS)       = %20.14f\n", Uele_IS0);
  fprintf(fp, "    Uele for down-spin (OS)       = %20.14f\n", Uele_OS1);
  fprintf(fp, "    Uele for down-spin (IS)       = %20.14f\n", Uele_IS1);
  fprintf(fp, "    Uxc for up-spin               = %20.14f\n", Uxc0);
  fprintf(fp, "    Uxc for down-spin             = %20.14f\n", Uxc1);
  fprintf(fp, "    UH0                           = %20.14f\n", UH0);
  fprintf(fp, "    UH1                           = %20.14f\n", UH1);
  fprintf(fp, "    UH2                           = %20.14f\n", UH2);
  fprintf(fp, "    Ucore                         = %20.14f\n", Ucore);
  fprintf(fp, "    Udc  (Uxc+UH0+UH1+UH2+Ucore)  = %20.14f\n", Udc);
  fprintf(fp, "    Utot (Uele+Udc)               = %20.14f\n", Utot);
  fprintf(fp, "    Ucoh (Utot-Uatom)             = %20.14f\n", Ucoh);

  fprintf(fp, "\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "                  Energies/atom (Hartree)                  \n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "    Uatom                         = %20.14f\n", Uatom / atomnum);
  fprintf(fp, "    Uele for   up-spin (OS)       = %20.14f\n", Uele_OS0 / atomnum);
  fprintf(fp, "    Uele for   up-spin (IS)       = %20.14f\n", Uele_IS0 / atomnum);
  fprintf(fp, "    Uele for down-spin (OS)       = %20.14f\n", Uele_OS1 / atomnum);
  fprintf(fp, "    Uele for down-spin (IS)       = %20.14f\n", Uele_IS1 / atomnum);
  fprintf(fp, "    Uxc for up-spin               = %20.14f\n", Uxc0 / atomnum);
  fprintf(fp, "    Uxc for down-spin             = %20.14f\n", Uxc1 / atomnum);
  fprintf(fp, "    UH0                           = %20.14f\n", UH0 / atomnum);
  fprintf(fp, "    UH1                           = %20.14f\n", UH1 / atomnum);
  fprintf(fp, "    UH2                           = %20.14f\n", UH2 / atomnum);
  fprintf(fp, "    Ucore                         = %20.14f\n", Ucore / atomnum);
  fprintf(fp, "    Udc  (Uxc+UH0+UH1+UH2+Ucore)  = %20.14f\n", Udc / atomnum);
  fprintf(fp, "    Utot (Uele+Udc)               = %20.14f\n", Utot / atomnum);
  fprintf(fp, "    Ucoh (Utot-Uatom)             = %20.14f\n", Ucoh / atomnum);

  fprintf(fp, "\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "               Force on Atom (Hartree/bohr)                \n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "\n");

  fprintf(fp, "                   Fx            Fy            Fz\n");

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;

  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
  {
    fprintf(fp, "   %i %i        %12.9f  %12.9f  %12.9f\n",
            ct_AN, WhatSpecies[ct_AN],
            -Gxyz[ct_AN][17], -Gxyz[ct_AN][18], -Gxyz[ct_AN][19]);

    sumx = sumx - Gxyz[ct_AN][17];
    sumy = sumy - Gxyz[ct_AN][18];
    sumz = sumz - Gxyz[ct_AN][19];
  }
  fprintf(fp, "\n");
  fprintf(fp, "   Sum of F   %12.9f  %12.9f  %12.9f\n", sumx, sumy, sumz);

  /****************************************************
                   Correction of Force
  ****************************************************/

  AEx = sumx / (double)atomnum;
  AEy = sumy / (double)atomnum;
  AEz = sumz / (double)atomnum;

  fprintf(fp, "\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "            Corrected Force on Atom (Hartree/bohr)         \n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "***********************************************************\n");
  fprintf(fp, "\n");

  fprintf(fp, "                   Fx            Fy            Fz\n");

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;

  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++)
  {
    Gxyz[ct_AN][17] = Gxyz[ct_AN][17] + AEx;
    Gxyz[ct_AN][18] = Gxyz[ct_AN][18] + AEy;
    Gxyz[ct_AN][19] = Gxyz[ct_AN][19] + AEz;

    fprintf(fp, "   %i %i        %12.9f  %12.9f  %12.9f\n",
            ct_AN, WhatSpecies[ct_AN],
            -Gxyz[ct_AN][17], -Gxyz[ct_AN][18], -Gxyz[ct_AN][19]);

    sumx = sumx - Gxyz[ct_AN][17];
    sumy = sumy - Gxyz[ct_AN][18];
    sumz = sumz - Gxyz[ct_AN][19];
  }
  fprintf(fp, "\n");
  fprintf(fp, "   Sum of F   %12.9f  %12.9f  %12.9f\n", sumx, sumy, sumz);
}

double Calc_optical_conductivities_dielectric_functions(
    int myid0,
    double **ko_col,
    double Eele0[2],
    double Eele1[2],
    int *MP,
    int *order_GA,
    double *ko,
    double *koS,
    double ***EIGEN_Band,
    double *S1,
    double *H1,
    double *CDM1,
    dcomplex **H_Band_Col,
    dcomplex **S_Band,
    double *EDM1,
    int myworld1,
    int myworld2,
    int *NPROCS_ID1,
    int *Comm_World1,
    int *NPROCS_WD1,
    int *Comm_World_StartID1,
    MPI_Comm *MPI_CommWD1,
    int *NPROCS_ID2,
    int *NPROCS_WD2,
    int *Comm_World2,
    int *Comm_World_StartID2,
    MPI_Comm *MPI_CommWD2,
    int *is2,
    int *ie2,
    int size_H1,
    int *SP_NZeros,
    int *SP_Atoms,
    double **EVec1_Re,
    double *Work1,
    double *Ss_Re,
    double *Cs_Re,
    double *Hs_Re,
    dcomplex *Ss_Cx,
    dcomplex *Cs_Cx,
    dcomplex *Hs_Cx)
{
  /* YTL-start */
  double Stime_CDDF, Etime_CDDF;
  double time5;

  dtime(&Stime_CDDF);

  /* show a message */
  if (myid0 == Host_ID)
    printf("\n<Optical calculation start>\n");

  switch (Solver)
  {

  case 2:

    time5 = Cluster_DFT_Optical_ScaLAPACK("scf", 1, SpinP_switch,
                                          ko_col, H, iHNL, OLP[0], DM[0], EDM,
                                          NULL, NULL, NULL, Eele0, Eele1,
                                          myworld1, NPROCS_ID1, Comm_World1, NPROCS_WD1,
                                          Comm_World_StartID1, MPI_CommWD1, is2, ie2,
                                          Ss_Re, Cs_Re, Hs_Re,
                                          CDM1, size_H1, SP_NZeros, SP_Atoms, EVec1_Re, Work1);
    break;

  case 3:

    if (SpinP_switch <= 1)
    {

      time5 = Band_DFT_Col_Optical_ScaLAPACK(1,
                                             CDDF_Kspace_grid1, CDDF_Kspace_grid2, CDDF_Kspace_grid3,
                                             SpinP_switch, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1,
                                             MP, order_GA, ko, koS,
                                             H1, S1,
                                             CDM1, EDM1,
                                             H_Band_Col);
    }
    else
    {

      time5 = Band_DFT_NonCol_Optical(1,
                                      koS, S_Band,
                                      CDDF_Kspace_grid1, CDDF_Kspace_grid2, CDDF_Kspace_grid3,
                                      SpinP_switch, H, iHNL, OLP[0], DM[0], EDM, Eele0, Eele1);
    }

    break;
  }

  dtime(&Etime_CDDF);

  if (myid0 == Host_ID)
    printf("<Optical calculations end, time=%7.5f (s)>\n", Etime_CDDF - Stime_CDDF);

  return (Etime_CDDF - Stime_CDDF);
}

void Allocate_Free_Band_Col(int todo_flag)
{
  static int firsttime = 1;
  int ZERO = 0, ONE = 1, info, myid0, numprocs0, myid1, numprocs1, myid2, numprocs2;
  ;
  int spinmax, i, j, k, ii, ij, ik, nblk_m, nblk_m2, wanA, spin, size_EVec1_Cx;
  double tmp, tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1, &numprocs0);
  MPI_Comm_rank(mpi_comm_level1, &myid0);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    spinmax = SpinP_switch + 1;
    Num_Comm_World1 = SpinP_switch + 1;

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = 2 * n;

    is2 = (int *)malloc(sizeof(int) * numprocs0);
    ie2 = (int *)malloc(sizeof(int) * numprocs0);

    MP = (int *)malloc(sizeof(int) * List_YOUSO[1]);
    order_GA = (int *)malloc(sizeof(int) * (List_YOUSO[1] + 1));
    My_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_Atoms = (int *)malloc(sizeof(int) * numprocs0);

    ko = (double *)malloc(sizeof(double) * (n + 1));
    koS = (double *)malloc(sizeof(double) * (n + 1));

    /* find size_H1 */

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    H1 = (double *)malloc(sizeof(double) * size_H1);
    S1 = (double *)malloc(sizeof(double) * size_H1);
    CDM1 = (double *)malloc(sizeof(double) * size_H1);
    EDM1 = (double *)malloc(sizeof(double) * size_H1);

    k_op = (int ***)malloc(sizeof(int **) * Kspace_grid1);
    for (i = 0; i < Kspace_grid1; i++)
    {
      k_op[i] = (int **)malloc(sizeof(int *) * Kspace_grid2);
      for (j = 0; j < Kspace_grid2; j++)
      {
        k_op[i][j] = (int *)malloc(sizeof(int) * Kspace_grid3);
      }
    }

    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        for (k = 0; k < Kspace_grid3; k++)
        {
          k_op[i][j][k] = -999;
        }
      }
    }

    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        for (k = 0; k < Kspace_grid3; k++)
        {

          if (k_op[i][j][k] == -999)
          {

            k_inversion(i, j, k, Kspace_grid1, Kspace_grid2, Kspace_grid3, &ii, &ij, &ik);

            if (i == ii && j == ij && k == ik)
            {
              k_op[i][j][k] = 1;
            }
            else
            {
              k_op[i][j][k] = 2;
              k_op[ii][ij][ik] = 0;
            }
          }
        } /* k */
      }   /* j */
    }     /* i */

    /* find T_knum */

    T_knum = 0;
    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        for (k = 0; k < Kspace_grid3; k++)
        {
          if (0 < k_op[i][j][k])
          {
            T_knum++;
          }
        }
      }
    }

    T_KGrids1 = (double *)malloc(sizeof(double) * T_knum);
    T_KGrids2 = (double *)malloc(sizeof(double) * T_knum);
    T_KGrids3 = (double *)malloc(sizeof(double) * T_knum);
    T_k_op = (int *)malloc(sizeof(int) * T_knum);

    T_k_ID = (int **)malloc(sizeof(int *) * 2);
    for (i = 0; i < 2; i++)
    {
      T_k_ID[i] = (int *)malloc(sizeof(int) * T_knum);
    }

    EIGEN_Band = (double ***)malloc(sizeof(double **) * spinmax);
    for (i = 0; i < spinmax; i++)
    {
      EIGEN_Band[i] = (double **)malloc(sizeof(double *) * T_knum);
      for (j = 0; j < T_knum; j++)
      {
        EIGEN_Band[i][j] = (double *)malloc(sizeof(double) * (n + 1));
        for (k = 0; k < (n + 1); k++)
          EIGEN_Band[i][j][k] = 1.0e+5;
      }
    }

    if (CDDF_on == 1)
    {
      H_Band_Col = (dcomplex **)malloc(sizeof(dcomplex *) * (n + 1));
      for (j = 0; j < n + 1; j++)
      {
        H_Band_Col[j] = (dcomplex *)malloc(sizeof(dcomplex) * (n + 1));
      }
    }

    /***********************************************
      allocation of arrays for the first world
      and
      make the first level worlds
    ***********************************************/

    NPROCS_ID1 = (int *)malloc(sizeof(int) * numprocs0);
    Comm_World1 = (int *)malloc(sizeof(int) * numprocs0);
    NPROCS_WD1 = (int *)malloc(sizeof(int) * Num_Comm_World1);
    Comm_World_StartID1 = (int *)malloc(sizeof(int) * Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm *)malloc(sizeof(MPI_Comm) * Num_Comm_World1);

    Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1,
                     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    /***********************************************
        allocation of arrays for the second world
        and
        make the second level worlds
    ***********************************************/

    if (T_knum <= numprocs1)
      Num_Comm_World2 = T_knum;
    else
      Num_Comm_World2 = numprocs1;

    NPROCS_ID2 = (int *)malloc(sizeof(int) * numprocs1);
    Comm_World2 = (int *)malloc(sizeof(int) * numprocs1);
    NPROCS_WD2 = (int *)malloc(sizeof(int) * Num_Comm_World2);
    Comm_World_StartID2 = (int *)malloc(sizeof(int) * Num_Comm_World2);
    MPI_CommWD2 = (MPI_Comm *)malloc(sizeof(MPI_Comm) * Num_Comm_World2);

    Make_Comm_Worlds(MPI_CommWD1[myworld1], myid1, numprocs1, Num_Comm_World2,
                     &myworld2, MPI_CommWD2, NPROCS_ID2, Comm_World2,
                     NPROCS_WD2, Comm_World_StartID2);

    MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

    double av_num;
    int ke, ks, n3;

    n3 = n;
    av_num = (double)n3 / (double)numprocs2;
    ke = (int)(av_num * (double)(myid2 + 1));
    ks = (int)(av_num * (double)myid2) + 1;
    if (myid1 == 0)
      ks = 1;
    if (myid1 == (numprocs2 - 1))
      ke = n3;
    k = ke - ks + 2;

    EVec1_Cx = (dcomplex **)malloc(sizeof(dcomplex *) * spinmax);
    for (spin = 0; spin < spinmax; spin++)
    {
      EVec1_Cx[spin] = (dcomplex *)malloc(sizeof(dcomplex) * k * n3);
    }
    size_EVec1_Cx = spinmax * k * n3;

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    np_cols = (int)(sqrt((float)numprocs2));
    do
    {
      if ((numprocs2 % np_cols) == 0)
        break;
      np_cols--;
    } while (np_cols >= 2);
    np_rows = numprocs2 / np_cols;

    nblk_m = NBLK;
    while ((nblk_m * np_rows > n || nblk_m * np_cols > n) && (nblk_m > 1))
    {
      nblk_m /= 2;
    }
    if (nblk_m < 1)
      nblk_m = 1;

    MPI_Allreduce(&nblk_m, &nblk, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    my_prow = myid2 / np_cols;
    my_pcol = myid2 % np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols);

    bhandle2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt2 = bhandle2;
    Cblacs_gridinit(&ictxt2, "Row", np_rows, np_cols);

    Ss_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    Hs_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);

    MPI_Allreduce(&na_rows, &na_rows_max, 1, MPI_INT, MPI_MAX, MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols, &na_cols_max, 1, MPI_INT, MPI_MAX, MPI_CommWD2[myworld2]);
    Cs_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows_max * na_cols_max);

    descinit_(descS, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descH, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descC, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_Band_Col: is2", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: ie2", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: MP", sizeof(int) * List_YOUSO[1], NULL);
      PrintMemory("Allocate_Free_Band_Col: order_GA", sizeof(int) * (List_YOUSO[1] + 1), NULL);
      PrintMemory("Allocate_Free_Band_Col: My_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: SP_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: SP_Atoms", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: ko", sizeof(double) * (n + 1), NULL);
      PrintMemory("Allocate_Free_Band_Col: koS", sizeof(double) * (n + 1), NULL);
      PrintMemory("Allocate_Free_Band_Col: H1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Band_Col: S1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Band_Col: CDM1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Band_Col: EDM1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Band_Col: ko_op", sizeof(double) * Kspace_grid1 * Kspace_grid2 * Kspace_grid3, NULL);
      PrintMemory("Allocate_Free_Band_Col: T_KGrids1", sizeof(double) * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_Col: T_KGrids2", sizeof(double) * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_Col: T_KGrids3", sizeof(double) * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_Col: T_k_ID", sizeof(int) * 2 * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_Col: EIGEN_Band", sizeof(double) * spinmax * T_knum * (n + 1), NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_ID1", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World1", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_WD1", sizeof(int) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World_StartID1", sizeof(int) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Band_Col: MPI_CommWD1", sizeof(MPI_Comm) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_ID2", sizeof(int) * numprocs1, NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World2", sizeof(int) * numprocs1, NULL);
      PrintMemory("Allocate_Free_Band_Col: NPROCS_WD2", sizeof(int) * Num_Comm_World2, NULL);
      PrintMemory("Allocate_Free_Band_Col: Comm_World_StartID2", sizeof(int) * Num_Comm_World2, NULL);
      PrintMemory("Allocate_Free_Band_Col: MPI_CommWD2", sizeof(MPI_Comm) * Num_Comm_World2, NULL);
      PrintMemory("Allocate_Free_Band_Col: EVec1_Cx", sizeof(dcomplex) * size_EVec1_Cx, NULL);
      PrintMemory("Allocate_Free_Band_Col: Ss_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_Col: Hs_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_Col: Cs_Cx", sizeof(dcomplex) * na_rows_max * na_cols_max, NULL);
      if (CDDF_on == 1)
      {
        PrintMemory("Allocate_Free_Band_Col: H_Band_Col", sizeof(dcomplex) * (n + 1) * (n + 1), NULL);
      }
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = 2 * n;

    spinmax = SpinP_switch + 1;

    free(is2);
    free(ie2);

    free(MP);
    free(order_GA);
    free(My_NZeros);
    free(SP_NZeros);
    free(SP_Atoms);

    free(ko);
    free(koS);

    /* find size_H1 */

    free(H1);
    free(S1);
    free(CDM1);
    free(EDM1);

    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        free(k_op[i][j]);
      }
      free(k_op[i]);
    }
    free(k_op);

    free(T_KGrids1);
    free(T_KGrids2);
    free(T_KGrids3);
    free(T_k_op);

    for (i = 0; i < 2; i++)
    {
      free(T_k_ID[i]);
    }
    free(T_k_ID);

    for (i = 0; i < spinmax; i++)
    {
      for (j = 0; j < T_knum; j++)
      {
        free(EIGEN_Band[i][j]);
      }
      free(EIGEN_Band[i]);
    }
    free(EIGEN_Band);

    if (CDDF_on == 1)
    {
      for (j = 0; j < (n + 1); j++)
      {
        free(H_Band_Col[j]);
      }
      free(H_Band_Col);
    }

    /***********************************************
        allocation of arrays for the second world
        and
        make the second level worlds
    ***********************************************/

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);

    if (Num_Comm_World2 <= numprocs1)
      MPI_Comm_free(&MPI_CommWD2[myworld2]);

    free(NPROCS_ID2);
    free(Comm_World2);
    free(NPROCS_WD2);
    free(Comm_World_StartID2);
    free(MPI_CommWD2);

    for (spin = 0; spin < spinmax; spin++)
    {
      free(EVec1_Cx[spin]);
    }
    free(EVec1_Cx);

    /***********************************************
      allocation of arrays for the first world
      and
      make the first level worlds
    ***********************************************/

    if (Num_Comm_World1 <= numprocs0)
      MPI_Comm_free(&MPI_CommWD1[myworld1]);

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    free(Ss_Cx);
    free(Hs_Cx);
    free(Cs_Cx);

    Cfree_blacs_system_handle(bhandle2);
    Cblacs_gridexit(ictxt2);
  }
}

void Allocate_Free_Band_NonCol(int todo_flag)
{
  static int firsttime = 1;
  int ZERO = 0, ONE = 1, info, myid0, numprocs0, myid1, numprocs1, myid2, numprocs2;
  ;
  int spinmax, i, j, k, ii, ij, ik, nblk_m, nblk_m2, wanA, spin, size_EVec1_Cx;
  double tmp, tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1, &numprocs0);
  MPI_Comm_rank(mpi_comm_level1, &myid0);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    spinmax = 1;
    Num_Comm_World1 = 1;

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = 2 * n;

    is2 = (int *)malloc(sizeof(int) * numprocs0);
    ie2 = (int *)malloc(sizeof(int) * numprocs0);

    MP = (int *)malloc(sizeof(int) * List_YOUSO[1]);
    order_GA = (int *)malloc(sizeof(int) * (List_YOUSO[1] + 1));
    My_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_Atoms = (int *)malloc(sizeof(int) * numprocs0);

    ko = (double *)malloc(sizeof(double) * (n + 1));
    koS = (double *)malloc(sizeof(double) * (n + 1));
    ko_noncol = (double *)malloc(sizeof(double) * (n2 + 1));

    /* find size_H1 */

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    H1 = (double *)malloc(sizeof(double) * size_H1);
    S1 = (double *)malloc(sizeof(double) * size_H1);

    k_op = (int ***)malloc(sizeof(int **) * Kspace_grid1);
    for (i = 0; i < Kspace_grid1; i++)
    {
      k_op[i] = (int **)malloc(sizeof(int *) * Kspace_grid2);
      for (j = 0; j < Kspace_grid2; j++)
      {
        k_op[i][j] = (int *)malloc(sizeof(int) * Kspace_grid3);
      }
    }

    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        for (k = 0; k < Kspace_grid3; k++)
        {
          k_op[i][j][k] = -999;
        }
      }
    }

    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        for (k = 0; k < Kspace_grid3; k++)
        {

          if (k_op[i][j][k] == -999)
          {

            k_inversion(i, j, k, Kspace_grid1, Kspace_grid2, Kspace_grid3, &ii, &ij, &ik);

            if (i == ii && j == ij && k == ik)
            {
              k_op[i][j][k] = 1;
            }
            else
            {
              k_op[i][j][k] = 1;
              k_op[ii][ij][ik] = 1;
            }
          }
        } /* k */
      }   /* j */
    }     /* i */

    /* find T_knum */

    T_knum = 0;
    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        for (k = 0; k < Kspace_grid3; k++)
        {
          if (0 < k_op[i][j][k])
          {
            T_knum++;
          }
        }
      }
    }

    T_KGrids1 = (double *)malloc(sizeof(double) * T_knum);
    T_KGrids2 = (double *)malloc(sizeof(double) * T_knum);
    T_KGrids3 = (double *)malloc(sizeof(double) * T_knum);
    T_k_op = (int *)malloc(sizeof(int) * T_knum);

    T_k_ID = (int **)malloc(sizeof(int *) * 2);
    for (i = 0; i < 2; i++)
    {
      T_k_ID[i] = (int *)malloc(sizeof(int) * T_knum);
    }

    EIGEN_Band = (double ***)malloc(sizeof(double **) * spinmax);
    for (i = 0; i < spinmax; i++)
    {
      EIGEN_Band[i] = (double **)malloc(sizeof(double *) * T_knum);
      for (j = 0; j < T_knum; j++)
      {
        EIGEN_Band[i][j] = (double *)malloc(sizeof(double) * (n2 + 1));
        for (k = 0; k < (n + 1); k++)
          EIGEN_Band[i][j][k] = 1.0e+5;
      }
    }

    if (CDDF_on == 1)
    {
      S_Band = (dcomplex **)malloc(sizeof(dcomplex *) * (n + 1));
      for (i = 0; i < n + 1; i++)
      {
        S_Band[i] = (dcomplex *)malloc(sizeof(dcomplex) * (n + 1));
      }
    }

    /***********************************************
      allocation of arrays for the first world
      and
      make the first level worlds
    ***********************************************/

    NPROCS_ID1 = (int *)malloc(sizeof(int) * numprocs0);
    Comm_World1 = (int *)malloc(sizeof(int) * numprocs0);
    NPROCS_WD1 = (int *)malloc(sizeof(int) * Num_Comm_World1);
    Comm_World_StartID1 = (int *)malloc(sizeof(int) * Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm *)malloc(sizeof(MPI_Comm) * Num_Comm_World1);

    Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1, &myworld1, MPI_CommWD1,
                     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    /***********************************************
        allocation of arrays for the second world
        and
        make the second level worlds
    ***********************************************/

    if (T_knum <= numprocs1)
      Num_Comm_World2 = T_knum;
    else
      Num_Comm_World2 = numprocs1;

    NPROCS_ID2 = (int *)malloc(sizeof(int) * numprocs1);
    Comm_World2 = (int *)malloc(sizeof(int) * numprocs1);
    NPROCS_WD2 = (int *)malloc(sizeof(int) * Num_Comm_World2);
    Comm_World_StartID2 = (int *)malloc(sizeof(int) * Num_Comm_World2);
    MPI_CommWD2 = (MPI_Comm *)malloc(sizeof(MPI_Comm) * Num_Comm_World2);

    Make_Comm_Worlds(MPI_CommWD1[myworld1], myid1, numprocs1, Num_Comm_World2,
                     &myworld2, MPI_CommWD2, NPROCS_ID2, Comm_World2,
                     NPROCS_WD2, Comm_World_StartID2);

    MPI_Comm_size(MPI_CommWD2[myworld2], &numprocs2);
    MPI_Comm_rank(MPI_CommWD2[myworld2], &myid2);

    double av_num;
    int ke, ks, n3;

    n3 = n2;

    av_num = (double)n3 / (double)numprocs2;
    ke = (int)(av_num * (double)(myid2 + 1));
    ks = (int)(av_num * (double)myid2) + 1;
    if (myid1 == 0)
      ks = 1;
    if (myid1 == (numprocs2 - 1))
      ke = n3;
    k = ke - ks + 2;

    EVec1_Cx = (dcomplex **)malloc(sizeof(dcomplex *) * spinmax);
    for (spin = 0; spin < spinmax; spin++)
    {
      EVec1_Cx[spin] = (dcomplex *)malloc(sizeof(dcomplex) * k * n3);
    }
    size_EVec1_Cx = spinmax * k * n3;

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    np_cols = (int)(sqrt((float)numprocs2));
    do
    {
      if ((numprocs2 % np_cols) == 0)
        break;
      np_cols--;
    } while (np_cols >= 2);
    np_rows = numprocs2 / np_cols;

    nblk_m = NBLK;
    while ((nblk_m * np_rows > n || nblk_m * np_cols > n) && (nblk_m > 1))
    {
      nblk_m /= 2;
    }
    if (nblk_m < 1)
      nblk_m = 1;

    MPI_Allreduce(&nblk_m, &nblk, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    my_prow = myid2 / np_cols;
    my_pcol = myid2 % np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols);

    bhandle2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt2 = bhandle2;
    Cblacs_gridinit(&ictxt2, "Row", np_rows, np_cols);

    Ss_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    Hs_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);

    MPI_Allreduce(&na_rows, &na_rows_max, 1, MPI_INT, MPI_MAX, MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols, &na_cols_max, 1, MPI_INT, MPI_MAX, MPI_CommWD2[myworld2]);
    Cs_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows_max * na_cols_max);

    rHs11_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    rHs12_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    rHs22_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    iHs11_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    iHs12_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);
    iHs22_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows * na_cols);

    descinit_(descS, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descH, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);
    descinit_(descC, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt2, &na_rows, &info);

    /* ***************************************************
          setting for BLACS in the matrix size of n2
    *************************************************** */

    int nblk_m2;

    np_cols2 = (int)(sqrt((float)numprocs2));
    do
    {
      if ((numprocs2 % np_cols2) == 0)
        break;
      np_cols2--;
    } while (np_cols2 >= 2);
    np_rows2 = numprocs2 / np_cols2;

    nblk_m2 = NBLK;
    while ((nblk_m2 * np_rows2 > n2 || nblk_m2 * np_cols2 > n2) && (nblk_m2 > 1))
    {
      nblk_m2 /= 2;
    }
    if (nblk_m2 < 1)
      nblk_m2 = 1;

    MPI_Allreduce(&nblk_m2, &nblk2, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    my_prow2 = myid2 / np_cols2;
    my_pcol2 = myid2 % np_cols2;

    na_rows2 = numroc_(&n2, &nblk2, &my_prow2, &ZERO, &np_rows2);
    na_cols2 = numroc_(&n2, &nblk2, &my_pcol2, &ZERO, &np_cols2);

    bhandle1_2 = Csys2blacs_handle(MPI_CommWD2[myworld2]);
    ictxt1_2 = bhandle1_2;

    Cblacs_gridinit(&ictxt1_2, "Row", np_rows2, np_cols2);
    Hs2_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows2 * na_cols2);
    Ss2_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows2 * na_cols2);

    MPI_Allreduce(&na_rows2, &na_rows_max2, 1, MPI_INT, MPI_MAX, MPI_CommWD2[myworld2]);
    MPI_Allreduce(&na_cols2, &na_cols_max2, 1, MPI_INT, MPI_MAX, MPI_CommWD2[myworld2]);
    Cs2_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows_max2 * na_cols_max2);

    descinit_(descH2, &n2, &n2, &nblk2, &nblk2, &ZERO, &ZERO, &ictxt1_2, &na_rows2, &info);
    descinit_(descC2, &n2, &n2, &nblk2, &nblk2, &ZERO, &ZERO, &ictxt1_2, &na_rows2, &info);
    descinit_(descS2, &n2, &n2, &nblk2, &nblk2, &ZERO, &ZERO, &ictxt1_2, &na_rows2, &info);

    /* for generalized Bloch theorem */
    if (GB_switch)
    {
      koSU = (double *)malloc(sizeof(double) * (n + 1));
      koSL = (double *)malloc(sizeof(double) * (n + 1));
      SU_Band = (dcomplex **)malloc(sizeof(dcomplex *) * (n + 1));
      for (i = 0; i < n + 1; i++)
      {
        SU_Band[i] = (dcomplex *)malloc(sizeof(dcomplex) * (n + 1));
      }

      SL_Band = (dcomplex **)malloc(sizeof(dcomplex *) * (n + 1));
      for (i = 0; i < n + 1; i++)
      {
        SL_Band[i] = (dcomplex *)malloc(sizeof(dcomplex) * (n + 1));
      }
    }
    /* -------------------------------------- */

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_Band_NonCol: is2", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ie2", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: MP", sizeof(int) * List_YOUSO[1], NULL);
      PrintMemory("Allocate_Free_Band_NonCol: order_GA", sizeof(int) * (List_YOUSO[1] + 1), NULL);
      PrintMemory("Allocate_Free_Band_NonCol: My_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: SP_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: SP_Atoms", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ko", sizeof(double) * (n + 1), NULL);
      PrintMemory("Allocate_Free_Band_NonCol: koS", sizeof(double) * (n + 1), NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ko_noncol", sizeof(double) * (n2 + 1), NULL);
      PrintMemory("Allocate_Free_Band_NonCol: H1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: S1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: ko_op", sizeof(double) * Kspace_grid1 * Kspace_grid2 * Kspace_grid3, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_KGrids1", sizeof(double) * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_KGrids2", sizeof(double) * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_KGrids3", sizeof(double) * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: T_k_ID", sizeof(int) * 2 * T_knum, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: EIGEN_Band", sizeof(double) * spinmax * T_knum * (n2 + 1), NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_ID1", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World1", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_WD1", sizeof(int) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World_StartID1", sizeof(int) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: MPI_CommWD1", sizeof(MPI_Comm) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_ID2", sizeof(int) * numprocs1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World2", sizeof(int) * numprocs1, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: NPROCS_WD2", sizeof(int) * Num_Comm_World2, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Comm_World_StartID2", sizeof(int) * Num_Comm_World2, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: MPI_CommWD2", sizeof(MPI_Comm) * Num_Comm_World2, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: EVec1_Cx", sizeof(dcomplex) * size_EVec1_Cx, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Ss_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Hs_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Cs_Cx", sizeof(dcomplex) * na_rows_max * na_cols_max, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: rHs11_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: rHs12_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: rHs22_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: iHs11_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: iHs12_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: iHs22_Cx", sizeof(dcomplex) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Hs2_Cx", sizeof(dcomplex) * na_rows2 * na_cols2, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Ss2_Cx", sizeof(dcomplex) * na_rows2 * na_cols2, NULL);
      PrintMemory("Allocate_Free_Band_NonCol: Cs2_Cx", sizeof(dcomplex) * na_rows_max2 * na_cols_max2, NULL);
      if (GB_switch)
      {
        PrintMemory("Allocate_Free_Band_NonCol: koSU", sizeof(double) * (n + 1), NULL);
        PrintMemory("Allocate_Free_Band_NonCol: koSL", sizeof(double) * (n + 1), NULL);
        PrintMemory("Allocate_Free_Band_NonCol: SU_Band", sizeof(double) * (n + 1) * (n + 1), NULL);
        PrintMemory("Allocate_Free_Band_NonCol: SL_Band", sizeof(double) * (n + 1) * (n + 1), NULL);
      }
    }
    firsttime = 0;

  } /* if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    spinmax = 1;
    Num_Comm_World1 = 1;

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }

    free(is2);
    free(ie2);

    free(MP);
    free(order_GA);
    free(My_NZeros);
    free(SP_NZeros);
    free(SP_Atoms);

    free(ko);
    free(koS);
    free(ko_noncol);

    free(H1);
    free(S1);

    for (i = 0; i < Kspace_grid1; i++)
    {
      for (j = 0; j < Kspace_grid2; j++)
      {
        free(k_op[i][j]);
      }
      free(k_op[i]);
    }
    free(k_op);

    free(T_KGrids1);
    free(T_KGrids2);
    free(T_KGrids3);
    free(T_k_op);

    for (i = 0; i < 2; i++)
    {
      free(T_k_ID[i]);
    }
    free(T_k_ID);

    for (i = 0; i < spinmax; i++)
    {
      for (j = 0; j < T_knum; j++)
      {
        free(EIGEN_Band[i][j]);
      }
      free(EIGEN_Band[i]);
    }
    free(EIGEN_Band);

    if (CDDF_on == 1)
    {
      for (i = 0; i < n + 1; i++)
      {
        free(S_Band[i]);
      }
      free(S_Band);
    }

    /* ***************************************************
          setting for BLACS in the matrix size of n
    *************************************************** */

    free(Ss_Cx);
    free(Hs_Cx);
    free(Cs_Cx);
    free(rHs11_Cx);
    free(rHs12_Cx);
    free(rHs22_Cx);
    free(iHs11_Cx);
    free(iHs12_Cx);
    free(iHs22_Cx);

    Cfree_blacs_system_handle(bhandle2);
    Cblacs_gridexit(ictxt2);

    /* ***************************************************
          setting for BLACS in the matrix size of n2
    *************************************************** */

    free(Hs2_Cx);
    free(Ss2_Cx);
    free(Cs2_Cx);

    Cfree_blacs_system_handle(bhandle1_2);
    Cblacs_gridexit(ictxt1_2);

    /***********************************************
        freeing of arrays for the second world
        and
        make the second level worlds
    ***********************************************/

    for (spin = 0; spin < spinmax; spin++)
    {
      free(EVec1_Cx[spin]);
    }
    free(EVec1_Cx);

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    if (Num_Comm_World2 <= numprocs1)
      MPI_Comm_free(&MPI_CommWD2[myworld2]);

    free(MPI_CommWD2);
    free(Comm_World_StartID2);
    free(NPROCS_WD2);
    free(Comm_World2);
    free(NPROCS_ID2);

    /***********************************************
      freeing of arrays for the first world
      and
      make the first level worlds
    ***********************************************/

    if (Num_Comm_World2 <= numprocs1)
      MPI_Comm_free(&MPI_CommWD1[myworld1]);

    free(MPI_CommWD1);
    free(Comm_World_StartID1);
    free(NPROCS_WD1);
    free(Comm_World1);
    free(NPROCS_ID1);

    /* for generalized Bloch theorem */
    if (GB_switch)
    {
      free(koSU);
      free(koSL);
      for (i = 0; i < n + 1; i++)
      {
        free(SU_Band[i]);
      }
      free(SU_Band);

      for (i = 0; i < n + 1; i++)
      {
        free(SL_Band[i]);
      }
      free(SL_Band);
    }
  }
}

void Allocate_Free_Cluster_Col(int todo_flag)
{
  static int firsttime = 1;
  int ZERO = 0, ONE = 1, info, myid0, numprocs0, myid1, numprocs1;
  int i, k, nblk_m, nblk_m2, wanA, spin, size_EVec1;
  double tmp, tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1, &numprocs0);
  MPI_Comm_rank(mpi_comm_level1, &myid0);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = n * 2;

    MP = (int *)malloc(sizeof(int) * List_YOUSO[1]);
    order_GA = (int *)malloc(sizeof(int) * (List_YOUSO[1] + 1));
    My_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_Atoms = (int *)malloc(sizeof(int) * numprocs0);

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    CDM1 = (double *)malloc(sizeof(double) * size_H1);
    Work1 = (double *)malloc(sizeof(double) * size_H1);
    EDM1 = (double *)malloc(sizeof(double) * size_H1);

    if (cal_partial_charge)
    {
      PDM1 = (double *)malloc(sizeof(double) * size_H1);
    }

    ko_col = (double **)malloc(sizeof(double *) * List_YOUSO[23]);
    for (i = 0; i < List_YOUSO[23]; i++)
    {
      ko_col[i] = (double *)malloc(sizeof(double) * (n + 2));
    }

    /***********************************************
        allocation of arrays for the first world
    ***********************************************/

    Num_Comm_World1 = SpinP_switch + 1;

    NPROCS_ID1 = (int *)malloc(sizeof(int) * numprocs0);
    Comm_World1 = (int *)malloc(sizeof(int) * numprocs0);
    NPROCS_WD1 = (int *)malloc(sizeof(int) * Num_Comm_World1);
    Comm_World_StartID1 = (int *)malloc(sizeof(int) * Num_Comm_World1);
    MPI_CommWD1 = (MPI_Comm *)malloc(sizeof(MPI_Comm) * Num_Comm_World1);

    /***********************************************
             make the first level worlds
    ***********************************************/

    Make_Comm_Worlds(mpi_comm_level1, myid0, numprocs0, Num_Comm_World1,
                     &myworld1, MPI_CommWD1, NPROCS_ID1, Comm_World1,
                     NPROCS_WD1, Comm_World_StartID1);

    /***********************************************
         for pallalel calculations in myworld1
    ***********************************************/

    MPI_Comm_size(MPI_CommWD1[myworld1], &numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1], &myid1);

    double av_num;
    int ke, ks;
    av_num = (double)n / (double)numprocs1;
    ke = (int)(av_num * (double)(myid1 + 1));
    ks = (int)(av_num * (double)myid1) + 1;
    if (myid1 == 0)
      ks = 1;
    if (myid1 == (numprocs1 - 1))
      ke = n;
    k = ke - ks + 2;

    EVec1_Re = (double **)malloc(sizeof(double *) * (SpinP_switch + 1));
    for (spin = 0; spin < (SpinP_switch + 1); spin++)
    {
      EVec1_Re[spin] = (double *)malloc(sizeof(double) * k * n);
    }
    size_EVec1 = (SpinP_switch + 1) * k * n;

    is2 = (int *)malloc(sizeof(int) * numprocs1);
    ie2 = (int *)malloc(sizeof(int) * numprocs1);

    /* setting for BLACS */

    np_cols = (int)(sqrt((float)numprocs1));
    do
    {
      if ((numprocs1 % np_cols) == 0)
        break;
      np_cols--;
    } while (np_cols >= 2);
    np_rows = numprocs1 / np_cols;

    nblk_m = NBLK;
    while ((nblk_m * np_rows > n || nblk_m * np_cols > n) && (nblk_m > 1))
    {
      nblk_m /= 2;
    }
    if (nblk_m < 1)
      nblk_m = 1;

    MPI_Allreduce(&nblk_m, &nblk, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    my_prow = myid1 / np_cols;
    my_pcol = myid1 % np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols);

    bhandle1 = Csys2blacs_handle(MPI_CommWD1[myworld1]);
    ictxt1 = bhandle1;

    Cblacs_gridinit(&ictxt1, "Row", np_rows, np_cols);
    Ss_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    Hs_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);

    MPI_Allreduce(&na_rows, &na_rows_max, 1, MPI_INT, MPI_MAX, MPI_CommWD1[myworld1]);
    MPI_Allreduce(&na_cols, &na_cols_max, 1, MPI_INT, MPI_MAX, MPI_CommWD1[myworld1]);
    Cs_Re = (double *)malloc(sizeof(double) * na_rows_max * na_cols_max);

    descinit_(descS, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt1, &na_rows, &info);
    descinit_(descH, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt1, &na_rows, &info);
    descinit_(descC, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt1, &na_rows, &info);

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_Cluster_Col: MP", sizeof(int) * List_YOUSO[1], NULL);
      PrintMemory("Allocate_Free_Cluster_Col: order_GA", sizeof(int) * (List_YOUSO[1] + 1), NULL);
      PrintMemory("Allocate_Free_Cluster_Col: My_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: SP_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: SP_Atoms", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: CDM1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Work1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: EDM1", sizeof(double) * size_H1, NULL);
      if (cal_partial_charge)
      {
        PrintMemory("Allocate_Free_Cluster_Col: PDM1", sizeof(double) * size_H1, NULL);
      }

      PrintMemory("Allocate_Free_Cluster_Col: ko_col", sizeof(double) * List_YOUSO[23] * (n + 2), NULL);
      PrintMemory("Allocate_Free_Cluster_Col: NPROCS_ID1", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Comm_World1", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: NPROCS_WD1", sizeof(int) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Comm_World_StartID1", sizeof(int) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: MPI_CommWD1", sizeof(MPI_Comm) * Num_Comm_World1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: EVec1_Re", sizeof(dcomplex) * size_EVec1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: is2", sizeof(int) * numprocs1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: ie2", sizeof(int) * numprocs1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Ss_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Hs_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Cs_Re", sizeof(double) * na_rows_max * na_cols_max, NULL);
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    free(MP);
    free(order_GA);
    free(My_NZeros);
    free(SP_NZeros);
    free(SP_Atoms);
    free(CDM1);
    free(Work1);
    free(EDM1);

    if (cal_partial_charge)
    {
      free(PDM1);
    }

    for (i = 0; i < List_YOUSO[23]; i++)
    {
      free(ko_col[i]);
    }
    free(ko_col);

    /***********************************************
       freeing of arrays for the first world
    ***********************************************/

    if (Num_Comm_World1 <= numprocs0)
      MPI_Comm_free(&MPI_CommWD1[myworld1]);

    free(NPROCS_ID1);
    free(Comm_World1);
    free(NPROCS_WD1);
    free(Comm_World_StartID1);
    free(MPI_CommWD1);

    /***********************************************
         for pallalel calculations in myworld1
    ***********************************************/

    for (spin = 0; spin < (SpinP_switch + 1); spin++)
    {
      free(EVec1_Re[spin]);
    }
    free(EVec1_Re);

    free(is2);
    free(ie2);

    /* setting for BLACS */

    free(Ss_Re);
    free(Hs_Re);
    free(Cs_Re);

    Cfree_blacs_system_handle(bhandle1);
    Cblacs_gridexit(ictxt1);
  }
}

void Allocate_Free_Cluster_NonCol(int todo_flag)
{
  static int firsttime = 1;
  int ZERO = 0, ONE = 1, info, size_EVec1;
  int i, k, nblk_m, nblk_m2, wanA, myid0, numprocs0, numprocs1;
  double tmp, tmp1;

  MPI_Barrier(mpi_comm_level1);
  MPI_Comm_size(mpi_comm_level1, &numprocs0);
  MPI_Comm_rank(mpi_comm_level1, &myid0);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }
    n2 = n * 2;

    is2 = (int *)malloc(sizeof(int) * numprocs0);
    ie2 = (int *)malloc(sizeof(int) * numprocs0);

    MP = (int *)malloc(sizeof(int) * List_YOUSO[1]);
    order_GA = (int *)malloc(sizeof(int) * (List_YOUSO[1] + 1));
    My_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_NZeros = (int *)malloc(sizeof(int) * numprocs0);
    SP_Atoms = (int *)malloc(sizeof(int) * numprocs0);
    ko_noncol = (double *)malloc(sizeof(double) * (n + 1) * 2);

    size_H1 = Get_OneD_HS_Col(0, H[0], &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

    CDM1 = (double *)malloc(sizeof(double) * size_H1);
    Work1 = (double *)malloc(sizeof(double) * size_H1);

    double av_num;
    int ke, ks;
    av_num = (double)(n * 2) / (double)numprocs0;
    ke = (int)(av_num * (double)(myid0 + 1));
    ks = (int)(av_num * (double)myid0) + 1;
    if (myid0 == 0)
      ks = 1;
    if (myid0 == (numprocs0 - 1))
      ke = n * 2;
    k = ke - ks + 2;
    EVec1_NonCol = (dcomplex *)malloc(sizeof(dcomplex) * k * n * 2);
    size_EVec1 = k * n * 2;

    /* ***************************************************
       setting for BLACS in the matrix size of n
       *************************************************** */

    np_cols = (int)(sqrt((float)numprocs0));
    do
    {
      if ((numprocs0 % np_cols) == 0)
        break;
      np_cols--;
    } while (np_cols >= 2);
    np_rows = numprocs0 / np_cols;

    nblk_m = NBLK;
    while ((nblk_m * np_rows > n || nblk_m * np_cols > n) && (nblk_m > 1))
    {
      nblk_m /= 2;
    }
    if (nblk_m < 1)
      nblk_m = 1;

    MPI_Allreduce(&nblk_m, &nblk, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    my_prow = myid0 / np_cols;
    my_pcol = myid0 % np_cols;

    na_rows = numroc_(&n, &nblk, &my_prow, &ZERO, &np_rows);
    na_cols = numroc_(&n, &nblk, &my_pcol, &ZERO, &np_cols);

    bhandle1 = Csys2blacs_handle(mpi_comm_level1);
    ictxt1 = bhandle1;

    Cblacs_gridinit(&ictxt1, "Row", np_rows, np_cols);
    Ss_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    rHs11_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    rHs12_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    rHs22_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    iHs11_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    iHs12_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);
    iHs22_Re = (double *)malloc(sizeof(double) * na_rows * na_cols);

    MPI_Allreduce(&na_rows, &na_rows_max, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    MPI_Allreduce(&na_cols, &na_cols_max, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    Cs_Re = (double *)malloc(sizeof(double) * na_rows_max * na_cols_max);

    descinit_(descS, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt1, &na_rows, &info);
    descinit_(descH, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt1, &na_rows, &info);
    descinit_(descC, &n, &n, &nblk, &nblk, &ZERO, &ZERO, &ictxt1, &na_rows, &info);

    /* ***************************************************
       setting for BLACS in the matrix size of n2
       *************************************************** */

    np_cols2 = (int)(sqrt((float)numprocs0));
    do
    {
      if ((numprocs0 % np_cols2) == 0)
        break;
      np_cols2--;
    } while (np_cols2 >= 2);
    np_rows2 = numprocs0 / np_cols2;

    nblk_m2 = NBLK;
    while ((nblk_m2 * np_rows2 > n2 || nblk_m2 * np_cols2 > n2) && (nblk_m2 > 1))
    {
      nblk_m2 /= 2;
    }
    if (nblk_m2 < 1)
      nblk_m2 = 1;

    MPI_Allreduce(&nblk_m2, &nblk2, 1, MPI_INT, MPI_MIN, mpi_comm_level1);

    my_prow2 = myid0 / np_cols2;
    my_pcol2 = myid0 % np_cols2;

    na_rows2 = numroc_(&n2, &nblk2, &my_prow2, &ZERO, &np_rows2);
    na_cols2 = numroc_(&n2, &nblk2, &my_pcol2, &ZERO, &np_cols2);

    bhandle1_2 = Csys2blacs_handle(mpi_comm_level1);
    ictxt1_2 = bhandle1_2;

    Cblacs_gridinit(&ictxt1_2, "Row", np_rows2, np_cols2);
    Hs2_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows2 * na_cols2);
    Ss2_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows2 * na_cols2);

    MPI_Allreduce(&na_rows2, &na_rows_max2, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    MPI_Allreduce(&na_cols2, &na_cols_max2, 1, MPI_INT, MPI_MAX, mpi_comm_level1);
    Cs2_Cx = (dcomplex *)malloc(sizeof(dcomplex) * na_rows_max2 * na_cols_max2);

    descinit_(descH2, &n2, &n2, &nblk2, &nblk2, &ZERO, &ZERO, &ictxt1_2, &na_rows2, &info);
    descinit_(descC2, &n2, &n2, &nblk2, &nblk2, &ZERO, &ZERO, &ictxt1_2, &na_rows2, &info);
    descinit_(descS2, &n2, &n2, &nblk2, &nblk2, &ZERO, &ZERO, &ictxt1_2, &na_rows2, &info);

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_Cluster_Col: is2", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: ie2", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: MP", sizeof(int) * List_YOUSO[1], NULL);
      PrintMemory("Allocate_Free_Cluster_Col: order_GA", sizeof(int) * (List_YOUSO[1] + 1), NULL);
      PrintMemory("Allocate_Free_Cluster_Col: My_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: SP_NZeros", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: SP_Atoms", sizeof(int) * numprocs0, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: ko_col", sizeof(double) * (n + 1) * 2, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: CDM1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Work1", sizeof(double) * size_H1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: EVec1_NonCol", sizeof(dcomplex) * size_EVec1, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Ss_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: rHs11_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: rHs12_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: rHs22_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: iHs11_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: iHs12_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: iHs22_Re", sizeof(double) * na_rows * na_cols, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Cs_Re", sizeof(double) * na_rows_max * na_cols_max, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Hs2_Cx", sizeof(double) * na_rows2 * na_cols2, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Ss2_Cx", sizeof(double) * na_rows2 * na_cols2, NULL);
      PrintMemory("Allocate_Free_Cluster_Col: Cs2_Cx", sizeof(double) * na_rows_max2 * na_cols_max2, NULL);
    }
    firsttime = 0;

  } /* if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    free(is2);
    free(ie2);
    free(MP);
    free(order_GA);
    free(My_NZeros);
    free(SP_NZeros);
    free(SP_Atoms);
    free(ko_noncol);
    free(CDM1);
    free(Work1);
    free(EVec1_NonCol);

    /* ***************************************************
       setting for BLACS in the matrix size of n
       *************************************************** */

    free(Ss_Re);
    free(rHs11_Re);
    free(rHs12_Re);
    free(rHs22_Re);
    free(iHs11_Re);
    free(iHs12_Re);
    free(iHs22_Re);
    free(Cs_Re);

    Cfree_blacs_system_handle(bhandle1);
    Cblacs_gridexit(ictxt1);

    /* ***************************************************
       setting for BLACS in the matrix size of n2
       *************************************************** */

    free(Hs2_Cx);
    free(Ss2_Cx);
    free(Cs2_Cx);

    Cfree_blacs_system_handle(bhandle1_2);
    Cblacs_gridexit(ictxt1_2);
  }
}

void Allocate_Free_NEGF_NonCol(int todo_flag)
{
  static int firsttime = 1;
  int i, n, wanA;

  MPI_Barrier(mpi_comm_level1);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }

    koS = (double *)malloc(sizeof(double) * (n + 1));

    S_Band = (dcomplex **)malloc(sizeof(dcomplex *) * (n + 1));
    for (i = 0; i < n + 1; i++)
    {
      S_Band[i] = (dcomplex *)malloc(sizeof(dcomplex) * (n + 1));
    }

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_NEGF_NonCol: koS", sizeof(int) * (n + 1), NULL);
      PrintMemory("Allocate_Free_NEGF_NonCol: ie2", sizeof(int) * (n + 1) * (n + 1), NULL);
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    n = 0;
    for (i = 1; i <= atomnum; i++)
    {
      wanA = WhatSpecies[i];
      n += Spe_Total_CNO[wanA];
    }

    free(koS);

    for (i = 0; i < n + 1; i++)
    {
      free(S_Band[i]);
    }
    free(S_Band);
  }
}

void Allocate_Free_GridData(int todo_flag)
{
  static int firsttime = 1;
  int i, m, spinmax, spin;

  MPI_Barrier(mpi_comm_level1);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    ReVk = (double *)malloc(sizeof(double) * My_Max_NumGridB);
    for (i = 0; i < My_Max_NumGridB; i++)
      ReVk[i] = 0.0;

    ImVk = (double *)malloc(sizeof(double) * My_Max_NumGridB);
    for (i = 0; i < My_Max_NumGridB; i++)
      ImVk[i] = 0.0;

    if (Mixing_switch == 3 || Mixing_switch == 4)
    {

      if (SpinP_switch == 0)
        spinmax = 1;
      else if (SpinP_switch == 1)
        spinmax = 2;
      else if (SpinP_switch == 3)
        spinmax = 3;

      ReRhoAtomk = (double *)malloc(sizeof(double) * My_Max_NumGridB);
      for (i = 0; i < My_Max_NumGridB; i++)
        ReRhoAtomk[i] = 0.0;

      ImRhoAtomk = (double *)malloc(sizeof(double) * My_Max_NumGridB);
      for (i = 0; i < My_Max_NumGridB; i++)
        ImRhoAtomk[i] = 0.0;

      ReRhok = (double ***)malloc(sizeof(double **) * List_YOUSO[38]);
      for (m = 0; m < List_YOUSO[38]; m++)
      {
        ReRhok[m] = (double **)malloc(sizeof(double *) * spinmax);
        for (spin = 0; spin < spinmax; spin++)
        {
          ReRhok[m][spin] = (double *)malloc(sizeof(double) * My_Max_NumGridB);
          for (i = 0; i < My_Max_NumGridB; i++)
            ReRhok[m][spin][i] = 0.0;
        }
      }

      ImRhok = (double ***)malloc(sizeof(double **) * List_YOUSO[38]);
      for (m = 0; m < List_YOUSO[38]; m++)
      {
        ImRhok[m] = (double **)malloc(sizeof(double *) * spinmax);
        for (spin = 0; spin < spinmax; spin++)
        {
          ImRhok[m][spin] = (double *)malloc(sizeof(double) * My_Max_NumGridB);
          for (i = 0; i < My_Max_NumGridB; i++)
            ImRhok[m][spin][i] = 0.0;
        }
      }

      Residual_ReRhok = (double **)malloc(sizeof(double *) * spinmax);
      for (spin = 0; spin < spinmax; spin++)
      {
        Residual_ReRhok[spin] = (double *)malloc(sizeof(double) * My_NumGridB_CB * List_YOUSO[38]);
        for (i = 0; i < My_NumGridB_CB * List_YOUSO[38]; i++)
          Residual_ReRhok[spin][i] = 0.0;
      }

      Residual_ImRhok = (double **)malloc(sizeof(double *) * spinmax);
      for (spin = 0; spin < spinmax; spin++)
      {
        Residual_ImRhok[spin] = (double *)malloc(sizeof(double) * My_NumGridB_CB * List_YOUSO[38]);
        for (i = 0; i < My_NumGridB_CB * List_YOUSO[38]; i++)
          Residual_ImRhok[spin][i] = 0.0;
      }
    }

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_GridData: ReVk", sizeof(double) * My_Max_NumGridB, NULL);
      PrintMemory("Allocate_Free_GridData: ImVk", sizeof(double) * My_Max_NumGridB, NULL);
      PrintMemory("Allocate_Free_GridData: ReRhoAtomk", sizeof(double) * My_Max_NumGridB, NULL);
      PrintMemory("Allocate_Free_GridData: ImRhoAtomk", sizeof(double) * My_Max_NumGridB, NULL);
      PrintMemory("Allocate_Free_GridData: ReRhok", sizeof(double) * List_YOUSO[38] * spinmax * My_Max_NumGridB, NULL);
      PrintMemory("Allocate_Free_GridData: ImRhok", sizeof(double) * List_YOUSO[38] * spinmax * My_Max_NumGridB, NULL);
      PrintMemory("Allocate_Free_GridData: Residual_ReRhok", sizeof(double) * List_YOUSO[38] * spinmax * My_NumGridB_CB, NULL);
      PrintMemory("Allocate_Free_GridData: Residual_ImRhok", sizeof(double) * List_YOUSO[38] * spinmax * My_NumGridB_CB, NULL);
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    free(ReVk);
    free(ImVk);

    if (Mixing_switch == 3 || Mixing_switch == 4)
    {

      if (SpinP_switch == 0)
        spinmax = 1;
      else if (SpinP_switch == 1)
        spinmax = 2;
      else if (SpinP_switch == 3)
        spinmax = 3;

      free(ReRhoAtomk);
      free(ImRhoAtomk);

      for (m = 0; m < List_YOUSO[38]; m++)
      {
        for (spin = 0; spin < spinmax; spin++)
        {
          free(ReRhok[m][spin]);
        }
        free(ReRhok[m]);
      }
      free(ReRhok);

      for (m = 0; m < List_YOUSO[38]; m++)
      {
        for (spin = 0; spin < spinmax; spin++)
        {
          free(ImRhok[m][spin]);
        }
        free(ImRhok[m]);
      }
      free(ImRhok);

      for (spin = 0; spin < spinmax; spin++)
      {
        free(Residual_ReRhok[spin]);
      }
      free(Residual_ReRhok);

      for (spin = 0; spin < spinmax; spin++)
      {
        free(Residual_ImRhok[spin]);
      }
      free(Residual_ImRhok);
    }
  }
}

void Allocate_Free_OrbOpt(int todo_flag)
{
  static int firsttime = 1;
  int i, j, k, p, al, wan, dim_H;

  MPI_Barrier(mpi_comm_level1);

  /********************************************
              allocation of arrays
  ********************************************/

  if (todo_flag == 1)
  {

    His_CntCoes = (double ****)malloc(sizeof(double ***) * (orbitalOpt_History + 1));
    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      His_CntCoes[k] = (double ***)malloc(sizeof(double **) * (Matomnum + 1));
      for (i = 0; i <= Matomnum; i++)
      {
        His_CntCoes[k][i] = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          His_CntCoes[k][i][j] = (double *)malloc(sizeof(double) * List_YOUSO[24]);
        }
      }
    }

    His_D_CntCoes = (double ****)malloc(sizeof(double ***) * (orbitalOpt_History + 1));
    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      His_D_CntCoes[k] = (double ***)malloc(sizeof(double **) * (Matomnum + 1));
      for (i = 0; i <= Matomnum; i++)
      {
        His_D_CntCoes[k][i] = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          His_D_CntCoes[k][i][j] = (double *)malloc(sizeof(double) * List_YOUSO[24]);
        }
      }
    }

    His_CntCoes_Species = (double ****)malloc(sizeof(double ***) * (orbitalOpt_History + 1));
    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      His_CntCoes_Species[k] = (double ***)malloc(sizeof(double **) * (SpeciesNum + 1));
      for (i = 0; i <= SpeciesNum; i++)
      {
        His_CntCoes_Species[k][i] = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          His_CntCoes_Species[k][i][j] = (double *)malloc(sizeof(double) * List_YOUSO[24]);
        }
      }
    }

    His_D_CntCoes_Species = (double ****)malloc(sizeof(double ***) * (orbitalOpt_History + 1));
    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      His_D_CntCoes_Species[k] = (double ***)malloc(sizeof(double **) * (SpeciesNum + 1));
      for (i = 0; i <= SpeciesNum; i++)
      {
        His_D_CntCoes_Species[k][i] = (double **)malloc(sizeof(double *) * List_YOUSO[7]);
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          His_D_CntCoes_Species[k][i][j] = (double *)malloc(sizeof(double) * List_YOUSO[24]);
        }
      }
    }

    dim_H = 0;
    for (wan = 0; wan < SpeciesNum; wan++)
    {
      for (al = 0; al < Spe_Total_CNO[wan]; al++)
      {
        for (p = 0; p < Spe_Specified_Num[wan][al]; p++)
        {
          dim_H++;
        }
      }
    }

    OrbOpt_Hessian = (double **)malloc(sizeof(double *) * (dim_H + 2));
    for (i = 0; i < (dim_H + 2); i++)
    {
      OrbOpt_Hessian[i] = (double *)malloc(sizeof(double) * (dim_H + 2));
    }

    His_OrbOpt_Etot = (double *)malloc(sizeof(double) * (orbitalOpt_History + 2));

    /* PrintMemory */

    if (firsttime && memoryusage_fileout)
    {
      PrintMemory("Allocate_Free_OrbOpt: His_CntCoes", sizeof(double) * (orbitalOpt_History + 1) * (Matomnum + 1) * List_YOUSO[7] * List_YOUSO[24], NULL);
      PrintMemory("Allocate_Free_OrbOpt: His_D_CntCoes", sizeof(double) * (orbitalOpt_History + 1) * (Matomnum + 1) * List_YOUSO[7] * List_YOUSO[24], NULL);
      PrintMemory("Allocate_Free_OrbOpt: His_CntCoes_Species", sizeof(double) * (orbitalOpt_History + 1) * (Matomnum + 1) * List_YOUSO[7] * List_YOUSO[24], NULL);
      PrintMemory("Allocate_Free_OrbOpt: His_D_CntCoes_Species", sizeof(double) * (orbitalOpt_History + 1) * (Matomnum + 1) * List_YOUSO[7] * List_YOUSO[24], NULL);
      PrintMemory("Allocate_Free_OrbOpt: OrbOpt_Hessian", sizeof(double) * (dim_H + 2) * (dim_H + 2), NULL);
      PrintMemory("Allocate_Free_OrbOpt: His_OrbOpt_Etot", sizeof(double) * (orbitalOpt_History + 2), NULL);
    }
    firsttime = 0;

  } /* end of if (todo_flag==1) */

  /********************************************
               freeing of arrays
  ********************************************/

  if (todo_flag == 2)
  {

    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      for (i = 0; i <= Matomnum; i++)
      {
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          free(His_CntCoes[k][i][j]);
        }
        free(His_CntCoes[k][i]);
      }
      free(His_CntCoes[k]);
    }
    free(His_CntCoes);

    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      for (i = 0; i <= Matomnum; i++)
      {
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          free(His_D_CntCoes[k][i][j]);
        }
        free(His_D_CntCoes[k][i]);
      }
      free(His_D_CntCoes[k]);
    }
    free(His_D_CntCoes);

    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      for (i = 0; i <= SpeciesNum; i++)
      {
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          free(His_CntCoes_Species[k][i][j]);
        }
        free(His_CntCoes_Species[k][i]);
      }
      free(His_CntCoes_Species[k]);
    }
    free(His_CntCoes_Species);

    for (k = 0; k < (orbitalOpt_History + 1); k++)
    {
      for (i = 0; i <= SpeciesNum; i++)
      {
        for (j = 0; j < List_YOUSO[7]; j++)
        {
          free(His_D_CntCoes_Species[k][i][j]);
        }
        free(His_D_CntCoes_Species[k][i]);
      }
      free(His_D_CntCoes_Species[k]);
    }
    free(His_D_CntCoes_Species);

    dim_H = 0;
    for (wan = 0; wan < SpeciesNum; wan++)
    {
      for (al = 0; al < Spe_Total_CNO[wan]; al++)
      {
        for (p = 0; p < Spe_Specified_Num[wan][al]; p++)
        {
          dim_H++;
        }
      }
    }

    for (i = 0; i < (dim_H + 2); i++)
    {
      free(OrbOpt_Hessian[i]);
    }
    free(OrbOpt_Hessian);

    free(His_OrbOpt_Etot);
  }
}
