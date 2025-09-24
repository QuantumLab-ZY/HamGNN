/**********************************************************************
TRAN_Channel_Output.c:

Routines used in MTRAN_EigenChannel and TRAN_Main_Analysis
to output eigenchanels to files.

Log of TRAN_Channel_Output.c:

xx/Xxx/2015  Released by M. Kawamura

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "openmx_common.h"

static void Print_CubeTitle_EigenChannel(
  FILE *fp, 
  double *TRAN_Channel_kpoint, 
  double TRAN_Channel_energy, 
  double eigentrans,
  int ispin)
{
  int ct_AN;
  int spe;

  fprintf(fp, " Transmission of this channel : %10.6f \n", eigentrans);
  fprintf(fp,
    " E - E_{F Left} [eV] : %10.6f,  k(Frac.) = %10.6f %10.6f,  ", 
    TRAN_Channel_energy * eV2Hartree, TRAN_Channel_kpoint[0], TRAN_Channel_kpoint[1]);
 
  if (SpinP_switch < 2){
    if (ispin == 0)  fprintf(fp, "Spin = Up \n");
    else             fprintf(fp, "Spin = Down \n");
  }
  else{
    fprintf(fp, "Spin = Non-colinear \n");
  }

  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    atomnum, Grid_Origin[1], Grid_Origin[2], Grid_Origin[3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    Ngrid1, gtv[1][1], gtv[1][2], gtv[1][3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    Ngrid2, gtv[2][1], gtv[2][2], gtv[2][3]);
  fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf\n",
    Ngrid3, gtv[3][1], gtv[3][2], gtv[3][3]);

  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++){
    spe = WhatSpecies[ct_AN];
    fprintf(fp, "%5d%12.6lf%12.6lf%12.6lf%12.6lf\n",
      Spe_WhatAtom[spe],
      Spe_Core_Charge[spe] - InitN_USpin[ct_AN] - InitN_DSpin[ct_AN],
      Gxyz[ct_AN][1], Gxyz[ct_AN][2], Gxyz[ct_AN][3]);
  } /* for (ct_AN = 1; ct_AN <= atomnum; ct_AN++) */

} /* static void Print_CubeTitle_EigenChannel */

static void Print_CubeTitle_EigenChannel_Bin(
  FILE *fp,
  double *TRAN_Channel_kpoint,
  double TRAN_Channel_energy,
  double eigentrans,
  int ispin)
{
  int ct_AN, n;
  int spe;
  char clist[300];
  double *dlist;

  sprintf(clist, " Transmission of this channel : %10.6f \n", eigentrans);
  fwrite(clist, sizeof(char), 200, fp);

  if (SpinP_switch < 2){
    if (ispin == 0)
      sprintf(clist,
      " E - E_{F Left} [eV] : %10.6f,  k(Frac.) = %10.6f %10.6f,  Spin = Up \n",
      TRAN_Channel_energy * eV2Hartree, TRAN_Channel_kpoint[0], TRAN_Channel_kpoint[1]);
    else
      sprintf(clist,
      " E - E_{F Left} [eV] : %10.6f,  k(Frac.) = %10.6f %10.6f,  Spin = Down \n",
      TRAN_Channel_energy * eV2Hartree, TRAN_Channel_kpoint[0], TRAN_Channel_kpoint[1]);
  }
  else{
    sprintf(clist,
      " E - E_{F Left} [eV] : %10.6f,  k(Frac.) = %10.6f %10.6f,  Spin = Non-colinear \n",
      TRAN_Channel_energy * eV2Hartree, TRAN_Channel_kpoint[0], TRAN_Channel_kpoint[1]);
  }
  fwrite(clist, sizeof(char), 200, fp);

  fwrite(&atomnum, sizeof(int), 1, fp);
  fwrite(&Grid_Origin[1], sizeof(double), 3, fp);
  fwrite(&Ngrid1, sizeof(int), 1, fp);
  fwrite(&gtv[1][1], sizeof(double), 3, fp);
  fwrite(&Ngrid2, sizeof(int), 1, fp);
  fwrite(&gtv[2][1], sizeof(double), 3, fp);
  fwrite(&Ngrid3, sizeof(int), 1, fp);
  fwrite(&gtv[3][1], sizeof(double), 3, fp);

  dlist = (double*)malloc(sizeof(double)*atomnum * 5);

  n = 0;
  for (ct_AN = 1; ct_AN <= atomnum; ct_AN++){
    spe = WhatSpecies[ct_AN];
    dlist[n] = (double)Spe_WhatAtom[spe];                                  n++;
    dlist[n] = Spe_Core_Charge[spe] - InitN_USpin[ct_AN] - InitN_DSpin[ct_AN]; n++;
    dlist[n] = Gxyz[ct_AN][1];                                             n++;
    dlist[n] = Gxyz[ct_AN][2];                                             n++;
    dlist[n] = Gxyz[ct_AN][3];                                             n++;
  }

  fwrite(dlist, sizeof(double), atomnum * 5, fp);
  free(dlist);
}/* static void Print_CubeTitle_Bin */

static void Print_CubeCData_MO(
  FILE *fp, 
  dcomplex *data, 
  char *op)
{
  int i1, i2, i3;
  int GN;
  int cmd;

  if (strcmp(op, "r") == 0) { cmd = 1; }
  else if (strcmp(op, "i") == 0) { cmd = 2; }
  else {
    printf("Print_CubeCData: op=%s not supported\n", op);
    return;
  }

  for (i1 = 0; i1<Ngrid1; i1++){
    for (i2 = 0; i2<Ngrid2; i2++){
      for (i3 = 0; i3<Ngrid3; i3++){
        GN = i1*Ngrid2*Ngrid3 + i2*Ngrid3 + i3;
        switch (cmd) {
        case 1:
          fprintf(fp, "%13.3E", data[GN].r);
          break;
        case 2:
          fprintf(fp, "%13.3E", data[GN].i);
          break;
        }
        if ((i3 + 1) % 6 == 0) { fprintf(fp, "\n"); }
      }
      /* avoid double \n\n when Ngrid3%6 == 0  */
      if (Ngrid3 % 6 != 0) fprintf(fp, "\n");
    }
  }
} /* static void Print_CubeCData_MO */

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
  double **eigentrans_sum)
#define GC_R_ref(i,j) GC_R[NUM_c*((j)-1)+(i)-1]
{
  int i, j, l;
  int num0, num1, mul, m, wan1, Gc_AN;
  int i1, j1;
  char *Name_Angular[Supported_MaxL + 1][2 * (Supported_MaxL + 1) + 1];
  char *Name_Multiple[20];
  FILE *fp_EV;
  char fname[100];
  double sumeval;

  /*
    Transmission of each channel
  */

  sprintf(fname, "%s%s.traneval%i_%i_%i", filepath, filename, kloop, iw, ispin);

  eigentrans_sum[iw][ispin] = 0.0;
  for (i = 0; i < NUM_c; i++){
    eigentrans_sum[iw][ispin] += eval[i];
  }

  if ((fp_EV = fopen(fname, "w")) == NULL) {
    printf("\ncannot open file to write EigenTransmisson \n"); fflush(stdout);
    exit(0);
  } /* if ((fp_EV = fopen(fname, "w")) == NULL) */

  fprintf(fp_EV, "#\n");
  fprintf(fp_EV, "# Eigen transmission of each channel \n");
  fprintf(fp_EV, "#\n");
  fprintf(fp_EV, "#   Index of k-point = %i\n", kloop);
  fprintf(fp_EV, "#   k2=%10.5f k3=%10.5f\n#\n", TRAN_Channel_kpoint[0], TRAN_Channel_kpoint[1]);
  fprintf(fp_EV, "#   Index of Energy = %i\n", iw);
  fprintf(fp_EV, "#   e=%10.5f \n#\n", TRAN_Channel_energy * eV2Hartree);

  if (SpinP_switch < 2){
    if (ispin == 0)  fprintf(fp_EV, "#   Spin = Up \n#\n");
    else             fprintf(fp_EV, "#   Spin = Down \n#\n");
  }
  else{
    fprintf(fp_EV, "#   Spin = Non-colinear \n#\n");
  }
  for (i = 0; i < NUM_c; i++){
    fprintf(fp_EV, "%5d  %25.15e \n", i, eval[i]);
  }
  fclose(fp_EV);

  /*
    LCAO Coefficients of each channel
  */
  sprintf(fname, "%s%s.tranevec%i_%i_%i", filepath, filename, kloop, iw, ispin);

  if ((fp_EV = fopen(fname, "w")) == NULL) {
    printf("\ncannot open file to write EigenChannel(LCAO) \n"); fflush(stdout);
    exit(0);
  } /* if ((fp_EV = fopen(fname, "w")) == NULL) */

  fprintf(fp_EV, "\n");
  fprintf(fp_EV, "***********************************************************\n");
  fprintf(fp_EV, "***********************************************************\n");
  fprintf(fp_EV, "        Eigenvalues and LCAO coefficients                  \n");
  fprintf(fp_EV, "        at the k-points specified in the input file.       \n");
  fprintf(fp_EV, "***********************************************************\n");
  fprintf(fp_EV, "***********************************************************\n");

  fprintf(fp_EV, "\n\n");
  fprintf(fp_EV, "   # of k-point = %i\n", kloop);
  fprintf(fp_EV, "   k2=%10.5f k3=%10.5f\n\n", TRAN_Channel_kpoint[0], TRAN_Channel_kpoint[1]);

  fprintf(fp_EV, "   # of Energy = %i\n", iw);
  fprintf(fp_EV, "   e=%10.5f \n\n", TRAN_Channel_energy * eV2Hartree);

  if (SpinP_switch < 2){
    if (ispin == 0)  fprintf(fp_EV, "   Spin = Up \n\n");
    else             fprintf(fp_EV, "   Spin = Down \n\n");
  }
  else{
    fprintf(fp_EV, "   Spin = Non-colinear \n\n");
  }

  fprintf(fp_EV, "   Real (Re) and imaginary (Im) parts of LCAO coefficients\n\n");

  if (SpinP_switch < 2){
    num0 = 4;
  }
  else{
    num0 = 2;
  }
  num1 = NUM_c / num0 + 1 * (NUM_c%num0 != 0);

  for (i = 1; i <= num1; i++){

    fprintf(fp_EV, "\n");

    for (i1 = -2; i1 <= 0; i1++){

      fprintf(fp_EV, "                     ");

      for (j = 1; j <= num0; j++){

        j1 = num0*(i - 1) + j;

        if (j1 <= NUM_c){

          if (i1 == -2){
            if (SpinP_switch < 2){
              fprintf(fp_EV, "  %4d    ", j1);
              fprintf(fp_EV, "          ");
            }
            else{
              fprintf(fp_EV, " %4d", j1);
              fprintf(fp_EV, "                                   ");
            }
          } /* if (i1 == -2) */

          else if (i1 == -1){
            if (SpinP_switch < 2){
              fprintf(fp_EV, "  %8.4f", eval[j1 - 1]);
              fprintf(fp_EV, "          ");
            }
            else{
              fprintf(fp_EV, "   %8.5f", eval[j1 - 1]);
              fprintf(fp_EV, "                             ");
            }
          } /* else if (i1 == -1) */

          else if (i1 == 0){
            if (SpinP_switch < 2){
              fprintf(fp_EV, "     Re   ");
              fprintf(fp_EV, "     Im   ");
            }
            else{
              fprintf(fp_EV, "     Re(U)");
              fprintf(fp_EV, "     Im(U)");
              fprintf(fp_EV, "     Re(D)");
              fprintf(fp_EV, "     Im(D)");
            }
          } /* else if (i1 == 0) */

        } /* if (j1 <= NUM_c) */
      } /* for (j = 1; j <= num0; j++) */
      fprintf(fp_EV, "\n");
      if (i1 == -1)  fprintf(fp_EV, "\n");
      if (i1 == 0)   fprintf(fp_EV, "\n");
    } /* for (i1 = -2; i1 <= 0; i1++) */

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

    Name_Multiple[0] = "0";
    Name_Multiple[1] = "1";
    Name_Multiple[2] = "2";
    Name_Multiple[3] = "3";
    Name_Multiple[4] = "4";
    Name_Multiple[5] = "5";

    i1 = 1;

    for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++){

      wan1 = WhatSpecies[Gc_AN];

      for (l = 0; l <= Supported_MaxL; l++){ /* Supported_MaxL */
        for (mul = 0; mul<Spe_Num_CBasis[wan1][l]; mul++){
          for (m = 0; m<(2 * l + 1); m++){

            if (l == 0 && mul == 0 && m == 0)
              fprintf(fp_EV, "%4d %3s %s %s",
              Gc_AN, SpeName[wan1], Name_Multiple[mul], Name_Angular[l][m]);
            else
              fprintf(fp_EV, "         %s %s",
              Name_Multiple[mul], Name_Angular[l][m]);

            for (j = 1; j <= num0; j++){

              j1 = num0*(i - 1) + j;

              if (SpinP_switch < 2){
                if (0 < i1 && i1 <= NUM_c && 0 < j1 && j1 <= NUM_c){
                  fprintf(fp_EV, "  %8.5f", GC_R_ref(i1, j1).r);
                  fprintf(fp_EV, "  %8.5f", GC_R_ref(i1, j1).i);
                } /* if (0<i1 && i1 <= NUM_c && 0<j1 && j1 <= NUM_c) */
              }
              else{
                if (0 < i1 && i1 <= NUM_c / 2 && 0 < j1 && j1 <= NUM_c){
                  fprintf(fp_EV, "  %8.5f", GC_R_ref(i1, j1).r);
                  fprintf(fp_EV, "  %8.5f", GC_R_ref(i1, j1).i);
                  fprintf(fp_EV, "  %8.5f", GC_R_ref(i1 + NUM_c / 2, j1).r);
                  fprintf(fp_EV, "  %8.5f", GC_R_ref(i1 + NUM_c / 2, j1).i);
                } /* if (0<i1 && i1 <= NUM_c && 0<j1 && j1 <= NUM_c) */
              }

            } /* for (j = 1; j <= num0; j++) */

            fprintf(fp_EV, "\n");
            i1++;

          } /* for (m = 0; m<(2 * l + 1); m++) */
        } /* for (mul = 0; mul<Spe_Num_CBasis[wan1][l]; mul++) */
      } /* for (l = 0; l <= Supported_MaxL; l++) */
    } /* for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++) */

  } /* for (i = 1; i <= num1; i++) */

  fclose(fp_EV);

} /* void TRAN_Output_ChannelLCAO */

/* Output EigenChannel in the real space representaion */

TRAN_Output_ChannelCube(
  int kloop, 
  int iw, 
  int ispin, 
  int orbit, 
  int NUM_c, 
  double *TRAN_Channel_kpoint, 
  dcomplex *EChannel,
  int *MP,
  double eigentrans,
  double TRAN_Channel_energy)
{
  int GN, Mc_AN, Gc_AN, Cwan, NO0, Nc, fd;
  int l1, l2, l3, Rn;
  int i;
  double co, si, kRn, ReCoef, ImCoef;
  double *RMO_Grid;
  double *IMO_Grid;
  double *RMO_Grid_tmp;
  double *IMO_Grid_tmp;
  dcomplex *MO_Grid;
  char file1[YOUSO10], Out_Extention[YOUSO10], write_mode[YOUSO10];
  FILE *fp;
  int numprocs, myid;
  char buf[fp_bsize];          /* setvbuf */

  MPI_Comm_size(mpi_comm_level1, &numprocs);
  MPI_Comm_rank(mpi_comm_level1, &myid);

  MO_Grid = (dcomplex*)malloc(sizeof(dcomplex)*TNumGrid);
  RMO_Grid = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid = (double*)malloc(sizeof(double)*TNumGrid);
  RMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);

  /* calc. MO on grids */

  for (GN = 0; GN < TNumGrid; GN++){
    RMO_Grid_tmp[GN] = 0.0;
    IMO_Grid_tmp[GN] = 0.0;
  } /* for (GN = 0; GN < TNumGrid; GN++)for (GN = 0; GN < TNumGrid; GN++) */

  for (Mc_AN = 1; Mc_AN <= Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];

    for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++){
      GN = GridListAtom[Mc_AN][Nc];
      Rn = CellListAtom[Mc_AN][Nc];

      l1 = -atv_ijk[Rn][1];
      l2 = -atv_ijk[Rn][2];
      l3 = -atv_ijk[Rn][3];

      if (l1 != 0) continue;

      kRn = TRAN_Channel_kpoint[0] *(double)l2 + TRAN_Channel_kpoint[1] * (double)l3;
      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);

      for (i = 0; i < NO0; i++){
        ReCoef = co * EChannel[MP[Gc_AN] + i - 1].r
               - si * EChannel[MP[Gc_AN] + i - 1].i;
        ImCoef = co * EChannel[MP[Gc_AN] + i - 1].i
               + si * EChannel[MP[Gc_AN] + i - 1].r;
        RMO_Grid_tmp[GN] += ReCoef*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
        IMO_Grid_tmp[GN] += ImCoef*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
      } /* for (i = 0; i < NO0; i++) */
    } /* for (Nc = 0; Nc < GridN_Atom[Gc_AN]; Nc++) */
  }

  MPI_Reduce(&RMO_Grid_tmp[0], &RMO_Grid[0], TNumGrid, MPI_DOUBLE,
    MPI_SUM, Host_ID, mpi_comm_level1);
  MPI_Reduce(&IMO_Grid_tmp[0], &IMO_Grid[0], TNumGrid, MPI_DOUBLE,
    MPI_SUM, Host_ID, mpi_comm_level1);

  for (GN = 0; GN < TNumGrid; GN++){
    MO_Grid[GN].r = RMO_Grid[GN];
    MO_Grid[GN].i = IMO_Grid[GN];
  } /* for (GN = 0; GN < TNumGrid; GN++) */

  /* output the real part of HOMOs on grids */

  if (myid == Host_ID){

    if (OutData_bin_flag){
      sprintf(Out_Extention, ".cube.bin");
      sprintf(write_mode, "wb");
    }
    else{
      sprintf(Out_Extention, ".cube");
      sprintf(write_mode, "w");
    }

    sprintf(file1, "%s%s.tranec%i_%i_%i_%i_r%s",
      filepath, filename, kloop, iw, ispin, orbit, Out_Extention);

    printf("  %s   ", file1);

    if ((fp = fopen(file1, write_mode)) != NULL){

#ifdef xt3
      setvbuf(fp, buf, _IOFBF, fp_bsize);  /* setvbuf */
#endif

      if (OutData_bin_flag){
        Print_CubeTitle_EigenChannel_Bin(
          fp, TRAN_Channel_kpoint, TRAN_Channel_energy, eigentrans, ispin);
        fwrite(RMO_Grid, sizeof(double), Ngrid1 * Ngrid2 * Ngrid3, fp);
      }
      else{
        Print_CubeTitle_EigenChannel(
          fp, TRAN_Channel_kpoint, TRAN_Channel_energy, eigentrans, ispin);
        Print_CubeCData_MO(fp, MO_Grid, "r");
        fd = fileno(fp);
        fsync(fd);
      }

      fclose(fp);
    }
    else{
      printf("Failure of saving MOs\n");
    }

    /* output the imaginary part of HOMOs on grids */

    sprintf(file1, "%s%s.tranec%i_%i_%i_%i_i%s",
      filepath, filename, kloop, iw, ispin, orbit, Out_Extention);
    printf("  %s \n", file1);

    if ((fp = fopen(file1, write_mode)) != NULL){

#ifdef xt3
      setvbuf(fp, buf, _IOFBF, fp_bsize);  /* setvbuf */
#endif

      if (OutData_bin_flag){
        Print_CubeTitle_EigenChannel_Bin(
          fp, TRAN_Channel_kpoint, TRAN_Channel_energy, eigentrans,ispin);
        fwrite(IMO_Grid, sizeof(double), Ngrid1 * Ngrid2 * Ngrid3, fp);
      }
      else{
        Print_CubeTitle_EigenChannel(
          fp, TRAN_Channel_kpoint, TRAN_Channel_energy, eigentrans,ispin);
        Print_CubeCData_MO(fp, MO_Grid, "i");
        fd = fileno(fp);
        fsync(fd);
      }

      fclose(fp);
    }
    else{
      printf("Failure of saving MOs\n");
    }

  } /* if (myid == Host_ID) */

  /* MPI_Barrier */
  MPI_Barrier(mpi_comm_level1);

  free(MO_Grid);
  free(RMO_Grid);
  free(IMO_Grid);
  free(RMO_Grid_tmp);
  free(IMO_Grid_tmp);
} /* TRAN_Output_ChannelCube */

void TRAN_Output_eigentrans_sum(
  int TRAN_Channel_Nkpoint,
  int TRAN_Channel_Nenergy,
 double ***eigentrans_sum)
{
  FILE *fp;
  int myid, kloop, iw, k;
  char file1[YOUSO10];

  MPI_Comm_rank(mpi_comm_level1, &myid);

  if (myid == Host_ID) {
    sprintf(file1, "%s%s.paranegf",filepath, filename);

    if ((fp = fopen(file1, "a")) != NULL) {

      fprintf(fp, "\nSum of transmission eigenvalues\n\n");

      for (kloop = 0; kloop < TRAN_Channel_Nkpoint; kloop++) {
        for (iw = 0; iw < TRAN_Channel_Nenergy; iw++) {
          if (SpinP_switch == 1) {
            for (k = 0; k < SpinP_switch + 1; k++) {
              fprintf(fp, "  Sum.Eigentrans.k%i.E%i.S%i   %20.10e\n",
                kloop, iw, k, eigentrans_sum[kloop][iw][k]);
            }/*for (k = 0; k < SpinP_switch + 1; k++)*/
          }
          else {
            fprintf(fp, "  Sum.Eigentrans.k%i.E%i.S%i   %20.10e\n",
              kloop, iw, 0, eigentrans_sum[kloop][iw][0]);
          }
        }/*for (iw = 0; iw < TRAN_Channel_Nenergy; iw++)*/
      }/**/
      fclose(fp);
    }
    else {
      printf("Failure of saving MOs\n");
    }
  }
}
