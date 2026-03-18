/**********************************************************************
  Calc_optical.c:

     Calc_optical.c is a subroutine to calculate electric conductivity
     and dielectric function, and related optical properties.

  Log of Calc_optical.c:

     3/September/2018  Released by YT Lee 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"

int i,j,k,l,d=3,total_spins;
int CDDF_freq_grid_total_number;
double conductivity_unit = 4599848.23346488111; /* (Ohm*m)^{-1} */
double dielectricless_unit = 366044.28814557; /* (F/(m*s) = Ohm*m) */
double eta, eta2; /* setup eta and eta^2 in Hartree */
double range_Ha; /* range in frequency domain in Hartree */
double step; /* difference omega at frequency domain for conductivity and dielectric function */
dcomplex**** cd_tensor; /* for omega = 1~N */
dcomplex**** df_tensor; /* for omega = 1~N */
dcomplex*** cd_tensor_omega0; /* for omega = 0 */
dcomplex*** df_tensor_omega0; /* for omega = 0 */

/* MPI */
int myid2=0;
int numprocs2=1; /* initial value = 1 when MPI_CommWD2 doesn't be activated or used for parallelization of bands. */

void Set_MPIworld_for_optical(int myid_for_MPIWD2,int numprocs_for_MPIWD2){
  myid2 = myid_for_MPIWD2;
  numprocs2 = numprocs_for_MPIWD2;
}

void Free_optical_1(){
  int Gc_AN,h_AN,tno0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    if (Gc_AN==0){ Gc_AN = 0; tno0 = 1; }else{ tno0 = Spe_Total_CNO[ WhatSpecies[Gc_AN] ]; }
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      for (k=0; k<tno0; k++){
        for (l=0; l< Spe_Total_CNO[ WhatSpecies[ natn[Gc_AN][h_AN] ] ] ; l++) free(MME_allorb[Gc_AN][h_AN][k][l]);
        free(MME_allorb[Gc_AN][h_AN][k]);
      }
      free(MME_allorb[Gc_AN][h_AN]);
    }
    free(MME_allorb[Gc_AN]);
  }
  free(MME_allorb);
}

void Free_optical_2(int n){
  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++){
      for (j=0;j<d;j++) free(cd_tensor[l][i][j]);
      free(cd_tensor[l][i]);
    }
    free(cd_tensor[l]);
  }
  free(cd_tensor);

  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++){
      for (j=0;j<d;j++) free(df_tensor[l][i][j]);
      free(df_tensor[l][i]);
    }
    free(df_tensor[l]);
  }
  free(df_tensor);

  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++) free(cd_tensor_omega0[l][i]);
    free(cd_tensor_omega0[l]);
  }
  free(cd_tensor_omega0);

  for (l=0;l<total_spins;l++){
    for (i=0;i<d;i++) free(df_tensor_omega0[l][i]);
    free(df_tensor_omega0[l]);
  }
  free(df_tensor_omega0);
}

void Initialize_optical(){
  eta      = CDDF_FWHM*0.5/eV2Hartree; eta2=eta*eta; /* setup eta and eta^2 in Hartree */
  range_Ha = (CDDF_max_eV-CDDF_min_eV)/eV2Hartree; /* in Hartree */
  step     = range_Ha/CDDF_freq_grid_number;
  CDDF_freq_grid_total_number = CDDF_freq_grid_number + ( CDDF_FWHM/(step*eV2Hartree) );

  if (SpinP_switch==0){
    total_spins=1;
  }else if (SpinP_switch==1 || SpinP_switch==3){
    total_spins=2;
  }

  /* declare conductivity tensor [1 (SpinP_Switch=0 or 3) or 2 (SpinP_Switch=1)][3][3][freq_grid] */
  cd_tensor = (dcomplex****)malloc(sizeof(dcomplex***)*total_spins);
  for (l=0;l<total_spins;l++){
    cd_tensor[l] = (dcomplex***)malloc(sizeof(dcomplex**)*d);
    for (i=0;i<d;i++){
      cd_tensor[l][i] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
      for (j=0;j<d;j++){
        cd_tensor[l][i][j] = (dcomplex*)malloc(sizeof(dcomplex)*CDDF_freq_grid_total_number);
        for (k=0;k<CDDF_freq_grid_total_number;k++) cd_tensor[l][i][j][k] = Complex(0.0,0.0);
      }
    }
  }

  df_tensor = (dcomplex****)malloc(sizeof(dcomplex***)*total_spins);
  for (l=0;l<total_spins;l++){
    df_tensor[l] = (dcomplex***)malloc(sizeof(dcomplex**)*d);
    for (i=0;i<d;i++){
      df_tensor[l][i] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
      for (j=0;j<d;j++){
        df_tensor[l][i][j] = (dcomplex*)malloc(sizeof(dcomplex)*CDDF_freq_grid_total_number);
        for (k=0;k<CDDF_freq_grid_total_number;k++) df_tensor[l][i][j][k] = Complex(0.0,0.0);
      }
    }
  }

  cd_tensor_omega0 = (dcomplex***)malloc(sizeof(dcomplex**)*total_spins);
  for (l=0;l<total_spins;l++){
    cd_tensor_omega0[l] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
    for (i=0;i<d;i++){
      cd_tensor_omega0[l][i] = (dcomplex*)malloc(sizeof(dcomplex)*d);
      for (j=0;j<d;j++) cd_tensor_omega0[l][i][j] = Complex(0.0,0.0);
    }
  }

  df_tensor_omega0 = (dcomplex***)malloc(sizeof(dcomplex**)*total_spins);
  for (l=0;l<total_spins;l++){
    df_tensor_omega0[l] = (dcomplex**)malloc(sizeof(dcomplex*)*d);
    for (i=0;i<d;i++){
      df_tensor_omega0[l][i] = (dcomplex*)malloc(sizeof(dcomplex)*d);
      for (j=0;j<d;j++) df_tensor_omega0[l][i][j] = Complex(0.0,0.0);
    }
  }
}

void Print_optical(int doesPrintCD,int doesPrintDF,int doesPrintOP){
  /* doesPrintCD = does it print out conductivity? */
  /* doesPrintDF = does it print out dielectric function? */
  /* doesPrintOP = does it print out optical properties? */

  /* start to save conductivity tensor, dielectric function, and other optical properties */
  double k1,k2,k3,k4,omega;

  char fname1[300];
  FILE *fp;

  /* print conductivity and dielectric function */
  if (SpinP_switch==0){

    if (doesPrintCD==1){

      sprintf(fname1,"%s%s.cd_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (real part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].r, cd_tensor_omega0[0][0][1].r, cd_tensor_omega0[0][0][2].r,
          cd_tensor_omega0[0][1][0].r, cd_tensor_omega0[0][1][1].r, cd_tensor_omega0[0][1][2].r,
          cd_tensor_omega0[0][2][0].r, cd_tensor_omega0[0][2][1].r, cd_tensor_omega0[0][2][2].r,
          (cd_tensor_omega0[0][0][0].r + cd_tensor_omega0[0][1][1].r + cd_tensor_omega0[0][2][2].r )/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
            cd_tensor[0][0][0][k].r, cd_tensor[0][0][1][k].r, cd_tensor[0][0][2][k].r,
            cd_tensor[0][1][0][k].r, cd_tensor[0][1][1][k].r, cd_tensor[0][1][2][k].r,
            cd_tensor[0][2][0][k].r, cd_tensor[0][2][1][k].r, cd_tensor[0][2][2][k].r,
            (cd_tensor[0][0][0][k].r + cd_tensor[0][1][1][k].r + cd_tensor[0][2][2][k].r )/3.0 );
      }
      fclose(fp);

      sprintf(fname1,"%s%s.cd_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (imaginery part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].i, cd_tensor_omega0[0][0][1].i, cd_tensor_omega0[0][0][2].i,
          cd_tensor_omega0[0][1][0].i, cd_tensor_omega0[0][1][1].i, cd_tensor_omega0[0][1][2].i,
          cd_tensor_omega0[0][2][0].i, cd_tensor_omega0[0][2][1].i, cd_tensor_omega0[0][2][2].i,
          (cd_tensor_omega0[0][0][0].i + cd_tensor_omega0[0][1][1].i + cd_tensor_omega0[0][2][2].i )/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
            cd_tensor[0][0][0][k].i, cd_tensor[0][0][1][k].i, cd_tensor[0][0][2][k].i,
            cd_tensor[0][1][0][k].i, cd_tensor[0][1][1][k].i, cd_tensor[0][1][2][k].i,
            cd_tensor[0][2][0][k].i, cd_tensor[0][2][1][k].i, cd_tensor[0][2][2][k].i,
            (cd_tensor[0][0][0][k].i + cd_tensor[0][1][1][k].i + cd_tensor[0][2][2][k].i )/3.0 );
      }
      fclose(fp);

    }

    if (doesPrintDF==1){

      sprintf(fname1,"%s%s.df_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (real part)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].r+1, df_tensor_omega0[0][0][1].r, df_tensor_omega0[0][0][2].r,
          df_tensor_omega0[0][1][0].r, df_tensor_omega0[0][1][1].r+1, df_tensor_omega0[0][1][2].r,
          df_tensor_omega0[0][2][0].r, df_tensor_omega0[0][2][1].r, df_tensor_omega0[0][2][2].r+1,
          (df_tensor_omega0[0][0][0].r + df_tensor_omega0[0][1][1].r + df_tensor_omega0[0][2][2].r +3)/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].r+1, df_tensor[0][0][1][k].r, df_tensor[0][0][2][k].r,
          df_tensor[0][1][0][k].r, df_tensor[0][1][1][k].r+1, df_tensor[0][1][2][k].r,
          df_tensor[0][2][0][k].r, df_tensor[0][2][1][k].r, df_tensor[0][2][2][k].r+1,
          (df_tensor[0][0][0][k].r + df_tensor[0][1][1][k].r + df_tensor[0][2][2][k].r +3)/3.0 );
      }
      fclose(fp);

      sprintf(fname1,"%s%s.df_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (imaginery part)\n");
      fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].i, df_tensor_omega0[0][0][1].i, df_tensor_omega0[0][0][2].i,
          df_tensor_omega0[0][1][0].i, df_tensor_omega0[0][1][1].i, df_tensor_omega0[0][1][2].i,
          df_tensor_omega0[0][2][0].i, df_tensor_omega0[0][2][1].i, df_tensor_omega0[0][2][2].i,
          (df_tensor_omega0[0][0][0].i + df_tensor_omega0[0][1][1].i + df_tensor_omega0[0][2][2].i )/3.0 );
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].i, df_tensor[0][0][1][k].i, df_tensor[0][0][2][k].i,
          df_tensor[0][1][0][k].i, df_tensor[0][1][1][k].i, df_tensor[0][1][2][k].i,
          df_tensor[0][2][0][k].i, df_tensor[0][2][1][k].i, df_tensor[0][2][2][k].i,
          (df_tensor[0][0][0][k].i + df_tensor[0][1][1][k].i + df_tensor[0][2][2][k].i )/3.0  );
      }
      fclose(fp);

    }
  }else if (SpinP_switch==1 || SpinP_switch==3){

    if (doesPrintCD==1){

      sprintf(fname1,"%s%s.cd_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (real part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].r+cd_tensor_omega0[1][0][0].r, cd_tensor_omega0[0][0][1].r+cd_tensor_omega0[1][0][1].r, cd_tensor_omega0[0][0][2].r+cd_tensor_omega0[1][0][2].r,
          cd_tensor_omega0[0][1][0].r+cd_tensor_omega0[1][1][0].r, cd_tensor_omega0[0][1][1].r+cd_tensor_omega0[1][1][1].r, cd_tensor_omega0[0][1][2].r+cd_tensor_omega0[1][1][2].r,
          cd_tensor_omega0[0][2][0].r+cd_tensor_omega0[1][2][0].r, cd_tensor_omega0[0][2][1].r+cd_tensor_omega0[1][2][1].r, cd_tensor_omega0[0][2][2].r+cd_tensor_omega0[1][2][2].r,
          (cd_tensor_omega0[0][0][0].r + cd_tensor_omega0[0][1][1].r + cd_tensor_omega0[0][2][2].r + cd_tensor_omega0[1][0][0].r + cd_tensor_omega0[1][1][1].r + cd_tensor_omega0[1][2][2].r)/3.0,
          cd_tensor_omega0[0][0][0].r, cd_tensor_omega0[0][0][1].r, cd_tensor_omega0[0][0][2].r, /* for up-spin tensor */
          cd_tensor_omega0[0][1][0].r, cd_tensor_omega0[0][1][1].r, cd_tensor_omega0[0][1][2].r,
          cd_tensor_omega0[0][2][0].r, cd_tensor_omega0[0][2][1].r, cd_tensor_omega0[0][2][2].r,
          cd_tensor_omega0[1][0][0].r, cd_tensor_omega0[1][0][1].r, cd_tensor_omega0[1][0][2].r, /* for down-spin tensor */
          cd_tensor_omega0[1][1][0].r, cd_tensor_omega0[1][1][1].r, cd_tensor_omega0[1][1][2].r,
          cd_tensor_omega0[1][2][0].r, cd_tensor_omega0[1][2][1].r, cd_tensor_omega0[1][2][2].r);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          cd_tensor[0][0][0][k].r+cd_tensor[1][0][0][k].r, cd_tensor[0][0][1][k].r+cd_tensor[1][0][1][k].r, cd_tensor[0][0][2][k].r+cd_tensor[1][0][2][k].r,
          cd_tensor[0][1][0][k].r+cd_tensor[1][1][0][k].r, cd_tensor[0][1][1][k].r+cd_tensor[1][1][1][k].r, cd_tensor[0][1][2][k].r+cd_tensor[1][1][2][k].r,
          cd_tensor[0][2][0][k].r+cd_tensor[1][2][0][k].r, cd_tensor[0][2][1][k].r+cd_tensor[1][2][1][k].r, cd_tensor[0][2][2][k].r+cd_tensor[1][2][2][k].r,
          (cd_tensor[0][0][0][k].r + cd_tensor[0][1][1][k].r + cd_tensor[0][2][2][k].r + cd_tensor[1][0][0][k].r + cd_tensor[1][1][1][k].r + cd_tensor[1][2][2][k].r)/3.0,
          cd_tensor[0][0][0][k].r, cd_tensor[0][0][1][k].r, cd_tensor[0][0][2][k].r, /* for up-spin tensor */
          cd_tensor[0][1][0][k].r, cd_tensor[0][1][1][k].r, cd_tensor[0][1][2][k].r,
          cd_tensor[0][2][0][k].r, cd_tensor[0][2][1][k].r, cd_tensor[0][2][2][k].r,
          cd_tensor[1][0][0][k].r, cd_tensor[1][0][1][k].r, cd_tensor[1][0][2][k].r, /* for down-spin tensor */
          cd_tensor[1][1][0][k].r, cd_tensor[1][1][1][k].r, cd_tensor[1][1][2][k].r,
          cd_tensor[1][2][0][k].r, cd_tensor[1][2][1][k].r, cd_tensor[1][2][2][k].r);
      }
      fclose(fp);

      sprintf(fname1,"%s%s.cd_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# conductivity tensor (imaginery part) , unit = Siemens/meter = Mho/meter = 1/(Ohm*meter)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          cd_tensor_omega0[0][0][0].i+cd_tensor_omega0[1][0][0].i, cd_tensor_omega0[0][0][1].i+cd_tensor_omega0[1][0][1].i, cd_tensor_omega0[0][0][2].i+cd_tensor_omega0[1][0][2].i,
          cd_tensor_omega0[0][1][0].i+cd_tensor_omega0[1][1][0].i, cd_tensor_omega0[0][1][1].i+cd_tensor_omega0[1][1][1].i, cd_tensor_omega0[0][1][2].i+cd_tensor_omega0[1][1][2].i,
          cd_tensor_omega0[0][2][0].i+cd_tensor_omega0[1][2][0].i, cd_tensor_omega0[0][2][1].i+cd_tensor_omega0[1][2][1].i, cd_tensor_omega0[0][2][2].i+cd_tensor_omega0[1][2][2].i,
          (cd_tensor_omega0[0][0][0].i + cd_tensor_omega0[0][1][1].i + cd_tensor_omega0[0][2][2].i + cd_tensor_omega0[1][0][0].i + cd_tensor_omega0[1][1][1].i + cd_tensor_omega0[1][2][2].i)/3.0,
          cd_tensor_omega0[0][0][0].i, cd_tensor_omega0[0][0][1].i, cd_tensor_omega0[0][0][2].i, /* for up-spin tensor */
          cd_tensor_omega0[0][1][0].i, cd_tensor_omega0[0][1][1].i, cd_tensor_omega0[0][1][2].i,
          cd_tensor_omega0[0][2][0].i, cd_tensor_omega0[0][2][1].i, cd_tensor_omega0[0][2][2].i,
          cd_tensor_omega0[1][0][0].i, cd_tensor_omega0[1][0][1].i, cd_tensor_omega0[1][0][2].i, /* for down-spin tensor */
          cd_tensor_omega0[1][1][0].i, cd_tensor_omega0[1][1][1].i, cd_tensor_omega0[1][1][2].i,
          cd_tensor_omega0[1][2][0].i, cd_tensor_omega0[1][2][1].i, cd_tensor_omega0[1][2][2].i);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          cd_tensor[0][0][0][k].i+cd_tensor[1][0][0][k].i, cd_tensor[0][0][1][k].i+cd_tensor[1][0][1][k].i, cd_tensor[0][0][2][k].i+cd_tensor[1][0][2][k].i,
          cd_tensor[0][1][0][k].i+cd_tensor[1][1][0][k].i, cd_tensor[0][1][1][k].i+cd_tensor[1][1][1][k].i, cd_tensor[0][1][2][k].i+cd_tensor[1][1][2][k].i,
          cd_tensor[0][2][0][k].i+cd_tensor[1][2][0][k].i, cd_tensor[0][2][1][k].i+cd_tensor[1][2][1][k].i, cd_tensor[0][2][2][k].i+cd_tensor[1][2][2][k].i,
          (cd_tensor[0][0][0][k].i + cd_tensor[0][1][1][k].i + cd_tensor[0][2][2][k].i + cd_tensor[1][0][0][k].i + cd_tensor[1][1][1][k].i + cd_tensor[1][2][2][k].i )/3.0,
          cd_tensor[0][0][0][k].i, cd_tensor[0][0][1][k].i, cd_tensor[0][0][2][k].i, /* for up-spin tensor */
          cd_tensor[0][1][0][k].i, cd_tensor[0][1][1][k].i, cd_tensor[0][1][2][k].i,
          cd_tensor[0][2][0][k].i, cd_tensor[0][2][1][k].i, cd_tensor[0][2][2][k].i,
          cd_tensor[1][0][0][k].i, cd_tensor[1][0][1][k].i, cd_tensor[1][0][2][k].i, /* for down-spin tensor */
          cd_tensor[1][1][0][k].i, cd_tensor[1][1][1][k].i, cd_tensor[1][1][2][k].i,
          cd_tensor[1][2][0][k].i, cd_tensor[1][2][1][k].i, cd_tensor[1][2][2][k].i);
      }
      fclose(fp);

    }

    if (doesPrintDF==1){

      sprintf(fname1,"%s%s.df_re",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (real part)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].r+df_tensor_omega0[1][0][0].r+1, df_tensor_omega0[0][0][1].r+df_tensor_omega0[1][0][1].r, df_tensor_omega0[0][0][2].r+df_tensor_omega0[1][0][2].r,
          df_tensor_omega0[0][1][0].r+df_tensor_omega0[1][1][0].r, df_tensor_omega0[0][1][1].r+df_tensor_omega0[1][1][1].r+1, df_tensor_omega0[0][1][2].r+df_tensor_omega0[1][1][2].r,
          df_tensor_omega0[0][2][0].r+df_tensor_omega0[1][2][0].r, df_tensor_omega0[0][2][1].r+df_tensor_omega0[1][2][1].r, df_tensor_omega0[0][2][2].r+df_tensor_omega0[1][2][2].r+1,
          (3+df_tensor_omega0[0][0][0].r + df_tensor_omega0[0][1][1].r + df_tensor_omega0[0][2][2].r + df_tensor_omega0[1][0][0].r + df_tensor_omega0[1][1][1].r + df_tensor_omega0[1][2][2].r)/3.0,
          df_tensor_omega0[0][0][0].r+1, df_tensor_omega0[0][0][1].r, df_tensor_omega0[0][0][2].r, /* for up-spin tensor */
          df_tensor_omega0[0][1][0].r, df_tensor_omega0[0][1][1].r+1, df_tensor_omega0[0][1][2].r,
          df_tensor_omega0[0][2][0].r, df_tensor_omega0[0][2][1].r, df_tensor_omega0[0][2][2].r+1,
          df_tensor_omega0[1][0][0].r+1, df_tensor_omega0[1][0][1].r, df_tensor_omega0[1][0][2].r, /* for down-spin tensor */
          df_tensor_omega0[1][1][0].r, df_tensor_omega0[1][1][1].r+1, df_tensor_omega0[1][1][2].r,
          df_tensor_omega0[1][2][0].r, df_tensor_omega0[1][2][1].r, df_tensor_omega0[1][2][2].r+1);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].r+df_tensor[1][0][0][k].r+1, df_tensor[0][0][1][k].r+df_tensor[1][0][1][k].r, df_tensor[0][0][2][k].r+df_tensor[1][0][2][k].r,
          df_tensor[0][1][0][k].r+df_tensor[1][1][0][k].r, df_tensor[0][1][1][k].r+df_tensor[1][1][1][k].r+1, df_tensor[0][1][2][k].r+df_tensor[1][1][2][k].r,
          df_tensor[0][2][0][k].r+df_tensor[1][2][0][k].r, df_tensor[0][2][1][k].r+df_tensor[1][2][1][k].r, df_tensor[0][2][2][k].r+df_tensor[1][2][2][k].r+1,
          (df_tensor[0][0][0][k].r+df_tensor[0][1][1][k].r+df_tensor[0][2][2][k].r+df_tensor[1][0][0][k].r+df_tensor[1][1][1][k].r+df_tensor[1][2][2][k].r+3)/3.0,
          df_tensor[0][0][0][k].r, df_tensor[0][0][1][k].r, df_tensor[0][0][2][k].r, /* for up-spin tensor */
          df_tensor[0][1][0][k].r, df_tensor[0][1][1][k].r, df_tensor[0][1][2][k].r,
          df_tensor[0][2][0][k].r, df_tensor[0][2][1][k].r, df_tensor[0][2][2][k].r,
          df_tensor[1][0][0][k].r, df_tensor[1][0][1][k].r, df_tensor[1][0][2][k].r, /* for down-spin tensor */
          df_tensor[1][1][0][k].r, df_tensor[1][1][1][k].r, df_tensor[1][1][2][k].r,
          df_tensor[1][2][0][k].r, df_tensor[1][2][1][k].r, df_tensor[1][2][2][k].r);
      }
      fclose(fp);

      sprintf(fname1,"%s%s.df_im",filepath,filename);
      fp = fopen(fname1,"w");
      fprintf(fp,"# dielectric function (imaginery part)\n");
      fprintf(fp,"# index: energy-grid=1, xx~zz=2~10, trace=11, up-xx ~ up-zz=12~20, down-xx ~ down-zz=21~29\n");
      fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz   (xx+yy+zz)/3");
      fprintf(fp,"        up-xx         up-xy         up-xz         up-yx         up-yy         up-yz         up-zx         up-zy         up-zz");
      fprintf(fp,"       down-xx       down-xy       down-xz       down-yx       down-yy       down-yz       down-zx       down-zy       down-zz\n");
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", 0.0,
          df_tensor_omega0[0][0][0].i+df_tensor_omega0[1][0][0].i, df_tensor_omega0[0][0][1].i+df_tensor_omega0[1][0][1].i, df_tensor_omega0[0][0][2].i+df_tensor_omega0[1][0][2].i,
          df_tensor_omega0[0][1][0].i+df_tensor_omega0[1][1][0].i, df_tensor_omega0[0][1][1].i+df_tensor_omega0[1][1][1].i, df_tensor_omega0[0][1][2].i+df_tensor_omega0[1][1][2].i,
          df_tensor_omega0[0][2][0].i+df_tensor_omega0[1][2][0].i, df_tensor_omega0[0][2][1].i+df_tensor_omega0[1][2][1].i, df_tensor_omega0[0][2][2].i+df_tensor_omega0[1][2][2].i,
          (df_tensor_omega0[0][0][0].i + df_tensor_omega0[0][1][1].i + df_tensor_omega0[0][2][2].i + df_tensor_omega0[1][0][0].i + df_tensor_omega0[1][1][1].i + df_tensor_omega0[1][2][2].i)/3.0,
          df_tensor_omega0[0][0][0].i, df_tensor_omega0[0][0][1].i, df_tensor_omega0[0][0][2].i, /* for up-spin tensor */
          df_tensor_omega0[0][1][0].i, df_tensor_omega0[0][1][1].i, df_tensor_omega0[0][1][2].i,
          df_tensor_omega0[0][2][0].i, df_tensor_omega0[0][2][1].i, df_tensor_omega0[0][2][2].i,
          df_tensor_omega0[1][0][0].i, df_tensor_omega0[1][0][1].i, df_tensor_omega0[1][0][2].i, /* for down-spin tensor */
          df_tensor_omega0[1][1][0].i, df_tensor_omega0[1][1][1].i, df_tensor_omega0[1][1][2].i,
          df_tensor_omega0[1][2][0].i, df_tensor_omega0[1][2][1].i, df_tensor_omega0[1][2][2].i);
      for (k=0;k<CDDF_freq_grid_number;k++){
        omega=(k+1)*step*eV2Hartree;
        fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n", omega,
          df_tensor[0][0][0][k].i+df_tensor[1][0][0][k].i, df_tensor[0][0][1][k].i+df_tensor[1][0][1][k].i, df_tensor[0][0][2][k].i+df_tensor[1][0][2][k].i,
          df_tensor[0][1][0][k].i+df_tensor[1][1][0][k].i, df_tensor[0][1][1][k].i+df_tensor[1][1][1][k].i, df_tensor[0][1][2][k].i+df_tensor[1][1][2][k].i,
          df_tensor[0][2][0][k].i+df_tensor[1][2][0][k].i, df_tensor[0][2][1][k].i+df_tensor[1][2][1][k].i, df_tensor[0][2][2][k].i+df_tensor[1][2][2][k].i,
          (df_tensor[0][0][0][k].i+df_tensor[0][1][1][k].i+df_tensor[0][2][2][k].i+df_tensor[1][0][0][k].i+df_tensor[1][1][1][k].i+df_tensor[1][2][2][k].i)/3.0,
          df_tensor[0][0][0][k].i, df_tensor[0][0][1][k].i, df_tensor[0][0][2][k].i, /* for up-spin tensor */
          df_tensor[0][1][0][k].i, df_tensor[0][1][1][k].i, df_tensor[0][1][2][k].i,
          df_tensor[0][2][0][k].i, df_tensor[0][2][1][k].i, df_tensor[0][2][2][k].i,
          df_tensor[1][0][0][k].i, df_tensor[1][0][1][k].i, df_tensor[1][0][2][k].i, /* for down-spin tensor */
          df_tensor[1][1][0][k].i, df_tensor[1][1][1][k].i, df_tensor[1][1][2][k].i,
          df_tensor[1][2][0][k].i, df_tensor[1][2][1][k].i, df_tensor[1][2][2][k].i);
      }
      fclose(fp);

    }

  }


  if (doesPrintOP==1){

    /* start to write refractive index, extinction_coefficient, absorption coefficient, reflection coefficient , and transmission coefficient. */
    sprintf(fname1,"%s%s.refractive_index",filepath,filename); /* rft = refractive */
    fp = fopen(fname1,"w");
    fprintf(fp,"# refractive index\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");

    double refractive_index[9],extinction_coefficient[9],absorption_coefficient[9],reflection_coefficient[9],transmission_coefficient[9];
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r+df_tensor_omega0[1][i][j].r;
          k3 = df_tensor_omega0[0][i][j].i+df_tensor_omega0[1][i][j].i;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + k3*k3 ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
        }
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,refractive_index[0],refractive_index[1],refractive_index[2],refractive_index[3],refractive_index[4],refractive_index[5],refractive_index[6],refractive_index[7],refractive_index[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          }
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,refractive_index[0],refractive_index[1],refractive_index[2],refractive_index[3],refractive_index[4],refractive_index[5],refractive_index[6],refractive_index[7],refractive_index[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.extinction",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# extinction coefficient\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){ 
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r + df_tensor_omega0[1][i][j].r;
          k3 = df_tensor_omega0[0][i][j].i + df_tensor_omega0[1][i][j].i;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + k3*k3 ) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,extinction_coefficient[0],extinction_coefficient[1],extinction_coefficient[2],extinction_coefficient[3],extinction_coefficient[4],extinction_coefficient[5],extinction_coefficient[6],extinction_coefficient[7],extinction_coefficient[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r + df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i + df_tensor[1][i][j][k].i;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,extinction_coefficient[0],extinction_coefficient[1],extinction_coefficient[2],extinction_coefficient[3],extinction_coefficient[4],extinction_coefficient[5],extinction_coefficient[6],extinction_coefficient[7],extinction_coefficient[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.absorption",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# absorption coefficient (unit = 10^6 1/m)\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    double hbar_ineV = pow(10,9)/0.65821195144 ; /* absorption coefficient (10^6 1/m) */
    double speed_of_light = 299792458 ;
    /* at omega = 0 */
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
          absorption_coefficient[l] = 2 * omega * extinction_coefficient[l] * hbar_ineV / speed_of_light ; 
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,absorption_coefficient[0],absorption_coefficient[1],absorption_coefficient[2],absorption_coefficient[3],absorption_coefficient[4],absorption_coefficient[5],absorption_coefficient[6],absorption_coefficient[7],absorption_coefficient[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.reflection",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# reflection coefficient (%%)\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){ 
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2+=1;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r+df_tensor_omega0[1][i][j].r;
          if (i==j) k2+=1;
          k3 = df_tensor_omega0[0][i][j].i+df_tensor_omega0[1][i][j].i;
          k1 = sqrt ( k2*k2 + k3*k3 ) ; //// check
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }
        k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
        k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
        k3 = extinction_coefficient[l]*extinction_coefficient[l];
        reflection_coefficient[l] = 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,reflection_coefficient[0],reflection_coefficient[1],reflection_coefficient[2],reflection_coefficient[3],reflection_coefficient[4],reflection_coefficient[5],reflection_coefficient[6],reflection_coefficient[7],reflection_coefficient[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2+=1;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            if (i==j) k2+=1;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            k1 = sqrt ( k2*k2 + k3*k3 ) ; //// check
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
          k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
          k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
          k3 = extinction_coefficient[l]*extinction_coefficient[l];
          reflection_coefficient[l] = 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
        }
      }
      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,reflection_coefficient[0],reflection_coefficient[1],reflection_coefficient[2],reflection_coefficient[3],reflection_coefficient[4],reflection_coefficient[5],reflection_coefficient[6],reflection_coefficient[7],reflection_coefficient[8]);
    }
    fclose(fp);

    sprintf(fname1,"%s%s.transmission",filepath,filename);
    fp = fopen(fname1,"w");
    fprintf(fp,"# transmission coefficient (%%)\n");
    fprintf(fp,"# index: energy-grid=1, xx=2, xy=3, xz=4, yx=5, yy=6, yz=7, zx=8, zy=9, zz=10, trace=11\n");
    fprintf(fp,"#energy-grid(eV)     xx            xy            xz            yx            yy            yz            zx            zy            zz\n");
    /* at omega = 0 */
    for (i=0;i<d;i++){
      for (j=0;j<d;j++){ 
        l=i*d+j;
        if (SpinP_switch==0){
          k2 = df_tensor_omega0[0][i][j].r;
          if (i==j) k2++;
          k1 = sqrt ( k2*k2 + df_tensor_omega0[0][i][j].i*df_tensor_omega0[0][i][j].i ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }else if (SpinP_switch==1 || SpinP_switch==3){
          k2 = df_tensor_omega0[0][i][j].r+df_tensor_omega0[1][i][j].r;
          k3 = df_tensor_omega0[0][i][j].i+df_tensor_omega0[1][i][j].i;
          if (i==j) k2++;
          k1 = sqrt ( k2*k2 + k3*k3 ) ;
          refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
          extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
        }
        k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
        k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
        k3 = extinction_coefficient[l]*extinction_coefficient[l];
        transmission_coefficient[l] = 100.0 - 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
      }
    }
    fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",0.0,transmission_coefficient[0],transmission_coefficient[1],transmission_coefficient[2],transmission_coefficient[3],transmission_coefficient[4],transmission_coefficient[5],transmission_coefficient[6],transmission_coefficient[7],transmission_coefficient[8]);
    for (k=0;k<CDDF_freq_grid_number;k++){
      omega=(k+1)*step*eV2Hartree;
      for (i=0;i<d;i++){
        for (j=0;j<d;j++){ 
          l=i*d+j;
          if (SpinP_switch==0){
            k2 = df_tensor[0][i][j][k].r;
            if (i==j) k2++;
            k1 = sqrt ( k2*k2 + df_tensor[0][i][j][k].i*df_tensor[0][i][j][k].i ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }else if (SpinP_switch==1 || SpinP_switch==3){
            k2 = df_tensor[0][i][j][k].r+df_tensor[1][i][j][k].r;
            k3 = df_tensor[0][i][j][k].i+df_tensor[1][i][j][k].i;
            if (i==j) k2++;
            k1 = sqrt ( k2*k2 + k3*k3 ) ;
            refractive_index[l] = sqrt( ( k1 + k2 ) * 0.5) ;
            extinction_coefficient[l] = sqrt( ( k1 - k2 ) * 0.5) ;
          }
          k1 = (refractive_index[l]-1)*(refractive_index[l]-1);
          k2 = (refractive_index[l]+1)*(refractive_index[l]+1);
          k3 = extinction_coefficient[l]*extinction_coefficient[l];
          transmission_coefficient[l] = 100.0 - 100 * ( k1 + k3 ) / ( k2 + k3 ) ; 
        }
      }

      fprintf(fp," %8.5lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf  %12.7lf\n",omega,transmission_coefficient[0],transmission_coefficient[1],transmission_coefficient[2],transmission_coefficient[3],transmission_coefficient[4],transmission_coefficient[5],transmission_coefficient[6],transmission_coefficient[7],transmission_coefficient[8]);
    }
    fclose(fp);
  }
}

void Calc_band_optical_col_1(double kx,double ky,double kz,int spin_index,int n,double* EIGEN, dcomplex** H, double* fd_dist,double ChemP){
  double k2,k3,omega,p1,p5,p2,p3,p4,kRn,sx;
  double dFD,dEuv,dEuv2,dEuv_p_omega,dFDdEuvdeno,dFDdEuvdeno2;
  double dE_limit = 1.0E-6; /* for intra-band and degenerate states */
  int Mc_AN,Gc_AN,tno0,tno1,h_AN,Gh_AN;
  /* int Cwan,Hwan; */
  int atom1,atom2,Rn,l1,l2,l3,o,state_m,state_n,sp;
  int tnos; // total number of states

  /* p1 = CDDF_FWHM*5/eV2Hartree; */ /* range of extended frequency, outside of spectrum. */
  p1 = CDDF_AddMaxE/eV2Hartree;
  tnos = CDDF_max_unoccupied_state+1;

  /* create array to check whether the state does require to calculate */
  int** is_dE_within_range = (int**)malloc(sizeof(int*)*tnos);
  for (i=0; i<tnos; i++) is_dE_within_range[i] = (int*)malloc(sizeof(int)*tnos);

  /* check whether the state does require to calculate */
  for (state_m=0;state_m<tnos;state_m++)
    for (state_n=0;state_n<tnos;state_n++){
      k3 = fabs( EIGEN[state_m+1] - EIGEN[state_n+1] ) ; /* dE */
      if (CDDF_material_type==0){ /* for insulator */
        if ( k3 <= range_Ha + p1 && (fd_dist[state_n] - fd_dist[state_m]) != 0.0 ){ /* for metal - Check (1) is it within energy ragne? */
          is_dE_within_range[state_m][state_n]=1; /* Decide whether it needs to calculate MME from state_m to state_n or not. */
        }else{
          is_dE_within_range[state_m][state_n]=0;
        }
        /* printf("%i-0,",CDDF_material_type); */
      }else if (CDDF_material_type==1){ /* for metal */
        if ( k3 <= range_Ha + p1 ){ /* for insulator - Check (1) is it within energy ragne and (2) difference of Fermi-Dirace distribution !=0 */
          is_dE_within_range[state_m][state_n]=1; /* Decide whether it needs to calculate MME from state_m to state_n or not. */
        }else{
          is_dE_within_range[state_m][state_n]=0;
        }
        /* printf("%i-1,",CDDF_material_type);*/
      }
    }
  /* ### MPI for bands ### */
  unsigned int n0 = tnos*tnos, n1 = n0/numprocs2, n2 = n0%numprocs2, n3[numprocs2], n5[numprocs2+1], ui, ul, uk; /* for step 1 */

  /* by Ozaki */
  for (i=0; i<(numprocs2+1); i++){
    n5[i] = numprocs2 + 1 - i; 
  }

  if (numprocs2>1){

    for (ui=0; ui<numprocs2; ui++){
      n3[ui]=n1;
      if (ui<n2) n3[ui]++;
    }

    uk=0; ul=n3[0]; n5[0]=0; n5[1]=ul;
    for (ui=0; ui<n0; ui++){
      if (ui>=ul){
        uk++;
        ul += n3[uk];
        n5[uk+1] = ul;
      }
    }

  }
  else{
    n5[0] = 0; 
    n5[1] = n0;
    n3[0] = n0;
  }

  /* initial atomic orbital index */
  int atomic_orbital_index[atomnum];
  atomic_orbital_index[0]=0;
  i=0;
  for (Gc_AN=1;Gc_AN<atomnum;Gc_AN++){
    /* Cwan = WhatSpecies[Gc_AN]; */
    /* tno0 = Spe_Total_CNO[Cwan]; */
    tno0 = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];
    i += tno0;
    atomic_orbital_index[Gc_AN]=i;
  }

  dcomplex MME_x_cuj_x_pf[3],MME_x_cui_x_cuj_x_pf[3]; // 3 => xyz
  dcomplex cuj_x_pf,conjg_lcao_coef,p6[9],phase_factor;

  /* for (l=0;l<n0;l++){ */
  for (ul=n5[myid2]; ul<n5[myid2+1]; ul++){
    state_m = ul/tnos; /* occupied state */
    state_n = ul%tnos; /* unoccupied state */

    if (is_dE_within_range[state_m][state_n]==1){ /* if dE is within range in Hartree */

      /* initialize */
      double MME[6]={0.0,0.0,0.0,0.0,0.0,0.0}; /* for saving momentum matrix elements, i.e. (x_re,y_re,z_re,x_im,y_im,z_im) */

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){ /* the third loop  - index of the first atoms within primitive cell */
        /* Cwan = WhatSpecies[Gc_AN]; */
        tno0 = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];

        atom1=Gc_AN-1;

        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){ /* the fourth loop - index of the second atoms within primtive cell */
          Gh_AN = natn[Gc_AN][h_AN];
          /* Hwan = WhatSpecies[Gh_AN]; */
          tno1 = Spe_Total_CNO[ WhatSpecies[Gh_AN] ];

          atom2=Gh_AN-1;

          Rn=ncn[Gc_AN][h_AN]; l1 = atv_ijk[Rn][1]; l2 = atv_ijk[Rn][2]; l3 = atv_ijk[Rn][3];

          kRn = kx*(double)l1 + ky*(double)l2 + kz*(double)l3;

          /* phase factor */
          phase_factor = Complex( cos(PIx2*kRn) , sin(PIx2*kRn) );

          MME_x_cui_x_cuj_x_pf[0]=Complex(0.0,0.0);
          MME_x_cui_x_cuj_x_pf[1]=Complex(0.0,0.0);
          MME_x_cui_x_cuj_x_pf[2]=Complex(0.0,0.0);

          for (k=0; k<tno0; k++){ /* the fifth loop  - orbital index k within atom i */
            MME_x_cuj_x_pf[0]=Complex(0.0,0.0);
            MME_x_cuj_x_pf[1]=Complex(0.0,0.0);
            MME_x_cuj_x_pf[2]=Complex(0.0,0.0);

            cuj_x_pf = Complex(0,0);
            for (o=0; o<tno1; o++){ /* the fixth loop  - orbital index o within atom j */
              /* step (1) */
              cuj_x_pf = Cmul( H[atomic_orbital_index[atom2]+o+1][state_n+1], phase_factor ) ; /* LCAO[ atomic orbitals ][ states ] */

              /* step (2) */
              MME_x_cuj_x_pf[0] = Cadd( MME_x_cuj_x_pf[0], RCmul( MME_allorb[Gc_AN][h_AN][k][o][0], cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[1] = Cadd( MME_x_cuj_x_pf[1], RCmul( MME_allorb[Gc_AN][h_AN][k][o][1], cuj_x_pf ) ) ;
              MME_x_cuj_x_pf[2] = Cadd( MME_x_cuj_x_pf[2], RCmul( MME_allorb[Gc_AN][h_AN][k][o][2], cuj_x_pf ) ) ;
            } /* 8th loop for orbital index o of atom j */

            /* step (3) */
            conjg_lcao_coef = Conjg( H[atomic_orbital_index[atom1]+k+1][state_m+1] ); /* LCAO[ atomic orbitals ][ states ] */

            MME_x_cui_x_cuj_x_pf[0] = Cadd( MME_x_cui_x_cuj_x_pf[0], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[0] ) );
            MME_x_cui_x_cuj_x_pf[1] = Cadd( MME_x_cui_x_cuj_x_pf[1], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[1] ) );
            MME_x_cui_x_cuj_x_pf[2] = Cadd( MME_x_cui_x_cuj_x_pf[2], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[2] ) );
          } /* 7th loop for orbital index k of atom i */

          /* step (4) : momentum matrix elements - MME[ state_m ][ state_n ][ i ] */
          MME[0] += MME_x_cui_x_cuj_x_pf[0].i ;
          MME[1] += MME_x_cui_x_cuj_x_pf[1].i ;
          MME[2] += MME_x_cui_x_cuj_x_pf[2].i ;
          MME[3] -= MME_x_cui_x_cuj_x_pf[0].r ;
          MME[4] -= MME_x_cui_x_cuj_x_pf[1].r ;
          MME[5] -= MME_x_cui_x_cuj_x_pf[2].r ; 
        } /* 6th loop for atomic index j */
      } /* 5th loop for atomic index i */
      /* ### end of MME from state m to state n ### */

      double dfd_x_kw = fd_dist[state_n] - fd_dist[state_m];
      dEuv  = EIGEN[state_m+1] - EIGEN[state_n+1] ;

      p6[0] = Complex( MME[0]*MME[0] + MME[3]*MME[3] , 0.0 );
      p6[1] = Complex( MME[0]*MME[1] + MME[3]*MME[4] , MME[3]*MME[1] - MME[0]*MME[4] );
      p6[2] = Complex( MME[0]*MME[2] + MME[3]*MME[5] , MME[3]*MME[2] - MME[0]*MME[5] );
      p6[3] = Complex( MME[1]*MME[0] + MME[4]*MME[3] , -p6[1].i );
      p6[4] = Complex( MME[1]*MME[1] + MME[4]*MME[4] , 0.0 );
      p6[5] = Complex( MME[1]*MME[2] + MME[4]*MME[5] , MME[4]*MME[2] - MME[1]*MME[5] );
      p6[6] = Complex( MME[2]*MME[0] + MME[5]*MME[3] , MME[5]*MME[0] - MME[2]*MME[3] );
      p6[7] = Complex( MME[2]*MME[1] + MME[5]*MME[4] , -p6[5].i );
      p6[8] = Complex( MME[2]*MME[2] + MME[5]*MME[5] , 0.0 );

      sx = (EIGEN[state_m+1] - ChemP) * Beta;
      if (100<sx) sx = 100.0; 
      k2 = exp(sx);
      k3 = - k2 * Beta / ( (1+k2) * (1+k2) ) ;

      for (i=0;i<d;i++){ /* tensor index, i = alpha */
        for (j=0;j<d;j++){ /* tensor index, j = beta */

          int j1 = i*3+j;
          /* if ((p6[j1].r)==0 && p6[j1].i==0) continue; */
          p1 = p6[j1].r*eta ;
          p5 = p6[j1].i*eta ;

          for (k=0; k<CDDF_freq_grid_total_number; k++){ /* scan all frequencies */

            omega=(k+1)*step;
            dEuv_p_omega = dEuv + omega ;
         
            /* At the same k-point */
            /* (1) intra-band : state_m == state_n ( dEuv == 0 ) */
            if ( fabs(dEuv) < dE_limit && state_m == state_n ){

              dFDdEuvdeno  = k3 / ( ( omega * omega ) + eta2 ) ;
              dFDdEuvdeno2 = dFDdEuvdeno / omega ;
            }
            /* (2) degenerate states : state_m != state_n && dEuv -> 0 */
            else if ( fabs(dEuv) < dE_limit && state_m != state_n ){
              dFDdEuvdeno  = k3 / ( ( omega * omega ) + eta2 ) ;
              dFDdEuvdeno2 = dFDdEuvdeno / omega ;
            }
            /* (3) inter-band : state_m != state_n && , dEuv is not close to 0 */
            else{
              dFDdEuvdeno  = dfd_x_kw / ( dEuv * ( ( dEuv_p_omega * dEuv_p_omega ) + eta2 ) );
              dFDdEuvdeno2 = dFDdEuvdeno / dEuv ;
              /* dFDdEuvdeno2 = - dFDdEuvdeno / omega ; */
            }

            /* p2 = p6[j1].i*dEuv_p_omega; */
            p3 = p1 - p6[j1].i*dEuv_p_omega ;
            p4 = p6[j1].r*dEuv_p_omega + p5 ;

            /* real part of conductivity tensor */
            cd_tensor[spin_index][i][j][k].r += dFDdEuvdeno  * p3 ;
            /* imaginery part of conductivity tensor */
            cd_tensor[spin_index][i][j][k].i += dFDdEuvdeno  * p4 ;
            /* real part of dielectric function */
            df_tensor[spin_index][i][j][k].r += dFDdEuvdeno2 * p4 ;
            /* imaginery part of dielectric function */
            df_tensor[spin_index][i][j][k].i -= dFDdEuvdeno2 * p3 ;

          } /* frequency */
        } /* tensor index, j = beta */
      } /* tensor index, i = alpha */
    }
  } /* state_m and state_n */

  for (i=0;i<tnos;i++) free(is_dE_within_range[i]);
  free(is_dE_within_range);
}

void Calc_band_optical_noncol_1(double kx,double ky,double kz,int n,double* EIGEN, dcomplex** LCAO, double* fd_dist,double ChemP){
  double k2,k3,omega,p1,p5,p2,p3,p4,kRn,sx;
  double dFD,dEuv,dEuv_p_omega,dFDdEuvdeno,dFDdEuvdeno2;
  double dE_limit = 1.0E-6; /* for intra-band and degenerate states */
  int Mc_AN,Gc_AN,tno0,tno1,h_AN,Gh_AN;
  /* int Cwan,Hwan; */
  int atom1,atom2,Rn,l1,l2,l3,o,sp,state_m,state_n;
  int tnos; /* total number of states */
  int total_number_of_orbitals = n*0.5; /* for spin-up and spin-down */

  /* p1 = CDDF_FWHM*5/eV2Hartree; */ /* range of extended frequency, outside of spectrum. */
  p1 = CDDF_AddMaxE/eV2Hartree;
  tnos = CDDF_max_unoccupied_state+1;

  /* create array to check whether the state does require to calculate */
  int** is_dE_within_range = (int**)malloc(sizeof(int*)*tnos);
  for (i=0;i<tnos;i++) is_dE_within_range[i] = (int*)malloc(sizeof(int)*tnos);

  /* check whether the state does require to calculate */
  for (state_m=0;state_m<tnos;state_m++)
    for (state_n=0;state_n<tnos;state_n++){
      k3 = fabs( EIGEN[state_m+1] - EIGEN[state_n+1] ) ; /* dE */
      if (CDDF_material_type==0){ /* for insulator */
        if ( k3 <= range_Ha + p1 ){ /* for metal - Check (1) is it within energy ragne? */
          is_dE_within_range[state_m][state_n]=1; /* Decide whether it needs to calculate MME from state_m to state_n or not. */
        }else{
          is_dE_within_range[state_m][state_n]=0;
        }
      }else if (CDDF_material_type==1){ /* for metal */
        if ( k3 <= range_Ha + p1 && (fd_dist[state_n] - fd_dist[state_m]) != 0.0 ){ /* for insulator - Check (1) is it within energy ragne and (2) difference of Fermi-Dirace distribution !=0 */
          is_dE_within_range[state_m][state_n]=1; /* Decide whether it needs to calculate MME from state_m to state_n or not. */
        }else{
          is_dE_within_range[state_m][state_n]=0;
        }
      }
    }

  /* ### MPI for bands ### */
  unsigned int n0 = tnos*tnos, n1 = n0/numprocs2, n2 = n0%numprocs2, n3[numprocs2], n5[numprocs2+1], ui, ul, uk; /* for step 1 */

  if (numprocs2>1){
    for (ui=0;ui<numprocs2;ui++){
      n3[ui]=n1;
      if (ui<n2) n3[ui]++;
    }

    uk=0; ul=n3[0]; n5[0]=0; n5[1]=ul;
    for (ui=0;ui<n0;ui++){
      if (ui>=ul){
        uk++;
        ul+=n3[uk];
        n5[uk+1]=ul;
      }
    }
  }else{
    n5[0]=0; n5[1]=n0;
    n3[0]=n0;
  }

  /* initial atomic orbital index */
  int atomic_orbital_index[2][atomnum];
  atomic_orbital_index[0][0]=0; /* for spin-up */
  atomic_orbital_index[1][0]=total_number_of_orbitals; /* for spin-down */
  i=0;
  for (Gc_AN=1;Gc_AN<atomnum;Gc_AN++){
    /*Cwan = WhatSpecies[Gc_AN]; */
    /*tno0 = Spe_Total_CNO[Cwan]; */
    tno0 = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];
    i += tno0;
    atomic_orbital_index[0][Gc_AN]=i; /* for spin-up */
    atomic_orbital_index[1][Gc_AN]=i+total_number_of_orbitals; /* for spin-down */
  }

  dcomplex MME_x_cuj_x_pf[3],MME_x_cui_x_cuj_x_pf[3]; /* 3 => xyz */
  dcomplex cuj_x_pf,conjg_lcao_coef,p6[9],phase_factor;

  for (sp=0;sp<total_spins;sp++){ /* for spin-up and spin-down */
    for (ul=n5[myid2];ul<n5[myid2+1];ul++){ /* occupied state u */
    /* for (l=0;l<n0;l++){ */ /* occupied state u */
      state_m = ul/tnos; /* occupied state */
      state_n = ul%tnos; /* unoccupied state */

      if (is_dE_within_range[state_m][state_n]==1){ // if dE is within range in Hartree */

        /* initialize */
        double MME[6]={0.0,0.0,0.0,0.0,0.0,0.0}; // for saving momentum matrix elements, i.e. (x_re,y_re,z_re,x_im,y_im,z_im) */

        for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){ // the third loop  - index of the first atoms within primitive cell */
          /* Cwan = WhatSpecies[Gc_AN]; */
          tno0 = Spe_Total_CNO[ WhatSpecies[Gc_AN] ];

          atom1=Gc_AN-1;

          for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){ /* the fourth loop - index of the second atoms within primtive cell */
            Gh_AN = natn[Gc_AN][h_AN];
            /*Hwan = WhatSpecies[Gh_AN]; */
            tno1 = Spe_Total_CNO[ WhatSpecies[Gh_AN] ];

            atom2=Gh_AN-1;

            Rn=ncn[Gc_AN][h_AN]; l1 = atv_ijk[Rn][1]; l2 = atv_ijk[Rn][2]; l3 = atv_ijk[Rn][3];

            kRn = kx*(double)l1 + ky*(double)l2 + kz*(double)l3;

            /* phase factor */
            /* double si = sin(PIx2*kRn),co = cos(PIx2*kRn); */ /* for cluster si = 0 , phase_factor = Complex (1,0) */
            phase_factor = Complex( cos(PIx2*kRn) , sin(PIx2*kRn) );

            MME_x_cui_x_cuj_x_pf[0]=Complex(0.0,0.0);
            MME_x_cui_x_cuj_x_pf[1]=Complex(0.0,0.0);
            MME_x_cui_x_cuj_x_pf[2]=Complex(0.0,0.0);

            for (k=0; k<tno0; k++){ /* the fifth loop  - orbital index k within atom i */
              MME_x_cuj_x_pf[0]=Complex(0.0,0.0);
              MME_x_cuj_x_pf[1]=Complex(0.0,0.0);
              MME_x_cuj_x_pf[2]=Complex(0.0,0.0);

              cuj_x_pf = Complex(0,0);

              for (o=0; o<tno1; o++){ /* the fixth loop  - orbital index o within atom j */
                /* step (1) */
                cuj_x_pf = Cmul( LCAO[state_n][atomic_orbital_index[sp][atom2]+o], phase_factor ) ; /* LCAO[ states ][ atomic orbitals ] */

                /* step (2) */
                MME_x_cuj_x_pf[0] = Cadd( MME_x_cuj_x_pf[0], RCmul( MME_allorb[Gc_AN][h_AN][k][o][0], cuj_x_pf ) ) ;
                MME_x_cuj_x_pf[1] = Cadd( MME_x_cuj_x_pf[1], RCmul( MME_allorb[Gc_AN][h_AN][k][o][1], cuj_x_pf ) ) ;
                MME_x_cuj_x_pf[2] = Cadd( MME_x_cuj_x_pf[2], RCmul( MME_allorb[Gc_AN][h_AN][k][o][2], cuj_x_pf ) ) ;
              } /* 8th loop for orbital index o of atom j */

              /* step (3) */
              conjg_lcao_coef = Conjg( LCAO[state_m][atomic_orbital_index[sp][atom1]+k] ); /* LCAO[ states ][ atomic orbitals ] */

              MME_x_cui_x_cuj_x_pf[0] = Cadd( MME_x_cui_x_cuj_x_pf[0], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[0] ) );
              MME_x_cui_x_cuj_x_pf[1] = Cadd( MME_x_cui_x_cuj_x_pf[1], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[1] ) );
              MME_x_cui_x_cuj_x_pf[2] = Cadd( MME_x_cui_x_cuj_x_pf[2], Cmul( conjg_lcao_coef, MME_x_cuj_x_pf[2] ) );
            } /* 7th loop for orbital index k of atom i */

            /* step (4) : momentum matrix elements - MME[ state_m ][ state_n ][ i ] */
            MME[0] += MME_x_cui_x_cuj_x_pf[0].i ;
            MME[1] += MME_x_cui_x_cuj_x_pf[1].i ;
            MME[2] += MME_x_cui_x_cuj_x_pf[2].i ;
            MME[3] -= MME_x_cui_x_cuj_x_pf[0].r ;
            MME[4] -= MME_x_cui_x_cuj_x_pf[1].r ;
            MME[5] -= MME_x_cui_x_cuj_x_pf[2].r ; 
          } /* 6th loop for atomic index j */
        } /* 5th loop for atomic index i */
        /* ### end of MME from state m to state n ### */

        double dfd_x_kw = fd_dist[state_n] - fd_dist[state_m];
        dEuv = EIGEN[state_m+1] - EIGEN[state_n+1] ;

        p6[0] = Complex( MME[0]*MME[0] + MME[3]*MME[3] , 0.0 );
        p6[1] = Complex( MME[0]*MME[1] + MME[3]*MME[4] , MME[3]*MME[1] - MME[0]*MME[4] );
        p6[2] = Complex( MME[0]*MME[2] + MME[3]*MME[5] , MME[3]*MME[2] - MME[0]*MME[5] );
        p6[3] = Complex( MME[1]*MME[0] + MME[4]*MME[3] , -p6[1].i );
        p6[4] = Complex( MME[1]*MME[1] + MME[4]*MME[4] , 0.0 );
        p6[5] = Complex( MME[1]*MME[2] + MME[4]*MME[5] , MME[4]*MME[2] - MME[1]*MME[5] );
        p6[6] = Complex( MME[2]*MME[0] + MME[5]*MME[3] , MME[5]*MME[0] - MME[2]*MME[3] );
        p6[7] = Complex( MME[2]*MME[1] + MME[5]*MME[4] , -p6[5].i );
        p6[8] = Complex( MME[2]*MME[2] + MME[5]*MME[5] , 0.0 );

        sx = (EIGEN[state_m+1] - ChemP) * Beta;
        if (100<sx) sx = 100.0; 
        k2 = exp(sx);
        k3 = - k2 * Beta / ( (1+k2) * (1+k2) ) ;

        for (i=0;i<d;i++){ /* tensor index, i = alpha */
          for (j=0;j<d;j++){ /* tensor index, j = beta */

            int j1 = i*3+j;
            /* if ((p6[j1].r)==0 && p6[j1].i==0) continue; */
            p1 = p6[j1].r*eta ;
            p5 = p6[j1].i*eta ;

            for (k=0;k<CDDF_freq_grid_total_number;k++){ /* scan all frequencies */

              omega=(k+1)*step;
              dEuv_p_omega = dEuv + omega ;
           
              /* At the same k-point */
              /* (1) intra-band : state_m == state_n ( dEuv == 0 ) */
              if ( fabs(dEuv) < dE_limit && state_m == state_n ){
                dFDdEuvdeno  = k3 / ( ( omega * omega ) + eta2 ) ;
                dFDdEuvdeno2 = dFDdEuvdeno / omega ;
              }
              /* (2) degenerate states : state_m != state_n && dEuv -> 0 */
              else if ( fabs(dEuv) < dE_limit && state_m != state_n ){
                dFDdEuvdeno  = k3 / ( ( omega * omega ) + eta2 ) ;
                dFDdEuvdeno2 = dFDdEuvdeno / omega ;
              }
              /* (3) inter-band : state_m != state_n && , dEuv is not close to 0 */
              else{
                dFDdEuvdeno  = dfd_x_kw / ( dEuv * ( ( dEuv_p_omega * dEuv_p_omega ) + eta2 ) );
                dFDdEuvdeno2 = dFDdEuvdeno / dEuv ;
                /* dFDdEuvdeno2 = - dFDdEuvdeno / omega ; */
              }

              /* p2 = p6[j1].i*dEuv_p_omega; */
              p3 = p1 - p6[j1].i*dEuv_p_omega ;
              p4 = p6[j1].r*dEuv_p_omega + p5 ;

              /* real part of conductivity tensor */
              cd_tensor[sp][i][j][k].r += dFDdEuvdeno  * p3 ;
              /* imaginery part of conductivity tensor */
              cd_tensor[sp][i][j][k].i += dFDdEuvdeno  * p4 ;
              /* real part of dielectric function */
              df_tensor[sp][i][j][k].r += dFDdEuvdeno2 * p4 ;
              /* imaginery part of dielectric function */
              df_tensor[sp][i][j][k].i -= dFDdEuvdeno2 * p3 ;
            } /* frequency */
          } /* tensor index, j = beta */
        } /* tensor index, i = alpha */
      }
    } /* state_m and state_n */
  } /* spin */

  for (i=0;i<tnos;i++) free(is_dE_within_range[i]);
  free(is_dE_within_range);
}


void Calc_optical_col_2(int n,double sum_weights){
  Free_optical_1(); /* free Nabra matrix elements, i.e. MME_allorb array. */

  int numprocs,myid,sp,o;
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* if (myid==Host_ID) printf("The highest state = %i , total number of states = %i\n",CDDF_max_unoccupied_state,n); */

  double q2=conductivity_unit/(Cell_Volume*sum_weights), q=q2/dielectricless_unit; /* unit */
  double collected_temp_tensor[CDDF_freq_grid_total_number*4],temp_tensor[CDDF_freq_grid_total_number*4];

  l = 2*CDDF_freq_grid_total_number; 
  o = l + CDDF_freq_grid_total_number;

  for (sp=0;sp<total_spins;sp++){ /* tensor index, i = alpha */
    for (i=0;i<d;i++){ /* tensor index, i = alpha */
      for (j=0;j<d;j++){ /* tensor index, j = beta */

        for (k=0; k<CDDF_freq_grid_total_number; k++){ /* scan all frequencies */

          /* save conductivity tensor */
          cd_tensor[sp][i][j][k].r *= q2 ; /* conductivity unit = (Ohm m)^{-1} */
          cd_tensor[sp][i][j][k].i *= q2 ; /* conductivity unit = (Ohm m)^{-1} */

          /* calculate dielectric function */
          /* the second term */
          df_tensor[sp][i][j][k].r *= q ;
          df_tensor[sp][i][j][k].i *= q ;

          /* the first term */
          /*if (i==j) df_tensor[kp][i][j][k].r += 1; */

          /* store data at each cpus, except Host_ID */
          collected_temp_tensor[k                            ] = 0.0;
          collected_temp_tensor[CDDF_freq_grid_total_number+k] = 0.0;
          collected_temp_tensor[l+k                          ] = 0.0;
          collected_temp_tensor[o+k                          ] = 0.0;
          temp_tensor[k                            ] = cd_tensor[sp][i][j][k].r;
          temp_tensor[CDDF_freq_grid_total_number+k] = cd_tensor[sp][i][j][k].i;
          temp_tensor[l+k                          ] = df_tensor[sp][i][j][k].r;
          temp_tensor[o+k                          ] = df_tensor[sp][i][j][k].i;
        }

        MPI_Reduce(&temp_tensor, &collected_temp_tensor, CDDF_freq_grid_total_number*4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_level1);

        /*restore data of conductivity and dielectric function */
        for (k=0;k<CDDF_freq_grid_total_number;k++){
          cd_tensor[sp][i][j][k] = Complex(collected_temp_tensor[  k],collected_temp_tensor[CDDF_freq_grid_total_number+k]);
          df_tensor[sp][i][j][k] = Complex(collected_temp_tensor[l+k],collected_temp_tensor[o+k]);
        }

        /* linear extrapolation at omega = 0 for conductivity and dielectric function (and assume x1-x0 = x2-x1) */
        /* y(0) = y1 + (y2-y1)(0-x1)/(x2-x1) = y1 + (y1-y2)*x1/(x2-x1) = y1 + (y1-y2)*dx/dx = y1 + (y1-y2)*1 = y1 + (y1-y2) = 2*y1 - y2 */
        cd_tensor_omega0[sp][i][j].r = 2.0*cd_tensor[sp][i][j][0].r - cd_tensor[sp][i][j][1].r;
        cd_tensor_omega0[sp][i][j].i = 2.0*cd_tensor[sp][i][j][0].i - cd_tensor[sp][i][j][1].i;
        df_tensor_omega0[sp][i][j].r = 2.0*df_tensor[sp][i][j][0].r - df_tensor[sp][i][j][1].r;
        df_tensor_omega0[sp][i][j].i = 2.0*df_tensor[sp][i][j][0].i - df_tensor[sp][i][j][1].i;
      } /* tensor index, j = beta */
    } /* tensor index, i = alpha */
  }

  /* print out */
  if (myid==Host_ID) Print_optical(1,1,1);
  Free_optical_2(n);
}


void Calc_optical_noncol_2(int n,double sum_weights){
  Free_optical_1();

  int numprocs,myid,sp,o;
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  double q2=conductivity_unit/(Cell_Volume*sum_weights), q=q2/dielectricless_unit; /* unit */
  double collected_temp_tensor[CDDF_freq_grid_total_number*4],temp_tensor[CDDF_freq_grid_total_number*4];

  l=2*CDDF_freq_grid_total_number; o=l+CDDF_freq_grid_total_number;
  for (sp=0;sp<total_spins;sp++){ /* tensor index, i = alpha */
    for (i=0;i<d;i++){ /* tensor index, i = alpha */
      for (j=0;j<d;j++){ /* tensor index, j = beta */
        for (k=0;k<CDDF_freq_grid_total_number;k++){ /* scan all frequencies */
          /* save conductivity tensor */
          cd_tensor[sp][i][j][k].r *= q2 ; /* conductivity unit = (Ohm m)^{-1} */
          cd_tensor[sp][i][j][k].i *= q2 ; /* conductivity unit = (Ohm m)^{-1} */

          /* calculate dielectric function */
          /* the second term */
          df_tensor[sp][i][j][k].r *= q ;
          df_tensor[sp][i][j][k].i *= q ;

          /* the first term */
          /*if (i==j) df_tensor[kp][i][j][k].r += 1; */

          /* store data at each cpus, except Host_ID */
          collected_temp_tensor[k]=0.0;
          collected_temp_tensor[CDDF_freq_grid_total_number+k]=0.0;
          collected_temp_tensor[l+k]=0.0;
          collected_temp_tensor[o+k]=0.0;
          temp_tensor[  k] = cd_tensor[sp][i][j][k].r;
          temp_tensor[CDDF_freq_grid_total_number+k] = cd_tensor[sp][i][j][k].i;
          temp_tensor[l+k] = df_tensor[sp][i][j][k].r;
          temp_tensor[o+k] = df_tensor[sp][i][j][k].i;
        }

        MPI_Reduce(&temp_tensor, &collected_temp_tensor, CDDF_freq_grid_total_number*4, MPI_DOUBLE, MPI_SUM, 0, mpi_comm_level1);

        /*restore data of conductivity and dielectric function */
        for (k=0;k<CDDF_freq_grid_total_number;k++){
          cd_tensor[sp][i][j][k] = Complex(collected_temp_tensor[k],collected_temp_tensor[CDDF_freq_grid_total_number+k]);
          df_tensor[sp][i][j][k] = Complex(collected_temp_tensor[l+k],collected_temp_tensor[o+k]);
        }

        /* linear extrapolation at omega = 0 for conductivity and dielectric function (and assume x1-x0 = x2-x1) */
        /* y(0) = y1 + (y2-y1)(0-x1)/(x2-x1) = y1 + (y1-y2)*x1/(x2-x1) = y1 + (y1-y2)*dx/dx = y1 + (y1-y2)*1 = y1 + (y1-y2) = 2*y1 - y2 */
        cd_tensor_omega0[sp][i][j].r = 2.0*cd_tensor[sp][i][j][0].r - cd_tensor[sp][i][j][1].r;
        cd_tensor_omega0[sp][i][j].i = 2.0*cd_tensor[sp][i][j][0].i - cd_tensor[sp][i][j][1].i;
        df_tensor_omega0[sp][i][j].r = 2.0*df_tensor[sp][i][j][0].r - df_tensor[sp][i][j][1].r;
        df_tensor_omega0[sp][i][j].i = 2.0*df_tensor[sp][i][j][0].i - df_tensor[sp][i][j][1].i;
      } /* tensor index, j = beta */
    } /* tensor index, i = alpha */
  }

  if (myid==Host_ID) Print_optical(1,1,1); /* (print CD=1=yes, print DF=1=yes, print optical properties=1=yes) */
  Free_optical_2(n);
}
