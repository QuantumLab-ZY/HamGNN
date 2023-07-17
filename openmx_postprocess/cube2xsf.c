/**********************************************************************
  cube2xsf.c:

  Log of cube2xsf.c:

    30/Jun/2015  Released by Mitsuaki Kawamura 
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define BohrR           0.529177249              /* Angstrom       */

static void cube2xsf_bin(char *fname1);
static void cube2xsf(char *fname1);

int main(int argc, char *argv[]) 
{
  int i;

  /* check the number of arguments */

  if (argc<2){
    printf("\nUsage : \n");
    printf("$ cube2xsf {prefix1}.cube {prefix2}.cube {prefix3}.cube ... \n");
    printf("$ cube2xsf {prefix1}.cube.bin {prefix2}.cube.bin, {prefix3}.cube.bin ... \n\n");
    printf("Then {prefix1}.xsf, {prefix2}.xsf, {prefix3}.xsf ... are generated. \n\n");
    exit(0);
  } 

  /* read each file */

  for (i=1; i<argc; i++){

    if (strstr(argv[i], ".cube.bin") != NULL){
      printf("converting %s\n", argv[i]);
      cube2xsf_bin(argv[i]);
    }
    else if (strstr(argv[i], ".cube") != NULL){
      printf("converting %s\n", argv[i]);
      cube2xsf(argv[i]);
    }

  }

}

static void cube2xsf(char *fname1)
{
  FILE *fp1, *fp2;
  int i, j, k, ii, jj, kk, n, ct_AN, atomnum, Ngrid1, Ngrid2, Ngrid3, spe;
  double Grid_Origin[4], gtv[4][4], Gxyz[4], charge;
  char clist[300];
  char fname2[300];
  char *p;
  double *dlist;

  i = 0;
  for (p = &fname1[0]; p<strstr(fname1, ".cube"); p++) fname2[i++] = *p;
  sprintf(&fname2[i],".xsf");

  if ((fp1 = fopen(fname1, "r")) != NULL){

    if ((fp2 = fopen(fname2, "w")) != NULL){

      fgets(clist, 300, fp1);
      fprintf(fp2, "#%s", clist);
      fgets(clist, 300, fp1);
      fprintf(fp2, "#%s", clist);
      fprintf(fp2, "CRYSTAL\n");
      fprintf(fp2, "PRIMVEC\n");

      fscanf(fp1, "%d %lf %lf %lf", &atomnum, &Grid_Origin[1], &Grid_Origin[2], &Grid_Origin[3]);
      fscanf(fp1, "%d %lf %lf %lf", &Ngrid1, &gtv[1][1], &gtv[1][2], &gtv[1][3]);
      fscanf(fp1, "%d %lf %lf %lf", &Ngrid2, &gtv[2][1], &gtv[2][2], &gtv[2][3]);
      fscanf(fp1, "%d %lf %lf %lf", &Ngrid3, &gtv[3][1], &gtv[3][2], &gtv[3][3]);

      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid3 * gtv[3][1] * BohrR,
        (double)Ngrid3 * gtv[3][2] * BohrR,
        (double)Ngrid3 * gtv[3][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid2 * gtv[2][1] * BohrR,
        (double)Ngrid2 * gtv[2][2] * BohrR,
        (double)Ngrid2 * gtv[2][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid1 * gtv[1][1] * BohrR,
        (double)Ngrid1 * gtv[1][2] * BohrR,
        (double)Ngrid1 * gtv[1][3] * BohrR);

      fprintf(fp2, "PRIMCOORD\n");
      fprintf(fp2, "%2d  1\n", atomnum);

      /* fread Gxyz and fprintf them */

      for (ct_AN = 1; ct_AN <= atomnum; ct_AN++){
        fscanf(fp1, "%d %lf %lf %lf %lf", &spe, &charge, &Gxyz[1], &Gxyz[2], &Gxyz[3]);
        fprintf(fp2, "%3d  %12.6lf%12.6lf%12.6lf\n", spe, 
          (Gxyz[1] - Grid_Origin[1]) * BohrR,
          (Gxyz[2] - Grid_Origin[2]) * BohrR,
          (Gxyz[3] - Grid_Origin[3]) * BohrR);
      }

      /* Print infomation of the volme data Grid */

      fprintf(fp2, "BEGIN_BLOCK_DATAGRID_3D\n");
      fprintf(fp2, "Transmission eigenchannel\n");
      fprintf(fp2, "DATAGRID_3D_UNKNOWN\n");

      fprintf(fp2, "%12d %12d %12d\n", Ngrid3 + 1, Ngrid2 + 1, Ngrid1 + 1);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n", 0.0, 0.0, 0.0);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid3 * gtv[3][1] * BohrR,
        (double)Ngrid3 * gtv[3][2] * BohrR,
        (double)Ngrid3 * gtv[3][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid2 * gtv[2][1] * BohrR,
        (double)Ngrid2 * gtv[2][2] * BohrR,
        (double)Ngrid2 * gtv[2][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid1 * gtv[1][1] * BohrR,
        (double)Ngrid1 * gtv[1][2] * BohrR,
        (double)Ngrid1 * gtv[1][3] * BohrR);

      /* fread data on grid and fprintf them */

      dlist = (double*)malloc(sizeof(double)*Ngrid1*Ngrid2*Ngrid3);

      for (i = 0; i < Ngrid1 * Ngrid2 * Ngrid3; i++){
        fscanf(fp1, "%lf", &dlist[i]);
      }

      for (i = 0; i <= Ngrid1; i++){
        for (j = 0; j <= Ngrid2; j++){
          for (k = 0; k <= Ngrid3; k++){

            ii = i % Ngrid1;
            jj = j % Ngrid2;
            kk = k % Ngrid3;

            n = ii * Ngrid2 * Ngrid3 + jj * Ngrid3 + kk;
            fprintf(fp2, "%13.3E", dlist[n]);
            if ((k + 1) % 6 == 0) { fprintf(fp2, "\n"); }
          } /* for (k = 0; k <= Ngrid3; k++) */
          /* avoid double \n\n when Ngrid3%6 == 0  */
          if ((Ngrid3 + 1) % 6 != 0) fprintf(fp2, "\n");
        } /* for (j = 0; j <= Ngrid2; j++) */
      } /* for (i = 0; i <= Ngrid1; i++) */

      free(dlist);

      fprintf(fp2, "END_DATAGRID_3D \n");
      fprintf(fp2, "END_BLOCK_DATAGRID_3D \n");

      fclose(fp2);
    }

    fclose(fp1);

    sprintf(clist, "rm %s", fname1);
    /*system(clist);*/
  }
  else{
    printf("Failure of reading %s.\n", fname1);
    exit(0);
  }
}



static void cube2xsf_bin(char *fname1)
{
  FILE *fp1, *fp2;
  int i, j, k, ii, jj, kk, n, ct_AN, atomnum, Ngrid1, Ngrid2, Ngrid3;
  double Grid_Origin[4], gtv[4][4];
  char clist[300];
  char fname2[300];
  char *p;
  double *dlist;

  i = 0;
  for (p = &fname1[0]; p<strstr(fname1, ".cube.bin"); p++) fname2[i++] = *p;
  sprintf(&fname2[i], ".xsf");

  if ((fp1 = fopen(fname1, "rb")) != NULL){

    if ((fp2 = fopen(fname2, "w")) != NULL){

      fread(clist, sizeof(char), 200, fp1);
      fprintf(fp2, "#%s", clist);
      fread(clist, sizeof(char), 200, fp1);
      fprintf(fp2, "#%s", clist);
      fprintf(fp2, "CRYSTAL\n");
      fprintf(fp2, "PRIMVEC\n");

      fread(&atomnum, sizeof(int), 1, fp1);
      fread(&Grid_Origin[1], sizeof(double), 3, fp1);
      fread(&Ngrid1, sizeof(int), 1, fp1);
      fread(&gtv[1][1], sizeof(double), 3, fp1);
      fread(&Ngrid2, sizeof(int), 1, fp1);
      fread(&gtv[2][1], sizeof(double), 3, fp1);
      fread(&Ngrid3, sizeof(int), 1, fp1);
      fread(&gtv[3][1], sizeof(double), 3, fp1);

      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid3 * gtv[3][1] * BohrR,
        (double)Ngrid3 * gtv[3][2] * BohrR,
        (double)Ngrid3 * gtv[3][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid2 * gtv[2][1] * BohrR,
        (double)Ngrid2 * gtv[2][2] * BohrR,
        (double)Ngrid2 * gtv[2][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid1 * gtv[1][1] * BohrR,
        (double)Ngrid1 * gtv[1][2] * BohrR,
        (double)Ngrid1 * gtv[1][3] * BohrR);

      fprintf(fp2, "PRIMCOORD\n");
      fprintf(fp2, "%2d  1\n", atomnum);

      /* fread Gxyz and fprintf them */

      dlist = (double*)malloc(sizeof(double)*atomnum * 5);
      fread(dlist, sizeof(double), atomnum * 5, fp1);

      n = 0;
      for (ct_AN = 1; ct_AN <= atomnum; ct_AN++){
        fprintf(fp2, "%3d  %12.6lf%12.6lf%12.6lf\n",
          (int)dlist[n],
          (dlist[n + 2] - Grid_Origin[1]) * BohrR,
          (dlist[n + 3] - Grid_Origin[2]) * BohrR,
          (dlist[n + 4] - Grid_Origin[3]) * BohrR);
        n += 5;
      }

      free(dlist);

      /* Print infomation of the volme data Grid */

      fprintf(fp2, "BEGIN_BLOCK_DATAGRID_3D\n");
      fprintf(fp2, "Transmission eigenchannel\n");
      fprintf(fp2, "DATAGRID_3D_UNKNOWN\n");

      fprintf(fp2, "%12d %12d %12d\n", Ngrid3 + 1, Ngrid2 + 1, Ngrid1 + 1);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n", 0.0, 0.0, 0.0);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid3 * gtv[3][1] * BohrR, 
        (double)Ngrid3 * gtv[3][2] * BohrR, 
        (double)Ngrid3 * gtv[3][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid2 * gtv[2][1] * BohrR, 
        (double)Ngrid2 * gtv[2][2] * BohrR, 
        (double)Ngrid2 * gtv[2][3] * BohrR);
      fprintf(fp2, "%20.12le%20.12le%20.12le\n",
        (double)Ngrid1 * gtv[1][1] * BohrR, 
        (double)Ngrid1 * gtv[1][2] * BohrR, 
        (double)Ngrid1 * gtv[1][3] * BohrR);

      /* fread data on grid and fprintf them */

      dlist = (double*)malloc(sizeof(double)*Ngrid1*Ngrid2*Ngrid3);

      fread(dlist, sizeof(double), Ngrid1 * Ngrid2 * Ngrid3, fp1);

      for (i = 0; i <= Ngrid1; i++){
        for (j = 0; j <= Ngrid2; j++){
          for (k = 0; k <= Ngrid3; k++){

            ii = i % Ngrid1;
            jj = j % Ngrid2;
            kk = k % Ngrid3;

            n = ii * Ngrid2 * Ngrid3 + jj * Ngrid3 + kk;
            fprintf(fp2, "%13.3E", dlist[n]);
            if ((k + 1) % 6 == 0) { fprintf(fp2, "\n"); }
          } /* for (k = 0; k <= Ngrid3; k++) */
          /* avoid double \n\n when Ngrid3%6 == 0  */
          if ((Ngrid3 + 1) % 6 != 0) fprintf(fp2, "\n");
        } /* for (j = 0; j <= Ngrid2; j++) */
      } /* for (i = 0; i <= Ngrid1; i++) */

      free(dlist);

      fprintf(fp2, "END_DATAGRID_3D \n");
      fprintf(fp2, "END_BLOCK_DATAGRID_3D \n");

      fclose(fp2);
    }

    fclose(fp1);

    sprintf(clist, "rm %s", fname1);
    /* system(clist); */
  }
  else{
    printf("Failure of reading %s.\n", fname1);
    exit(0);
  }
}

