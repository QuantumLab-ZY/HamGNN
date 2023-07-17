#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"

void TRAN_Print_Grid_Cell1(
  char *filename,
  int n1, int n2, int n3, 
   int *My_Cell1,
   double *realgrid
)
{
  FILE *fp;
  int i,j,k,ie;
  int MN;
  double val;
  static double eps=1.0e-7;
    fp = fopen(filename,"w");
    for (i=0;i<n1;i++)  {
      ie = My_Cell1[i];
      for (j=0;j<n2; j++) {
        for (k=0;k<n3;k++) {
           MN = (i)*n2*n3+(j)*n3+(k);
           val = realgrid[ MN ] ;
          if (fabs(val) > eps) {
          fprintf(fp,"%d %d %d %le\n",ie,j,k, val);
          }
        }
      }

    }
    fclose(fp);

}


void TRAN_Print_Grid_Cell0(
   char *filename,
   double origin[4], /* use 1:3 */
   double gtv[4][4], /* use [1:3][1:3] */
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,  /* 0:Ngrid1, 0:Ngrid2, Ngrid3s:Ngrid3e */
   double  R[4],  /* use 1:3 */
   int *Cell0,
   double *grid_value
)
{
  int i,j,k;
  int ie;

   FILE *fp;

   printf("TRAN_Print_Grid outputfile=%s\n",filename);

   fp=fopen(filename,"w");

   fprintf(fp,"%lf %lf %lf\n",
          origin[1], origin[2], origin[3]);

   for (i=1;i<=3;i++) {
   fprintf(fp,"%lf %lf %lf\n",gtv[i][1], gtv[i][2], gtv[i][3]);
   }
   fprintf(fp,"%d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3s,Ngrid3e);
   fprintf(fp,"%lf %lf %lf\n",R[1], R[2], R[3]);


   for (i=0;i<Ngrid1;i++){
       ie =Cell0[i];
      for (j=0;j<Ngrid2;j++) {
        fprintf(fp,"%d %d ",i,j);
        for (k=Ngrid3s; k<=Ngrid3e;k++) {
          fprintf(fp,"%le ", grid_value[Ngrid2*(Ngrid3e-Ngrid3s+1)*ie+(Ngrid3e-Ngrid3s+1)*j+ k-Ngrid3s] );
        }
        fprintf(fp,"\n");
      }
    }

   fclose(fp);
}


void TRAN_Print_Grid(
   char *filename,
   double origin[4], /* use 1:3 */
   double gtv[4][4], /* use [1:3][1:3] */
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,  /* 0:Ngrid1, 0:Ngrid2, Ngrid3s:Ngrid3e */
   double  R[4],  /* use 1:3 */
   double *grid_value 
)
{
  int i,j,k;

   FILE *fp;

   printf("TRAN_Print_Grid outputfile=%s\n",filename);

   fp=fopen(filename,"w");

   fprintf(fp,"%lf %lf %lf\n",
          origin[1], origin[2], origin[3]);

   for (i=1;i<=3;i++) {
   fprintf(fp,"%lf %lf %lf\n",gtv[i][1], gtv[i][2], gtv[i][3]);
   }
   fprintf(fp,"%d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3s,Ngrid3e);
   fprintf(fp,"%lf %lf %lf\n",R[1], R[2], R[3]);


   for (i=0;i<Ngrid1;i++){
      for (j=0;j<Ngrid2;j++) {
        fprintf(fp,"%d %d ",i,j);
        for (k=Ngrid3s; k<=Ngrid3e;k++) {
          fprintf(fp,"%le ", grid_value[Ngrid2*(Ngrid3e-Ngrid3s+1)*i+(Ngrid3e-Ngrid3s+1)*j+ k-Ngrid3s] );
        }
        fprintf(fp,"\n");
      }
    }

   fclose(fp);
}


void TRAN_Print_Grid_z(
   char *filename,
   double origin[4], /* use 1:3 */
   double gtv[4][4], /* use [1:3][1:3] */
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,  /* 0:Ngrid1, 0:Ngrid2, Ngrid3s:Ngrid3e */
   double  R[4],  /* use 1:3 */
   double *grid_value
)
{
  int i,j,k;

   FILE *fp;

   printf("TRAN_Print_Grid outputfile=%s\n",filename);

   fp=fopen(filename,"w");

   fprintf(fp,"%lf %lf %lf\n",
           origin[1], origin[2],origin[3]);


   for (i=1;i<=3;i++) {
   fprintf(fp,"%lf %lf %lf\n",gtv[i][1], gtv[i][2], gtv[i][3]);
   }
   fprintf(fp,"%d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3s,Ngrid3e);
   fprintf(fp,"%lf %lf %lf\n", R[1],R[2], R[3]);


      for (j=0;j<Ngrid2;j++) {
        for (k=Ngrid3s; k<=Ngrid3e;k++) {
        fprintf(fp,"%d %d ",j,k);

          for (i=0;i<Ngrid1;i++){

          fprintf(fp,"%le ", grid_value[Ngrid2*(Ngrid3e-Ngrid3s+1)*i+(Ngrid3e-Ngrid3s+1)*j+ k-Ngrid3s] );
        }
        fprintf(fp,"\n");
      }
    }

   fclose(fp);
}


void TRAN_Print_Grid_c(
   char *filenamer,
   char *filenamei,
   double origin[4], /* use 1:3 */
   double gtv[4][4], /* use [1:3][1:3] */
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,  /* 0:Ngrid1, 0:Ngrid2, Ngrid3s:Ngrid3e */
   double  R[4],  /* use 1:3 */
   dcomplex *grid_value
)
{
  int i,j,k;

   FILE *fp;

   printf("TRAN_Print_Grid outputfile=%s\n",filenamer);

   fp=fopen(filenamer,"w");

   fprintf(fp,"%lf %lf %lf\n",
          origin[1], origin[2], origin[3]);

   for (i=1;i<=3;i++) {
   fprintf(fp,"%lf %lf %lf\n",gtv[i][1], gtv[i][2], gtv[i][3]);
   }
   fprintf(fp,"%d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3s,Ngrid3e);
   fprintf(fp,"%lf %lf %lf\n",R[1], R[2], R[3]);


   for (i=0;i<Ngrid1;i++){
      for (j=0;j<Ngrid2;j++) {
        fprintf(fp,"%d %d ",i,j);
        for (k=Ngrid3s; k<=Ngrid3e;k++) {
          fprintf(fp,"%le ", grid_value[Ngrid2*(Ngrid3e-Ngrid3s+1)*i+(Ngrid3e-Ngrid3s+1)*j+ k-Ngrid3s].r );
        }
        fprintf(fp,"\n");
      }
    }

   fclose(fp);

   /*******************************************************************************************************/

   printf("TRAN_Print_Grid outputfile=%s\n",filenamei);

   fp=fopen(filenamei,"w");

   fprintf(fp,"%lf %lf %lf\n",
          origin[1], origin[2], origin[3]);

   for (i=1;i<=3;i++) {
   fprintf(fp,"%lf %lf %lf\n",gtv[i][1], gtv[i][2], gtv[i][3]);
   }
   fprintf(fp,"%d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3s,Ngrid3e);
   fprintf(fp,"%lf %lf %lf\n",R[1], R[2], R[3]);


   for (i=0;i<Ngrid1;i++){
      for (j=0;j<Ngrid2;j++) {
        fprintf(fp,"%d %d ",i,j);
        for (k=Ngrid3s; k<=Ngrid3e;k++) {
          fprintf(fp,"%le ", grid_value[Ngrid2*(Ngrid3e-Ngrid3s+1)*i+(Ngrid3e-Ngrid3s+1)*j+ k-Ngrid3s].i );
        }
        fprintf(fp,"\n");
      }
    }

   fclose(fp);

}



void TRAN_Print_Grid_v(
   char *filename,
   double origin[4], /* use 1:3 */
   double gtv[4][4], /* use [1:3][1:3] */
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,  /* 0:Ngrid1, 0:Ngrid2, Ngrid3s:Ngrid3e */
   double  R[4],  /* use 1:3 */
   double ***grid_value
)
{
  int i,j,k;
  int je;

   FILE *fp;

   printf("TRAN_Print_Grid outputfile=%s\n",filename);

   fp=fopen(filename,"w");

   fprintf(fp,"%lf %lf %lf\n",
          origin[1], origin[2], origin[3]);

   for (i=1;i<=3;i++) {
   fprintf(fp,"%lf %lf %lf\n",gtv[i][1], gtv[i][2], gtv[i][3]);
   }
   fprintf(fp,"%d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3s,Ngrid3e);
   fprintf(fp,"%lf %lf %lf\n",R[1], R[2], R[3]);


   for (i=0;i<Ngrid1;i++){
      for (j=0;j<Ngrid2;j++) {
        je = j;
        fprintf(fp,"%d %d ",i,j);
        for (k=Ngrid3s; k<=Ngrid3e;k++) {

          fprintf(fp,"%d %d %d %le\n",i,je,k, grid_value[j][i][k]); 

        }
        fprintf(fp,"\n");
      }
    }

   fclose(fp);

}


void TRAN_Print_Grid_Startv(
   char *filename,
   int Ngrid1, int Ngrid2,  int Ngrid3,  /* 0:Ngrid1, 0:Ngrid2, Ngrid3s:Ngrid3e */
   int Start,
   double ***grid_value
)
{
  int i,j,k;
  int je;
  static double eps=1.0e-7;
  FILE *fp;
  double val;

   printf("TRAN_Print_Grid outputfile=%s\n",filename);

   fp=fopen(filename,"w");

   for (i=0;i<Ngrid1;i++){
      for (j=0;j<Ngrid2;j++) {
        je = j+Start;
        for (k=0; k<Ngrid3;k++) {
          val = grid_value[j][i][k];
          if ( fabs(val) > eps) 
          fprintf(fp,"%d %d %d %le\n",i,je,k, val);

        }
      }
    }

   fclose(fp);

}

