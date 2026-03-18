/**********************************************************************
  gcube2oned.c

   gcube2oned.c is a program which transforms the data of 3D cube data
   to an 1D data along a chosen direction by integrating over the 
   remaining 2D. 

      Usage:

         ./gcube2oned input.cube 

  Log of gcube2onede.c:

     05/Oct./2019  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define fp_bsize         1048576     /* buffer size for setvbuf */
#define Hartree2eV      27.2113845                
#define BohrR           0.529177249              /* Angstrom       */

int main(int argc, char *argv[]) 
{
  int i,j,itmp,n1,n2,n3,po,axis;
  int atomnum1,atomnum2;
  int Ngrid1_1,Ngrid1_2,Ngrid1_3;
  int Ngrid2_1,Ngrid2_2,Ngrid2_3;
  double **Gxyz1,**Gxyz2;
  double ***CubeData1,***CubeData2;
  double Grid_Origin1[4];
  double Grid_Origin2[4];
  double gtv1[4][4],gtv2[4][4];
  double dtmp,step;
  char ctmp[100];
  char buf[1000],buf2[1000],*c;
  FILE *fp1,*fp2;
  char fp_buf[fp_bsize];          /* setvbuf */

  if (argc!=3){
    printf("Usage:\n");
    printf("  ./gcube2oned input.cube axis_number\n");
    exit(0);
  }

  axis = atoi(argv[2]); 

  /*******************************************
               read the first file 
  *******************************************/

  if ((fp1 = fopen(argv[1],"r")) != NULL){

    setvbuf(fp1,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */

    /* scanf cube tile */

    fscanf(fp1,"%s",ctmp);
    fscanf(fp1,"%s",ctmp);
    fscanf(fp1,"%d",&atomnum1);
    fscanf(fp1,"%lf %lf %lf",&Grid_Origin1[1],&Grid_Origin1[2],&Grid_Origin1[3]);
    fscanf(fp1,"%d",&Ngrid1_1);
    fscanf(fp1,"%lf %lf %lf",&gtv1[1][1],&gtv1[1][2],&gtv1[1][3]);
    fscanf(fp1,"%d",&Ngrid1_2);
    fscanf(fp1,"%lf %lf %lf",&gtv1[2][1],&gtv1[2][2],&gtv1[2][3]);
    fscanf(fp1,"%d",&Ngrid1_3);
    fscanf(fp1,"%lf %lf %lf",&gtv1[3][1],&gtv1[3][2],&gtv1[3][3]);

    /* allocation of arrays */

    Gxyz1 = (double**)malloc(sizeof(double*)*(atomnum1+1)); 
    for (i=0; i<(atomnum1+1); i++){
      Gxyz1[i] = (double*)malloc(sizeof(double)*4); 
    }

    CubeData1 = (double***)malloc(sizeof(double**)*Ngrid1_1); 
    for (i=0; i<Ngrid1_1; i++){
      CubeData1[i] = (double**)malloc(sizeof(double*)*Ngrid1_2); 
      for (j=0; j<Ngrid1_2; j++){
        CubeData1[i][j] = (double*)malloc(sizeof(double)*Ngrid1_3); 
      }
    }

    /* scanf xyz coordinates */

    for (i=1; i<=atomnum1; i++){
      fscanf(fp1,"%lf %lf %lf %lf %lf",&Gxyz1[i][0],&dtmp,&Gxyz1[i][1],&Gxyz1[i][2],&Gxyz1[i][3]);
    }

    /* scanf cube data */

    for (n1=0; n1<Ngrid1_1; n1++){
      for (n2=0; n2<Ngrid1_2; n2++){
        for (n3=0; n3<Ngrid1_3; n3++){
          fscanf(fp1,"%lf",&CubeData1[n1][n2][n3]);
	}
      }
    }

    fclose(fp1);
  }
  else{
    printf("error in scanfing %s\n",argv[1]);
  }

  /*******************************************
                make a file 
  *******************************************/

  step = BohrR*sqrt(gtv1[axis][1]*gtv1[axis][1]+gtv1[axis][2]*gtv1[axis][2]+gtv1[axis][3]*gtv1[axis][3]);

  if (axis==1){

    for (n1=0; n1<Ngrid1_1; n1++){
      double sum = 0.0;
      for (n2=0; n2<Ngrid1_2; n2++){
	for (n3=0; n3<Ngrid1_3; n3++){
	  sum += CubeData1[n1][n2][n3];
	}
      }
      printf("%5d %18.15f %18.15f\n",n1,step*(double)n1,sum/(double)(n2*n3));
    }
  }

  else if (axis==2){

    for (n2=0; n2<Ngrid1_2; n2++){
      double sum = 0.0;
      for (n1=0; n1<Ngrid1_1; n1++){
	for (n3=0; n3<Ngrid1_3; n3++){
	  sum += CubeData1[n1][n2][n3];
	}
      }
      printf("%5d %18.15f %18.15f\n",n1,step*(double)n2,sum/(double)(n1*n3));
    }
  }

  else if (axis==3){

    for (n3=0; n3<Ngrid1_3; n3++){
      double sum = 0.0;
      for (n1=0; n1<Ngrid1_1; n1++){
	for (n2=0; n2<Ngrid1_2; n2++){
	  sum += CubeData1[n1][n2][n3];
	}
      }
      printf("%5d %18.15f %18.15f\n",n1,step*(double)n3,sum/(double)(n1*n2));
    }
  }

  /* freeing of arrays */

  for (i=0; i<(atomnum1+1); i++){
    free(Gxyz1[i]);
  }
  free(Gxyz1);

  for (i=0; i<Ngrid1_1; i++){
    for (j=0; j<Ngrid1_2; j++){
      free(CubeData1[i][j]);
    }
    free(CubeData1[i]);
  }
  free(CubeData1);
}




