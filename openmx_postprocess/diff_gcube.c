/**********************************************************************
  diff_gcube.c

     diff_gcube.c is a program which calculates the difference between
     two densities stored in Gaussian cube files, and output the result
     to a file in the Gaussian cube format.

      Usage:

         ./diff_density input1.cube input2.cube output.cube 

      Definition of difference:

         input1 - input2 = output

  Log of diff_gcube.c:

     19/Apr/2004  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define fp_bsize         1048576     /* buffer size for setvbuf */

int main(int argc, char *argv[]) 
{
  static int i,j,itmp,n1,n2,n3,po;
  static int atomnum1,atomnum2;
  static int Ngrid1_1,Ngrid1_2,Ngrid1_3;
  static int Ngrid2_1,Ngrid2_2,Ngrid2_3;
  static double **Gxyz1,**Gxyz2;
  static double ***CubeData1,***CubeData2;
  static double Grid_Origin1[4];
  static double Grid_Origin2[4];
  static double gtv1[4][4],gtv2[4][4];
  static double dtmp;
  static char ctmp[100];
  char buf[1000],buf2[1000],*c;
  FILE *fp1,*fp2;
  char fp_buf[fp_bsize];          /* setvbuf */

  if (argc!=4){
    printf("Usage:\n");
    printf("  ./diff_density input1.cube input2.cube output.cube\n");
    exit(0);
  }

  /*******************************************
               read the first file 
  *******************************************/

  if ((fp1 = fopen(argv[1],"r")) != NULL){

#ifdef xt3
    setvbuf(fp1,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

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
               scanf the second file 
  *******************************************/

  if ((fp2 = fopen(argv[2],"r")) != NULL){

    /* scanf cube tile */

    fscanf(fp2,"%s",ctmp);
    fscanf(fp2,"%s",ctmp);
    fscanf(fp2,"%d",&atomnum2);
    fscanf(fp2,"%lf %lf %lf",&Grid_Origin2[1],&Grid_Origin2[2],&Grid_Origin2[3]);
    fscanf(fp2,"%d",&Ngrid2_1);
    fscanf(fp2,"%lf %lf %lf",&gtv2[1][1],&gtv2[1][2],&gtv2[1][3]);
    fscanf(fp2,"%d",&Ngrid2_2);
    fscanf(fp2,"%lf %lf %lf",&gtv2[2][1],&gtv2[2][2],&gtv2[2][3]);
    fscanf(fp2,"%d",&Ngrid2_3);
    fscanf(fp2,"%lf %lf %lf",&gtv2[3][1],&gtv2[3][2],&gtv2[3][3]);

    /* allocation of arrays */

    Gxyz2 = (double**)malloc(sizeof(double*)*(atomnum2+1)); 
    for (i=0; i<(atomnum2+1); i++){
      Gxyz2[i] = (double*)malloc(sizeof(double)*4); 
    }

    CubeData2 = (double***)malloc(sizeof(double**)*Ngrid2_1); 
    for (i=0; i<Ngrid2_1; i++){
      CubeData2[i] = (double**)malloc(sizeof(double*)*Ngrid2_2); 
      for (j=0; j<Ngrid2_2; j++){
        CubeData2[i][j] = (double*)malloc(sizeof(double)*Ngrid2_3); 
      }
    }

    /* check */
   
    po = 0;

    if (Ngrid1_1!=Ngrid2_1){
      printf("Found a difference in the number of grid on a-axis\n");  
      po = 1;      
    }
    
    if (Ngrid1_2!=Ngrid2_2){
      printf("Found a difference in the number of grid on b-axis\n");  
      po = 1;      
    }

    if (Ngrid1_3!=Ngrid2_3){
      printf("Found a difference in the number of grid on c-axis\n");  
      po = 1;      
    }

    if (atomnum1!=atomnum2){
      printf("Found a difference in the number of atoms\n");  
      po = 1;      
    }

    if (Grid_Origin1[1]!=Grid_Origin2[1]){
      printf("Found a difference in x-coordinate of the origin\n");  
      po = 1;      
    }

    if (Grid_Origin1[2]!=Grid_Origin2[2]){
      printf("Found a difference in y-coordinate of the origin\n");  
      po = 1;      
    }

    if (Grid_Origin1[3]!=Grid_Origin2[3]){
      printf("Found a difference in z-coordinate of the origin\n");  
      po = 1;      
    }

    if ( (gtv1[1][1]!=gtv2[1][1]) 
        || 
         (gtv1[1][2]!=gtv2[1][2]) 
        || 
         (gtv1[1][3]!=gtv2[1][3]) 
       ){
      printf("Found a difference in the vector of a-axis\n");  
      po = 1;      
    }

    if ( (gtv1[2][1]!=gtv2[2][1]) 
        || 
         (gtv1[2][2]!=gtv2[2][2]) 
        || 
         (gtv1[2][3]!=gtv2[2][3]) 
       ){
      printf("Found a difference in the vector of b-axis\n");  
      po = 1;      
    }

    if ( (gtv1[3][1]!=gtv2[3][1]) 
        || 
         (gtv1[3][2]!=gtv2[3][2]) 
        || 
         (gtv1[3][3]!=gtv2[3][3]) 
       ){
      printf("Found a difference in the vector of c-axis\n");  
      po = 1;      
    }

    /*
    if (po==1){
      do {
        printf("Are you sure you want to continue?, yes(1) or no(2)\n");
        fgets(buf,1000,stdin); sscanf(buf,"%d",itmp);

        if      (itmp==2) exit(0);
        else if (itmp==1) po = 0;

      } while (po==1);
    }   
    */

    /* scanf xyz coordinates */

    for (i=1; i<=atomnum2; i++){
      fscanf(fp2,"%lf %lf %lf %lf %lf",&Gxyz2[i][0],&dtmp,&Gxyz2[i][1],&Gxyz2[i][2],&Gxyz2[i][3]);
    }

    /* scanf cube data */

    for (n1=0; n1<Ngrid2_1; n1++){
      for (n2=0; n2<Ngrid2_2; n2++){
        for (n3=0; n3<Ngrid2_3; n3++){
          fscanf(fp2,"%lf",&CubeData2[n1][n2][n3]);
	}
      }
    }

    fclose(fp2);
  }
  else{
    printf("error in reading %s\n",argv[2]);
  }

  /*******************************************
                make a file 
  *******************************************/

  if ((fp1 = fopen(argv[3],"w")) != NULL){

    fprintf(fp1,"%s\n",ctmp);
    fprintf(fp1,"%s\n",ctmp);
    fprintf(fp1,"%5d %12.6f %12.6f %12.6f\n",atomnum1,Grid_Origin1[1],Grid_Origin1[2],Grid_Origin1[3]);
    fprintf(fp1,"%5d %12.6f %12.6f %12.6f\n",Ngrid1_1,gtv1[1][1],gtv1[1][2],gtv1[1][3]);
    fprintf(fp1,"%5d %12.6f %12.6f %12.6f\n",Ngrid1_2,gtv1[2][1],gtv1[2][2],gtv1[2][3]);
    fprintf(fp1,"%5d %12.6f %12.6f %12.6f\n",Ngrid1_3,gtv1[3][1],gtv1[3][2],gtv1[3][3]);

    for (i=1; i<=atomnum1; i++){
      fprintf(fp1,"%5.0f %12.6f %12.6f %12.6f %12.6f\n",
              Gxyz1[i][0],0.0,Gxyz1[i][1],Gxyz1[i][2],Gxyz1[i][3]);
    }

    for (n1=0; n1<Ngrid1_1; n1++){
      for (n2=0; n2<Ngrid1_2; n2++){
        for (n3=0; n3<Ngrid1_3; n3++){
          fprintf(fp1,"%13.3E",CubeData1[n1][n2][n3]-CubeData2[n1][n2][n3]);
          if ((n3+1)%6==0) { fprintf(fp1,"\n"); }
	}
        /* avoid double \n\n when Ngrid3%6 == 0  */
        if (Ngrid1_3%6!=0) fprintf(fp1,"\n");
      }
    }

    fclose(fp1);
  }
  else{
    printf("error in saving %s\n",argv[3]);
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

  for (i=0; i<(atomnum2+1); i++){
    free(Gxyz2[i]);
  }
  free(Gxyz2);

  for (i=0; i<Ngrid2_1; i++){
    for (j=0; j<Ngrid2_2; j++){
      free(CubeData2[i][j]);
    }
    free(CubeData2[i]);
  }
  free(CubeData2);

}




