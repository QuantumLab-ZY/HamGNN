/**********************************************************************
  esp.c:

     esp.c is a routine to calculate electrostatic potential (ESP).
     This code follows GNU-GPL.

     usage:  ./esp name -c 0 -s 1.4 2.0   

       Note: name.out and name.vhart.cube must be in the same directory.

       -c      constraint parameter 
               '-c 0' means charge conservation 
               '-c 1' means charge and dipole moment conservation  
       -s      scale factors for vdw radius
               '-s 1.4 2.0' means that 1.4 and 2.0 are 1st and 2nd scale factors  

     ref
       U.C.Singh and P.A.Kollman, J.Comp.Chem. 5,129(1984)
       L.E.Chirlian and M.M.Francl, J.Com.Chem. 8, 894(1987)
       B.H.Besler, K.M.Merz Jr. and P.A.Kollman, J.Comp.Chem.11,431(1990)

    van der Waals radii were taken from a website, WebElements
                                         ( http://www.webelements.com ).

  Log of esp.c:

     4/Feb/2004  Released by T.Ozaki 
    25/Sep/2004  local ESP scheme added by T.Ozaki
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "Inputtools.h"
#include "lapack_prototypes.h"
#include "f77func.h"

#define MaxRn1        1     /* the number of periodic cell along a-axis */
#define MaxRn2        1     /* the number of periodic cell along b-axis */
#define MaxRn3        1     /* the number of periodic cell along c-axis */

#define BohrR         0.529177249
#define AU2Debye      2.54174776
#define YOUSO10       500
#define MAXBUF        1024
#define fp_bsize      1048576     /* buffer size for setvbuf */

#include "mpi.h"

void read_input(char *file);
void read_vhart(char *file);
void read_vna(char *file);
void set_vdw();
void find_grids();
void calc_esp();
void LESP(char *file);
int Species2int(char Species[YOUSO10]);
int SEQ(char str1[YOUSO10], char str2[YOUSO10]);
void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);

int atomnum,SpeciesNum;
int Ngrid1,Ngrid2,Ngrid3;
int Modified_MK;
int c_i,s_i,l_i,po_c,po_s,po_l;
double Grid_Origin[4];
double gtv[4][4];
double tv[4][4];
double Atom_vdw[105];
double Ref_DipMx,Ref_DipMy,Ref_DipMz;
double scale1,scale2;

int *WhatSpecies;
double **Gxyz;
char **SpeName;
char **SpeBasis;
char **SpeVPS;
double ***VHart;
double ***VNA;
double ***X_grid;
double ***Y_grid;
double ***Z_grid;
int ***grid_flag;

int main(int argc, char *argv[]) 
{
  static int i,po_MK;
  static char file[20][100];
  static char *s_vec[20];

  printf("\n******************************************************************\n"); 
  printf("******************************************************************\n"); 
  printf(" esp: effective charges by a ESP fitting method\n"); 
  printf(" Copyright (C), 2004-2019, Taisuke Ozaki \n"); 
  printf(" This is free software, and you are welcome to         \n"); 
  printf(" redistribute it under the constitution of the GNU-GPL.\n");
  printf("******************************************************************\n"); 
  printf("******************************************************************\n\n"); 

  po_c = 0;
  po_s = 0;
  po_l = 0;

  for (i=1; i<argc; i++){
    if (strcmp(argv[i],"-c")==0){
      po_c = 1;
      c_i = i;
    }
    if (strcmp(argv[i],"-s")==0){
      po_s = 1;
      s_i = i;
    }
    if (strcmp(argv[i],"-l")==0){
      po_l = 1;
      l_i = i;
    }
  }

  if ( po_c==0 && po_s==0 && po_l==0 ){
    if (argc!=2){
      printf("Invalid argument\n");
      exit(0);
    }  
  }

  if (po_c==1 && po_s==0){
    if (argc!=4){
      printf("Invalid argument\n");
      exit(0);
    }  
  }

  if (po_c==0 && po_s==1){
    if (argc!=5){
      printf("Invalid argument\n");
      exit(0);
    }  
  }

  if (po_c==1 && po_s==1){
    if (argc!=7){
      printf("Invalid argument\n");
      exit(0);
    }  
  }

  if (po_l==1){
    if (argc!=3){
      printf("Invalid argument\n");
      exit(0);
    }  
  }

  /* -c */
  if (po_c==1){
    if (strcmp(argv[c_i+1],"0")==0 || strcmp(argv[c_i+1],"1")==0){
      Modified_MK = atoi(argv[c_i+1]);
    }
    else {
      printf("Invalid value for -c\n");
      exit(1);
    }
  }

  /* -s */
  if (po_s==1){
    scale1 = atof(argv[s_i+1]);
    scale2 = atof(argv[s_i+2]);

    if (scale1<0.0){
      printf("1st scale factor is negative.\n");
      exit(0);
    } 
    if (scale2<0.0){
      printf("2nd scale factor is negative.\n");
      exit(0);
    } 

    if (scale2<=scale1){
      printf("2nd scale factor should be larger than 1st.\n");
      exit(0);
    } 
  }

  /* -l */
  if (po_l==1){
    sprintf(file[1],"%s.vhart.cube",argv[1]);  
    sprintf(file[2],"%s.LESP",argv[1]);
    read_vhart(file[1]);
    LESP(file[2]);   
    exit(0);
  }

  /* default */
  if (po_c==0){
    Modified_MK = 0;
  }

  /* default */
  if (po_s==0){
    scale1 = 1.4;
    scale2 = 5.0;
  }
  
  sprintf(file[0],"%s.out",argv[1]);  
  sprintf(file[1],"%s.vhart.cube",argv[1]);  

  s_vec[0]="Constraint: charge "; s_vec[1]="Constraint: charge + dipole moment";
  printf("%s\n",s_vec[Modified_MK]);
  printf("Scale factors for vdw radius %10.5f %10.5f\n",scale1,scale2);
 
  /****************************************************
           Read a input file
  ****************************************************/

  read_input(file[0]);

  /****************************************************
           Read a vhart file
  ****************************************************/

  read_vhart(file[1]);

  /****************************************************
           Read a vna file
  ****************************************************/

  /*
   read_vna(argv[2]);
  */

  /****************************************************
               find grids in a shell
      defined by two scaled van der Waals radii
  ****************************************************/

   set_vdw();
   find_grids();

  /****************************************************
      calculate ESP
  ****************************************************/

   calc_esp();

}


void LESP(char *file)
{
  static int i,j,k,ct_AN;
  static double *EZ0;  
  static double *EZ;  
  static double **A,*A2;  
  static double x,y,z; 
  static double tdp,dpx,dpy,dpz;
  static FILE *fp;
  static char ctmp1[YOUSO10];
  static INTEGER N, NRHS, LDA, *IPIV, LDB, INFO;
  char fp_buf[fp_bsize];          /* setvbuf */

  printf("Effective charge estimated by a local ESP method\n");

  if ((fp = fopen(file,"r")) != NULL){

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    /* atomnum */
    fscanf(fp,"%d",&atomnum);

    /* allocation of arrays */
    EZ0 = (double*)malloc(sizeof(double)*(atomnum+10));
    EZ  = (double*)malloc(sizeof(double)*(atomnum+10));
    A = (double**)malloc(sizeof(double*)*(atomnum+10));
    for (i=0; i<(atomnum+10); i++){
      A[i] = (double*)malloc(sizeof(double)*(atomnum+10));
    }  

    A2 = (double*)malloc(sizeof(double)*(atomnum+10)*(atomnum+10));

    /* read data */
    for (i=0; i<atomnum; i++){
      fscanf(fp,"%d %lf",&j,&EZ0[i]);
      EZ[i] = EZ0[i];
    }

    /* set data */
    for (i=1; i<(atomnum+10); i++){
      for (j=1; j<(atomnum+10); j++){
        A[i][j] = 0.0;
      }
    }

    for (i=1; i<=atomnum; i++){
      A[i][i] = 2.0;
      A[i][atomnum+1] = 1.0;
      A[atomnum+1][i] = 1.0;
    }

    /* A to A2 */
    i = 0;
    for (k=1; k<=(atomnum+1); k++){
      for (j=1; j<=(atomnum+1); j++){
	A2[i] = A[j][k];
	i++;
      }
    }
    
    /* solve A*EZ = EZ0 */
     
    N = atomnum + 1;
    NRHS = 1;
    LDA = N;
    LDB = N;
    IPIV = (INTEGER*)malloc(sizeof(INTEGER)*N);

    F77_NAME(dgesv,DGESV)(&N, &NRHS, A2, &LDA, IPIV, EZ, &LDB, &INFO);

    if( INFO==0 ){
      printf("Success\n" ); 
    }
    else{
      printf("Failure: linear dependent\n" ); 
      exit(0); 
    }

    printf("\n");    
    printf("                                   without       with charge conservation\n");
    for(i=0; i<atomnum; i++){
      printf("  Atom=%4d  Local ESP Charge= %12.8f  %12.8f\n",i+1,EZ0[i],EZ[i]);
    }

    fclose(fp);

    printf("\n");

    /* calculate dipole moment */

    dpx = 0.0;
    dpy = 0.0;
    dpz = 0.0;

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      x = Gxyz[ct_AN][1];
      y = Gxyz[ct_AN][2];
      z = Gxyz[ct_AN][3];
      dpx += AU2Debye*EZ[ct_AN-1]*x;
      dpy += AU2Debye*EZ[ct_AN-1]*y;
      dpz += AU2Debye*EZ[ct_AN-1]*z;
    }
    tdp = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);

    printf("\n");
    printf("  Magnitude of dipole moment %15.10f (Debye)\n",tdp);
    printf("  Component x y z  %15.10f %15.10f %15.10f\n\n",dpx,dpy,dpz);

    /* freeing of arrays */

    free(IPIV);
    free(EZ0);
    free(EZ);
    for (i=0; i<(atomnum+10); i++){
      free(A[i]);
    }  
    free(A);
    free(A2);

  }
  else{
    printf("Failure of reading LESP file.\n\n");
    exit(0);
  }

}




void calc_esp()
{
  static int ct_AN,n1,n2,n3,po,spe;
  static int i,j,k;
  static int Rn1,Rn2,Rn3;
  static int num_grid;
  static double sum0,sum1,rij,rik;
  static double cx,cy,cz;
  static double bik,bij;
  static double x,y,z;
  static double dif,total_diff;
  static double GridVol;
  static double dx,dy,dz;
  static double dpx,dpy,dpz,tdp;
  static double tmp[4];
  static double **A,*B;
  static double *A2;
  static INTEGER N, NRHS, LDA, *IPIV, LDB, INFO;

  /* find the number of grids in the shell */
  num_grid = 0;
  for (n1=0; n1<Ngrid1; n1++){
    for (n2=0; n2<Ngrid2; n2++){
      for (n3=0; n3<Ngrid3; n3++){
        if (grid_flag[n1][n2][n3]==1) num_grid++;
      }
    }
  }

  printf("Number of grids in a van der Waals shell = %2d\n",num_grid);

  Cross_Product(gtv[2],gtv[3],tmp);
  GridVol = fabs( Dot_Product(gtv[1],tmp) );
  printf("Volume per grid = %15.10f (Bohr^3)\n",GridVol);

  /* make a matrix A and a vector B */
  A = (double**)malloc(sizeof(double*)*(atomnum+10));
  for (i=0; i<(atomnum+10); i++){
    A[i] = (double*)malloc(sizeof(double)*(atomnum+10));
    for (j=0; j<(atomnum+10); j++) A[i][j] = 0.0;
  }    

  A2 = (double*)malloc(sizeof(double)*(atomnum+10)*(atomnum+10));
  B = (double*)malloc(sizeof(double)*(atomnum+10));

  for (j=1; j<=atomnum; j++){
    for (k=1; k<=atomnum; k++){

      sum0 = 0.0;
      sum1 = 0.0;

      for (n1=0; n1<Ngrid1; n1++){
	for (n2=0; n2<Ngrid2; n2++){
	  for (n3=0; n3<Ngrid3; n3++){
	    if (grid_flag[n1][n2][n3]==1){

	      x = X_grid[n1][n2][n3];
	      y = Y_grid[n1][n2][n3];
	      z = Z_grid[n1][n2][n3];
     
              bij = 0.0;
              bik = 0.0;

              for (Rn1=-MaxRn1; Rn1<=MaxRn1; Rn1++){
                for (Rn2=-MaxRn2; Rn2<=MaxRn2; Rn2++){
                  for (Rn3=-MaxRn3; Rn3<=MaxRn3; Rn3++){

                    cx = (double)Rn1*tv[1][1] + (double)Rn2*tv[2][1] + (double)Rn3*tv[3][1]; 
                    cy = (double)Rn1*tv[1][2] + (double)Rn2*tv[2][2] + (double)Rn3*tv[3][2]; 
                    cz = (double)Rn1*tv[1][3] + (double)Rn2*tv[2][3] + (double)Rn3*tv[3][3]; 

                    /* rij */
                    dx = x - (Gxyz[j][1] + cx); 
                    dy = y - (Gxyz[j][2] + cy); 
                    dz = z - (Gxyz[j][3] + cz); 
                    rij = sqrt(dx*dx + dy*dy + dz*dz); 
                    bij += 1.0/rij;
 
		    /* rik */
		    dx = x - (Gxyz[k][1] + cx); 
		    dy = y - (Gxyz[k][2] + cy); 
		    dz = z - (Gxyz[k][3] + cz); 
		    rik = sqrt(dx*dx + dy*dy + dz*dz); 
                    bik += 1.0/rik;
                  }
                }
              }

              sum0 += bij*bik;

              if (j==1){
		
                /*
                sum1 -= (VHart[n1][n2][n3] + VNA[n1][n2][n3])*bik;
		*/

		sum1 -= VHart[n1][n2][n3]*bik;
              }

            }
	  }
	}
      }

      A[j][k] = sum0;
      if (j==1) B[k-1] = sum1;
    }
  }


  /* MK */
  if (Modified_MK==0){

    for (k=1; k<=atomnum; k++){
      A[atomnum+1][k] = 1.0;
      A[k][atomnum+1] = 1.0;
    }
    A[atomnum+1][atomnum+1] = 0.0;
    B[atomnum] = 1.0;

    /* A to A2 */

    i = 0;
    for (k=1; k<=(atomnum+1); k++){
      for (j=1; j<=(atomnum+1); j++){
	A2[i] = A[j][k];         
	i++;
      }
    }

    /* solve Aq = B */

    N = atomnum + 1;
    NRHS = 1;
    LDA = N;
    LDB = N;
    IPIV = (INTEGER*)malloc(sizeof(INTEGER)*N);

    F77_NAME(dgesv,DGESV)(&N, &NRHS, A2, &LDA, IPIV, B, &LDB, &INFO);

    if( INFO==0 ){
      printf("Success\n" ); 
    }
    else{
      printf("Failure: linear dependent\n" ); 
      exit(0); 
    }

    printf("\n");    
    for(i=0; i<atomnum; i++){
      printf("  Atom=%4d  Fitting Effective Charge=%15.11f\n",i+1,B[i]);
    }

  }

  /* Modified MK */
  else if (Modified_MK==1){

    for (k=1; k<=atomnum; k++){

      A[atomnum+1][k] = 1.0;
      A[k][atomnum+1] = 1.0;

      A[atomnum+2][k] = Gxyz[k][1];
      A[atomnum+3][k] = Gxyz[k][2];
      A[atomnum+4][k] = Gxyz[k][3];

      A[k][atomnum+2] = Gxyz[k][1];
      A[k][atomnum+3] = Gxyz[k][2];
      A[k][atomnum+4] = Gxyz[k][3];
    }

    B[atomnum  ] = 0.0;
    B[atomnum+1] = Ref_DipMx/AU2Debye;
    B[atomnum+2] = Ref_DipMy/AU2Debye;
    B[atomnum+3] = Ref_DipMz/AU2Debye;

    /* A to A2 */

    i = 0;
    for (k=1; k<=(atomnum+4); k++){
      for (j=1; j<=(atomnum+4); j++){
	A2[i] = A[j][k];         
	i++;
      }
    }

    /* solve Aq = B */

    N = atomnum + 4;
    NRHS = 1;
    LDA = N;
    LDB = N;
    IPIV = (INTEGER*)malloc(sizeof(INTEGER)*N);

    F77_NAME(dgesv,DGESV)(&N, &NRHS, A2, &LDA, IPIV, B, &LDB, &INFO); 

    if( INFO==0 ){
      printf("Success\n" ); 
    }
    else{
      printf("Failure: linear dependent\n" ); 
      exit(0); 
    }

    printf("\n");    
    for(i=0; i<atomnum; i++){
      printf("  Atom=%4d  Fitting Effective Charge=%15.11f\n",i+1,B[i]);
    }

  }

  dpx = 0.0;
  dpy = 0.0;
  dpz = 0.0;

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    x = Gxyz[ct_AN][1];
    y = Gxyz[ct_AN][2];
    z = Gxyz[ct_AN][3];
    dpx += AU2Debye*B[ct_AN-1]*x;
    dpy += AU2Debye*B[ct_AN-1]*y;
    dpz += AU2Debye*B[ct_AN-1]*z;
  }
  tdp = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);

  printf("\n");
  printf("  Magnitude of dipole moment %15.10f (Debye)\n",tdp);
  printf("  Component x y z  %15.10f %15.10f %15.10f\n",dpx,dpy,dpz);

  /* calc diff */

  total_diff = 0.0; 

  for (n1=0; n1<Ngrid1; n1++){
    for (n2=0; n2<Ngrid2; n2++){
      for (n3=0; n3<Ngrid3; n3++){
	if (grid_flag[n1][n2][n3]==1){

          x = X_grid[n1][n2][n3];
	  y = Y_grid[n1][n2][n3];
	  z = Z_grid[n1][n2][n3];

          for (Rn1=-MaxRn1; Rn1<=MaxRn1; Rn1++){
            for (Rn2=-MaxRn2; Rn2<=MaxRn2; Rn2++){
              for (Rn3=-MaxRn3; Rn3<=MaxRn3; Rn3++){

                cx = (double)Rn1*tv[1][1] + (double)Rn2*tv[2][1] + (double)Rn3*tv[3][1]; 
                cy = (double)Rn1*tv[1][2] + (double)Rn2*tv[2][2] + (double)Rn3*tv[3][2]; 
                cz = (double)Rn1*tv[1][3] + (double)Rn2*tv[2][3] + (double)Rn3*tv[3][3]; 

                for (j=1; j<=atomnum; j++){

                  dx = x - (Gxyz[j][1] + cx); 
                  dy = y - (Gxyz[j][2] + cy); 
                  dz = z - (Gxyz[j][3] + cz); 
                  rij = sqrt(dx*dx + dy*dy + dz*dz); 

                  dif = -VHart[n1][n2][n3] + B[j-1]/rij;
                  total_diff += dif*dif;
 	        }
	      }
	    }
	  }

	}
      }
    }
  }

  total_diff = sqrt(total_diff)/(GridVol*num_grid); 
  printf("RMS between the given ESP and fitting charges (Hartree/Bohr^3)=%15.12f\n\n",
         total_diff);

  /* freeing of arrays */
  for (i=0; i<(atomnum+10); i++){
    free(A[i]);
  }    
  free(A);

  free(B);
  free(A2);
  free(IPIV);

}


void find_grids()
{

  static int ct_AN,n1,n2,n3,po,wsp,wan;
  static double x,y,z,dx,dy,dz,r;
  
  static int nn1;

  for (n1=0; n1<Ngrid1; n1++){
    for (n2=0; n2<Ngrid2; n2++){
      for (n3=0; n3<Ngrid3; n3++){
        grid_flag[n1][n2][n3] = 0;
      }
    }
  }
  
  /* find outer grids for the fisrt vdw surface */

  for (n1=0; n1<Ngrid1; n1++){
    for (n2=0; n2<Ngrid2; n2++){
      for (n3=0; n3<Ngrid3; n3++){
        x = X_grid[n1][n2][n3];
        y = Y_grid[n1][n2][n3];
        z = Z_grid[n1][n2][n3];

        po = 0;
        ct_AN = 1;
        do{

          wan = WhatSpecies[ct_AN];

          dx = x - Gxyz[ct_AN][1]; 
          dy = y - Gxyz[ct_AN][2]; 
          dz = z - Gxyz[ct_AN][3]; 

          r = sqrt(dx*dx + dy*dy + dz*dz);  

          if (Atom_vdw[wan]==0.0){
            printf("unknown van der Waal radius of atom %d\n",wan);          
            printf("Please set your value for van der Waal radius of atom %d\n",wan);          
            exit(1);
          } 

          if ( r < (scale2*Atom_vdw[wan]) ) po = 1;

          ct_AN++; 
        } while (ct_AN<=atomnum && po==0);  

        if (po==1) grid_flag[n1][n2][n3] = 1; 
      }
    }
  }

  /* find inner grids for the second vdw surface among surviving grids */

  for (n1=0; n1<Ngrid1; n1++){
    for (n2=0; n2<Ngrid2; n2++){
      for (n3=0; n3<Ngrid3; n3++){

        if (grid_flag[n1][n2][n3]==1){

	  x = X_grid[n1][n2][n3];
	  y = Y_grid[n1][n2][n3];
	  z = Z_grid[n1][n2][n3];

	  po = 1;
	  ct_AN = 1;
	  do{

            wan = WhatSpecies[ct_AN];

	    dx = x - Gxyz[ct_AN][1]; 
	    dy = y - Gxyz[ct_AN][2]; 
	    dz = z - Gxyz[ct_AN][3]; 

	    r = sqrt(dx*dx + dy*dy + dz*dz);
 
	    if ( r < (scale1*Atom_vdw[wan]) ) po = 0;

	    ct_AN++;
	  } while (ct_AN<=atomnum && po==1);

	  if (po==0)  grid_flag[n1][n2][n3] = 0; 

	}

      }
    }
  }

}

void read_vhart(char *file)
{
  static int i,ct_AN,n1,n2,n3;
  static double tmp0;
  static FILE *fp;
  static char ctmp1[YOUSO10];
  char fp_buf[fp_bsize];          /* setvbuf */

  if ((fp = fopen(file,"r")) != NULL){

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    fscanf(fp,"%s",ctmp1);
    fscanf(fp,"%s",ctmp1);

    /* atomnum */
    fscanf(fp,"%d",&atomnum);

    Gxyz = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
      Gxyz[ct_AN] = (double*)malloc(sizeof(double)*4);
    }
    WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));

    /* orgin of grid */
    fscanf(fp,"%lf %lf %lf",&Grid_Origin[1],&Grid_Origin[2],&Grid_Origin[3]);
    
    /* Ngrid and gtv */
    fscanf(fp,"%d %lf %lf %lf",&Ngrid1,&gtv[1][1],&gtv[1][2],&gtv[1][3]);
    fscanf(fp,"%d %lf %lf %lf",&Ngrid2,&gtv[2][1],&gtv[2][2],&gtv[2][3]);
    fscanf(fp,"%d %lf %lf %lf",&Ngrid3,&gtv[3][1],&gtv[3][2],&gtv[3][3]);

    tv[1][1] = (double)Ngrid1*gtv[1][1];
    tv[1][2] = (double)Ngrid1*gtv[1][2];
    tv[1][3] = (double)Ngrid1*gtv[1][3];

    tv[2][1] = (double)Ngrid2*gtv[2][1];
    tv[2][2] = (double)Ngrid2*gtv[2][2];
    tv[2][3] = (double)Ngrid2*gtv[2][3];

    tv[3][1] = (double)Ngrid3*gtv[3][1];
    tv[3][2] = (double)Ngrid3*gtv[3][2];
    tv[3][3] = (double)Ngrid3*gtv[3][3];

    VHart = (double***)malloc(sizeof(double**)*Ngrid1);
    for (n1=0; n1<Ngrid1; n1++){
      VHart[n1] = (double**)malloc(sizeof(double*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
        VHart[n1][n2] = (double*)malloc(sizeof(double)*Ngrid3);
      }
    }

    X_grid = (double***)malloc(sizeof(double**)*Ngrid1);
    for (n1=0; n1<Ngrid1; n1++){
      X_grid[n1] = (double**)malloc(sizeof(double*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
        X_grid[n1][n2] = (double*)malloc(sizeof(double)*Ngrid3);
      }
    }

    Y_grid = (double***)malloc(sizeof(double**)*Ngrid1);
    for (n1=0; n1<Ngrid1; n1++){
      Y_grid[n1] = (double**)malloc(sizeof(double*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
        Y_grid[n1][n2] = (double*)malloc(sizeof(double)*Ngrid3);
      }
    }

    Z_grid = (double***)malloc(sizeof(double**)*Ngrid1);
    for (n1=0; n1<Ngrid1; n1++){
      Z_grid[n1] = (double**)malloc(sizeof(double*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
        Z_grid[n1][n2] = (double*)malloc(sizeof(double)*Ngrid3);
      }
    }

    grid_flag = (int***)malloc(sizeof(int**)*Ngrid1);
    for (n1=0; n1<Ngrid1; n1++){
      grid_flag[n1] = (int**)malloc(sizeof(int*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
        grid_flag[n1][n2] = (int*)malloc(sizeof(int)*Ngrid3);
      }
    }

    /* Gxyz */
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      fscanf(fp,"%d %lf %lf %lf %lf",
             &WhatSpecies[ct_AN],
             &tmp0,
             &Gxyz[ct_AN][1],&Gxyz[ct_AN][2],&Gxyz[ct_AN][3]);
    }

    /* Vhart */
    for (n1=0; n1<Ngrid1; n1++){
      for (n2=0; n2<Ngrid2; n2++){
        for (n3=0; n3<Ngrid3; n3++){
	  fscanf(fp,"%lf", &VHart[n1][n2][n3]);
	}
      }
    }

    /* XYZ_grid */
    for (n1=0; n1<Ngrid1; n1++){
      for (n2=0; n2<Ngrid2; n2++){
        for (n3=0; n3<Ngrid3; n3++){

  	  X_grid[n1][n2][n3] = (double)n1*gtv[1][1] + (double)n2*gtv[2][1]
 	                     + (double)n3*gtv[3][1] + Grid_Origin[1];
	  Y_grid[n1][n2][n3] = (double)n1*gtv[1][2] + (double)n2*gtv[2][2]
	                     + (double)n3*gtv[3][2] + Grid_Origin[2];
	  Z_grid[n1][n2][n3] = (double)n1*gtv[1][3] + (double)n2*gtv[2][3]
	                     + (double)n3*gtv[3][3] + Grid_Origin[3];
	}
      }
    }

    fclose(fp);
  }
  else{
    printf("Failure of reading vhart file.\n");
    exit(0);
  }

}




void read_vna(char *file)
{

  static int i,ct_AN,n1,n2,n3;
  static double tmp0;
  static FILE *fp;
  static char ctmp1[YOUSO10];
  char fp_buf[fp_bsize];          /* setvbuf */

  if ((fp = fopen(file,"r")) != NULL){

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    fscanf(fp,"%s",ctmp1);
    fscanf(fp,"%s",ctmp1);

    /* atomnum */
    fscanf(fp,"%d",&atomnum);

    /* orgin of grid */
    fscanf(fp,"%lf %lf %lf",&Grid_Origin[1],&Grid_Origin[2],&Grid_Origin[3]);
    
    /* Ngrid and gtv */
    fscanf(fp,"%d %lf %lf %lf",&Ngrid1,&gtv[1][1],&gtv[1][2],&gtv[1][3]);
    fscanf(fp,"%d %lf %lf %lf",&Ngrid2,&gtv[2][1],&gtv[2][2],&gtv[2][3]);
    fscanf(fp,"%d %lf %lf %lf",&Ngrid3,&gtv[3][1],&gtv[3][2],&gtv[3][3]);

    VNA = (double***)malloc(sizeof(double**)*Ngrid1);
    for (n1=0; n1<Ngrid1; n1++){
      VNA[n1] = (double**)malloc(sizeof(double*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
        VNA[n1][n2] = (double*)malloc(sizeof(double)*Ngrid3);
      }
    }

    /* Gxyz */
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      fscanf(fp,"%d %lf %lf %lf %lf",
             &WhatSpecies[ct_AN],
             &tmp0,
             &Gxyz[ct_AN][1],&Gxyz[ct_AN][2],&Gxyz[ct_AN][3]);
    }

    /* VNA */
    for (n1=0; n1<Ngrid1; n1++){
      for (n2=0; n2<Ngrid2; n2++){
        for (n3=0; n3<Ngrid3; n3++){
	  fscanf(fp,"%lf", &VNA[n1][n2][n3]);
	}
      }
    }

    fclose(fp);
  }
  else{
    printf("Failure of reading vna file.\n");
    exit(0);
  }
}



void read_input(char *file)
{
  static int i,j,spe,po;
  static int coordinates_unit;
  static int ct_AN;
  double r_vec[20],r_vec2[20];
  int i_vec[20],i_vec2[20];
  char *s_vec[20],Species[20];
  char buf[MAXBUF];
  FILE *fp;


  if (input_open(file)){

    /****************************************************
                 find dipole moment
    ****************************************************/

    if (fp=input_find("Dipole") ) {
      fscanf(fp,"%s",buf);
      fscanf(fp,"%s",buf);
      fscanf(fp,"%s %lf %lf %lf",buf,&Ref_DipMx,&Ref_DipMy,&Ref_DipMz);
    }

    input_close();
  }
  else {
    exit(0);
  }

  /****************************************************
              Definition of Atomic Species
  ****************************************************/

  /*
  input_int("Species.Number",&SpeciesNum,0);
  if (SpeciesNum<=0){
    printf("Species.Number may be wrong.\n");
    po++;
  }

  SpeName = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (spe=0; spe<SpeciesNum; spe++){
    SpeName[spe] = (char*)malloc(sizeof(char)*YOUSO10);
  }  

  SpeBasis = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (spe=0; spe<SpeciesNum; spe++){
    SpeBasis[spe] = (char*)malloc(sizeof(char)*YOUSO10);
  }  

  SpeVPS = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (spe=0; spe<SpeciesNum; spe++){
    SpeVPS[spe] = (char*)malloc(sizeof(char)*YOUSO10);
  }  

  if (fp=input_find("<Definition.of.Atomic.Species")) {
    for (i=0; i<SpeciesNum; i++){
      fscanf(fp,"%s %s %s",SpeName[i],SpeBasis[i],SpeVPS[i]);
    }
    if (! input_last("Definition.of.Atomic.Species>")) {

      po++;

      printf("Format error for Definition.of.Atomic.Species\n");
      exit(0);
    }
  }
  */

  /****************************************************
                        atoms
  ****************************************************/

  /*
  input_int("Atoms.Number",&atomnum,0);
  s_vec[0]="Ang";  s_vec[1]="AU";
  i_vec[0]= 0;     i_vec[1]= 1;
  input_string2int("Atoms.SpeciesAndCoordinates.Unit",
                   &coordinates_unit,2,s_vec,i_vec);

  if (fp=input_find("<Atoms.SpeciesAndCoordinates") ) {

    for (i=1; i<=atomnum; i++){
      fgets(buf,MAXBUF,fp);

      sscanf(buf,"%i %s %lf %lf %lf",
             &j, &Species, &Gxyz[i][1],&Gxyz[i][2],&Gxyz[i][3]);

      WhatSpecies[i] = Species2int(Species);

      printf("i=%3d spe=%2d %15.12f %15.12f %15.12f\n",
              i,WhatSpecies[i],Gxyz[i][1],Gxyz[i][2],Gxyz[i][3]);

      if (i!=j){
	printf("Format error of the sequential number %i in <Atoms.SpeciesAndCoordinates\n",j);
	po++;
      }
    }

    ungetc('\n',fp);
    if (!input_last("Atoms.SpeciesAndCoordinates>")) {

      printf("Format error for Atoms.SpeciesAndCoordinates\n");
      po++;
    }

    if (coordinates_unit==0){
      for (i=1; i<=atomnum; i++){
	Gxyz[i][1] = Gxyz[i][1]/BohrR;
	Gxyz[i][2] = Gxyz[i][2]/BohrR;
	Gxyz[i][3] = Gxyz[i][3]/BohrR;
      }
    }
  }
  */


}



int Species2int(char Species[YOUSO10])
{
  static int i,po;

  i = 0;
  po = 0; 
  while (i<SpeciesNum && po==0){
    if (SEQ(Species,SpeName[i])==1){
      po = 1;
    }
    if (po==0) i++;
  };

  if (po==1) return i;
  else {
    printf("%s is an invalid species name in Atoms.SpeciesAndCoordinates\n",
            Species);
    printf("Please check your input file\n");
    MPI_Finalize();
    exit(1);
  }
}



int SEQ(char str1[YOUSO10], char str2[YOUSO10])
{
  
  int i,result,l1,l2;

  l1 = strlen(str1);
  l2 = strlen(str2);

  result = 1; 
  if (l1 == l2){
    for (i=0; i<=l1-1; i++){
      if (str1[i]!=str2[i])  result = 0;   
    }
  }
  else
    result = 0; 

  return result;
}



void set_vdw()
{
  /*
    van der Waals radii were taken from a website, WebElements
                                 ( http://www.webelements.com ).
  */
  /* length unit: converted to Bohr */

  Atom_vdw[  1] = 1.20/BohrR;
  Atom_vdw[  2] = 1.40/BohrR;
  Atom_vdw[  3] = 1.82/BohrR;
  Atom_vdw[  4] = 1.53/BohrR;
  Atom_vdw[  5] = 1.92/BohrR;
  Atom_vdw[  6] = 1.70/BohrR;
  Atom_vdw[  7] = 1.55/BohrR;
  Atom_vdw[  8] = 1.52/BohrR;
  Atom_vdw[  9] = 1.47/BohrR;
  Atom_vdw[ 10] = 1.54/BohrR;
  Atom_vdw[ 11] = 2.27/BohrR;
  Atom_vdw[ 12] = 1.73/BohrR;
  Atom_vdw[ 13] = 1.84/BohrR;
  Atom_vdw[ 14] = 2.10/BohrR;
  Atom_vdw[ 15] = 1.80/BohrR;
  Atom_vdw[ 16] = 1.80/BohrR;
  Atom_vdw[ 17] = 1.75/BohrR;
  Atom_vdw[ 18] = 1.88/BohrR;
  Atom_vdw[ 19] = 2.75/BohrR;
  Atom_vdw[ 20] = 2.31/BohrR;
  Atom_vdw[ 21] = 0.00;
  Atom_vdw[ 22] = 0.00;
  Atom_vdw[ 23] = 0.00;
  Atom_vdw[ 24] = 0.00;
  Atom_vdw[ 25] = 0.00;
  Atom_vdw[ 26] = 0.00;
  Atom_vdw[ 27] = 0.00;
  Atom_vdw[ 28] = 1.63/BohrR;
  Atom_vdw[ 29] = 1.40/BohrR;
  Atom_vdw[ 30] = 1.31/BohrR;
  Atom_vdw[ 31] = 1.87/BohrR;
  Atom_vdw[ 32] = 0.00;
  Atom_vdw[ 33] = 1.85/BohrR;
  Atom_vdw[ 34] = 1.90/BohrR;
  Atom_vdw[ 35] = 1.85/BohrR;
  Atom_vdw[ 36] = 2.02/BohrR;
  Atom_vdw[ 37] = 2.75/BohrR;
  Atom_vdw[ 38] = 0.00;
  Atom_vdw[ 39] = 0.00;
  Atom_vdw[ 40] = 0.00;
  Atom_vdw[ 41] = 0.00;
  Atom_vdw[ 42] = 0.00;
  Atom_vdw[ 43] = 0.00;
  Atom_vdw[ 44] = 0.00;
  Atom_vdw[ 45] = 0.00;
  Atom_vdw[ 46] = 1.63/BohrR;
  Atom_vdw[ 47] = 1.72/BohrR;
  Atom_vdw[ 48] = 1.58/BohrR;
  Atom_vdw[ 49] = 1.93/BohrR;
  Atom_vdw[ 50] = 2.17/BohrR;
  Atom_vdw[ 51] = 0.00;
  Atom_vdw[ 52] = 2.06/BohrR;
  Atom_vdw[ 53] = 1.98/BohrR;
  Atom_vdw[ 54] = 2.16/BohrR;
  Atom_vdw[ 55] = 0.00;
  Atom_vdw[ 56] = 0.00;
  Atom_vdw[ 57] = 0.00;
  Atom_vdw[ 58] = 0.00;
  Atom_vdw[ 59] = 0.00;
  Atom_vdw[ 60] = 0.00;
  Atom_vdw[ 61] = 0.00;
  Atom_vdw[ 62] = 0.00;
  Atom_vdw[ 63] = 0.00;
  Atom_vdw[ 64] = 0.00;
  Atom_vdw[ 65] = 0.00;
  Atom_vdw[ 66] = 0.00;
  Atom_vdw[ 67] = 0.00;
  Atom_vdw[ 68] = 0.00;
  Atom_vdw[ 69] = 0.00;
  Atom_vdw[ 70] = 0.00;
  Atom_vdw[ 71] = 0.00;
  Atom_vdw[ 72] = 0.00;
  Atom_vdw[ 73] = 0.00;
  Atom_vdw[ 74] = 0.00;
  Atom_vdw[ 75] = 0.00;
  Atom_vdw[ 76] = 0.00;
  Atom_vdw[ 77] = 0.00;
  Atom_vdw[ 78] = 1.75/BohrR;
  Atom_vdw[ 79] = 1.66/BohrR;
  Atom_vdw[ 80] = 1.55/BohrR;
  Atom_vdw[ 81] = 1.96/BohrR;
  Atom_vdw[ 82] = 2.02/BohrR;
  Atom_vdw[ 83] = 0.00;
  Atom_vdw[ 84] = 0.00;
  Atom_vdw[ 85] = 0.00;
  Atom_vdw[ 86] = 0.00;
  Atom_vdw[ 87] = 0.00;
  Atom_vdw[ 88] = 0.00;
  Atom_vdw[ 89] = 0.00;
  Atom_vdw[ 90] = 0.00;
  Atom_vdw[ 91] = 0.00;
  Atom_vdw[ 92] = 1.86/BohrR;
  Atom_vdw[ 93] = 0.00;
  Atom_vdw[ 94] = 0.00;
  Atom_vdw[ 95] = 0.00;
  Atom_vdw[ 96] = 0.00;
  Atom_vdw[ 97] = 0.00;
  Atom_vdw[ 98] = 0.00;
  Atom_vdw[ 99] = 0.00;
  Atom_vdw[100] = 0.00;
  Atom_vdw[101] = 0.00;
  Atom_vdw[102] = 0.00;
  Atom_vdw[103] = 0.00;

}

void Cross_Product(double a[4], double b[4], double c[4])
{
  c[1] = a[2]*b[3] - a[3]*b[2]; 
  c[2] = a[3]*b[1] - a[1]*b[3]; 
  c[3] = a[1]*b[2] - a[2]*b[1];
}

double Dot_Product(double a[4], double b[4])
{
  static double sum;
  sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
  return sum;
}
