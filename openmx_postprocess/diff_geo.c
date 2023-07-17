/**********************************************************************
  diff_geo.c:

     diff_geo.c is a routine to calculate a root mean square of deviation,
     a mean deviation between two geometrical Cartesian coordinates,
     and a mean deviation between bond lengths.

     This code follows GNU-GPL.

     usage: 
              ./diff_geo file1.xyz file2.xyz -d rmsd

        option
           -d rmsd      a root mean square of deviation
           -d md        a mean deviation
           -d mdbl 2.2  a mean deviation between bond lengths, 
                        2.2 (Ang) means a cutoff bond length which
                        can be taken into account in the calculation

  Log of diff_geo.c:

     26/Feb/2004  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define fp_bsize         1048576     /* buffer size for setvbuf */
#define PI            3.1415926535897932384626

void read_xyzfiles(char file1[100], char file2[100]);
double rmsd(double **xyz1, double **xyz2);
double minimize_rmsd();
double md(double **xyz1, double **xyz2);
double minimize_md();
void calc_md_BL();
void generate_vec();

int atomnum1,atomnum2;
double **Gxyz1;
double **Gxyz2;
double BL0;
char **Species;

int main(int argc, char *argv[]) 
{
  static int i,po_d,d_i,anal_switch;
  static char file1[100];
  static char file2[100];

  printf("\n******************************************************************\n"); 
  printf("******************************************************************\n"); 
  printf(" diff_geo: analysis of structural deviation.\n"); 
  printf(" Copyright (C), 2004-2019, Taisuke Ozaki \n"); 
  printf(" This is free software, and you are welcome to         \n"); 
  printf(" redistribute it under the constitution of the GNU-GPL.\n");
  printf("******************************************************************\n"); 
  printf("******************************************************************\n\n"); 

  if (argc!=5 && argc!=6){
    printf("Usage:\n");
    printf("  ./diff_geo file1.xyz file2.xyz -d rmsd\n\n");
    printf("   option\n");    
    printf("    -d rmsd      a root mean square of deviation\n");
    printf("    -d md        a mean deviation\n");
    printf("    -d mdbl 2.2  a mean deviation between bond lengths\n");
    printf("                 2.2 (Ang) means a cutoff bond length which\n");
    printf("                 can be taken into account in the calculation\n\n");
    exit(0);
  }

  po_d = 0;
  for (i=1; i<argc; i++){
    if (strcmp(argv[i],"-d")==0){
      po_d = 1;
      d_i = i;
    }
  }

  if (strcmp(argv[d_i+1],"rmsd")==0){
    anal_switch = 1;
    if (argc!=5){
      printf("Usage:\n");
      printf("  ./diff_geo file1.xyz file2.xyz -d rmsd\n\n");
      printf("   option\n");    
      printf("    -d rmsd      a root mean square of deviation\n");
      printf("    -d md        a mean deviation\n");
      printf("    -d mdbl      a mean deviation between bond lengths\n\n");
      printf("                 2.2 (Ang) means a cutoff bond length which\n");
      printf("                 can be taken into account in the calculation\n\n");
      exit(0);
    }
  }  
  else if (strcmp(argv[d_i+1],"md")==0){
    anal_switch = 2;
    if (argc!=5){
      printf("Usage:\n");
      printf("  ./diff_geo file1.xyz file2.xyz -d md\n\n");
      printf("   option\n");    
      printf("    -d rmsd      a root mean square of deviation\n");
      printf("    -d md        a mean deviation\n");
      printf("    -d mdbl      a mean deviation between bond lengths\n\n");
      printf("                 2.2 (Ang) means a cutoff bond length which\n");
      printf("                 can be taken into account in the calculation\n\n");
      exit(0);
    }
  }  
  else if (strcmp(argv[d_i+1],"mdbl")==0){
    anal_switch = 3;

    if (argc!=6){
      printf("Usage:\n");
      printf("  ./diff_geo file1.xyz file2.xyz -d mdbl 2.2\n\n");
      printf("   option\n");    
      printf("    -d rmsd      a root mean square of deviation\n");
      printf("    -d md        a mean deviation\n");
      printf("    -d mdbl      a mean deviation between bond lengths\n\n");
      printf("                 2.2 (Ang) means a cutoff bond length which\n");
      printf("                 can be taken into account in the calculation\n\n");
      exit(0);
    }

    BL0 = atof(argv[d_i+2]); 
    if (BL0<0.0){
      printf("Cutoff bond length is negative.\n");
      exit(0);
    }
  }  

  else{
    printf("  This option is not supported.\n");
    exit(0);
  }  

  sprintf(file1,"%s",argv[1]);  
  sprintf(file2,"%s",argv[2]);  
  read_xyzfiles(file1, file2);

  /* a root mean square of deviation */ 
  if (anal_switch==1){
    minimize_rmsd(); 
  }
  /* a mean deviation */ 
  else if (anal_switch==2){
    minimize_md(); 
  }
  /* a mean deviation between bond lengths */ 
  else if (anal_switch==3){
    calc_md_BL(); 
  }
  /* generate a vector file */ 
  else if (anal_switch==4){
    generate_vec(); 
  }


}

double minimize_rmsd()
{
  static int ct_AN,num1,num2,po,num_trial;
  static int ix,iy,iz,iq1,iq2,iq3;
  static double sumx,sumy,sumz;
  static double cx1,cy1,cz1;
  static double cx2,cy2,cz2;
  static double **xyz2;
  static double MaxX,MaxY,MaxZ,MaxQ1,MaxQ2,MaxQ3;
  static double MinX,MinY,MinZ,MinQ1,MinQ2,MinQ3;
  static double dX,dY,dZ,dQ1,dQ2,dQ3;
  static double Min_RMSD,Last_Min_RMSD;
  static double Sx,Sy,Sz,Cx,Cy,Cz;
  static double Q1,Q2,Q3,x,y,z,dum;
  static double Fx,Fy,Fz,FQ1,FQ2,FQ3;
  static double Rot[5][5];

  /* allocation of array */  
  xyz2 = (double**)malloc(sizeof(double*)*(atomnum2+1));
  for (ct_AN=0; ct_AN<=atomnum2; ct_AN++){
    xyz2[ct_AN] = (double*)malloc(sizeof(double)*4);
  }

  /* find the centroid */

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;
  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    sumx += Gxyz1[ct_AN][1];
    sumy += Gxyz1[ct_AN][2];
    sumz += Gxyz1[ct_AN][3];
  }
  cx1 = cx1/(double)atomnum1;
  cy1 = cy1/(double)atomnum1;
  cz1 = cz1/(double)atomnum1;

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;
  for (ct_AN=1; ct_AN<=atomnum2; ct_AN++){
    sumx += Gxyz2[ct_AN][1];
    sumy += Gxyz2[ct_AN][2];
    sumz += Gxyz2[ct_AN][3];
  }
  cx2 = cx2/(double)atomnum2;
  cy2 = cy2/(double)atomnum2;
  cz2 = cz2/(double)atomnum2;
  
  /* shift the centroid to the origin */
  
  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    Gxyz1[ct_AN][1] -= cx1;
    Gxyz1[ct_AN][2] -= cy1;
    Gxyz1[ct_AN][3] -= cz1;
  }

  for (ct_AN=1; ct_AN<=atomnum2; ct_AN++){
    Gxyz2[ct_AN][1] -= cx2;
    Gxyz2[ct_AN][2] -= cy2;
    Gxyz2[ct_AN][3] -= cz2;
  }


  po = 0;
  num1 = 12;
  num2 = 20;
  num_trial = 1;
  Last_Min_RMSD = 10000.0;
  Min_RMSD = 1.0e+10;

  do {

    if (num_trial==1){

      MaxX =  0.05;
      MinX = -0.05;
      MaxY =  0.05;
      MinY = -0.05;
      MaxZ =  0.05;
      MinZ = -0.05;

      MaxQ1 = 0.2;
      MinQ1 = 0.0;
      MaxQ2 = 0.2;
      MinQ2 = 0.0;
      MaxQ3 = 0.2;
      MinQ3 = 0.0;
    }

    else {
      MaxX = Fx + 2.0*dX;
      MinX = Fx - 2.0*dX;
      MaxY = Fy + 2.0*dY;
      MinY = Fy - 2.0*dY;
      MaxZ = Fz + 2.0*dZ;
      MinZ = Fz - 2.0*dZ;

      MaxQ1 = FQ1 + 2.0*dQ1;
      MinQ1 = FQ1 - 2.0*dQ1;
      MaxQ2 = FQ2 + 2.0*dQ2;
      MinQ2 = FQ2 - 2.0*dQ2;
      MaxQ3 = FQ3 + 2.0*dQ3;
      MinQ3 = FQ3 - 2.0*dQ3;
    }

    dX = (MaxX - MinX)/(double)num1; 
    dY = (MaxY - MinY)/(double)num1; 
    dZ = (MaxZ - MinZ)/(double)num1; 
    dQ1 = (MaxQ1 - MinQ1)/(double)num2; 
    dQ2 = (MaxQ2 - MinQ2)/(double)num2; 
    dQ3 = (MaxQ3 - MinQ3)/(double)num2; 

    for (ix=0; ix<num1; ix++){
      x = MinX + (double)ix*dX;
      for (iy=0; iy<num1; iy++){
	y = MinY + (double)iy*dY;
	for (iz=0; iz<num1; iz++){
	  z = MinZ + (double)iz*dZ;
	  for (iq1=0; iq1<num2; iq1++){
	    Q1 = MinQ1 + (double)iq1*dQ1;
	    Sx = sin(Q1);
	    Cx = cos(Q1);

	    for (iq2=0; iq2<num2; iq2++){
	      Q2 = MinQ2 + (double)iq2*dQ2;
	      Sy = sin(Q2);
	      Cy = cos(Q2);

	      for (iq3=0; iq3<num2; iq3++){
		Q3 = MinQ3 + (double)iq3*dQ3;
		Sz = sin(Q3);
		Cz = cos(Q3);

		/* make a transform matrix */

		Rot[1][1] = Cy*Cz;
		Rot[1][2] = Sx*Sy*Cz - Cx*Sz;
		Rot[1][3] = Cx*Sy*Cz + Sx*Sz;

		Rot[2][1] = Cy*Sz;
		Rot[2][2] = Sx*Sy*Sz + Cx*Cz;
		Rot[2][3] = Cx*Sy*Sz - Sx*Cz;

		Rot[3][1] = -Sy;
		Rot[3][2] = Sx*Cy;
		Rot[3][3] = Cx*Cy;

		for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
		  xyz2[ct_AN][1] =  Rot[1][1]*(Gxyz2[ct_AN][1]-x) 
		                  + Rot[1][2]*(Gxyz2[ct_AN][2]-y)  
		                  + Rot[1][3]*(Gxyz2[ct_AN][3]-z);
  
		  xyz2[ct_AN][2] =  Rot[2][1]*(Gxyz2[ct_AN][1]-x) 
		                  + Rot[2][2]*(Gxyz2[ct_AN][2]-y)  
		                  + Rot[2][3]*(Gxyz2[ct_AN][3]-z);
  

		  xyz2[ct_AN][3] =  Rot[3][1]*(Gxyz2[ct_AN][1]-x) 
	 	                  + Rot[3][2]*(Gxyz2[ct_AN][2]-y)  
		                  + Rot[3][3]*(Gxyz2[ct_AN][3]-z);
		}

		dum = rmsd(Gxyz1, xyz2);
		if (dum<Min_RMSD){
		  Min_RMSD = dum;
		  Fx = x;
		  Fy = y;
		  Fz = z;
		  FQ1 = Q1;
		  FQ2 = Q2;
		  FQ3 = Q3;

         printf(" trial=%2d x,y,z=%6.3f %6.3f %6.3f Q1,Q2,Q3=%6.3f %6.3f %6.3f RMSD=%12.9f\n",
                  num_trial,x,y,z,Q1/PI*180.0,Q2/PI*180.0,Q3/PI*180.0,dum);

		}

	      }
	    }
	  }
	}
      }
    }


    if ( fabs(Last_Min_RMSD-Min_RMSD)<1.0e-5 ) po = 1;

    printf(" RMSD (previous trial)=%15.12f RMSD (current trial)=%15.12f\n",
            Last_Min_RMSD,Min_RMSD); 

    Last_Min_RMSD = Min_RMSD; 

    num_trial++;

  } while (po==0); 

  printf("\n");
  printf(" Final RMSD (Ang/Sqrt(N))=%15.12f\n",Min_RMSD); 
  printf(" Translated x,y,z (Ang) of centroid = %6.3f %6.3f %6.3f\n",Fx,Fy,Fz); 
  printf(" Rotation Angles  Q1,Q2,Q3 (Deg)    = %6.3f %6.3f %6.3f\n",
          FQ1/PI*180.0,FQ2/PI*180.0,FQ3/PI*180.0);

  /* generate a vector file */

  Sx = sin(FQ1);
  Cx = cos(FQ1);
  Sy = sin(FQ2);
  Cy = cos(FQ2);
  Sz = sin(FQ3);
  Cz = cos(FQ3);

  Rot[1][1] = Cy*Cz;
  Rot[1][2] = Sx*Sy*Cz - Cx*Sz;
  Rot[1][3] = Cx*Sy*Cz + Sx*Sz;

  Rot[2][1] = Cy*Sz;
  Rot[2][2] = Sx*Sy*Sz + Cx*Cz;
  Rot[2][3] = Cx*Sy*Sz - Sx*Cz;

  Rot[3][1] = -Sy;
  Rot[3][2] = Sx*Cy;
  Rot[3][3] = Cx*Cy;

  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    xyz2[ct_AN][1] =  Rot[1][1]*(Gxyz2[ct_AN][1]-Fx) 
                    + Rot[1][2]*(Gxyz2[ct_AN][2]-Fy)  
                    + Rot[1][3]*(Gxyz2[ct_AN][3]-Fz);
  
    xyz2[ct_AN][2] =  Rot[2][1]*(Gxyz2[ct_AN][1]-Fx) 
                    + Rot[2][2]*(Gxyz2[ct_AN][2]-Fy)  
                    + Rot[2][3]*(Gxyz2[ct_AN][3]-Fz);
  

    xyz2[ct_AN][3] =  Rot[3][1]*(Gxyz2[ct_AN][1]-Fx) 
                    + Rot[3][2]*(Gxyz2[ct_AN][2]-Fy)  
                    + Rot[3][3]*(Gxyz2[ct_AN][3]-Fz);
  }

  printf(" generate a vector file, dgeo_vec.xsf\n\n");
  generate_vec(Gxyz1, xyz2);

  /* freeing of array */  
  for (ct_AN=0; ct_AN<=atomnum2; ct_AN++){
    free(xyz2[ct_AN]);
  }
  free(xyz2);
} 


double rmsd(double **xyz1, double **xyz2)
{
  static int ct_AN;
  static double dx,dy,dz,r2;
  static double result,sum;

  sum = 0.0;
  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    dx = xyz1[ct_AN][1] - xyz2[ct_AN][1];    
    dy = xyz1[ct_AN][2] - xyz2[ct_AN][2];    
    dz = xyz1[ct_AN][3] - xyz2[ct_AN][3];    
    r2 = dx*dx + dy*dy + dz*dz;
    sum += r2; 
  }

  result = sqrt(sum/(double)atomnum1);
  
  return result;
}





void read_xyzfiles(char file1[100], char file2[100])
{
  static int ct_AN;
  static double tmp1,tmp2,tmp3;
  static char stmp[20];
  static FILE *fp;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* read the first file */
  if ((fp = fopen(file1,"r")) != NULL){

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    fscanf(fp,"%d",&atomnum1);
    Gxyz1 = (double**)malloc(sizeof(double*)*(atomnum1+1));
    for (ct_AN=0; ct_AN<=atomnum1; ct_AN++){
      Gxyz1[ct_AN] = (double*)malloc(sizeof(double)*4);
    }

    Species = (char**)malloc(sizeof(char*)*(atomnum1+1));
    for (ct_AN=0; ct_AN<=atomnum1; ct_AN++){
      Species[ct_AN] = (char*)malloc(sizeof(char)*15);
    }

    for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
      fscanf(fp,"%s %lf %lf %lf %lf %lf %lf",
             stmp,
             &Gxyz1[ct_AN][1],&Gxyz1[ct_AN][2],&Gxyz1[ct_AN][3],
             &tmp1,&tmp2,&tmp3);
    }
 
    fclose(fp);
  }
  else{
    printf("Failure of reading %s\n",file1);
    exit(0);
  }
  
  /* read the second file */
  if ((fp = fopen(file2,"r")) != NULL){

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    fscanf(fp,"%d",&atomnum2);
    Gxyz2 = (double**)malloc(sizeof(double*)*(atomnum2+1));
    for (ct_AN=0; ct_AN<=atomnum2; ct_AN++){
      Gxyz2[ct_AN] = (double*)malloc(sizeof(double)*4);
    }

    for (ct_AN=1; ct_AN<=atomnum2; ct_AN++){
      fscanf(fp,"%s %lf %lf %lf %lf %lf %lf",
             Species[ct_AN],
             &Gxyz2[ct_AN][1],&Gxyz2[ct_AN][2],&Gxyz2[ct_AN][3],
             &tmp1,&tmp2,&tmp3);
    }
 
    fclose(fp);
  }
  else{
    printf("Failure of reading %s\n",file2);
    exit(0);
  }

  if (atomnum1!=atomnum2){
    printf("different in the number of atoms\n"); 
    exit(1);
  }

}


double minimize_md()
{
  static int ct_AN,num1,num2,po,num_trial;
  static int ix,iy,iz,iq1,iq2,iq3;
  static double sumx,sumy,sumz;
  static double cx1,cy1,cz1;
  static double cx2,cy2,cz2;
  static double **xyz2;
  static double MaxX,MaxY,MaxZ,MaxQ1,MaxQ2,MaxQ3;
  static double MinX,MinY,MinZ,MinQ1,MinQ2,MinQ3;
  static double dX,dY,dZ,dQ1,dQ2,dQ3;
  static double Min_MD,Last_Min_MD;
  static double Sx,Sy,Sz,Cx,Cy,Cz;
  static double Q1,Q2,Q3,x,y,z,dum;
  static double Fx,Fy,Fz,FQ1,FQ2,FQ3;
  static double Rot[5][5];

  /* allocation of array */  
  xyz2 = (double**)malloc(sizeof(double*)*(atomnum2+1));
  for (ct_AN=0; ct_AN<=atomnum2; ct_AN++){
    xyz2[ct_AN] = (double*)malloc(sizeof(double)*4);
  }

  /* find the centroid */

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;
  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    sumx += Gxyz1[ct_AN][1];
    sumy += Gxyz1[ct_AN][2];
    sumz += Gxyz1[ct_AN][3];
  }
  cx1 = cx1/(double)atomnum1;
  cy1 = cy1/(double)atomnum1;
  cz1 = cz1/(double)atomnum1;

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;
  for (ct_AN=1; ct_AN<=atomnum2; ct_AN++){
    sumx += Gxyz2[ct_AN][1];
    sumy += Gxyz2[ct_AN][2];
    sumz += Gxyz2[ct_AN][3];
  }
  cx2 = cx2/(double)atomnum2;
  cy2 = cy2/(double)atomnum2;
  cz2 = cz2/(double)atomnum2;
  
  /* shift the centroid to the origin */
  
  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    Gxyz1[ct_AN][1] -= cx1;
    Gxyz1[ct_AN][2] -= cy1;
    Gxyz1[ct_AN][3] -= cz1;
  }

  for (ct_AN=1; ct_AN<=atomnum2; ct_AN++){
    Gxyz2[ct_AN][1] -= cx2;
    Gxyz2[ct_AN][2] -= cy2;
    Gxyz2[ct_AN][3] -= cz2;
  }


  po = 0;
  num1 = 12;
  num2 = 20;
  num_trial = 1;
  Last_Min_MD = 10000.0;
  Min_MD = 1.0e+10;

  do {

    if (num_trial==1){

      MaxX =  0.05;
      MinX = -0.05;
      MaxY =  0.05;
      MinY = -0.05;
      MaxZ =  0.05;
      MinZ = -0.05;

      MaxQ1 = 0.2;
      MinQ1 = 0.0;
      MaxQ2 = 0.2;
      MinQ2 = 0.0;
      MaxQ3 = 0.2;
      MinQ3 = 0.0;
    }

    else {
      MaxX = Fx + 2.0*dX;
      MinX = Fx - 2.0*dX;
      MaxY = Fy + 2.0*dY;
      MinY = Fy - 2.0*dY;
      MaxZ = Fz + 2.0*dZ;
      MinZ = Fz - 2.0*dZ;

      MaxQ1 = FQ1 + 2.0*dQ1;
      MinQ1 = FQ1 - 2.0*dQ1;
      MaxQ2 = FQ2 + 2.0*dQ2;
      MinQ2 = FQ2 - 2.0*dQ2;
      MaxQ3 = FQ3 + 2.0*dQ3;
      MinQ3 = FQ3 - 2.0*dQ3;
    }

    dX = (MaxX - MinX)/(double)num1; 
    dY = (MaxY - MinY)/(double)num1; 
    dZ = (MaxZ - MinZ)/(double)num1; 
    dQ1 = (MaxQ1 - MinQ1)/(double)num2; 
    dQ2 = (MaxQ2 - MinQ2)/(double)num2; 
    dQ3 = (MaxQ3 - MinQ3)/(double)num2; 

    for (ix=0; ix<num1; ix++){
      x = MinX + ix*dX;
      for (iy=0; iy<num1; iy++){
	y = MinY + iy*dY;
	for (iz=0; iz<num1; iz++){
	  z = MinZ + iz*dZ;
	  for (iq1=0; iq1<num2; iq1++){
	    Q1 = MinQ1 + iq1*dQ1;
	    Sx = sin(Q1);
	    Cx = cos(Q1);

	    for (iq2=0; iq2<num2; iq2++){
	      Q2 = MinQ2 + iq2*dQ2;
	      Sy = sin(Q2);
	      Cy = cos(Q2);

	      for (iq3=0; iq3<num2; iq3++){
		Q3 = MinQ3 + iq3*dQ3;
		Sz = sin(Q3);
		Cz = cos(Q3);

		/* make a transform matrix */

		Rot[1][1] = Cy*Cz;
		Rot[1][2] = Sx*Sy*Cz - Cx*Sz;
		Rot[1][3] = Cx*Sy*Cz + Sx*Sz;

		Rot[2][1] = Cy*Sz;
		Rot[2][2] = Sx*Sy*Sz + Cx*Cz;
		Rot[2][3] = Cx*Sy*Sz - Sx*Cz;

		Rot[3][1] = -Sy;
		Rot[3][2] = Sx*Cy;
		Rot[3][3] = Cx*Cy;

		for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
		  xyz2[ct_AN][1] =  Rot[1][1]*(Gxyz2[ct_AN][1]-x) 
		                  + Rot[1][2]*(Gxyz2[ct_AN][2]-y)  
		                  + Rot[1][3]*(Gxyz2[ct_AN][3]-z);
  
		  xyz2[ct_AN][2] =  Rot[2][1]*(Gxyz2[ct_AN][1]-x) 
		                  + Rot[2][2]*(Gxyz2[ct_AN][2]-y)  
		                  + Rot[2][3]*(Gxyz2[ct_AN][3]-z);
  

		  xyz2[ct_AN][3] =  Rot[3][1]*(Gxyz2[ct_AN][1]-x) 
	 	                  + Rot[3][2]*(Gxyz2[ct_AN][2]-y)  
		                  + Rot[3][3]*(Gxyz2[ct_AN][3]-z);
		}

		dum = md(Gxyz1, xyz2);
		if (dum<Min_MD){
		  Min_MD = dum;
		  Fx = x;
		  Fy = y;
		  Fz = z;
		  FQ1 = Q1;
		  FQ2 = Q2;
		  FQ3 = Q3;

         printf(" trial=%2d x,y,z=%6.3f %6.3f %6.3f Q1,Q2,Q3=%6.3f %6.3f %6.3f MD=%12.9f\n",
                  num_trial,x,y,z,Q1/PI*180.0,Q2/PI*180.0,Q3/PI*180.0,dum);

		}

	      }
	    }
	  }
	}
      }
    }


    if ( fabs(Last_Min_MD-Min_MD)<1.0e-5 ) po = 1;

    printf(" MD (previous trial)=%15.12f MD (current trial)=%15.12f\n",
            Last_Min_MD,Min_MD); 

    Last_Min_MD = Min_MD; 

    num_trial++;

  } while (po==0); 

  printf("\n");
  printf(" Final MD (Ang/N)=%15.12f\n",Min_MD); 
  printf(" Translated x,y,z (Ang) of centroid = %6.3f %6.3f %6.3f\n",Fx,Fy,Fz); 
  printf(" Rotation Angles  Q1,Q2,Q3 (Deg)    = %6.3f %6.3f %6.3f\n",
          FQ1/PI*180.0,FQ2/PI*180.0,FQ3/PI*180.0);

  /* generate a vector file */

  Sx = sin(FQ1);
  Cx = cos(FQ1);
  Sy = sin(FQ2);
  Cy = cos(FQ2);
  Sz = sin(FQ3);
  Cz = cos(FQ3);

  Rot[1][1] = Cy*Cz;
  Rot[1][2] = Sx*Sy*Cz - Cx*Sz;
  Rot[1][3] = Cx*Sy*Cz + Sx*Sz;

  Rot[2][1] = Cy*Sz;
  Rot[2][2] = Sx*Sy*Sz + Cx*Cz;
  Rot[2][3] = Cx*Sy*Sz - Sx*Cz;

  Rot[3][1] = -Sy;
  Rot[3][2] = Sx*Cy;
  Rot[3][3] = Cx*Cy;

  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    xyz2[ct_AN][1] =  Rot[1][1]*(Gxyz2[ct_AN][1]-Fx) 
                    + Rot[1][2]*(Gxyz2[ct_AN][2]-Fy)  
                    + Rot[1][3]*(Gxyz2[ct_AN][3]-Fz);
  
    xyz2[ct_AN][2] =  Rot[2][1]*(Gxyz2[ct_AN][1]-Fx) 
                    + Rot[2][2]*(Gxyz2[ct_AN][2]-Fy)  
                    + Rot[2][3]*(Gxyz2[ct_AN][3]-Fz);
  

    xyz2[ct_AN][3] =  Rot[3][1]*(Gxyz2[ct_AN][1]-Fx) 
                    + Rot[3][2]*(Gxyz2[ct_AN][2]-Fy)  
                    + Rot[3][3]*(Gxyz2[ct_AN][3]-Fz);
  }

  printf(" generate a vector file, dgeo_vec.xsf\n\n");
  generate_vec(Gxyz1, xyz2);

  /* freeing of array */  
  for (ct_AN=0; ct_AN<=atomnum2; ct_AN++){
    free(xyz2[ct_AN]);
  }
  free(xyz2);
} 


double md(double **xyz1, double **xyz2)
{
  static int ct_AN;
  static double dx,dy,dz,r;
  static double result,sum;

  sum = 0.0;
  for (ct_AN=1; ct_AN<=atomnum1; ct_AN++){
    dx = xyz1[ct_AN][1] - xyz2[ct_AN][1];    
    dy = xyz1[ct_AN][2] - xyz2[ct_AN][2];    
    dz = xyz1[ct_AN][3] - xyz2[ct_AN][3];    
    r = sqrt(dx*dx + dy*dy + dz*dz);
    sum += r; 
  }

  result = sum/(double)atomnum1;
  
  return result;
}

void calc_md_BL()
{
  static int AN1,AN2,num;
  static double dx,dy,dz;
  static double r1,r2,dr,sum;

  num = 0;
  sum = 0.0;
  for (AN1=1; AN1<=atomnum1; AN1++){
    for (AN2=1; AN2<=atomnum2; AN2++){
       
      dx = Gxyz1[AN1][1] - Gxyz1[AN2][1];    
      dy = Gxyz1[AN1][2] - Gxyz1[AN2][2];    
      dz = Gxyz1[AN1][3] - Gxyz1[AN2][3];    
      r1 = sqrt(dx*dx + dy*dy + dz*dz);

      if (r1<BL0){
        num++;
        dx = Gxyz2[AN1][1] - Gxyz2[AN2][1];    
        dy = Gxyz2[AN1][2] - Gxyz2[AN2][2];    
        dz = Gxyz2[AN1][3] - Gxyz2[AN2][3];    
        r2 = sqrt(dx*dx + dy*dy + dz*dz);
        dr = r1 - r2;
        sum += sqrt( dr*dr );        
      }
 
    }
  }

  sum = sum/(double)num;
  printf(" For bonds with a bond lenth below %5.2f(Ang)\n",BL0);
  printf(" MD_BL(Ang/bond) = %15.12f\n\n",sum);
} 



void generate_vec(double **xyz1, double **xyz2)
{
  static int AN1;
  static double dx,dy,dz;
  static double xmin,ymin,zmin,xmax,ymax,zmax;
  static char fname[100];
  FILE *fp;
  char fp_buf[fp_bsize];          /* setvbuf */

  sprintf(fname,"dgeo_vec.xsf");

  if ((fp = fopen(fname,"w")) != NULL){

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    fprintf(fp,"ATOMS\n");
    for (AN1=1; AN1<=atomnum1; AN1++){
      dx = xyz2[AN1][1] - xyz1[AN1][1];    
      dy = xyz2[AN1][2] - xyz1[AN1][2];    
      dz = xyz2[AN1][3] - xyz1[AN1][3];    
      fprintf(fp,"%3s  %13.3E %13.3E %13.3E %13.3E %13.3E %13.3E\n",
	      Species[AN1],xyz1[AN1][1],xyz1[AN1][2],xyz1[AN1][3],
	      dx,dy,dz);
    }  
    fclose(fp);

  }
  else{
    printf("Failure of saving the xsf file.\n");
  }
  
} 




