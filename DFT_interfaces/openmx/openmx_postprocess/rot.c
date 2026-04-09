#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI            3.1415926535897932384626

int main(int argc, char *argv[]) 
{

  static int ct_AN;
  static double Gxyz2[10][10];
  static double xyz2[10][10];
  static double Q1,Q2,Q3;
  static double Sx,Sy,Sz,Cx,Cy,Cz;
  static double Rot[5][5],r;

  Gxyz2[1][1] = 0.0;
  Gxyz2[1][2] = 0.0;
  Gxyz2[1][3] = 0.0;

  Gxyz2[2][1] = 0.76;
  Gxyz2[2][2] = 0.59;
  Gxyz2[2][3] = 0.00;

  Gxyz2[3][1] = -0.76;
  Gxyz2[3][2] =  0.59;
  Gxyz2[3][3] =  0.00;

  Q1 = 30.0/180.0*PI;
  Q2 = 40.0/180.0*PI;
  Q3 = 50.0/180.0*PI;


  Sx = sin(Q1);
  Cx = cos(Q1);

  Sy = sin(Q2);
  Cy = cos(Q2);
  
  Sz = sin(Q3);
  Cz = cos(Q3);

  Rot[1][1] = Cy*Cz;
  Rot[1][2] = Sx*Sy*Cz - Cx*Sz;
  Rot[1][3] = Cx*Sy*Cz + Sx*Sz;

  Rot[2][1] = Cy*Sz;
  Rot[2][2] = Sx*Sy*Sz + Cx*Cz;
  Rot[2][3] = Cx*Sy*Sz - Sx*Cz;

  Rot[3][1] = -Sy;
  Rot[3][2] = Sx*Cy;
  Rot[3][3] = Cx*Cy;

  for (ct_AN=1; ct_AN<=3; ct_AN++){

    xyz2[ct_AN][1] =  Rot[1][1]*Gxyz2[ct_AN][1] 
                    + Rot[1][2]*Gxyz2[ct_AN][2]  
                    + Rot[1][3]*Gxyz2[ct_AN][3];
  
    xyz2[ct_AN][2] =  Rot[2][1]*Gxyz2[ct_AN][1] 
                    + Rot[2][2]*Gxyz2[ct_AN][2]  
                    + Rot[2][3]*Gxyz2[ct_AN][3];
  

    xyz2[ct_AN][3] =  Rot[3][1]*Gxyz2[ct_AN][1] 
                    + Rot[3][2]*Gxyz2[ct_AN][2]  
                    + Rot[3][3]*Gxyz2[ct_AN][3];

    r = sqrt(  xyz2[ct_AN][1]*xyz2[ct_AN][1] 
             + xyz2[ct_AN][2]*xyz2[ct_AN][2]
             + xyz2[ct_AN][3]*xyz2[ct_AN][3]
             );

    //  printf("r=%15.12f\n",r); 
    printf(" %15.12f %15.12f %15.12f\n",xyz2[ct_AN][1],xyz2[ct_AN][2],xyz2[ct_AN][3]);
  }

}
