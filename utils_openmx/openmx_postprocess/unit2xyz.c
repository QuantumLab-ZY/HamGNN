#define PI            3.1415926535897932384626

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void fnjoint();
void chcp();
double sgn();
void ATV(double atvx[15],double atvy[15], double atvz[15],
         double tvx[5],double tvy[5],double tvz[5]);

main()
{
  static int atomnum,n,i,j,k,l,m,i1,i2,j1,j2;
  static double za0,za1,za2,za3;
  static double ka1,ka2,ka3,al,be,ga,c1,c2,c3,s2,s3;
  static double **xyz,rxyz[10],tvxyz[5][4];
  static double tvx[5],tvy[5],tvz[5];
  static double atvx[15],atvy[15],atvz[15];
  static double la1,la2,la3,ang1,ang2,ang3;
  static double r,dx,dy,dz; 
  static char Name[100];

 /*-----------  input data  ------------------------------------*/

  scanf("%lf %lf %lf",&la1,&la2,&la3);
  scanf("%lf %lf %lf",&ang1,&ang2,&ang3);
  scanf("%i",&atomnum);

  xyz = (double**)malloc(sizeof(double*)*(atomnum+1)); 
  for (i=0; i<=atomnum; i++){
    xyz[i] = (double*)malloc(sizeof(double)*4); 
  }

  ang1 = PI*ang1/180.0;
  ang2 = PI*ang2/180.0;
  ang3 = PI*ang3/180.0;

  tvx[1] = la1*cos(ang2);
  tvz[1] = (la1*cos(ang3)-tvx[1]*cos(ang1))/sin(ang1);
  tvy[1] = sqrt(la1*la1-tvx[1]*tvx[1]-tvz[1]*tvz[1]);
  tvx[2] = la2*cos(ang1);
  tvy[2] = 0;
  tvz[2] = la2*sin(ang1);
  tvx[3] = la3;
  tvy[3] = 0;
  tvz[3] = 0;

  ATV(atvx,atvy,atvz,tvx,tvy,tvz);

  for (i=1; i<=atomnum; i++){
    scanf("%lf  %s %lf %lf %lf",&za0,&Name,&za1,&za2,&za3);
    xyz[i][0] = za0;
    xyz[i][1] = za1*tvx[1] + za2*tvx[2] + za3*tvx[3];
    xyz[i][2] = za1*tvy[1] + za2*tvy[2] + za3*tvy[3];
    xyz[i][3] = za1*tvz[1] + za2*tvz[2] + za3*tvz[3];
    printf("%4d   %3s %16.12f  %16.12f  %16.12f\n",
             i,Name,xyz[i][1],xyz[i][2],xyz[i][3]);
  }

  /***************************
   distance between two atoms
  ***************************/

  /*
  for (i=1; i<=atomnum; i++){
    for (j=1; j<=atomnum; j++){
      if (i!=j){
        dx = xyz[i][1] - xyz[j][1];
        dy = xyz[i][2] - xyz[j][2];
        dz = xyz[i][3] - xyz[j][3];
        r = sqrt(dx*dx + dy*dy + dz*dz);
        printf("Atoms %4d and %4d   R=%15.12f\n",i,j,r);
      }
    }
  }
  */

  /* freeing of array */
  for (i=0; i<=atomnum; i++){
    free(xyz[i]);
  }
  free(xyz);
}

void ATV(double atvx[15],double atvy[15], double atvz[15],
         double tvx[5],double tvy[5],double tvz[5])

{

  atvx[0] = 0;
  atvy[0] = 0;
  atvz[0] = 0;
  atvx[1] = tvx[1];
  atvy[1] = tvy[1];
  atvz[1] = tvz[1];
  atvx[2] = tvx[2];
  atvy[2] = tvy[2];
  atvz[2] = tvz[2];
  atvx[3] = tvx[3];
  atvy[3] = tvy[3];
  atvz[3] = tvz[3];
  atvx[4] = tvx[1] + tvx[3];
  atvy[4] = tvy[1] + tvy[3];
  atvz[4] = tvz[1] + tvz[3];
  atvx[5] = tvx[1] - tvx[3];
  atvy[5] = tvy[1] - tvy[3];
  atvz[5] = tvz[1] - tvz[3];
  atvx[6] = tvx[1] + tvx[2];
  atvy[6] = tvy[1] + tvy[2];
  atvz[6] = tvz[1] + tvz[2];
  atvx[7] = tvx[1] - tvx[2];
  atvy[7] = tvy[1] - tvy[2];
  atvz[7] = tvz[1] - tvz[2];
  atvx[8] = tvx[2] + tvx[3];
  atvy[8] = tvy[2] + tvy[3];
  atvz[8] = tvz[2] + tvz[3];
  atvx[9] = tvx[2] - tvx[3];
  atvy[9] = tvy[2] - tvy[3];
  atvz[9] = tvz[2] - tvz[3];
  atvx[10] = tvx[1] + tvx[2] + tvx[3];
  atvy[10] = tvy[1] + tvy[2] + tvy[3];
  atvz[10] = tvz[1] + tvz[2] + tvz[3];
  atvx[11] = -tvx[1] + tvx[2] + tvx[3];
  atvy[11] = -tvy[1] + tvy[2] + tvy[3];
  atvz[11] = -tvz[1] + tvz[2] + tvz[3];
  atvx[12] = tvx[1] - tvx[2] + tvx[3];
  atvy[12] = tvy[1] - tvy[2] + tvy[3];
  atvz[12] = tvz[1] - tvz[2] + tvz[3];
  atvx[13] = tvx[1] + tvx[2] - tvx[3];
  atvy[13] = tvy[1] + tvy[2] - tvy[3];
  atvz[13] = tvz[1] + tvz[2] - tvz[3];

}


void fnjoint(char name1[100],char name2[100],char name3[100])
{
 static char name4[100];
	char *f1 = name1,
	     *f2 = name2,
	     *f3 = name3,
	     *f4 = name4;

   while(*f1)
    {
     *f4 = *f1;
     *f1++;
     *f4++;
    }
   while(*f2)
    {
     *f4 = *f2;
     *f2++;
     *f4++;
    }
   while(*f3)
    {
     *f4 = *f3;
     *f3++;
     *f4++;
    }
    *f4 = *f3;
    chcp(name3,name4);
}

void chcp(char name1[100],char name2[100])
{
	  char *f1 = name1,
	       *f2 = name2;

   while(*f2)
    {
     *f1 = *f2;
     *f1++;
     *f2++;
    }
    *f1 = *f2;
}


double sgn(double nu)
 {
 double result;
  if (nu<0)
    result = -1;
   else
    result = 1;
  return result;
 }

