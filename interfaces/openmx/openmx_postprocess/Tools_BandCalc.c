/**********************************************************************

 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <complex.h>

#include "lapack_prototypes.h"
#include "read_scfout.h"
#include "Tools_BandCalc.h"


double kgrid_dist(double x1, double x2, double y1, double y2, double z1, double z2)
{
  double k1,k2,k3;
  k1 = rtv[1][1]*(x2 -x1) +rtv[2][1]*(y2 -y1) +rtv[3][1]*(z2 -z1);
  k2 = rtv[1][2]*(x2 -x1) +rtv[2][2]*(y2 -y1) +rtv[3][2]*(z2 -z1);
  k3 = rtv[1][3]*(x2 -x1) +rtv[2][3]*(y2 -y1) +rtv[3][3]*(z2 -z1);

  return sqrt(k1*k1 +k2*k2 +k3*k3);
}


void Print_kxyzEig(char *Pdata_s, double kx, double ky, double kz, int l, double EIG){
  int i,j,k;
  sprintf(Pdata_s, "%10.6lf  %10.6lf  %10.6lf     %6d  %10.6lf "
      ,rtv[1][1]*kx+ rtv[2][1]*ky+ rtv[3][1]*kz
      ,rtv[1][2]*kx+ rtv[2][2]*ky+ rtv[3][2]*kz
      ,rtv[1][3]*kx+ rtv[2][3]*ky+ rtv[3][3]*kz, l, EIG);
}


void dtime(double *t){
  // real time 
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;
}


void name_Nband(char *fname1, char *fname2, int l){
  int i,j,k;
  char *num;
  double a;

  a = log10((double)l);
  i = a;
  num = (char*)malloc(sizeof(char)*i+2);
  for(j=i;j>=0;j--){
    k = l%10;
    num[j] = (char)(k+48);
    l/= 10;
  }num[i+1] = '\0';
  //printf("kotaka %s\n",num);

  //  strcpy(fname1,fname_out);
  strcat(fname1,fname2);
  strcat(fname1,num);
  //  printf("Output Filename is \"%s\"\n",fname1);
  free(num);
}


void k_inversion(int i,  int j,  int k, 
    int mi, int mj, int mk, 
    int *ii, int *ij, int *ik )
{
  *ii= mi-i-1;
  *ij= mj-j-1;
  *ik= mk-k-1;
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



void xyz2spherical(double x, double y, double z,
    double xo, double yo, double zo,
    double S_coordinate[3])
{
  double dx,dy,dz,r,r1,theta,phi,dum,dum1,Min_r;

  Min_r = 10e-15;

  dx = x - xo;  dy = y - yo;  dz = z - zo;

  dum = dx*dx + dy*dy; 
  r = sqrt(dum + dz*dz);
  r1 = sqrt(dum);

  if (Min_r<=r){

    if (r<fabs(dz))
      dum1 = sgn(dz)*1.0;
    else
      dum1 = dz/r;

    theta = acos(dum1);

    if (Min_r<=r1){
      if (0.0<=dx){

	if (r1<fabs(dy))
	  dum1 = sgn(dy)*1.0;
	else
	  dum1 = dy/r1;        

	phi = asin(dum1);
      }
      else{

	if (r1<fabs(dy))
	  dum1 = sgn(dy)*1.0;
	else
	  dum1 = dy/r1;        

	phi = PI - asin(dum1);
      }
    }
    else{
      phi = 0.0;
    }
  }
  else{
    theta = 0.5*PI;
    phi = 0.0;
  }

  S_coordinate[0] = r;  S_coordinate[1] = theta;  S_coordinate[2] = phi;
}



void EulerAngle_Spin( int quickcalc_flag,
    double Re11, double Re22,
    double Re12, double Im12, double Re21, double Im21,
    double Nup[2], double Ndown[2], double t[2], double p[2] )
{
  double phi,theta,d1,d2,d3,d4;
  double cop,sip,sit,cot,tmp,tmp1,tmp2,prod;
  double mx,my,mz,tn,absm;
  double S_coordinate[3];
  dcomplex bunbo,bunsi;
  dcomplex cd1,cd2,cd3,cd4,cNup,cNdown;

  /* Disabled by N. Yamaguchi
   * double complex ctmp1,ctheta,csit,ccot;
   *
   *
   * if (fabs(Re12)<1.0e-14){
   * phi = PI*90.0/180.0;
   * }
   * else{
   *
   * bunbo.r = Re12 + Re21;
   * bunbo.i = Im12 + Im21;
   * bunsi.r = Re12 - Re21;
   * bunsi.i = Im12 - Im21;
   *
   * tmp = -(bunsi.i*bunbo.r - bunsi.r*bunbo.i)/(bunbo.r*bunbo.r+bunbo.i*bunbo.i);
   * phi = atan(tmp);
   * }
   *
   * cop = cos(phi);
   * sip = sin(phi);
   *
   * if (fabs(Re11 - Re22)<1.0e-14){
   * ctheta = PI*90.0/180.0 + 0.0*I;
   * }
   * else {
   * tmp1 = (Re12*cop - Im12*sip + Re21*cop + Im21*sip)/(Re11 - Re22);
   * tmp2 = (Re12*sip + Im12*cop - Re21*sip + Im21*cop)/(Re11 - Re22);
   *
   * ctmp1 = tmp1 + tmp2*I;
   * ctheta = catan(ctmp1);
   * }
   *
   * csit = csin(ctheta);
   * ccot = ccos(ctheta);
   *
   * cd1.r = 0.5*(Re11 + Re22);
   * cd1.i = 0.0;
   *
   * cd2.r = 0.5*creal(ccot)*(Re11 - Re22);
   * cd2.i = 0.5*cimag(ccot)*(Re11 - Re22);
   *
   * cd3.r = 0.5*( (Re12*creal(csit)-Im12*cimag(csit))*cop
   * -(Re12*cimag(csit)+Im12*creal(csit))*sip );
   *
   * cd3.i = 0.5*( (Re12*creal(csit)-Im12*cimag(csit))*sip
   * +(Re12*cimag(csit)+Im12*creal(csit))*cop );
   *
   * cd4.r = 0.5*( (Re21*creal(csit)-Im21*cimag(csit))*cop
   * +(Re21*cimag(csit)+Im21*creal(csit))*sip );
   *
   * cd4.i = 0.5*(-(Re21*creal(csit)-Im21*cimag(csit))*sip
   * +(Re21*cimag(csit)+Im21*creal(csit))*cop );
   *
   * cNup.r   = cd1.r + cd2.r + cd3.r + cd4.r;
   * cNup.i   = cd1.i + cd2.i + cd3.i + cd4.i;
   * cNdown.r = cd1.r - cd2.r - cd3.r - cd4.r;
   * cNdown.i = cd1.i - cd2.i - cd3.i - cd4.i;
   *
   * Nup[0] = cNup.r;
   * Nup[1] = cNup.i;
   *
   * Ndown[0] = cNdown.r;
   * Ndown[1] = cNdown.i;
   *
   * t[0] = creal(ctheta);
   * t[1] = cimag(ctheta);
   *
   * p[0] = phi;
   * //  p[1] = 0.0;
   */

  mx =  2.0*Re12;
  my = -2.0*Im12;
  mz = Re11 - Re22;

  xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

  absm = S_coordinate[0];
  theta = S_coordinate[1];
  phi = S_coordinate[2];
  tn = Re11 + Re22;

  Nup[0]   = 0.5*(tn + absm);
  Nup[1]   = 0.0;

  Ndown[0] = 0.5*(tn - absm);
  Ndown[1] = 0.0;

  t[0] = theta;
  t[1] = 0.0;

  p[0] = phi;
  p[1] = 0.0;

}



