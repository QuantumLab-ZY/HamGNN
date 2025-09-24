/*********************************************************************

 program for generating a structure of a bented carbon nanotube
 
   USAGE:

    ./BentNT < inp > out

    where inp and out are files.
    The file, inp, must be like this:
     
     5 5
     14.0
     10.0 

    '5 5' is the index of a nanotube,
    '14.0' is the length (ang.) of the nanotube,
    '10.0' is the radius (ang.) of curvature.

*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI      3.1415926535897932384626
#define asize1 200000
#define asize2  10000
     
double rd();

main()
{
  int i,j,k,i1,j1,num,tnum;
  int atomnum,nn;
  int bit[asize1];
  double sx,sy,sz;
  double sumx,sumy,sumz;
  double di,dj,dk;
  double m,n,l,C,r,theta,phi,dtheta;
  double x,y,z,length,dum;
  double dx,dy,dz,scaleF;
  double gx[asize1],gy[asize1];
  double nx[asize1],ny[asize1],nz[asize1];
  double tx[5],ty[5],ax[3],ay[3];
  double xyz[asize2][4]; 
  double curveR;

  /* Index of nanotube; (m n) */
  scanf("%lf %lf",&m,&n);

  /* Length (ang.) */
  scanf("%lf",&length);

  /* Radius of curvature (ang.) */
  scanf("%lf",&curveR);

  l = 1.42;

  theta = atan((n*sqrt(3.0)/2.0)/(m+0.50*n));
  phi = PI/6.0-theta;
  C = l*sqrt((m+0.50*n)*(m+0.50*n)*3.0+9.0/4.0*n*n);
  r = 0.50*(C-0.0000000000000)/PI;

  tx[1] = 0.0;
  ty[1] = 0.0;
  tx[2] = l*0.50;
  ty[2] = sqrt(3.0)/2.0*l;
  tx[3] = 1.50*l;
  ty[3] = sqrt(3.0)/2.0*l;
  tx[4] = 2.0*l;
  ty[4] = 0.0;
  ax[1] = 0.0;
  ay[1] = sqrt(3.0)*l;
  ax[2] = 3.0*l;
  ay[2] = 0.0;

  num = 0;
  di = -31;
  for (i=-30; i<=100; i++){
    di = di + 1.0;
    dj = -31;
    for (j=-30; j<=10; j++){
      dj = dj + 1.0;
      for (k=1; k<=4; k++){
        num++;

        if (asize1<=num){
          printf("asize1 is small.\n");
          exit(0);
        }

        gx[num] = tx[k] + di*ax[1] + dj*ax[2]; 
        gy[num] = ty[k] + di*ay[1] + dj*ay[2]; 
      }     
    }
  } 

  /* roll up */

  for (i=1; i<=num; i++){
    x = gx[i]*cos(-phi)-gy[i]*sin(-phi);
    y = gx[i]*sin(-phi)+gy[i]*cos(-phi);
    gx[i] = x;
    gy[i] = y;
  }

  tnum = 0;
  for (i=1; i<=num; i++){
   if (0<=gy[i] && gy[i]<=length){
     if (0<=gx[i] && gx[i]<=C){
       tnum++;
       dtheta = gx[i]/r;
       nx[tnum] = r*cos(dtheta);    
       ny[tnum] = r*sin(dtheta);
       nz[tnum] = gy[i];
     } 
   }
  }

  /* pick up */

  for (i=1; i<=tnum; i++){
    bit[i] = 0;
  }

  nn = 0;
  for (i=1; i<=tnum; i++){
   for (j=i+1; j<=tnum; j++){
     if (i!=j){
       dx = nx[i] - nx[j];
       dy = ny[i] - ny[j];
       dz = nz[i] - nz[j];
       dum = dx*dx + dy*dy + dz*dz;
       if (sqrt(dum)<0.3){
         bit[j] = 1;
         nn++;
       }
     }
   }
  }

  /* Translate the center of mass to origin*/

  atomnum = tnum - nn;

  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;
  for (i=1; i<=tnum; i++){
    if (bit[i]==0){ 
      sumx = sumx + nx[i];
      sumy = sumy + ny[i];
      sumz = sumz + nz[i];
    }
  }

  sx = sumx/(double)atomnum;
  sy = sumy/(double)atomnum;
  sz = sumz/(double)atomnum;

  num = 0;
  for (i=1; i<=tnum; i++){
    if (bit[i]==0){
      num++;       
      xyz[num][1] = nx[i] - sx + rd();
      xyz[num][2] = ny[i] - sy + rd();
      xyz[num][3] = nz[i] - sz + rd();
    }
  }

  /* bent */

  for (i=1; i<=atomnum; i++){
    dx = fabs(curveR - xyz[i][1]);
    theta = xyz[i][3]/dx;
    x = curveR + dx*cos(theta);
    y = xyz[i][2];
    z = dx*sin(theta);
    xyz[i][1] = x;
    xyz[i][2] = y;
    xyz[i][3] = z;
  }

  /*
  for (i=1; i<=atomnum; i++){
    dx = fabs(curveR - xyz[i][1]);
    scaleF = dx/curveR;
    theta = scaleF*xyz[i][3]/dx;
    x = curveR + dx*cos(theta);
    y = xyz[i][2];
    z = dx*sin(theta);
    xyz[i][1] = x;
    xyz[i][2] = y;
    xyz[i][3] = z;
  }
  */

  /* Translate the center of mass to origin*/
  sumx = 0.0;
  sumy = 0.0;
  sumz = 0.0;
  for (i=1; i<=atomnum; i++){
    sumx = sumx + xyz[i][1];
    sumy = sumy + xyz[i][2];
    sumz = sumz + xyz[i][3];
  }
  sx = sumx/(double)atomnum;
  sy = sumy/(double)atomnum;
  sz = sumz/(double)atomnum;

  for (i=1; i<=atomnum; i++){
    xyz[i][1] = xyz[i][1] - sx;
    xyz[i][2] = xyz[i][2] - sy;
    xyz[i][3] = xyz[i][3] - sz;
  }

  /* output */

  printf("%i\n",atomnum);
  for (i=1; i<=atomnum; i++){
    printf("%4d  C  %15.10f %15.10f %15.10f  2.0 2.0\n",
            i,xyz[i][3],xyz[i][1],xyz[i][2]);
  }

  /*
  printf("%i\n",atomnum);
  for (i=1; i<=atomnum; i++){
    printf("C  %15.10f %15.10f %15.10f\n",xyz[i][1],xyz[i][2],xyz[i][3]);
  }
  */

}


double rd()
{
  static double result,width;
  result = rand();
  /* This rd() function generates random number -width/2 to width/2 */
  width = 0.010;
  while (width<result){
     result = result/2.0;
  }
  result = result - width*0.75;
  return result;
}




