#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double rnd(double width);
static double find_Emin(int n, double **a);


int main() 
{

  static int n,i,j;
  static double **a,Emin;

  /* size of matrix */

  n = 2000;

  /* allocation of array */

  a = (double**)malloc(sizeof(double*)*(n+1)); 
  for (i=0; i<(n+1); i++){
    a[i] = (double*)malloc(sizeof(double)*(n+1)); 
  }  

  /* initialize the matrix a */

  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      a[i][j] = rnd(1.0);
      a[j][i] = a[i][j];
    }
  }
      
  /* initialize the matrix a */

  Emin = find_Emin(n,a);

  printf("Emin=%18.15f\n",Emin);

  /* freeing of array */

  for (i=0; i<(n+1); i++){
    free(a[i]);
  }  
  free(a);

}




double find_Emin(int n, double **a)
{
  static int i,j,po,num;
  static double *v1,*v2,tmp,e0,e1;
  static double diff=1.0e-13;
  static double shiftE=35.0;
  static int nummax=31000; 

  /* allocation of arrays */

  v1 = (double*)malloc(sizeof(double)*(n+1)); 
  v2 = (double*)malloc(sizeof(double)*(n+1)); 

  /* shift diagonal parts */
  for (i=1; i<=n; i++) a[i][i] -= shiftE;

  /* initial vector */

  tmp = 1.0/sqrt((double)n);
  for (i=1; i<=n; i++) v1[i] = tmp;

  /* power method */

  e0 = 1.0e+100;
  po = 0;
  num = 0;

  do {

    num++;

    /* a by v1 -> v2 */

    for (i=1; i<=n; i++){
      tmp = 0.0;
      for (j=1; j<=n; j++){
	tmp += a[i][j]*v1[j];
      }
      v2[i] = tmp;
    }

    /* e1 = <v1|v2> */

    e1 = 0.0;
    for (i=1; i<=n; i++){
      e1 += v1[i]*v2[i];
    }

    /* converge? */

    if (fabs(e1-e0)<diff) po = 1;
    else                  e0 = e1;

    /* normalize v2 -> v1 */

    tmp = 0.0;
    for (i=1; i<=n; i++){
      tmp += v2[i]*v2[i];
    }

    tmp = 1.0/sqrt(tmp);
    for (i=1; i<=n; i++){
      v1[i] = tmp*v2[i];
    }

    printf("num=%6d e1=%18.15f\n",num,e1);


  } while(po==0 && num<nummax);

  /* freeing of arrays */

  free(v1);
  free(v2);

  /* back shift */

  for (i=1; i<=n; i++) a[i][i] += shiftE;
  e1 += shiftE;

  /* return */

  return e1;
}







double rnd(double width)
{

  /****************************************************
       This rnd() function generates random number
                -width/2 to width/2
  ****************************************************/

  static double result;

  result = rand();

  while (width<result){
    result = result/2.0;
  }
  result = result - width*0.75;
  return result;
}
