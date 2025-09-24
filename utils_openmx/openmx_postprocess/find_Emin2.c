#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double rnd(double width);
static double find_Emin(int n, double **a);


int main() 
{

  static int n,i,j;
  static double **a,Emin,sum;

  /* size of matrix */

  n = 400;

  /* allocation of array */

  a = (double**)malloc(sizeof(double*)*(n+1)); 
  for (i=0; i<(n+1); i++){
    a[i] = (double*)malloc(sizeof(double)*(n+1)); 
  }  

  /* initialize the matrix a */

  /*
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      a[i][j] = rnd(1.0);
      a[j][i] = a[i][j];
    }
  }
  */

  for (i=1; i<=n; i++){
    for (j=i; j<=n; j++){
      a[i][j] = (double)i+(double)j;
      a[j][i] = a[i][j];
    }
  }


  sum = 0.0;
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      sum += fabs(a[i][j]); 
    }
  }
  printf("sum=%18.15f\n",sum);

      
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
  static int i,j,po,num,rl,rlnum;
  static double *v0,*v1,*v2;
  static double tmp,tmp0,tmp1,e0,e1;
  static double *al,*be;
  static double shiftE=5.0;
  /* !!! change below for accuracy and efficiency !!! */
  static double diff=1.0e-13;
  static int rlmax=20; 
  static int nummax=21000; 

  /* allocation of arrays */

  v0 = (double*)malloc(sizeof(double)*(n+1)); 
  v1 = (double*)malloc(sizeof(double)*(n+1)); 
  v2 = (double*)malloc(sizeof(double)*(n+1)); 
  al = (double*)malloc(sizeof(double)*(n+1)); 
  be = (double*)malloc(sizeof(double)*(n+1)); 

  /******************************************
     tridiagonalization by Lanczos method 
  ******************************************/

  /* initial vector */

  tmp = 1.0/sqrt((double)n);
  for (i=0; i<=n; i++){
    v0[i] = 0.0;
    v1[i] = tmp;
    v2[i] = 0.0;
    al[i] = 0.0;
    be[i] = 0.0;
  }

  if (n<rlmax) rlnum = n-1;
  else         rlnum = rlmax;

  for (rl=1; rl<=rlnum; rl++){
    
    /* a by v1 -> v2 */

    for (i=1; i<=n; i++){
      tmp = 0.0;
      for (j=1; j<=n; j++){
	tmp += a[i][j]*v1[j];
      }
      v2[i] = tmp;
    }

    /* a[rl] = <v1|v2> */

    tmp0 = 0.0; 
    for (i=1; i<=n; i++){
      tmp0 += v1[i]*v2[i];
    }
    al[rl] = tmp0;

    /* v2[i] = v2[i] - be[rl-1]*v0[i] - al[rl]*v1[i] */

    tmp0 = be[rl];
    tmp1 = al[rl];

    for (i=1; i<=n; i++){
      v2[i] = v2[i] - tmp0*v0[i] - tmp1*v1[i];
    }

    /* be_{rl+1}^2 = <v2|v2> */

    tmp0 = 0.0; 
    for (i=1; i<=n; i++){
      tmp0 += v2[i]*v2[i];
    }
    be[rl+1] = sqrt(tmp0);

    /* v1 -> v0, and v2/be[rl+1] -> v1 */

    tmp0 = be[rl+1];
    for (i=1; i<=n; i++){
      v0[i] = v1[i];
      v1[i] = v2[i]/tmp0;
    }
  }

  /*********************************************
   find the smallest eigenvalue by power method 
  *********************************************/

  /* shift diagonal parts */
  for (i=1; i<=rlnum; i++) al[i] -= shiftE;

  /* initial vector */

  for (i=0; i<=n; i++) v1[i] = 0.0;
  v1[1] = 1.0;

  /* power method */

  e0 = 1.0e+100;
  po = 0;
  num = 0;

  do {

    num++;

    /* tri_a by v1 -> v2 */

    for (i=1; i<=n; i++){
      v2[i] = be[i]*v1[i-1] + al[i]*v1[i] + be[i+1]*v1[i+1];
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

    /*
    printf("num=%6d e1=%18.15f\n",num,e1);
    */


  } while(po==0 && num<nummax);

  /* freeing of arrays */

  free(v0);
  free(v1);
  free(v2);
  free(al);
  free(be);

  /* back shist */

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
