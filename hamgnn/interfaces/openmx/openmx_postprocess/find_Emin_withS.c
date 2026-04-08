#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double rnd(double width);
static double find_Emin(int n, double **a, double **s);

int main() 
{
  static int n,i,j;
  static double **a,**s,Emin;

  /* size of matrix */

  n = 10;

  /* allocation of array */

  a = (double**)malloc(sizeof(double*)*(n+3)); 
  for (i=0; i<(n+3); i++){
    a[i] = (double*)malloc(sizeof(double)*(n+3)); 
  }  

  s = (double**)malloc(sizeof(double*)*(n+3)); 
  for (i=0; i<(n+3); i++){
    s[i] = (double*)malloc(sizeof(double)*(n+3)); 
  }  

  for (i=1; i<=n; i++){
    for (j=i; j<=n; j++){
      a[i][j] = 0.0;
      s[i][j] = 0.0;
    }
  }


  a[1][1] = 1.0;
  a[1][2] = 1.0;
  a[2][1] = 1.0;
  a[2][2] =-1.0;
  a[2][3] = 0.5;
  a[3][2] = 0.5;
  a[3][3] =-0.5;

  s[1][1] = 1.0;
  s[1][2] = 0.1;
  s[2][1] = 0.1;
  s[2][2] = 1.0;
  s[2][3] = 0.2;
  s[3][2] = 0.2;
  s[3][3] = 1.0;

  Emin = find_Emin(3,a,s);

  printf("Emin=%18.15f\n",Emin);

  exit(0);

  /* initialize the matrix a */

  /*
  for (i=1; i<=n; i++){
    for (j=i; j<=n; j++){
      a[i][j] = rnd(1.0);
      a[j][i] = a[i][j];
    }
  }

  for (i=1; i<=n; i++){
    s[i][i] = 1.0;
    for (j=i+1; j<=n; j++){
      s[i][j] = rnd(0.1);
      s[j][i] = s[i][j];
    }
  }
  */

      
  /* call find_Emin */

  Emin = find_Emin(n,a,s);

  printf("Emin=%18.15f\n",Emin);

  /* freeing of array */

  for (i=0; i<(n+3); i++){
    free(a[i]);
  }  
  free(a);

  for (i=0; i<(n+3); i++){
    free(s[i]);
  }  
  free(s);

}




double find_Emin(int n, double **a, double **s)
{
  static int i,j,k,k1,k2,po,num;
  static double *v1,*v2;
  static double tmp,e0,e1;
  /* !!! change below for accuracy and efficiency !!! */
  static double diff=1.0e-10;
  static int nummax=10000; 

  /* allocation of arrays */

  v1 = (double*)malloc(sizeof(double)*(n+3)); 
  v2 = (double*)malloc(sizeof(double)*(n+3)); 

  /* initial vector */

  tmp = 1.0/sqrt((double)n);
  for (i=1; i<=n; i++) v1[i] = tmp;

  for (i=1; i<=n; i++){
    tmp = 0.0;
    for (j=1; j<=n; j++) tmp += s[i][j]*v1[j];
    v2[i] = tmp;        
  }

  tmp = 0.0;
  for (i=1; i<=n; i++)  tmp += v1[i]*v2[i];

  tmp = 1.0/sqrt(fabs(tmp));
  for (i=1; i<=n; i++)  v1[i] = v1[i]*tmp;

  /* steepest decent method */

  e0 = 1.0e+100;
  e1 = 0.0;
  po = 0;
  num = 0;

  do {

    num++;

    for (i=1; i<=n; i++){
      tmp = 0.0;
      for (j=1; j<=n; j++) tmp += (a[i][j] - e1*s[i][j])*v1[j];
      v2[i] = tmp;        
    }

    for (i=1; i<=n; i++) v1[i] -= 0.05*v2[i]; 

    for (i=1; i<=n; i++){
      tmp = 0.0;
      for (j=1; j<=n; j++) tmp += s[i][j]*v1[j];
      v2[i] = tmp;        
    }

    tmp = 0.0;
    for (i=1; i<=n; i++) tmp += v1[i]*v2[i];   

    tmp = 1.0/sqrt(fabs(tmp));

    for (i=1; i<=n; i++) v1[i] = v1[i]*tmp;
     
    for (i=1; i<=n; i++){
      tmp = 0.0;
      for (j=1; j<=n; j++) tmp += a[i][j]*v1[j];
      v2[i] = tmp;        
    }

    tmp = 0.0;
    for (i=1; i<=n; i++)  tmp += v1[i]*v2[i];   
    e1 = tmp;

    /* converge? */

    if (fabs(e1-e0)<diff) po = 1;
    else                  e0 = e1;


    printf("num=%5d  e1=%18.15f\n",num,e1);


  } while(po==0 && num<nummax);

  /* freeing of arrays */

  free(v1);
  free(v2);

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
