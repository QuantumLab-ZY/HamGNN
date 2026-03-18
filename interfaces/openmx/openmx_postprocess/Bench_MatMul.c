#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 

#define  asize1  520


struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};


void dtime(double *t);
double rnd(double width);


int main(int argc, char *argv[]) 
{
  int i,j,k,l,i1,j1,n;
  double **a;
  double **b;
  double **c;
  double *a1;
  double *b1;
  double *c1;
  double sum,alpha,beta;
  double Stime,Etime;

  a = (double**)malloc(sizeof(double*)*asize1);
  for (i=0; i<asize1; i++){
    a[i] = (double*)malloc(sizeof(double)*asize1);
  }

  b = (double**)malloc(sizeof(double*)*asize1);
  for (i=0; i<asize1; i++){
    b[i] = (double*)malloc(sizeof(double)*asize1);
  }

  c = (double**)malloc(sizeof(double*)*asize1);
  for (i=0; i<asize1; i++){
    c[i] = (double*)malloc(sizeof(double)*asize1);
  }

  a1 = (double*)malloc(sizeof(double)*asize1*asize1);
  b1 = (double*)malloc(sizeof(double)*asize1*asize1);
  c1 = (double*)malloc(sizeof(double)*asize1*asize1);


  /*
  printf("a1\n");
  for (i=0; i<asize1; i++){
    for (j=0; j<asize1; j++){
      printf("%6.3f ",b1[j*asize1+i]);
    }
    printf("\n");
  }

  printf("b1\n");
  for (i=0; i<asize1; i++){
    for (j=0; j<asize1; j++){
      printf("%6.3f ",b1[j*asize1+i]);
    }
    printf("\n");
  }

  printf("c1\n");
  for (i=0; i<asize1; i++){
    for (j=0; j<asize1; j++){
      printf("%6.3f ",c1[j*asize1+i]);
    }
    printf("\n");
  }
  */


  for (l=0; l<5; l++){

    /* set up matrices */

    for (i=0; i<asize1; i++){
      for (j=0; j<asize1; j++){
	a[i][j] = rnd(1.0); 
	a1[j*asize1+i] = a[i][j];
      }
    }

    for (i=0; i<asize1; i++){
      for (j=0; j<asize1; j++){
	b[i][j] = rnd(1.0); 
	b1[j*asize1+i] = b[i][j];
      }
    }


    /* 3 */

    dtime(&Stime);
      
    n = asize1;
    alpha = 1.0;
    beta = 0.0;

    dgemm_("N","N", &n,&n, &n,&alpha, a1, &n, b1,&n, &beta,c1,&n);

    dtime(&Etime);
    printf("time(s) 3 %10.5f\n",Etime-Stime);


    /* 1 */

    dtime(&Stime);
  
    for (i=0; i<asize1; i++){
      for (j=0; j<asize1; j++){
   
	sum = 0.0;
	for (k=0; k<asize1; k++){

	  sum += a[i][k]*b[j][k];     

	  /*
	  sum += a[i][k]*b[k][j];     
	  */

	}

	c[i][j] = sum;
  
      }
    }

    dtime(&Etime);
    printf("time(s) 1 %10.5f\n",Etime-Stime);

    /*
    for (i=0; i<asize1; i++){
      for (j=0; j<asize1; j++){
        printf("%6.3f ",c[i][j]);
      } 
      printf("\n");
    } 
    printf("\n");
    */

    /* 2 */

    dtime(&Stime);
  
    for (i=0; i<asize1; i++){

      i1 = i*asize1;

      for (j=0; j<asize1; j++){

	j1 = j*asize1;   

	sum = 0.0;
	for (k=0; k<asize1; k++){
	  /*
	  sum += a1[i+k*asize1]*b1[k+j*asize1];
	  */

	  sum += a1[i1+k]*b1[j1+k];
	}

	c1[j*asize1+i] = sum;
      }
    }

    dtime(&Etime);
    printf("time(s) 2 %10.5f\n",Etime-Stime);

    /*
    for (i=0; i<asize1; i++){
      for (j=0; j<asize1; j++){
        printf("%6.3f ",c1[asize1*j+i]);
      } 
      printf("\n");
    } 
    printf("\n");
    */

    /*
    for (i=0; i<asize1; i++){
      for (j=0; j<asize1; j++){
        printf("%6.3f ",c1[asize1*j+i]);
      } 
      printf("\n");
    } 
    printf("\n");
    */

  }

  for (i=0; i<asize1; i++){
    free(a[i]);
  }
  free(a);

  for (i=0; i<asize1; i++){
    free(b[i]);
  }
  free(b);

  for (i=0; i<asize1; i++){
    free(c[i]);
  }
  free(c);

  free(a1);
  free(b1);
  free(c1);
 
}



double rnd(double width)
{
  /****************************************************
       This rnd() function generates random number
                -width/2 to width/2
  ****************************************************/

  double result;

  result = rand();

  while (width<result){
    result = result/2.0;
  }
  result = result - width*0.75;
  return result;
}


void dtime(double *t)
{

  /* real time */
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;

  /* user time + system time */
  /*
  float tarray[2];
  clock_t times(), wall;
  struct tms tbuf;
  wall = times(&tbuf);
  tarray[0] = (float) (tbuf.tms_utime / (float)CLK_TCK);
  tarray[1] = (float) (tbuf.tms_stime / (float)CLK_TCK);
  *t = (double) (tarray[0]+tarray[1]);
  printf("dtime: %lf\n",*t);
  */

}


