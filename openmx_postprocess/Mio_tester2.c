/*************************************************************

   a simple code for measuring the elapsed time for I/O 
   by Taisuke Ozaki (AIST-RICS), 28. Dec. 2005
 
   compiling:
    e.g.
       gcc -O3 io_tester.c -lm -o io_tester

   usage:
    ./io_tester

     Then, we will find the elapsed time for writing data to
     the disk space in your display like this:

     elapased time for I/O    5.83752 (s)

*************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 

#define Msize1      100
#define Msize2      200
#define Msize3      300
#define bsize       1048576

struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};


double rnd(double width);
void dtime(double *t);



int main(int argc, char *argv[])
{
  int i,j,k;
  double ***V;
  double stime,etime;
  FILE *fp;
  char fname[300];
  char buf[bsize];          /* setvbuf */

  /* allocate array */

  V = (double***)malloc(sizeof(double**)*Msize1); 
  for (i=0; i<Msize1; i++){
    V[i] = (double**)malloc(sizeof(double*)*Msize2); 
    for (j=0; j<Msize2; j++){
      V[i][j] = (double*)malloc(sizeof(double)*Msize3); 
    }
  }

  /* set data */

  for (i=0; i<Msize1; i++){
    for (j=0; j<Msize2; j++){
      for (k=0; k<Msize3; k++){
        V[i][j][k] = rnd(1.0);
      }
    }
  }

  /* write data */

  dtime(&stime);
  
  sprintf(fname,"IO_test.txt");
  if ((fp = fopen(fname,"w")) != NULL){

    setvbuf(fp,buf,_IOFBF,bsize);  /* setvbuf */

    for (i=0; i<Msize1; i++){
      for (j=0; j<Msize2; j++){
	for (k=0; k<Msize3; k++){
	  fprintf(fp,"%13.3E",V[i][j][k]);
	  if ((k+1)%6==0) fprintf(fp,"\n");
	}

	/* avoid double \n\n when Msize3%6 == 0  */
	if (Msize3%6!=0) fprintf(fp,"\n");
      }
    }

    fclose(fp);
  }
  else {
    printf("could not open a file\n");
  }

  dtime(&etime);
  printf("  elapased time for I/O %10.5f (s)\n",etime-stime);

  /* free array */

  for (i=0; i<Msize1; i++){
    for (j=0; j<Msize2; j++){
      free(V[i][j]);
    }
    free(V[i]);
  }
  free(V);

  /* return */

  return 0;
} 



double rnd(double width)
{
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
}
