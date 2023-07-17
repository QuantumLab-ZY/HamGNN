#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include <omp.h>


struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};


#define asize1   10
#define asize2    8
#define asize3    4

void dtime(double *t);


int main(int argc, char *argv[]) 
{


  int i,is,ie,id,N,NT;
  double sum;
  double stime,etime;

  dtime(&stime);

  N = 300000000; 
  NT = 4;

#pragma omp parallel private(i,id,is,ie,sum)
{

  id = omp_get_thread_num();
  is = id*N/NT;
  ie = (id+1)*N/NT;

  printf("%d is=%2d ie=%2d\n",id,is,ie);

  sum = 0.0;
  for (i=is; i<ie; i++){
    sum += sin((double)i)+cos((double)i);     
  }  
  printf("%15.12f\n",sum);
}

  dtime(&etime);

  printf("Elapsed time (s)=%15.12f\n",etime-stime);

  return 0;
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

