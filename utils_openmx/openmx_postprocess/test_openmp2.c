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


void dtime(double *t);


int main(int argc, char *argv[]) 
{


  int i,id,Nthrds,Nprocs,N,k,kp;
  double sum[100],tsum;
  double stime,etime;

  dtime(&stime);

  N = 30000000; 
  k = 0;


#pragma omp parallel shared(k,sum) private(i,id,kp,tsum)
{

  id = omp_get_thread_num();
  Nthrds = omp_get_num_threads();
  Nprocs = omp_get_num_procs();

  printf("Nprocs=%2d Nthrds=%2d id=%d\n",Nprocs,Nthrds,id);

  kp = id;
  k = Nthrds-1;

  do {

    sum[kp] = 0.0;
    for (i=0; i<N; i++){
      sum[kp] += sin((double)(i+kp))+cos((double)(i+2*kp));     
    }  

    printf("id=%2d kp=%2d sum=%15.12f\n",id,kp,sum[kp]);

#pragma omp critical
    kp = ++k;

  }
  while (k<20);

#pragma omp barrier  
{
  tsum = 0.0;  
  for (i=0; i<20; i++){
    tsum += sum[i];
  }
}      
  printf("tsum=%15.12f\n",tsum);

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

