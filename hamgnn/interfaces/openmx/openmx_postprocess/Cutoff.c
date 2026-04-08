#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"


static int factorize(int num0, int N, int *fund, int *pow, int must );

void Determine_Cell_from_ECutoff(double tv[4][4], double ECut)
{
  /*********************************************************
     assume cubic cell, determine cell size from ECut 
     input::  tv, Ecut
     output:: tv (tuned) 
  *********************************************************/
   
  double a,len;
  int pow[4],must[4];
  int i,j,k,Scale[4];

  int myid,numprocs;
 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* check whether tv is a rectangular solid or not */
  for (i=1;i<=3;i++) Scale[i]=0;
  for (i=1;i<=3;i++) 
    for (j=1;j<=3;j++) 
      if (fabs(tv[i][j])<1.0e-6)  Scale[i]++;
  for (i=1;i<=3;i++) {
  /*  printf("Scale[%d]=%d\n",i,Scale[i]); */
    if (Scale[i]!=2) {
       printf("Error: Cell is not cubic\n");
       MPI_Finalize();
       exit(1);
    }  
  }
 
  a=PI/sqrt(ECut);

  must[1]=1;
  must[2]=1;
  must[3]=1;

  for (i=1; i<=3; i++) {
     /* printf("axis=%d\n",i); */
     len= sqrt( tv[i][1]*tv[i][1] + tv[i][2]*tv[i][2] +tv[i][3]*tv[i][3] ); 
     Scale[i] = (int) (len/a);

     /* factorize */     
     for (j=Scale[i]; ; j++) {
        /* printf("try %d\n",j); */
        if ( factorize(j, NfundamentalNum,fundamentalNum,pow,must[i])
             && ( (double)(a*j) >=len) ) {

           break; 
        }
     } 

     /* j*a = |tv[i]| */
     for (k=1;k<=3;k++) {
       tv[i][k]*= (a*j)/len;
     }
     Scale[i]=j;

  } 

    
  if (myid==Host_ID) {
    if (unitvector_unit==1) {
      printf("widened unit cell to fit energy cutoff (Bohr)\n");
      i=1;printf("A = %lf %lf %lf (%d)\n", tv[i][1],tv[i][2],tv[i][3],Scale[i]);
      i=2;printf("B = %lf %lf %lf (%d)\n", tv[i][1],tv[i][2],tv[i][3],Scale[i]);
      i=3;printf("C = %lf %lf %lf (%d)\n", tv[i][1],tv[i][2],tv[i][3],Scale[i]);
    }
    else if (unitvector_unit==0 ) {
      printf("widened unit cell to fit energy cutoff (Ang.)\n");
      i=1;printf("A = %lf %lf %lf (%d)\n", tv[i][1]*BohrR,tv[i][2]*BohrR,
		 tv[i][3]*BohrR,Scale[i]);
      i=2;printf("B = %lf %lf %lf (%d)\n", tv[i][1]*BohrR,tv[i][2]*BohrR,
		 tv[i][3]*BohrR,Scale[i]);
      i=3;printf("C = %lf %lf %lf (%d)\n", tv[i][1]*BohrR,tv[i][2]*BohrR,
		 tv[i][3]*BohrR,Scale[i]);
    }
  }

}


int factorize(int num0, int N, int *fund, int *pow, int must )
{
  int i;
  int a,b;
  int num;
  int ret;

  /* must exclude  division 0 */
  if (must==0)  return 0;

  if (must==1) {

    for (i=0;i<N;i++) {
      pow[i] = 0;
    }

  }
  else {
    num=num0%must;
    if ( num==0 ) {
      ret=factorize(  must, N, fund, pow, 1);
      if (ret==0) {
	return ret;
      }
    }
    else {
      return 0;
    }
  }

  num=num0/must ;
  for (i=0; i<N; i++) {
    while (1) {
      b = num%fund[i];

      if (b==0) {
	num /= fund[i];
	pow[i]++;
      }
      else {
	break;
      }
    }

  }

  if (num==1) {
    return num0;
  }
  else {
    return 0;
  }

}
 



#if 0
main()
{

  double tv[3][3],ECut;
  int i,j;

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      tv[ i][j]= 0.0;
     }
  }

  tv[0][0]=15.0;
  tv[1][1]=16.0;
  tv[2][2]=17.0;

  ECut=150.1;
  Determine_Cell_from_ECutoff(tv,ECut);

}

#endif

