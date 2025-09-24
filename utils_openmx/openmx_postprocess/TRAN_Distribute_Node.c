/**********************************************************************
  TRAN_Distribute_Node.c:

  TRAN_Distribute_Node.c is a subroutine to set data for the MPI calculation.
  to the electrodes. 

  Log of TRAN_Distribute_Node.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void TRAN_Distribute_Node(int Start,int End,
			  int numprocs,
			  /* output */
			  int *IDStart,
			  int *IDEnd)
{

  int i;

  for (i=0;i<numprocs;i++) {
    IDStart[i] = Start + (End+1-Start)*i/(numprocs);
    IDEnd[i]   = Start + (End+1-Start)*(i+1)/(numprocs)-1;
    if (IDStart[i] >IDEnd[i]) {
      IDStart[i]=-1;
      IDEnd[i]= -1;
    }
  }  

  /* check consistency */

  /*
  for (i=0;i<numprocs-1;i++) {
    if ( IDStart[i+1]!=(IDEnd[i]+1) ) {
      printf("error i=%d IDEnd[i]=%d IDStart[i+1]=%d\n",i,IDEnd[i],IDStart[i+1]);fflush(stdout);
    }
  }
  */

  /*
  printf("IDStart   ");
  for (i=0;i<numprocs;i++) {
    printf("%3d ",IDStart[i]);fflush(stdout);
  }
  printf("\n");

  printf("IDEnd     ");
  for (i=0;i<numprocs;i++) {
    printf("%3d ",IDEnd[i]);fflush(stdout);
  }
  printf("\n");

  printf("IDEnd-IDStart+1 ");
  for (i=0;i<numprocs;i++) {
    printf("%3d ",IDEnd[i]-IDStart[i]+1);fflush(stdout);
  }
  printf("\n");
  */

}


void TRAN_Distribute_Node_Idx( int Start,
                               int End,
			       int numprocs,
			       int eachiwmax,
			       /* output */
			       int **Idxlist )
{
  int *start, *end;
  int ID;
  int i;

  start = (int*)malloc(sizeof(int)*numprocs);
  end  = (int*)malloc(sizeof(int)*numprocs);

  TRAN_Distribute_Node(Start,End,numprocs,start,end);

  for (ID=0;ID<numprocs;ID++) {
    if (eachiwmax < end[ID]-start[ID]+1 ) {
      printf("error in TRAN_Distribute_Node_Idx\n");
      exit(0);
    }
    for (i=0;i<eachiwmax; i++) Idxlist[ID][i]=-1; /* initialize */

    for (i=start[ID];i<=end[ID];i++) {
      Idxlist[ID][i-start[ID]] = i; /* set */
    }
  }

  free(end);
  free(start);
}




#if 0
main()
{
   int IDStart[100];
   int IDEnd[100];

   int **Idx;
   int numprocs=13;
   int i,j;

    int idxstart,idxend;
    int rowmax;

    idxstart=0;
    idxend=15;

   TRAN_Distribute_Node(idxstart,idxend,numprocs,IDStart,IDEnd);


   rowmax = (idxend-idxstart+1)/numprocs+1;
   printf("max=%d\n",rowmax);
   Idx = (int**)malloc(sizeof(int*)*numprocs); 
   for (i=0;i<numprocs;i++) {
       Idx[i]= (int*)malloc(sizeof(int*)*rowmax);
   }
   TRAN_Distribute_Node_Idx(idxstart,idxend,numprocs,rowmax, Idx);

   for (i=0;i<numprocs;i++) {
     for (j=0;j<rowmax;j++) {
        printf("%d ", Idx[i][j]);
     }
     printf("\n");
  }

  for (i=0;i<numprocs;i++) {
     free(Idx[i]);
  }
  free(Idx);
}

#endif

