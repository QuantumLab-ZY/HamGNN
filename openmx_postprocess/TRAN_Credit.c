/**********************************************************************
  TRAN_Credit.c:

  TRAN_Credit.c is a subroutine to show a credit.

  Log of TRAN_Credit.c:

     24/July/2008  Released by H.Kino and T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "tran_prototypes.h"

void TRAN_Credit(MPI_Comm comm1)
{

  int myid;

  MPI_Comm_rank(comm1,&myid);

  if (myid==Host_ID) {
    printf("\n***********************************************************\n"); 
    printf("***********************************************************\n"); 
    printf(" Welcome to the NEGF extension\n");
    printf(" The prototype fortran code: by Hisashi Kondo.\n");
    printf(" The current version: by Hiori Kino and Taisuke Ozaki.\n");
    printf("***********************************************************\n"); 
    printf("***********************************************************\n\n\n\n"); 
  }

}
