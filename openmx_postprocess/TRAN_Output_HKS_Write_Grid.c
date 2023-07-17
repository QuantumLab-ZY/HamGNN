/**********************************************************************
  TRAN_Output_HKS_Write_Grid.c:

  TRAN_Output_HKS_Write_Grid.c is a subroutine to save and load data 
  on grid of leads

  Log of TRAN_Output_HKS_Write_Grid.c:

     11/Dec/2005   Released by H. Kino
     24/Apr/2012   Rewritten by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "tran_prototypes.h"

void TRAN_Output_HKS_Write_Grid(
				MPI_Comm comm1,
                                int mode,
				int Ngrid1,
				int Ngrid2,
				int Ngrid3,
				double *data,
				double *data1,
				double *data2,
				FILE *fp
				)
{
  int tag=99,size;
  int BN_AB,Nb,N2D,ID;
  int myid,numprocs;
  double *v;
  MPI_Request request;
  MPI_Status  stat;

  MPI_Comm_rank(comm1,&myid);
  MPI_Comm_size(comm1,&numprocs);

  N2D = Ngrid1*Ngrid2;

  /* allocation of array */
  size =  (((N2D+numprocs-1)/numprocs)+2)*Ngrid3;
  v = (double*)malloc(sizeof(double)*size); 

  /* save data */

  for (ID=0; ID<numprocs; ID++){

    Nb = (((ID+1)*N2D+numprocs-1)/numprocs)*Ngrid3
        -((ID*N2D+numprocs-1)/numprocs)*Ngrid3;

    /* set v and send v to Host_ID */

    if (ID==myid){
      if (mode==0){
        for (BN_AB=0; BN_AB<Nb; BN_AB++){
          v[BN_AB] = data[BN_AB];
        }
      }
      else if (mode==1){
        for (BN_AB=0; BN_AB<Nb; BN_AB++){
          v[BN_AB] = data[BN_AB] + data1[BN_AB] - 2.0*data2[BN_AB];
        }
      }

      if (myid!=Host_ID){
        MPI_Send(v, Nb, MPI_DOUBLE, Host_ID, tag, comm1);
      }  
    }

    /* receive v */

    if (myid==Host_ID) {
      if (myid!=ID){
	MPI_Recv(v, Nb, MPI_DOUBLE, ID, tag, comm1, &stat);
      }

      fwrite(v, sizeof(double), Nb, fp); 
    }
  }

  /* freeing of array */
  free(v);
}
