/**********************************************************************
  File_CntCoes.c:

     File_CntCoes.c is a subroutine to write and read a file which
     stores conctraction coefficients.

  Log of File_CntCoes.c:

     30/Oct/2003  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


static int   Input_CntCoes();
static void Output_CntCoes();

int File_CntCoes(char *mode)
{
  int ret;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  ret = 1;

  if (     strcasecmp(mode,"write")==0 )   Output_CntCoes();
  else if (strcasecmp(mode,"read" )==0 )   ret = Input_CntCoes();

  return ret;
}

int Input_CntCoes()
{
  int Mc_AN,Gc_AN,i,j,k;
  int my_check,check;
  int numprocs,myid;
  double *tmp0;
  char fileCC[YOUSO10];
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  check = 1;
  my_check = 1;

  /* allocation */
  tmp0 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[24]);

  for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

    Gc_AN = F_M2G[Mc_AN];

    sprintf(fileCC,"%s%s.ccs%i",filepath,filename,Gc_AN);

    if ((fp = fopen(fileCC,"r")) != NULL){

#ifdef xt3
     setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

     fread(tmp0,sizeof(double),List_YOUSO[7]*List_YOUSO[24],fp);

      k = 0;
      for (i=0; i<List_YOUSO[7]; i++){
        for (j=0; j<List_YOUSO[24]; j++){
          CntCoes[Mc_AN][i][j] = tmp0[k];
          k++;
        }
      }

      fclose(fp);
    }
    else {
      printf("Failure of making the ccs file (%s).\n",fileCC);
      my_check = 0;
    }
  }

  /* free */
  free(tmp0);

  MPI_Allreduce(&my_check, &check, 1, MPI_INT, MPI_PROD, mpi_comm_level1);

  if (check==0){
    printf(" myid=%2d  Failed in reading contraction coefficients\n",myid);
  }  

  return check;
}


void Output_CntCoes()
{
  int Mc_AN,Gc_AN,i,j,k;
  int numprocs,myid;
  double *tmp0;
  char fileCC[YOUSO10];
  FILE *fp; 
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation */
  tmp0 = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[24]);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    sprintf(fileCC,"%s%s.ccs%i",filepath,filename,Gc_AN);

    if ((fp = fopen(fileCC,"w")) != NULL){

#ifdef xt3
      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      k = 0;
      for (i=0; i<List_YOUSO[7]; i++){
        for (j=0; j<List_YOUSO[24]; j++){
          tmp0[k] = CntCoes[Mc_AN][i][j];          
          k++;
        }
      }

      fwrite(tmp0,sizeof(double),k,fp);
      fclose(fp);
    }
    else {
      printf("Failure of making the ccs file (%s).\n",fileCC);
    }
  }

  /* free */
  free(tmp0);
}





