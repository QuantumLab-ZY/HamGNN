/**********************************************************************
  Output_CompTime.c:

     Output_CompTime.c is a subrutine to write computational time
     to filename.TRN.

  Log of Output_CompTime.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

#define Num_CompTime 21

void Output_CompTime()
{
  int ID,j;
  int ID_Mintime1,ID_Maxtime1;
  int MinID[Num_CompTime];
  int MaxID[Num_CompTime];
  double Mintime1,Maxtime1;
  double *time0;
  double *time1;
  double MinCompTime[Num_CompTime];
  double MaxCompTime[Num_CompTime];
  char file_CompTime[YOUSO10] = ".CompTime";
  FILE *fp;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocate arrays */

  time0 = (double*)malloc(sizeof(double)*numprocs);
  time1 = (double*)malloc(sizeof(double)*numprocs);

  /* MPI_Bcast CompTime */

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&CompTime[ID][0], Num_CompTime, MPI_DOUBLE, ID, mpi_comm_level1);
  }

  MPI_Barrier(mpi_comm_level1);

  /* find the min and max CompTime */
  for (j=0; j<Num_CompTime; j++){
    MaxCompTime[j] = -10.0;
    MinCompTime[j] = 1.0e+10;

    for (ID=0; ID<numprocs; ID++){ 
      if (CompTime[ID][j]<MinCompTime[j]){
        MinCompTime[j] = CompTime[ID][j];
        MinID[j] = ID;
      }         
      if (MaxCompTime[j]<CompTime[ID][j]){
        MaxCompTime[j] = CompTime[ID][j];
        MaxID[j] = ID;
      }         
    }
  }

  if (myid==Host_ID){

    fnjoint(filepath,filename,file_CompTime);

    for (ID=0; ID<numprocs; ID++){
      time0[ID] = CompTime[ID][5]
                + CompTime[ID][6]
                + CompTime[ID][7]
                + CompTime[ID][8]
                + CompTime[ID][9]
                + CompTime[ID][10]
                + CompTime[ID][11]
                + CompTime[ID][12]
                + CompTime[ID][13]
                + CompTime[ID][14]
                + CompTime[ID][15]
                + CompTime[ID][16]
                + CompTime[ID][17]
                + CompTime[ID][18]
                + CompTime[ID][19];

      time1[ID] = CompTime[ID][3] - time0[ID];
    }

    Maxtime1 = -100.0; 
    Mintime1 = 1.0e+10;

    for (ID=0; ID<numprocs; ID++){
      if (time1[ID]<Mintime1){
        Mintime1 = time1[ID];
        ID_Mintime1 = ID;
      }         
      if (Maxtime1<time1[ID]){
        Maxtime1 = time1[ID];
        ID_Maxtime1 = ID;
      }         
    }    

    if ((fp = fopen(file_CompTime,"w")) != NULL){

      /* write */ 

      fprintf(fp,"\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"               Computational Time (second)                 \n");
      fprintf(fp,"***********************************************************\n");
      fprintf(fp,"***********************************************************\n\n");

      fprintf(fp,"   Elapsed.Time.  %12.3f\n\n",MaxCompTime[0]);

      fprintf(fp,"                               Min_ID   Min_Time       Max_ID   Max_Time\n");

      fprintf(fp,"   Total Computational Time = %5d %12.3f      %5d %12.3f\n",
              MinID[0],MinCompTime[0],MaxID[0],MaxCompTime[0]);
      fprintf(fp,"   readfile                 = %5d %12.3f      %5d %12.3f\n",
              MinID[1],MinCompTime[1],MaxID[1],MaxCompTime[1]);
      fprintf(fp,"   truncation               = %5d %12.3f      %5d %12.3f\n",
              MinID[2],MinCompTime[2],MaxID[2],MaxCompTime[2]);
      fprintf(fp,"   MD_pac                   = %5d %12.3f      %5d %12.3f\n",
              MinID[4],MinCompTime[4],MaxID[4],MaxCompTime[4]);
      fprintf(fp,"   OutData                  = %5d %12.3f      %5d %12.3f\n",
              MinID[20],MinCompTime[20],MaxID[20],MaxCompTime[20]);
      fprintf(fp,"   DFT                      = %5d %12.3f      %5d %12.3f\n",
              MinID[3],MinCompTime[3],MaxID[3],MaxCompTime[3]);
      fprintf(fp,"\n");
      fprintf(fp,"*** In DFT ***\n\n");
      fprintf(fp,"   Set_OLP_Kin              = %5d %12.3f      %5d %12.3f\n",
              MinID[5],MinCompTime[5],MaxID[5],MaxCompTime[5]);
      fprintf(fp,"   Set_Nonlocal             = %5d %12.3f      %5d %12.3f\n",
              MinID[6],MinCompTime[6],MaxID[6],MaxCompTime[6]);
      fprintf(fp,"   Set_ProExpn_VNA          = %5d %12.3f      %5d %12.3f\n",
	       MinID[16],MinCompTime[16],MaxID[16],MaxCompTime[16]);
      fprintf(fp,"   Set_Hamiltonian          = %5d %12.3f      %5d %12.3f\n",
              MinID[7],MinCompTime[7],MaxID[7],MaxCompTime[7]);
      fprintf(fp,"   Poisson                  = %5d %12.3f      %5d %12.3f\n",
              MinID[8],MinCompTime[8],MaxID[8],MaxCompTime[8]);
      fprintf(fp,"   Diagonalization          = %5d %12.3f      %5d %12.3f\n",
              MinID[9],MinCompTime[9],MaxID[9],MaxCompTime[9]);
      fprintf(fp,"   Mixing_DM                = %5d %12.3f      %5d %12.3f\n",
              MinID[10],MinCompTime[10],MaxID[10],MaxCompTime[10]);
      fprintf(fp,"   Force                    = %5d %12.3f      %5d %12.3f\n",
              MinID[11],MinCompTime[11],MaxID[11],MaxCompTime[11]);
      fprintf(fp,"   Total_Energy             = %5d %12.3f      %5d %12.3f\n",
              MinID[12],MinCompTime[12],MaxID[12],MaxCompTime[12]);
      fprintf(fp,"   Set_Aden_Grid            = %5d %12.3f      %5d %12.3f\n",
              MinID[13],MinCompTime[13],MaxID[13],MaxCompTime[13]);
      fprintf(fp,"   Set_Orbitals_Grid        = %5d %12.3f      %5d %12.3f\n",
              MinID[14],MinCompTime[14],MaxID[14],MaxCompTime[14]);
      fprintf(fp,"   Set_Density_Grid         = %5d %12.3f      %5d %12.3f\n",
              MinID[15],MinCompTime[15],MaxID[15],MaxCompTime[15]);
      fprintf(fp,"   RestartFileDFT           = %5d %12.3f      %5d %12.3f\n",
              MinID[17],MinCompTime[17],MaxID[17],MaxCompTime[17]);
      fprintf(fp,"   Mulliken_Charge          = %5d %12.3f      %5d %12.3f\n",
              MinID[18],MinCompTime[18],MaxID[18],MaxCompTime[18]);
      fprintf(fp,"   FFT(2D)_Density          = %5d %12.3f      %5d %12.3f\n",
              MinID[19],MinCompTime[19],MaxID[19],MaxCompTime[19]);
      fprintf(fp,"   Others                   = %5d %12.3f      %5d %12.3f\n",
              ID_Mintime1,Mintime1,ID_Maxtime1,Maxtime1);
      fclose(fp);

      /* stdout */ 

      if (0<level_stdout){

	printf("\n");
	printf("***********************************************************\n");
	printf("***********************************************************\n");
	printf("               Computational Time (second)                 \n");
	printf("***********************************************************\n");
	printf("***********************************************************\n\n");

        printf("                               Min_ID   Min_Time       Max_ID   Max_Time\n");
	printf("   Total Computational Time = %5d %12.3f      %5d %12.3f\n",
	       MinID[0],MinCompTime[0],MaxID[0],MaxCompTime[0]);
	printf("   readfile                 = %5d %12.3f      %5d %12.3f\n",
	       MinID[1],MinCompTime[1],MaxID[1],MaxCompTime[1]);
	printf("   truncation               = %5d %12.3f      %5d %12.3f\n",
	       MinID[2],MinCompTime[2],MaxID[2],MaxCompTime[2]);
	printf("   MD_pac                   = %5d %12.3f      %5d %12.3f\n",
	       MinID[4],MinCompTime[4],MaxID[4],MaxCompTime[4]);
        printf("   OutData                  = %5d %12.3f      %5d %12.3f\n",
              MinID[20],MinCompTime[20],MaxID[20],MaxCompTime[20]);
	printf("   DFT                      = %5d %12.3f      %5d %12.3f\n",
	       MinID[3],MinCompTime[3],MaxID[3],MaxCompTime[3]);
	printf("\n");
	printf("*** In DFT ***\n\n");
	printf("   Set_OLP_Kin              = %5d %12.3f      %5d %12.3f\n",
	       MinID[5],MinCompTime[5],MaxID[5],MaxCompTime[5]);
	printf("   Set_Nonlocal             = %5d %12.3f      %5d %12.3f\n",
	       MinID[6],MinCompTime[6],MaxID[6],MaxCompTime[6]);
	printf("   Set_ProExpn_VNA          = %5d %12.3f      %5d %12.3f\n",
	       MinID[16],MinCompTime[16],MaxID[16],MaxCompTime[16]);
	printf("   Set_Hamiltonian          = %5d %12.3f      %5d %12.3f\n",
	       MinID[7],MinCompTime[7],MaxID[7],MaxCompTime[7]);
	printf("   Poisson                  = %5d %12.3f      %5d %12.3f\n",
	       MinID[8],MinCompTime[8],MaxID[8],MaxCompTime[8]);
	printf("   Diagonalization          = %5d %12.3f      %5d %12.3f\n",
	       MinID[9],MinCompTime[9],MaxID[9],MaxCompTime[9]);
	printf("   Mixing_DM                = %5d %12.3f      %5d %12.3f\n",
	       MinID[10],MinCompTime[10],MaxID[10],MaxCompTime[10]);
	printf("   Force                    = %5d %12.3f      %5d %12.3f\n",
	       MinID[11],MinCompTime[11],MaxID[11],MaxCompTime[11]);
	printf("   Total_Energy             = %5d %12.3f      %5d %12.3f\n",
	       MinID[12],MinCompTime[12],MaxID[12],MaxCompTime[12]);
	printf("   Set_Aden_Grid            = %5d %12.3f      %5d %12.3f\n",
	       MinID[13],MinCompTime[13],MaxID[13],MaxCompTime[13]);
	printf("   Set_Orbitals_Grid        = %5d %12.3f      %5d %12.3f\n",
	       MinID[14],MinCompTime[14],MaxID[14],MaxCompTime[14]);
	printf("   Set_Density_Grid         = %5d %12.3f      %5d %12.3f\n",
	       MinID[15],MinCompTime[15],MaxID[15],MaxCompTime[15]);
        printf("   RestartFileDFT           = %5d %12.3f      %5d %12.3f\n",
               MinID[17],MinCompTime[17],MaxID[17],MaxCompTime[17]);
        printf("   Mulliken_Charge          = %5d %12.3f      %5d %12.3f\n",
               MinID[18],MinCompTime[18],MaxID[18],MaxCompTime[18]);
        printf("   FFT(2D)_Density          = %5d %12.3f      %5d %12.3f\n",
               MinID[19],MinCompTime[19],MaxID[19],MaxCompTime[19]);
	printf("   Others                   = %5d %12.3f      %5d %12.3f\n",
	       ID_Mintime1,Mintime1,ID_Maxtime1,Maxtime1);
      }

    } 
    else{
      printf("could not save the CompTime file.\n");
    }
  }

  /* free arrays */

  free(time0);
  free(time1);

}


