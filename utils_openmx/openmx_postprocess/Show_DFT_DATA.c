/**********************************************************************
  Runtest.c:

     Runtest.c is a subroutine to check whether OpenMX runs normally 
     on many platforms or not by comparing the stored *.out and generated
     *.out on your machine.

  Log of Runtest.c:

     25/Oct/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
/*  end stat section */
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
 

static void SpeciesString2int(int p);

typedef struct {
  char fn[YOUSO10];
} fname_type;

 

void Show_DFT_DATA(char *argv[]) 
{
  FILE *fp,*fp0,*fp1,*fp2,*fp3;
  int Num_DatFiles,i,j,k,fp_OK;
  int Num_Atoms;
  int NGrid1_1,NGrid1_2,NGrid1_3;
  int NGrid2_1,NGrid2_2,NGrid2_3;
  double Utot1,Utot2,dU,dF;
  double gx,gy,gz,fx,fy,fz;
  double sum1,sum2;
  double time1,TotalTime;
  char SN[30][YOUSO10];
  char SB[30][YOUSO10];
  char SV[30][YOUSO10];
  char fname[YOUSO10];
  char fname0[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  char fname_dat[YOUSO10];
  char fname_dat2[YOUSO10];
  char fname_out1[YOUSO10];
  char fname_out2[YOUSO10];
  char ftmp[YOUSO10];
  fname_type *fndat;
  int numprocs,myid;

  char *dir;
  char input_dir[300];
  char *output_file;
  DIR *dp;
  struct dirent *entry;

  MPI_Request request;
  MPI_Status  status;

  /* set up MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  sprintf(input_dir,"%s",argv[2]);

  printf("%s\n",input_dir);
  
  /* print the header */

  if (myid==Host_ID){

    /* set dir */

    dir = input_dir;

    /* count the number of dat files */

    if(( dp = opendir(dir) ) == NULL ){
      printf("could not find the directory '%s'\n",input_dir);
      MPI_Finalize();
      exit(0);
    }

    Num_DatFiles = 0;
    while((entry = readdir(dp)) != NULL){

      if ( strstr(entry->d_name,".dat")!=NULL ){ 
          
        Num_DatFiles++;
      }
    }
    closedir(dp);

    fndat = (fname_type*)malloc(sizeof(fname_type)*Num_DatFiles);

    /* store the name of dat files */

    if(( dp = opendir(dir) ) == NULL ){
      printf("could not find the directory '%s'\n",input_dir);
      MPI_Finalize();
      exit(0);
    }

    Num_DatFiles = 0;
    while((entry = readdir(dp)) != NULL){
 
      if ( strstr(entry->d_name,".dat")!=NULL ){ 

        sprintf(fndat[Num_DatFiles].fn,"%s/%s",input_dir,entry->d_name);  
        Num_DatFiles++;
      }
    }
    closedir(dp);

  } /* if (myid==Host_ID) */

  if (myid==Host_ID){
    printf(" %2d dat files are found in the directory '%s'.\n\n\n",Num_DatFiles,input_dir);
  }

  /***********************************************************
         start calculations
  ***********************************************************/

  for (i=0; i<Num_DatFiles; i++){

    sprintf(fname_dat,"%s",fndat[i].fn);

    input_open(fname_dat);
    input_int("Species.Number",&SpeciesNum,0);

    /* read Definition.of.Atomic.Species */

    if (fp=input_find("<Definition.of.Atomic.Species")) {

      for (k=0; k<SpeciesNum; k++){
        fscanf(fp,"%s %s %s",SN[k],SB[k],SV[k]);

        printf("i=%2d k=%2d %s %s\n",i,k,SB[k],SV[k]);

	/*
        SpeciesString2int(i);
	*/
      }

      if (! input_last("Definition.of.Atomic.Species>")) {
	MPI_Finalize();
	exit(0);
      }
    }

    input_close();
  }

  if (myid==Host_ID){
    free(fndat);
  }

  MPI_Barrier(mpi_comm_level1);
  MPI_Finalize();
  exit(0);
}



void SpeciesString2int(int p)
{
  int i,l,n,po;
  char c,cstr[YOUSO10*3];
  
  /* Get basis name */

  sprintf(SpeBasisName[p],"");

  i = 0;
  po = 0;
  while( ((c=SpeBasis[p][i])!='\0' || po==0) && i<YOUSO10 ){
    if (c=='-'){
      po = 1;
      SpeBasisName[p][i] = '\0';  
    }
    if (po==0) SpeBasisName[p][i] = SpeBasis[p][i]; 
    i++;
  }


  if (2<=level_stdout){
    printf("<Input_std>  SpeBasisName=%s\n",SpeBasisName[p]);
  }

  /* Get basis type */

  for (l=0; l<5; l++){
    Spe_Num_Basis[p][l] = 0;
  }

  i = 0;
  po = 0;

  while((c=SpeBasis[p][i])!='\0'){
    if (po==1){
      if      (c=='s'){ l=0; n=0; }
      else if (c=='p'){ l=1; n=0; }
      else if (c=='d'){ l=2; n=0; }
      else if (c=='f'){ l=3; n=0; }
      else if (c=='g'){ l=4; n=0; }
      else{

        if (n==0){
          cstr[0] = c;
          cstr[1] = '\0';
          Spe_Num_Basis[p][l]  = atoi(cstr);
          Spe_Num_CBasis[p][l] = atoi(cstr);
          n++;
        }
        else if (n==1){
          cstr[0] = c;
          cstr[1] = '\0';
          Spe_Num_CBasis[p][l] = atoi(cstr);   
          if (Spe_Num_Basis[p][l]<Spe_Num_CBasis[p][l]){
            printf("# of contracted orbitals are larger than # of primitive oribitals\n");
            MPI_Finalize();
            exit(1); 
          } 

          n++;
        }
        else {
          printf("Format error in Definition of Atomic Species\n");
          MPI_Finalize();
          exit(1);
	}
      } 
    }  

    if (SpeBasis[p][i]=='-') po = 1;
    i++;
  }

  for (l=0; l<5; l++){
    if (Spe_Num_Basis[p][l]!=0) Spe_MaxL_Basis[p] = l;

    if (2<=level_stdout){
      printf("<Input_std>  p=%2d l=%2d %2d %2d\n",
              p,l,Spe_Num_Basis[p][l],Spe_Num_CBasis[p][l]);
    }
  }
}






