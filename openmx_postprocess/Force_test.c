/**********************************************************************
  Force_test.c:

     Force_test.c is a subroutine to check consistency between 
     analytic and numerical forces

  Log of Force_test.c:

     20/Oct/2005  Released by T.Ozaki

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
/*  end stat section */
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
 


void Check_Force(char *argv[]);

#define nflags1  9
#define nflags2  7


static int
  mbit[nflags1][nflags2]={ 
  {1,1,1,1,1,1,1},
  {0,0,0,0,0,0,0},
  {1,0,0,0,0,0,0},
  {1,1,0,0,0,0,0},
  {1,0,1,0,0,0,0},
  {1,0,0,1,0,0,0},
  {1,0,0,0,1,0,0},
  {1,0,0,0,0,1,0},
  {1,0,0,0,0,0,1}
};



void Force_test(int argc, char *argv[]) 
{
  FILE *fp,*fp0,*fp1,*fp2,*fp3;
  int Num_DatFiles,i,j,k;
  int Num_Atoms;
  int NGrid1_1,NGrid1_2,NGrid1_3;
  int NGrid2_1,NGrid2_2,NGrid2_3;
  double Utot1,Utot2,dU,dF;
  double gx,gy,gz,fx,fy,fz;
  double sum1,sum2;
  char fname[YOUSO10];
  char fname0[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  char fname_dat[YOUSO10];
  char fname_dat2[YOUSO10];
  char fname_out1[YOUSO10];
  char fname_out2[YOUSO10];
  char ftmp[YOUSO10];
  char operate[800];

  printf("\n*******************************************************\n");   fflush(stdout);
  printf("*******************************************************\n");     fflush(stdout);
  printf(" Welcome to OpenMX                                     \n");     fflush(stdout);
  printf(" Copyright (C), 2002-2019, T.Ozaki                     \n");     fflush(stdout);
  printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n");     fflush(stdout);
  printf(" This is free software, and you are welcome to         \n");     fflush(stdout);
  printf(" redistribute it under the constitution of the GNU-GPL.\n");     fflush(stdout);
  printf("*******************************************************\n");     fflush(stdout);
  printf("*******************************************************\n\n\n"); fflush(stdout);

  printf("\n");                                                fflush(stdout);
  printf(" OpenMX is now in the mode to check consistency\n"); fflush(stdout);
  printf(" between anaylic and numerical forces.\n");          fflush(stdout);
  printf("\n");                                                fflush(stdout);

  sprintf(operate,"ls force_example/*.dat > ls_dat000000");
  system(operate);

  sprintf(operate,"wc ls_dat000000 > ls_dat000001");
  system(operate);

  sprintf(fname0,"ls_dat000000");
  sprintf(fname1,"ls_dat000001");
  sprintf(fname2,"forcetest.result");

  fp = fopen(fname2, "r");   
  if (fp!=NULL){
    fclose(fp); 
    sprintf(operate,"rm %s",fname2);
    system(operate);
  }

  if ( (fp2 = fopen(fname2,"w")) != NULL ){

    fprintf(fp2,"\n flag          ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",i);
    }
    fprintf(fp2,"\n\n");

    fprintf(fp2," Kinetic       ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][0]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2," Non-local     ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][1]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2," Neutral atom  ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][2]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2," diff Hartree  ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][3]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2," Ex-Corr       ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][4]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2," E. Field      ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][5]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2," Hubbard U     ");
    for (i=0; i<nflags1; i++){
      fprintf(fp2,"%2d  ",mbit[i][6]);
    }
    fprintf(fp2,"\n");

    fprintf(fp2,"\n\n");

    fclose(fp2);
  }
  else{
    printf("Could not open forcetest.result\n");
    exit(1);
  }

  /*  run file by file */

  if ( ((fp0 = fopen(fname0,"r")) != NULL) &&
       ((fp1 = fopen(fname1,"r")) != NULL) )
    {

    fscanf(fp1,"%i",&Num_DatFiles);  

    printf(" %2d dat files are found in the directory 'force_example'.\n\n\n",
            Num_DatFiles);fflush(stdout);

    /* start i loop */

    for (i=0; i<Num_DatFiles; i++){

      fscanf(fp0,"%s",fname_dat);
 
      /* run openmx */

      if (argc==3){
        sprintf(operate,"./openmx %s -forcetest2 %i ",fname_dat,force_flag);
      }
      else if (argc==4){

        int p;
        char *tp;
	char str[10][300];
        char tmp_argv[300];

        sprintf(str[0],"");
        sprintf(str[1],"");
        sprintf(str[2],"");
        sprintf(str[3],"");
        sprintf(str[4],"");
        sprintf(str[5],"");

        strcpy(tmp_argv,argv[3]); 

        tp = strtok(tmp_argv," ");
	sprintf(str[0],"%s",tp); 

        p = 1; 
        while (tp != NULL ){
          tp = strtok(NULL," ");   
          if (tp!=NULL){ 
	    sprintf(str[p],"%s",tp);     
            p++;
          }
	}

        sprintf(operate,"%s %s %s %s %s %s %s -forcetest2 %i",
                str[0],str[1],str[2],str[3],fname_dat,str[4],str[5],force_flag);

      }

      system(operate);
    }

    fclose(fp0);
    fclose(fp1);
  }
  else{
    printf("Could not find ls_dat000000 or ls_dat000001\n");
    exit(1);
  }

  sprintf(operate,"rm ls_dat000000");
  system(operate);

  sprintf(operate,"rm ls_dat000001");
  system(operate);

  printf("\n\n\n\n");
  printf("The result of force consistecy test is found in a file 'forcetest.result'.\n\n\n");
}





void Check_Force(char *argv[])
{
  int ixyz;
  int time1,time2,MD_iter;
  double step;
  double original_x,original_y,original_z;
  double Analytic_Force[4];
  double Numerical_Force[4];
  double Utot1,Utot2,Utot3;
  FILE *fp1;
  char fname1[YOUSO10];
  static int numprocs,myid;

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* initialize */

  for (ixyz=0; ixyz<=3; ixyz++){
    Analytic_Force[ixyz]  = 0.0;
    Numerical_Force[ixyz] = 0.0;
  }

  /* step for calculation of the numerical derivatives */

  step = 0.0003;

  /* flags */

  F_Kin_flag    = mbit[force_flag][0];
  F_NL_flag     = mbit[force_flag][1];
  F_VNA_flag    = mbit[force_flag][2];
  F_dVHart_flag = mbit[force_flag][3];
  F_Vxc_flag    = mbit[force_flag][4];
  F_VEF_flag    = mbit[force_flag][5];
  F_U_flag      = mbit[force_flag][6];

  /* store original coordinates */

  original_x = Gxyz[1][1];
  original_y = Gxyz[1][2];
  original_z = Gxyz[1][3];

  /* loop for ixyz */

  for (ixyz=1; ixyz<=3; ixyz++){

    Gxyz[1][1] = original_x;
    Gxyz[1][2] = original_y;
    Gxyz[1][3] = original_z;

    MD_iter = 1;
    time1 = truncation(MD_iter,1);
    time2 = DFT(MD_iter,(MD_iter-1)%orbitalOpt_per_MDIter+1);
    Utot1 = Utot;

    /* analytic force */

    if (myid==G2ID[1]){
      Analytic_Force[ixyz] = -Gxyz[1][16+ixyz];
    }

    MPI_Bcast(&Analytic_Force[ixyz], 1, MPI_DOUBLE, G2ID[1], mpi_comm_level1);

    /*  do not use the restart files for later calculations */
    Scf_RestartFromFile = 0;

    MD_iter = 2;
    Gxyz[1][ixyz] -= step; 
    time1 = truncation(MD_iter,1);
    time2 = DFT(MD_iter,(MD_iter-1)%orbitalOpt_per_MDIter+1);
    Utot2 = Utot;

    /*  do not use the restart files for later calculations */
    Scf_RestartFromFile = 0;

    MD_iter = 3;
    Gxyz[1][ixyz] += 2.0*step; 
    time1 = truncation(MD_iter,1);
    time2 = DFT(MD_iter,(MD_iter-1)%orbitalOpt_per_MDIter+1);
    Utot3 = Utot;

    /* numerical force */

    if (myid==G2ID[1]){
      Numerical_Force[ixyz] = -0.5*(Utot3 - Utot2)/step;     
    }

    MPI_Bcast(&Numerical_Force[ixyz], 1, MPI_DOUBLE, G2ID[1], mpi_comm_level1);
  }

  /* save forces to a file */

  if (myid==Host_ID){

    sprintf(fname1,"forcetest.result");

    if ( (fp1 = fopen(fname1,"a")) != NULL ){

      fprintf(fp1,"\n");
      fprintf(fp1,"%s\n",argv[1]);
      fprintf(fp1,"  flag=%2d\n",force_flag);
      fprintf(fp1,"  Numerical force= -(Utot(s+ds)-Utot(s-ds))/(2*ds)\n");
      fprintf(fp1,"  ds=%15.10f\n",step);
      fprintf(fp1,"  Forces (Hartree/Bohr) on atom 1\n");
      fprintf(fp1,"                               x              y               z\n");
      fprintf(fp1,"  Analytic force       %15.12f %15.12f %15.12f\n",
	      Analytic_Force[1],Analytic_Force[2],Analytic_Force[3]);
      fprintf(fp1,"  Numerical force      %15.12f %15.12f %15.12f\n",
	      Numerical_Force[1],Numerical_Force[2],Numerical_Force[3]);
      fprintf(fp1,"  diff                 %15.12f %15.12f %15.12f\n",
	      Analytic_Force[1]-Numerical_Force[1],
	      Analytic_Force[2]-Numerical_Force[2],
	      Analytic_Force[3]-Numerical_Force[3]);

      fclose(fp1);
    }
    else{
      printf("Could not find %s in checking force consistency\n",fname1);
    }

  }

}
