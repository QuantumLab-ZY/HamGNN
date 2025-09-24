/**********************************************************************
  Stress_test.c:

     Stress_test.c is a subroutine to check consistency between 
     analytic and numerical stress tensor.

  Log of Stress_test.c:

     15/July/2015  Released by T.Ozaki

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
 


void Check_Stress(char *argv[]);

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



void Stress_test(int argc, char *argv[]) 
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
  char operate[1000];

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
  printf(" between anaylic and numerical stress tensor.\n");   fflush(stdout);
  printf("\n");                                                fflush(stdout);

  sprintf(operate,"ls stress_example/*.dat > ls_dat000000");
  system(operate);

  sprintf(operate,"wc ls_dat000000 > ls_dat000001");
  system(operate);

  sprintf(fname0,"ls_dat000000");
  sprintf(fname1,"ls_dat000001");
  sprintf(fname2,"stresstest.result");

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
    printf("Could not open stresstest.result\n");
    exit(1);
  }

  /*  run file by file */

  if ( ((fp0 = fopen(fname0,"r")) != NULL) &&
       ((fp1 = fopen(fname1,"r")) != NULL) )
    {

    fscanf(fp1,"%i",&Num_DatFiles);  

    printf(" %2d dat files are found in the directory 'stress_example'.\n\n\n",
            Num_DatFiles);fflush(stdout);

    /* start i loop */

    for (i=0; i<Num_DatFiles; i++){

      fscanf(fp0,"%s",fname_dat);

      /* run openmx */

      if (argc==3){
        sprintf(operate,"./openmx %s -stresstest2 %i",fname_dat,stress_flag);
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

        sprintf(operate,"%s %s %s %s %s %s %s -stresstest2 %i",
                str[0],str[1],str[2],str[3],fname_dat,str[4],str[5],stress_flag);

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
  printf("The result of stress consistecy test is found in a file 'stresstest.result'.\n\n\n");
}





void Check_Stress(char *argv[])
{
  int i,j,k,pq,Gc_AN;
  int time1,time2,MD_iter;
  double step,sum;
  double smat[4][4];
  double original_tv[4][4];
  double Analytic_Stress[9];
  double Numerical_Stress[9];
  double Utot1,Utot2,Utot3;
  FILE *fp1;
  char fname1[YOUSO10];
  static int numprocs,myid;

  /* MPI */ 

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* initialize */

  for (i=0; i<9; i++){
    Analytic_Stress[i]  = 0.0;
    Numerical_Stress[i] = 0.0;
  }

  /* step for calculation of the numerical derivatives */

  step = 0.0001;

  /* flags */

  F_Kin_flag    = mbit[stress_flag][0];
  F_NL_flag     = mbit[stress_flag][1];
  F_VNA_flag    = mbit[stress_flag][2];
  F_dVHart_flag = mbit[stress_flag][3];
  F_Vxc_flag    = mbit[stress_flag][4];
  F_VEF_flag    = mbit[stress_flag][5];
  F_U_flag      = mbit[stress_flag][6];

  /***************************************************************
    calculation of analytic stress tensor without any distortion 
  ****************************************************************/

  MD_iter = 1;
  time1 = truncation(MD_iter,1);
  time2 = DFT(MD_iter,0);

  for (i=0; i<9; i++){
    Analytic_Stress[i] = Stress_Tensor[i];
  }

  /* store original coordinates */

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      original_tv[i][j] = tv[i][j];
    }
  }

  /***************************************************************
    calculation of numerical stress tensor by finite difference
  ****************************************************************/

  /* loop for pq */

  for (pq=0; pq<9; pq++){

    /**********************
       In  case of -step
    **********************/

    /*  do not use the restart files for later calculations */
    Scf_RestartFromFile = 0;

    /* set scaled tv */ 

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
        smat[i][j] = 0.0;
      }
      smat[i][i] = 1.0;
    }    

    if      (pq==0) smat[1][1] -= step;
    else if (pq==1) smat[1][2] -= step;
    else if (pq==2) smat[1][3] -= step;
    else if (pq==3) smat[2][1] -= step;
    else if (pq==4) smat[2][2] -= step;
    else if (pq==5) smat[2][3] -= step;
    else if (pq==6) smat[3][1] -= step;
    else if (pq==7) smat[3][2] -= step;
    else if (pq==8) smat[3][3] -= step;

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
        sum = 0.0;
        for (k=1; k<=3; k++){
          sum += smat[i][k]*original_tv[j][k];
	}
        tv[j][i] = sum; 
      }
    }

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

      Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

      Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
    }

    /* perform the total energy calculation */

    MD_iter = 2;
    time1 = truncation(MD_iter,1);
    time2 = DFT(MD_iter,0);
    Utot2 = Utot;

    /**********************
       In  case of +step
    **********************/

    /*  do not use the restart files for later calculations */
    Scf_RestartFromFile = 0;

    /* set scaled tv */ 

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
        smat[i][j] = 0.0;
      }
      smat[i][i] = 1.0;
    }    

    if      (pq==0) smat[1][1] += step;
    else if (pq==1) smat[1][2] += step;
    else if (pq==2) smat[1][3] += step;
    else if (pq==3) smat[2][1] += step;
    else if (pq==4) smat[2][2] += step;
    else if (pq==5) smat[2][3] += step;
    else if (pq==6) smat[3][1] += step;
    else if (pq==7) smat[3][2] += step;
    else if (pq==8) smat[3][3] += step;

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
        sum = 0.0;
        for (k=1; k<=3; k++){
          sum += smat[i][k]*original_tv[j][k];
	}
        tv[j][i] = sum; 
      }
    }

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][1] + Grid_Origin[1];

      Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][2] + Grid_Origin[2];

      Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                      + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                      + Cell_Gxyz[Gc_AN][3]*tv[3][3] + Grid_Origin[3];
    }

    /* perform total energy calculation */

    MD_iter = 3;
    time1 = truncation(MD_iter,1);
    time2 = DFT(MD_iter,0);
    Utot3 = Utot;

    /**********************
       numerical stress
    **********************/

    Numerical_Stress[pq] = 0.5*(Utot3 - Utot2)/step;     
  }

  /***************************************************************
                       save stress to a file
  ****************************************************************/

  if (myid==Host_ID){

    sprintf(fname1,"stresstest.result");

    if ( (fp1 = fopen(fname1,"a")) != NULL ){

      fprintf(fp1,"\n");
      fprintf(fp1,"%s\n",argv[1]);
      fprintf(fp1,"  flag=%2d\n",stress_flag);
      fprintf(fp1,"  Numerical stress tensor = (Utot(s+dep)-Utot(s-dep))/(2*dep)\n");
      fprintf(fp1,"  dep=%15.10f\n",step);
      fprintf(fp1,"  Stress tensor\n");
      fprintf(fp1,"  Analytic stress\n");

      for (i=0; i<3; i++){
	for (j=0; j<3; j++){
	  fprintf(fp1," %15.12f ", Analytic_Stress[3*i+j]);
	}
	fprintf(fp1,"\n");
      }

      fprintf(fp1,"  Numerical stress\n");

      for (i=0; i<3; i++){
	for (j=0; j<3; j++){
	  fprintf(fp1," %15.12f ", Numerical_Stress[3*i+j]);
	}
	fprintf(fp1,"\n");
      }

      fprintf(fp1,"  Diff\n");

      for (i=0; i<3; i++){
	for (j=0; j<3; j++){
	  fprintf(fp1," %15.12f ", Analytic_Stress[3*i+j]-Numerical_Stress[3*i+j]);
	}
	fprintf(fp1,"\n");
      }

      fclose(fp1);
    }
    else{
      printf("Could not find %s in checking stress consistency\n",fname1);
    }

  }

}
