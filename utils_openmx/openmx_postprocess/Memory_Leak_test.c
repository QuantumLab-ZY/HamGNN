/**********************************************************************
  Memory_Leak_test.c:

     Memory_Leak_test.c is a subroutine to check memory leak
     by monitoring actual used memory

  Log of Memory_Leak_test.c:

     20/Oct/2005  Released by T. Ozaki

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
 


void Get_VSZ(int MD_iter);


void Memory_Leak_test(int argc, char *argv[]) 
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

  printf("\n*******************************************************\n"); fflush(stdout);
  printf("*******************************************************\n"); fflush(stdout);
  printf(" Welcome to OpenMX                                     \n"); fflush(stdout);
  printf(" Copyright (C), 2002-2019, T.Ozaki                     \n"); fflush(stdout);
  printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n"); fflush(stdout);
  printf(" This is free software, and you are welcome to         \n"); fflush(stdout);
  printf(" redistribute it under the constitution of the GNU-GPL.\n"); fflush(stdout);
  printf("*******************************************************\n"); fflush(stdout);
  printf("*******************************************************\n\n\n"); fflush(stdout);

  printf("\n");fflush(stdout);
  printf(" OpenMX is now in the mode to check memory leak\n"); fflush(stdout);
  printf(" by monitoring actual used memory size 'VSZ' and 'RSS'.\n"); fflush(stdout);
  printf("\n");fflush(stdout);

  sprintf(operate,"ls ml_example/*.dat > ls_dat000000");
  system(operate);

  sprintf(operate,"wc ls_dat000000 > ls_dat000001");
  system(operate);

  sprintf(fname0,"ls_dat000000");
  sprintf(fname1,"ls_dat000001");
  sprintf(fname2,"mltest.result");

  fp = fopen(fname2, "r");   
  if (fp!=NULL){
    fclose(fp); 
    sprintf(operate,"rm %s",fname2);
    system(operate);
  }

  if ( ((fp0 = fopen(fname0,"r")) != NULL) &&
       ((fp1 = fopen(fname1,"r")) != NULL) &&
       ((fp2 = fopen(fname2,"a")) != NULL) )
    {

    /* here close fp2, but later open fp2 again due to appending */
    fclose(fp2);  

    fscanf(fp1,"%i",&Num_DatFiles);  

    printf(" %2d dat files are found in the directory 'ml_example'.\n\n\n",Num_DatFiles);

    /* start i loop */

    for (i=0; i<Num_DatFiles; i++){

      fscanf(fp0,"%s",fname_dat);
 
      /* run openmx */

      if      (argc==2){
        sprintf(operate,"./openmx %s -mltest2",fname_dat);
      }
      else if (argc==3){

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

        strcpy(tmp_argv,argv[2]); 

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

        sprintf(operate,"%s %s %s %s %s %s %s -mltest2",
                str[0],str[1],str[2],str[3],fname_dat,str[4],str[5]);
      }

      system(operate);

      /* write the result to a file, mltest.result */

      if ( (fp2 = fopen(fname2,"a")) != NULL ){
        fprintf(fp2,"\n\n %4d     %-32.30s\n\n", i+1,fname_dat);
        fclose(fp2);
      }
                  
      sprintf(operate,"cat ls_dat000004 >> %s",fname2);
      system(operate);

      sprintf(operate,"rm ls_dat000004");
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
  printf("Monitored VSZ and RSS can be found in a file 'mltest.result'.\n\n\n");
}



void Get_VSZ(int MD_iter)
{
  int column_VSZ,po_VSZ;
  int column_RSS,po_RSS;
  int column_CPU,po_CPU;
  int line,po_line,num;
  int i,max;
  FILE *fp,*fp1,*fp2;
  char fname0[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  char tmpchar[300];
  char tmpline[300];
  char operate[800];
  double Used_CPU,Used_VSZ,Used_RSS;

  sprintf(fname0,"ls_dat000002");
  sprintf(fname1,"ls_dat000003");
  sprintf(fname2,"ls_dat000004");

  sprintf(operate,"ps uxw > %s",fname0);
  system(operate);
  
  column_VSZ = 0;
  column_RSS = 0;
  column_CPU = 0;
  po_VSZ = 0;
  po_RSS = 0;
  po_CPU = 0;
  num = 0;

  /* find the column number of VSZ, RSS, and CPU */

  if ( (fp = fopen(fname0,"r")) != NULL ){

    do {
      if (fscanf(fp,"%s",tmpchar)!=EOF){

        if (po_VSZ==0)  column_VSZ++;
        if (po_RSS==0)  column_RSS++;
        if (po_CPU==0)  column_CPU++;

        num++;

	if (strcmp(tmpchar,"VSZ")==0){
	  po_VSZ = 1;
	}

	if (strcmp(tmpchar,"RSS")==0){
	  po_RSS = 1;
	}

	if (strcmp(tmpchar,"%CPU")==0){
	  po_CPU = 1;
	}
      }

    } while ( (po_VSZ==0 || po_RSS==0 || po_CPU==0) && num<12 );

    /*
    if (po_CPU==1) {
      printf("ABC1 column_CPU=%i\n",column_CPU);
    }

    if (po_VSZ==1) {
      printf("ABC2 column_VSZ=%i\n",column_VSZ);
    }

    if (po_RSS==1) {
      printf("ABC3 column_RSS=%i\n",column_RSS);
    }
    */

    fclose(fp);
  }
  else{
    printf("Could not find %s in checking memory leak (1)\n",fname0);
  }  

  /* find the line which corresponds the OpenMX job */

  po_line = 0;
  line = 0;

  if ( (fp = fopen(fname0,"r")) != NULL ){

    do {

      if (fgets(tmpline, 300, fp)!=NULL){

        line++; 

        if ( strstr(tmpline,"openmx ml_example/")!=NULL ){
          po_line = 1;          

          if ( (fp1 = fopen(fname1,"w")) != NULL ){
            fprintf(fp1,"%s",tmpline);
            fclose(fp1);
	  }          
          else{
            printf("Could not find %s in checking memory leak (2)\n",fname1);
          }
        }
      }
      else{
        po_line = 2;          
      }

    } while ( po_line==0 );


    /*
    if (po_line==1){
      printf("line=%3d\n",line);
    }
    */

    fclose(fp);
  }
  else{
    printf("Could not find %s in checking memory leak(3)\n",fname0);
  }  

  /* get VSZ */

  if ( (fp1 = fopen(fname1,"r")) != NULL ){

    if (column_CPU<column_VSZ)
      max = column_VSZ;
    else
      max = column_CPU;

    if (max<column_RSS) max = column_RSS;

    for (i=1; i<=max; i++){
      if (fscanf(fp1,"%s",tmpchar)!=EOF){
        if (i==column_CPU) Used_CPU = atof(tmpchar);
        if (i==column_VSZ) Used_VSZ = atof(tmpchar);
        if (i==column_RSS) Used_RSS = atof(tmpchar);
      }
    }

    fclose(fp1);
  }
  else{
    printf("Could not find %s in checking memory leak (4)\n",fname1);
  }  
  
  /*
  printf("Used_CPU (percent) = %6.3f\n",Used_CPU);fflush(stdout);
  printf("Used_VSZ (kbyte)   = %6d\n", (long int)(Used_VSZ));fflush(stdout);
  printf("Used_RSS (kbyte)   = %6d\n", (long int)(Used_RSS));fflush(stdout);
  */

  /* write Used_CPU and Used_VSZ in a file */

  if (MD_iter==1){

    if ( (fp2 = fopen(fname2,"w")) != NULL ){
      fprintf(fp2,"                   CPU (%%)     VSZ (kbyte)    RSS (kbyte)\n");
      fprintf(fp2," MD_iter=%4d     %7.3f     %6ld         %6ld\n",
              MD_iter,Used_CPU,(long int)(Used_VSZ),(long int)(Used_RSS));
      fclose(fp2);
    }          
    else{
      printf("Could not find %s in checking memory leak (5)\n",fname2);
    }
  }
  else {
    if ( (fp2 = fopen(fname2,"a")) != NULL ){
      fprintf(fp2," MD_iter=%4d     %7.3f     %6ld         %6ld\n",
              MD_iter,Used_CPU,(long int)(Used_VSZ),(long int)(Used_RSS));
      fclose(fp2);
    }          
    else{
      printf("Could not find %s in checking memory leak (6)\n",fname2);
    }
  }

  /* delete files */

  sprintf(operate,"rm %s",fname0);
  system(operate);

  sprintf(operate,"rm %s",fname1);
  system(operate);

}

