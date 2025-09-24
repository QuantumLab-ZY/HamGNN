/**********************************************************************
  neb.c:

     neb is a program to perform the nudged elastic band (NEB) method 
     for finding a minimum energy path (MEP) connecting two structures, 
     which is based on JCP 113, 9978 (2000).

  Log of neb.c:

     Apr./06/2011  Released by T. Ozaki
     Feb./01/2013  Modified by Y. Kubota (supervised by Prof. F. Ishii)
     Feb./17/2015  Modified by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "lapack_prototypes.h"

#define MAXBUF 1024
#define Criterion_Max_Step       0.20

#ifdef MAX 
#undef MAX
#endif
#define MAX(a,b) (((a)>(b))?  (a):(b)) 

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) (((a)<(b))?  (a):(b))


static void allocate_arrays();
static void free_arrays();
static void read_input(char *file);
static void generate_input_files(char *file, int iter);
static void Calc_NEB_Gradients(double ***neb_atom_coordinates);
static void Update_Coordinates(int iter, double ***neb_atom_coordinates);
static void Make_XYZ_File(char fname[YOUSO10], double ***neb_atom_coordinates);
static void Make_MD_File(char fname[YOUSO10], double ***neb_atom_coordinates);
static void Generate_Restart_File(char fname[YOUSO10], double ***neb_atom_coordinates);
static void Steepest_Decent(int iter, double ***neb_atom_coordinates, double norm, double Max_Force);
static double DIIS_BFGS(int iter, double ***neb_atom_coordinates, double norm, double Max_Force);
static void Estimate_Initial_Hessian();
static void Inverse(int n, double **a, double **ia);

double **All_Grid_Origin;
double **Tmp_Grid_Origin;
double ***neb_atom_coordinates;
double ***tmp_neb_atom_coordinates;
double ****His_neb_atom_coordinates;
double ******InvHess;
double ***Vec0;
int **atomFixedXYZ;


static char fileXYZ[YOUSO10] = ".neb.xyz"; 
static char fileMD[YOUSO10] = ".neb.md"; 
char system_name[YOUSO10];
int *WhatSpecies_NEB;
int *Spe_WhatAtom_NEB;
char **SpeName_NEB;

int PN;





void neb(int argc, char *argv[])
{ 
  int iter,index_images;
  int po,i,j,k,p,h;
  int myid,numprocs;
  int myid1,numprocs1;  
  int myid2,numprocs2;
  int myid3,numprocs3;
  int parallel1_flag;  
  int myworld1,ID0,ID1;
  int Num_Comm_World1;
  int *NPROCS_ID1;
  int *Comm_World1;
  int *NPROCS_WD1;
  int *Comm_World_StartID1;
  int parallel2_flag; 
  int myworld2,myworld3;
  int Num_Comm_World2,Num_Comm_World3;
  int *NPROCS_ID2;
  int *NPROCS_ID3;
  int *Comm_World2;
  int *Comm_World3;
  int *NPROCS_WD2;
  int *NPROCS_WD3;
  int *Comm_World_StartID2;
  int *Comm_WOrld_StartID3;
  char fname_original[YOUSO10];
  char fname1[YOUSO10];
  MPI_Comm *MPI_CommWD1;
  MPI_Comm *MPI_CommWD2;
  MPI_Comm *MPI_CommWD3;
  char file_neb_utot[YOUSO10];
  double TStime,TEtime,a0time,a1time,f0time;
  double b0time,b1time,c0time,c1time,flatstime,flatetime,sumtime=0.0;
  FILE *fp;

  dtime(&TStime);

  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);
 
  /* check argv */

  if (argc==1){
    printf("\nCould not find an input file.\n\n");
    MPI_Finalize(); 
    exit(0);
  } 

  /****************************************************
             show a message by the NEB code
  ****************************************************/
  
  if (myid==Host_ID){  
    printf("\n*******************************************************\n"); 
    printf("*******************************************************\n"); 
    printf(" Welcome to the NEB extension of OpenMX                \n");
    printf(" Copyright (C), 2002-2019, T. Ozaki                    \n"); 
    printf(" OpenMX comes with ABSOLUTELY NO WARRANTY.             \n"); 
    printf(" This is free software, and you are welcome to         \n"); 
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("*******************************************************\n"); 
    printf("*******************************************************\n\n"); 
  } 

  /****************************************************
                  read the input file 
  ****************************************************/

  read_input(argv[1]);

  fnjoint(filepath,filename,fileXYZ);
  fnjoint(filepath,filename,fileMD);
  sprintf(fname_original,"%s",argv[1]);

  /****************************************************
      compare PN with numprocs and NEB_Num_Images 
  ****************************************************/

  if ((PN > numprocs)||(PN > NEB_Num_Images)){

    PN=0;
    if (myid==Host_ID){
      printf("The keyword MD.NEB.Parallel.Number will be ignored if the following conditions are satisfied:\n");
      printf("MD.NEB.Parallel.Number is equal to or smaller than the number of MPI processes.\n");
      printf("MD.NEB.Parallel.Number is equal to or smaller than the number of MD.NEB.Number.Images.\n");
    }
  } 

  /*******************************************************************************
                   If MD.NEB.Parallel.Number is not specifed,
  *******************************************************************************/

  if (PN==0){

    /****************************************************
       Two level parallelization is performed. 
       Outer loop: images
       Inner loop: calculation in each image
    ****************************************************/
    
    if ( NEB_Num_Images<=numprocs || numprocs==1 ){

      /****************************************************
         allocate processes to each image in the World1
      ****************************************************/

      Num_Comm_World1 = NEB_Num_Images;
      
      if ( Num_Comm_World1<=numprocs ){
      
	parallel1_flag = 1; 
	
	NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs); 
	Comm_World1 = (int*)malloc(sizeof(int)*numprocs); 
	NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
	Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
	MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);
	
	Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World1, &myworld1, MPI_CommWD1, 
			 NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);
	
	MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
	MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);
	
      }
      else {
	parallel1_flag = 0; 
	myworld1 = 0;
      }
      
      /****************************************************
         allocate processes to each image in the World2
      ****************************************************/
    
      Num_Comm_World2 = 2;
    
      if ( Num_Comm_World2<=numprocs ){
      
	parallel2_flag = 1; 
      
	NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs); 
	Comm_World2 = (int*)malloc(sizeof(int)*numprocs); 
	NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
	Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
	MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);
	
	Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World2, &myworld2, MPI_CommWD2, 
			 NPROCS_ID2, Comm_World2, NPROCS_WD2, Comm_World_StartID2);
	
	MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
	MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);
      }
      else {
	parallel2_flag = 0; 
	myworld2 = 0;
      }
      
      /****************************************************
        SCF calculations for the two terminal structures
      ****************************************************/

      /* check whether the restart is performed or not */

      sprintf(file_neb_utot,"%s%s_0_rst/%s.neb.utot",filepath,system_name,system_name);

      if (Scf_RestartFromFile==1 && (fp = fopen(file_neb_utot,"rb"))!=NULL){

        if (myid==Host_ID){
          fread(&neb_atom_coordinates[0][0][0],sizeof(double),1,fp);
          fread(&neb_atom_coordinates[NEB_Num_Images+1][0][0],sizeof(double),1,fp);
	}

        MPI_Bcast(&neb_atom_coordinates[0][0][0], 1, MPI_DOUBLE, Host_ID, MPI_COMM_WORLD1);
        MPI_Bcast(&neb_atom_coordinates[NEB_Num_Images+1][0][0], 1, MPI_DOUBLE, Host_ID, MPI_COMM_WORLD1);

        fclose(fp);
      }     

      /* if the restart is not performed, perform the SCF calulations for the two terminals */ 

      else {
    
	/* generate input files */

	generate_input_files(fname_original,1);
    
	/* In case of parallel2_flag==1 */
     
	if (parallel2_flag==1){
      
	  if (myworld2==0)  
	    index_images = 0;
	  else 
	    index_images = NEB_Num_Images + 1;
	
	  sprintf(fname1,"%s_%i",fname_original,index_images);
	  argv[1] = fname1;
	
	  neb_run( argv,MPI_CommWD2[myworld2],index_images,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	
	  /* find a representative ID in the top world for ID=0 in the world2 */
      
	  i = 0; po = 0; 
	  do {
	
	    if (Comm_World2[i]==0){
	      ID0 = i; 
	      po = 1;
	    }
	    i++;
	  } while (po==0);
	
	  /* find a representative ID in the top world for ID=1 in the world2 */
      
	  i = 0; po = 0; 
	  do {
	  
	    if (Comm_World2[i]==1){
	      ID1 = i; 
	      po = 1;
	    }
	    i++;
	  } while (po==0);
	
	  /* MPI broadcast of neb_atom_coordinates[0] and neb_atom_coordinates[NEB_Num_Images+1] */
      
	  MPI_Barrier(MPI_COMM_WORLD1);
	  for (i=0; i<=atomnum; i++){
	    MPI_Bcast(&neb_atom_coordinates[0][i][0], 20, MPI_DOUBLE, ID0, MPI_COMM_WORLD1);
	    MPI_Bcast(&neb_atom_coordinates[NEB_Num_Images+1][i][0], 20, MPI_DOUBLE, ID1, MPI_COMM_WORLD1);
	  }
	} /* if (parallel2_flag==1) */
    
	/* In case of parallel2_flag==0 */
    
	else {
      
	  /* SCF calculation of a terminal with the index of 0 */
	  index_images = 0;
	  sprintf(fname1,"%s_%i",fname_original,index_images);
	  argv[1] = fname1;
	  neb_run( argv,MPI_COMM_WORLD1,index_images,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
      
	  /* SCF calculation of a terminal with the index of (NEB_Num_Images + 1) */
	  index_images = NEB_Num_Images + 1;
	  sprintf(fname1,"%s_%i",fname_original,index_images);
	  argv[1] = fname1;
	  neb_run( argv,MPI_COMM_WORLD1,index_images,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );

	}

        // save *_0_rst/*.neb.utot in binary mode

        if (myid==Host_ID){

          sprintf(file_neb_utot,"%s%s_0_rst/%s.neb.utot",filepath,system_name,system_name);

	  if ((fp = fopen(file_neb_utot,"wb")) != NULL){

	    fwrite(&neb_atom_coordinates[0][0][0],sizeof(double),1,fp);
	    fwrite(&neb_atom_coordinates[NEB_Num_Images+1][0][0],sizeof(double),1,fp);
	    fclose(fp);
	  }
	  else{
	    printf("Could not open a file %s in neb\n",file_neb_utot);
	  }
	}

      } /* else */

      /****************************************************
        optimiziation for finding a minimum energy path
        connecting the two terminal structures.
      ****************************************************/
    
      iter = 1;  
      MD_Opt_OK = 0;
      dtime(&f0time);
      do {

	/* if iter==1, generate an input file for restarting  */
	
	if (iter==1) Generate_Restart_File(fname_original,neb_atom_coordinates); 
      
	/* generate input files */

	generate_input_files(fname_original,iter);
      
	/* In case of parallel1_mode==1 */
      
	if (parallel1_flag==1){
	
	  sprintf(fname1,"%s_%i",fname_original,myworld1+1);
	  argv[1] = fname1;
	  dtime(&flatstime);
	  neb_run( argv,MPI_CommWD1[myworld1],myworld1+1,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	  dtime(&flatetime);
	  sumtime += (flatetime-flatstime);	
	  /* MPI: All_Grid_Origin */
	
	  for (i=0; i<=(NEB_Num_Images+1); i++){
	    for (j=0; j<=3; j++) Tmp_Grid_Origin[i][j] = 0.0;
	  }  
	  
	  if (myid1==Host_ID){
	    Tmp_Grid_Origin[myworld1+1][1] = Grid_Origin[1];
	    Tmp_Grid_Origin[myworld1+1][2] = Grid_Origin[2];
	    Tmp_Grid_Origin[myworld1+1][3] = Grid_Origin[3];
	  }
	
	  MPI_Barrier(MPI_COMM_WORLD1);
	
	  for (p=1; p<=NEB_Num_Images; p++){ 
	    MPI_Allreduce( &Tmp_Grid_Origin[p][0], &All_Grid_Origin[p][0], 
			   4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD1 );
	  }
	
	  /* MPI: neb_atom_coordinates */
	
	  for (p=1; p<=NEB_Num_Images; p++){
	    for (i=0; i<=atomnum; i++){
	      for (j=0; j<20; j++){ 
		tmp_neb_atom_coordinates[p][i][j] = 0.0;
	      }
	    }
	  }
	
	  if (myid1==Host_ID){
	    for (i=0; i<=atomnum; i++){
	      for (j=0; j<20; j++){ 
		tmp_neb_atom_coordinates[myworld1+1][i][j] = neb_atom_coordinates[myworld1+1][i][j];;
	      }
	    }
	  }
	  
	  MPI_Barrier(MPI_COMM_WORLD1);
	  
	  for (p=1; p<=NEB_Num_Images; p++){ 
	    for (i=0; i<=atomnum; i++){
	      
	      MPI_Allreduce( &tmp_neb_atom_coordinates[p][i][0], &neb_atom_coordinates[p][i][0],
			     20, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD1 );
	      
	    }
	  }
	  
	} /* if (parallel1_flag==1) */
	
	/* In case of parallel1_flag==0 */
	else {
	
	  for (p=1; p<=NEB_Num_Images; p++){ 
	    
	    sprintf(fname1,"%s_%i",fname_original,p);
	    argv[1] = fname1;
	    neb_run( argv,MPI_COMM_WORLD1,p,neb_atom_coordinates, 
		     WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	    
	    /* store Grid_Origin */
	    All_Grid_Origin[p][1] = Grid_Origin[1];
	    All_Grid_Origin[p][2] = Grid_Origin[2];
	    All_Grid_Origin[p][3] = Grid_Origin[3];
	  }
	}
      
	MPI_Barrier(MPI_COMM_WORLD1);
      
	/* calculate the gradients defined by the NEB method */
      
	Calc_NEB_Gradients(neb_atom_coordinates);
      
	/* make a xyz file storing a set of structures of images at the current step */
      
	Make_XYZ_File(fileXYZ,neb_atom_coordinates);    
      
        /* make a md file storing a set of structures of images at the current step */
      
        Make_MD_File(fileMD,neb_atom_coordinates);    

	/* update the atomic structures of the images */
	
	Update_Coordinates(iter,neb_atom_coordinates);
	
	/* generate an input file for restarting */
	
	Generate_Restart_File(fname_original,neb_atom_coordinates); 
	
	/* increment of iter */
      
	iter++;
      
      } while (MD_Opt_OK==0 && iter<=MD_IterNumber);

      MPI_Barrier(MPI_COMM_WORLD1);
    
      /* freeing of arrays for the World1 */
    
      if ( Num_Comm_World1<=numprocs ){
      
        MPI_Comm_free(&MPI_CommWD1[myworld1]);

	free(MPI_CommWD1);
	free(Comm_World_StartID1);
	free(NPROCS_WD1);
	free(Comm_World1);
	free(NPROCS_ID1);
      }
    
      /* freeing of arrays for the World2 */
    
      if ( Num_Comm_World2<=numprocs ){

        MPI_Comm_free(&MPI_CommWD2[myworld2]);

	free(MPI_CommWD2);
	free(Comm_World_StartID2);
	free(NPROCS_WD2);
	free(Comm_World2);
	free(NPROCS_ID2);
      }
    
      /* freeing of arrays */
    
      free_arrays();
    
      /* show a final message */
    
    
      dtime(&TEtime);
  
      printf("\nThe calculation was normally finished. (proc=%3d) TIME=%lf (s) flat time=%lf \n",
               myid,(TEtime-TStime),sumtime/(iter-1));

      MPI_Finalize();
      exit(0);

    } /* if ( NEB_Num_Images<=numprocs || numprocs==1 ) */

    /****************************************************
       One level parallelization is performed. 
       Outer loop: images
    ****************************************************/
  
    else{

      dtime(&a0time);
      int syou,amari,g,bb;

      syou =  NEB_Num_Images/numprocs;
      amari = NEB_Num_Images%numprocs;
      
      if(amari==0){
	g=syou;
      }
      else{
	g=syou+1;
      }

      Num_Comm_World1=numprocs;
      
      parallel1_flag = 1; 
      
      NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs); 
      Comm_World1 = (int*)malloc(sizeof(int)*numprocs); 
      NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
      Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
      MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);
      
      Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World1,      &myworld1, MPI_CommWD1, 
		       NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);
      
      MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
      MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);
      
      /****************************************************
         allocate processes to each image in the World2
      ****************************************************/
      
      Num_Comm_World2 = 2;
	
      if ( Num_Comm_World2<=numprocs ){
	  
	parallel2_flag = 1; 
      
	NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs); 
	Comm_World2 = (int*)malloc(sizeof(int)*numprocs); 
	NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
	Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
	MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);
      
	Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World2, &myworld2, MPI_CommWD2, 
			 NPROCS_ID2, Comm_World2, NPROCS_WD2, Comm_World_StartID2);
	  
	MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
	MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);
      }
	
      /****************************************************
        SCF calculations for the two terminal structures
      ****************************************************/

      /* check whether the restart is performed or not */

      sprintf(file_neb_utot,"%s%s_0_rst/%s.neb.utot",filepath,system_name,system_name);

      if (Scf_RestartFromFile==1 && (fp = fopen(file_neb_utot,"rb"))!=NULL){

        if (myid==Host_ID){
          fread(&neb_atom_coordinates[0][0][0],sizeof(double),1,fp);
          fread(&neb_atom_coordinates[NEB_Num_Images+1][0][0],sizeof(double),1,fp);
	}

        MPI_Bcast(&neb_atom_coordinates[0][0][0], 1, MPI_DOUBLE, Host_ID, MPI_COMM_WORLD1);
        MPI_Bcast(&neb_atom_coordinates[NEB_Num_Images+1][0][0], 1, MPI_DOUBLE, Host_ID, MPI_COMM_WORLD1);

        fclose(fp);
      }     

      /* if the restart is not performed, perform the SCF calulations for the two terminals */ 

      else {
	
	/* generate input files */

	generate_input_files(fname_original,1);
	
	/* In case of parallel2_flag==1 */
    
	if (parallel2_flag==1){
	  
	  if (myworld2==0)  
	    index_images = 0;
	  else 
	    index_images = NEB_Num_Images + 1;
	
	  sprintf(fname1,"%s_%i",fname_original,index_images);
	  argv[1] = fname1;
      
	  neb_run( argv,MPI_CommWD2[myworld2],index_images,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
      
	  /* find a representative ID in the top world for ID=0 in the world2 */
      
	  i = 0; po = 0; 
	  do {
	
	    if (Comm_World2[i]==0){
	      ID0 = i; 
	      po = 1;
	    }
	    i++;
	  } while (po==0);
      
	  /* find a representative ID in the top world for ID=1 in the world2 */
      
	  i = 0; po = 0; 
	  do {
	
	    if (Comm_World2[i]==1){
	      ID1 = i; 
	      po = 1;
	    }
	    i++;
	  } while (po==0);
      
	  /* MPI broadcast of neb_atom_coordinates[0] and neb_atom_coordinates[NEB_Num_Images+1] */
      
	  MPI_Barrier(MPI_COMM_WORLD1);
	  for (i=0; i<=atomnum; i++){
	    MPI_Bcast(&neb_atom_coordinates[0][i][0], 20, MPI_DOUBLE, ID0, MPI_COMM_WORLD1);
	    MPI_Bcast(&neb_atom_coordinates[NEB_Num_Images+1][i][0], 20, MPI_DOUBLE, ID1, MPI_COMM_WORLD1);
	  }
      
	} /* if (parallel2_flag==1) */

        // save *_0_rst/*.neb.utot in binary mode 

        if (myid==Host_ID){

          sprintf(file_neb_utot,"%s%s_0_rst/%s.neb.utot",filepath,system_name,system_name);

	  if ((fp = fopen(file_neb_utot,"wb")) != NULL){

	    fwrite(&neb_atom_coordinates[0][0][0],sizeof(double),1,fp);
	    fwrite(&neb_atom_coordinates[NEB_Num_Images+1][0][0],sizeof(double),1,fp);
	    fclose(fp);
	  }
	  else{
	    printf("Could not open a file %s in neb\n",file_neb_utot);
	  }
	}

      } /* else */

      /* amari world */

      if(amari!=0){
	Num_Comm_World3=amari;
      
	NPROCS_ID3 = (int*)malloc(sizeof(int)*numprocs); 
	Comm_World3 = (int*)malloc(sizeof(int)*numprocs); 
	NPROCS_WD3 = (int*)malloc(sizeof(int)*Num_Comm_World3); 
	Comm_WOrld_StartID3 = (int*)malloc(sizeof(int)*Num_Comm_World3); 
	MPI_CommWD3 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World3);
	
	Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World3, &myworld3, MPI_CommWD3, 
			 NPROCS_ID3, Comm_World3, NPROCS_WD3, Comm_WOrld_StartID3);
	
	MPI_Comm_size(MPI_CommWD3[myworld3],&numprocs3);
	MPI_Comm_rank(MPI_CommWD3[myworld3],&myid3);
      }
      /* amari world */
    
      dtime(&a1time);
    
      /****************************************************
        optimization for finding a minimum energy path
        connecting the two terminal structures.
      ****************************************************/
      
      iter = 1;  
      MD_Opt_OK = 0;
      dtime(&b0time);
      
      do {
	
	/* if iter==1, generate an input file for restarting  */
	
	if (iter==1) Generate_Restart_File(fname_original,neb_atom_coordinates); 

	/* generate input files */
	
	generate_input_files(fname_original,iter);
	
	for(h=1;h<=g;h++){
	  if(h==g && g==syou+1){
	    sprintf(fname1,"%s_%i",fname_original,myworld3+1+(h-1)*numprocs);
	  }
	  else{
	    sprintf(fname1,"%s_%i",fname_original,myworld1+1+(h-1)*numprocs);
	  }
	  argv[1] = fname1;
	  if((h==g) && (g==syou+1)){
	    neb_run( argv,MPI_CommWD3[myworld3],myworld3+1+(h-1)*numprocs,neb_atom_coordinates,
		     WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	    
	    if (myid3==Host_ID){
	      Tmp_Grid_Origin[myworld3+1+(h-1)*numprocs][1] = Grid_Origin[1];
	      Tmp_Grid_Origin[myworld3+1+(h-1)*numprocs][2] = Grid_Origin[2];
	      Tmp_Grid_Origin[myworld3+1+(h-1)*numprocs][3] = Grid_Origin[3];
	    }
	  }
	  else{
	    neb_run( argv,MPI_CommWD1[myworld1],myworld1+1+(h-1)*numprocs,neb_atom_coordinates,
		     WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	    
	    if(h==1){
	      for (i=0; i<=(NEB_Num_Images+1); i++){
		for (j=0; j<=3; j++) Tmp_Grid_Origin[i][j] = 0.0;
	      }  
	    }
	    if (myid1==Host_ID){
	      Tmp_Grid_Origin[myworld1+1+(h-1)*numprocs][1] = Grid_Origin[1];
	      Tmp_Grid_Origin[myworld1+1+(h-1)*numprocs][2] = Grid_Origin[2];
	      Tmp_Grid_Origin[myworld1+1+(h-1)*numprocs][3] = Grid_Origin[3];
	    }
	  }
	
	  /* MPI: All_Grid_Origin */

          MPI_Barrier(MPI_COMM_WORLD1);

	}

	MPI_Barrier(MPI_COMM_WORLD1);
	
	for (p=1; p<=NEB_Num_Images; p++){ 
	  MPI_Allreduce( &Tmp_Grid_Origin[p][0], &All_Grid_Origin[p][0], 
			 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD1 );
	}
	  
	/* MPI: neb_atom_coordinates */
	
	for (p=1; p<=NEB_Num_Images; p++){
	  for (i=0; i<=atomnum; i++){
	    for (j=0; j<20; j++){ 
	      tmp_neb_atom_coordinates[p][i][j] = 0.0;
	    }
	  }
	}
	
	for(h=1;h<=g;h++){
	  if(h==g && g==syou+1){
	    if (myid3==Host_ID){
	      for (i=0; i<=atomnum; i++){
		for (j=0; j<20; j++){ 
		  tmp_neb_atom_coordinates[myworld3+1+(h-1)*numprocs][i][j] = neb_atom_coordinates[myworld3+1+(h-1)*numprocs][i][j];;
		}
	      }
	    }
	  }
	  
	  else{
	    if (myid1==Host_ID){
	      for (i=0; i<=atomnum; i++){
		for (j=0; j<20; j++){ 
		  tmp_neb_atom_coordinates[myworld1+1+(h-1)*numprocs][i][j] = neb_atom_coordinates[myworld1+1+(h-1)*numprocs][i][j];;
		}
	      }
	    }
	  }
	}
	
	MPI_Barrier(MPI_COMM_WORLD1);
	
	for (p=1; p<=NEB_Num_Images; p++){ 
	  for (i=0; i<=atomnum; i++){
	    
	    MPI_Allreduce( &tmp_neb_atom_coordinates[p][i][0], &neb_atom_coordinates[p][i][0],
			   20, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD1 );
	    
	  }
	}
	
	MPI_Barrier(MPI_COMM_WORLD1);
	
	/* calculate the gradients defined by the NEB method */
	
	Calc_NEB_Gradients(neb_atom_coordinates);
	
	/* make a xyz file storing a set of structures of images at the current step */
	
	Make_XYZ_File(fileXYZ,neb_atom_coordinates);    

        /* make a md file storing a set of structures of images at the current step */
      
        Make_MD_File(fileMD,neb_atom_coordinates);    
	
	/* update the atomic structures of the images */
	
	Update_Coordinates(iter,neb_atom_coordinates);
	
	/* generate an input file for restarting */
	
	Generate_Restart_File(fname_original,neb_atom_coordinates); 
	
	/* increment of iter */
	
	iter++;
	
      } while (MD_Opt_OK==0 && iter<=MD_IterNumber);
      
      dtime(&b1time);
      
      MPI_Barrier(MPI_COMM_WORLD1);
      
      /* freeing of arrays for the World1 */
      
      dtime(&c0time);

      if (Num_Comm_World1<=numprocs) MPI_Comm_free(&MPI_CommWD1[myworld1]);

      free(MPI_CommWD1);
      free(Comm_World_StartID1);
      free(NPROCS_WD1);
      free(Comm_World1);
      free(NPROCS_ID1);
            
      /* freeing of arrays for the World2 */
      
      if (Num_Comm_World2<=numprocs) MPI_Comm_free(&MPI_CommWD2[myworld2]);

      free(MPI_CommWD2);
      free(Comm_World_StartID2);
      free(NPROCS_WD2);
      free(Comm_World2);
      free(NPROCS_ID2);
      
      /* freeing of arrays for the World3 */
      
      if(amari!=0){

        if (Num_Comm_World3<=numprocs) MPI_Comm_free(&MPI_CommWD3[myworld3]);

	free(MPI_CommWD3);
	free(Comm_WOrld_StartID3);
	free(NPROCS_WD3);
	free(Comm_World3);
	free(NPROCS_ID3);
	
      }
      
      /* freeing of arrays */
      
      free_arrays();
      
      /* show a final message */
      dtime(&c1time);
      
      dtime(&TEtime);
      
      printf("\nThe calculation was normally finished. (proc=%3d) TIME=%lf (s)\n",myid,(TEtime-TStime));

      MPI_Finalize();
      exit(0); 
    } 

  } /* if (PN==0) */

  /****************************************************************************
                  If MD.NEB.Parallel.Number (PN) is specifed,
                  corresponding to 1<=MD.NEB.Parallel.Number (PN)
  *****************************************************************************/

  else{

    dtime(&a0time);
    int syou,amari,g,bb;

    syou  = NEB_Num_Images/PN;
    amari = NEB_Num_Images%PN;
    
    if (amari==0) g = syou;
    else          g = syou + 1;
    
    Num_Comm_World1 = PN;

    if(PN>numprocs){

      if (myid==Host_ID) printf("PN is larger than the number of MPI processes.\n");

      MPI_Finalize();
      exit(0);
    }
    
    parallel1_flag = 1; 
    
    NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs); 
    Comm_World1 = (int*)malloc(sizeof(int)*numprocs); 
    NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
    Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
    MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);
    
    Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World1, &myworld1, MPI_CommWD1, 
		     NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);
    
    MPI_Comm_size(MPI_CommWD1[myworld1],&numprocs1);
    MPI_Comm_rank(MPI_CommWD1[myworld1],&myid1);
    
    /****************************************************
       allocate processes to each image in the World2
    ****************************************************/
    
    Num_Comm_World2 = 2;
    
    if ( Num_Comm_World2<=numprocs ){
      
      parallel2_flag = 1; 
      
      NPROCS_ID2 = (int*)malloc(sizeof(int)*numprocs); 
      Comm_World2 = (int*)malloc(sizeof(int)*numprocs); 
      NPROCS_WD2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
      Comm_World_StartID2 = (int*)malloc(sizeof(int)*Num_Comm_World2); 
      MPI_CommWD2 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2);
      
      Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World2, &myworld2, MPI_CommWD2, 
		       NPROCS_ID2, Comm_World2, NPROCS_WD2, Comm_World_StartID2);
      
      MPI_Comm_size(MPI_CommWD2[myworld2],&numprocs2);
      MPI_Comm_rank(MPI_CommWD2[myworld2],&myid2);
    }
    else{
      parallel2_flag = 0; 
      myworld2 = 0;
    }
    
    /****************************************************
      SCF calculations for the two terminal structures
    ****************************************************/

    /* check whether the restart is performed or not */

    sprintf(file_neb_utot,"%s%s_0_rst/%s.neb.utot",filepath,system_name,system_name);

    if (Scf_RestartFromFile==1 && (fp = fopen(file_neb_utot,"rb"))!=NULL){

      if (myid==Host_ID){
        fread(&neb_atom_coordinates[0][0][0],sizeof(double),1,fp);
        fread(&neb_atom_coordinates[NEB_Num_Images+1][0][0],sizeof(double),1,fp);
      }
 
      MPI_Bcast(&neb_atom_coordinates[0][0][0], 1, MPI_DOUBLE, Host_ID, MPI_COMM_WORLD1);
      MPI_Bcast(&neb_atom_coordinates[NEB_Num_Images+1][0][0], 1, MPI_DOUBLE, Host_ID, MPI_COMM_WORLD1);

      fclose(fp);
    }     

    /* if the restart is not performed, perform the SCF calulations for the two terminals */ 

    else {

      /* generate input files */

      generate_input_files(fname_original,1);

      /***********************************
                if (PN==1)
      ***********************************/

      if (PN==1){

	for (i=0; i<=1; i++){

	  if (i==0) index_images = 0;
	  else      index_images = NEB_Num_Images + 1; 

	  sprintf(fname1,"%s_%i",fname_original,index_images);
	  argv[1] = fname1;

	  neb_run( argv,MPI_CommWD2[myworld2],index_images,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	}
      }

      /***********************************
                if (2<=PN)
      ***********************************/

      else {
          
	if (myworld2==0)  
	  index_images = 0;
	else 
	  index_images = NEB_Num_Images + 1;
      
	sprintf(fname1,"%s_%i",fname_original,index_images);
	argv[1] = fname1;
      
	neb_run( argv,MPI_CommWD2[myworld2],index_images,neb_atom_coordinates,
		 WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
      
	/* find a representative ID in the top world for ID=0 in the world2 */
      
	i = 0; po = 0; 
	do {
	
	  if (Comm_World2[i]==0){
	    ID0 = i; 
	    po = 1;
	  }
	  i++;
	} while (po==0);
      
	/* find a representative ID in the top world for ID=1 in the world2 */
      
	i = 0; po = 0; 
	do {
	
	  if (Comm_World2[i]==1){
	    ID1 = i; 
	    po = 1;
	  }
	  i++;
	} while (po==0);
      
	/* MPI broadcast of neb_atom_coordinates[0] and neb_atom_coordinates[NEB_Num_Images+1] */
      
	MPI_Barrier(MPI_COMM_WORLD1);
	for (i=0; i<=atomnum; i++){
	  MPI_Bcast(&neb_atom_coordinates[0][i][0], 20, MPI_DOUBLE, ID0, MPI_COMM_WORLD1);
	  MPI_Bcast(&neb_atom_coordinates[NEB_Num_Images+1][i][0], 20, MPI_DOUBLE, ID1, MPI_COMM_WORLD1);
	}

      } /* else which corresponds to if (2<=PN) */

      // save *_0_rst/*.neb.utot in binary mode 

      if (myid==Host_ID){

	sprintf(file_neb_utot,"%s%s_0_rst/%s.neb.utot",filepath,system_name,system_name);

	if ((fp = fopen(file_neb_utot,"wb")) != NULL){

	  fwrite(&neb_atom_coordinates[0][0][0],sizeof(double),1,fp);
	  fwrite(&neb_atom_coordinates[NEB_Num_Images+1][0][0],sizeof(double),1,fp);
	  fclose(fp);
	}
	else{
	  printf("Could not open a file %s in neb\n",file_neb_utot);
	}
      }

    } /* else */

    /****************************************************
                      make amari world
    ****************************************************/

    if (amari!=0){

      Num_Comm_World3 = amari;
      
      NPROCS_ID3 = (int*)malloc(sizeof(int)*numprocs); 
      Comm_World3 = (int*)malloc(sizeof(int)*numprocs); 
      NPROCS_WD3 = (int*)malloc(sizeof(int)*Num_Comm_World3); 
      Comm_WOrld_StartID3 = (int*)malloc(sizeof(int)*Num_Comm_World3); 
      MPI_CommWD3 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World3);
      
      Make_Comm_Worlds(MPI_COMM_WORLD1, myid, numprocs, Num_Comm_World3, &myworld3, MPI_CommWD3, 
		       NPROCS_ID3, Comm_World3, NPROCS_WD3, Comm_WOrld_StartID3);
      
      MPI_Comm_size(MPI_CommWD3[myworld3],&numprocs3);
      MPI_Comm_rank(MPI_CommWD3[myworld3],&myid3);
    }
    /* amari world */
    
    dtime(&a1time);
    
    /****************************************************
     optimization for finding a minimum energy path
     connecting the two terminal structures.
    ****************************************************/
    
    iter = 1;  
    MD_Opt_OK = 0;
    dtime(&b0time);
    
    do {

      /* if iter==1, generate an input file for restarting  */
	
      if (iter==1) Generate_Restart_File(fname_original,neb_atom_coordinates); 

      /* generate input files */
	
      generate_input_files(fname_original,iter);
      
      for (h=1; h<=g; h++){

	if( h==g && g==(syou+1) ){
	  sprintf(fname1,"%s_%i",fname_original,myworld3+1+(h-1)*PN);
	}
	else{
	  sprintf(fname1,"%s_%i",fname_original,myworld1+1+(h-1)*PN);
	}

	argv[1] = fname1;

	if( h==g && g==(syou+1) ){

	  neb_run( argv,MPI_CommWD3[myworld3],myworld3+1+(h-1)*PN,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	  
	  if (myid3==Host_ID){
	    Tmp_Grid_Origin[myworld3+1+(h-1)*PN][1] = Grid_Origin[1];
	    Tmp_Grid_Origin[myworld3+1+(h-1)*PN][2] = Grid_Origin[2];
	    Tmp_Grid_Origin[myworld3+1+(h-1)*PN][3] = Grid_Origin[3];
	  }
	}

	else{

	  neb_run( argv,MPI_CommWD1[myworld1],myworld1+1+(h-1)*PN,neb_atom_coordinates,
		   WhatSpecies_NEB,Spe_WhatAtom_NEB,SpeName_NEB );
	  
	  if(h==1){
	    for (i=0; i<=(NEB_Num_Images+1); i++){
	      for (j=0; j<=3; j++) Tmp_Grid_Origin[i][j] = 0.0;
	    }  
	  }
	  if (myid1==Host_ID){
	    Tmp_Grid_Origin[myworld1+1+(h-1)*PN][1] = Grid_Origin[1];
	    Tmp_Grid_Origin[myworld1+1+(h-1)*PN][2] = Grid_Origin[2];
	    Tmp_Grid_Origin[myworld1+1+(h-1)*PN][3] = Grid_Origin[3];
	  }
	}
	
        /* MPI: All_Grid_Origin */
        MPI_Barrier(MPI_COMM_WORLD1);

      } /* end of for (h=1; h<=g; h++) */

      MPI_Barrier(MPI_COMM_WORLD1);
      
      for (p=1; p<=NEB_Num_Images; p++){ 
	MPI_Allreduce( &Tmp_Grid_Origin[p][0], &All_Grid_Origin[p][0], 
		       4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD1 );
      }
      
      /* MPI: neb_atom_coordinates */
      
      for (p=1; p<=NEB_Num_Images; p++){
	for (i=0; i<=atomnum; i++){
	  for (j=0; j<20; j++){ 
	    tmp_neb_atom_coordinates[p][i][j] = 0.0;
	  }
	}
      }
      
      for(h=1;h<=g;h++){
	if(h==g && g==syou+1){
	  if (myid3==Host_ID){
	    for (i=0; i<=atomnum; i++){
	      for (j=0; j<20; j++){ 
		tmp_neb_atom_coordinates[myworld3+1+(h-1)*PN][i][j] = neb_atom_coordinates[myworld3+1+(h-1)*PN][i][j];;
	      }
	    }
	  }
	}
	
	else{
	  if (myid1==Host_ID){
	    for (i=0; i<=atomnum; i++){
	      for (j=0; j<20; j++){ 
		tmp_neb_atom_coordinates[myworld1+1+(h-1)*PN][i][j] = neb_atom_coordinates[myworld1+1+(h-1)*PN][i][j];;
	      }
	    }
	  }
	}
      }
      
      MPI_Barrier(MPI_COMM_WORLD1);
      
      for (p=1; p<=NEB_Num_Images; p++){ 
	for (i=0; i<=atomnum; i++){
	  
	  MPI_Allreduce( &tmp_neb_atom_coordinates[p][i][0], &neb_atom_coordinates[p][i][0],
			 20, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD1);
	  
	}
      }
      
      MPI_Barrier(MPI_COMM_WORLD1);
      
      /* calculate the gradients defined by the NEB method */
      
      Calc_NEB_Gradients(neb_atom_coordinates);
      
      /* make a xyz file storing a set of structures of images at the current step */
      
      Make_XYZ_File(fileXYZ,neb_atom_coordinates);    

      /* make a md file storing a set of structures of images at the current step */
      
      Make_MD_File(fileMD,neb_atom_coordinates);    
      
      /* update the atomic structures of the images */
      
      Update_Coordinates(iter,neb_atom_coordinates);
      
      /* generate an input file for restarting */
      
      Generate_Restart_File(fname_original,neb_atom_coordinates); 
      
      /* increment of iter */
      
      iter++;
      
    } while (MD_Opt_OK==0 && iter<=MD_IterNumber);
    
    dtime(&b1time);
    
    MPI_Barrier(MPI_COMM_WORLD1);
    
    /* freeing of arrays for the World1 */
    
    dtime(&c0time);
    
    if (Num_Comm_World1<=numprocs) MPI_Comm_free(&MPI_CommWD1[myworld1]);

    free(MPI_CommWD1);
    free(Comm_World_StartID1);
    free(NPROCS_WD1);
    free(Comm_World1);
    free(NPROCS_ID1);
    
    /* freeing of arrays for the World2 */
    
    if (Num_Comm_World2<=numprocs) MPI_Comm_free(&MPI_CommWD2[myworld2]);

    free(MPI_CommWD2);
    free(Comm_World_StartID2);
    free(NPROCS_WD2);
    free(Comm_World2);
    free(NPROCS_ID2);
    
    /* freeing of arrays for the World3 */
    
    if(amari!=0){

      if (Num_Comm_World3<=numprocs) MPI_Comm_free(&MPI_CommWD3[myworld3]);
      
      free(MPI_CommWD3);
      free(Comm_WOrld_StartID3);
      free(NPROCS_WD3);
      free(Comm_World3);
      free(NPROCS_ID3);
      
    }
    
    /* freeing of arrays */
    
    free_arrays();
    
    /* show a final message */
    dtime(&c1time);
    
    dtime(&TEtime);
    
    printf("\nThe calculation was normally finished. (proc=%3d) TIME=%lf (s)\n",myid,(TEtime-TStime));

    MPI_Finalize();
    exit(0); 
  }
}



void Generate_Restart_File(char fname_original[YOUSO10], double ***neb_atom_coordinates)
{
  int i,p,Gc_AN,c,n1,k,po;
  int restart_flag;
  int unit_flag;
  double c1,c2,c3;
  double tmpxyz[4];
  char st[800];
  char st1[800];
  char rm_operate[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  char keyword[YOUSO10];
  FILE *fp1,*fp2;
  char *tp;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  /* generate the input files */

  if (myid==Host_ID){

    /* initialize */

    restart_flag = 0;
    unit_flag = 0;

    /* the new input file */    

    sprintf(fname1,"%s#",fname_original);
    fp1 = fopen(fname1,"w");
    fseek(fp1,0,SEEK_END);

    /* the original input file */    

    fp2 = fopen(fname_original,"r");

    if (fp2!=NULL){

      while (fgets(st,800,fp2)!=NULL){

	string_tolower(st,st1); 

        /* find the specification of <atoms.speciesandcoordinates */

	if (strncmp(st1,"<atoms.speciesandcoordinates",28)==0){

	  fprintf(fp1,"%s",st);

	  /* replace the atomic coordinates */

	  for (i=1; i<=atomnum; i++){

	    fgets(st,800,fp2);
	    string_tolower(st,st1);

	    /* serial number */
	    tp = strtok(st, " ");
	    if (tp!=NULL) fprintf(fp1,"%4s",tp);

	    /* name of species */
	    tp =strtok(NULL, " ");  
	    if (tp!=NULL) fprintf(fp1," %4s",tp);

            /* "Ang" */ 
            if (coordinates_unit==0){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[0][i][1]*BohrR);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[0][i][2]*BohrR);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[0][i][3]*BohrR);
	    }

            /* AU */
            else if (coordinates_unit==1){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[0][i][1]);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[0][i][2]);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[0][i][3]);
	    }

            /* FRAC */
            else if (coordinates_unit==2){

              /* The zero is taken as the origin of the unit cell. */

              tmpxyz[1] = neb_atom_coordinates[0][i][1] - Grid_Origin[1];
              tmpxyz[2] = neb_atom_coordinates[0][i][2] - Grid_Origin[2];
              tmpxyz[3] = neb_atom_coordinates[0][i][3] - Grid_Origin[3];

   	      c1 = Dot_Product(tmpxyz,rtv[1])*0.5/PI;
              c2 = Dot_Product(tmpxyz,rtv[2])*0.5/PI;
              c3 = Dot_Product(tmpxyz,rtv[3])*0.5/PI;

	      /* a-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c1);

	      /* b-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c2);

	      /* c-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c3);
	    }

	    while (tp!=NULL){
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1,"     %s",tp);
	    }
	  } 
	}

        /* find the specification of <atoms.speciesandcoordinates */

	else if (strncmp(st1,"<neb.atoms.speciesandcoordinates",32)==0){

	  fprintf(fp1,"%s",st);

	  /* replace the atomic coordinates */

	  for (i=1; i<=atomnum; i++){

	    fgets(st,800,fp2);
	    string_tolower(st,st1);

	    /* serial number */
	    tp = strtok(st, " ");
	    if (tp!=NULL) fprintf(fp1,"%4s",tp);

	    /* name of species */
	    tp =strtok(NULL, " ");  
	    if (tp!=NULL) fprintf(fp1," %4s",tp);

            /* "Ang" */ 
            if (coordinates_unit==0){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[NEB_Num_Images+1][i][1]*BohrR);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[NEB_Num_Images+1][i][2]*BohrR);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[NEB_Num_Images+1][i][3]*BohrR);
	    }

            /* AU */
            else if (coordinates_unit==1){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[NEB_Num_Images+1][i][1]);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[NEB_Num_Images+1][i][2]);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",neb_atom_coordinates[NEB_Num_Images+1][i][3]);
	    }

            /* FRAC */
            else if (coordinates_unit==2){

              /* The zero is taken as the origin of the unit cell. */

              tmpxyz[1] = neb_atom_coordinates[NEB_Num_Images+1][i][1] - Grid_Origin[1];
              tmpxyz[2] = neb_atom_coordinates[NEB_Num_Images+1][i][2] - Grid_Origin[2];
              tmpxyz[3] = neb_atom_coordinates[NEB_Num_Images+1][i][3] - Grid_Origin[3];

   	      c1 = Dot_Product(tmpxyz,rtv[1])*0.5/PI;
              c2 = Dot_Product(tmpxyz,rtv[2])*0.5/PI;
              c3 = Dot_Product(tmpxyz,rtv[3])*0.5/PI;

	      /* a-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c1);

	      /* b-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c2);

	      /* c-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c3);
	    }

	    while (tp!=NULL){
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1,"     %s",tp);
	    }

	  } 
	}

	/* sc.restart */

	else if (strncmp(st1,"scf.restart",11)==0){
	  fprintf(fp1,"scf.restart    on\n");
	  restart_flag = 1;
	}

	else {

          po = 0;

          for (p=1; p<=NEB_Num_Images; p++){ 

             /* find the specification of <neb%d.atoms.speciesandcoordinates and forget it */

	     k = (int)log10(p) + 1;
             sprintf(keyword,"<neb%i.atoms.speciesandcoordinates",p);

	     if (strncmp(st1,keyword,32+k)==0){

	       po = 1;

               /* get atomic coordinates and forget them */
	       for (i=1; i<=atomnum; i++) fgets(st,800,fp2);

               /* get neb%i.atoms.speciesandcoordinates> and forget it */
 	       fgets(st,800,fp2);

 	     }
	  } /* p */

          /* other cases */           

          if (po==0) fprintf(fp1,"%s",st);

	} /* else */
      } /* while */

      /* close fp2 */
      fclose(fp2); 
    }

    /* add the restart flag if it was not found. */

    if (restart_flag==0){
      fprintf(fp1,"\n\nscf.restart    on\n");
    }

    /* add atomic coordinates of the images */

    for (p=1; p<=NEB_Num_Images; p++){ 

      fp2 = fopen(fname_original,"r");

      if (fp2!=NULL){

	while (fgets(st,800,fp2)!=NULL){

	  string_tolower(st,st1); 

	  if (strncmp(st1,"<atoms.speciesandcoordinates",28)==0){

	    fprintf(fp1,"\n<NEB%d.Atoms.SpeciesAndCoordinates\n",p);

	    for (k=1; k<=atomnum; k++){

	      fgets(st,800,fp2);
	      string_tolower(st,st1);

	      /* serial number */
	      tp = strtok(st, " ");
	      if (tp!=NULL) fprintf(fp1,"%4s",tp);

	      /* name of species */
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1," %4s",tp);

	      /* "Ang" */ 
	      if (coordinates_unit==0){
		/* x-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][k][1]*BohrR);

		/* y-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][k][2]*BohrR);

		/* z-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][k][3]*BohrR);
	      }

	      /* AU */
	      else if (coordinates_unit==1){

		/* x-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][k][1]);

		/* y-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][k][2]);

		/* z-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][k][3]);
	      }

	      /* FRAC */
	      else if (coordinates_unit==2){

		/* The zero is taken as the origin of the unit cell. */

		tmpxyz[1] = neb_atom_coordinates[p][k][1] - Grid_Origin[1];
		tmpxyz[2] = neb_atom_coordinates[p][k][2] - Grid_Origin[2];
		tmpxyz[3] = neb_atom_coordinates[p][k][3] - Grid_Origin[3];

		c1 = Dot_Product(tmpxyz,rtv[1])*0.5/PI;
		c2 = Dot_Product(tmpxyz,rtv[2])*0.5/PI;
		c3 = Dot_Product(tmpxyz,rtv[3])*0.5/PI;

		/* a-axis */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",c1);

		/* b-axis */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",c2);

		/* c-axis */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",c3);
	      }

	      while (tp!=NULL){
		tp =strtok(NULL, " ");  
		if (tp!=NULL) fprintf(fp1,"     %s",tp);
	      }

	    } /* k */  

	  }     

	  else if (strncmp(st1,"atoms.speciesandcoordinates>",28)==0){
	    fprintf(fp1,"NEB%d.Atoms.SpeciesAndCoordinates>\n",p);
	  }
	}

	/* close fp2 */
	fclose(fp2); 

      } /* if (fp2!=NULL){ */
    } /* p */

    /* fclose */
    fclose(fp1); 

  } /* if (myid==Host_ID) */

}





void Make_XYZ_File(char fname[YOUSO10], double ***neb_atom_coordinates)
{
  FILE *fp;
  int p,k,i,j;
  int myid;
  char file_neb_utot[YOUSO10];

  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  if (myid==Host_ID){

    /* save *.neb.xyz */

    if ((fp = fopen(fname,"w")) != NULL){

      for (p=0; p<=(NEB_Num_Images+1); p++){ 

	fprintf(fp,"%i \n",atomnum);
	fprintf(fp,"  Image index = %3d  Energy= %8.5f (Hatree)\n",p,neb_atom_coordinates[p][0][0]);

	for (k=1; k<=atomnum; k++){

	  i = WhatSpecies_NEB[k];
	  j = Spe_WhatAtom_NEB[i];

	  fprintf(fp,"%4s   %10.7f  %10.7f  %10.7f  %10.7f  %10.7f  %10.7f\n",
		  Atom_Symbol[j],                
		  neb_atom_coordinates[p][k][1]*BohrR,
		  neb_atom_coordinates[p][k][2]*BohrR,
		  neb_atom_coordinates[p][k][3]*BohrR,
		 -neb_atom_coordinates[p][k][4],
		 -neb_atom_coordinates[p][k][5],
		 -neb_atom_coordinates[p][k][6]);
	}
      }
      fclose(fp);
    }
    else{
      printf("failure of saving the xyz file.\n");
      fclose(fp);
    }

  } /* if (myid==Host_ID) */

}    


void Make_MD_File(char fname[YOUSO10], double ***neb_atom_coordinates)
{
  FILE *fp;
  int p,k,i,j;
  int myid;

  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  if (myid==Host_ID){

    /* save *.neb.md */

    if ((fp = fopen(fname,"w")) != NULL){

      for (p=0; p<=(NEB_Num_Images+1); p++){ 

	fprintf(fp,"%i \n",atomnum);
	fprintf(fp,"  Image index = %3d  Energy= %8.5f (Hatree) ",p,neb_atom_coordinates[p][0][0]);

        fprintf(fp,"Cell_Vectors= ");
        for (i=1; i<=3; i++){
          for (j=1; j<=3; j++){
            fprintf(fp,"%8.5f ",tv[i][j]*BohrR);
          }
        }
        fprintf(fp,"\n");

	for (k=1; k<=atomnum; k++){

	  i = WhatSpecies_NEB[k];
	  j = Spe_WhatAtom_NEB[i];

	  fprintf(fp,"%4s   %8.5f %8.5f %8.5f  %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f  %8.5f %8.5f %8.5f %8.5f\n",
		  Atom_Symbol[j],                
		  neb_atom_coordinates[p][k][1]*BohrR,
		  neb_atom_coordinates[p][k][2]*BohrR,
		  neb_atom_coordinates[p][k][3]*BohrR,
		 -neb_atom_coordinates[p][k][4],
		 -neb_atom_coordinates[p][k][5],
	         -neb_atom_coordinates[p][k][6],
                  0.0,0.0,0.0, /* velocity */
                  neb_atom_coordinates[p][k][7], /* Net charge, electron charge is defined to be negative. */
                  neb_atom_coordinates[p][k][8], /* magnetic moment (muB) */
                  neb_atom_coordinates[p][k][9], /* angle0 */
                  neb_atom_coordinates[p][k][10] /* angle1 */
		  );
	}
      }
      fclose(fp);
    }
    else{
      printf("failure of saving the md file.\n");
      fclose(fp);
    }

  } /* if (myid==Host_ID) */

}    




void Update_Coordinates(int iter, double ***neb_atom_coordinates)
{
  int p,j,k;
  double norm,dis,dif;
  double Max_Force,Max_Step;  
  double tmp0,sum,sum_E,sum_dis;
  int numprocs,myid;
  char fileOPT[YOUSO10];
  FILE *fp_OPT;
  char fileE[YOUSO10];
  FILE *fp_E;

  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  /*********************************************
          calculate the norm of gradient
  *********************************************/

  norm = 0.0;
  for (p=1; p<=NEB_Num_Images; p++){ 
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
        norm += neb_atom_coordinates[p][j][k+3]*neb_atom_coordinates[p][j][k+3];  
      }
    }    
  }

  norm = sqrt(norm);

  /****************************************************
    find the maximum value of force 
  ****************************************************/

  Max_Force = 0.0;
  for (p=1; p<=NEB_Num_Images; p++){ 
    for (j=1; j<=atomnum; j++){

      sum = 0.0;
      for (k=1; k<=3; k++){
        tmp0 = neb_atom_coordinates[p][j][k+3];
        sum += tmp0*tmp0;
      }
      sum = sqrt(sum);
      if (Max_Force<sum) Max_Force = sum;
    }
  }      

  if (Max_Force<MD_Opt_criterion) MD_Opt_OK = 1;

  /*********************************************
             update atomic coordinates
  *********************************************/

  Max_Step = DIIS_BFGS(iter, neb_atom_coordinates,norm,Max_Force);

  /*
  Steepest_Decent(iter, neb_atom_coordinates,norm,Max_Force);
  */

  /*********************************************
         save the history of optimization
  *********************************************/

  if (myid==Host_ID){ 

    sum_E = 0.0;
    for (p=1; p<=NEB_Num_Images; p++){ 
      sum_E += neb_atom_coordinates[p][0][0]; 
    }

    sprintf(fileOPT,"%s%s.neb.opt",filepath,system_name);
  
    if ((fp_OPT = fopen(fileOPT,"a")) != NULL){

      if (iter==1){
        fprintf(fp_OPT,"\n");
        fprintf(fp_OPT,"#***********************************************************\n");
        fprintf(fp_OPT,"#***********************************************************\n");
        fprintf(fp_OPT,"#         History of optimization by the NEB method         \n");
        fprintf(fp_OPT,"#***********************************************************\n");
        fprintf(fp_OPT,"#***********************************************************\n");
        fprintf(fp_OPT,"#\n");
        fprintf(fp_OPT,"#     iter   SD_scaling     |Maximum force|   Maximum step        Norm        Sum of Total Energy of Images\n");
        fprintf(fp_OPT,"#                           (Hartree/Bohr)        (Ang)       (Hartree/Bohr)       (Hartree)  \n\n");
      }

      fprintf(fp_OPT,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f    %15.8f\n",
              iter,SD_scaling,Max_Force,Max_Step,norm,sum_E);

      fclose(fp_OPT);
    }
    else
      printf("Error in saving the neb.opt file\n");

  } /* if (myid==Host_ID) */

  /*********************************************
     save the total DFT energy of each image
     and iterdistance between them.
  *********************************************/

  if (myid==Host_ID){ 
    
    sprintf(fileE,"%s%s.neb.ene",filepath,system_name);

    if ((fp_E = fopen(fileE,"w")) != NULL){

      fprintf(fp_E,"#\n");
      fprintf(fp_E,"# 1st column: index of images, where 0 and MD.NEB.Number.Images+1 are the terminals\n");
      fprintf(fp_E,"# 2nd column: Total energy (Hartree) of each image\n");
      fprintf(fp_E,"# 3rd column: distance (Bohr) between neighbors\n");
      fprintf(fp_E,"# 4th column: distance (Bohr) from the image of the index 0\n");
      fprintf(fp_E,"# 5th column: x distance\n " );
      fprintf(fp_E,"#\n");

      sum_dis = 0.0;

      for (p=0; p<=(NEB_Num_Images+1); p++){ 

        if (p!=0){
	  dis = 0.0;
	  for (j=1; j<=atomnum; j++){
	    for (k=1; k<=3; k++){
	      dif = neb_atom_coordinates[p][j][k] - neb_atom_coordinates[p-1][j][k];
	      dis += dif*dif;
	    }
	  }
	}
        else{
          dis = 0.0;
        }

        sum_dis += dis; 
    
        fprintf(fp_E,"  %3d  %15.8f  %15.8f  %15.8f  %15.8f\n",
                p, neb_atom_coordinates[p][0][0], dis, sum_dis, neb_atom_coordinates[p][2][1]*BohrR-neb_atom_coordinates[p][1][1]*BohrR);

      }

      fclose(fp_E);
    }
    else
      printf("Error in saving the neb.ene file\n");

  }

}




double DIIS_BFGS(int iter, double ***neb_atom_coordinates, double norm, double Max_Force)
{
  int p,i,j,k,m,n,dim;
  int p1,i1,j1,p2,i2,j2;
  double sum,tmp1,tmp2,tmp,c0,c1;
  double Max_Step,scaleF;
  double SD_min,SD_max,SD_init,dt;
  double dif_g,dif_x,dif_x1,dif_x2;
  double sum0,sum1;
  char *JOBB="L";
  double *A,*B,max_A,RR;
  double *work;
  int *ipiv;
  INTEGER LDA,LWORK,info,N;

  /****************************************************
                       SD method
  ****************************************************/

  if (iter<OptStartDIIS){

    /*************************************************************
                   store coordinates and gradients
    *************************************************************/

    for (p=1; p<=NEB_Num_Images; p++){ 
      for (j=1; j<=atomnum; j++){
	for (k=1; k<=6; k++){
	  His_neb_atom_coordinates[1][p][j][k] = His_neb_atom_coordinates[0][p][j][k];
	  His_neb_atom_coordinates[0][p][j][k] = neb_atom_coordinates[p][j][k];
	}
      }
    }      

    /****************************************************
                      set up SD_scaling
    ****************************************************/

    dt = 41.3411*4.0;
    SD_init = dt*dt/1836.1526;
    SD_max = SD_init*10.0;   /* default 10  */
    SD_min = SD_init*0.005;  /* default 0.2 */

    if (iter==1){

      SD_scaling_user = Max_Force/BohrR/5.0;
      SD_scaling = SD_scaling_user/(Max_Force+1.0e-10);

      if (SD_max<SD_scaling) SD_scaling = SD_max;
      if (SD_scaling<SD_min) SD_scaling = SD_min;

      Past_Norm[1] = norm;
    }
    else {

      if (Past_Norm[1]<norm && iter%4==1){ 
	SD_scaling = SD_scaling/4.0;
      }
      else if (Past_Norm[1]<Past_Norm[2] && norm<Past_Norm[1] && iter%4==1){
	SD_scaling = SD_scaling*1.2;
      }

      if (SD_max<SD_scaling) SD_scaling = SD_max;
      if (SD_scaling<SD_min) SD_scaling = SD_min;

      Past_Norm[5] = Past_Norm[4];
      Past_Norm[4] = Past_Norm[3];
      Past_Norm[3] = Past_Norm[2];
      Past_Norm[2] = Past_Norm[1];
      Past_Norm[1] = norm;
    }

    /*************************************************************
     update coordinate by the simple steepest decent (SD) method 
    *************************************************************/

    Max_Step = 0.0;
    for (p=1; p<=NEB_Num_Images; p++){ 
      for (j=1; j<=atomnum; j++){
	for (k=1; k<=3; k++){

	  tmp = SD_scaling*neb_atom_coordinates[p][j][k+3];
	  neb_atom_coordinates[p][j][k] -= tmp;
	  if (Max_Step<fabs(tmp)) Max_Step = fabs(tmp);
	}
      }
    }      

  } /* if (iter<OptStartDIIS) */

  /****************************************************
                    DIIS + BFGS method
  ****************************************************/

  else {

    /*************************************************************
                   store coordinates and gradients
    *************************************************************/

    for (p=1; p<=NEB_Num_Images; p++){ 
      for (j=1; j<=atomnum; j++){
	for (k=1; k<=6; k++){

          for (m=(M_GDIIS_HISTORY-1); 0<m; m--){
  	    His_neb_atom_coordinates[m][p][j][k] = His_neb_atom_coordinates[m-1][p][j][k];
	  }

	  His_neb_atom_coordinates[0][p][j][k] = neb_atom_coordinates[p][j][k];
	}
      }
    }      

    /*************************************************************
          estimate an optimum coordinates by the DIIS method
    *************************************************************/

    dim = iter - OptStartDIIS + 1;
    if (M_GDIIS_HISTORY<dim) dim = M_GDIIS_HISTORY;

    /* allocation of arrays */

    A = (double*)malloc(sizeof(double)*(dim+1)*(dim+1));
    B = (double*)malloc(sizeof(double)*(dim+1));

    /* construct the A matrix */

    for (m=0; m<dim; m++){
      for (n=0; n<dim; n++){

        sum = 0.0;

	for (p=1; p<=NEB_Num_Images; p++){ 
	  for (j=1; j<=atomnum; j++){
	    for (k=4; k<=6; k++){
              sum += His_neb_atom_coordinates[m][p][j][k]*His_neb_atom_coordinates[n][p][j][k];
	    }
	  }
	}

        A[m*(dim+1)+n] = sum;
      }
    }

    /* find max of A and scale elements by it. */

    max_A = 0.0;
    for (m=0; m<dim; m++){
      for (n=0; n<dim; n++){
        RR = fabs(A[m*(dim+1)+n]) ;
        if (max_A<RR ) max_A = RR;
      }
    }

    max_A = 1.0/max_A;

    for (m=0; m<dim; m++){
      for (n=0; n<dim; n++){
        A[m*(dim+1)+n] *= max_A;
      }
    }

    for (m=0; m<dim; m++){
      A[m*(dim+1)+dim] = 1.0;
      A[dim*(dim+1)+m] = 1.0;
    } 
    A[dim*(dim+1)+dim] = 0.0;

    /*
    printf("Mat A\n");
    for (m=0; m<=dim; m++){
      for (n=0; n<=dim; n++){
        printf("%10.5f ",A[m*(dim+1)+n]);
      }
      printf("\n");
    }
    */

    for (m=0; m<dim; m++) B[m] = 0.0;
    B[dim] = 1.0;

    /* solve Ax = B */

    N     = dim + 1;
    LDA   = dim + 1;
    LWORK = dim + 1;
    work  = (double*)malloc(sizeof(double)*LWORK);
    ipiv  = (int*)malloc(sizeof(int)*(dim+1));
    i     = 1; 

    F77_NAME(dsysv,DSYSV)( JOBB, &N, &i, A, &LDA,  ipiv, B, &LDA, work, &LWORK, &info);

    if (info!=0) {
      printf(" ERROR in dsysv_, info=%d\n",info);
      MD_Opt_OK =1; 
      goto Last_Step;
    }

    /*
    for (m=0; m<dim; m++){
      printf("m=%2d B=%15.12f\n",m,B[m]);
    }
    */
  
    /* update the coordinates by the DIIS method */

    for (p=1; p<=NEB_Num_Images; p++){ 
      for (j=1; j<=atomnum; j++){
	for (k=1; k<=3; k++){

          sum0 = 0.0; 
          sum1 = 0.0;
          for (m=0; m<dim; m++){
  	    sum0 += B[m]*His_neb_atom_coordinates[m][p][j][k  ];
  	    sum1 += B[m]*His_neb_atom_coordinates[m][p][j][k+3];
	  }

	  neb_atom_coordinates[p][j][k]   = sum0;
	  neb_atom_coordinates[p][j][k+3] = sum1;
	}
      }
    }      

    Last_Step:

    /* freeing of arrays */

    free(work);
    free(ipiv);
    free(A);
    free(B);

    /*************************************************************
           update an approximate Hessian by the BFGS method 
    *************************************************************/

    if (iter==OptStartDIIS){

      for (p1=1; p1<=NEB_Num_Images; p1++){
	for (i1=1; i1<=atomnum; i1++){
	  for (j1=1; j1<=3; j1++){
	    for (p2=1; p2<=NEB_Num_Images; p2++){
	      for (i2=1; i2<=atomnum; i2++){
		for (j2=1; j2<=3; j2++){
                  InvHess[p1][i1][j1][p2][i2][j2] = 0.0;
		}
	      }
	    }
            InvHess[p1][i1][j1][p1][i1][j1] = 1.0;
	  }
	}
      }
    }

    else {

      /* (H^(n-1))^{-1} Delta g^(n) */

      for (p1=1; p1<=NEB_Num_Images; p1++){
	for (i1=1; i1<=atomnum; i1++){
	  for (j1=1; j1<=3; j1++){

	    sum = 0.0; 
	    for (p2=1; p2<=NEB_Num_Images; p2++){
	      for (i2=1; i2<=atomnum; i2++){
		for (j2=1; j2<=3; j2++){
		  dif_g = His_neb_atom_coordinates[0][p2][i2][j2+3]-His_neb_atom_coordinates[1][p2][i2][j2+3];
		  sum += InvHess[p1][i1][j1][p2][i2][j2]*dif_g;
		}
	      }
	    }
        
	    Vec0[p1][i1][j1] = sum;
	  }
	}
      }  

      /* < Delta x^(n) | Delta g^(n) > */

      tmp1 = 0.0; 
      for (p2=1; p2<=NEB_Num_Images; p2++){
	for (i2=1; i2<=atomnum; i2++){
	  for (j2=1; j2<=3; j2++){

	    dif_x = His_neb_atom_coordinates[0][p2][i2][j2  ]-His_neb_atom_coordinates[1][p2][i2][j2  ];
	    dif_g = His_neb_atom_coordinates[0][p2][i2][j2+3]-His_neb_atom_coordinates[1][p2][i2][j2+3];
	    tmp1 += dif_x*dif_g;
	  }
	}
      }

      /* < Delta g^(n) | Vec0 > */
    
      tmp2 = 0.0; 
      for (p2=1; p2<=NEB_Num_Images; p2++){
	for (i2=1; i2<=atomnum; i2++){
	  for (j2=1; j2<=3; j2++){

	    dif_g = His_neb_atom_coordinates[0][p2][i2][j2+3]-His_neb_atom_coordinates[1][p2][i2][j2+3];
	    tmp2 += dif_g*Vec0[p2][i2][j2];
	  }
	}
      }

      /* update InvHess */

      c0 = (tmp1 + tmp2)/(tmp1*tmp1);
      c1 = -1.0/tmp1;

      for (p1=1; p1<=NEB_Num_Images; p1++){
	for (i1=1; i1<=atomnum; i1++){
	  for (j1=1; j1<=3; j1++){
	    for (p2=1; p2<=NEB_Num_Images; p2++){
	      for (i2=1; i2<=atomnum; i2++){
		for (j2=1; j2<=3; j2++){
              
		  dif_x1 = His_neb_atom_coordinates[0][p1][i1][j1]
		          -His_neb_atom_coordinates[1][p1][i1][j1];

		  dif_x2 = His_neb_atom_coordinates[0][p2][i2][j2]
                          -His_neb_atom_coordinates[1][p2][i2][j2];

		  InvHess[p1][i1][j1][p2][i2][j2] += c0*dif_x1*dif_x2;

		  dif_x1 = Vec0[p1][i1][j1];
		  dif_x2 = His_neb_atom_coordinates[0][p2][i2][j2]
                          -His_neb_atom_coordinates[1][p2][i2][j2];

		  InvHess[p1][i1][j1][p2][i2][j2] += c1*dif_x1*dif_x2;

		  dif_x1 = His_neb_atom_coordinates[0][p1][i1][j1]
                          -His_neb_atom_coordinates[1][p1][i1][j1];
		  dif_x2 = Vec0[p2][i2][j2];

		  InvHess[p1][i1][j1][p2][i2][j2] += c1*dif_x1*dif_x2;

		}
	      }
	    }
	  }
	}
      }     

    } /* else */

    /* determine the pre-factor */

    if      (0.7<Max_Force)    scaleF = 0.1;
    else if (0.30<Max_Force)   scaleF = 0.2;
    else if (0.10<Max_Force)   scaleF = 0.3;
    else if (0.05<Max_Force)   scaleF = 0.4;
    else if (0.020<Max_Force)  scaleF = 0.5;
    else if (0.010<Max_Force)  scaleF = 0.6;
    else if (0.0050<Max_Force) scaleF = 0.7;
    else if (0.0020<Max_Force) scaleF = 0.8;
    else if (0.0010<Max_Force) scaleF = 0.9;
    else                       scaleF = 1.0;

    /* update atomic coordinates */

    for (p1=1; p1<=NEB_Num_Images; p1++){
      for (i1=1; i1<=atomnum; i1++){
	for (j1=1; j1<=3; j1++){

          sum1 = 0.0;  
	  for (p2=1; p2<=NEB_Num_Images; p2++){
	    for (i2=1; i2<=atomnum; i2++){
	      for (j2=1; j2<=3; j2++){

		tmp = neb_atom_coordinates[p2][i2][j2+3]; 
		sum1 += InvHess[p1][i1][j1][p2][i2][j2]*tmp;
	      }
	    }
	  }

          neb_atom_coordinates[p1][i1][j1] -= scaleF*sum1;
	}
      }
    }  

    /* In case of a too large updating, do a modest updating */

    Max_Step = 0.0;
    for (p=1; p<=NEB_Num_Images; p++){ 
      for (j=1; j<=atomnum; j++){
	for (k=1; k<=3; k++){

          dif_x = fabs(neb_atom_coordinates[p][j][k] - His_neb_atom_coordinates[0][p][j][k]);
          if (Max_Step<dif_x) Max_Step = dif_x; 
	}
      }
    }    

    if (Criterion_Max_Step<Max_Step){
      
      for (p=1; p<=NEB_Num_Images; p++){ 

	for (j=1; j<=atomnum; j++){
	  for (k=1; k<=3; k++){

	    dif_x = neb_atom_coordinates[p][j][k] - His_neb_atom_coordinates[0][p][j][k];
	    neb_atom_coordinates[p][j][k] = His_neb_atom_coordinates[0][p][j][k]
                                          + dif_x/Max_Step*Criterion_Max_Step; 
	  }
	}
      }    

      Max_Step = Criterion_Max_Step;
    }

  } /* else */

  return Max_Step;
}







void Steepest_Decent(int iter, double ***neb_atom_coordinates, double norm, double Max_Force)
{
  int p,j,k;
  double SD_min,SD_max,SD_init;

  /****************************************************
   set up SD_scaling
  ****************************************************/

  SD_init = 0.1/(Max_Force+1.0e-10);
  SD_max = SD_init*20.0;  
  SD_min = SD_init*0.05;  

  if (iter==1){
    SD_scaling = 0.1/(Max_Force+1.0e-10);
  }
  else {

    if (Past_Norm[1]<norm && iter%4==1){ 
      SD_scaling = SD_scaling/4.0;
    }
    else if (Past_Norm[1]<Past_Norm[2] && norm<Past_Norm[1] && iter%4==1){
      SD_scaling = SD_scaling*1.2;
    }

    Past_Norm[5] = Past_Norm[4];
    Past_Norm[4] = Past_Norm[3];
    Past_Norm[3] = Past_Norm[2];
    Past_Norm[2] = Past_Norm[1];
    Past_Norm[1] = norm;
  }

  if (SD_max<SD_scaling) SD_scaling = SD_max;
  if (SD_scaling<SD_min) SD_scaling = SD_min;

  /*************************************************************
    update coordinate by the simple steepest decent (SD) method 
  *************************************************************/

  for (p=1; p<=NEB_Num_Images; p++){ 
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
        neb_atom_coordinates[p][j][k] -= SD_scaling*neb_atom_coordinates[p][j][k+3];
      }
    }
  }      
}





void Calc_NEB_Gradients(double ***neb_atom_coordinates)
{
  int i,j,k,p,myid,pmax;
  double ***tangents;
  double dtmp1,dtmp2;
  double sum,prod1;
  double sconst,vmax;
  double Dvmax,Dvmin,vm,vi,vp;
  double dis1,dis2,dif;

  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  /* allocation of arrays */

  tangents = (double***)malloc(sizeof(double**)*(NEB_Num_Images+2));
  for (i=0; i<(NEB_Num_Images+2); i++){
    tangents[i] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (j=0; j<(atomnum+1); j++){
      tangents[i][j] = (double*)malloc(sizeof(double)*4);
      for (k=0; k<4; k++) tangents[i][j][k] = 0.0;
    }
  }

  /***********************************************************************
     calculate tangents based on Eqs. (8)-(11) in JCP 113, 9978 (2000)
  ***********************************************************************/

  for (p=1; p<=NEB_Num_Images; p++){ 

    vm = neb_atom_coordinates[p-1][0][0];
    vi = neb_atom_coordinates[p+0][0][0];
    vp = neb_atom_coordinates[p+1][0][0];

    if ( vm<vi && vi<vp ){

      for (j=1; j<=atomnum; j++){
	for (k=1; k<=3; k++){
	  tangents[p][j][k] = neb_atom_coordinates[p+1][j][k] - neb_atom_coordinates[p][j][k];
	}
      }     
    }    

    else if ( vp<vi && vi<vm ){

      for (j=1; j<=atomnum; j++){
	for (k=1; k<=3; k++){
	  tangents[p][j][k] = neb_atom_coordinates[p][j][k] - neb_atom_coordinates[p-1][j][k];
	}
      }     
    }    

    else if ( (vp>=vi && vi<=vm) || (vp<vi && vi>vm) ){

      Dvmax = MAX(fabs(vp-vi),fabs(vm-vi));
      Dvmin = MIN(fabs(vp-vi),fabs(vm-vi));

      if (vm<=vp){

	for (j=1; j<=atomnum; j++){
	  for (k=1; k<=3; k++){
	    tangents[p][j][k] = Dvmax*(neb_atom_coordinates[p+1][j][k] - neb_atom_coordinates[p  ][j][k])
	                      + Dvmin*(neb_atom_coordinates[p  ][j][k] - neb_atom_coordinates[p-1][j][k]);
	  }
	}     
      }
      else {

	for (j=1; j<=atomnum; j++){
	  for (k=1; k<=3; k++){
	    tangents[p][j][k] = Dvmin*(neb_atom_coordinates[p+1][j][k] - neb_atom_coordinates[p  ][j][k])
	                      + Dvmax*(neb_atom_coordinates[p  ][j][k] - neb_atom_coordinates[p-1][j][k]);
	  }
	}     
      }

    }    
    else {
      printf("unknown situation\n");
    }

    /* normalization of tangent */

    sum = 0.0;
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
        sum += tangents[p][j][k]*tangents[p][j][k];
      }
    } 

    sum = 1.0/sqrt(sum);
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
        tangents[p][j][k] *= sum;
      }
    } 
  }

  /***********************************************************************
                  calculate forces defined by the NEB method
  ***********************************************************************/

  for (p=1; p<=NEB_Num_Images; p++){ 

    /****************************************************************
                           Perpendicular force 
    ****************************************************************/

    /* inner product of gradient[p] and tangents[p] */

    prod1 = 0.0;
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
        prod1 += neb_atom_coordinates[p][j][k+16]*tangents[p][j][k];
      }
    } 

    /* calculate the perpendicular gradient */

    if ( neb_type_switch==1 ){ /* NEB */
      for (j=1; j<=atomnum; j++){
	for (k=1; k<=3; k++){
	  neb_atom_coordinates[p][j][k+3] = neb_atom_coordinates[p][j][k+16] - prod1*tangents[p][j][k];
	}
      } 
    }

    /****************************************************************
                              Parallel force 
    ****************************************************************/

    /* force constant */

    sconst = NEB_Spring_Const;
    
    /* calculate the distance between Ri+1 and Ri */

    dis1 = 0.0; 
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
	dif = neb_atom_coordinates[p+1][j][k] - neb_atom_coordinates[p][j][k];
	dis1 += dif*dif;
      }
    } 
    dis1 = sqrt(dis1);

    /* calculate the distance between Ri and Ri-1 */

    dis2 = 0.0; 
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
	dif = neb_atom_coordinates[p][j][k] - neb_atom_coordinates[p-1][j][k];
	dis2 += dif*dif;
      }
    } 
    dis2 = sqrt(dis2);

    /* calculate the parallel gradient by springs, and add the contribution 
       based on Eq. (12) in JCP 113, 9978 (2000) */

    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
	neb_atom_coordinates[p][j][k+3] += -sconst*(dis1 - dis2)*tangents[p][j][k];
      }
    }    

  } /* p */

  /****************************************************************
              introduce the contraints defined by user 
              by setting gradients zero
              1: fixed 
              0: relaxed
  ****************************************************************/

  for (p=1; p<=NEB_Num_Images; p++){ 
    for (j=1; j<=atomnum; j++){
      for (k=1; k<=3; k++){
        if (atomFixedXYZ[j][k]==1){
          neb_atom_coordinates[p][j][k+3] = 0.0;
	}
      }
    }    
  }

  /* freeing of arrays */

  for (i=0; i<(NEB_Num_Images+2); i++){
    for (j=0; j<(atomnum+1); j++){
      free(tangents[i][j]);
    }
    free(tangents[i]);
  }
  free(tangents);
}





void generate_input_files(char *file, int iter)
{
  int i,p,Gc_AN,c,n1,k;
  int restart_flag,fixed_flag;
  int unit_flag,level_flag;
  double c1,c2,c3;
  double tmpxyz[4];
  char st[800];
  char st1[800];
  char rm_operate[YOUSO10];
  char fname[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  FILE *fp1,*fp2;
  char *tp;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  /* get system.name */

  if (input_open(file)==0){
    MPI_Finalize(); 
    exit(0);
  }

  input_string("System.CurrrentDirectory",filepath,"./");
  input_string("System.Name",filename,"default");

  input_close();

  /* generate the input files */

  if (myid==Host_ID){

    for (p=0; p<=(NEB_Num_Images+1); p++){ 

      /* initialize */

      restart_flag = 0;
      fixed_flag = 0;
      unit_flag = 0;
      level_flag = 0;

      /* the new input file */    

      sprintf(fname1,"%s_%i",file,p);
      fp1 = fopen(fname1,"w");
      fseek(fp1,0,SEEK_END);

      /* the original input file */    

      fp2 = fopen(file,"r");

      if (fp2!=NULL){

	while (fgets(st,800,fp2)!=NULL){

	  string_tolower(st,st1); 

	  /* find the specification of <atoms.speciesandcoordinates */

	  if (strncmp(st1,"<atoms.speciesandcoordinates",28)==0){

	    fprintf(fp1,"%s",st);

	    /* replace the atomic coordinates */

	    for (i=1; i<=atomnum; i++){

	      fgets(st,800,fp2);
	      string_tolower(st,st1);

	      /* serial number */
	      tp = strtok(st, " ");
	      if (tp!=NULL) fprintf(fp1,"%4s",tp);

	      /* name of species */
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1," %4s",tp);

	      /* "Ang" */ 
	      if (coordinates_unit==0){
		/* x-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][i][1]*BohrR);

		/* y-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][i][2]*BohrR);

		/* z-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][i][3]*BohrR);
	      }

	      /* AU */
	      else if (coordinates_unit==1){

		/* x-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][i][1]);

		/* y-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][i][2]);

		/* z-coordinate */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",neb_atom_coordinates[p][i][3]);
	      }

	      /* FRAC */
	      else if (coordinates_unit==2){

		/* The zero is taken as the origin of the unit cell. */

		tmpxyz[1] = neb_atom_coordinates[p][i][1] - Grid_Origin[1];
		tmpxyz[2] = neb_atom_coordinates[p][i][2] - Grid_Origin[2];
		tmpxyz[3] = neb_atom_coordinates[p][i][3] - Grid_Origin[3];

		c1 = Dot_Product(tmpxyz,rtv[1])*0.5/PI;
		c2 = Dot_Product(tmpxyz,rtv[2])*0.5/PI;
		c3 = Dot_Product(tmpxyz,rtv[3])*0.5/PI;

		/* a-axis */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",c1);

		/* b-axis */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",c2);

		/* c-axis */
		tp =strtok(NULL, " ");  
		fprintf(fp1,"  %18.14f",c3);
	      }

	      while (tp!=NULL){
		tp =strtok(NULL, " ");  
		if (tp!=NULL) fprintf(fp1,"     %s",tp);
	      }

	    } 
	  }

	  /* scf.restart for iter==1 */

	  else if (strncmp(st1,"scf.restart",11)==0 && iter==1 && Scf_RestartFromFile==0){
	    fprintf(fp1,"scf.restart    off\n");
	    restart_flag = 1;
	  }

	  else if (strncmp(st1,"scf.restart",11)==0 && iter==1 && Scf_RestartFromFile==1){
	    fprintf(fp1,"scf.restart    on\n");
	    restart_flag = 1;
	  }

	  /* scf.restart for iter!=1 */

	  else if (strncmp(st1,"scf.restart",11)==0 && iter!=1){
	    fprintf(fp1,"scf.restart    on\n");
	    restart_flag = 1;
	  }

	  /* scf.fixed.grid */

	  else if (strncmp(st1,"scf.fixed.grid",14)==0){
	    fprintf(fp1,"%s",st);
	    fixed_flag = 1;
	  }  

	  /* System.Name */

	  else if (strncmp(st1,"system.name",11)==0){
	    fprintf(fp1,"System.Name                   %s_%i\n",filename,p);
	  }  

	  /* level.of.stdout */

	  else if (strncmp(st1,"level.of.stdout",15)==0 && p<=1 ){
	    fprintf(fp1,"level.of.stdout                   1\n");
	    level_flag = 1;
	  }  

	  else if (strncmp(st1,"level.of.stdout",15)==0 && 1<p ){
	    fprintf(fp1,"level.of.stdout                  -1\n");
	    level_flag = 1;
	  }  

	  else{
	    fprintf(fp1,"%s",st);
	  }
	}

	fclose(fp2); 
      }

      /* add the restart flag if it was not found. */

      if (restart_flag==0 && iter==1){
	fprintf(fp1,"\n\nscf.restart    off\n");
      }
      else if (restart_flag==0 && iter!=1){
	fprintf(fp1,"\n\nscf.restart    on\n");
      }

      /* add scf.fixed.grid if it was not found. */

      if (fixed_flag==0 && 1<=p && p<=NEB_Num_Images && 1<iter){
	fprintf(fp1,"\n\nscf.fixed.grid   %18.14f  %18.14f  %18.14f\n",
		All_Grid_Origin[p][1],All_Grid_Origin[p][2],All_Grid_Origin[p][3]);
      }

      /* add level.of.stdout if it was not found. */

      if (level_flag==0 && p<=1){
	fprintf(fp1,"\n\nlevel.of.stdout        1\n");
      }
      else if (level_flag==0 && 1<p){
	fprintf(fp1,"\n\nlevel.of.stdout        0\n");
      }

      /* fclose */
      fclose(fp1); 

    } /* p */
  } /* if (myid==Host_ID) */

  MPI_Barrier(MPI_COMM_WORLD1);

}



void read_input(char *file)
{
  int i,j,k,p;
  int SpinP_switch;
  int po,itmp;
  int unitvector_unit;
  int unitvectors_flag;
  int itmp1,itmp2,itmp3,itmp4;
  int myid,numprocs;
  double tv[4][4];
  double dtmp1,dtmp2,dtmp3,dtmp4,dtmp5;
  double dtmp6,dtmp7,dtmp8,dtmp9;
  double tmpx,tmpy,tmpz,x,y,z;
  double dif,sx,sy,sz;
  double xc0,yc0,zc0,xc1,yc1,zc1;
  double xc,yc,zc;
  char Species[100];
  char OrbPol[100];
  char buf[MAXBUF];
  char keyword[YOUSO10];  
  int i_vec[40];
  char *s_vec[40];
  FILE *fp;

  MPI_Comm_size(MPI_COMM_WORLD1,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid);

  /* open the input file */

  if (input_open(file)==0){
    MPI_Finalize(); 
    exit(0);
  }

  input_string("System.CurrrentDirectory",filepath,"./");
  input_string("System.Name",filename,"default");
  sprintf(system_name,"%s",filename);

  /**************************************************************
   NEB_Num_Images is the number of images (p) excluding 
   the end points (0 and p+1). 
  *************************************************************/

  input_int("MD.NEB.Number.Images",&NEB_Num_Images,10);
  input_double("MD.NEB.Spring.Const",&NEB_Spring_Const,0.4); /* hartree/bohr^2  */
  input_int("MD.maxIter",&MD_IterNumber,1);
  
  input_int("MD.NEB.Parallel.Number",&PN,0);

  s_vec[0]="Off"; s_vec[1]="On"; s_vec[2]="NC";
  i_vec[0]=0    ; i_vec[1]=1   ; i_vec[2]=3;
  input_string2int("scf.SpinPolarization", &SpinP_switch, 3, s_vec,i_vec);

  input_int("Atoms.Number",&atomnum,0);
  input_int("Species.Number",&SpeciesNum,0);

  input_int("MD.Opt.DIIS.History",&M_GDIIS_HISTORY,3);

  /* allocate arrays */
  allocate_arrays();

  /*********************************************************
                     read the unit cell
  *********************************************************/

  s_vec[0]="Ang"; s_vec[1]="AU";
  i_vec[0]=0;  i_vec[1]=1;
  input_string2int("Atoms.UnitVectors.Unit",&unitvector_unit,2,s_vec,i_vec);

  unitvectors_flag = 0; 

  if (fp=input_find("<Atoms.Unitvectors")) {

    unitvectors_flag = 1; 

    for (i=1; i<=3; i++){
      fscanf(fp,"%lf %lf %lf",&tv[i][1],&tv[i][2],&tv[i][3]);
    }
    if ( ! input_last("Atoms.Unitvectors>") ) {
      /* format error */
      if (myid==Host_ID){
        printf("Format error for Atoms.Unitvectors\n");
      }
      MPI_Finalize();
      exit(0);
    }

    /* Ang to AU */
    if (unitvector_unit==0){
      for (i=1; i<=3; i++){
	tv[i][1] = tv[i][1]/BohrR;
	tv[i][2] = tv[i][2]/BohrR;
	tv[i][3] = tv[i][3]/BohrR;
      }
    }
  }

  if (unitvectors_flag==0){
    if (myid==Host_ID){
      printf("A common unit cell for two terminal structures has to be specified.\n");fflush(stdout);
    }
    MPI_Finalize();
    exit(0);
  }  

  /**************************************************************
   read atomic coodinates given by Atoms.SpeciesAndCoordinates.
  **************************************************************/

  s_vec[0]="Ang";  s_vec[1]="AU";   s_vec[2]="FRAC";
  i_vec[0]= 0;     i_vec[1]= 1;     i_vec[2]= 2;
  input_string2int("Atoms.SpeciesAndCoordinates.Unit",
                     &coordinates_unit,3,s_vec,i_vec);

  if (fp=input_find("<Atoms.SpeciesAndCoordinates") ) {

    for (i=1; i<=atomnum; i++){

      fgets(buf,MAXBUF,fp);

      /* spin non-collinear */ 
      if (SpinP_switch==3){

	/*******************************************************
               (1) spin non-collinear
	*******************************************************/

	sscanf(buf,"%i %s %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %s",
	       &j, Species,
	       &neb_atom_coordinates[0][i][1],
               &neb_atom_coordinates[0][i][2],
               &neb_atom_coordinates[0][i][3],
	       &dtmp1,&dtmp2,
	       &dtmp3,&dtmp4,
	       &dtmp5,&dtmp6,
	       &itmp2,
	       OrbPol );

      }

      /**************************************************
                  (2) spin collinear
      **************************************************/

      else{ 

	sscanf(buf,"%i %s %lf %lf %lf %lf %lf %s",
	       &j, Species,
	       &neb_atom_coordinates[0][i][1],
               &neb_atom_coordinates[0][i][2],
               &neb_atom_coordinates[0][i][3],
	       &dtmp1,&dtmp2, OrbPol );
      }

      if (i!=j){
        if (myid==Host_ID){
   	  printf("Format error of the sequential number %i in <Atoms.SpeciesAndCoordinates\n",j);
	}
        MPI_Finalize();
        exit(0);
      }

    }

    ungetc('\n',fp);
    if (!input_last("Atoms.SpeciesAndCoordinates>")) {
      /* format error */
      if (myid==Host_ID){
        printf("Format error for Atoms.SpeciesAndCoordinates\n");fflush(stdout);
      }

      MPI_Finalize();
      exit(0);
    }

    /* Ang to AU */

    if (coordinates_unit==0){
      for (i=1; i<=atomnum; i++){
        neb_atom_coordinates[0][i][1] /= BohrR;
        neb_atom_coordinates[0][i][2] /= BohrR;
        neb_atom_coordinates[0][i][3] /= BohrR;
      }
    }
  }

  /*  FRAC to AU */ 
  if (coordinates_unit==2){

    /* The fractional coordinates should be kept within 0 to 1. */

    for (i=1; i<=atomnum; i++){
      for (k=1; k<=3; k++){

	itmp = (int)neb_atom_coordinates[0][i][k]; 

	if (1.0<neb_atom_coordinates[0][i][k]){

	  /* ended by T.Ohwaki */

	  neb_atom_coordinates[0][i][k] = neb_atom_coordinates[0][i][k] - (double)itmp;

	  if (myid==Host_ID){
	    if (k==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",i);
	  }
	}
	else if (neb_atom_coordinates[0][i][k]<0.0){

	  neb_atom_coordinates[0][i][k] = neb_atom_coordinates[0][i][k] + (double)(abs(itmp)+1);

	  if (myid==Host_ID){
	    if (k==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",i);
	  }
	}
      }
    }

    /* calculation of xyz-coordinate in A.U. The grid origin is zero. */

    tmpx = 0.0;
    tmpy = 0.0;
    tmpz = 0.0;

    for (i=1; i<=atomnum; i++){

      x = neb_atom_coordinates[0][i][1]*tv[1][1]
        + neb_atom_coordinates[0][i][2]*tv[2][1] 
        + neb_atom_coordinates[0][i][3]*tv[3][1] + tmpx;

      y = neb_atom_coordinates[0][i][1]*tv[1][2]
        + neb_atom_coordinates[0][i][2]*tv[2][2]
        + neb_atom_coordinates[0][i][3]*tv[3][2] + tmpy;

      z = neb_atom_coordinates[0][i][1]*tv[1][3]
        + neb_atom_coordinates[0][i][2]*tv[2][3]
        + neb_atom_coordinates[0][i][3]*tv[3][3] + tmpz;

      neb_atom_coordinates[0][i][1] = x;
      neb_atom_coordinates[0][i][2] = y;
      neb_atom_coordinates[0][i][3] = z;
    }
  }

  /*****************************************************************
   read atomic coodinates given by NEB.Atoms.SpeciesAndCoordinates
  *****************************************************************/

  if (fp=input_find("<NEB.Atoms.SpeciesAndCoordinates") ) {

    for (i=1; i<=atomnum; i++){

      fgets(buf,MAXBUF,fp);

      /* spin non-collinear */ 
      if (SpinP_switch==3){

	/*******************************************************
               (1) spin non-collinear
	*******************************************************/

	sscanf(buf,"%i %s %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %s",
	       &j, Species,
	       &neb_atom_coordinates[NEB_Num_Images+1][i][1],
               &neb_atom_coordinates[NEB_Num_Images+1][i][2],
               &neb_atom_coordinates[NEB_Num_Images+1][i][3],
	       &dtmp1,&dtmp2,
	       &dtmp3,&dtmp4,
	       &dtmp5,&dtmp6,
	       &itmp2,
	       OrbPol );

      }

      /**************************************************
                  (2) spin collinear
      **************************************************/

      else{ 

	sscanf(buf,"%i %s %lf %lf %lf %lf %lf %s",
	       &j, Species,
	       &neb_atom_coordinates[NEB_Num_Images+1][i][1],
               &neb_atom_coordinates[NEB_Num_Images+1][i][2],
               &neb_atom_coordinates[NEB_Num_Images+1][i][3],
	       &dtmp1,&dtmp2, OrbPol );
      }

      if (i!=j){
        if (myid==Host_ID){
  	  printf("Format error of the sequential number %i in <NEB.Atoms.SpeciesAndCoordinates\n",j);
	}
        MPI_Finalize();
        exit(0);
      }

    }

    ungetc('\n',fp);
    if (!input_last("NEB.Atoms.SpeciesAndCoordinates>")) {
      /* format error */
      if (myid==Host_ID){
        printf("Format error for NEB.Atoms.SpeciesAndCoordinates\n");
      }
      MPI_Finalize();
      exit(0);
    }

    /* Ang to AU */

    if (coordinates_unit==0){
      for (i=1; i<=atomnum; i++){
        neb_atom_coordinates[NEB_Num_Images+1][i][1] /= BohrR;
        neb_atom_coordinates[NEB_Num_Images+1][i][2] /= BohrR;
        neb_atom_coordinates[NEB_Num_Images+1][i][3] /= BohrR;
      }
    }
  }

  /*  FRAC to AU */ 
  if (coordinates_unit==2){

    /* The fractional coordinates should be kept within 0 to 1. */

    for (i=1; i<=atomnum; i++){
      for (k=1; k<=3; k++){

	itmp = (int)neb_atom_coordinates[NEB_Num_Images+1][i][k]; 

	if (1.0<neb_atom_coordinates[NEB_Num_Images+1][i][k]){

	  /* ended by T.Ohwaki */

	  neb_atom_coordinates[NEB_Num_Images+1][i][k] = neb_atom_coordinates[NEB_Num_Images+1][i][k] - (double)itmp;

	  if (myid==Host_ID){
	    if (k==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",i);
	  }
	}
	else if (neb_atom_coordinates[NEB_Num_Images+1][i][k]<0.0){

	  neb_atom_coordinates[NEB_Num_Images+1][i][k] = neb_atom_coordinates[NEB_Num_Images+1][i][k] + (double)(abs(itmp)+1);

	  if (myid==Host_ID){
	    if (k==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",i);
	    if (k==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",i);
	  }
	}
      }
    }

    /* calculation of xyz-coordinate in A.U. The grid origin is zero. */

    tmpx = 0.0;
    tmpy = 0.0;
    tmpz = 0.0;

    for (i=1; i<=atomnum; i++){

      x = neb_atom_coordinates[NEB_Num_Images+1][i][1]*tv[1][1]
        + neb_atom_coordinates[NEB_Num_Images+1][i][2]*tv[2][1] 
        + neb_atom_coordinates[NEB_Num_Images+1][i][3]*tv[3][1] + tmpx;

      y = neb_atom_coordinates[NEB_Num_Images+1][i][1]*tv[1][2] 
        + neb_atom_coordinates[NEB_Num_Images+1][i][2]*tv[2][2] 
        + neb_atom_coordinates[NEB_Num_Images+1][i][3]*tv[3][2] + tmpy;

      z = neb_atom_coordinates[NEB_Num_Images+1][i][1]*tv[1][3]
        + neb_atom_coordinates[NEB_Num_Images+1][i][2]*tv[2][3]
        + neb_atom_coordinates[NEB_Num_Images+1][i][3]*tv[3][3] + tmpz;

      neb_atom_coordinates[NEB_Num_Images+1][i][1] = x;
      neb_atom_coordinates[NEB_Num_Images+1][i][2] = y;
      neb_atom_coordinates[NEB_Num_Images+1][i][3] = z;
    }
  }

  /*****************************************************************
     if scf.restart==on, then read the coordinates of images
     and read total energies of the two terminal structures. 
  *****************************************************************/

  input_logical("scf.restart",&Scf_RestartFromFile, 0); 

  if (Scf_RestartFromFile){

    /* read the structures */

    for (p=1; p<=NEB_Num_Images; p++){ 

      sprintf(keyword,"<NEB%d.Atoms.SpeciesAndCoordinates",p);
    
      if (fp=input_find(keyword) ) {

        for (i=1; i<=atomnum; i++){

          fscanf(fp,"%i %s %lf %lf %lf %lf %lf",
  	         &j, Species,
                 &neb_atom_coordinates[p][i][1],
		 &neb_atom_coordinates[p][i][2],
		 &neb_atom_coordinates[p][i][3],
		 &dtmp1,&dtmp2);
	}

        sprintf(keyword,"NEB%d.Atoms.SpeciesAndCoordinates>",p);

	if (!input_last(keyword)) {
	  /* format error */
	  if (myid==Host_ID){
	    printf("Format error for NEB%d.Atoms.SpeciesAndCoordinates\n",p);
	  }
	  MPI_Finalize();
	  exit(0);
	}

      }
    }

  }

  else {

    /*****************************************************************
      Making the centroid of two terminal coordinates equivalent. 
      This treatment allows us to hold the equivalence of the centroid
      of all the images. Thus, Grid_Origin also becomes equivalent to 
      eath other. 
    *****************************************************************/

    xc0 = 0.0;
    yc0 = 0.0;
    zc0 = 0.0;

    xc1 = 0.0;
    yc1 = 0.0;
    zc1 = 0.0;
    
    for (i=1; i<=atomnum; i++){

      xc0 += neb_atom_coordinates[0][i][1];
      yc0 += neb_atom_coordinates[0][i][2];
      zc0 += neb_atom_coordinates[0][i][3];

      xc1 += neb_atom_coordinates[NEB_Num_Images+1][i][1];
      yc1 += neb_atom_coordinates[NEB_Num_Images+1][i][2];
      zc1 += neb_atom_coordinates[NEB_Num_Images+1][i][3];
    }

    xc0 = xc0/(double)atomnum; 
    yc0 = yc0/(double)atomnum; 
    zc0 = zc0/(double)atomnum; 

    xc1 = xc1/(double)atomnum; 
    yc1 = yc1/(double)atomnum; 
    zc1 = zc1/(double)atomnum; 

    xc = 0.5*(xc0 + xc1);
    yc = 0.5*(yc0 + yc1);
    zc = 0.5*(zc0 + zc1);

    for (i=1; i<=atomnum; i++){

      neb_atom_coordinates[0][i][1] += -xc0 + xc;
      neb_atom_coordinates[0][i][2] += -yc0 + yc;
      neb_atom_coordinates[0][i][3] += -zc0 + zc;

      neb_atom_coordinates[NEB_Num_Images+1][i][1] += -xc1 + xc;
      neb_atom_coordinates[NEB_Num_Images+1][i][2] += -yc1 + yc;
      neb_atom_coordinates[NEB_Num_Images+1][i][3] += -zc1 + zc;
    }

    /*****************************************************************
                  generate atomic coordinates for images 
    *****************************************************************/

    for (p=1; p<=NEB_Num_Images; p++){
      for (i=1; i<=atomnum; i++){
	for (j=1; j<=3; j++){
	  dif = neb_atom_coordinates[NEB_Num_Images+1][i][j]-neb_atom_coordinates[0][i][j];
	  neb_atom_coordinates[p][i][j] = neb_atom_coordinates[0][i][j]
	                                  + dif*(double)p/(double)(NEB_Num_Images+1);
	}
      }
    }
  }

  /*****************************************************************
   read atomFixedXYZ
   set fixed atomic position in geometry optimization
   and MD:  

      1: fixed 
      0: relaxed
  *****************************************************************/

  if (fp=input_find("<MD.Fixed.XYZ")) {

    for (i=1; i<=atomnum; i++){  
      fscanf(fp,"%d %d %d %d",
             &j,&atomFixedXYZ[i][1],&atomFixedXYZ[i][2],&atomFixedXYZ[i][3]);
    }  

    if ( ! input_last("MD.Fixed.XYZ>") ) {
      /* format error */

      if (myid==Host_ID){
        printf("Format error for MD.Fixed.XYZ\n");
      }
      MPI_Finalize();
      exit(0);
    }
  }

  /* close the input file */

  input_close();

}


void allocate_arrays()
{
  int i,j,k,m;
  int p1,i1,j1,p2,i2,j2;

  neb_atom_coordinates = (double***)malloc(sizeof(double**)*(NEB_Num_Images+2));
  for (i=0; i<(NEB_Num_Images+2); i++){
    neb_atom_coordinates[i] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (j=0; j<(atomnum+1); j++){
      neb_atom_coordinates[i][j] = (double*)malloc(sizeof(double)*20);
      for (k=0; k<20; k++) neb_atom_coordinates[i][j][k] = 0.0;
    }
  }

  tmp_neb_atom_coordinates = (double***)malloc(sizeof(double**)*(NEB_Num_Images+2));
  for (i=0; i<(NEB_Num_Images+2); i++){
    tmp_neb_atom_coordinates[i] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (j=0; j<(atomnum+1); j++){
      tmp_neb_atom_coordinates[i][j] = (double*)malloc(sizeof(double)*20);
      for (k=0; k<20; k++) neb_atom_coordinates[i][j][k] = 0.0;
    }
  }

  His_neb_atom_coordinates = (double****)malloc(sizeof(double***)*(M_GDIIS_HISTORY+2));
  for (m=0; m<(M_GDIIS_HISTORY+2); m++){
    His_neb_atom_coordinates[m] = (double***)malloc(sizeof(double**)*(NEB_Num_Images+2));
    for (i=0; i<(NEB_Num_Images+2); i++){
      His_neb_atom_coordinates[m][i] = (double**)malloc(sizeof(double*)*(atomnum+1));
      for (j=0; j<(atomnum+1); j++){
	His_neb_atom_coordinates[m][i][j] = (double*)malloc(sizeof(double)*20);
	for (k=0; k<20; k++) His_neb_atom_coordinates[m][i][j][k] = 0.0;
      }
    }
  }

  All_Grid_Origin = (double**)malloc(sizeof(double*)*(NEB_Num_Images+2));
  for (i=0; i<(NEB_Num_Images+2); i++){
    All_Grid_Origin[i] = (double*)malloc(sizeof(double)*4);
    for (j=0; j<4; j++) All_Grid_Origin[i][j] = 0.0;
  }  

  Tmp_Grid_Origin = (double**)malloc(sizeof(double*)*(NEB_Num_Images+2));
  for (i=0; i<(NEB_Num_Images+2); i++){
    Tmp_Grid_Origin[i] = (double*)malloc(sizeof(double)*4);
    for (j=0; j<4; j++) Tmp_Grid_Origin[i][j] = 0.0;
  }  

  WhatSpecies_NEB = (int*)malloc(sizeof(int)*(atomnum+1));
  Spe_WhatAtom_NEB = (int*)malloc(sizeof(int)*SpeciesNum);

  SpeName_NEB = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeName_NEB[i] = (char*)malloc(sizeof(char)*YOUSO10);
  }  

  InvHess = (double******)malloc(sizeof(double*****)*(NEB_Num_Images+2));
  for (p1=0; p1<(NEB_Num_Images+2); p1++){
    InvHess[p1] = (double*****)malloc(sizeof(double****)*(atomnum+1));
    for (i1=0; i1<(atomnum+1); i1++){
      InvHess[p1][i1] = (double****)malloc(sizeof(double***)*4);
      for (j1=0; j1<4; j1++){
        InvHess[p1][i1][j1] = (double***)malloc(sizeof(double**)*(NEB_Num_Images+2));
        for (p2=0; p2<(NEB_Num_Images+2); p2++){
          InvHess[p1][i1][j1][p2] = (double**)malloc(sizeof(double*)*(atomnum+1));
          for (i2=0; i2<(atomnum+1); i2++){
            InvHess[p1][i1][j1][p2][i2] = (double*)malloc(sizeof(double)*4);
            for (j2=0; j2<4; j2++){
              InvHess[p1][i1][j1][p2][i2][j2] = 0.0;
	    }
	  }
	}
      }
    }
  }  

  /* initialize InvHess */
  for (p1=1; p1<=NEB_Num_Images; p1++){
    for (i1=1; i1<=atomnum; i1++){
      for (j1=1; j1<=3; j1++){
        InvHess[p1][i1][j1][p1][i1][j1] = 1.0;
      }
    }
  }

  /*
  Estimate_Initial_Hessian();  
  */

  Vec0 = (double***)malloc(sizeof(double**)*(NEB_Num_Images+2));
  for (p1=0; p1<(NEB_Num_Images+2); p1++){
    Vec0[p1] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (i1=0; i1<(atomnum+1); i1++){
      Vec0[p1][i1] = (double*)malloc(sizeof(double)*4);
      for (j1=0; j1<4; j1++) Vec0[p1][i1][j1] = 0.0;
    }
  }

  atomFixedXYZ = (int**)malloc(sizeof(int*)*(atomnum+1));
  for(i=0; i<=atomnum; i++){
    atomFixedXYZ[i] = (int*)malloc(sizeof(int)*4);
    /* default='relaxed' */
    atomFixedXYZ[i][1] = 0;  
    atomFixedXYZ[i][2] = 0;
    atomFixedXYZ[i][3] = 0;
  }

}



void free_arrays()
{
  int m,i,j,k;
  int p1,i1,j1,p2,i2,j2;

  for (i=0; i<(NEB_Num_Images+2); i++){
    for (j=0; j<(atomnum+1); j++){
      free(neb_atom_coordinates[i][j]);
    }
    free(neb_atom_coordinates[i]);
  }
  free(neb_atom_coordinates);

  for (i=0; i<(NEB_Num_Images+2); i++){
    for (j=0; j<(atomnum+1); j++){
      free(tmp_neb_atom_coordinates[i][j]);
    }
    free(tmp_neb_atom_coordinates[i]);
  }
  free(tmp_neb_atom_coordinates);

  for (m=0; m<(M_GDIIS_HISTORY+2); m++){
    for (i=0; i<(NEB_Num_Images+2); i++){
      for (j=0; j<(atomnum+1); j++){
	free(His_neb_atom_coordinates[m][i][j]);
      }
      free(His_neb_atom_coordinates[m][i]);
    }
    free(His_neb_atom_coordinates[m]);
  }
  free(His_neb_atom_coordinates);

  for (i=0; i<(NEB_Num_Images+2); i++){
    free(All_Grid_Origin[i]);
  }  
  free(All_Grid_Origin);

  for (i=0; i<(NEB_Num_Images+2); i++){
    free(Tmp_Grid_Origin[i]);
  }  
  free(Tmp_Grid_Origin);

  free(WhatSpecies_NEB);
  free(Spe_WhatAtom_NEB);

  for (i=0; i<SpeciesNum; i++){
    free(SpeName_NEB[i]);
  }  
  free(SpeName_NEB);

  for (p1=0; p1<(NEB_Num_Images+2); p1++){
    for (i1=0; i1<(atomnum+1); i1++){
      for (j1=0; j1<4; j1++){
        for (p2=0; p2<(NEB_Num_Images+2); p2++){
          for (i2=0; i2<(atomnum+1); i2++){
            free(InvHess[p1][i1][j1][p2][i2]);
	  }
          free(InvHess[p1][i1][j1][p2]);
	}
        free(InvHess[p1][i1][j1]);
      }
      free(InvHess[p1][i1]);
    }
    free(InvHess[p1]);
  }  
  free(InvHess);

  for (p1=0; p1<(NEB_Num_Images+2); p1++){
    for (i1=0; i1<(atomnum+1); i1++){
      free(Vec0[p1][i1]);
    }
    free(Vec0[p1]);
  }
  free(Vec0);

  for(i=0; i<=atomnum; i++){
    free(atomFixedXYZ[i]);
  }
  free(atomFixedXYZ);
}



void Estimate_Initial_Hessian()
{
  int numprocs,myid,ID;
  int p1,i1,j1,p2,i2,j2,p,i;
  double **Hessian,**invHessian;

  /* MPI communication */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*******************************************
              initialize InvHess
  *******************************************/

  for (p1=1; p1<=NEB_Num_Images; p1++){
    for (i1=1; i1<=atomnum; i1++){
      for (j1=1; j1<=3; j1++){
	for (p2=1; p2<=NEB_Num_Images; p2++){
	  for (i2=1; i2<=atomnum; i2++){
	    for (j2=1; j2<=3; j2++){
	      InvHess[p1][i1][j1][p2][i2][j2] = 0.0;
	    }
	  }
	}
	InvHess[p1][i1][j1][p1][i1][j1] = 0.0;
      }
    }
  }
	
  /*******************************************
            allocation of arrays
  *******************************************/

  Hessian = (double**)malloc(sizeof(double*)*(3*atomnum+2));
  for (i=0; i<(3*atomnum+2); i++){
    Hessian[i] = (double*)malloc(sizeof(double)*(3*atomnum+2));
  }

  invHessian = (double**)malloc(sizeof(double*)*(3*atomnum+2));
  for (i=0; i<(3*atomnum+2); i++){
    invHessian[i] = (double*)malloc(sizeof(double)*(3*atomnum+2));
  }

  /*******************************************
           set Initial_Hessian_flag
  *******************************************/

  Initial_Hessian_flag = 1;

  /*******************************************
                 loop for p
  *******************************************/

  for (p=0; p<=(NEB_Num_Images+1); p++){ 

    /*******************************************
               the identity matrix 
    *******************************************/

    if (Initial_Hessian_flag==0){

      int i,j;

      for (i=0; i<3*atomnum; i++){     
	for (j=0; j<3*atomnum; j++){     
	  invHessian[i][j] = 0.0;
	}
	invHessian[i][i] = 1.0;
      }
    }

    /**************************************************************
      A model Hessian proposed by H.B. Schlegel

      Refs.
      H.B. Schlegel, Theo. Chim. Acta (Berl.) 66, 333 (1984).
      J.M.Wittbrodt and H.B. Schlegel, 
      J. Mol. Str. (Theochem) 398-399, 55 (1997).
    **************************************************************/

    else if (Initial_Hessian_flag==1){

      int i,j,k,I,J;
      int Mc_AN,Gc_AN,h_AN,Gh_AN,Rn;
      int wsp1,wsp2,m1,m2,n1,n2;
      double r,g[4],gr,d;
      double *Hess_tmp;
      double B[8][8];

      B[0][0] =  0.5000; B[0][1] =  0.5000; B[0][2] =  0.5000; B[0][3] =  0.5000; B[0][4] =  0.5000; B[0][5] =  0.5000; B[0][6] =  0.5000; B[0][7] =  0.5000;
      B[1][0] =  0.5000; B[1][1] = -0.2573; B[1][2] =  0.3401; B[1][3] =  0.6937; B[1][4] =  0.7126; B[1][5] =  0.8335; B[1][6] =  0.9491; B[1][7] =  1.0000;
      B[2][0] =  0.5000; B[2][1] =  0.3401; B[2][2] =  0.9652; B[2][3] =  1.2843; B[2][4] =  1.4625; B[2][5] =  1.6549; B[2][6] =  1.7190; B[2][7] =  2.0000;
      B[3][0] =  0.5000; B[3][1] =  0.6937; B[3][2] =  1.2843; B[3][3] =  1.6925; B[3][4] =  1.8238; B[3][5] =  2.1164; B[3][6] =  2.3185; B[3][7] =  2.5000;
      B[4][0] =  0.5000; B[4][1] =  0.7126; B[4][2] =  1.4625; B[4][3] =  1.8238; B[4][4] =  2.0203; B[4][5] =  2.2137; B[4][6] =  2.5206; B[4][7] =  2.7000;
      B[5][0] =  0.5000; B[5][1] =  0.8335; B[5][2] =  1.6549; B[5][3] =  2.1164; B[5][4] =  2.2137; B[5][5] =  2.3718; B[5][6] =  2.5110; B[5][7] =  2.7000;
      B[6][0] =  0.5000; B[6][1] =  0.9491; B[6][2] =  1.7190; B[6][3] =  2.3185; B[6][4] =  2.5206; B[6][5] =  2.5110; B[6][6] =  2.5200; B[6][7] =  2.7000;
      B[7][0] =  0.5000; B[7][1] =  1.0000; B[7][2] =  2.0000; B[7][3] =  2.5000; B[7][4] =  2.7000; B[7][5] =  2.7000; B[7][6] =  2.7000; B[7][7] =  2.9000;

      /* initialize Hessian */

      for (i=0; i<3*atomnum; i++){     
	for (j=0; j<3*atomnum; j++){     
	  Hessian[i][j] = 0.0;
	}
	Hessian[i][i] = 0.02;
      }

      /* calculate the approximate Hessian */

      for (Gc_AN=1; Gc_AN<=Matomnum; Gc_AN++){

	wsp1 = WhatSpecies[Gc_AN];
	m1 = Spe_WhatAtom[wsp1];
	if      (m1==1)   n1 = 1;
	else if (m1<=10)  n1 = 2;
	else if (m1<=18)  n1 = 3;
	else if (m1<=36)  n1 = 4;
	else if (m1<=54)  n1 = 5;
	else if (m1<=86)  n1 = 6;
	else if (m1<=103) n1 = 7;
	if (m1==2 || m1==10 || m1==18 || m1==36 || m1==54 || m1==86) n1 = 0;

	for (h_AN=1; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  wsp2 = WhatSpecies[Gh_AN];
	  m2 = Spe_WhatAtom[wsp2];

	  if      (m2==1)   n2 = 1;
	  else if (m2<=10)  n2 = 2;
	  else if (m2<=18)  n2 = 3;
	  else if (m2<=36)  n2 = 4;
	  else if (m2<=54)  n2 = 5;
	  else if (m2<=86)  n2 = 6;
	  else if (m2<=103) n2 = 7;
	  if (m2==2 || m2==10 || m2==18 || m2==36 || m2==54 || m2==86) n2 = 0;

	  Rn = ncn[Gc_AN][h_AN];
	  r = Dis[Gc_AN][h_AN];

	  g[1] = (Gxyz[Gc_AN][1] - Gxyz[Gh_AN][1] - atv[Rn][1])/r;
	  g[2] = (Gxyz[Gc_AN][2] - Gxyz[Gh_AN][2] - atv[Rn][2])/r;
	  g[3] = (Gxyz[Gc_AN][3] - Gxyz[Gh_AN][3] - atv[Rn][3])/r;

	  d = r - B[n1][n2];
	  gr = 1.734/d/d/d;

	  /* diagonal terms */
   
	  for (i=1; i<=3; i++){
	    for (j=1; j<=3; j++){
	      Hessian[(Gc_AN-1)*3+(i-1)][(Gc_AN-1)*3+(j-1)] += g[i]*g[j]*gr;
	    }
	  }
      
	  /* off-diagonal terms */

	  for (i=1; i<=3; i++){
	    for (j=1; j<=3; j++){
	      Hessian[(Gc_AN-1)*3+(i-1)][(Gh_AN-1)*3+(j-1)] += -g[i]*g[j]*gr;
	    }
	  }

	} /* h_AN */
      } /* Gc_AN */

      /* calculate the inverse Hessian */
           
      Inverse(3*atomnum-1,Hessian,invHessian);

    } /* end of else if (Initial_Hessian_flag==1) */

    /* store inverse Hessian to InvHess */

    for (i1=1; i1<=atomnum; i1++){
      for (j1=1; j1<=3; j1++){
	for (i2=1; i2<=atomnum; i2++){
	  for (j2=1; j2<=3; j2++){
	    InvHess[p][i1][j1][p][i2][j2] = invHessian[(i1-1)*3+(j1-1)][(i2-1)*3+(j2-1)]; 
	  }
	}
      }
    }

  } /* p */

  /*******************************************
                freeing of arrays
  *******************************************/

  for (i=0; i<(3*atomnum+2); i++){
    free(Hessian[i]);
  }
  free(Hessian);

  for (i=0; i<(3*atomnum+2); i++){
    free(invHessian[i]);
  }
  free(invHessian);

}



void Inverse(int n, double **a, double **ia)
{
  int method_flag=2;

  if (method_flag==0){

  /****************************************************
                  LU decomposition
                      0 to n
   NOTE:
   This routine does not consider the reduction of rank
  ****************************************************/

  int i,j,k;
  double w;
  double *x,*y;
  double **da;

  /***************************************************
    allocation of arrays: 

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  x = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  y = (double*)malloc(sizeof(double)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    x[i] = 0.0;
    y[i] = 0.0;
  }

  da = (double**)malloc(sizeof(double*)*List_YOUSO[39]);
  for (i=0; i<List_YOUSO[39]; i++){
    da[i] = (double*)malloc(sizeof(double)*List_YOUSO[39]);
    for (j=0; j<List_YOUSO[39]; j++){
      da[i][j] = 0.0;
    }
  }

  /* start calc. */

  if (n==-1){
    for (i=0; i<List_YOUSO[39]; i++){
      for (j=0; j<List_YOUSO[39]; j++){
	a[i][j] = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	da[i][j] = a[i][j];
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/a[k][k];
      for (i=k+1; i<=n; i++){
	a[i][k] = w*a[i][k];
	for (j=k+1; j<=n; j++){
	  a[i][j] = a[i][j] - a[i][k]*a[k][j];
	}
      }
    }

    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - a[i][j]*y[j];
	}
      }

      /****************************************************
                             Ux = y 
      ****************************************************/

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - a[i][j]*x[j];
	}
	x[i] = x[i]/a[i][i];
	ia[i][k] = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	a[i][j] = da[i][j];
      }
    }
  }

  /***************************************************
    freeing of arrays: 

     x[List_YOUSO[39]]
     y[List_YOUSO[39]]
     da[List_YOUSO[39]][List_YOUSO[39]]
  ***************************************************/

  free(x);
  free(y);

  for (i=0; i<List_YOUSO[39]; i++){
    free(da[i]);
  }
  free(da);

  }
  
  else if (method_flag==1){

    int i,j,M,N,LDA,INFO;
    int *IPIV,LWORK;
    double *A,*WORK;

    A = (double*)malloc(sizeof(double)*(n+2)*(n+2));
    WORK = (double*)malloc(sizeof(double)*(n+2));
    IPIV = (int*)malloc(sizeof(int)*(n+2));

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        A[i*(n+1)+j] = a[i][j];
      }
    }

    M = n + 1;
    N = M;
    LDA = M;
    LWORK = M;

    F77_NAME(dgetrf,DGETRF)( &M, &N, A, &LDA, IPIV, &INFO);
    F77_NAME(dgetri,DGETRI)( &N, A, &LDA, IPIV, WORK, &LWORK, &INFO);

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        ia[i][j] = A[i*(n+1)+j];
      }
    }

    free(A);
    free(WORK);
    free(IPIV);
  }

  else if (method_flag==2){

    int N,i,j,k;
    double *A,*B,*ko;
    double sum;

    N = n + 1;

    A = (double*)malloc(sizeof(double)*(N+2)*(N+2));
    B = (double*)malloc(sizeof(double)*(N+2)*(N+2));
    ko = (double*)malloc(sizeof(double)*(N+2));

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        A[j*N+i] = a[i][j];
      }
    }

    Eigen_lapack3(A, ko, N, N); 

    for (i=0; i<N; i++){
      ko[i] = 1.0/(ko[i]+1.0e-13);
    } 

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        B[i*N+j] = A[i*N+j]*ko[i];
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        ia[i][j] = 0.0;
      }
    }

    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        sum = 0.0;
	for (k=0; k<N; k++){
	  sum += A[k*N+i]*B[k*N+j];
	}
        ia[i][j] = sum;
      }
    }

    free(A);
    free(B);
    free(ko);
  }
}
