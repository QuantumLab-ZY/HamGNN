/**********************************************************************
  OutData_Binary.c:

   OutData_Binary.c is a subrutine to output values of electron densities,
   potetianls, wave functions, and etc. on the grids in the format of
   Gaussian cube, and atomic cartesian coordinates. 
   However, the data are stored in a binary format, and the text format
   can be recovered by a post processing code "bin2txt.c".  

  Log of OutData_Binary.c:

     27/Dec/2012  Released by T. Ozaki
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define CUBE_EXTENSION ".cube.bin"

static void out_density();
static void out_Veff();
static void out_Vhart();
static void out_Vna();
static void out_Vxc();
static void out_grid();
static void out_atomxyz();
static void out_atomxsf();
static void out_coordinates_bulk();
static void out_Cluster_MO();
static void out_Cluster_NC_MO();
static void out_Bulk_MO();
static void out_OrbOpt(char *inputfile);
static void out_Partial_Charge_Density();
static void Set_Partial_Density_Grid(double *****CDM);
static void Print_CubeTitle(FILE *fp, int EigenValue_flag, double EigenValue);
static void Print_CubeData(FILE *fp, char fext[], double *data, double *data1,char *op);
static void Print_VectorData(FILE *fp, char fext[],
                             double *data0, double *data1,
                             double *data2, double *data3);

double OutData_Binary(char *inputfile)
{
  char operate[YOUSO10];
  int i,c;
  int numprocs,myid;
  char fname1[300];
  char fname2[300];
  FILE *fp1,*fp2;
  char buf[fp_bsize];          /* setvbuf */
  char buf1[fp_bsize];         /* setvbuf */
  char buf2[fp_bsize];         /* setvbuf */
  double Stime,Etime;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&Stime);

  if (myid==Host_ID && 0<level_stdout) printf("\noutputting data on grids to files...\n\n");

  out_atomxyz();

  if (1<=level_fileout){
    out_density(); 
    out_Veff();
    out_Vhart(); 
  }

  if (2<=level_fileout){
    out_grid();
    if (ProExpn_VNA==0) out_Vna();
    out_Vxc();
  }

  if (specified_system!=0) out_coordinates_bulk();

  if (MO_fileout==1 && specified_system==0){

    /* spin non-collinear */
    if (SpinP_switch==3)
      out_Cluster_NC_MO();
    /* spin collinear */
    else
      out_Cluster_MO();
  }
  else if (MO_fileout==1 && specified_system!=0){
    out_Bulk_MO();
  }

  if (Cnt_switch==1){
    out_OrbOpt(inputfile);    
  } 

  /*************************************************
     partial charge density for STM simulations 
     The routine, out_Partial_Charge_Density(), 
     has to be called after calling out_density(), 
     since an array Density_Grid is used in 
     out_Partial_Charge_Density().
  *************************************************/

  if ( cal_partial_charge ){
    out_Partial_Charge_Density();
  }

  /* divide-conquer, gdc, or Krylov */

  if (Dos_fileout && (Solver==5 || Solver==6 || Solver==8 || Solver==11) ) {  

    if (myid==Host_ID){

      /* merge files */

      sprintf(fname1,"%s%s.Dos.vec",filepath,filename);
      fp1 = fopen(fname1,"ab");

      if (fp1!=NULL){
	remove(fname1); 
	fclose(fp1); 
      }
  
      for (i=0; i<numprocs; i++){ 

	sprintf(fname1,"%s%s.Dos.vec",filepath,filename);
	fp1 = fopen(fname1,"ab");
	fseek(fp1,0,SEEK_END);
	sprintf(fname2,"%s%s.Dos.vec%i",filepath,filename,i);
	fp2 = fopen(fname2,"rb");

	if (fp2!=NULL){
	  for (c=getc(fp2); c!=EOF; c=getc(fp2))  putc(c,fp1); 
	  fclose(fp2); 
	}
	fclose(fp1); 
      }

      for (i=0; i<numprocs; i++){
	sprintf(fname2,"%s%s.Dos.vec%i",filepath,filename,i);
	remove(fname2);
      }

    }
  }

  /* elapsed time */
  dtime(&Etime);
  return Etime - Stime;
}


void out_density()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k;
  int MN;
  double x,y,z,vx,vy,vz;
  double phi,theta,sden,oden;
  double xmin,ymin,zmin,xmax,ymax,zmax;
  char fname[YOUSO10];
  char file1[YOUSO10] = ".tden";
  char file2[YOUSO10] = ".den0";
  char file3[YOUSO10] = ".den1";
  char file4[YOUSO10] = ".sden";
  char file9[YOUSO10] = ".nc.xsf";
  char file10[YOUSO10] = ".ncsden.xsf.bin";
  char file12[YOUSO10] = ".nco.xsf";
  char file11[YOUSO10] = ".dden";
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID,digit;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  strcat(file1,CUBE_EXTENSION);
  strcat(file2,CUBE_EXTENSION);
  strcat(file3,CUBE_EXTENSION);
  strcat(file4,CUBE_EXTENSION);
  strcat(file11,CUBE_EXTENSION);

  /****************************************************
       electron density - atomic charge density  
  ****************************************************/

  sprintf(fname,"%s%s%s_%0*i",filepath,filename,file11,digit,myid);

  if ((fp = fopen(fname,"wb")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);

    for (MN=0; MN<My_NumGridB_AB; MN++){
      ADensity_Grid_B[MN] = Density_Grid_B[0][MN] 
                           +Density_Grid_B[1][MN] 
 	                 -2.0*ADensity_Grid_B[MN];
    }

    Print_CubeData(fp,file11,ADensity_Grid_B,(void*)NULL,(void*)NULL);
  }
  else{
    printf("Failure of saving the electron density\n");
  }

  /****************************************************
                  total electron density
  ****************************************************/

  sprintf(fname,"%s%s%s_%0*i",filepath,filename,file1,digit,myid);

  if ((fp = fopen(fname,"wb")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);

    if (SpinP_switch==0) {
      for (MN=0; MN<My_NumGridB_AB; MN++){
        Density_Grid_B[0][MN] = 2.0*Density_Grid_B[0][MN];
      }
      Print_CubeData(fp,file1,Density_Grid_B[0],(void*)NULL,(void*)NULL);
    }
    else {
      Print_CubeData(fp,file1,Density_Grid_B[0],Density_Grid_B[1],"add");
    }

  }
  else{
    printf("Failure of saving the electron density\n");
  }

  /* spin polization */

  if (SpinP_switch==1 || SpinP_switch==3){

    /****************************************************
                  up-spin electron density
    ****************************************************/

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file2,digit,myid);

    if ((fp = fopen(fname,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file2,Density_Grid_B[0],NULL,NULL);
    }
    else{
      printf("Failure of saving the electron density\n");
    }

    /****************************************************
                  down-spin electron density
    ****************************************************/

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file3,digit,myid);

    if ((fp = fopen(fname,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file3,Density_Grid_B[1],NULL,NULL);
    }
    else{
      printf("Failure of saving the electron density\n");
    }

    /****************************************************
                  spin electron density
    ****************************************************/

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file4,digit,myid);

    if ((fp = fopen(fname,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file4,Density_Grid_B[0],Density_Grid_B[1],"diff");
    }
    else{
      printf("Failure of saving the electron density\n");
    }
  }

  /****************************************************
        spin electron density with a spin vector
  ****************************************************/

  if (SpinP_switch==3){

    /* for XCrysDen, *.ncsden.xsf */

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file10,digit,myid);

    if ((fp = fopen(fname,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  
 
      Print_VectorData(fp,file10,Density_Grid_B[0],Density_Grid_B[1],
		       Density_Grid_B[2],Density_Grid_B[3]);

    }
    else{
      printf("Failure of saving the electron spin density with a vector\n");
    }

    /* non-collinear spin density by Mulliken population */

    if (myid==Host_ID){

      /* for XCrysDen, *.nc.xsf */

      sprintf(fname,"%s%s%s",filepath,filename,file9);
      if ((fp = fopen(fname,"w")) != NULL){

        setvbuf(fp,buf,_IOFBF,fp_bsize); 

        fprintf(fp,"CRYSTAL\n");
        fprintf(fp,"PRIMVEC\n");
        fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[1][1], BohrR*tv[1][2], BohrR*tv[1][3]);
        fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[2][1], BohrR*tv[2][2], BohrR*tv[2][3]);
        fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[3][1], BohrR*tv[3][2], BohrR*tv[3][3]);
        fprintf(fp,"PRIMCOORD\n");
        fprintf(fp,"%4d 1\n",atomnum);

        for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	  i = WhatSpecies[ct_AN];
          j = Spe_WhatAtom[i];

	  theta = Angle0_Spin[ct_AN];
	  phi   = Angle1_Spin[ct_AN];
	  sden = InitN_USpin[ct_AN] - InitN_DSpin[ct_AN];

	  vx = sden*sin(theta)*cos(phi);
	  vy = sden*sin(theta)*sin(phi);
	  vz = sden*cos(theta);
  
	  fprintf(fp,"%s %13.3E %13.3E %13.3E %13.3E %13.3E %13.3E\n",
  	          Atom_Symbol[j],
		  BohrR*Gxyz[ct_AN][1],
		  BohrR*Gxyz[ct_AN][2],
		  BohrR*Gxyz[ct_AN][3],
		  vx,vy,vz);
 	}

        fclose(fp);
      }
      else{
        printf("Failure of saving the Mulliken spin vector\n");
      }

      /* non-collinear obital magnetic moment by 'on-site' approximation */
      /* for XCrysDen  .xsf */

      sprintf(fname,"%s%s%s",filepath,filename,file12);
      if ((fp = fopen(fname,"w")) != NULL){

        setvbuf(fp,buf,_IOFBF,fp_bsize); 

        fprintf(fp,"CRYSTAL\n");
        fprintf(fp,"PRIMVEC\n");
        fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[1][1], BohrR*tv[1][2], BohrR*tv[1][3]);
        fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[2][1], BohrR*tv[2][2], BohrR*tv[2][3]);
        fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[3][1], BohrR*tv[3][2], BohrR*tv[3][3]);
        fprintf(fp,"PRIMCOORD\n");
        fprintf(fp,"%4d 1\n",atomnum);

        for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

	  i = WhatSpecies[ct_AN];
          j = Spe_WhatAtom[i];

          theta = Angle0_Orbital[ct_AN];
          phi   = Angle1_Orbital[ct_AN];
          oden  = OrbitalMoment[ct_AN];

          vx = oden*sin(theta)*cos(phi);
          vy = oden*sin(theta)*sin(phi);
          vz = oden*cos(theta);

	  fprintf(fp,"%s %13.3E %13.3E %13.3E %13.3E %13.3E %13.3E\n",
  	          Atom_Symbol[j],
		  BohrR*Gxyz[ct_AN][1],
		  BohrR*Gxyz[ct_AN][2],
		  BohrR*Gxyz[ct_AN][3],
		  vx,vy,vz);
 	}

        fclose(fp);
      }
      else{
        printf("Failure of saving orbital magnetic moments\n");
      }
  
    } /* if (myid==Host_ID) */
  } /* if (SpinP_switch==3) */

}



static void out_Vhart()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k,digit;
  char fname[YOUSO10];
  char file1[YOUSO10] = ".vhart";
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  strcat(file1,CUBE_EXTENSION);

  /****************************************************
                     Hartree potential
  ****************************************************/

  sprintf(fname,"%s%s%s_%0*i",filepath,filename,file1,digit,myid);
  if ((fp = fopen(fname,"wb")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
    Print_CubeData(fp,file1,dVHart_Grid_B,NULL,NULL);
  }
  else{
    printf("Failure of saving the electron density\n");
  }

}




static void out_Vna()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k;
  char fname[YOUSO10];
  char file1[YOUSO10] = ".vna";
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID,digit;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  strcat(file1,CUBE_EXTENSION);

  /****************************************************
                   neutral atom potential
  ****************************************************/

  sprintf(fname,"%s%s%s_%0*i",filepath,filename,file1,digit,myid);

  if ((fp = fopen(fname,"wb")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
    Print_CubeData(fp,file1,VNA_Grid_B,NULL,NULL);
  }
  else{
    printf("Failure of saving the electron density\n");
  }
}





static void out_Vxc()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k;
  int n,Ng1,Ng2,Ng3,spin,DN,BN;
  char fname[YOUSO10];
  char file1[YOUSO10] = ".vxc0";
  char file2[YOUSO10] = ".vxc1";
  FILE *fp;
  int numprocs,myid,ID,IDS,IDR,tag=999;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  strcat(file1,CUBE_EXTENSION);
  strcat(file2,CUBE_EXTENSION);

  /****************************************************
                       Vxc on grid
  ****************************************************/

  Set_XC_Grid(2,1,XC_switch,
              Density_Grid_D[0],Density_Grid_D[1],
              Density_Grid_D[2],Density_Grid_D[3],
              Vxc_Grid_D[0], Vxc_Grid_D[1],
	      Vxc_Grid_D[2], Vxc_Grid_D[3],
              NULL,NULL);

  /****************************************************
             copy Vxc_Grid_D to Vxc_Grid_B
  ****************************************************/

  Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
  Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
  Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

  for (n=0; n<Num_Rcv_Grid_B2D[myid]; n++){
    DN = Index_Rcv_Grid_B2D[myid][n];
    BN = Index_Snd_Grid_B2D[myid][n];

    i = DN/(Ng2*Ng3);
    j = (DN-i*Ng2*Ng3)/Ng3;
    k = DN - i*Ng2*Ng3 - j*Ng3; 

    if ( !(i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k)){
      for (spin=0; spin<=SpinP_switch; spin++){
        Vxc_Grid_B[spin][BN] = Vxc_Grid_D[spin][DN];
      }
    }
  }

  /****************************************************
       exchange-correlation potential for up-spin
  ****************************************************/

  sprintf(fname,"%s%s%s%i",filepath,filename,file1,myid);
  if ((fp = fopen(fname,"w")) != NULL){

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
    Print_CubeData(fp,file1,Vxc_Grid_B[0],NULL,NULL);
  }
  else{
    printf("Failure of saving the electron density\n");
  }

  /****************************************************
     exchange-correlation potential for down-spin
  ****************************************************/

  if (1<=SpinP_switch){

    sprintf(fname,"%s%s%s%i",filepath,filename,file2,myid);

    if ((fp = fopen(fname,"w")) != NULL){

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file2,Vxc_Grid_B[1],NULL,NULL);
    }
    else{
      printf("Failure of saving the electron density\n");
    }
  }

}




static void out_Veff()
{
  char fname[YOUSO10];
  char file1[YOUSO10] = ".v0";
  char file2[YOUSO10] = ".v1";
  FILE *fp;
  int numprocs,myid,digit;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  strcat(file1,CUBE_EXTENSION);
  strcat(file2,CUBE_EXTENSION);

  /****************************************************
            Kohn-Sham potential for up-spin
  ****************************************************/

  sprintf(fname,"%s%s%s_%0*i",filepath,filename,file1,digit,myid);
  if ((fp = fopen(fname,"w")) != NULL){

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
    Print_CubeData(fp,file1,Vpot_Grid_B[0],NULL,NULL);
  }
  else{
    printf("Failure of saving the electron density\n");
  }

  /****************************************************
           Kohn-Sham potential for down-spin
  ****************************************************/

  if (1<=SpinP_switch){

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file2,digit,myid);
    if ((fp = fopen(fname,"w")) != NULL){

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file2,Vpot_Grid_B[1],NULL,NULL);
    }
    else{
      printf("Failure of saving the electron density\n");
    }
  }
}




static void out_grid()
{
  int N,n1,n2,n3,i;
  char file1[YOUSO10] = ".grid.bin";
  int numprocs,myid,ID;
  double x,y,z;
  double *dlist;
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
     output the real space grids to a file, *.grid
  ****************************************************/

  if (myid==Host_ID){

    dlist = (double*)malloc(sizeof(double)*Ngrid2*Ngrid3*3);

    fnjoint(filepath,filename,file1);

    if ((fp = fopen(file1,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fwrite(&Ngrid1, sizeof(int), 1, fp);
      fwrite(&Ngrid2, sizeof(int), 1, fp);
      fwrite(&Ngrid3, sizeof(int), 1, fp);

      for (n1=0; n1<Ngrid1; n1++){

        i = 0;
        for (n2=0; n2<Ngrid2; n2++){
          for (n3=0; n3<Ngrid3; n3++){

	    x = (double)n1*gtv[1][1] + (double)n2*gtv[2][1]
	      + (double)n3*gtv[3][1] + Grid_Origin[1];
	    y = (double)n1*gtv[1][2] + (double)n2*gtv[2][2]
	      + (double)n3*gtv[3][2] + Grid_Origin[2];
	    z = (double)n1*gtv[1][3] + (double)n2*gtv[2][3]
	      + (double)n3*gtv[3][3] + Grid_Origin[3];

            dlist[i] = BohrR*x; i++;
            dlist[i] = BohrR*y; i++;
            dlist[i] = BohrR*z; i++;
          }
        }
        fwrite(dlist, sizeof(double), i, fp);
      }

      fclose(fp);
    }
    else{
      printf("Failure of saving grids\n");
    }

    free(dlist);
  }
}







static void out_atomxyz()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k;
  char filexyz[YOUSO10] = ".xyz";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
                cartesian coordinates
  ****************************************************/

  if (myid==Host_ID){

    fnjoint(filepath,filename,filexyz);
    if ((fp = fopen(filexyz,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp,"%i \n\n",atomnum);
      for (k=1; k<=atomnum; k++){
        i = WhatSpecies[k];
        j = Spe_WhatAtom[i];
        fprintf(fp,"%s   %8.5f  %8.5f  %8.5f  %18.15f  %18.15f  %18.15f\n",
	        Atom_Symbol[j],                
	        Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	        Gxyz[k][17],Gxyz[k][18],Gxyz[k][19]);
      }
      fclose(fp);
    }
    else{
      printf("could not save the xyz file\n");
    }
  }
}



static void out_atomxsf()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k;
  char filexsf[YOUSO10] = ".coord.xsf";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
                cartesian coordinates
  ****************************************************/

  if (myid==Host_ID){

    fnjoint(filepath,filename,filexsf);
    if ((fp = fopen(filexsf,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp,"CRYSTAL\n");
      fprintf(fp,"PRIMVEC\n");
      fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[1][1], BohrR*tv[1][2], BohrR*tv[1][3]);
      fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[2][1], BohrR*tv[2][2], BohrR*tv[2][3]);
      fprintf(fp," %10.6f %10.6f %10.6f\n",BohrR*tv[3][1], BohrR*tv[3][2], BohrR*tv[3][3]);
      fprintf(fp,"PRIMCOORD 1\n");
      fprintf(fp,"%4d %d\n",atomnum, 1);

      for (k=1; k<=atomnum; k++){
        i = WhatSpecies[k];
        /*j = Spe_WhatAtom[i];*/
        fprintf(fp,"%4d  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f\n",
	        Spe_WhatAtom[i],               
	        Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR,
	        Gxyz[k][24],Gxyz[k][25],Gxyz[k][26]);
      }
      fclose(fp);
    }
    else{
      printf("failure of saving coord.xsf file\n");
    }
  }
}



void out_coordinates_bulk()
{
  int n,i1,i2,i3,ct_AN,i,j;
  double tx,ty,tz,x,y,z;
  char file1[YOUSO10] = ".bulk.xyz";
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
        atomic coordinates including copied cells
  ****************************************************/

  if (myid==Host_ID){

    n = 1;

    fnjoint(filepath,filename,file1);

    if ((fp = fopen(file1,"w")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp,"%d\n\n",atomnum*(2*n+1)*(2*n+1)*(2*n+1));

      for (i1=-n; i1<=n; i1++){
        for (i2=-n; i2<=n; i2++){
          for (i3=-n; i3<=n; i3++){

            tx = (double)i1*tv[1][1] + (double)i2*tv[2][1] + (double)i3*tv[3][1];
            ty = (double)i1*tv[1][2] + (double)i2*tv[2][2] + (double)i3*tv[3][2];
            tz = (double)i1*tv[1][3] + (double)i2*tv[2][3] + (double)i3*tv[3][3];

            for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
              i = WhatSpecies[ct_AN];
              j = Spe_WhatAtom[i];

              x = BohrR*(Gxyz[ct_AN][1] + tx); 
              y = BohrR*(Gxyz[ct_AN][2] + ty); 
              z = BohrR*(Gxyz[ct_AN][3] + tz); 
              fprintf(fp,"%s %8.5f %8.5f %8.5f\n",Atom_Symbol[j],x,y,z);
            } 
          }
        }
      }

      fclose(fp);
    }
    else{
      printf("Failure of saving atomic coordinates\n");
    }
  }

}



void out_Cluster_MO()
{ 
  int Mc_AN,Gc_AN,Cwan,NO0,spin,Nc;
  int orbit,GN,spe,i,i1,i2,i3,so;
  double *MO_Grid_tmp;
  double *MO_Grid;
  char file1[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:

    double MO_Grid_tmp[TNumGrid];
    double MO_Grid[TNumGrid];
  ****************************************************/

  MO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  MO_Grid     = (double*)malloc(sizeof(double)*TNumGrid);

  /*************
      HOMOs
  *************/

  for (so=0; so<=SO_switch; so++){
    for (spin=0; spin<=SpinP_switch; spin++){
      for (orbit=0; orbit<num_HOMOs; orbit++){ 

	/* calc. MO on grids */

	for (GN=0; GN<TNumGrid; GN++) MO_Grid_tmp[GN] = 0.0;

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  NO0 = Spe_Total_CNO[Cwan];

          if (so==0){
  	    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	      GN = GridListAtom[Mc_AN][Nc];
	      for (i=0; i<NO0; i++){
	        MO_Grid_tmp[GN] += HOMOs_Coef[0][spin][orbit][Gc_AN][i].r*
		  Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	      }
	    }  
	  }

          else if (so==1){
  	    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	      GN = GridListAtom[Mc_AN][Nc];
	      for (i=0; i<NO0; i++){
	        MO_Grid_tmp[GN] += HOMOs_Coef[0][spin][orbit][Gc_AN][i].i*
		  Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	      }
	    }  
	  }

	}

	MPI_Reduce(&MO_Grid_tmp[0], &MO_Grid[0], TNumGrid, MPI_DOUBLE,
		   MPI_SUM, Host_ID, mpi_comm_level1);

	/* output HOMO on grids */

	if (myid==Host_ID){ 

          if (so==0)
  	    sprintf(file1,"%s%s.homo%i_%i_r%s",filepath,filename,spin,orbit,CUBE_EXTENSION);
          else if (so==1)
  	    sprintf(file1,"%s%s.homo%i_%i_i%s",filepath,filename,spin,orbit,CUBE_EXTENSION);

	  if ((fp = fopen(file1,"wb")) != NULL){

            setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	    Print_CubeTitle(fp,1,HOMOs_Coef[0][spin][orbit][0][0].r);
            fwrite(MO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	    fclose(fp);
	  }
	  else{
	    printf("Failure of saving MOs\n");
	  }

	}

      }  /* orbit */ 
    }  /* spin */ 
  }  /* so */

  /*************
      LUMOs
  *************/

  for (so=0; so<=SO_switch; so++){
    for (spin=0; spin<=SpinP_switch; spin++){
      for (orbit=0; orbit<num_LUMOs; orbit++){ 

	/* calc. MO on grids */

	for (GN=0; GN<TNumGrid; GN++) MO_Grid_tmp[GN] = 0.0;

	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  NO0 = Spe_Total_CNO[Cwan];

          if (so==0){
	    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	      GN = GridListAtom[Mc_AN][Nc];
	      for (i=0; i<NO0; i++){
		MO_Grid_tmp[GN] += LUMOs_Coef[0][spin][orbit][Gc_AN][i].r*
		  Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	      }
	    } 
	  }

          else if (so==1){
	    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	      GN = GridListAtom[Mc_AN][Nc];
	      for (i=0; i<NO0; i++){
		MO_Grid_tmp[GN] += LUMOs_Coef[0][spin][orbit][Gc_AN][i].i*
		  Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	      }
	    } 
	  }
 
	}

	/* output LUMO on grids */

	MPI_Reduce(&MO_Grid_tmp[0], &MO_Grid[0], TNumGrid, MPI_DOUBLE,
		   MPI_SUM, Host_ID, mpi_comm_level1);

	if (myid==Host_ID){ 

          if (so==0)
  	    sprintf(file1,"%s%s.lumo%i_%i_r%s",filepath,filename,spin,orbit,CUBE_EXTENSION);
          else if (so==0)
  	    sprintf(file1,"%s%s.lumo%i_%i_i%s",filepath,filename,spin,orbit,CUBE_EXTENSION);


	  if ((fp = fopen(file1,"wb")) != NULL){

            setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	    Print_CubeTitle(fp,1,LUMOs_Coef[0][spin][orbit][0][0].r);
            fwrite(MO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	    fclose(fp);
	  }
	  else{
	    printf("Failure of saving MOs\n");
	  }
	}

      }  /* orbit */ 
    }  /* spin */ 
  }  /* so */

  /****************************************************
    freeing of arrays:

    double MO_Grid[TNumGrid];
    double MO_Grid_tmp[TNumGrid];
  ****************************************************/

  free(MO_Grid);
  free(MO_Grid_tmp);
} 


void out_Cluster_NC_MO()
{ 
  int Mc_AN,Gc_AN,Cwan,NO0,spin,Nc;
  int orbit,GN,spe,i,i1,i2,i3;
  double *RMO_Grid_tmp,*IMO_Grid_tmp;
  double *RMO_Grid,*IMO_Grid;
  char file1[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:

    double RMO_Grid_tmp[TNumGrid];
    double IMO_Grid_tmp[TNumGrid];
    double RMO_Grid[TNumGrid];
    double IMO_Grid[TNumGrid];
  ****************************************************/

  RMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  RMO_Grid     = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid     = (double*)malloc(sizeof(double)*TNumGrid);

  /*************
      HOMOs
  *************/

  for (spin=0; spin<=1; spin++){
    for (orbit=0; orbit<num_HOMOs; orbit++){ 

      /* calc. MO on grids */

      for (GN=0; GN<TNumGrid; GN++){
	RMO_Grid_tmp[GN] = 0.0;
	IMO_Grid_tmp[GN] = 0.0;
      }

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	NO0 = Spe_Total_CNO[Cwan];
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	  GN = GridListAtom[Mc_AN][Nc];
	  for (i=0; i<NO0; i++){
 	    RMO_Grid_tmp[GN] += HOMOs_Coef[0][spin][orbit][Gc_AN][i].r*
	      Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	    IMO_Grid_tmp[GN] += HOMOs_Coef[0][spin][orbit][Gc_AN][i].i*
	      Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	  }
	}  
      }

      MPI_Reduce(&RMO_Grid_tmp[0], &RMO_Grid[0], TNumGrid, MPI_DOUBLE,
		 MPI_SUM, Host_ID, mpi_comm_level1);
      MPI_Reduce(&IMO_Grid_tmp[0], &IMO_Grid[0], TNumGrid, MPI_DOUBLE,
		 MPI_SUM, Host_ID, mpi_comm_level1);

      if (myid==Host_ID){ 

	/* output the real part of HOMOs on grids */

	sprintf(file1,"%s%s.homo%i_%i_r%s",
		filepath,filename,spin,orbit,CUBE_EXTENSION);

	if ((fp = fopen(file1,"wb")) != NULL){

          setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	  Print_CubeTitle(fp,1,HOMOs_Coef[0][spin][orbit][0][0].r);
          fwrite(RMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	  fclose(fp);
	}
	else{
	  printf("Failure of saving MOs\n");
	}

	/* output the imaginary part of HOMOs on grids */

	sprintf(file1,"%s%s.homo%i_%i_i%s", 
		filepath,filename,spin,orbit,CUBE_EXTENSION);

	if ((fp = fopen(file1,"wb")) != NULL){

          setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	  Print_CubeTitle(fp,1,HOMOs_Coef[0][spin][orbit][0][0].r);
          fwrite(IMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	  fclose(fp);
	}
	else{
	  printf("Failure of saving MOs\n");
	}

      }
    } /* orbit */ 
  } /* spin  */

  /*************
      LUMOs
  *************/

  for (spin=0; spin<=1; spin++){
    for (orbit=0; orbit<num_HOMOs; orbit++){ 

      /* calc. MO on grids */

      for (GN=0; GN<TNumGrid; GN++){
	RMO_Grid_tmp[GN] = 0.0;
	IMO_Grid_tmp[GN] = 0.0;
      }

      for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	Gc_AN = M2G[Mc_AN];    
	Cwan = WhatSpecies[Gc_AN];
	NO0 = Spe_Total_CNO[Cwan];
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	  GN = GridListAtom[Mc_AN][Nc];
	  for (i=0; i<NO0; i++){
	    RMO_Grid_tmp[GN] += LUMOs_Coef[0][spin][orbit][Gc_AN][i].r*
	      Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	    IMO_Grid_tmp[GN] += LUMOs_Coef[0][spin][orbit][Gc_AN][i].i*
	      Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	  }
	}  
      }

      MPI_Reduce(&RMO_Grid_tmp[0], &RMO_Grid[0], TNumGrid, MPI_DOUBLE,
		 MPI_SUM, Host_ID, mpi_comm_level1);
      MPI_Reduce(&IMO_Grid_tmp[0], &IMO_Grid[0], TNumGrid, MPI_DOUBLE,
		 MPI_SUM, Host_ID, mpi_comm_level1);

      if (myid==Host_ID){ 

	/* output the real part of LUMOs on grids */

	sprintf(file1,"%s%s.lumo%i_%i_r%s",
		filepath,filename,spin,orbit,CUBE_EXTENSION);

	if ((fp = fopen(file1,"wb")) != NULL){

          setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	  Print_CubeTitle(fp,1,LUMOs_Coef[0][spin][orbit][0][0].r);
          fwrite(RMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	  fclose(fp);
	}
	else{
	  printf("Failure of saving MOs\n");
	}

	/* output the imaginary part of HOMOs on grids */

	sprintf(file1,"%s%s.lumo%i_%i_i%s", 
		filepath,filename,spin,orbit,CUBE_EXTENSION);

	if ((fp = fopen(file1,"w")) != NULL){

          setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

	  Print_CubeTitle(fp,1,LUMOs_Coef[0][spin][orbit][0][0].r);
          fwrite(IMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	  fclose(fp);
	}
	else{
	  printf("Failure of saving MOs\n");
	}

      }
    }  /* orbit */ 
  }  /* spin  */

  /****************************************************
    freeing of arrays:

    double RMO_Grid_tmp[TNumGrid];
    double IMO_Grid_tmp[TNumGrid];
    double RMO_Grid[TNumGrid];
    double IMO_Grid[TNumGrid];
  ****************************************************/

  free(RMO_Grid_tmp);
  free(IMO_Grid_tmp);
  free(RMO_Grid);
  free(IMO_Grid);
} 




void out_Bulk_MO()
{ 
  int Mc_AN,Gc_AN,Cwan,NO0,spin,Nc;
  int kloop,Mh_AN,h_AN,Gh_AN,Rnh,Hwan;
  int NO1,l1,l2,l3,Nog,RnG,Nh,Rn;
  int orbit,GN,spe,i,j,i1,i2,i3,spinmax;
  double co,si,k1,k2,k3,kRn,ReCoef,ImCoef;
  double *RMO_Grid;
  double *IMO_Grid;
  double *RMO_Grid_tmp;
  double *IMO_Grid_tmp;
  char file1[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  FILE *fp;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:

    double RMO_Grid[TNumGrid];
    double IMO_Grid[TNumGrid];
    double RMO_Grid_tmp[TNumGrid];
    double IMO_Grid_tmp[TNumGrid];
  ****************************************************/

  RMO_Grid = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid = (double*)malloc(sizeof(double)*TNumGrid);
  RMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);
  IMO_Grid_tmp = (double*)malloc(sizeof(double)*TNumGrid);

  if      (SpinP_switch==0) spinmax = 0;
  else if (SpinP_switch==1) spinmax = 1;
  else if (SpinP_switch==3) spinmax = 1;

  /*************
       HOMOs
  *************/

  for (kloop=0; kloop<MO_Nkpoint; kloop++){

    k1 = MO_kpoint[kloop][1];
    k2 = MO_kpoint[kloop][2];
    k3 = MO_kpoint[kloop][3];

    for (spin=0; spin<=spinmax; spin++){
      for (orbit=0; orbit<Bulk_Num_HOMOs[kloop]; orbit++){ 

	/* calc. MO on grids */

	for (GN=0; GN<TNumGrid; GN++){
          RMO_Grid_tmp[GN] = 0.0;
          IMO_Grid_tmp[GN] = 0.0;
	}

        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
          Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  NO0 = Spe_Total_CNO[Cwan];

          for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
            GN = GridListAtom[Mc_AN][Nc];
            Rn = CellListAtom[Mc_AN][Nc];

            l1 =-atv_ijk[Rn][1];
            l2 =-atv_ijk[Rn][2];
            l3 =-atv_ijk[Rn][3];

            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

	    for (i=0; i<NO0; i++){
              ReCoef = co*HOMOs_Coef[kloop][spin][orbit][Gc_AN][i].r 
	 	      -si*HOMOs_Coef[kloop][spin][orbit][Gc_AN][i].i; 
              ImCoef = co*HOMOs_Coef[kloop][spin][orbit][Gc_AN][i].i
		      +si*HOMOs_Coef[kloop][spin][orbit][Gc_AN][i].r;
              RMO_Grid_tmp[GN] += ReCoef*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
              IMO_Grid_tmp[GN] += ImCoef*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	    }
          }
	}

        MPI_Reduce(&RMO_Grid_tmp[0], &RMO_Grid[0], TNumGrid, MPI_DOUBLE,
		   MPI_SUM, Host_ID, mpi_comm_level1);
        MPI_Reduce(&IMO_Grid_tmp[0], &IMO_Grid[0], TNumGrid, MPI_DOUBLE,
		   MPI_SUM, Host_ID, mpi_comm_level1);

        if (myid==Host_ID){ 

  	  /* output the real part of HOMOs on grids */

	  sprintf(file1,"%s%s.homo%i_%i_%i_r%s",
                  filepath,filename,kloop,spin,orbit,CUBE_EXTENSION);
  
  	  if ((fp = fopen(file1,"wb")) != NULL){

            setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

            Print_CubeTitle(fp,1,HOMOs_Coef[kloop][spin][orbit][0][0].r);
            fwrite(RMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	    fclose(fp);
	  }
	  else{
	    printf("Failure of saving MOs\n");
	  }

  	  /* output the imaginary part of HOMOs on grids */

	  sprintf(file1,"%s%s.homo%i_%i_%i_i%s", 
                  filepath,filename,kloop,spin,orbit,CUBE_EXTENSION);

	  if ((fp = fopen(file1,"wb")) != NULL){

            setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

            Print_CubeTitle(fp,1,HOMOs_Coef[kloop][spin][orbit][0][0].r);
            fwrite(IMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	    fclose(fp);
	  }
  	  else{
	    printf("Failure of saving MOs\n");
  	  }
	}

        /* MPI_Barrier */
        MPI_Barrier(mpi_comm_level1);

      } /* orbit */
    } /* spin */ 
  } /* kloop */

  /*************
       LUMOs
  *************/

  for (kloop=0; kloop<MO_Nkpoint; kloop++){

    k1 = MO_kpoint[kloop][1];
    k2 = MO_kpoint[kloop][2];
    k3 = MO_kpoint[kloop][3];

    for (spin=0; spin<=spinmax; spin++){
      for (orbit=0; orbit<Bulk_Num_LUMOs[kloop]; orbit++){

	/* calc. MO on grids */

	for (GN=0; GN<TNumGrid; GN++){
          RMO_Grid_tmp[GN] = 0.0;
          IMO_Grid_tmp[GN] = 0.0;
	}

        for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
          Gc_AN = M2G[Mc_AN];    
	  Cwan = WhatSpecies[Gc_AN];
	  NO0 = Spe_Total_CNO[Cwan];

          for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
            GN = GridListAtom[Mc_AN][Nc];
            Rn = CellListAtom[Mc_AN][Nc];

            l1 =-atv_ijk[Rn][1];
            l2 =-atv_ijk[Rn][2];
            l3 =-atv_ijk[Rn][3];

            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

	    for (i=0; i<NO0; i++){
              ReCoef = co*LUMOs_Coef[kloop][spin][orbit][Gc_AN][i].r 
		      -si*LUMOs_Coef[kloop][spin][orbit][Gc_AN][i].i; 
              ImCoef = co*LUMOs_Coef[kloop][spin][orbit][Gc_AN][i].i
		      +si*LUMOs_Coef[kloop][spin][orbit][Gc_AN][i].r;
              RMO_Grid_tmp[GN] += ReCoef*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
              IMO_Grid_tmp[GN] += ImCoef*Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	    }
          }
	}

        MPI_Reduce(&RMO_Grid_tmp[0], &RMO_Grid[0], TNumGrid, MPI_DOUBLE,
		   MPI_SUM, Host_ID, mpi_comm_level1);
        MPI_Reduce(&IMO_Grid_tmp[0], &IMO_Grid[0], TNumGrid, MPI_DOUBLE,
		   MPI_SUM, Host_ID, mpi_comm_level1);

	/* output the real part of LUMOs on grids */

        if (myid==Host_ID){ 

  	  sprintf(file1,"%s%s.lumo%i_%i_%i_r%s",
                  filepath,filename,kloop,spin,orbit,CUBE_EXTENSION);

  	  if ((fp = fopen(file1,"wb")) != NULL){

            setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

            Print_CubeTitle(fp,1,LUMOs_Coef[kloop][spin][orbit][0][0].r);
            fwrite(RMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	    fclose(fp);
	  }
	  else{
	    printf("Failure of saving MOs\n");
	    fclose(fp);
	  }

	  /* output the imaginary part of LUMOs on grids */

	  sprintf(file1,"%s%s.lumo%i_%i_%i_i%s",
                  filepath,filename,kloop,spin,orbit,CUBE_EXTENSION);

  	  if ((fp = fopen(file1,"wb")) != NULL){

            setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

            Print_CubeTitle(fp,1,LUMOs_Coef[kloop][spin][orbit][0][0].r);
            fwrite(IMO_Grid, sizeof(double), Ngrid1*Ngrid2*Ngrid3, fp);
	    fclose(fp);
	  }
	  else{
	    printf("Failure of saving MOs\n");
 	  }
	}

        /* MPI_Barrier */
        MPI_Barrier(mpi_comm_level1);

      } /* orbit */
    }   /* spin  */
  }     /* kloop */


  /****************************************************
    allocation of arrays:

    double RMO_Grid[TNumGrid];
    double IMO_Grid[TNumGrid];
    double RMO_Grid_tmp[TNumGrid];
    double IMO_Grid_tmp[TNumGrid];
  ****************************************************/

  free(RMO_Grid);
  free(IMO_Grid);
  free(RMO_Grid_tmp);
  free(IMO_Grid_tmp);
} 




static void Print_CubeTitle(FILE *fp, int EigenValue_flag, double EigenValue)
{
  int ct_AN,n;
  int spe; 
  char clist[300];
  double *dlist;

  if (EigenValue_flag==0){
    sprintf(clist," SYS1\n"); 
    fwrite(clist, sizeof(char), 200, fp);
    sprintf(clist," SYS1\n"); 
    fwrite(clist, sizeof(char), 200, fp);
  }
  else {
    sprintf(clist," Absolute eigenvalue=%10.7f (Hartree)  Relative eigenvalue=%10.7f (Hartree)\n",
            EigenValue,EigenValue-ChemP);
    fwrite(clist, sizeof(char), 200, fp);

    sprintf(clist, " Chemical Potential=%10.7f (Hartree)\n",ChemP);
    fwrite(clist, sizeof(char), 200, fp);
  }

  fwrite(&atomnum, sizeof(int), 1, fp);
  fwrite(&Grid_Origin[1], sizeof(double), 3, fp);
  fwrite(&Ngrid1, sizeof(int), 1, fp);
  fwrite(&gtv[1][1], sizeof(double), 3, fp);
  fwrite(&Ngrid2, sizeof(int), 1, fp);
  fwrite(&gtv[2][1], sizeof(double), 3, fp);
  fwrite(&Ngrid3, sizeof(int), 1, fp);
  fwrite(&gtv[3][1], sizeof(double), 3, fp);

  dlist = (double*)malloc(sizeof(double)*atomnum*5);

  n = 0;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    spe = WhatSpecies[ct_AN];
    dlist[n] = (double)Spe_WhatAtom[spe];                                  n++;
    dlist[n] = Spe_Core_Charge[spe]-InitN_USpin[ct_AN]-InitN_DSpin[ct_AN]; n++;
    dlist[n] = Gxyz[ct_AN][1];                                             n++;
    dlist[n] = Gxyz[ct_AN][2];                                             n++;
    dlist[n] = Gxyz[ct_AN][3];                                             n++;
  }

  fwrite(dlist, sizeof(double), atomnum*5, fp);
  free(dlist);
}




static void Print_CubeData(FILE *fp, char fext[], double *data0, double *data1, char *op)
{
  int myid,numprocs,ID,BN_AB,n3,cmd,digit,fd,c,po,num;
  char operate[300],operate1[300],operate2[300];
  FILE *fp1,*fp2;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  /****************************************************
                    output data 
  ****************************************************/

  if (op==NULL){
    fwrite(data0, sizeof(double), My_NumGridB_AB, fp);
  }
  else if (strcmp(op,"add")==0){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      data0[BN_AB] += data1[BN_AB]; 
    }

    fwrite(data0, sizeof(double), My_NumGridB_AB, fp);

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      data0[BN_AB] -= data1[BN_AB]; 
    }
  }

  else if (strcmp(op,"diff")==0){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      data0[BN_AB] -= data1[BN_AB]; 
    }

    fwrite(data0, sizeof(double), My_NumGridB_AB, fp);

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      data0[BN_AB] += data1[BN_AB]; 
    }

  }
  else {
    printf("Print_CubeData: op=%s not supported\n",op);
    return;
  }

  /****************************************************
                       fclose(fp);
  ****************************************************/

  fd = fileno(fp); 
  fsync(fd);
  fclose(fp);

  /****************************************************
                       merge files
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID){

    /* check whether all the files exists. */

    po = 0;
    num = 0;
    do {
      for (ID=0; ID<numprocs; ID++){
	sprintf(operate2,"%s%s%s_%0*i",filepath,filename,fext,digit,ID);
	fp2 = fopen(operate2,"r");
	if   (fp2==NULL) po = 1;
        else fclose(fp2);
      }

      num++;

    } while (po==1 && num<100000);

    /* check whether the file exists, and if it exists, remove it. */

    sprintf(operate1,"%s%s%s",filepath,filename,fext);
    fp1 = fopen(operate1, "rb");   
    if (fp1!=NULL){
      fclose(fp1); 
      remove(operate1);
    }

    /* merge all the fraction files */

    for (ID=0; ID<numprocs; ID++){

      sprintf(operate1,"%s%s%s",filepath,filename,fext);
      fp1 = fopen(operate1, "ab");   
      fseek(fp1,0,SEEK_END);

      sprintf(operate2,"%s%s%s_%0*i",filepath,filename,fext,digit,ID);
      fp2 = fopen(operate2,"rb");
  
      if (fp2!=NULL){
	for (c=getc(fp2); c!=EOF; c=getc(fp2)) putc(c,fp1); 
	fclose(fp2); 
      }

      fclose(fp1); 
    }

    /* remove all the fraction files */

    sprintf(operate,"rm %s%s%s_*",filepath,filename,fext);
    system(operate);

    /*
    for (ID=0; ID<numprocs; ID++){
      sprintf(operate,"%s%s%s_%0*i",filepath,filename,fext,digit,ID);
      remove(operate);
    }
    */

  }

}


static void Print_VectorData(FILE *fp, char fext[],
                             double *data0, double *data1,
                             double *data2, double *data3)
{
  int i,j,k,i1,i2,i3,n,c,GridNum;
  int GN_AB,n1,n2,n3,interval,digit;
  int BN_AB,N2D,GNs,GN;
  double x,y,z,vx,vy,vz;
  double sden,Cxyz[4],theta,phi;
  double tv0[4][4];
  double *dlist;
  char operate[300];
  char operate1[300];
  char operate2[300];
  char clist[300];
  int numprocs,myid,ID;
  MPI_Status stat;
  MPI_Request request;
  FILE *fp1,*fp2;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  /****************************************************
                    output data 
  ****************************************************/

  /* for XCrysDen */ 

  interval = 4;

  if (myid==Host_ID){

    sprintf(clist,"CRYSTAL\n"); 
    fwrite(clist, sizeof(char), 200, fp);
    sprintf(clist,"PRIMVEC\n"); 
    fwrite(clist, sizeof(char), 200, fp);

    for (i=1; i<=3; i++){
      for (j=1; j<=3; j++){
        tv0[i][j] = BohrR*tv[i][j];
      }
    }

    fwrite(&tv0[1][1], sizeof(double), 3, fp);
    fwrite(&tv0[2][1], sizeof(double), 3, fp);
    fwrite(&tv0[3][1], sizeof(double), 3, fp);

    GridNum = 0;
    for (n1=0; n1<Ngrid1; n1++){
      for (n2=0; n2<Ngrid2; n2++){
	for (n3=0; n3<Ngrid3; n3++){
	  if (n1%interval==0 && n2%interval==0 && n3%interval==0) GridNum++;
	}
      }
    }

    fwrite(&atomnum,sizeof(int),1, fp);
    fwrite(&GridNum,sizeof(int),1, fp);

    for (k=1; k<=atomnum; k++){
      i = WhatSpecies[k];
      j = Spe_WhatAtom[i];
      
      fwrite(Atom_Symbol[j], sizeof(char), 4, fp);

      Cxyz[1] = Gxyz[k][1]*BohrR;
      Cxyz[2] = Gxyz[k][2]*BohrR;
      Cxyz[3] = Gxyz[k][3]*BohrR;
 
      fwrite(&Cxyz[1], sizeof(double), 3, fp);
    }
  }

  /****************************************************
                 fwrite vector data
  ****************************************************/

  dlist = (double*)malloc(sizeof(double)*My_NumGridB_AB*6);

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  n = 0;
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
   
    GN_AB = BN_AB + GNs;
    n1 = GN_AB/(Ngrid2*Ngrid3);
    n2 = (GN_AB - n1*(Ngrid2*Ngrid3))/Ngrid3;
    n3 = GN_AB - n1*(Ngrid2*Ngrid3) - n2*Ngrid3;

    if (n1%interval==0 && n2%interval==0 && n3%interval==0){

      GN = n1*Ngrid2*Ngrid3 + n2*Ngrid3 + n3; 

      Get_Grid_XYZ(GN,Cxyz);
      x = BohrR*Cxyz[1];
      y = BohrR*Cxyz[2];
      z = BohrR*Cxyz[3];

      sden  = data0[BN_AB] - data1[BN_AB];
      theta = data2[BN_AB];
      phi   = data3[BN_AB];

      vx = sden*sin(theta)*cos(phi);
      vy = sden*sin(theta)*sin(phi);
      vz = sden*cos(theta);

      dlist[n  ] = x;
      dlist[n+1] = y;
      dlist[n+2] = z;
      dlist[n+3] = vx;
      dlist[n+4] = vy;
      dlist[n+5] = vz;

      n += 6;
    }
  }

  fwrite(dlist, sizeof(double), n, fp);
  free(dlist);

  /****************************************************
  fclose(fp);
  ****************************************************/

  fclose(fp);

  /****************************************************
                   merge files
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID){

    /* check whether the file exists, and if it exists, remove it. */

    sprintf(operate1,"%s%s%s",filepath,filename,fext);
    fp1 = fopen(operate1, "rb");   
    if (fp1!=NULL){
      fclose(fp1); 
      remove(operate1);
    }

    /* merge all the fraction files */

    for (ID=0; ID<numprocs; ID++){

      sprintf(operate1,"%s%s%s",filepath,filename,fext);
      fp1 = fopen(operate1, "ab");   
      fseek(fp1,0,SEEK_END);

      sprintf(operate2,"%s%s%s_%0*i",filepath,filename,fext,digit,ID);
      fp2 = fopen(operate2,"rb");
  
      if (fp2!=NULL){
	for (c=getc(fp2); c!=EOF; c=getc(fp2)) putc(c,fp1); 
	fclose(fp2); 
      }

      fclose(fp1); 
    }

    /* remove all the fraction files */

    sprintf(operate,"rm %s%s%s_*",filepath,filename,fext);
    system(operate);

    /*
    for (ID=0; ID<numprocs; ID++){
      sprintf(operate,"%s%s%s_%0*i",filepath,filename,fext,digit,ID);
      remove(operate);
    }
    */

  }

}





void out_OrbOpt(char *inputfile)
{
  int i,j,natom,po,al,p,wan,L0,Mul0,M0;
  int num,Mc_AN,Gc_AN,pmax,Mul1;
  int al1,al0;
  double sum,Max0;
  double ***tmp_coes;
  char file1[YOUSO10] = ".oopt";
  char DirPAO[YOUSO10];
  char ExtPAO[YOUSO10] = ".pao";
  char FN_PAO[YOUSO10];
  char EndLine[YOUSO10] = "<pseudo.atomic.orbitals.L=0";
  char fname2[YOUSO10];
  char command0[YOUSO10];
  double tmp0,tmp1;
  double *Tmp_Vec,*Tmp_CntCoes;
  int ***tmp_index2;
  char *sp0;
  FILE *fp,*fp2;
  char *buf;          /* setvbuf */
  int numprocs,myid,ID,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* set DirPAO */

  sprintf(DirPAO,"%s/PAO/",DFT_DATA_PATH);

  /* allocate buf */
  buf = malloc(fp_bsize); /* setvbuf */

  /* allocation of Tmp_Vec */
  Tmp_Vec = (double*)malloc(sizeof(double)*List_YOUSO[7]*List_YOUSO[24]);

  fnjoint(filepath,filename,file1);

  if ((fp = fopen(file1,"w")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    Tmp_CntCoes = (double*)malloc(sizeof(double)*List_YOUSO[24]); 

    tmp_index2 = (int***)malloc(sizeof(int**)*(List_YOUSO[25]+1)); 
    for (i=0; i<(List_YOUSO[25]+1); i++){
      tmp_index2[i] = (int**)malloc(sizeof(int*)*List_YOUSO[24]); 
      for (j=0; j<List_YOUSO[24]; j++){
	tmp_index2[i][j] = (int*)malloc(sizeof(int)*(2*(List_YOUSO[25]+1)+1)); 
      }
    }

    /**********************************************
      transformation of optimized orbitals by 
      an extended Gauss elimination and 
      the Gram-Schmidt orthogonalization
    ***********************************************/

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];    
      wan = WhatSpecies[Gc_AN];

      al = -1;
      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	  for (M0=0; M0<=2*L0; M0++){
	    al++;
	    tmp_index2[L0][Mul0][M0] = al;
	  }
	}
      }
       
      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	for (M0=0; M0<=2*L0; M0++){

	  /**********************************************
                     extended Gauss elimination
	  ***********************************************/

	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    al0 = tmp_index2[L0][Mul0][M0]; 
	    for (Mul1=0; Mul1<Spe_Num_CBasis[wan][L0]; Mul1++){
	      al1 = tmp_index2[L0][Mul1][M0];

              if (Mul1!=Mul0){

                tmp0 = CntCoes[Mc_AN][al0][Mul0]; 
                tmp1 = CntCoes[Mc_AN][al1][Mul0]; 

  	        for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
                  CntCoes[Mc_AN][al1][p] -= CntCoes[Mc_AN][al0][p]/tmp0*tmp1;
		}
              }

	    }
	  }

	  /*
	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    al0 = tmp_index2[L0][Mul0][M0]; 
	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      printf("Mul0=%2d p=%2d Coes=%15.12f\n",Mul0,p,CntCoes[Mc_AN][al0][p]);
	    }
	  } 
	  */         

	  /**********************************************
             orthonormalization of optimized orbitals
	  ***********************************************/

	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    al0 = tmp_index2[L0][Mul0][M0]; 

	    /* x - sum_i <x|e_i>e_i */

	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      Tmp_CntCoes[p] = 0.0;
	    }
         
	    for (Mul1=0; Mul1<Mul0; Mul1++){
	      al1 = tmp_index2[L0][Mul1][M0];

	      sum = 0.0;
	      for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
		sum = sum + CntCoes[Mc_AN][al0][p]*CntCoes[Mc_AN][al1][p];
	      }

	      for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
		Tmp_CntCoes[p] = Tmp_CntCoes[p] + sum*CntCoes[Mc_AN][al1][p];
	      }
	    }

	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      CntCoes[Mc_AN][al0][p] = CntCoes[Mc_AN][al0][p] - Tmp_CntCoes[p];
	    }

	    /* Normalize */

	    sum = 0.0;
	    Max0 = -100.0;
	    pmax = 0;
	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      sum = sum + CntCoes[Mc_AN][al0][p]*CntCoes[Mc_AN][al0][p];
	      if (Max0<fabs(CntCoes[Mc_AN][al0][p])){
		Max0 = fabs(CntCoes[Mc_AN][al0][p]);
		pmax = p;
	      }
	    }

	    if (fabs(sum)<1.0e-11)
	      tmp0 = 0.0;
	    else 
	      tmp0 = 1.0/sqrt(sum); 

	    tmp1 = sgn(CntCoes[Mc_AN][al0][pmax]);
            
	    for (p=0; p<Spe_Specified_Num[wan][al0]; p++){
	      CntCoes[Mc_AN][al0][p] = tmp0*tmp1*CntCoes[Mc_AN][al0][p];
	    }

	  }
	}
      }

    } /* Mc_AN */

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
	free(tmp_index2[i][j]);
      }
      free(tmp_index2[i]);
    }
    free(tmp_index2);

    free(Tmp_CntCoes);

    /**********************************************
                    set Tmp_Vec
    ***********************************************/

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      wan = WhatSpecies[Gc_AN];
      ID = G2ID[Gc_AN];

      if (myid==ID){

        Mc_AN = F_G2M[Gc_AN];

        al = -1;
        num = 0;
        for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){
	      al++;
	      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	        Tmp_Vec[num] = CntCoes[Mc_AN][al][p];
                num++;
	      }
	    }
	  }
        }

        if (myid!=Host_ID){
          MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
          MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
        }
       
      }

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
        MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      }

      /**********************************************
                         write 
      ***********************************************/
      
      if (myid==Host_ID){

        fprintf(fp,"\nAtom=%2d\n",Gc_AN);
        fprintf(fp,"Basis specification  %s\n",SpeBasis[wan]);

        fprintf(fp,"Contraction coefficients  p=");
        for (i=0; i<List_YOUSO[24]; i++){
	  fprintf(fp,"       %i  ",i);
        }
        fprintf(fp,"\n");

        al = -1;
        num = 0;

        for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
  	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){
	      al++;

	      fprintf(fp,"Atom=%3d  L=%2d  Mul=%2d  M=%2d  ",Gc_AN,L0,Mul0,M0);
	      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	        fprintf(fp,"%9.5f ",Tmp_Vec[num]);
                num++;
	      }
	      for (p=Spe_Specified_Num[wan][al]; p<List_YOUSO[24]; p++){
	        fprintf(fp,"  0.00000 ");
	      }
	      fprintf(fp,"\n");
	    }
	  }
        }
      } /* if (myid==Host_ID) */
    }   /* Gc_AN */
    fclose(fp);
  }
  else{
    printf("Failure of saving contraction coefficients of basis orbitals\n");
  }

  /****************************************************
       outputting of contracted orbitals as *.pao   
  ****************************************************/

  if (CntOrb_fileout==1 && Cnt_switch==1 && RCnt_switch==1){

    /****************************************************
       allocation of array:

       tmp_coes[List_YOUSO[25]+1]
               [List_YOUSO[24]]
               [List_YOUSO[24]]
    ****************************************************/

    tmp_coes = (double***)malloc(sizeof(double**)*(List_YOUSO[25]+1));
    for (i=0; i<(List_YOUSO[25]+1); i++){
      tmp_coes[i] = (double**)malloc(sizeof(double*)*List_YOUSO[24]);
      for (j=0; j<List_YOUSO[24]; j++){
        tmp_coes[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }
    }

    for (natom=0; natom<Num_CntOrb_Atoms; natom++){

      Gc_AN = CntOrb_Atoms[natom];
      ID = G2ID[Gc_AN];
      wan = WhatSpecies[Gc_AN];

      if (myid==ID){

        Mc_AN = F_G2M[Gc_AN];

        al = -1;
        num = 0;
        for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	  for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	    for (M0=0; M0<=2*L0; M0++){
	      al++;
	      for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	        Tmp_Vec[num] = CntCoes[Mc_AN][al][p];
                num++;
	      }
	    }
	  }
        }

        if (myid!=Host_ID){
          MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
          MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
        }
       
      }

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
        MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      }

      /****************************************************
       generate a pao file
      ****************************************************/

      if (myid==Host_ID){

        /* 
          setting of initial vectors using contraction
             coefficients and unit vectors (1.0)
        */

        al = -1;
        num = 0;

        for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
          for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
            for (M0=0; M0<=2*L0; M0++){
              al++;
      
              if (M0==0){
                for (p=0; p<Spe_Specified_Num[wan][al]; p++){
	          tmp_coes[L0][Mul0][p] = Tmp_Vec[num];
                  num++; 
	        }
                for (p=Spe_Specified_Num[wan][al]; p<Spe_PAO_Mul[wan]; p++){
	          tmp_coes[L0][Mul0][p] = 0.0;
	        }
              }
              else{
                for (p=0; p<Spe_Specified_Num[wan][al]; p++){
                  num++; 
	        }
              } 
	    }
	  }

          for (Mul0=Spe_Num_CBasis[wan][L0]; Mul0<Spe_PAO_Mul[wan]; Mul0++){
            for (p=0; p<Spe_PAO_Mul[wan]; p++) tmp_coes[L0][Mul0][p] = 0.0;
            tmp_coes[L0][Mul0][Mul0] = 1.0;
	  }
        }

        /****************************************************
                            make *.pao
        ****************************************************/

        sprintf(fname2,"%s%s_%i%s",filepath,SpeName[wan],Gc_AN,ExtPAO);
        if ((fp2 = fopen(fname2,"w")) != NULL){

	  setvbuf(fp2,buf,_IOFBF,fp_bsize);  /* setvbuf */

  	  fnjoint2(DirPAO,SpeBasisName[wan],ExtPAO,FN_PAO); 

          /****************************************************
                      Log of the orbital optimization
                                  and 
                        contraction coefficients
          ****************************************************/

          fprintf(fp2,"*****************************************************\n");
          fprintf(fp2,"*****************************************************\n");
          fprintf(fp2," The numerical atomic orbitals were generated\n");
          fprintf(fp2," by the variational optimization.\n");
          fprintf(fp2," The original file of PAOs was %s.\n",FN_PAO);
          fprintf(fp2," Basis specification was %s.\n",SpeBasis[wan]);
          fprintf(fp2," The input file was %s.\n",inputfile);
          fprintf(fp2,"*****************************************************\n");
          fprintf(fp2,"*****************************************************\n\n");

  	  fprintf(fp2,"<Contraction.coefficients\n");

	  al = -1;
          num = 0;

  	  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
	      for (M0=0; M0<=2*L0; M0++){

	        al++;

                if (M0==0){

		  for (p=0; p<Spe_Specified_Num[wan][al]; p++){

  		    fprintf(fp2,"  Atom=%3d  L=%2d  Mul=%2d  p=%3d  %18.15f\n",
                            Gc_AN,L0,Mul0,p,Tmp_Vec[num]);

		    num++;
		  }
		  for (p=Spe_Specified_Num[wan][al]; p<List_YOUSO[24]; p++){

  		    fprintf(fp2,"  Atom=%3d  L=%2d  Mul=%2d  p=%3d  %18.15f\n",
                            Gc_AN,L0,Mul0,p,0.0);
		  }
		}
                else{
		  for (p=0; p<Spe_Specified_Num[wan][al]; p++){
		    num++;
		  }
                } 

	      }
	    }
	  }
  	  fprintf(fp2,"Contraction.coefficients>\n");

          fprintf(fp2,"\n*****************************************************\n");
          fprintf(fp2,"*****************************************************\n\n");

          /****************************************************
                        from the original file
          ****************************************************/

	  if ((fp = fopen(FN_PAO,"r")) != NULL){

	    po = 0;       
	    do{
	      if (fgets(command0,YOUSO10,fp)!=NULL){
	        command0[strlen(command0)-1] = '\0';
                sp0 = strstr(command0,EndLine);
	        if (sp0!=NULL){
		  po = 1;
	        }
	        else {
		  fprintf(fp2,"%s\n",command0); 
	        }
	      }
	      else{ 
	        po = 1;  
	      }
	    }while(po==0);
                 
	    fclose(fp);
	  }
  	  else{
	    printf("Could not find %s\n",FN_PAO);
	    exit(1);
	  }

          /****************************************************
                          contracted orbitals
          ****************************************************/

          for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
            fprintf(fp2,"<pseudo.atomic.orbitals.L=%d\n",L0);
            for (i=0; i<Spe_Num_Mesh_PAO[wan]; i++){
              fprintf(fp2,"%18.15f %18.15f ",Spe_PAO_XV[wan][i],Spe_PAO_RV[wan][i]); 
 	      for (Mul0=0; Mul0<Spe_PAO_Mul[wan]; Mul0++){
                sum = 0.0;
                for (p=0; p<Spe_PAO_Mul[wan]; p++){
                  sum += tmp_coes[L0][Mul0][p]*Spe_PAO_RWF[wan][L0][p][i];
	        }
                fprintf(fp2,"%18.15f ",sum);
              }
              fprintf(fp2,"\n");
            }
            fprintf(fp2,"pseudo.atomic.orbitals.L=%d>\n",L0);
	  }

          for (L0=(Spe_MaxL_Basis[wan]+1); L0<=Spe_PAO_LMAX[wan]; L0++){
            fprintf(fp2,"<pseudo.atomic.orbitals.L=%d\n",L0);
            for (i=0; i<Spe_Num_Mesh_PAO[wan]; i++){
              fprintf(fp2,"%18.15f %18.15f ",Spe_PAO_XV[wan][i],Spe_PAO_RV[wan][i]); 
 	      for (Mul0=0; Mul0<Spe_PAO_Mul[wan]; Mul0++){
                fprintf(fp2,"%18.15f ",Spe_PAO_RWF[wan][L0][Mul0][i]);
	      }
              fprintf(fp2,"\n");
	    }
            fprintf(fp2,"pseudo.atomic.orbitals.L=%d>\n",L0);
	  }

          fclose(fp2);

        }
        else{
          printf("Could not open %s\n",fname2);
          exit(0);
        }

      } /* if (myid==Host_ID) */
    }   /* for (natom=0; natom<Num_CntOrb_Atoms; natom++) */

    /****************************************************
       freeing of arrays:
    ****************************************************/

    for (i=0; i<(List_YOUSO[25]+1); i++){
      for (j=0; j<List_YOUSO[24]; j++){
        free(tmp_coes[i][j]);
      }
      free(tmp_coes[i]);
    }
    free(tmp_coes);

  }

  free(Tmp_Vec);
  free(buf);
}




static void Print_CubeData_MO(FILE *fp, double *data, double *data1,char *op)
{
  int i1,i2,i3;
  int GN;
  int cmd;

  if (op==NULL) { cmd=0; }
  else if (strcmp(op,"add")==0)  { cmd=1; }
  else if (strcmp(op,"diff")==0) { cmd=2; }
  else {
    printf("Print_CubeData: op=%s not supported\n",op);
    return;
  }

  for (i1=0; i1<Ngrid1; i1++){
    for (i2=0; i2<Ngrid2; i2++){
      for (i3=0; i3<Ngrid3; i3++){
	GN = i1*Ngrid2*Ngrid3 + i2*Ngrid3 + i3;
	switch (cmd) {
	case 0:
	  fprintf(fp,"%13.3E",data[GN]);
	  break;
	case 1:
	  fprintf(fp,"%13.3E",data[GN]+data1[GN]);
	  break;
	case 2:
	  fprintf(fp,"%13.3E",data[GN]-data1[GN]);
	  break;
	}
	if ((i3+1)%6==0) { fprintf(fp,"\n"); }
      }
      /* avoid double \n\n when Ngrid3%6 == 0  */
      if (Ngrid3%6!=0) fprintf(fp,"\n");
    }
  }
}






void out_Partial_Charge_Density()
{
  int ct_AN,spe,i1,i2,i3,GN,i,j,k;
  int MN,digit;
  double x,y,z,vx,vy,vz;
  double phi,theta,sden,oden;
  double xmin,ymin,zmin,xmax,ymax,zmax;
  char fname[YOUSO10];
  char file1[YOUSO10] = ".pden";
  char file2[YOUSO10] = ".pden0";
  char file3[YOUSO10] = ".pden1";
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  digit = (int)log10(numprocs) + 1;

  strcat(file1,CUBE_EXTENSION);
  strcat(file2,CUBE_EXTENSION);
  strcat(file3,CUBE_EXTENSION);

  /****************************************************
         calculation of partial charge density 
  ****************************************************/

  Set_Partial_Density_Grid(Partial_DM);

  /****************************************************
       partial electron density including both up 
       and down spin contributions
  ****************************************************/

  sprintf(fname,"%s%s%s_%0*i",filepath,filename,file1,digit,myid);

  if ((fp = fopen(fname,"wb")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);

    if (SpinP_switch==0) {
      for (MN=0; MN<My_NumGridB_AB; MN++){
        Density_Grid_B[0][MN] = 2.0*Density_Grid_B[0][MN];
      }
      Print_CubeData(fp,file1,Density_Grid_B[0],(void*)NULL,(void*)NULL);
    }
    else {
      Print_CubeData(fp,file1,Density_Grid_B[0],Density_Grid_B[1],"add");
    }

  }
  else{
    printf("Failure of saving total partial electron density\n");
  }

  /* spin polization */

  if (SpinP_switch==1){

    /****************************************************
                  up-spin electron density
    ****************************************************/

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file2,digit,myid);

    if ((fp = fopen(fname,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file2,Density_Grid_B[0],NULL,NULL);
    }
    else{
      printf("Failure of saving the electron density\n");
    }

    /****************************************************
                  down-spin electron density
    ****************************************************/

    sprintf(fname,"%s%s%s_%0*i",filepath,filename,file3,digit,myid);

    if ((fp = fopen(fname,"wb")) != NULL){

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (myid==Host_ID) Print_CubeTitle(fp,0,0.0);
      Print_CubeData(fp,file3,Density_Grid_B[1],NULL,NULL);
    }
    else{
      printf("Failure of saving the electron density\n");
    }
  }

}



void Set_Partial_Density_Grid(double *****CDM)
{
  int al,L0,Mul0,M0,p,size1,size2;
  int Gc_AN,Mc_AN,Mh_AN,MN;
  int n1,n2,n3,k1,k2,k3,N3[4];
  int Cwan,NO0,NO1,Rn,N,Hwan,i,j,k,n;
  int Max_Size,My_Max,top_num;
  double time0;
  int h_AN,Gh_AN,Rnh,spin,Nc,GRc,Nh,Nog;
  int Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3;
  unsigned long long int LN,BN,AN,n2D,N2D;
  unsigned long long int GNc,GN;
  double tmp0,tmp1,sk1,sk2,sk3,tot_den,sum;
  double tmp0_0,tmp0_1,tmp0_2,tmp0_3;
  double sum_0,sum_1,sum_2,sum_3;

  double d1,d2,d3,cop,sip,sit,cot;
  double Re11,Re22,Re12,Im12,phi,theta;
  double x,y,z,Cxyz[4];
  double TStime,TEtime;
  double ***Tmp_Den_Grid;
  double **Den_Snd_Grid_A2B;
  double **Den_Rcv_Grid_A2B; 
  double *orbs0,*orbs1;
  double *orbs0_0,*orbs0_1,*orbs0_2,*orbs0_3;
  double *orbs1_0,*orbs1_1,*orbs1_2,*orbs1_3;

  double ***tmp_CDM;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom, Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of arrays */

  Tmp_Den_Grid = (double***)malloc(sizeof(double**)*(SpinP_switch+1)); 
  for (i=0; i<(SpinP_switch+1); i++){
    Tmp_Den_Grid[i] = (double**)malloc(sizeof(double*)*(Matomnum+1)); 
    Tmp_Den_Grid[i][0] = (double*)malloc(sizeof(double)*1); 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = F_M2G[Mc_AN];
      Tmp_Den_Grid[i][Mc_AN] = (double*)malloc(sizeof(double)*GridN_Atom[Gc_AN]);
    }
  }

  Den_Snd_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Den_Snd_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Snd_Grid_A2B[ID]*(SpinP_switch+1));
  }

  Den_Rcv_Grid_A2B = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Den_Rcv_Grid_A2B[ID] = (double*)malloc(sizeof(double)*Num_Rcv_Grid_A2B[ID]*(SpinP_switch+1));
  }

  /**********************************************
              calculate Tmp_Den_Grid
  ***********************************************/
    
#pragma omp parallel shared(FNAN,GridN_Atom,Matomnum,myid,G2ID,Orbs_Grid_FNAN,List_YOUSO,time_per_atom,Tmp_Den_Grid,Orbs_Grid,COrbs_Grid,GListTAtoms2,GListTAtoms1,NumOLG,CDM,SpinP_switch,WhatSpecies,ncn,F_G2M,natn,Spe_Total_CNO,M2G) private(OMPID,Nthrds,Nprocs,Mc_AN,h_AN,Stime_atom,Etime_atom,Gc_AN,Cwan,NO0,Gh_AN,Mh_AN,Rnh,Hwan,NO1,spin,i,j,tmp_CDM,Nog,Nc_0,Nc_1,Nc_2,Nc_3,Nh_0,Nh_1,Nh_2,Nh_3,orbs0_0,orbs0_1,orbs0_2,orbs0_3,orbs1_0,orbs1_1,orbs1_2,orbs1_3,sum_0,sum_1,sum_2,sum_3,tmp0_0,tmp0_1,tmp0_2,tmp0_3,Nc,Nh,orbs0,orbs1,sum,tmp0)
  {

    int spinmax;
    
    if      (SpinP_switch==0) spinmax = 0;
    else if (SpinP_switch==1) spinmax = 1;
    else if (SpinP_switch==3) spinmax = 1;

    orbs0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    orbs0_0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs0_1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs0_2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs0_3 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
    orbs1_3 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

    tmp_CDM = (double***)malloc(sizeof(double**)*(SpinP_switch+1)); 
    for (i=0; i<(SpinP_switch+1); i++){
      tmp_CDM[i] = (double**)malloc(sizeof(double*)*List_YOUSO[7]); 
      for (j=0; j<List_YOUSO[7]; j++){
	tmp_CDM[i][j] = (double*)malloc(sizeof(double)*List_YOUSO[7]); 
      }
    }

    /* get info. on OpenMP */ 

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();
    Nprocs = omp_get_num_procs();

    for (Mc_AN=(OMPID*Matomnum/Nthrds+1); Mc_AN<((OMPID+1)*Matomnum/Nthrds+1); Mc_AN++){

      dtime(&Stime_atom);

      /* set data on Mc_AN */

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan]; 

      for (spin=0; spin<=spinmax; spin++){
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){
	  Tmp_Den_Grid[spin][Mc_AN][Nc] = 0.0;
	}
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	/* set data on h_AN */
    
	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];
	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* store CDM into tmp_CDM */

	for (spin=0; spin<=spinmax; spin++){
	  for (i=0; i<NO0; i++){
	    for (j=0; j<NO1; j++){
	      tmp_CDM[spin][i][j] = CDM[spin][Mc_AN][h_AN][i][j];
	    }
	  }
	}

	/* summation of non-zero elements */

	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]-3; Nog+=4){

	  Nc_0 = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nc_1 = GListTAtoms1[Mc_AN][h_AN][Nog+1];
	  Nc_2 = GListTAtoms1[Mc_AN][h_AN][Nog+2];
	  Nc_3 = GListTAtoms1[Mc_AN][h_AN][Nog+3];

	  Nh_0 = GListTAtoms2[Mc_AN][h_AN][Nog];
	  Nh_1 = GListTAtoms2[Mc_AN][h_AN][Nog+1];
	  Nh_2 = GListTAtoms2[Mc_AN][h_AN][Nog+2];
	  Nh_3 = GListTAtoms2[Mc_AN][h_AN][Nog+3];

	  for (i=0; i<NO0; i++){
	    orbs0_0[i] = Orbs_Grid[Mc_AN][Nc_0][i];/* AITUNE */
	    orbs0_1[i] = Orbs_Grid[Mc_AN][Nc_1][i];/* AITUNE */
	    orbs0_2[i] = Orbs_Grid[Mc_AN][Nc_2][i];/* AITUNE */
	    orbs0_3[i] = Orbs_Grid[Mc_AN][Nc_3][i];/* AITUNE */
	  }

	  if (G2ID[Gh_AN]==myid){
	    for (j=0; j<NO1; j++){
	      orbs1_0[j] = Orbs_Grid[Mh_AN][Nh_0][j];/* AITUNE */
	      orbs1_1[j] = Orbs_Grid[Mh_AN][Nh_1][j];/* AITUNE */
	      orbs1_2[j] = Orbs_Grid[Mh_AN][Nh_2][j];/* AITUNE */
	      orbs1_3[j] = Orbs_Grid[Mh_AN][Nh_3][j];/* AITUNE */
	    }
	  }
	  else{
	    for (j=0; j<NO1; j++){
	      orbs1_0[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog  ][j];/* AITUNE */
	      orbs1_1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+1][j];/* AITUNE */
	      orbs1_2[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+2][j];/* AITUNE */
	      orbs1_3[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog+3][j];/* AITUNE */
	    }
	  }
	  
	  for (spin=0; spin<=spinmax; spin++){

	    /* Tmp_Den_Grid */

	    sum_0 = 0.0;
	    sum_1 = 0.0;
	    sum_2 = 0.0;
	    sum_3 = 0.0;

	    for (i=0; i<NO0; i++){

	      tmp0_0 = 0.0;
	      tmp0_1 = 0.0;
	      tmp0_2 = 0.0;
	      tmp0_3 = 0.0;

	      for (j=0; j<NO1; j++){
		tmp0_0 += orbs1_0[j]*tmp_CDM[spin][i][j];
		tmp0_1 += orbs1_1[j]*tmp_CDM[spin][i][j];
		tmp0_2 += orbs1_2[j]*tmp_CDM[spin][i][j];
		tmp0_3 += orbs1_3[j]*tmp_CDM[spin][i][j];
	      }

	      sum_0 += orbs0_0[i]*tmp0_0;
	      sum_1 += orbs0_1[i]*tmp0_1;
	      sum_2 += orbs0_2[i]*tmp0_2;
	      sum_3 += orbs0_3[i]*tmp0_3;
	    }

	    Tmp_Den_Grid[spin][Mc_AN][Nc_0] += sum_0;
	    Tmp_Den_Grid[spin][Mc_AN][Nc_1] += sum_1;
	    Tmp_Den_Grid[spin][Mc_AN][Nc_2] += sum_2;
	    Tmp_Den_Grid[spin][Mc_AN][Nc_3] += sum_3;

	  } /* spin */
	} /* Nog */

	for (; Nog<NumOLG[Mc_AN][h_AN]; Nog++){
 
	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog]; 
 
	  for (i=0; i<NO0; i++){
	    orbs0[i] = Orbs_Grid[Mc_AN][Nc][i];/* AITUNE */
	  }

	  if (G2ID[Gh_AN]==myid){
	    for (j=0; j<NO1; j++){
	      orbs1[j] = Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */
	    }
	  }
	  else{
	    for (j=0; j<NO1; j++){
	      orbs1[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */  
	    }
	  }

	  for (spin=0; spin<=spinmax; spin++){
 
	    /* Tmp_Den_Grid */
 
	    sum = 0.0;
	    for (i=0; i<NO0; i++){
	      tmp0 = 0.0;
	      for (j=0; j<NO1; j++){
		tmp0 += orbs1[j]*tmp_CDM[spin][i][j];
	      }
	      sum += orbs0[i]*tmp0;
	    }
 
	    Tmp_Den_Grid[spin][Mc_AN][Nc] += sum;
	  }
	}

      } /* h_AN */

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

    } /* Mc_AN */

    /* freeing of arrays */ 

    free(orbs0);
    free(orbs1);

    free(orbs0_0);
    free(orbs0_1);
    free(orbs0_2);
    free(orbs0_3);
    free(orbs1_0);
    free(orbs1_1);
    free(orbs1_2);
    free(orbs1_3);

    for (i=0; i<(SpinP_switch+1); i++){
      for (j=0; j<List_YOUSO[7]; j++){
	free(tmp_CDM[i][j]);
      }
      free(tmp_CDM[i]);
    }
    free(tmp_CDM);

#pragma omp flush(Tmp_Den_Grid)

  } /* #pragma omp parallel */

  /******************************************************
      MPI communication from the partitions A to B 
  ******************************************************/

  int spinmax;
    
  if      (SpinP_switch==0) spinmax = 0;
  else if (SpinP_switch==1) spinmax = 1;
  else if (SpinP_switch==3) spinmax = 1;
  
  /* copy Tmp_Den_Grid to Den_Snd_Grid_A2B */

  for (ID=0; ID<numprocs; ID++) Num_Snd_Grid_A2B[ID] = 0;
  
  N2D = Ngrid1*Ngrid2;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];

    for (AN=0; AN<GridN_Atom[Gc_AN]; AN++){

      GN = GridListAtom[Mc_AN][AN];
      GN2N(GN,N3);
      n2D = N3[1]*Ngrid2 + N3[2];
      ID = (int)(n2D*(unsigned long long int)numprocs/N2D);

      if (SpinP_switch==0){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]] = Tmp_Den_Grid[0][Mc_AN][AN];
      }
      else if (SpinP_switch==1){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+1] = Tmp_Den_Grid[1][Mc_AN][AN];
      }
      else if (SpinP_switch==3){
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+0] = Tmp_Den_Grid[0][Mc_AN][AN];
        Den_Snd_Grid_A2B[ID][Num_Snd_Grid_A2B[ID]*2+1] = Tmp_Den_Grid[1][Mc_AN][AN];
      }

      Num_Snd_Grid_A2B[ID]++;
    }
  }    

  /* MPI: A to B */  

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (Num_Snd_Grid_A2B[IDS]!=0){
      MPI_Isend( &Den_Snd_Grid_A2B[IDS][0], Num_Snd_Grid_A2B[IDS]*(spinmax+1), 
                 MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
    }

    if (Num_Rcv_Grid_A2B[IDR]!=0){
      MPI_Recv( &Den_Rcv_Grid_A2B[IDR][0], Num_Rcv_Grid_A2B[IDR]*(spinmax+1), 
                MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
    }

    if (Num_Snd_Grid_A2B[IDS]!=0) MPI_Wait(&request,&stat);
  }
      
  /******************************************************
   superposition of rho_i to calculate charge density 
   in the partition B.
  ******************************************************/

  /* initialize arrays */

  for (spin=0; spin<(spinmax+1); spin++){
    for (BN=0; BN<My_NumGridB_AB; BN++){
      Density_Grid_B[spin][BN] = 0.0;
    }
  }

  /* superposition of densities rho_i */

  for (ID=0; ID<numprocs; ID++){

    for (LN=0; LN<Num_Rcv_Grid_A2B[ID]; LN++){

      BN    = Index_Rcv_Grid_A2B[ID][3*LN+0];      
      Gc_AN = Index_Rcv_Grid_A2B[ID][3*LN+1];        
      GRc   = Index_Rcv_Grid_A2B[ID][3*LN+2]; 

      if (Solver!=4 || (Solver==4 && atv_ijk[GRc][1]==0 )){

	/* spin collinear non-polarization */
	if ( SpinP_switch==0 ){
	  Density_Grid_B[0][BN] += Den_Rcv_Grid_A2B[ID][LN];
	}

	/* spin collinear polarization */
	else if ( SpinP_switch==1 ){
	  Density_Grid_B[0][BN] += Den_Rcv_Grid_A2B[ID][LN*2  ];
	  Density_Grid_B[1][BN] += Den_Rcv_Grid_A2B[ID][LN*2+1];
	} 

	/* spin non-collinear */
	else if ( SpinP_switch==3 ){
	  Density_Grid_B[0][BN] += Den_Rcv_Grid_A2B[ID][LN*2  ];
	  Density_Grid_B[1][BN] += Den_Rcv_Grid_A2B[ID][LN*2+1];
	} 

      } /* if (Solve!=4.....) */           

    } /* AN */ 
  } /* ID */  

  /******************************************************
             MPI: from the partitions B to C
  ******************************************************/

  Data_Grid_Copy_B2C_2( Density_Grid_B, Density_Grid );  

  /******************************************************
              diagonalize spinor density 
  ******************************************************/

  /* if (SpinP_switch==3) diagonalize_nc_density(); */

  /* freeing of arrays */

  for (i=0; i<(SpinP_switch+1); i++){
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){
      free(Tmp_Den_Grid[i][Mc_AN]);
    }
    free(Tmp_Den_Grid[i]);
  }
  free(Tmp_Den_Grid);

  for (ID=0; ID<numprocs; ID++){
    free(Den_Snd_Grid_A2B[ID]);
  }
  free(Den_Snd_Grid_A2B);

  for (ID=0; ID<numprocs; ID++){
    free(Den_Rcv_Grid_A2B[ID]);
  }
  free(Den_Rcv_Grid_A2B);
}


