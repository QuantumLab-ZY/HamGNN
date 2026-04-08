/**********************************************************************
  Orbital_Moment.c:

    Orbital_Moment.c.c is a subrutine to calculate the orbital moment at 
    each atomic site. 

  Log of Occupation_Number_LDA_U.c:

     20/Nov./2006    -- Released by T.Ozaki
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

#ifdef c_complex
#include <complex.h>
#endif



void OM_onsite( char *mode );
void OM_dual( char *mode );


void Orbital_Moment( char *mode )
{


  OM_dual(mode);

  /*
  OM_onsite(mode);
  */

}





void OM_dual( char *mode )
{
  int i,j,k,l,m,n,mul,Gc_AN,Mc_AN;
  int spin,wan1,wan2,tno0,tno2,num,kl;
  int h_AN,Gh_AN,tno1,Mh_AN;
  int Hwan,Cwan,size1,size2;
  double tmp0,tmp1,tmp2,tmp;
  double tmpB0,tmpB1,tmpB2;
  double tmpA0,tmpA1,tmpA2;
  double summx,summy,summz;
  double My_Total_OrbitalMoment;
  double My_Total_OrbitalMomentx;
  double My_Total_OrbitalMomenty;
  double My_Total_OrbitalMomentz;
  double MulP_LA[4],MulP_LB[4],MagML;
  double S_coordinate[3];
  double ***DecMulP_LA;
  double ***DecMulP_LB;
  double *tmp_array0;
  double *tmp_array;
  double *tmp_array2;
  double *sum_l,*sum_mul;
  double *****iDM0;
  int *Snd_DM0_Size,*Rcv_DM0_Size;

  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char file_OM[YOUSO10] = ".OM";
  FILE *fp_OM;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  DecMulP_LA = (double***)malloc(sizeof(double**)*3);
  for (k=0; k<3; k++){
    DecMulP_LA[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      DecMulP_LA[k][Gc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]; i++) DecMulP_LA[k][Gc_AN][i] = 0.0;
    }
  }

  DecMulP_LB = (double***)malloc(sizeof(double**)*3);
  for (k=0; k<3; k++){
    DecMulP_LB[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      DecMulP_LB[k][Gc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]; i++) DecMulP_LB[k][Gc_AN][i] = 0.0;
    }
  }

  tmp_array0 = (double*)malloc(sizeof(double)*(List_YOUSO[7]*(SpinP_switch+1)));
  sum_l   = (double*)malloc(sizeof(double)*(SpinP_switch+1));
  sum_mul = (double*)malloc(sizeof(double)*(SpinP_switch+1));

  /* Snd_DM0_Size and Rcv_DM0_Size */

  Snd_DM0_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DM0_Size = (int*)malloc(sizeof(int)*numprocs);

  /* allocation of iDM0 */

  iDM0 = (double*****)malloc(sizeof(double****)*2); 
  for (k=0; k<2; k++){
    iDM0[k] = (double****)malloc(sizeof(double***)*(Matomnum+MatomnumF+1)); 
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = F_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_NO[Cwan];  
      }    

      iDM0[k][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1)); 
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_NO[Hwan];
	} 

	iDM0[k][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0); 
	for (i=0; i<tno0; i++){
	  iDM0[k][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1); 
	}
      }
    }
  }

  /****************************************************
      iDM[0][k][Matomnum] -> iDM0              
  ****************************************************/

  for (k=0; k<2; k++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      wan1 = WhatSpecies[Gc_AN];
      tno1 = Spe_Total_NO[wan1];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];        
	Hwan = WhatSpecies[Gh_AN];
	tno2 = Spe_Total_NO[Hwan];
	for (i=0; i<tno1; i++){
	  for (j=0; j<tno2; j++){
            iDM0[k][Mc_AN][h_AN][i][j] = iDM[0][k][Mc_AN][h_AN][i][j];
	  }
	}
      }
    }
  }

  /****************************************************
    MPI: iDM0              
  ****************************************************/

  /***********************************
             set data size
  ************************************/

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){
      tag = 999;

      /* find data size to send block data */
      if (F_Snd_Num[IDS]!=0){

	size1 = 0;
	for (k=0; k<2; k++){
	  for (n=0; n<F_Snd_Num[IDS]; n++){
	    Mc_AN = Snd_MAN[IDS][n];
	    Gc_AN = Snd_GAN[IDS][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      size1 += tno1*tno2; 
	    }
	  }
	}
 
	Snd_DM0_Size[IDS] = size1;
	MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_comm_level1, &request);
      }
      else{
	Snd_DM0_Size[IDS] = 0;
      }

      /* receiving of size of data */

      if (F_Rcv_Num[IDR]!=0){
	MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_comm_level1, &stat);
	Rcv_DM0_Size[IDR] = size2;
      }
      else{
	Rcv_DM0_Size[IDR] = 0;
      }

      if (F_Snd_Num[IDS]!=0) MPI_Wait(&request,&stat);

    }
  }

  /***********************************
             data transfer
  ************************************/

  tag = 999;
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*****************************
              sending of data 
      *****************************/

      if (F_Snd_Num[IDS]!=0){

	size1 = Snd_DM0_Size[IDS];

	/* allocation of array */

	tmp_array = (double*)malloc(sizeof(double)*size1);

	/* multidimentional array to vector array */

	num = 0;
	for (k=0; k<2; k++){
	  for (n=0; n<F_Snd_Num[IDS]; n++){
	    Mc_AN = Snd_MAN[IDS][n];
	    Gc_AN = Snd_GAN[IDS][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];
	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  tmp_array[num] = iDM0[k][Mc_AN][h_AN][i][j];
		  num++;
		} 
	      } 
	    }
	  }
	}

	MPI_Isend(&tmp_array[0], size1, MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      /*****************************
         receiving of block data
      *****************************/

      if (F_Rcv_Num[IDR]!=0){
        
	size2 = Rcv_DM0_Size[IDR];
        
	/* allocation of array */
	tmp_array2 = (double*)malloc(sizeof(double)*size2);
        
	MPI_Recv(&tmp_array2[0], size2, MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
        
	num = 0;
	for (k=0; k<2; k++){
	  Mc_AN = F_TopMAN[IDR] - 1;
	  for (n=0; n<F_Rcv_Num[IDR]; n++){
	    Mc_AN++;
	    Gc_AN = Rcv_GAN[IDR][n];
	    Cwan = WhatSpecies[Gc_AN]; 
	    tno1 = Spe_Total_NO[Cwan];

	    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	      Gh_AN = natn[Gc_AN][h_AN];        
	      Hwan = WhatSpecies[Gh_AN];
	      tno2 = Spe_Total_NO[Hwan];
	      for (i=0; i<tno1; i++){
		for (j=0; j<tno2; j++){
		  iDM0[k][Mc_AN][h_AN][i][j] = tmp_array2[num];
		  num++;
		}
	      }
	    }
	  }        
	}

	/* freeing of array */
	free(tmp_array2);
      }

      if (F_Snd_Num[IDS]!=0){
	MPI_Wait(&request,&stat);
	free(tmp_array);  /* freeing of array */
      } 
    }
  }

  /****************************************************
                     calculate NC_OcpN
  ****************************************************/

  My_Total_OrbitalMoment  = 0.0;
  My_Total_OrbitalMomentx = 0.0;
  My_Total_OrbitalMomenty = 0.0;
  My_Total_OrbitalMomentz = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

    MulP_LA[0] = 0.0;
    MulP_LA[1] = 0.0;
    MulP_LA[2] = 0.0;

    MulP_LB[0] = 0.0;
    MulP_LB[1] = 0.0;
    MulP_LB[2] = 0.0;

    for (m=0; m<Spe_Total_NO[wan1]; m++){

      tmpA0 = 0.0;
      tmpA1 = 0.0;
      tmpA2 = 0.0;

      tmpB0 = 0.0;
      tmpB1 = 0.0;
      tmpB2 = 0.0;

      for (n=0; n<Spe_Total_NO[wan1]; n++){
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  Gh_AN = natn[Gc_AN][h_AN];
	  Mh_AN = F_G2M[Gh_AN]; 
	  wan2 = WhatSpecies[Gh_AN];
	  kl = RMI1[Mc_AN][h_AN][0];

	  for (k=0; k<Spe_Total_NO[wan2]; k++){

	    tmpA0 += 0.5*OLP_L[0][Mc_AN][0][m][n]*
                         (iDM0[0][Mc_AN][h_AN][m][k]*OLP[0][Mc_AN][h_AN][n][k]
 	                 +iDM0[0][Mh_AN][kl  ][k][n]*OLP[0][Mc_AN][h_AN][m][k]);

	    tmpA1 += 0.5*OLP_L[1][Mc_AN][0][m][n]*
                         (iDM0[0][Mc_AN][h_AN][m][k]*OLP[0][Mc_AN][h_AN][n][k]
 	                 +iDM0[0][Mh_AN][kl  ][k][n]*OLP[0][Mc_AN][h_AN][m][k]);

	    tmpA2 += 0.5*OLP_L[2][Mc_AN][0][m][n]*
                         (iDM0[0][Mc_AN][h_AN][m][k]*OLP[0][Mc_AN][h_AN][n][k]
 	                 +iDM0[0][Mh_AN][kl  ][k][n]*OLP[0][Mc_AN][h_AN][m][k]);

	    tmpB0 += 0.5*OLP_L[0][Mc_AN][0][m][n]*
                         (iDM0[1][Mc_AN][h_AN][m][k]*OLP[0][Mc_AN][h_AN][n][k]
			 +iDM0[1][Mh_AN][kl  ][k][n]*OLP[0][Mc_AN][h_AN][m][k]);

	    tmpB1 += 0.5*OLP_L[1][Mc_AN][0][m][n]*
                         (iDM0[1][Mc_AN][h_AN][m][k]*OLP[0][Mc_AN][h_AN][n][k]
			 +iDM0[1][Mh_AN][kl  ][k][n]*OLP[0][Mc_AN][h_AN][m][k]);

	    tmpB2 += 0.5*OLP_L[2][Mc_AN][0][m][n]*
                         (iDM0[1][Mc_AN][h_AN][m][k]*OLP[0][Mc_AN][h_AN][n][k]
			 +iDM0[1][Mh_AN][kl  ][k][n]*OLP[0][Mc_AN][h_AN][m][k]);
	  }
	}
      }

      DecMulP_LA[0][Gc_AN][m] -= tmpA0;
      DecMulP_LA[1][Gc_AN][m] -= tmpA1;
      DecMulP_LA[2][Gc_AN][m] -= tmpA2;

      DecMulP_LB[0][Gc_AN][m] -= tmpB0;
      DecMulP_LB[1][Gc_AN][m] -= tmpB1;
      DecMulP_LB[2][Gc_AN][m] -= tmpB2;

      MulP_LA[0] -= tmpA0;
      MulP_LA[1] -= tmpA1;
      MulP_LA[2] -= tmpA2;

      MulP_LB[0] -= tmpB0;
      MulP_LB[1] -= tmpB1;
      MulP_LB[2] -= tmpB2;
    }

    tmp0 = MulP_LA[0] + MulP_LB[0];
    tmp1 = MulP_LA[1] + MulP_LB[1];
    tmp2 = MulP_LA[2] + MulP_LB[2];

    Orbital_Moment_XYZ[Gc_AN][0] = tmp0;
    Orbital_Moment_XYZ[Gc_AN][1] = tmp1;
    Orbital_Moment_XYZ[Gc_AN][2] = tmp2;

    xyz2spherical( tmp0,tmp1,tmp2, 0.0,0.0,0.0, S_coordinate );

    OrbitalMoment[Gc_AN]  = S_coordinate[0];
    Angle0_Orbital[Gc_AN] = S_coordinate[1];
    Angle1_Orbital[Gc_AN] = S_coordinate[2];

    My_Total_OrbitalMomentx += tmp0;
    My_Total_OrbitalMomenty += tmp1;
    My_Total_OrbitalMomentz += tmp2;

  } /* Mc_AN */


  MPI_Allreduce(&My_Total_OrbitalMomentx, &Total_OrbitalMomentx, 1, MPI_DOUBLE,
		MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Total_OrbitalMomenty, &Total_OrbitalMomenty, 1, MPI_DOUBLE,
		MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Total_OrbitalMomentz, &Total_OrbitalMomentz, 1, MPI_DOUBLE,
		MPI_SUM, mpi_comm_level1);

  xyz2spherical( Total_OrbitalMomentx,Total_OrbitalMomenty,Total_OrbitalMomentz,
		 0.0,0.0,0.0, S_coordinate );

  Total_OrbitalMoment       = S_coordinate[0];
  Total_OrbitalMomentAngle0 = S_coordinate[1];
  Total_OrbitalMomentAngle1 = S_coordinate[2];

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];

    MPI_Bcast(&Angle0_Orbital[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Angle1_Orbital[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&OrbitalMoment[Gc_AN],  1, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Orbital_Moment_XYZ[Gc_AN][0],  3, MPI_DOUBLE, ID, mpi_comm_level1);

    /* DecMulP_LA */

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	tmp_array0[num] = DecMulP_LA[spin][Gc_AN][i];
	num++;
      }
    }

    MPI_Bcast(&tmp_array0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	DecMulP_LA[spin][Gc_AN][i] = tmp_array0[num];
	num++;
      }
    }

    /* DecMulP_LB */

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	tmp_array0[num] = DecMulP_LB[spin][Gc_AN][i];
	num++;
      }
    }

    MPI_Bcast(&tmp_array0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	DecMulP_LB[spin][Gc_AN][i] = tmp_array0[num];
	num++;
      }
    }
  }

  /*
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    printf(" Atom%4d Ml %4.2f (%6.2f %6.2f)\n",
             Gc_AN,
	     OrbitalMoment[Gc_AN],
	     Angle0_Orbital[Gc_AN]/PI*180.0, Angle1_Orbital[Gc_AN]/PI*180.0);
  }
  */

  /****************************************
   file, *.OM
  ****************************************/

  if ( myid==Host_ID && strcasecmp(mode,"write")==0 ){

    sprintf(file_OM,"%s%s.OM",filepath,filename);

    if ((fp_OM = fopen(file_OM,"w")) != NULL){

      setvbuf(fp_OM,buf,_IOFBF,fp_bsize);  /* setvbuf */

      /* orbital moment for spin non-collinear */

      if (SpinP_switch==3){

        fprintf(fp_OM,"\n");
        fprintf(fp_OM,"***********************************************************\n");
        fprintf(fp_OM,"***********************************************************\n");
        fprintf(fp_OM,"                     Orbital moments                       \n");
        fprintf(fp_OM,"***********************************************************\n");
        fprintf(fp_OM,"***********************************************************\n\n");

        /* total Mulliken charge */

        fprintf(fp_OM,"   Total Orbital Moment (muB)  %12.9f   Angles  (Deg) %12.9f  %12.9f\n\n",
                Total_OrbitalMoment,Total_OrbitalMomentAngle0/PI*180.0,Total_OrbitalMomentAngle1/PI*180.0);

        fprintf(fp_OM,"          Orbital moment (muB)   theta (Deg)  phi (Deg)\n");

        for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
          wan1 = WhatSpecies[Gc_AN];

          fprintf(fp_OM," %4d %4s  %8.5f          %10.5f %10.5f\n",
                  Gc_AN,
                  SpeName[wan1],
                  OrbitalMoment[Gc_AN],
                  Angle0_Orbital[Gc_AN]/PI*180.0,
                  Angle1_Orbital[Gc_AN]/PI*180.0 );
        }

	/* decomposed orbital moments */

	Name_Angular[0][0] = "s          ";
	Name_Angular[1][0] = "px         ";
	Name_Angular[1][1] = "py         ";
	Name_Angular[1][2] = "pz         ";
	Name_Angular[2][0] = "d3z^2-r^2  ";
	Name_Angular[2][1] = "dx^2-y^2   ";
	Name_Angular[2][2] = "dxy        ";
	Name_Angular[2][3] = "dxz        ";
	Name_Angular[2][4] = "dyz        ";
	Name_Angular[3][0] = "f5z^2-3r^2 ";
	Name_Angular[3][1] = "f5xz^2-xr^2";
	Name_Angular[3][2] = "f5yz^2-yr^2";
	Name_Angular[3][3] = "fzx^2-zy^2 ";
	Name_Angular[3][4] = "fxyz       ";
	Name_Angular[3][5] = "fx^3-3*xy^2";
	Name_Angular[3][6] = "f3yx^2-y^3 ";
	Name_Angular[4][0] = "g1         ";
	Name_Angular[4][1] = "g2         ";
	Name_Angular[4][2] = "g3         ";
	Name_Angular[4][3] = "g4         ";
	Name_Angular[4][4] = "g5         ";
	Name_Angular[4][5] = "g6         ";
	Name_Angular[4][6] = "g7         ";
	Name_Angular[4][7] = "g8         ";
	Name_Angular[4][8] = "g9         ";

	fprintf(fp_OM,"\n\n  Decomposed Orbital Moments\n");

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];

	  fprintf(fp_OM,"\n %4d %4s         Orbital Moment(muB)    Angles (Deg)\n",Gc_AN,SpeName[wan1]);
	  fprintf(fp_OM,"            multiple\n");

	  num = 0;
	  for (l=0; l<=Supported_MaxL; l++){

	    sum_l[0] = 0.0;

	    for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){

	      sum_mul[0] = 0.0;

              summx = 0.0;
              summy = 0.0;
              summz = 0.0;

	      for (m=0; m<(2*l+1); m++){

                tmp0 = DecMulP_LA[0][Gc_AN][num] + DecMulP_LB[0][Gc_AN][num];
                tmp1 = DecMulP_LA[1][Gc_AN][num] + DecMulP_LB[1][Gc_AN][num];
                tmp2 = DecMulP_LA[2][Gc_AN][num] + DecMulP_LB[2][Gc_AN][num];

                summx += tmp0;
                summy += tmp1;
                summz += tmp2;

                xyz2spherical( tmp0, tmp1, tmp2, 0.0,0.0,0.0, S_coordinate );

	        fprintf(fp_OM,"  %s%2d   %12.9f         %9.4f %9.4f\n",
		        Name_Angular[l][m], 
                        mul,
                        S_coordinate[0],
                        S_coordinate[1]/PI*180.0,
                        S_coordinate[2]/PI*180.0);
		num++;
	      }

              xyz2spherical( summx, summy, summz, 0.0,0.0,0.0, S_coordinate );

              fprintf(fp_OM,"   sum over m     %12.9f         %9.4f %9.4f\n",
                      S_coordinate[0],
                      S_coordinate[1]/PI*180.0,
                      S_coordinate[2]/PI*180.0);
	    } 

	  }
	}

      }

      fclose(fp_OM);
    }
    else{
      printf("Failure of saving the MC file.\n");
    }
  }

  /****************************************************
                      freeing of arrays:
  ****************************************************/

  /* Snd_DM0_Size and Rcv_DM0_Size */

  free(Snd_DM0_Size);
  free(Rcv_DM0_Size);

  /* iDM0 */

  for (k=0; k<2; k++){

    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = F_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_NO[Cwan];  
      }    

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_NO[Hwan];
	} 

	for (i=0; i<tno0; i++){
	  free(iDM0[k][Mc_AN][h_AN][i]);
	}
	free(iDM0[k][Mc_AN][h_AN]);
      }
      free(iDM0[k][Mc_AN]);
    }
    free(iDM0[k]);
  }
  free(iDM0);

  for (k=0; k<3; k++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(DecMulP_LA[k][Gc_AN]);
    }
    free(DecMulP_LA[k]);
  }
  free(DecMulP_LA);

  for (k=0; k<3; k++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(DecMulP_LB[k][Gc_AN]);
    }
    free(DecMulP_LB[k]);
  }
  free(DecMulP_LB);

  free(tmp_array0);
  free(sum_l);
  free(sum_mul);
}











void OM_onsite( char *mode )
{
  int i,j,k,l,m,mul,Gc_AN,Mc_AN;
  int spin,wan1,tno0,num;
  int numprocs,myid,ID;
  double tmp0,tmp1,tmp2,tmp;
  double tmpB0,tmpB1,tmpB2;
  double tmpA0,tmpA1,tmpA2;
  double summx,summy,summz;
  double My_Total_OrbitalMoment;
  double My_Total_OrbitalMomentx;
  double My_Total_OrbitalMomenty;
  double My_Total_OrbitalMomentz;
  double MulP_LA[4],MulP_LB[4],MagML;
  double S_coordinate[3];
  double ***DecMulP_LA;
  double ***DecMulP_LB;
  double *tmp_array0;
  double *sum_l,*sum_mul;
  char *Name_Angular[20][10];
  char file_OM[YOUSO10] = ".OM";
  FILE *fp_OM;
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  DecMulP_LA = (double***)malloc(sizeof(double**)*3);
  for (k=0; k<3; k++){
    DecMulP_LA[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      DecMulP_LA[k][Gc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]; i++) DecMulP_LA[k][Gc_AN][i] = 0.0;
    }
  }

  DecMulP_LB = (double***)malloc(sizeof(double**)*3);
  for (k=0; k<3; k++){
    DecMulP_LB[k] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      DecMulP_LB[k][Gc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]; i++) DecMulP_LB[k][Gc_AN][i] = 0.0;
    }
  }

  tmp_array0 = (double*)malloc(sizeof(double)*(List_YOUSO[7]*(SpinP_switch+1)));
  sum_l   = (double*)malloc(sizeof(double)*(SpinP_switch+1));
  sum_mul = (double*)malloc(sizeof(double)*(SpinP_switch+1));

  My_Total_OrbitalMoment  = 0.0;
  My_Total_OrbitalMomentx = 0.0;
  My_Total_OrbitalMomenty = 0.0;
  My_Total_OrbitalMomentz = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

    MulP_LA[0] = 0.0;
    MulP_LA[1] = 0.0;
    MulP_LA[2] = 0.0;

    MulP_LB[0] = 0.0;
    MulP_LB[1] = 0.0;
    MulP_LB[2] = 0.0;

    for (i=0; i<Spe_Total_CNO[wan1]; i++){

      tmpA0 = 0.0;
      tmpA1 = 0.0;
      tmpA2 = 0.0;

      tmpB0 = 0.0;
      tmpB1 = 0.0;
      tmpB2 = 0.0;

      for (j=0; j<Spe_Total_CNO[wan1]; j++){

	tmp = iDM[0][0][Mc_AN][0][i][j];

	tmpA0 += tmp*OLP_L[0][Mc_AN][0][i][j];
	tmpA1 += tmp*OLP_L[1][Mc_AN][0][i][j];
	tmpA2 += tmp*OLP_L[2][Mc_AN][0][i][j];

	tmp = iDM[0][1][Mc_AN][0][i][j];

	tmpB0 += tmp*OLP_L[0][Mc_AN][0][i][j];
	tmpB1 += tmp*OLP_L[1][Mc_AN][0][i][j];
	tmpB2 += tmp*OLP_L[2][Mc_AN][0][i][j];
      }

      DecMulP_LA[0][Gc_AN][i] -= tmpA0;
      DecMulP_LA[1][Gc_AN][i] -= tmpA1;
      DecMulP_LA[2][Gc_AN][i] -= tmpA2;

      DecMulP_LB[0][Gc_AN][i] -= tmpB0;
      DecMulP_LB[1][Gc_AN][i] -= tmpB1;
      DecMulP_LB[2][Gc_AN][i] -= tmpB2;

      MulP_LA[0] -= tmpA0;
      MulP_LA[1] -= tmpA1;
      MulP_LA[2] -= tmpA2;

      MulP_LB[0] -= tmpB0;
      MulP_LB[1] -= tmpB1;
      MulP_LB[2] -= tmpB2;
    }

    tmp0 = MulP_LA[0] + MulP_LB[0];
    tmp1 = MulP_LA[1] + MulP_LB[1];
    tmp2 = MulP_LA[2] + MulP_LB[2];

    Orbital_Moment_XYZ[Gc_AN][0] = tmp0;
    Orbital_Moment_XYZ[Gc_AN][1] = tmp1;
    Orbital_Moment_XYZ[Gc_AN][2] = tmp2;

    xyz2spherical( tmp0,tmp1,tmp2, 0.0,0.0,0.0, S_coordinate );

    OrbitalMoment[Gc_AN]  = S_coordinate[0];
    Angle0_Orbital[Gc_AN] = S_coordinate[1];
    Angle1_Orbital[Gc_AN] = S_coordinate[2];

    My_Total_OrbitalMomentx += tmp0;
    My_Total_OrbitalMomenty += tmp1;
    My_Total_OrbitalMomentz += tmp2;

  }

  MPI_Allreduce(&My_Total_OrbitalMomentx, &Total_OrbitalMomentx, 1, MPI_DOUBLE,
		MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Total_OrbitalMomenty, &Total_OrbitalMomenty, 1, MPI_DOUBLE,
		MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&My_Total_OrbitalMomentz, &Total_OrbitalMomentz, 1, MPI_DOUBLE,
		MPI_SUM, mpi_comm_level1);

  xyz2spherical( Total_OrbitalMomentx,Total_OrbitalMomenty,Total_OrbitalMomentz,
		 0.0,0.0,0.0, S_coordinate );

  Total_OrbitalMoment       = S_coordinate[0];
  Total_OrbitalMomentAngle0 = S_coordinate[1];
  Total_OrbitalMomentAngle1 = S_coordinate[2];

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];

    MPI_Bcast(&Angle0_Orbital[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Angle1_Orbital[Gc_AN], 1, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&OrbitalMoment[Gc_AN],  1, MPI_DOUBLE, ID, mpi_comm_level1);
    MPI_Bcast(&Orbital_Moment_XYZ[Gc_AN][0],  3, MPI_DOUBLE, ID, mpi_comm_level1);

    /* DecMulP_LA */

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	tmp_array0[num] = DecMulP_LA[spin][Gc_AN][i];
	num++;
      }
    }

    MPI_Bcast(&tmp_array0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	DecMulP_LA[spin][Gc_AN][i] = tmp_array0[num];
	num++;
      }
    }

    /* DecMulP_LB */

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	tmp_array0[num] = DecMulP_LB[spin][Gc_AN][i];
	num++;
      }
    }

    MPI_Bcast(&tmp_array0[0], num, MPI_DOUBLE, ID, mpi_comm_level1);

    num = 0; 
    for (spin=0; spin<3; spin++){
      wan1 = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[wan1];  
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	DecMulP_LB[spin][Gc_AN][i] = tmp_array0[num];
	num++;
      }
    }

  }

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    printf(" Atom%4d Ml %4.2f (%6.2f %6.2f)\n",
             Gc_AN,
	     OrbitalMoment[Gc_AN],
	     Angle0_Orbital[Gc_AN]/PI*180.0, Angle1_Orbital[Gc_AN]/PI*180.0);
  }

  /****************************************
   file, *.OM
  ****************************************/

  if ( myid==Host_ID && strcasecmp(mode,"write")==0 ){

    sprintf(file_OM,"%s%s.OM",filepath,filename);

    if ((fp_OM = fopen(file_OM,"w")) != NULL){

      setvbuf(fp_OM,buf,_IOFBF,fp_bsize);  /* setvbuf */

      /* orbital moment for spin non-collinear */

      if (SpinP_switch==3){

        fprintf(fp_OM,"\n");
        fprintf(fp_OM,"***********************************************************\n");
        fprintf(fp_OM,"***********************************************************\n");
        fprintf(fp_OM,"                     Orbital moments                       \n");
        fprintf(fp_OM,"***********************************************************\n");
        fprintf(fp_OM,"***********************************************************\n\n");

        /* total Mulliken charge */

        fprintf(fp_OM,"   Total Orbital Moment (muB)  %12.9f   Angles  (Deg) %12.9f  %12.9f\n\n",
                Total_OrbitalMoment,Total_OrbitalMomentAngle0/PI*180.0,Total_OrbitalMomentAngle1/PI*180.0);

        fprintf(fp_OM,"          Orbital moment (muB)   theta (Deg)  phi (Deg)\n");

        for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
          wan1 = WhatSpecies[Gc_AN];

          fprintf(fp_OM," %4d %4s  %8.5f          %10.5f %10.5f\n",
                  Gc_AN,
                  SpeName[wan1],
                  OrbitalMoment[Gc_AN],
                  Angle0_Orbital[Gc_AN]/PI*180.0,
                  Angle1_Orbital[Gc_AN]/PI*180.0 );
        }

	/* decomposed orbital moments */

	Name_Angular[0][0] = "s          ";
	Name_Angular[1][0] = "px         ";
	Name_Angular[1][1] = "py         ";
	Name_Angular[1][2] = "pz         ";
	Name_Angular[2][0] = "d3z^2-r^2  ";
	Name_Angular[2][1] = "dx^2-y^2   ";
	Name_Angular[2][2] = "dxy        ";
	Name_Angular[2][3] = "dxz        ";
	Name_Angular[2][4] = "dyz        ";
	Name_Angular[3][0] = "f5z^2-3r^2 ";
	Name_Angular[3][1] = "f5xz^2-xr^2";
	Name_Angular[3][2] = "f5yz^2-yr^2";
	Name_Angular[3][3] = "fzx^2-zy^2 ";
	Name_Angular[3][4] = "fxyz       ";
	Name_Angular[3][5] = "fx^3-3*xy^2";
	Name_Angular[3][6] = "f3yx^2-y^3 ";

	fprintf(fp_OM,"\n\n  Decomposed Orbital Moments\n");

	for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	  wan1 = WhatSpecies[Gc_AN];

	  fprintf(fp_OM,"\n %4d %4s         Orbital Moment(muB)    Angles (Deg)\n",Gc_AN,SpeName[wan1]);
	  fprintf(fp_OM,"            multiple\n");

	  num = 0;
	  for (l=0; l<=Supported_MaxL; l++){

	    sum_l[0] = 0.0;

	    for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){

	      sum_mul[0] = 0.0;

              summx = 0.0;
              summy = 0.0;
              summz = 0.0;

	      for (m=0; m<(2*l+1); m++){

                tmp0 = DecMulP_LA[0][Gc_AN][num] + DecMulP_LB[0][Gc_AN][num];
                tmp1 = DecMulP_LA[1][Gc_AN][num] + DecMulP_LB[1][Gc_AN][num];
                tmp2 = DecMulP_LA[2][Gc_AN][num] + DecMulP_LB[2][Gc_AN][num];

                summx += tmp0;
                summy += tmp1;
                summz += tmp2;

                xyz2spherical( tmp0, tmp1, tmp2, 0.0,0.0,0.0, S_coordinate );

	        fprintf(fp_OM,"  %s%2d   %12.9f         %9.4f %9.4f\n",
		        Name_Angular[l][m], 
                        mul,
                        S_coordinate[0],
                        S_coordinate[1]/PI*180.0,
                        S_coordinate[2]/PI*180.0);
		num++;
	      }

              xyz2spherical( summx, summy, summz, 0.0,0.0,0.0, S_coordinate );

              fprintf(fp_OM,"   sum over m     %12.9f         %9.4f %9.4f\n",
                      S_coordinate[0],
                      S_coordinate[1]/PI*180.0,
                      S_coordinate[2]/PI*180.0);
	    } 

	  }
	}

      }

      fclose(fp_OM);
    }
    else{
      printf("Failure of saving the MC file.\n");
    }
  }

  /* free arrays */

  for (k=0; k<3; k++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(DecMulP_LA[k][Gc_AN]);
    }
    free(DecMulP_LA[k]);
  }
  free(DecMulP_LA);

  for (k=0; k<3; k++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(DecMulP_LB[k][Gc_AN]);
    }
    free(DecMulP_LB[k]);
  }
  free(DecMulP_LB);

  free(tmp_array0);
  free(sum_l);
  free(sum_mul);

}


