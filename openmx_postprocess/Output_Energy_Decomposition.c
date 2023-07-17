/**********************************************************************
  Output_Energy_Decomposition.c:

   Output_Energy_Decomposition.c is a subrutine to output decomposed energies.
 
  Log of Output_Energy_Decomposition.c:

     03/Aug./2014  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


void Output_Energy_Decomposition()
{ 
  int Mc_AN,Gc_AN,tno0,Cwan,num,l,m,n,mul,Nene;
  int wan1,wan2,i,j,k,spin,tag=999;
  double Stime_atom,Etime_atom;
  double sum0,sum1,sum,wa[2],time0;
  double ****DecE,*tmp_array0;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char file_DecE[YOUSO10];
  FILE *fp_DecE;
  int numprocs,myid,ID;
  char buf[fp_bsize];          /* setvbuf */
  MPI_Status stat;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  Nene = 5;

  /* allocation of arrays */

  DecE = (double****)malloc(sizeof(double***)*(atomnum+1));
  for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
    DecE[Gc_AN] = (double***)malloc(sizeof(double**)*List_YOUSO[7]);
    for (i=0; i<List_YOUSO[7]; i++){
      DecE[Gc_AN][i] = (double**)malloc(sizeof(double*)*2);
      for (spin=0; spin<2; spin++){
        DecE[Gc_AN][i][spin] = (double*)malloc(sizeof(double)*Nene);
        for (n=0; n<Nene; n++) DecE[Gc_AN][i][spin][n] = 0.0;
      }
    }
  }

  tmp_array0 = (double*)malloc(sizeof(double)*List_YOUSO[7]*Nene*2);

  /****************************************
   MPI: DecE
  ****************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    ID = G2ID[Gc_AN];
    wan1 = WhatSpecies[Gc_AN];

    /* sending from ID to Host_ID */ 

    if (myid==ID){
  
      Mc_AN = F_G2M[Gc_AN];  

      num = 0; 

      if (SpinP_switch==0){

	for (i=0; i<Spe_Total_CNO[wan1]; i++){
	  tmp_array0[num] = DecEkin[0][Mc_AN][i]; num++; tmp_array0[num] = DecEkin[0][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEv[  0][Mc_AN][i]; num++; tmp_array0[num] = DecEv[  0][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEcon[0][Mc_AN][i]; num++; tmp_array0[num] = DecEcon[0][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEscc[0][Mc_AN][i]; num++; tmp_array0[num] = DecEscc[0][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEvdw[0][Mc_AN][i]; num++; tmp_array0[num] = DecEvdw[0][Mc_AN][i]; num++;
	}
      }

      else if (SpinP_switch==1) {

	for (i=0; i<Spe_Total_CNO[wan1]; i++){
	  tmp_array0[num] = DecEkin[0][Mc_AN][i]; num++; tmp_array0[num] = DecEkin[1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEv[  0][Mc_AN][i]; num++; tmp_array0[num] = DecEv[  1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEcon[0][Mc_AN][i]; num++; tmp_array0[num] = DecEcon[1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEscc[0][Mc_AN][i]; num++; tmp_array0[num] = DecEscc[1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEvdw[0][Mc_AN][i]; num++; tmp_array0[num] = DecEvdw[1][Mc_AN][i]; num++;
	}
      }

      else if (SpinP_switch==3) {

	for (i=0; i<Spe_Total_CNO[wan1]; i++){
	  tmp_array0[num] = DecEkin[0][Mc_AN][i]; num++; tmp_array0[num] = DecEkin[1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEv[  0][Mc_AN][i]; num++; tmp_array0[num] = DecEv[  1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEcon[0][Mc_AN][i]; num++; tmp_array0[num] = DecEcon[1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEscc[0][Mc_AN][i]; num++; tmp_array0[num] = DecEscc[1][Mc_AN][i]; num++;
	  tmp_array0[num] = DecEvdw[0][Mc_AN][i]; num++; tmp_array0[num] = DecEvdw[1][Mc_AN][i]; num++;
	}
      }

      if (myid!=Host_ID){
        MPI_Send(&tmp_array0[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1);
      }
    }

    /* receiving at Host_ID from ID */ 

    if (myid==Host_ID){

      num = 2*Nene*Spe_Total_CNO[wan1]; 

      if (ID!=Host_ID){
	MPI_Recv(&tmp_array0[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      }

      num = 0; 
      for (i=0; i<Spe_Total_CNO[wan1]; i++){
	for (n=0; n<Nene; n++){
	  DecE[Gc_AN][i][0][n] = tmp_array0[num]; num++;
	  DecE[Gc_AN][i][1][n] = tmp_array0[num]; num++;
	}
      }

    }
  }

  /****************************************
             making a file, *.DecE
  ****************************************/

  if ( myid==Host_ID ){

    sprintf(file_DecE,"%s%s.DecE",filepath,filename);

    if ((fp_DecE = fopen(file_DecE,"w")) != NULL){

#ifdef xt3
      setvbuf(fp_DecE,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      if (SpinP_switch==0 || SpinP_switch==1){

	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"*************************************************************************\n");
	fprintf(fp_DecE,"*************************************************************************\n");
	fprintf(fp_DecE,"            Decomposed energies in Hartree unit            \n");
	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"   Utot = Utot(up) + Utot(dn)\n");
	fprintf(fp_DecE,"        = Ukin(up) + Ukin(dn) + Uv(up) + Uv(dn)\n");
	fprintf(fp_DecE,"        + Ucon(up)+ Ucon(dn) + Ucore+UH0 + Uvdw\n");
	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"   Uele = Ukin(up) + Ukin(dn) + Uv(up) + Uv(dn)\n");
	fprintf(fp_DecE,"   Ucon arizes from a constant potential added in the formalism\n");
	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"           up: up spin state, dn: down spin state\n");
	fprintf(fp_DecE,"*************************************************************************\n");
	fprintf(fp_DecE,"*************************************************************************\n\n");
      }

      else if (SpinP_switch==3){

	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"*************************************************************************\n");
	fprintf(fp_DecE,"*************************************************************************\n");
	fprintf(fp_DecE,"            Decomposed energies in Hartree unit            \n");
	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"   Utot = Ukin + Uv + Ucon + Ucore+UH0 + Uvdw\n");
	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"   Uele = Ukin + Uv\n");
	fprintf(fp_DecE,"   Ucon arizes from a constant potential added in the formalism\n");
	fprintf(fp_DecE,"\n");
	fprintf(fp_DecE,"*************************************************************************\n");
	fprintf(fp_DecE,"*************************************************************************\n\n");
      }

      /* total energy */

      sum = 0.0;
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (i=0; i<Spe_Total_CNO[wan1]; i++){
          for (spin=0; spin<=1; spin++){
	    for (n=0; n<Nene; n++){
	      sum += DecE[Gc_AN][i][spin][n];
	    }
	  }
	}
      }

      fprintf(fp_DecE,"  Total energy (Hartree) = %18.15f\n",sum);

      /* Decomposed energies with respect to atom */

      fprintf(fp_DecE,"\n  Decomposed.energies.(Hartree).with.respect.to.atom\n\n");

      if (SpinP_switch==0 || SpinP_switch==1){
        fprintf(fp_DecE,"                 Utot              Utot(up)    Utot(dn)    Ukin(up)    Ukin(dn)    Uv(up)      Uv(dn)      Ucon(up)    Ucon(dn)    Ucore+UH0   Uvdw\n");
      }
      else if (SpinP_switch==3){
        fprintf(fp_DecE,"                 Utot              Ukin        Uv          Ucon        Ucore+UH0   Uvdw\n");
      }  
    
      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

	wan1 = WhatSpecies[Gc_AN];

        /* Utot */

        for (spin=0; spin<=1; spin++){
          wa[spin] = 0.0;
          for (i=0; i<Spe_Total_CNO[wan1]; i++){
	    for (n=0; n<Nene; n++){
	      wa[spin] += DecE[Gc_AN][i][spin][n];
	    }
	  }
	}

	fprintf(fp_DecE,"  %4d %4s  %18.12f",Gc_AN,SpeName[wan1],wa[0]+wa[1]);
        if (SpinP_switch==0 || SpinP_switch==1){
          fprintf(fp_DecE," %11.6f %11.6f",wa[0],wa[1]);
	}

        /* Ukin */

        n = 0;
        wa[0] = 0.0; wa[1] = 0.0;
        for (i=0; i<Spe_Total_CNO[wan1]; i++){
          wa[0] += DecE[Gc_AN][i][0][n];
          wa[1] += DecE[Gc_AN][i][1][n];
	}

        if (SpinP_switch==0 || SpinP_switch==1){
          fprintf(fp_DecE," %11.6f %11.6f",wa[0],wa[1]);
	}
        else if (SpinP_switch==3){
          fprintf(fp_DecE," %11.6f",wa[0]+wa[1]);
	}

        /* Uv */

        n = 1;
        wa[0] = 0.0; wa[1] = 0.0;
        for (i=0; i<Spe_Total_CNO[wan1]; i++){
          wa[0] += DecE[Gc_AN][i][0][n];
          wa[1] += DecE[Gc_AN][i][1][n];
	}

        if (SpinP_switch==0 || SpinP_switch==1){
          fprintf(fp_DecE," %11.6f %11.6f",wa[0],wa[1]);
	}
        else if (SpinP_switch==3){
          fprintf(fp_DecE," %11.6f",wa[0]+wa[1]);
	}

        /* Ucon */

        n = 2;
        wa[0] = 0.0; wa[1] = 0.0;
        for (i=0; i<Spe_Total_CNO[wan1]; i++){
          wa[0] += DecE[Gc_AN][i][0][n];
          wa[1] += DecE[Gc_AN][i][1][n];
	}
        if (SpinP_switch==0 || SpinP_switch==1){
          fprintf(fp_DecE," %11.6f %11.6f",wa[0],wa[1]);
	}
        else if (SpinP_switch==3){
          fprintf(fp_DecE," %11.6f",wa[0]+wa[1]);
	}

        /* Uscc */

        n = 3;
        wa[0] = 0.0; wa[1] = 0.0;
        for (i=0; i<Spe_Total_CNO[wan1]; i++){
          wa[0] += DecE[Gc_AN][i][0][n];
          wa[1] += DecE[Gc_AN][i][1][n];
	}
        fprintf(fp_DecE," %11.6f",wa[0]+wa[1]);

        /* Uvdw */

        n = 4;
        wa[0] = 0.0; wa[1] = 0.0;
        for (i=0; i<Spe_Total_CNO[wan1]; i++){
          wa[0] += DecE[Gc_AN][i][0][n];
          wa[1] += DecE[Gc_AN][i][1][n];
	}
        fprintf(fp_DecE," %11.6f",wa[0]+wa[1]);

	fprintf(fp_DecE,"\n");
      }

      /* Decomposed energies with respect to atomic orbital */

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

      fprintf(fp_DecE,"\n\n  Decomposed.energies.(Hartree).with.respect.to.atomic.orbital\n");

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

        wan1 = WhatSpecies[Gc_AN];

        if (SpinP_switch==0 || SpinP_switch==1){
          fprintf(fp_DecE,"\n %4d %4s          Utot        Utot(up)    Utot(dn)    Ukin(up)    Ukin(dn)    Uv(up)      Uv(dn)      Ucon(up)    Ucon(dn)    Ucore+UH0   Uvdw\n",
                Gc_AN,SpeName[wan1]);
	}
        else if (SpinP_switch==3){
          fprintf(fp_DecE,"\n %4d %4s          Utot        Ukin        Uv          Ucon        Ucore+UH0   Uvdw\n",
                Gc_AN,SpeName[wan1]);
	}

	fprintf(fp_DecE,"            multiple\n");

        /* Ucore+UH0 and Uvdw */

        if (SpinP_switch==0 || SpinP_switch==1){
          fprintf(fp_DecE,"  none           %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f",
		DecE[Gc_AN][0][0][3]+DecE[Gc_AN][0][1][3]+DecE[Gc_AN][0][0][4]+DecE[Gc_AN][0][1][4],
                DecE[Gc_AN][0][0][3]+DecE[Gc_AN][0][0][4],DecE[Gc_AN][0][1][3]+DecE[Gc_AN][0][1][4],
                0.0,0.0,0.0,0.0,0.0,0.0);
	}
        else if (SpinP_switch==3){
          fprintf(fp_DecE,"  none           %11.6f %11.6f %11.6f %11.6f",
		DecE[Gc_AN][0][0][3]+DecE[Gc_AN][0][1][3]+DecE[Gc_AN][0][0][4]+DecE[Gc_AN][0][1][4],
                0.0,0.0,0.0);
	}

        n = 3;
        fprintf(fp_DecE," %11.6f",DecE[Gc_AN][0][0][n]+DecE[Gc_AN][0][1][n]);
        n = 4;
        fprintf(fp_DecE," %11.6f",DecE[Gc_AN][0][0][n]+DecE[Gc_AN][0][1][n]);
        fprintf(fp_DecE,"\n");

	num = 0;
	for (l=0; l<=Supported_MaxL; l++){

	  for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
	    for (m=0; m<(2*l+1); m++){

	      fprintf(fp_DecE,"  %s%2d ",Name_Angular[l][m],mul);

              /* Utot */

              for (spin=0; spin<=1; spin++){
                wa[spin] = 0.0;
		for (n=0; n<Nene-2; n++){
                  wa[spin] += DecE[Gc_AN][num][spin][n];
		}
	      }

              if (SpinP_switch==0 || SpinP_switch==1){
                fprintf(fp_DecE," %11.6f %11.6f %11.6f",wa[0]+wa[1],wa[0],wa[1]);
	      }
              else if (SpinP_switch==3){
                fprintf(fp_DecE," %11.6f",wa[0]+wa[1]);
	      }

              /* Ukin */

              n = 0;
              if (SpinP_switch==0 || SpinP_switch==1){
                fprintf(fp_DecE," %11.6f %11.6f",DecE[Gc_AN][num][0][n],DecE[Gc_AN][num][1][n]);
	      }
              else if (SpinP_switch==3){
                fprintf(fp_DecE," %11.6f",DecE[Gc_AN][num][0][n]+DecE[Gc_AN][num][1][n]);
	      }
              /* Uv */

              n = 1;
              if (SpinP_switch==0 || SpinP_switch==1){
                fprintf(fp_DecE," %11.6f %11.6f",DecE[Gc_AN][num][0][n],DecE[Gc_AN][num][1][n]);
	      }
              else if (SpinP_switch==3){
                fprintf(fp_DecE," %11.6f",DecE[Gc_AN][num][0][n]+DecE[Gc_AN][num][1][n]);
	      }

              /* Ucon */

              n = 2;
              if (SpinP_switch==0 || SpinP_switch==1){
                fprintf(fp_DecE," %11.6f %11.6f",DecE[Gc_AN][num][0][n],DecE[Gc_AN][num][1][n]);
	      }
              else if (SpinP_switch==3){
                fprintf(fp_DecE," %11.6f",DecE[Gc_AN][num][0][n]+DecE[Gc_AN][num][1][n]);
	      }

              /* Uscc */

              fprintf(fp_DecE," %11.6f",0.0);

              /* Uvdw */

              fprintf(fp_DecE," %11.6f",0.0);

              fprintf(fp_DecE,"\n");

	      num++;
	    }
	  } 
	}
      }

      fclose(fp_DecE);
    }
    else{
      printf("Failure of saving the DecE file.\n");
    }
  }

  /* freeing of arrays */

  for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
    for (i=0; i<List_YOUSO[7]; i++){
      for (spin=0; spin<2; spin++){
        free(DecE[Gc_AN][i][spin]);
      }
      free(DecE[Gc_AN][i]);
    }
    free(DecE[Gc_AN]);
  }
  free(DecE);

  free(tmp_array0);
}

