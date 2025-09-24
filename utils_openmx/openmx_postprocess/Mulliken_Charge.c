/**********************************************************************
  Mulliken_Charge.c:

     Mulliken_Charge.c is a subrutine to calculate Mulliken charge.
 
  Log of Mulliken_Charge.c:

     27/Dec/2002  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


#define  stdout_MulP  1



double Mulliken_Charge( char *mode )
{
  int Mc_AN,Gc_AN,tno0,Cwan,num,l,m,mul;
  int wan1,wan2,i,j,k,Gs_AN,s_AN,spin;
  int tag=999;
  double MulP[4],My_Total_SpinS;
  double MulP_LA[4],MulP_LB[4],MagML;
  double summx,summy,summz;
  double x,y,z,TZ;
  double My_Total_SpinSx;
  double My_Total_SpinSy;
  double My_Total_SpinSz;
  double My_Total_OrbitalMoment;
  double My_Total_OrbitalMomentx;
  double My_Total_OrbitalMomenty;
  double My_Total_OrbitalMomentz;
  double sden,tmp,tmp0,tmp1,tmp2;  
  double Total_Mul_up,Total_Mul_dn,Total_Mul;
  double tmpA0,tmpA1,tmpA2;
  double tmpB0,tmpB1,tmpB2;
  double Re11,Re22,Re12,Im12,d1,d2,d3;
  double theta[2],phi[2],Nup[2],Ndown[2],sit,cot,sip,cop;
  double Stime_atom, Etime_atom;
  double my_data[4],data[4];
  double ***DecMulP;
  double *tmp_array0;
  double *sum_l,*sum_mul;
  double S_coordinate[3];
  double TStime,TEtime,time0;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char file_MC[YOUSO10] = ".MC";
  FILE *fp_MC;
  int numprocs,myid,ID,myid_original;
  char buf[fp_bsize];          /* setvbuf */
  MPI_Status stat;

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid_original);

  /* calculate TZ */

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan1 = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan1];
  }

  /* allocation of arrays */

  DecMulP = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    DecMulP[spin] = (double**)malloc(sizeof(double*)*(atomnum+1));
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      DecMulP[spin][Gc_AN] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
      for (i=0; i<List_YOUSO[7]; i++) DecMulP[spin][Gc_AN][i] = 0.0;
    }
  }

  tmp_array0 = (double*)malloc(sizeof(double)*(List_YOUSO[7]*(SpinP_switch+1)));
  sum_l   = (double*)malloc(sizeof(double)*(SpinP_switch+1));
  sum_mul = (double*)malloc(sizeof(double)*(SpinP_switch+1));

  My_Total_SpinS  = 0.0;
  My_Total_SpinSx = 0.0;
  My_Total_SpinSy = 0.0;
  My_Total_SpinSz = 0.0;

  My_Total_OrbitalMoment  = 0.0;
  My_Total_OrbitalMomentx = 0.0;
  My_Total_OrbitalMomenty = 0.0;
  My_Total_OrbitalMomentz = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    wan1 = WhatSpecies[Gc_AN];

    MulP[0] = 0.0;
    MulP[1] = 0.0;
    MulP[2] = 0.0;
    MulP[3] = 0.0;

    /****************************************************
      in case of NEGF, add partial decomposed Mulliken
             population coming from CL or CR 
    ****************************************************/

    if (Solver==4){
      for (spin=0; spin<=SpinP_switch; spin++){
        for (i=0; i<Spe_Total_CNO[wan1]; i++){
          DecMulP[spin][Gc_AN][i] += TRAN_DecMulP[spin][Mc_AN][i];
          MulP[spin] += TRAN_DecMulP[spin][Mc_AN][i];
	}
      }
    }

    /****************************************************
               loop for neighbouring atoms  
    ****************************************************/

    for (s_AN=0; s_AN<=FNAN[Gc_AN]; s_AN++){
      Gs_AN = natn[Gc_AN][s_AN];
      wan2 = WhatSpecies[Gs_AN];

      if (Cnt_switch==0){
        for (spin=0; spin<=SpinP_switch; spin++){
          for (i=0; i<Spe_Total_CNO[wan1]; i++){

            tmp0 = 0.0;
	    for (j=0; j<Spe_Total_CNO[wan2]; j++){
              tmp0 +=  DM[0][spin][Mc_AN][s_AN][i][j]*OLP[0][Mc_AN][s_AN][i][j];
   	    }

            /* due to difference in the definition between density matrix and density */
            if (spin==3){ 
              DecMulP[spin][Gc_AN][i] -= tmp0;
              MulP[spin] -= tmp0;
	    }
            else{
              DecMulP[spin][Gc_AN][i] += tmp0;
              MulP[spin] += tmp0;
	    }
	  }
        }
      }
      else{
        for (spin=0; spin<=SpinP_switch; spin++){
          for (i=0; i<Spe_Total_CNO[wan1]; i++){

            tmp0 = 0.0;
	    for (j=0; j<Spe_Total_CNO[wan2]; j++){
              tmp0 +=  DM[0][spin][Mc_AN][s_AN][i][j]*CntOLP[0][Mc_AN][s_AN][i][j];
   	    }

           /* due to difference in the definition between density matrix and density */
            if (spin==3){ 
              DecMulP[spin][Gc_AN][i] -= tmp0;
              MulP[spin] -= tmp0;
	    }
            else { 
              DecMulP[spin][Gc_AN][i] += tmp0;
              MulP[spin] += tmp0;
	    }
	  }
        }
      }
    }

    /****************************************************
      if (SpinP_switch==3)
      spin non-collinear

      U \rho U^+ = n 
    ****************************************************/

    if (SpinP_switch==3){

      Re11 = MulP[0];
      Re22 = MulP[1];
      Re12 = MulP[2];
      Im12 = MulP[3];

      EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

      MulP[0] = Nup[0];
      MulP[1] = Ndown[0];
      MulP[2] = theta[0];
      MulP[3] = phi[0];

      /* decomposed Mulliken populations */
      for (i=0; i<Spe_Total_CNO[wan1]; i++){

        Re11 = DecMulP[0][Gc_AN][i];
        Re22 = DecMulP[1][Gc_AN][i];
        Re12 = DecMulP[2][Gc_AN][i];
        Im12 = DecMulP[3][Gc_AN][i];

        EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndown, theta, phi );

        DecMulP[0][Gc_AN][i] = Nup[0];
        DecMulP[1][Gc_AN][i] = Ndown[0];
	DecMulP[2][Gc_AN][i] = theta[0];
	DecMulP[3][Gc_AN][i] = phi[0];
      }

    }

    /****************************************************
     set InitN_USpin and InitN_DSpin  
    ****************************************************/

    if (SpinP_switch==0){
      InitN_USpin[Gc_AN] = MulP[0];
      InitN_DSpin[Gc_AN] = MulP[0];
    }
    else if (SpinP_switch==1){
      InitN_USpin[Gc_AN] = MulP[0];
      InitN_DSpin[Gc_AN] = MulP[1];
    }
    else if (SpinP_switch==3){
      InitN_USpin[Gc_AN] = MulP[0];
      InitN_DSpin[Gc_AN] = MulP[1];
      Angle0_Spin[Gc_AN] = MulP[2];
      Angle1_Spin[Gc_AN] = MulP[3];
    }

    /****************************************************
     My_Total_SpinS
    ****************************************************/

    /* spin non-collinear */
    if (SpinP_switch==3){
        
      theta[0] = Angle0_Spin[Gc_AN];
      phi[0]   = Angle1_Spin[Gc_AN];
      sden = 0.5*(InitN_USpin[Gc_AN] - InitN_DSpin[Gc_AN]);
      My_Total_SpinSx += sden*sin(theta[0])*cos(phi[0]);
      My_Total_SpinSy += sden*sin(theta[0])*sin(phi[0]);
      My_Total_SpinSz += sden*cos(theta[0]);

    }
    /* spin collinear */
    else {
      My_Total_SpinS += 0.5*(InitN_USpin[Gc_AN] - InitN_DSpin[Gc_AN]);
    }

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;

  } /* Mc_AN */

  /****************************************
   MPI:

   My_Total_SpinS
  ****************************************/

  /* spin non-collinear */

  if (SpinP_switch==3){

    /* spin moment */

    my_data[0] = My_Total_SpinSx;
    my_data[1] = My_Total_SpinSy;
    my_data[2] = My_Total_SpinSz;

    MPI_Allreduce(&my_data[0], &data[0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    Total_SpinSx = data[0];
    Total_SpinSy = data[1];
    Total_SpinSz = data[2];
    
    xyz2spherical( Total_SpinSx,Total_SpinSy,Total_SpinSz, 0.0,0.0,0.0, S_coordinate ); 

    Total_SpinS      = S_coordinate[0];
    Total_SpinAngle0 = S_coordinate[1];
    Total_SpinAngle1 = S_coordinate[2];
  }

  /* spin collinear */
  else{
    MPI_Allreduce(&My_Total_SpinS, &Total_SpinS, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  /* MPI: InitN_USpin and InitN_DSpin */

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    
    ID = G2ID[Gc_AN];

    if (ID!=Host_ID){

      if (myid==ID){

	/* spin non-collinear */
	if (SpinP_switch==3){
	  data[0] = InitN_USpin[Gc_AN];
	  data[1] = InitN_DSpin[Gc_AN];
	  data[2] = Angle0_Spin[Gc_AN];
	  data[3] = Angle1_Spin[Gc_AN];
	  MPI_Send(&data[0], 4, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1);
	}
	else{
	  data[0] = InitN_USpin[Gc_AN];
	  data[1] = InitN_DSpin[Gc_AN];
	  MPI_Send(&data[0], 2, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1);
	}
      }

      else if (myid==Host_ID){

	/* spin non-collinear */
	if (SpinP_switch==3){
	  MPI_Recv(&data[0], 4, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
	  InitN_USpin[Gc_AN] = data[0];
	  InitN_DSpin[Gc_AN] = data[1];
	  Angle0_Spin[Gc_AN] = data[2];
	  Angle1_Spin[Gc_AN] = data[3];
	}
	else{
	  MPI_Recv(&data[0], 2, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
	  InitN_USpin[Gc_AN] = data[0];
	  InitN_DSpin[Gc_AN] = data[1];
	}
      }
    }
  }

  /* MPI: DecMulP */

  if ( strcasecmp(mode,"write")==0 ){

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID = G2ID[Gc_AN];

      if (ID!=Host_ID){

	if (myid==ID){

	  num = 0; 
	  for (spin=0; spin<(SpinP_switch+1); spin++){
	    wan1 = WhatSpecies[Gc_AN];
	    for (i=0; i<Spe_Total_CNO[wan1]; i++){
	      tmp_array0[num] = DecMulP[spin][Gc_AN][i];
	      num++;
	    }
	  }

	  MPI_Send(&tmp_array0[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1);

	}
	else if (myid==Host_ID){

	  wan1 = WhatSpecies[Gc_AN];
	  num = (SpinP_switch+1)*Spe_Total_CNO[wan1]; 
	  MPI_Recv(&tmp_array0[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);

	  num = 0; 
	  for (spin=0; spin<(SpinP_switch+1); spin++){
	    wan1 = WhatSpecies[Gc_AN];
	    for (i=0; i<Spe_Total_CNO[wan1]; i++){
	      DecMulP[spin][Gc_AN][i] = tmp_array0[num];
	      num++;
	    }
	  }
	}
      }
    }
  }

  /****************************************
   stdout MulP
  ****************************************/

  if (myid_original==Host_ID && stdout_MulP && strcasecmp(mode,"stdout")==0) {

    Total_Mul_up = 0.0;
    Total_Mul_dn = 0.0;

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      wan1 = WhatSpecies[Gc_AN];

      if (SpinP_switch==0){

        Total_Mul_up += InitN_USpin[Gc_AN];
        Total_Mul_dn += InitN_USpin[Gc_AN];

        if (Gc_AN<=20 && 1<=level_stdout){
          printf(" %4d %4s  MulP %8.4f%8.4f sum %8.4f\n",
                  Gc_AN,SpeName[wan1],
                  InitN_USpin[Gc_AN],InitN_USpin[Gc_AN],
                  InitN_USpin[Gc_AN]+InitN_USpin[Gc_AN]);
	}
      }
      else if (SpinP_switch==1){

        Total_Mul_up += InitN_USpin[Gc_AN];
        Total_Mul_dn += InitN_DSpin[Gc_AN];

        if (Gc_AN<=20 && level_stdout<=1){
          printf(" %4d %4s  MulP %8.4f %8.4f sum %8.4f diff %8.4f\n",
                    Gc_AN,SpeName[wan1],
                    InitN_USpin[Gc_AN],InitN_DSpin[Gc_AN],
                    InitN_USpin[Gc_AN]+InitN_DSpin[Gc_AN],
                    InitN_USpin[Gc_AN]-InitN_DSpin[Gc_AN]);
	}
      }
      else if (SpinP_switch==3){
    
        sden = (InitN_USpin[Gc_AN] - InitN_DSpin[Gc_AN]);
        theta[0] = Angle0_Spin[Gc_AN];
        phi[0]   = Angle1_Spin[Gc_AN];
        x = sden*sin(theta[0])*cos(phi[0]);
        y = sden*sin(theta[0])*sin(phi[0]);
        z = sden*cos(theta[0]);

        MagML = OrbitalMoment[Gc_AN];
        theta[0] = Angle0_Orbital[Gc_AN];
        phi[0]   = Angle1_Orbital[Gc_AN];
        x += MagML*sin(theta[0])*cos(phi[0]);
        y += MagML*sin(theta[0])*sin(phi[0]);
        z += MagML*cos(theta[0]);

        xyz2spherical( x,y,z, 0.0,0.0,0.0, S_coordinate ); 

        Total_Mul_up += InitN_USpin[Gc_AN];
        Total_Mul_dn += InitN_DSpin[Gc_AN];

        if (Gc_AN<=20 && level_stdout<=1){
          printf(" %4d %4s  MulP%5.2f%5.2f sum %5.2f diff %5.2f (%6.2f %6.2f)  Ml %4.2f (%6.2f %6.2f)  Ml+s %4.2f (%6.2f %6.2f)\n",
                 Gc_AN,SpeName[wan1],
                 InitN_USpin[Gc_AN],InitN_DSpin[Gc_AN],
                 InitN_USpin[Gc_AN]+InitN_DSpin[Gc_AN],
                 InitN_USpin[Gc_AN]-InitN_DSpin[Gc_AN],
 		 fmod(Angle0_Spin[Gc_AN]/PI*180.0, 360.0),
		 fmod(Angle1_Spin[Gc_AN]/PI*180.0, 360.0), 
 	         OrbitalMoment[Gc_AN],
  	         fmod(Angle0_Orbital[Gc_AN]/PI*180.0, 360.0),
                 fmod(Angle1_Orbital[Gc_AN]/PI*180.0, 360.0),
                 S_coordinate[0],
                 fmod(S_coordinate[1]/PI*180.0, 360.0),
                 fmod(S_coordinate[2]/PI*180.0, 360.0) );
	}

      }

    } /* Gc_AN */

    if (20<atomnum && level_stdout<=1){
      printf("     ..........\n");
      printf("     ......\n\n");
    }

    if (0<level_stdout){

      printf(" Sum of MulP: up   =%12.5f down          =%12.5f\n",
               Total_Mul_up,Total_Mul_dn);
      printf("              total=%12.5f ideal(neutral)=%12.5f\n",
                            Total_Mul_up+Total_Mul_dn,TZ);     
    }

    Total_Mul = Total_Mul_up + Total_Mul_dn;
  }

  MPI_Bcast(&Total_Mul, 1, MPI_DOUBLE, Host_ID, mpi_comm_level1);

  /********************************************************
    check the stability of the eigenvalue solver 
    by looking the total number of electrons. 
    In some cases with degenerate states, the eigenvectors
    are not properly calculated, resulting in violation of
    norm of the eigenstates.
  *********************************************************/

  rediagonalize_flag_overlap_matrix = 0;

  if (Solver!=4 && strcasecmp(mode,"stdout")==0){

    int tmp_flag;
    double Dnum;

    Dnum = TZ - Total_Mul - system_charge;
     
    if (1.0e-8<fabs(Dnum)){

      if      (dste_flag==2) tmp_flag = 3;   /* vx -> qr    */ 
      else if (dste_flag==3) tmp_flag = 1;   /* qr -> dc    */ 
      else if (dste_flag==1) tmp_flag = 0;   /* dc -> gr    */
      else if (dste_flag==0){                /* gr -> ELPA1 */
        tmp_flag = 2; 
        rediagonalize_flag_overlap_matrix_ELPA1 = 1;        
      }

      dste_flag = tmp_flag;

      /****************************************************
         In Cluster_DFT, Band_DFT_Col, or Band_DFT_NonCol, 
         the overlap matrix will be rediagonalized.
      ****************************************************/  

      if (myid==Host_ID && 1<level_stdout){
        printf("Eigensolver changed Dnum=%18.15f dste_flag=%2d rediagonalize_flag_overlap_matrix_ELPA1=%2d\n",
                Dnum,dste_flag,rediagonalize_flag_overlap_matrix_ELPA1);
      }

      rediagonalize_flag_overlap_matrix = 1;
    }
  }

  /****************************************
   file, *.MC
  ****************************************/

  if (strcasecmp(mode,"write")==0 ){
    /* MPI: InitN_USpin and InitN_DSpin */
    MPI_Bcast(&InitN_USpin[0], atomnum+1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    MPI_Bcast(&InitN_DSpin[0], atomnum+1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
  }

  if ( myid==Host_ID && strcasecmp(mode,"write")==0 ){

    Total_Mul_up = 0.0;
    Total_Mul_dn = 0.0;

    sprintf(file_MC,"%s%s.MC",filepath,filename);

    if ((fp_MC = fopen(file_MC,"w")) != NULL){

      setvbuf(fp_MC,buf,_IOFBF,fp_bsize);  /* setvbuf */

      fprintf(fp_MC,"\n");
      fprintf(fp_MC,"***********************************************************\n");
      fprintf(fp_MC,"***********************************************************\n");
      fprintf(fp_MC,"                   Mulliken populations                    \n");
      fprintf(fp_MC,"***********************************************************\n");
      fprintf(fp_MC,"***********************************************************\n\n");

      /* spin non-collinear */
      if (SpinP_switch==3){

        /* total Mulliken charge */

        fprintf(fp_MC,"   Total spin moment (muB)  %12.9f   Angles (Deg) %12.9f  %12.9f\n\n",
                2.0*Total_SpinS, fmod(Total_SpinAngle0/PI*180.0, 360.0), fmod(Total_SpinAngle1/PI*180.0, 360));
        fprintf(fp_MC,"               Up       Down      Sum      Diff        theta      phi\n");
        for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
          wan1 = WhatSpecies[Gc_AN];
          fprintf(fp_MC," %4d %4s  %8.5f %8.5f  %8.5f  %8.5f  %10.5f %10.5f\n",
                  Gc_AN,SpeName[wan1],InitN_USpin[Gc_AN],InitN_DSpin[Gc_AN],
                  InitN_USpin[Gc_AN]+InitN_DSpin[Gc_AN],
                  InitN_USpin[Gc_AN]-InitN_DSpin[Gc_AN],
                  fmod(Angle0_Spin[Gc_AN]/PI*180.0, 360.0), 
                  fmod(Angle1_Spin[Gc_AN]/PI*180.0, 360.0) );

          Total_Mul_up += InitN_USpin[Gc_AN];
          Total_Mul_dn += InitN_DSpin[Gc_AN];

        }

        fprintf(fp_MC,"\n");
        fprintf(fp_MC," Sum of MulP: up   =%12.5f down          =%12.5f\n",
                 Total_Mul_up,Total_Mul_dn);
        fprintf(fp_MC,"              total=%12.5f ideal(neutral)=%12.5f\n",
                              Total_Mul_up+Total_Mul_dn,TZ);     

      }
      else{

        /* total Mulliken charge */
        fprintf(fp_MC,"  Total spin moment (muB)  %12.9f\n\n",2.0*Total_SpinS);
        fprintf(fp_MC,"                    Up spin      Down spin     Sum           Diff\n");
        for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
          wan1 = WhatSpecies[Gc_AN];
          fprintf(fp_MC,"   %4d %4s     %12.9f %12.9f  %12.9f  %12.9f\n",
                  Gc_AN,SpeName[wan1],InitN_USpin[Gc_AN],InitN_DSpin[Gc_AN],
                  InitN_USpin[Gc_AN]+InitN_DSpin[Gc_AN],
                  InitN_USpin[Gc_AN]-InitN_DSpin[Gc_AN]);

          Total_Mul_up += InitN_USpin[Gc_AN];
          Total_Mul_dn += InitN_DSpin[Gc_AN];
        }

        fprintf(fp_MC,"\n");
        fprintf(fp_MC," Sum of MulP: up   =%12.5f down          =%12.5f\n",
                 Total_Mul_up,Total_Mul_dn);
        fprintf(fp_MC,"              total=%12.5f ideal(neutral)=%12.5f\n",
                              Total_Mul_up+Total_Mul_dn,TZ);     
      }

      /* decomposed Mulliken charge */

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

      fprintf(fp_MC,"\n\n  Decomposed Mulliken populations\n");

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

        wan1 = WhatSpecies[Gc_AN];

	/* spin collinear */
	if (SpinP_switch==0 || SpinP_switch==1){

	  fprintf(fp_MC,"\n %4d %4s          Up spin      Down spin     Sum           Diff\n",
                  Gc_AN,SpeName[wan1]);
	  fprintf(fp_MC,"            multiple\n");
	}

	/* spin non-collinear */
	else{
	  fprintf(fp_MC,"\n %4d %4s          Up spin      Down spin     Sum           Diff           Angles(Deg)\n",
		  Gc_AN,SpeName[wan1]);
	  fprintf(fp_MC,"            multiple\n");
	}

	num = 0;
	for (l=0; l<=Supported_MaxL; l++){

	  if (SpinP_switch==0){
	    sum_l[0] = 0.0;
	  }
	  else if (SpinP_switch==1){
	    sum_l[0] = 0.0;
	    sum_l[1] = 0.0;
	  }
	  else if (SpinP_switch==3){
	    sum_l[0] = 0.0;
	    sum_l[1] = 0.0;
	  }

	  for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){

	    if (SpinP_switch==0){
	      sum_mul[0] = 0.0;
	    }
	    else if (SpinP_switch==1){
	      sum_mul[0] = 0.0;
	      sum_mul[1] = 0.0;
	    }
	    else if (SpinP_switch==3){
	      sum_mul[0] = 0.0;
	      sum_mul[1] = 0.0;
	    }

	    for (m=0; m<(2*l+1); m++){

	      if (SpinP_switch==0){
		fprintf(fp_MC,"  %s%2d   %12.9f %12.9f  %12.9f  %12.9f\n",
			Name_Angular[l][m],mul,
			DecMulP[0][Gc_AN][num],DecMulP[0][Gc_AN][num],
			DecMulP[0][Gc_AN][num]+DecMulP[0][Gc_AN][num],
			DecMulP[0][Gc_AN][num]-DecMulP[0][Gc_AN][num]);

		sum_mul[0] += DecMulP[0][Gc_AN][num];
	      }
	      else if (SpinP_switch==1){
		fprintf(fp_MC,"  %s%2d   %12.9f %12.9f  %12.9f  %12.9f\n",
			Name_Angular[l][m],mul,
			DecMulP[0][Gc_AN][num],DecMulP[1][Gc_AN][num],
			DecMulP[0][Gc_AN][num]+DecMulP[1][Gc_AN][num],
			DecMulP[0][Gc_AN][num]-DecMulP[1][Gc_AN][num]);
                 
		sum_mul[0] += DecMulP[0][Gc_AN][num];
		sum_mul[1] += DecMulP[1][Gc_AN][num];
	      }

	      else if (SpinP_switch==3){
		fprintf(fp_MC,"  %s%2d   %12.9f %12.9f  %12.9f  %12.9f  %9.4f %9.4f\n",
			Name_Angular[l][m],mul,
			DecMulP[0][Gc_AN][num],DecMulP[1][Gc_AN][num],
			DecMulP[0][Gc_AN][num]+DecMulP[1][Gc_AN][num],
			DecMulP[0][Gc_AN][num]-DecMulP[1][Gc_AN][num],
			fmod(DecMulP[2][Gc_AN][num]/PI*180.0, 360.0),
                        fmod(DecMulP[3][Gc_AN][num]/PI*180.0, 360.0) );
                 
		sum_mul[0] += DecMulP[0][Gc_AN][num];
		sum_mul[1] += DecMulP[1][Gc_AN][num];
	      }

	      num++;
	    }

	    if (SpinP_switch==0){

	      fprintf(fp_MC,"   sum over m     %12.9f %12.9f  %12.9f  %12.9f\n",
		      sum_mul[0],sum_mul[0],
		      sum_mul[0]+sum_mul[0],
		      sum_mul[0]-sum_mul[0]);

	      sum_l[0] += sum_mul[0];
	    }
	    else if (SpinP_switch==1){
	      fprintf(fp_MC,"   sum over m     %12.9f %12.9f  %12.9f  %12.9f\n",
		      sum_mul[0],sum_mul[1],
		      sum_mul[0]+sum_mul[1],
		      sum_mul[0]-sum_mul[1]);

	      sum_l[0] += sum_mul[0];
	      sum_l[1] += sum_mul[1];
	    }

	    else if (SpinP_switch==3){
	      fprintf(fp_MC,"   sum over m     %12.9f %12.9f  %12.9f  %12.9f\n",
		      sum_mul[0],sum_mul[1],
		      sum_mul[0]+sum_mul[1],
		      sum_mul[0]-sum_mul[1]);

	      sum_l[0] += sum_mul[0];
	      sum_l[1] += sum_mul[1];
	    }


	  } 

	  if (Spe_Num_CBasis[wan1][l]!=0){

	    if (SpinP_switch==0){
	      fprintf(fp_MC,"   sum over m+mul %12.9f %12.9f  %12.9f  %12.9f\n",
		      sum_l[0],sum_l[0],
		      sum_l[0]+sum_l[0],
		      sum_l[0]-sum_l[0]);
	    }
	    else if (SpinP_switch==1){
	      fprintf(fp_MC,"   sum over m+mul %12.9f %12.9f  %12.9f  %12.9f\n",
		      sum_l[0],sum_l[1],
		      sum_l[0]+sum_l[1],
		      sum_l[0]-sum_l[1]);
	    }

	    else if (SpinP_switch==3){
	      fprintf(fp_MC,"   sum over m+mul %12.9f %12.9f  %12.9f  %12.9f\n",
		      sum_l[0],sum_l[1],
		      sum_l[0]+sum_l[1],
		      sum_l[0]-sum_l[1]);
	    }

	  }
	}
      }

      fclose(fp_MC);
    }
    else{
      printf("Failure of saving the MC file.\n");
    }
  }

  /* freeing of arrays */

  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (Gc_AN=0; Gc_AN<=atomnum; Gc_AN++){
      free(DecMulP[spin][Gc_AN]);
    }
    free(DecMulP[spin]);
  }
  free(DecMulP);

  free(tmp_array0);

  free(sum_l);
  free(sum_mul);

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}







