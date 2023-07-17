/**********************************************************************
  SCF2File.c:

     SCF2File.c is a subroutine to output connectivity, Hamiltonian,
     overlap, and etc. to a binary file, filename.scfout.

  Log of SCF2DFT.c:

     01/July/2003  Released by T.Ozaki 
     17/Sep./2019  Modified by N. Yamaguchi

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "openmx_common.h"
#include "tran_variables.h"
#include "mpi.h"

#define MAX_LINE_SIZE 256

/* Added by N. Yamaguchi ***/
#define SCFOUT_VERSION 3
/* ***/

static void Output( FILE *fp, char *inputfile );
static void Calc_OLPpo();
static void Calc_OLPmo();

/* Modified by N. Yamaguchi ***/
double ******OLPpo;
double *****OLPmo;
/* ***/

/* Added by N. Yamaguchi ***/
int order_max=1;
/* ***/

void SCF2File(char *mode, char *inputfile)
{
  int Mc_AN,Gc_AN,tno0,tno1;
  int Cwan,h_AN,Gh_AN,Hwan;
  int i;
  char fname[YOUSO10];
  char operate[300];
  FILE *fp;
  int numprocs,myid,ID;
  int succeed_open;
  char buf[fp_bsize];          /* setvbuf */

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocation of array */

  /* Modified by N. Yamaguchi ***/
  /* OLPpo: matrix for position operator */

  int direction, order;
  OLPpo=(double******)malloc(sizeof(double*****)*3);
  for (direction=0; direction<3; direction++){
    OLPpo[direction]=(double*****)malloc(sizeof(double****)*order_max);
    for (order=0; order<order_max; order++){
      OLPpo[direction][order]=(double****)malloc(sizeof(double***)*(Matomnum+1));
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];
	}

	OLPpo[direction][order][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  }

	  OLPpo[direction][order][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
	  for (i=0; i<tno0; i++){
	    OLPpo[direction][order][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	  }
	}
      }
    }
  }

  /* OLPmo: matrix for momentum operator */

  OLPmo=(double*****)malloc(sizeof(double****)*3);
  for (direction=0; direction<3; direction++){

    OLPmo[direction]=(double****)malloc(sizeof(double***)*(Matomnum+1));
    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = S_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      OLPmo[direction][Mc_AN] = (double***)malloc(sizeof(double**)*(FNAN[Gc_AN]+1));
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	OLPmo[direction][Mc_AN][h_AN] = (double**)malloc(sizeof(double*)*tno0);
	for (i=0; i<tno0; i++){
	  OLPmo[direction][Mc_AN][h_AN][i] = (double*)malloc(sizeof(double)*tno1);
	}
      }
    }
  }

  /* end of allocation of OLPpo and OLPp */

  /* write a file */

  if ( strcasecmp(mode,"write")==0 ) {
    //sprintf(fname,"%s%s.scfout",filepath,filename);
    sprintf(fname,"overlap.scfout"); // Yang Zhong

    if (myid==Host_ID){

      if ((fp = fopen(fname,"w")) != NULL){

	setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
	succeed_open = 1;
      }
      else {
	succeed_open = 0;
	printf("Failure of making the scfout file (%s).\n",fname);
      }
    }

    MPI_Bcast(&succeed_open, 1, MPI_INT, Host_ID, mpi_comm_level1);

    if (succeed_open==1){
      if (myid==Host_ID) printf("Save the scfout file (%s)\n",fname);

      /* matrix elements for position operator */
      Calc_OLPpo();

      /* matrix elements for momentum operator */
      Calc_OLPmo();

      Output( fp, inputfile );

      if (myid==Host_ID) fclose(fp);
    } 
  }

  /* free array */

  /* Modified by N. Yamaguchi ***/

  for (direction=0; direction<3; direction++){
    for (order=0; order<order_max; order++){
      FNAN[0] = 0;
      for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = S_M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  tno0 = Spe_Total_CNO[Cwan];
	}

	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	  if (Mc_AN==0){
	    tno1 = 1;
	  }
	  else{
	    Gh_AN = natn[Gc_AN][h_AN];
	    Hwan = WhatSpecies[Gh_AN];
	    tno1 = Spe_Total_CNO[Hwan];
	  }

	  for (i=0; i<tno0; i++){
	    free(OLPpo[direction][order][Mc_AN][h_AN][i]);
	  }
	  free(OLPpo[direction][order][Mc_AN][h_AN]);
	}
	free(OLPpo[direction][order][Mc_AN]);
      }
      free(OLPpo[direction][order]);
    }
    free(OLPpo[direction]);
  }
  free(OLPpo);
  OLPpo=NULL;

  /* OLPmo */

  for (direction=0; direction<3; direction++){

    FNAN[0] = 0;
    for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = S_M2G[Mc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_CNO[Cwan];
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;
	}
	else{
	  Gh_AN = natn[Gc_AN][h_AN];
	  Hwan = WhatSpecies[Gh_AN];
	  tno1 = Spe_Total_CNO[Hwan];
	}

	for (i=0; i<tno0; i++){
	  free(OLPmo[direction][Mc_AN][h_AN][i]);
	}
	free(OLPmo[direction][Mc_AN][h_AN]);
      }
      free(OLPmo[direction][Mc_AN]);
    }
    free(OLPmo[direction]);
  }
  free(OLPmo);
  OLPmo=NULL;
}




void Output( FILE *fp, char *inputfile )
{
  int Gc_AN,Mc_AN,ct_AN,h_AN,i,j,can,Gh_AN;
  int num,wan1,wan2,TNO1,TNO2,spin,Rn,count;
  int k,Mh_AN,Rnh,Hwan,NO0,NO1,Qwan;
  int q_AN,Mq_AN,Gq_AN,Rnq;
  int i_vec[20],*p_vec;
  int numprocs,myid,ID,tag=999;
  double *Tmp_Vec,d_vec[20];
  FILE *fp_inp;
  char strg[MAX_LINE_SIZE];
  char buf[fp_bsize];          /* setvbuf */

  //Added by Yang Zhong
#ifdef DERIVATIVE
  double *Tmp_Vec_3;
#endif

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /***************************************************************
                     information of connectivity
  ****************************************************************/

  if (myid==Host_ID){

    /****************************************************
       atomnum
       spinP_switch
       version (added by N. Yamaguchi)
    ****************************************************/

    i = 0;
    i_vec[0] = atomnum;

    /* Disabled by N. Yamaguchi
     * i_vec[1] = SpinP_switch;
     */

    /* Added by N. Yamaguchi ***/
    int version=SCFOUT_VERSION;
    i_vec[1]=SpinP_switch+4*version;
    /* ***/

    i_vec[2] = Catomnum;
    i_vec[3] = Latomnum;
    i_vec[4] = Ratomnum;
    i_vec[5] = TCpyCell;

    fwrite(i_vec,sizeof(int),6,fp);

    /****************************************************
      order_max (added by N. Yamaguchi for HWC)
     ****************************************************/

    i_vec[0]=order_max;
    fwrite(i_vec,sizeof(int),1,fp);

    /****************************************************
                       atv[Rn][4]
    ****************************************************/

    for (Rn=0; Rn<=TCpyCell; Rn++){
      fwrite(atv[Rn],sizeof(double),4,fp);
    }

    /****************************************************
                       atv_ijk[Rn][4]
    ****************************************************/

    for (Rn=0; Rn<=TCpyCell; Rn++){
      fwrite(atv_ijk[Rn],sizeof(int),4,fp);
    }

    /****************************************************
                 # of orbitals in each atom
    ****************************************************/

    p_vec = (int*)malloc(sizeof(int)*atomnum);
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      wan1 = WhatSpecies[ct_AN];
      TNO1 = Spe_Total_CNO[wan1];
      p_vec[ct_AN-1] = TNO1;
    }
    fwrite(p_vec,sizeof(int),atomnum,fp);
    free(p_vec);

    /****************************************************
     FNAN[]:
     number of first nearest neighbouring atoms
    ****************************************************/

    p_vec = (int*)malloc(sizeof(int)*atomnum);
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      p_vec[ct_AN-1] = FNAN[ct_AN];
    }
    fwrite(p_vec,sizeof(int),atomnum,fp);
    free(p_vec);

    /****************************************************
      natn[][]:
      grobal index of neighboring atoms of an atom ct_AN
     ****************************************************/

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      fwrite(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp);
    }

    /****************************************************
      ncn[][]:
      grobal index for cell of neighboring atoms
      of an atom ct_AN
    ****************************************************/

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      fwrite(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp);
    }

    /****************************************************
      tv[4][4]:
      unit cell vectors in Bohr
    ****************************************************/

    fwrite(tv[1],sizeof(double),4,fp);
    fwrite(tv[2],sizeof(double),4,fp);
    fwrite(tv[3],sizeof(double),4,fp);

    /****************************************************
      rtv[4][4]:
      reciprocal unit cell vectors in Bohr^{-1}
    ****************************************************/

    fwrite(rtv[1],sizeof(double),4,fp);
    fwrite(rtv[2],sizeof(double),4,fp);
    fwrite(rtv[3],sizeof(double),4,fp);

    /****************************************************
      Gxyz[][1-3]:
      atomic coordinates in Bohr
    ****************************************************/

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      fwrite(Gxyz[ct_AN],sizeof(double),4,fp);
    }

  } /* if (myid==Host_ID) */

  /***************************************************************
                         Hamiltonian matrix 
  ****************************************************************/

  Tmp_Vec = (double*)malloc(sizeof(double)*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);

  for (spin=0; spin<=SpinP_switch; spin++){

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      ID = G2ID[Gc_AN];

      if (myid==ID){

        num = 0;

        Mc_AN = F_G2M[Gc_AN];
        wan1 = WhatSpecies[Gc_AN];
        TNO1 = Spe_Total_CNO[wan1];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          wan2 = WhatSpecies[Gh_AN];
          TNO2 = Spe_Total_CNO[wan2];

          if (Cnt_switch==0){
            for (i=0; i<TNO1; i++){
              for (j=0; j<TNO2; j++){
                Tmp_Vec[num] = H[spin][Mc_AN][h_AN][i][j];
                num++;
	      }
            }
          }
          else{
            for (i=0; i<TNO1; i++){
              for (j=0; j<TNO2; j++){
                Tmp_Vec[num] = CntH[spin][Mc_AN][h_AN][i][j];
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
        else{
          fwrite(Tmp_Vec, sizeof(double), num, fp);
        }
      }

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
        MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
        fwrite(Tmp_Vec, sizeof(double), num, fp);
      }

    }
  }

  /***************************************************************
                                  iHNL
  ****************************************************************/

  if ( SpinP_switch==3 ){

    for (spin=0; spin<List_YOUSO[5]; spin++){

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	ID = G2ID[Gc_AN];

	if (myid==ID){

	  num = 0;

	  Mc_AN = F_G2M[Gc_AN];
	  wan1 = WhatSpecies[Gc_AN];
	  TNO1 = Spe_Total_CNO[wan1];
	  for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	    Gh_AN = natn[Gc_AN][h_AN];
	    wan2 = WhatSpecies[Gh_AN];
	    TNO2 = Spe_Total_CNO[wan2];

	    for (i=0; i<TNO1; i++){
	      for (j=0; j<TNO2; j++){
		Tmp_Vec[num] = iHNL[spin][Mc_AN][h_AN][i][j];
		num++;
	      }
	    }
	  }

	  if (myid!=Host_ID){
	    MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
	    MPI_Wait(&request,&stat);
	    MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
	    MPI_Wait(&request,&stat);
	  }
	  else{
	    fwrite(Tmp_Vec, sizeof(double), num, fp);
	  }
	}

	else if (ID!=myid && myid==Host_ID){
	  MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
	  MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
	  fwrite(Tmp_Vec, sizeof(double), num, fp);
	}

      }  
    }
  }

  /***************************************************************
                          overlap matrix 
  ****************************************************************/

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];

    if (myid==ID){

      num = 0;

      Mc_AN = F_G2M[Gc_AN];
      wan1 = WhatSpecies[Gc_AN];
      TNO1 = Spe_Total_CNO[wan1];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];
        TNO2 = Spe_Total_CNO[wan2];

        if (Cnt_switch==0){
          for (i=0; i<TNO1; i++){
            for (j=0; j<TNO2; j++){
              Tmp_Vec[num] = OLP[0][Mc_AN][h_AN][i][j];
              num++;
            }
          }
        }
        else{
          for (i=0; i<TNO1; i++){
            for (j=0; j<TNO2; j++){
              Tmp_Vec[num] = CntOLP[0][Mc_AN][h_AN][i][j];
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
      else{
        fwrite(Tmp_Vec, sizeof(double), num, fp);
      }
    }

    else if (ID!=myid && myid==Host_ID){
      MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
      MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      fwrite(Tmp_Vec, sizeof(double), num, fp);
    }

  }

  /***************************************************************
      The derivatives of overlap matrix (Added by Yang Zhong)
  ****************************************************************/
#ifdef DERIVATIVE
  Tmp_Vec_3 = (double *)malloc(3*sizeof(double) * List_YOUSO[8] * List_YOUSO[7] * List_YOUSO[7]);

  for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++)
  {
    ID = G2ID[Gc_AN];

    if (myid == ID)
    {

      num = 0;

      Mc_AN = F_G2M[Gc_AN];
      wan1 = WhatSpecies[Gc_AN];
      TNO1 = Spe_Total_CNO[wan1];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {
        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];
        TNO2 = Spe_Total_CNO[wan2];

        if (Cnt_switch == 0)
        {
          for (i = 0; i < TNO1; i++)
          {
            for (j = 0; j < TNO2; j++)
            {
              Tmp_Vec_3[num] = OLP[1][Mc_AN][h_AN][i][j];
              Tmp_Vec_3[num+1] = OLP[2][Mc_AN][h_AN][i][j];
              Tmp_Vec_3[num+2] = OLP[3][Mc_AN][h_AN][i][j];
              num = num + 3;
            }
          }
        }
        else
        {
          for (i = 0; i < TNO1; i++)
          {
            for (j = 0; j < TNO2; j++)
            {
              Tmp_Vec_3[num] = CntOLP[1][Mc_AN][h_AN][i][j];
              Tmp_Vec_3[num+1] = CntOLP[2][Mc_AN][h_AN][i][j];
              Tmp_Vec_3[num+2] = CntOLP[3][Mc_AN][h_AN][i][j];
              num = num + 3;
            }
          }
        }
      }

      if (myid != Host_ID)
      {
        MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
        MPI_Isend(&Tmp_Vec_3[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
      }
      else
      {
        fwrite(Tmp_Vec_3, sizeof(double), num, fp);
      }
    }

    else if (ID != myid && myid == Host_ID)
    {
      MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
      MPI_Recv(&Tmp_Vec_3[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      fwrite(Tmp_Vec_3, sizeof(double), num, fp);
    }
  }

  /***************************************************************
      overlap matrix elements with lx,y,z (Added by Yang Zhong)
  ****************************************************************/
  for (Gc_AN = 1; Gc_AN <= atomnum; Gc_AN++)
  {
    ID = G2ID[Gc_AN];

    if (myid == ID)
    {

      num = 0;

      Mc_AN = F_G2M[Gc_AN];
      wan1 = WhatSpecies[Gc_AN];
      TNO1 = Spe_Total_CNO[wan1];
      for (h_AN = 0; h_AN <= FNAN[Gc_AN]; h_AN++)
      {
        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];
        TNO2 = Spe_Total_CNO[wan2];

        if (Cnt_switch == 0)
        {
          for (i = 0; i < TNO1; i++)
          {
            for (j = 0; j < TNO2; j++)
            {
              Tmp_Vec_3[num] = OLP_L[0][Mc_AN][h_AN][i][j];
              Tmp_Vec_3[num + 1] = OLP_L[1][Mc_AN][h_AN][i][j];
              Tmp_Vec_3[num + 2] = OLP_L[2][Mc_AN][h_AN][i][j];
              num = num + 3;
            }
          }
        }
      }

      if (myid != Host_ID)
      {
        MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
        MPI_Isend(&Tmp_Vec_3[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request, &stat);
      }
      else
      {
        fwrite(Tmp_Vec_3, sizeof(double), num, fp);
      }
    }

    else if (ID != myid && myid == Host_ID)
    {
      MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
      MPI_Recv(&Tmp_Vec_3[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      fwrite(Tmp_Vec_3, sizeof(double), num, fp);
    }
  }

  free(Tmp_Vec_3);
#endif

  /***************************************************************
                  DM and iDM: density matrix

    DM_{alpha,alpha} = CDM[0] + I*iDM[0]
    DM_{alpha,beta} = CDM[2] + I*CDM[3]
    DM_{beta,alpha} = CDM[2] - I*CDM[3]
    DM_{beta,beta} = CDM[1] + I*iDM[1]
  ****************************************************************/

  /* DM */

  for (spin=0; spin<=SpinP_switch; spin++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID = G2ID[Gc_AN];

      if (myid==ID){

        num = 0;

        Mc_AN = F_G2M[Gc_AN];
        wan1 = WhatSpecies[Gc_AN];
        TNO1 = Spe_Total_CNO[wan1];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          wan2 = WhatSpecies[Gh_AN];
          TNO2 = Spe_Total_CNO[wan2];

          for (i=0; i<TNO1; i++){
            for (j=0; j<TNO2; j++){
              Tmp_Vec[num] = DM[0][spin][Mc_AN][h_AN][i][j];
              num++;
	    }
          }
        }

        if (myid!=Host_ID){
          MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
          MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
	}
        else{
          fwrite(Tmp_Vec, sizeof(double), num, fp);
        }
      }

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
        MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
        fwrite(Tmp_Vec, sizeof(double), num, fp);
      }

    }  
  }

  /* iDM */

  for (spin=0; spin<2; spin++){
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID = G2ID[Gc_AN];

      if (myid==ID){

        num = 0;

        Mc_AN = F_G2M[Gc_AN];
        wan1 = WhatSpecies[Gc_AN];
        TNO1 = Spe_Total_CNO[wan1];
        for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
          Gh_AN = natn[Gc_AN][h_AN];
          wan2 = WhatSpecies[Gh_AN];
          TNO2 = Spe_Total_CNO[wan2];

          for (i=0; i<TNO1; i++){
            for (j=0; j<TNO2; j++){
              Tmp_Vec[num] = iDM[0][spin][Mc_AN][h_AN][i][j];
              num++;
	    }
          }
        }

        if (myid!=Host_ID){
          MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
          MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&stat);
	}
        else{
          fwrite(Tmp_Vec, sizeof(double), num, fp);
        }
      }

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
        MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
        fwrite(Tmp_Vec, sizeof(double), num, fp);
      }

    }  
  }

  /* freeing of Tmp_Vec */
  free(Tmp_Vec);

  if (myid==Host_ID){

    /****************************************************
        Solver
    ****************************************************/

    if (PeriodicGamma_flag==1)  i_vec[0] = 3;
    else                        i_vec[0] = Solver;
    fwrite(i_vec,sizeof(int),1,fp);

    /****************************************************
        ChemP
        Electronic Temp
        dipole moment (x, y, z) from core charge
        dipole moment (x, y, z) from back ground charge
    ****************************************************/

    d_vec[0] = ChemP;
    d_vec[1] = E_Temp*eV2Hartree;
    d_vec[2] = dipole_moment[1][1]; 
    d_vec[3] = dipole_moment[1][2];
    d_vec[4] = dipole_moment[1][3];
    d_vec[5] = dipole_moment[3][1];
    d_vec[6] = dipole_moment[3][2];
    d_vec[7] = dipole_moment[3][3];
    d_vec[8] = Total_Num_Electrons;
    d_vec[9] = Total_SpinS;
    fwrite(d_vec,sizeof(double),10,fp);
  }

  /****************************************************
      input file
  ****************************************************/

  if (myid==Host_ID){
    
    /* Added by N. Yamaguchi ***/
    if (inputfile=='\0') {
      return;
    }
    /* ***/

    /* find the number of lines in the input file */

    if ((fp_inp = fopen(inputfile,"r")) != NULL){

      setvbuf(fp_inp,buf,_IOFBF,fp_bsize);  /* setvbuf */
      count = 0;
      /* fgets gets a carriage return */
      while( fgets(strg, MAX_LINE_SIZE, fp_inp ) != NULL ){
	count++;
      }

      fclose(fp_inp);
    }
    else{
      printf("error #1 in saving *.scfout\n"); 
    }

    /* write the input file */

    if ((fp_inp = fopen(inputfile,"r")) != NULL){

      setvbuf(fp_inp,buf,_IOFBF,fp_bsize);  /* setvbuf */
      i_vec[0] = count;
      fwrite(i_vec, sizeof(int), 1, fp);

      /* fgets gets a carriage return */
      while( fgets(strg, MAX_LINE_SIZE, fp_inp ) != NULL ){
	fwrite(strg, sizeof(char), MAX_LINE_SIZE, fp);
      }

      fclose(fp_inp);
    }
    else{
      printf("error #1 in saving *.scfout\n"); 
    }
  }

}





void Calc_OLPpo()
{
  double time0;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int i,j,Cwan,Hwan,NO0,NO1,spinmax;
  int Rnh,Rnk,spin,N,NumC[4];
  int n1,n2,n3,L0,Mul0,M0,L1,Mul1,M1;
  int Nc,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double x,y,z,dx,dy,dz,tmpx,tmpy,tmpz;
  double bc,dv,r,theta,phi,sum,tmp0,tmp1;
  double xo,yo,zo,S_coordinate[3];
  double Cxyz[4];
  double **ChiVx;
  double **ChiVy;
  double **ChiVz;
  double *tmp_ChiVx;
  double *tmp_ChiVy;
  double *tmp_ChiVz;
  double *tmp_Orbs_Grid;
  double **tmp_OLPpox;
  double **tmp_OLPpoy;
  double **tmp_OLPpoz;
  double TStime,TEtime;
  double TStime0,TEtime0;
  double TStime1,TEtime1;
  double TStime2,TEtime2;
  double TStime3,TEtime3;
  int numprocs,myid,tag=999,ID;
  double Stime_atom, Etime_atom;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  ChiVx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVx[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  ChiVy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVy[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  ChiVz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVz[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  tmp_ChiVx = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  tmp_ChiVy = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  tmp_ChiVz = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_Orbs_Grid = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_OLPpox = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPpox[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  tmp_OLPpoy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPpoy[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  tmp_OLPpoz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLPpoz[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  /*****************************************************
     matrix elements for OLPpo (added by N. Yamaguchi)
  *****************************************************/

  int order;
  for (order=0; order<order_max; order++){
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      dtime(&Stime_atom);

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      NO0 = Spe_Total_CNO[Cwan];

      for (i=0; i<NO0; i++){
	for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

	  GNc = GridListAtom[Mc_AN][Nc];
	  GRc = CellListAtom[Mc_AN][Nc];

	  Get_Grid_XYZ(GNc,Cxyz);
	  x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1];
	  y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2];
	  z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	  int power;
	  double xn=1, yn=1, zn=1;
	  for (power=0; power<=order; power++){
	    xn*=x;
	    yn*=y;
	    zn*=z;
	  }

	  ChiVx[i][Nc] = xn*Orbs_Grid[Mc_AN][Nc][i];
	  ChiVy[i][Nc] = yn*Orbs_Grid[Mc_AN][Nc][i];
	  ChiVz[i][Nc] = zn*Orbs_Grid[Mc_AN][Nc][i];
	}
      }

      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

	Gh_AN = natn[Gc_AN][h_AN];
	Mh_AN = F_G2M[Gh_AN];

	Rnh = ncn[Gc_AN][h_AN];
	Hwan = WhatSpecies[Gh_AN];
	NO1 = Spe_Total_CNO[Hwan];

	/* initialize */

	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){
	    tmp_OLPpox[i][j] = 0.0;
	    tmp_OLPpoy[i][j] = 0.0;
	    tmp_OLPpoz[i][j] = 0.0;
	  }
	}

	/* summation of non-zero elements */

	for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

	  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

	  /* store ChiVx,y,z in tmp_ChiVx,y,z */

	  for (i=0; i<NO0; i++){
	    tmp_ChiVx[i] = ChiVx[i][Nc];
	    tmp_ChiVy[i] = ChiVy[i][Nc];
	    tmp_ChiVz[i] = ChiVz[i][Nc];
	  }

	  /* store Orbs_Grid in tmp_Orbs_Grid */

	  if (G2ID[Gh_AN]==myid){
	    for (j=0; j<NO1; j++){
	      tmp_Orbs_Grid[j] = Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */
	    }
	  }
	  else{
	    for (j=0; j<NO1; j++){
	      tmp_Orbs_Grid[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */
	    }
	  }

	  /* integration */

	  for (i=0; i<NO0; i++){
	    tmpx = tmp_ChiVx[i];
	    tmpy = tmp_ChiVy[i];
	    tmpz = tmp_ChiVz[i];
	    for (j=0; j<NO1; j++){
	      tmp_OLPpox[i][j] += tmpx*tmp_Orbs_Grid[j];
	      tmp_OLPpoy[i][j] += tmpy*tmp_Orbs_Grid[j];
	      tmp_OLPpoz[i][j] += tmpz*tmp_Orbs_Grid[j];
	    }
	  }
	}

	/* OLPpoxn, OLPpoyn, OLPpozn */

	for (i=0; i<NO0; i++){
	  for (j=0; j<NO1; j++){
	    OLPpo[0][order][Mc_AN][h_AN][i][j] = tmp_OLPpox[i][j]*GridVol;
	    OLPpo[1][order][Mc_AN][h_AN][i][j] = tmp_OLPpoy[i][j]*GridVol;
	    OLPpo[2][order][Mc_AN][h_AN][i][j] = tmp_OLPpoz[i][j]*GridVol;
	  }
	}

      }

      dtime(&Etime_atom);
      time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
    }
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVx[i]);
  }
  free(ChiVx);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVy[i]);
  }
  free(ChiVy);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVz[i]);
  }
  free(ChiVz);

  free(tmp_ChiVx);
  free(tmp_ChiVy);
  free(tmp_ChiVz);

  free(tmp_Orbs_Grid);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPpox[i]);
  }
  free(tmp_OLPpox);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPpoy[i]);
  }
  free(tmp_OLPpoy);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLPpoz[i]);
  }
  free(tmp_OLPpoz);
}


void Calc_OLPmo()
{
  double time0;
  int Mc_AN,Gc_AN,Mh_AN,h_AN,Gh_AN;
  int i,j,Cwan,Hwan,NO0,NO1,spinmax;
  int Rnh,Rnk,spin,N,NumC[4];
  int n1,n2,n3,L0,Mul0,M0,L1,Mul1,M1;
  int Nc,GNc,GRc,Nog,Nh,MN,XC_P_switch;
  double x,y,z,dx,dy,dz,tmpx,tmpy,tmpz;
  double bc,dv,r,theta,phi,sum,tmp0,tmp1;
  double xo,yo,zo,S_coordinate[3];
  double Cxyz[4];
  double **dorbs;
  double **ChiVx;
  double **ChiVy;
  double **ChiVz;
  double *tmp_ChiVx;
  double *tmp_ChiVy;
  double *tmp_ChiVz;
  double *tmp_Orbs_Grid;
  double **tmp_OLP_px;
  double **tmp_OLP_py;
  double **tmp_OLP_pz;
  double TStime,TEtime;
  double TStime0,TEtime0;
  double TStime1,TEtime1;
  double TStime2,TEtime2;
  double TStime3,TEtime3;
  int numprocs,myid,tag=999,ID;
  double Stime_atom, Etime_atom;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  dorbs = (double**)malloc(sizeof(double*)*4);
  for (i=0; i<4; i++){
    dorbs[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  ChiVx = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVx[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  ChiVy = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVy[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  ChiVz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    ChiVz[i] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
  }

  tmp_ChiVx = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  tmp_ChiVy = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  tmp_ChiVz = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_Orbs_Grid = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  tmp_OLP_px = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLP_px[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  tmp_OLP_py = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLP_py[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  tmp_OLP_pz = (double**)malloc(sizeof(double*)*List_YOUSO[7]);
  for (i=0; i<List_YOUSO[7]; i++){
    tmp_OLP_pz[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  }

  /*****************************************************
      calculation of matrix elements for OLP_px,y,z
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];
    
    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

      GNc = GridListAtom[Mc_AN][Nc];
      GRc = CellListAtom[Mc_AN][Nc];

      Get_Grid_XYZ(GNc,Cxyz);
      x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
      y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
      z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

      Get_dOrbitals(Cwan,x,y,z,dorbs); 

      for (i=0; i<NO0; i++){
	ChiVx[i][Nc] = dorbs[1][i];
	ChiVy[i][Nc] = dorbs[2][i];
	ChiVz[i][Nc] = dorbs[3][i];
      }
    }
    
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];

      Rnh = ncn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      NO1 = Spe_Total_CNO[Hwan];

      /* initialize */

      for (i=0; i<NO0; i++){
	for (j=0; j<NO1; j++){
	  tmp_OLP_px[i][j] = 0.0;
	  tmp_OLP_py[i][j] = 0.0;
	  tmp_OLP_pz[i][j] = 0.0;
	}
      }

      /* summation of non-zero elements */

      for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){

        Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
        Nh = GListTAtoms2[Mc_AN][h_AN][Nog];

        /* store ChiVx,y,z in tmp_ChiVx,y,z */

        for (i=0; i<NO0; i++){
          tmp_ChiVx[i] = ChiVx[i][Nc];
          tmp_ChiVy[i] = ChiVy[i][Nc];
          tmp_ChiVz[i] = ChiVz[i][Nc];
	}

        /* store Orbs_Grid in tmp_Orbs_Grid */

        if (G2ID[Gh_AN]==myid){
	  for (j=0; j<NO1; j++){
	    tmp_Orbs_Grid[j] = Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */
	  }
	}
        else{
	  for (j=0; j<NO1; j++){
	    tmp_Orbs_Grid[j] = Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */ 
	  }
        }

        /* integration */

        for (i=0; i<NO0; i++){
          tmpx = tmp_ChiVx[i]; 
          tmpy = tmp_ChiVy[i]; 
          tmpz = tmp_ChiVz[i]; 
          for (j=0; j<NO1; j++){
            tmp_OLP_px[i][j] += tmpx*tmp_Orbs_Grid[j];
            tmp_OLP_py[i][j] += tmpy*tmp_Orbs_Grid[j];
            tmp_OLP_pz[i][j] += tmpz*tmp_Orbs_Grid[j];
	  }
	}
      }

      /* OLP_px,y,z */

      for (i=0; i<NO0; i++){
        for (j=0; j<NO1; j++){

          /* <ialpha|<-p |jbeta>, p was operated to <ialpha|. 
             Thus, we take conjugate complex */

          OLPmo[0][Mc_AN][h_AN][i][j] =-tmp_OLP_px[i][j]*GridVol;
          OLPmo[1][Mc_AN][h_AN][i][j] =-tmp_OLP_py[i][j]*GridVol;
          OLPmo[2][Mc_AN][h_AN][i][j] =-tmp_OLP_pz[i][j]*GridVol;
        }
      }

    }

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }
  
  /****************************************************
    freeing of arrays:
  ****************************************************/

  for (i=0; i<4; i++){
    free(dorbs[i]);
  }
  free(dorbs);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVx[i]);
  }
  free(ChiVx);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVy[i]);
  }
  free(ChiVy);

  for (i=0; i<List_YOUSO[7]; i++){
    free(ChiVz[i]);
  }
  free(ChiVz);

  free(tmp_ChiVx);
  free(tmp_ChiVy);
  free(tmp_ChiVz);

  free(tmp_Orbs_Grid);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLP_px[i]);
  }
  free(tmp_OLP_px);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLP_py[i]);
  }
  free(tmp_OLP_py);

  for (i=0; i<List_YOUSO[7]; i++){
    free(tmp_OLP_pz[i]);
  }
  free(tmp_OLP_pz);
}
