/**********************************************************************
  TRAN_Output_HKS.c:

  TRAN_Output_HKS.c is a subroutine to save and load data on leads

  Log of TRAN_Output_HKS.c:

     11/Dec/2005   Released by H. Kino

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_common.h"
#include <mpi.h>
#include "tran_variables.h"
#include "tran_prototypes.h" 


/****************************************************************
   purpose: save and load data to restart without input file 

   from RestartFileDFT.c
------------------------------------------

"read" mode allocates 

int *WhatSpecies_l;
int *Spe_Total_CNO_l;
int *Spe_Total_NO_l;
int *FNAN_l;
int **natn_l;
int **ncn_l;
int **atv_ijk_l;

double *****OLP_l;
double *****H_l;
double ******DM_l;

or 

int *WhatSpecies_r;
int *Spe_Total_CNO_r;
int *Spe_Total_NO_r;
int *FNAN_r;
int **natn_r;
int **ncn_r;
int **atv_ijk_r;

double *****OLP_r;
double *****H_r;
double ******DM_r;


*****************************************************************/

/*
  e.g. Overlap_Band, which gives hints of data to be saved
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
  GA_AN = M2G[MA_AN];  
  wanA = WhatSpecies[GA_AN]; int* WhatSpecies[atomnum+1]
  tnoA = Spe_Total_CNO[wanA]; int* Spe_Total_CNO[SpeciesNum]
  Anum = MP[GA_AN];   int *MP, neglect!

  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
  GB_AN = natn[GA_AN][LB_AN]; int** natn[atomnum+1][Max_FSNAN*ScaleSize+1]
  Rn = ncn[GA_AN][LB_AN];  int** ncn[atomnum+1][Max_FSNAN*ScaleSize+1]
  wanB = WhatSpecies[GB_AN]; 
  tnoB = Spe_Total_CNO[wanB];

  l1 = atv_ijk[Rn][1];  int** atv_ijk[TCpyCell+1][4];
  l2 = atv_ijk[Rn][2];
  l3 = atv_ijk[Rn][3];
  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

  si = sin(2.0*PI*kRn);
  co = cos(2.0*PI*kRn);
  Bnum = MP[GB_AN];
  for (i=0; i<tnoA; i++){
  for (j=0; j<tnoB; j++){
  s = OLP[MA_AN][LB_AN][i][j]; double****
  size: OLP[4]
  [Matomnum+MatomnumF+MatomnumS+1]
  [FNAN[Gc_AN]+1]
  [Spe_Total_NO[Cwan]]
  [Spe_Total_NO[Hwan]]

  int *Spe_Total_NO  Spe_Total_NO[SpeciesNum]
*/


/***************************************************************************/

int TRAN_Output_HKS(char *fileHKS)
{
  FILE *fp;
  int i_vec[100],i2_vec[2];
  double *d_vec;
  int i,id,j;
  int size1,size,vsize;
  int *ia_vec;
  int Gc_AN,Mc_AN, tno0, Cwan, h_AN, tno1, Gh_AN, Hwan,k,m,N;

  int myid,numprocs;
  int tag=99;
  double *v;

  MPI_Status status;
  MPI_Request request;

  MPI_Comm_rank(mpi_comm_level1, &myid);
  MPI_Comm_size(mpi_comm_level1, &numprocs);
   
  if (myid==Host_ID) {

    d_vec = (double*)malloc(sizeof(double)*(3*(atomnum+1)+100));

    /* make a filename */
    if ( (fp=fopen(fileHKS,"w"))==NULL) {
      printf("can not open %s\n",fileHKS);
      exit(0);
    }

    /* save data to the file (*fp) */

    /* parameter to allocate memory */
    i=0;
    i_vec[i++]=1; /* major version */
    i_vec[i++]=0; /* minor version*/

    fwrite(i_vec,sizeof(int),i,fp);

    i=0;
    i_vec[i++]= SpinP_switch;
    i_vec[i++]= atomnum;
    i_vec[i++]= SpeciesNum;
    i_vec[i++]= Max_FSNAN;
    i_vec[i++]= TCpyCell;
    i_vec[i++]= Matomnum;
    i_vec[i++]= MatomnumF;
    i_vec[i++]= MatomnumS;
    i_vec[i++]= Ngrid1;
    i_vec[i++]= Ngrid2;
    i_vec[i++]= Ngrid3;
    i_vec[i++]= Num_Cells0; 

    fwrite(i_vec,sizeof(int),i,fp);

    id=0;
    d_vec[id++]= ScaleSize;
    for(j=1;j<=3;j++) {
      for (k=1;k<=3;k++) {
	d_vec[id++]= tv[j][k];
      }
    }
    for(j=1;j<=3;j++) {
      for (k=1;k<=3;k++) {
	d_vec[id++]= gtv[j][k];
      }
    }
    for (i=1;i<=3;i++) {
      d_vec[id++] = Grid_Origin[i];
    }

    for (i=1; i<=atomnum; i++) {
      for (j=1; j<=3; j++) {
        d_vec[id++] = Gxyz[i][j];
      }
    }

    d_vec[id++] = ChemP; 

    fwrite(d_vec,sizeof(double),id,fp);

    /*  data in arrays */
    fwrite(WhatSpecies, sizeof(int), atomnum+1, fp);
    fwrite(Spe_Total_CNO, sizeof(int), SpeciesNum, fp);
    fwrite(Spe_Total_NO,sizeof(int),SpeciesNum,fp);

    fwrite(FNAN,sizeof(int),atomnum+1,fp);

    size1=(int)Max_FSNAN*ScaleSize+1;
    for (i=0;i<= atomnum; i++) {
      fwrite(natn[i],sizeof(int),size1,fp);
    }
    for (i=0;i<= atomnum; i++) {
      fwrite(ncn[i],sizeof(int),size1,fp);
    }

    /*  printf("atv_ijk\n"); */

    size1=(TCpyCell+1)*4;
    ia_vec=(int*)malloc(sizeof(int)*size1);
    id=0;
    for (i=0;i<TCpyCell+1;i++) {
      for (j=0;j<=3;j++) {
	ia_vec[id++]=atv_ijk[i][j];
      }
    }
    fwrite(ia_vec,sizeof(int),size1,fp);
    free(ia_vec);

    /* freeing of d_vec */

    free(d_vec);

  }  /* if (myid==Host_ID) */

  /* allocate v */

  v = (double*)malloc(sizeof(double)*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);

  /* OLP,  this is complex */

  for (k=0;k<4;k++) {

    int m,ID;

    /*global  Gc_AN  1:atomnum */
    /*variable ID = G2ID[Gc_AN] */
    /*variable Mc_AN = G2M[Gc_AN] */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      Mc_AN = F_G2M[Gc_AN];
      ID = G2ID[Gc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      /* OLP into v */

      if (myid==ID) {

	vsize = 0;

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
	    for (j=0;j<tno1;j++) {
	      v[vsize] =  OLP[k][Mc_AN][h_AN][i][j];
	      vsize++;
	    }
	  }
	} 

        /* Isend */

        if (myid!=Host_ID){
          MPI_Isend(&vsize, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&status);
          MPI_Isend(&v[0], vsize, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&status);
	}
        else{
          fwrite(v, sizeof(double), vsize, fp);
        }
      }

      /* Recv */

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&vsize, 1, MPI_INT, ID, tag, mpi_comm_level1, &status);
        MPI_Recv(&v[0], vsize, MPI_DOUBLE, ID, tag, mpi_comm_level1, &status);
        fwrite(v, sizeof(double), vsize, fp);
      }

    } /* Gc_AN */
  } /* k */


  /* H */
  for (k=0; k<=SpinP_switch; k++){

    int ID;

    /*global  Gc_AN  1:atomnum */
    /*variable ID = G2ID[Gc_AN] */
    /*variable Mc_AN = G2M[Gc_AN] */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID    = G2ID[Gc_AN]; 
      Mc_AN = F_G2M[Gc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];  

      /* H into v */

      if (myid==ID) {

	vsize = 0;

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
	    for (j=0;j<tno1; j++) {
	      v[vsize] = H[k][Mc_AN][h_AN][i][j];
	      vsize++;
	    }
	  }
	}

        /* Isend */

        if (myid!=Host_ID){
          MPI_Isend(&vsize, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&status);
          MPI_Isend(&v[0], vsize, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&status);
	}
        else{
          fwrite(v, sizeof(double), vsize, fp);
        }
      }

      /* Recv */

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&vsize, 1, MPI_INT, ID, tag, mpi_comm_level1, &status);
        MPI_Recv(&v[0], vsize, MPI_DOUBLE, ID, tag, mpi_comm_level1, &status);
        fwrite(v, sizeof(double), vsize, fp);
      }

    }
  } /* k */

/* revised by Y. Xiao for Noncollinear NEGF calculations */
  /* iHNL */
if(SpinP_switch==3) {

  for (k=0; k<=2; k++){

    int ID;

    /*global  Gc_AN  1:atomnum */
    /*variable ID = G2ID[Gc_AN] */
    /*variable Mc_AN = G2M[Gc_AN] */

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

      ID    = G2ID[Gc_AN];
      Mc_AN = F_G2M[Gc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];

      /* H into v */

      if (myid==ID) {

        vsize = 0;

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
            for (j=0;j<tno1; j++) {
              v[vsize] = iHNL[k][Mc_AN][h_AN][i][j];
              vsize++;
            }
          }
        }

        /* Isend */

        if (myid!=Host_ID){
          MPI_Isend(&vsize, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&status);
          MPI_Isend(&v[0], vsize, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
          MPI_Wait(&request,&status);
        }
        else{
          fwrite(v, sizeof(double), vsize, fp);
        }
      }

      /* Recv */

      else if (ID!=myid && myid==Host_ID){
        MPI_Recv(&vsize, 1, MPI_INT, ID, tag, mpi_comm_level1, &status);
        MPI_Recv(&v[0], vsize, MPI_DOUBLE, ID, tag, mpi_comm_level1, &status);
        fwrite(v, sizeof(double), vsize, fp);
      }

    }
  } /* k */
 } /* if (SpinP_switch==3)  */
/* until here by Y. Xiao for Noncollinear NEGF calculations */

  /* DM */

  for (m=0; m<1; m++){

    int ID;

    for (k=0; k<=SpinP_switch; k++){

      /*global  Gc_AN  1:atomnum */
      /*variable ID = G2ID[Gc_AN] */
      /*variable Mc_AN = G2M[Gc_AN] */

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

        ID = G2ID[ Gc_AN]; 
	Mc_AN = F_G2M[Gc_AN];
	Cwan = WhatSpecies[Gc_AN];
	tno0 = Spe_Total_NO[Cwan];  

        /* DM into v */

        if (myid==ID) {

    	  vsize = 0;

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
	      for (j=0;j<tno1;j++) {
		v[vsize] = DM[m][k][Mc_AN][h_AN][i][j] ;
  	        vsize++;
	      }
	    }
	  }

          /* Isend */

	  if (myid!=Host_ID){
	    MPI_Isend(&vsize, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
	    MPI_Wait(&request,&status);
	    MPI_Isend(&v[0], vsize, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
	    MPI_Wait(&request,&status);
	  }
	  else{
	    fwrite(v, sizeof(double), vsize, fp);
	  }
	}

        /* Recv */

        else if (ID!=myid && myid==Host_ID){
          MPI_Recv(&vsize, 1, MPI_INT, ID, tag, mpi_comm_level1, &status);
          MPI_Recv(&v[0], vsize, MPI_DOUBLE, ID, tag, mpi_comm_level1, &status);
          fwrite(v, sizeof(double), vsize, fp);
        }

      } /* Gc_AN */
    } /* k */
  } /* m */


  /* free v */

  free(v);

  /* Density_Grid[0] + Density_Grid[1] - 2.0*ADensity_Grid */

  TRAN_Output_HKS_Write_Grid(mpi_comm_level1, 
                             1,
 	                     Ngrid1,Ngrid2,Ngrid3,
			     Density_Grid_B[0],Density_Grid_B[1],ADensity_Grid_B,fp);
  /* Vpot_Grid */

  TRAN_Output_HKS_Write_Grid(mpi_comm_level1, 
                             0, 
			     Ngrid1,Ngrid2,Ngrid3,
			     dVHart_Grid_B,NULL,NULL,fp);

  if (myid==Host_ID) fclose(fp);

  return 1;

}



	  
  
