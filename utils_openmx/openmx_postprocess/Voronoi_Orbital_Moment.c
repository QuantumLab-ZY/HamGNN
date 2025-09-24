/**********************************************************************
  Voronoi_Orbital_Moment.c:

   Voronoi_Orbital_Moment.c is a subroutine to calculate orbital 
   magnetic moment at each atomic site b a Voronoi decomposition 
   method

  Log of Voronoi_Orbital_Moment.c:

    23/Nov./2006  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"


void Orbitals_on_Grid(int wan, int Lmax, int Nmul, double x, double y, double z, 
                      double ***Chi, double **RF, double **AF);

void Generate_L_Matrix(int Lmax, double ***Lx0, double ***Ly0, double ***Lz0);


void Voronoi_Orbital_Moment()
{
  int Lmax,Nmul;
  int L,M,p,Mc_AN,Gc_AN,tno0,tno1;
  int i,j,k,Cwan,h_AN,Mh_AN,Nog,Nc,Nh;
  int n,num,tno2,size1,size2,ML,MR;
  int hR_AN,hL_AN,MR_AN,ML_AN,kl,m,GL_AN,GR_AN;
  int GNc,GRc,Gh_AN,Hwan,wan1,Lwan,Rwan;
  FILE *fp_VOM;
  char file_VOM[YOUSO10];
  char buf[fp_bsize];          /* setvbuf */
  double sum,FuzzyW,dx,dy,dz,x,y,z;
  double idm0,idm1,MagL,tmp;
  double sumx,sumy,sumz,lx,ly,lz;
  double Cxyz[4];
  double ***Chi0,****Tmp_Orb;
  double **RF,**AF;
  double ******WOLP;
  double *****iDM0;
  double *tmp_array;
  double *tmp_array2;
  double ***Lx0,***Ly0,***Lz0;
  int *Snd_DM0_Size,*Rcv_DM0_Size;
  int numprocs,myid,ID,IDS,IDR,tag=999;

  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID) printf("\n<Voronoi_Orbital_Moment>  calculate Voronoi orbital moment\n");fflush(stdout);

  Lmax = 3;
  Nmul = 4;

  /****************************************************
    allocation of array:
  ****************************************************/

  Chi0 = (double***)malloc(sizeof(double**)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    Chi0[L] = (double**)malloc(sizeof(double*)*(2*L+1));
    for (M=0; M<(2*L+1); M++){
      Chi0[L][M] = (double*)malloc(sizeof(double)*Nmul);
      for (p=0; p<Nmul; p++) Chi0[L][M][p] = 0.0;
    }
  }

  Tmp_Orb = (double****)malloc(sizeof(double***)*(Lmax+1));
  for (L=0; L<=Lmax; L++){
    Tmp_Orb[L] = (double***)malloc(sizeof(double**)*(2*L+1));
    for (M=0; M<(2*L+1); M++){
      Tmp_Orb[L][M] = (double**)malloc(sizeof(double*)*Nmul);
      for (p=0; p<Nmul; p++){
        Tmp_Orb[L][M][p] = (double*)malloc(sizeof(double)*List_YOUSO[11]);
      }
    }
  }

  RF = (double**)malloc(sizeof(double*)*(Lmax+1));
  for (i=0; i<(Lmax+1); i++){
    RF[i] = (double*)malloc(sizeof(double)*Nmul);
  }

  AF = (double**)malloc(sizeof(double*)*(Lmax+1));
  for (i=0; i<(Lmax+1); i++){
    AF[i] = (double*)malloc(sizeof(double)*(2*(Lmax+1)+1));
  }

  WOLP = (double******)malloc(sizeof(double*****)*(Matomnum+1));
  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];  
    }    

    WOLP[Mc_AN] = (double*****)malloc(sizeof(double****)*(FNAN[Gc_AN]+1));
     for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      WOLP[Mc_AN][h_AN] = (double****)malloc(sizeof(double***)*(Lmax+1));
      for (L=0; L<=Lmax; L++){
        WOLP[Mc_AN][h_AN][L] = (double***)malloc(sizeof(double**)*(2*L+1));
        for (M=0; M<(2*L+1); M++){
          WOLP[Mc_AN][h_AN][L][M] = (double**)malloc(sizeof(double*)*Nmul);
          for (p=0; p<Nmul; p++){
            WOLP[Mc_AN][h_AN][L][M][p] = (double*)malloc(sizeof(double)*(List_YOUSO[7]+1));
	  }
	}
      }
    }
  }

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

  /* Snd_DM0_Size and Rcv_DM0_Size */

  Snd_DM0_Size = (int*)malloc(sizeof(int)*numprocs);
  Rcv_DM0_Size = (int*)malloc(sizeof(int)*numprocs);

  /* Lx0, Ly0, Lz0 */
 
  Lx0 = (double***)malloc(sizeof(double**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    Lx0[L] = (double**)malloc(sizeof(double*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      Lx0[L][M] = (double*)malloc(sizeof(double)*(2*L+1)); 
    }
  }

  Ly0 = (double***)malloc(sizeof(double**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    Ly0[L] = (double**)malloc(sizeof(double*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      Ly0[L][M] = (double*)malloc(sizeof(double)*(2*L+1)); 
    }
  }

  Lz0 = (double***)malloc(sizeof(double**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    Lz0[L] = (double**)malloc(sizeof(double*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      Lz0[L][M] = (double*)malloc(sizeof(double)*(2*L+1)); 
    }
  }
  
  /****************************************************
                generate the L-matrix
  ****************************************************/

  Generate_L_Matrix(Lmax,Lx0,Ly0,Lz0); 

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

  /*****************************************************
   *****************************************************
   *****************************************************

      MPI: iDM

  *****************************************************
  *****************************************************
  *****************************************************/

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

  /*****************************************************
   *****************************************************
   *****************************************************

           calculate a weighted overlap matrix

  *****************************************************
  *****************************************************
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    /****************************************
     calculate basis functions on atom Mc_AN,
     store them into Tmp_Orb 
    ****************************************/

    for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

      GNc = GridListAtom[Mc_AN][Nc]; 
      GRc = CellListAtom[Mc_AN][Nc];

      Get_Grid_XYZ(GNc,Cxyz);
      x = Cxyz[1] + atv[GRc][1]; 
      y = Cxyz[2] + atv[GRc][2]; 
      z = Cxyz[3] + atv[GRc][3];

      dx = x - Gxyz[Gc_AN][1];
      dy = y - Gxyz[Gc_AN][2];
      dz = z - Gxyz[Gc_AN][3];

      FuzzyW = Fuzzy_Weight(Gc_AN,Mc_AN,0,x,y,z);
      Orbitals_on_Grid(Cwan,Lmax,Nmul,dx,dy,dz,Chi0,RF,AF);

      for (L=0; L<=Lmax; L++){
	for (M=0; M<=(2*L); M++){
	  for (p=0; p<Nmul; p++){
            Tmp_Orb[L][M][p][Nc] = FuzzyW*Chi0[L][M][p];
	  }
	}
      }
    }

    /****************************************
     calculate the weighted overlap matrix
    ****************************************/

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Mh_AN = F_G2M[Gh_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_CNO[Hwan];

      for (L=0; L<=Lmax; L++){
        for (M=0; M<=(2*L); M++){
          for (p=0; p<Nmul; p++){

	    for (j=0; j<tno1; j++){

              sum = 0.0;

              if (G2ID[Gh_AN]==myid){
		for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){
		  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
		  Nh = GListTAtoms2[Mc_AN][h_AN][Nog];
		  sum += Tmp_Orb[L][M][p][Nc]*Orbs_Grid[Mh_AN][Nh][j];/* AITUNE */
		}
	      }
              else{
		for (Nog=0; Nog<NumOLG[Mc_AN][h_AN]; Nog++){
		  Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
		  sum += Tmp_Orb[L][M][p][Nc]*Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j];/* AITUNE */
		}
              } 

              WOLP[Mc_AN][h_AN][L][M][p][j] = sum*GridVol;
            } 
	  }          
        }
      }
    }
  }

  /*****************************************************
   *****************************************************
   *****************************************************

      calculate the Voronoi orbital magnetic moment
   
  *****************************************************
  *****************************************************
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    lx = 0.0;
    ly = 0.0;
    lz = 0.0;

    for (hL_AN=0; hL_AN<=FNAN[Gc_AN]; hL_AN++){

      GL_AN = natn[Gc_AN][hL_AN];
      ML_AN = F_G2M[GL_AN];
      Lwan = WhatSpecies[GL_AN];

      for (hR_AN=0; hR_AN<=FNAN[Gc_AN]; hR_AN++){
        GR_AN = natn[Gc_AN][hR_AN];
        MR_AN = F_G2M[GR_AN];
        Rwan = WhatSpecies[GR_AN];
        kl = RMI1[Mc_AN][hL_AN][hR_AN];

        if (0<=kl) {

	  for (m=0; m<Spe_Total_CNO[Lwan]; m++){
	    for (n=0; n<Spe_Total_CNO[Rwan]; n++){

              sumx = 0.0;
              sumy = 0.0;
              sumz = 0.0;

              for (L=0; L<=Lmax; L++){
                for (ML=0; ML<=(2*L); ML++){
                  for (MR=0; MR<=(2*L); MR++){
                    for (p=0; p<Nmul; p++){

                      tmp = WOLP[Mc_AN][hL_AN][L][ML][p][m]*WOLP[Mc_AN][hR_AN][L][MR][p][n];

                      sumx += tmp*Lx0[L][ML][MR];
                      sumy += tmp*Ly0[L][ML][MR];
                      sumz += tmp*Lz0[L][ML][MR];
		    }
 		  }
		}
	      }

              idm0 = iDM0[0][ML_AN][kl][m][n];
              idm1 = iDM0[1][ML_AN][kl][m][n];

              lx -= (idm0 + idm1)*sumx;
              ly -= (idm0 + idm1)*sumy;
              lz -= (idm0 + idm1)*sumz;

	    } /* n */
	  } /* m */
	}  /* if (0<=kl) */
      } /* hR_AN */
    } /* hL_AN */ 

    MagL = sqrt(lx*lx + ly*ly + lz*lz);

    printf("Gc_AN=%2d |L|=%15.12f lx=%15.12f ly=%15.12f lz=%15.12f\n",Gc_AN,MagL,lx,ly,lz); 

  } /* Mc_AN */  

  /****************************************************
    freeing of array:
  ****************************************************/

  for (L=0; L<=Lmax; L++){
    for (M=0; M<(2*L+1); M++){
      free(Chi0[L][M]);
    }
    free(Chi0[L]);
  }
  free(Chi0);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<(2*L+1); M++){
      for (p=0; p<Nmul; p++){
        free(Tmp_Orb[L][M][p]);
      }
      free(Tmp_Orb[L][M]);
    }
    free(Tmp_Orb[L]);
  }
  free(Tmp_Orb);

  for (i=0; i<(Lmax+1); i++){
    free(RF[i]);
  }
  free(RF);

  for (i=0; i<(Lmax+1); i++){
    free(AF[i]);
  }
  free(AF);

  for (Mc_AN=0; Mc_AN<=Matomnum; Mc_AN++){

    if (Mc_AN==0){
      Gc_AN = 0;
      tno0 = 1;
    }
    else{
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      tno0 = Spe_Total_NO[Cwan];  
    }    

     for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      for (L=0; L<=Lmax; L++){
        for (M=0; M<(2*L+1); M++){
          for (p=0; p<Nmul; p++){
            free(WOLP[Mc_AN][h_AN][L][M][p]);
	  }
          free(WOLP[Mc_AN][h_AN][L][M]);
	}
        free(WOLP[Mc_AN][h_AN][L]);
      }
      free(WOLP[Mc_AN][h_AN]);
    }
    free(WOLP[Mc_AN]);
  }
  free(WOLP);

  /* freeing of iDM0 */

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

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(Lx0[L][M]);
    }
    free(Lx0[L]);
  }
  free(Lx0);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(Ly0[L][M]);
    }
    free(Ly0[L]);
  }
  free(Ly0);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(Lz0[L][M]);
    }
    free(Lz0[L]);
  }
  free(Lz0);

  /* Snd_DM0_Size and Rcv_DM0_Size */

  free(Snd_DM0_Size);
  free(Rcv_DM0_Size);
}







void Orbitals_on_Grid(int wan, int Lmax, int Nmul, double x, double y, double z, 
                      double ***Chi, double **RF, double **AF)
{
  int i,L0,Mul0,M0,i1,L;
  double S_coordinate[3];
  double dum,dum1,dum2,dum3,dum4,a,b,c,d;
  double siQ,coQ,siP,coP,Q,P,R,Rmin;
  double rm,df;

  /* Radial */
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,y12,y22,f;

  /* start calc. */

  Rmin = 10e-14;

  xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
  R = S_coordinate[0];
  Q = S_coordinate[1];
  P = S_coordinate[2];

  po = 0;
  mp_min = 0;
  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

    for (L0=0; L0<=Lmax; L0++){
      for (Mul0=0; Mul0<Nmul; Mul0++){
        RF[L0][Mul0] = 0.0;
      }
    }

    po = 1;
  }

  else if (R<Spe_PAO_RV[wan][0]){

    m = 4;
    rm = Spe_PAO_RV[wan][m];

    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

    x1 = rm - Spe_PAO_RV[wan][m-1];
    x2 = rm - Spe_PAO_RV[wan][m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    for (L0=0; L0<=Lmax; L0++){

      if (L0<=Spe_PAO_LMAX[wan]) L = L0;
      else                       L = Spe_PAO_LMAX[wan];

      for (Mul0=0; Mul0<Nmul; Mul0++){

        if (Spe_PAO_Mul[wan]<=Mul0){
          printf("Fatal error in calculating Voronoi orbital moments\n");
        }
 
        f1 = Spe_PAO_RWF[wan][L][Mul0][m-2];
        f2 = Spe_PAO_RWF[wan][L][Mul0][m-1];
        f3 = Spe_PAO_RWF[wan][L][Mul0][m];
        f4 = Spe_PAO_RWF[wan][L][Mul0][m+1];

        if (m==1){
          h1 = -(h2+h3);
          f1 = f4;
        }
        else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
          h3 = -(h1+h2);
          f4 = f1;
        }

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y22*(2.0*f2 + h2*g1)/h2
           + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
           - y12*(2.0*f3 - h2*g2)/h2;

        if (L==0){
          a = 0.0;
          b = 0.5*df/rm;
          c = 0.0;
          d = f - b*rm*rm;
        }

        else if (L==1){
          a = (rm*df - f)/(2.0*rm*rm*rm);
          b = 0.0;
          c = df - 3.0*a*rm*rm;
          d = 0.0;
        }

        else{
          b = (3.0*f - rm*df)/(rm*rm);
          a = (f - b*rm*rm)/(rm*rm*rm);
          c = 0.0;
          d = 0.0;
        }

        RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

      }
    }

  }

  else{

    do{
      m = (mp_min + mp_max)/2;
      if (Spe_PAO_RV[wan][m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

    x1 = R - Spe_PAO_RV[wan][m-1];
    x2 = R - Spe_PAO_RV[wan][m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    for (L0=0; L0<=Lmax; L0++){

      if (L0<=Spe_PAO_LMAX[wan]) L = L0;
      else                       L = Spe_PAO_LMAX[wan];

      for (Mul0=0; Mul0<Nmul; Mul0++){

        f1 = Spe_PAO_RWF[wan][L][Mul0][m-2];
        f2 = Spe_PAO_RWF[wan][L][Mul0][m-1];
        f3 = Spe_PAO_RWF[wan][L][Mul0][m];
        f4 = Spe_PAO_RWF[wan][L][Mul0][m+1];

        if (m==1){
          h1 = -(h2+h3);
          f1 = f4;
        }
        else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
          h3 = -(h1+h2);
          f4 = f1;
        }

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        RF[L0][Mul0] = f;

      }
    } 
  }

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);

    for (L0=0; L0<=Lmax; L0++){

      if (L0==0){
        AF[0][0] = 0.282094791773878;
      }
      else if (L0==1){
        dum = 0.48860251190292*siQ;
        AF[1][0] = dum*coP;
        AF[1][1] = dum*siP;
        AF[1][2] = 0.48860251190292*coQ;
      }
      else if (L0==2){
        dum1 = siQ*siQ;
        dum2 = 1.09254843059208*siQ*coQ;
        AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
        AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
        AF[2][2] = 1.09254843059208*dum1*siP*coP;
        AF[2][3] = dum2*coP;
        AF[2][4] = dum2*siP;
      }
      else if (L0==3){
        AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
        AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
        AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
        AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
        AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
        AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
        AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
      }

    }
  }

  /* Chi */  

  for (L0=0; L0<=Lmax; L0++){
    for (M0=0; M0<=2*L0; M0++){
      for (Mul0=0; Mul0<Nmul; Mul0++){
        Chi[L0][M0][Mul0] = RF[L0][Mul0]*AF[L0][M0];
      }
    }
  }
}






void Generate_L_Matrix(int Lmax, double ***Lx0, double ***Ly0, double ***Lz0)
{
  int L,M1,M2,M3,M,m1,m2;      
  dcomplex ***CLx,***CLy,***CLz;     
  dcomplex ***CLx1,***CLy1,***CLz1;     
  dcomplex Ctmp1,Ctmpx,Ctmpy,Ctmpz;

  /* allocation of arrays */
 
  CLx = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    CLx[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      CLx[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1)); 
      for (M1=0; M1<=(2*L); M1++)  CLx[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLy = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    CLy[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      CLy[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1)); 
      for (M1=0; M1<=(2*L); M1++)  CLy[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLz = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    CLz[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      CLz[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1)); 
      for (M1=0; M1<=(2*L); M1++)  CLz[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLx1 = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    CLx1[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      CLx1[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1)); 
      for (M1=0; M1<=(2*L); M1++)  CLx1[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLy1 = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    CLy1[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      CLy1[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1)); 
      for (M1=0; M1<=(2*L); M1++)  CLy1[L][M][M1] = Complex(0.0,0.0);
    }
  }

  CLz1 = (dcomplex***)malloc(sizeof(dcomplex**)*(Lmax+1)); 
  for (L=0; L<=Lmax; L++){
    CLz1[L] = (dcomplex**)malloc(sizeof(dcomplex*)*(2*L+1)); 
    for (M=0; M<=(2*L); M++){
      CLz1[L][M] = (dcomplex*)malloc(sizeof(dcomplex)*(2*L+1)); 
      for (M1=0; M1<=(2*L); M1++)  CLz1[L][M][M1] = Complex(0.0,0.0);
    }
  } 

  /* set complex matrices */

  for (L=0; L<=Lmax; L++){

    for (M1=0; M1<=2*L; M1++){
      for (M2=0; M2<=2*L; M2++){
        
        m1 = M1 - L;
        m2 = M2 - L;

        if (M1==(M2+1)) CLx[L][M1][M2].r  = 0.5*sqrt((L-m2)*(L+m2+1.0));
        if (M1==(M2-1)) CLx[L][M1][M2].r += 0.5*sqrt((L+m2)*(L-m2+1.0));

        if (M1==(M2+1)) CLy[L][M1][M2].i  =-0.5*sqrt((L-m2)*(L+m2+1.0));
        if (M1==(M2-1)) CLy[L][M1][M2].i -=-0.5*sqrt((L+m2)*(L-m2+1.0));

        if (M1==M2)     CLz[L][M1][M2].r  = m1;
      }
    }
  }

  /* transformation */

  for (L=0; L<=Lmax; L++){

    for (M1=0; M1<=2*L; M1++){
      for (M2=0; M2<=2*L; M2++){

        Ctmpx = Complex(0.0,0.0);
        Ctmpy = Complex(0.0,0.0);
        Ctmpz = Complex(0.0,0.0);

        for (M3=0; M3<=2*L; M3++){

          Ctmp1 = Comp2Real[L][M2][M3];

          Ctmpx.r += CLx[L][M1][M3].r*Ctmp1.r - CLx[L][M1][M3].i*Ctmp1.i;  
          Ctmpx.i += CLx[L][M1][M3].r*Ctmp1.i + CLx[L][M1][M3].i*Ctmp1.r;  

          Ctmpy.r += CLy[L][M1][M3].r*Ctmp1.r - CLy[L][M1][M3].i*Ctmp1.i;  
          Ctmpy.i += CLy[L][M1][M3].r*Ctmp1.i + CLy[L][M1][M3].i*Ctmp1.r;  

          Ctmpz.r += CLz[L][M1][M3].r*Ctmp1.r - CLz[L][M1][M3].i*Ctmp1.i;  
          Ctmpz.i += CLz[L][M1][M3].r*Ctmp1.i + CLz[L][M1][M3].i*Ctmp1.r;  

	}

        CLx1[L][M1][M2] = Ctmpx;        
        CLy1[L][M1][M2] = Ctmpy;        
        CLz1[L][M1][M2] = Ctmpz;        
      }
    }   
  }  

  for (L=0; L<=Lmax; L++){

    for (M1=0; M1<=2*L; M1++){
      for (M2=0; M2<=2*L; M2++){

        Ctmpx = Complex(0.0,0.0);
        Ctmpy = Complex(0.0,0.0);
        Ctmpz = Complex(0.0,0.0);

        for (M3=0; M3<=2*L; M3++){
 
          Ctmp1 = Conjg(Comp2Real[L][M1][M3]);

          Ctmpx.r += CLx1[L][M3][M2].r*Ctmp1.r - CLx1[L][M3][M2].i*Ctmp1.i;  
          Ctmpx.i += CLx1[L][M3][M2].r*Ctmp1.i + CLx1[L][M3][M2].i*Ctmp1.r;  

          Ctmpy.r += CLy1[L][M3][M2].r*Ctmp1.r - CLy1[L][M3][M2].i*Ctmp1.i;  
          Ctmpy.i += CLy1[L][M3][M2].r*Ctmp1.i + CLy1[L][M3][M2].i*Ctmp1.r;  

          Ctmpz.r += CLz1[L][M3][M2].r*Ctmp1.r - CLz1[L][M3][M2].i*Ctmp1.i;  
          Ctmpz.i += CLz1[L][M3][M2].r*Ctmp1.i + CLz1[L][M3][M2].i*Ctmp1.r;  
	}

        Lx0[L][M1][M2] = Ctmpx.i;        
        Ly0[L][M1][M2] = Ctmpy.i;        
        Lz0[L][M1][M2] = Ctmpz.i;
      }
    }   
  }  
      
  /* freeing of arrays */
 
  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLx[L][M]);
    }
    free(CLx[L]);
  }
  free(CLx);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLy[L][M]);
    }
    free(CLy[L]);
  }
  free(CLy);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLz[L][M]);
    }
    free(CLz[L]);
  }
  free(CLz);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLx1[L][M]);
    }
    free(CLx1[L]);
  }
  free(CLx1);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLy1[L][M]);
    }
    free(CLy1[L]);
  }
  free(CLy1);

  for (L=0; L<=Lmax; L++){
    for (M=0; M<=(2*L); M++){
      free(CLz1[L][M]);
    }
    free(CLz1[L]);
  }
  free(CLz1);

} 
