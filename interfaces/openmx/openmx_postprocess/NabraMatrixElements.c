/**********************************************************************
  NabraMatrixElements.c:

    NabraMatrixElements.c is a subroutine to calculate
    < PAO(atom i, orbital alpha) | nabla | PAO(atom j, orbital beta) >

  Log of Calc_MME.c:

     28/June/2018  Released by YT Lee

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "openmx_common.h"
#include "mpi.h"

//YTL-start
// MME_orb = matrix elements of Nabra operator - < phi_i_alpha | Nabla operator | phi_j_beta >
// without coefficients at a k-point and an eigen-state.
//
// parameters :
// i,j = index of atoms
// alpha, beta = orbital index of atom i/j
// Nabla = the first derivative wiht respect to x/y/z

void Calc_NabraMatrixElements()
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

  MPI_Status stat;
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

  /***************************************************************
  Calculation of matrix elements of Nabra operator for MME_allorb 
  ***************************************************************/

  int My_Matomnum[numprocs];
  int My_NZeros[numprocs];
  int is1[numprocs];
  int ie1[numprocs];
  int is2[numprocs];
  int order_GA[atomnum+2];
  int k,l,tno0,tno1,num,tnum,AN;

  // initialize MME_allorb
  MME_allorb = (double*****)malloc(sizeof(double****)*(atomnum+1)); // real number
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Cwan = WhatSpecies[Gc_AN];
    tno0 = Spe_Total_CNO[Cwan];
    MME_allorb[Gc_AN] = (double****)malloc(sizeof(double***)*(FNAN[Gc_AN]+1));
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      tno1 = Spe_Total_CNO[Hwan];
      MME_allorb[Gc_AN][h_AN] = (double***)malloc(sizeof(double**)*tno0);
      for (k=0; k<tno0; k++){
        MME_allorb[Gc_AN][h_AN][k] = (double**)malloc(sizeof(double*)*tno1);
        for (l=0; l<tno1; l++){
          MME_allorb[Gc_AN][h_AN][k][l] = (double*)malloc(sizeof(double)*3);
          for (i=0;i<3;i++) MME_allorb[Gc_AN][h_AN][k][l][i] = 0;
        }
      }
    }
  }

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];

    num = 0;      
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      NO1 = Spe_Total_CNO[Hwan];
      num += NO1;
    }

    My_NZeros[myid] += NO0*num;
  }

  MPI_Barrier(mpi_comm_level1);
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }

  MPI_Barrier(mpi_comm_level1);
  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  // set is2 and order_GA

  MPI_Barrier(mpi_comm_level1);
  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    order_GA[is2[myid]+Mc_AN-1] = M2G[Mc_AN];
  }

  MPI_Barrier(mpi_comm_level1);
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  k=is1[myid];

  double** tmp_MME = (double**)malloc(sizeof(double*)*3);
  for (j=0;j<3;j++) tmp_MME[j] = (double*)malloc(sizeof(double)*tnum);

  // start to calculate MME_allorb
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    NO0 = Spe_Total_CNO[Cwan];
    
    Set_dOrbitals_Grid_xyz(Global_Cnt_kind,1);

    for (i=0; i<NO0; i++){
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        GNc = GridListAtom[Mc_AN][Nc];
        GRc = CellListAtom[Mc_AN][Nc];

        Get_Grid_XYZ(GNc,Cxyz);
        x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
        y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
        z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

        ChiVx[i][Nc] = Orbs_Grid[Mc_AN][Nc][i]; // x components
        /*
        ChiVx[i][Nc] = dOrbs_Grid[0][Mc_AN][Nc][i];
        ChiVy[i][Nc] = dOrbs_Grid[1][Mc_AN][Nc][i];
        ChiVz[i][Nc] = dOrbs_Grid[2][Mc_AN][Nc][i];
        */
      }
    }

    Set_dOrbitals_Grid_xyz(Global_Cnt_kind,2);

    for (i=0; i<NO0; i++){
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        GNc = GridListAtom[Mc_AN][Nc];
        GRc = CellListAtom[Mc_AN][Nc];

        Get_Grid_XYZ(GNc,Cxyz);
        x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
        y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
        z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

        ChiVy[i][Nc] = Orbs_Grid[Mc_AN][Nc][i]; // y components
      }
    }

    Set_dOrbitals_Grid_xyz(Global_Cnt_kind,3);

    for (i=0; i<NO0; i++){
      for (Nc=0; Nc<GridN_Atom[Gc_AN]; Nc++){

        GNc = GridListAtom[Mc_AN][Nc];
        GRc = CellListAtom[Mc_AN][Nc];

        Get_Grid_XYZ(GNc,Cxyz);
        x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
        y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
        z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

        ChiVz[i][Nc] = Orbs_Grid[Mc_AN][Nc][i]; // z components
      }
    }

    Set_dOrbitals_Grid_xyz(Global_Cnt_kind,0);
    
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

      /* OLPpox,y,z */
      for (i=0; i<NO0; i++){
        for (j=0; j<NO1; j++){
          // hbar / a0 = 1 a.u.
          // all of MME_orb are in imaginery parp, and saved in double array
          // store data in 1D-array
          tmp_MME[0][k] = tmp_OLPpox[i][j]*GridVol; // <psi_i_alpha| nabla_x | psi_j_beta> ( without mutliplying GridVol )
          tmp_MME[1][k] = tmp_OLPpoy[i][j]*GridVol; // <psi_i_alpha| nabla_y | psi_j_beta> ( without mutliplying GridVol )
          tmp_MME[2][k] = tmp_OLPpoz[i][j]*GridVol; // <psi_i_alpha| nabla_z | psi_j_beta> ( without mutliplying GridVol )
          k++; 
        }
      }
    }

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  // broadcast 1D-data to all cpus
  if (numprocs>1){
    MPI_Barrier(mpi_comm_level1);
    for (ID=0; ID<numprocs; ID++){
      k = is1[ID];
      MPI_Bcast(&tmp_MME[0][k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
      MPI_Bcast(&tmp_MME[1][k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
      MPI_Bcast(&tmp_MME[2][k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
    }
  }

  // restore 1D-data to MME_allorb
  k=0;
  for (AN=1; AN<=atomnum; AN++){
    Gc_AN = order_GA[AN];  // get correct order of atomic serial index
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++)
      for (i=0; i< Spe_Total_CNO[ WhatSpecies[Gc_AN] ] ; i++)
        for (j=0; j< Spe_Total_CNO[ WhatSpecies[ natn[Gc_AN][h_AN] ] ]; j++){
          MME_allorb[Gc_AN][h_AN][i][j][0] = tmp_MME[0][k]; // x
          MME_allorb[Gc_AN][h_AN][i][j][1] = tmp_MME[1][k]; // y
          MME_allorb[Gc_AN][h_AN][i][j][2] = tmp_MME[2][k]; // z
          k++;
        }
  }

/*
  // print out
  if (myid==Host_ID){
    k=0;
    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++)
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++)
        for (i=0; i< Spe_Total_CNO[ WhatSpecies[Gc_AN] ] ; i++)
          for (j=0; j< Spe_Total_CNO[ WhatSpecies[ natn[Gc_AN][h_AN] ] ]; j++){
//            if (fabs(MME_allorb[Gc_AN][h_AN][i][j][0] - tmp_MME[0][k])+fabs(MME_allorb[Gc_AN][h_AN][i][j][1] - tmp_MME[1][k])+ fabs(MME_allorb[Gc_AN][h_AN][i][j][2] - tmp_MME[2][k]) > 0.0000001 )
            printf("%i %i %i %i : %12.7lf %12.7lf %12.7lf  %12.7lf %12.7lf %12.7lf\n",Gc_AN,h_AN,i,j,MME_allorb[Gc_AN][h_AN][i][j][0],MME_allorb[Gc_AN][h_AN][i][j][1],MME_allorb[Gc_AN][h_AN][i][j][2],tmp_MME[0][k],tmp_MME[1][k],tmp_MME[2][k]);
            k++;
          }
  }
*/

  /****************************************************
    freeing of arrays:
  ****************************************************/

  for (i=0; i<3; i++) free(tmp_MME[i]);
  free(tmp_MME);

  for (i=0; i<List_YOUSO[7]; i++){ free(ChiVx[i]); }
  free(ChiVx);

  for (i=0; i<List_YOUSO[7]; i++){ free(ChiVy[i]); }
  free(ChiVy);

  for (i=0; i<List_YOUSO[7]; i++){ free(ChiVz[i]); }
  free(ChiVz);

  free(tmp_ChiVx);
  free(tmp_ChiVy);
  free(tmp_ChiVz);

  free(tmp_Orbs_Grid);

  for (i=0; i<List_YOUSO[7]; i++){ free(tmp_OLPpox[i]); }
  free(tmp_OLPpox);

  for (i=0; i<List_YOUSO[7]; i++){ free(tmp_OLPpoy[i]); }
  free(tmp_OLPpoy);

  for (i=0; i<List_YOUSO[7]; i++){ free(tmp_OLPpoz[i]); }
  free(tmp_OLPpoz);
}
