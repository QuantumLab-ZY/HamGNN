/**********************************************************************
  Overlap_NC_Ss2.c:

     Overlap_NC_Ss2.c is a subroutine to make the S^{-1/2} matrix 
     for non-collinear cluster and band calculation, which is distributed 
     over MPI cores according to data distribution of ScaLAPACK.

  Log of Overlap_NC_Ss2.c:

     10/Feb./2019  Released by T. Ozaki

    rule for the transformation of indices

    **** il, jl (na_rows x na_cols) -> ig, jg (n x n) 

    ig = np_rows*nblk*((il)/nblk) + (il)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
    jg = np_cols*nblk*((jl)/nblk) + (jl)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;

    **** ig, jg (n x n) -> il, jl (na_rows x na_cols) 
    
    brow = (ig-1)/nblk;
    bcol = (jg-1)/nblk;
    prow = brow%np_rows;
    pcol = bcol%np_cols;

    if (my_prow==prow && my_pcol==pcol){

      il = (brow/np_rows+1)*nblk+1;
      jl = (bcol/np_cols+1)*nblk+1;

      if(((my_prow+np_rows)%np_rows) >= (brow%np_rows)){

        if(my_prow==prow){
          il = il+(ig-1)%nblk;
        }
        il = il-nblk;
      }

      if(((my_pcol+np_cols)%np_cols) >= (bcol%np_cols)){

        if(my_pcol==pcol){
          jl = jl+(jg-1)%nblk;
        }
        jl = jl-nblk;
      }
    }
    else{

      il = -1; 
      jl = -1;
    }

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"

static void trans_indices_n_to_n2( int id1, int i1, int j1, 
				   int *id11, int *i11, int *j11,
				   int *id22, int *i22, int *j22,
				   int *id12, int *i12, int *j12,
				   int *id21, int *i21, int *j21);

void Overlap_Band_NC_Ss2( dcomplex *Ss, dcomplex *Ss2, MPI_Comm mpi_commWD )
{
  int k,i,i1,j1,i2,j2,id11,id22,id12,id21;
  int i11,i22,i12,i21;
  int j11,j22,j12,j21;
  int IDS,IDR,size1,size2;
  int ID,myid,numprocs,tag=999;
  int *Num_Snd,*Num_Rcv;
  int **Snd_i2j2,**Rcv_i2j2;
  double **Snd_Ss,**Rcv_Ss;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_commWD,&numprocs);
  MPI_Comm_rank(mpi_commWD,&myid);
  MPI_Barrier(mpi_commWD);

  /* allocation of arrays */

  Num_Snd = (int*)malloc(sizeof(int)*numprocs);
  Num_Rcv = (int*)malloc(sizeof(int)*numprocs);

  for (ID=0; ID<numprocs; ID++){
    Num_Snd[ID] = 0;
    Num_Rcv[ID] = 0;
  }

  /* count the number of elements to be sent to other processes */

  for (i1=0; i1<na_rows; i1++){
    for (j1=0; j1<na_cols; j1++){

      trans_indices_n_to_n2(myid, i1, j1, &id11, &i11, &j11, &id22, &i22, &j22, &id12, &i12, &j12, &id21, &i21, &j21 );
      Num_Snd[id11]++; 
      Num_Snd[id22]++; 
    }
  }

  /* MPI: Num_Snd -> Num_Rcv */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      /*  sending size */
      size1 = Num_Snd[IDS];
      MPI_Isend(&size1, 1, MPI_INT, IDS, tag, mpi_commWD, &request);

      /* receiving size */
      MPI_Recv(&size2, 1, MPI_INT, IDR, tag, mpi_commWD, &stat);
      Num_Rcv[IDR] = size2;

      /* MPI_Wait */
      MPI_Wait(&request,&stat);
    }
    else{
      Num_Rcv[IDR] = Num_Snd[IDS];
    }
  }

  /* allocation of arrays */

  Snd_i2j2 = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Snd_i2j2[ID] = (int*)malloc(sizeof(int)*(Num_Snd[ID]+1)*2);
  }

  Rcv_i2j2 = (int**)malloc(sizeof(int*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Rcv_i2j2[ID] = (int*)malloc(sizeof(int)*(Num_Rcv[ID]+1)*2);
  }

  Snd_Ss = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Snd_Ss[ID] = (double*)malloc(sizeof(double*)*(Num_Snd[ID]+1)*2);
  }

  Rcv_Ss = (double**)malloc(sizeof(double*)*numprocs);
  for (ID=0; ID<numprocs; ID++){
    Rcv_Ss[ID] = (double*)malloc(sizeof(double*)*(Num_Rcv[ID]+1)*2);
  }

  /* set Snd_i2j2 */

  for (ID=0; ID<numprocs; ID++){
    Num_Snd[ID] = 0;
  }

  for (i1=0; i1<na_rows; i1++){
    for (j1=0; j1<na_cols; j1++){

      trans_indices_n_to_n2(myid, i1, j1, &id11, &i11, &j11, &id22, &i22, &j22, &id12, &i12, &j12, &id21, &i21, &j21 );

      /* 11 */

      Snd_i2j2[id11][2*Num_Snd[id11]  ] = i11; 
      Snd_i2j2[id11][2*Num_Snd[id11]+1] = j11; 
      Snd_Ss[id11][2*Num_Snd[id11]    ] = Ss[j1*na_rows+i1].r;
      Snd_Ss[id11][2*Num_Snd[id11]+1  ] = Ss[j1*na_rows+i1].i;
      Num_Snd[id11]++; 

      /* 22 */

      Snd_i2j2[id22][2*Num_Snd[id22]  ] = i22; 
      Snd_i2j2[id22][2*Num_Snd[id22]+1] = j22;
      Snd_Ss[id22][2*Num_Snd[id22]    ] = Ss[j1*na_rows+i1].r;
      Snd_Ss[id22][2*Num_Snd[id22]+1  ] = Ss[j1*na_rows+i1].i;
      Num_Snd[id22]++;

    }
  }

  /* MPI: Snd_i2j2 and Snd_Ss -> Rcv_i2j2 and Rcv_Ss */

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /* Snd_i2j2 */

    if (ID!=0){

      if (Num_Snd[IDS]!=0){

        /*  sending Snd_i2j2 */
        MPI_Isend(&Snd_i2j2[IDS][0], Num_Snd[IDS]*2, MPI_INT, IDS, tag, mpi_commWD, &request);
      }

      if (Num_Rcv[IDR]!=0){

        /* receiving size */
        MPI_Recv(&Rcv_i2j2[IDR][0], Num_Rcv[IDR]*2, MPI_INT, IDR, tag, mpi_commWD, &stat);
      }

      if (Num_Snd[IDS]!=0){
        /* MPI_Wait */
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for (k=0; k<Num_Snd[IDS]*2; k++){
        Rcv_i2j2[IDR][k] = Snd_i2j2[IDS][k];
      }
    }

    /* Snd_Ss */

    if (ID!=0){

      if (Num_Snd[IDS]!=0){

        /* sending Snd_Ss */
        MPI_Isend(&Snd_Ss[IDS][0], Num_Snd[IDS]*2, MPI_DOUBLE, IDS, tag, mpi_commWD, &request);
      }

      if (Num_Rcv[IDR]!=0){

        /* receiving size */
        MPI_Recv(&Rcv_Ss[IDR][0], Num_Rcv[IDR]*2, MPI_DOUBLE, IDR, tag, mpi_commWD, &stat);
      }

      if (Num_Snd[IDS]!=0){
        /* MPI_Wait */
        MPI_Wait(&request,&stat);
      }
    }
    else{
      for (k=0; k<Num_Snd[IDS]*2; k++){
        Rcv_Ss[IDR][k] = Snd_Ss[IDS][k];
      }
    }
  } 

  /* set Ss2 */

  for (i=0; i<na_rows2*na_cols2; i++){
    Ss2[i].r = 0.0;
    Ss2[i].i = 0.0;
  }

  for (ID=0; ID<numprocs; ID++){
    for (k=0; k<Num_Rcv[ID]; k++){

      i2 = Rcv_i2j2[ID][2*k  ];
      j2 = Rcv_i2j2[ID][2*k+1];

      Ss2[(j2-1)*na_rows2+(i2-1)].r = Rcv_Ss[ID][2*k  ];
      Ss2[(j2-1)*na_rows2+(i2-1)].i = Rcv_Ss[ID][2*k+1];
    }
  }

  /* freeing of arrays */

  free(Num_Snd);
  free(Num_Rcv);

  for (ID=0; ID<numprocs; ID++){
    free(Snd_i2j2[ID]);
  }
  free(Snd_i2j2);

  for (ID=0; ID<numprocs; ID++){
    free(Rcv_i2j2[ID]);
  }
  free(Rcv_i2j2);

  for (ID=0; ID<numprocs; ID++){
    free(Snd_Ss[ID]);
  }
  free(Snd_Ss);

  for (ID=0; ID<numprocs; ID++){
    free(Rcv_Ss[ID]);
  }
  free(Rcv_Ss);
}


void trans_indices_n_to_n2( int id1, int i1, int j1, 
                            int *id11, int *i11, int *j11,
                            int *id22, int *i22, int *j22,
                            int *id12, int *i12, int *j12,
                            int *id21, int *i21, int *j21)
{
  int i,wanA,ig,jg,il,jl,loop,si,sj,n;
  int my_prow1,my_pcol1,id2;
  int brow2,bcol2,prow2,pcol2;

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA = WhatSpecies[i];
    n += Spe_Total_CNO[wanA];
  }

  my_prow1 = id1/np_cols;
  my_pcol1 = id1%np_cols;

  for (loop=1; loop<=4; loop++){

    if      (loop==1){ si = 0; sj = 0; }
    else if (loop==2){ si = n; sj = n; }
    else if (loop==3){ si = 0; sj = n; }
    else if (loop==4){ si = 0; sj = n; }

    if (loop!=4){
      ig = np_rows*nblk*((i1)/nblk) + (i1)%nblk + ((np_rows+my_prow1)%np_rows)*nblk + 1 + si;
      jg = np_cols*nblk*((j1)/nblk) + (j1)%nblk + ((np_cols+my_pcol1)%np_cols)*nblk + 1 + sj;
    }
    else{
      jg = np_rows*nblk*((i1)/nblk) + (i1)%nblk + ((np_rows+my_prow1)%np_rows)*nblk + 1 + si;
      ig = np_cols*nblk*((j1)/nblk) + (j1)%nblk + ((np_cols+my_pcol1)%np_cols)*nblk + 1 + sj;
    }

    brow2 = (ig-1)/nblk2;
    bcol2 = (jg-1)/nblk2;
    prow2 = brow2%np_rows2;
    pcol2 = bcol2%np_cols2;

    id2 = prow2*np_cols2 + pcol2; 

    il = (brow2/np_rows2+1)*nblk2+1;
    jl = (bcol2/np_cols2+1)*nblk2+1;

    if (((prow2+np_rows2)%np_rows2) >= (brow2%np_rows2)){
      il = il+(ig-1)%nblk2;
      il = il-nblk2;
    }

    if (((pcol2+np_cols2)%np_cols2) >= (bcol2%np_cols2)){
      jl = jl+(jg-1)%nblk2;
      jl = jl-nblk2;
    }

    if      (loop==1){ *id11 = id2; *i11 = il; *j11 = jl; }
    else if (loop==2){ *id22 = id2; *i22 = il; *j22 = jl; }
    else if (loop==3){ *id12 = id2; *i12 = il; *j12 = jl; }
    else if (loop==4){ *id21 = id2; *i21 = il; *j21 = jl; }
  }
}

