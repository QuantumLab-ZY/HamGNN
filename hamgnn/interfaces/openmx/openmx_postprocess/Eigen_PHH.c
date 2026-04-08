/**********************************************************************
    Eigen_PHH.c:

    Eigen_PHH.c is a MPI parallelized subroutine to solve a standard
    eigenvalue problem with a Hermite complex matrix using Householder
    method and lapack's dstegr_() or dstedc_().

    Log of Eigen_PHH.c:

       Dec/14/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include "f77func.h"

#define  measure_time   0


static void Eigen_Original_PHH(MPI_Comm MPI_Current_Comm_WD,
			       dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag);
static void Eigen_Improved_PHH(MPI_Comm MPI_Current_Comm_WD,
			       dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag);
static void Eigen_ELPA1_Co(MPI_Comm MPI_Current_Comm_WD, 
			   dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag);
static void Eigen_ELPA2_Co(MPI_Comm MPI_Current_Comm_WD, 
			   dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag);

static int numrocC(int N, int NB, int IPROC, int ISRCPROC, int NPROCS);


void Eigen_PHH(MPI_Comm MPI_Current_Comm_WD, 
               dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag)
{
  int numprocs;

  MPI_Comm_size(MPI_Current_Comm_WD,&numprocs);

  if (n<20 || n<numprocs)
    EigenBand_lapack(ac, ko, n, n, 1);

  else if (scf_eigen_lib_flag==0 || n<100)
    Eigen_Improved_PHH(MPI_Current_Comm_WD, ac, ko, n, EVmax, bcast_flag);

  else if (scf_eigen_lib_flag==1)
    Eigen_ELPA1_Co(MPI_Current_Comm_WD, ac, ko, n, EVmax, bcast_flag);

#ifndef kcomp
  else if (scf_eigen_lib_flag==2)
    Eigen_ELPA2_Co(MPI_Current_Comm_WD, ac, ko, n, EVmax, bcast_flag);
#endif

}


#ifndef kcomp
void Eigen_ELPA2_Co(MPI_Comm MPI_Current_Comm_WD, 
                    dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag)
{

 /*
   !-------------------------------------------------------------------------------
   ! na:   System size (of the global matrix)
   ! nev:  Number of eigenvectors to be calculated
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------
 */

  int na = n;
  int nev = EVmax;
  int nblk = 4;
  /*
  int nblk = 16;
  */

  int np_rows, np_cols, na_rows, na_cols;
  int myid, numprocs, my_prow, my_pcol;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int i, j, my_blacs_ctxt, info, nprow, npcol, ig, jg;
  int zero=0, one=1, LOCr, LOCc, node, irow, icol, mpiworld;

  dcomplex *a, *z, *lz;
  MPI_Status *stat_send;

  MPI_Comm_size(MPI_Current_Comm_WD,&numprocs);
  MPI_Comm_rank(MPI_Current_Comm_WD,&myid);

  stat_send = (MPI_Status*)malloc(sizeof(MPI_Status)*numprocs);
  mpiworld = MPI_Comm_c2f(MPI_Current_Comm_WD);

  /*
   !-------------------------------------------------------------------------------
   ! Selection of number of processor rows/columns
   ! We try to set up the grid square-like, i.e. start the search for possible
   ! divisors of nprocs with a number next to the square root of nprocs
   ! and decrement it until a divisor is found.
  */

  np_cols=(int)(sqrt((float)numprocs));
  do{
    if((numprocs%np_cols)==0) break;
    np_cols--;
  } while(np_cols>=2);

  np_rows = numprocs/np_cols;

  my_prow = myid/np_cols;
  my_pcol = myid%np_cols;

  /*
  if(myid==0){
    printf("Number of processor rows=%d, cols=%d, total=%d\n",np_rows,np_cols,numprocs);
   }
  printf("myid=%d, my_prow=%d, my_pcol=%d mpi_comm_rows=%d mpi_comm_cols=%d\n",myid,my_prow,my_pcol,mpi_comm_rows,mpi_comm_cols);
  */

	 /*
   !-------------------------------------------------------------------------------
   ! Set up BLACS context and MPI communicators
   ! For ELPA, the MPI communicators along rows/cols are sufficient,
   ! and the grid setup may be done in an arbitrary way as long as it is
   ! consistent (i.e. 0<=my_prow<np_rows, 0<=my_pcol<np_cols and every
   ! process has a unique (my_prow,my_pcol) pair).
   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_row_col_comms.
	 */

  MPI_Comm_split(MPI_Current_Comm_WD,my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_Current_Comm_WD,my_prow,my_pcol,&mpi_comm_cols);

       /*
   ! Determine the necessary size of the distributed matrices,
   ! we use the Scalapack tools routine NUMROC for that.
       */

  na_rows = numrocC(na, nblk, my_prow, 0, np_rows);
  na_cols = numrocC(na, nblk, my_pcol, 0, np_cols);

  /*
   ! Set up a scalapack descriptor for the checks below.
   ! For ELPA the following restrictions hold:
   ! - block sizes in both directions must be identical (args 4+5)
   ! - first row and column of the distributed matrix must be on row/col 0/0 (args 6+7)
   ! Allocate the local matrices and distribute the matrix elements from the global matrix to the local matrices
   ! using block-cyclic Scalapack distribution 
  */

  a = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
  z = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);

  for(i=0;i<na_rows;i++){
    for(j=0;j<na_cols;j++){
      ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
      jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
      a[j*na_rows+i].r=ac[ig][jg].r;
      a[j*na_rows+i].i=ac[ig][jg].i;
    }
  }

  /*
   ! Calculate eigenvalues/eigenvectors with ELPA
  */

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  F77_NAME(elpa_solve_evp_complex_2stage_double_impl,ELPA_SOLVE_EVP_COMPLEX_2STAGE_DOUBLE_IMPL)(&na, &nev, a, &na_rows, &ko[1], z, &na_rows, &nblk, &na_cols, &mpi_comm_rows_int, &mpi_comm_cols_int, &mpiworld);

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  /*
   ! The eigenvectors are distributed to the processes using block-cyclic Scalapack distribution 
   ! Collect the eigenvectors to the host process
  */

  if(myid==0){

   for(node=0;node<numprocs;node++){

     if(node==0){

       for(i=0;i<na_rows;i++){
	 for(j=0;j<na_cols;j++){
	   ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	   jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	   ac[ig][jg].r = z[j*na_rows+i].r;
	   ac[ig][jg].i = z[j*na_rows+i].i;
	 }
       }
     }

     else{
         
       MPI_Recv(&irow, 1, MPI_INT, node, 10, MPI_Current_Comm_WD, stat_send);
       MPI_Recv(&icol, 1, MPI_INT, node, 20, MPI_Current_Comm_WD, stat_send);
       MPI_Recv(&LOCr, 1, MPI_INT, node, 40, MPI_Current_Comm_WD, stat_send);
       MPI_Recv(&LOCc, 1, MPI_INT, node, 50, MPI_Current_Comm_WD, stat_send);
         
       lz = (dcomplex*)malloc(sizeof(dcomplex)*LOCr*LOCc);

       MPI_Recv(lz, LOCr*LOCc*2, MPI_DOUBLE, node, 30, MPI_Current_Comm_WD, stat_send);

       for(i=0;i<LOCr;i++){
	 for(j=0;j<LOCc;j++){
	   ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+irow)%np_rows)*nblk + 1;
	   jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+icol)%np_cols)*nblk + 1;
	   ac[ig][jg].r = lz[j*LOCr+i].r;
	   ac[ig][jg].i = lz[j*LOCr+i].i;
	 }
       }
       free(lz);
     }
   }
  }
  else{
    MPI_Send(&my_prow, 1, MPI_INT, 0, 10, MPI_Current_Comm_WD);
    MPI_Send(&my_pcol, 1, MPI_INT, 0, 20, MPI_Current_Comm_WD);
    MPI_Send(&na_rows, 1, MPI_INT, 0, 40, MPI_Current_Comm_WD);
    MPI_Send(&na_cols, 1, MPI_INT, 0, 50, MPI_Current_Comm_WD);
    MPI_Send(z, na_rows*na_cols*2, MPI_DOUBLE, 0, 30, MPI_Current_Comm_WD);
  } 

  /*
   ! Broadcast the eigenvectors to all proceses 
  */

  for(i=1;i<=n;i++){
    MPI_Bcast(ac[i],(nev+1)*2,MPI_DOUBLE,0,MPI_Current_Comm_WD);
    MPI_Barrier(MPI_Current_Comm_WD);
  }


  /*
  if (bcast_flag==0){
  for (j=0; j<n+1; j++){
    free(ac[j]);
  }
  free(ac);

  printf("ZZZ7 myid=%3d\n",myid);
  MPI_Finalize();
  exit(0);
  }
  */

  free(a);
  free(z);
  free(stat_send);
}
#endif



void Eigen_ELPA1_Co(MPI_Comm MPI_Current_Comm_WD, 
                    dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag)
{

 /*
   !-------------------------------------------------------------------------------
   ! na:   System size (of the global matrix)
   ! nev:  Number of eigenvectors to be calculated
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------
 */

  int na = n;
  int nev = EVmax;
  int nblk = 4;
  /*
  int nblk = 16;
  */

  int np_rows, np_cols, na_rows, na_cols;
  int myid, numprocs, my_prow, my_pcol;
  MPI_Comm mpi_comm_rows, mpi_comm_cols;
  int mpi_comm_rows_int,mpi_comm_cols_int;
  int i, j, my_blacs_ctxt, info, nprow, npcol, ig, jg;
  int zero=0, one=1, LOCr, LOCc, node, irow, icol, mpiworld;

  dcomplex *a, *z, *lz;
  MPI_Status *stat_send;

  MPI_Comm_size(MPI_Current_Comm_WD,&numprocs);
  MPI_Comm_rank(MPI_Current_Comm_WD,&myid);

  stat_send = (MPI_Status*)malloc(sizeof(MPI_Status)*numprocs);
  mpiworld = MPI_Comm_c2f(MPI_Current_Comm_WD);

  /*
   !-------------------------------------------------------------------------------
   ! Selection of number of processor rows/columns
   ! We try to set up the grid square-like, i.e. start the search for possible
   ! divisors of nprocs with a number next to the square root of nprocs
   ! and decrement it until a divisor is found.
  */

  np_cols=(int)(sqrt((float)numprocs));
  do{
    if((numprocs%np_cols)==0) break;
    np_cols--;
  } while(np_cols>=2);

  np_rows = numprocs/np_cols;

  my_prow = myid/np_cols;
  my_pcol = myid%np_cols;

  /*
  if(myid==0){
    printf("Number of processor rows=%d, cols=%d, total=%d\n",np_rows,np_cols,numprocs);
   }
  printf("myid=%d, my_prow=%d, my_pcol=%d mpi_comm_rows=%d mpi_comm_cols=%d\n",myid,my_prow,my_pcol,mpi_comm_rows,mpi_comm_cols);
  */

	 /*
   !-------------------------------------------------------------------------------
   ! Set up BLACS context and MPI communicators
   ! For ELPA, the MPI communicators along rows/cols are sufficient,
   ! and the grid setup may be done in an arbitrary way as long as it is
   ! consistent (i.e. 0<=my_prow<np_rows, 0<=my_pcol<np_cols and every
   ! process has a unique (my_prow,my_pcol) pair).
   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_row_col_comms.
	 */

  MPI_Comm_split(MPI_Current_Comm_WD,my_pcol,my_prow,&mpi_comm_rows);
  MPI_Comm_split(MPI_Current_Comm_WD,my_prow,my_pcol,&mpi_comm_cols);

       /*
   ! Determine the necessary size of the distributed matrices,
   ! we use the Scalapack tools routine NUMROC for that.
       */

  na_rows = numrocC(na, nblk, my_prow, 0, np_rows);
  na_cols = numrocC(na, nblk, my_pcol, 0, np_cols);

  /*
   ! Set up a scalapack descriptor for the checks below.
   ! For ELPA the following restrictions hold:
   ! - block sizes in both directions must be identical (args 4+5)
   ! - first row and column of the distributed matrix must be on row/col 0/0 (args 6+7)
   ! Allocate the local matrices and distribute the matrix elements from the global matrix to the local matrices
   ! using block-cyclic Scalapack distribution 
  */

  a = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);
  z = (dcomplex*)malloc(sizeof(dcomplex)*na_rows*na_cols);

  for(i=0;i<na_rows;i++){
    for(j=0;j<na_cols;j++){
      ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
      jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
      a[j*na_rows+i].r=ac[ig][jg].r;
      a[j*na_rows+i].i=ac[ig][jg].i;
    }
  }

  /*
   ! Calculate eigenvalues/eigenvectors with ELPA
  */

  mpi_comm_rows_int = MPI_Comm_c2f(mpi_comm_rows);
  mpi_comm_cols_int = MPI_Comm_c2f(mpi_comm_cols);

  F77_NAME(solve_evp_complex,SOLVE_EVP_COMPLEX)(&na, &nev, a, &na_rows, &ko[1], z, &na_rows, &nblk, &mpi_comm_rows_int, &mpi_comm_cols_int);

  MPI_Comm_free(&mpi_comm_rows);
  MPI_Comm_free(&mpi_comm_cols);

  /*
   ! The eigenvectors are distributed to the processes using block-cyclic Scalapack distribution 
   ! Collect the eigenvectors to the host process
  */

  if(myid==0){

   for(node=0;node<numprocs;node++){

     if(node==0){

       for(i=0;i<na_rows;i++){
	 for(j=0;j<na_cols;j++){
	   ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+my_prow)%np_rows)*nblk + 1;
	   jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+my_pcol)%np_cols)*nblk + 1;
	   ac[ig][jg].r = z[j*na_rows+i].r;
	   ac[ig][jg].i = z[j*na_rows+i].i;
	 }
       }
     }

     else{
         
       MPI_Recv(&irow, 1, MPI_INT, node, 10, MPI_Current_Comm_WD, stat_send);
       MPI_Recv(&icol, 1, MPI_INT, node, 20, MPI_Current_Comm_WD, stat_send);
       MPI_Recv(&LOCr, 1, MPI_INT, node, 40, MPI_Current_Comm_WD, stat_send);
       MPI_Recv(&LOCc, 1, MPI_INT, node, 50, MPI_Current_Comm_WD, stat_send);
         
       lz = (dcomplex*)malloc(sizeof(dcomplex)*LOCr*LOCc);

       MPI_Recv(lz, LOCr*LOCc*2, MPI_DOUBLE, node, 30, MPI_Current_Comm_WD, stat_send);

       for(i=0;i<LOCr;i++){
	 for(j=0;j<LOCc;j++){
	   ig = np_rows*nblk*((i)/nblk) + (i)%nblk + ((np_rows+irow)%np_rows)*nblk + 1;
	   jg = np_cols*nblk*((j)/nblk) + (j)%nblk + ((np_cols+icol)%np_cols)*nblk + 1;
	   ac[ig][jg].r = lz[j*LOCr+i].r;
	   ac[ig][jg].i = lz[j*LOCr+i].i;
	 }
       }
       free(lz);
     }
   }
  }
  else{
    MPI_Send(&my_prow, 1, MPI_INT, 0, 10, MPI_Current_Comm_WD);
    MPI_Send(&my_pcol, 1, MPI_INT, 0, 20, MPI_Current_Comm_WD);
    MPI_Send(&na_rows, 1, MPI_INT, 0, 40, MPI_Current_Comm_WD);
    MPI_Send(&na_cols, 1, MPI_INT, 0, 50, MPI_Current_Comm_WD);
    MPI_Send(z, na_rows*na_cols*2, MPI_DOUBLE, 0, 30, MPI_Current_Comm_WD);
  } 

  /*
   ! Broadcast the eigenvectors to all proceses 
  */

  for(i=1;i<=n;i++){
    MPI_Bcast(ac[i],(nev+1)*2,MPI_DOUBLE,0,MPI_Current_Comm_WD);
    MPI_Barrier(MPI_Current_Comm_WD);
  }


  /*
  if (bcast_flag==0){
  for (j=0; j<n+1; j++){
    free(ac[j]);
  }
  free(ac);

  printf("ZZZ7 myid=%3d\n",myid);
  MPI_Finalize();
  exit(0);
  }
  */

  free(a);
  free(z);
  free(stat_send);
}

int numrocC(int N, int NB, int IPROC, int ISRCPROC, int NPROCS)
{
  int EXTRABLKS, MYDIST, NBLOCKS, NUMROC;

  /* Figure PROC's distance from source process */

  MYDIST = (NPROCS+IPROC-ISRCPROC) % NPROCS;

  /* Figure the total number of whole NB blocks N is split up into */

  NBLOCKS = N / NB;

  /* Figure the minimum number of rows/cols a process can have */

  NUMROC = (NBLOCKS/NPROCS) * NB;

  /* See if there are any extra blocks */

  EXTRABLKS = NBLOCKS % NPROCS;

  /* If I have an extra block */

  if(MYDIST < EXTRABLKS)
    NUMROC = NUMROC + NB;

  /* If I have last block, it may be a partial block */

  else if(MYDIST==EXTRABLKS)
    NUMROC = NUMROC +  (N % NB);

  return NUMROC;
}


#pragma optimization_level 1
void Eigen_Improved_PHH(MPI_Comm MPI_Current_Comm_WD,
                        dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag)
{
  double ABSTOL=LAPACK_ABSTOL;
  dcomplex **ad,*u,*b1,*p,*q,tmp0,tmp1,tmp2,u1,u2,p1,ss,ss0,ss1;
  double *uu,*alphar,*alphai,
         s1,s2,s3,r,
         sum,ar,ai,br,bi,e,
         a1,a2,a3,a4,a5,a6,b7,
         r3,x1,x2,xap,my_r,
         bb,bb1,ui,uj,uij;

  int jj,jj1,jj2,k,ks,ii,ll,i3,i2,j2,
      i,j,i1,i1s,j1,n1,n2,ik,k0,k1,num0,num1,nump,
      jk,po1,nn,count,is0,num,ID0,ID1,IDS,IDR;

  double Stime,Etime;
  double Stime1, Etime1;
  double time1,time2,time1a,time1b,time1c,time1d,time1e;
  int *is1,*ie1,*is2,*ie2,*row_flag;
  double av_num,*a1d;
  int numprocs,myid,tag=999,ID;

  MPI_Status stat;
  MPI_Request request;

  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(MPI_Current_Comm_WD,&numprocs);
  MPI_Comm_rank(MPI_Current_Comm_WD,&myid);

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=EVmax ){

    av_num = (double)EVmax/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = EVmax; 
  }

  else {
    for (ID=0; ID<EVmax; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=EVmax; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /****************************************************
    allocation of arrays:
  ****************************************************/

  n2 = n + 5;

  is2 = (int*)malloc(sizeof(int)*numprocs);
  ie2 = (int*)malloc(sizeof(int)*numprocs);

  row_flag = (int*)malloc(sizeof(int)*n2);

  ad = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    ad[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  b1 = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  u = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  uu = (double*)malloc(sizeof(double)*n2);
  p = (dcomplex*)malloc(sizeof(dcomplex)*2*n2);
  q = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  a1d = (double*)malloc(sizeof(double)*4*n2);

  alphar = (double*)malloc(sizeof(double)*n2);
  alphai = (double*)malloc(sizeof(double)*n2);

  for (i=1; i<=(n+2); i++){
    uu[i] = 0.0;
    b1[i] = Complex(0.0,0.0); 
  }

  if (measure_time==1) printf("size n=%3d EVmax=%2d\n",n,EVmax);
  if (measure_time==1) dtime(&Stime);

  time1 = 0.0;
  time2 = 0.0;
  time1a = 0.0;
  time1b = 0.0;
  time1c = 0.0;
  time1d = 0.0;
  time1e = 0.0;

  /****************************************************
               Householder transformation
  ****************************************************/

  /* make is2 and ie2 */

  if ( numprocs<=(n-1) ){

    av_num = (double)(n-1)/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 2;
      ie2[ID] = (int)(av_num*(double)(ID+1)) + 1;
    }

    is2[0] = 2;
    ie2[numprocs-1] = n;
  }

  else{
    for (ID=0; ID<(n-1); ID++){
      is2[ID] = ID + 2;
      ie2[ID] = ID + 2;
    }
    for (ID=(n-1); ID<numprocs; ID++){
      is2[ID] = 2;
      ie2[ID] = 0;
    }
  }

  row_flag[1] = 0;
  for (ID=0; ID<numprocs; ID++){
    for (i=is2[ID]; i<=ie2[ID]; i++){
      row_flag[i] = ID;
    }
  }

  /* loop for i */

  for (i=1; i<=(n-1); i++){

    if (measure_time==1) dtime(&Stime1);

    /* make u vector */

    s2 = 0.0;

    u[i+1].r = ac[i][i+1].r;
    u[i+1].i =-ac[i][i+1].i;

    for (i1=i+2; i1<=(n-1); i1+=2){
      u[i1+0].r = ac[i][i1+0].r; 
      u[i1+0].i =-ac[i][i1+0].i; 
      u[i1+1].r = ac[i][i1+1].r; 
      u[i1+1].i =-ac[i][i1+1].i; 

      s2 += u[i1+0].r*u[i1+0].r + u[i1+0].i*u[i1+0].i
          + u[i1+1].r*u[i1+1].r + u[i1+1].i*u[i1+1].i;
    }

    i1s = n + 1 - (n+1-(i+2))%2;
    for (i1=i1s; ((i+2)<=i1 && i1<=n); i1++){
      u[i1].r = ac[i][i1].r; 
      u[i1].i =-ac[i][i1].i; 
      s2 += u[i1].r*u[i1].r + u[i1].i*u[i1].i;
    }

    u[i].r = s2;

    ID = row_flag[i];
    count = n - i + 1;

    if (myid==ID){

      i2 = 2*i - 1;
      for (i1=i; i1<=n; i1++){
        a1d[i2] = u[i1].r;  i2++;
        a1d[i2] = u[i1].i;  i2++;
      }

      for (IDS=ID+1; IDS<numprocs; IDS++){
        MPI_Isend(&a1d[2*i-1], 2*count, MPI_DOUBLE, IDS, tag, MPI_Current_Comm_WD, &request_send[IDS-(ID+1)]);
      }

      num = (numprocs-1) - (ID+1) + 1;  
      if (1<=num){
        MPI_Waitall(num,request_send,stat_send);
      }
    }
    else if ( (ID+1)<=myid ) {
      MPI_Recv(&a1d[2*i-1], 2*count, MPI_DOUBLE, ID, tag, MPI_Current_Comm_WD, &stat);

      i2 = 2*i - 1;
      for (i1=i; i1<=n; i1++){
        u[i1].r = a1d[i2];  i2++;
        u[i1].i = a1d[i2];  i2++;
      }
    }

    s2 = u[i].r;

    if (measure_time==1){ 
      dtime(&Etime1);
      time1a += Etime1 - Stime1;      
    }

    s1 = u[i+1].r * u[i+1].r + u[i+1].i * u[i+1].i;
    s3 = fabs(s1 + s2);

    if ( ABSTOL<(fabs(u[i+1].r)+fabs(u[i+1].i)) ){
      if (u[i+1].r<0.0)  s3 =  sqrt(s3);
      else               s3 = -sqrt(s3);
    }
    else{
      s3 = sqrt(s3);
    }

    if ( ABSTOL<fabs(s2) || 1.0e-10<fabs(u[i+1].i) || i==(n-1) ){

      if (measure_time==1) dtime(&Stime1);

      ss.r = u[i+1].r;
      ss.i = u[i+1].i;

      ac[i+1][i].r = s3;
      ac[i+1][i].i = 0.0;
      ac[i][i+1].r = s3;
      ac[i][i+1].i = 0.0;

      u[i+1].r = u[i+1].r - s3;
      u[i+1].i = u[i+1].i;
      
      u1.r = s3 * s3 - ss.r * s3;
      u1.i =         - ss.i * s3;
      u2.r = 2.0 * u1.r;
      u2.i = 2.0 * u1.i;
      
      e = u2.r/(u1.r*u1.r + u1.i*u1.i);
      ar = e*u1.r;
      ai = e*u1.i;

      /* store alpha */
      alphar[i] = ar;
      alphai[i] = ai;

      /* store u2 */
      uu[i] = u2.r;

      /* store the first component of u */
      b1[i].r = ss.r - s3;
      b1[i].i = ss.i;

      my_r = 0.0;
      ID0 = row_flag[i+1];

      if ( (i+1)<=is2[myid] || (i+1)<=ie2[myid] ){

	if (is2[myid]<(i+1)) is0 = i + 1;
	else                 is0 = is2[myid];

	for (i1=is0; i1<=ie2[myid]; i1++){

	  i2 = i1 + (myid - ID0);
	  p[i2].r = 0.0;
	  p[i2].i = 0.0;

	  for (j=i+1; j<=n; j++){
	    p[i2].r += ac[i1][j].r * u[j].r - ac[i1][j].i * u[j].i;
	    p[i2].i += ac[i1][j].r * u[j].i + ac[i1][j].i * u[j].r;
	  }

	  p[i2].r = p[i2].r / u1.r;
	  p[i2].i = p[i2].i / u1.r;

	  my_r += u[i1].r * p[i2].r + u[i1].i * p[i2].i;
	}
        
        p[ie2[myid]+(myid-ID0)+1].r = my_r; 
      }

      if (measure_time==1){
	dtime(&Etime1);
	time1b += Etime1 - Stime1;      
      }

      if (measure_time==1) dtime(&Stime1);

      /* MPI: p */

      if (ID0<=myid){ 

        /* the number of working processors */

	nump = numprocs - ID0;

        /* sending */

	num0 = ie2[myid] - is2[myid] + 2;
	k0 = is2[myid] + (myid - ID0);

	i2 = 2*k0 - 1;
	for (i1=k0; i1<=(num0+k0-1); i1++){
	  a1d[i2] = p[i1].r;  i2++;
	  a1d[i2] = p[i1].i;  i2++;
	}

	for (ID=0; ID<nump; ID++){
	  IDS = (myid + ID) % nump + ID0; 
          if(IDS==myid) continue;
          MPI_Isend(&a1d[2*k0-1], 2*num0, MPI_DOUBLE, IDS, tag, MPI_Current_Comm_WD, &request_send[IDS-ID0]);
	}

        /* receiving */

	for (ID=0; ID<nump; ID++){
	  IDR = (myid - ID + nump) % nump + ID0;
          if(IDR==myid) continue;
          num1 = ie2[IDR] - is2[IDR] + 2;
          k1 = is2[IDR] + (IDR - ID0);
	  MPI_Irecv(&a1d[2*k1-1], 2*num1, MPI_DOUBLE, IDR, tag, MPI_Current_Comm_WD, &request_recv[IDR-ID0]);
	}

        /* waitall */

	request_send[myid-ID0] = MPI_REQUEST_NULL; 
        request_recv[myid-ID0] = MPI_REQUEST_NULL;

	MPI_Waitall(nump,request_recv,stat_send);
	MPI_Waitall(nump,request_send,stat_send);

        /* a1d -> p */
         
        i2 = 2*is2[ID0] - 1;
        for (i1=is2[ID0]; i1<=ie2[numprocs-1]+(numprocs-ID0); i1++){
          p[i1].r = a1d[i2];  i2++;
          p[i1].i = a1d[i2];  i2++;
        }
      }

      if (measure_time==1){
	dtime(&Etime1);
	time1c += Etime1-Stime1;      
      }

      if (measure_time==1) dtime(&Stime1);

      /* calculate r */

      r = 0.0;
      for (ID=ID0; ID<numprocs; ID++){
        r += p[ie2[ID]+(ID-ID0)+1].r; 
      }

      /* shift p */
      
      for (ID=(ID0+1); ID<numprocs; ID++){
        ID1 = ID - ID0;
	for (i1=is2[ID]; i1<=ie2[ID]; i1++){
          p[i1] = p[i1+ID1];
	}
      }

      /* calculate q */

      if ( (i+1)<=is2[myid] || (i+1)<=ie2[myid] ){

        r = 0.5*r / u2.r;
        br =  ar*r;
        bi = -ai*r;

        for (i1=i+1; i1<=n; i1++){
 	  tmp1.r = 0.5*(p[i1].r - (br * u[i1].r - bi*u[i1].i));
	  tmp1.i = 0.5*(p[i1].i - (br * u[i1].i + bi*u[i1].r));
	  q[i1].r = ar * tmp1.r - ai * tmp1.i; 
	  q[i1].i = ar * tmp1.i + ai * tmp1.r; 
        }
      }

      if ( (i+1)<=is2[myid] || (i+1)<=ie2[myid] ){

	if (is2[myid]<(i+1)) is0 = i + 1;
	else                 is0 = is2[myid];

	for (i1=is0; i1<=ie2[myid]; i1++){

	  tmp1.r = u[i1].r;
	  tmp1.i = u[i1].i;
	  tmp2.r = q[i1].r; 
	  tmp2.i = q[i1].i;

	  for (j1=i+1; j1<=n; j1++){
	    ac[i1][j1].r -= ( tmp1.r * q[j1].r + tmp1.i * q[j1].i
			     +tmp2.r * u[j1].r + tmp2.i * u[j1].i );
	    ac[i1][j1].i -= (-tmp1.r * q[j1].i + tmp1.i * q[j1].r
			     -tmp2.r * u[j1].i + tmp2.i * u[j1].r );
	  }
	}
      }

      if (measure_time==1){
        dtime(&Etime1);
        time1d += Etime1-Stime1;
      }

      if (myid<row_flag[i+1]) break;

    }
  }

  MPI_Barrier(MPI_Current_Comm_WD);

  if (measure_time==1) dtime(&Stime1);

  /* broadcast uu, b1, alphar, and alphai */

  MPI_Bcast(&uu[0],n+1,MPI_DOUBLE,numprocs-1,MPI_Current_Comm_WD);

  i1 = 0;
  for (i=1; i<=n; i++){
    a1d[i1] = b1[i].r; i1++;
    a1d[i1] = b1[i].i; i1++;
  }

  for (i=1; i<=n; i++){
    a1d[i1] = alphar[i]; i1++;
    a1d[i1] = alphai[i]; i1++;
  }  
  
  MPI_Bcast(&a1d[0],4*n,MPI_DOUBLE,numprocs-1,MPI_Current_Comm_WD);
  
  i1 = 0;
  for (i=1; i<=n; i++){
    b1[i].r = a1d[i1]; i1++;
    b1[i].i = a1d[i1]; i1++;
  }
  
  for (i=1; i<=n; i++){
    alphar[i] = a1d[i1]; i1++;
    alphai[i] = a1d[i1]; i1++;
  } 

 /* broadcast ac */

  BroadCast_ComplexMatrix(MPI_Current_Comm_WD,ac,n,is2,ie2,myid,numprocs,
                          stat_send,request_send,request_recv);

  if (measure_time==1){
    dtime(&Etime1);
    time1e += Etime1-Stime1;      
  }

  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j].r = ac[i][j].r;
      ad[i][j].i = ac[i][j].i;
    }
  }

  if (measure_time==1){
    printf("T0  myid=%2d time1a=%15.12f\n",myid,time1a);
    printf("T0  myid=%2d time1b=%15.12f\n",myid,time1b);
    printf("T0  myid=%2d time1c=%15.12f\n",myid,time1c);
    printf("T0  myid=%2d time1d=%15.12f\n",myid,time1d);
    printf("T0  myid=%2d time1e=%15.12f\n",myid,time1e);
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T1 myid=%2d   %15.12f\n",myid,Etime-Stime);
  }

  /****************************************************
                call a lapack routine
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (i=1; i<=n; i++){
    a1d[i-1]   = ad[i][i].r;
    a1d[i-1+n] = ad[i][i+1].r;
  }

  if      (dste_flag==0) lapack_dstegr2(n,EVmax,&a1d[0],&a1d[n],ko,ac);
  else if (dste_flag==1) lapack_dstedc2(n,&a1d[0],&a1d[n],ko,ac);
  else if (dste_flag==2){

    /*
    lapack_dstevx2(n,EVmax,&a1d[0],&a1d[n],ko,ac,1);
    */

    if (is1[myid]<=ie1[myid]){
      lapack_dstevx5(n,is1[myid],ie1[myid],&a1d[0],&a1d[n],ko,ac,1);

      /* MPI_Bcast */
      for (ID=0; ID<numprocs; ID++){

	num1 = ie1[ID] - is1[ID] + 1;
	i = is1[ID];

	if (0<num1){
	  MPI_Bcast(&ko[i], num1, MPI_DOUBLE, ID, MPI_Current_Comm_WD);
	}
      }

    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T2 myid=%2d   %15.12f\n",myid,Etime-Stime); 
  }

  /****************************************************
    transformation of eigenvectors to original space
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  /* ad stores u */
  for (i=2; i<=n; i++){
    ad[i-1][i].r = b1[i-1].r;
    ad[i-1][i].i =-b1[i-1].i;
    ad[i][i-1].r = b1[i-1].r;
    ad[i][i-1].i = b1[i-1].i;
  }

  for (k=is1[myid]; k<=(ie1[myid]-1); k+=2){
    for (nn=1; nn<=n-1; nn++){

      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){

	tmp0.r = 0.0;	tmp0.i = 0.0;
	tmp1.r = 0.0;	tmp1.i = 0.0;

	for (i=n-nn+1; i<=n; i++){
	  tmp0.r += ad[n-nn][i].r * ac[k+0][i].r - ad[n-nn][i].i * ac[k+0][i].i;
	  tmp0.i += ad[n-nn][i].i * ac[k+0][i].r + ad[n-nn][i].r * ac[k+0][i].i;
	  tmp1.r += ad[n-nn][i].r * ac[k+1][i].r - ad[n-nn][i].i * ac[k+1][i].i;
	  tmp1.i += ad[n-nn][i].i * ac[k+1][i].r + ad[n-nn][i].r * ac[k+1][i].i;
	}

	ss0.r = (alphar[n-nn]*tmp0.r - alphai[n-nn]*tmp0.i) / uu[n-nn];
	ss0.i = (alphar[n-nn]*tmp0.i + alphai[n-nn]*tmp0.r) / uu[n-nn];

	ss1.r = (alphar[n-nn]*tmp1.r - alphai[n-nn]*tmp1.i) / uu[n-nn];
	ss1.i = (alphar[n-nn]*tmp1.i + alphai[n-nn]*tmp1.r) / uu[n-nn];

	for (i=n-nn+1; i<=n; i++){
	  ac[k+0][i].r -= ss0.r * ad[n-nn][i].r + ss0.i * ad[n-nn][i].i;
	  ac[k+0][i].i -=-ss0.r * ad[n-nn][i].i + ss0.i * ad[n-nn][i].r;
	  ac[k+1][i].r -= ss1.r * ad[n-nn][i].r + ss1.i * ad[n-nn][i].i;
	  ac[k+1][i].i -=-ss1.r * ad[n-nn][i].i + ss1.i * ad[n-nn][i].r;
	}

      }
    }
  }

  ks = ie1[myid] + 1 - (ie1[myid]+1-is1[myid])%2;

  for (k=ks; (is1[myid]<=k && k<=ie1[myid]); k++){
    for (nn=1; nn<=n-1; nn++){

      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){

	tmp1.r = 0.0;
	tmp1.i = 0.0;

	for (i=n-nn+1; i<=n; i++){
	  tmp1.r += ad[n-nn][i].r * ac[k][i].r - ad[n-nn][i].i * ac[k][i].i;
	  tmp1.i += ad[n-nn][i].i * ac[k][i].r + ad[n-nn][i].r * ac[k][i].i;
	}

	ss.r = (alphar[n-nn]*tmp1.r - alphai[n-nn]*tmp1.i) / uu[n-nn];
	ss.i = (alphar[n-nn]*tmp1.i + alphai[n-nn]*tmp1.r) / uu[n-nn];
	for (i=n-nn+1; i<=n; i++){
	  ac[k][i].r -= ss.r * ad[n-nn][i].r + ss.i * ad[n-nn][i].i;
	  ac[k][i].i -=-ss.r * ad[n-nn][i].i + ss.i * ad[n-nn][i].r;
	}

      }
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T3 myid=%2d   %15.12f\n",myid,Etime-Stime);
  }

  /****************************************************
                     normalization
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (j=is1[myid]; j<=ie1[myid]; j++){
    sum = 0.0;
    for (i=1; i<=n; i++){
      sum += ac[j][i].r * ac[j][i].r + ac[j][i].i * ac[j][i].i;
    }
    sum = 1.0/sqrt(sum);
    for (i=1; i<=n; i++){
      ac[j][i].r = ac[j][i].r * sum;
      ac[j][i].i = ac[j][i].i * sum;
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T4 myid=%2d   %15.12f\n",myid,Etime-Stime);
  }

  /******************************************************
   MPI:

    broadcast the full ac matrix in Host_ID
  ******************************************************/

  if (bcast_flag==1){

    if (measure_time==1) dtime(&Stime);

    BroadCast_ComplexMatrix(MPI_Current_Comm_WD,ac,n,is1,ie1,myid,numprocs,
                            stat_send,request_send,request_recv);

    if (measure_time==1){
      dtime(&Etime);
      printf("T5 myid=%2d   %15.12f\n",myid,Etime-Stime);
    }
  }

  /****************************************************
                     transpose ac
  ****************************************************/

  for (i=1; i<=n; i++){
    for (j=(i+1); j<=n; j++){
      tmp1 = ac[i][j];
      tmp2 = ac[j][i];
      ac[i][j] = tmp2;
      ac[j][i] = tmp1;
    }
  }

  /****************************************************
                  freeing of arrays:
  ****************************************************/

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);
  free(is2);
  free(ie2);

  free(row_flag);

  for (i=0; i<n2; i++){  
    free(ad[i]);
  }
  free(ad);

  free(b1);
  free(u);
  free(uu);
  free(p);  
  free(q);
  free(a1d);
  free(alphar);
  free(alphai);
}




void Eigen_Original_PHH(MPI_Comm MPI_Current_Comm_WD,
                        dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag)
{
  double ABSTOL=LAPACK_ABSTOL;
  dcomplex **ad,*u,*b1,*p,*q,tmp1,tmp2,u1,u2,p1,ss;
  double *D,*E,*uu,*alphar,*alphai,
         s1,s2,s3,r,
         sum,ar,ai,br,bi,e,
         a1,a2,a3,a4,a5,a6,b7,r1,r2,
         r3,x1,x2,xap,
         bb,bb1,ui,uj,uij;

  int jj,jj1,jj2,k,ii,ll,i3,i2,j2,
      i,j,i1,j1,n1,n2,ik,
      jk,po1,nn,count;

  double Stime, Etime;
  double Stime1, Etime1,time1,time2,time1a,time1b,time1c;
  int *is1,*ie1,*is2,*ie2;
  double av_num;
  int numprocs,myid,tag=999,ID;

  MPI_Status stat;
  MPI_Request request;

  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(MPI_Current_Comm_WD,&numprocs);
  MPI_Comm_rank(MPI_Current_Comm_WD,&myid);

  /****************************************************
         find the numbers of partions for MPI
  ****************************************************/

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  if ( numprocs<=EVmax ){

    av_num = (double)EVmax/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = EVmax; 
  }

  else {
    for (ID=0; ID<EVmax; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=EVmax; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /****************************************************
    allocation of arrays:
  ****************************************************/

  is2 = (int*)malloc(sizeof(int)*numprocs);
  ie2 = (int*)malloc(sizeof(int)*numprocs);

  n2 = n + 5;

  ad = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    ad[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  b1 = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  u = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  uu = (double*)malloc(sizeof(double)*n2);
  p = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  q = (dcomplex*)malloc(sizeof(dcomplex)*n2);

  D = (double*)malloc(sizeof(double)*n2);
  E = (double*)malloc(sizeof(double)*n2);

  alphar = (double*)malloc(sizeof(double)*n2);
  alphai = (double*)malloc(sizeof(double)*n2);

  for (i=1; i<=(n+2); i++){
    uu[i] = 0.0;
  }

  if (measure_time==1) printf("size n=%3d EVmax=%2d\n",n,EVmax);
  if (measure_time==1) dtime(&Stime);

  time1 = 0.0;
  time2 = 0.0;
  time1a = 0.0;
  time1b = 0.0;
  time1c = 0.0;

  /****************************************************
               Householder transformation
  ****************************************************/

  for (i=1; i<=(n-1); i++){

    /* make is2 and ie2 */

    /*
    if ( numprocs<=(n-i) ){

      av_num = (double)(n-i)/(double)numprocs;

      for (ID=0; ID<numprocs; ID++){
        is2[ID] = (int)(av_num*(double)ID) + i + 1; 
        ie2[ID] = (int)(av_num*(double)(ID+1)) + i; 
      }

      is2[0] = i + 1;
      ie2[numprocs-1] = n;
    }

    else{
      for (ID=0; ID<(n-i); ID++){
        is2[ID] = ID + i + 1; 
        ie2[ID] = ID + i + 1;
      }
      for (ID=(n-i); ID<numprocs; ID++){
        is2[ID] = i + 1; 
        ie2[ID] = i - 1;
      }
    }
    */

    s1 = ac[i+1][i].r * ac[i+1][i].r + ac[i+1][i].i * ac[i+1][i].i;
    s2 = 0.0;

    u[i+1].r = ac[i+1][i].r;
    u[i+1].i = ac[i+1][i].i;
 
    for (i1=i+2; i1<=n; i1++){

      tmp1.r = ac[i1][i].r; 
      tmp1.i = ac[i1][i].i; 

      s2 += tmp1.r*tmp1.r + tmp1.i*tmp1.i;

      u[i1].r = tmp1.r;
      u[i1].i = tmp1.i;
    }

    s3 = fabs(s1 + s2);

    if ( ABSTOL<(fabs(ac[i+1][i].r)+fabs(ac[i+1][i].i)) ){
      if (ac[i+1][i].r<0.0)  s3 =  sqrt(s3);
      else                   s3 = -sqrt(s3);
    }
    else{
      s3 = sqrt(s3);
    }

    if ( ABSTOL<fabs(s2) || i==(n-1) ){

      ss.r = ac[i+1][i].r;
      ss.i = ac[i+1][i].i;

      ac[i+1][i].r = s3;
      ac[i+1][i].i = 0.0;
      ac[i][i+1].r = s3;
      ac[i][i+1].i = 0.0;

      u[i+1].r = u[i+1].r - s3;
      u[i+1].i = u[i+1].i;
      
      u1.r = s3 * s3 - ss.r * s3;
      u1.i =         - ss.i * s3;
      u2.r = 2.0 * u1.r;
      u2.i = 2.0 * u1.i;
      
      e = u2.r/(u1.r*u1.r + u1.i*u1.i);
      ar = e*u1.r;
      ai = e*u1.i;

      /* store alpha */
      alphar[i] = ar;
      alphai[i] = ai;

      /* store u2 */
      uu[i] = u2.r;

      /* store the first component of u */
      b1[i].r = ss.r - s3;
      b1[i].i = ss.i;

      if (measure_time==1) dtime(&Stime1);

      r = 0.0;
      for (i1=i+1; i1<=n; i1++){

	p1.r = 0.0;
	p1.i = 0.0;
	for (j=i+1; j<=n; j++){
	  p1.r += ac[i1][j].r * u[j].r - ac[i1][j].i * u[j].i;
	  p1.i += ac[i1][j].r * u[j].i + ac[i1][j].i * u[j].r;
	}
	p[i1].r = p1.r / u1.r;
	p[i1].i = p1.i / u1.r;

	r += u[i1].r * p[i1].r + u[i1].i * p[i1].i;
      }

      r = 0.5*r / u2.r;

      br =  ar*r;
      bi = -ai*r;

      for (i1=i+1; i1<=n; i1++){
	tmp1.r = 0.5*(p[i1].r - (br * u[i1].r - bi*u[i1].i));
	tmp1.i = 0.5*(p[i1].i - (br * u[i1].i + bi*u[i1].r));
	q[i1].r = ar * tmp1.r - ai * tmp1.i; 
	q[i1].i = ar * tmp1.i + ai * tmp1.r; 
      }

      if (measure_time==1){
        dtime(&Etime1);
        time1 += Etime1 - Stime1;
      }


      if (measure_time==1) dtime(&Stime1);


      for (i1=i+1; i1<=n; i1++){
        tmp1.r = u[i1].r;
        tmp1.i = u[i1].i;
        tmp2.r = q[i1].r; 
        tmp2.i = q[i1].i; 
	for (j1=i+1; j1<=n; j1++){
	  ac[i1][j1].r -= ( tmp1.r * q[j1].r + tmp1.i * q[j1].i
                           +tmp2.r * u[j1].r + tmp2.i * u[j1].i );
	  ac[i1][j1].i -= (-tmp1.r * q[j1].i + tmp1.i * q[j1].r
                           -tmp2.r * u[j1].i + tmp2.i * u[j1].r );
	}
      }



      if (measure_time==1){
        dtime(&Etime1);
        time2 += Etime1-Stime1;
      }


    }
  }

  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j].r = ac[i][j].r;
      ad[i][j].i = ac[i][j].i;
    }
  }


  if (measure_time==1){
    printf("T0  myid=%2d time1 =%15.12f time2 =%15.12f\n",myid,time1,time2);
    printf("T0a myid=%2d time1a=%15.12f time1b=%15.12f time1c=%15.12f\n",myid,time1a,time1b,time1c);
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T1 myid=%2d   %15.12f\n",myid,Etime-Stime);
  }

  /****************************************************
                call a lapack routine
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (i=1; i<=n; i++){
    D[i-1] = ad[i][i].r;
    E[i-1] = ad[i][i+1].r;
  }

  if      (dste_flag==0) lapack_dstegr2(n,EVmax,D,E,ko,ac);
  else if (dste_flag==1) lapack_dstedc2(n,D,E,ko,ac);
  else if (dste_flag==2) lapack_dstevx2(n,EVmax,D,E,ko,ac,1);

  if (measure_time==1){
    dtime(&Etime);
    printf("T2   %15.12f\n",Etime-Stime);
  }

  /****************************************************
    transformation of eigenvectors to original space
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  /* ad stores u */
  for (i=2; i<=n; i++){
    ad[i-1][i].r = b1[i-1].r;
    ad[i-1][i].i =-b1[i-1].i;
    ad[i][i-1].r = b1[i-1].r;
    ad[i][i-1].i = b1[i-1].i;
  }
  

  for (k=is1[myid]; k<=ie1[myid]; k++){
    for (nn=1; nn<=n-1; nn++){

      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){

	tmp1.r = 0.0;
	tmp1.i = 0.0;

	for (i=n-nn+1; i<=n; i++){
	  tmp1.r += ad[n-nn][i].r * ac[k][i].r - ad[n-nn][i].i * ac[k][i].i;
	  tmp1.i += ad[n-nn][i].i * ac[k][i].r + ad[n-nn][i].r * ac[k][i].i;
	}

	ss.r = (alphar[n-nn]*tmp1.r - alphai[n-nn]*tmp1.i) / uu[n-nn];
	ss.i = (alphar[n-nn]*tmp1.i + alphai[n-nn]*tmp1.r) / uu[n-nn];
	for (i=n-nn+1; i<=n; i++){
	  ac[k][i].r -= ss.r * ad[n-nn][i].r + ss.i * ad[n-nn][i].i;
	  ac[k][i].i -=-ss.r * ad[n-nn][i].i + ss.i * ad[n-nn][i].r;
	}

      }
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T3 myid=%2d   %15.12f\n",myid,Etime-Stime);
  }

  /****************************************************
                     normalization
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (j=is1[myid]; j<=ie1[myid]; j++){
    sum = 0.0;
    for (i=1; i<=n; i++){
      sum += ac[j][i].r * ac[j][i].r + ac[j][i].i * ac[j][i].i;
    }
    sum = 1.0/sqrt(sum);
    for (i=1; i<=n; i++){
      ac[j][i].r = ac[j][i].r * sum;
      ac[j][i].i = ac[j][i].i * sum;
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T4 myid=%2d   %15.12f\n",myid,Etime-Stime);
  }

  /******************************************************
   MPI:

    broadcast the full ac matrix in Host_ID
  ******************************************************/

  if (bcast_flag==1){

    if (measure_time==1) dtime(&Stime);

    BroadCast_ComplexMatrix(MPI_Current_Comm_WD,ac,n,is1,ie1,myid,numprocs,
                            stat_send,request_send,request_recv);


    if (measure_time==1){
      dtime(&Etime);
      printf("T5 myid=%2d   %15.12f\n",myid,Etime-Stime);
    }
  }

  /****************************************************
                     transpose ac
  ****************************************************/

  for (i=1; i<=n; i++){
    for (j=(i+1); j<=n; j++){
      tmp1 = ac[i][j];
      tmp2 = ac[j][i];
      ac[i][j] = tmp2;
      ac[j][i] = tmp1;
    }
  }

  /****************************************************
                  freeing of arrays:
  ****************************************************/

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);
  free(is2);
  free(ie2);

  for (i=0; i<n2; i++){
    free(ad[i]);
  }
  free(ad);

  free(b1);
  free(u);
  free(uu);
  free(p);
  free(q);
  free(D);
  free(E);
  free(alphar);
  free(alphai);
}

