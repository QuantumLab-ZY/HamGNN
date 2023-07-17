/*----------------------------------------------------------------------
  exx_file_overlap.c 

  Coded by M. Toyoda, 25/NOV/2009
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "exx.h"
#include "exx_log.h"
#include "exx_file_overlap.h"


#ifdef EXX_USE_MPI
#include <mpi.h>
#endif /* EXX_USE_MPI */

void EXX_File_Overlap_Write(
  int          ndglf,     /* number of double for GLF matrix */
  int          nop_local, /* number of OPs asigned to each node */
  int          nbmax,     /* max number of basis */
  int          jmax,
  double     **buf,       /* [nop_local][ndglf*nbmax*nbmax] */
  const char  *path       /* path to the file */
)
{
  int myrank, nproc, nd1, iop;
  FILE *fp;
  size_t nd, sz;

#ifdef EXX_USE_MPI
  MPI_Comm comm;
  MPI_Status stat;
  double *buf_s;
  int nop_remote, iproc;
#endif /* EXX_USE_MPI */

#ifdef EXX_USE_MPI
  comm = g_exx_mpicomm;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nproc);
#else
  myrank = EXX_ROOT_RANK;
  nproc = 1;
#endif /* EXX_USE_MPI */

  nd = ndglf*nbmax*nbmax; /* length for GLF */

  /* MASTER process */
  if (EXX_ROOT_RANK==myrank) {
    fp = fopen(path, "wb");
    if (NULL==fp) { EXX_ERROR( "failed to open file" ); }
     
    /* save data on master itself */
    for (iop=0; iop<nop_local; iop++) {
      sz = fwrite(buf[iop], sizeof(double), nd, fp);
      if (sz != nd) { EXX_ERROR( "file io failed" ); }
    }

#ifdef EXX_USE_MPI
    /* recieve data from slaves */
    buf_s = (double*)malloc(sizeof(double)*nd);
    for (iproc=1; iproc<nproc; iproc++) {
      MPI_Recv(&nop_remote, 1, MPI_INT, iproc, 0, g_exx_mpicomm, &stat);
      for (iop=0; iop<nop_remote; iop++) {
        MPI_Recv(&buf_s[0], nd, MPI_DOUBLE, iproc, 2*iop+1, comm, &stat);
        sz = fwrite(buf_s, sizeof(double), nd, fp);
        if (sz != nd) { EXX_ERROR( "file io failed" ); }
      }
    }
    free(buf_s);
#endif /* EXX_USE_MPI */

    /* close file */ 
    fclose(fp);
  } 
#ifdef EXX_USE_MPI
  /* SLAVE processes */
  else { 
    MPI_Send(&nop_local, 1, MPI_INT, 0, 0,comm);
    for (iop=0; iop<nop_local; iop++) {
      MPI_Send(buf[iop], nd, MPI_DOUBLE, 0, 2*iop+1, comm);
    }
  }
#endif /* EXX_USE_MPI */  
}




void EXX_File_Overlap_Read(
  int         ndglf,
  int         nbmax,
  int         jmax,
  int         iop,
  double     *buf, /* [ndglf*nbmax*nbmax] */
  const char *path 
)
{
  size_t nd, sz;
  FILE *fp;

  /* open file */
  fp = fopen(path, "rb");
  if (NULL==fp) { EXX_ERROR( "failed to open file" ); }

  nd = ndglf*nbmax*nbmax; /* length for GLF */

  /* move file pointer */
  fseek(fp, iop*nd*sizeof(double), SEEK_SET);
 
  /* read GLF */ 
  sz = fread(buf, sizeof(double), nd, fp);
  if (sz != nd) { EXX_ERROR( "file io error" ); }

  /* close file */
  fclose(fp);
}




