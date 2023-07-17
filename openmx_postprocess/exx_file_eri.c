/*----------------------------------------------------------------------
  exx_file_eri.c

  Coded by M. Toyoda, 25/NOV/2009
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "exx.h"
#include "exx_log.h"
#include "exx_file_eri.h"
#include <mpi.h>

static void cachefile_path(char *path, size_t len, const char *cachedir)
{
  int rank;

#ifdef EXX_USE_MPI
  MPI_Comm_rank(g_exx_mpicomm, &rank);
#else
  rank = EXX_ROOT_RANK;
#endif

  snprintf(path, len, "%s/eri%03d.dat", cachedir, rank);
}



void EXX_File_ERI_Create(const EXX_t *exx)
{
  int n;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t sz;

  cachefile_path(path, EXX_PATHLEN, EXX_CacheDir(exx));

  fp = fopen(path, "wb");

  /* write record number which is zero at this stage */
  n = 0;
  sz = fwrite(&n, sizeof(int), 1, fp);

  fclose(fp);
}


void EXX_File_ERI_Write(
  const EXX_t  *exx,
  const double *eri, /* [nb1*nb2*nb3*nb4*nrn] */
  int           iop1,
  int           iop2,
  int           nb1,
  int           nb2,
  int           nb3,
  int           nb4,
  int           nrn,
  const int    *iRd  /* [nrn] */
)
{
  int neri, nr, stat;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t sz; 

  cachefile_path(path, EXX_PATHLEN, EXX_CacheDir(exx));

  /* open file in 'append' mode */
  fp = fopen(path, "ab");

  /* write header information */
  sz = fwrite(&iop1, sizeof(int), 1, fp);
  sz = fwrite(&iop2, sizeof(int), 1, fp);
  sz = fwrite(&nb1,  sizeof(int), 1, fp);
  sz = fwrite(&nb2,  sizeof(int), 1, fp);
  sz = fwrite(&nb3,  sizeof(int), 1, fp);
  sz = fwrite(&nb4,  sizeof(int), 1, fp);
  sz = fwrite(&nrn,  sizeof(int), 1, fp);
  sz = fwrite(iRd,   sizeof(int), nrn, fp);

  /* write ERIs */
  neri = nb1*nb2*nb3*nb4*nrn;
  sz = fwrite(eri, sizeof(double), neri, fp);
 
  /* close file */ 
  fclose(fp);

  /* open file again in 'write-read' mode */
  fp = fopen(path, "r+b");

  /* increment record number */
  fseek(fp, 0, SEEK_SET);
  sz = fread(&nr, sizeof(int), 1, fp);
  nr++;
  fseek(fp, 0, SEEK_SET);
  sz = fwrite(&nr, sizeof(int), 1, fp);

  /* close file */
  fclose(fp);
}




int EXX_File_ERI_Read_NRecord(const EXX_t *exx)
{
  int n;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t sz;
 
  cachefile_path(path, EXX_PATHLEN, EXX_CacheDir(exx));

  fp = fopen(path, "rb");
  sz = fread(&n, sizeof(int), 1, fp);
  fclose(fp);
 
  return n;
}



void EXX_File_ERI_Read_Data_Head(
  const EXX_t *exx,
  int         record,
  int        *out_iop1,
  int        *out_iop2,
  int        *out_nb1,
  int        *out_nb2,
  int        *out_nb3,
  int        *out_nb4,
  int        *out_nrn
)
{
  int i, nr, iop1, iop2, nb1, nb2, nb3, nb4, nrn;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t cb, sz;

  cachefile_path(path, EXX_PATHLEN, EXX_CacheDir(exx));

  /* open file */
  fp = fopen(path, "rb");

  /* boundary check */
  sz = fread(&nr, sizeof(int), 1, fp);
  if (nr<record) { 
    fprintf(stderr, "  record= %d nr= %d\n", record, nr);
    EXX_ERROR("record number is too large");
  }

  /* seeking */
  for (i=0; i<record; i++) {
    /* read header information */
    sz = fread(&iop1, sizeof(int), 1, fp);
    sz = fread(&iop2, sizeof(int), 1, fp);
    sz = fread(&nb1,  sizeof(int), 1, fp);
    sz = fread(&nb2,  sizeof(int), 1, fp);
    sz = fread(&nb3,  sizeof(int), 1, fp);
    sz = fread(&nb4,  sizeof(int), 1, fp);
    sz = fread(&nrn,  sizeof(int), 1, fp);

    /* skip iRd data */
    cb = sizeof(int)*nrn;
    fseek(fp, (long)cb, SEEK_CUR);

    /* skip ERIs */
    cb = sizeof(double)*nb1*nb2*nb3*nb4*nrn;
    fseek(fp, (long)cb, SEEK_CUR);
  }

  /* read requested header data */
  sz = fread(out_iop1, sizeof(int), 1, fp);
  sz = fread(out_iop2, sizeof(int), 1, fp);
  sz = fread(out_nb1,  sizeof(int), 1, fp);
  sz = fread(out_nb2,  sizeof(int), 1, fp);
  sz = fread(out_nb3,  sizeof(int), 1, fp);
  sz = fread(out_nb4,  sizeof(int), 1, fp);
  sz = fread(out_nrn,  sizeof(int), 1, fp);
            
  fclose(fp);
}


void EXX_File_ERI_Read(
  const EXX_t *exx,
  int         record,
  double     *out_eri, /* [in_n] */
  int        *out_iRd, /* [in_n] */
  int         in_iop1,
  int         in_iop2,
  int         in_nrn,
  int         in_neri
)
{
  int i, neri, nr, iop1, iop2, nb1, nb2, nb3, nb4, nrn;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t sz, cb;  

  cachefile_path(path, EXX_PATHLEN, EXX_CacheDir(exx));

  /* open file */
  fp = fopen(path, "rb");

  /* boundary check */
  sz = fread(&nr, sizeof(int), 1, fp);
  if (nr<record) { 
    fprintf(stderr, "  record= %d nr= %d\n", record, nr);
    EXX_ERROR("record number is too large");
  }

  /* seeking */
  for (i=0; i<record; i++) {
    sz = fread(&iop1, sizeof(int), 1, fp);
    sz = fread(&iop2, sizeof(int), 1, fp);
    sz = fread(&nb1,  sizeof(int), 1, fp);
    sz = fread(&nb2,  sizeof(int), 1, fp);
    sz = fread(&nb3,  sizeof(int), 1, fp);
    sz = fread(&nb4,  sizeof(int), 1, fp);
    sz = fread(&nrn,  sizeof(int), 1, fp);

    /* skip R data */
    cb = sizeof(int)*nrn;
    fseek(fp, (long)cb, SEEK_CUR);

    /* skip ERI */
    cb = sizeof(double)*nb1*nb2*nb3*nb4*nrn;
    fseek(fp, (long)cb, SEEK_CUR);
  }

  /* read header information */
  sz = fread(&iop1, sizeof(int), 1, fp);
  sz = fread(&iop2, sizeof(int), 1, fp);
  sz = fread(&nb1,  sizeof(int), 1, fp);
  sz = fread(&nb2,  sizeof(int), 1, fp);
  sz = fread(&nb3,  sizeof(int), 1, fp);
  sz = fread(&nb4,  sizeof(int), 1, fp);
  sz = fread(&nrn,  sizeof(int), 1, fp);

  /* buffer length */ 
  neri = nb1*nb2*nb3*nb4*nrn;

  /* check consistency */
  if (neri != in_neri || nrn != in_nrn) { EXX_ERROR("file is broken"); }

  /* read R data */
  if (out_iRd) { 
    sz = fread(out_iRd, sizeof(int), nrn, fp); 
  } else {
    fseek(fp, sizeof(int)*nrn, SEEK_CUR);
  }

  /* read ERI data */
  if (out_eri) { sz = fread(out_eri, sizeof(double), neri, fp); }
 
  /* close file */ 
  fclose(fp);
}
  



