/*----------------------------------------------------------------------
  exx_file_overlap.h
----------------------------------------------------------------------*/
#ifndef EXX_FILE_OVERLAP_H_INCLUDED
#define EXX_FILE_OVERLAP_H_INCLUDED


void EXX_File_Overlap_Write(
  int          ndglf,     /* number of double for GLF matrix */
  int          nop_local, /* number of OPs asigned to each node */
  int          nbmax,     /* max number of basis */
  int          jmax,
  double     **buf,       /* [nop_local][ndglf*nbmax*nbmax] */
  const char  *path       /* path to the file */
);


void EXX_File_Overlap_Read(
  int         ndglf,
  int         nbmax,
  int         jmax,
  int         iop,
  double     *buf,   /* [ndglf*nbmax*nbmax] */
  const char *path 
);


#endif /* EXX_FILE_OVERLAP_H_INCLUDED */
