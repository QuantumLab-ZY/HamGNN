/*----------------------------------------------------------------------
  exx_file_eri.h

----------------------------------------------------------------------*/
#ifndef EXX_FILE_ERI_H_INCLUDED
#define EXX_FILE_ERI_H_INCLUDED


void EXX_File_ERI_Create(const EXX_t *exx);

void EXX_File_ERI_Write(
  const EXX_t        *exx,
  const double *eri, /* [nb1*nb2*nb3*nb4*nrn] */
  int           iop1,
  int           iop2,
  int           nb1,
  int           nb2,
  int           nb3,
  int           nb4,
  int           nrn,
  const int    *iRd    /* [nrn] */
);


int EXX_File_ERI_Read_NRecord(const EXX_t *exx);

void EXX_File_ERI_Read_Data_Head(
  const EXX_t      *exx,
  int         record,
  int        *iop1,
  int        *iop2,
  int        *nb1,
  int        *nb2,
  int        *nb3,
  int        *nb4,
  int        *nrn
);


void EXX_File_ERI_Read(
  const EXX_t      *exx,
  int         record,
  double     *out_eri, /* [in_n] */
  int        *out_iRd, /* [in_n] */
  int         in_iop1,
  int         in_iop2,
  int         in_nrn,
  int         in_n
);


#endif /* EXX_FILE_ERI_H_INCLUDED */
