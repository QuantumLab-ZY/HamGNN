/*----------------------------------------------------------------------
  exx_debug.h
----------------------------------------------------------------------*/
#ifndef EXX_DEBUG_H_INCLUDED
#define EXX_DEBUG_H_INCLUDED

#include "exx.h"

void EXX_Debug_Copy_DM(
  int MaxN,
  double *****CDM,
  EXX_t  *exx,
  dcomplex ****exx_CDM,
  int symbrk
);


void EXX_Initial_DM(
  EXX_t *exx,
  dcomplex ****exx_CDM
);


void EXX_Debug_Check_DM(
  EXX_t *exx, 
  dcomplex ****exx_DM, 
  double *****DM
);

#endif
