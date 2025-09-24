/*----------------------------------------------------------------------
  exx_step1.h
----------------------------------------------------------------------*/
#ifndef EXX_STEP1_H_INCLUDED
#define EXX_STEP1_H_INCLUDED

#include "exx.h"

int EXX_Step1(
  EXX_t  *exx,
  int    natom,
  int    *atom_sp,
  int    nspec, 
  int    *spec_nb,     /* [nspec] */
  double *spec_rc,     /* [nspec] */ 
  int    **spec_l,    /* [nspec][nbmax] */
  int    **spec_m,    /* [nspec][nbmax] */
  int    **spec_mul,  /* [nspec][nbmax] */
  double ****spec_fr, /* [nspec][lmax][mulmax][nmeshmax] */
  double **spec_xr,   /* [nspec][nmeshmax] */  
  int    *spec_nmesh  /* [nspec] */
);

#endif /* EXX_STEP1_H_INCLUDED */
