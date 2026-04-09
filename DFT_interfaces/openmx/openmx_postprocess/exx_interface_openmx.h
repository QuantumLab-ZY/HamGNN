/*----------------------------------------------------------------------
  exx_interface_openmx.h
----------------------------------------------------------------------*/
#ifndef EXX_INTERFACE_OPENMX_H_INCLUDED
#define EXX_INTERFACE_OPENMX_H_INCLUDED

#include "exx.h"

/* global object */
EXX_t *g_exx;


/*
  size: exx_DM[List_YOUSO[16]]
          [SpinP_switch+1]
          [nop]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Hwan]] 
*/
extern dcomplex *****g_exx_DM; 
extern double g_exx_U[2];
extern int g_exx_switch;
extern int g_exx_switch_output_DM;
extern int g_exx_switch_rhox;

void EXX_on_OpenMX_Init(MPI_Comm comm);
void EXX_Time_Report(void);
void EXX_on_OpenMX_Free(void);



void EXX_Cluster(
  double **H,
  double *Ux,
  EXX_t *exx,
  dcomplex ***exx_CDM, 
  const int *MP
);


void EXX_Simple_Mixing_DM(
  EXX_t *exx,
  double mix_wgt,
  dcomplex ***exx_CDM,
  dcomplex ***exx_PDM,
  dcomplex ***exx_P2DM
);


void EXX_Fock_Band(
  dcomplex **H,
  EXX_t *exx,
  dcomplex ****exx_CDM, 
  int    *MP,
  double k1,  
  double k2,  
  double k3,
  int spin
);

void EXX_Energy_Band(
  double *Ux,
  EXX_t *exx,
  dcomplex ****exx_CDM, 
  int    *MP
);

void EXX_Reduce_DM(EXX_t *exx, dcomplex ***exx_DM);

#endif /* EXX_INTERFACE_OPENMX_H_INCLUDED */
