/**********************************************************************
  TRAN_Deallocate_RestartFile.c:

  TRAN_Deallocate_RestartFile.c is a subroutine to deallocate arrays
  storing a lots of data of electrodes.

  Log of TRAN_Deallocate_RestartFile.c:

     11/Dec/2005   Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include "tran_variables.h"


void TRAN_Deallocate_RestartFile(char *position)
{
  int side,spin;
  int i, k, Mc_AN, Gc_AN, tno0, Cwan,  tno1,  Gh_AN,m , h_AN, Hwan; 

  double ScaleSize_t;
  int  SpinP_switch_t, atomnum_t, SpeciesNum_t, Max_FSNAN_t, TCpyCell_t, Matomnum_t, MatomnumF_t, MatomnumS_t;
  int *WhatSpecies_t;
  int *Spe_Total_CNO_t;
  int *Spe_Total_NO_t;
  int *FNAN_t;
  int **natn_t;
  int **ncn_t;
  int **atv_ijk_t;
  double *****OLP_t;
  double *****H_t;
  double ******DM_t;

  double *dDen_Grid_t;
  double *dVHart_Grid_t;
  double **Gxyz_t;

  if ( strcasecmp(position,"left")==0) {
    side = 0;
  } else if ( strcasecmp(position,"right")==0) {
    side = 1;
  } 

  ScaleSize_t=ScaleSize_e[side];
  SpinP_switch_t=SpinP_switch_e[side];
  atomnum_t=atomnum_e[side];
  SpeciesNum_t=SpeciesNum_e[side];
  Max_FSNAN_t=Max_FSNAN_e[side];
  TCpyCell_t=TCpyCell_e[side];
  Matomnum_t=Matomnum_e[side];
  MatomnumF_t=MatomnumF_e[side];
  MatomnumS_t=MatomnumS_e[side];
  WhatSpecies_t=WhatSpecies_e[side];
  Spe_Total_CNO_t=Spe_Total_CNO_e[side];
  Spe_Total_NO_t=Spe_Total_NO_e[side];
  FNAN_t=FNAN_e[side];
  natn_t=natn_e[side];
  ncn_t=ncn_e[side];
  atv_ijk_t=atv_ijk_e[side];
  OLP_t=OLP_e[side];
  H_t=H_e[side];
  DM_t=DM_e[side];
  dDen_Grid_t=dDen_Grid_e[side];
  dVHart_Grid_t=dVHart_Grid_e[side];
  Gxyz_t=Gxyz_e[side];

  for (i=0;i<TCpyCell_t+1;i++) {
    free(atv_ijk_t[i]);
  }
  free( atv_ijk_t );

  for (k=0; k<4 ; k++){
    for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){
      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = Mc_AN;
	Cwan = WhatSpecies_t[Gc_AN];
	tno0 = Spe_Total_NO_t[Cwan];  
      }    
      for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){

	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn_t[Gc_AN][h_AN];
	  Hwan = WhatSpecies_t[Gh_AN];
	  tno1 = Spe_Total_NO_t[Hwan];
	} 

	for (i=0; i<tno0; i++){
	  free( OLP_t[k][Mc_AN][h_AN][i] );
	}
	free( OLP_t[k][Mc_AN][h_AN] );
      }
      free( OLP_t[k][Mc_AN] );
    }
    free( OLP_t[k] );
  }
  free( OLP_t );

  for (k=0; k<=SpinP_switch_t; k++){
    for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){
      if (Mc_AN==0){
	Gc_AN = 0;
	tno0 = 1;
      }
      else{
	Gc_AN = Mc_AN;
	Cwan = WhatSpecies_t[Gc_AN];
	tno0 = Spe_Total_NO_t[Cwan];  
      }    
      for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){
	if (Mc_AN==0){
	  tno1 = 1;  
	}
	else{
	  Gh_AN = natn_t[Gc_AN][h_AN];
	  Hwan = WhatSpecies_t[Gh_AN];
	  tno1 = Spe_Total_NO_t[Hwan];
	} 
	for (i=0; i<tno0; i++){
	  free( H_t[k][Mc_AN][h_AN][i] );
	}
	free( H_t[k][Mc_AN][h_AN] );
      }
      free( H_t[k][Mc_AN] );
    }
    free( H_t[k] );
  }
  free( H_t );

  for (m=0; m<1; m++){
    for (k=0; k<=SpinP_switch_t; k++){
      for (Mc_AN=1; Mc_AN<=atomnum_t; Mc_AN++){
	if (Mc_AN==0){
	  Gc_AN = 0;
	  tno0 = 1;
	}
	else{
	  Gc_AN = Mc_AN;
	  Cwan = WhatSpecies_t[Gc_AN];
	  tno0 = Spe_Total_NO_t[Cwan];  
	}    
	for (h_AN=0; h_AN<=FNAN_t[Gc_AN]; h_AN++){
	  if (Mc_AN==0){
	    tno1 = 1;  
	  }
	  else{
	    Gh_AN = natn_t[Gc_AN][h_AN];
	    Hwan = WhatSpecies_t[Gh_AN];
	    tno1 = Spe_Total_NO_t[Hwan];
	  } 
	  for (i=0; i<tno0; i++){
	    free( DM_t[m][k][Mc_AN][h_AN][i] );
	  }
	  free( DM_t[m][k][Mc_AN][h_AN] );
	}
	free(  DM_t[m][k][Mc_AN] );
      }
      free( DM_t[m][k] );
    }
    free( DM_t[m] );
  }
  free( DM_t );

  free( dDen_Grid_t );

  free( dVHart_Grid_t );

  for (i=0;i<= atomnum_t; i++) {
    free(natn_t[i]);
  }
  free(natn_t);

  for (i=0;i<= atomnum_t; i++) {
    free(ncn_t[i]);
  }
  free(ncn_t);

  free(WhatSpecies_t);
  free(Spe_Total_CNO_t);
  free(Spe_Total_NO_t);
  free(FNAN_t);

  for (k=0; k<(atomnum_t+1); k++){
    free(Gxyz_t[k]);
  }
  free(Gxyz_t);

}

