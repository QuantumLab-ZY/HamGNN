/**********************************************************************
  PrintMemory_Fix.c:

     PrintMemory_Fix.c is a subroutine to print the size of arrays 
     with fixed sizes.

  Log of PrintMemory_Fix.c:

     24/May/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_common.h"

void PrintMemory_Fix()
{
  int ASIZE15,ASIZE18,ASIZE19,ASIZE21;
  int ASIZE22,ASIZE24,ASIZE25;

  /****************************************************
    PrintMemory 
    Allocate_Arrays(6) in SetPara_DFT
  ****************************************************/

  ASIZE15 = List_YOUSO[15];
  ASIZE18 = List_YOUSO[18];
  ASIZE19 = List_YOUSO[19];
  ASIZE21 = List_YOUSO[21];
  ASIZE22 = List_YOUSO[22];
  ASIZE24 = List_YOUSO[24];
  ASIZE25 = List_YOUSO[25];

  PrintMemory("SetPara_DFT: Spe_PAO_XV",sizeof(double)*ASIZE18*ASIZE21,NULL);
  PrintMemory("SetPara_DFT: Spe_PAO_RV",sizeof(double)*ASIZE18*ASIZE21,NULL);
  PrintMemory("SetPara_DFT: Spe_Atomic_Den",sizeof(double)*ASIZE18*ASIZE21,NULL);
  PrintMemory("SetPara_DFT: Spe_PAO_RWF",sizeof(double)*ASIZE18*(ASIZE25+1)*
                                                        ASIZE24*ASIZE21,NULL);
  PrintMemory("SetPara_DFT: Spe_RF_Bessel",sizeof(double)*ASIZE18*(ASIZE25+1)*
                                                          ASIZE24*ASIZE15,NULL);

  /****************************************************
    PrintMemory 
    Allocate_Arrays(7) in SetPara_DFT
  ****************************************************/

  PrintMemory("SetPara_DFT: Spe_VPS_XV",sizeof(double)*ASIZE18*ASIZE22,NULL);
  PrintMemory("SetPara_DFT: Spe_VPS_RV",sizeof(double)*ASIZE18*ASIZE22,NULL);
  PrintMemory("SetPara_DFT: Spe_Vna",sizeof(double)*ASIZE18*ASIZE22,NULL);
  PrintMemory("SetPara_DFT: Spe_VH_Atom",sizeof(double)*ASIZE18*ASIZE22,NULL);
  PrintMemory("SetPara_DFT: Spe_Atomic_PCC",sizeof(double)*ASIZE18*ASIZE22,NULL);
  PrintMemory("SetPara_DFT: Spe_VNL",sizeof(double)*ASIZE18*ASIZE19*ASIZE22,NULL);
  PrintMemory("SetPara_DFT: Spe_VNLE",sizeof(double)*ASIZE18*ASIZE19,NULL);
  PrintMemory("SetPara_DFT: Spe_VPS_List",sizeof(double)*ASIZE18*ASIZE19,NULL);
  PrintMemory("SetPara_DFT: Spe_NLRF_Bessel",sizeof(double)*ASIZE18*(ASIZE19+2)*
                                                            ASIZE15,NULL);

  /* allocated in SetPara_DFT.c */

  PrintMemory("SetPara_DFT: HOMOs_Coef", sizeof(dcomplex)*List_YOUSO[33]*
                                                         List_YOUSO[23]*
                                                         List_YOUSO[31]*
                                                         List_YOUSO[1]*
                                                         List_YOUSO[7], NULL);
  PrintMemory("SetPara_DFT: LUMOs_Coef", sizeof(dcomplex)*List_YOUSO[33]*
                                                         List_YOUSO[23]*
                                                         List_YOUSO[32]*
                                                         List_YOUSO[1]*
                                                         List_YOUSO[7], NULL);
}


