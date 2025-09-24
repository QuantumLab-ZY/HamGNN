#include <stdio.h>
#include <stdlib.h>
#include "openmx_common.h"

void Init_List_YOUSO()
{
  int i;

  for (i=0; i<NYOUSO; i++){
    List_YOUSO[i] = 0;
  }

  List_YOUSO[6]  = 130;
  List_YOUSO[10] = 100;
  List_YOUSO[14] = 104;
  List_YOUSO[26] =  60;
}
