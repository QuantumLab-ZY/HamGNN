#include <stdio.h>
#include <math.h>
#include "openmx_common.h"

double Nonlocal_Basis(int wan, int Lnum_index, int Mnum, int so,
                      double r, double theta, double phi)
{
  int Lnum;
  double NWF,Radial_WF,Angular_WF;

  Lnum = Spe_VPS_List[wan][Lnum_index];
  Radial_WF = Nonlocal_RadialF(wan,Lnum_index-1,so,r);
  Angular_WF = AngularF(Lnum,Mnum,theta,phi,0,0.0,0.0,0.0,0.0);
  NWF = Radial_WF*Angular_WF;

  return NWF;
} 
