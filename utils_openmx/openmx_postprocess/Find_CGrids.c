#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"

static void Find_ln(int i0, int n0, int res[2]);

void Find_CGrids(int Real_Position, int n1, int n2, int n3,
                 double Cxyz[4], int NOC[4])
{
  /*********************************************************
   Real_Position==0
       gives the coordinates in the original cell (Rn==0)

   Real_Position==1
       gives the coordinates in the translated cell
  **********************************************************/

  int Rn,l1,l2,l3,N;
  int nn1,nn2,nn3,res[2];
  double x0,y0,z0;
  
  Find_ln(1,n1,res);
  l1  = res[0];
  nn1 = res[1];
  
  Find_ln(2,n2,res);
  l2  = res[0];
  nn2 = res[1];
  
  Find_ln(3,n3,res);
  l3  = res[0];
  nn3 = res[1];
   
  if (CpyCell<abs(l1) || CpyCell<abs(l2) || CpyCell<abs(l3)){

    /* outside of Tcell */

    Rn = 0;
    x0 = atv[Rn][1];
    y0 = atv[Rn][2];
    z0 = atv[Rn][3];

    Cxyz[1] = 10e+5;
    Cxyz[2] = 10e+5;
    Cxyz[3] = 10e+5;

  }
  else{
  
    Rn = R_atv(CpyCell,l1,l2,l3);
    x0 = atv[Rn][1];
    y0 = atv[Rn][2];
    z0 = atv[Rn][3];
  
    /*  N = nn1*Ngrid2*Ngrid3 + nn2*Ngrid3 + nn3;  */
  
    if (Real_Position==0){
      Cxyz[1] = (double)nn1*gtv[1][1] + (double)nn2*gtv[2][1]
              + (double)nn3*gtv[3][1] + Grid_Origin[1];
      Cxyz[2] = (double)nn1*gtv[1][2] + (double)nn2*gtv[2][2]
              + (double)nn3*gtv[3][2] + Grid_Origin[2];
      Cxyz[3] = (double)nn1*gtv[1][3] + (double)nn2*gtv[2][3]
              + (double)nn3*gtv[3][3] + Grid_Origin[3];
    }
    else{
      Cxyz[1] = x0 
              + (double)nn1*gtv[1][1] + (double)nn2*gtv[2][1]
              + (double)nn3*gtv[3][1] + Grid_Origin[1];
      Cxyz[2] = y0
              + (double)nn1*gtv[1][2] + (double)nn2*gtv[2][2]
              + (double)nn3*gtv[3][2] + Grid_Origin[2];
      Cxyz[3] = z0
              + (double)nn1*gtv[1][3] + (double)nn2*gtv[2][3]
              + (double)nn3*gtv[3][3] + Grid_Origin[3];
    }
  }

  NOC[0] = Rn;
  NOC[1] = nn1;
  NOC[2] = nn2;
  NOC[3] = nn3;
}


void Find_ln(int i0, int n0, int res[2])
{
  int l0,n1,N;

  if      (i0==1) N = Ngrid1;
  else if (i0==2) N = Ngrid2;
  else if (i0==3) N = Ngrid3;

  if (n0<0){
    l0 = -((abs(n0)-1)/N + 1);
    n1 = N - (abs(n0) - N*(abs(l0)-1));
  }  
  else if (N<=n0){
    l0 = n0/N;
    n1 = n0 - N*l0;
  }
  else{
    l0 = 0;  
    n1 = n0;
  }

  res[0] = l0;
  res[1] = n1; 
}

