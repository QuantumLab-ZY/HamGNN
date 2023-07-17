/**********************************************************************
  Get_Cnt_Orbitals.c:

     Get_Cnt_Orbitals.c is a subrutine to calculate
     contracted basis orbitals

  Log of Get_Cnt_Orbitals.c:

     24/April/2002  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"

void Get_Cnt_Orbitals(int Mc_AN, double x, double y, double z, double *Chi)
{
  static int firsttime=1;
  int Gc_AN,i,j,L0,Mul0,M0,i1,al,wan,p;
  double S_coordinate[3];
  double dum,dum1,dum2,dum3,dum4,a,b;
  double siQ,coQ,siP,coP,Q,P,R,Rmin,tmp0;
  double **RF;
  double **AF;
  double SH[Supported_MaxL*2+1][2];
  double dSHt[Supported_MaxL*2+1][2];
  double dSHp[Supported_MaxL*2+1][2];

  /* Radial */
  int mp_min,mp_max,m,po;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,y12,y22,f;

  /****************************************************
     allocation of arrays:

   double RF[List_YOUSO[25]+1][List_YOUSO[24]];
   double AF[List_YOUSO[25]+1][2*(List_YOUSO[25]+1)+1];
  ****************************************************/

  RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
  for (i=0; i<(List_YOUSO[25]+1); i++){
    RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
  }

  AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
  for (i=0; i<(List_YOUSO[25]+1); i++){
    AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
  }

  /* start calc. */

  Gc_AN = F_M2G[Mc_AN];
  Rmin = 10e-14;
  wan = WhatSpecies[Gc_AN];
  xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate);
  R = S_coordinate[0];
  Q = S_coordinate[1];
  P = S_coordinate[2];

  po = 0;
  mp_min = 0;
  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
        RF[L0][Mul0] = 0.0;
      }
    }

    po = 1;
  }

  else if (R<Spe_PAO_RV[wan][0]){
    h1 = Spe_PAO_RV[wan][0] - Spe_PAO_RV[wan][1];
    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

        y1 = Spe_PAO_RWF[wan][L0][Mul0][0] - Spe_PAO_RWF[wan][L0][Mul0][1];
        a = y1/h1;
        b = Spe_PAO_RWF[wan][L0][Mul0][0] - a*Spe_PAO_RV[wan][0];

        RF[L0][Mul0] = a*R + b;
      }
    }

  }

  else{

    do{
      m = (mp_min + mp_max)/2;
      if (Spe_PAO_RV[wan][m]<R)
        mp_min = m;
      else 
       mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

    x1 = R - Spe_PAO_RV[wan][m-1];
    x2 = R - Spe_PAO_RV[wan][m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;

    dum = h1 + h2;
    dum1 = h1/h2/dum;
    dum2 = h2/h1/dum;
    dum = h2 + h3;
    dum3 = h2/h3/dum;
    dum4 = h3/h2/dum;

    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

        f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
        f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
        f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
        f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

        if (m==1){
          h1 = -(h2+h3);
          f1 = f4;
        }
        else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
          h3 = -(h1+h2);
          f4 = f1;
        }

        dum = f3 - f2;
        g1 = dum*dum1 + (f2-f1)*dum2;
        g2 = (f4-f3)*dum3 + dum*dum4;

        f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
           + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

        RF[L0][Mul0] = f;
      }
    } 

  }

  if (po==0){

    /* Angular */
    siQ = sin(Q);
    coQ = cos(Q);
    siP = sin(P);
    coP = cos(P);

    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

      if (L0==0){
        AF[0][0] = 0.282094791773878;
      }
      else if (L0==1){
        dum = 0.48860251190292*siQ;
        AF[1][0] = dum*coP;
        AF[1][1] = dum*siP;
        AF[1][2] = 0.48860251190292*coQ;
      }
      else if (L0==2){
        dum1 = siQ*siQ;
        dum2 = 1.09254843059208*siQ*coQ;
        AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
        AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
        AF[2][2] = 1.09254843059208*dum1*siP*coP;
        AF[2][3] = dum2*coP;
        AF[2][4] = dum2*siP;
      }
      else if (L0==3){
        AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
        AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
        AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
        AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
        AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
        AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
        AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
      }
      else if (4<=L0){

        for(m=-L0; m<=L0; m++){ 
          ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
	}

        AF[L0][0] = -SH[L0][0];
 
        j = -1;
        for (i=1; i<(L0*2+1); i=i+4){
          j++;
          AF[L0][i] = -2.0/sqrt(2.0)*SH[L0-(2*j+1)][0];
	}

	j = 0;
	for (i=3; i<(L0*2+1); i=i+4){
	  j++;
          AF[L0][i] = -2.0/sqrt(2.0)*SH[L0-2*j][0];
	}

	j = -1;
	for (i=2; i<(L0*2+1); i=i+4){
	  j++;
          AF[L0][i] = -2.0/sqrt(2.0)*SH[L0-(2*j+1)][1];
	}

	j = 0;
	for (i=4; i<(L0*2+1); i=i+4){
	  j++;
          AF[L0][i] = -2.0/sqrt(2.0)*SH[L0-2*j][1];
	}
      }

    }
  }

  /* Contracted Chi */  
  al = -1;
  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
    for (Mul0=0; Mul0<Spe_Num_CBasis[wan][L0]; Mul0++){
      for (M0=0; M0<=2*L0; M0++){
        al++;

        tmp0 = 0.0;
        for (p=0; p<Spe_Specified_Num[wan][al]; p++){
          tmp0 = tmp0 + CntCoes[Mc_AN][al][p]*RF[L0][p];
        }        
        Chi[al] = tmp0*AF[L0][M0];
      }
    }
  }

  /****************************************************
     freeing of arrays:

   double RF[List_YOUSO[25]+1][List_YOUSO[24]];
   double AF[List_YOUSO[25]+1][2*(List_YOUSO[25]+1)+1];
  ****************************************************/

  for (i=0; i<(List_YOUSO[25]+1); i++){
    free(RF[i]);
  }
  free(RF);

  for (i=0; i<(List_YOUSO[25]+1); i++){
    free(AF[i]);
  }
  free(AF);

}
