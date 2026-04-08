/**********************************************************************

 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>

#include "read_scfout.h"
#include "Tools_BandCalc.h"
#include "lapack_prototypes.h"
#include "f77func.h"
#include "EigenValue_Problem.h"
#include "Eigen_HH.h"



void EigenValue_Problem(double k1,double k2,double k3, int calc_switch){

  int i,j,k;
  int GA_AN,LB_AN,GB_AN,tnoA,tnoB;
  int l,n,i1,j1,i2,j2;
  int l1,l2,l3,ia,ib,ja,jb;
  int Rn;
  int Anum,Bnum;
  int F_TNumOrbs,n_NC;
  double kRn;
  double co,si;
  double sum,ko1;
  double sum_r0,sum_i0;
  double sum_r1,sum_i1;
  double sum_r00,sum_i00;
  double sum_r01,sum_i01;
  double sum_r10,sum_i10;
  double sum_r11,sum_i11;
  dcomplex Ctmp1,Ctmp2;

  double RedumA,RedumB,RedumC, ImdumA,ImdumB,ImdumC,ImdumD;
  double Redum2A,Redum2B,Redum2C,Imdum2D;
  double d0,d1,d2,d3;
  double Re11,Re22,Re12,Im12,Nup[2],Ndw[2],theta[2],phi[2];

  int *MPF;
  double *ko;
  dcomplex **H,**S,**C;

  //allocate
  MPF = (int*)malloc(sizeof(int)*(atomnum+1));

  Anum = 1;
  for(i1=1; i1<=atomnum; i1++){
    MPF[i1] = Anum;
    Anum+= Total_NumOrbs[i1];
  }
  n = Anum - 1;
  F_TNumOrbs = Anum + 2;
  n_NC = 2*Anum + 2;

  ko = (double*)malloc(sizeof(double)*n_NC);

  H = (dcomplex**)malloc(sizeof(dcomplex*)*n_NC);
  for (j=0; j<n_NC; j++){
    H[j] = (dcomplex*)malloc(sizeof(dcomplex)*n_NC);
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*F_TNumOrbs);
  for (i=0; i<F_TNumOrbs; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*F_TNumOrbs);
  }

  C = (dcomplex**)malloc(sizeof(dcomplex*)*n_NC);
  for (j=0; j<n_NC; j++){
    C[j] = (dcomplex*)malloc(sizeof(dcomplex)*n_NC);
  }

  for (i1=1; i1<=2*n; i1++){
    for (j1=1; j1<=2*n; j1++){
      H[i1][j1].r = 0.0;
      H[i1][j1].i = 0.0;
      C[i1][j1].r = 0.0;
      C[i1][j1].i = 0.0;
    }
  }
  for (i1=1; i1<=n; i1++){
    for (j1=1; j1<=n; j1++){
      S[i1][j1].r = 0.0;
      S[i1][j1].i = 0.0;
    }
  }
  /****************************************************
    set H' and diagonalize it
   ****************************************************/

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MPF[GA_AN];
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];
      Bnum = MPF[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);

      for (i1=0; i1<tnoA; i1++){
	ia = Anum +i1;
	ib = Anum +i1 + n;
	for (j1=0; j1<tnoB; j1++){
	  ja = Bnum +j1;
	  jb = Bnum +j1 + n;

	  S[Anum+i1  ][Bnum+j1  ].r += co*OLP[GA_AN][LB_AN][i1][j1];
	  S[Anum+i1  ][Bnum+j1  ].i += si*OLP[GA_AN][LB_AN][i1][j1];

	  H[Anum+i1  ][Bnum+j1  ].r += co*Hks[0][GA_AN][LB_AN][i1][j1] - si*iHks[0][GA_AN][LB_AN][i1][j1];
	  H[Anum+i1  ][Bnum+j1  ].i += si*Hks[0][GA_AN][LB_AN][i1][j1] + co*iHks[0][GA_AN][LB_AN][i1][j1];

	  H[Anum+i1+n][Bnum+j1+n].r += co*Hks[1][GA_AN][LB_AN][i1][j1] - si*iHks[1][GA_AN][LB_AN][i1][j1];
	  H[Anum+i1+n][Bnum+j1+n].i += si*Hks[1][GA_AN][LB_AN][i1][j1] + co*iHks[1][GA_AN][LB_AN][i1][j1];

	  H[Anum+i1  ][Bnum+j1+n].r += co*Hks[2][GA_AN][LB_AN][i1][j1]
	    - si*(Hks[3][GA_AN][LB_AN][i1][j1]+iHks[2][GA_AN][LB_AN][i1][j1]);
	  H[Anum+i1  ][Bnum+j1+n].i += si*Hks[2][GA_AN][LB_AN][i1][j1]
	    + co*(Hks[3][GA_AN][LB_AN][i1][j1]+iHks[2][GA_AN][LB_AN][i1][j1]);
	}//j1
      }//i1
    }//LB_AN
  }//GA_AN

  for (i1=1; i1<=n; i1++){
    for (j1=1; j1<=n; j1++){
      H[j1+n][i1].r= H[i1][j1+n].r;
      H[j1+n][i1].i=-H[i1][j1+n].i;
    }
  }

  Eigen_HH(S,ko,n,n,1);

  /* calculate S*1/sqrt(ko) */
  for (l=1; l<=n; l++){
    if(ko[l]<0.0) ko1 = 1.0e-14;
    else ko1 = ko[l];
    ko[l] = 1.0/sqrt(ko1);
  }
  for (i1=1; i1<=n; i1++){
    for (j1=1; j1<=n; j1++){
      S[i1][j1].r = S[i1][j1].r*ko[j1];
      S[i1][j1].i = S[i1][j1].i*ko[j1];
    } 
  }

  /* U'^+ * H * U * M1 */

  /* transpose S */
  for (i1=1; i1<=n; i1++){
    for (j1=i1+1; j1<=n; j1++){
      Ctmp1 = S[i1][j1];
      Ctmp2 = S[j1][i1];
      S[i1][j1] = Ctmp2;
      S[j1][i1] = Ctmp1;
    }
  }

  for (j1=1; j1<=n; j1++){
    for (i1=1; i1<=2*n; i1++){

      sum_r0 = 0.0;
      sum_i0 = 0.0;
      sum_r1 = 0.0;
      sum_i1 = 0.0;

      for (l=1; l<=n; l++){
	sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
	sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;
	sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
	sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
      }
      C[2*j1-1][i1].r = sum_r0;
      C[2*j1-1][i1].i = sum_i0;
      C[2*j1  ][i1].r = sum_r1;
      C[2*j1  ][i1].i = sum_i1;
    }
  }     
  for (i1=1; i1<=n; i1++){
    for (j1=1; j1<=2*n; j1++){
      sum_r0 = 0.0;
      sum_i0 = 0.0;
      sum_r1 = 0.0;
      sum_i1 = 0.0;

      for (l=1; l<=n; l++){
	sum_r0 +=  S[i1][l].r*C[j1][l  ].r + S[i1][l].i*C[j1][l  ].i;
	sum_i0 +=  S[i1][l].r*C[j1][l  ].i - S[i1][l].i*C[j1][l  ].r;
	sum_r1 +=  S[i1][l].r*C[j1][l+n].r + S[i1][l].i*C[j1][l+n].i;
	sum_i1 +=  S[i1][l].r*C[j1][l+n].i - S[i1][l].i*C[j1][l+n].r;
      }
      H[2*i1-1][j1].r = sum_r0;
      H[2*i1-1][j1].i = sum_i0;
      H[2*i1  ][j1].r = sum_r1;
      H[2*i1  ][j1].i = sum_i1;
    }
  }

  /* H to C (transposition) */
  for (i1=1; i1<=2*n; i1++){
    for (j1=1; j1<=2*n; j1++){
      C[i1][j1].r = H[i1][j1].r;
      C[i1][j1].i = H[i1][j1].i;
    }
  }

  Eigen_HH(C,ko,2*n,2*n,1);
  for(l=1; l<2*n+1; l++){
    ko1 = ko[l] - ChemP;
    ko1*= eV2Hartree;
    EIGEN[l] =ko1;
  }//l


  if(calc_switch==1){

    /* transpose */
    for (i1=1; i1<=n; i1++){
      for (j1=i1+1; j1<=n; j1++){
	Ctmp1 = S[i1][j1];
	Ctmp2 = S[j1][i1];
	S[i1][j1] = Ctmp2;
	S[j1][i1] = Ctmp1;
      }
    }
    for (i1=1; i1<=2*n; i1++){
      for (j1=i1+1; j1<=2*n; j1++){
	Ctmp1 = C[i1][j1];
	Ctmp2 = C[j1][i1];
	C[i1][j1] = Ctmp2;
	C[j1][i1] = Ctmp1;
      }
    }
    for (i1=1; i1<=n; i1++){
      for (j1=1; j1<=2*n; j1++){
	sum_r0 = 0.0;    sum_i0 = 0.0;
	sum_r1 = 0.0;    sum_i1 = 0.0;

	l1 = 0;
	for (l=1; l<=n; l++){
	  l1++;
	  sum_r0 +=  S[i1][l].r*C[j1][l1].r - S[i1][l].i*C[j1][l1].i;
	  sum_i0 +=  S[i1][l].r*C[j1][l1].i + S[i1][l].i*C[j1][l1].r;
	  l1++;
	  sum_r1 +=  S[i1][l].r*C[j1][l1].r - S[i1][l].i*C[j1][l1].i;
	  sum_i1 +=  S[i1][l].r*C[j1][l1].i + S[i1][l].i*C[j1][l1].r;
	}
	H[i1  ][j1].r = sum_r0;
	H[i1  ][j1].i = sum_i0;
	H[i1+n][j1].r = sum_r1;
	H[i1+n][j1].i = sum_i1;
      }
    }

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      tnoA = Total_NumOrbs[GA_AN];
      for (i1=0; i1<=tnoA; i1++){

	/* Added by N. Yamaguchi ***/
#ifndef DEBUG_SIGMAEK_OLDER_20181126
	if (mode==(int(*)())FermiLoop){
	  for (j1=0; j1<4; j1++){
	    Data_MulP[j1][0][GA_AN][i1] = 0.0;
	  }
	} else {
#endif
	  /* ***/

	  for (l=0; l<l_cal; l++){
	    for (j1=0; j1<4; j1++){
	      Data_MulP[j1][l][GA_AN][i1] = 0.0;
	    }//j1
	  }//l

	  /* Added by N. Yamaguchi ***/
#ifndef DEBUG_SIGMAEK_OLDER_20181126
	}
#endif
	/* ***/

      }//i1
    }//GA_AN

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      tnoA = Total_NumOrbs[GA_AN];
      Anum = MPF[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	tnoB = Total_NumOrbs[GB_AN];

	l1 = atv_ijk[Rn][1];  l2 = atv_ijk[Rn][2];  l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
	si = sin(2.0*PI*kRn);  co = cos(2.0*PI*kRn);

	Bnum = MPF[GB_AN];

	for (i1=0; i1<tnoA; i1++){
	  ia = Anum +i1;       ib = Anum +i1 + n;

	  for (j1=0; j1<tnoB; j1++){
	    ja = Bnum +j1;       jb = Bnum +j1 + n;

	    //for (l=0; l<2*n+1; l++){
	    for(l=l_min; l<=l_max; l++){
	      // Re11
	      RedumA = H[ia][l].r*H[ja][l].r + H[ia][l].i*H[ja][l].i;
	      ImdumA = H[ia][l].r*H[ja][l].i - H[ia][l].i*H[ja][l].r;
	      Redum2A = co*RedumA - si*ImdumA;
	      //  
	      //            CDM0[k] += co*RedumA + si*ImdumA;
	      //            iDM0[k] += si*RedumA + co*ImdumA;
	      //
	      // Re22
	      RedumB = H[ib][l].r*H[jb][l].r + H[ib][l].i*H[jb][l].i;
	      ImdumB = H[ib][l].r*H[jb][l].i - H[ib][l].i*H[jb][l].r;
	      Redum2B = co*RedumB - si*ImdumB;
	      // Re12
	      RedumC = H[ia][l].r*H[jb][l].r + H[ia][l].i*H[jb][l].i;
	      ImdumC = H[ia][l].r*H[jb][l].i - H[ia][l].i*H[jb][l].r;
	      Redum2C = co*RedumC - si*ImdumC;
	      // Im12
	      Imdum2D = co*ImdumC + si*RedumC;

	      /* Disabled by N. Yamaguchi
	       * if (l>=l_min && l<=l_max){
	       * Data_MulP[0][l][GA_AN][i1] += Redum2A *OLP[GA_AN][LB_AN][i1][j1]; //MulP
	       * Data_MulP[1][l][GA_AN][i1] += Redum2B *OLP[GA_AN][LB_AN][i1][j1]; //MulP
	       * Data_MulP[2][l][GA_AN][i1] += Redum2C *OLP[GA_AN][LB_AN][i1][j1]; //MulP
	       * Data_MulP[3][l][GA_AN][i1] -= Imdum2D *OLP[GA_AN][LB_AN][i1][j1]; //MulP
	       * }//if
	       */
	      /* Added by N. Yamaguchi ***/
#ifndef DEBUG_SIGMAEK_OLDER_20181126
	      if (mode==(int(*)())FermiLoop){
		if (l==Nband[0]){
		  Data_MulP[0][0][GA_AN][i1]+=Redum2A*OLP[GA_AN][LB_AN][i1][j1];
		  Data_MulP[1][0][GA_AN][i1]+=Redum2B*OLP[GA_AN][LB_AN][i1][j1];
		  Data_MulP[2][0][GA_AN][i1]+=Redum2C*OLP[GA_AN][LB_AN][i1][j1];
		  Data_MulP[3][0][GA_AN][i1]-=Imdum2D*OLP[GA_AN][LB_AN][i1][j1];
		}
	      } else {
#endif
		Data_MulP[0][l-l_min][GA_AN][i1]+=Redum2A*OLP[GA_AN][LB_AN][i1][j1];
		Data_MulP[1][l-l_min][GA_AN][i1]+=Redum2B*OLP[GA_AN][LB_AN][i1][j1];
		Data_MulP[2][l-l_min][GA_AN][i1]+=Redum2C*OLP[GA_AN][LB_AN][i1][j1];
		Data_MulP[3][l-l_min][GA_AN][i1]-=Imdum2D*OLP[GA_AN][LB_AN][i1][j1];
#ifndef DEBUG_SIGMAEK_OLDER_20181126
	      }
#endif
	      /* ***/

	    }//l
	  }//j1(tnoB)
	  }//i1(tnoA)
	}//LB_AN(FNAN)
      }//GA_AN

    }//calc_switch
    /**********************************
      free array
     **********************************/
    free(MPF);

    free(ko);
    for (i=0; i<F_TNumOrbs; i++){
      free(S[i]);
    }free(S);
    for (j=0; j<n_NC; j++){
      free(H[j]);
    }free(H);
    for (j=0; j<n_NC; j++){
      free(C[j]);
    }free(C);

    //  return 0;
  }



