/**********************************************************************
  tp.c:

  This program code calculates transition probablity between two states
  based on Fermi's golden rule.

  Log of tp.c:

     10/Jan./2018  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include "read_scfout.h"
#include "lapack_prototypes.h"
#include "f77func.h"

static void EigenBand_lapack(dcomplex **A, double *W, int N);
static dcomplex Lapack_LU_Zinverse(int n, dcomplex *A);

#define Host_ID       0         /* ID of the host CPU in MPI */

#define printout  0    /* 0:off, 1:on */
#define PI   3.1415926535897932384626



int main(int argc, char *argv[]) 
{
  int ct_AN,h_AN,Gh_AN,i,j,k,TNO1,TNO2;  
  int spin,Rn,n,ii,jj,ia,jb,l1,l2,l3;
  int Nocc1[2],Nocc2[2],Anum,Bnum;
  int *MP,po;
  double **C1,**C2;
  double *B,*ko,N1,N2;
  double **FermiF1,**FermiF2;
  double k1,k2,k3,k1b,k2b,k3b;
  double tmpr,tmpi,co,si,kRn,thocc;
  double px2,py2,pz2,p2,Utot1,Utot2;
  double os,osx,osy,osz;
  dcomplex ctmp,sum,sumx,sumy,sumz,**Z;
  dcomplex fsumx,fsumy,fsumz,det;
  dcomplex **Ax,**Ay,**Az;

  static double *a;
  static FILE *fp;
  FILE *fp1;

  double Ebond[30],Es,Ep;

  read_scfout(argv);

  /*
  printf("atomnum=%i\n", atomnum);
  printf("Catomnum=%i\n",Catomnum);
  printf("Latomnum=%i\n",Latomnum);
  printf("Ratomnum=%i\n",Ratomnum);
  */

  /*
  printf("\n\nOverlap matrix with momentum operator x\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      Rn = ncn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      printf("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n",
              ct_AN,h_AN,Gh_AN,Rn);
      for (i=0; i<TNO1; i++){
        for (j=0; j<TNO2; j++){
          printf("%10.7f ",OLPmo[0][ct_AN][h_AN][i][j]); 
        }
        printf("\n");
      }
    }
  }
  */

  /*
  printf("\n\nOverlap matrix\n");
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      Rn = ncn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      printf("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n",
              ct_AN,h_AN,Gh_AN,Rn);
      for (i=0; i<TNO1; i++){
        for (j=0; j<TNO2; j++){
          printf("%10.7f ",OLP[ct_AN][h_AN][i][j]); 
        }
        printf("\n");
      }
    }
  }
  */

  /* read the first coes file */

  MP = (int*)malloc(sizeof(int)*(atomnum+1));

  n = 0; 
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    MP[ct_AN] = n+1; 
    TNO1 = Total_NumOrbs[ct_AN];
    n += TNO1;
  }

  FermiF1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    FermiF1[spin] = (double*)malloc(sizeof(double)*(n+1));
  }

  C1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    C1[spin] = (double*)malloc(sizeof(double)*(n+1)*(n+1)*2);
    for (i=0; i<(n+1)*(n+1)*2; i++) C1[spin][i] = -1.0;
  }

  B = (double*)malloc(sizeof(double)*(n+1)*(n+1)*2);

  Z = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Z[spin] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1)*(n+1));
  }

  Ax = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Ax[spin] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1)*(n+1));
  }

  Ay = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Ay[spin] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1)*(n+1));
  }

  Az = (dcomplex**)malloc(sizeof(dcomplex*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    Az[spin] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1)*(n+1));
  }

  ko = (double*)malloc(sizeof(double)*(n+1));

  if ((fp1 = fopen(argv[2],"rb")) != NULL){

    for (spin=0; spin<=SpinP_switch; spin++){

      fread(&k1, sizeof(double), 1, fp1);
      fread(&k2, sizeof(double), 1, fp1);
      fread(&k3, sizeof(double), 1, fp1);

      fread(&Utot1, sizeof(double), 1, fp1);

      fread(FermiF1[spin], sizeof(double), (n+1), fp1);
      fread(C1[spin], sizeof(double), (n+1)*(n+1)*2, fp1);
    }
  }

  fclose(fp1);

  /*
  printf("FermiF1\n");
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=n; i++){
      printf("spin=%2d i=%2d FermiF1=%10.5f\n",spin,i,FermiF1[spin][i]);  
    }
  }
  */

  /*
  printf("C1\n");
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      printf("%10.5f ",C1[0][2*(n+1)*i+2*j]);  
    }
    printf("\n");
  }
  */

  /* read the second coes file */

  FermiF2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    FermiF2[spin] = (double*)malloc(sizeof(double)*(n+1));
  }

  C2 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){
    C2[spin] = (double*)malloc(sizeof(double)*(n+1)*(n+1)*2);
  }

  if ((fp1 = fopen(argv[3],"rb")) != NULL){
    for (spin=0; spin<=SpinP_switch; spin++){

      fread(&k1b, sizeof(double), 1, fp1);
      fread(&k2b, sizeof(double), 1, fp1);
      fread(&k3b, sizeof(double), 1, fp1);

      if (k1!=k1b){
        printf("k1 is inconsistent with each other.\n");
      }

      if (k2!=k2b){
        printf("k2 is inconsistent with each other.\n");
      }

      if (k3!=k3b){
        printf("k3 is inconsistent with each other.\n");
      }

      fread(&Utot2, sizeof(double), 1, fp1);

      fread(FermiF2[spin], sizeof(double), (n+1), fp1);
      fread(C2[spin], sizeof(double), (n+1)*(n+1)*2, fp1);
    }
  }

  fclose(fp1);

  /*
  printf("Utot1=%15.12f Utot2=%15.12f\n",Utot1,Utot2);
  exit(0);
  */

  /*
  printf("FermiF2\n");
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=n; i++){
      printf("spin=%2d i=%2d FermiF2=%10.5f\n",spin,i,FermiF2[spin][i]);  
    }
  }
  */

  /*
  printf("C2\n");
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      printf("%10.5f ",C2[0][2*(n+1)*i+2*j]);  
    }
    printf("\n");
  }
  */


#define C1_r(spin,i,j) C1[spin][ 2*(n+1)*(i)+2*(j) ]
#define C1_i(spin,i,j) C1[spin][ 2*(n+1)*(i)+2*(j)+1 ]
#define C2_r(spin,i,j) C2[spin][ 2*(n+1)*(i)+2*(j) ]
#define C2_i(spin,i,j) C2[spin][ 2*(n+1)*(i)+2*(j)+1 ]
#define B_r(i,j) B[ 2*(n)*(i-1)+2*(j-1)   ]
#define B_i(i,j) B[ 2*(n)*(i-1)+2*(j-1)+1 ]

  /* reordering the coefficients based on FermiF */

  po = 0;
  thocc = 0.5;
   
  do {

    for (spin=0; spin<=SpinP_switch; spin++){
      Nocc1[spin] = 0; 
      Nocc2[spin] = 0; 
      for (i=1; i<=n; i++){
	if (thocc<FermiF1[spin][i]) Nocc1[spin]++;
	if (thocc<FermiF2[spin][i]) Nocc2[spin]++;
      }
    }  

    if (SpinP_switch==0){
      N1 = 2.0*Nocc1[0];
      N2 = 2.0*Nocc2[0];
    }
    else{
      N1 = Nocc1[0] + Nocc1[1];
      N2 = Nocc2[0] + Nocc2[1];
    } 
    if ( (N1+0.3)<Valence_Electrons || (N2+0.3)<Valence_Electrons ){
      thocc = thocc*0.5;
    }
    else {
      po = 1;        
    }

  } while (po==0);

  /*
  for (spin=0; spin<=SpinP_switch; spin++){
    printf("ABC1 spin=%2d Nocc1=%2d Nocc2=%2d\n",spin,Nocc1[spin],Nocc2[spin]); 
  }

  exit(0);
  */

  /*
  printf("C1\n");
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      printf("%10.5f ",C1_r(0,i,j));
    }
    printf("\n");
  }
  */

  for (spin=0; spin<=SpinP_switch; spin++){

    /* C1 */

    k = 0;
    for (i=1; i<=n; i++){

      if (thocc<FermiF1[spin][i]){
        for (j=1; j<=n; j++){
          B[k] = C1_r(spin,i,j); k++;
          B[k] = C1_i(spin,i,j); k++;
        }
      }
    }

    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
        C1_r(spin,i,j) = B_r(i,j);
        C1_i(spin,i,j) = B_i(i,j);
      }
    }

    /* C2 */

    k = 0;
    for (i=1; i<=n; i++){

      if (thocc<FermiF2[spin][i]){
        for (j=1; j<=n; j++){
          B[k] = C2_r(spin,i,j); k++;
          B[k] = C2_i(spin,i,j); k++;
        }
      }
    }

    for (i=1; i<=n; i++){
      for (j=1; j<=n; j++){
        C2_r(spin,i,j) = B_r(i,j);
        C2_i(spin,i,j) = B_i(i,j);
      }
    }
  }  

  /*
  printf("reorderd C1\n");
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      printf("%10.5f ",C1[0][2*(n+1)*i+2*j]);  
    }
    printf("\n");
  }

  printf("reordered C2\n");
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      printf("%10.5f ",C2[0][2*(n+1)*i+2*j]);  
    }
    printf("\n");
  }
  */

  /* calculation of Z which is an overlap matrix two Slater determinants */

  printf(" Calculation of the Z matrix.\n");

  /*
  printf("C1\n");
  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      printf("%10.5f ",C1_r(0,i,j));
    }
    printf("\n");
  }
  */

  /*
  printf("Z\n");
  */

  for (spin=0; spin<=SpinP_switch; spin++){
    for (ii=1; ii<=Nocc1[spin]; ii++){
      for (jj=1; jj<=Nocc2[spin]; jj++){

	sum.r = 0.0; sum.i = 0.0;        
 
        for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

          TNO1 = Total_NumOrbs[ct_AN];
          Anum = MP[ct_AN];

          for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
	    Gh_AN = natn[ct_AN][h_AN];
	    Rn = ncn[ct_AN][h_AN];
	    TNO2 = Total_NumOrbs[Gh_AN];
            Bnum = MP[Gh_AN];

	    l1 = atv_ijk[Rn][1];
            l2 = atv_ijk[Rn][2];
            l3 = atv_ijk[Rn][3];
            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

            for (i=0; i<TNO1; i++){

              ia = Anum + i;

              for (j=0; j<TNO2; j++){

                jb = Bnum + j;

                tmpr = C1_r(spin,ii,ia)*C2_r(spin,jj,jb) + C1_i(spin,ii,ia)*C2_i(spin,jj,jb);
                tmpi = C1_r(spin,ii,ia)*C2_i(spin,jj,jb) - C1_i(spin,ii,ia)*C2_r(spin,jj,jb);

                sum.r += (tmpr*co - tmpi*si)*OLP[ct_AN][h_AN][i][j];
                sum.i += (tmpr*si + tmpi*co)*OLP[ct_AN][h_AN][i][j];
	      }
	    }

	  }
	}
  
        Z[spin][Nocc1[spin]*((jj)-1)+(ii)-1] = sum;
      
	/*
        printf("%10.5f ",sum.r);
	*/

      }
      /*
      printf("\n");
      */
    }
  }  

#define Z_ref(spin,i,j) Z[spin][Nocc1[spin]*((j)-1)+(i)-1]
#define Ax_ref(spin,i,j) Ax[spin][Nocc1[spin]*((j)-1)+(i)-1]
#define Ay_ref(spin,i,j) Ay[spin][Nocc1[spin]*((j)-1)+(i)-1]
#define Az_ref(spin,i,j) Az[spin][Nocc1[spin]*((j)-1)+(i)-1]

  /* calculation of the inverse Z */

  printf(" Calculation of the inverse Z matrix.\n");

  for (spin=0; spin<=SpinP_switch; spin++){

    /*
    printf("ABC1 det %15.12f %15.12f\n",det.r,det.i);   
    printf("Z spin=%2d Nocc1[spin]=%2d\n",spin,Nocc1[spin]);
    for (i=1; i<=Nocc1[spin]; i++){
      for (j=1; j<=Nocc1[spin]; j++){
        printf("Z i=%3d j=%3d Z=%10.5f %10.5f\n",
                  i,j,Z[spin][Nocc1[spin]*((j)-1)+(i)-1].r,Z[spin][Nocc1[spin]*((j)-1)+(i)-1].i); 
      }  
    } 
    */

    det = Lapack_LU_Zinverse(Nocc1[spin], Z[spin]);

    /*
    printf("det.r=%15.12f det.i=%15.12f\n",det.r,det.i);
    */

    for (i=1; i<=Nocc1[spin]; i++){
      for (j=1; j<=Nocc1[spin]; j++){

        ctmp = Z[spin][Nocc1[spin]*((j)-1)+(i)-1]; 
        Z[spin][Nocc1[spin]*((j)-1)+(i)-1].r = ctmp.r*det.r - ctmp.i*det.i; 
        Z[spin][Nocc1[spin]*((j)-1)+(i)-1].i = ctmp.r*det.i + ctmp.i*det.r; 
      }  
    } 

    /*
    printf("Inverse Z spin=%2d Nocc1[spin]=%2d\n",spin,Nocc1[spin]);
    for (i=1; i<=Nocc1[spin]; i++){
      for (j=1; j<=Nocc1[spin]; j++){
        printf("Z i=%3d j=%3d Z=%10.5f %10.5f\n",
                  i,j,Z[spin][Nocc1[spin]*((j)-1)+(i)-1].r,Z[spin][Nocc1[spin]*((j)-1)+(i)-1].i); 
      }  
    } 
    */

  }

  /* calculation of <phi_i|p|phi_j> */

  printf(" Calculation of <phi_i|p|phi_j>.\n");

  for (spin=0; spin<=SpinP_switch; spin++){
    for (ii=1; ii<=Nocc1[spin]; ii++){
      for (jj=1; jj<=Nocc2[spin]; jj++){

	sumx.r = 0.0; sumx.i = 0.0;        
	sumy.r = 0.0; sumy.i = 0.0;        
	sumz.r = 0.0; sumz.i = 0.0;        
 
        for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

          TNO1 = Total_NumOrbs[ct_AN];
          Anum = MP[ct_AN];

          for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
	    Gh_AN = natn[ct_AN][h_AN];
	    Rn = ncn[ct_AN][h_AN];
	    TNO2 = Total_NumOrbs[Gh_AN];
            Bnum = MP[Gh_AN];

	    l1 = atv_ijk[Rn][1];
            l2 = atv_ijk[Rn][2];
            l3 = atv_ijk[Rn][3];
            kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;
            si = sin(2.0*PI*kRn);
            co = cos(2.0*PI*kRn);

            for (i=0; i<TNO1; i++){

              ia = Anum + i;

              for (j=0; j<TNO2; j++){

                jb = Bnum + j;

                tmpr = C1_r(spin,ii,ia)*C2_r(spin,jj,jb) + C1_i(spin,ii,ia)*C2_i(spin,jj,jb);
                tmpi = C1_r(spin,ii,ia)*C2_i(spin,jj,jb) - C1_i(spin,ii,ia)*C2_r(spin,jj,jb);

                sumx.r += (tmpr*co - tmpi*si)*OLPmo[0][ct_AN][h_AN][i][j];
                sumx.i += (tmpr*si + tmpi*co)*OLPmo[0][ct_AN][h_AN][i][j];

                sumy.r += (tmpr*co - tmpi*si)*OLPmo[1][ct_AN][h_AN][i][j];
                sumy.i += (tmpr*si + tmpi*co)*OLPmo[1][ct_AN][h_AN][i][j];

                sumz.r += (tmpr*co - tmpi*si)*OLPmo[2][ct_AN][h_AN][i][j];
                sumz.i += (tmpr*si + tmpi*co)*OLPmo[2][ct_AN][h_AN][i][j];

	      }
	    }

	  }
	}
  
        Ax[spin][Nocc1[spin]*((jj)-1)+(ii)-1] = sumx;
        Ay[spin][Nocc1[spin]*((jj)-1)+(ii)-1] = sumy;
        Az[spin][Nocc1[spin]*((jj)-1)+(ii)-1] = sumz;
      
	/*
        printf("spin=%2d ii=%3d jj=%3d Ax.r=%15.12f Ax.i=%15.12f\n",spin,ii,jj,sumx.r,sumx.i);
	*/

      }
    }
  }  

  /*
  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=1; i<=Nocc1[spin]; i++){
      for (j=1; j<=Nocc1[spin]; j++){
        Z_ref(spin,i,j).r = 0.0;
        Z_ref(spin,i,j).i = 0.0;
      }
      Z_ref(spin,i,i).r = 1.0;
    }
  }

  det.r = 1.0;
  det.i = 0.0;
  */

  /* calculation of Ax*Z^{-1}, Ay*Z^{-1}, and Az*Z^{-1} */

  printf(" Calculation of A_{x,y,z}*Z^{-1}.\n");

  printf("\n Computational result\n");

  for (spin=0; spin<=SpinP_switch; spin++){

    fsumx.r = 0.0; fsumx.i = 0.0;        
    fsumy.r = 0.0; fsumy.i = 0.0;        
    fsumz.r = 0.0; fsumz.i = 0.0;        

    for (i=1; i<=Nocc1[spin]; i++){

      sumx.r = 0.0; sumx.i = 0.0;        
      sumy.r = 0.0; sumy.i = 0.0;        
      sumz.r = 0.0; sumz.i = 0.0;        

      for (k=1; k<=Nocc1[spin]; k++){

	sumx.r = Ax_ref(spin,i,k).r*Z_ref(spin,k,i).r - Ax_ref(spin,i,k).i*Z_ref(spin,k,i).i; 
	sumx.i = Ax_ref(spin,i,k).r*Z_ref(spin,k,i).i + Ax_ref(spin,i,k).i*Z_ref(spin,k,i).r; 

	sumy.r = Ay_ref(spin,i,k).r*Z_ref(spin,k,i).r - Ay_ref(spin,i,k).i*Z_ref(spin,k,i).i; 
	sumy.i = Ay_ref(spin,i,k).r*Z_ref(spin,k,i).i + Ay_ref(spin,i,k).i*Z_ref(spin,k,i).r; 

	sumz.r = Az_ref(spin,i,k).r*Z_ref(spin,k,i).r - Az_ref(spin,i,k).i*Z_ref(spin,k,i).i; 
	sumz.i = Az_ref(spin,i,k).r*Z_ref(spin,k,i).i + Az_ref(spin,i,k).i*Z_ref(spin,k,i).r; 

      }

      /*
      printf("WWW1 i=%4d sumx.r=%15.12f sumx.i=%15.12f\n",i,sumx.r,sumx.i);
      */

      fsumx.r += sumx.r; fsumx.i += sumx.i;
      fsumy.r += sumy.r; fsumy.i += sumy.i;
      fsumz.r += sumz.r; fsumz.i += sumz.i;

    }

    px2 = fsumx.r*fsumx.r + fsumx.i*fsumx.i;
    py2 = fsumy.r*fsumy.r + fsumy.i*fsumy.i;
    pz2 = fsumz.r*fsumz.r + fsumz.i*fsumz.i;

    p2 = px2 + py2 + pz2;
    
    os = p2/fabs(Utot1-Utot2);
    osx = px2/fabs(Utot1-Utot2);
    osy = py2/fabs(Utot1-Utot2);
    osz = pz2/fabs(Utot1-Utot2);

    printf(" spin=%2d oscillator strength=%15.10f\n",spin,os);
    printf(" x-comp.=%15.10f y-comp.=%15.10f z-comp.=%15.10f\n",osx,osy,osz);
  }

  printf(" Transition energy (eV) =%15.10f\n",fabs(Utot1-Utot2)*27.2113845);


  /* free of arrays */

  for (spin=0; spin<=SpinP_switch; spin++){
    free(C1[spin]);
  }
  free(C1);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(C2[spin]);
  }
  free(C2);

  free(B);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Z[spin]);
  }
  free(Z);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Ax[spin]);
  }
  free(Ax);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Ay[spin]);
  }
  free(Ay);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(Az[spin]);
  }
  free(Az);

  free(ko);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(FermiF1[spin]);
  }
  free(FermiF1);

  for (spin=0; spin<=SpinP_switch; spin++){
    free(FermiF2[spin]);
  }
  free(FermiF2);

}



dcomplex Lapack_LU_Zinverse(int n, dcomplex *A)
#define C_ref(i,j) C[n*(j)+i]
{
    static char *thisprogram="Lapack_LU_inverse";
    int *ipiv;
    dcomplex *work,tmp,det;
    int lwork;
    int info,i,j;

    /* L*U factorization */

    ipiv = (int*) malloc(sizeof(int)*n);

    F77_NAME(zgetrf,ZGETRF)(&n,&n,A,&n,ipiv,&info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    /* calculation of determinant */

    det.r = 1.0; det.i = 0.0;
    for (i=0; i<n; i++){
      tmp = det;
      det.r = tmp.r*A[n*(i)+(i)].r - tmp.i*A[n*(i)+(i)].i;
      det.i = tmp.r*A[n*(i)+(i)].i + tmp.i*A[n*(i)+(i)].r;
    }

    for (i=0; i<n; i++){
      if (ipiv[i] != i+1) { det.r = -det.r; det.i = -det.i; }
    }

    /* 
    printf("det %15.12f %15.12f\n",det.r,det.i);   

    for (i=0; i<n; i++){
      printf("i=%2d ipiv=%2d\n",i,ipiv[i]);
    }
    */

    /* inverse L*U factorization */

    lwork = 4*n;
    work = (dcomplex*)malloc(sizeof(dcomplex)*lwork);

    F77_NAME(zgetri,ZGETRI)(&n, A, &n, ipiv, work, &lwork, &info);

    if ( info !=0 ) {
      printf("zgetrf failed, info=%i, %s\n",info,thisprogram);
    }

    free(work); free(ipiv);

    return det;
}



void EigenBand_lapack(dcomplex **A, double *W, int N)
{
  static char *JOBZ="V";
  static char *UPLO="L";
  int LWORK;
  dcomplex *A0;
  dcomplex *WORK;
  double *RWORK;
  int INFO;
  int i,j;

  A0=(dcomplex*)malloc(sizeof(dcomplex)*N*N);

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A0[(j-1)*N+i-1] = A[i][j];
    }
  }

  LWORK=3*N; 
  WORK=(dcomplex*)malloc(sizeof(dcomplex)*LWORK);
  RWORK=(double*)malloc(sizeof(double)*(3*N-2));
  F77_NAME(zheev,ZHEEV)(JOBZ,UPLO, &N, A0, &N, W, WORK, &LWORK, RWORK, &INFO  );

  if (INFO!=0) {
    printf("************************************************************\n");
    printf("  EigenBand_lapack: cheev_()=%d\n",INFO);
    printf("************************************************************\n");
    exit(10);
  }

  for (i=1;i<=N;i++) {
    for (j=1;j<=N;j++) {
      A[i][j].r = A0[(j-1)*N+i-1].r;
      A[i][j].i = A0[(j-1)*N+i-1].i;
    }
  }

  for (i=N;i>=1;i--) {
    W[i] =W[i-1];
  }

  free(A0); free(RWORK); free(WORK); 
}
