/**********************************************************************
  TRAN_Calc_SurfGreen_Sanvito.c:

  TRAN_Calc_SurfGreen_Sanvito.c is a subroutine to calculate 
  surface Green's function based on the method first developed 
  by S. Sanvito and then extended by J. Taylor.

  Please refer to the below two references:
                  S. Sanvito et al. Phys. Rev. B 59, 11936 (1999)
                  J. Taylor et al. Phys. Rev. B 63, 245407 (2001)

  Log of TRAN_Calc_SurfGreen_Sanvito.c

     4/Jan/2013   Released by Y. Xiao

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "tran_prototypes.h"
#include "lapack_prototypes.h"

static void TRAN_Calc_SurfGreen_Normal(
                                        /* input */
                                dcomplex w,
                                int n,
                                dcomplex *h00,
                                dcomplex *h01,
                                dcomplex *s00,
                                dcomplex *s01,
                                int iteration_max,
                                double eps,
                                dcomplex *gr /* output */
                                );


static void TRAN_Calc_SurfGreen_Sanvito12(
                               /* input */
                                int side,
                                dcomplex w,
                                int n,
                                dcomplex *h00,
                                dcomplex *h01,
                                dcomplex *s00,
                                dcomplex *s01,
                                dcomplex *gr  /* output */
                                );

void TRAN_Calc_SurfGreen_Sanvito(
                               /* input */
                                int side,
                                dcomplex w,
                                int n,
                                dcomplex *h00,
                                dcomplex *h01,
                                dcomplex *s00,
                                dcomplex *s01,
                                int iteration_max,
                                double eps,
                                dcomplex *gr  /* output */
                                )
{

   if ( w.i > 1.0e+8 ) {
     /*printf("Recursive method is used when the 0th-order moment of Green's function is calculated\n");*/

    TRAN_Calc_SurfGreen_Normal(w, n, h00, h01, s00, s01, iteration_max, eps, gr);

   } else {
     /*printf("Sanvito's method is used for all cases except the 0th-order moment of Green's function\n");*/

    TRAN_Calc_SurfGreen_Sanvito12(side, w, n, h00, h01, s00, s01, gr);

   }
}



void TRAN_Calc_SurfGreen_Normal(
                               /* input */
				dcomplex w,
				int n, 
				dcomplex *h00, 
				dcomplex *h01,
				dcomplex *s00,
				dcomplex *s01,
                                int iteration_max,
                                double eps,
	 			dcomplex *gr /* output */
				)

#define h00_ref(i,j) h00[ n*((j)-1)+(i)-1 ]
#define h01_ref(i,j) h01[ n*((j)-1)+(i)-1 ]
#define s00_ref(i,j) s00[ n*((j)-1)+(i)-1 ]
#define s01_ref(i,j) s01[ n*((j)-1)+(i)-1 ]

#define es0_ref(i,j) es0[ n*((j)-1)+(i)-1 ]
#define e00_ref(i,j) e00[ n*((j)-1)+(i)-1 ]
#define alp_ref(i,j) alp[ n*((j)-1)+(i)-1 ]
#define bet_ref(i,j) bet[ n*((j)-1)+(i)-1 ]

#define gr_ref(i,j) gr[ n*((j)-1)+(i)-1 ]

#define gr00_ref(i,j) gr00[ n*((j)-1)+(i)-1 ]
#define gr01_ref(i,j) gr01[ n*((j)-1)+(i)-1 ]
#define gr02_ref(i,j) gr02[ n*((j)-1)+(i)-1 ]
#define gt_ref(i,j) gt[ n*((j)-1)+(i)-1 ]  

{
  int i,j,iter;
  dcomplex a,b;
  double rms2,val;
  dcomplex cval;

  dcomplex *es0, *e00, *alp, *bet ;
  dcomplex *gr00, *gr02, *gr01;
  dcomplex *gt;

  /*  printf("w=%le %le, n=%d, ite_max=%d eps=%le\n",w.r, w.i, n, iteration_max, eps); */
 
  a.r=1.0; a.i=0.0;
  b.r=0.0; b.i=0.0;

  es0 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  e00 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  alp = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  bet = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  gr00 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  gr01 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  gr02 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  gt = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;

  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      es0_ref(i,j).r = w.r*s00_ref(i,j).r - w.i*s00_ref(i,j).i - h00_ref(i,j).r;
      es0_ref(i,j).i = w.i*s00_ref(i,j).r + w.r*s00_ref(i,j).i - h00_ref(i,j).i;
    }
  }
  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      e00_ref(i,j).r = w.r*s00_ref(i,j).r - w.i*s00_ref(i,j).i - h00_ref(i,j).r;
      e00_ref(i,j).i = w.i*s00_ref(i,j).r + w.r*s00_ref(i,j).i - h00_ref(i,j).i;
    }
  }
  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      alp_ref(i,j).r = -w.r*s01_ref(i,j).r + w.i*s01_ref(i,j).i + h01_ref(i,j).r;
      alp_ref(i,j).i = -w.i*s01_ref(i,j).r - w.r*s01_ref(i,j).i + h01_ref(i,j).i;
    }
  }
  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      /* taking account of the complex conjugate of H and S */
      bet_ref(i,j).r = -w.r*s01_ref(j,i).r - w.i*s01_ref(j,i).i + h01_ref(j,i).r;
      bet_ref(i,j).i = -w.i*s01_ref(j,i).r + w.r*s01_ref(j,i).i - h01_ref(j,i).i;
    }
  }

  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      gr00_ref(i,j).r = es0_ref(i,j).r;
      gr00_ref(i,j).i = es0_ref(i,j).i;
    }
  }

  Lapack_LU_Zinverse(n,gr00);

  /* save gr00 to calculate rms */
  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      gt_ref(i,j).r = gr00_ref(i,j).r;
      gt_ref(i,j).i = gr00_ref(i,j).i;
    }
  }

  
  for( iter=1; iter<iteration_max; iter++) {

    for (i=1;i<=n;i++) {   
      for (j=1;j<=n;j++) {
	gr02_ref(i,j).r = e00_ref(i,j).r;
        gr02_ref(i,j).i = e00_ref(i,j).i;
      }
    }

    Lapack_LU_Zinverse(n,gr02);
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, gr02,&n,bet,&n,&b, gr01,&n);
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, alp,&n,gr01,&n,&b, gr00,&n);

    for (i=1;i<=n;i++) {  
      for (j=1;j<=n;j++) {
	es0_ref(i,j).r = es0_ref(i,j).r - gr00_ref(i,j).r;
	es0_ref(i,j).i = es0_ref(i,j).i - gr00_ref(i,j).i;
      }
    }

     
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, gr02,&n,alp,&n,&b, gr01,&n);
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, gr02,&n,bet,&n,&b, gr00,&n);
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, bet,&n,gr01,&n,&b, gr02,&n);

    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	e00_ref(i,j).r=e00_ref(i,j).r-gr02_ref(i,j).r;
	e00_ref(i,j).i=e00_ref(i,j).i-gr02_ref(i,j).i;
      }
    }
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, alp,&n,gr00,&n,&b, gr02,&n);

    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	e00_ref(i,j).r = e00_ref(i,j).r - gr02_ref(i,j).r;
	e00_ref(i,j).i = e00_ref(i,j).i - gr02_ref(i,j).i;
      }
    }
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, alp,&n,gr01,&n,&b, gr02,&n);

    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	alp_ref(i,j).r=gr02_ref(i,j).r;
	alp_ref(i,j).i=gr02_ref(i,j).i;
      }
    }
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a, bet,&n,gr00,&n,&b, gr02,&n);
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	bet_ref(i,j).r = gr02_ref(i,j).r;
	bet_ref(i,j).i = gr02_ref(i,j).i;
      }
    }

    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	gr00_ref(i,j).r = es0_ref(i,j).r;
	gr00_ref(i,j).i = es0_ref(i,j).i;
      }
    }

    Lapack_LU_Zinverse(n,gr00);

    /* calculate rms */

    rms2=0.0;
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	cval.r = gt_ref(i,j).r - gr00_ref(i,j).r; 
	cval.i = gt_ref(i,j).i - gr00_ref(i,j).i;
	val = cval.r*cval.r + cval.i*cval.i;
	if ( rms2 <  val ) { rms2 = val ; }
      }
    }
    rms2 = sqrt(rms2);

    /*debug*/ 

    /*
    printf("TRAN_Calc_SurfGreen: iter=%d itermax=%d, rms2=%le, eps=%le\n",
    iter, iteration_max, rms2, eps);
    */

    /*
    printf("TRAN_Calc_SurfGreen: iter=%d itermax=%d, rms2=%15.12f, eps=%15.12f\n",
    iter, iteration_max, rms2, eps);
    */

    /*debug end*/
    if ( rms2 < eps ) {
      /* converged */
      goto last;
    }
    else {
      for (i=1;i<=n;i++) {
	for (j=1;j<=n;j++) {
	  gt_ref(i,j).r = gr00_ref(i,j).r;
	  gt_ref(i,j).i = gr00_ref(i,j).i;
	}
      }
    }

  } /* iteration */


 last:
  if (iter>=iteration_max) {
    printf("ERROR: TRAN_Calc_SurfGreen: iter=%d itermax=%d, rms2=%le, eps=%le\n",
            iter, iteration_max, rms2, eps);
  }

  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      gr_ref(i,j).r = gr00_ref(i,j).r;
      gr_ref(i,j).i = gr00_ref(i,j).i;
    }
  }
  
  free(gt);
  free(gr02);
  free(gr01);
  free(gr00);
  free(bet);
  free(alp);
  free(e00);
  free(es0);
}









void TRAN_Calc_SurfGreen_Sanvito12(
                               /* input */
                                int side,
                                dcomplex w,
                                int n,
                                dcomplex *h00,
                                dcomplex *h01,
                                dcomplex *s00,
                                dcomplex *s01,
                                dcomplex *gr  /* output */
                                )

#define h00_ref(i,j) h00[ n*((j)-1)+(i)-1 ]
#define h01_ref(i,j) h01[ n*((j)-1)+(i)-1 ]
#define s00_ref(i,j) s00[ n*((j)-1)+(i)-1 ]
#define s01_ref(i,j) s01[ n*((j)-1)+(i)-1 ]

#define K00_ref(i,j) K00[ n*((j)-1)+(i)-1 ]
#define K01_ref(i,j) K01[ n*((j)-1)+(i)-1 ]
#define K10_ref(i,j) K10[ n*((j)-1)+(i)-1 ]

#define t1_ref(i,j) t1[ n*((j)-1)+(i)-1 ]
#define block1_ref(i,j) block1[ n*((j)-1)+(i)-1 ]
#define block2_ref(i,j) block2[ n*((j)-1)+(i)-1 ]
#define unit_ref(i,j) unit[ n*((j)-1)+(i)-1 ]

#define EA00_ref(i,j) EA[ ((j)-1)*n*2+(i)-1 ] 
#define EA01_ref(i,j) EA[ ((j+n)-1)*n*2+(i)-1 ]
#define EA10_ref(i,j) EA[ ((j)-1)*n*2+(i+n)-1 ]
#define EA11_ref(i,j) EA[ ((j+n)-1)*n*2+(i+n)-1 ]

#define EB00_ref(i,j) EB[ ((j)-1)*n*2+(i)-1 ] 
#define EB01_ref(i,j) EB[ ((j+n)-1)*n*2+(i)-1 ]
#define EB10_ref(i,j) EB[ ((j)-1)*n*2+(i+n)-1 ]
#define EB11_ref(i,j) EB[ ((j+n)-1)*n*2+(i+n)-1 ]

{
  FILE *fp;
  double pi;
  double kr,ki,wr,wi;
  double tempx,tempy;
  double eps=1.0e-8;
  dcomplex a,b;
  int i,j,m,move[n*2];

  dcomplex *K00, *K01, *K10, *t1, *t2, *t3, *unit, *block1, *block2, *EA, *EB;
  dcomplex *EA0, *EB0, *E0, *EA1, *EA2, *EB1, *EB2;

  dcomplex *test1, *test2, *s1, *s2, *s3; 

  dcomplex *V1, *V2, *V1t, *V2t;
  dcomplex *V_inv;

  /*  ZGEEV Lapack arrays */
  char BWORK[n];
  int LWORK, info, ILO, IHI, *IWORK;
  dcomplex *ALPHA, *BETA, *VL, *VR, *WORKE;
  double *LSCALE, *RSCALE, ABNRM, BBNRM, *RCONDE, *RCONDV, *RWORK;

  K00 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  K01 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  K10 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  t1 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  t2 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  t3 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  unit = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  block1 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  block2 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  EA = (dcomplex*)malloc(sizeof(dcomplex)*n*n*2*2) ;
  EB = (dcomplex*)malloc(sizeof(dcomplex)*n*n*2*2) ;
  E0 = (dcomplex*)malloc(sizeof(dcomplex)*2*n) ;
  EA0 = (dcomplex*)malloc(sizeof(dcomplex)*n*2) ;
  EB0 = (dcomplex*)malloc(sizeof(dcomplex)*n*2) ;

  EA1 = (dcomplex*)malloc(sizeof(dcomplex)*n) ;
  EB1 = (dcomplex*)malloc(sizeof(dcomplex)*n) ;
  EA2 = (dcomplex*)malloc(sizeof(dcomplex)*n) ;
  EB2 = (dcomplex*)malloc(sizeof(dcomplex)*n) ;

  WORKE = (dcomplex*)malloc(sizeof(dcomplex)*(2*2*n)) ;  /* 4*n */
  VL = (dcomplex*)malloc(sizeof(dcomplex)*n*n*2*2) ;
  VR = (dcomplex*)malloc(sizeof(dcomplex)*n*n*2*2) ;
  ALPHA = (dcomplex*)malloc(sizeof(dcomplex)*2*n) ; 
  BETA = (dcomplex*)malloc(sizeof(dcomplex)*2*n) ;

  RWORK = (double*)malloc(sizeof(double)*6*n*2) ;
  LSCALE = (double*)malloc(sizeof(double)*2*n) ;
  RSCALE = (double*)malloc(sizeof(double)*2*n) ;
  RCONDE = (double*)malloc(sizeof(double)*2*n) ;
  RCONDV = (double*)malloc(sizeof(double)*2*n) ;

  IWORK = (int*)malloc(sizeof(int)*(2*n+2)) ;


  test1 = (dcomplex*)malloc(sizeof(dcomplex)*n) ;
  test2 = (dcomplex*)malloc(sizeof(dcomplex)*n) ;
  s1    = (dcomplex*)malloc(sizeof(dcomplex)*n) ;
  s2    = (dcomplex*)malloc(sizeof(dcomplex)*1) ;
  s3    = (dcomplex*)malloc(sizeof(dcomplex)*1) ;

  V1 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  V2 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  V1t = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  V2t = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;

  V_inv = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;

  pi=3.1415926535;
  b.r=0.0; b.i=0.0;

  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      K00_ref(i,j).r = h00_ref(i,j).r - (  w.r*s00_ref(i,j).r - w.i*s00_ref(i,j).i );
      K00_ref(i,j).i = h00_ref(i,j).i - (  w.i*s00_ref(i,j).r + w.r*s00_ref(i,j).i );
    }
  }

 if ( side == 0 ) {

  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      K10_ref(i,j).r = h01_ref(i,j).r - (  w.r*s01_ref(i,j).r - w.i*s01_ref(i,j).i );
      K10_ref(i,j).i = h01_ref(i,j).i - (  w.i*s01_ref(i,j).r + w.r*s01_ref(i,j).i );
    }
  }
  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      K01_ref(i,j).r =   h01_ref(j,i).r - (  w.r*s01_ref(j,i).r + w.i*s01_ref(j,i).i );  
      K01_ref(i,j).i =  -h01_ref(j,i).i - (  w.i*s01_ref(j,i).r - w.r*s01_ref(j,i).i );  
    }
  }

 } else {

  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      K01_ref(i,j).r = h01_ref(i,j).r - (  w.r*s01_ref(i,j).r - w.i*s01_ref(i,j).i );
      K01_ref(i,j).i = h01_ref(i,j).i - (  w.i*s01_ref(i,j).r + w.r*s01_ref(i,j).i );
    }
  }
  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      K10_ref(i,j).r =   h01_ref(j,i).r - (  w.r*s01_ref(j,i).r + w.i*s01_ref(j,i).i );
      K10_ref(i,j).i =  -h01_ref(j,i).i - (  w.i*s01_ref(j,i).r - w.r*s01_ref(j,i).i );
    }
  }
 } /*if ( side == 0 )*/

  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      t1[(j-1)*n+i-1].r = K01[(j-1)*n+i-1].r; 
      t1[(j-1)*n+i-1].i = K01[(j-1)*n+i-1].i;
      unit_ref(i,j).r = 0.0; unit_ref(i,j).i = 0.0;
      if(i==j) { unit_ref(i,j).r = 1.0; }
    }
  }


  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      EA00_ref(i,j).r =  -K00_ref(i,j).r;
      EA00_ref(i,j).i =  -K00_ref(i,j).i;

      EA01_ref(i,j).r =  -K10_ref(i,j).r;
      EA01_ref(i,j).i =  -K10_ref(i,j).i;

      EA10_ref(i,j).r =  unit_ref(i,j).r;
      EA10_ref(i,j).i =  unit_ref(i,j).i;

      EA11_ref(i,j).r =  0.0;
      EA11_ref(i,j).i =  0.0;

      EB00_ref(i,j).r =   K01_ref(i,j).r;
      EB00_ref(i,j).i =   K01_ref(i,j).i;

      EB01_ref(i,j).r =   0.0;
      EB01_ref(i,j).i =   0.0;

      EB10_ref(i,j).r =  0.0;
      EB10_ref(i,j).i =  0.0;

      EB11_ref(i,j).r =  unit_ref(i,j).r;
      EB11_ref(i,j).i =  unit_ref(i,j).i;

    }
  }

  m=2*n; LWORK=2*m; /*2*m;*/
  F77_NAME(zggevx,ZGGEVX)( "N", "V", "V", "N", &m, EA, &m, EB, &m,
                           ALPHA, BETA, VL, &m, VR, &m, &ILO, &IHI,
                           LSCALE, RSCALE, &ABNRM, &BBNRM, RCONDE, RCONDV,
                           WORKE, &LWORK, RWORK, IWORK, &BWORK, &info );

/* calculate the eigenvalue from alpha/beta */ 
{
  double ar,ai,br,bi;
  double at,bt;
  for (i=0; i<m; i++)  {
     ar = ALPHA[i].r;   ai = ALPHA[i].i; 
     br = BETA[i].r;    bi = BETA[i].i;

     if( sqrt(ar*ar+ai*ai) < 1.0e-10 ) {
      E0[i].r = 0.0; E0[i].i =  10.0; 
      EA0[i].r = 0.0;    EA0[i].i = 0.0; 
      EB0[i].r = 1.0e+8; EB0[i].i = 0.0;
     } 
     else if( sqrt(br*br+bi*bi) < 1.0e-10 ) {
      E0[i].r = 0.0; E0[i].i = -10.0;  
      EA0[i].r = 1.0e+8; EA0[i].i = 0.0;
      EB0[i].r = 0.0;    EB0[i].i = 0.0;     
     }
     else {

     wr = (br*ar+bi*ai)/(ar*ar+ai*ai);
     wi = (bi*ar-br*ai)/(ar*ar+ai*ai);
     EB0[i].r = wr;    EB0[i].i = wi;

     wr = (ar*br+ai*bi)/(br*br+bi*bi);
     wi = (ai*br-ar*bi)/(br*br+bi*bi);
     EA0[i].r = wr; EA0[i].i = wi;
     
     tempx = wr/sqrt(wr*wr+wi*wi);
     tempy = wi/sqrt(wr*wr+wi*wi);
     if ( fabs(tempx)<1.0e-10 ) {
       if (tempy > 0.0) { kr = pi/2.0; } else { kr = pi*3.0/2.0; }
     } 
     else if ( fabs(tempy)<1.0e-10 ) {
       if (tempx > 0.0) { kr = 0.0;    } else { kr = pi;         }
     }
     else {
       if ( tempx>0.0 && tempy>0.0) {kr = asin(tempy); }
       if ( tempx>0.0 && tempy<0.0) {kr = asin(tempy)+pi*2.0; }
       if ( tempx<0.0 && tempy>0.0) {kr = acos(tempx); }
       if ( tempx<0.0 && tempy<0.0) {kr = acos(-tempx)+pi; }
     }
     ki = -0.5 * log(wr*wr+wi*wi);
     E0[i].r = kr;
     E0[i].i = ki;
     } /*if( sqrt(ar*ar+ai*ai) < 1.0e-10 )*/
   }
} /* end of eigenvalues */


  /* calculate group velocity and determine left and right going modes*/
{
  int mode, n1, ns1, ns2;
  double t2r,t2i,r2r,r2i,t3r,t3i,r3r,r3i,temp2r,temp2i,temp3r,temp3i,tempr,tempi;

  for ( i=0; i<m; i++ ) {
    if ( E0[i].i > -eps && E0[i].i < eps ) { mode = 0; move[i] = 1000; }
    else if ( E0[i].i > eps )  { mode = -999; move[i] = 1; }
    else if ( E0[i].i < -eps ) { mode = -999; move[i] = -1; }
  
    if ( mode == 0 ) {
 
     for ( j=0; j<n; j++) {test1[j].r = VR[i*m+j].r; test1[j].i = VR[i*m+j].i; test2[j].r = test1[j].r; test2[j].i = -test1[j].i;}
     a.r = 1.0; a.i = 0.0; n1=1;
     F77_NAME(zgemm,ZGEMM)("N","N", &n1, &n, &n, &a, test2, &n1, K01, &n, &b, s1, &n1);
     F77_NAME(zgemm,ZGEMM)("N","N", &n1, &n1, &n, &a, s1, &n1, test1, &n, &b, s2, &n1);
     
     F77_NAME(zgemm,ZGEMM)("N","N", &n1, &n, &n, &a, test2, &n1, K10, &n, &b, s1, &n1);
     F77_NAME(zgemm,ZGEMM)("N","N", &n1, &n1, &n, &a, s1, &n1, test1, &n, &b, s3, &n1);         
     
     kr=E0[i].r; ki=E0[i].i;
     t2r = exp(-ki)*cos(kr); t2i =  exp(-ki)*sin(kr);
     r2r = s2[0].r;          r2i = s2[0].i;
     t3r = exp( ki)*cos(kr); t3i = -exp( ki)*sin(kr);
     r3r = s3[0].r;          r3i = s3[0].i;

     temp2i = t2r*r2i + t2i*r2r; temp2r = t2r*r2r - t2i*r2i;
     temp3i = t3r*r3i + t3i*r3r; temp3r = t3r*r3r - t3i*r3i;

     tempr = -(temp2i - temp3i);
     tempi =  (temp2r - temp3r);
     
     if ( tempr > 0.0 ) { move[i] = 1; } else { move[i] = -1; }
     /*printf("%12.8f %12.8f %12.8f %12.8f %d\n",kr,ki,tempr,tempi,move[i]);*/
    } /* if ( mode == 0) */
     /*printf("%16.11f %16.11f %d\n",E0[i].r,E0[i].i,move[i]); */
  } /*  for ( i=0; i<m; i++ ) */

     /* check if the determination of left and right going modes is correct */
   ns1=0; ns2=0;
   for (i=0; i<m; i++) {
      if ( move[i] ==  1) ns1++;
      if ( move[i] == -1) ns2++;
   }
   if ( ns1 != n || ns2 != n || ns1+ns2 != m) { printf("Determination of left and right going modes are problematic...\n");  }  

   /*printf("%d %d \n", ns1, ns2);*/

       /* determination of left and right going modes */
   ns1=0; ns2=0;
  for ( i=0; i<m; i++) {
    if ( move[i] == 1 ) { 
      
       EA1[ns1].r = EA0[i].r; EA1[ns1].i = EA0[i].i;
       EB1[ns1].r = EB0[i].r; EB1[ns1].i = EB0[i].i;

       for ( j=0; j<n; j++) { 
          V1[ns1*n+j].r   = VR[i*m+j].r;   V1[ns1*n+j].i  = VR[i*m+j].i;
          V1t[ns1*n+j].r  = VR[i*m+j].r;   V1t[ns1*n+j].i = VR[i*m+j].i;
       }
       ns1=ns1+1;
    } else if ( move[i] == -1 ) {
       /*E2[ns2].r=E0[i].r;E2[ns2].i=E0[i].i;*/

       EA2[ns2].r = EA0[i].r; EA2[ns2].i = EA0[i].i;
       EB2[ns2].r = EB0[i].r; EB2[ns2].i = EB0[i].i;

       for ( j=0; j<n; j++) { 
          V2[ns2*n+j].r  = VR[i*m+j].r; V2[ns2*n+j].i  = VR[i*m+j].i;
          V2t[ns2*n+j].r = VR[i*m+j].r; V2t[ns2*n+j].i = VR[i*m+j].i;
       }
       ns2=ns2+1;
    } 
  } /* for ( i=0; i<m; i++) */


} /* end of calculation of group velocity */

  /* calculate duals of each mode */
  Lapack_LU_Zinverse(n,V1t);
  Lapack_LU_Zinverse(n,V2t);


  /* calculate the inverse of V */
{

 double kr1,kr2,ki1,ki2;
 int n1;

   for ( j=0; j<n*n; j++) {
     t1[j].r = 0.0;     t1[j].i = 0.0;
     t2[j].r = 0.0;     t2[j].i = 0.0;
   }

  a.r = 1.0; a.i = 0.0; n1=1;

  for ( i=0; i<n; i++) {

     for ( j=0; j<n; j++) {test1[j].r = V1[i*n+j].r; test1[j].i = V1[i*n+j].i; test2[j].r = V1t[j*n+i].r; test2[j].i = V1t[j*n+i].i;}
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n1, &a, test1, &n, test2, &n1, &b, block1, &n);
     kr1 = EA1[i].r; ki1 = EA1[i].i;

     for ( j=0; j<n; j++) {test1[j].r = V2[i*n+j].r; test1[j].i = V2[i*n+j].i; test2[j].r = V2t[j*n+i].r; test2[j].i = V2t[j*n+i].i;}
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n1, &a, test1, &n, test2, &n1, &b, block2, &n);
     kr2 = EB2[i].r; ki2 = EB2[i].i;

     for ( j=0; j<n*n; j++) {
       t1[j].r += ( block1[j].r*kr1-block1[j].i*ki1 ) ;
       t1[j].i += ( block1[j].i*kr1+block1[j].r*ki1 ) ;
       t2[j].r += ( block2[j].r*kr2-block2[j].i*ki2 ) ;
       t2[j].i += ( block2[j].i*kr2+block2[j].r*ki2 ) ;
     }

  } /* for ( i=0; i<n; i++) */

     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K10, &n, t2, &n, &b, block2, &n);
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K01, &n, t1, &n, &b, block1, &n);

    for ( j=0; j<n*n; j++) {
      V_inv[j].r = block2[j].r + K00[j].r + block1[j].r;
      V_inv[j].i = block2[j].i + K00[j].i + block1[j].i;
    }

     Lapack_LU_Zinverse(n,V_inv);

} /* end of calculation of inverse of V */

  /* calculate surface Green's function */
{
 double kr1,kr2,ki1,ki2;
 int n1;

  a.r = 1.0; a.i = 0.0; n1=1;

   for ( j=0; j<n*n; j++) {
     t1[j].r = 0.0;     t1[j].i = 0.0;
     t2[j].r = 0.0;     t2[j].i = 0.0;
     t3[j].r = 0.0;     t3[j].i = 0.0;
   }

  if ( side == 0 ) { 

   for ( i=0; i<n; i++) {
       /* left going states with BAR */
     for ( j=0; j<n; j++) {test1[j].r = V2[i*n+j].r; test1[j].i = V2[i*n+j].i; test2[j].r = V2t[j*n+i].r; test2[j].i = V2t[j*n+i].i;}
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n1, &a, test1, &n, test2, &n1, &b, block2, &n);
     kr2 = EB2[i].r; ki2 = EB2[i].i;

       /* right going states without BAR */
     for ( j=0; j<n; j++) {test1[j].r = V1[i*n+j].r; test1[j].i = V1[i*n+j].i; test2[j].r = V1t[j*n+i].r; test2[j].i = V1t[j*n+i].i;}
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n1, &a, test1, &n, test2, &n1, &b, block1, &n);
     kr1 = EA1[i].r; ki1 = EA1[i].i;

     for ( j=0; j<n*n; j++) {
        
       t1[j].r += ( block1[j].r*kr1-block1[j].i*ki1 ) ;
       t1[j].i += ( block1[j].i*kr1+block1[j].r*ki1 ) ;

       t2[j].r += ( block2[j].r*kr2-block2[j].i*ki2 ) ;
       t2[j].i += ( block2[j].i*kr2+block2[j].r*ki2 ) ;

       t3[j].r += ( block2[j].r ) ;
       t3[j].i += ( block2[j].i ) ;
       }
    } /* for ( i=0; i<n; i++) */

      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, t1, &n, V_inv, &n, &b, block1, &n);
      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K01, &n, block1, &n, &b, t1, &n);

      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K10, &n, t2, &n, &b, block1, &n);
      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K00, &n, t3, &n, &b, block2, &n);

     for ( j=0; j<n*n; j++) {
       t2[j].r = block1[j].r + block2[j].r ;
       t2[j].i = block1[j].i + block2[j].i ;
     }

    } else { /*if ( side == 1 )*/
     
   for ( i=0; i<n; i++) {
       /* left going states with BAR */
     for ( j=0; j<n; j++) {test1[j].r = V2[i*n+j].r; test1[j].i = V2[i*n+j].i; test2[j].r = V2t[j*n+i].r; test2[j].i = V2t[j*n+i].i;}
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n1, &a, test1, &n, test2, &n1, &b, block2, &n); 
     kr2 = EA1[i].r; ki2 = EA1[i].i;

       /* right going states without BAR */
     for ( j=0; j<n; j++) {test1[j].r = V1[i*n+j].r; test1[j].i = V1[i*n+j].i; test2[j].r = V1t[j*n+i].r; test2[j].i = V1t[j*n+i].i;}
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n1, &a, test1, &n, test2, &n1, &b, block1, &n);
     kr1 = EB2[i].r; ki1 = EB2[i].i;

     for ( j=0; j<n*n; j++) {

       t1[j].r += ( block2[j].r*kr1-block2[j].i*ki1 ) ;
       t1[j].i += ( block2[j].i*kr1+block2[j].r*ki1 ) ;

       t2[j].r += ( block1[j].r*kr2-block1[j].i*ki2 ) ;
       t2[j].i += ( block1[j].i*kr2+block1[j].r*ki2 ) ;

       t3[j].r += ( block1[j].r ) ;
       t3[j].i += ( block1[j].i ) ;
       }
    } /* for ( i=0; i<n; i++) */

      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, t1, &n, V_inv, &n, &b, block1, &n);
      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K10, &n, block1, &n, &b, t1, &n);

      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K01, &n, t2, &n, &b, block1, &n);
      F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, K00, &n, t3, &n, &b, block2, &n);

     for ( j=0; j<n*n; j++) {
       t2[j].r = block1[j].r + block2[j].r ;
       t2[j].i = block1[j].i + block2[j].i ;
     }

    } /* if ( side ==0 ) */

     Lapack_LU_Zinverse(n,t2);

     /* obtain delta<l-infinity>*/
     F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, t2, &n, t1, &n, &b, t3, &n);

   for ( j=0; j<n*n; j++) {
     gr[j].r = - ( V_inv[j].r + t3[j].r );
     gr[j].i = - ( V_inv[j].i + t3[j].i );
   }


} /* end of calculation of surface Green's function */

  /* freeing arrays */

  free(K00);
  free(K01);
  free(K10);
  free(unit);
  free(t1);
  free(t2);
  free(t3);
  free(block1);
  free(block2);
  free(EA);
  free(EB);
  free(E0);
  free(test1);
  free(test2);
  free(s1);
  free(s2);
  free(s3);
  free(V1);
  free(V2);
  free(V1t);
  free(V2t);
  free(EA0);
  free(EB0);
  free(EA1);
  free(EB1);
  free(EA2);
  free(EB2);
  free(V_inv);
  free(WORKE);
  free(VL);
  free(VR);
  free(ALPHA);
  free(BETA);
  free(RWORK);
  free(LSCALE);
  free(RSCALE);
  free(RCONDE);
  free(RCONDV);
  free(IWORK);

} /* end of routine */


