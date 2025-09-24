/**********************************************************************
  TRAN_Calc_SurfGreen_direct.c:

  TRAN_Calc_SurfGreen_direct.c is a subroutine to calculate the  
  surface Green's function.

  Log of TRAN_Calc_SurfGreen_direct.o

     11/Dec/2005   Released by H.Kino

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


static void TRAN_Calc_SurfGreen_Multiple_Inverse(
                                /* input */
				dcomplex w,
				int n, 
				double *h00, 
				double *h01,
				double *s00,
				double *s01,
                                int iteration_max,
                                double eps,
	 			dcomplex *gr /* output */
				);


static void TRAN_Calc_SurfGreen_transfer(
				  /* input */
				  dcomplex w,
				  int n, 
				  dcomplex *h00, 
				  dcomplex *h01,
				  dcomplex *s00,
				  dcomplex *s01,
				  int iteration_max,
				  double eps,
				  dcomplex *G00 /* output */
				  );



/*
 * calculate surface green function
 *
 *    G00(w) = (w-e_i^s)^(-1) | i-> infinity 
 *
*/

void TRAN_Calc_SurfGreen_direct(
                                /* input */
				dcomplex w,
				int n, 
				dcomplex *h00, 
				dcomplex *h01,
				dcomplex *s00,
				dcomplex *s01,
                                int iteration_max,
                                double eps,
                                /* output */
	 			dcomplex *gr 
				)
{

  TRAN_Calc_SurfGreen_Normal(w, n, h00, h01, s00, s01, iteration_max, eps, gr);

  /*
  TRAN_Calc_SurfGreen_transfer(w, n, h00, h01, s00, s01, iteration_max, eps, gr);
  */

  /*
  TRAN_Calc_SurfGreen_Multiple_Inverse(w, n, h00, h01, s00, s01, iteration_max, eps, gr);
  */

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












void TRAN_Calc_SurfGreen_Multiple_Inverse(
                                /* input */
				dcomplex w,
				int n, 
				double *h00, 
				double *h01,
				double *s00,
				double *s01,
                                int iteration_max,
                                double eps,
	 			dcomplex *gr /* output */
				)
#define h00_ref(i,j) h00[ n*((j)-1)+(i)-1 ]
#define h01_ref(i,j) h01[ n*((j)-1)+(i)-1 ]
#define s00_ref(i,j) s00[ n*((j)-1)+(i)-1 ]
#define s01_ref(i,j) s01[ n*((j)-1)+(i)-1 ]

#define gr_ref(i,j) gr[ n*((j)-1)+(i)-1 ]
#define g0_ref(i,j) g0[ n*((j)-1)+(i)-1 ]
#define h0_ref(i,j) h0[ n*((j)-1)+(i)-1 ]
#define hl_ref(i,j) hl[ n*((j)-1)+(i)-1 ]
#define hr_ref(i,j) hr[ n*((j)-1)+(i)-1 ]
#define tmpv1_ref(i,j) tmpv1[ n*((j)-1)+(i)-1 ]
#define tmpv2_ref(i,j) tmpv2[ n*((j)-1)+(i)-1 ]
{
  int i,j,iter;
  dcomplex a,b;
  double rms2,val;
  dcomplex cval;

  dcomplex *g0;
  dcomplex *h0,*hl,*hr;
  dcomplex *tmpv1,*tmpv2;

  /*  printf("w=%le %le, n=%d, ite_max=%d eps=%le\n",w.r, w.i, n, iteration_max, eps); */
 
  a.r=1.0; a.i=0.0;
  b.r=0.0; b.i=0.0;

  g0 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  h0 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  hl = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  hr = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;

  tmpv1 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;
  tmpv2 = (dcomplex*)malloc(sizeof(dcomplex)*n*n) ;

  /* h0 = ws00-h00 */

  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      h0_ref(i,j).r = w.r*s00_ref(i,j) - h00_ref(i,j);
      h0_ref(i,j).i = w.i*s00_ref(i,j);
    }
  }

  /* hl = ws01-h01 */

  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      hl_ref(i,j).r = w.r*s01_ref(i,j) - h01_ref(i,j);
      hl_ref(i,j).i = w.i*s01_ref(i,j);
    }
  }

  /* hr = hl^t */

  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      hr_ref(i,j).r = hl_ref(j,i).r;
      hr_ref(i,j).i = hl_ref(j,i).i;
    }
  }

  /* initial g0 = h0 */

  for (i=1;i<=n;i++) {    
    for (j=1;j<=n;j++) {
      g0_ref(i,j).r = h0_ref(i,j).r;
      g0_ref(i,j).i = h0_ref(i,j).i;
    }
  }

  /* initial g0 -> g0^-1  */

  Lapack_LU_Zinverse(n,g0);



  /* solve iteratively the closed form */
  
  for( iter=1; iter<iteration_max; iter++) {

    /* hl*g0 -> tmpv1 */

    F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, hl, &n, g0,  &n,  &b, tmpv1, &n);

    /* tmpv1*hr (=hl*g0*hr) -> tmpv2 */

    F77_NAME(zgemm,ZGEMM)("N","N", &n, &n, &n, &a, tmpv1, &n, hr,  &n,  &b, tmpv2, &n);

    /* tmpv2 = h0 - tmpv2 (= h0-hl*g0*hr) */

    for (i=1; i<=n; i++) {    
      for (j=1; j<=n; j++) {
        tmpv2_ref(i,j).r = h0_ref(i,j).r - tmpv2_ref(i,j).r;
        tmpv2_ref(i,j).i = h0_ref(i,j).i - tmpv2_ref(i,j).i;
      }
    }

    /* tmpv2 -> tmpv2^-1 */

    Lapack_LU_Zinverse(n,tmpv2);

    /* calculate rms */

    rms2=0.0;
    for (i=1; i<=n; i++) {
      for (j=1; j<=n; j++) {
	cval.r = tmpv2_ref(i,j).r - g0_ref(i,j).r; 
	cval.i = tmpv2_ref(i,j).i - g0_ref(i,j).i;
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


    printf("TRAN_Calc_SurfGreen: iter=%d itermax=%d, rms2=%15.12f, eps=%15.12f\n",
            iter, iteration_max, rms2, eps);



    /* tmpv2 -> g0 */

    for (i=1; i<=n; i++) {
      for (j=1; j<=n; j++) {
        g0_ref(i,j).r = tmpv2_ref(i,j).r;
        g0_ref(i,j).i = tmpv2_ref(i,j).i;
      }
    }

    if ( rms2 < eps ) {
      /* converged */
      goto last;
    }

  } /* iteration */


 last:
  if (iter>=iteration_max) {
    /*
    printf("ERROR: TRAN_Calc_SurfGreen: iter=%d itermax=%d, rms2=%le, eps=%le\n",
            iter, iteration_max, rms2, eps);
    */
  }

  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      gr_ref(i,j).r = g0_ref(i,j).r;
      gr_ref(i,j).i = g0_ref(i,j).i;
    }
  }

  free(g0);
  free(h0);
  free(hl);
  free(hr);
  free(tmpv1);
  free(tmpv2);
}















/*
 * calculate surface green function
 *
 *    G00(w) = (w S00 - H00 - (H01-w S01)^-1 * T )^-1   ---(53) 
 *
 *
 *    t_0 = (w S00-H00)^-1 (H01-w S01)^+
 *    bar_t_0 =  (w S00-H00)^-1 (H01-w S01)
 *    T_0 = t_0
 *    bar_T_0 = bar_t_0
 *
 * * loop
 *
 *    t_i = (1-t_(i-1) bar_t_(i-1) - bar_t_(i-1) t_(i-1) )^-1 (t_(i-1))^2
 *    bar_t_i = (1-t_(i-1) bar_t_(i-1) - bar_t_(i-1) t_(i-1) )^-1 (bar_t_(i-1))^2
 *
 *    bar_T_i = bar_T_i bar_t_i 
 *    T_i = T_(i-1) + bar_T_(i-1) t_i  
 *
 * * loop_end
 *
 *    G00 = (w S00-H00-H01 T_i)^-1
 *    
 *
*/
void TRAN_Calc_SurfGreen_transfer(
				  /* input */
				  dcomplex w,
				  int n, 
				  dcomplex *h00, 
				  dcomplex *h01,
				  dcomplex *s00,
				  dcomplex *s01,
				  int iteration_max,
				  double eps,
				  dcomplex *G00 /* output */
				  )
#define h00_ref(i,j) h00[ n*((j)-1)+(i)-1 ]
#define h01_ref(i,j) h01[ n*((j)-1)+(i)-1 ]
#define s00_ref(i,j) s00[ n*((j)-1)+(i)-1 ]
#define s01_ref(i,j) s01[ n*((j)-1)+(i)-1 ]


#define gr00_ref(i,j) gr00[ n*((j)-1)+(i)-1 ]
#define H10_ref(i,j) H10[ n*((j)-1)+(i)-1 ]
#define H01_ref(i,j) H01[ n*((j)-1)+(i)-1 ]

#define G00_ref(i,j) G00[ n*((j)-1)+(i)-1 ]
#define G00_old_ref(i,j) G00_old[ n*((j)-1)+(i)-1 ]


#define t_i_ref(i,j) t_i[ n*((j)-1)+(i)-1 ]
#define bar_t_i_ref(i,j) bar_t_i[ n*((j)-1)+(i)-1 ]
#define T_i_ref(i,j) T_i[ n*((j)-1)+(i)-1 ]
#define T_i_old_ref(i,j) T_i_old[ n*((j)-1)+(i)-1 ]

#define bar_T_i_ref(i,j) bar_T_i[ n*((j)-1)+(i)-1 ]

#define tt1_ref(i,j) tt1[ n*((j)-1)+(i)-1 ]
#define tt2_ref(i,j) tt2[ n*((j)-1)+(i)-1 ]
#define tt3_ref(i,j) tt3[ n*((j)-1)+(i)-1 ]

{
  int i,j,iter;
  dcomplex a,b,cval;
  double rms2,val;

  dcomplex *gr00, *H10, *H01; 
  dcomplex *t_i,     *bar_t_i,     *bar_T_i,      *T_i; 
  dcomplex *t_i_old, *bar_t_i_old, *bar_T_i_old, *T_i_old;
  dcomplex  *tt1, *tt2;

  int n2,one;

  n2 = n*n;
  one=1;

  /*  printf("w=%le %le, n=%d, ite_max=%d eps=%le\n",w.r, w.i, n, iteration_max, eps); */

  /* parameters for BLAS */ 
  a.r=1.0; a.i=0.0;
  b.r=0.0; b.i=0.0;

  t_i = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  bar_t_i = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  T_i = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  bar_T_i = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  t_i_old = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  bar_t_i_old = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  T_i_old = (dcomplex*)malloc(sizeof(dcomplex)*n2) ; 
  bar_T_i_old = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  tt1 = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  tt2 = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;

  gr00 = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  H10 = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;
  H01 = (dcomplex*)malloc(sizeof(dcomplex)*n2) ;


  /*  gr02 = w*s00-h00 */
  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      gr00_ref(i,j).r = w.r*s00_ref(i,j).r - w.i*s00_ref(i,j).i - h00_ref(i,j).r;
      gr00_ref(i,j).i = w.i*s00_ref(i,j).r + w.r*s00_ref(i,j).i - h00_ref(i,j).i;
    }
  }

  /* gr00^-1 */
  Lapack_LU_Zinverse(n,gr00);

  /* H01 = -w * s01 + h01 */
  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      H01_ref(i,j).r = -w.r*s01_ref(i,j).r + w.i*s01_ref(i,j).i + h01_ref(i,j).r;
      H01_ref(i,j).i = -w.i*s01_ref(i,j).r - w.r*s01_ref(i,j).i + h01_ref(i,j).i;
    }
  }

  /* for (32) */
  /*  H10 = -w*s10 + h10 */
  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      H10_ref(i,j).r = H01_ref(j,i).r;
      H10_ref(i,j).i = H01_ref(j,i).i;
    }
  }

  /* t_0 = gr00*H10 */

  F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,gr00,&n,H10, &n,&b, t_i,&n);

  /* bar_t_0 = gr00*H01 */
  F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,gr00,&n,H01, &n,&b, bar_t_i,&n);


  F77_NAME(zcopy,ZCOPY)(&n2,  t_i,&one, t_i_old,&one);
  F77_NAME(zcopy,ZCOPY)(&n2, bar_t_i,&one,bar_t_i_old,&one);

  F77_NAME(zcopy,ZCOPY)(&n2,  t_i,&one, T_i_old,&one);  /* T_i  = (50) */
  F77_NAME(zcopy,ZCOPY)(&n2,  t_i,&one, T_i,&one);
  /* bar_T_(i) = bar_t_0 bar_t_1 ... bar_t_(i) */
  F77_NAME(zcopy,ZCOPY)(&n2,  bar_t_i,&one, bar_T_i_old,&one);  




  for (iter=1;iter<iteration_max; iter++) {

    /* t_i_old * bar_t_i_old */
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,t_i_old,&n,bar_t_i_old, &n,&b, tt1,&n);
    /* bar_t_i_old * t_i_old */
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,bar_t_i_old,&n,t_i_old, &n,&b, tt2,&n);

    /*  I - t_i-1 bar_t_i-1  -  bar_t_i-1 t_i-1 */
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	tt1_ref(i,j).r = -tt1_ref(i,j).r - tt2_ref(i,j).r;
	tt1_ref(i,j).i = -tt1_ref(i,j).i - tt2_ref(i,j).i;
      }
    }
    for (i=1;i<=n;i++) {
      j=i;
      tt1_ref(i,j).r += 1.0;
    }

    /*  tt1 = ( I - t_i-1 bar_t_i-1  -  bar_t_i-1 t_i-1 )^-1 */
    Lapack_LU_Zinverse(n,tt1); 

    /* tt2 = t_i-1 t_i-1 */
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,t_i_old,&n,t_i_old, &n,&b, tt2,&n);
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,tt1,&n,tt2, &n,&b, t_i,&n);
    /* update t_i  (40) */

    /* for (41) */
    /* tt2 = bar_t_i-1 bar_t_i-1 */
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,bar_t_i_old,&n,bar_t_i_old, &n,&b, tt2,&n);
    /* bar_t_i = tt1 * tt2 */
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,tt1,&n,tt2, &n,&b, bar_t_i,&n);
    /* update bar_t_i  (41) */

    /* update bar_T_i = bar_t_0 bar_t_1 bar_t_2 bar_t_3 ... bar_t_(i-i) */
    /* bar_T_i = bar_T_(i-1) * bar_t_i */
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,bar_T_i_old,&n,bar_t_i, &n,&b, bar_T_i,&n);

    /* T_i = t0+ bt0 t1 + bt0 bt1 t2 + ... */
    /* T_i = T_(i-1) + bar_T_(i-1) t_i */
    /* F77_NAME(zcopy,ZCOPY)(&n2,T_i_old,&one, T_i, &one);*/ /* needless */
    b.r=1.0; b.i=0.0;
    F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,bar_T_i_old,&n,t_i, &n,&b, T_i,&n);
    b.r=0.0; b.i=0.0;
    /* updated T_i,   (50) */


    /* RMS = max [ T_i - T_(i-1) ] */
    rms2 = 0.0;
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
	cval.r = T_i_ref(i,j).r- T_i_old_ref(i,j).r;
	cval.i = T_i_ref(i,j).i- T_i_old_ref(i,j).i;
	val = cval.r*cval.r+ cval.i*cval.i;
	rms2 =  (rms2> val)? rms2: val;
      }
    }
    /* printf("iter=%d rms2=%lf\n",iter,rms2); */
    rms2 =sqrt(rms2);
	  
    if ( rms2 < eps ) {
      goto last;
    }
                   
    /* loop again */

    F77_NAME(zcopy,ZCOPY)(&n2,   t_i,&one,  t_i_old,&one);
    F77_NAME(zcopy,ZCOPY)(&n2,  bar_t_i,&one, bar_t_i_old,&one);
    F77_NAME(zcopy,ZCOPY)(&n2,  T_i,&one, T_i_old,&one);
    F77_NAME(zcopy,ZCOPY)(&n2,  bar_T_i,&one, bar_T_i_old,&one);

  }


 last:
 /*  printf("iter=%d rms=%lf\n",iter,rms2);   */

  if (iter>=iteration_max) {
    printf("ERROR: TRAN_Calc_SurfGreen_trans: iter=%d itermax=%d, rms=%le, eps=%le\n",
            iter, iteration_max, rms2, eps);
  }


  /* (53) */
  F77_NAME(zgemm,ZGEMM)("N","N",&n,&n,&n,&a,H01,&n,T_i, &n,&b, G00,&n);

  /* (w S00 -H00 -H01 T_i) */
  for (i=1;i<=n;i++) {
    for (j=1;j<=n;j++) {
      G00_ref(i,j).r = w.r*s00_ref(i,j).r - w.i*s00_ref(i,j).i - h00_ref(i,j).r - G00_ref(i,j).r;
      G00_ref(i,j).i = w.i*s00_ref(i,j).r + w.r*s00_ref(i,j).i - h00_ref(i,j).i - G00_ref(i,j).i;
    }
  }

  Lapack_LU_Zinverse(n,G00); /* (53) */

  free(H01);
  free(H10);
  free(gr00);
  free(tt2);
  free(tt1);
  free(T_i_old);
  free(bar_T_i_old);
  free(bar_t_i_old);
  free(t_i_old);
  free(bar_T_i);
  free(T_i);
  free(bar_t_i);
  free(t_i);

}
