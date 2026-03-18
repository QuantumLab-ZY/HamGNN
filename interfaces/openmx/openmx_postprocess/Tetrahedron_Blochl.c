#include <stdio.h>

/*****************************************************************
   Improved analytical tetrahedron method 
   Idea: PRB 49, 16223 (1994), by Bloechl, Jepsen and Andersen 
*******************************************************************/

/*
  input e[0:3],n
  output: e[0]<e[1]<e[2]<e[4]
*/
void OrderE0(double *e, int n)
{

 int i,j;
 double t;

 for (i=0;i<n-1;i++) {
   for (j=i;j<n;j++) {
      if (e[j]<e[i]) {
          t = e[j];
          e[j]=e[i];
          e[i]=t;

      }
   }
 }
}



/*
  input e[0:3],a[0:3],n
  output: e[0]<e[1]<e[2]<e[4] 
          a[] is also interchanged corresponding to e[]:
*/
void OrderE(double *e,double *a, int n)
{

 int i,j;
 double t;

 for (i=0;i<n-1;i++) {
   for (j=i;j<n;j++) {
      if (e[j]<e[i]) {
          t = e[j];
          e[j]=e[i];
          e[i]=t;

          t = a[j];
          a[j]=a[i];
          a[i]=t;

      }
   }
 }
}


/*
 density of states 
 Appendix C of the paper

   assume that et[] is orderd 

   input et[0:3], *e
   output dos
*/

void ATM_Dos(double *et, double *e, double *dos)
{
    double e21, e31, e32, e41, e42, e43;
    double e1,e2,e4;

    /* Parameter adjustments */
    --et;
#if 0
    printf("ATM: %lf %lf %lf %lf %lf\n",*e,et[1],et[2],et[3],et[4]);
#endif

    e21 = et[2] - et[1];
    e31 = et[3] - et[1];
    e32 = et[3] - et[2];
    e41 = et[4] - et[1];
    e42 = et[4] - et[2];
    e43 = et[4] - et[3];
    if (*e < et[1]) {
        *dos = 0.;
    } else if (*e > et[1] && *e < et[2]) {
        e1=(*e - et[1]);
        *dos =  3.0 * e1*e1 / (e21 * e31 * e41);
    } else if (*e > et[2] && *e < et[3]) {
        e2=(*e - et[2]);
        *dos = (e21 * 3. + e2 * 6. - (e31 + e42) * 3. * e2 * e2 
                  / (e32 * e42)) / (e31 * e41);
    } else if (*e > et[3] && *e < et[4]) {
        e4= (et[4] - *e);
        *dos =  3.0* e4*e4 / (e41 * e42 * e43);
    } else if (*e > et[4]) {
        *dos = 0.;
    }
#if 0
    printf("ATM_Dos: %lf %lf %lf %lf %lf->%lf\n",*e, et[1],et[2],et[3],et[4],*dos);
#endif

}


/* 
 Appendix B of the paper
 In the Appendix B, integrated weight is written, differenciate it. 
  assume that et[] is ordered. 

  input et[0:3], at[0:3],*e
  output spectrum

  <at> = sum_i at[i] w[i]
  w[i] is a function of et[i]
  An integrated w[i]  is written in the paper. 

*/
void ATM_Spectrum(double *et,double *at, double *e, double *spectrum)
{
     double a21, a31, a32, a41;
     double e21, e31, e32, e41, e42, e43;
     double a42, a43;
     double dos;

    /* Parameter adjustments , ---  f2c technique */
    --at;

    ATM_Dos(et, e, &dos);
    --et; 
#if 0
    printf("DOS-> %lf %lf\n",*e,dos);
    printf("%lf %lf %lf %lf\n",et[1],et[2],et[3],et[4]);
#endif
    e21 = et[2] - et[1];
    e31 = et[3] - et[1];
    e32 = et[3] - et[2];
    e41 = et[4] - et[1];
    e42 = et[4] - et[2];
    e43 = et[4] - et[3];
    a21 = at[2] - at[1];
    a31 = at[3] - at[1];
    a32 = at[3] - at[2];
    a41 = at[4] - at[1];
    a42 = at[4] - at[2];
    a43 = at[4] - at[3];
    if (*e < et[1]) {
        *spectrum = 0.;
    } else if (*e > et[1] && *e < et[2]) {
        *spectrum = dos * (at[1] + (*e - et[1]) * .33333333333333331 * (a21 / 
                e21 + a31 / e31 + a41 / e41));
    } else if (*e > et[2] && *e < et[3]) {
        *spectrum = dos * (at[1] + (a21 + e21 * a31 / e31 + e21 * a41 / e41) *
                 .33333333333333331 * (et[3] - *e) / e32 + (at[4] - at[1] - (
                e43 * a41 / e41 + e43 * a42 / e42 + a43) * .33333333333333331)
                 * (*e - et[2]) / e32);
    } else if (*e > et[3] && *e < et[4]) {
        *spectrum = dos * (at[4] + (*e - et[4]) * .33333333333333331 * (a41 / 
                e41 + a42 / e42 + a43 / e43));
    } else if (*e > et[4]) {
        *spectrum = 0.;
    }

}

