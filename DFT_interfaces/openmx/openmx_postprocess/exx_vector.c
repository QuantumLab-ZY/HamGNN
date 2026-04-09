/*----------------------------------------------------------------------
  exx_vector.c

  Some simple geometrical calculations.

  Coded by M. Toyoda, 07/JAN/2010
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "exx.h"
#include "exx_vector.h"


double EXX_Vector_Distance(const double v[3], const double w[3])
{
  double dx, dy, dz;
  dx = v[0] - w[0];
  dy = v[1] - w[1];
  dz = v[2] - w[2];
  return sqrt(dx*dx + dy*dy + dz*dz);
}


/*----------------------------------------------------------------------
  EXX_Vector_F2C

  Fractional coordinate to Cartesian coordinate.
----------------------------------------------------------------------*/
void EXX_Vector_F2C(
  double v_c[3],       /* (OUT) v in cartesian coord */
  const double v[3],   /* (IN) a vector in fractional coord */
  const double pvec[9] /* (IN) primitive translational vectors */
)
{
  v_c[0] = pvec[0]*v[0] + pvec[3]*v[1] + pvec[6]*v[2];
  v_c[1] = pvec[1]*v[0] + pvec[4]*v[1] + pvec[7]*v[2];
  v_c[2] = pvec[2]*v[0] + pvec[5]*v[1] + pvec[8]*v[2];
}


static void mat_inverse(
  const double a[9],
  double b[9]
)
{
  double det, ood;

  /* determinanat */
  det = a[0]*a[4]*a[8] + a[1]*a[5]*a[6] + a[2]*a[3]*a[7]
        - a[0]*a[5]*a[7] - a[1]*a[3]*a[8] - a[2]*a[4]*a[6]; 

  /* 1/det */
  if (fabs(det)<1e-10) {  
    fprintf(stderr, "***ERROR: %s (%d)\n", __FILE__, __LINE__);
    abort(); 
  }
  ood = 1.0/det;

  b[0] = (a[4]*a[8] - a[5]*a[7])*ood;
  b[1] = (a[2]*a[7] - a[1]*a[8])*ood;
  b[2] = (a[1]*a[5] - a[2]*a[4])*ood;
  b[3] = (a[5]*a[6] - a[3]*a[8])*ood;
  b[4] = (a[0]*a[8] - a[2]*a[6])*ood;
  b[5] = (a[2]*a[3] - a[0]*a[5])*ood;
  b[6] = (a[3]*a[7] - a[4]*a[6])*ood;
  b[7] = (a[1]*a[6] - a[0]*a[7])*ood;
  b[8] = (a[0]*a[4] - a[1]*a[3])*ood;
}


/*----------------------------------------------------------------------
  EXX_Vector_C2F

  Cartesian coordinate to fractional coordinate.
----------------------------------------------------------------------*/
void EXX_Vector_C2F(
  double v_f[3],       /* (OUT) v in fractional coord. */
  const double v[3],   /* (IN) a vector in cartesian coord */
  const double pvec[9] /* (IN) primitive translational vectos */
)
{
  double pvec_i[9]; /* inverse of pvec */

  mat_inverse(pvec, pvec_i);
  
  v_f[0] = pvec_i[0]*v[0] + pvec_i[3]*v[1] + pvec_i[6]*v[2];
  v_f[1] = pvec_i[1]*v[0] + pvec_i[4]*v[1] + pvec_i[7]*v[2];
  v_f[2] = pvec_i[2]*v[0] + pvec_i[5]*v[1] + pvec_i[8]*v[2];
}

 

/*----------------------------------------------------------------------
  EXX_Vector_F2C_Offsite

  Fractional coordinate to Cartesian coordinate.
----------------------------------------------------------------------*/
void EXX_Vector_F2C_Offsite(
  double v_c[3],        /* (OUT) v in cartesian coord */
  const double v[3],    /* (IN) a vector in fractional coord */
  const double pvec[9], /* (IN) primitive translational vectors */
  int          icell,
  int          nshell
)
{
  int ncd;
  double x, y, z;
 
  ncd = 2*nshell+1;
  
  x = v[0] + (double)( icell%ncd - nshell );
  y = v[1] + (double)( (icell/ncd)%ncd - nshell );
  z = v[2] + (double)( (icell/ncd/ncd)%ncd - nshell );

  v_c[0] = pvec[0]*x + pvec[3]*y + pvec[6]*z;
  v_c[1] = pvec[1]*x + pvec[4]*y + pvec[7]*z;
  v_c[2] = pvec[2]*x + pvec[5]*y + pvec[8]*z;
}

/*----------------------------------------------------------------------
  EXX_Vector_C2S
 
  Cartesian coordinate to spherical coordinate.
----------------------------------------------------------------------*/
void EXX_Vector_C2S(
  const double v[3],
  double *r,
  double *theta,
  double *phi
)
{
  double x, y, z;
  x = v[0];
  y = v[1];
  z = v[2];

  *r     = sqrt(x*x + y*y + z*z);
  *theta = atan2(sqrt(x*x+y*y),z);
  *phi   = atan2(y, x);
}



void EXX_Vector_PAO_Overlap(
  double rc1,      /* (IN) cutoff of PAO 1 */
  double rc2,      /* (IN) cutoff of PAO 2 */
  double d,        /* (IN) displacement */
  double *pair_rc, /* (OUT) cutoff of overlap */
  double *pair_cx  /* (OUT) dividing ratio of center of overlap */
)
{
  double x, y;

  x = rc1*rc1/d/d;
  y = rc2*rc2/d/d;

  if (fabs(d)<1e-10) {
    *pair_cx = 0.5;
    *pair_rc = (rc1<rc2) ? rc1 : rc2;
    return;
  }
  
  *pair_cx = 0.5*(1.0 - x + y);
  *pair_rc = 0.5*d*sqrt( 2.0*(x+y) - 1.0 - (x-y)*(x-y) );
}


