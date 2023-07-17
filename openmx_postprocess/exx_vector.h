/*----------------------------------------------------------------------
  exx_vector.h

  Some simple geometrical calculations.

  Coded by M. Toyoda 07/JAN/2010
----------------------------------------------------------------------*/
#ifndef EXX_VECTOR_H_INCLUDED
#define EXX_VECTOR_H_INCLUDED

double EXX_Vector_Distance(const double v[3], const double w[3]);


void EXX_Vector_F2C(
  double v_c[3],       /* (OUT) v in cartesian coord */
  const double v[3],   /* (IN) a vector in fractional coord */
  const double pvec[9] /* (IN) primitive translational vectors */
);


void EXX_Vector_C2F(
  double v_f[3],       /* (OUT) v in cartesian coord */
  const double v[3],   /* (IN) a vector in fractional coord */
  const double pvec[9] /* (IN) primitive translational vectors */
);


void EXX_Vector_F2C_Offsite(
  double       v_c[3],  /* (OUT) v in cartesian coord */
  const double v[3],    /* (IN) a vector in fractional coord */
  const double pvec[9], /* (IN) primitive translational vectors */
  int          icell,
  int          nshell
);


void EXX_Vector_C2S(
  const double v[3], /* (IN) a vector in cartesian */
  double *r,         /* (OUT) v in spherical coord */
  double *theta,
  double *phi
);


void EXX_Vector_PAO_Overlap(
  double rc1,      /* (IN) cutoff of PAO 1 */
  double rc2,      /* (IN) cutoff of PAO 2 */
  double d,        /* (IN) displacement */
  double *pair_rc, /* (OUT) cutoff of overlap */
  double *pair_cx  /* (OUT) dividing ratio of center of overlap */
);

#endif /* EXX_VECTOR_H_INCLUDED */
