#ifdef REALCASE
#ifdef DOUBLE_PRECISION
  integer, parameter :: rk = C_DOUBLE
  integer, parameter :: rck = C_DOUBLE
#endif
#ifdef SINGLE_PRECISION
  integer, parameter :: rk = C_FLOAT
  integer, parameter :: rck = C_FLOAT
#endif
  real(kind=rck), parameter      :: ZERO=0.0_rk, ONE = 1.0_rk
#endif

#ifdef COMPLEXCASE
#ifdef DOUBLE_PRECISION
  integer, parameter :: rk = C_DOUBLE
  integer, parameter :: ck = C_DOUBLE_COMPLEX
  integer, parameter :: rck = C_DOUBLE_COMPLEX
#endif
#ifdef SINGLE_PRECISION
  integer, parameter :: rk = C_FLOAT
  integer, parameter :: ck = C_FLOAT_COMPLEX
  integer, parameter :: rck = C_FLOAT_COMPLEX
#endif
  complex(kind=rck), parameter     :: ZERO = (0.0_rk,0.0_rk), ONE = (1.0_rk,0.0_rk)
#endif
