!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "config-f90.h"
!> \brief Fortran module which contains the source of ELPA 1stage
module elpa1_compute_real
  use elpa_utilities
  use mpi
  implicit none

  PRIVATE ! set default to private

  public :: solve_tridi_double
  public :: solve_tridi_double_impl

  public :: hh_transform_real_double
  public :: hh_transform_real
  public :: elpa_reduce_add_vectors_real_double
  public :: elpa_reduce_add_vectors_real
  public :: elpa_transpose_vectors_real_double
  public :: elpa_transpose_vectors_real

  interface hh_transform_real
    module procedure hh_transform_real_double
  end interface

  interface elpa_reduce_add_vectors_real
    module procedure elpa_reduce_add_vectors_real_double
  end interface

  interface elpa_transpose_vectors_real
    module procedure elpa_transpose_vectors_real_double
  end interface

  contains

#define DOUBLE_PRECISION_REAL 1
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"
#include "elpa_transpose_vectors.F90"
#include "elpa_reduce_add_vectors.F90"

#ifdef DOUBLE_PRECISION_REAL
#define PRECISION_AND_SUFFIX double
#else
#define PRECISION_AND_SUFFIX single
#endif
#include "elpa1_solve_tridi_real_template.F90"
#undef PRECISION_AND_SUFFIX
#ifdef DOUBLE_PRECISION_REAL
#define PRECISION_AND_SUFFIX  double_impl
#else
#define PRECISION_AND_SUFFIX  single_impl
#endif
#include "elpa1_solve_tridi_real_template.F90"
#undef PRECISION_AND_SUFFIX
#include "elpa1_merge_systems_real_template.F90"
#include "elpa1_tools_template.F90"


#undef DOUBLE_PRECISION
#undef REALCASE


end module elpa1_compute_real
