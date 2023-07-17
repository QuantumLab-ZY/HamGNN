!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), fomerly known as
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
! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: Andreas Marek, MPCDF
#include "config-f90.h"

module elpa2_compute_complex

  use elpa_utilities
  USE elpa1_compute_complex
  use precision
  use mpi
  use aligned_mem
  use redist_complex
  use pack_unpack_cpu_complex
  use compute_hh_trafo_complex

  implicit none

  PRIVATE ! By default, all routines contained are private

!  public :: bandred_real_double
!  public :: tridiag_band_real_double
!  public :: trans_ev_tridi_to_band_real_double
!  public :: trans_ev_band_to_full_real_double

  public :: bandred_complex_double
  public :: tridiag_band_complex_double
  public :: trans_ev_tridi_to_band_complex_double
  public :: trans_ev_band_to_full_complex_double

!  public :: band_band_real_double
!  public :: divide_band

!  integer(kind=ik), public :: which_qr_decomposition = 1     ! defines, which QR-decomposition algorithm will be used
                                                    ! 0 for unblocked
                                                    ! 1 for blocked (maxrank: nblk)
  contains


#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"

#define COMPLEXCASE 1
#undef REALCASE
#include "elpa2_bandred_template.F90"
#undef COMPLEXCASE
#define COMPLEXCASE 1
#include "elpa2_herm_matrix_allreduce_complex_template.F90"
#undef COMPLEXCASE
#define COMPLEXCASE 1
#include "elpa2_trans_ev_band_to_full_template.F90"
#include "elpa2_tridiag_band_template.F90"
#include "elpa2_trans_ev_tridi_to_band_template.F90"

#undef COMPLEXCASE
#undef DOUBLE_PRECISION

end module elpa2_compute_complex
