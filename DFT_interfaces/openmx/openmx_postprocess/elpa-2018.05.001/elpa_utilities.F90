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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: Andreas Marek, MPCDF

#include "config-f90.h"

module ELPA_utilities

#ifdef HAVE_ISO_FORTRAN_ENV
  use iso_fortran_env, only : output_unit, error_unit
#endif
  use, intrinsic :: iso_c_binding
  implicit none

  private ! By default, all routines contained are private

  public :: output_unit, error_unit
  public :: check_alloc, check_alloc_CUDA_f, check_memcpy_CUDA_f, check_dealloc_CUDA_f
  public :: map_global_array_index_to_local_index
  public :: pcol, prow
  public :: local_index                ! Get local index of a block cyclic distributed matrix
  public :: least_common_multiple      ! Get least common multiple

#ifndef HAVE_ISO_FORTRAN_ENV
  integer, parameter :: error_unit = 0
  integer, parameter :: output_unit = 6
#endif

  !******
  contains

#include "prow_pcol.F90"

!-------------------------------------------------------------------------------
#include "map_global_to_local.F90"

 integer function local_index(idx, my_proc, num_procs, nblk, iflag)

!-------------------------------------------------------------------------------
!  local_index: returns the local index for a given global index
!               If the global index has no local index on the
!               processor my_proc behaviour is defined by iflag
!
!  Parameters
!
!  idx         Global index
!
!  my_proc     Processor row/column for which to calculate the local index
!
!  num_procs   Total number of processors along row/column
!
!  nblk        Blocksize
!
!  iflag       Controls the behaviour if idx is not on local processor
!              iflag< 0 : Return last local index before that row/col
!              iflag==0 : Return 0
!              iflag> 0 : Return next local index after that row/col
!-------------------------------------------------------------------------------
    implicit none

    integer(kind=c_int) :: idx, my_proc, num_procs, nblk, iflag

    integer(kind=c_int) :: iblk

    iblk = (idx-1)/nblk  ! global block number, 0 based

    if (mod(iblk,num_procs) == my_proc) then

    ! block is local, always return local row/col number

    local_index = (iblk/num_procs)*nblk + mod(idx-1,nblk) + 1

    else

    ! non local block

    if (iflag == 0) then

        local_index = 0

    else

        local_index = (iblk/num_procs)*nblk

        if (mod(iblk,num_procs) > my_proc) local_index = local_index + nblk

        if (iflag>0) local_index = local_index + 1
    endif
    endif

 end function local_index

 integer function least_common_multiple(a, b)

    ! Returns the least common multiple of a and b
    ! There may be more efficient ways to do this, we use the most simple approach
    implicit none
    integer(kind=c_int), intent(in) :: a, b

    do least_common_multiple = a, a*(b-1), a
    if(mod(least_common_multiple,b)==0) exit
    enddo
    ! if the loop is left regularly, least_common_multiple = a*b

 end function least_common_multiple

 subroutine check_alloc(function_name, variable_name, istat, errorMessage)

    implicit none

    character(len=*), intent(in)    :: function_name
    character(len=*), intent(in)    :: variable_name
    integer(kind=c_int), intent(in)    :: istat
    character(len=*), intent(in)    :: errorMessage

    if (istat .ne. 0) then
      print *, function_name, ": error when allocating ", variable_name, " ", errorMessage
      stop 1
    endif
 end subroutine

 subroutine check_alloc_CUDA_f(file_name, line, successCUDA)

    implicit none

    character(len=*), intent(in)    :: file_name
    integer(kind=c_int), intent(in)    :: line
    logical                         :: successCUDA

    if (.not.(successCUDA)) then
      print *, file_name, ":", line,  " error in cuda_malloc when allocating "
      stop 1
    endif
 end subroutine

 subroutine check_dealloc_CUDA_f(file_name, line, successCUDA)

    implicit none

    character(len=*), intent(in)    :: file_name
    integer(kind=c_int), intent(in)    :: line
    logical                         :: successCUDA

    if (.not.(successCUDA)) then
      print *, file_name, ":", line,  " error in cuda_free when deallocating "
      stop 1
    endif
 end subroutine

 subroutine check_memcpy_CUDA_f(file_name, line, successCUDA)

    implicit none

    character(len=*), intent(in)    :: file_name
    integer(kind=c_int), intent(in)    :: line
    logical                         :: successCUDA

    if (.not.(successCUDA)) then
      print *, file_name, ":", line,  " error in cuda_memcpy when copying "
      stop 1
    endif
 end subroutine

end module ELPA_utilities
