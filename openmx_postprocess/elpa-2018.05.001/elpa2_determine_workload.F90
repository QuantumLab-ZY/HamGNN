!   This file is part of ELPA.
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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".



! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "config-f90.h"

module elpa2_workload

  implicit none
  private

  public :: determine_workload
  public :: divide_band

  contains
    subroutine determine_workload(obj, na, nb, nprocs, limits)
      use precision
      implicit none

!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)  :: na, nb, nprocs
      integer(kind=ik), intent(out) :: limits(0:nprocs)

      integer(kind=ik)              :: i


      if (na <= 0) then
        limits(:) = 0

        return
      endif

      if (nb*nprocs > na) then
          ! there is not enough work for all
        do i = 0, nprocs
          limits(i) = min(na, i*nb)
        enddo
      else
         do i = 0, nprocs
           limits(i) = (i*na)/nprocs
         enddo
      endif

    end subroutine
    !---------------------------------------------------------------------------------------------------
    ! divide_band: sets the work distribution in band
    ! Proc n works on blocks block_limits(n)+1 .. block_limits(n+1)

    subroutine divide_band(obj, nblocks_total, n_pes, block_limits)
      use precision
      implicit none
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)  :: nblocks_total ! total number of blocks in band
      integer(kind=ik), intent(in)  :: n_pes         ! number of PEs for division
      integer(kind=ik), intent(out) :: block_limits(0:n_pes)

      integer(kind=ik)              :: n, nblocks, nblocks_left


      block_limits(0) = 0
      if (nblocks_total < n_pes) then
        ! Not enough work for all: The first tasks get exactly 1 block
        do n=1,n_pes
          block_limits(n) = min(nblocks_total,n)
        enddo
      else
        ! Enough work for all. If there is no exact loadbalance,
        ! the LAST tasks get more work since they are finishing earlier!
        nblocks = nblocks_total/n_pes
        nblocks_left = nblocks_total - n_pes*nblocks
        do n=1,n_pes
          if (n<=n_pes-nblocks_left) then
            block_limits(n) = block_limits(n-1) + nblocks
          else
            block_limits(n) = block_limits(n-1) + nblocks + 1
          endif
        enddo
      endif


    end subroutine
end module elpa2_workload
