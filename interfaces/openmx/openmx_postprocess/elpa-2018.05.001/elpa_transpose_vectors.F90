#if 0
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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
! Author: Andreas Marek, MPCDF
#endif

#include "config-f90.h"

subroutine elpa_transpose_vectors_&
&MATH_DATATYPE&
&_&
&PRECISION &
             (obj, vmat_s, ld_s, comm_s, vmat_t, ld_t, comm_t, nvs, nvr, nvc, nblk, nrThreads)

!-------------------------------------------------------------------------------
! This routine transposes an array of vectors which are distributed in
! communicator comm_s into its transposed form distributed in communicator comm_t.
! There must be an identical copy of vmat_s in every communicator comm_s.
! After this routine, there is an identical copy of vmat_t in every communicator comm_t.
!
! vmat_s    original array of vectors
! ld_s      leading dimension of vmat_s
! comm_s    communicator over which vmat_s is distributed
! vmat_t    array of vectors in transposed form
! ld_t      leading dimension of vmat_t
! comm_t    communicator over which vmat_t is distributed
! nvs       global index where to start in vmat_s/vmat_t
!           Please note: this is kind of a hint, some values before nvs will be
!           accessed in vmat_s/put into vmat_t
! nvr       global length of vmat_s/vmat_t
! nvc       number of columns in vmat_s/vmat_t
! nblk      block size of block cyclic distribution
!
!-------------------------------------------------------------------------------
    use precision
#ifdef WITH_OPENMP
   use omp_lib
#endif
   use mpi

   implicit none
!   class(elpa_abstract_impl_t), intent(inout) :: obj
   integer(kind=c_int), intent(in)                                     :: obj
   integer(kind=ik), intent(in)                      :: ld_s, comm_s, ld_t, comm_t, nvs, nvr, nvc, nblk
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(in)   :: vmat_s(ld_s,nvc)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout):: vmat_t(ld_t,nvc)

   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable  :: aux(:)
   integer(kind=ik)                                  :: myps, mypt, nps, npt
   integer(kind=ik)                                  :: n, lc, k, i, ips, ipt, ns, nl, mpierr
   integer(kind=ik)                                  :: lcm_s_t, nblks_tot, nblks_comm, nblks_skip
   integer(kind=ik)                                  :: auxstride
   integer(kind=ik), intent(in)                      :: nrThreads

   call mpi_comm_rank(comm_s,myps,mpierr)
   call mpi_comm_size(comm_s,nps ,mpierr)
   call mpi_comm_rank(comm_t,mypt,mpierr)
   call mpi_comm_size(comm_t,npt ,mpierr)

   ! The basic idea of this routine is that for every block (in the block cyclic
   ! distribution), the processor within comm_t which owns the diagonal
   ! broadcasts its values of vmat_s to all processors within comm_t.
   ! Of course this has not to be done for every block separately, since
   ! the communictation pattern repeats in the global matrix after
   ! the least common multiple of (nps,npt) blocks

   lcm_s_t   = least_common_multiple(nps,npt) ! least common multiple of nps, npt

   nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr

   ! Get the number of blocks to be skipped at the begin.
   ! This must be a multiple of lcm_s_t (else it is getting complicated),
   ! thus some elements before nvs will be accessed/set.

   nblks_skip = ((nvs-1)/(nblk*lcm_s_t))*lcm_s_t

   allocate(aux( ((nblks_tot-nblks_skip+lcm_s_t-1)/lcm_s_t) * nblk * nvc ))
#ifdef WITH_OPENMP
   !$omp parallel private(lc, i, k, ns, nl, nblks_comm, auxstride, ips, ipt, n)
#endif
   do n = 0, lcm_s_t-1

      ips = mod(n,nps)
      ipt = mod(n,npt)

      if(mypt == ipt) then

         nblks_comm = (nblks_tot-nblks_skip-n+lcm_s_t-1)/lcm_s_t
         auxstride = nblk * nblks_comm
!         if(nblks_comm==0) cycle
         if (nblks_comm .ne. 0) then
         if(myps == ips) then
!            k = 0
#ifdef WITH_OPENMP
            !$omp do
#endif
            do lc=1,nvc
               do i = nblks_skip+n, nblks_tot-1, lcm_s_t
                  k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * auxstride
                  ns = (i/nps)*nblk ! local start of block i
                  nl = min(nvr-i*nblk,nblk) ! length
                  aux(k+1:k+nl) = vmat_s(ns+1:ns+nl,lc)
!                  k = k+nblk
               enddo
            enddo
         endif

#ifdef WITH_OPENMP
         !$omp barrier
         !$omp master
#endif

#ifdef WITH_MPI

        call MPI_Bcast(aux, nblks_comm*nblk*nvc,    &
#if REALCASE == 1
                       MPI_REAL_PRECISION,    &
#endif
#if COMPLEXCASE == 1
                       MPI_COMPLEX_PRECISION, &
#endif
                       ips, comm_s, mpierr)


#endif /* WITH_MPI */

#ifdef WITH_OPENMP
         !$omp end master
         !$omp barrier

         !$omp do
#endif
!         k = 0
         do lc=1,nvc
            do i = nblks_skip+n, nblks_tot-1, lcm_s_t
               k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * auxstride
               ns = (i/npt)*nblk ! local start of block i
               nl = min(nvr-i*nblk,nblk) ! length
               vmat_t(ns+1:ns+nl,lc) = aux(k+1:k+nl)
!               k = k+nblk
            enddo
         enddo
         endif
      endif

   enddo
#ifdef WITH_OPENMP
   !$omp end parallel
#endif
   deallocate(aux)


end subroutine

