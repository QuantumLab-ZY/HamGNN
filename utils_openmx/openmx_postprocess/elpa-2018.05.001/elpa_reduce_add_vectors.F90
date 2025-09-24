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
! Author: Andreas Marek, MPCDF
#endif

#include "config-f90.h"

subroutine elpa_reduce_add_vectors_&
&MATH_DATATYPE&
&_&
&PRECISION &
            (obj, vmat_s, ld_s, comm_s, vmat_t, ld_t, comm_t, nvr, nvc, nblk, nrThreads)

!-------------------------------------------------------------------------------
! This routine does a reduce of all vectors in vmat_s over the communicator comm_t.
! The result of the reduce is gathered on the processors owning the diagonal
! and added to the array of vectors vmat_t (which is distributed over comm_t).
!
! Opposed to elpa_transpose_vectors, there is NO identical copy of vmat_s
! in the different members within vmat_t (else a reduce wouldn't be necessary).
! After this routine, an allreduce of vmat_t has to be done.
!
! vmat_s    array of vectors to be reduced and added
! ld_s      leading dimension of vmat_s
! comm_s    communicator over which vmat_s is distributed
! vmat_t    array of vectors to which vmat_s is added
! ld_t      leading dimension of vmat_t
! comm_t    communicator over which vmat_t is distributed
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

!   class(elpa_abstract_impl_t), intent(inout)         :: obj
   integer(kind=c_int), intent(in)                                     :: obj
   integer(kind=ik), intent(in)                       :: ld_s, comm_s, ld_t, comm_t, nvr, nvc, nblk
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(in)    :: vmat_s(ld_s,nvc)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout) :: vmat_t(ld_t,nvc)

   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable   :: aux1(:), aux2(:)
   integer(kind=ik)                                   :: myps, mypt, nps, npt
   integer(kind=ik)                                   :: n, lc, k, i, ips, ipt, ns, nl, mpierr
   integer(kind=ik)                                   :: lcm_s_t, nblks_tot
   integer(kind=ik)                                   :: auxstride
   integer(kind=ik), intent(in)                       :: nrThreads


   call mpi_comm_rank(comm_s,myps,mpierr)
   call mpi_comm_size(comm_s,nps ,mpierr)
   call mpi_comm_rank(comm_t,mypt,mpierr)
   call mpi_comm_size(comm_t,npt ,mpierr)

   ! Look to elpa_transpose_vectors for the basic idea!

   ! The communictation pattern repeats in the global matrix after
   ! the least common multiple of (nps,npt) blocks

   lcm_s_t   = least_common_multiple(nps,npt) ! least common multiple of nps, npt

   nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr

   allocate(aux1( ((nblks_tot+lcm_s_t-1)/lcm_s_t) * nblk * nvc ))
   allocate(aux2( ((nblks_tot+lcm_s_t-1)/lcm_s_t) * nblk * nvc ))
   aux1(:) = 0
   aux2(:) = 0
#ifdef WITH_OPENMP
   !call omp_set_num_threads(nrThreads)

   !$omp parallel private(ips, ipt, auxstride, lc, i, k, ns, nl) num_threads(nrThreads)
#endif
   do n = 0, lcm_s_t-1

      ips = mod(n,nps)
      ipt = mod(n,npt)

      auxstride = nblk * ((nblks_tot - n + lcm_s_t - 1)/lcm_s_t)

      if(myps == ips) then

!         k = 0
#ifdef WITH_OPENMP
         !$omp do
#endif
         do lc=1,nvc
            do i = n, nblks_tot-1, lcm_s_t
               k = (i - n)/lcm_s_t * nblk + (lc - 1) * auxstride
               ns = (i/nps)*nblk ! local start of block i
               nl = min(nvr-i*nblk,nblk) ! length
               aux1(k+1:k+nl) = vmat_s(ns+1:ns+nl,lc)
!               k = k+nblk
            enddo
         enddo

         k = nvc * auxstride
#ifdef WITH_OPENMP
         !$omp barrier
         !$omp master
#endif

#ifdef WITH_MPI

         if (k>0) call mpi_reduce(aux1, aux2, k, &
#if REALCASE == 1
                                  MPI_REAL_PRECISION,  &
#endif
#if COMPLEXCASE == 1
                                  MPI_COMPLEX_PRECISION, &
#endif
                                  MPI_SUM, ipt, comm_t, mpierr)


#else /* WITH_MPI */
         if(k>0) aux2 = aux1
#endif /* WITH_MPI */

#ifdef WITH_OPENMP
         !$omp end master
         !$omp barrier
#endif
         if (mypt == ipt) then
!            k = 0
#ifdef WITH_OPENMP
         !$omp do
#endif
            do lc=1,nvc
               do i = n, nblks_tot-1, lcm_s_t
                  k = (i - n)/lcm_s_t * nblk + (lc - 1) * auxstride
                  ns = (i/npt)*nblk ! local start of block i
                  nl = min(nvr-i*nblk,nblk) ! length
                  vmat_t(ns+1:ns+nl,lc) = vmat_t(ns+1:ns+nl,lc) + aux2(k+1:k+nl)
!                  k = k+nblk
               enddo
            enddo
         endif

      endif

   enddo
#ifdef WITH_OPENMP
   !$omp end parallel
#endif

   deallocate(aux1)
   deallocate(aux2)

end subroutine


