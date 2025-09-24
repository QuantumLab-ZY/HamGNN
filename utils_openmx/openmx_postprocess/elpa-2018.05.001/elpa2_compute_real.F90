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

module elpa2_compute_real

  use elpa_utilities
  USE elpa1_compute_real
  use precision
  use mpi
  use aligned_mem

  implicit none

  PRIVATE ! By default, all routines contained are private

  public :: bandred_real_double
  public :: tridiag_band_real_double
  public :: trans_ev_tridi_to_band_real_double
  public :: trans_ev_band_to_full_real_double

!  public :: bandred_complex_double
!  public :: tridiag_band_complex_double
!  public :: trans_ev_tridi_to_band_complex_double
!  public :: trans_ev_band_to_full_complex_double

  public :: band_band_real_double
!  public :: divide_band

  integer(kind=ik), public :: which_qr_decomposition = 1     ! defines, which QR-decomposition algorithm will be used
                                                    ! 0 for unblocked
                                                    ! 1 for blocked (maxrank: nblk)
  contains


#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"

#define REALCASE 1
#undef COMPLEXCASE
#include "elpa2_bandred_template.F90"
#define REALCASE 1
#include "elpa2_symm_matrix_allreduce_real_template.F90"
#undef REALCASE
#define REALCASE 1
#include "elpa2_trans_ev_band_to_full_template.F90"
#include "elpa2_tridiag_band_template.F90"
#include "elpa2_trans_ev_tridi_to_band_template.F90"


    subroutine band_band_real_&
&PRECISION &
                  (obj, na, nb, nbCol, nb2, nb2Col, ab, ab2, d, e, communicator)
    !-------------------------------------------------------------------------------
    ! band_band_real:
    ! Reduces a real symmetric banded matrix to a real symmetric matrix with smaller bandwidth. Householder transformations are not stored.
    ! Matrix size na and original bandwidth nb have to be a multiple of the target bandwidth nb2. (Hint: expand your matrix with
    ! zero entries, if this
    ! requirement doesn't hold)
    !
    !  na          Order of matrix
    !
    !  nb          Semi bandwidth of original matrix
    !
    !  nb2         Semi bandwidth of target matrix
    !
    !  ab          Input matrix with bandwidth nb. The leading dimension of the banded matrix has to be 2*nb. The parallel data layout
    !              has to be accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb+1 to min(na, block_limits(n+1)*nb)
    !              are located on rank n.
    !
    !  ab2         Output matrix with bandwidth nb2. The leading dimension of the banded matrix is 2*nb2. The parallel data layout is
    !              accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb2+1 to min(na, block_limits(n+1)*nb2) are located
    !              on rank n.
    !
    !  d(na)       Diagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
    !
    !  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
    !
    !  communicator
    !              MPI-Communicator for the total processor set
    !-------------------------------------------------------------------------------
      use elpa2_workload
      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)               :: na, nb, nbCol, nb2, nb2Col, communicator
      real(kind=rk), intent(inout)               :: ab(2*nb,nbCol) ! removed assumed size
      real(kind=rk), intent(inout)               :: ab2(2*nb2,nb2Col) ! removed assumed size
      real(kind=rk), intent(out)                 :: d(na), e(na) ! set only on PE 0

      real(kind=rk)                              :: hv(nb,nb2), w(nb,nb2), w_new(nb,nb2), tau(nb2), hv_new(nb,nb2), &
                                                  tau_new(nb2), ab_s(1+nb,nb2), ab_r(1+nb,nb2), ab_s2(2*nb2,nb2), hv_s(nb,nb2)

      real(kind=rk)                              :: work(nb*nb2), work2(nb2*nb2)
      integer(kind=ik)                         :: lwork, info

      integer(kind=ik)                         :: istep, i, n, dest
      integer(kind=ik)                         :: n_off, na_s
      integer(kind=ik)                         :: my_pe, n_pes, mpierr
      integer(kind=ik)                         :: nblocks_total, nblocks
      integer(kind=ik)                         :: nblocks_total2, nblocks2
      integer(kind=ik)                         :: ireq_ab, ireq_hv
#ifdef WITH_MPI
!      integer(kind=ik)                         :: MPI_STATUS_IGNORE(MPI_STATUS_SIZE)
#endif
!      integer(kind=ik), allocatable            :: mpi_statuses(:,:)
      integer(kind=ik), allocatable            :: block_limits(:), block_limits2(:), ireq_ab2(:)

      integer(kind=ik)                         :: j, nc, nr, ns, ne, iblk
      integer(kind=ik)                         :: istat
      character(200)                           :: errorMessage

      call mpi_comm_rank(communicator,my_pe,mpierr)
      call mpi_comm_size(communicator,n_pes,mpierr)

      ! Total number of blocks in the band:
      nblocks_total = (na-1)/nb + 1
      nblocks_total2 = (na-1)/nb2 + 1

      ! Set work distribution
      allocate(block_limits(0:n_pes), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"error allocating block_limits "//errorMessage
        stop 1
      endif
      call divide_band(obj, nblocks_total, n_pes, block_limits)

      allocate(block_limits2(0:n_pes), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"error allocating block_limits2 "//errorMessage
        stop 1
      endif

      call divide_band(obj, nblocks_total2, n_pes, block_limits2)

      ! nblocks: the number of blocks for my task
      nblocks = block_limits(my_pe+1) - block_limits(my_pe)
      nblocks2 = block_limits2(my_pe+1) - block_limits2(my_pe)

      allocate(ireq_ab2(1:nblocks2), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"error allocating ireq_ab2 "//errorMessage
        stop 1
      endif

#ifdef WITH_MPI

      ireq_ab2 = MPI_REQUEST_NULL

      if (nb2>1) then
        do i=0,nblocks2-1

          call mpi_irecv(ab2(1,i*nb2+1), 2*nb2*nb2, MPI_REAL_PRECISION, 0, 3, communicator, ireq_ab2(i+1), mpierr)
        enddo
      endif

#else /* WITH_MPI */
      ! carefull the "recieve" has to be done at the corresponding send or wait
!      if (nb2>1) then
!        do i=0,nblocks2-1
!          ab2(1:2*nb2*nb2,i*nb2+1:i*nb2+1+nb2-1) = ab_s2(1:2*nb2,i*nb2+1:nb2)
!        enddo
!      endif

#endif /* WITH_MPI */
      ! n_off: Offset of ab within band
      n_off = block_limits(my_pe)*nb
      lwork = nb*nb2
      dest = 0
#ifdef WITH_MPI
      ireq_ab = MPI_REQUEST_NULL
      ireq_hv = MPI_REQUEST_NULL
#endif
      ! ---------------------------------------------------------------------------
      ! Start of calculations

      na_s = block_limits(my_pe)*nb + 1

      if (my_pe>0 .and. na_s<=na) then
        ! send first nb2 columns to previous PE
        ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
        do i=1,nb2
          ab_s(1:nb+1,i) = ab(1:nb+1,na_s-n_off+i-1)
        enddo
#ifdef WITH_MPI

        call mpi_isend(ab_s, (nb+1)*nb2, MPI_REAL_PRECISION, my_pe-1, 1, communicator, ireq_ab, mpierr)
#endif /* WITH_MPI */
      endif

      do istep=1,na/nb2

        if (my_pe==0) then

          n = MIN(na-na_s-nb2+1,nb) ! number of rows to be reduced
          hv(:,:) = 0.0_rk
          tau(:) = 0.0_rk

          ! The last step (istep=na-1) is only needed for sending the last HH vectors.
          ! We don't want the sign of the last element flipped (analogous to the other sweeps)
          if (istep < na/nb2) then

            ! Transform first block column of remaining matrix
            call PRECISION_GEQRF(n, nb2, ab(1+nb2,na_s-n_off), 2*nb-1, tau, work, lwork, info)

            do i=1,nb2
              hv(i,i) = 1.0_rk
              hv(i+1:n,i) = ab(1+nb2+1:1+nb2+n-i,na_s-n_off+i-1)
              ab(1+nb2+1:2*nb,na_s-n_off+i-1) = 0.0_rk
            enddo

          endif

          if (nb2==1) then
            d(istep) = ab(1,na_s-n_off)
            e(istep) = ab(2,na_s-n_off)
            if (istep == na) then
              e(na) = 0.0_rk
            endif
          else
            ab_s2 = 0.0_rk
            ab_s2(:,:) = ab(1:nb2+1,na_s-n_off:na_s-n_off+nb2-1)
            if (block_limits2(dest+1)<istep) then
              dest = dest+1
            endif
#ifdef WITH_MPI
            call mpi_send(ab_s2, 2*nb2*nb2, MPI_REAL_PRECISION, dest, 3, communicator, mpierr)

#else /* WITH_MPI */
            ! do irecv here
            if (nb2>1) then
              do i= 0,nblocks2-1
                ab2(1:2*nb2*nb2,i*nb2+1:i+nb2+1+nb2-1) = ab_s2(1:2*nb2,1:nb2)
              enddo
            endif
#endif /* WITH_MPI */

          endif

        else
          if (na>na_s+nb2-1) then
            ! Receive Householder vectors from previous task, from PE owning subdiagonal
#ifdef WITH_MPI
            call mpi_recv(hv, nb*nb2, MPI_REAL_PRECISION, my_pe-1, 2, communicator, MPI_STATUS_IGNORE, mpierr)

#else /* WITH_MPI */
           hv(1:nb,1:nb2) = hv_s(1:nb,1:nb2)
#endif /* WITH_MPI */

            do i=1,nb2
              tau(i) = hv(i,i)
              hv(i,i) = 1.0_rk
            enddo
          endif
        endif

        na_s = na_s+nb2
        if (na_s-n_off > nb) then
          ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
          ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0.0_rk
          n_off = n_off + nb
        endif

        do iblk=1,nblocks
          ns = na_s + (iblk-1)*nb - n_off ! first column in block
          ne = ns+nb-nb2                    ! last column in block

          if (ns+n_off>na) exit

            nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
            nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                          ! Note that nr>=0 implies that diagonal block is full (nc==nb)!
            call wy_gen_&
      &PRECISION&
      &(obj,nc,nb2,w,hv,tau,work,nb)

            if (iblk==nblocks .and. nc==nb) then
              !request last nb2 columns
#ifdef WITH_MPI
              call mpi_recv(ab_r,(nb+1)*nb2, MPI_REAL_PRECISION, my_pe+1, 1, communicator, MPI_STATUS_IGNORE, mpierr)

#else /* WITH_MPI */
             ab_r(1:nb+1,1:nb2) = ab_s(1:nb+1,1:nb2)
#endif /* WITH_MPI */
              do i=1,nb2
                ab(1:nb+1,ne+i-1) = ab_r(:,i)
              enddo
            endif
            hv_new(:,:) = 0.0_rk ! Needed, last rows must be 0 for nr < nb
            tau_new(:) = 0.0_rk

            if (nr>0) then
              call wy_right_&
        &PRECISION&
        &(obj,nr,nb,nb2,ab(nb+1,ns),2*nb-1,w,hv,work,nb)
              call PRECISION_GEQRF(nr, nb2, ab(nb+1,ns), 2*nb-1, tau_new, work, lwork, info)
              do i=1,nb2
                hv_new(i,i) = 1.0_rk
                hv_new(i+1:,i) = ab(nb+2:2*nb-i+1,ns+i-1)
                ab(nb+2:,ns+i-1) = 0.0_rk
              enddo

              !send hh-Vector
              if (iblk==nblocks) then
#ifdef WITH_MPI

                call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

#endif
                hv_s = hv_new
                do i=1,nb2
                  hv_s(i,i) = tau_new(i)
                enddo
#ifdef WITH_MPI
                call mpi_isend(hv_s,nb*nb2, MPI_REAL_PRECISION, my_pe+1, 2, communicator, ireq_hv, mpierr)

#else /* WITH_MPI */

#endif /* WITH_MPI */
              endif
            endif

            call wy_symm_&
      &PRECISION&
      &(obj,nc,nb2,ab(1,ns),2*nb-1,w,hv,work,work2,nb)

            if (my_pe>0 .and. iblk==1) then
              !send first nb2 columns to previous PE
#ifdef WITH_MPI

              call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)

#endif
              do i=1,nb2
                ab_s(1:nb+1,i) = ab(1:nb+1,ns+i-1)
              enddo
#ifdef WITH_MPI
              call mpi_isend(ab_s,(nb+1)*nb2, MPI_REAL_PRECISION, my_pe-1, 1, communicator, ireq_ab, mpierr)

#else /* WITH_MPI */

#endif /* WITH_MPI */
            endif

            if (nr>0) then
              call wy_gen_&
        &PRECISION&
        &(obj,nr,nb2,w_new,hv_new,tau_new,work,nb)
              call wy_left_&
        &PRECISION&
        &(obj,nb-nb2,nr,nb2,ab(nb+1-nb2,ns+nb2),2*nb-1,w_new,hv_new,work,nb)
            endif

            ! Use new HH Vector for the next block
            hv(:,:) = hv_new(:,:)
            tau = tau_new
          enddo
        enddo

        ! Finish the last outstanding requests
#ifdef WITH_MPI

        call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
        call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
!        allocate(mpi_statuses(MPI_STATUS_SIZE,nblocks2), stat=istat, errmsg=errorMessage)
!        if (istat .ne. 0) then
!          print *,"error allocating mpi_statuses "//errorMessage
!          stop 1
!        endif

        call mpi_waitall(nblocks2,ireq_ab2,MPI_STATUSES_IGNORE,mpierr)
!        deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
!        if (istat .ne. 0) then
!          print *,"error deallocating mpi_statuses "//errorMessage
!          stop 1
!        endif

        call mpi_barrier(communicator,mpierr)

#endif /* WITH_MPI */

        deallocate(block_limits, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"error deallocating block_limits "//errorMessage
          stop 1
        endif

        deallocate(block_limits2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"error deallocating block_limits2 "//errorMessage
          stop 1
        endif

        deallocate(ireq_ab2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"error deallocating ireq_ab2 "//errorMessage
          stop 1
        endif


    end subroutine

    subroutine wy_gen_&
    &PRECISION&
    &(obj, n, nb, W, Y, tau, mem, lda)

      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)            :: n      !length of householder-vectors
      integer(kind=ik), intent(in)            :: nb     !number of householder-vectors
      integer(kind=ik), intent(in)            :: lda        !leading dimension of Y and W
      real(kind=rk), intent(in)               :: Y(lda,nb)  !matrix containing nb householder-vectors of length b
      real(kind=rk), intent(in)               :: tau(nb)    !tau values
      real(kind=rk), intent(out)              :: W(lda,nb)  !output matrix W
      real(kind=rk), intent(in)               :: mem(nb)    !memory for a temporary matrix of size nb

      integer(kind=ik)                        :: i


   W(1:n,1) = tau(1)*Y(1:n,1)
   do i=2,nb
     W(1:n,i) = tau(i)*Y(1:n,i)
     call PRECISION_GEMV('T', n, i-1,  1.0_rk, Y, lda, W(1,i), 1, 0.0_rk, mem,1)
     call PRECISION_GEMV('N', n, i-1, -1.0_rk, W, lda, mem, 1, 1.0_rk, W(1,i),1)
   enddo
    end subroutine

    subroutine wy_left_&
    &PRECISION&
    &(obj, n, m, nb, A, lda, W, Y, mem, lda2)

      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)            :: n      !width of the matrix A
      integer(kind=ik), intent(in)            :: m      !length of matrix W and Y
      integer(kind=ik), intent(in)            :: nb     !width of matrix W and Y
      integer(kind=ik), intent(in)            :: lda        !leading dimension of A
      integer(kind=ik), intent(in)            :: lda2       !leading dimension of W and Y
      real(kind=rk), intent(inout)            :: A(lda,*)   !matrix to be transformed   ! remove assumed size
      real(kind=rk), intent(in)               :: W(m,nb)    !blocked transformation matrix W
      real(kind=rk), intent(in)               :: Y(m,nb)    !blocked transformation matrix Y
      real(kind=rk), intent(inout)            :: mem(n,nb)  !memory for a temporary matrix of size n x nb

   call PRECISION_GEMM('T', 'N', nb, n, m, 1.0_rk, W, lda2, A, lda, 0.0_rk, mem, nb)
   call PRECISION_GEMM('N', 'N', m, n, nb, -1.0_rk, Y, lda2, mem, nb, 1.0_rk, A, lda)
    end subroutine

    subroutine wy_right_&
    &PRECISION&
    &(obj, n, m, nb, A, lda, W, Y, mem, lda2)

      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)            :: n      !height of the matrix A
      integer(kind=ik), intent(in)            :: m      !length of matrix W and Y
      integer(kind=ik), intent(in)            :: nb     !width of matrix W and Y
      integer(kind=ik), intent(in)            :: lda        !leading dimension of A
      integer(kind=ik), intent(in)            :: lda2       !leading dimension of W and Y
      real(kind=rk), intent(inout)            :: A(lda,*)   !matrix to be transformed  ! remove assumed size
      real(kind=rk), intent(in)               :: W(m,nb)    !blocked transformation matrix W
      real(kind=rk), intent(in)               :: Y(m,nb)    !blocked transformation matrix Y
      real(kind=rk), intent(inout)            :: mem(n,nb)  !memory for a temporary matrix of size n x nb


      call PRECISION_GEMM('N', 'N', n, nb, m, 1.0_rk, A, lda, W, lda2, 0.0_rk, mem, n)
      call PRECISION_GEMM('N', 'T', n, m, nb, -1.0_rk, mem, n, Y, lda2, 1.0_rk, A, lda)

    end subroutine

    subroutine wy_symm_&
    &PRECISION&
    &(obj, n, nb, A, lda, W, Y, mem, mem2, lda2)

      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)            :: n      !width/heigth of the matrix A; length of matrix W and Y
      integer(kind=ik), intent(in)            :: nb     !width of matrix W and Y
      integer(kind=ik), intent(in)            :: lda        !leading dimension of A
      integer(kind=ik), intent(in)            :: lda2       !leading dimension of W and Y
      real(kind=rk), intent(inout)            :: A(lda,*)   !matrix to be transformed  ! remove assumed size
      real(kind=rk), intent(in)               :: W(n,nb)    !blocked transformation matrix W
      real(kind=rk), intent(in)               :: Y(n,nb)    !blocked transformation matrix Y
      real(kind=rk)                           :: mem(n,nb)  !memory for a temporary matrix of size n x nb
      real(kind=rk)                           :: mem2(nb,nb)    !memory for a temporary matrix of size nb x nb

      call PRECISION_SYMM('L', 'L', n, nb, 1.0_rk, A, lda, W, lda2, 0.0_rk, mem, n)
      call PRECISION_GEMM('T', 'N', nb, nb, n, 1.0_rk, mem, n, W, lda2, 0.0_rk, mem2, nb)
      call PRECISION_GEMM('N', 'N', n, nb, nb, -0.5_rk, Y, lda2, mem2, nb, 1.0_rk, mem, n)
      call PRECISION_SYR2K('L', 'N', n, nb, -1.0_rk, Y, lda2, mem, n, 1.0_rk, A, lda)

    end subroutine

#define REALCASE 1
#undef COMPLEXCASE

end module elpa2_compute_real
