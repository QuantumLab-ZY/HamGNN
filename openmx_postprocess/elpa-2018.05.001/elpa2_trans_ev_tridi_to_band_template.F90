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


  subroutine trans_ev_tridi_to_band_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, na, nev, nblk, nbw, q, q_dev, ldq, matrixCols,         &
     hh_trans, mpi_comm_rows, mpi_comm_cols, wantDebug, useGPU, max_threads, success, &
     kernel)

    !-------------------------------------------------------------------------------
    !  trans_ev_tridi_to_band_real/complex:
    !  Transforms the eigenvectors of a tridiagonal matrix back to the eigenvectors of the band matrix
    !
    !  Parameters
    !
    !  na          Order of matrix a, number of rows of matrix q
    !
    !  nev         Number eigenvectors to compute (= columns of matrix q)
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  nb          semi bandwith
    !
    !  q           On input: Eigenvectors of tridiagonal matrix
    !              On output: Transformed eigenvectors
    !              Distribution is like in Scalapack.
    !
    !  q_dev       GPU device pointer to q
    !
    !  ldq         Leading dimension of q
    !  matrixCols  local columns of matrix q
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns/both
    !
    !-------------------------------------------------------------------------------
      use elpa2_workload
      use pack_unpack_cpu_real
      use compute_hh_trafo_real
      use precision
      use iso_c_binding
#ifdef WITH_OPENMP
      ! use omp_lib
#endif
      implicit none
#include "precision_kinds.F90"
#include "fortran_constants.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      logical, intent(in)                        :: useGPU

      integer(kind=ik), intent(in)               :: kernel
      integer(kind=ik), intent(in)               :: na, nev, nblk, nbw, ldq, matrixCols, mpi_comm_rows, mpi_comm_cols

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck)                    :: q(ldq,*)
#else
      MATH_DATATYPE(kind=rck)                    :: q(ldq,matrixCols)
#endif

      MATH_DATATYPE(kind=rck), intent(in)        :: hh_trans(:,:)
      integer(kind=c_intptr_t)                   :: q_dev

      integer(kind=ik)                           :: np_rows, my_prow, np_cols, my_pcol
      integer(kind=ik)                           :: i, j, ip, sweep, nbuf, l_nev, a_dim2
      integer(kind=ik)                           :: current_n, current_local_n, current_n_start, current_n_end
      integer(kind=ik)                           :: next_n, next_local_n, next_n_start, next_n_end
      integer(kind=ik)                           :: bottom_msg_length, top_msg_length, next_top_msg_length
      integer(kind=ik)                           :: stripe_width, last_stripe_width, stripe_count
#ifdef WITH_OPENMP
      integer(kind=ik)                           :: thread_width, csw, b_off, b_len
#endif
      integer(kind=ik)                           :: num_result_blocks, num_result_buffers, num_bufs_recvd
      integer(kind=ik)                           :: a_off, current_tv_off, max_blk_size
      integer(kind=ik)                           :: mpierr, src, src_offset, dst, offset, nfact, num_blk

      logical                                    :: flag
#ifdef WITH_OPENMP
      MATH_DATATYPE(kind=rck), pointer           :: aIntern(:,:,:,:)
#else
      MATH_DATATYPE(kind=rck), pointer           :: aIntern(:,:,:)
#endif
      MATH_DATATYPE(kind=rck)                    :: a_var

      type(c_ptr)                                :: aIntern_ptr

      MATH_DATATYPE(kind=rck)   , allocatable    :: row(:)
      MATH_DATATYPE(kind=rck)   , allocatable    :: row_group(:,:)

#ifdef WITH_OPENMP
      MATH_DATATYPE(kind=rck), allocatable       :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
      MATH_DATATYPE(kind=rck), allocatable       :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
#else
      MATH_DATATYPE(kind=rck), allocatable       :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
      MATH_DATATYPE(kind=rck), allocatable       :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
#endif

      integer(kind=c_intptr_t)                   :: aIntern_dev
      integer(kind=c_intptr_t)                   :: bcast_buffer_dev
      integer(kind=c_intptr_t)                   :: num
      integer(kind=c_intptr_t)                   :: dev_offset, dev_offset_1
      integer(kind=c_intptr_t)                   :: row_dev
      integer(kind=c_intptr_t)                   :: row_group_dev
      integer(kind=c_intptr_t)                   :: hh_tau_dev
      integer(kind=c_intptr_t)                   :: hh_dot_dev
      integer(kind=ik)                           :: row_group_size, unpack_idx

      integer(kind=ik)                           :: n_times
      integer(kind=ik)                           :: top, chunk, this_chunk

      MATH_DATATYPE(kind=rck), allocatable       :: result_buffer(:,:,:)
      MATH_DATATYPE(kind=rck), allocatable       :: bcast_buffer(:,:)

      integer(kind=ik)                           :: n_off

      integer(kind=ik), allocatable              :: result_send_request(:), result_recv_request(:), limits(:)
      integer(kind=ik), allocatable              :: top_send_request(:), bottom_send_request(:)
      integer(kind=ik), allocatable              :: top_recv_request(:), bottom_recv_request(:)

      ! MPI send/recv tags, arbitrary

      integer(kind=ik), parameter                :: bottom_recv_tag = 111
      integer(kind=ik), parameter                :: top_recv_tag    = 222
      integer(kind=ik), parameter                :: result_recv_tag = 333

      integer(kind=ik), intent(in)               :: max_threads
 
#ifdef WITH_OPENMP
      integer(kind=ik)                           :: my_thread
#endif


      ! Just for measuring the kernel performance
      real(kind=c_double)                        :: kernel_time, kernel_time_recv ! MPI_WTIME always needs double
      ! long integer
      integer(kind=lik)                          :: kernel_flops, kernel_flops_recv

      logical, intent(in)                        :: wantDebug
      logical                                    :: success
      integer(kind=ik)                           :: istat, print_flops
      character(200)                             :: errorMessage
      character(20)                              :: gpuString
      logical                                    :: successCUDA
#ifndef WITH_MPI
      integer(kind=ik)                           :: j1
#endif
      integer(kind=ik)                           :: error
!      integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
!                                                                     &PRECISION&
!                                                                     &_&
!                                                                     &MATH_DATATYPE

      if(useGPU) then
        gpuString = "_gpu"
      else
        gpuString = ""
      endif


      n_times = 0
      if (useGPU) then
        unpack_idx = 0
        row_group_size = 0
      endif

      success = .true.
      kernel_time = 0.0
      kernel_flops = 0

      call MPI_Comm_rank(mpi_comm_rows, my_prow, mpierr)
      call MPI_Comm_size(mpi_comm_rows, np_rows, mpierr)
      call MPI_Comm_rank(mpi_comm_cols, my_pcol, mpierr)
      call MPI_Comm_size(mpi_comm_cols, np_cols, mpierr)

      if (mod(nbw,nblk)/=0) then
        if (my_prow==0 .and. my_pcol==0) then
          if (wantDebug) then
            write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_&
                                &MATH_DATATYPE&
                                &: ERROR: nbw=',nbw,', nblk=',nblk
            write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_&
                                &MATH_DATATYPE&
                                &: band backtransform works only for nbw==n*nblk'
          endif
          success = .false.
          return
        endif
      endif

      nfact = nbw / nblk


      ! local number of eigenvectors
      l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

      if (l_nev==0) then
#ifdef WITH_OPENMP
        thread_width = 0
#endif
        stripe_width = 0
        stripe_count = 0
        last_stripe_width = 0

      else ! l_nev

#if WITH_OPENMP
        ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
        ! every primary cache
        ! Suggested stripe width is 48 - should this be reduced for the complex case ???

        if (useGPU) then
          stripe_width = 256 ! Must be a multiple of 4
          stripe_count = (l_nev - 1) / stripe_width + 1
        else ! useGPU
          ! openmp only in non-GPU case
          thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
          stripe_width = 48 ! Must be a multiple of 4
#else
          stripe_width = 96 ! Must be a multiple of 8
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
          stripe_width = 48 ! Must be a multiple of 2
#else
          stripe_width = 48 ! Must be a multiple of 4
#endif
#endif /* COMPLEXCASE */

          stripe_count = (thread_width-1)/stripe_width + 1

          ! Adapt stripe width so that last one doesn't get too small

          stripe_width = (thread_width-1)/stripe_count + 1

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
          if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6) then

            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                                  ! (8 * sizeof(double) == 64)

    else
            stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                            ! (4 * sizeof(double) == 32)
          endif
#else
          if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6) then


            stripe_width = ((stripe_width+15)/16)*16 ! Must be a multiple of 16 because of AVX-512 memory alignment of 64 bytes
                                               ! (16 * sizeof(float) == 64)

    else
            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX/SSE memory alignment of 32 bytes
                                            ! (8 * sizeof(float) == 32)
          endif
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
          if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
              kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2) then

            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 4 because of AVX-512 memory alignment of 64 bytes
                                            ! (4 * sizeof(double complex) == 64)

    else

            stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 2 because of AVX/SSE memory alignment of 32 bytes
                                            ! (2 * sizeof(double complex) == 32)
    endif
#else

          if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
              kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2) then

            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                            ! (8 * sizeof(float complex) == 64)

          else
            stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                            ! (4 * sizeof(float complex) == 32)
    endif
#endif
#endif /* COMPLEXCASE */

#if REALCASE == 1
        last_stripe_width = l_nev - (stripe_count-1)*stripe_width
#endif
#if COMPLEXCASE == 1
! only needed in no OMP case check thsis
! last_stripe_width = l_nev - (stripe_count-1)*stripe_width
#endif

        endif ! useGPU

#else /* WITH_OPENMP */

        ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
        ! every primary cache
        ! Suggested stripe width is 48 - should this be reduced for the complex case ???

        if (useGPU) then
          stripe_width = 256 ! Must be a multiple of 4
          stripe_count = (l_nev - 1) / stripe_width + 1

        else ! useGPU
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
          stripe_width = 48 ! Must be a multiple of 4
#else
          stripe_width = 96 ! Must be a multiple of 8
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
          stripe_width = 48 ! Must be a multiple of 2
#else
          stripe_width = 48 ! Must be a multiple of 4
#endif
#endif /* COMPLEXCASE */

          stripe_count = (l_nev-1)/stripe_width + 1

          ! Adapt stripe width so that last one doesn't get too small

          stripe_width = (l_nev-1)/stripe_count + 1

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
          if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6) then

            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                                  ! (8 * sizeof(double) == 64)

    else
            stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                            ! (4 * sizeof(double) == 32)
          endif
#else
          if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
              kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6) then


            stripe_width = ((stripe_width+15)/16)*16 ! Must be a multiple of 16 because of AVX-512 memory alignment of 64 bytes
                                               ! (16 * sizeof(float) == 64)

    else
            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX/SSE memory alignment of 32 bytes
                                            ! (8 * sizeof(float) == 32)
          endif
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX

          if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
              kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2) then

            stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 4 because of AVX-512 memory alignment of 64 bytes
                                            ! (4 * sizeof(double complex) == 64)

    else

            stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 2 because of AVX/SSE memory alignment of 32 bytes
                                            ! (2 * sizeof(double complex) == 32)
    endif
#else

          if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
              kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2) then

            stripe_width = ((stripe_width+15)/16)*16 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                            ! (8 * sizeof(float complex) == 64)

          else
            stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                            ! (4 * sizeof(float complex) == 32)
    endif
#endif
#endif /* COMPLEXCASE */
        endif ! useGPU

        last_stripe_width = l_nev - (stripe_count-1)*stripe_width

#endif /* WITH_OPENMP */
      endif ! l_nev

      ! Determine the matrix distribution at the beginning

      allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_tridi_to_band_&
                &MATH_DATATYPE&
                &: error when allocating limits"//errorMessage
        stop 1
      endif
      call determine_workload(obj,na, nbw, np_rows, limits)

      max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

      a_dim2 = max_blk_size + nbw

      if (useGPU) then
        print *,"trans_ev_tridi_to_band_&
                &MATH_DATATYPE&
                &: error when allocating limits"
        stop 1

      else ! GPUs are not used

#if 0
! realcase or complexcase
!DEC$ ATTRIBUTES ALIGN: 64:: aIntern
#endif

#ifdef WITH_OPENMP
        if (posix_memalign(aIntern_ptr, 64_c_intptr_t, stripe_width*a_dim2*stripe_count*max_threads*     &
               C_SIZEOF(a_var)) /= 0) then
          print *,"trans_ev_tridi_to_band_&
          &MATH_DATATYPE&
          &: error when allocating aIntern"//errorMessage
          stop 1
        endif

        call c_f_pointer(aIntern_ptr, aIntern, [stripe_width,a_dim2,stripe_count,max_threads])
        ! allocate(aIntern(stripe_width,a_dim2,stripe_count,max_threads), stat=istat, errmsg=errorMessage)

        ! aIntern(:,:,:,:) should be set to 0 in a parallel region, not here!

#else /* WITH_OPENMP */

        if (posix_memalign(aIntern_ptr, 64_c_intptr_t, stripe_width*a_dim2*stripe_count*  &
            C_SIZEOF(a_var)) /= 0) then
          print *,"trans_ev_tridi_to_band_real: error when allocating aIntern"//errorMessage
          stop 1
        endif

        call c_f_pointer(aIntern_ptr, aIntern,[stripe_width,a_dim2,stripe_count] )
        !allocate(aIntern(stripe_width,a_dim2,stripe_count), stat=istat, errmsg=errorMessage)

        aIntern(:,:,:) = 0.0_rck
#endif /* WITH_OPENMP */
      endif !useGPU

      allocate(row(l_nev), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_tridi_to_band_&
        &MATH_DATATYPE&
        &: error when allocating row"//errorMessage
        stop 1
      endif

      row(:) = 0.0_rck

      ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
      ! and transpose the matrix using stripes of given stripe_width for cache blocking.

      ! The peculiar way it is done below is due to the fact that the last row should be
      ! ready first since it is the first one to start below

#ifdef WITH_OPENMP
      ! Please note about the OMP usage below:
      ! This is not for speed, but because we want the matrix a in the memory and
      ! in the cache of the correct thread (if possible)

      !$omp parallel do private(my_thread), schedule(static, 1)
      do my_thread = 1, max_threads
        aIntern(:,:,:,my_thread) = 0.0_rck ! if possible, do first touch allocation!
      enddo
      !$omp end parallel do

#endif /* WITH_OPENMP */

      do ip = np_rows-1, 0, -1
        if (my_prow == ip) then
          ! Receive my rows which have not yet been received
          src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
          do i=limits(ip)+1,limits(ip+1)
            src = mod((i-1)/nblk, np_rows)

            if (src < my_prow) then
#ifdef WITH_OPENMP

#ifdef WITH_MPI
              call MPI_Recv(row, l_nev, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)

#else /* WITH_MPI */

!              row(1:l_nev) = row(1:l_nev)

#endif /* WITH_MPI */


!$omp parallel do private(my_thread), schedule(static, 1)
              do my_thread = 1, max_threads
                call unpack_row_&
                     &MATH_DATATYPE&
                     &_cpu_openmp_&
                     &PRECISION &
                                  (obj,aIntern, row, i-limits(ip), my_thread, stripe_count, &
                                   thread_width, stripe_width, l_nev)

              enddo
!$omp end parallel do


#else /* WITH_OPENMP */
              if (useGPU) then
                 print *,"trans_ev_tridi_to_band_&
                      &MATH_DATATYPE&
                      &: error when allocating limits"
                 stop 1

              else ! useGPU
#ifdef WITH_MPI
                call MPI_Recv(row, l_nev, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                              src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)

#else /* WITH_MPI */

!                row(1:l_nev) = row(1:l_nev)

#endif /* WITH_MPI */

                call unpack_row_&
                     &MATH_DATATYPE&
                     &_cpu_&
                     &PRECISION &
                                (obj,aIntern, row,i-limits(ip), stripe_count, stripe_width, last_stripe_width)
!                 print *,"trans_ev_tridi_to_band_&
!                      &MATH_DATATYPE&
!                      &: non openmp not supported"
!                 stop 1
                
              endif ! useGPU
#endif /* WITH_OPENMP */

            elseif (src == my_prow) then

              src_offset = src_offset+1

              if (useGPU) then
                 print *,"trans_ev_tridi_to_band_&
                      &MATH_DATATYPE&
                      &: error when allocating limits"
                 stop 1

              else
                row(:) = q(src_offset, 1:l_nev)
              endif

#ifdef WITH_OPENMP

!$omp parallel do private(my_thread), schedule(static, 1)
              do my_thread = 1, max_threads
                call unpack_row_&
                     &MATH_DATATYPE&
                     &_cpu_openmp_&
                     &PRECISION &
                                   (obj,aIntern, row, i-limits(ip), my_thread, stripe_count, thread_width, stripe_width, l_nev)

              enddo
!$omp end parallel do


#else /* WITH_OPENMP */

              if (useGPU) then
                 print *,"trans_ev_tridi_to_band_&
                      &MATH_DATATYPE&
                      &: error when allocating limits"
                 stop 1

              else
                call unpack_row_&
                     &MATH_DATATYPE&
                     &_cpu_&
                     &PRECISION &
                                (obj,aIntern, row,i-limits(ip),  stripe_count, stripe_width, last_stripe_width)
!                 print *,"trans_ev_tridi_to_band_&
!                      &MATH_DATATYPE&
!                      &: non openmp not supported"
!                 stop 1
              endif

#endif /* WITH_OPENMP */

            endif
          enddo

          ! Send all rows which have not yet been send
          src_offset = 0
          do dst = 0, ip-1
            do i=limits(dst)+1,limits(dst+1)
              if (mod((i-1)/nblk, np_rows) == my_prow) then
                src_offset = src_offset+1
                row(:) = q(src_offset, 1:l_nev)

#ifdef WITH_MPI
                call MPI_Send(row, l_nev, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                              dst, 0, mpi_comm_rows, mpierr)
#endif /* WITH_MPI */
              endif
            enddo
          enddo

        else if (my_prow < ip) then

          ! Send all rows going to PE ip
          src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
          do i=limits(ip)+1,limits(ip+1)
            src = mod((i-1)/nblk, np_rows)
            if (src == my_prow) then
              src_offset = src_offset+1
              row(:) = q(src_offset, 1:l_nev)
#ifdef WITH_MPI
              call MPI_Send(row, l_nev, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            ip, 0, mpi_comm_rows, mpierr)
#endif /* WITH_MPI */
            endif
          enddo

          ! Receive all rows from PE ip
          do i=limits(my_prow)+1,limits(my_prow+1)
            src = mod((i-1)/nblk, np_rows)
            if (src == ip) then
#ifdef WITH_OPENMP

#ifdef WITH_MPI
              call MPI_Recv(row, l_nev, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */

!              row(1:l_nev) = row(1:l_nev)

#endif /* WITH_MPI */

!$omp parallel do private(my_thread), schedule(static, 1)
              do my_thread = 1, max_threads
                call unpack_row_&
                     &MATH_DATATYPE&
                     &_cpu_openmp_&
                     &PRECISION &
                                 (obj,aIntern, row, i-limits(my_prow), my_thread, stripe_count, thread_width, stripe_width, l_nev)
              enddo
!$omp end parallel do

#else /* WITH_OPENMP */
              if (useGPU) then
                 print *,"trans_ev_tridi_to_band_&
                      &MATH_DATATYPE&
                      &: error when allocating limits"
                 stop 1

              else ! useGPU
#ifdef WITH_MPI
                call MPI_Recv(row, l_nev, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                              src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */

!                row(1:l_nev) = row(1:l_nev)

#endif
                call unpack_row_&
                     &MATH_DATATYPE&
                     &_cpu_&
                     &PRECISION &
                                (obj,aIntern, row,i-limits(my_prow), stripe_count, stripe_width, last_stripe_width)
!                 print *,"trans_ev_tridi_to_band_&
!                      &MATH_DATATYPE&
!                      &: non openmp not supported"
!                 stop 1
              endif ! useGPU

#endif /* WITH_OPENMP */

            endif
          enddo
        endif
      enddo


      ! Set up result buffer queue

      num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

      num_result_buffers = 4*nfact
      allocate(result_buffer(l_nev,nblk,num_result_buffers), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_tridi_to_band_&
        &MATH_DATATYPE&
        &: error when allocating result_buffer"//errorMessage
        stop 1
      endif

      allocate(result_send_request(num_result_buffers), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_tridi_to_band_&
        &MATH_DATATYPE&
        &: error when allocating result_send_request"//errorMessage
        stop 1
      endif

      allocate(result_recv_request(num_result_buffers), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_tridi_to_band_&
        &MATH_DATATYPE&
        &: error when allocating result_recv_request"//errorMessage
        stop 1
      endif

#ifdef WITH_MPI
      result_send_request(:) = MPI_REQUEST_NULL
      result_recv_request(:) = MPI_REQUEST_NULL
#endif

      ! Queue up buffers
#ifdef WITH_MPI

      if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
        do j = 1, min(num_result_buffers, num_result_blocks)
          call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_MATH_DATATYPE_PRECISION_EXPL,     &
                         0, result_recv_tag, mpi_comm_rows, result_recv_request(j), mpierr)
        enddo
      endif
#else /* WITH_MPI */

      ! carefull the "recv" has to be done at the corresponding wait or send
      ! result_buffer(1: l_nev*nblk,1,j) =result_buffer(1:l_nev*nblk,1,nbuf)

#endif /* WITH_MPI */

      num_bufs_recvd = 0 ! No buffers received yet

      ! Initialize top/bottom requests

      allocate(top_send_request(stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MPI_DATATYPE&
         &: error when allocating top_send_request"//errorMessage
         stop 1
       endif

      allocate(top_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating top_recv_request"//errorMessage
         stop 1
       endif

      allocate(bottom_send_request(stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bottom_send_request"//errorMessage
         stop 1
       endif

      allocate(bottom_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bottom_recv_request"//errorMessage
         stop 1
       endif

#ifdef WITH_MPI
      top_send_request(:) = MPI_REQUEST_NULL
      top_recv_request(:) = MPI_REQUEST_NULL
      bottom_send_request(:) = MPI_REQUEST_NULL
      bottom_recv_request(:) = MPI_REQUEST_NULL
#endif

#ifdef WITH_OPENMP
      allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating top_border_send_buffer"//errorMessage
         stop 1
       endif

      allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating top_border_recv_buffer"//errorMessage
         stop 1
       endif

      allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bottom_border_send_buffer"//errorMessage
         stop 1
       endif

      allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bottom_border_recv_buffer"//errorMessage
         stop 1
       endif

      top_border_send_buffer(:,:) = 0.0_rck
      top_border_recv_buffer(:,:) = 0.0_rck
      bottom_border_send_buffer(:,:) = 0.0_rck
      bottom_border_recv_buffer(:,:) = 0.0_rck
      ! Initialize broadcast buffer

#else /* WITH_OPENMP */

       allocate(top_border_send_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating top_border_send_bufer"//errorMessage
         stop 1
       endif

      allocate(top_border_recv_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating top_border_recv_buffer"//errorMessage
         stop 1
       endif

      allocate(bottom_border_send_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bottom_border_send_buffer"//errorMessage
         stop 1
       endif

      allocate(bottom_border_recv_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bottom_border_recv_buffer"//errorMessage
         stop 1
       endif

      top_border_send_buffer(:,:,:) = 0.0_rck
      top_border_recv_buffer(:,:,:) = 0.0_rck
      bottom_border_send_buffer(:,:,:) = 0.0_rck
      bottom_border_recv_buffer(:,:,:) = 0.0_rck
#endif /* WITH_OPENMP */

      ! Initialize broadcast buffer

      allocate(bcast_buffer(nbw, max_blk_size), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_&
         &MATH_DATATYPE&
         &: error when allocating bcast_buffer"//errorMessage
         stop 1
       endif

      bcast_buffer = 0.0_rck

      current_tv_off = 0 ! Offset of next row to be broadcast

      ! ------------------- start of work loop -------------------

      a_off = 0 ! offset in aIntern (to avoid unnecessary shifts)

      top_msg_length = 0
      bottom_msg_length = 0

      do sweep = 0, (na-1)/nbw

        current_n = na - sweep*nbw
        call determine_workload(obj,current_n, nbw, np_rows, limits)
        current_n_start = limits(my_prow)
        current_n_end   = limits(my_prow+1)
        current_local_n = current_n_end - current_n_start

        next_n = max(current_n - nbw, 0)
        call determine_workload(obj,next_n, nbw, np_rows, limits)
        next_n_start = limits(my_prow)
        next_n_end   = limits(my_prow+1)
        next_local_n = next_n_end - next_n_start

        if (next_n_end < next_n) then
          bottom_msg_length = current_n_end - next_n_end
        else
          bottom_msg_length = 0
        endif

        if (next_local_n > 0) then
          next_top_msg_length = current_n_start - next_n_start
        else
          next_top_msg_length = 0
        endif

        if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
#ifdef WITH_MPI
#endif
          do i = 1, stripe_count

#ifdef WITH_OPENMP


            csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
            b_len = csw*nbw*max_threads
#ifdef WITH_MPI
            call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           my_prow+1, bottom_recv_tag, mpi_comm_rows, bottom_recv_request(i), mpierr)

#else /* WITH_MPI */
!            carefull the "recieve" has to be done at the corresponding wait or send
!            bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
#endif /* WITH_MPI */

#else /* WITH_OPENMP */

#ifdef WITH_MPI
            call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           my_prow+1, bottom_recv_tag, mpi_comm_rows, bottom_recv_request(i), mpierr)
#else  /* WITH_MPI */
!            carefull the recieve has to be done at the corresponding wait or send
!            bottom_border_recv_buffer(1:nbw*stripe_width,1,i) = top_border_send_buffer(1:nbw*stripe_width,1,i)
#endif /* WITH_MPI */

#endif /* WITH_OPENMP */

          enddo
#if WITH_MPI
#endif
        endif

        if (current_local_n > 1) then
          if (my_pcol == mod(sweep,np_cols)) then
            bcast_buffer(:,1:current_local_n) =    &
                  hh_trans(:,current_tv_off+1:current_tv_off+current_local_n)
            current_tv_off = current_tv_off + current_local_n
          endif

#ifdef WITH_MPI
           call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          mod(sweep,np_cols), mpi_comm_cols, mpierr)

#endif /* WITH_MPI */


        else ! (current_local_n > 1) then

          ! for current_local_n == 1 the one and only HH Vector is 0 and not stored in hh_trans_real/complex
          bcast_buffer(:,1) = 0.0_rck
        endif ! (current_local_n > 1) then

        if (l_nev == 0) cycle

        if (current_local_n > 0) then

          do i = 1, stripe_count
#ifdef WITH_OPENMP
            if (useGPU) then
              print *,"trans_ev_tridi_to_band_real: not yet implemented"
              stop 1
            endif

            ! Get real stripe width for strip i;
            ! The last OpenMP tasks may have an even smaller stripe with,
            ! but we don't care about this, i.e. we send/recv a bit too much in this case.
            ! csw: current_stripe_width

            csw = min(stripe_width, thread_width-(i-1)*stripe_width)
#endif /* WITH_OPENMP */

            !wait_b
            if (current_n_end < current_n) then


#ifdef WITH_OPENMP
              if (useGPU) then
                print *,"trans_ev_tridi_to_band_real: not yet implemented"
                stop 1
              endif

#ifdef WITH_MPI

              call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
              do my_thread = 1, max_threads
                n_off = current_local_n+a_off
                b_len = csw*nbw
                b_off = (my_thread-1)*b_len
                aIntern(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
                  reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
              enddo
!$omp end parallel do
#else /* WITH_OPENMP */

#ifdef WITH_MPI
              call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#endif
              n_off = current_local_n+a_off

              if (useGPU) then
                if (.not.(successCUDA)) then
                  print *,"trans_ev_tridi_to_band_&
                  &MATH_DATATYPE&
                  &: error in cudaMemcpy"
                  stop 1
                endif

              else
                aIntern(:,n_off+1:n_off+nbw,i) = bottom_border_recv_buffer(:,1:nbw,i)
              endif

#endif /* WITH_OPENMP */

           if (next_n_end < next_n) then

#ifdef WITH_OPENMP

             if (useGPU) then
               print *,"trans_ev_tridi_to_band_real: not yet implemented"
               stop 1
             endif
#ifdef WITH_MPI
             call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            my_prow+1, bottom_recv_tag, mpi_comm_rows, bottom_recv_request(i), mpierr)

#else /* WTIH_MPI */
!                carefull the recieve has to be done at the corresponding wait or send
!                bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)

#endif /* WITH_MPI */

#else /* WITH_OPENMP */

#ifdef WITH_MPI
             call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            my_prow+1, bottom_recv_tag, mpi_comm_rows, bottom_recv_request(i), mpierr)

#else /* WITH_MPI */

!!                carefull the recieve has to be done at the corresponding wait or send
!!                bottom_border_recv_buffer(1:stripe_width,1:nbw,i) =  top_border_send_buffer(1:stripe_width,1:nbw,i)

#endif /* WITH_MPI */

#endif /* WITH_OPENMP */
           endif
         endif

         if (current_local_n <= bottom_msg_length + top_msg_length) then

           !wait_t
           if (top_msg_length>0) then

#ifdef WITH_OPENMP
             if (useGPU) then
               print *,"trans_ev_tridi_to_band_&
                       &MATH_DATATYPE&
                       &: not yet implemented"
               stop 1
             endif
#ifdef WITH_MPI

             call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif

#else /* WITH_OPENMP */

#ifdef WITH_MPI
             call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif

             if (useGPU) then
               if (.not.(successCUDA)) then
                 print *,"trans_ev_tridi_to_band_&
                         &MATH_DATATYPE&
                         &: error in cudaMemcpy"
                 stop 1
                endif
             else ! useGPU
               aIntern(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
             endif ! useGPU
#endif /* WITH_OPENMP */
           endif ! top_msg_length

           !compute
#ifdef WITH_OPENMP


!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
           do my_thread = 1, max_threads
             if (top_msg_length>0) then
               b_len = csw*top_msg_length
               b_off = (my_thread-1)*b_len
               aIntern(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
             endif

       call compute_hh_trafo_&
            &MATH_DATATYPE&
            &_openmp_&
            &PRECISION &
                              (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, max_threads, &
             l_nev, a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
#if REALCASE == 1
                               hh_dot_dev, &
#endif
                               hh_tau_dev, kernel_flops, kernel_time, n_times, 0, current_local_n, &
             i, my_thread, thread_width, kernel)
           enddo
!$omp end parallel do

#else /* WITH_OPENMP */

!                 print *,"trans_ev_tridi_to_band_&
!                      &MATH_DATATYPE&
!                      &: non openmp not supported"
!                 stop 1
           call compute_hh_trafo_&
                &MATH_DATATYPE&
                &_&
                &PRECISION&
     & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count,  max_threads,      &
              a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev,      &
#if REALCASE == 1
              hh_dot_dev, &
#endif
              hh_tau_dev, kernel_flops, kernel_time, n_times, 0, current_local_n, i,          &
              last_stripe_width, kernel)

#endif /* WITH_OPENMP */

           !send_b        1
#ifdef WITH_MPI
           call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif

           if (bottom_msg_length>0) then
             n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_OPENMP
             b_len = csw*bottom_msg_length*max_threads
             bottom_border_send_buffer(1:b_len,i) = &
                 reshape(aIntern(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
#ifdef WITH_MPI
             call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            my_prow+1, top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#else /* WITH_MPI */
             if (next_top_msg_length > 0) then
               top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = bottom_border_send_buffer(1:csw* &
                                            next_top_msg_length*max_threads,i)
             endif

#endif /* WITH_MPI */

!#if REALCASE == 1
           endif ! this endif is not here in complex -case is for bottom_msg_length
!#endif

#else /* WITH_OPENMP */

             if (useGPU) then
               if (.not.(successCUDA)) then
                 print *,"trans_ev_tridi_to_band_&
                 &MATH_DATATYPE&
                 &: error in cudaMemcpy"
                 stop 1
               endif
             else
               bottom_border_send_buffer(:,1:bottom_msg_length,i) = aIntern(:,n_off+1:n_off+bottom_msg_length,i)
             endif
#ifdef WITH_MPI
             call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width,  &
                   MPI_MATH_DATATYPE_PRECISION_EXPL, my_prow+1, top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)

#else /* WITH_MPI */
                if (next_top_msg_length > 0) then
                  top_border_recv_buffer(1:stripe_width,1:next_top_msg_length,i) =  &
                  bottom_border_send_buffer(1:stripe_width,1:next_top_msg_length,i)
                endif

#endif /* WITH_MPI */
           endif
#endif /* WITH_OPENMP */

         else ! current_local_n <= bottom_msg_length + top_msg_length

         !compute
#ifdef WITH_OPENMP
         if (useGPU) then
           print *,"trans_ev_tridi_to_band_real: not yet implemented"
           stop 1
         endif

        !$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
        do my_thread = 1, max_threads

          call compute_hh_trafo_&
               &MATH_DATATYPE&
               &_openmp_&
               &PRECISION&
               & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, max_threads, l_nev, a_off, &
                 nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev, &
#if REALCASE == 1
            hh_dot_dev,  &
#endif
            hh_tau_dev, kernel_flops, kernel_time, n_times, current_local_n - bottom_msg_length, &
      bottom_msg_length, i, my_thread, thread_width, kernel)
        enddo
!$omp end parallel do

        !send_b
#ifdef WITH_MPI
        call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
        if (bottom_msg_length > 0) then
          n_off = current_local_n+nbw-bottom_msg_length+a_off
          b_len = csw*bottom_msg_length*max_threads
          bottom_border_send_buffer(1:b_len,i) = &
              reshape(aIntern(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
#ifdef WITH_MPI
          call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         my_prow+1, top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)

#else /* WITH_MPI */
          if (next_top_msg_length > 0) then
            top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = bottom_border_send_buffer(1:csw* &
                                                                                                     next_top_msg_length*&
                                                          max_threads,i)
          endif
#endif /* WITH_MPI */
        endif

#else /* WITH_OPENMP */

!           print *,"trans_ev_tridi_to_band_real: non openmp not yet supported"
!           stop 1

        call compute_hh_trafo_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count,  max_threads,      &
                      a_off,  nbw, max_blk_size, bcast_buffer, bcast_buffer_dev,      &
#if REALCASE == 1
            hh_dot_dev, &
#endif
           hh_tau_dev, kernel_flops, kernel_time, n_times,                 &
           current_local_n - bottom_msg_length, bottom_msg_length, i,             &
           last_stripe_width, kernel)



        !send_b
#ifdef WITH_MPI

        call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
        if (bottom_msg_length > 0) then
          n_off = current_local_n+nbw-bottom_msg_length+a_off

          if (useGPU) then
            if (.not.(successCUDA)) then
              print *,"trans_ev_tridi_to_band_&
              &MATH_DATATYPE&
              &: error cudaMemcpy"
              stop 1
            endif
          else
            bottom_border_send_buffer(:,1:bottom_msg_length,i) = aIntern(:,n_off+1:n_off+bottom_msg_length,i)
          endif

#ifdef WITH_MPI
          call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, &
           MPI_MATH_DATATYPE_PRECISION_EXPL, my_prow+1, top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#else /* WITH_MPI */
                if (next_top_msg_length > 0) then
                  top_border_recv_buffer(1:stripe_width,1:next_top_msg_length,i) =  &
                  bottom_border_send_buffer(1:stripe_width,1:next_top_msg_length,i)
                endif

#endif /* WITH_MPI */

#if REALCASE == 1
        endif
#endif

#endif /* WITH_OPENMP */

#ifndef WITH_OPENMP
#if COMPLEXCASE == 1
        endif
#endif
#endif
        !compute
#ifdef WITH_OPENMP


!$omp parallel do private(my_thread), schedule(static, 1)
        do my_thread = 1, max_threads
          call compute_hh_trafo_&
          &MATH_DATATYPE&
          &_openmp_&
          &PRECISION&
          & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width ,a_dim2, stripe_count, max_threads, l_nev, a_off, &
             nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
#if REALCASE == 1
             hh_dot_dev, &
#endif
             hh_tau_dev, kernel_flops, kernel_time, n_times, top_msg_length,&
             current_local_n-top_msg_length-bottom_msg_length, i, my_thread, thread_width, &
             kernel)
        enddo
!$omp end parallel do

#else /* WITH_OPENMP */

!           print *,"trans_ev_tridi_to_band_real: non openmp not yet supported"
!           stop 1

        call compute_hh_trafo_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count,  max_threads,          &
                      a_off,  nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
#if REALCASE == 1
            hh_dot_dev,     &
#endif
           hh_tau_dev, kernel_flops, kernel_time, n_times, top_msg_length,                    &
           current_local_n-top_msg_length-bottom_msg_length, i,                       &
           last_stripe_width, kernel)

#endif /* WITH_OPENMP */

        !wait_t
        if (top_msg_length>0) then
#ifdef WITH_OPENMP

#ifdef WITH_MPI
          call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif

#else /* WITH_OPENMP */

#ifdef WITH_MPI
          call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
          if (useGPU) then
            if (.not.(successCUDA)) then
              print *,"trans_ev_tridi_to_band_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy"
              stop 1
            endif
          else
            aIntern(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
          endif
#endif /* WITH_OPENMP */
        endif

        !compute
#ifdef WITH_OPENMP


!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
        do my_thread = 1, max_threads
          if (top_msg_length>0) then
            b_len = csw*top_msg_length
            b_off = (my_thread-1)*b_len
            aIntern(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
              reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
          endif
          call compute_hh_trafo_&
               &MATH_DATATYPE&
               &_openmp_&
               &PRECISION&
               & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, max_threads, l_nev, a_off, &
                  nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev, &
#if REALCASE == 1
             hh_dot_dev, &
#endif
             hh_tau_dev, kernel_flops, kernel_time, n_times, 0, top_msg_length, i, my_thread, &
       thread_width, kernel)
        enddo
!$omp end parallel do

#else /* WITH_OPENMP */

!           print *,"trans_ev_tridi_to_band_real: non openmp not yet supported"
!           stop 1

        call compute_hh_trafo_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             & (obj, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count,  max_threads,          &
                      a_off, nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev,          &
#if REALCASE == 1
               hh_dot_dev,     &
#endif
           hh_tau_dev, kernel_flops, kernel_time, n_times, 0, top_msg_length, i,               &
           last_stripe_width, kernel)

#endif /* WITH_OPENMP */
      endif

      if (next_top_msg_length > 0) then
        !request top_border data
#ifdef WITH_OPENMP

        b_len = csw*next_top_msg_length*max_threads
#ifdef WITH_MPI
        call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_MATH_DATATYPE_PRECISION_EXPL, &
           my_prow-1, top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#else /* WITH_MPI */
!             carefull the "recieve" has to be done at the corresponding wait or send
!              top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = &
!                                     bottom_border_send_buffer(1:csw*next_top_msg_length*max_threads,i)
#endif /* WITH_MPI */

#else /* WITH_OPENMP */

#ifdef WITH_MPI
        call MPI_Irecv(top_border_recv_buffer(1,1,i), next_top_msg_length*stripe_width, &
              MPI_MATH_DATATYPE_PRECISION_EXPL, my_prow-1, top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#else /* WITH_MPI */
!             carefull the "recieve" has to be done at the corresponding wait or send
!              top_border_recv_buffer(1:stripe_width,1:next_top_msg_length,i) =  &
!               bottom_border_send_buffer(1:stripe_width,1:next_top_msg_length,i)
#endif /* WITH_MPI */

#endif /* WITH_OPENMP */

      endif

      !send_t
      if (my_prow > 0) then
#ifdef WITH_OPENMP

#ifdef WITH_MPI
        call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
        b_len = csw*nbw*max_threads
        top_border_send_buffer(1:b_len,i) = reshape(aIntern(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))

#ifdef WITH_MPI
        call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       my_prow-1, bottom_recv_tag, mpi_comm_rows, top_send_request(i), mpierr)
#else /* WITH_MPI */
              if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
                bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
              endif
              if (next_n_end < next_n) then
                bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
              endif
#endif /* WITH_MPI */

#else /* WITH_OPENMP */

#ifdef WITH_MPI
        call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
        if (useGPU) then
          if (.not.(successCUDA)) then
            print *,"trans_ev_tridi_to_band_&
                    &MATH_DATATYPE&
                    &: error in cudaMemcpy"
            stop 1
          endif

        else
          top_border_send_buffer(:,1:nbw,i) = aIntern(:,a_off+1:a_off+nbw,i)
        endif
#ifdef WITH_MPI
        call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       my_prow-1, bottom_recv_tag, mpi_comm_rows, top_send_request(i), mpierr)
#else /* WITH_MPI */
            if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
               bottom_border_recv_buffer(1:nbw*stripe_width,1,i) = top_border_send_buffer(1:nbw*stripe_width,1,i)
             endif
             if (next_n_end < next_n) then
               bottom_border_recv_buffer(1:stripe_width,1:nbw,i) =  top_border_send_buffer(1:stripe_width,1:nbw,i)
             endif
#endif /* WITH_MPI */

#endif /* WITH_OPENMP */
      endif

      ! Care that there are not too many outstanding top_recv_request's
      if (stripe_count > 1) then
        if (i>1) then

#ifdef WITH_MPI
          call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
#endif
        else

#ifdef WITH_MPI
          call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
#endif

        endif
      endif

    enddo

    top_msg_length = next_top_msg_length

  else
    ! wait for last top_send_request

#ifdef WITH_MPI
    do i = 1, stripe_count
      call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
    enddo
#endif
  endif

    ! Care about the result

    if (my_prow == 0) then

      ! topmost process sends nbw rows to destination processes

      do j=0, nfact-1
        num_blk = sweep*nfact+j ! global number of destination block, 0 based
        if (num_blk*nblk >= na) exit

        nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

#ifdef WITH_MPI
        call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)

#endif
        dst = mod(num_blk, np_rows)

        if (dst == 0) then
          if (useGPU) then
             print *,"trans_ev_tridi_to_band_&
                  &MATH_DATATYPE&
                  &: error in cudaMemcpy"
             stop 1

          else ! useGPU

            do i = 1, min(na - num_blk*nblk, nblk)
#ifdef WITH_OPENMP
              call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION&
                   &(obj,aIntern, row, j*nblk+i+a_off, stripe_width, stripe_count, max_threads, thread_width, l_nev)
#else /* WITH_OPENMP */

              call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_&
                   &PRECISION&
                   &(obj,aIntern, row, j*nblk+i+a_off, stripe_width, last_stripe_width, stripe_count)

!              call pack_row_MATH_DATATYPE_cpu_PRECISION(obj,aIntern, row, j*nblk+i+a_off, stripe_width, last_stripe_width, stripe_count)
!                 print *,"trans_ev_tridi_to_band_&
!                      &MATH_DATATYPE&
!                      &: non openmp not supported"
!                 stop 1

#endif /* WITH_OPENMP */
              q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
            enddo
          endif ! useGPU

        else ! (dst == 0)

          if (useGPU) then
             print *,"trans_ev_tridi_to_band_&
                  &MATH_DATATYPE&
                  &: error in cudaMemcpy"
             stop 1

          else  ! useGPU
            do i = 1, nblk
#if WITH_OPENMP
              call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION&
                   &(obj,aIntern, result_buffer(:,i,nbuf), j*nblk+i+a_off, stripe_width, stripe_count, &
                   max_threads, thread_width, l_nev)
#else /* WITH_OPENMP */

              call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_&
                   &PRECISION&
                   &(obj, aIntern, result_buffer(:,i,nbuf),j*nblk+i+a_off, stripe_width, last_stripe_width, stripe_count)

!              call pack_row_MATH_DATATYPE_cpu_PRECISION(obj, aIntern, result_buffer(:,i,nbuf),j*nblk+i+a_off, stripe_width, last_stripe_width, stripe_count)
!                 print *,"trans_ev_tridi_to_band_&
!                      &MATH_DATATYPE&
!                      &: non openmp not supported"
!                 stop 1


#endif /* WITH_OPENMP */
            enddo
          endif ! useGPU
#ifdef WITH_MPI
          call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          dst, result_recv_tag, mpi_comm_rows, result_send_request(nbuf), mpierr)

#else /* WITH_MPI */
          if (j+num_result_buffers < num_result_blocks) &
                   result_buffer(1:l_nev,1:nblk,nbuf) = result_buffer(1:l_nev,1:nblk,nbuf)
          if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
            do j1 = 1, min(num_result_buffers, num_result_blocks)
              result_buffer(1:l_nev,1:nblk,j1) = result_buffer(1:l_nev,1:nblk,nbuf)
            enddo
          endif

#endif /* WITH_MPI */
        endif ! (dst == 0)
      enddo  !j=0, nfact-1

    else ! (my_prow == 0)

      ! receive and store final result

      do j = num_bufs_recvd, num_result_blocks-1

        nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

        ! If there is still work to do, just test for the next result request
        ! and leave the loop if it is not ready, otherwise wait for all
        ! outstanding requests

        if (next_local_n > 0) then

#ifdef WITH_MPI
          call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)

#else /* WITH_MPI */
          flag = .true.
#endif

          if (.not.flag) exit

        else ! (next_local_n > 0)
#ifdef WITH_MPI
          call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
        endif ! (next_local_n > 0)

        ! Fill result buffer into q
        num_blk = j*np_rows + my_prow ! global number of current block, 0 based
        do i = 1, min(na - num_blk*nblk, nblk)
          q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
        enddo

        ! Queue result buffer again if there are outstanding blocks left
#ifdef WITH_MPI

        if (j+num_result_buffers < num_result_blocks) &
            call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           0, result_recv_tag, mpi_comm_rows, result_recv_request(nbuf), mpierr)

        ! carefull the "recieve" has to be done at the corresponding wait or send
!         if (j+num_result_buffers < num_result_blocks) &
!                result_buffer(1:l_nev*nblk,1,nbuf) =  result_buffer(1:l_nev*nblk,1,nbuf)

#else /* WITH_MPI */

#endif /* WITH_MPI */

      enddo ! j = num_bufs_recvd, num_result_blocks-1
      num_bufs_recvd = j

    endif ! (my_prow == 0)

    ! Shift the remaining rows to the front of aIntern (if necessary)

    offset = nbw - top_msg_length
    if (offset<0) then
      if (wantDebug) write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_&
                                         &MATH_DATATYPE&
                                         &: internal error, offset for shifting = ',offset
      success = .false.
      return
    endif

    a_off = a_off + offset
    if (a_off + next_local_n + nbw > a_dim2) then
#ifdef WITH_OPENMP
      if (useGPU) then
        print *,"trans_ev_tridi_to_band_real: not yet implemented"
        stop 1
      endif


!$omp parallel do private(my_thread, i, j), schedule(static, 1)
      do my_thread = 1, max_threads
        do i = 1, stripe_count
          do j = top_msg_length+1, top_msg_length+next_local_n
            aIntern(:,j,i,my_thread) = aIntern(:,j+a_off,i,my_thread)
          enddo
        enddo
      enddo
!$omp end parallel do

#else /* WITH_OPENMP */
         do i = 1, stripe_count
           if (useGPU) then
              print *,"trans_ev_tridi_to_band_real: not yet implemented"
              stop 1
           else ! not useGPU
             do j = top_msg_length+1, top_msg_length+next_local_n
               aIntern(:,j,i) = aIntern(:,j+a_off,i)
             enddo
           endif
         enddo ! stripe_count
#endif /* WITH_OPENMP */

         a_off = 0
       endif
     enddo

     ! Just for safety:
#ifdef WITH_MPI
     if (ANY(top_send_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_send_request ***',my_prow,my_pcol
     if (ANY(bottom_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_send_request ***',my_prow,my_pcol
     if (ANY(top_recv_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_recv_request ***',my_prow,my_pcol
     if (ANY(bottom_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_recv_request ***',my_prow,my_pcol
#endif

     if (my_prow == 0) then

#ifdef WITH_MPI
       call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
#endif
     endif

#ifdef WITH_MPI
     if (ANY(result_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_send_request ***',my_prow,my_pcol
     if (ANY(result_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_recv_request ***',my_prow,my_pcol


     print_flops = 0
#ifdef HAVE_DETAILED_TIMINGS
     if (print_flops == 1) then
       call MPI_ALLREDUCE(kernel_flops, kernel_flops_recv, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_ROWS, mpierr)
       kernel_flops = kernel_flops_recv
       call MPI_ALLREDUCE(kernel_flops, kernel_flops_recv, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_COLS, mpierr)
       kernel_flops = kernel_flops_recv

       call MPI_ALLREDUCE(kernel_time, kernel_time_recv, 1, MPI_REAL8, MPI_MAX, MPI_COMM_ROWS, mpierr)
       kernel_time_recv = kernel_time
       call MPI_ALLREDUCE(kernel_time, kernel_time_recv, 1, MPI_REAL8, MPI_MAX, MPI_COMM_COLS, mpierr)
       kernel_time_recv = kernel_time
     endif
#endif

#endif /* WITH_MPI */

     if (my_prow==0 .and. my_pcol==0 .and.print_flops == 1) &
         write(error_unit,'(" Kernel time:",f10.3," MFlops: ",es12.5)')  kernel_time, kernel_flops/kernel_time*1.d-6

     if (useGPU) then
        print *,"trans_ev_tridi_to_band_real: not yet implemented"
        stop 1
     endif !use GPU

     ! deallocate all working space

     if (.not.(useGPU)) then
       nullify(aIntern)
       call free(aIntern_ptr)
     endif

     deallocate(row, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating row "//errorMessage
       stop 1
     endif

     deallocate(limits, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating limits"//errorMessage
       stop 1
     endif

     deallocate(result_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating result_send_request "//errorMessage
       stop 1
     endif

     deallocate(result_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating result_recv_request "//errorMessage
       stop 1
     endif

     deallocate(top_border_send_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating top_border_send_buffer "//errorMessage
       stop 1
     endif

     deallocate(top_border_recv_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating top_border_recv_buffer "//errorMessage
       stop 1
     endif

     deallocate(bottom_border_send_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating bottom_border_send_buffer "//errorMessage
       stop 1
     endif

     deallocate(bottom_border_recv_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating bottom_border_recv_buffer "//errorMessage
       stop 1
     endif

     deallocate(result_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
       &MATH_DATATYPE&
       &: error when deallocating result_buffer "//errorMessage
       stop 1
     endif

     deallocate(bcast_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating bcast_buffer "//errorMessage
       stop 1
     endif

     deallocate(top_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating top_send_request "//errorMessage
       stop 1
     endif

     deallocate(top_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating top_recv_request "//errorMessage
       stop 1
     endif

     deallocate(bottom_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating bottom_send_request "//errorMessage
       stop 1
     endif

     deallocate(bottom_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_&
               &MATH_DATATYPE&
               &: error when deallocating bottom_recv_request "//errorMessage
       stop 1
     endif


     return
!#if COMPLEXCASE == 1
!     contains
!     ! The host wrapper for extracting "tau" from the HH reflectors (see the
!     ! kernel below)
!       subroutine extract_hh_tau_complex_gpu_&
!       &PRECISION&
!       &(nbw, n, is_zero)
!         use cuda_c_kernel
!         use pack_unpack_gpu
!         use precision
!         implicit none
!         integer(kind=ik), value :: nbw, n
!         logical, value          :: is_zero
!         integer(kind=ik)        :: val_is_zero
!
!         if (is_zero) then
!           val_is_zero = 1
!         else
!           val_is_zero = 0
!         endif
!         call launch_extract_hh_tau_c_kernel_complex_&
!   &PRECISION&
!   &(bcast_buffer_dev,hh_tau_dev, nbw, n,val_is_zero)
!       end subroutine
!#endif /* COMPLEXCASE */

    end subroutine

! vim: syntax=fortran
