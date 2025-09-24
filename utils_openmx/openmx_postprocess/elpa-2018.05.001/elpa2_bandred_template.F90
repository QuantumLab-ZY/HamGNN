#if 0
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
#endif
    subroutine bandred_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, na, a_mat, a_dev, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, tmat, &
     tmat_dev, wantDebug, useGPU, success, &
#if REALCASE == 1
     useQR, &
#endif
     max_threads)

  !-------------------------------------------------------------------------------
  !  bandred_real/complex: Reduces a distributed symmetric matrix to band form
  !
  !  Parameters
  !
  !  na          Order of matrix
  !
  !  a_mat(lda,matrixCols)    Distributed matrix which should be reduced.
  !              Distribution is like in Scalapack.
  !              Opposed to Scalapack, a_mat(:,:) must be set completely (upper and lower half)
  !              a_mat(:,:) is overwritten on exit with the band and the Householder vectors
  !              in the upper half.
  !
  !  lda         Leading dimension of a_mat
  !  matrixCols  local columns of matrix a_mat
  !
  !  nblk        blocksize of cyclic distribution, must be the same in both directions!
  !
  !  nbw         semi bandwith of output matrix
  !
  !  mpi_comm_rows
  !  mpi_comm_cols
  !              MPI-Communicators for rows/columns
  !
  !  tmat(nbw,nbw,numBlocks)    where numBlocks = (na-1)/nbw + 1
  !              Factors for the Householder vectors (returned), needed for back transformation
  !
  !-------------------------------------------------------------------------------

      use iso_c_binding
      use elpa1_compute_real
#ifdef WITH_OPENMP
      use omp_lib
#endif
      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik)                            :: na, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols

#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck)                    :: a_mat(lda,*), tmat(nbw,nbw,*)
#else
      MATH_DATATYPE(kind=rck)                    :: a_mat(lda,matrixCols), tmat(nbw,nbw,numBlocks)
#endif

#if REALCASE == 1
      real(kind=rk)                               :: eps
#endif
      logical, intent(in)                         :: useGPU
      character(20)                               :: gpuString

      integer(kind=ik)                            :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)                            :: l_cols, l_rows
#if REALCASE == 1
      integer(kind=ik)                            :: vmrCols
#endif
#ifdef WITH_OPENMP
      integer(kind=ik)                            :: mynlc, lrs, transformChunkSize
#endif
      integer(kind=ik)                            :: i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
      integer(kind=ik)                            :: istep, ncol, lch, lcx, nlc
      integer(kind=ik)                            :: tile_size, l_rows_tile, l_cols_tile

      real(kind=rk)                    :: vnorm2
      MATH_DATATYPE(kind=rck)                    :: xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

!      complex(kind=COMPLEX_DATATYPE), allocatable :: tmpCUDA(:,:), vmrCUDA(:,:), umcCUDA(:,:) ! note the different dimension in real case
      MATH_DATATYPE(kind=rck), allocatable :: tmpCUDA(:),  vmrCUDA(:),  umcCUDA(:)
      MATH_DATATYPE(kind=rck), allocatable :: tmpCPU(:,:), vmrCPU(:,:), umcCPU(:,:)
      MATH_DATATYPE(kind=rck), allocatable :: vr(:)

#if REALCASE == 1
      ! needed for blocked QR decomposition
      integer(kind=ik)                            :: PQRPARAM(11), work_size
      real(kind=rk)                    :: dwork_size(1)
      real(kind=rk), allocatable       :: work_blocked(:), tauvector(:), blockheuristic(:)
#endif
      ! a_dev is passed from bandred_real to trans_ev_band
      integer(kind=C_intptr_T)                    :: a_dev, vmr_dev, umc_dev, tmat_dev, vav_dev
#ifdef WITH_MPI
      integer(kind=ik), external                  :: numroc
#endif
      integer(kind=ik)                            :: ierr
      integer(kind=ik)                            :: cur_l_rows, cur_l_cols, vmr_size, umc_size
      integer(kind=c_intptr_t)                    :: lc_start, lc_end
#if COMPLEXCASE == 1
      integer(kind=c_intptr_t)                    :: lce_1, lcs_1, lre_1
#endif
      integer(kind=ik)                            :: lr_end
      integer(kind=ik)                            :: na_cols
#if COMPLEXCASE == 1
      integer(kind=ik)                            :: na_rows
#endif

      logical, intent(in)                         :: wantDebug
      logical, intent(out)                        :: success
      logical                                     :: successCUDA
      integer(kind=ik)                            :: istat
      character(200)                              :: errorMessage
      integer(kind=ik)                            :: min_tile_size, error

#if REALCASE == 1
      logical, intent(in)                         :: useQR
#endif
      integer(kind=ik)                            :: mystart, myend, m_way, n_way, work_per_thread, m_id, n_id, n_threads, &
                                                    ii, pp
      logical                                     :: useGPU_reduction_lower_block_to_tridiagonal
      integer(kind=ik), intent(in)                :: max_threads

!      integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
!                                                                        &PRECISION&
!                                                                        &_&
!                                                                        &MATH_DATATYPE

      if(useGPU) then
        gpuString = "_gpu"
      else
        gpuString = ""
      endif

      useGPU_reduction_lower_block_to_tridiagonal = .false.

      if (useGPU) then
        useGPU_reduction_lower_block_to_tridiagonal = .true.
#if REALCASE == 1
        if (useQR) then
          !in this case switch off GPU usage for step "reduce current block to lower triangular form"
          ! since this is done by QR decomposition
          useGPU_reduction_lower_block_to_tridiagonal = .false.
        endif
#endif
      endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      success = .true.


      ! Semibandwith nbw must be a multiple of blocksize nblk
      if (mod(nbw,nblk)/=0) then
        if (my_prow==0 .and. my_pcol==0) then
          success = .false.
          return
        endif
      endif

      ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

      tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size

      ! make tile_size a smallest possible multiple of previously defined tile size, such that it is
      ! larger or equal to min_tile_size
      ! min_tile_size has been originally hardcoded as 128 * max(np_rows, np_cols), so it is now the implicit value
      ! it can, however, be set by the user
      min_tile_size = 0
      if(min_tile_size == 0) then
        ! not set by the user, use the default value
        min_tile_size = 128*max(np_rows, np_cols)
      endif
      tile_size = ((min_tile_size-1)/tile_size+1)*tile_size

      l_rows_tile = tile_size/np_rows ! local rows of a tile
      l_cols_tile = tile_size/np_cols ! local cols of a tile


      do istep = (na-1)/nbw, 1, -1

        n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

        ! Number of local columns/rows of remaining matrix
        l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
        l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

        ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

        if(1==1) then ! GPU not used

          ! unify the the name vmr and vmrCPU, as well as vmrGPU
          ! the same for umcCPU and umcGPU
          ! Allocate vmr and umcCPU to their exact sizes so that they can be used in bcasts and reduces

          allocate(vmrCPU(max(l_rows,1),2*n_cols), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"bandred_&
                     &MATH_DATATYPE&
                     &: error when allocating vmrCPU "//errorMessage
            stop 1
          endif

          allocate(umcCPU(max(l_cols,1),2*n_cols), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"bandred_&
                    &MATH_DATATYPE&
                    &: error when allocating umcCPU "//errorMessage
            stop 1
          endif

          allocate(vr(l_rows+1), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"bandred_&
                    &MATH_DATATYPE&
                    &: error when allocating vr "//errorMessage
            stop 1
          endif

        endif ! use GPU

        if (useGPU) then
          vmrCUDA(1 : cur_l_rows * n_cols) = 0.0_rck
        else
          vmrCPU(1:l_rows,1:n_cols) = 0.0_rck
        endif ! useGPU

        vr(:) = 0.0_rck
        tmat(:,:,istep) = 0.0_rck

        ! Reduce current block to lower triangular form
#if REALCASE == 1
        if (useQR) then
           print *, "bandred_&
                &MATH_DATATYPE&
                &: qr unsupported "
           stop 1

       else !useQR
#endif /* REALCASE == 1 */
         do lc = n_cols, 1, -1

           ncol = istep*nbw + lc ! absolute column number of householder Vector
           nrow = ncol - nbw ! Absolute number of pivot row

           lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
           lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

           tau = 0

           if (nrow == 1) exit ! Nothing to do

           cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block

           if (my_pcol==cur_pcol) then

             ! Get Vector to be transformed; distribute last element and norm of
             ! remaining elements to all procs in current column

             vr(1:lr) = a_mat(1:lr,lch) ! Vector to be transformed

             if (my_prow==prow(nrow, nblk, np_rows)) then
               aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
               aux1(2) = vr(lr)
             else
               aux1(1) = dot_product(vr(1:lr),vr(1:lr))
               aux1(2) = 0.0_rck
             endif

#ifdef WITH_MPI
             call mpi_allreduce(aux1, aux2, 2, MPI_MATH_DATATYPE_PRECISION, &
                                MPI_SUM, mpi_comm_rows, mpierr)

#else /* WITH_MPI */
              aux2 = aux1 ! this should be optimized
#endif

#if REALCASE == 1
             vnorm2 = aux2(1)
#endif
#if COMPLEXCASE == 1
             vnorm2 = real(aux2(1),kind=rk)
#endif
             vrl    = aux2(2)

             ! Householder transformation
#if REALCASE == 1
       call hh_transform_real_double(obj, vrl, vnorm2, xf, tau, wantDebug)
#endif
#if COMPLEXCASE == 1
       call hh_transform_complex_double(obj, vrl, vnorm2, xf, tau, wantDebug)
#endif

!       call hh_transform_&
!             &MATH_DATATYPE&
!             &_&
!             &PRECISION&
!                         (obj, vrl, vnorm2, xf, tau, wantDebug)

             ! Scale vr and store Householder Vector for back transformation

             vr(1:lr) = vr(1:lr) * xf
             if (my_prow==prow(nrow, nblk, np_rows)) then
               a_mat(1:lr-1,lch) = vr(1:lr-1)
               a_mat(lr,lch) = vrl
               vr(lr) = 1.0_rck
             else
               a_mat(1:lr,lch) = vr(1:lr)
             endif

           endif

           ! Broadcast Householder Vector and tau along columns

           vr(lr+1) = tau
#ifdef WITH_MPI
           call MPI_Bcast(vr, lr+1, MPI_MATH_DATATYPE_PRECISION, &
                          cur_pcol, mpi_comm_cols, mpierr)

#endif /* WITH_MPI */

           if (useGPU_reduction_lower_block_to_tridiagonal) then
             vmrCUDA(cur_l_rows * (lc - 1) + 1 : cur_l_rows * (lc - 1) + lr) = vr(1:lr)
           else
             vmrCPU(1:lr,lc) = vr(1:lr)
           endif
           tau = vr(lr+1)

#if REALCASE == 1
           tmat(lc,lc,istep) = tau ! Store tau in diagonal of tmat
#endif
#if COMPLEXCASE == 1
           tmat(lc,lc,istep) = conjg(tau) ! Store tau in diagonal of tmat
#endif
           ! Transform remaining columns in current block with Householder Vector
           ! Local dot product

           aux1 = 0.0_rck

#ifdef WITH_OPENMP
#if 0
 ! original complex implementation without openmp. check performance
            nlc = 0 ! number of local columns
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0) then
               nlc = nlc+1
               aux1(nlc) = dot_product(vr(1:lr),a_mat(1:lr,lcx))
             endif
           enddo

           ! Get global dot products
#ifdef WITH_MPI
           if (nlc>0) call mpi_allreduce(aux1, aux2, nlc, MPI_COMPLEX_PRECISION, MPI_SUM, mpi_comm_rows, mpierr)

           ! Transform

           nlc = 0
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0) then
               nlc = nlc+1
               a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)

             endif
           enddo



#else /* WITH_MPI */
!          if (nlc>0) aux2=aux1

           ! Transform

           nlc = 0
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0) then
               nlc = nlc+1
               a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux1(nlc)*vr(1:lr)
             endif
           enddo

#endif /* WITH_MPI */
!
!           ! Transform
!
!           nlc = 0
!           do j=1,lc-1
!             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
!             if (lcx>0) then
!               nlc = nlc+1
!               a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)

!             endif
!           enddo
#endif /* if 0 */

           !Open up one omp region to avoid paying openmp overhead.
           !This does not help performance due to the addition of two openmp barriers around the MPI call,
           !But in the future this may be beneficial if these barriers are replaced with a faster implementation

           !$omp parallel private(mynlc, j, lcx, ii, pp ) shared(aux1)
           mynlc = 0 ! number of local columns

           !This loop does not have independent iterations,
           !'mynlc' is incremented each iteration, and it is difficult to remove this dependency
           !Thus each thread executes every iteration of the loop, except it only does the work if it 'owns' that iteration
           !That is, a thread only executes the work associated with an iteration if its thread id is congruent to
           !the iteration number modulo the number of threads
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0 ) then
               mynlc = mynlc+1
               if ( mod((j-1), omp_get_num_threads()) .eq. omp_get_thread_num() ) then
                   if (lr>0) aux1(mynlc) = dot_product(vr(1:lr),a_mat(1:lr,lcx))
               endif
             endif
           enddo

           ! Get global dot products

           !$omp barrier
           !$omp single
#ifdef WITH_MPI
           if (mynlc>0) call mpi_allreduce(aux1, aux2, mynlc, MPI_MATH_DATATYPE_PRECISION, &
                                           MPI_SUM, mpi_comm_rows, mpierr)
#else /* WITH_MPI */
           if (mynlc>0) aux2 = aux1
#endif /* WITH_MPI */
           !$omp end single
           !$omp barrier

           ! Transform
           transformChunkSize=32
           mynlc = 0
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0) then
               mynlc = mynlc+1
               !This loop could be parallelized with an openmp pragma with static scheduling and chunk size 32
               !However, for some reason this is slower than doing it manually, so it is parallelized as below.
               do ii=omp_get_thread_num()*transformChunkSize,lr,omp_get_num_threads()*transformChunkSize
                  do pp = 1,transformChunkSize
                      if (pp + ii > lr) exit
#if REALCASE == 1
                          a_mat(ii+pp,lcx) = a_mat(ii+pp,lcx) - tau*aux2(mynlc)*vr(ii+pp)
#endif
#if COMPLEXCASE == 1
                          a_mat(ii+pp,lcx) = a_mat(ii+pp,lcx) - conjg(tau)*aux2(mynlc)*vr(ii+pp)
#endif
                  enddo
               enddo
             endif
           enddo
           !$omp end parallel

#else /* WITH_OPENMP */

           nlc = 0 ! number of local columns
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0) then
               nlc = nlc+1
               if (lr>0) aux1(nlc) = dot_product(vr(1:lr),a_mat(1:lr,lcx))
             endif
           enddo

           ! Get global dot products
#ifdef WITH_MPI
           if (nlc>0) call mpi_allreduce(aux1, aux2, nlc, MPI_MATH_DATATYPE_PRECISION, &
                                         MPI_SUM, mpi_comm_rows, mpierr)
#else /* WITH_MPI */
           if (nlc>0) aux2=aux1
#endif /* WITH_MPI */
           ! Transform

           nlc = 0
           do j=1,lc-1
             lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
             if (lcx>0) then
               nlc = nlc+1
#if REALCASE == 1
               a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - tau*aux2(nlc)*vr(1:lr)
#endif
#if COMPLEXCASE == 1
               a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
#endif
             endif
           enddo
#endif /* WITH_OPENMP */
         enddo ! lc

         ! Calculate scalar products of stored Householder vectors.
         ! This can be done in different ways, we use dsyrk

         vav = 0
         if (useGPU_reduction_lower_block_to_tridiagonal) then
           if (l_rows>0) &
#if REALCASE == 1
             call PRECISION_SYRK('U', 'T',            &
#endif
#if COMPLEXCASE == 1
             call PRECISION_HERK('U', 'C',            &
#endif
                           n_cols, l_rows, ONE, &
                           vmrCUDA, cur_l_rows, &
                           ZERO, vav, ubound(vav,dim=1))

         else ! useGPU_reduction_to_tridiagonal
           if (l_rows>0) &
#if REALCASE == 1
             call PRECISION_SYRK('U', 'T',           &
#endif
#if COMPLEXCASE == 1
             call PRECISION_HERK('U', 'C',           &
#endif
                           n_cols, l_rows, ONE, vmrCPU, ubound(vmrCPU,dim=1), ZERO, vav, ubound(vav,dim=1))
         endif
#if REALCASE == 1
         call symm_matrix_allreduce_&
#endif
#if COMPLEXCASE == 1
         call herm_matrix_allreduce_&
#endif
         &PRECISION &
                         (obj, n_cols,vav, nbw, nbw,mpi_comm_rows)
         ! Calculate triangular matrix T for block Householder Transformation
         do lc=n_cols,1,-1
           tau = tmat(lc,lc,istep)
           if (lc<n_cols) then
             call PRECISION_TRMV('U', BLAS_TRANS_OR_CONJ, 'N',&
                                 n_cols-lc, tmat(lc+1,lc+1,istep), ubound(tmat,dim=1), vav(lc+1,lc), 1)

#if REALCASE == 1
             tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
#endif
#if COMPLEXCASE == 1
             tmat(lc,lc+1:n_cols,istep) = -tau * conjg(vav(lc+1:n_cols,lc))
#endif
           endif
         enddo
#if REALCASE == 1
       endif !useQR
#endif


       ! Transpose vmr -> vmc (stored in umc, second half)
       if (useGPU) then
         call elpa_transpose_vectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION &
                           (obj, vmrCUDA, cur_l_rows, mpi_comm_rows, &
                            umcCUDA(cur_l_cols * n_cols + 1), cur_l_cols, &
                            mpi_comm_cols, 1, istep*nbw, n_cols, nblk, max_threads)
       else ! useGPU
         call elpa_transpose_vectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION &
                                           (obj, vmrCPU, ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                            umcCPU(1,n_cols+1), ubound(umcCPU,dim=1), mpi_comm_cols, &
                                            1, istep*nbw, n_cols, nblk, max_threads)
       endif

       ! Calculate umc = A**T * vmr
       ! Note that the distributed A has to be transposed
       ! Opposed to direct tridiagonalization there is no need to use the cache locality
       ! of the tiles, so we can use strips of the matrix


#if 0
       ! original complex implemetation check for performance
       umcCPU(1:l_cols,1:n_cols) = 0.0_rck
       vmrCPU(1:l_rows,n_cols+1:2*n_cols) = 0.0_rck

       if (l_cols>0 .and. l_rows>0) then
         do i=0,(istep*nbw-1)/tile_size

           lcs = i*l_cols_tile+1
           lce = min(l_cols,(i+1)*l_cols_tile)
           if (lce<lcs) cycle

           lre = min(l_rows,(i+1)*l_rows_tile)

             call PRECISION_GEMM('C', 'N', lce-lcs+1, n_cols, lre, ONE, a_mat(1,lcs), ubound(a_mat,dim=1), &
                        vmrCPU, ubound(vmrCPU,dim=1), ONE, umcCPU(lcs,1), ubound(umcCPU,dim=1))

           if (i==0) cycle
           lre = min(l_rows,i*l_rows_tile)
             call PRECISION_GEMM('N', 'N', lre, n_cols, lce-lcs+1, ONE, a_mat(1,lcs), lda, &
                        umcCPU(lcs,n_cols+1), ubound(umcCPU,dim=1), ONE, vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1))
         enddo

       endif ! (l_cols>0 .and. l_rows>0)
#endif /* if 0 */

       !Code for Algorithm 4

       ! n_way is actually a branch for the number of OpenMP threads
       n_way = 1
#ifdef WITH_OPENMP

#if REALCASE == 1
       n_way = max_threads

       !$omp parallel private( i,lcs,lce,lrs,lre)
#endif
       if (n_way > 1) then
#if REALCASE == 1
         !$omp do
#endif
         do i=1,min(l_cols_tile, l_cols)
           umcCPU(i,1:n_cols) = 0.0_rck
         enddo

#if REALCASE == 1
         !$omp do
#endif
         do i=1,l_rows
           vmrCPU(i,n_cols+1:2*n_cols) = 0.0_rck
         enddo

         if (l_cols>0 .and. l_rows>0) then

           !SYMM variant 4
           !Partitioned Matrix Expression:
           ! Ct = Atl Bt + Atr Bb
           ! Cb = Atr' Bt + Abl Bb
           !
           !Loop invariant:
           ! Ct = Atl Bt + Atr Bb
           !
           !Update:
           ! C1 = A10'B0 + A11B1 + A21 B2
           !
           !This algorithm chosen because in this algoirhtm, the loop around the dgemm calls
           !is easily parallelized, and regardless of choise of algorithm,
           !the startup cost for parallelizing the dgemms inside the loop is too great
#if REALCASE == 1
           !$omp do schedule(static,1)
#endif
           do i=0,(istep*nbw-1)/tile_size
             lcs = i*l_cols_tile+1                   ! local column start
             lce = min(l_cols, (i+1)*l_cols_tile)    ! local column end

             lrs = i*l_rows_tile+1                   ! local row start
             lre = min(l_rows, (i+1)*l_rows_tile)    ! local row end

             !C1 += [A11 A12] [B1
             !                 B2]
             if ( lre > lrs .and. l_cols > lcs ) then
               call PRECISION_GEMM('N', 'N', lre-lrs+1, n_cols, l_cols-lcs+1,          &
                                   ONE, a_mat(lrs,lcs), ubound(a_mat,dim=1),                 &
                                   umcCPU(lcs,n_cols+1), ubound(umcCPU,dim=1),  &
                                   ZERO, vmrCPU(lrs,n_cols+1), ubound(vmrCPU,dim=1))
             endif

             ! C1 += A10' B0
             if ( lce > lcs .and. i > 0 ) then
               call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',     &
                        lce-lcs+1, n_cols, lrs-1,              &
                                    ONE, a_mat(1,lcs),   ubound(a_mat,dim=1),      &
                                    vmrCPU(1,1),   ubound(vmrCPU,dim=1),   &
                                    ZERO, umcCPU(lcs,1), ubound(umcCPU,dim=1))
             endif
           enddo
         endif ! l_cols>0 .and. l_rows>0

      else ! n_way > 1
#endif /* WITH_OPENMP */

        if (useGPU) then
          umcCUDA(1 : l_cols * n_cols) = 0.0_rck
          vmrCUDA(cur_l_rows * n_cols + 1 : cur_l_rows * n_cols * 2) = 0.0_rck
        else ! useGPU
          umcCPU(1:l_cols,1:n_cols) = 0.0_rck
          vmrCPU(1:l_rows,n_cols+1:2*n_cols) = 0.0_rck
        endif ! useGPU

        if (l_cols>0 .and. l_rows>0) then

          do i=0,(istep*nbw-1)/tile_size

            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            if (lce<lcs) cycle
            lre = min(l_rows,(i+1)*l_rows_tile)

            if (useGPU) then
              print *,"bandred_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy vmr_dev 4"
              stop 1
            else ! useGPU

              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',          &
                             lce-lcs+1, n_cols, lre, ONE, a_mat(1,lcs), ubound(a_mat,dim=1), &
                                   vmrCPU, ubound(vmrCPU,dim=1), ONE, umcCPU(lcs,1), ubound(umcCPU,dim=1))
              if (i==0) cycle
              lre = min(l_rows,i*l_rows_tile)
              call PRECISION_GEMM('N', 'N', lre, n_cols, lce-lcs+1, ONE, a_mat(1,lcs), lda, &
                                     umcCPU(lcs,n_cols+1), ubound(umcCPU,dim=1), ONE,      &
                                     vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1))
            endif ! useGPU
          enddo ! i=0,(istep*nbw-1)/tile_size

        endif ! l_cols>0 .and. l_rows>0

#ifdef WITH_OPENMP
      endif ! n_way > 1
#if REALCASE == 1
      !$omp end parallel
#endif
#endif
       ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
       ! on the processors containing the diagonal
       ! This is only necessary if ur has been calculated, i.e. if the
       ! global tile size is smaller than the global remaining matrix

       ! Or if we used the Algorithm 4
       if (tile_size < istep*nbw .or. n_way > 1) then

         if (useGPU) then

           call elpa_reduce_add_vectors_&
                &MATH_DATATYPE&
                &_&
                &PRECISION &
                                (obj, vmrCUDA(cur_l_rows * n_cols + 1),cur_l_rows,  &
                                 mpi_comm_rows, umcCUDA,                            &
                                 cur_l_cols, mpi_comm_cols, istep*nbw, n_cols, nblk, max_threads)
         else ! useGPU

           call elpa_reduce_add_vectors_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
                                            (obj, vmrCPU(1,n_cols+1),ubound(vmrCPU,dim=1),mpi_comm_rows, &
                                             umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                             istep*nbw, n_cols, nblk, max_threads)
         endif ! useGPU
       endif ! tile_size < istep*nbw .or. n_way > 1

       if (l_cols>0) then

         if (useGPU) then
              print *,"bandred_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy vmr_dev 4"
              stop 1

         else ! useGPU

           allocate(tmpCPU(l_cols,n_cols), stat=istat, errmsg=errorMessage)
           if (istat .ne. 0) then
             print *,"bandred_&
                     &MATH_DATATYPE&
                     &: error when allocating tmpCPU "//errorMessage
             stop 1
           endif

#ifdef WITH_MPI
           call mpi_allreduce(umcCPU, tmpCPU, l_cols*n_cols, MPI_MATH_DATATYPE_PRECISION,    &
            MPI_SUM, mpi_comm_rows, mpierr)
           umcCPU(1:l_cols,1:n_cols) = tmpCPU(1:l_cols,1:n_cols)
#else /* WITH_MPI */
!           tmpCPU(1:l_cols,1:n_cols) = umcCPU(1:l_cols,1:n_cols)
#endif /* WITH_MPI */

           deallocate(tmpCPU, stat=istat, errmsg=errorMessage)
           if (istat .ne. 0) then
             print *,"bandred_&
                     &MATH_DATATYPE&
                     &: error when deallocating tmpCPU "//errorMessage
             stop 1
           endif
         endif ! useGPU
       endif ! l_cols > 0

       ! U = U * Tmat**T

       if (useGPU) then
              print *,"bandred_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy vmr_dev 4"
              stop 1

       else ! useGPU

         call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',     &
                        l_cols,n_cols, ONE, tmat(1,1,istep), ubound(tmat,dim=1), &
                              umcCPU, ubound(umcCPU,dim=1))

         ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

         call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',              &
                       n_cols, n_cols, l_cols, ONE, umcCPU, ubound(umcCPU,dim=1), umcCPU(1,n_cols+1), &
                             ubound(umcCPU,dim=1), ZERO, vav, ubound(vav,dim=1))

         call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
                       n_cols, n_cols, ONE, tmat(1,1,istep),    &
                             ubound(tmat,dim=1), vav, ubound(vav,dim=1))

       endif ! useGPU

#if REALCASE == 1
       call symm_matrix_allreduce_&
#endif
#if COMPLEXCASE == 1
       call herm_matrix_allreduce_&
#endif
            &PRECISION &
                              (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols)

       ! U = U - 0.5 * V * VAV

       if (useGPU) then
              print *,"bandred_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy vmr_dev 4"
              stop 1
          
       else ! useGPU
         call PRECISION_GEMM('N', 'N', l_cols, n_cols, n_cols,     &
#if REALCASE == 1
                       -0.5_rk,                           &
#endif
#if COMPLEXCASE == 1
                              (-0.5_rk, 0.0_rk),     &
#endif
            umcCPU(1,n_cols+1), ubound(umcCPU,dim=1), vav, &
                              ubound(vav,dim=1), ONE, umcCPU, ubound(umcCPU,dim=1))

         ! Transpose umc -> umr (stored in vmr, second half)
         call elpa_transpose_vectors_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
                                  (obj, umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                         vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                         1, istep*nbw, n_cols, nblk, max_threads)

       endif  ! useGPU


       ! A = A - V*U**T - U*V**T

#ifdef WITH_OPENMP
       !$omp parallel private( ii, i, lcs, lce, lre, n_way, m_way, m_id, n_id, work_per_thread, mystart, myend  )
       n_threads = omp_get_num_threads()

       if (mod(n_threads, 2) == 0) then
         n_way = 2
       else
         n_way = 1
       endif

       m_way = n_threads / n_way

       m_id = mod(omp_get_thread_num(),  m_way)
       n_id = omp_get_thread_num() / m_way

       do ii=n_id*tile_size,(istep*nbw-1),tile_size*n_way
         i = ii / tile_size
         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         lre = min(l_rows,(i+1)*l_rows_tile)
         if (lce<lcs .or. lre<1) cycle

         !Figure out this thread's range
         work_per_thread = lre / m_way
         if (work_per_thread * m_way < lre) work_per_thread = work_per_thread + 1
         mystart = m_id * work_per_thread + 1
         myend   = mystart + work_per_thread - 1
         if ( myend > lre ) myend = lre
         if ( myend-mystart+1 < 1) cycle
         call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, myend-mystart+1, lce-lcs+1, 2*n_cols, -ONE, &
                    vmrCPU(mystart, 1), ubound(vmrCPU,1), umcCPU(lcs,1), ubound(umcCPU,1), &
                     ONE, a_mat(mystart,lcs), ubound(a_mat,1))
       enddo
       !$omp end parallel
!#if COMPLEXCASE == 1
!       do i=0,(istep*nbw-1)/tile_size
!         lcs = i*l_cols_tile+1
!         lce = min(l_cols,(i+1)*l_cols_tile)
!         lre = min(l_rows,(i+1)*l_rows_tile)
!         if (lce<lcs .or. lre<1) cycle
!         call obj%timer%start("blas")
!         call PRECISION_GEMM('N', 'C', lre,lce-lcs+1, 2*n_cols, -ONE, &
!                       vmrCPU, ubound(vmrCPU,dim=1), umcCPU(lcs,1), ubound(umcCPU,dim=1), &
!                       ONE, a_mat(1,lcs), lda)
!         call obj%timer%stop("blas")
!       enddo
!#endif

#else /* WITH_OPENMP */

       do i=0,(istep*nbw-1)/tile_size
         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         lre = min(l_rows,(i+1)*l_rows_tile)
         if (lce<lcs .or. lre<1) cycle

         if (useGPU) then
              print *,"bandred_&
                      &MATH_DATATYPE&
                      &: error in cudaMemcpy vmr_dev 4"
              stop 1

         else ! useGPU

           call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, lre,lce-lcs+1, 2*n_cols, -ONE, &
                               vmrCPU, ubound(vmrCPU,dim=1), umcCPU(lcs,1), ubound(umcCPU,dim=1), &
                               ONE, a_mat(1,lcs), lda)
         endif ! useGPU
       enddo ! i=0,(istep*nbw-1)/tile_size
#endif /* WITH_OPENMP */

       if (.not.(useGPU)) then
         if (allocated(vr)) then
           deallocate(vr, stat=istat, errmsg=errorMessage)
           if (istat .ne. 0) then
             print *,"bandred_&
                     &MATH_DATATYPE&
                     &: error when deallocating vr "//errorMessage
             stop 1
           endif
         endif

         if (allocated(umcCPU)) then
           deallocate(umcCPU, stat=istat, errmsg=errorMessage)
           if (istat .ne. 0) then
             print *,"bandred_&
                     &MATH_DATATYPE&
                     &: error when deallocating umcCPU "//errorMessage
             stop 1
           endif
         endif

         if (allocated(vmrCPU)) then
           deallocate(vmrCPU, stat=istat, errmsg=errorMessage)
           if (istat .ne. 0) then
             print *,"bandred_&
                     &MATH_DATATYPE&
                     &: error when deallocating vmrCPU "//errorMessage
             stop 1
           endif
         endif
       endif !useGPU

     enddo ! istep - loop


     if (allocated(vr)) then
       deallocate(vr, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_&
                 &MATH_DATATYPE&
                 &: error when deallocating vr "//errorMessage
         stop 1
       endif
     endif

     if (allocated(umcCPU)) then
       deallocate(umcCPU, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_&
                 &MATH_DATATYPE&
                 &: error when deallocating umcCPU "//errorMessage
         stop 1
       endif
     endif

     if (allocated(vmrCPU)) then
       deallocate(vmrCPU, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_&
                 &MATH_DATATYPE&
                 &: error when deallocating vmrCPU "//errorMessage
         stop 1
       endif
     endif

!#if COMPLEXCASE == 1
!       ! check this
!       if (useGPU) then
!         successCUDA = cuda_free(umc_dev)
!         if (.not.(successCUDA)) then
!           print *,"bandred_complex: error in cudaFree umc_dev 7a"
!           stop
!         endif
!       endif
!#endif



   end subroutine bandred_&
   &MATH_DATATYPE&
   &_&
   &PRECISION
#if REALCASE == 1
   ! slower for gpu on 10000 10000 ???
#endif

