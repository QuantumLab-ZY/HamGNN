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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif


    subroutine trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, na, nqc, nblk, nbw, a_mat, a_dev, lda, tmat, tmat_dev, q_mat, &
     q_dev, ldq, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, useGPU &
#if REALCASE == 1
     ,useQr)
#endif
#if COMPLEXCASE == 1
     )
#endif

    !-------------------------------------------------------------------------------
    !  trans_ev_band_to_full_real/complex:
    !  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
    !
    !  Parameters
    !
    !  na          Order of matrix a_mat, number of rows of matrix q_mat
    !
    !  nqc         Number of columns of matrix q_mat
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  nbw         semi bandwith
    !
    !  a_mat(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a_mat after bandred_real/complex)
    !              Distribution is like in Scalapack.
    !
    !  lda         Leading dimension of a_mat
    !  matrixCols  local columns of matrix a_mat and q_mat
    !
    !  tmat(nbw,nbw,numBlocks) Factors returned by bandred_real/complex
    !
    !  q_mat           On input: Eigenvectors of band matrix
    !              On output: Transformed eigenvectors
    !              Distribution is like in Scalapack.
    !
    !  ldq         Leading dimension of q_mat
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns
    !
    !-------------------------------------------------------------------------------
      use precision
      use iso_c_binding
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      logical, intent(in)                    :: useGPU
#if REALCASE == 1
     logical, intent(in)                     :: useQR
#endif
      integer(kind=ik)                       :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
      MATH_DATATYPE(kind=rck)               :: a_mat(lda,*), q_mat(ldq,*), tmat(nbw,nbw,*)
#else
      MATH_DATATYPE(kind=rck)               :: a_mat(lda,matrixCols), q_mat(ldq,matrixCols), tmat(nbw, nbw, numBlocks)
#endif
      integer(kind=C_intptr_T)               :: a_dev ! passed from bandred_real at the moment not used since copied in bandred_real

      integer(kind=ik)                       :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)                       :: max_blocks_row, max_blocks_col, max_local_rows, &
                                                max_local_cols
      integer(kind=ik)                       :: l_cols, l_rows, l_colh, n_cols
      integer(kind=ik)                       :: istep, lc, ncol, nrow, nb, ns

      MATH_DATATYPE(kind=rck), allocatable   :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)
      ! hvm_dev is fist used and set in this routine
      ! q_mat is changed in trans_ev_tridi on the host, copied to device and passed here. this can be adapted
      ! tmp_dev is first used in this routine
      ! tmat_dev is passed along from bandred_real
      integer(kind=C_intptr_T)               :: hvm_dev, q_dev, tmp_dev, tmat_dev

      integer(kind=ik)                       :: i

#ifdef BAND_TO_FULL_BLOCKING
      MATH_DATATYPE(kind=rck), allocatable   :: tmat_complete(:,:), t_tmp(:,:), t_tmp2(:,:)
      integer(kind=ik)                       :: cwy_blocking, t_blocking, t_cols, t_rows
#endif

      integer(kind=ik)                       :: istat
      character(200)                         :: errorMessage
      character(20)                          :: gpuString
      logical                                :: successCUDA
      integer                                :: blocking_factor, erro
!      integer(kind=c_intptr_t), parameter    :: size_of_datatype = size_of_&
!                                                                   &PRECISION&
!                                                                   &_&
!                                                                   &MATH_DATATYPE

      if(useGPU) then
        gpuString = "_gpu"
      else
        gpuString = ""
      endif


#ifdef BAND_TO_FULL_BLOCKING
      blocking_factor = 3
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)


      max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of a_mat
      max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q_mat!

      max_local_rows = max_blocks_row*nblk
      max_local_cols = max_blocks_col*nblk

      if (useGPU) then

          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp1 "//errorMessage
          stop 1

      else ! do not useGPU

#ifdef BAND_TO_FULL_BLOCKING
        ! t_blocking was formerly 2; 3 is a better choice
        t_blocking = blocking_factor ! number of matrices T (tmat) which are aggregated into a new (larger) T matrix (tmat_complete) and applied at once

        ! we only use the t_blocking if we could call it fully, this is might be better but needs to benchmarked.
!       if ( na >= ((t_blocking+1)*nbw) ) then
        cwy_blocking = t_blocking * nbw

        allocate(tmp1(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp1 "//errorMessage
          stop 1
        endif

        allocate(tmp2(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp2 "//errorMessage
          stop 1
        endif

        allocate(hvb(max_local_rows*cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvb "//errorMessage
          stop 1
        endif

        allocate(hvm(max_local_rows,cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvm "//errorMessage
          stop 1
        endif

#else /* BAND_TO_FULL_BLOCKING */

        allocate(tmp1(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating tmp1 "//errorMessage
          stop 1
        endif

        allocate(tmp2(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&: error when allocating tmp2 "//errorMessage
          stop 1
        endif

        allocate(hvb(max_local_rows*nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvb "//errorMessage
          stop 1
        endif

        allocate(hvm(max_local_rows,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating hvm "//errorMessage
          stop 1
        endif
#endif /* BAND_TO_FULL_BLOCKING */

#ifdef BAND_TO_FULL_BLOCKING
        allocate(tmat_complete(cwy_blocking,cwy_blocking), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                   &MATH_DATATYPE&
                   &: error when allocating tmat_complete "//errorMessage
          stop 1
        endif
        allocate(t_tmp(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating t_tmp "//errorMessage
          stop 1
        endif
        allocate(t_tmp2(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when allocating t_tmp2 "//errorMessage
          stop 1
        endif
#endif
!        else
!          allocate(tmp1(max_local_cols*nbw))
!          allocate(tmp2(max_local_cols*nbw))
!          allocate(hvb(max_local_rows*nbw))
!          allocate(hvm(max_local_rows,nbw))
!        endif

        hvm = 0.0_rck   ! Must be set to 0 !!!
        hvb = 0.0_rck   ! Safety only
        l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q_mat

!       if ( na >= ((t_blocking+1)*nbw) ) then

#ifdef BAND_TO_FULL_BLOCKING
        do istep=1,((na-1)/nbw-1)/t_blocking + 1
#else
        do istep=1,(na-1)/nbw
#endif

#ifdef BAND_TO_FULL_BLOCKING
          ! This the call when using  na >= ((t_blocking+1)*nbw)
          !      n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw
          ! Number of columns in current step
          ! As an alternative we add some special case handling if na < cwy_blocking
          IF (na < cwy_blocking) THEN
            n_cols = MAX(0, na-nbw)
            IF ( n_cols .eq. 0 ) THEN
              EXIT
            END IF
          ELSE
            n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw ! Number of columns in current step
          END IF
#else /* BAND_TO_FULL_BLOCKING */
          n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step
#endif /* BAND_TO_FULL_BLOCKING */
          ! Broadcast all Householder vectors for current step compressed in hvb

          nb = 0
          ns = 0

          do lc = 1, n_cols
#ifdef BAND_TO_FULL_BLOCKING
            ncol = (istep-1)*cwy_blocking + nbw + lc ! absolute column number of householder Vector
#else
            ncol = istep*nbw + lc ! absolute column number of householder Vector
#endif
            nrow = ncol - nbw ! absolute number of pivot row

            l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
            l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

            if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)

            nb = nb+l_rows

            if (lc==n_cols .or. mod(ncol,nblk)==0) then
#ifdef WITH_MPI
              call MPI_Bcast(hvb(ns+1), nb-ns, MPI_MATH_DATATYPE_PRECISION,    &
                             pcol(ncol, nblk, np_cols), mpi_comm_cols, mpierr)


#endif /* WITH_MPI */
              ns = nb
            endif
          enddo ! lc

          ! Expand compressed Householder vectors into matrix hvm

          nb = 0
          do lc = 1, n_cols
#ifdef BAND_TO_FULL_BLOCKING
            nrow = (istep-1)*cwy_blocking + lc ! absolute number of pivot row
#else
            nrow = (istep-1)*nbw+lc ! absolute number of pivot row
#endif
            l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

            hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
            if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.0_rck
            nb = nb+l_rows
          enddo

#ifdef BAND_TO_FULL_BLOCKING
          l_rows = local_index(MIN(na,(istep+1)*cwy_blocking), my_prow, np_rows, nblk, -1)

          ! compute tmat2 out of tmat(:,:,)
          tmat_complete = 0
          do i = 1, t_blocking
            t_cols = MIN(nbw, n_cols - (i-1)*nbw)
            if (t_cols <= 0) exit
            t_rows = (i - 1) * nbw
            tmat_complete(t_rows+1:t_rows+t_cols,t_rows+1:t_rows+t_cols) = tmat(1:t_cols,1:t_cols,(istep-1)*t_blocking + i)

            if (i > 1) then
              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',      &
                           t_rows, t_cols, l_rows, ONE, hvm(1,1), max_local_rows, hvm(1,(i-1)*nbw+1), &
                                 max_local_rows, ZERO, t_tmp, cwy_blocking)

#ifdef WITH_MPI

              call mpi_allreduce(t_tmp, t_tmp2, cwy_blocking*nbw, MPI_MATH_DATATYPE_PRECISION,    &
                                 MPI_SUM, mpi_comm_rows, mpierr)
              call PRECISION_TRMM('L', 'U', 'N', 'N', t_rows, t_cols, ONE, tmat_complete, cwy_blocking, t_tmp2, cwy_blocking)
              call PRECISION_TRMM('R', 'U', 'N', 'N', t_rows, t_cols, -ONE, tmat_complete(t_rows+1,t_rows+1), cwy_blocking, &
                         t_tmp2, cwy_blocking)

              tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)

#else /* WITH_MPI */
!              t_tmp2(1:cwy_blocking,1:nbw) = t_tmp(1:cwy_blocking,1:nbw)
              call PRECISION_TRMM('L', 'U', 'N', 'N', t_rows, t_cols, ONE, tmat_complete, cwy_blocking, t_tmp, cwy_blocking)
              call PRECISION_TRMM('R', 'U', 'N', 'N', t_rows, t_cols, -ONE, tmat_complete(t_rows+1,t_rows+1), cwy_blocking, &
                         t_tmp, cwy_blocking)

              tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp(1:t_rows,1:t_cols)

#endif /* WITH_MPI */

!              call PRECISION_TRMM('L', 'U', 'N', 'N', t_rows, t_cols, ONE, tmat_complete, cwy_blocking, t_tmp2, cwy_blocking)
!              call PRECISION_TRMM('R', 'U', 'N', 'N', t_rows, t_cols, -ONE, tmat_complete(t_rows+1,t_rows+1), cwy_blocking, &
!                         t_tmp2, cwy_blocking)

!              tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)
             endif
          enddo
#else /* BAND_TO_FULL_BLOCKING */
          l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)
#endif

          ! Q = Q - V * T**T * V**T * Q

          if (l_rows>0) then

            call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',         &
                          n_cols, l_cols, l_rows, ONE, hvm, ubound(hvm,dim=1), &
                                q_mat, ldq, ZERO, tmp1, n_cols)

          else ! l_rows>0

            tmp1(1:l_cols*n_cols) = 0.0_rck
          endif ! l_rows>0

#ifdef WITH_MPI
          call mpi_allreduce(tmp1, tmp2, n_cols*l_cols, MPI_MATH_DATATYPE_PRECISION, MPI_SUM, mpi_comm_rows ,mpierr)

          if (l_rows>0) then
#ifdef BAND_TO_FULL_BLOCKING

            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                          n_cols, l_cols, ONE, tmat_complete, cwy_blocking, tmp2, n_cols)
            call PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm, ubound(hvm,dim=1), tmp2, n_cols, ONE, q_mat, ldq)

#else /* BAND_TO_FULL_BLOCKING */

            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                n_cols, l_cols, ONE, tmat(1,1,istep), ubound(tmat,dim=1), tmp2, n_cols)
            call PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm, ubound(hvm,dim=1), &
                                tmp2, n_cols, ONE, q_mat, ldq)

#endif /* BAND_TO_FULL_BLOCKING */

          endif
#else /* WITH_MPI */
!          tmp2 = tmp1
          if (l_rows>0) then
#ifdef BAND_TO_FULL_BLOCKING
            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                n_cols, l_cols, ONE, tmat_complete, cwy_blocking, tmp1, n_cols)
            call PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm, ubound(hvm,dim=1), tmp1, n_cols, ONE, q_mat, ldq)
#else /* BAND_TO_FULL_BLOCKING */

            call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',        &
                                n_cols, l_cols, ONE, tmat(1,1,istep), ubound(tmat,dim=1), tmp1, n_cols)
            call PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm, ubound(hvm,dim=1), &
                                tmp1, n_cols, ONE, q_mat, ldq)

#endif  /* BAND_TO_FULL_BLOCKING */
          endif
#endif /* WITH_MPI */

!          if (l_rows>0) then
!            call PRECISION_TRMM('L', 'U', 'T', 'N', n_cols, l_cols, ONE, tmat_complete, cwy_blocking, tmp2, n_cols)
!            call PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm, ubound(hvm,dim=1), tmp2, n_cols, ONE, q_mat, ldq)
!          endif

        enddo ! istep

      endif ! useGPU

      deallocate(tmp1, tmp2, hvb, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_band_to_full_&
                 &MATH_DATATYPE&
                 &: error when deallocating tmp1 tmp2 hvb "//errorMessage
        stop 1
      endif


      deallocate(hvm, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_band_to_full_&
                &MATH_DATATYPE&
                &: error when deallocating hvm "//errorMessage
        stop 1
      endif

#if BAND_TO_FULL_BLOCKING
      if (.not.(useGPU)) then
        deallocate(tmat_complete, t_tmp, t_tmp2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_band_to_full_&
                  &MATH_DATATYPE&
                  &: error when deallocating tmat_complete, t_tmp, t_tmp2 "//errorMessage
          stop 1
        endif
      endif
#endif

    end subroutine trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &_&
    &PRECISION


