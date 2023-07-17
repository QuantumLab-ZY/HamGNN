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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

subroutine solve_tridi_&
&PRECISION_AND_SUFFIX &
    ( obj, na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                           mpi_comm_cols, useGPU, wantDebug, success, max_threads )

      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)               :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=REAL_DATATYPE), intent(inout)    :: d(na), e(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,*)
#else
      real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,matrixCols)
#endif
      logical, intent(in)                        :: useGPU, wantDebug
      logical, intent(out)                       :: success

      integer(kind=ik)                           :: i, j, n, np, nc, nev1, l_cols, l_rows
      integer(kind=ik)                           :: my_prow, my_pcol, np_rows, np_cols, mpierr

      integer(kind=ik), allocatable              :: limits(:), l_col(:), p_col(:), l_col_bc(:), p_col_bc(:)

      integer(kind=ik)                           :: istat
      character(200)                             :: errorMessage
      character(20)                              :: gpuString
      integer(kind=ik), intent(in)               :: max_threads

      if(useGPU) then
        gpuString = "_gpu"
      else
        gpuString = ""
      endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      success = .true.

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

      ! Set Q to 0
      q(1:l_rows, 1:l_cols) = 0.0_rk

      ! Get the limits of the subdivisons, each subdivison has as many cols
      ! as fit on the respective processor column

      allocate(limits(0:np_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating limits "//errorMessage
        stop 1
      endif

      limits(0) = 0
      do np=0,np_cols-1
        nc = local_index(na, np, np_cols, nblk, -1) ! number of columns on proc column np

        ! Check for the case that a column has have zero width.
        ! This is not supported!
        ! Scalapack supports it but delivers no results for these columns,
        ! which is rather annoying
        if (nc==0) then
          success = .false.
          return
        endif
        limits(np+1) = limits(np) + nc
      enddo

      ! Subdivide matrix by subtracting rank 1 modifications

      do i=1,np_cols-1
        n = limits(i)
        d(n) = d(n)-abs(e(n))
        d(n+1) = d(n+1)-abs(e(n))
      enddo

      ! Solve sub problems on processsor columns

      nc = limits(my_pcol) ! column after which my problem starts

      if (np_cols>1) then
        nev1 = l_cols ! all eigenvectors are needed
      else
        nev1 = MIN(nev,l_cols)
      endif
      call solve_tridi_col_&
           &PRECISION_AND_SUFFIX &
             (obj, l_cols, nev1, nc, d(nc+1), e(nc+1), q, ldq, nblk,  &
                        matrixCols, mpi_comm_rows, useGPU, wantDebug, success, max_threads)
      if (.not.(success)) then
        return
      endif
      ! If there is only 1 processor column, we are done

      if (np_cols==1) then
        deallocate(limits, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi: error when deallocating limits "//errorMessage
          stop 1
        endif

        return
      endif

      ! Set index arrays for Q columns

      ! Dense distribution scheme:

      allocate(l_col(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating l_col "//errorMessage
        stop 1
      endif

      allocate(p_col(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating p_col "//errorMessage
        stop 1
      endif

      n = 0
      do np=0,np_cols-1
        nc = local_index(na, np, np_cols, nblk, -1)
        do i=1,nc
          n = n+1
          l_col(n) = i
          p_col(n) = np
        enddo
      enddo

      ! Block cyclic distribution scheme, only nev columns are set:

      allocate(l_col_bc(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating l_col_bc "//errorMessage
        stop 1
      endif

      allocate(p_col_bc(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating p_col_bc "//errorMessage
        stop 1
      endif

      p_col_bc(:) = -1
      l_col_bc(:) = -1

      do i = 0, na-1, nblk*np_cols
        do j = 0, np_cols-1
          do n = 1, nblk
            if (i+j*nblk+n <= MIN(nev,na)) then
              p_col_bc(i+j*nblk+n) = j
              l_col_bc(i+j*nblk+n) = i/np_cols + n
             endif
           enddo
         enddo
      enddo

      ! Recursively merge sub problems
      call merge_recursive_&
           &PRECISION &
           (obj, 0, np_cols, useGPU, wantDebug, success)
      if (.not.(success)) then
        return
      endif

      deallocate(limits,l_col,p_col,l_col_bc,p_col_bc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when deallocating l_col "//errorMessage
        stop 1
      endif

      return

      contains
        recursive subroutine merge_recursive_&
                  &PRECISION &
           (obj, np_off, nprocs, useGPU, wantDebug, success)
           use precision
           implicit none

           ! noff is always a multiple of nblk_ev
           ! nlen-noff is always > nblk_ev

!           class(elpa_abstract_impl_t), intent(inout) :: obj
           integer(kind=c_int), intent(in)                                     :: obj
           integer(kind=ik)     :: np_off, nprocs
           integer(kind=ik)     :: np1, np2, noff, nlen, nmid, n
#ifdef WITH_MPI
!           integer(kind=ik)     :: my_mpi_status(mpi_status_size)
#endif
           logical, intent(in)  :: useGPU, wantDebug
           logical, intent(out) :: success

           success = .true.

           if (nprocs<=1) then
             ! Safety check only
             success = .false.
             return
           endif
           ! Split problem into 2 subproblems of size np1 / np2

           np1 = nprocs/2
           np2 = nprocs-np1

           if (np1 > 1) call merge_recursive_&
                        &PRECISION &
           (obj, np_off, np1, useGPU, wantDebug, success)
           if (.not.(success)) return
           if (np2 > 1) call merge_recursive_&
                        &PRECISION &
           (obj, np_off+np1, np2, useGPU, wantDebug, success)
           if (.not.(success)) return

           noff = limits(np_off)
           nmid = limits(np_off+np1) - noff
           nlen = limits(np_off+nprocs) - noff

#ifdef WITH_MPI
           if (my_pcol==np_off) then
             do n=np_off+np1,np_off+nprocs-1
               call mpi_send(d(noff+1), nmid, MPI_REAL_PRECISION, n, 1, mpi_comm_cols, mpierr)
             enddo
           endif
#endif /* WITH_MPI */

           if (my_pcol>=np_off+np1 .and. my_pcol<np_off+nprocs) then
#ifdef WITH_MPI
             call mpi_recv(d(noff+1), nmid, MPI_REAL_PRECISION, np_off, 1, mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */
!             d(noff+1:noff+1+nmid-1) = d(noff+1:noff+1+nmid-1)
#endif /* WITH_MPI */
           endif

           if (my_pcol==np_off+np1) then
             do n=np_off,np_off+np1-1
#ifdef WITH_MPI
               call mpi_send(d(noff+nmid+1), nlen-nmid, MPI_REAL_PRECISION, n, 1, mpi_comm_cols, mpierr)
#endif /* WITH_MPI */

             enddo
           endif
           if (my_pcol>=np_off .and. my_pcol<np_off+np1) then
#ifdef WITH_MPI
             call mpi_recv(d(noff+nmid+1), nlen-nmid, MPI_REAL_PRECISION, np_off+np1, 1,mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */
!             d(noff+nmid+1:noff+nmid+1+nlen-nmid-1) = d(noff+nmid+1:noff+nmid+1+nlen-nmid-1)
#endif /* WITH_MPI */
           endif
           if (nprocs == np_cols) then

             ! Last merge, result distribution must be block cyclic, noff==0,
             ! p_col_bc is set so that only nev eigenvalues are calculated
             call merge_systems_&
                  &PRECISION &
                                 (obj, nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, noff, &
                                 nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, l_col, p_col, &
                                 l_col_bc, p_col_bc, np_off, nprocs, useGPU, wantDebug, success, max_threads )
             if (.not.(success)) return
           else
             ! Not last merge, leave dense column distribution
             call merge_systems_&
                  &PRECISION &
                                (obj, nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, noff, &
                                 nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, l_col(noff+1), p_col(noff+1), &
                                 l_col(noff+1), p_col(noff+1), np_off, nprocs, useGPU, wantDebug, success, max_threads )
             if (.not.(success)) return
           endif
       end subroutine merge_recursive_&
           &PRECISION

    end subroutine solve_tridi_&
        &PRECISION_AND_SUFFIX

    subroutine solve_tridi_col_&
    &PRECISION_AND_SUFFIX &
      ( obj, na, nev, nqoff, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, useGPU, wantDebug, success, max_threads )

   ! Solves the symmetric, tridiagonal eigenvalue problem on one processor column
   ! with the divide and conquer method.
   ! Works best if the number of processor rows is a power of 2!
      use precision
      implicit none
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj

      integer(kind=ik)              :: na, nev, nqoff, ldq, nblk, matrixCols, mpi_comm_rows
      real(kind=REAL_DATATYPE)      :: d(na), e(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE)      :: q(ldq,*)
#else
      real(kind=REAL_DATATYPE)      :: q(ldq,matrixCols)
#endif

      integer(kind=ik), parameter   :: min_submatrix_size = 16 ! Minimum size of the submatrices to be used

      real(kind=REAL_DATATYPE), allocatable    :: qmat1(:,:), qmat2(:,:)
      integer(kind=ik)              :: i, n, np
      integer(kind=ik)              :: ndiv, noff, nmid, nlen, max_size
      integer(kind=ik)              :: my_prow, np_rows, mpierr

      integer(kind=ik), allocatable :: limits(:), l_col(:), p_col_i(:), p_col_o(:)
      logical, intent(in)           :: useGPU, wantDebug
      logical, intent(out)          :: success
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

      integer(kind=ik), intent(in)  :: max_threads

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      success = .true.
      ! Calculate the number of subdivisions needed.

      n = na
      ndiv = 1
      do while(2*ndiv<=np_rows .and. n>2*min_submatrix_size)
        n = ((n+3)/4)*2 ! the bigger one of the two halves, we want EVEN boundaries
        ndiv = ndiv*2
      enddo

      ! If there is only 1 processor row and not all eigenvectors are needed
      ! and the matrix size is big enough, then use 2 subdivisions
      ! so that merge_systems is called once and only the needed
      ! eigenvectors are calculated for the final problem.

      if (np_rows==1 .and. nev<na .and. na>2*min_submatrix_size) ndiv = 2

      allocate(limits(0:ndiv), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi_col: error when allocating limits "//errorMessage
        stop 1
      endif

      limits(0) = 0
      limits(ndiv) = na

      n = ndiv
      do while(n>1)
        n = n/2 ! n is always a power of 2
        do i=0,ndiv-1,2*n
          ! We want to have even boundaries (for cache line alignments)
          limits(i+n) = limits(i) + ((limits(i+2*n)-limits(i)+3)/4)*2
        enddo
      enddo

      ! Calculate the maximum size of a subproblem

      max_size = 0
      do i=1,ndiv
        max_size = MAX(max_size,limits(i)-limits(i-1))
      enddo

      ! Subdivide matrix by subtracting rank 1 modifications

      do i=1,ndiv-1
        n = limits(i)
        d(n) = d(n)-abs(e(n))
        d(n+1) = d(n+1)-abs(e(n))
      enddo

      if (np_rows==1)    then

        ! For 1 processor row there may be 1 or 2 subdivisions
        do n=0,ndiv-1
          noff = limits(n)        ! Start of subproblem
          nlen = limits(n+1)-noff ! Size of subproblem

          call solve_tridi_single_problem_&
          &PRECISION_AND_SUFFIX &
                                  (obj, nlen,d(noff+1),e(noff+1), &
                                    q(nqoff+noff+1,noff+1),ubound(q,dim=1), wantDebug, success)

          if (.not.(success)) return
        enddo

      else

        ! Solve sub problems in parallel with solve_tridi_single
        ! There is at maximum 1 subproblem per processor

        allocate(qmat1(max_size,max_size), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi_col: error when allocating qmat1 "//errorMessage
          stop 1
        endif

        allocate(qmat2(max_size,max_size), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi_col: error when allocating qmat2 "//errorMessage
          stop 1
        endif

        qmat1 = 0 ! Make sure that all elements are defined

        if (my_prow < ndiv) then

          noff = limits(my_prow)        ! Start of subproblem
          nlen = limits(my_prow+1)-noff ! Size of subproblem
          call solve_tridi_single_problem_&
          &PRECISION_AND_SUFFIX &
                                    (obj, nlen,d(noff+1),e(noff+1),qmat1, &
                                    ubound(qmat1,dim=1), wantDebug, success)

          if (.not.(success)) return
        endif

        ! Fill eigenvectors in qmat1 into global matrix q

        do np = 0, ndiv-1

          noff = limits(np)
          nlen = limits(np+1)-noff
#ifdef WITH_MPI
          call MPI_Bcast(d(noff+1), nlen, MPI_REAL_PRECISION, np, mpi_comm_rows, mpierr)
          qmat2 = qmat1
          call MPI_Bcast(qmat2, max_size*max_size, MPI_REAL_PRECISION, np, mpi_comm_rows, mpierr)
#else /* WITH_MPI */
!          qmat2 = qmat1 ! is this correct
#endif /* WITH_MPI */
          do i=1,nlen

#ifdef WITH_MPI
            call distribute_global_column_&
            &PRECISION &
                     (obj, qmat2(1,i), q(1,noff+i), nqoff+noff, nlen, my_prow, np_rows, nblk)
#else /* WITH_MPI */
            call distribute_global_column_&
            &PRECISION &
                     (obj, qmat1(1,i), q(1,noff+i), nqoff+noff, nlen, my_prow, np_rows, nblk)
#endif /* WITH_MPI */
          enddo

        enddo

        deallocate(qmat1, qmat2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi_col: error when deallocating qmat2 "//errorMessage
          stop 1
        endif

      endif

      ! Allocate and set index arrays l_col and p_col

      allocate(l_col(na), p_col_i(na),  p_col_o(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi_col: error when allocating l_col "//errorMessage
        stop 1
      endif

      do i=1,na
        l_col(i) = i
        p_col_i(i) = 0
        p_col_o(i) = 0
      enddo

      ! Merge subproblems

      n = 1
      do while(n<ndiv) ! if ndiv==1, the problem was solved by single call to solve_tridi_single

        do i=0,ndiv-1,2*n

          noff = limits(i)
          nmid = limits(i+n) - noff
          nlen = limits(i+2*n) - noff

          if (nlen == na) then
            ! Last merge, set p_col_o=-1 for unneeded (output) eigenvectors
            p_col_o(nev+1:na) = -1
          endif
          call merge_systems_&
          &PRECISION &
                              (obj, nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, nqoff+noff, nblk, &
                               matrixCols, mpi_comm_rows, mpi_comm_self, l_col(noff+1), p_col_i(noff+1), &
                               l_col(noff+1), p_col_o(noff+1), 0, 1, useGPU, wantDebug, success, max_threads)
          if (.not.(success)) return

        enddo

        n = 2*n

      enddo

      deallocate(limits, l_col, p_col_i, p_col_o, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi_col: error when deallocating l_col "//errorMessage
        stop 1
      endif


    end subroutine solve_tridi_col_&
    &PRECISION_AND_SUFFIX

    recursive subroutine solve_tridi_single_problem_&
    &PRECISION_AND_SUFFIX &
    (obj, nlen, d, e, q, ldq, wantDebug, success)

   ! Solves the symmetric, tridiagonal eigenvalue problem on a single processor.
   ! Takes precautions if DSTEDC fails or if the eigenvalues are not ordered correctly.
     use precision
     implicit none
!     class(elpa_abstract_impl_t), intent(inout) :: obj
     integer(kind=c_int), intent(in)                                     :: obj
     integer(kind=ik)                         :: nlen, ldq
     real(kind=REAL_DATATYPE)                 :: d(nlen), e(nlen), q(ldq,nlen)

     real(kind=REAL_DATATYPE), allocatable    :: work(:), qtmp(:), ds(:), es(:)
     real(kind=REAL_DATATYPE)                 :: dtmp

     integer(kind=ik)              :: i, j, lwork, liwork, info
     integer(kind=ik), allocatable :: iwork(:)

     logical, intent(in)           :: wantDebug
     logical, intent(out)          :: success
      integer(kind=ik)             :: istat
      character(200)               :: errorMessage


     success = .true.
     allocate(ds(nlen), es(nlen), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_tridi_single: error when allocating ds "//errorMessage
       stop 1
     endif

     ! Save d and e for the case that dstedc fails

     ds(:) = d(:)
     es(:) = e(:)

     ! First try dstedc, this is normally faster but it may fail sometimes (why???)

     lwork = 1 + 4*nlen + nlen**2
     liwork =  3 + 5*nlen
     allocate(work(lwork), iwork(liwork), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_tridi_single: error when allocating work "//errorMessage
       stop 1
     endif
     call PRECISION_STEDC('I', nlen, d, e, q, ldq, work, lwork, iwork, liwork, info)

     if (info /= 0) then

       ! DSTEDC failed, try DSTEQR. The workspace is enough for DSTEQR.


       d(:) = ds(:)
       e(:) = es(:)
       call PRECISION_STEQR('I', nlen, d, e, q, ldq, work, info)

       ! If DSTEQR fails also, we don't know what to do further ...

       if (info /= 0) then
         if (wantDebug) &
           success = .false.
           return
         endif
       end if

       deallocate(work,iwork,ds,es, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_tridi_single: error when deallocating ds "//errorMessage
         stop 1
       endif

      ! Check if eigenvalues are monotonically increasing
      ! This seems to be not always the case  (in the IBM implementation of dstedc ???)

      do i=1,nlen-1
        if (d(i+1)<d(i)) then
#ifdef DOUBLE_PRECISION_REAL
          if (abs(d(i+1) - d(i)) / abs(d(i+1) + d(i)) > 1e-14_rk8) then
#else
          if (abs(d(i+1) - d(i)) / abs(d(i+1) + d(i)) > 1e-14_rk4) then
#endif
            print *,"solve_tridi_single: error when deallocating qtmp "
          else
            print *,"solve_tridi_single: error when deallocating qtmp "
          end if
          allocate(qtmp(nlen), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"solve_tridi_single: error when allocating qtmp "//errorMessage
            stop 1
          endif

          dtmp = d(i+1)
          qtmp(1:nlen) = q(1:nlen,i+1)
          do j=i,1,-1
            if (dtmp<d(j)) then
              d(j+1)        = d(j)
              q(1:nlen,j+1) = q(1:nlen,j)
            else
              exit ! Loop
            endif
          enddo
          d(j+1)        = dtmp
          q(1:nlen,j+1) = qtmp(1:nlen)
          deallocate(qtmp, stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"solve_tridi_single: error when deallocating qtmp "//errorMessage
            stop 1
          endif

       endif
     enddo

    end subroutine solve_tridi_single_problem_&
    &PRECISION_AND_SUFFIX

