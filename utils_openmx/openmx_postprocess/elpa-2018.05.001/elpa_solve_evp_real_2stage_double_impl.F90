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

#include "config-f90.h"

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"


 subroutine elpa_solve_evp_&
  &MATH_DATATYPE&
  &_&
  &2stage_&
  &PRECISION&
  &_impl (na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)

   use elpa2_compute_real
   use elpa1_compute_real
   use mpi
#ifdef WITH_OPENMP
   use omp_lib
#endif
   use iso_c_binding
   use precision
   implicit none
#include "precision_kinds.F90"
#include "fortran_constants.F90"
!   class(elpa_abstract_impl_t), intent(inout)                         :: obj
   integer(kind=c_int)                                     :: obj = 1
   integer(kind=ik), intent(in) :: na, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   integer(kind=ik), intent(inout) :: nev
   logical                                                            :: useGPU
#if REALCASE == 1
   logical                                                            :: useQR
   logical                                                            :: useQRActual
#endif
   integer(kind=c_int)                                                :: kernel

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout)                 :: a(lda,*)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(ldq,*)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout)                 :: a(lda,matrixCols)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(ldq,matrixCols)
#endif
   real(kind=C_DATATYPE_KIND), intent(inout)                          :: ev(na)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: hh_trans(:,:)

   integer(kind=c_int)                                                :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer(kind=c_int)                                                :: nbw, num_blocks
#if COMPLEXCASE == 1
   integer(kind=c_int)                                                :: l_cols_nev, l_rows, l_cols
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable                   :: tmat(:,:,:)
   real(kind=C_DATATYPE_KIND), allocatable                            :: e(:)
#if COMPLEXCASE == 1
   real(kind=C_DATATYPE_KIND), allocatable                            :: q_real(:,:)
#endif
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable, target           :: q_dummy(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer                       :: q_actual(:,:)


   integer(kind=c_intptr_t)                                           :: tmat_dev, q_dev, a_dev

   integer(kind=c_int)                                                :: i
   logical                                                            :: success, successCUDA
   logical                                                            :: wantDebug
   logical                                                            :: eigenvalues_only
   integer(kind=c_int)                                                :: istat, gpu, debug, qr
   character(200)                                                     :: errorMessage
   logical                                                            :: do_useGPU, do_useGPU_bandred, &
                                                                         do_useGPU_tridiag_band, do_useGPU_solve_tridi, &
                                                                         do_useGPU_trans_ev_tridi_to_band, &
                                                                         do_useGPU_trans_ev_band_to_full
   integer(kind=c_int)                                                :: numberOfGPUDevices
!   integer(kind=c_intptr_t), parameter                                :: size_of_datatype = size_of_&
!                                                                                            &PRECISION&
!                                                                                            &_&
!                                                                                            &MATH_DATATYPE
    integer(kind=ik)                                                  :: check_pd, error

    logical                                                           :: do_bandred, do_tridiag, do_solve_tridi,  &
                                                                         do_trans_to_band, do_trans_to_full

    integer(kind=ik)                                                  :: nrThreads
#if REALCASE == 1
#undef GPU_KERNEL
#undef GENERIC_KERNEL
#undef KERNEL_STRING
#define GPU_KERNEL ELPA_2STAGE_REAL_GPU
#define GENERIC_KERNEL ELPA_2STAGE_REAL_GENERIC
#define KERNEL_STRING "real_kernel"
#endif
#if COMPLEXCASE == 1
#undef GPU_KERNEL
#undef GENERIC_KERNEL
#undef KERNEL_STRING
#define GPU_KERNEL ELPA_2STAGE_COMPLEX_GPU
#define GENERIC_KERNEL ELPA_2STAGE_COMPLEX_GENERIC
#define KERNEL_STRING "complex_kernel"
#endif

!    print *, "from elpa_solve_evp_real_2stage_double_impl.F90"

#ifdef WITH_OPENMP
    nrThreads = omp_get_max_threads()
    call omp_set_num_threads(nrThreads)
#else
    nrThreads = 1
#endif

    success = .true.

    if (present(q)) then
      eigenvalues_only = .false.
    else
      eigenvalues_only = .true.
    endif

    call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
    call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)


   ! special case na = 1
   if (na .eq. 1) then
#if REALCASE == 1
     ev(1) = a(1,1)
#endif
#if COMPLEXCASE == 1
     ev(1) = real(a(1,1))
#endif
     if (.not.(eigenvalues_only)) then
       q(1,1) = ONE
     endif
     return
   endif

   if (nev == 0) then
     nev = 1
     eigenvalues_only = .true.
   endif


#if REALCASE == 1
    kernel = ELPA_2STAGE_REAL_GENERIC
#endif
#if COMPLEXCASE == 1
    kernel = ELPA_2STAGE_COMPLEX_GENERIC
#endif

    ! GPU settings

    gpu = 0
    useGPU = .false.

    do_useGPU = .false.


    do_useGPU_bandred = do_useGPU
    do_useGPU_tridiag_band = do_useGPU
    do_useGPU_solve_tridi = do_useGPU
    do_useGPU_trans_ev_tridi_to_band = do_useGPU
    do_useGPU_trans_ev_band_to_full = do_useGPU


#if REALCASE == 1
    qr = 0
    if (qr .eq. 1) then
      useQR = .true.
    else
      useQR = .false.
    endif

#endif

    debug = 0
    wantDebug = debug == 1



#if REALCASE == 1
    useQRActual = .false.
    ! set usage of qr decomposition via API call
    if (useQR) useQRActual = .true.
    if (.not.(useQR)) useQRACtual = .false.

    if (useQRActual) then
      if (mod(na,2) .ne. 0) then
        print *, "Do not use QR-decomposition for this matrix and blocksize."
        success = .false.
        return
      endif
    endif
#endif /* REALCASE */



    if (.not. eigenvalues_only) then
      q_actual => q(1:lda,1:matrixCols)
    else
     allocate(q_dummy(1:lda,1:matrixCols))
     q_actual => q_dummy(1:lda,1:matrixCols)
    endif


    ! set the default values for each of the 5 compute steps
    do_bandred        = .true.
    do_tridiag        = .true.
    do_solve_tridi    = .true.
    do_trans_to_band  = .true.
    do_trans_to_full  = .true.

    if (eigenvalues_only) then
      do_trans_to_band  = .false.
      do_trans_to_full  = .false.
    endif

    if(1==1) then ! matrix is not banded, determine the intermediate bandwidth for full->banded->tridi
      !first check if the intermediate bandwidth was set by the user
      nbw = 0
      if(nbw == 0) then
        ! intermediate bandwidth was not specified, select one of the defaults

        ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32
        ! On older systems (IBM Bluegene/P, Intel Nehalem) a value of 32 was optimal.
        ! For Intel(R) Xeon(R) E5 v2 and v3, better use 64 instead of 32!
        ! For IBM Bluegene/Q this is not clear at the moment. We have to keep an eye
        ! on this and maybe allow a run-time optimization here
        if (do_useGPU) then
          nbw = nblk
        else
#if REALCASE == 1
          nbw = (63/nblk+1)*nblk
#elif COMPLEXCASE == 1
          nbw = (31/nblk+1)*nblk
#endif
        endif

      else
        ! intermediate bandwidth has been specified by the user, check, whether correctly
        if (mod(nbw, nblk) .ne. 0) then
          print *, "Specified bandwidth ",nbw," has to be mutiple of the blocksize ", nblk, ". Aborting..."
          success = .false.
          return
        endif
      endif !nbw == 0

      num_blocks = (na-1)/nbw + 1

      ! tmat is needed only in full->band and band->full steps, so alocate here
      ! (not allocated for banded matrix on input)
      allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_evp_&
        &MATH_DATATYPE&
        &_2stage_&
        &PRECISION&
        &" // ": error when allocating tmat "//errorMessage
        stop 1
      endif

      do_bandred       = .true.
      do_solve_tridi   = .true.
      do_trans_to_band = .true.
      do_trans_to_full = .true.
    endif  ! matrix not already banded on input

    ! start the computations in 5 steps


    if (do_bandred) then
      ! Reduction full -> band
      call bandred_&
      &MATH_DATATYPE&
      &_&
      &PRECISION &
      (obj, na, a, &
      a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, tmat, &
      tmat_dev,  wantDebug, do_useGPU_bandred, success, &
#if REALCASE == 1
      useQRActual, &
#endif
       nrThreads)
      if (.not.(success)) return
    endif



     ! Reduction band -> tridiagonal
     if (do_tridiag) then
       allocate(e(na), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when allocating e "//errorMessage
         stop 1
       endif

!       call tridiag_band_&
!       &MATH_DATATYPE&
!       &_&
!       &PRECISION&
!       (obj, na, nbw, nblk, a, a_dev, lda, ev, e, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
!       do_useGPU_tridiag_band, wantDebug, nrThreads)

#if REALCASE == 1
       call tridiag_band_real_double(obj, na, nbw, nblk, a, a_dev, lda, ev, e, &
            matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
            do_useGPU_tridiag_band, wantDebug, nrThreads)
#endif
#if COMPLEXCASE == 1
       call tridiag_band_complex_double(obj, na, nbw, nblk, a, a_dev, lda, ev, e, &
            matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
            do_useGPU_tridiag_band, wantDebug, nrThreads)
#endif


#ifdef WITH_MPI
       call mpi_bcast(ev, na, MPI_REAL_PRECISION, 0, mpi_comm_all, mpierr)
       call mpi_bcast(e, na, MPI_REAL_PRECISION, 0, mpi_comm_all, mpierr)
#endif /* WITH_MPI */
     endif ! do_tridiag

#if COMPLEXCASE == 1
     l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
     l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
     l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

     allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_&
       &MATH_DATATYPE&
       &_2stage: error when allocating q_real"//errorMessage
       stop 1
     endif
#endif


     ! Solve tridiagonal system
     if (do_solve_tridi) then
       call solve_tridi_&
       &PRECISION &
       (obj, na, nev, ev, e, &
#if REALCASE == 1
       q_actual, ldq,   &
#endif
#if COMPLEXCASE == 1
       q_real, ubound(q_real,dim=1), &
#endif
       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, do_useGPU_solve_tridi, wantDebug, success, nrThreads)
       if (.not.(success)) return
     endif ! do_solve_tridi

     deallocate(e, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_&
       &MATH_DATATYPE&
       &_2stage: error when deallocating e "//errorMessage
       stop 1
     endif

     if (eigenvalues_only) then
       do_trans_to_band = .false.
       do_trans_to_full = .false.
     else

        check_pd = 0
       if (check_pd .eq. 1) then
         check_pd = 0
         do i = 1, na
           if (ev(i) .gt. THRESHOLD) then
             check_pd = check_pd + 1
           endif
         enddo
         if (check_pd .lt. na) then
           ! not positiv definite => eigenvectors needed
           do_trans_to_band = .true.
           do_trans_to_full = .true.
         else
           do_trans_to_band = .false.
           do_trans_to_full = .false.
         endif
       endif
     endif ! eigenvalues only

     if (do_trans_to_band) then
#if COMPLEXCASE == 1
       ! q must be given thats why from here on we can use q and not q_actual

       q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

       deallocate(q_real, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage: error when deallocating q_real"//errorMessage
         stop 1
       endif
#endif


       ! Backtransform stage 1
       call trans_ev_tridi_to_band_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, na, nev, nblk, nbw, q, &
       q_dev, &
       ldq, matrixCols, hh_trans, mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU_trans_ev_tridi_to_band, &
       nrThreads, success=success, kernel=kernel)

       if (.not.(success)) return

       ! We can now deallocate the stored householder vectors
       deallocate(hh_trans, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *, "solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when deallocating hh_trans "//errorMessage
         stop 1
       endif
     endif ! do_trans_to_band


     if (do_trans_to_full) then

       ! Backtransform stage 2


       call trans_ev_band_to_full_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
       (obj, na, nev, nblk, nbw, a, &
       a_dev, lda, tmat, tmat_dev,  q,  &
       q_dev, &
       ldq, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev_band_to_full &
#if REALCASE == 1
       , useQRActual  &
#endif
       )


       deallocate(tmat, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_2stage_&
         &PRECISION " // ": error when deallocating tmat"//errorMessage
         stop 1
       endif
     endif ! do_trans_to_full

     if (eigenvalues_only) then
       deallocate(q_dummy, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_evp_&
         &MATH_DATATYPE&
         &_1stage_&
         &PRECISION&
         &" // ": error when deallocating q_dummy "//errorMessage
         stop 1
       endif
     endif

1    format(a,f10.3)

!     print *, "called elpa_solve"

   end subroutine elpa_solve_evp_&
   &MATH_DATATYPE&
   &_2stage_&
   &PRECISION&
   &_impl

! vim: syntax=fortran
