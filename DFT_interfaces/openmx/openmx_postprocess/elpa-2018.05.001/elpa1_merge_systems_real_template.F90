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

    subroutine merge_systems_&
    &PRECISION &
                         (obj, na, nm, d, e, q, ldq, nqoff, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                          l_col, p_col, l_col_out, p_col_out, npc_0, npc_n, useGPU, wantDebug, success, max_threads)
      use iso_c_binding
      use precision
#ifdef WITH_OPENMP
      use omp_lib
#endif
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout)  :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik), intent(in)                :: na, nm, ldq, nqoff, nblk, matrixCols, mpi_comm_rows, &
                                                     mpi_comm_cols, npc_0, npc_n
      integer(kind=ik), intent(in)                :: l_col(na), p_col(na), l_col_out(na), p_col_out(na)
      real(kind=REAL_DATATYPE), intent(inout)     :: d(na), e
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE), intent(inout)     :: q(ldq,*)
#else
      real(kind=REAL_DATATYPE), intent(inout)     :: q(ldq,matrixCols)
#endif
      logical, intent(in)                         :: useGPU, wantDebug
      logical, intent(out)                        :: success

      ! TODO: play with max_strip. If it was larger, matrices being multiplied
      ! might be larger as well!
      integer(kind=ik), parameter                 :: max_strip=128

      real(kind=REAL_DATATYPE)                    :: PRECISION_LAMCH, PRECISION_LAPY2
      real(kind=REAL_DATATYPE)                    :: beta, sig, s, c, t, tau, rho, eps, tol, &
                                                     qtrans(2,2), dmax, zmax, d1new, d2new
      real(kind=REAL_DATATYPE)                    :: z(na), d1(na), d2(na), z1(na), delta(na),  &
                                                     dbase(na), ddiff(na), ev_scale(na), tmp(na)
      real(kind=REAL_DATATYPE)                    :: d1u(na), zu(na), d1l(na), zl(na)
      real(kind=REAL_DATATYPE), allocatable       :: qtmp1(:,:), qtmp2(:,:), ev(:,:)
#ifdef WITH_OPENMP
      real(kind=REAL_DATATYPE), allocatable       :: z_p(:,:)
#endif

      integer(kind=ik)                            :: i, j, na1, na2, l_rows, l_cols, l_rqs, l_rqe, &
                                                     l_rqm, ns, info
      integer(kind=ik)                            :: l_rnm, nnzu, nnzl, ndef, ncnt, max_local_cols, &
                                                     l_cols_qreorg, np, l_idx, nqcols1, nqcols2
      integer(kind=ik)                            :: my_proc, n_procs, my_prow, my_pcol, np_rows, &
                                                     np_cols, mpierr
      integer(kind=ik)                            :: np_next, np_prev, np_rem
      integer(kind=ik)                            :: idx(na), idx1(na), idx2(na)
      integer(kind=ik)                            :: coltyp(na), idxq1(na), idxq2(na)

      integer(kind=ik)                            :: istat
      character(200)                              :: errorMessage
      integer(kind=ik)                            :: gemm_dim_k, gemm_dim_l, gemm_dim_m

      integer(kind=C_intptr_T)                    :: qtmp1_dev, qtmp2_dev, ev_dev
      logical                                     :: successCUDA
!      integer(kind=c_intptr_t), parameter         :: size_of_datatype = size_of_&
!                                                                      &PRECISION&
!                                                                      &_real
      integer(kind=ik), intent(in)                :: max_threads
#ifdef WITH_OPENMP
      integer(kind=ik)                            :: my_thread

      allocate(z_p(na,0:max_threads-1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"merge_systems: error when allocating z_p "//errorMessage
        stop 1
      endif
#endif

      success = .true.
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      ! If my processor column isn't in the requested set, do nothing

      if (my_pcol<npc_0 .or. my_pcol>=npc_0+npc_n) then
        return
      endif
      ! Determine number of "next" and "prev" column for ring sends

      if (my_pcol == npc_0+npc_n-1) then
        np_next = npc_0
      else
        np_next = my_pcol + 1
      endif

      if (my_pcol == npc_0) then
        np_prev = npc_0+npc_n-1
      else
        np_prev = my_pcol - 1
      endif
      call check_monotony_&
      &PRECISION&
      &(obj, nm,d,'Input1',wantDebug, success)
      if (.not.(success)) then
        return
      endif
      call check_monotony_&
      &PRECISION&
      &(obj,na-nm,d(nm+1),'Input2',wantDebug, success)
      if (.not.(success)) then
        return
      endif
      ! Get global number of processors and my processor number.
      ! Please note that my_proc does not need to match any real processor number,
      ! it is just used for load balancing some loops.

      n_procs = np_rows*npc_n
      my_proc = my_prow*npc_n + (my_pcol-npc_0) ! Row major


      ! Local limits of the rows of Q

      l_rqs = local_index(nqoff+1 , my_prow, np_rows, nblk, +1) ! First row of Q
      l_rqm = local_index(nqoff+nm, my_prow, np_rows, nblk, -1) ! Last row <= nm
      l_rqe = local_index(nqoff+na, my_prow, np_rows, nblk, -1) ! Last row of Q

      l_rnm  = l_rqm-l_rqs+1 ! Number of local rows <= nm
      l_rows = l_rqe-l_rqs+1 ! Total number of local rows


      ! My number of local columns

      l_cols = COUNT(p_col(1:na)==my_pcol)

      ! Get max number of local columns

      max_local_cols = 0
      do np = npc_0, npc_0+npc_n-1
        max_local_cols = MAX(max_local_cols,COUNT(p_col(1:na)==np))
      enddo

      ! Calculations start here

      beta = abs(e)
      sig  = sign(1.0_rk,e)

      ! Calculate rank-1 modifier z

      z(:) = 0

      if (MOD((nqoff+nm-1)/nblk,np_rows)==my_prow) then
        ! nm is local on my row
        do i = 1, na
          if (p_col(i)==my_pcol) z(i) = q(l_rqm,l_col(i))
         enddo
      endif

      if (MOD((nqoff+nm)/nblk,np_rows)==my_prow) then
        ! nm+1 is local on my row
        do i = 1, na
          if (p_col(i)==my_pcol) z(i) = z(i) + sig*q(l_rqm+1,l_col(i))
        enddo
      endif

      call global_gather_&
      &PRECISION&
      &(obj, z, na)
      ! Normalize z so that norm(z) = 1.  Since z is the concatenation of
      ! two normalized vectors, norm2(z) = sqrt(2).
      z = z/sqrt(2.0_rk)
      rho = 2.0_rk*beta
      ! Calculate index for merging both systems by ascending eigenvalues
      call PRECISION_LAMRG( nm, na-nm, d, 1, 1, idx )

! Calculate the allowable deflation tolerance

      zmax = maxval(abs(z))
      dmax = maxval(abs(d))
      EPS = PRECISION_LAMCH( 'Epsilon' )
      TOL = 8.0_rk*EPS*MAX(dmax,zmax)

      ! If the rank-1 modifier is small enough, no more needs to be done
      ! except to reorganize D and Q

      IF ( RHO*zmax <= TOL ) THEN

        ! Rearrange eigenvalues

        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo

        ! Rearrange eigenvectors
        call resort_ev_&
        &PRECISION &
                       (obj, idx, na)


        return
      ENDIF

      ! Merge and deflate system

      na1 = 0
      na2 = 0

      ! COLTYP:
      ! 1 : non-zero in the upper half only;
      ! 2 : dense;
      ! 3 : non-zero in the lower half only;
      ! 4 : deflated.

      coltyp(1:nm) = 1
      coltyp(nm+1:na) = 3

      do i=1,na

        if (rho*abs(z(idx(i))) <= tol) then

          ! Deflate due to small z component.

          na2 = na2+1
          d2(na2)   = d(idx(i))
          idx2(na2) = idx(i)
          coltyp(idx(i)) = 4

        else if (na1>0) then

          ! Check if eigenvalues are close enough to allow deflation.

          S = Z(idx(i))
          C = Z1(na1)

          ! Find sqrt(a**2+b**2) without overflow or
          ! destructive underflow.
          TAU = PRECISION_LAPY2( C, S )
          T = D1(na1) - D(idx(i))
          C = C / TAU
          S = -S / TAU
          IF ( ABS( T*C*S ) <= TOL ) THEN

            ! Deflation is possible.

            na2 = na2+1

            Z1(na1) = TAU

            d2new = D(idx(i))*C**2 + D1(na1)*S**2
            d1new = D(idx(i))*S**2 + D1(na1)*C**2

            ! D(idx(i)) >= D1(na1) and C**2 + S**2 == 1.0
            ! This means that after the above transformation it must be
            !    D1(na1) <= d1new <= D(idx(i))
            !    D1(na1) <= d2new <= D(idx(i))
            !
            ! D1(na1) may get bigger but it is still smaller than the next D(idx(i+1))
            ! so there is no problem with sorting here.
            ! d2new <= D(idx(i)) which means that it might be smaller than D2(na2-1)
            ! which makes a check (and possibly a resort) necessary.
            !
            ! The above relations may not hold exactly due to numeric differences
            ! so they have to be enforced in order not to get troubles with sorting.


            if (d1new<D1(na1)  ) d1new = D1(na1)
            if (d1new>D(idx(i))) d1new = D(idx(i))

            if (d2new<D1(na1)  ) d2new = D1(na1)
            if (d2new>D(idx(i))) d2new = D(idx(i))

            D1(na1) = d1new

            do j=na2-1,1,-1
              if (d2new<d2(j)) then
                d2(j+1)   = d2(j)
                idx2(j+1) = idx2(j)
              else
                exit ! Loop
              endif
            enddo

            d2(j+1)   = d2new
            idx2(j+1) = idx(i)

            qtrans(1,1) = C; qtrans(1,2) =-S
            qtrans(2,1) = S; qtrans(2,2) = C
            call transform_columns_&
            &PRECISION &
                        (obj, idx(i), idx1(na1))
            if (coltyp(idx(i))==1 .and. coltyp(idx1(na1))/=1) coltyp(idx1(na1)) = 2
            if (coltyp(idx(i))==3 .and. coltyp(idx1(na1))/=3) coltyp(idx1(na1)) = 2

            coltyp(idx(i)) = 4

          else
            na1 = na1+1
            d1(na1) = d(idx(i))
            z1(na1) = z(idx(i))
            idx1(na1) = idx(i)
          endif
        else
          na1 = na1+1
          d1(na1) = d(idx(i))
          z1(na1) = z(idx(i))
          idx1(na1) = idx(i)
        endif

      enddo
      call check_monotony_&
      &PRECISION&
      &(obj, na1,d1,'Sorted1', wantDebug, success)
      if (.not.(success)) then
        return
      endif
      call check_monotony_&
      &PRECISION&
      &(obj, na2,d2,'Sorted2', wantDebug, success)
      if (.not.(success)) then
        return
      endif

      if (na1==1 .or. na1==2) then
        ! if(my_proc==0) print *,'--- Remark solve_tridi: na1==',na1,' proc==',myid

        if (na1==1) then
          d(1) = d1(1) + rho*z1(1)**2 ! solve secular equation
        else ! na1==2
          call PRECISION_LAED5(1, d1, z1, qtrans(1,1), rho, d(1))
          call PRECISION_LAED5(2, d1, z1, qtrans(1,2), rho, d(2))
          call transform_columns_&
          &PRECISION&
          &(obj, idx1(1), idx1(2))
        endif

        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        ! Calculate arrangement of all eigenvalues  in output
        call PRECISION_LAMRG( na1, na-na1, d, 1, 1, idx )
        ! Rearrange eigenvalues

        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo

        ! Rearrange eigenvectors

        do i=1,na
          if (idx(i)<=na1) then
            idxq1(i) = idx1(idx(i))
          else
            idxq1(i) = idx2(idx(i)-na1)
          endif
        enddo
        call resort_ev_&
        &PRECISION&
        &(obj, idxq1, na)
      else if (na1>2) then

        ! Solve secular equation

        z(1:na1) = 1
#ifdef WITH_OPENMP
        z_p(1:na1,:) = 1
#endif
        dbase(1:na1) = 0
        ddiff(1:na1) = 0

        info = 0
#ifdef WITH_OPENMP

!$OMP PARALLEL PRIVATE(i,my_thread,delta,s,info,j)
        my_thread = omp_get_thread_num()
!$OMP DO
#endif
        DO i = my_proc+1, na1, n_procs ! work distributed over all processors
          call PRECISION_LAED4(na1, i, d1, z1, delta, rho, s, info) ! s is not used!
          if (info/=0) then
            ! If DLAED4 fails (may happen especially for LAPACK versions before 3.2)
            ! use the more stable bisection algorithm in solve_secular_equation
            ! print *,'ERROR DLAED4 n=',na1,'i=',i,' Using Bisection'
            call solve_secular_equation_&
            &PRECISION&
            &(obj, na1, i, d1, z1, delta, rho, s)
          endif

          ! Compute updated z

#ifdef WITH_OPENMP
          do j=1,na1
            if (i/=j)  z_p(j,my_thread) = z_p(j,my_thread)*( delta(j) / (d1(j)-d1(i)) )
          enddo
          z_p(i,my_thread) = z_p(i,my_thread)*delta(i)
#else
          do j=1,na1
            if (i/=j)  z(j) = z(j)*( delta(j) / (d1(j)-d1(i)) )
          enddo
          z(i) = z(i)*delta(i)
#endif
          ! store dbase/ddiff

          if (i<na1) then
            if (abs(delta(i+1)) < abs(delta(i))) then
              dbase(i) = d1(i+1)
              ddiff(i) = delta(i+1)
            else
              dbase(i) = d1(i)
              ddiff(i) = delta(i)
            endif
          else
            dbase(i) = d1(i)
            ddiff(i) = delta(i)
          endif
        enddo
#ifdef WITH_OPENMP
!$OMP END PARALLEL


        do i = 0, max_threads-1
          z(1:na1) = z(1:na1)*z_p(1:na1,i)
        enddo
#endif

        call global_product_&
        &PRECISION&
        (obj, z, na1)
        z(1:na1) = SIGN( SQRT( -z(1:na1) ), z1(1:na1) )

        call global_gather_&
        &PRECISION&
        &(obj, dbase, na1)
        call global_gather_&
        &PRECISION&
        &(obj, ddiff, na1)
        d(1:na1) = dbase(1:na1) - ddiff(1:na1)

        ! Calculate scale factors for eigenvectors
        ev_scale(:) = 0.0_rk

#ifdef WITH_OPENMP


!$OMP PARALLEL DO PRIVATE(i) SHARED(na1, my_proc, n_procs,  &
!$OMP d1,dbase, ddiff, z, ev_scale, obj) &
!$OMP DEFAULT(NONE)

#endif
        DO i = my_proc+1, na1, n_procs ! work distributed over all processors

          ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
          ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

          ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
          ! in exactly this order, but we want to prevent compiler optimization
!         ev_scale_val = ev_scale(i)
          call add_tmp_&
          &PRECISION&
          &(obj, d1, dbase, ddiff, z, ev_scale(i), na1,i)
!         ev_scale(i) = ev_scale_val
        enddo
#ifdef WITH_OPENMP
!$OMP END PARALLEL DO


#endif

        call global_gather_&
        &PRECISION&
        &(obj, ev_scale, na1)
        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        ! Calculate arrangement of all eigenvalues  in output
        call PRECISION_LAMRG( na1, na-na1, d, 1, 1, idx )
        ! Rearrange eigenvalues
        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo
        call check_monotony_&
        &PRECISION&
        &(obj, na,d,'Output', wantDebug, success)

        if (.not.(success)) then
          return
        endif
        ! Eigenvector calculations


        ! Calculate the number of columns in the new local matrix Q
        ! which are updated from non-deflated/deflated eigenvectors.
        ! idxq1/2 stores the global column numbers.

        nqcols1 = 0 ! number of non-deflated eigenvectors
        nqcols2 = 0 ! number of deflated eigenvectors
        DO i = 1, na
          if (p_col_out(i)==my_pcol) then
            if (idx(i)<=na1) then
              nqcols1 = nqcols1+1
              idxq1(nqcols1) = i
            else
              nqcols2 = nqcols2+1
              idxq2(nqcols2) = i
            endif
          endif
        enddo

        gemm_dim_k = MAX(1,l_rows)
        gemm_dim_l = max_local_cols
        gemm_dim_m = MIN(max_strip,MAX(1,nqcols1))

        allocate(qtmp1(gemm_dim_k, gemm_dim_l), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when allocating qtmp1 "//errorMessage
          stop 1
        endif

        allocate(ev(gemm_dim_l,gemm_dim_m), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when allocating ev "//errorMessage
          stop 1
        endif


        allocate(qtmp2(gemm_dim_k, gemm_dim_m), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when allocating qtmp2 "//errorMessage
          stop 1
        endif

        qtmp1 = 0 ! May contain empty (unset) parts
        qtmp2 = 0 ! Not really needed


        ! Gather nonzero upper/lower components of old matrix Q
        ! which are needed for multiplication with new eigenvectors

        nnzu = 0
        nnzl = 0
        do i = 1, na1
          l_idx = l_col(idx1(i))
          if (p_col(idx1(i))==my_pcol) then
            if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
              nnzu = nnzu+1
              qtmp1(1:l_rnm,nnzu) = q(l_rqs:l_rqm,l_idx)
            endif
            if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
              nnzl = nnzl+1
              qtmp1(l_rnm+1:l_rows,nnzl) = q(l_rqm+1:l_rqe,l_idx)
            endif
          endif
        enddo

        ! Gather deflated eigenvalues behind nonzero components

        ndef = max(nnzu,nnzl)
        do i = 1, na2
          l_idx = l_col(idx2(i))
          if (p_col(idx2(i))==my_pcol) then
            ndef = ndef+1
            qtmp1(1:l_rows,ndef) = q(l_rqs:l_rqe,l_idx)
          endif
        enddo

        l_cols_qreorg = ndef ! Number of columns in reorganized matrix

        ! Set (output) Q to 0, it will sum up new Q

        DO i = 1, na
          if(p_col_out(i)==my_pcol) q(l_rqs:l_rqe,l_col_out(i)) = 0
        enddo

        np_rem = my_pcol

        do np = 1, npc_n
          ! Do a ring send of qtmp1

          if (np>1) then

            if (np_rem==npc_0) then
              np_rem = npc_0+npc_n-1
            else
              np_rem = np_rem-1
            endif
#ifdef WITH_MPI
            call MPI_Sendrecv_replace(qtmp1, l_rows*max_local_cols, MPI_REAL_PRECISION, &
                                        np_next, 1111, np_prev, 1111, &
                                        mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#endif /* WITH_MPI */
          endif

          ! Gather the parts in d1 and z which are fitting to qtmp1.
          ! This also delivers nnzu/nnzl for proc np_rem

          nnzu = 0
          nnzl = 0
          do i=1,na1
            if (p_col(idx1(i))==np_rem) then
              if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
                nnzu = nnzu+1
                d1u(nnzu) = d1(i)
                zu (nnzu) = z (i)
              endif
              if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
                nnzl = nnzl+1
                d1l(nnzl) = d1(i)
                zl (nnzl) = z (i)
              endif
            endif
          enddo

          ! Set the deflated eigenvectors in Q (comming from proc np_rem)

          ndef = MAX(nnzu,nnzl) ! Remote counter in input matrix
          do i = 1, na
            j = idx(i)
            if (j>na1) then
              if (p_col(idx2(j-na1))==np_rem) then
                ndef = ndef+1
                if (p_col_out(i)==my_pcol) &
                      q(l_rqs:l_rqe,l_col_out(i)) = qtmp1(1:l_rows,ndef)
              endif
            endif
          enddo

          do ns = 0, nqcols1-1, max_strip ! strimining loop

            ncnt = MIN(max_strip,nqcols1-ns) ! number of columns in this strip

            ! Get partial result from (output) Q

            do i = 1, ncnt
              qtmp2(1:l_rows,i) = q(l_rqs:l_rqe,l_col_out(idxq1(i+ns)))
            enddo

            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with upper half of Q:

            do i = 1, ncnt
              j = idx(idxq1(i+ns))
              ! Calculate the j-th eigenvector of the deflated system
              ! See above why we are doing it this way!
              tmp(1:nnzu) = d1u(1:nnzu)-dbase(j)
              call v_add_s_&
              &PRECISION&
              &(obj,tmp,nnzu,ddiff(j))
              ev(1:nnzu,i) = zu(1:nnzu) / tmp(1:nnzu) * ev_scale(j)
            enddo

            ! Multiply old Q with eigenvectors (upper half)

            if (l_rnm>0 .and. ncnt>0 .and. nnzu>0) then
              if (useGPU) then
                 print *,"merge_systems: error when allocating qtmp2 wwith useGPU"
                 stop 1
              else
                call PRECISION_GEMM('N', 'N', l_rnm, ncnt, nnzu,   &
                                    1.0_rk, qtmp1, ubound(qtmp1,dim=1),    &
                                    ev, ubound(ev,dim=1), &
                                    1.0_rk, qtmp2(1,1), ubound(qtmp2,dim=1))
              endif ! useGPU
            endif


            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with lower half of Q:

            do i = 1, ncnt
              j = idx(idxq1(i+ns))
              ! Calculate the j-th eigenvector of the deflated system
              ! See above why we are doing it this way!
              tmp(1:nnzl) = d1l(1:nnzl)-dbase(j)
              call v_add_s_&
              &PRECISION&
              &(obj,tmp,nnzl,ddiff(j))
              ev(1:nnzl,i) = zl(1:nnzl) / tmp(1:nnzl) * ev_scale(j)
            enddo


            ! Multiply old Q with eigenvectors (lower half)

            if (l_rows-l_rnm>0 .and. ncnt>0 .and. nnzl>0) then
              if (useGPU) then
                 print *,"merge_systems: error when allocating qtmp2 "
                 stop 1

              else
                call PRECISION_GEMM('N', 'N', l_rows-l_rnm, ncnt, nnzl,   &
                                     1.0_rk, qtmp1(l_rnm+1,1), ubound(qtmp1,dim=1),    &
                                     ev,  ubound(ev,dim=1),   &
                                     1.0_rk, qtmp2(l_rnm+1,1), ubound(qtmp2,dim=1))
              endif ! useGPU
            endif


             ! Put partial result into (output) Q

            do i = 1, ncnt
              q(l_rqs:l_rqe,l_col_out(idxq1(i+ns))) = qtmp2(1:l_rows,i)
            enddo

          enddo   !ns = 0, nqcols1-1, max_strip ! strimining loop
        enddo    !do np = 1, npc_n

        deallocate(ev, qtmp1, qtmp2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when deallocating ev "//errorMessage
          stop 1
        endif

      endif !very outer test (na1==1 .or. na1==2) 
#ifdef WITH_OPENMP
      deallocate(z_p, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"merge_systems: error when deallocating z_p "//errorMessage
        stop 1
      endif
#endif


      return

      contains
        subroutine add_tmp_&
        &PRECISION&
        &(obj, d1, dbase, ddiff, z, ev_scale_value, na1,i)
          use precision
          implicit none
!          class(elpa_abstract_impl_t), intent(inout) :: obj
          integer(kind=c_int), intent(in)                                     :: obj
          integer(kind=ik), intent(in) :: na1, i

          real(kind=REAL_DATATYPE), intent(in)    :: d1(:), dbase(:), ddiff(:), z(:)
          real(kind=REAL_DATATYPE), intent(inout) :: ev_scale_value
          real(kind=REAL_DATATYPE)                :: tmp(1:na1)

               ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
               ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

               ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
               ! in exactly this order, but we want to prevent compiler optimization

          tmp(1:na1) = d1(1:na1) -dbase(i)
          call v_add_s_&
          &PRECISION&
          &(obj, tmp(1:na1),na1,ddiff(i))
          tmp(1:na1) = z(1:na1) / tmp(1:na1)
          ev_scale_value = 1.0_rk/sqrt(dot_product(tmp(1:na1),tmp(1:na1)))

        end subroutine add_tmp_&
        &PRECISION

        subroutine resort_ev_&
        &PRECISION&
        &(obj, idx_ev, nLength)
          use precision
          implicit none
!          class(elpa_abstract_impl_t), intent(inout) :: obj
          integer(kind=c_int), intent(in)                                     :: obj
          integer(kind=ik), intent(in) :: nLength
          integer(kind=ik)             :: idx_ev(nLength)
          integer(kind=ik)             :: i, nc, pc1, pc2, lc1, lc2, l_cols_out

          real(kind=REAL_DATATYPE), allocatable   :: qtmp(:,:)
          integer(kind=ik)             :: istat
          character(200)               :: errorMessage

          if (l_rows==0) return ! My processor column has no work to do

          ! Resorts eigenvectors so that q_new(:,i) = q_old(:,idx_ev(i))

          l_cols_out = COUNT(p_col_out(1:na)==my_pcol)
          allocate(qtmp(l_rows,l_cols_out), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"resort_ev: error when allocating qtmp "//errorMessage
            stop 1
          endif

          nc = 0

          do i=1,na

            pc1 = p_col(idx_ev(i))
            lc1 = l_col(idx_ev(i))
            pc2 = p_col_out(i)

            if (pc2<0) cycle ! This column is not needed in output

            if (pc2==my_pcol) nc = nc+1 ! Counter for output columns

            if (pc1==my_pcol) then
              if (pc2==my_pcol) then
                ! send and recieve column are local
                qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
              else
#ifdef WITH_MPI
                call mpi_send(q(l_rqs,lc1), l_rows, MPI_REAL_PRECISION, pc2, mod(i,4096), mpi_comm_cols, mpierr)
#endif /* WITH_MPI */
              endif
            else if (pc2==my_pcol) then
#ifdef WITH_MPI
              call mpi_recv(qtmp(1,nc), l_rows, MPI_REAL_PRECISION, pc1, mod(i,4096), mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */
              qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
#endif /* WITH_MPI */
            endif
          enddo

          ! Insert qtmp into (output) q

          nc = 0

          do i=1,na

            pc2 = p_col_out(i)
            lc2 = l_col_out(i)

            if (pc2==my_pcol) then
              nc = nc+1
              q(l_rqs:l_rqe,lc2) = qtmp(1:l_rows,nc)
            endif
          enddo

          deallocate(qtmp, stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"resort_ev: error when deallocating qtmp "//errorMessage
            stop 1
          endif
        end subroutine resort_ev_&
        &PRECISION

        subroutine transform_columns_&
        &PRECISION&
        &(obj, col1, col2)
          use precision
          implicit none
!          class(elpa_abstract_impl_t), intent(inout) :: obj
          integer(kind=c_int), intent(in)                                     :: obj

          integer(kind=ik)           :: col1, col2
          integer(kind=ik)           :: pc1, pc2, lc1, lc2

          if (l_rows==0) return ! My processor column has no work to do

          pc1 = p_col(col1)
          lc1 = l_col(col1)
          pc2 = p_col(col2)
          lc2 = l_col(col2)

          if (pc1==my_pcol) then
            if (pc2==my_pcol) then
              ! both columns are local
              tmp(1:l_rows)      = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + q(l_rqs:l_rqe,lc2)*qtrans(2,1)
              q(l_rqs:l_rqe,lc2) = q(l_rqs:l_rqe,lc1)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
              q(l_rqs:l_rqe,lc1) = tmp(1:l_rows)
            else
#ifdef WITH_MPI
              call mpi_sendrecv(q(l_rqs,lc1), l_rows, MPI_REAL_PRECISION, pc2, 1, &
                                tmp, l_rows, MPI_REAL_PRECISION, pc2, 1,          &
                                mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */
              tmp(1:l_rows) = q(l_rqs:l_rqe,lc1)
#endif /* WITH_MPI */
              q(l_rqs:l_rqe,lc1) = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + tmp(1:l_rows)*qtrans(2,1)
            endif
          else if (pc2==my_pcol) then
#ifdef WITH_MPI
            call mpi_sendrecv(q(l_rqs,lc2), l_rows, MPI_REAL_PRECISION, pc1, 1, &
                               tmp, l_rows, MPI_REAL_PRECISION, pc1, 1,         &
                               mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else /* WITH_MPI */
            tmp(1:l_rows) = q(l_rqs:l_rqe,lc2)
#endif /* WITH_MPI */

            q(l_rqs:l_rqe,lc2) = tmp(1:l_rows)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
          endif
        end subroutine transform_columns_&
        &PRECISION

        subroutine global_gather_&
        &PRECISION&
        &(obj, z, n)
          ! This routine sums up z over all processors.
          ! It should only be used for gathering distributed results,
          ! i.e. z(i) should be nonzero on exactly 1 processor column,
          ! otherways the results may be numerically different on different columns
          use precision
          implicit none
!          class(elpa_abstract_impl_t), intent(inout) :: obj
          integer(kind=c_int), intent(in)                                     :: obj
          integer(kind=ik)            :: n
          real(kind=REAL_DATATYPE)    :: z(n)
          real(kind=REAL_DATATYPE)    :: tmp(n)

          if (npc_n==1 .and. np_rows==1) return ! nothing to do

          ! Do an mpi_allreduce over processor rows
#ifdef WITH_MPI
          call mpi_allreduce(z, tmp, n, MPI_REAL_PRECISION, MPI_SUM, mpi_comm_rows, mpierr)
#else /* WITH_MPI */
          tmp = z
#endif /* WITH_MPI */
          ! If only 1 processor column, we are done
          if (npc_n==1) then
            z(:) = tmp(:)
            return
          endif

          ! If all processor columns are involved, we can use mpi_allreduce
          if (npc_n==np_cols) then
#ifdef WITH_MPI
            call mpi_allreduce(tmp, z, n, MPI_REAL_PRECISION, MPI_SUM, mpi_comm_cols, mpierr)
#else /* WITH_MPI */
            tmp = z
#endif /* WITH_MPI */

            return
          endif

          ! Do a ring send over processor columns
          z(:) = 0
          do np = 1, npc_n
            z(:) = z(:) + tmp(:)
#ifdef WITH_MPI
            call MPI_Sendrecv_replace(z, n, MPI_REAL_PRECISION, np_next, 1111, np_prev, 1111, &
                                       mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#endif /* WITH_MPI */
          enddo
        end subroutine global_gather_&
        &PRECISION

        subroutine global_product_&
        &PRECISION&
        &(obj, z, n)
          ! This routine calculates the global product of z.
          use precision
          implicit none
!          class(elpa_abstract_impl_t), intent(inout) :: obj
          integer(kind=c_int), intent(in)                                     :: obj


          integer(kind=ik)            :: n
          real(kind=REAL_DATATYPE)    :: z(n)

          real(kind=REAL_DATATYPE)    :: tmp(n)

          if (npc_n==1 .and. np_rows==1) return ! nothing to do

          ! Do an mpi_allreduce over processor rows
#ifdef WITH_MPI
          call mpi_allreduce(z, tmp, n, MPI_REAL_PRECISION, MPI_PROD, mpi_comm_rows, mpierr)
#else /* WITH_MPI */
          tmp = z
#endif /* WITH_MPI */
          ! If only 1 processor column, we are done
          if (npc_n==1) then
            z(:) = tmp(:)
            return
          endif

          ! If all processor columns are involved, we can use mpi_allreduce
          if (npc_n==np_cols) then
#ifdef WITH_MPI
            call mpi_allreduce(tmp, z, n, MPI_REAL_PRECISION, MPI_PROD, mpi_comm_cols, mpierr)
#else /* WITH_MPI */
            z = tmp
#endif /* WITH_MPI */
            return
          endif

          ! We send all vectors to the first proc, do the product there
          ! and redistribute the result.

          if (my_pcol == npc_0) then
            z(1:n) = tmp(1:n)
            do np = npc_0+1, npc_0+npc_n-1
#ifdef WITH_MPI
              call mpi_recv(tmp, n, MPI_REAL_PRECISION, np, 1111, mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else  /* WITH_MPI */
              tmp(1:n) = z(1:n)
#endif  /* WITH_MPI */
              z(1:n) = z(1:n)*tmp(1:n)
            enddo
            do np = npc_0+1, npc_0+npc_n-1
#ifdef WITH_MPI
              call mpi_send(z, n, MPI_REAL_PRECISION, np, 1111, mpi_comm_cols, mpierr)
#endif  /* WITH_MPI */
            enddo
          else
#ifdef WITH_MPI
            call mpi_send(tmp, n, MPI_REAL_PRECISION, npc_0, 1111, mpi_comm_cols, mpierr)
            call mpi_recv(z  ,n, MPI_REAL_PRECISION, npc_0, 1111, mpi_comm_cols, MPI_STATUS_IGNORE, mpierr)
#else  /* WITH_MPI */
            z(1:n) = tmp(1:n)
#endif  /* WITH_MPI */

          endif
        end subroutine global_product_&
        &PRECISION

        subroutine check_monotony_&
        &PRECISION&
        &(obj, n,d,text, wantDebug, success)
        ! This is a test routine for checking if the eigenvalues are monotonically increasing.
        ! It is for debug purposes only, an error should never be triggered!
          use precision
          implicit none

!          class(elpa_abstract_impl_t), intent(inout) :: obj
          integer(kind=c_int), intent(in)                                     :: obj
          integer(kind=ik)              :: n
          real(kind=REAL_DATATYPE)      :: d(n)
          character*(*)                 :: text

          integer(kind=ik)              :: i
          logical, intent(in)           :: wantDebug
          logical, intent(out)          :: success

          success = .true.
          do i=1,n-1
            if (d(i+1)<d(i)) then
              if (wantDebug) write(error_unit,'(a,a,i8,2g25.17)') 'ELPA1_check_monotony: Monotony error on ',text,i,d(i),d(i+1)
              success = .false.
              return
            endif
          enddo
        end subroutine check_monotony_&
        &PRECISION

    end subroutine merge_systems_&
    &PRECISION
