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


#if REALCASE == 1

    subroutine v_add_s_&
    &PRECISION&
    &(obj, v,n,s)
      use precision
      implicit none
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik)            :: n
      real(kind=REAL_DATATYPE)    :: v(n),s

      v(:) = v(:) + s
    end subroutine v_add_s_&
    &PRECISION

    subroutine distribute_global_column_&
    &PRECISION&
    &(obj, g_col, l_col, noff, nlen, my_prow, np_rows, nblk)
      use precision
      implicit none

!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      real(kind=REAL_DATATYPE)     :: g_col(nlen), l_col(*) ! chnage this to proper 2d 1d matching ! remove assumed size
      integer(kind=ik)             :: noff, nlen, my_prow, np_rows, nblk

      integer(kind=ik)  :: nbs, nbe, jb, g_off, l_off, js, je

      nbs = noff/(nblk*np_rows)
      nbe = (noff+nlen-1)/(nblk*np_rows)

      do jb = nbs, nbe

        g_off = jb*nblk*np_rows + nblk*my_prow
        l_off = jb*nblk

        js = MAX(noff+1-g_off,1)
        je = MIN(noff+nlen-g_off,nblk)

        if (je<js) cycle

        l_col(l_off+js:l_off+je) = g_col(g_off+js-noff:g_off+je-noff)

      enddo
    end subroutine distribute_global_column_&
    &PRECISION

    subroutine solve_secular_equation_&
    &PRECISION&
    &(obj, n, i, d, z, delta, rho, dlam)
    !-------------------------------------------------------------------------------
    ! This routine solves the secular equation of a symmetric rank 1 modified
    ! diagonal matrix:
    !
    !    1. + rho*SUM(z(:)**2/(d(:)-x)) = 0
    !
    ! It does the same as the LAPACK routine DLAED4 but it uses a bisection technique
    ! which is more robust (it always yields a solution) but also slower
    ! than the algorithm used in DLAED4.
    !
    ! The same restictions than in DLAED4 hold, namely:
    !
    !   rho > 0   and   d(i+1) > d(i)
    !
    ! but this routine will not terminate with error if these are not satisfied
    ! (it will normally converge to a pole in this case).
    !
    ! The output in DELTA(j) is always (D(j) - lambda_I), even for the cases
    ! N=1 and N=2 which is not compatible with DLAED4.
    ! Thus this routine shouldn't be used for these cases as a simple replacement
    ! of DLAED4.
    !
    ! The arguments are the same as in DLAED4 (with the exception of the INFO argument):
    !
    !
    !  N      (input) INTEGER
    !         The length of all arrays.
    !
    !  I      (input) INTEGER
    !         The index of the eigenvalue to be computed.  1 <= I <= N.
    !
    !  D      (input) DOUBLE PRECISION array, dimension (N)
    !         The original eigenvalues.  It is assumed that they are in
    !         order, D(I) < D(J)  for I < J.
    !
    !  Z      (input) DOUBLE PRECISION array, dimension (N)
    !         The components of the updating Vector.
    !
    !  DELTA  (output) DOUBLE PRECISION array, dimension (N)
    !         DELTA contains (D(j) - lambda_I) in its  j-th component.
    !         See remark above about DLAED4 compatibility!
    !
    !  RHO    (input) DOUBLE PRECISION
    !         The scalar in the symmetric updating formula.
    !
    !  DLAM   (output) DOUBLE PRECISION
    !         The computed lambda_I, the I-th updated eigenvalue.
    !-------------------------------------------------------------------------------

      use precision
      implicit none
#include "precision_kinds.F90"
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik)           :: n, i
      real(kind=REAL_DATATYPE)   :: d(n), z(n), delta(n), rho, dlam

      integer(kind=ik)           :: iter
      real(kind=REAL_DATATYPE)   :: a, b, x, y, dshift

      ! In order to obtain sufficient numerical accuracy we have to shift the problem
      ! either by d(i) or d(i+1), whichever is closer to the solution

      ! Upper and lower bound of the shifted solution interval are a and b

      if (i==n) then

       ! Special case: Last eigenvalue
       ! We shift always by d(n), lower bound is d(n),
       ! upper bound is determined by a guess:

       dshift = d(n)
       delta(:) = d(:) - dshift

       a = 0.0_rk ! delta(n)
       b = rho*SUM(z(:)**2) + 1.0_rk ! rho*SUM(z(:)**2) is the lower bound for the guess
      else

        ! Other eigenvalues: lower bound is d(i), upper bound is d(i+1)
        ! We check the sign of the function in the midpoint of the interval
        ! in order to determine if eigenvalue is more close to d(i) or d(i+1)
        x = 0.5_rk*(d(i)+d(i+1))
        y = 1.0_rk + rho*SUM(z(:)**2/(d(:)-x))
        if (y>0) then
          ! solution is next to d(i)
          dshift = d(i)
        else
          ! solution is next to d(i+1)
          dshift = d(i+1)
        endif

        delta(:) = d(:) - dshift
        a = delta(i)
        b = delta(i+1)

      endif

      ! Bisection:

      do iter=1,200

        ! Interval subdivision
        x = 0.5_rk*(a+b)
        if (x==a .or. x==b) exit   ! No further interval subdivisions possible
#ifdef DOUBLE_PRECISION_REAL
        if (abs(x) < 1.e-200_rk8) exit ! x next to pole
#else
        if (abs(x) < 1.e-20_rk4) exit ! x next to pole
#endif
        ! evaluate value at x

        y = 1. + rho*SUM(z(:)**2/(delta(:)-x))

        if (y==0) then
          ! found exact solution
          exit
        elseif (y>0) then
          b = x
        else
          a = x
        endif

      enddo

      ! Solution:

      dlam = x + dshift
      delta(:) = delta(:) - x

    end subroutine solve_secular_equation_&
    &PRECISION
    !-------------------------------------------------------------------------------
#endif

#if REALCASE == 1
    subroutine hh_transform_real_&
#endif
#if COMPLEXCASE == 1
    subroutine hh_transform_complex_&
#endif
    &PRECISION &
                   (obj, alpha, xnorm_sq, xf, tau, wantDebug)
#if REALCASE  == 1
      ! Similar to LAPACK routine DLARFP, but uses ||x||**2 instead of x(:)
#endif
#if COMPLEXCASE == 1
      ! Similar to LAPACK routine ZLARFP, but uses ||x||**2 instead of x(:)
#endif
      ! and returns the factor xf by which x has to be scaled.
      ! It also hasn't the special handling for numbers < 1.d-300 or > 1.d150
      ! since this would be expensive for the parallel implementation.
      use precision
      implicit none
!      class(elpa_abstract_impl_t), intent(inout)    :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      logical, intent(in)                           :: wantDebug
#if REALCASE == 1
      real(kind=REAL_DATATYPE), intent(inout)       :: alpha
#endif
#if COMPLEXCASE == 1
      complex(kind=COMPLEX_DATATYPE), intent(inout) :: alpha
#endif
      real(kind=REAL_DATATYPE), intent(in)          :: xnorm_sq
#if REALCASE == 1
      real(kind=REAL_DATATYPE), intent(out)         :: xf, tau
#endif
#if COMPLEXCASE == 1
      complex(kind=COMPLEX_DATATYPE), intent(out)   :: xf, tau
      real(kind=REAL_DATATYPE)                      :: ALPHR, ALPHI
#endif

      real(kind=REAL_DATATYPE)                      :: BETA


#if COMPLEXCASE == 1
      ALPHR = real( ALPHA, kind=REAL_DATATYPE )
      ALPHI = PRECISION_IMAG( ALPHA )
#endif

#if REALCASE == 1
      if ( XNORM_SQ==0. ) then
#endif
#if COMPLEXCASE == 1
      if ( XNORM_SQ==0. .AND. ALPHI==0. ) then
#endif

#if REALCASE == 1
        if ( ALPHA>=0. ) then
#endif
#if COMPLEXCASE == 1
        if ( ALPHR>=0. ) then
#endif
          TAU = 0.
        else
          TAU = 2.
          ALPHA = -ALPHA
        endif
        XF = 0.

      else

#if REALCASE == 1
        BETA = SIGN( SQRT( ALPHA**2 + XNORM_SQ ), ALPHA )
#endif
#if COMPLEXCASE == 1
        BETA = SIGN( SQRT( ALPHR**2 + ALPHI**2 + XNORM_SQ ), ALPHR )
#endif
        ALPHA = ALPHA + BETA
        IF ( BETA<0 ) THEN
          BETA = -BETA
          TAU  = -ALPHA / BETA
        ELSE
#if REALCASE == 1
          ALPHA = XNORM_SQ / ALPHA
#endif
#if COMPLEXCASE == 1
          ALPHR = ALPHI * (ALPHI/real( ALPHA , kind=KIND_PRECISION))
          ALPHR = ALPHR + XNORM_SQ/real( ALPHA, kind=KIND_PRECISION )
#endif

#if REALCASE == 1
          TAU = ALPHA / BETA
          ALPHA = -ALPHA
#endif
#if COMPLEXCASE == 1
          TAU = PRECISION_CMPLX( ALPHR/BETA, -ALPHI/BETA )
          ALPHA = PRECISION_CMPLX( -ALPHR, ALPHI )
#endif
       END IF
       XF = 1.0/ALPHA
       ALPHA = BETA
     endif

#if REALCASE == 1
    end subroutine hh_transform_real_&
#endif
#if COMPLEXCASE == 1
    end subroutine hh_transform_complex_&
#endif
    &PRECISION
