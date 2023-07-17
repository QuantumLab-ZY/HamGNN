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
!
! --------------------------------------------------------------------------------------------------
!
! This file contains the compute intensive kernels for the Householder transformations.
!
! This is the small and simple version (no hand unrolling of loops etc.) but for some
! compilers this performs better than a sophisticated version with transformed and unrolled loops.
!
! It should be compiled with the highest possible optimization level.
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! --------------------------------------------------------------------------------------------------
#endif


  subroutine double_hh_trafo_&
  &MATH_DATATYPE&
  &_generic_simple_&
  &PRECISION&
  & (q, hh, nb, nq, ldq, ldh)

    use precision
!    use elpa_abstract_impl
    implicit none

    !class(elpa_abstract_impl_t), intent(inout) :: obj
    integer(kind=ik), intent(in)    :: nb, nq, ldq, ldh
#if REALCASE==1

#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+1)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif
    real(kind=C_DATATYPE_KIND)                :: s, h1, h2, tau1, tau2, x(nq), y(nq)
#endif /* REALCASE==1 */

#if COMPLEXCASE==1

#ifdef USE_ASSUMED_SIZE
    complex(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    complex(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    complex(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+1)
    complex(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:2)
#endif
    complex(kind=C_DATATYPE_KIND)                :: s, h1, h2, tau1, tau2, x(nq), y(nq)
#endif /* COMPLEXCASE==1 */
    integer(kind=ik)                :: i

    !call obj%timer%start("kernel_&
    !&MATH_DATATYPE&
    !&_generic_simple: double_hh_trafo_&
    !&MATH_DATATYPE&
    !&_generic_simple" // &
    !&PRECISION_SUFFIX &
    !)

    ! Calculate dot product of the two Householder vectors
#if REALCASE==1
    s = hh(2,2)*1.0
    do i=3,nb
       s = s+hh(i,2)*hh(i-1,1)
    enddo
#endif

#if COMPLEXCASE==1
    s = conjg(hh(2,2))*1.0
    do i=3,nb
       s = s+(conjg(hh(i,2))*hh(i-1,1))
    enddo
#endif

    ! Do the Householder transformations

    x(1:nq) = q(1:nq,2)
#if REALCASE==1
    y(1:nq) = q(1:nq,1) + q(1:nq,2)*hh(2,2)
#endif

#if COMPLEXCASE==1
    y(1:nq) = q(1:nq,1) + q(1:nq,2)*conjg(hh(2,2))
#endif

    do i=3,nb
#if REALCASE==1
       h1 = hh(i-1,1)
       h2 = hh(i,2)
#endif

#if COMPLEXCASE==1
       h1 = conjg(hh(i-1,1))
       h2 = conjg(hh(i,2))
#endif
       x(1:nq) = x(1:nq) + q(1:nq,i)*h1
       y(1:nq) = y(1:nq) + q(1:nq,i)*h2
    enddo

#if REALCASE==1
    x(1:nq) = x(1:nq) + q(1:nq,nb+1)*hh(nb,1)
#endif

#if COMPLEXCASE==1
    x(1:nq) = x(1:nq) + q(1:nq,nb+1)*conjg(hh(nb,1))
#endif
    tau1 = hh(1,1)
    tau2 = hh(1,2)

    h1 = -tau1
    x(1:nq) = x(1:nq)*h1
    h1 = -tau2
    h2 = -tau2*s
    y(1:nq) = y(1:nq)*h1 + x(1:nq)*h2

    q(1:nq,1) = q(1:nq,1) + y(1:nq)
    q(1:nq,2) = q(1:nq,2) + x(1:nq) + y(1:nq)*hh(2,2)

    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       q(1:nq,i) = q(1:nq,i) + x(1:nq)*h1 + y(1:nq)*h2
    enddo

    q(1:nq,nb+1) = q(1:nq,nb+1) + x(1:nq)*hh(nb,1)


    !call obj%timer%stop("kernel_&
    !&MATH_DATATYPE&
    !&_generic_simple: double_hh_trafo_&
    !&MATH_DATATYPE&
    !&_generic_simple" // &
    !&PRECISION_SUFFIX &
    !)

  end subroutine
