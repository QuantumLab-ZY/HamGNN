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
! It should be compiled with the highest possible optimization level.
!
! On Intel use -O3 -xSSE4.2 (or the SSE level fitting to your CPU)
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! --------------------------------------------------------------------------------------------------
#endif

! the intel compiler creates a temp copy of array q!
! this should be prevented if possible without using assumed size arrays
  subroutine double_hh_trafo_&
  &MATH_DATATYPE&
  &_generic_&
  &PRECISION&
  & (q, hh, nb, nq, ldq, ldh)

    use precision
    use iso_c_binding
!    use elpa_abstract_impl
    implicit none

    !class(elpa_abstract_impl_t), intent(inout) :: obj
    integer(kind=ik), intent(in)            :: nb, nq, ldq, ldh
#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+1)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif

    real(kind=C_DATATYPE_KIND)                :: s
    integer(kind=ik)                        :: i

!    equivalence(q(1,1),q_complex(1,1))

    ! Safety only:

!    call obj%timer%start("kernel generic: double_hh_trafo_&
!                     &MATH_DATATYPE&
!		     &_generic" // &
!		     &PRECISION_SUFFIX &
!		     )

    if(mod(ldq,4) /= 0) STOP 'double_hh_trafo: ldq not divisible by 4!'

    ! Calculate dot product of the two Householder vectors

    s = hh(2,2)*1.0
    do i=3,nb
       s = s+hh(i,2)*hh(i-1,1)
    enddo

    ! Do the Householder transformations

#ifndef USE_ASSUMED_SIZE
!    ! assign real data to compplex pointer
!    call c_f_pointer(c_loc(q), q_complex, [size(q,dim=1)/2,size(q,dim=2)])
#endif
    ! Always a multiple of 4 Q-rows is transformed, even if nq is smaller

    do i=1,nq-8,12

#ifdef USE_ASSUMED_SIZE
       call hh_trafo_kernel_12_generic_&
       &PRECISION&
       & (q(i,1),hh, nb, ldq, ldh, s)
#else
       call hh_trafo_kernel_12_generic_&
       &PRECISION&
       & (q(i:ldq,1:nb+1),hh(1:ldh,1:2), nb, ldq, ldh, s)
#endif

    enddo

    ! i > nq-8 now, i.e. at most 8 rows remain

    if(nq-i+1 > 4) then

#ifdef USE_ASSUMED_SIZE
       call hh_trafo_kernel_8_generic_&
       &PRECISION&
       & (q(i,1),hh, nb, ldq, ldh, s)
#else
       call hh_trafo_kernel_8_generic_&
       &PRECISION&
       & (q(i:ldq,1:nb+1), hh(1:ldh,1:2), nb, ldq, ldh, s)
#endif

    else if(nq-i+1 > 0) then

#ifdef USE_ASSUMED_SIZE
       call hh_trafo_kernel_4_generic_&
       &PRECISION&
       & (q(i,1),hh, nb, ldq, ldh, s)
#else
       call hh_trafo_kernel_4_generic_&
       &PRECISION&
       & (q(i:ldq,1:+nb+1),hh(1:ldh,1:2), nb, ldq, ldh, s)
#endif

    endif

!    call obj%timer%stop("kernel generic: double_hh_trafo_&
!                     &MATH_DATATYPE&
!		     &_generic" // &
!		     &PRECISION_SUFFIX &
!		     )

  end subroutine

  ! --------------------------------------------------------------------------------------------------
  ! The following kernels perform the Householder transformation on Q for 12/8/4 rows.
  ! Please note that Q is declared complex*16 here.
  ! This is a hint for compilers that packed arithmetic can be used for Q
  ! (relevant for Intel SSE and BlueGene double hummer CPUs).
  ! --------------------------------------------------------------------------------------------------

  subroutine hh_trafo_kernel_12_generic_&
  &PRECISION&
  & (q, hh, nb, ldq, ldh, s)
    use precision
    implicit none
    integer(kind=ik), intent(in)    :: nb, ldq, ldh
#ifdef USE_ASSUMED_SIZE
#ifdef PACK_REAL_TO_COMPLEX
    complex(kind=SPECIAL_COMPLEX_DATATYPE), intent(inout) :: q(ldq/2,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
#endif
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(:,:)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,2)
#endif
    real(kind=C_DATATYPE_KIND), intent(in)    :: s

#ifdef PACK_REAL_TO_COMPLEX
    complex(kind=SPECIAL_COMPLEX_DATATYPE)    :: x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6
#else
    real(kind=C_DATATYPE_KIND)                :: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, &
                                       y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12
#endif
    real(kind=C_DATATYPE_KIND)                   :: h1, h2, tau1, tau2
    integer(kind=ik)                :: i


    !call obj%timer%start("kernel generic: hh_trafo_kernel_12_generic" // &
    !		     &PRECISION_SUFFIX &
    !		     )

    x1  = q(1,2)
    x2  = q(2,2)
    x3  = q(3,2)
    x4  = q(4,2)
    x5  = q(5,2)
    x6  = q(6,2)
#ifndef PACK_REAL_TO_COMPLEX
    x7  = q(7,2)
    x8  = q(8,2)
    x9  = q(9,2)
    x10 = q(10,2)
    x11 = q(11,2)
    x12 = q(12,2)
#endif

    y1  = q(1 ,1) + q(1, 2)*hh(2,2)
    y2  = q(2 ,1) + q(2, 2)*hh(2,2)
    y3  = q(3 ,1) + q(3, 2)*hh(2,2)
    y4  = q(4 ,1) + q(4, 2)*hh(2,2)
    y5  = q(5 ,1) + q(5, 2)*hh(2,2)
    y6  = q(6 ,1) + q(6, 2)*hh(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    y7  = q(7 ,1) + q(7, 2)*hh(2,2)
    y8  = q(8 ,1) + q(8, 2)*hh(2,2)
    y9  = q(9 ,1) + q(9, 2)*hh(2,2)
    y10 = q(10,1) + q(10,2)*hh(2,2)
    y11 = q(11,1) + q(11,2)*hh(2,2)
    y12 = q(12,1) + q(12,2)*hh(2,2)
#endif

#ifdef DOUBLE_PRECISION_REAL
#if defined(SSE_ALIGNED)
    !DEC$ VECTOR ALIGNED
#endif
#endif

    do i=3,nb
       h1  = hh(i-1,1)
       h2  = hh(i,2)
       x1  = x1 + q(1, i)*h1
       y1  = y1 + q(1, i)*h2
       x2  = x2 + q(2, i)*h1
       y2  = y2 + q(2, i)*h2
       x3  = x3 + q(3, i)*h1
       y3  = y3 + q(3, i)*h2
       x4  = x4 + q(4, i)*h1
       y4  = y4 + q(4, i)*h2
       x5  = x5 + q(5, i)*h1
       y5  = y5 + q(5, i)*h2
       x6  = x6 + q(6, i)*h1
       y6  = y6 + q(6, i)*h2
#ifndef PACK_REAL_TO_COMPLEX
       x7  = x7  + q(7, i)*h1
       y7  = y7  + q(7, i)*h2
       x8  = x8  + q(8, i)*h1
       y8  = y8  + q(8, i)*h2
       x9  = x9  + q(9, i)*h1
       y9  = y9  + q(9, i)*h2
       x10 = x10 + q(10,i)*h1
       y10 = y10 + q(10,i)*h2
       x11 = x11 + q(11,i)*h1
       y11 = y11 + q(11,i)*h2
       x12 = x12 + q(12,i)*h1
       y12 = y12 + q(12,i)*h2
#endif
    enddo

    x1  = x1  + q(1,nb+1)*hh(nb,1)
    x2  = x2  + q(2,nb+1)*hh(nb,1)
    x3  = x3  + q(3,nb+1)*hh(nb,1)
    x4  = x4  + q(4,nb+1)*hh(nb,1)
    x5  = x5  + q(5,nb+1)*hh(nb,1)
    x6  = x6  + q(6,nb+1)*hh(nb,1)
#ifndef PACK_REAL_TO_COMPLEX
    x7  = x7  + q(7, nb+1)*hh(nb,1)
    x8  = x8  + q(8, nb+1)*hh(nb,1)
    x9  = x9  + q(9, nb+1)*hh(nb,1)
    x10 = x10 + q(10,nb+1)*hh(nb,1)
    x11 = x11 + q(11,nb+1)*hh(nb,1)
    x12 = x12 + q(12,nb+1)*hh(nb,1)

#endif

    tau1 = hh(1,1)
    tau2 = hh(1,2)

    h1  = -tau1
    x1  = x1 *h1
    x2  = x2 *h1
    x3  = x3 *h1
    x4  = x4 *h1
    x5  = x5 *h1
    x6  = x6 *h1
#ifndef PACK_REAL_TO_COMPLEX
    x7  = x7 *h1
    x8  = x8 *h1
    x9  = x9 *h1
    x10 = x10*h1
    x11 = x11*h1
    x12 = x12*h1
#endif

    h1  = -tau2
    h2  = -tau2*s
    y1  = y1 *h1 + x1 *h2
    y2  = y2 *h1 + x2 *h2
    y3  = y3 *h1 + x3 *h2
    y4  = y4 *h1 + x4 *h2
    y5  = y5 *h1 + x5 *h2
    y6  = y6 *h1 + x6 *h2
#ifndef PACK_REAL_TO_COMPLEX
    y7  = y7 *h1 + x7 *h2
    y8  = y8 *h1 + x8 *h2
    y9  = y9 *h1 + x9 *h2
    y10 = y10*h1 + x10*h2
    y11 = y11*h1 + x11*h2
    y12 = y12*h1 + x12*h2
#endif
    q(1,1)  = q(1, 1) + y1
    q(2,1)  = q(2, 1) + y2
    q(3,1)  = q(3, 1) + y3
    q(4,1)  = q(4, 1) + y4
    q(5,1)  = q(5, 1) + y5
    q(6,1)  = q(6, 1) + y6
#ifndef PACK_REAL_TO_COMPLEX
    q(7 ,1) = q(7, 1) + y7
    q(8 ,1) = q(8, 1) + y8
    q(9 ,1) = q(9, 1) + y9
    q(10,1) = q(10,1) + y10
    q(11,1) = q(11,1) + y11
    q(12,1) = q(12,1) + y12
#endif

    q(1, 2) = q(1, 2) + x1  + y1 *hh(2,2)
    q(2, 2) = q(2, 2) + x2  + y2 *hh(2,2)
    q(3, 2) = q(3, 2) + x3  + y3 *hh(2,2)
    q(4, 2) = q(4, 2) + x4  + y4 *hh(2,2)
    q(5, 2) = q(5, 2) + x5  + y5 *hh(2,2)
    q(6, 2) = q(6, 2) + x6  + y6 *hh(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    q(7, 2) = q(7, 2) + x7  + y7 *hh(2,2)
    q(8, 2) = q(8, 2) + x8  + y8 *hh(2,2)
    q(9, 2) = q(9, 2) + x9  + y9 *hh(2,2)
    q(10,2) = q(10,2) + x10 + y10*hh(2,2)
    q(11,2) = q(11,2) + x11 + y11*hh(2,2)
    q(12,2) = q(12,2) + x12 + y12*hh(2,2)
#endif

#ifdef DOUBLE_PRECISION_REAL
#if defined(SSE_ALIGNED)
    !DEC$ VECTOR ALIGNED
#endif
#endif

    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       q(1, i) = q(1,i)  + x1 *h1 + y1 *h2
       q(2, i) = q(2,i)  + x2 *h1 + y2 *h2
       q(3, i) = q(3,i)  + x3 *h1 + y3 *h2
       q(4, i) = q(4,i)  + x4 *h1 + y4 *h2
       q(5, i) = q(5,i)  + x5 *h1 + y5 *h2
       q(6, i) = q(6,i)  + x6 *h1 + y6 *h2
#ifndef PACK_REAL_TO_COMPLEX
       q(7, i) = q(7, i) + x7 *h1 + y7 *h2
       q(8, i) = q(8, i) + x8 *h1 + y8 *h2
       q(9, i) = q(9, i) + x9 *h1 + y9 *h2
       q(10,i) = q(10,i) + x10*h1 + y10*h2
       q(11,i) = q(11,i) + x11*h1 + y11*h2
       q(12,i) = q(12,i) + x12*h1 + y12*h2
#endif
    enddo

    q(1, nb+1) = q(1, nb+1) + x1 *hh(nb,1)
    q(2, nb+1) = q(2, nb+1) + x2 *hh(nb,1)
    q(3, nb+1) = q(3, nb+1) + x3 *hh(nb,1)
    q(4, nb+1) = q(4, nb+1) + x4 *hh(nb,1)
    q(5, nb+1) = q(5, nb+1) + x5 *hh(nb,1)
    q(6, nb+1) = q(6, nb+1) + x6 *hh(nb,1)
#ifndef PACK_REAL_TO_COMPLEX
    q(7, nb+1) = q(7, nb+1) + x7 *hh(nb,1)
    q(8, nb+1) = q(8, nb+1) + x8 *hh(nb,1)
    q(9, nb+1) = q(9, nb+1) + x9 *hh(nb,1)
    q(10,nb+1) = q(10,nb+1) + x10*hh(nb,1)
    q(11,nb+1) = q(11,nb+1) + x11*hh(nb,1)
    q(12,nb+1) = q(12,nb+1) + x12*hh(nb,1)
#endif

!    call obj%timer%stop("kernel generic: hh_trafo_kernel_12_generic" // &
!		     &PRECISION_SUFFIX &
!		     )

  end subroutine
  ! --------------------------------------------------------------------------------------------------

  subroutine hh_trafo_kernel_8_generic_&
  &PRECISION&
  & (q, hh, nb, ldq, ldh, s)
    use precision
    implicit none
    integer(kind=ik), intent(in)     :: nb, ldq, ldh
#ifdef USE_ASSUMED_SIZE
#ifdef PACK_REAL_TO_COMPLEX
    complex(kind=SPECIAL_COMPLEX_DATATYPE), intent(inout)  :: q(ldq/2,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout)  :: q(ldq,*)
#endif
    real(kind=C_DATATYPE_KIND), intent(in)     :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND),   intent(inout):: q(:,:)
    real(kind=C_DATATYPE_KIND), intent(in)     :: hh(ldh,2)
#endif
    real(kind=C_DATATYPE_KIND), intent(in)     :: s
#ifdef PACK_REAL_TO_COMPLEX
    complex(kind=SPECIAL_COMPLEX_DATATYPE)     :: x1, x2, x3, x4, y1, y2, y3, y4
#else
    real(kind=C_DATATYPE_KIND)                 :: x1, x2, x3, x4, x5, x6, x7, x8, &
                                        y1, y2, y3, y4, y5, y6, y7, y8
#endif
    real(kind=C_DATATYPE_KIND)                    :: h1, h2, tau1, tau2
    integer(kind=ik)                 :: i

!    call obj%timer%start("kernel generic: hh_trafo_kernel_8_generic" // &
!		     &PRECISION_SUFFIX &
!		     )
    x1 = q(1,2)
    x2 = q(2,2)
    x3 = q(3,2)
    x4 = q(4,2)
#ifndef PACK_REAL_TO_COMPLEX
    x5 = q(5,2)
    x6 = q(6,2)
    x7 = q(7,2)
    x8 = q(8,2)
#endif

    y1 = q(1,1) + q(1,2)*hh(2,2)
    y2 = q(2,1) + q(2,2)*hh(2,2)
    y3 = q(3,1) + q(3,2)*hh(2,2)
    y4 = q(4,1) + q(4,2)*hh(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    y5 = q(5,1) + q(5,2)*hh(2,2)
    y6 = q(6,1) + q(6,2)*hh(2,2)
    y7 = q(7,1) + q(7,2)*hh(2,2)
    y8 = q(8,1) + q(8,2)*hh(2,2)
#endif

#ifdef DOUBLE_PRECISION_REAL
#if defined(SSE_ALIGNED)
    !DEC$ VECTOR ALIGNED
#endif
#endif

    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       x1 = x1 + q(1,i)*h1
       y1 = y1 + q(1,i)*h2
       x2 = x2 + q(2,i)*h1
       y2 = y2 + q(2,i)*h2
       x3 = x3 + q(3,i)*h1
       y3 = y3 + q(3,i)*h2
       x4 = x4 + q(4,i)*h1
       y4 = y4 + q(4,i)*h2
#ifndef PACK_REAL_TO_COMPLEX
       x5 = x5 + q(5,i)*h1
       y5 = y5 + q(5,i)*h2
       x6 = x6 + q(6,i)*h1
       y6 = y6 + q(6,i)*h2
       x7 = x7 + q(7,i)*h1
       y7 = y7 + q(7,i)*h2
       x8 = x8 + q(8,i)*h1
       y8 = y8 + q(8,i)*h2
#endif
    enddo

    x1 = x1 + q(1,nb+1)*hh(nb,1)
    x2 = x2 + q(2,nb+1)*hh(nb,1)
    x3 = x3 + q(3,nb+1)*hh(nb,1)
    x4 = x4 + q(4,nb+1)*hh(nb,1)
#ifndef PACK_REAL_TO_COMPLEX
    x5 = x5 + q(5,nb+1)*hh(nb,1)
    x6 = x6 + q(6,nb+1)*hh(nb,1)
    x7 = x7 + q(7,nb+1)*hh(nb,1)
    x8 = x8 + q(8,nb+1)*hh(nb,1)
#endif

    tau1 = hh(1,1)
    tau2 = hh(1,2)

    h1 = -tau1
    x1 = x1*h1
    x2 = x2*h1
    x3 = x3*h1
    x4 = x4*h1
#ifndef PACK_REAL_TO_COMPLEX
    x5 = x5*h1
    x6 = x6*h1
    x7 = x7*h1
    x8 = x8*h1
#endif
    h1 = -tau2
    h2 = -tau2*s
    y1 = y1*h1 + x1*h2
    y2 = y2*h1 + x2*h2
    y3 = y3*h1 + x3*h2
    y4 = y4*h1 + x4*h2
#ifndef PACK_REAL_TO_COMPLEX
    y5 = y5*h1 + x5*h2
    y6 = y6*h1 + x6*h2
    y7 = y7*h1 + x7*h2
    y8 = y8*h1 + x8*h2
#endif
    q(1,1) = q(1,1) + y1
    q(2,1) = q(2,1) + y2
    q(3,1) = q(3,1) + y3
    q(4,1) = q(4,1) + y4
#ifndef PACK_REAL_TO_COMPLEX
    q(5,1) = q(5,1) + y5
    q(6,1) = q(6,1) + y6
    q(7,1) = q(7,1) + y7
    q(8,1) = q(8,1) + y8
#endif
    q(1,2) = q(1,2) + x1 + y1*hh(2,2)
    q(2,2) = q(2,2) + x2 + y2*hh(2,2)
    q(3,2) = q(3,2) + x3 + y3*hh(2,2)
    q(4,2) = q(4,2) + x4 + y4*hh(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    q(5,2) = q(5,2) + x5 + y5*hh(2,2)
    q(6,2) = q(6,2) + x6 + y6*hh(2,2)
    q(7,2) = q(7,2) + x7 + y7*hh(2,2)
    q(8,2) = q(8,2) + x8 + y8*hh(2,2)
#endif

#ifdef DOUBLE_PRECISION_REAL
#if defined(SSE_ALIGNED)
    !DEC$ VECTOR ALIGNED
#endif
#endif

    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       q(1,i) = q(1,i) + x1*h1 + y1*h2
       q(2,i) = q(2,i) + x2*h1 + y2*h2
       q(3,i) = q(3,i) + x3*h1 + y3*h2
       q(4,i) = q(4,i) + x4*h1 + y4*h2
#ifndef PACK_REAL_TO_COMPLEX
       q(5,i) = q(5,i) + x5*h1 + y5*h2
       q(6,i) = q(6,i) + x6*h1 + y6*h2
       q(7,i) = q(7,i) + x7*h1 + y7*h2
       q(8,i) = q(8,i) + x8*h1 + y8*h2
#endif
    enddo

    q(1,nb+1) = q(1,nb+1) + x1*hh(nb,1)
    q(2,nb+1) = q(2,nb+1) + x2*hh(nb,1)
    q(3,nb+1) = q(3,nb+1) + x3*hh(nb,1)
    q(4,nb+1) = q(4,nb+1) + x4*hh(nb,1)
#ifndef PACK_REAL_TO_COMPLEX
    q(5,nb+1) = q(5,nb+1) + x5*hh(nb,1)
    q(6,nb+1) = q(6,nb+1) + x6*hh(nb,1)
    q(7,nb+1) = q(7,nb+1) + x7*hh(nb,1)
    q(8,nb+1) = q(8,nb+1) + x8*hh(nb,1)
#endif


!    call obj%timer%stop("kernel generic: hh_trafo_kernel_8_generic" // &
!		     &PRECISION_SUFFIX &
!		     )

  end subroutine
  ! --------------------------------------------------------------------------------------------------

  subroutine hh_trafo_kernel_4_generic_&
  &PRECISION&
  & (q, hh, nb, ldq, ldh, s)

    use precision
    implicit none
    integer(kind=ik), intent(in)    :: nb, ldq, ldh
#ifdef USE_ASSUMED_SIZE
#ifdef PACK_REAL_TO_COMPLEX
    complex(kind=SPECIAL_COMPLEX_DATATYPE), intent(inout) :: q(ldq/2,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
#endif
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(:,:) !q(1:ldq/2,1:nb+1)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,2)
#endif
    real(kind=C_DATATYPE_KIND), intent(in)    :: s

#ifdef PACK_REAL_TO_COMPLEX
    complex(kind=SPECIAL_COMPLEX_DATATYPE)    :: x1, x2, y1, y2
#else
    real(kind=C_DATATYPE_KIND)                :: x1, x2, x3, x4, y1, y2, y3, y4
#endif
    real(kind=C_DATATYPE_KIND)                :: h1, h2, tau1, tau2
    integer(kind=ik)                :: i

!    call obj%timer%start("kernel generic: hh_trafo_kernel_4_generic" // &
!		     &PRECISION_SUFFIX &
!		     )
    x1 = q(1,2)
    x2 = q(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    x3 = q(3,2)
    x4 = q(4,2)
#endif

    y1 = q(1,1) + q(1,2)*hh(2,2)
    y2 = q(2,1) + q(2,2)*hh(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    y3 = q(3,1) + q(3,2)*hh(2,2)
    y4 = q(4,1) + q(4,2)*hh(2,2)
#endif

#ifdef DOUBLE_PRECISION_REAL
#if defined(SSE_ALIGNED)
    !DEC$ VECTOR ALIGNED
#endif
#endif

    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       x1 = x1 + q(1,i)*h1
       y1 = y1 + q(1,i)*h2
       x2 = x2 + q(2,i)*h1
       y2 = y2 + q(2,i)*h2
#ifndef PACK_REAL_TO_COMPLEX
       x3 = x3 + q(3,i)*h1
       y3 = y3 + q(3,i)*h2
       x4 = x4 + q(4,i)*h1
       y4 = y4 + q(4,i)*h2
#endif
    enddo

    x1 = x1 + q(1,nb+1)*hh(nb,1)
    x2 = x2 + q(2,nb+1)*hh(nb,1)
#ifndef PACK_REAL_TO_COMPLEX
    x3 = x3 + q(3,nb+1)*hh(nb,1)
    x4 = x4 + q(4,nb+1)*hh(nb,1)
#endif

    tau1 = hh(1,1)
    tau2 = hh(1,2)

    h1 = -tau1
    x1 = x1*h1
    x2 = x2*h1
#ifndef PACK_REAL_TO_COMPLEX
    x3 = x3*h1
    x4 = x4*h1
#endif
    h1 = -tau2
    h2 = -tau2*s
    y1 = y1*h1 + x1*h2
    y2 = y2*h1 + x2*h2
#ifndef PACK_REAL_TO_COMPLEX
    y3 = y3*h1 + x3*h2
    y4 = y4*h1 + x4*h2
#endif

    q(1,1) = q(1,1) + y1
    q(2,1) = q(2,1) + y2
#ifndef PACK_REAL_TO_COMPLEX
    q(3,1) = q(3,1) + y3
    q(4,1) = q(4,1) + y4
#endif
    q(1,2) = q(1,2) + x1 + y1*hh(2,2)
    q(2,2) = q(2,2) + x2 + y2*hh(2,2)
#ifndef PACK_REAL_TO_COMPLEX
    q(3,2) = q(3,2) + x3 + y3*hh(2,2)
    q(4,2) = q(4,2) + x4 + y4*hh(2,2)
#endif

#ifdef DOUBLE_PRECISION_REAL
#if defined(SSE_ALIGNED)
    !DEC$ VECTOR ALIGNED
#endif
#endif
    do i=3,nb
       h1 = hh(i-1,1)
       h2 = hh(i,2)
       q(1,i) = q(1,i) + x1*h1 + y1*h2
       q(2,i) = q(2,i) + x2*h1 + y2*h2
#ifndef PACK_REAL_TO_COMPLEX
       q(3,i) = q(3,i) + x3*h1 + y3*h2
       q(4,i) = q(4,i) + x4*h1 + y4*h2
#endif
    enddo

    q(1,nb+1) = q(1,nb+1) + x1*hh(nb,1)
    q(2,nb+1) = q(2,nb+1) + x2*hh(nb,1)
#ifndef PACK_REAL_TO_COMPLEX
    q(3,nb+1) = q(3,nb+1) + x3*hh(nb,1)
    q(4,nb+1) = q(4,nb+1) + x4*hh(nb,1)
#endif

!    call obj%timer%stop("kernel generic: hh_trafo_kernel_4_generic" // &
!		     &PRECISION_SUFFIX &
!		     )

  end subroutine
