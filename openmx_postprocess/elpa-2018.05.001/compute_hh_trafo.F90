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
! This file was written by A. Marek, MPCDF
#endif

       subroutine compute_hh_trafo_&
       &MATH_DATATYPE&
#ifdef WITH_OPENMP
       &_openmp_&
#else
       &_&
#endif
       &PRECISION &
       (obj, useGPU, wantDebug, a, a_dev, stripe_width, a_dim2, stripe_count, max_threads, &
#ifdef WITH_OPENMP
       l_nev, &
#endif
       a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
#if REALCASE == 1
       hh_dot_dev, &
#endif
       hh_tau_dev, kernel_flops, kernel_time, n_times, off, ncols, istripe, &
#ifdef WITH_OPENMP
       my_thread, thread_width, &
#else
       last_stripe_width, &
#endif
       kernel)

         use precision
         use iso_c_binding

#if REALCASE == 1

         use single_hh_trafo_real

#if defined(WITH_REAL_GENERIC_KERNEL) && !(defined(USE_ASSUMED_SIZE))
         use real_generic_kernel !, only : double_hh_trafo_generic
#endif

#endif /* REALCASE */

#if COMPLEXCASE == 1

#if defined(WITH_COMPLEX_GENERIC_KERNEL) && !(defined(USE_ASSUMED_SIZE))
           use complex_generic_kernel !, only : single_hh_trafo_complex_generic
#endif

#endif /* COMPLEXCASE */


         implicit none
#include "fortran_constants.F90"
!         class(elpa_abstract_impl_t), intent(inout) :: obj
         integer(kind=c_int), intent(in)                                     :: obj
         logical, intent(in)                        :: useGPU, wantDebug
         real(kind=c_double), intent(inout)         :: kernel_time  ! MPI_WTIME always needs double
         integer(kind=lik)                          :: kernel_flops
         integer(kind=ik), intent(in)               :: nbw, max_blk_size
#if REALCASE == 1
         real(kind=C_DATATYPE_KIND)                 :: bcast_buffer(nbw,max_blk_size)
#endif
#if COMPLEXCASE == 1
         complex(kind=C_DATATYPE_KIND)              :: bcast_buffer(nbw,max_blk_size)
#endif
         integer(kind=ik), intent(in)               :: a_off

         integer(kind=ik), intent(in)               :: stripe_width,a_dim2,stripe_count

         integer(kind=ik), intent(in)               :: max_threads
#ifndef WITH_OPENMP
         integer(kind=ik), intent(in)               :: last_stripe_width
#if REALCASE == 1
!         real(kind=C_DATATYPE_KIND)                :: a(stripe_width,a_dim2,stripe_count)
         real(kind=C_DATATYPE_KIND), pointer        :: a(:,:,:)
#endif
#if COMPLEXCASE == 1
!          complex(kind=C_DATATYPE_KIND)            :: a(stripe_width,a_dim2,stripe_count)
          complex(kind=C_DATATYPE_KIND),pointer     :: a(:,:,:)
#endif

#else /* WITH_OPENMP */
         integer(kind=ik), intent(in)               :: l_nev, thread_width
#if REALCASE == 1
!         real(kind=C_DATATYPE_KIND)                :: a(stripe_width,a_dim2,stripe_count,max_threads)
         real(kind=C_DATATYPE_KIND), pointer        :: a(:,:,:,:)
#endif
#if COMPLEXCASE == 1
!          complex(kind=C_DATATYPE_KIND)            :: a(stripe_width,a_dim2,stripe_count,max_threads)
          complex(kind=C_DATATYPE_KIND),pointer     :: a(:,:,:,:)
#endif

#endif /* WITH_OPENMP */

         integer(kind=ik), intent(in)               :: kernel

         integer(kind=c_intptr_t)                   :: a_dev
   integer(kind=c_intptr_t)                         :: bcast_buffer_dev
#if REALCASE == 1
         integer(kind=c_intptr_t)                   :: hh_dot_dev ! why not needed in complex case
#endif
         integer(kind=c_intptr_t)                   :: hh_tau_dev
         integer(kind=c_intptr_t)                   :: dev_offset, dev_offset_1, dev_offset_2

         ! Private variables in OMP regions (my_thread) should better be in the argument list!
         integer(kind=ik)                           :: off, ncols, istripe
#ifdef WITH_OPENMP
         integer(kind=ik)                           :: my_thread, noff
#endif
         integer(kind=ik)                           :: j, nl, jj, jjj, n_times
#if REALCASE == 1
         real(kind=C_DATATYPE_KIND)                 :: w(nbw,6)
#endif
#if COMPLEXCASE == 1
         complex(kind=C_DATATYPE_KIND)              :: w(nbw,2)
#endif
         real(kind=c_double)                        :: ttt ! MPI_WTIME always needs double


         j = -99



#ifndef WITH_OPENMP
         nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
#else /* WITH_OPENMP */

         if (istripe<stripe_count) then
           nl = stripe_width
         else
           noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
           nl = min(my_thread*thread_width-noff, l_nev-noff)
           if (nl<=0) then

             return
           endif
         endif
#endif /* not WITH_OPENMP */






#if REALCASE == 1
! GPU kernel real
         if (kernel .eq. ELPA_2STAGE_REAL_GPU) then

#endif /* REALCASE */
#if COMPLEXCASE == 1
! GPU kernel complex
         if (kernel .eq. ELPA_2STAGE_COMPLEX_GPU) then

#endif /* COMPLEXCASE */

         else ! not CUDA kernel

#if REALCASE == 1
#ifndef WITH_FIXED_REAL_KERNEL
         if (kernel .eq. ELPA_2STAGE_REAL_AVX_BLOCK2 .or. &
             kernel .eq. ELPA_2STAGE_REAL_AVX2_BLOCK2 .or. &
             kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
             kernel .eq. ELPA_2STAGE_REAL_SSE_BLOCK2 .or. &
             kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK2 .or. &
             kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK2 .or. &
             kernel .eq. ELPA_2STAGE_REAL_GENERIC    .or. &
             kernel .eq. ELPA_2STAGE_REAL_GENERIC_SIMPLE .or. &
             kernel .eq. ELPA_2STAGE_REAL_SSE_ASSEMBLY .or. &
             kernel .eq. ELPA_2STAGE_REAL_BGP .or.        &
             kernel .eq. ELPA_2STAGE_REAL_BGQ) then
#endif /* not WITH_FIXED_REAL_KERNEL */

#endif /* REALCASE */
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

             !FORTRAN CODE / X86 INRINISIC CODE / BG ASSEMBLER USING 2 HOUSEHOLDER VECTORS
#if REALCASE == 1
! generic kernel real case
#if defined(WITH_REAL_GENERIC_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
             if (kernel .eq. ELPA_2STAGE_REAL_GENERIC) then
#endif /* not WITH_FIXED_REAL_KERNEL */

               do j = ncols, 2, -2
                 w(:,1) = bcast_buffer(1:nbw,j+off)
                 w(:,2) = bcast_buffer(1:nbw,j+off-1)

#ifdef WITH_OPENMP

#ifdef USE_ASSUMED_SIZE
                 call double_hh_trafo_&
                      &MATH_DATATYPE&
                      &_generic_&
                      &PRECISION&
                      & (a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)

#else
                 call double_hh_trafo_&
                      &MATH_DATATYPE&
                      &_generic_&
                      &PRECISION&
                      & (a(1:stripe_width,j+off+a_off-1:j+off+a_off+nbw-1, istripe,my_thread), w(1:nbw,1:6), &
                    nbw, nl, stripe_width, nbw)
#endif

#else /* WITH_OPENMP */

#ifdef USE_ASSUMED_SIZE
                 call double_hh_trafo_&
                      &MATH_DATATYPE&
                      &_generic_&
                      &PRECISION&
                      & (a(1,j+off+a_off-1,istripe),w, nbw, nl, stripe_width, nbw)

#else
                 call double_hh_trafo_&
                      &MATH_DATATYPE&
                      &_generic_&
                      &PRECISION&
                      & (a(1:stripe_width,j+off+a_off-1:j+off+a_off+nbw-1,istripe),w(1:nbw,1:6), nbw, nl, stripe_width, nbw)
#endif
#endif /* WITH_OPENMP */

               enddo

#ifndef WITH_FIXED_REAL_KERNEL
             endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_KERNEL */

#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
! generic kernel complex case
#if defined(WITH_COMPLEX_GENERIC_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
           if (kernel .eq. ELPA_2STAGE_COMPLEX_GENERIC .or. &
               kernel .eq. ELPA_2STAGE_COMPLEX_BGP .or. &
               kernel .eq. ELPA_2STAGE_COMPLEX_BGQ ) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
             ttt = mpi_wtime()
             do j = ncols, 1, -1
#ifdef WITH_OPENMP
#ifdef USE_ASSUMED_SIZE

              call single_hh_trafo_&
                   &MATH_DATATYPE&
                   &_generic_&
                   &PRECISION&
                   & (a(1,j+off+a_off,istripe,my_thread), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
              call single_hh_trafo_&
                   &MATH_DATATYPE&
                   &_generic_&
                   &PRECISION&
                   & (a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe,my_thread), &
                     bcast_buffer(1:nbw,j+off), nbw, nl, stripe_width)
#endif

#else /* WITH_OPENMP */

#ifdef USE_ASSUMED_SIZE
              call single_hh_trafo_&
                   &MATH_DATATYPE&
                   &_generic_&
                   &PRECISION&
                   & (a(1,j+off+a_off,istripe), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
              call single_hh_trafo_&
                   &MATH_DATATYPE&
                   &_generic_&
                   &PRECISION&
                   & (a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,j+off), &
                      nbw, nl, stripe_width)
#endif
#endif /* WITH_OPENMP */

            enddo
#ifndef WITH_FIXED_COMPLEX_KERNEL
          endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_GENERIC .or. kernel .eq. ELPA_2STAGE_COMPLEX_BGP .or. kernel .eq. ELPA_2STAGE_COMPLEX_BGQ )
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_GENERIC_KERNEL */

#endif /* COMPLEXCASE */




#if REALCASE == 1
#ifdef WITH_OPENMP
             if (j==1) call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_cpu_openmp_&
                 &PRECISION&
                 & (a(1:stripe_width, 1+off+a_off:1+off+a_off+nbw-1,istripe,my_thread), &
                          bcast_buffer(1:nbw,off+1), nbw, nl,stripe_width)
#else
             if (j==1) call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_cpu_&
                 &PRECISION&
                 & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl,&
                          stripe_width)
#endif

#endif /* REALCASE == 1 */


#if REALCASE == 1
#ifndef WITH_FIXED_REAL_KERNEL
           endif !
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* REALCASE == 1 */






         endif ! GPU_KERNEL

#ifdef WITH_OPENMP
         if (my_thread==1) then
#endif
           kernel_flops = kernel_flops + 4*int(nl,8)*int(ncols,8)*int(nbw,8)
           kernel_time = kernel_time + mpi_wtime()-ttt
     n_times = n_times + 1
#ifdef WITH_OPENMP
         endif
#endif

       end subroutine

! vim: syntax=fortran
