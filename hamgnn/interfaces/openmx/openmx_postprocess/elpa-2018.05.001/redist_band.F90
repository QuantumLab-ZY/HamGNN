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
! Author: Andreas Marek, MPCDF
#endif
! --------------------------------------------------------------------------------------------------
! redist_band: redistributes band from 2D block cyclic form to 1D band

#include "config-f90.h"

subroutine redist_band_&
&MATH_DATATYPE&
&_&
&PRECISION &
           (obj, a_mat, a_dev, lda, na, nblk, nbw, matrixCols, mpi_comm_rows, mpi_comm_cols, communicator, ab, useGPU)

   use elpa2_workload
   use precision
   use iso_c_binding
   use elpa_utilities, only : local_index
   use mpi
   implicit none

!   class(elpa_abstract_impl_t), intent(inout)       :: obj
   integer(kind=c_int), intent(in)                                     :: obj
   logical, intent(in)                              :: useGPU
   integer(kind=ik), intent(in)                     :: lda, na, nblk, nbw, matrixCols, mpi_comm_rows, mpi_comm_cols, communicator
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(in)  :: a_mat(lda, matrixCols)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(out) :: ab(:,:)

   integer(kind=ik), allocatable                    :: ncnt_s(:), nstart_s(:), ncnt_r(:), nstart_r(:), &
                                                       global_id(:,:), global_id_tmp(:,:), block_limits(:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable :: sbuf(:,:,:), rbuf(:,:,:), buf(:,:)

   integer(kind=ik)                                 :: i, j, my_pe, n_pes, my_prow, np_rows, my_pcol, np_cols, &
                                                       nfact, np, npr, npc, mpierr, is, js
   integer(kind=ik)                                 :: nblocks_total, il, jl, l_rows, l_cols, n_off

   logical                                          :: successCUDA
   integer(kind=c_intptr_t)                         :: a_dev
!   integer(kind=c_intptr_t), parameter              :: size_of_datatype = size_of_&
!                                                                        &PRECISION&
!                                                                        &_&
!                                                                        &MATH_DATATYPE

   call mpi_comm_rank(communicator,my_pe,mpierr)
   call mpi_comm_size(communicator,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1))
#ifdef WITH_OPENMP
   allocate(global_id_tmp(0:np_rows-1,0:np_cols-1))
#endif
   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe
#ifdef WITH_MPI
#ifdef WITH_OPENMP
   global_id_tmp(:,:) = global_id(:,:)
   call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, communicator, mpierr)
   deallocate(global_id_tmp)
#else
   call mpi_allreduce(mpi_in_place, global_id, np_rows*np_cols, mpi_integer, mpi_sum, communicator, mpierr)
#endif
#endif /* WITH_MPI */
   ! Set work distribution

   nblocks_total = (na-1)/nbw + 1

   allocate(block_limits(0:n_pes))
   call divide_band(obj, nblocks_total, n_pes, block_limits)


   allocate(ncnt_s(0:n_pes-1))
   allocate(nstart_s(0:n_pes-1))
   allocate(ncnt_r(0:n_pes-1))
   allocate(nstart_r(0:n_pes-1))


   nfact = nbw/nblk

   ! Count how many blocks go to which PE

   ncnt_s(:) = 0
   np = 0 ! receiver PE number
   do j=0,(na-1)/nblk ! loop over rows of blocks
     if (j/nfact==block_limits(np+1)) np = np+1
     if (mod(j,np_rows) == my_prow) then
       do i=0,nfact
         if (mod(i+j,np_cols) == my_pcol) then
           ncnt_s(np) = ncnt_s(np) + 1
         endif
       enddo
     endif
   enddo

   ! Allocate send buffer

   allocate(sbuf(nblk,nblk,sum(ncnt_s)))
   sbuf(:,:,:) = 0.

   ! Determine start offsets in send buffer

   nstart_s(0) = 0
   do i=1,n_pes-1
     nstart_s(i) = nstart_s(i-1) + ncnt_s(i-1)
   enddo

   ! Fill send buffer

   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a_mat
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of a_mat

   np = 0
   do j=0,(na-1)/nblk ! loop over rows of blocks
     if (j/nfact==block_limits(np+1)) np = np+1
     if (mod(j,np_rows) == my_prow) then
       do i=0,nfact
         if (mod(i+j,np_cols) == my_pcol) then
           nstart_s(np) = nstart_s(np) + 1
           js = (j/np_rows)*nblk
           is = ((i+j)/np_cols)*nblk
           jl = MIN(nblk,l_rows-js)
           il = MIN(nblk,l_cols-is)

           sbuf(1:jl,1:il,nstart_s(np)) = a_mat(js+1:js+jl,is+1:is+il)
         endif
       enddo
      endif
   enddo

   ! Count how many blocks we get from which PE

   ncnt_r(:) = 0
   do j=block_limits(my_pe)*nfact,min(block_limits(my_pe+1)*nfact-1,(na-1)/nblk)
     npr = mod(j,np_rows)
     do i=0,nfact
       npc = mod(i+j,np_cols)
       np = global_id(npr,npc)
       ncnt_r(np) = ncnt_r(np) + 1
     enddo
   enddo

   ! Allocate receive buffer

   allocate(rbuf(nblk,nblk,sum(ncnt_r)))

   ! Set send counts/send offsets, receive counts/receive offsets
   ! now actually in variables, not in blocks

   ncnt_s(:) = ncnt_s(:)*nblk*nblk

   nstart_s(0) = 0
   do i=1,n_pes-1
     nstart_s(i) = nstart_s(i-1) + ncnt_s(i-1)
   enddo

   ncnt_r(:) = ncnt_r(:)*nblk*nblk

   nstart_r(0) = 0
   do i=1,n_pes-1
     nstart_r(i) = nstart_r(i-1) + ncnt_r(i-1)
   enddo

   ! Exchange all data with MPI_Alltoallv
#ifdef WITH_MPI

    call MPI_Alltoallv(sbuf, ncnt_s, nstart_s, MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       rbuf, ncnt_r, nstart_r, MPI_MATH_DATATYPE_PRECISION_EXPL, communicator, mpierr)

#else /* WITH_MPI */
    rbuf = sbuf
#endif /* WITH_MPI */

! set band from receive buffer

   ncnt_r(:) = ncnt_r(:)/(nblk*nblk)

   nstart_r(0) = 0
   do i=1,n_pes-1
     nstart_r(i) = nstart_r(i-1) + ncnt_r(i-1)
   enddo

   allocate(buf((nfact+1)*nblk,nblk))

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nbw

   do j=block_limits(my_pe)*nfact,min(block_limits(my_pe+1)*nfact-1,(na-1)/nblk)
     npr = mod(j,np_rows)
     do i=0,nfact
       npc = mod(i+j,np_cols)
       np = global_id(npr,npc)
       nstart_r(np) = nstart_r(np) + 1
#if REALCASE==1
       buf(i*nblk+1:i*nblk+nblk,:) = transpose(rbuf(:,:,nstart_r(np)))
#endif
#if COMPLEXCASE==1
       buf(i*nblk+1:i*nblk+nblk,:) = conjg(transpose(rbuf(:,:,nstart_r(np))))
#endif
     enddo
     do i=1,MIN(nblk,na-j*nblk)
       ab(1:nbw+1,i+j*nblk-n_off) = buf(i:i+nbw,i)
     enddo
   enddo

   deallocate(ncnt_s, nstart_s)
   deallocate(ncnt_r, nstart_r)
   deallocate(global_id)
   deallocate(block_limits)

   deallocate(sbuf, rbuf, buf)


end subroutine

