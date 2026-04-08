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

    subroutine symm_matrix_allreduce_&
&PRECISION &
                    (obj, n, a, lda, ldb, comm)
    !-------------------------------------------------------------------------------
    !  symm_matrix_allreduce: Does an mpi_allreduce for a symmetric matrix A.
    !  On entry, only the upper half of A needs to be set
    !  On exit, the complete matrix is set
    !-------------------------------------------------------------------------------
      use precision
      implicit none
!      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=c_int), intent(in)                                     :: obj
      integer(kind=ik)             :: n, lda, ldb, comm
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE)     :: a(lda,*)
#else
      real(kind=REAL_DATATYPE)     :: a(lda,ldb)
#endif
      integer(kind=ik)             :: i, nc, mpierr
      real(kind=REAL_DATATYPE)     :: h1(n*n), h2(n*n)


      nc = 0
      do i=1,n
        h1(nc+1:nc+i) = a(1:i,i)
        nc = nc+i
      enddo

#ifdef WITH_MPI
      call mpi_allreduce(h1, h2, nc, MPI_REAL_PRECISION, MPI_SUM, comm, mpierr)
      nc = 0
      do i=1,n
        a(1:i,i) = h2(nc+1:nc+i)
        a(i,1:i-1) = a(1:i-1,i)
        nc = nc+i
      enddo

#else /* WITH_MPI */
!      h2=h1

      nc = 0
      do i=1,n
        a(1:i,i) = h1(nc+1:nc+i)
        a(i,1:i-1) = a(1:i-1,i)
        nc = nc+i
      enddo

#endif /* WITH_MPI */
!      nc = 0
!      do i=1,n
!        a(1:i,i) = h2(nc+1:nc+i)
!        a(i,1:i-1) = a(1:i-1,i)
!        nc = nc+i
!      enddo


    end subroutine symm_matrix_allreduce_&
    &PRECISION



