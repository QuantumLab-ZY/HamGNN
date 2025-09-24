
!-------------------------------------------------------------------------------

subroutine solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  solve_evp_real: Solves the real eigenvalue problem
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed.
!              The smallest nev eigenvalues/eigenvectors are calculated.
!
!  a(lda,*)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,*)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------

   use ELPA1

   implicit none

   include 'mpif.h'

   integer, intent(in) :: na, nev, lda, ldq, nblk, mpi_comm_rows, mpi_comm_cols
   real*8 :: a(lda,*), ev(na), q(ldq,*)

   integer my_prow, my_pcol, mpierr, i,j
   real*8, allocatable :: e(:), tau(:)
   real*8 ttt0, ttt1,sum1,sum2

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)

   allocate(e(na), tau(na))

   ttt0 = MPI_Wtime()
   call tridiag_real(na, a, lda, nblk, mpi_comm_rows, mpi_comm_cols, ev, e, tau)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) print *,'Time tridiag_real :',ttt1-ttt0
   time_evp_fwd = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) print *,'Time solve_tridi  :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call trans_ev_real(na, nev, a, lda, tau, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) print *,'Time trans_ev_real:',ttt1-ttt0
   time_evp_back = ttt1-ttt0

   deallocate(e, tau)

end subroutine solve_evp_real

!-------------------------------------------------------------------------------

