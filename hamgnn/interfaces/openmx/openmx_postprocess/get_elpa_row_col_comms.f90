
!-------------------------------------------------------------------------------

subroutine get_elpa_row_col_comms(mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
! get_elpa_row_col_comms:
! All ELPA routines need MPI communicators for communicating within
! rows or columns of processes, these are set here.
! mpi_comm_rows/mpi_comm_cols can be free'd with MPI_Comm_free if not used any more.
!
!  Parameters
!
!  mpi_comm_global   Global communicator for the calculations (in)
!
!  my_prow           Row coordinate of the calling process in the process grid (in)
!
!  my_pcol           Column coordinate of the calling process in the process grid (in)
!
!  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!
!  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!
!-------------------------------------------------------------------------------

   use ELPA1 

   implicit none

   integer, intent(in)  :: mpi_comm_global, my_prow, my_pcol
   integer, intent(out) :: mpi_comm_rows, mpi_comm_cols

   integer :: mpierr

   ! mpi_comm_rows is used for communicating WITHIN rows, i.e. all processes
   ! having the same column coordinate share one mpi_comm_rows.
   ! So the "color" for splitting is my_pcol and the "key" is my row coordinate.
   ! Analogous for mpi_comm_cols

   call mpi_comm_split(mpi_comm_global,my_pcol,my_prow,mpi_comm_rows,mpierr)
   call mpi_comm_split(mpi_comm_global,my_prow,my_pcol,mpi_comm_cols,mpierr)

end subroutine get_elpa_row_col_comms

!-------------------------------------------------------------------------------

