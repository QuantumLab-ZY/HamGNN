  !Processor col for global col number
  pure function pcol(global_col, nblk, np_cols) result(local_col)
    use iso_c_binding, only : c_int
    implicit none
    integer(kind=c_int), intent(in) :: global_col, nblk, np_cols
    integer(kind=c_int)             :: local_col
    local_col = MOD((global_col-1)/nblk,np_cols)
  end function

  !Processor row for global row number
  pure function prow(global_row, nblk, np_rows) result(local_row)
    use iso_c_binding, only : c_int
    implicit none
    integer(kind=c_int), intent(in) :: global_row, nblk, np_rows
    integer(kind=c_int)             :: local_row
    local_row = MOD((global_row-1)/nblk,np_rows)
  end function

