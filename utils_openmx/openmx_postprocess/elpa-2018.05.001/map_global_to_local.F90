
 function map_global_array_index_to_local_index(iGLobal, jGlobal, iLocal, jLocal , nblk, np_rows, np_cols, my_prow, my_pcol) &
   result(possible)
   use iso_c_binding, only : c_int
   implicit none

   integer(kind=c_int)              :: pi, pj, li, lj, xi, xj
   integer(kind=c_int), intent(in)  :: iGlobal, jGlobal, nblk, np_rows, np_cols, my_prow, my_pcol
   integer(kind=c_int), intent(out) :: iLocal, jLocal
   logical                       :: possible

   possible = .true.
   iLocal = 0
   jLocal = 0

   pi = prow(iGlobal, nblk, np_rows)

   if (my_prow .ne. pi) then
     possible = .false.
     return
   endif

   pj = pcol(jGlobal, nblk, np_cols)

   if (my_pcol .ne. pj) then
     possible = .false.
     return
   endif
   li = (iGlobal-1)/(np_rows*nblk) ! block number for rows
   lj = (jGlobal-1)/(np_cols*nblk) ! block number for columns

   xi = mod( (iGlobal-1),nblk)+1   ! offset in block li
   xj = mod( (jGlobal-1),nblk)+1   ! offset in block lj

   iLocal = li * nblk + xi
   jLocal = lj * nblk + xj

 end function

