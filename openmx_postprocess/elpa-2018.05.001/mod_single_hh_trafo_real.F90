module single_hh_trafo_real
  implicit none
#include "config-f90.h"

#ifdef WITH_OPENMP
  public single_hh_trafo_real_cpu_openmp_double
#else
  public single_hh_trafo_real_cpu_double
#endif

  contains

#ifdef WITH_OPENMP
    subroutine single_hh_trafo_real_cpu_openmp_double(q, hh, nb, nq, ldq)
#else
    subroutine single_hh_trafo_real_cpu_double(q, hh, nb, nq, ldq)
#endif

!      use elpa_abstract_impl
      use precision
      ! Perform single real Householder transformation.
      ! This routine is not performance critical and thus it is coded here in Fortran

      implicit none
 !     class(elpa_abstract_impl_t), intent(inout) :: obj

      integer(kind=ik), intent(in)   :: nb, nq, ldq
!      real(kind=rk8), intent(inout)   :: q(ldq, *)
!      real(kind=rk8), intent(in)      :: hh(*)
      real(kind=rk8), intent(inout)   :: q(1:ldq, 1:nb)
      real(kind=rk8), intent(in)      :: hh(1:nb)
      integer(kind=ik)               :: i
      real(kind=rk8)                  :: v(nq)

!#ifdef WITH_OPENMP
!      call obj%timer%start("single_hh_trafo_real_cpu_openmp_double")
!#else
!      call obj%timer%start("single_hh_trafo_real_cpu_double")
!#endif

      ! v = q * hh
      v(:) = q(1:nq,1)
      do i=2,nb
        v(:) = v(:) + q(1:nq,i) * hh(i)
      enddo

      ! v = v * tau
      v(:) = v(:) * hh(1)

      ! q = q - v * hh**T
      q(1:nq,1) = q(1:nq,1) - v(:)
      do i=2,nb
        q(1:nq,i) = q(1:nq,i) - v(:) * hh(i)
      enddo

!#ifdef WITH_OPENMP
!      call obj%timer%stop("single_hh_trafo_real_cpu_openmp_double")
!#else
!      call obj%timer%stop("single_hh_trafo_real_cpu_double")
!#endif
    end subroutine

end module
