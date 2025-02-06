program hsx_dump
  use hsx_m, only: read_hsx_file, hsx_t
  ! use json_module
  implicit none
  integer, parameter :: dp = selected_real_kind(14,100)
  integer, parameter :: sp = selected_real_kind(6,30)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! type(json_core) :: json
  ! type(json_value), pointer :: jhead, jHon, jHoff, jSon, jSoff
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type(hsx_t) :: hsx
  character(len=256) :: hsxfp
  character(len=16)  :: idx
  integer, allocatable :: no(:)
  ! integer, allocatable :: nquant(:,:)
  ! integer, allocatable :: lquant(:,:)
  ! integer, allocatable :: zeta(:,:)
  integer, allocatable :: iaorb(:)
  integer, allocatable :: iphorb(:)
  integer, allocatable  :: numh(:) 
  integer, allocatable  :: listhptr(:)
  integer, allocatable  :: listh(:)  
  integer, allocatable  :: indxuo(:)
  real(sp), allocatable :: hamilt(:,:)
  real(sp), allocatable :: Sover(:)
  real(sp), allocatable :: xij(:,:)
  integer, allocatable  :: isa(:)
  real(sp), allocatable :: zval(:)
  integer, parameter :: MAXNAO = 19

  ! complex(dp), allocatable :: H(:,:,:)
  ! complex(dp), allocatable :: S(:,:)
  ! real(dp) :: xij(3)
  ! real(dp) :: k(3) = [0,0,0]
  ! complex(dp) :: img = (0.0, 1.0)
  ! complex(dp) :: phase

  ! real(dp), allocatable :: edge_index(:,:)
  ! integer,  allocatable :: cell_shift(:,:)
  ! integer,  allocatable :: inv_edge_idx(:)
  ! real(dp), allocatable :: Hon(:,:) !! since diff species have different naos.
  ! real(dp), allocatable :: Hoff(:,:)
  ! real(dp), allocatable :: Son(:,:)
  ! real(dp), allocatable :: Soff(:,:)
  
  call get_command_argument(1, hsxfp)
  call get_command_argument(2, idx)
  if (len_trim(hsxfp) == 0) then
    write(*,*) 'Please input HSX file path.'
    stop
  end if
  call read_hsx_file(hsx, trim(hsxfp))

  open(unit=11, file='HSX'//trim(idx), status='unknown', form='unformatted', access='stream', action='write')
  write(11) hsx%nspecies
  write(11) hsx%na_u    
  write(11) hsx%no_u   
  write(11) hsx%no_s
  write(11) hsx%nspin
  write(11) hsx%nh
  write(11) hsx%gamma
  write(11) hsx%has_xij
  allocate(no(hsx%nspecies))
  ! allocate(nquant(hsx%nspecies, MAXNAO))
  ! allocate(lquant(hsx%nspecies, MAXNAO))
  ! allocate(zeta(hsx%nspecies, MAXNAO))
  allocate(iaorb(hsx%no_u))
  allocate(iphorb(hsx%no_u))
  allocate(numh(hsx%no_u))
  allocate(listhptr(hsx%no_u))
  allocate(listh(hsx%nh))
  allocate(indxuo(hsx%no_s))
  allocate(hamilt(hsx%nh, hsx%nspin))
  allocate(Sover(hsx%nh))
  allocate(xij(3, hsx%nh))
  allocate(isa(hsx%na_u))
  allocate(zval(hsx%nspecies))
  no = hsx%no
  ! nquant = hsx%nquant
  ! lquant = hsx%lquant
  ! zeta = hsx%zeta
  iaorb = hsx%iaorb
  iphorb = hsx%iphorb
  numh = hsx%numh
  listhptr = hsx%listhptr
  listh = hsx%listh
  indxuo = hsx%indxuo
  hamilt = hsx%hamilt
  Sover = hsx%Sover
  xij = hsx%xij
  isa = hsx%isa
  zval = hsx%zval
  write(11) no
  ! write(11) nquant
  ! write(11) lquant
  ! write(11) zeta
  write(11) iaorb
  write(11) iphorb
  write(11) numh
  write(11) listhptr
  write(11) listh
  write(11) indxuo
  write(11) hamilt
  write(11) Sover
  write(11) xij
  write(11) isa
  write(11) zval
  close(11)
  
  ! To transform from sparse to full format, for a given k point:
  ! S(1:no_u,1:no_u) = 0                          ! full complex overlap matrix
  ! H(1:no_u,1:no_u,1:nspin) = 0                  ! full complex hamiltonian
  ! do io = 1,no_u                                ! loop on unit cell orbitals
  !   do j = 1,hsx%numh(io)                       ! loop on connected orbitals
  !     ij = hsx%listhptr(io)+j                   ! sparse-matrix array index
  !     jos = hsx%listh(ij)                       ! index of connected orbital
  !     jo = hsx%indxuo(jos)                      ! equiv. orbital in unit cell
  !     phase = exp(img*sum(k(:)*hsx%xij(:,ij)))  ! phase factor between orbs.
  !     H(io,jo,1:nspin) = H(io,jo,1:nspin) + &   ! hamiltonian matrix element
  !                         phase*hsx%hamilt(ij,1:nspin)
  !     S(io,jo) = S(io,jo) + phase*hsx%Sover(ij) ! overlap matrix element
  !   enddo
  ! enddo
  ! Notice that io,jo are within unit cell, and jos is within supercell

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! dump to json
  ! call json%initialize()
  ! call json%create_object(jhead, '')

  ! call json%add(jhead, 'edge_index', edge_index)

  ! call json%print(jhead, 'HS.json')
  ! call json%destroy(p)
  ! if (json%failed()) then
  !   write(*,*) 'An error occured when dumping json.'
  ! end if
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end program hsx_dump