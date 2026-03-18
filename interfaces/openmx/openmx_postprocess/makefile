###################################################################
#                                                                 #
#  Please set a proper CC and LIB for the compilation.            #
#  Examples of CC and LIB on several platforms are shown below.   #
#                                                                 #
###################################################################

#MKLROOT = /opt/compiler/intel2018u4/compilers_and_libraries_2018.5.274/linux/mkl
#CC = mpicc -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include/fftw #-DORBIT
#FC = mpif90 -O3 -xHOST -ip -no-prec-div -qopenmp
#LIB= -L${MKLROOT}/lib/intel64 -L/opt/compiler/intel2018u4/compilers_and_libraries_2018.5.274/linux/lib -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -liomp5 -lpthread -lm -ldl -lifcore -lgsl -lgslcblas -lm

GSL_lib = /data/home/yzhong/Tools/gsl-2.7.1/lib
GSL_include = /data/home/yzhong/Tools/gsl-2.7.1/include
MKLROOT = /opt/compiler/intel2018u4/compilers_and_libraries_2018.5.274/linux/mkl
CMPLR_ROOT = /opt/compiler/intel2018u4/compilers_and_libraries_2018.5.274
CC = mpiicc -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include/fftw -I${MKLROOT}/include -I${GSL_include} -DDERIVATIVE
FC = mpiifort -O3 -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include
LIB = ${CMPLR_ROOT}/linux/compiler/lib/intel64_lin/libiomp5.a ${MKLROOT}/lib/intel64/libmkl_blas95_lp64.a ${MKLROOT}/lib/intel64/libmkl_lapack95_lp64.a ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group ${CMPLR_ROOT}/linux/compiler/lib/intel64_lin/libifcoremt.a -lpthread -lm -ldl ${GSL_lib}/libgsl.a


#
# System B (Sekirei) at ISSP, Univ. of Tokyo
#
# CC = mpicc -O3  -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include/fftw -Dkcomp -fp-model precise
# FC = mpif90 -O3 -xHOST -ip -no-prec-div -qopenmp  -I${MKLROOT}/include/fftw -Dkcomp -fp-model precise
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lifcore -lmkl_blacs_sgimpt_lp64 -lmpi -liomp5 -lpthread -lm -ldl
#

#
# System C (Enaga) at ISSP, Univ. of Tokyo
#
# CC = mpicc -O3  -xHOST -ip -no-prec-div -qopenmp -I${MKLROOT}/include/fftw -Dkcomp -fp-model precise
# FC = mpif90 -O3 -xHOST -ip -no-prec-div -qopenmp  -I${MKLROOT}/include/fftw -Dkcomp -fp-model precise
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lifcore -lmkl_blacs_sgimpt_lp64 -lmpi -liomp5 -lpthread -lm -ldl
#

#
# pauli at ISSP, Univ. of Tokyo (AMD EPYC 7351P)
#
# CC = mpicc -I/opt/gnu/include -Dkcomp -O3 -march=znver1 -mtune=znver1 -mfma -mavx2 -m3dnow -fomit-frame-pointer -fopenmp
# FC = mpif90 -I/opt/gnu/include -Dkcomp -O3 -march=znver1 -mtune=znver1 -mfma -mavx2 -m3dnow -fomit-frame-pointer -fopenmp
# LIB = -L/opt/gnu/lib/ -lscalapack -lfftw3 -lflame -lblis -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lgfortran -lm
#

#
# FX100 at Nagoya Univ. (PRIMEHPC FX100, SPARC64b XIfx)
# before compilation do 'ulimit -v 33554432', and comment out elpa related objects
# 
# CC = mpifccpx -Kfast -Kopenmp -Dnosse -Dkcomp
# FC = mpifrtpx -Kfast -Kopenmp -Dkcomp
# LIB = -lfftw3 -SCALAPACK -SSL2BLAMP
#

#
# hster at JAIST (Intel Xeon E5-2680v2, 2.80GHz)
# 
# MKLROOT = /opt/intel/mkl
# FFTW = -I/work/t-ozaki/fftw-3.3.4
# CC = icc -openmp -O3 -xAVX -ip -no-prec-div $(FFTW)
# FC = ifort -openmp -O3 -xAVX -ip -no-prec-div $(FFTW)
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_sgimpt_lp64 -lmpi -lifcore -liomp5 -lpthread -lm -ldl
#

#
# Cray-XC40 at JAIST (Intel Xeon E5-2695v4 2.1GHz 18Core)
# 
# before compilation, do the following on your console
# module swap PrgEnv-cray PrgEnv-intel
# module load fftw
#
# CC      = cc -Dxt3 -O3 -axCOMMON-AVX512,CORE-AVX512,CORE-AVX2,CORE-AVX-I,AVX,SSE4.2,SSE4.1,SSE3,SSSE3,SSE2 -qopenmp -I/opt/cray/pe/fftw/3.3.6.5/x86_64/include
# FC      = ftn -Dxt3 -O3 -axCOMMON-AVX512,CORE-AVX512,CORE-AVX2,CORE-AVX-I,AVX,SSE4.2,SSE4.1,SSE3,SSSE3,SSE2 -qopenmp
# LIB     = -L/opt/cray/pe/fftw/3.3.6.5/x86_64/lib/ -lfftw3
#

#
# mx17 at ISSP, Univ. of Tokyo (Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz)
#
# MKLROOT = /opt/intel/mkl
# CC = mpicc -O3 -xHOST -ip -no-prec-div -qopenmp -I/opt/intel/mkl/include/fftw
# FC = mpif90 -O3 -xHOST -ip -no-prec-div -qopenmp
# LIB= -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_openmpi_lp64 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -liomp5 -lpthread -lm -ldl
#


CFLAGS  = -g 

OBJS    = openmx.o openmx_common.o Input_std.o Inputtools.o init.o LU_inverse.o ReLU_inverse.o \
          truncation.o readfile.o FT_PAO.o FT_NLP.o FT_ProExpn_VNA.o FT_VNA.o FT_ProductPAO.o \
          Hamiltonian_Cluster.o Hamiltonian_Cluster_Hs.o Hamiltonian_Cluster_NC_Hs2.o Hamiltonian_Band_NC_Hs2.o \
          Overlap_Cluster_NC_Ss2.o Overlap_Band_NC_Ss2.o Overlap_Cluster.o Overlap_Cluster_Ss.o \
          Set_ContMat_Cluster_LNO.o Hamiltonian_Band.o Matrix_Band_LNO.o Overlap_Band.o Hamiltonian_Cluster_NC.o \
          Hamiltonian_Band_NC.o Hamiltonian_Cluster_SO.o Get_OneD_HS_Col.o SetPara_DFT.o XC_Ceperly_Alder.o XC_CA_LSDA.o \
          XC_PW92C.o XC_PBE.o XC_EX.o DFT.o Mixing_DM.o Mixing_H.o Mixing_V.o Force.o Stress.o Poisson.o Poisson_ESM.o \
          Cluster_DFT_Col.o Cluster_DFT_NonCol.o Cluster_DFT_Dosout.o Cluster_DFT_ON2.o Cluster_DFT_LNO.o \
          Band_DFT_Col.o Band_DFT_NonCol.o Band_DFT_NonCol_GB.o \
          Band_DFT_kpath.o Band_DFT_kpath_LNO.o Band_DFT_MO.o \
          Unfolding_Bands.o Band_DFT_Dosout.o Set_Density_Grid.o Set_Orbitals_Grid.o Set_Aden_Grid.o \
          Gauss_Legendre.o zero_cfrac.o xyz2spherical.o AngularF.o RadialF.o Dr_RadialF.o PhiF.o \
          VNAF.o Dr_VNAF.o VH_AtomF.o Dr_VH_AtomF.o RF_BesselF.o QuickSort.o Nonlocal_RadialF.o \
          KumoF.o Dr_KumoF.o Mulliken_Charge.o Occupation_Number_LDA_U.o Eff_Hub_Pot.o Coulomb_Interaction.o \
          EulerAngle_Spin.o Smoothing_Func.o Orbital_Moment.o Pot_NeutralAtom.o Simple_Mixing_DM.o \
          DIIS_Mixing_DM.o ADIIS_Mixing_DM.o GR_Pulay_DM.o Kerker_Mixing_Rhok.o DIIS_Mixing_Rhok.o \
          Total_Energy.o Contract_Hamiltonian.o Contract_iHNL.o Cont_Matrix0.o Cont_Matrix1.o Cont_Matrix2.o \
          Cont_Matrix3.o Cont_Matrix4.o Opt_Contraction.o Initial_CntCoes.o Initial_CntCoes2.o Set_XC_Grid.o \
          Get_Orbitals.o Get_dOrbitals.o Get_Cnt_Orbitals.o \
          Get_Cnt_dOrbitals.o Gaunt.o Find_CGrids.o MD_pac.o \
          RestartFileDFT.o Output_CompTime.o Merge_LogFile.o Make_FracCoord.o \
          Make_InputFile_with_FinalCoord.o Output_Energy_Decomposition.o \
          Divide_Conquer.o Divide_Conquer_LNO.o Krylov.o \
          Divide_Conquer_Dosout.o EGAC_DFT.o LNO.o \
          Eigen_lapack.o Eigen_lapack2.o Eigen_lapack3.o EigenBand_lapack.o \
          Eigen_PReHH.o BroadCast_ReMatrix.o \
          Eigen_PHH.o BroadCast_ComplexMatrix.o \
          lapack_dstedc1.o lapack_dstedc2.o lapack_dstedc3.o\
          lapack_dstegr1.o lapack_dstegr2.o lapack_dstegr3.o \
          lapack_dstevx1.o lapack_dstevx2.o lapack_dstevx3.o \
          lapack_dstevx4.o lapack_dstevx5.o lapack_dsteqr1.o \
          Nonlocal_Basis.o Set_OLP_Kin.o Set_Nonlocal.o Set_ProExpn_VNA.o \
          Set_CoreHoleMatrix.o Set_OLP_p.o Set_Hamiltonian.o Set_Vpot.o \
          Voronoi_Charge.o Voronoi_Orbital_Moment.o Fuzzy_Weight.o \
          dampingF.o deri_dampingF.o Spherical_Bessel.o \
          iterout.o iterout_md.o Allocate_Arrays.o Free_Arrays.o \
          Init_List_YOUSO.o outputfile1.o \
          malloc_multidimarray.o PrintMemory.o PrintMemory_Fix.o \
          dtime.o OutData.o OutData_Binary.o init_alloc_first.o File_CntCoes.o \
          SCF2File.o mimic_sse.o Make_Comm_Worlds.o \
          Set_Allocate_Atom2CPU.o Cutoff.o Generating_MP_Special_Kpt.o \
          Maketest.o Runtest.o Memory_Leak_test.o \
          Force_test.o Stress_test.o Show_DFT_DATA.o Generate_Wannier.o \
          TRAN_Allocate.o TRAN_DFT.o TRAN_DFT_Dosout.o TRAN_Apply_Bias2e.o \
          TRAN_Deallocate_Electrode_Grid.o TRAN_Deallocate_RestartFile.o \
          TRAN_RestartFile.o TRAN_Calc_CentGreen.o TRAN_Input_std.o \
          TRAN_Set_CentOverlap.o TRAN_Calc_CentGreenLesser.o \
          TRAN_Input_std_Atoms.o TRAN_Set_Electrode_Grid.o \
          TRAN_Calc_GridBound.o TRAN_Set_IntegPath.o TRAN_Output_HKS.o \
          TRAN_Set_MP.o TRAN_Calc_SelfEnergy.o TRAN_Output_Trans_HS.o \
          TRAN_Calc_Hopping_G.o TRAN_Calc_SurfGreen.o TRAN_Set_SurfOverlap.o \
          TRAN_Add_Density_Lead.o TRAN_Add_ADensity_Lead.o TRAN_Set_Value.o \
          TRAN_Poisson.o TRAN_adjust_Ngrid.o TRAN_Print.o TRAN_Print_Grid.o \
          Lapack_LU_inverse.o TRAN_Distribute_Node.o TRAN_Output_HKS_Write_Grid.o \
          TRAN_Credit.o TRAN_Check_Region_Lead.o TRAN_Check_Region.o TRAN_Check_Input.o \
          DFTDvdW_init.o DFTD3vdW_init.o neb.o neb_run.o neb_check.o \
          TRAN_Allocate_NC.o TRAN_DFT_NC.o TRAN_Set_CentOverlap_NC.o TRAN_Set_SurfOverlap_NC.o \
          TRAN_Calc_OneTransmission.o TRAN_Main_Analysis.o TRAN_Main_Analysis_NC.o \
          MTRAN_EigenChannel.o TRAN_Channel_Functions.o TRAN_Channel_Output.o \
          TRAN_Calc_CurrentDensity.o TRAN_CDen_Main.o \
          elpa1.o solve_evp_real.o solve_evp_complex.o \
          NBO_Cluster.o NBO_Krylov.o Population_Analysis_Wannier.o Population_Analysis_Wannier2.o \
          NabraMatrixElements.o Set_dOrbitals_Grid.o Calc_optical.o \
          Band_DFT_NonCol_Optical.o Cluster_DFT_Optical.o \
          Band_DFT_Col_Optical_ScaLAPACK.o Cluster_DFT_Optical_ScaLAPACK.o 

# PROG    = openmx.exe
# PROG    = openmx

#-----------------------------------------------------------------------
# LIBELPA
#-----------------------------------------------------------------------
LIBELPADIR = ./elpa-2018.05.001

OBJS    += mod_precision.o elpa_utilities.o\
	elpa1_compute_real.o elpa1_compute_complex.o\
	aligned_mem.o elpa2_determine_workload.o\
	mod_redist_band_real.o mod_redist_band_complex.o\
	mod_pack_unpack_cpu_real.o mod_pack_unpack_cpu_complex.o\
	real.o complex.o\
	mod_single_hh_trafo_real.o mod_compute_hh_trafo_real.o mod_compute_hh_trafo_complex.o\
	elpa2_compute_real.o elpa2_compute_complex.o \
	elpa_solve_evp_real_2stage_double_impl.o elpa_solve_evp_complex_2stage_double_impl.o\

CC      += -I$(LIBELPADIR)
FC      += -I$(LIBELPADIR)

#
# set program name
# destination directory
#

PROG    = openmx
DESTDIR = ../work
UTIL 	= DosMain jx analysis_example esp polB calB Z2FH bandgnu13 bin2txt cube2xsf intensity_map md2axsf tp kSpin BandDispersion ADenBand FermiLoop GridCalc MulPOnly MulPCalc example_mpi_spawn gcube2oned

#
# OpenMX
#
openmx_postprocess: $(OBJS)
	$(CC) $(OBJS) $(STACK) $(LIB) -lm -o openmx_postprocess
	$(CC) -O2 read_openmx.c -o read_openmx

#
#
# all
#
#

all: $(PROG) $(UTIL)
	cp $(PROG) $(UTIL) $(DESTDIR)/

openmx.o: openmx.c openmx_common.h tran_variables.h tran_prototypes.h 
	$(CC) -c openmx.c
openmx_common.o: openmx_common.c openmx_common.h
	$(CC) -c openmx_common.c
Input_std.o: Input_std.c openmx_common.h Inputtools.h tran_prototypes.h 
	$(CC) -c Input_std.c
Inputtools.o: Inputtools.c
	$(CC) -c Inputtools.c

init.o: init.c openmx_common.h
	$(CC) -c init.c

LU_inverse.o: LU_inverse.c openmx_common.h
	$(CC) -c LU_inverse.c
ReLU_inverse.o: ReLU_inverse.c openmx_common.h
	$(CC) -c ReLU_inverse.c
truncation.o: truncation.c openmx_common.h tran_prototypes.h 
	$(CC) -c truncation.c
Find_CGrids.o: Find_CGrids.c openmx_common.h
	$(CC) -c Find_CGrids.c
readfile.o: readfile.c openmx_common.h
	$(CC) -c readfile.c
#
#
#
Hamiltonian_Cluster.o: Hamiltonian_Cluster.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster.c
Hamiltonian_Cluster_Hs.o: Hamiltonian_Cluster_Hs.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_Hs.c
Hamiltonian_Cluster_NC_Hs2.o: Hamiltonian_Cluster_NC_Hs2.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_NC_Hs2.c
Hamiltonian_Band_NC_Hs2.o: Hamiltonian_Band_NC_Hs2.c openmx_common.h
	$(CC) -c Hamiltonian_Band_NC_Hs2.c
Overlap_Cluster_NC_Ss2.o: Overlap_Cluster_NC_Ss2.c openmx_common.h
	$(CC) -c Overlap_Cluster_NC_Ss2.c
Overlap_Band_NC_Ss2.o: Overlap_Band_NC_Ss2.c openmx_common.h
	$(CC) -c Overlap_Band_NC_Ss2.c
Overlap_Cluster.o: Overlap_Cluster.c openmx_common.h
	$(CC) -c Overlap_Cluster.c  
Overlap_Cluster_Ss.o: Overlap_Cluster_Ss.c openmx_common.h
	$(CC) -c Overlap_Cluster_Ss.c  
Set_ContMat_Cluster_LNO.o: Set_ContMat_Cluster_LNO.c openmx_common.h
	$(CC) -c Set_ContMat_Cluster_LNO.c
Hamiltonian_Band.o: Hamiltonian_Band.c openmx_common.h
	$(CC) -c Hamiltonian_Band.c
Overlap_Band.o: Overlap_Band.c openmx_common.h
	$(CC) -c Overlap_Band.c
Matrix_Band_LNO.o: Matrix_Band_LNO.c openmx_common.h
	$(CC) -c Matrix_Band_LNO.c
Hamiltonian_Cluster_NC.o: Hamiltonian_Cluster_NC.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_NC.c
Hamiltonian_Cluster_SO.o: Hamiltonian_Cluster_SO.c openmx_common.h
	$(CC) -c Hamiltonian_Cluster_SO.c
Hamiltonian_Band_NC.o: Hamiltonian_Band_NC.c openmx_common.h
	$(CC) -c Hamiltonian_Band_NC.c
Get_OneD_HS_Col.o: Get_OneD_HS_Col.c openmx_common.h
	$(CC) -c Get_OneD_HS_Col.c
#
#
#  
SetPara_DFT.o: SetPara_DFT.c openmx_common.h
	$(CC) -c SetPara_DFT.c
XC_Ceperly_Alder.o: XC_Ceperly_Alder.c openmx_common.h
	$(CC) -c XC_Ceperly_Alder.c
XC_CA_LSDA.o: XC_CA_LSDA.c openmx_common.h
	$(CC) -c XC_CA_LSDA.c
XC_PW92C.o: XC_PW92C.c openmx_common.h
	$(CC) -c XC_PW92C.c
XC_PBE.o: XC_PBE.c openmx_common.h
	$(CC) -c XC_PBE.c
XC_EX.o: XC_EX.c openmx_common.h
	$(CC) -c XC_EX.c
#
# SCF
#

DFT.o: DFT.c openmx_common.h tran_prototypes.h
	$(CC) -c DFT.c
Cluster_DFT_Col.o: Cluster_DFT_Col.c openmx_common.h
	$(CC) -c Cluster_DFT_Col.c
Cluster_DFT_NonCol.o: Cluster_DFT_NonCol.c openmx_common.h
	$(CC) -c Cluster_DFT_NonCol.c
Cluster_DFT_Dosout.o: Cluster_DFT_Dosout.c openmx_common.h
	$(CC) -c Cluster_DFT_Dosout.c
Cluster_DFT_ON2.o: Cluster_DFT_ON2.c openmx_common.h
	$(CC) -c Cluster_DFT_ON2.c
Cluster_DFT_LNO.o: Cluster_DFT_LNO.c openmx_common.h
	$(CC) -c Cluster_DFT_LNO.c
Band_DFT_Col.o: Band_DFT_Col.c openmx_common.h
	$(CC) -c Band_DFT_Col.c
Band_DFT_NonCol.o: Band_DFT_NonCol.c openmx_common.h
	$(CC) -c Band_DFT_NonCol.c
Band_DFT_NonCol_GB.o: Band_DFT_NonCol_GB.c openmx_common.h
	$(CC) -c Band_DFT_NonCol_GB.c
Band_DFT_kpath.o: Band_DFT_kpath.c openmx_common.h
	$(CC) -c Band_DFT_kpath.c
Band_DFT_kpath_LNO.o: Band_DFT_kpath_LNO.c openmx_common.h
	$(CC) -c Band_DFT_kpath_LNO.c
Band_DFT_MO.o: Band_DFT_MO.c openmx_common.h
	$(CC) -c Band_DFT_MO.c
Unfolding_Bands.o: Unfolding_Bands.c openmx_common.h
	$(CC) -c Unfolding_Bands.c
Band_DFT_Dosout.o: Band_DFT_Dosout.c openmx_common.h
	$(CC) -c Band_DFT_Dosout.c
Mixing_DM.o: Mixing_DM.c openmx_common.h
	$(CC) -c Mixing_DM.c
Mixing_H.o: Mixing_H.c openmx_common.h
	$(CC) -c Mixing_H.c
Mixing_V.o: Mixing_V.c openmx_common.h
	$(CC) -c Mixing_V.c
Force.o: Force.c openmx_common.h
	$(CC) -c Force.c
Stress.o: Stress.c openmx_common.h
	$(CC) -c Stress.c
Poisson.o: Poisson.c openmx_common.h
	$(CC) -c Poisson.c
Poisson_ESM.o: Poisson_ESM.c openmx_common.h
	$(CC) -c Poisson_ESM.c
Mulliken_Charge.o: Mulliken_Charge.c openmx_common.h
	$(CC) -c Mulliken_Charge.c
Occupation_Number_LDA_U.o: Occupation_Number_LDA_U.c openmx_common.h
	$(CC) -c Occupation_Number_LDA_U.c
Eff_Hub_Pot.o: Eff_Hub_Pot.c openmx_common.h
	$(CC) -c Eff_Hub_Pot.c
Coulomb_Interaction.o: Coulomb_Interaction.c openmx_common.h
	$(CC) -c Coulomb_Interaction.c
EulerAngle_Spin.o: EulerAngle_Spin.c openmx_common.h
	$(CC) -c EulerAngle_Spin.c
Orbital_Moment.o: Orbital_Moment.c openmx_common.h
	$(CC) -c Orbital_Moment.c
Smoothing_Func.o: Smoothing_Func.c openmx_common.h
	$(CC) -c Smoothing_Func.c
Gauss_Legendre.o: Gauss_Legendre.c openmx_common.h
	$(CC) -c Gauss_Legendre.c
zero_cfrac.o: zero_cfrac.c openmx_common.h
	$(CC) -c zero_cfrac.c
xyz2spherical.o: xyz2spherical.c openmx_common.h
	$(CC) -c xyz2spherical.c
AngularF.o: AngularF.c openmx_common.h
	$(CC) -c AngularF.c
RadialF.o: RadialF.c openmx_common.h
	$(CC) -c RadialF.c
Dr_RadialF.o: Dr_RadialF.c openmx_common.h
	$(CC) -c Dr_RadialF.c
PhiF.o: PhiF.c openmx_common.h
	$(CC) -c PhiF.c
VNAF.o: VNAF.c openmx_common.h
	$(CC) -c VNAF.c
Dr_VNAF.o: Dr_VNAF.c openmx_common.h
	$(CC) -c Dr_VNAF.c
VH_AtomF.o: VH_AtomF.c openmx_common.h
	$(CC) -c VH_AtomF.c
Dr_VH_AtomF.o: Dr_VH_AtomF.c openmx_common.h
	$(CC) -c Dr_VH_AtomF.c

RF_BesselF.o: RF_BesselF.c openmx_common.h
	$(CC) -c RF_BesselF.c
Nonlocal_RadialF.o: Nonlocal_RadialF.c openmx_common.h
	$(CC) -c Nonlocal_RadialF.c

Set_Orbitals_Grid.o: Set_Orbitals_Grid.c openmx_common.h
	$(CC) -c Set_Orbitals_Grid.c
Set_Density_Grid.o: Set_Density_Grid.c openmx_common.h
	$(CC) -c Set_Density_Grid.c
Set_Aden_Grid.o: Set_Aden_Grid.c openmx_common.h
	$(CC) -c Set_Aden_Grid.c

KumoF.o: KumoF.c openmx_common.h
	$(CC) -c KumoF.c
Dr_KumoF.o: Dr_KumoF.c openmx_common.h
	$(CC) -c Dr_KumoF.c
Pot_NeutralAtom.o: Pot_NeutralAtom.c openmx_common.h
	$(CC) -c Pot_NeutralAtom.c
Simple_Mixing_DM.o: Simple_Mixing_DM.c openmx_common.h
	$(CC) -c Simple_Mixing_DM.c
DIIS_Mixing_DM.o: DIIS_Mixing_DM.c openmx_common.h
	$(CC) -c DIIS_Mixing_DM.c
ADIIS_Mixing_DM.o: ADIIS_Mixing_DM.c openmx_common.h
	$(CC) -c ADIIS_Mixing_DM.c
GR_Pulay_DM.o: GR_Pulay_DM.c openmx_common.h
	$(CC) -c GR_Pulay_DM.c
Kerker_Mixing_Rhok.o: Kerker_Mixing_Rhok.c openmx_common.h
	$(CC) -c Kerker_Mixing_Rhok.c
DIIS_Mixing_Rhok.o: DIIS_Mixing_Rhok.c openmx_common.h
	$(CC) -c DIIS_Mixing_Rhok.c
Total_Energy.o: Total_Energy.c openmx_common.h
	$(CC) -c Total_Energy.c
Contract_Hamiltonian.o: Contract_Hamiltonian.c openmx_common.h
	$(CC) -c Contract_Hamiltonian.c
Contract_iHNL.o: Contract_iHNL.c openmx_common.h
	$(CC) -c Contract_iHNL.c
Cont_Matrix0.o: Cont_Matrix0.c openmx_common.h
	$(CC) -c Cont_Matrix0.c
Cont_Matrix1.o: Cont_Matrix1.c openmx_common.h
	$(CC) -c Cont_Matrix1.c
Cont_Matrix2.o: Cont_Matrix2.c openmx_common.h
	$(CC) -c Cont_Matrix2.c
Cont_Matrix3.o: Cont_Matrix3.c openmx_common.h
	$(CC) -c Cont_Matrix3.c
Cont_Matrix4.o: Cont_Matrix4.c openmx_common.h
	$(CC) -c Cont_Matrix4.c
Opt_Contraction.o: Opt_Contraction.c openmx_common.h
	$(CC) -c Opt_Contraction.c
Initial_CntCoes.o: Initial_CntCoes.c openmx_common.h
	$(CC) -c Initial_CntCoes.c
Initial_CntCoes2.o: Initial_CntCoes2.c openmx_common.h
	$(CC) -c Initial_CntCoes2.c


Set_XC_Grid.o: Set_XC_Grid.c openmx_common.h
	$(CC) -c Set_XC_Grid.c
Get_Orbitals.o: Get_Orbitals.c openmx_common.h
	$(CC) -c Get_Orbitals.c
Get_dOrbitals.o: Get_dOrbitals.c openmx_common.h
	$(CC) -c Get_dOrbitals.c
Get_Cnt_Orbitals.o: Get_Cnt_Orbitals.c openmx_common.h
	$(CC) -c Get_Cnt_Orbitals.c
Get_Cnt_dOrbitals.o: Get_Cnt_dOrbitals.c openmx_common.h
	$(CC) -c Get_Cnt_dOrbitals.c
Gaunt.o: Gaunt.c openmx_common.h
	$(CC) -c Gaunt.c
RestartFileDFT.o: RestartFileDFT.c openmx_common.h
	$(CC) -c RestartFileDFT.c
Output_CompTime.o: Output_CompTime.c openmx_common.h
	$(CC) -c Output_CompTime.c
Output_Energy_Decomposition.o: Output_Energy_Decomposition.c openmx_common.h
	$(CC) -c Output_Energy_Decomposition.c
Merge_LogFile.o: Merge_LogFile.c openmx_common.h
	$(CC) -c Merge_LogFile.c
Make_FracCoord.o: Make_FracCoord.c openmx_common.h
	$(CC) -c Make_FracCoord.c
Make_InputFile_with_FinalCoord.o: Make_InputFile_with_FinalCoord.c openmx_common.h
	$(CC) -c Make_InputFile_with_FinalCoord.c
#
#
#
QuickSort.o: QuickSort.c openmx_common.h
	$(CC) -c QuickSort.c
Eigen_lapack.o: Eigen_lapack.c openmx_common.h lapack_prototypes.h
	$(CC) -c Eigen_lapack.c
Eigen_lapack2.o: Eigen_lapack2.c openmx_common.h lapack_prototypes.h
	$(CC) -c Eigen_lapack2.c
Eigen_lapack3.o: Eigen_lapack3.c openmx_common.h lapack_prototypes.h
	$(CC) -c Eigen_lapack3.c
EigenBand_lapack.o: EigenBand_lapack.c openmx_common.h lapack_prototypes.h
	$(CC) -c EigenBand_lapack.c
Eigen_PReHH.o: Eigen_PReHH.c openmx_common.h
	$(CC) -c Eigen_PReHH.c
Eigen_PHH.o: Eigen_PHH.c openmx_common.h
	$(CC) -c Eigen_PHH.c
BroadCast_ReMatrix.o: BroadCast_ReMatrix.c openmx_common.h
	$(CC) -c BroadCast_ReMatrix.c
BroadCast_ComplexMatrix.o: BroadCast_ComplexMatrix.c openmx_common.h
	$(CC) -c BroadCast_ComplexMatrix.c
lapack_dstedc1.o: lapack_dstedc1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstedc1.c
lapack_dstedc2.o: lapack_dstedc2.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstedc2.c
lapack_dstedc3.o: lapack_dstedc3.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstedc3.c
lapack_dstegr1.o: lapack_dstegr1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstegr1.c
lapack_dstegr2.o: lapack_dstegr2.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstegr2.c
lapack_dstegr3.o: lapack_dstegr3.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstegr3.c
lapack_dstevx1.o: lapack_dstevx1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx1.c
lapack_dstevx2.o: lapack_dstevx2.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx2.c
lapack_dstevx3.o: lapack_dstevx3.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx3.c
lapack_dstevx4.o: lapack_dstevx4.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx4.c
lapack_dstevx5.o: lapack_dstevx5.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dstevx5.c
lapack_dsteqr1.o: lapack_dsteqr1.c openmx_common.h lapack_prototypes.h
	$(CC) -c lapack_dsteqr1.c
Nonlocal_Basis.o: Nonlocal_Basis.c openmx_common.h
	$(CC) -c Nonlocal_Basis.c
Set_OLP_Kin.o: Set_OLP_Kin.c openmx_common.h 
	$(CC) -c Set_OLP_Kin.c
Set_Nonlocal.o: Set_Nonlocal.c openmx_common.h
	$(CC) -c Set_Nonlocal.c
Set_ProExpn_VNA.o: Set_ProExpn_VNA.c openmx_common.h
	$(CC) -c Set_ProExpn_VNA.c
Set_CoreHoleMatrix.o: Set_CoreHoleMatrix.c openmx_common.h
	$(CC) -c Set_CoreHoleMatrix.c
Set_OLP_p.o: Set_OLP_p.c openmx_common.h 
	$(CC) -c Set_OLP_p.c
Set_Hamiltonian.o: Set_Hamiltonian.c openmx_common.h
	$(CC) -c Set_Hamiltonian.c
Set_Vpot.o: Set_Vpot.c openmx_common.h
	$(CC) -c Set_Vpot.c
#
#
#
FT_PAO.o: FT_PAO.c openmx_common.h
	$(CC) -c FT_PAO.c
FT_NLP.o: FT_NLP.c openmx_common.h
	$(CC) -c FT_NLP.c
FT_ProExpn_VNA.o: FT_ProExpn_VNA.c openmx_common.h
	$(CC) -c FT_ProExpn_VNA.c
FT_VNA.o: FT_VNA.c openmx_common.h
	$(CC) -c FT_VNA.c
FT_ProductPAO.o: FT_ProductPAO.c openmx_common.h
	$(CC) -c FT_ProductPAO.c
#
#
#
Divide_Conquer.o: Divide_Conquer.c openmx_common.h
	$(CC) -c Divide_Conquer.c
Divide_Conquer_LNO.o: Divide_Conquer_LNO.c openmx_common.h
	$(CC) -c Divide_Conquer_LNO.c
Divide_Conquer_Dosout.o: Divide_Conquer_Dosout.c openmx_common.h
	$(CC) -c Divide_Conquer_Dosout.c
Krylov.o: Krylov.c openmx_common.h
	$(CC) -c Krylov.c
EGAC_DFT.o: EGAC_DFT.c openmx_common.h
	$(CC) -c EGAC_DFT.c
LNO.o: LNO.c openmx_common.h
	$(CC) -c LNO.c
#
#
#
MD_pac.o: MD_pac.c openmx_common.h lapack_prototypes.h
	$(CC) -c MD_pac.c
#
#
#
iterout.o: iterout.c openmx_common.h
	$(CC) -c iterout.c
iterout_md.o: iterout_md.c openmx_common.h
	$(CC) -c iterout_md.c
Allocate_Arrays.o: Allocate_Arrays.c openmx_common.h
	$(CC) -c Allocate_Arrays.c
Free_Arrays.o: Free_Arrays.c openmx_common.h
	$(CC) -c Free_Arrays.c
Init_List_YOUSO.o: Init_List_YOUSO.c openmx_common.h
	$(CC) -c Init_List_YOUSO.c
outputfile1.o: outputfile1.c openmx_common.h
	$(CC) -c outputfile1.c
malloc_multidimarray.o: malloc_multidimarray.c
	$(CC) -c malloc_multidimarray.c 
PrintMemory.o: PrintMemory.c
	$(CC) -c PrintMemory.c 
PrintMemory_Fix.o: PrintMemory_Fix.c openmx_common.h
	$(CC) -c PrintMemory_Fix.c 
dtime.o: dtime.c
	$(CC) -c dtime.c 
OutData.o: OutData.c openmx_common.h
	$(CC) -c OutData.c
OutData_Binary.o: OutData_Binary.c openmx_common.h
	$(CC) -c OutData_Binary.c
init_alloc_first.o: init_alloc_first.c openmx_common.h
	$(CC) -c init_alloc_first.c
File_CntCoes.o: File_CntCoes.c openmx_common.h
	$(CC) -c File_CntCoes.c
SCF2File.o: SCF2File.c openmx_common.h
	$(CC) -c SCF2File.c
Cutoff.o: Cutoff.c openmx_common.h
	$(CC) -c Cutoff.c
Voronoi_Charge.o: Voronoi_Charge.c openmx_common.h
	$(CC) -c Voronoi_Charge.c
Voronoi_Orbital_Moment.o: Voronoi_Orbital_Moment.c openmx_common.h
	$(CC) -c Voronoi_Orbital_Moment.c
Fuzzy_Weight.o: Fuzzy_Weight.c openmx_common.h
	$(CC) -c Fuzzy_Weight.c
dampingF.o: dampingF.c openmx_common.h
	$(CC) -c dampingF.c
deri_dampingF.o: deri_dampingF.c openmx_common.h
	$(CC) -c deri_dampingF.c
Spherical_Bessel.o: Spherical_Bessel.c openmx_common.h
	$(CC) -c Spherical_Bessel.c
Generating_MP_Special_Kpt.o: Generating_MP_Special_Kpt.c openmx_common.h
	$(CC) -c Generating_MP_Special_Kpt.c
Generate_Wannier.o: Generate_Wannier.c openmx_common.h
	$(CC) -c Generate_Wannier.c
DFTDvdW_init.o: DFTDvdW_init.c openmx_common.h
	$(CC) -c DFTDvdW_init.c
DFTD3vdW_init.o: DFTD3vdW_init.c openmx_common.h
	$(CC) -c DFTD3vdW_init.c
neb.o:	neb.c openmx_common.h Inputtools.h lapack_prototypes.h
	$(CC) -c neb.c
neb_run.o: neb_run.c openmx_common.h
	$(CC) -c neb_run.c
neb_check.o: neb_check.c openmx_common.h Inputtools.h 
	$(CC) -c neb_check.c
NBO_Cluster.o: NBO_Cluster.c openmx_common.h Inputtools.h
	$(CC) -c NBO_Cluster.c
NBO_Krylov.o: NBO_Krylov.c openmx_common.h Inputtools.h
	$(CC) -c NBO_Krylov.c
Population_Analysis_Wannier.o: Population_Analysis_Wannier.c openmx_common.h Inputtools.h
	$(CC) -c Population_Analysis_Wannier.c
Population_Analysis_Wannier2.o: Population_Analysis_Wannier2.c openmx_common.h Inputtools.h
	$(CC) -c Population_Analysis_Wannier2.c

#
# codes for calculating optical conductivities and dielectric functions developed by YTL
#

NabraMatrixElements.o: NabraMatrixElements.c openmx_common.h
	$(CC) -c NabraMatrixElements.c
Set_dOrbitals_Grid.o: Set_dOrbitals_Grid.c openmx_common.h
	$(CC) -c Set_dOrbitals_Grid.c
Calc_optical.o: Calc_optical.c openmx_common.h
	$(CC) -c Calc_optical.c
Band_DFT_NonCol_Optical.o: Band_DFT_NonCol_Optical.c openmx_common.h
	$(CC) -c Band_DFT_NonCol_Optical.c
Cluster_DFT_Col_Optical.o: Cluster_DFT_Col_Optical.c openmx_common.h
	$(CC) -c Cluster_DFT_Col_Optical.c
Band_DFT_Col_Optical_ScaLAPACK.o: Band_DFT_Col_Optical_ScaLAPACK.c openmx_common.h
	$(CC) -c Band_DFT_Col_Optical_ScaLAPACK.c
Cluster_DFT_Col_Optical_ScaLAPACK.o: Cluster_DFT_Col_Optical_ScaLAPACK.c openmx_common.h
	$(CC) -c Cluster_DFT_Col_Optical_ScaLAPACK.c

#
#
#
mimic_sse.o: mimic_sse.c mimic_sse.h
	$(CC) -c mimic_sse.c
Make_Comm_Worlds.o: Make_Comm_Worlds.c
	$(CC) -c Make_Comm_Worlds.c
Set_Allocate_Atom2CPU.o: Set_Allocate_Atom2CPU.c openmx_common.h
	$(CC) -c Set_Allocate_Atom2CPU.c

#
#
# Maketest, Runtest, Memory_Leak_test, Force_test, Show_DFT_DATA
#
#

Maketest.o: Maketest.c openmx_common.h Inputtools.h
	$(CC) -c Maketest.c
Runtest.o: Runtest.c openmx_common.h Inputtools.h
	$(CC) -c Runtest.c
Memory_Leak_test.o: Memory_Leak_test.c openmx_common.h Inputtools.h
	$(CC) -c Memory_Leak_test.c
Force_test.o: Force_test.c openmx_common.h Inputtools.h
	$(CC) -c Force_test.c
Stress_test.o: Stress_test.c openmx_common.h Inputtools.h
	$(CC) -c Stress_test.c
Show_DFT_DATA.o: Show_DFT_DATA.c openmx_common.h Inputtools.h
	$(CC) -c Show_DFT_DATA.c

#
# install
#
#

install: $(PROG)
#	strip $(PROG)
	cp $(PROG) $(DESTDIR)/$(PROG)

#
#
# clean executable and object files 
#
#

clean:
	rm -f $(PROG) $(OBJS) $(OBJS_jx) $(UTIL) *.o *.mod *.dbg 

#
#
# programs for generating DOS from files *.Dos.val and *.Dos.vec
#
#

DosMain: DosMain.o Inputtools.o malloc_multidimarray.o Tetrahedron_Blochl.o 
	$(CC) -o $@ DosMain.o Inputtools.o malloc_multidimarray.o Tetrahedron_Blochl.o -lm 
	cp DosMain $(DESTDIR)/DosMain

DosMain.o :DosMain.c openmx_common.h
	$(CC) -o $@ -c DosMain.c
Tetrahedron_Blochl.o : Tetrahedron_Blochl.c
	$(CC) -o $@ -c Tetrahedron_Blochl.c 

#
#
#  exchange interaction coupling constant J between two atoms
#
#

OBJ_jx = Inputtools.o read_scfout.o jx_quicksort.o jx_LNO.o jx_config.o \
	jx_tools.o jx_cluster.o jx_band_psum.o jx_band_indiv.o jx.o

jx_quicksort.o: jx_quicksort.c
	$(CC) -c jx_quicksort.c

jx_LNO.o: jx_LNO.c jx_tools.h jx_LNO.h read_scfout.h
	$(CC) -c jx_LNO.c

jx_config.o: jx_config.c jx_config.h
	$(CC) -c jx_config.c

jx_tools.o: jx_tools.c jx_tools.h jx_total_mem.h
	$(CC) -c jx_tools.c

jx_cluster.o: jx_cluster.c jx_tools.c jx.h jx_tools.h jx_total_mem.h
	$(CC) -c jx_cluster.c

jx_band_psum.o: jx_band_psum.c jx_tools.c jx.h jx_tools.h jx_total_mem.h
	$(CC) -c jx_band_psum.c

jx_band_indiv.o: jx_band_indiv.c jx_tools.c jx.h jx_tools.h jx_total_mem.h
	$(CC) -c jx_band_indiv.c

jx.o: jx.c jx_tools.c read_scfout.h jx_tools.h jx_total_mem.h jx.h
	$(CC) -c jx.c

jx: $(OBJ_jx)
	$(CC) $(OBJ_jx) $(LIB) -lm -o jx
	cp jx $(DESTDIR)/jx

#
#
#  transition probability between two states */
#
#

tp: tp.o read_scfout.o
	$(CC) tp.o read_scfout.o $(LIB) -lm -o tp
	cp tp $(DESTDIR)/tp

tp.o: tp.c read_scfout.h 
	$(CC) -c tp.c

#
#
# analysis_example
#
#

analysis_example: analysis_example.o read_scfout.o
	$(CC) analysis_example.o read_scfout.o $(LIB)  -lm -o analysis_example
	cp analysis_example $(DESTDIR)/analysis_example

analysis_example.o: analysis_example.c read_scfout.h 
	$(CC) -c analysis_example.c

read_scfout.o: read_scfout.c read_scfout.h 
	$(CC) -c read_scfout.c

#
#
# program for generating EPS from files *.out and *.vhart
#
#

OBJS_ESP  = esp.o Inputtools.o
esp:	$(OBJS_ESP)
	$(CC) $(OBJS_ESP) $(LIB) -lm -o $@
	cp esp $(DESTDIR)/esp
esp.o : esp.c Inputtools.h
	$(CC) -o $@ -c esp.c

#
#
# check_lead
#
#

check_lead: check_lead.o Inputtools.o
	$(CC) check_lead.o Inputtools.o -lm -o check_lead
	cp check_lead $(DESTDIR)/check_lead

check_lead.o: check_lead.c Inputtools.h 
	$(CC) -c check_lead.c

#
#
#  optical conductivity 
#
#

OpticalConductivityMain: OpticalConductivityMain.o \
              Inputtools.o  malloc_multidimarray.o
	$(CC) -o $@   OpticalConductivityMain.o  Inputtools.o  malloc_multidimarray.o -lm 
	cp OpticalConductivityMain $(DESTDIR)/OpticalConductivityMain

#
#
#  electric polarization using Berry's phase
#
#

OBJS_polB = polB.o read_scfout.o
polB:	$(OBJS_polB)
	$(CC) $(OBJS_polB) $(LIB) -lm -o polB
	cp polB $(DESTDIR)/polB

polB.o: polB.c read_scfout.h 
	$(CC) -c polB.c

#
#
# Code for calculating Berry Curvature and Chern Number
#
#

OBJS_calB = calB.o read_scfout.o
calB:	$(OBJS_calB)
	$(CC) $(OBJS_calB) $(LIB) -lm -o calB
	cp calB $(DESTDIR)/calB

calB.o: calB.c read_scfout.h 
	$(CC) -c calB.c

#
#
# Code for calculating Z2 invariant by Fukui-Hatsugai Method
#
#

OBJS_Z2FH = Z2FH.o read_scfout.o
Z2FH:	$(OBJS_Z2FH)
	$(CC) $(OBJS_Z2FH) $(LIB) -lm -o Z2FH
	cp Z2FH $(DESTDIR)/Z2FH

Z2FH.o: Z2FH.c read_scfout.h 
	$(CC) -c Z2FH.c

#
#
# example_mpi_spawn
#
#

example_mpi_spawn: example_mpi_spawn.o
	$(CC) example_mpi_spawn.o $(LIB) -lm -o example_mpi_spawn
	cp example_mpi_spawn $(DESTDIR)/example_mpi_spawn

example_mpi_spawn.o: example_mpi_spawn.c
	$(CC) -c example_mpi_spawn.c


#
#
# test_mpi
#
#

test_mpi: test_mpi.o
	$(CC) test_mpi.o $(LIB) -lm -o test_mpi
	cp test_mpi $(DESTDIR)/test_mpi

test_mpi.o: test_mpi.c
	$(CC) -c test_mpi.c

MAIN_TRAN_Display_Gridvalue: MAIN_TRAN_Display_Gridvalue.o TRAN_Read.o TRAN_Print.o
	$(CC) -o $@  MAIN_TRAN_Display_Gridvalue.o TRAN_Read.o TRAN_Print.o -lm $(LIB)  

TRAN_Main_Analysis.o: TRAN_Main_Analysis.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Main_Analysis.c
TRAN_Main_Analysis_NC.o: TRAN_Main_Analysis_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Main_Analysis_NC.c

TRAN_Allocate.o: TRAN_Allocate.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Allocate.c
TRAN_Calc_GridBound.o: TRAN_Calc_GridBound.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Calc_GridBound.c
TRAN_DFT.o: TRAN_DFT.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_DFT.c
TRAN_DFT_Dosout.o: TRAN_DFT_Dosout.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_DFT_Dosout.c
TRAN_Input_std.o: TRAN_Input_std.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Input_std.c
TRAN_Input_std_Atoms.o: TRAN_Input_std_Atoms.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Input_std_Atoms.c
TRAN_Output_HKS.o: TRAN_Output_HKS.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Output_HKS.c
TRAN_Output_Trans_HS.o: TRAN_Output_Trans_HS.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Output_Trans_HS.c
TRAN_Add_Density_Lead.o: TRAN_Add_Density_Lead.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Add_Density_Lead.c
TRAN_Add_ADensity_Lead.o: TRAN_Add_ADensity_Lead.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Add_ADensity_Lead.c
TRAN_Poisson.o: TRAN_Poisson.c tran_variables.h tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Poisson.c
TRAN_RestartFile.o: TRAN_RestartFile.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_RestartFile.c
TRAN_Set_CentOverlap.o: TRAN_Set_CentOverlap.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_CentOverlap.c
TRAN_Set_Electrode_Grid.o: tran_variables.h tran_prototypes.h openmx_common.h
	$(CC) -c TRAN_Set_Electrode_Grid.c
TRAN_Set_IntegPath.o: TRAN_Set_IntegPath.c tran_variables.h tran_prototypes.h lapack_prototypes.h  
	$(CC) -c TRAN_Set_IntegPath.c
TRAN_Set_SurfOverlap.o: TRAN_Set_SurfOverlap.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_SurfOverlap.c
TRAN_adjust_Ngrid.o: TRAN_adjust_Ngrid.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_adjust_Ngrid.c
Lapack_LU_inverse.o: Lapack_LU_inverse.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c Lapack_LU_inverse.c
TRAN_Deallocate_Electrode_Grid.o: TRAN_Deallocate_Electrode_Grid.c tran_variables.h 
	$(CC) -c TRAN_Deallocate_Electrode_Grid.c
TRAN_Deallocate_RestartFile.o: TRAN_Deallocate_RestartFile.c tran_variables.h 
	$(CC) -c TRAN_Deallocate_RestartFile.c
TRAN_Apply_Bias2e.o: TRAN_Apply_Bias2e.c tran_prototypes.h 
	$(CC) -c TRAN_Apply_Bias2e.c
TRAN_Calc_CentGreen.o: TRAN_Calc_CentGreen.c tran_prototypes.h 
	$(CC) -c TRAN_Calc_CentGreen.c
TRAN_Calc_CentGreenLesser.o: TRAN_Calc_CentGreenLesser.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_CentGreenLesser.c
TRAN_Calc_OneTransmission.o: TRAN_Calc_OneTransmission.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_OneTransmission.c
TRAN_Calc_SelfEnergy.o: TRAN_Calc_SelfEnergy.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_SelfEnergy.c
TRAN_Calc_SurfGreen.o: TRAN_Calc_SurfGreen.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_SurfGreen.c
TRAN_Calc_Hopping_G.o: TRAN_Calc_Hopping_G.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_Hopping_G.c
TRAN_Credit.o: TRAN_Credit.c tran_prototypes.h 
	$(CC) -c TRAN_Credit.c
TRAN_Output_HKS_Write_Grid.o: TRAN_Output_HKS_Write_Grid.c tran_prototypes.h 
	$(CC) -c TRAN_Output_HKS_Write_Grid.c
TRAN_Print.o: TRAN_Print.c tran_prototypes.h 
	$(CC) -c TRAN_Print.c
TRAN_Print_Grid.o: TRAN_Print_Grid.c tran_prototypes.h 
	$(CC) -c TRAN_Print_Grid.c
TRAN_Read.o: TRAN_Read.c tran_prototypes.h 
	$(CC) -c TRAN_Read.c
TRAN_Set_Value.o: TRAN_Set_Value.c tran_prototypes.h 
	$(CC) -c TRAN_Set_Value.c
TRAN_Check_Region_Lead.o: TRAN_Check_Region_Lead.c tran_variables.h
	$(CC) -c TRAN_Check_Region_Lead.c
TRAN_Check_Region.o: TRAN_Check_Region.c tran_prototypes.h 
	$(CC) -c TRAN_Check_Region.c
TRAN_Check_Input.o: TRAN_Check_Input.c tran_prototypes.h 
	$(CC) -c TRAN_Check_Input.c
TRAN_Set_MP.o: TRAN_Set_MP.c
	$(CC) -c TRAN_Set_MP.c
TRAN_Distribute_Node.o: TRAN_Distribute_Node.c
	$(CC) -c TRAN_Distribute_Node.c
TRAN_Allocate_NC.o: TRAN_Allocate_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Allocate_NC.c
TRAN_DFT_NC.o: TRAN_DFT_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_DFT_NC.c
TRAN_Set_CentOverlap_NC.o: TRAN_Set_CentOverlap_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_CentOverlap_NC.c
TRAN_Set_SurfOverlap_NC.o: TRAN_Set_SurfOverlap_NC.c tran_variables.h tran_prototypes.h
	$(CC) -c TRAN_Set_SurfOverlap_NC.c

# S MitsuakiKAWAMURA                                                                     
MTRAN_EigenChannel.o: MTRAN_EigenChannel.c tran_prototypes.h \
	TRAN_Calc_SurfGreen.o TRAN_Calc_SelfEnergy.o TRAN_Calc_CentGreen.o
	$(CC) -c MTRAN_EigenChannel.c
TRAN_Channel_Functions.o: TRAN_Channel_Functions.c lapack_prototypes.h
	$(CC) -c TRAN_Channel_Functions.c
TRAN_Channel_Output.o: TRAN_Channel_Output.c openmx_common.h
	$(CC) -c TRAN_Channel_Output.c
TRAN_Calc_CurrentDensity.o: TRAN_Calc_CurrentDensity.c tran_prototypes.h lapack_prototypes.h
	$(CC) -c TRAN_Calc_CurrentDensity.c
TRAN_CDen_Main.o: TRAN_CDen_Main.c openmx_common.h lapack_prototypes.h tran_prototypes.h tran_variables.h
	$(CC) -c TRAN_CDen_Main.c
# E MitsuakiKAWAMURA                       

elpa1.o: elpa1.f90
	$(FC) -c elpa1.f90
solve_evp_real.o: solve_evp_real.f90 elpa1.o
	$(FC) -c solve_evp_real.f90
solve_evp_complex.o: solve_evp_complex.f90 elpa1.o
	$(FC) -c solve_evp_complex.f90

mod_precision.o: $(LIBELPADIR)/mod_precision.F90
	$(FC) -c $(LIBELPADIR)/mod_precision.F90
elpa_utilities.o: $(LIBELPADIR)/elpa_utilities.F90
	$(FC) -c $(LIBELPADIR)/elpa_utilities.F90
elpa1_compute_real.o: $(LIBELPADIR)/elpa1_compute_real.F90
	$(FC) -c $(LIBELPADIR)/elpa1_compute_real.F90
elpa1_compute_complex.o: $(LIBELPADIR)/elpa1_compute_complex.F90
	$(FC) -c $(LIBELPADIR)/elpa1_compute_complex.F90
aligned_mem.o: $(LIBELPADIR)/aligned_mem.F90
	$(FC) -c $(LIBELPADIR)/aligned_mem.F90
mod_pack_unpack_cpu_real.o: $(LIBELPADIR)/mod_pack_unpack_cpu_real.F90
	$(FC) -c $(LIBELPADIR)/mod_pack_unpack_cpu_real.F90
mod_pack_unpack_cpu_complex.o: $(LIBELPADIR)/mod_pack_unpack_cpu_complex.F90
	$(FC) -c $(LIBELPADIR)/mod_pack_unpack_cpu_complex.F90
mod_redist_band_real.o: $(LIBELPADIR)/mod_redist_band_real.F90
	$(FC) -c $(LIBELPADIR)/mod_redist_band_real.F90
mod_redist_band_complex.o: $(LIBELPADIR)/mod_redist_band_complex.F90
	$(FC) -c $(LIBELPADIR)/mod_redist_band_complex.F90
elpa2_determine_workload.o: $(LIBELPADIR)/elpa2_determine_workload.F90
	$(FC) -c $(LIBELPADIR)/elpa2_determine_workload.F90
real.o: $(LIBELPADIR)/real.F90
	$(FC) -c $(LIBELPADIR)/real.F90
complex.o: $(LIBELPADIR)/complex.F90
	$(FC) -c $(LIBELPADIR)/complex.F90
mod_single_hh_trafo_real.o: $(LIBELPADIR)/mod_single_hh_trafo_real.F90
	$(FC) -c $(LIBELPADIR)/mod_single_hh_trafo_real.F90
mod_compute_hh_trafo_real.o: $(LIBELPADIR)/mod_compute_hh_trafo_real.F90
	$(FC) -c $(LIBELPADIR)/mod_compute_hh_trafo_real.F90
mod_compute_hh_trafo_complex.o: $(LIBELPADIR)/mod_compute_hh_trafo_complex.F90
	$(FC) -c $(LIBELPADIR)/mod_compute_hh_trafo_complex.F90
elpa2_compute_real.o: $(LIBELPADIR)/elpa2_compute_real.F90
	$(FC) -c $(LIBELPADIR)/elpa2_compute_real.F90
elpa2_compute_complex.o: $(LIBELPADIR)/elpa2_compute_complex.F90
	$(FC) -c $(LIBELPADIR)/elpa2_compute_complex.F90
elpa_solve_evp_real_2stage_double_impl.o: $(LIBELPADIR)/elpa_solve_evp_real_2stage_double_impl.F90
	$(FC) -c $(LIBELPADIR)/elpa_solve_evp_real_2stage_double_impl.F90
elpa_solve_evp_complex_2stage_double_impl.o: $(LIBELPADIR)/elpa_solve_evp_complex_2stage_double_impl.F90
	$(FC) -c $(LIBELPADIR)/elpa_solve_evp_complex_2stage_double_impl.F90


#
#
# bandgnu13
#
#

bandgnu13: bandgnu13.c
	   gcc bandgnu13.c -lm -o bandgnu13
bin2txt: bin2txt.c
	   gcc bin2txt.c -lm -o bin2txt
cube2xsf: cube2xsf.c
	   gcc cube2xsf.c -lm -o cube2xsf
intensity_map: intensity_map.c
	   gcc intensity_map.c -lm -o intensity_map
md2axsf: md2axsf.c
	   gcc md2axsf.c -lm -o md2axsf

#
#
# kSpin
#
#

kSpin: kSpin.o BD.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o Tools_Search.o FL.o GC.o MO.o
	$(CC) $^ $(LIB) -lm -o $@

kSpin.o: kSpin.c BandDispersion.h read_scfout.h Inputtools.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK

Inputtools_kSpin.o: Inputtools_kSpin.c
	$(CC) -c $<

Tools_BandCalc.o: Tools_BandCalc.c lapack_prototypes.h read_scfout.h Tools_BandCalc.h
	$(CC) -c $<

GetOrbital.o: GetOrbital.c Inputtools.h read_scfout.h Tools_BandCalc.h
	$(CC) -c $<

EigenValue_Problem.o: EigenValue_Problem.c read_scfout.h Tools_BandCalc.h lapack_prototypes.h f77func.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $<

Eigen_HH.o: Eigen_HH.c read_scfout.h Tools_BandCalc.h lapack_prototypes.h f77func.h Eigen_HH.h
	$(CC) -c $<

Tools_Search.o: Tools_Search.c Tools_BandCalc.h EigenValue_Problem.h
	$(CC) -c $<

BandDispersion: BandDispersion.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

BandDispersion.o: BandDispersion.c GetOrbital.h Inputtools.h read_scfout.h Tools_BandCalc.h EigenValue_Problem.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $< 

BD.o: BandDispersion.c GetOrbital.h Inputtools.h read_scfout.h Tools_BandCalc.h EigenValue_Problem.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $< -DSIGMAEK -o $@

Circular_Search.o: Circular_Search.c GetOrbital.h Inputtools.h read_scfout.h Tools_BandCalc.h EigenValue_Problem.h Eigen_HH.h EigenValue_Problem.h
	$(CC) -c $<

#
#
# MulPCalc
#
#

MulPCalc: MulPCalc.o Tools_BandCalc.o Inputtools_kSpin.o
	$(CC) $^ $(LIB) -lm -o $@

MulPCalc.o: MulPCalc.c Tools_BandCalc.h Inputtools.h read_scfout.h
	$(CC) -c $<

#
#
# ADenBand
#
#

ADenBand: ADenBand.o Tools_BandCalc.o Inputtools_kSpin.o
	$(CC) $^ $(LIB) -lm -o $@

ADenBand.o: ADenBand.c Tools_BandCalc.h Inputtools.h read_scfout.h
	$(CC) -c $<

#
#
# FermiLoop
#
#

FermiLoop: FermiLoop.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

FermiLoop.o: FermiLoop.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< 

FL.o: FermiLoop.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK -o $@

#
#
# GridCalc
#
#

GridCalc: GridCalc.o read_scfout.o Inputtools_kSpin.o Tools_BandCalc.o GetOrbital.o EigenValue_Problem.o Eigen_HH.o SigmaEK.o
	$(CC) $^ $(LIB) -lm -o $@

GridCalc.o: GridCalc.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< 

GC.o: GridCalc.c read_scfout.h Inputtools.h EigenValue_Problem.h Eigen_HH.h Tools_BandCalc.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK -o $@

#
#
# MulPOnly
#
#

MulPOnly: MulPOnly.o read_scfout.o Inputtools_kSpin.o EigenValue_Problem.o Eigen_HH.o Tools_BandCalc.o GetOrbital.o SigmaEK.o
	$(CC) $^ $(LIB) -o $@

MulPOnly.o: MulPOnly.c Tools_BandCalc.h Inputtools.h read_scfout.h EigenValue_Problem.h GetOrbital.h SigmaEK.o
	$(CC) -c $< 

MO.o: MulPOnly.c Tools_BandCalc.h Inputtools.h read_scfout.h EigenValue_Problem.h GetOrbital.h
	$(CC) -c $< -DSIGMAEK -o $@

SigmaEK.o: SigmaEK.c
	$(CC) -c $< -o $@


