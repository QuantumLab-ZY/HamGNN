#-------------------------------------------------------------------------------
# defaults
#-------------------------------------------------------------------------------
# FC       = ifort # -assume byterecl 
FC       = gfortran
# FC       = x86_64-w64-mingw32-gfortran

# FFLAGS   = -g -O3 -traceback  -check bounds  -qopenmp # -parallel -heap-arrays
FFLAGS   = -g -fbacktrace -fbounds-check -fopenmp # -Wall -Wextra -Wconversion
MAKE     = make
JSONINC  = # -Ijson-fortran-build/include
LLIBS    = # json-fortran-build/lib/libjsonfortran.a

#-------------------------------------------------------------------------------
# Src
#-------------------------------------------------------------------------------

SRC = hsx_m.f90 dump.f90


OBJ = $(SRC:.f90=.o)
EXE = hsxdump

#-------------------------------------------------------------------------------
# Suffix rules
#-------------------------------------------------------------------------------
.SUFFIXES: .o .f90
.f90.o:
	$(FC) $(FFLAGS) -c $< $(JSONINC)


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------
tdm:	$(OBJ)
	$(FC) $(FFLAGS) -o $(EXE) $(OBJ) $(LLIBS) 

clean:
	rm -f *.mod
	rm -f $(OBJ)
tar:
	tar -czvf hfnamd.tgz *.f90 Makefile
tag:
	ctags *.f90
