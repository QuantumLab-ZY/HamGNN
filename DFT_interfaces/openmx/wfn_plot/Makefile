#-------------------------------------------------------------------------------
# compilers
#-------------------------------------------------------------------------------
CC      = gcc
MPICC   = mpicc

#-------------------------------------------------------------------------------
# flags
#-------------------------------------------------------------------------------
CFLAGS  = -O2 -Wall -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-but-set-variable -Wno-stringop-truncation
LDFLAGS =
LIBS    = -lm

# MPI+OpenMP build
MPICFLAGS = -O2 -Wall -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-but-set-variable -Wno-stringop-truncation -DUSE_MPI -fopenmp
MPILIBS = -fopenmp

#-------------------------------------------------------------------------------
# sources and targets
#-------------------------------------------------------------------------------
SRC   = wfn2cube.c
EXE   = wfn2cube
EXE_MPI = wfn2cube_mpi

#-------------------------------------------------------------------------------
# targets
#-------------------------------------------------------------------------------
.PHONY: all mpi test clean

all: $(EXE)

mpi: $(EXE_MPI)

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

$(EXE_MPI): $(SRC)
	$(MPICC) $(MPICFLAGS) -o $@ $< $(MPILIBS) $(LIBS)

test:
	$(CC) $(CFLAGS) -DTEST_DAT -o wfn2cube_test $(SRC) $(LDFLAGS) $(LIBS)
	./wfn2cube_test
	rm -f wfn2cube_test

clean:
	rm -f $(EXE) $(EXE_MPI) wfn2cube_test test_output.cube
