static char Version_OpenMX[30] = "3.9"; /* version of OpenMX */ 
   
#define PI              3.1415926535897932384626
#define BYTESIZE        8                        /* Don't change!! */
#define kB              0.00008617251324000000   /* eV/K           */          
#define BohrR           0.529177249              /* Angstrom       */
#define eV2Hartree      27.2113845                
#define electron_mass                 0.000910938291 /* [10^{-27} kg] */
#define unified_atomic_mass_unit      1.660538921    /* [10^{-27} kg] */

#define NYOUSO       60        /* # of YOUSO                                      */
/* #define YOUSO1                 # of atoms in the system                        */ 
/* #define YOUSO2                 # of atoms in hopping cluster                   */
/* #define YOUSO3                 maximum # of recursion levels for GEVP          */
/* #define YOUSO4                 # of 1D-copied cells                            */
/* #define YOUSO5                 size of the first index of HNL                  */
#define YOUSO6      130        /* no role */
/* #define YOUSO7                 max # of orbitals including an atom             */
/* #define YOUSO8                 max # of atoms in a rcut-off cluster            */
/* #define YOUSO9                 maximum # of recursion levels for inverse S     */
#define YOUSO10     500        /* length of a file name                           */
/* #define YOUSO11                max # of grids for an atom                      */
/* #define YOUSO12                max # of overlapping grids                      */
#define YOUSO14     104        /* number of elements in the periodic table        */
/* #define YOUSO15                max # of normK-grids                            */
/* #define YOUSO16                max # of previous steps in real-space Pulay mixing */
/* #define YOUSO17                max # of 1D-grids for FFT in the unit cell      */
/* #define YOUSO18                max # of species in the system                  */
/* #define YOUSO19                # of radial projection obitals in KB potential  */
/* #define YOUSO20                # of total projection obitals in KB potential   */
/* #define YOUSO21                # of mesh for radial wave functions             */
/* #define YOUSO22                # of mesh VPS                                   */
/* #define YOUSO23                # of spin polization                            */
/* #define YOUSO24                max multiplicity of radial wave functions       */
/* #define YOUSO25                max # of L for orbital                          */
#define YOUSO26      61         /* size of the second index in Gxyz[][]           */
/* #define YOUSO27                # of grids along "x"-axis in the k-space        */
/* #define YOUSO28                # of grids along "y"-axis in the k-space        */
/* #define YOUSO29                # of grids along "z"-axis in the k-space        */
/* #define YOUSO30                max # of L for KB projectors                    */
/* #define YOUSO31                # of HOMOs for output                           */
/* #define YOUSO32                # of LUMOs for output                           */
/* #define YOUSO33                # of MO_Nkpoint                                 */
/* #define YOUSO34                # of radial parts in VNA projector expansion    */
/* #define YOUSO35                max L of projectors in VNA projector expansion  */
#define YOUSO36      14           /* max L in Comp2Real array                       */
/* #define YOUSO37                max # of charge states in LESP                  */
/* #define YOUSO38                max # of previous steps in k-space Pulay mixing */
/* #define YOUSO39                max # of previous steps in h-space Pulay mixing */

#define Supported_MaxL      4        /* supported max angular momentum for basis orbital */

#define Radial_kmin       10e-7      /* the minimum radius in the radial k-space */
#define CoarseGL_Mesh     150        /* # of grids in coarse Gauss-Legendre quadrature */
#define GL_Mesh           300        /* # of grids in Gauss-Legendre quadrature */
#define FineGL_Mesh       1500       /* # of grids in fine Gauss-Legendre quadrature */

#define Threshold_OLP_Eigen  1.0e-9  /* threshold for cutting off eigenvalues of OLP */
#define fp_bsize         2097152     /* buffer size for setvbuf */
#define Shift_K_Point     1.0e-6     /* disturbance for stabilization of eigenvalue routine */

#define LAPACK_ABSTOL     6.0e-15    /* absolute error tolerance for lapack routines */

#define penalty_value_CoreHole   100    /* penalty value for creation of core hole */

#define Host_ID             0        /* ID of the host CPU in MPI */


int Temp_MD_iter;


typedef float     Type_DS_VNA;          /* type of DS_VNA */
#define MPI_Type_DS_VNA  MPI_FLOAT      /* type of DS_VNA */

typedef float     Type_Orbs_Grid;       /* type of Orbs_Grid */
#define ___Type_Orbs_Grid_definition___
#define MPI_Type_Orbs_Grid  MPI_FLOAT   /* type of Orbs_Grid */


#ifndef ___INTEGER_definition___
typedef int INTEGER; /* for fortran integer */
#define ___INTEGER_definition___ 
#endif

#ifndef ___dcomplex_definition___
typedef struct { double r,i; } dcomplex;
#define ___dcomplex_definition___ 
#endif

/* FFT radix */
static int NfundamentalNum=4;
static int fundamentalNum[4]={2,3,5,7};

#ifndef __sqr_definition___
#define sqr(x)   ( (x)*(x) )
#define __sqr_definition___
#endif

#ifdef nompi

#ifndef ___MPI_Comm_definition___
typedef int MPI_Comm;
#define ___MPI_Comm_definition___ 
#endif

#ifndef ___MPI_Status_definition___
typedef struct MPIStatus{int i;}  MPI_Status;  
#define ___MPI_Status_definition___ 
#endif

#ifndef ___MPI_Request_definition___
typedef struct MPIRequest{int i;} MPI_Request;  
#define ___MPI_Request_definition___ 
#endif

#else
#include "mpi.h"
#endif

MPI_Comm  mpi_comm_level1;
MPI_Comm  MPI_COMM_WORLD1;

#include "f77func.h"

/*---------- added by TOYODA 08/JAN/2010 */
#include "exx_interface_openmx.h"
/*---------- until here */


#ifndef ___logical_definition___
typedef long int logical;
#define ___logical_definition___ 
#endif

typedef long int integer;
typedef double doublereal;
typedef short ftnlen;
typedef short flag;
typedef short ftnint;


/*****************************************************************************
                             once allocated arrays
*****************************************************************************/
 
/*******************************************************
 char **SpeName; 
 character symbol of species
  size: SpeName[SpeciesNum][YOUSO10]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
char **SpeName;

/*******************************************************
 char **SpeBasis; 
 character symbol of a basis set assigned to species
  size: SpeBasis[SpeciesNum][YOUSO10]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
char **SpeBasis;

/*******************************************************
 char **SpeBasisName; 
 file name of a basis set assigned to species
  size: SpeBasisName[SpeciesNum][YOUSO10]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
char **SpeBasisName;

/*******************************************************
  char **SpeVPS; 
  file name of pseudo potentials set assigned to species
  size: SpeBasisVPS[SpeciesNum][YOUSO10]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
char **SpeVPS;

/*******************************************************
 double *Spe_AtomicMass; 
 atomic mass of each species, where hydrogen is 1.  
  size: Spe_AtomicMass[SpeciesNum]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Spe_AtomicMass;

/*******************************************************
 int *Spe_MaxL_Basis; 
 the maximum "l" component of used atomic orbitals inv
 each species
  size: Spe_MaxL_Basis[SpeciesNum]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_MaxL_Basis;

/*******************************************************
 int **Spe_Num_Basis; 
 the number of multiplicity of primitive radial parts
 for each "l" component in an species
  size: Spe_Num_Basis[SpeciesNum][6]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Spe_Num_Basis;

/*******************************************************
 int **Spe_Num_CBasis; 
 the number of multiplicity of contracted radial parts
 for each "l" component in an species
  size: Spe_Num_CBasis[SpeciesNum][6]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Spe_Num_CBasis;

/*******************************************************
 double **EH0_scaling;
  scaling factors to vanish Ecore plus EH0
  at the cutoff radius  
  size: EH0_scaling[SpeciesNum][SpeciesNum]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **EH0_scaling;

/*******************************************************
 double **SO_factor;
  scaling factors of spin-orbit coupling
  size: SO_factor[SpeciesNum][4]
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **SO_factor;

/*******************************************************
 double ***Hub_U_Basis          --- added by MJ
 the value of Hubbard U for LDA+U calculation
  size: Hub_U_Basis[SpeciesNum][Spe_MaxL_Basis+1][Spe_Num_Basis]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***Hub_U_Basis ;      /* --- added by MJ  */

/*******************************************************
 double ***Hund_J_Basis         --- by S.Ryee
 the value of Hund J for general LDA+U scheme
 (when Hub_type=2 is used)
  size: Hund_J_Basis[SpeciesNum][Spe_MaxL_Basis+1][Spe_Num_Basis]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***Hund_J_Basis;      /* by S.Ryee */

/*******************************************************
 int ***Nonzero_UJ         --- by S.Ryee
 Index for orbitals having nonzero input values of Hubbard U and Hund J
 (when Hub_type=2 is used)
  size: Nonzero_UJ[SpeciesNum][Spe_MaxL_Basis+1][Spe_Num_Basis]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int ***Nonzero_UJ;      /* by S.Ryee */

/*******************************************************
 double *****Coulomb_Array        --- by S.Ryee
 Coulomb interaction tensor for general LDA+U scheme 
 (when Hub_type=2 is used)
 free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****Coulomb_Array;      /* by S.Ryee */

/*******************************************************
 double *****AMF_Array        --- by S.Ryee
 Density matrix for LDA+U with AMF-type double counting.
 (when Hub_type=2 && dc_Type=2 or 4 is used)
 free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****AMF_Array;      /* by S.Ryee */

/*******************************************************
 double **Bessel_j_Array0        --- by S.Ryee
 Bessel function array for calculating Slater integrals. 
 (when Hub_type=2 && Yukawa_on=1 is used)
*******************************************************/
double **Bessel_j_Array0;      /* by S.Ryee */

/*******************************************************
 double **Bessel_h_Array0        --- by S.Ryee
 Hankel function array for calculating Slater integrals. 
 (when Hub_type=2 && Yukawa_on=1 is used)
*******************************************************/
double **Bessel_h_Array0;      /* by S.Ryee */

/*******************************************************
 int *OrbPol_flag          --- added by MJ and TO
  flag that spefifies how orbital is polarized
  size: OrbPol[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *OrbPol_flag ;          /* --- added by MJ and TO  */

/*******************************************************
 double **Gxyz; 
 atomic global coordinates, velocities, and gradients of
 the total energy with respect to the atomic coordinates
  size: Gxyz[atomnum+4][YOUSO26]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Gxyz;

/********************************************************
 double ***GxyzHistoryIn;
  History of atomic global coordinates
  size GxyzHistoryIn[M_GDIIS_HISTORY+1][atomnum+4][4];
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***GxyzHistoryIn;

/********************************************************
 double ***GxyzHistoryR;
  History of residual global coordinates
  size GxyzHistoryR[M_GDIIS_HISTORY+1][atomnum+4][4];
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***GxyzHistoryR;

/********************************************************
 double **His_Gxyz;
  history of atomic global coordinates for charge extrapolation
  size His_Gxyz[Extrapolated_Charge_History][3*atomnum];
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **His_Gxyz;

/*******************************************************
 int **atom_Fixed_XYZ;
  atomic global coordinates, =0 relax, =1 fix position
  size: atom_Fixed_XYZ[atomnum+1][3];
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **atom_Fixed_XYZ;

/*******************************************************
 double **Cell_Gxyz; 
 atomic global coordinates spanned
 by the unit cell vectors
  size: Cell_Gxyz[atomnum+1][4]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Cell_Gxyz;

/*******************************************************
 double *InitN_USpin; 
  the number of the upspin electon of initial atoms
  size: InitN_USpin[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitN_USpin;

/*******************************************************
 double *InitN_DSpin; 
  the number of the upspin electon of initial atoms
  size: InitN_DSpin[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitN_DSpin;

/*******************************************************
 double *InitMagneticMoment; 
  initial magnetic moment of each atom
  size: InitMagneticMoment[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitMagneticMoment;

/*******************************************************
 double *Angle0_Spin; 
  angle of theta for atomic projected spin moment
  size: Angle0_Spin[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Angle0_Spin;

/*******************************************************
 double *Angle1_Spin; 
  angle of phi for atomic projected spin moment
  size: Angle1_Spin[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Angle1_Spin;

/*******************************************************
 double *InitAngle0_Spin; 
  initial angle of theta for atomic projected spin moment
  size: InitAngle0_Spin[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitAngle0_Spin;

/*******************************************************
 double *InitAngle1_Spin; 
  initial angle of phi for atomic projected spin moment
  size: InitAngle1_Spin[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitAngle1_Spin;

/*******************************************************
 double *Angle0_Orbital; 
  angle of theta for atomic projected orbital moment
  size: Angle0_Orbital[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Angle0_Orbital;

/*******************************************************
 double *Angle1_Orbital; 
  angle of phi for atomic projected orbital moment
  size: Angle1_Orbital[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Angle1_Orbital;

/*******************************************************
 double *OrbitalMoment;
  magnitude of atomic projected orbital moment
  size: OrbitalMoment[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *OrbitalMoment;

/*******************************************************
 int *Constraint_SpinAngle; 
  flag for constraining the spin angle of atomic projected spin
  size: Constraint_SpinAngle[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Constraint_SpinAngle;

/*******************************************************
 double *InitAngle0_Orbital; 
  initial angle of theta for atomic projected orbital moment
  size: InitAngle0_Orbital[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitAngle0_Orbital;

/*******************************************************
 double *InitAngle1_Orbital; 
  initial angle of phi for atomic projected orbital moment
  size: InitAngle1_Orbital[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *InitAngle1_Orbital;

/*******************************************************
 double **Orbital_Moment_XYZ;
  x-, y-, and z- components of orbital moment calculated
  at each atomic site
  size: Orbital_Moment_XYZ[atomnum+1][3]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Orbital_Moment_XYZ;

/*******************************************************
 int *Constraint_OrbitalAngle; 
  flag for constraining the orbital moment angle of atomic 
  projected orbital moment
  size: Constraint_OrbitalAngle[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Constraint_OrbitalAngle;

/*******************************************************
 int *WhatSpecies; 
 array to specify species for each atom in the system 
  size: WhatSpecies[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *WhatSpecies;

/*******************************************************
 int *GridN_Atom; 
 the number of grids overlaping to each atom
  size: GridN_Atom[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *GridN_Atom;

/*******************************************************
 double *NormK; 
 radial grid values in the reciprocal space
  size: NormK[Ngrid_NormK+1]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *NormK;

/*******************************************************
 double *Spe_Atom_Cut1; 
 cutoff radius of atomic orbitals for each species
  size: Spe_Atom_Cut1[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Spe_Atom_Cut1;

/*******************************************************
 double *Spe_Core_Charge; 
 effective core charge of each species
  size: Spe_Core_Charge[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Spe_Core_Charge;

/*******************************************************
 int *TGN_EH0; 
 the number of 3D grids for calculating EH0 in
 Correction_Energy.c
  size: TGN_EH0[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *TGN_EH0;

/*******************************************************
 double *dv_EH0; 
 the volume of a grid for calculating EH0 in
 Correction_Energy.c
  size: dv_EH0[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *dv_EH0;

/*******************************************************
 int *Spe_Num_Mesh_VPS; 
 the number of grids for pseudo potentials
  size: Spe_Num_Mesh_VPS[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Num_Mesh_VPS;

/*******************************************************
 int *Spe_Num_Mesh_PAO; 
 the number of grids for atomic orbitals
  size: Spe_Num_Mesh_PAO[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Num_Mesh_PAO;

/*******************************************************
 int *Spe_Total_VPS_Pro; 
 the total number of projector in KB nonlocal potentials
  size: Spe_Total_VPS_Pro[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Total_VPS_Pro;

/*******************************************************
 int *Spe_Num_RVPS; 
 the number of radial projectors in KB nonlocal potentials
  size: Spe_Num_RVPS[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Num_RVPS;

/*******************************************************
 int *Spe_PAO_LMAX; 
 the maximum "l" component of atomic orbitals stored
 in the file of each species
  size: Spe_PAO_LMAX[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_PAO_LMAX;

/*******************************************************
 int *Spe_PAO_Mul; 
 the multiplicity of radial wave functions for each "l"
 component of atomic orbitals stored in the file of each
 species
  size: Spe_PAO_Mul[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_PAO_Mul;

/*******************************************************
 int *Spe_WhatAtom; 
 atomic number in the periodic table for each species
  size: Spe_WhatAtom[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_WhatAtom;

/*******************************************************
 int *Spe_Total_NO; 
 the number of primitive atomic orbitals in a species
  size: Spe_Total_NO[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Total_NO;

/*******************************************************
 int *Spe_Total_CNO; 
 the number of contracted atomic orbitals in a species
  size: Spe_Total_CNO[SpeciesNum]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Total_CNO;

/*******************************************************
 int *FNAN_DCLNO,*SNAN_DCLNO; 
 the number of first and second neighboring atoms 
 which are referred in the DC-LNO calculation.
  size: FNAN_DCLNO[atomnum+1], SNAN_DCLNO[atomnum+1]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *FNAN_DCLNO,*SNAN_DCLNO;

/*******************************************************
 int *FNAN; 
 the number of first neighboring atoms
  size: FNAN[atomnum+1]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *FNAN;

/*******************************************************
 int *SNAN; 
 the number of second neighboring atoms
  size: SNAN[atomnum+1]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *SNAN;

/*******************************************************
 int *ONAN; 
 the number of third neighboring atoms
  size: ONAN[atomnum+1]
  allocation: call as Allocate_Arrays(2) in readfile.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *ONAN;

/*******************************************************
 int **natn; 
  grobal index number of neighboring atoms of an atom ct_AN
  size: natn[atomnum+1][Max_FSNAN*ScaleSize+1]
  allocation: call as Allocate_Arrays(3) in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **natn;

/*******************************************************
 int **ncn; 
  grobal index number for cell of neighboring atoms of
  an atom ct_AN
  size: ncn[atomnum+1][Max_FSNAN*ScaleSize+1]
  allocation: call as Allocate_Arrays(3) in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **ncn;

/*******************************************************
 int **natn_onan; 
  grobal index number of neighboring ONAN atoms of an atom ct_AN
  size: natn_onan[atomnum+1][Max_FSNAN*ScaleSize+1]
  allocation: truncation.c 
  free:       truncation.c and Free_Arrays(0) in openmx.c
*******************************************************/
int **natn_onan;

/*******************************************************
 int **ncn_onan1,**ncn_onan2,**ncn_onan3; 
  cell indices of neighboring ONAN atoms of an atom ct_AN
  size: ncn_onan[atomnum+1][Max_FSNAN*ScaleSize+1]
  allocation: truncation.c 
  free:       truncation.c and Free_Arrays(0) in openmx.c
*******************************************************/
int **ncn_onan1,**ncn_onan2,**ncn_onan3; 

/*******************************************************
 double **Dis; 
  distance to neighboring atoms of an atom ct_AN
  size: Dis[atomnum+1][Max_FSNAN*ScaleSize+1]
  allocation: call as Allocate_Arrays(3) in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Dis;

/*******************************************************
 double **GridX_EH0, **GridY_EH0, **GridZ_EH0 ;
  x,y,and z-coordinates of grids for calculating EH0 in
  Correction_Energy.c
  size: GridX_EH0[SpeciesNum][Max_TGN_EH0]
  allocation: call as Allocate_Arrays(4)
              in Correcion_Energy.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **GridX_EH0,**GridY_EH0,**GridZ_EH0;

/*******************************************************
 double **Arho_EH0;
  atomic density on grids for calculating EH0 in
  Correction_Energy.c
  size: Arho_EH0[SpeciesNum][Max_TGN_EH0]
  allocation: call as Allocate_Arrays(4)
              in Correcion_Energy.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Arho_EH0;

/*******************************************************
 double ****Arho_EH0_Orb;
  orbitally resolved atomic density on grids for 
  calculating DecEH0 in Correction_Energy.c
  size: Arho_EH0_Orb[SpeciesNum][Max_TGN_EH0][Spe_MaxL_Basis][Spe_Num_Basis]
  allocation: call as Allocate_Arrays(4)
              in Correcion_Energy.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****Arho_EH0_Orb;

/*******************************************************
 double **Wt_EH0;
  quadrature weight of grids for calculating EH0
  in Correction_Energy.c
  size: Wt_EH0[SpeciesNum][Max_TGN_EH0]
  allocation: call as Allocate_Arrays(4)
              in Correcion_Energy.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Wt_EH0;

/*******************************************************
 double **atv;
  xyz translation vectors of periodically copied cells
  size: atv[TCpyCell+1][4]
  allocation: in Set_Periodic() of truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **atv;

/*******************************************************
 int **ratv;
  queue number of periodically copied cells
  size: ratv[TCpyCell*2+4][TCpyCell*2+4][TCpyCell*2+4];
  allocation: in Set_Periodic() of truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int ***ratv;

/*******************************************************
 int **atv_ijk;
  i,j,and j number of periodically copied cells
  size: atv_ijk[TCpyCell+1][4];
  allocation: in Set_Periodic() of truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **atv_ijk;

/*******************************************************
 double **MO_kpoint;
  kpoints at which wave functions are calculated.
  size: MO_kpoint[MO_Nkpoint+1][4]
  allocation: call as Allocate_Arrays(5) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **MO_kpoint;

/*******************************************************
 double **Spe_PAO_XV;
  radial mesh (x=log(r)) for PAO 
  size: Spe_PAO_XV[List_YOUSO[18]]
                  [List_YOUSO[21]]
  allocation: call as Allocate_Arrays(6) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_PAO_XV;

/*******************************************************
 double **Spe_PAO_RV;
  logarithmic radial mesh (r=exp(r)) for PAO 
  size: Spe_PAO_XV[List_YOUSO[18]]
                  [List_YOUSO[21]]
  allocation: call as Allocate_Arrays(6) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_PAO_RV;

/*******************************************************
 double **Spe_Atomic_Den;
  atomic charge density on radial mesh of PAO 
  size: Spe_Atomic_Den[List_YOUSO[18]]
                      [List_YOUSO[21]]
  allocation: call as Allocate_Arrays(6) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_Atomic_Den;

/*******************************************************
 double **Spe_Atomic_Den2;
  atomic charge density+PCC charge on radial mesh of PAO, 
  where both the edges are extended by adding one more. 
  size: Spe_Atomic_Den[List_YOUSO[18]]
                      [List_YOUSO[21]+2]
  allocation: call as Allocate_Arrays(6) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_Atomic_Den2;

/*******************************************************
 double ****Spe_PAO_RWF;
  radial parts of basis orbitals on radial mesh of PAO 
  size: Spe_PAO_RWF[List_YOUSO[18]]
                   [List_YOUSO[25]+1]
                   [List_YOUSO[24]]
                   [List_YOUSO[21]]
  allocation: call as Allocate_Arrays(6) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****Spe_PAO_RWF;

/*******************************************************
 double ****Spe_RF_Bessel;
  radial parts of basis orbitals on radial mesh in the
  momentum space
  size: Spe_RF_Bessel[List_YOUSO[18]]
                     [List_YOUSO[25]+1]
                     [List_YOUSO[24]]
                     [List_YOUSO[15]]
  allocation: call as Allocate_Arrays(6) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****Spe_RF_Bessel;

/*******************************************************
 double **Spe_VPS_XV;
  radial mesh (x=log(r)) for VPS 
  size: Spe_VPS_XV[List_YOUSO[18]]
                  [List_YOUSO[22]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_VPS_XV;

/*******************************************************
 double **Spe_VPS_XV;
  logarithmic radial mesh (r=exp(x)) for VPS 
  size: Spe_VPS_RV[List_YOUSO[18]]
                  [List_YOUSO[22]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_VPS_RV;

/*******************************************************
 double **Spe_Vna;
  neutral atom potentials on radial mesh of VPS
  size: Spe_Vna[List_YOUSO[18]]
               [List_YOUSO[22]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_Vna;

/*******************************************************
 double **Spe_VH_Atom;
  Hartree potentials of atomic charge densities
  on radial mesh of VPS
  size: Spe_VH_Atom[List_YOUSO[18]]
                   [List_YOUSO[22]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_VH_Atom;

/*******************************************************
 double **Spe_Atomic_PCC;
  partial core correction charge densities on radial
  mesh of VPS
  size: Spe_Atomic_PCC[List_YOUSO[18]]
                      [List_YOUSO[22]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_Atomic_PCC;

/*******************************************************
 double ****Spe_VNL;
  radial parts of projectors of non-local potentials
  on radial mesh of VPS
  size: Spe_VNL[SO_switch+1]
               [List_YOUSO[18]]
               [List_YOUSO[19]]
               [List_YOUSO[22]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****Spe_VNL;

/*******************************************************
 double ***Spe_VNLE;
  projection energies of projectors of non-local
  potentials on radial mesh of VPS
  size: Spe_VNLE[SO_switch+1]
                [List_YOUSO[18]]
                [List_YOUSO[19]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***Spe_VNLE;

/*******************************************************
 int **Spe_VPS_List;
  angular momentum numbers of projectors of non-local
  potentials
  size: Spe_VPS_List[List_YOUSO[18]]
                    [List_YOUSO[19]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Spe_VPS_List;

/*******************************************************
 double ****Spe_NLRF_Bessel;
  radial parts of projectors of non-local potentials
  on radial mesh in the momentum space
  size: Spe_RF_Bessel[SO_switch+1]
                     [List_YOUSO[18]]
                     [List_YOUSO[19]+2]
                     [List_YOUSO[15]]
  allocation: call as Allocate_Arrays(7) in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****Spe_NLRF_Bessel;

/*****************************************************************************
                   allocated arrays at every MD step 
*****************************************************************************/

/*******************************************************
 int ***GListTAtoms1;
  grid index (local for ct_AN) overlaping between
  two orbitals 
  size: GListTAtoms1[Matomnum+1]
                    [FNAN[Gc_AN]+1]
                    [NumOLG[Mc_AN][h_AN]]
  allocation: UCell_Box() of truncation.c
  free:       truncation.c and Free_Arrays(0) in openmx.c
*******************************************************/
int ***GListTAtoms1;

/*******************************************************
 int ***GListTAtoms2;
  grid index (local for h_AN) overlaping between
  two orbitals 
  size: GListTAtoms2[Matomnum+1]
                    [FNAN[Gc_AN]+1]
                    [NumOLG[Mc_AN][h_AN]]
  allocation: UCell_Box() of truncation.c
  free:       truncation.c and Free_Arrays(0) in openmx.c
*******************************************************/
int ***GListTAtoms2;

/*******************************************************
 int **GridListAtom; 
  neighboring grid points of an atom Mc_AN
  size: GridListAtom[Matomnum+1][Max_GridN_Atom*ScaleSize+1]
  allocation: allocate in UCell_Box() of truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **GridListAtom;

/*******************************************************
 int **CellListAtom; 
  cell number of neighboring grid points of an atom Mc_AN
  size: CellListAtom[Matomnum+1][Max_GridN_Atom*ScaleSize+1]
  allocation: allocate in UCell_Box() of truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **CellListAtom;

/*******************************************************
 int **MGridListAtom; 
  neighboring grid points (medium variable) of an atom Mc_AN
  size: MGridListAtom[Matomnum+1][Max_GridN_Atom*ScaleSize+1]
  allocation: allocate in UCell_Box() of truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **MGridListAtom;

/*******************************************************
 double **Density_Grid; 
  electron densities on grids in the partition C
  size: Density_Grid[2 or 4][My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Density_Grid;

/*******************************************************
 double **Density_Grid_B; 
  electron densities on grids in the partition B
  size: Density_Grid[2 or 4][My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Density_Grid_B;

/*******************************************************
 double **Density_Grid_D; 
  electron densities on grids in the partition D
  size: Density_Grid[2 or 4][My_NumGridD]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Density_Grid_D;

/*******************************************************
 double *ADensity_Grid_B; 
  superposed atomic density on grids in the partition B
  size: ADensity_Grid_B[My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *ADensity_Grid_B;

/*******************************************************
 double **PCCDensity_Grid_B; 
  electron densities by the superposition of partial 
  core correction densities on grids in the partition B
  size: PCCDensity_Grid[2][My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **PCCDensity_Grid_B;

/*******************************************************
 double **PCCDensity_Grid_D; 
  electron densities by the superposition of partial 
  core correction densities on grids in the partition D
  size: PCCDensity_Grid[2][My_NumGridD]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **PCCDensity_Grid_D;

/*******************************************************
 double **Vxc_Grid; 
  exchange-correlation potentials on grids in the partition C
  size: Vxc_Grid[2 or 4][My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Vxc_Grid;

/*******************************************************
 double **Vxc_Grid_B; 
  exchange-correlation potentials on grids in the partition B
  size: Vxc_Grid_B[2 or 4][My_NumGridB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Vxc_Grid_B;

/*******************************************************
 double **Vxc_Grid_D; 
  exchange-correlation potentials on grids in the partition D
  size: Vxc_Grid_D[2 or 4][My_NumGridD]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Vxc_Grid_D;

/*******************************************************
 double *RefVxc_Grid; 
  exchange-correlation potentials on grids in the partition C
  for the reference charge density
  size: RefVxc_Grid[My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *RefVxc_Grid;

/*******************************************************
 double *RefVxc_Grid_B; 
  exchange-correlation potentials on grids in the partition B
  for the reference charge density
  size: RefVxc_Grid[My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *RefVxc_Grid_B;

/*******************************************************
 double *VNA_Grid; 
  neutral atom potential on grids in the partition C
  size: VNA_Grid[My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *VNA_Grid;

/*******************************************************
 double *VNA_Grid_B; 
  neutral atom potential on grids in the partition B
  size: VNA_Grid[My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *VNA_Grid_B;

/*******************************************************
 double *VEF_Grid; 
  potential on grids in the partition C by external electric field
  size: VEF_Grid[My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *VEF_Grid;

/*******************************************************
 double *VEF_Grid_B; 
  potential on grids in the partition B by external electric field
  size: VEF_Grid_B[My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *VEF_Grid_B;

/*******************************************************
 double *dVHart_Grid; 
  Hartree potential of the differential
  electron density on grids in the partition C
  size: dVHart_Grid[My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *dVHart_Grid;

/*******************************************************
 double *dVHart_Grid_B; 
  Hartree potential of the difference electron density 
  on grids in the partition B
  size: dVHart_Grid_B[My_Max_NumGridB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *dVHart_Grid_B;

/*******************************************************
 double *dVHart_Periodic_Grid_B; 
  Hartree potential of the periodic difference electron 
  density on grids in the partition B, which appears 
  in the core hole calculation.
  size: dVHart_Periodic_Grid_B[My_Max_NumGridB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *dVHart_Periodic_Grid_B;

/*******************************************************
 double *Density_Periodic_Grid_B; 
  Pre-calculated periodic density of a system without 
  a core hole on grids in the partition B, which appears 
  in the core hole calculation.
  size: Density_Periodic_Grid_B[My_Max_NumGridB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *Density_Periodic_Grid_B;

/*******************************************************
 double **Vpot_Grid; 
  Kohn-Sham effective potentials on grids in the partition C
  size: Vpot_Grid[2 or 4][My_NumGridC]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Vpot_Grid;

/*******************************************************
 double **Vpot_Grid_B; 
  Kohn-Sham effective potentials on grids in the partition B
  size: Vpot_Grid[2 or 4][My_NumGridB_AB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Vpot_Grid_B;

/*******************************************************
 double ***ReVKSk;
  real part of Kohn-Sham effective potentials 
  on reciprocal grids in the partition B
  size: ReVKSk[List_YOUSO[38]]
              [1, 2, or 3]
              [My_Max_NumGridB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***ReVKSk;

/*******************************************************
 double ***ImVKSk;
  imaginary part of Kohn-Sham effective potentials 
  on reciprocal grids in the partition B
  size: ImVKSk[List_YOUSO[38]]
              [1, 2, or 3]
              [My_Max_NumGridB]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***ImVKSk;

/*******************************************************
 double **Residual_ReVKSk;
  real part of residual for Kohn-Sham effective potentials 
  on reciprocal grids in the partition B
  size: Residual_ReVKSk[1, 2, or 3]
                       [My_NumGridB_CB*List_YOUSO[38]]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Residual_ReVKSk;

/*******************************************************
 double **Residual_ImVKSk;
  imaginary part of residual for Kohn-Sham effective potentials 
  on reciprocal grids in the partition B
  size: Residual_ImVKSk[1, 2, or 3]
                       [My_NumGridB_CB*List_YOUSO[38]]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **Residual_ImVKSk;

/*******************************************************
 Type_Orbs_Grid ***Orbs_Grid;
  values of basis orbitals on grids
  size: Orbs_Grid[Matomnum+1]
                 [GridN_Atom[Gc_AN]]
                 [Spe_Total_NO[Cwan]]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
Type_Orbs_Grid ***Orbs_Grid;

/*******************************************************
 Type_Orbs_Grid ***COrbs_Grid;
  values of contrated basis orbitals on grids
  size: COrbs_Grid[Matomnum+MatomnumF+1]
                  [Spe_Total_NO[Cwan]]
                  [GridN_Atom[Gc_AN]]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
Type_Orbs_Grid ***COrbs_Grid;

/*******************************************************
 Type_Orbs_Grid ***Orbs_Grid_FNAN;
  values of basis orbitals on grids for neighbor atoms
  which do not belong to my process.
  size: Orbs_Grid_FNAN[Matomnum+1]
                      [FNAN[Gc_AN]+1]
                      [NumOLG[Mc_AN][h_AN]]
                      [Spe_Total_NO[Cwan]]
  allocation: allocate in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
Type_Orbs_Grid ****Orbs_Grid_FNAN;

/*******************************************************
 double ***LNO_coes;
  LCAO coefficients for localized natural orbitals
  size: LNO_coes[SpinP_switch+1]
                [Matomnum+MatomnumF+MatomnumS+1]
                [List_YOUSO[7]*List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***LNO_coes;

/*******************************************************
 int *LNOs_Num_predefined;
  the number of LNOs that user defines
  size: LNOs_Num_predefined[SpeciesNum]
  allocation: Allocate_Arrays.c
  free:       Free_Arrays.c
*******************************************************/
int *LNOs_Num_predefined;

/*******************************************************
 double ***LNO_pops;
  populations for localized natural orbitals
  size: LNO_pops[SpinP_switch+1]
                [Matomnum+MatomnumF+MatomnumS+1]
                [List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***LNO_pops;

/*******************************************************
 int *LNO_Num;
  the number of localized natural orbitals for each atom
  size: LNO_pops[atomnum+1]
  allocation: allocate in Allocation_Arrays.c
  free:       Free_Arrays(0) in openmx.c
*******************************************************/
int *LNO_Num;

/*******************************************************
 double *****H0;
  matrix elements of basis orbitals for T+VNL
  size: H0[4]
          [Matomnum+1]
          [FNAN[Gc_AN]+1]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****H0;

/*******************************************************
 double *****CntH0;
  matrix elements of contracted basis orbitals for T+VNL
  size: CntH0[4]
             [Matomnum+1]
             [FNAN[Gc_AN]+1]
             [Spe_Total_CNO[Cwan]]
             [Spe_Total_CNO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****CntH0;

/*******************************************************
 double *****HNL;
  real matrix elements of basis orbitals for non-local VPS
  size: HNL[List_YOUSO[5]]
           [Matomnum+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****HNL;

/*******************************************************
 double *****iHNL;
  imaginary matrix elements of basis orbitals for non-local VPS
  size: iHNL[List_YOUSO[5]]
            [Matomnum+MatomnumF+MatomnumS+1]
            [FNAN[Gc_AN]+1]
            [Spe_Total_NO[Cwan]]
            [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****iHNL;

/*******************************************************
 double *****iCntHNL;
  imaginary matrix elements of contracted basis orbitals
  for non-local VPS
  size: iCntHNL[List_YOUSO[5]]
               [Matomnum+MatomnumF+MatomnumS+1]
               [FNAN[Gc_AN]+1]
               [Spe_Total_CNO[Cwan]]
               [Spe_Total_CNO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****iCntHNL;

/*******************************************************
 double *****HCH;
  real matrix elements for a core hole potential
  size: HCH[List_YOUSO[5]]
           [Matomnum+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****HCH;

/*******************************************************
 double *****iHCH;
  imaginary matrix elements for a core hole potential
  size: iHCH[List_YOUSO[5]]
            [Matomnum+MatomnumF+MatomnumS+1]
            [FNAN[Gc_AN]+1]
            [Spe_Total_NO[Cwan]]
            [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****iHCH;

/*******************************************************
 double *****OLP_L;
  <i|lx,ly,lz|j> overlap matrix elements with lx,y,z
  operator of basis orbitals which are used to calculate
  orbital moment
  size: OLP_L[3]
             [Matomnum+1]
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****OLP_L;

/*******************************************************
 double *****OLP;
  overlap matrix elements of basis orbitals
  size: OLP[4]
           [Matomnum+MatomnumF+MatomnumS+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****OLP;

/*** added by Ohwaki ***/

/*******************************************************
   double *****OLP_p;
    overlap matrix elements of basis orbitals
    size: OLP_p[4]
               [Matomnum+MatomnumF+MatomnumS+1]
               [FNAN[Gc_AN]+1]
               [Spe_Total_NO[Cwan]]
               [Spe_Total_NO[Hwan]]
   allocation: allocate in truncation.c
   free:       in truncation.c
               and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****OLP_p;

/*** added by Ohwaki (end) ***/

/*******************************************************
 double *****OLP_CH;
  scaled overlap matrix elements of basis orbitals  
  used for the core hole calculation 
  size: OLP_CH
           [4]
           [Matomnum+MatomnumF+MatomnumS+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****OLP_CH;

/*******************************************************
 double *****CntOLP;
  overlap matrix elements of contracted basis orbitals
  size: CntOLP[4]
              [Matomnum+MatomnumF+MatomnumS+1]
              [FNAN[Gc_AN]+1]
              [Spe_Total_CNO[Cwan]]
              [Spe_Total_CNO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****CntOLP;

/*******************************************************
 double *****H;
  Kohn-Sham matrix elements of basis orbitals
  size: H[SpinP_switch+1]
         [Matomnum+MatomnumF+MatomnumS+1]
         [FNAN[Gc_AN]+1]
         [Spe_Total_NO[Cwan]]
         [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****H;

/*******************************************************
 double *****CntH;
  Kohn-Sham matrix elements of contracted basis orbitals
  size: CntH[SpinP_switch+1]
            [Matomnum+MatomnumF+MatomnumS+1]
            [FNAN[Gc_AN]+1]
            [Spe_Total_CNO[Cwan]]
            [Spe_Total_CNO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****CntH;

/*******************************************************
 double ******HisH1;
  historical matrix elements corresponding to "H" in
  the Kohn-Sham matrix elements
  size: HisH1[List_YOUSO[39]]
             [SpinP_switch+1]
             [Matomnum+1]
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******HisH1;

/*******************************************************
 double ******HisH2;
  historical matrix elements corresponding to "iHNL" in
  the Kohn-Sham matrix elements
  size: HisH2[List_YOUSO[39]]
             [SpinP_switch]
             [Matomnum+1]
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******HisH2;

/*******************************************************
 double ******ResidualH1;
  historical matrix elements corresponding to "H" in
  the Kohn-Sham matrix elements
  size: ResidualH1[List_YOUSO[39]]
                  [SpinP_switch+1]
                  [Matomnum+1]
                  [FNAN[Gc_AN]+1]
                  [Spe_Total_NO[Cwan]]
                  [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******ResidualH1;

/*******************************************************
 double ******ResidualH2;
  residual matrix elements corresponding to "iHNL" in
  the Kohn-Sham matrix elements
  size: ResidualH2[List_YOUSO[39]]
                  [SpinP_switch]
                  [Matomnum+1]
                  [FNAN[Gc_AN]+1]
                  [Spe_Total_NO[Cwan]]
                  [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******ResidualH2;

/*******************************************************
 double *****H_Hub;             --- added by MJ
  real part of effective Hubbard Hamiltonian;
  same structure with OLP but the first dimension of OLP,
  the dimensions for derivatives, is not required in H_Hub.
  instead of it, H_Hub should have spin index.
  size: H_Hub[spin]
             [Matomnum+1 --> Mc_AN]  
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****H_Hub;           /* --- added by MJ  */

/*******************************************************
 double *****iHNL0;             --- added by TO
  imaginary matrix elements for non-local VPS
  size: iHNL[List_YOUSO[5]]
            [Matomnum+1]
            [FNAN[Gc_AN]+1]
            [Spe_Total_NO[Cwan]]
            [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****iHNL0;           /* --- added by TO  */

/*******************************************************
 double ******DS_NL;
  overlap matrix elements between projectors,
  of non-local potentials, and basis orbitals 
  size: DS_NL[SO_switch+1]
             [4]
             [Matomnum+2]
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_VPS_Pro[Hwan]+2] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******DS_NL;

/*******************************************************
 double ******CntDS_NL;
  overlap matrix elements between projectors, of non-local
  potentials, and contracted basis orbitals 
  size: CntDS_NL[SO_switch+1] 
                [4]
                [Matomnum+2]
                [FNAN[Gc_AN]+1]
                [Spe_Total_CNO[Cwan]]
                [Spe_Total_VPS_Pro[Hwan]+2] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******CntDS_NL;

/*******************************************************
 double ***H_Zeeman_NCO;          
  matrix element introduced by the constraint for orbital
  magnetic moments. Note that the matrix elements are purely
  imaginary. 
  size: H_Zeeman_NCO[Matomnum+1]  
                    [Spe_Total_NO[Cwan]]
                    [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***H_Zeeman_NCO;

/*******************************************************
 double ***TRAN_DecMulP;
  partial decomposed Mulliken population by CL or CR 
  overlapping 
  size: TRAN_DecMulP
          [SpinP_switch+1]
          [Matomnum+1]
          [List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***TRAN_DecMulP;

/*******************************************************
 double ******DM;
  current and old density matrices
  size: DM[List_YOUSO[16]]
          [SpinP_switch+1]
          [Matomnum+1]
          [FNAN[Gc_AN]+1]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******DM;

/*******************************************************
 double *****DM0;
  current density matrix
  size: DM0[SpinP_switch+1]
           [Matomnum+MatomnumF+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****DM0;

/*******************************************************
 double *****Partial_DM;
  partial density matrix to calculate partial density 
  in an energy window specified by 
  a keyword, scf.energy.window.partial.charge.
 
  size: Partial_DM
          [2]
          [Matomnum+1]
          [FNAN[Gc_AN]+1]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****Partial_DM;

/*******************************************************
 double ****DM_onsite;     --- added by MJ
   current 'onsite' density matrices
  ; same with ******DM, but [List_YOUSO[16] = 0],[FNAN[Gc_AN]+1 = 0]
   i.e. DM_onsite[][][][][] is defined as DM[0][][][0][][].
   To make DM_onsite physically meaninful(H_Hub must be Hermitian),
   it is set to be diagonal in the U_Mulliken_Charge.c.
   Therefore, last two dimensions are equivalent, i.e.,
   if [Spe_Total_NO[Cwan]] != [Spe_Total_NO[Hwan]] then
   DM_onsite = 0.
   So, it is possible to reduce the dimension of this array into 3, 
   but it remains for the future generalization.

  size: DM_onsite
          [2]
          [SpinP_switch+1]
          [Matomnum+1]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Hwan]] 
  allocation: allocate in  truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****DM_onsite;     /* --- added by MJ  */

/*******************************************************
 double ****v_eff;     --- added by MJ
  temporary effective Hubbard type potential which will be used
  to construct the effective Hubbard potential, V_eff.
  (cf. MJ's LDA+U note at 14, April 2004)

  size: v_eff
         [SpinP_switch+1]
         [Matomnum+MatomnumF+1]
         [Spe_Total_NO[Cwan]]
         [Spe_Total_NO[Cwan]] 
  allocation: allocate in  truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****v_eff;     /* --- added by MJ  */

/*******************************************************
 dcomplex ******NC_OcpN;     
   matrix consisting of occupation numbers which are used in 
   the non-collinear LDA+U method and a constraint DFT for 
   the spin orientation at each site. 

  size: NC_OcpN
          [2]
          [2]
          [2]
          [Matomnum+1]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Cwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
dcomplex ******NC_OcpN;  

/*******************************************************
 dcomplex *****NC_v_eff;     
   effectiv potential which are used in the non-collinear
   LDA+U method and a constraint DFT for the spin orientation
   at each site. 

  size: NC_v_eff
          [2]
          [2]
          [Matomnum+MatomnumF+1]
          [Spe_Total_NO[Cwan]]
          [Spe_Total_NO[Cwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
dcomplex *****NC_v_eff;

/*******************************************************
 double ******ResidualDM;
  current and old residual real density matrices, which are
  defined as the difference between input and output
  density matrices
  size: ResidualDM[List_YOUSO[16]]
                  [SpinP_switch+1]
                  [Matomnum+1]
                  [FNAN[Gc_AN]+1]
                  [Spe_Total_NO[Cwan]]
                  [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******ResidualDM;

/*******************************************************
 double ******iResidualDM;
  current and old residual imaginary density matrices,
  which are defined as the difference between input and
  output density matrices
  size: iResidualDM[List_YOUSO[16]]
                   [2]
                   [Matomnum+1]
                   [FNAN[Gc_AN]+1]
                   [Spe_Total_NO[Cwan]]
                   [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******iResidualDM;

/*******************************************************
 double *****EDM;
  current energy density matrices
  size: EDM[SpinP_switch+1]
           [Matomnum+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****EDM;

/*******************************************************
 double *****PDM;
  density matrix at one-step before
  size: PDM[SpinP_switch+1]
           [Matomnum+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****PDM;

/*******************************************************
 double *****iDM;
  imaginary density matrix
  size: iDM[List_YOUSO[16]]
           [2]
           [Matomnum+1]
           [FNAN[Gc_AN]+1]
           [Spe_Total_NO[Cwan]]
           [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ******iDM;

/*******************************************************
 double ***S12;
  S^{-1/2} of overlap matrix in divide-conquer (DC) method.

  size: S12[Matomnum    +1][n2][n2] for  DC method
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***S12;

/*******************************************************
 int **NumOLG;
  the number of overlapping grids between atom Mc_AN 
  and atom Lh_AN
  size: NumOLG[Matomnum+1]
              [FNAN[Gc_AN]+1]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **NumOLG;

/*******************************************************
 int *RNUM;
  the number of initial recusion levels of each atom 
  size: RNUM[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *RNUM;

/*******************************************************
 int *RNUM2;
  the number of current recusion levels of each atom 
  size: RNUM2[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *RNUM2;

/*******************************************************
 int ***RMI1;
  a table which converts local atomic index to global
  atomic index. 
  size: RMI1[Matomnum+1]
            [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
            [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
  allocation: in Trn_System() of truncation.c
  free:       truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int ***RMI1;

/*******************************************************
 int ***RMI2;
  a table which converts local atomic index to global
  atomic index. 
  size: RMI2[Matomnum+1]
            [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
            [FNAN[Gc_AN]+SNAN[Gc_AN]+1]
  allocation: in Trn_System() of truncation.c
  free:       truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int ***RMI2;

/*******************************************************
 double *NE_KGrids1, *NE_KGrids2, *NE_KGrids3;
  non equivalent k-points along reciprocal a-, b-, and c-axes.
  size: NE_KGrids1[num_non_eq_kpt]
        NE_KGrids2[num_non_eq_kpt]
        NE_KGrids3[num_non_eq_kpt]
  allocation: in Generating_MP_Special_Kpt
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *NE_KGrids1,*NE_KGrids2,*NE_KGrids3;

/*******************************************************
 int *NE_T_k_op;
  weight of the non equivalent k-points
  size: NE_T_k_op[num_non_eq_kpt]
  allocation: in Generating_MP_Special_Kpt
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *NE_T_k_op;

/*******************************************************
 dcomplex *****HOMOs_Coef;
  LCAO coefficients of HOMOs 
  size: HOMOs_Coef[List_YOUSO[33]]
                  [2] 
                  [List_YOUSO[31]]
                  [List_YOUSO[1]]
                  [List_YOUSO[7]]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
dcomplex *****HOMOs_Coef;

/*******************************************************
 dcomplex *****LUMOs_Coef;
  LCAO coefficients of HOMOs 
  size: HOMOs_Coef[List_YOUSO[33]]
                  [2] 
                  [List_YOUSO[32]]
                  [List_YOUSO[1]]
                  [List_YOUSO[7]]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
dcomplex *****LUMOs_Coef;

/*******************************************************
 int **Spe_Specified_Num;
  a table which converts index of contracted orbitals
  to that of primitive orbitals
  size: Spe_Specified_Num[List_YOUSO[18]]
                         [Spe_Total_NO[spe]]  
  allocation: in Set_BasisPara() of SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Spe_Specified_Num;

/*******************************************************
 int ***Spe_Trans_Orbital;
  a table which converts index of contracted orbitals
  to that of primitive orbitals
  size: Spe_Trans_Orbital[List_YOUSO[18]]
                         [Spe_Total_NO[spe]]  
                         [List_YOUSO[24]]
  allocation: in Set_BasisPara() of SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int ***Spe_Trans_Orbital;

/*******************************************************
 int *Spe_OpenCore_flag;
  flag to open core pseudopotential. In case of 1, partial 
  core charge is fully spin-polarized. 
  size: Spe_Spe2Ban[List_YOUSO[18]]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_OpenCore_flag;

/*******************************************************
 int *Spe_Spe2Ban;
  intermediate variable used in Correction_Energy.c
  size: Spe_Spe2Ban[List_YOUSO[18]]
  allocation: Allocation_Arrays(0) in Input_std()
              of SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Spe_Spe2Ban;

/*******************************************************
 int *Bulk_Num_HOMOs;
  the number of HOMOs of bulk for outputting to files
  size: Bulk_Num_HOMOs[List_YOUSO[33]]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Bulk_Num_HOMOs;

/*******************************************************
 int *Bulk_Num_LUMOs;
  the number of LUMOs of bulk for outputting to files
  size: Bulk_Num_HOMOs[List_YOUSO[33]]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Bulk_Num_LUMOs;

/*******************************************************
 int **Bulk_HOMO;
  HOMOs of up and down spins in the bulk systems
  size: Bulk_HOMO[List_YOUSO[33]][2]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Bulk_HOMO;

/*******************************************************
 double ***CntCoes;
  contraction coefficients of basis orbitals
  size: CntCoes[Matomnum+MatomnumF+1]
               [List_YOUSO[7]]
               [List_YOUSO[24]]
  allocation: in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***CntCoes;

/*******************************************************
 double ***CntCoes_Species;
  contraction coefficients of basis orbitals
  size: CntCoes_Species
               [SpeciesNum+1]
               [List_YOUSO[7]]
               [List_YOUSO[24]]
  allocation: in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***CntCoes_Species;

/*******************************************************
 int *CntOrb_Atoms;
  atoms for outputting contraction coefficients of
  basis orbitals to files
  size: CntOrb_Atoms[Num_CntOrb_Atoms]
  allocation: in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *CntOrb_Atoms;

/*******************************************************
 double **S;
  a full overlap matrix
  size: S[Size_Total_Matrix+2][Size_Total_Matrix+2]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double **S;

/*******************************************************
 double *EV_S;
  the eigenvalues of a full overlap matrix
  size: EV_S[Size_Total_Matrix+2]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *EV_S;

/*******************************************************
 double *IEV_S;
  the inverse of eigenvalues of a full overlap matrix
  size: IEV_S[Size_Total_Matrix+2]
  allocation: in SetPara_DFT.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *IEV_S;

/*******************************************************
 int *M2G
  M2G gives a conversion from the medium
  index to the global indicies of atoms.
  size: M2G[Matomnum+1]
  allocation: in Set_Allocate_Atom2CPU.c
  free:       call as Free_Arrays(0) in openmx.c
              and in Set_Allocate_Atom2CPU.c
*******************************************************/
int *M2G;

/*******************************************************
 int *F_M2G, *S_M2G; 
  F_M2G, and S_M2G give a conversion from the medium
  index (Matomnum+MatomnumF,
         Matomnum+MatomnumF+MatomnumS)
  to the global indicies of atoms.
  size: F_M2G[Matomnum+MatomnumF+1],
        S_M2G[Matomnum+MatomnumF+MatomnumS+1],
  allocation: in truncation.c
  free:       call as Free_Arrays(0) in openmx.c
              and in truncation.c
*******************************************************/
int *F_M2G,*S_M2G;

/*******************************************************
 int *VPS_j_dependency;
  VPS_j_dependency gives a flag whethere the VPS depends
  on total moment j.
  size: VPS_j_dependency[SpeciesNum],
  allocation: call as Allocate_Arrays(0) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *VPS_j_dependency;

/*******************************************************
 double ***Projector_VNA;
  Projectors for a projector expansion of VNA
  size: Projector_VNA[List_YOUSO[18]][YOUSO35+1][YOUSO34][List_YOUSO[22]]
  allocation: call Allocate_Arrays(7) in SetPara_DFT.c
  free:       call Free_Arrays(0) in openmx.c
*******************************************************/
double ****Projector_VNA;

/*******************************************************
 double **VNA_proj_ene;
  Projector energy for a projector expansion of VNA
  size: VNA_proj_ene[List_YOUSO[18]][YOUSO35+1][YOUSO34]
  allocation: call Allocate_Arrays(7) in SetPara_DFT.c
  free:       call Free_Arrays(0) in openmx.c
*******************************************************/
double ***VNA_proj_ene;

/*******************************************************
 double ***Spe_VNA_Bessel;
  radial parts of projectors of VNA on radial mesh
  in the momentum space
  size: Spe_VNA_Bessel[List_YOUSO[18]][YOUSO35+1][YOUSO34][List_YOUSO[15]]
  allocation: call Allocate_Arrays(7) in SetPara_DFT.c
  free:       call Free_Arrays(0) in openmx.c
*******************************************************/
double ****Spe_VNA_Bessel;

/*******************************************************
 double **Spe_CrudeVNA_Bessel;
  radial parts of crude VNA potentials on
  Gauss-Legendre radial mesh in the momentum space
  size: Spe_CrudeVNA_Bessel[List_YOUSO[18]][GL_Mesh+2]
  allocation: call Allocate_Arrays(7) in SetPara_DFT.c
  free:       call Free_Arrays(0) in openmx.c
*******************************************************/
double **Spe_CrudeVNA_Bessel;

/*******************************************************
 double *******Spe_ProductRF_Bessel;
  radial parts of product of two PAOs on
  Gauss-Legendre radial mesh in the momentum space
  size: Spe_ProductRF_Bessel[List_YOUSO[18]]
                            [Spe_MaxL_Basis[i]+1]
                            [Spe_Num_Basis[i][j]]
                            [Spe_MaxL_Basis[i]+1]
                            [Spe_Num_Basis[i][l]]
                            [Lmax+1]
                            [GL_Mesh+2]
  allocation: call Allocate_Arrays(7) in SetPara_DFT.c
  free:       call Free_Arrays(0) in openmx.c
*******************************************************/
double *******Spe_ProductRF_Bessel;

/*******************************************************
 double ****HVNA;
  real matrix elements of basis orbitals for VNA projectors
  size: HVNA[Matomnum+1]
            [FNAN[Gc_AN]+1]
            [Spe_Total_NO[Cwan]]
            [Spe_Total_NO[Hwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****HVNA;

/*******************************************************
 Type_DS_VNA *****DS_VNA;
  overlap matrix elements between projectors of VNA
  potentials, and basis orbitals 
  size: DS_VNA[4]
              [Matomnum+4]
              [FNAN[Gc_AN]+1]
              [Spe_Total_NO[Cwan]]
              [(YOUSO35+1)*(YOUSO35+1)*YOUSO34]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
Type_DS_VNA *****DS_VNA;

/*******************************************************
 Type_DS_VNA *****CntDS_VNA;
  overlap matrix elements between projectors of VNA
  potentials, and contracted basis orbitals 
  size: CntDS_VNA[4]
                 [Matomnum+MatomnumF+1]
                 [FNAN[Gc_AN]+1]
                 [Spe_Total_CNO[Cwan]]
                 [(YOUSO35+1)*(YOUSO35+1)*YOUSO34]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
Type_DS_VNA *****CntDS_VNA;

/*******************************************************
 double ****HVNA2;
  real matrix elements of basis orbitals for VNA projectors
  <Phi_{LM,L'M'}|VNA>
  size: HVNA2[4]
             [Matomnum+1]
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_NO[Cwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****HVNA2;

/*******************************************************
 double ****HVNA3;
  real matrix elements of basis orbitals for VNA projectors
  <VNA|Phi_{LM,L'M'}>
  size: HVNA3[4]
             [Matomnum+1]
             [FNAN[Gc_AN]+1]
             [Spe_Total_NO[Cwan]]
             [Spe_Total_NO[Cwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****HVNA3;

/*******************************************************
 double ****CntHVNA2;
  real matrix elements of basis orbitals for VNA projectors
  <Phi_{LM,L'M'}|VNA>
  size: CntHVNA2[4]
                [Matomnum+1]
                [FNAN[Gc_AN]+1]
                [Spe_Total_CNO[Cwan]]
                [Spe_Total_CNO[Cwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****CntHVNA2;

/*******************************************************
 double ****CntHVNA3;
  real matrix elements of basis orbitals for VNA projectors
  <VNA|Phi_{LM,L'M'}>
  size: CntHVNA3[4]
                [Matomnum+1]
                [FNAN[Gc_AN]+1]
                [Spe_Total_CNO[Cwan]]
                [Spe_Total_CNO[Cwan]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *****CntHVNA3;

/*******************************************************
 double ***Krylov_U (for BLAS3 version) 
  a Krylov matrix used in the embedding cluster method
  size: Krylov_U[SpinP_switch+1]
                [Matomnum+1]
                [List_YOUSO[3]*List_YOUSO[7]*(Max_FNAN+1)*List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***Krylov_U;

/*******************************************************
 double ***First_Moment_EC
 double ***Second_Moment_EC
  First moments of projected density of states used 
  in the embedding cluster method
  size: First_Moment_EC[SpinP_switch+1]
                       [atomnum+1]
                       [List_YOUSO[7]]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***First_Moment_EC;
double ***Second_Moment_EC;

/*******************************************************
 double ****EC_matrix
  a perturbation matrix used in the embedding cluster method
  size: EC_matrix[SpinP_switch+1]
                 [Matomnum+1]
                 [List_YOUSO[3]*List_YOUSO[7]] 
                 [List_YOUSO[3]*List_YOUSO[7]] 
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ****EC_matrix;

/*******************************************************
 int *rlmax_EC;
  recursion level to generate the preconditioning matrix
  for Hamiltonian in EC method
  size: rlmax_EC[Matomnum+1]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int *rlmax_EC;

/*******************************************************
 int *rlmax_EC2;
  recursion level to generate the preconditioning matrix
  for overlap matrix in EC method
  size: rlmax_EC2[Matomnum+1]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int *rlmax_EC2;

/*******************************************************
 int *EKC_core_size;
  core size in EKC method
  size: EKC_core_size[Matomnum+1]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int *EKC_core_size;

/*******************************************************
 double *scale_rc_EKC;
  scale factor to determine the core size in EKC method
  size: scale_rc_EKC[Matomnum+1]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double *scale_rc_EKC;

/*******************************************************
 double **InvHessian
  an approximate inverse of Hessian matrix
  size: InvHessian[3*atomnum+2][3*atomnum+2]
  allocation: allocate as Allocate_Arrays(1)
  free:       Free_Arrays(0) in openmx.c
*******************************************************/
double **InvHessian;

/*******************************************************
 double **Hessian
  an approximate Hessian matrix
  size: Hessian[3*atomnum+2][3*atomnum+2]
  allocation: allocate as Allocate_Arrays(1)
  free:       Free_Arrays(0) in openmx.c
*******************************************************/
double **Hessian;

/*******************************************************
 double ***DecEkin
  decomposed kinetic energy 
  size: DecEkin[2][Matomnum+1][List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***DecEkin;

/*******************************************************
 double ***DecEv
  decomposition of expectation value of Kohn-Sham effective potetial
  size: DecEv[2][Matomnum+1][List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***DecEv;

/*******************************************************
 double ***DecEcon
  decomposition of energy arising from the constant term 
  size: DecEcon[2][Matomnum+1][List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***DecEcon;

/*******************************************************
 double ***DecEscc
  decomposed energy of screened core-core repulsion energy
  size: DecEscc[2][Matomnum+1][List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***DecEscc;

/*******************************************************
 double ***DecEvdw
  decomposed van der Waals energy (D2 or D3) by Dion
  size: DecEvdw[2][Matomnum+1][List_YOUSO[7]]
  allocation: allocate in truncation.c
  free:       in truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
double ***DecEvdw;

/*******************************************************
 int *empty_occupation_spin;
  spin index of states for which the occupation is set to empty.
  size: empty_occupation_spin[empty_occupation_num];
  allocation: allocate in Input_std.c
  free:       Free_Arrays(0) in openmx.c
*******************************************************/
int *empty_occupation_spin;

/*******************************************************
 int *empty_occupation_orbital;
  orbital index of states for which the occupation is set to empty.
  size: empty_occupation_orbital[empty_occupation_num];
  allocation: allocate in Input_std.c
  free:       Free_Arrays(0) in openmx.c
*******************************************************/
int *empty_occupation_orbital;


dcomplex *zp,*Ep,*Rp;
double GL_Abscissae[GL_Mesh+2],GL_Weight[GL_Mesh+2];
double FineGL_Abscissae[FineGL_Mesh+2],FineGL_Weight[FineGL_Mesh+2];
double CoarseGL_Abscissae[CoarseGL_Mesh+2],CoarseGL_Weight[CoarseGL_Mesh+2];
double GL_NormK[GL_Mesh+2];
char Atom_Symbol[YOUSO14][4];
double Atom_Weight[YOUSO14];
double tv[4][4],rtv[4][4],tv_velocity[4][4];
double Left_tv[4][4],Right_tv[4][4];
double gtv[4][4],rgtv[4][4],length_gtv[4];
double gtv_FE[4][4],rgtv_FE[4][4];
double Stress_Tensor[9];
double MD_applied_pressure,UpV;
int MD_applied_pressure_flag[3];

double Grid_Origin[4];
double dipole_moment[4][4];
double TempPara[30][3],PrePara[30][3];
double MD_TimeStep,ChemP,Beta;
double CN_Error,E_Temp,Original_E_Temp,FCR,BCR,LNO_Occ_Cutoff,orderN_LNO_Buffer;
double GP,GT,T,Weight,Cell_Volume,Uele,Uele2,Ukc,Uvdw,Uch;
double Uele_OS0,Uele_OS1,Uele_IS0,Uele_IS1,Uxc0,Uxc1;
double UH0,UH1,UH2,Ucore,Uhub,Ucs,Uef,Ukin,Unl,Una,Uzs,Uzo,UvdW;
double Ucc,Ucoh,Uatom,Udc,Utot,Uxc,Given_Total_Charge,Calc_Total_Charge;
double Min_Mixing_weight,Max_Mixing_weight,Max_Mixing_weight2,Mixing_weight;
double Kerker_factor,Criterion_MP_Special_Kpt;
double Gdiis_Mixing,system_charge,VNA_reduce_ratio,Total_Num_Electrons;
double Total_SpinS,Total_SpinSx,Total_SpinSy,Total_SpinSz;
double Total_OrbitalMoment,Total_OrbitalMomentx;
double Total_OrbitalMomenty,Total_OrbitalMomentz;
double Total_SpinAngle0,Total_SpinAngle1;
double Total_OrbitalMomentAngle0,Total_OrbitalMomentAngle1;
double ScaleSize,Pin[4][4],Ptot[4][4];
double LastBoxCenterX,LastBoxCenterY,LastBoxCenterZ;
double Vol,Vol1,Vol2,Vol3,W;
double SCF_Criterion,NormRD[5],BestNormRD,History_Uele[5];
double PAO_Nkmax,Grid_Ecut,Finite_Elements_Ecut,rcut_FEB;
double orbitalOpt_criterion,MD_Opt_criterion,orbitalOpt_SD_step;
double MD_EvsLattice_Step;
double Restart_Spin_Angle_Theta,Restart_Spin_Angle_Phi;
int MD_EvsLattice_flag[3];
int MD_OutABC;
double X_Center_Coordinate,Y_Center_Coordinate,Z_Center_Coordinate;
dcomplex Comp2Real[YOUSO36+1][2*(YOUSO36+1)+1][2*(YOUSO36+1)+1];
/* added by mari (May 2004) */
double TempScale[30],RatScale[30],Temp; 
int IntScale[30],NumScale[30];
/* added by mari (May 2004) */
/* for Nose-Hoover algorithm */
double NH_R,NH_nzeta,NH_czeta,TempQ,GivenTemp,NH_Ham;
/* added by MIZUHO for NPT-MD */
double PresW;
double TempTol;
double NPT_WV_F0[4][4];
int LatticeRestriction;

/* for VS4 (added by T.Ohwaki) */
int num_AtGr,*AtomGr,*atnum_AtGr;
double *Temp_AtGr;

/* for Langevin heat-bath (added by T.Ohwaki) */
double FricFac,RandomF;


/* for generalized Bloch Theorem (added by T. B. Prayitno and supervised by Prof. F. Ishii) */
int GB_switch;
double q1_GB,q2_GB,q3_GB;

int NUMPROCS_MPI_COMM_WORLD,MYID_MPI_COMM_WORLD;
int alloc_first[40],Last_TNumGrid;
int Scf_RestartFromFile,Band_disp_switch,Use_of_Collinear_Restart;
int GeoOpt_RestartFromFile,OutData_bin_flag,LNO_flag,LNOs_Num_predefined_flag;
int coordinates_unit,unitvector_unit;
int Size_Total_Matrix,SP_PEV,EKC_core_size_max;
int specified_system,MO_fileout,num_HOMOs,num_LUMOs;
int Cluster_HOMO[2],MO_Nkpoint,ML_flag,ForceConsistency_flag,force_flag;
int StressConsistency_flag,stress_flag,scf_stress_flag,MD_cellopt_flag,cellopt_swtich;
int rediagonalize_flag_overlap_matrix; 
int rediagonalize_flag_overlap_matrix_ELPA1;
int CntOrb_fileout,Num_CntOrb_Atoms;
int num_non_eq_kpt,way_of_kpoint;
int pop_anal_aow_flag,scf_dclno_threading;
int remake_headfile,OneD_Grid,Ngrid1,Ngrid2,Ngrid3;
int Ngrid1_FE,Ngrid2_FE,Ngrid3_FE;
int TNumGrid,Kspace_grid1,Kspace_grid2,Kspace_grid3;
int DFTSCF_loop,Ngrid_NormK,SCF_RENZOKU;
int Mixing_switch,MD_IterNumber,MD_Current_Iter,Av_num,T_switch,IS_switch;
int MD_Init_Velocity,Correct_Position_flag;
int rlmax_IS,XC_switch,PCC_switch,SpinP_switch,SpinP_switch_RestartFiles,SpeciesNum,real_SpeciesNum;
int Hub_U_switch,Hub_U_occupation,Hub_U_Enhance_OrbPol;  /* --- added by MJ */
int SO_switch,MPI_tunedgrid_flag,Voronoi_Charge_flag,Voronoi_OrbM_flag;
int Constraint_NCS_switch,openmp_threads_eq_procs,openmp_threads_num;
int Zeeman_NCS_switch,Zeeman_NCO_switch;
int atomnum;
//int Catomnum,Latomnum,Ratomnum;
int POLES,rlmax,Solver,dste_flag,Ngrid_fixed_flag,scf_eigen_lib_flag;
int KrylovH_order,KrylovS_order,recalc_EM,EKC_invS_flag;
int EC_Sub_Dim,Energy_Decomposition_flag;
int EKC_Exact_invS_flag,EKC_expand_core_flag,orderN_FNAN_SNAN_flag;
int MD_switch,PeriodicGamma_flag;
int Max_FNAN,Max_FSNAN,Max_GridN_Atom,Max_NumOLG,Max_OneD_Grids;
int Max_Nd,Max_TGN_EH0,CellNN_flag;
int NN_B_AB2CA_S,NN_B_AB2CA_R,NN_B_CA2CB_S,NN_B_CA2CB_R;
int NN_A2B_S,NN_A2B_R,NN_B2C_S,NN_B2C_R,NN_B2D_S,NN_B2D_R; 
int List_YOUSO[NYOUSO];
int PreNum,TempNum,TCpyCell,CpyCell;
int Runtest_flag;
int Num_Mixing_pDM,level_stdout,level_fileout,HS_fileout;
int memoryusage_fileout;  
int Pulay_SCF,Pulay_SCF_original,EveryPulay_SCF,SCF_Control_Temp;
int Cnt_switch,RCnt_switch,SICnt_switch,ACnt_switch,SCnt_switch;
int E_Field_switch,Simple_InitCnt[10];
int MD_Opt_OK,orbitalOpt_SCF,orbitalOpt_MD,orbitalOpt_per_MDIter;
int orbitalOpt_History,orbitalOpt_StartPulay,OrbOpt_OptMethod;
int orbitalOpt_Force_Skip,Initial_Hessian_flag;
int NOHS_L,NOHS_C,ProExpn_VNA,BufferL_ProVNA;
int M_GDIIS_HISTORY,OptStartDIIS,OptEveryDIIS;
int Extrapolated_Charge_History;
int orderN_Kgrid,FT_files_save,FT_files_read;
int NEB_Num_Images,neb_type_switch;
double NEB_Spring_Const;
int Min_Grid_Index[4],Max_Grid_Index[4];
int Min_Grid_Index_D[4],Max_Grid_Index_D[4];
int SO_factor_flag;
int Cell_Fixed_XYZ[4][4];
int empty_occupation_flag,empty_occupation_num;
int empty_states_flag,empty_states_atom;
int empty_states_orbitals_sidx,empty_states_orbitals_num;

double **CompTime;

char filename[YOUSO10],filepath[YOUSO10],command[YOUSO10];
char ref_diff_charge_filename[YOUSO10];
char restart_filename[YOUSO10];
char DFT_DATA_PATH[YOUSO10];
double Oopt_NormD[10];
double bias_weight,Past_Utot[10],Past_Norm[10];
double Max_Force,GridVol,W_OrthoNorm;
double SD_scaling,SD_scaling_user;
double Constraint_NCS_V;
double Mag_Field_Orbital,Mag_Field_Spin;
double scf_fixed_origin[4];
int F_dVHart_flag,F_Vxc_flag,F_VNA_flag;
int F_VEF_flag,F_Kin_flag,F_NL_flag,F_CH_flag,F_U_flag;
int F_dftD_flag; /* okuno */

/* calculation with a core hole state */
int core_hole_state_flag,Core_Hole_Atom;
int Core_Hole_J,scf_coulomb_cutoff,scf_coulomb_cutoff_CoreHole;
char Core_Hole_Orbital[40];
double Shortest_CellVec;

/* core level excitations */
int CLE_Type;
double CLE_Val_Window,CLE_Con_Window;
 
/* partial charge for STM simulation */
int cal_partial_charge;
double ene_win_partial_charge;

/* band dispersion */
int Band_Nkpath,Band_kPathUnit;
double Band_UnitCell[4][4];
int *Band_N_perpath;
double ***Band_kpath;
char ***Band_kname; 

/*  DOS */
int DosGauss_fileout;
int Dos_fileout;
int DosGauss_Num_Mesh;
double DosGauss_Width;
double Dos_Erange[2];
int Dos_Kgrid[3];
int fermisurfer_output;

/*  electric field */ 
double E_Field[3];

/* O(N^2) method */
int ON2_Npoles,ON2_Npoles_f;
dcomplex *ON2_zp,*ON2_Rp,*ON2_zp_f,*ON2_Rp_f;
int *ON2_method,*ON2_method_f;

/* EGAC method */

int Matomnum_EGAC;
dcomplex *EGAC_zp,*EGAC_Rp,*EGAC_zp_f,*EGAC_Rp_f;
int *EGAC_method,*EGAC_method_f;
int EGAC_Num,EGAC_Npoles,EGAC_Npoles_f,EGAC_Npoles_CF;
int EGAC_Npoles_new,EGAC_Npoles_near;
int *EGAC_Top,*EGAC_End;
int *Num_Rcv_HS_EGAC,*Num_Snd_HS_EGAC;
int *Top_Index_HS_EGAC,*M2G_EGAC;
int **Indx_Rcv_HS_EGAC,**Indx_Snd_HS_EGAC;
int *Num_Rcv_GA_EGAC,*Num_Snd_GA_EGAC;
int **Indx_Rcv_GA_EGAC,**Indx_Snd_GA_EGAC;
int *Top_Index_GA_EGAC,*M2G_JOB_EGAC;
int **L2L_ONAN,*G2M_EGAC,***RMI1_EGAC,***RMI2_EGAC;
int *Snd_GA_EGAC_Size,*Rcv_GA_EGAC_Size;
int *Snd_OLP_EGAC_Size,*Rcv_OLP_EGAC_Size;
int **Indx_Rcv_DM_EGAC,**Indx_Snd_DM_EGAC;
int *Num_Rcv_DM_EGAC,*Num_Snd_DM_EGAC;
int *M2G_DM_Snd_EGAC,*G2M_DM_Snd_EGAC;
int *Snd_DM_EGAC_Size,*Rcv_DM_EGAC_Size;
int Matomnum_DM_Snd_EGAC,Num_GA_EGAC; 
int Max_Snd_OLP_EGAC_Size,Max_Rcv_OLP_EGAC_Size;
int Max_Snd_GA_EGAC_Size,Max_Rcv_GA_EGAC_Size;
int *dim_GD_EGAC,*dim_IA_EGAC,Max_dim_GD_EGAC;
int Max_dim_GA_EGAC,Max_ONAN;
int DIIS_History_EGAC,AC_flag_EGAC,scf_GF_EGAC;
double *****H_EGAC,****OLP_EGAC,*****DM_Snd_EGAC;
dcomplex ****GD_EGAC,****GA_EGAC,***Sigma_EGAC;
dcomplex **fGD_EGAC;
int MPI_spawn_flag;
FILE *MPI_spawn_stream;

/* Wannier funtions by hmweng */

int Wannier_Func_Calc;
int Wannier_Func_Num;
double Wannier_Outer_Window_Bottom;
double Wannier_Outer_Window_Top;
double Wannier_Inner_Window_Bottom;
double Wannier_Inner_Window_Top;
int Wannier_Initial_Guess;
int Wannier_unit;
int Wannier_Num_Kinds_Projectors;
int Num_Wannier_Template_Projectors;
int Wannier_grid1,Wannier_grid2,Wannier_grid3;
int Wannier_MaxShells;
int Wannier_Minimizing_Max_Steps;
char **Wannier_ProSpeName;
char **Wannier_ProName;
double **Wannier_Pos, **Wannier_Guide;
double **Wannier_X_Direction;
double **Wannier_Z_Direction;
int *Wannier_Num_Pro, *Wannier_ProName2Num;
int **Wannier_NumL_Pro;
int *WannierPro2SpeciesNum;
int Wannier_Draw_Int_Bands,Wannier_Draw_MLWF;
int Wannier_Plot_SuperCells[3];
double Wannier_Dis_Mixing_Para, Wannier_Dis_Conv_Criterion;
int Wannier_Dis_SCF_Max_Steps;
int Wannier_Min_Scheme, Wannier_Min_Secant_Steps;
double Wannier_Min_StepLength, Wannier_Min_Secant_StepLength;
double Wannier_Min_Conv_Criterion;
int Wannier_Output_Overlap_Matrix, Wannier_Output_Projection_Matrix;
int Wannier_Output_kmesh, Wannier_Readin_Overlap_Matrix,Wannier_Readin_Projection_Matrix;

double **Wannier_Euler_Rotation_Angle; /* for each ProSpe */
double ****Wannier_RotMat_for_Real_Func; /* Rotation Matrix of real Orbitals for each kind of projectors */

int **Wannier_Select_Matrix;
double ***Wannier_Projector_Hybridize_Matrix;
/* For interface with Wannier90 */
int Wannier90_fileout;
/*-------------------------------------------------------------Wannier*/

void Show_DFT_DATA(char *argv[]);
void Maketest(char *mode, int argc, char *argv[]);
void Runtest(char *mode, int argc, char *argv[]);
void Memory_Leak_test(int argc, char *argv[]);
void Get_VSZ(int MD_iter);
void Force_test(int argc, char *argv[]);
void Check_Force(char *argv[]);
void Stress_test(int argc, char *argv[]); 
void Check_Stress(char *argv[]);

double RF_BesselF(int Gensi, int GL, int Mul, double R);
double RF_BesselF2(int Gensi, int GL, int Mul, int LB, double R);
double Nonlocal_RadialF(int Gensi, int l, int so, double R);
double PhiF(double R, double *phi0, double *MRV, int Grid_Num);
double AngularF(int l, int m, double Q, double P, int Use_switch,
                double siQ, double coQ, double siP, double coP);
double RadialF(int Gensi, int L, int Mul, double R);
void Dr_RadialF(int Gensi, int L, int Mul, double R, double Deri_RF[3]);
double Smoothing_Func(double rcut,double r1);
double VNAF(int Gensi, double R);
double Dr_VNAF(int Gensi, double R);
double VH_AtomF(int spe, int N, double x, double r, double *xv, double *rv, double *yv);
double Dr_VH_AtomF(int spe, int N, double x, double r, double *xv, double *rv, double *yv);
double KumoF(int N, double x, double *xv, double *rv, double *yv);
double Dr_KumoF(int N, double x, double r, double *xv, double *rv, double *yv);

double AtomicCoreDenF(int Gensi, double R);
double Nonlocal_Basis(int wan, int Lnum_index, int Mnum, int so,
                      double r, double theta, double phi);

void Get_Orbitals(int wan, double x, double y, double z, double *Chi);
void Get_dOrbitals(int wan, double R, double Q, double P, double **dChi);
/* AITUNE */
struct WORK_DORBITAL {
	double** RF; double** dRF; double** AF; double** dAFQ; double** dAFP;
};
void Get_dOrbitals_init(struct WORK_DORBITAL* buffer);
void Get_dOrbitals_work(int wan, double R, double Q, double P, double **dChi, struct WORK_DORBITAL buffer);
void Get_dOrbitals_free(struct WORK_DORBITAL buffer);
/* end of AITUNE */
void Get_Cnt_Orbitals(int Mc_AN, double x, double y, double z, double *Chi);
void Get_Cnt_dOrbitals(int Mc_AN, double x, double y, double z, double **dChi);

/* Fukuda+YTL-start */
void Get_dOrbitals2(int wan, double R, double Q, double P, double **dChi);
void Get_Cnt_dOrbitals2(int Mc_AN, double x, double y, double z, double **dChi);
double Set_dOrbitals_Grid(int Cnt_kind);
double Set_dOrbitals_Grid_xyz(int Cnt_kind,int xyz);
/* Fukuda+YTL-end */

double Set_Orbitals_Grid(int Cnt_kind);
double Set_Aden_Grid();
double Set_Density_Grid(int Cnt_kind, int Calc_CntOrbital_ON, double *****CDM, double **Density_Grid_B0);
void diagonalize_nc_density(double **Density_Grid_B0);
void Data_Grid_Copy_B2C_1(double *data_B, double *data_C); 
void Data_Grid_Copy_B2C_2(double **data_B, double **data_C); 
void Density_Grid_Copy_B2D(double **Density_Grid_B0);
double Set_Initial_DM(double *****CDM, double *****H);
double Mulliken_Charge( char *mode );
double LNO(char *mode,
           int SCF_iter,
           double ****OLP0,
           double *****Hks,
           double *****CDM);

/* added by MJ */
void Occupation_Number_LDA_U(int SCF_iter, int SucceedReadingDMfile, double dUele, double ECE[], char *mode);
/* added by MJ */
void Eff_Hub_Pot(int SCF_iter, double ****OLP0);
void EulerAngle_Spin( int quickcalc_flag,
                      double Re11, double Re22,
                      double Re12, double Im12,
                      double Re21, double Im21,
                      double Nup[2], double Ndown[2],
                      double t[2], double p[2] );

void Orbital_Moment(char *mode);

/* added by S.Ryee */
void Coulomb_Interaction();
double slater_ratio;
int Hub_Type,Yukawa_on,dc_Type,Nmul;
/* array size of 30 is an arbitrarily large number */
double U[30],J[30],Slater_F0[30],Slater_F2[30],Slater_F4[30],Slater_F6[30];
int B_spe[30],B_l[30],B_mul[30];
double B_cut[30],lambda[30];
/*******************/

double Mixing_DM(int MD_iter,
                 int SCF_iter,
                 int SCF_iter0,
                 int SucceedReadingDMfile,
                 double ***ReRhok,
                 double ***ImRhok,
                 double **Residual_ReRhok,
                 double **Residual_ImRhok,
                 double *ReVk,
                 double *ImVk,
                 double *ReRhoAtomk,
                 double *ImRhoAtomk);

double Mixing_H( int MD_iter,
                 int SCF_iter,
                 int SCF_iter0 );

double Mixing_V( int MD_iter,
		 int SCF_iter,
		 int SCF_iter0 );

void Simple_Mixing_DM(int Change_switch, 
                      double Mix_wgt,
                      double *****CDM,
                      double *****PDM,
                      double *****P2DM,
                      double *****iCDM,
                      double *****iPDM,
                      double *****iP2DM,
                      double *****RDM,
                      double *****iRDM);

void DIIS_Mixing_DM(int SCF_iter, double ******ResidualDM, double ******iResidualDM);
void ADIIS_Mixing_DM(int SCF_iter, double ******ResidualDM, double ******iResidualDM);
void GR_Pulay_DM(int SCF_iter, double ******ResidualDM);


void Kerker_Mixing_Rhok(int Change_switch,
                        double Mix_wgt,
                        double ***ReRhok,
                        double ***ImRhok,
                        double **Residual_ReRhok,
                        double **Residual_ImRhok,
                        double *ReVk,
                        double *ImVk,
                        double *ReRhoAtomk,
                        double *ImRhoAtomk);

void DIIS_Mixing_Rhok(int SCF_iter,
                      double Mix_wgt,
                      double ***ReRhok,
                      double ***ImRhok,
                      double **Residual_ReRhok,
                      double **Residual_ImRhok,
                      double *ReVk,
                      double *ImVk,
                      double *ReRhoAtomk,
                      double *ImRhoAtomk);

 
void Overlap_Cluster(double ****OLP, double **S,int *MP);
void Overlap_Cluster_Ss(double ****OLP0, double *Ss, int *MP, int myworld1);

void Set_ContMat_Cluster_LNO(double ****OLP0, double *****nh, double ***S, double ***H, int *MP);

void Hamiltonian_Cluster(double ****RH, double **H, int *MP);
void Hamiltonian_Cluster_Hs(double ****RH, double *Hs, int *MP, int spin, int myworld1);
void Hamiltonian_Cluster_NC_Hs2( double *rHs11, double *rHs22, double *rHs12, 
                                 double *iHs11, double *iHs22, double *iHs12,
                                 dcomplex *Hs2 );
void Hamiltonian_Band_NC_Hs2( dcomplex *Hs11, dcomplex *Hs22, dcomplex *Hs12, 
			      dcomplex *Hs2,  MPI_Comm mpi_commWD);
void Overlap_Cluster_NC_Ss2( double *Ss, dcomplex *Ss2);
void Overlap_Band_NC_Ss2( dcomplex *Ss, dcomplex *Ss2, MPI_Comm mpi_commWD );
void Hamiltonian_Cluster_NC(double *****RH, double *****IH,
                            dcomplex **H, int *MP);
void Hamiltonian_Cluster_SO(double ****RH, double ****IH, dcomplex **H, int *MP);
void Hamiltonian_Band(int Host_ID1, double ****RH,
                      dcomplex **H, int *MP,
                      double k1, double k2, double k3);
void Hamiltonian_Band_NC(int Host_ID1, double *****RH, double *****IH,
                         dcomplex **H, int *MP,
                         double k1, double k2, double k3);
int Get_OneD_HS_Col(int set_flag, double ****RH, double *H1, int *MP, 
                    int *order_GA, int *My_NZeros, int *is1, int *is2);
void Overlap_Band(int Host_ID1, double ****OLP, dcomplex **S, int *MP,
                  double k1, double k2, double k3);

void Matrix_Band_LNO(int Host_ID1, int spin, double ****OLP, dcomplex **S, int *MP,
                     double k1, double k2, double k3);

void Initial_CntCoes(double *****nh, double *****OLP);
void Initial_CntCoes2(double *****nh, double *****OLP);
double Opt_Contraction(
         int orbitalOpt_iter,
         double TotalE,
         double *****H, 
         double *****OLP,
         double *****CDM,
         double *****EDM,
         double ****His_CntCoes,
         double ****His_D_CntCoes,
         double ****His_CntCoes_Species,
         double ****His_D_CntCoes_Species,
         double *His_OrbOpt_Etot,
         double **OrbOpt_Hessian);

void Contract_Hamiltonian(double *****H,   double *****CntH,
                          double *****OLP, double *****CntOLP);
void Contract_iHNL(double *****iHNL, double *****iCntHNL);
void Cont_Matrix0(double ****Mat, double ****CMat);
void Cont_Matrix1(double ****Mat, double ****CMat);
void Cont_Matrix2(Type_DS_VNA ****Mat, Type_DS_VNA ****CMat);
void Cont_Matrix3(double ****Mat, double ****CMat);
void Cont_Matrix4(double ****Mat, double ****CMat);

/* hmweng */
void Generate_Wannier();

void Population_Analysis_Wannier(char *argv[]);
void Population_Analysis_Wannier2(char *argv[]);

double EC(char *mode,
          int SCF_iter,
          double *****Hks,
          double *****ImNL,
	  double ****OLP0,
	  double *****CDM,
	  double *****EDM,
	  double Eele0[2], double Eele1[2]);

double Divide_Conquer(char *mode,
                      int SCF_iter,
                      double *****Hks,
                      double *****ImNL,
                      double ****OLP0,
                      double *****CDM,
                      double *****EDM,
                      double Eele0[2], double Eele1[2]);

double Divide_Conquer_LNO(char *mode,
                          int MD_iter,
                          int SCF_iter,
                          int SucceedReadingDMfile,
                          double *****Hks,
                          double *****ImNL,
                          double ****OLP0,
                          double *****CDM,
                          double *****EDM,
                          double Eele0[2], double Eele1[2]);

double Krylov(char *mode,
              int SCF_iter,
              double *****Hks,
              double *****ImNL,
              double ****OLP0,
              double *****CDM,
              double *****EDM,
              double Eele0[2], double Eele1[2]);
double Divide_Conquer_Dosout(double *****Hks,
                             double *****ImNL,
                             double ****OLP0);
double EGAC_DFT( char *mode,
                 int SCF_iter,
                 int SpinP_switch,
                 double *****Hks,
                 double *****ImNL,
                 double ****OLP0,
                 double *****CDM,
                 double *****EDM,
                 double Eele0[2], double Eele1[2] );
void Gauss_Legendre(int n, double x[], double w[], int *ncof, int *flag);
void zero_cfrac(int n, dcomplex *zp, dcomplex *Rp );

double dampingF(double rcut, double r);
double deri_dampingF(double rcut, double r);

void xyz2spherical(double x, double y, double z,
                   double xo, double yo, double zo,
                   double S_coordinate[3]);
int RestartFileDFT(char *mode, int MD_iter, double *Uele, double *****H, double *****CntH, double *etime);
void FT_PAO();
void FT_NLP();
void FT_ProExpn_VNA();
void FT_VNA();
void FT_ProductPAO();

double Poisson(int fft_charge_flag,
               double *ReDenk, double *ImDenk);

double FFT_Density(int den_flag,
                   double *ReDenk, double *ImDenk);

void Get_Value_inReal(int complex_flag,
                      double *ReVr, double *ImVr, 
                      double *ReVk, double *ImVk);

 /** Effective Screening Medium (ESM) Method Calculation (added by T.Ohwaki) **/

double Poisson_ESM(int fft_charge_flag,
		   double *ReRhok, double *ImRhok);

 /**  ESM end  **/

double Set_Hamiltonian(char *mode,
                       int MD_iter,
                       int SCF_iter,
                       int SCF_iter0,
                       int TRAN_Poisson_flag2,
                       int SucceedReadingDMfile,
                       int Cnt_kind,
                       double *****H0,
                       double *****HNL,
                       double *****CDM,
		       double *****H);


double Total_Energy(int MD_iter, double *****CDM, double ECE[]);
double Force(double *****H0,
	     double ******DS_NL, 
	     double *****OLP,
	     double *****CDM, 
	     double *****EDM); 
double Stress(double *****H0,
	      double ******DS_NL,
	      double *****OLP,
	      double *****CDM,
	      double *****EDM);
double Set_OLP_Kin(double *****OLP, double *****H0);
double Set_Nonlocal(double *****HNL, double ******DS_NL);
double Set_CoreHoleMatrix(double *****HCH);
double Set_OLP_p(double *****OLP_p);

double Set_ProExpn_VNA(double ****HVNA, double *****HVNA2, Type_DS_VNA *****DS_VNA);
void Set_Vpot(int MD_iter,
              int SCF_iter, 
              int SCF_iter0,
              int TRAN_Poisson_flag2,
              int XC_P_switch);
void Set_XC_Grid(int SCF_iter, int XC_P_switch, int XC_switch, 
                 double *Den0, double *Den1, 
                 double *Den2, double *Den3,
                 double *Vxc0, double *Vxc1,
                 double *Vxc2, double *Vxc3,
                 double ***dEXC_dGD, 
                 double ***dDen_Grid);
double Pot_NeutralAtom(int ct_AN, double Gx, double Gy, double Gz);
double XC_Ceperly_Alder(double den, int P_switch);
void XC_CA_LSDA(int SCF_iter, double den0, double den1, double XC[2],int P_switch);
void XC_PW92C(int SCF_iter, double dens[2], double Ec[1], double Vc[2]);
void XC_PBE(int SCF_iter, double dens[2], double GDENS[3][2], double Exc[2],
            double DEXDD[2], double DECDD[2],
            double DEXDGD[3][2], double DECDGD[3][2]);
void XC_EX(int NSP, double DS0, double DS[2], double EX[1], double VX[2]);
void Voronoi_Charge();
void Voronoi_Orbital_Moment();

double Fuzzy_Weight(int ct_AN, int Mc_AN, int Rn, double x, double y, double z);

void neb(int argc, char *argv[]);
void neb_run(char *argv[], MPI_Comm mpi_commWD, int index_images, double ***neb_atom_coordinates,
             int *WhatSpecies_NEB, int *Spe_WhatAtom_NEB, char **SpeName_NEB);
int neb_check(char *argv[]); 
 
/** DCLNO **/

int *NPROCS_ID1_DCLNO;
int *Comm_World1_DCLNO;
int *NPROCS_WD1_DCLNO;
int *Comm_World_StartID1_DCLNO;
MPI_Comm *MPI_CommWD1_DCLNO;
int myworld1_DCLNO,Num_Comm_World1_DCLNO;

int *NPROCS_ID2_DCLNO;
int *Comm_World2_DCLNO;
int *NPROCS_WD2_DCLNO;
int *Comm_World_StartID2_DCLNO;
MPI_Comm *MPI_CommWD2_DCLNO;
int myworld2_DCLNO,Num_Comm_World2_DCLNO;

// for ouput_atomic_orbitals
int my_CpyCell;

/** Natural Bond Orbital (NBO) Analysis (added by T.Ohwaki) **/
int NBO_switch;
int NAO_only;
int Num_NBO_FCenter;
int *NBO_FCenter;
int *Num_NHOs;
int Total_Num_NHO,Total_Num_NBO;
int NHO_fileout,NBO_fileout;
int NBO_SmallCell_Switch;
double NAO_Occ_or_Ryd;
double **NBO_CDMn_tmp, **NBO_OLPn_tmp, **NBO_Fock_tmp, ***NAO_vec_tmp;
double **NAO_partial_pop, **NAO_ene_level, ***NAO_coefficient;
double ****NHOs_Coef,****NBOs_Coef_b,****NBOs_Coef_a;
double NBO_SmallCellFrac[4][3];

int NAO_Nkpoint;
double **NAO_kpoint;

int *rlmax_EC_NAO, *rlmax_EC2_NAO, *EKC_core_size_NAO;
int *F_Snd_Num_NAO, *S_Snd_Num_NAO, *F_Rcv_Num_NAO, *S_Rcv_Num_NAO;
int **Snd_MAN_NAO, **Snd_GAN_NAO, **Rcv_GAN_NAO;
int *F_TopMAN_NAO, *S_TopMAN_NAO;
int *F_G2M_NAO, *S_G2M_NAO;
int *F_M2G_NAO, *S_M2G_NAO;
int MatomnumF_NAO, MatomnumS_NAO;
int *Snd_HFS_Size_NAO, *Rcv_HFS_Size_NAO;

void Calc_NAO_Cluster(double *****CDM);
void Calc_NAO_Band(
		   int nkpoint, double **kpoint,
		   int SpinP_switch,
		   double *****nh,
		   double ****OLP);

void Calc_NAO_Krylov(double *****Hks, double ****OLP0, double *****CDM);
/** NBO end **/


/*-----------------------------------------------------------------------*/

double readfile(char *argv[]);
void Input_std(char *filename);

double truncation(int MD_iter, int UCell_flag);
double DFT(int MD_iter, int Cnt_Now);

double Cluster_DFT_Col(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **ko,
                   double *****nh, 
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *Hs,
		   double *CDM1,
		   double *EDM1,
		   double *PDM1,
		   int size_H1,
                   int *SP_NZeros,
                   int *SP_Atoms,
                   double **EVec1,
                   double *Work1);


double Cluster_DFT_NonCol(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double *ko,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   double Eele0[2], double Eele1[2],
                   int *MP,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *rHs11,
		   double *rHs12,
		   double *rHs22,
		   double *iHs11,
		   double *iHs12,
		   double *iHs22,
                   dcomplex *Ss2,
                   dcomplex *Hs2,
                   dcomplex *Cs2,
		   double *DM1,
		   int size_H1, 
                   dcomplex *EVec1,
                   double *Work1);



double Calc_DM_Cluster_non_collinear_ScaLAPACK(
    int calc_flag,
    int myid,
    int numprocs,
    int size_H1,
    int *is2,
    int *ie2,
    int *MP,
    int n,
    int n2,
    double *****CDM,
    double *****iDM0,
    double *****EDM,
    double *ko,
    double *DM1,
    double *Work1,
    dcomplex *EVec1 );


double Cluster_DFT_LNO(char *mode,
		       int SCF_iter,
		       int SpinP_switch,
		       double ***Cluster_ReCoes,
		       double **Cluster_ko,
		       double *****nh,
		       double *****ImNL,
		       double ****OLP0,
		       double *****CDM,
		       double *****EDM,
		       EXX_t *exx, 
		       dcomplex ****exx_CDM,
		       double *Uexx,
		       double Eele0[2], double Eele1[2]);

double Cluster_DFT_Dosout( int SpinP_switch,
                           double *****nh,
                           double *****ImNL,
                           double ****CntOLP);

double Cluster_DFT_ON2(char *mode,
		       int SCF_iter,
		       int SpinP_switch,
		       double *****nh,
		       double *****ImNL,
		       double ****CntOLP,
		       double *****CDM,
		       double *****EDM,
		       double Eele0[2], double Eele1[2]);


double Band_DFT_Col(
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2], 
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double ***EIGEN,
		    double *H1,
		    double *S1,
		    double *CDM1,
		    double *EDM1,
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    int ***k_op,
		    int *T_k_op,
		    int **T_k_ID,
		    double *T_KGrids1,
		    double *T_KGrids2,
		    double *T_KGrids3,
                    int myworld1,
		    int *NPROCS_ID1,
		    int *Comm_World1,
		    int *NPROCS_WD1,
		    int *Comm_World_StartID1,
		    MPI_Comm *MPI_CommWD1,
                    int myworld2,
		    int *NPROCS_ID2,
		    int *NPROCS_WD2,
		    int *Comm_World2,
		    int *Comm_World_StartID2,
		    MPI_Comm *MPI_CommWD2);



double Band_DFT_NonCol(
                    int SCF_iter,
                    int knum_i, int knum_j, int knum_k,
		    int SpinP_switch,
		    double *****nh,
		    double *****ImNL,
		    double ****CntOLP,
		    double *****CDM,
		    double *****EDM,
		    double Eele0[2], double Eele1[2], 
		    int *MP,
		    int *order_GA,
		    double *ko,
		    double *koS,
		    double ***EIGEN,
		    double *H1,   
		    double *S1,
		    dcomplex *rHs11,   
		    dcomplex *rHs22,   
		    dcomplex *rHs12,   
		    dcomplex *iHs11,   
		    dcomplex *iHs22,   
		    dcomplex *iHs12, 
		    dcomplex **EVec1,
		    dcomplex *Ss,
		    dcomplex *Cs,
                    dcomplex *Hs,
		    dcomplex *Ss2,
		    dcomplex *Cs2,
                    dcomplex *Hs2,
		    int ***k_op,
		    int *T_k_op,
		    int **T_k_ID,
		    double *T_KGrids1,
		    double *T_KGrids2,
		    double *T_KGrids3,
                    int myworld1,
		    int *NPROCS_ID1,
		    int *Comm_World1,
		    int *NPROCS_WD1,
		    int *Comm_World_StartID1,
		    MPI_Comm *MPI_CommWD1,
                    int myworld2,
		    int *NPROCS_ID2,
		    int *NPROCS_WD2,
		    int *Comm_World2,
		    int *Comm_World_StartID2,
		    MPI_Comm *MPI_CommWD2);


/*
  For generalized Bloch Theorem 
  (added by T. B. Prayitno and supervised by Prof. F. Ishii)
*/
double Band_DFT_NonCol_GB(int SCF_iter,
			  double *koSU,
			  double *koSL,
			  dcomplex **SU,
			  dcomplex **SL,
			  int knum_i, int knum_j, int knum_k,
			  int SpinP_switch,
			  double *****nh,
			  double *****ImNL,
			  double ****CntOLP,
			  double *****CDM,
			  double *****EDM,
			  double Eele0[2], double Eele1[2]);


void k_inversion(int i,  int j,  int k, 
                 int mi, int mj, int mk, 
                 int *ii, int *ij, int *ik ); 
void Band_DFT_kpath( int nkpath, int *n_perk,
                     double ***kpath, char ***kname,
                     int SpinP_switch,
                     double *****nh,
                     double *****ImNL,
                     double ****CntOLP);

void Band_DFT_kpath_LNO( int nkpath, int *n_perk,
                         double ***kpath, char ***kname, 
                         int  SpinP_switch, 
                         double *****nh,
                         double *****ImNL,
                         double ****CntOLP);

void Band_DFT_MO( int nkpoint, double **kpoint,
                  int SpinP_switch, 
                  double *****nh,
                  double *****ImNL,
                  double ****CntOLP);
double Band_DFT_Dosout( int knum_i, int knum_j, int knum_k,
                        int SpinP_switch,
                        double *****nh,
                        double *****ImNL,
                        double ****CntOLP );

void Unfolding_Bands( int nkpoint, double **kpoint,
		      int SpinP_switch, 
		      double *****nh,
		      double *****ImNL,
		      double ****CntOLP);

double MD_pac(int iter, char *fname_input);
void Calc_Temp_Atoms(int iter);
int Species2int(char Species[YOUSO10]);
int R_atv(int CpyCell, int i, int j, int k);
int SEQ(char str1[YOUSO10], char str2[YOUSO10]);
void Generation_ATV(int CpyCell);
char *string_tolower(char *buf, char *buf1);
void iterout(int iter,double drctime,char fileE[YOUSO10],char fileDRC[YOUSO10]);
void iterout_md(int iter,double drctime,char fileSE[YOUSO10]);
void outputfile1(int f_switch, int MD_iter, int orbitalOpt_iter,
                 int Cnt_Now, int SCF_iter, char fname[YOUSO10],
                 double ChemP_e0[2]);
void init();
void Find_CGrids(int Real_Position, int n1, int n2, int n3,
                 double Cxyz[4], int NOC[4]);
void Diamond_structure(double aa);
void HCP_structure(double aa, double coa);
void FCC_structure(double aa);
void SetPara_DFT();
void Output_CompTime();
void Output_Energy_Decomposition();
void Make_FracCoord(char *file);
void Merge_LogFile(char *file);
void Make_InputFile_with_FinalCoord(char *file, int MD_iter);
void Eigen_lapack(double **a, double *ko, int n, int EVmax);
void Eigen_lapack2(double *a, int csize, double *ko, int n, int EVmax);
void Eigen_lapack3(double *a, double *ko, int n, int EVmax);
void EigenBand_lapack(dcomplex **A, double *W, int N0, int MaxN, int ev_flag);
void Eigen_PReHH(MPI_Comm MPI_Current_Comm_WD, 
                 double **ac, double *ko, int n, int EVmax, int bcast_flag);
void Eigen_PHH(MPI_Comm MPI_Current_Comm_WD, 
               dcomplex **ac, double *ko, int n, int EVmax, int bcast_flag);
void BroadCast_ReMatrix(MPI_Comm MPI_Curret_Comm_WD, 
                        double **Mat, int n, int *is1,int *ie1, int myid, int numprocs,
                        MPI_Status *stat_send,
                        MPI_Request *request_send,
                        MPI_Request *request_recv);
void BroadCast_ComplexMatrix(MPI_Comm MPI_Current_Comm_WD, 
                             dcomplex **Mat, int n, int *is1, int *ie1, int myid, int numprocs, 
                             MPI_Status *stat_send,
                             MPI_Request *request_send,
                             MPI_Request *request_recv);
void lapack_dstedc1(INTEGER N, double *D, double *E, double *W, double **ev);
void lapack_dstedc2(INTEGER N, double *D, double *E, double *W, dcomplex **ev);
void lapack_dstedc3(INTEGER N, double *D, double *E, double *W, double *ev, INTEGER csize);
void lapack_dstegr1(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, double **ev);
void lapack_dstegr2(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, dcomplex **ev);
void lapack_dstegr3(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, double *ev, INTEGER csize);

void lapack_dstevx1(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, double **ev);
void lapack_dstevx2(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, dcomplex **ev, int ev_flag);
void lapack_dstevx3(INTEGER N, INTEGER EVmax, double *D, double *E, double *W, double *ev, INTEGER csize);
void lapack_dstevx4(INTEGER N, INTEGER IL, INTEGER IU, double *D, double *E, double *W, double **ev);
void lapack_dstevx5(INTEGER N, INTEGER IL, INTEGER IU, double *D, double *E, double *W, dcomplex **ev, int ev_flag);
void lapack_dsteqr1(INTEGER N, double *D, double *E, double *W, double **ev);

void LU_inverse(int n, dcomplex **a);
void ReLU_inverse(int n, double **a, double **ia);
void spline3(double r, double r1, double rcut,
             double g, double dg, double value[2]);
void Cswap(dcomplex *a, dcomplex *b);
void fnjoint(char name1[YOUSO10],char name2[YOUSO10],char name3[YOUSO10]);
void fnjoint2(char name1[YOUSO10], char name2[YOUSO10],
              char name3[YOUSO10], char name4[YOUSO10]);
void chcp(char name1[YOUSO10],char name2[YOUSO10]);
void Init_List_YOUSO();
void Allocate_Arrays(int wherefrom);
void Free_Arrays(int dokokara);
double OutData(char *inputfile);
double OutData_Binary(char *inputfile);
void init_alloc_first();
int File_CntCoes(char *mode);
void SCF2File(char *mode, char *inputfile);
void Determine_Cell_from_ECutoff(double tv[4][4], double ECut);
#ifdef kcomp
void Spherical_Bessel( double x, int lmax, double *sb, double *dsb );
#else
inline void Spherical_Bessel( double x, int lmax, double *sb, double *dsb ) ;
#endif


void Generating_MP_Special_Kpt(/* input */
                               int atomnum,
			       int SpeciesNum,
			       double tv[4][4],
			       double **Gxyz,
                               double *InitN_USpin, 
                               double *InitN_DSpin,
                               double criterion_geo,
                               int SpinP_switch,
			       int *WhatSpecies,
			       int knum_i, int knum_j, int knum_k
                               /* implicit output 
                               num_non_eq_kpt,
                               NE_KGrids1, NE_KGrids2, NE_KGrids3,
                               NE_T_k_op */ );

void Make_Comm_Worlds(
   MPI_Comm MPI_Current_Comm_WD,   
   int myid0,
   int numprocs0,
   int Num_Comm_World, 
   int *myworld1, 
   MPI_Comm *MPI_CommWD,     /* size: Num_Comm_World */
   int *NPROCS1_ID,          /* size: numprocs0 */
   int *Comm_World1,         /* size: numprocs0 */
   int *NPROCS1_WD,          /* size: Num_Comm_World */
   int *Comm_World_StartID   /* size: Num_Comm_World */
		      );

void Make_Comm_Worlds2(
   MPI_Comm MPI_Curret_Comm_WD,   
   int myid0,
   int numprocs0,
   int Num_Comm_World, 
   int *myworld1, 
   MPI_Comm *MPI_CommWD,     /* size: Num_Comm_World */
   int *Comm_World1,         /* size: numprocs0 */
   int *NPROCS1_WD           /* size: Num_Comm_World */
		       );


 
/***********************  openmx_common.c  **************************/
  
void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);
void ComplexSH(int l, int m, double theta, double phi,
               double SH[2], double dSHt[2], double dSHp[2]);
void asbessel(int n, double x, double sbe[2]);
double Gaunt(int l,int m,int l1,int m1,int l2,int m2);
void Associated_Legendre(int l, int m, double x, double ALeg[2]);
void qsort_double(long n, double *a, double *b);
void qsort_double3(long n, double *a, int *b, int *c);
void qsort_double3B(long n, double *a, int *b, int *c);
void qsort_int(long n, int *a, int *b);
void qsort_int1(long n, int *a);
void qsort_int3(long n, int *a, int *b, int *c);
void qsort_double_int(long n, double *a, int *b);
void qsort_double_int2(long n, double *a, int *b);
void GN2N(int GN, int N3[4]);
void GN2N_EGAC(int GN, int N3[4]);
int AproxFactN(int N0);
void Get_Grid_XYZ(int GN, double xyz[4]);
double rnd(double width);
double rnd0to1(void);
double largest(double a, double b);
double smallest(double a, double b);
double Cabs(dcomplex z);
double sgn(double nu);
double isgn(int nu);
dcomplex Complex(double re, double im);
dcomplex Cadd(dcomplex a, dcomplex b);
dcomplex Csub(dcomplex a, dcomplex b);
dcomplex Cmul(dcomplex a, dcomplex b);
dcomplex Conjg(dcomplex z);
dcomplex Cdiv(dcomplex a, dcomplex b);
dcomplex Csqrt(dcomplex z);
dcomplex RCadd(double x, dcomplex a);
dcomplex RCsub(double x, dcomplex a);
dcomplex RCmul(double x, dcomplex a);
dcomplex CRmul(dcomplex a, double x);
dcomplex RCdiv(double x, dcomplex a);
dcomplex CRC(dcomplex a, double x, dcomplex b);
dcomplex Im_pow(int fu, int Ls);
dcomplex Csin(dcomplex a);
dcomplex Ccos(dcomplex a);
dcomplex Cexp(dcomplex a);

double FermiFunc(double x, int spin, int orb, int *index, double *popn);
double FermiFunc_NC(double x, int orb);

void PrintMemory_Fix();
void PrintMemory(char *name, long int size0, char *mode);
void dtime(double *);


 
/* okuno */
void DFTDvdW_SetNeighborShell(double rij[3],double** distR,
                              double* distR2,int*nrm);
void DFTDvdW_init();
void DFTD3vdW_init(); /* Ellner */


/****************************************************************/

void *malloc_multidimarray(char *type, int N, int *size);
void free_multidimarray(void **p,  int N, int *size);

/****************************************************************
           subroutines and common variables for MPI
****************************************************************/

/*******************************************************
 int *G2ID;
  G2ID gives a proccesor ID allocated to each atom
  with a global atom index.
  size: G2ID[atomnum+1];
  allocation: Allocation_Arrays(1) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *G2ID;

/*******************************************************
 int *Species_Top,*Species_End;
  arrays, Species_Top and Species_End, give global
  indices of the first and last species in species
  allocated to each processor.
  size: Species_Top[numprocs],Species_End[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Species_Top,*Species_End;

/*******************************************************
 int *F_Snd_Num;

  F_Snd_Num gives the number of atoms of which informations,
  related by FNAN, are transfered from myid to ID.
  size: F_Snd_Num[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *F_Snd_Num;

/*******************************************************
 int *F_Snd_Num_WK;

  F_Snd_Num_WK is a work array for F_Snd_Num.
  size: F_Snd_Num_WK[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *F_Snd_Num_WK;

/*******************************************************
 int *S_Snd_Num;

  S_Snd_Num gives the number of atoms of which informations,
  related by SNAN, are transfered from myid to ID.
  size: S_Snd_Num[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *S_Snd_Num;

/*******************************************************
 int *F_Rcv_Num;

  F_Rcv_Num gives the number of atoms of which informations,
  related by FNAN, are recieved at myid from ID.
  size: F_Rcv_Num[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *F_Rcv_Num;

/*******************************************************
 int *F_Rcv_Num_WK;

  F_Rcv_Num_WK is a work array for F_Rcv_Num.
  size: F_Rcv_Num_WK[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *F_Rcv_Num_WK;

/*******************************************************
 int *S_Rcv_Num;

  S_Rcv_Num gives the number of atoms of which informations,
  related by SNAN, are recieved at myid from ID.
  size: S_Rcv_Num[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *S_Rcv_Num;

/*******************************************************
 int *Snd_DS_NL_Size;

  Snd_DS_NL_Size gives the size of data for DS_NL
  which are transfered from myid to ID.
  size: Snd_DS_NL_Size[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Snd_DS_NL_Size;

/*******************************************************
 int *Rcv_DS_NL_Size;

  Rcv_DS_NL_Size gives the size of data for DS_NL
  which are received from ID.
  size: Rcv_DS_NL_Size[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Rcv_DS_NL_Size;

/*******************************************************
 int *Snd_HFS_Size;

  Snd_HFS_Size gives the size of data for Hks including
  the FNAN and SNAN atoms which are transfered from myid
  to ID.
  size: Snd_HFS_Size[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Snd_HFS_Size;

/*******************************************************
 int *Rcv_HFS_Size;

  Rcv_HFS_Size gives the size of data for Hks including
  the FNAN and SNAN atoms which are received from ID.
  size: Rcv_HFS_Size[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *Rcv_HFS_Size;

/*******************************************************
 int *F_TopMAN,*S_TopMAN;

  F_TopMAN and S_TopMAN give the first medium
  atom number in atoms sent from ID in the size of
  F_Rcv_Num[ID] and F_Rcv_Num[ID] + S_Rcv_Num[ID],
  respectively.
  size: F_TopMAN[numprocs],S_TopMAN[numprocs]
  allocation: Allocation_Arrays(0) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *F_TopMAN,*S_TopMAN;

/*******************************************************
 int *F_G2M,*S_G2M;

  F_G2M and S_G2M give a conversion from the
  global atom number to the medium atom number
  for atoms sent from ID in the size of
  F_Rcv_Num[ID] and F_Rcv_Num[ID] + S_Rcv_Num[ID],
  respectively. 
  size: F_G2M[atomnum+1],S_G2M[atomnum+1]
  allocation: Allocation_Arrays(1) in Input_std()
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *F_G2M,*S_G2M;

/*******************************************************
 int **Snd_MAN;
  Snd_MAN is a medium atom index of which informations
  are sent to a processor ID.
  size: Snd_MAN[numprocs][FS_Snd_Num[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Snd_MAN;

/*******************************************************
 int **Snd_GAN;
  Snd_GAN and Snd_GAN are a global atom index of which 
  informations are sent to a processor ID.
  size: Snd_GAN[numprocs][FS_Snd_Num[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Snd_GAN;

/*******************************************************
 int **Rcv_GAN;
  Rcv_GAN are a global atom index cell index of which 
  informations are recieved at myid from a processor ID.
  size: Rcv_GAN[numprocs][F_Rcv_Num[ID]+S_Rcv_Num[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Rcv_GAN;

/*******************************************************
 int **Pro_Snd_GAtom;

  Pro_Snd_GAtom gives the global atomic number used 
  for MPI communication of DS_VNA and DS_NL
  size: Pro_Snd_GAtom[numprocs][Num_Pro_Snd[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Pro_Snd_GAtom;

/*******************************************************
 int **Pro_Snd_MAtom;

  Pro_Snd_MAtom gives the intermedium atomic number used 
  for MPI communication of DS_VNA and DS_NL
  size: Pro_Snd_MAtom[numprocs][Num_Pro_Snd[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Pro_Snd_MAtom;

/*******************************************************
 int **Pro_Snd_LAtom;

  Pro_Snd_MAtom gives the local atomic number used 
  for MPI communication of DS_VNA and DS_NL
  size: Pro_Snd_LAtom[numprocs][Num_Pro_Snd[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Pro_Snd_LAtom;

/*******************************************************
 int **Pro_Snd_LAtom2;

  Pro_Snd_MAtom2 gives the local atomic number used 
  for MPI communication of DS_VNA and DS_NL, and 
  tells us the position of array which should be stored.
  size: Pro_Snd_LAtom2[numprocs][Num_Pro_Snd[ID]]
  allocation: Set_Inf_SndRcv() of truncation.c
  free:       Set_Inf_SndRcv() of truncation.c
              and call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Pro_Snd_LAtom2;

/*******************************************************
 int *Num_Snd_Grid_A2B

  Num_Snd_Grid_A2B gives the number of grids data of 
  rho_i sent to ID.
  size: Num_Snd_Grid_A2B[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Snd_Grid_A2B;

/*******************************************************
 int *Num_Rcv_Grid_A2B

  Num_Rcv_Grid_A2B gives the number of grids data of 
  rho_i received from ID.
  size: Num_Rcv_Grid_A2B[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Rcv_Grid_A2B;

/*******************************************************
 int **Index_Snd_Grid_A2B

  Index_Snd_Grid_A2B gives indices BN, atom, and Rn 
  in the partition B associated with the grids data of 
  rho_i sent to ID.
  size: Index_Snd_Grid_A2B[numprocs][3*Num_Snd_Grid_A2B[ID]]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Snd_Grid_A2B;

/*******************************************************
 int **Index_Rcv_Grid_A2B

  Index_Rcv_Grid_A2B gives indices BN, atom, and Rn 
  in the partition B associated with the grids 
  data of rho_i received from ID.
  size: Index_Rcv_Grid_A2B[numprocs][3*Num_Rcv_Grid_A2B[ID]]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Rcv_Grid_A2B;

/*******************************************************
 int *Num_Snd_Grid_B2C

  Num_Snd_Grid_B2C gives the number of grids data of 
  rho sent to ID.
  size: Num_Snd_Grid_B2C[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Snd_Grid_B2C;

/*******************************************************
 int *Num_Rcv_Grid_B2C

  Num_Rcv_Grid_B2C gives the number of grids data of 
  rho received from ID.
  size: Num_Rcv_Grid_B2C[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Rcv_Grid_B2C;

/*******************************************************
 int **Index_Snd_Grid_B2C

  Index_Snd_Grid_B2C gives index BN in the partition B
  associated with the grids data of rho sent to ID.
  size: Index_Snd_Grid_B2C[numprocs][# of grid to sent].
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Snd_Grid_B2C;

/*******************************************************
 int *Index_Rcv_Grid_B2C

  Index_Rcv_Grid_B2C gives index CN in the partition C
  associated with the grids data of rho received from ID.
  size: Index_Rcv_Grid_B2C[numprocs][# of grid to receive]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Rcv_Grid_B2C;

/*******************************************************
 int *Num_Snd_Grid_B2D

  Num_Snd_Grid_B2D gives the number of grids data of 
  rho sent to ID.
  size: Num_Snd_Grid_B2D[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Snd_Grid_B2D;

/*******************************************************
 int *Num_Rcv_Grid_B2D

  Num_Rcv_Grid_B2D gives the number of grids data of 
  rho received from ID.
  size: Num_Rcv_Grid_B2D[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Rcv_Grid_B2D;

/*******************************************************
 int **Index_Snd_Grid_B2D

  Index_Snd_Grid_B2D gives index BN in the partition B
  associated with the grids data of rho sent to ID.
  size: Index_Snd_Grid_B2D[numprocs][# of grid to sent].
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Snd_Grid_B2D;

/*******************************************************
 int *Index_Rcv_Grid_B2D

  Index_Rcv_Grid_B2D gives index DN in the partition D
  associated with the grids data of rho received from ID.
  size: Index_Rcv_Grid_B2D[numprocs][# of grid to receive]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Rcv_Grid_B2D;

/*******************************************************
 int *Num_Snd_Grid_B_AB2CA

  Num_Snd_Grid_B_AB2CA gives the number of grid data
  sent from the AB to CA partitions in the partion B.
  size: Num_Snd_Grid_B_AB2CA[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Snd_Grid_B_AB2CA;

/*******************************************************
 int *Num_Rcv_Grid_B_AB2CA

  Num_Rcv_Grid_B_AB2CA gives the number of grid data
  received in the CA partition and sent from the AB 
  partition in the partion B.
  size: Num_Rcv_Grid_B_AB2CA[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Rcv_Grid_B_AB2CA;
/* added by mari 05.12.2014 */
int *Num_Snd_Grid_B_AB2C;

/*******************************************************
 int *Num_Snd_Grid_B_CA2CB

  Num_Snd_Grid_B_CA2CB gives the number of grid data
  sent from the CA to CB partitions in the partion B.
  size: Num_Snd_Grid_B_CA2CB[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Snd_Grid_B_CA2CB;
/* added by mari 05.12.2014 */
int *Num_Rcv_Grid_B_AB2C;

/*******************************************************
 int *Num_Rcv_Grid_B_CA2CB

  Num_Rcv_Grid_B_CA2CB gives the number of grid data
  received in the CB partition and sent from the CA
  partition in the partion B.
  size: Num_Rcv_Grid_B_CA2CB[numprocs]
  allocation: call Allocate_Arrays() in Input_std.c
  free:       call Free_Arrays in openmx.c
*******************************************************/
int *Num_Rcv_Grid_B_CA2CB;

/*******************************************************
 int **Index_Snd_Grid_B_AB2CA

  Index_Snd_Grid_B_AB2CA gives index, BN_AB in the partition 
  B_AB associated with the grids data of sent to ID.
  size: Index_Snd_Grid_B_AB2CA[numprocs][Num_Snd_Grid_B_AB2CA[ID]]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Snd_Grid_B_AB2CA;
/* added by mari 05.12.2014 */
int **Index_Snd_Grid_B_AB2C;

/*******************************************************
 int **Index_Rcv_Grid_B_AB2CA

  Index_Rcv_Grid_B_AB2CA gives index, BN_AB in the partition 
  B_AB associated with the grids data of sent to ID.
  size: Index_Rcv_Grid_B_AB2CA[numprocs][Num_Rcv_Grid_B_AB2CA[ID]]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Rcv_Grid_B_AB2CA;
/* added by mari 05.12.2014 */
int **Index_Rcv_Grid_B_AB2C;

/*******************************************************
 int **Index_Snd_Grid_B_CA2CB

  Index_Snd_Grid_B_CA2CB gives index, BN_CA in the partition 
  B_CA associated with the grids data of sent to ID.
  size: Index_Snd_Grid_B_CA2CB[numprocs][Num_Snd_Grid_B_CA2CB[ID]]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Snd_Grid_B_CA2CB;

/*******************************************************
 int **Index_Rcv_Grid_B_CA2CB

  Index_Rcv_Grid_B_CA2CB gives index, BN_CA in the partition 
  B_CA associated with the grids data of sent to ID.
  size: Index_Rcv_Grid_B_CA2CB[numprocs][Num_Rcv_Grid_B_CA2CB[ID]]
  allocation: allocate_grids2atoms() in truncation.c
  free:       allocate_grids2atoms() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int **Index_Rcv_Grid_B_CA2CB;

/*******************************************************
  int *ID_NN_B_AB2CA_S;
  int *ID_NN_B_AB2CA_R;

  global process ID used for sending and receiving data
  in MPI commucation (AB to CA) of the structure B.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *ID_NN_B_AB2CA_S;
int *ID_NN_B_AB2CA_R;

/*******************************************************
  int *GP_B_AB2CA_S;
  int *GP_B_AB2CA_R;

  starting index to data used for sending and receiving data
  in MPI commucation (AB to CA) of the structure B.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *GP_B_AB2CA_S;
int *GP_B_AB2CA_R;

/*******************************************************
  int *ID_NN_B_CA2CB_S;
  int *ID_NN_B_CA2CB_R;

  global process ID used for sending and receiving data
  in MPI commucation (CA to CB) of the structure B.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *ID_NN_B_CA2CB_S;
int *ID_NN_B_CA2CB_R;

/*******************************************************
  int *GP_B_CA2CB_S;
  int *GP_B_CA2CB_R;

  starting index to data used for sending and receiving data
  in MPI commucation (CA to CB) of the structure B.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *GP_B_CA2CB_S;
int *GP_B_CA2CB_R;

/*******************************************************
  int *ID_NN_B2C_S;
  int *ID_NN_B2C_R;

  global process ID used for sending and receiving data
  in MPI commucation from the structure B to C.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *ID_NN_B2C_S;
int *ID_NN_B2C_R;

/*******************************************************
  int *GP_B2C_S;
  int *GP_B2C_R;

  starting index to data used for sending and receiving data
  in MPI commucation from the structure B to C.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *GP_B2C_S;
int *GP_B2C_R;

/*******************************************************
  int *ID_NN_B2D_S;
  int *ID_NN_B2D_R;

  global process ID used for sending and receiving data
  in MPI commucation from the structure B to D.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *ID_NN_B2D_S;
int *ID_NN_B2D_R;

/*******************************************************
  int *GP_B2D_S;
  int *GP_B2D_R;

  starting index to data used for sending and receiving data
  in MPI commucation from the structure B to D.

  allocation: Construct_MPI_Data_Structure_Grid() in truncation.c
  free:       Construct_MPI_Data_Structure_Grid() and
              call as Free_Arrays(0) in openmx.c
*******************************************************/
int *GP_B2D_S;
int *GP_B2D_R;

/*******************************************************
 double *time_per_atom; 
  elapsed time which is required for each atom
  size: time_per_atom[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double *time_per_atom;

/*******************************************************
 int *orderN_FNAN,*orderN_SNAN;
  user defined FNAN and SNAN for O(N) calculations
  size: orderN_FNAN[atomnum+1]
  size: orderN_SNAN[atomnum+1]
  allocation: call as Allocate_Arrays(1) in Input_std.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
int *orderN_FNAN,*orderN_SNAN;

int Matomnum,MatomnumF,MatomnumS,Max_Matomnum;
int MSpeciesNum,Num_Procs,Num_Procs2;
int Num_Cells0,My_NumGrid1,FNAN2_Grid;
int Max_Num_Rcv_FNAN2_Grid;
int Max_Num_Snd_FNAN2_Grid;
int My_NGrid1_Poisson,My_NGrid2_Poisson;
int My_NumGridB_AB,My_NumGridB_CB,My_NumGridB_CA;
int My_Max_NumGridB,My_NumGridC,My_NumGridD;
int Max_Num_Snd_Grid_B2C,Max_Num_Rcv_Grid_B2C;
int Max_Num_Snd_Grid_B2D,Max_Num_Rcv_Grid_B2D;
int Max_Num_Snd_Grid_B_AB2CA;
int Max_Num_Rcv_Grid_B_AB2CA;
/* added by mari 05.12.2014 */
int Max_Num_Snd_Grid_B_AB2C;
int Max_Num_Rcv_Grid_B_AB2C;
int Max_Num_Snd_Grid_B_CA2CB;
int Max_Num_Rcv_Grid_B_CA2CB;

/** Effective Screening Medium (ESM) Method Calculation (added by T.Ohwaki) **/

int ESM_switch,ESM_wall_switch;
int ESM_direction; /* surface normal axis; 1:x,2:y,3:z (added by AdvanceSoft) */
int iESM[4]; /* axis indices rearranged for ESM (added by AdvanceSoft) */
double V_ESM;
double ESM_wall_position,ESM_wall_height;
double ESM_buffer_range;

 /**  ESM end  **/

int Set_Allocate_Atom2CPU(int MD_iter, int isw, int weight_flag);

 /* added by T.Ohwaki */
int Arti_Force;
double Arti_Grad;
 /* added by T.Ohwaki */

/* vdW  DFT-D added by okuno*/
int dftD_switch;     /* okuno */
int unit_dftD;       /* unit Au or Ang */
int maxn_dftD;       /* the maximum number of vectors dist2, disR */
double rcut_dftD;    /* cut-off parameter for DFT-D */
double beta_dftD;    /* damping factor for DFT-D */
double scal6_dftD;   /* global scaling  factor  */
double *C6_dftD;     /* for each species */
double *RvdW_dftD;   /* for each species */
double **C6ij_dftD;  /* C6 coefficient of each atom type pair */
double **Rsum_dftD;  /* sum of VdW radii */
int n1_DFT_D,n2_DFT_D,n3_DFT_D;
int DFTD_IntDir1,DFTD_IntDir2,DFTD_IntDir3;

/* vdW DFT-D3 added by Ellner*/
int version_dftD;             /* 1-->DFT-D2 (Okuno), 2-->DFT-D3 with zero damping, 3--> DFT-D3 with BJ damping */
int DFTD3_damp_dftD;       /* For DFTD3: 1 --> ZERO 2--> BJ */
double k1_dftD, k2_dftD, k3_dftD;    /* used for calculating coordination number */
double s6_dftD, s8_dftD;   /* global scaling factors (s6=1.0)*/ 
double sr6_dftD, sr8_dftD;  /* parameters for zero damping function (sr8=1.0)*/
double alp6_dftD, alp8_dftD; /* exponent in zero damping function (alp6=14)*/
double **r0ab_dftD;        /* parameters used in calculating zero damping function*/
double a1_dftD, a2_dftD;   /* parameters for BJ damping function */
double *r2r4_dftD;         /* intermediate used in r2r4ab_dftd */
double **r2r4ab_dftD;      /* used in calculating C8 */
double cncut_dftD;         /* coordination number cut-off radius. Also needs global cut-off radius defined above (okuno) as rcut_dftD */
int n1_CN_DFT_D,n2_CN_DFT_D,n3_CN_DFT_D; /* for cncut PBC */
int *maxcn_dftD;           /* max number of C6 ref parameters per atom */
double *****C6ab_dftD;     /* C6 info: m=1,2,3:parameter/CN_atomA/CN_atomB [atomA][atomB][CN_ref_atomA][CN_ref_atomB][m] */
double *rcov_dftD;         /* intermediate used in rcovab_dftD */
double **rcovab_dftD;      /* covalent radius used in calculating coordination number */

/* unfolding added by Chi-Cheng Lee */
double **unfold_abc;
double *unfold_origin;
int *unfold_mapN2n;
double unfold_lbound,unfold_ubound;
int unfold_electronic_band;
int unfold_Nkpoint;
int unfold_nkpts;
double **unfold_kpoint;
char **unfold_kpoint_name;
/* end unfolding */


/* scalapack */

static int NBLK=128;
/* for n */
int nblk,np_rows,np_cols,na_rows,na_cols,na_rows_max,na_cols_max;
int my_prow,my_pcol;
int bhandle1,bhandle2,ictxt1,ictxt2;
int descS[9],descH[9],descC[9];

/* for 2*n */
int nblk2,np_rows2,np_cols2,na_rows2,na_cols2,na_rows_max2,na_cols_max2;
int my_prow2,my_pcol2;
int bhandle1_2,bhandle2_2,ictxt1_2,ictxt2_2;
int descS2[9],descH2[9],descC2[9];


/* YTL-start */
#define PIx2  6.2831853071795864769252
int Global_Cnt_kind; /* for calculating matrix element of Nabra operator */
int CDDF_on; /* turn on / off CDDF calculation */
int CDDF_freq_grid_number; /* frequnecy grid number for each conductivity tensor and dielectric function */
int CDDF_max_unoccupied_state; /* maximum unoccupied state for calculating conductivity and dielectric function */
int CDDF_material_type; /* default = 0. 0 = insulator, 1 = metal */
int CDDF_Kspace_grid1,CDDF_Kspace_grid2,CDDF_Kspace_grid3;
double CDDF_FWHM; /* FWHM for calculating conductivity and dielectric function */
double CDDF_AddMaxE; /* addition maximum energy for calculating conductivity and dielectric function */
double CDDF_max_eV,CDDF_min_eV;

double *****MME_allorb; /* < phi( atom a, orbital alpha) | nabla | phi( atom b, orbital beta) > */

void Calc_NabraMatrixElements(); /* < PAO(atom i, orbital alpha) | nabla | PAO(atom j, orbital beta) > */

void Set_MPIworld_for_optical(int myid,int numprocs);
void Initialize_optical();
/* at each thread */
void Calc_band_optical_col_1(double kx,double ky,double kz,int spin_index,int n,double* EIGEN, dcomplex** H, double* fd_dist,double ChemP);
void Calc_band_optical_noncol_1(double kx,double ky,double kz,int n,double* EIGEN, dcomplex** H, double* fd_dist,double ChemP);
/* collect data from different theads, sum k-point weight, and then calculating conductivity and dielectric function */
void Calc_optical_col_2(int n,double sum_weights);
void Calc_optical_noncol_2(int n,double sum_weights); 


double Band_DFT_Col_Optical_ScaLAPACK(
				      int SCF_iter,
				      int knum_i, int knum_j, int knum_k,
				      int SpinP_switch,
				      double *****nh,
				      double *****ImNL,
				      double ****CntOLP,
				      double *****CDM,
				      double *****EDM,
				      double Eele0[2], double Eele1[2], 
				      int *MP,
				      int *order_GA,
				      double *ko,
				      double *koS,
				      double *H1,
				      double *S1,
				      double *CDM1,
				      double *EDM1,
				      dcomplex **H);

double Band_DFT_NonCol_Optical(int SCF_iter,
			       double *koS,
			       dcomplex **S,
			       int knum_i, int knum_j, int knum_k,
			       int SpinP_switch,
			       double *****nh,
			       double *****ImNL,
			       double ****CntOLP,
			       double *****CDM,
			       double *****EDM,
			       double Eele0[2], double Eele1[2]);

double Cluster_DFT_Optical(char *mode,
			   int SCF_iter,
			   int SpinP_switch,
			   double ***Cluster_ReCoes,
			   double **Cluster_ko,
			   double *****nh,
			   double *****ImNL,
			   double ****CntOLP,
			   double *****CDM,
			   double *****EDM,
			   EXX_t *exx, 
			   dcomplex ****exx_CDM,
			   double *Uexx,
			   double Eele0[2], double Eele1[2]);

double Cluster_DFT_Optical_ScaLAPACK(
                   char *mode,
                   int SCF_iter,
                   int SpinP_switch,
                   double **Cluster_ko,
                   double *****nh,
                   double *****ImNL,
                   double ****CntOLP,
                   double *****CDM,
                   double *****EDM,
                   EXX_t *exx, 
                   dcomplex ****exx_CDM,
                   double *Uexx,
                   double Eele0[2], double Eele1[2],
		   int myworld1,
		   int *NPROCS_ID1,
		   int *Comm_World1,
		   int *NPROCS_WD1,
		   int *Comm_World_StartID1,
		   MPI_Comm *MPI_CommWD1,
		   int *is2,
		   int *ie2,
		   double *Ss,
		   double *Cs,
		   double *Hs, 
		   double *CDM1,
		   int size_H1, 
                   int *SP_NZeros,
                   int *SP_Atoms, 
                   double **EVec1,
                   double *Work1);



/* YTL-end */
