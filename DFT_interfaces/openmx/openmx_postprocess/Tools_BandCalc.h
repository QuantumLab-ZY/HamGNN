

#define PI               3.1415926535897932384626   
#define kB               0.00008617251324000000     /* eV/K */
#define BohrR            0.529177249                /* BohrR   >> Angstrom */
#define eV2Hartree      27.2113845                  /* Hartree >> eV       */

#define asize10         10

#define Shift_K_Point    1.0e-6
#define measure_time     0


#ifdef nompi

#ifndef ___MPI_Comm_definition___
typedef int MPI_Comm;
#define ___MPI_Comm_definition___
#endif

#else
#include "mpi.h"
#endif

//MPI_Comm  mpi_comm_level1;
//MPI_Comm  MPI_COMM_WORLD1;


/**************************************************
  INPUT and OUTPUT FILENAME
  fname[] is input parameta file name.
  fname_wf[] is .scfout file name.
  fname_output[] is output data file name. 
  Calc_Type is a variable which decide Calclation type.
 **************************************************/


char fname[256], fname_wf[256], fname_out[256];
int Calc_Type;

/* Added by N. Yamaguchi ***/
#ifndef DEBUG_SIGMAEK_OLDER_20181126
typedef int (*kSpin)();
kSpin mode;
#ifdef SIGMAEK
int AtomInfo();
int CircularSearch();
int BandDispersion();
int GridCalc();
int FermiLoop();
int MulPOnly();
#else
void BandDispersion();
void GridCalc();
void FermiLoop();
void MulPOnly();
#endif
#endif
/* ***/

// ### Orbital Data    ###
int TNO_MAX ,ClaOrb_MAX[2] ,**ClaOrb;              // Orbital
char OrbSym[5][2], **OrbName, **An2Spe;


/*ISO_SURFACE_PROGRAM******************************
  (int)   
  Nband is a variable of calculate band number. 
  (double)  
  E_Range[2] are variables of energy range for iso-surface calculation.  (eV)
  Data_MulP is data of MulP (each TNO). 
  k_CNT[3] are variables of Central k_grid. 
  k_CNT1[3] are variables of Central k-point. (k-space(A-1))
 **************************************************/


double k_CNT[3], k_CNT1[3];
double E_Range[2]; //  EF;
double *EIGEN;
double MulP_VecScale[3];
double MulP_up, MulP_dw, MulP_theta, MulP_phi;
double ****Data_MulP;  

int Nband[2];
int Spin_Dege;


/* 3mesh ******************************************
 **************************************************/

int switch_Eigen_Newton;
int plane_3mesh;
int k1_3mesh, k2_3mesh;
int k1_domain, k2_domain;
double kRange_3mesh[2];
double kRange_domain[2]; 


/*Band_kpath***************************************
 **************************************************/

int l_max, l_min, l_cal;                        // band index
int Band_Nkpath,*Band_N_perpath;
double ***Band_kpath;
char ***Band_kname;


/**************************************************
  subroutine 
 **************************************************/
double kgrid_dist(double x1, double x2, double y1, double y2, double z1, double z2);
void Print_kxyzEig(char *Pdata_s, double kx, double ky, double kz, int l, double EIG);
void dtime(double *t);
void name_Nband(char *fname1, char *fname2, int l);
void k_inversion(int i,  int j,  int k, 
    int mi, int mj, int mk, 
    int *ii, int *ij, int *ik );
double sgn(double nu);
void xyz2spherical(double x, double y, double z,
    double xo, double yo, double zo,
    double S_coordinate[3]);
void EulerAngle_Spin( int quickcalc_flag,
    double Re11, double Re22,
    double Re12, double Im12, double Re21, double Im21,
    double Nup[2], double Ndown[2], double t[2], double p[2] );



