void free_scfout();

/*******************************************************
  int atomnun;
  the number of total atoms
 *******************************************************/
int atomnum;   

/*******************************************************
  int Catomnun;
  the number of atoms in the central region
 *******************************************************/
int Catomnum;

/*******************************************************
  int Latomnun;
  the number of atoms in the left lead
 *******************************************************/
int Latomnum;

/*******************************************************
  int Ratomnun;
  the number of atoms in the left lead
 *******************************************************/
int Ratomnum;   

/*******************************************************
  int SpinP_switch;
0: non-spin polarized 
1: spin polarized
3: non-collinear calculation
 *******************************************************/
int SpinP_switch;

/*******************************************************
  int TCpyCell;
  the total number of periodic cells
 *******************************************************/
int TCpyCell;

/*******************************************************
  int Solver;
  method for solving eigenvalue problem
 *******************************************************/
int Solver;

/*******************************************************
  double ChemP;
  chemical potential
 *******************************************************/
double ChemP;

/*******************************************************
  int Valence_Electrons;
  total number of valence electrons
 *******************************************************/
int Valence_Electrons;

/*******************************************************
  double Total_SpinS;
  total value of Spin (2*Total_SpinS = muB)
 *******************************************************/
double Total_SpinS;

/*******************************************************
  double E_Temp;
  electronic temperature
 *******************************************************/
double E_Temp;

/*******************************************************
  int *Total_NumOrbs; 
  the number of atomic orbitals in each atom
size: Total_NumOrbs[atomnum+1]
 *******************************************************/
int *Total_NumOrbs;

/*******************************************************
  int *FNAN; 
  the number of first neighboring atoms of each atom
size: FNAN[atomnum+1]
 *******************************************************/
int *FNAN;

/*******************************************************
  int **natn; 
  grobal index of neighboring atoms of an atom ct_AN
size: natn[atomnum+1][FNAN[ct_AN]+1]
 *******************************************************/
int **natn;

/*******************************************************
  int **ncn; 
  grobal index for cell of neighboring atoms of
  an atom ct_AN
size: ncn[atomnum+1][FNAN[ct_AN]+1]
 *******************************************************/
int **ncn;

/*******************************************************
  double **atv;
  x,y,and z-components of translation vector of  
  periodically copied cells
size: atv[TCpyCell+1][4];
 *******************************************************/
double **atv;

/*******************************************************
  int **atv_ijk;
  i,j,and k number of periodically copied cells
size: atv_ijk[TCpyCell+1][4];
 *******************************************************/
int **atv_ijk;

/*******************************************************
  double tv[4][4];
  unit cell vectors in Bohr
 *******************************************************/
double tv[4][4];

/*******************************************************
  double rtv[4][4]:
  reciprocal unit cell vectors in Bohr^{-1}

note:
tv_i \dot rtv_j = 2PI * Kronecker's delta_{ij}
 *******************************************************/
double rtv[4][4];

/*******************************************************
  double Gxyz[atomnum+1][60];
  atomic coordinates in Bohr
 *******************************************************/
double **Gxyz;

/*******************************************************
  double *****Hks;
  Kohn-Sham matrix elements of basis orbitals
size: Hks[SpinP_switch+1]
[atomnum+1]
[FNAN[ct_AN]+1]
[Total_NumOrbs[ct_AN]]
[Total_NumOrbs[h_AN]] 
 *******************************************************/
double *****Hks;

/*******************************************************
  double *****iHks;
  imaginary Kohn-Sham matrix elements of basis orbitals
  for alpha-alpha, beta-beta, and alpha-beta spin matrices
  of which contributions come from spin-orbit coupling 
  and Hubbard U effective potential.
size: iHks[3]
[atomnum+1]
[FNAN[ct_AN]+1]
[Total_NumOrbs[ct_AN]]
[Total_NumOrbs[h_AN]] 
 *******************************************************/
double *****iHks;

/*******************************************************
  double ****OLP;
  overlap matrix
size: OLP[atomnum+1]
[FNAN[ct_AN]+1]
[Total_NumOrbs[ct_AN]]
[Total_NumOrbs[h_AN]]
 *******************************************************/
double ****OLP;

/*******************************************************
  double *****DM;
  density matrix
size: DM[SpinP_switch+1]
[atomnum+1]
[FNAN[ct_AN]+1]
[Total_NumOrbs[ct_AN]]
[Total_NumOrbs[h_AN]]
 *******************************************************/
double *****DM;

/*******************************************************
  double *****iDM;
  density matrix
size: iDM[2]
[atomnum+1]
[FNAN[ct_AN]+1]
[Total_NumOrbs[ct_AN]]
[Total_NumOrbs[h_AN]]
 *******************************************************/
double *****iDM;

/*******************************************************
  double dipole_moment_core[4];
 *******************************************************/
double dipole_moment_core[4];

/*******************************************************
  int version; (added by N. Yamaguchi)
 *******************************************************/
int version;

/*******************************************************
  int order_max; (added by N. Yamaguchi for HWC)
 *******************************************************/
int order_max;

/*******************************************************
  double *cc_vec; (added by N. Yamaguchi for HWC)
 *******************************************************/
double *cc_vec;

/*******************************************************
  double ******OLPpo; (added by N. Yamaguchi for HWC)
 *******************************************************/
double ******OLPpo;

/*******************************************************
  double *****OLPmo; 
 *******************************************************/
double *****OLPmo;

/*******************************************************
  double dipole_moment_background[4];
 *******************************************************/
double dipole_moment_background[4];
