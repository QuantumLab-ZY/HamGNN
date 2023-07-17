
#ifndef ___dcomplex_definition___
typedef struct { double r,i; } dcomplex;
#define ___dcomplex_definition___ 
#endif


#ifndef YOUSO10
#define YOUSO10 500
#endif

#ifndef PI
#define PI    3.1415926535897932384626
#endif


int  TRAN_output_hks; 
int  TRAN_output_TranMain;
char TRAN_hksoutfilename[YOUSO10];
char TRAN_hksfilename[2][YOUSO10];


int *TRAN_region;
int *TRAN_Original_Id; 



/* [0] for L,  [1] for R  */

int Order_Lead_Side[2];
int TRAN_grid_bound[2];

int TRAN_grid_bound_e[2];       /* not used */
double TRAN_grid_bound_diff[2]; /* not used */
int TRAN_Kspace_grid2;
int TRAN_Kspace_grid3; 
int TRAN_TKspace_grid2;
int TRAN_TKspace_grid3;
int TRAN_integration;
int TRAN_Poisson_flag;
int TRAN_FFTE_CpyNum;
int TRAN_SCF_Iter_Band;
int TRAN_analysis,TRAN_SCF_skip;
double TRAN_Poisson_Gpara_Scaling;
 
double ScaleSize_e[2];
int SpinP_switch_e[2], atomnum_e[2], SpeciesNum_e[2], Max_FSNAN_e[2];
int TCpyCell_e[2], Matomnum_e[2], MatomnumF_e[2], MatomnumS_e[2];
int Latomnum,Ratomnum,Catomnum;
int *WhatSpecies_e[2];
int *Spe_Total_CNO_e[2];
int *Spe_Total_NO_e[2];
int *FNAN_e[2];
int **natn_e[2];
int **ncn_e[2];
int **atv_ijk_e[2];
double Grid_Origin_e[2][4];
double **Gxyz_e[2];

double *****OLP_e[2];
double *****H_e[2];
double ******DM_e[2]; 

double *dDen_Grid_e[2];
double *dVHart_Grid_e[2];

int IntNgrid1_e[2];
int Ngrid1_e[2], Ngrid2_e[2], Ngrid3_e[2];
int Num_Cells0_e[2];

double tv_e[2][4][4]; /* = Left_tv and Right_tv, merge them ! */
double gtv_e[2][4][4];

double ChemP_e[2];


double **ElectrodeDensity_Grid[2];
double *ElectrodeADensity_Grid[2];
double *ElectrodedVHart_Grid[2];
dcomplex ***VHart_Boundary[2];
dcomplex ***dDen_IntBoundary[2];
/* added by mari 09.12.2014 */
dcomplex *VHart_Boundary_G[2];

dcomplex **S00_e;
dcomplex **S01_e;
dcomplex ***H00_e;
dcomplex ***H01_e;

int NUM_e[2];


int **iLB_AN_e;  /* not used ? */



dcomplex *SCC;
dcomplex *SCL;
dcomplex *SCR;

dcomplex **HCC;
dcomplex **HCL;
dcomplex **HCR;

int NUM_c;



int tran_omega_n_scf; /* # of freq. to calculate density */
dcomplex *tran_omega_scf;
dcomplex *tran_omega_weight_scf;
int *tran_integ_method_scf; 

int tran_surfgreen_iteration_max;
double tran_surfgreen_eps; 

int tran_num_poles;


/* square path */
double tran_square_path_ene[4];
int tran_square_path_ene_fix[4]; 
int tran_square_path_div[4];
double tran_square_path_bias_expandenergy; /* for NEGF */
int tran_square_path_bias_div;    /* for NEGF */

double tran_thermalarc_path_ene[5];
int tran_thermalarc_path_ene_fix[5];
int tran_thermalarc_path_div[3];
double tran_thermalarc_path_bias_expandenergy; /* for NEGF */
int tran_thermalarc_path_bias_div;    /* for NEGF */



int tran_line_path_div;
char **tran_line_path_string; 

int    tran_transmission_on;
double tran_transmission_energyrange[3];
int    tran_transmission_energydiv;

int    tran_transmission_iv_on;
double tran_transmission_iv_energyrange[3];
int tran_transmission_iv_energydiv; 

double tran_dos_energyrange[3];
int    tran_dos_energydiv;
int    TRAN_dos_Kspace_grid2,TRAN_dos_Kspace_grid3;




dcomplex **tran_transmission;
dcomplex **tran_transmission_iv; 


int tran_bias_apply;      /* =1:NEGF  =0:no bias voltage */
double tran_biasvoltage_e[2];   /* bias voltage 0:left and 1:right */
/* modified by mari 12.22.2014 */
double tran_gate_voltage[2];
double Tran_bias_neq_im_energy;
double Tran_bias_neq_energy_step;
double Tran_bias_neq_cutoff;
double Tran_bias_neq_lower_bound;
double Tran_bias_neq_upper_bound;
int Tran_bias_neq_num_energy_step;


/* for the contour integrals */
double *TRAN_GL_Abs,*TRAN_GL_Wgt;

/* revised by Y. Xiao for Noncollinear NEGF calculations */
double *****iHNL_e[2];

dcomplex **S00_nc_e;
dcomplex **S01_nc_e;
dcomplex **H00_nc_e;
dcomplex **H01_nc_e;

dcomplex *SCC_nc;
dcomplex *SCL_nc;
dcomplex *SCR_nc;
dcomplex *HCC_nc;
dcomplex *HCL_nc;
dcomplex *HCR_nc;
/* until here by Y. Xiao for Noncollinear NEGF calculations */

