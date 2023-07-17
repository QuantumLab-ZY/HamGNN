#include "f77func.h"
#ifdef blaswrap
#define zgemm_ f2c_zgemm
#define zcopy_ f2c_zcopy
#endif
      
#ifndef ___dcomplex_definition___
typedef struct { double r,i; } dcomplex;
#define ___dcomplex_definition___ 
#endif

#define z_mul_inline(x,y,z) { z.r= x.r*y.r-x.i*y.i;  z.i = x.r * y.i + x.i*y.r; }
#define z_exp_inline(x,z) {double _t; _t=exp(x.r);  z.r = _t*cos(x.i); z.i = _t*sin(x.i); }

#ifndef __sqr_definition___
#define sqr(x)   ( (x)*(x) )
#define __sqr_definition___
#endif


#ifndef Host_ID
#define Host_ID 0
#endif


#ifndef YOUSO10
#define YOUSO10 500
#endif

#ifndef Shift_K_Point
#define Shift_K_Point    1.0e-6      /* disturbance for stabilization of eigenvalue routine */
#endif


#ifndef ___Type_Orbs_Grid_definition___
typedef float Type_Orbs_Grid; /* type of Orbs_Grid */
#define ___Type_Orbs_Grid_definition___
#endif

int Lapack_LU_Zinverse(int , dcomplex *);



/*** TRAN_PROTOTYPES ***/

/* TRAN_Allocate.c  */
void TRAN_Allocate_Atoms(
    int atomnum    
);

/* TRAN_Allocate.c  */
void TRAN_Deallocate_Atoms( void );

/* TRAN_Allocate.c  */
void TRAN_Allocate_Cregion( 
     MPI_Comm mpi_comm_level1,  
     int  SpinP_switch, 
     int atomnum,
     int *WhatSpecies,
     int *Spe_Total_CNO 
     ) ;

/* TRAN_Allocate.c  */
void TRAN_Deallocate_Cregion(int SpinP_switch);

/* in TRAN_Allocate.c  */
void TRAN_Allocate_Lead_Region( MPI_Comm mpi_comm_level1 ); 

/* in TRAN_Allocate.c  */
void TRAN_Deallocate_Lead_Region();

/* TRAN_Deallocate_Electrode_Grid.c */
void TRAN_Deallocate_Electrode_Grid(int Ngrid2);

/* TRAN_Deallocate_RestartFile.c */
void TRAN_Deallocate_RestartFile(char *position);


/* TRAN_Apply_Bias2e.c  */
void  TRAN_Apply_Bias2e(
       MPI_Comm comm1,
       int side,
       double voltage,  
       double TRAN_eV2Hartree,
       int SpinP_switch,
       int atomnum,
       int *WhatSpecies,
       int *Spe_Total_CNO,
       int *FNAN,
       int **natn,
       int Ngrid1,
       int Ngrid2,
       int Ngrid3,
        double ****OLP,
        double *ChemP, 
        double *****H, 
        double *dVHart_Grid
);

/* TRAN_Calc_CentGreen.c  */
void TRAN_Calc_CentGreen(
                      dcomplex w,
                      int nc, 
                      dcomplex *sigmaL,
                      dcomplex *sigmaR, 
                      dcomplex *HCC,
                      dcomplex *SCC,
                      dcomplex *GC  
                      );

/* TRAN_Calc_CentGreenLesser.c  */
void TRAN_Calc_CentGreenLesser(
                      /* input */
                      dcomplex w,
                      double ChemP_e[2],
                      int nc, 
                      int Order_Lead_Side[2],
                      dcomplex *SigmaL,
                      dcomplex *SigmaR, 
                      dcomplex *GC, 
                      dcomplex *HCCk, 
                      dcomplex *SCC, 

                      /* work, nc*nc */
                      dcomplex *v1, 
                      dcomplex *v2,
 
                      /*  output */ 
                      dcomplex *Gless 
                      );
  
 
/* TRAN_Calc_GridBound.c  */
void TRAN_Calc_GridBound(MPI_Comm mpi_comm_level1,
			 int atomnum,
			 int *WhatSpecies,
			 double *Spe_Atom_Cut1,
			 int Ngrid1, 
			 double *Grid_Origin, 
			 double **Gxyz,
			 double tv[4][4],
			 double gtv[4][4],
			 double rgtv[4][4],
			 double Left_tv[4][4],
			 double Right_tv[4][4]);

/* TRAN_Calc_OneTransmission.c  */
void TRAN_Calc_OneTransmission(
   int nc, 
   dcomplex *SigmaL_R,   /* at w, changed when exit */
   dcomplex *SigmaL_A,   /* at w, changed when exit */
   dcomplex *SigmaR_R,   /* at w, changed when exit */
   dcomplex *SigmaR_A,   /* at w, changed when exit */
   dcomplex *GC_R,       /* at w, changed when exit */
   dcomplex *GC_A,       /* at w, changed when exit */
   dcomplex *v1,         /* work */
   dcomplex *v2,         /* work */
   dcomplex *value       /* output, transmission */
   );


/* TRAN_Calc_OneTransmission2.c  */
void TRAN_Calc_OneTransmission2(
				dcomplex w,
				int nc, 
				dcomplex *SigmaL,   /* at w, changed when exit */
				dcomplex *SigmaR,   /* at w, changed when exit */
				double *HCC,
				double *SCC,
				dcomplex *GR,       /* at w, changed when exit */
				dcomplex *v1,       /* work */
				dcomplex *v2,       /* work */
				dcomplex *value     /* output, transmission */
				);


/* TRAN_Calc_SelfEnergy.c  */
void TRAN_Calc_SelfEnergy( 
            dcomplex w,  
            int ne,     
            dcomplex *gr,  
            int nc,
            dcomplex *hce,    
            dcomplex *sce,    
            dcomplex *sigma   
          );


/* TRAN_Calc_Hopping_G.c */
void TRAN_Calc_Hopping_G(
                         /* input */
			 dcomplex w,
			 int ne,          /* size of electrode */ 
			 int nc,          /* size of central region */
			 dcomplex *gs,    /* surface green function of electrode, size=ne*ne */
			 dcomplex *gc,    /* green function of central region, size=nc*nc    */
			 dcomplex *hce,   /* e.g., HCL,    size=nc*ne */
			 dcomplex *sce,   /* e.g., SCL,    size=nc*ne */

                         /* output */
			 dcomplex *gh     /* e.g., GCL_R , size=nc*ne */
			 );


/* TRAN_Calc_SurfGreen.c  */
void TRAN_Calc_SurfGreen_direct(
				dcomplex w,
				int n, 
				dcomplex *h00, 
				dcomplex *h01,
				dcomplex *s00,
				dcomplex *s01,
                                int iteration_max,
                                double eps,
	 			dcomplex *gr  
				);

/* TRAN_Calc_Transmission.c  */
double TRAN_Calc_Transmission(
		int iter, 
		int SpinP_switch,
		double *****nh,   
		double *****ImNL,  
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int **atv_ijk,
		int *List_YOUSO
);


/* TRAN_Connect_Read_Density.c  */
void TRAN_Connect_Read_Density(
    char *filename,
    int SpinP_switch,
    int Ngrid1, 
    int Ngrid2, 
    int Ngrid3,
    double *ChemP,
    double *minE, 
    double *dVHart_Grid,
    double **Vpot_Grid,
    double **Density_Grid
);

/* TRAN_Connect_Read_Hamiltonian.c  */
/* *static
 * void compare_and_print_error(char *buf,char *str,int val1,int val2);*/
/* TRAN_Connect_Read_Hamiltonian.c  */
void TRAN_Connect_Read_Hamiltonian(
    char *filename, 
    int SpinP_switch, 
    int *WhatSpecies,
    int *FNAN, 
    int **natn,
    int **ncn, 
    int *Spe_Total_CNO, 
    double *****H, 
    double ****OLP
);

/* TRAN_Credit.c  */
void TRAN_Credit(MPI_Comm comm1);

/* TRAN_DFT.c  */
/* *static
 * void TRAN_Set_CDM(
 *    MPI_Comm comm1, 
 *    int spin,
 *    int Matomnum,
 *    int *M2G,
 *    int *WhatSpecies,
 *    int *Spe_Total_CNO,
 *    int *MP,
 *    int *FNAN,
 *    int**natn,
 *    dcomplex *v,
 *    int NUM_c,
 *    dcomplex w_weight, 
 *    int mode,   
 *    double *****CDM,
 *    double *****EDM,
 *    double ***TRAN_DecMulP, 
 *    double Eele0[2], double Eele1[2], 
 *    double ChemP_e0[2]
 *);*/
/* TRAN_DFT.c  */
double TRAN_DFT(
                MPI_Comm comm1,
                int SucceedReadingDMfile, 
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,   
		double *****ImNL,  
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn, 
		int **ncn,
		int *M2G, 
		int *G2ID, 
                int *F_G2M, 
		int **atv_ijk,
		int *List_YOUSO,
		double *****CDM,   
		double *****EDM,   
                double ***TRAN_DecMulP,
		double Eele0[2], double Eele1[2], 
                double ChemP_e0[2]);

/* TRAN_DFT_Dosout.c  */
double TRAN_DFT_Dosout(
                MPI_Comm comm1,
                int level_stdout,
		int iter, 
		int SpinP_switch,
		double *****nh,   
		double *****ImNL,  
		double ****CntOLP, 
		int atomnum,
		int Matomnum,
		int *WhatSpecies,
		int *Spe_Total_CNO,
		int *FNAN,
		int **natn,  
		int **ncn,
		int *M2G, 
		int *G2ID, 
		int **atv_ijk,
		int *List_YOUSO,
                int **Spe_Num_CBasis,
                int SpeciesNum,
                char *filename,
                char *filepath,
		double *****CDM,   
		double *****EDM,   
		double Eele0[2], double Eele1[2])  ;

/* TRAN_Distribute_Node.c  */
void TRAN_Distribute_Node(
   int Start,int End,
   int numprocs,
   int *IDStart,
   int *IDEnd
 );

/* TRAN_Distribute_Node.c  */
void TRAN_Distribute_Node_Idx(
   int Start,int End,
   int numprocs,
   int eachiwmax,
   int **Idxlist
 );


/* TRAN_Input_std.c  */
void TRAN_Input_std(
  MPI_Comm comm1, 
  int Solver,           
  int SpinP_switch,  
  char *filepath,
  double kBvalue,
  double TRAN_eV2Hartree,  
  double Electronic_Temperature,
  int *output_hks
);

/* TRAN_Input_std_Atoms.c  */
void TRAN_Input_std_Atoms(  MPI_Comm comm1, int Solver );



/* TRAN_Output_HKS.c  */
int TRAN_Output_HKS(char *fileHKS);

/* TRAN_Output_HKS_Write_Grid.c  */
void TRAN_Output_HKS_Write_Grid(
				MPI_Comm comm1,
                                int mode,
				int Ngrid1,
				int Ngrid2,
				int Ngrid3,
				double *data,
				double *data1,
				double *data2,
				FILE *fp
				);

/* TRAN_Output_Trans_HS.c  */
/* revised by Y. Xiao for Noncollinear NEGF calculations */ /* iHNL is added */
void  TRAN_Output_Trans_HS(
        MPI_Comm comm1,
        int Solver,
        int SpinP_switch, 
        double ChemP ,
        double *****H,
        double *****iHNL,
        double *****OLP,
        double *****H0,
        int atomnum,
        int SpeciesNum,
        int *WhatSpecies,
        int *Spe_Total_CNO,
        int *FNAN,
        int **natn,
        int **ncn,
        int *G2ID,
        int **atv_ijk,
        int Max_FSNAN,
        double ScaleSize,
        int *F_G2M,      
        int TCpyCell,
        int *List_YOUSO,
        char *filepath,
        char *filename,
        char *fname  );
/* until here  by Y. Xiao for Noncollinear NEGF calculations */

/* TRAN_Output_Transmission.c  */ 
void TRAN_Output_Transmission(int SpinP_switch);

/* TRAN_Add_Density_Lead.c  */
void TRAN_Add_Density_Lead(
            MPI_Comm comm1,
            int SpinP_switch,
            int Ngrid1,
            int Ngrid2,
            int Ngrid3,
            int My_NumGridB_AB,
            double **Density_Grid_B);

/* TRAN_Add_ADensity_Lead.c */
void TRAN_Add_ADensity_Lead(
            MPI_Comm comm1,
            int SpinP_switch,
            int Ngrid1,
            int Ngrid2,
            int Ngrid3,
            int My_NumGridB_AB,
            double *ADensity_Grid_B);

/* TRAN_Poisson.c  */
double TRAN_Poisson(double *ReRhok, double *ImRhok);

/* FFT2D_Density.c  */
double FFT2D_Density(int den_flag, 
                     double *ReRhok, double *ImRhok);

/* added by mari 08.12.2014 */
double FFT1D_Density(int den_flag, 
                     double *ReRhok, double *ImRhok,
                     dcomplex ***VHart_Boundary_a, 
                     dcomplex **VHart_Boundary_b);

/* Get_Value_inReal2D.c  */
void Get_Value_inReal2D(int complex_flag,
                        double *ReVr, double *ImVr, 
                        double *ReVk, double *ImVk);
/* added by mari 18.12.2014 */
void Get_Value_inReal1D(int complex_flag,
                        double *ReVr, double *ImVr, 
                        double *ReVk, double *ImVk);

/* TRAN_Print.c  */
void TRAN_Print2_set_eps(double e1);

/* TRAN_Print.c  */
void TRAN_Print2_set_max(int m1);

/* TRAN_Print.c  */
void TRAN_Print2_dcomplex(char *name, int n1,int n2,dcomplex *gc);

/* TRAN_Print.c  */
void TRAN_Print2_double(char *name, int n1,int n2,double *gc);

/* TRAN_Print.c  */
void TRAN_Print2_dx_dcomplex(char *name, int n1,int dx1, int n2,int dx2, dcomplex *gc);

/* TRAN_Print.c  */
void TRAN_Print2_dx_double(char *name, int n1,int dx1,int n2,int dx2,double *gc);

/* TRAN_Print.c  */
void TRAN_FPrint2_double(char *name, int n1,int n2,double *gc);

/* TRAN_Print.c  */
void TRAN_FPrint2_dcomplex(char *name, int n1,int n2,dcomplex *gc);

/* TRAN_Print.c  */
void TRAN_FPrint2_binary_double(FILE *fp, int n1,int n2,double *gc);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid_Cell1(
  char *filename,
  int n1, int n2, int n3, 
   int *My_Cell1,
   double *realgrid
);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid_Cell0(
   char *filename,
   double origin[4],  
   double gtv[4][4],  
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,   
   double  R[4],   
   int *Cell0,
   double *grid_value
);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid(
   char *filename,
   double origin[4],  
   double gtv[4][4],  
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,   
   double  R[4],   
   double *grid_value 
);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid_z(
   char *filename,
   double origin[4],  
   double gtv[4][4],  
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,   
   double  R[4],   
   double *grid_value
);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid_c(
   char *filenamer,
   char *filenamei,
   double origin[4],  
   double gtv[4][4],  
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,   
   double  R[4],   
   dcomplex *grid_value
);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid_v(
   char *filename,
   double origin[4],  
   double gtv[4][4],  
   int Ngrid1, int Ngrid2, int Ngrid3s, int Ngrid3e,   
   double  R[4],   
   double ***grid_value
);

/* TRAN_Print_Grid.c  */
void TRAN_Print_Grid_Startv(
   char *filename,
   int Ngrid1, int Ngrid2,  int Ngrid3,   
   int Start,
   double ***grid_value
);

/* TRAN_Read.c  */
void TRAN_Read_double(char *str, int n1,int n2, double *a);

/* TRAN_Read.c  */
void TRAN_Read_dcomplex(char *str, int n1,int n2, dcomplex *a);

/* TRAN_Read.c  */
void TRAN_FRead2_binary_double(FILE *fp, int n1, int n2, double *gc);

/* TRAN_RestartFile.c  */
int TRAN_Input_HKS( MPI_Comm comm1, char *fileHKS);

/* TRAN_RestartFile.c  */
int TRAN_RestartFile(MPI_Comm comm1, char *mode, char *position,char *filepath, char *filename);

/* TRAN_Set_CentOverlap.c  */
void TRAN_Set_CentOverlap( 
			   MPI_Comm comm1,
			   int job, 
			   int SpinP_switch, 
                           double k2,
                           double k3,
                           int *order_GA,
			   double **H1,
			   double *S1,
			   double *****H,  
			   double ****OLP,  
			   int atomnum,
			   int Matomnum,
 			   int *M2G, 
			   int *G2ID, 
			   int *WhatSpecies,
			   int *Spe_Total_CNO,
			   int *FNAN,
			   int **natn,
			   int **ncn, 
			   int **atv_ijk
			   );

/* TRAN_Set_Electrode_Grid.c  */
void TRAN_Set_Electrode_Grid(MPI_Comm comm1,
                             int *TRAN_Poisson_flag2,
			     double *Grid_Origin,   /* origin of the grid */
			     double tv[4][4],       /* unit vector of the cell*/
			     double Left_tv[4][4],  /* unit vector  left */
			     double Right_tv[4][4], /* unit vector right */
			     double gtv[4][4],      /* unit vector of the grid point, which is gtv*integer */
			     int Ngrid1,
			     int Ngrid2,
			     int Ngrid3             /* # of c grid points */
			     );


/* TRAN_Set_Electrode_Grid.c  */
/* *static
 * void  TRAN_FFT_Electrode_Grid(MPI_Comm comm1, int isign);*/
/* TRAN_Set_IntegPath.c  */
void TRAN_Set_IntegPath_Square(void);

/* TRAN_Set_IntegPath.c  */
void      TRAN_Set_IntegPath_ThermalArc(void);

/* TRAN_Set_IntegPath.c  */
void TRAN_Set_IntegPath( MPI_Comm comm1,
			 double TRAN_eV2Hartree,
			 double kBvalue, double Electronic_Temperature );

/* TRAN_Check_Input.c */
void TRAN_Check_Input( MPI_Comm comm1, int Solver );

/* TRAN_Set_MP.c  */
void TRAN_Set_MP(
        int job, 
        int atomnum, int *WhatSpecies, int *Spe_Total_CNO, 
        int *NUM,   
        int *MP     
);

/* TRAN_Set_PathEnergyStr.c  */
/* *static
 * void TRAN_error_and_exit(
 *    char *buf
 *);*/
/* TRAN_Set_PathEnergyStr.c  */
void TRAN_Set_PathEnergyStr_Square(
   int m,
   char **str,
   double default_relative_ene[4],  
   double tran_square_path_ene[4],  
   int    tran_square_path_ene_fix[4]
);


/* TRAN_Set_SurfOverlap.c  */
void TRAN_Set_SurfOverlap( MPI_Comm comm1, char *position, double k2, double k3 );

/* TRAN_Set_Value.c  */
void TRAN_Set_Value_double(dcomplex *A, int n, double a, double b);

/* TRAN_adjust_Ngrid.c  */
void  TRAN_adjust_Ngrid( MPI_Comm comm1, int *Ngrid1,int *Ngrid2, int *Ngrid3);


/* TRAN_Check_Region_Lead.c */
int TRAN_Check_Region_Lead(
		  int atomnum,
		    int *WhatSpecies, 
		      double *Spe_Atom_Cut1,
		        double **Gxyz,
			  double tv[4][4]
		);

/* TRAN_Check_Region.c */
int TRAN_Check_Region(
                      int atomnum,
                      int *WhatSpecies, 
                      double *Spe_Atom_Cut1,
                      double **Gxyz
                      );

/* TRAN_Main_Analysis.c */
void TRAN_Main_Analysis(MPI_Comm comm1, 
                        int argc, char *argv[], 
                        int Matomnum, int *M2G, 
                        int *GridN_Atom, 
                        int **GridListAtom,
                        int **CellListAtom,
                        Type_Orbs_Grid ***Orbs_Grid,
                        int TNumGrid);

/* TRAN_Main_Analysis_NC.c */
void TRAN_Main_Analysis_NC( MPI_Comm comm1, 
                            int argc, char *argv[], 
                            int Matomnum, int *M2G, 
                            int *GridN_Atom, 
                            int **GridListAtom,
                            int **CellListAtom,
                            Type_Orbs_Grid ***Orbs_Grid,
                            int TNumGrid );

/* revised by Y. Xiao for Noncollinear NEGF calculations */
double TRAN_DFT_NC(
                MPI_Comm comm1,
                int SucceedReadingDMfile,
                int level_stdout,
                int iter,
                int SpinP_switch,
                double *****nh,
                double *****ImNL,
                double ****CntOLP,
                int atomnum,
                int Matomnum,
                int *WhatSpecies,
                int *Spe_Total_CNO,
                int *FNAN,
                int **natn,
                int **ncn,
                int *M2G,
                int *G2ID,
                int *F_G2M,
                int **atv_ijk,
                int *List_YOUSO,
                double *koS,
                dcomplex **S,                
                double *****CDM,
                double *****iCDM,
                double *****EDM,
                double ***TRAN_DecMulP,
                double Eele0[2], double Eele1[2],
                double ChemP_e0[2]);

void TRAN_Allocate_Cregion_NC(MPI_Comm mpi_comm_level1,
                           int SpinP_switch,
                           int atomnum,
                           int *WhatSpecies,
                           int *Spe_Total_CNO
                           );

void TRAN_Deallocate_Cregion_NC(int SpinP_switch);

void TRAN_Allocate_Lead_Region_NC( MPI_Comm mpi_comm_level1 );

void TRAN_Deallocate_Lead_Region_NC();

void TRAN_Set_CentOverlap_NC(
                          MPI_Comm comm1,
                          int job,
                          int SpinP_switch,
                          double k2,
                          double k3,
                          int *order_GA,
                          double **H1,
                          double **H2,
                          double *S1,
                          double *****H,
                          double ****OLP,
                          int atomnum,
                          int Matomnum,
                          int *M2G,
                          int *G2ID,
                          int *WhatSpecies,
                          int *Spe_Total_CNO,
                          int *FNAN,
                          int **natn,
                          int **ncn,
                          int **atv_ijk
                          );

void TRAN_Set_SurfOverlap_NC( MPI_Comm comm1,
                           char *position,
                           double k2,
                           double k3);
/* until here by Y. Xiao for Noncollinear NEGF calculations */

