/**********************************************************************

jx.c:

This program code calculates spin-spin interaction
coupling constant J between the selected two atoms

Log of jx.c:
   30/Aug/2003  Released by Myung Joon Han (supervised by Prof. J. Yu)
    7/Dec/2003  Modified by Taisuke Ozaki
   03/Mar/2011  Modified by Fumiyuki Ishii for MPI
   02/May/2018  Modified by Asako Terasawa for blas
***********************************************************************/

#pragma once

#define Host_ID       0         /* ID of the host CPU in MPI */
#define Flag_interactive      0
#define printout  0    /* 0:off, 1:on */
#define PI   3.1415926535897932384626

void matmul_dcomplex_lapack(char *typeA, char *typeB,
                            int m, int n, int k,
                            dcomplex **A_2d, dcomplex **B_2d, dcomplex **C_2d);

void matmul_double_lapack(char *typeA, char *typeB,
                          int m, int n, int k,
                          double **A_2d, double **B_2d, double **C_2d);

void Eigen_lapack(double **a, double *ko, int n);
void EigenBand_lapack(dcomplex **A, double *W, int N);
void k_inversion(int i,  int j,  int k,
                        int mi, int mj, int mk,
                        int *ii, int *ij, int *ik );
void Overlap_Band(double ****OLP,
                         dcomplex **S,int *MP,
                         double k1, double k2, double k3);
void Hamiltonian_Band(double ****RH, dcomplex **H, int *MP,
                             double k1, double k2, double k3);
void dtime(double *t);

void matinv_double_lapack(double **A, int N);
void matinv_dcplx_lapack(dcomplex **A, int N);
void zero_cfrac(int n, dcomplex *zp, dcomplex *Rp );

void gen_eigval_herm(dcomplex **A, dcomplex **B, double *V, int N);
void gen_eigval_herm_lapack(dcomplex **A, dcomplex **B, double *V, int N);

void dcplx_basis_transformation(int num, dcomplex **trns_mat_1, dcomplex **mat, dcomplex **trns_mat_2);
