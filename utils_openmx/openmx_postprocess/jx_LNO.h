#include "mpi.h"
void LNO_alloc(MPI_Comm comm1);
void LNO_free(MPI_Comm comm1);

double LNO_Col_Diag(MPI_Comm comm1);
double ***LNO_coes;
double ***LNO_pops;

void LNO_occ_trns_mat(MPI_Comm comm1);
dcomplex ****LNO_mat;

int *LNO_Num;

//int *LNOs_Num_predefined;
//double LNO_Occ_Cutoff;
//int LNO_flag,LNOs_Num_predefined_flag;
