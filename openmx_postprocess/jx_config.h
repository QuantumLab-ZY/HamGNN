char *filename_scfout;
char *filename_jxconfig;

int flag_periodic_sum;
int flag_minimal;

int num_poles;
int num_Kgrid[3];

int num_ij_total;
int num_ij_bunch;

int **id_atom;
int **id_cell;

int flag_LNO;
double LNO_Occ_Cutoff;

int read_jx_config(char *file, int Solver);
