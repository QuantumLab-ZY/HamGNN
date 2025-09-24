/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#ifndef EXX_H_INCLUDED
#define EXX_H_INCLUDED


/* CONSTANTS */
#define EXX_SYSTEM_PERIODIC  1 /* periodic */
#define EXX_SYSTEM_CLUSTER   2 /* non-periodic */

#define EXX_ROOT_RANK 0 
#define EXX_PATHLEN 256

#include "exx_def_openmx.h"
#include "exx_log.h"

typedef struct EXX_Struct EXX_t;


#ifdef EXX_USE_MPI
#include <mpi.h>
extern MPI_Comm g_exx_mpicomm;
#else
extern int g_exx_mpicomm;
#endif


extern int g_exx_skip1;
extern int g_exx_skip2;
extern char g_exx_cachedir[EXX_PATHLEN];
extern int g_exx_liberi_lmax;
extern int g_exx_liberi_ngrid;
extern int g_exx_liberi_ngl;
extern double g_exx_rc_cut;
extern double g_exx_w_scr;

EXX_t* EXX_New(
  int           natom,
  const double *atom_v,
  const int    *atom_sp,
  int           nspec,
  const double *spec_rc,
  const int    *spec_nb,
  const double *pvec,
  double        w_scr,
  double        rc_cut,
  int           mode,
  const char   *cachedir
);

void EXX_Free(EXX_t *);



int EXX_natom(const EXX_t *self);
const double* EXX_atom_rc(const EXX_t *self);
const double* EXX_atom_v(const EXX_t *self);
const int* EXX_atom_nb(const EXX_t *self);
const double* EXX_pvec(const EXX_t *self);
double EXX_w_scr(const EXX_t *self);
double EXX_rc_cut(const EXX_t *self);
int EXX_nbmax(const EXX_t *self);

int EXX_Number_of_OP_Shells(const EXX_t *self);
int EXX_Number_of_OP(const EXX_t *self);
const int* EXX_Array_OP_Atom1(const EXX_t *self);
const int* EXX_Array_OP_Atom2(const EXX_t *self);
const int* EXX_Array_OP_Cell(const EXX_t *self);

int EXX_Number_of_EP_Shells(const EXX_t *self);
int EXX_Number_of_EP(const EXX_t *self);
const int* EXX_Array_EP_Atom1(const EXX_t *self);
const int* EXX_Array_EP_Atom2(const EXX_t *self);
const int* EXX_Array_EP_Cell(const EXX_t *self);

const char* EXX_CacheDir(const EXX_t *self);


int EXX_Find_OP(
  const EXX_t *self,
  int          iatom1,
  int          iatom2,
  int          iR_x,
  int          iR_y,
  int          iR_z
);

int EXX_Find_EP(
  const EXX_t *self,
  int          iatom1,
  int          iatom2,
  int          iR_x,
  int          iR_y,
  int          iR_z
);

#if 0
int EXX_Make_Quartets(
  const EXX_t *xfm,
  int *q_op1, /* [nq] */
  int *q_op2, /* [nq] */
  int *q_opd, /* [nq] */
  int *q_wf,  /* [nq] */
  int *q_ep1, /* [nq*8] */
  int *q_ep2, /* [nq*8] */
  int *q_mul  /* [nq] */
);
#endif

#endif /* EXX_H_INCLUDED */
