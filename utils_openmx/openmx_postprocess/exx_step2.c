/*----------------------------------------------------------------------
  exx_step2.c
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "exx.h"
#include "exx_log.h"
#include "exx_file_overlap.h"
#include "exx_file_eri.h"
#include "exx_index.h"
#include "exx_vector.h"
#include "exx_step2.h"
#include "eri.h"
#include "eri_def.h"
#include "eri_gtbl.h"


#define SUMSKIP_THRESHOLD 1e-10
#define PI 3.1415926535897932384626

#define NRNMAX 1024
#define NBMAX 20

#define DIVISION_SEQUENTIAL  1
#ifdef EXX_USE_MPI
#define DIVISION_MASTERSLAVE 2
#endif

#ifdef EXX_USE_MPI
/*----------------------------------------------------------------------
  CONSTANT VALUES for master-slave division of tasks
----------------------------------------------------------------------*/
/* message tags */
#define MS_TAG_SLAVE_STATUS 11
#define MS_TAG_COMMAND      12
#define MS_TAG_TASKID       13

/* slave status */ 
#define MS_SLAVE_IS_BUSY   101
#define MS_SLAVE_IS_IDLE   102

/* commands */
#define MS_COMMAND_TASK    201
#define MS_COMMAND_QUIT    202
#endif


static void step2_core(
  const EXX_t *exx,
  ERI_t *solver,
  int iop1, 
  int iop2,
  double *et1, 
  double *et2, 
  unsigned long long *out_cnt1,
  unsigned long long *out_cnt2
)
{
  int i, k, f, j;
  int iatom1, iatom2, iatom3, iatom4;
  int icell1, icell2, icell3;
  int ib1, ib2, ib3, ib4, nb1, nb2, nb3, nb4;
  int ncd_ep, ncell_ep, ic0_ep;
  double rc1, rc2, rc3, rc4, rc12, rc34, d12, d34, d, cx12, cx34;
  double c1[3], c2[3], c3[3], c4[3], c12[3], c34[3], c34_off[3];
  double cc[3], v_R[3], R, Rtheta, Rphi;
  
  char path_cache[EXX_PATHLEN];

  /* EXX info */
  int nop, natom, nshop, nshep;
  const int *op_atom1;   /* [nop] */
  const int *op_atom2;   /* [nop] */
  const int *op_cell;    /* [nop] */
  const int *atom_nb;    /* [natom] */
  const double *atom_rc; /* [natom] */
  const double *atom_v;  /* [natom*3] */
  const double *pvec;    /* [9] */
  double rc_cut, w_scr;
  int nbmax;
  double scr;
  int iscr;

  int lmax, lmax_gl, ngrid, ngl, ndglf, nd;
  const double *glf1, *glf2;
  double *op1, *op2;
  int jmax_gl;

  /* packed matrix */
  int np1, np2, ip1, ip2; 
  const double *glf1_max;
  const double *glf2_max;
  
  /* timing */
  clock_t clk1, clk2, clk3;

  double rmin, thresh;
  
  unsigned long long cnt1, cnt2, cnt3;

  double *prej; /* [lmax0*ngl] */
  double *preY; /* [lmax0*lmax0*2] */

  /* Rn cache */
  int nrn, irn, nrn0;
  double list_R[NRNMAX];
  double list_Rt[NRNMAX];
  double list_Rp[NRNMAX];
  int    list_iRd[NRNMAX];
  double I4[NRNMAX], X[NRNMAX*ERI_LMAXMAX*ERI_LMAXMAX];  
  double *eri_buffer; /* nbmax^4*nrnmax */
 
  double *mm_A, *mul_gc, *glY;
  int    *mul_j2, *mul_n, *minimalR;
  int num_minimalR;
 
  thresh = 1e-8;

  nop   = EXX_Number_of_OP(exx);
  natom = EXX_natom(exx);
  nshop = EXX_Number_of_OP_Shells(exx);
  nshep = EXX_Number_of_EP_Shells(exx);
  
  ncd_ep   = 2*nshep+1;
  ncell_ep = ncd_ep*ncd_ep*ncd_ep;
  ic0_ep   = (ncell_ep-1)/2;
  
  op_atom1 = EXX_Array_OP_Atom1(exx);
  op_atom2 = EXX_Array_OP_Atom2(exx);
  op_cell  = EXX_Array_OP_Cell(exx);
  
  atom_rc  = EXX_atom_rc(exx);
  atom_v   = EXX_atom_v(exx);
  atom_nb  = EXX_atom_nb(exx);
  pvec     = EXX_pvec(exx);
  w_scr    = EXX_w_scr(exx);
  rc_cut   = EXX_rc_cut(exx);
  nbmax    = EXX_nbmax(exx);

  ngl     = ERI_ngl(solver);
  lmax_gl = ERI_lmax_gl(solver);
  ndglf   = ERI_Size_of_GLF(solver)/sizeof(double);
  jmax_gl = lmax_gl*lmax_gl;

  snprintf(path_cache, EXX_PATHLEN, "%s/exxovlp.dat", EXX_CacheDir(exx));

  /* bare/screen */
  iscr = (rc_cut>1e-10);

  /* allocation */
  nd    = ndglf*nbmax*nbmax;
  op1 = (double*)malloc(sizeof(double)*nd);
  op2 = (double*)malloc(sizeof(double)*nd);


  mul_j2 = (int*)malloc(sizeof(int)*jmax_gl*jmax_gl*jmax_gl);
  mul_gc = (double*)malloc(sizeof(double)*jmax_gl*jmax_gl*jmax_gl);
  mul_n  = (int*)malloc(sizeof(double)*jmax_gl*jmax_gl);
  prej   = (double*)malloc(sizeof(double)*NRNMAX*ngl*jmax_gl);
  preY   = (double*)malloc(sizeof(double)*NRNMAX*jmax_gl);
  minimalR = (int*)malloc(sizeof(int)*NRNMAX);

  iatom1 = op_atom1[iop1];
  iatom2 = op_atom2[iop1];
  icell1 = op_cell[iop1];

  EXX_Vector_F2C(c1, &atom_v[3*iatom1], pvec);
  EXX_Vector_F2C_Offsite(c2, &atom_v[3*iatom2], pvec, icell1, nshop);

  rc1 = atom_rc[iatom1];
  rc2 = atom_rc[iatom2];
  d12 = EXX_Vector_Distance(c1, c2);
  EXX_Vector_PAO_Overlap(rc1, rc2, d12, &rc12, &cx12);

  c12[0] = cx12*c1[0] + (1.0-cx12)*c2[0];
  c12[1] = cx12*c1[1] + (1.0-cx12)*c2[1];
  c12[2] = cx12*c1[2] + (1.0-cx12)*c2[2];
  
  iatom3 = op_atom1[iop2];
  iatom4 = op_atom2[iop2];
  icell2 = op_cell[iop2];

  EXX_Vector_F2C(c3, &atom_v[3*iatom3], pvec);
  EXX_Vector_F2C_Offsite(c4, &atom_v[3*iatom4], pvec, icell2, nshop);

  rc3 = atom_rc[iatom3];
  rc4 = atom_rc[iatom4];
  d34 = EXX_Vector_Distance(c3, c4);
  EXX_Vector_PAO_Overlap(rc3, rc4, d34, &rc34, &cx34);

  c34[0] = cx34*c3[0] + (1.0-cx34)*c4[0];
  c34[1] = cx34*c3[1] + (1.0-cx34)*c4[1];
  c34[2] = cx34*c3[2] + (1.0-cx34)*c4[2];

  nb1 = atom_nb[iatom1];
  nb2 = atom_nb[iatom2];
  nb3 = atom_nb[iatom3];
  nb4 = atom_nb[iatom4];

  /* load data */ 
  EXX_File_Overlap_Read(ndglf, nbmax, jmax_gl, iop1, op1, path_cache);
  EXX_File_Overlap_Read(ndglf, nbmax, jmax_gl, iop2, op2, path_cache);

  /* count num of Rn */
  nrn = 0;
  for (icell3=ic0_ep; icell3<ncell_ep; icell3++) {
    f = EXX_Index_Cmp_OP(iatom1, iatom2, iatom3, iatom4,
                         icell1, icell2, icell3, 
                         nshop, nshep);
    if (f>0) { continue; }
    
    EXX_Index_Cell2Cartesian(icell3, nshep, pvec, cc);
       
    c34_off[0] = c34[0] + cc[0];
    c34_off[1] = c34[1] + cc[1];
    c34_off[2] = c34[2] + cc[2];

    if (iscr) {
      d = EXX_Vector_Distance(c12, c34_off);
      rmin = d - rc12 - rc34;
      if (rmin > rc_cut) { continue; }
    }

    /* R = c34 + Rn - c12 */
    v_R[0] = c34_off[0] - c12[0]; 
    v_R[1] = c34_off[1] - c12[1]; 
    v_R[2] = c34_off[2] - c12[2]; 

    /* in spherical coord */
    EXX_Vector_C2S(v_R, &R, &Rtheta, &Rphi);

    if (nrn>=NRNMAX) { 
      EXX_ERROR("NRNMAX is too small!");
    }
    list_R[nrn]  = R;
    list_Rt[nrn] = Rtheta;
    list_Rp[nrn] = Rphi;
    list_iRd[nrn] = icell3;
    nrn++;
  }

  if (nrn) {

    scr = ERI_NOSCREEN;
    /*scr = 0.10A*/
    if (w_scr>1e-10) { scr = w_scr; }

    ERI_Integral_GL_PrejY(solver, list_R, list_Rt, list_Rp, nrn, 
      scr, prej, preY, mul_j2, mul_gc, mul_n,
      minimalR, &num_minimalR);

#if 0 /* EXX_LOG_SWITCH */
    EXX_Log_Print("IOP1= %4d  IOP2= %4d  NRN= %4d\n", iop1, iop2, nrn);
    EXX_Log_Print("NB1= %2d  NB2= %2d  NB3= %2d  NB4= %2d\n",
      nb1, nb2, nb3, nb4);
    for (i=0; i<nrn; i++) {
       EXX_Log_Print("IRd[%3d]= %3d\n", i, list_iRd[i]);
    }
#endif /* EXX_LOG_SWITCH */
  
    eri_buffer = (double*)malloc(sizeof(double)*nb1*nb2*nb3*nb4*nrn);

    clk1 = clock();

    cnt1 = 0;
    cnt2 = 0;
    for (ib1=0; ib1<nb1; ib1++) {
     for (ib2=0; ib2<nb2; ib2++) {
        glf1 = &op1[(ib1*nbmax+ib2)*ndglf]; 
        for (ib3=0; ib3<nb3; ib3++) {
          for (ib4=0; ib4<nb4; ib4++) {
            glf2 = &op2[(ib3*nbmax+ib4)*ndglf]; 

            i = (((ib1*nb2+ib2)*nb3+ib3)*nb4+ib4)*nrn;
            ERI_Integral_GL_Post(solver, &eri_buffer[i], glf1, glf2, nrn,
              prej, preY, mul_j2, mul_gc, mul_n, minimalR, num_minimalR);

            for (irn=0; irn<nrn; irn++) {
              if (fabs(eri_buffer[i+irn])>1e-6) { cnt2++; }
              cnt1++;
            }
          } /* ib4 */
        } /* ib3 */
      } /* ib2 */
    } /* ib1 */
 
    clk2 = clock();

    EXX_File_ERI_Write(exx, eri_buffer, iop1, iop2, 
      nb1, nb2, nb3, nb4, nrn, list_iRd);
   
#if 0 /* EXX_LOG_SWITCH */
    for (ib1=0; ib1<nb1; ib1++) {
      for (ib2=0; ib2<nb2; ib2++) {
        for (ib3=0; ib3<nb3; ib3++) {
          for (ib4=0; ib4<nb4; ib4++) {
            for (irn=0; irn<nrn; irn++) {
              EXX_Log_Print("IB1= %2d  IB2= %2d  ", ib1, ib2);
              EXX_Log_Print("IB3= %2d  IB4= %2d  ", ib3, ib4);
              EXX_Log_Print("IRN= %3d  ", irn);
              i = (((ib1*nb2+ib2)*nb3+ib3)*nb4+ib4)*nrn+irn;
              EXX_Log_Print("  ERI= %10.6f\n", eri_buffer[i]);
            }
          }
        }
      }
    }
#endif /* EXX_LOG_SWITCH */
  
    free(eri_buffer);
  
    clk3 = clock();
  } else {
    clk1 = clk2 = clk3 = clock();
  }

  free(op1);
  free(op2);

  free(mul_j2);
  free(mul_gc);
  free(mul_n);
  free(prej);
  free(preY);
  free(minimalR);

  *et1 = (double)(clk2-clk1)/(double)CLOCKS_PER_SEC; 
  *et2 = (double)(clk3-clk2)/(double)CLOCKS_PER_SEC; 
  *out_cnt1 = cnt1;
  *out_cnt2 = cnt2;
}


          
int EXX_Step2(const EXX_t *exx)
{
  int i, k, f, iop1, iop2, nop;
  int nproc, rank;
  
  int srank, tag, msg, flag, nrun; 

#ifdef EXX_USE_MPI
  MPI_Status stat;
  MPI_Comm comm;
#endif
  
  /* OPOP */ 
  int nopop, nopop_local, iopop0, iopop, n, m;
  int *g_iop1, *g_iop2;

  /* ERI info */ 
  int lmax, lmax_gl, ngrid, ngl, ndglf, nd;
  ERI_Init_Misc info;
  ERI_t *solver;

  /* timing */
  clock_t clk1, clk2;
  double et, et1_core, et1_core_sum, et2_core, et2_core_sum;

  unsigned long long cnt1, cnt2;  
  unsigned long long cnt1_sum, cnt2_sum;

  int division;

#ifdef EXX_USE_MPI
  comm = g_exx_mpicomm;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
#else
  rank = EXX_ROOT_RANK;
  nproc = 1;
#endif

  nop   = EXX_Number_of_OP(exx);
  
  /*----- initialize LIBERI -----*/
  /*lmax    = 16;*/
  /*lmax_gl = 8;*/
  lmax    = g_exx_liberi_lmax;
  lmax_gl = (g_exx_liberi_lmax+1)/2;
  ngrid   = g_exx_liberi_ngrid;
  ngl     = g_exx_liberi_ngl;

  info.sbttype = ERI_SBT_LINEAR;
  info.rmax = 80.0; 

  solver = ERI_Init(lmax, lmax_gl, ngrid, ngl, ERI_SH_REAL, &info);
  if (NULL==solver) { EXX_ERROR( "failed to initialize LIBERI" ); }

  /* allocation */
  nopop = nop*nop;
  g_iop1 = (int*)malloc(sizeof(int)*nopop);
  g_iop2 = (int*)malloc(sizeof(int)*nopop);

  /* OPOP */
  i = 0;
  for (iop1=0; iop1<nop; iop1++) {
    for (iop2=0; iop2<nop; iop2++) {
      g_iop1[i] = iop1;
      g_iop2[i] = iop2;
      i++;
   }
  }

#ifdef EXX_USE_MPI
  if (nproc < 4) {
    division = DIVISION_SEQUENTIAL;
  } else { 
    division = DIVISION_MASTERSLAVE;
  }
#else
  division = DIVISION_SEQUENTIAL;
#endif /* EXX_USE_MPI */

  EXX_File_ERI_Create(exx);

  clk1 = clock();
  et1_core_sum = 0.0;
  et2_core_sum = 0.0;
  cnt1_sum = 0;
  cnt2_sum = 0;

  switch (division) {
  case DIVISION_SEQUENTIAL:
    n = (nopop/nproc);
    if (nopop%nproc) { n++; }
    m = nopop - nproc*(n-1); 
  
    assert( nopop == (n-1)*nproc + m );

    nopop_local = (rank<m) ? (n) : (n-1);
    EXX_LOG_TRACE_INTEGER("NOPOP_LOCAL",  nopop_local);
    
    iopop0 = rank;
    for (i=0; i<nopop_local; i++) {
      iopop = iopop0 + i*nproc;
      iop1 = g_iop1[iopop];
      iop2 = g_iop2[iopop];
      step2_core(exx, solver, iop1, iop2, 
                 &et1_core, &et2_core, &cnt1, &cnt2);
      et1_core_sum += et1_core;
      et2_core_sum += et2_core;
      cnt1_sum += (unsigned long long)cnt1;
      cnt2_sum += (unsigned long long)cnt2;
    } /* i */ 

    break;

#ifdef EXX_USE_MPI
  case DIVISION_MASTERSLAVE: 
    nopop_local = 0;
    if (0==rank) {
      /* master */
  
      EXX_LOG_MESSAGE("M/S Master: started");
      EXX_LOG_TRACE_INTEGER("nproc", nproc);
      EXX_LOG_TRACE_INTEGER("ntask", nopop);

       /* wait loop */
       nrun = nproc - 1;
       iopop = 0;
       while (nrun) {

        for (srank=1; srank<nproc; srank++) { 
          /* check message from slave */
          tag = MS_TAG_SLAVE_STATUS;
          MPI_Iprobe(srank, tag, comm, &flag,  &stat);

          /* skip hereafter if no message */
          if (0==flag) { continue; } 

          /* revieve the message */
          MPI_Recv(&msg, 1, MPI_INT, srank, tag, comm, &stat);

          switch (msg) {
 
          case MS_SLAVE_IS_IDLE:
            if (iopop<nopop) {
              /* allocate slave another task */
              msg = MS_COMMAND_TASK;
              tag = MS_TAG_COMMAND;
              MPI_Send(&msg, 1, MPI_INT, srank, tag, comm);

              msg = iopop;
              tag = MS_TAG_TASKID;
              MPI_Send(&msg, 1, MPI_INT, srank, tag, comm);

#if 0
              EXX_Log_Print("M/S Master: task %d to slave %d\n", 
                            msg, srank);
#endif
              iopop++;
            } else {
              /* no more task and tell slave to quit */
              msg = MS_COMMAND_QUIT;
              tag = MS_TAG_COMMAND;
              MPI_Send(&msg, 1, MPI_INT, srank, tag, comm);
#if 0
              EXX_Log_Print("M/S Master: quit slave %d\n", srank);
#endif
              nrun--;
            } 
            break;

          default:
            /* program should never come here !! */
            EXX_ERROR("");
          }
        } /* loop of srank */
      } /* loop of nrun */
      EXX_LOG_MESSAGE("M/S Master: end");

    } else {
      /* slaves */
 
      EXX_LOG_MESSAGE("M/S Slave: started");

      flag = 1;
      while ( flag ) 
      {
        /* request next command */
        tag = MS_TAG_SLAVE_STATUS;
        msg = MS_SLAVE_IS_IDLE;
        MPI_Send(&msg, 1, MPI_INT, 0, tag, comm);
 
        /* recieve command */
        tag = MS_TAG_COMMAND;
        MPI_Recv(&msg, 1, MPI_INT, 0, tag, comm, &stat);

        switch (msg) {
        case MS_COMMAND_TASK:
          /* new task */
          tag = MS_TAG_TASKID;
          MPI_Recv(&msg, 1, MPI_INT, 0, tag, comm, &stat);
#if 0
          EXX_Log_Print("M/S Slave: task %d recieved\n", msg);
#endif
          iopop = msg;
          nopop_local++;
      
          iop1 = g_iop1[iopop];
          iop2 = g_iop2[iopop];
          step2_core(exx, solver, iop1, iop2, 
                     &et1_core, &et2_core, &cnt1, &cnt2); 
          et1_core_sum += et1_core;
          et2_core_sum += et2_core;
          cnt1_sum += (unsigned long long)cnt1;
          cnt2_sum += (unsigned long long)cnt2;
          break;
 
        case MS_COMMAND_QUIT:
#if 0
          EXX_LOG_MESSAGE("M/S Slave: quit command recieved");
#endif
          flag = 0;
          break;
       
        default:
          /* program should never come here !! */
          EXX_ERROR("program should never come here!!");
        }

      } /* loop of flag */
      
      EXX_LOG_MESSAGE("M/S Slave: end");
    } /* end if */
    break;
#endif /* EXX_USE_MPI */

  default:
    /* program should never come here !! */
    EXX_ERROR("program should never come here!!");
  } /* end switch */

  clk2 = clock();
  
  free(g_iop1);
  free(g_iop2);

  et = (double)(clk2-clk1)/(double)CLOCKS_PER_SEC;

  EXX_LOG_MESSAGE("STEP2 SUMMARY:");
  EXX_LOG_TRACE_INTEGER("NOPOP_LOCAL", nopop_local);
#if EXX_LOG_SWITCH
  EXX_Log_Print("TRACE: %15s= %.2e\n", "NQ_ALL", (double)cnt1_sum);
  EXX_Log_Print("TRACE: %15s= %.2e\n", "NQ_SIG", (double)cnt2_sum);
#endif
  EXX_LOG_TRACE_DOUBLE("TOTAL_TIME",    et);
  EXX_LOG_TRACE_DOUBLE("CORE_TIME" ,    et1_core_sum);
  EXX_LOG_TRACE_DOUBLE("AVG_CORE_TIME", et1_core_sum/(double)cnt1_sum);
  EXX_LOG_TRACE_DOUBLE("CORE_TIME(2)" ,    et2_core_sum);
  EXX_LOG_TRACE_DOUBLE("AVG_CORE_TIME(2)", et2_core_sum/(double)cnt1_sum);
  EXX_LOG_MESSAGE("");
  
  ERI_Free(solver);

  return 0;
}

