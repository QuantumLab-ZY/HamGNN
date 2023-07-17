/*----------------------------------------------------------------------
  exx_log.c

  simple logging tool

  Coded by M. Toyoda, 16 Dec. 2009 
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <assert.h>
#include "exx.h"
#include "exx_log.h"

#if (defined _BSD_SOURCE || _XOPEN_SOURCE >= 500)
#include <unistd.h>
#endif

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif 


static void my_gethostname(char *name, size_t length)
{
  char *env;

  /* name should be longer than HOST_NAME_MAX */
#if (defined _BSD_SOURCE || _XOPEN_SOURCE >= 500)
  gethostname(name, length);
  return;
#endif

  /* check env list */
  env = getenv("HOSTNAME");
  if (env && strlen(env)) {
    strncpy(name, env, length);
    return;
  }
  
  /* no way to know it */
  strncpy(name, "host-name-unknown", length);
}



void EXX_Log_StdOut(const char *fmt, ... )
{
  int myrank;
  va_list argp;

#ifdef EXX_USE_MPI
  MPI_Comm_rank(g_exx_mpicomm, &myrank);
#else
  myrank = EXX_ROOT_RANK;
#endif /* EXX_USE_MPI */

  if (EXX_ROOT_RANK == myrank) {
    va_start( argp, fmt );
    vfprintf(stdout, fmt, argp );
    va_end( argp ); 
    fflush( stdout );
  }
}


void EXX_Log_StdErr(const char *fmt, ... )
{
  int myrank;
  va_list argp;

#ifdef EXX_USE_MPI
  MPI_Comm_rank(g_exx_mpicomm, &myrank);
#else
  myrank = EXX_ROOT_RANK;
#endif /* EXX_USE_MPI */

  if (EXX_ROOT_RANK == myrank) {
    va_start( argp, fmt );
    vfprintf(stderr, fmt, argp );
    va_end( argp ); 
    fflush( stderr );
  }
}


void EXX_Error(const char *msg, const char *file, int line)
{
  fprintf(stderr, "***Error in %s (%d)\n", file, line);
  fprintf(stderr, "   %s\n", msg);

#ifdef EXX_USE_MPI
  MPI_Abort(g_exx_mpicomm, -1);
#else
  abort();
#endif
}


void EXX_Warn(const char *msg, const char *file, int line)
{
  fprintf(stderr, "***Warning in %s (%d)\n", file, line);
  fprintf(stderr, "   %s\n", msg);
}


 
#if EXX_LOG_SWITCH

static FILE* fp_log;


void EXX_Log_Open(void)
{
  int nproc, myrank;
  char path[EXX_PATHLEN], host[HOST_NAME_MAX];
  time_t loctim;

  assert( NULL == fp_log );

#ifdef EXX_USE_MPI
  MPI_Comm_rank(g_exx_mpicomm, &myrank);
  MPI_Comm_size(g_exx_mpicomm, &nproc);
#else
  myrank = EXX_ROOT_RANK;
  nproc = 1;
#endif /* EXX_USE_MPI */ 

  /* filename */
  sprintf(path, "log%03d.txt", myrank);
  
  /* open */
  fp_log = fopen(path, "wt");
 
  /* header message */
  fprintf(fp_log, "EXX-LOG OPEN\n");

  /* current date and time */
  time(&loctim);
  fprintf(fp_log, "  DATETIME= %s", ctime(&loctim));

  /* mpi */
  fprintf(fp_log, "  MPI_RANK= %d\n", myrank);
  fprintf(fp_log, "  MPI_NPROC= %d\n", nproc);

  /* host name */
  my_gethostname(host, HOST_NAME_MAX);
  fprintf(fp_log, "  HOSTNAME= %s\n", host);

  /* otuput imediately (if possible) */
  fflush(fp_log);
}


void EXX_Log_Close(void)
{
  time_t loctim;

  /* show tail message */
  fprintf(fp_log, "EXX-LOG CLOSING\n");

  /* current date and time */
  time(&loctim);
  fprintf(fp_log, "  DATETIME= %s", ctime(&loctim));

  fprintf(fp_log, "  GOOD-BYE.\n");

  /* close file */
  fclose(fp_log);

  /* reset the pointer */
  fp_log = NULL;
}


void EXX_Log_Print(const char *fmt, ... )
{
  va_list argp;

  va_start( argp, fmt );
  vfprintf(fp_log, fmt, argp );
  va_end( argp ); 

  fflush(fp_log);
}


void EXX_Log_Message(const char *message)
{
  EXX_Log_Print("%s\n", message);
}


void EXX_Log_Trace_Integer(const char *key, int val)
{
  EXX_Log_Print("TRACE: %15s= %10d\n", key, val);
}


void EXX_Log_Trace_Double(const char *key, double val)
{
  EXX_Log_Print("TRACE: %15s= %20.10f\n", key, val);
}


void EXX_Log_Trace_String(const char *key, const char *val)
{
  EXX_Log_Print("TRACE: %15s= \"%s\"\n", key, val);
}


void EXX_Log_Trace_Vector(const char *key, const double *val)
{
  EXX_Log_Print("TRACE %15s= ( %12.6f, %12.6f, %12.6f )\n", 
    key, val[0], val[1], val[2]);
}

#endif /* EXX_LOG_SWITCH */

