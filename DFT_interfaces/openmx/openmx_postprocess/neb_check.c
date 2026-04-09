/**********************************************************************
  neb_check.c:

     neb_check.c is a subroutine to check 
     whether the calculation is NEB or not.

  Log of neb_check.c:

    13/April/2011  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
#include <omp.h>

  

int neb_check(char *argv[]) 
{ 
  int i,j,flag;
  char *s_vec[50];
  int i_vec[50];

  if (input_open(argv[1])==0){
    MPI_Finalize(); 
    exit(0);
  }

  i=0;
  s_vec[i]="NOMD";                    i_vec[i]=0;  i++;
  s_vec[i]="NVE" ;                    i_vec[i]=1;  i++;
  s_vec[i]="NVT_VS";                  i_vec[i]=2;  i++; /* modified by mari */
  s_vec[i]="OPT";                     i_vec[i]=3;  i++;
  s_vec[i]="EF";                      i_vec[i]=4;  i++; 
  s_vec[i]="BFGS";                    i_vec[i]=5;  i++; 
  s_vec[i]="RF";                      i_vec[i]=6;  i++; /* RF method by hmweng */
  s_vec[i]="DIIS";                    i_vec[i]=7;  i++;
  s_vec[i]="Constraint_DIIS";         i_vec[i]=8;  i++; /* not used */
  s_vec[i]="NVT_NH";                  i_vec[i]=9;  i++; 
  s_vec[i]="Opt_LBFGS";               i_vec[i]=10; i++; 
  s_vec[i]="NVT_VS2";                 i_vec[i]=11; i++; /* modified by Ohwaki */
  s_vec[i]="EvsLC";                   i_vec[i]=12; i++; 
  s_vec[i]="NEB";                     i_vec[i]=13; i++; 
  s_vec[i]="NVT_VS4";                 i_vec[i]=14; i++; /* modified by Ohwaki */
  s_vec[i]="NVT_Langevin";            i_vec[i]=15; i++; /* modified by Ohwaki */
  s_vec[i]="DF";                      i_vec[i]=16; i++; /* delta-factor */
  s_vec[i]="OptC1";                   i_vec[i]=17; i++; /* cell opt with fixed fractional coordinates by SD */
  s_vec[i]="OptC2";                   i_vec[i]=18; i++; /* cell opt with fixed fractional coordinates and angles fixed by SD */
  s_vec[i]="OptC3";                   i_vec[i]=19; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|=|a3| by SD */
  s_vec[i]="OptC4";                   i_vec[i]=20; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|!=|a3| by SD */
  s_vec[i]="OptC5";                   i_vec[i]=21; i++; /* cell opt with no constraint for cell and coordinates by SD */
  s_vec[i]="RFC1";                    i_vec[i]=22; i++; /* cell opt with fixed fractional coordinates by RF */
  s_vec[i]="RFC2";                    i_vec[i]=23; i++; /* cell opt with fixed fractional coordinates and angles fixed by RF */
  s_vec[i]="RFC3";                    i_vec[i]=24; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|=|a3| by RF */
  s_vec[i]="RFC4";                    i_vec[i]=25; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|!=|a3| by RF */
  s_vec[i]="RFC5";                    i_vec[i]=26; i++; /* cell opt with no constraint for cell and coordinates by RF */

  /* added by MIZUHO for NPT-MD */
  s_vec[i]="NPT_VS_PR";               i_vec[i]=27; i++; /* NPT MD implemented by Velocity-Scaling method and Parrinelo-Rahman method */
  s_vec[i]="NPT_VS_WV";               i_vec[i]=28; i++; /* NPT MD implemented by Velocity-Scaling method and Wentzcovitch method */
  s_vec[i]="NPT_NH_PR";               i_vec[i]=29; i++; /* NPT MD implemented by Nose-Hoover method and Parrinelo-Rahman method */
  s_vec[i]="NPT_NH_WV";               i_vec[i]=30; i++; /* NPT MD implemented by Nose-Hoover method and Wentzcovitch method */

  /* variable cell optimization */
  s_vec[i]="RFC6";                    i_vec[i]=31; i++; /* cell opt with fixed a3 vector by RF */
  s_vec[i]="RFC7";                    i_vec[i]=32; i++; /* cell opt with fixed a2 and a3 vector by RF */
  s_vec[i]="OptC6";                   i_vec[i]=33; i++; /* cell opt with fixed a3 vector by SD */
  s_vec[i]="OptC7";                   i_vec[i]=34; i++; /* cell opt with fixed a2 and a3 vector by SD */

  j = input_string2int("MD.Type",&MD_switch, i, s_vec,i_vec);
  if (j==-1){
    MPI_Finalize();
    exit(0);
  }

  /* for NEB */

  if (MD_switch==13){
    neb_type_switch = 1;
  }

  input_close();

  flag = 0;
  if (MD_switch==13) flag = 1;

  return flag;
}













