#define MAXBUF 1024

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Inputtools.h"
//#include "lapack_prototypes.h"
#include "jx_total_mem.h"
#include "jx_config.h"
//#include "jx_LNO.h"

//#include <malloc/malloc.h>
//#include <assert.h>

int read_jx_config(char *file, int Solver){

  FILE *fp;
  int error_code;

  int j;
  int i_vec[3],i_vec2[3];
  char buf[MAXBUF];

  char *str;
  char substr1[16];
  char substr2[16];

  char *s1,*s2;
  int i1,i2,i3,i4,i5;
  double d1,d2,d3,d4,d5;

  if (input_open(file)==0){
    error_code=1;
    return error_code;
  }

  input_logical("Flag.PeriodicSum",&flag_periodic_sum,0);
//  input_logical("Flag.MinimalBasisSet",&flag_minimal,0);
  input_int("Num.Poles",&num_poles,60);
  i_vec2[0]=1;
  i_vec2[1]=1;
  i_vec2[2]=1;
  input_intv("Num.Kgrid",3,i_vec,i_vec2);
  num_Kgrid[0] = i_vec[0];
  num_Kgrid[1] = i_vec[1];
  num_Kgrid[2] = i_vec[2];

  input_int("Num.ij.pairs",&num_ij_total,0);
  //input_int("Bunch.ij.pairs",&num_ij_bunch,1);
  input_int("Bunch.ij.pairs",&num_ij_bunch,num_ij_total);

  if ( num_ij_bunch > num_ij_total ){
    num_ij_bunch=num_ij_total;
  }
//  printf("num_ij_total=%i, num_ij_bunch=%i\n",num_ij_total,num_ij_bunch);

  id_atom = (int**)malloc(sizeof(int*)*num_ij_total);
  for (j=0; j<num_ij_total; j++){
    id_atom[j] = (int*)malloc(sizeof(int)*2);
  }
  id_cell = (int**)malloc(sizeof(int*)*num_ij_total);
  for (j=0; j<num_ij_total; j++){
    id_cell[j] = (int*)malloc(sizeof(int)*2);
  }

  for (j=0; j<num_ij_total; j++){
    id_atom[j][0] = -1;
    id_atom[j][1] = -1;
    id_cell[j][0] = 0;
    id_cell[j][1] = 0;
    id_cell[j][2] = 0;
  }

  if (Solver==3 && flag_periodic_sum==0){
    if (fp=input_find("<ijpairs.cellid") ) {
      for (j=0; j<num_ij_total; j++){
        fgets(buf,MAXBUF,fp);
        sscanf(buf,"%i %i %i %i %i",&id_atom[j][0],&id_atom[j][1],&id_cell[j][0],&id_cell[j][1],&id_cell[j][2]);

      }
      ungetc('\n',fp);
    }
  }
  else{
    if (fp=input_find("<ijpairs.nocellid")) {

      for (j=0; j<num_ij_total; j++){
        fgets(buf,MAXBUF,fp);
        sscanf(buf,"%i %i",&id_atom[j][0],&id_atom[j][1]);
      }
      ungetc('\n',fp);
    }
    else if (fp=input_find("<ijpairs.cellid") ) {
      for (j=0; j<num_ij_total; j++){
        fgets(buf,MAXBUF,fp);
        sscanf(buf,"%i %i %i %i %i",&id_atom[j][0],&id_atom[j][1],&id_cell[j][0],&id_cell[j][1],&id_cell[j][2]);
      }
      ungetc('\n',fp);
    }
  }

  input_logical("Flag.LNO",&flag_LNO,0);
  input_double("LNO.occ.cutoff", &LNO_Occ_Cutoff, 0.0);

  error_code=0;
  return error_code;

}
