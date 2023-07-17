/**********************************************************************
  DFTDvdW_init.c:
  
     DFTDvdW_init.c is a subroutine to initialize the DFT-D calculation.

  Log of DFTDvdW_init.c:

     08/Apr/2011  Released by Yukihiro Okuno

     The original code is taken from the PWSCF Quantum Espresso. 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time   0


void DFTDvdW_init(){

  double vdW_C6[104];
  double vdW_R[104];
  double maxlentv,templen;
  int i,j,iZ;
  int myid,numprocs;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if(myid==Host_ID){
    printf("<DFTDvdW_init>\n");fflush(stdout);
  }

  /************************************************************
       The C6 unit is Rydberg. The length is in Bohr (AU)
    These values are taken from pwscf Quantum Espresso Program 
  *************************************************************/

  vdW_C6[0]=        0.000; vdW_R[0]=    4.000;
  vdW_C6[1]=        4.857; vdW_R[1]=    1.892;
  vdW_C6[2]=        2.775; vdW_R[2]=    1.912;
  vdW_C6[3]=       55.853; vdW_R[3]=    1.559;
  vdW_C6[4]=       55.853; vdW_R[4]=    2.661;
  vdW_C6[5]=      108.584; vdW_R[5]=    2.806;
  vdW_C6[6]=       60.710; vdW_R[6]=    2.744;
  vdW_C6[7]=       42.670; vdW_R[7]=    2.640;
  vdW_C6[8]=       24.284; vdW_R[8]=    2.536;
  vdW_C6[9]=       26.018; vdW_R[9]=    2.432;
  vdW_C6[10]=      21.855; vdW_R[10]=   2.349;
  vdW_C6[11]=     198.087; vdW_R[11]=   2.162;
  vdW_C6[12]=     198.087; vdW_R[12]=   2.578;
  vdW_C6[13]=     374.319; vdW_R[13]=   3.097;
  vdW_C6[14]=     320.200; vdW_R[14]=   3.243;
  vdW_C6[15]=     271.980; vdW_R[15]=   3.222;
  vdW_C6[16]=     193.230; vdW_R[16]=   3.180;
  vdW_C6[17]=     175.885; vdW_R[17]=   3.097;
  vdW_C6[18]=     159.927; vdW_R[18]=   3.014;
  vdW_C6[19]=     374.666; vdW_R[19]=   2.806;
  vdW_C6[20]=     374.666; vdW_R[20]=   2.785;
  vdW_C6[21]=     374.666; vdW_R[21]=   2.952;
  vdW_C6[22]=     374.666; vdW_R[22]=   2.952;
  vdW_C6[23]=     374.666; vdW_R[23]=   2.952;
  vdW_C6[24]=     374.666; vdW_R[24]=   2.952;
  vdW_C6[25]=     374.666; vdW_R[25]=   2.952;
  vdW_C6[26]=     374.666; vdW_R[26]=   2.952;
  vdW_C6[27]=     374.666; vdW_R[27]=   2.952;
  vdW_C6[28]=     374.666; vdW_R[28]=   2.952;
  vdW_C6[29]=     374.666; vdW_R[29]=   2.952;
  vdW_C6[30]=     374.666; vdW_R[30]=   2.952;
  vdW_C6[31]=     589.405; vdW_R[31]=   3.118;
  vdW_C6[32]=     593.221; vdW_R[32]=   3.264;
  vdW_C6[33]=     567.896; vdW_R[33]=   3.326;
  vdW_C6[34]=     438.498; vdW_R[34]=   3.347;
  vdW_C6[35]=     432.600; vdW_R[35]=   3.305;
  vdW_C6[36]=     416.642; vdW_R[36]=   3.264;
  vdW_C6[37]=     855.833; vdW_R[37]=   3.076;
  vdW_C6[38]=     855.833; vdW_R[38]=   3.035;
  vdW_C6[39]=     855.833; vdW_R[39]=   3.097;
  vdW_C6[40]=     855.833; vdW_R[40]=   3.097;
  vdW_C6[41]=     855.833; vdW_R[41]=   3.097;
  vdW_C6[42]=     855.833; vdW_R[42]=   3.097;
  vdW_C6[43]=     855.833; vdW_R[43]=   3.097;
  vdW_C6[44]=     855.833; vdW_R[44]=   3.097;
  vdW_C6[45]=     855.833; vdW_R[45]=   3.097;
  vdW_C6[46]=     855.833; vdW_R[46]=   3.097;
  vdW_C6[47]=     855.833; vdW_R[47]=   3.097;
  vdW_C6[48]=     855.833; vdW_R[48]=   3.097;
  vdW_C6[49]=    1294.678; vdW_R[49]=   3.160;
  vdW_C6[50]=    1342.899; vdW_R[50]=   3.409;
  vdW_C6[51]=    1333.532; vdW_R[51]=   3.555;
  vdW_C6[52]=    1101.101; vdW_R[52]=   3.575;
  vdW_C6[53]=    1092.775; vdW_R[53]=   3.575;
  vdW_C6[54]=    1040.391; vdW_R[54]=   3.555;
  vdW_C6[55]=       0.000; vdW_R[55]=   4.000;
  vdW_C6[56]=       0.000; vdW_R[56]=   4.000;
  vdW_C6[57]=       0.000; vdW_R[57]=   4.000;
  vdW_C6[58]=       0.000; vdW_R[58]=   4.000;
  vdW_C6[59]=       0.000; vdW_R[59]=   4.000;
  vdW_C6[60]=       0.000; vdW_R[60]=   4.000;
  vdW_C6[61]=       0.000; vdW_R[61]=   4.000;
  vdW_C6[62]=       0.000; vdW_R[62]=   4.000;
  vdW_C6[63]=       0.000; vdW_R[63]=   4.000;
  vdW_C6[64]=       0.000; vdW_R[64]=   4.000;
  vdW_C6[65]=       0.000; vdW_R[65]=   4.000;
  vdW_C6[66]=       0.000; vdW_R[66]=   4.000;
  vdW_C6[67]=       0.000; vdW_R[67]=   4.000;
  vdW_C6[68]=       0.000; vdW_R[68]=   4.000;
  vdW_C6[69]=       0.000; vdW_R[69]=   4.000;
  vdW_C6[70]=       0.000; vdW_R[70]=   4.000;
  vdW_C6[71]=       0.000; vdW_R[71]=   4.000;
  vdW_C6[72]=       0.000; vdW_R[72]=   4.000;
  vdW_C6[73]=       0.000; vdW_R[73]=   4.000;
  vdW_C6[74]=       0.000; vdW_R[74]=   4.000;
  vdW_C6[75]=       0.000; vdW_R[75]=   4.000;
  vdW_C6[76]=       0.000; vdW_R[76]=   4.000;
  vdW_C6[77]=       0.000; vdW_R[77]=   4.000;
  vdW_C6[78]=       0.000; vdW_R[78]=   4.000;
  vdW_C6[79]=       0.000; vdW_R[79]=   4.000;
  vdW_C6[80]=       0.000; vdW_R[80]=   4.000;
  vdW_C6[81]=       0.000; vdW_R[81]=   4.000;
  vdW_C6[82]=       0.000; vdW_R[82]=   4.000;
  vdW_C6[83]=       0.000; vdW_R[83]=   4.000;
  vdW_C6[84]=       0.000; vdW_R[84]=   4.000;
  vdW_C6[85]=       0.000; vdW_R[85]=   4.000;
  vdW_C6[86]=       0.000; vdW_R[86]=   4.000;
  vdW_C6[87]=       0.000; vdW_R[87]=   4.000;
  vdW_C6[88]=       0.000; vdW_R[88]=   4.000;
  vdW_C6[89]=       0.000; vdW_R[89]=   4.000;
  vdW_C6[90]=       0.000; vdW_R[90]=   4.000;
  vdW_C6[91]=       0.000; vdW_R[91]=   4.000;
  vdW_C6[92]=       0.000; vdW_R[92]=   4.000;
  vdW_C6[93]=       0.000; vdW_R[93]=   4.000;
  vdW_C6[94]=       0.000; vdW_R[94]=   4.000;
  vdW_C6[95]=       0.000; vdW_R[95]=   4.000;
  vdW_C6[96]=       0.000; vdW_R[96]=   4.000;
  vdW_C6[97]=       0.000; vdW_R[97]=   4.000;
  vdW_C6[98]=       0.000; vdW_R[98]=   4.000;
  vdW_C6[99]=       0.000; vdW_R[99]=   4.000;
  vdW_C6[100]=      0.000; vdW_R[100]=  4.000;
  vdW_C6[101]=      0.000; vdW_R[101]=  4.000;
  vdW_C6[102]=      0.000; vdW_R[102]=  4.000;
  vdW_C6[103]=      0.000; vdW_R[103]=  4.000;

  /* change energy unit from Ry to Hartree */ 
  for(i=0;i<=103; i++){
    vdW_C6[i] = vdW_C6[i]/2.0;
  }

  /* allocation of arrays */

  C6_dftD =(double*)malloc(sizeof(double)*SpeciesNum);
  RvdW_dftD=(double*)malloc(sizeof(double)*SpeciesNum);

  C6ij_dftD = (double**)malloc(sizeof(double*)*SpeciesNum);
  for(i=0; i<SpeciesNum; i++){
    C6ij_dftD[i]=(double*)malloc(sizeof(double)*SpeciesNum);
  }

  Rsum_dftD = (double**)malloc(sizeof(double*)*SpeciesNum);
  for(i=0; i<SpeciesNum; i++){
    Rsum_dftD[i]=(double*)malloc(sizeof(double)*SpeciesNum);
  }

  /* setting of parameters for single atoms */

  for(i=0; i<SpeciesNum; i++){
    C6_dftD[i] = -1.0;
    RvdW_dftD[i] = -1.0;
  }

  for(i=0; i<SpeciesNum; i++){
    iZ = Spe_WhatAtom[i];
    if(0<=iZ && iZ<=103){
      C6_dftD[i] = vdW_C6[iZ];
      RvdW_dftD[i] = vdW_R[iZ];
    }
  }

  for(i=0; i<SpeciesNum; i++){
    if((C6_dftD[i]<0.0)||(RvdW_dftD[i]<0.0)){
      printf("error in DFTDvdw_init: vdW parameters are not found.\n");fflush(stdout);
      MPI_Finalize();
      exit(0);
    }
  }

  /* setting of parameters for atom pairs */

  for(i=0; i< SpeciesNum; i++){
    for(j=0; j<SpeciesNum; j++){
      C6ij_dftD[i][j]= sqrt(C6_dftD[i]*C6_dftD[j]);
      Rsum_dftD[i][j]= RvdW_dftD[i]+RvdW_dftD[j];
    }
  }

  /*
  if(myid==Host_ID){
    printf("DFT-D parameter\n");fflush(stdout);
    for(i=0; i<SpeciesNum; i++){
      printf("%d species C6=%f R=%f \n",i,C6_dftD[i],RvdW_dftD[i]);fflush(stdout);
    }
  }
  */

  /* find n1_DFT_D, n2_DFT_D, and n3_DFT_D */

  i = 1;
  templen = sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);
  n1_DFT_D = (int)ceil(rcut_dftD/templen);
  if (DFTD_IntDir1==0) n1_DFT_D = 0;

  i = 2;
  templen = sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);
  n2_DFT_D = (int)ceil(rcut_dftD/templen);
  if (DFTD_IntDir2==0) n2_DFT_D = 0;

  i = 3;
  templen = sqrt(tv[i][1]*tv[i][1]+tv[i][2]*tv[i][2]+tv[i][3]*tv[i][3]);
  n3_DFT_D = (int)ceil(rcut_dftD/templen);
  if (DFTD_IntDir3==0) n3_DFT_D = 0;

  /*
  printf("n1=%2d n2=%2d n3=%2d\n",n1_DFT_D,n2_DFT_D,n3_DFT_D);
  */

  /* freeing of arrays */

  free(C6_dftD);
  free(RvdW_dftD);

  return;
}
