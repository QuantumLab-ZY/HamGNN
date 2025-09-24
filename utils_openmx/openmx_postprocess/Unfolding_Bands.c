/**********************************************************************
  Unfolding_Bands.c

     Unfolding_Bands.c is a subroutine to calculate unfolded weight
     at given k-points for the file output.

  Log of Band_Unfolding.c:

      6/Jan/2016  Released by Chi-Cheng Lee

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

static int NR;
static int Norb;
static int nr,natom;
static int* Norbperatom;
static int*** _MapR;
static int** Rlist;
static int** rlist;
static double***** Elem;
static int*** tabr4RN;
static int*** rnmap;
static int exitcode;
static int totnkpts;
static int* np;

static void determine_kpts(const int nk, double** klist);
static double volume(const double* a,const double* b,const double* c);
static double dot(const double* v1,const double* v2);
static double distwovec(const double* a,const double* b);
static void getnorbperatom();
static void buildMapRlist();
static void buildtabr4RN(const double* a,const double* b,const double* c,double* origin,const int* mapN2n);
static void abc_by_ABC(double** S);
static void buildrnmap(const int* mapN2n);

static void Unfolding_Bands_Col(
				int nkpoint, double **kpoint,
				int SpinP_switch, 
				double *****nh,
				double ****CntOLP);

static void Unfolding_Bands_NonCol(
				   int nkpoint, double **kpoint,
				   int SpinP_switch, 
				   double *****nh,
				   double *****ImNL,
				   double ****CntOLP);


void Unfolding_Bands( int nkpoint, double **kpoint,
		      int SpinP_switch, 
		      double *****nh,
		      double *****ImNL,
		      double ****CntOLP)
{
  if (SpinP_switch==0 || SpinP_switch==1){
    Unfolding_Bands_Col( nkpoint, kpoint, SpinP_switch, nh, CntOLP);
  }
  else if (SpinP_switch==3){
    Unfolding_Bands_NonCol( nkpoint, kpoint, SpinP_switch, nh, ImNL, CntOLP);
  }
}



static void Unfolding_Bands_Col(
				int nkpoint, double **kpoint,
				int SpinP_switch, 
				double *****nh,
				double ****CntOLP)
{
  double coe;
  double* a;
  double* b;
  double* c;
  double* K;
  double* K2;
  double* r;
  double* r0;
  double* kj_e;
  int* iskj_e;
  int countkj_e;
  double pk1,pk2,pk3;
  double dis2pk;
  double kdis;
  dcomplex** weight;
  dcomplex*** kj_v;
  dcomplex**** tmpelem;
  double **fracabc;

  int i,j,k,l,n,wan;
  int *MP,*order_GA,*My_NZeros,*SP_NZeros,*SP_Atoms;
  int i1,j1,po,spin,n1,size_H1;
  int num2,RnB,l1,l2,l3,kloop;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum;
  int ii,ij,ik,Rn,AN;
  int num0,num1,mul,m,wan1,Gc_AN;
  int LB_AN,GB_AN,Bnum;
  double time0,tmp,tmp1,av_num;
  double snum_i,snum_j,snum_k,k1,k2,k3,sum,sumi,Num_State,FermiF;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN,EV_cut0;
  double **ko,*M1,**EIGEN;
  double *koS;
  double *S1,**H1;
  dcomplex ***H,**S,***C;
  dcomplex Ctmp1,Ctmp2;
  double u2,v2,uv,vu;
  double dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2,Imdum;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double FermiEps = 1.0e-14;
  double x_cut = 30.0;
  double OLP_eigen_cut=Threshold_OLP_Eigen;
  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10];
  FILE *fp_EV0;
  FILE *fp_EV;
  FILE *fp_EV1;
  FILE *fp_EV2;
  FILE *fp_EV3;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;
  int *is1,*ie1;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout) {
    printf("\n*******************************************************\n");
    printf("                 Unfolding of Bands \n");
    printf("*******************************************************\n\n");fflush(stdout);
  } 
  dtime(&TStime);

  /****************************************************
                  allocation of arrays
  ****************************************************/
  
  getnorbperatom();
  exitcode=0;
  buildMapRlist();
  if (exitcode==1) {
    for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
    free(unfold_origin);
    free(unfold_mapN2n);
    for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
    return;
  }
  
  a = (double*)malloc(sizeof(double)*3);
  b = (double*)malloc(sizeof(double)*3);
  c = (double*)malloc(sizeof(double)*3);
  a[0]=unfold_abc[0][0];
  a[1]=unfold_abc[0][1];
  a[2]=unfold_abc[0][2];
  b[0]=unfold_abc[1][0];
  b[1]=unfold_abc[1][1];
  b[2]=unfold_abc[1][2];
  c[0]=unfold_abc[2][0];
  c[1]=unfold_abc[2][1];
  c[2]=unfold_abc[2][2];
  buildtabr4RN(a,b,c,unfold_origin,unfold_mapN2n);
  if (exitcode==1) {
    for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
    free(unfold_origin);
    free(unfold_mapN2n);
    for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
    free(a);
    free(b);
    free(c);
    return;
  }

  if (myid==Host_ID && 1<level_stdout) {
    printf("Reference origin is set to (%f %f %f) (Bohr)\n",unfold_origin[0],unfold_origin[1],unfold_origin[2]);
    printf("Supercell_lattice_vector atom Reference_lattice_vector atom\n");
    for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) 
      printf("(%i %i %i) %i <==> (%i %i %i) %i\n",Rlist[i][0],Rlist[i][1],Rlist[i][2],j+1,tabr4RN[i][j][0],tabr4RN[i][j][1],tabr4RN[i][j][2],unfold_mapN2n[j]);
    printf("\n");fflush(stdout);
  }

  coe=Cell_Volume/volume(a,b,c);
  fracabc=(double**)malloc(sizeof(double*)*3);
  for (i=0; i<3; i++) fracabc[i]=(double*)malloc(sizeof(double)*3);
  abc_by_ABC(fracabc);
  determine_kpts(nkpoint,kpoint);

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));
  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);
  
  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }

  ko = (double**)malloc(sizeof(double*)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    ko[i] = (double*)malloc(sizeof(double)*(n+1));
  }

  koS = (double*)malloc(sizeof(double)*(n+1));

  EIGEN = (double**)malloc(sizeof(double*)*List_YOUSO[23]);
  for (j=0; j<List_YOUSO[23]; j++){
    EIGEN[j] = (double*)malloc(sizeof(double)*(n+1));
  }

  H = (dcomplex***)malloc(sizeof(dcomplex**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    H[i] = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
    for (j=0; j<n+1; j++){
      H[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
    }
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
  for (i=0; i<n+1; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
  }

  M1 = (double*)malloc(sizeof(double)*(n+1));

  C = (dcomplex***)malloc(sizeof(dcomplex**)*List_YOUSO[23]);
  for (i=0; i<List_YOUSO[23]; i++){
    C[i] = (dcomplex**)malloc(sizeof(dcomplex*)*(n+1));
    for (j=0; j<n+1; j++){
      C[i][j] = (dcomplex*)malloc(sizeof(dcomplex)*(n+1));
    }
  }

  /*****************************************************
        allocation of arrays for parallelization 
  *****************************************************/

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  /* find size_H1 */
  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /* allocation of S1 and H1 */
  S1 = (double*)malloc(sizeof(double)*size_H1);
  H1 = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    H1[spin] = (double*)malloc(sizeof(double)*size_H1);
  }

  /* Get S1 */
  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SpinP_switch==0){ 
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1[0], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, nh[0], H1[0], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, nh[1], H1[1], MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  dtime(&SiloopTime);

  /*****************************************************
         Solve eigenvalue problem at each k-point
  *****************************************************/

  kj_e=(double*)malloc(sizeof(double)*4);
  iskj_e=(int*)malloc(sizeof(int)*4);
  K=(double*)malloc(sizeof(double)*3);
  K2=(double*)malloc(sizeof(double)*3);
  r=(double*)malloc(sizeof(double)*3);
  r0=(double*)malloc(sizeof(double)*3);
  weight=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (i=0; i<atomnum; i++) weight[i]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[i]);
  kj_v=(dcomplex***)malloc(sizeof(dcomplex**)*4);
  for (i=0; i<4; i++) {
    kj_v[i]=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
    for (j=0; j<atomnum; j++) kj_v[i][j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
  }
  tmpelem=(dcomplex****)malloc(sizeof(dcomplex***)*atomnum);
  for (i=0; i<atomnum; i++) tmpelem[i]=(dcomplex***)malloc(sizeof(dcomplex**)*atomnum);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) {
    tmpelem[i][j]=(dcomplex**)malloc(sizeof(dcomplex*)*Norbperatom[i]);
    for (k=0; k<Norbperatom[i]; k++) {
      tmpelem[i][j][k]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
    }
  }

  Name_Angular[0][0] = "s          ";
  Name_Angular[1][0] = "px         ";
  Name_Angular[1][1] = "py         ";
  Name_Angular[1][2] = "pz         ";
  Name_Angular[2][0] = "d3z^2-r^2  ";
  Name_Angular[2][1] = "dx^2-y^2   ";
  Name_Angular[2][2] = "dxy        ";
  Name_Angular[2][3] = "dxz        ";
  Name_Angular[2][4] = "dyz        ";
  Name_Angular[3][0] = "f5z^2-3r^2 ";
  Name_Angular[3][1] = "f5xz^2-xr^2";
  Name_Angular[3][2] = "f5yz^2-yr^2";
  Name_Angular[3][3] = "fzx^2-zy^2 ";
  Name_Angular[3][4] = "fxyz       ";
  Name_Angular[3][5] = "fx^3-3*xy^2";
  Name_Angular[3][6] = "f3yx^2-y^3 ";
  Name_Angular[4][0] = "g1         ";
  Name_Angular[4][1] = "g2         ";
  Name_Angular[4][2] = "g3         ";
  Name_Angular[4][3] = "g4         ";
  Name_Angular[4][4] = "g5         ";
  Name_Angular[4][5] = "g6         ";
  Name_Angular[4][6] = "g7         ";
  Name_Angular[4][7] = "g8         ";
  Name_Angular[4][8] = "g9         ";

  Name_Multiple[0] = "0";
  Name_Multiple[1] = "1";
  Name_Multiple[2] = "2";
  Name_Multiple[3] = "3";
  Name_Multiple[4] = "4";
  Name_Multiple[5] = "5";

  if (myid==Host_ID){
    strcpy(file_EV,".EV");
    fnjoint(filepath,filename,file_EV);
    if ((fp_EV = fopen(file_EV,"a")) != NULL){
      fprintf(fp_EV,"\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"          Unfolding calculation for band structure         \n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"                                                                          \n");
      fprintf(fp_EV," Origin of the Reference cell is set to (%f %f %f) (Bohr).\n\n",
              unfold_origin[0],unfold_origin[1],unfold_origin[2]);
      fprintf(fp_EV," Unfolded weights at specified k points are stored in System.Name.unfold_totup(dn).\n");
      fprintf(fp_EV," Individual orbital weights are stored in System.Name.unfold_orbup(dn).\n");
      fprintf(fp_EV," The format is: k_dis(Bohr^{-1})  energy(eV)  weight.\n\n");
      fprintf(fp_EV," The sequence for the orbital weights in System.Name.unfold_orbup(dn) is given below.\n\n");

      i1 = 1;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (l=0; l<=Supported_MaxL; l++){
	  for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
	    for (m=0; m<(2*l+1); m++){
	      fprintf(fp_EV,"  %4d ",i1);
	      if (l==0 && mul==0 && m==0)
		fprintf(fp_EV,"%4d %3s %s %s",
			Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
	      else
		fprintf(fp_EV,"         %s %s",
			Name_Multiple[mul],Name_Angular[l][m]);
	      fprintf(fp_EV,"\n");
	      i1++;
	    }
	  }
	}
      }

      fprintf(fp_EV,"\n"); 
      fprintf(fp_EV,"\n  The total number of calculated k points is %i.\n",totnkpts);
      fprintf(fp_EV,"  The number of calculated k points on each path is \n");
 
      fprintf(fp_EV,"  For each path: ("); 
      for (i=0; i<nkpoint; i++){
        fprintf(fp_EV," %i",np[i]); 
      }
      fprintf(fp_EV," )\n\n");

      fprintf(fp_EV,"                 ka         kb         kc\n");

      kloop = 0;
      for (i=0; i<nkpoint; i++){
	for (j=0; j<np[i]; j++) {

          kloop++;

	  if (np[i]==1) {
	    fprintf(fp_EV,"  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,kpoint[i][1],kpoint[i][2],kpoint[i][3]);
	  } 
          else {
	    fprintf(fp_EV,"  %3d/%3d   %10.6f %10.6f %10.6f\n",
                    kloop,totnkpts,
                    kpoint[i][1]+j*(kpoint[i+1][1]-kpoint[i][1])/np[i],
		    kpoint[i][2]+j*(kpoint[i+1][2]-kpoint[i][2])/np[i],
		    kpoint[i][3]+j*(kpoint[i+1][3]-kpoint[i][3])/np[i]);
	  }
	}
      }

      fprintf(fp_EV,"\n");
      fclose(fp_EV);

    }
    else{
      printf("Failure of saving the EV file.\n");
      fclose(fp_EV);
    }

    if (SpinP_switch==0) {

      strcpy(file_EV,".unfold_totup");
      fnjoint(filepath,filename,file_EV);

      fp_EV = fopen(file_EV,"w");
      if (fp_EV == NULL) {
	printf("Failure of saving the System.Name.unfold_totup file.\n");
	fclose(fp_EV);
      }

      strcpy(file_EV,".unfold_orbup");
      fnjoint(filepath,filename,file_EV);
      fp_EV1 = fopen(file_EV,"w");

      if (fp_EV1 == NULL) {
	printf("Failure of saving the System.Name.unfold_orbup file.\n");
	fclose(fp_EV1);
      }

    } 
    else if (SpinP_switch==1) {
      strcpy(file_EV,".unfold_totup");
      fnjoint(filepath,filename,file_EV);
      fp_EV = fopen(file_EV,"w");
      if (fp_EV == NULL) {
	printf("Failure of saving the System.Name.unfold_totup file.\n");
	fclose(fp_EV);
      }
      strcpy(file_EV,".unfold_orbup");
      fnjoint(filepath,filename,file_EV);
      fp_EV1 = fopen(file_EV,"w");
      if (fp_EV1 == NULL) {
	printf("Failure of saving the System.Name.unfold_orbup file.\n");
	fclose(fp_EV1);
      }
      strcpy(file_EV,".unfold_totdn");
      fnjoint(filepath,filename,file_EV);
      fp_EV2 = fopen(file_EV,"w");
      if (fp_EV2 == NULL) {
	printf("Failure of saving the System.Name.unfold_totdn file.\n");
	fclose(fp_EV2);
      }
      strcpy(file_EV,".unfold_orbdn");
      fnjoint(filepath,filename,file_EV);
      fp_EV3 = fopen(file_EV,"w");
      if (fp_EV3 == NULL) {
	printf("Failure of saving the System.Name.unfold_orbdn file.\n");
	fclose(fp_EV3);
      }
    }
  }

  int kloopi,kloopj;
  double kpt0,kpt1,kpt2;

  /* for gnuplot example */

  if (myid==Host_ID){
    strcpy(file_EV,".unfold_plotexample");
    fnjoint(filepath,filename,file_EV);
    fp_EV0 = fopen(file_EV,"w");
    if (fp_EV0 == NULL) {
      printf("Failure of saving the System.Name.unfold_plotexample file.\n");
      fclose(fp_EV0);
    }

    fprintf(fp_EV0,"set yrange [%f:%f]\n",unfold_lbound*eV2Hartree,unfold_ubound*eV2Hartree); 
    fprintf(fp_EV0,"set ylabel 'Energy (eV)'\n");
    fprintf(fp_EV0,"set xtics(");

    pk1 =  coe*kpoint[0][1]*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
          -coe*kpoint[0][2]*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
          +coe*kpoint[0][3]*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
    pk2 = -coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
          +coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
          -coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
    pk3 =  coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
          -coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
          +coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);

    kdis=0.;
    for (kloopi=0; kloopi<nkpoint; kloopi++) {

      kpt0=kpoint[kloopi][1];
      kpt1=kpoint[kloopi][2];
      kpt2=kpoint[kloopi][3];

      k1 =  coe*kpt0*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
           -coe*kpt1*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
           +coe*kpt2*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
      k2 = -coe*kpt0*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
           +coe*kpt1*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
           -coe*kpt2*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
      k3 = coe*kpt0*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
          -coe*kpt1*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
          +coe*kpt2*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);

      K[0]=k1*rtv[1][1]+k2*rtv[2][1]+k3*rtv[3][1];
      K[1]=k1*rtv[1][2]+k2*rtv[2][2]+k3*rtv[3][2];
      K[2]=k1*rtv[1][3]+k2*rtv[2][3]+k3*rtv[3][3];
      K2[0]=pk1*rtv[1][1]+pk2*rtv[2][1]+pk3*rtv[3][1];
      K2[1]=pk1*rtv[1][2]+pk2*rtv[2][2]+pk3*rtv[3][2];
      K2[2]=pk1*rtv[1][3]+pk2*rtv[2][3]+pk3*rtv[3][3];
      dis2pk=distwovec(K,K2);
      kdis+=dis2pk;

      if (kloopi==nkpoint-1) 
        fprintf(fp_EV0,"'%s' %f)\n",unfold_kpoint_name[kloopi],kdis); 
      else 
        fprintf(fp_EV0,"'%s' %f,",unfold_kpoint_name[kloopi],kdis);

      pk1=k1;
      pk2=k2;
      pk3=k3;
    }

    pk1=coe*kpoint[0][1]*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
      -coe*kpoint[0][2]*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
      +coe*kpoint[0][3]*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
    pk2=-coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
      +coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
      -coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
    pk3=coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
      -coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
      +coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);
    fprintf(fp_EV0,"set xrange [0:%f]\n",kdis);
    fprintf(fp_EV0,"set arrow nohead from 0,0 to %f,0\n",kdis);
    kdis=0.;
    for (kloopi=1; kloopi<nkpoint-1; kloopi++) {
      fprintf(fp_EV0,"set arrow nohead from ");
      kpt0=kpoint[kloopi][1];
      kpt1=kpoint[kloopi][2];
      kpt2=kpoint[kloopi][3];
      k1= coe*kpt0*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
        -coe*kpt1*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
        +coe*kpt2*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
      k2=-coe*kpt0*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
        +coe*kpt1*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
        -coe*kpt2*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
      k3= coe*kpt0*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
        -coe*kpt1*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
        +coe*kpt2*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);
      K[0]=k1*rtv[1][1]+k2*rtv[2][1]+k3*rtv[3][1];
      K[1]=k1*rtv[1][2]+k2*rtv[2][2]+k3*rtv[3][2];
      K[2]=k1*rtv[1][3]+k2*rtv[2][3]+k3*rtv[3][3];
      K2[0]=pk1*rtv[1][1]+pk2*rtv[2][1]+pk3*rtv[3][1];
      K2[1]=pk1*rtv[1][2]+pk2*rtv[2][2]+pk3*rtv[3][2];
      K2[2]=pk1*rtv[1][3]+pk2*rtv[2][3]+pk3*rtv[3][3];
      dis2pk=distwovec(K,K2);
      kdis+=dis2pk;
      fprintf(fp_EV0,"%f,%f to %f,%f\n",kdis,unfold_lbound*eV2Hartree,kdis,unfold_ubound*eV2Hartree);
      pk1=k1;
      pk2=k2;
      pk3=k3;
    }
    fprintf(fp_EV0,"set style circle radius 0\n");
    fprintf(fp_EV0,"plot '%s.unfold_totup' using 1:2:($3)*0.05 notitle with circles lc rgb 'red'\n",filename);
  }

  /* end gnuplot example */

  pk1=coe*kpoint[0][1]*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
    -coe*kpoint[0][2]*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
    +coe*kpoint[0][3]*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
  pk2=-coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
    +coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
    -coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
  pk3=coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
    -coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
    +coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);

  /* for standard output */

  if (myid==Host_ID && 0<level_stdout) {
    printf(" The number of selected k points is %i.\n",totnkpts);

    printf(" For each path: (");
    for (i=0; i<nkpoint; i++){
      printf(" %i",np[i]); 
    }
    printf(" )\n\n");
    printf("                 ka         kb         kc\n");
  }

  /*********************************************
                      kloopi 
  *********************************************/

  kdis=0.;
  kloop = 0;
  for (kloopi=0; kloopi<nkpoint; kloopi++)
    for (kloopj=0; kloopj<np[kloopi]; kloopj++) {

      kloop++;

      if (np[kloopi]==1) {
	kpt0=kpoint[kloopi][1];
	kpt1=kpoint[kloopi][2];
	kpt2=kpoint[kloopi][3];
      } else {
	kpt0=kpoint[kloopi][1]+kloopj*(kpoint[kloopi+1][1]-kpoint[kloopi][1])/np[kloopi];
	kpt1=kpoint[kloopi][2]+kloopj*(kpoint[kloopi+1][2]-kpoint[kloopi][2])/np[kloopi];
	kpt2=kpoint[kloopi][3]+kloopj*(kpoint[kloopi+1][3]-kpoint[kloopi][3])/np[kloopi];
      }
    
      /* for standard output */
     
      if (myid==Host_ID && 0<level_stdout) {

        if (kloop==totnkpts)
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n\n",kloop,totnkpts,kpt0,kpt1,kpt2);
        else 
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,kpt0,kpt1,kpt2);
      }

      k1= coe*kpt0*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
        -coe*kpt1*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
        +coe*kpt2*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
      k2=-coe*kpt0*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
        +coe*kpt1*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
        -coe*kpt2*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
      k3= coe*kpt0*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
        -coe*kpt1*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
        +coe*kpt2*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);

      /* make S */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  S[i1][j1] = Complex(0.0,0.0);
	} 
      } 

      k = 0;
      for (AN=1; AN<=atomnum; AN++){
	GA_AN = order_GA[AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];

	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  Rn = ncn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];

	  l1 = atv_ijk[Rn][1];
	  l2 = atv_ijk[Rn][2];
	  l3 = atv_ijk[Rn][3];
	  kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	  si = sin(2.0*PI*kRn);
	  co = cos(2.0*PI*kRn);

	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){

	      S[Anum+i][Bnum+j].r += S1[k]*co;
	      S[Anum+i][Bnum+j].i += S1[k]*si;

	      k++;
	    }
	  }
	}
      }

      /* diagonalization of S */
      Eigen_PHH(mpi_comm_level1,S,koS,n,n,1);

      if (3<=level_stdout){
	printf("  kloop %i, k1 k2 k3 %10.6f %10.6f %10.6f\n",kloop,k1,k2,k3);
	for (i1=1; i1<=n; i1++){
	  printf("  Eigenvalues of OLP  %2d  %15.12f\n",i1,koS[i1]);
	}
      }

      /* minus eigenvalues to 1.0e-14 */

      for (l=1; l<=n; l++){
	if (koS[l]<0.0) koS[l] = 1.0e-14;
      }

      /* calculate S*1/sqrt(koS) */

      for (l=1; l<=n; l++) M1[l] = 1.0/sqrt(koS[l]);

      /* S * M1  */

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  S[i1][j1].r = S[i1][j1].r*M1[j1];
	  S[i1][j1].i = S[i1][j1].i*M1[j1];
	} 
      } 

      /* loop for spin */

      for (spin=0; spin<=SpinP_switch; spin++){

	/* make H */

	for (i1=1; i1<=n; i1++){
	  for (j1=1; j1<=n; j1++){
	    H[spin][i1][j1] = Complex(0.0,0.0);
	  } 
	} 

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];

	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	    GB_AN = natn[GA_AN][LB_AN];
	    Rn = ncn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    l1 = atv_ijk[Rn][1];
	    l2 = atv_ijk[Rn][2];
	    l3 = atv_ijk[Rn][3];
	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

	    for (i=0; i<tnoA; i++){

	      for (j=0; j<tnoB; j++){

		H[spin][Anum+i][Bnum+j].r += H1[spin][k]*co;
		H[spin][Anum+i][Bnum+j].i += H1[spin][k]*si;

		k++;

	      }
	    }
	  }
	}

	/* first transpose of S */

	for (i1=1; i1<=n; i1++){
	  for (j1=i1+1; j1<=n; j1++){
	    Ctmp1 = S[i1][j1];
	    Ctmp2 = S[j1][i1];
	    S[i1][j1] = Ctmp2;
	    S[j1][i1] = Ctmp1;
	  }
	}

	/****************************************************
                      M1 * U^t * H * U * M1
	****************************************************/

	/* H * U * M1 */

#pragma omp parallel for shared(spin,n,myid,is1,ie1,S,H,C) private(i1,j1,l) 

	for (j1=is1[myid]; j1<=ie1[myid]; j1++){

	  for (i1=1; i1<=(n-1); i1+=2){

	    double sum0  = 0.0, sum1  = 0.0;
	    double sumi0 = 0.0, sumi1 = 0.0;

	    for (l=1; l<=n; l++){
	      sum0  += H[spin][i1+0][l].r*S[j1][l].r - H[spin][i1+0][l].i*S[j1][l].i;
	      sum1  += H[spin][i1+1][l].r*S[j1][l].r - H[spin][i1+1][l].i*S[j1][l].i;

	      sumi0 += H[spin][i1+0][l].r*S[j1][l].i + H[spin][i1+0][l].i*S[j1][l].r;
	      sumi1 += H[spin][i1+1][l].r*S[j1][l].i + H[spin][i1+1][l].i*S[j1][l].r;
	    }

	    C[spin][j1][i1+0].r = sum0;
	    C[spin][j1][i1+1].r = sum1;

	    C[spin][j1][i1+0].i = sumi0;
	    C[spin][j1][i1+1].i = sumi1;
	  }

	  for (; i1<=n; i1++){

	    double sum  = 0.0;
	    double sumi = 0.0;

	    for (l=1; l<=n; l++){
	      sum  += H[spin][i1][l].r*S[j1][l].r - H[spin][i1][l].i*S[j1][l].i;
	      sumi += H[spin][i1][l].r*S[j1][l].i + H[spin][i1][l].i*S[j1][l].r;
	    }

	    C[spin][j1][i1].r = sum;
	    C[spin][j1][i1].i = sumi;
	  }

	} /* i1 */ 

	/* M1 * U^+ H * U * M1 */

#pragma omp parallel for shared(spin,n,is1,ie1,myid,S,H,C) private(i1,j1,l)  

	for (i1=1; i1<=n; i1++){
	  for (j1=is1[myid]; j1<=ie1[myid]; j1++){
  
	    double sum  = 0.0;
	    double sumi = 0.0;

	    for (l=1; l<=n; l++){
	      sum  +=  S[i1][l].r*C[spin][j1][l].r + S[i1][l].i*C[spin][j1][l].i;
	      sumi +=  S[i1][l].r*C[spin][j1][l].i - S[i1][l].i*C[spin][j1][l].r;
	    }

	    H[spin][j1][i1].r = sum;
	    H[spin][j1][i1].i = sumi;

	  }
	} 

	/* broadcast H */

	BroadCast_ComplexMatrix(mpi_comm_level1,H[spin],n,is1,ie1,myid,numprocs,
				stat_send,request_send,request_recv);

	/* H to C */

	for (i1=1; i1<=n; i1++){
	  for (j1=1; j1<=n; j1++){
	    C[spin][j1][i1] = H[spin][i1][j1];
	  }
	}

	/* penalty for ill-conditioning states */

	EV_cut0 = 1.0e-9;

	for (i1=1; i1<=n; i1++){

	  if (koS[i1]<EV_cut0){
	    C[spin][i1][i1].r += pow((koS[i1]/EV_cut0),-2.0) - 1.0;
	  }
 
	  /* cutoff the interaction between the ill-conditioned state */
 
	  if (1.0e+3<C[spin][i1][i1].r){
	    for (j1=1; j1<=n; j1++){
	      C[spin][i1][j1] = Complex(0.0,0.0);
	      C[spin][j1][i1] = Complex(0.0,0.0);
	    }
	    C[spin][i1][i1].r = 1.0e+4;
	  }
	}

	/* diagonalization of C */
	Eigen_PHH(mpi_comm_level1,C[spin],ko[spin],n,n,0);

	for (i1=1; i1<=n; i1++){
	  EIGEN[spin][i1] = ko[spin][i1];
	}

	/****************************************************
          transformation to the original eigenvectors.
                 NOTE JRCAT-244p and JAIST-2122p 
	****************************************************/

	/*  The H matrix is distributed by row */

	for (i1=1; i1<=n; i1++){
	  for (j1=is1[myid]; j1<=ie1[myid]; j1++){
	    H[spin][j1][i1] = C[spin][i1][j1];
	  }
	}

	/* second transpose of S */

	for (i1=1; i1<=n; i1++){
	  for (j1=i1+1; j1<=n; j1++){
	    Ctmp1 = S[i1][j1];
	    Ctmp2 = S[j1][i1];
	    S[i1][j1] = Ctmp2;
	    S[j1][i1] = Ctmp1;
	  }
	}

	/* C is distributed by row in each processor */

#pragma omp parallel for shared(spin,n,is1,ie1,myid,S,H,C) private(i1,j1,l,sum,sumi)  

	for (j1=is1[myid]; j1<=ie1[myid]; j1++){
	  for (i1=1; i1<=n; i1++){

	    sum  = 0.0;
	    sumi = 0.0;
	    for (l=1; l<=n; l++){
	      sum  += S[i1][l].r*H[spin][j1][l].r - S[i1][l].i*H[spin][j1][l].i;
	      sumi += S[i1][l].r*H[spin][j1][l].i + S[i1][l].i*H[spin][j1][l].r;
	    }

	    C[spin][j1][i1].r = sum;
	    C[spin][j1][i1].i = sumi;
	  }
	}

	/* broadcast C:
	   C is distributed by row in each processor
	*/

	BroadCast_ComplexMatrix(mpi_comm_level1,C[spin],n,is1,ie1,myid,numprocs,
				stat_send,request_send,request_recv);

      } /* spin */

      /****************************************************
                          Output
      ****************************************************/

      if (myid==Host_ID){

        setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	K[0]=k1*rtv[1][1]+k2*rtv[2][1]+k3*rtv[3][1];
	K[1]=k1*rtv[1][2]+k2*rtv[2][2]+k3*rtv[3][2];
	K[2]=k1*rtv[1][3]+k2*rtv[2][3]+k3*rtv[3][3];
	K2[0]=pk1*rtv[1][1]+pk2*rtv[2][1]+pk3*rtv[3][1];
	K2[1]=pk1*rtv[1][2]+pk2*rtv[2][2]+pk3*rtv[3][2];
	K2[2]=pk1*rtv[1][3]+pk2*rtv[2][3]+pk3*rtv[3][3];
	dis2pk=distwovec(K,K2);
	kdis+=dis2pk;

	num0 = 4;
	num1 = n/num0 + 1*(n%num0!=0);

	for (spin=0; spin<=SpinP_switch; spin++){
	  for (i=1; i<=num1; i++){

            countkj_e=0;
            for (j=0; j<4; j++) iskj_e[j]=0;

	    for (i1=-2; i1<=0; i1++){

	      for (j=1; j<=num0; j++){

		j1 = num0*(i-1) + j;

		if (j1<=n){ 

		  if (i1==-1){

                    if (((EIGEN[spin][j1]-ChemP)<=unfold_ubound)&&((EIGEN[spin][j1]-ChemP)>=unfold_lbound)) {
                      kj_e[countkj_e]=(EIGEN[spin][j1]-ChemP);
                      iskj_e[countkj_e]=1; }
                    countkj_e++;

		  }

		}
	      }
	    }

	    i1 = 1; 
            int iorb;

	    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
              iorb=0;

	      wan1 = WhatSpecies[Gc_AN];
            
	      for (l=0; l<=Supported_MaxL; l++){
		for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		  for (m=0; m<(2*l+1); m++){

                    countkj_e = 0;

		    for (j=1; j<=num0; j++){

		      j1 = num0*(i-1) + j;

		      if (0<i1 && j1<=n){
			kj_v[countkj_e++][Gc_AN-1][iorb]=Complex(C[spin][j1][i1].r,C[spin][j1][i1].i);
		      }
		    }

		    i1++;
                    iorb++;
		  }
		}
	      }
	    }

            for (countkj_e=0; countkj_e<4; countkj_e++) if (iskj_e[countkj_e]==1) {

	      for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) weight[k][j]=Complex(0.,0.);

	      for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) for (l=0; l<Norbperatom[j]; l++)
		for (m=0; m<Norbperatom[k]; m++) tmpelem[j][k][l][m]=Cmul(Conjg(kj_v[countkj_e][j][l]),kj_v[countkj_e][k][m]);

	      int NA,ir,MA,MO,NO;
	      dcomplex dtmp;
	      for (NA=0; NA<atomnum; NA++) {
		int n=unfold_mapN2n[NA]; int r0x=tabr4RN[0][NA][0]; int r0y=tabr4RN[0][NA][1]; int r0z=tabr4RN[0][NA][2];
		r0[0]=r0x*a[0]+r0y*b[0]+r0z*c[0];
		r0[1]=r0x*a[1]+r0y*b[1]+r0z*c[1];
		r0[2]=r0x*a[2]+r0y*b[2]+r0z*c[2];
		dcomplex phase1=Cexp(Complex(0.,-dot(K,r0)));
		for (ir=0; ir<nr; ir++) {
		  if (rnmap[ir][n][1]==-1) continue;
		  r[0]=rlist[ir][0]*a[0]+rlist[ir][1]*b[0]+rlist[ir][2]*c[0];
		  r[1]=rlist[ir][0]*a[1]+rlist[ir][1]*b[1]+rlist[ir][2]*c[1];
		  r[2]=rlist[ir][0]*a[2]+rlist[ir][1]*b[2]+rlist[ir][2]*c[2];
		  dcomplex phase2=Cmul(phase1,Cexp(Complex(0.,dot(K,r))));

		  for (MA=0; MA<atomnum; MA++) {
		    if (Elem[rnmap[ir][n][0]][MA][rnmap[ir][n][1]][0][0]<-99999.) continue;
		    for (MO=0; MO<Norbperatom[MA]; MO++) for (NO=0; NO<Norbperatom[NA]; NO++) {
		      dtmp=RCmul(Elem[rnmap[ir][n][0]][MA][rnmap[ir][n][1]][MO][NO],tmpelem[MA][NA][MO][NO]);
		      weight[NA][NO]=Cadd(weight[NA][NO],Cmul(phase2,dtmp));
		    }}}}

	      double sumallorb=0.;

	      for (j=0; j<atomnum; j++){
                for (k=0; k<Norbperatom[j]; k++){
                   sumallorb += weight[j][k].r;
		}
	      }

	      if (spin==0) 
                fprintf(fp_EV,"%f %f %10.7f\n",kdis,kj_e[countkj_e]*eV2Hartree,fabs(sumallorb)/coe);
	      else 
                fprintf(fp_EV2,"%f %f %10.7f\n",kdis,kj_e[countkj_e]*eV2Hartree,fabs(sumallorb)/coe);

	      /* set negative weight to zero for plotting purpose */
	      for (j=0; j<atomnum; j++){
                for (k=0; k<Norbperatom[j]; k++){
                   if (weight[j][k].r<0.0) weight[j][k].r=0.0;
		}
	      }

	      if (spin==0) fprintf(fp_EV1,"%f %f ",kdis,kj_e[countkj_e]*eV2Hartree);
	      else fprintf(fp_EV3,"%f %f ",kdis,kj_e[countkj_e]*eV2Hartree);
	      for (j=0; j<atomnum; j++) {
		if (spin==0) for (k=0; k<Norbperatom[j]; k++) fprintf(fp_EV1,"%e ",weight[j][k].r/coe);
		else for (k=0; k<Norbperatom[j]; k++) fprintf(fp_EV3,"%e ",weight[j][k].r/coe);
	      }
	      if (spin==0) fprintf(fp_EV1,"\n");
	      else fprintf(fp_EV3,"\n");
	    }
	  }
	}

      } /* if (myid==Host_ID) */
   
      pk1=k1;
      pk2=k2;
      pk3=k3;
    }  /* kloop */

  if (myid==Host_ID){
    if (fp_EV0 != NULL) fclose(fp_EV0);
    if (fp_EV != NULL) fclose(fp_EV);
    if (fp_EV1 != NULL) fclose(fp_EV1);
    if ((SpinP_switch==1)&&(fp_EV2 != NULL)) fclose(fp_EV2);
    if ((SpinP_switch==1)&&(fp_EV3 != NULL)) fclose(fp_EV3);
  }

  /****************************************************
                       free arrays
  ****************************************************/

  free(K);
  free(K2);
  free(r);
  free(r0);
  free(kj_e);
  free(iskj_e);
  free(unfold_mapN2n);
  free(a);
  free(b);
  free(c);
  free(unfold_origin);
  free(np);
  for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
  for (i=0; i<4; i++) for (j=0; j<atomnum; j++) free(kj_v[i][j]);
  for (i=0; i<4; i++) free(kj_v[i]); free(kj_v);
  for (i=0; i<atomnum; i++) free(weight[i]); free(weight);
  for (i=0; i<NR; i++) free(Rlist[i]); free(Rlist);
  for (i=0; i<nr; i++) free(rlist[i]); free(rlist);
  for (i=0; i<21; i++) for (j=0; j<21; j++) free(_MapR[i][j]);
  for (i=0; i<21; i++) free(_MapR[i]); free(_MapR);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) free(tabr4RN[i][j]);
  for (i=0; i<NR; i++) free(tabr4RN[i]); free(tabr4RN);
  for (i=0; i<nr; i++) for (j=0; j<natom; j++) free(rnmap[i][j]);
  for (i=0; i<nr; i++) free(rnmap[i]); free(rnmap);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) for (l=0; l<Norbperatom[j]; l++) free(Elem[i][j][k][l]);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) free(Elem[i][j][k]);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) free(Elem[i][j]);
  for (i=0; i<NR; i++) free(Elem[i]); free(Elem);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) for (k=0; k<Norbperatom[i]; k++)
    free(tmpelem[i][j][k]);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) free(tmpelem[i][j]);
  for (i=0; i<atomnum; i++) free(tmpelem[i]); free(tmpelem);
  for (i=0; i<3; i++) free(fracabc[i]); free(fracabc);
  free(Norbperatom);
  for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
  for (i=0; i<(unfold_Nkpoint+1); i++){
    free(unfold_kpoint_name[i]);
  }
  free(unfold_kpoint_name);

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);

  free(MP);
  free(order_GA);
  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  for (i=0; i<List_YOUSO[23]; i++){
    free(ko[i]);
  }
  free(ko);

  free(koS);

  for (j=0; j<List_YOUSO[23]; j++){
    free(EIGEN[j]);
  }
  free(EIGEN);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n+1; j++){
      free(H[i][j]);
    }
    free(H[i]);
  }
  free(H);  

  for (i=0; i<n+1; i++){
    free(S[i]);
  }
  free(S);

  free(M1);

  for (i=0; i<List_YOUSO[23]; i++){
    for (j=0; j<n+1; j++){
      free(C[i][j]);
    }
    free(C[i]);
  }
  free(C);

  free(S1);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(H1[spin]);
  }
  free(H1);

  dtime(&TEtime);
}




static void Unfolding_Bands_NonCol(
				   int nkpoint, double **kpoint,
				   int SpinP_switch, 
				   double *****nh,
				   double *****ImNL,
				   double ****CntOLP)
{
  double coe;
  double* a;
  double* b;
  double* c;
  double* K;
  double* K2;
  double* r;
  double* r0;
  double* kj_e;
  int* iskj_e;
  int countkj_e;
  double pk1,pk2,pk3;
  double dis2pk;
  double kdis;
  dcomplex** weight;
  dcomplex** weight1;
  dcomplex*** kj_v;
  dcomplex*** kj_v1;
  dcomplex**** tmpelem;
  dcomplex**** tmpelem1;
  double **fracabc;

  int i,j,k,l,n,wan,m,ii1,jj1,jj2,n2;
  int *MP;
  int *order_GA;
  int *My_NZeros;
  int *SP_NZeros;
  int *SP_Atoms;
  int i1,j1,po,spin,n1,size_H1;
  int num2,RnB,l1,l2,l3,kloop,AN,Rn;
  int ct_AN,h_AN,wanA,tnoA,wanB,tnoB;
  int GA_AN,Anum;
  int ii,ij,ik,MaxN;
  int wan1,mul,Gc_AN,num0,num1;
  int LB_AN,GB_AN,Bnum;

  double time0,tmp,tmp1,av_num;
  double snum_i,snum_j,snum_k,k1,k2,k3,sum,sumi,Num_State,FermiF;
  double x,Dnum,Dnum2,AcP,ChemP_MAX,ChemP_MIN;
  double *S1;
  double *RH0;
  double *RH1;
  double *RH2;
  double *RH3;
  double *IH0;
  double *IH1;
  double *IH2;
  double *ko,*M1,*EIGEN;
  double *koS;
  double EV_cut0;
  dcomplex **H,**S,**C;
  dcomplex Ctmp1,Ctmp2;
  double **Ctmp;
  double u2,v2,uv,vu;
  double dum,sumE,kRn,si,co;
  double Resum,ResumE,Redum,Redum2,Imdum;
  double TStime,TEtime,SiloopTime,EiloopTime;
  double FermiEps = 1.0e-14;
  double x_cut = 30.0;
  double OLP_eigen_cut=Threshold_OLP_Eigen;

  char *Name_Angular[Supported_MaxL+1][2*(Supported_MaxL+1)+1];
  char *Name_Multiple[20];
  char file_EV[YOUSO10];
  FILE *fp_EV0;
  FILE *fp_EV;
  FILE *fp_EV1;
  char buf[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID;
  int OMPID,Nthrds,Nprocs;
  int *is1,*ie1;
  int *is2,*ie2;
  int *is12,*ie12;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  if (myid==Host_ID && 0<level_stdout) {
    printf("\n*******************************************************\n");
    printf("                 Unfolding of Bands \n");
    printf("*******************************************************\n\n");fflush(stdout);
  } 

  dtime(&TStime);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n + 2;

  /****************************************************
   Allocation
  ****************************************************/

  getnorbperatom();
  exitcode=0;
  buildMapRlist();
  if (exitcode==1) {
    for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
    free(unfold_origin);
    free(unfold_mapN2n);
    for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
    return;
  }

  a = (double*)malloc(sizeof(double)*3);
  b = (double*)malloc(sizeof(double)*3);
  c = (double*)malloc(sizeof(double)*3);
  a[0]=unfold_abc[0][0];
  a[1]=unfold_abc[0][1];
  a[2]=unfold_abc[0][2];
  b[0]=unfold_abc[1][0];
  b[1]=unfold_abc[1][1];
  b[2]=unfold_abc[1][2];
  c[0]=unfold_abc[2][0];
  c[1]=unfold_abc[2][1];
  c[2]=unfold_abc[2][2];
  buildtabr4RN(a,b,c,unfold_origin,unfold_mapN2n);
  if (exitcode==1) {
    for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
    free(unfold_origin);
    free(unfold_mapN2n);
    for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
    free(a);
    free(b);
    free(c);
    return;
  }

  if (myid==Host_ID && 1<level_stdout) {
    printf("Reference origin is set to (%f %f %f) (Bohr)\n",unfold_origin[0],unfold_origin[1],unfold_origin[2]);
    printf("Supercell_lattice_vector atom Reference_lattice_vector atom\n");
    for (i=0; i<NR; i++) for (j=0; j<atomnum; j++)
      printf("(%i %i %i) %i <==> (%i %i %i) %i\n",Rlist[i][0],Rlist[i][1],Rlist[i][2],j+1,tabr4RN[i][j][0],tabr4RN[i][j][1],tabr4RN[i][j][2],unfold_mapN2n[j]);
    printf("\n");fflush(stdout);
  }

  coe=Cell_Volume/volume(a,b,c);
  fracabc=(double**)malloc(sizeof(double*)*3);
  for (i=0; i<3; i++) fracabc[i]=(double*)malloc(sizeof(double)*3);
  abc_by_ABC(fracabc);
  determine_kpts(nkpoint,kpoint);

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);
  order_GA = (int*)malloc(sizeof(int)*(List_YOUSO[1]+1));

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_NZeros = (int*)malloc(sizeof(int)*numprocs);
  SP_Atoms = (int*)malloc(sizeof(int)*numprocs);

  ko = (double*)malloc(sizeof(double)*n2);
  koS = (double*)malloc(sizeof(double)*(n+1));

  EIGEN = (double*)malloc(sizeof(double)*n2);

  H = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    H[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  S = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  M1 = (double*)malloc(sizeof(double)*n2);

  C = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (j=0; j<n2; j++){
    C[j] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  Ctmp = (double**)malloc(sizeof(double*)*n2);
  for (j=0; j<n2; j++){
    Ctmp[j] = (double*)malloc(sizeof(double)*n2);
  }

  /*****************************************************
        allocation of arrays for parallelization 
  *****************************************************/

  stat_send = malloc(sizeof(MPI_Status)*numprocs);
  request_send = malloc(sizeof(MPI_Request)*numprocs);
  request_recv = malloc(sizeof(MPI_Request)*numprocs);

  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  is12 = (int*)malloc(sizeof(int)*numprocs);
  ie12 = (int*)malloc(sizeof(int)*numprocs);

  is2 = (int*)malloc(sizeof(int)*numprocs);
  ie2 = (int*)malloc(sizeof(int)*numprocs);

  if ( numprocs<=n ){

    av_num = (double)n/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is1[ID] = (int)(av_num*(double)ID) + 1; 
      ie1[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is1[0] = 1;
    ie1[numprocs-1] = n; 

  }

  else{

    for (ID=0; ID<n; ID++){
      is1[ID] = ID + 1; 
      ie1[ID] = ID + 1;
    }
    for (ID=n; ID<numprocs; ID++){
      is1[ID] =  1;
      ie1[ID] = -2;
    }
  }

  for (ID=0; ID<numprocs; ID++){
    is12[ID] = 2*is1[ID] - 1;
    ie12[ID] = 2*ie1[ID];
  }

  /* make is2 and ie2 */ 

  MaxN = 2*n;

  if ( numprocs<=MaxN ){

    av_num = (double)MaxN/(double)numprocs;

    for (ID=0; ID<numprocs; ID++){
      is2[ID] = (int)(av_num*(double)ID) + 1; 
      ie2[ID] = (int)(av_num*(double)(ID+1)); 
    }

    is2[0] = 1;
    ie2[numprocs-1] = MaxN; 
  }

  else{
    for (ID=0; ID<MaxN; ID++){
      is2[ID] = ID + 1; 
      ie2[ID] = ID + 1;
    }
    for (ID=MaxN; ID<numprocs; ID++){
      is2[ID] =  1;
      ie2[ID] = -2;
    }
  }

  /* find size_H1 */
  size_H1 = Get_OneD_HS_Col(0, CntOLP, &tmp, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  /* allocation of arrays */
  S1  = (double*)malloc(sizeof(double)*size_H1);
  RH0 = (double*)malloc(sizeof(double)*size_H1);
  RH1 = (double*)malloc(sizeof(double)*size_H1);
  RH2 = (double*)malloc(sizeof(double)*size_H1);
  RH3 = (double*)malloc(sizeof(double)*size_H1);
  IH0 = (double*)malloc(sizeof(double)*size_H1);
  IH1 = (double*)malloc(sizeof(double)*size_H1);
  IH2 = (double*)malloc(sizeof(double)*size_H1);

  /* set S1, RH0, RH1, RH2, RH3, IH0, IH1, IH2 */

  size_H1 = Get_OneD_HS_Col(1, CntOLP, S1,  MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[0],  RH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[1],  RH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[2],  RH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  size_H1 = Get_OneD_HS_Col(1, nh[3],  RH3, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);

  if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
      && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  
    
    /* nothing is done. */
  }
  else {
    size_H1 = Get_OneD_HS_Col(1, ImNL[0], IH0, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[1], IH1, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
    size_H1 = Get_OneD_HS_Col(1, ImNL[2], IH2, MP, order_GA, My_NZeros, SP_NZeros, SP_Atoms);
  }

  dtime(&SiloopTime);

  /*****************************************************
         Solve eigenvalue problem at each k-point
  *****************************************************/

  kj_e=(double*)malloc(sizeof(double)*2);
  iskj_e=(int*)malloc(sizeof(int)*2);
  K=(double*)malloc(sizeof(double)*3);
  K2=(double*)malloc(sizeof(double)*3);
  r=(double*)malloc(sizeof(double)*3);
  r0=(double*)malloc(sizeof(double)*3);
  weight=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (i=0; i<atomnum; i++) weight[i]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[i]);
  weight1=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
  for (i=0; i<atomnum; i++) weight1[i]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[i]);
  kj_v=(dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (i=0; i<2; i++) {
    kj_v[i]=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
    for (j=0; j<atomnum; j++) kj_v[i][j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
  }
  kj_v1=(dcomplex***)malloc(sizeof(dcomplex**)*2);
  for (i=0; i<2; i++) {
    kj_v1[i]=(dcomplex**)malloc(sizeof(dcomplex*)*atomnum);
    for (j=0; j<atomnum; j++) kj_v1[i][j]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
  }
  tmpelem=(dcomplex****)malloc(sizeof(dcomplex***)*atomnum);
  for (i=0; i<atomnum; i++) tmpelem[i]=(dcomplex***)malloc(sizeof(dcomplex**)*atomnum);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) {
    tmpelem[i][j]=(dcomplex**)malloc(sizeof(dcomplex*)*Norbperatom[i]);
    for (k=0; k<Norbperatom[i]; k++) {
      tmpelem[i][j][k]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
    }
  }
  tmpelem1=(dcomplex****)malloc(sizeof(dcomplex***)*atomnum);
  for (i=0; i<atomnum; i++) tmpelem1[i]=(dcomplex***)malloc(sizeof(dcomplex**)*atomnum);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) {
    tmpelem1[i][j]=(dcomplex**)malloc(sizeof(dcomplex*)*Norbperatom[i]);
    for (k=0; k<Norbperatom[i]; k++) {
      tmpelem1[i][j][k]=(dcomplex*)malloc(sizeof(dcomplex)*Norbperatom[j]);
    }
  }

  Name_Angular[0][0] = "s          ";
  Name_Angular[1][0] = "px         ";
  Name_Angular[1][1] = "py         ";
  Name_Angular[1][2] = "pz         ";
  Name_Angular[2][0] = "d3z^2-r^2  ";
  Name_Angular[2][1] = "dx^2-y^2   ";
  Name_Angular[2][2] = "dxy        ";
  Name_Angular[2][3] = "dxz        ";
  Name_Angular[2][4] = "dyz        ";
  Name_Angular[3][0] = "f5z^2-3r^2 ";
  Name_Angular[3][1] = "f5xz^2-xr^2";
  Name_Angular[3][2] = "f5yz^2-yr^2";
  Name_Angular[3][3] = "fzx^2-zy^2 ";
  Name_Angular[3][4] = "fxyz       ";
  Name_Angular[3][5] = "fx^3-3*xy^2";
  Name_Angular[3][6] = "f3yx^2-y^3 ";
  Name_Angular[4][0] = "g1         ";
  Name_Angular[4][1] = "g2         ";
  Name_Angular[4][2] = "g3         ";
  Name_Angular[4][3] = "g4         ";
  Name_Angular[4][4] = "g5         ";
  Name_Angular[4][5] = "g6         ";
  Name_Angular[4][6] = "g7         ";
  Name_Angular[4][7] = "g8         ";
  Name_Angular[4][8] = "g9         ";

  Name_Multiple[0] = "0";
  Name_Multiple[1] = "1";
  Name_Multiple[2] = "2";
  Name_Multiple[3] = "3";
  Name_Multiple[4] = "4";
  Name_Multiple[5] = "5";

  if (myid==Host_ID){
    strcpy(file_EV,".EV");
    fnjoint(filepath,filename,file_EV);
    if ((fp_EV = fopen(file_EV,"a")) != NULL){
      fprintf(fp_EV,"\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"          Unfolding calculation for band structure         \n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"***********************************************************\n");
      fprintf(fp_EV,"                                                                          \n");
      fprintf(fp_EV," Origin of the Reference cell is set to (%f %f %f) (Bohr).\n\n",
              unfold_origin[0],unfold_origin[1],unfold_origin[2]);
      fprintf(fp_EV," Unfolded weights at specified k points are stored in System.Name.unfold_totup(dn).\n");
      fprintf(fp_EV," Individual orbital weights are stored in System.Name.unfold_orbup(dn).\n");
      fprintf(fp_EV," The format is: k_dis(Bohr^{-1})  energy(eV)  weight.\n\n");
      fprintf(fp_EV," The sequence for the orbital weights in System.Name.unfold_orbup(dn) is given below.\n\n");

      i1 = 1;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	wan1 = WhatSpecies[Gc_AN];
	for (l=0; l<=Supported_MaxL; l++){
	  for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
	    for (m=0; m<(2*l+1); m++){
	      fprintf(fp_EV,"  %4d ",i1);
	      if (l==0 && mul==0 && m==0)
		fprintf(fp_EV,"%4d %3s %s %s",
			Gc_AN,SpeName[wan1],Name_Multiple[mul],Name_Angular[l][m]);
	      else
		fprintf(fp_EV,"         %s %s",
			Name_Multiple[mul],Name_Angular[l][m]);
	      fprintf(fp_EV,"\n");
	      i1++;
	    }
	  }
	}
      }
  
      fprintf(fp_EV,"\n"); 
      fprintf(fp_EV,"\n  The total number of calculated k points is %i.\n",totnkpts);
      fprintf(fp_EV,"  The number of calculated k points on each path is \n");

      fprintf(fp_EV,"  For each path: ("); 
      for (i=0; i<nkpoint; i++){
        fprintf(fp_EV," %i",np[i]); 
      }
      fprintf(fp_EV," )\n\n");

      fprintf(fp_EV,"                 ka         kb         kc\n");

      kloop = 0;
      for (i=0; i<nkpoint; i++){
	for (j=0; j<np[i]; j++) {

          kloop++;

	  if (np[i]==1) {
	    fprintf(fp_EV,"  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,kpoint[i][1],kpoint[i][2],kpoint[i][3]);
	  } 
          else {
	    fprintf(fp_EV,"  %3d/%3d   %10.6f %10.6f %10.6f\n",
                    kloop,totnkpts,
                    kpoint[i][1]+j*(kpoint[i+1][1]-kpoint[i][1])/np[i],
		    kpoint[i][2]+j*(kpoint[i+1][2]-kpoint[i][2])/np[i],
		    kpoint[i][3]+j*(kpoint[i+1][3]-kpoint[i][3])/np[i]);
	  }

	}
      }
      fprintf(fp_EV,"\n");
      fclose(fp_EV);
    }
    else{
      printf("Failure of saving the EV file.\n");
      fclose(fp_EV);
    }

    strcpy(file_EV,".unfold_tot");
    fnjoint(filepath,filename,file_EV);
    fp_EV = fopen(file_EV,"w");
    if (fp_EV == NULL) {
      printf("Failure of saving the System.Name.unfold_totup file.\n");
      fclose(fp_EV);
    }
    strcpy(file_EV,".unfold_orb");
    fnjoint(filepath,filename,file_EV);
    fp_EV1 = fopen(file_EV,"w");
    if (fp_EV1 == NULL) {
      printf("Failure of saving the System.Name.unfold_orbup file.\n");
      fclose(fp_EV1);
    }
  }

  int kloopi,kloopj;
  double kpt0,kpt1,kpt2;

  /* for gnuplot example */
  if (myid==Host_ID){

    strcpy(file_EV,".unfold_plotexample");
    fnjoint(filepath,filename,file_EV);
    fp_EV0 = fopen(file_EV,"w");
    if (fp_EV0 == NULL) {
      printf("Failure of saving the System.Name.unfold_plotexample file.\n");
      fclose(fp_EV0);
    }
    fprintf(fp_EV0,"set yrange [%f:%f]\n",unfold_lbound*eV2Hartree,unfold_ubound*eV2Hartree);
    fprintf(fp_EV0,"set ylabel 'Energy (eV)'\n");
    fprintf(fp_EV0,"set xtics(");
    pk1=coe*kpoint[0][1]*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
      -coe*kpoint[0][2]*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
      +coe*kpoint[0][3]*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
    pk2=-coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
      +coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
      -coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
    pk3=coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
      -coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
      +coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);
    kdis=0.;

    for (kloopi=0; kloopi<nkpoint; kloopi++) {

      kpt0=kpoint[kloopi][1];
      kpt1=kpoint[kloopi][2];
      kpt2=kpoint[kloopi][3];

      k1= coe*kpt0*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
        -coe*kpt1*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
        +coe*kpt2*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
      k2=-coe*kpt0*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
        +coe*kpt1*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
        -coe*kpt2*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
      k3= coe*kpt0*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
        -coe*kpt1*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
        +coe*kpt2*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);
      K[0]=k1*rtv[1][1]+k2*rtv[2][1]+k3*rtv[3][1];
      K[1]=k1*rtv[1][2]+k2*rtv[2][2]+k3*rtv[3][2];
      K[2]=k1*rtv[1][3]+k2*rtv[2][3]+k3*rtv[3][3];
      K2[0]=pk1*rtv[1][1]+pk2*rtv[2][1]+pk3*rtv[3][1];
      K2[1]=pk1*rtv[1][2]+pk2*rtv[2][2]+pk3*rtv[3][2];
      K2[2]=pk1*rtv[1][3]+pk2*rtv[2][3]+pk3*rtv[3][3];
      dis2pk=distwovec(K,K2);
      kdis+=dis2pk;

      if (kloopi==(nkpoint-1)) 
        fprintf(fp_EV0,"'%s' %f)\n",unfold_kpoint_name[kloopi],kdis); 
      else 
        fprintf(fp_EV0,"'%s' %f,",unfold_kpoint_name[kloopi],kdis);

      pk1=k1;
      pk2=k2;
      pk3=k3;
    }

    pk1=coe*kpoint[0][1]*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
      -coe*kpoint[0][2]*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
      +coe*kpoint[0][3]*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
    pk2=-coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
      +coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
      -coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
    pk3=coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
      -coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
      +coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);
    fprintf(fp_EV0,"set xrange [0:%f]\n",kdis);
    fprintf(fp_EV0,"set arrow nohead from 0,0 to %f,0\n",kdis);
    kdis=0.;
    for (kloopi=1; kloopi<nkpoint-1; kloopi++) {
      fprintf(fp_EV0,"set arrow nohead from ");
      kpt0=kpoint[kloopi][1];
      kpt1=kpoint[kloopi][2];
      kpt2=kpoint[kloopi][3];
      k1= coe*kpt0*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
        -coe*kpt1*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
        +coe*kpt2*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
      k2=-coe*kpt0*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
        +coe*kpt1*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
        -coe*kpt2*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
      k3= coe*kpt0*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
        -coe*kpt1*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
        +coe*kpt2*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);
      K[0]=k1*rtv[1][1]+k2*rtv[2][1]+k3*rtv[3][1];
      K[1]=k1*rtv[1][2]+k2*rtv[2][2]+k3*rtv[3][2];
      K[2]=k1*rtv[1][3]+k2*rtv[2][3]+k3*rtv[3][3];
      K2[0]=pk1*rtv[1][1]+pk2*rtv[2][1]+pk3*rtv[3][1];
      K2[1]=pk1*rtv[1][2]+pk2*rtv[2][2]+pk3*rtv[3][2];
      K2[2]=pk1*rtv[1][3]+pk2*rtv[2][3]+pk3*rtv[3][3];
      dis2pk=distwovec(K,K2);
      kdis+=dis2pk;
      fprintf(fp_EV0,"%f,%f to %f,%f\n",kdis,unfold_lbound*eV2Hartree,kdis,unfold_ubound*eV2Hartree);
      pk1=k1;
      pk2=k2;
      pk3=k3;
    }
    fprintf(fp_EV0,"set style circle radius 0\n");
    fprintf(fp_EV0,"plot '%s.unfold_tot' using 1:2:($3)*0.05 notitle with circles lc rgb 'red'\n",filename);
  }
  /* end gnuplot example */

  pk1=coe*kpoint[0][1]*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
     -coe*kpoint[0][2]*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
     +coe*kpoint[0][3]*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
  pk2=-coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
     +coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
     -coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
  pk3=coe*kpoint[0][1]*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
     -coe*kpoint[0][2]*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
     +coe*kpoint[0][3]*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);

  /* for standard output */

  if (myid==Host_ID && 0<level_stdout) {
    printf(" The number of selected k points is %i.\n",totnkpts);

    printf(" For each path: (");
    for (i=0; i<nkpoint; i++){
      printf(" %i",np[i]); 
    }
    printf(" )\n\n");
    printf("                 ka         kb         kc\n");
  }

  /*********************************************
                      kloopi 
  *********************************************/

  kloop = 0;
  kdis=0.;
  for (kloopi=0; kloopi<nkpoint; kloopi++)
    for (kloopj=0; kloopj<np[kloopi]; kloopj++) {

      kloop++;

      if (np[kloopi]==1) {
	kpt0=kpoint[kloopi][1];
	kpt1=kpoint[kloopi][2];
	kpt2=kpoint[kloopi][3];
      } else {
	kpt0=kpoint[kloopi][1]+kloopj*(kpoint[kloopi+1][1]-kpoint[kloopi][1])/np[kloopi];
	kpt1=kpoint[kloopi][2]+kloopj*(kpoint[kloopi+1][2]-kpoint[kloopi][2])/np[kloopi];
	kpt2=kpoint[kloopi][3]+kloopj*(kpoint[kloopi+1][3]-kpoint[kloopi][3])/np[kloopi];
      }

      /* for standard output */
     
      if (myid==Host_ID && 0<level_stdout) {

        if (kloop==totnkpts)
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n\n",kloop,totnkpts,kpt0,kpt1,kpt2);
        else 
          printf("  %3d/%3d   %10.6f %10.6f %10.6f\n",kloop,totnkpts,kpt0,kpt1,kpt2);
      }

      k1= coe*kpt0*(fracabc[1][1]*fracabc[2][2]-fracabc[1][2]*fracabc[2][1])
         -coe*kpt1*(fracabc[0][1]*fracabc[2][2]-fracabc[2][1]*fracabc[0][2])
         +coe*kpt2*(fracabc[0][1]*fracabc[1][2]-fracabc[1][1]*fracabc[0][2]);
      k2=-coe*kpt0*(fracabc[1][0]*fracabc[2][2]-fracabc[2][0]*fracabc[1][2])
         +coe*kpt1*(fracabc[0][0]*fracabc[2][2]-fracabc[2][0]*fracabc[0][2])
         -coe*kpt2*(fracabc[0][0]*fracabc[1][2]-fracabc[1][0]*fracabc[0][2]);
      k3= coe*kpt0*(fracabc[1][0]*fracabc[2][1]-fracabc[1][1]*fracabc[2][0])
         -coe*kpt1*(fracabc[0][0]*fracabc[2][1]-fracabc[2][0]*fracabc[0][1])
         +coe*kpt2*(fracabc[0][0]*fracabc[1][1]-fracabc[1][0]*fracabc[0][1]);

      /* make S and H */

      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  S[i][j] = Complex(0.0,0.0);
	} 
      } 

      for (i=1; i<=2*n; i++){
	for (j=1; j<=2*n; j++){
	  H[i][j] = Complex(0.0,0.0);
	} 
      } 

      /* non-spin-orbit coupling and non-LDA+U */
      if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
	  && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){  

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];

	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	    GB_AN = natn[GA_AN][LB_AN];
	    Rn = ncn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    l1 = atv_ijk[Rn][1];
	    l2 = atv_ijk[Rn][2];
	    l3 = atv_ijk[Rn][3];
	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

	    for (i=0; i<tnoA; i++){
	      for (j=0; j<tnoB; j++){

		H[Anum+i  ][Bnum+j  ].r += co*RH0[k];
		H[Anum+i  ][Bnum+j  ].i += si*RH0[k];

		H[Anum+i+n][Bnum+j+n].r += co*RH1[k];
		H[Anum+i+n][Bnum+j+n].i += si*RH1[k];
            
		H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*RH3[k];
		H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*RH3[k];

		S[Anum+i  ][Bnum+j  ].r += co*S1[k];
		S[Anum+i  ][Bnum+j  ].i += si*S1[k];

		k++;
	      }
	    }
	  }
	}
      }

      /* spin-orbit coupling or LDA+U */
      else {  

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];

	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	    GB_AN = natn[GA_AN][LB_AN];
	    Rn = ncn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    Bnum = MP[GB_AN];

	    l1 = atv_ijk[Rn][1];
	    l2 = atv_ijk[Rn][2];
	    l3 = atv_ijk[Rn][3];
	    kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	    si = sin(2.0*PI*kRn);
	    co = cos(2.0*PI*kRn);

	    for (i=0; i<tnoA; i++){
	      for (j=0; j<tnoB; j++){

		H[Anum+i  ][Bnum+j  ].r += co*RH0[k] - si*IH0[k];
		H[Anum+i  ][Bnum+j  ].i += si*RH0[k] + co*IH0[k];

		H[Anum+i+n][Bnum+j+n].r += co*RH1[k] - si*IH1[k];
		H[Anum+i+n][Bnum+j+n].i += si*RH1[k] + co*IH1[k];
            
		H[Anum+i  ][Bnum+j+n].r += co*RH2[k] - si*(RH3[k]+IH2[k]);
		H[Anum+i  ][Bnum+j+n].i += si*RH2[k] + co*(RH3[k]+IH2[k]);

		S[Anum+i  ][Bnum+j  ].r += co*S1[k];
		S[Anum+i  ][Bnum+j  ].i += si*S1[k];

		k++;
	      }
	    }
	  }
	}
      }

      /* set off-diagonal part */

      for (i=1; i<=n; i++){
	for (j=1; j<=n; j++){
	  H[j+n][i].r = H[i][j+n].r;
	  H[j+n][i].i =-H[i][j+n].i;
	} 
      } 

      /* diagonalization of S */
      Eigen_PHH(mpi_comm_level1,S,koS,n,n,1);

      /* minus eigenvalues to 1.0e-10 */

      for (l=1; l<=n; l++){
	if (koS[l]<1.0e-10) koS[l] = 1.0e-10;
      }

      /* calculate S*1/sqrt(koS) */

      for (l=1; l<=n; l++) koS[l] = 1.0/sqrt(koS[l]);

      /* S * 1.0/sqrt(koS[l]) */

#pragma omp parallel for shared(n,S,koS) private(i1,j1) 

      for (i1=1; i1<=n; i1++){
	for (j1=1; j1<=n; j1++){
	  S[i1][j1].r = S[i1][j1].r*koS[j1];
	  S[i1][j1].i = S[i1][j1].i*koS[j1];
	} 
      } 

      /****************************************************
                  set H' and diagonalize it
      ****************************************************/

      /* U'^+ * H * U * M1 */

      /* transpose S */

      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /* H * U' */
      /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (i1=1+OMPID; i1<=2*n; i1+=Nthrds){
	  for (j1=is1[myid]; j1<=ie1[myid]; j1++){

	    double sum_r0 = 0.0;
	    double sum_i0 = 0.0;

	    double sum_r1 = 0.0;
	    double sum_i1 = 0.0;

	    for (l=1; l<=n; l++){
	      sum_r0 += H[i1][l  ].r*S[j1][l].r - H[i1][l  ].i*S[j1][l].i;
	      sum_i0 += H[i1][l  ].r*S[j1][l].i + H[i1][l  ].i*S[j1][l].r;

	      sum_r1 += H[i1][n+l].r*S[j1][l].r - H[i1][n+l].i*S[j1][l].i;
	      sum_i1 += H[i1][n+l].r*S[j1][l].i + H[i1][n+l].i*S[j1][l].r;
	    }

	    C[2*j1-1][i1].r = sum_r0;
	    C[2*j1-1][i1].i = sum_i0;

	    C[2*j1  ][i1].r = sum_r1;
	    C[2*j1  ][i1].i = sum_i1;
	  }
	} 

      } /* #pragma omp parallel */

      /* U'^+ H * U' */
      /* H is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is1,ie1,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l,jj1,jj2)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (j1=is1[myid]+OMPID; j1<=ie1[myid]; j1+=Nthrds){
	  for (i1=1; i1<=n; i1++){

	    double sum_r00 = 0.0;
	    double sum_i00 = 0.0;

	    double sum_r01 = 0.0;
	    double sum_i01 = 0.0;

	    double sum_r10 = 0.0;
	    double sum_i10 = 0.0;

	    double sum_r11 = 0.0;
	    double sum_i11 = 0.0;

	    jj1 = 2*j1 - 1;
	    jj2 = 2*j1;

	    for (l=1; l<=n; l++){

	      sum_r00 += S[i1][l].r*C[jj1][l  ].r + S[i1][l].i*C[jj1][l  ].i;
	      sum_i00 += S[i1][l].r*C[jj1][l  ].i - S[i1][l].i*C[jj1][l  ].r;

	      sum_r01 += S[i1][l].r*C[jj1][l+n].r + S[i1][l].i*C[jj1][l+n].i;
	      sum_i01 += S[i1][l].r*C[jj1][l+n].i - S[i1][l].i*C[jj1][l+n].r;

	      sum_r10 += S[i1][l].r*C[jj2][l  ].r + S[i1][l].i*C[jj2][l  ].i;
	      sum_i10 += S[i1][l].r*C[jj2][l  ].i - S[i1][l].i*C[jj2][l  ].r;

	      sum_r11 += S[i1][l].r*C[jj2][l+n].r + S[i1][l].i*C[jj2][l+n].i;
	      sum_i11 += S[i1][l].r*C[jj2][l+n].i - S[i1][l].i*C[jj2][l+n].r;
	    }

	    H[jj1][2*i1-1].r = sum_r00;
	    H[jj1][2*i1-1].i = sum_i00;

	    H[jj1][2*i1  ].r = sum_r01;
	    H[jj1][2*i1  ].i = sum_i01;

	    H[jj2][2*i1-1].r = sum_r10;
	    H[jj2][2*i1-1].i = sum_i10;

	    H[jj2][2*i1  ].r = sum_r11;
	    H[jj2][2*i1  ].i = sum_i11;

	  }
	}

      } /* #pragma omp parallel */

      /* broadcast H */

      BroadCast_ComplexMatrix(mpi_comm_level1,H,2*n,is12,ie12,myid,numprocs,
			      stat_send,request_send,request_recv);

      /* H to C (transposition) */

#pragma omp parallel for shared(n,C,H)  

      for (i1=1; i1<=2*n; i1++){
	for (j1=1; j1<=2*n; j1++){
	  C[j1][i1].r = H[i1][j1].r;
	  C[j1][i1].i = H[i1][j1].i;
	}
      }

      /* solve the standard eigenvalue problem */
      /*  The output C matrix is distributed by column. */

      Eigen_PHH(mpi_comm_level1,C,ko,2*n,MaxN,0);

      for (i1=1; i1<=MaxN; i1++){
	EIGEN[i1] = ko[i1];
      }

      /* calculation of wave functions */

      /*  The H matrix is distributed by row */

      for (i1=1; i1<=2*n; i1++){
	for (j1=is2[myid]; j1<=ie2[myid]; j1++){
	  H[j1][i1] = C[i1][j1];
	}
      }

      /* transpose */

      for (i1=1; i1<=n; i1++){
	for (j1=i1+1; j1<=n; j1++){
	  Ctmp1 = S[i1][j1];
	  Ctmp2 = S[j1][i1];
	  S[i1][j1] = Ctmp2;
	  S[j1][i1] = Ctmp1;
	}
      }

      /* C is distributed by row in each processor */

#pragma omp parallel shared(C,S,H,n,is2,ie2,myid) private(OMPID,Nthrds,Nprocs,i1,j1,l,l1)
      { 

	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for (j1=is2[myid]+OMPID; j1<=ie2[myid]; j1+=Nthrds){
	  for (i1=1; i1<=n; i1++){

	    double sum_r0 = 0.0; 
	    double sum_i0 = 0.0;

	    double sum_r1 = 0.0; 
	    double sum_i1 = 0.0;

	    l1 = 0; 

	    for (l=1; l<=n; l++){

	      l1++; 

	      sum_r0 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
	      sum_i0 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;

	      l1++; 

	      sum_r1 +=  S[i1][l].r*H[j1][l1].r - S[i1][l].i*H[j1][l1].i;
	      sum_i1 +=  S[i1][l].r*H[j1][l1].i + S[i1][l].i*H[j1][l1].r;
	    } 

	    C[j1][i1  ].r = sum_r0;
	    C[j1][i1  ].i = sum_i0;

	    C[j1][i1+n].r = sum_r1;
	    C[j1][i1+n].i = sum_i1;

	  }
	}

      } /* #pragma omp parallel */

      /* broadcast C: C is distributed by row in each processor */

      BroadCast_ComplexMatrix(mpi_comm_level1,C,2*n,is2,ie2,myid,numprocs,
			      stat_send,request_send,request_recv);

      /* C to H (transposition)
	 H consists of column vectors
      */ 

      for (i1=1; i1<=MaxN; i1++){
	for (j1=1; j1<=2*n; j1++){
	  H[j1][i1] = C[i1][j1];
	}
      }

      /****************************************************
                        Output
      ****************************************************/

      if (myid==Host_ID){

	setvbuf(fp_EV,buf,_IOFBF,fp_bsize);  /* setvbuf */

	K[0]=k1*rtv[1][1]+k2*rtv[2][1]+k3*rtv[3][1];
	K[1]=k1*rtv[1][2]+k2*rtv[2][2]+k3*rtv[3][2];
	K[2]=k1*rtv[1][3]+k2*rtv[2][3]+k3*rtv[3][3];
	K2[0]=pk1*rtv[1][1]+pk2*rtv[2][1]+pk3*rtv[3][1];
	K2[1]=pk1*rtv[1][2]+pk2*rtv[2][2]+pk3*rtv[3][2];
	K2[2]=pk1*rtv[1][3]+pk2*rtv[2][3]+pk3*rtv[3][3];
	dis2pk=distwovec(K,K2);
	kdis+=dis2pk;

	num0 = 2;
	num1 = 2*n/num0 + 1*((2*n)%num0!=0);
  
	for (i=1; i<=num1; i++){

          countkj_e=0;
          for (j=0; j<2; j++) iskj_e[j]=0;

	  for (i1=-2; i1<=0; i1++){

	    for (j=1; j<=num0; j++){

	      j1 = num0*(i-1) + j;

	      if (j1<=2*n){ 

		if (i1==-1){

                  if (((EIGEN[j1]-ChemP)<=unfold_ubound)&&((EIGEN[j1]-ChemP)>=unfold_lbound)) {
                    kj_e[countkj_e]=(EIGEN[j1]-ChemP);
                    iskj_e[countkj_e]=1; }
                  countkj_e++;

		}
	      }
	    }
	  }

	  i1 = 1; 
          int iorb;

	  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
            iorb=0;

	    wan1 = WhatSpecies[Gc_AN];
            
	    for (l=0; l<=Supported_MaxL; l++){
	      for (mul=0; mul<Spe_Num_CBasis[wan1][l]; mul++){
		for (m=0; m<(2*l+1); m++){

                  countkj_e=0;

		  for (j=1; j<=num0; j++){

		    j1 = num0*(i-1) + j;

		    if (0<i1 && j1<=2*n){
                      kj_v[countkj_e][Gc_AN-1][iorb]=Complex(H[i1][j1].r,H[i1][j1].i);
                      kj_v1[countkj_e++][Gc_AN-1][iorb]=Complex(H[i1+n][j1].r,H[i1+n][j1].i);
		    }
		  }

		  i1++;
                  iorb++;
		}
	      }
	    }
	  }

          for (countkj_e=0; countkj_e<2; countkj_e++) if (iskj_e[countkj_e]==1) {
	    for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) weight[k][j]=Complex(0.,0.);
	    for (k=0; k<atomnum; k++) for (j=0; j<Norbperatom[k]; j++) weight1[k][j]=Complex(0.,0.);

	    for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) for (l=0; l<Norbperatom[j]; l++)
	      for (m=0; m<Norbperatom[k]; m++) tmpelem[j][k][l][m]=Cmul(Conjg(kj_v[countkj_e][j][l]),kj_v[countkj_e][k][m]);
	    for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) for (l=0; l<Norbperatom[j]; l++)
	      for (m=0; m<Norbperatom[k]; m++) tmpelem1[j][k][l][m]=Cmul(Conjg(kj_v1[countkj_e][j][l]),kj_v1[countkj_e][k][m]);

	    int NA,ir,MA,MO,NO;
	    dcomplex dtmp,dtmp1;

	    for (NA=0; NA<atomnum; NA++) {

	      int n=unfold_mapN2n[NA]; 
              int r0x=tabr4RN[0][NA][0]; 
              int r0y=tabr4RN[0][NA][1]; 
              int r0z=tabr4RN[0][NA][2];

	      r0[0]=r0x*a[0]+r0y*b[0]+r0z*c[0];
	      r0[1]=r0x*a[1]+r0y*b[1]+r0z*c[1];
	      r0[2]=r0x*a[2]+r0y*b[2]+r0z*c[2];

	      dcomplex phase1=Cexp(Complex(0.,-dot(K,r0)));

	      for (ir=0; ir<nr; ir++) {

		if (rnmap[ir][n][1]==-1) continue;

		r[0]=rlist[ir][0]*a[0]+rlist[ir][1]*b[0]+rlist[ir][2]*c[0];
		r[1]=rlist[ir][0]*a[1]+rlist[ir][1]*b[1]+rlist[ir][2]*c[1];
		r[2]=rlist[ir][0]*a[2]+rlist[ir][1]*b[2]+rlist[ir][2]*c[2];

		dcomplex phase2=Cmul(phase1,Cexp(Complex(0.,dot(K,r))));

		for (MA=0; MA<atomnum; MA++) {

		  if (Elem[rnmap[ir][n][0]][MA][rnmap[ir][n][1]][0][0]<-99999.) continue;

		  for (MO=0; MO<Norbperatom[MA]; MO++) for (NO=0; NO<Norbperatom[NA]; NO++) {
		    dtmp=RCmul(Elem[rnmap[ir][n][0]][MA][rnmap[ir][n][1]][MO][NO],tmpelem[MA][NA][MO][NO]);
		    dtmp1=RCmul(Elem[rnmap[ir][n][0]][MA][rnmap[ir][n][1]][MO][NO],tmpelem1[MA][NA][MO][NO]);
		    weight[NA][NO]=Cadd(weight[NA][NO],Cmul(phase2,dtmp));
		    weight1[NA][NO]=Cadd(weight1[NA][NO],Cmul(phase2,dtmp1));
		  }}}}

	    double sumallorb=0.;
	    for (j=0; j<atomnum; j++) for (k=0; k<Norbperatom[j]; k++) sumallorb+=weight[j][k].r;
	    for (j=0; j<atomnum; j++) for (k=0; k<Norbperatom[j]; k++) sumallorb+=weight1[j][k].r;
	    fprintf(fp_EV,"%f %f %10.7f\n",kdis,kj_e[countkj_e]*eV2Hartree,fabs(sumallorb)/coe);

	    /* set negative weight to zero for plotting purpose */
	    for (j=0; j<atomnum; j++){
              for (k=0; k<Norbperatom[j]; k++){

  	        if ( (weight[j][k].r+weight1[j][k].r)<0.0) {
                   weight[j][k].r  = 0.0; 
                   weight1[j][k].r = 0.0; 
                }
	      }
	    }

	    fprintf(fp_EV1,"%f %f ",kdis,kj_e[countkj_e]*eV2Hartree);
	    for (j=0; j<atomnum; j++) {
	      for (k=0; k<Norbperatom[j]; k++) fprintf(fp_EV1,"%e ",(weight[j][k].r+weight1[j][k].r)/coe);
	    }
	    fprintf(fp_EV1,"\n");
          }  
        }
      } /* if (myid==Host_ID) */

      MPI_Barrier(mpi_comm_level1);

      pk1=k1;
      pk2=k2;
      pk3=k3;

    }  /* kloopj */

  if (myid==Host_ID){
    if (fp_EV0 != NULL) fclose(fp_EV0);
    if (fp_EV != NULL) fclose(fp_EV);
    if (fp_EV1 != NULL) fclose(fp_EV1);
  }

  /****************************************************
                       free arrays
  ****************************************************/

  free(K);
  free(K2);
  free(r);
  free(r0);
  free(kj_e);
  free(iskj_e);
  free(unfold_mapN2n);
  free(a);
  free(b);
  free(c);
  free(unfold_origin);
  free(np);

  for (i=0; i<3; i++) free(unfold_abc[i]); free(unfold_abc);
  for (i=0; i<2; i++) for (j=0; j<atomnum; j++) free(kj_v[i][j]);
  for (i=0; i<2; i++) free(kj_v[i]); free(kj_v);
  for (i=0; i<2; i++) for (j=0; j<atomnum; j++) free(kj_v1[i][j]);
  for (i=0; i<2; i++) free(kj_v1[i]); free(kj_v1);
  for (i=0; i<atomnum; i++) free(weight[i]); free(weight);
  for (i=0; i<atomnum; i++) free(weight1[i]); free(weight1);
  for (i=0; i<NR; i++) free(Rlist[i]); free(Rlist);
  for (i=0; i<nr; i++) free(rlist[i]); free(rlist);
  for (i=0; i<21; i++) for (j=0; j<21; j++) free(_MapR[i][j]);
  for (i=0; i<21; i++) free(_MapR[i]); free(_MapR);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) free(tabr4RN[i][j]);
  for (i=0; i<NR; i++) free(tabr4RN[i]); free(tabr4RN);
  for (i=0; i<nr; i++) for (j=0; j<natom; j++) free(rnmap[i][j]);
  for (i=0; i<nr; i++) free(rnmap[i]); free(rnmap);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) for (l=0; l<Norbperatom[j]; l++) free(Elem[i][j][k][l]);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) for (k=0; k<atomnum; k++) free(Elem[i][j][k]);
  for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) free(Elem[i][j]);
  for (i=0; i<NR; i++) free(Elem[i]); free(Elem);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) for (k=0; k<Norbperatom[i]; k++)
    free(tmpelem[i][j][k]);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) free(tmpelem[i][j]);
  for (i=0; i<atomnum; i++) free(tmpelem[i]); free(tmpelem);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) for (k=0; k<Norbperatom[i]; k++)
    free(tmpelem1[i][j][k]);
  for (i=0; i<atomnum; i++) for (j=0; j<atomnum; j++) free(tmpelem1[i][j]);
  for (i=0; i<atomnum; i++) free(tmpelem1[i]); free(tmpelem1);
  for (i=0; i<3; i++) free(fracabc[i]); free(fracabc);
  free(Norbperatom);
  for (i=0; i<unfold_Nkpoint+1; i++) free(unfold_kpoint[i]); free(unfold_kpoint);
  for (i=0; i<(unfold_Nkpoint+1); i++){
    free(unfold_kpoint_name[i]);
  }
  free(unfold_kpoint_name);

  free(stat_send);
  free(request_send);
  free(request_recv);

  free(is1);
  free(ie1);
  free(is2);
  free(ie2);
  free(is12);
  free(ie12);

  free(MP);
  free(order_GA);

  free(My_NZeros);
  free(SP_NZeros);
  free(SP_Atoms);

  free(ko);
  free(koS);

  free(S1);
  free(RH0);
  free(RH1);
  free(RH2);
  free(RH3);
  free(IH0);
  free(IH1);
  free(IH2);

  free(EIGEN);

  for (j=0; j<n2; j++){
    free(H[j]);
  }
  free(H);

  for (i=0; i<n2; i++){
    free(S[i]);
  }
  free(S);

  free(M1);

  for (j=0; j<n2; j++){
    free(C[j]);
  }
  free(C);

  for (j=0; j<n2; j++){
    free(Ctmp[j]);
  }
  free(Ctmp);

  dtime(&TEtime);

}


static double volume(const double* a,const double* b,const double* c) {
  return fabs(a[0]*b[1]*c[2]+b[0]*c[1]*a[2]+c[0]*a[1]*b[2]-c[0]*b[1]*a[2]-a[1]*b[0]*c[2]-a[0]*c[1]*b[2]);}

int chkr(int a, int b, int c) {int i; for (i=0; i<nr; i++) if ((rlist[i][0]==a)&&(rlist[i][1]==b)&&(rlist[i][2]==c)) return i;
  return -99999;}

int addr(int a, int b, int c) { int i, j, ck; ck=chkr(a,b,c);
  if (ck!=-99999) return ck;
  else {
    int** tmprlist;
    tmprlist = (int**)malloc(sizeof(int*)*nr);
    for (i=0; i<nr; i++) {
      tmprlist[i]=(int*)malloc(sizeof(int)*3);
      for (j=0; j<3; j++) tmprlist[i][j]=rlist[i][j];
    }
    for (i=0; i<nr; i++) free(rlist[i]); free(rlist);
    rlist=(int**)malloc(sizeof(int*)*++nr);
    for (i=0; i<nr; i++) rlist[i]=(int*)malloc(sizeof(int)*3);
    for (i=0; i<nr-1; i++) for (j=0; j<3; j++)
      rlist[i][j]=tmprlist[i][j];
    rlist[nr-1][0]=a; rlist[nr-1][1]=b; rlist[nr-1][2]=c;
    for (i=0; i<nr-1; i++) free(tmprlist[i]);
    free(tmprlist);
    return nr-1;
  }}

void buildrnmap(const int* mapN2n) {
  int i, j, k;
  natom=-999; for (i=0; i<atomnum; i++) if (mapN2n[i]>natom) natom=mapN2n[i];
  natom++;
  rnmap=(int***)malloc(sizeof(int**)*nr);
  for (i=0; i<nr; i++) {
    rnmap[i]=(int**)malloc(sizeof(int*)*natom);
    for (j=0; j<natom; j++) {
      rnmap[i][j]=(int*)malloc(sizeof(int)*2);
      rnmap[i][j][1]=-1;
    }
  }
  int iR, iN;
  for (iR=0; iR<NR; iR++) for (iN=0; iN<atomnum; iN++) {
    int tmpi; tmpi=addr(tabr4RN[iR][iN][0],tabr4RN[iR][iN][1],tabr4RN[iR][iN][2]);
    rnmap[tmpi][mapN2n[iN]][0]=iR; rnmap[tmpi][mapN2n[iN]][1]=iN;
  }
}

static double dis(const double* a) {return sqrt(a[2]*a[2]+a[1]*a[1]+a[0]*a[0]);}

static double distwovec(const double* a, const double* b) {return sqrt((a[2]-b[2])*(a[2]-b[2])+(a[1]-b[1])*(a[1]-b[1])+(a[0]-b[0])*(a[0]-b[0]));}

static double det(const double* a,const double* b,const double* c) {
  return a[0]*b[1]*c[2]+b[0]*c[1]*a[2]+c[0]*a[1]*b[2]-c[0]*b[1]*a[2]-a[1]*b[0]*c[2]-a[0]*c[1]*b[2];}

/* abc = S ABC */
void abc_by_ABC(double ** S) { 
  double detABC=tv[1][1]*tv[2][2]*tv[3][3]+tv[2][1]*tv[3][2]*tv[1][3]+tv[3][1]*tv[1][2]*tv[2][3]-tv[3][1]*tv[2][2]*tv[1][3]-tv[1][2]*tv[2][1]*tv[3][3]-tv[1][1]*tv[3][2]*tv[2][3];
  int i,j,k;
  double** inv = (double**)malloc(sizeof(double*)*3);
  for (i=0; i<3; i++) inv[i]=(double*)malloc(sizeof(double)*3); 
  inv[0][0]=(tv[2][2]*tv[3][3]-tv[3][2]*tv[2][3])/detABC;
  inv[0][1]=(tv[1][3]*tv[3][2]-tv[3][3]*tv[1][2])/detABC;
  inv[0][2]=(tv[1][2]*tv[2][3]-tv[2][2]*tv[1][3])/detABC;
  inv[1][0]=(tv[2][3]*tv[3][1]-tv[3][3]*tv[2][1])/detABC;
  inv[1][1]=(tv[1][1]*tv[3][3]-tv[3][1]*tv[1][3])/detABC;
  inv[1][2]=(tv[1][3]*tv[2][1]-tv[2][3]*tv[1][1])/detABC;
  inv[2][0]=(tv[2][1]*tv[3][2]-tv[3][1]*tv[2][2])/detABC;
  inv[2][1]=(tv[1][2]*tv[3][1]-tv[3][2]*tv[1][1])/detABC;
  inv[2][2]=(tv[1][1]*tv[2][2]-tv[2][1]*tv[1][2])/detABC;
  for (i=0; i<3; i++) for (j=0; j<3; j++) S[i][j]=0.;
  for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++) S[i][j]+=unfold_abc[i][k]*inv[k][j];
  for (i=0; i<3; i++) free(inv[i]); free(inv);
}

/* det(v-a-b-c,b,c) */
static double det1(const double* a,const double* b,const double* c,const double* v) {
  return (v[0]-a[0]-b[0]-c[0])*b[1]*c[2]+b[0]*c[1]*(v[2]-a[2]-b[2]-c[2])+c[0]*(v[1]-a[1]-b[1]-c[1])*b[2]-c[0]*b[1]*(v[2]-a[2]-b[2]-c[2])-(v[1]-a[1]-b[1]-c[1])*b[0]*c[2]-(v[0]-a[0]-b[0]-c[0])*c[1]*b[2];}

int insidecube(const double* va,const double* vb,const double* vc,const double* vatom) {
  double chk0=det(vatom,va,vb);
  double chk1=det(vatom,vb,vc);
  double chk2=det(vatom,vc,va);
  double chk3=det1(vc,va,vb,vatom); 
  double chk4=det1(va,vb,vc,vatom); 
  double chk5=det1(vb,vc,va,vatom); 
  if (fabs(chk5)<0.0000001) return 0;
  if (fabs(chk4)<0.0000001) return 0;
  if (fabs(chk3)<0.0000001) return 0;
  if (fabs(chk2)<0.0000001) chk2=0.;
  if (fabs(chk1)<0.0000001) chk1=0.;
  if (fabs(chk0)<0.0000001) chk0=0.;
  if (chk0*chk3>0.0000001) return 0;
  if (chk1*chk4>0.0000001) return 0;
  if (chk2*chk5>0.0000001) return 0;
  return 1;
}

static double dot(const double* v1,const double* v2) {
  double dotsum=0.;
  int i;
  for (i=0; i<3; i++) dotsum+=v1[i]*v2[i];
  return dotsum;
}

void getnorbperatom() {
  Norbperatom = (int*)malloc(sizeof(int)*atomnum);
  int ct_AN, wan1, TNO1;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    wan1 = WhatSpecies[ct_AN];
    TNO1 = Spe_Total_CNO[wan1];
    Norbperatom[ct_AN-1] = TNO1;
  }

  Norb=0;
  int* Ibegin;
  Ibegin = (int*)malloc(sizeof(int)*atomnum);
  int i;
  for (i=0; i<atomnum; i++) {Ibegin[i]=Norb; Norb+=Norbperatom[i];}
  free(Ibegin);
}

static int MapR(const int i,const int j,const int k) { return _MapR[i+10][j+10][k+10];}

int AddR(const int a,const int b,const int c) { if (MapR(a,b,c)==-1) {
    int i, j;
    int** tmpRlist;
    tmpRlist=(int**)malloc(sizeof(int*)*NR);
    for (i=0; i<NR; i++) {
      tmpRlist[i]=(int*)malloc(sizeof(int)*3);
      for (j=0; j<3; j++) {
	tmpRlist[i][j]=Rlist[i][j];
      }
    }

    for (i=0; i<NR; i++) free(Rlist[i]);
    free(Rlist);
    Rlist=(int**)malloc(sizeof(int*)*++NR);
    for (i=0; i<NR; i++) {
      Rlist[i]=(int*)malloc(sizeof(int)*3);
    }

    Rlist[NR-1][0]=a; Rlist[NR-1][1]=b; Rlist[NR-1][2]=c;
    for (i=0; i<NR-1; i++) for (j=0; j<3; j++)
      Rlist[i][j]=tmpRlist[i][j];
    if ((a>10)||(a<-10)||(b>10)||(b<-10)||(c>10)||(c<-10)) {
      free(tmpRlist);
      for (i=0; i<NR-1; i++) free(tmpRlist[i]);
      free(tmpRlist);
      printf("R in overlap matrix is larger than expected\n");
      exitcode=1;
      return -9999;
    }
    _MapR[a+10][b+10][c+10]=NR-1;

    for (i=0; i<NR-1; i++) free(tmpRlist[i]);
    free(tmpRlist);

    return MapR(a,b,c);
  } else return MapR(a,b,c);
}

void buildMapRlist() {
  int i, j, k, l, m;
  _MapR=(int***)malloc(sizeof(int**)*21);         
  for (i=0; i<21; i++) {
    _MapR[i]=(int**)malloc(sizeof(int*)*21);
    for (j=0; j<21; j++) {
      _MapR[i][j]=(int*)malloc(sizeof(int)*21);
      for (k=0; k<21; k++) _MapR[i][j][k]=-1;
    }
  }
  _MapR[10][10][10]=0;

  NR=1;
  Rlist=(int**)malloc(sizeof(int*)*NR);         
  for (i=0; i<NR; i++) {
    Rlist[i]=(int*)malloc(sizeof(int)*3);
    for (j=0; j<3; j++) {
      Rlist[i][j]=0;  
    }
  }

  int Gc_AN,Mc_AN,Gh_AN,h_AN;
  int num,wan1,wan2,TNO1,TNO2,Rn;
  int numprocs,myid,ID,tag=999;
  double *Tmp_Vec;
  Tmp_Vec = (double*)malloc(sizeof(double)*List_YOUSO[8]*List_YOUSO[7]*List_YOUSO[7]);
   
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Rn = ncn[Gc_AN][h_AN];
      AddR(atv_ijk[Rn][1], atv_ijk[Rn][2], atv_ijk[Rn][3]);
      if (exitcode==1) {
	for (i=0; i<21; i++) for (j=0; j<21; j++) free(_MapR[i][j]);
	for (i=0; i<21; i++) free(_MapR[i]); free(_MapR);
	for (i=0; i<NR; i++) free(Rlist[i]); free(Rlist);
	free(Tmp_Vec);
	return;
      }
    }
  }

  Elem=(double*****)malloc(sizeof(double****)*NR);
  for (i=0; i<NR; i++) {
    Elem[i]=(double****)malloc(sizeof(double***)*atomnum);
    for (j=0; j<atomnum; j++) { 
      Elem[i][j]=(double***)malloc(sizeof(double**)*atomnum);
      for (k=0; k<atomnum; k++) {
	Elem[i][j][k]=(double**)malloc(sizeof(double*)*Norbperatom[j]);
	for (l=0; l<Norbperatom[j]; l++) {
	  Elem[i][j][k][l]=(double*)malloc(sizeof(double)*Norbperatom[k]);
	  for (m=0; m<Norbperatom[k]; m++) {
	    Elem[i][j][k][l][m]=-9999999.88888;
	  }}}}}

  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    ID = G2ID[Gc_AN];

    if (myid==ID){

      num = 0;

      Mc_AN = F_G2M[Gc_AN];
      wan1 = WhatSpecies[Gc_AN];
      TNO1 = Spe_Total_CNO[wan1];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
        Rn = ncn[Gc_AN][h_AN];
        Gh_AN = natn[Gc_AN][h_AN];
        wan2 = WhatSpecies[Gh_AN];
        TNO2 = Spe_Total_CNO[wan2];

        if (Cnt_switch==0){
          for (i=0; i<TNO1; i++){
            for (j=0; j<TNO2; j++){
              Tmp_Vec[num] = OLP[0][Mc_AN][h_AN][i][j];
              num++;
            }
          }
        }
        else{
          for (i=0; i<TNO1; i++){
            for (j=0; j<TNO2; j++){
              Tmp_Vec[num] = CntOLP[0][Mc_AN][h_AN][i][j];
              num++;
            }
          }
        }
      }

      if (myid!=Host_ID){
        MPI_Isend(&num, 1, MPI_INT, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request,&stat);
        MPI_Isend(&Tmp_Vec[0], num, MPI_DOUBLE, Host_ID, tag, mpi_comm_level1, &request);
        MPI_Wait(&request,&stat);
      }
      else{
	/*        fwrite(Tmp_Vec, sizeof(double), num, fp); */
      }
    }

    else if (ID!=myid && myid==Host_ID){
      MPI_Recv(&num, 1, MPI_INT, ID, tag, mpi_comm_level1, &stat);
      MPI_Recv(&Tmp_Vec[0], num, MPI_DOUBLE, ID, tag, mpi_comm_level1, &stat);
      /*      fwrite(Tmp_Vec, sizeof(double), num, fp);*/
    }

    num = 0;
    wan1 = WhatSpecies[Gc_AN];
    TNO1 = Spe_Total_CNO[wan1];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Rn = ncn[Gc_AN][h_AN];
      Gh_AN = natn[Gc_AN][h_AN];
      wan2 = WhatSpecies[Gh_AN];
      TNO2 = Spe_Total_CNO[wan2];
      for (i=0; i<TNO1; i++){
	for (j=0; j<TNO2; j++){
	  Elem[MapR(atv_ijk[Rn][1],atv_ijk[Rn][2],atv_ijk[Rn][3])][Gc_AN-1][Gh_AN-1][i][j]=Tmp_Vec[num];
	  num++;
	}
      }
    }
  }

  free(Tmp_Vec);
}

/* assign each R N with r */
void buildtabr4RN(const double* a,const double* b,const double*c,double* origin,const int* mapN2n) {
  double* A;
  double* B;
  double* C;
  double** Atoms;
  double*tmp;
  A = (double*)malloc(sizeof(double)*3);
  B = (double*)malloc(sizeof(double)*3);
  C = (double*)malloc(sizeof(double)*3);
  A[0]=tv[1][1];
  A[1]=tv[1][2];
  A[2]=tv[1][3];
  B[0]=tv[2][1];
  B[1]=tv[2][2];
  B[2]=tv[2][3];
  C[0]=tv[3][1];
  C[1]=tv[3][2];
  C[2]=tv[3][3];
  int i, j, k;
  int iR, iN;
  int esti=0,estj=0,estk=0;
  Atoms = (double**)malloc(sizeof(double*)*atomnum);
  for (i=0; i<atomnum; i++) {
    Atoms[i] = (double*)malloc(sizeof(double)*3);
    for (j=0; j<3; j++) Atoms[i][j]=Gxyz[i+1][j+1];
  }

  double X,Y,Z;
  tmp = (double*)malloc(sizeof(double)*3);
  tabr4RN=(int***)malloc(sizeof(int**)*NR);
  for (i=0; i<NR; i++) {
    tabr4RN[i]=(int**)malloc(sizeof(int*)*atomnum);
    for (j=0; j<atomnum; j++) {
      tabr4RN[i][j]=(int*)malloc(sizeof(int)*3);
      tabr4RN[i][j][0]=-99999;
    }
  }

  /*  suggesting the origin of the reference cell */
  if ((fabs(origin[0]+0.9999900023114)<0.00000000001)&&
      (fabs(origin[1]+0.9999900047614)<0.00000000001)&&
      (fabs(origin[2]+0.9999900058914)<0.00000000001)) {
    double orgx,orgy,orgz;
    double reforgx,reforgy,reforgz;
    reforgx=Atoms[0][0]-a[0]-b[0]-c[0]+0.124966998688568*(a[0]+b[0]+c[0]);
    reforgy=Atoms[0][1]-a[1]-b[1]-c[1]+0.124966997688568*(a[1]+b[1]+c[1]);
    reforgz=Atoms[0][2]-a[2]-b[2]-c[2]+0.124966996688568*(a[2]+b[2]+c[2]);
    int*** nrtable;
    nrtable=(int***)malloc(sizeof(int**)*8);
    for (i=0; i<8; i++) {
      nrtable[i]=(int**)malloc(sizeof(int*)*8);
      for (j=0; j<8; j++) {
	nrtable[i][j]=(int*)malloc(sizeof(int)*8);
      }}

    int tmpi,tmpj,tmpk;
    for (tmpi=0; tmpi<8; tmpi++) for (tmpj=0; tmpj<8; tmpj++) for (tmpk=0; tmpk<8; tmpk++) {
      nr=1;
      rlist=(int**)malloc(sizeof(int*)*1);
      for (i=0; i<1; i++) {
	rlist[i]=(int*)malloc(sizeof(int)*3);
	for (j=0; j<3; j++) {
	  rlist[i][j]=0;
	}
      }
      for (j=0; j<atomnum; j++) tabr4RN[0][j][0]=-99999;
      orgx=reforgx+(double)tmpi*a[0]/8.+(double)tmpj*b[0]/8.+(double)tmpk*c[0]/8.;
      orgy=reforgy+(double)tmpi*a[1]/8.+(double)tmpj*b[1]/8.+(double)tmpk*c[1]/8.;
      orgz=reforgz+(double)tmpi*a[2]/8.+(double)tmpj*b[2]/8.+(double)tmpk*c[2]/8.;

      for (iN=0; iN<atomnum; iN++) {
        double estl=9999.;
	double X=(double)Rlist[0][0]*A[0]+(double)Rlist[0][1]*B[0]+(double)Rlist[0][2]*C[0]+Atoms[iN][0];
	double Y=(double)Rlist[0][0]*A[1]+(double)Rlist[0][1]*B[1]+(double)Rlist[0][2]*C[1]+Atoms[iN][1];
	double Z=(double)Rlist[0][0]*A[2]+(double)Rlist[0][1]*B[2]+(double)Rlist[0][2]*C[2]+Atoms[iN][2];
	for (i=-2*atomnum; i<=2*atomnum; i+=5) for (j=-2*atomnum; j<=2*atomnum; j+=5) for (k=-2*atomnum; k<=2*atomnum; k+=5) {
	  double x=(double)i*a[0]+(double)j*b[0]+(double)k*c[0]+orgx;
	  double y=(double)i*a[1]+(double)j*b[1]+(double)k*c[1]+orgy;
	  double z=(double)i*a[2]+(double)j*b[2]+(double)k*c[2]+orgz;
	  tmp[0]=X-x;
	  tmp[1]=Y-y;
	  tmp[2]=Z-z;
	  if (dis(tmp)<estl) { estl=dis(tmp); esti=i; estj=j; estk=k; }
	}
	int leave=0;
	for (i=esti-10; i<=esti+10; i++) {
	  if (leave==1) break;
	  for (j=estj-10; j<=estj+10; j++) {
            if (leave==1) break;
	    for (k=estk-10; k<=estk+10; k++) {
              double x=(double)i*a[0]+(double)j*b[0]+(double)k*c[0]+orgx;
              double y=(double)i*a[1]+(double)j*b[1]+(double)k*c[1]+orgy;
              double z=(double)i*a[2]+(double)j*b[2]+(double)k*c[2]+orgz;
              tmp[0]=X-x;
              tmp[1]=Y-y;
              tmp[2]=Z-z;
	      if (insidecube(a,b,c,tmp)) {
		tabr4RN[0][iN][0]=i;
		tabr4RN[0][iN][1]=j;
		tabr4RN[0][iN][2]=k;
		addr(i,j,k); leave=1; break;
	      }
            }}}
      }
      nrtable[tmpi][tmpj][tmpk]=nr;
      for (iN=0; iN<atomnum; iN++) {
        int chka=tabr4RN[0][iN][0];
        int chkb=tabr4RN[0][iN][1];
        int chkc=tabr4RN[0][iN][2];
        int chkn=mapN2n[iN];
        int iNp;
        for (iNp=iN+1; iNp<atomnum; iNp++)
          if (((chka==tabr4RN[0][iNp][0])&&
               (chkb==tabr4RN[0][iNp][1])&&
               (chkc==tabr4RN[0][iNp][2])&&
               (chkn==mapN2n[iNp]))) nrtable[tmpi][tmpj][tmpk]=-1;
      }
      for (i=0; i<nr; i++) free(rlist[i]); free(rlist);
    }

    int smallestnr;
    int leave=0;
    for (tmpi=0; tmpi<8; tmpi++) {
      if (leave==1) break;
      for (tmpj=0; tmpj<8; tmpj++) {
        if (leave==1) break;
	for (tmpk=0; tmpk<8; tmpk++) 
	  if (nrtable[tmpi][tmpj][tmpk]>0) { smallestnr=nrtable[tmpi][tmpj][tmpk]; leave=1; break; }
      }}

    for (tmpi=0; tmpi<8; tmpi++) for (tmpj=0; tmpj<8; tmpj++) for (tmpk=0; tmpk<8; tmpk++) 
      if ((nrtable[tmpi][tmpj][tmpk]>0)&&(nrtable[tmpi][tmpj][tmpk]<smallestnr)) smallestnr=nrtable[tmpi][tmpj][tmpk];
    for (tmpi=0; tmpi<8; tmpi++) for (tmpj=0; tmpj<8; tmpj++) for (tmpk=0; tmpk<8; tmpk++) 
      if (nrtable[tmpi][tmpj][tmpk]==smallestnr) {
        origin[0]=reforgx+(double)tmpi*a[0]/8.+(double)tmpj*b[0]/8.+(double)tmpk*c[0]/8.;
        origin[1]=reforgy+(double)tmpi*a[1]/8.+(double)tmpj*b[1]/8.+(double)tmpk*c[1]/8.;
        origin[2]=reforgz+(double)tmpi*a[2]/8.+(double)tmpj*b[2]/8.+(double)tmpk*c[2]/8.;
      } 

    for (i=0; i<8; i++) for (j=0; j<8; j++) free(nrtable[i][j]);
    for (i=0; i<8; i++) free(nrtable[i]);
    free(nrtable);
  } 
  /*  finish suggesting the reference origin      */

  nr=1;
  rlist=(int**)malloc(sizeof(int*)*1);
  for (i=0; i<1; i++) {
    rlist[i]=(int*)malloc(sizeof(int)*3);
    for (j=0; j<3; j++) {
      rlist[i][j]=0;
    }
  }

  for (j=0; j<atomnum; j++) tabr4RN[0][j][0]=-99999;

  int shiftorigin=0;
  /* try to find r vector for RN */
  for (iR=0; iR<NR; iR++) { for (iN=0; iN<atomnum; iN++) {
      double estl=9999.;
      double X=(double)Rlist[iR][0]*A[0]+(double)Rlist[iR][1]*B[0]+(double)Rlist[iR][2]*C[0]+Atoms[iN][0];
      double Y=(double)Rlist[iR][0]*A[1]+(double)Rlist[iR][1]*B[1]+(double)Rlist[iR][2]*C[1]+Atoms[iN][1];
      double Z=(double)Rlist[iR][0]*A[2]+(double)Rlist[iR][1]*B[2]+(double)Rlist[iR][2]*C[2]+Atoms[iN][2];
      for (i=-2*atomnum; i<=2*atomnum; i+=5) for (j=-2*atomnum; j<=2*atomnum; j+=5) for (k=-2*atomnum; k<=2*atomnum; k+=5) {
	double x=(double)i*a[0]+(double)j*b[0]+(double)k*c[0]+origin[0];
	double y=(double)i*a[1]+(double)j*b[1]+(double)k*c[1]+origin[1];
	double z=(double)i*a[2]+(double)j*b[2]+(double)k*c[2]+origin[2];
	tmp[0]=X-x;
	tmp[1]=Y-y;
	tmp[2]=Z-z;
	if (dis(tmp)<estl) { estl=dis(tmp); esti=i; estj=j; estk=k; }
      }
      int leave=0;
      for (i=esti-10; i<=esti+10; i++) {
        if (leave==1) break;
        for (j=estj-10; j<=estj+10; j++) {
	  if (leave==1) break;
          for (k=estk-10; k<=estk+10; k++) {
	    double x=(double)i*a[0]+(double)j*b[0]+(double)k*c[0]+origin[0];
	    double y=(double)i*a[1]+(double)j*b[1]+(double)k*c[1]+origin[1];
	    double z=(double)i*a[2]+(double)j*b[2]+(double)k*c[2]+origin[2];
	    tmp[0]=X-x;
	    tmp[1]=Y-y;
	    tmp[2]=Z-z;
            if (insidecube(a,b,c,tmp)) {
              tabr4RN[iR][iN][0]=i;
              tabr4RN[iR][iN][1]=j;
              tabr4RN[iR][iN][2]=k;
              addr(i,j,k); leave=1; break;
            }
	  }}}
      if (tabr4RN[iR][iN][0]==-99999) { shiftorigin=1; break; }
    }
    if (shiftorigin==1) break;
  }

  /*
    checking if two same kinds of normal cell atoms are assigned to the same r vector (could be due to locating around boundaries)
  */

  for (iR=0; iR<NR; iR++) { if (shiftorigin==1) break;
    for (iN=0; iN<atomnum; iN++) {
      int chka=tabr4RN[iR][iN][0];
      int chkb=tabr4RN[iR][iN][1];
      int chkc=tabr4RN[iR][iN][2];
      int chkn=mapN2n[iN];
      int iNp, iRp;
      for (iNp=iN+1; iNp<atomnum; iNp++)
        if ((shiftorigin==1)||((chka==tabr4RN[iR][iNp][0])&&
			       (chkb==tabr4RN[iR][iNp][1])&&
			       (chkc==tabr4RN[iR][iNp][2])&&
			       (chkn==mapN2n[iNp]))) {shiftorigin=1; break;}
      for (iRp=iR+1; iRp<NR; iRp++) for (iNp=0; iNp<atomnum; iNp++)
        if ((chka==tabr4RN[iRp][iNp][0])&&
            (chkb==tabr4RN[iRp][iNp][1])&&
            (chkc==tabr4RN[iRp][iNp][2])&&
            (chkn==mapN2n[iNp])) {shiftorigin=1; break;}
    }}

  if (shiftorigin==0) buildrnmap(mapN2n);

  free(A);
  free(B);
  free(C);
  free(tmp);
  for (i=0; i<atomnum; i++) free(Atoms[i]);
  free(Atoms);

  if (shiftorigin==1) { printf("Cannot assign atoms in the reference cell properly! Could be due to more than one same atom in the reference cell!\nCheck the input file, maybe the structure is highly disordered or you need to set the reference origin by yourself!\n\n"); 
    exitcode=1;
    for (i=0; i<nr; i++) free(rlist[i]); free(rlist);
    for (i=0; i<NR; i++) for (j=0; j<atomnum; j++) free(tabr4RN[i][j]);
    for (i=0; i<NR; i++) free(tabr4RN[i]); free(tabr4RN);
    return;
  }
}

void determine_kpts(const int nk, double** klist) {

  int i,j;
  double** reciplatt;
  reciplatt = (double**)malloc(sizeof(double*)*3);
  for (i=0; i<3; i++) reciplatt[i] = (double*)malloc(sizeof(double)*3);
  double V=2.*PI/(unfold_abc[0][0]*unfold_abc[1][1]*unfold_abc[2][2]
		  +unfold_abc[1][0]*unfold_abc[2][1]*unfold_abc[0][2]+unfold_abc[2][0]*unfold_abc[1][2]*unfold_abc[0][1]
		  -unfold_abc[2][0]*unfold_abc[1][1]*unfold_abc[0][2]-unfold_abc[0][1]*unfold_abc[1][0]*unfold_abc[2][2]
		  -unfold_abc[0][0]*unfold_abc[2][1]*unfold_abc[1][2]);
  reciplatt[0][0]=V*(unfold_abc[1][1]*unfold_abc[2][2]-unfold_abc[2][1]*unfold_abc[1][2]);
  reciplatt[0][1]=V*(unfold_abc[2][0]*unfold_abc[1][2]-unfold_abc[1][0]*unfold_abc[2][2]);
  reciplatt[0][2]=V*(unfold_abc[1][0]*unfold_abc[2][1]-unfold_abc[2][0]*unfold_abc[1][1]);
  reciplatt[1][0]=V*(unfold_abc[2][1]*unfold_abc[0][2]-unfold_abc[0][1]*unfold_abc[2][2]);
  reciplatt[1][1]=V*(unfold_abc[0][0]*unfold_abc[2][2]-unfold_abc[2][0]*unfold_abc[0][2]);
  reciplatt[1][2]=V*(unfold_abc[2][0]*unfold_abc[0][1]-unfold_abc[0][0]*unfold_abc[2][1]);
  reciplatt[2][0]=V*(unfold_abc[0][1]*unfold_abc[1][2]-unfold_abc[1][1]*unfold_abc[0][2]);
  reciplatt[2][1]=V*(unfold_abc[1][0]*unfold_abc[0][2]-unfold_abc[0][0]*unfold_abc[1][2]);
  reciplatt[2][2]=V*(unfold_abc[0][0]*unfold_abc[1][1]-unfold_abc[1][0]*unfold_abc[0][1]);
    
  double dis=0.;
  for (i=0; i<nk-1; i++) 
    dis+=sqrt(pow((klist[i][1]-klist[i+1][1])*reciplatt[0][0]+(klist[i][2]-klist[i+1][2])*reciplatt[1][0]+(klist[i][3]-klist[i+1][3])*reciplatt[2][0],2)+
              pow((klist[i][1]-klist[i+1][1])*reciplatt[0][1]+(klist[i][2]-klist[i+1][2])*reciplatt[1][1]+(klist[i][3]-klist[i+1][3])*reciplatt[2][1],2)+
              pow((klist[i][1]-klist[i+1][1])*reciplatt[0][2]+(klist[i][2]-klist[i+1][2])*reciplatt[1][2]+(klist[i][3]-klist[i+1][3])*reciplatt[2][2],2));

  np = (int*)malloc(sizeof(int)*(nk));

  if (unfold_nkpts<=nk) {
    for (i=0; i<nk; i++) np[i]=1;
    totnkpts=nk;
  } 
  else {
    double intvl=dis/(unfold_nkpts-1);
    for (i=0; i<nk-1; i++) {
      np[i]=
	(int)(sqrt(pow((klist[i][1]-klist[i+1][1])*reciplatt[0][0]+(klist[i][2]-klist[i+1][2])*reciplatt[1][0]+(klist[i][3]-klist[i+1][3])*reciplatt[2][0],2)+
		   pow((klist[i][1]-klist[i+1][1])*reciplatt[0][1]+(klist[i][2]-klist[i+1][2])*reciplatt[1][1]+(klist[i][3]-klist[i+1][3])*reciplatt[2][1],2)+
		   pow((klist[i][1]-klist[i+1][1])*reciplatt[0][2]+(klist[i][2]-klist[i+1][2])*reciplatt[1][2]+(klist[i][3]-klist[i+1][3])*reciplatt[2][2],2))/
              intvl);
      if (np[i]==0) np[i]=1;
    }
    np[nk-1]=1;
    totnkpts=1;
    for (i=0; i<nk-1; i++) totnkpts+=np[i];
  }

  for (i=0; i<3; i++) free(reciplatt[i]);
  free(reciplatt);
}



















