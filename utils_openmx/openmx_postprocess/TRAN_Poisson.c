/**********************************************************************
  TRAN_Poisson.c:

     TRAN_Poisson.c is a subroutine to solve Poisson's equation with 
     the boundary conditions using FFT and a finite difference method.

  Log of TRAN_Poisson.c:

     09/July/2008  Released by T.Ozaki

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"
#include "lapack_prototypes.h"

#include <fftw3.h> 

/*S MitsuakiKAWAMURA*/
void FFT2D_Poisson(double *ReRhor, double *ImRhor, 
                          double *ReRhok, double *ImRhok);
/*E MitsuakiKAWAMURA*/
static void Inverse_FFT2D_Poisson(double *ReRhor, double *ImRhor,
				  double *ReRhok, double *ImRhok); 

/* added by mari 09.12.2014 */
static double TRAN_Poisson_FDG(double *ReRhok, double *ImRhok);
static void FFT1D_Poisson(double *ReRhok, double *ImRhok,
                          dcomplex ***VHart_Boundary_a,
                          dcomplex **VHart_Boundary_b);
static void Inverse_FFT1D_Poisson(double *ReRhok, double *ImRhok); 

static double TRAN_Poisson_FD(double *ReRhok, double *ImRhok);
static double TRAN_Poisson_FFT(double *ReRhok, double *ImRhok);


  
double TRAN_Poisson(double *ReRhok, double *ImRhok)
{ 
  double time;

  switch (TRAN_Poisson_flag){

    case 1:
      time = TRAN_Poisson_FD(ReRhok, ImRhok);
    break;

    case 2:
      time = TRAN_Poisson_FFT(ReRhok, ImRhok);
    break;

    /* added by mari 09.12.2014 */
    case 3:
      time = TRAN_Poisson_FDG(ReRhok, ImRhok);
    break;
  }

  return time;
}




double TRAN_Poisson_FFT(double *ReRhok, double *ImRhok)
{
  int i,j,k,BN_AB,k1,k2,k3;
  int N2D,GNs,GN,BN_CB,ip,MN1;
  int LN,BN,CN,N3[4];
  double time0;
  double tmp0,sk1,sk2,sk3;
  double Gx,Gy,Gz,fac_invG2;
  double gateV,cv,cc,a,b,v,v0;
  double v1,x0,x1,x;
  double Av_dVH0,Av_dVH1;
  double MySum_dV0,MySum_dV1;
  double Av_dVH_FFT0,Av_dVH_FFT1; 
  double TStime,TEtime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID) printf("<TRAN_Poisson_FFT>  Solving Poisson's equation...\n");

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
                 FFT of charge density 
  ****************************************************/

  FFT_Density(0,ReRhok,ImRhok);

  /****************************************************
                       4*PI/G2/N^3
  ****************************************************/

  tmp0 = 4.0*PI/(double)(Ngrid1*Ngrid2*Ngrid3);

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){

    GN = BN_CB + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

    if (k1<Ngrid1/2) sk1 = (double)k1;
    else             sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2) sk2 = (double)k2;
    else             sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    fac_invG2 = tmp0/(Gx*Gx + Gy*Gy + Gz*Gz);

    if (k1==0 && k2==0 && k3==0){
      ReRhok[BN_CB] = 0.0;
      ImRhok[BN_CB] = 0.0;
    }
    else{
      ReRhok[BN_CB] *= fac_invG2;
      ImRhok[BN_CB] *= fac_invG2;
    }
  }  

  /****************************************************
        find the Hartree potential in real space
  ****************************************************/
 
  Get_Value_inReal(0,dVHart_Grid_B,NULL,ReRhok,ImRhok);

  /****************************************************
    calculate average value of the difference Hartree 
    potential at the boudaries.
  ****************************************************/

  MySum_dV0 = 0.0;
  MySum_dV1 = 0.0;

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN = BN_AB + GNs;     
    i = GN/(Ngrid2*Ngrid3);    
    j = (GN - i*Ngrid2*Ngrid3)/Ngrid3;
    k = GN - i*Ngrid2*Ngrid3 - j*Ngrid3; 

    if      (i==0)          MySum_dV0 += dVHart_Grid_B[BN_AB]; 
    else if (i==(Ngrid1-1)) MySum_dV1 += dVHart_Grid_B[BN_AB]; 
  }

  MPI_Allreduce(&MySum_dV0, &Av_dVH_FFT0, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  MPI_Allreduce(&MySum_dV1, &Av_dVH_FFT1, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  Av_dVH0 = 0.0;
  Av_dVH1 = 0.0;

  for (j=0; j<Ngrid2; j++){
    for (k=0; k<Ngrid3; k++){

      MN1 = j*Ngrid3 + k;

      Av_dVH0 += dVHart_Grid_e[0][MN1];
      Av_dVH1 += dVHart_Grid_e[1][MN1];
    }
  } 

  Av_dVH_FFT0 = Av_dVH_FFT0/(double)(Ngrid2*Ngrid3);
  Av_dVH_FFT1 = Av_dVH_FFT1/(double)(Ngrid2*Ngrid3);
  Av_dVH0 = Av_dVH0/(double)(Ngrid2*Ngrid3);
  Av_dVH1 = Av_dVH1/(double)(Ngrid2*Ngrid3);

  /****************************************************
                add the boundary condition
  ****************************************************/

  if      (0.1<fabs(tv[1][1])) ip = 1;
  else if (0.1<fabs(tv[1][2])) ip = 2;
  else if (0.1<fabs(tv[1][3])) ip = 3;

  x0 = Grid_Origin[ip];
  x1 = (double)Ngrid1*length_gtv[1] + Grid_Origin[ip];

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN = BN_AB + GNs;     
    i = GN/(Ngrid2*Ngrid3);    
    j = (GN - i*Ngrid2*Ngrid3)/Ngrid3;
    k = GN - i*Ngrid2*Ngrid3 - j*Ngrid3; 

    x = (double)i*length_gtv[1] + Grid_Origin[ip];
    v0 = Av_dVH0 - Av_dVH_FFT0;
    v1 = Av_dVH1 - Av_dVH_FFT1;
    a = (v1 - v0)/(x1 - x0);
    b = v0 - a*x0;
    v = a*x + b;

    dVHart_Grid_B[BN_AB] += v;
  }

  /****************************************************
                 apply the gate voltage
  ****************************************************/

  if      (0.1<fabs(tv[1][1])) ip = 1;
  else if (0.1<fabs(tv[1][2])) ip = 2;
  else if (0.1<fabs(tv[1][3])) ip = 3;

  cc = 0.0;
  for (i=1; i<=Catomnum; i++){
    cc += Gxyz[Latomnum+i][ip];
  }  
  cc /= (double)Catomnum;

  a = 1.0*( fabs(tv[1][ip]) - fabs(Left_tv[1][ip]) - fabs(Right_tv[1][ip]) );

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN = BN_AB + GNs;     
    i = GN/(Ngrid2*Ngrid3);    
    j = (GN - i*Ngrid2*Ngrid3)/Ngrid3;
    k = GN - i*Ngrid2*Ngrid3 - j*Ngrid3; 

    cv = (double)i*length_gtv[1] + Grid_Origin[ip];
    /* modified by mari 12.22.2014 */
    gateV = tran_gate_voltage[0]*exp( -pow( (cv-cc)/a, 8.0) );
    dVHart_Grid_B[BN_AB] += gateV;
  }

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}




double TRAN_Poisson_FD(double *ReRhok, double *ImRhok)
{ 
  int i,j,k,k1,k2,k3,ip;
  int GN,BN_AB,GNs,BN_CB,N2D;
  double sk1,sk2,sk3,Gx,Gy,Gz;
  double da2,Gpara2,a,cc,cv,gateV;
  double TStime,TEtime,etime;
  int myid,numprocs;
  dcomplex *DL,*D,*DU,*B;
  INTEGER n,nrhs,ldb,info;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID) printf("<TRAN_Poisson_FD>  Solving Poisson's equation...\n");

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
                 allocation of arrays
  ****************************************************/

  DL = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);
  D  = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);
  DU = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);
  B  = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1);

  /****************************************************
          FFT of charge density on the b-c plane
  ****************************************************/
  
  etime = FFT2D_Density(0,ReRhok,ImRhok);

  /****************************************************
          solve finite difference equations
  ****************************************************/
        
  da2 = Dot_Product(gtv[1],gtv[1]);
  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    GN = BN_CB + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    
    if (k2<Ngrid2/2) sk2 = (double)k2;
    else             sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk2*rtv[2][3] + sk3*rtv[3][3];
    Gpara2 = Gx*Gx + Gy*Gy + Gz*Gz;

    for (k1=0; k1<(Ngrid1-1); k1++){
      DL[k1].r = 1.0;
      DL[k1].i = 0.0;
      DU[k1].r = 1.0;
      DU[k1].i = 0.0;
    }         

    for (k1=0; k1<(Ngrid1-0); k1++){
      D[k1].r = -2.0 - da2*Gpara2;
      D[k1].i = 0.0;
    }

    /* scale terms with Gpara=0 */ 

    if (k2==0 && k3==0){
      for (k1=0; k1<Ngrid1; k1++){
	ReRhok[BN_CB+k1] *= TRAN_Poisson_Gpara_Scaling;
      }
    }

    /* set B */

    for (k1=0; k1<(Ngrid1-0); k1++){
      B[k1].r = -4.0*PI*da2*ReRhok[BN_CB+k1];
      B[k1].i = -4.0*PI*da2*ImRhok[BN_CB+k1];
    }

    /* add the boundary condition */

    B[0       ].r -= VHart_Boundary[0][Ngrid1_e[0]-1][k2][k3].r;
    B[0       ].i -= VHart_Boundary[0][Ngrid1_e[0]-1][k2][k3].i;
    B[Ngrid1-1].r -= VHart_Boundary[1][0            ][k2][k3].r;
    B[Ngrid1-1].i -= VHart_Boundary[1][0            ][k2][k3].i;

    /* solve the linear equation */ 

    n = Ngrid1-0;
    nrhs = 1;
    ldb = Ngrid1-0;
     
    F77_NAME(zgtsv,ZGTSV)(&n, &nrhs, DL, D, DU, B, &ldb, &info);

    /* store B to ReRhok and ImRhok */

    for (k1=0; k1<(Ngrid1-0); k1++){
      ReRhok[BN_CB+k1] = B[k1].r;
      ImRhok[BN_CB+k1] = B[k1].i;
    }

  } /* BN2 */

  /****************************************************
        find the Hartree potential in real space
  ****************************************************/
 
  Get_Value_inReal2D(0,dVHart_Grid_B,NULL,ReRhok,ImRhok);

  /****************************************************
               apply the gate voltage
  ****************************************************/

  if      (0.1<fabs(tv[1][1])) ip = 1;
  else if (0.1<fabs(tv[1][2])) ip = 2;
  else if (0.1<fabs(tv[1][3])) ip = 3;

  cc = 0.0;
  for (i=1; i<=Catomnum; i++){
    cc += Gxyz[Latomnum+i][ip];
  }  
  cc /= (double)Catomnum;

  a = 1.0*( fabs(tv[1][ip]) - fabs(Left_tv[1][ip]) - fabs(Right_tv[1][ip]) );

  N2D = Ngrid1*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid3;

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){

    GN = BN_AB + GNs;     
    i = GN/(Ngrid2*Ngrid3);    
    j = (GN - i*Ngrid2*Ngrid3)/Ngrid3;
    k = GN - i*Ngrid2*Ngrid3 - j*Ngrid3; 

    cv = (double)i*length_gtv[1] + Grid_Origin[ip];
    /* modified by mari 12.22.2014 */
    gateV = tran_gate_voltage[0]*exp( -pow( (cv-cc)/a, 8.0) );
    dVHart_Grid_B[BN_AB] += gateV;
  }

  /****************************************************
                   freeing of arrays
  ****************************************************/

  free(DL);
  free(D);
  free(DU);
  free(B);

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  return (TEtime-TStime);
}




double FFT2D_Density(int den_flag, 
                     double *ReRhok, double *ImRhok)
{
  int BN_AB,BN_CB;
  int numprocs,myid;
  double tmp0;
  double *ReRhor,*ImRhor;
  double TStime,TEtime,time0;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  if (4<den_flag){
    printf("invalid den_flag for FFT2D_Density\n");
    MPI_Finalize();
    exit(0);
  }

  /* allocation of arrays */

  ReRhor = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImRhor = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

  /* set ReRhor and ImRhor */

  switch (den_flag){

    case 0:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Density_Grid_B[0][BN_AB] + Density_Grid_B[1][BN_AB]
                       -2.0*ADensity_Grid_B[BN_AB]; 
        ImRhor[BN_AB] = 0.0;
      }

    break;

    case 1:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Density_Grid_B[0][BN_AB];
        ImRhor[BN_AB] = 0.0;
      }

    break;

    case 2:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Density_Grid_B[1][BN_AB];
        ImRhor[BN_AB] = 0.0;
      }

    break;

    case 3:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = 2.0*ADensity_Grid_B[BN_AB];
        ImRhor[BN_AB] = 0.0;
      }

    break;

    case 4:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Density_Grid_B[2][BN_AB];
        ImRhor[BN_AB] = Density_Grid_B[3][BN_AB];
      }

    break;

    case 5:
      /* not defined */

    break;

    case 6:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Vpot_Grid_B[0][BN_AB];
        ImRhor[BN_AB] = 0.0;
      }

    break;

    case 7:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Vpot_Grid_B[1][BN_AB];
        ImRhor[BN_AB] = 0.0;
      }

    break;

    case 8:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Vpot_Grid_B[2][BN_AB];
        ImRhor[BN_AB] = Vpot_Grid_B[3][BN_AB];
      }

    break;

  }
  
  /****************************************************
                       FFT of Dens
  ****************************************************/

  FFT2D_Poisson(ReRhor, ImRhor, ReRhok, ImRhok);

  tmp0 = 1.0/(double)(Ngrid3*Ngrid2);
  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB++){
    ReRhok[BN_CB] *= tmp0;
    ImRhok[BN_CB] *= tmp0;
  }

  /* freeing of arrays */

  free(ReRhor);
  free(ImRhor);

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}



void FFT2D_Poisson(double *ReRhor, double *ImRhor, 
                   double *ReRhok, double *ImRhok) 
{
  int i,BN_AB,BN_CB,BN_CA;
  double *array0,*array1;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_proc, Etime_proc;
  MPI_Status stat;
  MPI_Request request;
  fftw_complex *in, *out;
  fftw_plan p;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:
  ****************************************************/

  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 

  /*------------------ FFT along the C-axis in the AB partition ------------------*/

  if (measure_time==1) dtime(&Stime_proc);

  p = fftw_plan_dft_1d(Ngrid3,in,out,-1,FFTW_ESTIMATE);

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB+=Ngrid3){

    for (i=0; i<Ngrid3; i++){
      in[i][0] = ReRhor[BN_AB+i];
      in[i][1] = ImRhor[BN_AB+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid3; i++){
      ReRhor[BN_AB+i] = out[i][0];
      ImRhor[BN_AB+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time FFT-C  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ MPI: AB to CA partitions ------------------*/

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime_proc);
  }

  array0 = (double*)malloc(sizeof(double)*2*Max_Num_Snd_Grid_B_AB2CA); 
  array1 = (double*)malloc(sizeof(double)*2*Max_Num_Rcv_Grid_B_AB2CA); 
  
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /* if ID!=0 */
    if (ID!=0){

      for (i=0; i<Num_Snd_Grid_B_AB2CA[IDS]; i++){
	BN_AB = Index_Snd_Grid_B_AB2CA[IDS][i];
	array0[2*i  ] = ReRhor[BN_AB];
	array0[2*i+1] = ImRhor[BN_AB];
      }

      if (Num_Snd_Grid_B_AB2CA[IDS]!=0){
	MPI_Isend( &array0[0], Num_Snd_Grid_B_AB2CA[IDS]*2, 
		   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Rcv_Grid_B_AB2CA[IDR]!=0){
	MPI_Recv( &array1[0], Num_Rcv_Grid_B_AB2CA[IDR]*2, 
		  MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      }

      if (Num_Snd_Grid_B_AB2CA[IDS]!=0) MPI_Wait(&request,&stat);

      for (i=0; i<Num_Rcv_Grid_B_AB2CA[IDR]; i++){
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_CA] = array1[2*i  ];
	ImRhok[BN_CA] = array1[2*i+1];
      }
    }

    /* if ID==0 */
    else{
      for (i=0; i<Num_Rcv_Grid_B_AB2CA[IDR]; i++){
	BN_AB = Index_Snd_Grid_B_AB2CA[IDS][i];
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_CA] = ReRhor[BN_AB];
	ImRhok[BN_CA] = ImRhor[BN_AB];
      }
    }
  }    

  free(array0);
  free(array1);

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Etime_proc);
    printf("myid=%2d  Time MPI: AB to CB = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ FFT along the B-axis in the CA partition ------------------*/

  if (measure_time==1) dtime(&Stime_proc);

  p = fftw_plan_dft_1d(Ngrid2,in,out,-1,FFTW_ESTIMATE);

  for (BN_CA=0; BN_CA<My_NumGridB_CA; BN_CA+=Ngrid2){

    for (i=0; i<Ngrid2; i++){
      in[i][0] = ReRhok[BN_CA+i];
      in[i][1] = ImRhok[BN_CA+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid2; i++){
      ReRhor[BN_CA+i] = out[i][0];
      ImRhor[BN_CA+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  
  fftw_cleanup();

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time FFT-B  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ MPI: CA to CB partitions ------------------*/

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime_proc);
  }

  array0 = (double*)malloc(sizeof(double)*2*Max_Num_Snd_Grid_B_CA2CB); 
  array1 = (double*)malloc(sizeof(double)*2*Max_Num_Rcv_Grid_B_CA2CB); 
  
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /* if ID!=0 */
    if (ID!=0){

      for (i=0; i<Num_Snd_Grid_B_CA2CB[IDS]; i++){
	BN_CA = Index_Snd_Grid_B_CA2CB[IDS][i];
	array0[2*i  ] = ReRhor[BN_CA];
	array0[2*i+1] = ImRhor[BN_CA];
      }

      if (Num_Snd_Grid_B_CA2CB[IDS]!=0){

	MPI_Isend( &array0[0], Num_Snd_Grid_B_CA2CB[IDS]*2, 
		   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Rcv_Grid_B_CA2CB[IDR]!=0){

	MPI_Recv( &array1[0], Num_Rcv_Grid_B_CA2CB[IDR]*2, 
		  MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      }

      if (Num_Snd_Grid_B_CA2CB[IDS]!=0) MPI_Wait(&request,&stat);

      for (i=0; i<Num_Rcv_Grid_B_CA2CB[IDR]; i++){
	BN_CB = Index_Rcv_Grid_B_CA2CB[IDR][i];
	ReRhok[BN_CB] = array1[2*i  ];
	ImRhok[BN_CB] = array1[2*i+1];
      }
    }

    /* if ID==0 */
    else{
      for (i=0; i<Num_Rcv_Grid_B_CA2CB[IDR]; i++){
	BN_CA = Index_Snd_Grid_B_CA2CB[IDS][i];
	BN_CB = Index_Rcv_Grid_B_CA2CB[IDR][i];
	ReRhok[BN_CB] = ReRhor[BN_CA];
	ImRhok[BN_CB] = ImRhor[BN_CA];
      }
    }

  }    

  free(array0);
  free(array1);

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Etime_proc);
    printf("myid=%2d  Time MPI: CA to CB = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  fftw_free(in);
  fftw_free(out);
}






void Inverse_FFT2D_Poisson(double *ReRhor, double *ImRhor, 
                           double *ReRhok, double *ImRhok) 
{
  int i,BN_AB,BN_CB,BN_CA;
  double *array0,*array1;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_proc, Etime_proc;

  MPI_Status stat;
  MPI_Request request;
  fftw_complex *in, *out;
  fftw_plan p;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
    allocation of arrays:

    fftw_complex  in[List_YOUSO[17]];
    fftw_complex out[List_YOUSO[17]];
  ****************************************************/

  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 

  /*------------------ MPI: CB to CA partitions ------------------*/

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime_proc);
  }

  array0 = (double*)malloc(sizeof(double)*2*Max_Num_Rcv_Grid_B_CA2CB); 
  array1 = (double*)malloc(sizeof(double)*2*Max_Num_Snd_Grid_B_CA2CB); 
  
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /* if ID!=0 */
    if (ID!=0){

      for (i=0; i<Num_Rcv_Grid_B_CA2CB[IDS]; i++){
	BN_CB = Index_Rcv_Grid_B_CA2CB[IDS][i];
	array0[2*i  ] = ReRhok[BN_CB];
	array0[2*i+1] = ImRhok[BN_CB];
      }

      if (Num_Rcv_Grid_B_CA2CB[IDS]!=0){
	MPI_Isend( &array0[0], Num_Rcv_Grid_B_CA2CB[IDS]*2, 
		   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Snd_Grid_B_CA2CB[IDR]!=0){
	MPI_Recv( &array1[0], Num_Snd_Grid_B_CA2CB[IDR]*2, 
		  MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      }

      if (Num_Rcv_Grid_B_CA2CB[IDS]!=0)  MPI_Wait(&request,&stat);

      for (i=0; i<Num_Snd_Grid_B_CA2CB[IDR]; i++){
	BN_CA = Index_Snd_Grid_B_CA2CB[IDR][i];
	ReRhor[BN_CA] = array1[2*i  ];
	ImRhor[BN_CA] = array1[2*i+1];
      }
    }

    /* if ID==0 */
    else{
      for (i=0; i<Num_Snd_Grid_B_CA2CB[IDR]; i++){
	BN_CB = Index_Rcv_Grid_B_CA2CB[IDS][i];
	BN_CA = Index_Snd_Grid_B_CA2CB[IDR][i];
	ReRhor[BN_CA] = ReRhok[BN_CB];
	ImRhor[BN_CA] = ImRhok[BN_CB];
      }
    }
  }    

  free(array0);
  free(array1);

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Etime_proc);
    printf("myid=%2d  Time MPI: CB to CA = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ Inverse FFT along the B-axis in the CA partition ------------------*/

  if (measure_time==1) dtime(&Stime_proc);

  p = fftw_plan_dft_1d(Ngrid2,in,out,1,FFTW_ESTIMATE);

  for (BN_CA=0; BN_CA<My_NumGridB_CA; BN_CA+=Ngrid2){

    for (i=0; i<Ngrid2; i++){
      in[i][0] = ReRhor[BN_CA+i];
      in[i][1] = ImRhor[BN_CA+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid2; i++){
      ReRhor[BN_CA+i] = out[i][0];
      ImRhor[BN_CA+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time Inverse FFT-B  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ MPI: CA to AB partitions ------------------*/

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime_proc);
  }

  array0 = (double*)malloc(sizeof(double)*2*Max_Num_Rcv_Grid_B_AB2CA); 
  array1 = (double*)malloc(sizeof(double)*2*Max_Num_Snd_Grid_B_AB2CA); 
  
  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    /* if ID!=0 */
    if (ID!=0){

      for (i=0; i<Num_Rcv_Grid_B_AB2CA[IDS]; i++){
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDS][i];
	array0[2*i  ] = ReRhor[BN_CA];
	array0[2*i+1] = ImRhor[BN_CA];
      }

      if (Num_Rcv_Grid_B_AB2CA[IDS]!=0){
	MPI_Isend( &array0[0], Num_Rcv_Grid_B_AB2CA[IDS]*2, 
		   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Snd_Grid_B_AB2CA[IDR]!=0){
	MPI_Recv( &array1[0], Num_Snd_Grid_B_AB2CA[IDR]*2, 
		  MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      }

      if (Num_Rcv_Grid_B_AB2CA[IDS]!=0)  MPI_Wait(&request,&stat);

      for (i=0; i<Num_Snd_Grid_B_AB2CA[IDR]; i++){
	BN_AB = Index_Snd_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_AB] = array1[2*i  ];
	ImRhok[BN_AB] = array1[2*i+1];
      }
    }

    /* if ID==0 */
    else{
      for (i=0; i<Num_Snd_Grid_B_AB2CA[IDR]; i++){
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDS][i];
	BN_AB = Index_Snd_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_AB] = ReRhor[BN_CA];
	ImRhok[BN_AB] = ImRhor[BN_CA];
      }
    }

  }    

  free(array0);
  free(array1);

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Etime_proc);
    printf("myid=%2d  Time MPI: CA to AB = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ Inverse FFT along the C-axis in the AB partition ------------------*/

  if (measure_time==1) dtime(&Stime_proc);

  p = fftw_plan_dft_1d(Ngrid3,in,out,1,FFTW_ESTIMATE);

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB+=Ngrid3){

    for (i=0; i<Ngrid3; i++){
      in[i][0] = ReRhok[BN_AB+i];
      in[i][1] = ImRhok[BN_AB+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid3; i++){
      ReRhor[BN_AB+i] = out[i][0];
      ImRhor[BN_AB+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  
  fftw_cleanup();

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time Inverse FFT-C  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /****************************************************
    freeing of arrays:

    fftw_complex  in[List_YOUSO[17]];
    fftw_complex out[List_YOUSO[17]];
  ****************************************************/

  fftw_free(in);
  fftw_free(out);
}





void Get_Value_inReal2D(int complex_flag,
                        double *ReVr, double *ImVr, 
                        double *ReVk, double *ImVk)
{
  int BN_AB;
  double *ReTmpr,*ImTmpr;

  /* allocation of arrays */

  ReTmpr = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpr = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

  /* call Inverse_FFT2D_Poisson */
 
  Inverse_FFT2D_Poisson(ReTmpr, ImTmpr, ReVk, ImVk);

  /* copy ReTmpr and ImTmpr into ReVr and ImVr */

  if (complex_flag==0){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      ReVr[BN_AB] = ReTmpr[BN_AB];
    }  
  }
  else if (complex_flag==1){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      ReVr[BN_AB] = ReTmpr[BN_AB];
      ImVr[BN_AB] = ImTmpr[BN_AB];
    }  
  }

  /* freeing of arrays */

  free(ReTmpr);
  free(ImTmpr);
}


/* beginning: added by mari 09.12.2014 */
double TRAN_Poisson_FDG(double *ReRhok, double *ImRhok)
{ 
  int myid,numprocs;
  int N2D,My_NumGridB_C,GN,GNs;
  double *ReRhok_C,*ImRhok_C;
  dcomplex *A,*B,***VHart_Boundary_a,**VHart_Boundary_b;
  int *IPIV;
  int i,k,k1,k2,k3,ku1,ku2,kl1,kl2;
  double *array0,*array1;
  int ID,IDS,IDR,tag=999,BN_AB,BN_C;
  double da2,db2,sk3,Gx,Gy,Gz,Gpara2;
  INTEGER n,nrhs,lda,ldb,info;
  MPI_Status stat;
  MPI_Request request;
  double TStime,TEtime,etime;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID) printf("<TRAN_Poisson_FDG>  Solving Poisson's equation...\n");

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
                  allocation of arrays
  ****************************************************/


  N2D = Ngrid1*Ngrid2;
  My_NumGridB_C = (((myid+1)*Ngrid3+numprocs-1)/numprocs)*N2D 
                - ((myid*Ngrid3+numprocs-1)/numprocs)*N2D;
  ReRhok_C = (double*)malloc(sizeof(double)*My_NumGridB_C); 
  ImRhok_C = (double*)malloc(sizeof(double)*My_NumGridB_C); 

  A  = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1*Ngrid2*Ngrid1*Ngrid2);
  B  = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid1*Ngrid2);
  IPIV  = (int*)malloc(sizeof(int)*Ngrid1*Ngrid2);

  VHart_Boundary_G[0] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
  VHart_Boundary_G[1] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);

  VHart_Boundary_a = (dcomplex***)malloc(sizeof(dcomplex)*2);
  VHart_Boundary_a[0] = (dcomplex**)malloc(sizeof(dcomplex)*Ngrid2);
  VHart_Boundary_a[1] = (dcomplex**)malloc(sizeof(dcomplex)*Ngrid2);
  for (k2=0; k2<Ngrid2; k2++) {
    VHart_Boundary_a[0][k2] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
    VHart_Boundary_a[1][k2] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
  } 
  VHart_Boundary_b = (dcomplex**)malloc(sizeof(dcomplex)*2);
  VHart_Boundary_b[0] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
  VHart_Boundary_b[1] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);

  for (k3=0; k3<Ngrid3; k3++) {
    VHart_Boundary_G[0][k3].r=tran_gate_voltage[0];
    VHart_Boundary_G[0][k3].i=0.0;
    VHart_Boundary_G[1][k3].r=tran_gate_voltage[1];
    VHart_Boundary_G[1][k3].i=0.0;
  }

  /****************************************************
      FFT of Rho and VHart_Boundary along the c axis
  ****************************************************/

  etime = FFT1D_Density(0,ReRhok,ImRhok,VHart_Boundary_a,VHart_Boundary_b);

/* AB to C (MPI) */

  array0 = (double*)malloc(sizeof(double)*2*Max_Num_Snd_Grid_B_AB2C);
  array1 = (double*)malloc(sizeof(double)*2*Max_Num_Rcv_Grid_B_AB2C);

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      for (i=0; i<Num_Snd_Grid_B_AB2C[IDS]; i++){
        BN_AB = Index_Snd_Grid_B_AB2C[IDS][i];
        array0[2*i  ] = ReRhok[BN_AB];
        array0[2*i+1] = ImRhok[BN_AB];
      }

      if (Num_Snd_Grid_B_AB2C[IDS]!=0){
        MPI_Isend( &array0[0], Num_Snd_Grid_B_AB2C[IDS]*2,
                   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }

      if (Num_Rcv_Grid_B_AB2C[IDR]!=0){
        MPI_Recv( &array1[0], Num_Rcv_Grid_B_AB2C[IDR]*2,
                  MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      }

      if (Num_Snd_Grid_B_AB2C[IDS]!=0) MPI_Wait(&request,&stat);

      for (i=0; i<Num_Rcv_Grid_B_AB2C[IDR]; i++){
        BN_C = Index_Rcv_Grid_B_AB2C[IDR][i];
        ReRhok_C[BN_C] = array1[2*i  ];
        ImRhok_C[BN_C] = array1[2*i+1];
      }
    }

    else{
      for (i=0; i<Num_Rcv_Grid_B_AB2C[IDR]; i++){
        BN_AB = Index_Snd_Grid_B_AB2C[IDS][i];
        BN_C = Index_Rcv_Grid_B_AB2C[IDR][i];
        ReRhok_C[BN_C] = ReRhok[BN_AB];
        ImRhok_C[BN_C] = ImRhok[BN_AB];
      }
    }
  }

  free(array0);
  free(array1);

  /****************************************************
           solve finite difference equations
  ****************************************************/

  da2 = Dot_Product(gtv[1],gtv[1]);
  db2 = Dot_Product(gtv[2],gtv[2]);
  GNs = ((myid*Ngrid3+numprocs-1)/numprocs)*N2D;

  for (BN_C=0; BN_C<My_NumGridB_C; BN_C+=N2D){

    GN = BN_C + GNs;
    k3 = GN/N2D;
    
    if (k3<Ngrid3/2) sk3 = (double)k3;
    else             sk3 = (double)(k3 - Ngrid3);

    Gx = sk3*rtv[3][1];
    Gy = sk3*rtv[3][2]; 
    Gz = sk3*rtv[3][3];
    Gpara2 = Gx*Gx + Gy*Gy + Gz*Gz;

    /* set A */

    for (k=0; k<N2D; k++) {
      for (i=0; i<N2D; i++) {
        A[k*N2D+i].r=0.0;
        A[k*N2D+i].i=0.0;
      }
    }
    for (k=0; k<N2D; k++){
      A[k*N2D+k].r = - 2.0/da2 - 2.0/db2 - Gpara2;
      A[k*N2D+k].i = 0.0;
    }
    for (k2=0; k2<Ngrid2; k2++){
      for (k1=0; k1<Ngrid1; k1++){
        k = k1*Ngrid2 + k2;
        ku1 = (k1+1)*Ngrid2 + k2;
        kl1 = (k1-1)*Ngrid2 + k2;
        ku2 = k1*Ngrid2 + k2+1;
        kl2 = k1*Ngrid2 + k2-1;
        if(k1!=0) A[k*N2D+kl1].r = 1.0/da2;
        if(k1!=0) A[k*N2D+kl1].i = 0.0;
        if(k1!=Ngrid1-1) A[k*N2D+ku1].r = 1.0/da2;
        if(k1!=Ngrid1-1) A[k*N2D+ku1].i = 0.0;
        if(k2!=0) A[k*N2D+kl2].r = 1.0/db2;
        if(k2!=0) A[k*N2D+kl2].i = 0.0;
        if(k2!=Ngrid2-1) A[k*N2D+ku2].r = 1.0/db2;
        if(k2!=Ngrid2-1) A[k*N2D+ku2].i = 0.0;
      }         
    }         

    /* scale terms with Gpara=0 */ 

/*    if (k2==0 && k3==0){
        for (k1=0; k1<Ngrid1; k1++){
          ReRhok[BN_CB+k1] *= TRAN_Poisson_Gpara_Scaling;
        }
      } because default value = 1.0 */

    /* set B */

    for (k=0; k<N2D; k++){
        B[k].r = -4.0*PI*ReRhok_C[BN_C+k];
        B[k].i = -4.0*PI*ImRhok_C[BN_C+k];
    }

    /* add the boundary condition */

    for (k2=0; k2<Ngrid2; k2++){
      k = k2;
      B[k].r -= VHart_Boundary_a[0][k2][k3].r/da2;
      B[k].i -= VHart_Boundary_a[0][k2][k3].i/da2;
      k = (Ngrid1-1)*Ngrid2 + k2;
      B[k].r -= VHart_Boundary_a[1][k2][k3].r/da2;
      B[k].i -= VHart_Boundary_a[1][k2][k3].i/da2;
    }
    for (k1=0; k1<Ngrid1; k1++){
      k = k1*Ngrid2;
      B[k].r -= VHart_Boundary_b[0][k3].r/db2;
      B[k].i -= VHart_Boundary_b[0][k3].i/db2;
      k = k1*Ngrid2 + Ngrid2-1;
      B[k].r -= VHart_Boundary_b[1][k3].r/db2;
      B[k].i -= VHart_Boundary_b[1][k3].i/db2;
    }

    /* solve the linear equation */

    n = N2D;
    nrhs = 1;
    lda = n;
    ldb = n;
     
    F77_NAME(zgesv,ZGESV)(&n, &nrhs, A, &lda, IPIV, B, &ldb, &info);

    /* store B to ReRhor and ImRhor */
    for (k=0; k<N2D; k++){
      ReRhok_C[BN_C+k] = B[k].r;
      ImRhok_C[BN_C+k] = B[k].i;
    }

  }

/* C to AB (MPI) */

  array0 = (double*)malloc(sizeof(double)*2*Max_Num_Rcv_Grid_B_AB2C);
  array1 = (double*)malloc(sizeof(double)*2*Max_Num_Snd_Grid_B_AB2C);

  for (ID=0; ID<numprocs; ID++){

    IDS = (myid + ID) % numprocs;
    IDR = (myid - ID + numprocs) % numprocs;

    if (ID!=0){

      for (i=0; i<Num_Rcv_Grid_B_AB2C[IDS]; i++){
        BN_C = Index_Rcv_Grid_B_AB2C[IDS][i];
        array0[2*i  ] = ReRhok_C[BN_C];
        array0[2*i+1] = ImRhok_C[BN_C];
      }

      if (Num_Rcv_Grid_B_AB2C[IDS]!=0){
        MPI_Isend( &array0[0], Num_Rcv_Grid_B_AB2C[IDS]*2,
                   MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request);
      }


      if (Num_Snd_Grid_B_AB2C[IDR]!=0){
        MPI_Recv( &array1[0], Num_Snd_Grid_B_AB2C[IDR]*2,
                  MPI_DOUBLE, IDR, tag, mpi_comm_level1, &stat);
      }

      if (Num_Rcv_Grid_B_AB2C[IDS]!=0)  MPI_Wait(&request,&stat);

      for (i=0; i<Num_Snd_Grid_B_AB2C[IDR]; i++){
        BN_AB = Index_Snd_Grid_B_AB2C[IDR][i];
        ReRhok[BN_AB] = array1[2*i  ];
        ImRhok[BN_AB] = array1[2*i+1];
      }
    }

    else{
      for (i=0; i<Num_Snd_Grid_B_AB2C[IDR]; i++){
        BN_C = Index_Rcv_Grid_B_AB2C[IDS][i];
        BN_AB = Index_Snd_Grid_B_AB2C[IDR][i];
        ReRhok[BN_AB] = ReRhok_C[BN_C];
        ImRhok[BN_AB] = ImRhok_C[BN_C];
      }
    }

  }

  free(array0);
  free(array1);

  /****************************************************
         find the Hartree potential in real space
  ****************************************************/

  Get_Value_inReal1D(0,dVHart_Grid_B,NULL,ReRhok,ImRhok);
 
  /****************************************************
                    freeing of arrays
  ****************************************************/

  free(ReRhok_C);
  free(ImRhok_C);
  free(A);
  free(B);
  free(IPIV);
  free(VHart_Boundary_G[0]);
  free(VHart_Boundary_G[1]);
  free(VHart_Boundary_b[0]);
  free(VHart_Boundary_b[1]);
  for (k2=0; k2<Ngrid2; k2++) {
    free(VHart_Boundary_a[0][k2]);
    free(VHart_Boundary_a[1][k2]);
  }
  free(VHart_Boundary_a[0]);
  free(VHart_Boundary_a[1]);

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  return (TEtime-TStime);
}

double FFT1D_Density(int den_flag, 
                     double *ReRhok, double *ImRhok,
                     dcomplex ***VHart_Boundary_a,
                     dcomplex **VHart_Boundary_b)
{
  int BN_AB,k3;
  int numprocs,myid;
  double tmp0;
  double TStime,TEtime,time0;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&TStime);

  if (4<den_flag){
    printf("invalid den_flag for FFT1D_Density\n");
    MPI_Finalize();
    exit(0);
  }

  /* set ReRhor and ImRhor */

  switch (den_flag){

    case 0:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhok[BN_AB] = Density_Grid_B[0][BN_AB] + Density_Grid_B[1][BN_AB]
                       -2.0*ADensity_Grid_B[BN_AB]; 
        ImRhok[BN_AB] = 0.0;
      }

    break;

    case 1:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhok[BN_AB] = Density_Grid_B[0][BN_AB];
        ImRhok[BN_AB] = 0.0;
      }

    break;

    case 2:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhok[BN_AB] = Density_Grid_B[1][BN_AB];
        ImRhok[BN_AB] = 0.0;
      }

    break;

    case 3:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhok[BN_AB] = 2.0*ADensity_Grid_B[BN_AB];
        ImRhok[BN_AB] = 0.0;
      }

    break;

    case 4:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhok[BN_AB] = Density_Grid_B[2][BN_AB];
        ImRhok[BN_AB] = Density_Grid_B[3][BN_AB];
      }

    break;
  }
  
  /****************************************************
                        FFT of Dens
  ****************************************************/

  FFT1D_Poisson(ReRhok, ImRhok, VHart_Boundary_a, VHart_Boundary_b);

  tmp0 = 1.0/(double)(Ngrid3);
  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
    ReRhok[BN_AB] *= tmp0;
    ImRhok[BN_AB] *= tmp0;
  }
  for (k3=0; k3<Ngrid3; k3++){
    VHart_Boundary_b[0][k3].r *= tmp0;
    VHart_Boundary_b[0][k3].i *= tmp0;
    VHart_Boundary_b[1][k3].r *= tmp0;
    VHart_Boundary_b[1][k3].i *= tmp0;
  }

  /* for time */
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}

void FFT1D_Poisson(double *ReRhok, double *ImRhok,
                   dcomplex ***VHart_Boundary_a, 
                   dcomplex **VHart_Boundary_b) 
{
  int i,k3,BN_AB;
  fftw_complex *in, *out;
  fftw_plan p;
  
  /****************************************************
    allocation of arrays:
  ****************************************************/

  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 

/*------------- FFT of Rhok along the C-axis -------------*/

  p = fftw_plan_dft_1d(Ngrid3,in,out,-1,FFTW_ESTIMATE);

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB+=Ngrid3){

    for (i=0; i<Ngrid3; i++){
      in[i][0] = ReRhok[BN_AB+i];
      in[i][1] = ImRhok[BN_AB+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid3; i++){
      ReRhok[BN_AB+i] = out[i][0];
      ImRhok[BN_AB+i] = out[i][1];
    }

  }

/*------- FFT of VHart_Boundary_G along the C-axis -------*/

  for (i=0; i<Ngrid3; i++){
    in[i][0] = VHart_Boundary_G[0][i].r;
    in[i][1] = VHart_Boundary_G[0][i].i;
  }

  fftw_execute(p);

  for (i=0; i<Ngrid3; i++){
    VHart_Boundary_b[0][i].r = out[i][0];
    VHart_Boundary_b[0][i].i = out[i][1];
  }

  for (i=0; i<Ngrid3; i++){
    in[i][0] = VHart_Boundary_G[1][i].r;
    in[i][1] = VHart_Boundary_G[1][i].i;
  }

  fftw_execute(p);

  for (i=0; i<Ngrid3; i++){
    VHart_Boundary_b[1][i].r = out[i][0];
    VHart_Boundary_b[1][i].i = out[i][1];
  }

  fftw_destroy_plan(p);  

/*---- Inverse FFT of VHart_Boundary along the B-axis ----*/

  p = fftw_plan_dft_1d(Ngrid2,in,out,1,FFTW_ESTIMATE);

  for (k3=0; k3<Ngrid3; k3+=1){

    for (i=0; i<Ngrid2; i++){
      in[i][0] = VHart_Boundary[0][Ngrid1_e[0]-1][i][k3].r;
      in[i][1] = VHart_Boundary[0][Ngrid1_e[0]-1][i][k3].i;
    }

    fftw_execute(p);

    for (i=0; i<Ngrid2; i++){
      VHart_Boundary_a[0][i][k3].r = out[i][0];
      VHart_Boundary_a[0][i][k3].i = out[i][1];
    }
  }

  for (k3=0; k3<Ngrid3; k3+=1){

    for (i=0; i<Ngrid2; i++){
      in[i][0] = VHart_Boundary[1][0][i][k3].r;
      in[i][1] = VHart_Boundary[1][0][i][k3].i;
    }

    fftw_execute(p);

    for (i=0; i<Ngrid2; i++){
      VHart_Boundary_a[1][i][k3].r = out[i][0];
      VHart_Boundary_a[1][i][k3].i = out[i][1];
    }
  }

  fftw_destroy_plan(p);  
  fftw_cleanup();

  /****************************************************
    freeing of arrays:
  ****************************************************/

  fftw_free(in);
  fftw_free(out);
}

void Inverse_FFT1D_Poisson(double *ReRhok, double *ImRhok) 
{
  int i,BN_AB;

  fftw_complex *in, *out;
  fftw_plan p;
  
  /****************************************************
    allocation of arrays:
  ****************************************************/

  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 

/*--------- Inverse FFT of Rhok along the C-axis ---------*/

  p = fftw_plan_dft_1d(Ngrid3,in,out,1,FFTW_ESTIMATE);

  for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB+=Ngrid3){

    for (i=0; i<Ngrid3; i++){
      in[i][0] = ReRhok[BN_AB+i];
      in[i][1] = ImRhok[BN_AB+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid3; i++){
      ReRhok[BN_AB+i] = out[i][0];
      ImRhok[BN_AB+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  
  fftw_cleanup();

  /****************************************************
    freeing of arrays:
  ****************************************************/

  fftw_free(in);
  fftw_free(out);
}

void Get_Value_inReal1D(int complex_flag,
                        double *ReVr, double *ImVr, 
                        double *ReVk, double *ImVk)
{
  int BN_AB;

  /* allocation of arrays */

  /* call Inverse_FFT1D_Poisson */
 
  Inverse_FFT1D_Poisson(ReVk, ImVk);
  
  if (complex_flag==0){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      ReVr[BN_AB] = ReVk[BN_AB];
    }  
  }
  else if (complex_flag==1){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      ReVr[BN_AB] = ReVk[BN_AB];
      ImVr[BN_AB] = ImVk[BN_AB];
    }  
  }

}
/* ending: added by mari 09.12.2014 */
