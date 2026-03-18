/**********************************************************************
  Poisson.c:
  
     Poisson.c is a subrutine to solve Poisson's equation using
     fast Fourier transformation.

  Log of Poisson.c:

     22/Nov/2001  Released by T. Ozaki
     06/Apr/2012  Rewritten by T. Ozaki

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <fftw3.h> 


static void Inverse_FFT_Poisson(double *ReRhor, double *ImRhor, 
                                double *ReRhok, double *ImRhok);

static void FFT_Poisson(double *ReRhor, double *ImRhor, 
                        double *ReRhok, double *ImRhok);  

double Poisson(int fft_charge_flag,
               double *ReRhok, double *ImRhok)
{ 
  int k1,k2,k3;
  int N2D,GNs,GN,BN_CB,BN_AB;
  int N3[4];
  double time0,Rc;
  double tmp0,sk1,sk2,sk3;
  double Gx,Gy,Gz,fac_invG2;
  double TStime,TEtime,etime;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID && 0<level_stdout){
    printf("<Poisson>  Poisson's equation using FFT...\n");
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /****************************************************
            FFT of difference charge density 
  ****************************************************/

  if (fft_charge_flag<=1 && scf_coulomb_cutoff_CoreHole==1){
    etime = FFT_Density(5,ReRhok,ImRhok);
  }
  else if (fft_charge_flag==1 || fft_charge_flag==2){
    etime = FFT_Density(0,ReRhok,ImRhok);
  }  

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

    /* spherical Coulomb cutoff */ 

    if ( (scf_coulomb_cutoff==1 && fft_charge_flag!=2) || fft_charge_flag==11 ){

      Rc = 0.5*Shortest_CellVec;
      fac_invG2 = tmp0/(Gx*Gx + Gy*Gy + Gz*Gz)*(1.0-cos(sqrt(Gx*Gx+Gy*Gy+Gz*Gz)*Rc));

      if (k1==0 && k2==0 && k3==0){
	ReRhok[BN_CB] *= 2.0*PI*Rc*Rc/(double)(Ngrid1*Ngrid2*Ngrid3);
	ImRhok[BN_CB] *= 2.0*PI*Rc*Rc/(double)(Ngrid1*Ngrid2*Ngrid3);
      }
      else{
	ReRhok[BN_CB] *= fac_invG2;
	ImRhok[BN_CB] *= fac_invG2;
      }
    }

    /* no Coulomb cutoff */ 

    else {

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
  }  

  /****************************************************
        find the Hartree potential in real space
  ****************************************************/
  
  Get_Value_inReal(0,dVHart_Grid_B,dVHart_Grid_B,ReRhok,ImRhok);

  /****************************************************
    if (fft_charge_flag==2),
    copy the difference charge Hartree potential
    to dVHart_Periodic_Grid_B.
  ****************************************************/

  if (fft_charge_flag==2){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      dVHart_Periodic_Grid_B[BN_AB] = dVHart_Grid_B[BN_AB];
    }
  }

  /****************************************************
   Periodic part of dVHart potential is added 
   to dVHart_Grid in the core hole calculation.
  ****************************************************/

  if (fft_charge_flag<=1 && scf_coulomb_cutoff_CoreHole==1){

    for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
      dVHart_Grid_B[BN_AB] += dVHart_Periodic_Grid_B[BN_AB];
    }
  }

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}




void FFT_Poisson(double *ReRhor, double *ImRhor, 
                 double *ReRhok, double *ImRhok) 
{
  static int firsttime=1;
  int i,BN_AB,BN_CB,BN_CA,gp,NN_S,NN_R;
  double *array0,*array1;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_proc, Etime_proc;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

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

  array0 = (double*)malloc(sizeof(double)*2*GP_B_AB2CA_S[NN_B_AB2CA_S]); 
  array1 = (double*)malloc(sizeof(double)*2*GP_B_AB2CA_R[NN_B_AB2CA_R]); 

  request_send = malloc(sizeof(MPI_Request)*NN_B_AB2CA_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B_AB2CA_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B_AB2CA_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B_AB2CA_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B_AB2CA_R; ID++){

    IDR = ID_NN_B_AB2CA_R[ID];
    gp = GP_B_AB2CA_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &array1[2*gp], Num_Rcv_Grid_B_AB2CA[IDR]*2, 
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++; 
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B_AB2CA_S; ID++){

    IDS = ID_NN_B_AB2CA_S[ID];
    gp = GP_B_AB2CA_S[ID];

    if (IDS!=myid){

      for (i=0; i<Num_Snd_Grid_B_AB2CA[IDS]; i++){
        BN_AB = Index_Snd_Grid_B_AB2CA[IDS][i];
        array0[2*gp+2*i  ] = ReRhor[BN_AB];
        array0[2*gp+2*i+1] = ImRhor[BN_AB];
      }     

      MPI_Isend( &array0[2*gp], Num_Snd_Grid_B_AB2CA[IDS]*2, 
    	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;  
    }
  }
  
  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  /* copy them to ReRhok and ImRhok */

  for (ID=0; ID<NN_B_AB2CA_R; ID++){

    IDR = ID_NN_B_AB2CA_R[ID];
    gp = GP_B_AB2CA_R[ID];

    if (IDR!=myid){
      for (i=0; i<Num_Rcv_Grid_B_AB2CA[IDR]; i++){
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_CA] = array1[2*gp+2*i  ];
	ImRhok[BN_CA] = array1[2*gp+2*i+1];
      }
    }

    else{
      for (i=0; i<Num_Snd_Grid_B_AB2CA[IDR]; i++){
	BN_AB = Index_Snd_Grid_B_AB2CA[IDR][i];
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_CA] = ReRhor[BN_AB];
	ImRhok[BN_CA] = ImRhor[BN_AB];
      }
    }
  }

  free(array0);
  free(array1);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

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
      ReRhok[BN_CA+i] = out[i][0];
      ImRhok[BN_CA+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time FFT-B  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ MPI: CA to CB partitions ------------------*/

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime_proc);
  }

  array0 = (double*)malloc(sizeof(double)*2*GP_B_CA2CB_S[NN_B_CA2CB_S]); 
  array1 = (double*)malloc(sizeof(double)*2*GP_B_CA2CB_R[NN_B_CA2CB_R]); 

  request_send = malloc(sizeof(MPI_Request)*NN_B_CA2CB_S);
  request_recv = malloc(sizeof(MPI_Request)*NN_B_CA2CB_R);
  stat_send = malloc(sizeof(MPI_Status)*NN_B_CA2CB_S);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B_CA2CB_R);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B_CA2CB_R; ID++){

    IDR = ID_NN_B_CA2CB_R[ID];
    gp = GP_B_CA2CB_R[ID];

    if (IDR!=myid){ 
      MPI_Irecv( &array1[2*gp], Num_Rcv_Grid_B_CA2CB[IDR]*2, 
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++; 
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B_CA2CB_S; ID++){

    IDS = ID_NN_B_CA2CB_S[ID];
    gp = GP_B_CA2CB_S[ID];

    if (IDS!=myid){

      for (i=0; i<Num_Snd_Grid_B_CA2CB[IDS]; i++){
        BN_CA = Index_Snd_Grid_B_CA2CB[IDS][i];
        array0[2*gp+2*i  ] = ReRhok[BN_CA];
        array0[2*gp+2*i+1] = ImRhok[BN_CA];
      }     

      MPI_Isend( &array0[2*gp], Num_Snd_Grid_B_CA2CB[IDS]*2, 
  	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;  
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  /* copy them to ReRhor and ImRhor */

  for (ID=0; ID<NN_B_CA2CB_R; ID++){

    IDR = ID_NN_B_CA2CB_R[ID];
    gp = GP_B_CA2CB_R[ID];

    if (IDR!=myid){
      for (i=0; i<Num_Rcv_Grid_B_CA2CB[IDR]; i++){
        BN_CB = Index_Rcv_Grid_B_CA2CB[IDR][i];
        ReRhor[BN_CB] = array1[2*gp+2*i  ];
        ImRhor[BN_CB] = array1[2*gp+2*i+1];
      }
    }

    else{
      for (i=0; i<Num_Snd_Grid_B_CA2CB[IDR]; i++){
	BN_CA = Index_Snd_Grid_B_CA2CB[IDR][i];
	BN_CB = Index_Rcv_Grid_B_CA2CB[IDR][i];
	ReRhor[BN_CB] = ReRhok[BN_CA];
	ImRhor[BN_CB] = ImRhok[BN_CA];
      }
    }
  }

  free(array0);
  free(array1);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Etime_proc);
    printf("myid=%2d  Time MPI: CA to CB = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ FFT along the A-axis in the CB partition ------------------*/

  if (measure_time==1) dtime(&Stime_proc);

  p = fftw_plan_dft_1d(Ngrid1,in,out,-1,FFTW_ESTIMATE);

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    for (i=0; i<Ngrid1; i++){
      in[i][0] = ReRhor[BN_CB+i];
      in[i][1] = ImRhor[BN_CB+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid1; i++){
      ReRhok[BN_CB+i] = out[i][0];
      ImRhok[BN_CB+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  
  fftw_cleanup();

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time FFT-A  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /****************************************************
    freeing of arrays:
  ****************************************************/

  fftw_free(in);
  fftw_free(out);

  /* PrintMemory */

  if (firsttime) {

    if (GP_B_CA2CB_S[NN_B_CA2CB_S]<GP_B_AB2CA_S[NN_B_AB2CA_S]){
      PrintMemory("Poisson: array0",sizeof(double)*2*GP_B_AB2CA_S[NN_B_AB2CA_S],NULL);
    } 
    else{
      PrintMemory("Poisson: array0",sizeof(double)*2*GP_B_CA2CB_S[NN_B_CA2CB_S],NULL);
    }

    if (GP_B_CA2CB_R[NN_B_CA2CB_R]<GP_B_AB2CA_R[NN_B_AB2CA_R]){
      PrintMemory("Poisson: array1",sizeof(double)*2*GP_B_AB2CA_R[NN_B_AB2CA_R],NULL);
    } 
    else{
      PrintMemory("Poisson: array1",sizeof(double)*2*GP_B_CA2CB_R[NN_B_CA2CB_R],NULL);
    }
 
    if (NN_B_CA2CB_S<NN_B_AB2CA_S){
      PrintMemory("Poisson: request_send",sizeof(MPI_Request)*NN_B_AB2CA_S,NULL);
      PrintMemory("Poisson: stat_send   ",sizeof(MPI_Status)*NN_B_AB2CA_S,NULL);
    }
    else{
      PrintMemory("Poisson: request_send",sizeof(MPI_Request)*NN_B_CA2CB_S,NULL);
      PrintMemory("Poisson: stat_send   ",sizeof(MPI_Status)*NN_B_CA2CB_S,NULL);
    } 

    if (NN_B_CA2CB_R<NN_B_AB2CA_R){
      PrintMemory("Poisson: request_recv",sizeof(MPI_Request)*NN_B_AB2CA_R,NULL);
      PrintMemory("Poisson: stat_recv   ",sizeof(MPI_Status)*NN_B_AB2CA_R,NULL);
    }
    else{
      PrintMemory("Poisson: request_recv",sizeof(MPI_Request)*NN_B_CA2CB_R,NULL);
      PrintMemory("Poisson: stat_recv   ",sizeof(MPI_Status)*NN_B_CA2CB_R,NULL);
    } 

    /* turn off firsttime flag */
    firsttime=0;
  }

}




void Inverse_FFT_Poisson(double *ReRhor, double *ImRhor, 
                         double *ReRhok, double *ImRhok) 
{
  int i,BN_AB,BN_CB,BN_CA,gp,NN_S,NN_R;
  double *array0,*array1;
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_proc, Etime_proc;
  MPI_Status stat;
  MPI_Request request;
  MPI_Status *stat_send;
  MPI_Status *stat_recv;
  MPI_Request *request_send;
  MPI_Request *request_recv;

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

  /*------------------ Inverse FFT along the A-axis in the CB partition ------------------*/

  if (measure_time==1) dtime(&Stime_proc);

  p = fftw_plan_dft_1d(Ngrid1,in,out,1,FFTW_ESTIMATE);

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    for (i=0; i<Ngrid1; i++){
      in[i][0] = ReRhok[BN_CB+i];
      in[i][1] = ImRhok[BN_CB+i];
    }

    fftw_execute(p);

    for (i=0; i<Ngrid1; i++){
      ReRhok[BN_CB+i] = out[i][0];
      ImRhok[BN_CB+i] = out[i][1];
    }
  }

  fftw_destroy_plan(p);  

  if (measure_time==1){
    dtime(&Etime_proc);
    printf("myid=%2d  Time Inverse FFT-A  = %15.12f\n",myid,Etime_proc-Stime_proc);
  }

  /*------------------ MPI: CB to CA partitions ------------------*/

  if (measure_time==1){
    MPI_Barrier(mpi_comm_level1);
    dtime(&Stime_proc);
  }

  array0 = (double*)malloc(sizeof(double)*2*GP_B_CA2CB_R[NN_B_CA2CB_R]); 
  array1 = (double*)malloc(sizeof(double)*2*GP_B_CA2CB_S[NN_B_CA2CB_S]); 

  request_send = malloc(sizeof(MPI_Request)*NN_B_CA2CB_R);
  request_recv = malloc(sizeof(MPI_Request)*NN_B_CA2CB_S);
  stat_send = malloc(sizeof(MPI_Status)*NN_B_CA2CB_R);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B_CA2CB_S);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B_CA2CB_S; ID++){

    IDR = ID_NN_B_CA2CB_S[ID];
    gp = GP_B_CA2CB_S[ID];

    if (IDR!=myid){

      MPI_Irecv( &array1[2*gp], Num_Snd_Grid_B_CA2CB[IDR]*2, 
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B_CA2CB_R; ID++){

    IDS = ID_NN_B_CA2CB_R[ID];
    gp = GP_B_CA2CB_R[ID];

    if (IDS!=myid){ 

      for (i=0; i<Num_Rcv_Grid_B_CA2CB[IDS]; i++){
        BN_CB = Index_Rcv_Grid_B_CA2CB[IDS][i];
        array0[2*gp+2*i  ] = ReRhok[BN_CB];
        array0[2*gp+2*i+1] = ImRhok[BN_CB];
      }     

      MPI_Isend( &array0[2*gp], Num_Rcv_Grid_B_CA2CB[IDS]*2, 
  	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  /* copy them to ReRhor and ImRhor */

  for (ID=0; ID<NN_B_CA2CB_S; ID++){

    IDR = ID_NN_B_CA2CB_S[ID];
    gp = GP_B_CA2CB_S[ID];

    if (IDR!=myid){
      for (i=0; i<Num_Snd_Grid_B_CA2CB[IDR]; i++){
        BN_CA = Index_Snd_Grid_B_CA2CB[IDR][i];
        ReRhor[BN_CA] = array1[2*gp+2*i  ];
        ImRhor[BN_CA] = array1[2*gp+2*i+1];
      }
    }
    else{
      for (i=0; i<Num_Rcv_Grid_B_CA2CB[IDR]; i++){
	BN_CB = Index_Rcv_Grid_B_CA2CB[IDR][i];
	BN_CA = Index_Snd_Grid_B_CA2CB[IDR][i];
	ReRhor[BN_CA] = ReRhok[BN_CB];
	ImRhor[BN_CA] = ImRhok[BN_CB];
      }
    }
  }

  free(array0);
  free(array1);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

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

  array0 = (double*)malloc(sizeof(double)*2*GP_B_AB2CA_R[NN_B_AB2CA_R]); 
  array1 = (double*)malloc(sizeof(double)*2*GP_B_AB2CA_S[NN_B_AB2CA_S]); 

  request_send = malloc(sizeof(MPI_Request)*NN_B_AB2CA_R);
  request_recv = malloc(sizeof(MPI_Request)*NN_B_AB2CA_S);
  stat_send = malloc(sizeof(MPI_Status)*NN_B_AB2CA_R);
  stat_recv = malloc(sizeof(MPI_Status)*NN_B_AB2CA_S);

  NN_S = 0;
  NN_R = 0;

  /* MPI_Irecv */

  for (ID=0; ID<NN_B_AB2CA_S; ID++){

    IDR = ID_NN_B_AB2CA_S[ID];
    gp = GP_B_AB2CA_S[ID];

    if (IDR!=myid){

      MPI_Irecv( &array1[2*gp], Num_Snd_Grid_B_AB2CA[IDR]*2, 
                 MPI_DOUBLE, IDR, tag, mpi_comm_level1, &request_recv[NN_R]);
      NN_R++;
    }
  }

  /* MPI_Isend */

  for (ID=0; ID<NN_B_AB2CA_R; ID++){

    IDS = ID_NN_B_AB2CA_R[ID];
    gp = GP_B_AB2CA_R[ID];

    if (IDS!=myid){ 

      for (i=0; i<Num_Rcv_Grid_B_AB2CA[IDS]; i++){
        BN_CB = Index_Rcv_Grid_B_AB2CA[IDS][i];
        array0[2*gp+2*i  ] = ReRhor[BN_CB];
        array0[2*gp+2*i+1] = ImRhor[BN_CB];
      }     

      MPI_Isend( &array0[2*gp], Num_Rcv_Grid_B_AB2CA[IDS]*2, 
  	         MPI_DOUBLE, IDS, tag, mpi_comm_level1, &request_send[NN_S]);
      NN_S++;
    }
  }

  /* MPI_Waitall */

  if (NN_S!=0) MPI_Waitall(NN_S,request_send,stat_send);
  if (NN_R!=0) MPI_Waitall(NN_R,request_recv,stat_recv);

  /* copy them to ReRhok and ImRhok */

  for (ID=0; ID<NN_B_AB2CA_S; ID++){

    IDR = ID_NN_B_AB2CA_S[ID];
    gp = GP_B_AB2CA_S[ID];

    if (IDR!=myid){
      for (i=0; i<Num_Snd_Grid_B_AB2CA[IDR]; i++){
	BN_AB = Index_Snd_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_AB] = array1[2*gp+2*i  ];
	ImRhok[BN_AB] = array1[2*gp+2*i+1];
      }
    }
    else {
      for (i=0; i<Num_Rcv_Grid_B_AB2CA[IDR]; i++){
	BN_CA = Index_Rcv_Grid_B_AB2CA[IDR][i];
	BN_AB = Index_Snd_Grid_B_AB2CA[IDR][i];
	ReRhok[BN_AB] = ReRhor[BN_CA];
	ImRhok[BN_AB] = ImRhor[BN_CA];
      }
    }
  }

  free(array0);
  free(array1);

  free(request_send);
  free(request_recv);
  free(stat_send);
  free(stat_recv);

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




double FFT_Density(int den_flag,
                   double *ReRhok, double *ImRhok)
{
  int BN_AB;
  int numprocs,myid;
  double *ReRhor,*ImRhor;
  double TStime,TEtime,time0;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  if (11<den_flag){
    printf("invalid den_flag for FFT_Density\n");
    MPI_Finalize();
    exit(0);
  }

  /* allocation of arrays */

  ReRhor = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImRhor = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

  /* set ReRhor and ImRhor */

  switch(den_flag) {

    case 0:

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Density_Grid_B[0][BN_AB] + Density_Grid_B[1][BN_AB] - 2.0*ADensity_Grid_B[BN_AB]; 
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

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
        ReRhor[BN_AB] = Density_Grid_B[0][BN_AB] + Density_Grid_B[1][BN_AB] - Density_Periodic_Grid_B[BN_AB];
        ImRhor[BN_AB] = 0.0;
      }

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

  /* FFT of Density */

  FFT_Poisson(ReRhor, ImRhor, ReRhok, ImRhok);

  /* freeing of arrays */

  free(ReRhor);
  free(ImRhor);

  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}






void Get_Value_inReal(int complex_flag,
                      double *ReVr, double *ImVr, 
                      double *ReVk, double *ImVk)
{
  int BN_AB;
  double *ReTmpr,*ImTmpr;

  /* allocation of arrays */

  ReTmpr = (double*)malloc(sizeof(double)*My_Max_NumGridB); 
  ImTmpr = (double*)malloc(sizeof(double)*My_Max_NumGridB); 

  /* call Inverse_FFT_Poisson */
 
  Inverse_FFT_Poisson(ReTmpr, ImTmpr, ReVk, ImVk);

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



