/**********************************************************************
  TRAN_Set_IntegPath.c:

  TRAN_Set_IntegPath.c is a subroutine to set integral paths for calculating
  density matrix by ther NEGF method.

  Log of TRAN_Set_IntegPath.c:

     24/July/2008  Released by T.Ozaki

***********************************************************************/
    /* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "openmx_common.h"
#include "tran_prototypes.h"
#include "tran_variables.h"
#include "lapack_prototypes.h"

#define EPS 1.0e-14

static void zero_fermi( int N, dcomplex *zp0, dcomplex *Rp0 );
static void TRAN_Set_IntegPath_CF( double kBvalue, double Electronic_Temperature );

void CF( MPI_Comm comm1, double TRAN_eV2Hartree,
	 double kBvalue, double Electronic_Temperature );

void OLD( MPI_Comm comm1, double TRAN_eV2Hartree,
	  double kBvalue, double Electronic_Temperature );

void Set_GL( double x1, double x2,  double *x,  double *w, int n );


void TRAN_Set_IntegPath( MPI_Comm comm1,
			 double TRAN_eV2Hartree,
			 double kBvalue, double Electronic_Temperature )
{

  if (TRAN_integration==0){
    CF(comm1, TRAN_eV2Hartree, kBvalue, Electronic_Temperature); 
  }
  else if (TRAN_integration==1){
    OLD(comm1, TRAN_eV2Hartree, kBvalue, Electronic_Temperature); 
  }

}





void CF( MPI_Comm comm1, double TRAN_eV2Hartree,
         double kBvalue, double Electronic_Temperature )
{
  int i,p,po,k;
  int side,side0,side1;
  double beta,R;
  double Dx,Sx,xs,xe;
  double Av_ChemP;
  double x0,x1,f0,f1;
  double tmpx0,tmpx1,x;
  char *s_vec[20];
  FILE *fp;
  char file[100];
  int myid;

  MPI_Comm_rank(comm1,&myid);

  /* set beta */
  beta = 1.0/kBvalue/Electronic_Temperature;

  /* set R for the half circle contour integral */
  R = 1.0e+12;

  /* set Order_Lead_Side */

  if (ChemP_e[0]<ChemP_e[1]){
    Order_Lead_Side[0] = 0;
    Order_Lead_Side[1] = 1;
  }
  else {
    Order_Lead_Side[0] = 1;
    Order_Lead_Side[1] = 0;
  }

  /************************************
           finite-bias voltage 
  ************************************/

  if ( tran_bias_apply ){

    /* integration for "non-equilibrium" part */
      
    side0 = Order_Lead_Side[0];
    side1 = Order_Lead_Side[1];

    Av_ChemP = 0.5*(ChemP_e[side0] + ChemP_e[side1]);
    x = Av_ChemP;   

    po = 0;
    do {

      f0 = 1.0/(1.0+exp((x-ChemP_e[side0])*beta));
      f1 = 1.0/(1.0+exp((x-ChemP_e[side1])*beta));

      if ( fabs(f1-f0)<Tran_bias_neq_cutoff ){

        po = 1;
        x1 = x; 
      };

      x += Tran_bias_neq_energy_step*0.1;

    } while (po==0);

    x0 = Av_ChemP - (x1 - Av_ChemP);
    Tran_bias_neq_lower_bound = x0;
    Tran_bias_neq_upper_bound = x1;
    Tran_bias_neq_num_energy_step = (x1-x0)/(double)Tran_bias_neq_energy_step;

    if (myid==Host_ID){
      printf("\n");
      printf("Parameters for the integration of the non-equilibrium part\n");
      printf("  lower bound:           %15.12f (eV)\n",x0*TRAN_eV2Hartree);
      printf("  upper bound:           %15.12f (eV)\n",x1*TRAN_eV2Hartree);
      printf("  energy step:           %15.12f (eV)\n",Tran_bias_neq_energy_step*TRAN_eV2Hartree);
      printf("  number of steps:           %3d     \n",Tran_bias_neq_num_energy_step);

      if (fabs(x1-x0)<Tran_bias_neq_energy_step){
        printf("  Warning: the energy step you specified is larger than the integration range.\n");
      }

      /* output the data to a file which will be merged to *.out. */

      sprintf(file,"%s%s.paranegf",filepath,filename);

      if ((fp = fopen(file,"w")) != NULL){
      
	fprintf(fp,"\n");
	fprintf(fp,"***********************************************************\n");
	fprintf(fp,"***********************************************************\n");
	fprintf(fp,"            Informaions for NEGF calculation               \n");
	fprintf(fp,"***********************************************************\n");
	fprintf(fp,"***********************************************************\n\n");

	fprintf(fp,"\n");
	fprintf(fp,"Intrinsic chemical potential (eV) of the leads\n");
	fprintf(fp,"  Left lead:  %15.12f\n",ChemP_e[0]*TRAN_eV2Hartree);
	fprintf(fp,"  Right lead: %15.12f\n",(ChemP_e[1]-tran_biasvoltage_e[1])*TRAN_eV2Hartree);

        s_vec[0] = "left";
        s_vec[1] = "right";

        side = 0;
        fprintf(fp,"  add voltage =%8.4f (eV) to the %5s lead: new ChemP (eV): %8.4f\n",
                  tran_biasvoltage_e[side]*TRAN_eV2Hartree,s_vec[side],ChemP_e[side]*TRAN_eV2Hartree);
        side = 1;
        fprintf(fp,"  add voltage =%8.4f (eV) to the %5s lead: new ChemP (eV): %8.4f\n\n",
                  tran_biasvoltage_e[side]*TRAN_eV2Hartree,s_vec[side],ChemP_e[side]*TRAN_eV2Hartree);

	fprintf(fp,"  Intrinsic bias volatage:  %15.12f (eV)\n",(ChemP_e[1]-tran_biasvoltage_e[1]-ChemP_e[0])*TRAN_eV2Hartree);
	fprintf(fp,"  Net bias volatage:        %15.12f (eV)\n",(ChemP_e[1]-ChemP_e[0])*TRAN_eV2Hartree);

	fprintf(fp,"\n");

	fprintf(fp,"Parameters for the integration of the non-equilibrium part\n");
	fprintf(fp,"  lower bound:           %15.12f (eV)\n",x0*TRAN_eV2Hartree);
	fprintf(fp,"  upper bound:           %15.12f (eV)\n",x1*TRAN_eV2Hartree);
	fprintf(fp,"  energy step:           %15.12f (eV)\n",Tran_bias_neq_energy_step*TRAN_eV2Hartree);
	fprintf(fp,"  number of steps:           %3d     \n",Tran_bias_neq_num_energy_step);

	fclose(fp);
      }
      else{
	printf("Failure of saving the NEGF data file.\n");
      }

    }            

    /* set tran_omega_n_scf */
    tran_omega_n_scf = tran_num_poles + 1 + Tran_bias_neq_num_energy_step;
  }

  /************************************
            zero-bias voltage
  ************************************/

  else{ 

    /* set tran_omega_n_scf */
    tran_omega_n_scf = tran_num_poles + 1;
  }

  /* allocation */
  tran_omega_scf = (dcomplex*)malloc(sizeof(dcomplex)*tran_omega_n_scf);
  tran_omega_weight_scf = (dcomplex*)malloc(sizeof(dcomplex)*tran_omega_n_scf);
  tran_integ_method_scf = (int*)malloc(sizeof(int)*tran_omega_n_scf);

  zero_fermi( tran_num_poles, tran_omega_scf, tran_omega_weight_scf ); 

  /*  
  for (p=0; p<tran_num_poles; p++){
    printf("p=%3d zp.r=%15.12f zp.i=%15.12f Rp.r=%15.12f Rp.i=%15.12f\n",
            p, tran_omega_scf[p].r, tran_omega_scf[p].i, tran_omega_weight_scf[p].r, tran_omega_weight_scf[p].i ); 
  }
  */

  /* finite bias case */

  if ( tran_bias_apply ){

    /*************************************
     (1) finite bias

           p=0, tran_num_poles-1  

              poles for ChemP_e[side0]

           p=tran_num_poles  

              iR for ChemP_e[side0]

           p=tran_num_poles+1,tran_num_poles+Tran_bias_neq_num_energy_step

           tran_omega_scf[tran_num_poles+1+i] = 
             Tran_bias_neq_lower_bound + (double)i*Tran_bias_neq_num_energy_step;

           tran_omega_weight_scf[tran_num_poles+1+i] = (f1 - f0)*Tran_bias_neq_energy_step;
    
    *************************************/

    /* contribution of poles for the "equilibrium" region */

    side = Order_Lead_Side[0];

    for (p=0; p<tran_num_poles; p++){

      tran_omega_scf[p].r = ChemP_e[side];
      tran_omega_scf[p].i = tran_omega_scf[p].i/beta;
      tran_omega_weight_scf[p].r = -tran_omega_weight_scf[p].r/beta;
      tran_omega_weight_scf[p].i = 0.0;
      tran_integ_method_scf[p] = 0;  
    }

    /* contribution of the half circle contour integral for the "equilibrium" region */
  
    tran_omega_scf[tran_num_poles].r = 0.0;
    tran_omega_scf[tran_num_poles].i = R;
    tran_omega_weight_scf[tran_num_poles].r = 0.0;
    tran_omega_weight_scf[tran_num_poles].i = 0.5*R;
    tran_integ_method_scf[tran_num_poles] = 1;

    /* "non-equilibrium" part */

    side0 = Order_Lead_Side[0];
    side1 = Order_Lead_Side[1];

    xs = Tran_bias_neq_lower_bound;
    xe = Tran_bias_neq_upper_bound;

    Sx = xe + xs;
    Dx = xe - xs;

    for (i=0; i<Tran_bias_neq_num_energy_step; i++){

      x = Tran_bias_neq_lower_bound + (double)i*Tran_bias_neq_energy_step;

      tran_omega_scf[tran_num_poles+1+i].r = x;
      tran_omega_scf[tran_num_poles+1+i].i = Tran_bias_neq_im_energy;

      f0 = 1.0/(1.0+exp((x-ChemP_e[side0])*beta));
      f1 = 1.0/(1.0+exp((x-ChemP_e[side1])*beta));

      tran_omega_weight_scf[tran_num_poles+1+i].r = (f1 - f0)*Tran_bias_neq_energy_step;
      tran_omega_weight_scf[tran_num_poles+1+i].i = 0.0;
      tran_integ_method_scf[tran_num_poles+1+i] = 2;
    }

  }

  /* zero-bias case */

  else {

    /*************************************
     (3) zero-bias

           p=0, tran_num_poles-1  

              poles for ChemP_e[side0]

           p=tran_num_poles  

              iR for ChemP_e[side0]
    *************************************/

    /* contribution of poles */

    for (p=0; p<tran_num_poles; p++){
      tran_omega_scf[p].r = ChemP_e[0];
      tran_omega_scf[p].i = tran_omega_scf[p].i/beta;
      tran_omega_weight_scf[p].r = -2.0*tran_omega_weight_scf[p].r/beta;
      tran_omega_weight_scf[p].i =  0.0;
      tran_integ_method_scf[p] = 0; 
    }

    /* contribution of the half circle contour integral */

    tran_omega_scf[tran_num_poles].r = 0.0;
    tran_omega_scf[tran_num_poles].i = R;

    tran_omega_weight_scf[tran_num_poles].r = 0.0; 
    tran_omega_weight_scf[tran_num_poles].i = 0.5*R;
    tran_integ_method_scf[tran_num_poles] = 1; 
  }  

}






void OLD( MPI_Comm comm1, double TRAN_eV2Hartree,
          double kBvalue, double Electronic_Temperature )
{
  int i,p,po,k,num;
  int num_path1,num_path2,num_path3;
  int side,side0,side1;
  double beta,R,fr,fi,A,B;
  double dxp,dxm,dx,xi,xr;
  double theta,theta1,theta2;
  double Av_ChemP,dtheta;
  double Delta,EB,a,zr,zi;
  double x0,x1,f0,f1;
  double St,Dt,Sx,Dx,xs,xe;
  double tmpx0,tmpx1,x;
  int myid;

  MPI_Comm_rank(comm1,&myid);

  /* set beta */
  beta = 1.0/kBvalue/Electronic_Temperature;

  if ( 1.0e-8<fabs(tran_biasvoltage_e[1]) ){
    printf("\nThe integration scheme does not support the finite bias case.\n\n");
    MPI_Finalize();
    exit(0);
  }

  /****************************************
    set the number of points for each path 
  ****************************************/

  Delta = 1.2/27.2113845;
  x0 = 0.5*(Delta*beta/PI-1.0);    
  num_path1 = (int)x0 + 1;

  num_path2 = 400;
  num_path3 = 40;  

  TRAN_GL_Abs = (double*)malloc(sizeof(double)*(num_path3+2));
  TRAN_GL_Wgt = (double*)malloc(sizeof(double)*(num_path3+2));

  printf("num %2d %2d %2d\n",num_path1,num_path2,num_path3);

  tran_omega_n_scf = num_path1 + num_path2 + num_path3;

  /* allocation */
  tran_omega_scf = (dcomplex*)malloc(sizeof(dcomplex)*tran_omega_n_scf);
  tran_omega_weight_scf = (dcomplex*)malloc(sizeof(dcomplex)*tran_omega_n_scf);
  tran_integ_method_scf = (int*)malloc(sizeof(int)*tran_omega_n_scf);

  /****************************************
       path1 = Matsubara poles
  ****************************************/

  for (i=0; i<num_path1; i++){
    tran_omega_scf[i].r = ChemP_e[0];
    tran_omega_scf[i].i = (2.0*(double)i + 1.0)*PI/beta;

    /*
    tran_omega_weight_scf[i].r = 0.0;
    tran_omega_weight_scf[i].i = -2.0*PI/beta;
    */

    tran_omega_weight_scf[i].r = 2.0/beta;
    tran_omega_weight_scf[i].i = 0.0;

    tran_integ_method_scf[i] = 1;
  }
  
  /****************************************
      path2 = L
  ****************************************/

  dxp = 40.0/beta;
  dxm =-40.0/beta;

  xs = ChemP_e[0] + dxm;
  xe = ChemP_e[0] + dxp;

  TRAN_GL_Abs = (double*)malloc(sizeof(double)*(num_path2+2));
  TRAN_GL_Wgt = (double*)malloc(sizeof(double)*(num_path2+2));

  Set_GL(-1.0, 1.0, TRAN_GL_Abs, TRAN_GL_Wgt, num_path2);

  Sx = xe + xs;
  Dx = xe - xs;

  for (i=0; i<num_path2; i++){

    x = 0.50*(Dx*TRAN_GL_Abs[i] + Sx);

    tran_omega_scf[i+num_path1].r = x;
    tran_omega_scf[i+num_path1].i = Delta;

    xr = beta*(tran_omega_scf[i+num_path1].r - ChemP_e[0]);
    xi = beta*tran_omega_scf[i+num_path1].i;

    A = 1.0 + exp(xr)*cos(xi);
    B = exp(xr)*sin(xi);
  
    fr = A/(A*A+B*B);
    fi =-B/(A*A+B*B);

    tran_omega_weight_scf[i+num_path1].r =-0.5*Dx*TRAN_GL_Wgt[i]*fi/PI;    
    tran_omega_weight_scf[i+num_path1].i = 0.5*Dx*TRAN_GL_Wgt[i]*fr/PI;

    tran_integ_method_scf[i+num_path1] = 1;
  }

  /*
  dxp = 40.0/beta;
  dxm =-40.0/beta;
  dx = (dxp - dxm)/(double)num_path2;

  for (i=0; i<num_path2; i++){

    tran_omega_scf[i+num_path1].r = ChemP_e[0] + dxp - (double)i*dx;
    tran_omega_scf[i+num_path1].i = Delta;

    xr = beta*(tran_omega_scf[i+num_path1].r - ChemP_e[0]);
    xi = beta*tran_omega_scf[i+num_path1].i;

    A = 1.0 + exp(xr)*cos(xi);
    B = exp(xr)*sin(xi);
  
    fr = A/(A*A+B*B);
    fi =-B/(A*A+B*B);

    tran_omega_weight_scf[i+num_path1].r =-dx*fi/PI;    
    tran_omega_weight_scf[i+num_path1].i = dx*fr/PI;

    tran_integ_method_scf[i+num_path1] = 1;
  }
  */

  /****************************************
      path3 = L
  ****************************************/

  EB = -1.0;
  zr = ChemP_e[0] + dxm;
  zi = Delta;
  a = 0.5*(EB*EB - zr*zr - zi*zi)/(EB - zr);
  R = a - EB;

  theta1 = acos((zr-a)/R); 
  theta2 = PI;

  printf("EB=%15.12f a=%15.12f ChemP=%15.12f theta1=%15.12f theta2=%15.12f\n",EB,a,ChemP_e[0],theta1,theta2); 

  TRAN_GL_Abs = (double*)malloc(sizeof(double)*(num_path3+2));
  TRAN_GL_Wgt = (double*)malloc(sizeof(double)*(num_path3+2));

  Set_GL(-1.0, 1.0, TRAN_GL_Abs, TRAN_GL_Wgt, num_path3);

  St = theta2 + theta1;
  Dt = theta2 - theta1;  

  for (i=0; i<num_path3; i++){

    theta = 0.50*(Dt*TRAN_GL_Abs[i] + St);

    tran_omega_scf[i+num_path1+num_path2].r = a + R*cos(theta);
    tran_omega_scf[i+num_path1+num_path2].i = R*sin(theta);

    xr = beta*(tran_omega_scf[i+num_path1+num_path2].r - ChemP_e[0]);
    xi = beta*tran_omega_scf[i+num_path1+num_path2].i;

    A = 1.0 + exp(xr)*cos(xi);
    B = exp(xr)*sin(xi);

    fr = A/(A*A+B*B);
    fi =-B/(A*A+B*B);

    tran_omega_weight_scf[i+num_path1+num_path2].r = 0.5*Dt*TRAN_GL_Wgt[i]*(-fi*R*sin(theta) + fr*R*cos(theta))/PI;
    tran_omega_weight_scf[i+num_path1+num_path2].i =-0.5*Dt*TRAN_GL_Wgt[i]*(-fi*R*cos(theta) - fr*R*sin(theta))/PI; 

    tran_integ_method_scf[i+num_path1+num_path2] = 1;
  }


  free(TRAN_GL_Abs);
  free(TRAN_GL_Wgt);


  /*
  for (i=0; i<num_path3; i++){

    dtheta =(theta2-theta1)/(double)num_path3; 
    theta = theta1 + dtheta*(double)i;

    tran_omega_scf[i+num_path1+num_path2].r = a + R*cos(theta);
    tran_omega_scf[i+num_path1+num_path2].i = R*sin(theta);

    xr = beta*(tran_omega_scf[i+num_path1+num_path2].r - ChemP_e[0]);
    xi = beta*tran_omega_scf[i+num_path1+num_path2].i;

    A = 1.0 + exp(xr)*cos(xi);
    B = exp(xr)*sin(xi);

    fr = A/(A*A+B*B);
    fi =-B/(A*A+B*B);

    tran_omega_weight_scf[i+num_path1+num_path2].r = dtheta*(-fi*R*sin(theta) + fr*R*cos(theta))/PI;
    tran_omega_weight_scf[i+num_path1+num_path2].i =-dtheta*(-fi*R*cos(theta) - fr*R*sin(theta))/PI; 

    tran_integ_method_scf[i+num_path1+num_path2] = 1;
  }
  */



  /*

  if (1){

  if (myid==0){
   
    dcomplex dsum,w,g,z;
    double e;

    e = -1.0; 
    dsum.r = 0.0;
    dsum.i = 0.0;

    for (i=0; i<tran_omega_n_scf; i++){

      z.r = tran_omega_scf[i].r;
      z.i = tran_omega_scf[i].i;

      w.r = tran_omega_weight_scf[i].r;
      w.i = tran_omega_weight_scf[i].i;

      g.r = (z.r-e)/((z.r-e)*(z.r-e)+z.i*z.i); 
      g.i =-z.i/((z.r-e)*(z.r-e)+z.i*z.i); 

      dsum.r += g.r*w.r - g.i*w.i;
      dsum.i += g.r*w.i + g.i*w.r;

    }

    printf("A %4d %18.15f %18.15f\n",tran_omega_n_scf,dsum.r,dsum.i);


    double es,ee,de;
    int num;

    e = -1.0; 

    num = 100000000;
    es = EB;
    ee = ChemP_e[0] + dxp;
    de = (ee-es)/(double)(num-1);
    dsum.r = 0.0;
    dsum.i = 0.0;

    for (i=0; i<num; i++){

      z.r = es + de*(double)i;
      z.i = 0.0001;

      fr = 1.0/(1.0+exp(beta*(z.r-ChemP_e[0])));

      w.r = 0.0*fr;
      w.i = de/PI*fr;

      g.r = (z.r-e)/((z.r-e)*(z.r-e)+z.i*z.i); 
      g.i =-z.i/((z.r-e)*(z.r-e)+z.i*z.i); 

      dsum.r += g.r*w.r - g.i*w.i;
      dsum.i += g.r*w.i + g.i*w.r;

    }

    printf("B %4d %18.15f %18.15f\n",num,dsum.r,dsum.i);
  }


  MPI_Finalize();
  exit(0);
  }

  */


}



void Set_GL( double x1, double x2,  double *x,  double *w, int n )
{

  int m,j,i;
  double z1,z,xm,xl,pp,p3,p2,p1; 

  m=(n+1)/2;
  xm=0.50*(x2+x1);
  xl=0.50*(x2-x1);

  for (i=1;i<=m;i++) {

    z=cos(PI*(i-0.250)/(n+0.50));

    do {

      p1=1.0;
      p2=0.0;

      for (j=1;j<=n;j++) {
	p3=p2;
	p2=p1;
	p1=((2.0*(double)j-1.0)*z*p2-((double)j-1.0)*p3)/(double)j;
      }

      pp=(double)n*(z*p1-p2)/(z*z-1.0);
      z1=z;
      z=z1-p1/pp;

    } while (fabs(z-z1) > EPS);

    x[i]=xm-xl*z;     
    x[n+1-i]=xm+xl*z; 
    w[i]=2.0*xl/((1.0-z*z)*pp*pp);
    w[n+1-i]=w[i];    
  }

  for (i=1; i<=n; i++) {
    x[i-1] = x[i];    
    w[i-1] = w[i];    
  }  

}





  
void zero_fermi( int N, dcomplex *zp0, dcomplex *Rp0 ) 
{
  int i,j,M;
  double **A,**B;
  double *zp,*Rp;

  /* check input parameters */

  if (N<=0){
    printf("\ncould not find the number of zeros\n\n");
    MPI_Finalize();
    exit(0);
  }

  /* find the number of zeros */

  M = 2*N;

  /* allocation of arrays */

  A = (double**)malloc(sizeof(double*)*(M+2));
  for (i=0; i<(M+2); i++){
    A[i] = (double*)malloc(sizeof(double)*(M+2));
  }

  B = (double**)malloc(sizeof(double*)*(M+2));
  for (i=0; i<(M+2); i++){
    B[i] = (double*)malloc(sizeof(double)*(M+2));
  }

  zp = (double*)malloc(sizeof(double)*(M+2));
  Rp = (double*)malloc(sizeof(double)*(M+2));

  /* initialize arrays */

  for (i=0; i<(M+2); i++){
    for (j=0; j<(M+2); j++){
      A[i][j] = 0.0;
      B[i][j] = 0.0;
    }

    zp[i] = 0.0;
    Rp[i] = 0.0;
  }

  /* set matrix elements */

  for (i=1; i<=M; i++){
    B[i][i] = (2.0*(double)i - 1.0);
  }

  for (i=1; i<=(M-1); i++){
    A[i][i+1] = -0.5;
    A[i+1][i] = -0.5;
  }

  /* diagonalization */

  {
  int i,j;
  char jobz = 'V';
  char uplo ='U';
  INTEGER itype;
  INTEGER n,lda,ldb,lwork,info;
  double *a,*b;
  double *work;

  itype = 1;
  n = M;
  lda = M;
  ldb = M;
  lwork = 3*M;

  a = (double*)malloc(sizeof(double)*n*n);
  b = (double*)malloc(sizeof(double)*n*n);
  work = (double*)malloc(sizeof(double)*3*n);

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      a[j*n+i] = A[i+1][j+1];
      b[j*n+i] = B[i+1][j+1];
    }
  }

  F77_NAME(dsygv,DSYGV)( &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, zp, work, &lwork, &info);

  /*
  printf("info=%2d\n",info);
  */

  /* store residue */

  for (i=0; i<N; i++){
    zp0[i].r = 0.0;
    zp0[i].i =-1.0/zp[i];
  }

  for (i=0; i<N; i++) {
    Rp0[i].r = -a[i*n]*a[i*n]*zp0[i].i*zp0[i].i*0.250;
    Rp0[i].i = 0.0;
  }

  free(a);
  free(b);
  free(work);
  }

  /* free of arrays */

  free(Rp);
  free(zp);

  for (i=0; i<(M+2); i++){
    free(B[i]);
  }
  free(B);

  for (i=0; i<(M+2); i++){
    free(A[i]);
  }
  free(A);
}



















