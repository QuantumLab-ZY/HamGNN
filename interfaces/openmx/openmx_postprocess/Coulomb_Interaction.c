/**********************************************************************
  Coulomb_Interaction.c:

     Coulomb_Interaction.c is a subroutine to calculate
     Coulomb interaction tensor for general LDA+U scheme.

  by Siheon Ryee.
***********************************************************************/

#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"

/* Coulomb interaction tensor */
static double FAC(int num); 
static int MIN(int a, int b);
static int MAX(int a, int b);
static double Wigner3j(int l1, int l2, int l3, int m1, int m2, int m3);
static double Gaunt_SR(int l, int k, int m1, int m2);
static dcomplex StoR(int l, int m1, int m2);
static double Coulomb_Matrix(int num, double F0, double JH, int l, int a, int b, int c, int d);
/*****************************/

/* Yukawa potential */
static double Bessel_j(int l2max, int k_in, double x);
static double Bessel_h(int k_in, double x);
static double Integrate_Bessel(int mode, int N, double start, double end, int isp, int l, int mul, int k, double Ndi, double lambda);
static double Integrate_Bessel0(int mode, int N, double start, double end, int isp, int l, int mul, int k, double Ndi);
/********************/


void Coulomb_Interaction(){

  int Npoints,Nmesh,N_iter,i,j,l,mul,mm; 
  double Ndi,Nend,lambda_lower,lambda_upper,lambda_test; 
  int dum_Nmul,dum_l,dum_mul;
  double Uval,Jval,delta_U,J_temp;
  
  int NZUJ;
 
  int a,b,c,d;

  int numprocs,myid;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* Calculate Yukawa potential */
  if(Yukawa_on==1){
    if (myid==Host_ID && 0<level_stdout){
      printf("\n");
      printf("*******************************************************\n");   fflush(stdout);
      printf("      Calculating Thomas-Fermi screening length        \n" );  fflush(stdout);
      printf("*******************************************************\n\n"); fflush(stdout); 
    }


    /* Allocating arrays */
    Npoints = 1600; /* Arbitrarily large value */

    Bessel_j_Array0 = (double**)malloc(sizeof(double*)*(Nmul+1));
    for(i=0; i<=Nmul; i++){
      Bessel_j_Array0[i] = (double*)malloc(sizeof(double)*Npoints);
      for(mm=0; mm<Npoints; mm++){
        Bessel_j_Array0[i][mm] = 0.0;
      }
    }

    Bessel_h_Array0 = (double**)malloc(sizeof(double*)*(Nmul+1));
    for(i=0; i<=Nmul; i++){
      Bessel_h_Array0[i] = (double*)malloc(sizeof(double)*Npoints);
      for(mm=0; mm<Npoints; mm++){
        Bessel_h_Array0[i][mm] = 0.0;
      }
    }
   
    dum_Nmul=1;
    for(i=0; i<SpeciesNum; i++){
      for(dum_l=0; dum_l<=Spe_MaxL_Basis[i]; dum_l++){
        for(dum_mul=0; dum_mul<Spe_Num_Basis[i][dum_l]; dum_mul++){
          Uval = Hub_U_Basis[i][dum_l][dum_mul];
          Jval = Hund_J_Basis[i][dum_l][dum_mul];
          if (Uval > 1.0e-5 || Jval > 1.0e-5){
            U[dum_Nmul] = Uval;
            B_spe[dum_Nmul] = i;
            B_l[dum_Nmul] = dum_l;
            B_mul[dum_Nmul] = dum_mul;
            B_cut[dum_Nmul] = Spe_Atom_Cut1[i];
            dum_Nmul++;
          }
        }
      }
    }
    /* End allocating arrays */
    
    
    for(i=1; i<=Nmul; i++){

      /* parameters for calculating Slater integrals */
      /* unit of lambda: 1/a.u. */
      /* All quantities are in atomic unit */
      lambda_lower = 0.1;
      if (U[i] >= 1.5) lambda_upper = 10.0;
      else if (U[i] >= 0.5) lambda_upper = 20.0;
      else lambda_upper = 40.0;

      delta_U = 0.0005/eV2Hartree;  /* criterion for convergence */
      
      Ndi=0.01;
      Nend=B_cut[i]+1.0;
      Nmesh=(int)(Nend/Ndi);

      N_iter=200; /* maximum # of iteration */
      /***********************************************/

      /* Calculating unscreened Slater integrals */
/*      for(mm=0; mm<Nmesh; mm++){
        Bessel_j_Array0[i][mm] = Integrate_Bessel0(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],0,Ndi);
        Bessel_h_Array0[i][mm] = Integrate_Bessel0(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],0,Ndi);
      }
      Slater_F0[i] = Integrate_Bessel0(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],0,Ndi);

      for(mm=0; mm<Nmesh; mm++){
        Bessel_j_Array0[i][mm] = Integrate_Bessel0(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],2,Ndi);
        Bessel_h_Array0[i][mm] = Integrate_Bessel0(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],2,Ndi);
      }
      Slater_F2[i] = Integrate_Bessel0(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],2,Ndi);

      for(mm=0; mm<Nmesh; mm++){
        Bessel_j_Array0[i][mm] = Integrate_Bessel0(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],4,Ndi);
        Bessel_h_Array0[i][mm] = Integrate_Bessel0(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],4,Ndi);
      }
      Slater_F4[i] = Integrate_Bessel0(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],4,Ndi);

      if(B_l[i]==3){
        for(mm=0; mm<Nmesh; mm++){
          Bessel_j_Array0[i][mm] = Integrate_Bessel0(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],6,Ndi);
          Bessel_h_Array0[i][mm] = Integrate_Bessel0(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],6,Ndi);
        }
        Slater_F6[i] = Integrate_Bessel0(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],6,Ndi);
      }
      printf("F0= %f, F2= %f, F4= %f\n", Slater_F0[i], Slater_F2[i], Slater_F4[i]); */


      /* Finding Thomas-Fermi screening length of Hubbard U(=F0) */
      for(j=1; j<=N_iter; j++){
        lambda_test = 0.5*(lambda_lower+lambda_upper);
      
        for(mm=0; mm<Nmesh; mm++){
          Bessel_j_Array0[i][mm] = Integrate_Bessel(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],0,Ndi,lambda_test);
          Bessel_h_Array0[i][mm] = Integrate_Bessel(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],0,Ndi,lambda_test);
        }
        Slater_F0[i] = Integrate_Bessel(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],0,Ndi,lambda_test);

        if(Slater_F0[i] <= U[i]+delta_U && Slater_F0[i] >= U[i]-delta_U){
          lambda[i]=lambda_test;
          break;
        }
  
        if((Slater_F0[i]-U[i])>0.0) lambda_lower = lambda_test;
        else lambda_upper = lambda_test;

        if(j==N_iter){
          if(myid==Host_ID && 0<level_stdout){
            printf("!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!! \n");  fflush(stdout);
            printf("    Slater integrals are not correctly calculated.\n");  fflush(stdout);
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");  fflush(stdout);
            printf("\n");  fflush(stdout);
          }
        }
           
      } /* N_iter */
      
      /* Calculating F2, F4, and F6 by using Yukawa potential */
      for(mm=0; mm<Nmesh; mm++){
        Bessel_j_Array0[i][mm] = Integrate_Bessel(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],2,Ndi,lambda[i]);
        Bessel_h_Array0[i][mm] = Integrate_Bessel(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],2,Ndi,lambda[i]);
      }
      Slater_F2[i] = Integrate_Bessel(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],2,Ndi,lambda[i]);

      for(mm=0; mm<Nmesh; mm++){
        Bessel_j_Array0[i][mm] = Integrate_Bessel(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],4,Ndi,lambda[i]);
        Bessel_h_Array0[i][mm] = Integrate_Bessel(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],4,Ndi,lambda[i]);
      }
      Slater_F4[i] = Integrate_Bessel(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],4,Ndi,lambda[i]);

      if(B_l[i]==3){
        for(mm=0; mm<Nmesh; mm++){
          Bessel_j_Array0[i][mm] = Integrate_Bessel(0,i,0.0,(double)(mm)*Ndi,B_spe[i],B_l[i],B_mul[i],6,Ndi,lambda[i]);
          Bessel_h_Array0[i][mm] = Integrate_Bessel(1,i,(double)(mm)*Ndi,Nend,B_spe[i],B_l[i],B_mul[i],6,Ndi,lambda[i]);
        }
        Slater_F6[i] = Integrate_Bessel(2,i,0.0,(Nend-1.0),B_spe[i],B_l[i],B_mul[i],6,Ndi,lambda[i]);
      }

      if(B_l[i]==1){
        J_temp = (Slater_F2[i])/5.0;
      }
      if(B_l[i]==2){
        J_temp = (Slater_F2[i] + Slater_F4[i])/14.0;
      }
      if(B_l[i]==3){
        J_temp = (286.0*Slater_F2[i] + 195.0*Slater_F4[i] + 250.0*Slater_F6[i])/6435.0;
      }

      J[i] = J_temp;

      if (myid==Host_ID && level_stdout>0){
        printf("<species: %s, angular momentum= %d, multiplicity number= %d>\n", SpeName[B_spe[i]], B_l[i], B_mul[i]);
        printf(" TF-screening-length lambda= %f 1/au\n", lambda[i]);
        printf(" Hubbard U= %f eV\n", U[i]*eV2Hartree);
        printf(" Hund J= %f eV\n", J[i]*eV2Hartree);
        printf(" Slater F0= %f eV\n", Slater_F0[i]*eV2Hartree); 
        printf(" Slater F2= %f eV\n", Slater_F2[i]*eV2Hartree); 
        printf(" Slater F4= %f eV\n", Slater_F4[i]*eV2Hartree); 
        if(B_l[i]==3){
          printf(" Slater F6= %f eV\n", Slater_F6[i]*eV2Hartree); 
        }
        if(B_l[i]==2){
          printf(" F4/F2= %f\n", Slater_F4[i]/Slater_F2[i]); 
        }
        if(B_l[i]==3){
          printf(" F4/F2= %f\n", Slater_F4[i]/Slater_F2[i]); 
          printf(" F6/F4= %f\n", Slater_F6[i]/Slater_F4[i]); 
        }
        printf("\n");
      }

    } /* Nmul */


    /* Freeing arrays */
    for(i=0; i<=Nmul; i++){
      free(Bessel_j_Array0[i]);
      free(Bessel_h_Array0[i]);
    }
  } /* Yukawa_on==1 */



  /**********************************************
    Storing values to Coulomb interaction array
      (only when Hub_Type=2 is used)
  **********************************************/
  if(Yukawa_on != 1){  /* U and J from input */

    /* Storing U,J values to the array */
    dum_Nmul=1;
    for(i=0; i<SpeciesNum; i++){
      for(dum_l=0; dum_l<=Spe_MaxL_Basis[i]; dum_l++){
        for(dum_mul=0; dum_mul<Spe_Num_Basis[i][dum_l]; dum_mul++){
          Uval = Hub_U_Basis[i][dum_l][dum_mul];
          Jval = Hund_J_Basis[i][dum_l][dum_mul];
          NZUJ = Nonzero_UJ[i][dum_l][dum_mul];
          if((Uval > 1.0e-5 || Jval > 1.0e-5) && (NZUJ>0)){
            for(a=0; a<(2*dum_l+1); a++){
              for(b=0; b<(2*dum_l+1); b++){
                for(c=0; c<(2*dum_l+1); c++){
                  for(d=0; d<(2*dum_l+1); d++){
                    Coulomb_Array[NZUJ][a][b][c][d]
                    = Coulomb_Matrix(dum_Nmul,Uval,Jval,dum_l,a,b,c,d);
                  }
                }
              }
            }
	    dum_Nmul++;
          } 
        } 
      } 
    } 
  
  /*  for(a=0; a<(2*2+1); a++){
      for(b=0; b<(2*2+1); b++){
        printf("a=%d b=%d, U=%f\n", a,b,Coulomb_Array[0][a][b][a][b]*eV2Hartree); 
      }
    }*/
  }


  /*************************************************************
    Allocating Hubbard U & Hund J values from Slater integrals
  *************************************************************/
  if(Yukawa_on==1){  /* U from input and J from Yukawa */

    /* initialize J values */ 
    for (i=0; i<SpeciesNum; i++){
      for (l=0; l<=Spe_MaxL_Basis[i]; l++){
        for (mul=0; mul<Spe_Num_Basis[i][l]; mul++){
          Hund_J_Basis[i][l][mul]=0.0;
	}
      }
    }

    /* Storing U,J values to the array */
    dum_Nmul=1;
    for(i=0; i<SpeciesNum; i++){
      for(dum_l=0; dum_l<=Spe_MaxL_Basis[i]; dum_l++){
        for(dum_mul=0; dum_mul<Spe_Num_Basis[i][dum_l]; dum_mul++){
          Uval = Hub_U_Basis[i][dum_l][dum_mul];
          NZUJ = Nonzero_UJ[i][dum_l][dum_mul];
          if ((Uval > 1.0e-5) && (NZUJ>0)){
            Hund_J_Basis[i][dum_l][dum_mul] = J[dum_Nmul];
            Jval = J[dum_Nmul]; 
            for(a=0; a<(2*dum_l+1); a++){
              for(b=0; b<(2*dum_l+1); b++){
                for(c=0; c<(2*dum_l+1); c++){
                  for(d=0; d<(2*dum_l+1); d++){
                    Coulomb_Array[NZUJ][a][b][c][d]
                    = Coulomb_Matrix(dum_Nmul,Uval,Jval,dum_l,a,b,c,d);
                  }
                }
              }
            }
	    dum_Nmul++;
          } 
        } 
      } 
    } 


  }
  


  /****************************************************
                  Generating AMF array
  (only when around-mean-field double-counting is used)
  ****************************************************/
  
  if(Hub_U_switch==1 && Hub_Type==2 && (dc_Type==2 || dc_Type==4)){

    AMF_Array = (double*****)malloc(sizeof(double****)*(Nmul+1));
    for(i=0; i<=Nmul; i++){
      AMF_Array[i] = (double****)malloc(sizeof(double***)*4);
      for(a=0; a<4; a++){
        AMF_Array[i][a] = (double***)malloc(sizeof(double**)*2);
        for(b=0; b<2; b++){
          AMF_Array[i][a][b] = (double**)malloc(sizeof(double*)*7);
          for(c=0; c<7; c++){
            AMF_Array[i][a][b][c] = (double*)malloc(sizeof(double)*7);
            for(d=0; d<7; d++){
              AMF_Array[i][a][b][c][d] = 0.0;
            }
          }
        }
      }
    }
  }
 



} /* Yukawa_potential */ 






/******************************************************
   Integrate r^2*R*R*j form "start" to "end"
   
   R: Radial function of atomic orbital
      with angular moment "l" and multiplicity "mul"
   j: Spherical Bessel function
******************************************************/


/******************************************************
   Integrate r^2*R*R*h form "start" to "end"
   
   R: Radial function of atomic orbital
      with angular moment "l" and multiplicity "mul"
   h: Spherical Hankel function of the first kind
******************************************************/

/*static double Integrate_Bessel0(int mode, int N, double start, double end, int isp, int l, int mul, int k, double Ndi){

  double i,di,f_i1,f_i2,f_i3,sum_f;
  double j0,j0_di,j0_2di;
  double h0,h0_di,h0_2di;
  int step;  

  di = 0.001;
  sum_f = 0.0;


  switch (mode){
  case 0:  
    if(end>start){
      for(i=start; i<=end; i+=2.0*di){  
        j0 = pow(i,(double)k);
        j0_di = pow(i+di,(double)k);
        j0_2di = pow(i+2.0*di,(double)k);

        f_i1 = i*i*RadialF(isp, l, mul, i)*RadialF(isp, l, mul, i) * j0;
        f_i2 = (i+di)*(i+di)*RadialF(isp, l, mul, i+di)*RadialF(isp, l, mul, i+di) * j0_di;
        f_i3 = (i+2.0*di)*(i+2.0*di)*RadialF(isp, l, mul, i+2.0*di)*RadialF(isp, l, mul, i+2.0*di) * j0_2di;
        if (i > end-2.0*di) break;

        sum_f += (di/3.0)*(f_i1+4.0*f_i2+f_i3);
      }
    }  
    else{
      sum_f = 0.0;
    } 
 
    printf("Bessel = %f\n", sum_f); 
  break;

  case 1: 
    if(end>start){
      for(i=start; i<=end; i+=2.0*di){  
        h0 = 1.0/(i*pow(i,(double)k));
        h0_di = 1.0/((i+di)*pow(i+di,(double)k));
        h0_2di = 1.0/((i+2.0*di)*pow(i+2.0*di,(double)k));

        f_i1 = i*i*RadialF(isp, l, mul, i)*RadialF(isp, l, mul, i) * h0;
        f_i2 = (i+di)*(i+di)*RadialF(isp, l, mul, i+di)*RadialF(isp, l, mul, i+di) * h0_di;
        f_i3 = (i+2.0*di)*(i+2.0*di)*RadialF(isp, l, mul, i+2.0*di)*RadialF(isp, l, mul, i+2.0*di) * h0_2di;
        if (i > end-2.0*di) break;

        sum_f += (di/3.0)*(f_i1+4.0*f_i2+f_i3);
      }
    }  
    else{
      sum_f = 0.0;
    }

    printf("Hankel = %f\n", sum_f); 
  break;

  step = 0;
  case 2:
    for(i=start; i<=end; i+=2.0*Ndi){
      j0 = pow(i,(double)k);
      j0_di = pow(i+Ndi,(double)k);
      j0_2di = pow(i+2.0*Ndi,(double)k);
      h0 = 1.0/(i*pow(i,(double)k));
      h0_di = 1.0/((i+Ndi)*pow(i+Ndi,(double)k));
      h0_2di = 1.0/((i+2.0*Ndi)*pow(i+2.0*Ndi,(double)k));

      f_i1 = i*i*RadialF(isp,l,mul,i)*RadialF(isp,l,mul,i)
            *(Bessel_j_Array0[N][step]*h0
             +Bessel_h_Array0[N][step]*j0);
  
      f_i2 = (i+Ndi)*(i+Ndi)*RadialF(isp,l,mul,i+Ndi)*RadialF(isp,l,mul,i+Ndi)
            *(Bessel_j_Array0[N][step+1]*h0_di
             +Bessel_h_Array0[N][step+1]*j0_di);
  
      f_i3 = (i+2.0*Ndi)*(i+2.0*Ndi)*RadialF(isp,l,mul,i+2.0*Ndi)*RadialF(isp,l,mul,i+2.0*Ndi)
            *(Bessel_j_Array0[N][step+2]*h0_2di
             +Bessel_h_Array0[N][step+2]*j0_2di);

      sum_f += (Ndi/3.0)*(f_i1+4.0*f_i2+f_i3);

      step+=2; 
    }

    printf("Total = %f\n", sum_f); 

  break;
  }
 
  return sum_f;

}
*/




static double Integrate_Bessel(int mode, int N, double start, double end, int isp, int l, int mul, int k, double Ndi, double lambda){

  double i,di,f_i1,f_i2,f_i3,sum_f;
  int step;  

  di = 0.001;
  sum_f = 0.0;

  switch (mode){
  case 0:   /* Spherical Bessel function */
    if(end>start){
      for(i=start; i<=end; i+=2.0*di){  
        /* Integration using Simpson's rule */
        f_i1 = i*i*RadialF(isp, l, mul, i)*RadialF(isp, l, mul, i) * Bessel_j(2*l,k,lambda*i);
        f_i2 = (i+di)*(i+di)*RadialF(isp, l, mul, i+di)*RadialF(isp, l, mul, i+di) * Bessel_j(2*l,k,lambda*(i+di));
        f_i3 = (i+2.0*di)*(i+2.0*di)*RadialF(isp, l, mul, i+2.0*di)*RadialF(isp, l, mul, i+2.0*di) * Bessel_j(2*l,k,lambda*(i+2.0*di));
        if (i > end-2.0*di) break;

        sum_f += (di/3.0)*(f_i1+4.0*f_i2+f_i3);
      }
    }  
    else{
      sum_f = 0.0;
    } 
  break;

  case 1:   /* Spherical Hankel function */
    if(end>start){
      for(i=start; i<=end; i+=2.0*di){  
        /* Integration using Simpson's rule */
        f_i1 = i*i*RadialF(isp, l, mul, i)*RadialF(isp, l, mul, i) * Bessel_h(k,lambda*i);
        f_i2 = (i+di)*(i+di)*RadialF(isp, l, mul, i+di)*RadialF(isp, l, mul, i+di) * Bessel_h(k,lambda*(i+di));
        f_i3 = (i+2.0*di)*(i+2.0*di)*RadialF(isp, l, mul, i+2.0*di)*RadialF(isp, l, mul, i+2.0*di) * Bessel_h(k,lambda*(i+2.0*di));
        if (i > end-2.0*di) break;

        sum_f += (di/3.0)*(f_i1+4.0*f_i2+f_i3);
      }
    }  
    else{
      sum_f = 0.0;
    }

  break;

  step = 0;
  case 2:   /* Calculating Slater integral */
    for(i=start; i<=end; i+=2.0*Ndi){
      
      /* Integration using Simpson's rule */
      f_i1 = i*i*RadialF(isp,l,mul,i)*RadialF(isp,l,mul,i)
            *(Bessel_j_Array0[N][step]*Bessel_h(k,lambda*i)
             +Bessel_h_Array0[N][step]*Bessel_j(2*l,k,lambda*i));
  
      f_i2 = (i+Ndi)*(i+Ndi)*RadialF(isp,l,mul,i+Ndi)*RadialF(isp,l,mul,i+Ndi)
            *(Bessel_j_Array0[N][step+1]*Bessel_h(k,lambda*(i+Ndi))
             +Bessel_h_Array0[N][step+1]*Bessel_j(2*l,k,lambda*(i+Ndi)));
  
      f_i3 = (i+2.0*Ndi)*(i+2.0*Ndi)*RadialF(isp,l,mul,i+2.0*Ndi)*RadialF(isp,l,mul,i+2.0*Ndi)
            *(Bessel_j_Array0[N][step+2]*Bessel_h(k,lambda*(i+2.0*Ndi))
             +Bessel_h_Array0[N][step+2]*Bessel_j(2*l,k,lambda*(i+2.0*Ndi)));

      sum_f += (Ndi/3.0)*(f_i1+4.0*f_i2+f_i3);

      step+=2; 
    }

    sum_f = sum_f*lambda*(((double)(2*k+1))*pow(-1.0,(double)(k+1)));

  break;
  }
 
  return sum_f;

}


/****************************************************
   Bessel_h and Bessel_j are based on the
   subroutine of Elk code created by F. Bultmark,
   F. Cricchio, L. Nordstrom and J. K. Dewhurst.


   Computes variations of the spherical Bessel function
   $b_l(x)=i^lh^{(1)}_l(ix)$, for real argument $x$ and
   $l=0,1,\ldots,l_{\rm max}$. The recursion relation
   $$ j_{l+1}(x)=\frac{2l+1}{x}j_l(x)+j_{l-1}(x) $$
   is used upwards. For starting values there are
   $$ b_0(x)=-\frac{e^{-x}}{x};\qquad b_1(x)=b_0(x)\left\{1+\frac{1}{x}
   \right\}. $$
   For $x\ll 1$ the asymtotic forms
   $$ b_l(x)\approx\frac{-(2l-1)!!}{(-x)^{l+1}} $$
   are used.
****************************************************/

static double Bessel_h(int k_in, double x){

  int l;
  double xi,xmin,t1,t2;
  double b0,b1,b_dum;
  double f;
 
  if (x==0.0){
    x = 1.0e-9;
  }

 
  xmin = 1.0e-8;
  xi = 1.0/x; 


  b0 = -xi*exp(-x);
  b1 = b0*(1.0+xi);

  if(x <= xmin){
    b0 = -xi;
    t1 = -1.0;
    t2 = xi;
  
    f = b0;
    if(k_in > 0){
      for(l=1; l<=k_in; l++){
        t1 = t1*((double)(2*l-1));
        t2 = t2*xi;
        b_dum = t2*t1;
      }
    f = b_dum;
    }
  }

  if(x > xmin){
    if (k_in==0){
      f = b0;
    }
    else if (k_in==1){
      f = b1;
    }
    else{
      b_dum = 0.0;
      for(l=2; l<=k_in; l++){
        b_dum = ((double)(2*l-1))*b1*xi + b0;
        b0 = b1;
        b1 = b_dum;
      }
      f = b_dum;
    }
  }
  return f;

} 



/***********************************************************************
   Computes variations of the spherical Bessel function, $a_l(x)=i^lj_l(ix)$,
   for real argument $x$ and $l=0,1,\ldots,l_{\rm max}$. The recursion relation
   $$ j_{l+1}(x)=\frac{2l+1}{x}j_l(x)+j_{l-1}(x) $$
   is used upwards. For starting values there are
   $$ a_0(x)=\frac{\sinh(x)}{x};\qquad a_1(x)=\frac{a_0(x)-\cosh(x)}{x} $$.
   For $x\ll 1$ the asymtotic forms
   $$ a_l(x)\approx\frac{(-x)^l}{(2l+1)!!} $$
   are used.
***********************************************************************/


static double Bessel_j(int l2max, int k_in, double x){

  int l;
  double xi,xmin,t1,t2;
  double a0,a1,a_dum;
  double f;

  if(l2max==0){ 
    xmin = 1.0e-6;
  }
  else if(l2max==2){
    xmin = 1.0e-4;
  }
  else if(l2max==4){
    xmin = 1.0e-2;
  }
  else{
    xmin = 1.0;
  }

  xi = 1.0/x; 


  a0 = xi*sinh(x);
  a1 = xi*(a0-cosh(x));

  if(x <= xmin){
    a0 = 1.0;
    t1 = 1.0;
    t2 = 1.0;
  
    f = a0;
    if(k_in > 0){
      for(l=1; l<=k_in; l++){
        t1 = t1/((double)(2*l+1));
        t2 = -t2*x;
        a_dum = t2*t1;
      }
    f = a_dum;
    }
  }

  if(x > xmin){
    if (k_in==0){
      f = a0;
    }
    else if (k_in==1){
      f = a1;
    }
    else{
      a_dum = 0.0;
      for(l=2; l<=k_in; l++){
        a_dum = ((double)(2*l-1))*a1*xi + a0;
        a0 = a1;
        a1 = a_dum;
      }
      f = a_dum;
    }
  }
  return f;

} 




/**********************************************************************
  Calculating Coulomb interaction matrix in real harmonics basis.
  U(a,b,c,d) = <a, b | Vee | c, d> 
             = Sum_over_m1,m2,m3,m4[S_a,m1 * S_b,m2 * 
               {sum_over_k=0,2,4,..[a_k(m1,m2,m3,m4)*F[k]}* 
               S^*_c,m3 * S^*_d,m4
***********************************************************************/

static double Coulomb_Matrix(int num, double F0, double JH, int l, int a, int b, int c, int d){
	
  double Slater_F[7];
  double Coulomb_Sph;
  dcomplex Coulomb_Real, TRANS;
  int i,j,m,n,kk;

  if(Yukawa_on==1){
    switch (l){
    case 1:
      Slater_F[0] = F0;
      Slater_F[2] = Slater_F2[num];
    break;
  
    case 2:	
      Slater_F[0] = F0;
      Slater_F[2] = Slater_F2[num];
      Slater_F[4] = Slater_F4[num];
    break;
  
    case 3:
      Slater_F[0] = F0; 
      Slater_F[2] = Slater_F2[num];
      Slater_F[4] = Slater_F4[num];
      Slater_F[6] = Slater_F6[num];
    break;
    }
  }
  else{
    switch (l){
    case 1:
      Slater_F[0] = F0;
      Slater_F[2] = 5.0*JH;
    break;
  
    case 2:	
      Slater_F[0] = F0;
      Slater_F[2] = (14.0/(1.0 + slater_ratio))*JH;
      Slater_F[4] = slater_ratio*Slater_F[2]; 
    break;
  
    case 3:
      Slater_F[0] = F0;
      Slater_F[2] = (6435.0/(286.0 + 195.0*0.668 + 250.0*0.494))*JH;
      Slater_F[4] = 0.668*Slater_F[2];	
      Slater_F[6] = 0.494*Slater_F[2];
    break;
    }
  }

  Coulomb_Real = Complex(0.0,0.0);
    for (i=0; i<(2*l+1); i++){
      for (j=0; j<(2*l+1); j++){
  	for (m=0; m<(2*l+1); m++){
    	  for (n=0; n<(2*l+1); n++){
	    TRANS = Cmul(Cmul(StoR(l,a,i),StoR(l,b,j)),Cmul(Conjg(StoR(l,c,m)),Conjg(StoR(l,d,n))));
	    Coulomb_Sph = 0.0;
	    for (kk=0; kk<=2*l; kk+=2){
	      if ((i+j) == (m+n)){
	 	Coulomb_Sph += Gaunt_SR(l,kk,l-i,l-m)*Gaunt_SR(l,kk,l-n,l-j)*Slater_F[kk];
	      }
 	      else{
		Coulomb_Sph += 0.0;
	      }
	    }
	    Coulomb_Real.r += Cmul(TRANS, Complex(Coulomb_Sph,0.0)).r;
	  }
	}
      }
    }


  return Coulomb_Real.r;

}



/*********************************************************************
  Transformation matrix from spherical to real harmonics, 
  S_m1,m2 (0 <= (m1, m2) < 2l+1).

  Real | Spherical
  y_l0 = Y^l_0
  y_lm = 1/sqrt(2)*[Y^l_-m + (-1)^m*Y^l_m]
  y_l(-m) = i/sqrt(2)*[Y^l_-m - (-1)^m*Y^l_m] 


  Real  |   Spherical 
 
  px  	S   |11>
  py   <--  |10>
  pz 	    |1-1>
 -----------------------------
  dz^2			|22>
  dx^2-y^2	 S	|21>
  dxy		<--	|20>
  dzx			|2-1>
  dyz			|2-2>
 -----------------------------
  fz^3			|33>
  fxz^2			|32>
  fyz^2		 S	|31>
  fz(x^2-y^2)	<--	|30>
  fxyz			|3-1>
  fx^3-3xy^2		|3-2>
  f3yx^2-y^3		|3-3>
**********************************************************************/

static dcomplex StoR(int l, int m1, int m2){

  dcomplex result, S2R;

  result = Complex(0.0,0.0);
  switch(l){
  case 1:
    if (m1 == 0){
      if (m2 != 1){
 	result = Complex((double)(m2-l), 0.0);
      }
      else{
	result = Complex(0.0,0.0);
      } 
    }
    else if (m1 == 1){
      if (m2 != 1){
	result = Complex(0.0,1.0);
      }	
      else{
	result = Complex(0.0,0.0);
      }
    }
    else{
      if (m1 ==2 && m2 == 1){
	result = Complex(sqrt(2.0),0.0);
      }
      else{
	result = Complex(0.0,0.0);
      }
    }
  break;

  case 2:
    if (m1 == 0){
      if (m2 == 2){
        result = Complex(sqrt(2.0),0.0);
      }
      else{
        result = Complex(0.0,0.0);
      }
    }
    else if (m1 == 1){
      if (m2 == 0 || m2 == 4){
        result = Complex(1.0,0.0);
      }
      else{
        result = Complex(0.0,0.0);
      }
    }
    else if (m1 == 2){
      if (m2 == 0 || m2 == 4){
        result = Complex(0.0,((double)(m2-l)/2.0));
      }
      else{
        result = Complex(0.0,0.0);
      }
    }
    else if (m1 == 3){
      if (m2 == 1 || m2 == 3){
	result = Complex((double)(m2-l),0.0);
      }
      else{
        result = Complex(0.0,0.0);
      }
    }
    else{
      if (m1 == 4 && (m2 == 1 || m2 == 3)){
        result = Complex(0.0,1.0);
      }
      else{
        result = Complex(0.0,0.0);
      }
    }
  break;
    
  case 3:
    if (m1 == 0 && m2 == 3){
      result = Complex(sqrt(2.0),0.0);
    }
    else if (m1 == 1){
      if (m2 == 2 || m2 == 4){
        result = Complex((double)(m2-l),0.0);
      }
    }
    else if (m1 == 2){
      if (m2 == 2 || m2 == 4){
        result = Complex(0.0,1.0);
      }
    }	
    else if (m1 == 3){
      if (m2 == 1 || m2 == 5){
        result = Complex(1.0,0.0);
      }
    }
    else if (m1 == 4){
      if (m2 == 1 || m2 == 5){
        result = Complex(0.0,((double)(m2-l)/2.0));
      }
    }
    else if (m1 == 5){
      if (m2 == 0 || m2 == 6){
        result = Complex((double)(m2-l)/3.0,0.0);
      }
    }
    else if (m1 == 6){
      if (m2 == 0 || m2 == 6){
        result = Complex(0.0,1.0);
      }
    }
    else{
      result = Complex(0.0,0.0);
    }
  break;
  }

  S2R = Cmul(result, Complex(1.0/sqrt(2.0),0.0));
  return S2R;
}




/*******************************************************************************
  Gaunt coeffcient calculator.
  c_k(l,m1;l,m3) = sqrt(4*pi/(2k+1)) * integral(Y^*_l,m1 * Y_k,m-m' * Y_l,m2)
********************************************************************************/

static double Gaunt_SR(int l, int k, int m1, int m2){

  double ck;
  ck = pow(-1.0, -(double)m1)*sqrt((2.0*(double)l+1.0)*(2.0*(double)l+1.0))*
       Wigner3j(l,k,l,0,0,0)*Wigner3j(l,k,l,-m1,m1-m2,m2);
	
  return ck;
}




/**************************************************************
  Wigner3-j symbol calculator using Racah formula.
  Algorithm is based on the python code by David Terr.

  / l1 l2 l3 \ 
  |	     |
  \ m1 m2 m3 /
***************************************************************/

static double Wigner3j(int l1, int l2, int l3, int m1, int m2, int m3){

  double t_coeff, tabc;
  int tc1, tc2, tc3, tc4;
  int a1, a2, b1, b2, c1, c2;
  int count;	
  int x1, x2, x3, x4, x5;
  double x;

  int t1, t2, t3, t4, t5;
  int tmin, tmax;



  tc1 = l1 + l2 - l3;
  tc2 = l1 - l2 + l3;
  tc3 = -l1 + l2 + l3;
  tc4 = l1 + l2 + l3 + 1;
  t_coeff = FAC(tc1)*FAC(tc2)*FAC(tc3)/FAC(tc4);

  a1 = (l1 + m1);
  a2 = (l1 - m1);
  b1 = (l2 + m2);
  b2 = (l2 - m2);
  c1 = (l3 + m3);
  c2 = (l3 - m3);
  tabc = pow(-1.0,(double)(l1-l2-m3))*sqrt(t_coeff*FAC(a1)*FAC(a2)*FAC(b1)*FAC(b2)*FAC(c1)*FAC(c2));

  t1 = l2 - m1 - l3;
  t2 = l1 + m2 - l3;
  t3 = l1 + l2 - l3;
  t4 = l1 - m1;
  t5 = l2 + m2;

  tmin = MAX(0, MAX(t1,t2));
  tmax = MIN(t3, MIN(t4,t5));
	
  x = 0.0;
  for (count=tmin; count <= tmax; count++){
    x1 = l3 - l2 + count + m1;
    x2 = l3 - l1 + count - m2;
    x3 = l1 + l2 - l3 - count;
    x4 = l1 - count - m1;
    x5 = l2 - count + m2;
		
    x = x + pow(-1.0, (double)count)/(FAC(count)*FAC(x1)*FAC(x2)*FAC(x3)*FAC(x4)*FAC(x5));
	
  }

  return tabc*x;

}



/****************************************************
  Functions to calculate the max. and min. values
*****************************************************/

static int MAX(int a, int b){
	
  if (a >= b){
    return a;
  }
  else{
    return b;
  }
}

static int MIN(int a, int b){
	
  if (a >= b){
    return b;
  }
  else{
    return a;
  }
}




/***********************************************
  FAC(n) = n!
************************************************/

static double FAC(int num){
  int i = 0;
  int result = 1;
  if (num != 0){
    for (i=num; i>0; i--){
      result = result*i;
    }
  }
  else if( num == 0){
    result = 1;	
  }

  return result;
}



