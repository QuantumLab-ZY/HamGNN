/**********************************************************************
  Poisson_ESM.c:

     Poisson_ESM.c is a subroutine to solve Poisson's equations
     for effective screening medium (ESM) method calculations.

  References for ESM method:

     [1] M.Otani and O.Sugino, PRB 73, 115407 (2006).
     [2] O.Sugino et al., Surf.Sci., 601, 5237 (2007).
     [3] M.Otani et al., J.Phys.Soc.Jp., 77, 024802 (2008).
     [4] T. Ohwaki et al., J. Chem. Phys., 136, 134101 (2012).

  Log of Poisson_ESM.c:

     01/10/2011  Released by T.Ohwaki and M.Otani
     01/05/2012  Revised by T.Ohwaki (for new division structure)
     04/06/2012  Revised by T.Ohwaki
     01/31/2019  modified by AdvanceSoft
     09/13/2019  modified by S.Hagiwara

***********************************************************************/

#define  measure_time   0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <fftw3.h> 

static void One_dim_FFT(FILE *fp, 
			int sgn1, int sgn2, int k23, 
			double *ReF2, double *ImF2,
                        double prefac);


double Poisson_ESM(int fft_charge_flag,
                   double *ReRhok, double *ImRhok)
{ 
  int i,j,k,k1,k2,k3,kz1;
  int BN_CB,N2D,GNs,GN;
  double GridArea,tmpv[4];
  double sk1,sk2,sk3;
  double x,y,z,Gx,Gy,Gz,G2;
  double time0,TStime,TEtime;

  double Gz2,Gp,Gp2,z0,z1,zz;
  double ReRhok_tmp,ImRhok_tmp,sin_tmp,cos_tmp,exp_tmp1,exp_tmp2,exp_tmp3;
  double tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,PI2,PI4;
  double tmp1r,tmp1i,tmp2r,tmp2i,tmp3r,tmp3i,tmp4r,tmp4i,tmp5r,tmp5i;

  FILE *fp;
  char fname[YOUSO10];

  /* MPI */
  int numprocs,myid,tag=999,ID,IDS,IDR;
  MPI_Status stat;
  MPI_Request request;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  fftw_complex *in0, *out0;
  fftw_plan p0;
    
  double *ReRhok_r,*ImRhok_r;
  int nz_l,nz_r;
  double z_l,z_r,f1,f2,f3,f4,a0,a1,a2,a3;
    
  // !!! This parameter must be set from the input file !!!
  int esm_nfit;
  esm_nfit = 4;
    
  /****************************************************
    allocation of arrays:

    fftw_complex  in[List_YOUSO[17]];
    fftw_complex out[List_YOUSO[17]];
  ****************************************************/

  in0  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]);
  out0 = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]);

 ReRhok_r = (double*)malloc(sizeof(double)*Ngrid1);
 ImRhok_r = (double*)malloc(sizeof(double)*Ngrid1);

  if (myid==Host_ID) { 
    printf("<Poisson_ESM> Poisson's equation using FFT & ESM...\n");
    if (ESM_switch == 1) printf("<Poisson_ESM> Boundary condition = vacuum | cell | vacuum\n");
    if (ESM_switch == 2) printf("<Poisson_ESM> Boundary condition = metal| | cell | metal \n");
    if (ESM_switch == 3) printf("<Poisson_ESM> Boundary condition = vacuum | cell | metal \n");
    if (ESM_switch == 4) printf("<Poisson_ESM> Boundary condition = uniform electric field \n");
  }

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  Cross_Product(gtv[2],gtv[3],tmpv);
  GridArea = sqrt(tmpv[1]*tmpv[1] + tmpv[2]*tmpv[2] + tmpv[3]*tmpv[3]) * 0.529177249;

  PI2 = 2.0*PI;
  PI4 = 4.0*PI;

  /****************************************************
                 FFT of charge density 
  ****************************************************/

  if (fft_charge_flag==1) FFT_Density(0,ReRhok,ImRhok);

  if (myid==Host_ID){ 
    printf("<Poisson_ESM> Total number of electrons = %12.9f \n",ReRhok[0]*GridVol);fflush(stdout);
  }
/*
  if (myid==Host_ID){

    sprintf(fname,"%s%s.ESM.dcharge",filepath,filename);

    if ( (fp=fopen(fname,"w"))!=NULL ) {

      fprintf(fp,"## Check of difference charge density (e/Ang)      ##\n");fflush(stdout);
      fprintf(fp,"## Grid : x-corrdinate (Ang) : delta-rho(G_||=0,z) ##\n");fflush(stdout);

      One_dim_FFT(fp,1,1,0,ReRhok,ImRhok,GridArea);

      fprintf(fp,"\n");
      fclose(fp);
    }
    else {
      printf("Failure in saving *.ESM.dcharge.\n");
    }
  }
*/
  /************************************************************

      One_dim_FFT(fp,n1,n2,m1,m2,ReV,ImV,prefac)

       n1= 1: for check of output (ReV & ImV are not changed)
         = 2: for calculation (ReV & ImV are changed)
       n2= 1: FFT
         =-1: inverse FFT
       m1   : Gx + Gy
       prefac: prefactor of output data

  *************************************************************/

  tmp0 =1.0/(double)(Ngrid1*Ngrid2*Ngrid3);

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  /*******************************************************************************/
  /*  taking phase factor coming from grid-origin shift ( rho{Gz}*exp{i*Gz*dz} ) */
  /*******************************************************************************/

  z0 = tv[1][ESM_direction]/2.0; /* modified by AdvanceSoft */

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

      for (k1=0; k1<Ngrid1; k1++){

	if (k1<Ngrid1/2) sk1 = (double)k1;
	else             sk1 = (double)(k1 - Ngrid1);

	Gz  = sk1*rtv[1][ESM_direction]; /* modified by AdvanceSoft */

        ReRhok_tmp = ReRhok[BN_CB+k1];
        ImRhok_tmp = ImRhok[BN_CB+k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

	tmp1r = ReRhok_tmp*cos_tmp + ImRhok_tmp*sin_tmp;
	tmp1i = ImRhok_tmp*cos_tmp - ReRhok_tmp*sin_tmp;

	ReRhok[BN_CB+k1] = tmp1r;
	ImRhok[BN_CB+k1] = tmp1i;

      }
    }


  /***************************************************************************/
  /*  "vacuum|vacuum|vacuum" boundary condition (for isolated slab systems)  */
  /***************************************************************************/

  if (ESM_switch==1){

    p0 = fftw_plan_dft_1d(Ngrid1, in0, out0, 1, FFTW_ESTIMATE);

    /* * * *  G_|| != 0  * * * */

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    GN = BN_CB + GNs;
    k3 = GN / (Ngrid2 * Ngrid1);
    k2 = (GN - k3 * Ngrid2 * Ngrid1) / Ngrid1;

        if (k2<Ngrid2/2) sk2 = (double)k2;
        else             sk2 = (double)(k2 - Ngrid2);

        if (k3<Ngrid3/2) sk3 = (double)k3;
        else             sk3 = (double)(k3 - Ngrid3);

	/* modified by AdvanceSoft */
	Gx  = sk2*rtv[2][iESM[2]] + sk3*rtv[3][iESM[2]];  /* original Gy,Gz -> Gx,Gy */
	Gy  = sk2*rtv[2][iESM[3]] + sk3*rtv[3][iESM[3]];
	Gp2 = Gx*Gx + Gy*Gy;
	Gp  = sqrt(Gp2);

	if (k2!=0 || k3!=0){

	  tmp1r = 0.0; tmp1i = 0.0; tmp2r = 0.0; tmp2i = 0.0;

	  for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

	    if (k1<Ngrid1/2) sk1 = (double)k1;
	    else             sk1 = (double)(k1 - Ngrid1);

	    Gz  = sk1*rtv[1][ESM_direction];  /* original Gx -> Gz */ /* modified by AdvanceSoft */
	    Gz2 = Gz*Gz;
	    G2  = Gp2 + Gz2;

        ReRhok_tmp = ReRhok[BN_CB+k1];
        ImRhok_tmp = ImRhok[BN_CB+k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

	    tmp1r +=  (ReRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
		     - ImRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*tmp0;
	    tmp1i +=  (ImRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
		     + ReRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*tmp0;

	    tmp2r +=  (ReRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
		     + ImRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*tmp0;
	    tmp2i +=  (ImRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
		     - ReRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*tmp0;

	    ReRhok[BN_CB+k1] = PI4*ReRhok_tmp/G2*tmp0;
	    ImRhok[BN_CB+k1] = PI4*ImRhok_tmp/G2*tmp0;

	  } /* end of k1-loop */

     /* 1-D FFT for Gz -> z */

       for (k1=0; k1<Ngrid1; k1++){

         in0[k1][0] = ReRhok[BN_CB+k1];
         in0[k1][1] = ImRhok[BN_CB+k1];

       }

       fftw_execute(p0);

       for (k1=0; k1<Ngrid1; k1++){

           ReRhok[BN_CB+k1] = out0[k1][0];
           ImRhok[BN_CB+k1] = out0[k1][1];

       }

     /* end of 1-D FFT */

	  for (k1=0; k1<Ngrid1; k1++){  /* z-loop */

	    if (k1<Ngrid1/2) sk1 = (double)k1;
	    else             sk1 = (double)(k1 - Ngrid1);

	    zz = sk1*tv[1][ESM_direction]/(double)Ngrid1; /* modified by AdvanceSoft */

            exp_tmp1 = -PI2*exp( Gp*(zz-z0))/Gp;
            exp_tmp2 = -PI2*exp(-Gp*(zz+z0))/Gp;

	    ReRhok[BN_CB+k1] += exp_tmp1*tmp1r + exp_tmp2*tmp2r;
	    ImRhok[BN_CB+k1] += exp_tmp1*tmp1i + exp_tmp2*tmp2i;

	  } /* end of k1-loop */

	} /* end of if */

      } /* end of BN_CB-loop */

    /* * * *  End of G_|| != 0 case  * * * */


    /* * * *  G_|| = 0  * * * */

    if(myid==Host_ID){

      tmp5r = ReRhok[0];
      tmp5i = ImRhok[0];

      ReRhok[0] = -PI2*(z0*z0)*ReRhok[0]*tmp0;
      ImRhok[0] = -PI2*(z0*z0)*ImRhok[0]*tmp0;

      tmp1r = 0.0; tmp1i = 0.0; tmp2r = 0.0; tmp2i = 0.0; tmp3r = 0.0; tmp3i = 0.0;

      for (k1=1; k1<Ngrid1; k1++){  /* Gz-loop */

	if (k1<Ngrid1/2) sk1 = (double)k1;
	else             sk1 = (double)(k1 - Ngrid1);

	Gz  = sk1*rtv[1][ESM_direction];  /* original Gx -> Gz */ /* modified by AdvanceSoft */
	Gz2 = Gz*Gz;

        ReRhok_tmp = ReRhok[k1];
        ImRhok_tmp = ImRhok[k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

	tmp1r += (-ReRhok_tmp*sin_tmp - ImRhok_tmp*cos_tmp)/Gz*tmp0;
	tmp1i += (-ImRhok_tmp*sin_tmp + ReRhok_tmp*cos_tmp)/Gz*tmp0;

	tmp2r += ( ReRhok_tmp*sin_tmp - ImRhok_tmp*cos_tmp)/Gz*tmp0;
	tmp2i += ( ImRhok_tmp*sin_tmp + ReRhok_tmp*cos_tmp)/Gz*tmp0;

	tmp3r += ReRhok_tmp*cos_tmp/Gz2*tmp0;
	tmp3i += ImRhok_tmp*cos_tmp/Gz2*tmp0;

	ReRhok[k1] = PI4*ReRhok_tmp/Gz2*tmp0;
	ImRhok[k1] = PI4*ImRhok_tmp/Gz2*tmp0;

      } /* end of k1-loop */

     /* 1-D FFT for Gz -> z */

       for (k1=0; k1<Ngrid1; k1++){

         in0[k1][0] = ReRhok[k1];
         in0[k1][1] = ImRhok[k1];

       }

       fftw_execute(p0);

       for (k1=0; k1<Ngrid1; k1++){

           ReRhok[k1] = out0[k1][0];
           ImRhok[k1] = out0[k1][1];

       }

     /* end of 1-D FFT */

      for (k1=0; k1<Ngrid1; k1++){  /* z-loop */

	if (k1<Ngrid1/2) sk1 = (double)k1;
	else             sk1 = (double)(k1 - Ngrid1);

	zz = sk1*tv[1][ESM_direction]/(double)Ngrid1; /* modified by AdvanceSoft */

	ReRhok[k1] += -PI2*(zz*zz)*tmp5r*tmp0
	              -PI2*(zz-z0)*tmp1r    
                      -PI2*(zz+z0)*tmp2r    
                      -PI4*tmp3r;

	ImRhok[k1] += -PI2*(zz*zz)*tmp5i*tmp0
                      -PI2*(zz-z0)*tmp1i    
                      -PI2*(zz+z0)*tmp2i    
                      -PI4*tmp3i;

      } /* end of k1-loop */
   
        
    nz_l = Ngrid1/2+esm_nfit;
    nz_r = Ngrid1/2-esm_nfit;
    z_l = tv[1][ESM_direction]/(double)Ngrid1*(double)(nz_l-Ngrid1);
    z_r = tv[1][ESM_direction]/(double)Ngrid1*(double)(nz_r);
    
    f1 = ReRhok[nz_r];
    f2 = ReRhok[nz_l];
    f3 = -PI4*z_r*tmp5r*tmp0
        -PI2*tmp1r
        -PI2*tmp2r;
    f4 = -PI4*z_l*tmp5r*tmp0
        -PI2*tmp1r
        -PI2*tmp2r;
    z_l += tv[1][ESM_direction];
    a0 = (f1*z_l*z_l*(z_l-3.0*z_r)+z_r*(f3*z_l*z_l*(-z_l+z_r)
                                        +z_r*(f2*(3.0*z_l-z_r)+f4*z_l*(-z_l+z_r))))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a1 = (f3*z_l*z_l*z_l+z_l*(6.0*f1-6.0*f2+(f3+2.0*f4)*z_l)*z_r
          -(2*f3+f4)*z_l*z_r*z_r-f4*z_r*z_r*z_r)/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a2 = (-3*f1*(z_l+z_r)+3.0*f2*(z_l+z_r)-(z_l-z_r)*(2*f3*z_l
                                                      +f4*z_l+f3*z_r+2*f4*z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a3 = (2.0*f1-2.0*f2+(f3+f4)*(z_l-z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    for (k1=nz_r; k1<=nz_l; k1++){
        zz = (double)k1*tv[1][ESM_direction]/(double)Ngrid1;
        ReRhok[k1] = a0+(a1+(a2+a3*zz)*zz)*zz;
    }
    
    z_l -= tv[1][ESM_direction];
    f1 = ImRhok[nz_r];
    f2 = ImRhok[nz_l];
    f3 = -PI4*z_r*tmp5i*tmp0
        -PI2*tmp1i
        -PI2*tmp2i;
    f4 = -PI4*z_l*tmp5i*tmp0
        -PI2*tmp1i
        -PI2*tmp2i;
    z_l += tv[1][ESM_direction];
    a0 = (f1*z_l*z_l*(z_l-3.0*z_r)+z_r*(f3*z_l*z_l*(-z_l+z_r)
                                        +z_r*(f2*(3.0*z_l-z_r)+f4*z_l*(-z_l+z_r))))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a1 = (f3*z_l*z_l*z_l+z_l*(6.0*f1-6.0*f2+(f3+2.0*f4)*z_l)*z_r
          -(2*f3+f4)*z_l*z_r*z_r-f4*z_r*z_r*z_r)/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a2 = (-3*f1*(z_l+z_r)+3.0*f2*(z_l+z_r)-(z_l-z_r)*(2*f3*z_l
                                                      +f4*z_l+f3*z_r+2*f4*z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a3 = (2.0*f1-2.0*f2+(f3+f4)*(z_l-z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    for (k1=nz_r; k1<=nz_l; k1++){
        zz = (double)k1*tv[1][ESM_direction]/(double)Ngrid1;
        ImRhok[k1] = a0+(a1+(a2+a3*zz)*zz)*zz;
        }

 
 
        
 /*
    for (k1=0; k1<Ngrid1; k1++){
        ReRhok[k1] += ReRhok_r[k1];
        ImRhok[k1] += ImRhok_r[k1];
        }
*/
        
    } /* end of if myid==Host_ID */

    /* * * *  End of G_|| = 0 case  * * * */

  } /* ESM_switch==1 */  


  /************************************************************/
  /*  "metal|vacuum|metal" boundary condition (ESM_switch==2) */
  /*  "inside-capacitor"   boundary condition (ESM_switch==4) */
  /************************************************************/

  else if (ESM_switch==2 || ESM_switch==4){

    /* Hagiwara modified */
    z1 = z0;  /* z1: position of ideal metal surface */

    p0 = fftw_plan_dft_1d(Ngrid1, in0, out0, 1, FFTW_ESTIMATE);

    /* * * *  G_|| != 0  * * * */

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    GN = BN_CB + GNs;
    k3 = GN / (Ngrid2 * Ngrid1);
    k2 = (GN - k3 * Ngrid2 * Ngrid1) / Ngrid1;

        if (k2<Ngrid2/2) sk2 = (double)k2;
        else             sk2 = (double)(k2 - Ngrid2);

        if (k3<Ngrid3/2) sk3 = (double)k3;
        else             sk3 = (double)(k3 - Ngrid3);

        /* modified by AdvanceSoft */
        Gx  = sk2*rtv[2][iESM[2]] + sk3*rtv[3][iESM[2]]; 
        Gy  = sk2*rtv[2][iESM[3]] + sk3*rtv[3][iESM[3]];
        Gp2 = Gx*Gx + Gy*Gy;
        Gp  = sqrt(Gp2);

        if (k2!=0 || k3!=0){

          tmp1r = 0.0; tmp1i = 0.0; tmp2r = 0.0; tmp2i = 0.0;
          tmp3r = 0.0; tmp3i = 0.0; tmp4r = 0.0; tmp4i = 0.0;

          for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

            if (k1<Ngrid1/2) sk1 = (double)k1;
            else             sk1 = (double)(k1 - Ngrid1);

            Gz  = sk1*rtv[1][ESM_direction]; /* modified by AdvanceSoft */
            Gz2 = Gz*Gz;
            G2  = Gp2 + Gz2;

        ReRhok_tmp = ReRhok[BN_CB+k1];
        ImRhok_tmp = ImRhok[BN_CB+k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

        exp_tmp1 = G2*exp(Gp*(z1-z0))/tmp0;
        exp_tmp2 = G2*exp(Gp*(z0-z1))/tmp0;

            tmp1r +=  (ReRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     - ImRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/exp_tmp1;
            tmp1i +=  (ImRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     + ReRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/exp_tmp1;

            tmp2r +=  (ReRhok_tmp*(Gp*cos_tmp + Gz*sin_tmp)
                     - ImRhok_tmp*(Gp*sin_tmp - Gz*cos_tmp))/exp_tmp2;
            tmp2i +=  (ImRhok_tmp*(Gp*cos_tmp + Gz*sin_tmp)
                     + ReRhok_tmp*(Gp*sin_tmp - Gz*cos_tmp))/exp_tmp2;

            tmp3r +=  (ReRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     + ImRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/exp_tmp1;
            tmp3i +=  (ImRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     - ReRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/exp_tmp1;

            tmp4r +=  (ReRhok_tmp*(Gp*cos_tmp + Gz*sin_tmp)
                     + ImRhok_tmp*(Gp*sin_tmp - Gz*cos_tmp))/exp_tmp2;
            tmp4i +=  (ImRhok_tmp*(Gp*cos_tmp + Gz*sin_tmp)
                     - ReRhok_tmp*(Gp*sin_tmp - Gz*cos_tmp))/exp_tmp2;

            ReRhok[BN_CB+k1] = PI4*ReRhok_tmp/G2*tmp0;
            ImRhok[BN_CB+k1] = PI4*ImRhok_tmp/G2*tmp0;

          } /* end of k1-loop */

     /* 1-D FFT for Gz -> z */

       for (k1=0; k1<Ngrid1; k1++){

         in0[k1][0] = ReRhok[BN_CB+k1];
         in0[k1][1] = ImRhok[BN_CB+k1];

       }

       fftw_execute(p0);

       for (k1=0; k1<Ngrid1; k1++){

           ReRhok[BN_CB+k1] = out0[k1][0];
           ImRhok[BN_CB+k1] = out0[k1][1];

       }

     /* end of 1-D FFT */

          for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

            if (k1<Ngrid1/2) sk1 = (double)k1;
            else             sk1 = (double)(k1 - Ngrid1);

            zz = sk1*tv[1][ESM_direction]/(double)Ngrid1; /* modified by AdvanceSoft */

        exp_tmp1 = -PI4/Gp/(1.0-exp(-4.0*Gp*z1));
        exp_tmp2 = exp(Gp*(zz-z1)) - exp(-Gp*(zz+3.0*z1));
        exp_tmp3 = exp(Gp*(zz-3.0*z1)) - exp(-Gp*(zz+z1));

            ReRhok[BN_CB+k1] += exp_tmp1*(exp_tmp2*(tmp1r + tmp2r) + exp_tmp3*(tmp3r + tmp4r)); 
            ImRhok[BN_CB+k1] += exp_tmp1*(exp_tmp2*(tmp1i + tmp2i) + exp_tmp3*(tmp3i + tmp4i));

          } /* end of k1-loop */

        } /* end of if */

    } /* end of BN_CB-loop */

    /* * * *  End of G_|| != 0 case  * * * */


    /* * * *  G_|| = 0  * * * */

    if(myid==Host_ID){

      tmp5r = ReRhok[0];
      tmp5i = ImRhok[0];

      ReRhok[0] = -PI2*(z0*z0 - 2.0*z0*z1)*ReRhok[0]*tmp0;
      ImRhok[0] = -PI2*(z0*z0 - 2.0*z0*z1)*ImRhok[0]*tmp0;

      tmp1r = 0.0; tmp1i = 0.0; tmp2r = 0.0; tmp2i = 0.0;
      tmp3r = 0.0; tmp3i = 0.0; tmp4r = 0.0; tmp4i = 0.0;

      for (k1=1; k1<Ngrid1; k1++){  /* k1-loop */

        if (k1<Ngrid1/2) sk1 = (double)k1;
        else             sk1 = (double)(k1 - Ngrid1);

        Gz  = sk1*rtv[1][ESM_direction]; /* modified by AdvanceSoft */
        Gz2 = Gz*Gz;

        ReRhok_tmp = ReRhok[k1];
        ImRhok_tmp = ImRhok[k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

        tmp1r += ( ReRhok_tmp*cos_tmp - ImRhok_tmp*sin_tmp)/Gz2*tmp0;
        tmp1i += ( ImRhok_tmp*cos_tmp + ReRhok_tmp*sin_tmp)/Gz2*tmp0;

        tmp2r += ( ReRhok_tmp*cos_tmp + ImRhok_tmp*sin_tmp)/Gz2*tmp0;
        tmp2i += ( ImRhok_tmp*cos_tmp - ReRhok_tmp*sin_tmp)/Gz2*tmp0;

        tmp3r += ( ReRhok_tmp*sin_tmp)/Gz*tmp0;
        tmp3i += ( ImRhok_tmp*sin_tmp)/Gz*tmp0;

        tmp4r += (-ImRhok_tmp*cos_tmp)/Gz*tmp0;
        tmp4i += ( ReRhok_tmp*cos_tmp)/Gz*tmp0;

        ReRhok[k1] = PI4*ReRhok_tmp/Gz2*tmp0;
        ImRhok[k1] = PI4*ImRhok_tmp/Gz2*tmp0;

      } /* end of k1-loop */

     /* 1-D FFT for Gz -> z */

       for (k1=0; k1<Ngrid1; k1++){

         in0[k1][0] = ReRhok[k1];
         in0[k1][1] = ImRhok[k1];

       }

       fftw_execute(p0);

       for (k1=0; k1<Ngrid1; k1++){

           ReRhok[k1] = out0[k1][0];
           ImRhok[k1] = out0[k1][1];

       }

     /* end of 1-D FFT */

      for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

        if (k1<Ngrid1/2) sk1 = (double)k1;
        else             sk1 = (double)(k1 - Ngrid1);

        zz = sk1*tv[1][ESM_direction]/(double)Ngrid1; /* modified by AdvanceSoft */

        ReRhok[k1] += -PI2*(zz*zz)*tmp5r*tmp0
                      -PI2*(zz+z1)/z1*tmp1r
                      +PI2*(zz-z1)/z1*tmp2r
                      +PI4*(z1-z0)*tmp3r
                      -PI4*(z1-z0)*tmp4r*zz/z1
                      -0.5*V_ESM*(zz-z1)/z1; /* <-- for the case ESM_switch==4 */

        ImRhok[k1] += -PI2*(zz*zz)*tmp5i*tmp0
                      -PI2*(zz+z1)/z1*tmp1i
                      +PI2*(zz-z1)/z1*tmp2i
                      +PI4*(z1-z0)*tmp3i
                      -PI4*(z1-z0)*tmp4i*zz/z1;

      } /* end of k1-loop */
        
    /* start smoothing */
    nz_l = Ngrid1/2+esm_nfit;
    nz_r = Ngrid1/2-esm_nfit;
    z_l = tv[1][ESM_direction]/(double)Ngrid1*(double)(nz_l-Ngrid1);
    z_r = tv[1][ESM_direction]/(double)Ngrid1*(double)(nz_r);
    
    f1 = ReRhok[nz_r];
    f2 = ReRhok[nz_l];
    f3 = -PI4*z_r*tmp5r*tmp0
    -PI2/z1*tmp1r
    +PI2/z1*tmp2r
    -PI4*(z1-z0)*tmp4r/z1
    -0.5*V_ESM/z1;
    f4 = -PI4*z_l*tmp5r*tmp0
    -PI2/z1*tmp1r
    +PI2/z1*tmp2r
    -PI4*(z1-z0)*tmp4r/z1
    -0.5*V_ESM/z1;
    z_l += tv[1][ESM_direction];
    a0 = (f1*z_l*z_l*(z_l-3.0*z_r)+z_r*(f3*z_l*z_l*(-z_l+z_r)
                                        +z_r*(f2*(3.0*z_l-z_r)+f4*z_l*(-z_l+z_r))))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a1 = (f3*z_l*z_l*z_l+z_l*(6.0*f1-6.0*f2+(f3+2.0*f4)*z_l)*z_r
          -(2*f3+f4)*z_l*z_r*z_r-f4*z_r*z_r*z_r)/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a2 = (-3*f1*(z_l+z_r)+3.0*f2*(z_l+z_r)-(z_l-z_r)*(2*f3*z_l
                                                      +f4*z_l+f3*z_r+2*f4*z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a3 = (2.0*f1-2.0*f2+(f3+f4)*(z_l-z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    for (k1=nz_r; k1<=nz_l; k1++){ /* z-loop */
        zz = (double)k1*tv[1][ESM_direction]/(double)Ngrid1;
        ReRhok[k1] = a0+(a1+(a2+a3*zz)*zz)*zz;
    }
    
    z_l -= tv[1][ESM_direction];
    f1 = ImRhok[nz_r];
    f2 = ImRhok[nz_l];
    f3 = -PI4*z_r*tmp5i*tmp0
    -PI2/z1*tmp1i
    +PI2/z1*tmp2i
    -PI4*(z1-z0)*tmp4i/z1;
    f4 = -PI4*z_l*tmp5i*tmp0
    -PI2/z1*tmp1i
    +PI2/z1*tmp2i
    -PI4*(z1-z0)*tmp4i/z1;
    z_l += tv[1][ESM_direction];
    a0 = (f1*z_l*z_l*(z_l-3.0*z_r)+z_r*(f3*z_l*z_l*(-z_l+z_r)
                                        +z_r*(f2*(3.0*z_l-z_r)+f4*z_l*(-z_l+z_r))))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a1 = (f3*z_l*z_l*z_l+z_l*(6.0*f1-6.0*f2+(f3+2.0*f4)*z_l)*z_r
          -(2*f3+f4)*z_l*z_r*z_r-f4*z_r*z_r*z_r)/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a2 = (-3*f1*(z_l+z_r)+3.0*f2*(z_l+z_r)-(z_l-z_r)*(2*f3*z_l
                                                      +f4*z_l+f3*z_r+2*f4*z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a3 = (2.0*f1-2.0*f2+(f3+f4)*(z_l-z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    for (k1=nz_r; k1<=nz_l; k1++){ /* z-loop */
        zz = (double)k1*tv[1][ESM_direction]/(double)Ngrid1;
        ImRhok[k1] = a0+(a1+(a2+a3*zz)*zz)*zz;
        }
    /* end of smoothing */
    
        /*
         for (k1=0; k1<Ngrid1; k1++){
         ReRhok[k1] += ReRhok_r[k1];
         ImRhok[k1] += ImRhok_r[k1];
         }
         */
    } /* end of if myid==Host_ID */
      
    /* * * *  End of G_|| = 0 case  * * * */

  } /* ESM_switch==2 or 4 */


  /****************************************************************************/
  /*  "vacuum|vacuum|metal" boundary condition (for electrochemical systems)  */
  /****************************************************************************/

  else if (ESM_switch==3){

    z1 = z0;  /* z1: position of ideal metal surface */

    p0 = fftw_plan_dft_1d(Ngrid1, in0, out0, 1, FFTW_ESTIMATE);

    /* * * *  G_|| != 0  * * * */

  for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    GN = BN_CB + GNs;
    k3 = GN / (Ngrid2 * Ngrid1);
    k2 = (GN - k3 * Ngrid2 * Ngrid1) / Ngrid1;

        if (k2<Ngrid2/2) sk2 = (double)k2;
        else             sk2 = (double)(k2 - Ngrid2);

        if (k3<Ngrid3/2) sk3 = (double)k3;
        else             sk3 = (double)(k3 - Ngrid3);

        /* modified by AdvanceSoft */
        Gx  = sk2*rtv[2][iESM[2]] + sk3*rtv[3][iESM[2]]; 
        Gy  = sk2*rtv[2][iESM[3]] + sk3*rtv[3][iESM[3]];
        Gp2 = Gx*Gx + Gy*Gy;
        Gp  = sqrt(Gp2);

        if (k2!=0 || k3!=0){

          tmp1r = 0.0; tmp1i = 0.0; tmp2r = 0.0; tmp2i = 0.0; tmp3r = 0.0; tmp3i = 0.0;

          for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

            if (k1<Ngrid1/2) sk1 = (double)k1;
            else             sk1 = (double)(k1 - Ngrid1);

            Gz  = sk1*rtv[1][ESM_direction]; /* modified by AdvanceSoft */
            Gz2 = Gz*Gz;
            G2  = Gp2 + Gz2;

        ReRhok_tmp = ReRhok[BN_CB+k1];
        ImRhok_tmp = ImRhok[BN_CB+k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

            tmp1r +=  (ReRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     - ImRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*exp(Gp*(z1-z0))*tmp0;
            tmp1i +=  (ImRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     + ReRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*exp(Gp*(z1-z0))*tmp0;

            tmp2r +=  (ReRhok_tmp*(Gp*cos_tmp + Gz*sin_tmp)
                     - ImRhok_tmp*(Gp*sin_tmp - Gz*cos_tmp))/G2*exp(Gp*(z0-z1))*tmp0;
            tmp2i +=  (ImRhok_tmp*(Gp*cos_tmp + Gz*sin_tmp)
                     + ReRhok_tmp*(Gp*sin_tmp - Gz*cos_tmp))/G2*exp(Gp*(z0-z1))*tmp0;

            tmp3r +=  (ReRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     + ImRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*tmp0;
            tmp3i +=  (ImRhok_tmp*(Gp*cos_tmp - Gz*sin_tmp)
                     - ReRhok_tmp*(Gp*sin_tmp + Gz*cos_tmp))/G2*tmp0;

            ReRhok[BN_CB+k1] = PI4*ReRhok_tmp/G2*tmp0;
            ImRhok[BN_CB+k1] = PI4*ImRhok_tmp/G2*tmp0;

          } /* end of k1-loop */

     /* 1-D FFT for Gz -> z */

       for (k1=0; k1<Ngrid1; k1++){

         in0[k1][0] = ReRhok[BN_CB+k1];
         in0[k1][1] = ImRhok[BN_CB+k1];

       }

       fftw_execute(p0);

       for (k1=0; k1<Ngrid1; k1++){

           ReRhok[BN_CB+k1] = out0[k1][0];
           ImRhok[BN_CB+k1] = out0[k1][1];

       }

     /* end of 1-D FFT */

          for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

            if (k1<Ngrid1/2) sk1 = (double)k1;
            else             sk1 = (double)(k1 - Ngrid1);

            zz = sk1*tv[1][ESM_direction]/(double)Ngrid1 /*- Grid_Origin[1]*/; /* modified by AdvanceSoft */

            exp_tmp1 = -PI2*exp(Gp*(zz-z1))/Gp;
            exp_tmp2 =  PI2*(exp(Gp*(zz-z0-2.0*z1))-exp(-Gp*(zz+z0)))/Gp;

            ReRhok[BN_CB+k1] += exp_tmp1*(tmp1r + tmp2r) + exp_tmp2*tmp3r;
            ImRhok[BN_CB+k1] += exp_tmp1*(tmp1i + tmp2i) + exp_tmp2*tmp3i;

          } /* end of k1-loop */

        } /* end of if */

    } /* end of BN_CB-loop */

    /* * * *  End of G_|| != 0 case  * * * */

    /* * * *  G_|| = 0  * * * */

    if(myid==Host_ID){

      tmp5r = ReRhok[0];
      tmp5i = ImRhok[0];

      ReRhok[0] = -PI2*(z0*z0 - 4.0*z0*z1)*ReRhok[0]*tmp0;
      ImRhok[0] = -PI2*(z0*z0 - 4.0*z0*z1)*ImRhok[0]*tmp0;

      tmp1r = 0.0; tmp1i = 0.0; tmp2r = 0.0; tmp2i = 0.0; tmp3r = 0.0; tmp3i = 0.0;

      for (k1=1; k1<Ngrid1; k1++){  /* k1-loop */

        if (k1<Ngrid1/2) sk1 = (double)k1;
        else             sk1 = (double)(k1 - Ngrid1);

        Gz  = sk1*rtv[1][ESM_direction]; /* modified by AdvanceSoft */
        Gz2 = Gz*Gz;

        ReRhok_tmp = ReRhok[k1];
        ImRhok_tmp = ImRhok[k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

        tmp1r += ( ReRhok_tmp*cos_tmp - ImRhok_tmp*sin_tmp)/Gz2*tmp0;
        tmp1i += ( ImRhok_tmp*cos_tmp + ReRhok_tmp*sin_tmp)/Gz2*tmp0;

        tmp2r += ( ReRhok_tmp*sin_tmp - ImRhok_tmp*cos_tmp)/Gz*tmp0;
        tmp2i += ( ImRhok_tmp*sin_tmp + ReRhok_tmp*cos_tmp)/Gz*tmp0;

        tmp3r += (-ReRhok_tmp*sin_tmp - ImRhok_tmp*cos_tmp)/Gz*tmp0;
        tmp3i += (-ImRhok_tmp*sin_tmp + ReRhok_tmp*cos_tmp)/Gz*tmp0;

        ReRhok[k1] = PI4*ReRhok_tmp/Gz2*tmp0;
        ImRhok[k1] = PI4*ImRhok_tmp/Gz2*tmp0;

      } /* end of k1-loop */



     /* 1-D FFT for Gz -> z */

       for (k1=0; k1<Ngrid1; k1++){

         in0[k1][0] = ReRhok[k1];
         in0[k1][1] = ImRhok[k1];

       }

       fftw_execute(p0);

       for (k1=0; k1<Ngrid1; k1++){

         ReRhok[k1] = out0[k1][0];
         ImRhok[k1] = out0[k1][1];

       }

     /* end of 1-D FFT */

      for (k1=0; k1<Ngrid1; k1++){  /* k1-loop */

        if (k1<Ngrid1/2) sk1 = (double)k1;
        else             sk1 = (double)(k1 - Ngrid1);

        zz = sk1*tv[1][ESM_direction]/(double)Ngrid1 /*- Grid_Origin[1]*/; /* modified by AdvanceSoft */

        ReRhok[k1] += -PI2*(zz*zz + 2.0*z0*zz)*tmp5r*tmp0
                      -PI4*tmp1r
                      -PI4*(zz-z1)*tmp2r
                      -PI4*(z1-z0)*tmp3r;

        ImRhok[k1] += -PI2*(zz*zz + 2.0*z0*zz)*tmp5i*tmp0
                      -PI4*tmp1i
                      -PI4*(zz-z1)*tmp2i
                      -PI4*(z1-z0)*tmp3i;

      } /* end of k1-loop */
        
    /* start smoothing */
    nz_l = Ngrid1/2+esm_nfit;
    nz_r = Ngrid1/2-esm_nfit;
    z_l = tv[1][ESM_direction]/(double)Ngrid1*(double)(nz_l-Ngrid1);
    z_r = tv[1][ESM_direction]/(double)Ngrid1*(double)(nz_r);
    
    f1 = ReRhok[nz_r];
    f2 = ReRhok[nz_l];
    f3 = -PI4*(z_r+z0)*tmp5r*tmp0
    -PI4*tmp2r;
    f4 = -PI4*(z_l+z0)*tmp5r*tmp0
    -PI4*tmp2r;
    z_l += tv[1][ESM_direction];
    a0 = (f1*z_l*z_l*(z_l-3.0*z_r)+z_r*(f3*z_l*z_l*(-z_l+z_r)
                                        +z_r*(f2*(3.0*z_l-z_r)+f4*z_l*(-z_l+z_r))))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a1 = (f3*z_l*z_l*z_l+z_l*(6.0*f1-6.0*f2+(f3+2.0*f4)*z_l)*z_r
          -(2*f3+f4)*z_l*z_r*z_r-f4*z_r*z_r*z_r)/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a2 = (-3*f1*(z_l+z_r)+3.0*f2*(z_l+z_r)-(z_l-z_r)*(2*f3*z_l
                                                      +f4*z_l+f3*z_r+2*f4*z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a3 = (2.0*f1-2.0*f2+(f3+f4)*(z_l-z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    for (k1=nz_r; k1<=nz_l; k1++){ /* z-loop */
        zz = (double)k1*tv[1][ESM_direction]/(double)Ngrid1;
        ReRhok[k1] = a0+(a1+(a2+a3*zz)*zz)*zz;
    }
    
    z_l -= tv[1][ESM_direction];
    f1 = ImRhok[nz_r];
    f2 = ImRhok[nz_l];
    f3 = -PI4*(z_r+z0)*tmp5i*tmp0
    -PI4*tmp2i;
    f4 = -PI4*(z_l+z0)*tmp5i*tmp0
    -PI4*tmp2i;
    z_l += tv[1][ESM_direction];
    a0 = (f1*z_l*z_l*(z_l-3.0*z_r)+z_r*(f3*z_l*z_l*(-z_l+z_r)
                                        +z_r*(f2*(3.0*z_l-z_r)+f4*z_l*(-z_l+z_r))))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a1 = (f3*z_l*z_l*z_l+z_l*(6.0*f1-6.0*f2+(f3+2.0*f4)*z_l)*z_r
          -(2*f3+f4)*z_l*z_r*z_r-f4*z_r*z_r*z_r)/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a2 = (-3*f1*(z_l+z_r)+3.0*f2*(z_l+z_r)-(z_l-z_r)*(2*f3*z_l
                                                      +f4*z_l+f3*z_r+2*f4*z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    a3 = (2.0*f1-2.0*f2+(f3+f4)*(z_l-z_r))/((z_l-z_r)*(z_l-z_r)*(z_l-z_r));
    for (k1=nz_r; k1<=nz_l; k1++){
        zz = (double)k1*tv[1][ESM_direction]/(double)Ngrid1;
        ImRhok[k1] = a0+(a1+(a2+a3*zz)*zz)*zz;
    }
    /* end of smoothing */
    
    /*
    for (k1=0; k1<Ngrid1; k1++){
        ReRhok[k1] += ReRhok_r[k1];
        ImRhok[k1] += ImRhok_r[k1];
    }
    */
      
    } /* end of if myid==Host_ID */

    /* * * *  End of G_|| = 0 case  * * * */

  } /* ESM_switch==3 */


  fftw_destroy_plan(p0);
  fftw_cleanup();


  /*************************************************************************
     V(G_||,z') -> V(G_||,z) change the order of z-coordinate mesh points  
  *************************************************************************/

    p0 = fftw_plan_dft_1d(Ngrid1, in0, out0, -1, FFTW_ESTIMATE);

    for (BN_CB=0; BN_CB<My_NumGridB_CB; BN_CB+=Ngrid1){

    /* 1-D FFT for z -> Gz */

      for (k1=0; k1<Ngrid1; k1++){

        in0[k1][0] = ReRhok[BN_CB+k1];
        in0[k1][1] = ImRhok[BN_CB+k1];

      }

      fftw_execute(p0);

      for (k1=0; k1<Ngrid1; k1++){

        ReRhok[BN_CB+k1] = out0[k1][0];
        ImRhok[BN_CB+k1] = out0[k1][1];

      }

    /* end of 1-D FFT */

	for (k1=0; k1<Ngrid1; k1++){

	  if (k1<Ngrid1/2) sk1 = (double)k1;
	  else             sk1 = (double)(k1 - Ngrid1);

	  Gz  = sk1*rtv[1][ESM_direction]; /* modified by AdvanceSoft */

        ReRhok_tmp = ReRhok[BN_CB+k1];
        ImRhok_tmp = ImRhok[BN_CB+k1];

        sin_tmp = sin(Gz*z0);
        cos_tmp = cos(Gz*z0);

	  tmp1r = ReRhok_tmp*cos_tmp - ImRhok_tmp*sin_tmp;
	  tmp1i = ImRhok_tmp*cos_tmp + ReRhok_tmp*sin_tmp;

	  ReRhok[BN_CB+k1] = tmp1r/(double)Ngrid1;
	  ImRhok[BN_CB+k1] = tmp1i/(double)Ngrid1;
  
	}
  
    }
/*
    if (myid==Host_ID){

      sprintf(fname,"%s%s.ESM.dhart",filepath,filename);

      if ( (fp=fopen(fname,"w"))!=NULL ) {

        fprintf(fp,"## check of difference Hartree potential (Hartree) ##\n");fflush(stdout);
        fprintf(fp,"## Grid : x-corrdinate (Ang) : delta-V_H(G_||=0,z) ##\n");fflush(stdout);

        One_dim_FFT(fp,1,1,0,ReRhok,ImRhok,1.0); 

        fprintf(fp,"\n");
        fclose(fp);
      }
      else {
        printf("Failure in saving *.ESM.dhart.\n");
      }
    }*/

  /****************************************************
        find the Hartree potential in real space
  ****************************************************/
  
  Get_Value_inReal(0,dVHart_Grid_B,dVHart_Grid_B,ReRhok,ImRhok); 


  fftw_destroy_plan(p0);
  fftw_cleanup();


  /****************************************************
    freeing of arrays:

    fftw_complex  in[List_YOUSO[17]];
    fftw_complex out[List_YOUSO[17]];
  ****************************************************/

  fftw_free(in0);
  fftw_free(out0);



  /* for time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;

}



void One_dim_FFT(FILE *fp, 
                 int sgn1, int sgn2, int k23,
                 double *ReF2, double *ImF2,
                 double prefac)
{
  int k1,n1;
  double xcord,tv11 = tv[1][ESM_direction]; /* modified by AdvanceSoft */

  /**************************************************************

      One_dim_FFT(fp,sgn1,sgn2,k23,ReF3,ImF3,prefac)

       sgn1= 1: for check of output (ReV & ImV are not changed)
           = 2: for calculation (ReV & ImV are changed)
       sgn2= 1: FFT
           =-1: inverse FFT
       k23    : Gx + Gy
       prefac: prefactor of output data

  ***************************************************************/

  fftw_complex *in, *out;
  fftw_plan p;

  /****************************************************
    allocation of arrays:

    fftw_complex  in[List_YOUSO[17]];
    fftw_complex out[List_YOUSO[17]];
  ****************************************************/

  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]);
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]);

  p = fftw_plan_dft_1d(Ngrid1, in, out, sgn2, FFTW_ESTIMATE);


  for (k1=0; k1<Ngrid1; k1++){

    in[k1][0] = ReF2[k23+k1];
    in[k1][1] = ImF2[k23+k1];

  }

  fftw_execute(p);

  for (n1=0; n1<Ngrid1; n1++){

    if (sgn1 == 1){

      xcord = tv11 / Ngrid1 * n1 * 0.529177249;

      fprintf(fp,"   %4d  %12.9f  %12.9f \n",n1,xcord,out[n1][0]*prefac);

    }
    else if (sgn1 == 2){

      ReF2[k23+n1] = out[n1][0];
      ImF2[k23+n1] = out[n1][1];

    }

  }

  fftw_destroy_plan(p);
  fftw_cleanup();

  /****************************************************
    freeing of arrays:

    fftw_complex  in[List_YOUSO[17]];
    fftw_complex out[List_YOUSO[17]];
  ****************************************************/

  fftw_free(in);
  fftw_free(out);

}



