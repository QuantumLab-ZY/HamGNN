#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>

/* input argument "SCF_iter" was added by S.Ryee for LDA+U. 
 * This was added so that spin-density XC energy is ignored
 * for SCF_iter>=2 when cFLL or cAMF LDA+U scheme is used. */
void Set_XC_Grid(int SCF_iter, int XC_P_switch, int XC_switch, 
                 double *Den0, double *Den1, 
                 double *Den2, double *Den3,
                 double *Vxc0, double *Vxc1,
                 double *Vxc2, double *Vxc3,
                 double ***dEXC_dGD, 
                 double ***dDen_Grid)
{
  /***********************************************************************************************
   XC_P_switch:
      0  \epsilon_XC (XC energy density)  
      1  \mu_XC      (XC potential)  
      2  \epsilon_XC - \mu_XC
      3  dfxc/d|\nabra n| *|d|\nabra n|/d\nabra n and nabla n

   Switch 3 is implemetend by yshiihara to calculate the
   terms for GGA stress. In this case, the input and output parameters are:
            
   INPUT     Den0 : density of up spin     n_up  
   INPUT     Den1 : density of down spin   n_down
   INPUT     Den2 : theta
   INPUT     Den3 : phi
   OUTPUT    Vxc0 : NULL
   OUTPUT    Vxc1 : NULL
   OUTPUT    Vxc2 : NULL
   OUTPUT    Vxc3 : NULL
   OUTPUT    Axc[2(spin)][3(x,y,z)][My_NumGridD] : dfxc/d|\nabra n| *|d|\nabra n|/d\nabra n
   OUTPUT    dDen_Grid[2(spin)][3(x,y,z)][My_NumGridD] : dn/dx, dn/dy, dn/dz 
  ***********************************************************************************************/

  static int firsttime=1;
  int MN,MN1,MN2,i,j,k,ri,ri1,ri2;
  int i1,i2,j1,j2,k1,k2,n,nmax;
  int Ng1,Ng2,Ng3;
  int dDen_Grid_NULL_flag;
  int dEXC_dGD_NULL_flag;
  double den_min=1.0e-14; 
  double Ec_unif[1],Vc_unif[2],Exc[2],Vxc[2];
  double Ex_unif[1],Vx_unif[2],tot_den;
  double ED[2],GDENS[3][2];
  double DEXDD[2],DECDD[2];
  double DEXDGD[3][2],DECDGD[3][2];
  double up_x_a,up_x_b,up_x_c;
  double up_y_a,up_y_b,up_y_c;
  double up_z_a,up_z_b,up_z_c;
  double dn_x_a,dn_x_b,dn_x_c;
  double dn_y_a,dn_y_b,dn_y_c;
  double dn_z_a,dn_z_b,dn_z_c;
  double up_a,up_b,up_c;
  double dn_a,dn_b,dn_c;

  double tmp0,tmp1;
  double cot,sit,sip,cop,phi,theta;
  double detA,igtv[4][4];
  int numprocs,myid;

  /* for OpenMP */
  int OMPID,Nthrds;
  
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
                 allocation of arrays
  ****************************************************/

  dDen_Grid_NULL_flag = 0;

  if (XC_switch==4 && dDen_Grid==NULL){

    dDen_Grid_NULL_flag = 1;

    dDen_Grid = (double***)malloc(sizeof(double**)*2); 
    for (k=0; k<=1; k++){
      dDen_Grid[k] = (double**)malloc(sizeof(double*)*3); 
      for (i=0; i<3; i++){
        dDen_Grid[k][i] = (double*)malloc(sizeof(double)*My_NumGridD); 
        for (j=0; j<My_NumGridD; j++) dDen_Grid[k][i][j] = 0.0;
      }
    }
  }

  dEXC_dGD_NULL_flag = 0;

  if (XC_switch==4 && dEXC_dGD==NULL){

    dEXC_dGD_NULL_flag = 1;

    dEXC_dGD = (double***)malloc(sizeof(double**)*2); 
    for (k=0; k<=1; k++){
      dEXC_dGD[k] = (double**)malloc(sizeof(double*)*3); 
      for (i=0; i<3; i++){
	dEXC_dGD[k][i] = (double*)malloc(sizeof(double)*My_NumGridD); 
	for (j=0; j<My_NumGridD; j++) dEXC_dGD[k][i][j] = 0.0;
      }
    }
  }

  /****************************************************
     calculate dDen_Grid
  ****************************************************/

  if (XC_switch==4){
 
    detA =   gtv[1][1]*gtv[2][2]*gtv[3][3]
           + gtv[1][2]*gtv[2][3]*gtv[3][1]
           + gtv[1][3]*gtv[2][1]*gtv[3][2]
           - gtv[1][3]*gtv[2][2]*gtv[3][1]
           - gtv[1][2]*gtv[2][1]*gtv[3][3]
           - gtv[1][1]*gtv[2][3]*gtv[3][2];     

    igtv[1][1] =  (gtv[2][2]*gtv[3][3] - gtv[2][3]*gtv[3][2])/detA;
    igtv[2][1] = -(gtv[2][1]*gtv[3][3] - gtv[2][3]*gtv[3][1])/detA;
    igtv[3][1] =  (gtv[2][1]*gtv[3][2] - gtv[2][2]*gtv[3][1])/detA; 

    igtv[1][2] = -(gtv[1][2]*gtv[3][3] - gtv[1][3]*gtv[3][2])/detA;
    igtv[2][2] =  (gtv[1][1]*gtv[3][3] - gtv[1][3]*gtv[3][1])/detA;
    igtv[3][2] = -(gtv[1][1]*gtv[3][2] - gtv[1][2]*gtv[3][1])/detA; 

    igtv[1][3] =  (gtv[1][2]*gtv[2][3] - gtv[1][3]*gtv[2][2])/detA;
    igtv[2][3] = -(gtv[1][1]*gtv[2][3] - gtv[1][3]*gtv[2][1])/detA;
    igtv[3][3] =  (gtv[1][1]*gtv[2][2] - gtv[1][2]*gtv[2][1])/detA; 

#pragma omp parallel shared(My_NumGridD,Min_Grid_Index_D,Max_Grid_Index_D,igtv,dDen_Grid,PCCDensity_Grid_D,PCC_switch,Den0,Den1,Den2,Den3,den_min) private(OMPID,Nthrds,nmax,n,i,j,k,ri,ri1,ri2,i1,i2,j1,j2,k1,k2,MN,MN1,MN2,up_a,dn_a,up_b,dn_b,up_c,dn_c,Ng1,Ng2,Ng3)
    {

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();

      Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
      Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
      Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

      for (MN=OMPID; MN<My_NumGridD; MN+=Nthrds){

        i = MN/(Ng2*Ng3);
	j = (MN-i*Ng2*Ng3)/Ng3;
	k = MN - i*Ng2*Ng3 - j*Ng3; 

        if ( i==0 || i==(Ng1-1) || j==0 || j==(Ng2-1) || k==0 || k==(Ng3-1) ){

	  dDen_Grid[0][0][MN] = 0.0;
	  dDen_Grid[0][1][MN] = 0.0;
	  dDen_Grid[0][2][MN] = 0.0;
	  dDen_Grid[1][0][MN] = 0.0;
	  dDen_Grid[1][1][MN] = 0.0;
	  dDen_Grid[1][2][MN] = 0.0;
        }

        else {

          /* set i1, i2, j1, j2, k1, and k2 */ 

          i1 = i - 1;
          i2 = i + 1;

          j1 = j - 1;
          j2 = j + 1;

          k1 = k - 1;
          k2 = k + 1;

	  /* set dDen_Grid */

	  if ( den_min<(Den0[MN]+Den1[MN]) ){

	    /* a-axis */

	    MN1 = i1*Ng2*Ng3 + j*Ng3 + k;
	    MN2 = i2*Ng2*Ng3 + j*Ng3 + k;

	    if (PCC_switch==0) {
	      up_a = Den0[MN2] - Den0[MN1];
	      dn_a = Den1[MN2] - Den1[MN1];
	    }
	    else if (PCC_switch==1) {
	      up_a = Den0[MN2] + PCCDensity_Grid_D[0][MN2]
	           - Den0[MN1] - PCCDensity_Grid_D[0][MN1];
	      dn_a = Den1[MN2] + PCCDensity_Grid_D[1][MN2]
	           - Den1[MN1] - PCCDensity_Grid_D[1][MN1];
	    }

	    /* b-axis */

	    MN1 = i*Ng2*Ng3 + j1*Ng3 + k; 
	    MN2 = i*Ng2*Ng3 + j2*Ng3 + k; 

	    if (PCC_switch==0) {
	      up_b = Den0[MN2] - Den0[MN1];
	      dn_b = Den1[MN2] - Den1[MN1];
	    }
	    else if (PCC_switch==1) {
	      up_b = Den0[MN2] + PCCDensity_Grid_D[0][MN2]
	           - Den0[MN1] - PCCDensity_Grid_D[0][MN1];
	      dn_b = Den1[MN2] + PCCDensity_Grid_D[1][MN2]
	           - Den1[MN1] - PCCDensity_Grid_D[1][MN1];
	    }

	    /* c-axis */

	    MN1 = i*Ng2*Ng3 + j*Ng3 + k1; 
	    MN2 = i*Ng2*Ng3 + j*Ng3 + k2; 

	    if (PCC_switch==0) {
	      up_c = Den0[MN2] - Den0[MN1];
	      dn_c = Den1[MN2] - Den1[MN1];
	    }
	    else if (PCC_switch==1) {
	      up_c = Den0[MN2] + PCCDensity_Grid_D[0][MN2]
	           - Den0[MN1] - PCCDensity_Grid_D[0][MN1];
	      dn_c = Den1[MN2] + PCCDensity_Grid_D[1][MN2]
	           - Den1[MN1] - PCCDensity_Grid_D[1][MN1];
	    }

	    /* up */

	    dDen_Grid[0][0][MN] = 0.5*(igtv[1][1]*up_a + igtv[1][2]*up_b + igtv[1][3]*up_c);
	    dDen_Grid[0][1][MN] = 0.5*(igtv[2][1]*up_a + igtv[2][2]*up_b + igtv[2][3]*up_c);
	    dDen_Grid[0][2][MN] = 0.5*(igtv[3][1]*up_a + igtv[3][2]*up_b + igtv[3][3]*up_c);

	    /* down */

	    dDen_Grid[1][0][MN] = 0.5*(igtv[1][1]*dn_a + igtv[1][2]*dn_b + igtv[1][3]*dn_c);
	    dDen_Grid[1][1][MN] = 0.5*(igtv[2][1]*dn_a + igtv[2][2]*dn_b + igtv[2][3]*dn_c);
	    dDen_Grid[1][2][MN] = 0.5*(igtv[3][1]*dn_a + igtv[3][2]*dn_b + igtv[3][3]*dn_c);
	  }

	  else{
	    dDen_Grid[0][0][MN] = 0.0;
	    dDen_Grid[0][1][MN] = 0.0;
	    dDen_Grid[0][2][MN] = 0.0;
	    dDen_Grid[1][0][MN] = 0.0;
	    dDen_Grid[1][1][MN] = 0.0;
	    dDen_Grid[1][2][MN] = 0.0;
	  }

	} /* else */

      } /* n */

#pragma omp flush(dDen_Grid)

    } /* #pragma omp parallel */
  } /* if (XC_switch==4) */ 

  /****************************************************
   loop MN
  ****************************************************/

#pragma omp parallel shared(dDen_Grid,dEXC_dGD,den_min,Vxc0,Vxc1,Vxc2,Vxc3,My_NumGridD,XC_P_switch,XC_switch,Den0,Den1,Den2,Den3,PCC_switch,PCCDensity_Grid_D) private(OMPID,Nthrds,MN,tot_den,tmp0,tmp1,ED,Exc,Ec_unif,Vc_unif,Vxc,Ex_unif,Vx_unif,GDENS,DEXDD,DECDD,DEXDGD,DECDGD)
  {

    OMPID = omp_get_thread_num();
    Nthrds = omp_get_num_threads();

    for (MN=OMPID; MN<My_NumGridD; MN+=Nthrds){

      switch(XC_switch){
        
	/******************************************************************
         LDA (Ceperly-Alder)

         constructed by Ceperly and Alder,
         ref.
         D. M. Ceperley, Phys. Rev. B18, 3126 (1978)
         D. M. Ceperley and B. J. Alder, Phys. Rev. Lett., 45, 566 (1980) 

         and parametrized by Perdew and Zunger.
         ref.
         J. Perdew and A. Zunger, Phys. Rev. B23, 5048 (1981)
	******************************************************************/
        
      case 1:
        
	tot_den = Den0[MN] + Den1[MN];

	/* partial core correction */
	if (PCC_switch==1) {
	  tot_den += PCCDensity_Grid_D[0][MN] + PCCDensity_Grid_D[1][MN];
	}

	tmp0 = XC_Ceperly_Alder(tot_den,XC_P_switch);

	Vxc0[MN] = tmp0;
	Vxc1[MN] = tmp0;
        
	break;

	/******************************************************************
         LSDA-CA (Ceperly-Alder)

         constructed by Ceperly and Alder,
         ref.
         D. M. Ceperley, Phys. Rev. B18, 3126 (1978)
         D. M. Ceperley and B. J. Alder, Phys. Rev. Lett., 45, 566 (1980) 

         and parametrized by Perdew and Zunger.
         ref.
         J. Perdew and A. Zunger, Phys. Rev. B23, 5048 (1981)
	******************************************************************/

      case 2:

	ED[0] = Den0[MN];
	ED[1] = Den1[MN];

	/* partial core correction */
	if (PCC_switch==1) {
	  ED[0] += PCCDensity_Grid_D[0][MN];
	  ED[1] += PCCDensity_Grid_D[1][MN];
	}

	XC_CA_LSDA(SCF_iter, ED[0], ED[1], Exc, XC_P_switch);
	Vxc0[MN] = Exc[0];
	Vxc1[MN] = Exc[1];

	break;

	/******************************************************************
         LSDA-PW (PW92)

         ref.
         J.P.Perdew and Yue Wang, Phys. Rev. B45, 13244 (1992) 
	******************************************************************/

      case 3:

	ED[0] = Den0[MN];
	ED[1] = Den1[MN];

	/* partial core correction */
	if (PCC_switch==1) {
	  ED[0] += PCCDensity_Grid_D[0][MN];
	  ED[1] += PCCDensity_Grid_D[1][MN];
	}

	if ((ED[0]+ED[1])<den_min){
	  Vxc0[MN] = 0.0;
	  Vxc1[MN] = 0.0;
	}
	else{

	  if (XC_P_switch==0){

	    XC_PW92C(SCF_iter,ED,Ec_unif,Vc_unif);

	    Vxc[0] = Vc_unif[0];
	    Vxc[1] = Vc_unif[1];
	    Exc[0] = Ec_unif[0];

            if((dc_Type==3 || dc_Type==4) && SCF_iter>1){  /* for LDA+U with cFLL by S.Ryee */
  	      XC_EX(1,ED[0]+ED[1],ED,Ex_unif,Vx_unif);
	      Vxc[0] = Vxc[0] + Vx_unif[0];
	      Exc[1] = 2.0*((ED[0]+ED[1])/2.0)*Ex_unif[0];

	      XC_EX(1,ED[0]+ED[1],ED,Ex_unif,Vx_unif);
	      Vxc[1] += Vx_unif[0];
	      Exc[1] += 2.0*((ED[0]+ED[1])/2.0)*Ex_unif[0];
            }
            else{
  	      XC_EX(1,2.0*ED[0],ED,Ex_unif,Vx_unif);
	      Vxc[0] = Vxc[0] + Vx_unif[0];
	      Exc[1] = 2.0*ED[0]*Ex_unif[0];

	      XC_EX(1,2.0*ED[1],ED,Ex_unif,Vx_unif);
	      Vxc[1] += Vx_unif[0];
	      Exc[1] += 2.0*ED[1]*Ex_unif[0];
            }

	    Exc[1] = 0.5*Exc[1]/(ED[0]+ED[1]);

	    Vxc0[MN] = Exc[0] + Exc[1];
	    Vxc1[MN] = Exc[0] + Exc[1];
	  }

	  else if (XC_P_switch==1){
	    XC_PW92C(SCF_iter,ED,Ec_unif,Vc_unif);
	    Vxc0[MN] = Vc_unif[0];
	    Vxc1[MN] = Vc_unif[1];
           
            if((dc_Type==3 || dc_Type==4) && SCF_iter>1){  /* for LDA+U with cFLL by S.Ryee */
	      XC_EX(1,ED[0]+ED[1],ED,Ex_unif,Vx_unif);
	      Vxc0[MN] = Vxc0[MN] + Vx_unif[0];

	      XC_EX(1,ED[0]+ED[1],ED,Ex_unif,Vx_unif);
	      Vxc1[MN] = Vxc1[MN] + Vx_unif[0];
            }
            else{
	      XC_EX(1,2.0*ED[0],ED,Ex_unif,Vx_unif);
	      Vxc0[MN] = Vxc0[MN] + Vx_unif[0];

	      XC_EX(1,2.0*ED[1],ED,Ex_unif,Vx_unif);
	      Vxc1[MN] = Vxc1[MN] + Vx_unif[0];
            }

	  }

	  else if (XC_P_switch==2){

	    XC_PW92C(SCF_iter,ED,Ec_unif,Vc_unif);

	    Vxc[0] = Vc_unif[0];
	    Vxc[1] = Vc_unif[1];
	    Exc[0] = Ec_unif[0];

            if((dc_Type==3 || dc_Type==4) && SCF_iter>1){  /* for LDA+U with cFLL by S.Ryee */
	      XC_EX(1,ED[0]+ED[1],ED,Ex_unif,Vx_unif);
	      Vxc[0]  = Vxc[0] + Vx_unif[0];
	      Exc[1]  = 2.0*((ED[0]+ED[1])/2.0)*Ex_unif[0];

	      XC_EX(1,ED[0]+ED[1],ED,Ex_unif,Vx_unif);
	      Vxc[1] += Vx_unif[0];
	      Exc[1] += 2.0*((ED[0]+ED[1])/2.0)*Ex_unif[0];
            }
            else{
	      XC_EX(1,2.0*ED[0],ED,Ex_unif,Vx_unif);
	      Vxc[0]  = Vxc[0] + Vx_unif[0];
	      Exc[1]  = 2.0*ED[0]*Ex_unif[0];

	      XC_EX(1,2.0*ED[1],ED,Ex_unif,Vx_unif);
	      Vxc[1] += Vx_unif[0];
	      Exc[1] += 2.0*ED[1]*Ex_unif[0];
            }

	    Exc[1] = 0.5*Exc[1]/(ED[0]+ED[1]);

	    Vxc0[MN] = Exc[0] + Exc[1] - Vxc[0];
	    Vxc1[MN] = Exc[0] + Exc[1] - Vxc[1];
	  }
	}

	break;

	/******************************************************************
         GGA-PBE
         ref.
         J. P. Perdew, K. Burke, and M. Ernzerhof,
         Phys. Rev. Lett. 77, 3865 (1996).
	******************************************************************/

      case 4:

	/****************************************************
         ED[0]       density of up spin:     n_up   
         ED[1]       density of down spin:   n_down

         GDENS[0][0] derivative (x) of density of up spin
         GDENS[1][0] derivative (y) of density of up spin
         GDENS[2][0] derivative (z) of density of up spin
         GDENS[0][1] derivative (x) of density of down spin
         GDENS[1][1] derivative (y) of density of down spin
         GDENS[2][1] derivative (z) of density of down spin

         DEXDD[0]    d(fx)/d(n_up) 
         DEXDD[1]    d(fx)/d(n_down) 
         DECDD[0]    d(fc)/d(n_up) 
         DECDD[1]    d(fc)/d(n_down) 

         n'_up_x   = d(n_up)/d(x)
         n'_up_y   = d(n_up)/d(y)
         n'_up_z   = d(n_up)/d(z)
         n'_down_x = d(n_down)/d(x)
         n'_down_y = d(n_down)/d(y)
         n'_down_z = d(n_down)/d(z)
       
         DEXDGD[0][0] d(fx)/d(n'_up_x) 
         DEXDGD[1][0] d(fx)/d(n'_up_y) 
         DEXDGD[2][0] d(fx)/d(n'_up_z) 
         DEXDGD[0][1] d(fx)/d(n'_down_x) 
         DEXDGD[1][1] d(fx)/d(n'_down_y) 
         DEXDGD[2][1] d(fx)/d(n'_down_z) 

         DECDGD[0][0] d(fc)/d(n'_up_x) 
         DECDGD[1][0] d(fc)/d(n'_up_y) 
         DECDGD[2][0] d(fc)/d(n'_up_z) 
         DECDGD[0][1] d(fc)/d(n'_down_x) 
         DECDGD[1][1] d(fc)/d(n'_down_y) 
         DECDGD[2][1] d(fc)/d(n'_down_z) 
	****************************************************/

	ED[0] = Den0[MN];
	ED[1] = Den1[MN];

	if ((ED[0]+ED[1])<den_min){

          if (XC_P_switch!=3){
  	    Vxc0[MN] = 0.0;
	    Vxc1[MN] = 0.0;
	  }

	  /* later add its derivatives */
	  if (XC_P_switch!=0){
	    dEXC_dGD[0][0][MN] = 0.0;
	    dEXC_dGD[0][1][MN] = 0.0;
	    dEXC_dGD[0][2][MN] = 0.0;

	    dEXC_dGD[1][0][MN] = 0.0;
	    dEXC_dGD[1][1][MN] = 0.0;
	    dEXC_dGD[1][2][MN] = 0.0;
	  }
	}
     
	else{

	  GDENS[0][0] = dDen_Grid[0][0][MN];
	  GDENS[1][0] = dDen_Grid[0][1][MN];
	  GDENS[2][0] = dDen_Grid[0][2][MN];
	  GDENS[0][1] = dDen_Grid[1][0][MN];
	  GDENS[1][1] = dDen_Grid[1][1][MN];
	  GDENS[2][1] = dDen_Grid[1][2][MN];

	  if (PCC_switch==1) {
	    ED[0] += PCCDensity_Grid_D[0][MN];
	    ED[1] += PCCDensity_Grid_D[1][MN];
	  }

	  XC_PBE(SCF_iter, ED, GDENS, Exc, DEXDD, DECDD, DEXDGD, DECDGD);

	  /* XC energy density */
	  if      (XC_P_switch==0){
	    Vxc0[MN] = Exc[0] + Exc[1];
	    Vxc1[MN] = Exc[0] + Exc[1];
	  }

	  /* XC potential */
	  else if (XC_P_switch==1){
	    Vxc0[MN] = DEXDD[0] + DECDD[0];
	    Vxc1[MN] = DEXDD[1] + DECDD[1];
	  }

	  /* XC energy density - XC potential */
	  else if (XC_P_switch==2){
	    Vxc0[MN] = Exc[0] + Exc[1] - DEXDD[0] - DECDD[0];
	    Vxc1[MN] = Exc[0] + Exc[1] - DEXDD[1] - DECDD[1];
	  }

	  /* later add its derivatives */
	  if (XC_P_switch!=0){
	    dEXC_dGD[0][0][MN] = DEXDGD[0][0] + DECDGD[0][0];
	    dEXC_dGD[0][1][MN] = DEXDGD[1][0] + DECDGD[1][0];
	    dEXC_dGD[0][2][MN] = DEXDGD[2][0] + DECDGD[2][0];

	    dEXC_dGD[1][0][MN] = DEXDGD[0][1] + DECDGD[0][1];
	    dEXC_dGD[1][1][MN] = DEXDGD[1][1] + DECDGD[1][1];
	    dEXC_dGD[1][2][MN] = DEXDGD[2][1] + DECDGD[2][1];
	  }

	}
	
	break;
	
      } /* switch(XC_switch) */
    }   /* MN */

    if (XC_switch==4){
#pragma omp flush(dEXC_dGD)
    }

  } /* #pragma omp parallel */

  /****************************************************
        calculate the second part of XC potential
               when GGA and XC_P_switch!=0
  ****************************************************/

  if (XC_switch==4 && (XC_P_switch==1 || XC_P_switch==2)){
    
#pragma omp parallel shared(Min_Grid_Index_D,Max_Grid_Index_D,My_NumGridD,XC_P_switch,Vxc0,Vxc1,Vxc2,Vxc3,igtv,dEXC_dGD,Den0,Den1,Den2,Den3,den_min) private(OMPID,Nthrds,nmax,i,j,k,ri,ri1,ri2,i1,i2,j1,j2,k1,k2,MN,MN1,MN2,up_x_a,up_y_a,up_z_a,dn_x_a,dn_y_a,dn_z_a,up_x_b,up_y_b,up_z_b,dn_x_b,dn_y_b,dn_z_b,up_x_c,up_y_c,up_z_c,dn_x_c,dn_y_c,dn_z_c,tmp0,tmp1,Ng1,Ng2,Ng3)
    {

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();

      Ng1 = Max_Grid_Index_D[1] - Min_Grid_Index_D[1] + 1;
      Ng2 = Max_Grid_Index_D[2] - Min_Grid_Index_D[2] + 1;
      Ng3 = Max_Grid_Index_D[3] - Min_Grid_Index_D[3] + 1;

      for (MN=OMPID; MN<My_NumGridD; MN+=Nthrds){

        i = MN/(Ng2*Ng3);
	j = (MN-i*Ng2*Ng3)/Ng3;
	k = MN - i*Ng2*Ng3 - j*Ng3; 

        if ( i<=1 || (Ng1-2)<=i || j<=1 || (Ng2-2)<=j || k<=1 || (Ng3-2)<=k ){

	  Vxc0[MN] = 0.0;
	  Vxc1[MN] = 0.0;
	}
 
        else {

          /* set i1, i2, j1, j2, k1, and k2 */ 

          i1 = i - 1;
          i2 = i + 1;

          j1 = j - 1;
          j2 = j + 1;

          k1 = k - 1;
          k2 = k + 1;

	  /* set Vxc_Grid */

	  if ( den_min<(Den0[MN]+Den1[MN]) ){

	    /* a-axis */

	    MN1 = i1*Ng2*Ng3 + j*Ng3 + k;
	    MN2 = i2*Ng2*Ng3 + j*Ng3 + k;

	    up_x_a = dEXC_dGD[0][0][MN2] - dEXC_dGD[0][0][MN1];
	    up_y_a = dEXC_dGD[0][1][MN2] - dEXC_dGD[0][1][MN1];
	    up_z_a = dEXC_dGD[0][2][MN2] - dEXC_dGD[0][2][MN1];

	    dn_x_a = dEXC_dGD[1][0][MN2] - dEXC_dGD[1][0][MN1];
	    dn_y_a = dEXC_dGD[1][1][MN2] - dEXC_dGD[1][1][MN1];
	    dn_z_a = dEXC_dGD[1][2][MN2] - dEXC_dGD[1][2][MN1];

	    /* b-axis */

	    MN1 = i*Ng2*Ng3 + j1*Ng3 + k; 
	    MN2 = i*Ng2*Ng3 + j2*Ng3 + k; 

	    up_x_b = dEXC_dGD[0][0][MN2] - dEXC_dGD[0][0][MN1];
	    up_y_b = dEXC_dGD[0][1][MN2] - dEXC_dGD[0][1][MN1];
	    up_z_b = dEXC_dGD[0][2][MN2] - dEXC_dGD[0][2][MN1];

	    dn_x_b = dEXC_dGD[1][0][MN2] - dEXC_dGD[1][0][MN1];
	    dn_y_b = dEXC_dGD[1][1][MN2] - dEXC_dGD[1][1][MN1];
	    dn_z_b = dEXC_dGD[1][2][MN2] - dEXC_dGD[1][2][MN1];

	    /* c-axis */

	    MN1 = i*Ng2*Ng3 + j*Ng3 + k1; 
	    MN2 = i*Ng2*Ng3 + j*Ng3 + k2; 

	    up_x_c = dEXC_dGD[0][0][MN2] - dEXC_dGD[0][0][MN1];
	    up_y_c = dEXC_dGD[0][1][MN2] - dEXC_dGD[0][1][MN1];
	    up_z_c = dEXC_dGD[0][2][MN2] - dEXC_dGD[0][2][MN1];

	    dn_x_c = dEXC_dGD[1][0][MN2] - dEXC_dGD[1][0][MN1];
	    dn_y_c = dEXC_dGD[1][1][MN2] - dEXC_dGD[1][1][MN1];
	    dn_z_c = dEXC_dGD[1][2][MN2] - dEXC_dGD[1][2][MN1];

	    /* up */

	    tmp0 = igtv[1][1]*up_x_a + igtv[1][2]*up_x_b + igtv[1][3]*up_x_c
	         + igtv[2][1]*up_y_a + igtv[2][2]*up_y_b + igtv[2][3]*up_y_c
	         + igtv[3][1]*up_z_a + igtv[3][2]*up_z_b + igtv[3][3]*up_z_c;
	    tmp0 = 0.5*tmp0;

	    /* down */

	    tmp1 = igtv[1][1]*dn_x_a + igtv[1][2]*dn_x_b + igtv[1][3]*dn_x_c
	         + igtv[2][1]*dn_y_a + igtv[2][2]*dn_y_b + igtv[2][3]*dn_y_c
	         + igtv[3][1]*dn_z_a + igtv[3][2]*dn_z_b + igtv[3][3]*dn_z_c;
	    tmp1 = 0.5*tmp1;

	    /* XC potential */

	    if (XC_P_switch==1){
	      Vxc0[MN] -= tmp0; 
	      Vxc1[MN] -= tmp1;
	    }

	    /* XC energy density - XC potential */

	    else if (XC_P_switch==2){
	      Vxc0[MN] += tmp0; 
	      Vxc1[MN] += tmp1;
	    }

	  }
	}
      }

#pragma omp flush(Vxc0,Vxc1,Vxc2,Vxc3)

    } /* #pragma omp parallel */
  } /* if (XC_switch==4 && XC_P_switch!=0) */

  /****************************************************
            In case of non-collinear spin DFT 
  ****************************************************/

  if (SpinP_switch==3 && (XC_P_switch==1 || XC_P_switch==2)){

#pragma omp parallel shared(Den0,Den1,Den2,Den3,Vxc0,Vxc1,Vxc2,Vxc3,My_NumGridD) private(OMPID,Nthrds,MN,tmp0,tmp1,theta,phi,sit,cot,sip,cop)
    {

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();

      for (MN=OMPID; MN<My_NumGridD; MN+=Nthrds){

	tmp0 = 0.5*(Vxc0[MN] + Vxc1[MN]);
	tmp1 = 0.5*(Vxc0[MN] - Vxc1[MN]);
	theta = Den2[MN];
	phi   = Den3[MN];

	sit = sin(theta);
	cot = cos(theta);
	sip = sin(phi);
	cop = cos(phi);

        /*******************************************************************
           Since Set_XC_Grid is called as  
           
           XC_P_switch = 1;
           Set_XC_Grid( XC_P_switch, 1,
                        ADensity_Grid_D, ADensity_Grid_D,
                        ADensity_Grid_D, ADensity_Grid_D,
                        RefVxc_Grid_D,   RefVxc_Grid_D,
                        RefVxc_Grid_D,   RefVxc_Grid_D );

           XC_P_switch = 0;
           Set_XC_Grid( XC_P_switch, 1,
                        ADensity_Grid_D, ADensity_Grid_D,
                        ADensity_Grid_D, ADensity_Grid_D,
                        RefVxc_Grid_D,   RefVxc_Grid_D,
                        RefVxc_Grid_D,   RefVxc_Grid_D );

           from Force.c and Total_Energy.c, respectively, 

           data have to be stored in order of Vxc3, Vxc2, Vxc1, and Vxc0.
           Note that only Vxc0 will be used for those calling. 
        ********************************************************************/

	Vxc3[MN] = -tmp1*sit*sip;     /* Im Vxc12 */ 
	Vxc2[MN] =  tmp1*sit*cop;     /* Re Vxc12 */
	Vxc1[MN] =  tmp0 - cot*tmp1;  /* Re Vxc22 */
	Vxc0[MN] =  tmp0 + cot*tmp1;  /* Re Vxc11 */
      }

#pragma omp flush(Vxc0,Vxc1,Vxc2,Vxc3)

    } /* #pragma omp parallel */ 
  }

  /****************************************************
                 freeing of arrays
  ****************************************************/

  if (dDen_Grid_NULL_flag==1){
    for (k=0; k<=1; k++){
      for (i=0; i<3; i++){
        free(dDen_Grid[k][i]);
      }
      free(dDen_Grid[k]);
    }
    free(dDen_Grid);
  }

  if (dEXC_dGD_NULL_flag==1){
    for (k=0; k<=1; k++){
      for (i=0; i<3; i++){
	free(dEXC_dGD[k][i]);
      }
      free(dEXC_dGD[k]);
    }
    free(dEXC_dGD);
  }

}
