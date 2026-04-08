/**********************************************************************
  init.c:

     init.c is a subroutine to initialize several parameters at
     the starting point of calculations.

  Log of init.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "openmx_common.h"
#include "mpi.h"


static void InitV();
static void HGF();
static void Matsubara();
static void MM();
static dcomplex G(dcomplex z);
static void Set_Poles_Residues_EGAC();


void init()
{
  int i,p,wsp,wan;
  double kappa,R,R1,R2;
  int myid,numprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
         rediagonalize_flag_overlap_matrix
         rediagonalize_flag_overlap_matrix_ELPA1
  ****************************************************/

  rediagonalize_flag_overlap_matrix = 0;
  rediagonalize_flag_overlap_matrix_ELPA1 = 0;

  /****************************************************
                 Correct_Position_flag
  ****************************************************/

  Correct_Position_flag = 0;

  /****************************************************
                      force flag
  ****************************************************/

  F_Kin_flag    = 1;
  F_NL_flag     = 1;
  F_CH_flag     = 1;
  F_VNA_flag    = 1;
  F_VEF_flag    = 1;
  F_dVHart_flag = 1;
  F_Vxc_flag    = 1;
  F_U_flag      = 1;
  F_dftD_flag   = 1; /* okuno */

  /****************************************************
     Initialization of the pV term in the enthalpy
  ****************************************************/

  UpV = 0.0;

  /****************************************************
                Setting of atomic weight
  ****************************************************/

  for (i=1; i<=atomnum; i++){
    wsp = WhatSpecies[i];

    if (0.0<Spe_AtomicMass[wsp]){
     Gxyz[i][20] = Spe_AtomicMass[wsp];
    }
    else{
      wan = Spe_WhatAtom[wsp];
      Gxyz[i][20] = Atom_Weight[wan];
    }
  }

  InitV();

  /* set Beta */

  Beta = 1.0/kB/E_Temp;

  /****************************************************
   (1+(1+x/n)^n)^{-1}
   Complex poles for the modified Matsubara summation
  ****************************************************/

  /*
  dM = POLES*1.000;
  dp = -1.0;
  dum1 = 0.5*PI/dM;
  Beta = 1.0/kB/E_Temp;

  for (p=0; p<POLES; p++){
    dp = dp + 1.0;
    dum = (2.0*dp+1.0)*dum1;
    zp[p] = Complex(cos(dum),sin(dum));
    Ep[p] = Complex(2.0*dM/Beta*(zp[p].r-1.0),2.0*dM/Beta*zp[p].i);
  }
  */


  /****************************************************
  CONTINUED FRACTION
  zero points and residues for the continued fraction
  expansion of exponential function used in integration
  of Green's functions with Fermi-Dirac distribution
  ****************************************************/

  if (Solver==1) {

    zero_cfrac( POLES, zp, Rp ); 

    for (p=0; p<POLES; p++){
      Ep[p].r = 0.0;
      Ep[p].i = zp[p].i/Beta;
    }
  }

  /**********************************************************
                 O(N^2) cluster diagonalization 
  **********************************************************/

  if (Solver==9) {

    ON2_Npoles_f = ON2_Npoles; 

    /************************************
      for calculation of density matrix
    ************************************/

    /* set up poles */

    zero_cfrac( ON2_Npoles, ON2_zp, ON2_Rp ); 

    for (p=0; p<ON2_Npoles; p++){
      ON2_method[p] = 1; 
    }

    /* for the zero-th moment */

    R = 1.0e+10;

    ON2_zp[ON2_Npoles].r = 0.0;
    ON2_zp[ON2_Npoles].i = R;

    ON2_Rp[ON2_Npoles].r = 0.0;
    ON2_Rp[ON2_Npoles].i = 0.5*R;

    ON2_method[ON2_Npoles] = 2; 

    ON2_Npoles++;

    /*********************************************
      for calculation of energy density matrix
    *********************************************/

    /* calculation of kappa */

    kappa = 0.0; 
    for (p=0; p<ON2_Npoles_f; p++){
      kappa += ON2_Rp[p].r; 
    }
    kappa = 4.0*kappa/Beta;

    /* set up poles */

    for (p=0; p<ON2_Npoles_f; p++){
      ON2_zp_f[p] = ON2_zp[p];
      ON2_Rp_f[p] = ON2_Rp[p];
      ON2_method_f[p] = 1; 
    }

    R = 1.0e+7;

    ON2_zp_f[ON2_Npoles_f].r = 0.0;
    ON2_zp_f[ON2_Npoles_f].i = R*R;

    ON2_Rp_f[ON2_Npoles_f].r = 0.5*R*R*R*R/(R-1.0);
    ON2_Rp_f[ON2_Npoles_f].i = 0.5*R*R*R*kappa/(R-1.0);

    ON2_method_f[ON2_Npoles_f] = 2; 
    ON2_Npoles_f++;

    /* for the 1st order moment */

    ON2_zp_f[ON2_Npoles_f].r = 0.0;
    ON2_zp_f[ON2_Npoles_f].i = R;

    ON2_Rp_f[ON2_Npoles_f].r =-0.5*R*R*R/(R-1.0);
    ON2_Rp_f[ON2_Npoles_f].i =-0.5*R*kappa/(R-1.0);

    ON2_method_f[ON2_Npoles_f] = 2; 
    ON2_Npoles_f++;

  }

  /****************************************************
      for the EGAC method
  ****************************************************/

  if (Solver==10) {
    Set_Poles_Residues_EGAC();
  }

}



void Set_Poles_Residues_EGAC()
{
  int p,i,j,k,m,n,pole;
  int LDA,LDB,LDC,M,N,K;
  double x,xmin,xmax,width,R,kappa;
  dcomplex t,alpha,sum,al,be;
  int myid,numprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /************************************************************************
    defining of poles
  
    near: original poles close to the real axis, which are not 
          explicity calculated.
          The number of poles is EGAC_Npoles_near

    new: poles which are newly created by analytic continuation.
         The number of poles is EGAC_Npoles_new.

  ***************************************************************************/

  if (AC_flag_EGAC==0){

    EGAC_Npoles_near = 0; 
    EGAC_Npoles_new =  0; 
  }
  else {

    EGAC_Npoles_near = 1; /* the number of original poles close to the real axis */
    EGAC_Npoles_new  = 6; /* odd includes the zero real case. even not. */
  }

  /* set up the original poles */

  zero_cfrac( EGAC_Npoles, EGAC_zp, EGAC_Rp ); 

  if (EGAC_Npoles_near!=0 && EGAC_Npoles_new!=0){

    /* shift the original poles and residues */
    for (i=(EGAC_Npoles-1); EGAC_Npoles_near<=i; i--){

      p = i + EGAC_Npoles_new - EGAC_Npoles_near;

      EGAC_zp[p] = EGAC_zp[i];
      EGAC_Rp[p] = EGAC_Rp[i];
      EGAC_method[p] = 1; 
    }

    /* set EGAC_method */

    for (i=0; i<EGAC_Npoles_new; i++){
      EGAC_method[i] = 1; 
    }

    /* update EGAC_Npoles */

    EGAC_Npoles = EGAC_Npoles + EGAC_Npoles_new - EGAC_Npoles_near;

    /* set EGAC_Rp and RGAC_zp */

    if (EGAC_Npoles_near==1 && EGAC_Npoles_new==2){

      EGAC_Rp[0].r = -1.667647058823529e+1; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 9.424777960769380;
      EGAC_Rp[1].r = 1.567647058823529e+1; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.021017612416683e+1;
    }

    if (EGAC_Npoles_near==1 && EGAC_Npoles_new==3){
      EGAC_Rp[0].r = -1.286764705882353e+2; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 9.424777960769380;
      EGAC_Rp[1].r = 2.536764705882353e+2; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.021017612416683e+1;
      EGAC_Rp[2].r = -1.260000000000000e+2; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.099557428756428e+1;
    }

    if (EGAC_Npoles_near==1 && EGAC_Npoles_new==4){
      EGAC_Rp[0].r = -6.917569659442724e+2; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 9.424777960769380;
      EGAC_Rp[1].r = 2.130611455108359e+3; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.021017612416683e+1;
      EGAC_Rp[2].r = -2.201890092879257e+3; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.099557428756428e+1;
      EGAC_Rp[3].r = 7.620356037151703e+2; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 1.178097245096172e+1;
    }

    if (EGAC_Npoles_near==1 && EGAC_Npoles_new==5){
      EGAC_Rp[0].r = -2.958506191950464e+3; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 9.424777960769380;
      EGAC_Rp[1].r = 1.261432662538700e+4; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.021017612416683e+1;
      EGAC_Rp[2].r = -2.027921517027864e+4; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.099557428756428e+1;
      EGAC_Rp[3].r = 1.454125541795666e+4; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 1.178097245096172e+1;
      EGAC_Rp[4].r = -3.918860681114551e+3; EGAC_Rp[4].i = 0; EGAC_zp[4].r = 0; EGAC_zp[4].i = 1.256637061435917e+1;
    }

    if (EGAC_Npoles_near==1 && EGAC_Npoles_new==6){
      EGAC_Rp[0].r = -1.074824303405573e+4; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 9.424777960769380;
      EGAC_Rp[1].r = 5.935274767801858e+4; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.021017612416683e+1;
      EGAC_Rp[2].r = -1.316724520123839e+5; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.099557428756428e+1;
      EGAC_Rp[3].r = 1.464274922600619e+5; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 1.178097245096172e+1;
      EGAC_Rp[4].r = -8.152328173374613e+4; EGAC_Rp[4].i = 0; EGAC_zp[4].r = 0; EGAC_zp[4].i = 1.256637061435917e+1;
      EGAC_Rp[5].r = 1.816273684210526e+4; EGAC_Rp[5].i = 0; EGAC_zp[5].r = 0; EGAC_zp[5].i = 1.335176877775662e+1;
    }

    if (EGAC_Npoles_near==1 && EGAC_Npoles_new==10){
      EGAC_Rp[0].r = -6.716949473684211e+5; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 9.424777960769380;
      EGAC_Rp[1].r = 7.588901299771167e+6; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.021017612416683e+1;
      EGAC_Rp[2].r = -3.812890970800915e+7; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.099557428756428e+1;
      EGAC_Rp[3].r = 1.116370481354691e+8; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 1.178097245096172e+1;
      EGAC_Rp[4].r = -2.096937273885584e+8; EGAC_Rp[4].i = 0; EGAC_zp[4].r = 0; EGAC_zp[4].i = 1.256637061435917e+1;
      EGAC_Rp[5].r = 2.618671615304348e+8; EGAC_Rp[5].i = 0; EGAC_zp[5].r = 0; EGAC_zp[5].i = 1.335176877775662e+1;
      EGAC_Rp[6].r = -2.173199156869565e+8; EGAC_Rp[6].i = 0; EGAC_zp[6].r = 0; EGAC_zp[6].i = 1.413716694115407e+1;
      EGAC_Rp[7].r = 1.155371703652174e+8; EGAC_Rp[7].i = 0; EGAC_zp[7].r = 0; EGAC_zp[7].i = 1.492256510455152e+1;
      EGAC_Rp[8].r = -3.570049758260870e+7; EGAC_Rp[8].i = 0; EGAC_zp[8].r = 0; EGAC_zp[8].i = 1.570796326794897e+1;
      EGAC_Rp[9].r = 4.884462982608696e+6; EGAC_Rp[9].i = 0; EGAC_zp[9].r = 0; EGAC_zp[9].i = 1.649336143134641e+1;
    }

    if (EGAC_Npoles_near==2 && EGAC_Npoles_new==10){
 
      EGAC_Rp[0].r = -1.763345918711892e+6; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 1.570796326794897e+1;
      EGAC_Rp[1].r = 5.991420275120774e+6; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.649336143134641e+1;
      EGAC_Rp[2].r = -1.202829289432434e+7; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.792940473702815e+1;
      EGAC_Rp[3].r = 1.869832007305604e+7; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 1.978901183747596e+1;
      EGAC_Rp[4].r = -2.227096667349770e+7; EGAC_Rp[4].i = 0; EGAC_zp[4].r = 0; EGAC_zp[4].i = 2.199114857512855e+1;
      EGAC_Rp[5].r = 1.997073980800229e+7; EGAC_Rp[5].i = 0; EGAC_zp[5].r = 0; EGAC_zp[5].i = 2.448898168174987e+1;
      EGAC_Rp[6].r = -1.304341956379009e+7; EGAC_Rp[6].i = 0; EGAC_zp[6].r = 0; EGAC_zp[6].i = 2.725091173940574e+1;
      EGAC_Rp[7].r = 5.850019384845002e+6; EGAC_Rp[7].i = 0; EGAC_zp[7].r = 0; EGAC_zp[7].i = 3.025374081156461e+1;
      EGAC_Rp[8].r = -1.608625985250872e+6; EGAC_Rp[8].i = 0; EGAC_zp[8].r = 0; EGAC_zp[8].i = 3.347949502058243e+1;
      EGAC_Rp[9].r = 2.041494945507869e+5; EGAC_Rp[9].i = 0; EGAC_zp[9].r = 0; EGAC_zp[9].i = 3.691371367968007e+1;

      /*
      EGAC_Rp[0].r = -1.756335580203082e+8; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 1.570796326794897e+1;
      EGAC_Rp[1].r = 1.855623295995325e+9; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 1.649336143134641e+1;
      EGAC_Rp[2].r = -8.701099775645099e+9; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 1.727875959474386e+1;
      EGAC_Rp[3].r = 2.376222199424616e+10; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 1.806415775814131e+1;
      EGAC_Rp[4].r = -4.164598641159953e+10; EGAC_Rp[4].i = 0; EGAC_zp[4].r = 0; EGAC_zp[4].i = 1.884955592153876e+1;
      EGAC_Rp[5].r = 4.857231661976266e+10; EGAC_Rp[5].i = 0; EGAC_zp[5].r = 0; EGAC_zp[5].i = 1.963495408493621e+1;
      EGAC_Rp[6].r = -3.769709124271910e+10; EGAC_Rp[6].i = 0; EGAC_zp[6].r = 0; EGAC_zp[6].i = 2.042035224833366e+1;
      EGAC_Rp[7].r = 1.877236005968703e+10; EGAC_Rp[7].i = 0; EGAC_zp[7].r = 0; EGAC_zp[7].i = 2.120575041173110e+1;
      EGAC_Rp[8].r = -5.442698744808659e+9; EGAC_Rp[8].i = 0; EGAC_zp[8].r = 0; EGAC_zp[8].i = 2.199114857512855e+1;
      EGAC_Rp[9].r = 6.999877611015162e+8; EGAC_Rp[9].i = 0; EGAC_zp[9].r = 0; EGAC_zp[9].i = 2.277654673852600e+1;
      */
    }

    if (EGAC_Npoles_near==3 && EGAC_Npoles_new==8){

      EGAC_Rp[0].r = -1.103590519846028e+5; EGAC_Rp[0].i = 0; EGAC_zp[0].r = 0; EGAC_zp[0].i = 2.199114857512855e+1;
      EGAC_Rp[1].r = 3.470701755330101e+5; EGAC_Rp[1].i = 0; EGAC_zp[1].r = 0; EGAC_zp[1].i = 2.356194490192345e+1;
      EGAC_Rp[2].r = -5.995768863834235e+5; EGAC_Rp[2].i = 0; EGAC_zp[2].r = 0; EGAC_zp[2].i = 2.643403151328692e+1;
      EGAC_Rp[3].r = 7.501813113567999e+5; EGAC_Rp[3].i = 0; EGAC_zp[3].r = 0; EGAC_zp[3].i = 3.015324571418253e+1;
      EGAC_Rp[4].r = -6.615835107902539e+5; EGAC_Rp[4].i = 0; EGAC_zp[4].r = 0; EGAC_zp[4].i = 3.455751918948773e+1;
      EGAC_Rp[5].r = 3.899632880594315e+5; EGAC_Rp[5].i = 0; EGAC_zp[5].r = 0; EGAC_zp[5].i = 3.955318540273037e+1;
      EGAC_Rp[6].r = -1.376557903249500e+5; EGAC_Rp[6].i = 0; EGAC_zp[6].r = 0; EGAC_zp[6].i = 4.507704551804211e+1;
      EGAC_Rp[7].r = 2.195746453398864e+4; EGAC_Rp[7].i = 0; EGAC_zp[7].r = 0; EGAC_zp[7].i = 5.108270366235984e+1;
    }

  }
  else{
    for (i=0; i<EGAC_Npoles; i++){
      EGAC_method[i] = 1; 
    }
  }

  /***********************************************
   copy EGAC_zp, EGAC_Rp, and EGAC_method to 
   EGAC_zp_f, EGAC_Rp_f, and EGAC_method_f.
  ***********************************************/

  for (i=0; i<EGAC_Npoles; i++){
    EGAC_zp_f[i] = EGAC_zp[i];
    EGAC_Rp_f[i] = EGAC_Rp[i]; 
    EGAC_method_f[i] = EGAC_method[i];
  }
  EGAC_Npoles_f = EGAC_Npoles; 

  /************************************
        contribution of moments
  ************************************/

  /* for the zero-th moment */

  R = 1.0e+9;

  EGAC_zp[EGAC_Npoles].r = 0.0;
  EGAC_zp[EGAC_Npoles].i = R*R;
  EGAC_Rp[EGAC_Npoles].r = 0.0;
  EGAC_Rp[EGAC_Npoles].i = 0.5*R*R;
  EGAC_method[EGAC_Npoles] = 2; 
  EGAC_Npoles++;



  /*
  EGAC_zp[EGAC_Npoles].r = 0.0;
  EGAC_zp[EGAC_Npoles].i = R;
  EGAC_Rp[EGAC_Npoles].r = 0.0;
  EGAC_Rp[EGAC_Npoles].i = 0.0;
  EGAC_method[EGAC_Npoles] = 2; 
  EGAC_Npoles++;
  */

  /* calculation of kappa */

  /*
  kappa = 0.0; 
  for (p=0; p<EGAC_Npoles_f; p++){
    kappa += EGAC_Rp[p].r; 
  }
  kappa = 4.0*kappa/Beta;
  */

  /* for the zero-th order moment */

  /*
  EGAC_zp_f[EGAC_Npoles_f].r = 0.0;
  EGAC_zp_f[EGAC_Npoles_f].i = R*R;
  EGAC_Rp_f[EGAC_Npoles_f].r = 0.5*R*R*R*R/(R-1.0);
  EGAC_Rp_f[EGAC_Npoles_f].i = 0.5*R*R*R*kappa/(R-1.0);
  EGAC_method_f[EGAC_Npoles_f] = 2; 
  EGAC_Npoles_f++;
  */

  /* for the 1st order moment */

  /*
  EGAC_zp_f[EGAC_Npoles_f].r = 0.0;
  EGAC_zp_f[EGAC_Npoles_f].i = R;
  EGAC_Rp_f[EGAC_Npoles_f].r =-0.5*R*R*R/(R-1.0);
  EGAC_Rp_f[EGAC_Npoles_f].i =-0.5*R*kappa/(R-1.0);
  EGAC_method_f[EGAC_Npoles_f] = 2; 
  EGAC_Npoles_f++;
  */

  /*
  for (i=0; i<EGAC_Npoles; i++){
    printf("ABC0 i=%2d zp=%18.15f %18.15f Rp=%18.15f %18.15f\n",
	   i,EGAC_zp[i].r/Beta,EGAC_zp[i].i/Beta,EGAC_Rp[i].r,EGAC_Rp[i].i); 
  }

  MPI_Finalize();
  exit(0);
  */

  /*
  sum.r = 0.0; sum.i = 0.0;
  x = 0.0;
  for (i=0; i<EGAC_Npoles_new; i++){
    sum.r += EGAC_Rp[i].r;
    sum.i += EGAC_Rp[i].i;

    if (x<fabs(EGAC_Rp[i].r)) x = fabs(EGAC_Rp[i].r);
    if (x<fabs(EGAC_Rp[i].i)) x = fabs(EGAC_Rp[i].i);
  }
  printf("GGG1 sum.r=%18.15f sum.i=%18.15f max=%18.15f\n",sum.r,sum.i,x);
  */

  /* 
     At this moment, 
     note that 
     EGAC_Npoles = EGAC_Npoles_f = EGAC_Npoles - EGAC_Npoles_near + EGAC_Npoles_new + 2. 
  */

}






void MM()
{
  int p;
  double ChemP,N;
  dcomplex EpP,CN;
  dcomplex G0; 

  ChemP = 0.0;

  CN.r = 0.0;
  CN.i = 0.0;

  for (p=0; p<POLES; p++){
    EpP.r = ChemP + Ep[p].r;
    EpP.i = Ep[p].i;

    G0 = G(EpP);
    CN.r += (G0.r*zp[p].r - G0.i*zp[p].i)*2.0/Beta;
    CN.i += (G0.r*zp[p].i + G0.i*zp[p].r)*2.0/Beta;
  }

  N = CN.r;

  printf("CN.r=%18.15f  CN.i=%18.15f\n",CN.r,CN.i);
  printf("N=%18.15f\n",N);

}



void Matsubara()
{
  int p;
  double ChemP,N,R,mu0;
  dcomplex EpP,CN;
  dcomplex G0; 

  ChemP = 0.0;
  R = 1.0e+12;

  CN.r = 0.0;
  CN.i = 0.0;

  for (p=0; p<POLES; p++){
    EpP.r = ChemP + Ep[p].r;
    EpP.i = Ep[p].i;

    G0 = G(EpP);
    CN.r += (G0.r*Rp[p].r - G0.i*Rp[p].i)*2.0/Beta;
    CN.i += (G0.r*Rp[p].i + G0.i*Rp[p].r)*2.0/Beta;
  }

  

  EpP.r = 0.0;
  EpP.i = R;

  mu0 = -R*G(EpP).i;
  N = 0.5*mu0 - CN.r;

  printf("CN.r=%18.15f  CN.i=%18.15f\n",CN.r,CN.i);
  printf("mu0=%18.15f  N=%18.15f\n",mu0,N);
}





void HGF()
{
  int p;
  double ChemP,N,R,mu0;
  dcomplex EpP,CN;
  dcomplex G0; 

  ChemP = 0.0;
  R = 1.0e+12;

  CN.r = 0.0;
  CN.i = 0.0;

  for (p=0; p<POLES; p++){
    EpP.r = ChemP + Ep[p].r;
    EpP.i = Ep[p].i;

    G0 = G(EpP);
    CN.r += (G0.r*Rp[p].r - G0.i*Rp[p].i)*2.0/Beta;
    CN.i += (G0.r*Rp[p].i + G0.i*Rp[p].r)*2.0/Beta;
  }

  

  EpP.r = 0.0;
  EpP.i = R;

  mu0 = -R*G(EpP).i;
  N = 0.5*mu0 - CN.r;

  printf("CN.r=%18.15f  CN.i=%18.15f\n",CN.r,CN.i);
  printf("mu0=%18.15f  N=%18.15f\n",mu0,N);

}



dcomplex G(dcomplex z)
{
  int i;
  dcomplex e[10];
  dcomplex d,sum,dum;
  
  e[1].r =-10.0/eV2Hartree;  e[1].i = 0.0;
  e[2].r = -5.0/eV2Hartree;  e[2].i = 0.0;
  e[3].r = -2.0/eV2Hartree;  e[3].i = 0.0;
  e[4].r =  5.0/eV2Hartree;  e[4].i = 0.0;

  sum.r = 0.0;
  sum.i = 0.0;
 
  for (i=1; i<=4; i++){
    d = Csub(z,e[i]);
    dum = RCdiv(1.0,d);
    sum.r += dum.r;
    sum.i += dum.i;
  }

  return sum; 
}








void InitV()
{
  /******************************************************* 
   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ********************************************************/

  double ax,ay,az,tmp,Wscale,sum,v,sumVx,sumVy,sumVz,scale;
  int j,Gc_AN,myid,numprocs;

  Wscale = unified_atomic_mass_unit/electron_mass;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if ( ( MD_switch==2  || // NVT_VS
         MD_switch==9  || // NVT_NH
         MD_switch==11 || // NVT_VS2 
         MD_switch==14 || // NVT_VS4
         MD_switch==15 || // NVT_Langevin
         MD_switch==27 || // NPT_VS_PR
         MD_switch==28 || // NPT_VS_WV
         MD_switch==29 || // NPT_NH_PR
         MD_switch==30 )  // NPT_NH_WV
       && MD_Init_Velocity==0 ){

    if (myid==Host_ID){

      sumVx = 0.0;
      sumVy = 0.0;
      sumVz = 0.0;

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

        v = sqrt(3.0*kB*TempPara[1][2]/(Gxyz[Gc_AN][20]*Wscale*eV2Hartree));

	ax = rnd(1.0);
	ay = rnd(1.0);
	az = rnd(1.0);

        tmp = 1.0/sqrt(ax*ax+ay*ay+az*az);

        ax *= tmp; 
        ay *= tmp; 
        az *= tmp; 
        
	Gxyz[Gc_AN][24] = v*ax;
	Gxyz[Gc_AN][25] = v*ay;
	Gxyz[Gc_AN][26] = v*az;

        sumVx += Gxyz[Gc_AN][24];
        sumVy += Gxyz[Gc_AN][25];
        sumVz += Gxyz[Gc_AN][26];
      }

      /* correction so that the sum of velocities can be zero.  */

      sumVx /= (double)atomnum; 
      sumVy /= (double)atomnum; 
      sumVz /= (double)atomnum; 

      for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
	Gxyz[Gc_AN][24] -= sumVx;
	Gxyz[Gc_AN][25] -= sumVy;
	Gxyz[Gc_AN][26] -= sumVz;

	Gxyz[Gc_AN][27] = Gxyz[Gc_AN][24];
	Gxyz[Gc_AN][28] = Gxyz[Gc_AN][25];
	Gxyz[Gc_AN][29] = Gxyz[Gc_AN][26];
      }

    } /* end of if (myid==Host_ID) */

    /****************
    MPI:  Gxyz
    *****************/

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      MPI_Bcast(&Gxyz[Gc_AN][24], 6, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    }
  }

  else if (MD_Init_Velocity==0) {

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      Gxyz[Gc_AN][24] = 0.0;
      Gxyz[Gc_AN][25] = 0.0;
      Gxyz[Gc_AN][26] = 0.0;
      Gxyz[Gc_AN][27] = 0.0;
      Gxyz[Gc_AN][28] = 0.0;
      Gxyz[Gc_AN][29] = 0.0;
    }
  }

  /*********************************************************** 
   calculate the initial Ukc:

   1 a.u.=2.4189*10^-2 fs, 1fs=41.341105 a.u. 
  ***********************************************************/

  Ukc = 0.0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    sum = 0.0;
    for (j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
	sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
      }
    }
    Ukc += 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

  /* calculation of temperature (K) and rescale v */
  Temp = Ukc/(1.5*kB*(double)atomnum)*eV2Hartree;
  scale = sqrt(TempPara[1][2]/(Temp+1.0e-13));

  if (MD_Init_Velocity==0){

    for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
      Gxyz[Gc_AN][24] *= scale;
      Gxyz[Gc_AN][25] *= scale;
      Gxyz[Gc_AN][26] *= scale;
    }
  }

  /* recalculate Ukc */
  Ukc = 0.0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    sum = 0.0;
    for (j=1; j<=3; j++){
      if (atom_Fixed_XYZ[Gc_AN][j]==0){
	sum += Gxyz[Gc_AN][j+23]*Gxyz[Gc_AN][j+23];
      }
    }
    Ukc += 0.5*Gxyz[Gc_AN][20]*Wscale*sum;
  }

}




