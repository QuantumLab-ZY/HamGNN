/**********************************************************************
  TRAN_Input_std.c:

  TRAN_Input_std.c is a subroutine to read the input data.

  Log of TRAN_Input_std.c:

     24/July/2008  Released by H.Kino and T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "Inputtools.h"
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"

#define MAXBUF  256




void TRAN_Input_std(
  MPI_Comm comm1, 
  int Solver,          /* input */
  int SpinP_switch,  
  char *filepath,
  double kBvalue,
  double TRAN_eV2Hartree,
  double Electronic_Temperature,
                      /* output */
  int *output_hks
)
{
  FILE *fp;
  int i,po;
  int i_vec[20],i_vec2[20];
  double r_vec[20];
  char *s_vec[20];
  char buf[MAXBUF];
  int myid;
  /* S MitsuakiKAWAMURA*/
  int TRAN_Analysis_Dummy;
  /* E MitsuakiKAWAMURA*/

  MPI_Comm_rank(comm1,&myid);

  /****************************************************
               parameters for TRANSPORT
  ****************************************************/

  input_logical("NEGF.Output_HKS",&TRAN_output_hks,0);
  *output_hks = TRAN_output_hks;

  /* printf("NEGF.OutputHKS=%d\n",TRAN_output_hks); */
  input_string("NEGF.filename.HKS",TRAN_hksoutfilename,"NEGF.hks");
  /* printf("TRAN_hksoutfilename=%s\n",TRAN_hksoutfilename); */

  input_logical("NEGF.Output.for.TranMain",&TRAN_output_TranMain,0);

  if ( Solver!=4 ) { return; }

  /**** show transport credit ****/
  TRAN_Credit(comm1);

  input_string("NEGF.filename.hks.l",TRAN_hksfilename[0],"NEGF.hks.l");
  input_string("NEGF.filename.hks.r",TRAN_hksfilename[1],"NEGF.hks.r");

  /* read data of leads */

  TRAN_RestartFile(comm1, "read","left", filepath,TRAN_hksfilename[0]);
  TRAN_RestartFile(comm1, "read","right",filepath,TRAN_hksfilename[1]);

  /* check b-, and c-axes of the unit cell of leads. */

  po = 0;
  for (i=2; i<=3; i++){
    if (1.0e-10<fabs(tv_e[0][i][1]-tv_e[1][i][1])) po = 1;
    if (1.0e-10<fabs(tv_e[0][i][2]-tv_e[1][i][2])) po = 1;
    if (1.0e-10<fabs(tv_e[0][i][3]-tv_e[1][i][3])) po = 1;
  }

  if (po==1){

    if (myid==Host_ID){
      printf("Warning: The b- or c-axis of the unit cell for the left lead is not same as that for the right lead.\n");
    }

    MPI_Finalize();
    exit(1);
  }

  /* show chemical potentials */

  if (myid==Host_ID){
    printf("\n");
    printf("Intrinsic chemical potential (eV) of the leads\n");
    printf("  Left lead:  %15.12f\n",ChemP_e[0]*TRAN_eV2Hartree);
    printf("  Right lead: %15.12f\n",ChemP_e[1]*TRAN_eV2Hartree);
  }

  /* check the conflict of SpinP_switch */

  if ( (SpinP_switch!=SpinP_switch_e[0]) || (SpinP_switch!=SpinP_switch_e[1]) ){

    if (myid==Host_ID){
      printf ("scf.SpinPolarization conflicts between leads or lead and center.\n");
    }

    MPI_Finalize();
    exit(0);
  }

  input_int(   "NEGF.Surfgreen.iterationmax", &tran_surfgreen_iteration_max, 600);
  input_double("NEGF.Surfgreen.convergeeps", &tran_surfgreen_eps, 1.0e-12); 

  /****  k-points parallel to the layer, which are used for the SCF calc. ****/

  i_vec2[0]=1;
  i_vec2[1]=1;
  input_intv("NEGF.scf.Kgrid",2,i_vec,i_vec2);
  TRAN_Kspace_grid2 = i_vec[0];
  TRAN_Kspace_grid3 = i_vec[1];

  if (TRAN_Kspace_grid2<=0){

    if (myid==Host_ID){
      printf("NEGF.scf.Kgrid should be over 1\n");
    }

    MPI_Finalize();
    exit(1);
  } 
  if (TRAN_Kspace_grid3<=0){

    if (myid==Host_ID){
      printf("NEGF.scf.Kgrid should be over 1\n");
    }

    MPI_Finalize();
    exit(1);
  } 

  /* Poisson solver */

  TRAN_Poisson_flag = 1;

  /* beginning added by mari 07.23.2014 */
  s_vec[0]="FD";  s_vec[1]="FFT";  s_vec[2]="FDG";
  i_vec[0]=1   ;  i_vec[1]=2    ;  i_vec[2]=3    ;

  input_string2int("NEGF.Poisson.Solver", &TRAN_Poisson_flag, 3, s_vec,i_vec);
  /* end added by mari 07.23.2014 */

  /* parameter to scale terms with Gpara=0 */

  input_double("NEGF.Poisson_Gparazero.scaling", &TRAN_Poisson_Gpara_Scaling, 1.0); 

  /* the number of buffer cells in FFTE */

  input_int("NEGF.FFTE.Num.Buffer.Cells", &TRAN_FFTE_CpyNum, 1); 

  /* the number of iterations by the Band calculation in the initial SCF iterations */

  input_int("NEGF.SCF.Iter.Band", &TRAN_SCF_Iter_Band, 3); 

  /* integration method */

  TRAN_integration = 0;

  s_vec[0]="CF"; s_vec[1]="OLD"; 
  i_vec[0]=0    ; i_vec[1]=1    ;
  input_string2int("NEGF.Integration", &TRAN_integration, 2, s_vec,i_vec);

  /* check whether analysis is made or not */

  input_logical("NEGF.tran.analysis",&TRAN_analysis,1);

  /* S MitsuakiKAWAMURA*/
  input_logical("NEGF.tran.channel", &TRAN_Analysis_Dummy, 1);
  if (TRAN_Analysis_Dummy == 1 && TRAN_analysis != 1){
    TRAN_analysis = 1;
    if (myid == Host_ID)
      printf("Since NEGF.tran.channel is on, NEGF.tran.analysis is automatically set to on.\n");
  }
  input_logical("NEGF.tran.CurrentDensity", &TRAN_Analysis_Dummy, 1);
  if (TRAN_Analysis_Dummy == 1 && TRAN_analysis != 1){
    TRAN_analysis = 1;
    if (myid == Host_ID)
      printf("Since NEGF.tran.CurrentDensity is on, NEGF.tran.analysis is automatically set to on.\n");
  }
  input_logical("NEGF.OffDiagonalCurrent", &TRAN_Analysis_Dummy, 0);
  if (TRAN_Analysis_Dummy == 1 && TRAN_analysis != 1) {
    TRAN_analysis = 1;
    if (myid == Host_ID)
      printf("Since NEGF.OffDiagonalCurrent is on, NEGF.tran.analysis is automatically set to on.\n");
  }
  /* E MitsuakiKAWAMURA*/

  /* check whether the SCF calcultion is skipped or not */

  i = 0;
  s_vec[i] = "OFF";         i_vec[i] = 0;  i++;
  s_vec[i] = "ON";          i_vec[i] = 1;  i++;
  s_vec[i] = "Periodic";    i_vec[i] = 2;  i++;

  input_string2int("NEGF.tran.SCF.skip", &TRAN_SCF_skip, i, s_vec, i_vec);

  /****  k-points parallel to the layer, which are used for the transmission calc. ****/
  
  i_vec2[0]=TRAN_Kspace_grid2;
  i_vec2[1]=TRAN_Kspace_grid3;
  input_intv("NEGF.tran.Kgrid",2,i_vec,i_vec2);
  TRAN_TKspace_grid2 = i_vec[0];
  TRAN_TKspace_grid3 = i_vec[1];

  if (TRAN_TKspace_grid2<=0){

    if (myid==Host_ID){
      printf("NEGF.tran.Kgrid should be over 1\n");
    }

    MPI_Finalize();
    exit(1);
  } 
  if (TRAN_TKspace_grid3<=0){

    if (myid==Host_ID){
      printf("NEGF.tran.Kgrid should be over 1\n");
    }

    MPI_Finalize();
    exit(1);
  } 

  /**** source and drain bias voltage ****/

  input_logical("NEGF.bias.apply",&tran_bias_apply,1); /* default=on */

  if ( tran_bias_apply ) {

    double tmp;

    tran_biasvoltage_e[0] = 0.0;
    input_double("NEGF.bias.voltage", &tmp, 0.0);  /* in eV */
    tran_biasvoltage_e[1] = tmp/TRAN_eV2Hartree; 

  }
  else {
    tran_biasvoltage_e[0]=0.0;
    tran_biasvoltage_e[1]=0.0;
  }

  if (tran_bias_apply) {

    int side;
    side=0;
    TRAN_Apply_Bias2e(comm1,  side, tran_biasvoltage_e[side], TRAN_eV2Hartree,
		      SpinP_switch_e[side], atomnum_e[side],
		      WhatSpecies_e[side], Spe_Total_CNO_e[side], FNAN_e[side], natn_e[side],
		      Ngrid1_e[side], Ngrid2_e[side], Ngrid3_e[side], OLP_e[side][0],
		      &ChemP_e[side],H_e[side], dVHart_Grid_e[side] ); /* output */
    side=1;
    TRAN_Apply_Bias2e(comm1,  side, tran_biasvoltage_e[side], TRAN_eV2Hartree,
		      SpinP_switch_e[side], atomnum_e[side],
		      WhatSpecies_e[side], Spe_Total_CNO_e[side], FNAN_e[side], natn_e[side],
		      Ngrid1_e[side], Ngrid2_e[side], Ngrid3_e[side], OLP_e[side][0],
		      &ChemP_e[side], H_e[side], dVHart_Grid_e[side] ); /* output */
  }

  /**** gate voltage ****/

  /* beginning added by mari 07.30.2014 */
  if(TRAN_Poisson_flag == 3) {
    r_vec[0] = 0.0;
    r_vec[1] = 0.0;
    input_doublev("NEGF.gate.voltage", 2, tran_gate_voltage, r_vec); 
    tran_gate_voltage[0] /= TRAN_eV2Hartree; 
    tran_gate_voltage[1] /= TRAN_eV2Hartree; 
  } else {
    r_vec[0] = 0.0;
    input_doublev("NEGF.gate.voltage", 1, tran_gate_voltage, r_vec); 
    tran_gate_voltage[0] /= TRAN_eV2Hartree; 
  }
  /* end added by mari 07.30.2014 */

  /******************************************************
            parameters for the DOS calculation         
  ******************************************************/
  
  i=0;
  r_vec[i++] = -10.0;
  r_vec[i++] =  10.0;
  r_vec[i++] = 5.0e-3;
  input_doublev("NEGF.Dos.energyrange",i, tran_dos_energyrange, r_vec); /* in eV */
  /* change the unit from eV to Hartree */
  tran_dos_energyrange[0] /= TRAN_eV2Hartree;
  tran_dos_energyrange[1] /= TRAN_eV2Hartree;
  tran_dos_energyrange[2] /= TRAN_eV2Hartree;

  input_int("NEGF.Dos.energy.div",&tran_dos_energydiv,200);
  
  i_vec2[0]=1;
  i_vec2[1]=1;
  input_intv("NEGF.Dos.Kgrid",2,i_vec,i_vec2);
  TRAN_dos_Kspace_grid2 = i_vec[0];
  TRAN_dos_Kspace_grid3 = i_vec[1];

  /********************************************************
    integration on real axis with a small imaginary part
    for the "non-equilibrium" region
  ********************************************************/

  input_double("NEGF.bias.neq.im.energy", &Tran_bias_neq_im_energy, 1.0e-2);  /* in eV */
  if (Tran_bias_neq_im_energy<0.0) {
    if (myid==Host_ID)  printf("NEGF.bias.neq.im.energy should be positive.\n");
    MPI_Finalize();
    exit(1);
  }
  /* change the unit from eV to Hartree */
  Tran_bias_neq_im_energy /= TRAN_eV2Hartree; 

  input_double("NEGF.bias.neq.energy.step", &Tran_bias_neq_energy_step, 0.02);  /* in eV */
  if (Tran_bias_neq_energy_step<0.0) {
    if (myid==Host_ID)  printf("NEGF.bias.neq.energy.step should be positive.\n");
    MPI_Finalize();
    exit(1);
  }
  /* change the unit from eV to Hartree */
  Tran_bias_neq_energy_step /= TRAN_eV2Hartree; 

  input_double("NEGF.bias.neq.cutoff", &Tran_bias_neq_cutoff, 1.0e-8);  /* dimensionless */

  /********************************************************
     contour integration based on a continued 
     fraction representation of the Fermi function 
  ********************************************************/

  input_int("NEGF.Num.Poles", &tran_num_poles,150);

  TRAN_Set_IntegPath( comm1, TRAN_eV2Hartree, kBvalue, Electronic_Temperature );
}



