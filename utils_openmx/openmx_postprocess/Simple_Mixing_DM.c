/**********************************************************************
  Simple_Mixing_DM.c:

     Simple_Mixing_DM.c is a subroutine to achieve self-consistent
     field using the simple mixing method.

  Log of Simple_Mixing_DM.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"


void Simple_Mixing_DM(int Change_switch,
                      double Mix_wgt,
                      double *****CDM,
                      double *****PDM,
                      double *****P2DM,
                      double *****iCDM,
                      double *****iPDM,
                      double *****iP2DM,
                      double *****RDM,
                      double *****iRDM)
{
  int ian,jan,Mc_AN,Gc_AN;
  int h_AN,Gh_AN,m,n,spin,i;
  double Mix_wgt1,Mix_wgt2,Norm,My_Norm,tmp0;
  double Min_Weight,Max_Weight;
  double nc_weight[4];
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* start... */

  Min_Weight = Min_Mixing_weight;
  if (SCF_RENZOKU==-1){
    Max_Weight = Max_Mixing_weight;
    Max_Mixing_weight2 = Max_Mixing_weight;
  }
  else if (SCF_RENZOKU==1000){  /* past 3 */
    Max_Mixing_weight2 = 2.0*Max_Mixing_weight2;
    if (0.7<Max_Mixing_weight2) Max_Mixing_weight2 = 0.7;
    Max_Weight = Max_Mixing_weight2;
    SCF_RENZOKU = 0;
  }
  else{
    Max_Weight = Max_Mixing_weight2;
  }

  /****************************************************
                        NormRD
  ****************************************************/

  My_Norm = 0.0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];      
      jan = Spe_Total_CNO[WhatSpecies[Gh_AN]];

      for (spin=0; spin<=SpinP_switch; spin++){
        for (m=0; m<ian; m++){
          for (n=0; n<jan; n++){

            RDM[spin][Mc_AN][h_AN][m][n] = CDM[spin][Mc_AN][h_AN][m][n]
                                          -PDM[spin][Mc_AN][h_AN][m][n];
            My_Norm += RDM[spin][Mc_AN][h_AN][m][n]*RDM[spin][Mc_AN][h_AN][m][n];
          }
        }
      }

      if (SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch 
          || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )){ 

        for (spin=0; spin<2; spin++){
  	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      iRDM[spin][Mc_AN][h_AN][m][n] = iCDM[spin][Mc_AN][h_AN][m][n]
                                             -iPDM[spin][Mc_AN][h_AN][m][n];
	      My_Norm += iRDM[spin][Mc_AN][h_AN][m][n]*iRDM[spin][Mc_AN][h_AN][m][n];
	    }
	  }
	}
      }
    }
  }

  /****************************************************
    MPI: 

    My_Norm
  ****************************************************/

  MPI_Allreduce(&My_Norm, &Norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /****************************************************
    find an optimum mixing weight
  ****************************************************/

  for (i=4; 1<=i; i--){
    NormRD[i] = NormRD[i-1];
    History_Uele[i] = History_Uele[i-1];
  }
  NormRD[0] = Norm;
  History_Uele[0] = Uele;




  if (Change_switch==1){

    if ( (int)sgn(History_Uele[0]-History_Uele[1])
	  ==(int)sgn(History_Uele[1]-History_Uele[2])
       && NormRD[0]<NormRD[1]){

      /* tmp0 = 1.6*Mixing_weight; */

      tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0){
          Mixing_weight = tmp0;
	}
        else{ 
          Mixing_weight = Min_Weight;
	}
      }
      else{ 
        Mixing_weight = Max_Weight;
        SCF_RENZOKU++;  
      }
    }
   
    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
            ==(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[1]<NormRD[0]){

      tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

      /* tmp0 = Mixing_weight/1.6; */

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else 
        Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;  
    }

    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
	       !=(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[0]<NormRD[1]){

      /* tmp0 = Mixing_weight*1.2; */

      tmp0 = NormRD[1]/(largest(NormRD[1]-NormRD[0],10e-10))*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else{ 
        Mixing_weight = Max_Weight;
        SCF_RENZOKU++;
      }
    }

    else if ( (int)sgn(History_Uele[0]-History_Uele[1])
	       !=(int)sgn(History_Uele[1]-History_Uele[2])
             && NormRD[1]<NormRD[0]){

      /* tmp0 = Mixing_weight/2.0; */

      tmp0 = NormRD[1]/(largest(NormRD[1]+NormRD[0],10e-10))*Mixing_weight;

      if (tmp0<Max_Weight){
        if (Min_Weight<tmp0)
          Mixing_weight = tmp0;
        else 
          Mixing_weight = Min_Weight;
      }
      else 
        Mixing_weight = Max_Weight;

      SCF_RENZOKU = -1;
    }

    Mix_wgt = Mixing_weight;
  }

  /****************************************************
                        Mixing
  ****************************************************/

  nc_weight[0] = 1.0;
  nc_weight[1] = 1.0;
  nc_weight[2] = 1.0;
  nc_weight[3] = 1.0;

  for (spin=0; spin<=SpinP_switch; spin++){

    Mix_wgt1 = nc_weight[spin]*Mix_wgt;
    Mix_wgt2 = 1.0 - Mix_wgt1;
 
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      ian = Spe_Total_CNO[WhatSpecies[Gc_AN]];
      for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	Gh_AN = natn[Gc_AN][h_AN];      
	jan = Spe_Total_CNO[WhatSpecies[Gh_AN]];

	for (m=0; m<ian; m++){
	  for (n=0; n<jan; n++){

            CDM[spin][Mc_AN][h_AN][m][n] = Mix_wgt1 * CDM[spin][Mc_AN][h_AN][m][n]
 	                                 + Mix_wgt2 * PDM[spin][Mc_AN][h_AN][m][n];

            P2DM[spin][Mc_AN][h_AN][m][n] = PDM[spin][Mc_AN][h_AN][m][n];
            PDM[spin][Mc_AN][h_AN][m][n]  = CDM[spin][Mc_AN][h_AN][m][n];
          }
        }

        if ( (SpinP_switch==3 && ( SO_switch==1 || Hub_U_switch==1 || 1<=Constraint_NCS_switch
              || Zeeman_NCS_switch==1 || Zeeman_NCO_switch==1 )) && spin<=1 ){ 

	  for (m=0; m<ian; m++){
	    for (n=0; n<jan; n++){

	      iCDM[spin][Mc_AN][h_AN][m][n] = Mix_wgt1 * iCDM[spin][Mc_AN][h_AN][m][n]
                                            + Mix_wgt2 * iPDM[spin][Mc_AN][h_AN][m][n];

	      iP2DM[spin][Mc_AN][h_AN][m][n] = iPDM[spin][Mc_AN][h_AN][m][n];
	      iPDM[spin][Mc_AN][h_AN][m][n]  = iCDM[spin][Mc_AN][h_AN][m][n];
	    }
	  }
	}

      }
    }
  }

}
