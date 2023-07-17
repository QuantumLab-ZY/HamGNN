/**********************************************************************
  Kerker_Mixing_Rhok.c:

     Kerker_Mixing_Rhok.c is a subroutine to achieve self-consistent
     field using the Kerker mixing in k-space. 

  Log of Kerker_Mixing_Rhok.c

     30/Dec/2004  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "tran_prototypes.h"
#include "tran_variables.h"
#include "mpi.h"
#include <omp.h>


#define  maxima_step  1000000.0

static void Kerker_Mixing_Rhok_Normal(
                               int Change_switch,
			       double Mix_wgt,
			       double ***ReRhok,
			       double ***ImRhok,
			       double **Residual_ReRhok,
			       double **Residual_ImRhok,
			       double *ReVk,
			       double *ImVk,
			       double *ReRhoAtomk,
			       double *ImRhoAtomk);


static void Kerker_Mixing_Rhok_NEGF(
                             int Change_switch,
			     double Mix_wgt,
			     double ***ReRhok,
			     double ***ImRhok,
			     double **Residual_ReRhok,
			     double **Residual_ImRhok,
			     double *ReVk,
			     double *ImVk,
			     double *ReRhoAtomk,
			     double *ImRhoAtomk);




void Kerker_Mixing_Rhok(int Change_switch,
                        double Mix_wgt,
                        double ***ReRhok,
                        double ***ImRhok,
                        double **Residual_ReRhok,
                        double **Residual_ImRhok,
                        double *ReVk,
                        double *ImVk,
                        double *ReRhoAtomk,
                        double *ImRhoAtomk)
{

  if (Solver!=4 || TRAN_Poisson_flag==2){

    Kerker_Mixing_Rhok_Normal(Change_switch,
			      Mix_wgt,
			      ReRhok,
			      ImRhok,
			      Residual_ReRhok,
			      Residual_ImRhok,
			      ReVk,
			      ImVk,
			      ReRhoAtomk,
			      ImRhoAtomk);
  }

  else if (Solver==4){

    Kerker_Mixing_Rhok_NEGF(Change_switch,
			    Mix_wgt,
			    ReRhok,
			    ImRhok,
			    Residual_ReRhok,
			    Residual_ImRhok,
			    ReVk,
			    ImVk,
			    ReRhoAtomk,
			    ImRhoAtomk);
  }

}










void Kerker_Mixing_Rhok_Normal(int Change_switch,
			       double Mix_wgt,
			       double ***ReRhok,
			       double ***ImRhok,
			       double **Residual_ReRhok,
			       double **Residual_ImRhok,
			       double *ReVk,
			       double *ImVk,
			       double *ReRhoAtomk,
			       double *ImRhoAtomk)
{
  static int firsttime=1;
  int ian,jan,Mc_AN,Gc_AN,spin,spinmax;
  int h_AN,Gh_AN,m,n,i,j,k,k1,k2,k3;
  int MN,pSCF_iter,p0,p1;
  int GN,GNs,N2D,BN_AB,N3[4];
  double Mix_wgt2,Norm,My_Norm,tmp0,tmp1;
  double Min_Weight,Max_Weight,wgt0,wgt1;
  double Gx,Gy,Gz,G2,size_Kerker_weight;
  double sk1,sk2,sk3,G12,G22,G32;
  double G0,G02,G02p,weight;
  int numprocs,myid,ID;
  double *Kerker_weight;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocate arrays */

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = (double*)malloc(sizeof(double)*My_NumGridB_CB); 

  if (firsttime)
  PrintMemory("Kerker_Mixing_Rhok: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
  firsttime=0;

  /* find an optimum G0 */

  G12 = rtv[1][1]*rtv[1][1] + rtv[1][2]*rtv[1][2] + rtv[1][3]*rtv[1][3]; 
  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G12<G22) G0 = G12;
  else         G0 = G22;
  if (G32<G0)  G0 = G32;

  if (Change_switch==0) G0 = 0.0;
  else                  G0 = sqrt(G0);

  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.2*Kerker_factor*G0)*(0.2*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /***********************************
            set Kerker_weight 
  ************************************/

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  for (k=0; k<My_NumGridB_CB; k++){

    /* get k3, k2, and k1 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
    k1 = GN - k3*Ngrid2*Ngrid1 - k2*Ngrid1; 

    if (k1<Ngrid1/2)  sk1 = (double)k1;
    else              sk1 = (double)(k1 - Ngrid1);

    if (k2<Ngrid2/2)  sk2 = (double)k2;
    else              sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2)  sk3 = (double)k3;
    else              sk3 = (double)(k3 - Ngrid3);

    Gx = sk1*rtv[1][1] + sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk1*rtv[1][2] + sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk1*rtv[1][3] + sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    if (k1==0 && k2==0 && k3==0)  weight = 1.0;
    else                          weight = (G2 + G02)/(G2+G02p);

    Kerker_weight[k] = sqrt(weight);

  } /* k */

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
       norm of residual charge density in k-space
  ****************************************************/

  MPI_Barrier(mpi_comm_level1);

  My_Norm = 0.0;
  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = Kerker_weight[k];
      tmp0 = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*weight;
      tmp1 = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*weight;
      Residual_ReRhok[spin][My_NumGridB_CB+k] = tmp0;
      Residual_ImRhok[spin][My_NumGridB_CB+k] = tmp1;
      My_Norm += tmp0*tmp0 + tmp1*tmp1;

    } /* k */
  } 

  /****************************************************
    MPI: 

    My_Norm
  ****************************************************/

  MPI_Allreduce(&My_Norm, &Norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  /* normalization by the number of grids */
  Norm = Norm/(double)(Ngrid1*Ngrid2*Ngrid3);

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

    if ((int)sgn(History_Uele[0]-History_Uele[1])
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
   
    else if ((int)sgn(History_Uele[0]-History_Uele[1])
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

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
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

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
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

  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = 1.0/Kerker_weight[k];

      if (Change_switch==0){
	wgt0 = 0.0;
	wgt1 = 1.0;
      }
      else{
	wgt0  = Mix_wgt*weight;
	wgt1 =  1.0 - wgt0;
      }

      ReRhok[0][spin][k] = wgt0*ReRhok[0][spin][k] + wgt1*ReRhok[1][spin][k];
      ImRhok[0][spin][k] = wgt0*ImRhok[0][spin][k] + wgt1*ImRhok[1][spin][k];

      /* correction to largely changing components */

      tmp0 = ReRhok[0][spin][k] - ReRhok[1][spin][k];  
      tmp1 = ImRhok[0][spin][k] - ImRhok[1][spin][k];  

      if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
	ReRhok[0][spin][k] = sgn(tmp0)*maxima_step + ReRhok[1][spin][k]; 
	ImRhok[0][spin][k] = sgn(tmp1)*maxima_step + ImRhok[1][spin][k]; 
      }

    } /* k */
  } /* spin */

  /****************************************************
                         shift of rho
  ****************************************************/

  for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){
    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReRhok[pSCF_iter][spin][k] = ReRhok[pSCF_iter-1][spin][k]; 
	ImRhok[pSCF_iter][spin][k] = ImRhok[pSCF_iter-1][spin][k]; 
      }
    }
  }

  /****************************************************
                    shift of residual rho
  ****************************************************/

  for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

    p0 = pSCF_iter*My_NumGridB_CB;
    p1 = (pSCF_iter-1)*My_NumGridB_CB; 

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	Residual_ReRhok[spin][p0+k] = Residual_ReRhok[spin][p1+k];
	Residual_ImRhok[spin][p0+k] = Residual_ImRhok[spin][p1+k]; 
      }
    }
  }

  /************************************************************
    find the charge density for the partition B in real space 
  ************************************************************/

  tmp0 = 1.0/(double)(Ngrid1*Ngrid2*Ngrid3);

  for (spin=0; spin<spinmax; spin++){

    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][spin][k];
      ImVk[k] = ImRhok[0][spin][k];
    }

    if (spin==0 || spin==1){

      Get_Value_inReal(0,Density_Grid_B[spin],Density_Grid_B[spin],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Density_Grid_B[spin][BN_AB] = Density_Grid_B[spin][BN_AB]*tmp0;
      }
    }

    else if (spin==2){

      Get_Value_inReal(1,Density_Grid_B[2],Density_Grid_B[3],ReVk,ImVk);

      for (BN_AB=0; BN_AB<My_NumGridB_AB; BN_AB++){
	Density_Grid_B[2][BN_AB] = Density_Grid_B[2][BN_AB]*tmp0;
	Density_Grid_B[3][BN_AB] = Density_Grid_B[3][BN_AB]*tmp0;
      }
    }
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B);

  /****************************************************
      set ReVk and ImVk which are used in Poisson.c
  ****************************************************/

  if (SpinP_switch==0){
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = 2.0*ReRhok[0][0][k] - ReRhoAtomk[k];
      ImVk[k] = 2.0*ImRhok[0][0][k] - ImRhoAtomk[k];
    }
  }
  
  else {
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][0][k] + ReRhok[0][1][k] - ReRhoAtomk[k];
      ImVk[k] = ImRhok[0][0][k] + ImRhok[0][1][k] - ImRhoAtomk[k];
    }
  }

  /* freeing of arrays */

  free(Kerker_weight);
}



void Kerker_Mixing_Rhok_NEGF(int Change_switch,
			     double Mix_wgt,
			     double ***ReRhok,
			     double ***ImRhok,
			     double **Residual_ReRhok,
			     double **Residual_ImRhok,
			     double *ReVk,
			     double *ImVk,
			     double *ReRhoAtomk,
			     double *ImRhoAtomk)
{
  static int firsttime=1;
  int ian,jan,Mc_AN,Gc_AN,spin,spinmax;
  int h_AN,Gh_AN,m,n,n1,i,j,k,k1,k2,k3;
  int MN,pSCF_iter,p0,p1;
  int BN_AB,N2D,GN,GNs,N3[4];
  double Mix_wgt2,Norm,My_Norm,tmp0,tmp1;
  double Min_Weight,Max_Weight,wgt0,wgt1;
  double Gx,Gy,Gz,G2,size_Kerker_weight;
  double sk1,sk2,sk3,G12,G22,G32;
  double G0,G02,G02p,weight;
  double Q,sumL,sumR;
  int numprocs,myid,ID;
  double *Kerker_weight;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs,Nloop,Nthrds0;
  double *My_Norm_threads;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* allocate arrays */

  size_Kerker_weight = My_NumGridB_CB;
  Kerker_weight = (double*)malloc(sizeof(double)*My_NumGridB_CB);
  
  /* find an optimum G0 */

  G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
  G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

  if (G22<G32) G0 = G22;
  else         G0 = G32;

  if (Change_switch==0) G0 = 0.0;
  else                  G0 = sqrt(G0);

  G02 = Kerker_factor*Kerker_factor*G0*G0;
  G02p = (0.1*Kerker_factor*G0)*(0.1*Kerker_factor*G0);

  if      (SpinP_switch==0)  spinmax = 1;
  else if (SpinP_switch==1)  spinmax = 2;
  else if (SpinP_switch==3)  spinmax = 3;

  /***********************************
            set Kerker_weight 
  ************************************/

  N2D = Ngrid3*Ngrid2;
  GNs = ((myid*N2D+numprocs-1)/numprocs)*Ngrid1;

  if (firsttime)
  PrintMemory("Kerker_Mixing_Rhok: Kerker_weight",sizeof(double)*size_Kerker_weight,NULL);
  firsttime=0;

  for (k=0; k<My_NumGridB_CB; k+=Ngrid1){

    /* get k3, k2, and k1 */

    GN = k + GNs;     
    k3 = GN/(Ngrid2*Ngrid1);    
    k2 = (GN - k3*Ngrid2*Ngrid1)/Ngrid1;
 
    if (k2<Ngrid2/2)  sk2 = (double)k2;
    else              sk2 = (double)(k2 - Ngrid2);

    if (k3<Ngrid3/2)  sk3 = (double)k3;
    else              sk3 = (double)(k3 - Ngrid3);

    Gx = sk2*rtv[2][1] + sk3*rtv[3][1];
    Gy = sk2*rtv[2][2] + sk3*rtv[3][2]; 
    Gz = sk2*rtv[2][3] + sk3*rtv[3][3];
    G2 = Gx*Gx + Gy*Gy + Gz*Gz;

    for (k1=0; k1<Ngrid1; k1++){

      if (Change_switch==0){
	weight = (G2 + G02 + 0.1)/(G2 + 0.01);
      }
      else{

	if (k2==0 && k3==0){

	  sumL = 0.0;
	  sumR = 0.0; 

	  if (SpinP_switch==0){

	    for (n1=0; n1<k1; n1++){
	      sumL += 2.0*ReRhok[0][0][k+n1];
	    }

	    for (n1=k1+1; n1<Ngrid1; n1++){
	      sumR += 2.0*ReRhok[0][0][k+n1];
	    }
	  }
        
	  else if (SpinP_switch==1 || SpinP_switch==3){

	    for (n1=0; n1<k1; n1++){
	      sumL += ReRhok[0][0][k+n1] + ReRhok[0][1][k+n1];
	    }

	    for (n1=k1+1; n1<Ngrid1; n1++){
	      sumR += ReRhok[0][0][k+n1] + ReRhok[0][1][k+n1];
	    }
	  }

	  Q = 4.0*fabs(sumL - sumR)*GridVol + 1.0;
	}    
	else {
	  Q = 1.0;
	}

	weight = (G2 + G02)/(G2 + G02p)*Q;
      }

      Kerker_weight[k+k1] = sqrt(weight);

    } /* k1 */
  } /* k */

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
       norm of residual charge density in k-space
  ****************************************************/

  My_Norm = 0.0;
  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = Kerker_weight[k];
      tmp0 = (ReRhok[0][spin][k] - ReRhok[1][spin][k])*weight;
      tmp1 = (ImRhok[0][spin][k] - ImRhok[1][spin][k])*weight;
      Residual_ReRhok[spin][My_NumGridB_CB+k] = tmp0;
      Residual_ImRhok[spin][My_NumGridB_CB+k] = tmp1;
      My_Norm += tmp0*tmp0 + tmp1*tmp1;

    } /* k */
  }/* spin */

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

    if ((int)sgn(History_Uele[0]-History_Uele[1])
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
   
    else if ((int)sgn(History_Uele[0]-History_Uele[1])
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

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
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

    else if ((int)sgn(History_Uele[0]-History_Uele[1])
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

  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){

      weight = 1.0/Kerker_weight[k];
      wgt0  = Mix_wgt*weight;
      wgt1 =  1.0 - wgt0;

      /***********************************
       [0]: output at n -> input at n+1
       [1]: input  at n
      ***********************************/

      ReRhok[0][spin][k] = wgt0*ReRhok[0][spin][k] + wgt1*ReRhok[1][spin][k];
      ImRhok[0][spin][k] = wgt0*ImRhok[0][spin][k] + wgt1*ImRhok[1][spin][k];

      /* correction to large changing components */

      tmp0 = ReRhok[0][spin][k] - ReRhok[1][spin][k];  
      tmp1 = ImRhok[0][spin][k] - ImRhok[1][spin][k];  

      if ( maxima_step<(fabs(tmp0)+fabs(tmp1)) ){
	ReRhok[0][spin][k] = sgn(tmp0)*maxima_step + ReRhok[1][spin][k]; 
	ImRhok[0][spin][k] = sgn(tmp1)*maxima_step + ImRhok[1][spin][k]; 
      }

    } /* k */
  } /* spin */

  /****************************************************
                       shift of rho
  ****************************************************/

  for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){
    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	ReRhok[pSCF_iter][spin][k] = ReRhok[pSCF_iter-1][spin][k]; 
	ImRhok[pSCF_iter][spin][k] = ImRhok[pSCF_iter-1][spin][k];
      }
    }
  }

  /****************************************************
                    shift of residual rho
  ****************************************************/

  for (pSCF_iter=(List_YOUSO[38]-1); 0<pSCF_iter; pSCF_iter--){

    p0 = pSCF_iter*My_NumGridB_CB;
    p1 = (pSCF_iter-1)*My_NumGridB_CB; 

    for (spin=0; spin<spinmax; spin++){
      for (k=0; k<My_NumGridB_CB; k++){
	Residual_ReRhok[spin][p0+k] = Residual_ReRhok[spin][p1+k];
	Residual_ImRhok[spin][p0+k] = Residual_ImRhok[spin][p1+k]; 
      }
    }
  }

  /****************************************************
        find the charge density in real space 
  ****************************************************/

  for (spin=0; spin<spinmax; spin++){
    for (k=0; k<My_NumGridB_CB; k++){
      ReVk[k] = ReRhok[0][spin][k];
      ImVk[k] = ImRhok[0][spin][k];
    }

    /* revised by Y. Xiao for Noncollinear NEGF calculations */
    if (spin==0 || spin==1) {
      Get_Value_inReal2D(0, Density_Grid_B[spin], NULL, ReVk, ImVk);
    } else {
      Get_Value_inReal2D(1, Density_Grid_B[2], Density_Grid_B[3], ReVk, ImVk);
    }
    /* until here by Y. Xiao for Noncollinear NEGF calculations */

    /*  Get_Value_inReal2D(0, Density_Grid_B[spin], NULL, ReVk, ImVk); */
  }

  /******************************************************
             MPI: from the partitions B to D
  ******************************************************/

  Density_Grid_Copy_B2D(Density_Grid_B);

  /* freeing of arrays */
  free(Kerker_weight);
}
