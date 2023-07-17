#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"



double Mixing_DM(int MD_iter,
                 int SCF_iter,
                 int SCF_iter0,
                 int SucceedReadingDMfile,
                 double ***ReRhok,
                 double ***ImRhok,
                 double **Residual_ReRhok,
                 double **Residual_ImRhok,
                 double *ReVk,
                 double *ImVk,
                 double *ReRhoAtomk,
                 double *ImRhoAtomk)
{
  int pSCF_iter,NumMix,NumSlide;
  int spin,ct_AN,wan1,TNO1,h_AN,Gh_AN,wan2;
  int TNO2,i,j,ian,jan,m,n;
  int spinmax,k1,k2,k3;
  int Mc_AN,Gc_AN;
  double time0;
  double TStime,TEtime;
  int numprocs,myid,ID;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  /* In case of rmm-diish or rmm-diisv, then return. */
  if ( Cnt_switch!=1 && (Mixing_switch==5 || Mixing_switch==7) ) return 0.0; 

  /* In case of (rmm-diis || rmm-adiis) && SCF_iter<(Pulay_SCF/2)), then return. */
  if ( Cnt_switch!=1 && SucceedReadingDMfile!=1 
       && (Mixing_switch==1 || Mixing_switch==6) && SCF_iter<=(Pulay_SCF/2) && Solver!=4){

    return 0.0; 
  }

  /*******************************************************
                     Simple Mixing
  *******************************************************/

  if (Mixing_switch==0){

    if (MD_iter==1 && SCF_iter==1){

      Simple_Mixing_DM(0,1.0,DM[0],DM[1],DM[2], 
		       iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);
    }
    else{

      Simple_Mixing_DM(1,Mixing_weight,DM[0],DM[1],DM[2],
		       iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);
    }
  }

  /*******************************************************
    Pulay's method

    Residual Minimazation Method (RMM) using
    Direct Inversion in the Iterative Subspace (DIIS)
  *******************************************************/

  else if (Mixing_switch==1){

    if (SCF_iter==1){
      DIIS_Mixing_DM(1,ResidualDM,iResidualDM);
    }
    else if (SCF_iter<=(Pulay_SCF-1)){

      Simple_Mixing_DM(1,Mixing_weight,DM[0],DM[1],DM[2],
		       iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);
    }
    else if (SCF_iter==Pulay_SCF)
      DIIS_Mixing_DM(2,ResidualDM,iResidualDM);
    else 
      DIIS_Mixing_DM(SCF_iter-(Pulay_SCF-2),ResidualDM,iResidualDM);
  }

  /*********************************************************
     Guaranteed-Reduction Pulay's method (GR-Pulay method)
  *********************************************************/

  else if (Mixing_switch==2){
    if (SCF_iter==1){
      GR_Pulay_DM(1,ResidualDM);
    }
    else if (SCF_iter<=(Pulay_SCF-1))
      Simple_Mixing_DM(1,Mixing_weight,DM[0],DM[1],DM[2],
		       iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);
    else if (SCF_iter==Pulay_SCF)
      GR_Pulay_DM(2,ResidualDM);
    else 
      GR_Pulay_DM(SCF_iter-(Pulay_SCF-2),ResidualDM);
  }
  
  /*********************************************************
               Kerker simple mixing in k-space
  *********************************************************/

  else if (Mixing_switch==3){

    if (MD_iter==1 && SCF_iter0==1 && SucceedReadingDMfile==0){
      Kerker_Mixing_Rhok(0, Mixing_weight,
                         ReRhok,ImRhok,
                         Residual_ReRhok,Residual_ImRhok,
                         ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
    }

    else if (MD_iter==1 && SCF_iter==1){
      Kerker_Mixing_Rhok(0, Mixing_weight,
                         ReRhok,ImRhok,
                         Residual_ReRhok,Residual_ImRhok,
                         ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
    }

    else if ( (SCF_iter%30)==1) {
      Kerker_Mixing_Rhok(1,Max_Mixing_weight,
                         ReRhok,ImRhok,
                         Residual_ReRhok,Residual_ImRhok,
                         ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
    }

    else{
      Kerker_Mixing_Rhok(1,Mixing_weight,
                         ReRhok,ImRhok,
                         Residual_ReRhok,Residual_ImRhok,
                         ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
    }
  }

  /*********************************************************
   RMM-DIIS mixing with Kerker's weighting factor in k-space
  *********************************************************/

  else if (Mixing_switch==4){
 
    if (MD_iter==1 && SCF_iter==1){
      Kerker_Mixing_Rhok(0,Mixing_weight,
                         ReRhok,ImRhok,
                         Residual_ReRhok,Residual_ImRhok,
                         ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
    }

    else if (SCF_iter<Pulay_SCF){
      Kerker_Mixing_Rhok(1,Mixing_weight,
                         ReRhok,ImRhok,
                         Residual_ReRhok,Residual_ImRhok,
                         ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
    }

    else{

      if (6<SCF_iter){
	DIIS_Mixing_Rhok(SCF_iter-(Pulay_SCF-3),
			 Mixing_weight,
			 ReRhok,ImRhok,
			 Residual_ReRhok,Residual_ImRhok,
			 ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
      }
      else{
	DIIS_Mixing_Rhok(SCF_iter,
			 Mixing_weight,
			 ReRhok,ImRhok,
			 Residual_ReRhok,Residual_ImRhok,
			 ReVk,ImVk,ReRhoAtomk,ImRhoAtomk);
      }  

    }
  }

  /*******************************************************
    ADIIS method:
    Augmented Roothaan-Hall (ARH) energy function 
    +
    Direct Inversion in the Iterative Subspace (DIIS)
  *******************************************************/

  else if (Mixing_switch==6){

    if (SCF_iter<=(Pulay_SCF-1)){
      Simple_Mixing_DM(1,Mixing_weight,DM[0],DM[1],DM[2],
		       iDM[0],iDM[1],iDM[2],ResidualDM[2],iResidualDM[2]);

    }
    else 
      ADIIS_Mixing_DM(SCF_iter-Pulay_SCF+2,ResidualDM,iResidualDM);
  }

  /*******************************************************
                       not supported 
  *******************************************************/

  else {
    printf("Mixing_switch=%i is not supported.\n",Mixing_switch);
    MPI_Finalize();
    exit(0);
  }

  /* if SCF_iter0==1, then NormRD[0]=1 */

  if (SCF_iter0==1) NormRD[0] = 1.0;

  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;
} 



