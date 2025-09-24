#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

void outputfile1(int f_switch, int MD_iter, int orbitalOpt_iter,
                 int Cnt_Now, int SCF_iter, char fname[YOUSO10], 
                 double ChemP_e0[2])
{
  FILE *fp;
  int myid;
  char buf[fp_bsize];          /* setvbuf */
  int i; /* S.Ryee */

  /* MPI */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){

    if (f_switch==1){
      fp = fopen(fname, "a");   
      if (fp==NULL){
        printf("error in outputfile1\n");
        exit(0); 
      }
      else{

        if (SCF_iter==1){
          fprintf(fp,"\n***********************************************************\n");
          fprintf(fp,"***********************************************************\n");
          fprintf(fp,"                  SCF history at MD=%2d                    \n",
                  MD_iter);
          fprintf(fp,"***********************************************************\n");
          fprintf(fp,"***********************************************************\n\n");
        }

        if ( Cnt_switch==0 || Cnt_Now!=1 ){
          fprintf(fp,"   SCF= %3d  NormRD= %15.12f  Uele= %15.12f\n",
                  SCF_iter,sqrt(fabs(NormRD[0])),Uele);
	}
        else if ( Cnt_switch==1 && Cnt_Now==1 ){
          fprintf(fp,"  OrbOpt=%3d  SCF= %3d  NormRD= %15.12f  Uele= %15.12f\n",
                  orbitalOpt_iter,SCF_iter,sqrt(fabs(NormRD[0])),Uele);
	}

        fclose(fp); 
      } 
    }

    else if (f_switch==2){

      fp = fopen(fname, "a");   

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (fp==NULL){
        printf("error in outputfile1\n");
        exit(0); 
      }
      else{
        fprintf(fp,"\n*******************************************************\n"); 
        fprintf(fp,"        Total energy (Hartree) at MD =%2d        \n",MD_iter);
        fprintf(fp,"*******************************************************\n\n"); 

	fprintf(fp,"  Uele.    %20.12f\n\n",Uele);
	fprintf(fp,"  Ukin.    %20.12f\n",Ukin);
	fprintf(fp,"  UH0.     %20.12f\n",UH0);
	fprintf(fp,"  UH1.     %20.12f\n",UH1);
	fprintf(fp,"  Una.     %20.12f\n",Una);
	fprintf(fp,"  Unl.     %20.12f\n",Unl);
	fprintf(fp,"  Uxc0.    %20.12f\n",Uxc0);
	fprintf(fp,"  Uxc1.    %20.12f\n",Uxc1);
	fprintf(fp,"  Ucore.   %20.12f\n",Ucore);
	fprintf(fp,"  Uhub.    %20.12f\n",Uhub);
	fprintf(fp,"  Ucs.     %20.12f\n",Ucs);
        fprintf(fp,"  Uzs.     %20.12f\n",Uzs);
        fprintf(fp,"  Uzo.     %20.12f\n",Uzo);
        fprintf(fp,"  Uef.     %20.12f\n",Uef);
        fprintf(fp,"  UvdW.    %20.12f\n",UvdW);
        fprintf(fp,"  Uch.     %20.12f\n",Uch);
        fprintf(fp,"  Utot.    %20.12f\n\n",Utot);
        fprintf(fp,"  UpV.     %20.12f\n",UpV);
        fprintf(fp,"  Enpy.    %20.12f\n",Utot+UpV);

	fprintf(fp,"  Note:\n\n");
	fprintf(fp,"  Utot = Ukin+UH0+UH1+Una+Unl+Uxc0+Uxc1+Ucore+Uhub+Ucs+Uzs+Uzo+Uef+UvdW+Uch\n\n");
        fprintf(fp,"  Uele:   band energy\n");
	fprintf(fp,"  Ukin:   kinetic energy\n");
	fprintf(fp,"  UH0:    electric part of screened Coulomb energy\n");
	fprintf(fp,"  UH1:    difference electron-electron Coulomb energy\n");
	fprintf(fp,"  Una:    neutral atom potential energy\n");
	fprintf(fp,"  Unl:    non-local potential energy\n");
	fprintf(fp,"  Uxc0:   exchange-correlation energy for alpha spin\n");
	fprintf(fp,"  Uxc1:   exchange-correlation energy for beta spin\n");
	fprintf(fp,"  Ucore:  core-core Coulomb energy\n");
	fprintf(fp,"  Uhub:   LDA+U energy\n");
	fprintf(fp,"  Ucs:    constraint energy for spin orientation\n");
        fprintf(fp,"  Uzs:    Zeeman term for spin magnetic moment\n");
        fprintf(fp,"  Uzo:    Zeeman term for orbital magnetic moment\n");
        fprintf(fp,"  Uef:    electric energy by electric field\n");
        fprintf(fp,"  UvdW:   semi-empirical vdW energy\n"); /* okuno */
        fprintf(fp,"  Uch:    penalty term to create a core hole\n\n"); 
        fprintf(fp,"  UpV:    pressure times volume\n"); 
        fprintf(fp,"  Enpy:   Enthalpy = Utot + UpV\n"); 
        fprintf(fp,"  (see also PRB 72, 045121(2005) for the energy contributions)\n\n");

        fprintf(fp,"\n\n");

        if (Solver==4){ /* NEGF */
          fprintf(fp,"  Chemical potential of left lead  (Hartree) %20.12f\n",ChemP_e0[0]);
          fprintf(fp,"  Chemical potential of right lead (Hartree) %20.12f\n",ChemP_e0[1]);
	}
        else{ 
          fprintf(fp,"  Chemical potential (Hartree) %20.12f\n",ChemP);
	}

        /***** start *****/ /* by S.Ryee */
        if(Hub_U_switch==1){	
          fprintf(fp,"\n");
          fprintf(fp,"*******************************************************\n");
          fprintf(fp,"                  DFT+U Type and DC                    \n");
          fprintf(fp,"*******************************************************\n"); 
          if(Hub_Type==2){
            switch (dc_Type){
            case 1:
              fprintf(fp,"     scf.DFTU.Type: 2(General)   scf.dc.Type: sFLL     \n" );
            break;

            case 2:
              fprintf(fp,"     scf.DFTU.Type: 2(General)   scf.dc.Type: sAMF     \n" );
            break;

            case 3:
              fprintf(fp,"     scf.DFTU.Type: 2(General)   scf.dc.Type: cFLL     \n" );
            break;

            case 4:
              fprintf(fp,"     scf.DFTU.Type: 2(General)   scf.dc.Type: cAMF     \n" );
            break;
            }
          }
          if(Hub_Type==1){
            fprintf(fp,"             scf.DFTU.Type: 1(Simplified)           \n" ); 
          }
          fprintf(fp, "\n");
        }
        if (Hub_U_switch==1 && Hub_Type==2 && Yukawa_on==1){  
          fprintf(fp,"\n*******************************************************\n"); 
          fprintf(fp,"    Thomas-Fermi screening length & Slater integrals     \n");
          fprintf(fp,"*******************************************************\n\n"); 

          for(i=1; i<=Nmul; i++){
            fprintf(fp, "<species: %s, angular momentum= %d, multiplicity number= %d>\n", SpeName[B_spe[i]], B_l[i], B_mul[i]);
            fprintf(fp, " TF-screening-length lambda= %f 1/au\n", lambda[i]);
            fprintf(fp, " Hubbard U= %f eV\n", U[i]*eV2Hartree);
            fprintf(fp, " Hund J= %f eV\n", J[i]*eV2Hartree);
            fprintf(fp, " Slater F0= %f eV\n", Slater_F0[i]*eV2Hartree); 
            fprintf(fp, " Slater F2= %f eV\n", Slater_F2[i]*eV2Hartree); 
            fprintf(fp, " Slater F4= %f eV\n", Slater_F4[i]*eV2Hartree); 
            if(B_l[i]==3){
              fprintf(fp, " Slater F6= %f eV\n", Slater_F6[i]*eV2Hartree); 
            }
            if(B_l[i]==2){
              fprintf(fp, " F4/F2= %f\n", Slater_F4[i]/Slater_F2[i]); 
            }
            if(B_l[i]==3){
              fprintf(fp, " F4/F2= %f\n", Slater_F4[i]/Slater_F2[i]); 
              fprintf(fp, " F6/F4= %f\n", Slater_F6[i]/Slater_F4[i]); 
            }
            fprintf(fp, "\n");
          }
        }
        /***** end *****/


        fclose(fp); 
      } 
    }

    else if (f_switch==3){

      fp = fopen(fname, "a");   

      setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

      if (fp==NULL){
        printf("error in outputfile1\n");
        exit(0); 
      }

      else{

        if (orbitalOpt_iter==1){
          fprintf(fp,"\n***********************************************************\n");
          fprintf(fp,"***********************************************************\n");
          fprintf(fp,"         History of orbital optimization   MD=%2d          \n",
                  MD_iter);
          fprintf(fp,"*********     Gradient Norm ((Hartree/borh)^2)     ********\n");
          fprintf(fp,"              Required criterion= %15.12f                  \n",
                  orbitalOpt_criterion);
          fprintf(fp,"***********************************************************\n\n");
        }

        fprintf(fp,"   iter= %3d  Gradient Norm= %15.12f  Uele= %15.12f\n",
                orbitalOpt_iter,Oopt_NormD[1],Uele);
        fclose(fp); 
      } 
    }
  }

}

