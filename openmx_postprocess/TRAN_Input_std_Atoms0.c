/**********************************************************************
  TRAN_Input_std_Atoms.c:

  TRAN_Input_std_Atoms.c is a subroutine to read the input data.

  Log of TRAN_Input_std_Atoms.c:

     24/July/2008  Released by H.Kino and T.Ozaki

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Inputtools.h"
#include "openmx_common.h"
#include <mpi.h>
#include "tran_prototypes.h"
#include "tran_variables.h"

#define MAXBUF 1024

int OrbPol2int(char OrbPol[YOUSO10]);


void TRAN_Input_std_Atoms(  MPI_Comm comm1, int Solver )
{
  int po=0;
  int idim=1;
  int myid;

  FILE *fp;
  char *s_vec[20];
  int i_vec[20];
  double r_vec[20];

  int i,j,spe,spe_e; 
  char Species[YOUSO10];
  double GO_XR,GO_YR,GO_ZR;
  double GO_XL,GO_YL,GO_ZL;
  char OrbPol[YOUSO10];
  char buf[MAXBUF];
/* revised by Y. Xiao for Noncollinear NEGF calculations */
  double mx,my,mz,tmp,S_coordinate[3];
/*until here by Y. Xiao for Noncollinear NEGF calculations */

  if (Solver!=4) return; 

  MPI_Comm_rank(comm1,&myid);

  /* center */
  input_int("Atoms.Number",&Catomnum,0);
  if (Catomnum<=0){

    if (myid==Host_ID){
      printf("Atoms.Number may be wrong.\n");
    }
    MPI_Finalize();
    exit(1);
  }

  /* left */
  input_int("LeftLeadAtoms.Number",&Latomnum,0);
  if (Latomnum<=0){

    if (myid==Host_ID){
      printf("LeftLeadAtoms.Number may be wrong.\n");
    }
    MPI_Finalize();
    exit(1);
  }

  /* right */
  input_int("RightLeadAtoms.Number",&Ratomnum,0);
  if (Ratomnum<=0){

    if (myid==Host_ID){
      printf("RightLeadAtoms.Number may be wrong.\n");
    }
    MPI_Finalize();
    exit(1);
  }
    
  atomnum = Catomnum + Latomnum + Ratomnum;
  List_YOUSO[1] = atomnum + 1;

  /* memory allocation */

  Allocate_Arrays(1);

  /* memory allocation for TRAN_* */
  TRAN_Allocate_Atoms(atomnum);

  s_vec[0]="Ang";  s_vec[1]="AU";
  i_vec[0]= 0;     i_vec[1]=1;
  input_string2int("Atoms.SpeciesAndCoordinates.Unit",
		   &coordinates_unit,2,s_vec,i_vec);

  /* left lead */

  if (fp=input_find("<LeftLeadAtoms.SpeciesAndCoordinates") ) {

    for (i=1; i<=Latomnum; i++){

      fgets(buf,MAXBUF,fp);
/* revised by Y. Xiao for Noncollinear NEGF calculations */

        /* spin non-collinear */
        if (SpinP_switch==3){

          sscanf(buf,"%i %s %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %s",
                 &j, Species,
                 &Gxyz[i][1],&Gxyz[i][2],&Gxyz[i][3],
                 &InitN_USpin[i],&InitN_DSpin[i],
                 &Angle0_Spin[i], &Angle1_Spin[i],
                 &Angle0_Orbital[i], &Angle1_Orbital[i],
                 &Constraint_SpinAngle[i],
                 OrbPol );

          if (fabs(Angle0_Spin[i])<1.0e-10){
            Angle0_Spin[i] = Angle0_Spin[i] + rnd(1.0e-5);
          }

          Angle0_Spin[i] = PI*Angle0_Spin[i]/180.0;
          Angle1_Spin[i] = PI*Angle1_Spin[i]/180.0;
          InitAngle0_Spin[i] = Angle0_Spin[i];
          InitAngle1_Spin[i] = Angle1_Spin[i];

          if (fabs(Angle0_Orbital[i])<1.0e-10){
            Angle0_Orbital[i] = Angle0_Orbital[i] + rnd(1.0e-5);
          }

          Angle0_Orbital[i] = PI*Angle0_Orbital[i]/180.0;
          Angle1_Orbital[i] = PI*Angle1_Orbital[i]/180.0;
          InitAngle0_Orbital[i] = Angle0_Orbital[i];
          InitAngle1_Orbital[i] = Angle1_Orbital[i];

      /*check whether the Euler angle measured from the direction (1,0) is used*/
      if ( (InitN_USpin[i]-InitN_DSpin[i]) < 0.0){
            
            mx = -sin(InitAngle0_Spin[i])*cos(InitAngle1_Spin[i]);
            my = -sin(InitAngle0_Spin[i])*sin(InitAngle1_Spin[i]);
            mz = -cos(InitAngle0_Spin[i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

            Angle0_Spin[i] = S_coordinate[1];
            Angle1_Spin[i] = S_coordinate[2];
            InitAngle0_Spin[i] = Angle0_Spin[i];
            InitAngle1_Spin[i] = Angle1_Spin[i];

            tmp = InitN_USpin[i];
            InitN_USpin[i] = InitN_DSpin[i];
            InitN_DSpin[i] = tmp;

            mx = -sin(InitAngle0_Orbital[i])*cos(InitAngle1_Orbital[i]);
            my = -sin(InitAngle0_Orbital[i])*sin(InitAngle1_Orbital[i]);
            mz = -cos(InitAngle0_Orbital[i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

            Angle0_Orbital[i] = S_coordinate[1];
            Angle1_Orbital[i] = S_coordinate[2];
            InitAngle0_Orbital[i] = Angle0_Orbital[i];
            InitAngle1_Orbital[i] = Angle1_Orbital[i];

      }  /* if ( (InitN_USpin[i]-InitN_DSpin[i]) < 0.0)  */  
     } /* if (SpinP_switch == 3) */

     /* spin collinear */
     else {

      sscanf(buf,"%i %s %lf %lf %lf %lf %lf %s",&j,Species,
	     &Gxyz[i][1],&Gxyz[i][2],&Gxyz[i][3],
	     &InitN_USpin[i],&InitN_DSpin[i], OrbPol);

     }

      WhatSpecies[i] = Species2int(Species);
      TRAN_region[i]=2;
      TRAN_Original_Id[i]=j;

      if (Hub_U_switch==1) OrbPol_flag[i] = OrbPol2int(OrbPol);

      /* check the consistency of basis set */

      spe   = WhatSpecies[i];
      spe_e = WhatSpecies_e[0][i];

      if (i!=j){

        if (myid==Host_ID){
	  printf("Error of sequential number %i in <LeftLeadAtoms.SpeciesAndCoordinates\n",j);
        }
        MPI_Finalize();
        exit(1);
      }

      if (2<=level_stdout && myid==Host_ID){

	printf("<Input_std> L_AN=%2d T_AN=%2d WhatSpecies=%2d USpin=%7.4f DSpin=%7.4f\n",
	       i,i,
	       WhatSpecies[i],
	       InitN_USpin[i],
	       InitN_DSpin[i]);
      }
    }

    ungetc('\n',fp);

    if (!input_last("LeftLeadAtoms.SpeciesAndCoordinates>")) {
      /* format error */

      if (myid==Host_ID){
        printf("Format error for LeftLeadAtoms.SpeciesAndCoordinates\n");
      }
      MPI_Finalize();
      exit(1);
    }
  }

  /* center */
  if (fp=input_find("<Atoms.SpeciesAndCoordinates") ) {
    for (i=1; i<=Catomnum; i++){

      fgets(buf,MAXBUF,fp);
        /* spin non-collinear */
        if (SpinP_switch==3){

          sscanf(buf,"%i %s %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %s",
                 &j, Species,
                 &Gxyz[Latomnum+i][1],&Gxyz[Latomnum+i][2],&Gxyz[Latomnum+i][3],
                 &InitN_USpin[Latomnum+i],&InitN_DSpin[Latomnum+i],
                 &Angle0_Spin[Latomnum+i], &Angle1_Spin[Latomnum+i],
                 &Angle0_Orbital[Latomnum+i], &Angle1_Orbital[Latomnum+i],
                 &Constraint_SpinAngle[Latomnum+i],
                 OrbPol );

          if (fabs(Angle0_Spin[Latomnum+i])<1.0e-10){
            Angle0_Spin[Latomnum+i] = Angle0_Spin[Latomnum+i] + rnd(1.0e-5);
          }

          Angle0_Spin[Latomnum+i] = PI*Angle0_Spin[Latomnum+i]/180.0;
          Angle1_Spin[Latomnum+i] = PI*Angle1_Spin[Latomnum+i]/180.0;
          InitAngle0_Spin[Latomnum+i] = Angle0_Spin[Latomnum+i];
          InitAngle1_Spin[Latomnum+i] = Angle1_Spin[Latomnum+i];

          if (fabs(Angle0_Orbital[Latomnum+i])<1.0e-10){
            Angle0_Orbital[Latomnum+i] = Angle0_Orbital[Latomnum+i] + rnd(1.0e-5);
          }

          Angle0_Orbital[Latomnum+i] = PI*Angle0_Orbital[Latomnum+i]/180.0;
          Angle1_Orbital[Latomnum+i] = PI*Angle1_Orbital[Latomnum+i]/180.0;
          InitAngle0_Orbital[Latomnum+i] = Angle0_Orbital[Latomnum+i];
          InitAngle1_Orbital[Latomnum+i] = Angle1_Orbital[Latomnum+i];

      /*check whether the Euler angle measured from the direction (1,0) is used*/
      if ( (InitN_USpin[Latomnum+i]-InitN_DSpin[Latomnum+i]) < 0.0){

            mx = -sin(InitAngle0_Spin[Latomnum+i])*cos(InitAngle1_Spin[Latomnum+i]);
            my = -sin(InitAngle0_Spin[Latomnum+i])*sin(InitAngle1_Spin[Latomnum+i]);
            mz = -cos(InitAngle0_Spin[Latomnum+i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

            Angle0_Spin[Latomnum+i] = S_coordinate[1];
            Angle1_Spin[Latomnum+i] = S_coordinate[2];
            InitAngle0_Spin[Latomnum+i] = Angle0_Spin[Latomnum+i];
            InitAngle1_Spin[Latomnum+i] = Angle1_Spin[Latomnum+i];

            tmp = InitN_USpin[Latomnum+i];
            InitN_USpin[Latomnum+i] = InitN_DSpin[Latomnum+i];
            InitN_DSpin[Latomnum+i] = tmp;

            mx = -sin(InitAngle0_Orbital[Latomnum+i])*cos(InitAngle1_Orbital[Latomnum+i]);
            my = -sin(InitAngle0_Orbital[Latomnum+i])*sin(InitAngle1_Orbital[Latomnum+i]);
            mz = -cos(InitAngle0_Orbital[Latomnum+i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

            Angle0_Orbital[Latomnum+i] = S_coordinate[1];
            Angle1_Orbital[Latomnum+i] = S_coordinate[2];
            InitAngle0_Orbital[Latomnum+i] = Angle0_Orbital[Latomnum+i];
            InitAngle1_Orbital[Latomnum+i] = Angle1_Orbital[Latomnum+i];

      }  /* if ( (InitN_USpin[i]-InitN_DSpin[i]) < 0.0)  */
     } /* if (SpinP_switch == 3) */

     /* spin collinear */
     else {

      sscanf(buf,"%i %s %lf %lf %lf %lf %lf %s",
             &j,Species,
	     &Gxyz[Latomnum+i][1],
             &Gxyz[Latomnum+i][2],
             &Gxyz[Latomnum+i][3],
	     &InitN_USpin[Latomnum+i],
             &InitN_DSpin[Latomnum+i], 
             OrbPol);

     }

      WhatSpecies[Latomnum+i] = Species2int(Species);
      TRAN_region[Latomnum+i]= 1;
      TRAN_Original_Id[Latomnum+i]= j;

      if (Hub_U_switch==1) OrbPol_flag[Latomnum+i] = OrbPol2int(OrbPol);

      if (i!=j){

        if (myid==Host_ID){
	  printf("Error of sequential number %i in <Atoms.SpeciesAndCoordinates\n",j);
        }
        MPI_Finalize();
        exit(1);
      }

      if (2<=level_stdout && myid==Host_ID){
	printf("<Input_std>  ct_AN=%2d WhatSpecies=%2d USpin=%7.4f DSpin=%7.4f\n",
	       Latomnum+i,WhatSpecies[Latomnum+i],InitN_USpin[Latomnum+i],InitN_DSpin[Latomnum+i]);
      }
    }

    ungetc('\n',fp);

    if (!input_last("Atoms.SpeciesAndCoordinates>")) {
      /* format error */
      if (myid==Host_ID){
        printf("Format error for Atoms.SpeciesAndCoordinates\n");
      }
      MPI_Finalize();
      exit(1);
 
    }
  }

  /* right */

  if (fp=input_find("<RightLeadAtoms.SpeciesAndCoordinates") ) {

    for (i=1; i<=Ratomnum; i++){

      fgets(buf,MAXBUF,fp);
        /* spin non-collinear */
        if (SpinP_switch==3){

          sscanf(buf,"%i %s %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %s",
                 &j, Species,
                 &Gxyz[Catomnum+Latomnum+i][1],&Gxyz[Catomnum+Latomnum+i][2],&Gxyz[Catomnum+Latomnum+i][3],
                 &InitN_USpin[Catomnum+Latomnum+i],&InitN_DSpin[Catomnum+Latomnum+i],
                 &Angle0_Spin[Catomnum+Latomnum+i], &Angle1_Spin[Catomnum+Latomnum+i],
                 &Angle0_Orbital[Catomnum+Latomnum+i], &Angle1_Orbital[Catomnum+Latomnum+i],
                 &Constraint_SpinAngle[Catomnum+Latomnum+i],
                 OrbPol );

          if (fabs(Angle0_Spin[Catomnum+Latomnum+i])<1.0e-10){
            Angle0_Spin[Catomnum+Latomnum+i] = Angle0_Spin[Catomnum+Latomnum+i] + rnd(1.0e-5);
          }

          Angle0_Spin[Catomnum+Latomnum+i] = PI*Angle0_Spin[Catomnum+Latomnum+i]/180.0;
          Angle1_Spin[Catomnum+Latomnum+i] = PI*Angle1_Spin[Catomnum+Latomnum+i]/180.0;
          InitAngle0_Spin[Catomnum+Latomnum+i] = Angle0_Spin[Catomnum+Latomnum+i];
          InitAngle1_Spin[Catomnum+Latomnum+i] = Angle1_Spin[Catomnum+Latomnum+i];

          if (fabs(Angle0_Orbital[Catomnum+Latomnum+i])<1.0e-10){
            Angle0_Orbital[Catomnum+Latomnum+i] = Angle0_Orbital[Catomnum+Latomnum+i] + rnd(1.0e-5);
          }

          Angle0_Orbital[Catomnum+Latomnum+i] = PI*Angle0_Orbital[Catomnum+Latomnum+i]/180.0;
          Angle1_Orbital[Catomnum+Latomnum+i] = PI*Angle1_Orbital[Catomnum+Latomnum+i]/180.0;
          InitAngle0_Orbital[Catomnum+Latomnum+i] = Angle0_Orbital[Catomnum+Latomnum+i];
          InitAngle1_Orbital[Catomnum+Latomnum+i] = Angle1_Orbital[Catomnum+Latomnum+i];

      /*check whether the Euler angle measured from the direction (1,0) is used*/
      if ( (InitN_USpin[Catomnum+Latomnum+i]-InitN_DSpin[Catomnum+Latomnum+i]) < 0.0){

            mx = -sin(InitAngle0_Spin[Catomnum+Latomnum+i])*cos(InitAngle1_Spin[Catomnum+Latomnum+i]);
            my = -sin(InitAngle0_Spin[Catomnum+Latomnum+i])*sin(InitAngle1_Spin[Catomnum+Latomnum+i]);
            mz = -cos(InitAngle0_Spin[Catomnum+Latomnum+i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

            Angle0_Spin[Catomnum+Latomnum+i] = S_coordinate[1];
            Angle1_Spin[Catomnum+Latomnum+i] = S_coordinate[2];
            InitAngle0_Spin[Catomnum+Latomnum+i] = Angle0_Spin[Catomnum+Latomnum+i];
            InitAngle1_Spin[Catomnum+Latomnum+i] = Angle1_Spin[Catomnum+Latomnum+i];

            tmp = InitN_USpin[Catomnum+Latomnum+i];
            InitN_USpin[Catomnum+Latomnum+i] = InitN_DSpin[Catomnum+Latomnum+i];
            InitN_DSpin[Catomnum+Latomnum+i] = tmp;

            mx = -sin(InitAngle0_Orbital[Catomnum+Latomnum+i])*cos(InitAngle1_Orbital[Catomnum+Latomnum+i]);
            my = -sin(InitAngle0_Orbital[Catomnum+Latomnum+i])*sin(InitAngle1_Orbital[Catomnum+Latomnum+i]);
            mz = -cos(InitAngle0_Orbital[Catomnum+Latomnum+i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

            Angle0_Orbital[Catomnum+Latomnum+i] = S_coordinate[1];
            Angle1_Orbital[Catomnum+Latomnum+i] = S_coordinate[2];
            InitAngle0_Orbital[Catomnum+Latomnum+i] = Angle0_Orbital[Catomnum+Latomnum+i];
            InitAngle1_Orbital[Catomnum+Latomnum+i] = Angle1_Orbital[Catomnum+Latomnum+i];

      }  /* if ( (InitN_USpin[i]-InitN_DSpin[i]) < 0.0)  */
     } /* if (SpinP_switch == 3) */

     /* spin collinear */
     else {
      sscanf(buf,"%i %s %lf %lf %lf %lf %lf %s",
             &j,Species,
	     &Gxyz[Catomnum+Latomnum+i][1],
	     &Gxyz[Catomnum+Latomnum+i][2],
	     &Gxyz[Catomnum+Latomnum+i][3],
	     &InitN_USpin[Catomnum+Latomnum+i],
	     &InitN_DSpin[Catomnum+Latomnum+i], OrbPol);

     }

      WhatSpecies[Catomnum+Latomnum+i] = Species2int(Species);

      TRAN_region[Catomnum+Latomnum+i]= 3;
      TRAN_Original_Id[Catomnum+Latomnum+i]= j;

      if (Hub_U_switch==1) OrbPol_flag[Catomnum+Latomnum+i] = OrbPol2int(OrbPol);

      if (i!=j){
        if (myid==Host_ID){
 	  printf("Error of sequential number %i in <RightLeadAtoms.SpeciesAndCoordinates\n",j);
        }
        MPI_Finalize();
        exit(1);
      }

      if (2<=level_stdout && myid==Host_ID){
	printf("<Input_std> R_AN=%2d T_AN=%2d WhatSpecies=%2d USpin=%7.4f DSpin=%7.4f\n",
	       i,Catomnum+Latomnum+i,
	       WhatSpecies[Catomnum+Latomnum+i],
	       InitN_USpin[Catomnum+Latomnum+i],
	       InitN_DSpin[Catomnum+Latomnum+i]);
      }
    }

    ungetc('\n',fp);

    if (!input_last("RightLeadAtoms.SpeciesAndCoordinates>")) {
      /* format error */

      if (myid==Host_ID){
        printf("Format error for RightLeadAtoms.SpeciesAndCoordinates\n");
      }
      MPI_Finalize();
      exit(1);

    }
  }

  if (coordinates_unit==0){
    for (i=1; i<=atomnum; i++){
      Gxyz[i][1] = Gxyz[i][1]/BohrR;
      Gxyz[i][2] = Gxyz[i][2]/BohrR;
      Gxyz[i][3] = Gxyz[i][3]/BohrR;
    }
  }

  /* compare the coordinates with those used for the band calculation of the left lead */

    for (i=1; i<=Latomnum; i++){
      printf("ABC1 X i=%2d %15.12f %15.12f %15.12f %15.12f\n",i,Gxyz_e[0][i][1],Gxyz[i][1],Gxyz[1][1],Gxyz_e[0][1][1]);
      printf("ABC1 Y i=%2d %15.12f %15.12f %15.12f %15.12f\n",i,Gxyz_e[0][i][2],Gxyz[i][2],Gxyz[1][2],Gxyz_e[0][1][2]);
      printf("ABC1 Z i=%2d %15.12f %15.12f %15.12f %15.12f\n",i,Gxyz_e[0][i][3],Gxyz[i][3],Gxyz[1][3],Gxyz_e[0][1][3]);
    }

    MPI_Finalize();
    exit(1);

  for (i=1; i<=Latomnum; i++){

    if (    1.0e-12<fabs(Gxyz_e[0][i][1]-Gxyz[i][1]+Gxyz[1][1]-Gxyz_e[0][1][1])
	 || 1.0e-12<fabs(Gxyz_e[0][i][2]-Gxyz[i][2]+Gxyz[1][2]-Gxyz_e[0][1][2])
	 || 1.0e-12<fabs(Gxyz_e[0][i][3]-Gxyz[i][3]+Gxyz[1][3]-Gxyz_e[0][1][3]) ){  

      if (myid==Host_ID){
	printf("\n\n");
	printf("The LEFT lead cannot be superposed on the original cell even after the translation.\n");
	printf("Check your atomic coordinates of the LEFT lead.\n\n");
      }

      MPI_Finalize();
      exit(1);
    }
  }

  /* compare the coordinates with those used for the band calculation of the right lead */

  for (i=1; i<=Ratomnum; i++){

    if (    1.0e-12<fabs((Gxyz_e[1][i][1]+(Gxyz[Catomnum+Latomnum+1][1]-Gxyz_e[1][1][1]))-Gxyz[Catomnum+Latomnum+i][1])
         || 1.0e-12<fabs((Gxyz_e[1][i][2]+(Gxyz[Catomnum+Latomnum+1][2]-Gxyz_e[1][1][2]))-Gxyz[Catomnum+Latomnum+i][2])
         || 1.0e-12<fabs((Gxyz_e[1][i][3]+(Gxyz[Catomnum+Latomnum+1][3]-Gxyz_e[1][1][3]))-Gxyz[Catomnum+Latomnum+i][3]) ){  

      if (myid==Host_ID){
	printf("\n\n");
	printf("The RIGHT lead cannot be superposed on the original cell even after the translation.\n");
	printf("Check your atomic coordinates of the RIGHT lead.\n\n");
      }

      MPI_Finalize();
      exit(1);
    }

  }

  /****************************************************
                          Unit cell
  ****************************************************/

  for (i=1; i<=3; i++){

    Left_tv[i][1]  = tv_e[0][i][1];
    Left_tv[i][2]  = tv_e[0][i][2];
    Left_tv[i][3]  = tv_e[0][i][3];

    Right_tv[i][1] = tv_e[1][i][1];
    Right_tv[i][2] = tv_e[1][i][2];
    Right_tv[i][3] = tv_e[1][i][3];
  }

  for (i=2; i<=3; i++){
    tv[i][1]  = tv_e[0][i][1];
    tv[i][2]  = tv_e[0][i][2];
    tv[i][3]  = tv_e[0][i][3];
  }

  /*****************************************************
  set the grid origin and the unit vectors for the NEGF
  calculations so that the boundaries between leads and 
  the extended central region match with those in the 
  band calculations for the leads. 
  *****************************************************/

  GO_XL = Gxyz[1][1] - Gxyz_e[0][1][1] + Grid_Origin_e[0][1];
  GO_YL = Gxyz[1][2] - Gxyz_e[0][1][2] + Grid_Origin_e[0][2];
  GO_ZL = Gxyz[1][3] - Gxyz_e[0][1][3] + Grid_Origin_e[0][3];

  GO_XR = Gxyz[Catomnum+Latomnum+1][1] - Gxyz_e[1][1][1] + Grid_Origin_e[1][1];
  GO_YR = Gxyz[Catomnum+Latomnum+1][2] - Gxyz_e[1][1][2] + Grid_Origin_e[1][2];
  GO_ZR = Gxyz[Catomnum+Latomnum+1][3] - Gxyz_e[1][1][3] + Grid_Origin_e[1][3];

  /* large cell = left lead + central region + right lead */    

  tv[idim][1] = GO_XR + Right_tv[idim][1] - GO_XL;
  tv[idim][2] = GO_YR + Right_tv[idim][2] - GO_YL;
  tv[idim][3] = GO_ZR + Right_tv[idim][3] - GO_ZL;

  scf_fixed_origin[0] = GO_XL;
  scf_fixed_origin[1] = GO_YL;
  scf_fixed_origin[2] = GO_ZL;

  /*
  if (myid==0){

  printf("Right_tv\n");
  printf("%15.12f %15.12f %15.12f\n",Right_tv[1][1],Right_tv[1][2],Right_tv[1][3]);  

  printf("GO_XR=%15.12f GO_XL=%15.12f\n",GO_XR,GO_XL);
  printf("GO_YR=%15.12f GO_YL=%15.12f\n",GO_YR,GO_YL);
  printf("GO_ZR=%15.12f GO_ZL=%15.12f\n",GO_ZR,GO_ZL);

  printf("tv\n");
  printf("%15.12f %15.12f %15.12f\n",tv[1][1],tv[1][2],tv[1][3]);  
  printf("%15.12f %15.12f %15.12f\n",tv[2][1],tv[2][2],tv[2][3]);  
  printf("%15.12f %15.12f %15.12f\n",tv[3][1],tv[3][2],tv[3][3]);  

  }

  MPI_Finalize();
  exit(0);
  */


}
