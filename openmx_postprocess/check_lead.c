#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Inputtools.h"

#define asize10  100
#define BohrR    0.529177249
#define print_std   0

static void Input_NEGF(char *argv[]);
static void SpeciesString2int(int p);
static int Species2int(char Species[asize10]);
static int SEQ(char str1[asize10], char str2[asize10]);
static void fnjoint2(char name1[asize10], char name2[asize10],
                     char name3[asize10], char name4[asize10]);
static int Find_Size_LeftLeads();
static int Find_Size_RightLeads();
static void Make_new_inputfile(char *argv[], int NL, int NR);
static void FreeArrays();

static int atomnum,Catomnum;
static int Latomnum,Ratomnum;
static int SpeciesNum;
static int coordinates_unit;
static int unitvector_unit;
static int Solver,Lbuffer,Rbuffer;

static int *C_WhatSpecies;
static int *L_WhatSpecies;
static int *R_WhatSpecies;
static int **Spe_Num_Basis;
static int **Spe_Num_CBasis;
static int *Spe_MaxL_Basis;

static double tv[4][4];
static double Left_tv[4][4];
static double Right_tv[4][4];
static double **Cxyz;
static double **Lxyz;
static double **Rxyz;
static double **Cdens;
static double **Ldens;
static double **Rdens;
static double *Spe_Atom_Cut1;

static char **SpeName; 
static char **SpeBasis;
static char **SpeBasisName;
static char **SpeVPS;

int main(int argc, char *argv[]) 
{    
  static int NL,NR;

  Input_NEGF(argv);  
  NL = Find_Size_LeftLeads();
  NR = Find_Size_RightLeads();
  Make_new_inputfile(argv,NL,NR); 
  FreeArrays();
}    

void Make_new_inputfile(char *argv[], int NL, int NR)
{

  FILE *fp,*fp1,*fp2;
  char *sp0,*sp1,*sp2;
  char *sp3,*sp4,*sp5;
  char fname2[asize10];
  char command0[3*asize10];
  static int po,po1,i,num,cell,spe;
  static double x,y,z;

  sprintf(fname2,"%s_NEGF1",argv[1]);

  printf("\n");
  printf("The new input file, %s, was generated.\n",fname2);
  printf("\n");

  if ((fp2 = fopen(fname2,"w")) != NULL){
    if ((fp = fopen(argv[1],"r")) != NULL){

      po = 0;
      do {
	if (fgets(command0,asize10,fp)!=NULL){
	  command0[strlen(command0)-1] = '\0';
           
          sp0 = strstr(command0,"LeftLeadAtoms.Number");
          sp1 = strstr(command0,"<LeftLeadAtoms.SpeciesAndCoordinates");
          sp2 = strstr(command0,"<LeftLeadAtoms.UnitVectors");
          sp3 = strstr(command0,"RightLeadAtoms.Number");
          sp4 = strstr(command0,"<RightLeadAtoms.SpeciesAndCoordinates");
          sp5 = strstr(command0,"<RightLeadAtoms.UnitVectors");

          /****************************************************
                         LeftLeadAtoms.Number
          ****************************************************/

	  if (sp0!=NULL){
            fprintf(fp2,"LeftLeadAtoms.Number         %i\n",NL*Latomnum); 
          }

          /****************************************************
                   LeftLeadAtoms.SpeciesAndCoordinates
          ****************************************************/

	  else if (sp1!=NULL){
            fprintf(fp2,"%s\n",command0); 

            for (i=0; i<=Latomnum; i++){
              fgets(command0,asize10,fp);
              command0[strlen(command0)-1] = '\0';
	    }

            num = 0;
            for (cell=0; cell<NL; cell++){
              for (i=1; i<=Latomnum; i++){

                spe = L_WhatSpecies[i];

                x = Lxyz[i][1] + (double)cell*Left_tv[3][1];                 
                y = Lxyz[i][2] + (double)cell*Left_tv[3][2];                 
                z = Lxyz[i][3] + (double)cell*Left_tv[3][3];                 
                num++;               

               if (coordinates_unit==0){
                 fprintf(fp2," %4d %3s  %15.12f %15.12f %15.12f   %6.3f  %6.3f\n",
                         num,SpeName[spe],x*BohrR,y*BohrR,z*BohrR,
                         Ldens[i][0],Ldens[i][1]); 
	       }
	       else{
                 fprintf(fp2," %4d %3s  %15.12f %15.12f %15.12f   %6.3f  %6.3f\n",
                         num,SpeName[spe],x,y,z,Ldens[i][0],Ldens[i][1]);
               }

	      }
	    }
            fprintf(fp2,"LeftLeadAtoms.SpeciesAndCoordinates>\n");

	  }

          /****************************************************
                          LeftLeadAtoms.UnitVectors
          ****************************************************/

	  else if (sp2!=NULL){
            fprintf(fp2,"%s\n",command0);

            for (i=1; i<=2; i++){
              fgets(command0,asize10,fp);
              command0[strlen(command0)-1] = '\0';
              fprintf(fp2,"%s\n",command0);
	    }
            for (i=1; i<=2; i++){
              fgets(command0,asize10,fp);
              command0[strlen(command0)-1] = '\0';
	    }

            if (coordinates_unit==0){
              fprintf(fp2,"  %15.12f %15.12f %15.12f\n",
                      (double)NL*Left_tv[3][1]*BohrR,
                      (double)NL*Left_tv[3][2]*BohrR,
                      (double)NL*Left_tv[3][3]*BohrR);
	    }
            else {
              fprintf(fp2,"  %15.12f %15.12f %15.12f\n",
                      (double)NL*Left_tv[3][1],
                      (double)NL*Left_tv[3][2],
                      (double)NL*Left_tv[3][3]);
            }

            fprintf(fp2,"LeftLeadAtoms.UnitVectors>\n");
	  }

          /****************************************************
                           RightLeadAtoms.Number
          ****************************************************/

	  else if (sp3!=NULL){
            fprintf(fp2,"RightLeadAtoms.Number         %i\n",NR*Ratomnum); 
          }

          /****************************************************
                   RightLeadAtoms.SpeciesAndCoordinates
          ****************************************************/

	  else if (sp4!=NULL){
            fprintf(fp2,"%s\n",command0); 

            for (i=0; i<=Ratomnum; i++){
              fgets(command0,asize10,fp);
              command0[strlen(command0)-1] = '\0';
	    }

            num = 0;
            for (cell=0; cell<NR; cell++){
              for (i=1; i<=Ratomnum; i++){

                spe = R_WhatSpecies[i];

                x = Rxyz[i][1] + (double)cell*Right_tv[3][1];                 
                y = Rxyz[i][2] + (double)cell*Right_tv[3][2];                 
                z = Rxyz[i][3] + (double)cell*Right_tv[3][3];                 
                num++;               

               if (coordinates_unit==0){
                 fprintf(fp2," %4d %3s  %15.12f %15.12f %15.12f   %6.3f  %6.3f\n",
                         num,SpeName[spe],x*BohrR,y*BohrR,z*BohrR,Rdens[i][0],Rdens[i][1]); 
	       }
	       else{
                 fprintf(fp2," %4d %3s  %15.12f %15.12f %15.12f   %6.3f  %6.3f\n",
                         num,SpeName[spe],x,y,z,Rdens[i][0],Rdens[i][1]);
               }

	      }
	    }
            fprintf(fp2,"RightLeadAtoms.SpeciesAndCoordinates>\n");

	  }

          /****************************************************
                          RightLeadAtoms.UnitVectors
          ****************************************************/


	  else if (sp5!=NULL){

            fprintf(fp2,"%s\n",command0);

            for (i=1; i<=2; i++){
              fgets(command0,asize10,fp);
              command0[strlen(command0)-1] = '\0';
              fprintf(fp2,"%s\n",command0);
	    }
            for (i=1; i<=2; i++){
              fgets(command0,asize10,fp);
              command0[strlen(command0)-1] = '\0';
	    }

            if (coordinates_unit==0){
              fprintf(fp2,"  %15.12f %15.12f %15.12f\n",
                      (double)NR*Right_tv[3][1]*BohrR,
                      (double)NR*Right_tv[3][2]*BohrR,
                      (double)NR*Right_tv[3][3]*BohrR);
	    }
            else {
              fprintf(fp2,"  %15.12f %15.12f %15.12f\n",
                      (double)NR*Right_tv[3][1],
                      (double)NR*Right_tv[3][2],
                      (double)NR*Right_tv[3][3]);
            }

            fprintf(fp2,"RightLeadAtoms.UnitVectors>\n");
	  }

          /****************************************************
                                  Others
          ****************************************************/

          else{ 
            fprintf(fp2,"%s\n",command0); 
	  }  
         
	} 
        else{
	  po = 1;  
        }
      }while(po==0);

      fclose(fp);
    }
    else{
      printf("Could not find %s\n",argv[1]);
      exit(0);
    }

    fclose(fp2);
  }
  else{
    printf("Could not open %s\n",fname2);
    exit(0);
  }

} 

int Find_Size_LeftLeads()
{
  
  static int cell,i,j,po,spe1,spe2;
  static int Num_LCcell,Num_LLcell,Num_Lcell;
  static int TFNAN_p,TFNAN_c;
  static double dx,dy,dz;
  static double R,r1,r2;

  /****************************************************
     interations between central and left regions
  ****************************************************/

  po = 0;
  Num_LCcell = 0;  
  TFNAN_c = 0;

  do{ 

    Num_LCcell++;

    TFNAN_p = TFNAN_c;
    TFNAN_c = 0;

    for (i=1; i<=Catomnum; i++){
      spe1 = C_WhatSpecies[i];       
      r1 = Spe_Atom_Cut1[spe1];

      for (cell=0; cell<Num_LCcell; cell++){
        for (j=1; j<=Latomnum; j++){

          spe2 = L_WhatSpecies[j];
          r2 = Spe_Atom_Cut1[spe2];
 
          dx = Cxyz[i][1] - Lxyz[j][1] - (double)cell*Left_tv[3][1];
          dy = Cxyz[i][2] - Lxyz[j][2] - (double)cell*Left_tv[3][2];
          dz = Cxyz[i][3] - Lxyz[j][3] - (double)cell*Left_tv[3][3];
          R = sqrt(dx*dx + dy*dy + dz*dz);
          if ( R<(r1+r2) ) TFNAN_c++;
        }
      }
    }

    if ((TFNAN_c-TFNAN_p)==0) po = 1;

  }while(po==0);   

  Num_LCcell--;

  /****************************************************
       interations between left0 and left regions
  ****************************************************/

  po = 0;
  Num_LLcell = 0;  
  TFNAN_c = 0;

  do{ 

    Num_LLcell++;

    TFNAN_p = TFNAN_c;
    TFNAN_c = 0;

    for (i=1; i<=Latomnum; i++){
      spe1 = L_WhatSpecies[i];       
      r1 = Spe_Atom_Cut1[spe1];

      for (cell=0; cell<Num_LLcell; cell++){
        for (j=1; j<=Latomnum; j++){
          spe2 = L_WhatSpecies[j];
          r2 = Spe_Atom_Cut1[spe2];

          dx = Lxyz[i][1] - Lxyz[j][1] - (double)cell*Left_tv[3][1];
          dy = Lxyz[i][2] - Lxyz[j][2] - (double)cell*Left_tv[3][2];
          dz = Lxyz[i][3] - Lxyz[j][3] - (double)cell*Left_tv[3][3];
          R = sqrt(dx*dx + dy*dy + dz*dz);
          if ( R<(r1+r2) ) TFNAN_c++;
	}
      }
    }

    if ((TFNAN_c-TFNAN_p)==0) po = 1;

  }while(po==0);   

  Num_LLcell = Num_LLcell - 2;

  if (Num_LLcell<Num_LCcell) Num_Lcell = Num_LCcell;
  else                       Num_Lcell = Num_LLcell;

  printf("\n");
  printf("************  Left lead  ************\n");
  printf("  Minimum cell size of the left lead\n");
  printf("    for the central region       = %i\n",Num_LCcell);
  printf("    in the left lead             = %i\n",Num_LLcell);
  printf("  Requested buffer size          = %i\n",Lbuffer);
  printf("  Size of an adjusted left lead  = %i\n",Num_Lcell+Lbuffer);

  Num_Lcell = Num_Lcell + Lbuffer;

  return Num_Lcell;
}


int Find_Size_RightLeads()
{
  
  static int cell,i,j,po,spe1,spe2;
  static int Num_RCcell,Num_RRcell,Num_Rcell;
  static int TFNAN_p,TFNAN_c;
  static double dx,dy,dz;
  static double R,r1,r2;

  /****************************************************
     interations between central and right regions
  ****************************************************/

  po = 0;
  Num_RCcell = 0;  
  TFNAN_c = 0;

  do{ 

    Num_RCcell++;

    TFNAN_p = TFNAN_c;
    TFNAN_c = 0;

    for (i=1; i<=Catomnum; i++){
      spe1 = C_WhatSpecies[i];       
      r1 = Spe_Atom_Cut1[spe1];

      for (cell=0; cell<Num_RCcell; cell++){
        for (j=1; j<=Ratomnum; j++){

          spe2 = R_WhatSpecies[j];
          r2 = Spe_Atom_Cut1[spe2];
 
          dx = Cxyz[i][1] - Rxyz[j][1] - (double)cell*Right_tv[3][1];
          dy = Cxyz[i][2] - Rxyz[j][2] - (double)cell*Right_tv[3][2];
          dz = Cxyz[i][3] - Rxyz[j][3] - (double)cell*Right_tv[3][3];
          R = sqrt(dx*dx + dy*dy + dz*dz);
          if ( R<(r1+r2) ) TFNAN_c++;
        }
      }
    }

    if ((TFNAN_c-TFNAN_p)==0) po = 1;

  }while(po==0);   

  Num_RCcell--;

  /****************************************************
     interations between right0 and right regions
  ****************************************************/

  po = 0;
  Num_RRcell = 0;  
  TFNAN_c = 0;

  do{ 

    Num_RRcell++;

    TFNAN_p = TFNAN_c;
    TFNAN_c = 0;

    for (i=1; i<=Ratomnum; i++){
      spe1 = R_WhatSpecies[i];       
      r1 = Spe_Atom_Cut1[spe1];

      for (cell=0; cell<Num_RRcell; cell++){
        for (j=1; j<=Ratomnum; j++){
          spe2 = R_WhatSpecies[j];
          r2 = Spe_Atom_Cut1[spe2];

          dx = Rxyz[i][1] - Rxyz[j][1] - (double)cell*Right_tv[3][1];
          dy = Rxyz[i][2] - Rxyz[j][2] - (double)cell*Right_tv[3][2];
          dz = Rxyz[i][3] - Rxyz[j][3] - (double)cell*Right_tv[3][3];
          R = sqrt(dx*dx + dy*dy + dz*dz);
          if ( R<(r1+r2) ) TFNAN_c++;
	}
      }
    }

    if ((TFNAN_c-TFNAN_p)==0) po = 1;

  }while(po==0);   

  Num_RRcell = Num_RRcell - 2;

  if (Num_RRcell<Num_RCcell) Num_Rcell = Num_RCcell;
  else                       Num_Rcell = Num_RRcell;

  printf("\n");
  printf("************  Right lead  ************\n");
  printf("  Minimum cell size of the right lead\n");
  printf("    for the central region       = %i\n",Num_RCcell);
  printf("    in the right lead            = %i\n",Num_RRcell);
  printf("  Requested buffer size          = %i\n",Rbuffer);
  printf("  Size of an adjusted left lead  = %i\n",Num_Rcell+Rbuffer);

  Num_Rcell = Num_Rcell + Rbuffer;

  return Num_Rcell;
}




void Input_NEGF(char *argv[])
{
  FILE *fp;
  int i,j,ct_AN,spe;
  int po=0; /* error count */
  double r_vec[20],r_vec2[20];
  int i_vec[20],i_vec2[20];
  char *s_vec[20];
  char Species[20];
  char c; 
  double tmp0,tmp1;
  char DirPAO[asize10];
  char ExtPAO[asize10] = ".pao";
  char FN_PAO[asize10];

  printf("\n*******************************************\n");
  printf("  This program adjusts the size of leads.  \n");
  printf("*******************************************\n\n");

  /****************************************************
                       open a file
  ****************************************************/
  
  input_open(argv[1]);

  /****************************************************
   set DirPAO and DirVPS 
  ****************************************************/

  sprintf(DirPAO,"%s/PAO/",DFT_DATA_PATH);
  
  /****************************************************
                Number of Atomic Species
  ****************************************************/

  input_int("Species.Number",&SpeciesNum,0);

  if (SpeciesNum<=0){
    printf("Species.Number may be wrong.\n");
    po++;
  }

  /****************************************************
      allocation of arrays:

      char SpeName[SpeciesNum][asize10]
      char SpeBasis[SpeciesNum][asize10]
      char SpeBasisName[SpeciesNum][asize10]
      char SpeVPS[SpeciesNum][asize10]
      int Spe_Num_Basis[SpeciesNum][6];
      int Spe_Num_CBasis[SpeciesNum][6];
      int Spe_MaxL_Basis[SpeciesNum];
  ****************************************************/

  SpeName = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeName[i] = (char*)malloc(sizeof(long)*asize10);
  }

  SpeBasis = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeBasis[i] = (char*)malloc(sizeof(char)*asize10);
  }

  SpeBasisName = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeBasisName[i] = (char*)malloc(sizeof(char)*asize10);
  }

  SpeVPS = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeVPS[i] = (char*)malloc(sizeof(char)*asize10);
  }

  Spe_Num_Basis = (int**)malloc(sizeof(int*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    Spe_Num_Basis[i] = (int*)malloc(sizeof(int)*6);
  }

  Spe_Num_CBasis = (int**)malloc(sizeof(int*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    Spe_Num_CBasis[i] = (int*)malloc(sizeof(int)*6);
  }

  Spe_MaxL_Basis = (int*)malloc(sizeof(int)*SpeciesNum);
  Spe_Atom_Cut1 = (double*)malloc(sizeof(double)*SpeciesNum);

  /****************************************************
              definition of Atomic Species
  ****************************************************/

  if (fp=input_find("<Definition.of.Atomic.Species")) {
    for (i=0; i<SpeciesNum; i++){
      fscanf(fp,"%s %s %s",SpeName[i],SpeBasis[i],SpeVPS[i]);
      SpeciesString2int(i);
    }
    if (! input_last("Definition.of.Atomic.Species>")) {
      /* format error */
      printf("Format error for Definition.of.Atomic.Species\n");
      po++;
    }
  }

  if (print_std==1){ 
    for (i=0; i<SpeciesNum; i++){
      printf("<Input_std>  %i Name  %s\n",i,SpeName[i]);
      printf("<Input_std>  %i Basis %s\n",i,SpeBasis[i]);
      printf("<Input_std>  %i VPS   %s\n",i,SpeVPS[i]);
    }
  }

  /****************************************************
                      eigensolver 
  ****************************************************/

  s_vec[0]="Recursion";     s_vec[1]="Cluster"; s_vec[2]="Band";
  s_vec[3]="NEGF";          s_vec[4]="DC";      s_vec[5]="GDC";
  s_vec[6]="Cluster-DIIS";  s_vec[7]="Krylov";  s_vec[8]="Cluster2";  
  s_vec[9]="EGAC";          s_vec[10]="DC-LNO";   s_vec[11]="Cluster-LNO";
  
  i_vec[0]=1;  i_vec[1]=2;   i_vec[2 ]=3;
  i_vec[3]=4;  i_vec[4]=5;   i_vec[5 ]=6;
  i_vec[6]=7;  i_vec[7]=8;   i_vec[8 ]=9;
  i_vec[9]=10; i_vec[10]=11; i_vec[11]=12;

  input_string2int("scf.EigenvalueSolver", &Solver, 12, s_vec,i_vec);

  /****************************************************
                         Atoms
  ****************************************************/

  /* except for NEGF */

  if (Solver!=4){  
    printf("Plese set NEGF for scf.EigenvalueSolve\n");
    exit(0);
  } /* if (Solver!=4){  */

  /*  NEGF */

  else{

    /* center */
    input_int("Atoms.Number",&Catomnum,0);
    if (Catomnum<=0){
      printf("Atoms.Number may be wrong.\n");
      po++;
    }

    /* left */
    input_int("LeftLeadAtoms.Number",&Latomnum,0);
    if (Latomnum<=0){
      printf("LeftLeadAtoms.Number may be wrong.\n");
      po++;
    }

    /* right */
    input_int("RightLeadAtoms.Number",&Ratomnum,0);
    if (Ratomnum<=0){
      printf("RightLeadAtoms.Number may be wrong.\n");
      po++;
    }
    
    atomnum = Catomnum + Latomnum + Ratomnum;

    /****************************************************
        allocation of arrays:

        int C_WhatSpecies[Catomnum+1];
        int L_WhatSpecies[Latomnum+1];
        int R_WhatSpecies[Ratomnum+1];
        double Cxyz[Catomnum+1][4]; 
        double Lxyz[Latomnum+1][4]; 
        double Rxyz[Ratomnum+1][4]; 
        double Cdens[Catomnum+1][2]; 
        double Ldens[Latomnum+1][2]; 
        double Rdens[Ratomnum+1][2]; 
    ****************************************************/

    C_WhatSpecies = (int*)malloc(sizeof(int)*(Catomnum+1));
    L_WhatSpecies = (int*)malloc(sizeof(int)*(Latomnum+1));
    R_WhatSpecies = (int*)malloc(sizeof(int)*(Ratomnum+1));

    Cxyz = (double**)malloc(sizeof(double*)*(Catomnum+1));
    for (ct_AN=0; ct_AN<=Catomnum; ct_AN++){
      Cxyz[ct_AN] = (double*)malloc(sizeof(double)*4);
    }

    Lxyz = (double**)malloc(sizeof(double*)*(Latomnum+1));
    for (ct_AN=0; ct_AN<=Latomnum; ct_AN++){
      Lxyz[ct_AN] = (double*)malloc(sizeof(double)*4);
    }

    Rxyz = (double**)malloc(sizeof(double*)*(Ratomnum+1));
    for (ct_AN=0; ct_AN<=Ratomnum; ct_AN++){
      Rxyz[ct_AN] = (double*)malloc(sizeof(double)*4);
    }

    Cdens = (double**)malloc(sizeof(double*)*(Catomnum+1));
    for (ct_AN=0; ct_AN<=Catomnum; ct_AN++){
      Cdens[ct_AN] = (double*)malloc(sizeof(double)*2);
    }

    Ldens = (double**)malloc(sizeof(double*)*(Latomnum+1));
    for (ct_AN=0; ct_AN<=Latomnum; ct_AN++){
      Ldens[ct_AN] = (double*)malloc(sizeof(double)*2);
    }

    Rdens = (double**)malloc(sizeof(double*)*(Ratomnum+1));
    for (ct_AN=0; ct_AN<=Ratomnum; ct_AN++){
      Rdens[ct_AN] = (double*)malloc(sizeof(double)*2);
    }

    /****************************************************
                      coordinates unit
    ****************************************************/

    s_vec[0]="Ang";  s_vec[1]="AU";
    i_vec[0]= 0;     i_vec[1]=1;
    input_string2int("Atoms.SpeciesAndCoordinates.Unit",
                     &coordinates_unit,2,s_vec,i_vec);

    /****************************************************
                  coordinates of center
    ****************************************************/

    if (fp=input_find("<Atoms.SpeciesAndCoordinates") ) {
      for (i=1; i<=Catomnum; i++){
        fscanf(fp,"%i %s %lf %lf %lf %lf %lf",&j,&Species,
	       &Cxyz[i][1],
               &Cxyz[i][2],
               &Cxyz[i][3],
               &Cdens[i][0],
               &Cdens[i][1]);

        C_WhatSpecies[i] = Species2int(Species);

        if (i!=j){
          printf("Error of sequential number %i in <Atoms.SpeciesAndCoordinates\n",j);
          po++;
        }

        if (print_std==1){ 
          printf("<Input_std>  ct_AN=%2d C_WhatSpecies=%2d\n",i,C_WhatSpecies[i]);
	}
      }

      if (!input_last("Atoms.SpeciesAndCoordinates>")) {
        /* format error */
        printf("Format error for Atoms.SpeciesAndCoordinates\n");
        po++;
      }
    }

    /****************************************************
                     coordinates of left
    ****************************************************/

    if (fp=input_find("<LeftLeadAtoms.SpeciesAndCoordinates") ) {
      for (i=1; i<=Latomnum; i++){
        fscanf(fp,"%i %s %lf %lf %lf %lf %lf",&j,&Species,
	       &Lxyz[i][1],
               &Lxyz[i][2],
               &Lxyz[i][3],
               &Ldens[i][0],
               &Ldens[i][1]);

        L_WhatSpecies[i] = Species2int(Species);

        if (i!=j){
          printf("Error of sequential number %i in <LeftLeadAtoms.SpeciesAndCoordinates\n",j);
          po++;
        }

        if (print_std==1){ 
          printf("<Input_std> L_AN=%2d L_WhatSpecies=%2d\n",
                 i,L_WhatSpecies[i]);
	}
        
      }

      if (!input_last("LeftLeadAtoms.SpeciesAndCoordinates>")) {
        /* format error */
        printf("Format error for LeftLeadAtoms.SpeciesAndCoordinates\n");
        po++;
      }
    }

    /****************************************************
                    coordinates of right
    ****************************************************/

    if (fp=input_find("<RightLeadAtoms.SpeciesAndCoordinates") ) {
      for (i=1; i<=Ratomnum; i++){
        fscanf(fp,"%i %s %lf %lf %lf %lf %lf",&j,&Species,
	       &Rxyz[i][1],
               &Rxyz[i][2],
               &Rxyz[i][3],
               &Rdens[i][0],
               &Rdens[i][1]);

        R_WhatSpecies[i] = Species2int(Species);

        if (i!=j){
          printf("Error of sequential number %i in <RightLeadAtoms.SpeciesAndCoordinates\n",j);
          po++;
        }

        if (print_std==1){ 
          printf("<Input_std> R_AN=%2d R_WhatSpecies=%2d\n",
                 i,R_WhatSpecies[i]);
	}

      }

      if (!input_last("RightLeadAtoms.SpeciesAndCoordinates>")) {
        /* format error */
        printf("Format error for RightLeadAtoms.SpeciesAndCoordinates\n");
        po++;
      }
    }

    if (coordinates_unit==0){
      for (i=1; i<=Catomnum; i++){
        Cxyz[i][1] = Cxyz[i][1]/BohrR;
        Cxyz[i][2] = Cxyz[i][2]/BohrR;
        Cxyz[i][3] = Cxyz[i][3]/BohrR;
      }

      for (i=1; i<=Latomnum; i++){
        Lxyz[i][1] = Lxyz[i][1]/BohrR;
        Lxyz[i][2] = Lxyz[i][2]/BohrR;
        Lxyz[i][3] = Lxyz[i][3]/BohrR;
      }

      for (i=1; i<=Latomnum; i++){
        Rxyz[i][1] = Rxyz[i][1]/BohrR;
        Rxyz[i][2] = Rxyz[i][2]/BohrR;
        Rxyz[i][3] = Rxyz[i][3]/BohrR;
      }
    }

    /****************************************************
                          Unit cell
    ****************************************************/
    
    s_vec[0]="Ang"; s_vec[1]="AU";
    i_vec[1]=0;  i_vec[1]=1;
    input_string2int("Atoms.UnitVectors.Unit",&unitvector_unit,2,s_vec,i_vec);

    /* center */
    if (fp=input_find("<Atoms.Unitvectors")) {
      for (i=1; i<=3; i++){
        fscanf(fp,"%lf %lf %lf",&tv[i][1],&tv[i][2],&tv[i][3]);
      }
      if ( ! input_last("Atoms.Unitvectors>") ) {
        /* format error */
        printf("Format error for Atoms.Unitvectors\n");
        po++;
      }
    }

    /* left */
    if (fp=input_find("<LeftLeadAtoms.Unitvectors")) {
      for (i=1; i<=3; i++){
        fscanf(fp,"%lf %lf %lf",&Left_tv[i][1],&Left_tv[i][2],&Left_tv[i][3]);
      }
      if ( ! input_last("LeftLeadAtoms.Unitvectors>") ) {
        /* format error */
        printf("Format error for LeftLeadAtoms.Unitvectors\n");
        po++;
      }
    }

    /* right */
    if (fp=input_find("<RightLeadAtoms.Unitvectors")) {
      for (i=1; i<=3; i++){
        fscanf(fp,"%lf %lf %lf",&Right_tv[i][1],&Right_tv[i][2],&Right_tv[i][3]);
      }
      if ( ! input_last("RightLeadAtoms.Unitvectors>") ) {
        /* format error */
        printf("Format error for RightLeadAtoms.Unitvectors\n");
        po++;
      }
    }

    if (unitvector_unit==0){
      for (i=1; i<=3; i++){
        tv[i][1] = tv[i][1]/BohrR;
        tv[i][2] = tv[i][2]/BohrR;
        tv[i][3] = tv[i][3]/BohrR;
      }

      for (i=1; i<=3; i++){
        Left_tv[i][1] = Left_tv[i][1]/BohrR;
        Left_tv[i][2] = Left_tv[i][2]/BohrR;
        Left_tv[i][3] = Left_tv[i][3]/BohrR;
      }

      for (i=1; i<=3; i++){
        Right_tv[i][1] = Right_tv[i][1]/BohrR;
        Right_tv[i][2] = Right_tv[i][2]/BohrR;
        Right_tv[i][3] = Right_tv[i][3]/BohrR;
      }
    }

    /****************************************************
                       buffer size
    ****************************************************/

     input_int("Buffer.Size.LeftLead",  &Lbuffer, 1);
     input_int("Buffer.Size.RightLead", &Rbuffer, 1);

  } /* else{ */ 


  /****************************************************
                       input_close
  ****************************************************/

  input_close();

  if (po>0 || input_errorCount()>0) {
    printf("errors in the inputfile\n");
    exit(1);
  } 

  /****************************************************
                   print out to std
  ****************************************************/

  printf("<Input_NEGF>  Your input file was normally read.\n");
  printf("<Input_NEGF>  The system includes %i species and %i atoms.\n",
          SpeciesNum,atomnum);

  /****************************************************
                   find rcut of pao
  ****************************************************/

  for (spe=0; spe<SpeciesNum; spe++){
    fnjoint2(DirPAO,SpeBasisName[spe],ExtPAO,FN_PAO);
    if ((fp = fopen(FN_PAO,"r")) != NULL){
      input_open(FN_PAO);
      input_double("radial.cutoff.pao",&Spe_Atom_Cut1[spe],(double)0.0);
      input_close();
      fclose(fp);
      printf("<Input_NEGF>  PAOs of species %s were normally found.\n",SpeName[spe]);
      printf("<Input_NEGF>  rcut of PAOs of species %s = %5.3f (Bohr)\n",
              SpeName[spe],Spe_Atom_Cut1[spe]);
    }
    else{
      printf("Could not find %s\n",FN_PAO);
      exit(1);
    }
  }

}

void SpeciesString2int(int p)
{
  static int i,l,n,po;
  char c,cstr[asize10];

  /* Get basis name */

  i = 0;
  po = 0;
  while((c=SpeBasis[p][i])!='\0'){
    if (c=='-') po = 1;
    if (po==0) SpeBasisName[p][i] = SpeBasis[p][i]; 
    i++;
  }

  if (print_std==1){ 
    printf("<Input_std>  SpeBasisName=%s\n",SpeBasisName[p]);
  }

  /* Get basis type */

  for (l=0; l<5; l++){
    Spe_Num_Basis[p][l] = 0;
  }

  i = 0;
  po = 0;

  while((c=SpeBasis[p][i])!='\0'){
    if (po==1){
      if      (c=='s'){ l=0; n=0; }
      else if (c=='p'){ l=1; n=0; }
      else if (c=='d'){ l=2; n=0; }
      else if (c=='f'){ l=3; n=0; }
      else{

        if (n==0){
          cstr[0] = c;
          cstr[1] = '\0';
          Spe_Num_Basis[p][l]  = atoi(cstr);
          Spe_Num_CBasis[p][l] = atoi(cstr);
          n++;
        }
        else if (n==1){
          cstr[0] = c;
          cstr[1] = '\0';
          Spe_Num_CBasis[p][l] = atoi(cstr);   
          if (Spe_Num_Basis[p][l]<Spe_Num_CBasis[p][l]){
            printf("# of contracted orbitals are larger than # of primitive oribitals\n");
            exit(1); 
          } 

          n++;
        }
        else {
          printf("Format error in Definition of Atomic Species\n");
          exit(1);
	}
      } 
    }  

    if (SpeBasis[p][i]=='-') po = 1;
    i++;
  }

  for (l=0; l<5; l++){
    if (Spe_Num_Basis[p][l]!=0) Spe_MaxL_Basis[p] = l;

    if (print_std==1){ 
      printf("<Input_std>  p=%2d l=%2d %2d %2d\n",
              p,l,Spe_Num_Basis[p][l],Spe_Num_CBasis[p][l]);
    }
  }
}

int Species2int(char Species[asize10])
{
  static int i,po;

  i = 0;
  po = 0; 
  while (i<SpeciesNum && po==0){
    if (SEQ(Species,SpeName[i])==1){
      po = 1;
    }
    if (po==0) i++;
  };

  if (po==1) return i;
  else {
    printf("%s is an invalid species name in Atoms.SpeciesAndCoordinates\n",
            Species);
    printf("Please check your input file\n");
    exit(1);
  }
}


static int SEQ(char str1[asize10],
               char str2[asize10])
{
  
  int i,result,l1,l2;

  l1 = strlen(str1);
  l2 = strlen(str2);

  result = 1; 
  if (l1 == l2){
    for (i=0; i<=l1-1; i++){
      if (str1[i]!=str2[i])  result = 0;   
    }
  }
  else
    result = 0; 

  return result;
}


void fnjoint2(char name1[asize10], char name2[asize10],
              char name3[asize10], char name4[asize10])
{
  char *f1 = name1,
       *f2 = name2,
       *f3 = name3,
       *f4 = name4;

  while(*f1)
    {
      *f4 = *f1;
      *f1++;
      *f4++;
    }
  while(*f2)
    {
      *f4 = *f2;
      *f2++;
      *f4++;
    }
  while(*f3)
    {
      *f4 = *f3;
      *f3++;
      *f4++;
    }
  *f4 = *f3;
}

void FreeArrays()
{
  static int i,ct_AN,spe; 

  free(C_WhatSpecies);
  free(L_WhatSpecies);
  free(R_WhatSpecies);

  for (i=0; i<SpeciesNum; i++){
    free(Spe_Num_Basis[i]);
  }
  free(Spe_Num_Basis);

  for (i=0; i<SpeciesNum; i++){
    free(Spe_Num_CBasis[i]);
  }
  free(Spe_Num_CBasis);

  free(Spe_MaxL_Basis);

  for (ct_AN=0; ct_AN<=Catomnum; ct_AN++){
    free(Cxyz[ct_AN]);
  }
  free(Cxyz);

  for (ct_AN=0; ct_AN<=Latomnum; ct_AN++){
    free(Lxyz[ct_AN]);
  }
  free(Lxyz);

  for (ct_AN=0; ct_AN<=Ratomnum; ct_AN++){
    free(Rxyz[ct_AN]);
  }
  free(Rxyz);

  for (ct_AN=0; ct_AN<=Catomnum; ct_AN++){
    free(Cdens[ct_AN]);
  }
  free(Cdens);

  for (ct_AN=0; ct_AN<=Latomnum; ct_AN++){
    free(Ldens[ct_AN]);
  }
  free(Ldens);

  for (ct_AN=0; ct_AN<=Ratomnum; ct_AN++){
    free(Rdens[ct_AN]);
  }
  free(Rdens);

  free(Spe_Atom_Cut1);

  for (i=0; i<SpeciesNum; i++){
    free(SpeName[i]);
  }
  free(SpeName);

  for (i=0; i<SpeciesNum; i++){
    free(SpeBasis[i]);
  }
  free(SpeBasis);

  for (i=0; i<SpeciesNum; i++){
    free(SpeBasisName[i]);
  }
  free(SpeBasisName);

  for (i=0; i<SpeciesNum; i++){
    free(SpeVPS[i]);
  }
  free(SpeVPS);

}




