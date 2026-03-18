#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>

#include "Inputtools.h"
#include "read_scfout.h"
#include "Tools_BandCalc.h"


void SpeciesString2int(int p, char **SpeName, char **SpeBasis, int **Spe_Num_Basis);
static int SEQ(char str1[asize10], char str2[asize10]);


void Classify_OrbNum(int **ClaOrb, char **An2Spe, int Print_AtomInfo){
  // ### FILE ################
  FILE *fp, *fp1;

  /* Disabled by N. Yamaguchi ***
  char fnm[10], fname_atominfo[256];
  * ***/
  /* Added by N. Yamaguchi ***/
  char fnm[21], fname_atominfo[256];
  /* ***/

  char c;
  char buf[256];
  int i,j,k,l, i1,i2,i3;
  int po = 0;
  double d0,d1,d2,d3;

  // ### ATOMIC INFO #########
  int TNO_MAX, *WhatSpecies;
  int SpeciesNum, **Spe_Num_Basis;
  char **SpeName, **SpeBasis, SpeAtom[asize10];
  char *Name_Angular[6][9];

  // ### ORBITAL INFO #########
  //Name_Angular and Name_Multiple
  Name_Angular[0][0] = "s          ";
  Name_Angular[1][0] = "px         ";
  Name_Angular[1][1] = "py         ";
  Name_Angular[1][2] = "pz         ";
  Name_Angular[2][0] = "d3z^2-r^2  ";
  Name_Angular[2][1] = "dx^2-y^2   ";
  Name_Angular[2][2] = "dxy        ";
  Name_Angular[2][3] = "dxz        ";
  Name_Angular[2][4] = "dyz        ";
  Name_Angular[3][0] = "f5z^2-3r^2 ";
  Name_Angular[3][1] = "f5xz^2-xr^2";
  Name_Angular[3][2] = "f5yz^2-yr^2";
  Name_Angular[3][3] = "fzx^2-zy^2 ";
  Name_Angular[3][4] = "fxyz       ";
  Name_Angular[3][5] = "fx^3-3*xy^2";
  Name_Angular[3][6] = "f3yx^2-y^3 ";
  Name_Angular[4][0] = "g1         ";
  Name_Angular[4][1] = "g2         ";
  Name_Angular[4][2] = "g3         ";
  Name_Angular[4][3] = "g4         ";
  Name_Angular[4][4] = "g5         ";
  Name_Angular[4][5] = "g6         ";
  Name_Angular[4][6] = "g7         ";
  Name_Angular[4][7] = "g8         ";
  Name_Angular[4][8] = "g9         ";

  // ### Open *.dat file ###

  /* Disabled by N. Yamaguchi
   * strcpy(fnm,"input.dat");
   */

  /* Added by N. Yamaguchi ***/
  strcpy(fnm,"temporal_12345.input");
  /* ***/

  input_open(fnm);


  // ### Malloc ###
  input_int("Species.Number",&SpeciesNum,0);
  SpeName = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeName[i] = (char*)malloc(sizeof(char)*(asize10));
  }
  SpeBasis = (char**)malloc(sizeof(char*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    SpeBasis[i] = (char*)malloc(sizeof(char)*20);
  }
  Spe_Num_Basis = (int**)malloc(sizeof(int*)*SpeciesNum);
  for (i=0; i<SpeciesNum; i++){
    Spe_Num_Basis[i] = (int*)malloc(sizeof(int)*6);
  }

  // ### Decide Species Info ###
  if ((fp=input_find("<Definition.of.Atomic.Species")) != NULL) {
    for (i=0; i<SpeciesNum; i++){
      fscanf(fp,"%s %s %s",SpeName[i],SpeBasis[i],buf);
      SpeciesString2int(i, SpeName, SpeBasis, Spe_Num_Basis);
    }
  }
  // ### Decide Atoms Info ###
  WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));

  if ((fp=input_find("<Atoms.SpeciesAndCoordinates")) != NULL) {
    for (i=1; i<=atomnum; i++){
      fscanf(fp,"%d %s",&j,SpeAtom);
      strcpy(An2Spe[i],SpeAtom);
      while((c=fgetc(fp))!='\n'){};
      j=0;    po=0;
      while (j<SpeciesNum && po==0){
	if (SEQ(SpeAtom,SpeName[j])==1)po = 1;
	if (po==0) j++;
      }
      if (po==1) { WhatSpecies[i] = j; }
      //else if(po==0){ if (Print_AtomInfo==1) printf("error\n"); }
    }//i
  }//if
  input_close();

  // ### PRINT PART ###
  if (Print_AtomInfo==1) {
    strcpy(fname_atominfo, fname_out);
    strcat(fname_atominfo, ".atominfo");
    fp1 = fopen(fname_atominfo, "w");

    fprintf(fp1, "############ CELL INFO ####################\n");
    fprintf(fp1," %10.6lf %10.6lf %10.6lf \n", rtv[1][1], rtv[2][1], rtv[3][1] );
    fprintf(fp1," %10.6lf %10.6lf %10.6lf \n", rtv[1][2], rtv[2][2], rtv[3][2] );
    fprintf(fp1," %10.6lf %10.6lf %10.6lf \n", rtv[1][3], rtv[2][3], rtv[3][3] );
    fprintf(fp1, "  \n"); 
    fprintf(fp1, "############ ATOM INFO ####################\n");
    fprintf(fp1, "          s    p    d    f    -       \n");
    for (i=0; i<SpeciesNum; i++){
      fprintf(fp1, "     %2s",SpeName[i]);
      for (j=0; j<5; j++)  fprintf(fp1, "%4d ",Spe_Num_Basis[i][j]);
      fprintf(fp1, "\n");
    }//i
    fprintf(fp1, "###                                     ###\n");
  }//if
  for (i=1; i<=atomnum; i++){
    if (Print_AtomInfo==1) fprintf(fp1, "%4d %s ",i,SpeName[WhatSpecies[i]]);
    if (Print_AtomInfo==1) fprintf(fp1, "(%10.6lf %10.6lf %10.6lf) \n", BohrR*Gxyz[i][1], BohrR*Gxyz[i][2], BohrR*Gxyz[i][3]);
    j=0;
    if (Print_AtomInfo==1) fprintf(fp1, "       (Orbital + Multiple) -> Total\n");
    for (i1=0; i1<4; i1++){
      for (i2=0; i2<Spe_Num_Basis[WhatSpecies[i]][i1]; i2++){
	for (i3=0; i3<(2*i1+1); i3++){
	  ClaOrb[i][j] = (i1*i1)+i3;
	  if (Print_AtomInfo==1) fprintf(fp1, "      %s ( %d + %d ) ->%4d\n", Name_Angular[i1][i3], ((i1*i1)+i3), (i2+1), j);
	  j++;
	}//for(i3)
      }//for(i2)
    }//for(i1)
  }//i
  if (Print_AtomInfo==1){
    fprintf(fp1, "###########################################\n\n");
    fclose(fp1);
  }

  // ### Malloc Free ###
  for (i=0; i<SpeciesNum; i++){
    free(SpeName[i]);
  } free(SpeName);
  for (i=0; i<SpeciesNum; i++){
    free(SpeBasis[i]);
  } free(SpeBasis);
  for (i=0; i<SpeciesNum; i++){
    free(Spe_Num_Basis[i]);
  } free(Spe_Num_Basis);
  free(WhatSpecies);
}



void SpeciesString2int(int p, char **SpeName, char **SpeBasis, int **Spe_Num_Basis){
  static int i,l,n,po;
  char c,cstr[4];

  // Get basis name 
  for (l=0; l<5; l++){
    Spe_Num_Basis[p][l] = 0;
  }
  i = 0;  po = 0;

  while((c=SpeBasis[p][i])!='\0'){
    if(po == 1){
      if      (c=='s'){ l=0; n=0; }
      else if (c=='p'){ l=1; n=0; }
      else if (c=='d'){ l=2; n=0; }
      else if (c=='f'){ l=3; n=0; }
      else{
	if (n==0){
	  cstr[0] = c;
	  cstr[1] = '\0';
	  Spe_Num_Basis[p][l] = atoi(cstr);
	  n++;
	}else {
	  exit(1);
	}
      }
    } 
    if (SpeBasis[p][i]=='-') po = 1;
    i++;
  }
}


static int SEQ(char str1[asize10], char str2[asize10]){

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


