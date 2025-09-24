/****************************************************************
 USAGE:

   ./pdb2pao *.pdb P > *.paoinp

   or 

   ./pdb2pao *.pdb O > *.paoinp

****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define asize1  100
#define asize2  100
#define asize3   20

void Set_Names();   
void readfile();
void pdb2pao(char pao[asize1],
             char res[asize1], char atom[asize1]);
void strcpy2(char names[2][asize2],
             char pdb[asize2], char pao[asize2]);
char *substr( char *buffer, const char *string,
              int nStart, int nStop );

char Name_Residues[asize1][asize2];
char NameAtoms[30][asize1][2][asize2];

int Num_Atoms[asize1];

main(int argc, char *argv[]) 
{
  Set_Names();   
  readfile(argv);
}


void readfile(char *argv[])
{

  static int po,po1,num,cres,rnum,i,num_pao;
  char st[300],buffer[asize1];
  char buffer0[asize1],buffer1[asize1];
  char res[asize1],res1[asize1];
  char atom[asize1],pao[asize1];
  char x[asize1],y[asize1],z[asize1];
  char list_pao[100][asize1];
  FILE *fp;

  /* find the number of residues */

  if ((fp=fopen(argv[1],"r"))!=NULL){

    while (fgets(st, 300, fp) != NULL){
      if (strcmp(substr(buffer, st, 0, 4), "ATOM")==0){
        substr(buffer, st, 23, 26);
        rnum = atoi(buffer);  
      }
    }
    fclose(fp);
  }
  else {
    printf("could not open.\n");
  }

  /*
  printf("rnum=%i\n",rnum);
  */

  /* read and convert */

  num_pao = 0;
  
  if ((fp=fopen(argv[1],"r"))!=NULL){

    printf("<Atoms.SpeciesAndCoordinates\n");

    po = 0;
    num = 0;

    while (fgets(st, 300, fp) != NULL){

      substr(buffer, st, 0, 4);

      if (strcmp(buffer, "END\n")==0){
	po = 1;
      }
      else if (strcmp(buffer, "ATOM")==0){
	num++;
	substr(res,  st, 17, 20);
	substr(atom, st, 13, 16);
	pdb2pao(pao,res,atom);
	substr(x,  st, 31, 38);
	substr(y,  st, 39, 46);
	substr(z,  st, 47, 54);

	substr(buffer0, st, 23, 26);
	cres = atoi(buffer0);  

	/* N-terminus */
	if (cres==1 && strcmp(atom,"H  ")==0){
	  strcpy(pao, "HNA");          
	}
	else if (cres==1 && strcmp(atom,"N  ")==0){
	  strcpy(pao, "NA");
	}

	/* C-terminus */
	else if (cres==rnum && strcmp(atom,"O  ")==0){
	  strcpy(pao, "OC");
	}
	else if (cres==rnum && strcmp(atom,"C  ")==0){
	  strcpy(pao, "CC");
	}

	/*
	  printf("cres=%i rnum=%i atom=%s\n",cres,rnum,atom);  
	*/

	/* initial charge */

	substr(buffer,  pao, 0, 1);

	if (strcmp(buffer, "C")==0){
	  if (strcmp(argv[2], "P")==0) 
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,buffer,atof(x),atof(y),atof(z),2.0,2.0); 
	  else
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,pao,atof(x),atof(y),atof(z),2.0,2.0); 
	}
	else if (strcmp(buffer, "N")==0){
	  if (strcmp(argv[2], "P")==0) 
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,buffer,atof(x),atof(y),atof(z),2.5,2.5); 
	  else
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,pao,atof(x),atof(y),atof(z),2.5,2.5); 
	}
	else if (strcmp(buffer, "O")==0){
	  if (strcmp(argv[2], "P")==0) 
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,buffer,atof(x),atof(y),atof(z),3.0,3.0); 
	  else 
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,pao,atof(x),atof(y),atof(z),3.0,3.0); 
	}
	else if (strcmp(buffer, "S")==0){
	  if (strcmp(argv[2], "P")==0) 
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,buffer,atof(x),atof(y),atof(z),3.0,3.0); 
	  else
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,pao,atof(x),atof(y),atof(z),3.0,3.0); 
	}
	else if (strcmp(buffer, "H")==0){
	  if (strcmp(argv[2], "P")==0) 
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,buffer,atof(x),atof(y),atof(z),0.5,0.5); 
	  else
	    printf(" %4d   %3s  %10.5f  %10.5f  %10.5f    %3.2f %3.2f\n",
		   num,pao,atof(x),atof(y),atof(z),0.5,0.5); 
	}


	i = 0;
	po1 = 0; 
	do {
	  if (strcmp(list_pao[i], pao)==0){
	    po1 = 1;
	  }
	  else{
	    i++;
	  }
	} while(po1==0 && i<num_pao);         

	if (po1==0){
	  num_pao++;
	  strcpy(list_pao[num_pao-1], pao);
	}

      }
    }

    fclose(fp);
    printf("Atoms.SpeciesAndCoordinates>\n");

    strcpy(buffer0, "-s2p1");
    strcpy(buffer1, "-s2p2d1");

    printf("\n\n");
    printf("<Definition.of.Atomic.Species\n");
    for (i=0; i<num_pao; i++){
      substr(buffer,  list_pao[i], 0, 1);
      if (strcmp(buffer, "H")==0){
        printf("   %3s  %3s%5s     H_TM\n",list_pao[i],list_pao[i],buffer0); 
      }
    }
    for (i=0; i<num_pao; i++){
      substr(buffer,  list_pao[i], 0, 1);
      if (strcmp(buffer, "C")==0){
        printf("   %3s  %3s%7s    C_TM_PCC\n",list_pao[i],list_pao[i],buffer1); 
      }
    }
    for (i=0; i<num_pao; i++){
      substr(buffer,  list_pao[i], 0, 1);
      if (strcmp(buffer, "N")==0){
        printf("   %3s  %3s%7s    N_TM_PCC\n",list_pao[i],list_pao[i],buffer1); 
      }
    }
    for (i=0; i<num_pao; i++){
      substr(buffer,  list_pao[i], 0, 1);
      if (strcmp(buffer, "O")==0){
        printf("   %3s  %3s%7s    O_TM_PCC\n",list_pao[i],list_pao[i],buffer1); 
      }
    }
    for (i=0; i<num_pao; i++){
      substr(buffer,  list_pao[i], 0, 1);
      if (strcmp(buffer, "S")==0){
        printf("   %3s  %3s%7s    S_TM_PCC\n",list_pao[i],list_pao[i],buffer1); 
      }
    }
    printf("Definition.of.Atomic.Species>\n");

  }
  else {
    printf("could not open.\n");
  }
}

void pdb2pao(char pao[asize1], char res[asize1], char atom[asize1])
{

  static int po,i,ir,ia; 

  po = 0;
  i = 0;
  do {
    if (strcmp(res, Name_Residues[i])==0){
      po = 1;
      ir = i;
    }
    else {
      i++;
    }
  } while (po==0 && i<asize3);

  po = 0;
  i = 0;
  do {
    if (strcmp(atom, NameAtoms[ir][i][0])==0){
      po = 1;
      ia = i; 
    }
    else{
      i++;
    }  
  } while (po==0 && i<Num_Atoms[ir]);


  strcpy(pao, NameAtoms[ir][i][1]);
}


char *substr( char *buffer, const char *string, int nStart, int nStop )
{
  memmove( buffer, string + nStart, nStop - nStart );
  buffer[nStop-nStart] = '\0';

  return buffer;
}


void Set_Names()
{

  /*******************
    Glycine, Gly, G
  *******************/

  Num_Atoms[0] = 7;
  strcpy(Name_Residues[0], "GLY");

  strcpy2(NameAtoms[0][ 0], "CA ","CA" );
  strcpy2(NameAtoms[0][ 1], "C  ","C"  );
  strcpy2(NameAtoms[0][ 2], "N  ","N"  );
  strcpy2(NameAtoms[0][ 3], "O  ","O"  );
  strcpy2(NameAtoms[0][ 4], "HA ","HA" );
  strcpy2(NameAtoms[0][ 5], "H  ","HN" );
  strcpy2(NameAtoms[0][ 6], "OXT","OC" ); /* C-terminus */

  /*******************
    Alanine, Ala, A
  *******************/

  Num_Atoms[1] = 9;
  strcpy(Name_Residues[1], "ALA");

  strcpy2(NameAtoms[1][ 0], "CA ","CA" );
  strcpy2(NameAtoms[1][ 1], "C  ","C"  );
  strcpy2(NameAtoms[1][ 2], "CB ","C1" );
  strcpy2(NameAtoms[1][ 3], "N  ","N"  );
  strcpy2(NameAtoms[1][ 4], "O  ","O"  );
  strcpy2(NameAtoms[1][ 5], "HA ","HA" );
  strcpy2(NameAtoms[1][ 6], "H  ","HN" );
  strcpy2(NameAtoms[1][ 7], "HB ","H1" );
  strcpy2(NameAtoms[1][ 8], "OXT","OC" ); /* C-terminus */

  /*******************
    Valine, Val, V
  *******************/

  Num_Atoms[2] = 13;
  strcpy(Name_Residues[2], "VAL");

  strcpy2(NameAtoms[2][ 0], "CA ","CA" );
  strcpy2(NameAtoms[2][ 1], "C  ","C"  );
  strcpy2(NameAtoms[2][ 2], "CB ","C3" );
  strcpy2(NameAtoms[2][ 3], "CG1","C1" );
  strcpy2(NameAtoms[2][ 4], "CG2","C1" );
  strcpy2(NameAtoms[2][ 5], "N  ","N"  );
  strcpy2(NameAtoms[2][ 6], "O  ","O"  );
  strcpy2(NameAtoms[2][ 7], "HA ","HA" );
  strcpy2(NameAtoms[2][ 8], "H  ","HN" );
  strcpy2(NameAtoms[2][ 9], "HB ","H3" );
  strcpy2(NameAtoms[2][10], "HG1","H1" );
  strcpy2(NameAtoms[2][11], "HG2","H1" );
  strcpy2(NameAtoms[2][12], "OXT","OC" ); /* C-terminus */

  /*******************
    Leucine, Leu, L
  *******************/

  Num_Atoms[3] = 15;
  strcpy(Name_Residues[3], "LEU");

  strcpy2(NameAtoms[3][ 0], "CA ","CA" );
  strcpy2(NameAtoms[3][ 1], "C  ","C"  );
  strcpy2(NameAtoms[3][ 2], "CB ","C2" );
  strcpy2(NameAtoms[3][ 3], "CG ","C3" );
  strcpy2(NameAtoms[3][ 4], "CD1","C1" );
  strcpy2(NameAtoms[3][ 5], "CD2","C1" );
  strcpy2(NameAtoms[3][ 6], "N  ","N"  );
  strcpy2(NameAtoms[3][ 7], "O  ","O"  );
  strcpy2(NameAtoms[3][ 8], "HA ","HA" );
  strcpy2(NameAtoms[3][ 9], "H  ","HN" );
  strcpy2(NameAtoms[3][10], "HB ","H2" );
  strcpy2(NameAtoms[3][11], "HG ","H3" );
  strcpy2(NameAtoms[3][12], "HD1","H1" );
  strcpy2(NameAtoms[3][13], "HD2","H1" );
  strcpy2(NameAtoms[3][14], "OXT","OC" ); /* C-terminus */

  /*******************
   Isoleucine, Ile, I
  *******************/

  Num_Atoms[4] = 15;
  strcpy(Name_Residues[4], "ILE");

  strcpy2(NameAtoms[4][ 0], "CA ","CA" );
  strcpy2(NameAtoms[4][ 1], "C  ","C"  );
  strcpy2(NameAtoms[4][ 2], "CB ","C3" );
  strcpy2(NameAtoms[4][ 3], "CG1","C2" );
  strcpy2(NameAtoms[4][ 4], "CG2","C1" );
  strcpy2(NameAtoms[4][ 5], "CD1","C1" );
  strcpy2(NameAtoms[4][ 6], "N  ","N"  );
  strcpy2(NameAtoms[4][ 7], "O  ","O"  );
  strcpy2(NameAtoms[4][ 8], "HA ","HA" );
  strcpy2(NameAtoms[4][ 9], "H  ","HN" );
  strcpy2(NameAtoms[4][10], "HB ","H3" );
  strcpy2(NameAtoms[4][11], "HG1","H2" );
  strcpy2(NameAtoms[4][12], "HG2","H1" );
  strcpy2(NameAtoms[4][13], "HD1","H1" );
  strcpy2(NameAtoms[4][14], "OXT","OC" ); /* C-terminus */

  /*******************
    Proline, Pro, P
  *******************/

  Num_Atoms[5] = 13;
  strcpy(Name_Residues[5], "PRO");

  strcpy2(NameAtoms[5][ 0], "CA ","CA" );
  strcpy2(NameAtoms[5][ 1], "C  ","C"  );
  strcpy2(NameAtoms[5][ 2], "CB ","C2" );
  strcpy2(NameAtoms[5][ 3], "CG ","C2" );
  strcpy2(NameAtoms[5][ 4], "CD ","CA" );
  strcpy2(NameAtoms[5][ 5], "N  ","N"  );
  strcpy2(NameAtoms[5][ 6], "O  ","O"  );
  strcpy2(NameAtoms[5][ 7], "HA ","HA" );
  strcpy2(NameAtoms[5][ 8], "H  ","HN" );
  strcpy2(NameAtoms[5][ 9], "HB ","H2" );
  strcpy2(NameAtoms[5][10], "HG ","H2" );
  strcpy2(NameAtoms[5][11], "HD ","HA" );
  strcpy2(NameAtoms[5][12], "OXT","OC" ); /* C-terminus */

  /************************
    Phenylalanine, Phe, F
  ************************/

  Num_Atoms[6] = 21;
  strcpy(Name_Residues[6], "PHE");

  strcpy2(NameAtoms[6][ 0], "CA ","CA" );
  strcpy2(NameAtoms[6][ 1], "C  ","C"  );
  strcpy2(NameAtoms[6][ 2], "CB ","C2" );
  strcpy2(NameAtoms[6][ 3], "CG ","CB3");
  strcpy2(NameAtoms[6][ 4], "CD1","CB2");
  strcpy2(NameAtoms[6][ 5], "CD2","CB2");
  strcpy2(NameAtoms[6][ 6], "CE1","CB2");
  strcpy2(NameAtoms[6][ 7], "CE2","CB2");
  strcpy2(NameAtoms[6][ 8], "CZ ","CB2");
  strcpy2(NameAtoms[6][ 9], "N  ","N"  );
  strcpy2(NameAtoms[6][10], "O  ","O"  );
  strcpy2(NameAtoms[6][11], "HA ","HA" );
  strcpy2(NameAtoms[6][12], "H  ","HN" );
  strcpy2(NameAtoms[6][13], "HB ","H2" );
  strcpy2(NameAtoms[6][14], "HG ","H2" );
  strcpy2(NameAtoms[6][15], "HD1","HB" );
  strcpy2(NameAtoms[6][16], "HD2","HB" );
  strcpy2(NameAtoms[6][17], "HE1","HB" );
  strcpy2(NameAtoms[6][18], "HE2","HB" );
  strcpy2(NameAtoms[6][19], "HZ ","HB" );
  strcpy2(NameAtoms[6][20], "OXT","OC" ); /* C-terminus */

  /************************
     Methionine, Met, M
  ************************/

  Num_Atoms[7] = 14;
  strcpy(Name_Residues[7], "MET");

  strcpy2(NameAtoms[7][ 0], "CA ","CA" );
  strcpy2(NameAtoms[7][ 1], "C  ","C"  );
  strcpy2(NameAtoms[7][ 2], "CB ","C2" );
  strcpy2(NameAtoms[7][ 3], "CG ","CS2");
  strcpy2(NameAtoms[7][ 4], "CE ","CS1");
  strcpy2(NameAtoms[7][ 5], "N  ","N"  );
  strcpy2(NameAtoms[7][ 6], "O  ","O"  );
  strcpy2(NameAtoms[7][ 7], "SD ","S2" );
  strcpy2(NameAtoms[7][ 8], "HA ","HA" );
  strcpy2(NameAtoms[7][ 9], "H  ","HN" );
  strcpy2(NameAtoms[7][10], "HB ","H2" );
  strcpy2(NameAtoms[7][11], "HG ","H2" );
  strcpy2(NameAtoms[7][12], "HE ","H1" );
  strcpy2(NameAtoms[7][13], "OXT","OC" ); /* C-terminus */

  /************************
     Tryptophan, Trp, W
  ************************/

  Num_Atoms[8] = 24;
  strcpy(Name_Residues[8], "TRP");

  strcpy2(NameAtoms[8][ 0], "CA ","CA" );
  strcpy2(NameAtoms[8][ 1], "C  ","C"  );
  strcpy2(NameAtoms[8][ 2], "CB ","C2" );
  strcpy2(NameAtoms[8][ 3], "CG ","CB3");
  strcpy2(NameAtoms[8][ 4], "CD1","CW1");
  strcpy2(NameAtoms[8][ 5], "CD2","CB3");
  strcpy2(NameAtoms[8][ 6], "CE2","CW2");
  strcpy2(NameAtoms[8][ 7], "CE3","CB2");
  strcpy2(NameAtoms[8][ 8], "CZ2","CB2");
  strcpy2(NameAtoms[8][ 9], "CZ3","CB2");
  strcpy2(NameAtoms[8][10], "CH2","CB2");
  strcpy2(NameAtoms[8][11], "N  ","N"  );
  strcpy2(NameAtoms[8][12], "NE1","NW" );
  strcpy2(NameAtoms[8][13], "O  ","O"  );
  strcpy2(NameAtoms[8][14], "HA ","HA" );
  strcpy2(NameAtoms[8][15], "H  ","HN" );
  strcpy2(NameAtoms[8][16], "HB ","H2" );
  strcpy2(NameAtoms[8][17], "HD1","HB" );
  strcpy2(NameAtoms[8][18], "HE1","HNW");
  strcpy2(NameAtoms[8][19], "HE3","HB" );
  strcpy2(NameAtoms[8][20], "HZ2","HB" );
  strcpy2(NameAtoms[8][21], "HZ3","HB" );
  strcpy2(NameAtoms[8][22], "HH2","HB" );
  strcpy2(NameAtoms[8][23], "OXT","OC" ); /* C-terminus */

  /************************
      Cysteine, Cys, C
  ************************/

  Num_Atoms[9] = 20;
  strcpy(Name_Residues[9], "CYS");
  
  /************************
       Lysine, Lys, K
  ************************/

  Num_Atoms[10] = 17;
  strcpy(Name_Residues[10], "LYS");

  strcpy2(NameAtoms[10][ 0], "CA ","CA" );
  strcpy2(NameAtoms[10][ 1], "C  ","C"  );
  strcpy2(NameAtoms[10][ 2], "CB ","C2" );
  strcpy2(NameAtoms[10][ 3], "CG ","C2" );
  strcpy2(NameAtoms[10][ 4], "CD ","C2" );
  strcpy2(NameAtoms[10][ 5], "CE ","CA" );
  strcpy2(NameAtoms[10][ 6], "N  ","N"  );
  strcpy2(NameAtoms[10][ 7], "NZ ","NA" );
  strcpy2(NameAtoms[10][ 8], "O  ","O"  );
  strcpy2(NameAtoms[10][ 9], "HA ","HA" );
  strcpy2(NameAtoms[10][10], "H  ","HN" );
  strcpy2(NameAtoms[10][11], "HB ","H2" );
  strcpy2(NameAtoms[10][12], "HG ","H2" );
  strcpy2(NameAtoms[10][13], "HD ","H2" );
  strcpy2(NameAtoms[10][14], "HE ","HA" );
  strcpy2(NameAtoms[10][15], "HZ ","HNA");
  strcpy2(NameAtoms[10][16], "OXT","OC" ); /* C-terminus */

  /************************
      Arginine, Arg, R
  ************************/

  Num_Atoms[11] = 20;
  strcpy(Name_Residues[11], "ARG");

  strcpy2(NameAtoms[11][ 0], "CA ","CA" );
  strcpy2(NameAtoms[11][ 1], "C  ","C"  );
  strcpy2(NameAtoms[11][ 2], "CB ","C2" );
  strcpy2(NameAtoms[11][ 3], "CG ","C2" );
  strcpy2(NameAtoms[11][ 4], "CD ","CA" );
  strcpy2(NameAtoms[11][ 5], "CZ ","CR" );
  strcpy2(NameAtoms[11][ 6], "N  ","N"  );
  strcpy2(NameAtoms[11][ 7], "NE ","N"  );
  strcpy2(NameAtoms[11][ 8], "NH1","NR" );
  strcpy2(NameAtoms[11][ 9], "NH2","NR" );
  strcpy2(NameAtoms[11][10], "O  ","O"  );
  strcpy2(NameAtoms[11][11], "HA ","HA" );
  strcpy2(NameAtoms[11][12], "H  ","HN" );
  strcpy2(NameAtoms[11][13], "HB ","H2" );
  strcpy2(NameAtoms[11][14], "HG ","H2" );
  strcpy2(NameAtoms[11][15], "HD ","HA" );
  strcpy2(NameAtoms[11][16], "HE ","HN" );
  strcpy2(NameAtoms[11][17], "HH1","HR" );
  strcpy2(NameAtoms[11][18], "HH2","HR" );
  strcpy2(NameAtoms[11][19], "OXT","OC" ); /* C-terminus */

  /************************
      Histidine, His, H
  ************************/

  Num_Atoms[12] = 20;
  strcpy(Name_Residues[12], "HIS");

  strcpy2(NameAtoms[12][ 0], "CA ","CA" );
  strcpy2(NameAtoms[12][ 1], "C  ","C"  );
  strcpy2(NameAtoms[12][ 2], "CB ","C2" );
  strcpy2(NameAtoms[12][ 3], "CG ","CH2");
  strcpy2(NameAtoms[12][ 4], "CD2","CW1");
  strcpy2(NameAtoms[12][ 5], "CE1","CH1");
  strcpy2(NameAtoms[12][ 6], "CE1","CH1");
  strcpy2(NameAtoms[12][ 7], "N  ","N"  );
  strcpy2(NameAtoms[12][ 8], "ND1","NW" );
  strcpy2(NameAtoms[12][ 9], "NE2","NW" );
  strcpy2(NameAtoms[12][10], "O  ","O"  );
  strcpy2(NameAtoms[12][11], "HA ","HA" );
  strcpy2(NameAtoms[12][12], "H  ","HN" );
  strcpy2(NameAtoms[12][13], "HB ","H2" );
  strcpy2(NameAtoms[12][14], "HG ","H2" );
  strcpy2(NameAtoms[12][15], "HD1","HNW");
  strcpy2(NameAtoms[12][16], "HD2","HB" );
  strcpy2(NameAtoms[12][17], "HE1","HB" );
  strcpy2(NameAtoms[12][18], "HE2","HNW");
  strcpy2(NameAtoms[12][19], "OXT","OC" ); /* C-terminus */

  /************************
    Aspartic acid, Asp, D
  ************************/

  Num_Atoms[13] = 12;
  strcpy(Name_Residues[13], "ASP");

  strcpy2(NameAtoms[13][ 0], "CA ","CA" );
  strcpy2(NameAtoms[13][ 1], "C  ","C"  );
  strcpy2(NameAtoms[13][ 2], "CB ","C2" );
  strcpy2(NameAtoms[13][ 3], "CG ","CC" );
  strcpy2(NameAtoms[13][ 4], "N  ","N"  );
  strcpy2(NameAtoms[13][ 5], "O  ","O"  );
  strcpy2(NameAtoms[13][ 6], "OD1","OC" );
  strcpy2(NameAtoms[13][ 7], "OD2","OC" );
  strcpy2(NameAtoms[13][ 8], "HA ","HA" );
  strcpy2(NameAtoms[13][ 9], "H  ","HN" );
  strcpy2(NameAtoms[13][10], "HB ","H2" );
  strcpy2(NameAtoms[13][11], "OXT","OC" ); /* C-terminus */

  /************************
    Glutamic acid, Glu, E
  ************************/

  Num_Atoms[14] = 14;
  strcpy(Name_Residues[14], "GLU");

  strcpy2(NameAtoms[14][ 0], "CA ","CA" );
  strcpy2(NameAtoms[14][ 1], "C  ","C"  );
  strcpy2(NameAtoms[14][ 2], "CB ","C2" );
  strcpy2(NameAtoms[14][ 3], "CG ","C2" );
  strcpy2(NameAtoms[14][ 4], "CD ","CC" );
  strcpy2(NameAtoms[14][ 5], "N  ","N"  );
  strcpy2(NameAtoms[14][ 6], "O  ","O"  );
  strcpy2(NameAtoms[14][ 7], "OE1","OC" );
  strcpy2(NameAtoms[14][ 8], "OE2","OC" );
  strcpy2(NameAtoms[14][ 9], "HA ","HA" );
  strcpy2(NameAtoms[14][10], "H  ","HN" );
  strcpy2(NameAtoms[14][11], "HB ","H2" );
  strcpy2(NameAtoms[14][12], "HG ","H2" );
  strcpy2(NameAtoms[14][13], "OXT","OC" ); /* C-terminus */

  /************************
     Asparagine, Asn, N
  ************************/

  Num_Atoms[15] = 13;
  strcpy(Name_Residues[15], "ASN");

  strcpy2(NameAtoms[15][ 0], "CA ","CA" );
  strcpy2(NameAtoms[15][ 1], "C  ","C"  );
  strcpy2(NameAtoms[15][ 2], "CB ","C2" );
  strcpy2(NameAtoms[15][ 3], "CG ","C"  );
  strcpy2(NameAtoms[15][ 4], "N  ","N"  );
  strcpy2(NameAtoms[15][ 5], "ND2","N"  );
  strcpy2(NameAtoms[15][ 6], "O  ","O"  );
  strcpy2(NameAtoms[15][ 7], "OD1","O"  );
  strcpy2(NameAtoms[15][ 8], "HA ","HA" );
  strcpy2(NameAtoms[15][ 9], "H  ","HN" );
  strcpy2(NameAtoms[15][10], "HB ","H2" );
  strcpy2(NameAtoms[15][11], "HD2","HN" );
  strcpy2(NameAtoms[15][12], "OXT","OC" ); /* C-terminus */

  /************************
     Glutamine, Gln, Q
  ************************/

  Num_Atoms[16] = 15;
  strcpy(Name_Residues[16], "GLN");

  strcpy2(NameAtoms[16][ 0], "CA ","CA" );
  strcpy2(NameAtoms[16][ 1], "C  ","C"  );
  strcpy2(NameAtoms[16][ 2], "CB ","C2" );
  strcpy2(NameAtoms[16][ 3], "CG ","C2" );
  strcpy2(NameAtoms[16][ 4], "CD ","C"  );
  strcpy2(NameAtoms[16][ 5], "N  ","N"  );
  strcpy2(NameAtoms[16][ 6], "NE2","N"  );
  strcpy2(NameAtoms[16][ 7], "O  ","O"  );
  strcpy2(NameAtoms[16][ 8], "OE1","O"  );
  strcpy2(NameAtoms[16][ 9], "HA ","HA" );
  strcpy2(NameAtoms[16][10], "H  ","HN" );
  strcpy2(NameAtoms[16][11], "HB ","H2" );
  strcpy2(NameAtoms[16][12], "HG ","H2" );
  strcpy2(NameAtoms[16][13], "HE2","HN" );
  strcpy2(NameAtoms[16][14], "OXT","OC" ); /* C-terminus */

  /************************
       Serine, Ser, S
  ************************/

  Num_Atoms[17] = 11;
  strcpy(Name_Residues[17], "SER");

  strcpy2(NameAtoms[17][ 0], "CA ","CA" );
  strcpy2(NameAtoms[17][ 1], "C  ","C"  );
  strcpy2(NameAtoms[17][ 2], "CB ","C2" );
  strcpy2(NameAtoms[17][ 3], "N  ","N"  );
  strcpy2(NameAtoms[17][ 4], "O  ","O"  );
  strcpy2(NameAtoms[17][ 5], "OG ","OH" );
  strcpy2(NameAtoms[17][ 6], "HA ","HA" );
  strcpy2(NameAtoms[17][ 7], "H  ","HN" );
  strcpy2(NameAtoms[17][ 8], "HB ","H2" );
  strcpy2(NameAtoms[17][ 9], "HG ","HO" );
  strcpy2(NameAtoms[17][10], "OXT","OC" ); /* C-terminus */

  /************************
      Threonine, Thr, T
  ************************/

  Num_Atoms[18] = 13;
  strcpy(Name_Residues[18], "THR");

  strcpy2(NameAtoms[18][ 0], "CA ","CA" );
  strcpy2(NameAtoms[18][ 1], "C  ","C"  );
  strcpy2(NameAtoms[18][ 2], "CB ","COH");
  strcpy2(NameAtoms[18][ 3], "CG2","C1" );
  strcpy2(NameAtoms[18][ 4], "N  ","N"  );
  strcpy2(NameAtoms[18][ 5], "O  ","O"  );
  strcpy2(NameAtoms[18][ 6], "OG1","OH" );
  strcpy2(NameAtoms[18][ 7], "HA ","HA" );
  strcpy2(NameAtoms[18][ 8], "H  ","HN" );
  strcpy2(NameAtoms[18][ 9], "HB ","H2" );
  strcpy2(NameAtoms[18][10], "HG ","HO" );
  strcpy2(NameAtoms[18][11], "HG2","H1" );
  strcpy2(NameAtoms[18][12], "OXT","OC" ); /* C-terminus */

  /************************
      Tyrosine, Tyr, Y
  ************************/

  Num_Atoms[19] = 21;
  strcpy(Name_Residues[19], "TYR");

  strcpy2(NameAtoms[19][ 0], "CA ","CA" );
  strcpy2(NameAtoms[19][ 1], "C  ","C"  );
  strcpy2(NameAtoms[19][ 2], "CB ","C2" );
  strcpy2(NameAtoms[19][ 3], "CG ","CB3");
  strcpy2(NameAtoms[19][ 4], "CD1","CB2");
  strcpy2(NameAtoms[19][ 5], "CD2","CB2");
  strcpy2(NameAtoms[19][ 6], "CE1","CB2");
  strcpy2(NameAtoms[19][ 7], "CE2","CB2");
  strcpy2(NameAtoms[19][ 8], "CZ ","CY" );
  strcpy2(NameAtoms[19][ 9], "N  ","N"  );
  strcpy2(NameAtoms[19][10], "O  ","O"  );
  strcpy2(NameAtoms[19][11], "OH ","OH" );
  strcpy2(NameAtoms[19][12], "HA ","HA" );
  strcpy2(NameAtoms[19][13], "H  ","HN" );
  strcpy2(NameAtoms[19][14], "HB ","H2" );
  strcpy2(NameAtoms[19][15], "HD1","HB" );
  strcpy2(NameAtoms[19][16], "HD2","HB" );
  strcpy2(NameAtoms[19][17], "HE1","HB" );
  strcpy2(NameAtoms[19][18], "HE2","HB" );
  strcpy2(NameAtoms[19][19], "HH ","HO" );
  strcpy2(NameAtoms[19][20], "OXT","OC" ); /* C-terminus */

}  


void strcpy2(char names[2][asize2], char pdb[asize2], char pao[asize2])
{
  strcpy(names[0], pdb);
  strcpy(names[1], pao);
}







