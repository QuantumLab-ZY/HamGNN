/**********************************************************************
  md2axsf.c:

     md2axsf.c is a routine to convert a md file to a axsf file.
     This code follows GNU-GPL.

     Compile: gcc md2axsf.c -lm -o md2axsf

     usage:  ./md2axsf input.md output.axsf

  Log of md2axsf.c:

     March/15/2016  Released by T. Ozaki 
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define strsize  600

char Atom_Symbol[105][4];

void Set_Atom_Symbol();
int Atomic_Number(char species[4]);


int main(int argc, char *argv[]) 
{
  int i,j,k,ns,na;
  int nline,atomnum,numsnapshot;
  FILE *fp1,*fp0;
  char read_str[strsize];
  double tv[4][4],tmp,x,y,z;

  Set_Atom_Symbol();

  /* count the number of lines */

  if ((fp1 = fopen(argv[1],"r")) != NULL){

    nline = 0;
    while (fgets(read_str, strsize, fp1)!=NULL){
      nline++;
    }

    fclose(fp1);
  }  
  else{
    printf("could not find %s\n",argv[1]);
  }

  /* read the number of atoms */

  if ((fp1 = fopen(argv[1],"r")) != NULL){

    fscanf(fp1,"%ld",&atomnum); 
    fclose(fp1);
  }  
  else{
    printf("could not find %s\n",argv[1]);
  }

  numsnapshot = nline/(atomnum+2);

  /* read the file again, and save the corresponding axsf file */

  if ((fp0 = fopen(argv[2],"w")) != NULL){

    fprintf(fp0,"ANIMSTEPS %d\n",numsnapshot);
    fprintf(fp0,"CRYSTAL\n");

    if ((fp1 = fopen(argv[1],"r")) != NULL){

      for (ns=1; ns<=numsnapshot; ns++){

	fscanf(fp1,"%s",read_str);  /* atomnum */         

	fscanf(fp1,"%s",read_str);  /* time= */         
	fscanf(fp1,"%s",read_str);  /* 0.0000 */         
	fscanf(fp1,"%s",read_str);  /* (fs) */         
	fscanf(fp1,"%s",read_str);  /* Energy= */         
	fscanf(fp1,"%s",read_str);  /* -8.20219 */         
	fscanf(fp1,"%s",read_str);  /* (Hartree) */         
	fscanf(fp1,"%s",read_str);  /*  Cell_Vectors= */         

	fscanf(fp1,"%lf",&tv[1][1]); 
	fscanf(fp1,"%lf",&tv[1][2]); 
	fscanf(fp1,"%lf",&tv[1][3]); 

	fscanf(fp1,"%lf",&tv[2][1]); 
	fscanf(fp1,"%lf",&tv[2][2]); 
	fscanf(fp1,"%lf",&tv[2][3]); 

	fscanf(fp1,"%lf",&tv[3][1]); 
	fscanf(fp1,"%lf",&tv[3][2]); 
	fscanf(fp1,"%lf",&tv[3][3]); 

        fprintf(fp0,"PRIMVEC %d\n",ns);

	for (i=1; i<=3; i++){
	  for (j=1; j<=3; j++){
	    fprintf(fp0," %15.7f ",tv[i][j]);
	  }
	  fprintf(fp0,"\n");
	}

        fprintf(fp0,"PRIMCOORD %d\n",ns);
        fprintf(fp0,"  %d %d\n",atomnum,1);

        for (na=1; na<=atomnum; na++){
  	  fscanf(fp1,"%s",read_str); 
  	  fscanf(fp1,"%lf",&x); 
  	  fscanf(fp1,"%lf",&y); 
  	  fscanf(fp1,"%lf",&z); 
 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 
  	  fscanf(fp1,"%lf",&tmp); 

          fprintf(fp0,"  %3d ",Atomic_Number(read_str));
          fprintf(fp0,"%15.7f",x);
          fprintf(fp0,"%15.7f",y);
          fprintf(fp0,"%15.7f\n",z);
        }
      }

      fclose(fp1);
    }  
    else{
      printf("could not find %s\n",argv[1]);
    }

    fclose(fp0);
  }
  else{
    printf("could not find %s\n",argv[2]);
  }

}

int Atomic_Number(char species[4])
{
  int i,po;

  i = 0; 
  po = 0;
  do {
    if (strncmp(species,Atom_Symbol[i],4)==0)
      po = 1;
    else 
      i++; 
  } while (po==0 && i<104);

  return i;

}




void Set_Atom_Symbol()
{
  strcpy(Atom_Symbol[  0], "E");
  strcpy(Atom_Symbol[  1], "H");
  strcpy(Atom_Symbol[  2], "He");
  strcpy(Atom_Symbol[  3], "Li");
  strcpy(Atom_Symbol[  4], "Be");
  strcpy(Atom_Symbol[  5], "B");
  strcpy(Atom_Symbol[  6], "C");
  strcpy(Atom_Symbol[  7], "N");
  strcpy(Atom_Symbol[  8], "O");
  strcpy(Atom_Symbol[  9], "F");
  strcpy(Atom_Symbol[ 10], "Ne");
  strcpy(Atom_Symbol[ 11], "Na");
  strcpy(Atom_Symbol[ 12], "Mg");
  strcpy(Atom_Symbol[ 13], "Al");
  strcpy(Atom_Symbol[ 14], "Si");
  strcpy(Atom_Symbol[ 15], "P");
  strcpy(Atom_Symbol[ 16], "S");
  strcpy(Atom_Symbol[ 17], "Cl");
  strcpy(Atom_Symbol[ 18], "Ar");
  strcpy(Atom_Symbol[ 19], "K");
  strcpy(Atom_Symbol[ 20], "Ca");
  strcpy(Atom_Symbol[ 21], "Sc");
  strcpy(Atom_Symbol[ 22], "Ti");
  strcpy(Atom_Symbol[ 23], "V");
  strcpy(Atom_Symbol[ 24], "Cr");
  strcpy(Atom_Symbol[ 25], "Mn");
  strcpy(Atom_Symbol[ 26], "Fe");
  strcpy(Atom_Symbol[ 27], "Co");
  strcpy(Atom_Symbol[ 28], "Ni");
  strcpy(Atom_Symbol[ 29], "Cu");
  strcpy(Atom_Symbol[ 30], "Zn");
  strcpy(Atom_Symbol[ 31], "Ga");
  strcpy(Atom_Symbol[ 32], "Ge");
  strcpy(Atom_Symbol[ 33], "As");
  strcpy(Atom_Symbol[ 34], "Se");
  strcpy(Atom_Symbol[ 35], "Br");
  strcpy(Atom_Symbol[ 36], "Kr");
  strcpy(Atom_Symbol[ 37], "Rb");
  strcpy(Atom_Symbol[ 38], "Sr");
  strcpy(Atom_Symbol[ 39], "Y");
  strcpy(Atom_Symbol[ 40], "Zr");
  strcpy(Atom_Symbol[ 41], "Nb");
  strcpy(Atom_Symbol[ 42], "Mo");
  strcpy(Atom_Symbol[ 43], "Tc");
  strcpy(Atom_Symbol[ 44], "Ru");
  strcpy(Atom_Symbol[ 45], "Rh");
  strcpy(Atom_Symbol[ 46], "Pd");
  strcpy(Atom_Symbol[ 47], "Ag");
  strcpy(Atom_Symbol[ 48], "Cd");
  strcpy(Atom_Symbol[ 49], "In");
  strcpy(Atom_Symbol[ 50], "Sn");
  strcpy(Atom_Symbol[ 51], "Sb");
  strcpy(Atom_Symbol[ 52], "Te");
  strcpy(Atom_Symbol[ 53], "I");
  strcpy(Atom_Symbol[ 54], "Xe");
  strcpy(Atom_Symbol[ 55], "Cs");
  strcpy(Atom_Symbol[ 56], "Ba");
  strcpy(Atom_Symbol[ 57], "La");
  strcpy(Atom_Symbol[ 58], "Ce");
  strcpy(Atom_Symbol[ 59], "Pr");
  strcpy(Atom_Symbol[ 60], "Nd");
  strcpy(Atom_Symbol[ 61], "Pm");
  strcpy(Atom_Symbol[ 62], "Sm");
  strcpy(Atom_Symbol[ 63], "Eu");
  strcpy(Atom_Symbol[ 64], "Gd");
  strcpy(Atom_Symbol[ 65], "Tb");
  strcpy(Atom_Symbol[ 66], "Dy");
  strcpy(Atom_Symbol[ 67], "Ho");
  strcpy(Atom_Symbol[ 68], "Er");
  strcpy(Atom_Symbol[ 69], "Tm");
  strcpy(Atom_Symbol[ 70], "Yb");
  strcpy(Atom_Symbol[ 71], "Lu");
  strcpy(Atom_Symbol[ 72], "Hf");
  strcpy(Atom_Symbol[ 73], "Ta");
  strcpy(Atom_Symbol[ 74], "W");
  strcpy(Atom_Symbol[ 75], "Re");
  strcpy(Atom_Symbol[ 76], "Os");
  strcpy(Atom_Symbol[ 77], "Ir");
  strcpy(Atom_Symbol[ 78], "Pt");
  strcpy(Atom_Symbol[ 79], "Au");
  strcpy(Atom_Symbol[ 80], "Hg");
  strcpy(Atom_Symbol[ 81], "Tl");
  strcpy(Atom_Symbol[ 82], "Pb");
  strcpy(Atom_Symbol[ 83], "Bi");
  strcpy(Atom_Symbol[ 84], "Po");
  strcpy(Atom_Symbol[ 85], "At");
  strcpy(Atom_Symbol[ 86], "Rn");
  strcpy(Atom_Symbol[ 87], "Fr");
  strcpy(Atom_Symbol[ 88], "Ra");
  strcpy(Atom_Symbol[ 89], "Ac");
  strcpy(Atom_Symbol[ 90], "Th");
  strcpy(Atom_Symbol[ 91], "Pa");
  strcpy(Atom_Symbol[ 92], "U");
  strcpy(Atom_Symbol[ 93], "Np");
  strcpy(Atom_Symbol[ 94], "Pu");
  strcpy(Atom_Symbol[ 95], "Am");
  strcpy(Atom_Symbol[ 96], "Cm");
  strcpy(Atom_Symbol[ 97], "Bk");
  strcpy(Atom_Symbol[ 98], "Cf");
  strcpy(Atom_Symbol[ 99], "Es");
  strcpy(Atom_Symbol[100], "Fm");
  strcpy(Atom_Symbol[101], "Md");
  strcpy(Atom_Symbol[102], "No");
  strcpy(Atom_Symbol[103], "Lr");
}
