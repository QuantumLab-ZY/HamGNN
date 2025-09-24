/*

  Usage:   thisprogram   Cgra.Dos.val Cgra.Dos.vec 

  output: Cgra.DOS.Tetrahedron
          Cgra.PDOS.Tetrahedron

*/

 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Inputtools.h"

#define BINARYWRITE  

#define eV2Hartree 27.2113845
#define PI 3.1415926535897932384626
#define fp_bsize         1048576     /* buffer size for setvbuf */
 
#define YOUSO10  500

typedef struct DCOMPLEX{double r,i;} dcomplex;  

double ***EigenE;
double ****PDOS;
int **Msize1;
 
void *malloc_multidimarray(char *type, int N, int *size);
void OrderE0(double *e, int n);
void OrderE(double *e,double *a, int n);
void ATM_Dos(double *et, double *e, double *dos);
void ATM_Spectrum(double *et,double *at, double *e, double *spectrum);
char *Delete_path_extension(char *s0, char *r0);
void Dos_Gaussian( char *basename, int Dos_N, double Dos_Gaussian, 
		   int knum_i,int knum_j, int knum_k,
		   int SpinP_switch, double *****EIGEN, double *******ev,
                   int neg,
		   double Dos_emin, double Dos_emax, 
		   int iemin, int iemax,
                   int *WhatSpecies, int *Spe_Total_CNO, int atomnum );
void DosDC_Gaussian( char *basename, int atomnum, 
                     int Dos_N, double Dos_Gaussian, 
		     int SpinP_switch,
  		     double Dos_emin, double Dos_emax, 
		     int iemin, int iemax,
                     int *WhatSpecies, int *Spe_Total_CNO );
void Dos_NEGF( char *basename,
               char *extension,
               int Dos_N,
	       int SpinP_switch, double *****EIGEN, double *******ev,
               int neg,
	       double Dos_emin, double Dos_emax, 
	       int iemin, int iemax,
	       int *WhatSpecies, int *Spe_Total_CNO, int atomnum );
void Dos_Tetrahedron( char *basename, int Dos_N, 
		      int knum_i,int knum_j, int knum_k,
		      int SpinP_switch, double *****EIGEN, double *******ev,
                      int neg, 
		      double Dos_emin, double Dos_emax, 
		      int iemin, int iemax,
                      int *WhatSpecies, int *Spe_Total_CNO, int atomnum );
void Spectra_Gaussian( int pdos_n, int *pdos_atoms, char *basename, int Dos_N, 
		       double Dos_Gaussian, 
		       int knum_i, int knum_j, int knum_k,
		       int SpinP_switch, double *****EIGEN, double *******ev, int neg,
		       double Dos_emin, double Dos_emax,
		       int iemin, int iemax,
		       int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
		       int MaxL, int **Spe_Num_CBasis, int **Spe_Num_Relation);
void SpectraDC_Gaussian( int pdos_n, int *pdos_atoms, char *basename, int Dos_N, 
  		         double Dos_Gaussian, 
		         int knum_i,int knum_j, int knum_k,
		         int SpinP_switch, double *****EIGEN, double *******ev, int neg,
		         double Dos_emin, double Dos_emax,
		         int iemin, int iemax,
		         int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
		         int MaxL,int **Spe_Num_CBasis, int **Spe_Num_Relation);
void Spectra_Tetrahedron( int pdos_n, int *pdos_atoms, char *basename, int Dos_N, 
			  int knum_i,int knum_j, int knum_k,
			  int SpinP_switch, double *****EIGEN, double *******ev, int neg, 
			  double Dos_emin, double Dos_emax, 
			  int iemin, int iemax,
			  int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
                          int MaxL,int **Spe_Num_CBasis, int **Spe_Num_Relation);
void Spectra_NEGF( int pdos_n, int *pdos_atoms, char *basename, char *extension, int Dos_N, 
		   int SpinP_switch, double *****EIGEN, double *******ev, int neg,
		   double Dos_emin, double Dos_emax,
		   int iemin, int iemax,
		   int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
		   int MaxL, int **Spe_Num_CBasis, int **Spe_Num_Relation);
void  Spe_Num_CBasis2Relation(
			      int SpeciesNum,int MaxL,int *Spe_Total_CNO,int **Spe_Num_CBasis,
			      int **Spe_Num_Relation);
void input_file_eg(char *file_eg,
		   int *mode, int *NonCol,
                   int *n, double *emin, double *emax, int *iemin,int *iemax, 
		   int *SpinP_switch, int Kgrid[3], int *atomnum, int **WhatSpecies,
		   int *SpeciesNum, int **Spe_Total_CNO, int * Max_tnoA, int *MaxL,
		   int ***Spe_Num_CBasis, int ***Spe_Num_Relation, double *ChemP, 
                   double **Angle0_Spin, double **Angle1_Spin,
		   double ******ko);
void input_file_ev( char *file_ev, int mode, int NonCol, int Max_tnoA, 
		    int Kgrid[3], int SpinP_switch, int iemin, int iemax, 
                    int atomnum,  int *WhatSpecies, int *Spe_Total_CNO, 
		    double ********ev, double ChemP);
void input_main( int mode, int Kgrid[3], int atomnum, 
		 int *method, int *todo, 
		 double *gaussian, 
		 int *pdos_n, int **pdos_atoms);



int main(int argc, char **argv)
{

  char *file_eg, *file_ev;
 
  int mode,NonCol;
  int n;
  double emax,emin;
  int iemax,iemin;
  int SpinP_switch;
  int Kgrid[3];
  int atomnum,SpeciesNum;
  double ChemP;
  int *WhatSpecies, *Spe_Total_CNO,MaxL;

  double *Angle0_Spin,*Angle1_Spin;
  double *****ko;
  double *******ev; 
  int   **Spe_Num_CBasis; 
  int **Spe_Num_Relation; 

  double gaussian; 
  int Dos_N;
  int Max_tnoA;

  char basename[YOUSO10];

  /**************************************************
                read data from file_eg
  ***************************************************/

  file_eg=argv[1];
  file_ev=argv[2];

  input_file_eg(file_eg,
		&mode, &NonCol, &n, &emin, &emax, &iemin,&iemax,
		&SpinP_switch,  Kgrid, &atomnum, &WhatSpecies,
		&SpeciesNum, &Spe_Total_CNO, &Max_tnoA, &MaxL,
		&Spe_Num_CBasis, &Spe_Num_Relation, &ChemP,
                &Angle0_Spin, &Angle1_Spin,
		&ko); 


  input_file_ev( file_ev, mode, NonCol, Max_tnoA,
		 Kgrid,  SpinP_switch,  iemin,  iemax,
		 atomnum,  WhatSpecies,  Spe_Total_CNO,
		 &ev, ChemP);


  Delete_path_extension(file_eg,basename);

  {
    int todo,method;
    int pdos_n, *pdos_atoms;
 
    input_main(  mode, Kgrid,  atomnum,
		 &method, &todo,
		 &gaussian,
		 &pdos_n, &pdos_atoms); 

    /* set a suitable Dos_N */

    if (method==1){
      Dos_N = 500;
    }
    else if (method==4){
      Dos_N = iemax - iemin + 1;
    }
    else{
      Dos_N = (emax-emin)/gaussian*5;
    }


    /* Total DOS */
    if (todo==1) {

      switch (method) {
      case 1:
	Dos_Tetrahedron(  basename,Dos_N,
			  Kgrid[0],Kgrid[1],Kgrid[2], 
			  SpinP_switch, ko, ev, iemax-iemin+1, 
			  emin, emax, 
			  iemin,  iemax, WhatSpecies, Spe_Total_CNO, atomnum );
	break;
      case 2: 

        /* DC */
        if (mode==5){

	DosDC_Gaussian(  basename, atomnum, Dos_N, gaussian,
		         SpinP_switch, emin, emax,
		         iemin, iemax, WhatSpecies, Spe_Total_CNO );
	}
	else{

	Dos_Gaussian(  basename, Dos_N,  gaussian,  
		       Kgrid[0],Kgrid[1],Kgrid[2],
		       SpinP_switch, ko, ev, iemax-iemin+1,
		       emin, emax,
		       iemin,  iemax, WhatSpecies, Spe_Total_CNO, atomnum );

	}

	break;
#if 0
      case 3:
	Dos_Histgram(  basename,Dos_N,
		       Kgrid[0],Kgrid[1],Kgrid[2],
		       SpinP_switch, ko, iemax-iemin+1,
		       emin, emax,
		       iemin,  iemax );
	break;
#endif

      case 4: 

        if (mode==6){

	Dos_NEGF( basename, "NEGF", Dos_N,
		  SpinP_switch, ko, ev, iemax-iemin+1,
		  emin, emax,
		  iemin,  iemax, WhatSpecies, Spe_Total_CNO, atomnum );
	}

        else if (mode==7){

	Dos_NEGF( basename, "Gaussian", Dos_N,
		  SpinP_switch, ko, ev, iemax-iemin+1,
		  emin, emax,
		  iemin,  iemax, WhatSpecies, Spe_Total_CNO, atomnum );
	}

	break;
      }
    }

    /* Projected DOS */

    else if (todo==2) {

      switch(method) {
      case 1:
	Spectra_Tetrahedron( pdos_n, pdos_atoms, basename,Dos_N,
			     Kgrid[0],Kgrid[1],Kgrid[2],
			     SpinP_switch, ko,ev, iemax-iemin+1,
			     emin, emax,
			     iemin,  iemax,
			     atomnum, WhatSpecies, Spe_Total_CNO, 
			     MaxL,Spe_Num_CBasis, Spe_Num_Relation );
	break;
      case 2:

        /* DC */
        if (mode==5){

	SpectraDC_Gaussian( pdos_n, pdos_atoms, basename,Dos_N, gaussian, 
			    Kgrid[0],Kgrid[1],Kgrid[2],
			    SpinP_switch, ko,ev, iemax-iemin+1,
			    emin, emax,
			    iemin,  iemax,
			    atomnum, WhatSpecies, Spe_Total_CNO,
			    MaxL,Spe_Num_CBasis, Spe_Num_Relation );
	}
        else {

	Spectra_Gaussian( pdos_n, pdos_atoms, basename,Dos_N, gaussian, 
			  Kgrid[0],Kgrid[1],Kgrid[2],
			  SpinP_switch, ko,ev, iemax-iemin+1,
			  emin, emax,
			  iemin,  iemax,
			  atomnum, WhatSpecies, Spe_Total_CNO,
			  MaxL,Spe_Num_CBasis, Spe_Num_Relation );
        }

	break;

      case 4: 

        if (mode==6){

	Spectra_NEGF( pdos_n, pdos_atoms, basename, "NEGF", Dos_N,
		      SpinP_switch, ko,ev, iemax-iemin+1,
		      emin, emax,
		      iemin,  iemax,
		      atomnum, WhatSpecies, Spe_Total_CNO,
		      MaxL,Spe_Num_CBasis, Spe_Num_Relation );
	}

        else if (mode==7){

	Spectra_NEGF( pdos_n, pdos_atoms, basename, "Gaussian", Dos_N,
		      SpinP_switch, ko,ev, iemax-iemin+1,
		      emin, emax,
		      iemin,  iemax,
		      atomnum, WhatSpecies, Spe_Total_CNO,
		      MaxL,Spe_Num_CBasis, Spe_Num_Relation );
	}

	break;

      }
    }
  }

  exit(0);
}





/* 
   input s0
   output r0
   if input='../test.Dos.val' output='test'
*/

char *Delete_path_extension(char *s0, char *r0)
{
  char *c;
  /* char s[YOUSO10]; */

  /* find '/' */
  c=rindex(s0,'/');
  if (c)  {
    strcpy(r0,c+1);
  }
  else {
    strcpy(r0,s0);
  }
  printf("<%s>\n",r0);

  if (strlen(r0)==0 ) { return  NULL; }
  
  c =index(r0,'.');
  if (c) {
    *c='\0';
  }
  printf("<%s>\n",r0);
  return r0;

}


void Dos_Histgram( char *basename, int Dos_N,
		   int knum_i,int knum_j, int knum_k,
		   int SpinP_switch, double *****EIGEN, int neg,
		   double Dos_emin, double Dos_emax, 
		   int iemin, int iemax )
{
  int ie,spin,i,j,k,ieg,iecenter;
  double *DosE, **Dos;
  double eg,x,factor;
  int N_Dos , i_Dos[10];
  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  printf("<Dos_Histgram> start\n");

  if (strlen(basename)==0) {
    strcpy(file_Dos,"DOS.Histgram");
  }
  else {
    sprintf(file_Dos,"%s.DOS.Histgram",basename);
  }
  if ( (fp_Dos=fopen(file_Dos,"w"))==NULL ) {

#ifdef xt3
    setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("can not open a file %s\n",file_Dos);
    return;
  }

  printf("<Dos_Histgram> make %s\n",file_Dos);

  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);
  N_Dos=2; i_Dos[0]=SpinP_switch+1; i_Dos[1]=Dos_N;
  Dos=(double**)malloc_multidimarray("double",N_Dos,i_Dos);

  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie]=0.0;
    }
  }

  for (spin=0;spin<=SpinP_switch;spin++) {
    for (i=0; i<=(knum_i-1); i++){
      for (j=0; j<=(knum_j-1); j++){
	for (k=0; k<=(knum_k-1); k++){
	  for (ieg=0;ieg<neg; ieg++) {

	    eg = EIGEN[spin][i][j][k][ieg] ;
	    x = (eg-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1);
	    iecenter = (int)x ;

	    if (iecenter>=0 && iecenter <Dos_N )
	      Dos[spin][iecenter] = Dos[spin][iecenter] + 1.0;
	  } /* ieg */
	} /* k */
      } /* j */
    } /* i */
  } /* spin */

  /* normalize */
  factor =  eV2Hartree * knum_i * knum_j * knum_k * (Dos_emax-Dos_emin)/(Dos_N-1); 
  factor = 1.0/factor;
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie] = Dos[spin][ie] * factor;
    }
  }

  if (SpinP_switch==1) {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf\n",(DosE[ie])*eV2Hartree, Dos[0][ie], -Dos[1][ie]);
    }
  }
  else {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf\n",(DosE[ie])*eV2Hartree, Dos[0][ie]*2.0);
    }
  }

  if (fp_Dos) fclose(fp_Dos);


  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    free(Dos[i]);
  }
  free(Dos);

}


void Dos_Gaussian( char *basename, int Dos_N, double Dos_Gaussian, 
		   int knum_i,int knum_j, int knum_k,
		   int SpinP_switch, double *****EIGEN, double *******ev,
                   int neg,
		   double Dos_emin, double Dos_emax, 
		   int iemin, int iemax,
                   int *WhatSpecies, int *Spe_Total_CNO, int atomnum )
{

  /*****************************************************************
                            Method = Gaussian 
  *****************************************************************/
  /*     exp( -(x/a)^2 ) 
	 a= Dos_Gaussian 
	 x=-3a : 3a is counted */
  double pi2;
  int iewidth,ie;

  int spin,i,j,k,ieg,iecenter;
  int wanA,GA_AN,tnoA,i1;
  double eg,x,xa,factor,wval;

  double *DosE, **Dos;
  int N_Dos,i_Dos[10];

  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* sawada */

  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N];

  printf("<Dos_Gaussian> start\n");

  if (strlen(basename)==0) {
    strcpy(file_Dos,"DOS.Gaussian");
  }
  else {
    sprintf(file_Dos,"%s.DOS.Gaussian",basename);
  }
  if ( (fp_Dos=fopen(file_Dos,"w"))==NULL ) {
    printf("can not open a file %s\n",file_Dos);
    return;
  }
  
  printf("<Dos_Gaussian> make %s\n",file_Dos);

  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);
  N_Dos=2; i_Dos[0]=SpinP_switch+1; i_Dos[1]=Dos_N;
  Dos=(double**)malloc_multidimarray("double",N_Dos,i_Dos);

  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie]=0.0;
    }
  }

  pi2 = sqrt(PI);
  iewidth = Dos_Gaussian*3.0/(Dos_emax-Dos_emin)*(Dos_N-1)+3;
  printf("Gaussian width=%d\n",iewidth); 

  for (spin=0;spin<=SpinP_switch;spin++) {
    for (i=0; i<=(knum_i-1); i++){
      for (j=0; j<=(knum_j-1); j++){
	for (k=0; k<=(knum_k-1); k++){
	  for (ieg=0;ieg<neg; ieg++) {


            /* generalization for spin non-collinear */
            wval = 0.0;
   	    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
  	      wanA = WhatSpecies[GA_AN];
	      tnoA = Spe_Total_CNO[wanA];
  	      for (i1=0; i1<tnoA; i1++){
                wval += ev[spin][i][j][k][GA_AN][i1][ieg];
	      }
	    }

	    eg = EIGEN[spin][i][j][k][ieg] ;
	    x = (eg-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1);
	    iecenter = (int)x ; 

	    iemin = iecenter - iewidth;
	    iemax = iecenter + iewidth;

	    if (iemin<0) iemin=0;
	    if (iemax>=Dos_N) iemax=Dos_N-1;
	    if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax <Dos_N ) {

	      for (ie=iemin;ie<=iemax;ie++) {
		xa = (eg-DosE[ie])/Dos_Gaussian; 

		/*
            printf("Dos_N=%3d iemin=%3d iemax=%3d Dos_Gaussian=%15.12f %15.12f\n",
                    Dos_N,iemin,iemax,Dos_Gaussian,wval*exp( -xa*xa)/(Dos_Gaussian*pi2));
		*/

		Dos[spin][ie] += wval*exp(- xa*xa)/(Dos_Gaussian*pi2);
	      }
	    } 
	  } /* ieg */
	} /* k */
      } /* j */
    } /* i */
  } /* spin */

  /* normalize */
  factor = 1.0/(double)( eV2Hartree * knum_i * knum_j * knum_k );
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie] = Dos[spin][ie] * factor;
    }

  /* sawada */
  
  h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
  sw = 2;
  
  if (sw == 1) {

    /* Trapezoidal rule */

    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      for (ie=1;ie<q;ie++) {
        s1 += Dos[spin][ie];
      }   
      ssum[spin][q] = (0.5*(Dos[spin][0]+Dos[spin][q])+s1)*h;
    }
  } else {

    /* Simpson's rule */

    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      s2 = 0.0;
      for (ie=1;ie<q;ie+=2) {
        s1 += Dos[spin][ie];
      }
      for (ie=2;ie<q;ie+=2) {
        s2 += Dos[spin][ie];
      }
      ssum[spin][q] = (Dos[spin][0]+4.0*s1+2.0*s2+Dos[spin][q])*h/3.0;
    }
  }
  
  }

  if (SpinP_switch==1) {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie], -Dos[1][ie], ssum[0][ie], ssum[1][ie]);
    }
  }

  else {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie]*2.0, ssum[0][ie]*2.0);
    }
  }

  if (fp_Dos) fclose(fp_Dos);

  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    free(Dos[i]);
  }
  free(Dos);
}



void DosDC_Gaussian( char *basename, int atomnum, 
                     int Dos_N, double Dos_Gaussian, 
		     int SpinP_switch,
  		     double Dos_emin, double Dos_emax, 
		     int iemin, int iemax,
                     int *WhatSpecies, int *Spe_Total_CNO )
{

  /*****************************************************************
                          Method = Gaussian 
  *****************************************************************/
  /********************************
          exp( -(x/a)^2 ) 
	  a= Dos_Gaussian 
 	  x=-3a : 3a is counted 
  ********************************/

  double pi2;
  int iewidth,ie;
  int GA_AN,wanA,tnoA;

  int spin,i,j,k,ieg,iecenter;
  double eg,x,xa,factor,sum;

  double *DosE, **Dos;
  int N_Dos,i_Dos[10];
  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* sawada */
   
  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N];

  printf("<Dos_Gaussian> start\n");

  if (strlen(basename)==0) {
    strcpy(file_Dos,"DOS.Gaussian");
  }
  else {
    sprintf(file_Dos,"%s.DOS.Gaussian",basename);
  }
  if ( (fp_Dos=fopen(file_Dos,"w"))==NULL ) {

#ifdef xt3
    setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("can not open a file %s\n",file_Dos);
    return;
  }
  
  printf("<Dos_Gaussian> make %s\n",file_Dos);

  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);
  N_Dos=2; i_Dos[0]=SpinP_switch+1; i_Dos[1]=Dos_N;
  Dos=(double**)malloc_multidimarray("double",N_Dos,i_Dos);


  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie]=0.0;
    }
  }

  pi2=sqrt(PI);
  iewidth = Dos_Gaussian*3.0/(Dos_emax-Dos_emin)*(Dos_N-1)+3;
  printf("Gaussian width=%d\n",iewidth); 

  for (spin=0;spin<=SpinP_switch;spin++) {

    for (GA_AN=0; GA_AN<atomnum; GA_AN++){

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (ieg=0; ieg<Msize1[spin][GA_AN]; ieg++) {

        eg = EigenE[spin][GA_AN][ieg];
        x = (eg-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1);
        iecenter = (int)x ; 
        iemin = iecenter - iewidth;
        iemax = iecenter + iewidth ;

        if (iemin<0) iemin=0;
        if (iemax>=Dos_N) iemax=Dos_N-1;

        if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax <Dos_N ) {
          sum = 0.0;
          for (i=0; i<tnoA; i++) sum += PDOS[spin][GA_AN][ieg][i];
          for (ie=iemin; ie<=iemax; ie++) {
            xa = (eg-DosE[ie])/Dos_Gaussian; 
            Dos[spin][ie] += sum*exp(- xa*xa)/(Dos_Gaussian*pi2);
	  }
        } 

      }   /* ieg */
    }     /* GA_AN */
  }       /* spin */
 
  /* normalize */
  factor = 1.0/eV2Hartree;
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie] = Dos[spin][ie] * factor;
    }

  /* sawada */

  h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
  sw = 2;

  if (sw == 1) {

    /* Trapezoidal rule */

    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      for (ie=1;ie<q;ie++) {
        s1 += Dos[spin][ie];
      }
      ssum[spin][q] = (0.5*(Dos[spin][0]+Dos[spin][q])+s1)*h;
    }
  } else {

    /* Simpson's rule */

    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      s2 = 0.0;
      for (ie=1;ie<q;ie+=2) {
        s1 += Dos[spin][ie];
      }
      for (ie=2;ie<q;ie+=2) {
        s2 += Dos[spin][ie];
      }
      ssum[spin][q] = (Dos[spin][0]+4.0*s1+2.0*s2+Dos[spin][q])*h/3.0;
    }
  }
  
  }

  if (SpinP_switch==1) {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie], -Dos[1][ie], ssum[0][ie], ssum[1][ie]);
    }
  }
  else {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie]*2.0, ssum[0][ie]*2.0);
    }
  }

  if (fp_Dos) fclose(fp_Dos);

  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    free(Dos[i]);
  }
  free(Dos);
}






void Dos_NEGF( char *basename,
               char *extension,
               int Dos_N,
	       int SpinP_switch, double *****EIGEN, double *******ev,
               int neg,
	       double Dos_emin, double Dos_emax, 
	       int iemin, int iemax,
	       int *WhatSpecies, int *Spe_Total_CNO, int atomnum )
{
  /*****************************************************************
                            Method = NEGF
  *****************************************************************/

  double pi2;
  int iewidth,ie;

  int spin,i,j,k,ieg,iecenter;
  int wanA,GA_AN,tnoA,i1;
  double eg,x,xa,factor,wval;

  double *DosE, **Dos;
  int N_Dos,i_Dos[10];

  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* sawada */

  int q,sw;
  double h,s1,s2;
  double ssum[2][Dos_N];

  printf("<Dos_%s> start\n",extension);

  if (strlen(basename)==0) {
    sprintf(file_Dos,"DOS.%s",extension);
  }
  else {
    sprintf(file_Dos,"%s.DOS.%s",basename,extension);
  }
  if ( (fp_Dos=fopen(file_Dos,"w"))==NULL ) {
    printf("can not open a file %s\n",file_Dos);
    return;
  }
  
  printf("<Dos_%s> make %s\n",extension,file_Dos);

  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);
  N_Dos=2; i_Dos[0]=SpinP_switch+1; i_Dos[1]=Dos_N;
  Dos=(double**)malloc_multidimarray("double",N_Dos,i_Dos);

  /* note: Dos_N == neg */ 

  /* initialize */

  for (ie=0; ie<Dos_N; ie++) {
    DosE[ie] = EIGEN[0][0][0][0][ie];
  }

  for (spin=0; spin<=SpinP_switch; spin++){
    for (ie=0; ie<Dos_N; ie++){
      Dos[spin][ie] = 0.0;
    }
  }

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (ieg=0; ieg<neg; ieg++) {

      i = 0;
      j = 0;
      k = 0; 

      wval = 0.0;

      for (GA_AN=0; GA_AN<atomnum; GA_AN++){
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<tnoA; i1++){
	  wval += ev[spin][i][j][k][GA_AN][i1][ieg];
	}
      }

      Dos[spin][ieg] = wval/eV2Hartree;

    } /* ieg */

  /* sawada */
   
  h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
  sw = 2; 
   
  if (sw == 1) {
 
    /* Trapezoidal rule */
 
    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      for (ie=1;ie<q;ie++) {
        s1 += Dos[spin][ie];
      }   
      ssum[spin][q] = (0.5*(Dos[spin][0]+Dos[spin][q])+s1)*h;
    } 
  } else {
 
    /* Simpson's rule */
 
    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      s2 = 0.0;
      for (ie=1;ie<q;ie+=2) {
        s1 += Dos[spin][ie];
      } 
      for (ie=2;ie<q;ie+=2) {
        s2 += Dos[spin][ie];
      } 
      ssum[spin][q] = (Dos[spin][0]+4.0*s1+2.0*s2+Dos[spin][q])*h/3.0;
    } 
  } 
    
  } /* spin */
 
  /* save */

  if (SpinP_switch==1) {
    for (ie=0; ie<Dos_N; ie++) {
      fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie], -Dos[1][ie], ssum[0][ie], ssum[1][ie]);
    }
  }
  else {
    for (ie=0; ie<Dos_N; ie++) {
      fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie]*2.0, ssum[0][ie]*2.0);
    }
  }

  if (fp_Dos) fclose(fp_Dos);

  /* free arrays */
 
  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    free(Dos[i]);
  }
  free(Dos);
}






void Dos_Tetrahedron( char *basename, int Dos_N, 
		      int knum_i,int knum_j, int knum_k,
		      int SpinP_switch, double *****EIGEN, double *******ev,
                      int neg, 
		      double Dos_emin, double Dos_emax, 
		      int iemin, int iemax,
                      int *WhatSpecies, int *Spe_Total_CNO, int atomnum )
{
  int spin,ieg,i,j,k,ie,i1,tnoA,wanA,GA_AN;
  int i_in,j_in,k_in,ic,itetra; 
  double x,result,factor,wval; 
  double cell_e[8], tetra_e[4];
  static int tetra_id[6][4]= { {0,1,2,5}, {1,2,3,5}, {2,3,5,7},
			       {0,2,4,5}, {2,4,5,6}, {2,5,6,7} };

  double *DosE, **Dos;
  int N_Dos,i_Dos[10];
  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* sawada */

  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N];

  printf("<Dos_Tetrahedron> start\n");

  if (strlen(basename)==0) {
    strcpy(file_Dos,"DOS.Tetrahedron");
  }
  else {
    sprintf(file_Dos,"%s.DOS.Tetrahedron",basename);
  }
  if ( (fp_Dos=fopen(file_Dos,"w"))==NULL ) {

#ifdef xt3
    setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("can not open a file %s\n",file_Dos);
    return;
  }

  printf("<Dos_Tetrahedron> make %s\n",file_Dos);

  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);
  N_Dos=2; i_Dos[0]=SpinP_switch+1; i_Dos[1]=Dos_N;
  Dos=(double**)malloc_multidimarray("double",N_Dos,i_Dos);
   
  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie]=0.0;
    }
  }

  /********************************************************************
                           Method = tetrahedron 
  *******************************************************************/

  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ieg=0;ieg<neg; ieg++) {
      for (i=0; i<=(knum_i-1); i++){
	for (j=0; j<=(knum_j-1); j++){
	  for (k=0; k<=(knum_k-1); k++){

            /* generalization for spin non-collinear */
            wval = 0.0;
   	    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
  	      wanA = WhatSpecies[GA_AN];
	      tnoA = Spe_Total_CNO[wanA];
  	      for (i1=0; i1<tnoA; i1++){
                wval += ev[spin][i][j][k][GA_AN][i1][ieg];
	      }
	    }

	    for (i_in=0;i_in<2;i_in++) {
	      for (j_in=0;j_in<2;j_in++) {
		for (k_in=0;k_in<2;k_in++) {
		  cell_e[i_in*4+j_in*2+k_in] = 
		    EIGEN[spin][ (i+i_in)%knum_i ][ (j+j_in)%knum_j ][ (k+k_in)%knum_k ][ieg] ;
		}
	      }
	    }
	    for (itetra=0;itetra<6;itetra++) {
	      for (ic=0;ic<4;ic++) {
		tetra_e[ic]=cell_e[ tetra_id[itetra][ic] ];
	      }
	      OrderE0(tetra_e,4);
	      x = (tetra_e[0]-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1)-1.0;
	      iemin=(int)x;
	      x = (tetra_e[3]-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1)+1.0;
	      iemax=(int)x;
	      if (iemin<0) { iemin=0; }
	      if (iemax>=Dos_N) {iemax=Dos_N-1; }
	      if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax<Dos_N ) {
		for (ie=iemin;ie<=iemax;ie++) {
		  ATM_Dos( tetra_e, &DosE[ie], &result);
		  Dos[spin][ie] += wval*result;
		}
	      }

	    } /* itetra */
	  } /* k */
	} /* j */
      } /* i */
    } /* ieg */
  } /* spin */

  /* normalize */
  factor = 1.0/(double)( eV2Hartree * knum_i * knum_j * knum_k * 6 );
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ie=0;ie<Dos_N;ie++) {
      Dos[spin][ie] = Dos[spin][ie] * factor;
    }

  /* sawada */
   
  h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
  sw = 2; 
   
  if (sw == 1) {
 
    /* Trapezoidal rule */
 
    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      for (ie=1;ie<q;ie++) {
        s1 += Dos[spin][ie];
      }   
      ssum[spin][q] = (0.5*(Dos[spin][0]+Dos[spin][q])+s1)*h;
    } 
  } else {
 
    /* Simpson's rule */
 
    for (q=0;q<Dos_N;q++) {
      s1 = 0.0;
      s2 = 0.0;
      for (ie=1;ie<q;ie+=2) {
        s1 += Dos[spin][ie];
      } 
      for (ie=2;ie<q;ie+=2) {
        s2 += Dos[spin][ie];
      } 
      ssum[spin][q] = (Dos[spin][0]+4.0*s1+2.0*s2+Dos[spin][q])*h/3.0;
    } 
  } 

  }

  if (SpinP_switch==1) {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie], -Dos[1][ie], ssum[0][ie], ssum[1][ie]);
    }
  } 
  else {
    for (ie=0;ie<Dos_N;ie++) {
      fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, Dos[0][ie]*2.0, ssum[0][ie]*2.0);
    }
  }

  if (fp_Dos) fclose(fp_Dos);

  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    free(Dos[i]);
  }
  free(Dos);
}
   



void Spectra_Gaussian( int pdos_n, int *pdos_atoms, char *basename, int Dos_N, 
		       double Dos_Gaussian, 
		       int knum_i, int knum_j, int knum_k,
		       int SpinP_switch, double *****EIGEN, double *******ev, int neg,
		       double Dos_emin, double Dos_emax,
		       int iemin, int iemax,
		       int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
		       int MaxL, int **Spe_Num_CBasis, int **Spe_Num_Relation)
{

  /*****************************************************************
                           Method = Gaussian 
  *****************************************************************/
  /*     exp( -(x/a)^2 ) 
         a= Dos_Gaussian 
         x=-3a : 3a is counted */

  int Max_tnoA;
  int i,spin,j,k;
  int wanA, GA_AN,tnoA,i1,ieg;
  double eg,x,rval,xa;
  double *DosE, ****Dos;
  int N_Dos, i_Dos[10];
  double pi2,factor;
  int  iewidth,ie,iecenter;
  int iatom,L,M,LM;
  static char *Lname[5]={"s","p","d","f","g"};
  double dossum[2],MulP[5],dE;

  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */
  static char *extension="PDOS.Gaussian";

  /* sawada */

  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N], Doss[2][Dos_N];

  printf("<Spectra_Gaussian> start\n");

  pi2=sqrt(PI);
  iewidth = Dos_Gaussian*3.0/(Dos_emax-Dos_emin)*(Dos_N-1)+3;


  Max_tnoA=0;
  for (i=0;i<atomnum;i++) {
    wanA=WhatSpecies[i];
    if (Max_tnoA<Spe_Total_CNO[wanA]) Max_tnoA=Spe_Total_CNO[wanA];
  }
  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);

  N_Dos=4; i_Dos[0]=SpinP_switch+1; 
  i_Dos[1] = atomnum; i_Dos[2]= Max_tnoA;
  i_Dos[3]=Dos_N;
  Dos=(double****)malloc_multidimarray("double",N_Dos,i_Dos);


  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i1=0; i1<=(tnoA-1); i1++){

        for (ie=0;ie<Dos_N;ie++) {
          Dos[spin][GA_AN][i1][ie]=0.0;
        }
      }
    }
  }

  /********************************************************************
                           Method = gaussian
  *******************************************************************/

  factor = 1.0/(double)(knum_i * knum_j * knum_k); /* normalization factor */

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (i=0; i<=(knum_i-1); i++){
      for (j=0; j<=(knum_j-1); j++){
	for (k=0; k<=(knum_k-1); k++){
	  for (ieg=0;ieg<neg; ieg++) {

	    eg = EIGEN[spin][i][j][k][ieg];

	    x = (eg-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1);
	    iecenter = (int)x ;
	    iemin = iecenter - iewidth;
	    iemax = iecenter + iewidth ;
	    if (iemin<0) iemin=0;
	    if (iemax>=Dos_N) iemax=Dos_N-1;
         
	    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
	      wanA = WhatSpecies[GA_AN];
	      tnoA = Spe_Total_CNO[wanA];
	      for (i1=0; i1<=(tnoA-1); i1++){
 
		rval = ev[spin][i][j][k][GA_AN][i1][ieg];

		if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax <Dos_N ) {
		  for (ie=iemin; ie<=iemax; ie++) {
		    xa = (eg-DosE[ie])/Dos_Gaussian; 

		    Dos[spin][GA_AN][i1][ie] += factor*rval* exp(-xa*xa)/(Dos_Gaussian*pi2*eV2Hartree);
		  }
		} /* if */
	      } /* i1 */
	    } /* GA_AN */
	  } /* ieg */
	} /* k */
      } /* j */
    } /* i */
  } /* spin */

  /* orbital projection */
  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (L=0;L<=MaxL; L++) {
      if (Spe_Num_CBasis[wanA][L]>0) {
        for (M=0;M<2*L+1;M++) {
          LM=100*L+M+1;
	  sprintf(file_Dos,"%s.%s.atom%d.%s%d",basename,extension,pdos_atoms[iatom],Lname[L],M+1);
	  if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
            setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	    printf("can not open %s\n",file_Dos);
	  }
	  else {
	    printf("make %s\n",file_Dos);

            /* sawada */ 
 
            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2; 
     
            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }

            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) { 
              /* Trapezoidal rule */ 
                for (q=0;q<Dos_N;q++) { 
                  s1 = 0.0; 
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      } 
                    }
                    s1 += Doss[spin][ie];
                  } 
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                } 
              } else { 
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }  
                    }
                    s1 += Doss[spin][ie];
                  } 
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }  
                    }
                    s2 += Doss[spin][ie];
                  } 
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                } 
              }             
            }

	    for (ie=0;ie<Dos_N;ie++) {
	      for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	      for (i1=0;i1< tnoA; i1++) {
		if (LM== Spe_Num_Relation[wanA][i1]) {
		  for (spin=0;spin<=SpinP_switch;spin++) {
		    dossum[spin]+=  Dos[spin][GA_AN][i1][ie];
                  }
		}
	      }
             
	      if (SpinP_switch==1) {
		fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
                        dossum[0],-dossum[1], ssum[0][ie], ssum[1][ie]);
	      }
	      else {
		fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, dossum[0]*2.0, ssum[0][ie]*2.0);
	      }

	    }
	  }
	  if (fp_Dos) fclose(fp_Dos);
	} /* M */
      }
    } /* L */
  } /* iatom */

  /* atom projection */
  for (iatom=0; iatom<pdos_n; iatom++) {

    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    sprintf(file_Dos,"%s.%s.atom%d",basename,extension,pdos_atoms[iatom]);

    if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
      setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open %s\n",file_Dos);
    }
    else {
      printf("make %s\n",file_Dos);

      /*
      for (spin=0; spin<=SpinP_switch; spin++){
        MulP[spin] = 0.0;
      }
      */

      /* sawada */
  
      h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
      sw = 2;
     
      for (spin=0;spin<=SpinP_switch;spin++) {
        for (ie=1;ie<Dos_N;ie++) {
          Doss[spin][ie] = 0.0;
        } 
      } 
 
      for (spin=0;spin<=SpinP_switch;spin++) {
        if (sw == 1) {
        /* Trapezoidal rule */
          for (q=0;q<Dos_N;q++) {
            s1 = 0.0;
            for (ie=1;ie<q;ie++) {
              Doss[spin][ie] = 0.0;
              for (i1=0;i1< tnoA; i1++) {
                Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
              }
              s1 += Doss[spin][ie];
            }
            ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
          } 
        } else {
        /* Simpson's rule */
          for (q=0;q<Dos_N;q++) {
            s1 = 0.0;
            s2 = 0.0;
            for (ie=1;ie<q;ie+=2) {
              Doss[spin][ie] = 0.0;
              for (i1=0;i1< tnoA; i1++) {
                Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
              }
              s1 += Doss[spin][ie];
            }
            for (ie=2;ie<q;ie+=2) {
              Doss[spin][ie] = 0.0;
              for (i1=0;i1< tnoA; i1++) {
                Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
              }
              s2 += Doss[spin][ie];
            }
            ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
          } 
        }   
      }


      for (ie=0;ie<Dos_N;ie++) {
	for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	for (i1=0;i1< tnoA; i1++) {
	  for (spin=0;spin<=SpinP_switch;spin++)
	    dossum[spin]+=  Dos[spin][GA_AN][i1][ie];
	}

        /* integration */
	/*
        if (DosE[ie]<=0.00){
          dE = DosE[1] - DosE[0];
          for (spin=0; spin<=SpinP_switch; spin++){
            MulP[spin] += dossum[spin]*dE;
          }
	}
	*/

	if (SpinP_switch==1) {
	  fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
		  dossum[0],-dossum[1], ssum[0][ie], ssum[1][ie]);
	}
	else {
	  fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, dossum[0]*2.0, ssum[0][ie]*2.0);
	}

      } /* ie */
    } /* fp_Dos */
    if (fp_Dos) fclose(fp_Dos);


    /*
    for (spin=0; spin<=SpinP_switch; spin++){
      printf("spin=%2d MulP=%15.12f\n",spin,MulP[spin]*eV2Hartree);
    }
    */


  } /* iatom */
  

  /*********************************************************
                        close and free
  *********************************************************/

  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    for (j=0; j<i_Dos[1]; j++){
      for (k=0; k<i_Dos[2]; k++){
        free(Dos[i][j][k]);
      }
      free(Dos[i][j]);
    }
    free(Dos[i]);
  }
  free(Dos);
}


void SpectraDC_Gaussian( int pdos_n, int *pdos_atoms, char *basename, int Dos_N, 
  		         double Dos_Gaussian, 
		         int knum_i,int knum_j, int knum_k,
		         int SpinP_switch, double *****EIGEN, double *******ev, int neg,
		         double Dos_emin, double Dos_emax,
		         int iemin, int iemax,
		         int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
		         int MaxL,int **Spe_Num_CBasis, int **Spe_Num_Relation)
{

  /*****************************************************************
                           Method = Gaussian 
  *****************************************************************/
  /*     exp( -(x/a)^2 ) 
         a= Dos_Gaussian 
         x=-3a : 3a is counted */


  int Max_tnoA;
  int i,spin,j,k;
  int wanA,GA_AN,tnoA,i1,ieg;
  double eg,x,rval,xa,tmp1;
  double *DosE, ****Dos;
  int N_Dos, i_Dos[10];
  double pi2;
  int  iewidth,ie,iecenter;
  int iatom,L,M,LM;
  static char *Lname[5]={"s","p","d","f","g"};
  double dossum[2];

  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */
  static char *extension="PDOS.Gaussian";

  /* sawada */

  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N], Doss[2][Dos_N];

  printf("<Spectra_Gaussian> start\n");

  pi2=sqrt(PI);
  iewidth = Dos_Gaussian*3.0/(Dos_emax-Dos_emin)*(Dos_N-1)+3;

  Max_tnoA=0;
  for (i=0;i<atomnum;i++) {
    wanA=WhatSpecies[i];
    if (Max_tnoA<Spe_Total_CNO[wanA]) Max_tnoA=Spe_Total_CNO[wanA];
  }
  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);

  Dos = (double****)malloc(sizeof(double***)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++) {
    Dos[spin] =(double***)malloc(sizeof(double**)*pdos_n);
    for (iatom=0;iatom<pdos_n;iatom++) {
      GA_AN = pdos_atoms[iatom]-1;
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Dos[spin][iatom] =(double**)malloc(sizeof(double*)*tnoA);
      for (i=0; i<tnoA; i++){
        Dos[spin][iatom][i] =(double*)malloc(sizeof(double)*Dos_N);
        for (ie=0; ie<Dos_N; ie++) {
          Dos[spin][iatom][i][ie] =0.0;
	}
      }
    }
  }

  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }

  /********************************************************************
                           Method = gaussian
  *******************************************************************/

  for (spin=0;spin<=SpinP_switch;spin++) {
    for (iatom=0;iatom<pdos_n;iatom++) {

      GA_AN = pdos_atoms[iatom]-1;
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      for (ieg=0; ieg<Msize1[spin][GA_AN]; ieg++) {

        eg = EigenE[spin][GA_AN][ieg];
        x = (eg-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1);
        iecenter = (int)x ; 
        iemin = iecenter - iewidth;
        iemax = iecenter + iewidth ;
        if (iemin<0) iemin=0;
        if (iemax>=Dos_N) iemax=Dos_N-1;

        if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax <Dos_N ) {
          for (ie=iemin; ie<=iemax; ie++) {
            xa = (eg-DosE[ie])/Dos_Gaussian; 

            tmp1 = exp( -xa*xa)/(Dos_Gaussian*pi2*eV2Hartree);

            for (i=0; i<tnoA; i++){
              Dos[spin][iatom][i][ie] += PDOS[spin][GA_AN][ieg][i]*tmp1;
	    }
	  }
	}

      } /* ieg */
    }   /* iatom */
   }    /* spin */

  /* orbital projection */
  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (L=0;L<=MaxL; L++) {
      if (Spe_Num_CBasis[wanA][L]>0) {
        for (M=0;M<2*L+1;M++) {
          LM=100*L+M+1;
	  sprintf(file_Dos,"%s.%s.atom%d.%s%d",basename,extension,pdos_atoms[iatom],
		  Lname[L],M+1);
	  if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
            setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	    printf("can not open %s\n",file_Dos);
	  }
	  else {
	    printf("make %s\n",file_Dos);

            /* sawada */

            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2;

            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }

            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) {
              /* Trapezoidal rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s1 += Doss[spin][ie];
                  } 
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                } 
              } else { 
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s1 += Doss[spin][ie];
                  } 
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s2 += Doss[spin][ie];
                  } 
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                }
              }
            }         

	    for (ie=0;ie<Dos_N;ie++) {
	      for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	      for (i1=0;i1< tnoA; i1++) {
		if (LM== Spe_Num_Relation[wanA][i1]) {
		  for (spin=0;spin<=SpinP_switch;spin++) 
		    dossum[spin]+=  Dos[spin][iatom][i1][ie];
		}
	      }
	      if (SpinP_switch==1) {
		fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
                        dossum[0],-dossum[1], ssum[0][ie], ssum[1][ie]);
	      }
	      else {
		fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, dossum[0]*2.0, ssum[0][ie]*2.0);
	      }

	    }
	  }
	  if (fp_Dos) fclose(fp_Dos);
	} /* M */
      }
    }     /* L */
  }       /* iatom */

  /* atom projection */
  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    sprintf(file_Dos,"%s.%s.atom%d",basename,extension,pdos_atoms[iatom]);
    if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
      setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open %s\n",file_Dos);
    }
    else {
      printf("make %s\n",file_Dos);

      /* sawada */

            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2;

            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }

            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) {
              /* Trapezoidal rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s1 += Doss[spin][ie];
                  } 
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                } 
              } else { 
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s1 += Doss[spin][ie];
                  } 
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s2 += Doss[spin][ie];
                  } 
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                }
              }
            } 
 
      for (ie=0;ie<Dos_N;ie++) {
	for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	for (i1=0;i1< tnoA; i1++) {
	  for (spin=0;spin<=SpinP_switch;spin++)
	    dossum[spin]+=  Dos[spin][iatom][i1][ie];
	}
	if (SpinP_switch==1) {
	  fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
		  dossum[0],-dossum[1], ssum[0][ie], ssum[1][ie]);
	}
	else {
	  fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, dossum[0]*2.0, ssum[0][ie]*2.0);
	}

      } /* ie */
    } /* fp_Dos */
    if (fp_Dos) fclose(fp_Dos);
  } /* iatom */
  

  /*********************************************************
                        close and free
  *********************************************************/

  free(DosE);

  for (spin=0; spin<=SpinP_switch; spin++) {
    for (iatom=0;iatom<pdos_n;iatom++) {
      GA_AN = pdos_atoms[iatom]-1;
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i=0; i<tnoA; i++){
        free(Dos[spin][iatom][i]);
      }
      free(Dos[spin][iatom]);
    }
    free(Dos[spin]);
  }
  free(Dos);

}

void Spectra_Tetrahedron( int pdos_n, int *pdos_atoms, char *basename, int Dos_N, 
			  int knum_i,int knum_j, int knum_k,
			  int SpinP_switch, double *****EIGEN, double *******ev, int neg, 
			  double Dos_emin, double Dos_emax, 
			  int iemin, int iemax,
			  int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
                          int MaxL,int **Spe_Num_CBasis, int **Spe_Num_Relation)
{
  int spin,ieg,i,j,k,ie;
  int i_in,j_in,k_in,ic,itetra; 
  double x,result,factor; 
  double cell_e[8], tetra_e[4],tetra_a[4];
  static int tetra_id[6][4]= { {0,1,2,5}, {1,2,3,5}, {2,3,5,7},
			       {0,2,4,5}, {2,4,5,6}, {2,5,6,7} };
  int wanA,Max_tnoA,GA_AN,tnoA,i1,iatom;
  int L,M,LM;
  static char *Lname[5]={"s","p","d","f","g"};
  double dossum[2];

  double *DosE, ****Dos, rval;
  int N_Dos,i_Dos[10];
  double ***cell_a;
  int N_cell_a, i_cell_a[10];

  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* sawada */

  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N], Doss[2][Dos_N];

  static char *extension="PDOS.Tetrahedron";

  printf("<Spectra_Tetrahedron> start\n");

#if 0
  if (strlen(basename)==0) {
    strcpy(file_Dos,extension);
  }
  else {
    sprintf(file_Dos,"%s.%s",basename,extension);
  }

  if ( (fp_Dos=fopen(file_Dos,"w"))==NULL ) {

#ifdef xt3
    setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("can not open a file %s\n",file_Dos);
    return;
  }

  printf("<Spectra_Tetrahedron> make %s\n",file_Dos);
#endif

  Max_tnoA=Spe_Total_CNO[0];
  for (i=0;i<atomnum;i++) {
    wanA=WhatSpecies[i];
    if (Max_tnoA<Spe_Total_CNO[wanA]) Max_tnoA=Spe_Total_CNO[wanA];
  }



  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);

  N_Dos=4; i_Dos[0]=SpinP_switch+1; 
  i_Dos[1] = atomnum; i_Dos[2]= Max_tnoA;
  i_Dos[3]=Dos_N;
  Dos=(double****)malloc_multidimarray("double",N_Dos,i_Dos);


  N_cell_a=3; i_cell_a[0]= atomnum; i_cell_a[1]=Max_tnoA; i_cell_a[2]= 8;
  cell_a = (double***) malloc_multidimarray("double",N_cell_a,i_cell_a);
   
  /* initialize */
  for (ie=0;ie<Dos_N;ie++) {
    DosE[ie] = Dos_emin+(Dos_emax-Dos_emin)*(double)ie/(double)(Dos_N-1);
  }
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i1=0; i1<=(tnoA-1); i1++){

	for (ie=0;ie<Dos_N;ie++) {
	  Dos[spin][GA_AN][i1][ie]=0.0;
	}
      }
    }
  }

  /********************************************************************
                          Method = tetrahedron 
  *******************************************************************/

  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ieg=0;ieg<neg; ieg++) {
      for (i=0; i<=(knum_i-1); i++){
	for (j=0; j<=(knum_j-1); j++){
	  for (k=0; k<=(knum_k-1); k++){

	    for (i_in=0;i_in<2;i_in++) {
	      for (j_in=0;j_in<2;j_in++) {
		for (k_in=0;k_in<2;k_in++) {
		  cell_e[i_in*4+j_in*2+k_in] = 
		    EIGEN[spin][ (i+i_in)%knum_i ][ (j+j_in)%knum_j ][ (k+k_in)%knum_k ][ieg] ;
                  for (GA_AN=0; GA_AN<atomnum; GA_AN++){
                    wanA = WhatSpecies[GA_AN];
                    tnoA = Spe_Total_CNO[wanA];
                    for (i1=0; i1<=(tnoA-1); i1++){
                      rval= ev[spin][ (i+i_in)%knum_i ][ (j+j_in)%knum_j ][ (k+k_in)%knum_k ][GA_AN][i1][ieg];
                      cell_a[GA_AN][i1][i_in*4+j_in*2+k_in] = rval;
                    }
                  }
		}
	      }
	    }

	    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
	      wanA = WhatSpecies[GA_AN];
	      tnoA = Spe_Total_CNO[wanA];
	      for (i1=0; i1<=(tnoA-1); i1++){

		for (itetra=0;itetra<6;itetra++) {
		  for (ic=0;ic<4;ic++) {
		    tetra_e[ic]=cell_e[ tetra_id[itetra][ic] ];
		    tetra_a[ic]=cell_a[ GA_AN ][i1][ tetra_id[itetra][ic] ];
		  }
		  OrderE(tetra_e,tetra_a,4);

		  for (ie=0;ie<4;ie++) 
#if 0
 	          printf("%lf(%lf) ",tetra_e[ie],tetra_a[ie]);
		  printf("\n");
#endif

		  x = (tetra_e[0]-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1)-1.0;
		  iemin=(int)x;
		  x = (tetra_e[3]-Dos_emin)/(Dos_emax-Dos_emin)*(Dos_N-1)+1.0;
		  iemax=(int)x;
		  if (iemin<0) { iemin=0; }
		  if (iemax>=Dos_N) {iemax=Dos_N-1; }
#if 0
		  printf("%lf %lf %lf %lf\n", DosE[iemin],tetra_e[0],tetra_e[3], DosE[iemax] );
#endif
		  if ( 0<=iemin && iemin<Dos_N && 0<=iemax && iemax<Dos_N ) {
		    for (ie=iemin;ie<=iemax;ie++) {
		      ATM_Spectrum( tetra_e,tetra_a, &DosE[ie], &result);
		      Dos[spin][GA_AN][i1][ie] += result;
#if 0
		      printf("%lf-> %lf\n",DosE[ie],result);
#endif
		    }
		  }
		} /* itetra */
	      } 
	    }

	  } /* k */
	} /* j */
      } /* i */
    } /* ieg */
  } /* spin */

  /* normalize */
  factor = 1.0/(double)( eV2Hartree * knum_i * knum_j * knum_k * 6 );
  for (spin=0;spin<=SpinP_switch;spin++) {
    for (GA_AN=0; GA_AN<atomnum; GA_AN++){
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i1=0; i1<=(tnoA-1); i1++){

	for (ie=0;ie<Dos_N;ie++) {
	  Dos[spin][GA_AN][i1][ie] = Dos[spin][GA_AN][i1][ie] * factor;
	}
      }
    }
  }

  /***************************************************
                         output 
  ***************************************************/

#if 0
  if (SpinP_switch==1) {
    for (ie=0;ie<Dos_N;ie++) {
      for (sum=0.0,sum2=0.0,GA_AN=0; GA_AN<atomnum; GA_AN++){
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<=(tnoA-1); i1++){
	  sum += Dos[0][GA_AN][i1][ie];
	  sum2 += Dos[1][GA_AN][i1][ie];
	}
      }

      fprintf(fp_Dos,"%lf %lf %lf", (DosE[ie])*eV2Hartree,
	      sum, -sum2 );
    }
  } 
  else {
    for (ie=0;ie<Dos_N;ie++) {
      for (sum=0.0,sum2=0.0,GA_AN=0; GA_AN<atomnum; GA_AN++){
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<=(tnoA-1); i1++){
	  sum += Dos[0][GA_AN][i1][ie];
	}
      }

      fprintf(fp_Dos,"%lf %lf\n", (DosE[ie])*eV2Hartree, sum*2.0);
    }
  }
#else

  /*
  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i1=0;i1< tnoA; i1++) {
      printf("Spe_Num_Relation %d %d %d\n",wanA,i1,Spe_Num_Relation[wanA][i1]);
    }
  }
  */

  /* orbital projection */
  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (L=0;L<=MaxL; L++) {
      if (Spe_Num_CBasis[wanA][L]>0) {
        for (M=0;M<2*L+1;M++) {
          LM=100*L+M+1;
	  sprintf(file_Dos,"%s.%s.atom%d.%s%d",basename,extension,pdos_atoms[iatom],
		  Lname[L],M+1);
	  if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
            setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

	    printf("can not open %s\n",file_Dos);
	  }
	  else {
	    printf("make %s\n",file_Dos);

            /* sawada */
 
            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2;
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) {
              /* Trapezoidal rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s1 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                }
              } else {
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s1 += Doss[spin][ie];
                  }
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s2 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                }
              }
            }
           
	    for (ie=0;ie<Dos_N;ie++) {
	      for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	      for (i1=0;i1< tnoA; i1++) {
		/*          if (i1==4) {
			    printf("%d %d\n",LM,Spe_Num_Relation[wanA][i1]);
			    } 
		*/
		if (LM== Spe_Num_Relation[wanA][i1]) {
		  for (spin=0;spin<=SpinP_switch;spin++) 
		    dossum[spin]+=  Dos[spin][GA_AN][i1][ie];
		}
	      }
	      if (SpinP_switch==1) {
		fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", 
                        (DosE[ie])*eV2Hartree, 
			dossum[0],-dossum[1],ssum[0][ie],ssum[1][ie]);
	      }
	      else {
		fprintf(fp_Dos,"%lf %lf %lf\n", 
                        (DosE[ie])*eV2Hartree,
			dossum[0]*2.0,ssum[0][ie]*2.0);
	      }

	    }
	  }
	} /* M */
      }
    } /* L */
  } /* iatom */
 

  /* atom projection */
  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    sprintf(file_Dos,"%s.%s.atom%d",basename,extension,pdos_atoms[iatom]);
    if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
      setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open %s\n",file_Dos);
    }
    else {
      printf("make %s\n",file_Dos);

      /* sawada */
 
            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2;
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) {
              /* Trapezoidal rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s1 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                }
              } else {
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s1 += Doss[spin][ie];
                  }
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s2 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                }
              }
            }
  
      for (ie=0;ie<Dos_N;ie++) {
	for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	for (i1=0;i1< tnoA; i1++) {
	  for (spin=0;spin<=SpinP_switch;spin++)
	    dossum[spin]+=  Dos[spin][GA_AN][i1][ie];
	}
	if (SpinP_switch==1) {
	  fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
		  dossum[0],-dossum[1],ssum[0][ie],ssum[1][ie]);
	}
	else {
	  fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree,
		  dossum[0]*2.0, ssum[0][ie]*2.0);
	}

      } /* ie */
    } /* fp_Dos */
  } /* iatom */



#endif


  /*********************************************************
                       close and free
  *********************************************************/

#if 0
  if (fp_Dos) fclose(fp_Dos);
#endif

  free(DosE);

  for (spin=0; spin<i_Dos[0]; spin++) {
    for (iatom=0; iatom<i_Dos[1]; iatom++) {
      for (i=0; i<i_Dos[2]; i++){
        free(Dos[spin][iatom][i]);
      }
      free(Dos[spin][iatom]);
    }
    free(Dos[spin]);
  }
  free(Dos);

  for (i=0; i<i_cell_a[0]; i++){
    for (j=0; j<i_cell_a[1]; j++){
      free(cell_a[i][j]);
    }
    free(cell_a[i]);
  }
  free(cell_a);

}
   




void Spectra_NEGF( int pdos_n, int *pdos_atoms, char *basename, char *extension, int Dos_N, 
		   int SpinP_switch, double *****EIGEN, double *******ev, int neg,
		   double Dos_emin, double Dos_emax,
		   int iemin, int iemax,
		   int atomnum, int *WhatSpecies, int *Spe_Total_CNO,
		   int MaxL, int **Spe_Num_CBasis, int **Spe_Num_Relation)
{
  /*****************************************************************
                               Method = NEGF
  *****************************************************************/

  int Max_tnoA;
  int i,spin,j,k;
  int wanA, GA_AN,tnoA,i1,ieg;
  double eg,x,rval,xa;
  double *DosE, ****Dos;
  int N_Dos, i_Dos[10];
  double pi2,factor;
  int  iewidth,ie,iecenter;
  int iatom,L,M,LM;
  static char *Lname[5]={"s","p","d","f","g"};
  double dossum[2],MulP[5],dE;

  char file_Dos[YOUSO10];
  FILE *fp_Dos;
  char fp_buf[fp_bsize];          /* setvbuf */

  /* sawada */

  int q, sw;
  double h,s1,s2;
  double ssum[2][Dos_N], Doss[2][Dos_N];

  printf("<Spectra_%s> start\n",extension);

  Max_tnoA=0;
  for (i=0; i<atomnum; i++) {
    wanA=WhatSpecies[i];
    if (Max_tnoA<Spe_Total_CNO[wanA]) Max_tnoA=Spe_Total_CNO[wanA];
  }
  /* allocation */
  DosE=(double*)malloc(sizeof(double)*Dos_N);

  N_Dos=4; i_Dos[0]=SpinP_switch+1; 
  i_Dos[1] = atomnum; i_Dos[2]= Max_tnoA;
  i_Dos[3]=Dos_N;
  Dos=(double****)malloc_multidimarray("double",N_Dos,i_Dos);

  /* note: Dos_N == neg */ 

  /* initialize */

  for (ie=0; ie<Dos_N; ie++) {
    DosE[ie] = EIGEN[0][0][0][0][ie];
  }

  /********************************************************************
                             Method = NEGF
  *******************************************************************/

  for (spin=0;spin<=SpinP_switch;spin++) {
    for (ieg=0; ieg<neg; ieg++) {
      for (GA_AN=0; GA_AN<atomnum; GA_AN++){
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<tnoA; i1++){
	  rval = ev[spin][0][0][0][GA_AN][i1][ieg];
          Dos[spin][GA_AN][i1][ieg] = rval/eV2Hartree;
	} /* i1 */
      } /* GA_AN */
    } /* ieg */
  } /* spin */

  /* orbital projection */

  for (iatom=0; iatom<pdos_n; iatom++) {

    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    for (L=0;L<=MaxL; L++) {
      if (Spe_Num_CBasis[wanA][L]>0) {
        for (M=0;M<2*L+1;M++) {
          LM=100*L+M+1;

	  sprintf(file_Dos,"%s.PDOS.%s.atom%d.%s%d",
                  basename,extension,pdos_atoms[iatom],
		  Lname[L],M+1);

	  if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
            setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif
	    printf("can not open %s\n",file_Dos);
	  }

	  else {
	    printf("make %s\n",file_Dos);

            /* sawada */
 
            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2;
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) {
              /* Trapezoidal rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s1 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                }
              } else {
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s1 += Doss[spin][ie];
                  }
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      if (LM== Spe_Num_Relation[wanA][i1]) {
                        Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                      }
                    }
                    s2 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                }
              }
            }
            
	    for (ie=0;ie<Dos_N;ie++) {
	      for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	      for (i1=0;i1< tnoA; i1++) {
		if (LM== Spe_Num_Relation[wanA][i1]) {
		  for (spin=0;spin<=SpinP_switch;spin++) {
		    dossum[spin]+=  Dos[spin][GA_AN][i1][ie];
                  }
		}
	      }
	      if (SpinP_switch==1) {
		fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
                        dossum[0],-dossum[1],ssum[0][ie],ssum[1][ie]);
	      }
	      else {
		fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, dossum[0]*2.0, ssum[0][ie]*2.0);
	      }

	    }
	  }
	  if (fp_Dos) fclose(fp_Dos);
	} /* M */
      }
    } /* L */
  } /* iatom */

  /* atom projection */

  for (iatom=0;iatom<pdos_n;iatom++) {
    GA_AN = pdos_atoms[iatom]-1;
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    sprintf(file_Dos,"%s.PDOS.%s.atom%d",basename,extension,pdos_atoms[iatom]);
    if ( (fp_Dos=fopen(file_Dos,"w")) ==NULL )  {

#ifdef xt3
      setvbuf(fp_Dos,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open %s\n",file_Dos);
    }
    else {

      printf("make %s\n",file_Dos);

      /*
      for (spin=0; spin<=SpinP_switch; spin++){
        MulP[spin] = 0.0;
      }
      */

      /* sawada */
 
            h = (DosE[Dos_N-1] - DosE[0])/(Dos_N-1)*eV2Hartree;
            sw = 2;
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              for (ie=1;ie<Dos_N;ie++) {
                Doss[spin][ie] = 0.0;
              }
            }
 
            for (spin=0;spin<=SpinP_switch;spin++) {
              if (sw == 1) {
              /* Trapezoidal rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  for (ie=1;ie<q;ie++) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s1 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (0.5*(Doss[spin][0]+Doss[spin][q])+s1)*h;
                }
              } else {
              /* Simpson's rule */
                for (q=0;q<Dos_N;q++) {
                  s1 = 0.0;
                  s2 = 0.0;
                  for (ie=1;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s1 += Doss[spin][ie];
                  }
                  for (ie=2;ie<q;ie+=2) {
                    Doss[spin][ie] = 0.0;
                    for (i1=0;i1< tnoA; i1++) {
                      Doss[spin][ie]+=  Dos[spin][GA_AN][i1][ie];
                    }
                    s2 += Doss[spin][ie];
                  }
                  ssum[spin][q] = (Doss[spin][0]+4.0*s1+2.0*s2+Doss[spin][q])*h/3.0;
                }
              }
            }

      for (ie=0;ie<Dos_N;ie++) {
	for (spin=0;spin<=SpinP_switch;spin++) dossum[spin]=0.0;
	for (i1=0;i1< tnoA; i1++) {
	  for (spin=0;spin<=SpinP_switch;spin++)
	    dossum[spin]+=  Dos[spin][GA_AN][i1][ie];
	}

        /* integration */
	/*
        if (DosE[ie]<=0.00){
          dE = DosE[1] - DosE[0];
          for (spin=0; spin<=SpinP_switch; spin++){
            MulP[spin] += dossum[spin]*dE;
          }
	}
	*/

	if (SpinP_switch==1) {
	  fprintf(fp_Dos,"%lf %lf %lf %lf %lf\n", (DosE[ie])*eV2Hartree,
		  dossum[0],-dossum[1],ssum[0][ie],ssum[1][ie]);
	}
	else {
	  fprintf(fp_Dos,"%lf %lf %lf\n", (DosE[ie])*eV2Hartree, dossum[0]*2.0, ssum[0][ie]*2.0);
	}

      } /* ie */
    } /* fp_Dos */
    if (fp_Dos) fclose(fp_Dos);

    /*
    for (spin=0; spin<=SpinP_switch; spin++){
      printf("spin=%2d MulP=%15.12f\n",spin,MulP[spin]*eV2Hartree);
    }
    */

  } /* iatom */
  

  /*********************************************************
                        close and free
  *********************************************************/

  free(DosE);

  for (i=0; i<i_Dos[0]; i++){
    for (j=0; j<i_Dos[1]; j++){
      for (k=0; k<i_Dos[2]; k++){
        free(Dos[i][j][k]);
      }
      free(Dos[i][j]);
    }
    free(Dos[i]);
  }
  free(Dos);
}





/* input    int SpeciesNum,int MaxL,int *Spe_Total_CNO,int **Spe_Num_CBasis,
   output Spe_Num_Relation
*/
void  Spe_Num_CBasis2Relation(
			      int SpeciesNum,int MaxL,int *Spe_Total_CNO,int **Spe_Num_CBasis,
			      int **Spe_Num_Relation)
{
  int i,j,k,l,id;

  for (i=0;i<SpeciesNum;i++) {
    id=0;
    for (j=0;j<=MaxL;j++) {
      if (Spe_Num_CBasis[i][j]>0) {
	for (k=0;k<Spe_Num_CBasis[i][j];k++) {
	  for (l=0;l<2*j+1;l++) {
	    Spe_Num_Relation[i][id++]=100*j+l+1;
	  }
	}
      } 
    } 
    for (j=0;j<Spe_Total_CNO[i];j++) {
      printf("%d ",Spe_Num_Relation[i][j]);
    }
    printf("\n");
  }


}





void input_file_eg(char *file_eg,
		   int *mode, int *NonCol,
                   int *n, double *emin, double *emax, int *iemin,int *iemax, 
		   int *SpinP_switch, int Kgrid[3], int *atomnum, int **WhatSpecies,
		   int *SpeciesNum, int **Spe_Total_CNO, int * Max_tnoA, int *MaxL,
		   int ***Spe_Num_CBasis, int ***Spe_Num_Relation, double *ChemP, 
                   double **Angle0_Spin, double **Angle1_Spin,
		   double ******ko)
{
  double r_vec[10], r_vec2[10];
  int i_vec[10], i_vec2[10];
  FILE *fp;
  int N_Spe_Num_CBasis, i_Spe_Num_CBasis[10];
  int j,k;
  int N_Spe_Num_Relation, i_Spe_Num_Relation[10];
  int N_ko, i_ko[10];
  int i,i1,j1,k1,ie,spin;

  input_open(file_eg);

  /*****************************************
   mode

    1: normal for cluster and band   
    2: 
    3: 
    4:
    5: DC, GDC, and Krylov
    6: NEGF
    7: Gaussian DOS for collinear case
  *****************************************/

  input_int("mode",mode,1);

  /*****************************************
   NonCol

    0: collinear calc.
    1: non-collinear calc. 
  *****************************************/

  input_int("NonCol",NonCol,0);

  /*****************************************
          In case of divide-conquer
  *****************************************/

  if (*mode==5){

    input_int("Nspin",SpinP_switch,0);

    r_vec2[0]=0.0; r_vec2[1]=0.0;
    input_doublev("Erange",2,r_vec,r_vec2);
    *emin=r_vec[0]; *emax=r_vec[1];

    input_int("atomnum",atomnum,0);

    *WhatSpecies = (int*)malloc(sizeof(int)*(*atomnum));
    if (fp=input_find("<WhatSpecies") ) {
      for (i=0;i<*atomnum;i++) {
	fscanf( fp, "%d", &(*WhatSpecies)[i]);
      }
      if (!input_last("WhatSpecies>")) {
	printf("format error near WhatSpecies>\n");
	exit(10);
      }
    }
    else {
      printf("<WhatSpecies is not found\n");
      exit(10);
    }

    input_int("SpeciesNum",SpeciesNum,0);
    *Spe_Total_CNO= (int*)malloc(sizeof(int)*(*SpeciesNum));
    if (fp=input_find("<Spe_Total_CNO") ) {
      for (i=0;i<*SpeciesNum;i++) {
	fscanf( fp, "%d", &(*Spe_Total_CNO)[i]);
      }
      if (!input_last("Spe_Total_CNO>")) {
	printf("format error near Spe_Total_CNO>\n");
	exit(10);
      }
    }
    else {
      printf("<Spe_Total_CNO is not found\n");
      exit(10);
    }

    *Max_tnoA = (*Spe_Total_CNO)[0];
    for (i=1;i<*SpeciesNum;i++) {
      if ( *Max_tnoA  < (*Spe_Total_CNO)[i] ) *Max_tnoA=(*Spe_Total_CNO)[i];
    }
    printf("Max of Spe_Total_CNO = %d\n",*Max_tnoA);

    input_int("MaxL",MaxL,4);
    N_Spe_Num_CBasis=2; i_Spe_Num_CBasis[0]=*SpeciesNum; i_Spe_Num_CBasis[1]=*MaxL+1;
    *Spe_Num_CBasis=(int**)malloc_multidimarray("int",N_Spe_Num_CBasis, i_Spe_Num_CBasis);
    if (fp=input_find("<Spe_Num_CBasis") ) {
      for (i=0;i<*SpeciesNum;i++) {
	for (j=0;j<=*MaxL;j++) {
	  fscanf(fp,"%d",&(*Spe_Num_CBasis)[i][j]);
	}
      }
      if (!input_last("Spe_Num_CBasis>")) {
	printf("format error near Spe_Num_CBasis>\n");
	exit(10);
      }
    }
    else {
      printf("<Spe_Num_CBasis is not found\n");
      exit(10);
    }

    /* relation, index to orbital, 0->s, 1->p ... and so on. */
    N_Spe_Num_Relation=2; i_Spe_Num_Relation[0]=*SpeciesNum; i_Spe_Num_Relation[1]=*Max_tnoA;
    *Spe_Num_Relation=(int**)malloc_multidimarray("int",N_Spe_Num_Relation, i_Spe_Num_Relation);
    Spe_Num_CBasis2Relation(*SpeciesNum,*MaxL,*Spe_Total_CNO,*Spe_Num_CBasis, *Spe_Num_Relation);

    input_double("ChemP",ChemP,0.0);

    if (*NonCol==1){
      *Angle0_Spin = (double*)malloc(sizeof(double)*(*atomnum));
      *Angle1_Spin = (double*)malloc(sizeof(double)*(*atomnum));
      if (fp=input_find("<SpinAngle") ) {
        for (i=0; i<*atomnum; i++) {
	  fscanf( fp, "%lf %lf", &(*Angle0_Spin)[i],&(*Angle0_Spin)[i]);
	}
	if (!input_last("SpinAngle>")) {
	  printf("format error near SpinAngle>\n");
	  exit(10);
	}
      }
      else {
	printf("<SpinAngle is not found\n");
	exit(10);
      }
    }
  }

  /*****************************************
         in case of cluster and band
  *****************************************/

  else {

    input_int("N",n,0);
    r_vec2[0]=0.0; r_vec2[1]=0.0;
    input_doublev("Erange",2,r_vec,r_vec2);

    *emin=r_vec[0]; *emax=r_vec[1];
    i_vec2[0]=0; i_vec2[1]=0; i_vec2[2]=0;

    input_intv("irange" , 2,i_vec,i_vec2);
    *iemin=i_vec[0];  *iemax= i_vec[1];

    input_int("Nspin",SpinP_switch,0);

    i_vec2[0]=i_vec2[1]=i_vec2[2]=0;
    input_intv("Kgrid",3, Kgrid,i_vec2);
  
    input_int("atomnum",atomnum,0);
    *WhatSpecies = (int*)malloc(sizeof(int)* (*atomnum));
    if (fp=input_find("<WhatSpecies") ) {
      for (i=0;i<*atomnum;i++) {
	fscanf( fp, "%d", &(*WhatSpecies)[i]);
      }
      if (!input_last("WhatSpecies>")) {
	printf("format error near WhatSpecies>\n");
	exit(10);
      }
    }
    else {
      printf("<WhatSpecies is not found\n");
      exit(10);
    }
  
    input_int("SpeciesNum",SpeciesNum,0);
    *Spe_Total_CNO= (int*)malloc(sizeof(int)*(*SpeciesNum));
    if (fp=input_find("<Spe_Total_CNO") ) {
      for (i=0;i<*SpeciesNum;i++) {
	fscanf( fp, "%d", &(*Spe_Total_CNO)[i]);
      }
      if (!input_last("Spe_Total_CNO>")) {
	printf("format error near Spe_Total_CNO>\n");
	exit(10);
      }
    }
    else {
      printf("<Spe_Total_CNO is not found\n");
      exit(10);
    }

    *Max_tnoA = (*Spe_Total_CNO)[0];
    for (i=1;i<*SpeciesNum;i++) {
      if ( *Max_tnoA  < (*Spe_Total_CNO)[i] ) *Max_tnoA=(*Spe_Total_CNO)[i];
    }
    printf("Max of Spe_Total_CNO = %d\n",*Max_tnoA);

    input_int("MaxL",MaxL,4);
    N_Spe_Num_CBasis=2; i_Spe_Num_CBasis[0]=*SpeciesNum; i_Spe_Num_CBasis[1]=*MaxL+1;
    *Spe_Num_CBasis=(int**)malloc_multidimarray("int",N_Spe_Num_CBasis, i_Spe_Num_CBasis);
    if (fp=input_find("<Spe_Num_CBasis") ) {
      for (i=0;i<*SpeciesNum;i++) {
	for (j=0;j<=*MaxL;j++) {
	  fscanf(fp,"%d",&(*Spe_Num_CBasis)[i][j]);
	}
      }
      if (!input_last("Spe_Num_CBasis>")) {
	printf("format error near Spe_Num_CBasis>\n");
	exit(10);
      }
    }
    else {
      printf("<Spe_Num_CBasis is not found\n");
      exit(10);
    }

    /* relation, index to orbital, 0->s, 1->p ... and so on. */
    N_Spe_Num_Relation=2; i_Spe_Num_Relation[0]=*SpeciesNum; i_Spe_Num_Relation[1]=*Max_tnoA;
    *Spe_Num_Relation=(int**)malloc_multidimarray("int",N_Spe_Num_Relation, i_Spe_Num_Relation);
    Spe_Num_CBasis2Relation(*SpeciesNum,*MaxL,*Spe_Total_CNO,*Spe_Num_CBasis, *Spe_Num_Relation);
  
    input_double("ChemP",ChemP,0.0);

    if (*NonCol==1){
      *Angle0_Spin = (double*)malloc(sizeof(double)*(*atomnum));
      *Angle1_Spin = (double*)malloc(sizeof(double)*(*atomnum));
      if (fp=input_find("<SpinAngle") ) {
        for (i=0; i<*atomnum; i++) {
	  fscanf( fp, "%lf %lf", &(*Angle0_Spin)[i],&(*Angle0_Spin)[i]);
	}
	if (!input_last("SpinAngle>")) {
	  printf("format error near SpinAngle>\n");
	  exit(10);
	}
      }
      else {
	printf("<SpinAngle is not found\n");
	exit(10);
      }
    }

    if (fp=input_find("<Eigenvalues") ) {
      N_ko=5; 
      i=0; i_ko[i++]=*SpinP_switch+1; 
      i_ko[i++]=Kgrid[0]; i_ko[i++]=Kgrid[1]; i_ko[i++]=Kgrid[2]; i_ko[i++]= *iemax-*iemin+1; 
      *ko=(double*****)malloc_multidimarray("double", N_ko,i_ko);

      for (i=0;i<Kgrid[0];i++) {
	for (j=0;j<Kgrid[1];j++) {
	  for (k=0;k<Kgrid[2];k++) {
	    for (spin=0;spin<=*SpinP_switch;spin++) {
	      fscanf(fp,"%d%d%d",&i1,&j1,&k1); 

	      for (ie=*iemin;ie<=*iemax;ie++) {
		fscanf(fp,"%lf",&(*ko)[spin][i1][j1][k1][ie-*iemin]);
		/*   printf("%lf ",ko[spin][i1][j1][k1][ie-iemin]); */
		(*ko)[spin][i1][j1][k1][ie-*iemin]=(*ko)[spin][i1][j1][k1][ie-*iemin]-*ChemP;

	      }
	    } /* spin */
	  } /* k*/
	} /* j */
      } /* i */
      if (!input_last("Eigenvalues>")) {
	printf("format error near Eigenvalues>\n");
	exit(10);
      }
    } /* if */
    else {
      printf("can not find <Eigenvalues\n");
      exit(10);
    }
  }

  input_close();
}






void input_file_ev( char *file_ev, int mode, int NonCol, int Max_tnoA, 
		    int Kgrid[3], int SpinP_switch, int iemin, int iemax, 
                    int atomnum,  int *WhatSpecies, int *Spe_Total_CNO, 
		    double ********ev, double ChemP)

{ 
  int N_ev, i_ev[10];
  int i,j,k,spin,ie,i1,j1,k1,n;
  int ii,jj,kk;
  int GA_AN, wanA, tnoA;
  int i_vec[10];
  float *fSD;
  double sum;
  char keyword[YOUSO10];
  char fp_buf[fp_bsize];          /* setvbuf */
  FILE *fp; 
  
  /*****************************************
          In case of divide-conquer
  *****************************************/
    
  if (mode==5){

    input_open(file_ev);

    Msize1 = (int**)malloc(sizeof(int*)*(SpinP_switch+1));
    EigenE = (double***)malloc(sizeof(double**)*(SpinP_switch+1));
    PDOS = (double****)malloc(sizeof(double***)*(SpinP_switch+1));

    if (NonCol==0){

      for (spin=0; spin<=SpinP_switch; spin++){

	Msize1[spin] = (int*)malloc(sizeof(int)*atomnum);
	EigenE[spin] = (double**)malloc(sizeof(double*)*atomnum);
	PDOS[spin] = (double***)malloc(sizeof(double**)*atomnum);

	for (GA_AN=0; GA_AN<atomnum; GA_AN++){

	  printf("read GA_AN=%4d\n",GA_AN);

	  sprintf(keyword,"<AN%dAN%d",GA_AN+1,spin);

	  if (fp=input_find(keyword) ) {
	    wanA = WhatSpecies[GA_AN];
	    tnoA = Spe_Total_CNO[wanA];

	    fscanf(fp,"%d",&Msize1[spin][GA_AN]);

	    EigenE[spin][GA_AN] = (double*)malloc(sizeof(double)*Msize1[spin][GA_AN]);
	    PDOS[spin][GA_AN] = (double**)malloc(sizeof(double*)*Msize1[spin][GA_AN]);

	    for (n=0; n<Msize1[spin][GA_AN]; n++){

	      PDOS[spin][GA_AN][n] = (double*)malloc(sizeof(double)*tnoA);

	      fscanf(fp,"%d", &j);
	      fscanf(fp,"%lf",&EigenE[spin][GA_AN][n]);
	      EigenE[spin][GA_AN][n] = EigenE[spin][GA_AN][n] - ChemP;

	      for (i=0; i<tnoA; i++) {
		fscanf(fp,"%lf",&PDOS[spin][GA_AN][n][i]);
	      }
	    }

	    sprintf(keyword,"AN%dAN%d>",GA_AN+1,spin);
	    if (!input_last(keyword)) {
	      printf("format error near AN%dAN>\n",GA_AN+1);
	      exit(10);
	    }

	  }
	  else {
	    printf("<AN%dAN is not found\n",GA_AN+1);
	    exit(10);
	  }
	}
      }
    }

    else{

      Msize1[0] = (int*)malloc(sizeof(int)*atomnum);
      Msize1[1] = (int*)malloc(sizeof(int)*atomnum);
      EigenE[0] = (double**)malloc(sizeof(double*)*atomnum);
      EigenE[1] = (double**)malloc(sizeof(double*)*atomnum);
      PDOS[0] = (double***)malloc(sizeof(double**)*atomnum);
      PDOS[1] = (double***)malloc(sizeof(double**)*atomnum);


      for (GA_AN=0; GA_AN<atomnum; GA_AN++){

	printf("read GA_AN=%4d\n",GA_AN);

	sprintf(keyword,"<AN%d",GA_AN+1);

	if (fp=input_find(keyword) ) {

	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];

	  fscanf(fp,"%d %d",&Msize1[0][GA_AN],&Msize1[1][GA_AN]);

	  EigenE[0][GA_AN] = (double*)malloc(sizeof(double)*Msize1[0][GA_AN]);
	  EigenE[1][GA_AN] = (double*)malloc(sizeof(double)*Msize1[1][GA_AN]);
	  PDOS[0][GA_AN] = (double**)malloc(sizeof(double*)*Msize1[0][GA_AN]);
	  PDOS[1][GA_AN] = (double**)malloc(sizeof(double*)*Msize1[1][GA_AN]);

	  for (n=0; n<Msize1[0][GA_AN]; n++){

	    PDOS[0][GA_AN][n] = (double*)malloc(sizeof(double)*tnoA);
	    PDOS[1][GA_AN][n] = (double*)malloc(sizeof(double)*tnoA);

	    fscanf(fp,"%d", &j);
	    fscanf(fp,"%lf %lf",&EigenE[0][GA_AN][n],&EigenE[1][GA_AN][n]);
	    EigenE[0][GA_AN][n] = EigenE[0][GA_AN][n] - ChemP;
	    EigenE[1][GA_AN][n] = EigenE[1][GA_AN][n] - ChemP;

	    for (i=0; i<tnoA; i++) {
	      fscanf(fp,"%lf %lf",&PDOS[0][GA_AN][n][i],&PDOS[1][GA_AN][n][i]);
	    }
	  }

	  sprintf(keyword,"AN%d>",GA_AN+1);
	  if (!input_last(keyword)) {
	    printf("format error near AN%dAN>\n",GA_AN+1);
	    exit(10);
	  }

	}
	else {
	  printf("<AN%dAN is not found\n",GA_AN+1);
	  exit(10);
	}
      }

    }

    input_close();
  
  }  
  
  /*****************************************
         In case of cluster and band
  *****************************************/

  else {

    if ( (fp=fopen(file_ev,"r"))==NULL ) {

#ifdef xt3
      setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("cannot open a file %s\n", file_ev);
      exit(10);
    }

    N_ev=7;
    i=0; i_ev[i++]=SpinP_switch+1;
    i_ev[i++]=Kgrid[0]; i_ev[i++]=Kgrid[1]; i_ev[i++]=Kgrid[2]; 
    i_ev[i++] = atomnum ;  i_ev[i++] = Max_tnoA;
    i_ev[i++]= iemax-iemin+1;
    (*ev)=(double*******)malloc_multidimarray("double", N_ev,i_ev);

    fSD=(float*)malloc(sizeof(float)*Max_tnoA*2);

    if (NonCol==0){

      for (i=0;i<Kgrid[0];i++) {
	for (j=0;j<Kgrid[1];j++) {
	  for (k=0;k<Kgrid[2];k++) {
	    for (spin=0;spin<=SpinP_switch;spin++) {
	      for (ie=iemin;ie<=iemax;ie++) {

		fread(i_vec,sizeof(int),3,fp);
		ii=i_vec[0]; jj=i_vec[1]; kk=i_vec[2];

		/*
		  This part was generalized for the parallel calculation. 
		  if (ii!=i || jj!=j || kk!=k) {
		  printf("parameter error in reading Eigenvectors\n");
		  exit(10);
		  }
		*/

		for (GA_AN=0; GA_AN<atomnum; GA_AN++){
		  wanA = WhatSpecies[GA_AN];
		  tnoA = Spe_Total_CNO[wanA];

		  fread(fSD,sizeof(float),tnoA,fp);
		  for (i1=0; i1<=(tnoA-1); i1++){
		    (*ev)[spin][ii][jj][kk][GA_AN][i1][ie-iemin] = fSD[i1];
		  }
		}

		/*   printf("\n"); */
		/* check normalization */
		sum=0.0;
		for (GA_AN=0; GA_AN<atomnum; GA_AN++){
		  wanA = WhatSpecies[GA_AN];
		  tnoA = Spe_Total_CNO[wanA];
		  for (i1=0; i1<=(tnoA-1); i1++){
		    sum+= (*ev)[spin][ii][jj][kk][GA_AN][i1][ie-iemin];
		  }
		}

		/*
		  if (fabs(sum-1.0) > 1.0e-3) {
		  printf("%d %d %d %d %d, sum=%lf\n",spin,ii,jj,kk,ie,sum);
		  }
		*/

	      }
	    }
	  }
	}
      }
    }
 
    else{

      for (i=0;i<Kgrid[0];i++) {
	for (j=0;j<Kgrid[1];j++) {
	  for (k=0;k<Kgrid[2];k++) {

	    for (ie=iemin; ie<=iemax; ie++) {

	      fread(i_vec,sizeof(int),3,fp);
	      ii=i_vec[0]; jj=i_vec[1]; kk=i_vec[2];

	      /*
		This part was generalized for the parallel calculation. 
		if (ii!=i || jj!=j || kk!=k) {
		printf("parameter error in reading Eigenvectors\n");
		exit(10);
		}
	      */

	      for (GA_AN=0; GA_AN<atomnum; GA_AN++){
		wanA = WhatSpecies[GA_AN];
		tnoA = Spe_Total_CNO[wanA];

		fread(fSD,sizeof(float),2*tnoA,fp);

		for (i1=0; i1<tnoA; i1++){
		  (*ev)[0][ii][jj][kk][GA_AN][i1][ie-iemin] = fSD[i1];
		}

		for (i1=0; i1<tnoA; i1++){
		  (*ev)[1][ii][jj][kk][GA_AN][i1][ie-iemin] = fSD[tnoA+i1];
		}
	      }

	      /*   printf("\n"); */
	      /* check normalization */
	      /*
	      sum=0.0;
	      for (GA_AN=0; GA_AN<atomnum; GA_AN++){
		wanA = WhatSpecies[GA_AN];
		tnoA = Spe_Total_CNO[wanA];
		for (i1=0; i1<=(tnoA-1); i1++){
		  sum+= (*ev)[0][ii][jj][kk][GA_AN][i1][ie-iemin];
		}
	      }
	      */

	      /*
		if (fabs(sum-1.0) > 1.0e-3) {
		printf("%d %d %d %d %d, sum=%lf\n",spin,ii,jj,kk,ie,sum);
		}
	      */

	    }
	  }
	}
      }
    }

    free(fSD);
    fclose(fp);
  }


}



void input_main( int mode, int Kgrid[3], int atomnum, 
		 int *method, int *todo, 
		 double *gaussian, 
		 int *pdos_n, int **pdos_atoms)
{
  char buf[1000],buf2[1000],*c;
  int i;

  if (mode==5){
    printf("The input data caluculated by the divide-conquer method\n"); 
    printf("Gaussian Broadening is employed\n"); 
    *method=2;
  }

  else if (mode==6){
    printf("The input data caluculated by the NEGF method\n"); 
    *method=4;
  }

  else if (mode==7){
    printf("The input data caluculated by the Gaussian broadening method\n"); 
    *method=4;
  }

  else{
    if (Kgrid[0]==1 && Kgrid[1]==1 && Kgrid[2]==1 ) {
      printf("Kgrid= 1 1 1 => Gaussian Broadening is employed\n");
      *method=2;
    }
    else {
      printf("Which method do you use?, Tetrahedron(1), Gaussian Broadening(2)\n"); 
      fgets(buf,1000,stdin); sscanf(buf,"%d",method);
    }
  }

  if (*method==2) {
    printf("Please input a value of gaussian (double) (eV)\n"); 
    fgets(buf,1000,stdin); sscanf(buf,"%lf",gaussian); *gaussian = *gaussian/eV2Hartree; 
  }
    
  printf("Do you want Dos(1) or PDos(2)?\n");
  fgets(buf,1000,stdin);sscanf(buf,"%d",todo);
  
  if (*todo==2) {
    printf("\nNumber of atoms=%d\n",atomnum);
    if (atomnum==1) {
      (*pdos_atoms)=(int*)malloc(sizeof(int)*1);
      *pdos_n=1;
      (*pdos_atoms)[0]=1;
    }
    else {
      printf("Which atoms for PDOS : (1,...,%d), ex 1 2\n",atomnum);
      fgets(buf,1000,stdin);
      strcpy(buf2,buf);
      /*	printf("<%s>\n",buf); */
      *pdos_n=0;
      c=strtok(buf," ")  ;
      while ( c  ) {
	/*  printf("<%s> ",c); */
	if (sscanf(c,"%d",&i)==1)    (*pdos_n)++;
	c=strtok(NULL," ");
      }
      /*  printf("pdos_n=%d\n",*pdos_n); */
      *pdos_atoms=(int*)malloc(sizeof(int)*(*pdos_n));
      for (c=strtok(buf2," ") , i=0;i<*pdos_n;c=strtok(NULL," "),i++)  {
	/*  printf("<%s> ",c); */
	sscanf(c,"%d",&(*pdos_atoms)[i]);
      }
    }
    printf("pdos_n=%d\n",*pdos_n);
    for (i=0;i<*pdos_n;i++) {
      printf("%d ",(*pdos_atoms)[i]);
    }
    printf("\n");
  }

}
