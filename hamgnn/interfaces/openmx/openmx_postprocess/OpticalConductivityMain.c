/**********************************************************************
  OpticalConductivityMain.c:

     OpticalConductivityMain.c is a program code for calculating
     optical conductivity of molecules within LDA or GGA.

   usage:   thisprogram file.optical file.Dos.val file.opticaloutput

  Log of OpticalConductivityMain.c:

     1/Apr./2004  Released by H.Kino

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Inputtools.h"

#define fp_bsize         1048576     /* buffer size for setvbuf */

void *malloc_multidimarray(char *type, int N, int *size); 

#define YOUSO10  500
#define eV2Hartree 27.2113845
#define PI 3.1415926535897932384626

int *iemin0, *iemax0; 
int iemin, iemax; 

double ****Jij; 
int N_Jij=4, i_Jij[4];

double **eig;
int N_eig=2, i_eig[2];

int     nspin; 
double  gaussian; /* gaussian broadening */
double  Temp;     /* temperature */
int     Nbin;     /* # of dos */

double  ****dos; 
int N_dos, i_dos[10];
double *ene; 
double freqmax; 


void file_read(char *fileopt,char *fileeig)
{
   double erange[2],d_vec[20];
   char buf[YOUSO10];
   int spin;
   int i_vec[20],i_vec2[20];
   int ie,ia,ib,iaxes;
   int i1,i2,i3;
   FILE *fp;
   int NJ;

   input_open(fileopt);
   input_int("nspin",&nspin,0);

   iemin0=(int*)malloc(sizeof(int)*(nspin+1));
   iemax0=(int*)malloc(sizeof(int)*(nspin+1));


#if 0
   d_vec[0]=d_vec[1]=0.0;
   input_doublev("erange",2,erange, d_vec);
   if ( -erange[0] > erange[1] )  { freqmax=erange[1]; }
   else { freqmax=-erange[0]; }
   printf(" # freqmax=%lf\n",freqmax); 
#else
 
  erange[0]= -100.;
  erange[1]=  100.;
  freqmax=100.;
#endif
   printf(" # freqmax=%lf\n",freqmax);

   input_int("N", &NJ,0);
   for (spin=0;spin<=nspin;spin++) {

#if 0
     sprintf(buf,"ierange.spin=%d",spin);
     i_vec2[0]=0; i_vec2[0]=0; 
     input_intv(buf, 2, i_vec, i_vec2);
     iemin0[spin]=i_vec[0];
     iemax0[spin]=i_vec[1];
#else
     iemin0[spin]=1;
     iemax0[spin]=NJ;
#endif

   }

   /* for max. size of iemax-iemin */
   iemin =  iemin0[0];
   for (spin=0;spin<=nspin;spin++) {
      if (iemin > iemin0[spin] ) iemin = iemin0[spin];
   }
   iemax= iemax0[0];
   for (spin=0;spin<=nspin;spin++) {
      if (iemax < iemax0[spin] ) iemax = iemax0[spin];
   }

   
   N_Jij=4;
   i_Jij[0]=nspin+1; i_Jij[1]=iemax-iemin+1; i_Jij[2]=i_Jij[1]; i_Jij[3]=3;
   Jij=(double****)malloc_multidimarray("double",N_Jij,i_Jij);


   N_eig=2;
   i_eig[0]=nspin+1; i_eig[1]=iemax-iemin+1; 
   eig=(double**)malloc_multidimarray("double",N_eig, i_eig); 

   for (spin=0; spin<=nspin; spin++)  {
   
#if 0
    sprintf(buf,"<eig.spin=%d",spin);
        
    if (fp=input_find(buf) ) {
      for (ie= 0; ie<=iemax0[spin]-iemin0[spin]; ie++) {
        fscanf(fp,"%lf", &eig[spin][ie]);
      }
    }
#endif

    for (iaxes=0;iaxes<3;iaxes++) {
      sprintf(buf,"<barJ.spin=%d.axis=%d",spin,iaxes+1);
      if (fp=input_find(buf) ) {
        for (ia=0;ia<=iemax0[spin]-iemin0[spin];ia++) {
         for (ib=0;ib<=iemax0[spin]-iemin0[spin];ib++) { 
           fscanf(fp,"%le",&Jij[spin][ia][ib][iaxes]);
         }
        }
      }  /* if fp=input_find() */
      else {
        printf("can not find the keyword %s\n",buf);
        exit(0);
      }
    }  /* for iaxes */

  } /* for spin */

  input_close();

   /**************************************************/

  input_open(fileeig);
  sprintf(buf,"<Eigenvalues");
  if (fp=input_find(buf) ) {
    for (spin=0; spin<=nspin; spin++) {
      fscanf(fp,"%d %d %d",&i1,&i1,&i1); /* dummy */
      for (ie=iemin0[spin];ie<=iemax0[spin];ie++) {
        fscanf(fp,"%lf",&eig[spin][ie-iemin0[spin]]);
      }
    }
  }

  input_close();

}


void input_ask()
{
    gaussian=1/eV2Hartree;
    Temp=0.03/eV2Hartree;
    Nbin=1000;

    printf(" # gaussian=%lf\n",gaussian); 

    printf(" freqmax (Hartree)=? ");
    scanf(" %lf",&freqmax);

    printf(" freq mesh=? ");
    scanf(" %d",&Nbin);

}

/* Fermi distribution function */
double Fermi(double ene)
{
   if (ene> 10.0) return 0.0;
   if (ene<-10.0) return 1.0;
   return 1.0/(1.0+exp(ene)); 
}


void calc_opticalconductivity(char *fileout)
{
  int ie,ia,ib,iaxis1,iaxis2,spin;
  int iecenter, iewidth,iestart,ieend; 
  double w,val,xa,pi2,thermal; 
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */

  if ( (fp=fopen(fileout,"w"))==NULL ) {

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
    printf("can not open a file %s\n",fileout);
    exit(0);
  }

  pi2 = sqrt(PI);
  /* allocate */

  N_dos=4; i_dos[0]=nspin+1; i_dos[1]=Nbin; i_dos[2]=3; i_dos[3]=3;
  dos = (double****)malloc_multidimarray("double",N_dos,i_dos);

  ene = (double*)malloc(sizeof(double)*Nbin);

  printf(" # D_freq=%lf\n",freqmax/(double)Nbin);

  /* initialize */
  for (ie=0;ie<Nbin;ie++) {
    ene[ie]= 0.0+ freqmax*(double)ie/(double)Nbin;
  }
  for (spin=0;spin<=nspin;spin++) {
    for (ie=0;ie<Nbin;ie++) {
      for (iaxis1=0; iaxis1<3;iaxis1++) {
	for (iaxis2=0; iaxis2<3;iaxis2++) {
	  dos[spin][ie][iaxis1][iaxis2]=0.0;
	} 
      }
    }
  }

  /* calculate */
  iewidth = gaussian/freqmax* Nbin * 5.0;
  printf("# iewidth=%d\n",iewidth);
  for (spin=0;spin<=nspin;spin++) {
    for (iaxis1=2; iaxis1<=2;iaxis1++) {
      for (iaxis2=2; iaxis2<=2;iaxis2++) {
	for (ia=0;ia<iemax0[spin]-iemin0[spin]+1; ia++) {
	  for (ib=0;ib<iemax0[spin]-iemin0[spin]+1; ib++) {
	    w = eig[spin][ib]  - eig[spin][ia];
	    thermal = Fermi( eig[spin][ia]/Temp ) - Fermi( eig[spin][ib]/Temp ); 

	    printf("%d %d eig=%lf\n",ia,ib,w);  /* debug */

	    if (w>0.0 && thermal >1.0e-3) {
	      val= -thermal * 
		Jij[spin][ia][ib][iaxis1] * Jij[spin][ib][ia][iaxis2]/w;

	      printf("%d %lf (%lf %lf) %le \n",spin,w,eig[spin][ib],eig[spin][ia],val); 

	      /* debug */
	      /* broadening */
	      xa = w/freqmax * Nbin; 
	      iecenter =  xa; 
	      iestart= iecenter-iewidth; if (iestart<0) iestart=0;
	      ieend = iecenter+iewidth ; if (ieend >=Nbin) ieend = Nbin-1;
	      if (iestart>=Nbin) ieend = -1;
	      if (ieend < 0 ) iestart=Nbin;
	      /*debug*/
	      /*
	      printf("iestart, iecenter, ieend= %d %d %d\n",iestart, iecenter, ieend);
	      */
	      for (ie=iestart; ie<ieend; ie++) {
		xa = (ene[ie] - w )/gaussian;
		/*
                printf("%d %lf %lf %lf %lf %lf\n",ie, ene[ie],w, xa,exp(-xa*xa)/gaussian/pi2,val);
		*/
		dos[spin][ie][iaxis1][iaxis2] += val* exp(-xa*xa) / (gaussian*pi2) ;
		/*
		printf("dos: %d %d %d %d %lf\n",spin,ie,iaxis1,iaxis2, dos[spin][ie][iaxis1][iaxis2] );
		*/
	      }

	    } /* if (w */
	  } /* ib */


	} /* ia */
      } /* iaxis2 */
    } /* iaxis1 */
  } /* spin */

#if 1
  iaxis1=2;iaxis2=2;
  for (ie=0;ie<Nbin;ie++) {

    if (nspin==0){
      fprintf(fp,"%d %lf %lf %lf \n",
              ie, ene[ie], dos[0][ie][iaxis1][iaxis2],dos[0][ie][iaxis1][iaxis2]);
    }
    else if (nspin==1){
      fprintf(fp,"%d %lf %lf %lf \n",
              ie, ene[ie], dos[0][ie][iaxis1][iaxis2],dos[1][ie][iaxis1][iaxis2]);
    }
  }

  fclose(fp);
#endif 

}


/***************************************************************
****************************************************************/
main(int argc, char **argv)
{

  if (argc<4 )  {
    printf("usage :  thisprogram file.optical file.Dos.val file.opticaloutput\n");
    exit(0);
  }

  file_read(argv[1],argv[2]);

  input_ask();
  calc_opticalconductivity(argv[3]);

  return 0;
}
