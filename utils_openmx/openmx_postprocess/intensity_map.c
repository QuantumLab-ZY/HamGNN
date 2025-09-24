/**********************************************************************
  intensity_map.c:

     intensity_map.c is a code to generate data for intensity map of the 
     unfolded spectral weight obtained by the unfolding procedure for bands.
     The unfolded spectral weight w is smeared out by a Lorentzian function 
     w/((k/hk)^2+(e/he)^2+1), where k is momentum, and e is energy. 

     This code follows GNU-GPL.

     Compile: gcc intensity_map.c -lm -o intensity_map

     Usage:  ./intensity_map file -c 3 -k 0.2 -e 0.1 -l -10 -u 10 > outputfile

       -c     column of spectral weight you analyze
       -k     degree of smearing (Bohr^{-1}) in k-vector
       -e     degree of smearing (eV) in energy
       -l     lower bound of energy for drawing the map
       -u     upper bound of energy for drawing the map

  Log of intensity_map.c:

     04/Feb/2016  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int main(int argc, char *argv[]) 
{
  int i,j,q,po_c,po_k,po_e,n,po,ret;
  int po_l,po_u,l_i,u_i; 
  int c_i,k_i,e_i,nline,nelement,tnelement;
  int column_index;
  int kmesh,emesh; 
  double dk,de,hk,he,tmp,eu,el;
  double *momentum,*energy,*weight;
  double kmin,kmax,emin,emax;
  double kv,ev,sum,se,sk;
  char *buf,*token;
  FILE *fp1,*fp2;

  /***********************************
        analyze the arguments
  ************************************/

  po_c = 0;
  po_k = 0;
  po_e = 0;
  po_l = 0;
  po_u = 0;

  for (i=1; i<argc; i++){

    if (strcmp(argv[i],"-c")==0){
      po_c = 1;
      c_i = i;
    }

    if (strcmp(argv[i],"-k")==0){
      po_k = 1;
      k_i = i;
    }

    if (strcmp(argv[i],"-e")==0){
      po_e = 1;
      e_i = i;
    }

    if (strcmp(argv[i],"-l")==0){
      po_l = 1;
      l_i = i;
    }

    if (strcmp(argv[i],"-u")==0){
      po_u = 1;
      u_i = i;
    }

  }

  if ( po_c==0 || po_k==0 || po_e==0 || po_l==0 || po_u==0 || argc!=12 ){
    printf("Invalid argument\n");
    exit(0);
  }

  /* -c */

  column_index = atoi(argv[c_i+1]);

  /* -k */

  hk = atof(argv[k_i+1]);

  /* -e */

  he = atof(argv[e_i+1]);

  /* -l */

  el = atof(argv[l_i+1]);

  /* -u */

  eu = atof(argv[u_i+1]);

  /***********************************
             read the file 
  ************************************/

  if ((fp1 = fopen(argv[1],"r")) != NULL){

    /* get the length of one line */
   
    po = 0;
    n = 0;
    do {
      if (( q=fgetc(fp1))=='\n') po = 1;
      n++;
    } while (po==0);

    buf = (char*)malloc(sizeof(char)*(n+100));

    /* get the nline by reading line by line */

    rewind( fp1 );

    nline = 0;
    while ( fgets(buf, n+30, fp1) != NULL ) {
      nline++;
    }

    /* allocate arrays */

    momentum = (double*)malloc(sizeof(double)*nline); 
    energy = (double*)malloc(sizeof(double)*nline); 
    weight = (double*)malloc(sizeof(double)*nline); 

    /* get nelement by reading the file again */

    rewind( fp1 );

    tnelement = 0;
    while(( ret = fscanf( fp1 , "%lf" , buf )) != EOF ) {
      tnelement++;
    }

    nelement = tnelement/nline;

    /* get momentum, energy, and weight */
    
    rewind( fp1 );

    for (i=0; i<nline; i++){
      for (j=0; j<nelement; j++){
        fscanf(fp1,"%lf",&tmp);
        if      (j==0)                momentum[i] = tmp;
        else if (j==1)                energy[i]   = tmp;
        else if (j==(column_index-1)) weight[i]   = tmp;
      }
    }

    /* close fp1 */
    fclose(fp1);
  }
  else{
    printf("could not find %s\n",argv[1]);
  }
    
  /***********************************
      calculate the intensity map
  ************************************/

  /* find min and max */

  kmin = 100000.0;
  kmax =-100000.0;
  emin = 100000.0;
  emax =-100000.0;

  for (i=0; i<nline; i++){
    if (momentum[i]<kmin) kmin = momentum[i];
    if (kmax<momentum[i]) kmax = momentum[i];
    if (energy[i]<emin) emin = energy[i];
    if (emax<energy[i]) emax = energy[i];
  }

  /* determine kmesh and emesh */

  kmesh = 200;
  emesh = 200;

  /* calculated dk and ek */

  dk = (kmax-kmin)/kmesh;
  de = (eu-el)/emesh;

  /* loop for kmesh and emesh */

  for (i=0; i<=kmesh; i++){

    kv = kmin + dk*(double)i;

    for (j=0; j<=emesh; j++){

      ev = el + de*(double)j;

      /* loop for nline */

      sum = 0.0;
      for (q=0; q<nline; q++){
        
        se = (ev - energy[q])/hk; 
        sk = (kv - momentum[q])/he; 
        sum += weight[q]/(se*se+sk*sk+1);
      }

      printf("%15.12f %15.12f %15.12f\n",kv,ev,sum);
    }

    printf("\n");
  }

}

