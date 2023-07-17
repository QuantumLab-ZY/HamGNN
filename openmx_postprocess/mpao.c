/**********************************************************************
  mpao.c:

     mpao.c is a program which creats a new PAO file by merging optimized PAOs

      Compile: 

          gcc mpao.c -o mpao

      Usage:

         ./mpao pao.in In.pao Out.pao

      Input: pao.in, In.pao
      Output: Out.pao

     Format of pao.in 

number.optpao 2

# comments

<Contraction.coefficients1
  45
  Atom=  1  L= 0  Mul= 0  p=  0   0.987339105757399
  Atom=  1  L= 0  Mul= 0  p=  1   0.144572333164555
  Atom=  1  L= 0  Mul= 0  p=  2   0.056087183684438
  Atom=  1  L= 0  Mul= 0  p=  3   0.027261193718410
  Atom=  1  L= 0  Mul= 0  p=  4   0.015958449984304
  Atom=  1  L= 0  Mul= 0  p=  5   0.007910570725120
  Atom=  1  L= 0  Mul= 0  p=  6   0.005736873547430
  Atom=  1  L= 0  Mul= 0  p=  7   0.003146706448984
  Atom=  1  L= 0  Mul= 0  p=  8   0.002595307626407
  Atom=  1  L= 0  Mul= 0  p=  9   0.001465792640795
  Atom=  1  L= 0  Mul= 0  p= 10   0.001284605091222
  Atom=  1  L= 0  Mul= 0  p= 11   0.000603683749152
  Atom=  1  L= 0  Mul= 0  p= 12   0.000608890530101
  Atom=  1  L= 0  Mul= 0  p= 13   0.000110255389979
  Atom=  1  L= 0  Mul= 0  p= 14   0.000203509900251
  Atom=  1  L= 1  Mul= 0  p=  0   0.769132336574609
  Atom=  1  L= 1  Mul= 0  p=  1  -0.520650430202424
  Atom=  1  L= 1  Mul= 0  p=  2  -0.303581841748251
  Atom=  1  L= 1  Mul= 0  p=  3  -0.180366698518022
  Atom=  1  L= 1  Mul= 0  p=  4  -0.088337151236981
  Atom=  1  L= 1  Mul= 0  p=  5  -0.059297067314670
  Atom=  1  L= 1  Mul= 0  p=  6  -0.029007088451895
  Atom=  1  L= 1  Mul= 0  p=  7  -0.019093007338914
  Atom=  1  L= 1  Mul= 0  p=  8  -0.007349579879166
  Atom=  1  L= 1  Mul= 0  p=  9  -0.008181220209327
  Atom=  1  L= 1  Mul= 0  p= 10  -0.001565548729985
  Atom=  1  L= 1  Mul= 0  p= 11  -0.003591940563818
  Atom=  1  L= 1  Mul= 0  p= 12  -0.000930140414519
  Atom=  1  L= 1  Mul= 0  p= 13  -0.001058335479029
  Atom=  1  L= 1  Mul= 0  p= 14   0.000813816776307
  Atom=  1  L= 2  Mul= 0  p=  0   0.969345399226012
  Atom=  1  L= 2  Mul= 0  p=  1  -0.201796897918964
  Atom=  1  L= 2  Mul= 0  p=  2  -0.080389536810392
  Atom=  1  L= 2  Mul= 0  p=  3  -0.080687317560867
  Atom=  1  L= 2  Mul= 0  p=  4  -0.031372587939204
  Atom=  1  L= 2  Mul= 0  p=  5  -0.061985377639792
  Atom=  1  L= 2  Mul= 0  p=  6  -0.019024982663290
  Atom=  1  L= 2  Mul= 0  p=  7  -0.033988635073689
  Atom=  1  L= 2  Mul= 0  p=  8  -0.007170045929676
  Atom=  1  L= 2  Mul= 0  p=  9  -0.014118301811566
  Atom=  1  L= 2  Mul= 0  p= 10  -0.001837123240428
  Atom=  1  L= 2  Mul= 0  p= 11  -0.006954581255371
  Atom=  1  L= 2  Mul= 0  p= 12  -0.000905329100295
  Atom=  1  L= 2  Mul= 0  p= 13  -0.005258826612580
  Atom=  1  L= 2  Mul= 0  p= 14   0.000180128264926
Contraction.coefficients1>

# comments

<Contraction.coefficients2
  45
  Atom=  1  L= 0  Mul= 0  p=  0   1.987339105757399
  Atom=  1  L= 0  Mul= 0  p=  1   0.144572333164555
  Atom=  1  L= 0  Mul= 0  p=  2   0.056087183684438
  Atom=  1  L= 0  Mul= 0  p=  3   0.027261193718410
  Atom=  1  L= 0  Mul= 0  p=  4   0.015958449984304
  Atom=  1  L= 0  Mul= 0  p=  5   0.007910570725120
  Atom=  1  L= 0  Mul= 0  p=  6   0.005736873547430
  Atom=  1  L= 0  Mul= 0  p=  7   0.003146706448984
  Atom=  1  L= 0  Mul= 0  p=  8   0.002595307626407
  Atom=  1  L= 0  Mul= 0  p=  9   0.001465792640795
  Atom=  1  L= 0  Mul= 0  p= 10   0.001284605091222
  Atom=  1  L= 0  Mul= 0  p= 11   0.000603683749152
  Atom=  1  L= 0  Mul= 0  p= 12   0.000608890530101
  Atom=  1  L= 0  Mul= 0  p= 13   0.000110255389979
  Atom=  1  L= 0  Mul= 0  p= 14   0.000203509900251
  Atom=  1  L= 1  Mul= 0  p=  0   1.769132336574609
  Atom=  1  L= 1  Mul= 0  p=  1  -0.520650430202424
  Atom=  1  L= 1  Mul= 0  p=  2  -0.303581841748251
  Atom=  1  L= 1  Mul= 0  p=  3  -0.180366698518022
  Atom=  1  L= 1  Mul= 0  p=  4  -0.088337151236981
  Atom=  1  L= 1  Mul= 0  p=  5  -0.059297067314670
  Atom=  1  L= 1  Mul= 0  p=  6  -0.029007088451895
  Atom=  1  L= 1  Mul= 0  p=  7  -0.019093007338914
  Atom=  1  L= 1  Mul= 0  p=  8  -0.007349579879166
  Atom=  1  L= 1  Mul= 0  p=  9  -0.008181220209327
  Atom=  1  L= 1  Mul= 0  p= 10  -0.001565548729985
  Atom=  1  L= 1  Mul= 0  p= 11  -0.003591940563818
  Atom=  1  L= 1  Mul= 0  p= 12  -0.000930140414519
  Atom=  1  L= 1  Mul= 0  p= 13  -0.001058335479029
  Atom=  1  L= 1  Mul= 0  p= 14   0.000813816776307
  Atom=  1  L= 2  Mul= 0  p=  0   1.969345399226012
  Atom=  1  L= 2  Mul= 0  p=  1  -0.201796897918964
  Atom=  1  L= 2  Mul= 0  p=  2  -0.080389536810392
  Atom=  1  L= 2  Mul= 0  p=  3  -0.080687317560867
  Atom=  1  L= 2  Mul= 0  p=  4  -0.031372587939204
  Atom=  1  L= 2  Mul= 0  p=  5  -0.061985377639792
  Atom=  1  L= 2  Mul= 0  p=  6  -0.019024982663290
  Atom=  1  L= 2  Mul= 0  p=  7  -0.033988635073689
  Atom=  1  L= 2  Mul= 0  p=  8  -0.007170045929676
  Atom=  1  L= 2  Mul= 0  p=  9  -0.014118301811566
  Atom=  1  L= 2  Mul= 0  p= 10  -0.001837123240428
  Atom=  1  L= 2  Mul= 0  p= 11  -0.006954581255371
  Atom=  1  L= 2  Mul= 0  p= 12  -0.000905329100295
  Atom=  1  L= 2  Mul= 0  p= 13  -0.005258826612580
  Atom=  1  L= 2  Mul= 0  p= 14   0.000180128264926
Contraction.coefficients2>

  Log of mpao.c:

     8/Nov/2010  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXNEST   14

#define SizeL    10
#define SizeMul  30
#define SizeCCoes 30

static FILE *fp=NULL;
static int  nestlevel=-1;
static FILE *fpnest[MAXNEST];

static int printlevel=0;
static int errorlevel=0;

char *string_tolower(char *buf, char *buf1);
int input_open(const  char *fname);
int input_close();
int input_cmpstring(const char *str,  int *ret, int nvals,
                    char **strval,
                    int *ivals);
int input_logical(const char *key, int *ret,const  int defval);
int input_int(const char *key, int *ret, const int defval);
int input_double(const char *key, double *ret, const double defval);
int input_string(const char *key, char *ret,const char *defval);
int input_stringv(const char *key,const int nret, char **ret, char  **defval);
int input_doublev(const char *key,const int nret, double *ret, double *defval);
int input_intv(const char *key,const int nret, int *ret, int *defval)
;
FILE *input_find(const char *key);
int input_last(const char *key);
int input_string2int(const char *key, int *ret,int nvals,  char **strval,  int *ivals);
int input_errorCount();

#define fp_bsize         1048576     /* buffer size for setvbuf */

int main(int argc, char *argv[]) 
{

  FILE *fp1,*fp2,*fp3;
  int PAO_Lmax,PAO_Mul,Mul2,Mesh;

  int exflag[100]; 

  int num,loopnum,loop;
  int i,j,L,L0,Mul,p;
  int GMul[SizeMul]; 
  int num_optpao;
  char str1[100];
  double tmp1,tmp2;
  double ccoes[SizeL][SizeMul][SizeCCoes];
  double *XV,*RV;

  double ***PAO;
  double dum;
  char buf[1000];
  char file1[200],file2[200],file3[200];
  char st[3000];
  char st1[3000];

  if (argc!=4){
    printf("Usage:\n");
    printf("  ./mpao pao.in In.pao Out.pao\n");
    exit(0);
  }

  /*******************************************
              read the first file 
  *******************************************/

  for (L=0; L<SizeL; L++){

    GMul[L] = 0;

    for (Mul=0; Mul<SizeMul; Mul++){
      for (i=0; i<SizeCCoes; i++){
        ccoes[L][Mul][i] = 0.0;
      }
      ccoes[L][Mul][Mul] = 1.0;
    }
  }

  input_open(argv[1]);

  input_int("number.optpao",&num_optpao,1);

  for (num=1; num<=num_optpao; num++){

    sprintf(file1,"<Contraction.coefficients%i",num);
    sprintf(file2,"Contraction.coefficients%i>",num);

    if (fp=input_find(file1)) {

      fscanf(fp,"%d",&loopnum); 

      for (loop=0; loop<loopnum; loop++){  

	fscanf(fp,"%s",str1);   /* Atom= */
	fscanf(fp,"%lf",&tmp1); 

	fscanf(fp,"%s",str1);   /* L= */
	fscanf(fp,"%d",&L); 

	fscanf(fp,"%s",str1);   /* Mul= */
	fscanf(fp,"%d",&Mul); 

	fscanf(fp,"%s",str1);   /* p= */
	fscanf(fp,"%d",&p); 

	fscanf(fp,"%lf",&tmp1); /* contraction coefficient */ 

	if (p==0) GMul[L]++;

	ccoes[L][GMul[L]-1][p] = tmp1;                    

      }
    }

    if (!input_last(file2)) {
      /* format error */
      printf("Contraction.coefficients%i\n",num);
      exit(1);
    }
  } 

  input_close();

  for (L=0; L<5; L++){
    printf("L=%2d Mul=%2d\n",L,GMul[L]);
  }  

  for (L=0; L<5; L++){
    for (Mul=0; Mul<10; Mul++){
      for (p=0; p<15; p++){
        printf("L=%2d Mul=%2d p=%2d CCoes=%18.15f\n",L,Mul,p,ccoes[L][Mul][p]);
      }
    }
  }  

  /*******************************************
              read the second file 
  *******************************************/

  input_open(argv[2]);

  input_int("PAO.Lmax",&PAO_Lmax,0);
  input_int("PAO.Mul",&PAO_Mul,0);
  input_int("grid.num.output",&Mesh,0);

  XV = (double*)malloc(sizeof(double)*Mesh); 
  RV = (double*)malloc(sizeof(double)*Mesh); 

  PAO = (double***)malloc(sizeof(double**)*(PAO_Lmax+1)); 
  for (i=0; i<(PAO_Lmax+1); i++){
    PAO[i] = (double**)malloc(sizeof(double*)*PAO_Mul); 
    for (j=0; j<PAO_Mul; j++){
      PAO[i][j] = (double*)malloc(sizeof(double)*Mesh); 
    }
  }

  for (L=0; L<=PAO_Lmax; L++){  

    sprintf(file1,"<pseudo.atomic.orbitals.L=%i",L);
    sprintf(file2,"pseudo.atomic.orbitals.L=%i>",L);

    if (fp=input_find(file1)) {
      for (i=0; i<Mesh; i++){
	for (j=0; j<=(PAO_Mul+1); j++){
	  if (fscanf(fp,"%lf",&dum)==EOF){
	    printf("File error in pseudo.atomic.orbitals.L=%i\n",L);
	  }
	  else{
	    if (j==0)
	      XV[i] = dum;
	    else if (j==1)
	      RV[i] = dum;
	    else
	      PAO[L][j-2][i] = dum;
	  }
	}
      }

    }

    if (!input_last(file2)) {
      /* format error */
      printf("Format error in pseudo.atomic.orbitals.L=%i\n",L);
      exit(1);
    }

  }

  input_close();

  /*******************************************
     modified Gram-Schmidt orthonormalization 
  *******************************************/

  for (L=0; L<=PAO_Lmax; L++){
    for (Mul=0; Mul<PAO_Mul; Mul++){

      /* orthogonalization */

      for (Mul2=0; Mul2<Mul; Mul2++){

        tmp1 = 0.0;
        for (i=0; i<PAO_Mul; i++){
          tmp1 += ccoes[L][Mul2][i]*ccoes[L][Mul][i];
        }

        for (i=0; i<PAO_Mul; i++){
          ccoes[L][Mul][i] -= tmp1*ccoes[L][Mul2][i];
        }
      }

      /* normalization */
 
      tmp1 = 0.0;     
      for (i=0; i<PAO_Mul; i++){
        tmp1 += ccoes[L][Mul][i]*ccoes[L][Mul][i];
      }
      
      tmp1 = 1.0/sqrt(tmp1); 

      for (i=0; i<PAO_Mul; i++){
        ccoes[L][Mul][i] *= tmp1;
      }
    }
  }

  /*
  for (L=0; L<=PAO_Lmax; L++){
    for (Mul=0; Mul<PAO_Mul; Mul++){
      for (Mul2=0; Mul2<PAO_Mul; Mul2++){

        tmp1 = 0.0;
        for (i=0; i<PAO_Mul; i++){
          tmp1 += ccoes[L][Mul][i]*ccoes[L][Mul2][i];
        }

        printf("ABC L=%2d Mul=%2d Mul2=%2d %18.15f\n",L,Mul,Mul2,tmp1);

      }
    }
  }
  */

  printf("%s %s %s %s\n",argv[0],argv[1],argv[2],argv[3]);

  /*******************************************
                output a file 
  *******************************************/

  /* the new pao file */    

  fp1 = fopen(argv[3],"w");
  fseek(fp1,0,SEEK_END);

  /* the input file for contraction coefficients */    

  fp3 = fopen(argv[1],"r");

  if (fp3!=NULL){

    fprintf(fp1,"*****************************************************\n");
    fprintf(fp1,"*****************************************************\n");
    fprintf(fp1," The numerical atomic orbitals were generated\n");
    fprintf(fp1," by the variational optimization with OpenMX,\n");
    fprintf(fp1," and a patch work with mpao.\n");
    fprintf(fp1," A set of contraction coefficients can be found below.\n");
    fprintf(fp1,"*****************************************************\n");
    fprintf(fp1,"*****************************************************\n");

    fprintf(fp1,"\n\n");

    while (fgets(st,800,fp3)!=NULL){
      fprintf(fp1,"%s",st);
    }

    fprintf(fp1,"\n\n");
  }

  fclose(fp3); 

  /* the original pao file */    

  fp2 = fopen(argv[2],"r");

  if (fp2!=NULL){

    while (fgets(st,800,fp2)!=NULL){

      string_tolower(st,st1); 

      if (strncmp(st1,"pao.lmax",8)==0){

        fprintf(fp1,"PAO.Lmax   %d\n",PAO_Lmax);
        fprintf(fp1,"PAO.Mul    %d\n",PAO_Mul);

        for (L=0; L<=PAO_Lmax; L++){  

          fprintf(fp1,"<pseudo.atomic.orbitals.L=%d\n",L);
           
          for (i=0; i<Mesh; i++){

            fprintf(fp1,"%18.15f %18.15f ",XV[i],RV[i]);

            for (Mul=0; Mul<PAO_Mul; Mul++){

              tmp1 = 0.0; 
              for (Mul2=0; Mul2<PAO_Mul; Mul2++){
                tmp1 += ccoes[L][Mul][Mul2]*PAO[L][Mul2][i];
	      }

              fprintf(fp1,"%18.15f ",tmp1);
	    }

            fprintf(fp1,"\n");
	  }
          fprintf(fp1,"pseudo.atomic.orbitals.L=%d>\n",L);
	}  

goto LAST_Proc;

      }
      else{
        fprintf(fp1,"%s",st);
      }

    }
    
 LAST_Proc:

    fclose(fp2); 
  }    

  fclose(fp1); 

  /* freeing of arrays */

  for (i=0; i<(PAO_Lmax+1); i++){
    for (j=0; j<PAO_Mul; j++){
      free(PAO[i][j]);
    }
    free(PAO[i]);
  }
  free(PAO);

  free(XV);
  free(RV);
}





char *string_tolower(char *buf, char *buf1)
{
  char *c=buf;
  char *c1=buf1;

  while (*c){
    *c1=tolower(*c);
    c++;
    c1++;
  }
 return buf;
}

int input_open(const  char *fname)
{

  nestlevel++;
  if (nestlevel>=MAXNEST) {
    printf("input_open: can not open a further file. nestlevel=%d\n",nestlevel);
    return 0; 
  }

  fp=fpnest[nestlevel]=fopen(fname, "r");

  #ifdef DEBUG
    printf("input_open: nestlevel=%d fp=%x\n",nestlevel,fp);
  #endif

  errorlevel = 0;

  if (fp==NULL) {
    printf("input_open: can not open %s\n",fname);
    return 0;
  }

  return 1;
}

int input_close()
{
  int ret;
#ifdef DEBUG
  printf("input_close: closing fp=%x\n",fp);
#endif
  ret= fclose(fp);
  fp=NULL;
  nestlevel--;
  if (nestlevel>=0) fp=fpnest[nestlevel];
#ifdef DEBUG
  printf("input_close: nestlevel=%d fp=%x\n",nestlevel,fp);
#endif
  return ret;
}



static int strlen_trim(const char *str) 
{
  int len;
  int i;
  len = strlen(str);
  for (i=len-1;i>=0;i++) {
    if (str[i]!=' ' || str[i]!='\t') return i+1;
  }
  return 0;
}

static char *string_toupper(char *buf)
{

  char *c=buf;
  while (*c){
    *c=toupper(*c);
    c++;
  }
 return buf;
}


/***************************************************
  The name of strcasestr was changed to mystrcasestr
  due to confliction to MAC gcc Ver. 4.0 
  by T.Ozaki at 19 Dec. 2005
****************************************************/

char* mystrcasestr(  char *str1, const char *str2) 
{
  char s1[500],s2[500];
  char *c;
  strcpy(s1,str1);
  strcpy(s2,str2);

  string_toupper(s1);
  string_toupper(s2);
  c=strstr(s1,s2);
  if (c==NULL) {return NULL; }
  else {
    return &str1[c-s1];
  }
}



static char *nexttoken(char *c)
{
  while (*c==' ' || *c=='\t' ) c++;
  return c;
}


int input_cmpstring(const char *str,  int *ret, int nvals,
                    char **strval,
                    int *ivals)
{
  int i;
  int lenstr,lenstrval;

  /*   for (i=0;i<nvals;i++) {
       printf("<%s> <%d>| ",strval[i],ivals[i]);
       }
       printf("\n");
  */

  lenstr=strlen_trim(str);
  for (i=0;i<nvals;i++) {
    lenstrval = strlen_trim(strval[i]);
    if (printlevel>10) {
       printf("%d %d %s %s\n",lenstr,lenstrval,str,strval[i]);
    }
    if (lenstr==lenstrval && strncasecmp(str,strval[i],lenstr)==0 ) {
      if (printlevel>10) {
              printf("<%s> found\n",strval[i]); 
      }
      *ret= ivals[i] ;
               if (printlevel>0) {
		 printf("%s=%s %d\n",str,strval[i],ivals[i]);
		 }  
      return 1;
    }
  }
  return 0;
}

int input_errorCount() {
  return errorlevel;
}

#define SIZE_LOGICAL_DEFAULT 10
#define BUFSIZE 200

int input_logical(const char *key, int *ret,const  int defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE];
  int keylen,buflen,nread;

   char *strval[SIZE_LOGICAL_DEFAULT]={"on","yes","true",".true.","ok","off","no","false",".false.","ng"};
   int ival[SIZE_LOGICAL_DEFAULT]={1,1,1,1,1,0,0,0,0,0};

  char key_inbuf[BUFSIZE],val_inbuf[BUFSIZE];

  keylen=strlen_trim(key);
  
  *ret=defval;
  rewind(fp);
  while ( fgets(buf,size,fp) ) {
    nread=sscanf(buf,"%s %s",key_inbuf,val_inbuf); 
    buflen=strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      if (nread!=2) {
	goto ERROR;
      }
      if ( input_cmpstring( val_inbuf,
			    ret, SIZE_LOGICAL_DEFAULT, strval, ival)) {
	if (printlevel>0) {
	  printf("%s=%s=%d\n",key,val_inbuf,*ret);
	}
	return 1;
      }
      else { 
	goto ERROR;
      }
    }
  }
  return 0;

 ERROR:
  printf("\nERROR, key=%s value=logical\n\n",key);
  errorlevel++;
  return -1;

}

int input_int(const char *key, int *ret, const int defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE];
  int keylen,buflen, nread;
  char key_inbuf[BUFSIZE],val_inbuf[BUFSIZE];


  keylen=strlen_trim(key);

  *ret=defval;
  rewind(fp);
  while ( fgets(buf,size,fp) ) {
    nread=sscanf(buf,"%s %s",key_inbuf,val_inbuf);
    buflen =strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      if (nread!=2) {
	goto ERROR;
      }
      nread=sscanf(val_inbuf,"%d",ret);
      if (printlevel>0) {
	printf("%s= %d\n",key,*ret);
      }
      if (nread!=1) {
	goto ERROR;
      }
      return 1;
    }
  }       
  return 0;

 ERROR:
  printf("\nERROR, key=%s value=int\n\n",key);
  errorlevel++;
  return -1;

}

int input_double(const char *key, double *ret, const double defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE];
  int keylen,buflen,nread;
  char key_inbuf[BUFSIZE],val_inbuf[BUFSIZE];


  keylen=strlen_trim(key); 

  *ret=defval;
  rewind(fp);
  while ( fgets(buf,size,fp) ) {
    nread=sscanf(buf,"%s %s",key_inbuf,val_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      if (nread!=2) {
	goto ERROR;
      }
      nread=sscanf(val_inbuf,"%lf",ret);
      if (nread!=1) {
	goto ERROR;
      } 
      if (printlevel>0) {
	printf("%s= %E\n",key,*ret);
      }
      return 1;
    }
  }
  return 0;

 ERROR:
  printf("\nERROR, key=%s value=double\n\n",key);
  errorlevel++;
  return 0;

}

int input_string(const char *key, char *ret,const char *defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE];
  int keylen,buflen,nread;
  char key_inbuf[BUFSIZE],val_inbuf[BUFSIZE];

  keylen=strlen_trim(key);

  strcpy(ret,defval);
  rewind(fp);
  while ( fgets(buf,size,fp) ) {
    nread=sscanf(buf,"%s %s",key_inbuf,val_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      /*      printf("<%s> found val=<%s>\n",key,val_inbuf); */
      if (nread!=2) {
	goto ERROR;
      }
      strcpy(ret,val_inbuf);
      if (printlevel>0) {
	printf("%s=%s\n",key,ret);
      }
      return 1;
    }
  }
  return 0;

 ERROR:
  printf("\nERROR key=%s value=string\n\n",key);
  errorlevel++;
  return -1;

}


int input_string2int(const char *key, int *ret, int nvals, 
                     char **strval,  int *ivals)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE];
  int keylen,buflen,iret,nread;
  char key_inbuf[BUFSIZE],val_inbuf[BUFSIZE];

  keylen=strlen_trim(key);

  *ret=ivals[0]; 
  rewind(fp);
  while ( fgets(buf,size,fp) ) {
    nread=sscanf(buf,"%s %s",key_inbuf,val_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      if (nread!=2) {
	goto ERROR;
      }
      iret= input_cmpstring(val_inbuf,ret,nvals,strval,ivals);
      if (printlevel>0) {
	printf("%s=%d\n",key,*ret);
      }
      if (iret==0) {
	goto ERROR;
      }
      return iret;
    }
  }
  return 0;
 ERROR:
  printf("\nERROR key=%s\n\n",key);
  errorlevel++;
  return -1;

}



int input_stringv(const char *key,const int nret, char **ret, char  **defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE], *c;
  int keylen,buflen;
  char key_inbuf[BUFSIZE];
  int i;


  keylen=strlen_trim(key); 
  for (i=0;i<nret;i++) {
    strcpy(ret[i],defval[i]);
  }
  rewind(fp);
  while ( c=fgets(buf,size,fp) ) {
    sscanf(buf,"%s",key_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      c=strstr(buf,key)+ keylen+1;
      for (i=0;i<nret;i++) {
	c= nexttoken(c);
	if (sscanf(c,"%s",ret[i])!=1) {goto ERROR;}
	c=c+strlen(ret[i]);
      }
      if (printlevel>0) {
	printf("%s= ",key);
	for (i=0;i<nret;i++) {
	  printf("%s|",ret[i]);
	}
	printf("\n");
      }
      return 1;
    }
  }
  return 0;

 ERROR:
  printf("\nERROR key=%s value= %d strings\n\n",key,nret);
  errorlevel++;
  return -1;


}

int input_doublev(const char *key,const int nret, 
double *ret, double *defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE], *c;
  int keylen,buflen;
  char key_inbuf[BUFSIZE];
  int i;

  keylen=strlen_trim(key);
  /*  printf("input_doblev in\n"); */
  for (i=0;i<nret;i++) {
    ret[i]=defval[i];
  }
  rewind(fp);
  while ( c=fgets(buf,size,fp) ) {
    sscanf(buf,"%s",key_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen == buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      c=mystrcasestr(buf,key); c=&c[keylen+1];
      for (i=0;i<nret;i++) {
	c= nexttoken(c);
	if (sscanf(c,"%s",buf)!=1) goto ERROR;
	if (sscanf(buf,"%lf",&ret[i])!=1) goto ERROR;
	c=c+strlen(buf);
      }
      if (printlevel>0){
	printf("%s= ",key);
	for (i=0;i<nret;i++) {
	  printf("%lf|",ret[i]);
	}
	printf("\n");
      }

      return 1;
    }
  }
  return 0;

 ERROR:
  printf("\nERROR key=%s value= %d doubles\n\n",key,nret);
  errorlevel++;
  return -1;

}



int input_intv(const char *key, int nret, int *ret, int *defval)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE], *c;
  int keylen,buflen;
  char key_inbuf[BUFSIZE];
  int i;

  keylen=strlen_trim(key);
  /*  printf("input_doblev in\n"); */
  for (i=0;i<nret;i++) {
    ret[i]=defval[i];
  }
  rewind(fp);
  while ( c=fgets(buf,size,fp) ) {
    sscanf(buf,"%s",key_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      /*    printf("<%s> found\n",key); */
      c=mystrcasestr(buf,key)+keylen+1;
      /*    printf("0 <%s>\n",c); */
      for (i=0;i<nret;i++) {
	c= nexttoken(c);
	/*       printf("%d <%s>\n",i,c); */
	if (sscanf(c,"%s",buf)!=1) goto ERROR;
	if (sscanf(buf,"%d",&ret[i])!=1) goto ERROR;
	c=c+strlen(buf);
      }
      if (printlevel>0) {
	printf("%s=",key);
	for (i=1;i<nret;i++) {
	  printf("%d|",ret[i]);
	}
	printf("\n");
      }
      return 1;
    }
  }
  return 0;
 ERROR:
  printf("\nERROR key=%s value= %d intv\n\n",key,nret);
  errorlevel++;
  return -1;

}


FILE *input_find(const char *key)
{
  const int size=BUFSIZE;
  char buf[BUFSIZE] ;
  char key_inbuf[BUFSIZE];
  int keylen,buflen;

  keylen=strlen_trim(key);

  rewind(fp);
  while ( fgets(buf,size,fp) ) {
    sscanf(buf,"%s",key_inbuf);
    buflen = strlen_trim(key_inbuf);
    if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
      return fp;
    }
  }
  return NULL;

}

int input_last(const char *key) 
{
  const int size=BUFSIZE;
  char buf[BUFSIZE];
  char key_inbuf[BUFSIZE];
  int keylen,buflen;

  keylen=strlen_trim(key);

  fgets(buf,size,fp); 
  fgets(buf,size,fp); 
  sscanf(buf,"%s",key_inbuf);
  buflen=strlen_trim(key_inbuf);
  /*  printf("last=<%s>\n",buf); */
  if (keylen==buflen && strncasecmp(key,key_inbuf,keylen)==0) {
    return 1;
  }

  return 0;
}




