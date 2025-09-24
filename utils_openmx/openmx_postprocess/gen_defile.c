/**********************************************************************
  gen_defile.c

      Compile: 
         gcc gen_defile.c -lm -o gen_defile

      Usage: 
         ./gen_defile input1.out input2.out output.de

      Definition of difference: 
         input2 - input1 = output

  Log of gen_defile.c:

     16/Aug/2016  Released by T. Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#define MAXNEST   14
#define MAXBUF    1024
#define BohrR     0.529177249

static FILE *fp=NULL;
static int  nestlevel=-1;
static FILE *fpnest[MAXNEST];

static int printlevel=0;
static int errorlevel=0;

int input_open(const  char *fname);
int input_close();
int input_cmpstring(const char *str,  int *ret, int nvals, char **strval, int *ivals);
int input_logical(const char *key, int *ret,const  int defval);
int input_int(const char *key, int *ret, const int defval);
int input_double(const char *key, double *ret, const double defval);
int input_string(const char *key, char *ret,const char *defval);
int input_stringv(const char *key,const int nret, char **ret, char  **defval);
int input_doublev(const char *key,const int nret, double *ret, double *defval);
int input_intv(const char *key,const int nret, int *ret, int *defval);
FILE *input_find(const char *key);
int input_last(const char *key);
int input_string2int(const char *key, int *ret,int nvals,  char **strval,  int *ivals);
int input_errorCount();




int main(int argc, char *argv[]) 
{
  int i,j,po;
  int atomnum1,atomnum2;
  int i_vec[40],i_vec2[40];
  int unitvector_unit;
  double **Gxyz1,**Gxyz2;
  double tv[4][4];
  double tmp0,tmp1,tmp2,tmp3;
  char *s_vec[40];
  char **AtomChar;
  char ctmp[100];
  char buf[MAXBUF],buf2[MAXBUF],*c;
  FILE *fp1,*fp2,*fp3;

  if (argc!=4){
    printf("Usage:\n");
    printf("  ./gen_defile input1.out input2.out output.de\n");
    exit(0);
  }

  /*******************************************
     read the first file: file1.out
  *******************************************/

  if (input_open(argv[1])==0){
    printf("Could not open %s\n",argv[1]);    
    exit(0);
  }

  input_int("Atoms.Number",&atomnum1,0);

  Gxyz1 = (double**)malloc(sizeof(double*)*(atomnum1+1)); 
  for (i=0; i<(atomnum1+1); i++){
    Gxyz1[i] = (double*)malloc(sizeof(double)*30); 
  }

  AtomChar = (char**)malloc(sizeof(char*)*(atomnum1+1)); 
  for (i=0; i<(atomnum1+1); i++){
    AtomChar[i] = (char*)malloc(sizeof(char)*100); 
  }

  if (fp=input_find("<coordinates.forces") ) {

    fscanf(fp,"%lf",&tmp1);

    for (i=1; i<=atomnum1; i++){
      fscanf(fp,"%lf %s %lf %lf %lf %lf %lf %lf",
             &tmp0,AtomChar[i],&Gxyz1[i][1],&Gxyz1[i][2],&Gxyz1[i][3],
             &tmp1,&tmp2,&tmp3);
    }

    if ( ! input_last("coordinates.forces>") ) {
      /* format error */
      printf("Format error for coordinates.forces\n");
      po++;
    }
  }

  s_vec[0]="Ang"; s_vec[1]="AU";
  i_vec[0]=0;  i_vec[1]=1;
  input_string2int("Atoms.UnitVectors.Unit",&unitvector_unit,2,s_vec,i_vec);

  if (fp=input_find("<Atoms.Unitvectors")) {

    for (i=1; i<=3; i++){
      fscanf(fp,"%lf %lf %lf",&tv[i][1],&tv[i][2],&tv[i][3]);
    }
    if ( ! input_last("Atoms.Unitvectors>") ) {
      /* format error */
      printf("Format error for Atoms.Unitvectors\n");
      po++;
    }

    /* AU to Ang */
    if (unitvector_unit==1){
      for (i=1; i<=3; i++){
	tv[i][1] = tv[i][1]*BohrR;
	tv[i][2] = tv[i][2]*BohrR;
	tv[i][3] = tv[i][3]*BohrR;
      }
    }
  }

  if (fp=input_find("Decomposed.energies.(Hartree).with.respect.to.atom") ) {

    fgets(buf,MAXBUF,fp);
    fgets(buf,MAXBUF,fp);

    for (i=1; i<=atomnum1; i++){
      fgets(buf,MAXBUF,fp);
      sscanf(buf,"%d %s %lf",&j,ctmp,&tmp1);
      Gxyz1[i][4] = -tmp1;
    }
  }

  input_close();

  /*******************************************
     read the second file: file2.out
  *******************************************/

  if (input_open(argv[2])==0){
    printf("Could not open %s\n",argv[2]);
    exit(0);
  }

  input_int("Atoms.Number",&atomnum2,0);

  if (atomnum1!=atomnum2){
    printf("The number of atoms is different from each other.\n");
    exit(0);
  } 

  if (fp=input_find("Decomposed.energies.(Hartree).with.respect.to.atom") ) {

    fgets(buf,MAXBUF,fp);
    fgets(buf,MAXBUF,fp);

    for (i=1; i<=atomnum1; i++){
      fgets(buf,MAXBUF,fp);
      sscanf(buf,"%d %s %lf",&j,ctmp,&tmp1);
      Gxyz1[i][4] += tmp1;
    }
  }

  input_close();

  /*******************************************
   write results in the third file: file2.de
  *******************************************/

  if ((fp1 = fopen(argv[3],"w")) != NULL){
    fprintf(fp1,"%14.7f %14.7f %14.7f\n",tv[1][1],tv[1][2],tv[1][3]);
    fprintf(fp1,"%14.7f %14.7f %14.7f\n",tv[2][1],tv[2][2],tv[2][3]);
    fprintf(fp1,"%14.7f %14.7f %14.7f\n",tv[3][1],tv[3][2],tv[3][3]);

    fprintf(fp1,"%d\n",atomnum1);

    for (i=1; i<=atomnum1; i++){
      fprintf(fp1,"%4d %3s %10.5f %10.5f %10.5f %10.5f\n",
                  i,AtomChar[i],Gxyz1[i][1],Gxyz1[i][2],Gxyz1[i][3],Gxyz1[i][4]*27.2113845);
    }

    fclose(fp1);
  }
  else{
    printf("error in saving %s\n",argv[3]);
  }

  /* freeing of arrays */

  for (i=0; i<(atomnum1+1); i++){
    free(Gxyz1[i]);
  }
  free(Gxyz1);

  for (i=0; i<(atomnum1+1); i++){
    free(AtomChar[i]);
  }
  free(AtomChar);

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


int input_cmpstring(const char *str,  int *ret, int nvals, char **strval, int *ivals)
{
  int i;
  int lenstr,lenstrval;

  /*
  for (i=0;i<nvals;i++) {
    printf("<%s> <%d>\n",strval[i],ivals[i]);
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
#define BUFSIZE 500

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

