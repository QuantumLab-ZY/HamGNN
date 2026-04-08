/**********************************************************************
  expao.c:

     expao.c is a program which creats a new PAO file by merging two 
     PAO files. 

      Compile: 

          gcc expao.c -o expao

      Usage:

         ./expao In1.pao In2.pao Out.pao

         Then, you will be interactively asked as 

         How many PAOs do you employ from In2.pao with L= 0 (Max: 5) :    2
         which means that PAOs with L= 0 consist of In1.pao (0 to  2) and In2.pao (0 to  1)

         How many PAOs do you employ from In2.pao with L= 1 (Max: 5) :    1
         which means that PAOs with L= 1 consist of In1.pao (0 to  3) and In2.pao (0 to  0)

         How many PAOs do you employ from In2.pao with L= 2 (Max: 5) :    3
         which means that PAOs with L= 2 consist of In1.pao (0 to  1) and In2.pao (0 to  2)

         How many PAOs do you employ from In2.pao with L= 3 (Max: 5) :    2
         which means that PAOs with L= 3 consist of In1.pao (0 to  2) and In2.pao (0 to  1)

      Input: In1.pao, In2.pao
      Output: Out.pao

  Log of expao.c:

     27/Aug/2010  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXNEST   14

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
  int i,j,L,Mul;
  FILE *fp1,*fp2;
  int PAO_Lmax1,PAO_Mul1,Mesh1;
  int PAO_Lmax2,PAO_Mul2,Mesh2;
  int exflag[20];
  double ***PAO1,***PAO2;
  double *XV,*RV;
  double dum;
  char buf[1000];
  char file1[200],file2[200],file3[200];
  char st[3000];
  char st1[3000];

  if (argc!=4){
    printf("Usage:\n");
    printf("  ./expao In1.pao In2.pao Out.pao\n");
    exit(0);
  }

  /*******************************************
              read the first file 
  *******************************************/

  input_open(argv[1]);

  input_int("PAO.Lmax",&PAO_Lmax1,0);
  input_int("PAO.Mul",&PAO_Mul1,0);

  input_int("grid.num.output",&Mesh1,0);

  PAO1 = (double***)malloc(sizeof(double**)*(PAO_Lmax1+1)); 
  for (i=0; i<(PAO_Lmax1+1); i++){
    PAO1[i] = (double**)malloc(sizeof(double*)*PAO_Mul1); 
    for (j=0; j<PAO_Mul1; j++){
      PAO1[i][j] = (double*)malloc(sizeof(double)*Mesh1); 
    }
  }

  XV = (double*)malloc(sizeof(double)*Mesh1); 
  RV = (double*)malloc(sizeof(double)*Mesh1); 

  for (L=0; L<=PAO_Lmax1; L++){  

    sprintf(file1,"<pseudo.atomic.orbitals.L=%i",L);
    sprintf(file2,"pseudo.atomic.orbitals.L=%i>",L);

    if (fp=input_find(file1)) {
      for (i=0; i<Mesh1; i++){
	for (j=0; j<=(PAO_Mul1+1); j++){
	  if (fscanf(fp,"%lf",&dum)==EOF){
	    printf("File error in pseudo.atomic.orbitals.L=%i\n",L);
	  }
	  else{
	    if (j==0)
	      XV[i] = dum;
	    else if (j==1)
	      RV[i] = dum;
	    else
	      PAO1[L][j-2][i] = dum;
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
              read the second file 
  *******************************************/

  input_open(argv[2]);

  input_int("PAO.Lmax",&PAO_Lmax2,0);
  input_int("PAO.Mul",&PAO_Mul2,0);
  input_int("grid.num.output",&Mesh2,0);

  if (PAO_Lmax1!=PAO_Lmax2){
    printf("PAO.Lmax1 is inconsistent.\n");
    exit(0);
  }   

  if (PAO_Mul1!=PAO_Mul2){
    printf("PAO.Mul1 is inconsistent.\n");
    exit(0);
  }   

  if (Mesh1!=Mesh2){
    printf("grid.num.output is inconsistent.\n");
    exit(0);
  }   

  PAO2 = (double***)malloc(sizeof(double**)*(PAO_Lmax2+1)); 
  for (i=0; i<(PAO_Lmax2+1); i++){
    PAO2[i] = (double**)malloc(sizeof(double*)*PAO_Mul2); 
    for (j=0; j<PAO_Mul2; j++){
      PAO2[i][j] = (double*)malloc(sizeof(double)*Mesh2); 
    }
  }

  for (L=0; L<=PAO_Lmax2; L++){  

    sprintf(file1,"<pseudo.atomic.orbitals.L=%i",L);
    sprintf(file2,"pseudo.atomic.orbitals.L=%i>",L);

    if (fp=input_find(file1)) {
      for (i=0; i<Mesh2; i++){
	for (j=0; j<=(PAO_Mul2+1); j++){
	  if (fscanf(fp,"%lf",&dum)==EOF){
	    printf("File error in pseudo.atomic.orbitals.L=%i\n",L);
	  }
	  else{
	    if (j==0)
	      XV[i] = dum;
	    else if (j==1)
	      RV[i] = dum;
	    else
	      PAO2[L][j-2][i] = dum;
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
              interactive queries
  *******************************************/

  printf("%s %s %s %s\n",argv[0],argv[1],argv[2],argv[3]);

  for (L=0; L<=PAO_Lmax1; L++){  

    printf("\n");
    printf("How many PAOs do you employ from %s with L=%2d (Max:%2d) :    ",argv[2],L,PAO_Mul1);
    fgets(buf,1000,stdin); sscanf(buf,"%d",&exflag[L]);
    printf("which means that PAOs with L=%2d consist of %s (0 to %2d) and %s (0 to %2d)\n",
            L,argv[1],PAO_Mul1-exflag[L]-1,argv[2],exflag[L]-1);
  }

  /*******************************************
                output a file 
  *******************************************/

  /* the new pao file */    

  fp1 = fopen(argv[3],"w");
  fseek(fp1,0,SEEK_END);

  /* the original pao file */    

  fp2 = fopen(argv[1],"r");

  if (fp2!=NULL){

    while (fgets(st,800,fp2)!=NULL){

      string_tolower(st,st1); 

      /* find PAO.Lmax */

      if (strncmp(st1,"pao.lmax",8)==0){

        fprintf(fp1,"PAO.Lmax   %d\n",PAO_Lmax1);
        fprintf(fp1,"PAO.Mul    %d\n",PAO_Mul1);

        for (L=0; L<=PAO_Lmax1; L++){  

          fprintf(fp1,"<pseudo.atomic.orbitals.L=%d\n",L);
           
          for (i=0; i<Mesh1; i++){

            fprintf(fp1,"%18.15f %18.15f ",XV[i],RV[i]);

            for (Mul=0; Mul<(PAO_Mul1-exflag[L]); Mul++){
              fprintf(fp1,"%18.15f ",PAO1[L][Mul][i]);
	    }

            for (Mul=0; Mul<exflag[L]; Mul++){
              fprintf(fp1,"%18.15f ",PAO2[L][Mul][i]);
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

  for (i=0; i<(PAO_Lmax1+1); i++){
    for (j=0; j<PAO_Mul1; j++){
      free(PAO1[i][j]);
    }
    free(PAO1[i]);
  }
  free(PAO1);

  for (i=0; i<(PAO_Lmax2+1); i++){
    for (j=0; j<PAO_Mul2; j++){
      free(PAO2[i][j]);
    }
    free(PAO2[i]);
  }
  free(PAO2);

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




