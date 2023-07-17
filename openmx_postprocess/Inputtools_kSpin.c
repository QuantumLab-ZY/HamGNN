/*************************************************** 
   Copyright (C), 2002, Hiori Kino.  
   This library follows the terms of GNU GPL.

   Aug. 14, 2003
   allow opening multiple files (only nesting)
***************************************************/

#include <stdio.h>
#include  <math.h>
#include <ctype.h>
#include <string.h>

#define MAXNEST   14

static FILE *fp=NULL;
static int  nestlevel=-1;
static FILE *fpnest[MAXNEST];

static int printlevel=0;
static int errorlevel=0;

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

/* Added by N. Yamaguchi ***/
#if 0
/* ***/

#define BUFSIZE 200

/* Added by N. Yamaguchi ***/
#endif
#define BUFSIZE 4096
/* ***/

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
  while ((c=fgets(buf,size,fp)) != NULL) { 
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
  while ((c=fgets(buf,size,fp)) != NULL) { 
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
  while ((c=fgets(buf,size,fp)) != NULL) {
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

