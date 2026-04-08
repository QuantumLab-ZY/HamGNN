/*  
  H. Kino

  usage: 

  > bandgnu  foo.Band 
  foo.Band is an output of openmx

  > gnuplot foo.GNUBAND


*  2002/5/13 bug fix, and change gnuplot output

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAXBUF 1024
#define fp_bsize         1048576     /* buffer size for setvbuf */

/* #define DEBUG  */

#ifdef DEBUG
#undef DEBUG
#endif


double len3(double a[4], double b[4])
{
  int i;
  double r,v[4];
  
  r=0.0;
  for (i=1;i<=3;i++) {
    v[i]=a[i]-b[i];
    r+= v[i]*v[i];
  }
  return sqrt(r);
}

void vk_rtv(double vk[4], double rtv[4][4])
{
   /* vk[1:3], rtv[1:3][1:3] */
  int i,j;
  double v[4];
#ifdef DEBUG
  printf("rtv\n");
  for (i=1;i<=3;i++) {
     printf("%lf ",vk[i]);
  } printf("\n");
  for (i=1;i<=3;i++) {
    for (j=1;j<=3;j++) {
      printf("%lf ",rtv[i][j]);
    }
  } printf("\n");
#endif
  for (i=1;i<=3;i++) {
    v[i]=0.0;
    for (j=1;j<=3;j++) {
       v[i] = v[i] + vk[j]*rtv[j][i] ;
    }
  }
  for (i=1;i<=3;i++) vk[i]=v[i];

}






#define MUNIT 2

main(int argc, char **argv)
{
  static double Unit0[MUNIT]={27.2113845,1.0}; /* Hartree-> eV */
  static char *Unitname0[MUNIT]={"eV","Hartree"};

  int iunit;
  double Unit;
  char *Unitname;

  int maxneig, mspin;
  double ChemP;
  char buf[MAXBUF];
  FILE *fp;
  double rtv[4][4];
 
  static int m_perpath,i_perpath;
  int nkpath,  *n_perpath;
  double ***kpath;
  char ***kname;
  double ****EIGEN;
  double **kline;

  int i,j,k,l,spin;
  int ik,n1,meig;
  double vk[4], ovk[3][4];

  double ymin,ymax,ymin1,ymax1;
  char fname[200],fnamedat1[3][200],*pos_dot,ticsname[200];
  char fp_buf[fp_bsize];          /* setvbuf */


  if (argc!=2) {
    printf("usage: %s foo.Band\n",argv[0]);
    exit(10);
  }

  iunit=0;
  Unit=Unit0[iunit];
  Unitname=Unitname0[iunit];

  if ( (fp=fopen(argv[1],"r") ) ==0 ) {

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("can not open a file: %s\n",argv[1]);
    exit(10);
  }

  fscanf(fp,"%d %d  %lf", &maxneig,&mspin,&ChemP);

#ifdef DEBUG
  printf("%d %d %lf\n",maxneig,mspin,ChemP);
#endif


  fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",
	 &rtv[1][1], &rtv[1][2], &rtv[1][3],
	 &rtv[2][1], &rtv[2][2], &rtv[2][3],
	 &rtv[3][1], &rtv[3][2], &rtv[3][3] );
#ifdef DEBUG
  printf(" %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
	 rtv[1][1],  rtv[1][2],  rtv[1][3],
	 rtv[2][1],  rtv[2][2],  rtv[2][3],
	 rtv[3][1],  rtv[3][2],  rtv[3][3] );
#endif
 
  fscanf(fp,"%d",&nkpath);
#ifdef DEBUG
  printf("nkpath=%d\n",nkpath);
#endif

  n_perpath = (int*)malloc(sizeof(int)*(nkpath+1));

  kpath = (double***)malloc(sizeof(double**)*(nkpath+1));
  for (i=0;i<=nkpath;i++) {
    kpath[i]= (double**)malloc(sizeof(double*)*4);
    for (j=0;j<=3;j++) {
      kpath[i][j]=(double*)malloc(sizeof(double)*4);
    }
  }

  kname = (char***)malloc(sizeof(char**)*(nkpath+1));
  for (i=0;i<=nkpath;i++) {
    kname[i]= (char**)malloc(sizeof(char*)*4);
    for (j=0;j<=3;j++) {
      kname[i][j]= (char*)malloc(sizeof(char)*(100));
    }
  }


  for (i=1;i<=nkpath;i++) {
    fscanf(fp,"%d %lf %lf %lf  %lf %lf %lf  %s %s",
	   &n_perpath[i],
	   &kpath[i][1][1], &kpath[i][1][2], &kpath[i][1][3],
	   &kpath[i][2][1], &kpath[i][2][2], &kpath[i][2][3],
	   kname[i][1],kname[i][2]);
#ifdef DEBUG
    printf("%d %lf %lf %lf  %lf %lf %lf  %s %s\n",
           n_perpath[i],
           kpath[i][1][1],  kpath[i][1][2],  kpath[i][1][3],
           kpath[i][2][1],  kpath[i][2][2],  kpath[i][2][3],
	   kname[i][1],kname[i][2]);
#endif

  }

  m_perpath=0;
  for (i=1;i<=nkpath;i++) if (m_perpath<n_perpath[i]) m_perpath=n_perpath[i];
#ifdef DEBUG
  printf("m_perpath=%d\n",m_perpath);
#endif
  EIGEN = (double****)malloc(sizeof(double***)*(mspin+1));
  for (i=0;i<mspin+1;i++) {
    EIGEN[i]=(double***)malloc(sizeof(double**)*(nkpath+1));
    for (j=0;j<(nkpath+1);j++) {
      EIGEN[i][j]=(double**)malloc(sizeof(double*)*(m_perpath+1));
      for (k=0;k<(m_perpath+1);k++) {
	EIGEN[i][j][k]=(double*)malloc(sizeof(double)*(maxneig+1));
	for (l=0;l<=maxneig;l++) {
	  EIGEN[i][j][k][l]=0.0;
	}
      }
    }
  }

  kline=(double**)malloc(sizeof(double*)*(nkpath+1));
  for (j=0;j<(nkpath+1);j++) {
    kline[j]=(double*)malloc(sizeof(double)*(m_perpath+1));
  }

#ifdef DEBUG
  printf("<kpath end>\n");
#endif

  meig=maxneig;

  kline[1][1]=0.0;


  for (ik=1;ik<=nkpath;ik++) {
    for (i_perpath=1;i_perpath<=n_perpath[ik]; i_perpath++) {
      for (spin=0;spin<=mspin;spin++) {
#ifdef DEBUG
	printf("ik,i_perpath,spin= %d %d %d\n",ik,i_perpath,spin);
#endif
	fscanf(fp,"%d %lf %lf %lf\n", &n1,&vk[1],&vk[2],&vk[3]);
#ifdef DEBUG
	printf("%d %lf %lf %lf\n",n1,vk[1],vk[2],vk[3]);
#endif

	vk_rtv(vk,rtv);
#ifdef DEBUG
	printf("%d %lf %lf %lf\n",n1,vk[1],vk[2],vk[3]);
#endif
	if (i_perpath==1) {
	  if (ik==1) {
	    kline[ik][i_perpath]=0.0;
	  }
	  else {
	    kline[ik][i_perpath] =kline[ik-1][n_perpath[ik-1]];
	  }
	}
	else {
	  kline[ik][i_perpath] = kline[ik][i_perpath-1]+len3(vk,ovk[spin]);
	}
#ifdef DEBUG
	printf("kline= %d %d %lf\n",ik,i_perpath,kline[ik][i_perpath]);
#endif
	for (i=1;i<=3;i++) ovk[spin][i] = vk[i];
	if (meig>n1) meig=n1;
	for (l=1;l<=n1;l++) {
	  fscanf(fp,"%lf",&EIGEN[spin][ik][i_perpath][l]);
	}
#ifdef DEBUG
	for (l=1;l<=n1;l++) {
	  printf("%lf ",EIGEN[spin][ik][i_perpath][l]);
	}
	printf("\n");
#endif

      }
    }
  }

  ymin=100000.0;
  for (ik=1;ik<=nkpath;ik++) {
    for (i_perpath=1;i_perpath<=n_perpath[ik]; i_perpath++) {
      for (spin=0;spin<=mspin;spin++) {
	if (ymin> EIGEN[spin][ik][i_perpath][1]) {
	  ymin=EIGEN[spin][ik][i_perpath][1];
	} 
      }
    }
  } 
  ymax=-100000.0;
  for (ik=1;ik<=nkpath;ik++) {
    for (i_perpath=1;i_perpath<=n_perpath[ik]; i_perpath++) {
      for (spin=0;spin<=mspin;spin++) {
	if (ymax< EIGEN[spin][ik][i_perpath][1]) {
	  ymax=EIGEN[spin][ik][i_perpath][meig]; 
	}
      }
    }
  }

  ymax1=(ymax-ymin)*1.1+ymin;
  ymin1=-(ymax-ymin)*1.1+ymax;


  fclose(fp);


  for (spin=0;spin<= mspin;spin++) {

    strcpy(fnamedat1[spin],argv[1]);
    pos_dot =rindex(fnamedat1[spin],'.');
    if (pos_dot) *pos_dot='\0';
    sprintf(fnamedat1[spin],"%s%s%d",fnamedat1[spin],".BANDDAT",spin+1);


    if ( (fp=fopen(fnamedat1[spin],"w"))==NULL ) {

#ifdef xt3
      setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open a file %s\n",fname);
      exit(10);
    }


    for (ik=2;ik<=nkpath;ik++){
      k = 0;
      for (i=0;i<=10;i++) {
	if (((ymin1-ChemP)*Unit+(ymax1-ymin1)*Unit*(double)i/10) > 0 && k == 0)
	  {
	    fprintf(fp, "%lf %lf\n", kline[ik][1], 0.0);
	    k = 1;
	  }

	fprintf(fp, "%lf %lf\n", kline[ik][1],
		(ymin1-ChemP)*Unit+(ymax1-ymin1)*Unit*(double)i/10);
      }
      fprintf(fp, "\n\n");
    }


    /*
    for (ik=2;ik<=nkpath;ik++) {
      for (i=0;i<=10;i++) {
        fprintf(fp, "%lf %lf\n", kline[ik][1], 
		(ymin1-ChemP)*Unit+(ymax1-ymin1)*Unit*(double)i/10);

        

      }
      fprintf(fp, "\n\n");
    }
    */


#if 0
    fprintf(fp, "%lf %lf\n", kline[nkpath][n_perpath[nkpath]],
	    (ymin1-ChemP)*Unit);
    fprintf(fp, "%lf %lf\n", kline[nkpath][n_perpath[nkpath]], 
	    (ymax1-ChemP)*Unit);

    fprintf(fp, "\n\n");
#endif



    for(l=1;l<=meig;l++) {
      for (ik=1;ik<=nkpath;ik++) {
	for (i_perpath=1;i_perpath<=n_perpath[ik]; i_perpath++) {
	  fprintf(fp, "%lf %15.12f\n",kline[ik][i_perpath],
		  (EIGEN[spin][ik][i_perpath][l]-ChemP)*Unit );
	}
	fprintf(fp, "\n\n");
      }
    } 

    fclose(fp);


  }

  strcpy(fname,argv[1]);
  pos_dot =rindex(fname,'.');
  if (pos_dot) *pos_dot='\0';
#if 0
  printf("name=%s\n",fname);
#endif
  sprintf(fname,"%s%s",fname,".GNUBAND");
  if ( (fp=fopen(fname,"w"))==NULL ) {

#ifdef xt3
    setvbuf(fp,fp_buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("can not open a file %s\n",fname);
    exit(10);
  }
  printf("%s is made\n",fname);

  fprintf(fp,"set style data lines\n");
  fprintf(fp,"set nokey\n");
  fprintf(fp,"set zeroaxis\n");
  fprintf(fp,"set ytics 1\n");
  fprintf(fp,"set mytics 5\n");

  fprintf(fp,"set xra [%lf:%lf]\n",
          kline[1][1],kline[nkpath][n_perpath[nkpath]]);
  fprintf(fp,"set yra [%lf:%lf]\n",
	  (ymin1-ChemP)*Unit, (ymax1-ChemP)*Unit);

  fprintf(fp,"set ylabel \"%s\"\n",Unitname);

  fprintf(fp,"set xtics (");
  for (ik=1;ik<=nkpath;ik++) {
    if (ik==1) {
      strcpy(ticsname,kname[ik][1]);
    }
    else {
      if ( strcmp(kname[ik][1],kname[ik-1][2])==0 ) {
	strcpy(ticsname,kname[ik][1]);
      }
      else {
	sprintf(ticsname,"%s,%s",kname[ik-1][2],kname[ik][1]);  
      }
    }
    fprintf(fp,"\"%s\" %lf, ", ticsname,kline[ik][1]);
  }
  fprintf(fp,"\"%s\" %lf)\n", kname[nkpath][2],  kline[nkpath][n_perpath[nkpath]] );


  fprintf(fp,"plot \"%s\"",fnamedat1[0]);
  if (mspin==1) {
    fprintf(fp,", \"%s\"\n",fnamedat1[1]);
  }
  else {
    fprintf(fp,"\n");
  }


  fprintf(fp,"pause -1");

  fclose(fp);


  /* free arrays */ 

  free(n_perpath);

  for (i=0;i<=nkpath;i++) {
    for (j=0;j<=3;j++) {
      free(kpath[i][j]);
    }
    free(kpath[i]);
  }
  free(kpath);

  for (i=0;i<=nkpath;i++) {
    for (j=0;j<=3;j++) {
      free(kname[i][j]);
    }
    free(kname[i]);
  }
  free(kname);

  for (i=0;i<mspin+1;i++) {
    for (j=0;j<(nkpath+1);j++) {
      for (k=0;k<(m_perpath+1);k++) {
	free(EIGEN[i][j][k]);
      }
      free(EIGEN[i][j]);
    }
    free(EIGEN[i]);
  }
  free(EIGEN);

  for (j=0;j<(nkpath+1);j++) {
    free(kline[j]);
  }
  free(kline);

}
