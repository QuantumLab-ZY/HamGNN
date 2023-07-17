/**********************************************************************
  bin2txt.c:

  Log of bin2txt.c:

    29/Dec/2012  Released by T. Ozaki 
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void convert_cube(char *fname);
void convert_xsf(char *fname1);
void convert_grid(char *fname1);



int main(int argc, char *argv[]) 
{
  int i;

  /* check the number of arguments */

  if (argc<2){
    printf("Invalid argument\n");
    exit(0);
  } 

  /* read each file */

  for (i=1; i<argc; i++){

    /* in case of a cube file */

    if (strstr(argv[i],".cube.bin")!=NULL){
      printf("converting %s\n",argv[i]);
      convert_cube(argv[i]);
    }

    /* in case of a ncsden.xsf file */

    else if (strstr(argv[i],".xsf.bin")!=NULL){
      printf("converting %s\n",argv[i]);
      convert_xsf(argv[i]);
    }
   
    /* in case of a grid file */

    else if (strstr(argv[i],".grid.bin")!=NULL){
      printf("converting %s\n",argv[i]);
      convert_grid(argv[i]);
    }

  }

}



void convert_cube(char *fname1)
{
  FILE *fp1,*fp2;
  int i,j,k,n,ct_AN,atomnum,Ngrid1,Ngrid2,Ngrid3;
  double Grid_Origin[4],gtv[4][4];
  char clist[300];
  char fname2[300];
  char *p;
  double *dlist;

  i = 0;
  for (p=&fname1[0]; p<strstr(fname1,".bin"); p++) fname2[i++] = *p;   
  fname2[i] = '\0';

  if ((fp1 = fopen(fname1,"rb")) != NULL){

    if ((fp2 = fopen(fname2,"w")) != NULL){

      fread(clist, sizeof(char), 200, fp1);
      fprintf(fp2,"%s",clist);
      fread(clist, sizeof(char), 200, fp1);
      fprintf(fp2,"%s",clist);

      fread(&atomnum, sizeof(int), 1, fp1);
      fread(&Grid_Origin[1], sizeof(double), 3, fp1);
      fread(&Ngrid1, sizeof(int), 1, fp1);
      fread(&gtv[1][1], sizeof(double), 3, fp1);
      fread(&Ngrid2, sizeof(int), 1, fp1);
      fread(&gtv[2][1], sizeof(double), 3, fp1);
      fread(&Ngrid3, sizeof(int), 1, fp1);
      fread(&gtv[3][1], sizeof(double), 3, fp1);

      fprintf(fp2,"%5d%12.6lf%12.6lf%12.6lf\n",
	      atomnum,Grid_Origin[1],Grid_Origin[2],Grid_Origin[3]);
      fprintf(fp2,"%5d%12.6lf%12.6lf%12.6lf\n",
	      Ngrid1,gtv[1][1],gtv[1][2],gtv[1][3]);
      fprintf(fp2,"%5d%12.6lf%12.6lf%12.6lf\n",
	      Ngrid2,gtv[2][1],gtv[2][2],gtv[2][3]);
      fprintf(fp2,"%5d%12.6lf%12.6lf%12.6lf\n",
	      Ngrid3,gtv[3][1],gtv[3][2],gtv[3][3]);

      /* fread Gxyz and fprintf them */

      dlist = (double*)malloc(sizeof(double)*atomnum*5);
      fread(dlist, sizeof(double), atomnum*5, fp1);

      n = 0;
      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	fprintf(fp2,"%5d%12.6lf%12.6lf%12.6lf%12.6lf\n",
		(int)dlist[n  ],
                     dlist[n+1],
                     dlist[n+2],
                     dlist[n+3],
                     dlist[n+4]);
        n += 5;
      }

      free(dlist);

      /* fread data on grid and fprintf them */

      dlist = (double*)malloc(sizeof(double)*Ngrid2*Ngrid3);

      for (i=0; i<Ngrid1; i++){

        fread(dlist, sizeof(double), Ngrid2*Ngrid3, fp1);

        n = 0;
        for (j=0; j<Ngrid2; j++){
          for (k=0; k<Ngrid3; k++){
	    fprintf(fp2,"%13.3E",dlist[n]);
            n++;

	    if ((k+1)%6==0) { fprintf(fp2,"\n"); }
  	  }
	  /* avoid double \n\n when Ngrid3%6 == 0 */
	  if (Ngrid3%6!=0) fprintf(fp2,"\n");
	}
      }

      free(dlist);

      fclose(fp2);
    }

    fclose(fp1);

    sprintf(clist,"rm %s",fname1);
    system(clist);
  }  
  else{
    printf("Failure of reading %s.\n",fname1);
    exit(0);
  }
}



void convert_xsf(char *fname1)
{
  FILE *fp1,*fp2;
  int i,j,k,n,ct_AN,atomnum;
  int GridNum,Ngrid1,Ngrid2,Ngrid3;
  double Grid_Origin[4],tv[4][4];
  double xyz[4];
  char clist[300];
  char fname2[300];
  char *p;
  double *dlist;

  i = 0;
  for (p=&fname1[0]; p<strstr(fname1,".bin"); p++) fname2[i++] = *p;   
  fname2[i] = '\0';

  if ((fp1 = fopen(fname1,"rb")) != NULL){

    if ((fp2 = fopen(fname2,"w")) != NULL){

      fread(clist, sizeof(char), 200, fp1);
      fprintf(fp2,"%s",clist);
      fread(clist, sizeof(char), 200, fp1);
      fprintf(fp2,"%s",clist);

      fread(&tv[1][1], sizeof(double), 3, fp1);
      fread(&tv[2][1], sizeof(double), 3, fp1);
      fread(&tv[3][1], sizeof(double), 3, fp1);

      fprintf(fp2," %10.6f %10.6f %10.6f\n",tv[1][1], tv[1][2], tv[1][3]);
      fprintf(fp2," %10.6f %10.6f %10.6f\n",tv[2][1], tv[2][2], tv[2][3]);
      fprintf(fp2," %10.6f %10.6f %10.6f\n",tv[3][1], tv[3][2], tv[3][3]);

      fread(&atomnum, sizeof(int), 1, fp1);
      fread(&GridNum, sizeof(int), 1, fp1);
      
      fprintf(fp2,"PRIMCOORD\n");
      fprintf(fp2,"%4d 1\n",atomnum+GridNum);

      /* fread Gxyz and fprintf them */

      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

	fread(clist, sizeof(char), 4, fp1);
	fread(&xyz[1], sizeof(double), 3, fp1);
	fprintf(fp2, "%s   %8.5f  %8.5f  %8.5f   0.0 0.0 0.0\n",
		clist,xyz[1],xyz[2],xyz[3]);
      }

      /* fread vector data on grid and fprintf them */

      dlist = (double*)malloc(sizeof(double)*GridNum*6);
      fread(dlist, sizeof(double), GridNum*6, fp1);

      for (i=0; i<GridNum; i++){

	fprintf(fp2,"X %13.3E %13.3E %13.3E %13.3E %13.3E %13.3E\n",
		dlist[6*i  ],
		dlist[6*i+1],
		dlist[6*i+2],
		dlist[6*i+3],
		dlist[6*i+4],
		dlist[6*i+5]);
      }

      free(dlist);
      fclose(fp2);
    }

    fclose(fp1);

    sprintf(clist,"rm %s",fname1);
    system(clist);
  }  
  else{
    printf("Failure of reading %s.\n",fname1);
    exit(0);
  }
}




void convert_grid(char *fname1)
{
  FILE *fp1,*fp2;
  int i,j,k,n,n1,n2,n3,Ngrid1,Ngrid2,Ngrid3;
  double Grid_Origin[4],gtv[4][4];
  char clist[300];
  char fname2[300];
  char *p;
  double x,y,z; 
  double *dlist;

  i = 0;
  for (p=&fname1[0]; p<strstr(fname1,".bin"); p++) fname2[i++] = *p;   
  fname2[i] = '\0';

  if ((fp1 = fopen(fname1,"rb")) != NULL){

    if ((fp2 = fopen(fname2,"w")) != NULL){

      fread(&Ngrid1, sizeof(int), 1, fp1);
      fread(&Ngrid2, sizeof(int), 1, fp1);
      fread(&Ngrid3, sizeof(int), 1, fp1);

      dlist = (double*)malloc(sizeof(double)*Ngrid2*Ngrid3*3);

      n = 0;
      for (n1=0; n1<Ngrid1; n1++){

        fread(dlist, sizeof(double), Ngrid2*Ngrid3*3, fp1);

        i = 0;
        for (n2=0; n2<Ngrid2; n2++){
          for (n3=0; n3<Ngrid3; n3++){
            x = dlist[i]; i++; 
            y = dlist[i]; i++; 
            z = dlist[i]; i++; n++; 
	    fprintf(fp2,"%5d  %19.12f %19.12f %19.12f\n", n-1,x,y,z);
          }
        }
      }

      free(dlist);
      fclose(fp2);
    }

    fclose(fp1);

    sprintf(clist,"rm %s",fname1);
    system(clist);
  }  
  else{
    printf("Failure of reading %s.\n",fname1);
    exit(0);
  }
}
