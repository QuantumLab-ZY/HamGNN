/**********************************************************************
  iterout_md.c:

     iterout_md.c is a subroutine to output calculated quantities
     at each MD step to filename.ene.

  Log of iterout_md.c:

     20/Sep./2007  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include "openmx_common.h"

void iterout_md(int iter, double drctime, char fileSE[YOUSO10])
{
  int i,j,k;
  double dt,itermax,aa,dx,dy,dz,xido,yido,zido;
  FILE *fp;
  char buf[fp_bsize];          /* setvbuf */
  char fileEvsLC[YOUSO10];  

  if ((fp = fopen(fileSE,"a")) != NULL){

    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */

    /*  1 */      fprintf(fp,"%5d ",iter);
    /*  2 */      fprintf(fp,"%10.7f ",drctime);
    /*  3 */      fprintf(fp,"%10.7f ",Uele_OS0);
    /*  4 */      fprintf(fp,"%10.7f ",Uele_IS0);
    /*  5 */      fprintf(fp,"%10.7f ",Uele_OS1);
    /*  6 */      fprintf(fp,"%10.7f ",Uele);
    /*  7 */      fprintf(fp,"%10.7f ",Uxc0);
    /*  8 */      fprintf(fp,"%10.7f ",Uxc1);
    /*  9 */      fprintf(fp,"%10.7f ",UH0);
    /* 10 */      fprintf(fp,"%10.7f ",UH1);
    /* 11 */      fprintf(fp,"%10.7f ",Ucore);
    /* 12 */      fprintf(fp,"%10.7f ",Udc);
    /* 13 */      fprintf(fp,"%10.7f ",Ucoh);
    /* 14 */      fprintf(fp,"%10.7f ",Ukc);
    /* 15 */      fprintf(fp,"%10.7f ",Utot);
    /* 16 */      fprintf(fp,"%10.7f ",Utot+Ukc);
    /* 17 */      fprintf(fp,"%10.7f ",ChemP);
    /* 18 */      fprintf(fp,"%10.7f ",GivenTemp);
    /* 19 */      fprintf(fp,"%10.7f ",Temp);
    /* 20 */      fprintf(fp,"%10.7f ",NH_czeta);
    /* 21 */      fprintf(fp,"%10.7f ",NH_nzeta);
    /* 22 */      fprintf(fp,"%10.7f ",NH_Ham);
    /* 23 */      fprintf(fp,"%10.7f ",tv[1][1]*BohrR);
    /* 24 */      fprintf(fp,"%10.7f ",tv[1][2]*BohrR); 
    /* 25 */      fprintf(fp,"%10.7f ",tv[1][3]*BohrR);
    /* 26 */      fprintf(fp,"%10.7f ",tv[3][3]*BohrR);
    /* 27 */      fprintf(fp,"%10.7f ",Weight/Cell_Volume*10.0/6.022); 
    /* 28 */      fprintf(fp,"%10.7f ",GP*1.60219*100);
    /* 29 */      fprintf(fp,"%10.7f ",GT);
    /* 30 */      fprintf(fp,"%17.15f ",Max_Force);
    /* 31 */      fprintf(fp,"%10.7f ",Total_SpinS*2.0);
    /* 32 */      fprintf(fp,"%10.7f ",ChemP);
    /* 33 */      fprintf(fp,"%10.7f ", Gxyz[1][17]);
    /* 34 */      fprintf(fp,"%10.7f\n",Gxyz[1][17]);
    fclose(fp);
  }
  else
    printf("error in saving *.ene\n");

  /* E vs. LC */

  if (MD_switch==12){

    sprintf(fileEvsLC,"%s%s.EvsLC",filepath,filename);

    if ((fp = fopen(fileEvsLC,"a")) != NULL){

      fprintf(fp,"%15.10f ",tv[1][1]*BohrR);
      fprintf(fp,"%15.10f ",tv[1][2]*BohrR); 
      fprintf(fp,"%15.10f ",tv[1][3]*BohrR);
      fprintf(fp,"%15.10f ",tv[2][1]*BohrR);
      fprintf(fp,"%15.10f ",tv[2][2]*BohrR); 
      fprintf(fp,"%15.10f ",tv[2][3]*BohrR);
      fprintf(fp,"%15.10f ",tv[3][1]*BohrR);
      fprintf(fp,"%15.10f ",tv[3][2]*BohrR); 
      fprintf(fp,"%15.10f ",tv[3][3]*BohrR);
      fprintf(fp,"%15.10f\n",Utot);

      fclose(fp);
    }
    else
      printf("error in saving *.EvsLC\n");

  }  

  /* for calculation of the delta factor */

  if (MD_switch==16){

    sprintf(fileEvsLC,"%s%s.DF",filepath,filename);

    if ((fp = fopen(fileEvsLC,"a")) != NULL){

      fprintf(fp,"%18.12f ",Cell_Volume*BohrR*BohrR*BohrR/(double)atomnum);
      fprintf(fp,"%18.12f\n",Utot*eV2Hartree/(double)atomnum);

      fclose(fp);
    }
    else
      printf("error in saving *.EvsLC\n");

  }  

}


