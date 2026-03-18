/**********************************************************************
  Cluster_DFT_ON2.c:

     Cluster_DFT_ON2.c is a subroutine to perform cluster calculations
     with an O(N^2~) method.

  Log of Cluster_DFT_ON2.c:

     11/Sep./2009  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"
#include "mpi.h"
#include <omp.h>

#define  measure_time       0


typedef struct {
   long int a;
   double complex b;
} clists;



static double Find_Trial_ChemP_by_Muller(
        int SCF_iter, int num_loop, 
        double pTrial_ChemP[4], double pDiff_Num[5], 
        double DeltaMu[7], double DeltaN[3], double DeltaNp[3]);

static void qsort_complex(long n, long *a, double complex *b);
static int clists_cmp(const clists *x, const clists *y);
int Lapack_LU_Zinverse(int , dcomplex *);

static void Nested_Dissection( int free_flag, int Nmat, int *MP, 
                               int *Depth_Level, int *Number_Division);

static double Cluster_collinear_ON2_Force(
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh, double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
				    double Eele0[2], double Eele1[2]);

static double Cluster_collinear_ON2_Iter(
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh, double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
				    double Eele0[2], double Eele1[2]);




static void OND_Solver(
  char *mode, 
  int myid,
  int numprocs,
  int myid2,
  int numprocs2,
  int Nloop1,
  MPI_Comm MPI_Comm,
  int spin,
  int Epoint,
  double Real_Ene,
  int Nmat,
  int *NZE,
  int *PZE,
  int nl, 
  int mpmax,
  double Trial_ChemP,
  double **Hall,
  double *Sall,
  double **DMall,
  double **WRG,
  int *conv_index_row,
  int *conv_index_col,
  int **conv_ind2,
  int TNZE);

static void Bisection_System(
    int Latomnum, 
    double **Cell_Gxyz2, 
    int **OG2G, 
    int **G2OG, 
    int *NN_FNAN,
    int **NN_natn,
    int **NN_ncn, 
    int *OGmin,
    int *OGmax, 
    int *num0, 
    int *numb,
    int *num1);




int **Index_BC;
int **SindB,**EindB;
int **SindC,**EindC;
int *invp;
int ***Bnum[2],***Cnum,**Anum;
int ****Bi[2],****Ci,***Ai;
double complex ****Bx[2];
double complex ****Cx;
double complex ****IBx[2];
double complex ****ICx;
double complex ***Ax;
double complex ***IAx;
dcomplex *IA0,***IS;
double **IA;
dcomplex ***Lvec[2];
double complex **Vvec,**Vvec2;
double complex **Q0,**Q1;
double *VvecTmp,*ISTmp;


double Cluster_DFT_ON2(char *mode,
		       int SCF_iter,
		       int SpinP_switch,
		       double *****nh,
		       double *****ImNL,
		       double ****CntOLP,
		       double *****CDM,
		       double *****EDM,
		       double Eele0[2], double Eele1[2])
{
  static double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){

    if (strcasecmp(mode,"scf")==0){

      time0 = Cluster_collinear_ON2_Iter( SCF_iter,SpinP_switch,
                                          nh,CntOLP,CDM,EDM,Eele0,Eele1 );

    }
   
    else if (strcasecmp(mode,"force")==0){

      time0 = Cluster_collinear_ON2_Force( SCF_iter,SpinP_switch,
                                           nh,CntOLP,CDM,EDM,Eele0,Eele1 );

    }

  }

  /****************************************************
           collinear with spin-orbit coupling
  ****************************************************/

  else if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==1 ){
    printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");
    MPI_Finalize();
    exit(1);
  }

  /****************************************************
   non-collinear with and without spin-orbit coupling
  ****************************************************/

  else if (SpinP_switch==3){
    /*
    time0 = Cluster_non_collinear(mode,SCF_iter,SpinP_switch,nh,ImNL,CntOLP,CDM,EDM,Eele0,Eele1);
    */
  }

  return time0;
}







static void Nested_Dissection(int free_flag, int Nmat, int *MP, int *Depth_Level, int *Number_Division)
{
  int i,j,k,l,n,nmin,nmax,ct_AN;
  int FNAN_min,po,n0,n1,nb,k0,i0,GA,p,k1;
  int Gc_AN,AN0,Mc_AN,Gh_AN,h_AN,OGh_AN,Rn;
  int OG_min,OG_max,OG_max0,diff_n,nmin0,nmax0;
  int Anum,wanA,Latomnum;
  int abc_n0[4],abc_nb[4],abc_n1[4];
  int abc_OG_max[4],abc_OG_min[4];
  int abc_nmax[4],abc_nmin[4];
  int *OGmin,*OGmax,**G2OG,**OG2G;
  double **Cell_Gxyz2;
  int avsize_cluster;
  int m,mp,mpmax,nl,np;
  int num0,numb,num1;
  int **Atom_Index_BC;
  int **Atom_SindB;
  int **Atom_EindB;
  int **Atom_SindC;
  int **Atom_EindC;
  int **NN_natn,**NN_ncn,*NN_FNAN,*MP2;

  /**********************************************
   calulate Depth_Level and Number_Division
   Depth_Level = nl
   Number_Division = mpmax
  **********************************************/

  avsize_cluster = 20;

  if (avsize_cluster<atomnum){
    np = atomnum/avsize_cluster;
    nl = log((double)np)/log(2.0);
    if (nl<0) nl = 0;
  }

  else {
    nl = 0;
  }

  mpmax = 1;
  for (i=0; i<nl; i++) mpmax = 2*mpmax;

  /************************************************
   if (free_flag==1) 
     free Index_BC, SindB, EindB, SindC, EindC
   else 
     allocate them
  ************************************************/

  if (free_flag==1){

    for (n=0; n<(nl+1); n++){
      free(Index_BC[n]);
    }
    free(Index_BC);

    for (n=0; n<(nl+1); n++){
      free(SindB[n]);
    }  
    free(SindB);

    for (n=0; n<(nl+1); n++){
      free(EindB[n]);
    }  
    free(EindB);

    for (n=0; n<nl; n++){
      free(SindC[n]);
    }  
    free(SindC);

    for (n=0; n<nl; n++){
      free(EindC[n]);
    }  
    free(EindC);

  }

  else {

    Index_BC = (int**)malloc(sizeof(int*)*(nl+1));
    for (n=0; n<(nl+1); n++){
      Index_BC[n] = (int*)malloc(sizeof(int)*Nmat);
      for (i=0; i<Nmat; i++){
	Index_BC[n][i] = i;
      }
    }

    SindB = (int**)malloc(sizeof(int*)*(nl+1));
    mp = mpmax;
    for (n=0; n<(nl+1); n++){
      SindB[n] = (int*)malloc(sizeof(int)*mp);
      for (m=0; m<mp; m++) SindB[n][m] = 0;
      mp /= 2;
    }  

    EindB = (int**)malloc(sizeof(int*)*(nl+1));
    mp = mpmax;
    for (n=0; n<(nl+1); n++){
      EindB[n] = (int*)malloc(sizeof(int)*mp);
      for (m=0; m<mp; m++) EindB[n][m] = 0;
      mp /= 2;
    }  

    SindC = (int**)malloc(sizeof(int*)*nl);
    mp = mpmax/2;
    for (n=0; n<nl; n++){
      SindC[n] = (int*)malloc(sizeof(int)*mp);
      for (m=0; m<mp; m++) SindC[n][m] = 0;
      mp /= 2;
    }  

    EindC = (int**)malloc(sizeof(int*)*nl);
    mp = mpmax/2;
    for (n=0; n<nl; n++){
      EindC[n] = (int*)malloc(sizeof(int)*mp);
      for (m=0; m<mp; m++) EindC[n][m] = 0;
      mp /= 2;
    }  
  }

  if (free_flag==1) return ;

  /* allocation of arrays */

  Atom_Index_BC = (int**)malloc(sizeof(int*)*(nl+1));
  for (n=0; n<(nl+1); n++){
    Atom_Index_BC[n] = (int*)malloc(sizeof(int)*(atomnum+1));
    for (i=0; i<(atomnum+1); i++){
      Atom_Index_BC[n][i] = i;
    }
  }

  Atom_SindB = (int**)malloc(sizeof(int*)*(nl+1));
  mp = mpmax;
  for (n=0; n<(nl+1); n++){
    Atom_SindB[n] = (int*)malloc(sizeof(int)*mp);
    for (m=0; m<mp; m++) Atom_SindB[n][m] = 0;
    mp /= 2;
  }  

  Atom_EindB = (int**)malloc(sizeof(int*)*(nl+1));
  mp = mpmax;
  for (n=0; n<(nl+1); n++){
    Atom_EindB[n] = (int*)malloc(sizeof(int)*mp);
    for (m=0; m<mp; m++) Atom_EindB[n][m] = 0;
    mp /= 2;
  }  

  Atom_SindC = (int**)malloc(sizeof(int*)*nl);
  mp = mpmax/2;
  for (n=0; n<nl; n++){
    Atom_SindC[n] = (int*)malloc(sizeof(int)*mp);
    for (m=0; m<mp; m++) Atom_SindC[n][m] = 0;
    mp /= 2;
  }  

  Atom_EindC = (int**)malloc(sizeof(int*)*nl);
  mp = mpmax/2;
  for (n=0; n<nl; n++){
    Atom_EindC[n] = (int*)malloc(sizeof(int)*mp);
    for (m=0; m<mp; m++) Atom_EindC[n][m] = 0;
    mp /= 2;
  }  

  Cell_Gxyz2 = (double**)malloc(sizeof(double*)*4);
  for (k=0; k<4; k++){
    Cell_Gxyz2[k] = (double*)malloc(sizeof(double)*(atomnum+1)); 
  }

  OG2G = (int**)malloc(sizeof(int*)*4);
  for (k=0; k<4; k++){
    OG2G[k] = (int*)malloc(sizeof(int)*(atomnum+1)); 
  }

  G2OG = (int**)malloc(sizeof(int*)*4);
  for (k=0; k<4; k++){
    G2OG[k] = (int*)malloc(sizeof(int)*(atomnum+1)); 
  }

  OGmin = (int*)malloc(sizeof(int)*(atomnum+1)); 
  OGmax = (int*)malloc(sizeof(int)*(atomnum+1)); 

  NN_ncn = (int**)malloc(sizeof(int*)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    NN_ncn[ct_AN]  = (int*)malloc(sizeof(int)*((int)(Max_FSNAN*ScaleSize)+1));
  }

  NN_natn = (int**)malloc(sizeof(int*)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    NN_natn[ct_AN] = (int*)malloc(sizeof(int)*((int)(Max_FSNAN*ScaleSize)+1));
  }

  NN_FNAN = (int*)malloc(sizeof(int)*(atomnum+1));
  MP2 = (int*)malloc(sizeof(int)*(atomnum+1));

  /**********************************************
      perform a nested dissection by bisection 
  **********************************************/

  Atom_SindB[nl][0] = 1;
  Atom_EindB[nl][0] = atomnum;

  mp = 1;

  /* n: level of hierarchy */

  for (n=(nl-1); 0<=n; n--){

    /* m: child number */

    for (m=0; m<mp; m++){

      /* construct the adjacency information */

      for (i=Atom_SindB[n+1][m]; i<=Atom_EindB[n+1][m]; i++){

        GA = Atom_Index_BC[n+1][i];
        i0 = 0;
 
        for (j=0; j<=FNAN[GA]; j++){
 
          po = 0;
          for (p=Atom_SindB[n+1][m]; p<=Atom_EindB[n+1][m]; p++){

            if (natn[GA][j]==Atom_Index_BC[n+1][p]){

              k1 = p - Atom_SindB[n+1][m] + 1; 
              po = 1; 

              break; 
            } 
	  } /* p */        

          if (po==1){

            NN_natn[i-Atom_SindB[n+1][m]+1][i0] = k1;
            NN_ncn[i-Atom_SindB[n+1][m]+1][i0] = ncn[GA][j];
            i0++;
	  }

        } /* j */

        NN_FNAN[i-Atom_SindB[n+1][m]+1] = i0 - 1;

        for (k=1; k<=3; k++){
          Cell_Gxyz2[k][i-Atom_SindB[n+1][m]+1] = Cell_Gxyz[GA][k];
	  OG2G[k][i-Atom_SindB[n+1][m]+1] = GA;
	  G2OG[k][i-Atom_SindB[n+1][m]+1] = i-Atom_SindB[n+1][m]+1;
	}       

      } /* i */

      Latomnum = Atom_EindB[n+1][m] - Atom_SindB[n+1][m] + 1;

      if (0<Latomnum){ 
        Bisection_System( Latomnum,Cell_Gxyz2,OG2G,G2OG,
                          NN_FNAN,NN_natn,NN_ncn,
                          OGmin,OGmax,&num0,&numb,&num1);
      }
      else {
        num0 = 0;
        numb = 0;
        num1 = 0;  
      }

      /* construct Atom_Index_BC for the C part */

      Atom_EindC[n][m] = Atom_EindB[n+1][m]; 
      Atom_SindC[n][m] = Atom_EindC[n][m] + 1 - numb;

      for (i=Atom_SindC[n][m]; i<=Atom_EindC[n][m]; i++){
        Atom_Index_BC[n][i] = OG2G[0][num0+i-Atom_SindC[n][m]+1];
      }

      /* construct Index_BC for the B parts */

      Atom_SindB[n][2*m] = Atom_SindB[n+1][m];
      Atom_EindB[n][2*m] = Atom_SindB[n][2*m] + num0 - 1;

      Atom_SindB[n][2*m+1] = Atom_EindB[n][2*m] + 1;
      Atom_EindB[n][2*m+1] = Atom_SindB[n][2*m+1] + num1 - 1;

      for (i=Atom_SindB[n][2*m]; i<=Atom_EindB[n][2*m]; i++){
        Atom_Index_BC[n][i] = OG2G[0][i-Atom_SindB[n][2*m]+1];
      }

      for (i=Atom_SindB[n][2*m+1]; i<=Atom_EindB[n][2*m+1]; i++){
        Atom_Index_BC[n][i] = OG2G[0][num0+numb+i-Atom_SindB[n][2*m+1]+1];
      }

    } /* m */

    mp *= 2; 

  } /* n */

  /*************************************************************
   reorder Atom_Index_BC so that the order of the global index 
   can be equivalent in all the Atom_Index_BCs 
  *************************************************************/

  mp = mpmax;

  for (n=0; n<nl; n++){

    for (m=0; m<mp; m++){
      for (i=Atom_SindB[n][m]; i<=Atom_EindB[n][m]; i++){
	Atom_Index_BC[n+1][i] = Atom_Index_BC[n][i];
      }
    }

    for (m=0; m<(mp/2); m++){
      for (i=Atom_SindC[n][m]; i<=Atom_EindC[n][m]; i++){
	Atom_Index_BC[n+1][i] = Atom_Index_BC[n][i];
      }
    }

    mp /= 2;
  }  

  /* printing Atom_Index_BC for debugging */

  /*
  printf("\n\n");
  printf("nl=%2d mpmax=%2d\n",nl,mpmax);

  mp = mpmax;
  for (n=0; n<(nl+1); n++){

    for (m=0; m<mp; m++){
      for (i=Atom_SindB[n][m]; i<=Atom_EindB[n][m]; i++){
        printf("B n=%2d m=%2d i=%2d Atom_Index_BC=%2d\n",n,m,i,Atom_Index_BC[n][i]);
      }
    }

    mp /= 2;
  }

  mp = mpmax/2;
  for (n=0; n<nl; n++){

    for (m=0; m<mp; m++){
      for (i=Atom_SindC[n][m]; i<=Atom_EindC[n][m]; i++){
        printf("C n=%2d m=%2d i=%2d Atom_Index_BC=%2d\n",n,m,i,Atom_Index_BC[n][i]);
      }
    }

    mp /= 2;
  }
  */

  /*************************************************************
             construct Index_BC using Atom_Index_BC 
  *************************************************************/

  /* set up Depth_Level and Number_Division */

  *Depth_Level = nl;
  *Number_Division = mpmax;

  /* set up MP2 where MP2[i=ordered index for atom] 
     gives an ordered index for orbital. */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    Gc_AN = Atom_Index_BC[nl][i];
    MP2[i] = Anum;
    wanA = WhatSpecies[Gc_AN];
    Anum += Spe_Total_CNO[wanA];
  }

  /* Index_BC for B */

  mp = mpmax;
  for (n=0; n<(nl+1); n++){
    for (m=0; m<mp; m++){

      i = Atom_SindB[n][m]; 
      SindB[n][m] = MP2[i]-1;

      i = Atom_EindB[n][m]; 
      Gc_AN = Atom_Index_BC[n][i];
      wanA = WhatSpecies[Gc_AN];
      EindB[n][m] = MP2[i] + Spe_Total_CNO[wanA]-2;

      for (i=Atom_SindB[n][m]; i<=Atom_EindB[n][m]; i++){

        Gc_AN = Atom_Index_BC[n][i];
        wanA = WhatSpecies[Gc_AN];
        i0 = MP2[i];

        for (j=MP[Gc_AN]; j<(MP[Gc_AN]+Spe_Total_CNO[wanA]); j++){
          Index_BC[n][i0-1] = j-1; 
          i0++;
	}
      }
    }

    mp /= 2;
  }

  /* Index_BC for C */

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){

      i = Atom_SindC[n][m]; 
      SindC[n][m] = MP2[i]-1;

      i = Atom_EindC[n][m]; 
      Gc_AN = Atom_Index_BC[n][i];
      wanA = WhatSpecies[Gc_AN];
      EindC[n][m] = MP2[i] + Spe_Total_CNO[wanA]-2;

      for (i=Atom_SindC[n][m]; i<=Atom_EindC[n][m]; i++){

        Gc_AN = Atom_Index_BC[n][i];
        wanA = WhatSpecies[Gc_AN];
        i0 = MP2[i];

        for (j=MP[Gc_AN]; j<(MP[Gc_AN]+Spe_Total_CNO[wanA]); j++){
          Index_BC[n][i0-1] = j-1; 
          i0++;
	}
      }
    }

    mp /= 2;
  }


  /* printing Index_BC for debugging */

  /*
  printf("\n\n");
  printf("nl=%2d mpmax=%2d\n",nl,mpmax);

  mp = mpmax;
  for (n=0; n<(nl+1); n++){

    for (m=0; m<mp; m++){
      for (i=SindB[n][m]; i<=EindB[n][m]; i++){
        printf("B n=%2d m=%2d i=%2d Index_BC=%2d\n",n,m,i,Index_BC[n][i]);
      }
    }

    mp /= 2;
  }

  mp = mpmax/2;
  for (n=0; n<nl; n++){

    for (m=0; m<mp; m++){
      for (i=SindC[n][m]; i<=EindC[n][m]; i++){
        printf("C n=%2d m=%2d i=%2d Index_BC=%2d\n",n,m,i,Index_BC[n][i]);
      }
    }

    mp /= 2;
  }
  */


  /* printing structures */

  /*
  {

    int n3;
    char NameC[100][100];

    n3 = nl-6;  

  sprintf(NameC[0],"He");
  sprintf(NameC[1],"Li");
  sprintf(NameC[2],"Be");
  sprintf(NameC[3],"B");
  sprintf(NameC[4],"C");
  sprintf(NameC[5],"N");
  sprintf(NameC[6],"F");
  sprintf(NameC[7],"Ne");
  sprintf(NameC[8],"Na");
  sprintf(NameC[9],"Mg");
  sprintf(NameC[10],"Al");
  sprintf(NameC[11],"Si");
  sprintf(NameC[12],"P");
  sprintf(NameC[13],"S");
  sprintf(NameC[14],"Cl");
  sprintf(NameC[15],"Ar");
  sprintf(NameC[16],"K");

  printf("%2d\n\n",atomnum);

  mp = mpmax;
  for (n=0; n<(nl+1); n++){

    for (m=0; m<mp; m++){

      if (n==n3){ 
	for (i=SindB[n][m]; i<=EindB[n][m]; i++){

          j = Index_BC[n][i] + 1;
          printf("H  %12.7f %12.7f %12.7f   0.0 0.0 0.0\n",
                  Gxyz[j][1]*BohrR,Gxyz[j][2]*BohrR,Gxyz[j][3]*BohrR);
	}
      }

    }

    mp /= 2;
  }

  mp = mpmax/2;
  for (n=0; n<nl; n++){

    for (m=0; m<mp; m++){

      if (n3<=n){ 
	for (i=SindC[n][m]; i<=EindC[n][m]; i++){

          j = Index_BC[n][i] + 1;
                
          printf("%s %12.7f %12.7f %12.7f   0.0 0.0 0.0\n",
                  NameC[n],Gxyz[j][1]*BohrR,Gxyz[j][2]*BohrR,Gxyz[j][3]*BohrR);



	}
      }

    }

    mp /= 2;
  }

  }
  MPI_Finalize();
  exit(0);
  */

  /* freeing of arrays */

  free(MP2);
  free(NN_FNAN);

  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    free(NN_natn[ct_AN]);
  }
  free(NN_natn);

  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    free(NN_ncn[ct_AN]);
  }
  free(NN_ncn);

  free(OGmax);
  free(OGmin);

  for (k=0; k<4; k++){
    free(G2OG[k]);
  }
  free(G2OG);

  for (k=0; k<4; k++){
    free(OG2G[k]);
  }
  free(OG2G);

  for (k=0; k<4; k++){
    free(Cell_Gxyz2[k]);
  }
  free(Cell_Gxyz2);

  for (n=0; n<nl; n++){
    free(Atom_EindC[n]);
  }  
  free(Atom_EindC);

  for (n=0; n<nl; n++){
    free(Atom_SindC[n]);
  }  
  free(Atom_SindC);

  for (n=0; n<(nl+1); n++){
    free(Atom_EindB[n]);
  }  
  free(Atom_EindB);

  for (n=0; n<(nl+1); n++){
    free(Atom_SindB[n]);
  }  
  free(Atom_SindB);

  for (n=0; n<(nl+1); n++){
    free(Atom_Index_BC[n]);
  }
  free(Atom_Index_BC);
}







static void Bisection_System(
    int Latomnum, 
    double **Cell_Gxyz2, 
    int **OG2G, 
    int **G2OG, 
    int *NN_FNAN,
    int **NN_natn,
    int **NN_ncn, 
    int *OGmin,
    int *OGmax, 
    int *num0, 
    int *numb,
    int *num1)
{
  int i,k,AN0,AN1,AN2,Gc_AN,h_AN,Gh_AN;
  int nmin,nmax,l,n,Rn,OGh_AN;
  int FNAN_min,OG_max,OG_min,po;
  int OG_min0,OG_max0,diff_n,nmin0,nmax0;
  int abc_n0[4],abc_nb[4],abc_n1[4];
  int abc_OG_max[4],abc_OG_min[4];
  int abc_nmax[4],abc_nmin[4],abc_mag[4];
  int n0,n1,nb,n00,nb0,n10,k0,wanA;
  int *longtail_flag;
  double rcut,max_rcut,min_rcut;

  /***************************************************************
       eliminate atoms which have longer cutoff of basis set, 
       where we let the eliminated atoms be in the C region.
  ***************************************************************/

  longtail_flag = (int*)malloc(sizeof(int)*(Latomnum+1));
  for (i=0; i<(Latomnum+1); i++) longtail_flag[i] = 0;

  max_rcut = -100000.0;
  min_rcut =  100000.0;

  for (AN0=1; AN0<=Latomnum; AN0++){

    AN1 = OG2G[1][AN0];
    wanA = WhatSpecies[AN1];
    rcut = Spe_Atom_Cut1[wanA];

    if (max_rcut<rcut) max_rcut = rcut;
    if (rcut<min_rcut) min_rcut = rcut;
  }

  if (  1.5<=(max_rcut/min_rcut) ){

    for (AN0=1; AN0<=Latomnum; AN0++){

      AN1 = OG2G[1][AN0];
      wanA = WhatSpecies[AN1];
      rcut = Spe_Atom_Cut1[wanA];

      if ( 1.5<=(rcut/min_rcut) ){
        longtail_flag[AN0] = 1;                 
      } 
    }
  }

  /***************************************************************
                      bisection of the system 
  ***************************************************************/

  for (k=1; k<=3; k++){

    /*
    for (i=1; i<=Latomnum; i++){
      printf("A1 k=%2d i=%2d Cell_Gxyz2=%15.12f OG2G=%2d G2OG=%2d\n",
              k,i,Cell_Gxyz2[k][i],OG2G[k][i],G2OG[k][i]);
    }
    */

    /* order atoms based on Cell_Gxyz2 */

    qsort_double3((long)Latomnum,Cell_Gxyz2[k],OG2G[k],G2OG[k]); 

    /*
    for (i=1; i<=Latomnum; i++){
      printf("A2 k=%2d i=%2d Cell_Gxyz2=%15.12f OG2G=%2d G2OG=%2d\n",
              k,i,Cell_Gxyz2[k][i],OG2G[k][i],G2OG[k][i]);
    }
    */

    for (i=1; i<=Latomnum; i++){
      AN0 = G2OG[k][i];
      G2OG[0][AN0] = i;
    }

    /* find the reachable range of each atom */

    for (AN0=1; AN0<=Latomnum; AN0++){

      AN1 = G2OG[k][AN0]; 

      if (longtail_flag[AN1]==0){  /* eliminate atoms with "longtail_flag[AN1]==0" */

	nmin = 1000000;
	nmax =-1000000;

	for (h_AN=0; h_AN<=NN_FNAN[AN1]; h_AN++){

	  AN2 = NN_natn[AN1][h_AN];
	  Rn = NN_ncn[AN1][h_AN];
	  l = atv_ijk[Rn][k];

	  n = (G2OG[0][AN2]-1) + Latomnum*l;

	  if (longtail_flag[AN2]==0){ /* eliminate atoms with "longtail_flag[AN2]==0" */
	    if (nmax<n) nmax = n;
	    if (n<nmin) nmin = n;
	  }
	}
      }

      else {
        nmin = AN0-1;
        nmax = AN0-1;
      }

      OGmin[AN0] = nmin; 
      OGmax[AN0] = nmax;

    } /* AN0 */

    /* find the starting atom which has the lowest NN_FNAN. */

    FNAN_min = 1000000;
    for (AN0=1; AN0<=Latomnum; AN0++){

      AN1 = G2OG[k][AN0]; 

      if (longtail_flag[AN1]==0){  /* eliminate atoms with "longtail_flag[AN1]==0" */
	if (NN_FNAN[AN1]<FNAN_min){

	  FNAN_min = NN_FNAN[AN1];
	  OG_min = AN0;
	}
      }
    }

    /********************************************************** 
          extend the size of the part n0 step by step
    **********************************************************/
    
    po = 0;
    OG_max = OG_min;
    nmin = OGmin[OG_min];
    nmax = OGmax[OG_min];
    diff_n = 1000000;
    OG_min0 = OG_min;
    OG_max0 = OG_max;
    nmin0 = nmin;
    nmax0 = nmax;

    do {

      if (OGmin[OG_max]<nmin) nmin = OGmin[OG_max];
      if (nmax<OGmax[OG_max]) nmax = OGmax[OG_max];

      if (OGmin[OG_min]<nmin) nmin = OGmin[OG_min];
      if (nmax<OGmax[OG_min]) nmax = OGmax[OG_min];
  
      n0 = OG_max - OG_min + 1;
      nb = (nmax - nmin + 1) - n0;
      n1 = Latomnum - (n0 + nb); 

      /********************************************************** 
           set up flag into OG2G[0][AN],
           where AN is the local index when the routine was called.
      **********************************************************/

      for (AN0=0; AN0<=Latomnum; AN0++) OG2G[0][AN0] = 0;

      for (AN0=OG_min; AN0<=OG_max; AN0++){ 
	AN1 = G2OG[k][AN0]; 
	OG2G[0][AN1] = 10;  /* flag in the n0 part */
      }

      for (AN0=nmin; AN0<=nmax; AN0++){

	if (0<=AN0){
	  AN1 = AN0%Latomnum + 1;    
	}
	else {

	  AN1 = AN0; 
	  do {
	    AN1 += Latomnum;
	  } while (AN1<0);
	  AN1 = AN1 + 1;
	}
         
	AN2 = G2OG[k][AN1]; 

	if (OG2G[0][AN2]==0)  OG2G[0][AN2] = 20;  /* flag in the nb part */
      }

      for (AN0=1; AN0<=Latomnum; AN0++){
	if (OG2G[0][AN0]==0) OG2G[0][AN0] = 30; /* flag in the n1 part */
      }

      /********************************************************** 
             update n0, nb, n1 using OG2G[0] and longtail_flag
      **********************************************************/

      for (AN0=1; AN0<=Latomnum; AN0++){

	if (longtail_flag[AN0]==1 && OG2G[0][AN0]==10){
	  n0--;
	  nb++;
	}          
          
	if (longtail_flag[AN0]==1 && OG2G[0][AN0]==30){
	  n1--;
	  nb++;
	}
      }         

      /********************************************************** 
              if (fabs(n1-n0)<=diff_n) then, update parameters
      **********************************************************/

      if (fabs(n1-n0)<=diff_n){

	diff_n = fabs(n1-n0);
	OG_min0 = OG_min;
	OG_max0 = OG_max;
	nmin0 = nmin;
	nmax0 = nmax;
	n00 = n0;
	nb0 = nb;
	n10 = n1;
      }
      else {
	po = 1;
      }

      if ( Latomnum<=(nmax-nmin+1) ) {
	po = 2; /* partitioning is impossible along the direction. */
      }

      /*
      printf("FFF po=%2d k=%2d OG_min=%2d OG_max=%2d nmin=%2d nmax=%2d n0=%2d nb=%2d n1=%2d\n",
              po,k,OG_min,OG_max,nmin,nmax,n0,nb,n1);
      */


      if (po==0){ 
        if (OG_max<Latomnum) OG_max++;
        else if (1<OG_min)   OG_min--;
      }

    } while (po==0);

    abc_OG_max[k] = OG_max0;
    abc_OG_min[k] = OG_min0;
    abc_nmax[k] = nmax0;
    abc_nmin[k] = nmin0;
    abc_n0[k] = n00;
    abc_nb[k] = nb0;
    abc_n1[k] = n10;

    /*
    printf("k=%2d n0=%2d nb=%2d n1=%2d\n",k,abc_n0[k],abc_nb[k],abc_n1[k]);
    */

  } /* k */

  /********************************************************** 
     set up OG2G
  **********************************************************/

  abc_mag[1] = fabs(abc_n0[1]-abc_n1[1])+abc_nb[1];
  abc_mag[2] = fabs(abc_n0[2]-abc_n1[2])+abc_nb[2];
  abc_mag[3] = fabs(abc_n0[3]-abc_n1[3])+abc_nb[3];

  /*
  printf("abc_mag[1]=%2d\n",abc_mag[1]);
  printf("abc_mag[2]=%2d\n",abc_mag[2]);
  printf("abc_mag[3]=%2d\n",abc_mag[3]);
  */

  if (abc_mag[1]<=abc_mag[2])  k0 = 1;  
  else                         k0 = 2;
  if (abc_mag[3]<abc_mag[k0])  k0 = 3;

  /*
  printf("k0=%2d abc_n0[k0]=%2d\n",k0,abc_n0[k0]);
  */

  /* The n0 part */

  n = 1;
  for (AN0=abc_OG_min[k0]; AN0<(abc_OG_min[k0]+abc_n0[k0]); AN0++){

    AN1 = G2OG[k0][AN0]; 

    if (longtail_flag[AN1]!=1){
      G2OG[0][n] = OG2G[k0][AN0];
      OG2G[k0][AN0] = -1;
      n++;
    }
  }

  /* The nb part from the n0 part */

  for (AN0=abc_OG_min[k0]; AN0<(abc_OG_min[k0]+abc_n0[k0]); AN0++){

    AN1 = G2OG[k0][AN0]; 

    if (longtail_flag[AN1]==1){
      G2OG[0][n] = OG2G[k0][AN0];
      OG2G[k0][AN0] = -1;
      n++;
    }
  }

  /* The nb part from the original nb part */

  for (AN0=abc_nmin[k0]; AN0<=abc_nmax[k0]; AN0++){

    if (0<=AN0){
      AN1 = AN0%Latomnum + 1;    
    }
    else {

      AN1 = AN0; 
      do {
        AN1 += Latomnum;
      } while (AN1<0);
      AN1 = AN1 + 1;
    }

    if (OG2G[k0][AN1]!=-1) {
      G2OG[0][n] = OG2G[k0][AN1];
      OG2G[k0][AN1] = -1;
      n++;
    }
  }

  /* The nb part from the n1 part */

  for (AN0=1; AN0<=Latomnum; AN0++){

    AN1 = G2OG[k0][AN0]; 

    if (OG2G[k0][AN0]!=-1 && longtail_flag[AN1]==1) {
      G2OG[0][n] = OG2G[k0][AN0];
      OG2G[k0][AN0] = -1;
      n++;
    }
  }
    
  /* The n1 part */

  for (AN0=1; AN0<=Latomnum; AN0++){

    AN1 = G2OG[k0][AN0]; 

    if (OG2G[k0][AN0]!=-1 && longtail_flag[AN1]!=1) {
      G2OG[0][n] = OG2G[k0][AN0];
      OG2G[k0][AN0] = -1;
      n++;
    }
  }

  if ( (n-1)!=Latomnum ){
    printf("Could not bisect\n");

    MPI_Finalize();
    exit(1);
  }

  /* set n0, nb, n1, and OG2G[0] */

  n0 = abc_n0[k0];
  nb = abc_nb[k0];
  n1 = abc_n1[k0];

  for (n=1; n<=Latomnum; n++){
    OG2G[0][n] = G2OG[0][n];
  }

  /* print OG2G for debugging */

  /*
  for (n=1; n<=n0; n++){
    printf("n0: n=%2d OG2G=%2d\n",n,OG2G[0][n]);
  }

  for (n=n0+1; n<=n0+nb; n++){
    printf("nb: n=%2d OG2G=%2d\n",n,OG2G[0][n]);
  }

  for (n=n0+nb+1; n<=n0+nb+n1; n++){
    printf("n1: n=%2d OG2G=%2d\n",n,OG2G[0][n]);
  }
  */

  *num0 = abc_n0[k0];
  *numb = abc_nb[k0];
  *num1 = abc_n1[k0];

  free(longtail_flag);
}






static void OND_Solver(
  char *mode, 
  int myid,
  int numprocs,
  int myid2,
  int numprocs2,
  int Nloop1,
  MPI_Comm MPI_Comm,
  int spin,
  int Epoint,
  double Real_Ene,
  int Nmat,
  int *NZE,
  int *PZE,
  int nl, 
  int mpmax,
  double Trial_ChemP,
  double **Hall,
  double *Sall,
  double **DMall,
  double **WRG,
  int *conv_index_row,
  int *conv_index_col,
  int **conv_ind2,
  int TNZE)
{
  int po,TNZE2,TNZE3,col,row;
  int i,j,k,j1,k1,l,p,q;
  double complex *Tx;
  int *Ti,*Tp,*Tp0;
  double tol,sum;
  double stime,etime;
  double stime1,etime1;
  double stime2,etime2;
  double complex alpha,weight;
  double complex ctmp,ctmp3;
  double av_num;
  int io,jo,jo_min,n;
  int m0,m1,m2,m3,m4,m5;
  int n0,n1,n2,mp0,nsr;
  int nb0,nb1,nc,k2,kk;
  int mp,mm,numa;
  int m,i1,i2,j2,i3,j3,nsa,nsc,nsc2,nsb;
  int max_nsc,max_nsb,max_nsa;
  dcomplex dcsum0,dcsum1,dcsum2,dcsum3,dcsum4;
  double complex csum2,csum3,csum4;
  double time1a,time1a2,time1a1,time1a0,time1b,time1c,time1d;
  double time1,time2,time3,time4,time5,time6,time41,time42;
  double time31,time32,time33,time34;
  double time51,time52,time53,time54,time55;
  double numop1,numop2,numop0;
  dcomplex al,be;
  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;
  /* for MPI */
  int ID,*is1,*ie1,*Row2ID,tag=999;
  int size0,size1,k0;
  MPI_Status *stat_send;
  MPI_Request *request_send;
  MPI_Request *request_recv;

  stat_send = malloc(sizeof(MPI_Status)*numprocs2);
  request_send = malloc(sizeof(MPI_Request)*numprocs2);
  request_recv = malloc(sizeof(MPI_Request)*numprocs2);

  al.r = 1.0;
  al.i = 0.0;
  be.r = 0.0;
  be.i = 0.0;

  numop1 = 0.0;
  numop2 = 0.0;

  time1 = 0.0;
  time1a= 0.0;
  time1a0= 0.0;
  time1a1= 0.0;
  time1a2= 0.0;
  time1b= 0.0;
  time1c= 0.0;
  time1d= 0.0;
  time2 = 0.0;
  time3 = 0.0;
  time31= 0.0;
  time32= 0.0;
  time33= 0.0;
  time34= 0.0;
  time4 = 0.0;
  time41= 0.0;
  time42= 0.0;
  time5 = 0.0;
  time51= 0.0;
  time52= 0.0;
  time53= 0.0;
  time54= 0.0;
  time55= 0.0;
  time6 = 0.0;

  dtime(&stime);
  dtime(&stime1);

  /* set alpha and weight */

  if (strcasecmp(mode,"contour")==0){  

    if (ON2_method[Epoint]==1){ /* poles */
      alpha  = Trial_ChemP + I*(ON2_zp[Epoint].i/Beta);
      weight = -2.0*ON2_Rp[Epoint].r/Beta;
    }
    else if (ON2_method[Epoint]==2){ /* zeroth moment */
      alpha  = ON2_zp[Epoint].r + I*ON2_zp[Epoint].i;
      weight = I*ON2_Rp[Epoint].i;
    }
  }

  else if (strcasecmp(mode,"retarded")==0){  
    alpha  = Real_Ene + 0.01/eV2Hartree*I;
    weight = I;
  }

  else if (strcasecmp(mode,"force")==0){  

    if (ON2_method_f[Epoint]==1){ /* poles */
      alpha  = Trial_ChemP + I*(ON2_zp_f[Epoint].i/Beta);
      weight = -2.0*ON2_Rp_f[Epoint].r/Beta*alpha;
    }
    else if (ON2_method_f[Epoint]==2){ /* zeroth and 1st order moments */
      alpha  = ON2_zp_f[Epoint].r + I*ON2_zp_f[Epoint].i;
      weight = ON2_Rp_f[Epoint].r + I*ON2_Rp_f[Epoint].i;
    }
  }

  /***********************************************************
      construct the A-matrix of which form is defined in 
      CSparse as "compressed-column form".
  ***********************************************************/

  Tp  = (int*)malloc(sizeof(int)*(Nmat+1));
  Tp0 = (int*)malloc(sizeof(int)*(Nmat+1));

  if ( 1<measure_time ){
    printf("myid=%2d Epoint =%2d %10.5f %10.5f\n",myid,Epoint,Trial_ChemP,cimag(alpha) );fflush(stdout);
  }

  dtime(&stime2);

  for (i=0; i<TNZE; i++){
    row = conv_index_row[i];                /* row index    */
    col = conv_index_col[i];                /* column index */
    conv_ind2[row][col] = 1;
  }

  dtime(&etime2);
  time1a0 = etime2 - stime2;
  
  dtime(&stime2);
  
  for (j=0; j<Nmat; j++) Tp0[j] = 0;   

  for (i=0; i<TNZE; i++){
  
    row = conv_index_row[i];                /* row index    */
    col = conv_index_col[i];                /* column index */

    if (conv_ind2[row][col]==1){
      conv_ind2[row][col] = -(Tp0[col]+1);      
      Tp0[col]++;
    }
  }

  Tp[0] = 0;
  for (j=1; j<=Nmat; j++){    /* j: col */
    Tp[j] = Tp[j-1] + Tp0[j-1];    
  }
  TNZE2 = Tp[Nmat];
  
  for (i=0; i<TNZE; i++){
  
    row = conv_index_row[i];                /* row index    */
    col = conv_index_col[i];                /* column index */
  
    if (conv_ind2[row][col]<0){
      conv_ind2[row][col] = -conv_ind2[row][col] - 1 + Tp[col];  
    }
  }

  Tx = (double complex*)malloc(sizeof(double complex)*TNZE2);
  Ti = (int*)malloc(sizeof(int)*TNZE2);
  for (i=0; i<TNZE2; i++) Tx[i] = 0.0;

  dtime(&etime2);
  time1a1 = etime2 - stime2;

  dtime(&stime2);

  for (i=0; i<TNZE; i++){

    row = conv_index_row[i];                /* row index    */
    col = conv_index_col[i];                /* column index */

    k = conv_ind2[row][col]; 

    Ti[k] = row;                                 /* row index    */    
    Tx[k] += (alpha*Sall[i] - Hall[spin][i]);    /* value        */    
  }

  dtime(&etime2);
  time1a2 = etime2 - stime2;

  /* make invp */

  invp = (int*)malloc(sizeof(int)*Nmat);

  for (i=0; i<Nmat; i++){
    j = Index_BC[nl][i];
    invp[j] = i;
  }

  dtime(&etime1);
  time1a = etime1 - stime1;

  /***********************************************************
             set up block matrices, B0, B1, and C
  ***********************************************************/

  dtime(&stime1);

  Bnum[0] = (int***)malloc(sizeof(int**)*nl);
  Bnum[1] = (int***)malloc(sizeof(int**)*nl);
  Cnum  = (int***)malloc(sizeof(int**)*nl);
  Bi[0] = (int****)malloc(sizeof(int***)*nl);
  Bi[1] = (int****)malloc(sizeof(int***)*nl);
  Ci  = (int****)malloc(sizeof(int***)*nl);

  Bx[0] = (double complex****)malloc(sizeof(double complex***)*nl);
  Bx[1] = (double complex****)malloc(sizeof(double complex***)*nl);
  Cx  = (double complex****)malloc(sizeof(double complex***)*nl);

  IBx[0] = (double complex****)malloc(sizeof(double complex***)*nl);
  IBx[1] = (double complex****)malloc(sizeof(double complex***)*nl);
  ICx  = (double complex****)malloc(sizeof(double complex***)*nl);

  mp = mpmax/2;

  for (n=0; n<nl; n++){

    Bnum[0][n] = (int**)malloc(sizeof(int*)*mp);
    Bnum[1][n] = (int**)malloc(sizeof(int*)*mp);
    Cnum[n]  = (int**)malloc(sizeof(int*)*mp);

    Bi[0][n] = (int***)malloc(sizeof(int**)*mp);
    Bi[1][n] = (int***)malloc(sizeof(int**)*mp);
    Ci[n]  = (int***)malloc(sizeof(int**)*mp);

    Bx[0][n] = (double complex***)malloc(sizeof(double complex**)*mp);
    Bx[1][n] = (double complex***)malloc(sizeof(double complex**)*mp);
    Cx[n]  = (double complex***)malloc(sizeof(double complex**)*mp);

    IBx[0][n] = (double complex***)malloc(sizeof(double complex**)*mp);
    IBx[1][n] = (double complex***)malloc(sizeof(double complex**)*mp);
    ICx[n]  = (double complex***)malloc(sizeof(double complex**)*mp);

    for (m=0; m<mp; m++){

      Bnum[0][n][m] = (int*)malloc(sizeof(int)*(EindC[n][m]-SindC[n][m]+1));
      Bnum[1][n][m] = (int*)malloc(sizeof(int)*(EindC[n][m]-SindC[n][m]+1));
      Cnum[n][m]  = (int*)malloc(sizeof(int)*(EindC[n][m]-SindC[n][m]+1));

      Bi[0][n][m] = (int**)malloc(sizeof(int*)*(EindC[n][m]-SindC[n][m]+1));
      Bi[1][n][m] = (int**)malloc(sizeof(int*)*(EindC[n][m]-SindC[n][m]+1));
      Ci[n][m]  = (int**)malloc(sizeof(int*)*(EindC[n][m]-SindC[n][m]+1));

      Bx[0][n][m] = (double complex**)malloc(sizeof(double complex*)*(EindC[n][m]-SindC[n][m]+1));
      Bx[1][n][m] = (double complex**)malloc(sizeof(double complex*)*(EindC[n][m]-SindC[n][m]+1));
      Cx[n][m]  = (double complex**)malloc(sizeof(double complex*)*(EindC[n][m]-SindC[n][m]+1));

      IBx[0][n][m] = (double complex**)malloc(sizeof(double complex*)*(EindC[n][m]-SindC[n][m]+1));
      IBx[1][n][m] = (double complex**)malloc(sizeof(double complex*)*(EindC[n][m]-SindC[n][m]+1));
      ICx[n][m]  = (double complex**)malloc(sizeof(double complex*)*(EindC[n][m]-SindC[n][m]+1));

      for (i=SindC[n][m]; i<=EindC[n][m]; i++){

        i1 = Index_BC[n][i];      

        nb0 = 0;
        nb1 = 0;
        nc  = 0;

        for (j=Tp[i1]; j<Tp[i1+1]; j++){

          j1 = invp[Ti[j]];

          if      ( SindB[n][2*m  ]<=j1 && j1<=EindB[n][2*m  ] ) nb0++;
          else if ( SindB[n][2*m+1]<=j1 && j1<=EindB[n][2*m+1] ) nb1++;
          else if ( SindC[n][m    ]<=j1 && j1<=EindC[n][m    ] ) nc++;
        }

        Bnum[0][n][m][i-SindC[n][m]] = nb0;
        Bnum[1][n][m][i-SindC[n][m]] = nb1;
        Cnum[n][m][i-SindC[n][m]] = nc;

        Bi[0][n][m][i-SindC[n][m]] = (int*)malloc(sizeof(int)*nb0);
        Bi[1][n][m][i-SindC[n][m]] = (int*)malloc(sizeof(int)*nb1);
        Ci[n][m][i-SindC[n][m]] = (int*)malloc(sizeof(int)*nc);

        Bx[0][n][m][i-SindC[n][m]] = (double complex*)malloc(sizeof(double complex)*nb0);
        Bx[1][n][m][i-SindC[n][m]] = (double complex*)malloc(sizeof(double complex)*nb1);
        Cx[n][m][i-SindC[n][m]] = (double complex*)malloc(sizeof(double complex)*nc);

        IBx[0][n][m][i-SindC[n][m]] = (double complex*)malloc(sizeof(double complex)*nb0);
        IBx[1][n][m][i-SindC[n][m]] = (double complex*)malloc(sizeof(double complex)*nb1);
        ICx[n][m][i-SindC[n][m]] = (double complex*)malloc(sizeof(double complex)*nc);

        for (k=0; k<nb0; k++) IBx[0][n][m][i-SindC[n][m]][k] = 0.0;
        for (k=0; k<nb1; k++) IBx[1][n][m][i-SindC[n][m]][k] = 0.0;
        for (k=0; k<nc; k++)  ICx[n][m][i-SindC[n][m]][k] = 0.0;

        nb0 = 0;
        nb1 = 0;
        nc  = 0;

        for (j=Tp[i1]; j<Tp[i1+1]; j++){

          j1 = invp[Ti[j]];
          i2 = i - SindC[n][m];           

          if      ( SindB[n][2*m]<=j1 && j1<=EindB[n][2*m] ){

            Bi[0][n][m][i2][nb0] = j1 - SindB[n][2*m];
            Bx[0][n][m][i2][nb0] = Tx[j];

            nb0++;
	  }

          else if ( SindB[n][2*m+1]<=j1 && j1<=EindB[n][2*m+1] ){

            Bi[1][n][m][i2][nb1] = j1 - SindB[n][2*m+1];
            Bx[1][n][m][i2][nb1] = Tx[j];

            nb1++;
	  }
          else if ( SindC[n][m]<=j1 && j1<=EindC[n][m] ){

            Ci[n][m][i2][nc] = j1 - SindC[n][m];
            Cx[n][m][i2][nc] = Tx[j];

            nc++;
	  }

        } /* j */
      } /* i */
    } /* m */

    mp /= 2;

  } /* n */

  dtime(&etime1);
  time1b = etime1 - stime1;

  /***********************************************************
             calculate A^-1 for the smallest blocks
  ***********************************************************/

  Anum = (int**)malloc(sizeof(int*)*mpmax);
  Ai = (int***)malloc(sizeof(int**)*mpmax);
  Ax = (double complex***)malloc(sizeof(double complex**)*mpmax);
  IAx = (double complex***)malloc(sizeof(double complex**)*mpmax);

  for (m=0; m<mpmax; m++){

    nsa = EindB[0][m] - SindB[0][m] + 1;

    Anum[m] = (int*)malloc(sizeof(int)*nsa);
    Ai[m] = (int**)malloc(sizeof(int*)*nsa);
    Ax[m] = (double complex**)malloc(sizeof(double complex*)*nsa);
    IAx[m] = (double complex**)malloc(sizeof(double complex*)*nsa);

    for (i=SindB[0][m]; i<=EindB[0][m]; i++){

      i1 = Index_BC[0][i];
      i2 = i - SindB[0][m];
      numa = 0;

      for (j=Tp[i1]; j<Tp[i1+1]; j++){
        j1 = invp[Ti[j]];
	if ( SindB[0][m]<=j1 && j1<=EindB[0][m] ) numa++;
      }

      Anum[m][i2] = numa;

      Ai[m][i2] = (int*)malloc(sizeof(int)*numa);
      Ax[m][i2] = (double complex*)malloc(sizeof(double complex)*numa);
      IAx[m][i2] = (double complex*)malloc(sizeof(double complex)*numa);
      for (k=0; k<numa; k++) IAx[m][i2][k] = 0.0;

      numa = 0;

      for (j=Tp[i1]; j<Tp[i1+1]; j++){

        j1 = invp[Ti[j]];

	if ( SindB[0][m]<=j1 && j1<=EindB[0][m] ){

          Ai[m][i2][numa] = j1 - SindB[0][m];
          Ax[m][i2][numa] = Tx[j];

          numa++; 
	}
      }
    }
  }

  /* calculate IA */

  mp = mpmax;
  n = 0;

  max_nsa = 0;
  for (m=0; m<mp; m++){
    nsa = EindB[n][m] - SindB[n][m] + 1;
    if (max_nsa<nsa) max_nsa = nsa; 
  }

  IA0 = (dcomplex*)malloc(sizeof(dcomplex)*max_nsa*max_nsa);

  IA = (double**)malloc(sizeof(double*)*mp);
  for (m=0; m<mp; m++){
    nsa = EindB[n][m] - SindB[n][m] + 1;
    IA[m] = (double*)malloc(sizeof(double)*2*nsa*nsa);
    for (i=0; i<2*nsa*nsa; i++) IA[m][i] = 0.0;
  }

  mp = mpmax;
  n = 0;

  for ( m=myid2; m<mp; m+=numprocs2 ){

    nsa = EindB[n][m] - SindB[n][m] + 1;

    for (i=0; i<nsa; i++){
      for (j=0; j<nsa; j++){
        IA0[nsa*j+i].r = 0.0;
        IA0[nsa*j+i].i = 0.0;
      }
    }

    for (i=0; i<nsa; i++){
      for (j=0; j<Anum[m][i]; j++){
        j2 = Ai[m][i][j];
        IA0[nsa*j2+i].r = creal(Ax[m][i][j]); 
        IA0[nsa*j2+i].i = cimag(Ax[m][i][j]); 
      }
    }

    dtime(&stime1);

    if (0<nsa) Lapack_LU_Zinverse(nsa,IA0);

    dtime(&etime1);
    time1c += etime1 - stime1;

    for (i=0; i<nsa; i++){
      for (j=0; j<nsa; j++){
        IA[m][2*nsa*j+2*i  ] = IA0[nsa*j+i].r;
        IA[m][2*nsa*j+2*i+1] = IA0[nsa*j+i].i;
      }   
    }   

    /* store IA into IAx */

    for (i=0; i<(EindB[n][m]-SindB[n][m]+1); i++){
      for (j=0; j<Anum[m][i]; j++){
        j2 = Ai[m][i][j];
        IAx[m][i][j] = IA0[nsa*j2+i].r + I*IA0[nsa*j2+i].i;
      }
    }

  } /* m */


  dtime(&stime1);

  /* MPI: IA */

  if (1<numprocs2){
    for ( m=0; m<mp; m++ ){
      nsa = EindB[n][m] - SindB[n][m] + 1;
      ID = m % numprocs2;
      MPI_Bcast(&IA[m][0], nsa*nsa*2, MPI_DOUBLE, ID, MPI_Comm);
    }
  }

  /***********************************************************
       allocation of IS, VvecTmp, Vvec, and Lvec
  ***********************************************************/

  mp = mpmax/2;
  IS = (dcomplex***)malloc(sizeof(dcomplex**)*nl);
  for (n=0; n<nl; n++){
    IS[n] = (dcomplex**)malloc(sizeof(dcomplex*)*mp);
    for (m=0; m<mp; m++){
      nsa = EindC[n][m] - SindC[n][m] + 1;

      IS[n][m] = (dcomplex*)malloc(sizeof(dcomplex)*nsa*nsa);
      for (i=0; i<nsa*nsa; i++) IS[n][m][i] = Complex(0.0,0.0);
    }
    mp /= 2;
  }  

  mp = mpmax/2;
  max_nsc = 0;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      nsc = EindC[n][m] - SindC[n][m] + 1;
      if (max_nsc<nsc) max_nsc = nsc;
    }
    mp /= 2;
  }

  mp = mpmax/2;
  max_nsb = 0;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      nsb = EindB[n][2*m]-SindB[n][2*m] + 1;
      if (max_nsb<nsb) max_nsb = nsb;

      nsb = EindB[n][2*m+1]-SindB[n][2*m+1] + 1;
      if (max_nsb<nsb) max_nsb = nsb;
    }
    mp /= 2;
  }

  ISTmp = (double*)malloc(sizeof(double)*max_nsc*max_nsc*2);
  VvecTmp = (double*)malloc(sizeof(double)*max_nsc*Nmat*2);

  Vvec = (double complex**)malloc(sizeof(double complex*)*max_nsc);
  for (i=0; i<max_nsc; i++){
    Vvec[i] = (double complex*)malloc(sizeof(double complex)*Nmat);
  }

  Vvec2 = (double complex**)malloc(sizeof(double complex*)*Nmat);
  for (i=0; i<Nmat; i++){
    Vvec2[i] = (double complex*)malloc(sizeof(double complex)*max_nsc);
  }

  mp = mpmax/2;
  Lvec[0] = (dcomplex***)malloc(sizeof(dcomplex**)*nl);
  for (n=0; n<nl; n++){
    Lvec[0][n] = (dcomplex**)malloc(sizeof(dcomplex*)*mp);
    for (m=0; m<mp; m++){
      kk = (EindB[n][2*m]-SindB[n][2*m]+1)*(EindC[n][m]-SindC[n][m]+1);
      Lvec[0][n][m] = (dcomplex*)malloc(sizeof(dcomplex)*kk);
    }
    mp /= 2;
  }

  mp = mpmax/2;
  Lvec[1] = (dcomplex***)malloc(sizeof(dcomplex**)*nl);
  for (n=0; n<nl; n++){
    Lvec[1][n] = (dcomplex**)malloc(sizeof(dcomplex*)*mp);
    for (m=0; m<mp; m++){
      kk = (EindB[n][2*m+1]-SindB[n][2*m+1]+1)*(EindC[n][m]-SindC[n][m]+1);
      Lvec[1][n][m] = (dcomplex*)malloc(sizeof(dcomplex)*kk);
    }
    mp /= 2;
  }

  Q0 = (double complex**)malloc(sizeof(double complex*)*max_nsc);
  for (i=0; i<max_nsc; i++){
    Q0[i] = (double complex*)malloc(sizeof(double complex)*max_nsc);
  }

  Q1 = (double complex**)malloc(sizeof(double complex*)*max_nsc);
  for (i=0; i<max_nsc; i++){
    Q1[i] = (double complex*)malloc(sizeof(double complex)*max_nsc);
  }

  is1 = (int*)malloc(sizeof(int)*max_nsc);
  ie1 = (int*)malloc(sizeof(int)*max_nsc);

  Row2ID = (int*)malloc(sizeof(int)*Nmat);

  dtime(&etime1);
  time1d = etime1 - stime1;

  dtime(&etime);

  time1 = etime - stime;

  /***********************************************************
         main calculations by the recurrence relations
  ***********************************************************/
 
  mp = mpmax/2;
  m2 = 1;

  for (n1=0; n1<nl; n1++){

    /*******************************
        calculate the initial V0 
    *******************************/

    dtime(&stime);

    for (m=0; m<mpmax/2; m++){
      for (m0=0; m0<2; m0++){

	nsa = EindB[0][2*m+m0] - SindB[0][2*m+m0] + 1;

	m3 = (2*m+m0)/(m2*2);
	m4 = ((2*m+m0)/m2)%2;

        nsc = EindC[n1][m3] - SindC[n1][m3] + 1;

	/***********************************
        division of nsc for 
        MPI parallelization in MPI_Comm
	***********************************/

	if ( numprocs2<=nsc ){

	  av_num = (double)nsc/(double)numprocs2;

	  for (ID=0; ID<numprocs2; ID++){
	    is1[ID] = (int)( (av_num+1.0e-12)*(double)ID);
	    ie1[ID] = (int)( (av_num+1.0e-12)*(double)(ID+1)) - 1; 
	  }

	  is1[0] = 0;
	  ie1[numprocs2-1] = nsc-1; 
	}

	else{

	  for (ID=0; ID<nsc; ID++){
	    is1[ID] = ID; 
	    ie1[ID] = ID;
	  }
	  for (ID=nsc; ID<numprocs2; ID++){
	    is1[ID] =  0;
	    ie1[ID] = -1;
	  }
	}

#pragma omp parallel shared(myid2,is1,ie1,nsc,nsa,m4,n1,m3,Bnum,Bi,SindB,EindB,m,m0,IA,Bx,Vvec) private(OMPID,Nthrds,Nprocs,j,i,k,k1,k2)
	{

	  double complex ctmp1,csum1;

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for ( j=is1[myid2]+OMPID; j<=ie1[myid2]; j+=Nthrds ){

	    for (i=0; i<nsa; i++){

	      csum1 = 0.0;
            
	      for (k=0; k<Bnum[m4][n1][m3][j]; k++){

		k1 = Bi[m4][n1][m3][j][k] + SindB[n1][2*m3+m4];
		k2 = k1 - SindB[0][2*m+m0];

		if (SindB[0][2*m+m0]<=k1 && k1<=EindB[0][2*m+m0]){

		  ctmp1 = IA[2*m+m0][2*nsa*k2+2*i] + I*IA[2*m+m0][2*nsa*k2+2*i+1];
		  csum1 += ctmp1*Bx[m4][n1][m3][j][k];
		}
	      }

	      Vvec[j][SindB[0][2*m+m0]+i] = csum1;

	    }
	  } /* j */

	} /* #pragma omp parallel */

      } /* m0 */
    } /* m */

    dtime(&etime);
    time2 += etime - stime;

    /*******************************
          recurrence relations
    *******************************/

    for (m=0; m<mp; m++){

      dtime(&stime);
      dtime(&stime1);

      nsc = EindC[n1][m] - SindC[n1][m] + 1;
      mp0 = m2; 

      /***********************************
        division of nsc for 
        MPI parallelization in MPI_Comm
      ***********************************/

      if ( numprocs2<=nsc ){

	av_num = (double)nsc/(double)numprocs2;

	for (ID=0; ID<numprocs2; ID++){
	  is1[ID] = (int)( (av_num+1.0e-12)*(double)ID);
	  ie1[ID] = (int)( (av_num+1.0e-12)*(double)(ID+1)) - 1; 
	}

	is1[0] = 0;
	ie1[numprocs2-1] = nsc-1; 
      }

      else{

	for (ID=0; ID<nsc; ID++){
	  is1[ID] = ID; 
	  ie1[ID] = ID;
	}
	for (ID=nsc; ID<numprocs2; ID++){
	  is1[ID] =  0;
	  ie1[ID] = -1;
	}
      }

      dtime(&etime1);
      time31 += etime1 - stime1;

      dtime(&stime2);

      /***********************************
       loop for n2: which means 
       climbing the ladder at the level n1
      ***********************************/

      for (n2=0; n2<n1; n2++){
        for (m1=0; m1<mp0; m1++){

          mm = mp0*m + m1;

          dtime(&stime1);

          nsr = EindC[n2][mm] - SindC[n2][mm] + 1;

          /***********************************
                     calculation of Q 
          ***********************************/

#pragma omp parallel shared(is1,ie1,myid2,nsr,Bnum,n1,n2,m1,mm,SindB,Bi,Bx,Vvec,IS,Q1,Q0,SindC,numop1) private(OMPID,Nthrds,Nprocs,j,i,k,k1,m5,l,m3,m4,numop0)

          { 

	    double complex csum1;

            numop0 = 0.0;

    	    /* get info. on OpenMP */ 

  	    OMPID = omp_get_thread_num();
	    Nthrds = omp_get_num_threads();
	    Nprocs = omp_get_num_procs();

	    for ( j=is1[myid2]+OMPID; j<=ie1[myid2]; j+=Nthrds ){

	      for (i=0; i<nsr; i++){

		csum1 = 0.0;

		/* B_{even} V_{even} */

		for (k=0; k<Bnum[0][n2][mm][i]; k++){
		  k1 = SindB[n2][2*mm] + Bi[0][n2][mm][i][k];   
		  csum1 += Bx[0][n2][mm][i][k]*Vvec[j][k1]; 
		}

		numop0 += (double)Bnum[0][n2][mm][i]; 

		/* B_{odd} V_{odd} */

		for (k=0; k<Bnum[1][n2][mm][i]; k++){
		  k1 = SindB[n2][2*mm+1] + Bi[1][n2][mm][i][k];
		  csum1 += Bx[1][n2][mm][i][k]*Vvec[j][k1]; 
		}

		numop0 += (double)Bnum[1][n2][mm][i]; 

		/* B[C] */

		m5 = 1;
		for (l=0; l<(n1-n2); l++) m5 *= 2;

		m3 = mm/m5;
		m4 = (mm/(m5/2))%2;

		for (k=0; k<Bnum[m4][n1][m3][j]; k++){
		  k1 = Bi[m4][n1][m3][j][k] + SindB[n1][2*m3+m4] - SindC[n2][mm];
		  if (k1==i) csum1 -= Bx[m4][n1][m3][j][k]; 
		}

		/* store the temporal result into Q0 */

		Q0[j][i] = csum1;

	      } /* i */

	      /* calculate Q */

	      for (i=0; i<nsr; i++){

		csum1 = 0.0;
		for (k=0; k<nsr; k++){
		  csum1 += (IS[n2][mm][i*nsr+k].r+I*IS[n2][mm][i*nsr+k].i)*Q0[j][k];  
		}           

		Q1[j][i] = csum1;
	      }

	    } /* j */

            numop1 += numop0;

	  } /* #pragma omp parallel */

          dtime(&etime1);
          time32 += etime1 - stime1;

          /**********************************
                        update V 
          **********************************/

          dtime(&stime1);

#pragma omp parallel shared(is1,ie1,myid2,EindB,SindB,n2,mm,nsr,Lvec,Q1,Vvec,SindC,numop1) \
 private(OMPID,Nthrds,Nprocs,j,i,i2,k,i1,numop0)

          { 

	    double complex csum0;

            numop0 = 0.0;

    	    /* get info. on OpenMP */ 

  	    OMPID = omp_get_thread_num();
	    Nthrds = omp_get_num_threads();
	    Nprocs = omp_get_num_procs();

	    for ( j=is1[myid2]+OMPID; j<=ie1[myid2]; j+=Nthrds ){

  	      /* V_{even} */

	      for (i=0; i<(EindB[n2][2*mm]-SindB[n2][2*mm]+1); i++){

		i2 = i*nsr; 
		csum0 = 0.0;

		for (k=0; k<nsr; k++){
		  csum0 += (Lvec[0][n2][mm][i2+k].r+I*Lvec[0][n2][mm][i2+k].i)*Q1[j][k];
		}

		Vvec[j][SindB[n2][2*mm]+i] += csum0;
	      }

              numop0 += (double)(EindB[n2][2*mm]-SindB[n2][2*mm]+1)*nsr;

	      /* V_{odd} */

	      for (i=0; i<(EindB[n2][2*mm+1]-SindB[n2][2*mm+1]+1); i++){

		i2 = i*nsr; 
		csum0 = 0.0;

		for (k=0; k<nsr; k++){
		  csum0 += (Lvec[1][n2][mm][i2+k].r+I*Lvec[1][n2][mm][i2+k].i)*Q1[j][k];
		}

		Vvec[j][SindB[n2][2*mm+1]+i] += csum0;
	      }

              numop0 += (double)(EindB[n2][2*mm+1]-SindB[n2][2*mm+1]+1)*nsr;

	      /* add the contribution of -Q */

	      for (i=0; i<nsr; i++){
		i1 = SindC[n2][mm] + i;
		Vvec[j][i1] = -Q1[j][i];
	      }

	    } /* j */   

            numop1 += numop0;

	  } /* #pragma omp parallel */

          dtime(&etime1);
          time33 += etime1 - stime1;

	} /* m1 */

        /* update mp0 */

        mp0 /= 2;

      } /* n2 */

      dtime(&etime2);
      time34 += etime2 - stime2;

      dtime(&etime);
      time3 += etime - stime;

      /*******************************
           calculate the S-matrix
      *******************************/

      dtime(&stime);

      /* initialize IS */

      for (j=0; j<nsc; j++){
        for (i=0; i<nsc; i++){

	  ISTmp[2*j*nsc+2*i  ] = 0.0;
	  ISTmp[2*j*nsc+2*i+1] = 0.0;
	}
      }

      /* put C to IS */

      for (i=0; i<nsc; i++){
	for (j=0; j<Cnum[n1][m][i]; j++){

	  j1 = Ci[n1][m][i][j];

          if (is1[myid2]<=j1 && j1<=ie1[myid2]){ 

  	    ISTmp[2*j1*nsc+2*i  ] = creal(Cx[n1][m][i][j]);
	    ISTmp[2*j1*nsc+2*i+1] = cimag(Cx[n1][m][i][j]);
	  }
	}
      }

      /* calculate the inner products */

      dtime(&stime1);

#pragma omp parallel shared(is1,ie1,myid2,nsc,Bnum,n1,m,SindB,Bi,Bx,Vvec,ISTmp) private(OMPID,Nthrds,Nprocs,j,i,k,k1)

      { 

	double complex csum1;

 
	/* get info. on OpenMP */ 

	OMPID = omp_get_thread_num();
	Nthrds = omp_get_num_threads();
	Nprocs = omp_get_num_procs();

	for ( j=is1[myid2]+OMPID; j<=ie1[myid2]; j+=Nthrds ){

	  for (i=0; i<nsc; i++){

	    csum1 = 0.0;

	    /* contribution of even m */

	    for (k=0; k<Bnum[0][n1][m][i]; k++){

	      k1 = SindB[n1][2*m] + Bi[0][n1][m][i][k];
	      csum1 += Bx[0][n1][m][i][k]*Vvec[j][k1];
	    }

	    /* contribution of odd m */

	    for (k=0; k<Bnum[1][n1][m][i]; k++){

	      k1 = SindB[n1][2*m+1] + Bi[1][n1][m][i][k];
	      csum1 += Bx[1][n1][m][i][k]*Vvec[j][k1];
	    }

	    ISTmp[2*j*nsc+2*i  ] -= creal(csum1);
	    ISTmp[2*j*nsc+2*i+1] -= cimag(csum1);
	  }
	}

      } /* #pragma omp parallel */

      /* MPI: IS */

      if (1<numprocs2){

	/* sending */

	size0 = (ie1[myid2]-is1[myid2]+1)*nsc*2;       
	if (size0<0) size0 = 0; 
	k0 = 2*is1[myid2]*nsc;
	if (k0<0) k0 = 0;

	for (ID=0; ID<numprocs2; ID++){
	  MPI_Isend(&ISTmp[k0], size0, MPI_DOUBLE, ID, tag, MPI_Comm, &request_send[ID]);
	}

	/* receiving */

	for (ID=0; ID<numprocs2; ID++){

	  size1 = (ie1[ID]-is1[ID]+1)*nsc*2;       
	  if (size1<0) size1 = 0; 
	  k1 = 2*is1[ID]*nsc;
	  if (k1<0) k1 = 0;

	  MPI_Irecv(&ISTmp[k1], size1, MPI_DOUBLE, ID, tag, MPI_Comm, &request_recv[ID]);
	}

        /* Waitall */

	MPI_Waitall(numprocs2,request_recv,stat_send);
	MPI_Waitall(numprocs2,request_send,stat_send);
      }      

      /* set up IS */

      for (j=0; j<nsc; j++){
        for (i=0; i<nsc; i++){
	  IS[n1][m][j*nsc+i].r = ISTmp[2*j*nsc+2*i  ];
	  IS[n1][m][j*nsc+i].i = ISTmp[2*j*nsc+2*i+1];
	}
      }
 
      dtime(&etime1);
      time41 += etime1 - stime1;

      /*************************************f******************
       MPI communication of Vvec using VvecTmp:
       In order to hide the communication, the MPI_Waitall are 
       called after the calculation of the inverse of S 
      ********************************************************/

      for (j=is1[myid2]; j<=ie1[myid2]; j++){
        for (i=0; i<Nmat; i++){
          VvecTmp[2*j*Nmat+2*i  ] = creal(Vvec[j][i]);
          VvecTmp[2*j*Nmat+2*i+1] = cimag(Vvec[j][i]);
	}
      }

      if (1<numprocs2){

	/* sending */

	size0 = (ie1[myid2]-is1[myid2]+1)*Nmat*2;       
	if (size0<0) size0 = 0; 
	k0 = 2*is1[myid2]*Nmat;
	if (k0<0) k0 = 0;
    
	for (ID=0; ID<numprocs2; ID++){
	  MPI_Isend(&VvecTmp[k0], size0, MPI_DOUBLE, ID, tag, MPI_Comm, &request_send[ID]);
	}

	/* receiving */

	for (ID=0; ID<numprocs2; ID++){

	  size1 = (ie1[ID]-is1[ID]+1)*Nmat*2;       
	  if (size1<0) size1 = 0; 
	  k1 = 2*is1[ID]*Nmat;
	  if (k1<0) k1 = 0;

	  MPI_Irecv(&VvecTmp[k1], size1, MPI_DOUBLE, ID, tag, MPI_Comm, &request_recv[ID]);
	}
      }

      /*******************************
        call Lapack_LU_Zinverse
        to calculate the inverse of S
      *******************************/

      dtime(&stime1);

      if (0<nsc) Lapack_LU_Zinverse(nsc,IS[n1][m]);

      dtime(&etime1);
      time42 += etime1 - stime1;

      dtime(&etime);
      time4 += etime - stime;

      /****************************************************
       To end MPI communication of VvecTmp, call waitall, 
      ****************************************************/
       
      if (1<numprocs2){
	MPI_Waitall(numprocs2,request_recv,stat_send);
	MPI_Waitall(numprocs2,request_send,stat_send);
      }

      /*******************************
            store Vvec as Lvec
      *******************************/

      for (j=0; j<nsc; j++){
        for (i=SindB[n1][2*m]; i<=EindB[n1][2*m]; i++){
	  Lvec[0][n1][m][ (i-SindB[n1][2*m])*nsc+j ].r = VvecTmp[2*j*Nmat+2*i  ]; 
	  Lvec[0][n1][m][ (i-SindB[n1][2*m])*nsc+j ].i = VvecTmp[2*j*Nmat+2*i+1]; 
	}        
      }    

      for (j=0; j<nsc; j++){
        for (i=SindB[n1][2*m+1]; i<=EindB[n1][2*m+1]; i++){
	  Lvec[1][n1][m][ (i-SindB[n1][2*m+1])*nsc+j ].r = VvecTmp[2*j*Nmat+2*i  ]; 
	  Lvec[1][n1][m][ (i-SindB[n1][2*m+1])*nsc+j ].i = VvecTmp[2*j*Nmat+2*i+1]; 
	}        
      }    

      /*****************************************
                     set up Row2ID
      *****************************************/

      ID = 0;
      for (i=0; i<(EindB[n1][2*m+1]-SindB[n1][2*m]+1); i++){

	i2 = i + SindB[n1][2*m];
        Row2ID[i2] = ID;
        ID++;
        ID = ID % numprocs2;
      }

      /*****************************************
          update entries in the inverse of A
      *****************************************/

      dtime(&stime);

      /*
      printf("n1=%2d m=%2d nsc=%2d\n",n1,m,nsc);
      */

      if (0<nsc){

	/* calculate L_{even}^t S^{-1} */

        dtime(&stime1);

#pragma omp parallel shared(n1,m,EindB,SindB,Row2ID,myid2,nsc,IS,Lvec,Vvec2,numop2) private(OMPID,Nthrds,Nprocs,i,i2,j,j2,k,i3,j3,dcsum0,numop0)
        { 

          numop0 = 0.0;

  	  /* get info. on OpenMP */ 

  	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for ( i=OMPID; i<(EindB[n1][2*m]-SindB[n1][2*m]+1); i+=Nthrds ){

	    if ( Row2ID[i+SindB[n1][2*m]]==myid2 ){

	      i2 = i*nsc;

	      for (j=0; j<nsc; j++){
             
		j2 = j*nsc;

		dcsum0.r = 0.0; dcsum0.i = 0.0;
    
		for (k=0; k<nsc; k++){

		  i3 = i2 + k;
		  j3 = j2 + k;   

		  dcsum0.r += IS[n1][m][j3].r*Lvec[0][n1][m][i3].r - IS[n1][m][j3].i*Lvec[0][n1][m][i3].i;
		  dcsum0.i += IS[n1][m][j3].i*Lvec[0][n1][m][i3].r + IS[n1][m][j3].r*Lvec[0][n1][m][i3].i;
		}
        
		Vvec2[i+SindB[n1][2*m]][j] = dcsum0.r + I*dcsum0.i;
	      }        

  	      numop0 += (double)nsc*nsc; 
	    }
	  }    

	  /* calculate L_{odd}^t S^{-1} */

	  for ( i=OMPID; i<(EindB[n1][2*m+1]-SindB[n1][2*m+1]+1); i+=Nthrds ){

	    if ( Row2ID[i+SindB[n1][2*m+1]]==myid2 ){

	      i2 = i*nsc;

	      for (j=0; j<nsc; j++){

		j2 = j*nsc;

		dcsum0.r = 0.0; dcsum0.i = 0.0;

		for (k=0; k<nsc; k++){

		  j3 = j2 + k;   
		  i3 = i2 + k;

		  dcsum0.r += IS[n1][m][j3].r*Lvec[1][n1][m][i3].r - IS[n1][m][j3].i*Lvec[1][n1][m][i3].i;
		  dcsum0.i += IS[n1][m][j3].i*Lvec[1][n1][m][i3].r + IS[n1][m][j3].r*Lvec[1][n1][m][i3].i;
		}

		Vvec2[i+SindB[n1][2*m+1]][j] = dcsum0.r + I*dcsum0.i;
	      }        

 	      numop0 += (double)nsc*nsc; 

	    }
	  }    

          numop2 += numop0;

	} /* #pragma omp parallel */

        dtime(&etime1);
        time51 += etime1 - stime1;

	/* update IAx */

        dtime(&stime1);

	n2 = 0;

	for (m1=0; m1<m2; m1++){

	  m3 = m2*m + m1;

#pragma omp parallel shared(m2,m1,SindB,EindB,n2,m3,Row2ID,myid2,Anum,Ai,n1,m,EindC,SindC,Lvec,Vvec2,IAx) private(OMPID,Nthrds,Nprocs,k2,i,i2,j,j2,nsc2,k)
	  { 

	    double complex csum1,ctmp2;
         
  	    /* get info. on OpenMP */ 

  	    OMPID = omp_get_thread_num();
	    Nthrds = omp_get_num_threads();
	    Nprocs = omp_get_num_procs();

	    /* for even */  

	    if (m2==1) k2 = 0;
	    else       k2 = ((m2/2)<=m1);

	    for ( i=SindB[n2][2*m3]+OMPID; i<=EindB[n2][2*m3]; i+=Nthrds ){

	      if (Row2ID[i]==myid2){ 

		i2 = i - SindB[n2][2*m3];

		for (j=0; j<Anum[2*m3][i2]; j++){

		  j2 = Ai[2*m3][i2][j] + SindB[0][2*m3] - SindB[n1][2*m+k2];
		  csum1 = 0.0;

		  nsc2 = EindC[n1][m]-SindC[n1][m]+1;

		  for (k=0; k<nsc2; k++){
		    ctmp2 = Lvec[k2][n1][m][j2*nsc2+k].r + I*Lvec[k2][n1][m][j2*nsc2+k].i;
		    csum1 += Vvec2[i][k]*ctmp2;
		  }

		  IAx[2*m3][i2][j] += csum1; 
		}
	      }      
	    }

	    /* for odd */  

	    if (m2==1) k2 = 1;
	    else       k2 = ((m2/2)<=m1);

	    for ( i=SindB[n2][2*m3+1]+OMPID; i<=EindB[n2][2*m3+1]; i+=Nthrds ){

	      if (Row2ID[i]==myid2){ 

		i2 = i - SindB[n2][2*m3+1];

		for (j=0; j<Anum[2*m3+1][i2]; j++){

		  j2 = Ai[2*m3+1][i2][j] + SindB[0][2*m3+1] - SindB[n1][2*m+k2];
		  csum1 = 0.0;

		  nsc2 = EindC[n1][m] - SindC[n1][m] + 1;

		  for (k=0; k<nsc2; k++){
		    ctmp2 = Lvec[k2][n1][m][j2*nsc2+k].r + I*Lvec[k2][n1][m][j2*nsc2+k].i;
		    csum1 += Vvec2[i][k]*ctmp2;
		  }

		  IAx[2*m3+1][i2][j] += csum1; 
		}
	      }      
	    }

	  } /* #pragma omp parallel */

	} /* m1 */

        dtime(&etime1);
        time52 += etime1 - stime1;

	/* update the outer IBx */

        dtime(&stime1);

#pragma omp parallel shared(EindC,SindC,n1,m,Bnum,SindB,Bi,Row2ID,myid2,IBx,Vvec2) \
                     private(OMPID,Nthrds,Nprocs,i,j,j2)
	{ 

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for ( i=OMPID; i<(EindC[n1][m]-SindC[n1][m]+1); i+=Nthrds ){

	    /* for even */  

	    for (j=0; j<Bnum[0][n1][m][i]; j++){
             
	      j2 = SindB[n1][2*m] + Bi[0][n1][m][i][j];

	      if (Row2ID[j2]==myid2){ 
		IBx[0][n1][m][i][j] = -Vvec2[j2][i];
	      }
	    }

	    /* for odd */  

	    for (j=0; j<Bnum[1][n1][m][i]; j++){
             
	      j2 = SindB[n1][2*m+1] + Bi[1][n1][m][i][j];

	      if (Row2ID[j2]==myid2){ 
		IBx[1][n1][m][i][j] = -Vvec2[j2][i];
	      }
	    }

	  } /* i */             

        } /* #pragma omp parallel */

        dtime(&etime1);
        time53 += etime1 - stime1;

	/* update the inner IBx */

        dtime(&stime1);

#pragma omp parallel shared(m2,n1,m,SindC,EindC,Row2ID,myid2,Bnum,Bi,SindB,Lvec,Vvec2,IBx) \
                     private(OMPID,Nthrds,Nprocs,mp0,n2,m1,m3,k2,i,i2,j,j2,nsc2,k)
	{ 

	  double complex ctmp1,csum1;

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  mp0 = m2;
      
	  for (n2=0; n2<n1; n2++){
	    for (m1=0; m1<mp0; m1++){

	      m3 = mp0*m + m1;
	      k2 = ((mp0/2)<=m1);

	      for ( i=SindC[n2][m3]+OMPID; i<=EindC[n2][m3]; i+=Nthrds ){

		if (Row2ID[i]==myid2){

		  i2 = i - SindC[n2][m3];

		  /* for even */  

		  for (j=0; j<Bnum[0][n2][m3][i2]; j++){

		    j2 = Bi[0][n2][m3][i2][j] + SindB[n2][2*m3] - SindB[n1][2*m+k2]; 
		    csum1 = 0.0;

		    nsc2 = EindC[n1][m]-SindC[n1][m]+1;

		    for (k=0; k<nsc2; k++){

		      ctmp1 = Lvec[k2][n1][m][j2*nsc2+k].r + I*Lvec[k2][n1][m][j2*nsc2+k].i;
		      csum1 += Vvec2[i][k]*ctmp1;
		    }

		    IBx[0][n2][m3][i2][j] += csum1;
		  }

		  /* for odd */  

		  for (j=0; j<Bnum[1][n2][m3][i2]; j++){

		    j2 = Bi[1][n2][m3][i2][j] + SindB[n2][2*m3+1] - SindB[n1][2*m+k2]; 
		    csum1 = 0.0;

		    nsc2 = EindC[n1][m]-SindC[n1][m]+1;  

		    for (k=0; k<nsc2; k++){
		      ctmp1 = Lvec[k2][n1][m][j2*nsc2+k].r + I*Lvec[k2][n1][m][j2*nsc2+k].i;
		      csum1 += Vvec2[i][k]*ctmp1;
		    }

		    IBx[1][n2][m3][i2][j] += csum1;
		  }

		} /* if (Row2ID[i]==myid2) */
	      }

	    } /* m1 */             

	    mp0 /= 2;

	  } /* n2 */

        } /* #pragma omp parallel */

        dtime(&etime1);
        time54 += etime1 - stime1;

	/* update the inner ICx */

        dtime(&stime1);

#pragma omp parallel shared(m2,n1,m,SindC,EindC,Row2ID,myid2,Cnum,Ci,SindB,Lvec,Vvec2,ICx) \
                     private(OMPID,Nthrds,Nprocs,mp0,n2,m1,m3,k2,i,i2,j,j2,nsc2,k)
	{ 

	  double complex ctmp1,csum1;

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  mp0 = m2;

	  for (n2=0; n2<n1; n2++){
	    for (m1=0; m1<mp0; m1++){

	      m3 = mp0*m + m1;
	      k2 = ((mp0/2)<=m1);

	      for ( i=SindC[n2][m3]+OMPID; i<=EindC[n2][m3]; i+=Nthrds ){

		if (Row2ID[i]==myid2){ 

		  i2 = i - SindC[n2][m3];

		  for (j=0; j<Cnum[n2][m3][i2]; j++){

		    j2 = Ci[n2][m3][i2][j] + SindC[n2][m3] - SindB[n1][2*m+k2]; 
		    csum1 = 0.0;

		    nsc2 = EindC[n1][m]-SindC[n1][m]+1;

		    for (k=0; k<nsc2; k++){
		      ctmp1 = Lvec[k2][n1][m][j2*nsc2+k].r + I*Lvec[k2][n1][m][j2*nsc2+k].i;
		      csum1 += Vvec2[i][k]*ctmp1;
		    }

		    ICx[n2][m3][i2][j] += csum1; 

		  } /* j */
		} /* if (Row2ID[i]==myid2) */
	      } /* i */          
	    } /* m1 */

	    mp0 /= 2;

	  } /* n2 */

        } /* #pragma omp parallel */

        dtime(&etime1);
        time55 += etime1 - stime1;

	/* update the outer ICx */

        if (myid2==0){

	  for (i=0; i<nsc; i++){
	    for (j=0; j<Cnum[n1][m][i]; j++){

	      j2 = Ci[n1][m][i][j];
	      ICx[n1][m][i][j] = IS[n1][m][j2*nsc+i].r + I*IS[n1][m][j2*nsc+i].i;
	    }
	  }
	}

      } /* if (0<nsc), end of "update entries in the inverse of A" */

      dtime(&etime);
      time5 += etime - stime;

    } /* m */                    
  
    /* update m2 and mp */

    m2 *= 2;
    mp /= 2;

  } /* n1 */

  /***********************************************************
              add the part of inverse to DMall
  ***********************************************************/

  dtime(&stime);

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      for (i=SindC[n][m]; i<=EindC[n][m]; i++){
        for (j=0; j<Bnum[0][n][m][i-SindC[n][m]]; j++){

          j1 = Bi[0][n][m][i-SindC[n][m]][j];
          j2 = j1 + SindB[n][2*m];
          i2 = i - SindC[n][m];

          i3 = Index_BC[nl][i];          
          j3 = Index_BC[nl][j2]; 
          k = conv_ind2[i3][j3]; 
  	  Tx[k] = creal(weight*IBx[0][n][m][i2][j]);

          k = conv_ind2[j3][i3];  
  	  Tx[k] = creal(weight*IBx[0][n][m][i2][j]);
	}
      }
    }

    mp /= 2;
  }

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      for (i=SindC[n][m]; i<=EindC[n][m]; i++){
        for (j=0; j<Bnum[1][n][m][i-SindC[n][m]]; j++){

          j1 = Bi[1][n][m][i-SindC[n][m]][j];
          j2 = j1 + SindB[n][2*m+1];
          i2 = i - SindC[n][m];

          i3 = Index_BC[nl][i];          
          j3 = Index_BC[nl][j2]; 
          k = conv_ind2[i3][j3]; 
  	  Tx[k] = creal(weight*IBx[1][n][m][i2][j]);

          k = conv_ind2[j3][i3]; 
  	  Tx[k] = creal(weight*IBx[1][n][m][i2][j]);
	}
      }
    }

    mp /= 2;
  }

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      for (i=SindC[n][m]; i<=EindC[n][m]; i++){
        for (j=0; j<Cnum[n][m][i-SindC[n][m]]; j++){

          j1 = Ci[n][m][i-SindC[n][m]][j];
          j2 = j1 + SindC[n][m];
          i2 = i - SindC[n][m];

          i3 = Index_BC[nl][i];          
          j3 = Index_BC[nl][j2]; 
          k = conv_ind2[i3][j3]; 

  	  Tx[k] = creal(weight*ICx[n][m][i2][j]);
	}
      }
    }

    mp /= 2;
  }

  mp = mpmax;
  for (m=0; m<mp; m++){
    for (i=SindB[0][m]; i<=EindB[0][m]; i++){
      for (j=0; j<Anum[m][i-SindB[0][m]]; j++){

	j1 = Ai[m][i-SindB[0][m]][j];
	j2 = j1 + SindB[0][m];
	i2 = i - SindB[0][m];

	i3 = Index_BC[nl][i];          
	j3 = Index_BC[nl][j2]; 

	k = conv_ind2[i3][j3]; 

	Tx[k] = creal(weight*IAx[m][i2][j]);
      }
    }
  }

  if (strcasecmp(mode,"contour")==0){  

    for (i=0; i<TNZE; i++){

      row = conv_index_row[i];                /* row index    */
      col = conv_index_col[i];                /* column index */
      k = conv_ind2[row][col];  

      DMall[spin][i] += Tx[k];
    }  
  }
  else if (strcasecmp(mode,"retarded")==0){  

    sum = 0.0;
    for (i=0; i<TNZE; i++){

      row = conv_index_row[i];                /* row index    */
      col = conv_index_col[i];                /* column index */
      k = conv_ind2[row][col]; 
      sum += Tx[k]*Sall[i];
    }

    WRG[spin][Epoint] = sum;
  }

  else if (strcasecmp(mode,"force")==0){  

    for (i=0; i<TNZE; i++){

      row = conv_index_row[i];                /* row index    */
      col = conv_index_col[i];                /* column index */
      k = conv_ind2[row][col];  

      DMall[spin][i] += Tx[k];
    }  
  }

  dtime(&etime);
  time6 = etime - stime;

  if (0<measure_time){

    printf("OND myid=%2d numop1=%15.12f\n",myid,numop1); fflush(stdout);
    printf("OND myid=%2d numop2=%15.12f\n",myid,numop2); fflush(stdout);
    printf("OND time1 =%15.12f\n",time1); fflush(stdout);
    printf("OND time1a=%15.12f\n",time1a); fflush(stdout);
    printf("ONDtime1a0=%15.12f\n",time1a0); fflush(stdout);
    printf("ONDtime1a1=%15.12f\n",time1a1); fflush(stdout);
    printf("ONDtime1a2=%15.12f\n",time1a2); fflush(stdout);
    printf("OND time1b=%15.12f\n",time1b); fflush(stdout);
    printf("OND time1c=%15.12f\n",time1c); fflush(stdout);
    printf("OND time1d=%15.12f\n",time1d); fflush(stdout);
    printf("OND time2 =%15.12f\n",time2); fflush(stdout);
    printf("OND time3 =%15.12f\n",time3); fflush(stdout);
    printf("OND time31=%15.12f\n",time31); fflush(stdout);
    printf("OND time32=%15.12f\n",time32); fflush(stdout);
    printf("OND time33=%15.12f\n",time33); fflush(stdout);
    printf("OND time34=%15.12f\n",time34); fflush(stdout);
    printf("OND time4 =%15.12f\n",time4); fflush(stdout);
    printf("OND time41=%15.12f\n",time41); fflush(stdout);
    printf("OND time42=%15.12f\n",time42); fflush(stdout);
    printf("OND time5 =%15.12f\n",time5); fflush(stdout);
    printf("OND time51=%15.12f\n",time51); fflush(stdout);
    printf("OND time52=%15.12f\n",time52); fflush(stdout);
    printf("OND time53=%15.12f\n",time53); fflush(stdout);
    printf("OND time54=%15.12f\n",time54); fflush(stdout);
    printf("OND time55=%15.12f\n",time55); fflush(stdout);
    printf("OND myid=%2d time6 =%15.12f\n",myid,time6); fflush(stdout);
  }

  /* free arrays */

  free(Row2ID);

  free(ie1);
  free(is1);

  for (i=0; i<max_nsc; i++){
    free(Q0[i]);
  }
  free(Q0);

  for (i=0; i<max_nsc; i++){
    free(Q1[i]);
  }
  free(Q1);

  free(VvecTmp);
  free(ISTmp);

  for (i=0; i<max_nsc; i++){
    free(Vvec[i]);
  }
  free(Vvec);

  for (i=0; i<Nmat; i++){
    free(Vvec2[i]);
  }
  free(Vvec2);

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      free(Lvec[0][n][m]);
    }
    free(Lvec[0][n]);
    mp /= 2;
  }
  free(Lvec[0]);

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      free(Lvec[1][n][m]);
    }
    free(Lvec[1][n]);
    mp /= 2;
  }
  free(Lvec[1]);

  for (m=0; m<mpmax; m++){

    nsa = EindB[0][m] - SindB[0][m] + 1;

    Anum[m] = (int*)malloc(sizeof(int)*nsa);

    for (i=SindB[0][m]; i<=EindB[0][m]; i++){

      i2 = i - SindB[0][m];

      free(Ai[m][i2]);
      free(Ax[m][i2]);
      free(IAx[m][i2]);
    }

    free(Ai[m]);
    free(Ax[m]);
    free(IAx[m]);
    free(Anum[m]);
  }
  free(Ai);
  free(Ax);
  free(IAx);
  free(Anum);

  for (m=0; m<mp; m++){
    free(IA[m]);
  }
  free(IA);

  free(IA0);

  mp = mpmax/2;
  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      free(IS[n][m]);
    }
    free(IS[n]);
    mp /= 2;
  }  
  free(IS);

  mp = mpmax/2;

  for (n=0; n<nl; n++){
    for (m=0; m<mp; m++){
      for (i=SindC[n][m]; i<=EindC[n][m]; i++){

        free(IBx[0][n][m][i-SindC[n][m]]);
        free(IBx[1][n][m][i-SindC[n][m]]);
        free(ICx[n][m][i-SindC[n][m]]);

        free(Bx[0][n][m][i-SindC[n][m]]);
        free(Bx[1][n][m][i-SindC[n][m]]);
        free(Cx[n][m][i-SindC[n][m]]);

        free(Bi[0][n][m][i-SindC[n][m]]);
        free(Bi[1][n][m][i-SindC[n][m]]);
        free(Ci[n][m][i-SindC[n][m]]);

      } /* i */

      free(IBx[0][n][m]);
      free(IBx[1][n][m]);
      free(ICx[n][m]);

      free(Bx[0][n][m]);
      free(Bx[1][n][m]);
      free(Cx[n][m]);

      free(Bi[0][n][m]);
      free(Bi[1][n][m]);
      free(Ci[n][m]);

      free(Bnum[0][n][m]);
      free(Bnum[1][n][m]);
      free(Cnum[n][m]);

    } /* m */

    free(IBx[0][n]);
    free(IBx[1][n]);
    free(ICx[n]);

    free(Bx[0][n]);
    free(Bx[1][n]);
    free(Cx[n]);

    free(Bi[0][n]);
    free(Bi[1][n]);
    free(Ci[n]);

    free(Bnum[0][n]);
    free(Bnum[1][n]);
    free(Cnum[n]);

    mp /= 2;

  } /* n */

  free(IBx[0]);
  free(IBx[1]);
  free(ICx);

  free(Bx[0]);
  free(Bx[1]);
  free(Cx);

  free(Bi[0]);
  free(Bi[1]);
  free(Ci);

  free(Bnum[0]);
  free(Bnum[1]);
  free(Cnum);

  free(invp);

  free(Ti);
  free(Tx);
  free(Tp);
  free(Tp0);

  free(request_recv);
  free(request_send);
  free(stat_send);
}







static void qsort_complex(long n, long *a, double complex *b)
{
  long int i;
  clists *AB;

  AB = (clists*)malloc(sizeof(clists)*n);

  for (i=0; i<n; i++){
    AB[i].a = a[i];     
    AB[i].b = b[i];
  }

  qsort(AB, n, sizeof(clists), (int(*)(const void*, const void*))clists_cmp);

  for (i=0; i<n; i++){
    a[i] = AB[i].a;
    b[i] = AB[i].b;
  }

  free(AB);
}


static int clists_cmp(const clists *x, const clists *y)
{
  return (x->a < y->a ? -1 :
          y->a < x->a ?  1 : 0);
}




static double Cluster_collinear_ON2_Force(
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh, double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
				    double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  double TStime,TEtime;
  double stime,etime;
  int numprocs,myid,ID,tag=999;
  int i,j,k,k2,wan,wanA,wanB,Epoint,column;
  int tnoA,tnoB,LB_AN,i1,j1,l;
  int MA_AN,GA_AN,GB_AN,Gc_AN,h_AN,ni,nj,TNZE;
  int Cwan,Hwan,Gh_AN,Anum,tnum,num,num4;
  int loop1,Nloop1,Nmat,spin,NEV,imax;
  int *NZE,*PZE,*MP;
  double **Hall,*Sall;
  double **EDMall;
  double **eval,**evec;
  double **WRG;
  double eachsum,totalsum,pChemP; 
  double TZ,ChemP_MIN,ChemP_MAX,ymax;
  double Dnum,spin_degeneracy;
  double totalsum1,f0,f1,x1,y1,x0,y0;
  double x,Trial_ChemP,dE_step;
  double cTrial_ChemP,ChemP0;
  double My_Eele1[2]; 
  int po,po3,po4,po6,loopN,free_flag; 
  int num_loop,end_flag;
  int Depth_Level,Number_Division;
  int *conv_index_row;
  int *conv_index_col;
  int **conv_ind2;
  int *My_NZeros;
  int *is1,*ie1;
  /* for MPI */
  int myworld1;
  int numprocs1,myid1;
  int numprocs2,myid2;
  int Num_Comm_World1;
  int *NPROCS_ID1,*NPROCS_WD1;
  int *Comm_World1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;
  int myworld2B;
  int numprocs2B,myid2B;
  int Num_Comm_World2B;
  int *NPROCS_ID2B,*NPROCS_WD2B;
  int *Comm_World2B;
  int *Comm_World_StartID2B;
  MPI_Comm *MPI_CommWD2B;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* for elapsed time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  if (0<measure_time) dtime(&stime);

  /****************************************************
         set up the first communication world
  ****************************************************/

  Num_Comm_World1 = SpinP_switch + 1; 

  NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs); 
  NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

  Make_Comm_Worlds(mpi_comm_level1, myid, numprocs, Num_Comm_World1, &myworld1, MPI_CommWD1, 
                   NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

  /****************************************************
           calculation of dimension of matrix
  ****************************************************/

  Nmat = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    Nmat += Spe_Total_CNO[wanA];
  }

  /****************************************************
                  total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }

  /****************************************************
                   calculate TNZE
  ****************************************************/

  TNZE = 0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Cwan = WhatSpecies[Gc_AN];

    nj = 0;
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      nj += Spe_Total_NO[Hwan];
    }

    TNZE += Spe_Total_NO[Cwan]*nj;
  }

  /****************************************************
                   allocate arrays
  ****************************************************/

  WRG = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    WRG[spin] = (double*)malloc(sizeof(double)*ON2_Npoles_f);
    for (i=0; i<ON2_Npoles_f; i++) WRG[spin][i] = 0.0;
  }

  conv_ind2 = (int**)malloc(sizeof(int*)*Nmat);
  for (i=0; i<Nmat; i++){
    conv_ind2[i] = (int*)malloc(sizeof(int)*Nmat);
  } 

  EDMall = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    EDMall[spin] = (double*)malloc(sizeof(double)*TNZE);
    for (i=0; i<TNZE; i++) EDMall[spin][i] = 0.0;
  }

  Hall = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    Hall[spin] = (double*)malloc(sizeof(double)*TNZE);
  }

  Sall = (double*)malloc(sizeof(double)*TNZE);

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
    
  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }

  conv_index_row = (int*)malloc(sizeof(int)*TNZE);
  conv_index_col = (int*)malloc(sizeof(int)*TNZE);

  /******************************************
      MPI communication of nh and CntOLP
  ******************************************/

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /******************
       for nh
  ******************/

  for (spin=0; spin<=SpinP_switch; spin++){

    /* set nh */

    k = is1[myid];
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  for (j=0; j<tnoB; j++){
	    Hall[spin][k] = nh[spin][MA_AN][LB_AN][i][j]; 
	    k++;
	  }
	}
      }
    }

    /* MPI Hall */

    MPI_Barrier(mpi_comm_level1);
    
    for (ID=0; ID<numprocs; ID++){
      k = is1[ID];
      MPI_Bcast(&Hall[spin][k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
    }

  }

  /******************
      for CntOLP
  ******************/

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          Sall[k] = CntOLP[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  /* MPI S1 */

  MPI_Barrier(mpi_comm_level1);

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&Sall[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /******************************************
   set up conv_index_row and conv_index_col
  ******************************************/

  k = is1[myid];

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){

          conv_index_row[k] = MP[GA_AN] + i - 1;
          conv_index_col[k] = MP[GB_AN] + j - 1;
          k++;
	}
      }
    }
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&conv_index_row[k], My_NZeros[ID], MPI_INT, ID, mpi_comm_level1);
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&conv_index_col[k], My_NZeros[ID], MPI_INT, ID, mpi_comm_level1);
  }

  /****************************************************
                 set up NZE and PZE
  ****************************************************/

  NZE = (int*)malloc(sizeof(int)*Nmat);
  PZE = (int*)malloc(sizeof(int)*Nmat);

  for (i=0; i<Nmat; i++){
    NZE[i] = 0;
    PZE[i] = -1;
  }

  for (k=0; k<TNZE; k++){

    i = conv_index_row[k];
    
    NZE[i]++;
    if (PZE[i]==-1) PZE[i] = k;
  }

  /* freeing of arrays */

  free(My_NZeros);
  free(ie1);

  if (0<measure_time){
    dtime(&etime);
    printf("myid=%2d OND_Solver time1=%15.12f\n",myid,etime-stime);fflush(stdout);
  }

  /*************************************************************************
   *************************************************************************

     Part 0: nested dissection of matrix 

   *************************************************************************
  *************************************************************************/

  free_flag = 0;
  Nested_Dissection(free_flag,Nmat,MP,&Depth_Level,&Number_Division);

  /*************************************************************************
   *************************************************************************

     Part 1: contour integration with a chemical potential \mu_0

     main loop which is decomposed to 
     spin and energy point

   *************************************************************************
  *************************************************************************/

  MPI_Barrier(mpi_comm_level1);

  Nloop1 = (SpinP_switch+1)*ON2_Npoles_f;

  if (Nloop1<=numprocs){

    Num_Comm_World2B = Nloop1;
                  
    NPROCS_ID2B = (int*)malloc(sizeof(int)*numprocs);
    Comm_World2B = (int*)malloc(sizeof(int)*numprocs);
    NPROCS_WD2B = (int*)malloc(sizeof(int)*Num_Comm_World2B);
    Comm_World_StartID2B = (int*)malloc(sizeof(int)*Num_Comm_World2B);
    MPI_CommWD2B = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2B);

    Make_Comm_Worlds(mpi_comm_level1, myid, numprocs, Num_Comm_World2B, &myworld2B, MPI_CommWD2B, 
		     NPROCS_ID2B, Comm_World2B, NPROCS_WD2B, Comm_World_StartID2B);

    MPI_Comm_size(MPI_CommWD2B[myworld2B],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2B[myworld2B],&myid2);
  }
  else {
    numprocs2 = 1;
    myid2 = 0;
  }

  for (spin=0; spin<(SpinP_switch+1); spin++){
    for (i=0; i<TNZE; i++) EDMall[spin][i] = 0.0;
  }

  if (Nloop1<=numprocs){

    loop1 = myworld2B; 

    /* get spin, Epoint, and column */
    spin = loop1/ON2_Npoles_f;
    Epoint = loop1 - spin*ON2_Npoles_f;

    OND_Solver("force",
	       myid,numprocs,
	       myid2,numprocs2,
	       Nloop1,
	       MPI_CommWD2B[myworld2B],
	       spin,Epoint,0.0,
	       Nmat,NZE,PZE,
	       Depth_Level,
	       Number_Division,
	       ChemP, 
	       Hall,Sall,EDMall,
	       WRG,
	       conv_index_row, 
	       conv_index_col,
	       conv_ind2,
	       TNZE);
  }  

  else {

    for ( loop1=myid; loop1<Nloop1; loop1+=numprocs ){

      /* get spin, Epoint, and column */
      spin = loop1/ON2_Npoles_f;
      Epoint = loop1 - spin*ON2_Npoles_f;

      /* call OND_Solver */

      OND_Solver("force",
		 myid,numprocs, 
		 myid2,numprocs2,
		 Nloop1,
		 mpi_comm_level1,
		 spin,Epoint,0.0,
		 Nmat,NZE,PZE,
		 Depth_Level,
		 Number_Division, 
		 ChemP, 
		 Hall,Sall,EDMall,
		 WRG,
		 conv_index_row, 
		 conv_index_col,
		 conv_ind2,
		 TNZE);

    }
  } /* else */

  /*************************************************************************
        store the energy density matrix into an array EDM
  **************************************************************************/

  MPI_Barrier(mpi_comm_level1);
  
  for (spin=0; spin<=SpinP_switch; spin++){

    MPI_Allreduce(&EDMall[spin][0], &Sall[0], TNZE, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    k = is1[myid];
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  for (j=0; j<tnoB; j++){
	    EDM[spin][MA_AN][LB_AN][i][j] = Sall[k];
	    k++;
	  }
	}
      }
    }
  }
 
  /****************************************************
     freeing Index_BC, SindB, EindB, SindC, EindE
  ****************************************************/

  free_flag = 1;
  Nested_Dissection(free_flag,Nmat,MP,&Depth_Level,&Number_Division);

  /****************************************************
                   freeing of arrays
  ****************************************************/

  free(is1);

  free(NZE);
  free(PZE);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(WRG[spin]);
  }
  free(WRG);

  for (i=0; i<Nmat; i++){
    free(conv_ind2[i]);
  } 
  free(conv_ind2);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(EDMall[spin]);
  }
  free(EDMall);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(Hall[spin]);
  }
  free(Hall);

  free(Sall);

  free(MP);
  free(conv_index_row);
  free(conv_index_col);

  /* freeing of arrays for the first world */

  if (Num_Comm_World1<=numprocs){
    MPI_Comm_free(&MPI_CommWD1[myworld1]);
  }

  free(NPROCS_ID1);
  free(Comm_World1);
  free(NPROCS_WD1);
  free(Comm_World_StartID1);
  free(MPI_CommWD1);

  /* freeing of arrays for the second B world */

  if (Nloop1<=numprocs){

    MPI_Comm_free(&MPI_CommWD2B[myworld2B]);
    free(NPROCS_ID2B);
    free(Comm_World2B);
    free(NPROCS_WD2B);
    free(Comm_World_StartID2B);
    free(MPI_CommWD2B);
  }

  /* for elapsed time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  return (TEtime-TStime);
}












static double Cluster_collinear_ON2_Iter(
				    int SCF_iter,
				    int SpinP_switch,
				    double *****nh, double ****CntOLP,
				    double *****CDM,
				    double *****EDM,
				    double Eele0[2], double Eele1[2])
{
  static int firsttime=1;
  double TStime,TEtime;
  double stime,etime;
  int numprocs,myid,ID,tag=999;
  int i,j,k,k2,wan,wanA,wanB,Epoint,column;
  int tnoA,tnoB,LB_AN,i1,j1,l;
  int MA_AN,GA_AN,GB_AN,Gc_AN,h_AN,ni,nj,TNZE;
  int Cwan,Hwan,Gh_AN,Anum,tnum,num,num4;
  int loop1,Nloop1,Nmat,spin,NEV,imax;
  int *NZE,*PZE,*MP;
  double **Hall,*Sall;
  double **DMall;
  double **eval,**evec;
  double **WRG;
  double eachsum,totalsum,pChemP; 
  double TZ,ChemP_MIN,ChemP_MAX,ymax;
  double Dnum,spin_degeneracy;
  double totalsum1,f0,f1,x1,y1,x0,y0;
  double x,Trial_ChemP,dE_step;
  double cTrial_ChemP,ChemP0;
  double x3,y3,startE,Real_Ene=0.0;
  double My_Eele1[2]; 
  static double pTrial_ChemP[4],pDiff_Num[5]; 
  static double DeltaMu[7],DeltaN[3],DeltaNp[3];  
  int po,po3,po4,po6,loopN,free_flag; 
  int num_loop,end_flag;
  int Depth_Level,Number_Division;
  int *conv_index_row;
  int *conv_index_col;
  int **conv_ind2;
  int *My_NZeros;
  int *is1,*ie1;
  /* for MPI */
  int myworld1;
  int numprocs1,myid1;
  int numprocs2,myid2;
  int Num_Comm_World1;
  int *NPROCS_ID1,*NPROCS_WD1;
  int *Comm_World1;
  int *Comm_World_StartID1;
  MPI_Comm *MPI_CommWD1;
  int myworld2B;
  int numprocs2B,myid2B;
  int Num_Comm_World2B;
  int *NPROCS_ID2B,*NPROCS_WD2B;
  int *Comm_World2B;
  int *Comm_World_StartID2B;
  MPI_Comm *MPI_CommWD2B;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* for elapsed time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TStime);

  if (0<measure_time) dtime(&stime);

  /****************************************************
         set up the first communication world
  ****************************************************/

  Num_Comm_World1 = SpinP_switch + 1; 

  NPROCS_ID1 = (int*)malloc(sizeof(int)*numprocs); 
  Comm_World1 = (int*)malloc(sizeof(int)*numprocs); 
  NPROCS_WD1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  Comm_World_StartID1 = (int*)malloc(sizeof(int)*Num_Comm_World1); 
  MPI_CommWD1 = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World1);

  Make_Comm_Worlds(mpi_comm_level1, myid, numprocs, Num_Comm_World1, &myworld1, MPI_CommWD1, 
                   NPROCS_ID1, Comm_World1, NPROCS_WD1, Comm_World_StartID1);

  /****************************************************
           calculation of dimension of matrix
  ****************************************************/

  Nmat = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    Nmat += Spe_Total_CNO[wanA];
  }

  /****************************************************
                  total core charge
  ****************************************************/

  TZ = 0.0;
  for (i=1; i<=atomnum; i++){
    wan = WhatSpecies[i];
    TZ = TZ + Spe_Core_Charge[wan];
  }

  /****************************************************
                   calculate TNZE
  ****************************************************/

  TNZE = 0;
  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    Cwan = WhatSpecies[Gc_AN];

    nj = 0;
    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
      Gh_AN = natn[Gc_AN][h_AN];
      Hwan = WhatSpecies[Gh_AN];
      nj += Spe_Total_NO[Hwan];
    }

    TNZE += Spe_Total_NO[Cwan]*nj;
  }

  /****************************************************
                   allocate arrays
  ****************************************************/

  WRG = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    WRG[spin] = (double*)malloc(sizeof(double)*ON2_Npoles);
    for (i=0; i<ON2_Npoles; i++) WRG[spin][i] = 0.0;
  }

  conv_ind2 = (int**)malloc(sizeof(int*)*Nmat);
  for (i=0; i<Nmat; i++){
    conv_ind2[i] = (int*)malloc(sizeof(int)*Nmat);
  } 

  DMall = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    DMall[spin] = (double*)malloc(sizeof(double)*TNZE);
    for (i=0; i<TNZE; i++) DMall[spin][i] = 0.0;
  }

  Hall = (double**)malloc(sizeof(double*)*(SpinP_switch+1));
  for (spin=0; spin<(SpinP_switch+1); spin++){
    Hall[spin] = (double*)malloc(sizeof(double)*TNZE);
  }

  Sall = (double*)malloc(sizeof(double)*TNZE);

  MP = (int*)malloc(sizeof(int)*(atomnum+1));
    
  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum += Spe_Total_CNO[wanA];
  }

  conv_index_row = (int*)malloc(sizeof(int)*TNZE);
  conv_index_col = (int*)malloc(sizeof(int)*TNZE);

  /******************************************
      MPI communication of nh and CntOLP
  ******************************************/

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /******************
       for nh
  ******************/

  for (spin=0; spin<=SpinP_switch; spin++){

    /* set nh */

    k = is1[myid];
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  for (j=0; j<tnoB; j++){
	    Hall[spin][k] = nh[spin][MA_AN][LB_AN][i][j]; 
	    k++;
	  }
	}
      }
    }

    /* MPI Hall */

    MPI_Barrier(mpi_comm_level1);
    
    for (ID=0; ID<numprocs; ID++){
      k = is1[ID];
      MPI_Bcast(&Hall[spin][k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
    }

  }

  /******************
      for CntOLP
  ******************/

  k = is1[myid];
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){
          Sall[k] = CntOLP[MA_AN][LB_AN][i][j]; 
          k++;
	}
      }
    }
  }

  /* MPI S1 */

  MPI_Barrier(mpi_comm_level1);

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&Sall[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
  }

  /******************************************
   set up conv_index_row and conv_index_col
  ******************************************/

  k = is1[myid];

  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];    
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];
    for (i=0; i<tnoA; i++){
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
        GB_AN = natn[GA_AN][LB_AN];
        wanB = WhatSpecies[GB_AN];
        tnoB = Spe_Total_CNO[wanB];
        for (j=0; j<tnoB; j++){

          conv_index_row[k] = MP[GA_AN] + i - 1;
          conv_index_col[k] = MP[GB_AN] + j - 1;
          k++;
	}
      }
    }
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&conv_index_row[k], My_NZeros[ID], MPI_INT, ID, mpi_comm_level1);
  }

  for (ID=0; ID<numprocs; ID++){
    k = is1[ID];
    MPI_Bcast(&conv_index_col[k], My_NZeros[ID], MPI_INT, ID, mpi_comm_level1);
  }

  /****************************************************
                 set up NZE and PZE
  ****************************************************/

  NZE = (int*)malloc(sizeof(int)*Nmat);
  PZE = (int*)malloc(sizeof(int)*Nmat);

  for (i=0; i<Nmat; i++){
    NZE[i] = 0;
    PZE[i] = -1;
  }

  for (k=0; k<TNZE; k++){

    i = conv_index_row[k];
    
    NZE[i]++;
    if (PZE[i]==-1) PZE[i] = k;
  }

  /* freeing of arrays */

  free(My_NZeros);
  free(ie1);

  if (0<measure_time){
    dtime(&etime);
    printf("myid=%2d OND_Solver time1=%15.12f\n",myid,etime-stime);fflush(stdout);
  }

  /*************************************************************************
   *************************************************************************

     Part 0: nested dissection of matrix 

   *************************************************************************
  *************************************************************************/

  free_flag = 0;
  Nested_Dissection(free_flag,Nmat,MP,&Depth_Level,&Number_Division);

  /*************************************************************************
   *************************************************************************

     Part 1: contour integration with a chemical potential \mu_0

     main loop which is decomposed to 
     spin and energy point

   *************************************************************************
  *************************************************************************/

  MPI_Barrier(mpi_comm_level1);

  Nloop1 = (SpinP_switch+1)*ON2_Npoles;

  if (Nloop1<=numprocs){

    Num_Comm_World2B = Nloop1;
                  
    NPROCS_ID2B = (int*)malloc(sizeof(int)*numprocs);
    Comm_World2B = (int*)malloc(sizeof(int)*numprocs);
    NPROCS_WD2B = (int*)malloc(sizeof(int)*Num_Comm_World2B);
    Comm_World_StartID2B = (int*)malloc(sizeof(int)*Num_Comm_World2B);
    MPI_CommWD2B = (MPI_Comm*)malloc(sizeof(MPI_Comm)*Num_Comm_World2B);

    Make_Comm_Worlds(mpi_comm_level1, myid, numprocs, Num_Comm_World2B, &myworld2B, MPI_CommWD2B, 
		     NPROCS_ID2B, Comm_World2B, NPROCS_WD2B, Comm_World_StartID2B);

    MPI_Comm_size(MPI_CommWD2B[myworld2B],&numprocs2);
    MPI_Comm_rank(MPI_CommWD2B[myworld2B],&myid2);
  }
  else {
    numprocs2 = 1;
    myid2 = 0;
  }

  end_flag = 0;
  num_loop = 1;  
  Trial_ChemP = ChemP;

  do {

    for (spin=0; spin<(SpinP_switch+1); spin++){
      for (i=0; i<TNZE; i++) DMall[spin][i] = 0.0;
    }

    if (0<measure_time) dtime(&stime);

    if (Nloop1<=numprocs){

      loop1 = myworld2B; 

      /* get spin, Epoint, and column */
      spin = loop1/ON2_Npoles;
      Epoint = loop1 - spin*ON2_Npoles;

      OND_Solver("contour",
                 myid,numprocs,
                 myid2,numprocs2,
		 Nloop1,
		 MPI_CommWD2B[myworld2B],
		 spin,Epoint,Real_Ene,
		 Nmat,NZE,PZE,
		 Depth_Level,
		 Number_Division,
		 Trial_ChemP, 
		 Hall,Sall,DMall,
                 WRG,
		 conv_index_row, 
		 conv_index_col,
		 conv_ind2,
		 TNZE);
    }  

    else {

      for ( loop1=myid; loop1<Nloop1; loop1+=numprocs ){

	/* get spin, Epoint, and column */
	spin = loop1/ON2_Npoles;
	Epoint = loop1 - spin*ON2_Npoles;

	/* call OND_Solver */

	OND_Solver("contour",
                   myid,numprocs, 
                   myid2,numprocs2,
		   Nloop1,
		   mpi_comm_level1,
		   spin,Epoint,Real_Ene,
		   Nmat,NZE,PZE,
		   Depth_Level,
		   Number_Division, 
		   Trial_ChemP, 
		   Hall,Sall,DMall,
                   WRG,
		   conv_index_row, 
		   conv_index_col,
		   conv_ind2,
		   TNZE);



      }

    } /* else */

    if (0<measure_time){
      dtime(&etime);
      printf("myid=%2d OND_Solver time2 =%15.12f\n",myid,etime-stime);fflush(stdout);
    }

    /*******************************************************
     calculate "each" number of electrons on each processor, 
     and MPI communicate to get the total sum.
    ********************************************************/

    MPI_Barrier(mpi_comm_level1);

    if (0<measure_time) dtime(&stime);

    eachsum = 0.0;
    for (spin=0; spin<=SpinP_switch; spin++){
      for (i=0; i<TNZE; i++){
	eachsum += DMall[spin][i]*Sall[i];
      }
    }

    MPI_Allreduce(&eachsum, &totalsum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    if (SpinP_switch==0){
      eachsum  *= 2.0;
      totalsum *= 2.0;
    }

    if (0<measure_time){
      dtime(&etime);
      printf("myid=%2d OND_Solver time3=%15.12f\n",myid,etime-stime);fflush(stdout);
    }

    /* set x3 and y3 */

    x3 = Trial_ChemP;
    y3 = -TZ + totalsum + system_charge;
    pTrial_ChemP[3] = x3;
    pDiff_Num[3] = y3;

    if (0<measure_time){
      printf("myid=%2d eachsum=%15.12f totalsum=%15.12f\n",myid,eachsum,totalsum);
      if (myid==0)  
      printf("SSS SCF_iter=%2d num_loop=%2d x3=%18.15f y3=%18.15f\n",SCF_iter,num_loop,x3,y3);
    }

    /*******************************************************
      find the next Trial_ChemP.
      updating Trial_ChemP is performed by two ways:

      (1) Step 1:
      if fabs(y3) is large enough so that finding a proper 
      Trial_ChemP can be difficult, then we try to find 
      Trial_ChemP using the retarded Green function, G(E+i0^+).

      (2) Step 2:
      if fabs(y3) is small, then a linear interpolation or 
      the Muller method is used to estimate Trial_ChemP.  
    ********************************************************/

    /**********************************************
      Step 1:
    ***********************************************/

    if (0.01<fabs(y3) && num_loop==1){

      cTrial_ChemP = Trial_ChemP;
      dE_step = 0.02/eV2Hartree; 

      if (0.0<y3){
	startE = x3 - dE_step*(double)ON2_Npoles*0.8; 
      } 
      else {
	startE = x3 - dE_step*(double)ON2_Npoles*0.2; 
      }

      if (Nloop1<=numprocs){

	loop1 = myworld2B; 

	/* get spin, Epoint, and column */
	spin = loop1/ON2_Npoles;
	Epoint = loop1 - spin*ON2_Npoles;
	Real_Ene = startE + dE_step*(double)Epoint; 

	OND_Solver("retarded",
		   myid,numprocs,
		   myid2,numprocs2,
		   Nloop1,
		   MPI_CommWD2B[myworld2B],
		   spin,Epoint,Real_Ene,
		   Nmat,NZE,PZE,
		   Depth_Level,
		   Number_Division,
		   Trial_ChemP, 
		   Hall,Sall,DMall,
		   WRG,
		   conv_index_row, 
		   conv_index_col,
		   conv_ind2,
		   TNZE);
      }  

      else {

	for ( loop1=myid; loop1<Nloop1; loop1+=numprocs ){

	  /* get spin, Epoint, and column */
	  spin = loop1/ON2_Npoles;
	  Epoint = loop1 - spin*ON2_Npoles;
	  Real_Ene = startE + dE_step*(double)Epoint; 

	  /* call OND_Solver */

	  OND_Solver("retarded",
		     myid,numprocs, 
		     myid2,numprocs2,
		     Nloop1,
		     mpi_comm_level1,
		     spin,Epoint,Real_Ene,
		     Nmat,NZE,PZE,
		     Depth_Level,
		     Number_Division, 
		     Trial_ChemP, 
		     Hall,Sall,DMall,
		     WRG,
		     conv_index_row, 
		     conv_index_col,
		     conv_ind2,
		     TNZE);

	}
      } /* else */

      /*******************************
        search the next Trial_ChemP
      *******************************/

      if (0.0<y3)
	ChemP0 = x3 - dE_step*(double)ON2_Npoles*0.8; 
      else 
	ChemP0 = x3 + dE_step*(double)ON2_Npoles*0.8; 

      eachsum = 0.0;  

      for (spin=0; spin<=SpinP_switch; spin++){
	for (i=0; i<ON2_Npoles; i++){

	  Real_Ene = startE + dE_step*(double)i;
	  x = (Real_Ene - Trial_ChemP)*Beta;
	  f0 = 1.0/(1.0 + exp(x));
	  x = (Real_Ene - ChemP0)*Beta;
	  f1 = 1.0/(1.0 + exp(x));
	  eachsum += WRG[spin][i]*(f1-f0);
	}
      }

      eachsum = eachsum/PI*dE_step;
      if (SpinP_switch==0) eachsum *= 2.0;

      MPI_Allreduce(&eachsum, &totalsum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

     if ( 1<measure_time ){
        printf("GGG1 SCF_iter=%2d Trial_ChemP=%15.12f ChemP0=%15.12f y3=%15.12f totalsum=%15.12f\n",
               SCF_iter,Trial_ChemP,ChemP0,y3,totalsum);
     }
    
      if ( 0.0<(y3*(y3+totalsum)) ){

        /* just use ChemP0 */
         
	Trial_ChemP = ChemP0;
      }
      else {

	/* refinement of Tria_ChemP by a bisection method */

	/* set up the range of Trial_ChemP */

	if (0.0<y3){
	  ChemP_MAX = Trial_ChemP;
	  ChemP_MIN = ChemP0;
	}
	else {
	  ChemP_MAX = ChemP0;
	  ChemP_MIN = Trial_ChemP;
	} 

	/* do refinement */

	do {
        
	  ChemP0 = 0.50*(ChemP_MAX + ChemP_MIN);
         
	  eachsum = 0.0;  

	  for (spin=0; spin<=SpinP_switch; spin++){
	    for (i=0; i<ON2_Npoles; i++){

	      Real_Ene = startE + dE_step*(double)i;
	      x = (Real_Ene - Trial_ChemP)*Beta;
	      f0 = 1.0/(1.0 + exp(x));
	      x = (Real_Ene - ChemP0)*Beta;
	      f1 = 1.0/(1.0 + exp(x));
	      eachsum += WRG[spin][i]*(f1-f0);
	    }
	  }

          eachsum = eachsum/PI*dE_step;
          if (SpinP_switch==0) eachsum *= 2.0;

	  MPI_Allreduce(&eachsum, &totalsum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

          if ( 1<measure_time ){
            printf("EEE myid=%2d SCF_iter=%2d ChemP=%15.12f diff=%15.12f\n",myid,SCF_iter,ChemP0,y3+totalsum); 
	  }

	  if ( 0.0<=(y3+totalsum) ){
	    ChemP_MAX = ChemP0;
	  } 
	  else {
	    ChemP_MIN = ChemP0;
	  } 

	  if (fabs(y3+totalsum)<1.0e-8) po6 = 1;

	} while (po6==0);         

	/* update Trial_ChemP */

	Trial_ChemP = cTrial_ChemP + 2.0*(ChemP0-cTrial_ChemP);
      }

      pTrial_ChemP[0] = x3;
      pDiff_Num[0] = y3;

      DeltaMu[6] = Trial_ChemP - cTrial_ChemP;
      pDiff_Num[4] = y3;
      
    }

    /**********************************************
      Step 2:
    ***********************************************/
 
    else {

      Trial_ChemP = Find_Trial_ChemP_by_Muller( SCF_iter, num_loop, 
                                                pTrial_ChemP,  pDiff_Num, 
                                                DeltaMu, DeltaN, DeltaNp );
    }

    /* check its convergence */

    if (     fabs(y3)<1.0e-8
         || (fabs(y3)<1.0e-7 && SCF_iter<5) ){

      end_flag = 1;
    }
    else {
      num_loop++;
    }

  } while (end_flag==0 && num_loop<13);

  /* update the new ChemP */

  ChemP = Trial_ChemP;

  /* show num_loop */

  if (myid==Host_ID){
    printf("<Cluster2>  SCF_iter=%2d ChemP_Iter=%2d\n",SCF_iter,num_loop); 
  }

  /*************************************************************************
           store the density matrix by the part 1 into an array DM
  **************************************************************************/

  MPI_Barrier(mpi_comm_level1);
  
  for (spin=0; spin<=SpinP_switch; spin++){

    MPI_Allreduce(&DMall[spin][0], &Sall[0], TNZE, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    k = is1[myid];
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  for (j=0; j<tnoB; j++){
	    CDM[spin][MA_AN][LB_AN][i][j] = Sall[k];
	    k++;
	  }
	}
      }
    }
  }

  /****************************************************
               calculation of band energy
  ****************************************************/

  My_Eele1[0] = 0.0;
  My_Eele1[1] = 0.0;

  for (spin=0; spin<=SpinP_switch; spin++){
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (j=0; j<=FNAN[GA_AN]; j++){
	wanB = WhatSpecies[natn[GA_AN][j]];
	tnoB = Spe_Total_CNO[wanB];
	for (k=0; k<tnoA; k++){
	  for (l=0; l<tnoB; l++){
	    My_Eele1[spin] += CDM[spin][MA_AN][j][k][l]*nh[spin][MA_AN][j][k][l];
	  }
	}
      }
    }
  }

  MPI_Barrier(mpi_comm_level1);
  
  /* MPI, My_Eele1 */
  for (spin=0; spin<=SpinP_switch; spin++){
    MPI_Allreduce(&My_Eele1[spin], &Eele1[spin], 1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);
  }

  if (SpinP_switch==0) Eele1[1] = Eele1[0];

  /****************************************************
     freeing Index_BC, SindB, EindB, SindC, EindE
  ****************************************************/

  free_flag = 1;
  Nested_Dissection(free_flag,Nmat,MP,&Depth_Level,&Number_Division);

  /****************************************************
                   freeing of arrays
  ****************************************************/

  free(is1);

  free(NZE);
  free(PZE);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(WRG[spin]);
  }
  free(WRG);

  for (i=0; i<Nmat; i++){
    free(conv_ind2[i]);
  } 
  free(conv_ind2);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(DMall[spin]);
  }
  free(DMall);

  for (spin=0; spin<(SpinP_switch+1); spin++){
    free(Hall[spin]);
  }
  free(Hall);

  free(Sall);

  free(MP);
  free(conv_index_row);
  free(conv_index_col);

  /* freeing of arrays for the first world */

  if (Num_Comm_World1<=numprocs){
    MPI_Comm_free(&MPI_CommWD1[myworld1]);
  }

  free(NPROCS_ID1);
  free(Comm_World1);
  free(NPROCS_WD1);
  free(Comm_World_StartID1);
  free(MPI_CommWD1);

  /* freeing of arrays for the second B world */

  if (Nloop1<=numprocs){

    MPI_Comm_free(&MPI_CommWD2B[myworld2B]);
    free(NPROCS_ID2B);
    free(Comm_World2B);
    free(NPROCS_WD2B);
    free(Comm_World_StartID2B);
    free(MPI_CommWD2B);
  }

  /* for elapsed time */
  MPI_Barrier(mpi_comm_level1);
  dtime(&TEtime);
  return (TEtime-TStime);
}




static double Find_Trial_ChemP_by_Muller(
        int SCF_iter, int num_loop,   
        double pTrial_ChemP[4], double pDiff_Num[5], 
        double DeltaMu[7], double DeltaN[3], double DeltaNp[3])
{
  static double stepw;
  double y31,x0,x1,x2,x3,y0,y1,y2,y3;
  int i,i0,i1,i2,num4,po3,po4;
  double a,b,c,d,g0,g1,g2,dFa,dFb,dFc,dF,dF0,F;
  double xx[5],yy[5],zz[5],yy0[5];
  double z0,z1,z2,a1,a2,a3,det;
  double a11,a12,a13,a21,a22,a23,a31,a32,a33;
  double scaling4;
  double Trial_ChemP,Trial_ChemP0;
  int flag_nan;
  char nanchar[300];
    
  Trial_ChemP = pTrial_ChemP[3];

  x0 = pTrial_ChemP[0];
  x1 = pTrial_ChemP[1];
  x2 = pTrial_ChemP[2];
  x3 = pTrial_ChemP[3];

  y0  = pDiff_Num[0];
  y1  = pDiff_Num[1];
  y2  = pDiff_Num[2];
  y3  = pDiff_Num[3];
  y31 = pDiff_Num[4];

  /*******************************************************
               store history of stepw and y3
  *******************************************************/

  if (num_loop==2){

    DeltaMu[2] = DeltaMu[1];
    DeltaN[2]  = DeltaN[1];
    DeltaNp[2] = DeltaNp[1];

    DeltaMu[1] = DeltaMu[0];
    DeltaN[1]  = DeltaN[0];
    DeltaNp[1] = DeltaNp[0];
  
    DeltaMu[0] = DeltaMu[6];
    DeltaN[0]  = y31;
    DeltaNp[0] = y3;
  }

  /*******************************************************
                estimate a new chemical potential
  *******************************************************/

  /* num_loop==1 */

  if (num_loop==1){

    y31 = y3; 

    if (SCF_iter<4){

      stepw = fabs(y3);
      if (0.02<stepw) stepw = -sgn(y3)*0.02;
      else            stepw = -(y3 + 10.0*y3*y3);
      Trial_ChemP = Trial_ChemP + stepw;
    }

    else {

      /* if (4<=SCF_iter), estimate Trial_ChemP using an extrapolation. */

      z0 = DeltaNp[0];
      z1 = DeltaNp[1];
      z2 = DeltaNp[2];

      a11 = DeltaN[0]; a12 = DeltaMu[0]; a13 = DeltaN[0]*DeltaMu[0];
      a21 = DeltaN[1]; a22 = DeltaMu[1]; a23 = DeltaN[1]*DeltaMu[1];
      a31 = DeltaN[2]; a32 = DeltaMu[2]; a33 = DeltaN[2]*DeltaMu[2];

      det = a11*a22*a33+a21*a32*a13+a31*a12*a23-a11*a32*a23-a31*a22*a13-a21*a12*a33;

      a1 = ((a22*a33-a23*a32)*z0 + (a13*a32-a12*a33)*z1 + (a12*a23-a13*a22)*z2)/det;
      a2 = ((a23*a31-a21*a33)*z0 + (a11*a33-a13*a31)*z1 + (a13*a21-a11*a23)*z2)/det;
      a3 = ((a21*a32-a22*a31)*z0 + (a12*a31-a11*a32)*z1 + (a11*a22-a12*a21)*z2)/det;

      stepw = -a1*y3/(a2+a3*y3);

      flag_nan = 0;
      sprintf(nanchar,"%8.4f",stepw);

      if (strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	  || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	flag_nan = 1;

	stepw = fabs(y3);
	if (0.02<stepw) stepw = -sgn(y3)*0.02;
	else            stepw = -(y3 + y3*y3);
	Trial_ChemP = Trial_ChemP + stepw;
      }
      else {
	Trial_ChemP = Trial_ChemP + stepw;
      }

    }

    x0 = x3;
    y0 = y3;
  } 

  /* num_loop==2 */

  else if (num_loop==2){

    x1 = x3;
    y1 = y3;

    Trial_ChemP0 = (x1*y0-x0*y1)/(y0-y1);

    if ( 0.1<fabs(Trial_ChemP-Trial_ChemP0) ){
      Trial_ChemP = Trial_ChemP + 0.1*sgn(Trial_ChemP0 - Trial_ChemP);
    }
    else {

      Trial_ChemP = Trial_ChemP0;
    }
  }

  /* num_loop==3 */

  else if (num_loop==3){

    x2 = x3;
    y2 = y3;

    a = y0/(x0-x2)/(x0-x1) - y1/(x1-x2)/(x0-x1) + y2/(x0-x2)/(x1-x2);
    b = -(x2+x1)*y0/(x0-x2)/(x0-x1) + (x0+x2)*y1/(x1-x2)/(x0-x1) - (x1+x0)*y2/(x0-x2)/(x1-x2);
    c = x1*x2*y0/(x0-x2)/(x0-x1) - x2*x0*y1/(x1-x2)/(x0-x1) + x0*x1*y2/(x0-x2)/(x1-x2);

    /* refinement of a, b, and c by the iterative method */

    po4 = 0;
    num4 = 0;
    scaling4 = 0.3;
    dF = 1000.0;

    do {

      g0 = a*x0*x0 + b*x0 + c - y0;
      g1 = a*x1*x1 + b*x1 + c - y1;
      g2 = a*x2*x2 + b*x2 + c - y2;

      dFa = 2.0*(g0*x0*x0 + g1*x1*x1 + g2*x2*x2);   
      dFb = 2.0*(g0*x0 + g1*x1 + g2*x2);
      dFc = 2.0*(g0 + g1 + g2);

      F = g0*g0 + g1*g1 + g2*g2;
      dF0 = dF;
      dF = sqrt(fabs(dFa*dFa + dFb*dFb + dFc*dFc));

      if (dF0<dF) scaling4 /= 1.5;

      /*
        printf("3 F=%18.15f dF=%18.14f scaling4=%15.12f\n",F,dF,scaling4);
      */

      a -= scaling4*dFa;
      b -= scaling4*dFb;
      c -= scaling4*dFc;

      if (dF<1.0e-11) po4 = 1;
      num4++; 

    } while (po4==0 && num4<100);

    /* calculte Trial_ChemP */

    /*
      printf("3 a=%18.15f\n",a); 
      printf("3 b=%18.15f\n",b); 
      printf("3 c=%18.15f\n",c); 
    */

    if (0.0<=b)
      Trial_ChemP0 = -2.0*c/(b+sqrt(fabs(b*b-4*a*c)));
    else 
      Trial_ChemP0 = (-b+sqrt(fabs(b*b-4*a*c)))/(2.0*a); 

    flag_nan = 0;
    sprintf(nanchar,"%8.4f",Trial_ChemP0);

    if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	   || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

      flag_nan = 1;

      stepw = fabs(y3);
      if (0.02<stepw) stepw = -sgn(y3)*0.02;
      else            stepw = -(y3 + y3*y3);
      Trial_ChemP = Trial_ChemP + stepw;
    }
    else {

      if ( 0.02<fabs(Trial_ChemP-Trial_ChemP0) ){
	Trial_ChemP = Trial_ChemP + 0.02*sgn(Trial_ChemP0 - Trial_ChemP);
      }
      else {
	Trial_ChemP = Trial_ChemP0;
      }
    }

    /*
      printf("3 Trial_ChemP0=%18.15f\n",Trial_ChemP0); 
      printf("3 Trial_ChemP =%18.15f\n",Trial_ChemP); 
    */

  }

  /* if (4<=num_loop) */

  else {

    Trial_ChemP0 = Trial_ChemP;

    xx[1] = x0;
    xx[2] = x1;
    xx[3] = x2;
    xx[4] = x3;

    yy[1] = y0;
    yy[2] = y1;
    yy[3] = y2; 
    yy[4] = y3; 

    qsort_double(4, xx, yy);      

    /* check whether the zero point is passed or not. */

    po3 = 0; 
    for (i=1; i<4; i++){
      if (yy[i]*yy[i+1]<0.0){
	i0 = i;
	po3 = 1;
      }
    } 

    /*
      for (i=1; i<=4; i++){
      printf("num_loop=%2d i0=%2d i=%2d xx=%18.15f yy=%18.15f\n",num_loop,i0,i,xx[i],yy[i]);
      }
    */
      
    if (po3==1){

      if (i0==1){
	x0 = xx[i0+0];
	x1 = xx[i0+1];
	x2 = xx[i0+2];
 
	y0 = yy[i0+0];
	y1 = yy[i0+1];
	y2 = yy[i0+2];
      }

      else if (i0==2) {

	if (fabs(yy[i0-1])<fabs(yy[i0+2])){
          
	  x0 = xx[i0-1];
	  x1 = xx[i0+0];
	  x2 = xx[i0+1];
 
	  y0 = yy[i0-1];
	  y1 = yy[i0+0];
	  y2 = yy[i0+1];
	}
	else {
	  x0 = xx[i0+0];
	  x1 = xx[i0+1];
	  x2 = xx[i0+2];
 
	  y0 = yy[i0+0];
	  y1 = yy[i0+1];
	  y2 = yy[i0+2];
	}
      }

      else if (i0==3) {
	x0 = xx[i0-1];
	x1 = xx[i0+0];
	x2 = xx[i0+1];
 
	y0 = yy[i0-1];
	y1 = yy[i0+0];
	y2 = yy[i0+1];
      }

      a = y0/(x0-x2)/(x0-x1) - y1/(x1-x2)/(x0-x1) + y2/(x0-x2)/(x1-x2);
      b = -(x2+x1)*y0/(x0-x2)/(x0-x1) + (x0+x2)*y1/(x1-x2)/(x0-x1) - (x1+x0)*y2/(x0-x2)/(x1-x2);
      c = x1*x2*y0/(x0-x2)/(x0-x1) - x2*x0*y1/(x1-x2)/(x0-x1) + x0*x1*y2/(x0-x2)/(x1-x2);

      /*
	printf("x0=%18.15f y0=%18.15f\n",x0,y0); 
	printf("x1=%18.15f y1=%18.15f\n",x1,y1); 
	printf("x2=%18.15f y2=%18.15f\n",x2,y2); 

	printf("a=%18.15f\n",a); 
	printf("b=%18.15f\n",b); 
	printf("c=%18.15f\n",c); 
      */

      /* refinement of a, b, and c by the iterative method */

      po4 = 0;
      num4 = 0;
      scaling4 = 0.3;
      dF = 1000.0;

      do {

	g0 = a*x0*x0 + b*x0 + c - y0;
	g1 = a*x1*x1 + b*x1 + c - y1;
	g2 = a*x2*x2 + b*x2 + c - y2;

	dFa = 2.0*(g0*x0*x0 + g1*x1*x1 + g2*x2*x2);   
	dFb = 2.0*(g0*x0 + g1*x1 + g2*x2);
	dFc = 2.0*(g0 + g1 + g2);

	F = g0*g0 + g1*g1 + g2*g2;
	dF0 = dF;
	dF = sqrt(fabs(dFa*dFa + dFb*dFb + dFc*dFc));

	if (dF0<dF) scaling4 /= 1.5;

	/*
          printf("F=%18.15f dF=%18.14f scaling4=%15.12f\n",F,dF,scaling4);
	*/

	a -= scaling4*dFa;
	b -= scaling4*dFb;
	c -= scaling4*dFc;

	if (dF<1.0e-11) po4 = 1;
	num4++; 

      } while (po4==0 && num4<100);

      /* calculte Trial_ChemP */

      if (0.0<=b)
	Trial_ChemP0 = -2.0*c/(b+sqrt(fabs(b*b-4*a*c)));
      else 
	Trial_ChemP0 = (-b+sqrt(fabs(b*b-4*a*c)))/(2.0*a); 

      flag_nan = 0;
      sprintf(nanchar,"%8.4f",Trial_ChemP0);

      if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	     || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	flag_nan = 1;

	stepw = fabs(y3);
	if (0.02<stepw) stepw = -sgn(y3)*0.02;
	else            stepw = -(y3 + y3*y3);
	Trial_ChemP = Trial_ChemP + stepw;
      }
      else {

	if ( 0.02<fabs(Trial_ChemP-Trial_ChemP0) ){
	  Trial_ChemP = Trial_ChemP + 0.02*sgn(Trial_ChemP0 - Trial_ChemP);
	}
	else {
	  Trial_ChemP = Trial_ChemP0;
	}
      }

      /*
        printf("F=%18.15f dF=%18.14f Trial_ChemP=%18.15f\n",F,dF,Trial_ChemP);
      */
    }

    else {

      yy[1] = fabs(y0);
      yy[2] = fabs(y1);
      yy[3] = fabs(y2); 
      yy[4] = fabs(y3); 

      zz[1] = 1;
      zz[2] = 2;
      zz[3] = 3;
      zz[4] = 4;

      xx[1] = x0;
      xx[2] = x1;
      xx[3] = x2;
      xx[4] = x3;

      yy0[1] = y0;
      yy0[2] = y1;
      yy0[3] = y2; 
      yy0[4] = y3; 

      qsort_double(4, yy, zz);

      i0 = (int)zz[1];
      i1 = (int)zz[2];
      i2 = (int)zz[3];

      x0 = xx[i0];
      x1 = xx[i1];
      x2 = xx[i2];

      y0 = yy0[i0];
      y1 = yy0[i1];
      y2 = yy0[i2];

      Trial_ChemP0 = (x1*y0-x0*y1)/(y0-y1);

      flag_nan = 0;
      sprintf(nanchar,"%8.4f",Trial_ChemP0);

      if (   strstr(nanchar,"nan")!=NULL || strstr(nanchar,"NaN")!=NULL 
	     || strstr(nanchar,"inf")!=NULL || strstr(nanchar,"Inf")!=NULL){

	flag_nan = 1;

	stepw = fabs(y3);
	if (0.02<stepw) stepw = -sgn(y3)*0.02;
	else            stepw = -(y3 + y3*y3);
	Trial_ChemP = Trial_ChemP + stepw;
      }
      else {

	if ( 0.02<fabs(Trial_ChemP-Trial_ChemP0) ){
	  Trial_ChemP = Trial_ChemP + 0.02*sgn(Trial_ChemP0 - Trial_ChemP);
	}
	else {
	  Trial_ChemP = Trial_ChemP0;
	}
      }

    } /* else */
  } /* else */

  /* store pTrial_ChemP and pDiff_Num */

  pTrial_ChemP[0] = x0;
  pTrial_ChemP[1] = x1;
  pTrial_ChemP[2] = x2;
  pTrial_ChemP[3] = x3;

  pDiff_Num[0] = y0;
  pDiff_Num[1] = y1;
  pDiff_Num[2] = y2;
  pDiff_Num[3] = y3;
  pDiff_Num[4] = y31;

  DeltaMu[6] = stepw;

  /* return Traial_ChemP */
  return Trial_ChemP;
}
