/**********************************************************************
  calB.c

  Code for calculating Berry Curvature and Chern Number.
  11/Sep/2015    Rename polB.c -> calB.c by Hikaru Sawahata
  20/Oct/2015    Modified for MPI        by Hikaru Sawahata
  25/Oct/2015    Modified                by Hikaru Sawahata
  04/Sep/2019    Modified for efficiency and stability by Naoya Yamaguchi
  15/Nov/2019    Modified by Naoya Yamaguchi

  polB.c:

  code for calculating the electric polarization of bulk system using 
  Berry's phase.

  Log of polB.c:

     30/Nov/2005   Released by Taisuke Ozaki
     27/Feb/2006   Modified by Fumiyuki Ishii
     28/July/2006  Modified for MPI by Fumiyuki Ishii
     19/Jan/2007   Modified by Taisuke Ozaki
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h> 
#include "read_scfout.h"
#include "lapack_prototypes.h"
#include "f77func.h"
#include "mpi.h"

#define Host_ID       0         /* ID of the host CPU in MPI */

#define printout  0    /* 0:off, 1:on */
#define PI   3.1415926535897932384626
#define measure_time   0
#define dste_flag      2
#define AU2Debye   2.54174776  
#define AU2Mucm    5721.52891433 /* 100 e/bohr/bohr */
#define Bohr       0.529177 

/* Added by N. Yamaguchi ***/
#define ALLOCATE(TYPE, NAME, SIZE) TYPE *NAME=(TYPE*)malloc(sizeof(TYPE)*SIZE)
#define DEALLOCATE(NAME) free(NAME); NAME=NULL
#define PRINTFF(...)\
  do{\
    printf(__VA_ARGS__);\
    fflush(stdout);\
  }while(0)
/* ***/

struct timeval2 {
  long tv_sec;    /* second */
  long tv_usec;   /* microsecond */
};

/* Added by N. Yamaguchi ***/
dcomplex ****expOLP;
dcomplex *****expOLP_HWF;
static void expOLP_Band(dcomplex ****expOLP,
    dcomplex **T, int *MP,
    double k1, double k2, double k3,
    double dka, double dkb, double dkc, char sign);
static void setexpOLP(double dka, double dkb, double dkc, int calcOrderMax, dcomplex ****expOLP);
static dcomplex ****memoryAllocation_dcomplex(dcomplex ****in);
/* ***/

/* Modified by N. Yamaguchi ***/
static void Overlap_k1k2(int diag_flag,
    double k1[4], double k2[4],
    int spinsize,
    int fsize, int fsize2, int fsize3,
    int calcBandMin, int calcBandMax, int calcOrderMax,
    int *MP,
    dcomplex ***Sop, dcomplex ***Wk1, dcomplex ***Wk2,
    double **EigenVal1, double **EigenVal2);
/* ***/

static void Eigen_HH(dcomplex **ac, double *ko, int n, int EVmax);
static void lapack_dstevx2(INTEGER N, double *D, double *E, double *W, dcomplex **ev);
static void Overlap_Band(double ****OLP,
                         dcomplex **S,int *MP,
                         double k1, double k2, double k3);
static void Hamiltonian_Band(double ****RH, dcomplex **H, int *MP,
                             double k1, double k2, double k3);
static void Hamiltonian_Band_NC(double *****RH, double *****IH,
                                dcomplex **H, int *MP,
                                double k1, double k2, double k3);
static void determinant( int spin, int N, dcomplex **a, INTEGER *ipiv, dcomplex *a1d,
                         dcomplex *work, dcomplex Cdet[2] );

static void dtime(double *t);
static double tmp[4];

void LU_fact(int n, dcomplex **a);
dcomplex RCdiv(double x, dcomplex a);
dcomplex Csub(dcomplex a, dcomplex b);
dcomplex Cmul(dcomplex a, dcomplex b);
dcomplex Cdiv(dcomplex a, dcomplex b);

void Cross_Product(double a[4], double b[4], double c[4]);
double Dot_Product(double a[4], double b[4]);




int main(int argc, char *argv[]) 
{
  int fsize,fsize2,fsize3,fsize4;
  int spin,spinsize,diag_flag;
  int i,j,k,hos,po,wan,hog2,valnonmag;
  int n1,n2,n3,i1,i2,i3;
  int Nk[4],kloop[4][4];
  int pflag[4];
  int hog[1];
  int metal;
  double k1[4],k2[4],k3[4],k4[4];
  double Gabs[4],Parb[4];
  double Phidden[4], Phidden0;
  double tmpr,tmpi;
  double mulr[2],muli[2]; 
  double **kg;
  double sum,d;
  double norm;
  double pele,piony,pall;
  double gdd;
  double CellV;
  double Cell_Volume;
  double psi,sumpsi[2];
  double detr;
  double TZ;
  int ct_AN,h_AN;
  char *s_vec[20];
  int *MP;  
  double **EigenVal1;
  double **EigenVal2;
  double **EigenVal3;
  double **EigenVal4;
  dcomplex ***Sop;
  dcomplex ***Wk1;
  dcomplex ***Wk2;
  dcomplex ***Wk3;
  dcomplex ***Wk4;
  MPI_Comm comm1;
  int numprocs,myid,ID,ID1;
  double TStime,TEtime;

  dcomplex *work1;   /* for determinant */
  dcomplex *work2;   /* for determinant */
  dcomplex Cdet[2];  /* for determinant */
  INTEGER *ipiv;     /* for determinant */
  /* for MPI*/
  int *arpo;
  int AB_knum,S_knum,E_knum,num_ABloop0;
  int AB_knum0;
  int ik1,ik2,ik3,ABloop,ABloop0,abcount;
  double tmp4;
  /* int *AB_Nk2, *AB_Nk3, *ABmesh2ID; */

  FILE *Cfp;

  /* MPI initialize */

  MPI_Status stat;
  MPI_Request request;
  MPI_Init(&argc, &argv);
  comm1 = MPI_COMM_WORLD;
  MPI_Comm_size(comm1,&numprocs);
  MPI_Comm_rank(comm1,&myid);

  dtime(&TStime);
  if(myid==Host_ID){
    printf("\n******************************************************************\n"); 
    printf("******************************************************************\n"); 
    printf(" calB:\n"); 
    printf(" code for calculating the Chern number and Berry curvature of bulk systems\n"); 
    printf(" by Fukui-Hatsugai-Suzuki method.\n"); 
    printf(" Copyright (C), 2019, Hikaru Sawahata, Naoya Yamaguchi,\n"); 
    printf(" Fumiyuki Ishii and Taisuke Ozaki \n"); 
    printf(" This is free software, and you are welcome to         \n"); 
    printf(" redistribute it under the constitution of the GNU-GPL.\n");
    printf("******************************************************************\n"); 
    printf("******************************************************************\n"); 
  } 

  /* ****************************************
   *          scan Mesh,BandNumber 
   * ****************************************/
  int    BandFlag;
  int    mesh1,mesh2;
  if(myid==Host_ID){
    // Input FILE
    // 1.Each Or Sum
    // 2.Band Number
    // 3.Mesh Number
    // 4.Direction
    printf("Each[0] or Sum[1] Curvature:");
    scanf("%d",&BandFlag);
    printf("Band Number[0:Occupy Band]:");
    scanf("%d",&hog2);
    printf("Specify the direction:\n");
    scanf("%d %d %d",&pflag[1],&pflag[2],&pflag[3]);
    printf("Mesh Number:");
    scanf("%d %d",&mesh1, &mesh2);
  } //if Host_id
  MPI_Barrier(comm1);

  MPI_Bcast(&mesh1   ,1,MPI_INT,0,comm1);
  MPI_Bcast(&mesh2   ,1,MPI_INT,0,comm1);
  MPI_Bcast(&hog2    ,1,MPI_INT,0,comm1);
  MPI_Bcast(&BandFlag,1,MPI_INT,0,comm1);
  MPI_Bcast(pflag,    4,MPI_INT,0,comm1);

  double chernNum;
  double chernSum = 0.0;
  double kx,ky; 
  /* double BerryCMAX = 0.0; */
  dcomplex UUUU;
  dcomplex U12,U23,U34,U41;

  /* Disabled by N. Yamaguchi ***
  double BerryC[mesh2][mesh1];
  * ***/

  /* Added by N. Yamaguchi ***/
  double *BerryC;
  if (myid==Host_ID){
    BerryC=(double*)malloc(sizeof(double)*mesh2*mesh1);
  }
  /* ***/

  int Summesh = mesh1 * mesh2 ;


  int ABNumprocs;


  /* Disabled by N. Yamaguchi ***
  int Recvcount[numprocs];
  * ***/

  /* Added by N. Yamaguchi ***/
  ALLOCATE(int, Recvcount, numprocs);
  /* ***/

  int myRecvcount;

  if(myid < Summesh % numprocs){
    ABNumprocs   = Summesh / numprocs + 1;
    myRecvcount = ABNumprocs;
  }
  else{
    ABNumprocs = Summesh / numprocs;
    myRecvcount = ABNumprocs;
  }

  MPI_Gather( &myRecvcount, 1,  MPI_INT,
              Recvcount,    1,  MPI_INT,
              Host_ID, comm1);
  /*  */
  /* if ( myid == Host_ID ){ */
  /*   for ( i = 0; i < numprocs; i++ ) */
  /*     printf("Recvcount[%d] = %d\n", i, Recvcount[i]); */
  /* } */
  /*  */
  MPI_Barrier(comm1);


  /* Disabled by N. Yamaguchi ***
  double BerryCproc[ ABNumprocs ]; 
  * ***/

  /* Added by N. Yamaguchi ***/
  ALLOCATE(double, BerryCproc, ABNumprocs);
  /* ***/

  double Kspace;
  double A12,A23,A34,A41;
  /******************************************
              read a scfout file
  ******************************************/

  read_scfout(argv);

  /* Added by N. Yamaguchi ***/
  if (SpinP_switch!=3){
    PRINTFF("Error: \"calB\" is available only for non-collinear cases.\n");
    MPI_Finalize();
    return 0;
  }
  /* ***/

  /*  */
  /* s_vec[0]="Recursion"; s_vec[1]="Cluster"; s_vec[2]="Band"; */
  /* s_vec[3]="NEGF";      s_vec[4]="DC";      s_vec[5]="GDC"; */
  /* s_vec[6]="Cluster2";  s_vec[7]="Krylov"; */
  /*  */
  /* s_vec[0]="collinear spin-unpolarized"; */
  /* s_vec[1]="collinear spin-polarized"; */
  /* s_vec[3]="non-collinear"; */
  /******************************************
      find the size of the full matrix 
  ******************************************/

  /* MP:
     a pointer which shows the starting number
     of basis orbitals associated with atom i
     in the full matrix */

  MP = (int*)malloc(sizeof(int)*(atomnum+1)); 
  arpo = (int*)malloc(sizeof(int)*numprocs);

  fsize = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = fsize;
    fsize += Total_NumOrbs[i];
  }
  fsize--;


  /******************************************
               allocation arrays
  ******************************************/
  if      (SpinP_switch==0){ spinsize=1; fsize2=fsize;  fsize3=fsize+2; fsize4=Valence_Electrons/2;}
  else if (SpinP_switch==1){ spinsize=2; fsize2=fsize;  fsize3=fsize+2; 
  fsize4=Valence_Electrons/2+abs(floor(Total_SpinS))*2+1;}
  else if (SpinP_switch==3){ spinsize=1; fsize2=2*fsize;fsize3=2*fsize+2; fsize4=Valence_Electrons;}


  /* Sop:
     overlap matrix between one-particle wave functions
     calculated at two k-points, k1 and k2 */
  Sop = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize); 
  for (spin=0; spin<spinsize; spin++){
    Sop[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3); 
    for (i=0; i<fsize3; i++){
      Sop[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
      for (j=0; j<fsize3; j++){ Sop[spin][i][j].r=0.0; Sop[spin][i][j].i=0.0;}
    } 
  }

  /* Wk1:
     wave functions at k1 */

  Wk1 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize); 
  for (spin=0; spin<spinsize; spin++){
    Wk1[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3); 
    for (i=0; i<fsize3; i++){
      Wk1[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
      for (j=0; j<fsize3; j++){ Wk1[spin][i][j].r=0.0; Wk1[spin][i][j].i=0.0;}
    } 
  }

  /* Wk2:
     wave functions at k2 */

  Wk2 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize); 
  for (spin=0; spin<spinsize; spin++){
    Wk2[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3); 
    for (i=0; i<fsize3; i++){
      Wk2[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
      for (j=0; j<fsize3; j++){ Wk2[spin][i][j].r=0.0; Wk2[spin][i][j].i=0.0;}
    } 
  }

  /* Wk3:
     wave functions at k3 */

  Wk3 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize); 
  for (spin=0; spin<spinsize; spin++){
    Wk3[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3); 
    for (i=0; i<fsize3; i++){
      Wk3[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
      for (j=0; j<fsize3; j++){ Wk3[spin][i][j].r=0.0; Wk3[spin][i][j].i=0.0;}
    } 
  }

 /* Wk4:
     wave functions at k4 */

  Wk4 = (dcomplex***)malloc(sizeof(dcomplex**)*spinsize); 
  for (spin=0; spin<spinsize; spin++){
    Wk4[spin] = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3); 
    for (i=0; i<fsize3; i++){
      Wk4[spin][i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3); 
      for (j=0; j<fsize3; j++){ Wk4[spin][i][j].r=0.0; Wk4[spin][i][j].i=0.0;}
    } 
  }

 /* EigenVal1:
     eigenvalues at k1 */

  EigenVal1 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal1[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal1[spin][j] = 0.0;
  }

  /* EigenVal2:
     eigenvalues at k2 */

  EigenVal2 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal2[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal2[spin][j] = 0.0;
  }

  /* EigenVal3:
     eigenvalues at k3 */

  EigenVal3 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal3[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal3[spin][j] = 0.0;
  }

  /* EigenVal4:
   * eigenvalues at k4*/
  EigenVal4 = (double**)malloc(sizeof(double*)*spinsize);
  for (spin=0; spin<spinsize; spin++){
    EigenVal4[spin] = (double*)malloc(sizeof(double)*fsize3);
    for (j=0; j<fsize3; j++) EigenVal4[spin][j] = 0.0;
  }


  /* for determinant */

  ipiv = (INTEGER*)malloc(sizeof(INTEGER)*fsize3);
  work1 = (dcomplex*)malloc(sizeof(dcomplex)*fsize3*fsize3);
  work2 = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);

 
  /* Cross_Product(tv[2],tv[3],tmp); */
  /* CellV = Dot_Product(tv[1],tmp);  */
  /* Cell_Volume = fabs(CellV);  */
  
   for (i=1; i<=3; i++){
    Gabs[i] = sqrt(rtv[i][1]*rtv[i][1]+rtv[i][2]*rtv[i][2]+rtv[i][3]*rtv[i][3]);
  }
  
  kloop[1][1] = 1;
  kloop[1][2] = 2;
  kloop[1][3] = 3;

  kloop[2][1] = 2;
  kloop[2][2] = 3;
  kloop[2][3] = 1;

  kloop[3][1] = 3;
  kloop[3][2] = 1;
  kloop[3][3] = 2;

  for (k=1; k<=3; k++){
    if(pflag[k]==1){
      n1 = kloop[k][1];
      n2 = kloop[k][2];
      n3 = kloop[k][3];
    }
  }

   /* A direction of plane */
  
  // (1,0,0) -> b-c plane
  // (0,1,0) -> c-a plane
  // (0,0,1) -> a-b plane
  
  Cross_Product(rtv[n2],rtv[n3],tmp);
  Kspace = fabs(tmp[n1]);

  Nk[n1] = 1;
  Nk[n2] = mesh1;
  Nk[n3] = mesh2;

  kg = (double**)malloc(sizeof(double*)*4);
  kg[0] = (double*)malloc(sizeof(double)*1);
  for (i=1; i<=3; i++){
    kg[i] = (double*)malloc(sizeof(double)*(Nk[i]+1));
  }
  
  /* make  k-point */
  
  for (i=1; i<=3; i++){
    d = 1.0/(double)Nk[i];
    for (j=0; j<=Nk[i]; j++){
      kg[i][j] = d*(double)j;
    }
  } 
  double dx,dy;

  dx = 1.0 / (double)mesh1;
  dy = 1.0 / (double)mesh2;
 
  chernNum   = 0.0;
  if(hog2 <= 0){
    hog2 = Valence_Electrons;
  }

  /* Added by N. Yamaguchi ***/

  int direction;
  expOLP_HWF=(dcomplex*****)malloc(sizeof(dcomplex****)*4);
  for (direction=0; direction<4; direction++){
    expOLP_HWF[direction]=memoryAllocation_dcomplex(expOLP_HWF[direction]);
  }
  if (n1==3){
    setexpOLP(1.0/Nk[1], 0.0, 0.0, 1, expOLP_HWF[0]);
    setexpOLP(0.0, 1.0/Nk[2], 0.0, 1, expOLP_HWF[1]);
    setexpOLP(-1.0/Nk[1], 0.0, 0.0, 1, expOLP_HWF[2]);
    setexpOLP(0.0, -1.0/Nk[2], 0.0, 1, expOLP_HWF[3]);
  } else if (n1==2){
    setexpOLP(0.0, 0.0, 1.0/Nk[3], 1, expOLP_HWF[0]);
    setexpOLP(1.0/Nk[1], 0.0, 0.0, 1, expOLP_HWF[1]);
    setexpOLP(0.0, 0.0, -1.0/Nk[3], 1, expOLP_HWF[2]);
    setexpOLP(-1.0/Nk[1], 0.0, 0.0, 1, expOLP_HWF[3]);
  } else {
    setexpOLP(0.0, 1.0/Nk[2], 0.0, 1, expOLP_HWF[0]);
    setexpOLP(0.0, 0.0, 1.0/Nk[3], 1, expOLP_HWF[1]);
    setexpOLP(0.0, -1.0/Nk[2], 0.0, 1, expOLP_HWF[2]);
    setexpOLP(0.0, 0.0, -1.0/Nk[3], 1, expOLP_HWF[3]);
  }
  /** **/

  for(i3 = 0;i3 < Nk[n1];i3++){
    /* for(i2=0;i2<Nk[n3];i2++){ */
      /* for(i1=0;i1<Nk[n2];i1++){ */
    //
    // Parallelization
    ABloop = 0;
    /* AB_knum0 = ceil( myid * (double)Summesh / numprocs ); */

     if( Summesh % numprocs == 0){
        AB_knum0 = myid * ABNumprocs;
      }
      else if( myid < Summesh % numprocs ){
        AB_knum0 = myid * ABNumprocs;
      }
      else if( myid == Summesh % numprocs ){
        AB_knum0 = ( ABNumprocs + 1 ) * myid ;
      }
      else{
        AB_knum0 = Summesh % numprocs + myid * ABNumprocs;
      }
      MPI_Barrier(comm1);


  for  ( AB_knum  =  AB_knum0; 
         AB_knum  < AB_knum0 + ABNumprocs; 
         AB_knum++){
    //Parallelization
      i1 =  AB_knum       % mesh1 ;
      i2 = (AB_knum - i1) / mesh1 ;

    // make k-point k1,k2,k3,k4
    // kg[x,y,z][meshNumber]
        k1[n2] = kg[n2][i1];
        k1[n3] = kg[n3][i2];
        k1[n1] = kg[n1][i3];
    //
        k2[n2] = kg[n2][i1+1];
        k2[n3] = kg[n3][i2];
        k2[n1] = kg[n1][i3];
    //
        k3[n2] = kg[n2][i1+1];
        k3[n3] = kg[n3][i2+1];
        k3[n1] = kg[n1][i3];
    //
        k4[n2] = kg[n2][i1];
        k4[n3] = kg[n3][i2+1];
        k4[n1] = kg[n1][i3];
	    /* calculate the overlap matrix */
      /* calculate UUUU */
      // <u1|u2>  -> diag_flag = 0 diagonalize u1,u2
      // <u2|u3>  -> diag_flag = 2 diagonalize only u3
      // <u3|u4>  -> diag_flag = 2 diagonalize only u4
      // <u4|u1>  -> diag_flag = 3 no diagonalize u4,u1
          //calculate <u1|u2>
          //diagonalize u1,u2
          diag_flag = 0;

	  /* Modified by N. Yamaguchi ***/
	  expOLP=expOLP_HWF[0];
          Overlap_k1k2( diag_flag, k1, k2, spinsize,
              fsize, fsize2, fsize3, 1, fsize4, 1,
              MP, Sop, Wk1, Wk2, EigenVal1, EigenVal2 );
	  /* ***/

          for (spin=0; spin<spinsize; spin++){
            if(BandFlag == 1){
              determinant(spin,hog2,Sop[spin],ipiv,work1,work2,Cdet);
            }
            else{
              Cdet[spin] = Sop[spin][hog2][hog2];
            }
	    /* norm  = sqrt( Cdet[spin].r*Cdet[spin].r
	       + Cdet[spin].i*Cdet[spin].i ); */
	    U12.r = Cdet[spin].r;
	    U12.i = Cdet[spin].i;
          } /* spin */
          /* A12 = atan2(U12.i,U12.r); */
          //calculate <u2|u3>
          //diagonalize only u3
          diag_flag = 2;

	  /* Modified by N. Yamaguchi ***/
	  expOLP=expOLP_HWF[1];
          Overlap_k1k2( diag_flag, k2, k3, spinsize,
              fsize, fsize2, fsize3, 1, fsize4, 1,
              MP, Sop, Wk2, Wk3, EigenVal2, EigenVal3);
	  /* ***/

          for (spin=0; spin<spinsize; spin++){
            if(BandFlag == 1){
              determinant(spin,hog2,Sop[spin],ipiv,work1,work2,Cdet);
            }
            else{
              Cdet[spin] = Sop[spin][hog2][hog2];
            }
	    /* norm  = sqrt( Cdet[spin].r*Cdet[spin].r
	       + Cdet[spin].i*Cdet[spin].i ); */
	    U23.r = Cdet[spin].r;
	    U23.i = Cdet[spin].i;
          } /* spin */
          /* A23 = atan2(U23.i,U23.r); */
         //calculate <u3|u4>
         //diagonalize only u4
          diag_flag = 2;

	  /* Modified by N. Yamaguchi ***/
	  expOLP=expOLP_HWF[2];
          Overlap_k1k2( diag_flag, k3, k4,
              spinsize, fsize, fsize2, fsize3, 1, fsize4, 1,
              MP, Sop, Wk3, Wk4, EigenVal3, EigenVal4 );
	  /* ***/

          for (spin=0; spin<spinsize; spin++){
            if(BandFlag == 1){
              determinant(spin,hog2,Sop[spin],ipiv,work1,work2,Cdet);
            }
            else{
              Cdet[spin] = Sop[spin][hog2][hog2];
            }
	    /* norm  = sqrt( Cdet[spin].r*Cdet[spin].r
	       + Cdet[spin].i*Cdet[spin].i ); */
	    U34.r = Cdet[spin].r;
	    U34.i = Cdet[spin].i;
          } /* spin */
          /* A34 = atan2(U34.i,U34.r); */
          //calculate <u4|u1>
          //no diagonalize
          diag_flag = 3;

	  /* Modified by N. Yamaguchi ***/
	  expOLP=expOLP_HWF[3];
          Overlap_k1k2( diag_flag, k4, k1,
              spinsize, fsize, fsize2, fsize3, 1, fsize4, 1,
              MP, Sop, Wk4, Wk1, EigenVal4, EigenVal1 );
	  /* ***/

          for (spin=0; spin<spinsize; spin++){
            if(BandFlag == 1){
              determinant(spin,hog2,Sop[spin],ipiv,work1,work2,Cdet);
            }
            else{
              Cdet[spin] = Sop[spin][hog2][hog2];
            }
	    /* norm  = sqrt( Cdet[spin].r*Cdet[spin].r
	       + Cdet[spin].i*Cdet[spin].i ); */
	    U41.r = Cdet[spin].r;
	    U41.i = Cdet[spin].i;
          } /* spin */
          /* A41 = atan2(U41.i,U41.r); */
          //calculate F = Im ln(UUUU)
          UUUU                = Cmul(Cmul(U12,U23),Cmul(U34,U41));
          BerryCproc[ABloop]  = atan2(UUUU.i,UUUU.r);
          chernNum           += BerryCproc[ABloop];
          printf("<BerryFlux> %f %f %f\n"
              ,(double)i1 * dx,(double)i2 * dy,BerryCproc[ABloop]);
          ABloop++;
    } /* AB_knum */
  }   /* i3   */

 
  
  MPI_Barrier(comm1);
   
  //Sum chernNum,Gather BerryCurvature

  MPI_Reduce( &chernNum ,&chernSum      ,    1,
              MPI_DOUBLE,MPI_SUM,Host_ID,comm1);
  
  MPI_Barrier(comm1);



  /* Disabled by N. Yamaguchi ***
  int displs[numprocs];
  * ***/

  /* Added by N. Yamaguchi ***/
  ALLOCATE(int, displs, numprocs);
  /* ***/

  int Rem, Remcount = 0;
  displs[0] = 0;
  Rem = Summesh % numprocs;
  for ( i = 1; i <= numprocs - 1; i++ ){
    if( Remcount < Rem ){
      displs[i] = displs[i-1] + (int) Summesh / numprocs + 1;
      Remcount++;
    }
    else{
      displs[i] = displs[i-1] + (int) Summesh / numprocs;
    }
  }


  MPI_Gatherv( 
      BerryCproc, ABNumprocs, MPI_DOUBLE,
      BerryC,     Recvcount, displs,
      MPI_DOUBLE, Host_ID, comm1);

  /* Added by N. Yamaguchi ***/
  DEALLOCATE(BerryCproc);
  DEALLOCATE(Recvcount);
  DEALLOCATE(displs);
  /* ***/

  /*************************************************/
  /* Output Results                                */
  /*************************************************/
  MPI_Barrier(comm1);  

  if(myid == Host_ID){
    char CdataFILE[30];
    double kx,ky;
    sprintf(CdataFILE, "BerryC%d.dat",hog2);
    Cfp = fopen(CdataFILE,"w");
    fprintf(Cfp,"#Mesh Number:%d*%d\n",mesh1,mesh2);fflush(stdout);
    fprintf(Cfp,"#Band Number:%d\n",hog2);
    fprintf(Cfp,"#ChernNumber = %f\n",0.5*chernSum/PI);fflush(stdout);
    fprintf(Cfp,"#k1          k2           F (ang-2)\n");fflush(stdout);
    for(j = 0; j < mesh2; j++){
      for(i = 0; i < mesh1; i++){
        kx = (double)i*dx+dx*0.5;
        ky = (double)j*dx+dx*0.5;

	/* Disabled by N. Yamaguchi ***
          fprintf(Cfp,"%f %f %10.8e\n"
              ,kx,ky,BerryC[j][i]* Summesh / Kspace 
              * Bohr * Bohr);
	  * ***/

	/* Added by N. Yamaguchi ***/
          fprintf(Cfp,"%f %f %10.8e\n", kx, ky, BerryC[j*mesh1+i]*Summesh/Kspace*Bohr*Bohr);
	/* ***/

	fflush(stdout);
      }
      fprintf(Cfp,"\n");
          fflush(stdout);
    }
    printf("ChernNumber = %f\n",0.5 * chernSum / PI);fflush(stdout);
    fclose(Cfp);

    /* Added by N. Yamaguchi ***/
    free(BerryC);
    /* ***/

  }
  /******************************************
                  free arrays
  ******************************************/

  /* Added by N. Yamaguchi ***/
  for (direction=0; direction<4; direction++){
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      int TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
	      for (i=0; i<TNO1; i++){
	        free(expOLP_HWF[direction][ct_AN][h_AN][i]);
	      }
        free(expOLP_HWF[direction][ct_AN][h_AN]);
      }
      free(expOLP_HWF[direction][ct_AN]);
    }
    free(expOLP_HWF[direction]);
  }
  free(expOLP_HWF);
  expOLP_HWF=NULL;
  /** **/

  free(MP);

  for (spin=0; spin<spinsize; spin++){
    for (i=0; i<fsize3; i++){
      free(Sop[spin][i]);
    } 
    free(Sop[spin]);
  }
  free(Sop);

  for (spin=0; spin<spinsize; spin++){
    for (i=0; i<fsize3; i++){
      free(Wk1[spin][i]);
    } 
    free(Wk1[spin]);
  }
  free(Wk1);

  for (spin=0; spin<spinsize; spin++){
    for (i=0; i<fsize3; i++){
      free(Wk2[spin][i]);
    } 
    free(Wk2[spin]);
  }
  free(Wk2);

  for (spin=0; spin<spinsize; spin++){
    for (i=0; i<fsize3; i++){
      free(Wk3[spin][i]);
    } 
    free(Wk3[spin]);
  }
  free(Wk3);

  for (spin=0; spin<spinsize; spin++){
    for (i=0; i<fsize3; i++){
      free(Wk4[spin][i]);
    } 
    free(Wk4[spin]);
  }
  free(Wk4);


  for (spin=0; spin<spinsize; spin++){
    free(EigenVal1[spin]);
  }
  free(EigenVal1);

  for (spin=0; spin<spinsize; spin++){
    free(EigenVal2[spin]);
  }
  free(EigenVal2);

  for (spin=0; spin<spinsize; spin++){
    free(EigenVal3[spin]);
  }
  free(EigenVal3);

  for (spin=0; spin<spinsize; spin++){
    free(EigenVal4[spin]);
  }
  free(EigenVal4);



  free(ipiv);
  free(work1);
  free(work2);

  for (i=0; i<=3; i++){
    free(kg[i]);
  }
  free(kg);
 
  /* print message */

  MPI_Barrier(comm1); 

  dtime(&TEtime); 

  printf(" \nElapsed time = %lf (s) for myid=%3d\n",TEtime-TStime,myid);fflush(stdout);
  /*  */
  MPI_Barrier(comm1); 
  
  printf(" \nThe calculation was finished normally in myid=%2d.\n",myid);fflush(stdout);
  /*  */

  
  
  /* MPI_Finalize */

  MPI_Finalize();

  /* return */

  return 0;
}














void Overlap_Band(double ****OLP,
                  dcomplex **S,int *MP,
                  double k1, double k2, double k3)
{
  static int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  static int l1,l2,l3,Rn,n2;
  static double **S1,**S2;
  static double kRn,si,co,s;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
                       Allocation
  ****************************************************/

  n2 = NUM + 2;

  S1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    S1[i] = (double*)malloc(sizeof(double)*n2);
  }

  S2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    S2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                       set overlap
  ****************************************************/

  S[0][0].r = NUM;

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S1[i][j] = 0.0;
      S2[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  s = OLP[GA_AN][LB_AN][i][j];
	  S1[Anum+i][Bnum+j] += s*co;
	  S2[Anum+i][Bnum+j] += s*si;
	}
      }
    }
  }

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      S[i][j].r =  S1[i][j];
      S[i][j].i =  S2[i][j];
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(S1[i]);
    free(S2[i]);
  }
  free(S1);
  free(S2);

}





void Hamiltonian_Band(double ****RH, dcomplex **H, int *MP,
                      double k1, double k2, double k3)
{
  static int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  static int l1,l2,l3,Rn,n2;
  static double **H1,**H2;
  static double kRn,si,co,h;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
                       Allocation
  ****************************************************/

  n2 = NUM + 2;

  H1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H1[i] = (double*)malloc(sizeof(double)*n2);
  }

  H2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                    set Hamiltonian
  ****************************************************/

  H[0][0].r = 2.0*NUM;
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H1[i][j] = 0.0;
      H2[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];
      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){
	  h = RH[GA_AN][LB_AN][i][j];
	  H1[Anum+i][Bnum+j] += h*co;
	  H2[Anum+i][Bnum+j] += h*si;
	}
      }
    }
  }

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H[i][j].r = H1[i][j];
      H[i][j].i = H2[i][j];
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(H1[i]);
    free(H2[i]);
  }
  free(H1);
  free(H2);
}

















void Hamiltonian_Band_NC(double *****RH, double *****IH,
                         dcomplex **H, int *MP,
                         double k1, double k2, double k3)
{
  static int i,j,k,wanA,wanB,tnoA,tnoB,Anum,Bnum;
  static int NUM,GA_AN,LB_AN,GB_AN;
  static int l1,l2,l3,Rn,n2;
  static double **H11r,**H11i;
  static double **H22r,**H22i;
  static double **H12r,**H12i;
  static double kRn,si,co,h;

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;
  n2 = NUM + 2;

  /*******************************************
   allocation of H11r, H11i,
                 H22r, H22i,
                 H12r, H12i
  *******************************************/

  H11r = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H11r[i] = (double*)malloc(sizeof(double)*n2);
  }

  H11i = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H11i[i] = (double*)malloc(sizeof(double)*n2);
  }

  H22r = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H22r[i] = (double*)malloc(sizeof(double)*n2);
  }

  H22i = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H22i[i] = (double*)malloc(sizeof(double)*n2);
  }

  H12r = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H12r[i] = (double*)malloc(sizeof(double)*n2);
  }

  H12i = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    H12i[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
                    set Hamiltonian
  ****************************************************/

  H[0][0].r = 2.0*NUM;
  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H11r[i][j] = 0.0;
      H11i[i][j] = 0.0;
      H22r[i][j] = 0.0;
      H22i[i][j] = 0.0;
      H12r[i][j] = 0.0;
      H12i[i][j] = 0.0;
    }
  }

  for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
    tnoA = Total_NumOrbs[GA_AN];
    Anum = MP[GA_AN];

    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      Rn = ncn[GA_AN][LB_AN];
      tnoB = Total_NumOrbs[GB_AN];

      l1 = atv_ijk[Rn][1];
      l2 = atv_ijk[Rn][2];
      l3 = atv_ijk[Rn][3];
      kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

      si = sin(2.0*PI*kRn);
      co = cos(2.0*PI*kRn);
      Bnum = MP[GB_AN];

      for (i=0; i<tnoA; i++){
	for (j=0; j<tnoB; j++){

	  H11r[Anum+i][Bnum+j] += co*RH[0][GA_AN][LB_AN][i][j] -  si*IH[0][GA_AN][LB_AN][i][j];
	  H11i[Anum+i][Bnum+j] += si*RH[0][GA_AN][LB_AN][i][j] +  co*IH[0][GA_AN][LB_AN][i][j];
	  H22r[Anum+i][Bnum+j] += co*RH[1][GA_AN][LB_AN][i][j] -  si*IH[1][GA_AN][LB_AN][i][j];
	  H22i[Anum+i][Bnum+j] += si*RH[1][GA_AN][LB_AN][i][j] +  co*IH[1][GA_AN][LB_AN][i][j];
	  H12r[Anum+i][Bnum+j] += co*RH[2][GA_AN][LB_AN][i][j] - si*(RH[3][GA_AN][LB_AN][i][j]
								     + IH[2][GA_AN][LB_AN][i][j]);
	  H12i[Anum+i][Bnum+j] += si*RH[2][GA_AN][LB_AN][i][j] + co*(RH[3][GA_AN][LB_AN][i][j]
								     + IH[2][GA_AN][LB_AN][i][j]);
        }
      }

    }
  }

  /******************************************************
    the full complex matrix of H
  ******************************************************/

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      H[i    ][j    ].r =  H11r[i][j];
      H[i    ][j    ].i =  H11i[i][j];
      H[i+NUM][j+NUM].r =  H22r[i][j];
      H[i+NUM][j+NUM].i =  H22i[i][j];
      H[i    ][j+NUM].r =  H12r[i][j];
      H[i    ][j+NUM].i =  H12i[i][j];
      H[j+NUM][i    ].r =  H[i][j+NUM].r;
      H[j+NUM][i    ].i = -H[i][j+NUM].i;
    }
  }

  /****************************************************
                       free arrays
  ****************************************************/

  for (i=0; i<n2; i++){
    free(H11r[i]);
  }
  free(H11r);

  for (i=0; i<n2; i++){
    free(H11i[i]);
  }
  free(H11i);

  for (i=0; i<n2; i++){
    free(H22r[i]);
  }
  free(H22r);

  for (i=0; i<n2; i++){
    free(H22i[i]);
  }
  free(H22i);

  for (i=0; i<n2; i++){
    free(H12r[i]);
  }
  free(H12r);

  for (i=0; i<n2; i++){
    free(H12i[i]);
  }
  free(H12i);

}



#pragma optimization_level 1
void Eigen_HH(dcomplex **ac, double *ko, int n, int EVmax)
{
  /**********************************************************************
    Eigen_HH:

    Eigen_HH.c is a subroutine to solve a standard eigenvalue problem
    with a Hermite complex matrix using Householder method and lapack's
    F77_NAME(dstegr,DSTEGR)() or dstedc_().

    Log of Eigen_HH.c:

       Dec/07/2004  Released by T.Ozaki

  ***********************************************************************/

  static double ABSTOL=1.0e-13;

  static dcomplex **ad,*u,*b1,*p,*q,tmp1,tmp2,u1,u2,p1,ss;
  static double *D,*E,*uu,*alphar,*alphai,
    s1,s2,s3,r,
    sum,ar,ai,br,bi,e,
    a1,a2,a3,a4,a5,a6,b7,r1,r2,
    r3,x1,x2,xap,
    bb,bb1,ui,uj,uij;

  static int jj,jj1,jj2,k,ii,ll,i3,i2,j2,
    i,j,i1,j1,n1,n2,ik,
    jk,po1,nn,count;

  static double Stime, Etime;
  static double Stime1, Etime1;
  static double Stime2, Etime2;
  static double time1,time2;

  /****************************************************
    allocation of arrays:
  ****************************************************/

  n2 = n + 5;

  ad = (dcomplex**)malloc(sizeof(dcomplex*)*n2);
  for (i=0; i<n2; i++){
    ad[i] = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  }

  b1 = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  u = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  uu = (double*)malloc(sizeof(double)*n2);
  p = (dcomplex*)malloc(sizeof(dcomplex)*n2);
  q = (dcomplex*)malloc(sizeof(dcomplex)*n2);

  D = (double*)malloc(sizeof(double)*n2);
  E = (double*)malloc(sizeof(double)*n2);

  alphar = (double*)malloc(sizeof(double)*n2);
  alphai = (double*)malloc(sizeof(double)*n2);

  for (i=1; i<=(n+2); i++){
    uu[i] = 0.0;
  }

  if (measure_time==1) printf("size n=%3d EVmax=%2d\n",n,EVmax);
  if (measure_time==1) dtime(&Stime);

  /****************************************************
               Householder transformation
  ****************************************************/

  for (i=1; i<=(n-1); i++){

    s1 = ac[i+1][i].r * ac[i+1][i].r + ac[i+1][i].i * ac[i+1][i].i;
    s2 = 0.0;

    u[i+1].r = ac[i+1][i].r;
    u[i+1].i = ac[i+1][i].i;
 
    for (i1=i+2; i1<=n; i1++){

      tmp1.r = ac[i1][i].r; 
      tmp1.i = ac[i1][i].i; 

      s2 += tmp1.r*tmp1.r + tmp1.i*tmp1.i;

      u[i1].r = tmp1.r;
      u[i1].i = tmp1.i;
    }

    s3 = fabs(s1 + s2);

    if ( ABSTOL<(fabs(ac[i+1][i].r)+fabs(ac[i+1][i].i)) ){
      if (ac[i+1][i].r<0.0)  s3 =  sqrt(s3);
      else                   s3 = -sqrt(s3);
    }
    else{
      s3 = sqrt(s3);
    }

    if ( ABSTOL<fabs(s2) || i==(n-1) ){

      ss.r = ac[i+1][i].r;
      ss.i = ac[i+1][i].i;

      ac[i+1][i].r = s3;
      ac[i+1][i].i = 0.0;
      ac[i][i+1].r = s3;
      ac[i][i+1].i = 0.0;

      u[i+1].r = u[i+1].r - s3;
      u[i+1].i = u[i+1].i;
      
      u1.r = s3 * s3 - ss.r * s3;
      u1.i =         - ss.i * s3;
      u2.r = 2.0 * u1.r;
      u2.i = 2.0 * u1.i;
      
      e = u2.r/(u1.r*u1.r + u1.i*u1.i);
      ar = e*u1.r;
      ai = e*u1.i;

      /* store alpha */
      alphar[i] = ar;
      alphai[i] = ai;

      /* store u2 */
      uu[i] = u2.r;

      /* store the first component of u */
      b1[i].r = ss.r - s3;
      b1[i].i = ss.i;

      r = 0.0;
      for (i1=i+1; i1<=n; i1++){

	p1.r = 0.0;
	p1.i = 0.0;
	for (j=i+1; j<=n; j++){
	  p1.r += ac[i1][j].r * u[j].r - ac[i1][j].i * u[j].i;
	  p1.i += ac[i1][j].r * u[j].i + ac[i1][j].i * u[j].r;
	}
	p[i1].r = p1.r / u1.r;
	p[i1].i = p1.i / u1.r;

	r += u[i1].r * p[i1].r + u[i1].i * p[i1].i;
      }

      r = 0.5*r / u2.r;

      br =  ar*r;
      bi = -ai*r;

      for (i1=i+1; i1<=n; i1++){
	tmp1.r = 0.5*(p[i1].r - (br * u[i1].r - bi*u[i1].i));
	tmp1.i = 0.5*(p[i1].i - (br * u[i1].i + bi*u[i1].r));
	q[i1].r = ar * tmp1.r - ai * tmp1.i; 
	q[i1].i = ar * tmp1.i + ai * tmp1.r; 
      }

      for (i1=i+1; i1<=n; i1++){
        tmp1.r = u[i1].r;
        tmp1.i = u[i1].i;
        tmp2.r = q[i1].r; 
        tmp2.i = q[i1].i; 
	for (j1=i+1; j1<=n; j1++){
	  ac[i1][j1].r -= ( tmp1.r * q[j1].r + tmp1.i * q[j1].i
                           +tmp2.r * u[j1].r + tmp2.i * u[j1].i );
	  ac[i1][j1].i -= (-tmp1.r * q[j1].i + tmp1.i * q[j1].r
                           -tmp2.r * u[j1].i + tmp2.i * u[j1].r );
	}
      }
    }
  }

  for (i=1; i<=n; i++){
    for (j=1; j<=n; j++){
      ad[i][j].r = ac[i][j].r;
      ad[i][j].i = ac[i][j].i;
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T1   %15.12f\n",Etime-Stime);
  }

  /****************************************************
                     call a lapack routine
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  for (i=1; i<=n; i++){
    D[i-1] = ad[i][i  ].r;
    E[i-1] = ad[i][i+1].r;
  }

  /*
  if      (dste_flag==0) lapack_dstegr2(n,D,E,ko,ac);
  else if (dste_flag==1) lapack_dstedc2(n,D,E,ko,ac);
  else if (dste_flag==2) lapack_dstevx2(n,D,E,ko,ac);
  */

  lapack_dstevx2(n,D,E,ko,ac);

  if (measure_time==1){
    dtime(&Etime);
    printf("T2   %15.12f\n",Etime-Stime);
  }

  /****************************************************
    transformation of eigenvectors to original space
  ****************************************************/

  if (measure_time==1) dtime(&Stime);

  /* ad stores u */
  for (i=2; i<=n; i++){
    ad[i-1][i].r = b1[i-1].r;
    ad[i-1][i].i =-b1[i-1].i;
    ad[i][i-1].r = b1[i-1].r;
    ad[i][i-1].i = b1[i-1].i;
  }
  
  for (k=1; k<=EVmax; k++){
  
    for (nn=1; nn<=n-1; nn++){

      if ( (1.0e-3*ABSTOL)<fabs(uu[n-nn])){

	tmp1.r = 0.0;
	tmp1.i = 0.0;

	for (i=n-nn+1; i<=n; i++){
	  tmp1.r += ad[n-nn][i].r * ac[k][i].r - ad[n-nn][i].i * ac[k][i].i;
	  tmp1.i += ad[n-nn][i].i * ac[k][i].r + ad[n-nn][i].r * ac[k][i].i;
	}

	ss.r = (alphar[n-nn]*tmp1.r - alphai[n-nn]*tmp1.i) / uu[n-nn];
	ss.i = (alphar[n-nn]*tmp1.i + alphai[n-nn]*tmp1.r) / uu[n-nn];

	for (i=n-nn+1; i<=n; i++){
	  ac[k][i].r -= ss.r * ad[n-nn][i].r + ss.i * ad[n-nn][i].i;
	  ac[k][i].i -=-ss.r * ad[n-nn][i].i + ss.i * ad[n-nn][i].r;
	}
      }
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T4   %15.12f\n",Etime-Stime);
  }

  /****************************************************
                     normalization
  ****************************************************/

  if (measure_time==1) dtime(&Stime);
  
  for (j=1; j<=EVmax; j++){
    sum = 0.0;
    for (i=1; i<=n; i++){
      sum += ac[j][i].r * ac[j][i].r + ac[j][i].i * ac[j][i].i;
    }
    sum = 1.0/sqrt(sum);
    for (i=1; i<=n; i++){
      ac[j][i].r = ac[j][i].r * sum;
      ac[j][i].i = ac[j][i].i * sum;
    }
  }

  if (measure_time==1){
    dtime(&Etime);
    printf("T5   %15.12f\n",Etime-Stime);
  }

  /****************************************************
                     transpose ac
  ****************************************************/

  for (i=1; i<=n; i++){
    for (j=(i+1); j<=n; j++){
      tmp1 = ac[i][j];
      tmp2 = ac[j][i];
      ac[i][j] = tmp2;
      ac[j][i] = tmp1;
    }
  }

  /****************************************************
                  freeing of arrays:
  ****************************************************/

  for (i=0; i<n2; i++){
    free(ad[i]);
  }
  free(ad);

  free(b1);
  free(u);
  free(uu);
  free(p);
  free(q);
  free(D);
  free(E);
  free(alphar);
  free(alphai);
}



void lapack_dstevx2(INTEGER N, double *D, double *E, double *W, dcomplex **ev)
{
  static int i,j;

  char  *JOBZ="V";
  char  *RANGE="A";

  double VL,VU; /* dummy */
  INTEGER IL,IU; 
  double ABSTOL=1.0e-12;
  INTEGER M;
  double *Z;
  INTEGER LDZ;
  double *WORK;
  INTEGER *IWORK;
  INTEGER *IFAIL;
  INTEGER INFO;

  M = N;
  LDZ = N;

  Z = (double*)malloc(sizeof(double)*LDZ*N);
  WORK = (double*)malloc(sizeof(double)*5*N);
  IWORK = (INTEGER*)malloc(sizeof(INTEGER)*5*N);
  IFAIL = (INTEGER*)malloc(sizeof(INTEGER)*N);

  F77_NAME(dstevx,DSTEVX)( JOBZ, RANGE, &N,  D, E, &VL, &VU, &IL, &IU, &ABSTOL,
           &M, W, Z, &LDZ, WORK, IWORK, IFAIL, &INFO );

  /* store eigenvectors */

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      ev[i+1][j+1].r = Z[i*N+j];
      ev[i+1][j+1].i = 0.0;
    }
  }

  /* shift ko by 1 */
  for (i=N; i>=1; i--){
    W[i]= W[i-1];
  }

  if (INFO>0) {
    printf("\n error in dstevx_, info=%d\n\n",INFO);
  }
  if (INFO<0) {
    printf("info=%d in dstevx_\n",INFO);
    exit(0);
  }

  free(Z);
  free(WORK);
  free(IWORK);
  free(IFAIL);
}



void determinant( int spin, int N, dcomplex **a, INTEGER *ipiv, dcomplex *a1d,
                  dcomplex *work, dcomplex Cdet[2] )
{
  /********************************************************************
   void determinant( int spin, int N, dcomplex **a, INTEGER *ipiv,
                     dcomplex *a1d,
                     dcomplex *work, dcomplex Cdet[2] )
  
   a routine for calculating the determinant of overlap matrix
   for occupied states at k1- and k2-points.

   int spin (input variable) 
   
     collinear spin-unpolarized:    spin = 0
     collinear spin-polarized:      spin = 0 or 1
     non-collinear spin-polarized:  spin = 0
 
   int N (input variable)

     the number of occupied states

   dcomplex **a (input variable)

     overlap matrix whose size is [fize3][fsize3].

   INTEGER *ipiv (work space)

     work space for a lapack routine, zgetrf, 
     whose size is fsize3. 

   dcomplex *a1d (work space)

     work space for a lapack routine, zgetrf,
     whose size is fsize3*fsize3.

   dcomplex *work (work space)

     work space for a lapack routine, zgetrf, 
     whose size is fsize3. 

   dcomplex Cdet[2] (output variable)   

     the determinant of the matrix 'a' for occupied states
     at k1- and k2-points. Cdet[0] and Cdet[1] correspond
     to those with spin index 0 and 1.
  ********************************************************************/

  static int i,j;
  static INTEGER lda,info,lwork;
  dcomplex Ctmp;

  lda = N;
  lwork = N;

  /****************************************************
      a -> a1d
  ****************************************************/

  for (i=0;i<N;i++) {
    for (j=0;j<N;j++) {
      a1d[j*N+i] = a[i+1][j+1];
    }
  }

  /****************************************************
                call zgetrf_() in clapack
  ****************************************************/

  F77_NAME(zgetrf,ZGETRF)(&N, &N, a1d, &lda, ipiv, &info);
  if (info!=0){
    printf("ERROR in zgetrf_(), info=%2d\n",info);
  }

  /*
  for (i=0;i<=N;i++) {
    ipiv[i] = i + 1; 
  }
  LU_fact(N,a);

  for (i=0; i<N; i++){
    a1d[i*N+i] = a[i+1][i+1];
  }
  */

  /****************************************************
               a1d -> a
  ****************************************************/

  /*
  printf("Re \n");
  for (i=0;i<N;i++) {
    for (j=0;j<N;j++) {
      printf("%15.12f ",a1d[j*N+i].r);
    }
    printf("\n");
  }

  printf("Im\n");
  for (i=0;i<N;i++) {
    for (j=0;j<N;j++) {
      printf("%15.12f ",a1d[j*N+i].i);
    }
    printf("\n");
  }
  */

  /*
  for (i=0;i<=N;i++) {
    printf("i=%2d ipiv[i]=%3d\n",i,ipiv[i]); 
  }
  */

  Cdet[spin].r = 1.0;
  Cdet[spin].i = 0.0;
  for (i=0; i<N; i++){

    Ctmp = Cdet[spin];
    if ( ipiv[i]!=(i+1) ){ 
      Ctmp.r = -Ctmp.r;
      Ctmp.i = -Ctmp.i;
    } 

    Cdet[spin].r = Ctmp.r*a1d[i*N+i].r - Ctmp.i*a1d[i*N+i].i;
    Cdet[spin].i = Ctmp.r*a1d[i*N+i].i + Ctmp.i*a1d[i*N+i].r;
  }
}





void dtime(double *t)
{
  /* real time */
  struct timeval timev;
  gettimeofday(&timev, NULL);
  *t = timev.tv_sec + (double)timev.tv_usec*1e-6;

  /* user time + system time */
  /*
  float tarray[2];
  clock_t times(), wall;
  struct tms tbuf;
  wall = times(&tbuf);
  tarray[0] = (float) (tbuf.tms_utime / (float)CLOCKS_PER_SEC);
  tarray[1] = (float) (tbuf.tms_stime / (float)CLOCKS_PER_SEC);
  *t = (double) (tarray[0]+tarray[1]);
  printf("dtime: %lf\n",*t);
  */
}







void LU_fact(int n, dcomplex **a)
{
  /*    LU factorization */
  int i,j,k;
  dcomplex w,sum;

  for (k=1; k<=n-1; k++){
    w = RCdiv(1.0,a[k][k]);
    for (i=k+1; i<=n; i++){
      a[i][k] = Cmul(w,a[i][k]);
      for (j=k+1; j<=n; j++){
	a[i][j] = Csub(a[i][j], Cmul(a[i][k],a[k][j]));
      }
    }
  }

}




dcomplex RCdiv(double x, dcomplex a)
{
  dcomplex c;
  double xx,yy,w;
  xx = a.r;
  yy = a.i;
  w = xx*xx+yy*yy;
  c.r = x*a.r/w;
  c.i = -x*a.i/w;
  return c;
}


dcomplex Csub(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r - b.r;
  c.i = a.i - b.i;
  return c;
}

dcomplex Cmul(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = a.r*b.r - a.i*b.i;
  c.i = a.i*b.r + a.r*b.i;
  return c;
}
dcomplex Cdiv(dcomplex a, dcomplex b)
{
  dcomplex c;
  c.r = (a.r*b.r+a.i*b.i)/(b.r*b.r+b.i*b.i);
  c.i = (a.i*b.r-a.r*b.i)/(b.r*b.r+b.i*b.i);
  return c;
}

void Cross_Product(double a[4], double b[4], double c[4])
{
  c[1] = a[2]*b[3] - a[3]*b[2]; 
  c[2] = a[3]*b[1] - a[1]*b[3]; 
  c[3] = a[1]*b[2] - a[2]*b[1];
}

double Dot_Product(double a[4], double b[4])
{
  static double sum;
  sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
  return sum;
}

/* Added by N. Yamaguchi ***/
static void expOLP_Band(dcomplex ****expOLP,
    dcomplex **T, int *MP,
    double k1, double k2, double k3,
    double dka, double dkb, double dkc, char sign)
{
  int i,j,wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM,GA_AN,LB_AN,GB_AN;
  int l1,l2,l3,Rn,n2;
  double **T1,**T2;
  double kRn,si,co;
  dcomplex t;
  double si2, co2;

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    Anum += Total_NumOrbs[i];
  }
  NUM = Anum - 1;

  /****************************************************
    Allocation
   ****************************************************/

  n2 = NUM + 2;

  T1 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    T1[i] = (double*)malloc(sizeof(double)*n2);
  }

  T2 = (double**)malloc(sizeof(double*)*n2);
  for (i=0; i<n2; i++){
    T2[i] = (double*)malloc(sizeof(double)*n2);
  }

  /****************************************************
    set T
   ****************************************************/

  T[0][0].r = NUM;

  for (i=1; i<=NUM; i++){
    for (j=1; j<=NUM; j++){
      T1[i][j] = 0.0;
      T2[i][j] = 0.0;
    }
  }

  if (sign=='-'){
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      tnoA = Total_NumOrbs[GA_AN];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	tnoB = Total_NumOrbs[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);
	Bnum = MP[GB_AN];
	kRn-=dka*l1+dkb*l2+dkc*l3;
	si2 = sin(2.0*PI*kRn);
	co2 = cos(2.0*PI*kRn);
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    t=expOLP[GA_AN][LB_AN][i][j];
	    T1[Anum+i][Bnum+j]+=t.r*co-t.i*si;
	    T2[Anum+i][Bnum+j]+=t.i*co+t.r*si;
	    T1[Bnum+j][Anum+i]+=t.r*co2+t.i*si2;
	    T2[Bnum+j][Anum+i]+=t.i*co2-t.r*si2;
	  }
	}
      }
    }
  } else if (sign=='+'){
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      tnoA = Total_NumOrbs[GA_AN];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	tnoB = Total_NumOrbs[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);
	Bnum = MP[GB_AN];
	kRn+=dka*l1+dkb*l2+dkc*l3;
	si2 = sin(2.0*PI*kRn);
	co2 = cos(2.0*PI*kRn);
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    t=expOLP[GA_AN][LB_AN][i][j];
	    T1[Anum+i][Bnum+j]+=t.r*co+t.i*si;
	    T2[Anum+i][Bnum+j]-=t.i*co-t.r*si;
	    T1[Bnum+j][Anum+i]+=t.r*co2-t.i*si2;
	    T2[Bnum+j][Anum+i]-=t.i*co2+t.r*si2;
	  }
	}
      }
    }
  } else if (sign=='M'){
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      tnoA = Total_NumOrbs[GA_AN];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	tnoB = Total_NumOrbs[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);
	Bnum = MP[GB_AN];
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    t=expOLP[GA_AN][LB_AN][i][j];
	    T1[Anum+i][Bnum+j]+=t.r*co-t.i*si;
	    T2[Anum+i][Bnum+j]+=t.i*co+t.r*si;
	  }
	}
      }
    }
  } else if (sign=='P'){
    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
      tnoA = Total_NumOrbs[GA_AN];
      Anum = MP[GA_AN];

      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	Rn = ncn[GA_AN][LB_AN];
	tnoB = Total_NumOrbs[GB_AN];

	l1 = atv_ijk[Rn][1];
	l2 = atv_ijk[Rn][2];
	l3 = atv_ijk[Rn][3];
	kRn = k1*(double)l1 + k2*(double)l2 + k3*(double)l3;

	si = sin(2.0*PI*kRn);
	co = cos(2.0*PI*kRn);
	Bnum = MP[GB_AN];
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){
	    t=expOLP[GA_AN][LB_AN][i][j];
	    T1[Anum+i][Bnum+j]+=t.r*co+t.i*si;
	    T2[Anum+i][Bnum+j]+=-t.i*co+t.r*si;
	  }
	}
      }
    }
  }

  if (sign=='+' || sign=='-'){
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
	T[i][j].r =  0.5*T1[i][j];
	T[i][j].i =  0.5*T2[i][j];
      }
    }
  } else {
    for (i=1; i<=NUM; i++){
      for (j=1; j<=NUM; j++){
	T[i][j].r =  T1[i][j];
	T[i][j].i =  T2[i][j];
      }
    }
  }

  /****************************************************
    free arrays
   ****************************************************/

  for (i=0; i<n2; i++){
    free(T1[i]);
    free(T2[i]);
  }
  free(T1);
  free(T2);

}


static void Overlap_k1k2(int diag_flag,
    double k1[4], double k2[4],
    int spinsize,
    int fsize, int fsize2, int fsize3,
    int calcBandMin, int calcBandMax, int calcOrderMax,
    int *MP,
    dcomplex ***Sop, dcomplex ***Wk1, dcomplex ***Wk2,
    double **EigenVal1, double **EigenVal2)
{
  int spin;
  int ik,i1,j1,i,j,l,k;
  int ct_AN,h_AN,mu1,mu2;
  int Anum,Bnum,tnoA,tnoB;
  int mn,jj1,ii1,m;
  int Rnh,Gh_AN,l1,l2,l3;
  int recalc[2];
  double kpoints[2][4];
  double *ko,*M1;
  double sumr,sumi;
  double tmp1r,tmp1i;
  double tmp2r,tmp2i;
  double tmp3r,tmp3i;
  double si,co,kRn,k1da,k1db,k1dc;
  double si1,co1,si2,co2;
  double tmp2r1,tmp2i1;
  double tmp2r2,tmp2i2;
  double dx,dy,dz;
  double dkx,dky,dkz;
  double dka,dkb,dkc;
  double k1x,k1y,k1z;
  dcomplex **S,**H,**C;
  double OLP_eigen_cut = 1.0e-12;
  dcomplex Ctmp1,Ctmp2;

  /****************************************************
    allocation of arrays:
   ****************************************************/

  ko = (double*)malloc(sizeof(double)*fsize3);
  M1 = (double*)malloc(sizeof(double)*fsize3);

  S = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
  for (i=0; i<fsize3; i++){
    S[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
    for (j=0; j<fsize3; j++){ S[i][j].r = 0.0; S[i][j].i = 0.0; }
  }

  H = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
  for (i=0; i<fsize3; i++){
    H[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
    for (j=0; j<fsize3; j++){ H[i][j].r = 0.0; H[i][j].i = 0.0; }
  }

  C = (dcomplex**)malloc(sizeof(dcomplex*)*fsize3);
  for (i=0; i<fsize3; i++){
    C[i] = (dcomplex*)malloc(sizeof(dcomplex)*fsize3);
    for (j=0; j<fsize3; j++){ C[i][j].r = 0.0; C[i][j].i = 0.0; }
  }

  /* set kpoints */

  kpoints[0][1] = k1[1];
  kpoints[0][2] = k1[2];
  kpoints[0][3] = k1[3];

  kpoints[1][1] = k2[1];
  kpoints[1][2] = k2[2];
  kpoints[1][3] = k2[3];

  k1x = k1[1]*rtv[1][1] + k1[2]*rtv[2][1] + k1[3]*rtv[3][1];
  k1y = k1[1]*rtv[1][2] + k1[2]*rtv[2][2] + k1[3]*rtv[3][2];
  k1z = k1[1]*rtv[1][3] + k1[2]*rtv[2][3] + k1[3]*rtv[3][3];

  dka = k2[1] - k1[1];
  dkb = k2[2] - k1[2];
  dkc = k2[3] - k1[3];

  dkx = dka*rtv[1][1] + dkb*rtv[2][1] + dkc*rtv[3][1];
  dky = dka*rtv[1][2] + dkb*rtv[2][2] + dkc*rtv[3][2];
  dkz = dka*rtv[1][3] + dkb*rtv[2][3] + dkc*rtv[3][3];

  if      (diag_flag==0) {recalc[0]=1; recalc[1]=1; }
  else if (diag_flag==1) {recalc[0]=1; recalc[1]=0; }
  else if (diag_flag==2) {recalc[0]=0; recalc[1]=1; }
  else if (diag_flag==3) {recalc[0]=0; recalc[1]=0; }

  /****************************************************
    diagonalize Bloch matrix at k-points, k1 and k2
   ****************************************************/

  for (ik=0; ik<2; ik++){

    if (recalc[ik]){

      Overlap_Band(OLP,S,MP,kpoints[ik][1],kpoints[ik][2],kpoints[ik][3]);

      Eigen_HH(S,ko,fsize,fsize);

      for (l=1; l<=fsize; l++){
	if (ko[l]<OLP_eigen_cut){
	  printf("found an overcomplete basis set\n");

	  /* Modified by N. Yamaguchi ***/
	  MPI_Abort(MPI_COMM_WORLD, 1);
	  /* ***/

	}
      }

      for (l=1; l<=fsize; l++) M1[l] = 1.0/sqrt(ko[l]);

      /* S * M1  */

      for (i1=1; i1<=fsize; i1++){
	for (j1=1; j1<=fsize; j1++){
	  S[i1][j1].r = S[i1][j1].r*M1[j1];
	  S[i1][j1].i = S[i1][j1].i*M1[j1];
	}
      }

      for (spin=0; spin<spinsize; spin++){

	/* transpose S */

	for (i1=1; i1<=fsize; i1++){
	  for (j1=i1+1; j1<=fsize; j1++){
	    Ctmp1 = S[i1][j1];
	    Ctmp2 = S[j1][i1];
	    S[i1][j1] = Ctmp2;
	    S[j1][i1] = Ctmp1;
	  }
	}

	/****************************************************
	  collinear case
	 ****************************************************/

	if (SpinP_switch==0 || SpinP_switch==1){

	    Hamiltonian_Band(Hks[spin],H,MP,kpoints[ik][1],kpoints[ik][2],kpoints[ik][3]);

	  /****************************************************
	    M1 * U^t * H * U * M1
	   ****************************************************/

	  /* H * U * M1 */

	  for (j1=1; j1<=fsize; j1++){
	    for (i1=1; i1<=fsize; i1++){

	      sumr = 0.0;
	      sumi = 0.0;

	      for (l=1; l<=fsize; l++){
		sumr += H[i1][l].r*S[j1][l].r - H[i1][l].i*S[j1][l].i;
		sumi += H[i1][l].r*S[j1][l].i + H[i1][l].i*S[j1][l].r;
	      }

	      C[j1][i1].r = sumr;
	      C[j1][i1].i = sumi;
	    }
	  }

	  /* M1 * U^+ H * U * M1 */

	  for (i1=1; i1<=fsize; i1++){
	    for (j1=1; j1<=fsize; j1++){
	      sumr = 0.0;
	      sumi = 0.0;
	      for (l=1; l<=fsize; l++){
		sumr +=  S[i1][l].r*C[j1][l].r + S[i1][l].i*C[j1][l].i;
		sumi +=  S[i1][l].r*C[j1][l].i - S[i1][l].i*C[j1][l].r;
	      }
	      H[i1][j1].r = sumr;
	      H[i1][j1].i = sumi;
	    }
	  }

	  /* H to C */

	  for (i1=1; i1<=fsize; i1++){
	    for (j1=1; j1<=fsize; j1++){
	      C[i1][j1] = H[i1][j1];
	    }
	  }

	  /* solve eigenvalue problem */

	  if (ik==0){
	    Eigen_HH(C,EigenVal1[spin],fsize,fsize);
	  }
	  else if (ik==1){
	    Eigen_HH(C,EigenVal2[spin],fsize,fsize);
	  }

	  /****************************************************
	    transformation to the original eigenvectors.
	    NOTE JRCAT-244p and JAIST-2122p
	   ****************************************************/

	  /* transpose */
	  for (i1=1; i1<=fsize; i1++){
	    for (j1=i1+1; j1<=fsize; j1++){
	      Ctmp1 = S[i1][j1];
	      Ctmp2 = S[j1][i1];
	      S[i1][j1] = Ctmp2;
	      S[j1][i1] = Ctmp1;
	    }
	  }

	  /* transpose */
	  for (i1=1; i1<=fsize; i1++){
	    for (j1=i1+1; j1<=fsize; j1++){
	      Ctmp1 = C[i1][j1];
	      Ctmp2 = C[j1][i1];
	      C[i1][j1] = Ctmp2;
	      C[j1][i1] = Ctmp1;
	    }
	  }

	  /* calculate wave functions */

	  for (i1=1; i1<=fsize; i1++){
	    for (j1=1; j1<=fsize; j1++){
	      sumr = 0.0;
	      sumi = 0.0;
	      for (l=1; l<=fsize; l++){
		sumr +=  S[i1][l].r*C[j1][l].r - S[i1][l].i*C[j1][l].i;
		sumi +=  S[i1][l].r*C[j1][l].i + S[i1][l].i*C[j1][l].r;
	      }

	      if (ik==0){
		Wk1[spin][j1][i1].r = sumr;
		Wk1[spin][j1][i1].i = sumi;
	      }
	      else if (ik==1){
		Wk2[spin][j1][i1].r = sumr;
		Wk2[spin][j1][i1].i = sumi;
	      }
	    }
	  }
	}

	/****************************************************
	  non-collinear case
	 ****************************************************/

	else if (SpinP_switch==3){

	    Hamiltonian_Band_NC(Hks,iHks,H,MP,kpoints[ik][1],kpoints[ik][2],kpoints[ik][3]);

	  /* H * U * M1 */

	  for (j1=1; j1<=fsize; j1++){
	    for (i1=1; i1<=fsize2; i1++){

	      /* Modified by N. Yamaguchi ***/
	      sumr=0.0;
	      sumi=0.0;
	      for (l=1; l<=fsize; l++){
		sumr+=H[i1][l].r*S[j1][l].r-H[i1][l].i*S[j1][l].i;
		sumi+=H[i1][l].r*S[j1][l].i+H[i1][l].i*S[j1][l].r;
	      }
	      C[2*j1-1][i1].r=sumr;
	      C[2*j1-1][i1].i=sumi;
	      sumr=0.0;
	      sumi=0.0;
	      for (l=1; l<=fsize; l++){
		sumr+=H[i1][l+fsize].r*S[j1][l].r-H[i1][l+fsize].i*S[j1][l].i;
		sumi+=H[i1][l+fsize].r*S[j1][l].i+H[i1][l+fsize].i*S[j1][l].r;
	      }
	      C[2*j1][i1].r=sumr;
	      C[2*j1][i1].i=sumi;
	      /* ***/

	    }
	  }

	  /* M1 * U^+ H * U * M1 */

	  for (i1=1; i1<=fsize; i1++){

	    /* Modified by N. Yamaguchi ***/
	    for (j1=1; j1<=fsize2; j1++){
	      sumr=0.0;
	      sumi=0.0;
	      for (l=1; l<=fsize; l++){
		sumr+=S[i1][l].r*C[j1][l].r+S[i1][l].i*C[j1][l].i;
		sumi+=S[i1][l].r*C[j1][l].i-S[i1][l].i*C[j1][l].r;
	      }
	      H[2*i1-1][j1].r=sumr;
	      H[2*i1-1][j1].i=sumi;
	    }
	    for (j1=1; j1<=fsize2; j1++){
	      sumr=0.0;
	      sumi=0.0;
	      for (l=1; l<=fsize; l++){
		sumr+=S[i1][l].r*C[j1][l+fsize].r+S[i1][l].i*C[j1][l+fsize].i;
		sumi+=S[i1][l].r*C[j1][l+fsize].i-S[i1][l].i*C[j1][l+fsize].r;
	      }
	      H[2*i1][j1].r=sumr;
	      H[2*i1][j1].i=sumi;
	    }
	    /* ***/

	  }

	  /* solve eigenvalue problem */

	  if (ik==0){
	    Eigen_HH(H,EigenVal1[0],fsize2,fsize2);
	  }
	  else if (ik==1){
	    Eigen_HH(H,EigenVal2[0],fsize2,fsize2);
	  }

	  /****************************************************
	    transformation to the original eigenvectors
	    NOTE JRCAT-244p and JAIST-2122p
	    C = U * lambda^{-1/2} * D
	   ****************************************************/

	  /* transpose */

	  for (i1=1; i1<=fsize; i1++){
	    for (j1=i1+1; j1<=fsize; j1++){
	      Ctmp1 = S[i1][j1];
	      Ctmp2 = S[j1][i1];
	      S[i1][j1] = Ctmp2;
	      S[j1][i1] = Ctmp1;
	    }
	  }

	  for (i1=1; i1<=fsize2; i1++){
	    for (j1=1; j1<=fsize2; j1++){
	      C[i1][j1].r = 0.0;
	      C[i1][j1].i = 0.0;
	    }
	  }

	  /* Modified by N. Yamaguchi ***/
	  for (i1=1; i1<=fsize; i1++){
	    for (j1=1; j1<=fsize2; j1++){
	      sumr=0.0;
	      sumi=0.0;
	      for (l=1; l<=fsize; l++){
		sumr+=S[i1][l].r*H[2*l-1][j1].r-S[i1][l].i*H[2*l-1][j1].i;
		sumi+=S[i1][l].r*H[2*l-1][j1].i+S[i1][l].i*H[2*l-1][j1].r;
	      }
	      if (ik==0){
		Wk1[0][j1][i1].r=sumr;
		Wk1[0][j1][i1].i=sumi;
	      }
	      else if (ik==1){
		Wk2[0][j1][i1].r=sumr;
		Wk2[0][j1][i1].i=sumi;
	      }
	      sumr=0.0;
	      sumi=0.0;
	      for (l=1; l<=fsize; l++){
		sumr+=S[i1][l].r*H[2*l][j1].r-S[i1][l].i*H[2*l][j1].i;
		sumi+=S[i1][l].r*H[2*l][j1].i+S[i1][l].i*H[2*l][j1].r;
	      }
	      if (ik==0){
		Wk1[0][j1][i1+fsize].r=sumr;
		Wk1[0][j1][i1+fsize].i=sumi;
	      }
	      else if (ik==1){
		Wk2[0][j1][i1+fsize].r=sumr;
		Wk2[0][j1][i1+fsize].i=sumi;
	      }
	    }
	  }
	  /* ***/

	}

      } /* spin */
    } /* if (recalc[ik]) */
  } /* ik */

  /****************************************************
    calculate an overlap matrix between one-particle
    wave functions  calculated at two k-points, k1 and k2
   ****************************************************/

  /* Modified by N. Yamaguchi ***/
  expOLP_Band(expOLP, H, MP, k2[1], k2[2], k2[3], dka, dkb, dkc, '-');
  if (SpinP_switch==3){
    for (mu1=calcBandMin; mu1<=calcBandMax; mu1++){
      for (j1=1; j1<=fsize; j1++){
	sumr=0.0;
	sumi=0.0;
	for (l=1; l<=fsize; l++){
	  sumr+=Wk1[0][mu1][l].r*H[l][j1].r+Wk1[0][mu1][l].i*H[l][j1].i;
	  sumi+=Wk1[0][mu1][l].r*H[l][j1].i-Wk1[0][mu1][l].i*H[l][j1].r;
	}
	C[mu1][j1].r=sumr;
	C[mu1][j1].i=sumi;
      }
    }
    for (mu1=calcBandMin; mu1<=calcBandMax; mu1++){
      for (mu2=calcBandMin; mu2<=calcBandMax; mu2++){
	sumr=0.0;
	sumi=0.0;
	for (l=1; l<=fsize; l++){
	  sumr+=C[mu1][l].r*Wk2[0][mu2][l].r-C[mu1][l].i*Wk2[0][mu2][l].i;
	  sumi+=C[mu1][l].i*Wk2[0][mu2][l].r+C[mu1][l].r*Wk2[0][mu2][l].i;
	}
	Sop[0][mu1][mu2].r=sumr;
	Sop[0][mu1][mu2].i=sumi;
      }
    }
    for (mu1=calcBandMin; mu1<=calcBandMax; mu1++){
      for (j1=1; j1<=fsize; j1++){
	sumr=0.0;
	sumi=0.0;
	for (l=1; l<=fsize; l++){
	  sumr+=Wk1[0][mu1][l+fsize].r*H[l][j1].r+Wk1[0][mu1][l+fsize].i*H[l][j1].i;
	  sumi+=Wk1[0][mu1][l+fsize].r*H[l][j1].i-Wk1[0][mu1][l+fsize].i*H[l][j1].r;
	}
	C[mu1][j1].r=sumr;
	C[mu1][j1].i=sumi;
      }
    }
    for (mu1=calcBandMin; mu1<=calcBandMax; mu1++){
      for (mu2=calcBandMin; mu2<=calcBandMax; mu2++){
	sumr=0.0;
	sumi=0.0;
	for (l=1; l<=fsize; l++){
	  sumr+=C[mu1][l].r*Wk2[0][mu2][l+fsize].r-C[mu1][l].i*Wk2[0][mu2][l+fsize].i;
	  sumi+=C[mu1][l].i*Wk2[0][mu2][l+fsize].r+C[mu1][l].r*Wk2[0][mu2][l+fsize].i;
	}
	Sop[0][mu1][mu2].r+=sumr;
	Sop[0][mu1][mu2].i+=sumi;
      }
    }
  } else {
    for (spin=0; spin<spinsize; spin++){
      for (mu1=calcBandMin; mu1<=calcBandMax; mu1++){
	for (j1=1; j1<=fsize; j1++){
	  sumr=0.0;
	  sumi=0.0;
	  for (l=1; l<=fsize; l++){
	    sumr+=Wk1[spin][mu1][l].r*H[l][j1].r+Wk1[spin][mu1][l].i*H[l][j1].i;
	    sumi+=Wk1[spin][mu1][l].r*H[l][j1].i-Wk1[spin][mu1][l].i*H[l][j1].r;
	  }
	  C[mu1][j1].r=sumr;
	  C[mu1][j1].i=sumi;
	}
      }
      for (mu1=calcBandMin; mu1<=calcBandMax; mu1++){
	for (mu2=calcBandMin; mu2<=calcBandMax; mu2++){
	  sumr=0.0;
	  sumi=0.0;
	  for (l=1; l<=fsize; l++){
	    sumr+=C[mu1][l].r*Wk2[spin][mu2][l].r-C[mu1][l].i*Wk2[spin][mu2][l].i;
	    sumi+=C[mu1][l].i*Wk2[spin][mu2][l].r+C[mu1][l].r*Wk2[spin][mu2][l].i;
	  }
	  Sop[spin][mu1][mu2].r = sumr;
	  Sop[spin][mu1][mu2].i = sumi;
	}
      }
    }
  }
  /* ***/

  /****************************************************
    deallocation of arrays:
    (Modified by N. Yamaguchi about this comment)
   ****************************************************/

  free(ko);
  free(M1);

  for (i=0; i<fsize3; i++){
    free(S[i]);
  }
  free(S);

  for (i=0; i<fsize3; i++){
    free(H[i]);
  }
  free(H);

  for (i=0; i<fsize3; i++){
    free(C[i]);
  }
  free(C);

}
static void setexpOLP(double dka, double dkb, double dkc, int calcOrderMax, dcomplex ****expOLP)
{
  double dkx = dka*rtv[1][1] + dkb*rtv[2][1] + dkc*rtv[3][1];
  double dky = dka*rtv[1][2] + dkb*rtv[2][2] + dkc*rtv[3][2];
  double dkz = dka*rtv[1][3] + dkb*rtv[2][3] + dkc*rtv[3][3];
#ifdef HWF
  double ******OLPpo=OLPpo_HWF;
#endif
  int ct_AN;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    double phase=-(dkx*Gxyz[ct_AN][1]+dky*Gxyz[ct_AN][2]+dkz*Gxyz[ct_AN][3]);
    double co=cos(phase);
    double si=sin(phase);
    int tnoA = Total_NumOrbs[ct_AN];
    int h_AN;
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      int i1;
      for (i1=0; i1<tnoA; i1++){
	int Gh_AN = natn[ct_AN][h_AN];
	int tnoB = Total_NumOrbs[Gh_AN];
	int j1;
	for (j1=0; j1<tnoB; j1++){
	  double tmp3r, tmp3i;
	  int order, fac=1, sw=-1;
	  double dkxn=1, dkyn=1, dkzn=1;
	  for (order=0; order<=calcOrderMax; order++){
	    if (sw==-1){
	      tmp3r=OLP[ct_AN][h_AN][i1][j1];
	      tmp3i=0;
	      fac*=-1;
	      sw=1;
	      continue;
	    }
	    dkxn*=dkx;
	    dkyn*=dky;
	    dkzn*=dkz;
	    fac*=order;
	    if (sw==1){
	      tmp3i+=
		+dkxn/fac*OLPpo[0][order-1][ct_AN][h_AN][i1][j1]
		+dkyn/fac*OLPpo[1][order-1][ct_AN][h_AN][i1][j1]
		+dkzn/fac*OLPpo[2][order-1][ct_AN][h_AN][i1][j1];
	      sw=0;
	    } else {
	      tmp3r+=
		+dkxn/fac*OLPpo[0][order-1][ct_AN][h_AN][i1][j1]
		+dkyn/fac*OLPpo[1][order-1][ct_AN][h_AN][i1][j1]
		+dkzn/fac*OLPpo[2][order-1][ct_AN][h_AN][i1][j1];
	      fac*=-1;
	      sw=1;
	    }
	  }
	  expOLP[ct_AN][h_AN][i1][j1].r=co*tmp3r-si*tmp3i;
	  expOLP[ct_AN][h_AN][i1][j1].i=co*tmp3i+si*tmp3r;
	}
      }
    }
  }
}
static dcomplex ****memoryAllocation_dcomplex(dcomplex ****in)
{
  in=(dcomplex****)malloc(sizeof(dcomplex***)*(atomnum+1));
  int ct_AN;
  in[0]=NULL;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    int TNO1=Total_NumOrbs[ct_AN];
    in[ct_AN]=(dcomplex***)malloc(sizeof(dcomplex**)*(FNAN[ct_AN]+1));
    int h_AN;
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      in[ct_AN][h_AN]=(dcomplex**)malloc(sizeof(dcomplex*)*TNO1);
      int TNO2;
      int Gh_AN=natn[ct_AN][h_AN];
      TNO2=Total_NumOrbs[Gh_AN];
      int i;
      for (i=0; i<TNO1; i++){
	in[ct_AN][h_AN][i]=(dcomplex*)malloc(sizeof(dcomplex)*TNO2);
      }
    }
  }
  return in;
}
/* ***/
