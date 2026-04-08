/**********************************************************************
  Cluster_DFT_Dosout.c:

     Cluster_DFT_Dosout.c is a subroutine to set up density of states
     based on cluster calculations

  Log of Cluster_DFT_Dosout.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"



static double Cluster_Col_Dosout( int SpinP_switch, 
                                  double *****nh, double ****CntOLP);

static double Cluster_NonCol_Dosout(
                              int SpinP_switch, 
                              double *****nh,
                              double *****ImNL,
                              double ****CntOLP);


static void CurrentOpt_Cluster( 
                          int n,
                          double **H,
                          double **Ovlp, 
                          double **Sinv, 
                          double ***J );

double Cluster_DFT_Dosout( int SpinP_switch,
                           double *****nh,
                           double *****ImNL,
                           double ****CntOLP)
{
  static double time0;

  /****************************************************
         collinear without spin-orbit coupling
  ****************************************************/

  if ( (SpinP_switch==0 || SpinP_switch==1) && SO_switch==0 ){
    time0 = Cluster_Col_Dosout(SpinP_switch, nh, CntOLP);
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
    time0 = Cluster_NonCol_Dosout(SpinP_switch, nh, ImNL, CntOLP);
  }

  return time0;
}


double Cluster_Col_Dosout( int SpinP_switch,  double *****nh, double ****CntOLP)
{
  int n,i,j,l,wanA,n2;
  int *MP;
  int n1min,iemin,iemax,spin,i1,j1,iemin0,iemax0,n1,ie;
  int MA_AN, GA_AN, tnoA, Anum, LB_AN, GB_AN,wanB, tnoB, Bnum, k;
  int MaxL;
  int i_vec[10];  
  int file_ptr_size;

  double EV_cut0;
  double **ko; int N_ko, i_ko[10];
  double ***H; int N_H, i_H[10];
  double ***C; int N_C, i_C[10];
  double **Ctmp; int N_Ctmp, i_Ctmp[10];
  double ****CDM; int N_CDM=4,i_CDM[10];
  double **SD; int N_SD=2, i_SD[10];
  double TStime,TEtime,time0;

  /* optical conductivity */
  double **Ovlp; int N_Ovlp, i_Ovlp[10];
  double **Sinv; int N_Sinv, i_Sinv[10];
  double ***J;  int N_J, i_J[10];
  FILE *fp_opt;
  char file_opt[YOUSO10];

  double sum,dum;
  float *fSD; 

  char buf1[fp_bsize];          /* setvbuf */
  char buf2[fp_bsize];          /* setvbuf */
  char buf3[fp_bsize];          /* setvbuf */
  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  int numprocs,myid,ID,tag;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID  && 0<level_stdout) {
    printf("Cluster_DFT_Dosout: start\n"); fflush(stdout);
  }

  dtime(&TStime);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = n + 2;

  /****************************************************
   Allocation

   int     MP[List_YOUSO[1]]
   double  ko[List_YOUSO[23]][n2]
   double  H[List_YOUSO[23]][n2][n2]
   double  C[List_YOUSO[23]][n2][n2]
   double  Ctmp[n2][n2]
   double  CDM[Matomnum+1][List_YOUSO[8]]
              [List_YOUSO[7]][List_YOUSO[7]]
   double  SD[List_YOUSO[1]][List_YOUSO[7]]
   float   fSD[List_YOUSO[7]]
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);

  N_ko=2; i_ko[0]=List_YOUSO[23]; i_ko[1]=n2;
  ko=(double**)malloc_multidimarray("double",N_ko,i_ko);

  N_H=3; i_H[0]=List_YOUSO[23]; i_H[1]=n2; i_H[2]=n2;
  H=(double***)malloc_multidimarray("double",N_H, i_H);

  N_C=3; i_C[0]=List_YOUSO[23]; i_C[1]=n2; i_C[2]=n2;
  C=(double***)malloc_multidimarray("double",N_C, i_C);

  N_Ctmp=2; i_Ctmp[0]=n2; i_Ctmp[1]=n2;
  Ctmp=(double**)malloc_multidimarray("double",N_Ctmp, i_Ctmp);

  N_CDM=4; i_CDM[0]=Matomnum+1; i_CDM[1]=List_YOUSO[8];
  i_CDM[2]=List_YOUSO[7]; i_CDM[3]=List_YOUSO[7];
  CDM =(double****)malloc_multidimarray("double",N_CDM,i_CDM);

  N_SD=2; i_SD[0]=List_YOUSO[1]; i_SD[1]=List_YOUSO[7];
  SD = (double**)malloc_multidimarray("double",N_SD, i_SD);

  fSD=(float*)malloc(sizeof(float)*List_YOUSO[7]);

  if (myid==Host_ID){
    strcpy(file_eig,".Dos.val");
    fnjoint(filepath,filename,file_eig);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {

#ifdef xt3
      setvbuf(fp_eig,buf1,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open a file %s\n",file_eig);
    }

    strcpy(file_ev,".Dos.vec");
    fnjoint(filepath,filename,file_ev);
    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {

#ifdef xt3
      setvbuf(fp_ev,buf2,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open a file %s\n",file_ev);
    }

  }

  file_ptr_size=sizeof(FILE *);

  MPI_Bcast(&fp_eig,file_ptr_size,MPI_BYTE,Host_ID,mpi_comm_level1);
  MPI_Bcast(&fp_ev,file_ptr_size,MPI_BYTE,Host_ID,mpi_comm_level1);


#if 0
  /*debug*/
  MPI_Barrier(mpi_comm_level1);
  printf("%d: fp_opt=%d\n",myid,fp_opt);
  MPI_Barrier(mpi_comm_level1);
#endif

  if ( fp_eig==NULL || fp_ev==NULL  ) {
      goto Finishing;
    }


  if (myid==Host_ID){

    fprintf(fp_eig,"mode        1\n");
    fprintf(fp_eig,"NonCol      0\n");
    fprintf(fp_eig,"N           %d\n",n);
    fprintf(fp_eig,"Nspin       %d\n",SpinP_switch);
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
    /*  fprintf(fp_eig,"irange      %d %d\n",iemin,iemax); */
    fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
    fprintf(fp_eig,"atomnum     %d\n",atomnum);
    fprintf(fp_eig,"<WhatSpecies\n");
    for (i=1;i<=atomnum;i++) {
      fprintf(fp_eig,"%d ",WhatSpecies[i]);
    }
    fprintf(fp_eig,"\nWhatSpecies>\n");
    fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
    fprintf(fp_eig,"<Spe_Total_CNO\n");
    for (i=0;i<SpeciesNum;i++) {
      fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
    }
    fprintf(fp_eig,"\nSpe_Total_CNO>\n");
    MaxL=Supported_MaxL; 
    fprintf(fp_eig,"MaxL           %d\n",Supported_MaxL);
    fprintf(fp_eig,"<Spe_Num_CBasis\n");
    for (i=0;i<SpeciesNum;i++) {
      for (l=0;l<=MaxL;l++) {
	fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
      }
      fprintf(fp_eig,"\n");
    }
    fprintf(fp_eig,"Spe_Num_CBasis>\n");
    fprintf(fp_eig,"ChemP       %lf\n",ChemP);

    /* optical conductivity */
    if (fp_opt) {
      fprintf(fp_opt,"nspin   %d\n",SpinP_switch);
      fprintf(fp_opt,"N       %d\n",n);
    }

    printf("write eigenvalues\n");
    printf("write eigenvectors\n");

  } /* if (myid==Host_ID */

  Overlap_Cluster(CntOLP,S,MP);

  if (myid==Host_ID){

    n = S[0][0];

    Eigen_lapack(S,ko[0],n,n);

    /* S[i][j] contains jth eigenvector, not ith ! */

    /****************************************************
              searching of negative eigenvalues
    ****************************************************/

    /* minus eigenvalues to 1.0e-14 */
    
    for (l=1; l<=n; l++){
      if (ko[0][l]<0.0) ko[0][l] = 1.0e-14;
      EV_S[l] = ko[0][l];
    }

    /* print to the standard output */

    if (2<=level_stdout && myid==Host_ID){
      for (l=1; l<=n; l++){
	printf(" <Cluster_DFT_Dosout>  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[0][l]);
      }
    }

    /* calculate S*1/sqrt(ko) */

    for (l=1; l<=n; l++){
      IEV_S[l] = 1.0/sqrt(ko[0][l]);
    }

    /*********************************************************************
     A = U^+ S U    : A diagonal
     A[n] delta_nm =  U[j][n] S[j][i] U[i][m] =U^+[n][j] S[j][i] U[i][m]
     1 = U A^-1 U^+ S
     S^-1 =U A^-1 U^+
     S^-1[i][j]= U[i][n] A^-1[n] U^+[n][j]
     S^-1[i][j]= U[i][n] A^-1[n] U[j][n]
    **********************************************************************/
        
    /*
    printf("Error check S S^{-1} =\n");
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
        sum=0.0;
        for (k=1;k<=n;k++) {
           sum+= Ovlp[i][k]*  Sinv[k][j];
        }
        printf("%lf ",sum);
      }
      printf("\n");
    }
   */

  }  /*  if (myid==Host_ID */

#if 0
      for (i=1;i<=n;i++) {
        printf("%d: ",myid);
        for (j=1;j<=n;j++) {
          sum=S[i][j];
          printf("%lf ",sum);
        }
        printf("\n");
      }

  MPI_Finalize();
  exit(0);
#endif
  /****************************************************
    Calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  n1min=n;
  iemin=n; iemax=1;

  for (spin=0; spin<=SpinP_switch; spin++){

    Hamiltonian_Cluster(nh[spin],H[spin],MP);

    if (myid==Host_ID){

      /* optical conductivity */
      if (fp_opt) {
        CurrentOpt_Cluster( n, H[spin], Ovlp, Sinv, J );
      }

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          sum = 0.0;
          for (l=1; l<=n; l++){
            sum = sum + H[spin][i1][l]*S[l][j1]*IEV_S[j1]; 
          }
          C[spin][i1][j1] = sum;
        }
      }

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          sum = 0.0;
          for (l=1; l<=n; l++){
            sum = sum + IEV_S[i1]*S[l][i1]*C[spin][l][j1];
          }
          H[spin][i1][j1] = sum;
        }
      }

      /*****   H -> B_nl in the note  *****/

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          C[spin][i1][j1] = H[spin][i1][j1];
        }
      }

      /* penalty for ill-conditioning states */

      EV_cut0 = Threshold_OLP_Eigen;

      for (i1=1; i1<=n; i1++){

        if (EV_S[i1]<EV_cut0){
          C[spin][i1][i1] += pow((EV_S[i1]/EV_cut0),-2.0) - 1.0;
        }

        /* cutoff the interaction between the ill-conditioned state */
 
        if (1.0e+3<C[spin][i1][i1]){
          for (j1=1; j1<=n; j1++){
            C[spin][i1][j1] = 0.0;
            C[spin][j1][i1] = 0.0;
	  }
          C[spin][i1][i1] = 1.0e+4;
        }
      }

      /* diagonalize the matrix */

      n1 = n;
      Eigen_lapack(C[spin],ko[spin],n1,n1);

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n1; j1++){
	  H[spin][i1][j1] = C[spin][i1][j1];
        }
      }

      /* print to the standard output */

      if (2<=level_stdout && myid==Host_ID){
        for (l=1; l<=n; l++){
	  printf(" <Cluster_DFT_Dosout>  Eigenvalues of H spin=%2d  %2d  %18.15f\n",
                    spin,l,ko[spin][l]);
        }
      }

      /*****  H => D  in the note *****/

      if (n1min>n1) n1min=n1;

      iemin0=1;
      for (i1=1;i1<n1;i1++) {
        if (ko[spin][i1]>ChemP+Dos_Erange[0]) {
  	  iemin0=i1-1;
  	  break;
        }
      }
      if (iemin0<1)  iemin0=1;
      
      iemax0=n1;
      for (i1=iemin0;i1<n1;i1++) {
        if (ko[spin][i1]>ChemP+Dos_Erange[1]) {
	  iemax0=i1;
  	  break;
        }
      }
      if (iemax0>n1)  iemax0=n1;
    
      if (iemin>iemin0) iemin = iemin0;
      if (iemax<iemax0) iemax = iemax0;

      /****************************************************
          Transformation to the original eigenvectors.
                        AIST NOTE 244P
      ****************************************************/

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n; j1++){
          C[spin][i1][j1] = 0.0;
        }
      }

      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n1; j1++){
          sum = 0.0;
          for (l=1; l<=n; l++){
            sum = sum + S[i1][l]*IEV_S[l]*H[spin][l][j1];
          }
          C[spin][i1][j1] = sum;
        }
      }

      /***** C -> c_pn in the note *****/

      /* optical conductivity */

      if (fp_opt) {

#if 0
	printf("C=\n");
	for (i1=1;i1<=n;i1++) { /* a */
	  for (j1=1;j1<=n;j1++) { /* m */
	    printf("%lf ",C[spin][i1][j1]);
	  }
	  printf("\n");
	}
	printf("%d: calculation of barJ start n=%d\n",myid,n);
#endif

	/***  <n|\bar{J} |m> = sum  C_na <a|J|b> C_mb 
	      = sum_ab  C[a][n] J[a][b] C[b][m] ***/
	for (k=0;k<3;k++) { /* direction */

	  /* first J * C */
	  for (i1=1;i1<=n;i1++) { /* a */
	    for (j1=1;j1<=n;j1++) { /* m */
	      H[spin][i1][j1] =0.0;
	      for (i=1;i<=n;i++) {  /* b */
		H[spin][i1][j1] += J[k][i1][i]* C[spin][i][j1];
	      }
	    }
	  }
	  /*  C * (J * C) */
	  for (i1=1;i1<=n;i1++) { /* n */
	    for (j1=1;j1<=n;j1++) { /* m */
	      Ctmp[i1][j1]=0.0;
	      for (i=1;i<=n;i++) {  /* a */
		Ctmp[i1][j1] += C[spin][i][i1] * H[spin][i][j1]; 
	      }
	    }
	  }
        
	  /* output */
	  fprintf(fp_opt, "<barJ.spin=%d.axis=%d\n",spin,k+1);
	  for (i1=1;i1<=n;i1++) { /* n */
	    for (j1=1;j1<=n;j1++) { /* m */
	      fprintf( fp_opt, "%lf ",Ctmp[i1][j1]);
	    }
	    fprintf( fp_opt,"\n");
	  }       
	  fprintf(fp_opt, "barJ.spin=%d.dim=%d>\n",spin,k+1);

	}  /* k, direction */

#if 0
	printf("%d: calculation of barJ end\n",myid);
#endif

      } /* fp_opt */

    } /* if (myid==Host_ID) */
  }   /* spin */

  if (myid==Host_ID){
    if (iemax>n1min) iemax=n1min;
    printf("iemin iemax= %d %d\n",iemin,iemax);
  }

  /* MPI, iemin, iemax */
  MPI_Bcast(&iemin, 1, MPI_INT, Host_ID, mpi_comm_level1);
  MPI_Bcast(&iemax, 1, MPI_INT, Host_ID, mpi_comm_level1);

  if (myid==Host_ID){

    fprintf(fp_eig,"irange      %d %d\n",iemin,iemax);
    fprintf(fp_eig,"<Eigenvalues\n");

    for (spin=0; spin<=SpinP_switch; spin++) {
      fprintf(fp_eig,"%d %d %d ",0,0,0);
      for (ie=iemin;ie<=iemax;ie++) {
	fprintf(fp_eig,"%lf ",ko[spin][ie]);
	/* printf("%lf ",ko[spin][ie]); */
      }
      fprintf(fp_eig,"\n");
      /* printf("\n"); */
    }
    fprintf(fp_eig,"Eigenvalues>\n");
  }

  /****************************************************
    MPI:

    C
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){

    for (i1=0; i1<=n; i1++){
      for (j1=0; j1<=n; j1++){
        Ctmp[i1][j1] = C[spin][i1][j1];
      }
    }

    for (i1=0; i1<=n; i1++){
      MPI_Bcast(&Ctmp[i1][0], n+1, MPI_DOUBLE, Host_ID, mpi_comm_level1);
    }

    for (i1=0; i1<=n; i1++){
      for (j1=0; j1<=n; j1++){
        C[spin][i1][j1] = Ctmp[i1][j1];
      }
    }

  }

#if  0
  printf("%d: Bcast C end %d %d\n",myid,iemin,iemax);
  MPI_Barrier(mpi_comm_level1);
#endif

  /****************************************************
            calculation of density matrices 
            for up and down spins
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (k=iemin; k<=iemax; k++){

      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
        GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	Anum = MP[GA_AN];
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];
	  Bnum = MP[GB_AN];
	  for (i=0; i<tnoA; i++){
	    for (j=0; j<tnoB; j++){
              dum = C[spin][Anum+i][k]*C[spin][Bnum+j][k];
	      CDM[MA_AN][LB_AN][i][j] = dum;
	    }
	  }
	}
      }

      /*******************************************
                     M_i = S_ij D_ji
                     D_ji = C_nj C_ni
                     S_ij : CntOLP
                     D_ji : CDM
      ******************************************/

      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<=(tnoA-1); i1++){
	  SD[GA_AN][i1]=0.0;
	}
      }

      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
        GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<tnoA; i1++){
	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	    GB_AN = natn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    for (j1=0; j1<tnoB; j1++){
	      SD[GA_AN][i1] += CDM[MA_AN][LB_AN][i1][j1]*
                            CntOLP[MA_AN][LB_AN][i1][j1];
	    }
	  }
	}
      }

      i_vec[0]=i_vec[1]=i_vec[2]=0;
      if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);
      for (GA_AN=1; GA_AN<=atomnum; GA_AN++){
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	ID = G2ID[GA_AN];  
	if (ID==myid){
	  for (i1=0; i1<tnoA; i1++){
	    fSD[i1]=SD[GA_AN][i1];
	  }

	  if (myid!=Host_ID){
            tag = 999;
	    MPI_Isend(&fSD[0],tnoA,MPI_FLOAT,Host_ID,
		      tag,mpi_comm_level1,&request);
	    MPI_Wait(&request,&stat);
	  }
	}

	if (myid==Host_ID && ID!=Host_ID){
          tag = 999;
	  MPI_Recv(&fSD[0], tnoA, MPI_FLOAT, ID, tag, mpi_comm_level1, &stat);
	}

	if (myid==Host_ID) fwrite(fSD,sizeof(float),tnoA,fp_ev);
        MPI_Barrier(mpi_comm_level1);

      }


    } /* k */
  } /* spin */

 Finishing: ;

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
    if (fp_ev)  fclose(fp_ev);

    /* optical conductivity */
    if (fp_opt) fclose(fp_opt);
  }

  /****************************************************
                          Free
  ****************************************************/

#if 0
  printf("%d: free start\n",myid); 
#endif

  free(MP);

  for (spin=0; spin<i_ko[0]; spin++){
    free(ko[spin]);
  }
  free(ko);

  for (spin=0; spin<i_H[0]; spin++){
    for (i=0; i<i_H[1]; i++){
      free(H[spin][i]);
    }
    free(H[spin]);
  } 
  free(H);

  for (spin=0; spin<i_C[0]; spin++){
    for (i=0; i<i_C[1]; i++){
      free(C[spin][i]);
    }
    free(C[spin]);
  } 
  free(C);

  for (i=0; i<i_Ctmp[0]; i++){
    free(Ctmp[i]);
  }
  free(Ctmp);

  for (i=0; i<i_CDM[0]; i++){
    for (j=0; j<i_CDM[1]; j++){
      for (k=0; k<i_CDM[2]; k++){
        free(CDM[i][j][k]);
      }
      free(CDM[i][j]);
    }
    free(CDM[i]);
  }
  free(CDM);

  for (i=0; i<i_SD[0]; i++){
    free(SD[i]);
  }
  free(SD);

  free(fSD);

#if 0
   printf("%d: Dosout  Barrier start\n",myid);
   MPI_Barrier(mpi_comm_level1);
   printf("%d: Dosout Barrier end\n",myid);

#endif

  /* for elapsed time */

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;

}





double Cluster_NonCol_Dosout( int SpinP_switch, 
                              double *****nh,
                              double *****ImNL,
                              double ****CntOLP)
{
  int n,i,j,wanA,n2;
  int l,ii1,jj1;
  int *MP;
  int n1min,iemin,iemax,spin,i1,j1,iemin0,iemax0,n1,ie;
  int MA_AN, GA_AN, tnoA, Anum, LB_AN, GB_AN,wanB, tnoB, Bnum, k;
  int MaxL;
  int i_vec[10];  
  int file_ptr_size;

  double EV_cut0;
  double *ko; int N_ko, i_ko[10];
  dcomplex **H; int N_H, i_H[10];
  dcomplex **C; int N_C, i_C[10];
  double **Ctmp; int N_Ctmp, i_Ctmp[10];
  double *****CDM; int N_CDM,i_CDM[10];
  double ***SD; int N_SD, i_SD[10];
  double **SDup,**SDdn;
  double TStime,TEtime,time0;
  double sum_r,sum_i;
  double sit,cot,sip,cop,theta,phi;
  double sum,dum,tmp1,tmp2,tmp3;
  float *fSD; 

  char file_eig[YOUSO10],file_ev[YOUSO10];
  FILE *fp_eig, *fp_ev;
  char buf1[fp_bsize];          /* setvbuf */
  char buf2[fp_bsize];          /* setvbuf */
  int numprocs,myid,ID,tag;
  MPI_Status stat;
  MPI_Request request;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  printf("Cluster_DFT_Dosout: start\n"); fflush(stdout);

  dtime(&TStime);

  /****************************************************
             calculation of the array size
  ****************************************************/

  n = 0;
  for (i=1; i<=atomnum; i++){
    wanA  = WhatSpecies[i];
    n  = n + Spe_Total_CNO[wanA];
  }
  n2 = 2*n + 2;

  /****************************************************
   Allocation

   int     MP[List_YOUSO[1]]
   double  ko[n2]
   double  H[n2][n2]
   double  C[n2][n2]
   double  Ctmp[n2][n2]
   double  CDM[5][Matomnum+1][List_YOUSO[8]]
              [List_YOUSO[7]][List_YOUSO[7]]
   double  S[3]D[List_YOUSO[1]][List_YOUSO[7]]
   float   fSD[List_YOUSO[7]]
  ****************************************************/

  MP = (int*)malloc(sizeof(int)*List_YOUSO[1]);

  N_ko=1; i_ko[0]=n2;
  ko=(double*)malloc_multidimarray("double",N_ko,i_ko);

  N_H=2; i_H[0]=n2; i_H[1]=n2;
  H=(dcomplex**)malloc_multidimarray("dcomplex",N_H, i_H);

  N_C=2; i_C[0]=n2; i_C[1]=n2;
  C=(dcomplex**)malloc_multidimarray("dcomplex",N_C, i_C);

  N_Ctmp=2; i_Ctmp[0]=n2; i_Ctmp[1]=n2;
  Ctmp=(double**)malloc_multidimarray("double",N_Ctmp, i_Ctmp);

  N_CDM=5;  i_CDM[0]=4; i_CDM[1]=Matomnum+1; i_CDM[2]=List_YOUSO[8];
  i_CDM[3]=List_YOUSO[7]; i_CDM[4]=List_YOUSO[7];
  CDM =(double*****)malloc_multidimarray("double",N_CDM,i_CDM);

  N_SD=3; i_SD[0]=4;  i_SD[1]=List_YOUSO[1]; i_SD[2]=List_YOUSO[7];
  SD = (double***)malloc_multidimarray("double",N_SD, i_SD);

  SDup = (double**)malloc(sizeof(double*)*List_YOUSO[1]); 
  for (i=0; i<List_YOUSO[1]; i++){ 
    SDup[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]); 
  }

  SDdn = (double**)malloc(sizeof(double*)*List_YOUSO[1]); 
  for (i=0; i<List_YOUSO[1]; i++){ 
    SDdn[i] = (double*)malloc(sizeof(double)*List_YOUSO[7]); 
  }

  fSD=(float*)malloc(sizeof(float)*List_YOUSO[7]*2);

  if (myid==Host_ID){
    strcpy(file_eig,".Dos.val");
    fnjoint(filepath,filename,file_eig);
    if ( (fp_eig=fopen(file_eig,"w"))==NULL ) {

#ifdef xt3
      setvbuf(fp_eig,buf1,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open a file %s\n",file_eig);
    }

    strcpy(file_ev,".Dos.vec");
    fnjoint(filepath,filename,file_ev);
    if ( (fp_ev=fopen(file_ev,"w"))==NULL ) {

#ifdef xt3
      setvbuf(fp_ev,buf2,_IOFBF,fp_bsize);  /* setvbuf */
#endif

      printf("can not open a file %s\n",file_ev);
    }
  }


  file_ptr_size=sizeof(FILE *);

  MPI_Bcast(&fp_eig,file_ptr_size,MPI_BYTE,Host_ID,mpi_comm_level1);
  MPI_Bcast(&fp_ev,file_ptr_size,MPI_BYTE,Host_ID,mpi_comm_level1);

 
  if ( fp_eig==NULL || fp_ev==NULL ) {
    goto Finishing;
  }

  if (myid==Host_ID){

    fprintf(fp_eig,"mode        1\n");
    fprintf(fp_eig,"NonCol      1\n");
    fprintf(fp_eig,"N           %d\n",n);
    fprintf(fp_eig,"Nspin       %d\n",1); /* switch to 1 */ 
    fprintf(fp_eig,"Erange      %lf %lf\n",Dos_Erange[0],Dos_Erange[1]);
    /*  fprintf(fp_eig,"irange      %d %d\n",iemin,iemax); */
    fprintf(fp_eig,"Kgrid       %d %d %d\n",1,1,1);
    fprintf(fp_eig,"atomnum     %d\n",atomnum);
    fprintf(fp_eig,"<WhatSpecies\n");
    for (i=1;i<=atomnum;i++) {
      fprintf(fp_eig,"%d ",WhatSpecies[i]);
    }
    fprintf(fp_eig,"\nWhatSpecies>\n");
    fprintf(fp_eig,"SpeciesNum     %d\n",SpeciesNum);
    fprintf(fp_eig,"<Spe_Total_CNO\n");
    for (i=0;i<SpeciesNum;i++) {
      fprintf(fp_eig,"%d ",Spe_Total_CNO[i]);
    }
    fprintf(fp_eig,"\nSpe_Total_CNO>\n");
    MaxL=Supported_MaxL; 
    fprintf(fp_eig,"MaxL           %d\n",Supported_MaxL);
    fprintf(fp_eig,"<Spe_Num_CBasis\n");
    for (i=0;i<SpeciesNum;i++) {
      for (l=0;l<=MaxL;l++) {
	fprintf(fp_eig,"%d ",Spe_Num_CBasis[i][l]);
      }
      fprintf(fp_eig,"\n");
    }
    fprintf(fp_eig,"Spe_Num_CBasis>\n");
    fprintf(fp_eig,"ChemP       %lf\n",ChemP);

    fprintf(fp_eig,"<SpinAngle\n");
    for (i=1; i<=atomnum; i++) {
      fprintf(fp_eig,"%lf %lf\n",Angle0_Spin[i],Angle1_Spin[i]);
    }
    fprintf(fp_eig,"SpinAngle>\n");


    printf("write eigenvalues\n");
    printf("write eigenvectors\n");

  }

  Overlap_Cluster(CntOLP,S,MP);

  if (myid==Host_ID){

    n = S[0][0];
    Eigen_lapack(S,ko,n,n);

    /* minus eigenvalues to 1.0e-14 */
    for (l=1; l<=n; l++){
      if (ko[l]<0.0) ko[l] = 1.0e-14;
      EV_S[l] = ko[l];
    }

    /* print to the standard output */

    if (2<=level_stdout && myid==Host_ID){
      for (l=1; l<=n; l++){
	printf("  <Cluster_DFT_Dosout>  Eigenvalues of OLP  %2d  %18.15f\n",l,ko[l]);
      }
    }

    /* calculate S*1/sqrt(ko) */
  
    for (l=1; l<=n; l++){
      IEV_S[l] = 1.0/sqrt(ko[l]);
    }
  }

  /****************************************************
    Calculations of eigenvalues for up and down spins

     Note:
         MP indicates the starting position of
              atom i in arraies H and S
  ****************************************************/

  Hamiltonian_Cluster_NC(nh, ImNL, H, MP);

  if (myid==Host_ID){

    /* H * U * lambda^{-1/2} */

    for (i1=1; i1<=2*n; i1++){
      for (j1=1; j1<=n; j1++){
                
        for (k=0; k<=1; k++){

          sum_r = 0.0;
          sum_i = 0.0;

          for (l=1; l<=n; l++){
            sum_r = sum_r + H[i1][l+k*n].r*S[l][j1]*IEV_S[j1];
            sum_i = sum_i + H[i1][l+k*n].i*S[l][j1]*IEV_S[j1];
          }

          jj1 = 2*j1 - 1 + k;
          C[i1][jj1].r = sum_r;
          C[i1][jj1].i = sum_i;
	}
      }
    }

    /* lambda^{-1/2} * U^+ H * U * lambda^{-1/2} */

    for (i1=1; i1<=n; i1++){

      for (k=0; k<=1; k++){

        ii1 = 2*i1 - 1 + k;

        for (j1=1; j1<=2*n; j1++){
          sum_r = 0.0;
          sum_i = 0.0;
          for (l=1; l<=n; l++){
	    sum_r = sum_r + IEV_S[i1]*S[l][i1]*C[l+k*n][j1].r;
	    sum_i = sum_i + IEV_S[i1]*S[l][i1]*C[l+k*n][j1].i;
          }

          H[ii1][j1].r = sum_r;
          H[ii1][j1].i = sum_i;
        }
      }
    }

    /* H to C */

    for (i1=1; i1<=2*n; i1++){
      for (j1=1; j1<=2*n; j1++){
        C[i1][j1].r = H[i1][j1].r;
        C[i1][j1].i = H[i1][j1].i;
      }
    }

    /* penalty for ill-conditioning states */

    EV_cut0 = Threshold_OLP_Eigen;

    for (i1=1; i1<=n; i1++){

      if (EV_S[i1]<EV_cut0){
	C[2*i1-1][2*i1-1].r += pow((EV_S[i1]/EV_cut0),-2.0) - 1.0;
	C[2*i1  ][2*i1  ].r += pow((EV_S[i1]/EV_cut0),-2.0) - 1.0;
      }

      /* cutoff the interaction between the ill-conditioned state */

      if (1.0e+3<C[2*i1-1][2*i1-1].r){
	for (j1=1; j1<=2*n; j1++){
	  C[2*i1-1][j1    ] = Complex(0.0,0.0);
	  C[j1    ][2*i1-1] = Complex(0.0,0.0);
	  C[2*i1  ][j1    ] = Complex(0.0,0.0);
	  C[j1    ][2*i1  ] = Complex(0.0,0.0);
	}
	C[2*i1-1][2*i1-1] = Complex(1.0e+4,0.0);
	C[2*i1  ][2*i1  ] = Complex(1.0e+4,0.0);
      }
    }

    /* solve eigenvalue problem */

    n1 = 2*n;
    EigenBand_lapack(C,ko,n1,n1,1);

    for (i1=1; i1<=n1; i1++){
      for (j1=1; j1<=n1; j1++){
        H[i1][j1].r = C[i1][j1].r;
        H[i1][j1].i = C[i1][j1].i;
      }
    }

    /****************************************************
        Transformation to the original eigenvectors.
        JRCAT NOTE 244P  C = U * lambda^{-1/2} * D
    ****************************************************/

    for (i1=1; i1<=2*n; i1++){
      for (j1=1; j1<=2*n; j1++){
        C[i1][j1].r = 0.0;
        C[i1][j1].i = 0.0;
      }
    }

    for (k=0; k<=1; k++){
      for (i1=1; i1<=n; i1++){
        for (j1=1; j1<=n1; j1++){
          sum_r = 0.0;
          sum_i = 0.0;
          for (l=1; l<=n; l++){
            sum_r = sum_r + S[i1][l]*IEV_S[l]*H[2*(l-1)+1+k][j1].r;
            sum_i = sum_i + S[i1][l]*IEV_S[l]*H[2*(l-1)+1+k][j1].i;
          }
          C[i1+k*n][j1].r = sum_r;
          C[i1+k*n][j1].i = sum_i;
        }
      }
    }
  } /* if (myid==Host_ID) */

  if (myid==Host_ID){
 
    iemin = 1;
    for (i1=1;i1<=n1;i1++) {
      if (ko[i1]>ChemP+Dos_Erange[0]){
        iemin=i1-1;
        break;
      }
    }
    if (iemin<1)  iemin=1;

    iemax = n1;
    for (i1=iemin; i1<=n1; i1++) {
      if (ko[i1]>ChemP+Dos_Erange[1]) {
        iemax=i1;
        break;
      }
    }
    if (iemax>n1)  iemax=n1;
  }

  /* MPI, iemin, iemax */
  MPI_Bcast(&iemin, 1, MPI_INT, Host_ID, mpi_comm_level1);
  MPI_Bcast(&iemax, 1, MPI_INT, Host_ID, mpi_comm_level1);

  if (myid==Host_ID){

    fprintf(fp_eig,"irange      %d %d\n",iemin,iemax);
    fprintf(fp_eig,"<Eigenvalues\n");

    for (spin=0; spin<=1; spin++) {
      fprintf(fp_eig,"%d %d %d ",0,0,0);
      for (ie=iemin; ie<=iemax; ie++) {
	fprintf(fp_eig,"%lf ",ko[ie]);
      }
      fprintf(fp_eig,"\n");
      /* printf("\n"); */
    }
    fprintf(fp_eig,"Eigenvalues>\n");

  }

  /****************************************************
     MPI:

     C
  ****************************************************/

  for (i1=0; i1<=2*n; i1++){
    for (j1=0; j1<=2*n; j1++){
      Ctmp[i1][j1] = C[i1][j1].r;
    }
  }

  for (i1=1; i1<=2*n; i1++){
     MPI_Bcast(&Ctmp[i1][0], 2*n, MPI_DOUBLE, Host_ID, mpi_comm_level1);
  }

  for (i1=0; i1<=2*n; i1++){
    for (j1=0; j1<=2*n; j1++){
      C[i1][j1].r = Ctmp[i1][j1];
    }
  }

  for (i1=0; i1<=2*n; i1++){
    for (j1=0; j1<=2*n; j1++){
      Ctmp[i1][j1] = C[i1][j1].i;
    }
  }

  for (i1=1; i1<=2*n; i1++){
     MPI_Bcast(&Ctmp[i1][0], 2*n, MPI_DOUBLE, Host_ID, mpi_comm_level1);
  }

  for (i1=0; i1<=2*n; i1++){
    for (j1=0; j1<=2*n; j1++){
      C[i1][j1].i = Ctmp[i1][j1];
    }
  }

  /****************************************************
     calculate fraction of density matrix
  ****************************************************/

  for (k=iemin; k<=iemax; k++){

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){

      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      Anum = MP[GA_AN];
      for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	GB_AN = natn[GA_AN][LB_AN];
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	Bnum = MP[GB_AN];
	for (i=0; i<tnoA; i++){
	  for (j=0; j<tnoB; j++){

	    /* Re11 */
	    dum = C[Anum+i][k].r*C[Bnum+j][k].r + C[Anum+i][k].i*C[Bnum+j][k].i;
	    CDM[0][MA_AN][LB_AN][i][j] = dum;

	    /* Re22 */
	    dum = C[Anum+i+n][k].r*C[Bnum+j+n][k].r + C[Anum+i+n][k].i*C[Bnum+j+n][k].i;
	    CDM[1][MA_AN][LB_AN][i][j] = dum;

	    /* Re12 */
	    dum = C[Anum+i][k].r*C[Bnum+j+n][k].r + C[Anum+i][k].i*C[Bnum+j+n][k].i;
	    CDM[2][MA_AN][LB_AN][i][j] = dum;

	    /* Im12
	       conjugate complex of Im12 due to difference in the definition
	       between density matrix and charge density
	    */

	    dum = -(C[Anum+i][k].r*C[Bnum+j+n][k].i - C[Anum+i][k].i*C[Bnum+j+n][k].r);
	    CDM[3][MA_AN][LB_AN][i][j] = dum;

	  }
	}
      }
    }

    /*******************************************
                   M_i = S_ij D_ji
                   D_ji = C_nj C_ni
                   S_ij : CntOLP
                   D_ji : CDM
    *******************************************/

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){ 
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i1=0; i1<tnoA; i1++){
	SD[0][GA_AN][i1] = 0.0;
	SD[1][GA_AN][i1] = 0.0;
	SD[2][GA_AN][i1] = 0.0;
	SD[3][GA_AN][i1] = 0.0;
      }
    }

    for (spin=0; spin<=3; spin++){
      for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
	GA_AN = M2G[MA_AN];
	wanA = WhatSpecies[GA_AN];
	tnoA = Spe_Total_CNO[wanA];
	for (i1=0; i1<tnoA; i1++){
	  for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	    GB_AN = natn[GA_AN][LB_AN];
	    wanB = WhatSpecies[GB_AN];
	    tnoB = Spe_Total_CNO[wanB];
	    for (j1=0; j1<tnoB; j1++){
	      SD[spin][GA_AN][i1] += CDM[spin][MA_AN][LB_AN][i1][j1]*
	 	                        CntOLP[MA_AN][LB_AN][i1][j1];
	    }
	  }
	}
      }
    }

    /*  transform to up and down states */

    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];

      theta = Angle0_Spin[GA_AN];
      phi   = Angle1_Spin[GA_AN];

      sit = sin(theta);
      cot = cos(theta);
      sip = sin(phi);
      cop = cos(phi);     

      for (i1=0; i1<tnoA; i1++){

        tmp1 = 0.5*(SD[0][GA_AN][i1] + SD[1][GA_AN][i1]);
        tmp2 = 0.5*cot*(SD[0][GA_AN][i1] - SD[1][GA_AN][i1]);
        tmp3 = (SD[2][GA_AN][i1]*cop - SD[3][GA_AN][i1]*sip)*sit;

        SDup[GA_AN][i1] = tmp1 + tmp2 + tmp3;
        SDdn[GA_AN][i1] = tmp1 - tmp2 - tmp3;
      }
    }

    /*  writting a binary file */

    i_vec[0]=i_vec[1]=i_vec[2]=0;
    if (myid==Host_ID) fwrite(i_vec,sizeof(int),3,fp_ev);

    for (GA_AN=1; GA_AN<=atomnum; GA_AN++){

      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      ID = G2ID[GA_AN];  

      if (ID==myid){

	for (i1=0; i1<tnoA; i1++){
	  fSD[i1] = SDup[GA_AN][i1];
	}
	for (i1=0; i1<tnoA; i1++){
	  fSD[tnoA+i1] = SDdn[GA_AN][i1];
	}

	if (myid!=Host_ID){
	  tag = 999;
	  MPI_Isend(&fSD[0], 2*tnoA, MPI_FLOAT, Host_ID, tag, mpi_comm_level1, &request);
	  MPI_Wait(&request,&stat);
	}
      }

      if (myid==Host_ID && ID!=Host_ID){
	tag = 999;
	MPI_Recv(&fSD[0], 2*tnoA, MPI_FLOAT, ID, tag, mpi_comm_level1, &stat);
      }

      if (myid==Host_ID) fwrite(fSD,sizeof(float),2*tnoA,fp_ev);
      MPI_Barrier(mpi_comm_level1);

    } /* GA_AN */ 
  } /* for (k=iemin; k<=iemax; k++){ */



Finishing: ;

  if (myid==Host_ID){
    if (fp_eig) fclose(fp_eig);
    if (fp_ev)  fclose(fp_ev);
  }

  /****************************************************
                          Free
  ****************************************************/

  free(MP);
  free(ko);

  for (i=0; i<i_H[0]; i++){
    free(H[i]);
  }
  free(H);

  for (i=0; i<i_C[0]; i++){
    free(C[i]);
  }
  free(C);

  for (i=0; i<i_Ctmp[0]; i++){
    free(Ctmp[i]);
  }
  free(Ctmp);

  for (i=0; i<i_CDM[0]; i++){
    for (j=0; j<i_CDM[1]; j++){
      for (k=0; k<i_CDM[2]; k++){
        for (l=0; l<i_CDM[3]; l++){
          free(CDM[i][j][k][l]);
	}
        free(CDM[i][j][k]);
      }
      free(CDM[i][j]);
    }
    free(CDM[i]);
  }
  free(CDM);

  for (i=0; i<i_SD[0]; i++){
    for (j=0; j<i_SD[1]; j++){
      free(SD[i][j]);
    }
    free(SD[i]);
  }
  free(SD);

  for (i=0; i<List_YOUSO[1]; i++){ 
    free(SDup[i]);
  }
  free(SDup);

  for (i=0; i<List_YOUSO[1]; i++){ 
    free(SDdn[i]);
  }
  free(SDdn);

  free(fSD);

  /* for elapsed time */

  dtime(&TEtime);
  time0 = TEtime - TStime;
  return time0;

}






/***************************************************************************
 calculate <a|J|b> = i <a| H r - r H |a> 
                   = i <a| H |c><\bar{c}| r |a> - i<a| r |\bar{c}><c| H |a>

   <a|r|\bar{c}> = <a|r|d> S^-1_{dc} 

   <a|r|d> ~ 1/2 (R_a + R_d ) S_ad

 input:  n
         H[1:n][1:n] Hamiltonian 
         Ovlp[1:n][1:n] S
         Sinv[1:n][1:n] S^-1 
 output:  J[0:2][1:n][1:n]  <=  <a|J|b>/i , without imaginary part
***************************************************************************/


void  CurrentOpt_Cluster( int n,double **H,double **Ovlp, double **Sinv, double ***J )
{
  int Anum, GA_AN, wanA, tnoA;
  int Bnum, GB_AN, wanB, tnoB;
  int i,j,k,idim;
  double ***rij; int N_rij, i_rij[10];

  /* allocate work array */
  N_rij=3;
  i_rij[0]=3;
  i_rij[1]=n+1;
  i_rij[2]=n+1;
  rij = (double***)malloc_multidimarray("double",N_rij,i_rij);

  /*** initialize ***/
  for (idim=0;idim<3;idim++)
    for (i=1;i<=n;i++)
      for (j=1;j<=n;j++)  {
        J[idim][i][j]= 0.0;
        rij[idim][i][j]=0.0 ;
      }

#if 0
    printf("H=\n");
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++)  {
        printf("%lf ",H[i][j]);
      }
     printf("\n");
    }

    printf("\nS=\n");
    for (i=1;i<=n;i++){
      for (j=1;j<=n;j++)  {
        printf("%lf ",Ovlp[i][j]);
      }
    printf("\n");
    }

    printf("\nSinv=\n");
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++)  {
        printf("%lf ",Sinv[i][j]);
      }
      printf("\n");
    }
    printf("\n");
#endif


  /*** <a|r|b> ***/
  Anum=0;
  for (GA_AN=1; GA_AN<=atomnum; GA_AN++) {
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];  
    for (i=0;i<tnoA;i++) {  /* loop i to increase Anum */
      Anum++;
      Bnum=0;
      for (GB_AN=1; GB_AN<=atomnum; GB_AN++) {
	wanB = WhatSpecies[GB_AN];
	tnoB = Spe_Total_CNO[wanB];
	for (j=0;j<tnoB;j++) {  /* loop j to increase Bnum */
	  Bnum++;
	  for (idim=0;idim<3;idim++) {
	    
            /*  <a|r|d>  = 1/2 (R_a +R_d) S_ad */
            /* J[0:2][][] , whereas Gxyz[][1:3], confusing */
             J[idim][Anum][Bnum]=  0.5*(Gxyz[GA_AN][idim+1]+Gxyz[GB_AN][idim+1])*Ovlp[Anum][Bnum];

	  }  /* idim */
	} /* j */
      } /* GA_AN */
    } /* i */
  } /* GA_AN */

#if 0
   /* print  <a|r|d> */
  for (idim=0;idim<3;idim++) {
    printf("<a|r|d> idim=%d\n",idim);
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
        printf("%lf ",J[idim][i][j]);
      }
      printf("\n");
    }
  }
#endif

  /*** <a|r|\bar{c}> = <a|r|d> S^-1_{dc} ***/
  for (idim=0;idim<3;idim++)
    for (i=1;i<=n;i++)
      for (j=1;j<=n;j++)  
        for (k=1;k<=n;k++) {
          rij[idim][i][j] += J[idim][i][k]*Sinv[k][j]; 
      }
#if 0
   /* print  <a|r|d> */
  for (idim=0;idim<3;idim++) {
    printf("<a|r|bar{d}> idim=%d\n",idim);
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
        printf("%lf ",rij[idim][i][j]);
      }
      printf("\n");
    }
  }
#endif

  
   /*** rij <=  <a|r|\bar{c}> ***/
  
   /* initialize J */
  for (idim=0;idim<3;idim++)
    for (i=1;i<=n;i++)
      for (j=1;j<=n;j++)
          J[idim][i][j]=0.0;

   /*** <a|J|b> =  <a| H |c><\bar{c}| r |a> - <a| r |\bar{c}><c| H |a> ***/
  for (idim=0;idim<3;idim++)
    for (i=1;i<=n;i++)
      for (j=1;j<=n;j++)
        for (k=1;k<=n;k++) 
          J[idim][i][j] += H[i][k]*rij[idim][j][k] - rij[idim][i][k]*H[k][j] ;
        
#if 0
   /* print  <a|J|b> */
  for (idim=0;idim<3;idim++) {
    printf("<a|J|d> idim=%d\n",idim);
    for (i=1;i<=n;i++) {
      for (j=1;j<=n;j++) {
        printf("%lf ",J[idim][i][j]);
      }
      printf("\n");
    }
  }
#endif



  /*** free rij ***/
  free_multidimarray((void**)rij,N_rij,i_rij);

#if 1
   printf("CurrentOpt_Cluster end\n");
#endif

}

