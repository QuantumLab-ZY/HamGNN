/**********************************************************************
  Set_Allocate_Atom2CPU.c:

    Set_Allocate_Atom2CPU.c is a subroutine to allocate atoms to processors
    for the MPI parallel computation.

  Log of Set_Allocate_Atom2CPU.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "openmx_common.h"
#include "mpi.h"

static void Output_Atom2CPU();
static void Allocation_Species();
static void Allocation_Atoms_3D(int MD_iter, int NL_switch);



int Set_Allocate_Atom2CPU(int MD_iter, int species_flag, int weight_flag)
{
  double time0;
  time_t TStime,TEtime;

  time(&TStime);

  if (species_flag==1)
    Allocation_Species(); 
  else{ 
    Allocation_Atoms_3D(MD_iter, weight_flag);
  }
  
  time(&TEtime);
  time0 = TEtime - TStime;
  return time0;
}



#pragma optimization_level 1
void Allocation_Atoms_3D(int MD_iter, int weight_flag)
{
  /***************************************
        allocate atoms to processes
      by Modified Recursive Bisection
  ***************************************/
  int m,i,j,k,k0,Na,np,ID,n0; 
  int myid,numprocs,numprocs0;
  int max_depth,n,depth,child;
  int **Num_Procs_in_Child;
  int **Num_Atoms_in_Child;
  int ***List_AN_in_Child;
  int *MatomN,*ID2ID;
  double t,ax,ay,az,sum;
  double w0,sumw,min_diff;
  double WMatomnum,longest_time;
  double ***List_T_in_Child;
  double **IMT,*weight;
  double *ko,*WMatomN;
  double xyz_c[4];

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* set numprocs0 */
  if (numprocs<atomnum) numprocs0 = numprocs;
  else                  numprocs0 = atomnum;

  /* max_depth of level */

  if (numprocs==1 || atomnum==1)
    max_depth = 0;
  else  
    max_depth = (int)(log(((double)numprocs0-1.0+1.0e-7))/log(2.0)) + 1;

  /****************************
      allocation of arrays
  ****************************/

  Num_Procs_in_Child = (int**)malloc(sizeof(int*)*(max_depth+1));
  n = 1; 
  for (depth=0; depth<(max_depth+1); depth++){
    Num_Procs_in_Child[depth] = (int*)malloc(sizeof(int)*n);
    n *= 2;
  }

  Num_Atoms_in_Child = (int**)malloc(sizeof(int*)*(max_depth+1));
  n = 1; 
  for (depth=0; depth<(max_depth+1); depth++){
    Num_Atoms_in_Child[depth] = (int*)malloc(sizeof(int)*n);
    for (i=0; i<n; i++) Num_Atoms_in_Child[depth][i] = 0; 
    n *= 2;
  }

  IMT = (double**)malloc(sizeof(double*)*4);
  for (i=0; i<4; i++){
    IMT[i] = (double*)malloc(sizeof(double)*4);
  }

  ko = (double*)malloc(sizeof(double)*4);

  weight = (double*)malloc(sizeof(double)*(atomnum+1));

  /* set weight */    

  if (weight_flag==0){
    for (i=1; i<=atomnum; i++){
      weight[i] = 1.0;
    }
  }
  else if (weight_flag==1){

    longest_time = 0.0;
    for (i=1; i<=atomnum; i++){
      if (longest_time<time_per_atom[i]) longest_time = time_per_atom[i];
    }

    for (i=1; i<=atomnum; i++){
      weight[i] = time_per_atom[i]/longest_time;
    }
  }

  /* set Num_Procs_in_Child */    

  n = 2; 
  Num_Procs_in_Child[0][0] = numprocs0;

  for (depth=1; depth<(max_depth+1); depth++){
    for (i=0; i<n; i++){

      if (i%2==0){
        Num_Procs_in_Child[depth][i] = (Num_Procs_in_Child[depth-1][i/2]-1)/2+1;
      }  
      else{
        Num_Procs_in_Child[depth][i] = Num_Procs_in_Child[depth-1][i/2]-Num_Procs_in_Child[depth][i-1];  
      } 
    }
    n *= 2;
  }    

  /* set Num_Atoms_in_Child at depth=0 */    

  depth = 0; child = 0;
  Num_Atoms_in_Child[depth][child] = atomnum;

  /***************************************************************
   modified recursive bisection to set AN_in_Child at each depth 
  ***************************************************************/ 

  /**************************************************************************
   Since the size of the last index of List_AN_in_Child and List_T_in_Child
   is determined by the modified recursive bisection, they are allocated 
   on-the-fly. 
  **************************************************************************/

  /* allocation of List_AN_in_Child and List_T_in_Child */
  List_AN_in_Child = (int***)malloc(sizeof(int**)*(max_depth+1)); 
  List_T_in_Child = (double***)malloc(sizeof(double**)*(max_depth+1)); 
  List_AN_in_Child[0] = (int**)malloc(sizeof(int*)*1);
  List_T_in_Child[0] = (double**)malloc(sizeof(double*)*1);
  List_AN_in_Child[0][0] = (int*)malloc(sizeof(int)*(atomnum+1));
  List_T_in_Child[0][0]  = (double*)malloc(sizeof(double)*(atomnum+1));
  for (k=0; k<atomnum; k++)  List_AN_in_Child[depth][0][k] = k+1;

  n = 1; 
  for (depth=0; depth<max_depth; depth++){

    /* allocation of List_AN_in_Child and List_T_in_Child */
    List_AN_in_Child[depth+1] = (int**)malloc(sizeof(int*)*n*2);
    List_T_in_Child[depth+1] = (double**)malloc(sizeof(double*)*n*2);

    /**********************************************************
     reordering of atoms at depth using the inertia tensor
    **********************************************************/

    for (child=0; child<n; child++){

      /* get the number of atoms in the child */

      Na = Num_Atoms_in_Child[depth][child];

      /* calculate the centroid of atoms in the child */

      xyz_c[1] = 0.0; 
      xyz_c[2] = 0.0; 
      xyz_c[3] = 0.0; 

      for (k=0; k<Na; k++){
        m = List_AN_in_Child[depth][child][k];
        xyz_c[1] += Gxyz[m][1]*weight[m];
        xyz_c[2] += Gxyz[m][2]*weight[m];
        xyz_c[3] += Gxyz[m][3]*weight[m];
      }

      xyz_c[1] /= (double)Na;
      xyz_c[2] /= (double)Na;
      xyz_c[3] /= (double)Na;

      /* make inertia moment tensor */

      for (i=1; i<=3; i++){
        for (j=1; j<=3; j++){

          sum = 0.0;
          for (k=0; k<Na; k++){
	    m = List_AN_in_Child[depth][child][k];
	    sum += weight[m]*(Gxyz[m][i]-xyz_c[i])*(Gxyz[m][j]-xyz_c[j]);
	  }

          IMT[i][j] = sum;
        }
      }

      /* diagonalize the inertia moment tensor */

      Eigen_lapack(IMT,ko,3,3);

      /* find the principal axis */
  
      ax = IMT[1][3];
      ay = IMT[2][3];
      az = IMT[3][3];

      /* calculate the intervening variable, t */

      for (k=0; k<Na; k++){
        m = List_AN_in_Child[depth][child][k];
        t = ax*(Gxyz[m][1]-xyz_c[1]) + ay*(Gxyz[m][2]-xyz_c[2]) + az*(Gxyz[m][3]-xyz_c[3]);
        List_T_in_Child[depth][child][k] = t;
      }

      /* sorting atoms in the child based on t */

      qsort_double_int((long)Na,List_T_in_Child[depth][child],List_AN_in_Child[depth][child]);

      /* calculate the sum of weight in the child */

      sumw = 0.0;
      for (k=0; k<Na; k++){
        m = List_AN_in_Child[depth][child][k];
        sumw += weight[m];
      }

      /* find atomic index at which the bisection is made. */

      np = Num_Procs_in_Child[depth+1][2*child] + Num_Procs_in_Child[depth+1][2*child+1];
      w0 = (sumw*(double)Num_Procs_in_Child[depth+1][2*child])/(double)np;

      sumw = 0.0;
      min_diff = 10000000; 
      for (k=0; k<Na; k++){
        m = List_AN_in_Child[depth][child][k];
        sumw += weight[m];
        if (fabs(w0-sumw)<min_diff){
          min_diff = fabs(w0-sumw);
          k0 = k;
        }
      }

      /* adjust k0 to avoid the case that (# of atoms)<(# of processes) */

      if ( (((k0+1)<Num_Procs_in_Child[depth+1][2*child])
           ||
           ((Na-(k0+1))<Num_Procs_in_Child[depth+1][2*child+1]))
           && 
           1<Na ){

        k0 = Num_Procs_in_Child[depth+1][2*child] - 1; 
      }

      /* bisection of atoms in the child based on Num_Procs_in_Child */

      Num_Atoms_in_Child[depth+1][2*child  ] = k0 + 1;
      Num_Atoms_in_Child[depth+1][2*child+1] = Na - (k0+1);

      /* allocation of List_AN_in_Child and List_T_in_Child */
      List_AN_in_Child[depth+1][2*child]   = (int*)malloc(sizeof(int)*Num_Atoms_in_Child[depth+1][2*child]);
      List_T_in_Child[depth+1][2*child]   = (double*)malloc(sizeof(double)*Num_Atoms_in_Child[depth+1][2*child]);
      List_AN_in_Child[depth+1][2*child+1] = (int*)malloc(sizeof(int)*Num_Atoms_in_Child[depth+1][2*child+1]);
      List_T_in_Child[depth+1][2*child+1] = (double*)malloc(sizeof(double)*Num_Atoms_in_Child[depth+1][2*child+1]);

      /* copy depth -> depth+1 */

      for (k=0; k<Num_Atoms_in_Child[depth+1][2*child]; k++){
        List_AN_in_Child[depth+1][2*child][k] = List_AN_in_Child[depth][child][k];
      }

      for (k=0; k<Num_Atoms_in_Child[depth+1][2*child+1]; k++){
        m = Num_Atoms_in_Child[depth+1][2*child]+k;
        List_AN_in_Child[depth+1][2*child+1][k] = List_AN_in_Child[depth][child][m];
      }

    } /* child */     

    /* doubling of n */
    n *= 2;

  } /* depth */

  /*
  if (myid==0){
    n = 1; 
    for (depth=0; depth<=max_depth; depth++){
      for (child=0; child<n; child++){

	Na = Num_Atoms_in_Child[depth][child];

	for (k=0; k<Na; k++){
	  m = List_AN_in_Child[depth][child][k];
	  t = List_T_in_Child[depth][child][k];
	  printf("depth=%2d child=%2d k=%2d m=%2d t=%15.12f\n",depth,child,k,m,t);
	}
      }
      n *= 2;
    }
  }
  */

  /***************************************************************
                 allocation of atoms to processes
  ***************************************************************/ 

  /*
  sorting of atoms in each child at max_depth. 
  if the sorting is not performed, the force calculations
  related to HNL3 and 4B will be failed.
  */

  n = 1; 
  for (depth=0; depth<max_depth; depth++) n *= 2;

  for (child=0; child<n; child++){
    Na = Num_Atoms_in_Child[max_depth][child];
    qsort_int1((long)Na,List_AN_in_Child[depth][child]);
  }  

  n = 1; 
  for (depth=0; depth<max_depth; depth++) n *= 2;

  /* set ID2ID */

  ID2ID = (int*)malloc(sizeof(int)*numprocs);

  if ( (Solver==8 || Solver==11) && atomnum<=numprocs){

    for (ID=0; ID<numprocs; ID++){
      ID2ID[ID] = (int)( (long int)((long int)numprocs*(long int)ID)/(long int)atomnum);
    }
  }

  else{
    for (ID=0; ID<numprocs; ID++){
      ID2ID[ID] = ID;
    }
  } 

  /* set G2ID, M2G, Matomnum, and WMatomnum */

  Matomnum = 0;
  WMatomnum = 0.0;
  ID = 0;

  for (child=0; child<n; child++){

    Na = Num_Atoms_in_Child[max_depth][child];

    if (Na!=0){ 

      for (k=0; k<Na; k++){
	m = List_AN_in_Child[max_depth][child][k];
	G2ID[m] = ID2ID[ID]; 
      }

      if (myid==ID2ID[ID]){

	Matomnum = Na; 
	if (alloc_first[10]==0) free(M2G);

	M2G = (int*)malloc(sizeof(int)*(Matomnum+2));
	alloc_first[10] = 0;

	for (k=0; k<Na; k++){
	  m = List_AN_in_Child[max_depth][child][k];
	  M2G[k+1] = m;
	}

	WMatomnum = 0.0;
	for (k=0; k<Na; k++){
	  m = List_AN_in_Child[max_depth][child][k];
	  WMatomnum += weight[m];
	}
      }

      ID++; 
    }
  }

  /* allocate of M2G for Divide_Conquer_LNO */

  if (Matomnum==0){
    if (alloc_first[10]==0) free(M2G);
    M2G = (int*)malloc(sizeof(int)*2);
    alloc_first[10] = 0;
  }

  /****************************************
    find Max_Matomnum, MatomN and WMatomN
  ****************************************/

  MatomN = (int*)malloc(sizeof(int)*numprocs);
  WMatomN = (double*)malloc(sizeof(double)*numprocs);

  MatomN[myid]  = Matomnum;
  WMatomN[myid] = WMatomnum;

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&MatomN[ID],  1, MPI_INT, ID, mpi_comm_level1);
    MPI_Bcast(&WMatomN[ID], 1, MPI_DOUBLE, ID, mpi_comm_level1);
  } 

  /* find Max_Matomnum */

  Max_Matomnum = 0;
  for (ID=0; ID<numprocs; ID++){
    if (Max_Matomnum<MatomN[ID]) Max_Matomnum = MatomN[ID];
  }     

  /*********************************************
              print the information 
  *********************************************/

  if (myid==Host_ID && 1<=MD_iter && 0<level_stdout){
    printf("\n");
    printf("*******************************************************\n"); 
    printf("  Allocation of atoms to proccesors at MD_iter=%5d     \n", MD_iter );
    printf("*******************************************************\n\n"); 
  }

  for (ID=0; ID<numprocs; ID++){

    if (myid==Host_ID && 1<=MD_iter && 0<level_stdout){
      printf(" proc = %3d  # of atoms=%4d  estimated weight=%16.5f\n",
	     ID,MatomN[ID],WMatomN[ID]);
    }
  }     

  if (myid==Host_ID && 1<=MD_iter && 0<level_stdout) printf("\n\n\n");

  /****************************************
         initialize time_per_atom
  ****************************************/

  for (k=1; k<=atomnum; k++) time_per_atom[k] = 0.0;

  /*
  if (myid==Host_ID){

    int i,k;
    char AtomName[38][10]=
      {"E","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si",
       "P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
       "Ga","Ge","As","Se","Br","Kr"
      };

    for (i=1; i<=atomnum; i++){
      k = (G2ID[i])%36+4;
      printf("%5s  %15.12f %15.12f %15.12f  0.0 0.0 0.0\n",
               AtomName[k],Gxyz[i][1]*BohrR,Gxyz[i][2]*BohrR,Gxyz[i][3]*BohrR);
    }
  }
  MPI_Finalize();
  exit(0);
  */
  
  /*
  {
    int i1,i2,k1,k2;
    double dx,dy,dz,r,rmax;

    rmax = 0.0;
    for (i1=1; i1<=Matomnum; i1++){
      k1 = M2G[i1];

      for (i2=1; i2<=Matomnum; i2++){
        k2 = M2G[i2];

        dx = Gxyz[k1][1] - Gxyz[k2][1];
        dy = Gxyz[k1][2] - Gxyz[k2][2];
        dz = Gxyz[k1][3] - Gxyz[k2][3];
        r = sqrt(dx*dx+dy*dy+dz*dz);
        if (rmax<r) rmax = r;
      }
    }

    printf("ABC1 myid=%2d rmax=%15.12f\n",myid,rmax);fflush(stdout);
  }

  MPI_Finalize();
  exit(0);
  */


  /****************************************
            freeing of arrays
  ****************************************/

  free(ID2ID);

  free(WMatomN);
  free(MatomN);

  free(List_AN_in_Child[0][0]);
  free(List_T_in_Child[0][0]);
  free(List_AN_in_Child[0]);
  free(List_T_in_Child[0]);

  n = 1; 
  for (depth=0; depth<max_depth; depth++){

    for (i=0; i<n*2; i++){
      free(List_T_in_Child[depth+1][i]);
      free(List_AN_in_Child[depth+1][i]);
    }
    free(List_T_in_Child[depth+1]);
    free(List_AN_in_Child[depth+1]);

    n *= 2;
  }
  free(List_T_in_Child);
  free(List_AN_in_Child);

  free(weight);
  free(ko);

  for (i=0; i<4; i++){
    free(IMT[i]);
  }
  free(IMT);

  for (depth=0; depth<(max_depth+1); depth++){
    free(Num_Atoms_in_Child[depth]);
  }
  free(Num_Atoms_in_Child);

  for (depth=0; depth<(max_depth+1); depth++){
    free(Num_Procs_in_Child[depth]);
  }
  free(Num_Procs_in_Child);
}


#pragma optimization_level 1
void Allocation_Species()
{
  int i,num1,num2;
  int numprocs,myid;
  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /****************************************************
                  partition of species
  ****************************************************/

  for (i=0; i<numprocs; i++){
    Species_Top[i] = 0;
    Species_End[i] = 0;
  }

  if (SpeciesNum<numprocs) Num_Procs2 = SpeciesNum;
  else                     Num_Procs2 = numprocs;

  num1 = SpeciesNum/Num_Procs2;
  num2 = SpeciesNum%Num_Procs2;
  for (i=0; i<Num_Procs2; i++){
    Species_Top[i] = num1*i;
    Species_End[i] = num1*(i + 1) - 1;
  }
  if (num2!=0){
    for (i=0; i<num2; i++){
      Species_Top[i] = Species_Top[i] + i;
      Species_End[i] = Species_End[i] + i + 1;
    }
    for (i=num2; i<Num_Procs2; i++){
      Species_Top[i] = Species_Top[i] + num2;
      Species_End[i] = Species_End[i] + num2;
    }
  }

  if (myid==Host_ID && 2<=level_stdout){
    for (i=0; i<Num_Procs2; i++){
      printf("proc=%4d  Species_Top=%4d  Species_End=%4d\n",
	     i,Species_Top[i],Species_End[i]);
    }
  }

  if (myid<Num_Procs2)
    MSpeciesNum = Species_End[myid] - Species_Top[myid] + 1;
  else 
    MSpeciesNum = 0;

  if (myid==Host_ID && 2<=level_stdout){
    printf("myid=%i  MSpeciesNum=%i\n",myid,MSpeciesNum);
  }

}









/*
void Output_Atom2CPU()
{

  int i,numprocs;
  char file_A2C[YOUSO10] = ".A2C";
  FILE *fp;

  MPI_Comm_size(mpi_comm_level1,&numprocs);

  fnjoint(filepath,filename,file_A2C);

  if ((fp = fopen(file_A2C,"w")) != NULL){

    fprintf(fp,"\n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"               Allocation of atoms to CPUs                 \n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"***********************************************************\n");
    fprintf(fp,"\n");

    fprintf(fp,"   Average weight = %5.4f\n\n",eachw);
    fprintf(fp,"   CPU    Atoms    Top     End     CPU_Weight\n");
    for (i=0; i<numprocs; i++){
      fprintf(fp,"  %4d   %4d    %4d    %4d    %4d\n",
              i,
              Gatom_End[i]-Gatom_Top[i]+1,
              Gatom_Top[i], 
              Gatom_End[i],
              CPU_Weight[i]);
    } 
    fclose(fp);
  }
  else
    printf("Failure of saving the Set_Allocate_Atom2CPU.\n");
}
*/
