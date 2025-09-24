#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"


#define debug   0 /* ??? */ 
#define debug1  0 /* for general verbose output */
#define debug2  0 /* for symmetry with atomic arrangement */ 
#define debug3  0 /* for lattice vector symmetry */ 
#define debug4  0 /* for generating k points */
#define debug5  0 /* open the files for comparision */
double pi,smallvalue;

void MP_Special_Kpt(/* input */
 		    int atom_num, /* Number of total atoms in cell. Read from OpenMX */  
		    int atom_type, /* Number of elements in cell. Read from OpenMX */  
		    /* Lattice vectors. Read from openMX 
		       VectorA --> Read_in_Lattice[0][0:2]
		       VectorB --> Read_in_Lattice[1][0:2]
		       VectorC --> Read_in_Lattice[2][0:2]
		    */  
		    double **Read_in_Lattice,
		    double **Read_in_Atom_Pos,
                    int inversion_flag,
		    int knum_i, int knum_j, int knum_k, /* number of k-points along a,b,c-axis */ 
		    int *Read_in_Species
		    /* implicit output 
		    num_non_eq_kpt,
		    NE_KGrids1, NE_KGrids2, NE_KGrids3,
		    NE_T_k_op */ );



double Cal_Cell_Volume(double **lattice_vector);
void Cal_Reciprocal_Vectors(double **latt_vec, double **rec_latt_vec);
void Symmetry_Operation_Transform(int ***InputSymop, int ***OutputSymop, int op_num, 
                                  double **In_Lattice, double **Out_Lattice);
void Chk_Shorter_Lattice_Vector(double **rlatt);
int Bravais_Type(double **lattice_vector,double *cell_parameters);
int Finding_Bravais_Lattice_Type(double **lattice_vector, double *cell_parameters);
void Matrix_Copy23(int **source,int n_source,int ***symop, int k_symop);
void Matrix_Copy32(int **target,int n_target,int ***symop, int k_symop);
void Matrix_Copy22(int **source,int **target, int dim);
int Matrix_Equal(int **m1, int **m2, int dim);
void Matrix_Productor(int dim, int **m1, int **m2, int **pro);
int Symmtry_Operator_Generation(int ***symgen,int gen_num,int ***symop);
int Bravais_Lattice_Symmetry(int bravais_type, int ***symop);
void Ascend_Ordering(double *xyz_value, int *ordering, int tot_atom);
void Ordering_Atomic_Position(double **atomic_position, int start_atom_indx, int end_atom_indx);
int Chk_Pure_Point_Group_Op(int **exop, double **atomic_position, int *atom_species, 
			    int atom_num, int atom_type, double *trans_vec);
void Get_Symmetry_Operation(int ***symop, int *opnum, double **atomic_position, int *atom_species,
			    int atom_num, int atom_type, int *num_pnt_op, double **trans_op_vec);
int Chk_Primitive_Cell(int bravais_type, double *cell_parameters, double **lattice_vector,
		       double **atomic_position, int *atom_species,int atom_num, int atom_type,
		       double **platt, double **ptran_vec, double *pcell_parameters,int *npcell);
int Generate_MP_Special_Kpt(int knum_i, int knum_j, int knum_k, int ***sym_op, int op_num,
			    int ***pureGsym, int pureG_num, double *shift, 
                            double *KGrids1, double *KGrids2, double *KGrids3, int *T_k_op);
void Atomic_Coordinates_Transform(double **Read_in_Atom_Pos, double **atomic_position, int atom_num, 
                                  double **Read_in_Lattice, double **lattice_vector);














void Generating_MP_Special_Kpt(/* input */
                               int atomnum,
			       int SpeciesNum,
			       double tv[4][4],
			       double **Gxyz,
                               double *InitN_USpin, 
                               double *InitN_DSpin,
                               double criterion_geo,
                               int SpinP_switch,
			       int *WhatSpecies,
			       int knum_i, int knum_j, int knum_k
                               /* implicit output 
                               num_non_eq_kpt,
                               NE_KGrids1, NE_KGrids2, NE_KGrids3,
                               NE_T_k_op */ )
{
  int i,j,k,wan;
  int myid,numprocs;
  int MG_SpeciesNum;
  int *MG_WhatSpecies;
  int inversion_flag;
  double **Read_in_Lattice;
  double **Read_in_Atom_Pos, **rlatt;
  int *Read_in_Species;
  int *check_flag;

  /***************************************************************
                           print stdout
  ***************************************************************/

  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    printf("\n");
    printf("*******************************************************\n");
    printf("      Generation of Monkhorst-Pack k-points\n");
    printf("*******************************************************\n\n");
  }

  /***************************************************************
                      allocation of arrays
  ***************************************************************/
  
  Read_in_Atom_Pos = (double**)malloc(sizeof(double*)*(atomnum+1));
  for (i=0; i<(atomnum+1); i++){
    Read_in_Atom_Pos[i] = (double*)malloc(sizeof(double)*3);
  }

  Read_in_Lattice = (double**)malloc(sizeof(double*)*3);
  for(i=0; i<3; i++){
    Read_in_Lattice[i]=(double*)malloc(sizeof(double)*3);
  }

  rlatt = (double**)malloc(sizeof(double*)*3);
  for(i=0; i<3; i++){
    rlatt[i]=(double*)malloc(sizeof(double)*3);
  }

  /* set inversion_flag */

  inversion_flag = (int)(0*(SpinP_switch==0 || SpinP_switch==1)+(SpinP_switch==3));

  /* copy tv to Read_in_Lattice */

  for (i=0; i<3; i++){
    for (j=0; j<3; j++){
      Read_in_Lattice[i][j] = tv[i+1][j+1]; 
      rlatt[i][j]=0.0;
    }
  }

  /* calculation of reciprocal lattice vectors */

  pi=PI;
  Cal_Reciprocal_Vectors(Read_in_Lattice, rlatt);

  for(i=0;i<3;i++){
    rlatt[i][0]=rlatt[i][0]/2.0/pi;
    rlatt[i][1]=rlatt[i][1]/2.0/pi;
    rlatt[i][2]=rlatt[i][2]/2.0/pi;
  }

  if(debug1==1){
    printf("Reciprocal lattice:\n");
    for(i=0;i<3;i++){
      printf("K%1d (%10.5f, %10.5f, %10.5f)\n",i+1,rlatt[i][0],rlatt[i][1],rlatt[i][2]);
    }
  }

  /***************************************************************
      redefinition of species by considering magnetic structure 
  ***************************************************************/

  /* non-magnetic case */

  if (SpinP_switch==0){

    MG_WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));
    MG_SpeciesNum = SpeciesNum;
    Read_in_Species = (int*)malloc(sizeof(int)*(MG_SpeciesNum+1));

    for (i=1; i<=atomnum; i++){
      MG_WhatSpecies[i] = WhatSpecies[i];
    }
  }   

  /* collinear magnetic case */
  else if (SpinP_switch==1){

    /* allocation of array */

    MG_WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));

    check_flag = (int*)malloc(sizeof(int)*(atomnum+1));
    for (i=1; i<=atomnum; i++) check_flag[i] = 0;

    k = 0;

    for (i=1; i<=atomnum; i++){

      if (check_flag[i]==0){ 

        wan = WhatSpecies[i];
        MG_WhatSpecies[i] = k;
        check_flag[i] = 1;

        for (j=i+1; j<=atomnum; j++){

          if (   check_flag[j]==0
              && 
                 WhatSpecies[j]==wan 
              &&
                 (criterion_geo>fabs(InitN_USpin[i]-InitN_USpin[j])) 
              &&
                 (criterion_geo>fabs(InitN_DSpin[i]-InitN_DSpin[j])) 
	     )
            {
              MG_WhatSpecies[j] = k;
              check_flag[j] = 1;
            }    
	}

        k++; 
      }
    }

    MG_SpeciesNum = k;
    Read_in_Species = (int*)malloc(sizeof(int)*(MG_SpeciesNum+1));

    /* freeing of array */
    free(check_flag);
  }

  /* non-collinear magnetic case */
  else if (SpinP_switch==3){

    MG_WhatSpecies = (int*)malloc(sizeof(int)*(atomnum+1));
    MG_SpeciesNum = atomnum;
    Read_in_Species = (int*)malloc(sizeof(int)*(MG_SpeciesNum+1));

    for (i=1; i<=atomnum; i++){
      MG_WhatSpecies[i] = i-1;
    }
  }

  /*
  printf("MG_SpeciesNum=%2d\n",MG_SpeciesNum);
  for (i=1; i<=atomnum; i++){
    printf("i=%3d MG_WhatSpecies=%3d\n",i,MG_WhatSpecies[i]);
  }
  */

  /***************************************************************
      copy tv to Read_in_Lattice
  ***************************************************************/

  k = 0;
  for (i=0; i<MG_SpeciesNum; i++){
    for (j=1; j<=atomnum; j++){

      wan = MG_WhatSpecies[j];

      if (wan==i){
        Read_in_Atom_Pos[k][0] = Gxyz[j][1]*rlatt[0][0]+Gxyz[j][2]*rlatt[0][1]+Gxyz[j][3]*rlatt[0][2];
        Read_in_Atom_Pos[k][1] = Gxyz[j][1]*rlatt[1][0]+Gxyz[j][2]*rlatt[1][1]+Gxyz[j][3]*rlatt[1][2];
        Read_in_Atom_Pos[k][2] = Gxyz[j][1]*rlatt[2][0]+Gxyz[j][2]*rlatt[2][1]+Gxyz[j][3]*rlatt[2][2];
        k++;
      }
    }

    Read_in_Species[i] = k;
  }

  if(debug3==1){
    for(i=0;i<MG_SpeciesNum;i++){
      if(i==0){
	k=0;
      }else{
	k=Read_in_Species[i-1];
      }
      for(j=k;j<Read_in_Species[i];j++){
	printf("Species %2d  (%10.5f,%10.5f,%10.5f)\n",i,Read_in_Atom_Pos[j][0],Read_in_Atom_Pos[j][1],Read_in_Atom_Pos[j][2]);
      }
    }
  } 

  /***************************************************************
      call Generating_MP_Special_Kpt
  ***************************************************************/

  smallvalue = criterion_geo;
  MP_Special_Kpt(/* input */
                 atomnum,
                 MG_SpeciesNum,
                 Read_in_Lattice,
                 Read_in_Atom_Pos,
                 inversion_flag,
                 knum_i,knum_j,knum_k,
                 Read_in_Species 
		 /* implicit output 
		 num_non_eq_kpt,
		 NE_KGrids1, NE_KGrids2, NE_KGrids3,
		 NE_T_k_op */ );

  /***************************************************************
      freeing of arrays
  ***************************************************************/

  free(Read_in_Species);
  free(MG_WhatSpecies);

  for (i=0; i<(atomnum+1); i++){
    free(Read_in_Atom_Pos[i]);
  }
  free(Read_in_Atom_Pos);

  for(i=0; i<3; i++){
    free(rlatt[i]);
  }
  free(rlatt);

  for(i=0; i<3; i++){
    free(Read_in_Lattice[i]);
  }
  free(Read_in_Lattice);


}




void MP_Special_Kpt(/* input */
 		    int atom_num, /* Number of total atoms in cell. Read from OpenMX */  
		    int atom_type, /* Number of elements in cell. Read from OpenMX */  
		    /* Lattice vectors. Read from openMX 
		       VectorA --> Read_in_Lattice[0][0:2]
		       VectorB --> Read_in_Lattice[1][0:2]
		       VectorC --> Read_in_Lattice[2][0:2]
		    */  
		    double **Read_in_Lattice,
		    double **Read_in_Atom_Pos,
                    int inversion_flag,
		    int knum_i, int knum_j, int knum_k, /* number of k-points along a,b,c-axis */ 
		    int *Read_in_Species
		    /* implicit output 
		    num_non_eq_kpt,
		    NE_KGrids1, NE_KGrids2, NE_KGrids3,
		    NE_T_k_op */ )
{
  /* matrix for symmetry operation. 
     rsym_op is for storing crystal symmetry operation matrix defined in original lattice vectors space
     Gsym_op is for those defined in reciprocal space of original lattice vectors
     pureGsys is for those from puer reciprocal lattice vectors
     ksym_op is the symmetry operation matrix in reciprocal space including time-inversion symmetry.
     sym_op is for temporary storage. 
  */
  int ***sym_op, ***rsym_op, ***Gsym_op, ***pureGsym, ***ksym_op; 
  /* number of symmetry operation matrix */
  int op_num, op_num_max, op_num_original, ksym_num, pureG_num; 
  /* number of pure point group operation matrix */
  int num_pure_pnt_op;
  /* Translation operation vectors associated with pure point group operation found from
     real crystal structure 
  */
  double **trans_op_vec;
  /* for detecting and applying inversion symmetry  */ 
  int inversion_sym, **inv, **tmpsym, **tmpisym;
    
  /* The following is defined to read VASP files and do comparision*/  
  int ***vasp_symop;
  int vasp_opnum;
  double **vasp_transopvec;     
  int vasp_numpurepntop;
  int opok,transok;
  int kopen,iopen,jopen;


  /* total number of non-equivalent k points. Return to openMX */ 
  int kpt_num;  
  /* coordinates of non-equivalent k points. Return to openMX */  
  /*
  double *KGrids1, *KGrids2, *KGrids3; 
  */
  /* T_k_op is weight of each non-equivalent k point. Return to openMX */ 
  /*
  int *T_k_op; 
  */

  int *tmpWeight; 

  int shift_keep;/* shift is for shifting of k points */
  /* Temporary array used for generating k points*/
  double *tmpK1,*tmpK2,*tmpK3;
  double ktmp[3], kx, ky, kz, shift[3], tmp_shift[3];

  /* rlatt is used for reciprocal lattice vector.
     platt is used for primitive unit cell vector  
     lattice_vector is used for storing the interchanged 
     or updated lattice vectors after identifying its 
     bravaise lattice type. */
  double **rlatt, **platt, **lattice_vector, **klatt; 
  
  /* bravais_type is the Bravais lattice type of inputted lattice vectors
     pcell_brav is the primitive uint cell's bravais lattice type. */     
  int bravais_type, pcell_brav; 
  /* Number of primitive unit cell contained in the original lattice */
  int npcell;
  /*cell_parameters is for unit cell's lattice parameters: 
    cell_parameters[0] length of a vector
    cell_parameters[1] length of b vector
    cell_parameters[2] length of c vector
    cell_parameters[3] angle between b and c vector in radian (alpha)
    cell_parameters[4] angle between a and c vector in radian (beta)
    cell_parameters[5] angle between a and b vector in radian (gamma)
    pcell_parameters is for primitive cell parameters.     
  */ 
  double pcell_parameters[6], cell_parameters[6];
  /* volume of unit cell */  
  double cell_vol; 
  /* Used when trying to find the primitive unit cell.
     It has the same structure as atomic_position while 
     it is 2 elements larger.
  */    
  double **ptran_vec;
  /* Temporary array used for finding lattice type and primitive unit cell */  
  double con_cell[6], cos1,cos2,cos3;
    
    
  /* Read_in_Atom_Pos: Atomic postions in crystal structure read from openMX.
     These corrdinats are supposed to be in fractional coordinates and all atoms
     of same elements should be stored successively. The last atom of each element
     is indicated by Read_in_Species array.
     The map is as follwoing:
     atom index       x,y,z
     |              |
     |              |
     \|/            \|/
     Read_in_Atom_Pos[atom0_of_element_0][0:2]
     Read_in_Atom_Pos[atom1_of_element_0][0:2]
     Read_in_Atom_Pos[atom2_of_element_0][0:2]
     Read_in_Atom_Pos[atom0_of_element_1][0:2]  <--- Read_in_Species[element0]
     Read_in_Atom_Pos[atom1_of_element_1][0:2]
     Read_in_Atom_Pos[atom2_of_element_1][0:2]
     Read_in_Atom_Pos[atom3_of_element_1][0:2]
     Read_in_Atom_Pos[atom0_of_element_2][0:2]  <--- Read_in_Species[element1]
     Read_in_Atom_Pos[atom1_of_element_2][0:2]
     
  */
    
  /* atomic_position[atom_indx][0:2] coordinates of atom with index as atom_indx. */
  double **atomic_position;     
  /* This array stores the atomic index of different species in atomic_position array.    
     Atoms of element 0 is stored with index from 0 to atom_species[0]-1 in atomic_position[atom_indx][3].
     Atoms of element 1 is stored from atom_species[0] to atom_species[1]-1
     Atoms of element i is stored from atom_species[i-1] to atom_species[i]-1.
     Atoms with atom_indx equal to j can be termined to be element i as following:
     Searching atom_species[] array, if atom_species[i-1]< j+1 <=atom_species[i], then this
     atom belongs to species i
  */
  int *atom_species;           
  /* Temporary used variables */                                 
  int i,j,k,r,p,s;
  int whetherNonEquiv;
  char c; 
  FILE *fp;
  int myid,numprocs;
    
  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  pi=PI; 
		
  lattice_vector=(double**)malloc(sizeof(double*)*3);
  rlatt=(double**)malloc(sizeof(double*)*3);
  platt=(double**)malloc(sizeof(double*)*3);
  klatt=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    lattice_vector[i]=(double*)malloc(sizeof(double)*3);
    platt[i]=(double*)malloc(sizeof(double)*3);
    rlatt[i]=(double*)malloc(sizeof(double)*3);
    klatt[i]=(double*)malloc(sizeof(double)*3);
  }

  op_num_max = 48;

  sym_op = (int***)malloc(sizeof(int**)*op_num_max);
  for(k=0; k<op_num_max; k++){
    sym_op[k] = (int**)malloc(sizeof(int*)*3);
    for(i=0; i<3; i++){
      sym_op[k][i] = (int*)malloc(sizeof(int)*3);
      for(j=0; j<3; j++) sym_op[k][i][j] = 0;
    }
  }

  pureGsym = (int***)malloc(sizeof(int**)*op_num_max);
  for(k=0; k<op_num_max; k++){
    pureGsym[k] = (int**)malloc(sizeof(int*)*3);
    for(i=0; i<3; i++){
      pureGsym[k][i] = (int*)malloc(sizeof(int)*3);
      for(j=0; j<3; j++) pureGsym[k][i][j] = 0;
    }
  }

  trans_op_vec = (double**)malloc(sizeof(double*)*op_num_max);
  for(k=0; k<op_num_max; k++){
    trans_op_vec[k]=(double*)malloc(sizeof(double)*3);
    for(i=0; i<3; i++) trans_op_vec[k][i] = 0.0;
  }

  if(atom_num<atom_type){
    printf("Error!\nAtomic number should not be smaller than atomic species number.\n");
    return;
  }
    
  atomic_position = (double**)malloc(sizeof(double*)*(atom_num+2));
  for(k=0; k<(atom_num+2); k++){
    atomic_position[k] = (double*)malloc(sizeof(double)*3);
    for(i=0; i<3; i++) atomic_position[k][i] = 0.0;
  }

  ptran_vec=(double**)malloc(sizeof(double*)*(atom_num+2));
  for(k=0; k<(atom_num+2); k++){
    ptran_vec[k]=(double*)malloc(sizeof(double)*3);
    for(i=0; i<3; i++) ptran_vec[k][i] = 0.0;
  }

  atom_species=(int*)malloc(sizeof(int)*atom_type);
  for(i=0; i<atom_type; i++) atom_species[i]=0;

  for(i=0;i<atom_num;i++){
    for(j=0;j<3;j++){
      atomic_position[i][j]=Read_in_Atom_Pos[i][j];
    }
  }

  for(i=0;i<atom_type;i++){
    atom_species[i]=Read_in_Species[i];
  }

  /*******************************************
      for debugging 
  *******************************************/

  if (debug5){

    r=15;
    switch(r){
    case 1:
      /* sc */
      Read_in_Lattice[0][0]=1.0;    Read_in_Lattice[0][1]=0.0;    Read_in_Lattice[0][2]=0.0;
      Read_in_Lattice[1][0]=0.0;    Read_in_Lattice[1][1]=1.0;    Read_in_Lattice[1][2]=0.0;
      Read_in_Lattice[2][0]=0.0;    Read_in_Lattice[2][1]=0.0;    Read_in_Lattice[2][2]=1.0;
      if((fp=fopen("sc.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 2:
      /*bcc */    
      Read_in_Lattice[0][0]=-0.5;    Read_in_Lattice[0][1]=0.5;    Read_in_Lattice[0][2]=0.5;
      Read_in_Lattice[1][0]=0.5;    Read_in_Lattice[1][1]=-0.5;    Read_in_Lattice[1][2]=0.5;
      Read_in_Lattice[2][0]=0.5;    Read_in_Lattice[2][1]=0.5;    Read_in_Lattice[2][2]=-0.5;
      if((fp=fopen("bcc.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;        
    case 3:
      /* fcc */
      Read_in_Lattice[0][0]=0.0;    Read_in_Lattice[0][1]=1.5;    Read_in_Lattice[0][2]=1.5;
      Read_in_Lattice[1][0]=1.5;    Read_in_Lattice[1][1]=0.0;    Read_in_Lattice[1][2]=1.5;
      Read_in_Lattice[2][0]=1.5;    Read_in_Lattice[2][1]=1.5;    Read_in_Lattice[2][2]=0.0;
      if((fp=fopen("fcc.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 4:
      /* hP */
      Read_in_Lattice[0][0]=1.0;    Read_in_Lattice[0][1]=-1.0;    Read_in_Lattice[0][2]=0.0;
      Read_in_Lattice[1][0]=-1.0;    Read_in_Lattice[1][1]=0.0;    Read_in_Lattice[1][2]=1.0;
      Read_in_Lattice[2][0]=-1.0;    Read_in_Lattice[2][1]=-1.0;    Read_in_Lattice[2][2]=-1.0;
      if((fp=fopen("hP.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;				
    case 5:
      /* tP */
      Read_in_Lattice[0][0]=1.0;    Read_in_Lattice[0][1]=0.0;    Read_in_Lattice[0][2]=0.0;
      Read_in_Lattice[1][0]=0.0;    Read_in_Lattice[1][1]=1.0;    Read_in_Lattice[1][2]=0.0;
      Read_in_Lattice[2][0]=0.0;    Read_in_Lattice[2][1]=0.0;    Read_in_Lattice[2][2]=2.0;
      if((fp=fopen("tP.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 6:
      /* tI */                                                                                
      Read_in_Lattice[0][0]=-0.5;    Read_in_Lattice[0][1]=0.5;    Read_in_Lattice[0][2]=1.0;
      Read_in_Lattice[1][0]=0.5;    Read_in_Lattice[1][1]=-0.5;    Read_in_Lattice[1][2]=1.0;
      Read_in_Lattice[2][0]=0.5;    Read_in_Lattice[2][1]=0.5;    Read_in_Lattice[2][2]=-1.0;
      if((fp=fopen("tI.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 7:
      /* hR  */
      Read_in_Lattice[0][0]=0.7848063699;    Read_in_Lattice[0][1]=0.2957148683;    Read_in_Lattice[0][2]=0.544639035;
      Read_in_Lattice[1][0]=0.0;    Read_in_Lattice[1][1]=0.8386705679;    Read_in_Lattice[1][2]=0.544639035;
      Read_in_Lattice[2][0]=0.0;    Read_in_Lattice[2][1]=0.0;    Read_in_Lattice[2][2]=1.0;
      if((fp=fopen("hR.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 8:
      /* oP */
      Read_in_Lattice[0][0]=4.0;    Read_in_Lattice[0][1]=0.0;    Read_in_Lattice[0][2]=0.0;
      Read_in_Lattice[1][0]=0.0;    Read_in_Lattice[1][1]=2.0;    Read_in_Lattice[1][2]=0.0;
      Read_in_Lattice[2][0]=0.0;    Read_in_Lattice[2][1]=0.0;    Read_in_Lattice[2][2]=3.0;

      if((fp=fopen("oP.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 9:
      /* oI  */
      Read_in_Lattice[0][0]=-0.5;    Read_in_Lattice[0][1]=1.0;    Read_in_Lattice[0][2]=1.5;
      Read_in_Lattice[1][0]=0.5;    Read_in_Lattice[1][1]=-1.0;    Read_in_Lattice[1][2]=1.5;
      Read_in_Lattice[2][0]=0.5;    Read_in_Lattice[2][1]=1.0;    Read_in_Lattice[2][2]=-1.5;
      if((fp=fopen("oI.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 10:
      /* oF  */
      Read_in_Lattice[0][0]=0.0;    Read_in_Lattice[0][1]=1.0;    Read_in_Lattice[0][2]=1.5;
      Read_in_Lattice[1][0]=0.5;    Read_in_Lattice[1][1]=0.0;    Read_in_Lattice[1][2]=1.5;
      Read_in_Lattice[2][0]=0.5;    Read_in_Lattice[2][1]=1.0;    Read_in_Lattice[2][2]=0.0;
      if((fp=fopen("oF.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;    		
    case 11:
      /* oC  */
      Read_in_Lattice[0][0]=1.5;    Read_in_Lattice[0][1]=1.0;    Read_in_Lattice[0][2]=0.0;
      Read_in_Lattice[1][0]=-1.5;    Read_in_Lattice[1][1]=1.0;    Read_in_Lattice[1][2]=0.0;
      Read_in_Lattice[2][0]=0.0;    Read_in_Lattice[2][1]=0.0;    Read_in_Lattice[2][2]=3.0;
      if((fp=fopen("oC.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 12:
      /* mP */
      Read_in_Lattice[0][0]=1.0;    Read_in_Lattice[0][1]=-0.5;    Read_in_Lattice[0][2]=0.0;
      Read_in_Lattice[1][0]=-0.2;    Read_in_Lattice[1][1]=2.0;    Read_in_Lattice[1][2]=0.0;
      Read_in_Lattice[2][0]=0.0;    Read_in_Lattice[2][1]=0.0;    Read_in_Lattice[2][2]=3.0;
      if((fp=fopen("mP.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 13:
      /* mB */
      Read_in_Lattice[0][0]=0.0;    Read_in_Lattice[0][1]=1.0;    Read_in_Lattice[0][2]=-1.5;
      Read_in_Lattice[1][0]=0.0;    Read_in_Lattice[1][1]=1.0;    Read_in_Lattice[1][2]=1.5;
      Read_in_Lattice[2][0]=0.9993908270;    Read_in_Lattice[2][1]=-0.0348994967;    Read_in_Lattice[2][2]=0.0;
      if((fp=fopen("mB.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    case 14:
      /* aP triclinic */
      Read_in_Lattice[0][0]= 0.7546287487;     Read_in_Lattice[0][1]=-0.1317556090;     Read_in_Lattice[0][2]= 0.6427876097;
      Read_in_Lattice[1][0]= 0.7546287487;     Read_in_Lattice[1][1]= 1.3309517942;     Read_in_Lattice[1][2]=-0.9932156702;
      Read_in_Lattice[2][0]=-0.7546287487;     Read_in_Lattice[2][1]= 1.5944630122;     Read_in_Lattice[2][2]= 0.7212091104;

      if((fp=fopen("aP.dat","wt"))==NULL){
	printf("Error in open file\n");
	return;
      }
      break;
    default:
      if((fp=fopen("sym.dat","wt"))==NULL){
        printf("Error in open file\n");
        return;
      }
      break;
    }/*End of swich bravais_type and initialize Read_in_Lattice */
		
    /*******************************************
      for debugging 
    *******************************************/
  }

	
  /* First step: Determin the bravais lattice type*/
  /* Using lattice_vector instead of original one, since lattice_vector
     would be changed according to convential choice of lattice vectors */
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      lattice_vector[i][j]=Read_in_Lattice[i][j];
    }
  }
  for(i=0;i<6;i++){
    cell_parameters[i]=0.0;
    pcell_parameters[i]=0.0;
  }
  bravais_type=Finding_Bravais_Lattice_Type(lattice_vector,cell_parameters);

  if (myid==Host_ID){
  
    switch(bravais_type){
    case 1:
      /* sc */
      printf("  Found a simple cubic lattice.\n");
      break;
    case 2:
      /*bcc */    
      printf("  Found a body-centered cubic lattice.\n");
      break;        
    case 3:
      /* fcc */
      printf("  Found a face-centered cubic lattice.\n");
      break;
    case 4:
      /* hP */
      printf("  Found a primitive hexagonal lattice.\n");
      break;				
    case 5:
      /* tP */
      printf("  Found a primitive tetragonal lattice.\n");
      break;
    case 6:
      /* tI */                                                                                
      printf("  Found a body-centered tetragonal lattice.\n");
      break;
    case 7:
      /* hR  */
      printf("  Found a rhombohedral lattice.\n");
      break;
    case 8:
      /* oP */
      printf("  Found a primitive orthorhombic lattice.\n");
      break;
    case 9:
      /* oI  */
      printf("  Found a body-centered orthorhombic lattice.\n");
      break;
    case 10:
      /* oF  */
      printf("  Found a face-centered orthorhombic lattice.\n");
      break;    		
    case 11:
      /* oC  */
      printf("  Found a base-centered orthorhombic lattice.\n");
      break;
    case 12:
      /* mP */
      printf("  Found a primitive monoclinic lattice.\n");
      break;
    case 13:
      /* mB */
      printf("  Found a base-centered monoclinic lattice.\n");
      break;
    case 14:
      /* aP triclinic */
      printf("  Found a primitive triclinic lattice.\n");
      break;
    }
  
    printf("  Lattice parameters are:\n");
    for(i=0;i<6;i++){
      if(cell_parameters[i]!=0.0){
	if(i==0){ 
	  printf("  a=%10.5f\n",cell_parameters[i]);
	}else if(i==1){
	  printf("  b/a=%10.5f\n",cell_parameters[i]);
	}else if(i==2){
	  printf("  c/a=%10.5f\n",cell_parameters[i]);
	}else{
	  if(fabs(cos(cell_parameters[i]))>smallvalue){
	    if(i==3){
	      printf("  Alpha is %10.5f and cos(Alpha) is %10.5f.\n",cell_parameters[i]/pi*180.0,cos(cell_parameters[i]));
	    }else if(i==4){
	      printf("  Beta is %10.5f and cos(Beta) is %10.5f.\n",cell_parameters[i]/pi*180.0,cos(cell_parameters[i]));
	    }else{
	      printf("  Gamma is %10.5f and cos(Gamma) is %10.5f.\n",cell_parameters[i]/pi*180.0,cos(cell_parameters[i]));
	    }
	  }
	}
      }
    }
    printf("  Lattice vectors are:\n");
    for(i=0;i<3;i++){
      printf("  (%5.2f, %5.2f, %5.2f)\n",lattice_vector[i][0],lattice_vector[i][1],lattice_vector[i][2]);
    }
    printf("  The volume of unit cell is %10.5f\n",Cal_Cell_Volume(lattice_vector));    

  }

  /*Second(optional):
    Check whether the inputed unit cell is a primitive cell or not.
    INPUT 
    bravais_type            Bravais lattice type
    cell_parameters         a, b, c length and angles between vectors 
    lattice_vector          the original lattice vectors
    atomic_position         the atomic positions transformed from OpenMX
    atom_species            the indicator of last atom's index of each species in atomic_position
    atom_num                number of atoms in atomic_position array
    atom_type               number of species in array atom_species

    OUTPUT
    platt                   the primitive lattice vectors founded
    ptran_vec               array to store the translation vectors for each atom
    pcell_parameters        the cell parameters for primitive cell
    npcell                  number of primitive cell in cell defined by lattice_vector
  */    
  pcell_brav=Chk_Primitive_Cell(bravais_type,cell_parameters,Read_in_Lattice,
				atomic_position,atom_species,atom_num,atom_type,
				platt,ptran_vec,pcell_parameters,&npcell);

  if (myid==Host_ID){

    printf("  In Chk_Primitive_Cell,\n  the found primitive cell is ");
	
    switch(pcell_brav){

    case 1:
      /* sc */
      printf("  a simple cubic lattice.\n");
      break;
    case 2:
      /*bcc */    
      printf("  a body-centered cubic lattice.\n");
      break;  
    case 3:
      /* fcc */
      printf("  a face-centered cubic lattice.\n");
      break;
    case 4:
      /* hP */
      printf("  a primitive hexagonal lattice.\n");
      break;				
    case 5:
      /* tP */
      printf("  a primitive tetragonal lattice.\n");
      break;
    case 6:
      /* tI */                                                                          
      printf("  a body-centered tetragonal lattice.\n");
      break;
    case 7:
      /* hR  */
      printf("  a rhombohedral lattice.\n");
      break;
    case 8:
      /* oP */
      printf("  a primitive orthorhombic lattice.\n");
      break;
    case 9:
      /* oI  */
      printf("  a body-centered orthorhombic lattice.\n");
      break;
    case 10:
      /* oF  */
      printf("  a face-centered orthorhombic lattice.\n");
      break;    		
    case 11:
      /* oC  */
      printf("  a base-centered orthorhombic lattice.\n");
      break;
    case 12:
      /* mP */
      printf("  a primitive monoclinic lattice.\n");
      break;
    case 13:
      /* mB */
      printf("  a base-centered monoclinic lattice.\n");
      break;
    case 14:
      /* aP triclinic */
      printf("  a primitive triclinic lattice.\n");
      break;
    }
    printf("  Primitive lattice vectors are:\n");
    for(i=0;i<3;i++){
      printf("  P%1d (%10.5f,%10.5f,%10.5f)\n",i+1,platt[i][0],platt[i][1],platt[i][2]);
    }
    printf("  There are %2d primitive cells contained in your original cell.\n",npcell);

  }

  /* If the lattice vectors above have been interchanged or modified, the atomic position
     which stored as fractrion coordinates should also be changed.*/    
  if(debug2==1){
    printf("Transfer atomic coordinates.\n");
    printf("Original:\n");
    for(i=0;i<atom_num;i++){
      printf("atom%2d %10.5f, %10.5f, %10.5f\n",i+1,Read_in_Atom_Pos[i][0],Read_in_Atom_Pos[i][1],Read_in_Atom_Pos[i][2]);
    }
  }

  Atomic_Coordinates_Transform(Read_in_Atom_Pos, atomic_position, atom_num,
			       Read_in_Lattice, lattice_vector);

  if(debug2==1){                             
    printf("Transfered:\n");
    for(i=0;i<atom_num;i++){
      printf("atom%2d %10.5f, %10.5f, %10.5f\n",i+1,atomic_position[i][0],atomic_position[i][1],atomic_position[i][2]);
    }                                 
  }
    
  /*Bravais_Lattice_Symmetry generates the symmetry operation of the pure bravais lattice determined above,
    without consideration of actual atomic arrangement (lattice basis) 
    Input
    bravais_type
    OUTPUT
    sym_op              symmetry operation matrix   
    op_num              number of symmetry operation 
  */    

  op_num=Bravais_Lattice_Symmetry(bravais_type,sym_op);

  if (debug5){
    
    for(k=0;k<op_num;k++){
      fprintf(fp," openmx            %2d\n",k+1);
      for(i=0;i<3;i++){
	fprintf(fp," %4d%4d%4d\n",sym_op[k][i][0],sym_op[k][i][1],sym_op[k][i][2]);
      }
    }
    fclose(fp);
  }
  if(debug1==1){
    printf("In subroutine Bravais_Lattice_Symmetry:\n");
    printf("Totally found %4d symmtry operations\n",op_num);
  }
  if(debug1 == 1){
    for(k=0;k<op_num;k++){
      for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	  if(sym_op[k][i][j]!=0){
	    printf("    sym_op[%2d][%1d][%1d]=%2d.0 ;\n",k,i,j,sym_op[k][i][j]);
	  }
	}
      }
    }
  }
		
		
  /* Find the possible symmetry operations according to the atomic arrangement 
     (lattic basis, atomic_position) from the pool of symmetry operations 
     (sym_op[op_num][3][3]) found from pure bravais lattice. sym_op matix is changed
     after Get_Symmetry_Operation. Totally it returns op_num operations while the first
     num_pure_pnt_op is the pure point symmetry operation. The left are those connected
     with translation vectors, which are stored in trans_op[op_num-num_pure_pnt_op][3].
  */     

  Get_Symmetry_Operation(sym_op,&op_num,atomic_position,atom_species,atom_num,atom_type,&num_pure_pnt_op,trans_op_vec);

  if(debug3==1){
    printf("In subroutine Get_Symmetry_Operation:\n");
  }

  if (myid==Host_ID){
    printf("  Found allowed symmetry operation number is %2d.\n",op_num);
    printf("  Among them there are %2d pure point group operations.\n",num_pure_pnt_op);
  }  
  
  if(debug5){
    if((fp=fopen("getgrp.dat","wt"))==NULL){
      printf("Error in open file\n");
      return;
    }
    fprintf(fp,"GETGRP tot number is %2d\n",op_num);
    fprintf(fp,"GETGRP pure point num %2d\n",num_pure_pnt_op);
    for(k=0;k<num_pure_pnt_op;k++){
      for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	  if(sym_op[k][i][j]!=0){
	    fprintf(fp,"    purepoint[%2d][%1d][%1d]=%2d.0 ;\n",k,i,j,sym_op[k][i][j]);
	  }
	}
      }
    }
  } 
       
  if(debug5){

    for(k=num_pure_pnt_op;k<op_num;k++){
      for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	  if(sym_op[k][i][j]!=0){
	    fprintf(fp,"    space[%2d][%1d][%1d]=%2d.0 ;\n",k,i,j,sym_op[k][i][j]);
	  }
	}
      }
      fprintf(fp,"    Translation vector:%10.5f%10.5f%10.5f\n",trans_op_vec[k][0],trans_op_vec[k][1],trans_op_vec[k][2]);
    } 
    fclose(fp);
  }

  if(debug5){
    /* Reading the output from VASP to make comparision */    
    if((fp=fopen("vasp.dat","rt"))==NULL){
      printf("Error in open file\n");
      return;
    }
    fscanf(fp,"%d",&vasp_opnum);
    fscanf(fp,"%d",&vasp_numpurepntop);
    /*    printf("%d\n",vasp_opnum);
	  printf("%d",vasp_numpurepntop); */
    
    if(vasp_opnum!=op_num){
      printf("Different in total operation number. %2d vs. %2d\n",vasp_opnum,op_num);
      /*    	return; */
    }
    if(vasp_numpurepntop!=num_pure_pnt_op){
      printf("Different in pure point group operation number. %2d vs. %2d\n",vasp_numpurepntop,num_pure_pnt_op);
      /*   	return; */
    }

    vasp_symop=(int***)malloc(sizeof(int**)*vasp_opnum);
    vasp_transopvec=(double**)malloc(sizeof(double*)*vasp_opnum);
    for(k=0;k<vasp_opnum;k++){
      vasp_symop[k]=(int**)malloc(sizeof(int*)*3);
      vasp_transopvec[k]=(double*)malloc(sizeof(double)*3);
      for(i=0;i<3;i++){
	vasp_symop[k][i]=(int*)malloc(sizeof(int)*3);
	vasp_transopvec[k][i]=0.0;
      }
    }
    for (k=0; k<vasp_opnum; k++){
      for (i=0;i<3;i++){
	for (j=0;j<3;j++){
	  vasp_symop[k][i][j]=0;
	}
      }
    }
    /*    c=getchar(); */
    for(k=0;k<vasp_opnum;k++){
      for(i=0;i<3;i++){
	fscanf(fp,"%2d%2d%2d",&vasp_symop[k][i][0],&vasp_symop[k][i][1],&vasp_symop[k][i][2]);
	
      }
      fscanf(fp,"%lf%lf%lf",&vasp_transopvec[k][0],&vasp_transopvec[k][1],&vasp_transopvec[k][2]);
  
    }
    /* after reading, Now comparing */
    
    transok=0;
    for(k=0;k<vasp_opnum;k++){/* for each operation from vasp */
      for(kopen=0;kopen<vasp_opnum;kopen++){ /*scan those from openMX */
	opok=kopen;
	for(i=0;i<3;i++){
	  for(j=0;j<3;j++){
	    if(vasp_symop[k][i][j]!=sym_op[kopen][i][j]){
	      opok=-1;/* if there is one element not equal, then non-equal */
	    }
	  }
	}
	if(opok==kopen){
	  break;
	}
      }
      if(opok!=-1){/* opok remember that equals*/
	printf("SYMOP vasp %2d == openMX %d  ",k,opok);
	if(fabs(vasp_transopvec[k][0]-trans_op_vec[opok][0])<smallvalue
	   &&fabs(vasp_transopvec[k][1]-trans_op_vec[opok][1])<smallvalue
	   &&fabs(vasp_transopvec[k][2]-trans_op_vec[opok][2])<smallvalue){
	  printf("trans vec also equal\n");
	}else{
	  printf("************TRANS VECTOR NOT EQUAL***************\n");
	}
      }else{
	printf("!!!!!!!!!!!!!!!!%2d NO SAME!!!!!!!!!!!\n",k);
      }
    }/* end comparision */    
    
    for(k=0;k<vasp_opnum; k++){
      for(i=0; i<3; i++){
	free(vasp_symop[k][i]);
      }
      free(vasp_symop[k]);
      free(vasp_transopvec[k]);
    }  
    free(vasp_symop);
    free(vasp_transopvec);
  } /* end of debug5 */

  /* Now, we can generate the k-points and their weights  */
  kpt_num = 0;

  if (myid==Host_ID){
    printf("  Sampling grids are %3dx%3dx%3d\n",knum_i, knum_j, knum_k);
  }

  tmpK1=(double*)malloc(sizeof(double)*(8*knum_i*knum_j*knum_k));
  tmpK2=(double*)malloc(sizeof(double)*(8*knum_i*knum_j*knum_k));
  tmpK3=(double*)malloc(sizeof(double)*(8*knum_i*knum_j*knum_k));
  tmpWeight=(int*)malloc(sizeof(int)*(8*knum_i*knum_j*knum_k));

  for(i=0;i<(8*knum_i*knum_j*knum_k);i++){
    tmpK1[i]=-1234567;
    tmpK2[i]=-1234567;
    tmpK3[i]=-1234567;
    tmpWeight[i]=0;
  }
    
  /* Transfer the symmetry operation into original lattice vector */
  /* Here op_num must be equal or smaller than 48*/

  op_num_original = op_num;

  Gsym_op = (int***)malloc(sizeof(int**)*op_num_original);
  for(k=0; k<op_num_original; k++){
    Gsym_op[k] = (int**)malloc(sizeof(int*)*3);
    for(i=0; i<3; i++){
      Gsym_op[k][i] = (int*)malloc(sizeof(int)*3);
      for(j=0; j<3; j++) Gsym_op[k][i][j] = 0;
    }
  }

  rsym_op = (int***)malloc(sizeof(int**)*op_num_original);
  for(k=0; k<op_num_original; k++){
    rsym_op[k] = (int**)malloc(sizeof(int*)*3);
    for(i=0; i<3; i++){
      rsym_op[k][i] = (int*)malloc(sizeof(int)*3);
      for(j=0; j<3; j++) rsym_op[k][i][j] = 0;
    }
  }

  if(debug4==1){
    printf("Cryatal Lattice:\n");
    for(i=0;i<3;i++){
      printf("A%1d (%10.5f, %10.5f, %10.5f)\n",i+1,Read_in_Lattice[i][0],Read_in_Lattice[i][1],Read_in_Lattice[i][2]);
    }
  }
    
  Symmetry_Operation_Transform(sym_op,rsym_op,op_num,lattice_vector,Read_in_Lattice);

  /* The reciprocal lattice vectors */    
  Cal_Reciprocal_Vectors(Read_in_Lattice, rlatt);
  for(i=0;i<3;i++){
    rlatt[i][0]=rlatt[i][0]/2.0/pi;
    rlatt[i][1]=rlatt[i][1]/2.0/pi;
    rlatt[i][2]=rlatt[i][2]/2.0/pi;
    
    klatt[i][0]=rlatt[i][0]/knum_i;
    klatt[i][1]=rlatt[i][1]/knum_j;
    klatt[i][2]=rlatt[i][2]/knum_k;
  }
  if(debug1==1){
    printf("Reciprocal lattice:\n");
    for(i=0;i<3;i++){
      printf("K%1d (%10.5f, %10.5f, %10.5f)\n",i+1,rlatt[i][0],rlatt[i][1],rlatt[i][2]);
    }
  }
  if(debug4==1){
    printf("Basis of k-points:\n");
    for(i=0;i<3;i++){
      printf("K%1d (%10.5f, %10.5f, %10.5f)\n",i+1,klatt[i][0],klatt[i][1],klatt[i][2]);
    }
  }
    
  /* Convert the symmetry operation from real space lattice to reciprocal lattice 
     And now we have the symmetry operation matrix rsym_op and Gsym_op in both original
     real space lattice vectors and reciprocal lattice vectors, respectively.
     These symmetry operations are the real crystal structures symmetrical operations.
  */
  if(debug4==1){
    printf("before transfer rsym_op is:\n");
    for(k=0;k<op_num;k++){
      printf("openmx rsym_op %2d\n",k+1);
      printf("%2d %2d %2d \n",rsym_op[k][0][0],rsym_op[k][0][1],rsym_op[k][0][2]);
      printf("%2d %2d %2d \n",rsym_op[k][1][0],rsym_op[k][1][1],rsym_op[k][1][2]);
      printf("%2d %2d %2d \n",rsym_op[k][2][0],rsym_op[k][2][1],rsym_op[k][2][2]);
    }
    /* c=getchar(); */
  }

  Symmetry_Operation_Transform(rsym_op,Gsym_op,op_num,Read_in_Lattice,rlatt);   

  if(debug4==1){
    printf("after transfer Gsym_op is:\n");
    for(k=0;k<op_num;k++){
      printf("openmx Gsym_op %2d\n",k+1);
      printf("%2d %2d %2d \n",Gsym_op[k][0][0],Gsym_op[k][0][1],Gsym_op[k][0][2]);
      printf("%2d %2d %2d \n",Gsym_op[k][1][0],Gsym_op[k][1][1],Gsym_op[k][1][2]);
      printf("%2d %2d %2d \n",Gsym_op[k][2][0],Gsym_op[k][2][1],Gsym_op[k][2][2]);
    }
    /* c=getchar(); */
  }
          
  for(i=0;i<3;i++){
    rlatt[i][0]=rlatt[i][0]/knum_i;
    rlatt[i][1]=rlatt[i][1]/knum_j;
    rlatt[i][2]=rlatt[i][2]/knum_k;
  }
        
  /* Get the lattice type of reciprocal lattice */  
  
  bravais_type=Finding_Bravais_Lattice_Type(rlatt,cell_parameters);

  /* shift of grids: if total grid numer is even, shift 0.5
     if total grid number is odd, no shift
  */

  shift[0] = 0.0;
  shift[1] = 0.0;
  shift[2] = 0.0;

  shift[0]=shift[0]+0.5*fmod((double)(knum_i+1),2.0);
  shift[1]=shift[1]+0.5*fmod((double)(knum_j+1),2.0);
  shift[2]=shift[2]+0.5*fmod((double)(knum_k+1),2.0);
  
  Cal_Reciprocal_Vectors(rlatt,platt);
  for(i=0;i<3;i++){
    platt[i][0]=platt[i][0]/2.0/pi;
    platt[i][1]=platt[i][1]/2.0/pi;
    platt[i][2]=platt[i][2]/2.0/pi;
  }
  if(debug4==1){
    printf("shift test\n");
    for(i=0;i<3;i++){
      printf("K%1d (%10.5f, %10.5f, %10.5f)\n",i+1,klatt[i][0],klatt[i][1],klatt[i][2]);
    }
    for(i=0;i<3;i++){
      printf("PINV%1d (%10.5f, %10.5f, %10.5f)\n",i+1,platt[i][0],platt[i][1],platt[i][2]);
    }
  }
  kx=shift[0]*klatt[0][0]+shift[1]*klatt[1][0]+shift[2]*klatt[2][0];
  ky=shift[0]*klatt[0][1]+shift[1]*klatt[1][1]+shift[2]*klatt[2][1];
  kz=shift[0]*klatt[0][2]+shift[1]*klatt[1][2]+shift[2]*klatt[2][2];
  if(debug4==1){
    printf("T (%10.5f, %10.5f, %10.5f)\n",kx,ky,kz);
  }
  tmp_shift[0]=kx*platt[0][0]+ky*platt[0][1]+kz*platt[0][2];
  tmp_shift[1]=kx*platt[1][0]+ky*platt[1][1]+kz*platt[1][2];
  tmp_shift[2]=kx*platt[2][0]+ky*platt[2][1]+kz*platt[2][2];
  if(debug4==1){
    printf("tmp shift (%10.5f, %10.5f, %10.5f)\n",tmp_shift[0],tmp_shift[1],tmp_shift[2]);
  }
  tmp_shift[0]=fmod(tmp_shift[0]+1000.25,1.0)-0.25;
  tmp_shift[1]=fmod(tmp_shift[1]+1000.25,1.0)-0.25;
  tmp_shift[2]=fmod(tmp_shift[2]+1000.25,1.0)-0.25;
  
  kx=shift[0];ky=shift[1];kz=shift[2];
  shift[0]=tmp_shift[0];
  shift[1]=tmp_shift[1];
  shift[2]=tmp_shift[2];
  
  if(debug4==1){
    printf("General checking shift is %10.5f, %10.5f, %10.5f\n",shift[0],shift[1],shift[2]);
  }
  
  shift_keep=1;
  if( ((fabs(shift[0])>smallvalue) && (fabs(shift[0]-0.5)>smallvalue)) 
      ||((fabs(shift[1])>smallvalue) && (fabs(shift[1]-0.5)>smallvalue))
      ||((fabs(shift[2])>smallvalue) && (fabs(shift[2]-0.5)>smallvalue))
      ){
    shift_keep=0;
  }
   
  if( (bravais_type==1 || bravais_type==3 || bravais_type==6 || bravais_type==7 || bravais_type==10)
      && ((fabs(shift[0])>smallvalue) || (fabs(shift[1])>smallvalue) || (fabs(shift[2])>smallvalue))
      && ((fabs(shift[0]-0.5)>smallvalue) || (fabs(shift[1]-0.5)>smallvalue) || (fabs(shift[2]-0.5)>smallvalue))){
    shift_keep=0;
  }
  if( bravais_type==2 
      && ( (fabs(shift[0])>smallvalue) || 
	   (fabs(shift[1])>smallvalue) || 
	   (fabs(shift[2])>smallvalue)
	   )
      ){
    shift_keep=0;
  }
  if( bravais_type==4 && ((fabs(shift[0])>smallvalue) || (fabs(shift[1])>smallvalue) || (fabs(shift[2])>smallvalue))){
    shift_keep=0;
  }
  if( (bravais_type==5 || bravais_type==11 || bravais_type==13) 
      && ((fabs(shift[0])>smallvalue) || (fabs(shift[1])>smallvalue) || (fabs(shift[2])>smallvalue))
      && ((fabs(shift[0]-0.5)>smallvalue) || (fabs(shift[1]-0.5)>smallvalue) || (fabs(shift[2])>smallvalue))
      && ((fabs(shift[2]-0.5)>smallvalue) || (fabs(shift[0])>smallvalue) || (fabs(shift[1])>smallvalue))
      && ((fabs(shift[0]-0.5)>smallvalue) || (fabs(shift[1]-0.5)>smallvalue) || (fabs(shift[2]-0.5)>smallvalue))){
    shift_keep=0;
  }
  if( bravais_type==9 
      && ((fabs(shift[0])>smallvalue) || (fabs(shift[1])>smallvalue) || (fabs(shift[2])>smallvalue))
      && ((fabs(shift[0]-0.5)>smallvalue) || (fabs(shift[1]-0.5)>smallvalue) || (fabs(shift[2])>smallvalue))
      && ((fabs(shift[0]-0.5)>smallvalue) || (fabs(shift[1])>smallvalue) || (fabs(shift[2]-0.5)>smallvalue))
      && ((fabs(shift[0])>smallvalue) || (fabs(shift[1]-0.5)>smallvalue) || (fabs(shift[2]-0.5)>smallvalue))){
    shift_keep=0;
  }
  
  shift[0]=kx;shift[1]=ky;shift[2]=kz;
  
  if(debug4==1){
    printf("shift_keep=%1d\n",shift_keep);
  }
  if(shift_keep==0){
    for(i=0;i<3;i++){
      atomic_position[0][i]=0.0;
    }
    atom_species[0]=1;
    pureG_num=Bravais_Lattice_Symmetry(bravais_type,sym_op);
    Get_Symmetry_Operation(sym_op,&pureG_num,atomic_position,atom_species,1,1,&num_pure_pnt_op,trans_op_vec);

    Cal_Reciprocal_Vectors(Read_in_Lattice, platt);
    for(i=0;i<3;i++){
      platt[i][0]=platt[i][0]/2.0/pi;
      platt[i][1]=platt[i][1]/2.0/pi;
      platt[i][2]=platt[i][2]/2.0/pi;
    }
    if(debug4==1){
      for(i=0;i<3;i++){
	printf("K%1d (%10.5f, %10.5f, %10.5f)\n",i+1,platt[i][0],platt[i][1],platt[i][2]);
      }    	
    }

    Symmetry_Operation_Transform(sym_op, pureGsym, pureG_num, rlatt, platt);

    if(debug4==1){
      for(k=0;k<pureG_num;k++){
	printf("pureGsym %2d\n",k+1);
	printf("%2d %2d %2d \n",pureGsym[k][0][0],pureGsym[k][0][1],pureGsym[k][0][2]);
	printf("%2d %2d %2d \n",pureGsym[k][1][0],pureGsym[k][1][1],pureGsym[k][1][2]);
	printf("%2d %2d %2d \n",pureGsym[k][2][0],pureGsym[k][2][1],pureGsym[k][2][2]);
      }
    }
  }
  else{
    pureG_num=1;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	pureGsym[0][i][j]=0;
      }
    }
    pureGsym[0][0][0]=1; 
    pureGsym[0][1][1]=1;
    pureGsym[0][2][2]=1;
  }
    
  /* checking whether inversion symmetry exist or not 
     but for non-collinear case, time-inversion symmtry should not be
     enforced. 
     inversion_flag == 0   collinear calculation
     inversion_flag == 1   noncollinear calculation 
  */

  inversion_sym=0;
  for(k=0;k<op_num;k++){
    if(Gsym_op[k][0][0]==-1&&Gsym_op[k][0][1]==0&&Gsym_op[k][0][2]==0
       &&Gsym_op[k][1][0]==0&&Gsym_op[k][1][1]==-1&&Gsym_op[k][1][2]==0
       &&Gsym_op[k][2][0]==0&&Gsym_op[k][2][1]==0&&Gsym_op[k][2][2]==-1){inversion_sym=1;break;}
  }

  /* c=getchar(); */

  if (inversion_sym==0 && inversion_flag==0){

    if(debug4==1){
      printf("Collinear case and inversion symmetry does not exist.\n");
    }

    ksym_num = 2*op_num;

    ksym_op = (int***)malloc(sizeof(int**)*ksym_num);
    for(k=0; k<ksym_num; k++){
      ksym_op[k] = (int**)malloc(sizeof(int*)*3);
      for(i=0; i<3; i++){
	ksym_op[k][i] = (int*)malloc(sizeof(int)*3);
      }
    }

    inv=(int**)malloc(sizeof(int*)*3);
    tmpsym=(int**)malloc(sizeof(int*)*3);
    tmpisym=(int**)malloc(sizeof(int*)*3);
    for(i=0;i<3;i++){
      inv[i]=(int*)malloc(sizeof(int)*3);
      tmpsym[i]=(int*)malloc(sizeof(int)*3);
      tmpisym[i]=(int*)malloc(sizeof(int)*3);
    }

    inv[0][0]=-1;inv[0][1]=0; inv[0][2]=0;
    inv[1][0]=0 ;inv[1][1]=-1;inv[1][2]=0;
    inv[2][0]=0 ;inv[2][1]=0; inv[2][2]=-1;

    for(k=0; k<op_num; k++){
      Matrix_Copy32(tmpsym, 3, Gsym_op, k);
      Matrix_Productor(3, inv, tmpsym, tmpisym);
      Matrix_Copy23(tmpisym, 3, ksym_op, k+op_num);
    }

    for(i=0;i<3;i++){
      free(inv[i]);
      free(tmpsym[i]);
      free(tmpisym[i]);
    }
    free(inv);
    free(tmpsym);
    free(tmpisym);
  }

  else{ /*Inversion symmetry exists and noncollinear calculation.*/

    if(debug4==1){
      if(inversion_sym==1){
        printf("Inversion does EXIST.\n");
      }else{
        printf("Noncollinear calculation.\n"); 
      }
    }

    ksym_num = op_num;

    ksym_op = (int***)malloc(sizeof(int**)*ksym_num);

    for(k=0; k<ksym_num; k++){
      ksym_op[k] = (int**)malloc(sizeof(int*)*3);
      for(i=0; i<3; i++){
	ksym_op[k][i] = (int*)malloc(sizeof(int)*3);
      }
    }

  }

  for(k=0; k<op_num; k++){
    for (i=0;i<3;i++){
      for (j=0;j<3;j++){
	ksym_op[k][i][j] = Gsym_op[k][i][j]; /* ksym_op correspondes to G in VASP */
      }
    }
  }

  if(debug4==1){
    for(k=0; k<ksym_num; k++){
      printf("openmx ksym_op %2d\n",k+1);
      printf("%2d %2d %2d \n",ksym_op[k][0][0],ksym_op[k][0][1],ksym_op[k][0][2]);
      printf("%2d %2d %2d \n",ksym_op[k][1][0],ksym_op[k][1][1],ksym_op[k][1][2]);
      printf("%2d %2d %2d \n",ksym_op[k][2][0],ksym_op[k][2][1],ksym_op[k][2][2]);
    }
    /* c=getchar(); */
  }

  kpt_num = Generate_MP_Special_Kpt(knum_i, knum_j, knum_k, ksym_op,ksym_num, pureGsym, pureG_num,
		  		    shift, tmpK1, tmpK2, tmpK3, tmpWeight);

  num_non_eq_kpt = kpt_num;

  NE_KGrids1 = (double*)malloc(sizeof(double)*kpt_num);
  NE_KGrids2 = (double*)malloc(sizeof(double)*kpt_num);
  NE_KGrids3 = (double*)malloc(sizeof(double)*kpt_num);
  NE_T_k_op = (int*)malloc(sizeof(int)*kpt_num);
  alloc_first[23] = 0;

  for(k=0; k<kpt_num; k++){
    NE_KGrids1[k] = tmpK1[k];
    NE_KGrids2[k] = tmpK2[k];
    NE_KGrids3[k] = tmpK3[k];
    NE_T_k_op[k] = tmpWeight[k];
    tmpK1[k] = 0.0;
    tmpK2[k] = 0.0;
    tmpK3[k] = 0.0;
    tmpWeight[k] = 0;
  } 
    
  /* hmweng Output non-equivalent k points*/

  if(debug5){

    /* For comparing with VASP IBZKPT file */
    if((fp=fopen("IBZKPT","rt"))==NULL){
      printf("Error in open file\n");
      return;
    }
    fscanf(fp,"%d",&i);
    
    if(i!=kpt_num){
      printf("K point number are different. VASP %2d vs. %2d\n",i,kpt_num);
      return;
    }

    for(k=0;k<kpt_num;k++){
      fscanf(fp,"%lf%lf%lf%d",&tmpK1[k],&tmpK2[k],&tmpK3[k],&tmpWeight[k]);
      /*    	printf("%20.14f%20.14f%20.14f%14d\n",tmpK1[k],tmpK2[k],tmpK3[k],tmpWeight[k]); */
    }
    /* after reading, Now comparing */
    
    for(k=0;k<kpt_num;k++){/* for each k point from vasp */
      for(kopen=0;kopen<kpt_num;kopen++){ /*scan those from openMX */
	opok=-1;
	if(fabs(tmpK1[k]-NE_KGrids1[kopen])<smallvalue
	   &&fabs(tmpK2[k]-NE_KGrids2[kopen])<smallvalue
	   &&fabs(tmpK3[k]-NE_KGrids3[kopen])<smallvalue
	   &&tmpWeight[k]==NE_T_k_op[kopen]){
	  opok=kopen;/* if there is one element not equal, then non-equal */
	}
	if(opok==kopen){
	  break;
	}
      }
      if(opok!=-1){/* opok remember that equals*/
	printf("VASP k point %2d == openMX %2d\n",k,opok);
      }else{
	printf("!!!!!!!!!!!!!!!!%2d NO SAME!!!!!!!!!!!\n",k);
      }
    }/* end comparision */    

  } /* end of debug5 */

  /* end output non-equivalent k points */

  if (myid==Host_ID){
    printf("\n");
    printf("  Number of non-equivalent k points is %2d\n",kpt_num);
    printf("  Generated Monkhorst-Pack k-points with a weight\n\n");  
    for(k=0;k<kpt_num;k++){
      printf("%20.14f%20.14f%20.14f%14d\n",NE_KGrids1[k],NE_KGrids2[k],NE_KGrids3[k],NE_T_k_op[k]); 
    }
  }  

  free(tmpK1);
  free(tmpK2);
  free(tmpK3);
  free(tmpWeight);

  for(i=0; i<3; i++){
    free(lattice_vector[i]);
    free(platt[i]);
    free(rlatt[i]);
    free(klatt[i]);
  }

  free(lattice_vector);
  free(platt);
  free(rlatt);
  free(klatt);

  for(k=0; k<op_num_original; k++){
    for(i=0; i<3; i++){
      free(Gsym_op[k][i]);
    }
    free(Gsym_op[k]);
  }
  free(Gsym_op);

  for(k=0; k<op_num_original; k++){
    for(i=0; i<3; i++){
      free(rsym_op[k][i]);
    }
    free(rsym_op[k]);
  }
  free(rsym_op);

  for(k=0; k<ksym_num; k++){
    for(i=0;i<3;i++){
      free(ksym_op[k][i]);
    }
    free(ksym_op[k]);
  }
  free(ksym_op);

  for(k=0; k<op_num_max; k++){
    for(i=0; i<3; i++){
      free(sym_op[k][i]);
    }
    free(sym_op[k]);
  }
  free(sym_op);

  for(k=0; k<op_num_max; k++){
    for(i=0; i<3; i++){
      free(pureGsym[k][i]);
    }
    free(pureGsym[k]);
  }
  free(pureGsym);

  for(k=0; k<op_num_max; k++){
    free(trans_op_vec[k]);
  }
  free(trans_op_vec);

  for(k=0; k<(atom_num+2); k++){
    free(atomic_position[k]);
  }
  free(atomic_position);

  for(k=0; k<(atom_num+2); k++){
    free(ptran_vec[k]);
  }
  free(ptran_vec);

  free(atom_species);

}


double Cal_Cell_Volume(double **lattice_vector){
  /* calculate the volume determined by lattice_vector[3][3] */
  int i, j, k;
  double omega;
  omega = 0.0;
  for(i=0; i<3; i++){
    omega = omega + lattice_vector[0][i]*(lattice_vector[1][(i+1)%3]*lattice_vector[2][(i+2)%3]-lattice_vector[1][(i+2)%3]*lattice_vector[2][(i+1)%3]);
    /* debug 
       printf("a%3d*(b%3d*c%3d-b%3d*c%3d)\n",i,(i+1)%3,(i+2)%3,(i+2)%3,(i+1)%3); */
  }
  return omega;
}

void Cal_Reciprocal_Vectors(double **latt_vec, double **rec_latt_vec)
{
  /* calculating the reciprocal lattice vectors of inputted latt_vec. The unit is 2*pi/a */
  int i,j,k;
  double omega;
  omega=0.0;
  omega=Cal_Cell_Volume(latt_vec);
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rec_latt_vec[i][j]=latt_vec[(i+1)%3][(j+1)%3]*latt_vec[(i+2)%3][(j+2)%3]-latt_vec[(i+1)%3][(j+2)%3]*latt_vec[(i+2)%3][(j+1)%3];
      rec_latt_vec[i][j]=rec_latt_vec[i][j]*2*pi/omega;
    }
  }
  return;
}



void Atomic_Coordinates_Transform(double **Read_in_Atom_Pos, double **atomic_position, int atom_num, 
                                  double **Read_in_Lattice, double **lattice_vector){
  /* Convert the atomic fractional coordinates from one lattice vectors (Read_in_Lattice) to another one (lattice_vector).*/
  int i,j,k;
  double r[3];
  double **rec_latt_vec;

  rec_latt_vec=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    rec_latt_vec[i]=(double*)malloc(sizeof(double)*3); 
  }

  Cal_Reciprocal_Vectors(lattice_vector, rec_latt_vec);

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rec_latt_vec[i][j]=rec_latt_vec[i][j]/2.0/pi;
    }
  }

  for(i=0;i<atom_num;i++){
    r[0]=Read_in_Lattice[0][0]*Read_in_Atom_Pos[i][0]+Read_in_Lattice[1][0]*Read_in_Atom_Pos[i][1]+Read_in_Lattice[2][0]*Read_in_Atom_Pos[i][2];
    r[1]=Read_in_Lattice[0][1]*Read_in_Atom_Pos[i][0]+Read_in_Lattice[1][1]*Read_in_Atom_Pos[i][1]+Read_in_Lattice[2][1]*Read_in_Atom_Pos[i][2];
    r[2]=Read_in_Lattice[0][2]*Read_in_Atom_Pos[i][0]+Read_in_Lattice[1][2]*Read_in_Atom_Pos[i][1]+Read_in_Lattice[2][2]*Read_in_Atom_Pos[i][2];
    atomic_position[i][0]=rec_latt_vec[0][0]*r[0]+rec_latt_vec[0][1]*r[1]+rec_latt_vec[0][2]*r[2];
    atomic_position[i][1]=rec_latt_vec[1][0]*r[0]+rec_latt_vec[1][1]*r[1]+rec_latt_vec[1][2]*r[2];
    atomic_position[i][2]=rec_latt_vec[2][0]*r[0]+rec_latt_vec[2][1]*r[1]+rec_latt_vec[2][2]*r[2];
    
    /*    
	  atomic_position[i][0]=atomic_position[i][0]-floor(atomic_position[i][0]);
	  atomic_position[i][0]=fmod(atomic_position[i][0]+1000.5,1.0)-0.5;
	  
	  atomic_position[i][1]=atomic_position[i][1]-floor(atomic_position[i][1]);
	  atomic_position[i][1]=fmod(atomic_position[i][1]+1000.5,1.0)-0.5;
	  
	  atomic_position[i][2]=atomic_position[i][2]-floor(atomic_position[i][2]);
	  atomic_position[i][2]=fmod(atomic_position[i][2]+1000.5,1.0)-0.5;
    */ 
  }
  
  for(i=0;i<3;i++){
    free(rec_latt_vec[i]);
  }
  free(rec_latt_vec);
  
  return;
}

void Symmetry_Operation_Transform(int ***InputSymop, int ***OutputSymop, int op_num, 
                                  double **In_Lattice, double **Out_Lattice){
  /* Convert the atomic fractional coordinates from one lattice vectors (Read_in_Lattice) to another one (lattice_vector).*/
  int i,j,k,kk,ibi;
  double Isym[3][3], Osym[3][3];
  double **rIn, **rOut, tmp;
  double adbi[3][3],bdai[3][3],prod[3][3],sum;
  char c;

  rIn=(double**)malloc(sizeof(double*)*3);
  rOut=(double**)malloc(sizeof(double*)*3);

  for(i=0;i<3;i++){
    rOut[i]=(double*)malloc(sizeof(double)*3);
    rIn[i]=(double*)malloc(sizeof(double)*3);
  }
	
  Cal_Reciprocal_Vectors(In_Lattice, rIn);
  Cal_Reciprocal_Vectors(Out_Lattice, rOut);
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rIn[i][j]=rIn[i][j]/2.0/pi;
      rOut[i][j]=rOut[i][j]/2.0/pi;
      Isym[i][j]=0.0;
      Osym[i][j]=0.0;
    }
  }	
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      bdai[i][j]=Out_Lattice[i][0]*rIn[j][0]+Out_Lattice[i][1]*rIn[j][1]+Out_Lattice[i][2]*rIn[j][2];
      /* printf("bdotai[%1d][%1d]=%10.5f\n",i+1,j+1,bdai[i][j]); */
    }
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      adbi[i][j]=In_Lattice[i][0]*rOut[j][0]+In_Lattice[i][1]*rOut[j][1]+In_Lattice[i][2]*rOut[j][2];
      /* printf("adotbi[%1d][%1d]=%10.5f\n",i+1,j+1,adbi[i][j]); */
    }
  }

  for(k=0; k<op_num; k++){

    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	Isym[i][j]=(double)InputSymop[k][i][j];
      }
    }

    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	sum=0.0;
	for(kk=0;kk<3;kk++){
	  sum=sum+bdai[i][kk]*Isym[kk][j];
	}
	prod[i][j]=sum;
      }
    }

    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	sum=0.0;
	for(kk=0;kk<3;kk++){
	  sum=sum+prod[i][kk]*adbi[kk][j];
	}
	Osym[i][j]=sum;
      }
    }

    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	/* printf("FLOAT=%20.18f\n",Osym[i][j]); */
	if(fabs(Osym[i][j]-1.0)<smallvalue){Osym[i][j]=1.0;}
	if(fabs(Osym[i][j]+1.0)<smallvalue){Osym[i][j]=-1.0;}
	OutputSymop[k][i][j]=(int)Osym[i][j];
	/* printf("OutputSymop[%2d][%1d][%1d]=%3d %20.18f\n",k,i,j,OutputSymop[k][i][j],Osym[i][j]); */
	Osym[i][j]=0.0;
      }
    }

  }
  
  for(i=0;i<3;i++){
    free(rOut[i]);
    free(rIn[i]);
  }
  free(rIn);
  free(rOut);
}
    
void Chk_Shorter_Lattice_Vector(double **rlatt){
  int i,j,k;
  int shorter;
  double abc[3], alpha, beta, gamma, a, b, c;
  double ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  double absv;
    
  ax = rlatt[0][0];
  ay = rlatt[0][1];
  az = rlatt[0][2];
  bx = rlatt[1][0];
  by = rlatt[1][1];
  bz = rlatt[1][2];
  cx = rlatt[2][0];
  cy = rlatt[2][1];
  cz = rlatt[2][2];    
    
  for(k=0;k<3;k++){
    abc[k]=sqrt(rlatt[k][0]*rlatt[k][0]+rlatt[k][1]*rlatt[k][1]+rlatt[k][2]*rlatt[k][2]);
  }
  a=abc[0];b=abc[1];c=abc[2];
  dx=bx-cx; dy=by-cy; dz=bz-cz;   
  alpha = acos((b*b+c*c-dx*dx-dy*dy-dz*dz)/(2.0*b*c));
    
  dx=ax-cx; dy=ay-cy; dz=az-cz;
  beta = acos((a*a+c*c-dx*dx-dy*dy-dz*dz)/(2.0*a*c));
    
  dx=ax-bx; dy=ay-by; dz=az-bz;
  gamma = acos((a*a+b*b-dx*dx-dy*dy-dz*dz)/(2.0*a*b));

  shorter=1;i=0;j=0;
    
  while(shorter==1){
    shorter=0;/* try to be confident with the present vectors */
    for(k=0;k<3;k++){/* for three vectors */
      j=0;
      while(1==1){/* for vector[k]-vector[(k+1+k%2)%3] */
	i++;
	rlatt[k][0]=rlatt[k][0]-rlatt[(k+1+k%2)%3][0];
	rlatt[k][1]=rlatt[k][1]-rlatt[(k+1+k%2)%3][1];
	rlatt[k][2]=rlatt[k][2]-rlatt[(k+1+k%2)%3][2];
	absv=sqrt(rlatt[k][0]*rlatt[k][0]+rlatt[k][1]*rlatt[k][1]+rlatt[k][2]*rlatt[k][2]);
	/* printf("%2d-%2d vector: new length is %10.5f, old one is %10.5f\n",k,(k+1)%3,absv,abc[k]); */
	if(absv>(abc[k]+smallvalue)){
	  break; /* no search for vector[k]-vector[(k+1k%2)%3], break to + case */
	}
	if(absv<(abc[k]-smallvalue)){
	  shorter=1; /* there is shorter choice. Then continue searching */
	  abc[k]=absv;
	}
	j=1; /* in - case, found one, mark it and skip + case */
      }
      i++;
      rlatt[k][0]=rlatt[k][0]+rlatt[(k+1+k%2)%3][0];
      rlatt[k][1]=rlatt[k][1]+rlatt[(k+1+k%2)%3][1];
      rlatt[k][2]=rlatt[k][2]+rlatt[(k+1+k%2)%3][2];
      while(j==0){/*start + case if - case can not find shorter vector*/
	i++;
	rlatt[k][0]=rlatt[k][0]+rlatt[(k+1+k%2)%3][0];
	rlatt[k][1]=rlatt[k][1]+rlatt[(k+1+k%2)%3][1];
	rlatt[k][2]=rlatt[k][2]+rlatt[(k+1+k%2)%3][2];
	absv=sqrt(rlatt[k][0]*rlatt[k][0]+rlatt[k][1]*rlatt[k][1]+rlatt[k][2]*rlatt[k][2]);
	/* printf("%2d+%2d vector: new length is %10.5f, old one is %10.5f\n",k,(k+1)%3,absv,abc[k]);*/
	if(absv>(abc[k]+smallvalue)){
	  break;/* no search for vector[k]+vector[(k+1+k%2)%3], break */
	}
	if(absv<(abc[k]-smallvalue)){
	  shorter=1;
	  abc[k]=absv;
	}
      }
      if(j==0){
	rlatt[k][0]=rlatt[k][0]-rlatt[(k+1+k%2)%3][0];
	rlatt[k][1]=rlatt[k][1]-rlatt[(k+1+k%2)%3][1];
	rlatt[k][2]=rlatt[k][2]-rlatt[(k+1+k%2)%3][2];
      }
      j=0;
      while(1==1){/* for vector[k]-vector[(k+2-k%2)%3] */
	i++;
	rlatt[k][0]=rlatt[k][0]-rlatt[(k+2-k%2)%3][0];
	rlatt[k][1]=rlatt[k][1]-rlatt[(k+2-k%2)%3][1];
	rlatt[k][2]=rlatt[k][2]-rlatt[(k+2-k%2)%3][2];
	absv=sqrt(rlatt[k][0]*rlatt[k][0]+rlatt[k][1]*rlatt[k][1]+rlatt[k][2]*rlatt[k][2]);
	/* printf("%2d-%2d vector: new length is %10.5f, old one is %10.5f\n",k,(k+2)%3,absv,abc[k]);*/
	if(absv>(abc[k]+smallvalue)){
	  break; /* no search for vector[k]-vector[(k+2-k%2)%3], break to + case */
	}
	if(absv<(abc[k]-smallvalue)){
	  shorter=1; /* there is shorter choice. Then continue searching */
	  abc[k]=absv;
	}
	j=1; /* in - case, found one, mark it and skip + case */
      }
      i++;
      rlatt[k][0]=rlatt[k][0]+rlatt[(k+2-k%2)%3][0];
      rlatt[k][1]=rlatt[k][1]+rlatt[(k+2-k%2)%3][1];
      rlatt[k][2]=rlatt[k][2]+rlatt[(k+2-k%2)%3][2];
      while(j==0){/*start + case if - case can not find shorter vector*/
	i++;
	rlatt[k][0]=rlatt[k][0]+rlatt[(k+2-k%2)%3][0];
	rlatt[k][1]=rlatt[k][1]+rlatt[(k+2-k%2)%3][1];
	rlatt[k][2]=rlatt[k][2]+rlatt[(k+2-k%2)%3][2];
	absv=sqrt(rlatt[k][0]*rlatt[k][0]+rlatt[k][1]*rlatt[k][1]+rlatt[k][2]*rlatt[k][2]);
	/* printf("%2d+%2d vector: new length is %10.5f, old one is %10.5f\n",k,(k+2)%3,absv,abc[k]);*/
	if(absv>(abc[k]+smallvalue)){
	  break;/* no search for vector[k]+vector[(k+2-k%2)%3], break */
	}
	if(absv<(abc[k]-smallvalue)){
	  shorter=1;
	  abc[k]=absv;
	}
      }
      if(j==0){
	rlatt[k][0]=rlatt[k][0]-rlatt[(k+2-k%2)%3][0];
	rlatt[k][1]=rlatt[k][1]-rlatt[(k+2-k%2)%3][1];
	rlatt[k][2]=rlatt[k][2]-rlatt[(k+2-k%2)%3][2];
      }
    }/*finished + and - case for all vectors*/
  }

  ax = rlatt[0][0];
  ay = rlatt[0][1];
  az = rlatt[0][2];
  bx = rlatt[1][0];
  by = rlatt[1][1];
  bz = rlatt[1][2];
  cx = rlatt[2][0];
  cy = rlatt[2][1];
  cz = rlatt[2][2];    
    
  for(k=0;k<3;k++){
    abc[k]=sqrt(rlatt[k][0]*rlatt[k][0]+rlatt[k][1]*rlatt[k][1]+rlatt[k][2]*rlatt[k][2]);
  }
  a=abc[0];b=abc[1];c=abc[2];
  dx=bx-cx; dy=by-cy; dz=bz-cz;   
  alpha = acos((b*b+c*c-dx*dx-dy*dy-dz*dz)/(2.0*b*c));
    
  dx=ax-cx; dy=ay-cy; dz=az-cz;
  beta = acos((a*a+c*c-dx*dx-dy*dy-dz*dz)/(2.0*a*c));
    
  dx=ax-bx; dy=ay-by; dz=az-bz;
  gamma = acos((a*a+b*b-dx*dx-dy*dy-dz*dz)/(2.0*a*b));
    
  /*    printf("After trying to find the shorter lattice vector, the lattice vectors are:\n");
        printf("a=%10.5f,b=%10.5f,c=%10.5f\n",abc[0],abc[1],abc[2]);
        printf("Angles:\n");
        printf("b-c alpha=%10.5f,cos(alpha)=%10.5f\n",alpha/pi*180.0,cos(alpha));
        printf("a-c beta=%10.5f,cos(beta)=%10.5f\n",beta/pi*180.0,cos(beta));
        printf("a-b gamma=%10.5f,cos(gamma)=%10.5f\n",gamma/pi*180.0,cos(gamma)); 
    
	bra_typ=Bravais_Type(rlatt,cell_parameters);
	printf("This lattice type is %2d\n",bra_typ);
  */
  return;   
}

int Bravais_Type(double **lattice_vector,double *cell_parameters){
  /*
    INPUT 
    double lattice_vector[3][3]  Lattice Vectors given by users;
  
    OUTPUT
    int bravais_type             Bravais Lattice Type: 1 --- Primitive cubic (cP) or simple-cubic sc 
    2 --- Body-centered cubic (cI) bcc
    3 --- Face-centered cubic (cF) fcc
    4 --- Primitive hexagonal (hP)
    5 --- Primitive tetragonal (tP)
    6 --- Body-centerd tetragonal (tI)
    7 --- Rhombohedral (hR)
    8 --- Primitive orthorhombic (oP)
    9 --- Body-centered orthorhombic (oI)
    10 --- Face-centered orthorhombic (oF)
    11 --- Base-centered orthorhombic (oC)
    12 --- Primitive monoclinic (mP)
    13 --- Base-centred monoclinic (mB)
    14 --- Primitive triclinic (aP)
    Defined to be consistent with VASP.                                                    
    double cell_parameters[6]   lattice parameters like a, b, c and alpha, beta, gamma
  */
  double cell_volume;
  double ax, ay, az, bx, by, bz, cx, cy, cz;
  double a, b, c, alpha, beta, gamma;  
  double dx,dy,dz;
  int bravais_type,i,j,k;
  char ch;
  bravais_type=15;
  /* 1. Are the lattice vectors given in right-hand coordinates */
  cell_volume = Cal_Cell_Volume(lattice_vector);
  /*debug
    printf("cell volume is %10.5f\n",cell_volume);
  */ 
  if(fabs(cell_volume)<(smallvalue*smallvalue*smallvalue)){/* lattice volume is too small */
    printf("WARNING!!! The volume of given lattice is too small.\n");
  }
  if(cell_volume<0.0){/* not in right-hand coordinates, then reversing all the lattice vectors */
    lattice_vector[0][0]=-lattice_vector[0][0];
    lattice_vector[0][1]=-lattice_vector[0][1];
    lattice_vector[0][2]=-lattice_vector[0][2];
    lattice_vector[1][0]=-lattice_vector[1][0];
    lattice_vector[1][1]=-lattice_vector[1][1];
    lattice_vector[1][2]=-lattice_vector[1][2];
    lattice_vector[2][0]=-lattice_vector[2][0];
    lattice_vector[2][1]=-lattice_vector[2][1];
    lattice_vector[2][2]=-lattice_vector[2][2];
  }
  ax = lattice_vector[0][0];
  ay = lattice_vector[0][1];
  az = lattice_vector[0][2];
  bx = lattice_vector[1][0];
  by = lattice_vector[1][1];
  bz = lattice_vector[1][2];
  cx = lattice_vector[2][0];
  cy = lattice_vector[2][1];
  cz = lattice_vector[2][2];    
  a = sqrt(ax*ax+ay*ay+az*az);
  b = sqrt(bx*bx+by*by+bz*bz);
  c = sqrt(cx*cx+cy*cy+cz*cz);
  dx=bx-cx; dy=by-cy; dz=bz-cz;
  alpha = acos((b*b+c*c-dx*dx-dy*dy-dz*dz)/(2.0*b*c));
    
  dx=ax-cx; dy=ay-cy; dz=az-cz;
  beta = acos((a*a+c*c-dx*dx-dy*dy-dz*dz)/(2.0*a*c));
    
  dx=ax-bx; dy=ay-by; dz=az-bz;
  gamma = acos((a*a+b*b-dx*dx-dy*dy-dz*dz)/(2.0*a*b));
  /*    
	printf("vectors information:\n");
	printf("a=%10.5f,   b=%10.5f,   c=%10.5f\n",a,b,c);
	printf("Angles:\n");
	printf("b-c alpha=%10.5f,cos(alpha)=%10.5f\n",alpha/pi*180.0,cos(alpha));
	printf("a-c beta=%10.5f,cos(beta)=%10.5f\n",beta/pi*180.0,cos(beta));
	printf("a-b gamma=%10.5f,cos(gamma)=%10.5f\n",gamma/pi*180.0,cos(gamma)); 
  */
  for(i=0; i<6; i++){
    cell_parameters[i] = 0.0;
  }
  cell_parameters[3] = pi/2;
  cell_parameters[4] = pi/2;
  cell_parameters[5] = pi/2;
  if(fabs(alpha-beta)<smallvalue&&fabs(alpha-gamma)<smallvalue){
    /*if all three angles are equal, it would be cubic/tetragonal/orthorhombic or hR lattices*/
    if(fabs(alpha-pi/2.0)<smallvalue){
      /* if all three angles are right angles, it might be cubic/tetragonal/orthorhombic lattice*/
      if(fabs(a-b)<smallvalue&&fabs(b-c)<smallvalue){/* a=b=c and alpha=beta=gamma=90 degree, Primitive cubic (cP) */
	bravais_type=1;
	cell_parameters[0]=a;
      }else if(fabs(a-b)<smallvalue){
	/*two of a, b, c are equal, Primitive tetragonal (tP)*/
	/* Attention! Generally, the lattice vectors are chosen to have a=b 
	   If fabs(b-c)<smallvalue or fabs(a-c)<smallvalue){ */
	bravais_type = 5;
	/*if(fabs(a-b)<smallvalue){*/
	cell_parameters[0]=a;
	cell_parameters[2]=c/a;
	/*}else if(fabs(b-c)<smallvalue){
	  cell_parameters[0]=b;
	  cell_parameters[2]=a/b;
	  }else{
	  cell_parameters[0]=a;
	  cell_parameters[2]=b/a;
	  }*/
      }else if((c-b)>smallvalue && (b-a)>smallvalue){/* a,b,c are non-equivlent, primitive orthorhombic (oP) */
	/*Attention! Generally, lattic vectors are taken so that a, b, c are in ascending order.*/
	bravais_type = 8;
	cell_parameters[0]=a;
	cell_parameters[1]=b/a;
	cell_parameters[2]=c/a;
      }		
    }else{/*all angles are equal but not right angle, it would be cF, cI, or hR*/
      if(fabs(a-b)<smallvalue&&fabs(b-c)<smallvalue){ /*all the vectors' length are equal */
	if(fabs(cos(alpha)-1.0/2.0)<smallvalue){/* if all angles are 60 degree, it is Face-centred cubic (cF) */
	  bravais_type = 3;
	  cell_parameters[0]=sqrt(2.0)*a;
	}else if(fabs(cos(alpha)+1.0/3.0)<smallvalue){/* if all angles are arccos(-1/3) degree, it is Body-centred cubic (cI) */
	  bravais_type = 2;
	  cell_parameters[0]=2.0*a/sqrt(3.0);
	}else{/* other angle, Rhombohedral (hR)*/
	  bravais_type = 7;
	  cell_parameters[0]=a;
	  cell_parameters[3]=alpha;
	}
      }
    }	
  }else if(fabs(beta-gamma)<smallvalue){
    /* two of three angles are equal. alpha==beta or alpha==gamma or beta==gamma*/
    /* Attention! Generally the lattice vectors are chosen so that alpha=beta */
  }else if(fabs(alpha-beta)<smallvalue){
    if(fabs(alpha-pi/2.0)<smallvalue){/* If two equal angles are right angle, it would be hP, oC or mP */
      if(fabs(a-b)<smallvalue){
        /*If the two vectors perpendicular and their lengths are equal, it would be hP or oC instead of mP*/
	if(fabs(cos(gamma)+1.0/2.0)<smallvalue){ /*If gamma is 120 degree, it is hP*/
	  bravais_type = 4;
	  cell_parameters[0]=a;
	  cell_parameters[2]=c/a;
	}else{ /*gamma is other angel, oC*/
	  /* Attention! Generally, gamma is taken as an obtuse angle. */
	  if(cos(gamma)<-1.0*smallvalue){
	    bravais_type = 11;
	    cell_parameters[0]=sqrt(a*a+b*b+2.0*a*b*cos(gamma));
	    cell_parameters[1]=sqrt(a*a+b*b-2.0*a*b*cos(gamma))/cell_parameters[0];
	    cell_parameters[2]=c/cell_parameters[0];
	  }
	}
      }else{/* mP */
	/*Attention! Generally, gamma is taken as an obtuse angle and special axis 
	  is conventionally labelled as 'b-axis' and a<c 
	  a->c, b->a, c->b*/
	if(cos(gamma)<-1.0*smallvalue && (a-b)>smallvalue){
	  bravais_type = 12;
	  cell_parameters[0]=b;
	  cell_parameters[1]=c/b;
	  cell_parameters[2]=a/b;
	  cell_parameters[4]=gamma;
	  cx = lattice_vector[0][0];
	  cy = lattice_vector[0][1];
	  cz = lattice_vector[0][2];
	  ax = lattice_vector[1][0];
	  ay = lattice_vector[1][1];
	  az = lattice_vector[1][2];
	  bx = lattice_vector[2][0];
	  by = lattice_vector[2][1];
	  bz = lattice_vector[2][2];
	  /* printf("Attention! In primitive monoclinic case, adjust lattice vectors!\n"); */
	}
      }
    }else{/* two equal angles are other angle than right angle. */
      if(fabs(a-b)<smallvalue && fabs(a-c)<smallvalue){/*all vectors' length are equal, tI*/
	/* Additional criterions are: (a+b), (a+c) and (b+c) are orthogonal to one another,
	   since the face constructed by a, b, c are diamond shape. 
	   (a+c)(b+c)=a.b+a.c+b.c+c.c=0
	   (a+b)(a+c)=a.b+a.c+b.c+a.a=0
	   (a+b)(b+c)=a.b+a.c+b.c+b.b=0
	   Attention! Generally, the lattice vectors are chose so that |a+c|=|b+c|<|a+b|
        */
	/* debug
	   printf("tI right\n");
	   printf("%20.15f",a*a+c*c+2.0*a*c*cos(beta));
	   printf("%20.15f",-(b*b+c*c+2.0*b*c*cos(alpha)));
	   printf("%20.15f",fabs(a*a+c*c+2.0*a*c*cos(beta)-(a*a+b*b+2.0*a*b*cos(gamma))));
	   printf("%20.15f",fabs(c*c+a*b*cos(gamma)+a*c*cos(beta)+b*c*cos(alpha)));
	*/
	if(fabs(a*a+c*c+2.0*a*c*cos(beta)-(b*b+c*c+2.0*b*c*cos(alpha)))<smallvalue
	   && fabs(a*a+c*c+2.0*a*c*cos(beta)-(a*a+b*b+2.0*a*b*cos(gamma)))>smallvalue
	   && fabs(c*c+a*b*cos(gamma)+a*c*cos(beta)+b*c*cos(alpha))<smallvalue){
	  bravais_type = 6;
	  cell_parameters[0]=sqrt(c*c+a*a+2.0*a*c*cos(beta));
	  cell_parameters[2]=sqrt(b*b+a*a+2.0*a*b*cos(gamma))/cell_parameters[0];
	}
      }else if(fabs(a-b)<smallvalue){/*only two vectors' length are equal, mB*/
	if(cos(alpha)<-1.0*smallvalue&&cos(beta)<-1.0*smallvalue){
	  /* Attention! Generally, alpha and beta are taken as obtuse angles */
	  bravais_type = 13;
	  cell_parameters[0]=sqrt(b*b+a*a+2.0*a*b*cos(gamma));
	  cell_parameters[1]=sqrt(b*b+a*a-2.0*a*b*cos(gamma))/cell_parameters[0];
	  cell_parameters[2]=c/cell_parameters[0];
	  cell_parameters[4]=acos(cos(alpha)/cos(gamma/2.0));
	}
      }
    }
  }else if(fabs(gamma-beta)<smallvalue){
    /* 
       Attention! Generally the lattice vectors are chosen so that alpha=beta
    */
  }else{/* all of the angles are different */
    if(fabs(a-b)<smallvalue&&fabs(b-c)<smallvalue){/*all the vectors' length are equal, oI*/
      if(fabs(c*c+a*b*cos(gamma)+a*c*cos(beta)+b*c*cos(alpha))<smallvalue
	 && (a*a+c*c+2.0*a*c*cos(beta)-(c*c+b*b+2.0*c*b*cos(alpha)))>smallvalue
	 && (a*a+b*b+2.0*a*b*cos(gamma)-(a*a+c*c+2.0*a*c*cos(beta)))>smallvalue){
	/*Attention!
    	  Additional criterions are: (a+b), (a+c) and (b+c) are orthogonal to one another,
	  since the face constructed by a, b, c are diamond shape. 
	  (a+c)(b+c)=a.b+a.c+b.c+c.c=0
	  (a+b)(a+c)=a.b+a.c+b.c+a.a=0
	  (a+b)(b+c)=a.b+a.c+b.c+b.b=0
	  Generally, the lattice vectors are chose so that |a+b|>|a+c|>|c+b|*/
	bravais_type = 9;
	cell_parameters[0]=sqrt(c*c+b*b+2.0*c*b*cos(alpha));
	cell_parameters[1]=sqrt(c*c+a*a+2.0*c*a*cos(beta))/cell_parameters[0];
	cell_parameters[2]=sqrt(b*b+a*a+2.0*b*a*cos(gamma))/cell_parameters[0];
      }
    }else if(fabs(sqrt(b*b+c*c-2.0*b*c*cos(alpha))-a)<smallvalue 
	     && fabs(sqrt(a*a+c*c-2.0*a*c*cos(beta))-b)<smallvalue 
	     && fabs(sqrt(b*b+a*a-2.0*b*a*cos(gamma))-c)<smallvalue 
	     && fabs((ax+bx-cx)*(ax+bx-cx)+(ay+by-cy)*(ay+by-cy)+(az+bz-cz)*(az+bz-cz)-
		     (ax+cx-bx)*(ax+cx-bx)-(ay+cy-by)*(ay+cy-by)-(az+cz-bz)*(az+cz-bz))>smallvalue
	     && fabs((ax+cx-bx)*(ax+cx-bx)+(ay+cy-by)*(ay+cy-by)+(az+cz-bz)*(az+cz-bz)-
		     (bx+cx-ax)*(bx+cx-ax)-(by+cy-ay)*(by+cy-ay)-(bz+cz-az)*(bz+cz-az))>smallvalue				
	     ){/* if |a|=|b-c| and |b|=|a-c| and |c|=|b-a| then oF, 
		  and take the general choice |a+b-c|>|a+c-b|>|b+c-a| 
		  Attention!*/
      bravais_type = 10;
      cell_parameters[0]=sqrt(a*a+b*b+c*c+2.0*b*c*cos(alpha)-2.0*a*b*cos(gamma)-2.0*a*c*cos(beta));
      cell_parameters[1]=sqrt(a*a+b*b+c*c+2.0*a*c*cos(beta)-2.0*b*c*cos(alpha)-2.0*a*b*cos(gamma))/cell_parameters[0];
      cell_parameters[2]=sqrt(a*a+b*b+c*c+2.0*a*b*cos(gamma)-2.0*a*c*cos(beta)-2.0*b*c*cos(alpha))/cell_parameters[0];
    }else{/*last case, it must be triclinic aP */
      /* All angles should be sharp angle and ordered so that cos(gamma)>cos(beta)>cos(alpha)>0
	 Attention!*/
      if(cos(gamma)>cos(beta)&&cos(beta)>cos(alpha)&&cos(alpha)>smallvalue){
	bravais_type = 14;
	cell_parameters[0]=a;
	cell_parameters[1]=b/a;
	cell_parameters[2]=c/a;
	cell_parameters[3]=alpha;
	cell_parameters[4]=beta;
	cell_parameters[5]=gamma;
      }
    }
  }
  lattice_vector[0][0]=ax;
  lattice_vector[0][1]=ay;
  lattice_vector[0][2]=az;
  lattice_vector[1][0]=bx;
  lattice_vector[1][1]=by;
  lattice_vector[1][2]=bz;
  lattice_vector[2][0]=cx;
  lattice_vector[2][1]=cy;
  lattice_vector[2][2]=cz;
    
  return bravais_type;
}

int Finding_Bravais_Lattice_Type(double **lattice_vector, double *cell_parameters){
	
  int bravais_type1st, bravais_type; /* Bravais lattice type */
  int i,j,k,pcell_brav;
  int L11,L12,L13,L21,L22,L23,L31,L32,L33,ID1,ID2,ID3,ID4,ID5;
  double pcell_parameters[6]; /* a, b, c and alpha, beta, gamma */
  double con_cell[6], cos1, cos2, cos3;
  double **latt1st, **rlatt, **platt, **newlatt; 
    
  rlatt=(double**)malloc(sizeof(double*)*3);
  platt=(double**)malloc(sizeof(double*)*3);
  newlatt=(double**)malloc(sizeof(double*)*3);
  latt1st=(double**)malloc(sizeof(double*)*3);
  for(i=0;i<3;i++){
    platt[i]=(double*)malloc(sizeof(double)*3);
    rlatt[i]=(double*)malloc(sizeof(double)*3);
    newlatt[i]=(double*)malloc(sizeof(double)*3);
    latt1st[i]=(double*)malloc(sizeof(double)*3);
  }                                                      
  for(i=0;i<3;i++){/*saving the original lattice vector*/
    for(j=0;j<3;j++){
      rlatt[i][j]=lattice_vector[i][j];
      latt1st[i][j]=lattice_vector[i][j];
      platt[i][j]=lattice_vector[i][j];
      newlatt[i][j]=lattice_vector[i][j];
    }
  }
  if(debug3==1){
    printf("Original lattice vectors are:\n");
    for(i=0;i<3;i++){
      printf("A%1d (%5.2f, %5.2f, %5.2f)\n",i+1,lattice_vector[i][0],lattice_vector[i][1],lattice_vector[i][2]);
    }
  }
  bravais_type1st=Bravais_Type(latt1st,cell_parameters);
  if(debug3==1){   
    if(0<bravais_type1st && bravais_type1st<15){
      printf("In subroutine Bravais_Type, it is found the Lattice is %d\n",bravais_type1st);
      printf("Cell parameters are\n");
      for(i=0;i<6;i++){
	if(cell_parameters[i]!=0.0){
	  if(i==0){ 
	    printf("Lattice parameter %d is %10.5f\n",i,cell_parameters[i]);
	  }else if(i==1||i==2){
	    printf("Lattice parameter %d is %10.5f\n",i,cell_parameters[i]*cell_parameters[0]);
	  }else{
	    printf("Angle %d information is %10.5f, cosin is %10.5f\n",i,cell_parameters[i]/pi*180.0,cos(cell_parameters[i]));
	  }
	}
      }
    }
  }
  for(i=0;i<6;i++){
    con_cell[i]=cell_parameters[i];
  }
  /* Find the shortest lattice vectors. The volume could be kept.*/
  Chk_Shorter_Lattice_Vector(rlatt);
  if(debug3==1){
    printf("The possible shortest lattice vectors are:\n");
    for(i=0;i<3;i++){
      printf("A%1d (%5.2f, %5.2f, %5.2f)\n",i+1,rlatt[i][0],rlatt[i][1],rlatt[i][2]);
    }
  }
  /*Do several combinations of these lattice vectors
    A=L11*a+L12*b+L13*c;
    B=L21*a+L22*b+L23*c;
    C=L31*a+L32*b+L33*c;
    a,b,c are the vectors found from Chk_Shorter_Lattice_Vector;
    A,B,C are the obtained vectors after linear combination of a, b, c with integer L11,L12,L13, ...
    The idea is to find the highest symmetrical bravais lattice from vectors 
    with possible combinations which conserving the volume surrounded by them.
  */
  bravais_type=bravais_type1st;
  cos1=1.0;cos2=1.0;cos3=1.0;
  for(L33=-2;L33<3;L33++){/*N9*/
    for(L32=-2;L32<3;L32++){/*N8*/
      for(L31=-2;L31<3;L31++){/*N7*/
	for(L23=-2;L23<3;L23++){/*N6*/
	  for(L22=-2;L22<3;L22++){/*N5*/
	    ID1=L22*L33-L23*L32;
	    for(L21=-2;L21<3;L21++){/*N4*/
	      ID2=L23*L31-L21*L33;ID3=L21*L32-L22*L31;  
	      for(L13=-2;L13<3;L13++){/*N3*/
		ID4=L13*ID3; 
		for(L12=-2;L12<3;L12++){/*N2*/
		  ID5=L12*ID2+ID4;
		  for(L11=-2;L11<3;L11++){/*N1*/
		    if(L11*ID1+ID5==1){/*volume conservation condition*/
		      platt[0][0] = L11*rlatt[0][0]+L12*rlatt[1][0]+L13*rlatt[2][0];
		      platt[0][1] = L11*rlatt[0][1]+L12*rlatt[1][1]+L13*rlatt[2][1];
		      platt[0][2] = L11*rlatt[0][2]+L12*rlatt[1][2]+L13*rlatt[2][2];
		      platt[1][0] = L21*rlatt[0][0]+L22*rlatt[1][0]+L23*rlatt[2][0];
		      platt[1][1] = L21*rlatt[0][1]+L22*rlatt[1][1]+L23*rlatt[2][1];
		      platt[1][2] = L21*rlatt[0][2]+L22*rlatt[1][2]+L23*rlatt[2][2];
		      platt[2][0] = L31*rlatt[0][0]+L32*rlatt[1][0]+L33*rlatt[2][0];
		      platt[2][1] = L31*rlatt[0][1]+L32*rlatt[1][1]+L33*rlatt[2][1];
		      platt[2][2] = L31*rlatt[0][2]+L32*rlatt[1][2]+L33*rlatt[2][2];    
		      pcell_brav=Bravais_Type(platt,pcell_parameters);
		      if((pcell_brav<bravais_type)||(pcell_brav==bravais_type
						     &&((fabs(cos(pcell_parameters[3]))<(cos1-smallvalue))
							&& (fabs(cos(pcell_parameters[4]))<(cos2-smallvalue))
							&& (fabs(cos(pcell_parameters[5]))<(cos3-smallvalue))))){
			bravais_type=pcell_brav;
			for(i=0;i<3;i++){/*saving the original lattice vector*/
			  for(j=0;j<3;j++){
			    newlatt[i][j]=platt[i][j];
			  }
			}
			for(i=0;i<6;i++){
			  con_cell[i]=pcell_parameters[i];
			}
			cos1=fabs(cos(pcell_parameters[3]));
			cos2=fabs(cos(pcell_parameters[4]));
			cos3=fabs(cos(pcell_parameters[5]));
		      }/*found higher symmetrical lattice*/
		    }/*keep volume*/
		  }/*L11*/
		}/*L12*/
	      }/*L13*/  	
	    }/*L21*/
	  }/*L22*/
	}/*L23*/
      }/*L31*/
    }/*L32*/
  }/*L33*/
  if(bravais_type==bravais_type1st){/* Two types are the same */
    ID1=0;
    for(i=0;i<6;i++){
      if(fabs(cell_parameters[i]-con_cell[i])>smallvalue){
	ID1=1;
	break;
      }
    }
    if(ID1==0){
      for(i=0;i<3;i++){
	for(j=0;j<3;j++){
	  newlatt[i][j]=latt1st[i][j];
	}
      }
    }else{
      if(debug3==1){
	printf("WARNING!Taking the origianl lattice vectors \nwhile the cell parameters are different from the original ones.\n");
      }
    }
  }
    
  for(i=0;i<6;i++){
    cell_parameters[i]=con_cell[i];
  }
  for(i=0;i<3;i++){
    lattice_vector[i][0]=newlatt[i][0];
    lattice_vector[i][1]=newlatt[i][1];
    lattice_vector[i][2]=newlatt[i][2];
  }
    
  for(i=0;i<3;i++){
    free(platt[i]);
    free(rlatt[i]);
    free(newlatt[i]);
    free(latt1st[i]);
  }                                                      
  free(rlatt);
  free(platt);
  free(newlatt);
  free(latt1st);
  
  return bravais_type;
}
   
void Matrix_Copy23(int **source,int n_source,int ***symop, int k_symop){
  /* copy source to symop*/	
  int i,j;

  for(i=0; i<n_source; i++){
    for(j=0; j<n_source; j++){
      symop[k_symop][i][j]=source[i][j];
    }
  }
  return;
}

void Matrix_Copy32(int **target,int n_target,int ***symop, int k_symop)
{
  /* copy from symop to target*/	
  int i,j;

  for(i=0; i<n_target; i++){
    for(j=0; j<n_target; j++){
      target[i][j] = symop[k_symop][i][j];
    }
  }
}

void Matrix_Copy22(int **source,int **target, int dim)
{
  /* copy from symop to target*/	
  int i,j;

  for(i=0; i<dim; i++){
    for(j=0; j<dim; j++){
      target[i][j] = source[i][j];
    }
  }
}

int Matrix_Equal(int **m1, int **m2, int dim){
  int i,j;
	
  for(i=0;i<dim;i++){
    for(j=0;j<dim;j++){
      if(m1[i][j]!=m2[i][j]){
	return 0;
      }
    }
  }
  return 1;
}

void Matrix_Productor(int dim, int **m1, int **m2, int **pro){
  int i,j,k,sum;	
  for(i=0;i<dim;i++){
    for(j=0;j<dim;j++){
      sum=0;
      for(k=0;k<dim;k++){
	sum+=m1[i][k]*m2[k][j];
      }
      pro[i][j]=sum;
    }
  }
  return;
}

int Symmtry_Operator_Generation(int ***symgen,int gen_num,int ***symop){
  /* Generating the symmetry operation matrix by using gen_num generators stored in symop */
  int op_num,op_i,existing, op_count, op2;
  int i,j,k,ii;
  int **iden,**genop, **exop, **tmpop, **tmpop1;
  int op_order, iorder;
  int new1,new2,m,n;		
    
  existing=0;
    
  iden=(int**)malloc(sizeof(int*)*3);
  genop=(int**)malloc(sizeof(int*)*3);
  exop=(int**)malloc(sizeof(int*)*3);
  tmpop=(int**)malloc(sizeof(int*)*3);
  tmpop1=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    iden[i]=(int*)malloc(sizeof(int)*3);
    genop[i]=(int*)malloc(sizeof(int)*3);
    exop[i]=(int*)malloc(sizeof(int)*3);
    tmpop[i]=(int*)malloc(sizeof(int)*3);
    tmpop1[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      iden[i][j]=0;
      genop[i][j]=0;
      exop[i][j]=0;
      tmpop[i][j]=0;
      tmpop1[i][j]=0;
    }
  }
  iden[0][0]=1;iden[1][1]=1;iden[2][2]=1;

  Matrix_Copy23(iden,3,symop,0);
  op_num=1;
		
  for(k=0;k<gen_num;k++){
    existing=0;
    Matrix_Copy32(genop,3,symgen,k);/* take out one generator */
    if(debug3 == 1){
      printf("taking %2d generator\n",k);
      for(i=0;i<3;i++){
	printf("%3d%3d%3d\n",genop[i][0],genop[i][1],genop[i][2]);
      }
    }
    for(op_i=0;op_i<op_num;op_i++){/* compare this generator with existing operators */
      Matrix_Copy32(exop,3,symop,op_i);
      if(debug3 == 1){
	printf("existing symmetry %2d\n",op_i);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",exop[i][0],exop[i][1],exop[i][2]);
	}
      }
      if(Matrix_Equal(genop,exop,3)==1){/* whether this generator already exists */
	existing=1;
	break;
      }
    }
    if(existing==1){/* If already exists, go to the next generator */
      continue;
    }
    Matrix_Productor(3, genop, iden, exop);/*find the order of generator */
    if(debug3 == 1){
      printf("after productor\n");
      for(i=0;i<3;i++){
	printf("%3d%3d%3d\n",exop[i][0],exop[i][1],exop[i][2]);
      }
    }	
    for(iorder=1;iorder<50;iorder++){
      if(Matrix_Equal(exop,iden,3)==1){
	op_order=iorder;
	if(debug3 == 1){
	  printf("Found the Order is %d\n",op_order);
	}
	break;	
      }
      Matrix_Productor(3,genop,exop,tmpop);
      Matrix_Copy22(tmpop,exop,3);
    }
    /* printf("Haha\n Order is %d\n",op_order); */
    op_count=op_num; 
    for(op_i=0;op_i<op_num;op_i++){/* for all existing symmetry operator  exop */
      Matrix_Copy32(exop,3,symop,op_i);
      for(iorder=1;iorder<op_order;iorder++){
	Matrix_Productor(3,genop,exop,tmpop);
	Matrix_Copy22(tmpop,exop,3); /*exop=genop**Iorder*symop  exop == H */
	for(i=0;i<op_num;i++){ /* time another existing operator */
	  Matrix_Copy32(tmpop,3,symop,i);
	  Matrix_Productor(3,tmpop,exop,tmpop1);/* tmpop1 == HH */
	  existing=0;
	  for(j=0;j<op_count;j++){
	    Matrix_Copy32(tmpop,3,symop,j);
	    if(Matrix_Equal(tmpop,tmpop1,3)==1){
	      existing=1;
	      break;
	    }
	  }
	  if(existing==1){continue;}
	  /* if not existing, found one new symmetry operator */
	  Matrix_Copy23(tmpop1,3,symop,op_count);	
	  if(debug3 == 1){
	    printf("111found one now is %d\n",op_count+1);
	    for(ii=0;ii<3;ii++){
	      printf("%3d%3d%3d\n",tmpop1[ii][0],tmpop1[ii][1],tmpop1[ii][2]);
	    }
	  }
	  op_count++;
	  if(op_count>48){printf("111Error!Over 48\n");return 0;/*error*/}
	}
      }
      if(op_i==0){op2=op_count;}
    }
    /*Products with more than one sandwiched SIGMA-factor:*/
    new1=op_num;
    new2=op_count;
    for(i=1;i<50;i++){
      for(n=op_num;n<op2;n++){
	for(m=new1;m<new2;m++){
	  Matrix_Copy32(tmpop,3,symop,n);
	  Matrix_Copy32(tmpop1,3,symop,m);
	  Matrix_Productor(3,tmpop,tmpop1,exop);
	  existing=0;
	  for(j=0;j<op_count;j++){
	    Matrix_Copy32(tmpop,3,symop,j);
	    if(Matrix_Equal(tmpop,exop,3)==1){
	      existing=1;
	      break;
	    }
	  }
	  if(existing!=1){
	    Matrix_Copy23(exop,3,symop,op_count);
	    if(debug3 == 1){
	      printf("222found one now is %d\n",op_count+1);
	      for(ii=0;ii<3;ii++){
		printf("%3d%3d%3d\n",tmpop1[ii][0],tmpop1[ii][1],tmpop1[ii][2]);
	      }
	    }
	    op_count++;
	    if(op_count>48){printf("222Error!Over 48\n");return 0;}
	  }
	}
      }
      if(new2==op_count){break;}
      new1=new2+1;
      new2=op_count;	
    }
    op_num=op_count;
  }

  for(i=0;i<3;i++){
    free(iden[i]);
    free(genop[i]);
    free(exop[i]);
    free(tmpop[i]);
    free(tmpop1[i]);
  }
  free(iden);
  free(genop);
  free(exop);
  free(tmpop);
  free(tmpop1);
  
  return op_num;
}



int Bravais_Lattice_Symmetry(int bravais_type, int ***symop){
  /*Bravais_Lattice_Symmetry generates the symmetry operation of the pure bravais lattice determined above,
    without consideration of actual atomic arrangement (lattice basis) 
    Input
    bravais_type
    OUTPUT
    sym_op              symmetry operation matrix   
    op_num              number of symmetry operation 
  */    
  int op_num;
		
  int ***symgen;
  int **inv, **rot3D, **rot6Z, **rot2hex, **rot2Ybc, **rot2Zbc, **rot2Ybas, **rot2Yfc, **rot2Zfc;
  int **rot2Tri, **rot4Zpri, **rot2Ypri, **rot4Zbc, **rot4Zfc, **rot2Zpri;
  
  int i,j,k;

  /*initiallize the operation generators */        
  inv=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    inv[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      inv[i][j]=0;
    }
  }
  inv[0][0]=-1;inv[1][1]=-1;inv[2][2]=-1;
  /*
    -1   0    0
    0   -1   0 
    0   0    -1
    DATA INV /-1,0,0,0,-1,0,0,0,-1/
  */
    
  rot3D=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot3D[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot3D[i][j]=0;
    }
  }
  rot3D[0][1]=1;rot3D[1][2]=1;rot3D[2][0]=1;
  /*
    0   1   0
    0   0   1 
    1   0   0
    R3D /0,0,1,1,0,0,0,1,0/
  */
    
  rot4Zpri=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot4Zpri[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot4Zpri[i][j]=0;
    }
  }
  rot4Zpri[0][1]=1;rot4Zpri[1][0]=-1;rot4Zpri[2][2]=1;
  /*
    0   1  0
    -1  0  0
    0   0  1
    R4ZP /0,-1,0,1,0,0,0,0,1/
  */
        
  rot2hex=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2hex[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2hex[i][j]=0;
    }
  }
  rot2hex[0][0]=1;rot2hex[1][0]=-1;rot2hex[1][1]=-1;rot2hex[2][2]=-1;
  /*
    1   0    0
    -1  -1   0
    0   0    -1
    R2HEX /1,-1,0,0,-1,0,0,0,-1/
  */

  rot6Z=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot6Z[i]=(int*)malloc(sizeof(int)*3);
  }

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot6Z[i][j]=0;
    }
  }
  rot6Z[0][0]=1;rot6Z[0][1]=1;rot6Z[1][0]=-1;rot6Z[2][2]=1;
  /*
    1    1     0
    -1   0     0
    0    0     1
    DATA R6Z /1,-1,0,1,0,0,0,0,1/
  */
    
  rot2Tri=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Tri[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Tri[i][j]=0;
    }
  }
  rot2Tri[0][0]=-1;rot2Tri[1][2]=-1;rot2Tri[2][1]=-1;
  /*
    -1  0   0
    0   0   -1
    0   -1  0
    DATA R2TRI /-1,0,0,0,0,-1,0,-1,0/
  */

    
    
  rot2Ypri=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Ypri[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Ypri[i][j]=0;
    }
  }
  rot2Ypri[0][0]=-1;rot2Ypri[1][1]=1;rot2Ypri[2][2]=-1;
  /*
    -1  0   0
    0   1   0
    0   0   -1
    DATA R2YP /-1,0,0,0,1,0,0,0,-1/
  */
    
  rot4Zbc=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot4Zbc[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot4Zbc[i][j]=0;
    }
  }
  rot4Zbc[0][2]=-1;rot4Zbc[1][0]=1;rot4Zbc[1][1]=1;rot4Zbc[1][2]=1;rot4Zbc[2][1]=-1;
  /*
    0  0   -1
    1  1   1
    0  -1  0
    R4ZBC /0,1,0,0,1,-1,-1,1,0/
  */

  rot4Zfc=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot4Zfc[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot4Zfc[i][j]=0;
    }
  }
  rot4Zfc[0][0]=1;rot4Zfc[0][2]=-1;rot4Zfc[1][0]=1;rot4Zfc[2][0]=1;rot4Zfc[2][1]=-1;
  /*
    1  0   -1
    1  0   0
    1  -1  0
    DATA R4ZFC /1,1,1,0,0,-1,-1,0,0/
  */

  rot2Zpri=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Zpri[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Zpri[i][j]=0;
    }
  }
  rot2Zpri[0][0]=-1;rot2Zpri[1][1]=-1;rot2Zpri[2][2]=1;
  /*
    -1  0   0
    0  -1   0
    0   0   1
    R2ZP /-1,0,0,0,-1,0,0,0,1/
  */

  rot2Ybc=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Ybc[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Ybc[i][j]=0;
    }
  }
  rot2Ybc[0][2]=1;rot2Ybc[1][0]=-1;rot2Ybc[1][1]=-1;rot2Ybc[1][2]=-1;rot2Ybc[2][0]=1;
  /*
    0   0   1
    -1  -1  -1
    1   0   0
    DATA R2YBC /0,-1,1,0,-1,0,1,-1,0/
  */
    
  rot2Zbc=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Zbc[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Zbc[i][j]=0;
    }
  }
  rot2Zbc[0][1]=1;rot2Zbc[1][0]=1;rot2Zbc[2][0]=-1;rot2Zbc[2][1]=-1;rot2Zbc[2][2]=-1;
  /*
    0   1    0
    1   0    0
    -1  -1  -1
    R2ZBC /0,1,-1,1,0,-1,0,0,-1/
  */

  rot2Ybas=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Ybas[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Ybas[i][j]=0;
    }
  }
  rot2Ybas[0][1]=-1;rot2Ybas[1][0]=-1;rot2Ybas[2][2]=-1;
  /*
    0   -1  0
    -1  0   0
    0   0   -1
    DATA R2YBAS /0,-1,0,-1,0,0,0,0,-1/
  */
    
  rot2Yfc=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Yfc[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Yfc[i][j]=0;
    }
  }
  rot2Yfc[0][1]=-1;rot2Yfc[0][2]=1;rot2Yfc[1][1]=-1;rot2Yfc[2][0]=1;rot2Yfc[2][1]=-1;
  /*
    0  -1   1
    0  -1   0
    1  -1   0
    R2YFC /0,0,1,-1,-1,-1,1,0,0/
  */
    
  rot2Zfc=(int**)malloc(sizeof(int*)*3);
  for(i=0;i<3;i++){
    rot2Zfc[i]=(int*)malloc(sizeof(int)*3);
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      rot2Zfc[i][j]=0;
    }
  }
  rot2Zfc[0][1]=1;rot2Zfc[0][2]=-1;rot2Zfc[1][0]=1;rot2Zfc[1][2]=-1;rot2Zfc[2][2]=-1;
  /*
    0  1   -1
    1  0   -1
    0  0   -1
    DATA R2ZFC /0,1,0,1,0,0,-1,-1,-1/
  */

    
  symgen=(int***)malloc(sizeof(int**)*48);
  for(k=0; k<48; k++){
    symgen[k] = (int**)malloc(sizeof(int*)*3);
    for(i=0;i<3;i++){
      symgen[k][i] = (int*)malloc(sizeof(int)*3);
    }
  }
    
  for (k=0; k<48; k++){
    for (i=0;i<3;i++){
      for (j=0;j<3;j++){
	symgen[k][i][j]=0;
	symop[k][i][j]=0;
      }
    }
  }
  /* debug printf("in Bravais_lattice_Symmetry lattice is %d\n",bravais_type); */
  Matrix_Copy23(inv,3,symgen,0);
    
  switch(bravais_type){
  case 1: /* Primitive cubic cP */
    Matrix_Copy23(rot3D,3,symgen,1);
    Matrix_Copy23(rot4Zpri,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;

  case 2: /* Body-centered cubic (cI)*/
    Matrix_Copy23(rot3D,3,symgen,1);
    Matrix_Copy23(rot4Zbc,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;
    	
  case 3: /* Face-centered cubic (cF)*/
    Matrix_Copy23(rot3D,3,symgen,1);
    Matrix_Copy23(rot4Zfc,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;

  case 4: /* Primitive hexagonal (hP) */
    Matrix_Copy23(rot6Z,3,symgen,1);
    Matrix_Copy23(rot2hex,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;
    	
  case 5: /* Primitive tetragonal (tP)*/
    Matrix_Copy23(rot4Zpri,3,symgen,1);
    Matrix_Copy23(rot2Ypri,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;
    	
  case 6: /* Body-centred tetragonal (tI) */
    Matrix_Copy23(rot4Zbc,3,symgen,1);
    Matrix_Copy23(rot2Ybc,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;

  case 7: /*Rhombohedral (hR) */
    Matrix_Copy23(rot2Tri,3,symgen,1);
    Matrix_Copy23(rot3D,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;
    	
  case 8: /* Primitive Orthorhombic (oP)*/
    Matrix_Copy23(rot2Zpri,3,symgen,1);
    Matrix_Copy23(rot2Ypri,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;
    	
  case 9: /* Body-centred Orthorhombic (oI) */
    Matrix_Copy23(rot2Zbc,3,symgen,1);
    Matrix_Copy23(rot2Ybc,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;

  case 10: /* Face-centred Orthorhombic (oF) */
    Matrix_Copy23(rot2Zfc,3,symgen,1);
    Matrix_Copy23(rot2Yfc,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;
    	
  case 11: /* Base-centred Orthorhombic (oC) */
    Matrix_Copy23(rot2Zpri,3,symgen,1);
    Matrix_Copy23(rot2Ybas,3,symgen,2);
    if(debug3 == 1){
      printf("3 generator\n");
      for(k=0;k<3;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,3,symop);
    break;

  case 12: /* Primitive monoclinic (mP) */
    Matrix_Copy23(rot2Ypri,3,symgen,1);
    if(debug3 == 1){
      printf("2 generator\n");
      for(k=0;k<2;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,2,symop);
    break;
    		
  case 13:/* Base-centred monoclinic (mB) */
    Matrix_Copy23(rot2Ybas,3,symgen,1);
    if(debug3 == 1){
      printf("2 generator\n");
      for(k=0;k<2;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }    	  
    op_num=Symmtry_Operator_Generation(symgen,2,symop);
    break;	 
    	
  case 14:/* Primitive triclinic (aP) */
    if(debug3 == 1){
      printf("1 generator\n");
      for(k=0;k<1;k++){
	printf("%3d\n",k);
	for(i=0;i<3;i++){
	  printf("%3d%3d%3d\n",symgen[k][i][0],symgen[k][i][1],symgen[k][i][2]);
	}
      }
    }
    op_num=Symmtry_Operator_Generation(symgen,1,symop);
    break;	 
  default:
    printf("Eorr in lattice type!\n");
    return 0;
    break;
  }      
  
  for(i=0; i<3; i++){
    free(inv[i]);
    free(rot3D[i]);
    free(rot6Z[i]);
    free(rot2hex[i]);
    free(rot2Ybc[i]);
    free(rot2Zbc[i]);
    free(rot2Ybas[i]);
    free(rot2Yfc[i]);
    free(rot2Zfc[i]);
    free(rot2Tri[i]);
    free(rot4Zpri[i]);
    free(rot2Ypri[i]);
    free(rot4Zbc[i]);
    free(rot4Zfc[i]);
    free(rot2Zpri[i]);
  }
  free(inv);
  free(rot3D);
  free(rot6Z);
  free(rot2hex);
  free(rot2Ybc);
  free(rot2Zbc);
  free(rot2Ybas);
  free(rot2Yfc);
  free(rot2Zfc);
  free(rot2Tri);
  free(rot4Zpri);
  free(rot2Ypri);
  free(rot4Zbc);
  free(rot4Zfc);
  free(rot2Zpri);
  
  for(k=0; k<48; k++){
    for(i=0; i<3; i++){
      free(symgen[k][i]);
    }
    free(symgen[k]);
  }
  free(symgen);

  return op_num;
}

void Ascend_Ordering(double *xyz_value, int *ordering, int tot_atom){
  int i,j,k, tmp_order;
  double tmp_xyz;
	
  for(i=1;i<tot_atom; i++){/* taking one value */
    for(j=i;j>0;j--){ /* compare with all the other lower index value */
      if(xyz_value[j]<xyz_value[j-1]){/* if it is smaller than lower index value, exchange */
	tmp_xyz=xyz_value[j];
	xyz_value[j]=xyz_value[j-1];
	xyz_value[j-1]=tmp_xyz;
	tmp_order = ordering[j];
	ordering[j]=ordering[j-1];
	ordering[j-1]=tmp_order;
      }
    }
  }
  return;
}

void Ordering_Atomic_Position(double **atomic_position, int start_atom_indx, int end_atom_indx){
  int atom_indx, tot_atom;
  int i,j,k, num_x, num_y, start_i;
  double **ordered_atom_pos;
  double *xyz_value;
  double tmp_value;
	
  int *ordering;/* memorizing the atom_indx in the ascending order */
	
  tot_atom=end_atom_indx-start_atom_indx;
  if(debug2==1){
    printf("To be ordered atomic position is:\n");
    for(i=0;i<tot_atom;i++){
      atom_indx=i+start_atom_indx;
      printf("atom %2d %12.8f  %12.8f  %12.8f\n",atom_indx,atomic_position[atom_indx][0],atomic_position[atom_indx][1],atomic_position[atom_indx][2]);
    }
  }

  ordered_atom_pos=(double**)malloc(sizeof(double*)*tot_atom);
  xyz_value=(double*)malloc(sizeof(double)*tot_atom);
  ordering=(int*)malloc(sizeof(int*)*tot_atom);
  for(k=0;k<tot_atom;k++){
    ordered_atom_pos[k]=(double*)malloc(sizeof(double)*3);
    xyz_value[k]=0.0;
    ordering[k]=0;
  }
	
  /* Firstly, arrange x-coordinates in ascending order */
  for(i=0;i<tot_atom;i++){
    atom_indx=i+start_atom_indx;
    xyz_value[i]=atomic_position[atom_indx][0];
    ordering[i]=atom_indx;
  }
  Ascend_Ordering(xyz_value, ordering, tot_atom);
	
  for(k=0;k<tot_atom;k++){
    ordered_atom_pos[k][0]=atomic_position[ordering[k]][0];
    ordered_atom_pos[k][1]=atomic_position[ordering[k]][1];
    ordered_atom_pos[k][2]=atomic_position[ordering[k]][2];
  }
  for(k=0;k<tot_atom;k++){
    atomic_position[k+start_atom_indx][0]=ordered_atom_pos[k][0];
    atomic_position[k+start_atom_indx][1]=ordered_atom_pos[k][1];
    atomic_position[k+start_atom_indx][2]=ordered_atom_pos[k][2];
  }
  if(debug2==1){
    printf("after ordering x-value:\n");
    for(i=0;i<tot_atom;i++){
      atom_indx=i+start_atom_indx;
      printf("atom %2d %12.8f  %12.8f  %12.8f\n",atom_indx,atomic_position[atom_indx][0],atomic_position[atom_indx][1],atomic_position[atom_indx][2]);
    }
  }
  /* Secondly, arrange y-coordinats those have same x-coordinates in ascending order */
  tmp_value=atomic_position[start_atom_indx][0];
  num_x=1;
  start_i=0;
  xyz_value[num_x-1]=atomic_position[start_atom_indx][1];
  ordering[num_x-1]=start_atom_indx;
  
  for(i=1;i<tot_atom;i++){
    atom_indx=i+start_atom_indx;
    if(tmp_value==atomic_position[atom_indx][0]){
      num_x++;
      xyz_value[num_x-1]=atomic_position[atom_indx][1];
      ordering[num_x-1]=atom_indx;
      if(i==tot_atom-1){
	Ascend_Ordering(xyz_value, ordering, num_x);
	for(k=0;k<num_x;k++){
	  ordered_atom_pos[k][0]=atomic_position[ordering[k]][0];
	  ordered_atom_pos[k][1]=atomic_position[ordering[k]][1];
	  ordered_atom_pos[k][2]=atomic_position[ordering[k]][2];
	}
	for(k=0;k<num_x;k++){
	  atomic_position[start_i+k+start_atom_indx][0]=ordered_atom_pos[k][0];
	  atomic_position[start_i+k+start_atom_indx][1]=ordered_atom_pos[k][1];
	  atomic_position[start_i+k+start_atom_indx][2]=ordered_atom_pos[k][2];
	}
      }
    }else{
      Ascend_Ordering(xyz_value, ordering, num_x);
      for(k=0;k<num_x;k++){
	ordered_atom_pos[k][0]=atomic_position[ordering[k]][0];
	ordered_atom_pos[k][1]=atomic_position[ordering[k]][1];
	ordered_atom_pos[k][2]=atomic_position[ordering[k]][2];
      }
      for(k=0;k<num_x;k++){
	atomic_position[start_i+k+start_atom_indx][0]=ordered_atom_pos[k][0];
	atomic_position[start_i+k+start_atom_indx][1]=ordered_atom_pos[k][1];
	atomic_position[start_i+k+start_atom_indx][2]=ordered_atom_pos[k][2];
      }
      start_i=i;
      num_x=1;
      tmp_value=atomic_position[atom_indx][0];
      xyz_value[num_x-1]=atomic_position[atom_indx][1];
      ordering[num_x-1]=atom_indx;
    }
  }
  if(debug2==1){
    printf("after ordering x,y-value:\n");
    for(i=0;i<tot_atom;i++){
      atom_indx=i+start_atom_indx;
      printf("atom %2d %12.8f  %12.8f  %12.8f\n",atom_indx,atomic_position[atom_indx][0],atomic_position[atom_indx][1],atomic_position[atom_indx][2]);
    }
  }
  /* At last, arrange z-coordinats those have same y-coordinates in ascending order 
     Now just looking at the y values, x values are not necessarily monitored.*/
  tmp_value=atomic_position[start_atom_indx][1];
  num_x=1;
  start_i=0;
  xyz_value[num_x-1]=atomic_position[start_atom_indx][2];
  ordering[num_x-1]=start_atom_indx;
  for(i=1;i<tot_atom;i++){
    atom_indx=i+start_atom_indx;
    if(tmp_value==atomic_position[atom_indx][1]){
      num_x++;
      xyz_value[num_x-1]=atomic_position[atom_indx][2];
      ordering[num_x-1]=atom_indx;
      if(i==tot_atom-1){
	Ascend_Ordering(xyz_value, ordering, num_x);
	for(k=0;k<num_x;k++){
	  ordered_atom_pos[k][0]=atomic_position[ordering[k]][0];
	  ordered_atom_pos[k][1]=atomic_position[ordering[k]][1];
	  ordered_atom_pos[k][2]=atomic_position[ordering[k]][2];
	}
	for(k=0;k<num_x;k++){
	  atomic_position[start_i+k+start_atom_indx][0]=ordered_atom_pos[k][0];
	  atomic_position[start_i+k+start_atom_indx][1]=ordered_atom_pos[k][1];
	  atomic_position[start_i+k+start_atom_indx][2]=ordered_atom_pos[k][2];
	}
      }
    }else{
      Ascend_Ordering(xyz_value, ordering, num_x);
      for(k=0;k<num_x;k++){
	ordered_atom_pos[k][0]=atomic_position[ordering[k]][0];
	ordered_atom_pos[k][1]=atomic_position[ordering[k]][1];
	ordered_atom_pos[k][2]=atomic_position[ordering[k]][2];
      }
      for(k=0;k<num_x;k++){
	atomic_position[start_i+k+start_atom_indx][0]=ordered_atom_pos[k][0];
	atomic_position[start_i+k+start_atom_indx][1]=ordered_atom_pos[k][1];
	atomic_position[start_i+k+start_atom_indx][2]=ordered_atom_pos[k][2];
      }
      start_i=i;
      num_x=1;
      tmp_value=atomic_position[atom_indx][1];
      xyz_value[num_x-1]=atomic_position[atom_indx][2];
      ordering[num_x-1]=atom_indx;
    }
  }
  /* OK, now all the ordering work done */
  if(debug2==1){
    printf("after ordering x,y,z-value:\n");
    for(i=0;i<tot_atom;i++){
      atom_indx=i+start_atom_indx;
      printf("atom %2d %12.8f  %12.8f  %12.8f\n",atom_indx,atomic_position[atom_indx][0],atomic_position[atom_indx][1],atomic_position[atom_indx][2]);
    }  
  }
  
  for(k=0;k<tot_atom;k++){
    free(ordered_atom_pos[k]);
  }
  free(ordered_atom_pos);
  free(xyz_value);
  free(ordering);

  return;
}

int Chk_Pure_Point_Group_Op(int **exop, double **atomic_position, int *atom_species, 
			    int atom_num, int atom_type, double *trans_vec){
  /* 
     check whether this operation (exop) is pure point group operation or not, if not, find the
     connected translation vector (tmp_vec)
     INPUT  
     exop                                                  one symmetry operation
     atomic_position[atom_indx][atom_type_indx][3]         lattice basis (atomic arrangement)
     atom_num                                              total number of atoms in the lattice
     atom_type                                             number of atoms' species
     atom_species                                          
     OUTPUT
     pure_point                                             Whether the given operation is pure point group operation or not
     tmp_vec                                                the translation vector for non pure point group operation
     pure_point = Chk_Pure_Point_Group_Op(exop,atomic_position,atom_species,atom_num,atom_type,tmp_vec);
  */
    	
  int atom_indx, atom_type_indx, start_atom_indx, end_atom_indx,atom_indx2;
  int pure_point,foundone;
  int i,j,k;
  double **roted_xyz;
  double *markone, *tmp_vec, *diff_vec;
  char c;
  int smallest_species;
    
  foundone=0;    
  roted_xyz=(double**)malloc(sizeof(double*)*atom_num);
  markone=(double*)malloc(sizeof(double)*3);
  tmp_vec=(double*)malloc(sizeof(double)*3);
  diff_vec=(double*)malloc(sizeof(double)*3);
  for(k=0; k<atom_num; k++){
    roted_xyz[k] = (double*)malloc(sizeof(double)*3);
    for(i=0; i<3; i++){
      roted_xyz[k][i] = 0.0;
      tmp_vec[i]=3.0;
    }
  }
  /*to be consistent wit VASP, finding the species having smallest number of atom*/    
  j=atom_species[0];
  smallest_species=0;
  for(i=1;i<atom_type;i++){
    if((atom_species[i]-atom_species[i-1])<j){
      j=atom_species[i]-atom_species[i-1];
      smallest_species=i;
    }	
  }
  if(debug3==1){
    printf("smallest number of species is %2d\n",smallest_species);
  }
  for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
    if(atom_type_indx==0){
      start_atom_indx=0;/* The atoms of first species are stored from 0 index */
    }else{
      start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
    }
    end_atom_indx=atom_species[atom_type_indx];
      
    for(atom_indx=start_atom_indx;atom_indx<end_atom_indx;atom_indx++){
      /* 1. Precess the atomic coordinates, make them be [-0.5, 0.5) 
	 (Applying periodic boundary condition)
      */ 	
      if(debug3==1){
	printf("atom %3d ",atom_indx);
      }
      for(i=0;i<3;i++){
	/* make it positive and less than +1.0 */
	if(debug3==1){
	  printf("%12.8f-->",atomic_position[atom_indx][i]);
	}
	atomic_position[atom_indx][i]=atomic_position[atom_indx][i]-floor(atomic_position[atom_indx][i]);
	atomic_position[atom_indx][i]=fmod(atomic_position[atom_indx][i]+1000.5,1.0)-0.5;
	if(debug3==1){
	  printf("%10.8f",atomic_position[atom_indx][i]);
	}
      }
      if(debug3==1){
	printf("\n");
      }
    } 
    /* 2. Ordering the atomic coordinates of same species with index from start to end. 
       firstly x- then y-, z- last. */
    Ordering_Atomic_Position(atomic_position, start_atom_indx, end_atom_indx);  
    /* 3. Applying symmetry operation to each atom's coordinates */
    if(debug3==1){
      printf("NOW applying Symmetry Operation:\n");
    }
    for(atom_indx=start_atom_indx;atom_indx<end_atom_indx;atom_indx++){
      /*
	roted_xyz[atom_indx][0]=exop[0][0]*atomic_position[atom_indx][0]+exop[0][1]*atomic_position[atom_indx][1]+exop[0][2]*atomic_position[atom_indx][2];
	roted_xyz[atom_indx][1]=exop[1][0]*atomic_position[atom_indx][0]+exop[1][1]*atomic_position[atom_indx][1]+exop[1][2]*atomic_position[atom_indx][2];
	roted_xyz[atom_indx][2]=exop[2][0]*atomic_position[atom_indx][0]+exop[2][1]*atomic_position[atom_indx][1]+exop[2][2]*atomic_position[atom_indx][2];
      */
      roted_xyz[atom_indx][0]=exop[0][0]*atomic_position[atom_indx][0]+exop[1][0]*atomic_position[atom_indx][1]+exop[2][0]*atomic_position[atom_indx][2];
      roted_xyz[atom_indx][1]=exop[0][1]*atomic_position[atom_indx][0]+exop[1][1]*atomic_position[atom_indx][1]+exop[2][1]*atomic_position[atom_indx][2];
      roted_xyz[atom_indx][2]=exop[0][2]*atomic_position[atom_indx][0]+exop[1][2]*atomic_position[atom_indx][1]+exop[2][2]*atomic_position[atom_indx][2];

      /* Similarly, applying periodic boundary condition*/ 	    
      if(debug3==1){
	printf("atom %3d ",atom_indx);
      }
      for(i=0;i<3;i++){
	/* make it positive and less than +1.0 */
	if(debug3==1){
	  printf("%12.8f-->",atomic_position[atom_indx][i]);
	}
	roted_xyz[atom_indx][i]=roted_xyz[atom_indx][i]-floor(roted_xyz[atom_indx][i]);
	roted_xyz[atom_indx][i]=fmod(roted_xyz[atom_indx][i]+1000.5,1.0)-0.5;
	if(debug3==1){
	  printf("%10.8f",roted_xyz[atom_indx][i]);
	}
      }
      if(debug3==1){
	printf("\n");
      }
    }
    Ordering_Atomic_Position(roted_xyz, start_atom_indx, end_atom_indx);  
  }
  /* Comparing the original and rotated atomic positions to check the symmetry operation 
     or to find the translation vector connecting rotated and original coordinates*/
    
  if(smallest_species==0){
    start_atom_indx=0;
  }else{
    start_atom_indx=atom_species[smallest_species-1];
  }
  markone[0]=roted_xyz[start_atom_indx][0];
  markone[1]=roted_xyz[start_atom_indx][1];
  markone[2]=roted_xyz[start_atom_indx][2];
  if(debug3==1){
    printf("MARKED atom is %12.8f%12.8f%12.8f\n",markone[0],markone[1],markone[2]);
  }
  for(atom_indx=start_atom_indx;atom_indx<atom_species[smallest_species];atom_indx++){
    trans_vec[0]=atomic_position[atom_indx][0]-markone[0];
    trans_vec[1]=atomic_position[atom_indx][1]-markone[1];
    trans_vec[2]=atomic_position[atom_indx][2]-markone[2];
    	
    trans_vec[0]=fmod(trans_vec[0]+1000.0,1.0);
    trans_vec[1]=fmod(trans_vec[1]+1000.0,1.0);
    trans_vec[2]=fmod(trans_vec[2]+1000.0,1.0);
    if(fabs(trans_vec[0]-1.0)<smallvalue){trans_vec[0]=0.0;}
    if(fabs(trans_vec[1]-1.0)<smallvalue){trans_vec[1]=0.0;}
    if(fabs(trans_vec[2]-1.0)<smallvalue){trans_vec[2]=0.0;}
    if(debug3==1){	
      printf("atom %2d and ",atom_indx);
      printf("TRANS_VEC %12.8f%12.8f%12.8f\n",trans_vec[0],trans_vec[1],trans_vec[2]);
      printf("tmp_vec %12.8f%12.8f%12.8f\n",tmp_vec[0],tmp_vec[1],tmp_vec[2]);
      /* c=getchar(); */
    }
    	
    if(trans_vec[0]>(smallvalue+tmp_vec[0])
       ||trans_vec[1]>(smallvalue+tmp_vec[1])
       ||trans_vec[2]>(smallvalue+tmp_vec[2])){
      if(debug3==1){
	printf("Ignore too large vector\n");
      }
    }else{
      /* translate the rotated coordinates with trans_vec and compare with original ones*/
      if(debug3==1){
	printf("applying TRANSLATION vectors:\n");
      }
      for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
	if(atom_type_indx==0){
	  start_atom_indx=0;/* The atoms of first species are stored from 0 index */
      	}else{
	  start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
      	}
      	end_atom_indx=atom_species[atom_type_indx];
      
      	for(atom_indx2=start_atom_indx;atom_indx2<end_atom_indx;atom_indx2++){
	  /*      Applying periodic boundary condition */ 	      
          if(debug3==1){
	    printf("atom %3d ",atom_indx2);
      	  }
	  for(i=0;i<3;i++){
	    /* make it positive and less than +1.0 */
	    if(debug3==1){
	      printf("%12.8f-->",roted_xyz[atom_indx2][i]);
	    }
	    roted_xyz[atom_indx2][i]=roted_xyz[atom_indx2][i]+trans_vec[i];
	    roted_xyz[atom_indx2][i]=roted_xyz[atom_indx2][i]-floor(roted_xyz[atom_indx2][i]);
	    roted_xyz[atom_indx2][i]=fmod(roted_xyz[atom_indx2][i]+1000.5,1.0)-0.5;
	    if(debug3==1){
	      printf("%10.8f",roted_xyz[atom_indx2][i]);
	    }
	  }
	  if(debug3==1){
	    printf("\n");
      	  }
      	} 
	/*    Ordering the atomic coordinates of same species with index from start to end. 
	      firstly x- then y-, z- last. */
	Ordering_Atomic_Position(roted_xyz, start_atom_indx, end_atom_indx);  
      }/*applying translation */
      /* after translation, now comparing */
      pure_point=0;
      for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
	if(atom_type_indx==0){
	  start_atom_indx=0;/* The atoms of first species are stored from 0 index */
	}else{
	  start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
	}
	end_atom_indx=atom_species[atom_type_indx];
	for(atom_indx2=start_atom_indx;atom_indx2<end_atom_indx;atom_indx2++){
	  diff_vec[0]=atomic_position[atom_indx2][0]-roted_xyz[atom_indx2][0];
	  diff_vec[1]=atomic_position[atom_indx2][1]-roted_xyz[atom_indx2][1];
	  diff_vec[2]=atomic_position[atom_indx2][2]-roted_xyz[atom_indx2][2];
    	
	  diff_vec[0]=fmod(diff_vec[0]+1000.0,1.0);
	  diff_vec[1]=fmod(diff_vec[1]+1000.0,1.0);
	  diff_vec[2]=fmod(diff_vec[2]+1000.0,1.0);
	  if(fabs(diff_vec[0]-1.0)<smallvalue){diff_vec[0]=0.0;}
	  if(fabs(diff_vec[1]-1.0)<smallvalue){diff_vec[1]=0.0;}
	  if(fabs(diff_vec[2]-1.0)<smallvalue){diff_vec[2]=0.0;}	
	  if(fabs(diff_vec[0])<smallvalue
	     &&fabs(diff_vec[1])<smallvalue
	     &&fabs(diff_vec[2])<smallvalue){
	    pure_point=1;
	  }else{
	    pure_point=0;
	    if(debug3==1){
	      printf("NOT TRANSLATED!!!!!\n");
	    }
	    break;
	  }
	}
	if(pure_point==0){break;}
      }/*comparing */
      if(pure_point==1){
	/*translation vector can reproduce all the atomic position, mark it */
	tmp_vec[0]=trans_vec[0];
	tmp_vec[1]=trans_vec[1];
	tmp_vec[2]=trans_vec[2];
	/* printf("Haha!!!!!!!!!!!!!!!!!!!!!! FOUND ONE\n"); */
    	/* printf("tmp_vec %12.8f%12.8f%12.8f\n",tmp_vec[0],tmp_vec[1],tmp_vec[2]); */
	foundone=1;
      }
      for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
	if(atom_type_indx==0){
	  start_atom_indx=0;/* The atoms of first species are stored from 0 index */
	}else{
	  start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
	}
	end_atom_indx=atom_species[atom_type_indx];
      
	for(atom_indx2=start_atom_indx;atom_indx2<end_atom_indx;atom_indx2++){
	  for(i=0;i<3;i++){
	    roted_xyz[atom_indx2][i]=roted_xyz[atom_indx2][i]-trans_vec[i];
	    roted_xyz[atom_indx2][i]=roted_xyz[atom_indx2][i]-floor(roted_xyz[atom_indx2][i]);
	    roted_xyz[atom_indx2][i]=fmod(roted_xyz[atom_indx2][i]+1000.5,1.0)-0.5;
	  }
	} 
      }
    }/*find a translation vector */
  }
  if(foundone==1){
    trans_vec[0]=tmp_vec[0];
    trans_vec[1]=tmp_vec[1];
    trans_vec[2]=tmp_vec[2];
  }
  
  for(k=0; k<atom_num; k++){
    free(roted_xyz[k]);
  }
  free(roted_xyz);
  free(markone);
  free(tmp_vec);
  free(diff_vec);
  
  return foundone;
}       

void Get_Symmetry_Operation(int ***symop, int *opnum, double **atomic_position, int *atom_species,
			    int atom_num, int atom_type, int *num_pnt_op, double **trans_op_vec){
  /* Find the possible symmetry operations according to the atomic arrangement 
     (lattic basis, atomic_position) from the pool of symmetry operations 
     (sym_op[op_num][3][3]) found from pure bravais lattice. sym_op matix is changed
     after Get_Symmetry_Operation. Totally it returns op_num operations while the first
     num_pure_pnt_op is the pure point symmetry operation. The left are those connected
     with translation vectors, which are stored in trans_op[op_num-num_pure_pnt_op][3].
  */     
  int i,j,k,pure_point;
  int **exop;
  int ***spg_op; /* temp array for store founded space group (non pure point group) operation */
  double **trans_vec;
  double *tmp_vec;
  char c;
  int op_num,sizeof_matrix;
  int num_pure_pnt_op,num_spg_op;
	  
	  
  num_pure_pnt_op=0;
  num_spg_op=0;
  pure_point = 0;
		
  op_num=*opnum;
  sizeof_matrix=op_num;
  		
  spg_op = (int***)malloc(sizeof(int**)*op_num);
  for (k=0; k<op_num; k++){
    spg_op[k] = (int**)malloc(sizeof(int*)*3);
    for (i=0; i<3; i++){
      spg_op[k][i] = (int*)malloc(sizeof(int)*3);
      for (j=0; j<3; j++) spg_op[k][i][j] = 0;
    }
  }

  trans_vec = (double**)malloc(sizeof(double*)*op_num);
  for (k=0; k<op_num; k++){
    trans_vec[k]=(double*)malloc(sizeof(double)*3);
    for (i=0; i<3; i++) trans_vec[k][i] = 0.0;
  }
		
  exop=(int**)malloc(sizeof(int*)*3);
  tmp_vec=(double*)malloc(sizeof(double)*3);
  for(i=0;i<3;i++){
    exop[i]=(int*)malloc(sizeof(int)*3);
    tmp_vec[i]=0.0;
  }
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      exop[i][j]=0;
    }
  }

  for(k=0;k<op_num;k++){ /* for each operation */
    	
    Matrix_Copy32(exop,3,symop,k);
    if(debug2 ==1 ){
      printf("operation %2d:\n",k);
      for(i=0;i<3;i++){
	printf("%4d%4d%4d\n",exop[i][0],exop[i][1],exop[i][2]);
      }
    }
    /* 
       check whether this operation (exop) is pure point group operation or not, if not, 
       find the connected translation vector (tmp_vec)
       INPUT  
       exop                                                  one symmetry operation
       atomic_position[atom_indx][atom_type_indx][3]         lattice basis (atomic arrangement)
       atom_num                                              total number of atoms in the lattice
       atom_type                                             number of atoms' species
    */
    pure_point = Chk_Pure_Point_Group_Op(exop,atomic_position,atom_species,atom_num,atom_type,tmp_vec);
    if(pure_point==1){/* if it is an allowed symmetry */
      if(debug2 ==1 ){
	printf("********************This is an allowed symmetry. Number %2d\n",k);
      }
      if(fabs(tmp_vec[0])<smallvalue
	 &&fabs(tmp_vec[1])<smallvalue
	 &&fabs(tmp_vec[2])<smallvalue){
	/* pure point group symmetry */
	Matrix_Copy23(exop,3,symop,num_pure_pnt_op);
	trans_op_vec[num_pure_pnt_op][0]=0.0;
	trans_op_vec[num_pure_pnt_op][1]=0.0;
	trans_op_vec[num_pure_pnt_op][2]=0.0;
	num_pure_pnt_op++;
      }else{
	/* space group symmetry */

	Matrix_Copy23(exop,3,spg_op,num_spg_op);
	trans_vec[num_spg_op][0]=tmp_vec[0];
	trans_vec[num_spg_op][1]=tmp_vec[1];
	trans_vec[num_spg_op][2]=tmp_vec[2];  
	if(debug2 ==1 ){
	  printf("********************But with translation vector%8.5f%8.5f%8.5f\n",tmp_vec[0],tmp_vec[1],tmp_vec[2]);
	}
	num_spg_op++;
      }
    }else{
      if(debug2 ==1 ){
	printf("!!!NOT allowed symmetry. Number %2d\n",k);
      }
    }
  }

  for(i=0;i<num_spg_op;i++){/* put space group operations and corresponding translation vector*/
    Matrix_Copy32(exop,3,spg_op,i);
    Matrix_Copy23(exop,3,symop,i+num_pure_pnt_op);
    trans_op_vec[i+num_pure_pnt_op][0]=trans_vec[i][0];
    trans_op_vec[i+num_pure_pnt_op][1]=trans_vec[i][1];
    trans_op_vec[i+num_pure_pnt_op][2]=trans_vec[i][2];
  }

  op_num=num_pure_pnt_op+num_spg_op;
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      exop[i][j]=0;
    }
  }
  for(i=op_num;i<*opnum;i++){
    Matrix_Copy23(exop,3,symop,i);
    trans_op_vec[i][0]=0.0;    	
    trans_op_vec[i][1]=0.0;
    trans_op_vec[i][2]=0.0;
  }
  *opnum=op_num;
  *num_pnt_op=num_pure_pnt_op;
  /* printf("the allowed symmetry number is %2d and there are %2d space group operation\n",*opnum, *num_pnt_op); */

  for (k=0; k<sizeof_matrix; k++){
    free(trans_vec[k]);
  }
  free(trans_vec);

  for (k=0; k<sizeof_matrix; k++){
    for (i=0; i<3; i++){
      free(spg_op[k][i]);
    }
    free(spg_op[k]);
  }
  free(spg_op);

  for(i=0;i<3;i++){
    free(exop[i]);
  }
  free(tmp_vec);
  free(exop);
}

int Chk_Primitive_Cell(int bravais_type, double *cell_parameters, double **lattice_vector,
		       double **atomic_position, int *atom_species,int atom_num, int atom_type,
		       double **platt, double **ptran_vec, double *pcell_parameters,int *npcell){    
  /* Check whether the inputed unit cell is a primitive cell or not
     INPUT 
     bravais_type            Bravais lattice type
     cell_parameters         a, b, c length and angles between vectors 
     lattice_vector          the original lattice vectors
     atomic_position         the atomic positions transformed from OpenMX
     atom_species            the indicator of atomic index of each species in atomic_position
     atom_num                number of atoms in atomic_position array
     atom_type               number of species in array atom_species

     OUTPUT
     platt                   the primitive lattice vectors founded
     ptran_vec               array to store the translation vectors for each atom
     pcell_parameters        the cell parameters for primitive cell
     npcell                  number of primitive cell in cell defined by lattice_vector
     pcell_brav              return the bravais type of primitive lattice
  */                        
  int atom_indx, atom_type_indx, start_atom_indx, end_atom_indx,atom_indx2;
  int i,j,k,ip,jp,kp;
  double **roted_xyz;
  double *markone, *tmp_vec, *diff_vec, *trans_vec;
  char c;
  int smallest_species, pure_point, ptrans_num, pcell_brav;
  double cellvol,pcellvol;
    
        
  roted_xyz=(double**)malloc(sizeof(double*)*atom_num);
  markone=(double*)malloc(sizeof(double)*3);
  tmp_vec=(double*)malloc(sizeof(double)*3);
  diff_vec=(double*)malloc(sizeof(double)*3);
  trans_vec=(double*)malloc(sizeof(double)*3);
  for(k=0; k<atom_num; k++){
    roted_xyz[k] = (double*)malloc(sizeof(double)*3);
    for(i=0; i<3; i++){
      roted_xyz[k][i] = 0.0;
      ptran_vec[k][i] = 0.0;
      tmp_vec[i]=3.0;
      trans_vec[i]=0.0;
    }
  }
  for(i=0;i<3;i++){
    ptran_vec[atom_num][i]=0.0;
    ptran_vec[atom_num+1][i]=0.0;
  }
  /*to be consistent wit VASP, finding the species having smallest number of atom*/    
  j=atom_species[0];
  smallest_species=0;
  for(i=1;i<atom_type;i++){
    if((atom_species[i]-atom_species[i-1])<j){
      j=atom_species[i]-atom_species[i-1];
      smallest_species=i;
    }	
  }
  if(debug3==1){
    printf("smallest number of species is %2d\n",smallest_species);
  }
  if(smallest_species==0){
    start_atom_indx=0;
  }else{
    start_atom_indx=atom_species[smallest_species-1];
  }
  if((atom_species[smallest_species]-start_atom_indx)==1){
    printf("Original cell was already a primitive cell.\n");
    *npcell=1;
    
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	platt[i][j]=lattice_vector[i][j];
      }
    }
    
    for(i=0;i<6;i++){
      pcell_parameters[i]=cell_parameters[i];
    }
    
    for(k=0; k<atom_num; k++){
      free(roted_xyz[k]);
    }
    free(roted_xyz);
    free(markone);
    free(tmp_vec);
    free(diff_vec);
    free(trans_vec);
    
    return bravais_type;
  }
 	
  for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
    if(atom_type_indx==0){
      start_atom_indx=0;/* The atoms of first species are stored from 0 index */
    }else{
      start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
    }
    end_atom_indx=atom_species[atom_type_indx];
      
    for(atom_indx=start_atom_indx;atom_indx<end_atom_indx;atom_indx++){
      /* 1. Precess the atomic coordinates, make them be [-0.5, 0.5) 
	 (Applying periodic boundary condition)
      */ 	
      if(debug3==1){
      	printf("atom %3d ",atom_indx);
      }
      for(i=0;i<3;i++){
	/* make it positive and less than +1.0 */
	if(debug3==1){
	  printf("%12.8f-->",atomic_position[atom_indx][i]);
	}
	atomic_position[atom_indx][i]=atomic_position[atom_indx][i]-floor(atomic_position[atom_indx][i]);
	atomic_position[atom_indx][i]=fmod(atomic_position[atom_indx][i]+1000.5,1.0)-0.5;
	if(debug3==1){
	  printf("%10.8f",atomic_position[atom_indx][i]);
	}
      }
      if(debug3==1){
      	printf("\n");
      }
    } 
    /* 2. Ordering the atomic coordinates of same species with index from start to end. 
       firstly x- then y-, z- last. */
    Ordering_Atomic_Position(atomic_position, start_atom_indx, end_atom_indx);  
  }
 
  /* mark the first atom */ 
  if(smallest_species==0){
    start_atom_indx=0;
  }else{
    start_atom_indx=atom_species[smallest_species-1];
  }   
  markone[0]=atomic_position[start_atom_indx][0];
  markone[1]=atomic_position[start_atom_indx][1];
  markone[2]=atomic_position[start_atom_indx][2];
  ptrans_num=3;
  for(i=0;i<3;i++){
    ptran_vec[i][i]=1.0;
  }
  if(debug3==1){
    printf("MARKED atom is %12.8f%12.8f%12.8f\n",markone[0],markone[1],markone[2]);
  }
  for(atom_indx=start_atom_indx+1;atom_indx<atom_species[smallest_species];atom_indx++){
    trans_vec[0]=atomic_position[atom_indx][0]-markone[0];
    trans_vec[1]=atomic_position[atom_indx][1]-markone[1];
    trans_vec[2]=atomic_position[atom_indx][2]-markone[2];
    	
    trans_vec[0]=fmod(trans_vec[0]+1000.0,1.0);
    trans_vec[1]=fmod(trans_vec[1]+1000.0,1.0);
    trans_vec[2]=fmod(trans_vec[2]+1000.0,1.0);
    if(debug3==1){	
      printf("atom %2d and ",atom_indx);
      printf("TRANS_VEC %12.8f%12.8f%12.8f\n",trans_vec[0],trans_vec[1],trans_vec[2]);
    }
    /* c=getchar(); */
    if(fabs(trans_vec[0]-1.0)<smallvalue){trans_vec[0]=0.0;}
    if(fabs(trans_vec[0]-1.0)<smallvalue){trans_vec[0]=0.0;}
    if(fabs(trans_vec[0]-1.0)<smallvalue){trans_vec[0]=0.0;}	
    /* translate the rotated coordinates with trans_vec and compare with original ones*/
    if(debug3==1){
      printf("applying TRANSLATION vectors:\n");
    }
    for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
      if(atom_type_indx==0){
	start_atom_indx=0;/* The atoms of first species are stored from 0 index */
      }else{
	start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
      }
      end_atom_indx=atom_species[atom_type_indx];
      
      for(atom_indx2=start_atom_indx;atom_indx2<end_atom_indx;atom_indx2++){
	/*      Applying periodic boundary condition */ 	       
	if(debug3==1){
	  printf("atom %3d ",atom_indx2);
	}
	for(i=0;i<3;i++){
	  /* make it positive and less than +1.0 */
	  if(debug3==1){
	    printf("%12.8f-->",atomic_position[atom_indx2][i]);
	  }
	  roted_xyz[atom_indx2][i]=atomic_position[atom_indx2][i]+trans_vec[i];
	  roted_xyz[atom_indx2][i]=roted_xyz[atom_indx2][i]-floor(roted_xyz[atom_indx2][i]);
	  roted_xyz[atom_indx2][i]=fmod(roted_xyz[atom_indx2][i]+1000.5,1.0)-0.5;
	  if(debug3==1){
	    printf("%10.8f",roted_xyz[atom_indx2][i]);
	  }
	}
	if(debug3==1){
	  printf("\n");
	}
      } 
      /*    Ordering the atomic coordinates of same species with index from start to end. 
	    firstly x- then y-, z- last. */
      Ordering_Atomic_Position(roted_xyz, start_atom_indx, end_atom_indx);  
    }/*applying translation */
    /* after translation, now comparing */
    pure_point=0;
    for(atom_type_indx=0;atom_type_indx<atom_type;atom_type_indx++){ 	
      if(atom_type_indx==0){
	start_atom_indx=0;/* The atoms of first species are stored from 0 index */
      }else{
	start_atom_indx=atom_species[atom_type_indx-1]; /*Other species atoms are stored from atom_species[atom_type_indx-1] */
      }
      end_atom_indx=atom_species[atom_type_indx];
      for(atom_indx2=start_atom_indx;atom_indx2<end_atom_indx;atom_indx2++){
	diff_vec[0]=atomic_position[atom_indx2][0]-roted_xyz[atom_indx2][0];
	diff_vec[1]=atomic_position[atom_indx2][1]-roted_xyz[atom_indx2][1];
	diff_vec[2]=atomic_position[atom_indx2][2]-roted_xyz[atom_indx2][2];
    	
	diff_vec[0]=fmod(diff_vec[0]+1000.0,1.0);
	diff_vec[1]=fmod(diff_vec[1]+1000.0,1.0);
	diff_vec[2]=fmod(diff_vec[2]+1000.0,1.0);
	if(fabs(diff_vec[0]-1.0)<smallvalue){diff_vec[0]=0.0;}
	if(fabs(diff_vec[1]-1.0)<smallvalue){diff_vec[1]=0.0;}
	if(fabs(diff_vec[2]-1.0)<smallvalue){diff_vec[2]=0.0;}	
	if(fabs(diff_vec[0])<smallvalue
	   &&fabs(diff_vec[1])<smallvalue
	   &&fabs(diff_vec[2])<smallvalue){
	  pure_point=1;
	}else{
	  pure_point=0;
	  if(debug3==1){
	    printf("NOT TRANSLATED!!!!!\n");
	  }
	  break;
	}
      }
      if(pure_point==0){break;}
    }/*comparing */
    if(pure_point==1){
      /*translation vector can reproduce all the atomic position, mark it */
      if(debug3==1){
	printf("OK find one! ptrans_num=%2d\n",ptrans_num);
      }
      ptran_vec[ptrans_num][0]=trans_vec[0];
      ptran_vec[ptrans_num][1]=trans_vec[1];
      ptran_vec[ptrans_num][2]=trans_vec[2];
      ptrans_num++;  
    }
  }
  /* c=getchar(); */
  if(ptrans_num==3){/*no translation vectors are founded */
    printf("Original cell was already a primitive cell.\n");
    *npcell=1;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	platt[i][j]=lattice_vector[i][j];
      }
    }
    for(i=0;i<6;i++){
      pcell_parameters[i]=cell_parameters[i];
    }
    for(k=0; k<atom_num; k++){
      free(roted_xyz[k]);
    }
    free(roted_xyz);
    free(markone);
    free(tmp_vec);
    free(diff_vec);
    free(trans_vec);
    return bravais_type;
  }
  if(debug3==1){
    printf("before ordering, ptrans_num=%2d\n",ptrans_num);
  }
  Ordering_Atomic_Position(ptran_vec, 0, ptrans_num);  
  /* y-z plane 
   *markone, *tmp_vec, *diff_vec will be used.*/
  for(i=1;i<ptrans_num;i++){
    if(fabs(ptran_vec[i][0]-ptran_vec[0][0])>smallvalue){
      ip=i-1;
      jp=i;
      if(debug3==1){
	printf("ip=%d,jp=%d\n",ip,jp);
      }
      break;
    }
  }     
  markone[0]=ptran_vec[jp][0];
  markone[1]=ptran_vec[jp][1];
  markone[2]=ptran_vec[jp][2];
  for(i=1;i<(ip+1);i++){
    if(fabs(ptran_vec[i][1]-ptran_vec[0][1])>smallvalue){
      kp=i;
      if(debug3==1){
	printf("kp=%d\n",kp);
      }
      break;
    }
  }
  tmp_vec[0]=ptran_vec[kp][0];
  tmp_vec[1]=ptran_vec[kp][1];
  tmp_vec[2]=ptran_vec[kp][2];
  diff_vec[0]=ptran_vec[0][0];
  diff_vec[1]=ptran_vec[0][1];
  diff_vec[2]=ptran_vec[0][2];
  platt[0][0]=markone[0]*lattice_vector[0][0]+markone[1]*lattice_vector[1][0]+markone[2]*lattice_vector[2][0];
  platt[0][1]=markone[0]*lattice_vector[0][1]+markone[1]*lattice_vector[1][1]+markone[2]*lattice_vector[2][1];
  platt[0][2]=markone[0]*lattice_vector[0][2]+markone[1]*lattice_vector[1][2]+markone[2]*lattice_vector[2][2];
      
  platt[1][0]=tmp_vec[0]*lattice_vector[0][0]+tmp_vec[1]*lattice_vector[1][0]+tmp_vec[2]*lattice_vector[2][0];
  platt[1][1]=tmp_vec[0]*lattice_vector[0][1]+tmp_vec[1]*lattice_vector[1][1]+tmp_vec[2]*lattice_vector[2][1];
  platt[1][2]=tmp_vec[0]*lattice_vector[0][2]+tmp_vec[1]*lattice_vector[1][2]+tmp_vec[2]*lattice_vector[2][2];
      
  platt[2][0]=diff_vec[0]*lattice_vector[0][0]+diff_vec[1]*lattice_vector[1][0]+diff_vec[2]*lattice_vector[2][0];
  platt[2][1]=diff_vec[0]*lattice_vector[0][1]+diff_vec[1]*lattice_vector[1][1]+diff_vec[2]*lattice_vector[2][1];
  platt[2][2]=diff_vec[0]*lattice_vector[0][2]+diff_vec[1]*lattice_vector[1][2]+diff_vec[2]*lattice_vector[2][2];
      
  pcell_brav=Finding_Bravais_Lattice_Type(platt,pcell_parameters);
  cellvol=Cal_Cell_Volume(lattice_vector);
  pcellvol=Cal_Cell_Volume(platt);
  if(debug3==1){
    printf("Original cell's volume=%10.5f. Primitive cell 's volume=%10.5f\n",cellvol,pcellvol);
  }
  cellvol=cellvol/pcellvol;
  if((cellvol-(int)cellvol)<smallvalue){
    *npcell=(int)cellvol;
  }else{
    *npcell=(int)cellvol+1;
  }    
  
  for(k=0; k<atom_num; k++){
    free(roted_xyz[k]);
  }
  free(roted_xyz);
  free(markone);
  free(tmp_vec);
  free(diff_vec);
  free(trans_vec);
    
  return pcell_brav;     
}                    



int Generate_MP_Special_Kpt(int knum_i, int knum_j, int knum_k, int ***sym_op, int op_num,
			    int ***pureGsym, int pureG_num, double *shift, 
                            double *KGrids1, double *KGrids2, double *KGrids3, int *T_k_op)
     /*
       kpt_num=Generate_MP_Special_Kpt(knum_i, knum_j, knum_k, ksym_op,ksym_num, 
       pureGsym, pureG_num, shift, tmpK1, tmpK2, tmpK3, tmpWeight);
       INPUT 
       knum_i, knum_j, knum_k     sampling grids knum_i*knum_j*knum_k                                         
       sym_op                     Symmetry operations of the given crystal
       op_num                     The first op_num operator in sym_op is pure point group operation
  
       OUTPUT                                                  
       return the total number of non-equivalent k points
       T_KGrids1, T_KGrids2, T_KGrids3  k point coordinates kx,ky,kz
       T_k_op       weight of the k point                      
       return the total non-equivlent k point number.
     */  
{
  int kpt_num, ktest; /*  total number of non-equivalent k points */

  double ktmp[48][3],kx,ky,kz;
  int i,j,k,r,p,s, itst,ksym;
  int whetherNonEquiv;
  double *tmpWeight;
  char c;
  kpt_num = 0;
  
  tmpWeight=(double*)malloc(sizeof(double)*(8*knum_i*knum_j*knum_k));
  for(i=0;i<(8*knum_i*knum_j*knum_k);i++){
    tmpWeight[i]=0.0;
  }
  if(debug4==1){
    printf("pureG number is %2d, and op_num=%2d\n",pureG_num,op_num);
  }
  for(p=0;p<knum_k;p++){
    for(r=0;r<knum_j;r++){
      for(s=0;s<knum_i;s++){
	/*generate one k point */
	ktmp[0][0]=((double)s+shift[0])/(double)knum_i;
	ktmp[0][1]=((double)r+shift[1])/(double)knum_j;
	ktmp[0][2]=((double)p+shift[2])/(double)knum_k;
        if(debug4==1){
	  printf("%10.5f%10.5f%10.5f\n",ktmp[0][0],ktmp[0][1],ktmp[0][2]); 
        }
	ktest=0;
	for(k=0;k<pureG_num;k++){
	  kx=ktmp[0][0]*(double)pureGsym[k][0][0]+ktmp[0][1]*(double)pureGsym[k][1][0]+ktmp[0][2]*(double)pureGsym[k][2][0];
	  ky=ktmp[0][0]*(double)pureGsym[k][0][1]+ktmp[0][1]*(double)pureGsym[k][1][1]+ktmp[0][2]*(double)pureGsym[k][2][1];
	  kz=ktmp[0][0]*(double)pureGsym[k][0][2]+ktmp[0][1]*(double)pureGsym[k][1][2]+ktmp[0][2]*(double)pureGsym[k][2][2];
	  kx=kx-floor(kx);
          kx=fmod(kx+1000.5-0.5*smallvalue,1.0)-0.5+0.5*smallvalue;
          
          ky=ky-floor(ky);
          ky=fmod(ky+1000.5-0.5*smallvalue,1.0)-0.5+0.5*smallvalue;
          
          kz=kz-floor(kz);
          kz=fmod(kz+1000.5-0.5*smallvalue,1.0)-0.5+0.5*smallvalue;
          
	  whetherNonEquiv=1;	
	  for(itst=0;itst<ktest;itst++){
	    if( fabs(kx-ktmp[itst][0])<smallvalue 
		&& fabs(ky-ktmp[itst][1])<smallvalue 
		&& fabs(kz-ktmp[itst][2])<smallvalue) {
	      whetherNonEquiv=0;
              if(debug4==1){
	        printf("k=%2d,existing with itst=%2d and ktest=%2d\n",k,itst,ktest);
              } 
	    }
	  }
	  if(whetherNonEquiv==1){
	    ktmp[ktest][0]=kx;
	    ktmp[ktest][1]=ky;
	    ktmp[ktest][2]=kz;
	    ktest++;
            if(debug4==1){
              printf("k=%2d,Find one! ktest=%5d\n",k,ktest);
            } 
	  }
	}/* End of pure G symmetry operation */
        /* printf("ktest=%5d",ktest); */
        /* c=getchar(); */
	for(itst=0;itst<ktest;itst++){
	  whetherNonEquiv=1;
	  for(ksym=0;ksym<op_num;ksym++){
	    /* Symmtry operating this k point*/
	    kx=sym_op[ksym][0][0]*ktmp[itst][0]+sym_op[ksym][1][0]*ktmp[itst][1]+sym_op[ksym][2][0]*ktmp[itst][2];
	    ky=sym_op[ksym][0][1]*ktmp[itst][0]+sym_op[ksym][1][1]*ktmp[itst][1]+sym_op[ksym][2][1]*ktmp[itst][2];
	    kz=sym_op[ksym][0][2]*ktmp[itst][0]+sym_op[ksym][1][2]*ktmp[itst][1]+sym_op[ksym][2][2]*ktmp[itst][2];
	    /* Applying periodic boundary condition*/   				
            kx=kx-floor(kx);
            kx=fmod(kx+1000.5-0.5*smallvalue,1.0)-0.5+0.5*smallvalue;
          
            ky=ky-floor(ky);
            ky=fmod(ky+1000.5-0.5*smallvalue,1.0)-0.5+0.5*smallvalue;
          
            kz=kz-floor(kz);
            kz=fmod(kz+1000.5-0.5*smallvalue,1.0)-0.5+0.5*smallvalue;
            for(i=0;i<kpt_num;i++){
	      if( fabs(kx-KGrids1[i])<smallvalue 
		  && fabs(ky-KGrids2[i])<smallvalue 
		  && fabs(kz-KGrids3[i])<smallvalue) {
		whetherNonEquiv=0;
		tmpWeight[i]+=(double)(pureG_num/ktest);
		break;
	      }
	    }
	    if(whetherNonEquiv==0){
	      break;
	    }
	  }
	  if(whetherNonEquiv==1){
	    if(kpt_num>=(8*knum_i*knum_j*knum_k)){
	      printf("!***************!\n");
	      printf("!Something Error!\n");
	      printf("!***************!\n");
	      return 0;
	    }
	    KGrids1[kpt_num]=ktmp[itst][0];
	    KGrids2[kpt_num]=ktmp[itst][1];
	    KGrids3[kpt_num]=ktmp[itst][2];
	    tmpWeight[kpt_num]=(double)(pureG_num/ktest);
	    kpt_num++;
            if(debug4==1){
	      printf("kpt_num=%2d",kpt_num); 
            }	    /* c=getchar(); */
	  }
	}
      }
    }
  }	
  /* printf("kpt_num=%2d\n",kpt_num); */
  
  for(j=0;j<kpt_num;j++){
    T_k_op[j]=(int)tmpWeight[j];
    /*printf("%20.14f%20.14f%20.14f%14d\n",KGrids1[j],KGrids2[j],KGrids3[j],T_k_op[j]);*/ 
  }	  
  free(tmpWeight);
  return kpt_num;     
}
