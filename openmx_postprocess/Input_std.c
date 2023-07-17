#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "openmx_common.h"
#include "Inputtools.h"
#include "mpi.h"
#include <omp.h>



#include "tran_prototypes.h"

#define MAXBUF 1024
#define Max_Num_WF_Projs 15


void SpeciesString2int(int p);
void kpath_changeunit(double tv[4][4],double tv0[4][4],int Band_Nkpath,
                      double ***Band_kpath);
void kpoint_changeunit(double tv[4][4],double tv0[4][4],int MO_Nkpoint,
                       double **MO_kpoint);
void Set_Cluster_UnitCell(double tv[4][4],int unitflag);

int OrbPol2int(char OrbPol[YOUSO10]);
char *ToCapital(char *s);
int divisible_cheker(int N);
static void Set_In_First_Cell();
static void Remake_RestartFile(int numprocs_new, int numprocs_old, int N1, int N2, int N3, int SpinFlag_old);

/* hmweng */
int Analyze_Wannier_Projectors(int p, char ctmp[YOUSO10], 
                               int **tmp_Wannier_Pro_SelMat,
                               double ***tmp_Wannier_Projector_Hybridize_Matrix);
void Get_Rotational_Matrix(double alpha, double beta, double gamma, int L, double tmpRotMat[7][7]);
int Calc_Factorial(int arg);
void Get_Euler_Rotation_Angle(
      double zx, double zy, double zz,
      double xx, double xy, double xz,
      double *alpha_r, double *beta_r, double *gamma_r);



void Input_std(char *file)
{
  FILE *fp,*fp_check;
  int i,j,k,itmp;
  int num_wannier_total_projectors;
  int l,mul; /* added by MJ */
  int po=0;  /* error count */
  double r_vec[40],r_vec2[20];
  int i_vec[60],i_vec2[60];
  char *s_vec[60],Species[YOUSO10];
  char OrbPol[YOUSO10];
  double ecutoff1dfft;
  double mx,my,mz,tmp;
  double tmpx,tmpy,tmpz;
  double S_coordinate[3];
  double length,x,y,z;
  int orbitalopt;
  char buf[MAXBUF];
  char file_check[YOUSO10];
  int numprocs,myid,myid_original;
  int output_hks,ret;
  int numprocs1;

  /* added by S.Ryee */
  int a,b,c,d; 
  int dum_Nmul,dum_l,dum_mul;
  double Uval,Jval;
  /******************/

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Comm_rank(MPI_COMM_WORLD1,&myid_original);

  /****************************************************
                       open a file
  ****************************************************/

  if (input_open(file)==0){
    MPI_Finalize(); 
    exit(0);
  }

  /****************************************************
                   read basis information
  ****************************************************/

  input_string("System.CurrrentDirectory",filepath,"./");
  input_string("System.Name",filename,"default");
  input_string("DATA.PATH",DFT_DATA_PATH,"../DFT_DATA19");
  input_int("level.of.stdout", &level_stdout,1);
  input_int("level.of.fileout",&level_fileout,1);
  input_logical("memory.usage.fileout",&memoryusage_fileout,0); /* default=off */

  /* if OpenMX was called by MPI_spawn. */

  if (MPI_spawn_flag==1){
    char fname[YOUSO10];

    sprintf(fname,"%s%s.std",filepath,filename);
    MPI_spawn_stream = freopen(fname, "w", stdout);
  }

  /* check range of level_stdout and level_fileout */

  if (level_stdout<-1 || 3<level_stdout){
    printf("Invalid value of level.of.stdout\n");
    po++;
  }
  
  if (level_fileout<0 || 3<level_fileout){
    printf("Invalid value of level.of.fileout\n");
    po++;
  }
  /****************************************************
               projector expansion of VNA
  ****************************************************/

  input_logical("scf.ProExpn.VNA",&ProExpn_VNA,1); /* default=on */
  input_int("scf.BufferL.VNA", &BufferL_ProVNA,6);
  input_int("scf.RadialF.VNA", &List_YOUSO[34],12);
  
  /****************************************************
                      cutoff energy 
  ****************************************************/

  /* for cutoff energy */
  
  input_double("scf.energycutoff",&Grid_Ecut,(double)150.0);
  input_logical("scf.MPI.tuned.grids",&MPI_tunedgrid_flag,0);

  /* for fixed Ngrids */

  i_vec2[0]=0;
  i_vec2[1]=0;
  i_vec2[2]=0;
  input_intv("scf.Ngrid",3,i_vec,i_vec2);
  Ngrid1 = i_vec[0];
  Ngrid2 = i_vec[1];
  Ngrid3 = i_vec[2];

  if (Ngrid1==0 && Ngrid2==0 && Ngrid3==0)
    Ngrid_fixed_flag = 0;
  else 
    Ngrid_fixed_flag = 1;

  if (Ngrid_fixed_flag==1){
    i = divisible_cheker(Ngrid1);
    j = divisible_cheker(Ngrid2);
    k = divisible_cheker(Ngrid3);
   
    if ( (i*j*k)==0 ) {

      printf("scf.Ngrid must be divisible by \n");

      printf("    ");
      for (i=0; i<NfundamentalNum; i++){
        printf("%3d ",fundamentalNum[i]);
      } 
      printf("\n");

      MPI_Finalize(); 
      exit(0);
    }
  }

  /****************************************************
               definition of atomic species
  ****************************************************/

  input_int("Species.Number",&SpeciesNum,0);
  real_SpeciesNum = SpeciesNum;

  if (SpeciesNum<=0){
    printf("Species.Number may be wrong.\n");
    po++;
  }
  List_YOUSO[18] = SpeciesNum;

  /* memory allocation */
  Allocate_Arrays(0);

  /*************************************************************
     for LDA+U
     Hub_U_switch should be called before Allocate_Arrays(1);
  *************************************************************/ 

  input_logical("scf.Hubbard.U",&Hub_U_switch, 0);     /* --- MJ */

  /*************************************************************
     Choose LDA+U form when Hub_U_switch = 1				
     For simplified(Dudarev) form, Hub_Type = 1 (default)
     For general form, Hub_Type = 2 
  *************************************************************/

  if (Hub_U_switch==1){			/* by S.Ryee */
    input_int("scf.DFTU.Type",&Hub_Type,1);	
  }

  /*************************************************************
     Generating Coulomb interaction matrix from Yukawa potential
     (only for Hub_Type==2)				
  *************************************************************/

  if (Hub_Type==2){			/* by S.Ryee */
    input_logical("scf.Yukawa",&Yukawa_on,0);	
  }

  /*************************************************************
     Double-counting scheme for general LDA+U  				
     For sFLL, dc_Type = 1 (default)
     For sAMF, dc_Type = 2
     For cFLL, dc_Type = 3
     For cAMF, dc_Type = 4
  *************************************************************/
  if (Hub_U_switch==1 && Hub_Type==2){	/* by S.Ryee */
    s_vec[0]="sFLL";	i_vec[0]=1;
    s_vec[1]="sAMF";	i_vec[1]=2;
    s_vec[2]="cFLL";    i_vec[2]=3;
    s_vec[3]="cAMF";	i_vec[3]=4;
    input_string2int("scf.dc.Type",&dc_Type,4,s_vec,i_vec);
  }

  /*************************************************************
     Controlling the ratio of two Slater integrals, F4 over F2,
     for general LDA+U scheme.				
     F4/F2 = 0.625 (default)
  *************************************************************/
  if (Hub_U_switch==1 && Hub_Type==2){	/* by S.Ryee */
    input_double("scf.Slater.Ratio",&slater_ratio,(double)0.625);
  }

  /* default Hub_U_occupation = 2; */

  s_vec[0]="DUAL";            i_vec[0]=2;
  s_vec[1]="ONSITE";          i_vec[1]=0;
  s_vec[2]="FULL" ;           i_vec[2]=1;

  input_string2int("scf.Hubbard.Occupation",&Hub_U_occupation, 3, s_vec,i_vec);

  /****************************************************
                   Orbital optimization
  ****************************************************/

  s_vec[0]="OFF";  s_vec[1]="Atoms";  s_vec[2]="Species";  s_vec[3]="Atoms2";  s_vec[4]="Species2";
  i_vec[0]=0; i_vec[1]=1; i_vec[2]=2; i_vec[3]=3; i_vec[4]=4; 
  input_string2int("orbitalOpt.Method",&orbitalopt,5,s_vec,i_vec);

  switch (orbitalopt) {
    case 0: { Cnt_switch=0; }                                              break;
    case 1: { Cnt_switch=1; RCnt_switch=1; SCnt_switch=0; }                break;
    case 2: { Cnt_switch=1; RCnt_switch=1; ACnt_switch=1; SCnt_switch=0; } break;
    case 3: { Cnt_switch=1; RCnt_switch=1; SCnt_switch=1; }                break;
    case 4: { Cnt_switch=1; RCnt_switch=1; ACnt_switch=1; SCnt_switch=1; } break;
  }

  /*************************************************************
                           read species
  *************************************************************/ 

  if (fp=input_find("<Definition.of.Atomic.Species")) {

    for (i=0; i<SpeciesNum; i++){
      fgets(buf,MAXBUF,fp);
      sscanf(buf,"%s %s %s %lf",SpeName[i],SpeBasis[i],SpeVPS[i],&Spe_AtomicMass[i]);
      SpeciesString2int(i);
    }

    ungetc('\n',fp);

    if (! input_last("Definition.of.Atomic.Species>")) {
      /* format error */

      po++;

      if (myid==Host_ID){
        printf("Format error for Definition.of.Atomic.Species\n");
      }
      MPI_Finalize();
      exit(0);
    }
  }

  if (2<=level_stdout){
    for (i=0; i<SpeciesNum; i++){
      printf("<Input_std>  %i Name  %s\n",i,SpeName[i]);
      printf("<Input_std>  %i Basis %s\n",i,SpeBasis[i]);
      printf("<Input_std>  %i VPS   %s\n",i,SpeVPS[i]);
    }
  }

  List_YOUSO[35] = 0;
  for (i=0; i<SpeciesNum; i++){
    if (List_YOUSO[35]<Spe_MaxL_Basis[i]) List_YOUSO[35] = Spe_MaxL_Basis[i];
  }
  List_YOUSO[35] = List_YOUSO[35] + BufferL_ProVNA;

  /****************************************************
       Molecular dynamics or geometry optimization
  ****************************************************/

  i=0;
  s_vec[i]="NOMD";                    i_vec[i]=0;  i++;
  s_vec[i]="NVE" ;                    i_vec[i]=1;  i++;
  s_vec[i]="NVT_VS";                  i_vec[i]=2;  i++; /* modified by mari */
  s_vec[i]="Opt";                     i_vec[i]=3;  i++;
  s_vec[i]="EF";                      i_vec[i]=4;  i++; 
  s_vec[i]="BFGS";                    i_vec[i]=5;  i++; 
  s_vec[i]="RF";                      i_vec[i]=6;  i++; /* RF method by hmweng */
  s_vec[i]="DIIS";                    i_vec[i]=7;  i++;
  s_vec[i]="Constraint_DIIS";         i_vec[i]=8;  i++; /* not used */
  s_vec[i]="NVT_NH";                  i_vec[i]=9;  i++; 
  s_vec[i]="Opt_LBFGS";               i_vec[i]=10; i++; 
  s_vec[i]="NVT_VS2";                 i_vec[i]=11; i++; /* modified by Ohwaki */
  s_vec[i]="EvsLC";                   i_vec[i]=12; i++; 
  s_vec[i]="NEB";                     i_vec[i]=13; i++; 
  s_vec[i]="NVT_VS4";                 i_vec[i]=14; i++; /* modified by Ohwaki */
  s_vec[i]="NVT_Langevin";            i_vec[i]=15; i++; /* modified by Ohwaki */
  s_vec[i]="DF";                      i_vec[i]=16; i++; /* delta-factor */
  s_vec[i]="OptC1";                   i_vec[i]=17; i++; /* cell opt with fixed fractional coordinates by SD */
  s_vec[i]="OptC2";                   i_vec[i]=18; i++; /* cell opt with fixed fractional coordinates and angles fixed by SD */
  s_vec[i]="OptC3";                   i_vec[i]=19; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|=|a3| by SD */
  s_vec[i]="OptC4";                   i_vec[i]=20; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|!=|a3| by SD */
  s_vec[i]="OptC5";                   i_vec[i]=21; i++; /* cell opt with no constraint for cell and coordinates by SD */
  s_vec[i]="RFC1";                    i_vec[i]=22; i++; /* cell opt with fixed fractional coordinates by RF */
  s_vec[i]="RFC2";                    i_vec[i]=23; i++; /* cell opt with fixed fractional coordinates and angles fixed by RF */
  s_vec[i]="RFC3";                    i_vec[i]=24; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|=|a3| by RF */
  s_vec[i]="RFC4";                    i_vec[i]=25; i++; /* cell opt with fixed fractional coordinates, angles fixed and |a1|=|a2|!=|a3| by RF */
  s_vec[i]="RFC5";                    i_vec[i]=26; i++; /* cell opt with no constraint for cell and coordinates by RF */

  /* added by MIZUHO for NPT-MD */
  s_vec[i]="NPT_VS_PR";               i_vec[i]=27; i++; /* NPT MD implemented by Velocity-Scaling method and Parrinelo-Rahman method */
  s_vec[i]="NPT_VS_WV";               i_vec[i]=28; i++; /* NPT MD implemented by Velocity-Scaling method and Wentzcovitch method */
  s_vec[i]="NPT_NH_PR";               i_vec[i]=29; i++; /* NPT MD implemented by Nose-Hoover method and Parrinelo-Rahman method */
  s_vec[i]="NPT_NH_WV";               i_vec[i]=30; i++; /* NPT MD implemented by Nose-Hoover method and Wentzcovitch method */

  /* variable cell optimization */
  s_vec[i]="RFC6";                    i_vec[i]=31; i++; /* cell opt with fixed a3 vector by RF */
  s_vec[i]="RFC7";                    i_vec[i]=32; i++; /* cell opt with fixed a2 and a3 vector by RF */
  s_vec[i]="OptC6";                   i_vec[i]=33; i++; /* cell opt with fixed a3 vector by SD */
  s_vec[i]="OptC7";                   i_vec[i]=34; i++; /* cell opt with fixed a2 and a3 vector by SD */

  j = input_string2int("MD.Type",&MD_switch, i, s_vec,i_vec);

  if (j==-1){
    MPI_Finalize();
    exit(0);
  }

  /* for NEB */

  if (MD_switch==13){
    neb_type_switch = 1;
  }

  /* cell optimization by SD */

  if      (MD_switch==17){
    MD_switch = 17;
    cellopt_swtich = 1; /* OptC1: no constraint for cell vectors */
  }

  else if (MD_switch==18){
    MD_switch = 17;
    cellopt_swtich = 2; /* OptC2: angles fixed for cell vectors */
  }

  else if (MD_switch==19){

    MD_switch = 17;
    cellopt_swtich = 3; /* OptC3: angles fixed and |a1|=|a2|=|a3| for cell vectors */

    /*
    double a1,a2,a3;

    a1 = tv[1][1]*tv[1][1]+tv[1][2]*tv[1][2]+tv[1][3]*tv[1][3];
    a2 = tv[2][1]*tv[2][1]+tv[2][2]*tv[2][2]+tv[2][3]*tv[2][3];
    a3 = tv[3][1]*tv[3][1]+tv[3][2]*tv[3][2]+tv[3][3]*tv[3][3];

    if ( 0.000000001<fabs(a1-a2) || 0.000000001<fabs(a1-a3) ){

      if (myid==Host_ID){  
        printf("\nThe condition |a1|=|a2|=|a3| should be satisfied for the use of OptC3\n\n");
      }

      MPI_Finalize();
      exit(0);  
    }
    */

  }

  else if (MD_switch==20){

    MD_switch = 17;
    cellopt_swtich = 4; /* OptC4: angles fixed and |a1|=|a2|!=|a3| for cell vectors */

    /*
    double a1,a2,a3;

    a1 = tv[1][1]*tv[1][1]+tv[1][2]*tv[1][2]+tv[1][3]*tv[1][3];
    a2 = tv[2][1]*tv[2][1]+tv[2][2]*tv[2][2]+tv[2][3]*tv[2][3];
    a3 = tv[3][1]*tv[3][1]+tv[3][2]*tv[3][2]+tv[3][3]*tv[3][3];

    if ( 0.000000001<fabs(a1-a2) ){

      if (myid==Host_ID){  
        printf("\nThe condition |a1|=|a2| should be satisfied for the use of OptC4\n\n");
      }

      MPI_Finalize();
      exit(0);  
    }
    */
  }

  else if (MD_switch==21){
    MD_switch = 17;
    cellopt_swtich = 5; /* OptC5: no constraint for cell and coordinates */
  }

  else if (MD_switch==33){
    MD_switch = 17;
    cellopt_swtich = 6; /* OptC6: cell opt with fixed a3 vector by SD */
  }

  else if (MD_switch==34){
    MD_switch = 17;
    cellopt_swtich = 7; /* OptC7: cell opt with fixed a2 and a3 vector by SD */
  }

  /* cell optimization by RF */

  if      (MD_switch==22){
    MD_switch = 18;
    cellopt_swtich = 1; /* RFC1: no constraint for cell vectors */

    if (myid==Host_ID){
      printf("The optimizer is not supported.\n");
    }

    MPI_Finalize();
    exit(0);
  }

  else if (MD_switch==23){
    MD_switch = 18;
    cellopt_swtich = 2; /* RFC2: angles fixed for cell vectors */

    if (myid==Host_ID){
      printf("The optimizer is not supported.\n");
    }

    MPI_Finalize();
    exit(0);
  }

  else if (MD_switch==24){

    MD_switch = 18;
    cellopt_swtich = 3; /* RFC3: angles fixed and |a1|=|a2|=|a3| for cell vectors */

    /*
    double a1,a2,a3;
    a1 = tv[1][1]*tv[1][1]+tv[1][2]*tv[1][2]+tv[1][3]*tv[1][3];
    a2 = tv[2][1]*tv[2][1]+tv[2][2]*tv[2][2]+tv[2][3]*tv[2][3];
    a3 = tv[3][1]*tv[3][1]+tv[3][2]*tv[3][2]+tv[3][3]*tv[3][3];

    if ( 0.000000001<fabs(a1-a2) || 0.000000001<fabs(a1-a3) ){

      if (myid==Host_ID){  
        printf("\nThe condition |a1|=|a2|=|a3| should be satisfied for the use of OptC3\n\n");
      }

      MPI_Finalize();
      exit(0);  
    }
    */

    if (myid==Host_ID){
      printf("The optimizer is not supported.\n");
    }

    MPI_Finalize();
    exit(0);
  }

  else if (MD_switch==25){

    MD_switch = 18;
    cellopt_swtich = 4; /* RFC4: angles fixed and |a1|=|a2|!=|a3| for cell vectors */

    /*
    double a1,a2,a3;

    a1 = tv[1][1]*tv[1][1]+tv[1][2]*tv[1][2]+tv[1][3]*tv[1][3];
    a2 = tv[2][1]*tv[2][1]+tv[2][2]*tv[2][2]+tv[2][3]*tv[2][3];
    a3 = tv[3][1]*tv[3][1]+tv[3][2]*tv[3][2]+tv[3][3]*tv[3][3];

    if ( 0.000000001<fabs(a1-a2) ){

      if (myid==Host_ID){  
        printf("\nThe condition |a1|=|a2| should be satisfied for the use of OptC4\n\n");
      }

      MPI_Finalize();
      exit(0);  
    }
    */

    if (myid==Host_ID){
      printf("The optimizer is not supported.\n");
    }

    MPI_Finalize();
    exit(0);
  }

  else if (MD_switch==26){
    MD_switch = 18;
    cellopt_swtich = 5; /* RFC5: no constraint for cell and coordinates */
  }

  else if (MD_switch==31){
    MD_switch = 18;
    cellopt_swtich = 6; /* RFC6: cell opt with fixed a3 vector by RF */
  }

  else if (MD_switch==32){
    MD_switch = 18;
    cellopt_swtich = 7; /* RFC6: cell opt with fixed a2 and a3 vector by RF */
  }

  /* MD.maxIter */

  input_int("MD.maxIter",&MD_IterNumber,1);
  if (MD_IterNumber<1){
    printf("MD_IterNumber=%i should be over 0.\n",MD_IterNumber);
    po++;
  }

  if (MD_switch==16 && MD_IterNumber!=7){ /* delta-factor */
    printf("MD_IterNumber=%i should be 7 for the delta-factor calculation.\n",MD_IterNumber);
    po++;
  }

  input_int("MD.Current.Iter",&MD_Current_Iter,0);

  input_double("MD.TimeStep",&MD_TimeStep,(double)0.5);
  if (MD_TimeStep<0.0){
    printf("MD.TimeStep=%lf should be over 0.\n",MD_TimeStep);
    po++;
  }

  input_double("MD.Opt.criterion",&MD_Opt_criterion,(double)0.0003);
  input_int("MD.Opt.DIIS.History",&M_GDIIS_HISTORY,3);
  input_int("MD.Opt.StartDIIS",&OptStartDIIS,5);
  input_int("MD.Opt.EveryDIIS",&OptEveryDIIS,200);

  input_double("MD.EvsLC.Step",&MD_EvsLattice_Step,(double)0.4);

  i_vec[0]=1; i_vec[1]=1, i_vec[2]=1;
  input_intv("MD.EvsLC.flag",3,MD_EvsLattice_flag,i_vec);

  input_logical("MD.Out.ABC",&MD_OutABC,0); /* default=off */

  /*
  input_double("MD.Initial.MaxStep",&SD_scaling_user,(double)0.02); 
  */

  i=0;
  s_vec[i]="Schlegel" ;                  i_vec[i]=1;  i++;
  s_vec[i]="iden";                       i_vec[i]=0;  i++;
  s_vec[i]="FF" ;                        i_vec[i]=2;  i++;
  j = input_string2int("MD.Opt.Init.Hessian",&Initial_Hessian_flag, i, s_vec,i_vec);

  /* Ang -> a.u. */
  /*
  SD_scaling_user /= BohrR;
  */

  /*
  input_double("MD.Opt.DIIS.Mixing",&Gdiis_Mixing,(double)0.1);
  */

  if (19<M_GDIIS_HISTORY){
    printf("MD.Opt.DIIS.History should be lower than 19.\n");
    MPI_Finalize();
    exit(0);
  }

  if (MD_switch==2 || MD_switch==9 || MD_switch==11 || MD_switch==14 || MD_switch==15
      || MD_switch==27 || MD_switch==28 || MD_switch==29 || MD_switch==30){

    if (fp=input_find("<MD.TempControl")) {

      fscanf(fp,"%i",&TempNum);         

      /* added by mari */
      /* added by MIZUHO for NPT-MD */

      if (MD_switch==2 || MD_switch==11 || MD_switch==14 || MD_switch==27 || MD_switch==28 ) {

	/* NVT_VS or NVT_VS2 or NVT_VS4 or NPT_VS_PR or NPT_VS_WV */

	NumScale[0] = 0;

	for (i=1; i<=TempNum; i++){  
	  fscanf(fp,"%d %d %lf %lf",&NumScale[i],&IntScale[i],
                                    &TempScale[i],&RatScale[i]);
	  TempPara[i][1] = NumScale[i];
	  TempPara[i][2] = TempScale[i];
	}

        TempPara[0][1] = 0;
	TempPara[0][2] = TempPara[1][2];
      }

      /* added by mari */
      /* added by MIZUHO for NPT-MD */
      else if (MD_switch==9 || MD_switch==15 || MD_switch==29 || MD_switch==30) {

	/* NVT_NH or NVT_Langevin or NPT_NH_PR or NPT_NH_WV  */

	for (i=1; i<=TempNum; i++){  
	  fscanf(fp,"%lf %lf",&TempPara[i][1],&TempPara[i][2]);
	}  

        TempPara[0][1] = 0;
	TempPara[0][2] = TempPara[1][2];
      }

      if ( ! input_last("MD.TempControl>") ) {
	/* format error */
	printf("Format error for MD.TempControl\n");
	po++;
      }

    }
  }

  if (fp=input_find("<MD.CellPressureControl")) {
    fscanf(fp,"%i",&PreNum);  
    /* modified by MIZUHO for NPT-MD */
    for (i=1; i<=PreNum; i++){  
      fscanf(fp,"%lf %lf",&PrePara[i][1],&PrePara[i][2]);

      /* 1 [GPa] = 0.3398827*0.0001 [Hartree/Bohr^3] */

      PrePara[i][2] = PrePara[i][2]*0.3398827*0.0001;  /* change from GPa to a.u. */
    }  
    /* added by MIZUHO for NPT-MD */
    PrePara[0][1] = 0;
    PrePara[0][2] = PrePara[1][2];

    if ( ! input_last("MD.CellPressureControl>") ) {
      /* format error */
      printf("Format error for MD.CellPressureControl\n");
      po++;
    }
  }

  input_double("NH.Mass.HeatBath",&TempQ,(double)20.0);

  /* added by MIZUHO for NPT-MD */
  /* default value of PresW will be given later */
  input_double("NPT.Mass.Barostat",&PresW,(double)0.0);
  PresW = PresW*(unified_atomic_mass_unit/electron_mass);   
  input_double("MD.TempTolerance", &TempTol,(double)100.0);
  i=0;
  s_vec[i]="none"; i_vec[i]=i;  i++;  // 0
  s_vec[i]="cubic"; i_vec[i]=i;  i++; // 1
  s_vec[i]="ortho"; i_vec[i]=i;  i++; // 2
  input_string2int("NPT.LatticeRestriction",&LatticeRestriction, i, s_vec,i_vec);
  input_double("Langevin.Friction.Factor",&FricFac,(double)0.001);

  if (fp=input_find("<NPT.WV.F0")) {

    for (i=1; i<=3; i++){
      fscanf(fp,"%lf %lf %lf",&NPT_WV_F0[i][1],&NPT_WV_F0[i][2],&NPT_WV_F0[i][3]);
    }
    if ( ! input_last("NPT.WV.F0>") ) {
      /* format error */
      printf("Format error for NPT_WV_F0\n");
      po++;
    }
  }

  /* LNO_flag */

  input_logical("LNO.flag",&LNO_flag,0);

  /****************************************************
             solver of the eigenvalue problem
  ****************************************************/

  s_vec[0]="Recursion";     s_vec[1]="Cluster"; s_vec[2]="Band";
  s_vec[3]="NEGF";          s_vec[4]="DC";      s_vec[5]="GDC";
  s_vec[6]="Cluster-DIIS";  s_vec[7]="Krylov";  s_vec[8]="Cluster2";  
  s_vec[9]="EGAC";          s_vec[10]="DC-LNO"; s_vec[11]="Cluster-LNO";
  
  i_vec[0]=1;  i_vec[1]=2;   i_vec[2 ]=3;
  i_vec[3]=4;  i_vec[4]=5;   i_vec[5 ]=6;
  i_vec[6]=7;  i_vec[7]=8;   i_vec[8 ]=9;
  i_vec[9]=10; i_vec[10]=11; i_vec[11]=12;

  input_string2int("scf.EigenvalueSolver", &Solver, 12, s_vec,i_vec);

  if (Solver==11 || Solver==12){
    LNO_flag = 1;
  }

  if (Solver==1){
    if (myid==Host_ID){
      printf("The recursion method is not supported anymore.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (Solver==6){
    if (myid==Host_ID){
      printf("The GDC method is not supported anymore.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /* default=dstevx */
  s_vec[0]="dstevx"; s_vec[1]="dstegr"; s_vec[2]="dstedc"; s_vec[3]="dsteqr"; 
  i_vec[0]=2;        i_vec[1]=0;        i_vec[2]=1;        i_vec[3]=3;        
  input_string2int("scf.lapack.dste", &dste_flag, 4, s_vec,i_vec);

  s_vec[0]="elpa1"; s_vec[1]="lapack"; s_vec[2]="elpa2";
  i_vec[0]=1;       i_vec[1]=0;        i_vec[2]=2; 
  input_string2int("scf.eigen.lib", &scf_eigen_lib_flag, 3, s_vec,i_vec);

  if (Solver==1){
    if (myid==Host_ID){
      printf("Recursion method is not supported in this version.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  input_logical("scf.dclno.threading",&scf_dclno_threading,0);

  /****************************************************
      for generation of Monkhorst-Pack k-points
  ****************************************************/

  input_double("scf.MP.criterion",&Criterion_MP_Special_Kpt,(double)1.0e-5);

  s_vec[0]="REGULAR"; s_vec[1]="MP";
  i_vec[0]=1        ; i_vec[1]=2   ; 
  input_string2int("scf.Generation.Kpoint", &way_of_kpoint, 2, s_vec,i_vec);

  if (Solver==4 && way_of_kpoint==2){
    if (myid==Host_ID){
      printf("The Monkhorst-Pack is not supported for NEGF.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /****************************************************
   flags for saving and reading Fourier transformed
   quantities generated in FT_*.c
  ****************************************************/

  input_logical("FT.files.save",&FT_files_save,0);
  input_logical("FT.files.read",&FT_files_read,0);

  /****************************************************
                SCF or electronic system
  ****************************************************/

  s_vec[0]="LDA"; s_vec[1]="LSDA-CA"; s_vec[2]="LSDA-PW"; s_vec[3]="GGA-PBE";s_vec[4]="EXX-TEST";
  i_vec[0]=1; i_vec[1]=2; i_vec[2]= 3; i_vec[3]= 4; i_vec[4]=5;
  input_string2int("scf.XcType", &XC_switch, 5, s_vec,i_vec);

  s_vec[0]="Off"; s_vec[1]="On"; s_vec[2]="NC";
  i_vec[0]=0    ; i_vec[1]=1   ; i_vec[2]=3;
  input_string2int("scf.SpinPolarization", &SpinP_switch, 3, s_vec,i_vec);
  if      (SpinP_switch==0) List_YOUSO[23] = 1;  
  else if (SpinP_switch==1) List_YOUSO[23] = 2;
  else if (SpinP_switch==3) List_YOUSO[23] = 4;

  if (XC_switch==3 && Solver==8){
    if (myid==Host_ID){
      printf("Krylov subspace method is not supported for non-collinear calculations.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (XC_switch==1 && 1<=SpinP_switch){
    if (myid==Host_ID){
      printf("SpinP_switch should be OFF for this exchange functional.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /* scf.Constraint.NC.Spin */

  s_vec[0]="off"; s_vec[1]="on"; s_vec[2]="on2";
  i_vec[0]=0    ; i_vec[1]=1   ; i_vec[2]=2;
  input_string2int("scf.Constraint.NC.Spin", &Constraint_NCS_switch, 3, s_vec,i_vec);

  if (SpinP_switch!=3 && 1<=Constraint_NCS_switch){
    if (myid==Host_ID){
      printf("The constraint scheme is not supported for a collinear DFT calculation.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (1<=Constraint_NCS_switch && Hub_U_occupation!=2){
    if (myid==Host_ID){
      printf("The constraint scheme is supported in case of scf.Hubbard.Occupation=dual.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  input_double("scf.Constraint.NC.Spin.V",&Constraint_NCS_V,(double)0.0);  /* in eV */
  /* eV to Hartree */
  Constraint_NCS_V = Constraint_NCS_V/eV2Hartree;

  /* scf.SpinOrbit.Coupling */

  input_logical("scf.SpinOrbit.Coupling",&SO_switch,0);

  if (SpinP_switch!=3 && SO_switch==1){
    if (myid==Host_ID){
      printf("Spin-orbit coupling is not supported for collinear DFT calculations.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (SpinP_switch==0 && SO_switch==1){
    if (myid==Host_ID){
      printf("scf.SpinOrbit.Coupling must be OFF when scf.SpinPolarization=OFF\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /* scf.SO.factor */

  SO_factor_flag = 0; 

  if (SpinP_switch==3 && SO_switch==1){

    if (fp=input_find("<scf.SO.factor")) {    

      /* switch on SO_factor_flag */

      SO_factor_flag = 1; 

      /* initialize the SO_factor */ 
      for (i=0; i<SpeciesNum; i++){
	for (l=0; l<=3; l++){
          SO_factor[i][l] = 1.0 ;
	}
      }
	
      /* read SO_factor from the '.dat' file  */ 
      for (i=0; i<SpeciesNum; i++){
	fscanf(fp,"%s",Species);
        j = Species2int(Species);

	for (l=0; l<=3; l++){
          fscanf(fp,"%s %lf", buf, &SO_factor[j][l]) ;
	}
      }

      if (! input_last("scf.SO.factor>") ) {
	/* format error */
	printf("Format error for scf.SO.factor\n");
	po++;
      }

    }   /*  if (fp=input_find("<scf.SO.factor")) */

  } /* if (SpinP_switch==3 && SO_switch==1) */

  /* scf.partialCoreCorrection */

  input_logical("scf.partialCoreCorrection",&PCC_switch,1);

  if (PCC_switch==0){
    if (myid==Host_ID){
      printf("scf.partialCoreCorrection should be always switched on.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }  

  /* scf.pcc.opencore */

  if (fp=input_find("<scf.pcc.opencore")) {    

    for (i=0; i<SpeciesNum; i++){
      fscanf(fp,"%s",Species);
      j = Species2int(Species);
      fscanf(fp,"%d", &Spe_OpenCore_flag[j]) ;

      if (Spe_OpenCore_flag[j]==0 || Spe_OpenCore_flag[j]==1 || Spe_OpenCore_flag[j]==-1){
      }
      else{
        /* format error */
        printf("Valid input for scf.pcc.opencore is 0, -1, or 1.\n");
        po++;
      }
    }
 
    if (! input_last("scf.pcc.opencore>") ) {
      /* format error */
      printf("Format error for scf.pcc.opencore\n");
      po++;
    }
   
  } /* if (fp=input_find("<scf.pcc.opencore")) */

  /* scf.NC.Zeeman.Spin */

  input_logical("scf.NC.Zeeman.Spin",&Zeeman_NCS_switch,0);

  if (SpinP_switch!=3 && Zeeman_NCS_switch==1){
    if (myid==Host_ID){
      printf("The Zeeman term is not supported for a collinear DFT calculation.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (Zeeman_NCS_switch==1 && Hub_U_occupation!=2){
    if (myid==Host_ID){
      printf("The Zeeman term for spin is supported in case of scf.Hubbard.Occupation=dual.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (1<=Constraint_NCS_switch && Zeeman_NCS_switch==1){
    if (myid==Host_ID){
      printf("For spin magnetic moment, the constraint scheme and the Zeeman term\n");
      printf("are mutually exclusive.  Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  input_double("scf.NC.Mag.Field.Spin",&Mag_Field_Spin,(double)0.0);

  /**************************************
     Tesla to a.u.
     1 Tesla = 1/(2.35051742*10^5) a.u. 
  ***************************************/

  Mag_Field_Spin = Mag_Field_Spin/(2.35051742*100000.0);

  /* scf.NC.Zeeman.Orbital */

  input_logical("scf.NC.Zeeman.Orbital",&Zeeman_NCO_switch,0);

  if (SpinP_switch!=3 && Zeeman_NCO_switch==1){
    if (myid==Host_ID){
      printf("The Zeeman term is not supported for a collinear DFT calculation.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (Zeeman_NCO_switch==1 && Hub_U_occupation!=2){
    if (myid==Host_ID){
      printf("The Zeeman term for orbital is supported in case of scf.Hubbard.Occupation=dual.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (Zeeman_NCO_switch==1 && SO_switch==0){
    if (myid==Host_ID){
      printf("The Zeeman term for orbital moment is not supported without the SO term.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  input_double("scf.NC.Mag.Field.Orbital",&Mag_Field_Orbital,(double)0.0);

  /**************************************
     Tesla to a.u.
     1 Tesla = 1/(2.35051742*10^5) a.u. 
  ***************************************/

  Mag_Field_Orbital = Mag_Field_Orbital/(2.35051742*100000.0);

  if      (SpinP_switch==0)                 List_YOUSO[5] = 1;
  else if (SpinP_switch==1)                 List_YOUSO[5] = 2;
  else if (SpinP_switch==3 && SO_switch==0) List_YOUSO[5] = 3;
  else if (SpinP_switch==3 && SO_switch==1) List_YOUSO[5] = 3;

  i_vec2[0]=4;
  i_vec2[1]=4;
  i_vec2[2]=4;
  input_intv("scf.Kgrid",3,i_vec,i_vec2);
  Kspace_grid1 = i_vec[0];
  Kspace_grid2 = i_vec[1];
  Kspace_grid3 = i_vec[2];

  if (Kspace_grid1<=0){
    printf("Kspace_grid1 should be over 1\n");
    MPI_Finalize();
    exit(0);
  } 
  if (Kspace_grid2<=0){
    printf("Kspace_grid2 should be over 1\n");
    MPI_Finalize();
    exit(0);
  } 
  if (Kspace_grid3<=0){
    printf("Kspace_grid3 should be over 1\n");
    MPI_Finalize();
    exit(0);
  } 

  if (Solver!=3 && Solver!=4){
    List_YOUSO[27] = 1;
    List_YOUSO[28] = 1;
    List_YOUSO[29] = 1;
  }
  else{
    List_YOUSO[27] = Kspace_grid1;
    List_YOUSO[28] = Kspace_grid2;
    List_YOUSO[29] = Kspace_grid3;
  }

  /* set PeriodicGamma_flag in 1 in the band calc. with only the gamma point */
  PeriodicGamma_flag = 0;
  if (Solver==3 && Kspace_grid1==1 && Kspace_grid2==1 && Kspace_grid3==1){

    PeriodicGamma_flag = 1;
    Solver = 2;
   
    if (myid==Host_ID){
    printf("When only the gamma point is considered, the eigenvalue solver is changed to 'Cluster' with the periodic boundary condition.\n");fflush(stdout);
    }
  }

  input_double("scf.ElectronicTemperature",&E_Temp,(double)300.0);
  E_Temp = E_Temp/eV2Hartree;
  Original_E_Temp = E_Temp;

  s_vec[0]="Simple";     i_vec[0]=0;
  s_vec[1]="RMM-DIIS";   i_vec[1]=1;  
  s_vec[2]="GR-Pulay";   i_vec[2]=2;
  s_vec[3]="Kerker";     i_vec[3]=3; 
  s_vec[4]="RMM-DIISK";  i_vec[4]=4;
  s_vec[5]="RMM-DIISH";  i_vec[5]=5;
  s_vec[6]="RMM-ADIIS";  i_vec[6]=6;
  s_vec[7]="RMM-DIISV";  i_vec[7]=7;

  input_string2int("scf.Mixing.Type",&Mixing_switch,8,s_vec,i_vec);

  if ( (Mixing_switch==5 || Mixing_switch==7) && Cnt_switch==1 ){
    if (myid==Host_ID){ 
      printf("RMM-DIISH and RMM-DIISV are not compatible with orbital optimization.\n\n");
    }
    MPI_Finalize(); 
    exit(0);
  }

  input_double("scf.Init.Mixing.Weight",&Mixing_weight,(double)0.3);
  input_double("scf.Min.Mixing.Weight",&Min_Mixing_weight,(double)0.001);
  input_double("scf.Max.Mixing.Weight",&Max_Mixing_weight,(double)0.4);
  /* if Kerker_factor is not set here, later Kerker factor is automatically determined. */
  input_double("scf.Kerker.factor",&Kerker_factor,(double)-1.0);
  input_int("scf.Mixing.History",&Num_Mixing_pDM,5);
  input_int("scf.Mixing.StartPulay",&Pulay_SCF,6); 
  Pulay_SCF_original = Pulay_SCF;
  input_int("scf.Mixing.EveryPulay",&EveryPulay_SCF,1);
  input_int("scf.ExtCharge.History",&Extrapolated_Charge_History,3);
  
  if (Pulay_SCF<4 && Mixing_switch==1){
    if (myid==Host_ID){
      printf("For RMM-DIIS, scf.Mixing.StartPulay should be set to larger than 3.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /* increase electric temperature in case of SCF oscillation, default=off */
  input_logical("scf.Mixing.Control.Temp", &SCF_Control_Temp, 0); 

  if (Mixing_switch==0){
    List_YOUSO[16] = 3;
    List_YOUSO[38] = 1;
  }
  else if (Mixing_switch==1 || Mixing_switch==2 || Mixing_switch==6) {
    List_YOUSO[16] = Num_Mixing_pDM + 2;
    List_YOUSO[38] = 1;
    List_YOUSO[39] = Num_Mixing_pDM + 3;
  }
  else if (Mixing_switch==3) {
    List_YOUSO[16] = 3;
    List_YOUSO[38] = 3;
  }
  else if (Mixing_switch==4) {
    List_YOUSO[16] = 3;
    List_YOUSO[38] = Num_Mixing_pDM + 2;
  }
  else if (Mixing_switch==5){
    List_YOUSO[16] = 3;
    List_YOUSO[38] = 1;
    List_YOUSO[39] = Num_Mixing_pDM + 3;
  }
  else if (Mixing_switch==7) {
    List_YOUSO[16] = 3;
    List_YOUSO[38] = Num_Mixing_pDM + 2;
  }

  input_double("scf.criterion",&SCF_Criterion,(double)1.0e-6);
  if (SCF_Criterion<0.0){
    printf("SCF_Criterion=%10.9f should be larger than 0.\n",SCF_Criterion);
    po++;
  }

  input_double("scf.system.charge",&system_charge,(double)0.0);

  /* scf.fixed.grid */

  r_vec[0]=1.0e+9; r_vec[1]=1.0e+9; r_vec[2]=1.0e+9;
  input_doublev("scf.fixed.grid",3,scf_fixed_origin,r_vec);

  TRAN_Input_std(mpi_comm_level1, Solver, SpinP_switch, filepath, kB, 
                 eV2Hartree, E_Temp, &output_hks);

  /* YTL-start */
  /*************************************************************************
    Calculate:
    (1) nabra operator
    (2) momentum matrix elements from occupied state to unoccupied state
    (3) conductivity tensor
    (4) dielectric function
  *************************************************************************/

  input_logical("CDDF.start",&CDDF_on,0); /* default=off */
  input_double("CDDF.FWHM",&CDDF_FWHM,(double)0.2); /* default: 0.2 eV */
  input_double("CDDF.maximum_energy",&CDDF_max_eV,(double)10.0); /* default: 10.0 eV */
  input_double("CDDF.minimum_energy",&CDDF_min_eV,(double) 0.0); /* default: 0.0 eV */
  input_double("CDDF.additional_maximum_energy",&CDDF_AddMaxE,(double)0.0); /* default: 0.0 eV (if CDDF_max_unoccupied_state==0) */
  input_int("CDDF.frequency.grid.total_number",&CDDF_freq_grid_number,10000); /* default: 10000 grids */
  input_int("CDDF.maximum_unoccupied_state",&CDDF_max_unoccupied_state,0); /* default: 0 => off , >=1 => on */
  input_int("CDDF.material_type",&CDDF_material_type,0); /* default: 0, 0 = insulator , 1 = metal */

  i_vec2[0]=Kspace_grid1;
  i_vec2[1]=Kspace_grid2;
  i_vec2[2]=Kspace_grid3;
  input_intv("CDDF.Kgrid",3,i_vec,i_vec2);
  CDDF_Kspace_grid1 = i_vec[0];
  CDDF_Kspace_grid2 = i_vec[1];
  CDDF_Kspace_grid3 = i_vec[2];

  if (Kspace_grid1==1 && Kspace_grid2==1 && Kspace_grid3==1 
      && (CDDF_Kspace_grid1!=1 || CDDF_Kspace_grid2!=1 || CDDF_Kspace_grid3!=1) ){

    if (myid==Host_ID){
      printf("You want to 'Cluster' calculation for SCF, but 'Band' calculation for CDDF calculation.\n ");
      printf("This is not allowed. Please use the same solver.\n ");
      printf("For this purpose, scf.Kgrid or CDDF.Kgrid can be changed.\n ");
    }

    MPI_Finalize();
    exit(0);
  }

  if (CDDF_Kspace_grid1==1 && CDDF_Kspace_grid2==1 && CDDF_Kspace_grid3==1 
      && (Kspace_grid1!=1 || Kspace_grid2!=1 || Kspace_grid3!=1) ){

    if (myid==Host_ID){
      printf("You want to 'Band' calculation for SCF, but 'Cluster' calculation for CDDF calculation.\n ");
      printf("This is not allowed. Please use the same solver.\n ");
      printf("For this purpose, scf.Kgrid or CDDF.Kgrid can be changed.\n ");
    }

    MPI_Finalize();
    exit(0);
  }

  /* YTL-end */

  /********************************************************
    Effective Screening Medium (ESM) method Calculation 
                                      added by T.Ohwaki                                   
  *********************************************************/

  s_vec[0]="off"; s_vec[1]="on1"; s_vec[2]="on2"; s_vec[3]="on3"; s_vec[4]="on4";
  i_vec[0]=0    ; i_vec[1]=1    ; i_vec[2]=2    ; i_vec[3]=3    ; i_vec[4]=4    ;
  input_string2int("ESM.switch", &ESM_switch, 5, s_vec,i_vec);

  s_vec[0]="off"; s_vec[1]="on";
  i_vec[0]=0    ; i_vec[1]=1   ;
  input_string2int("ESM.wall.switch", &ESM_wall_switch, 2, s_vec,i_vec);

  /* added by AdvanceSoft */
  s_vec[0]="x"; s_vec[1]="y"; s_vec[2]="z";
  i_vec[0]=1  ; i_vec[1]=2  ; i_vec[2]=3  ;
  input_string2int("ESM.direction", &ESM_direction, 3, s_vec,i_vec);
  switch (ESM_direction){
  case 1:
    iESM[1]=1; iESM[2]=2; iESM[3]=3;
    break;
  case 2:
    iESM[1]=2; iESM[2]=3; iESM[3]=1;
    break;
  case 3:
    iESM[1]=3; iESM[2]=1; iESM[3]=2;
    break;
  }

  if (myid==Host_ID && 0<level_stdout){

    if (ESM_switch==1){
      printf("\n");
      printf("********************************************************** \n");
      printf("   Effective Screening Medium (ESM) method calculation     \n");
      printf("                                                           \n");
      printf("    The following calc. is implemented with ESM method.    \n");
      printf("    Boundary condition = Vacuum|Vacuum|Vacuum              \n");
      printf("                                                           \n");
      printf("        Copyright (C), 2011-2019, T.Ohwaki and M.Otani     \n");
      printf("********************************************************** \n");
      printf("\n");
    }

    else if (ESM_switch==2){
      printf("\n");
      printf("********************************************************** \n");
      printf("   Effective Screening Medium (ESM) method calculation     \n");
      printf("                                                           \n");
      printf("    The following calc. is implemented with ESM method.    \n");
      printf("    Boundary condition = Metal|Vacuum|Metal                \n");
      printf("                                                           \n");
      printf("        Copyright (C), 2011-2019, T.Ohwaki and M.Otani     \n");
      printf("********************************************************** \n");
      printf("\n");
    }

    else if (ESM_switch==3){
      printf("\n");
      printf("********************************************************** \n");
      printf("   Effective Screening Medium (ESM) method calculation     \n");
      printf("                                                           \n");
      printf("    The following calc. is implemented with ESM method.    \n");
      printf("    Boundary condition = Vacuum|Vacuum|Metal               \n");
      printf("                                                           \n");
      printf("        Copyright (C), 2011-2019, T.Ohwaki and M.Otani     \n");
      printf("********************************************************** \n");
      printf("\n");
    }

    else if (ESM_switch==4){
      printf("\n");
      printf("********************************************************** \n");
      printf("   Effective Screening Medium (ESM) method calculation     \n");
      printf("                                                           \n");
      printf("    The following calc. is implemented with ESM method.    \n");
      printf("    Boundary condition = Metal|Vacuum|Metal                \n");
      printf("                         plus Uniform electric field       \n");
      printf("                                                           \n");
      printf("        Copyright (C), 2011-2019, T.Ohwaki and M.Otani     \n");
      printf("********************************************************** \n");
      printf("\n");
    }

  }

  input_double("ESM.potential.diff",&V_ESM,(double)0.0);  /* eV */
  /* change the unit from eV to Hartree */
  V_ESM = V_ESM/eV2Hartree;

  if (ESM_switch!=0 && ESM_switch!=4 && 1.0e-13<fabs(V_ESM)){
    if (myid==Host_ID){
      printf("<ESM:Warning> Non-zero ESM.ElectricField is not valid except for on4.\n\n");
    }

    MPI_Finalize();
    exit(0);
  }

  /* This means the distance from the upper edge along the a-axis (x-coordinate). */
  input_double("ESM.wall.position",&ESM_wall_position,(double)1.0); /* Angstrom */
  /* change the unit from ang. to a.u. */
  ESM_wall_position /= BohrR; 

  if (ESM_wall_switch==1){

    if (ESM_wall_position<0.0){
      if (myid==Host_ID){
	printf("<ESM:Warning> ESM.wall.position must be positive.\n\n");
      }

      MPI_Finalize();
      exit(0);
    }

    input_double("ESM.wall.height",&ESM_wall_height,(double)100.0); /* eV */
    /* change the unit from eV to Hartree */
    ESM_wall_height /= eV2Hartree;

    input_double("ESM.buffer.range",&ESM_buffer_range,(double)10.0); /* Angstrom */
    /* change the unit from ang. to a.u. */
    ESM_buffer_range /= BohrR; 

  } /* ESM_wall_switch */

  /* Artificially imposed force  */
  /* added by T.Ohwaki */

  s_vec[0]="off"; s_vec[1]="on";
  i_vec[0]=0    ; i_vec[1]=1   ;
  Arti_Force = 0; /* default = off */
  input_string2int("MD.Artificial_Force", &Arti_Force, 2, s_vec,i_vec);
  input_double("MD.Artificial_Grad",&Arti_Grad,(double)0.0); /* Hartree/Bohr */

  if (myid==Host_ID && Arti_Force==1 && 0<level_stdout){
    printf("\n");
    printf("##################################################### \n");
    printf("##                                                 ## \n");
    printf("## An artificial force is imposed on the 1st atom. ## \n");
    printf("##  * Gradient = %12.9f (Hartree/Bohr)                \n",Arti_Grad);
    printf("##                                                 ## \n");
    printf("##################################################### \n");
    printf("\n");
  }

  /*****************************************************
  if restart files for geometry optimization exist, 
  read data from them. 
  default = off
  *****************************************************/

  input_logical("geoopt.restart",&GeoOpt_RestartFromFile, 0); 

  /****************************************************
                         atoms
  ****************************************************/

  /* except for NEGF */

  if (Solver!=4){  

    /* atom */

    input_int("Atoms.Number",&atomnum,0);

    if (atomnum<=0){
      printf("Atoms.Number may be wrong.\n");
      po++;
    }
    List_YOUSO[1] = atomnum + 1;

    /* memory allocation */
    Allocate_Arrays(1);

    /* initialize */

    s_vec[0]="Ang";  s_vec[1]="AU";   s_vec[2]="FRAC";
    i_vec[0]= 0;     i_vec[1]= 1;     i_vec[2]= 2;
    input_string2int("Atoms.SpeciesAndCoordinates.Unit",
                     &coordinates_unit,3,s_vec,i_vec);

    if (fp=input_find("<Atoms.SpeciesAndCoordinates") ) {

      for (i=1; i<=atomnum; i++){
        fgets(buf,MAXBUF,fp);

	/* spin non-collinear */ 
	if (SpinP_switch==3){

	  /*******************************************************
               (1) spin non-collinear
	  *******************************************************/

	  sscanf(buf,"%i %s %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %s",
		 &j, Species,
		 &Gxyz[i][1],&Gxyz[i][2],&Gxyz[i][3],
		 &InitN_USpin[i],&InitN_DSpin[i],
		 &Angle0_Spin[i], &Angle1_Spin[i], 
		 &Angle0_Orbital[i], &Angle1_Orbital[i],
		 &Constraint_SpinAngle[i],
		 OrbPol );

	  /* atomMove is initialzed as 1 */

	  if (fabs(Angle0_Spin[i])<1.0e-10){
	    Angle0_Spin[i] = Angle0_Spin[i] + rnd(1.0e-5);
	  }

	  Angle0_Spin[i] = PI*Angle0_Spin[i]/180.0 + rnd(1.0e-10);
	  Angle1_Spin[i] = PI*Angle1_Spin[i]/180.0 + rnd(1.0e-10);
	  InitAngle0_Spin[i] = Angle0_Spin[i];
	  InitAngle1_Spin[i] = Angle1_Spin[i];

	  if (fabs(Angle0_Orbital[i])<1.0e-10){
	    Angle0_Orbital[i] = Angle0_Orbital[i] + rnd(1.0e-5);
	  }

	  Angle0_Orbital[i] = PI*Angle0_Orbital[i]/180.0;
	  Angle1_Orbital[i] = PI*Angle1_Orbital[i]/180.0;
	  InitAngle0_Orbital[i] = Angle0_Orbital[i];
	  InitAngle1_Orbital[i] = Angle1_Orbital[i];


          /*************************************************************************
           check whether the Euler angle measured from the direction (1,0) is used,
           if not, change the Euler angle 
	  *************************************************************************/

          if ( (InitN_USpin[i]-InitN_DSpin[i])<0.0 ){

	    mx = -sin(InitAngle0_Spin[i])*cos(InitAngle1_Spin[i]);
	    my = -sin(InitAngle0_Spin[i])*sin(InitAngle1_Spin[i]);
            mz = -cos(InitAngle0_Spin[i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);

	    Angle0_Spin[i] = S_coordinate[1];
  	    Angle1_Spin[i] = S_coordinate[2];
	    InitAngle0_Spin[i] = Angle0_Spin[i];
  	    InitAngle1_Spin[i] = Angle1_Spin[i];

            tmp = InitN_USpin[i];
            InitN_USpin[i] = InitN_DSpin[i];
            InitN_DSpin[i] = tmp; 

	    mx = -sin(InitAngle0_Orbital[i])*cos(InitAngle1_Orbital[i]);
	    my = -sin(InitAngle0_Orbital[i])*sin(InitAngle1_Orbital[i]);
            mz = -cos(InitAngle0_Orbital[i]);

            xyz2spherical(mx,my,mz,0.0,0.0,0.0,S_coordinate);
             
            Angle0_Orbital[i] = S_coordinate[1];
            Angle1_Orbital[i] = S_coordinate[2]; 
  	    InitAngle0_Orbital[i] = Angle0_Orbital[i];
	    InitAngle1_Orbital[i] = Angle1_Orbital[i];
          }
	}

	/**************************************************
                  (2) spin collinear
	**************************************************/

	else{ 

	  sscanf(buf,"%i %s %lf %lf %lf %lf %lf %s",
		 &j, Species,
		 &Gxyz[i][1],&Gxyz[i][2],&Gxyz[i][3],
		 &InitN_USpin[i],&InitN_DSpin[i], OrbPol );
	}

        WhatSpecies[i] = Species2int(Species);

        if (Hub_U_switch==1) OrbPol_flag[i] = OrbPol2int(OrbPol);
 
        if (i!=j){
          printf("Format error of the sequential number %i in <Atoms.SpeciesAndCoordinates\n",j);
          po++;
        }

        if (2<=level_stdout){
          printf("<Input_std>  ct_AN=%2d WhatSpecies=%2d USpin=%7.4f DSpin=%7.4f\n",
                  i,WhatSpecies[i],InitN_USpin[i],InitN_DSpin[i]);
        }
      }

      ungetc('\n',fp);
      if (!input_last("Atoms.SpeciesAndCoordinates>")) {
        /* format error */
        printf("Format error for Atoms.SpeciesAndCoordinates\n");
        po++;
      }

    }

    /****************************************************
       the unit of atomic coordinates is transformed 
    ****************************************************/

    /*  Ang to AU */ 
    if (coordinates_unit==0){
      for (i=1; i<=atomnum; i++){
	Gxyz[i][1] = Gxyz[i][1]/BohrR;
	Gxyz[i][2] = Gxyz[i][2]/BohrR;
	Gxyz[i][3] = Gxyz[i][3]/BohrR;
      }
    }

    /****************************************************
                          unit cell
    ****************************************************/

    s_vec[0]="Ang"; s_vec[1]="AU";
    i_vec[0]=0;  i_vec[1]=1;
    input_string2int("Atoms.UnitVectors.Unit",&unitvector_unit,2,s_vec,i_vec);

    if (fp=input_find("<Atoms.Unitvectors")) {

      for (i=1; i<=3; i++){
        fscanf(fp,"%lf %lf %lf",&tv[i][1],&tv[i][2],&tv[i][3]);
      }
      if ( ! input_last("Atoms.Unitvectors>") ) {
        /* format error */
        printf("Format error for Atoms.Unitvectors\n");
        po++;
      }

      /* Ang to AU */
      if (unitvector_unit==0){
        for (i=1; i<=3; i++){
          tv[i][1] = tv[i][1]/BohrR;
          tv[i][2] = tv[i][2]/BohrR;
          tv[i][3] = tv[i][3]/BohrR;
        }
      }
      
      /* check the cell vectors for the band calculation of the lead */
      
      if (output_hks==1){
        
        if ( tv[1][1]<0.0 || tv[1][2]<0.0 || tv[1][3]<0.0 || 
	      1.0e-12<fabs(tv[1][1]*tv[1][2]) ||
	      1.0e-12<fabs(tv[1][2]*tv[1][3]) ||
	      1.0e-12<fabs(tv[1][3]*tv[1][1]) 
           ){

          if (myid==Host_ID){
	    printf("The a-axis of the unit cell of leads used for the NEGF calculation\n");
	    printf("must be x, y, or, z-axis in the cartesian coordinate.\n");
            printf("The following a-axis will be accepted:\n\n");
 
            printf(" ax   0.0  0.0\n");
            printf("       or\n");
            printf(" 0.0  ay   0.0\n");
            printf("       or\n");
            printf(" 0.0  0.0  az\n\n");
            printf("where ax, ay, and az must be POSITIVE.\n\n");
          }
                
          MPI_Finalize();
          exit(0);
        }
      }

      /* Effective Screening Medium (ESM) method Calculation */
      /* added by T.Ohwaki                                   */

      /* modified by AdvanceSoft */
      if (ESM_switch!=0){

        if ( tv[1][iESM[1]]<0.0 || 1.0e-13<fabs(tv[1][iESM[2]]) || 1.0e-13<fabs(tv[1][iESM[3]]) || 
	      1.0e-13<fabs(tv[1][iESM[1]]*tv[1][iESM[2]]) ||
	      1.0e-13<fabs(tv[1][iESM[2]]*tv[1][iESM[3]]) ||
	      1.0e-13<fabs(tv[1][iESM[3]]*tv[1][iESM[1]]) 
           ){

          if (myid==Host_ID){
            printf("<ESM:Warning>\n");
	    printf("The a-axis of the unit cell used for the ESM calculation\n");
	    printf("must be parallel to surface normal axis,\n");
            printf("where its component must be POSITIVE.\n\n");
	  }

          MPI_Finalize();
          exit(0);
	}

        if ( 1.0e-13<fabs(tv[2][iESM[1]]) || 1.0e-13<fabs(tv[3][iESM[1]]) ){

          if (myid==Host_ID){
            printf("<ESM:Warning>\n");
	    printf("The b- and c-axes of the unit cell used for the ESM calculation\n");
	    printf("must be orthogonal to the a-axis.\n\n");
	  }

          MPI_Finalize();
          exit(0);
        }
      }      

    }

    else {
      /* automatically set up the unit cell */

      /* Effective Screening Medium (ESM) method Calculation */
      /* added by T.Ohwaki                                   */

      if (ESM_switch!=0){
	if (myid==Host_ID){
	  printf("<ESM:Warning> A unit cell must be specified in the ESM calculation.\n\n");
	}

	MPI_Finalize();
	exit(0);
      }

      /* In case of Solver==3 */

      if (Solver==3){
	if (myid==Host_ID){
	  printf("You have to give a unit cell for the band calc.\n");
	}
	MPI_Finalize();
	exit(0);
      }

      /* The other case */

      if (coordinates_unit==2){
	if (myid==Host_ID){
	  printf("You have to give a unit cell in case of use of fractional coordinates.\n");
	}
	MPI_Finalize();
	exit(0);
      }

      /* Set a unit cell */
      Set_Cluster_UnitCell(tv,unitvector_unit);
      Determine_Cell_from_ECutoff(tv,Grid_Ecut+0.001);
    }

    /*  FRAC to AU */ 
    if (coordinates_unit==2){

      /* The fractional coordinates should be kept within 0 to 1. */

      for (i=1; i<=atomnum; i++){
        for (k=1; k<=3; k++){

          itmp = (int)Gxyz[i][k]; 

          if (1.0<Gxyz[i][k]){

            /* modified by AdvanceSoft */
	    if (ESM_switch!=0 && k==ESM_direction){
	      if (myid==Host_ID){
                printf("<ESM:Warning>\n");
                printf("The fractional coordinate of a-axis for atom %d = %16.9f \n",i,Gxyz[i][k]);
		printf("The fractional coordinate of a-axis should be kept within 0 to 1 in the ESM calculation.\n\n");
	      }

	      MPI_Finalize();
	      exit(0);
	    }

            /* ended by T.Ohwaki */

            if (GeoOpt_RestartFromFile==0){

	      Gxyz[i][k] = Gxyz[i][k] - (double)itmp;

	      if (myid==Host_ID){
		if (k==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",i);
		if (k==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",i);
		if (k==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",i);
	      }
	    }
	  }
          else if (Gxyz[i][k]<0.0){

	    /* Effective Screening Medium (ESM) method Calculation */
	    /* added by T.Ohwaki */

            /* modified by AdvanceSoft */
	    if (ESM_switch!=0 && k==ESM_direction){
	      if (myid==Host_ID){
                printf("<ESM:Warning>\n");
                printf("The fractional coordinate of a-axis for atom %d = %16.9f \n",i,Gxyz[i][k]);
		printf("The fractional coordinate of a-axis should be kept within 0 to 1 in the ESM calculation.\n\n");
	      }

	      MPI_Finalize();
	      exit(0);
	    }

            /* ended by T.Ohwaki */

            if (GeoOpt_RestartFromFile==0){

	      Gxyz[i][k] = Gxyz[i][k] + (double)(abs(itmp)+1);

	      if (myid==Host_ID){
		if (k==1) printf("The fractional coordinate of a-axis for atom %d was translated within the range (0 to 1).\n",i);
		if (k==2) printf("The fractional coordinate of b-axis for atom %d was translated within the range (0 to 1).\n",i);
		if (k==3) printf("The fractional coordinate of c-axis for atom %d was translated within the range (0 to 1).\n",i);
	      }
	    }
	  }

	}
      }

      /* calculation of xyz-coordinate in A.U. The grid origin is Grid_Origin or zero. */

      if ( 1.0e+8<scf_fixed_origin[0] &&
           1.0e+8<scf_fixed_origin[1] &&
	   1.0e+8<scf_fixed_origin[2] ){

        scf_fixed_origin[0] = 0.0;
        scf_fixed_origin[1] = 0.0;
        scf_fixed_origin[2] = 0.0;

        tmpx = 0.0;
        tmpy = 0.0;
        tmpz = 0.0;
      }
      else {
        tmpx = scf_fixed_origin[0];
        tmpy = scf_fixed_origin[1];
        tmpz = scf_fixed_origin[2];
      }

      for (i=1; i<=atomnum; i++){
	x = Gxyz[i][1]*tv[1][1] + Gxyz[i][2]*tv[2][1] + Gxyz[i][3]*tv[3][1] + tmpx;
	y = Gxyz[i][1]*tv[1][2] + Gxyz[i][2]*tv[2][2] + Gxyz[i][3]*tv[3][2] + tmpy;
	z = Gxyz[i][1]*tv[1][3] + Gxyz[i][2]*tv[2][3] + Gxyz[i][3]*tv[3][3] + tmpz;
	Gxyz[i][1] = x;
	Gxyz[i][2] = y;
	Gxyz[i][3] = z;
      }
    }

  } /* if (Solver!=4){ */

  /* Atoms.Unitvectors.Velocity */

  for (i=1; i<=3; i++){
    for (j=1; j<=3; j++){
      tv_velocity[i][j] = 0.0;
    }
  }

  if (fp=input_find("<Atoms.Unitvectors.Velocity")) {

    for (i=1; i<=3; i++){
      fscanf(fp,"%lf %lf %lf",&tv_velocity[i][1],&tv_velocity[i][2],&tv_velocity[i][3]);
    }
    if ( ! input_last("Atoms.Unitvectors.Velocity>") ) {
      /* format error */
      printf("Format error for Atoms.Unitvectors.Velocity\n");
      po++;
    }
  }

  /*******************************
                NEGF 
  *******************************/

  else{

    TRAN_Input_std_Atoms(mpi_comm_level1, Solver);
  }

  /**************************************************
                          LNO
  **************************************************/

  input_double("orderN.LNO.Occ.Cutoff",&LNO_Occ_Cutoff,(double)1.0e-1);
  input_double("orderN.LNO.Buffer",&orderN_LNO_Buffer,(double)0.2);

  LNOs_Num_predefined_flag = 0;

  if (LNO_flag==1){

    if (fp=input_find("<LNOs.Num")) {    

      LNOs_Num_predefined_flag = 1;

      for (i=0; i<SpeciesNum; i++){
	fscanf(fp,"%s",Species);
        j = Species2int(Species);
	fscanf(fp,"%d",&k);
        LNOs_Num_predefined[j] = k;
      }

      if (! input_last("LNOs.Num>") ) {
	/* format error */
	printf("Format error for LNOs.Num\n");
	po++;
      }

    }  /* if (fp=input_find("<LNOs.Num"))  */
  }

  /**************************************************
          store the initial magnetic moment
  **************************************************/

  if (Constraint_NCS_switch==2){
    for (i=1; i<=atomnum; i++){
      InitMagneticMoment[i] = fabs(InitN_USpin[i] - InitN_DSpin[i]);
    }    
  }

  /*************************************
    calculation with a core hole state
  *************************************/

  input_logical("scf.core.hole",&core_hole_state_flag, 0); 

  if (fp=input_find("<core.hole.state")) {
    
    fgets(buf,MAXBUF,fp);
    sscanf(buf,"%i %s %i",&Core_Hole_Atom,Core_Hole_Orbital,&Core_Hole_J);

    /* check SpinP_switch */

    if (SpinP_switch==0){
      if (myid==Host_ID){
        printf("The calculation with a core hole is not supported for non-spin polarized calculations.\n\n");
      }
      MPI_Finalize();
      exit(0);
    }

    /* check atom index */

    if (Core_Hole_Atom<1 || atomnum<Core_Hole_Atom){
      if (myid==Host_ID){
        printf("The selected atom is invalid.\n\n");
      }
      MPI_Finalize();
      exit(0);
    }

    /* check orbital */

    if (  strcmp(Core_Hole_Orbital,"s")!=0 
       && strcmp(Core_Hole_Orbital,"p")!=0
       && strcmp(Core_Hole_Orbital,"d")!=0
       && strcmp(Core_Hole_Orbital,"f")!=0
       ){

      if (myid==Host_ID){
        printf("The selected orbital is invalid.\n\n");
      }
      MPI_Finalize();
      exit(0);
    }

    /* check the j-channel */

    if (strcmp(Core_Hole_Orbital,"s")==0) l = 0;   
    if (strcmp(Core_Hole_Orbital,"p")==0) l = 1;   
    if (strcmp(Core_Hole_Orbital,"d")==0) l = 2;   
    if (strcmp(Core_Hole_Orbital,"f")==0) l = 3;   

    if ( Core_Hole_J<0 || 2*(2*l+1)<Core_Hole_J ){

      if (myid==Host_ID){
        printf("The orbital index should be within 0 to %i.\n\n",2*(2*l+1));
      }
      MPI_Finalize();
      exit(0);
    }

  }

  /* find the shortest cell vector */

  double len1,len2,len3;

  len1 = sqrt( Dot_Product(tv[1], tv[1]) );
  len2 = sqrt( Dot_Product(tv[2], tv[2]) );
  len3 = sqrt( Dot_Product(tv[3], tv[3]) );

  Shortest_CellVec = len1;
  if (len2<Shortest_CellVec) Shortest_CellVec = len2;
  if (len3<Shortest_CellVec) Shortest_CellVec = len3;

  /* scf.coulomb.cutoff */    

  s_vec[0]="off"; s_vec[1]="on"; 
  i_vec[0]=0    ; i_vec[1]=1   ;
  input_string2int("scf.coulomb.cutoff", &scf_coulomb_cutoff, 2, s_vec,i_vec);

  /*********************************************************
               core level excitation spectrum

   1: X-ray Absorption Near Edge Structure (XANES) 
      with single electron excitations.

   2: X-ray Absorption Near Edge Structure (XANES) 
      with single and double electron excitations.

  *********************************************************/

  s_vec[0]="NONE"; s_vec[1]="XANES"; s_vec[2]="XANES1"; s_vec[3]="XANES2";

  i_vec[0]=-1;  i_vec[1]=0;  i_vec[2]=1;  i_vec[3]=2;

  input_string2int("CLE.Type", &CLE_Type, 4, s_vec,i_vec);

  input_double("CLE.Val.Window",&CLE_Val_Window,(double)1.0); /* eV */
  input_double("CLE.Con.Window",&CLE_Con_Window,(double)1.0); /* eV */

  /********************************************************
    Automatic determination of Kerker factor 

    Kerker_factor = 1/G0 inplies 
    Kerker_factor * G0 -> constant 

    Furthermore, to take account of anisotropy of system, 
    an enhancement factor is introduced by 

    y = a*(DG/AG) + 1

    where, 
    AG = (G12+G22+G32)/3;
    DG = (fabs(G22-G12)+fabs(G32-G22)+fabs(G12-G32))/3;

    In an extreme case that G22<<G12 and G32<<G12, 
    AG approaches to G12/3, and DG approaches to 2/3 G12. 
    So, DG/AG becomes 2 in the case. 

    On the other hand, if G12=G22=G32, y becomes 1.

    Taking appropriate prefactors, we define as    
    Kerker_factor = 0.5/G0*y = 0.5/G0*(4.0*DG/AG+1.0);
  *******************************************************/

  if ( Kerker_factor<0.0 && (Mixing_switch==3 || Mixing_switch==4 || Mixing_switch==7) ){

    double tmp[4];
    double CellV,G0,G12,G22,G32,DG,AG;  

    Cross_Product(tv[2],tv[3],tmp);
    CellV = Dot_Product(tv[1],tmp); 
    Cell_Volume = fabs(CellV);

    Cross_Product(tv[2],tv[3],tmp);
    rtv[1][1] = 2.0*PI*tmp[1]/CellV;
    rtv[1][2] = 2.0*PI*tmp[2]/CellV;
    rtv[1][3] = 2.0*PI*tmp[3]/CellV;

    Cross_Product(tv[3],tv[1],tmp);
    rtv[2][1] = 2.0*PI*tmp[1]/CellV;
    rtv[2][2] = 2.0*PI*tmp[2]/CellV;
    rtv[2][3] = 2.0*PI*tmp[3]/CellV;
  
    Cross_Product(tv[1],tv[2],tmp);
    rtv[3][1] = 2.0*PI*tmp[1]/CellV;
    rtv[3][2] = 2.0*PI*tmp[2]/CellV;
    rtv[3][3] = 2.0*PI*tmp[3]/CellV;

    G12 = rtv[1][1]*rtv[1][1] + rtv[1][2]*rtv[1][2] + rtv[1][3]*rtv[1][3]; 
    G22 = rtv[2][1]*rtv[2][1] + rtv[2][2]*rtv[2][2] + rtv[2][3]*rtv[2][3]; 
    G32 = rtv[3][1]*rtv[3][1] + rtv[3][2]*rtv[3][2] + rtv[3][3]*rtv[3][3]; 

    G12 = sqrt(G12);
    G22 = sqrt(G22);
    G32 = sqrt(G32);

    if (G12<G22) G0 = G12;
    else         G0 = G22;
    if (G32<G0)  G0 = G32;

    AG = (G12+G22+G32)/3.0;
    DG = (fabs(G22-G12)+fabs(G32-G22)+fabs(G12-G32))/3.0;

    Kerker_factor = 0.5/G0*(2.0*DG/AG+1.0);

    if (myid==Host_ID && 0<level_stdout){
      printf("Automatic determination of Kerker_factor:  %15.12f\n",Kerker_factor);fflush(stdout);
    }
  }

  /****************************************************
      DFT-D: 
      Added by Okuno in March 2011
  ****************************************************/

  /* for vdW  okuno */
  input_logical("scf.dftD",&dftD_switch,0);
  input_int("version.dftD",&version_dftD,2);   /* Ellner */

  if (version_dftD!=2 && version_dftD!=3){

    if (myid==Host_ID){
      printf("version.dftD should be 1 or 2.\n");
    }

    MPI_Finalize();
    exit(0);
  }
  
  /* DFT D okuno */
  if(dftD_switch){
    s_vec[1]="Ang"; s_vec[0]="AU"; /* Ellner CHANGED ORDER */
    i_vec[1]=0;  i_vec[0]=1;
    input_string2int("DFTD.Unit",&unit_dftD,2,s_vec,i_vec);
    input_double("DFTD.rcut_dftD",&rcut_dftD,100.0);

    input_double("DFTD.d",&beta_dftD,20.0);
    if(unit_dftD==0){ /* change Ang to AU */
      rcut_dftD = rcut_dftD/BohrR;
    }

    input_double("DFTD.scale6",&scal6_dftD,0.75);

    i_vec2[0]=1;
    i_vec2[1]=1;
    i_vec2[2]=1;
    input_intv("DFTD.IntDirection",3,i_vec,i_vec2);
    DFTD_IntDir1 = i_vec[0];
    DFTD_IntDir2 = i_vec[1];
    DFTD_IntDir3 = i_vec[2];

    /*
    if(myid==Host_ID){
      printf("DFTD.Unit = %i  \n",unit_dftD);
      printf("DFTD.rcut_dftD =%f\n",rcut_dftD);
      printf("DFTD.beta =%f\n",beta_dftD);
      printf("DFTD.scale6 =%f\n",scal6_dftD);
    }
    */

    for (i=1; i<=atomnum; i++){  
      Gxyz[i][60] = 1.0;
    }

    if (fp=input_find("<DFTD.periodicity")) {

      for (i=1; i<=atomnum; i++){  
	fscanf(fp,"%d %lf",&j,&Gxyz[i][60]);
      }     

      if ( ! input_last("DFTD.periodicity>") ) {
	/* format error */
	printf("Format error for DFTD.periodicity\n");
	po++;
      }
    }

    /* Ellner DFT-D3 */
    s_vec[0]="ZERO"; s_vec[1]="BJ";
    i_vec[0]=1;  i_vec[1]=2;
    input_string2int("DFTD3.damp",&DFTD3_damp_dftD,2,s_vec,i_vec);
    if(version_dftD==3) input_double("DFTD.scale6",&s6_dftD,1.0); /* FIXED */

    if ( XC_switch==4 ){
      input_double("DFTD.sr6",&sr6_dftD,1.217);
      input_double("DFTD.a1",&a1_dftD,0.4289);
      input_double("DFTD.a2",&a2_dftD,4.4407);
      if (DFTD3_damp_dftD==1)  input_double("DFTD.scale8",&s8_dftD,0.722); /* GGA-PBE ZERO DAMP */
      if (DFTD3_damp_dftD==2)  input_double("DFTD.scale8",&s8_dftD,0.7875);/* GGA-PBE BJ DAMP */
    }
    else{
      input_double("DFTD.scale8",&s8_dftD,0.5);
      input_double("DFTD.sr6",&sr6_dftD,1.0);
      input_double("DFTD.a1",&a1_dftD,0.5);
      input_double("DFTD.a2",&a2_dftD,5.0);
    }

    input_double("DFTD.cncut_dftD",&cncut_dftD,40.0);
     if(unit_dftD==0){ /* change Ang to AU */
      cncut_dftD = cncut_dftD/BohrR;
    }

  } /* if (dftD_switch) */

  /************************************************************
   set fixed components of cell vectors in cell optimization 

    1: fixed 
    0: relaxed
  ************************************************************/

  for (i=1; i<=3; i++){  
    for (j=1; j<=3; j++){  
      Cell_Fixed_XYZ[i][j] = 0;
    }
  }

  if (fp=input_find("<MD.Fixed.Cell.Vectors")) {

    for (i=1; i<=3; i++){  
      fscanf(fp,"%d %d %d",&Cell_Fixed_XYZ[i][1],&Cell_Fixed_XYZ[i][2],&Cell_Fixed_XYZ[i][3]);
    }  

    if ( ! input_last("MD.Fixed.Cell.Vectors>") ) {
      /* format error */
      printf("Format error for MD.Fixed.Cell.Vectors\n");
      po++;
    }
  }

  /* RFC5 */

  if (MD_switch==18 && cellopt_swtich==5){
    for (i=1; i<=3; i++){  
      for (j=1; j<=3; j++){  
	Cell_Fixed_XYZ[i][j] = 0;
      }
    }
  }

  /* RFC6 */

  else if (MD_switch==18 && cellopt_swtich==6){
    Cell_Fixed_XYZ[1][1] = 0; Cell_Fixed_XYZ[1][2] = 0; Cell_Fixed_XYZ[1][3] = 0;
    Cell_Fixed_XYZ[2][1] = 0; Cell_Fixed_XYZ[2][2] = 0; Cell_Fixed_XYZ[2][3] = 0;
    Cell_Fixed_XYZ[3][1] = 1; Cell_Fixed_XYZ[3][2] = 1; Cell_Fixed_XYZ[3][3] = 1;
  }

  /* RFC7 */

  else if (MD_switch==18 && cellopt_swtich==7){
    Cell_Fixed_XYZ[1][1] = 0; Cell_Fixed_XYZ[1][2] = 0; Cell_Fixed_XYZ[1][3] = 0;
    Cell_Fixed_XYZ[2][1] = 1; Cell_Fixed_XYZ[2][2] = 1; Cell_Fixed_XYZ[2][3] = 1;
    Cell_Fixed_XYZ[3][1] = 1; Cell_Fixed_XYZ[3][2] = 1; Cell_Fixed_XYZ[3][3] = 1;
  }

  /* OptC6 */

  else if (MD_switch==17 && cellopt_swtich==6){
    Cell_Fixed_XYZ[1][1] = 0; Cell_Fixed_XYZ[1][2] = 0; Cell_Fixed_XYZ[1][3] = 0;
    Cell_Fixed_XYZ[2][1] = 0; Cell_Fixed_XYZ[2][2] = 0; Cell_Fixed_XYZ[2][3] = 0;
    Cell_Fixed_XYZ[3][1] = 1; Cell_Fixed_XYZ[3][2] = 1; Cell_Fixed_XYZ[3][3] = 1;
  }

  /* OptC7 */

  else if (MD_switch==17 && cellopt_swtich==7){
    Cell_Fixed_XYZ[1][1] = 0; Cell_Fixed_XYZ[1][2] = 0; Cell_Fixed_XYZ[1][3] = 0;
    Cell_Fixed_XYZ[2][1] = 1; Cell_Fixed_XYZ[2][2] = 1; Cell_Fixed_XYZ[2][3] = 1;
    Cell_Fixed_XYZ[3][1] = 1; Cell_Fixed_XYZ[3][2] = 1; Cell_Fixed_XYZ[3][3] = 1;
  }

  /****************************************************
   set fixed atomic position in geometry optimization
   and MD:  

      1: fixed 
      0: relaxed
  ****************************************************/

  if (fp=input_find("<MD.Fixed.XYZ")) {

    for (i=1; i<=atomnum; i++){  
      fscanf(fp,"%d %d %d %d",
             &j,&atom_Fixed_XYZ[i][1],&atom_Fixed_XYZ[i][2],&atom_Fixed_XYZ[i][3]);
    }  

    if ( ! input_last("MD.Fixed.XYZ>") ) {
      /* format error */
      printf("Format error for MD.Fixed.XYZ\n");
      po++;
    }
  }

  /****************************************************
             set initial velocities for MD
  ****************************************************/

  MD_Init_Velocity = 0;

  /* at this moment */

  if (fp=input_find("<MD.Init.Velocity")) {

    MD_Init_Velocity = 1;

    for (i=1; i<=atomnum; i++){  

      fscanf(fp,"%d %lf %lf %lf",&j,&Gxyz[i][24],&Gxyz[i][25],&Gxyz[i][26]);

      /***********************************************
          5.291772083*10^{-11} m / 2.418884*10^{-17} s 
          = 2.1876917*10^6 m/s                         
          = 1 a.u. for velocity 

          1 m/s = 0.4571028 * 10^{-6} a.u.
      ***********************************************/
         
      for (j=1; j<=3; j++){        
	if (atom_Fixed_XYZ[i][j]==0){
	  Gxyz[i][23+j] = Gxyz[i][23+j]*0.4571028*0.000001;
	  Gxyz[i][26+j] = Gxyz[i][23+j];
	}

	else{
	  Gxyz[i][23+j] = 0.0;
	  Gxyz[i][26+j] = 0.0;
	}
      }
    }  

    if ( ! input_last("MD.Init.Velocity>") ) {
      /* format error */
      printf("Format error for MD.Init.Velocity\n");
      po++;
    }
  }

  /* one step before */

  if (fp=input_find("<MD.Init.Velocity.Prev")) {

    MD_Init_Velocity = 1;

    for (i=1; i<=atomnum; i++){  

      fscanf(fp,"%d %lf %lf %lf",&j,&Gxyz[i][27],&Gxyz[i][28],&Gxyz[i][29]);

      /***********************************************
          5.291772083*10^{-11} m / 2.418884*10^{-17} s 
          = 2.1876917*10^6 m/s                         
          = 1 a.u. for velocity 

          1 m/s = 0.4571028 * 10^{-6} a.u.
      ***********************************************/
         
      for (j=1; j<=3; j++){        
	if (atom_Fixed_XYZ[i][j]==0){
	  Gxyz[i][26+j] = Gxyz[i][26+j]*0.4571028*0.000001;
	}

	else{
	  Gxyz[i][26+j] = 0.0;
	}
      }
    }  

    if ( ! input_last("MD.Init.Velocity.Prev>") ) {
      /* format error */
      printf("Format error for MD.Init.Velocity.Prev\n");
      po++;
    }
  }

  /* parameters for the Nose-Hoover thermostat */

  input_double("NH.R",    &NH_R,(double)0.0);
  input_double("NH.nzeta",&NH_nzeta,(double)0.0);
  input_double("NH.czeta",&NH_czeta,(double)0.0);

  /*************************************************************
   set atom groups for multi heat-bath thermostat for MD (VS4):
                                     added by T. Ohwaki
  *************************************************************/

  input_int("MD.num.AtomGroup",&num_AtGr,1);

  Allocate_Arrays(10);
  int chk_vs4;

  for (k=1; k<=num_AtGr; k++){
    atnum_AtGr[k] = 0;
  }

  if (fp=input_find("<MD.AtomGroup")){

    for (i=1; i<=atomnum; i++){
      fscanf(fp,"%d %d", &j, &AtomGr[i]);
      chk_vs4 = 0;

      for (k=1; k<=num_AtGr; k++){
        if (AtomGr[i] == k){
          atnum_AtGr[k]+=1;
          chk_vs4 = 1;
        }
      }

      if (chk_vs4 ==0){
        printf("Please check your input for atom group!!\n");
        po++;
      }
    }

    if ( ! input_last("MD.AtomGroup>") ) {
      /* format error */
      printf("Format error for MD.AtomGroup\n");
      po++;
    }

    if (myid==Host_ID && 0<level_stdout){
      printf("\n");
      printf("*************************************************************** \n");
      printf("  Multi heat-bath MD calculation with velocity scaling method   \n");
      printf("                                                                \n");
      printf("  Number of atom groups = %d \n", num_AtGr);
      for (k=1; k<=num_AtGr; k++){
	printf("  Number of atoms in group  %d  =  %d \n", k, atnum_AtGr[k]);
      }
      for (i=1; i<=atomnum; i++){
	printf("  Atom  %d  : group  %d  \n", i, AtomGr[i]);
      }
      printf("                                                                \n");
      printf("*************************************************************** \n");
      printf("\n");
    }
  }

  /****************************************************
         Starting point of  LDA+U    --- by MJ
  ****************************************************/

  /* for LDA+U */ 

  if (Hub_U_switch == 1){                              /* --- MJ */

    if (fp=input_find("<Hubbard.U.values")) {    

      /* initialize the U-values */ 
      for (i=0; i<SpeciesNum; i++){
	for (l=0; l<=Spe_MaxL_Basis[i]; l++){
	  for (mul=0; mul<Spe_Num_Basis[i][l]; mul++){
	    Hub_U_Basis[i][l][mul]=0.0 ;
	  }
	}
      }
	
      /* read the U-values from the '.dat' file  */    /* --- MJ */
      for (i=0; i<SpeciesNum; i++){
	fscanf(fp,"%s",Species);
        j = Species2int(Species);

	for (l=0; l<=Spe_MaxL_Basis[j]; l++){
	  for (mul=0; mul<Spe_Num_Basis[j][l]; mul++){
	    fscanf(fp,"%s %lf", buf, &Hub_U_Basis[j][l][mul]) ;
	  }
	}
      }

      if (! input_last("Hubbard.U.values>") ) {
	/* format error */
	printf("Format error for Hubbard.U.values\n");
	po++;
      }

    }   /*  if (fp=input_find("<Hubbard.U.values"))  */

    for (i=0; i<SpeciesNum; i++){
      for (l=0; l<=Spe_MaxL_Basis[i]; l++){
	for (mul=0; mul<Spe_Num_Basis[i][l]; mul++){
          Hub_U_Basis[i][l][mul] = Hub_U_Basis[i][l][mul]/eV2Hartree;
	}
      }
    }

  }   /*  if (Hub_U_switch == 1)  */ 

  /****************************************************
      Finished reading Hubbard U values   --- by MJ
  ****************************************************/

  /****************************************************
	Reading Hund J values by S.Ryee
  ****************************************************/

  /* for general LDA+U scheme */ 

  if (Hub_U_switch == 1 && Hub_Type == 2 && Yukawa_on!=1){      
    if (fp=input_find("<Hund.J.values")) {    

      /* initialize the J values */ 
      for (i=0; i<SpeciesNum; i++){
	for (l=0; l<=Spe_MaxL_Basis[i]; l++){
	  for (mul=0; mul<Spe_Num_Basis[i][l]; mul++){
	    Hund_J_Basis[i][l][mul]=0.0 ;
	  }
	}
      }
	
      /* read the J values from the '.dat' file  */    
      for (i=0; i<SpeciesNum; i++){
	fscanf(fp,"%s",Species);
        j = Species2int(Species);

	for (l=0; l<=Spe_MaxL_Basis[j]; l++){
	  for (mul=0; mul<Spe_Num_Basis[j][l]; mul++){
	    fscanf(fp,"%s %lf", buf, &Hund_J_Basis[j][l][mul]) ;
	  }
	}
      }

      if (! input_last("Hund.J.values>") ) {
	/* format error */
	printf("Format error for Hund.J.values\n");
	po++;
      }

    }   

    for (i=0; i<SpeciesNum; i++){
      for (l=0; l<=Spe_MaxL_Basis[i]; l++){
	for (mul=0; mul<Spe_Num_Basis[i][l]; mul++){
          Hund_J_Basis[i][l][mul] = Hund_J_Basis[i][l][mul]/eV2Hartree;
	}
      }
    }

  }   /*  if (Hub_Type == 2)  */ 

  /****************************************************
	Finished reading Hund J values by S.Ryee
  ****************************************************/

  /****************************************************
      Generating Coulomb interaction array by S.Ryee
        (only when Hub_Type=2 is used)
  ****************************************************/

  if (Hub_U_switch==1 && Hub_Type==2){

    Nmul=0;
    for(i=0; i<SpeciesNum; i++){
      for(dum_l=0; dum_l<=Spe_MaxL_Basis[i]; dum_l++){
        for(dum_mul=0; dum_mul<Spe_Num_Basis[i][dum_l]; dum_mul++){
	  Uval = Hub_U_Basis[i][dum_l][dum_mul];
          Jval = Hund_J_Basis[i][dum_l][dum_mul];
          if (Uval > 1.0e-5 || Jval > 1.0e-5){
            Nmul++;
            Nonzero_UJ[i][dum_l][dum_mul] = Nmul;
          }
          else{
            Nonzero_UJ[i][dum_l][dum_mul] = 0;
          }
        }
      }
    } 

    /* Allocating array */
    Coulomb_Array = (double*****)malloc(sizeof(double****)*(Nmul+1));
    for(i=0; i<=Nmul; i++){
      Coulomb_Array[i] = (double****)malloc(sizeof(double***)*7);
      for(a=0; a<7; a++){
        Coulomb_Array[i][a] = (double***)malloc(sizeof(double**)*7);
        for(b=0; b<7; b++){
          Coulomb_Array[i][a][b] = (double**)malloc(sizeof(double*)*7);
          for(c=0; c<7; c++){
            Coulomb_Array[i][a][b][c] = (double*)malloc(sizeof(double)*7);
            for(d=0; d<7; d++){
              Coulomb_Array[i][a][b][c][d] = 0.0;
            }
          }
        }
      }
    }

  } 

  /*********************************************
   Finished generating Coulomb interaction array
      (only when Hub_Type=2 is used)
  *********************************************/

  /****************************************************
           the maximum number of SCF iterations
  ****************************************************/

  input_int("scf.maxIter",&DFTSCF_loop,40);
  if (DFTSCF_loop<0) {
    printf("scf.maxIter should be over 0.\n");
    po++;
  }

  /****************************************************
             parameters for O(N^2) method
  ****************************************************/

  input_int("scf.Npoles.ON2",&ON2_Npoles,100);
  if (ON2_Npoles<0) {
    printf("scf.Npoles.ON2 should be over 0.\n");
    po++;
  }

  /****************************************************
             parameters for the EGAC method
  ****************************************************/

  input_int("scf.Npoles.EGAC",&EGAC_Npoles,80);
  if (EGAC_Npoles<0) {
    printf("scf.Npoles.EGAC should be over 0.\n");
    po++;
  }

  EGAC_Npoles_CF = EGAC_Npoles;

  input_int("scf.DIIS.History.EGAC",&DIIS_History_EGAC,2);
  input_logical("scf.AC.flag.EGAC",&AC_flag_EGAC,0); /* default off */
  input_logical("scf.GF.EGAC",&scf_GF_EGAC,0);       /* default off */

  /****************************************************
                 Net charge of the system
  ****************************************************/

  input_double("Atoms.NetCharge",&Given_Total_Charge,(double)0.0);

  /*****************************************************
  if restart files exist, read data from them. 
  default = off
  *****************************************************/

  s_vec[0]="off";      i_vec[0] = 0;   /* default, off for first MD, later on */ 
  s_vec[1]="alloff";   i_vec[1] = -1;  /* switch off explicitly */ 
  s_vec[2]="on";       i_vec[2] = 1;   /* switch on explicitly */ 
  s_vec[3]="c2n";      i_vec[3] = 2;   /* restart for non-collinear calc. with a collinear restart file */ 

  input_string2int("scf.restart", &Scf_RestartFromFile, 4, s_vec,i_vec);

  /* specify the name of restart files, default is the same as System.Name */

  ret = input_string("scf.restart.filename",restart_filename,filename);

  scf_coulomb_cutoff_CoreHole = 0;
  if (scf_coulomb_cutoff==1 && Scf_RestartFromFile==1 && ret==1){
    scf_coulomb_cutoff_CoreHole = 1;
  }

  /****************************************************
         non-self-consistent one-shot calculation
         added by Po-Hao Chang (pohao)
  ****************************************************/

  Use_of_Collinear_Restart = 0;

  if (Scf_RestartFromFile==2){
    Scf_RestartFromFile = 1;
    Use_of_Collinear_Restart = 1;
  }

  input_double("scf.Restart.Spin.Angle.Theta",&Restart_Spin_Angle_Theta,(double)0.0);
  input_double("scf.Restart.Spin.Angle.Phi",&Restart_Spin_Angle_Phi,(double)0.0);

  if (Use_of_Collinear_Restart==1 && Scf_RestartFromFile==1){
 
    sprintf(file_check,"%s%s_rst/%s.crst_check",filepath,restart_filename,restart_filename);

    if ((fp = fopen(file_check,"r")) != NULL){
      fscanf(fp,"%d %d %d %d %d",&i_vec[0],&i_vec[1],&i_vec[2],&i_vec[3],&i_vec[4]);
      fclose(fp);

      SpinP_switch_RestartFiles = i_vec[4]; 

      if ( SpinP_switch!=3 ){

	if (myid==Host_ID){
	  printf("Restart.with.Collinear.Restart.Files=on is supported only for non-collinear calculations.\n");
	}

	MPI_Finalize(); 
	exit(0);
      }

      if (i_vec[4]==3){

	if (myid==Host_ID){
	  printf("Restart files should be obtained from a collinear calculation in case of Restart.with.Collinear.Restart.Files=on.\n.");
	}

	MPI_Finalize(); 
	exit(0);
      }

    }
    else{
      printf("Failure of reading %s\n",file_check);
    }
  } 

  /* check the number of processors, and if the number of processors in the current job 
     is different from that for the previous calculation which generated the restart files,
     the restart files are restructured.  
  */

  if (Scf_RestartFromFile==1){

    MPI_Comm_size(mpi_comm_level1,&numprocs1);
    sprintf(file_check,"%s%s_rst/%s.crst_check",filepath,restart_filename,restart_filename);

    if ((fp = fopen(file_check,"r")) != NULL){
      fscanf(fp,"%d %d %d %d %d",&i_vec[0],&i_vec[1],&i_vec[2],&i_vec[3],&i_vec[4]);
      fclose(fp);
      MPI_Barrier(mpi_comm_level1);

      if (i_vec[0]!=numprocs1){
        Remake_RestartFile(numprocs1,i_vec[0],i_vec[1],i_vec[2],i_vec[3],i_vec[4]); 
      }
    }
    else{
      printf("Failure of reading %s\n",file_check);
    }
  }

  /****************************************************                                                                        Generalized Bloch 
   added by T. B. Prayitno and supervised by Prof. F. Ishii
  ****************************************************/

  input_logical("scf.Generalized.Bloch",&GB_switch,0);

  if (SpinP_switch!=3 && GB_switch==1){
    if (myid==Host_ID){
      printf("Generalized Bloch theorem is not supported for the collinear DFT calculation.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (GB_switch==1 && SO_switch==1){
    if (myid==Host_ID){
      printf("Spin-orbit coupling is not supported for the generalized Bloch theorem.\n");
      printf("Check your input file.\n\n");
    }
    MPI_Finalize();
    exit(0);
  }

  else {
    r_vec[0]=0.0; r_vec[1]=0.0; r_vec[2]=0.0;
    input_doublev("Spin.Spiral.Vector",3,r_vec2,r_vec);
    q1_GB = r_vec2[0]; q2_GB = r_vec2[1]; q3_GB = r_vec2[2];
  }

  /****************************************************
                    Band dispersion
  ****************************************************/

  input_logical("Band.dispersion",&Band_disp_switch, 0);

  Band_kpath=NULL;
  Band_kname=NULL;
  Band_N_perpath=NULL;
  input_int("Band.Nkpath",&Band_Nkpath,0);
  if (2<=level_stdout) printf("Band.Nkpath=%d\n",Band_Nkpath);

  if (Band_Nkpath>0) {

       Band_kPathUnit=0;
       if (fp=input_find("<Band.kpath.UnitCell") ) {   
          Band_kPathUnit=1;
          for (i=1;i<=3;i++) {
            fscanf(fp,"%lf %lf %lf",&Band_UnitCell[i][1],
                      &Band_UnitCell[i][2],&Band_UnitCell[i][3]);
          }
          if ( ! input_last("Band.kpath.UnitCell>") ) {
          /* format error */
           printf("Format error near Band.kpath.UnitCell>\n");
           po++;
          }

          if (myid==Host_ID){
            for (i=1;i<=3;i++) {
              printf("%lf %lf %lf\n",Band_UnitCell[i][1],
                        Band_UnitCell[i][2],Band_UnitCell[i][3]);
            }
	  }
          
          if (unitvector_unit==0) {
            for (i=1;i<=3;i++)
              for (j=1;j<=3;j++)
               Band_UnitCell[i][j]=Band_UnitCell[i][j]/BohrR;
          }
       }
       else {
          for (i=1;i<=3;i++) 
            for (j=1;j<=3;j++) 
               Band_UnitCell[i][j]=tv[i][j];
       }

    /* allocate */

    Band_N_perpath=(int*)malloc(sizeof(int)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++) Band_N_perpath[i] = 0;

    Band_kpath = (double***)malloc(sizeof(double**)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++){
      Band_kpath[i] = (double**)malloc(sizeof(double*)*3);
      for (j=0; j<3; j++){
        Band_kpath[i][j] = (double*)malloc(sizeof(double)*4);
        for (k=0; k<4; k++) Band_kpath[i][j][k] = 0.0;
      }
    }

    Band_kname = (char***)malloc(sizeof(char**)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++){
      Band_kname[i] = (char**)malloc(sizeof(char*)*3);
      for (j=0; j<3; j++){
        Band_kname[i][j] = (char*)malloc(sizeof(char)*YOUSO10);
      }
    }

    /* end of allocation */

    if (myid==Host_ID && 2<=level_stdout) printf("kpath\n");

    if (fp=input_find("<Band.kpath") ) {
      for (i=1; i<=Band_Nkpath; i++) {
        fscanf(fp,"%d %lf %lf %lf %lf %lf %lf %s %s",
         &Band_N_perpath[i]  , 
         &Band_kpath[i][1][1], &Band_kpath[i][1][2],&Band_kpath[i][1][3],
         &Band_kpath[i][2][1], &Band_kpath[i][2][2],&Band_kpath[i][2][3],
         Band_kname[i][1],Band_kname[i][2]);

	if (myid==Host_ID && 2<=level_stdout){
          printf("%d (%lf %lf %lf) (%lf %lf %lf) %s %s\n",
          Band_N_perpath[i]  ,
          Band_kpath[i][1][1], Band_kpath[i][1][2],Band_kpath[i][1][3],
          Band_kpath[i][2][1], Band_kpath[i][2][2],Band_kpath[i][2][3],
          Band_kname[i][1],Band_kname[i][2]);
	}

      }
      if ( ! input_last("Band.kpath>") ) {
         /* format error */
         printf("Format error near Band.kpath>\n");
         po++;
      }
    }
    else {
      /* format error */
            printf("<Band.kpath is necessary.\n");
            po++;
    }
    if (Band_kPathUnit){
      kpath_changeunit( tv, Band_UnitCell, Band_Nkpath, Band_kpath );
    }
 
  }

  /****************************************************
                   One dimentional FFT
  ****************************************************/

  input_int("1DFFT.NumGridK",&Ngrid_NormK,900);

  if (Ngrid_NormK<GL_Mesh)
    List_YOUSO[15] = GL_Mesh + 10;
  else 
    List_YOUSO[15] = Ngrid_NormK + 2;

  input_double("1DFFT.EnergyCutoff",&ecutoff1dfft,(double)3600.0);
  PAO_Nkmax=sqrt(ecutoff1dfft);

  input_int("1DFFT.NumGridR",&OneD_Grid,900);

  /************************************************************
    it it not allowed to make the simultaneuos specification
    of spin non-collinear and orbital optimization
  ************************************************************/

  if (SpinP_switch==3 && Cnt_switch==1){
    if (myid==Host_ID){
      printf("unsupported:\n");
      printf("simultaneuos specification of spin non-collinear\n");
      printf("and orbital optimization\n");
    }

    MPI_Finalize();
    exit(0);
  }

  /************************************************************
    it it not allowed to make the simultaneuos specification
    of NEGF and orbital optimization
  ************************************************************/

  if (Solver==4 && Cnt_switch==1){
    if (myid==Host_ID){
      printf("unsupported:\n");
      printf("simultaneuos specification of NEGF and orbital optimization\n");
    }

    MPI_Finalize();
    exit(0);
  }

  s_vec[0]="Symmetrical"; s_vec[1]="Free"; s_vec[2]="Simple";
  i_vec[0]=1; i_vec[1]=0; i_vec[2]=2;
  input_string2int("orbitalOpt.InitCoes",&SICnt_switch,3,s_vec,i_vec);
  input_int("orbitalOpt.scf.maxIter",&orbitalOpt_SCF,40);
  input_int("orbitalOpt.Opt.maxIter",&orbitalOpt_MD,100);
  input_int("orbitalOpt.per.MDIter",&orbitalOpt_per_MDIter,1000000);
  input_double("orbitalOpt.criterion",&orbitalOpt_criterion,(double)1.0e-4);
  input_double("orbitalOpt.SD.step",&orbitalOpt_SD_step,(double)0.001);
  input_int("orbitalOpt.HistoryPulay",&orbitalOpt_History,20);
  input_int("orbitalOpt.StartPulay",&orbitalOpt_StartPulay,10);
  input_logical("orbitalOpt.Force.Skip" ,&orbitalOpt_Force_Skip,0);

  s_vec[0]="DIIS"; s_vec[1]="EF";
  i_vec[0]=1; i_vec[1]=2;
  input_string2int("orbitalOpt.Opt.Method",&OrbOpt_OptMethod,2,s_vec,i_vec);

  /****************************************************
                  order-N method for SCF
  ****************************************************/

  input_double("orderN.HoppingRanges",&BCR,(double)7.0);
  BCR=BCR/BohrR;

  input_int("orderN.NumHoppings",&NOHS_L,2);
  if (Solver==2 || Solver==3 || Solver==4 || Solver==7 || Solver==9){
    NOHS_L = 1;
    BCR = 1.0;
  }

  NOHS_C = NOHS_L;

  /* start EC */

  input_int("orderN.EC.Sub.Dim",&EC_Sub_Dim,400);

  /* end EC */

  /* start Krylov */

  /* input_int("orderN.Kgrid",&orderN_Kgrid,5); */

  input_int("orderN.KrylovH.order",&KrylovH_order,400);
  input_int("orderN.KrylovS.order",&KrylovS_order,4*KrylovH_order);
  input_logical("orderN.Recalc.Buffer",&recalc_EM,1);
  input_logical("orderN.Exact.Inverse.S",&EKC_Exact_invS_flag,1);
  input_logical("orderN.Expand.Core",&EKC_expand_core_flag,1);
  input_logical("orderN.Inverse.S",&EKC_invS_flag,0); /* ?? */

  if (EKC_Exact_invS_flag==0){
    EKC_invS_flag = 1;
  }
  else {
    EKC_invS_flag = 0;
  }

  orderN_FNAN_SNAN_flag = 0;

  if (fp=input_find("<orderN.FNAN.SNAN")) {

    orderN_FNAN_SNAN_flag = 1;

    for (i=1; i<=atomnum; i++){
      fscanf(fp,"%i %i %i",&j,&orderN_FNAN[i],&orderN_SNAN[i]);

      if (i!=j) {
        if (myid==Host_ID){
          printf("Format error for orderN.FNAN.SNAN\n");
	}

        MPI_Finalize();
        exit(0);
      }
    }

    if (! input_last("orderN.FNAN.SNAN>")) {
      /* format error */

      po++;

      if (myid==Host_ID){
        printf("Format error for orderN.FNAN+SNAN\n");
      }
      MPI_Finalize();
      exit(0);
    }
  }

  /* end Krylov */

  input_int("orderN.RecursiveLevels",&rlmax,10);
  List_YOUSO[3] = rlmax + 3;

  s_vec[0]="NO"; s_vec[1]="SQRT";
  i_vec[0]=1   ; i_vec[1]=2;
  input_string2int("orderN.TerminatorType",&T_switch,2, s_vec,i_vec);

  s_vec[0]="Recursion"; s_vec[1]="Divide";
  i_vec[0]=1; i_vec[1]=2;
  input_string2int("orderN.InverseSolver",&IS_switch,2,s_vec,i_vec);

  input_int("orderN.InvRecursiveLevels",&rlmax_IS,30);
  List_YOUSO[9] = rlmax_IS + 3;

  input_double("orderN.ChargeDeviation",&CN_Error,(double)1.0e-7);
  input_double("orderN.InitChemPot",&ChemP,(double)-3.0); /* in eV */
  ChemP = ChemP/eV2Hartree;

  input_int("orderN.AvNumTerminater",&Av_num,1);
  input_int("orderN.NumPoles",&POLES,10);

  /****************************************************
    control patameters for outputting wave functions
  ****************************************************/

  input_logical("MO.fileout",&MO_fileout,0);
  input_int("num.HOMOs",&num_HOMOs,1);
  input_int("num.LUMOs",&num_LUMOs,1);

  if (MO_fileout==0){
    num_HOMOs = 1;
    num_LUMOs = 1;
  }

  if ((Solver!=2 && Solver!=3 && Solver!=7) && MO_fileout==1){

    s_vec[0]="Recursion";     s_vec[1]="Cluster"; s_vec[2]="Band";
    s_vec[3]="NEGF";          s_vec[4]="DC";      s_vec[5]="GDC";
    s_vec[6]="Cluster-DIIS";  s_vec[7]="Krylov";  s_vec[8]="Cluster2";  
    s_vec[9]="EGAC";          s_vec[10]="DC-LNO";   s_vec[11]="Cluster-LNO";

    printf("MO.fileout=ON is not supported in case of scf.EigenvalueSolver=%s\n",
           s_vec[Solver-1]);  
    printf("MO.fileout is changed to OFF\n");
    MO_fileout = 0;
  }

  if ( (Solver==2 || Solver==3 || Solver==7) && MO_fileout==1 ){
    List_YOUSO[31] = num_HOMOs;
    List_YOUSO[32] = num_LUMOs;
  }
  else{
    List_YOUSO[31] = 1;
    List_YOUSO[32] = 1;
  }

  /* for bulk */  

  input_int("MO.Nkpoint",&MO_Nkpoint,1);
  if (MO_Nkpoint<0){
    printf("MO_Nkpoint must be positive.\n");
    po++;
  }

  if (MO_fileout && Solver==2 && MO_Nkpoint!=1){
    if (myid==Host_ID){
      printf("MO.Nkpoint should be 1 for the cluster calculation.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if ( (Solver==2 || Solver==3 || Solver==7) && MO_fileout==1 ){
    List_YOUSO[33] = MO_Nkpoint;
  }
  else{
    List_YOUSO[33] = 1;
  }

  /* memory allocation */
  Allocate_Arrays(5);

  if (fp=input_find("<MO.kpoint")) {
    for (i=0; i<MO_Nkpoint; i++){
      fscanf(fp,"%lf %lf %lf",&MO_kpoint[i][1],&MO_kpoint[i][2],&MO_kpoint[i][3]);

      if (2<=level_stdout){
        printf("<Input_std>  MO_kpoint %2d  %10.6f  %10.6f  %10.6f\n",
	       i,MO_kpoint[i][1],MO_kpoint[i][2],MO_kpoint[i][3]);
      }
    }
    if (!input_last("MO.kpoint>")) {
      /* format error */
      printf("Format error for MO.kpoint\n");
      po++;
    }

    if (Band_kPathUnit){
      kpoint_changeunit(tv,Band_UnitCell,MO_Nkpoint,MO_kpoint);
    }
  }

  /****************************************************
             Natural Bond Orbital Analysis
  ****************************************************/

  s_vec[0]="off"; s_vec[1]="on1"; s_vec[2]="on2"; s_vec[3]="on3"; s_vec[4]="on4";
  i_vec[0]=0    ; i_vec[1]=1    ; i_vec[2]=2    ; i_vec[3]=3    ; i_vec[4]=4    ;
  input_string2int("NBO.switch", &NBO_switch, 5, s_vec,i_vec);

  if (NBO_switch!=0){

    input_logical("NAO.only",&NAO_only,1);

    input_double("NAO.threshold",&NAO_Occ_or_Ryd,0.85);

    input_int("NBO.Num.CenterAtoms",&Num_NBO_FCenter,1);
    if (Num_NBO_FCenter<0){
      printf("NBO.Num.CenterAtoms must be positive.\n");
      po++;
    }

    if (Num_NBO_FCenter>atomnum){
      printf("NBO ERROR: NBO.Num.CenterAtoms should be less than the total number of atoms.");
      po++;
    }

    Allocate_Arrays(11);

    if (fp=input_find("<NBO.CenterAtoms")) {
      for (i=0; i<Num_NBO_FCenter; i++){
	fscanf(fp,"%d",&NBO_FCenter[i]);
	if (NBO_FCenter[i] <= 0 || NBO_FCenter[i] > atomnum){
	  printf ("NBO ERROR: Index of center atom is wrong.");
	  po++;
	}

	if (0<=level_stdout && myid == Host_ID){
	  printf("<Input_std>  NBO_Center %d : %d \n",i,NBO_FCenter[i]);
	}
      }
    }

    input_logical("NHO.fileout",&NHO_fileout,0);
    input_logical("NBO.fileout",&NBO_fileout,0);

    input_logical("NBO.SmallCell.Switch",&NBO_SmallCell_Switch,0);

    if (fp=input_find("<NBO.SmallCell")) {
      for (i=1; i<=3; i++){
	fscanf(fp,"%lf %lf",&NBO_SmallCellFrac[i][1],&NBO_SmallCellFrac[i][2]);

	if (1<=level_stdout && myid == Host_ID){
	  printf("<Input_std>  NBO_SmallCell %10.6f %10.6f \n",
		 NBO_SmallCellFrac[i][1],NBO_SmallCellFrac[i][2]);
	}
      }

      if (!input_last("NBO.SmallCell>")) {
	printf("Format error for NBO.SmallCell\n");
	po++;
      }
    }

    /*
      input_int("NAO.Nkpoint",&NAO_Nkpoint,1);
      if (NAO_Nkpoint<0){
      printf("NAO_Nkpoint must be positive.\n");
      po++;
      }

      if (fp=input_find("<NAO.kpoint")) {
      for (i=0; i<NAO_Nkpoint; i++){
      fscanf(fp,"%lf %lf %lf",&NAO_kpoint[i][1],&NAO_kpoint[i][2],&NAO_kpoint[i][3]);

      if (0<=level_stdout && myid == Host_ID){
      printf("<Input_std>  NAO_kpoint %2d  %10.6f  %10.6f  %10.6f\n",
      i,NAO_kpoint[i][1],NAO_kpoint[i][2],NAO_kpoint[i][3]);
      }
      }
      if (!input_last("NAO.kpoint>")) {
      printf("Format error for MO.kpoint\n");
      po++;
      }

      if (Band_kPathUnit){
      kpoint_changeunit2(tv,Band_UnitCell,NAO_Nkpoint,NAO_kpoint);
      }

      }
    */

  }

  /* added by Chi-Cheng for unfolding */
  /****************************************************
               Unfolding Band Structure
  ****************************************************/

  input_logical("Unfolding.Electronic.Band",&unfold_electronic_band,0);

  if (unfold_electronic_band==1) {

    unfold_abc=(double**)malloc(sizeof(double*)*3);
    for (i=0; i<3; i++) unfold_abc[i]=(double*)malloc(sizeof(double)*3);
    if (fp=input_find("<Unfolding.ReferenceVectors")) {
      for (i=0; i<3; i++){
        fscanf(fp,"%lf %lf %lf",&unfold_abc[i][0],&unfold_abc[i][1],&unfold_abc[i][2]);
      }
      if (!input_last("Unfolding.ReferenceVectors>")) {
        /* format error */
        printf("Format error in Unfolding.ReferenceVectors\n");
        po++;
      }
      /* Ang to AU */
      if (unitvector_unit==0) {
        for (i=0; i<3; i++){
          unfold_abc[i][0] = unfold_abc[i][0]/BohrR;
          unfold_abc[i][1] = unfold_abc[i][1]/BohrR;
          unfold_abc[i][2] = unfold_abc[i][2]/BohrR;
        }
      }
    }
    else {
      for (i=0; i<3; i++) {
        unfold_abc[i][0] = tv[i+1][1];
        unfold_abc[i][1] = tv[i+1][2];
        unfold_abc[i][2] = tv[i+1][3];
      }
    }

    unfold_origin=(double*)malloc(sizeof(double)*3);
    if (fp=input_find("<Unfolding.Referenceorigin")) {
        fscanf(fp,"%lf %lf %lf",&unfold_origin[0],&unfold_origin[1],&unfold_origin[2]);
      if (!input_last("Unfolding.Referenceorigin>")) {
        /* format error */
        printf("Format error in Unfolding.Referenceorigin\n");
        po++;
      }
      /* Ang to AU */
      if (unitvector_unit==0) {
        for (i=0; i<3; i++) unfold_origin[i] = unfold_origin[i]/BohrR;
      }
    }
    else {
      /* default value will be calculated later on */
      unfold_origin[0]=-0.9999900023114;
      unfold_origin[1]=-0.9999900047614;
      unfold_origin[2]=-0.9999900058914;
    }

    input_double("Unfolding.LowerBound",&unfold_lbound,-10.); 
    input_double("Unfolding.UpperBound",&unfold_ubound,10.);  /* in eV */
    if (unfold_lbound>=unfold_ubound){
      printf("Unfolding: Lower bound must be lower than the upper bound.\n");
      po++;
    }
    unfold_lbound=unfold_lbound/eV2Hartree;
    unfold_ubound=unfold_ubound/eV2Hartree;

    unfold_mapN2n=(int*)malloc(sizeof(int)*atomnum);
    if (fp=input_find("<Unfolding.Map")) {
      for (i=0; i<atomnum; i++){
        fscanf(fp,"%i %i",&j,&unfold_mapN2n[i]);
        if ((j!=i+1)||(unfold_mapN2n[i]<0)) { printf("Format error in Unfolding.Map! (Values cannot be negative.)\n"); po++;}
      }
      if (!input_last("Unfolding.Map>")) {
      /* format error */
      printf("Format error in Unfolding.Map\n");
      po++;
      }
    }
    else {
      for (i=0; i<atomnum; i++) unfold_mapN2n[i]=i;
    }

    if (Solver!=2 && Solver!=3){

      s_vec[0]="Recursion";     s_vec[1]="Cluster"; s_vec[2]="Band";
      s_vec[3]="NEGF";          s_vec[4]="DC";      s_vec[5]="GDC";
      s_vec[6]="Cluster-DIIS";  s_vec[7]="Kryloqv";  s_vec[8]="Cluster2";  
      s_vec[9]="EGAC";          s_vec[10]="DC-LNO";   s_vec[11]="Cluster-LNO";

      printf("Unfolding.fileout=ON is not supported in case of scf.EigenvalueSolver=%s\n",
             s_vec[Solver-1]);
      printf("Unfolding.fileout is changed to OFF\n");
      unfold_electronic_band = 0;
    }

    /* for bulk */
    
    input_int("Unfolding.Nkpoint",&unfold_Nkpoint,0);
    if (unfold_Nkpoint<0){
      printf("Unfolding.Nkpoint must be positive.\n");
      po++;
    }
    
    input_int("Unfolding.desired_totalnkpt",&unfold_nkpts,0);
    if (unfold_nkpts<0){
      printf("Unfolding.desired_totalnkpt must be positive.\n");
      po++;
    }
    
    /* memory allocation */
    unfold_kpoint = (double**)malloc(sizeof(double*)*(unfold_Nkpoint+1));
    for (i=0; i<(unfold_Nkpoint+1); i++){
      unfold_kpoint[i] = (double*)malloc(sizeof(double)*4);
    }

    unfold_kpoint_name = (char**)malloc(sizeof(char*)*(unfold_Nkpoint+1));
    for (i=0; i<(unfold_Nkpoint+1); i++){
      unfold_kpoint_name[i] = (char*)malloc(sizeof(char)*YOUSO10);
    }

    if (fp=input_find("<Unfolding.kpoint")) {
      for (i=0; i<unfold_Nkpoint; i++){
        fscanf(fp,"%s %lf %lf %lf",
               unfold_kpoint_name[i],
               &unfold_kpoint[i][1],
               &unfold_kpoint[i][2],
               &unfold_kpoint[i][3]);

        if (2<=level_stdout){
          printf("<Input_std>  Unfolding.kpoint %2d  %10.6f  %10.6f  %10.6f\n",
                 i,unfold_kpoint[i][1],unfold_kpoint[i][2],unfold_kpoint[i][3]);
        }
      }
      if (!input_last("Unfolding.kpoint>")) {
        /* format error */
        printf("Format error for Unfolding.kpoint\n");
        po++;
      }
    }
  }
  /* end unfolding */

  /**********************************************************
   control of occupation for orbitals
   The check for the input parameters will be done 
   in SetPara_DFT.c.

   In the scheme, the empty states are explicitly defined by
   orbital indecies.
   *********************************************************/

  input_logical("empty.occupation.flag",&empty_occupation_flag,0); /* default=off */
  
  if (empty_occupation_flag==1){
    input_int("empty.occupation.num", &empty_occupation_num,0);
    empty_occupation_spin = (int*)malloc(sizeof(int)*empty_occupation_num);
    empty_occupation_orbital = (int*)malloc(sizeof(int)*empty_occupation_num);

    if (fp=input_find("<empty.occupation.orbitals")) {

       for (i=0; i<empty_occupation_num; i++){
         fscanf(fp,"%d %d",&empty_occupation_spin[i],&empty_occupation_orbital[i]);
       }
    
      if ( ! input_last("empty.occupation.orbitals>") ) {
	/* format error */
	printf("Format error for empty.occupation.orbitals\n");
	po++;
      }
    }
  }

  /***********************************************************
   specification of unoccupied orbitals:
   a constraint method for occupation

   In the scheme, the empty states are automatically searched 
   by monitoring populations of targeted orbitals in an atom.
  ***********************************************************/

  input_logical("empty.states.flag",&empty_states_flag,0); /* default=off */

  if (empty_occupation_flag==1 && empty_states_flag==1){

    if (myid==Host_ID){
      printf("empty.occupation.flag and empty.states.flag are mutually exclusive. Check your input file.\n");
      MPI_Finalize();
      exit(0);
    }
  }

  if (empty_states_flag==1){

    char empty_states_orbitals[YOUSO10]; 
    char c0,c1,cstr[YOUSO10*3];
    int num,l0;

    if (fp=input_find("<empty.states.orbitals")) {

      fscanf(fp,"%d %s %s", &empty_states_atom, Species, empty_states_orbitals) ;
  
      if ( ! input_last("empty.states.orbitals>") ) {
	/* format error */
        if (myid==Host_ID){
  	  printf("Format error for empty.states.orbitals\n");
	}
	po++;
      }
    }
    
    i = Species2int(Species);

    if (i!=WhatSpecies[empty_states_atom]){
      if (myid==Host_ID){
	printf("Species in empty.states.orbitals is not consistent with Atoms.SpeciesAndCoordinates.\n");

        MPI_Finalize();
        exit(0);
      }
    }

    c0 = empty_states_orbitals[0];
    c1 = empty_states_orbitals[1];

    if      (c0=='s') l0 = 0;
    else if (c0=='p') l0 = 1;
    else if (c0=='d') l0 = 2;
    else if (c0=='f') l0 = 3;
    else if (c0=='g') l0 = 4;
    else {
      if (myid==Host_ID){
	printf("Wrong orbital in empty.states.orbitals (1).\n");
        MPI_Finalize();
        exit(0);
      }
    }

    cstr[0] = c1;
    cstr[1] = '\0';
    j = atoi(cstr);

    if (Spe_Num_Basis[i][l0]<j){
      if (myid==Host_ID){
	printf("Wrong orbital in empty.states.orbitals (2).\n");
        MPI_Finalize();
        exit(0);
      }
    }

    num = 0;
    for (l=0; l<l0; l++){
      num += (2*l+1)*Spe_Num_Basis[i][l];
    }

    num += (2*l0+1)*(j-1);
    empty_states_orbitals_sidx = num;
    empty_states_orbitals_num = (2*l0+1);
  }

  /****************************************************
                  OutData_bin_flag
  ****************************************************/

  input_logical("OutData.bin.flag",&OutData_bin_flag,0); /* default=off */

  /****************************************************
           for output of contracted orbitals    
  ****************************************************/

  input_logical("CntOrb.fileout",&CntOrb_fileout,0);
  if ((!(Cnt_switch==1 && RCnt_switch==1)) && CntOrb_fileout==1){
    printf("CntOrb.fileout=on is valid in case of orbitalOpt.Method=Restricted or Speciese\n");
    po++;    
  }

  if (CntOrb_fileout==1){
    input_int("Num.CntOrb.Atoms",&Num_CntOrb_Atoms,1);
    CntOrb_Atoms = (int*)malloc(sizeof(int)*Num_CntOrb_Atoms);      

    if (fp=input_find("<Atoms.Cont.Orbitals")) {
      for (i=0; i<Num_CntOrb_Atoms; i++){
        fscanf(fp,"%i",&CntOrb_Atoms[i]);
        if (CntOrb_Atoms[i]<1 || atomnum<CntOrb_Atoms[i]){
          printf("Invalid atom in <Atoms.Cont.Orbitals\n" ); 
          po++;
        }
      }

      if (!input_last("Atoms.Cont.Orbitals>")) {
        /* format error */
        printf("Format error for Atoms.Cont.Orbitals\n");
        po++;
      }
    }
  }

  /****************************************************
                 external electric field
  ****************************************************/

  r_vec[0]=0.0; r_vec[1]=0.0; r_vec[2]=0.0;
  input_doublev("scf.Electric.Field",3,E_Field,r_vec);

  if ( 1.0e-50<fabs(E_Field[0]) || 1.0e-50<fabs(E_Field[1]) || 1.0e-50<fabs(E_Field[2]) ){

    E_Field_switch = 1;    

    /*******************************************
                 unit transformation
         
        V/m = J/C/m
            = 1/(4.35975*10^{-18}) Hatree
	      *(1/(1.602177*10^{-19}) e )^{-1}
              *(1/(0.5291772*10^{-10}) a0 )^{-1}
            = 0.1944688 * 10^{-11} Hartree/e/a0

       input unit:  GV/m = 10^9 V/m
       used unit:   Hartree/e/a0

       GV/m = 0.1944688 * 10^{-2} Hartree/e/a0 
    *******************************************/

    length = sqrt( Dot_Product(tv[1], tv[1]) );
    x = E_Field[0]*tv[1][1]/length;
    y = E_Field[0]*tv[1][2]/length;
    z = E_Field[0]*tv[1][3]/length;

    length = sqrt( Dot_Product(tv[2], tv[2]) );
    x += E_Field[1]*tv[2][1]/length;
    y += E_Field[1]*tv[2][2]/length;
    z += E_Field[1]*tv[2][3]/length;

    length = sqrt( Dot_Product(tv[3], tv[3]) );
    x += E_Field[2]*tv[3][1]/length;
    y += E_Field[2]*tv[3][2]/length;
    z += E_Field[2]*tv[3][3]/length;

    length = sqrt( x*x + y*y + z*z );
    x = x/length;
    y = y/length;
    z = z/length;

    if (myid==Host_ID && 0<level_stdout){  
      printf("\n");
      printf("<Applied External Electric Field>\n");
      printf("  direction (x, y, z)  = %10.5f %10.5f %10.5f\n",x,y,z);
      printf("  magnitude (10^9 V/m) = %10.5f\n\n",length);
    }

    E_Field[0] = 0.1944688*0.01*E_Field[0];
    E_Field[1] = 0.1944688*0.01*E_Field[1];
    E_Field[2] = 0.1944688*0.01*E_Field[2];
  }
  else {
    E_Field_switch = 0;    
  }

  /****************************************************
                      DOS, PDOS
  ****************************************************/

  input_logical("Dos.fileout",&Dos_fileout,0);
  input_logical("DosGauss.fileout",&DosGauss_fileout,0);
  input_int("DosGauss.Num.Mesh",&DosGauss_Num_Mesh,200);
  input_double("DosGauss.Width",&DosGauss_Width,0.2);   /* in eV */
  input_logical("FermiSurfer.fileout",&fermisurfer_output,0);

  /* change the unit from eV to Hartree */
  DosGauss_Width = DosGauss_Width/eV2Hartree;

  if (Dos_fileout && DosGauss_fileout){

    if (myid==Host_ID){
      printf("Dos.fileout and DosGauss.fileout are mutually exclusive.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if ( DosGauss_fileout && (Solver!=3 && PeriodicGamma_flag!=1) ){  /* band */

    if (myid==Host_ID){
      printf("DosGauss.fileout is supported for only band calculation.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  r_vec[0]=-20.0; r_vec[1]=20.0;
  input_doublev("Dos.Erange",2,Dos_Erange,r_vec);
  /* change the unit from eV to Hartree */
  Dos_Erange[0]= Dos_Erange[0]/eV2Hartree;
  Dos_Erange[1]= Dos_Erange[1]/eV2Hartree;

  i_vec[0]=Kspace_grid1; i_vec[1]=Kspace_grid2, i_vec[2]=Kspace_grid3;
  input_intv("Dos.Kgrid",3,Dos_Kgrid,i_vec);

  /**********************************************************
   calculation of partial charge for scanning tunneling 
   microscopy (STM) simulations by the Tersoff-Hamann scheme
  ***********************************************************/

  input_logical("partial.charge",&cal_partial_charge,0); /* default=off */
   
  if (cal_partial_charge && (Dos_fileout==0 && DosGauss_fileout==0)){
    if (myid==Host_ID){
      printf("partial.charge can be switched on in case that\n");
      printf("either Dos.fileout or DosGauss.fileout is switched on.\n");

    }
    MPI_Finalize();
    exit(0);
  }

  if ( cal_partial_charge && (Solver!=2 && Solver!=3) ){
    if (myid==Host_ID){
      printf("The calculation of partial charge is supported only for the Cluster and Band methods.\n");

    }
    MPI_Finalize();
    exit(0);
  }

  input_double("partial.charge.energy.window",&ene_win_partial_charge,0.0); /* in eV */
  ene_win_partial_charge /= eV2Hartree;  /* eV to Hartee */  

  /****************************************************
   write a binary file, filename.scfout, which includes
   Hamiltonian, overlap, and density matrices.
  ****************************************************/
 
  input_logical("HS.fileout",&HS_fileout,0);

  /****************************************************
                   Energy decomposition
  ****************************************************/

  input_logical("Energy.Decomposition",&Energy_Decomposition_flag,0);

  if (Energy_Decomposition_flag==1 && Cnt_switch==1){
    if (myid==Host_ID){
      printf("Energy decomposition is not supported for orbital optimization.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /****************************************************
                   Voronoi charge
  ****************************************************/

  input_logical("Voronoi.charge",&Voronoi_Charge_flag,0);

  /****************************************************
                 Voronoi orbital moment
  ****************************************************/

  input_logical("Voronoi.orbital.moment",&Voronoi_OrbM_flag,0);

  /****************************************************
       parameters on Wannier funtions by hmweng
  ****************************************************/

  input_logical("Wannier.Func.Calc",&Wannier_Func_Calc,0); /* default=off */
  input_logical("Wannier90.fileout",&Wannier90_fileout,0); /* default=off */

  if (Wannier_Func_Calc){

    int **tmp_Wannier_Pro_SelMat;
    double ***tmp_Wannier_Projector_Hybridize_Matrix;

    input_int("Wannier.Func.Num", &Wannier_Func_Num,0);
    input_double("Wannier.Outer.Window.Bottom",&Wannier_Outer_Window_Bottom,(double)-10.0); /* eV */
    input_double("Wannier.Outer.Window.Top",   &Wannier_Outer_Window_Top,(double)10.0);  /* eV */

    input_double("Wannier.Inner.Window.Bottom",&Wannier_Inner_Window_Bottom,(double)0.0);   /* eV */
    input_double("Wannier.Inner.Window.Top",   &Wannier_Inner_Window_Top,(double)0.0);   /* eV */
    input_logical("Wannier.Initial.Guess",   &Wannier_Initial_Guess,1);         /* on|off */

    if( (Wannier_Outer_Window_Top-Wannier_Outer_Window_Bottom)<=0.0 ){      

      if (myid==Host_ID){
        printf("Error:WF  The top of the OUTER window should be higher than the bottom.\n");
      }

      MPI_Finalize();
      exit(0);
    }

    if( (Wannier_Inner_Window_Top-Wannier_Inner_Window_Bottom)<0.0){

      if (myid==Host_ID){
        printf("Error:WF  The top of the INNER window should be higher than the bottom.\n");
      }

      MPI_Finalize();
      exit(0);
    }

    if ( 0.0<(Wannier_Inner_Window_Top-Wannier_Inner_Window_Bottom) ){

      if( Wannier_Inner_Window_Bottom<Wannier_Outer_Window_Bottom 
	  || Wannier_Inner_Window_Top>Wannier_Outer_Window_Top ){

	if (myid==Host_ID){
	  printf("Error:WF  INNER window (%10.5f,%10.5f) is not inside of OUTER window (%10.5f,%10.5f).\n",
		 Wannier_Inner_Window_Bottom,Wannier_Inner_Window_Top,
		 Wannier_Outer_Window_Bottom,Wannier_Outer_Window_Top);
	}

	MPI_Finalize();
	exit(0);
      }
    }


    if(fabs(Wannier_Inner_Window_Bottom-Wannier_Inner_Window_Top)<1e-6){

      if (myid==Host_ID){
        printf("Message:WF  The top and bottom of INNER window is the same.\n");
        printf("Message:WF  We assume that you DO NOT want to use an inner window and no states will be frozen.\n");
      }

      Wannier_Inner_Window_Top=Wannier_Inner_Window_Bottom;
    }

    s_vec[0]="Ang";  s_vec[1]="AU";   s_vec[2]="FRAC";
    i_vec[0]= 0;     i_vec[1]= 1;     i_vec[2]= 2;
    input_string2int("Wannier.Initial.Projectors.Unit",
		     &Wannier_unit,3,s_vec,i_vec);


    if (fp=input_find("<Wannier.Initial.Projectors")) {

      {
	int po,num_lines;
	char ctmp[200];
	double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9;
	po = 0;
	num_lines = 0; 
	do{
	  fscanf(fp,"%s %lf %lf %lf  %lf %lf %lf  %lf %lf %lf",
		 ctmp,&tmp1,&tmp2,&tmp3,&tmp4,&tmp5,&tmp6,&tmp7,&tmp8,&tmp9);


	  if (strcmp(ctmp,"Wannier.Initial.Projectors>")==0){ 
	    po = 1;
	  }
	  else{ 
	    num_lines++; 
	  }

	} while (po==0 && num_lines<10000);

	if (strcmp(ctmp,"Wannier.Initial.Projectors>")!=0){ 
	  if (myid==Host_ID){
	    printf("Format error for Wannier.Initial.Projectors\n");
	  }
	  MPI_Finalize();
	  exit(0);
	}

	Wannier_Num_Kinds_Projectors = num_lines;     
      }
    }

    MPI_Barrier(mpi_comm_level1);

    Allocate_Arrays(8);  

    /* allocation of temporal arrays */
    tmp_Wannier_Pro_SelMat=(int**)malloc(sizeof(int*)*Wannier_Num_Kinds_Projectors);
    for(i=0; i<Wannier_Num_Kinds_Projectors; i++){
      tmp_Wannier_Pro_SelMat[i]=(int*)malloc(sizeof(int)*Max_Num_WF_Projs);
      for(j=0;j<Max_Num_WF_Projs;j++){
	tmp_Wannier_Pro_SelMat[i][j]=-1;
      } 
    }
    tmp_Wannier_Projector_Hybridize_Matrix=(double***)malloc(sizeof(double**)*Wannier_Num_Kinds_Projectors);
    for(i=0; i<Wannier_Num_Kinds_Projectors; i++){
      tmp_Wannier_Projector_Hybridize_Matrix[i]=(double**)malloc(sizeof(double*)*Max_Num_WF_Projs);
      for(j=0;j<Max_Num_WF_Projs;j++){
	tmp_Wannier_Projector_Hybridize_Matrix[i][j]=(double*)malloc(sizeof(double)*Max_Num_WF_Projs);
      }
    }

    num_wannier_total_projectors = 0;
   
    if (fp=input_find("<Wannier.Initial.Projectors")) {

      {
	char ctmp[YOUSO10];

	for (i=0; i<Wannier_Num_Kinds_Projectors; i++){

	  fscanf(fp,"%s %lf %lf %lf  %lf %lf %lf  %lf %lf %lf",
		 ctmp,
		 &Wannier_Pos[i][1],
		 &Wannier_Pos[i][2],
		 &Wannier_Pos[i][3],
        
		 &Wannier_Z_Direction[i][1], 
		 &Wannier_Z_Direction[i][2], 
		 &Wannier_Z_Direction[i][3], 
		 &Wannier_X_Direction[i][1], 
		 &Wannier_X_Direction[i][2], 
		 &Wannier_X_Direction[i][3]);

	  num_wannier_total_projectors += Analyze_Wannier_Projectors(i,ctmp,tmp_Wannier_Pro_SelMat,tmp_Wannier_Projector_Hybridize_Matrix);
 
	  if (Wannier_unit==2){

	    /* adjust position of Wannier funtion within 0 to 1 */ 

	    for (k=1; k<=3; k++){

	      /*adjust it to -0.5 to 0.5 */

              if (2<=level_stdout) printf(" %10.5f -> ",Wannier_Pos[i][k]);        

	      itmp = (int)Wannier_Pos[i][k];
	      if (1.0<Wannier_Pos[i][k]){
		Wannier_Pos[i][k] = Wannier_Pos[i][k] - (double)itmp;
	      }
	      else if (Wannier_Pos[i][k]<0.0){
		Wannier_Pos[i][k] = Wannier_Pos[i][k] + (double)(abs(itmp)+1);
	      }

              if (2<=level_stdout)  printf(" %10.5f  ",Wannier_Pos[i][k]);
	    }
          
	    /* calculate Cartesian coordiantes */ 

	    x = Wannier_Pos[i][1]*tv[1][1] + Wannier_Pos[i][2]*tv[2][1] + Wannier_Pos[i][3]*tv[3][1];
	    y = Wannier_Pos[i][1]*tv[1][2] + Wannier_Pos[i][2]*tv[2][2] + Wannier_Pos[i][3]*tv[3][2];
	    z = Wannier_Pos[i][1]*tv[1][3] + Wannier_Pos[i][2]*tv[2][3] + Wannier_Pos[i][3]*tv[3][3];

	    Wannier_Pos[i][1] = x;
	    Wannier_Pos[i][2] = y;
	    Wannier_Pos[i][3] = z;

	  }

	  if(Wannier_unit==0){ 

	    /* transfer the unit to AU since all the atomic site are experessed in AU */

	    Wannier_Pos[i][1]=Wannier_Pos[i][1]/BohrR;
	    Wannier_Pos[i][2]=Wannier_Pos[i][2]/BohrR;
	    Wannier_Pos[i][3]=Wannier_Pos[i][3]/BohrR;
	  }

          if (myid==Host_ID){
	    printf("Projector Center(Ang.): (%10.5f, %10.5f, %10.5f)\n",
                    Wannier_Pos[i][1]*BohrR,
                    Wannier_Pos[i][2]*BohrR,
                    Wannier_Pos[i][3]*BohrR);
	  }

	  /* hmweng */        
       
	  Get_Euler_Rotation_Angle(Wannier_Z_Direction[i][1],
                                   Wannier_Z_Direction[i][2],
                                   Wannier_Z_Direction[i][3],
				   Wannier_X_Direction[i][1],
                                   Wannier_X_Direction[i][2],
                                   Wannier_X_Direction[i][3],
				   &Wannier_Euler_Rotation_Angle[i][0],
				   &Wannier_Euler_Rotation_Angle[i][1],
				   &Wannier_Euler_Rotation_Angle[i][2]);

          /* for each kind of Projector, find rotation matrix to the new coordinate system */

	  for(k=1; k<=3; k++){  /* L=0 is not included since s orbital is rotationally invariant */
 
            double tmpRotMat[7][7];
            int jmp, jm; 

	    if(Wannier_NumL_Pro[i][k]!=0){  /* if this l component is involved */

	      Get_Rotational_Matrix(Wannier_Euler_Rotation_Angle[i][0],
				    Wannier_Euler_Rotation_Angle[i][1],
				    Wannier_Euler_Rotation_Angle[i][2],
				    k, tmpRotMat);

	      for(jmp=0;jmp<2*k+1;jmp++){
		for(jm=0;jm<2*k+1;jm++){

		  Wannier_RotMat_for_Real_Func[i][k][jmp][jm]=tmpRotMat[jmp][jm]; 
		}
	      } 
	    }
	  }

	}
      }
    }

    if (num_wannier_total_projectors!=Wannier_Func_Num && Wannier_Initial_Guess==1){

      if (myid==Host_ID){
	printf("Error:WF  The number (%d) of projectors is not same as that (%d) of Wannier functions\n",
	       num_wannier_total_projectors,Wannier_Func_Num);
      }

      MPI_Finalize();
      exit(0);
    }
  
    { /* for each kind of projectors, find the selection matrix which can tell the position of
	 projectors among the envolved basis set.
	 For example,  
	 if projector(s) is(are) definded as dxy, which means the envolved basis set is local d
	 orbitals, the projector is dxy which is in the third one arranged in the following order:
	 dz2 --> 0, dx2-y2 --> 1, dxy -->2, dxz --> 3, dyz -->4.
	 if projectors are defined as sp2, which means the envolved basis set are local s and p
	 orbitals, the used local orbitals are s, px, py, this seletion matrix will have the values
	 like:
	 s --> 0, px -->1, py --> 2  (pz is in the basis set, but will not be selected in
	 the final Amn matrix)
      */

      Allocate_Arrays(9);

      j=0;

      for(i=0; i<Wannier_Num_Kinds_Projectors; i++){

	for(k=0; k<Wannier_Num_Pro[i]; k++){
	  Wannier_Select_Matrix[i][k] = tmp_Wannier_Pro_SelMat[i][k];
	}

	for(k=0;k<Wannier_Num_Pro[i];k++){
	  for(l=0;l<Wannier_Num_Pro[i];l++){
	    Wannier_Projector_Hybridize_Matrix[i][k][l] = tmp_Wannier_Projector_Hybridize_Matrix[i][k][l];
	  }
	}
      

	for(k=0;k<Wannier_Num_Pro[i];k++){
	  Wannier_Guide[0][j]=Wannier_Pos[i][1];
	  Wannier_Guide[1][j]=Wannier_Pos[i][2];   
	  Wannier_Guide[2][j]=Wannier_Pos[i][3];
	  j++;
	}

      } /*for each kind of projector*/
    }

    i_vec2[0]=4;
    i_vec2[1]=4;
    i_vec2[2]=4;
    input_intv("Wannier.Kgrid",3,i_vec,i_vec2);
    Wannier_grid1 = i_vec[0];
    Wannier_grid2 = i_vec[1];
    Wannier_grid3 = i_vec[2];

    input_int("Wannier.MaxShells", &Wannier_MaxShells,12);
    input_int("Wannier.Minimizing.Max.Steps", &Wannier_Minimizing_Max_Steps,200);

    input_logical("Wannier.Interpolated.Bands", &Wannier_Draw_Int_Bands,0);

    if (Wannier_Draw_Int_Bands){
      if (!Band_disp_switch){

        if (myid==Host_ID){
  	  printf("Error:WF You are requiring for plotting the interpolated bands but without turn on key word Band.dispersion.\n");
	}
        MPI_Finalize();
	exit(0); 

      }
    } 

    input_logical("Wannier.Function.Plot", &Wannier_Draw_MLWF,0);

    i_vec2[0]=0;  i_vec2[1]=0;  i_vec2[2]=0;
    input_intv("Wannier.Function.Plot.SuperCells",3,Wannier_Plot_SuperCells,i_vec2);

    if (     Wannier_grid1<Wannier_Plot_SuperCells[0]
          || Wannier_grid2<Wannier_Plot_SuperCells[1]
	  || Wannier_grid3<Wannier_Plot_SuperCells[2]){

        if (myid==Host_ID){
  	  printf("Error:WF For each component, the following condition should be satisfied\n");
  	  printf("      Wannier.Function.Plot.SuperCells<=Wannier.Kgrid\n");

	}
        MPI_Finalize();
	exit(0); 
    }

    input_double("Wannier.Dis.Mixing.Para",&Wannier_Dis_Mixing_Para,(double)0.5);
    input_double("Wannier.Dis.Conv.Criterion",&Wannier_Dis_Conv_Criterion,(double)1e-8); 
    input_int("Wannier.Dis.SCF.Max.Steps",&Wannier_Dis_SCF_Max_Steps, 200);
    input_int("Wannier.Minimizing.Scheme",&Wannier_Min_Scheme, 0);
    input_double("Wannier.Minimizing.StepLength",&Wannier_Min_StepLength,(double)2.0); 
    input_int("Wannier.Minimizing.Secant.Steps",&Wannier_Min_Secant_Steps, 5);
    input_double("Wannier.Minimizing.Secant.StepLength",&Wannier_Min_Secant_StepLength, (double)2.0);
    input_double("Wannier.Minimizing.Conv.Criterion",&Wannier_Min_Conv_Criterion, (double)1e-8);
    input_logical("Wannier.Output.kmesh",&Wannier_Output_kmesh, 0);

    input_logical("Wannier.Readin.Overlap.Matrix",&Wannier_Readin_Overlap_Matrix,0);

    if (Wannier_Readin_Overlap_Matrix==1){
      sprintf(file_check,"%s%s.mmn",filepath,filename);
      if ((fp_check = fopen(file_check,"r")) != NULL){
	fclose(fp_check);
      }    
      else {
	if (myid==Host_ID){  
	  printf("\nError:WF  not found %s\n",file_check);
	}

	MPI_Finalize();
	exit(0);
      }
    }

    if (Wannier_Readin_Overlap_Matrix==0) 
      Wannier_Output_Overlap_Matrix = 1;
    else  
      Wannier_Output_Overlap_Matrix = 0;

    Wannier_Readin_Projection_Matrix = 0;

    if (Wannier_Initial_Guess==1)
      Wannier_Output_Projection_Matrix = 1;
    else 
      Wannier_Output_Projection_Matrix = 0;

    if(Wannier_Readin_Overlap_Matrix==1) DFTSCF_loop=1;

    /* freeing of temporal arrays */

    for(i=0; i<Wannier_Num_Kinds_Projectors; i++){
      free(tmp_Wannier_Pro_SelMat[i]);
    }
    free(tmp_Wannier_Pro_SelMat);

    for(i=0; i<Wannier_Num_Kinds_Projectors; i++){
      free(tmp_Wannier_Projector_Hybridize_Matrix[i]);
    }
    free(tmp_Wannier_Projector_Hybridize_Matrix);

  } /* if(Wannier_Func_Calc) */

  /****************************************************
           parameters for cell optimizaiton 
  ****************************************************/

  /* scf.stress.tensor */

  input_logical("scf.stress.tensor",&scf_stress_flag,0); /* default=off */

  /* applied pressure */
  
  input_double("MD.applied.pressure",&MD_applied_pressure,(double)0.0); /* GPa */

  i_vec[0]=1; i_vec[1]=1, i_vec[2]=1;
  input_intv("MD.applied.pressure.flag",3,MD_applied_pressure_flag,i_vec);

  /****************************************************
     set flags for calculations of stress tensor
     scf_stress_flag: default=off
     MD_cellopt_flag: default=off 
  ****************************************************/

  /* In case of cell optimization,  */

  MD_cellopt_flag = 0;    

  if (   MD_switch==17 
      || MD_switch==18 
     )
  {
    scf_stress_flag = 1;
    MD_cellopt_flag = 1;
  }

  /* add by MIZUHO for NPT-MD */
  /* In case of NPT-MD, calc stress and truncation */
  if( MD_switch==27 || MD_switch==28 || MD_switch==29 || MD_switch==30 ){
    scf_stress_flag = 1;
    MD_cellopt_flag = 1;
  }

  /* check the compatibility with other functionlities */

  if (scf_stress_flag==1 && Cnt_switch==1){
    if (myid==Host_ID){
      printf("Orbital optimization is not compatible with stress calculation.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (scf_stress_flag==1 && Hub_U_switch==1 && Hub_U_occupation==1){
    if (myid==Host_ID){
      printf("\n'full' local propjector in DFT+U is not compatible with stress calculation.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  if (scf_stress_flag==1 && SpinP_switch==3){
    if (myid==Host_ID){
      printf("\n'Non-collinear calculation is not compatible with stress calculation.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  /******************************************************
     Population analysis by atomic orbital resembling 
     Wannier funtions with maximum occupation
  ******************************************************/

  input_logical("Population.Analysis.AO.Wanniers",&pop_anal_aow_flag,0); /* default=off */

  /****************************************************
                       input_close
  ****************************************************/

  input_close();

  if (po>0 || input_errorCount()>0) {
    printf("errors in the inputfile\n");
    MPI_Finalize();
    exit(0);
  } 

  /****************************************************
           adjustment of atomic position
       Atomic positions are not adjusted in Ver. 3.6
  ****************************************************/

  /*  if (Solver!=4) Set_In_First_Cell();  */

  /****************************************************
                 Gxyz -> His_Gxyz[0]
  ****************************************************/

  k = 0;
  for (i=1; i<=atomnum; i++){
    His_Gxyz[0][k] = Gxyz[i][1]; k++;       
    His_Gxyz[0][k] = Gxyz[i][2]; k++;       
    His_Gxyz[0][k] = Gxyz[i][3]; k++;       
  }
  
  /****************************************************
          generate Monkhorst-Pack k-points 
  ****************************************************/

  if (Solver==3 && way_of_kpoint==2){

    Generating_MP_Special_Kpt(/* input */
			      atomnum, SpeciesNum,
			      tv, Gxyz,
			      InitN_USpin, InitN_DSpin,
			      Criterion_MP_Special_Kpt,
			      SpinP_switch,
			      WhatSpecies, 
			      Kspace_grid1, Kspace_grid2, Kspace_grid3
			      /* implicit output 
			      num_non_eq_kpt,
			      NE_KGrids1, NE_KGrids2, NE_KGrids3,
			      NE_T_k_op */ );
  }

  /**************************************************************
   Effective Screening Medium (ESM) method Calculation 
                                     added by T.Ohwaki     
  ***************************************************************/

#if 0

  if (ESM_switch!=0){

    double xb0,xb1;

    xb0 = 0.0;
    xb1 = tv[1][1];

    /* check whether the wall position does not overlap with atoms. */
    
    for (i=1; i<=atomnum; i++){

      if ( ESM_wall_switch==1){
	if ( Gxyz[i][1]>(xb1-ESM_wall_position)){

	  if (myid==Host_ID){
	    printf("<ESM:Warning>\n");
	    printf("The coordinate of a-axis for atom %d = %16.9f \n",i,Gxyz[i][1]);
	    printf("The coordinate of the wall position = %16.9f \n",xb1-ESM_wall_position);
	    printf("The axis of the unit cell is too short.\n");
	    printf("The wall position overlaps with atoms.\n");
	    printf("Please increase the length of the a-axis.\n");
	  }

	  MPI_Finalize();
	  exit(0);
	}
      }

      /* check the size of the upper vaccum */ 

      if ( Gxyz[i][1]>(xb1-ESM_buffer_range)){

        if (myid==Host_ID){
          printf("<ESM:Warning>\n");
          printf("The coordinate of a-axis for atom %d = %16.9f \n",i,Gxyz[i][1]);
          printf("The coordinate of the upper vacuum edge = %16.9f \n",xb1-ESM_buffer_range);
          printf("The region of the upper vaccum along the a-axis is too short.\n");
          printf("Please increase the length of the a-axis.\n");
        }

        MPI_Finalize();
        exit(0);
      }

      /* check the size of the lower vaccum */ 

      if ( Gxyz[i][1]<(xb0+ESM_buffer_range)){

        if (myid==Host_ID){
          printf("<ESM:Warning>\n");
          printf("The coordinate of a-axis for atom %d = %16.9f \n",i,Gxyz[i][1]);
          printf("The coordinate of the lower vacuum edge = %16.9f \n",xb0+ESM_buffer_range);
          printf("The region of the lower vaccum along the a-axis is too short.\n");
          printf("Please increase the length of the a-axis.\n");
        }

        MPI_Finalize();
        exit(0);
      }

    }
  }

#endif

  /****************************************************
                   print out to std
  ****************************************************/

  if (myid==Host_ID && 0<level_stdout){  
    printf("\n\n<Input_std>  Your input file was normally read.\n");
    printf("<Input_std>  The system includes %i species and %i atoms.\n",
            real_SpeciesNum,atomnum);
  }

}


void Remake_RestartFile(int numprocs_new, int numprocs_old, int N1, int N2, int N3, int SpinFlag_old)
{
  int myid,ID,N2D,n2Ds_new,n2De_new;
  int n2Ds_old,n2De_old,num,spin,i,n,n3,m;
  int IDs_old,IDe_old,n2D;
  FILE *fp,*fp2;
  char fileCD1[YOUSO10];
  char fileCD2[YOUSO10];
  char file_check[YOUSO10];
  double *tmp_array0,*tmp_array1;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /*
  printf("myid=%2d numprocs_new=%2d numprocs_old=%2d\n",
          myid,numprocs_new,numprocs_old);
  */

  N2D = N1*N2;
  n2Ds_new = (myid*N2D+numprocs_new-1)/numprocs_new;
  n2De_new = ((myid+1)*N2D+numprocs_new-1)/numprocs_new;
  IDs_old = n2Ds_new*numprocs_old/N2D;
  IDe_old = n2De_new*numprocs_old/N2D;

  /* allocation of array */     
  num = (n2De_new-n2Ds_new)*N3;
  tmp_array1 = (double*)malloc(sizeof(double)*num);

  /********************************************************
                  Restructure files of crst
  ********************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<Extrapolated_Charge_History; i++){

      n = 0;
      for (ID=IDs_old; ID<=IDe_old; ID++){

	n2Ds_old = (ID*N2D+numprocs_old-1)/numprocs_old;
	n2De_old = ((ID+1)*N2D+numprocs_old-1)/numprocs_old;

	/* allocation of array */     
	num = (n2De_old-n2Ds_old)*N3;
	tmp_array0 = (double*)malloc(sizeof(double)*num);

        /* set file name */ 
	sprintf(fileCD1,"%s%s_rst/%s.crst%i_%i_%i",filepath,restart_filename,restart_filename,spin,ID,i);

        /* read files */ 

	if ((fp = fopen(fileCD1,"rb")) != NULL){

	  fread(tmp_array0,sizeof(double),num,fp);
	  fclose(fp);
 
          for (n2D=n2Ds_old; n2D<n2De_old; n2D++){
            if (n2Ds_new<=n2D && n2D<n2De_new){
              for (n3=0; n3<N3; n3++){
                m = (n2D-n2Ds_old)*N3 + n3; 
                tmp_array1[n] = tmp_array0[m];              
                n++;
	      }
            }
          } 
	}

	/* freeing of array */     
	free(tmp_array0);

      } /* ID */

      /* set file name */ 
      sprintf(fileCD2,"%s%s_rst/%s.crst%i_%i_%i_tmp",filepath,restart_filename,restart_filename,spin,myid,i);

      if ((n2De_new-n2Ds_new)*N3!=n && n!=0){
        printf("Warning!!: the restart files were not properly read\n");
      }

      if (n!=0){  
	/* save data */
	if ((fp2 = fopen(fileCD2,"wb")) != NULL){
	  fwrite(tmp_array1,sizeof(double),n,fp2);
	  fclose(fp2);
	}
	else{
	  printf("Could not open a file %s\n",fileCD2);
	}
      }

    } /* i */
  } /* spin */

  /* rewrite "crst_check" files */

  sprintf(file_check,"%s%s_rst/%s.crst_check",filepath,restart_filename,restart_filename);

  if ((fp = fopen(file_check,"w")) != NULL){
    fprintf(fp,"%d %d %d %d",numprocs_new,N1,N2,N3);
    fclose(fp);
  }
  else{
    printf("Failure of saving %s\n",file_check);
  }

  /* remove old files */

  MPI_Barrier(mpi_comm_level1);
   
  if (myid==0){
    for (spin=0; spin<=SpinP_switch; spin++){
      for (ID=0; ID<numprocs_old; ID++){
	for (i=0; i<Extrapolated_Charge_History; i++){

          sprintf(fileCD1,"%s%s_rst/%s.crst%i_%i_%i",filepath,restart_filename,restart_filename,spin,ID,i);
          fp = fopen(fileCD1, "r");   

	  if (fp!=NULL){
	    fclose(fp); 
	    remove(fileCD1);
	  }
	}
      }
    }
  }

  /* rename *_tmp */

  MPI_Barrier(mpi_comm_level1);

  for (spin=0; spin<=SpinP_switch; spin++){
    for (i=0; i<Extrapolated_Charge_History; i++){

      sprintf(fileCD1,"%s%s_rst/%s.crst%i_%i_%i",filepath,restart_filename,restart_filename,spin,myid,i);
      sprintf(fileCD2,"%s%s_rst/%s.crst%i_%i_%i_tmp",filepath,restart_filename,restart_filename,spin,myid,i);
      rename(fileCD2,fileCD1); 
    }
  }

  /********************************************************
                 Restructure files of adrst
  ********************************************************/

  n = 0;
  for (ID=IDs_old; ID<=IDe_old; ID++){

    n2Ds_old = (ID*N2D+numprocs_old-1)/numprocs_old;
    n2De_old = ((ID+1)*N2D+numprocs_old-1)/numprocs_old;

    /* allocation of array */     
    num = (n2De_old-n2Ds_old)*N3;
    tmp_array0 = (double*)malloc(sizeof(double)*num);

    /* set file name */ 
    sprintf(fileCD1,"%s%s_rst/%s.adrst%i",filepath,restart_filename,restart_filename,ID);

    /* read files */ 

    if ((fp = fopen(fileCD1,"rb")) != NULL){

      fread(tmp_array0,sizeof(double),num,fp);
      fclose(fp);
 
      for (n2D=n2Ds_old; n2D<n2De_old; n2D++){
	if (n2Ds_new<=n2D && n2D<n2De_new){
	  for (n3=0; n3<N3; n3++){
	    m = (n2D-n2Ds_old)*N3 + n3; 
	    tmp_array1[n] = tmp_array0[m];              
	    n++;
	  }
	}
      } 
    }

    /* freeing of array */
    free(tmp_array0);

  } /* ID */

  /* set file name */ 
  sprintf(fileCD2,"%s%s_rst/%s.adrst%i_tmp",filepath,restart_filename,restart_filename,myid);

  if ((n2De_new-n2Ds_new)*N3!=n && n!=0){
    printf("Warning!!: the restart files were not properly read\n");
  }

  if (n!=0){  
    /* save data */
    if ((fp2 = fopen(fileCD2,"wb")) != NULL){
      fwrite(tmp_array1,sizeof(double),n,fp2);
      fclose(fp2);
    }
    else{
      printf("Could not open a file %s\n",fileCD2);
    }
  }

  /* remove old files */

  MPI_Barrier(mpi_comm_level1);
   
  if (myid==0){

    for (ID=0; ID<numprocs_old; ID++){

      sprintf(fileCD1,"%s%s_rst/%s.adrst%i",filepath,restart_filename,restart_filename,ID);
      fp = fopen(fileCD1, "r");   

      if (fp!=NULL){
	fclose(fp); 
	remove(fileCD1);
      }
    }
  }

  /* rename *_tmp */

  MPI_Barrier(mpi_comm_level1);

  sprintf(fileCD1,"%s%s_rst/%s.adrst%i",filepath,restart_filename,restart_filename,myid);
  sprintf(fileCD2,"%s%s_rst/%s.adrst%i_tmp",filepath,restart_filename,restart_filename,myid);
  rename(fileCD2,fileCD1); 

  /* freeing of array */     
  free(tmp_array1);
}





static void Set_In_First_Cell()
{
  int i,Gc_AN;
  int itmp;
  double Cxyz[4];
  double tmp[4];
  double xc,yc,zc;
  double CellV;

  /* calculate the reciprocal vectors */

  Cross_Product(tv[2],tv[3],tmp);
  CellV = Dot_Product(tv[1],tmp); 
  Cell_Volume = fabs(CellV);

  Cross_Product(tv[2],tv[3],tmp);
  rtv[1][1] = 2.0*PI*tmp[1]/CellV;
  rtv[1][2] = 2.0*PI*tmp[2]/CellV;
  rtv[1][3] = 2.0*PI*tmp[3]/CellV;

  Cross_Product(tv[3],tv[1],tmp);
  rtv[2][1] = 2.0*PI*tmp[1]/CellV;
  rtv[2][2] = 2.0*PI*tmp[2]/CellV;
  rtv[2][3] = 2.0*PI*tmp[3]/CellV;
  
  Cross_Product(tv[1],tv[2],tmp);
  rtv[3][1] = 2.0*PI*tmp[1]/CellV;
  rtv[3][2] = 2.0*PI*tmp[2]/CellV;
  rtv[3][3] = 2.0*PI*tmp[3]/CellV;

  /* find the center of coordinates */

  xc = 0.0;
  yc = 0.0;
  zc = 0.0;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){
    xc += Gxyz[Gc_AN][1];
    yc += Gxyz[Gc_AN][2];
    zc += Gxyz[Gc_AN][3];
  }

  xc = xc/(double)atomnum;
  yc = yc/(double)atomnum;
  zc = zc/(double)atomnum;

  X_Center_Coordinate = xc;
  Y_Center_Coordinate = yc;
  Z_Center_Coordinate = zc;

  for (Gc_AN=1; Gc_AN<=atomnum; Gc_AN++){

    Cxyz[1] = Gxyz[Gc_AN][1] - xc;
    Cxyz[2] = Gxyz[Gc_AN][2] - yc;
    Cxyz[3] = Gxyz[Gc_AN][3] - zc;

    Cell_Gxyz[Gc_AN][1] = Dot_Product(Cxyz,rtv[1])*0.5/PI;
    Cell_Gxyz[Gc_AN][2] = Dot_Product(Cxyz,rtv[2])*0.5/PI;
    Cell_Gxyz[Gc_AN][3] = Dot_Product(Cxyz,rtv[3])*0.5/PI;

    for (i=1; i<=3; i++){
      if (1.0<fabs(Cell_Gxyz[Gc_AN][i])){
        if (0.0<=Cell_Gxyz[Gc_AN][i]){
          itmp = (int)Cell_Gxyz[Gc_AN][i]; 
          Cell_Gxyz[Gc_AN][i] = Cell_Gxyz[Gc_AN][i] - (double)itmp;
	}
        else{
          itmp = abs((int)Cell_Gxyz[Gc_AN][i]) + 1; 
          Cell_Gxyz[Gc_AN][i] = Cell_Gxyz[Gc_AN][i] + (double)itmp;
        }
      }
    }

    Gxyz[Gc_AN][1] =  Cell_Gxyz[Gc_AN][1]*tv[1][1]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][1]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][1] + xc;

    Gxyz[Gc_AN][2] =  Cell_Gxyz[Gc_AN][1]*tv[1][2]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][2]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][2] + yc;

    Gxyz[Gc_AN][3] =  Cell_Gxyz[Gc_AN][1]*tv[1][3]
                    + Cell_Gxyz[Gc_AN][2]*tv[2][3]
                    + Cell_Gxyz[Gc_AN][3]*tv[3][3] + zc;
  }

}






int Analyze_Wannier_Projectors(int p, char ctmp[YOUSO10], 
                               int **tmp_Wannier_Pro_SelMat,
                               double ***tmp_Wannier_Projector_Hybridize_Matrix)
{
  /*******************************************
   list of supported projectors:
   s(1),
   px(1),py(1),pz(1),
   dz2(1),dx2-y2(1),dxy(1),dxz(1),dyz(1),
   fz3(1),fxz2(1),fyz2(1),fzx2(1),fxyz(1),fx3-3xy2(1),f3yx2-y3(1),
   sp  (s+px)(2),
   sp2 (s+px+py)(3),
   sp3 (s+px+py+pz)(4),
   sp3dz2(5),sp3deg(6),
   p (all p)(3)
   d (all d)(5)
   f (all f)(7)
   where the number of parenthesis is 
   the total number of projectors. 
   The number should be less than Max_Num_WF_Projs.
  *******************************************/

  int i,po,ip,num,L,j,k,po_ok;
  /* Here 100 means the all the posibilitty of combination of initial
     guess. Total number of initial guess orbital should smaller than 100. */
  int Num_Projectors[100];
  int NumL[100][4];
  /* The following array has dimension 20, that means the maximume number
     of projectors included in one orbital combination. sp3 means 4 orbitals. 
     sp3deg means 6 orbitals */ 
  int SelMat[100][Max_Num_WF_Projs];
  double HybridMat[100][Max_Num_WF_Projs][Max_Num_WF_Projs];
  char Name_Wannier_Projectors[100][30];
  char c;
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* analysis of projectors used for calculation of Wannier functions */

  i = 0;
  po = 0;

  while( ((c=ctmp[i])!='\0' || po<=1) && i<YOUSO10 ){

    if (c=='-' && po==0){
      po = 1;
      Wannier_ProSpeName[p][i] = '\0';
      ip = i;
    }
    else if (po==0){
      Wannier_ProSpeName[p][i] = ctmp[i];
    } 
    else if (po==1){
      Wannier_ProName[p][i-ip-1] = ctmp[i];
    }
    else if (c=='\n'){
      po = 2;
      Wannier_ProName[p][i-ip-1] = '\0';
    }
       
    i++;
  }

  if (myid==Host_ID){  
    printf("Projector name %s, orbitals %s.\n",Wannier_ProSpeName[p],Wannier_ProName[p]);
  }

  /* check whether Wannier_ProSpeName is found in SpeBasisName or not */

  po = 0;
  for (i=0; i<SpeciesNum; i++){

    /* 
    if (myid==Host_ID) printf("i=%2d %s\n",i,SpeName[i]);
    */

    if (strcmp(Wannier_ProSpeName[p],SpeName[i])==0){
      po = 1;
      WannierPro2SpeciesNum[p] = i;

      /* check the multiple basis specification of the same angular moment */

      po_ok = 0;

      /* s1 case */
      if (Spe_Num_Basis[i][0]==1 &&
          Spe_Num_Basis[i][1]==0 &&
          Spe_Num_Basis[i][2]==0 &&
          Spe_Num_Basis[i][3]==0 &&
          Spe_Num_Basis[i][4]==0)  po_ok = 1;

      /* s1p1 case */
      if (Spe_Num_Basis[i][0]==1 &&
          Spe_Num_Basis[i][1]==1 &&
          Spe_Num_Basis[i][2]==0 &&
          Spe_Num_Basis[i][3]==0 &&
          Spe_Num_Basis[i][4]==0)  po_ok = 2;

      /* s1p1d1 case */
      if (Spe_Num_Basis[i][0]==1 &&
          Spe_Num_Basis[i][1]==1 &&
          Spe_Num_Basis[i][2]==1 &&
          Spe_Num_Basis[i][3]==0 &&
          Spe_Num_Basis[i][4]==0)  po_ok = 3;

      /* s1p1d1f1 case */
      if (Spe_Num_Basis[i][0]==1 &&
          Spe_Num_Basis[i][1]==1 &&
          Spe_Num_Basis[i][2]==1 &&
          Spe_Num_Basis[i][3]==1 &&
          Spe_Num_Basis[i][4]==0)  po_ok = 4;

      if (po_ok==0){
	if (myid==Host_ID){
	  printf("Could not assign the multiple projector for the same angular part of '%s'\n",
		 Wannier_ProSpeName[p]);
	  printf("The following specification can be allowed:\n");
	  printf("   s1, s1p1, s1p1d1, or s1p1d1f1\n");
	}

	MPI_Finalize();
	exit(0);
      }

    }
  }

  if (po==0){
    if (myid==Host_ID){
      printf("could not find %s among species\n",Wannier_ProSpeName[p]);
    }
    MPI_Finalize();
    exit(0);
  }

  /* find the number of projectors */

  Num_Wannier_Template_Projectors = 24;

  for (i=0; i<Num_Wannier_Template_Projectors; i++){
    for (L=0; L<=3; L++){
      NumL[i][L] = 0;  
    }
  }   

  /* selection matrix and Hybridize Matrix*/

  /* s */
  strcpy(Name_Wannier_Projectors[  0], "s");          Num_Projectors[0]  = 1;  NumL[0][0]=1; 
  SelMat[0][0]=0; 
  HybridMat[0][0][0]=1.0;

  /* px */
  strcpy(Name_Wannier_Projectors[  1], "px");         Num_Projectors[1]  = 1;  NumL[1][1]=1;
  SelMat[1][0]=0;
  HybridMat[1][0][0]=1.0;

  /* py */
  strcpy(Name_Wannier_Projectors[  2], "py");         Num_Projectors[2]  = 1;  NumL[2][1]=1;
  SelMat[2][0]=1;
  HybridMat[2][0][0]=1.0;

  /* pz */
  strcpy(Name_Wannier_Projectors[  3], "pz");         Num_Projectors[3]  = 1;  NumL[3][1]=1;
  SelMat[3][0]=2;
  HybridMat[3][0][0]=1.0;

  /* dz2 */
  strcpy(Name_Wannier_Projectors[  4], "dz2");        Num_Projectors[4]  = 1;  NumL[4][2]=1;
  SelMat[4][0]=0;
  HybridMat[4][0][0]=1.0;

  /* dx2-y2 */
  strcpy(Name_Wannier_Projectors[  5], "dx2-y2");     Num_Projectors[5]  = 1;  NumL[5][2]=1;
  SelMat[5][0]=1;
  HybridMat[5][0][0]=1.0;

  /* dxy */
  strcpy(Name_Wannier_Projectors[  6], "dxy");        Num_Projectors[6]  = 1;  NumL[6][2]=1;
  SelMat[6][0]=2;
  HybridMat[6][0][0]=1.0;

  /* dxz */
  strcpy(Name_Wannier_Projectors[  7], "dxz");        Num_Projectors[7]  = 1;  NumL[7][2]=1;
  SelMat[7][0]=3;
  HybridMat[7][0][0]=1.0;

  /* dyz */
  strcpy(Name_Wannier_Projectors[  8], "dyz");        Num_Projectors[8]  = 1;  NumL[8][2]=1;
  SelMat[8][0]=4;
  HybridMat[8][0][0]=1.0;

  /* fz3 */
  strcpy(Name_Wannier_Projectors[  9], "fz3");        Num_Projectors[9] = 1;  NumL[9][3]=1;
  SelMat[9][0]=0;
  HybridMat[9][0][0]=1.0;

  /* fxz2 */
  strcpy(Name_Wannier_Projectors[ 10], "fxz2");       Num_Projectors[10] = 1;  NumL[10][3]=1;
  SelMat[10][0]=1;
  HybridMat[10][0][0]=1.0;

  /* fyz2 */
  strcpy(Name_Wannier_Projectors[ 11], "fyz2");       Num_Projectors[11] = 1;  NumL[11][3]=1;
  SelMat[11][0]=2;
  HybridMat[11][0][0]=1.0;

  /* fzx2 */
  strcpy(Name_Wannier_Projectors[ 12], "fzx2");       Num_Projectors[12] = 1;  NumL[12][3]=1;
  SelMat[12][0]=3;
  HybridMat[12][0][0]=1.0;

  /* fxyz */
  strcpy(Name_Wannier_Projectors[ 13], "fxyz");       Num_Projectors[13] = 1;  NumL[13][3]=1;
  SelMat[13][0]=4;
  HybridMat[13][0][0]=1.0;

  /* fx3-3xy2 */
  strcpy(Name_Wannier_Projectors[ 14], "fx3-3xy2");   Num_Projectors[14] = 1;  NumL[14][3]=1;
  SelMat[14][0]=5;
  HybridMat[14][0][0]=1.0;

  /* f3yx2-y3 */
  strcpy(Name_Wannier_Projectors[ 15], "f3yx2-y3");   Num_Projectors[15] = 1;  NumL[15][3]=1;
  SelMat[15][0]=6;
  HybridMat[15][0][0]=1.0;

  /* sp */
  strcpy(Name_Wannier_Projectors[ 16], "sp");         Num_Projectors[16] = 2;  NumL[16][0]=1; NumL[16][1]=1;
  SelMat[16][0]=0;  /* this sp means the s and px hybirdization, therefore s and px should be selected. */
  SelMat[16][1]=1;
  HybridMat[16][0][0]=1.0/sqrt(2.0); HybridMat[16][0][1]=1.0/sqrt(2.0);
  HybridMat[16][1][0]=1.0/sqrt(2.0); HybridMat[16][1][1]=-1.0/sqrt(2.0);

  /* sp2 */
  strcpy(Name_Wannier_Projectors[ 17], "sp2");        Num_Projectors[17] = 3;  NumL[17][0]=1; NumL[17][1]=1;
  SelMat[17][0]=0;  /* this sp2 means the s, px and py hybirdization, therefore s, px, py should be selected. */
  SelMat[17][1]=1;
  SelMat[17][2]=2;

  HybridMat[17][0][0] = 1.0/sqrt(3.0); 
  HybridMat[17][0][1] =-1.0/sqrt(6.0); 
  HybridMat[17][0][2] = 1.0/sqrt(2.0);

  HybridMat[17][1][0] = 1.0/sqrt(3.0); 
  HybridMat[17][1][1] =-1.0/sqrt(6.0); 
  HybridMat[17][1][2]= -1.0/sqrt(2.0);

  HybridMat[17][2][0] = 1.0/sqrt(3.0); 
  HybridMat[17][2][1] = 2.0/sqrt(6.0);
  HybridMat[17][2][2] = 0.0;

  /* sp3 */
  strcpy(Name_Wannier_Projectors[ 18], "sp3");        Num_Projectors[18] = 4;  NumL[18][0]=1; NumL[18][1]=1;
  SelMat[18][0]=0;  /* this sp3 means the s, px, py and pz hybirdization, therefore s, px, py, pz should be selected. */
  SelMat[18][1]=1;
  SelMat[18][2]=2;
  SelMat[18][3]=3;

  HybridMat[18][0][0] = 1.0/2.0; 
  HybridMat[18][0][1] = 1.0/2.0; 
  HybridMat[18][0][2] = 1.0/2.0; 
  HybridMat[18][0][3] = 1.0/2.0;

  HybridMat[18][1][0] = 1.0/2.0; 
  HybridMat[18][1][1] = 1.0/2.0; 
  HybridMat[18][1][2] =-1.0/2.0; 
  HybridMat[18][1][3] =-1.0/2.0;
 
  HybridMat[18][2][0] = 1.0/2.0; 
  HybridMat[18][2][1] =-1.0/2.0; 
  HybridMat[18][2][2] = 1.0/2.0; 
  HybridMat[18][2][3] =-1.0/2.0;

  HybridMat[18][3][0] = 1.0/2.0; 
  HybridMat[18][3][1] =-1.0/2.0; 
  HybridMat[18][3][2] =-1.0/2.0; 
  HybridMat[18][3][3] = 1.0/2.0;

  /* sp3dz2 */
  strcpy(Name_Wannier_Projectors[ 19], "sp3dz2");      
  Num_Projectors[19] = 5;  NumL[19][0]=1; NumL[19][1]=1; NumL[19][2]=1;

  /* this sp3dz2 means the s, px, py and pz, and dz2 hybirdization, 
     therefore s, px, py, pz and dz2 should be selected. */

  SelMat[19][0]=0;  
  SelMat[19][1]=1;
  SelMat[19][2]=2;
  SelMat[19][3]=3;
  SelMat[19][4]=4;

  HybridMat[19][0][0] = 1.0/sqrt(3.0); 
  HybridMat[19][0][1] =-1.0/sqrt(6.0); 
  HybridMat[19][0][2] = 1.0/sqrt(2.0); 
  HybridMat[19][0][3] = 0.0; 
  HybridMat[19][0][4] = 0.0;
 
  HybridMat[19][1][0] = 1.0/sqrt(3.0); 
  HybridMat[19][1][1] =-1.0/sqrt(6.0); 
  HybridMat[19][1][2] =-1.0/sqrt(2.0); 
  HybridMat[19][1][3] = 0.0; 
  HybridMat[19][1][4] = 0.0;

  HybridMat[19][2][0] = 1.0/sqrt(3.0); 
  HybridMat[19][2][1] = 2.0/sqrt(6.0); 
  HybridMat[19][2][2] = 0.0; 
  HybridMat[19][2][3] = 0.0; 
  HybridMat[19][2][4] = 0.0;

  HybridMat[19][3][0] = 0.0; 
  HybridMat[19][3][1] = 0.0; 
  HybridMat[19][3][2] = 0.0; 
  HybridMat[19][3][3] = 1.0/sqrt(2.0); 
  HybridMat[19][3][4] = 1.0/sqrt(2.0);

  HybridMat[19][4][0] = 0.0; 
  HybridMat[19][4][1] = 0.0; 
  HybridMat[19][4][2] = 0.0; 
  HybridMat[19][4][3] =-1.0/sqrt(2.0); 
  HybridMat[19][4][4] = 1.0/sqrt(2.0);

  /* sp3deg */

  strcpy(Name_Wannier_Projectors[ 20], "sp3deg");   
  Num_Projectors[20] = 6;  NumL[20][0]=1; NumL[20][1]=1; NumL[20][2]=1;

  /* this sp3deg means the s, px, py and pz, and dz2, dx2-y2 hybirdization, 
     therefore these orbitals should be selected. */

  SelMat[20][0]=0;  
  SelMat[20][1]=1;
  SelMat[20][2]=2;
  SelMat[20][3]=3;
  SelMat[20][4]=4;
  SelMat[20][5]=5;

  HybridMat[20][0][0] = 1.0/sqrt(6.0); 
  HybridMat[20][0][1] =-1.0/sqrt(2.0); 
  HybridMat[20][0][2] = 0.0; 
  HybridMat[20][0][3] = 0.0; 
  HybridMat[20][0][4] =-1.0/sqrt(12.0); 
  HybridMat[20][0][5] = 0.5;

  HybridMat[20][1][0] = 1.0/sqrt(6.0); 
  HybridMat[20][1][1] = 1.0/sqrt(2.0); 
  HybridMat[20][1][2] = 0.0; 
  HybridMat[20][1][3] = 0.0; 
  HybridMat[20][1][4] =-1.0/sqrt(12.0); 
  HybridMat[20][1][5] = 0.5;

  HybridMat[20][2][0] = 1.0/sqrt(6.0); 
  HybridMat[20][2][1] = 0.0; 
  HybridMat[20][2][2] =-1.0/sqrt(2.0); 
  HybridMat[20][2][3] = 0.0; 
  HybridMat[20][2][4] =-1.0/sqrt(12.0); 
  HybridMat[20][2][5] =-0.5;

  HybridMat[20][3][0] = 1.0/sqrt(6.0); 
  HybridMat[20][3][1] = 0.0; 
  HybridMat[20][3][2] = 1.0/sqrt(2.0); 
  HybridMat[20][3][3] = 0.0; 
  HybridMat[20][3][4] =-1.0/sqrt(12.0); 
  HybridMat[20][3][5] =-0.5;

  HybridMat[20][4][0] = 1.0/sqrt(6.0); 
  HybridMat[20][4][1] = 0.0; 
  HybridMat[20][4][2] = 0.0; 
  HybridMat[20][4][3] =-1.0/sqrt(2.0); 
  HybridMat[20][4][4] = 1.0/sqrt(3.0); 
  HybridMat[20][4][5] = 0.0;

  HybridMat[20][5][0] = 1.0/sqrt(6.0); 
  HybridMat[20][5][1] = 0.0; 
  HybridMat[20][5][2] = 0.0; 
  HybridMat[20][5][3] = 1.0/sqrt(2.0); 
  HybridMat[20][5][4] = 1.0/sqrt(3.0); 
  HybridMat[20][5][5] = 0.0;

  /* p, px, py and pz. no hybirdization. */

  strcpy(Name_Wannier_Projectors[ 21], "p");   
  Num_Projectors[21] = 3;  NumL[21][0]=0; NumL[21][1]=1; NumL[21][2]=0;

  SelMat[21][0]=0;  
  SelMat[21][1]=1;
  SelMat[21][2]=2;

  HybridMat[21][0][0]=1.0; HybridMat[21][0][1]=0.0; HybridMat[21][0][2]=0.0;
  HybridMat[21][1][0]=0.0; HybridMat[21][1][1]=1.0; HybridMat[21][1][2]=0.0;
  HybridMat[21][2][0]=0.0; HybridMat[21][2][1]=0.0; HybridMat[21][2][2]=1.0;

  /* d, all five d orbitals.*/

  strcpy(Name_Wannier_Projectors[ 22], "d");     
  Num_Projectors[22] = 5;  NumL[22][0]=0; NumL[22][1]=0; NumL[22][2]=1;

  SelMat[22][0]=0;  
  SelMat[22][1]=1;
  SelMat[22][2]=2;
  SelMat[22][3]=3;
  SelMat[22][4]=4;

  for (i=0; i<=4; i++){
    for (j=0; j<=4; j++){
      HybridMat[22][i][j]=0.0; 
      if (i==j) HybridMat[22][i][j]=1.0;
    }
  }
  /* f, all five f orbitals.*/

  strcpy(Name_Wannier_Projectors[ 23], "f");     
  Num_Projectors[23] = 7;  NumL[23][0]=0; NumL[23][1]=0; NumL[23][2]=0; NumL[23][3]=1;

  SelMat[23][0]=0;  
  SelMat[23][1]=1;
  SelMat[23][2]=2;
  SelMat[23][3]=3;
  SelMat[23][4]=4;
  SelMat[23][5]=5;
  SelMat[23][6]=6;

  for (i=0; i<=6; i++){
    for (j=0; j<=6; j++){
      HybridMat[23][i][j]=0.0; 
      if (i==j) HybridMat[23][i][j]=1.0;
    }
  }

  po = 0;  
  i = 0;
  do {
    if (strcmp(Name_Wannier_Projectors[i],Wannier_ProName[p])==0){
      po = 1;
      num = Num_Projectors[i];

      Wannier_ProName2Num[p]=i;

      Wannier_Num_Pro[p]  = Num_Projectors[i];

      Wannier_NumL_Pro[p][0] = NumL[i][0];
      Wannier_NumL_Pro[p][1] = NumL[i][1];
      Wannier_NumL_Pro[p][2] = NumL[i][2];
      Wannier_NumL_Pro[p][3] = NumL[i][3];

      /* for each kind of projectors, find the selection matrix which can tell the position of
	 projectors among the envolved basis set.
	 For example,
	 if projector(s) is(are) definded as dxy, which means the envolved basis set is local d
	 orbitals, the projector is dxy which is in the third one arranged in the following order:
	 dz2 --> 0, dx2-y2 --> 1, dxy -->2, dxz --> 3, dyz -->4.
	 if projectors are defined as sp2, which means the envolved basis set are local s and p
	 orbitals, the used local orbitals are s, px, py, this seletion matrix will have the values
	 like:
	 s --> 0, px -->1, py --> 2  (pz is in the basis set, but will not be selected in
	 the final Amn matrix)
      */

      for(j=0;j<num;j++){
	tmp_Wannier_Pro_SelMat[p][j]=SelMat[i][j];
	for(k=0;k<num;k++){
	  tmp_Wannier_Projector_Hybridize_Matrix[p][j][k]=HybridMat[i][j][k];
	}
      }
    }
    i++;
  } while(po==0 && i<Num_Wannier_Template_Projectors);

  if (po==0){

    if (myid==Host_ID){  
      printf("Error:WF could not find %s among defined projectors\n",Wannier_ProName[p]);
    }

    MPI_Finalize();
    exit(0);
  }

  return num;
}



void SpeciesString2int(int p)
{
  int i,l,n,po,k,base;
  int tmp;
  int nlist[10]; 
  char c,cstr[YOUSO10*3];
  int numprocs,myid;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* initialize nlist */

  for (i=0; i<10; i++) nlist[i] = 0;
  
  /* Get basis name */

  sprintf(SpeBasisName[p],"");

  i = 0;
  po = 0;
  while( ((c=SpeBasis[p][i])!='\0' || po==0) && i<YOUSO10 ){
    if (c=='-'){
      po = 1;
      SpeBasisName[p][i] = '\0';  
    }
    if (po==0) SpeBasisName[p][i] = SpeBasis[p][i]; 
    i++;
  }

  if (2<=level_stdout && myid==Host_ID){
    printf("<Input_std>  SpeBasisName=%s\n",SpeBasisName[p]);
  }

  /* Get basis type */

  for (l=0; l<5; l++){
    Spe_Num_Basis[p][l] = 0;
  }

  i = 0;
  po = 0;
  n = 0; 

  /* get informations of basis set */

  while((c=SpeBasis[p][i])!='\0'){

    if (po==1){

      /* in case of orbital optimization, set Spe_Num_CBasis */
      if ( (c=='s' || c=='p' || c=='d' || c=='f' || c=='g') 
          && n!=0 && Cnt_switch==1){

	Spe_Num_CBasis[p][l] = 0;
	base = 1;
	for (k=(n-1); 0<=k; k--){
	  Spe_Num_CBasis[p][l] += nlist[k]*base;
	  base *= 10;
	}
      }

      /* analysis of the string */

      if      (c=='s'){ l=0; n=0; }
      else if (c=='p'){ l=1; n=0; }
      else if (c=='d'){ l=2; n=0; }
      else if (c=='f'){ l=3; n=0; }
      else if (c=='g'){ l=4; n=0; }
      else{

        /* no orbital optimization */
        if (Cnt_switch==0){

          if (c=='>') {
	    printf("Format error in Definition of Atomic Species\n");
	    MPI_Finalize();
	    exit(0);
          }

	  if (n==0){
	    cstr[0] = c;
	    cstr[1] = '\0';
	    Spe_Num_Basis[p][l] = atoi(cstr); /* corrected by t.ohwaki */
	    n++;
	  }
	  else if (n==1){
	    cstr[0] = c;
	    cstr[1] = '\0';
            tmp = Spe_Num_Basis[p][l];
	    Spe_Num_Basis[p][l] = 10*tmp + atoi(cstr);  /* corrected by t.ohwaki */ 
	    n++;
	  }
	  else if (n==2){
	    cstr[0] = c;
	    cstr[1] = '\0';
            tmp = Spe_Num_Basis[p][l];
	    Spe_Num_Basis[p][l] = 10*tmp + atoi(cstr);  /* corrected by t.ohwaki */ 
	    n++;
	  }
	  else {
	    printf("Format error in Definition of Atomic Species\n");
	    MPI_Finalize();
	    exit(0);
	  }
	}

        /* orbital optimization */
        else if (Cnt_switch==1){

          if (c=='>'){

            /* set Spe_Num_Basis */
	    Spe_Num_Basis[p][l] = 0;
            base = 1;
            for (k=(n-1); 0<=k; k--){
   	      Spe_Num_Basis[p][l] += nlist[k]*base;
              base *= 10;
            }
            
            /* reset n */
            n = 0;
          } 
          
          else {
	    cstr[0] = c;
	    cstr[1] = '\0';
            nlist[n] = atoi(cstr);
            n++;
          }
	}

      } 
    }  

    if (SpeBasis[p][i]=='-') po = 1;
    i++;
  }

  /* in case of orbital optimization, set Spe_Num_CBasis */

  if ( n!=0 && Cnt_switch==1){

    Spe_Num_CBasis[p][l] = 0;
    base = 1;
    for (k=(n-1); 0<=k; k--){
      Spe_Num_CBasis[p][l] += nlist[k]*base;
      base *= 10;
    }
  }

  /* check the number of primitive and contracted basis functions */

  for (l=0; l<5; l++){
    if (Spe_Num_Basis[p][l]!=0) Spe_MaxL_Basis[p] = l;

    if (Spe_Num_Basis[p][l]<Spe_Num_CBasis[p][l]){

      printf("# of contracted orbitals are larger than # of primitive oribitals\n");
      printf("Primitive=%3d Contracted=%3d\n",Spe_Num_Basis[p][l],Spe_Num_CBasis[p][l]);
      MPI_Finalize();
      exit(0); 
    } 

    if (2<=level_stdout && Cnt_switch==0 && myid==Host_ID){
      printf("<Input_std>  p=%2d l=%2d Primitive=%3d\n",
              p,l,Spe_Num_Basis[p][l]);
    }
    else if (Cnt_switch==1 && myid==Host_ID) {   /* added by t.ohwaki */
      printf("<Input_std>  p=%2d l=%2d Primitive=%3d Contracted=%3d\n",
              p,l,Spe_Num_Basis[p][l],Spe_Num_CBasis[p][l]);
    }

  }

  if (2<=level_stdout || Cnt_switch==1) printf("\n");
}





int Species2int(char Species[YOUSO10])
{
  int i,po;

  i = 0;
  po = 0; 
  while (i<SpeciesNum && po==0){
    if (SEQ(Species,SpeName[i])==1){
      po = 1;
    }
    if (po==0) i++;
  };

  if (po==0){
    printf("Found an undefined species name %s\n",Species);
    printf("in Atoms.SpeciesAndCoordinates or Hubbard.U.values\n");
    printf("Please check your input file\n");
    MPI_Finalize();
    exit(0);
  }

  return i;
}



int OrbPol2int(char OrbPol[YOUSO10])
{
  int i,po;
  char opns[3][YOUSO10]={"OFF","ON","EX"};

  i = 0;
  po = 0; 

  ToCapital(OrbPol);  

  while (i<3 && po==0){
    if (SEQ(OrbPol,opns[i])==1){
      po = 1;
    }
    if (po==0) i++;
  };

  if (po==0){
    printf("Invalid flag for LDA+U (Atoms.SpeciesAndCoordinates)  %s\n",OrbPol);
    printf("Please check your input file\n");
    MPI_Finalize();
    exit(0);
  }

  return i;
}



char *ToCapital(char *s)
{
  char *p;
  for (p=s; *p; p++) *p = toupper(*p);
  return (s);  
}




void kpath_changeunit( double tv[4][4], double tv0[4][4], int Band_Nkpath,
                       double ***Band_kpath )
{
  /***********************************************************************
    k1 rtv0[0] + k2 rtv0[1] + k3 rtv0[2] = l rtv[0] + m rtv[1] + n rtv[2] 
      rtv = reciptical vector of tv
      rtv0 = reciptical vector of tv0
    e.g.   l is given by 
     tv[0] ( k1 rtv0[0] + k2 rtv0[1] + k3 rtv0[2]) = l tv[0] rtv[0] 
  ************************************************************************/

  double tmp[4], CellV;
  double rtv[4][4],rtv0[4][4];
  int i,j;
  double r;
  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  Cross_Product(tv[2],tv[3],tmp);
  CellV = Dot_Product(tv[1],tmp); 
  
  Cross_Product(tv[2],tv[3],tmp);
  rtv[1][1] = 2.0*PI*tmp[1]/CellV;
  rtv[1][2] = 2.0*PI*tmp[2]/CellV;
  rtv[1][3] = 2.0*PI*tmp[3]/CellV;
  
  Cross_Product(tv[3],tv[1],tmp);
  rtv[2][1] = 2.0*PI*tmp[1]/CellV;
  rtv[2][2] = 2.0*PI*tmp[2]/CellV;
  rtv[2][3] = 2.0*PI*tmp[3]/CellV;
  
  Cross_Product(tv[1],tv[2],tmp);
  rtv[3][1] = 2.0*PI*tmp[1]/CellV;
  rtv[3][2] = 2.0*PI*tmp[2]/CellV;
  rtv[3][3] = 2.0*PI*tmp[3]/CellV; 

  Cross_Product(tv0[2],tv0[3],tmp);
  CellV = Dot_Product(tv0[1],tmp);

  Cross_Product(tv0[2],tv0[3],tmp);
  rtv0[1][1] = 2.0*PI*tmp[1]/CellV;
  rtv0[1][2] = 2.0*PI*tmp[2]/CellV;
  rtv0[1][3] = 2.0*PI*tmp[3]/CellV;

  Cross_Product(tv0[3],tv0[1],tmp);
  rtv0[2][1] = 2.0*PI*tmp[1]/CellV;
  rtv0[2][2] = 2.0*PI*tmp[2]/CellV;
  rtv0[2][3] = 2.0*PI*tmp[3]/CellV;

  Cross_Product(tv0[1],tv0[2],tmp);
  rtv0[3][1] = 2.0*PI*tmp[1]/CellV;
  rtv0[3][2] = 2.0*PI*tmp[2]/CellV;
  rtv0[3][3] = 2.0*PI*tmp[3]/CellV;

  if (myid==Host_ID){
    printf("kpath (converted)\n");
  }

  for (i=1;i<=Band_Nkpath;i++) {
    for (j=1;j<=3;j++) tmp[j]=Band_kpath[i][1][j];
    for (j=1;j<=3;j++) {
      r =    tmp[1]* Dot_Product(tv[j],rtv0[1]) 
           + tmp[2]* Dot_Product(tv[j],rtv0[2])
	   + tmp[3]* Dot_Product(tv[j],rtv0[3]);
      Band_kpath[i][1][j] = r/ Dot_Product(tv[j],rtv[j]);
    }
    for (j=1;j<=3;j++) tmp[j]=Band_kpath[i][2][j];
    for (j=1;j<=3;j++) {
      r =    tmp[1]* Dot_Product(tv[j],rtv0[1])
             + tmp[2]* Dot_Product(tv[j],rtv0[2])
             + tmp[3]* Dot_Product(tv[j],rtv0[3]);
      Band_kpath[i][2][j] = r/ Dot_Product(tv[j],rtv[j]);
    }

    if (myid==Host_ID){
      printf("(%lf %lf %lf) (%lf %lf %lf)\n",
             Band_kpath[i][1][1],Band_kpath[i][1][2],Band_kpath[i][1][3],
	     Band_kpath[i][2][1],Band_kpath[i][2][2],Band_kpath[i][2][3]);
    }
  }   

}


void kpoint_changeunit(double tv[4][4],double tv0[4][4],int MO_Nkpoint,
                       double **MO_kpoint)
{
  /***********************************************************************
    k1 rtv0[0] + k2 rtv0[1] + k3 rtv0[2] = l rtv[0] + m rtv[1] + n rtv[2] 
      rtv = reciptical vector of tv
      rtv0 = reciptical vector of tv0
    e.g.   l is given by 
     tv[0] ( k1 rtv0[0] + k2 rtv0[1] + k3 rtv0[2]) = l tv[0] rtv[0] 
  ************************************************************************/

  double tmp[4], CellV;
  double rtv[4][4],rtv0[4][4];
  int i,j;
  double r;
  int myid;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  Cross_Product(tv[2],tv[3],tmp);
  CellV = Dot_Product(tv[1],tmp); 
  
  Cross_Product(tv[2],tv[3],tmp);
  rtv[1][1] = 2.0*PI*tmp[1]/CellV;
  rtv[1][2] = 2.0*PI*tmp[2]/CellV;
  rtv[1][3] = 2.0*PI*tmp[3]/CellV;
  
  Cross_Product(tv[3],tv[1],tmp);
  rtv[2][1] = 2.0*PI*tmp[1]/CellV;
  rtv[2][2] = 2.0*PI*tmp[2]/CellV;
  rtv[2][3] = 2.0*PI*tmp[3]/CellV;
  
  Cross_Product(tv[1],tv[2],tmp);
  rtv[3][1] = 2.0*PI*tmp[1]/CellV;
  rtv[3][2] = 2.0*PI*tmp[2]/CellV;
  rtv[3][3] = 2.0*PI*tmp[3]/CellV; 

  Cross_Product(tv0[2],tv0[3],tmp);
  CellV = Dot_Product(tv0[1],tmp);

  Cross_Product(tv0[2],tv0[3],tmp);
  rtv0[1][1] = 2.0*PI*tmp[1]/CellV;
  rtv0[1][2] = 2.0*PI*tmp[2]/CellV;
  rtv0[1][3] = 2.0*PI*tmp[3]/CellV;

  Cross_Product(tv0[3],tv0[1],tmp);
  rtv0[2][1] = 2.0*PI*tmp[1]/CellV;
  rtv0[2][2] = 2.0*PI*tmp[2]/CellV;
  rtv0[2][3] = 2.0*PI*tmp[3]/CellV;

  Cross_Product(tv0[1],tv0[2],tmp);
  rtv0[3][1] = 2.0*PI*tmp[1]/CellV;
  rtv0[3][2] = 2.0*PI*tmp[2]/CellV;
  rtv0[3][3] = 2.0*PI*tmp[3]/CellV;

  if (myid==Host_ID){
    printf("kpoint at which wave functions are calculated (converted)\n");
  }

  for (i=0;i<MO_Nkpoint;i++){
    for (j=1;j<=3;j++) tmp[j]= MO_kpoint[i][j];
    for (j=1;j<=3;j++) {
      r =    tmp[1]* Dot_Product(tv[j],rtv0[1]) 
           + tmp[2]* Dot_Product(tv[j],rtv0[2])
           + tmp[3]* Dot_Product(tv[j],rtv0[3]);
      MO_kpoint[i][j] = r/ Dot_Product(tv[j],rtv[j]);
    }
    if (myid==Host_ID){
      printf("%lf %lf %lf\n",MO_kpoint[i][1],MO_kpoint[i][2],MO_kpoint[i][3]);
    }
  }
}




/* *** calculate an unit cell of a cluster,    ***
 * assuming that unit of Gxyz and Rc is A.U. 
 * cell size = (max[ xyz-Rc ] - min[ xyz+Rc ])*1.01
*/


void Set_Cluster_UnitCell(double tv[4][4], int unitflag)
{
/* 
 * Species: int SpeciesNum, char SpeName[], char SpeBasis[]
 *
 * Coordinate:   int WhatSpecies[]; double Gxyz[][]
 *
 * unitflag=0 (Ang.)  unitflag=1 (a.u.),  used only to print them
 *
 * tv[][] is always in a.u.
*/
  FILE *fp;
  int i,id,spe,myid; 
  double *sperc;
  double min[4],max[4],gmin[4],gmax[4]; 
  char FN_PAO[YOUSO10];
  char ExtPAO[YOUSO10] = ".pao";
  char DirPAO[YOUSO10];

  double margin=1.10;
  double unit;
  char *unitstr[2]={"Ang.","a.u."};

  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* set DirPAO */

  sprintf(DirPAO,"%s/PAO/",DFT_DATA_PATH);

  unit=1.0;
  if (unitflag==0) unit=BohrR;

  sperc=(double*)malloc(sizeof(double)*SpeciesNum);

  for (spe=0; spe<SpeciesNum; spe++){
    fnjoint2(DirPAO,SpeBasisName[spe],ExtPAO,FN_PAO);    
    if ((fp = fopen(FN_PAO,"r")) != NULL){
      input_open(FN_PAO);
      input_double("radial.cutoff.pao",&sperc[spe],(double)0.0);
      input_close();
      fclose(fp);
    }
    else {

      if (myid==Host_ID){ 
        printf("Set_Cluster_UnitCell: can not open %s\n", FN_PAO); 
      }

      MPI_Finalize();     
      exit(0); 
    }
  }

  if (level_stdout>=2) {
    for (spe=0;spe<SpeciesNum;spe++) {
    printf("<Set_Cluster_UnitCell> %d %s rc=%lf\n",spe,SpeName[spe],sperc[spe]);
    }
  }


  if (level_stdout>=2) {
   printf("<Set_Cluster_UnitCell> x y z   Rc\n");
  }

  for (i=1;i<=3;i++) {
     gmax[i]=gmin[i]=Gxyz[1][i];
  }

  for (i=1;i<=atomnum;i++) {
    id=WhatSpecies[i];   
    /*  printf("%d %d %lf\n",i,id,sperc[id]); */
    if (level_stdout>=2) {
      printf("<Set_Cluster_UnitCell> %lf %lf %lf %lf\n",
              Gxyz[i][1],Gxyz[i][2],Gxyz[i][3],sperc[id]);
    }
    min[1]=Gxyz[i][1]-sperc[id];
    min[2]=Gxyz[i][2]-sperc[id];
    min[3]=Gxyz[i][3]-sperc[id];
    max[1]=Gxyz[i][1]+sperc[id];
    max[2]=Gxyz[i][2]+sperc[id];
    max[3]=Gxyz[i][3]+sperc[id];
    if (min[1]<gmin[1]) gmin[1]=min[1];
    if (min[2]<gmin[2]) gmin[2]=min[2];
    if (min[3]<gmin[3]) gmin[3]=min[3];
    if (max[1]>gmax[1]) gmax[1]=max[1];
    if (max[2]>gmax[2]) gmax[2]=max[2];
    if (max[3]>gmax[3]) gmax[3]=max[3];
  }

  /* initialize */
  for (id=1;id<=3;id++){
    for(i=1;i<=3;i++) {
      tv[id][i]=0.0;
    }
  }

  tv[1][1]=(gmax[1]-gmin[1])*margin;
  tv[2][2]=(gmax[2]-gmin[2])*margin;
  tv[3][3]=(gmax[3]-gmin[3])*margin;

  if (level_stdout>=2) {
    printf("<Set_Cluster_UnitCell> to determine the unit cell, min and max includes effects of Rc\n");
    for (i=1;i<=3;i++) 
    printf("<Set_Cluster_UnitCell> axis=%d min,max=%lf %lf\n", i, gmin[i]*unit,gmax[i]*unit);
  } 

  if (myid==Host_ID && 0<level_stdout){

    printf("<Set_Cluster_UnitCell> automatically determied UnitCell(%s)\n<Set_Cluster_UnitCell> from atomic positions and Rc of PAOs (margin= %5.2lf%%)\n",unitstr[unitflag],(margin-1.0)*100.0);
    for(i=1;i<=3;i++) {
      printf("<Set_Cluster_UnitCell> %lf %lf %lf\n",
               tv[i][1]*unit,tv[i][2]*unit, tv[i][3]*unit);
    }
    printf("\n");
  }

  free(sperc); 
}



int divisible_cheker(int N)
{
  /************************
   return 0; non-divisible 
   return 1; divisible 
  ************************/

  int i,po;

  if (N!=1){
    po = 1; 
    for (i=0; i<NfundamentalNum; i++){
      if ( N!=1 && (N % fundamentalNum[i])==0 ){
	po = 0;
	N = N/fundamentalNum[i];
      }
    }
  }
  else{
    po = 0;
  }

  if (po==0 && N!=1){
    divisible_cheker(N);
  }

  if (po==0) return 1;
  else       return 0;
}











/* hmweng */
void Get_Euler_Rotation_Angle(
      double zx, double zy, double zz,
      double xx, double xy, double xz,
      double *alpha_r, double *beta_r, double *gamma_r)
{
  double norm, coszx, yx,yy,yz, alpha, beta, gamma, tmp1;
  int numprocs,myid;

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  *alpha_r=0.0;
  *beta_r=0.0;
  *gamma_r=0.0;
  alpha=0.0;
  beta=0.0;
  gamma=0.0;

  /* check orthogonality of z-axis and x-axis and normalise them. */

  norm=sqrt(zx*zx+zy*zy+zz*zz);
  zx=zx/norm;
  zy=zy/norm;
  zz=zz/norm;

  norm=sqrt(xx*xx+xy*xy+xz*xz);
  xx=xx/norm;
  xy=xy/norm;
  xz=xz/norm;

  coszx=(zx*xx+zy*xy+zz*xz);

  if(fabs(coszx)>1e-6){ /* not orthogonal */

    if (myid==Host_ID){
      printf("Error:WF  z-axis and x-axis are not orthogonal, please check it and try again.\n");
    }

    MPI_Finalize();
    exit(0);
  }
  /* orthogonal */
  /* then get y-axis = z-axis cross product x-axis to make a right-hand coordinate. */

  yx=zy*xz-zz*xy;
  yy=zz*xx-zx*xz;
  yz=zx*xy-zy*xx;

  norm=sqrt(yx*yx+yy*yy+yz*yz);
  yx=yx/norm;
  yy=yy/norm;
  yz=yz/norm;
       
  if(fabs(fabs(zz)-1.0)<1e-6){ /* new z-axis is along old z-axis */
    if(zz>0){ /* along the positive direciton */
      beta=0.0; 
    }else{
      beta=PI; /* have a 180-degree rotation */ 
    }
    /* in this case, alpha and gamma are the same, rotating around  the same z-axis.*/
    gamma=0.0;
    /* alpha is determined by x-axis */ 
    alpha=asin(xy/cos(beta)); /*  sin(alpha)*cos(beta)=xy */
    if(xx/zz<0){    /* cos(alpha)*cos(beta)=xx */
      alpha=PI-alpha;
    }
    if(alpha<0){
      alpha=2.0*PI+alpha;
    }  
  }

  else{

    beta=acos(zz);           /* beta is always in [0, PI] */ 
    tmp1=sqrt(zx*zx+zy*zy);  /* sin(beta) */
    alpha=asin(zy/tmp1);     /* zy=sin(beta)*sin(alpha)   asin() gives vale between [-PI/2, PI/2] */ 
    if(zx<0){   /* zx=sin(beta)*cos(alpha) and sin(beta)>=0.0, this means cos(alpha)<0.0 */
      alpha=PI-alpha;  
    }
    if(alpha<0.0){
      alpha=2.0*PI+alpha;  /* make alpha in [0,2PI] */
    } 

    /* determin gamma now */

    if(fabs(fabs(-xz/sin(beta))-1.0)<1e-5){
      if(-xz/sin(beta)<0.0){
	gamma=PI;
      }else{
	gamma=0.0;
      }
    }else{ 
      gamma=acos(-xz/sin(beta));  /* xz=-cos(gamma)*sin(beta). acos() gieve a value between [0, PI] */
    }
    /* we need sin(gamma) to finally determin gamma. xx=cos(alpha)*cos(beta)*cos(gamma)-sin(alpha)*sin(gamma) */
    tmp1=-(xx-(-xz/sin(beta))*zz*cos(alpha))/sin(alpha); /* tmp1 is sin(gamma) */ 
    if(tmp1<0){
      gamma=2*PI-gamma;
    }

  } /* cos(beta)!=1.0 or -1.0 */

  *alpha_r=alpha;
  *beta_r=beta;
  *gamma_r=gamma;

  if (myid==Host_ID){
    printf("z-axis %10.5f %10.5f %10.5f, x-axis %10.5f %10.5f %10.5f\n",zx,zy,zz,xx,xy,xz);
    printf("y-axis %10.5f %10.5f %10.5f\n",yx,yy,yz);
    printf("Euler Angles are %10.5f, %10.5f, %10.5f.(in degree)\n",
            alpha/PI*180.0,beta/PI*180.0,gamma/PI*180.0);
    printf("Euler Angles are %10.5f, %10.5f, %10.5f.(in rad)\n",alpha,beta,gamma);
  }

} /* end of Get_Euler_Rotation_Angle */ 


/* hmweng */

int Calc_Factorial(int arg)
{

  /* calculate Factorial of arg. (arg)! */
  int result,n;

  result = 1;

  if(arg<0){
    printf("Error. For Calc_Factorial, positive integer is needed!\n");
    exit(0);
  }

  if(arg==0 || arg==1){
    result=1;
  }

  else{
    for(n=arg;n>0;n--){
      result=n*result;
    }
  }

  return result;
}



/* hmweng */
void Get_Rotational_Matrix(double alpha, double beta, double gamma, int L, double tmpRotMat[7][7])
{
  int j,i,k;
  int m, mp; 
  int jm, jmp, maxk;
  double fac1,tmp1,tmp2,fac2,sumk,dj[2*L+1][2*L+1],tmp3, sumr,sumi;
  dcomplex Dlm[7][7];
  dcomplex RotMat_for_Real_Func[7][7];
  dcomplex Umat[7][7], Utmp[7][7], Umat_inv[7][7];
  int myid;

  /* get MPI ID */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  j=L;  

  for(jmp=0;jmp<7;jmp++){
    for(jm=0;jm<7;jm++){
      Dlm[jmp][jm].r=0.0;
      Dlm[jmp][jm].i=0.0;
      RotMat_for_Real_Func[jmp][jm].r=0.0;
      RotMat_for_Real_Func[jmp][jm].i=0.0;
      Umat[jmp][jm].r=0.0;
      Umat[jmp][jm].i=0.0;
      Umat_inv[jmp][jm].r=0.0;
      Umat_inv[jmp][jm].i=0.0;
      tmpRotMat[jmp][jm]=0.0;
      if(jmp==jm){
	Dlm[jmp][jm].r=1.0;
	RotMat_for_Real_Func[jmp][jm].r=1.0;
      }
    }
  }
 

  for(jmp=0;jmp<2*j+1;jmp++){
    for(jm=jmp;jm<2*j+1;jm++){
      mp=j-jmp; /* mp goes from  j to -j */
      m=j-jm;   /* m goes from mp to -j to satisfy condition mp>=m */

      tmp1=(double)(Calc_Factorial(j+m)*Calc_Factorial(j-m)*Calc_Factorial(j+mp)*Calc_Factorial(j-mp));

      fac1=sqrt(tmp1);
      if((j-mp)>(j+m)){
	maxk=j+m;
      }else{
	maxk=j-mp;
      }  

      sumk=0.0;
      for(k=0;k<=maxk;k++){
	tmp1=(double)(Calc_Factorial(j-mp-k)*Calc_Factorial(j+m-k)*Calc_Factorial(k+mp-m)*Calc_Factorial(k));

	if(fabs(cos(beta/2.0))<1e-8 && 2*j+m-mp-2*k==0){
	  tmp2=1.0;
	}else{
	  tmp2=exp(log(fabs(cos(beta/2.0)))*(double)(2*j+m-mp-2*k));
	}
	if(fabs(sin(beta/2.0))<1e-8 && 2*k+mp-m==0){
	  tmp3=1.0;
	}else{
	  tmp3=exp(log(fabs(sin(beta/2.0)))*(double)(2*k+mp-m));
	}
	fac2=tmp2*tmp3/tmp1;
	if((k+mp-m)%2==0){
	  sumk=sumk+fac2;
	}else{
	  sumk=sumk-fac2;
	} 
      }/* sum over all k */
      dj[jmp][jm]=sumk*fac1;
      tmp1= cos((double)mp*alpha+(double)m*gamma);
      tmp2=-sin((double)m*gamma+(double)mp*alpha);
      Dlm[jmp][jm].r=dj[jmp][jm]*tmp1;
      Dlm[jmp][jm].i=dj[jmp][jm]*tmp2;
    } /* m */
  }/* m' */

   /* for those when m'<m */
  for(jmp=0;jmp<2*j+1;jmp++){
    for(jm=0;jm<jmp;jm++){
      mp=j-jmp; /* mp runs from j-1 to -j+1 */
      m=j-jm;  /* m runs from mp+1 to j to satisfy mp<m */
      if((mp-m)%2==0){
	dj[jmp][jm]=dj[jm][jmp];
      }else{
	dj[jmp][jm]=-dj[jm][jmp];
      }
      tmp1= cos((double)mp*alpha)*cos((double)m*gamma)-sin((double)mp*alpha)*sin((double)m*gamma);
      tmp2=-cos((double)mp*alpha)*sin((double)m*gamma)-sin((double)mp*alpha)*cos((double)m*gamma);
      Dlm[jmp][jm].r=dj[jmp][jm]*tmp1;
      Dlm[jmp][jm].i=dj[jmp][jm]*tmp2;
    }
  }

  if (0){
    for(jmp=0;jmp<2*j+1;jmp++){
      for(jm=0;jm<2*j+1;jm++){
	printf("(%10.6f %10.6f) ",Dlm[jmp][jm].r,Dlm[jmp][jm].i);
      }
      printf("\n");
    }
    if(j==1){
      printf("Compare with those in Rose's books:\n");
      printf("(%10.6f %10.6f) ",(cos(alpha)*cos(gamma)-sin(alpha)*sin(gamma))*(1.0+cos(beta))/2.0,
	     (-cos(alpha)*sin(gamma)-sin(alpha)*cos(gamma))*(1.0+cos(beta))/2.0);     

      printf("(%10.6f %10.6f) ",-cos(alpha)*sin(beta)/sqrt(2.0),sin(alpha)*sin(beta)/sqrt(2.0));

      printf("(%10.6f %10.6f) ",(cos(alpha)*cos(gamma)+sin(alpha)*sin(gamma))*(1.0-cos(beta))/2.0,
	     (cos(alpha)*sin(gamma)-sin(alpha)*cos(gamma))*(1.0-cos(beta))/2.0);
      printf("\n");
    
      printf("(%10.6f %10.6f) ",cos(gamma)*sin(beta)/sqrt(2.0),-sin(gamma)*sin(beta)/sqrt(2.0));
    
      printf("(%10.6f %10.6f) ",cos(beta),0.0);

      printf("(%10.6f %10.6f) ",-cos(gamma)*sin(beta)/sqrt(2.0),-sin(gamma)*sin(beta)/sqrt(2.0));
      printf("\n");
      printf("(%10.6f %10.6f) ",(cos(alpha)*cos(gamma)+sin(alpha)*sin(gamma))*(1.0-cos(beta))/2.0,
	     (-cos(alpha)*sin(gamma)+sin(alpha)*cos(gamma))*(1.0-cos(beta))/2.0);

      printf("(%10.6f %10.6f) ",cos(alpha)*sin(beta)/sqrt(2.0),sin(alpha)*sin(beta)/sqrt(2.0));

      printf("(%10.6f %10.6f) ",(cos(alpha)*cos(gamma)-sin(alpha)*sin(gamma))*(1.0+cos(beta))/2.0,
	     (cos(alpha)*sin(gamma)+sin(alpha)*cos(gamma))*(1.0+cos(beta))/2.0);
      printf("\n");
    }else if(j==2){
      printf("for Comparision:\n");
      /* m'= 2 */
      printf("(%10.6f %10.6f) ", (1.0+cos(beta))*(1.0+cos(beta))/4.0*cos(2.0*alpha+2.0*gamma),-(1.0+cos(beta))*(1.0+cos(beta))/4.0*sin(2.0*alpha+2.0*gamma));
      printf("(%10.6f %10.6f) ", -0.5*sin(beta)*(1.0+cos(beta))*cos(2.0*alpha+gamma),0.5*sin(beta)*(1.0+cos(beta))*sin(2.0*alpha+gamma));
      printf("(%10.6f %10.6f) ", sqrt(6.0)/4.0*sin(beta)*sin(beta)*cos(2.0*alpha),-sqrt(6.0)/4.0*sin(beta)*sin(beta)*sin(2.0*alpha));
      printf("(%10.6f %10.6f) ", 0.5*sin(beta)*(cos(beta)-1.0)*cos(2.0*alpha-gamma),-0.5*sin(beta)*(cos(beta)-1.0)*sin(2.0*alpha-gamma));
      printf("(%10.6f %10.6f)\n", (1.0-cos(beta))*(1.0-cos(beta))/4.0*cos(2.0*alpha-2.0*gamma),-(1.0-cos(beta))*(1.0-cos(beta))/4.0*sin(2.0*alpha-2.0*gamma));
      /* m'= 1 */     
      printf("(%10.6f %10.6f) ",0.5*sin(beta)*(1.0+cos(beta))*cos(alpha+2.0*gamma),-0.5*sin(beta)*(1.0+cos(beta))*sin(alpha+2.0*gamma));
      printf("(%10.6f %10.6f) ", ((1.0+cos(beta))*(1.0+cos(beta))/4.0-3.0/4.0*sin(beta)*sin(beta))*cos(alpha+gamma),-((1.0+cos(beta))*(1.0+cos(beta))/4.0-3.0/4.0*sin(beta)*sin(beta))*sin(alpha+gamma));
      printf("(%10.6f %10.6f) ", -sqrt(6.0)/4.0*sin(2.0*beta)*cos(alpha), sqrt(6.0)/4.0*sin(2.0*beta)*sin(alpha));
      printf("(%10.6f %10.6f) ",-(cos(2.0*beta)-cos(beta))*0.5*cos(alpha-gamma),(cos(2.0*beta)-cos(beta))*0.5*sin(alpha-gamma));
      printf("(%10.6f %10.6f)\n", 0.5*sin(beta)*(cos(beta)-1.0)*cos(alpha-2.0*gamma),-0.5*sin(beta)*(cos(beta)-1.0)*sin(alpha-2.0*gamma));
      /* m'= 0 */     
      printf("(%10.6f %10.6f) ",sqrt(6.0)/4.0*sin(beta)*sin(beta)*cos(2.0*gamma),-sqrt(6.0)/4.0*sin(beta)*sin(beta)*sin(2.0*gamma));
      printf("(%10.6f %10.6f) ", sqrt(6.0)/4.0*sin(2.0*beta)*cos(gamma),-sqrt(6.0)/4.0*sin(2.0*beta)*sin(gamma));
      printf("(%10.6f %10.6f) ", 0.5*(3.0*cos(beta)*cos(beta)-1.0),0.0);
      printf("(%10.6f %10.6f) ", -sqrt(6.0)/4.0*sin(2.0*beta)*cos(-gamma),sqrt(6.0)/4.0*sin(2.0*beta)*sin(-gamma));
      printf("(%10.6f %10.6f)\n", sqrt(6.0)/4.0*sin(beta)*sin(beta)*cos(-2.0*gamma),-sqrt(6.0)/4.0*sin(beta)*sin(beta)*sin(-2.0*gamma));
      /* m'=-1 */
      printf("(%10.6f %10.6f) ", -0.5*sin(beta)*(cos(beta)-1.0)*cos(-1.0*alpha+2.0*gamma), 0.5*sin(beta)*(cos(beta)-1.0)*sin(-1.0*alpha+2.0*gamma));
      printf("(%10.6f %10.6f) ",-(cos(2.0*beta)-cos(beta))*0.5*cos(-alpha+gamma),(cos(2.0*beta)-cos(beta))*0.5*sin(-alpha+gamma));
      printf("(%10.6f %10.6f) ", sqrt(6.0)/4.0*sin(2.0*beta)*cos(-alpha),-sqrt(6.0)/4.0*sin(2.0*beta)*sin(-alpha));
      printf("(%10.6f %10.6f) ", (cos(2.0*beta)+cos(beta))*0.5*cos(-alpha-gamma),-(cos(2.0*beta)+cos(beta))*0.5*sin(-alpha-gamma));
      printf("(%10.6f %10.6f)\n", -0.5*sin(beta)*(1.0+cos(beta))*cos(-1.0*alpha-2.0*gamma),0.5*sin(beta)*(1.0+cos(beta))*sin(-1.0*alpha-2.0*gamma));
      /* m'=-2 */
      printf("(%10.6f %10.6f) ", (1.0-cos(beta))*(1.0-cos(beta))/4.0*cos(-2.0*alpha+2.0*gamma),-(1.0-cos(beta))*(1.0-cos(beta))/4.0*sin(-2.0*alpha+2.0*gamma));
      printf("(%10.6f %10.6f) ", -0.5*sin(beta)*(cos(beta)-1.0)*cos(-2.0*alpha+1.0*gamma),0.5*sin(beta)*(cos(beta)-1.0)*sin(-2.0*alpha+1.0*gamma));
      printf("(%10.6f %10.6f) ", sqrt(6.0)/4.0*sin(beta)*sin(beta)*cos(-2.0*alpha),-sqrt(6.0)/4.0*sin(beta)*sin(beta)*sin(-2.0*alpha));
      printf("(%10.6f %10.6f) ", 0.5*sin(beta)*(1.0+cos(beta))*cos(-2.0*alpha-1.0*gamma),-0.5*sin(beta)*(1.0+cos(beta))*sin(-2.0*alpha-1.0*gamma));
      printf("(%10.6f %10.6f)\n", (1.0+cos(beta))*(1.0+cos(beta))/4.0*cos(-2.0*alpha-2.0*gamma),-(1.0+cos(beta))*(1.0+cos(beta))/4.0*sin(-2.0*alpha-2.0*gamma));
    }
    
  }
  /* The rotation matrix connecting real function is defined as M=U^(-1)*Dlm^(T)*U, U is the transfer matrix
     from real to imaginary function for orbitals
     p1         px
     p0   =  U* py
     p-1        pz

     d2           dz2
     d1           dx2-y2
     d0    =  U*  dxy
     d-1          dxz
     d-2          dyz

     px'         px
     py'  =  M * py
     pz'         pz

     dz2'          dz2
     dx2-y2'       dx2-y2
     dxy'    = M * dxy
     dxz'          dxz
     dyz'          dyz
     Here ' means those in the rotated coordinate, without ' means those in original coordinate.
  */ 
  switch(j){
  case 1:
    Umat[0][0].r=-1.0/sqrt(2.0);
    Umat[0][0].i=0.0;

    Umat[0][1].r=0.0;
    Umat[0][1].i=-1.0/sqrt(2.0);    

    Umat[0][2].r=0.0;
    Umat[0][2].i=0.0;

    Umat[1][0].r=0.0;
    Umat[1][0].i=0.0;

    Umat[1][1].r=0.0;
    Umat[1][1].i=0.0;

    Umat[1][2].r=1.0;
    Umat[1][2].i=0.0;

    Umat[2][0].r=1.0/sqrt(2.0);
    Umat[2][0].i=0.0;

    Umat[2][1].r=0.0;
    Umat[2][1].i=-1.0/sqrt(2.0);

    Umat[2][2].r=0.0;
    Umat[2][2].i=0.0;
    break;
  case 2:
    Umat[0][0].r=0.0;
    Umat[0][0].i=0.0;

    Umat[0][1].r=1.0/sqrt(2.0);
    Umat[0][1].i=0.0;

    Umat[0][2].r=0.0;
    Umat[0][2].i=1.0/sqrt(2.0);

    Umat[0][3].r=0.0;
    Umat[0][3].i=0.0;
     
    Umat[0][4].r=0.0;
    Umat[0][4].i=0.0;

    Umat[1][0].r=0.0;
    Umat[1][0].i=0.0;
     
    Umat[1][1].r=0.0;
    Umat[1][1].i=0.0;

    Umat[1][2].r=0.0;
    Umat[1][2].i=0.0;

    Umat[1][3].r=-1.0/sqrt(2.0);
    Umat[1][3].i=0.0;
     
    Umat[1][4].r=0.0;
    Umat[1][4].i=-1.0/sqrt(2.0);

    Umat[2][0].r=1.0;
    Umat[2][0].i=0.0;
  
    Umat[2][1].r=0.0;
    Umat[2][1].i=0.0;

    Umat[2][2].r=0.0;
    Umat[2][2].i=0.0;

    Umat[2][3].r=0.0;
    Umat[2][3].i=0.0;

    Umat[2][4].r=0.0;
    Umat[2][4].i=0.0;

    Umat[3][0].r=0.0;
    Umat[3][0].i=0.0;

    Umat[3][1].r=0.0;
    Umat[3][1].i=0.0;

    Umat[3][2].r=0.0;
    Umat[3][2].i=0.0;

    Umat[3][3].r=1.0/sqrt(2.0);
    Umat[3][3].i=0.0;

    Umat[3][4].r=0.0;
    Umat[3][4].i=-1.0/sqrt(2.0);

    Umat[4][0].r=0.0;
    Umat[4][0].i=0.0;

    Umat[4][1].r=1.0/sqrt(2.0);
    Umat[4][1].i=0.0;

    Umat[4][2].r=0.0;
    Umat[4][2].i=-1.0/sqrt(2.0);

    Umat[4][3].r=0.0;
    Umat[4][3].i=0.0;

    Umat[4][4].r=0.0;
    Umat[4][4].i=0.0;
    break;

  case 3:

    break;

  }

  for(jmp=0;jmp<2*j+1;jmp++){
    for(jm=0;jm<2*j+1;jm++){
      Umat_inv[jmp][jm].r=Umat[jm][jmp].r;
      Umat_inv[jmp][jm].i=-Umat[jm][jmp].i;
    }
  }

  /* U^{-1}*Dlm^{T} ==> Utmp */
  for(jmp=0;jmp<2*j+1;jmp++){
    for(jm=0;jm<2*j+1;jm++){
      sumr=0.0; sumi=0.0;
      for(i=0;i<2*j+1;i++){
	sumr=sumr+Umat_inv[jmp][i].r*Dlm[jm][i].r-Umat_inv[jmp][i].i*Dlm[jm][i].i;
	sumi=sumi+Umat_inv[jmp][i].r*Dlm[jm][i].i+Umat_inv[jmp][i].i*Dlm[jm][i].r;
      }
      Utmp[jmp][jm].r=sumr;
      Utmp[jmp][jm].i=sumi;
    }
  }

  /* Utmp * U */
  for(jmp=0;jmp<2*j+1;jmp++){
    for(jm=0;jm<2*j+1;jm++){
      sumr=0.0; sumi=0.0;
      for(i=0;i<2*j+1;i++){
	sumr=sumr+Utmp[jmp][i].r*Umat[i][jm].r-Utmp[jmp][i].i*Umat[i][jm].i;
	sumi=sumi+Utmp[jmp][i].r*Umat[i][jm].i+Utmp[jmp][i].i*Umat[i][jm].r;
      }
      RotMat_for_Real_Func[jmp][jm].r=sumr;
      RotMat_for_Real_Func[jmp][jm].i=sumi;
    }
  }

  for(jmp=0;jmp<2*j+1;jmp++){
    for(jm=0;jm<2*j+1;jm++){

      if(fabs(RotMat_for_Real_Func[jmp][jm].i)>1e-8){

        if (myid==Host_ID){
	  printf("ERROR:WF The rotational matrix should be real.mp=%i,m=%i\n",j-jmp,j-jm);
        }

        MPI_Finalize();
        exit(0);
      }

      tmpRotMat[jmp][jm]=RotMat_for_Real_Func[jmp][jm].r;
    }
  }

  if(myid==Host_ID && 0<level_stdout){
    printf("Final matrix for real functional rotation:\n");
    for(jmp=0;jmp<2*j+1;jmp++){
      for(jm=0;jm<2*j+1;jm++){
	printf("%10.6f  ",RotMat_for_Real_Func[jmp][jm].r);
      }
      printf("\n");
    }
    printf("\n");
  }

} /* end of Get_Rotational_Matrix */






