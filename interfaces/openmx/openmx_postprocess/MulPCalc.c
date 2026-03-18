#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Inputtools.h"
#include "Tools_BandCalc.h"

/* Added by N. Yamaguchi ***/
#include "read_scfout.h"
/* ***/

int main(int argc, char *argv[]){
  FILE *fp,*fp1,*fp2;
  char c;
  int i,j,k,l;
  double d0,d1,d2,d3;

  int Atom_Num, Data_size;
  int l_sum, l_max, l_min, *Data_l;
  double Pxyz[3], MulP_VecScale[3];
  int Num_EA, switch_deg, Num_EB;
  int *EA, *Extract_Atom;
  int Data_Reduction;

  double *line_k;  //Band
  double **Data_k, *Data_kpath, *Data_E, ***Data_atomMulP;
  double **sum_atomMulP;
  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2

  int i_vec[20];           // input variable
  int *i_vec2;             // input variable
  char *s_vec[20];         // input variable
  double r_vec[20];        // input variable

  int fl_data  = 0;
  char fname[256], fname_1[256], fname_aMulP[256], fname_out[256];

  // ### INPUT_FILE ##########################################
  // check argv 
  if (argc==1){
    printf("\nCould not find an input file.\n\n");

    /* Disabled by N. Yamaguchi ***
    exit(0);
    * ***/

    /* Added by N. Yamaguchi ***/
    return 0;
    /* ***/

  }
  sprintf(fname,"%s",argv[1]);
  input_open(fname);
  input_string("Filename.atomMulP",fname_aMulP,"default");
  input_string("Filename.xyzdata",fname_out,"default");

  s_vec[0]="FermiSurface";
  s_vec[1]="CircularSearch";
  s_vec[2]="BandDispersion";

  /* Disabled by N. Yamaguchi ***
     s_vec[3]="EigenGrid";
   * ***/

  /* Added by N. Yamaguchi ***/
  s_vec[3]="GridCalc";
  /* ***/

  /* Disabled by N. Yamaguchi ***
  s_vec[4]="TriMesh";
  * ***/

  /* Added by N. Yamaguchi ***/
  s_vec[4]="FermiLoop";
  /* ***/

  /* Disabled by N. Yamaguchi ***
  s_vec[5]="MulPonly";
  * ***/

  /* Added by N. Yamaguchi ***/
  s_vec[5]="MulPOnly";
  /* ***/

  i_vec[0]=1;  i_vec[1]=2;  i_vec[2]=3;   i_vec[3]=4;   i_vec[4]=5;  i_vec[5]=6;
  i = input_string2int("Calc.Type", &Calc_Type, 6, s_vec, i_vec);
  printf("\nStart \"%s\" Calculation (%d).  \n", s_vec[Calc_Type-1], Calc_Type);

  r_vec[0]=1.0; r_vec[1]=1.0; r_vec[2]=1.0;
  input_doublev("MulP.Vec.Scale",3, MulP_VecScale, r_vec);
  input_int("Num.of.Extract.Atom",&Num_EA,1);
  if (Num_EA > 0){
    i_vec2 = (int*)malloc(sizeof(int)*(Num_EA+1));
    Extract_Atom = (int*)malloc(sizeof(int)*(Num_EA+1));
    for(i=0;i<Num_EA;i++) i_vec2[i]=i+1;
    input_intv("Extract.Atom",Num_EA,Extract_Atom,i_vec2);
    free(i_vec2);
  }
  input_int("Data.Reduction",&Data_Reduction,1);

  /* Added by N. Yamaguchi ***/
  int sw=i<0 ? 0: 1;
  char fname_out0[256];
  double k_CNT[3];
  int plane_3mesh;
  int Band_Nkpath;
  input_string("Filename.outdata",fname_out0,"default");
  r_vec[0]=0.0; r_vec[1]=0.0; r_vec[2]=0.0;
  input_doublev("Search.kCentral",3,k_CNT,r_vec);
  input_int("Calc.Type.3mesh",&plane_3mesh,1);
  r_vec[0]=-0.5; r_vec[1]=0.5;
  input_doublev("Energy.Range",2,E_Range,r_vec);
  if (E_Range[0]>E_Range[1]){
    d0 = E_Range[0];
    E_Range[0] = E_Range[1];
    E_Range[1] = d0;
  }//if
  input_int("Band.Nkpath",&Band_Nkpath,0);
  if (Band_Nkpath>0) {

    Band_N_perpath=(int*)malloc(sizeof(int)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++) Band_N_perpath[i] = 0;

    Band_kpath = (double***)malloc(sizeof(double**)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++){
      Band_kpath[i] = (double**)malloc(sizeof(double*)*3);
      for (j=0; j<3; j++){
	Band_kpath[i][j] = (double*)malloc(sizeof(double)*4);
	for (k=0; k<4; k++) Band_kpath[i][j][k] = 0.0;
      }//for(i)
    }//for(j)
    Band_kname = (char***)malloc(sizeof(char**)*(Band_Nkpath+1));
    for (i=0; i<(Band_Nkpath+1); i++){
      Band_kname[i] = (char**)malloc(sizeof(char*)*3);
      for (j=0; j<3; j++){
	Band_kname[i][j] = (char*)malloc(sizeof(char)*16);
      }//for(i)
    }//for(j)
    if( (fp=input_find("<Band.kpath"))  != NULL) {
      for (i=1; i<=Band_Nkpath; i++) {
	fscanf(fp,"%d %lf %lf %lf %lf %lf %lf %s %s",
	    &Band_N_perpath[i]  ,
	    &Band_kpath[i][1][1], &Band_kpath[i][1][2], &Band_kpath[i][1][3],
	    &Band_kpath[i][2][1], &Band_kpath[i][2][2], &Band_kpath[i][2][3],
	    Band_kname[i][1], Band_kname[i][2]);
      }//for(Band_Nkpath)
      if ( ! input_last("Band.kpath>") ) { // format error
	printf("Format error near Band.kpath>\n");
	return -1;
      }
    }//else
    else { // format error
      printf("<Band.kpath is necessary.\n");
      return -1;
    }//else
  }//if(Nkpath>0)
  /* ***/

  input_close();

  /* Disabled by N. Yamaguchi ***
     if (i < 0)  exit(0);
   * ***/

  /* Added by N. Yamaguchi ***/
  if (!sw){
    return 0;
  }
  if (Calc_Type==1){
    puts("\"FermiSurface\" is not supported in this version of MulPCalc.");
    return 0;
  } else if (Calc_Type==2){
    puts("\"CircularSearch\" is not supported in this version of MulPCalc.");
    return 0;
  }
  /* ***/

  // ### Get Calculation Data ##############################
  if((fp = fopen(fname_aMulP,"r")) != NULL){ 
    printf("\nInput filename is \"%s\"  \n\n", fname_aMulP);
  }else{
    printf("Cannot open MulP File. \"%s\" is not found.\n" ,fname_aMulP);

    /* Disabled by N. Yamaguchi ***
    exit(0);
    * ***/

    /* Added by N. Yamaguchi ***/
    return 0;
    /* ***/

  }

  fscanf(fp,"%d", &Data_size);
  fscanf(fp,"%d", &Atom_Num);

  /* Added by N. Yamaguchi ***/
  double *line_Nk;
  if (Calc_Type == 3) {
    fscanf(fp, "%lf%lf%lf", &rtv[1][1], &rtv[2][1], &rtv[3][1]);
    fscanf(fp, "%lf%lf%lf", &rtv[1][2], &rtv[2][2], &rtv[3][2]);
    fscanf(fp, "%lf%lf%lf", &rtv[1][3], &rtv[2][3], &rtv[3][3]);
    line_Nk=(double*)malloc(sizeof(double)*(Band_Nkpath+1));
    line_Nk[0] = 0;
    for (i=1; i<(Band_Nkpath+1); i++){
      line_Nk[i] = kgrid_dist(Band_kpath[i][1][1], Band_kpath[i][2][1], Band_kpath[i][1][2], Band_kpath[i][2][2], Band_kpath[i][1][3], Band_kpath[i][2][3]) +line_Nk[i-1]; 
      printf("line_Nk[%d]:%10.6f\n",i,line_Nk[i]);
      printf("(%9.6f, %9.6f, %9.6f ) -> (%9.6f, %9.6f, %9.6f )\n", Band_kpath[i][1][1], Band_kpath[i][1][2], Band_kpath[i][1][3], Band_kpath[i][2][1], Band_kpath[i][2][2], Band_kpath[i][2][3]);
    }
  }
  /* ***/

  EA = (int*)malloc(sizeof(int)*(Atom_Num+1));
  for(i=0;i<=Atom_Num;i++) EA[i] = 0;
  if (Num_EA > 0){
    printf("Calculate Atom:");
    for(i=0;i<Num_EA;i++){
      /*kotaka*/
      if (Extract_Atom[i]>0 && Extract_Atom[i]<=Atom_Num){
	EA[Extract_Atom[i]-1] = 1;
	printf("%4d ",Extract_Atom[i]);
      }else{
	printf("%4d is illegal value.\n",Extract_Atom[i]);
	printf("kotaka %4d\n",Atom_Num);
      }     
    } printf("\n");
    printf("MulP Scale:(%10.6lf %10.6lf %10.6lf)\n",MulP_VecScale[0],MulP_VecScale[1],MulP_VecScale[2]);  
  }//if
  Data_k = (double**)malloc(sizeof(double*)*(Data_size+1));
  for(j=0;j<=Data_size;j++){
    Data_k[j] = (double*)malloc(sizeof(double)*3);
  }
  Data_l = (int*)malloc(sizeof(int)*(Data_size+1));
  Data_E = (double*)malloc(sizeof(double)*(Data_size+1));
  Data_atomMulP = (double***)malloc(sizeof(double**)*(Data_size+1));
  for(j=0;j<=Data_size;j++){
    Data_atomMulP[j] = (double**)malloc(sizeof(double*)*(Atom_Num));
    for(i=0;i<Atom_Num;i++){
      Data_atomMulP[j][i] = (double*)malloc(sizeof(double)*4);
    }
  }
  line_k = (double*)malloc(sizeof(double)*(Data_size+1));

  for(j=0;j<Data_size;j++){
    for(i=0;i<3;i++) fscanf(fp,"%lf",&Data_k[j][i]);
    fscanf(fp,"%d",&Data_l[j]);
    fscanf(fp,"%lf",&Data_E[j]);
    for(i=0;i<Atom_Num;i++){
      for(k=0;k<4;k++) fscanf(fp,"%lf",&Data_atomMulP[j][i][k]);
    }//i 
    if (Calc_Type == 3) fscanf(fp,"%lf",&line_k[j]);
  }//j

  fclose(fp);
  l_max = Data_l[0];
  l_min = Data_l[0];
  for(j=0;j<Data_size;j++){
    if (l_min>Data_l[j]) l_min=Data_l[j];
    if (l_max<Data_l[j]) l_max=Data_l[j];
  }
  sum_atomMulP = (double**)malloc(sizeof(double*)*(Data_size+1));
  for(j=0;j<=Data_size;j++){
    sum_atomMulP[j] = (double*)malloc(sizeof(double)*7);
  }

  //############ PRINT PART #########################
  for(j=0;j<Data_size;j++){
    sum_atomMulP[j][0] = 0.0;  sum_atomMulP[j][1] = 0.0;  sum_atomMulP[j][2] = 0.0;
    sum_atomMulP[j][3] = 0.0;  sum_atomMulP[j][4] = 0.0;  sum_atomMulP[j][5] = 0.0;  sum_atomMulP[j][6] = 0.0;
  }  k = 0;

  strcpy(fname_1,fname_out);
  strcat(fname_1,".MulPop");

  fp1 = fopen(fname_1,"w");

  /* Disabled by N. Yamaguchi ***
     fprintf(fp1,"# kx[Ang-1]   ky[Ang-1]   kz[Ang-1]   Eig[eV]     Num_Ele     Spin_x      Spin_y      Spin_z  \n");
   * ***/

  /* Added by N. Yamaguchi ***/
  fputs("# kx[/Ang.]   ky[/Ang.]   kz[/Ang.]   Eig[eV]     Num_Ele     N_{alpha}   N_{beta}    Spin_x      Spin_y      Spin_z", fp1);
  if (Calc_Type==3){
    fputs("      k1D[/Ang.]", fp1);
  }
  fputs("\n", fp1);
  /* ***/

  for(l=l_min;l<=l_max;l++){
    strcpy(fname_1,fname_out);
    strcat(fname_1,".MulPop");
    name_Nband(fname_1,"_",l);

    fp = fopen(fname_1,"w");

    /* Disabled by N. Yamaguchi ***
       fprintf(fp,"# kx[Ang-1]   ky[Ang-1]   kz[Ang-1]   Eig[eV]     Num_Ele     Spin_x      Spin_y      Spin_z  \n");
     * ***/

    /* Added by N. Yamaguchi ***/
    fputs("# kx[/Ang.]   ky[/Ang.]   kz[/Ang.]   Eig[eV]     Num_Ele     N_{alpha}   N_{beta}    Spin_x      Spin_y      Spin_z", fp);
    if (Calc_Type==3){
      fputs("      k1D[/Ang.]", fp);
    }
    fputs("\n", fp);
    /* ***/

    for(j=0;j<Data_size;j++){
      if (Data_l[j] == l){
	k++;
	sum_atomMulP[j][0]=Data_k[j][0];  sum_atomMulP[j][1]=Data_k[j][1];  sum_atomMulP[j][2]=Data_k[j][2];
	sum_atomMulP[j][3]=0;      sum_atomMulP[j][4]=0;      sum_atomMulP[j][5]=0;      sum_atomMulP[j][6]=0;
	for(i=0;i<Atom_Num;i++){
	  if (EA[i]==1){
	    sum_atomMulP[j][3]+= Data_atomMulP[j][i][0]; //Re11
	    sum_atomMulP[j][4]+= Data_atomMulP[j][i][1];  //Re22
	    sum_atomMulP[j][5]+= Data_atomMulP[j][i][2];  //Re12
	    sum_atomMulP[j][6]+= Data_atomMulP[j][i][3];  //Im12
	  }//if
	}//i
	if (k%Data_Reduction == 0  || Data_Reduction == 0){
	  Re11 = sum_atomMulP[j][3];       Re22 = sum_atomMulP[j][4];
	  Re12 = sum_atomMulP[j][5];       Im12 = sum_atomMulP[j][6];
	  EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndw, Ntheta, Nphi );
	  fprintf(fp, "%10.6lf  " ,sum_atomMulP[j][0]/BohrR);  fprintf(fp1, "%10.6lf  " ,sum_atomMulP[j][0]/BohrR);
	  fprintf(fp, "%10.6lf  " ,sum_atomMulP[j][1]/BohrR);  fprintf(fp1, "%10.6lf  " ,sum_atomMulP[j][1]/BohrR);
	  fprintf(fp, "%10.6lf  " ,sum_atomMulP[j][2]/BohrR);  fprintf(fp1, "%10.6lf  " ,sum_atomMulP[j][2]/BohrR);
	  fprintf(fp, "%10.6lf  ", Data_E[j]);  fprintf(fp1, "%10.6lf  ", Data_E[j]);

	  fprintf(fp, "%10.6lf  " ,Nup[0] +Ndw[0]);

	  /* Added by N. Yamaguchi ***/
	  fprintf(fp, "%10.6lf  " , Re11);
	  fprintf(fp, "%10.6lf  " , Re22);
	  /* ***/

	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]) *MulP_VecScale[0] );
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]) *MulP_VecScale[1] );
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *cos(Ntheta[0]) *MulP_VecScale[2] );
	  fprintf(fp1, "%10.6lf  " ,Nup[0] +Ndw[0]);

	  /* Added by N. Yamaguchi ***/
	  fprintf(fp1, "%10.6lf  " , Re11);
	  fprintf(fp1, "%10.6lf  " , Re22);
	  /* ***/

	  fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]) *MulP_VecScale[0] );
	  fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]) *MulP_VecScale[1] );
	  fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *cos(Ntheta[0]) *MulP_VecScale[2] );
	  if(Calc_Type == 3){  fprintf(fp, "%10.6lf  " ,line_k[j]);  fprintf(fp1, "%10.6lf  " ,line_k[j]);  }
	  fprintf(fp, "\n");           fprintf(fp1, "\n"); 
	}//if(Reduction)
      }//if(l)
    }//j
    fclose(fp);

    /* Added by N. Yamaguchi ***/
    if (Calc_Type==3){
      fputs("\n", fp1);
    }
    /* ***/

  }//l
  fclose(fp1);

  /* Added by N. Yamaguchi ***/
  if (Calc_Type==5){
    char fname_gp[256];
    strcpy(fname_gp,fname_out);
    strcat(fname_gp,".plotexample");
    fp1=fopen(fname_gp, "w");
    fputs("Bohr2Ang=1./0.529177249\n", fp1);
    if (plane_3mesh==1){
      fprintf(fp1, "xc=%f\n", k_CNT[0]);
      fprintf(fp1, "yc=%f\n", k_CNT[1]);
    } else if (plane_3mesh==2){
      fprintf(fp1, "xc=%f\n", k_CNT[1]);
      fprintf(fp1, "yc=%f\n", k_CNT[2]);
    } else if (plane_3mesh==3){
      fprintf(fp1, "xc=%f\n", k_CNT[2]);
      fprintf(fp1, "yc=%f\n", k_CNT[0]);
    }
    fputs("tics=0.2\n", fp1);
    fputs("linewidth=3\n", fp1);
    fputs("set size ratio -1\n", fp1);
    fputs("set encoding iso\n", fp1);
    fputs("set xlabel 'k_x (/\\305)'\n", fp1);
    fputs("set ylabel 'k_y (/\\305)'\n", fp1);
    fputs("#set xtics tics\n", fp1);
    fputs("#set ytics tics\n", fp1);
    fputs("set label 1 center at first xc, yc '+' front\n", fp1);
    fputs("#set label 2 center at first xc, yc-0.1 '{/Symbol G}' front\n", fp1);
    for (l=l_min; l<=l_max; l++){
      if (l>l_min){
	fputs("re", fp1);
      }
      if (plane_3mesh==1){
	fprintf(fp1, "plot '%s.FermiSurf_%d' using ($1*Bohr2Ang):($2*Bohr2Ang) with lines notitle linewidth linewidth\n", fname_out0, l);
	fprintf(fp1, "#replot '%s.MulPop_%d' using 1:2:8:9 with vectors notitle linewidth linewidth\n", fname_out, l);
      } else if (plane_3mesh==2){
	fprintf(fp1, "plot '%s.FermiSurf_%d' using ($2*Bohr2Ang):($3*Bohr2Ang) with lines notitle linewidth linewidth\n", fname_out0, l);
	fprintf(fp1, "#replot '%s.MulPop_%d' using 2:3:9:10 with vectors notitle linewidth linewidth\n", fname_out, l);
      } else if (plane_3mesh==3){
	fprintf(fp1, "plot '%s.FermiSurf_%d' using ($3*Bohr2Ang):($1*Bohr2Ang) with lines notitle linewidth linewidth\n", fname_out0, l);
	fprintf(fp1, "#replot '%s.MulPop_%d' using 3:1:10:8 with vectors notitle linewidth linewidth\n", fname_out, l);
      }
    }
    if (plane_3mesh==1){
      fprintf(fp1, "replot '%s.MulPop' using 1:2:8:9 with vectors notitle linewidth linewidth\n", fname_out);
    } else if (plane_3mesh==2){
      fprintf(fp1, "replot '%s.MulPop' using 2:3:9:10 with vectors notitle linewidth linewidth\n", fname_out);
    } else if (plane_3mesh==3){
      fprintf(fp1, "replot '%s.MulPop' using 3:1:10:8 with vectors notitle linewidth linewidth\n", fname_out);
    }
    fputs("pause -1\n", fp1);
    fclose(fp1);
  } else if (Calc_Type==3){
    char fname_gp[256];
    strcpy(fname_gp, fname_out);
    strcat(fname_gp, ".plotexample");
    fp1=fopen(fname_gp, "w");
    fputs("linewidth=3\n", fp1);
    fputs("scale=0.01\n", fp1);
    fputs("set nokey\n", fp1);
    fputs("set zeroaxis\n", fp1);
    fputs("set mytics 5\n", fp1);
    fputs("set grid\n", fp1);
    fputs("set ylabel 'E-E_F (eV)'\n", fp1);
    fprintf(fp1, "set xrange [%f:%f]\n", line_Nk[0], line_Nk[Band_Nkpath]);
    fprintf(fp1, "set yrange [%f:%f]\n", E_Range[0], E_Range[1]);
    for (i=1; i<=Band_Nkpath; i++) {
      if (i==1){
	fprintf(fp1, "set xtics ('%s' %f", Band_kname[i][1], line_Nk[i-1]);
      }
      if (i==Band_Nkpath){
	fprintf(fp1, ", '%s' %f)\n", Band_kname[i][2], line_Nk[i]);
      } else if (!strcmp(Band_kname[i][2], Band_kname[i+1][1])){
	fprintf(fp1, ", '%s' %f", Band_kname[i+1][1], line_Nk[i]);
      } else {
	fprintf(fp1, ", '%s, %s' %f", Band_kname[i][2], Band_kname[i+1][1], line_Nk[i]);
      }
    }
    fprintf(fp1, "plot '%s.BAND' with lines notitle\n", fname_out0);
    fprintf(fp1, "replot '%s.MulPop' using 11:4:($6>=$7 ? ($6-$7)*scale : 0.0) with circles notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "replot '%s.MulPop' using 11:4:($6>=$7 ? 0.0 : ($7-$6)*scale) with circles notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "#replot '%s.MulPop' using 11:4:($5*scale) with circles notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "#replot '%s.MulPop' using 11:4:($6*scale) with circles notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "#replot '%s.MulPop' using 11:4:($7*scale) with circles notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "#replot '%s.MulPop' using 11:4:8:9 with vectors notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "#replot '%s.MulPop' using 11:4:9:10 with vectors notitle linewidth linewidth\n", fname_out);
    fprintf(fp1, "#replot '%s.MulPop' using 11:4:10:8 with vectors notitle linewidth linewidth\n", fname_out);
    for (l=l_min; l<=l_max; l++){
    fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:($6>=$7 ? ($6-$7)*scale : 0.0) with circles notitle linewidth linewidth\n", fname_out, l);
    fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:($6>=$7 ? 0.0 : ($7-$6)*scale) with circles notitle linewidth linewidth\n", fname_out, l);
      fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:($5*scale) with circles notitle linewidth linewidth\n", fname_out, l);
      fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:($6*scale) with circles notitle linewidth linewidth\n", fname_out, l);
      fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:($7*scale) with circles notitle linewidth linewidth\n", fname_out, l);
      fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:8:9 with vectors notitle linewidth linewidth\n", fname_out, l);
      fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:9:10 with vectors notitle linewidth linewidth\n", fname_out, l);
      fprintf(fp1, "#replot '%s.MulPop_%d' using 11:4:10:8 with vectors notitle linewidth linewidth\n", fname_out, l);
    }
    fputs("pause -1\n", fp1);
    fclose(fp1);
  } else if (Calc_Type==4){
    for(l=l_min;l<=l_max;l++){
      char fname_gp[256];
      strcpy(fname_gp,fname_out);
      name_Nband(fname_gp,".plotexample_",l);
      fp1=fopen(fname_gp, "w");
      fputs("Bohr2Ang=1./0.529177249\n", fp1);
      if (plane_3mesh==1){
	fprintf(fp1, "xc=%f*Bohr2Ang\n", k_CNT1[0]);
	fprintf(fp1, "yc=%f*Bohr2Ang\n", k_CNT1[1]);
      } else if (plane_3mesh==2){
	fprintf(fp1, "xc=%f*Bohr2Ang\n", k_CNT1[1]);
	fprintf(fp1, "yc=%f*Bohr2Ang\n", k_CNT1[2]);
      } else if (plane_3mesh==3){
	fprintf(fp1, "xc=%f*Bohr2Ang\n", k_CNT1[2]);
	fprintf(fp1, "yc=%f*Bohr2Ang\n", k_CNT1[0]);
      }
      fputs("tics=0.2\n", fp1);
      fputs("linewidth=3\n", fp1);
      fputs("set size ratio -1\n", fp1);
      fputs("set encoding iso\n", fp1);
      fputs("set xlabel 'k_x (/\\305)'\n", fp1);
      fputs("set ylabel 'k_y (/\\305)'\n", fp1);
      fputs("set cblabel 'eV'\n", fp1);
      fputs("set pm3d map\n", fp1);
      fputs("set palette defined (-2 'blue', -1 'white', 0 'red')\n", fp1);
      fputs("set pm3d interpolate 5, 5\n", fp1);
      fputs("#set xtics tics\n", fp1);
      fputs("#set ytics tics\n", fp1);
      fputs("set label 1 center at first xc, yc '+' front\n", fp1);
      fputs("#set label 2 center at first xc, yc-0.1 '{/Symbol G}' front\n", fp1);
      if (plane_3mesh==1){
	fprintf(fp1, "splot '%s.EigenMap_%d' using ($1*Bohr2Ang):($2*Bohr2Ang):4 notitle\n", fname_out0, l);
	fprintf(fp1, "replot '%s.MulPop_%d' using 1:2:3:8:9:10 with vectors notitle linetype -1 linewidth linewidth\n", fname_out, l);
      } else if (plane_3mesh==2){
	fprintf(fp1, "splot '%s.EigenMap_%d' using ($2*Bohr2Ang):($3*Bohr2Ang):4 notitle\n", fname_out0, l);
	fprintf(fp1, "replot '%s.MulPop_%d' using 2:3:1:9:10:8 with vectors notitle linetype -1 linewidth linewidth\n", fname_out, l);
      } else if (plane_3mesh==3){
	fprintf(fp1, "splot '%s.EigenMap_%d' using ($3*Bohr2Ang):($1*Bohr2Ang):4 notitle\n", fname_out0, l);
	fprintf(fp1, "replot '%s.MulPop_%d' using 3:1:2:10:8:9 with vectors notitle linetype -1 linewidth linewidth\n", fname_out, l);
      }
      fputs("pause -1\n", fp1);
      fclose(fp1);
    }
  } else if (Calc_Type==6){
    char fname_gp[256];
    strcpy(fname_gp,fname_out);
    strcat(fname_gp,".plotexample");
    fp1=fopen(fname_gp, "w");
    fputs("Bohr2Ang=1./0.529177249\n", fp1);
    fputs("tics=0.2\n", fp1);
    fputs("linewidth=3\n", fp1);
    fputs("set size ratio -1\n", fp1);
    fputs("set encoding iso\n", fp1);
    fputs("set xlabel 'k_x (/\\305)'\n", fp1);
    fputs("set ylabel 'k_y (/\\305)'\n", fp1);
    fputs("#set xtics tics\n", fp1);
    fputs("#set ytics tics\n", fp1);
    if (plane_3mesh==1){
      fprintf(fp1, "plot '%s.MulPop' using 1:2:8:9 with vectors notitle linewidth linewidth\n", fname_out);
    } else if (plane_3mesh==2){
      fprintf(fp1, "plot '%s.MulPop' using 2:3:9:10 with vectors notitle linewidth linewidth\n", fname_out);
    } else if (plane_3mesh==3){
      fprintf(fp1, "plot '%s.MulPop' using 3:1:10:8 with vectors notitle linewidth linewidth\n", fname_out);
    }
    for (l=l_min; l<=l_max; l++){
      if (plane_3mesh==1){
	fprintf(fp1, "#replot '%s.MulPop_%d' using 1:2:8:9 with vectors notitle linewidth linewidth\n", fname_out, l);
      } else if (plane_3mesh==2){
	fprintf(fp1, "#replot '%s.MulPop_%d' using 2:3:9:10 with vectors notitle linewidth linewidth\n", fname_out, l);
      } else if (plane_3mesh==3){
	fprintf(fp1, "#replot '%s.MulPop_%d' using 3:1:10:8 with vectors notitle linewidth linewidth\n", fname_out, l);
      }
    }
    fputs("pause -1\n", fp1);
    fclose(fp1);
  }
  /* ***/

  //############ FREE MALLOC ########################
  free(Extract_Atom);

  free(EA);
  for(j=0;j<=Data_size;j++){
    free(sum_atomMulP[j]);
  } free(sum_atomMulP);
  for(j=0;j<=Data_size;j++)
    free(Data_k[j]);
  free(Data_k);
  free(Data_l);
  free(Data_E);
  for(j=0;j<=Data_size;j++){
    for(i=0;i<Atom_Num;i++){
      free(Data_atomMulP[j][i]);
    }free(Data_atomMulP[j]);
  }free(Data_atomMulP);
  free(line_k);
}
