#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "Inputtools.h"
#include "Tools_BandCalc.h"
#include "read_scfout.h"


char D2Hex(int i){
  if(i<0) return 0;
  else if(i<10) return i+48;
  else if(i<16) return i+55;
  else return 0;
}

int main(int argc, char *argv[]){

  FILE *fp,*fp1,*fp2;
  char c;
  int i,j,j1,k,l;
  double d0,d1,d2,d3;

  int Atom_Num, Data_size;
  int l_sum, l_max, l_min, *Data_l;
  double Pxyz[3], MulP_VecScale[3];
  int Num_EA, Num_EB;
  int *EA, *Extract_Atom;
  int Data_Reduction;

  double *line_k, *line_Nk;  //Band
  double **Data_k, *Data_kpath, *Data_E, ***Data_atomMulP;
  double **sum_atomMulP;
  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2

  int i_vec[20];           // input variable
  int *i_vec2;             // input variable
  char *s_vec[20];         // input variable
  double r_vec[20];        // input variable

  char fname[256], fname_1[256], fname_aMulP[256], fname_out[256];
  char fname_plot[256];

  int iR, iG, iB;
  char ColorCode[7];

  int BandAden_size;
  int ADen_Scale = 2; 
  double ADen_Max, ADen_Min;

  //DEGE
  int switch_dege;
  int *BandTotal;
  double **sum_atomMulP_D, *line_k_D, *Data_E_D;

  // ### INPUT_FILE ##########################################
  // check argv 
  if (argc==1){
    printf("\nCould not find an input file.\n\n");
    exit(0);
  }
  sprintf(fname,"%s",argv[1]);
  input_open(fname);
  input_string("Filename.atomMulP",fname_aMulP,"default");
  input_string("Filename.xyzdata",fname_out,"default");

  s_vec[0]="FermiSurface";
  s_vec[1]="CircularSearch";
  s_vec[2]="BandDispersion";
  s_vec[3]="EigenGrid"; 
  s_vec[4]="TriMesh";
  s_vec[5]="MulPonly";
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
  input_int("Dege",&switch_dege,0);

  //BAND
  r_vec[0]=-2.0; r_vec[1]=1.0;
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

  input_close();
  if (i < 0)  exit(0);

  // ### Get Calculation Data ##############################
  if((fp = fopen(fname_aMulP,"r")) != NULL){ 
    printf("\nInput filename is \"%s\"  \n\n", fname_aMulP);
  }else{
    printf("Cannot open MulP File. \"%s\" is not found.\n" ,fname_aMulP);
    exit(0);
  }

  fscanf(fp,"%d", &Data_size);
  fscanf(fp,"%d", &Atom_Num);
  if (Calc_Type == 3) {
    fscanf(fp," %lf %lf %lf ", &rtv[1][1], &rtv[2][1], &rtv[3][1] );
    fscanf(fp," %lf %lf %lf ", &rtv[1][2], &rtv[2][2], &rtv[3][2] );
    fscanf(fp," %lf %lf %lf ", &rtv[1][3], &rtv[2][3], &rtv[3][3] );
  }
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
	printf("AtomNum %4d\n",Atom_Num);
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

  line_Nk=(double*)malloc(sizeof(double)*(Band_Nkpath+1));
  line_Nk[0] = 0.0;
  for (i=1; i<(Band_Nkpath+1); i++){
    line_Nk[i] = kgrid_dist(Band_kpath[i][1][1], Band_kpath[i][2][1], Band_kpath[i][1][2], Band_kpath[i][2][2], Band_kpath[i][1][3], Band_kpath[i][2][3]) +line_Nk[i-1]; 
    printf("line_Nk[%d]:%10.6lf\n",i,line_Nk[i]);
    printf("%10.6lf-%10.6lf,%10.6lf-%10.6lf,%10.6lf-%10.6lf\n",Band_kpath[i][1][1], Band_kpath[i][2][1], Band_kpath[i][1][2], Band_kpath[i][2][2], Band_kpath[i][1][3], Band_kpath[i][2][3]);
  }

  l_max = Data_l[0];
  l_min = Data_l[0];
  for(j=0;j<Data_size;j++){
    if (l_min>Data_l[j]) l_min=Data_l[j];
    if (l_max<Data_l[j]) l_max=Data_l[j];
  }
  l_cal = l_max-l_min +1;
  if(l_cal<1) l_cal=1;
  printf("l_min:%4d   l_max:%4d   l_cal:%4d \n" ,l_min ,l_max , l_cal);

  BandTotal = (int*)malloc(sizeof(int)*(l_cal));
  for(l=0;l<l_cal;l++) BandTotal[l] = 0.0;

  sum_atomMulP = (double**)malloc(sizeof(double*)*(Data_size+1));
  for(j=0;j<=Data_size;j++){
    sum_atomMulP[j] = (double*)malloc(sizeof(double)*7);
  }
  //############ PRINT PART #########################
  for(j=0;j<Data_size;j++){
    sum_atomMulP[j][0] = 0.0;  sum_atomMulP[j][1] = 0.0;  sum_atomMulP[j][2] = 0.0;
    sum_atomMulP[j][3] = 0.0;  sum_atomMulP[j][4] = 0.0;  sum_atomMulP[j][5] = 0.0;  sum_atomMulP[j][6] = 0.0;
  }
  strcpy(fname_1,fname_out);
  strcat(fname_1,".MulPop");

  fp1 = fopen(fname_1,"w");
  fprintf(fp1,"# kx[Ang-1]   ky[Ang-1]   kz[Ang-1]   Eig[eV]     Num_Ele     Spin_x      Spin_y      Spin_z  \n");

  for(l=l_min;l<=l_max;l++){
    strcpy(fname_1,fname_out);
    strcat(fname_1,".MulPop");
    name_Nband(fname_1,"_",l);

    fp = fopen(fname_1,"w");
    fprintf(fp,"# kx[Ang-1]   ky[Ang-1]   kz[Ang-1]   Eig[eV]     Num_Ele     Spin_x      Spin_y      Spin_z  \n");

    k = 0;
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
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]) *MulP_VecScale[0] );
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]) *MulP_VecScale[1] );
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *cos(Ntheta[0]) *MulP_VecScale[2] );
	  fprintf(fp1, "%10.6lf  " ,Nup[0] +Ndw[0]);
	  fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]) *MulP_VecScale[0] );
	  fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]) *MulP_VecScale[1] );
	  fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *cos(Ntheta[0]) *MulP_VecScale[2] );
	  if(Calc_Type == 3){  fprintf(fp, "%10.6lf  " ,line_k[j]);  fprintf(fp1, "%10.6lf  " ,line_k[j]);  }
	  fprintf(fp, "\n");           fprintf(fp1, "\n"); 
	}//if(Reduction)
      }//if(l)
    }//j
    BandTotal[l-l_min] = k;
    //printf("%4dBand: %4d\n",l,BandTotal[l-l_min]);
    fclose(fp);
  }//l
  fclose(fp1);

  if (Calc_Type == 3 && switch_dege==1){

    sum_atomMulP_D = (double**)malloc(sizeof(double*)*(Data_size+1));
    for(j=0;j<=Data_size;j++){
      sum_atomMulP_D[j] = (double*)malloc(sizeof(double)*7);
    }
    line_k_D = (double*)malloc(sizeof(double)*(Data_size+1));
    Data_E_D = (double*)malloc(sizeof(double)*(Data_size+1));

    strcpy(fname_1,fname_out);
    strcat(fname_1,".MulPop_D");

    fp1 = fopen(fname_1,"w");
    fprintf(fp1,"# kx[Ang-1]   ky[Ang-1]   kz[Ang-1]   Eig[eV]     Num_Ele     Spin_x      Spin_y      Spin_z      line-k  \n");


    BandAden_size=0;
    for(l=l_min;l<=l_max;l+=2){
      if (BandTotal[l-l_min]==BandTotal[l-l_min+1]){

	for(j=0;j<Data_size;j++){
	  if (Data_l[j] == l){

	    for(j1=0;j1<Data_size;j1++){
	      if (Data_l[j1]==l+1 && line_k[j]==line_k[j1] && 
		  sum_atomMulP[j][0]== sum_atomMulP[j1][0] && 
		  sum_atomMulP[j][1]== sum_atomMulP[j1][1] && 
		  sum_atomMulP[j][2]== sum_atomMulP[j1][2]   ){
		sum_atomMulP_D[BandAden_size][0] = sum_atomMulP[j][0];
		sum_atomMulP_D[BandAden_size][1] = sum_atomMulP[j][1];
		sum_atomMulP_D[BandAden_size][2] = sum_atomMulP[j][2];
		sum_atomMulP_D[BandAden_size][3] = sum_atomMulP[j][3] +sum_atomMulP[j1][3];
		sum_atomMulP_D[BandAden_size][4] = sum_atomMulP[j][4] +sum_atomMulP[j1][4];
		sum_atomMulP_D[BandAden_size][5] = sum_atomMulP[j][5] +sum_atomMulP[j1][5];
		sum_atomMulP_D[BandAden_size][6] = sum_atomMulP[j][6] +sum_atomMulP[j1][6];
		line_k_D[BandAden_size] = line_k[j];
		Data_E_D[BandAden_size] = Data_E[j];

		Re11 = sum_atomMulP_D[BandAden_size][3];       Re22 = sum_atomMulP_D[BandAden_size][4];
		Re12 = sum_atomMulP_D[BandAden_size][5];       Im12 = sum_atomMulP_D[BandAden_size][6];
		EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndw, Ntheta, Nphi );
		fprintf(fp1, "%10.6lf  " ,sum_atomMulP_D[BandAden_size][0]/BohrR);
		fprintf(fp1, "%10.6lf  " ,sum_atomMulP_D[BandAden_size][1]/BohrR);
		fprintf(fp1, "%10.6lf  " ,sum_atomMulP_D[BandAden_size][2]/BohrR);
		fprintf(fp1, "%10.6lf  ", Data_E_D[BandAden_size]);

		fprintf(fp1, "%10.6lf  " ,Nup[0] +Ndw[0]);
		fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]) *MulP_VecScale[0] );
		fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]) *MulP_VecScale[1] );
		fprintf(fp1, "%10.6lf  " ,(Nup[0] -Ndw[0]) *cos(Ntheta[0]) *MulP_VecScale[2] );
		fprintf(fp1, "%10.6lf  \n" ,line_k_D[BandAden_size]); 

		BandAden_size++;
	      }//if(Data_l)
	    }//j1

	  }//if(Data_l)
	}//j

      }//if(BandTotal)
    }//l
    fclose(fp1);
  }//if(sw)
  else {BandAden_size = Data_size;}

  if (Calc_Type == 3){ 

    // ### BAND_OUTPUT ###  
    strcpy(fname_plot,fname_out);
    strcat(fname_plot,".GNUADenB");
    fp1 = fopen(fname_plot,"w");
    fprintf(fp1,"se para\n");
    fprintf(fp1,"se si ra 1\n");
    fprintf(fp1,"se yl 'E-E_F (eV)'\n");
    fprintf(fp1,"se xtics (");
    for (i=1; i<(Band_Nkpath+1); i++){
      fprintf(fp1,"\"%s\" %lf, ", Band_kname[i][1], line_Nk[i-1]);
    } fprintf(fp1,"\"%s\"  %lf)\n", Band_kname[Band_Nkpath][2], line_Nk[Band_Nkpath]);
    fprintf(fp1,"se xra [%10.6lf:%10.6lf]\n",line_Nk[0],line_Nk[Band_Nkpath]);
    fprintf(fp1,"se yra [%10.6lf:%10.6lf]\n",E_Range[0],E_Range[1]);
    fprintf(fp1,"se tra [%10.6lf:%10.6lf]\n",E_Range[0],E_Range[1]);
    for (i=1; i<Band_Nkpath; i++)  fprintf(fp1,"const%d = %lf\n",i,line_Nk[i]);
    for (i=1; i<Band_Nkpath; i++){
      if (i==1){ fprintf(fp1,"pl const%d,t lc -1 \n",i); }
      else{ fprintf(fp1,"repl const%d,t lc -1\n",i); }
    } fprintf(fp1,"PointStyle = 7 \n");

    for(i=0;i<100;i+=ADen_Scale){
      /*
	 iR = 128;
	 iG = (75-2*i/3)*2.55;
	 iB = 255;
       */
      iR = 2.55*(100-(i+ADen_Scale));
      iG = 255;
      iB = 2.55*(100-(i+ADen_Scale));
      ColorCode[0] = D2Hex(iR/16);    ColorCode[1] = D2Hex(iR%16);
      ColorCode[2] = D2Hex(iG/16);    ColorCode[3] = D2Hex(iG%16);
      ColorCode[4] = D2Hex(iB/16);    ColorCode[5] = D2Hex(iB%16);
      ColorCode[6] = '\0';

      strcpy(fname_1,fname_out);
      strcat(fname_1,".ADenBand");
      if (switch_dege==0){
	if (i+ADen_Scale<9) {name_Nband(fname_1,"_00",i+ADen_Scale);      }
	else if(i+ADen_Scale<99) {name_Nband(fname_1,"_0",i+ADen_Scale);  }
	d0 = ((double)i)*0.01;    d1 = ((double)i+ADen_Scale)*0.01;
      }else if(switch_dege==1){ strcat(fname_1,"D");
	if (2*(i+ADen_Scale)<9) {name_Nband(fname_1,"_00",2*(i+ADen_Scale));      }
	else if(2*(i+ADen_Scale)<99) {name_Nband(fname_1,"_0",2*(i+ADen_Scale));  }
	d0 = ((double)i)*0.02;    d1 = ((double)i+ADen_Scale)*0.02;
      }
      fp = fopen(fname_1,"w");
      fprintf(fp,"# k-line     Eig[eV]     Num_Ele     Spin_x      Spin_y      Spin_z  \n");
      k = 0;
      for(j=0;j<BandAden_size;j++){
	if (switch_dege==0){
	  Re11 = sum_atomMulP[j][3];       Re22 = sum_atomMulP[j][4];
	  Re12 = sum_atomMulP[j][5];       Im12 = sum_atomMulP[j][6];
	}else if(switch_dege==1){
	  Re11 = sum_atomMulP_D[j][3];     Re22 = sum_atomMulP_D[j][4];
	  Re12 = sum_atomMulP_D[j][5];     Im12 = sum_atomMulP_D[j][6];
	}
	EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndw, Ntheta, Nphi );

	if ( ((Nup[0] +Ndw[0])>d0) && ((Nup[0] +Ndw[0])<=d1) ){
	  k++;
	  if (switch_dege==0){
	    fprintf(fp, "%10.6lf  " ,line_k[j]);    fprintf(fp, "%10.6lf  " ,Data_E[j]);
	  }else if(switch_dege==1){
	    fprintf(fp, "%10.6lf  " ,line_k_D[j]);  fprintf(fp, "%10.6lf  " ,Data_E_D[j]);
	  }
	  fprintf(fp, "%10.6lf  " ,Nup[0] +Ndw[0]);
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]) *MulP_VecScale[0] );
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]) *MulP_VecScale[1] );
	  fprintf(fp, "%10.6lf  " ,(Nup[0] -Ndw[0]) *cos(Ntheta[0]) *MulP_VecScale[2] );
	  fprintf(fp, "\n");
	}
      }//j
      fclose(fp);
      if (k>0) {
	/*
	   fprintf(fp1,"rep \"%s\" pt PointStyle ps %lf lc rgb \"#%s\"\n"
	   , fname_1, (((double)i+1)/100), ColorCode);
	 */
	fprintf(fp1,"rep \"%s\" w d lc rgb \"#%s\"\n"
	    , fname_1, ColorCode);
      }
    }//i
    fprintf(fp1,"se ter pdf enh col\n");
    fprintf(fp1,"se out \"%s_BandADen.pdf\" \n", fname_out); 
    fprintf(fp1,"unse key\n");
    fprintf(fp1,"rep\n");
    fclose(fp1);
  }//if(CalcType)

  //############ FREE MALLOC ########################
  free(Extract_Atom);

  free(EA);
  free(BandTotal);

  for(j=0;j<=Data_size;j++){
    free(sum_atomMulP[j]);
  } free(sum_atomMulP);
  if (switch_dege==1){
    for(j=0;j<=Data_size;j++){
      free(sum_atomMulP_D[j]);
    } free(sum_atomMulP_D);
    free(line_k_D);
    free(Data_E_D);
  }//sw

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
  // ### (Band Calculation)   ###
  free(line_Nk);
  free(Band_N_perpath);
  if (Band_Nkpath>0) {
    for (i=0; i<(Band_Nkpath+1); i++){
      for (j=0; j<3; j++){
	free(Band_kpath[i][j]);
      }free(Band_kpath[i]);
    }free(Band_kpath);
    for (i=0; i<(Band_Nkpath+1); i++){
      for (j=0; j<3; j++){
	free(Band_kname[i][j]);
      }free(Band_kname[i]);
    }free(Band_kname);
  }//if(Band_Nkpath>0)

  return 0;
}

