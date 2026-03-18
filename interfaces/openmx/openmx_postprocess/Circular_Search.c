#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>

#ifdef nompi
#include "mimic_mpi.h"
#else
#include "mpi.h"
#endif

#include "Tools_BandCalc.h"
#include "Tools_Search.h"
#include "Inputtools.h"
#include "read_scfout.h"
#include "EigenValue_Problem.h"
#include "GetOrbital.h"


double kgrid_dotprod(double x1, double x2, double y1, double y2, double z1, double z2)
{
  double kg1[3],kg2[3];
  kg1[0] = rtv[1][1]*(x1) +rtv[2][1]*(y1) +rtv[3][1]*(z1);
  kg1[1] = rtv[1][2]*(x1) +rtv[2][2]*(y1) +rtv[3][2]*(z1);
  kg1[2] = rtv[1][3]*(x1) +rtv[2][3]*(y1) +rtv[3][3]*(z1);

  kg2[0] = rtv[1][1]*(x2) +rtv[2][1]*(y2) +rtv[3][1]*(z2);
  kg2[1] = rtv[1][2]*(x2) +rtv[2][2]*(y2) +rtv[3][2]*(z2);
  kg2[2] = rtv[1][3]*(x2) +rtv[2][3]*(y2) +rtv[3][3]*(z2);

  return kg1[0]*kg2[0] +kg1[1]*kg2[1] +kg1[2]*kg2[2];
}


int Circular_Search(){
  
  FILE *fp, *fp1, *fp2;
  int i_debug = 0;
  int i,j,k,l,m,n,n2, i1,i2, j1,j2, l1,l2;           // loop variable
  int *S_knum, *E_knum, T_knum, size_Tknum;       // MPI variable (k-point divide)
  int *S2_knum, *E2_knum, T2_knum;                // MPI variable (k-point divide)
  int trial_Newton = 5;
  int namelen, num_procs, myrank;                 // MPI_variable

  char fname_FS[256], fname_MP[256], fname_Spin[256];
  char Pdata_s[256];

  double k1, k2, k3;                              // k-point variable                         
  double d0, d1, d2, d3;

  double TStime, TEtime, Stime, Etime;            // Time variable

  double Time_EIG, *Time_Contour, *Time_MulP;     // Time measurement variable

  int i_vec[20],i_vec2[20];                       // input variable
  char *s_vec[20];                                // input variable
  double r_vec[20];                               // input variable

  // ### k-point Data    ###
  double **k_xyz, *EIGEN_MP;                      // Eigen solve array
  double E1, E2, EF;                              // Eigen value 

  // ### Circular Search ###
  int phiMesh;
  int LoopMax = 100;
  int *Count_Cc, hit_Total;
  double phi, *phi_Cc, Ccheck[2];
  double *phi1_Cc, *phi2_Cc;
  double ***k_xyz_Cc, **EIG_Cc, *EIG_l;
  double StartGrid_1[3],StartGrid_2[3];
  double SearchLength;
  double data1[5], data2[5], data3[5];            // Newton Method
  double SeaLen_Cs;

  // ### Orbital Data    ###
  double **MulP_Cc, ***OrbMulP_Cc, ***Orb2MulP_Cc;
  double Re11, Re22, Re12, Im12;                  // MulP Calc variable-1
  double Nup[2], Ndw[2], Ntheta[2], Nphi[2];      // MulP Calc variable-2


  // ### MPI_Init ############################################
//  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//  MPI_Get_processor_name(processor_name, &namelen);
//  printf("myrank:%d \n",myrank);

//  dtime(&Stime);

  // ### INPUT_FILE ##########################################

  input_open(fname);
  r_vec[0]=-0.5; r_vec[1]=0.1;
  input_doublev("Energy.Range",2,E_Range,r_vec);
  if (E_Range[0]>E_Range[1]){
    d0 = E_Range[0];
    E_Range[0] = E_Range[1];
    E_Range[1] = d0;
  }//if

  input_int("Eigen.Newton",&switch_Eigen_Newton,0);
  if (switch_Eigen_Newton != 0) switch_Eigen_Newton = 1;
  input_int("Calc.Type.3mesh",&plane_3mesh,1);
  
  r_vec[0]=0.3; r_vec[1]=0.3; r_vec[2]=0.0;
  input_doublev("Start.kGrid1",3,StartGrid_1,r_vec);

  r_vec[0]=0.0; r_vec[1]=0.0; r_vec[2]=0.0;
  input_doublev("Start.kGrid2",3,StartGrid_2,r_vec);

  input_double("Search.kLength", &SearchLength, 0.1);
  input_int("Search.phi.Mesh",&phiMesh,7);

  input_int("Trial.Newton",&trial_Newton,5);
  if (trial_Newton<2) trial_Newton = 2;
 
  r_vec[0]=1.0; r_vec[1]=1.0; r_vec[2]=1.0;
  input_doublev("MulP.Vec.Scale",3, MulP_VecScale, r_vec);

  input_close();


  // ### Band Total (n2) ###
  k = 1;
  for (i=1; i<=atomnum; i++){ k+= Total_NumOrbs[i]; }
  n = k - 1;    n2 = 2*k + 2;

  // ### MALLOC ############################################
  S_knum = (int*)malloc(sizeof(int)*num_procs);
  E_knum = (int*)malloc(sizeof(int)*num_procs);
  S2_knum = (int*)malloc(sizeof(int)*num_procs);
  E2_knum = (int*)malloc(sizeof(int)*num_procs);
   
  // ### (EigenValue Problem) ###
  EIGEN = (double*)malloc(sizeof(double)*n2);
  for (i = 0; i < n2; i++) EIGEN[i] = 0.0;

  if (myrank == 0)  printf("########### SEARCH START ##################\n");

  // ### DIVISION CALC BANDS ################################
  if (myrank == 0)  printf("########### Division Calc Bands ###########\n");

  // ### Malloc ###
  size_Tknum = phiMesh+1;
  k_xyz = (double**)malloc(sizeof(double*)*3);
  for(i=0; i<3; i++){
    k_xyz[i] = (double*)malloc(sizeof(double)*(size_Tknum+1));
  }
  EIGEN_MP = (double*)malloc(sizeof(double)*((size_Tknum+1)*n2));
  for (i = 0; i < ((size_Tknum+1)*n2); i++) EIGEN_MP[i] = 0.0;
  EIG_l = (double*)malloc(sizeof(double*)*(size_Tknum+1));
  
  // ### k-GRID & EIGEN(intersection) ###
  for(i=0; i<=phiMesh; i++){
    if(phiMesh<2){ break; }else{
      k_xyz[0][i] = StartGrid_1[0] + (StartGrid_2[0] - StartGrid_1[0])*(2.0*(double)i)/(2.0*(double)phiMesh) +   Shift_K_Point; 
      k_xyz[1][i] = StartGrid_1[1] + (StartGrid_2[1] - StartGrid_1[1])*(2.0*(double)i)/(2.0*(double)phiMesh) -   Shift_K_Point; 
      k_xyz[2][i] = StartGrid_1[2] + (StartGrid_2[2] - StartGrid_1[2])*(2.0*(double)i)/(2.0*(double)phiMesh) + 2*Shift_K_Point; 
    }//if
  }//i
  // ### Division CALC_PART ###
  T_knum = size_Tknum;
  for(i=0; i<num_procs; i++){
    if (T_knum <= i){
      S_knum[i] = -10;   E_knum[i] = -100;
    } else if (T_knum < num_procs) {
     S_knum[i] = i;     E_knum[i] = i;
    } else {
      d0 = (double)T_knum/(double)num_procs;
      S_knum[i] = (int)((double)i*(d0+0.0001));
      E_knum[i] = (int)((double)(i+1)*(d0+0.0001)) - 1;
      if (i==(num_procs-1)) E_knum[i] = T_knum - 1;
      if (E_knum[i]<0)      E_knum[i] = 0;
    }
  }
  // ### EIGEN_VALUE_PROBLEM ###
  for (i = S_knum[myrank]; i <= E_knum[myrank]; i++){
    EigenValue_Problem(k_xyz[0][i], k_xyz[1][i], k_xyz[2][i], 0);
    for(l=1; l<=2*n; l++){ EIGEN_MP[i*n2+l] = EIGEN[l]; }//l
  }//i
  // ### MPI part ###
  for (i=0; i<num_procs; i++){
    k = S_knum[i]*n2;
    l = abs(E_knum[i]-S_knum[i]+1)*n2;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&EIGEN_MP[k], l, MPI_DOUBLE, i, MPI_COMM_WORLD);
  }
  // ### SELECT CALC_BAND ###
  l_min = 0;  l_max = 0;  l_cal = 0; 
  EF = (E_Range[1]+E_Range[0])/2;
  for(l=1; l<=2*n; l++){
    E1 = EIGEN_MP[0+l];  E2 = EIGEN_MP[0+l];
    for (i = 0; i < size_Tknum; i++){
      if (E1 > EIGEN_MP[i*n2+l])  E1 = EIGEN_MP[i*n2+l];  //min
      if (E2 < EIGEN_MP[i*n2+l])  E2 = EIGEN_MP[i*n2+l];  //max
    }//i
    if ((E1-EF)*(E2-EF) <=0){
      if (l_cal == 0){        l_cal = 1;  l_min = l;  l_max = l;     }
      else if(l_cal > 0){    l_max = l;    }
    }//if
  }//l

  if (l_cal > 0){
    l_cal = (l_max-l_min+1);
    if (myrank==0) printf("The number of BANDs %4d (%4d->%4d)\n", l_cal, l_min, l_max); 
  } else{
    if (myrank ==0) printf("No Bands in this Energy level(%lf).\n", EF);
    return -1;
  }
  Count_Cc = (int*)malloc(sizeof(int)*(l_cal+1));
  for(i=0; i<=l_cal; i++) Count_Cc[i] = 0;
  // ### Malloc ###
  k_xyz_Cc = (double***)malloc(sizeof(double**)*(l_cal+1));
  for(i=0; i<=l_cal; i++){
    k_xyz_Cc[i] = (double**)malloc(sizeof(double*)*3);
    for(j=0; j<3; j++){
      k_xyz_Cc[i][j] = (double*)malloc(sizeof(double)*(LoopMax+2));
    }
  }
  phi_Cc = (double*)malloc(sizeof(double)*(LoopMax+2));
  phi1_Cc = (double*)malloc(sizeof(double)*(LoopMax+2));
  phi2_Cc = (double*)malloc(sizeof(double)*(LoopMax+2));

  for(i=0; i<LoopMax+2; i++) phi_Cc[i] = 0.0;
  EIG_Cc = (double**)malloc(sizeof(double*)*(l_cal+1));
  for(i=0; i<=l_cal; i++){
    EIG_Cc[i] = (double*)malloc(sizeof(double)*(LoopMax+2));
  }

  // ### (MulP Calculation)   ###
  Data_MulP = (double****)malloc(sizeof(double***)*4);
  for (i1=0; i1<4; i1++){
    Data_MulP[i1] = (double***)malloc(sizeof(double**)*(l_cal+1));
    for (l=0; l<=l_cal; l++){
      Data_MulP[i1][l] = (double**)malloc(sizeof(double*)*(atomnum+1));
      for (j=0; j<=atomnum; j++){
        Data_MulP[i1][l][j] = (double*)malloc(sizeof(double)*(TNO_MAX+1));
        for (k=0; k<=TNO_MAX; k++) Data_MulP[i1][l][j][k] = 0.0;
      }
    }
  }

  if (myrank == 0){
    strcpy(fname_FS, fname_out);  strcat(fname_FS, ".FS");
    fp2 = fopen(fname_FS, "w");
    if (i_debug ==1){
      strcat(fname_FS, "1");
      fp1 = fopen(fname_FS, "w");
    }//if
  }//if

  for(l=l_min; l<=l_max; l++){

    for(i=0; i<=phiMesh; i++){
      if(phiMesh<2){ break; }else{
        k_xyz[0][i] = StartGrid_1[0] + (StartGrid_2[0] - StartGrid_1[0])*(2.0*(double)i)/(2.0*(double)phiMesh) +   Shift_K_Point;
        k_xyz[1][i] = StartGrid_1[1] + (StartGrid_2[1] - StartGrid_1[1])*(2.0*(double)i)/(2.0*(double)phiMesh) -   Shift_K_Point;
        k_xyz[2][i] = StartGrid_1[2] + (StartGrid_2[2] - StartGrid_1[2])*(2.0*(double)i)/(2.0*(double)phiMesh) + 2*Shift_K_Point;
      }//if
    }//i
    // ### SEARCH FIRST STEP ################################
    j2 = 0;
    for(i=0; i< phiMesh; i++){
      if ((EIGEN_MP[i*n2 + l]-EF)*(EIGEN_MP[(i+1)*n2 + l]-EF) <=0){
        for(k=0; k<3; k++){
          data1[k] = k_xyz[k][i];
          data2[k] = k_xyz[k][i+1];
        }//k
        func_Newton(data1, EIGEN_MP[i*n2+l], data2, EIGEN_MP[(i+1)*n2+l], data3, &E1,  EF, l, trial_Newton);
        for(k=0; k<3; k++){
          k_xyz_Cc[l-l_min][k][0] = data3[k];
        } EIG_Cc[l-l_min][0] = E1; 
      }//if
    }//i
    MPI_Barrier(MPI_COMM_WORLD);
 
    // ### SEARCH SECOND STEP ###############################
    Ccheck[0] = 0.00;    Ccheck[1] = 0.00;
    for(i1=0; i1<100; i1++){
      for(i=0; i<=phiMesh; i++){
        phi = ((double)i/(double)phiMesh-0.5)*PI;
        if (i1>0) phi+= phi_Cc[i1-1]; 
        if (plane_3mesh == 1){     k1= SearchLength*cos(phi);  k2= SearchLength*sin(phi);  k3= 0.0;                   }
        else if(plane_3mesh == 2){ k1= 0.0;                    k2= SearchLength*cos(phi);  k3= SearchLength*sin(phi); }
        else if(plane_3mesh == 3){ k1= SearchLength*sin(phi);  k2= 0.0;                    k3= SearchLength*cos(phi); }
        
        k1+= rtv[1][1]*k_xyz_Cc[l-l_min][0][i1] +rtv[2][1]*k_xyz_Cc[l-l_min][1][i1] +rtv[3][1]*k_xyz_Cc[l-l_min][2][i1];
        k2+= rtv[1][2]*k_xyz_Cc[l-l_min][0][i1] +rtv[2][2]*k_xyz_Cc[l-l_min][1][i1] +rtv[3][2]*k_xyz_Cc[l-l_min][2][i1];
        k3+= rtv[1][3]*k_xyz_Cc[l-l_min][0][i1] +rtv[2][3]*k_xyz_Cc[l-l_min][1][i1] +rtv[3][3]*k_xyz_Cc[l-l_min][2][i1];
        k_xyz[0][i] = (tv[1][1]*k1 +tv[1][2]*k2 +tv[1][3]*k3)/2/PI;
        k_xyz[1][i] = (tv[2][1]*k1 +tv[2][2]*k2 +tv[2][3]*k3)/2/PI;
        k_xyz[2][i] = (tv[3][1]*k1 +tv[3][2]*k2 +tv[3][3]*k3)/2/PI;
      }//i
      // ### EIGEN_VALUE_PROBLEM ###
      for (i = S_knum[myrank]; i <= E_knum[myrank]; i++){
        EigenValue_Problem(k_xyz[0][i], k_xyz[1][i], k_xyz[2][i], 0);
        EIG_l[i] = EIGEN[l];
      }//i
      // ### MPI part ###
      for (i=0; i<num_procs; i++){
        k = S_knum[i];
        l1 = abs(E_knum[i]-S_knum[i]+1);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&EIG_l[k], l1, MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      for(i=0; i< phiMesh; i++){
        if ((EIG_l[i]-EF)*(EIG_l[i+1]-EF) <=0){
          for(k=0; k<3; k++){
            data1[k] = k_xyz[k][i];  data2[k] = k_xyz[k][i+1];
          }//k
          func_Newton(data1, EIG_l[i], data2, EIG_l[i+1], data3, &E1,  EF, l, trial_Newton);
        
          for(k=0; k<3; k++) {            k_xyz_Cc[l-l_min][k][i1+1] = data3[k];          }
          EIG_Cc[l-l_min][i1+1] = E1;   
          phi_Cc[i1] = ((double)i/(double)phiMesh-0.5)*PI;
          
          d0 = kgrid_dotprod( k_xyz_Cc[l-l_min][0][i1+1]-StartGrid_1[0], k_xyz_Cc[l-l_min][0][i1]-StartGrid_1[0],
                              k_xyz_Cc[l-l_min][1][i1+1]-StartGrid_1[1], k_xyz_Cc[l-l_min][1][i1]-StartGrid_1[1],
                              k_xyz_Cc[l-l_min][2][i1+1]-StartGrid_1[2], k_xyz_Cc[l-l_min][2][i1]-StartGrid_1[2] );
          d0/= kgrid_dist(k_xyz_Cc[l-l_min][0][i1+1], StartGrid_1[0],
                          k_xyz_Cc[l-l_min][1][i1+1], StartGrid_1[1],
                          k_xyz_Cc[l-l_min][2][i1+1], StartGrid_1[2] );
          d0/= kgrid_dist(k_xyz_Cc[l-l_min][0][i1],   StartGrid_1[0],
                          k_xyz_Cc[l-l_min][1][i1],   StartGrid_1[1],
                          k_xyz_Cc[l-l_min][2][i1],   StartGrid_1[2] );
          
          d1 = kgrid_dotprod( k_xyz_Cc[l-l_min][0][i1+1]-StartGrid_2[0], k_xyz_Cc[l-l_min][0][i1]-StartGrid_2[0],
                              k_xyz_Cc[l-l_min][1][i1+1]-StartGrid_2[1], k_xyz_Cc[l-l_min][1][i1]-StartGrid_2[1],
                              k_xyz_Cc[l-l_min][2][i1+1]-StartGrid_2[2], k_xyz_Cc[l-l_min][2][i1]-StartGrid_2[2] );
          d1/= kgrid_dist(k_xyz_Cc[l-l_min][0][i1+1], StartGrid_2[0],
                          k_xyz_Cc[l-l_min][1][i1+1], StartGrid_2[1],
                          k_xyz_Cc[l-l_min][2][i1+1], StartGrid_2[2] );
          d1/= kgrid_dist(k_xyz_Cc[l-l_min][0][i1],   StartGrid_2[0],
                          k_xyz_Cc[l-l_min][1][i1],   StartGrid_2[1],
                          k_xyz_Cc[l-l_min][2][i1],   StartGrid_2[2] );
          phi1_Cc[i1] = acos(d0);
          phi2_Cc[i1] = acos(d1);
          break;
        }//if(EIG)
        MPI_Barrier(MPI_COMM_WORLD);
      }//i(for)
      if (i1>0) phi_Cc[i1]+= phi_Cc[i1-1]; 
      Ccheck[0]+= acos(d0);    Ccheck[1]+= acos(d1);
      if(myrank==0) printf("%lf    %lf\n", Ccheck[0], Ccheck[1]);
      // LOOP-FINISH
      if ( Ccheck[0]>= 2*PI  ||  Ccheck[1]>= 2*PI ) {
        for(k=0; k<3; k++) {  k_xyz_Cc[l-l_min][k][i1+1] = k_xyz_Cc[l-l_min][k][0];  }   
        EIG_Cc[l-l_min][i1+1] = EIG_Cc[l-l_min][0];
      }
      if ( Ccheck[0]>= 2*PI  ||  Ccheck[1]>= 2*PI ) {    break;    }
    }//i1(for)
    
    Count_Cc[l-l_min] = i1+1;

    if (i_debug ==1 && myrank == 0){
      for(i1=0; i1<=Count_Cc[l-l_min]; i1++){
        Print_kxyzEig(Pdata_s, k_xyz_Cc[l-l_min][0][i1],
                               k_xyz_Cc[l-l_min][1][i1],
                               k_xyz_Cc[l-l_min][2][i1],
                               l,EIG_Cc[l-l_min][i1]    );
        fprintf(fp1, "%s\n", Pdata_s);        //printf("%s\n", Pdata_s);
      }//i1(for)
    }//if
    if (i_debug ==1 && myrank == 0)  fprintf(fp1, "\n\n");
    
    if (Count_Cc[l-l_min]<= 0) continue;
    d0 = 0.0;
    for(i1=0; i1<Count_Cc[l-l_min]; i1++) {
      d0+= kgrid_dist(k_xyz_Cc[l-l_min][0][i1+1],     k_xyz_Cc[l-l_min][0][i1],
                      k_xyz_Cc[l-l_min][1][i1+1],     k_xyz_Cc[l-l_min][1][i1],
                      k_xyz_Cc[l-l_min][2][i1+1],     k_xyz_Cc[l-l_min][2][i1] );
    }//i1
    if(myrank==0) printf("Next Search Length: %10.6lf (%10.6lf)\n" ,d0/(Count_Cc[l-l_min]) ,SearchLength  );
/*
    d0 = 0.0;
    for(i1=0; i1<Count_Cc[l-l_min]; i1++) {
      d1 = kgrid_dist(k_xyz_Cc[l-l_min][0][i1],     StartGrid_1[0],
                      k_xyz_Cc[l-l_min][1][i1],     StartGrid_1[1],
                      k_xyz_Cc[l-l_min][2][i1],     StartGrid_1[2] );
      d0+= phi1_Cc[i1]*d1;
    }//i1
    if(myrank==0) printf("                  : %10.6lf (%10.6lf)\n" ,d0/(Count_Cc[l-l_min]) ,SearchLength  );
*/
    //kotaka
    SeaLen_Cs = d0/(Count_Cc[l-l_min]);

    // ### SEARCH THIRD STEP  ###############################
    for(i1=0; i1<Count_Cc[l-l_min]-1; i1++) {
    
      for(i=0; i<=phiMesh; i++){
        phi = ((double)i/(double)phiMesh-0.5)*PI;
        if (i1>0) phi+= phi_Cc[i1-1]; 
        if (plane_3mesh == 1){     k1= SeaLen_Cs *cos(phi);  k2= SeaLen_Cs *sin(phi);  k3= 0.0;                 }
        else if(plane_3mesh == 2){ k1= 0.0;                  k2= SeaLen_Cs *cos(phi);  k3= SeaLen_Cs *sin(phi); }
        else if(plane_3mesh == 3){ k1= SeaLen_Cs *sin(phi);  k2= 0.0;                  k3= SeaLen_Cs *cos(phi); }
        
        k1+= rtv[1][1]*k_xyz_Cc[l-l_min][0][i1] +rtv[2][1]*k_xyz_Cc[l-l_min][1][i1] +rtv[3][1]*k_xyz_Cc[l-l_min][2][i1];
        k2+= rtv[1][2]*k_xyz_Cc[l-l_min][0][i1] +rtv[2][2]*k_xyz_Cc[l-l_min][1][i1] +rtv[3][2]*k_xyz_Cc[l-l_min][2][i1];
        k3+= rtv[1][3]*k_xyz_Cc[l-l_min][0][i1] +rtv[2][3]*k_xyz_Cc[l-l_min][1][i1] +rtv[3][3]*k_xyz_Cc[l-l_min][2][i1];
        k_xyz[0][i] = (tv[1][1]*k1 +tv[1][2]*k2 +tv[1][3]*k3)/2/PI;
        k_xyz[1][i] = (tv[2][1]*k1 +tv[2][2]*k2 +tv[2][3]*k3)/2/PI;
        k_xyz[2][i] = (tv[3][1]*k1 +tv[3][2]*k2 +tv[3][3]*k3)/2/PI;
      }//i
      // ### EIGEN_VALUE_PROBLEM ###
      for (i = S_knum[myrank]; i <= E_knum[myrank]; i++){
        EigenValue_Problem(k_xyz[0][i], k_xyz[1][i], k_xyz[2][i], 0);
        EIG_l[i] = EIGEN[l];
      }//i
      // ### MPI part ###
      for (i=0; i<num_procs; i++){
        k = S_knum[i];
        l1 = abs(E_knum[i]-S_knum[i]+1);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&EIG_l[k], l1, MPI_DOUBLE, i, MPI_COMM_WORLD);
      }
      for(i=0; i< phiMesh; i++){
        if ((EIG_l[i]-EF)*(EIG_l[i+1]-EF) <=0){
          
          for(k=0; k<3; k++){
            data1[k] = k_xyz[k][i];          data2[k] = k_xyz[k][i+1];
          }//k
          func_Newton(data1, EIG_l[i], data2, EIG_l[i+1], data3, &E1,  EF, l, trial_Newton);
        
          for(k=0; k<3; k++) {            k_xyz_Cc[l-l_min][k][i1+1] = data3[k];          }
          EIG_Cc[l-l_min][i1+1] = E1;   
          phi_Cc[i1] = ((double)i/(double)phiMesh-0.5)*PI;
          break;
        }//if(EIG)
      }//i(for)
      if (i1>0) phi_Cc[i1]+= phi_Cc[i1-1]; 
    }//i1

    if (myrank == 0) {
      for(i1=0; i1<=Count_Cc[l-l_min]; i1++){
        Print_kxyzEig(Pdata_s, k_xyz_Cc[l-l_min][0][i1], k_xyz_Cc[l-l_min][1][i1]
                             , k_xyz_Cc[l-l_min][2][i1], l, EIG_Cc[l-l_min][i1]    );
        fprintf(fp2, "%s\n", Pdata_s);
      }//i1(for)
      Print_kxyzEig(Pdata_s, k_xyz_Cc[l-l_min][0][0], k_xyz_Cc[l-l_min][1][0]
                           , k_xyz_Cc[l-l_min][2][0], l, EIG_Cc[l-l_min][0]    );
      fprintf(fp2, "%s\n\n\n", Pdata_s);
    }//if(myrank)

  }//l
  hit_Total = 0.0;
  for(l=l_min; l<=l_max; l++)  hit_Total+= Count_Cc[l-l_min];
  
  if (i_debug ==1 &&  myrank == 0){    fclose(fp1);    }
  if (myrank == 0){    fclose(fp2);    }

  //MULP CALC
  if (myrank == 0){
    strcpy(fname_MP,fname_out);    strcat(fname_MP,".AtomMulP");
    fp1= fopen(fname_MP,"w");
    fprintf(fp1,"                                      \n");
    fclose(fp1);
    for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
      strcpy(fname_MP,fname_out);     strcat(fname_MP,".MulP_");      strcat(fname_MP,OrbSym[i1]);
      fp1= fopen(fname_MP,"w");
      fprintf(fp1,"                                    \n");
      fclose(fp1);
    }//i1
    for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
      strcpy(fname_MP,fname_out);     strcat(fname_MP,".MulP_");     strcat(fname_MP,OrbName[i1]);
      fp1= fopen(fname_MP,"w");
      fprintf(fp1,"                                    \n");
      fclose(fp1);
    }//i1
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  MulP_Cc = (double**)malloc(sizeof(double*)*4);
  for (i=0; i<4; i++){
    MulP_Cc[i] = (double*)malloc(sizeof(double)*((atomnum+1)*(LoopMax+2)));
    for (k=0; k<((atomnum+1)*(LoopMax+2)); k++) MulP_Cc[i][k] = 0.0;
  }
  OrbMulP_Cc = (double***)malloc(sizeof(double**)*4);
  for (i=0; i<4; i++){
    OrbMulP_Cc[i] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[0]+1));
    for (j=0; j<=ClaOrb_MAX[0]; j++){
      OrbMulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(LoopMax+2)));
      for (k=0; k<((atomnum+1)*(LoopMax+2)); k++) OrbMulP_Cc[i][j][k] = 0.0;
    }//j
  }//i
  Orb2MulP_Cc = (double***)malloc(sizeof(double**)*4);
  for (i=0; i<4; i++){
    Orb2MulP_Cc[i] = (double**)malloc(sizeof(double*)*(ClaOrb_MAX[1]+1));
    for (j=0; j<=ClaOrb_MAX[1]; j++){
      Orb2MulP_Cc[i][j] = (double*)malloc(sizeof(double)*((atomnum+1)*(LoopMax+2)));
      for (k=0; k<((atomnum+1)*(LoopMax+2)); k++) Orb2MulP_Cc[i][j][k] = 0.0;
    }//j
  }//i



  for(l=l_min; l<=l_max; l++){
    if (Count_Cc[l-l_min] <1) continue;

    // ### MulP_Calculation ################################
    T2_knum = Count_Cc[l-l_min];
    // Division CALC_PART 
    for(i=0; i<num_procs; i++){
      if (T2_knum <= i){
        S2_knum[i] = -10;   E2_knum[i] = -100; 
      } else if (T2_knum < num_procs) {
        S2_knum[i] = i;     E2_knum[i] = i;   
      } else {
        d0 = (double)T2_knum/(double)num_procs;
        S2_knum[i] = (int)((double)i*(d0+0.0001));
        E2_knum[i] = (int)((double)(i+1)*(d0+0.0001)) - 1;
        if (i==(num_procs-1)) E2_knum[i] = T2_knum - 1;
            if (E2_knum[i]<0)      E2_knum[i] = 0;
      }
    }
    // #######################################################
    for (j = S2_knum[myrank]; j <= E2_knum[myrank]; j++){
      for (i=0; i < atomnum; i++){
        for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
          for (j1=0; j1<4; j1++){ OrbMulP_Cc[j1][i1][j*(atomnum+1)+i] = 0;  }//j1
        }//for(i1)
        for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
          for (j1=0; j1<4; j1++){ Orb2MulP_Cc[j1][i1][j*(atomnum+1)+i] = 0; }//j1
        }//for(i1)
      }//for(i)
    }//for(j)

 //   printf("kotaka\n");

    MPI_Barrier(MPI_COMM_WORLD);
    for (j = S2_knum[myrank]; j <= E2_knum[myrank]; j++){
      EigenValue_Problem( k_xyz_Cc[l-l_min][0][j], k_xyz_Cc[l-l_min][1][j], k_xyz_Cc[l-l_min][2][j], 1);
      for (i=0; i < atomnum; i++){
        for (j1=0; j1<4; j1++){
          MulP_Cc[j1][j*(atomnum+1)+i] = 0.0;

          for (i1=0; i1 < Total_NumOrbs[i+1]; i1++){
            Orb2MulP_Cc[j1][ClaOrb[i+1][i1]][j*(atomnum+1)+i]+= Data_MulP[j1][l-l_min][i+1][i1];
            MulP_Cc[j1][j*(atomnum+1)+i]+= Data_MulP[j1][l-l_min][i+1][i1];
          }//for(i1)

        }//for(j1)
      }//for(i)
    }//for(j)

    MPI_Barrier(MPI_COMM_WORLD);
    for (i=0; i<num_procs; i++){
      k = S2_knum[i]*(atomnum+1);
      i2 = abs(E2_knum[i]-S2_knum[i]+1)*(atomnum+1);
      MPI_Barrier(MPI_COMM_WORLD);
      for (j1=0; j1<4; j1++){
        MPI_Bcast(&MulP_Cc[j1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
        for (i1=0; i1 <=ClaOrb_MAX[1]; i1++){
          MPI_Bcast(&Orb2MulP_Cc[j1][i1][k], i2, MPI_DOUBLE, i, MPI_COMM_WORLD);
        }//for(i1)
      }//for(j1)
    }//i

    for(j=0; j<Count_Cc[l-l_min]; j++){
      for (i=0; i<atomnum; i++){
        for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
          for (i2=0; i2 <=ClaOrb_MAX[1]; i2++){
            if (OrbSym[i1][0] == OrbName[i2][0]){
              for (j1=0; j1<4; j1++){
                OrbMulP_Cc[j1][i1][j*(atomnum+1)+i]+= Orb2MulP_Cc[j1][i2][j*(atomnum+1)+i];
              }//j1
            }//if
          }//i2
        }//i1
      }//j
    }//i

    if (myrank == 0){
      // ###################################################
      strcpy(fname_MP,fname_out);  strcat(fname_MP,".AtomMulP");
      fp1= fopen(fname_MP,"a");
      for(j=0; j<Count_Cc[l-l_min]; j++){
        Print_kxyzEig(Pdata_s, k_xyz_Cc[l-l_min][0][j], k_xyz_Cc[l-l_min][1][j], k_xyz_Cc[l-l_min][2][j], l, EIG_Cc[l-l_min][j]);
        fprintf(fp1, "%s", Pdata_s);
        for (i=0; i<atomnum; i++){
          fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[0][j*(atomnum+1)+i], MulP_Cc[1][j*(atomnum+1)+i]);
          fprintf(fp1,"%10.6lf %10.6lf ", MulP_Cc[2][j*(atomnum+1)+i], MulP_Cc[3][j*(atomnum+1)+i]);
        } fprintf(fp1,"\n");
      }//j
      fclose(fp1);
      // ###################################################
      for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
        strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbSym[i1]);
        fp1= fopen(fname_MP,"a");
        for(j=0; j<Count_Cc[l-l_min]; j++){
          Print_kxyzEig(Pdata_s, k_xyz_Cc[l-l_min][0][j], k_xyz_Cc[l-l_min][1][j], k_xyz_Cc[l-l_min][2][j], l, EIG_Cc[l-l_min][j]);
          fprintf(fp1, "%s", Pdata_s);
          for (i=0; i<atomnum; i++){
            fprintf(fp1,"%10.6lf %10.6lf ", OrbMulP_Cc[0][i1][j*(atomnum+1)+i], OrbMulP_Cc[1][i1][j*(atomnum+1)+i]);
            fprintf(fp1,"%10.6lf %10.6lf ", OrbMulP_Cc[2][i1][j*(atomnum+1)+i], OrbMulP_Cc[3][i1][j*(atomnum+1)+i]);
          } fprintf(fp1,"\n");
        }//j
        fclose(fp1);
      }//i1
      // ###################################################
      for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
        strcpy(fname_MP,fname_out);  strcat(fname_MP,".MulP_");  strcat(fname_MP,OrbName[i1]);
        fp1= fopen(fname_MP,"a");
        for(j=0; j<Count_Cc[l-l_min]; j++){
          Print_kxyzEig(Pdata_s, k_xyz_Cc[l-l_min][0][j], k_xyz_Cc[l-l_min][1][j], k_xyz_Cc[l-l_min][2][j], l, EIG_Cc[l-l_min][j]);
          fprintf(fp1, "%s", Pdata_s);
          for (i=0; i<atomnum; i++){
            fprintf(fp1,"%10.6lf %10.6lf ", Orb2MulP_Cc[0][i1][j*(atomnum+1)+i], Orb2MulP_Cc[1][i1][j*(atomnum+1)+i]);
            fprintf(fp1,"%10.6lf %10.6lf ", Orb2MulP_Cc[2][i1][j*(atomnum+1)+i], Orb2MulP_Cc[3][i1][j*(atomnum+1)+i]);
          } fprintf(fp1,"\n");
        }//j
        fclose(fp1);
      }//i1
      // ###################################################
    }//if(myrank)

    if (myrank == 0){
      // ###################################################
      strcpy(fname_Spin,fname_out);
      name_Nband(fname_Spin,".Pxyz_",l);
      fp1 = fopen(fname_Spin,"a");
      for(j=0; j<Count_Cc[l-l_min]; j++){
        for(i=0; i<3; i++){
          fprintf(fp1,"%10.6lf ",rtv[1][i+1]*k_xyz_Cc[l-l_min][0][j]+ rtv[2][i+1]*k_xyz_Cc[l-l_min][1][j]+ rtv[3][i+1]*k_xyz_Cc[l-l_min][2][j]);
        }
        Re11 = 0;      Re22 = 0;      Re12 = 0;      Im12 = 0;
        for (i=0; i<atomnum; i++){
          Re11+= MulP_Cc[0][j*(atomnum+1)+i];        Re22+= MulP_Cc[1][j*(atomnum+1)+i];
          Re12+= MulP_Cc[2][j*(atomnum+1)+i];        Im12+= MulP_Cc[3][j*(atomnum+1)+i];
        } EulerAngle_Spin( 1, Re11, Re22, Re12, Im12, Re12, -Im12, Nup, Ndw, Ntheta, Nphi );
        fprintf(fp1,"%10.6lf  ",MulP_VecScale[0]* (Nup[0] -Ndw[0]) *sin(Ntheta[0]) *cos(Nphi[0]));
        fprintf(fp1,"%10.6lf  ",MulP_VecScale[1]* (Nup[0] -Ndw[0]) *sin(Ntheta[0]) *sin(Nphi[0]));
        fprintf(fp1,"%10.6lf\n",MulP_VecScale[2]* (Nup[0] -Ndw[0]) *cos(Ntheta[0]));
      }//j      
      fclose(fp1);
    }//if(myrank)

  }//l

  if (myrank == 0){
    //### atomnum & data_size ###
    strcpy(fname_MP,fname_out);     strcat(fname_MP,".AtomMulP");
    fp1= fopen(fname_MP,"r+");
    fseek(fp1, 0L, SEEK_SET);
    fprintf(fp1,"%6d %4d", hit_Total, atomnum);
    fclose(fp1);
    for (i1=0; i1 <=ClaOrb_MAX[0]; i1++){
      strcpy(fname_MP,fname_out);    strcat(fname_MP,".MulP_");    strcat(fname_MP,OrbSym[i1]);
      fp1= fopen(fname_MP,"r+");
      fseek(fp1, 0L, SEEK_SET);
      fprintf(fp1,"%6d %4d", hit_Total, atomnum);
      fclose(fp1);
    }//i1
    for (i1=1; i1 <=ClaOrb_MAX[1]; i1++){
      strcpy(fname_MP,fname_out);    strcat(fname_MP,".MulP_");    strcat(fname_MP,OrbName[i1]);
      fp1= fopen(fname_MP,"r+");
      fseek(fp1, 0L, SEEK_SET);
      fprintf(fp1,"%6d %4d", hit_Total, atomnum);
      fclose(fp1);
    }//i1
  }//if(myrank)


  // ### Malloc Free ###
  for (i=0; i<4; i++){
    free(MulP_Cc[i]);
  } free(MulP_Cc);
  for (i=0; i<4; i++){
    for (j=0; j<=ClaOrb_MAX[0]; j++){
      free(OrbMulP_Cc[i][j]);
    } free(OrbMulP_Cc[i]);
  } free(OrbMulP_Cc);
  for (i=0; i<4; i++){
    for (j=0; j<=ClaOrb_MAX[1]; j++){
      free(Orb2MulP_Cc[i][j]);
    } free(Orb2MulP_Cc[i]);
  } free(Orb2MulP_Cc);

  // ### (Closed circle) ###
  free(Count_Cc);
  for(i=0; i<=l_cal; i++){
    for(j=0; j<3; j++){
      free(k_xyz_Cc[i][j]);
    } free(k_xyz_Cc[i]);
  } free(k_xyz_Cc);
  free(phi_Cc);
  free(phi1_Cc);
  free(phi2_Cc);
  for(i=0; i<=l_cal; i++){
    free(EIG_Cc[i]);
  } free(EIG_Cc);

  // ### (MPI Calculation)    ###
  free(S_knum);
  free(E_knum);

  // ### (EigenValue Problem) ###
  for(i=0; i<3; i++){
    free(k_xyz[i]);
  } free(k_xyz);
  free(EIGEN_MP);
  free(EIGEN); 
  free(EIG_l); 
  
  // ### (MulP Calculation)   ###
  for (i1=0; i1<4; i1++){
    for (l=0; l<=l_cal; l++){
      for (j=0; j<=atomnum; j++){
        free(Data_MulP[i1][l][j]);
      } free(Data_MulP[i1][l]);
    } free(Data_MulP[i1]);
  } free(Data_MulP);

  return 0;
}


