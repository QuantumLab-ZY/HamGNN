/**********************************************************************
  Set_Orbitals_Grid.c:

   Set_Orbitals_Grid.c is a subroutine to calculate the value of basis
   functions on each grid point.

  Log of Set_Orbitals_Grid.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include <omp.h>




double Set_Orbitals_Grid(int Cnt_kind)
{
  int i,j,n,Mc_AN,Gc_AN,Cwan,NO0,GNc,GRc;
  int Gh_AN,Mh_AN,Rnh,Hwan,NO1,Nog,h_AN;
  long int k,Nc;
  double time0;
  double x,y,z,dx,dy,dz;
  double TStime,TEtime;
  double Cxyz[4];
  int numprocs,myid,tag=999,ID,IDS,IDR;
  double Stime_atom,Etime_atom;

  MPI_Status stat;
  MPI_Request request;

  /* for OpenMP */
  int OMPID,Nthrds,Nprocs;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  
  dtime(&TStime);

  /*****************************************************
                Calculate orbitals on grids
  *****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    dtime(&Stime_atom);

    Gc_AN = M2G[Mc_AN];    
    Cwan = WhatSpecies[Gc_AN];

    if (Cnt_kind==0)  NO0 = Spe_Total_NO[Cwan];
    else              NO0 = Spe_Total_CNO[Cwan]; 

#pragma omp parallel shared(Comp2Real,Spe_PAO_RWF,Spe_Num_Basis,Spe_MaxL_Basis,Spe_PAO_RV,Spe_Num_Mesh_PAO,List_YOUSO,Orbs_Grid,Cnt_kind,Gxyz,atv,CellListAtom,GridListAtom,GridN_Atom,Gc_AN,Cwan,Mc_AN,NO0) private(OMPID,Nthrds,Nprocs,Nc,GNc,GRc,Cxyz,x,y,z,dx,dy,dz,i,j)
    {
      double *Chi0;
      double Cxyz0[4]; 
      double **RF;
      double **AF;
      int i,L0,Mul0,M0,i1;
      double S_coordinate[3];
      double dum,dum1,dum2,dum3,dum4,a,b,c,d;
      double siQ,coQ,siP,coP,Q,P,R;
      double rm,df,sum0,sum1;
      double SH[Supported_MaxL*2+1][2];
      double dSHt[Supported_MaxL*2+1][2];
      double dSHp[Supported_MaxL*2+1][2];

      /* Radial */
      int mp_min,mp_max,m,po,wan;
      double h1,h2,h3,f1,f2,f3,f4;
      double g1,g2,x1,x2,y1,y2,y12,y22,f;
      double r,r1,theta,phi,Min_r;

      /* allocation of array */

      Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

      RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
      }

      AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
      for (i=0; i<(List_YOUSO[25]+1); i++){
	AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
      }

      /* get info. on OpenMP */ 

      OMPID = omp_get_thread_num();
      Nthrds = omp_get_num_threads();
      Nprocs = omp_get_num_procs();

      for (Nc=OMPID*GridN_Atom[Gc_AN]/Nthrds; Nc<(OMPID+1)*GridN_Atom[Gc_AN]/Nthrds; Nc++){

	GNc = GridListAtom[Mc_AN][Nc]; 
	GRc = CellListAtom[Mc_AN][Nc];

	Get_Grid_XYZ(GNc,Cxyz);
	x = Cxyz[1] + atv[GRc][1] - Gxyz[Gc_AN][1]; 
	y = Cxyz[2] + atv[GRc][2] - Gxyz[Gc_AN][2]; 
	z = Cxyz[3] + atv[GRc][3] - Gxyz[Gc_AN][3];

	if (Cnt_kind==0){

          /* Get_Orbitals(Cwan,x,y,z,Chi0); */
          /* start of inlining of Get_Orbitals */

          wan = Cwan;

          /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
          /* start of inlining of xyz2spherical */

	  Min_r = 10e-15;
	  dum = x*x + y*y; 
	  r = sqrt(dum + z*z);
	  r1 = sqrt(dum);

	  if (Min_r<=r){

	    if (r<fabs(z))
	      dum1 = sgn(z)*1.0;
	    else
	      dum1 = z/r;

	    theta = acos(dum1);

	    if (Min_r<=r1){
	      if (0.0<=x){

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;        
  
		phi = asin(dum1);
	      }
	      else{

		if (r1<fabs(y))
		  dum1 = sgn(y)*1.0;
		else
		  dum1 = y/r1;        

		phi = PI - asin(dum1);
	      }
	    }
	    else{
	      phi = 0.0;
	    }
	  }
	  else{
	    theta = 0.5*PI;
	    phi = 0.0;
	  }

	  R = r;
	  Q = theta;
	  P = phi;

	  /* end of inlining of xyz2spherical */

	  po = 0;
	  mp_min = 0;
	  mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	  if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		RF[L0][Mul0] = 0.0;
	      }
	    }

	    po = 1;
	  }

	  else if (R<Spe_PAO_RV[wan][0]){

	    m = 4;
	    rm = Spe_PAO_RV[wan][m];

	    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = rm - Spe_PAO_RV[wan][m-1];
	    x2 = rm - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		if (m==1){
		  h1 = -(h2+h3);
		  f1 = f4;
		}
		else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		  h3 = -(h1+h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y22*(2.0*f2 + h2*g1)/h2
		  + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		  - y12*(2.0*f3 - h2*g2)/h2;

		if (L0==0){
		  a = 0.0;
		  b = 0.5*df/rm;
		  c = 0.0;
		  d = f - b*rm*rm;
		}

		else if (L0==1){
		  a = (rm*df - f)/(2.0*rm*rm*rm);
		  b = 0.0;
		  c = df - 3.0*a*rm*rm;
		  d = 0.0;
		}

		else{
		  b = (3.0*f - rm*df)/(rm*rm);
		  a = (f - b*rm*rm)/(rm*rm*rm);
		  c = 0.0;
		  d = 0.0;
		}

		RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

	      }
	    }

	  }

	  else{

	    do{
	      m = (mp_min + mp_max)/2;
	      if (Spe_PAO_RV[wan][m]<R)
		mp_min = m;
	      else 
		mp_max = m;
	    }
	    while((mp_max-mp_min)!=1);
	    m = mp_max;

	    h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
	    h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
	    h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

	    x1 = R - Spe_PAO_RV[wan][m-1];
	    x2 = R - Spe_PAO_RV[wan][m];
	    y1 = x1/h2;
	    y2 = x2/h2;
	    y12 = y1*y1;
	    y22 = y2*y2;

	    dum = h1 + h2;
	    dum1 = h1/h2/dum;
	    dum2 = h2/h1/dum;
	    dum = h2 + h3;
	    dum3 = h2/h3/dum;
	    dum4 = h3/h2/dum;

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	      for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		if (m==1){
		  h1 = -(h2+h3);
		  f1 = f4;
		}
		else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		  h3 = -(h1+h2);
		  f4 = f1;
		}

		dum = f3 - f2;
		g1 = dum*dum1 + (f2-f1)*dum2;
		g2 = (f4-f3)*dum3 + dum*dum4;

		f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		  + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		RF[L0][Mul0] = f;

	      }
	    } 

	  }

	  if (po==0){

	    /* Angular */
	    siQ = sin(Q);
	    coQ = cos(Q);
	    siP = sin(P);
	    coP = cos(P);

	    for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

	      if (L0==0){
		AF[0][0] = 0.282094791773878;
	      }
	      else if (L0==1){
		dum = 0.48860251190292*siQ;
		AF[1][0] = dum*coP;
		AF[1][1] = dum*siP;
		AF[1][2] = 0.48860251190292*coQ;
	      }
	      else if (L0==2){
		dum1 = siQ*siQ;
		dum2 = 1.09254843059208*siQ*coQ;
		AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		AF[2][2] = 1.09254843059208*dum1*siP*coP;
		AF[2][3] = dum2*coP;
		AF[2][4] = dum2*siP;
	      }

	      else if (L0==3){
		AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
	      }

	      else if (4<=L0){

		/* calculation of complex spherical harmonics functions */
		for(m=-L0; m<=L0; m++){ 
		  ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		}

		/* transformation of complex to real */
		for (i=0; i<(L0*2+1); i++){

		  sum0 = 0.0;
		  sum1 = 0.0; 
		  for (j=0; j<(L0*2+1); j++){
		    sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
		    sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		  }
		  AF[L0][i] = sum0 + sum1; 
		}              

	      }
	    }
	  }

	  /* Chi0 */  
	  i1 = -1;
	  for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	    for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
	      for (M0=0; M0<=2*L0; M0++){
		i1++;
		Chi0[i1] = RF[L0][Mul0]*AF[L0][M0];
	      }
	    }
	  }
	  /* end of inlining of Get_Orbitals */

	}
	else{
          Get_Cnt_Orbitals(Mc_AN,x,y,z,Chi0);
	}

	for (i=0; i<NO0; i++){
	  Orbs_Grid[Mc_AN][Nc][i] = (Type_Orbs_Grid)Chi0[i];/* AITUNE */
	}

      } /* Nc */

      /* freeing of array */

      free(Chi0);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(RF[i]);
      }
      free(RF);

      for (i=0; i<(List_YOUSO[25]+1); i++){
	free(AF[i]);
      }
      free(AF);

    } /* #pragma omp parallel */

    dtime(&Etime_atom);
    time_per_atom[Gc_AN] += Etime_atom - Stime_atom;
  }

  /****************************************************
     Calculate Orbs_Grid_FNAN
  ****************************************************/

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];    

    for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){

      Gh_AN = natn[Gc_AN][h_AN];

      if (G2ID[Gh_AN]!=myid){

        Mh_AN = F_G2M[Gh_AN];
        Rnh = ncn[Gc_AN][h_AN];
        Hwan = WhatSpecies[Gh_AN];

        if (Cnt_kind==0)  NO1 = Spe_Total_NO[Hwan];
        else              NO1 = Spe_Total_CNO[Hwan];

#pragma omp parallel shared(List_YOUSO,Orbs_Grid_FNAN,NO1,Mh_AN,Hwan,Cnt_kind,Rnh,Gh_AN,Gxyz,atv,NumOLG,Mc_AN,h_AN,GListTAtoms1,GridListAtom,CellListAtom) private(OMPID,Nthrds,Nprocs,Nog,Nc,GNc,GRc,x,y,z,j)
        {

          double *Chi0;
	  double Cxyz0[4]; 
          double **RF;
          double **AF;
	  int i,L0,Mul0,M0,i1;
	  double S_coordinate[3];
	  double dum,dum1,dum2,dum3,dum4,a,b,c,d;
	  double siQ,coQ,siP,coP,Q,P,R;
	  double rm,df,sum0,sum1;
	  double SH[Supported_MaxL*2+1][2];
	  double dSHt[Supported_MaxL*2+1][2];
	  double dSHp[Supported_MaxL*2+1][2];

	  /* Radial */
	  int mp_min,mp_max,m,po,wan;
	  double h1,h2,h3,f1,f2,f3,f4;
	  double g1,g2,x1,x2,y1,y2,y12,y22,f;
          double r,r1,theta,phi,Min_r;

          /* allocation of arrays */

	  Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

	  RF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    RF[i] = (double*)malloc(sizeof(double)*List_YOUSO[24]);
	  }

	  AF = (double**)malloc(sizeof(double*)*(List_YOUSO[25]+1));
	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    AF[i] = (double*)malloc(sizeof(double)*(2*(List_YOUSO[25]+1)+1));
	  }

	  /* get info. on OpenMP */ 

	  OMPID = omp_get_thread_num();
	  Nthrds = omp_get_num_threads();
	  Nprocs = omp_get_num_procs();

	  for (Nog=OMPID*NumOLG[Mc_AN][h_AN]/Nthrds; Nog<(OMPID+1)*NumOLG[Mc_AN][h_AN]/Nthrds; Nog++){

	    Nc = GListTAtoms1[Mc_AN][h_AN][Nog];
	    GNc = GridListAtom[Mc_AN][Nc];
	    GRc = CellListAtom[Mc_AN][Nc]; 

	    Get_Grid_XYZ(GNc,Cxyz0);

	    x = Cxyz0[1] + atv[GRc][1] - Gxyz[Gh_AN][1] - atv[Rnh][1];
	    y = Cxyz0[2] + atv[GRc][2] - Gxyz[Gh_AN][2] - atv[Rnh][2];
	    z = Cxyz0[3] + atv[GRc][3] - Gxyz[Gh_AN][3] - atv[Rnh][3];

	    if (Cnt_kind==0){

              /* Get_Orbitals(Hwan,x,y,z,Chi0); */
              /* start of inlining of Get_Orbitals */

              wan = Hwan; 

	      /* xyz2spherical(x,y,z,0.0,0.0,0.0,S_coordinate); */
              /* start of inlining of xyz2spherical */

	      Min_r = 10e-15;
	      dum = x*x + y*y; 
	      r = sqrt(dum + z*z);
	      r1 = sqrt(dum);

	      if (Min_r<=r){

		if (r<fabs(z))
		  dum1 = sgn(z)*1.0;
		else
		  dum1 = z/r;

		theta = acos(dum1);

		if (Min_r<=r1){
		  if (0.0<=x){

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;        
  
		    phi = asin(dum1);
		  }
		  else{

		    if (r1<fabs(y))
		      dum1 = sgn(y)*1.0;
		    else
		      dum1 = y/r1;        

		    phi = PI - asin(dum1);
		  }
		}
		else{
		  phi = 0.0;
		}
	      }
	      else{
		theta = 0.5*PI;
		phi = 0.0;
	      }

	      R = r;
	      Q = theta;
	      P = phi;

              /* end of inlining of xyz2spherical */

	      po = 0;
	      mp_min = 0;
	      mp_max = Spe_Num_Mesh_PAO[wan] - 1;

	      if (Spe_PAO_RV[wan][Spe_Num_Mesh_PAO[wan]-1]<R){

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		    RF[L0][Mul0] = 0.0;
		  }
		}

		po = 1;
	      }

	      else if (R<Spe_PAO_RV[wan][0]){

		m = 4;
		rm = Spe_PAO_RV[wan][m];

		h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = rm - Spe_PAO_RV[wan][m-1];
		x2 = rm - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		    if (m==1){
		      h1 = -(h2+h3);
		      f1 = f4;
		    }
		    else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		      h3 = -(h1+h2);
		      f4 = f1;
		    }

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y22*(2.0*f2 + h2*g1)/h2
		      + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
		      - y12*(2.0*f3 - h2*g2)/h2;

		    if (L0==0){
		      a = 0.0;
		      b = 0.5*df/rm;
		      c = 0.0;
		      d = f - b*rm*rm;
		    }

		    else if (L0==1){
		      a = (rm*df - f)/(2.0*rm*rm*rm);
		      b = 0.0;
		      c = df - 3.0*a*rm*rm;
		      d = 0.0;
		    }

		    else{
		      b = (3.0*f - rm*df)/(rm*rm);
		      a = (f - b*rm*rm)/(rm*rm*rm);
		      c = 0.0;
		      d = 0.0;
		    }

		    RF[L0][Mul0] = a*R*R*R + b*R*R + c*R + d;

		  }
		}

	      }

	      else{

		do{
		  m = (mp_min + mp_max)/2;
		  if (Spe_PAO_RV[wan][m]<R)
		    mp_min = m;
		  else 
		    mp_max = m;
		}
		while((mp_max-mp_min)!=1);
		m = mp_max;

		h1 = Spe_PAO_RV[wan][m-1] - Spe_PAO_RV[wan][m-2];
		h2 = Spe_PAO_RV[wan][m]   - Spe_PAO_RV[wan][m-1];
		h3 = Spe_PAO_RV[wan][m+1] - Spe_PAO_RV[wan][m];

		x1 = R - Spe_PAO_RV[wan][m-1];
		x2 = R - Spe_PAO_RV[wan][m];
		y1 = x1/h2;
		y2 = x2/h2;
		y12 = y1*y1;
		y22 = y2*y2;

		dum = h1 + h2;
		dum1 = h1/h2/dum;
		dum2 = h2/h1/dum;
		dum = h2 + h3;
		dum3 = h2/h3/dum;
		dum4 = h3/h2/dum;

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		  for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){

		    f1 = Spe_PAO_RWF[wan][L0][Mul0][m-2];
		    f2 = Spe_PAO_RWF[wan][L0][Mul0][m-1];
		    f3 = Spe_PAO_RWF[wan][L0][Mul0][m];
		    f4 = Spe_PAO_RWF[wan][L0][Mul0][m+1];

		    if (m==1){
		      h1 = -(h2+h3);
		      f1 = f4;
		    }
		    else if (m==(Spe_Num_Mesh_PAO[wan]-1)){
		      h3 = -(h1+h2);
		      f4 = f1;
		    }

		    dum = f3 - f2;
		    g1 = dum*dum1 + (f2-f1)*dum2;
		    g2 = (f4-f3)*dum3 + dum*dum4;

		    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
		      + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);

		    RF[L0][Mul0] = f;

		  }
		} 

	      }

	      if (po==0){

		/* Angular */
		siQ = sin(Q);
		coQ = cos(Q);
		siP = sin(P);
		coP = cos(P);

		for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){

		  if (L0==0){
		    AF[0][0] = 0.282094791773878;
		  }
		  else if (L0==1){
		    dum = 0.48860251190292*siQ;
		    AF[1][0] = dum*coP;
		    AF[1][1] = dum*siP;
		    AF[1][2] = 0.48860251190292*coQ;
		  }
		  else if (L0==2){
		    dum1 = siQ*siQ;
		    dum2 = 1.09254843059208*siQ*coQ;
		    AF[2][0] = 0.94617469575756*coQ*coQ - 0.31539156525252;
		    AF[2][1] = 0.54627421529604*dum1*(1.0 - 2.0*siP*siP);
		    AF[2][2] = 1.09254843059208*dum1*siP*coP;
		    AF[2][3] = dum2*coP;
		    AF[2][4] = dum2*siP;
		  }

		  else if (L0==3){
		    AF[3][0] = 0.373176332590116*(5.0*coQ*coQ*coQ - 3.0*coQ);
		    AF[3][1] = 0.457045799464466*coP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][2] = 0.457045799464466*siP*siQ*(5.0*coQ*coQ - 1.0);
		    AF[3][3] = 1.44530572132028*siQ*siQ*coQ*(coP*coP-siP*siP);
		    AF[3][4] = 2.89061144264055*siQ*siQ*coQ*siP*coP;
		    AF[3][5] = 0.590043589926644*siQ*siQ*siQ*(4.0*coP*coP*coP - 3.0*coP);
		    AF[3][6] = 0.590043589926644*siQ*siQ*siQ*(3.0*siP - 4.0*siP*siP*siP);
		  }

		  else if (4<=L0){

		    /* calculation of complex spherical harmonics functions */
		    for(m=-L0; m<=L0; m++){ 
		      ComplexSH(L0,m,Q,P,SH[L0+m],dSHt[L0+m],dSHp[L0+m]);
		    }

		    /* transformation of complex to real */
		    for (i=0; i<(L0*2+1); i++){

		      sum0 = 0.0;
		      sum1 = 0.0; 
		      for (j=0; j<(L0*2+1); j++){
			sum0 += Comp2Real[L0][i][j].r*SH[j][0] - Comp2Real[L0][i][j].i*SH[j][1]; 
			sum1 += Comp2Real[L0][i][j].r*SH[j][1] + Comp2Real[L0][i][j].i*SH[j][0]; 
		      }
		      AF[L0][i] = sum0 + sum1; 
		    }              

		  }
		}
	      }

	      /* Chi0 */  
	      i1 = -1;
	      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
		for (Mul0=0; Mul0<Spe_Num_Basis[wan][L0]; Mul0++){
		  for (M0=0; M0<=2*L0; M0++){
		    i1++;
		    Chi0[i1] = RF[L0][Mul0]*AF[L0][M0];
		  }
		}
	      }

              /* end of inlining of Get_Orbitals */

	    } /* if (Cnt_kind==0) */

	    else{
              Get_Cnt_Orbitals(Mh_AN,x,y,z,Chi0);
	    }

	    for (j=0; j<NO1; j++){
	      Orbs_Grid_FNAN[Mc_AN][h_AN][Nog][j] = (Type_Orbs_Grid)Chi0[j];/* AITUNE */
	    }

	  } /* Nog */

          /* freeing of arrays */

	  free(Chi0);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(RF[i]);
	  }
	  free(RF);

	  for (i=0; i<(List_YOUSO[25]+1); i++){
	    free(AF[i]);
	  }
	  free(AF);

        } 
      }
    }
  }

  /* time */
  dtime(&TEtime);
  time0 = TEtime - TStime;

  return time0;
}
