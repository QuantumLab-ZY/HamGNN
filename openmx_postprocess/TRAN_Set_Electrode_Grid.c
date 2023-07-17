/**********************************************************************
  TRAN_Set_Electrode_Grid.c:

  TRAN_Set_Electrode_Grid.c is a subroutine to calculate charge density 
  near the boundary region of the extended system,  contributed from 
  the electrodes.

  The contribution is added to the regions [0:TRAN_grid_bound[0]] and 
  [TRAN_grid_bound[1]:Ngrid1-1].

  Log of TRAN_Set_Electrode_Grid.c:

     24/July/2008  Released by T.Ozaki

***********************************************************************/
/* revised by Y. Xiao for Noncollinear NEGF calculations */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"
#include <fftw3.h> 
#include "tran_variables.h"
#include "tran_prototypes.h" 

#define print_stdout 0

double Interpolated_Func(double R, double *xg, double *v, int N);
double Dot_Product(double a[4], double b[4]);



/*
 *   input
 *             dDen_Grid_e
 *             dVHart_Grid_e
 *   
 *   output 
 *             double **ElectrodeDensity_Grid
 *             double **VHart_Boundary
 */


/*
 *  This is also used for Vtot_Grid 
 */


/* implicit input from trans_variables.h 
 *      double  **dDen_Grid_e
 *      double  **dVHart_Grid_e
 */






void TRAN_Set_Electrode_Grid(MPI_Comm comm1,
                             int *TRAN_Poisson_flag2,
			     double *Grid_Origin,   /* origin of the grid */
			     double tv[4][4],       /* unit vector of the cell*/
			     double Left_tv[4][4],  /* unit vector  left */
			     double Right_tv[4][4], /* unit vector right */
			     double gtv[4][4],      /* unit vector of the grid point, which is gtv*integer */
			     int Ngrid1,
			     int Ngrid2,
			     int Ngrid3             /* # of c grid points */
			     )
{
  int l1[2];
  int i,j,k,k2,k3;
  int tnoA,tnoB,wanB,wanA;
  int GA_AN,MA_AN,Nc,GNc,LB_AN_e;
  int GB_AN;
  int Rn_e,GA_AN_e,GB_AN_e,GRc;
  int side,direction;
  int spin,p2,p3;
  int n1,n2,n3;
  int id,gidx;
  double rcutA,rcutB,r,r1,r2;
  double dx,dy,dz,xx;
  double dx1,dy1,dz1;
  double dx2,dy2,dz2;
  double x1,y1,z1;
  double x2,y2,z2;
  double sum,tmp;
  double offset[4];
  double R[4];
  double xyz[4];
  double *Chi0,*Chi1,*Chi2;
  int idim=1;
  int myid;
  fftw_complex *in, *out;
  fftw_plan p;
   
  MPI_Comm_rank(comm1, &myid);

  /* for passing TRAN_Poisson_flag to "DFT" */
  *TRAN_Poisson_flag2 = TRAN_Poisson_flag;
 
  if (myid==Host_ID){
    printf("<TRAN_Set_Electrode_Grid>\n");
  }

  /* allocation of array */

  Chi0 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  Chi1 = (double*)malloc(sizeof(double)*List_YOUSO[7]);
  Chi2 = (double*)malloc(sizeof(double)*List_YOUSO[7]);

  in  = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 
  out = fftw_malloc(sizeof(fftw_complex)*List_YOUSO[17]); 

  /* allocation of arrays */
  if (print_stdout){
    printf("%d %d %d %d %d\n",Ngrid1,Ngrid2,Ngrid3,TRAN_grid_bound[0], TRAN_grid_bound[1]);
  }

  for (side=0; side<2; side++) {
    ElectrodeDensity_Grid[side] = (double**)malloc(sizeof(double*)*(SpinP_switch_e[side]+1));
  }

  /* left lead */

  side  = 0;
  l1[0] = 0;
  l1[1] = TRAN_grid_bound[0];
  for (spin=0; spin<=SpinP_switch_e[side]; spin++) {
    ElectrodeDensity_Grid[side][spin] = (double*)malloc(sizeof(double)*Ngrid3*Ngrid2*(l1[1]-l1[0]+1));
  }

  ElectrodeADensity_Grid[side] = (double*)malloc(sizeof(double)*Ngrid3*Ngrid2*(l1[1]-l1[0]+1));
  ElectrodedVHart_Grid[side] = (double*)malloc(sizeof(double)*Ngrid3*Ngrid2*(l1[1]-l1[0]+1));

  VHart_Boundary[side] = (dcomplex***)malloc(sizeof(dcomplex**)*Ngrid1_e[side]);
  for (n1=0; n1<Ngrid1_e[side]; n1++){
    VHart_Boundary[side][n1] = (dcomplex**)malloc(sizeof(dcomplex*)*Ngrid2);
    for (n2=0; n2<Ngrid2; n2++){
      VHart_Boundary[side][n1][n2] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
    }
  }

  /* right lead */

  side  = 1;
  l1[0] = TRAN_grid_bound[1];
  l1[1] = Ngrid1 - 1;
  for (spin=0; spin<=SpinP_switch_e[side]; spin++) {
    ElectrodeDensity_Grid[side][spin] = (double*)malloc(sizeof(double)*Ngrid3*Ngrid2*(l1[1]-l1[0]+1));
  }

  ElectrodeADensity_Grid[side] = (double*)malloc(sizeof(double)*Ngrid3*Ngrid2*(l1[1]-l1[0]+1));
  ElectrodedVHart_Grid[side] = (double*)malloc(sizeof(double)*Ngrid3*Ngrid2*(l1[1]-l1[0]+1));

  VHart_Boundary[side] = (dcomplex***)malloc(sizeof(dcomplex**)*Ngrid1_e[side]);
  for (n1=0; n1<Ngrid1_e[side]; n1++){
    VHart_Boundary[side][n1] = (dcomplex**)malloc(sizeof(dcomplex*)*Ngrid2);
    for (n2=0; n2<Ngrid2; n2++){
      VHart_Boundary[side][n1][n2] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
    }
  }

  /*******************************************************
   charge density contributed by the left and right sides 
  *******************************************************/

  for (side=0; side<=1; side++){

    if (side==0){
      direction = -1;
      l1[0] = 0;
      l1[1] = TRAN_grid_bound[0];
    }
    else{
      direction = 1;
      l1[0] = TRAN_grid_bound[1];
      l1[1] = Ngrid1-1;
    }

    /* initialize ElectrodeDensity_Grid and ElectrodeADensity_Grid */

    for (spin=0; spin<=SpinP_switch_e[side]; spin++) {
      for (i=0; i<Ngrid3*Ngrid2*(l1[1]-l1[0]+1); i++){
	ElectrodeDensity_Grid[side][spin][i] = 0.0;
      }
    }
    for (i=0; i<Ngrid3*Ngrid2*(l1[1]-l1[0]+1); i++){
      ElectrodeADensity_Grid[side][i] = 0.0;
    }

    /* calculate charge density */
 
    for (n1=l1[0]; n1<=l1[1]; n1++){
      for (n2=0; n2<Ngrid2; n2++){
	for (n3=0; n3<Ngrid3; n3++){

	  GNc = n1*Ngrid2*Ngrid3 + n2*Ngrid3 + n3;
	  Get_Grid_XYZ(GNc, xyz); 

	  for (p2=-1; p2<=1; p2++){
	    for (p3=-1; p3<=1; p3++){
	      for (GA_AN_e=1; GA_AN_e<=atomnum_e[side]; GA_AN_e++){

                if (side==0) GA_AN = GA_AN_e;
                else         GA_AN = GA_AN_e + Latomnum + Catomnum;

		wanA = WhatSpecies[GA_AN];
		tnoA = Spe_Total_CNO[wanA];
		rcutA = Spe_Atom_Cut1[wanA];

		x1 = Gxyz[GA_AN][1]
                   + (double)direction*tv_e[side][1][1] 
                   +        (double)p2*tv_e[side][2][1] 
                   +        (double)p3*tv_e[side][3][1];

		y1 = Gxyz[GA_AN][2] 
                   + (double)direction*tv_e[side][1][2] 
                   +        (double)p2*tv_e[side][2][2] 
                   +        (double)p3*tv_e[side][3][2];

		z1 = Gxyz[GA_AN][3] 
                   + (double)direction*tv_e[side][1][3] 
                   +        (double)p2*tv_e[side][2][3] 
                   +        (double)p3*tv_e[side][3][3];

		dx1 = xyz[1] - x1; 
		dy1 = xyz[2] - y1; 
		dz1 = xyz[3] - z1; 
		r1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1);

		if (r1<=rcutA){

                  /******************************
                       ElectrodeDensity_Grid
                  ******************************/

		  Get_Orbitals(wanA,dx1,dy1,dz1,Chi1);

		  for (LB_AN_e=0; LB_AN_e<=FNAN_e[side][GA_AN_e]; LB_AN_e++){

		    GB_AN_e = natn_e[side][GA_AN_e][LB_AN_e];
		    Rn_e    =  ncn_e[side][GA_AN_e][LB_AN_e];

		    if (side==0) GB_AN = GB_AN_e;
		    else         GB_AN = GB_AN_e + Latomnum + Catomnum;

		    wanB = WhatSpecies[GB_AN];
		    tnoB = Spe_Total_CNO[wanB];
		    rcutB = Spe_Atom_Cut1[wanB];

		    x2 = Gxyz[GB_AN][1]
		      + (double)(direction+atv_ijk_e[side][Rn_e][1])*tv_e[side][1][1]
		      + (double)(p2       +atv_ijk_e[side][Rn_e][2])*tv_e[side][2][1]
		      + (double)(p3       +atv_ijk_e[side][Rn_e][3])*tv_e[side][3][1];

		    y2 = Gxyz[GB_AN][2] 
		      + (double)(direction+atv_ijk_e[side][Rn_e][1])*tv_e[side][1][2]
		      + (double)(p2       +atv_ijk_e[side][Rn_e][2])*tv_e[side][2][2]
		      + (double)(p3       +atv_ijk_e[side][Rn_e][3])*tv_e[side][3][2];

		    z2 = Gxyz[GB_AN][3] 
		      + (double)(direction+atv_ijk_e[side][Rn_e][1])*tv_e[side][1][3]
		      + (double)(p2       +atv_ijk_e[side][Rn_e][2])*tv_e[side][2][3]
		      + (double)(p3       +atv_ijk_e[side][Rn_e][3])*tv_e[side][3][3];

		    dx2 = xyz[1] - x2;
		    dy2 = xyz[2] - y2; 
		    dz2 = xyz[3] - z2; 
		    r2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);

		    if (r2<=rcutB){

		      Get_Orbitals(wanB,dx2,dy2,dz2,Chi2);

		      for (spin=0; spin<=SpinP_switch; spin++) {

			if (atv_ijk_e[side][Rn_e][1]==(-direction)){

			  sum = 0.0;
			  for (i=0; i<tnoA; i++){
			    for (j=0; j<tnoB; j++){
                   /* revised by Y. Xiao for Noncollinear NEGF calculations */
			    /*  sum += 2.0*DM_e[side][0][spin][GA_AN_e][LB_AN_e][i][j]*Chi1[i]*Chi2[j]; */
                               if(spin==3){
                                  sum -= 2.0*DM_e[side][0][spin][GA_AN_e][LB_AN_e][i][j]*Chi1[i]*Chi2[j];
                               } else {
                                  sum += 2.0*DM_e[side][0][spin][GA_AN_e][LB_AN_e][i][j]*Chi1[i]*Chi2[j];
                               }
                    /* until here by Y. Xiao for Noncollinear NEGF calculations */
			    }
			  }
			}

			else if (atv_ijk_e[side][Rn_e][1]==0){

			  sum = 0.0;
			  for (i=0; i<tnoA; i++){
			    for (j=0; j<tnoB; j++){
                     /* revised by Y. Xiao for Noncollinear NEGF calculations */
			    /*  sum += DM_e[side][0][spin][GA_AN_e][LB_AN_e][i][j]*Chi1[i]*Chi2[j]; */
                               if(spin==3){
                                  sum -= DM_e[side][0][spin][GA_AN_e][LB_AN_e][i][j]*Chi1[i]*Chi2[j];
                               } else {
                                  sum += DM_e[side][0][spin][GA_AN_e][LB_AN_e][i][j]*Chi1[i]*Chi2[j];
                               }
                     /* until here by Y. Xiao for Noncollinear NEGF calculations */
			    }
			  }
			}

			gidx = (n1-l1[0])*Ngrid2*Ngrid3 + n2*Ngrid3 + n3; 
			ElectrodeDensity_Grid[side][spin][gidx] += sum;

		      }
		    }
		  }

                  /******************************
                       ElectrodeADensity_Grid
                  ******************************/

  		  xx = 0.5*log(dx1*dx1 + dy1*dy1 + dz1*dz1);
                  gidx = (n1-l1[0])*Ngrid2*Ngrid3 + n2*Ngrid3 + n3; 
                  ElectrodeADensity_Grid[side][gidx] += 0.5*KumoF( Spe_Num_Mesh_PAO[wanA], xx, 
                                                            Spe_PAO_XV[wanA], Spe_PAO_RV[wanA], 
                                                            Spe_Atomic_Den[wanA]);

		} /* if (r1<=rcutA) */              
	      }
	    }
	  }
	}
      }
    }

  } /* side */

  /*******************************************************
   2D FFT of dDen_Grid_e, which is difference between
   charge density calculated by KS wave functions and 
   the superposition of atomic charge density, of the 
   left and right leads on the bc plane for TRAN_Poisson
  *******************************************************/

  for (side=0; side<=1; side++){

    /* set VHart_Boundary in real space */

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (n2=0; n2<Ngrid2; n2++){
	for (n3=0; n3<Ngrid3; n3++){

          /* borrow the array "VHart_Boundary" */
          VHart_Boundary[side][n1][n2][n3].r = dDen_Grid_e[side][n1*Ngrid2*Ngrid3+n2*Ngrid3+n3];
	  VHart_Boundary[side][n1][n2][n3].i = 0.0;
	}
      }
    }

    /* FFT of VHart_Boundary for c-axis */

    p = fftw_plan_dft_1d(Ngrid3,in,out,-1,FFTW_ESTIMATE);

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (n2=0; n2<Ngrid2; n2++){

	for (n3=0; n3<Ngrid3; n3++){

	  in[n3][0] = VHart_Boundary[side][n1][n2][n3].r;
	  in[n3][1] = VHart_Boundary[side][n1][n2][n3].i;
	}  

	fftw_execute(p);

	for (k3=0; k3<Ngrid3; k3++){

	  VHart_Boundary[side][n1][n2][k3].r = out[k3][0];
	  VHart_Boundary[side][n1][n2][k3].i = out[k3][1];
	}
      }
    }

    fftw_destroy_plan(p);  

    /* FFT of VHart_Boundary for b-axis */

    p = fftw_plan_dft_1d(Ngrid2,in,out,-1,FFTW_ESTIMATE);

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (k3=0; k3<Ngrid3; k3++){

	for (n2=0; n2<Ngrid2; n2++){

	  in[n2][0] = VHart_Boundary[side][n1][n2][k3].r;
	  in[n2][1] = VHart_Boundary[side][n1][n2][k3].i;
	}

	fftw_execute(p);

	for (k2=0; k2<Ngrid2; k2++){

	  VHart_Boundary[side][n1][k2][k3].r = out[k2][0];
	  VHart_Boundary[side][n1][k2][k3].i = out[k2][1];
	}
      }
    }

    fftw_destroy_plan(p);  

    tmp = 1.0/(double)(Ngrid2*Ngrid3); 

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (k2=0; k2<Ngrid2; k2++){
	for (k3=0; k3<Ngrid3; k3++){
	  VHart_Boundary[side][n1][k2][k3].r *= tmp;
	  VHart_Boundary[side][n1][k2][k3].i *= tmp;
	}
      }
    }

  } /* side */

  /*******************************************************
   interpolation of 2D Fourier transformed difference
   charge of the left and right leads on the bc plane for 
   TRAN_Poisson_FFT_Extended
  *******************************************************/

  {
    int ip,Num;
    double x; 
    double *vr,*vi,*xg;

    if      (1.0<fabs(tv[1][1])) ip = 1;
    else if (1.0<fabs(tv[1][2])) ip = 2;
    else if (1.0<fabs(tv[1][3])) ip = 3;

    side = 0;
    IntNgrid1_e[side] = (int)(fabs((double)TRAN_FFTE_CpyNum*tv_e[side][1][ip]+1.0e-8)/length_gtv[1]);

    dDen_IntBoundary[side] = (dcomplex***)malloc(sizeof(dcomplex**)*IntNgrid1_e[side]);
    for (n1=0; n1<IntNgrid1_e[side]; n1++){
      dDen_IntBoundary[side][n1] = (dcomplex**)malloc(sizeof(dcomplex*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
	dDen_IntBoundary[side][n1][n2] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
      }
    }

    side = 1;
    IntNgrid1_e[side] = (int)(fabs((double)TRAN_FFTE_CpyNum*tv_e[side][1][ip]+1.0e-8)/length_gtv[1]);

    dDen_IntBoundary[side] = (dcomplex***)malloc(sizeof(dcomplex**)*IntNgrid1_e[side]);
    for (n1=0; n1<IntNgrid1_e[side]; n1++){
      dDen_IntBoundary[side][n1] = (dcomplex**)malloc(sizeof(dcomplex*)*Ngrid2);
      for (n2=0; n2<Ngrid2; n2++){
	dDen_IntBoundary[side][n1][n2] = (dcomplex*)malloc(sizeof(dcomplex)*Ngrid3);
      }
    }

    for (side=0; side<=1; side++){

      vr = (double*)malloc(sizeof(double)*(TRAN_FFTE_CpyNum+2)*Ngrid1_e[side]);
      vi = (double*)malloc(sizeof(double)*(TRAN_FFTE_CpyNum+2)*Ngrid1_e[side]);
      xg = (double*)malloc(sizeof(double)*(TRAN_FFTE_CpyNum+2)*Ngrid1_e[side]);

      for (k2=0; k2<Ngrid2; k2++){
	for (k3=0; k3<Ngrid3; k3++){

          /* set up the 1D data */

          for (i=0; i<(TRAN_FFTE_CpyNum+2); i++){ 

  	    for (n1=0; n1<Ngrid1_e[side]; n1++){
	      vr[n1+i*Ngrid1_e[side]] = VHart_Boundary[side][n1][k2][k3].r;
	      vi[n1+i*Ngrid1_e[side]] = VHart_Boundary[side][n1][k2][k3].i;
	    } 

	    for (n1=0; n1<Ngrid1_e[side]; n1++){
              xg[n1+i*Ngrid1_e[side]] =
                            (double)(n1+i*Ngrid1_e[side])*fabs(tv_e[side][1][ip]/(double)Ngrid1_e[side])
                          - (double)(Ngrid1_e[side]*(TRAN_FFTE_CpyNum+1))*fabs(tv_e[side][1][ip]/(double)Ngrid1_e[side])
                          + Grid_Origin[ip];
	    }
	  }

	  /*
	  if (myid==0 && side==0 && k2==1 && k3==0){
	    for (n1=0; n1<Ngrid1_e[side]*(TRAN_FFTE_CpyNum+2); n1++){
	      printf("A %15.12f %15.12f %15.12f\n",xg[n1],vr[n1],vi[n1]);
	    }
          }
	  */

          /* interpolation */

	  for (n1=0; n1<IntNgrid1_e[side]; n1++){

            x = (double)n1*length_gtv[1] - (double)IntNgrid1_e[side]*length_gtv[1] + Grid_Origin[ip];

            dDen_IntBoundary[side][n1][k2][k3].r = Interpolated_Func(x, xg, vr, (TRAN_FFTE_CpyNum+2)*Ngrid1_e[side]);
            dDen_IntBoundary[side][n1][k2][k3].i = Interpolated_Func(x, xg, vi, (TRAN_FFTE_CpyNum+2)*Ngrid1_e[side]);


	    /*
            if (myid==0 && side==0 && k2==1 && k3==0){
            printf("B %15.12f %15.12f %15.12f\n",
                   x,dDen_IntBoundary[side][n1][k2][k3].r,
                     dDen_IntBoundary[side][n1][k2][k3].i);
	    }
	    */

	  }
	}
      }    

      free(xg);
      free(vi);
      free(vr);

    }
  }

  /****************************************************
    2D FFT of dVHartree of the left and right leads 
    on the bc plane for TRAN_Poisson
  ****************************************************/

  for (side=0; side<=1; side++){

    /* set VHart_Boundary in real space */

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (n2=0; n2<Ngrid2; n2++){
	for (n3=0; n3<Ngrid3; n3++){

          VHart_Boundary[side][n1][n2][n3].r = dVHart_Grid_e[side][n1*Ngrid2*Ngrid3+n2*Ngrid3+n3];
	  VHart_Boundary[side][n1][n2][n3].i = 0.0;
	}
      }
    }

    /* FFT of VHart_Boundary for c-axis */

    p = fftw_plan_dft_1d(Ngrid3,in,out,-1,FFTW_ESTIMATE);

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (n2=0; n2<Ngrid2; n2++){

	for (n3=0; n3<Ngrid3; n3++){

	  in[n3][0] = VHart_Boundary[side][n1][n2][n3].r;
	  in[n3][1] = VHart_Boundary[side][n1][n2][n3].i;
	}  

	fftw_execute(p);

	for (k3=0; k3<Ngrid3; k3++){

	  VHart_Boundary[side][n1][n2][k3].r = out[k3][0];
	  VHart_Boundary[side][n1][n2][k3].i = out[k3][1];
	}
      }
    }

    fftw_destroy_plan(p);  

    /* FFT of VHart_Boundary for b-axis */

    p = fftw_plan_dft_1d(Ngrid2,in,out,-1,FFTW_ESTIMATE);

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (k3=0; k3<Ngrid3; k3++){

	for (n2=0; n2<Ngrid2; n2++){

	  in[n2][0] = VHart_Boundary[side][n1][n2][k3].r;
	  in[n2][1] = VHart_Boundary[side][n1][n2][k3].i;
	}

	fftw_execute(p);

	for (k2=0; k2<Ngrid2; k2++){

	  VHart_Boundary[side][n1][k2][k3].r = out[k2][0];
	  VHart_Boundary[side][n1][k2][k3].i = out[k2][1];
	}
      }
    }

    fftw_destroy_plan(p);  

    tmp = 1.0/(double)(Ngrid2*Ngrid3); 

    for (n1=0; n1<Ngrid1_e[side]; n1++){
      for (k2=0; k2<Ngrid2; k2++){
	for (k3=0; k3<Ngrid3; k3++){
	  VHart_Boundary[side][n1][k2][k3].r *= tmp;
	  VHart_Boundary[side][n1][k2][k3].i *= tmp;
	}
      }
    }

  } /* side */

  /* freeing of arrays */
  free(Chi0);
  free(Chi1);
  free(Chi2);

  fftw_free(in);
  fftw_free(out);
}




double Interpolated_Func(double R, double *xg, double *v, int N)
{
  int mp_min,mp_max,m;
  double h1,h2,h3,f1,f2,f3,f4;
  double g1,g2,x1,x2,y1,y2,f;
  double result;

  mp_min = 0;
  mp_max = N - 1;

  if (R<xg[0]){
    printf("Error #1 in Interpolated_Func\n");
    exit(0);
  }
  else if (xg[N-1]<R){
    printf("Error #2 in Interpolated_Func\n");
    exit(0);
  }
  else{

    do{
      m = (mp_min + mp_max)/2;
      if (xg[m]<R)
        mp_min = m;
      else 
        mp_max = m;
    }
    while((mp_max-mp_min)!=1);
    m = mp_max;

    if (m<2)
      m = 2;
    else if (N<=m)
      m = N - 2;

    /****************************************************
                   spline like interpolation
    ****************************************************/

    if (m==1){

      h2 = xg[m]   - xg[m-1];
      h3 = xg[m+1] - xg[m];

      f2 = v[m-1];
      f3 = v[m];
      f4 = v[m+1];

      h1 = -(h2+h3);
      f1 = f4;
    }
    else if (m==(N-1)){

      h1 = xg[m-1] - xg[m-2];
      h2 = xg[m]   - xg[m-1];

      f1 = v[m-2];
      f2 = v[m-1];
      f3 = v[m];

      h3 = -(h1+h2);
      f4 = f1;
    }
    else{
      h1 = xg[m-1] - xg[m-2];
      h2 = xg[m]   - xg[m-1];
      h3 = xg[m+1] - xg[m];

      f1 = v[m-2];
      f2 = v[m-1];
      f3 = v[m];
      f4 = v[m+1];
    }

    /****************************************************
                calculate the value at R
    ****************************************************/

    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);

    x1 = R - xg[m-1];
    x2 = R - xg[m];
    y1 = x1/h2;
    y2 = x2/h2;

    f =  y2*y2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y1*y1*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
  }
  result = f;
  return result;
}
