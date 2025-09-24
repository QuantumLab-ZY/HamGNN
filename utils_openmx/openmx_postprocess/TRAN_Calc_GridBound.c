/**********************************************************************
  TRAN_Calc_GridBound.c:

  TRAN_Calc_GridBound.c is a subroutine to find the grid boundary region.

  output: TRAN_region[] is changed 
          TRAN_grid_bound[2];
 
  The boundary regions are defined by [0:TRAN_grid_bound[0]] and [TRAN_grid_bound[1]: Ngrid1-1] 


  Log of TRAN_Calc_GridBound.c:

     24/July/2008  Released by H.Kino and T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "tran_variables.h"
#include "tran_prototypes.h"

 
#ifdef MAX 
#undef MAX
#endif
#define MAX(a,b) (((a)>(b))?  (a):(b)) 

#ifdef MIN
#undef MIN
#endif
#define MIN(a,b) (((a)<(b))?  (a):(b))


static void Cross_Product(double a[4], double b[4], double c[4]);
static double Dot_Product(double a[4], double b[4]); 


void TRAN_Calc_GridBound(MPI_Comm mpi_comm_level1,
			 int atomnum,
			 int *WhatSpecies,
			 double *Spe_Atom_Cut1,
			 int Ngrid1, 
			 double *Grid_Origin, 
			 double **Gxyz,
			 double tv[4][4],
			 double gtv[4][4],
			 double rgtv[4][4],
			 double Left_tv[4][4],
			 double Right_tv[4][4])
{
  int ct_AN,ct_AN2,wanA,wanB;
  int i,j,po;
  double r,rcutA,rcutB;
  double dx,dy,dz;
  double MinV,MaxV;
  int MaxGridNum;
  double R[4],Cxyz[4];
  double b[4],c[4],v[4];
  double coef,Vec0,Vec1;
  int myid,numprocs;

  /* MPI */ 
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (i=1; i<=3; i++) { R[i] = tv[1][i] - Right_tv[1][i]; }

  if (myid==Host_ID){
    printf("\n\n<TRAN_Calc_GridBound>\n\n");
  }

  /******************************************************
    find the unit vector perpendicular to the bc-plane
  ******************************************************/

  b[1] = tv[2][1];
  b[2] = tv[2][2];
  b[3] = tv[2][3];

  c[1] = tv[3][1];
  c[2] = tv[3][2];
  c[3] = tv[3][3];

  Cross_Product(b,c,v);
  coef = 1.0/sqrt(fabs( Dot_Product(v,v) ));

  v[1] = coef*v[1];
  v[2] = coef*v[2];
  v[3] = coef*v[3];

  /********************************************************** 
                          left side
  ***********************************************************/

  MinV =  1.0e+10;
  MaxV = -1.0e+10;

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

    if ( TRAN_region[ct_AN]==2 ) {  /* left region */

      wanA = WhatSpecies[ct_AN];
      rcutA = Spe_Atom_Cut1[wanA];

      /* find Vec0 and Vec1 */

      Cxyz[1] = Gxyz[ct_AN][1] + rcutA*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[ct_AN][2] + rcutA*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[ct_AN][3] + rcutA*v[3] - Grid_Origin[3];

      Vec0 = Dot_Product(Cxyz,rgtv[1])*0.5/PI;

      Cxyz[1] = Gxyz[ct_AN][1] - rcutA*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[ct_AN][2] - rcutA*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[ct_AN][3] - rcutA*v[3] - Grid_Origin[3];

      Vec1 = Dot_Product(Cxyz,rgtv[1])*0.5/PI;

      if (Vec0<MinV) MinV = Vec0;
      if (Vec1<MinV) MinV = Vec1;
      if (MaxV<Vec0) MaxV = Vec0;   
      if (MaxV<Vec1) MaxV = Vec1;

      /* set "12" atoms having non-zero overlap with atoms in the left lead */ 

      ct_AN2 = 1;
      po = 0;

      do { 

        if ( TRAN_region[ct_AN2]==2 ) {  /* left region */

          wanB = WhatSpecies[ct_AN2];
          rcutB = Spe_Atom_Cut1[wanB];

          dx = Gxyz[ct_AN][1] - (Gxyz[ct_AN2][1] - Left_tv[1][1]);
          dy = Gxyz[ct_AN][2] - (Gxyz[ct_AN2][2] - Left_tv[1][2]);
          dz = Gxyz[ct_AN][3] - (Gxyz[ct_AN2][3] - Left_tv[1][3]);

          r = sqrt(dx*dx+dy*dy+dz*dz);

          if (r<(rcutA+rcutB) ){
            TRAN_region[ct_AN] += 10;
	    po = 1;
          }
	}

        ct_AN2++;

      } while (po==0 && ct_AN2<=atomnum);
    }
  }

  /***************************************************
   TRAN_grid_bound[0] is the maximum grid point that 
   basis functions of atoms belonging to the left lead    
   can ovelap.  
  ***************************************************/

  Vec0 = fabs(Dot_Product(Left_tv[1],rgtv[1])*0.5/PI);
  TRAN_grid_bound[0] = (int)(MaxV - Vec0) + 1;
  
  /*
  if (myid==Host_ID){
    printf("Left MinV=%15.12f MaxV=%15.12f\n",MinV,MaxV);
    printf("grid_bound (left) =%d\n",TRAN_grid_bound[0]);
  }
  */
  
  /********************************************************** 
                           right side 
  ***********************************************************/

  MinV =  1.0e+10;
  MaxV = -1.0e+10;

  MaxGridNum = Dot_Product(tv[1],rgtv[1])*0.5/PI;

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){

    if ( TRAN_region[ct_AN]==3 ) {  /* right region */

      wanA = WhatSpecies[ct_AN];
      rcutA = Spe_Atom_Cut1[wanA];

      /* find Vec0 and Vec1 */

      Cxyz[1] = Gxyz[ct_AN][1] + rcutA*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[ct_AN][2] + rcutA*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[ct_AN][3] + rcutA*v[3] - Grid_Origin[3];

      Vec0 = Dot_Product(Cxyz,rgtv[1])*0.5/PI;

      Cxyz[1] = Gxyz[ct_AN][1] - rcutA*v[1] - Grid_Origin[1];
      Cxyz[2] = Gxyz[ct_AN][2] - rcutA*v[2] - Grid_Origin[2];
      Cxyz[3] = Gxyz[ct_AN][3] - rcutA*v[3] - Grid_Origin[3];

      Vec1 = Dot_Product(Cxyz,rgtv[1])*0.5/PI;

      if (Vec0<MinV) MinV = Vec0;
      if (Vec1<MinV) MinV = Vec1;
      if (MaxV<Vec0) MaxV = Vec0;   
      if (MaxV<Vec1) MaxV = Vec1;

      /* set "13" atoms having non-zero overlap with atoms in the right lead */ 

      ct_AN2 = 1;
      po = 0;

      do { 

        if ( TRAN_region[ct_AN2]==3 ) {  /* right region */

          wanB = WhatSpecies[ct_AN2];
          rcutB = Spe_Atom_Cut1[wanB];

          dx = Gxyz[ct_AN][1] - (Gxyz[ct_AN2][1] + Right_tv[1][1]);
          dy = Gxyz[ct_AN][2] - (Gxyz[ct_AN2][2] + Right_tv[1][2]);
          dz = Gxyz[ct_AN][3] - (Gxyz[ct_AN2][3] + Right_tv[1][3]);

          r = sqrt(dx*dx+dy*dy+dz*dz);

          if (r<(rcutA+rcutB) ){
            TRAN_region[ct_AN] += 10;
	    po = 1;
          }
	}

        ct_AN2++;

      } while (po==0 && ct_AN2<=atomnum);

    }
  }

  /***************************************************
   TRAN_grid_bound[1] is the minimum grid point that 
   basis functions of atoms belonging to the right lead    
   can ovelap.  
  ***************************************************/

  Vec0 = fabs(Dot_Product(Right_tv[1],rgtv[1])*0.5/PI);
  TRAN_grid_bound[1]= (int)(MinV + Vec0);

  /*
  if (myid==Host_ID){
    printf("Right MinV=%15.12f MaxV=%15.12f\n",MinV,MaxV);
    printf("grid_bound (right) =%d\n",TRAN_grid_bound[1]);
  }
  */

  if (myid==Host_ID){

    printf("\n*******************************************************\n");      fflush(stdout);
    printf("The extended cell consists of Left0-Center-Right0.\n");             fflush(stdout);
    printf("The cells of left and right reads are connected as.\n");            fflush(stdout);
    printf("...|Left2|Left1|Left0-Center-Right0|Right1|Right2...\n\n");         fflush(stdout); 
    printf("Each atom in the extended cell is assigned as follows:\n");         fflush(stdout);
    printf("where '12' and '2' mean that they are in 'Left0', and\n");          fflush(stdout);
    printf("'12' has overlap with atoms in the Left1,\n");                      fflush(stdout);
    printf("and '13' and '3' mean that they are in 'Right0', and\n");           fflush(stdout);
    printf("'13' has overlap with atoms in the 'Right1', and also\n");          fflush(stdout);
    printf("'1' means atom in the 'Center'.\n");                                fflush(stdout);
    printf("********************************************************\n\n");     fflush(stdout);

    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      if (ct_AN<10) 
        printf("Atom%d  =%3d ",ct_AN,TRAN_region[ct_AN]);
      else if (ct_AN<100) 
        printf("Atom%d =%3d ",ct_AN,TRAN_region[ct_AN]);
      else if (ct_AN<1000) 
        printf("Atom%d=%3d ",ct_AN,TRAN_region[ct_AN]);

      if (ct_AN%7==0) printf("\n");
    }
    printf("\n\n");
  }

}




void Cross_Product(double a[4], double b[4], double c[4])
{
  c[1] = a[2]*b[3] - a[3]*b[2]; 
  c[2] = a[3]*b[1] - a[1]*b[3]; 
  c[3] = a[1]*b[2] - a[2]*b[1];
}

double Dot_Product(double a[4], double b[4])
{
  double sum;
  sum = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; 
  return sum;
}
