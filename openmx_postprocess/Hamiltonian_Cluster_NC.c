/**********************************************************************
  Hamiltonian_Cluster_NC.c:

     Hamiltonian_Cluster_NC.c is a subroutine to make a Hamiltonian
     matrix for cluster or molecular systems with non-collinear spin.

  Log of Hamiltonian_Cluster_NC.c:

     17/Dec/2003  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "mpi.h"


void Hamiltonian_Cluster_NC(double *****RH, double *****IH,
                            dcomplex **H, int *MP)
{
  int i,j,k,q;
  int MA_AN,GA_AN,LB_AN,GB_AN,AN;
  int wanA,wanB,tnoA,tnoB,Anum,Bnum,NUM;
  int num,tnum,num_orbitals,qmax;
  int ID,myid,numprocs,tag=999;
  int *My_NZeros;
  int *is1,*ie1,*is2;
  int *My_Matomnum,*order_GA;
  double *H1,sum;

  MPI_Status stat;
  MPI_Request request;

  /* MPI */

  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);
  MPI_Barrier(mpi_comm_level1);

  /* allocation of arrays */

  My_NZeros = (int*)malloc(sizeof(int)*numprocs);
  My_Matomnum = (int*)malloc(sizeof(int)*numprocs);
  is1 = (int*)malloc(sizeof(int)*numprocs);
  ie1 = (int*)malloc(sizeof(int)*numprocs);
  is2 = (int*)malloc(sizeof(int)*numprocs);
  order_GA = (int*)malloc(sizeof(int)*(atomnum+2));

  /* find my total number of non-zero elements in myid */

  My_NZeros[myid] = 0;
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    GA_AN = M2G[MA_AN];
    wanA = WhatSpecies[GA_AN];
    tnoA = Spe_Total_CNO[wanA];

    num = 0;      
    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
      GB_AN = natn[GA_AN][LB_AN];
      wanB = WhatSpecies[GB_AN];
      tnoB = Spe_Total_CNO[wanB];
      num += tnoB;
    }

    My_NZeros[myid] += tnoA*num;
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_NZeros[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  tnum = 0;
  for (ID=0; ID<numprocs; ID++){
    tnum += My_NZeros[ID];
  }  

  is1[0] = 0;
  ie1[0] = My_NZeros[0] - 1;

  for (ID=1; ID<numprocs; ID++){
    is1[ID] = ie1[ID-1] + 1;
    ie1[ID] = is1[ID] + My_NZeros[ID] - 1;
  }  

  /* set is2 and order_GA */

  My_Matomnum[myid] = Matomnum;
  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&My_Matomnum[ID],1,MPI_INT,ID,mpi_comm_level1);
  }

  is2[0] = 1;
  for (ID=1; ID<numprocs; ID++){
    is2[ID] = is2[ID-1] + My_Matomnum[ID-1];
  }
  
  for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
    order_GA[is2[myid]+MA_AN-1] = M2G[MA_AN];
  }

  for (ID=0; ID<numprocs; ID++){
    MPI_Bcast(&order_GA[is2[ID]],My_Matomnum[ID],MPI_INT,ID,mpi_comm_level1);
  }

  /* set MP */

  Anum = 1;
  for (i=1; i<=atomnum; i++){
    MP[i] = Anum;
    wanA = WhatSpecies[i];
    Anum = Anum + Spe_Total_CNO[wanA];
  }
  NUM = Anum - 1;
  H[0][0].r = NUM;

  /* set H1 */

  H1 = (double*)malloc(sizeof(double)*(tnum+1));

  /* initialize H */

  for (i=1; i<=2*NUM; i++){
    for (j=1; j<=2*NUM; j++){
      H[i][j].r = 0.0;
      H[i][j].i = 0.0;
    }
  }

  /****************************************************************************
    in case of SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
               && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0

    H[i    ][j    ].r = RH[0];
    H[i    ][j    ].i = 0.0;
    H[i+NUM][j+NUM].r = RH[1];
    H[i+NUM][j+NUM].i = 0.0;
    H[i    ][j+NUM].r = RH[2];
    H[i    ][j+NUM].i = RH[3];

    in case of SO_switch==1 or Hub_U_switch==1 or 1<=Constraint_NCS_switch 
               or Zeeman_NCS_switch==1 or Zeeman_NCO_switch==1 

    H[i    ][j    ].r = RH[0];  
    H[i    ][j    ].i = IH[0];
    H[i+NUM][j+NUM].r = RH[1];
    H[i+NUM][j+NUM].i = IH[1];
    H[i    ][j+NUM].r = RH[2];
    H[i    ][j+NUM].i = RH[3] + IH[2];
  ****************************************************************************/

  /* non-spin-orbit coupling and non-LDA+U */

  if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
      && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0)
    qmax = 4;
  else 
    qmax = 6;

  for (q=0; q<qmax; q++){

    k = is1[myid];
    for (MA_AN=1; MA_AN<=Matomnum; MA_AN++){
      GA_AN = M2G[MA_AN];    
      wanA = WhatSpecies[GA_AN];
      tnoA = Spe_Total_CNO[wanA];
      for (i=0; i<tnoA; i++){
	for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	  GB_AN = natn[GA_AN][LB_AN];
	  wanB = WhatSpecies[GB_AN];
	  tnoB = Spe_Total_CNO[wanB];

          /* non-spin-orbit coupling and non-LDA+U */

          if (SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
              && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){
	    for (j=0; j<tnoB; j++){
	      H1[k] = RH[q][MA_AN][LB_AN][i][j];
	      k++;
	    }
	  }

          /* spin-orbit coupling or LDA+U */

          else{

            switch(q){

  	    case 0:
        
	      for (j=0; j<tnoB; j++){
		H1[k] = RH[0][MA_AN][LB_AN][i][j]; 
		k++;
	      }

              break;

  	    case 1:
        
	      for (j=0; j<tnoB; j++){
		H1[k] = RH[1][MA_AN][LB_AN][i][j]; 
		k++;
	      }

              break;

  	    case 2:
        
	      for (j=0; j<tnoB; j++){
		H1[k] = RH[2][MA_AN][LB_AN][i][j]; 
		k++;
	      }

              break;

  	    case 3:
        
	      for (j=0; j<tnoB; j++){
		H1[k] = IH[0][MA_AN][LB_AN][i][j]; 
		k++;
	      }

              break;

  	    case 4:
        
	      for (j=0; j<tnoB; j++){
		H1[k] = IH[1][MA_AN][LB_AN][i][j]; 
		k++;
	      }

              break;

  	    case 5:
        
	      for (j=0; j<tnoB; j++){
		H1[k] = RH[3][MA_AN][LB_AN][i][j] + IH[2][MA_AN][LB_AN][i][j]; 
		k++;
	      }

              break;

            } /* switch(q) */
	  } /* else */
	}
      }
    }

    /* MPI H1 */
    
    for (ID=0; ID<numprocs; ID++){
      k = is1[ID];
      MPI_Bcast(&H1[k], My_NZeros[ID], MPI_DOUBLE, ID, mpi_comm_level1);
    }

    /* H1 -> H */

    /* non-spin-orbit coupling and non-LDA+U */

    if ( SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 
         && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0){

      switch(q){

      case 0:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j].r += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 1:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i+NUM][Bnum+j+NUM].r += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 2:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j+NUM].r += H1[k];
		H[Bnum+j+NUM][Anum+i].r += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 3:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j+NUM].i += H1[k];
		H[Bnum+j+NUM][Anum+i].i +=-H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      } /* switch(q) */     
    } /* if ( SO_switch==0 && Hub_U_switch==0 && Constraint_NCS_switch==0 && Zeeman_NCS_switch==0 && Zeeman_NCO_switch==0) */

    /* in case of SO_switch==1 or Hub_U_switch==1 or 1<=Constraint_NCS_switch or Zeeman_NCS_switch==1 or Zeeman_NCO_switch==1 */

    else{ 

      switch(q){

      case 0:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j].r += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 1:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i+NUM][Bnum+j+NUM].r += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 2:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j+NUM].r += H1[k];
		H[Bnum+j+NUM][Anum+i].r += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 3:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j].i += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 4:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i+NUM][Bnum+j+NUM].i += H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      case 5:

	k = 0;
	for (AN=1; AN<=atomnum; AN++){
	  GA_AN = order_GA[AN];
	  wanA = WhatSpecies[GA_AN];
	  tnoA = Spe_Total_CNO[wanA];
	  Anum = MP[GA_AN];
	  for (i=0; i<tnoA; i++){
	    for (LB_AN=0; LB_AN<=FNAN[GA_AN]; LB_AN++){
	      GB_AN = natn[GA_AN][LB_AN];
	      wanB = WhatSpecies[GB_AN];
	      tnoB = Spe_Total_CNO[wanB];
	      Bnum = MP[GB_AN];
	      for (j=0; j<tnoB; j++){
		H[Anum+i][Bnum+j+NUM].i += H1[k];
		H[Bnum+j+NUM][Anum+i].i +=-H1[k];
		k++;
	      }
	    }
	  }
	}

	break;

      } /* switch(q) */     
    } /* else */
  } /* for (q=0;... */

  /* freeing of arrays */

  free(My_NZeros);
  free(My_Matomnum);
  free(is1);
  free(ie1);
  free(is2);
  free(order_GA);
  free(H1);
}


