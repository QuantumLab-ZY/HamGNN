/*----------------------------------------------------------------------
  exx_rhox.c
----------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "openmx_common.h"
#include "exx_rhox.h"
#include "exx_log.h"
#include "exx_vector.h"


void EXX_Output_DM(EXX_t *exx, dcomplex ****exx_DM)
{
  int myrank, nproc;
  int i, j, iep, spin, iatm1, iatm2, icell;  
  int nep, nspin, natom, nb1, nb2;
  const int *ep_atom1, *ep_atom2, *ep_cell, *atom_nb;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t sz;

  natom = atomnum; 
  nspin = SpinP_switch+1; 
  nep = EXX_Number_of_EP(exx);
  ep_atom1 = EXX_Array_EP_Atom1(exx);
  ep_atom2 = EXX_Array_EP_Atom2(exx);
  ep_cell  = EXX_Array_EP_Cell(exx);
  atom_nb  = EXX_atom_nb(exx);

  snprintf(path, EXX_PATHLEN, "%s/exxdm.dat", EXX_CacheDir(exx));

  fp = fopen(path, "wb");
 
  /* spin */
  sz = fwrite(&nspin, sizeof(int), 1, fp); 
 
  /* atom */
  sz = fwrite(&natom, sizeof(int), 1, fp); 

  /* number of basis */
  sz = fwrite(&atom_nb[0], sizeof(int), natom, fp);

  /* exchange pairs */  
  sz = fwrite(&nep, sizeof(int), 1, fp); 
  sz = fwrite(ep_atom1, sizeof(int), nep, fp); 
  sz = fwrite(ep_atom2, sizeof(int), nep, fp); 
  sz = fwrite(ep_cell , sizeof(int), nep, fp); 
 
  for (spin=0; spin<nspin; spin++) {
    for (iep=0; iep<nep; iep++) {
      nb1 = atom_nb[ep_atom1[iep]];
      nb2 = atom_nb[ep_atom2[iep]];
      for (i=0; i<nb1; i++) {
        for (j=0; j<nb2; j++) {
          fwrite(&exx_DM[spin][iep][i][j].r, sizeof(double), 1, fp); 
          fwrite(&exx_DM[spin][iep][i][j].i, sizeof(double), 1, fp); 
        }
      }
    }  /* loop of iep */
  } /* loop of spin */

#if 0
  printf("EXX-DM:\n");
  for (spin=0; spin<nspin; spin++) {
    for (iep=0; iep<nep; iep++) {
      nb1 = atom_nb[ep_atom1[iep]];
      nb2 = atom_nb[ep_atom2[iep]];
      for (i=0; i<nb1; i++) {
        for (j=0; j<nb2; j++) {
          printf("  %1d %5d %2d %2d %20.12f %20.12f\n",
            spin, iep, i, j, exx_DM[spin][iep][i][j].r,
            exx_DM[spin][iep][i][j].i);
        }
      }
    }  /* loop of iep */
  } /* loop of spin */
  printf("\n");
#endif

  fclose(fp);
}


void EXX_Input_DM(EXX_t *exx, dcomplex ****exx_DM)
{
  int i, j, iep, spin, iatm1, iatm2, icell;  
  int nep, nspin, natom, nb1, nb2;
  int *ep_atom1, *ep_atom2, *ep_cell, *atom_nb;
  FILE *fp;
  char path[EXX_PATHLEN];
  size_t sz;

  snprintf(path, EXX_PATHLEN, "%s/exxdm.dat", EXX_CacheDir(exx));
  
  fp = fopen(path, "rb");
 
  /* spin */
  sz = fread(&nspin, sizeof(int), 1, fp); 
  if (nspin != SpinP_switch+1) { 
    fprintf(stderr, "  nspin= %d  SpinP_switch= %d\n", nspin, SpinP_switch);
    EXX_ERROR("file is wrong");
  }
  
  /* atom */
  sz = fread(&natom, sizeof(int), 1, fp); 
  if (natom != atomnum) {
    fprintf(stderr, "  natom= %d  atomnum= %d\n", natom, atomnum);
    EXX_ERROR("file is wrong");
  }
    
  atom_nb = (int*)malloc(sizeof(int)*natom);

  /* number of basis */
  sz = fread(&atom_nb[0], sizeof(int), natom, fp);
  for (i=0; i<natom; i++) {
    if (atom_nb[i] != EXX_atom_nb(exx)[i]) {
      fprintf(stderr, "  i= %d  atom_nb= %d %d\n", 
        i, atom_nb[i], EXX_atom_nb(exx)[i]);
      EXX_ERROR("file is wrong");
    }
  }
  
  /* exchange pairs */  
  sz = fread(&nep, sizeof(int), 1, fp); 
  if (nep != EXX_Number_of_EP(exx)) {
    fprintf(stderr, "  nep= %d %d\n", nep, EXX_Number_of_EP(exx));
    EXX_ERROR("file is wrong");
  }

  ep_atom1 = (int*)malloc(sizeof(int)*nep);
  ep_atom2 = (int*)malloc(sizeof(int)*nep);
  ep_cell  = (int*)malloc(sizeof(int)*nep);
      
  sz = fread(&ep_atom1[0], sizeof(int), nep, fp); 
  sz = fread(&ep_atom2[0], sizeof(int), nep, fp); 
  sz = fread(&ep_cell[0] , sizeof(int), nep, fp); 

  for (iep=0; iep<nep; iep++) {
    if (ep_atom1[iep] != EXX_Array_EP_Atom1(exx)[iep]) {
      fprintf(stderr, "  i= %d ep_atom1= %d %d\n",
        i, ep_atom1[iep], EXX_Array_EP_Atom1(exx)[iep]);
      EXX_ERROR("file is wrong");
    }
    if (ep_atom2[iep] != EXX_Array_EP_Atom2(exx)[iep]) {
      fprintf(stderr, "  i= %d ep_atom2= %d %d\n",
        i, ep_atom2[iep], EXX_Array_EP_Atom2(exx)[iep]);
      EXX_ERROR("file is wrong");
    }
    if (ep_cell[iep] != EXX_Array_EP_Cell(exx)[iep]) {
      fprintf(stderr, "  i= %d ep_cell= %d %d\n",
        i, ep_cell[iep], EXX_Array_EP_Cell(exx)[iep]);
      EXX_ERROR("file is wrong");
    }
  }
  
  for (spin=0; spin<nspin; spin++) {
    for (iep=0; iep<nep; iep++) {
      nb1 = atom_nb[ep_atom1[iep]];
      nb2 = atom_nb[ep_atom2[iep]];
      for (i=0; i<nb1; i++) {
        for (j=0; j<nb2; j++) {
          fread(&exx_DM[spin][iep][i][j].r, sizeof(double), 1, fp); 
          fread(&exx_DM[spin][iep][i][j].i, sizeof(double), 1, fp); 
        }
      }
    }  /* loop of iep */
  } /* loop of spin */

#if 0
  printf("EXX-DM:\n");
  for (spin=0; spin<nspin; spin++) {
    for (iep=0; iep<nep; iep++) {
      nb1 = atom_nb[ep_atom1[iep]];
      nb2 = atom_nb[ep_atom2[iep]];
      for (i=0; i<nb1; i++) {
        for (j=0; j<nb2; j++) {
          printf("  %1d %5d %2d %2d %20.12f %20.12f\n",
            spin, iep, i, j, exx_DM[spin][iep][i][j].r,
            exx_DM[spin][iep][i][j].i);
        }
      }
    }  /* loop of iep */
  } /* loop of spin */
  printf("\n");
#endif
    
  free(atom_nb);
  free(ep_atom1);
  free(ep_atom2);
  free(ep_cell);

  fclose(fp);
}


