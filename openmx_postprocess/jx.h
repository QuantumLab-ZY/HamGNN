/**********************************************************************

jx.c:

This program code calculates spin-spin interaction
coupling constant J between the selected two atoms

Log of jx.c:
   30/Aug/2003  Released by Myung Joon Han (supervised by Prof. J. Yu)
    7/Dec/2003  Modified by Taisuke Ozaki
   03/Mar/2011  Modified by Fumiyuki Ishii for MPI
   02/May/2018  Modified by Asako Terasawa for blas
***********************************************************************/

#pragma once

// functions for Jij of atoms in a cluster
void Jij_cluster(int argc, char *argv[], MPI_Comm comm1);
// functions for Jij of atoms in a single cell in band calculation
void Jij_band_psum( MPI_Comm comm1 );
void Jij_band_indiv( MPI_Comm comm1 );
