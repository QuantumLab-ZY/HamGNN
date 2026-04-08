/**********************************************************************
  Make_InputFile_with_FinalCoord.c:

     Make_InputFile_with_FinalCoord.c is a subrutine to make an input file
     with the final coordinate of the system.

  Log of Make_InputFile_with_FinalCoord.c:

     19/Sep./2007  Released by T. Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <string.h>

/*  stat section */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
/*  end stat section */
#include "openmx_common.h"
#include "tran_variables.h"
#include "mpi.h"


void Make_InputFile_with_FinalCoord_Normal(char *file, int MD_iter);
void Make_InputFile_with_FinalCoord_NEGF(char *file, int MD_iter);



void Make_InputFile_with_FinalCoord(char *file, int MD_iter)
{

  if (Solver!=4)
    Make_InputFile_with_FinalCoord_Normal(file, MD_iter);
  else 
    Make_InputFile_with_FinalCoord_NEGF(file, MD_iter);

}




void Make_InputFile_with_FinalCoord_Normal(char *file, int MD_iter)
{
  int i,j,Gc_AN,c,n1,k;
  int restart_flag,fixed_flag;
  int geoopt_restart_flag;
  int velocity_flag;
  int MD_Current_Iter_flag;
  int Atoms_UnitVectors_flag;
  int Atoms_UnitVectors_Velocity_flag;
  int Atoms_UnitVectors_Unit_flag;
  int NPT_WV_F0_flag;
  int rstfile_num;
  double c1,c2,c3;
  double vx,vy,vz;
  double tmpxyz[4];
  char st[800];
  char st1[800];
  char rm_operate[YOUSO10];
  char fname[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  FILE *fp1,*fp2;
  char *tp;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){

    /* initialize */

    restart_flag = 0;
    geoopt_restart_flag = 0;
    fixed_flag = 0;
    velocity_flag = 0; 
    MD_Current_Iter_flag = 0;
    Atoms_UnitVectors_flag = 0;
    Atoms_UnitVectors_Unit_flag = 0;
    Atoms_UnitVectors_Velocity_flag = 0;
    NPT_WV_F0_flag = 0;
 
    /* the new input file */    

    sprintf(fname1,"%s#",file);
    fp1 = fopen(fname1,"w");
    fseek(fp1,0,SEEK_END);

    /* the original input file */    

    fp2 = fopen(file,"r");

    if (fp2!=NULL){

      while (fgets(st,800,fp2)!=NULL){

        string_tolower(st,st1); 

        /* find the specification of <atoms.speciesandcoordinates */

        if (strncmp(st1,"<atoms.speciesandcoordinates",28)==0){

          fprintf(fp1,"%s",st);

          /* replace the atomic coordinates */

          for (i=1; i<=atomnum; i++){

            fgets(st,800,fp2);
            string_tolower(st,st1);

            /* serial number */
	    tp = strtok(st, " ");
	    if (tp!=NULL) fprintf(fp1,"%4s",tp);

            /* name of species */
            tp =strtok(NULL, " ");  
	    if (tp!=NULL) fprintf(fp1," %4s",tp);

            /* "Ang" */ 
            if (coordinates_unit==0){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][1]*BohrR);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][2]*BohrR);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][3]*BohrR);
            }

            /* AU */
            else if (coordinates_unit==1){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][1]);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][2]);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][3]);
            }

            /* FRAC */
            else if (coordinates_unit==2){

              /* The zero is taken as the origin of the unit cell. */

              tmpxyz[1] = Gxyz[i][1] - Grid_Origin[1];
              tmpxyz[2] = Gxyz[i][2] - Grid_Origin[2];
              tmpxyz[3] = Gxyz[i][3] - Grid_Origin[3];

   	      c1 = Dot_Product(tmpxyz,rtv[1])*0.5/PI;
              c2 = Dot_Product(tmpxyz,rtv[2])*0.5/PI;
              c3 = Dot_Product(tmpxyz,rtv[3])*0.5/PI;

	      /* a-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c1);

	      /* b-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c2);

	      /* c-axis */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",c3);
            }

	    while (tp!=NULL){
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1,"     %s",tp);
	    }

          } 

        }

        /* scf.restart */

        else if (strncmp(st1,"scf.restart",11)==0){
          fprintf(fp1,"scf.restart    on\n");
          restart_flag = 1;
	}

        /* geoopt.restart */

        else if (strncmp(st1,"geoopt.restart",11)==0){
          fprintf(fp1,"geoopt.restart    on\n");
          geoopt_restart_flag = 1;
	}

        /* scf.fixed.grid */

        else if (strncmp(st1,"scf.fixed.grid",14)==0){
          fprintf(fp1,"scf.fixed.grid   %18.14f  %18.14f  %18.14f\n",
                  Grid_Origin[1],Grid_Origin[2],Grid_Origin[3]);
          fixed_flag = 1;
        }  

        /* MD.Init.Velocity && VerletXYZ (NEV) */

        else if (strncmp(st1,"<md.init.velocity",17)==0
		 && ( MD_switch==1   ||  // NVE
                      MD_switch==2   ||  // NVT_VS 
                      MD_switch==9   ||  // NVT_NH
                      MD_switch==11  ||  // NVT_VS2 
                      MD_switch==14  ||  // NVT_VS4
                      MD_switch==15  ||  // NVT_Langevin
                      MD_switch==27  ||  // NPT_VS_PR
                      MD_switch==28  ||  // NPT_VS_WV
                      MD_switch==29  ||  // NPT_NH_PR
                      MD_switch==30      // NPT_NH_WV
	         )){

          for (i=1; i<=(atomnum+1); i++){
            fgets(st,800,fp2);
	  }

          /* velocity at this moment */ 

          fprintf(fp1,"<MD.Init.Velocity\n");

          for (i=1; i<=atomnum; i++){

            vx = Gxyz[i][24]/(0.4571028*0.000001);
            vy = Gxyz[i][25]/(0.4571028*0.000001);
            vz = Gxyz[i][26]/(0.4571028*0.000001);
            fprintf(fp1," %4d    %20.14f  %20.14f  %20.14f\n", i, vx, vy, vz);
	  }

          fprintf(fp1,"MD.Init.Velocity>\n");

          /* velocity one step before */ 

          if ( MD_switch==1  || // NVE
               MD_switch==9  || // NVT_NH
               MD_switch==15 || // NVT_Langevin
               MD_switch==29 || // NPT_NH_PR
               MD_switch==30    // NPT_NH_WV
	       ){

	    fprintf(fp1,"<MD.Init.Velocity.Prev\n");

	    for (i=1; i<=atomnum; i++){

	      vx = Gxyz[i][27]/(0.4571028*0.000001);
	      vy = Gxyz[i][28]/(0.4571028*0.000001);
	      vz = Gxyz[i][29]/(0.4571028*0.000001);
	      fprintf(fp1," %4d    %20.14f  %20.14f  %20.14f\n", i, vx, vy, vz);
	    }

	    fprintf(fp1,"MD.Init.Velocity.Prev>\n");
	  }

          /* parameters for the Nose-Hoover thermostat */

          if ( MD_switch==9 || MD_switch==29 || MD_switch==30 ){
            fprintf(fp1,"NH.R      %20.15f\n",NH_R);
            fprintf(fp1,"NH.nzeta  %20.15f\n",NH_nzeta);
            fprintf(fp1,"NH.czeta  %20.15f\n",NH_czeta);
	  }

          /* velocity_flag */

          velocity_flag = 1;
        }  

        /* MD.Current.Iter */

        else if (strncmp(st1,"md.current.iter",15)==0){

          fprintf(fp1,"\n\nMD.Current.Iter  %2d\n",MD_iter);

          MD_Current_Iter_flag = 1;
	}

        /* Atoms.UnitVectors */

        else if (strncmp(st1,"<atoms.unitvectors",18)==0){

          for (i=1; i<=4; i++){
            fgets(st,800,fp2);
	  }

          fprintf(fp1,"<Atoms.UnitVectors\n");

	  for (i=1; i<=3; i++){
	    for (j=1; j<=3; j++){

	      if (unitvector_unit==0){ /* Ang */
		fprintf(fp1," %18.15f ",tv[i][j]*BohrR);
	      }
	      else{                    /* AU */  
		fprintf(fp1," %18.15f ",tv[i][j]);
	      }
	    }
	    fprintf(fp1,"\n");
	  }

          fprintf(fp1,"Atoms.UnitVectors>\n");

          Atoms_UnitVectors_flag = 1;
	}

        /* Atoms.UnitVectors.Unit */

        else if (strncmp(st1,"atoms.unitvectors.unit",22)==0){

          if      (unitvector_unit==0)  fprintf(fp1,"Atoms.UnitVectors.Unit  Ang\n");
          else if (unitvector_unit==1)  fprintf(fp1,"Atoms.UnitVectors.Unit  AU\n");

          Atoms_UnitVectors_Unit_flag = 1;
	}

        else if (strncmp(st1,"<atoms.unitvectors.velocity",27)==0
                 && ( MD_switch==27  ||  // NPT_VS_PR
                      MD_switch==28  ||  // NPT_VS_WV
                      MD_switch==29  ||  // NPT_NH_PR
                      MD_switch==30      // NPT_NH_WV
	      )){

          /* velocity of lattice vectors at this moment */ 

          fprintf(fp1,"<Atoms.Unitvectors.Velocity\n");

	  for (i=1; i<=3; i++){
	    for (j=1; j<=3; j++){
              fprintf(fp1," %25.22f ",tv_velocity[i][j]); 
   	    } 
	    fprintf(fp1,"\n");
	  } 

          fprintf(fp1,"Atoms.Unitvectors.Velocity>\n");

          Atoms_UnitVectors_Velocity_flag = 1;

	} /* <atoms.unitvectors.velocity */

        else if (strncmp(st1,"<NPT.WV.F0",10)==0){

          fprintf(fp1,"<NPT.WV.F0\n");

	  for (i=1; i<=3; i++){
	    for (j=1; j<=3; j++){
              fprintf(fp1," %25.22f ",NPT_WV_F0[i][j]); 
   	    } 
	    fprintf(fp1,"\n");
	  } 

          fprintf(fp1,"NPT.WV.F0>\n");

          NPT_WV_F0_flag = 1;
	}

        else{
          fprintf(fp1,"%s",st);
	}
      }

      fclose(fp2); 
    }

    /* add the restart flag if it was not found. */

    if (restart_flag==0){
      fprintf(fp1,"\n\nscf.restart    on\n");
    }

    /* add the restart flag for geometry optimization if it was not found. */

    if (geoopt_restart_flag==0 &&
       (MD_switch==3 || MD_switch==4 || MD_switch==5 || MD_switch==6 || MD_switch==7) ){
      fprintf(fp1,"\n\ngeoopt.restart    on\n");
    }

    /* add scf.fixed.grid if it was not found. */

    if (fixed_flag==0){
      fprintf(fp1,"\n\nscf.fixed.grid   %18.14f  %18.14f  %18.14f\n",
                   Grid_Origin[1],Grid_Origin[2],Grid_Origin[3]);
    }

    /* add velocity frag if it was not found. */

    if ( velocity_flag==0 && 
         ( MD_switch==1  || // NVE
           MD_switch==2  || // NVT_VS
           MD_switch==9  || // NVT_NH
           MD_switch==11 || // NVT_VS2
           MD_switch==14 || // NVT_VS4
           MD_switch==15 || // NVT_Langevin
           MD_switch==27 || // NPT_VS_PR
           MD_switch==28 || // NPT_VS_WV
           MD_switch==29 || // NPT_NH_PR
           MD_switch==30    // NPT_NH_WV
      )){

      /* velocity at this moment */ 

      fprintf(fp1,"\n\n<MD.Init.Velocity\n");

      for (i=1; i<=atomnum; i++){
	fprintf(fp1," %4d    %20.14f  %20.14f  %20.14f\n",
		i,
		Gxyz[i][24]/(0.4571028*0.000001),
		Gxyz[i][25]/(0.4571028*0.000001),
		Gxyz[i][26]/(0.4571028*0.000001));
      }

      fprintf(fp1,"MD.Init.Velocity>\n");

      /* velocity one step before */ 

      fprintf(fp1,"\n\n<MD.Init.Velocity.Prev\n");

      for (i=1; i<=atomnum; i++){
	fprintf(fp1," %4d    %20.14f  %20.14f  %20.14f\n",
		i,
		Gxyz[i][27]/(0.4571028*0.000001),
		Gxyz[i][28]/(0.4571028*0.000001),
		Gxyz[i][29]/(0.4571028*0.000001));
      }

      fprintf(fp1,"MD.Init.Velocity.Prev>\n");

      /* parameters for the Nose-Hoover thermostat */

      if ( MD_switch==9 || MD_switch==29 || MD_switch==30 ){

	fprintf(fp1,"\n");
	fprintf(fp1,"NH.R      %20.15f\n",NH_R);
	fprintf(fp1,"NH.nzeta  %20.15f\n",NH_nzeta);
	fprintf(fp1,"NH.czeta  %20.15f\n",NH_czeta);
      }
    }

    /* add MD.Current.Iter if it was not found. */

    if (MD_Current_Iter_flag==0){

      fprintf(fp1,"\n\nMD.Current.Iter  %2d\n",MD_iter);
    }

    /* add Atoms.UnitVectors if it was not found. */

    if (Atoms_UnitVectors_flag==0){

      fprintf(fp1,"\n\n");
      fprintf(fp1,"<Atoms.UnitVectors\n");

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){

          if (unitvector_unit==0){ /* Ang */
  	    fprintf(fp1," %18.15f ",tv[i][j]*BohrR);
	  }
          else{                    /* AU */  
  	    fprintf(fp1," %18.15f ",tv[i][j]);
	  }

	}
	fprintf(fp1,"\n");
      }

      fprintf(fp1,"Atoms.UnitVectors>\n");
    }

    /* add Atoms.UnitVectors.Unit if it was not found. */

    if (Atoms_UnitVectors_Unit_flag==0){

      fprintf(fp1,"\n\n");

      if (unitvector_unit==0){ /* Ang */
        fprintf(fp1,"Atoms.UnitVectors.Unit  Ang\n");
      }
      else{
        fprintf(fp1,"Atoms.UnitVectors.Unit  AU\n");
      }
    }

    /* add Atoms.UnitVectors.Velocity if it was not found. */

    if (Atoms_UnitVectors_Velocity_flag==0){

      fprintf(fp1,"\n\n");

      /* velocity of lattice vectors at this moment */ 

      fprintf(fp1,"<Atoms.Unitvectors.Velocity\n");

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
	  fprintf(fp1," %25.22f ",tv_velocity[i][j]); 
	} 
	fprintf(fp1,"\n");
      } 

      fprintf(fp1,"Atoms.Unitvectors.Velocity>\n");

    }

    /* add NPT.WV.F0 if it was not found. */

    if (NPT_WV_F0_flag==0){

      fprintf(fp1,"\n\n");
      fprintf(fp1,"<NPT.WV.F0\n");

      for (i=1; i<=3; i++){
	for (j=1; j<=3; j++){
	  fprintf(fp1," %25.22f ",NPT_WV_F0[i][j]); 
	} 
	fprintf(fp1,"\n");
      } 

      fprintf(fp1,"NPT.WV.F0>\n");
    }

    /* fclose */
    fclose(fp1); 

  } /* if (myid==Host_ID) */

}



void Make_InputFile_with_FinalCoord_NEGF(char *file, int MD_iter)
{
  int i,Gc_AN,c,n1,k;
  int restart_flag,fixed_flag;
  int velocity_flag;
  int rstfile_num;
  double c1,c2,c3;
  char st[800];
  char st1[800];
  char rm_operate[YOUSO10];
  char fname[YOUSO10];
  char fname1[YOUSO10];
  char fname2[YOUSO10];
  FILE *fp1,*fp2;
  char *tp;
  int numprocs,myid;

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){

    /* initialize */

    restart_flag = 0;
    fixed_flag = 0;
    velocity_flag = 0; 

    /* the new input file */    

    sprintf(fname1,"%s#",file);
    fp1 = fopen(fname1,"w");
    fseek(fp1,0,SEEK_END);

    /* the original input file */    

    fp2 = fopen(file,"r");

    if (fp2!=NULL){

      while (fgets(st,800,fp2)!=NULL){

        string_tolower(st,st1); 

        /* find the specification of <leftleadatoms.speciesandcoordinates */

        if (strncmp(st1,"<leftleadatoms.speciesandcoordinates",35)==0){

          fprintf(fp1,"%s",st);

          /* replace the atomic coordinates */

          for (i=1; i<=Latomnum; i++){

            fgets(st,800,fp2);
            string_tolower(st,st1);

            /* serial number */
	    tp = strtok(st, " ");
	    if (tp!=NULL) fprintf(fp1,"%4s",tp);

            /* name of species */
            tp =strtok(NULL, " ");  
	    if (tp!=NULL) fprintf(fp1," %4s",tp);

            /* "Ang" */ 
            if (coordinates_unit==0){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][1]*BohrR);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][2]*BohrR);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][3]*BohrR);
            }

            /* AU */
            else if (coordinates_unit==1){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][1]);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][2]);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i][3]);
            }

	    while (tp!=NULL){
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1,"     %s",tp);
	    }

          } 
	}

        /* find the specification of <atoms.speciesandcoordinates */

        else if (strncmp(st1,"<atoms.speciesandcoordinates",28)==0){

          fprintf(fp1,"%s",st);

          /* replace the atomic coordinates */

          for (i=1; i<=Catomnum; i++){

            fgets(st,800,fp2);
            string_tolower(st,st1);

            /* serial number */
	    tp = strtok(st, " ");
	    if (tp!=NULL) fprintf(fp1,"%4s",tp);

            /* name of species */
            tp =strtok(NULL, " ");  
	    if (tp!=NULL) fprintf(fp1," %4s",tp);

            /* "Ang" */ 
            if (coordinates_unit==0){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum][1]*BohrR);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum][2]*BohrR);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum][3]*BohrR);
            }

            /* AU */
            else if (coordinates_unit==1){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum][1]);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum][2]);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum][3]);
            }

	    while (tp!=NULL){
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1,"     %s",tp);
	    }
          }
        }

        /* find the specification of <rightleadatoms.speciesandcoordinates */

        else if (strncmp(st1,"<rightleadatoms.speciesandcoordinates",36)==0){

          fprintf(fp1,"%s",st);

          /* replace the atomic coordinates */

          for (i=1; i<=Ratomnum; i++){

            fgets(st,800,fp2);
            string_tolower(st,st1);

            /* serial number */
	    tp = strtok(st, " ");
	    if (tp!=NULL) fprintf(fp1,"%4s",tp);

            /* name of species */
            tp =strtok(NULL, " ");  
	    if (tp!=NULL) fprintf(fp1," %4s",tp);

            /* "Ang" */ 
            if (coordinates_unit==0){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum+Catomnum][1]*BohrR);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum+Catomnum][2]*BohrR);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum+Catomnum][3]*BohrR);
            }

            /* AU */
            else if (coordinates_unit==1){

	      /* x-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum+Catomnum][1]);

	      /* y-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum+Catomnum][2]);

	      /* z-coordinate */
	      tp =strtok(NULL, " ");  
	      fprintf(fp1,"  %18.14f",Gxyz[i+Latomnum+Catomnum][3]);
            }

	    while (tp!=NULL){
	      tp =strtok(NULL, " ");  
	      if (tp!=NULL) fprintf(fp1,"     %s",tp);
	    }

          } 

        }

        /* scf.restart */

        else if (strncmp(st1,"scf.restart",11)==0){
          fprintf(fp1,"scf.restart    on\n");
          restart_flag = 1;
	}

        /* scf.fixed.grid */

        else if (strncmp(st1,"scf.fixed.grid",14)==0){

          fprintf(fp1,"scf.fixed.grid   %18.14f  %18.14f  %18.14f\n",
                       Grid_Origin[1],Grid_Origin[2],Grid_Origin[3]);
          fixed_flag = 1;
        }  

        /* MD.Init.Velocity && VerletXYZ (NEV) */

        else if (strncmp(st1,"<md.init.velocity",16)==0
		 && (MD_switch==1 || MD_switch==2 || MD_switch==9 || MD_switch==11
		     || MD_switch==14 || MD_switch==15 || MD_switch==29 || MD_switch==30 )){

          for (i=1; i<=(atomnum+1); i++){
            fgets(st,800,fp2);
	  }

          fprintf(fp1,"<MD.Init.Velocity\n");

          for (i=1; i<=atomnum; i++){
            fprintf(fp1," %4d    %20.14f  %20.14f  %20.14f\n",
                    i,
                    Gxyz[i][24]/(0.4571028*0.000001),
                    Gxyz[i][25]/(0.4571028*0.000001),
                    Gxyz[i][26]/(0.4571028*0.000001));
	  }

          fprintf(fp1,"MD.Init.Velocity>\n");

          velocity_flag = 1;
        }  

        else{
          fprintf(fp1,"%s",st);
	}
      }

      fclose(fp2); 
    }

    /* add the restart flag if it was not found. */

    if (restart_flag==0){
      fprintf(fp1,"\n\nscf.restart    on\n");
    }

    /* add scf.fixed.grid if it was not found. */

    if (fixed_flag==0){
      fprintf(fp1,"\n\nscf.fixed.grid   %18.14f  %18.14f  %18.14f\n",
                   Grid_Origin[1],Grid_Origin[2],Grid_Origin[3]);
    }

    /* add velocity frag if it was not found. */

    if (velocity_flag==0 && (MD_switch==1 || MD_switch==2 || MD_switch==9 || MD_switch==11 || MD_switch==29 || MD_switch==30)){

      fprintf(fp1,"\n\n<MD.Init.Velocity\n");

      for (i=1; i<=atomnum; i++){
	fprintf(fp1," %4d    %20.14f  %20.14f  %20.14f\n",
		i,
		Gxyz[i][24]/(0.4571028*0.000001),
		Gxyz[i][25]/(0.4571028*0.000001),
		Gxyz[i][26]/(0.4571028*0.000001));
      }

      fprintf(fp1,"MD.Init.Velocity>\n");
    }

    /* fclose */
    fclose(fp1); 

  } /* if (myid==Host_ID) */

}










