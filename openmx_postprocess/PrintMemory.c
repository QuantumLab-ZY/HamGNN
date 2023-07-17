/**********************************************************************
  PrintMemory.c:

     PrintMemory.c is a subroutine to save memory size of each array.

  Log of PrintMemory.c:

     13/Mar/2003  Released by H. Kino
     14/Oct/2011  Rewritten by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_common.h"

static void PrintMemory_Output(char *name, long int size0, char *mode);

void PrintMemory(char *name, long int size0, char *mode)
{

  if (memoryusage_fileout){
    PrintMemory_Output(name, size0, mode);
  }
}


void PrintMemory_Output(char *name, long int size0, char *mode)
{
  FILE *fp;
  double size;
  static char *filename;
  static long double sumsize;

  if (mode!=NULL) {

    /* initialization */

    if (strcmp(mode,"init")==0) {

      sumsize = 0.0; 
      filename = name;
      fp = fopen(filename,"w");

      if (fp==NULL) {
	printf("PrintMemory: fail to open a file: %s\n",name);
      }
      else {
	fclose(fp);
      }

      return;
    }

    /* summation of memory */

    if (strcmp(mode,"sum")==0) {
      fp = fopen(filename,"a");
      fprintf(fp,"Memory: %-45s %10.2f MBytes\n","total",(double)sumsize);
      fclose(fp);
      return;
    }
  }

  /* add data to the file */

  size = (double)size0/(1024*1024);
  sumsize += size;
  fp = fopen(filename,"a");
  fprintf(fp,"Memory: %-45s %10.2f MBytes\n",name,size);
  fclose(fp);

}
