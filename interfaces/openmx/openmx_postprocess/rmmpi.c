#include <stdio.h>
#include <stdlib.h>
#include <time.h>

char *substr( char *buffer, const char *string,
              int nStart, int nStop );



int main()
{
  static int pid;
  static char st[300],buffer[800];
  static char operate[800];
  FILE *fp;

  sprintf(operate,"ps aux > rmmpi_tmp");
  system(operate);

  if ((fp=fopen("rmmpi_tmp","r"))!=NULL){

    while (fgets(st, 300, fp) != NULL){
      if (strcmp(substr(buffer, st, 0, 5), "ozaki")==0){
        substr(buffer, st, 6, 16);
        pid = atoi(buffer);  
        printf("%s %2d\n",substr(buffer, st, 0, 5),pid);

        sprintf(operate,"kill -9 %2d",pid);
        system(operate);
      } 
    }

    fclose(fp);
  }
  else {
    printf("could not open.\n");
  }

  sprintf(operate,"rm rmmpi_tmp");
  system(operate);

}

char *substr( char *buffer, const char *string, int nStart, int nStop )
{
  memmove( buffer, string + nStart, nStop - nStart );
  buffer[nStop-nStart] = '\0';

  return buffer;
}
