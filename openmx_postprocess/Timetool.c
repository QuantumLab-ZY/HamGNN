#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/times.h>
#include <string.h>
#include <math.h>

typedef struct _Tlist {
   char *name;
   float stime;
   float etime;
   int   count;
   int   flag;
   struct _Tlist *next;
   struct _Tlist *child; 
} Tlist;  


static Tlist  *listtop=NULL; 
static Tlist  *listlast=NULL;

#define MAXSTACKLIST 50
static Tlist *stacklist[MAXSTACKLIST];
static int stacklast=0;

static int Timetool_printlevel=0;


static void fillspc(char *buf,int  n)
{
 int i,len;

 len=strlen(buf);
 if (len>n) {
    buf[n]='\0';
    return;
 }
 for (i=len;i<n ;i++) {
    buf[i]=' ';
 }
 buf[n]='\0';

}

static double Timetool_printlist(int level, Tlist *top,double grandtotal) 
{
 int i;
 Tlist *now;
 char *tab="  ";
 char *key="<TIME>";
 int nbuf=60;
 char buf[200];
 double total,subtotal,t0;

 total=0.0;
 for ( now=top; now; now=now->next ) {
     total += now->etime;
 }

 if (level==0) {
    grandtotal=total;
    printf(
"------------------------  Summay of elapsed times ----------------------\n"
);
    printf(
  "%s %-30s %-8s %-8s %10s  %6s  %6s\n",
key,"name", "", "#_called","sec","(%)", "%_per_call");
 }

 for ( now=top; now; now=now->next ) {

   subtotal=0.0;
   if (now->child) {
      subtotal = Timetool_printlist(level+1,now->child,grandtotal);
      printf("%s",key);
      *buf='\0';
      for (i=0;i<level;i++) { strcat(buf,tab); }
      strcat(buf,now->name);
      fillspc(buf,30);
      t0=(now->etime-subtotal); t0=(t0>0)? t0:0.0;
      printf(" %-30s %8s %8d %10.2f  %6.2f  %6.2f\n", 
            buf,"ex_child", now->count,t0,
              t0*100.0/grandtotal, t0*100.0/grandtotal/now->count);

   }
   else  {
      printf("%s",key);
      *buf='\0';
      for (i=0;i<level;i++) { strcat(buf,tab); }
      strcat(buf,now->name);
      fillspc(buf,30);
      t0=(now->etime-subtotal); t0=(t0>0)? t0:0.0;
      printf(" %-30s %8s %8d %10.2f  %6.2f  %6.2f\n", 
            buf,"________", now->count,t0,
               t0*100.0/grandtotal, t0*100.0/grandtotal/now->count);

   }


 }

 if (level==0) {
      printf("%s %-30s %-8s %-8s %10.2f  %-6.2f\n",
             key,"total","","",total, 100.0);
    printf(
"------------------------------------------------------------------------\n");

 }

 return total;

}

static Tlist *Timetool_findname(Tlist *top, char *name)
{

 Tlist *now, *tmp;
 
 for ( now=top; now; now=now->next ) {
   if (strcasecmp(now->name,name)==0) {
     return now;
   }
   if ( now->child ) {
      if ( tmp=Timetool_findname( now->child, name) ) {
         return tmp;
       }
   }
 }
 return NULL;
}

static void Timetool_initlist(Tlist *now)
{
   now->name=NULL;
   now->stime=0.0;
   now->etime=0.0;
   now->count=0;
   now->flag=0;
   now->next=NULL;
   now->child=NULL;

}

void Timetool_times(char *name, char *flag)
{
   struct tms ST;

   if ( strcasecmp(flag , "init")==0 ) {
      int i;
      for (i=0;i<MAXSTACKLIST;i++) {
        stacklist[i]=NULL;
      }
      stacklast=0;
      return;
   }


   if ( strcasecmp(flag , "print")==0 || strcasecmp(flag , "show")==0 ||
        strcasecmp(flag , "output")==0 ) {
      Timetool_printlist(0,listtop,0.0);
      return;
   }

   if ( strcasecmp(flag , "enter")==0 || strcasecmp(flag , "start")==0 ) {
      Tlist *now, *tmp;
      times(&ST);

      stacklast++;

      if ( now =Timetool_findname(listtop,name) ) { /* already there is */
         /* set starting time and flag */
         if (Timetool_printlevel) 
         printf("Timetool: name reenter %s\n",name);
      }
      else { /* new list */
 
         now= stacklist[stacklast-1];

         if (now) {
            if (now->child==NULL) { /* make child */
               if (Timetool_printlevel) 
               printf("%d name child %s\n", stacklast,name);
            now=now->child=(Tlist*)malloc(sizeof(Tlist));
            Timetool_initlist(now);
            now->name = strdup(name);
            stacklist[stacklast]=now;
            }
            else { /* make next; */
              if (Timetool_printlevel)
              printf("Timetool: %d name next %s\n", stacklast,name);
              now = stacklist[stacklast];
              now=now->next=(Tlist*)malloc(sizeof(Tlist));
            Timetool_initlist(now);
            now->name = strdup(name);
            stacklist[stacklast]=now;
            }
         }
         else {
            if (Timetool_printlevel)
            printf("Timetool: name next %s\n", name);
            now = stacklist[stacklast];
            if (now) {
              now=now->next=(Tlist*)malloc(sizeof(Tlist));
            }
            else {
                listtop =now= (Tlist*)malloc(sizeof(Tlist));
            }
            Timetool_initlist(now);
            now->name = strdup(name);
            stacklist[stacklast]=now;
         }

      }

      now->stime=(float)(ST.tms_utime+ST.tms_stime)/ (float)CLOCKS_PER_SEC;
      now->count++;
      now->flag=1;
      return; 
   }
   if ( strcasecmp(flag , "exit")==0 || strcasecmp(flag , "end")==0 ) {
      Tlist *now, *tmp;
      times(&ST);

      if (Timetool_printlevel)
      printf("Timetool: %d name exit %s\n",stacklast,name);
   
      now = Timetool_findname(listtop,name);
      /* omit to check the name */
      /* name must be in the stack */
      now->flag=0;
      now->etime += (float)(ST.tms_utime+ST.tms_stime)/(float)CLOCKS_PER_SEC
                      - now->stime;

      stacklist[stacklast]=now;
      stacklast--;

     return;
   }

   {
     printf("Timetool: flag=%s undefined for name=%s\n", flag,name);
     exit(100);

   }
   

}


#if 0

void foo(int n){

 int i;
  double p;

  p=10.;
  for (i=0;i<n*n*n;i++) {
    p= p*cos( (double)i/(double)n );
  }

}    
  

main()
{
   Timetool_times(NULL,"init");

   Timetool_times("1","enter");
   foo(50);

   Timetool_times("2","enter");
   Timetool_times("2","exit");


   Timetool_times("3","enter");

   Timetool_times("4","enter");
   foo(100);
   Timetool_times("4","exit");

   Timetool_times("5","enter");
   Timetool_times("5","exit");

   foo(90);

   Timetool_times("4","enter");
   foo(100);
   Timetool_times("4","exit");

   Timetool_times("5","enter");
   Timetool_times("5","exit");

   foo(90);


   Timetool_times("3","exit");

   Timetool_times("1","exit");



   Timetool_times("6","enter");
   Timetool_times("6","exit");




   Timetool_times(NULL,"print");

}


#endif



