int input_open(const  char *fname);
int input_close();
int input_cmpstring( char *str,  int *ret, int nvals,  char **strval,  int *ivals);
int input_logical(const char *key, int *ret,const  int defval);
int input_int(const char *key, int *ret, const int defval);
int input_double(const char *key, double *ret, const double defval);
int input_string(const char *key, char *ret,const char *defval);
int input_stringv(const char *key,const int nret, char **ret, char  **defval);
int input_doublev(const char *key,const int nret, double *ret, double *defval);
int input_intv(const char *key,const int nret, int *ret, int *defval)
;
FILE *input_find(const char *key);
int input_last(const char *key);
int input_string2int(const char *key, int *ret,int nvals,  char **strval,  int *ivals);
int input_errorCount();
