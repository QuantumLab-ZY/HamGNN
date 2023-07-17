void qsort_int1(long n, int *a);
void qsort_int(long n, int *a, int *b);
void qsort_int3(long n, int *a, int *b, int *c);
void qsort_double(long n, double *a, double *b);
void qsort_double_int(long n, double *a, int *b);
void qsort_double_int2(long n, double *a, int *b);
void qsort_double3(long n, double *a, int *b, int *c);
void qsort_double3B(long n, double *a, int *b, int *c);

typedef struct {
    double a,b;
} dlists;

typedef struct {
  double a;
  int b;
} dilists;

typedef struct {
  double a;
  int b;
  int c;
} dlist3;

typedef struct {
    int a,b;
} ilists;

typedef struct {
    int a,b,c;
} ilist3;

int dlists_cmp(const dlists *x, const dlists *y);
int dlists_cmp2(const dlists *x, const dlists *y);
int ilists_cmp(const ilists *x, const ilists *y);
