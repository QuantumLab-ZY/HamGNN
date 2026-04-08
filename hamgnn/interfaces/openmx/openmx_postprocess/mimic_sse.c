#ifdef nosse
#include "mimic_sse.h"

 __m128d _mm_setzero_pd(void);
 __m128d _mm_add_pd(__m128d a, __m128d b);
 __m128d _mm_mul_pd(__m128d a, __m128d b);
 __m128d _mm_load_pd(double const*dp);
 __m128d _mm_loadu_pd(double const*dp);
 void _mm_store_pd(double *dp, __m128d a);
 void _mm_storeu_pd(double *dp, __m128d a);

#endif
