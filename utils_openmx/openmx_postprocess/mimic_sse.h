
typedef struct {
    double             m128d_f64[2];
} __m128d;

extern __m128d _mm_setzero_pd(void);
extern __m128d _mm_add_pd(__m128d a, __m128d b);
extern __m128d _mm_mul_pd(__m128d a, __m128d b);
extern __m128d _mm_load_pd(double const*dp);
extern __m128d _mm_loadu_pd(double const*dp);
extern void _mm_store_pd(double *dp, __m128d a);
extern void _mm_storeu_pd(double *dp, __m128d a);
