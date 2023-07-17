 integer(kind=C_INT), parameter :: ELPA_OK = 0 
 integer(kind=C_INT), parameter :: ELPA_ERROR = -1 
 integer(kind=C_INT), parameter :: ELPA_ERROR_ENTRY_NOT_FOUND = -2 
 integer(kind=C_INT), parameter :: ELPA_ERROR_ENTRY_INVALID_VALUE = -3 
 integer(kind=C_INT), parameter :: ELPA_ERROR_ENTRY_ALREADY_SET = -4 
 integer(kind=C_INT), parameter :: ELPA_ERROR_ENTRY_NO_STRING_REPRESENTATION = -5 
 integer(kind=C_INT), parameter :: ELPA_ERROR_ENTRY_READONLY = -6 

 integer(kind=C_INT), parameter :: ELPA_SOLVER_1STAGE = 1 
 integer(kind=C_INT), parameter :: ELPA_SOLVER_2STAGE = 2 

 integer(kind=C_INT), parameter :: ELPA_NUMBER_OF_SOLVERS = (0 +1 +1) 

 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_GENERIC = 1 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_GENERIC_SIMPLE = 2 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_BGP = 3 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_BGQ = 4 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SSE_ASSEMBLY = 5 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SSE_BLOCK2 = 6 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SSE_BLOCK4 = 7 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SSE_BLOCK6 = 8 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX_BLOCK2 = 9 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX_BLOCK4 = 10 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX_BLOCK6 = 11 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX2_BLOCK2 = 12 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX2_BLOCK4 = 13 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX2_BLOCK6 = 14 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX512_BLOCK2 = 15 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX512_BLOCK4 = 16 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_AVX512_BLOCK6 = 17 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_GPU = 18 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SPARC64_BLOCK2 = 19 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SPARC64_BLOCK4 = 20 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_SPARC64_BLOCK6 = 21 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_VSX_BLOCK2 = 22 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_VSX_BLOCK4 = 23 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_VSX_BLOCK6 = 24 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_INVALID = -1 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_REAL_DEFAULT = 15 

 integer(kind=C_INT), parameter :: ELPA_2STAGE_NUMBER_OF_REAL_KERNELS = & 
 (0 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1) 

 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_GENERIC = 1 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE = 2 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_BGP = 3 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_BGQ = 4 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_SSE_ASSEMBLY = 5 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_SSE_BLOCK1 = 6 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_SSE_BLOCK2 = 7 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_AVX_BLOCK1 = 8 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_AVX_BLOCK2 = 9 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_AVX2_BLOCK1 = 10 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_AVX2_BLOCK2 = 11 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 = 12 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_AVX512_BLOCK2 = 13 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_GPU = 14 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_INVALID = -1 
 integer(kind=C_INT), parameter :: ELPA_2STAGE_COMPLEX_DEFAULT = 12 

 integer(kind=C_INT), parameter :: ELPA_2STAGE_NUMBER_OF_COMPLEX_KERNELS = & 
 (0 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1 +1) 

 integer(kind=C_INT), parameter :: ELPA_AUTOTUNE_NOT_TUNABLE = 0 
 integer(kind=C_INT), parameter :: ELPA_AUTOTUNE_FAST = 1 
 integer(kind=C_INT), parameter :: ELPA_AUTOTUNE_MEDIUM = 2 

 integer(kind=C_INT), parameter :: ELPA_AUTOTUNE_DOMAIN_REAL = 1 
 integer(kind=C_INT), parameter :: ELPA_AUTOTUNE_DOMAIN_COMPLEX = 2 
 integer(kind=C_INT), parameter :: ELPA_AUTOTUNE_DOMAIN_ANY = 3 

