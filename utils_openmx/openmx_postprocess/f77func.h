
#ifndef __f77_function_definition__
#define __f77_function_definition__


  /* f77 name is uppercase */
/* #define _F77_FUNC_UPPERCASE_ */ 

    /* f77 name has a underscore */
/* #define _F77_FUNC_UNDERSCORE1_ */

    /* f77 name has two undrescores */
/*#define _F77_FUNC_UNDERSCORE2_ */


/* _F77_FUNC_UNDERSCORE1_ and _F77_FUNC_UNDERSCORE2_ are mutually exclusive. */

#ifdef f77
#endif

#ifdef f77_
#define _F77_FUNC_UNDERSCORE1_
#endif

#ifdef f77__
#define _F77_FUNC_UNDERSCORE2_
#endif

#ifdef F77
#define _F77_FUNC_UPPERCASE_
#endif

#ifdef F77_
#define _F77_FUNC_UPPERCASE_
#define _F77_FUNC_UNDERSCORE1_
#endif

#ifdef F77__
#define _F77_FUNC_UPPERCASE_
#define _F77_FUNC_UNDERSCORE2_
#endif

#if !defined(f77) && !defined(f77_) && !defined(f77__) && !defined(F77) && !defined(F77_)  && !defined(F77__)
#define _F77_FUNC_UNDERSCORE1_
#endif

/*********************************************************************/


#if  defined ( _F77_FUNC_UPPERCASE_ )  && !defined(_F77_FUNC_UNDERSCORE1_)  && defined(_F77_FUNC_UNDERSCORE2_)
    #define F77_NAME(a,A) A ## __
#endif

#if  !defined ( _F77_FUNC_UPPERCASE_ )  && !defined(_F77_FUNC_UNDERSCORE1_)   && defined(_F77_FUNC_UNDERSCORE2_)
    #define F77_NAME(a,A) a ## __
#endif

#if  defined ( _F77_FUNC_UPPERCASE_ )  && defined(_F77_FUNC_UNDERSCORE1_)    && !defined(_F77_FUNC_UNDERSCORE2_)
    #define F77_NAME(a,A) A ## _
#endif

#if  !defined ( _F77_FUNC_UPPERCASE_ )  && defined(_F77_FUNC_UNDERSCORE1_) && !defined(_F77_FUNC_UNDERSCORE2_)
    #define F77_NAME(a,A) a ## _
#endif

#if  defined ( _F77_FUNC_UPPERCASE_ )  && !defined(_F77_FUNC_UNDERSCORE1_) && !defined(_F77_FUNC_UNDERSCORE2_)
    #define F77_NAME(a,A) A 
#endif

#if  !defined ( _F77_FUNC_UPPERCASE_ )  && !defined(_F77_FUNC_UNDERSCORE1_) && !defined(_F77_FUNC_UNDERSCORE2_)
    #define F77_NAME(a,A) a 
#endif



#endif
