/*----------------------------------------------------------------------
  exx_log.h

  simple logging tool

  Coded by M. Toyoda, 16 Dec. 2009 
----------------------------------------------------------------------*/
#ifndef EXX_LOG_H_INCLUDED
#define EXX_LOG_H_INCLUDED

#define EXX_LOG_SWITCH 1

/* works only for root process */
void EXX_Log_StdOut(const char *fmt, ... );
void EXX_Log_StdErr(const char *fmt, ... );


#define EXX_ERROR( msg ) EXX_Error( msg, __FILE__, __LINE__ )
#define EXX_WARN( msg )  EXX_Warn( msg, __FILE__, __LINE__ )

void EXX_Error(const char *msg, const char *file, int line);
void EXX_Warn(const char *msg, const char *file, int line);


#if EXX_LOG_SWITCH

void EXX_Log_Open(void);
void EXX_Log_Close(void);

void EXX_Log_Print(const char *fmt, ... );

void EXX_Log_Message(const char *message);
void EXX_Log_Trace_Integer(const char *key, int val);
void EXX_Log_Trace_Double(const char *key, double val);
void EXX_Log_Trace_String(const char *key, const char *val);
void EXX_Log_Trace_Vector(const char *ket, const double *val);

#define EXX_LOG_OPEN               EXX_Log_Open()
#define EXX_LOG_CLOSE              EXX_Log_Close()
#define EXX_LOG_MESSAGE( message ) EXX_Log_Message( message )
#define EXX_LOG_TRACE_INTEGER( key , val ) EXX_Log_Trace_Integer( key, val )
#define EXX_LOG_TRACE_DOUBLE( key , val )  EXX_Log_Trace_Double( key, val )
#define EXX_LOG_TRACE_STRING( key , val )  EXX_Log_Trace_String( key, val )
#define EXX_LOG_TRACE_VECTOR( key , val )  EXX_Log_Trace_Vector( key, val )

#if __STDC_VERSION__ >= 199901L
#define EXX_LOG_PRINT( fmt , ... ) EXX_Log_Print( fmt, __VA_ARGS__ )
#endif

#else

#define EXX_LOG_OPEN
#define EXX_LOG_OPEN
#define EXX_LOG_CLOSE
#define EXX_LOG_MESSAGE( message ) 
#define EXX_LOG_TRACE_INTEGER( key , val )
#define EXX_LOG_TRACE_DOUBLE( key , val ) 
#define EXX_LOG_TRACE_STRING( key , val )
#define EXX_LOG_TRACE_VECTOR( key , val )

#endif /* EXX_LOG_SWITCH */

#endif /* EXX_LOG_H_INCLUDED */
