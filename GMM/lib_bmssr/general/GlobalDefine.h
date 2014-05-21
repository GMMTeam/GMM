#ifndef __GLOBAL_DEFINE
#define __GLOBAL_DEFINE

#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
//#include <stdexcpt.h>

#define RETURN_OK                           1                
#define RETURN_CANNOT_OPEN_FILE             2                
#define RETURN_SETTING_ILLEGAL_SECTION      3                
#define RETURN_SETTING_ILLEGAL_ITEM         4                
#define RETURN_SETTING_ILLEGAL_DATA         5                
#define RETURN_SETTING_DATA_INCOMPLETE      6                
#define RETURN_NOT_ENOUGH_MEMORY            7                
#define RETURN_FILE_NOT_OPEN                8

#define RETURN_FAIL                         -1                

#define RETURN_WRONG_MATRIX                 9
#define RETURN_EIG_IMAG                     10
#define RETURN_WRONG_MATRIX_SIZE            11

#define RETURN_NOT_ENOUGH_DATA			    12

#define SETTING_MAX_LINE_LENGTH      500
#define _MAX_EXT_LENGTH_			 50
#define _MAX_                        1000

#ifndef PI
#define PI   3.14159265358979
#endif

#define dvaPI  6.28318530717959     /* PI*2 */
#define MAX_PATH                      500


#define LZERO  (-1.0E10)   // ~log(0) 
#define LSMALL (-0.5E10)   // log values < LSMALL are set to LZERO
#define MINEARG (-708.3)   // lowest exp() arg  = log(MINLARG)
#define MINLARG 2.45E-308  // lowest log() arg  = exp(MINEARG)


inline double Exp(double x){ return (x>MINEARG) ? exp(x) : 0.0;};
inline double Log(double x){ return (x<MINLARG) ? LZERO : log(x);};


#define _NO_THROW               throw ()
#define _THROW_MATRIX_ERROR     throw (matrix_error)
#define REPORT_ERROR(ErrormMsg)  throw matrix_error( ErrormMsg);


#define STD

#ifndef _NO_EXCEPTION
#  define TRYBEGIN()    try {
#  define CATCHERROR()  } catch (const STD::exception& e) {printf("Error: %s\n",e.what());}
#else
#  define TRYBEGIN()
#  define CATCHERROR()
#endif

#endif

