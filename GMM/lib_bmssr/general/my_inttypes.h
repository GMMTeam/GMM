#ifndef _MY_INTTYPES_H
#define _MY_INTTYPES_H

#ifdef __GNUC__ //LINUX

#include <inttypes.h>

#define __int8 int8_t
#define __int16 int16_t
#define __int32 int32_t
#define __int64 int64_t

#define __uint8 uint8_t
#define __uint16 uint16_t
#define __uint32 uint32_t
#define __uint64 uint64_t

#else //WIN32 or WIN64

#define __uint8 unsigned __int8
#define __uint16 unsigned __int16
#define __uint32 unsigned __int32
#define __uint64 unsigned __int64

#define int8_t __int8
#define int16_t __int16
#define int32_t __int32
#define int64_t __int64

#define uint8_t __uint8
#define uint16_t __uint16
#define uint32_t __uint32
#define uint64_t __uint64

#endif

#endif //endfile
