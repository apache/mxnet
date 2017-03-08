#pragma once
/* MP Programming shortcuts */
#ifdef VISUAL_STUDIO
#include <intrin.h>
#define AtomicAdd(ptr,val) InterlockedExchangeAdd64((volatile LONGLONG*)ptr,val)
#define AtomicAdd16(ptr,val) _InterlockedExchangeAdd16((volatile short*)(ptr),val)
#define AtomicAdd32(ptr,val) InterlockedExchangeAdd((volatile int*)(ptr),val)
#define AtomicInc32(ptr) InterlockedIncrement((volatile int*)ptr)
#define AtomicInc(ptr) InterlockedIncrement64((volatile LONGLONG*)ptr)
#define AtomicInc16(ptr) InterlockedIncrement16((volatile short*)ptr)
#define AtomicDec(ptr) InterlockedDecrement64((volatile LONGLONG*)ptr)
#define AtomicDec32(ptr) InterlockedDecrement((volatile int*)ptr)
#define AtomicDec16(ptr) InterlockedDecrement16((volatile short*)ptr)
#define CompareExchange(ptr,newValue,oldValue) _InterlockedCompareExchange64((volatile LONG64*)ptr,(LONG64)newValue,(LONG64)oldValue)
#define CompareExchange32(ptr,newValue,oldValue) _InterlockedCompareExchange((volatile LONG*)ptr,(LONG)newValue,(LONG)oldValue)
#define CompareExchange16(ptr,newValue,oldValue) _InterlockedCompareExchange16((volatile short*)ptr,(short)newValue,(short)oldValue)
#define GET_THREAD_IDENTITY() GetCurrentThreadId()
#else
#define AtomicAdd(ptr,val) __sync_add_and_fetch((long long*)ptr,val)
#define AtomicAdd32(ptr,val) __sync_add_and_fetch((int*)ptr, val)
#define AtomicAdd16(ptr,val) __sync_add_and_fetch((short*)ptr,val)
#define AtomicInc(ptr) __sync_add_and_fetch((long long*)ptr,1)
#define AtomicInc32(ptr) __sync_add_and_fetch((int*)ptr, 1)
#define AtomicDec(ptr) __sync_sub_and_fetch((long long*)ptr,1)
#define AtomicDec32(ptr) __sync_sub_and_fetch((int*)ptr,1)
#define AtomicInc16(ptr) __sync_add_and_fetch((short*)ptr,1)
#define AtomicDec16(ptr) __sync_sub_and_fetch((short*)ptr,1)
#define CompareExchange(ptr,newValue,oldValue) __sync_val_compare_and_swap((long long*)ptr, (long long)oldValue,(long long)newValue)
#define CompareExchange32(ptr,newValue,oldValue) __sync_val_compare_and_swap((int*)ptr,(int)oldValue,(int)newValue)
#define CompareExchange16(ptr,newValue,oldValue) __sync_val_compare_and_swap((short*)ptr,(short)oldValue,(short)newValue)
#endif