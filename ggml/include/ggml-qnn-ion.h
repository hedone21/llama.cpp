#pragma once
#include <stdint.h>

/**
* Defination: void* rpcmem_alloc(int heapid, uint32 flags, int size);
* Allocate a buffer via ION and register it with the FastRPC framework.
* @param[in] heapid  Heap ID to use for memory allocation.
* @param[in] flags   ION flags to use for memory allocation.
* @param[in] size    Buffer size to allocate.
* @return            Pointer to the buffer on success; NULL on failure.
*/
typedef void *(*RpcMemAllocFn_t)(int, uint32_t, int);

/**
* Defination: void rpcmem_free(void* po);
* Free a buffer and ignore invalid buffers.
*/
typedef void (*RpcMemFreeFn_t)(void *);

/**
* Defination: int rpcmem_to_fd(void* po);
* Return an associated file descriptor.
* @param[in] po  Data pointer for an RPCMEM-allocated buffer.
* @return        Buffer file descriptor.
*/
typedef int (*RpcMemToFdFn_t)(void *);
