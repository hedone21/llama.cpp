#pragma once

#include <ggml-qnn-ion.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stddef.h>

#define ggml_offsetof(type, member) ((size_t)&(((type *)0)->member))
#define ggml_container_of(ptr, type, member) \
    reinterpret_cast<type*>( \
        reinterpret_cast<char*>(ptr) - offsetof(type, member) \
    )

#ifdef  __cplusplus
extern "C" {
#endif
typedef struct ggml_shared_mem* ggml_shared_mem_t;
typedef struct ggml_shared_mem_pool* ggml_shared_mem_pool_t;

extern cl_context GGML_SHARED_CL_CONTEXT;
extern cl_command_queue GGML_SHARED_CL_QUEUE;

struct ggml_shared_mem_private;
struct ggml_shared_mem {
    // shared_memory fd
    int fd;
    // Shared memory for CPU
    void* mem;
    // Shared memory for opencl
    cl_mem cmem;

    size_t mem_size;

    struct ggml_shared_mem_private* priv;

    // int (*alloc)(ggml_shared_mem_t shared_mem, size_t size);
    int (*alloc_cl)(ggml_shared_mem_t shared_mem, cl_context context, size_t size);
    void (*free)(ggml_shared_mem_t shared_mem);
};

struct ggml_shared_mem_pool_private;
struct ggml_shared_mem_pool {
    struct ggml_shared_mem_pool_private* priv;

    ggml_shared_mem_t (*get)(ggml_shared_mem_pool_t pool);
    void (*free)(ggml_shared_mem_pool_t pool);
};

ggml_shared_mem_t ggml_shared_mem_new(void);
ggml_shared_mem_pool_t ggml_get_shared_mem_pool(void);

#ifdef  __cplusplus
}
#endif
