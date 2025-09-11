#pragma once

#include <ggml-qnn-ion.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct ggml_shared_mem* ggml_shared_mem_t;

struct ggml_shared_mem_private;
struct ggml_shared_mem {
    // shared_memory fd
    int fd;
    // Shared memory for CPU
    void* mem;
    // Shared memory for opencl
    cl_mem cmem;

    struct ggml_shared_mem_private* priv;

    int (*alloc)(ggml_shared_mem_t shared_mem, size_t size);
    int (*alloc_cl)(ggml_shared_mem_t shared_mem, cl_context context, size_t size);
    void (*free)(ggml_shared_mem_t shared_mem);
};

ggml_shared_mem_t ggml_shared_mem_new(void);

#ifdef  __cplusplus
}
#endif
