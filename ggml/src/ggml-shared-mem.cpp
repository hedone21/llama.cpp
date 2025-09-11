#include <ggml.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <cassert>
#include <sys/mman.h>
#include "ggml-impl.h"
#include "ggml-shared-mem.h"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define CL_QCOM_ion_host_ptr 1
#define CL_MEM_EXT_HOST_PTR_QCOM                   (1 << 29)

struct ggml_shared_mem_private {
    void* handle;
    RpcMemAllocFn_t rpcmem_alloc;
    RpcMemFreeFn_t rpcmem_free;
    RpcMemToFdFn_t rpcmem_to_fd;
};

int ggml_shared_mem_alloc(ggml_shared_mem_t shared_mem, size_t size) {
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid shared_mem\n", __func__);
        return -1;
    }

    struct ggml_shared_mem_private* priv = shared_mem->priv;
    assert(priv != NULL);
    assert(priv->rpcmem_alloc != NULL);
    assert(priv->rpcmem_free != NULL);
    assert(priv->rpcmem_to_fd != NULL);

    void* mem_ptr = priv->rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, 1, size);
    int mem_fd = priv->rpcmem_to_fd(mem_ptr);
    if (mem_fd < 0) {
        GGML_LOG_ERROR("[MYGO] %s: failed to get file descriptor from allocated memory\n", __func__);
        shared_mem->free(shared_mem);
        return -1;
    }

    void* aligned_memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);
    if (aligned_memory == MAP_FAILED) {
        GGML_LOG_ERROR("[MYGO] %s: mmap failed\n", __func__);
        shared_mem->free(shared_mem);
        return -1;
    }

    shared_mem->fd = mem_fd;
    shared_mem->mem = aligned_memory;

    return 0;
}

int ggml_shared_mem_alloc_cl(ggml_shared_mem_t shared_mem, cl_context context, size_t size) {
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid shared_mem\n", __func__);
        return -1;
    }

    cl_int err;
    cl_mem_ion_host_ptr host_ptr = {0};
    host_ptr.ext_host_ptr.allocation_type = CL_MEM_EXT_HOST_PTR_QCOM;
    host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    host_ptr.ion_hostptr = shared_mem->mem;
    host_ptr.ion_filedesc = shared_mem->fd;

    cl_mem mem = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                size, &host_ptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("[MYGO] %s: clCreateBuffer failed with error %d\n", __func__, err);
        return -1;
    }

    return 0;
}

void ggml_shared_mem_free(ggml_shared_mem_t shared_mem) {
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid shared_mem\n", __func__);
        return;
    }

    struct ggml_shared_mem_private* priv = shared_mem->priv;
    assert(priv != NULL);
    assert(priv->rpcmem_free != NULL);

    if (shared_mem->mem != NULL) {
        // Assuming we have a way to get the original pointer used in rpcmem_alloc
        priv->rpcmem_free(shared_mem->mem);
        shared_mem->mem = NULL;
    }

    dlclose(priv->handle);
    free(priv);
    free(shared_mem);
}

ggml_shared_mem_t ggml_shared_mem_new(void) {
    void* handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: failed to load libcdsprpc.so\n", __func__);
        return NULL;
    }

    RpcMemAllocFn_t rpcmem_alloc = (RpcMemAllocFn_t)dlsym(handle, "rpcmem_alloc");
    RpcMemFreeFn_t rpcmem_free = (RpcMemFreeFn_t)dlsym(handle, "rpcmem_free");
    RpcMemToFdFn_t rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(handle, "rpcmem_to_fd");

    if (rpcmem_alloc == NULL || rpcmem_free == NULL || rpcmem_to_fd == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: failed to load symbols from libcdsprpc.so\n", __func__);
        dlclose(handle);
        return NULL;
    }

    struct ggml_shared_mem_private* priv = (struct ggml_shared_mem_private*)malloc(sizeof(struct ggml_shared_mem_private));
    if (priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: failed to allocate memory for ggml_shared_mem_private\n", __func__);
        dlclose(handle);
        return NULL;
    }

    priv->handle = handle;
    priv->rpcmem_alloc = rpcmem_alloc;
    priv->rpcmem_free = rpcmem_free;
    priv->rpcmem_to_fd = rpcmem_to_fd;

    ggml_shared_mem_t shared_mem = (ggml_shared_mem_t)malloc(sizeof(struct ggml_shared_mem));
    if (shared_mem == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: failed to allocate memory for ggml_shared_mem\n", __func__);
        free(priv);
        dlclose(handle);
        return NULL;
    }

    shared_mem->priv = priv;
    shared_mem->mem = NULL;
    shared_mem->cmem = NULL;

    shared_mem->alloc = ggml_shared_mem_alloc;
    shared_mem->alloc_cl = ggml_shared_mem_alloc_cl;
    shared_mem->free = ggml_shared_mem_free;

    return shared_mem;
}


