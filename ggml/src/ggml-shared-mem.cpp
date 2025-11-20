#include <ggml.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <sys/mman.h>
#include <stddef.h>
#include <stdbool.h>
#include "ggml-impl.h"
#include "ggml-shared-mem.h"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
// #define CL_MEM_ION_HOST_PTR_QCOM         0x40B2
#define CL_MEM_EXT_HOST_PTR_QCOM                   (1 << 29)
#define LOAD_FACTOR_THRESHOLD 0.75 // 크기 조절을 결정하는 임계값
#define GGML_MAX_SHARED_MEM_POOLS 128

cl_context GGML_SHARED_CL_CONTEXT = NULL;
cl_command_queue GGML_SHARED_CL_QUEUE = NULL;

static int IDX = 0;
static ggml_shared_mem_pool_t global_pool = NULL;

struct ggml_shared_mem_pool_private {
    ggml_shared_mem_t shared_mem[GGML_MAX_SHARED_MEM_POOLS];
    int mem_pos;
};

struct ggml_shared_mem_private {
    int idx;
    void* handle;
    RpcMemAllocFn_t rpcmem_alloc;
    RpcMemFreeFn_t rpcmem_free;
    RpcMemToFdFn_t rpcmem_to_fd;
};

static ggml_shared_mem_t ggml_shared_mem_pool_get(ggml_shared_mem_pool_t pool) {
    if (pool == NULL || pool->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool\n", __func__);
        return NULL;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    assert(priv != NULL);

    // This is cirular allocation
    if (priv->mem_pos >= GGML_MAX_SHARED_MEM_POOLS) {
        priv->mem_pos = 0;
    }

    return priv->shared_mem[priv->mem_pos++];
}

static void ggml_shared_mem_pool_free(ggml_shared_mem_pool_t pool) {
    if (pool == NULL || pool->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool\n", __func__);
        return;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    assert(priv != NULL);

    for (int i = 0; i < GGML_MAX_SHARED_MEM_POOLS; i++) {
        ggml_shared_mem_t shared_mem = priv->shared_mem[i];
        if (shared_mem != NULL) {
            shared_mem->free(shared_mem);
            priv->shared_mem[i] = NULL;
        }
    }

    free(priv);
    free(pool);
    global_pool = NULL;
}

static ggml_shared_mem_pool_t ggml_shared_mem_pool_new(void) {
    ggml_shared_mem_pool_t pool = (ggml_shared_mem_pool_t)malloc(sizeof(struct ggml_shared_mem_pool));
    if (pool == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: failed to allocate memory for ggml_shared_mem_pool\n", __func__);
        return NULL;
    }

    struct ggml_shared_mem_pool_private* priv = (struct ggml_shared_mem_pool_private*)malloc(sizeof(struct ggml_shared_mem_pool_private));
    if (priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: failed to allocate memory for ggml_shared_mem_pool_private\n", __func__);
        free(pool);
        return NULL;
    }

    priv->mem_pos = 0;

    for (int i = 0; i < GGML_MAX_SHARED_MEM_POOLS; i++) {
        ggml_shared_mem_t shared_mem = ggml_shared_mem_new();
        priv->shared_mem[i] = shared_mem;
        shared_mem->alloc_cl(shared_mem, GGML_SHARED_CL_CONTEXT, 128 * 1024); // Preallocate 128KB

        // GGML_LOG_ERROR("[MYGO] %s: preloading shared memory pool %d with size %zu\n", __func__, i, shared_mem->mem_size);
    }

    pool->priv = priv;

    pool->get = ggml_shared_mem_pool_get;
    pool->free = ggml_shared_mem_pool_free;

    return pool;
}

ggml_shared_mem_pool_t ggml_get_shared_mem_pool(void) {
    if (global_pool == NULL) {
        global_pool = ggml_shared_mem_pool_new();
    }
    return global_pool;
}

int ggml_shared_mem_alloc(ggml_shared_mem_t shared_mem, size_t size) {
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s/%d: invalid shared_mem\n", __func__, __LINE__);
        return -1;
    }

    struct ggml_shared_mem_private* priv = shared_mem->priv;
    assert(priv != NULL);
    assert(priv->rpcmem_alloc != NULL);
    assert(priv->rpcmem_free != NULL);
    assert(priv->rpcmem_to_fd != NULL);

    void* mem_ptr = priv->rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
    int mem_fd = priv->rpcmem_to_fd(mem_ptr);
    if (mem_fd < 0) {
        GGML_LOG_ERROR("[MYGO] %s: failed to get file descriptor from allocated memory\n", __func__);
        shared_mem->free(shared_mem);
        return -1;
    }

    // Note: mmap is not needed as rpcmem_alloc already provides a usable pointer
    // void* aligned_memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);
    // if (aligned_memory == MAP_FAILED) {
    //     GGML_LOG_ERROR("[MYGO] %s: mmap failed\n", __func__);
    //     shared_mem->free(shared_mem);
    //     return -1;
    // }

    shared_mem->mem = mem_ptr;
    shared_mem->fd = mem_fd;
    shared_mem->mem_size = size;

    // GGML_LOG_ERROR("[MYGO] %s: creating buffer with size %zu\n", __func__, shared_mem->mem_size);

    return 0;
}

int ggml_shared_mem_alloc_cl(ggml_shared_mem_t shared_mem, cl_context context, size_t size) {
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s/%d: invalid shared_mem\n", __func__, __LINE__);
        return -1;
    }

    // if (shared_mem->mem == NULL || shared_mem->fd < 0) {
    //     GGML_LOG_ERROR("[MYGO] %s: shared memory not allocated\n", __func__);
    //     return -1;
    // }

    shared_mem->mem_size = size;
    // if (shared_mem->mem_size < size) {
    //     GGML_LOG_ERROR("[MYGO] %s: shared memory is insuffient\n", __func__);
    //     return -1;
    // }

    cl_int err;
    cl_mem_ion_host_ptr host_ptr = {0};
    host_ptr.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
    host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    host_ptr.ion_hostptr = shared_mem->mem;
    host_ptr.ion_filedesc = shared_mem->fd;

    // cl_mem mem = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
    //                             shared_mem->mem_size, &host_ptr, &err);
    cl_mem mem = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                                shared_mem->mem_size, NULL, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("[MYGO] %s: clCreateBuffer failed with error %d\n", __func__, err);
        return -1;
    }

    shared_mem->cmem = mem;
    shared_mem->mem = clEnqueueMapBuffer(GGML_SHARED_CL_QUEUE, mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
        size, 0, NULL, NULL, &err);

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
    shared_mem->mem_size = 0;
    shared_mem->fd = -1;

    // shared_mem->alloc = ggml_shared_mem_alloc;
    shared_mem->alloc_cl = ggml_shared_mem_alloc_cl;
    shared_mem->free = ggml_shared_mem_free;

    return shared_mem;
}
