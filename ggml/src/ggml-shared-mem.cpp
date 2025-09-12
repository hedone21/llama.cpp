#include <ggml.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <cassert>
#include <sys/mman.h>
#include "ggml-impl.h"
#include "ggml-shared-mem.h"

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
#define CL_MEM_EXT_HOST_PTR_QCOM                   (1 << 29)

static const int GGML_MAX_SHARED_MEM_POOLS = 4;
static int IDX = 0;
static ggml_shared_mem_pool_t global_pool = NULL;

struct ggml_shared_mem_pool_private {
    struct ggml_shared_mem *shared_mem[GGML_MAX_SHARED_MEM_POOLS];
    bool used[GGML_MAX_SHARED_MEM_POOLS];
    int n_used;
};

struct ggml_shared_mem_private {
    int idx;
    void* handle;
    RpcMemAllocFn_t rpcmem_alloc;
    RpcMemFreeFn_t rpcmem_free;
    RpcMemToFdFn_t rpcmem_to_fd;
};

static ggml_shared_mem_t ggml_shared_mem_pool_get(ggml_shared_mem_pool_t pool) {
    GGML_LOG_ERROR("[MYGO] %s: called\n", __func__);
    if (pool == NULL || pool->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool\n", __func__);
        return NULL;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    assert(priv != NULL);

    if (priv->n_used >= GGML_MAX_SHARED_MEM_POOLS) {
        GGML_LOG_ERROR("[MYGO] %s: all shared memory pools are in use\n", __func__);
        return NULL;
    }

    for (int i = 0; i < GGML_MAX_SHARED_MEM_POOLS; i++) {
        if (!priv->used[i]) {
            priv->used[i] = true;
            priv->n_used++;
            GGML_LOG_ERROR("[MYGO] %s: allocated shared memory pool at index %d\n", __func__, i);
            return priv->shared_mem[i];
        }
    }

    GGML_LOG_ERROR("[MYGO] %s: no available shared memory pool found\n", __func__);
    return NULL;
}

void ggml_shared_mem_pool_put(ggml_shared_mem_pool_t pool, ggml_shared_mem_t shared_mem) {
    if (pool == NULL || pool->priv == NULL || shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool or shared_mem\n", __func__);
        return;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    struct ggml_shared_mem_private* sm_priv = shared_mem->priv;
    assert(priv != NULL);
    assert(sm_priv != NULL);

    int idx = sm_priv->idx;
    if (idx < 0 || idx >= GGML_MAX_SHARED_MEM_POOLS) {
        GGML_LOG_ERROR("[MYGO] %s: index out of range\n", __func__);
        return;
    }

    if (!priv->used[idx]) {
        GGML_LOG_ERROR("[MYGO] %s: shared memory at index %d is not in use\n", __func__, idx);
        return;
    }

    priv->used[idx] = false;
    priv->n_used--;
}

static void ggml_shared_mem_pool_free(ggml_shared_mem_pool_t pool) {
    if (pool == NULL || pool->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool\n", __func__);
        return;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    assert(priv != NULL);

    for (int i = 0; i < GGML_MAX_SHARED_MEM_POOLS; i++) {
        if (priv->shared_mem[i] != NULL) {
            priv->shared_mem[i]->free(priv->shared_mem[i]);
            priv->shared_mem[i] = NULL;
            priv->used[i] = false;
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

    priv->n_used = 0;

    for (int i = 0; i < 4; i++) {
        ggml_shared_mem_t shared_mem = ggml_shared_mem_new();
        shared_mem->alloc(shared_mem, 1024 * 1024 * 300); // pre-allocate 300MB

        priv->shared_mem[i] = shared_mem;
        priv->used[i] = false;
    }

    pool->priv = priv;

    pool->get = ggml_shared_mem_pool_get;
    pool->put = ggml_shared_mem_pool_put;
    pool->free = ggml_shared_mem_pool_free;
}

ggml_shared_mem_pool_t ggml_get_shared_mem_pool(void) {
    if (global_pool == NULL) {
        global_pool = ggml_shared_mem_pool_new();
    }
    return global_pool;
}

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

    return 0;
}

int ggml_shared_mem_alloc_cl(ggml_shared_mem_t shared_mem, cl_context context, size_t size) {
    GGML_LOG_ERROR("[MYGO] %s: size=%zu\n", __func__, size);
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid shared_mem\n", __func__);
        return -1;
    }

    if (shared_mem->mem == NULL || shared_mem->fd < 0 || shared_mem->mem_size < size) {
        GGML_LOG_ERROR("[MYGO] %s: shared memory not allocated or insufficient size\n", __func__);
        return -1;
    }

    cl_int err;
    cl_mem_ion_host_ptr host_ptr = {0};
    host_ptr.ext_host_ptr.allocation_type = CL_MEM_EXT_HOST_PTR_QCOM;
    host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    host_ptr.ion_hostptr = shared_mem->mem;
    host_ptr.ion_filedesc = shared_mem->fd;

    cl_mem mem = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                shared_mem->mem_size, &host_ptr, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("[MYGO] %s: clCreateBuffer failed with error %d\n", __func__, err);
        return -1;
    }

    shared_mem->cmem = mem;

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

    priv->idx = IDX++;
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

    shared_mem->alloc = ggml_shared_mem_alloc;
    shared_mem->alloc_cl = ggml_shared_mem_alloc_cl;
    shared_mem->free = ggml_shared_mem_free;

    return shared_mem;
}


