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

// 해시맵의 각 항목(슬롯)을 나타내는 구조체
typedef struct {
    void* key;                 // 키 (포인터 주소)
    ggml_shared_mem_t value;  // 값 (구조체 포인터)
    bool is_occupied;          // 해당 슬롯이 사용 중인지 여부
} hashmap_entry_t;

// 해시맵 전체를 나타내는 구조체
typedef struct {
    hashmap_entry_t* entries;  // 항목 배열
    size_t capacity;           // 총 용량
    size_t size;               // 현재 저장된 항목의 수
} hashmap_t;

hashmap_t* hashmap_create(size_t initial_capacity);
void hashmap_destroy(hashmap_t* map);
bool hashmap_put(hashmap_t* map, void* key, ggml_shared_mem_t value);
ggml_shared_mem_t hashmap_get(hashmap_t* map, void* key);
bool hashmap_remove(hashmap_t* map, void* key);

// 간단한 포인터 주소 해시 함수
static size_t hash_pointer(void* ptr, size_t capacity) {
    // 포인터 주소(정수)를 비트 연산하여 해시 값을 만듭니다.
    // 이는 간단한 예시이며, 더 정교한 해시 함수를 사용할 수도 있습니다.
    size_t hash = (size_t)ptr;
    hash = (hash >> 16) ^ hash;
    return hash % capacity;
}

// 내부 사용 함수: 크기 조절
static bool hashmap_resize(hashmap_t* map, size_t new_capacity);

static ggml_shared_mem_pool_t global_pool = NULL;

struct ggml_shared_mem_pool_private {
    hashmap_t *map;
};

struct ggml_shared_mem_private {
    int idx;
    void* handle;
    RpcMemAllocFn_t rpcmem_alloc;
    RpcMemFreeFn_t rpcmem_free;
    RpcMemToFdFn_t rpcmem_to_fd;
};

static ggml_shared_mem_t ggml_shared_mem_pool_get(ggml_shared_mem_pool_t pool, void* key) {
    if (pool == NULL || pool->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool\n", __func__);
        return NULL;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    assert(priv != NULL);

    ggml_shared_mem_t mem = hashmap_get(priv->map, key);
    if (mem == NULL) {
        mem = ggml_shared_mem_new();
        hashmap_put(priv->map, key, mem);
    }

    return mem;
}

void ggml_shared_mem_pool_put(ggml_shared_mem_pool_t pool, void* key, ggml_shared_mem_t shared_mem) {
    if (pool == NULL || pool->priv == NULL || shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool or shared_mem\n", __func__);
        return;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;

    hashmap_put(priv->map, key, shared_mem);
}

static void ggml_shared_mem_pool_free(ggml_shared_mem_pool_t pool) {
    if (pool == NULL || pool->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid pool\n", __func__);
        return;
    }

    struct ggml_shared_mem_pool_private* priv = pool->priv;
    assert(priv != NULL);

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

    priv->map = hashmap_create(256);

    pool->priv = priv;

    pool->get = ggml_shared_mem_pool_get;
    pool->put = ggml_shared_mem_pool_put;
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

    GGML_LOG_ERROR("[MYGO] %s: creating buffer with size %zu\n", __func__, shared_mem->mem_size);

    return 0;
}

int ggml_shared_mem_alloc_cl(ggml_shared_mem_t shared_mem, cl_context context, size_t size) {
    if (shared_mem == NULL || shared_mem->priv == NULL) {
        GGML_LOG_ERROR("[MYGO] %s: invalid shared_mem\n", __func__);
        return -1;
    }

    if (shared_mem->mem == NULL || shared_mem->fd < 0) {
        GGML_LOG_ERROR("[MYGO] %s: shared memory not allocated\n", __func__);
        return -1;
    }

    if (shared_mem->mem_size < size) {
        GGML_LOG_ERROR("[MYGO] %s: shared memory is insuffient\n", __func__);
        return -1;
    }

    cl_int err;
    cl_mem_ion_host_ptr host_ptr = {0};
    host_ptr.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
    host_ptr.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    host_ptr.ion_hostptr = shared_mem->mem;
    host_ptr.ion_filedesc = shared_mem->fd;

    GGML_LOG_ERROR("[MYGO] %s: creating cl buffer with size %zu\n", __func__, shared_mem->mem_size);

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

hashmap_t* hashmap_create(size_t initial_capacity) {
    if (initial_capacity == 0) {
        initial_capacity = 16; // 기본 초기 용량
    }

    hashmap_t* map = (hashmap_t*)malloc(sizeof(hashmap_t));
    if (!map) {
        return NULL;
    }

    map->entries = (hashmap_entry_t*)calloc(initial_capacity, sizeof(hashmap_entry_t));
    if (!map->entries) {
        free(map);
        return NULL;
    }

    map->capacity = initial_capacity;
    map->size = 0;

    return map;
}

void hashmap_destroy(hashmap_t* map) {
    if (!map) {
        return;
    }
    free(map->entries);
    free(map);
}

bool hashmap_put(hashmap_t* map, void* key, ggml_shared_mem_t value) {
    if (!map || !key) {
        return false;
    }

    // 로드 팩터가 임계값을 넘으면 크기를 두 배로 늘립니다.
    if ((double)map->size / map->capacity > LOAD_FACTOR_THRESHOLD) {
        if (!hashmap_resize(map, map->capacity * 2)) {
            return false;
        }
    }

    size_t index = hash_pointer(key, map->capacity);

    // 선형 탐사 (Linear Probing)
    for (size_t i = 0; i < map->capacity; ++i) {
        size_t current_index = (index + i) % map->capacity;
        hashmap_entry_t* entry = &map->entries[current_index];

        // 키가 이미 존재하면 값만 업데이트
        if (entry->is_occupied && entry->key == key) {
            entry->value = value;
            return true;
        }

        // 비어있는 슬롯을 찾으면 데이터 삽입
        if (!entry->is_occupied) {
            entry->key = key;
            entry->value = value;
            entry->is_occupied = true;
            map->size++;
            return true;
        }
    }

    // 해시맵이 가득 찬 경우 (이론상 resize 로직 때문에 발생하기 어려움)
    return false;
}

ggml_shared_mem_t hashmap_get(hashmap_t* map, void* key) {
    if (!map || !key) {
        return NULL;
    }

    size_t index = hash_pointer(key, map->capacity);

    for (size_t i = 0; i < map->capacity; ++i) {
        size_t current_index = (index + i) % map->capacity;
        hashmap_entry_t* entry = &map->entries[current_index];

        // 해당 슬롯이 비어있으면 키가 존재하지 않음
        if (!entry->is_occupied) {
            // 단, 삭제된 항목이 있을 수 있으므로 계속 탐색해야 하지만,
            // 이 간단한 구현에서는 삭제 시 is_occupied만 false로 바꾸므로 여기서 중단.
            // (더 정교한 구현에서는 DELETED 상태를 둘 수 있음)
            return NULL;
        }

        if (entry->key == key) {
            return entry->value;
        }
    }

    return NULL;
}

bool hashmap_remove(hashmap_t* map, void* key) {
    if (!map || !key) {
        return false;
    }

    size_t index = hash_pointer(key, map->capacity);

    for (size_t i = 0; i < map->capacity; ++i) {
        size_t current_index = (index + i) % map->capacity;
        hashmap_entry_t* entry = &map->entries[current_index];

        if (entry->is_occupied && entry->key == key) {
            entry->is_occupied = false;
            entry->key = NULL;
            entry->value = NULL;
            map->size--;
            // TODO: 삭제 후 탐색 문제를 해결하기 위해 삭제된 항목에 대한 표시(tombstone)를
            // 추가하고 get() 로직을 수정하는 것이 더 안정적입니다.
            // 이 구현은 간단함을 위해 생략합니다.
            return true;
        }
    }

    return false;
}


static bool hashmap_resize(hashmap_t* map, size_t new_capacity) {
    if (new_capacity < map->capacity) return false;

    hashmap_entry_t* new_entries = (hashmap_entry_t*)calloc(new_capacity, sizeof(hashmap_entry_t));
    if (!new_entries) {
        return false;
    }

    // 기존 항목들을 새로운 해시맵에 재배치
    for (size_t i = 0; i < map->capacity; ++i) {
        if (map->entries[i].is_occupied) {
            void* key = map->entries[i].key;
            ggml_shared_mem_t value = map->entries[i].value;
            size_t new_index = hash_pointer(key, new_capacity);

            // 새 위치에서 충돌 해결
            for (size_t j = 0; j < new_capacity; ++j) {
                size_t current_index = (new_index + j) % new_capacity;
                if (!new_entries[current_index].is_occupied) {
                    new_entries[current_index].key = key;
                    new_entries[current_index].value = value;
                    new_entries[current_index].is_occupied = true;
                    break;
                }
            }
        }
    }

    free(map->entries);
    map->entries = new_entries;
    map->capacity = new_capacity;

    return true;
}

