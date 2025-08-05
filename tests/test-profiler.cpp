#include <unistd.h>
#include <array>
#include <cstring>
#define TEST_GGML_PROFILER
#include <cstdio>
#include <cstdbool>
#include <ggml-profiler.h>
#include <ggml.h>
#include <ggml-alloc.h>

typedef bool (*TestFunc)();

// Define how a simple GGML compute graph can be constructed for the new GGML op.
ggml_tensor * build_graph(ggml_context * ctx, ggml_type type = GGML_TYPE_F32, std::array<int, 4> ne = {1, 2, 3, 4}) {
    // Step 1: create input tensors that don't depend on any other tensors:
    ggml_tensor * a = ggml_new_tensor(ctx, type, 4, (const int64_t*)ne.data());
    ggml_set_name(a, "a"); // Setting names is optional but it's useful for debugging.

    ggml_tensor * b = ggml_new_tensor(ctx, type, 4, (const int64_t*)ne.data());
    ggml_set_name(b, "b");

    // Step 2: use the op that you want to test in the GGML compute graph.
    ggml_tensor * out = ggml_mul(ctx, a, b); // For this example we're just doing a simple addition.
    ggml_set_name(out, "out");

    // Step 3: return the output tensor.
    return out;
}

bool test_profiler_new() {
    auto profiler = ggml_profiler_new();
    if (profiler == nullptr) {
        printf("\033[1;31mFailed to create profiler\033[0m\n");
        return false;
    }

    ggml_profiler_free(profiler);

    return true;
}

bool test_profiler_profile_start() {
    auto profiler = ggml_profiler_new();
    if (profiler == nullptr) {
        printf("\033[1;31mFailed to create profiler\033[0m\n");
        return false;
    }

    ggml_profiler_start(profiler);

    if (!ggml_profiler_is_active(profiler)) {
        printf("\033[1;31mProfiler should be active after start\033[0m\n");
        ggml_profiler_free(profiler);
        return false;
    }
    // Simulate some work
    for (volatile int i = 0; i < 1000000; i++);
    ggml_profiler_stop(profiler);

    if (ggml_profiler_is_active(profiler)) {
        printf("\033[1;31mProfiler should not be active after stop\033[0m\n");
        ggml_profiler_free(profiler);
        return false;
    }

    ggml_profiler_free(profiler);
    return true;
}


bool test_profiler_record_node() {
    auto profiler = ggml_profiler_new();
    if (profiler == nullptr) {
        printf("\033[1;31mFailed to create profiler\033[0m\n");
        return false;
    }

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * out = build_graph(ctx);

    ggml_profiler_start(profiler);

    // Record a node
    ggml_profiler_record_node_start(profiler, out);

    if (ggml_profiler_get_record_count(profiler) != 0) {
        printf("\033[1;31mProfiler should have no records before recording a node\033[0m\n");
        ggml_profiler_stop(profiler);
        ggml_profiler_free(profiler);
        return false;
    }

    usleep(1000); // Simulate some work

    ggml_profiler_record_node_end(profiler, out);

    if (ggml_profiler_get_record_count(profiler) != 1) {
        printf("\033[1;31mProfiler should have one record after recording a node\033[0m\n");
        ggml_profiler_stop(profiler);
        ggml_profiler_free(profiler);
        return false;
    }

    if (ggml_profiler_tensor_info_get_time(profiler->tensors[0]) < 1000) {
        printf("\033[1;31mProfiler recorded time should be greater than 1000 microseconds\033[0m\n");
        ggml_profiler_stop(profiler);
        ggml_profiler_free(profiler);
        return false;
    }

    ggml_profiler_stop(profiler);
    ggml_profiler_free(profiler);
    return true;
}

bool test_profiler_parse_tensor() {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * out = build_graph(ctx);

    auto info = ggml_profiler_tensor_info_new(out);
    if (strcmp(info->name, out->name) != 0) {
        printf("\033[1;31mTensor name mismatch: expected %s, got %s\033[0m\n", out->name, info->name);
        ggml_profiler_tensor_info_free(info);
        return false;
    }

    if (info->type != out->type) {
        printf("\033[1;31mTensor type mismatch: expected %d, got %d\033[0m\n", out->type, info->type);
        ggml_profiler_tensor_info_free(info);
        return false;
    }

    if (info->ne[0] != out->ne[0] || info->ne[1] != out->ne[1] ||
        info->ne[2] != out->ne[2] || info->ne[3] != out->ne[3]) {
        printf("\033[1;31mTensor shape mismatch: expected [%d, %d, %d, %d], got [%d, %d, %d, %d]\033[0m\n",
               out->ne[0], out->ne[1], out->ne[2], out->ne[3],
               info->ne[0], info->ne[1], info->ne[2], info->ne[3]);
        ggml_profiler_tensor_info_free(info);
        return false;
    }

    if (info->nbytes != ggml_nbytes(out)) {
        printf("\033[1;31mTensor size mismatch: expected %zu, got %zu\033[0m\n",
               ggml_nbytes(out), info->nbytes);
        ggml_profiler_tensor_info_free(info);
        return false;
    }

    ggml_profiler_tensor_info_free(info);

    return true;
}

static TestFunc tests[] = {
    test_profiler_new,
    test_profiler_profile_start,
    test_profiler_record_node,
    test_profiler_parse_tensor,
};

int main(int argc, char ** argv) {
    int npass = 0;
    int ntest = 0;

    for (int i = 0; i < sizeof(tests) / sizeof(TestFunc); i++) {
        ntest++;
        if (tests[i]()) {
            npass++;
        } else {
            printf("\033[1;31mTest %d failed\033[0m\n", i + 1);
        }
    }

    printf("%d/%d tests passed\n", npass, ntest);
    if (npass != ntest) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }
    printf("\033[1;32mOK\033[0m\n");

    return 0;
}
