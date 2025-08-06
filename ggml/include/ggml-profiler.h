#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <stddef.h>

#define GGML_PROFILER_RECORD_MAX (1024 * 1024)
#define GGML_PROFILER_RECORDING_MAX 32
#define GGML_PROFILER_BACKEND_DEVICE_NAME_MAX 8

typedef struct ggml_profiler * ggml_profiler_t;
typedef struct ggml_profiler_tensor_info * ggml_profiler_tensor_info_t;

// Tensor information structure for profiling
struct ggml_profiler_tensor_info {
    // Tensor information
    const char * name; // tensor name
    enum ggml_type type; // tensor type
    enum ggml_op op; // tensor operation

    int64_t ne[GGML_MAX_DIMS]; // tensor dimensions
    size_t nbytes; // number of bytes in the tensor
    int64_t n_src; // number of source tensors
    ggml_profiler_tensor_info_t src[GGML_MAX_SRC]; // source tensors info

    // Profiling information
    char backend[GGML_PROFILER_BACKEND_DEVICE_NAME_MAX]; // backend device name
    int64_t start_time; // start time of the tensor computation
    int64_t end_time; // end time of the tensor computation
    size_t memory_usage; // memory usage during the tensor computation
    size_t io_usage; // I/O usage during the tensor computation

    int start_core; // start core for the tensor computation
    int end_core; // end core for the tensor computation. To check if the tensor is computed on multiple cores
    bool core_changed; // true if the core changed during the tensor computation
};

// Create a new tensor info instance from a ggml tensor
ggml_profiler_tensor_info_t ggml_profiler_tensor_info_new(const struct ggml_tensor * tensor);

// Free the tensor info instance
void ggml_profiler_tensor_info_free(ggml_profiler_tensor_info_t info);

// Set the backend device name for the tensor info
inline void ggml_profiler_tensor_info_set_backend(ggml_profiler_tensor_info_t info, const char * backend) {
    if (info == NULL || backend == NULL) {
        return; // If info or backend is null, do nothing
    }
    snprintf(info->backend, sizeof(info->backend), "%s", backend); // Set the backend device name
}

// Get tensor compuation time in microseconds
inline int64_t ggml_profiler_tensor_info_get_time(const ggml_profiler_tensor_info_t info) {
    if (info == NULL || info->end_time <= info->start_time) {
        return 0; // Return 0 if info is null or end time is not greater than start time
    }
    return info->end_time - info->start_time; // Return the computation time in microseconds
}

struct ggml_profiler {
    bool on_started; // true if profiling is started

    ggml_profiler_tensor_info_t tensors[GGML_PROFILER_RECORD_MAX]; // list of tensors being profiled
    size_t n_tensors; // number of tensors being profiled
    ggml_profiler_tensor_info_t recording[GGML_PROFILER_RECORDING_MAX]; // list of recording tensors
    size_t n_recording; // number of recording tensors
};

// Function to create a new profiler instance
// This function is used only in test code.
// Use ggml_profiler_get_instance() to get the singleton instance in production code.
ggml_profiler_t ggml_profiler_new(void);
// Get the singleton profiler instance
ggml_profiler_t ggml_profiler_get_instance(void);

// Start profiling
void ggml_profiler_start(ggml_profiler_t profiler);
// Stop profiling
void ggml_profiler_stop(ggml_profiler_t profiler);
// Check if the profiler is started
inline bool ggml_profiler_is_active(ggml_profiler_t profiler) {
    return profiler != NULL && profiler->on_started;
}

// Record node(tensor) computation start
void ggml_profiler_record_node_start(ggml_profiler_t profiler, const struct ggml_tensor * tensor);
// Record node(tensor) computation end
void ggml_profiler_record_node_end(ggml_profiler_t profiler, const struct ggml_tensor * tensor);

// Report profiling information
// If path is not null, save the report to the file at the specified path.
void ggml_profiler_report(ggml_profiler_t profiler, const char * path);

// Get recoreded tensor info count
size_t ggml_profiler_get_record_count(ggml_profiler_t profiler);

// Get recording tensor infos
const ggml_profiler_tensor_info_t* ggml_profiler_get_recording(ggml_profiler_t profiler);

static bool ggml_profiler_init(ggml_profiler_t profiler) {
    if (profiler == NULL) {
        return false;
    }
    profiler->on_started = false;
    return true; // Return true if initialization is successful
}

static void ggml_profiler_free(ggml_profiler_t profiler) {
    if (profiler == NULL) {
        return; // If profiler is null, do nothing
    }

    // Free all recorded tensor infos
    for (size_t i = 0; i < profiler->n_tensors; i++) {
        ggml_profiler_tensor_info_free(profiler->tensors[i]);
    }

    // Free all recording tensor infos
    for (size_t i = 0; i < profiler->n_recording; i++) {
        ggml_profiler_tensor_info_free(profiler->recording[i]);
    }

    free(profiler); // Free the profiler instance
}

#ifdef  __cplusplus
}
#endif


