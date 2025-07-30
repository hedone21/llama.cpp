#include "ggml-impl.h"
#include <ggml-profiler.h>
#include <chrono>
#include <string>

static std::string get_tensor_info(ggml_profiler_tensor_info_t tensor) {
    std::string info = "Tensor: " + std::string(tensor->name) + ", Type: " + ggml_type_name(tensor->type) +
                       ", Op: " + ggml_op_name(tensor->op) + ", Shape: [" +
                       std::to_string(tensor->ne[0]) + ", " +
                       std::to_string(tensor->ne[1]) + ", " +
                       std::to_string(tensor->ne[2]) + ", " +
                       std::to_string(tensor->ne[3]) + "]";
    return info;
}

static int64_t get_current_time() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

ggml_profiler_tensor_info_t ggml_profiler_tensor_info_new(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Tensor is null");
        return nullptr; // Return nullptr if tensor is null
    }

    ggml_profiler_tensor_info_t info = new ggml_profiler_tensor_info;

    info->name = tensor->name;
    info->type = tensor->type;
    info->op = tensor->op;

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        info->ne[i] = tensor->ne[i];
    }
    info->nbytes = ggml_nbytes(tensor);
    info->n_src = 0;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            info->src[info->n_src++] = ggml_profiler_tensor_info_new(tensor->src[i]);
        }
    }

    // Initialize profiling information
    snprintf(info->backend, sizeof(info->backend), "-");
    info->start_time = get_current_time();
    info->end_time = 0;
    info->memory_usage = 0;
    info->io_usage = 0;

    return info;
}

void ggml_profiler_tensor_info_free(ggml_profiler_tensor_info_t info) {
    if (info == nullptr) {
        return; // If info is null, do nothing
    }

    // Free source tensor infos
    for (int i = 0; i < info->n_src; i++) {
        ggml_profiler_tensor_info_free(info->src[i]);
    }

    delete info; // Free the info structure itself
}

ggml_profiler_t ggml_profiler_new(void) {
    ggml_profiler_t profiler = new ggml_profiler;

    if (!ggml_profiler_init(profiler)) {
        ggml_profiler_free(profiler);
        GGML_LOG_ERROR("Failed to initialize ggml_profiler");
        return nullptr; // Return nullptr if initialization fails
    }

    return profiler;
}

ggml_profiler_t ggml_profiler_get_instance(void) {
    static ggml_profiler_t instance = nullptr;
    if (instance == nullptr) {
        instance = ggml_profiler_new();
    }
    return instance;
}

void ggml_profiler_start(ggml_profiler_t profiler) {
    if (profiler == nullptr) {
        GGML_LOG_ERROR("Profiler instance is null");
        return;
    }
    profiler->on_started = true;
}

void ggml_profiler_stop(ggml_profiler_t profiler) {
    if (profiler == nullptr) {
        GGML_LOG_ERROR("Profiler instance is null");
        return;
    }
    profiler->on_started = false;
}

void ggml_profiler_record_node_start(ggml_profiler_t profiler, const struct ggml_tensor * tensor) {
    if (!ggml_profiler_is_active(profiler)) {
        return; // If profiling is not active, do nothing
    }

    auto info = ggml_profiler_tensor_info_new(tensor);
    if (info == nullptr) {
        GGML_LOG_ERROR("Failed to create tensor info for %s", tensor->name);
        return; // If tensor info creation fails, do nothing
    }

    profiler->recording.push_back(info);

    GGML_LOG_DEBUG("Node start: %s\n", get_tensor_info(info).c_str());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            GGML_LOG_DEBUG("  Source[%d]: %s\n", i, get_tensor_info(info->src[i]).c_str());
        }
    }
}

void ggml_profiler_record_node_end(ggml_profiler_t profiler, const struct ggml_tensor * tensor) {
    if (!ggml_profiler_is_active(profiler)) {
        return; // If profiling is not active, do nothing
    }

    if (tensor == nullptr) {
        GGML_LOG_ERROR("Tensor is null");
        return; // If tensor is null, do nothing
    }

    // Find the tensor info in the recording list
    for (auto it = profiler->recording.begin(); it != profiler->recording.end(); ++it) {
        if ((*it)->name == tensor->name) {
            // Update the end time and other profiling information
            (*it)->end_time = get_current_time();
            // (*it)->memory_usage = ggml_nbytes(tensor);
            // (*it)->io_usage = 0; // Placeholder for I/O usage, if applicable
            profiler->tensors.push_back(*it); // Add to the main tensor list
            profiler->recording.erase(it); // Remove from the recording list
            break;
        }
    }

    // Record the end of the tensor computation
    GGML_LOG_DEBUG("Node end: %s\n", tensor->name);
}

void ggml_profiler_report(ggml_profiler_t profiler, const char * path) {
    if (profiler == nullptr || !ggml_profiler_is_active(profiler)) {
        GGML_LOG_ERROR("Profiler instance is null or not active");
        return; // If profiler is null or not active, do nothing
    }

    // Print profiling information for each tensor
    for (const auto & tensor_info : profiler->tensors) {
        int64_t time = ggml_profiler_tensor_info_get_time(tensor_info);
        GGML_LOG_INFO("Tensor: %s, Type: %s, Op: %s, Shape: [%d, %d, %d, %d], Time: %lld us, Memory: %zu bytes",
                      tensor_info->name, ggml_type_name(tensor_info->type), ggml_op_name(tensor_info->op),
                      tensor_info->ne[0], tensor_info->ne[1], tensor_info->ne[2], tensor_info->ne[3],
                      time, tensor_info->nbytes);
    }

    // Optionally write to a file if path is provided
    if (path != nullptr) {
        FILE *file = fopen(path, "w");
        if (file == nullptr) {
            GGML_LOG_ERROR("Failed to open file %s for writing", path);
            return; // If file opening fails, do nothing
        }

        for (const auto & tensor_info : profiler->tensors) {
            int64_t time = ggml_profiler_tensor_info_get_time(tensor_info);
            fprintf(file, "Tensor: %s, Type: %s, Op: %s, Shape: [%d, %d, %d, %d], Time: %lld us, Memory: %zu bytes\n",
                    tensor_info->name, ggml_type_name(tensor_info->type), ggml_op_name(tensor_info->op),
                    tensor_info->ne[0], tensor_info->ne[1], tensor_info->ne[2], tensor_info->ne[3],
                    time, tensor_info->nbytes);
        }

        fclose(file);
    }
    GGML_LOG_INFO("Profiling report generated successfully.");
}
