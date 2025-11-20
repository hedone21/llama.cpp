#include <time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <unistd.h> // for sysconf

/**
 * 안드로이드/리눅스 CPU 사용량 모니터링 (싱글턴 패턴)
 * * @return 0.0 ~ 100.0 (사용률 %), -1.0 (에러)
 * @note 첫 번째 호출 시에는 기준점이 없으므로 0.0을 반환합니다.
 */
double ggml_utils_get_cpu_usage() {
    // [상태 보존 변수]
    static struct timespec prev_time = {0, 0};
    static double prev_cpu_time_sec = 0.0;
    static int is_initialized = 0;

    // [1. 현재 실제 시간 측정 (Wall Clock)]
    struct timespec curr_time;
    if (clock_gettime(CLOCK_MONOTONIC, &curr_time) == -1) {
        return -1.0;
    }

    // [2. 내 프로세스 CPU 누적 시간 측정 (User + Kernel)]
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == -1) {
        return -1.0;
    }

    // timeval 구조체(sec, usec)를 초(double) 단위로 합산
    double curr_cpu_time_sec =
        (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec * 1e-6 +
        (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec * 1e-6;

    // [3. 초기화 체크]
    if (!is_initialized) {
        prev_time = curr_time;
        prev_cpu_time_sec = curr_cpu_time_sec;
        is_initialized = 1;
        return 0.0; // 첫 호출은 기준점이 없으므로 0 리턴
    }

    // [4. 델타 계산]
    // 실제 흐른 시간 (초 단위)
    double time_delta = (curr_time.tv_sec - prev_time.tv_sec) +
                        (curr_time.tv_nsec - prev_time.tv_nsec) * 1e-9;

    // 프로세스가 CPU를 쓴 시간 (초 단위)
    double cpu_delta = curr_cpu_time_sec - prev_cpu_time_sec;

    // 상태 업데이트
    prev_time = curr_time;
    prev_cpu_time_sec = curr_cpu_time_sec;

    // 예외 처리: 시간이 거의 안 흘렀을 때 (Division by Zero 방지)
    if (time_delta < 0.0001) {
        return 0.0;
    }

    // [5. 사용률 계산]
    double percentage = (cpu_delta / time_delta) * 100.0;

    // (선택사항) 만약 전체 시스템 대비 % (0~100)로 정규화하고 싶다면 아래 주석 해제
    /*
    static long num_cores = 0;
    if (num_cores == 0) num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores > 0) percentage /= num_cores;
    */

    return percentage;
}
