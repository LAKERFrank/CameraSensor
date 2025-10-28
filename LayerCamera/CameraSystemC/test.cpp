#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring> // For memset
#include <unistd.h>

timespec get_clock_time(clockid_t clock_type) {
    timespec ts;
    if (clock_gettime(clock_type, &ts) != 0) {
        throw std::runtime_error("Failed to get clock time");
    }
    return ts;
}

std::chrono::system_clock::time_point timespec_to_timepoint(const timespec &ts) {
    return std::chrono::system_clock::time_point(
        std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec));
}

long long timespec_to_ns(timespec ts) {
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    try {
        // Get current real time
        timespec realtime1 = get_clock_time(CLOCK_REALTIME);
        // Get boot time (real time - uptime)
        timespec boottime = get_clock_time(CLOCK_BOOTTIME);
        // Get current real time
        timespec realtime2 = get_clock_time(CLOCK_REALTIME);

        long long r1 = timespec_to_ns(realtime1);
        long long r2 = timespec_to_ns(realtime2);

        printf("check1:%10lld.%9lld\n", (long long)(r1/1e9), r1 - (long long)((long long)(r1/1e9)*1e9));
        printf("check2:%10lld.%9lld\n", (long long)(r2/1e9), r2 - (long long)((long long)(r2/1e9)*1e9));

        long long diff = r2 - r1;
        printf("diff:%10lld.%9lld\n", (long long)(diff/1e9), diff - (long long)((long long)(diff/1e9)*1e9));

        long long offset = r1 - timespec_to_ns(boottime);

        //// Convert to high-resolution time points
        //auto real_time_point = timespec_to_timepoint(realtime);
        //auto boot_time_point = timespec_to_timepoint(boottime);

        //// Output results
        //auto real_time_t = std::chrono::system_clock::to_time_t(real_time_point);
        //auto boot_time_t = std::chrono::system_clock::to_time_t(boot_time_point);

        //std::cout << "Current time: " << std::ctime(&real_time_t);
        //std::cout << "Boot time: " << std::ctime(&boot_time_t);

    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
