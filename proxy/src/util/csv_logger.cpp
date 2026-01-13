#include <mutex>

using std::lock_guard;

#include "util/csv_logger.h"

CSV_Logger::CSV_Logger() {
    file.open(
        "proxy_metrics.csv",
        std::ios::out | std::ios::app
    );

    if (file.tellp() == 0) {
        file << "timestamp_us,"
             << "client_fd,"
             << "feature_us,"
             << "ai_us,"
             << "policy_us,"
             << "total_us,"
             << "decision,"
             << "traffic_class,"
             << "confidence\n";
    }
}

CSV_Logger& CSV_Logger::instance() {
    static CSV_Logger logger;
    return logger;
}

void CSV_Logger::log(const string& line) {
    lock_guard lock(mtx);
    file << line << "\n";
    file.flush();
}