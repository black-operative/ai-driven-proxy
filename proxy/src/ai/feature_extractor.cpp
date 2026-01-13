#include <cstring>

#include "ai/feature_extractor.h"

Feature_Vector Feature_Extractor::extract(
    const char* data, 
    size_t len
) {
    auto now = steady_clock::now();
    uint64_t inter_arrival = 0;

    if (request_count > 0) {
        inter_arrival = duration_cast<microseconds>(
            now - last_request
        ).count();
    }

    last_request = now;
    request_count++;

    Feature_Vector f_vec {};
    f_vec.payload_size     = len;
    f_vec.request_count    = request_count;
    f_vec.inter_arrival_us = inter_arrival;
    f_vec.header_size      = strstr(data, "\r\n\r\n") ? 
                             (strstr(data, "\r\n") - data) : 0;

    return f_vec;
}
