#pragma once

#include <chrono>

#include "common.h"

using namespace std::chrono;

class Feature_Extractor {
    private:
        steady_clock::time_point last_request;
        uint32_t                 request_count;

    public:
        Feature_Vector extract(const char* data, size_t len);
};