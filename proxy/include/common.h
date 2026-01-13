#pragma once

#include <cstdint>

enum class TRAFFIC_TYPE : uint8_t {
    BENIGN = 0,
    BOT    = 1,
    ATTACK = 2
};

struct Feature_Vector {
    uint32_t payload_size;
    uint32_t header_size;
    uint32_t request_count;
    uint64_t inter_arrival_us;   
};

struct AI_Result {
    TRAFFIC_TYPE type;
    float        confidence;
};
