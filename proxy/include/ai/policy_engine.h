#pragma once

#include "common.h"

class Policy_Engine {
    public:
        bool allow(const AI_Result& result);
};