#include "ai/policy_engine.h"

bool Policy_Engine::allow(const AI_Result& result) {
    if (
        result.type == TRAFFIC_TYPE::ATTACK && 
        result.confidence > 0.8f
    ) return false;
    
    return true;
}