cmake_minimum_required(VERSION 3.15)

project(AI_Proxy VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCES
    src/main.cpp
    src/ai/ai_client.cpp
    src/ai/feature_extractor.cpp
    src/ai/policy_engine.cpp
    src/util/csv_logger.cpp
    src/util/net_utils.cpp
    src/net/proxy_server.cpp
    src/thread/thread_pool.cpp
)

add_executable(
    AI_Proxy
    ${SOURCES}
)

target_include_directories(
    AI_Proxy PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)