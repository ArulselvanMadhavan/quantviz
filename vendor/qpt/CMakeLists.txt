cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(qpt)

set(VERBOSE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(quantcpu_static STATIC)
target_sources(quantcpu_static PRIVATE src/quant_cpu.cpp)
target_include_directories(quantcpu_static PUBLIC include)
target_link_libraries(quantcpu_static "${TORCH_LIBRARIES}")
set_property(TARGET quantcpu_static PROPERTY CXX_STANDARD 14)
set_property(TARGET quantcpu_static PROPERTY POSITION_INDEPENDENT_CODE
ON)

add_library(quantcpu_shared SHARED)
target_sources(quantcpu_shared PRIVATE src/quant_cpu.cpp)
target_include_directories(quantcpu_shared PUBLIC include)
target_link_libraries(quantcpu_shared "${TORCH_LIBRARIES}")
set_property(TARGET quantcpu_shared PROPERTY CXX_STANDARD 14)

# target_include_directories(quantcpu_static PRIVATE
# qtorch/quant/quant_cpu/include)
# target_include_directories(quantcpu_shared PRIVATE
# qtorch/quant/quant_cpu/include)

# add_custom_command(
#         TARGET quantcpu_static  POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy
#                 ${CMAKE_CURRENT_SOURCE_DIR}/qtorch/quant/quant_cpu/include/libquantcpu.h
#                 ${CMAKE_CURRENT_BINARY_DIR}/libquantcpu.h)
