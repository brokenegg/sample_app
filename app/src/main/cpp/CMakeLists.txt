cmake_minimum_required(VERSION 3.4.1)

add_library(translator
            SHARED
            translator.cpp
        )

include_directories(${CMAKE_SOURCE_DIR}/inc)

target_link_libraries(translator
    ${CMAKE_SOURCE_DIR}/libsentencepiece.a
    ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so
    android
    log)
