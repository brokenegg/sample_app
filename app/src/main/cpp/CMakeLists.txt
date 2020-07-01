cmake_minimum_required(VERSION 3.4.1)

add_library(tokenize
            SHARED
            tokenize.cpp
        )

target_link_libraries(tokenize
        ${CMAKE_SOURCE_DIR}/libsentencepiece.a
      android
      log)
