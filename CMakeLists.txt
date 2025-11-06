cmake_minimum_required(VERSION 3.16)

project(myMPITimer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(MPI REQUIRED)
link_libraries(MPI::MPI_CXX)

add_executable(MPI_Timer src/MPI_Timer.cpp)
target_include_directories(MPI_Timer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)