cmake_minimum_required(VERSION 3.18)
project(mnist_perceptron LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


# sm_75 = Turing (RTX 20xx)
# sm_86 = Ampere (RTX 30xx)
# sm_89 = Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(mnist_perceptron
    main.cpp
    mnist_reader.cpp
    perceptron10.cu
)

target_include_directories(mnist_perceptron PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(mnist_perceptron PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
