cmake_minimum_required(VERSION 3.10)

project(main)

add_executable(main main.c ct_util.c ct_network.c ct_backprop.c)

target_link_libraries(main PRIVATE m)