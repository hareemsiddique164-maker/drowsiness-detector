cmake_minimum_required(VERSION 3.16...4.3)
project(DrowsinessDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Path to your OpenCV build folder
set(OpenCV_DIR "C:/opencv/build")

find_package(OpenCV REQUIRED)

add_executable(DrowsinessDetector main.cpp)

target_include_directories(DrowsinessDetector PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DrowsinessDetector PRIVATE ${OpenCV_LIBS})

# Copy OpenCV DLLs next to the executable so it can run on Windows
add_custom_command(TARGET DrowsinessDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:DrowsinessDetector>
        $<TARGET_FILE_DIR:DrowsinessDetector>
    COMMAND_EXPAND_LISTS
)

# Copy Haar cascade XML files next to the executable
set(HAAR_DIR "C:/opencv/build/etc/haarcascades")
add_custom_command(TARGET DrowsinessDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HAAR_DIR}/haarcascade_frontalface_default.xml"
        "${HAAR_DIR}/haarcascade_eye.xml"
        $<TARGET_FILE_DIR:DrowsinessDetector>
)
