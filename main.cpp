#include <iostream>
#include <vector>
#include <CL/opencl.hpp>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

int main(int argc, char* argv[]) {
    
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "Platform: " << platforms.front().getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    // Select the platform.
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    std::cout << "Device: " << devices.front().getInfo<CL_DEVICE_NAME>()
              << std::endl;

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices.front());
}