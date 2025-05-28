#include "ConwayGame.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define BLOCK_SIZE 32

ConwayGameOpenCL::ConwayGameOpenCL(int rows, int cols) : ConwayGame(rows, cols) {
    initializeOpenCL();
}
ConwayGameOpenCL::ConwayGameOpenCL(int rows, int cols, std::vector<int> vectorGrid) 
    : ConwayGame(rows, cols, vectorGrid) {
    initializeOpenCL();
}
ConwayGameOpenCL::ConwayGameOpenCL(std::vector<std::vector<int>> matrixGrid) 
    : ConwayGame(matrixGrid) {
    initializeOpenCL();
}

void ConwayGameOpenCL::initializeOpenCL() {
    std::ifstream kernelFile(KERNEL_PATH);
    if (!kernelFile.is_open())
        throw std::runtime_error("Failed to open ConwayKernel.cl");

    std::stringstream kernelSource;
    kernelSource << kernelFile.rdbuf();

    const int bufferSize = rows * cols * sizeof(int);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.empty()) throw std::runtime_error("No OpenCL devices found.");

    device = devices.front();
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    program = cl::Program(context, kernelSource.str(), true);
    kernel = cl::Kernel(program, "conwayStep");

    deviceGrid = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSize);
    deviceNextGrid = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSize);

    // Make sure nextGrid is resized before usage
    nextGrid.resize(rows * cols);
}


void ConwayGameOpenCL::update() {
    try {
        size_t worldSize = rows * cols * sizeof(int);
        queue.enqueueWriteBuffer(deviceGrid, CL_TRUE, 0, worldSize, grid.data());

        kernel.setArg(0, deviceGrid);
        kernel.setArg(1, deviceNextGrid);
        kernel.setArg(2, rows);
        kernel.setArg(3, cols);

        size_t globalSizeValue = ((rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        cl::NDRange globalSize(globalSizeValue);
        cl::NDRange localSize(BLOCK_SIZE);

        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);
        queue.enqueueReadBuffer(deviceNextGrid, CL_TRUE, 0, worldSize, nextGrid.data());

        std::swap(grid, nextGrid);
    } catch (const std::exception& err) {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
}
