#include "ConwayGame.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define BLOCK_SIZE 32

void ConwayGameOpenCL::initializeOpenCL() {
    // Load OpenCL kernel source code
    std::ifstream kernelFile("kernel/ConwayKernel.cl");
    std::stringstream kernelSource;
    kernelSource << kernelFile.rdbuf();

    const int bufferSize = rows * cols * sizeof(int);
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Device> devices;

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices.front());

    cl::Program program(context, kernelSource.str(), true);

    cl::Kernel kernel(program, "conwayStep");

    // Initialize device buffers
    cl::Buffer deviceGrid(context, CL_MEM_READ_ONLY, bufferSize);
    cl::Buffer deviceNextGrid(context, CL_MEM_WRITE_ONLY, bufferSize);
}

void ConwayGameOpenCL::update() {
    try{
        size_t worldSize = rows * cols * sizeof(int);
        queue.enqueueWriteBuffer(deviceGrid, CL_TRUE, 0, worldSize, grid.data());

        kernel.setArg(0, deviceGrid);
        kernel.setArg(1, deviceNextGrid);
        kernel.setArg(2, rows);
        kernel.setArg(3, cols);

        cl::NDRange globalSize(rows*cols);
        cl::NDRange localSize(BLOCK_SIZE);

        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &event);
        queue.enqueueReadBuffer(deviceNextGrid, CL_TRUE, 0, worldSize, nextGrid.data());

        // Swap grids
        std::swap(grid, nextGrid);
    } catch (const std::exception& err) {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
}