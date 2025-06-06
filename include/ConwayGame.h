#include <vector>
#include <iostream>
#include <CL/opencl.hpp>

class ConwayGame {
public:
    int rows;
    int cols;
    std::vector<int> grid;
    std::vector<int> nextGrid;
    int getIndex(int row, int col);
    int getValue(int row, int col);
    int countNeighbors(int row, int col);
    // Constructors
    ConwayGame(int rows, int cols);
    ConwayGame(int rows, int cols, std::vector<int> vectorGrid);
    ConwayGame(std::vector<std::vector<int>> matrixGrid);

    // Destructor
    ~ConwayGame();

    // Get the grid
    std::vector<int> getGrid();

    // Randomize the grid
    void randomizeGrid();

    // Print the grid
    void printGrid();

    // Set value in the grid
    void setValue(int row, int col, int value);

    virtual void update() = 0; // Pure virtual function for updating the grid
};

class ConwayGameSequential : public ConwayGame {
public:
    ConwayGameSequential(int rows, int cols);
    ConwayGameSequential(int rows, int cols, std::vector<int> vectorGrid);
    ConwayGameSequential(std::vector<std::vector<int>> matrixGrid);
    void update() override;
};

class ConwayGameOpenCL : public ConwayGame {
public:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Buffer deviceGrid;
    cl::Buffer deviceNextGrid;
    ConwayGameOpenCL(int rows, int cols);
    ConwayGameOpenCL(int rows, int cols, std::vector<int> vectorGrid);
    ConwayGameOpenCL(std::vector<std::vector<int>> matrixGrid);

    void initializeOpenCL();
    void update() override;
};

class ConwayGameCuda : public ConwayGame {
public:
    ConwayGameCuda(int rows, int cols) : ConwayGame(rows, cols) {};
    void update() override;
};

//funcion para version CUDA
ConwayGameCuda* createConwayGameCuda(int rows, int cols, bool useIfs = false, int blockSize = 16);