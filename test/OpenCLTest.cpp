#include <gtest/gtest.h>
#include "ConwayGame.h"  // Your OpenCL subclass

class OpenCLTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initial grid (3x3)
        initialGrid = {
            0, 1, 0,
            1, 1, 0,
            0, 0, 1
        };

        rows = 3;
        cols = 3;
    }

    int rows;
    int cols;
    std::vector<int> initialGrid;
};

TEST_F(OpenCLTest, InitializeOpenCL_DoesNotThrow) {
    EXPECT_NO_THROW({
        ConwayGameOpenCL game(rows, cols, initialGrid);
    });
}

TEST_F(OpenCLTest, Update_MatchesCPUUpdate) {
    ConwayGameOpenCL clGame(rows, cols, initialGrid);
    ConwayGameOpenCL cpuGame(rows, cols, initialGrid);

    cpuGame.update();
    clGame.update();

    std::vector<int> cpuGrid = cpuGame.getGrid();
    std::vector<int> clGrid = clGame.getGrid();

    ASSERT_EQ(cpuGrid.size(), clGrid.size());

    for (size_t i = 0; i < cpuGrid.size(); ++i) {
        EXPECT_EQ(clGrid[i], cpuGrid[i]) << "Mismatch at index " << i;
    }
}
