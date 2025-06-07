#include <gtest/gtest.h>
#include "ConwayGame.h"

class SequentialTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Example grid (3x3)
        // 0 1 0
        // 1 1 0
        // 0 0 1
        std::vector<int> vecGrid = {
            0, 1, 0,
            1, 1, 0,
            0, 0, 1
        };
        game = new ConwayGameSequential(3, 3, vecGrid);
    }

    void TearDown() override {
        delete game;
    }

    ConwayGameSequential* game;
};

TEST_F(SequentialTest, GetIndex_CorrectIndex) {
    EXPECT_EQ(game->getIndex(0, 0), 0);
    EXPECT_EQ(game->getIndex(0, 2), 2);
    EXPECT_EQ(game->getIndex(1, 0), 3);
    EXPECT_EQ(game->getIndex(2, 2), 8);
}

TEST_F(SequentialTest, GetValue_WithinBounds) {
    EXPECT_EQ(game->getValue(0, 1), 1);
    EXPECT_EQ(game->getValue(1, 1), 1);
    EXPECT_EQ(game->getValue(2, 2), 1);
    EXPECT_EQ(game->getValue(0, 0), 0);
}

TEST_F(SequentialTest, GetValue_OutOfBounds) {
    EXPECT_EQ(game->getValue(-1, 0), 0);
    EXPECT_EQ(game->getValue(0, -1), 0);
    EXPECT_EQ(game->getValue(3, 0), 0);
    EXPECT_EQ(game->getValue(0, 3), 0);
}

TEST_F(SequentialTest, CountNeighbors_CorrectCount) {
    // Position (1,1) has neighbors: (0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)
    // Values: 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1 = 3
    EXPECT_EQ(game->countNeighbors(1,1), 3);

    // Position (0,0) neighbors: out-of-bounds + out-of-bounds + out-of-bounds + out-of-bounds + 1 + out-of-bounds + 1 + 1 = 3
    EXPECT_EQ(game->countNeighbors(0,0), 3);

    // Position (2,2) neighbors: 1 + 0 + out-of-bounds + 0 + out-of-bounds + out-of-bounds + out-of-bounds + out-of-bounds = 1
    EXPECT_EQ(game->countNeighbors(2,2), 1);
}

TEST_F(SequentialTest, SetValue_WithinBounds) {
    game->setValue(0, 0, 1);
    EXPECT_EQ(game->getValue(0, 0), 1);

    game->setValue(2, 2, 0);
    EXPECT_EQ(game->getValue(2, 2), 0);
}

TEST_F(SequentialTest, SetValue_OutOfBounds_DoesNothing) {
    game->setValue(-1, 0, 1);
    EXPECT_EQ(game->getValue(-1, 0), 0);

    game->setValue(3, 3, 1);
    EXPECT_EQ(game->getValue(3, 3), 0);
}

TEST_F(SequentialTest, Update_ChangesGridCorrectly) {
    /*
    Initial grid (3x3):
    0 1 0
    1 1 0
    0 0 1

    Expected next grid after update (apply Game of Life rules):
    - Cell (0,0): neighbors=3 (alive)
    - Cell (0,1): neighbors=3 (alive)
    - Cell (0,2): neighbors=2 (dead -> stays dead)
    - Cell (1,0): neighbors=2 (alive -> stays alive)
    - Cell (1,1): neighbors=3 (alive)
    - Cell (1,2): neighbors=3 (alive)
    - Cell (2,0): neighbors=2 (dead -> stays dead)
    - Cell (2,1): neighbors=3 (alive)
    - Cell (2,2): neighbors=1 (alive -> dies)

    Resulting grid:
    1 1 0
    1 1 1
    0 1 0
    */

    game->update();

    std::vector<int> expectedGrid = {
        1, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    std::vector<int> updatedGrid = game->getGrid();
    EXPECT_EQ(updatedGrid.size(), expectedGrid.size());

    for (size_t i = 0; i < expectedGrid.size(); ++i) {
        EXPECT_EQ(updatedGrid[i], expectedGrid[i]) << "Mismatch at index " << i;
    }
}