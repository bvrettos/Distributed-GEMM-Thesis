#include <DistributedMatrix.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <validation.hpp>

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);
    
    int result = Catch::Session().run(argc, argv);

    MPI_Finalize();
    return result;
}

TEST_CASE("Testing Empty Tile Constructor") {
    Tile<double> tile;
    REQUIRE(tile.getRows() == 0);
    REQUIRE(tile.getColumns() == 0);
    REQUIRE(tile.getLayout() == MatrixLayout::ColumnMajor);
}

TEST_CASE("Testing Constructor from array of data - LAPACK Style") {
    int rows = 600;
    int columns = 512;
    double* a = (double*) malloc(sizeof(double) * rows * columns);
    for (int i = 0; i < rows*columns; i++)
        a[i] = i;

    Tile<double> tile(rows, columns, a, rows, MemoryLocation::Host, MatrixLayout::ColumnMajor);

    REQUIRE(tile.getRows() == rows);
    REQUIRE(tile.getColumns() == columns);
    REQUIRE(tile.getLayout() == MatrixLayout::ColumnMajor);
}

