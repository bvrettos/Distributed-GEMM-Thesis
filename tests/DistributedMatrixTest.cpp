#include <DistributedMatrix.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <validation.hpp>
#include "pblasDecomposition.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int result = Catch::Session().run(argc, argv);

    MPI_Finalize();
    return result;
}

TEST_CASE("Blank Constructor") {
    DistributedMatrix<double> matrix();

    REQUIRE(matrix.getRows() == 0);
}



TEST_CASE("Block Cyclic Decomposition Test") {

    /* Need to test if 2D Block Cyclic Decomp is correct. Use PBLAS Decomp (we know it works) for validation */
    

}