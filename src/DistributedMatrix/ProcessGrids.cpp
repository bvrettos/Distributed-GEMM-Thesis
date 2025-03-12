#include <DistributedMatrix.hpp>

void calculateCubicGrid(int size, int& dRow, int& dCol, int& dStack)
{
    int Px = std::cbrt(size);   // Start with the cube root of the total for the first size
    
    std::vector<int> dims;
    // Iterate over x, lowering the number to find the most cubic dims
    for (int i = Px; i >= 1; --i) {
        if ((size % i) == 0) {
            int remainder = size / i;
            int Py = std::sqrt(remainder);

            for (int j = Py; j >= 1; --j) {
                if ((remainder % j) == 0) {
                    int z = remainder / j;
                    std::vector<int> dims = {i, j, z};
                    std::sort(dims.begin(), dims.end(), std::greater<int>());
                }
            }
        }
    }

    dRow = dims[0];
    dCol = dims[1];
    dStack = dims[2];

    #ifdef DEBUG
        if (rank == 0) printf("Cubic Dimensions: %d x %d x %d\n", dRow, dCol, dStack);
    #endif

    return;
}

template <typename scalar_t>
void DistributedMatrix<scalar_t>::calculate2DProcessGrid()
{
    int Px = std::sqrt(size);
    int Py = Px;
    /* If less than 4 devices */
    if (Px == 0) {
        Py = size;
        Px = 1;
    }
    /* If more than 4 devices, find the most square decomposition */
    int counter;
    for (counter = Px; counter > 0; --counter) 
        if (size % counter == 0) break;
    
    if (counter==0) {
        Px = size;
        Py = 1;
    }
    else {
        Px = counter;
        Py = size/counter;
    }
    this->dRow = Py;
    this->dCol = Px;

    this->processRow = rank/dRow;
    this->processColumn = rank%dRow;
}

template class DistributedMatrix<double>;
template class DistributedMatrix<float>;