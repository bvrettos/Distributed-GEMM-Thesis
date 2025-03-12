#include <Tile.hpp>

/* Create empty tile */
template <typename scalar_t>
Tile<scalar_t>::Tile() :
    rows(0),
    columns(0),
    data(nullptr),
    location(MemoryLocation::Host),
    layout(MatrixLayout::ColumnMajor),
    ld(0),
    allocated(false)
{

}

/* Create tile from start - use existing memory buffer */
template <typename scalar_t>
Tile<scalar_t>::Tile(int64_t rows, int64_t columns, scalar_t* data, int64_t ld, MemoryLocation location, MatrixLayout layout) :
    rows(rows),
    columns(columns),
    data(data),
    location(location),
    ld(ld),
    layout(layout),
    allocated(false)
{
    assert(rows > 0);
    assert(columns > 0);
    assert(data != nullptr);
    assert(((layout == MatrixLayout::ColumnMajor) && ld >= rows) || ((layout == MatrixLayout::RowMajor) && ld >= columns));
}

/* Create a tile with specific dimensions and location - No Input Data */
template <typename scalar_t>
Tile<scalar_t>::Tile(int64_t rows, int64_t columns, MemoryLocation location, MatrixLayout layout) :
    rows(rows),
    columns(columns),
    location(location),
    layout(layout),
    allocated(false)
{
    assert(rows > 0);
    assert(columns > 0);

    if (layout == MatrixLayout::ColumnMajor)
        ld = rows;
    else
        ld = columns;
}

/* Create a tile based on existing tile - use existing memory buffer */
template <typename scalar_t>
Tile<scalar_t>::Tile(Tile<scalar_t> sourceTile, scalar_t* data, int64_t ld) :
    rows(sourceTile.rows),
    columns(sourceTile.columns),
    ld(ld),
    location(sourceTile.location),
    layout(sourceTile.layout)
{
    assert(rows > 0);
    assert(columns > 0);
    assert(((layout == MatrixLayout::ColumnMajor) && ld >= rows) || ((layout == MatrixLayout::RowMajor) && ld >= columns));

    /* Memory already allocated - free it first */
    if (data != nullptr) {
        if (location == MemoryLocation::Host)
            free(this->data);
        else
            cudaFree(this->data);
    }

    /* Change pointers to actual data */
    this->data = data;
}

/* Memory Utilities */

template <typename scalar_t>
void Tile<scalar_t>::allocateMemory()
{
    /* Allocate data on pointer*/
    if (location == MemoryLocation::Host)
        cudaMallocHost((void**)&data, sizeof(scalar_t) * rows * columns, cudaMemAttachHost);
    else
        cudaMalloc((void**)&data, sizeof(scalar_t) * rows * columns);
}

template <typename scalar_t>
bool Tile<scalar_t>::isAllocated() { return this->allocated; }
    
/* Class Getters */
template <typename scalar_t>
int64_t Tile<scalar_t>::getRows() { return this->rows; }

template <typename scalar_t>
int64_t Tile<scalar_t>::getColumns() { return this->columns; }

template <typename scalar_t>
int64_t Tile<scalar_t>::getStride() { return this->ld; }

template <typename scalar_t>
MatrixLayout Tile<scalar_t>::getLayout() { return this->layout; }

template <typename scalar_t>
scalar_t* Tile<scalar_t>::getDataPointer() { return this->data; }

template <typename scalar_t>
MemoryLocation Tile<scalar_t>::getLocation() { return this->location; }

template <typename scalar_t>
void Tile<scalar_t>::generateRandomValues(int deviceLoc)
{
    generateMatrixGPU(this->data, this->rows, this->columns, deviceLoc);
}

template <typename scalar_t>
void Tile<scalar_t>::printTile(int rank)
{
    printMatrix(this->data, this->rows, this->columns, rank, this->layout);

    return;
}

template <typename scalar_t>
void Tile<scalar_t>::writeTile(int tileRow, int tileColumn)
{
    std::string filename= "Tile(" + std::to_string(tileRow) + ", " + std::to_string(tileColumn) + ").txt";
    writeMatrix(this->data, this->rows, this->columns, filename, this->layout);
}

/* Class Instatiations */
template class Tile<double>;
template class Tile<float>;