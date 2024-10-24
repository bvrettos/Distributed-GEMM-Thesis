#ifndef ENUMS_HPP
#define ENUMS_HPP

enum class MatrixLayout {
    RowMajor,
    ColumnMajor
};

enum class MemoryLocation {
    Host,
    Device
};

enum class TileType {
    MatrixTile,
    Workspace
};

enum class DistributionStrategy {
    BlockCyclic,
    BlockSequential
};


#endif