#include <DistributedMatrix.hpp>

template <typename scalar_t>
void multiply(scalar_t alpha, DistributedMatrix<scalar_t>& A, DistributedMatrix<scalar_t>& B, DistributedMatrix<scalar_t>& C, scalar_t beta)
{
	/* First Checks 
		- K should be same for A, B
		- Blocking Dimensions should be equal between each other
		- if alpha == 0, just scale C by beta
	*/
	assert(A.getColumns() == B.getRows()); // MxK * KxN => K should be common;

	if ((A.getColumns() == 0) || alpha == 0) {
		/* C = b*C, just scale C */
		scaleMatrix(beta, C);
		return;
	}

	/* Keep Metadata of previous executions to see if executor needs to be reset/updated */
	


	/* Instead of using temporary betas... */
	const scalar_t one = 1.0;

	/* 
	   Generic SUMMA: 
		- Broadcasts can be parallelized on a mutliple streams, but since distribution sets processes next to each other, they will be serialized
		- GEMM calls must be serialized, for parallel GEMM calls, extra memory and reduction is needed 
	*/  

	/* Calculate First C-Tile (+ scale) */


	for (int k = 1; k < A.gridColumns(); k++) {
		Tile<scalar_t>& currentTileA = A.tileMap[processRow][k];
		if (A.tileIsMine(processRow, k)) {
			/* Broadcast A(i,k) to ranks owning block row C(i, :) */
			currentTileA.bcast();
		}

		else {
			/* Receive Tile A from other */
			workspaceA.bcast();
		}

		Tile<scalar_t>& currentTileB = B.tileMap[k][processColumn];
		if (B.tileIsMine(k, processColumn)) {
			/* Broadcast B(k,j) to ranks owning block row C(:, j) */
			currentTileB.bcast();
		}

		else {
			/* Receive Tile B from other */
			workspaceB.bcast();
		}	

		/* Multiply Tiles */
		internalGemm(alpha, workspaceA, workspaceB, workspaceC, one);
	}
	
	return;
}

/* Add Transposes */
template <typename scalar_t>
void internalGemm(scalar_t alpha, char TransA, char TransB, Tile<scalar_t>& A, Tile<scalar_t>& B, Tile<scalar_t>& C, scalar_t beta)
{
	assert(A.getColumns() == B.getRows());
	int64_t m, n, k;
	m = A.getRows();
	n = B.getColumns();
	k = A.getColumns();

	cublasGemm(handle, charToCublasTransOp(TransA), charToCublasTransOp(TransB), m, n, k, &alpha, A.getDataPointer(), A.getStride(), 
		B.getDataPointer(), B.getStride(),&beta, C.getDataPointer(), C.getStride());

	return;
}

