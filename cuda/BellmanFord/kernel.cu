//bellman ford;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
//depth of bellman ford = O(V) in best case  where V is the no. of vertices;


using namespace std;

__global__
void relaxEdges(int* dMat, int* ddist, int* dn)
{
	int blockid = blockIdx.x;
	int threadId = threadIdx.x;
	int element_idx = (*dn) * blockid + threadIdx.x;
	int u = blockid;
	if (dMat[element_idx] != -1000) // implies there is an edges -1000 is a sentinal value implying there is no edge
	{
		atomicMin(&ddist[threadId], ddist[u] + dMat[element_idx]);

	}


}
void BellmanFord(int adjMatrix[], int n, int src)
{
	int* dMat;
	int mat_size = n * n;
	int* dn;
	int* distance = new int[n];
	int* ddist;
	for (int i = 0; i < n; i++)
	{
		distance[i] = 100000;
	}
	distance[src] = 0;
	cudaMalloc(&dMat, sizeof(int) * mat_size);
	cudaMalloc(&dn, sizeof(int));

	cudaMemcpy(dn, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dMat, adjMatrix, sizeof(int) * mat_size, cudaMemcpyHostToDevice);
	for (int i = 0; i < n; i++)
	{
		cudaMalloc(&ddist, sizeof(int) * n);
		cudaMemcpy(ddist, distance, sizeof(int) * n, cudaMemcpyHostToDevice);
		relaxEdges << <n, n >> > (dMat, ddist, dn);
		cudaDeviceSynchronize();
		cudaMemcpy(distance, ddist, sizeof(int) * n, cudaMemcpyDeviceToHost);
	}
	for (int i = 0; i < n; i++)
	{
		cout << distance[i] << " ";
	}
	cout << endl;

}


int main(void)
{
	int n = 5;
	int adjMatrix[] = { 0, -1, 4, -1000, -1000,
						-1000,0,3,2,2,
						-1000,-1000,0,-1000,-1000,
						-1000,1,5,0,-1000,
						-1000,-1000,-1000,-3,0 };//size 5*5;
	int src = 0;
	BellmanFord(adjMatrix, n, src);

	return 0;
}
