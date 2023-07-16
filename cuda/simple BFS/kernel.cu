
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/sort.h>
#include<thrust/iterator/discard_iterator.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<vector>
#include <stdio.h>
#include <fstream>
#include <sstream>




__global__
void BFSUtil(int csr_ptr[], int edges[], int levelOrder[], int* level, int* visited, int* flag,int *nodes) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < *nodes) {
		if (levelOrder[tid] == *level)
		{
			for (int i = csr_ptr[tid]; i < csr_ptr[tid + 1]; i++) {
				if (visited[edges[i]] == 0) {
					levelOrder[edges[i]] = *level + 1;
					visited[edges[i]] = 1;
					*flag = 1;
				}
			}
		}
	}

}

void BFS(std::vector<int> &h_csr_ptr, std::vector<int> &h_edges) {
	int h_nodes = h_csr_ptr.size() - 1;
	std::vector<int> h_levelOrder(h_nodes, -1);
	std::vector<int> h_visited(h_nodes, 0);
	int src = 0;
	int h_level = 0;
	int h_flag = 1;
	h_visited[src] = 1;
	h_levelOrder[src] = 0;

	//copying to device
	thrust::device_vector<int> csr_ptr = h_csr_ptr;
	thrust::device_vector<int> edges = h_edges;
	thrust::device_vector<int> levelOrder = h_levelOrder;
	thrust::device_vector<int> visited = h_visited;
	int* flag = nullptr;
	int* level = nullptr;
	int* nodes = nullptr;

	cudaMalloc((void**)&flag, sizeof(int));
	cudaMalloc((void**)&level, sizeof(int));
	cudaMalloc((void**)&nodes, sizeof(int));
	//copy h_nodes to nodes
	cudaMemcpy(nodes, &h_nodes, sizeof(int), cudaMemcpyHostToDevice);


	int threadsPerBlock = 256;
	int blocksPerGrid = (h_nodes + threadsPerBlock - 1) / threadsPerBlock;

	while (h_flag) {

		h_flag = 0;
		cudaMemcpy(flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(level, &h_level, sizeof(int), cudaMemcpyHostToDevice);
		BFSUtil << <blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(csr_ptr.data()), thrust::raw_pointer_cast(edges.data()), thrust::raw_pointer_cast(levelOrder.data()), level, thrust::raw_pointer_cast(visited.data()), flag, nodes);
		cudaDeviceSynchronize();
		cudaMemcpy(&h_flag, flag, sizeof(int), cudaMemcpyDeviceToHost);
		h_level++;

	}

	//printing level order
	thrust::copy(levelOrder.begin(), levelOrder.end(), std::ostream_iterator<int>(std::cout, " \n"));
	std::ofstream outFile("levelOrder.txt");
	for (auto i = levelOrder.begin(); i != levelOrder.end(); i++) {
		outFile << *i << "\n";
	}
	outFile.close();
}



std::pair<std::vector<int>,std::vector<int>> EdgeListToCSR(thrust::host_vector<int>& u, thrust::host_vector<int>& v)
{

	thrust::device_vector<int> d_u = u;
	thrust::device_vector<int> d_v = v;

	thrust::sort_by_key(d_u.begin(), d_u.end(), d_v.begin());


	thrust::device_vector<int> ones(d_u.size(), 1);
	thrust::device_vector<int> csr_ptr(d_u.size());

	auto end = thrust::reduce_by_key(d_u.begin(), d_u.end(), ones.begin(), thrust::make_discard_iterator(), csr_ptr.begin());

	int newSize = end.second - csr_ptr.begin() + 1;
	csr_ptr.resize(newSize);

	thrust::exclusive_scan(csr_ptr.begin(), csr_ptr.end(), csr_ptr.begin());

	std::vector<int> h_csr_ptr; 
	std::vector<int> h_v; 
	//copying to host
	thrust::copy(csr_ptr.begin(), csr_ptr.end(), std::back_inserter(h_csr_ptr));
	thrust::copy(d_v.begin(), d_v.end(), std::back_inserter(h_v));
	return std::make_pair(h_csr_ptr, h_v);
}

// ... Rest of your code (BFSUtil, BFS, EdgeListToCSR functions)

int main(void) {
	std::vector<int> u;
	std::vector<int> v;
	// open the file
	std::ifstream file("random_graph.txt");

	// check if the file is open
	if (!file.is_open()) {
		std::cerr << "could not open the file - 'random_graph.txt'" << std::endl;
		return EXIT_FAILURE;
	}

	// read data from the file
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream ss(line);
		int node1, node2;
		char comma;

		ss >> node1 >> comma >> node2;

		u.push_back(node1);
		v.push_back(node2);
	}

	// close the file
	file.close();

	thrust::host_vector<int> h_u = u;
	thrust::host_vector<int> h_v = v;
	std::pair<std::vector<int>, std::vector<int>> csr_edges = EdgeListToCSR(h_u, h_v);
	BFS(csr_edges.first, csr_edges.second);

	return 0;
}