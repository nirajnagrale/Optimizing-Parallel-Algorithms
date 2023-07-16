#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<stdio.h>
#define INF 1e9;
using namespace std;

struct bfsTree {
    int* nodelevel;
    int* parent;
};



__global__
void bfsUtil(int dptrVertices[], int dptrEdges[], int* dn,
    int* dm, int dlevelOrder[], int* dptrLevel, int* dptrVisited, int* dptrFlag, int* dptrParent)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (dlevelOrder[id] == *dptrLevel)
    {
        int startingidx;
        int endingidx;
        if (id != *dn - 1)
        {
            startingidx = dptrVertices[id];
            endingidx = dptrVertices[id + 1] - 1;
        }
        else
        {
            startingidx = dptrVertices[id];
            endingidx = *dm - 1;
        }
        for (int i = startingidx; i <= endingidx; i++)
        {
            if (!dptrVisited[dptrEdges[i]])
            {
                atomicExch(&dptrParent[dptrEdges[i]], id);
                atomicExch(&dlevelOrder[dptrEdges[i]], *dptrLevel + 1);
                atomicExch(&dptrVisited[dptrEdges[i]], 1);
                atomicExch(dptrFlag, 1);
            }
        }
    }
    __syncthreads();
}

struct bfsTree bfs(int vertices[], int edges[], int n, int m, int src)
{
    int* levelOrder = new int[n];
    int level = 0;
    int* parent = new int[n];
    parent[src] = src;
    int flag = 1;
    int* dv;
    int* de;
    int* dn;
    int* dm;
    int* dparent;
    int* visited = new int[n];

    for (int i = 0; i < n; i++)
    {
        visited[i] = 0;
        levelOrder[i] = INF;
    }
    visited[src] = 1;
    levelOrder[src] = 0;

    //allocating mem in gpu
    cudaMalloc(&dv, n * sizeof(int));
    cudaMalloc(&de, m * sizeof(int));
    cudaMalloc(&dn, sizeof(int));
    cudaMalloc(&dm, sizeof(int));
    cudaMalloc(&dparent, sizeof(int) * n);
    // copying data into device;
    cudaMemcpy(dv, vertices, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(de, edges, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dm, &m, sizeof(int), cudaMemcpyHostToDevice);
    while (flag)
    {
        flag = 0;
        int* dlevelOrder;
        int* dl;
        int* di;
        int* dvisited;
        int* dflag;

        //allocating mem in gpu
        cudaMalloc(&dvisited, sizeof(int) * n);
        cudaMalloc(&dlevelOrder, sizeof(int) * n);
        cudaMalloc(&dl, sizeof(int));
        cudaMalloc(&di, sizeof(int));
        cudaMalloc(&dflag, sizeof(int));

        // copying data into device;
        cudaMemcpy(dvisited, visited, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dlevelOrder, levelOrder, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dl, &level, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dflag, &flag, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dparent, parent, sizeof(int) * n, cudaMemcpyHostToDevice);


        // 1 block n threads;
        //int dptrVertices[], int dptrEdges[], int* dn,
        //int* dm, int dlevelOrder[], int* dptrLevel, int* dptrVisited, int* dptrFlag, int* dptrParent
        //bfsUtil << <1, n >> > (dv, de, dn, dm, dlevelOrder, dl, dvisited, dflag, dparent);
        bfsUtil<<<1,n>>>(dv, de, dn, dm, dlevelOrder, dl, dvisited, dflag, dparent);
        // copying data back to host;
        cudaMemcpy(levelOrder, dlevelOrder, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&level, dl, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(visited, dvisited, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag, dflag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, dparent, sizeof(int) * n, cudaMemcpyDeviceToHost);

        //freeing data from gpu;
        cudaFree(dlevelOrder);
        cudaFree(dl);
        level++;

    }


    struct bfsTree bfsT;
    bfsT.nodelevel = new int[n];
    bfsT.parent = new int[n];

    for (int i = 0; i < n; i++)
    {
        bfsT.nodelevel[i] = levelOrder[i];
        bfsT.parent[i] = parent[i];
        // std::cout << visited[i] << " ";
        std::cout << "node " << i << " has level " << levelOrder[i] << " ";
        std::cout << std::endl;
    }
    return bfsT;
    //freeing data from gpu;
    cudaFree(dv);
    cudaFree(de);
    cudaFree(dn);
    cudaFree(dm);
    cudaFree(dparent);
}
__device__ int dsafe[5] = { 0,0,0,0,0 };
__global__
void cutVertUtil(int *dVertices,int *dEdges,int * dn,int * dm,int *droot,int * dParent,int * dvis,int * dLevelOrder,int * dptrsafe)
{
    
    int nodeId = blockIdx.x;
    /*if(nodeId == 0 && threadIdx.x==0)
    {
        for (int i = 0; i < *dn; i++)
        {
            printf("%d ", dVertices[i]);
        }
        printf("\n");
        for (int i = 0; i < *dm; i++)
        {
            printf("%d ", dEdges[i]);
        }
        printf("\n");

        for (int i = 0; i < *dn; i++)
        {
            printf("%d ", dParent[i]);
        }
        printf("\n");
        for (int i = 0; i < *dn; i++)
        {
            printf("%d ", dLevelOrder[i]);
        }
        printf("\n");

    }*/
    //printf("%d %d %d ", *dn, *dm, *droot);
   /* if (nodeId == 4 && threadIdx.x == 0)
    {
        for (int i = 0; i < *dn; i++)
        {
            printf("%d is visited\n", dvis[i]);
        }
    }*/
    if (nodeId != *droot && dParent[nodeId] != *droot)
    {
        __shared__ int level ;
        __shared__ int newLevelOrder[5];
        for (int i = 0; i < 5; i++)
        {
            newLevelOrder[i] = 10000;
        }
        __shared__ int flag;
        level = 0;
        flag = 1;
        newLevelOrder[nodeId] = 0; 
        dvis[dParent[nodeId]] = 1;
      /*  if (nodeId == 4)
        {
            printf("%d", dvis[dParent[nodeId]]);
        }*/
        __syncthreads();
       
        while (flag)
        {
            flag = 0;
            __syncthreads();
            int startingIdx;
            int endingIdx;
            if (newLevelOrder[threadIdx.x] == level)
            {
                if (threadIdx.x != *dn - 1)
                {
                    startingIdx = dVertices[threadIdx.x];
                    endingIdx = dVertices[threadIdx.x + 1] - 1;
                }
                else
                {
                    startingIdx = dVertices[threadIdx.x];
                    endingIdx = *dm - 1;
                }
                for (int i = startingIdx; i <= endingIdx; i++)
                {
                    if (dvis[dEdges[i]]==0)
                    {
                        
                        atomicExch(&newLevelOrder[dEdges[i]], level + 1);
                        atomicExch(&dvis[i], 1);
                        atomicExch(&flag, 1);
                        /*if (nodeId == 4)
                        {
                            printf("%d %d\n ",i, dEdges[i]);
                        }*/
                        if (dLevelOrder[dEdges[i]]<=dLevelOrder[dParent[nodeId]])
                        {
                            atomicExch(&dsafe[nodeId], 1);
                            //printf("safe %d\n",nodeId);
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == nodeId)
            {
                level++;
            }
           

        }


    }
    __syncthreads();
}

void rootCutVert(int vertices[], int edges[], int LevelOrder[], int parent[], int n, int m, int src,int safe[])
{
    int* levelOrder = new int[n];
    int level = 0;
    int flag = 1;
    int* dv;
    int* de;
    int* dn;
    int* dm;
    int root = src;
    int* visited = new int[n];
    int first_child = edges[src];
    src = first_child;
    for (int i = 0; i < n; i++)
    {
        visited[i] = 0;
        levelOrder[i] = INF;
    }
    levelOrder[src] = 0;
    visited[src] = 1;


    //allocating mem in gpu
    cudaMalloc(&dv, n * sizeof(int));
    cudaMalloc(&de, m * sizeof(int));
    cudaMalloc(&dn, sizeof(int));
    cudaMalloc(&dm, sizeof(int));

    // copying data into device;
    cudaMemcpy(dv, vertices, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(de, edges, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dm, &m, sizeof(int), cudaMemcpyHostToDevice);
    while (flag)
    {
        flag = 0;
        int* dlevelOrder;
        int* dl;
        int* di;
        int* dvisited;
        int* dflag;
        int* dparent;
        //allocating mem in gpu
        cudaMalloc(&dvisited, sizeof(int) * n);
        cudaMalloc(&dlevelOrder, sizeof(int) * n);
        cudaMalloc(&dl, sizeof(int));
        cudaMalloc(&di, sizeof(int));
        cudaMalloc(&dflag, sizeof(int));
        cudaMalloc(&dparent, sizeof(int) * n);

        // copying data into device;
        cudaMemcpy(dvisited, visited, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dlevelOrder, levelOrder, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dl, &level, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dflag, &flag, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dparent, parent, sizeof(int), cudaMemcpyHostToDevice);

        // 1 block n threads;
        bfsUtil << <1, n >> > (dv, de, dn, dm, dlevelOrder, dl, dvisited, dflag, dparent);

        // copying data back to host;
        cudaMemcpy(levelOrder, dlevelOrder, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&level, dl, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(visited, dvisited, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag, dflag, sizeof(int), cudaMemcpyDeviceToHost);


        //freeing data from gpu;
        cudaFree(dlevelOrder);
        cudaFree(dl);
        level++;

    }
    //freeing data from gpu;
    cudaFree(dv);
    cudaFree(de);
    cudaFree(dn);
    cudaFree(dm);
    for (int i = 0; i < n; i++)
    {
        if (levelOrder[i] != 1e9 && parent[i] == root)
        {
            safe[i] = 1;
        }
    }
}

void cutVert(int vertices[], int edges[], int LevelOrder[], int parent[], int n, int m, int src)
{
    //int *dVertices,int *dEdges,int * dn,int * dm,int *droot,int * dParent,int * dsafe,int * dvisited[]
   
    int* visited = new int[n];
    int* safe = new int[n];
    int* cutvertex = new int[n];
    for (int i = 0; i < n; i++)
    {
        visited[i] = 0;
        safe[i] = 0;
        cutvertex[i] = 0;
    }
    
   
    int* dVertices;
    int* dEdges;
    int* dn;
    int* dm;
    int* droot;
    int* dParent;
    int* dvis;
    int* dLevelOrder;
    int* dptrsafe;
  
    
    cudaMalloc(&dVertices, sizeof(int) * n);
    cudaMalloc(&dEdges, sizeof(int) * m);
    cudaMalloc(&dn, sizeof(int));
    cudaMalloc(&dm, sizeof(int));
    cudaMalloc(&droot, sizeof(int));
    cudaMalloc(&dParent, sizeof(int) * n);
    cudaMalloc(&dvis, sizeof(int) * n);
   
    cudaMalloc(&dLevelOrder, sizeof(int) * n);
    cudaMalloc(&dptrsafe, sizeof(int) * n);

   
    cudaMemcpy(dVertices, vertices, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dEdges, edges, sizeof(int) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dm, &m, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(droot, &src, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dParent, parent, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dvis, visited, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dLevelOrder, LevelOrder, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dptrsafe, safe, sizeof(int) * n, cudaMemcpyHostToDevice);
    //cutVertUtil << <n, n >> > (dVertices,dEdges,dn,dm,droot,dParent,dvisited,dLevelOrder);
    cutVertUtil << <n, n >> > (dVertices,dEdges,dn,dm,droot,dParent,dvis,dLevelOrder,dptrsafe);
    cudaDeviceSynchronize();

    //cudaMemcpy(safe, dptrsafe, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaMemcpyFromSymbol(safe, dsafe, sizeof(int) * n);
    
   

    //freeing variables
 
    cudaFree(dEdges);
    cudaFree(dn);
    cudaFree(dm);
    cudaFree(droot);
    cudaFree(dParent);
    cudaFree(dvis);
    cudaFree(dLevelOrder);
    cudaFree(dptrsafe);

    //for root;
    rootCutVert(vertices, edges, LevelOrder, parent, n, m, src,safe);
    for (int i = 0; i < n; i++)
    {
        if (safe[i] == 0)
        {
            cutvertex[parent[i]] = 1;
        }
    }
    for (int i = 0; i < n; i++)
    {
        cout << "node "<<i<<"  is cutvertex(1)/notcutvertex(0) "<<cutvertex[i] << endl;
    }

}

int main(void)
{
    int edges[] = { 1,2,3,0,2,0,1,0,4,3 };
    int vertices[] = { 0,3,5,7,9 };
    int src = 2;
    int n = sizeof(vertices) / sizeof(vertices[0]);
    int m = sizeof(edges) / sizeof(edges[0]);

    struct bfsTree b = bfs(vertices, edges, n, m, src);
    int* levelOrder = new int[n];
    int* parent = new int[n];
    for (int i = 0; i < n; i++)
    {
        levelOrder[i] = b.nodelevel[i];
        parent[i] = b.parent[i];

        std::cout << "node " << i << " has level " << b.nodelevel[i] << " & parent " << b.parent[i] << std::endl;
    }
    cutVert(vertices, edges, levelOrder, parent, n, m, src);
    return 0;
}
