#include <cuda_runtime.h>

#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return 1; }
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        char bus_id[32]{};
        cudaGetDeviceProperties(&props, device);
        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device);
        int managed = 0;
        int concurrent_managed = 0;
        cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device);
        cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, device);
        std::printf("cuda=%d bus=%s name=%s sm=%d%d managed=%d concurrent_managed=%d\n", device,
                    bus_id, props.name, props.major, props.minor, managed, concurrent_managed);
    }
    for (int source = 0; source < count; ++source) {
        for (int destination = 0; destination < count; ++destination) {
            if (source == destination) { continue; }
            int access = 0;
            cudaDeviceCanAccessPeer(&access, source, destination);
            std::printf("peer %d->%d=%d\n", source, destination, access);
        }
    }
    return 0;
}
