/* Minimal QNN HTP runner - loads .so model, runs on HTP */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "QnnInterface.h"
#include "QnnGraph.h"
#include "QnnContext.h"
#include "QnnBackend.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.so>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* backend_path = "/home/fibo/qnn_sdk_262/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so";
    const char* system_lib = "/home/fibo/qnn_sdk_262/lib/aarch64-ubuntu-gcc9.4/libQnnSystem.so";

    printf("QNN HTP Runner\n");
    printf("  Model: %s\n", model_path);
    printf("  Backend: %s\n", backend_path);

    // Load backend
    void* backend_handle = dlopen(backend_path, RTLD_NOW);
    if (!backend_handle) {
        printf("ERROR: Failed to load backend: %s\n", dlerror());
        return 1;
    }
    printf("Backend loaded\n");

    // Get providers
    typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t***, uint32_t*);
    GetProvidersFn getProviders = (GetProvidersFn)dlsym(backend_handle, "QnnInterface_getProviders");
    if (!getProviders) {
        printf("ERROR: getProviders not found\n");
        return 1;
    }

    const QnnInterface_t** providers = NULL;
    uint32_t num_providers = 0;
    Qnn_ErrorHandle_t err = getProviders(&providers, &num_providers);
    printf("Providers: %d (err=%d)\n", num_providers, err);

    if (num_providers > 0) {
        printf("  API version: %d.%d\n",
               QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
        printf("  Backend ID: %d\n", providers[0]->backendId);
    }

    dlclose(backend_handle);
    return 0;
}
