#include <cuda_runtime_api.h>

namespace msdga_module {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace msdga_module
