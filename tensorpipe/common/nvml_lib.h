/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#if defined(TP_USE_ROCM)
#include <rocm_smi/rocm_smi.h>
#else
#include <nvml.h>
#endif

#include <tensorpipe/common/defs.h>
#include <tensorpipe/common/dl.h>

#define TP_NVML_CHECK(nvml_lib, a)                                \
  do {                                                            \
    nvmlReturn_t error = (a);                                     \
    if (error != NVML_SUCCESS) {                                  \
      const char* errorStr;                                       \
      errorStr = (nvml_lib).errorString(error);                   \
      TP_THROW_ASSERT() << __TP_EXPAND_OPD(a) << " " << errorStr; \
    }                                                             \
  } while (false)

namespace tensorpipe {

// Master list of all symbols we care about from libnvidia-ml.

#if defined(TP_USE_ROCM)
#define TP_FORALL_NVML_SYMBOLS(_)                                             \
  _(tp_rsmi_deviceGetComputeRunningProcesses,                                 \
    rsmi_compute_process_info_get,                                            \
    rsmi_status_t,                                                            \
    (rsmi_process_info_t*, uint32_t*))                                        \
  _(tp_rsmi_deviceGetRunningComputeGpus,                                             \
    rsmi_compute_process_gpus_get,                                            \
    rsmi_status_t,                                                            \
    (uint32_t, uint32_t*, uint32_t*))                                         \
  _(tp_rsmi_deviceGetCount_v2,                                                \
    rsmi_num_monitor_devices,                                                 \
    rsmi_status_t,                                                            \
    (uint32_t*))                                                              \
  _(tp_rsmi_deviceGetP2PStatus,                                               \
    rsmi_is_P2P_accessible,                                                   \
    rsmi_status_t,                                                            \
    (uint32_t, uint32_t, bool*))                                              \
  _(tp_rsmi_deviceGetUUID,                                                    \
    rsmi_dev_unique_id_get,                                                   \
    rsmi_status_t,                                                            \
    (uint32_t, uint64_t*))                                                    \
  _(tp_rsmi_errorString,                                                      \
    rsmi_status_string,                                                       \
    rsmi_status_t,                                                            \
    (rsmi_status_t, const char **))                                           \
  _(tp_rsmi_init,                                                             \
    rsmi_init,                                                                \
    rsmi_status_t,                                                            \
    (uint64_t))                                                               \
  _(tp_rsmi_shutdown,                                                         \
    rsmi_shut_down,                                                           \
    rsmi_status_t,                                                            \
    (void))
#else
#define TP_FORALL_NVML_SYMBOLS(_)                                             \
  _(deviceGetComputeRunningProcesses,                                         \
    nvmlDeviceGetComputeRunningProcesses,                                     \
    nvmlReturn_t,                                                             \
    (nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*))                        \
  _(deviceGetCount_v2, nvmlDeviceGetCount_v2, nvmlReturn_t, (unsigned int*))  \
  _(deviceGetHandleByIndex_v2,                                                \
    nvmlDeviceGetHandleByIndex_v2,                                            \
    nvmlReturn_t,                                                             \
    (unsigned int, nvmlDevice_t*))                                            \
  _(deviceGetHandleByUUID,                                                    \
    nvmlDeviceGetHandleByUUID,                                                \
    nvmlReturn_t,                                                             \
    (const char*, nvmlDevice_t*))                                             \
  _(deviceGetP2PStatus,                                                       \
    nvmlDeviceGetP2PStatus,                                                   \
    nvmlReturn_t,                                                             \
    (nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t*)) \
  _(deviceGetUUID,                                                            \
    nvmlDeviceGetUUID,                                                        \
    nvmlReturn_t,                                                             \
    (nvmlDevice_t, char*, unsigned int))                                      \
  _(errorString, nvmlErrorString, const char*, (nvmlReturn_t))                \
  _(init_v2, nvmlInit_v2, nvmlReturn_t, ())                                   \
  _(shutdown, nvmlShutdown, nvmlReturn_t, ())
#endif

#ifdef TP_USE_ROCM
#define NVML_DEVICE_UUID_BUFFER_SIZE 8
#endif

#if defined(TP_USE_CUDA)
#define SMI_LIB_NAME "libnvidia-ml.so.1"
#elif defined(TP_USE_ROCM)
#define SMI_LIB_NAME "librocm_smi64.so"
#endif

// Wrapper for libnvidia-ml.

class NvmlLib {
 private:
  explicit NvmlLib(DynamicLibraryHandle dlhandle)
      : dlhandle_(std::move(dlhandle)) {}

  DynamicLibraryHandle dlhandle_;
  bool inited_ = false;

#define TP_DECLARE_FIELD(method_name, function_name, return_type, args_types) \
  return_type(*function_name##_ptr_) args_types = nullptr;
  TP_FORALL_NVML_SYMBOLS(TP_DECLARE_FIELD)
#undef TP_DECLARE_FIELD

 public:
  NvmlLib() = default;

  // Implement another RAII layer (on top of the one of DynamicLibraryHandle) to
  // deal with nvmlInit_v2 and nvmlShutdown. The default move assignment would
  // fail to shutdown NVML when another instance is moved into it, and it would
  // cause the destructor to shutdown a moved-out instance.
  NvmlLib(const NvmlLib&) = delete;
  NvmlLib& operator=(const NvmlLib&) = delete;
  NvmlLib(NvmlLib&& other) {
    *this = std::move(other);
  }
  NvmlLib& operator=(NvmlLib&& other) {
    std::swap(dlhandle_, other.dlhandle_);
    std::swap(inited_, other.inited_);
#define TP_SWAP_FIELD(method_name, function_name, return_type, args_types) \
  std::swap(function_name##_ptr_, other.function_name##_ptr_);
    TP_FORALL_NVML_SYMBOLS(TP_SWAP_FIELD)
#undef TP_SWAP_FIELD
    return *this;
  }

#define TP_FORWARD_CALL(method_name, function_name, return_type, args_types) \
  template <typename... Args>                                                \
  auto method_name(Args&&... args) const {                                   \
    return (*function_name##_ptr_)(std::forward<Args>(args)...);             \
  }
  TP_FORALL_NVML_SYMBOLS(TP_FORWARD_CALL)
#undef TP_FORWARD_CALL

  static std::tuple<Error, NvmlLib> create() {
    Error error;
    DynamicLibraryHandle dlhandle;
    // To keep things "neat" and contained, we open in "local" mode (as
    // opposed to global) so that the cuda symbols can only be resolved
    // through this handle and are not exposed (a.k.a., "leaked") to other
    // shared objects.
    std::tie(error, dlhandle) = DynamicLibraryHandle::create(
        SMI_LIB_NAME, RTLD_LOCAL | RTLD_LAZY);
    if (error) {
      return std::make_tuple(std::move(error), NvmlLib());
    }
    // Log at level 9 as we can't know whether this will be used in a transport
    // or channel, thus err on the side of this being as low-level as possible
    // because we don't expect this to be of interest that often.
    TP_VLOG(9) << [&]() -> std::string {
      std::string filename;
      std::tie(error, filename) = dlhandle.getFilename();
      if (error) {
        return "Couldn't determine location of shared library libnvidia-ml.so.1: " +
            error.what();
      }
      return "Found shared library libnvidia-ml.so.1 at " + filename;
    }();
    NvmlLib lib(std::move(dlhandle));
#define TP_LOAD_SYMBOL(method_name, function_name, return_type, args_types) \
  {                                                                         \
    void* ptr;                                                              \
    std::tie(error, ptr) = lib.dlhandle_.loadSymbol(#function_name);        \
    if (error) {                                                            \
      return std::make_tuple(std::move(error), NvmlLib());                  \
    }                                                                       \
    TP_THROW_ASSERT_IF(ptr == nullptr);                                     \
    lib.function_name##_ptr_ =                                              \
        reinterpret_cast<decltype(function_name##_ptr_)>(ptr);              \
  }
    TP_FORALL_NVML_SYMBOLS(TP_LOAD_SYMBOL)
#undef TP_LOAD_SYMBOL
    TP_NVML_CHECK(lib, lib.init_v2());
    lib.inited_ = true;
    return std::make_tuple(Error::kSuccess, std::move(lib));
  }

  ~NvmlLib() {
    if (inited_) {
      TP_DCHECK(dlhandle_.hasValue());
      TP_NVML_CHECK(*this, shutdown());
    }
  }

// #define TP_USE_ROCM
#if defined(TP_USE_ROCM)
  rsmi_status_t deviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int* num_items, nvmlProcessInfo_t* procs) const {
    return tp_rsmi_deviceGetComputeRunningProcesses(procs, num_items);
  }

  rsmi_status_t deviceGetCount_v2(unsigned int* num_devices) const {
    return tp_rsmi_deviceGetCount_v2(num_devices);
  }

  rsmi_status_t deviceGetHandleByIndex_v2 (unsigned int index, nvmlDevice_t* handle) const {
    *handle = index;
    return RSMI_STATUS_SUCCESS;
  }

  rsmi_status_t deviceGetComputeRunningGpus(uint32_t pid, nvmlDevice_t* devices, uint32_t* num_devices) const {
    return tp_rsmi_deviceGetRunningComputeGpus(pid, devices, num_devices);
  }

  rsmi_status_t deviceGetP2PStatus(nvmlDevice_t dev_src, nvmlDevice_t dev_dst, nvmlGpuP2PCapsIndex_t p2pCaps, nvmlGpuP2PStatus_t* res) const {
    rsmi_status_t ret;
    bool accessible;
    ret = tp_rsmi_deviceGetP2PStatus(dev_src, dev_dst, &accessible);
    // Need to change the logic at calling function
    *res = static_cast<nvmlGpuP2PStatus_t>(!accessible);
    return ret;
  }

  rsmi_status_t deviceGetUUID(nvmlDevice_t dev_id, char* uuid_str, unsigned int size) const {
    // arg1 - uint32_t
    // arg2 - uint64_t*
    return tp_rsmi_deviceGetUUID(dev_id, reinterpret_cast<uint64_t *>(uuid_str));
  }
  
  rsmi_status_t deviceGetHandleByUUID(const char* uuid_str, nvmlDevice_t* device) const {
    // printf("UUID recv'd: %lu\n", *(uint64_t*)uuid_str);
    rsmi_status_t ret;
    uint32_t num_devices;
    ret = tp_rsmi_deviceGetCount_v2(&num_devices);
    if (ret != 0) {
      return ret;
    }

    for (uint32_t i = 0; i < num_devices; i++) {
      uint64_t uuid;
      ret = tp_rsmi_deviceGetUUID(i, &uuid);
      if (ret != 0) {
        return ret;
      }
      // printf("UUID: %lu\n", uuid);
      if (uuid == *reinterpret_cast<const uint64_t*>(uuid_str)) {
        *device = i;
        return RSMI_STATUS_SUCCESS;
      }
    }

    return RSMI_STATUS_NOT_FOUND;
  }

  const char* errorString(nvmlReturn_t result) const {
    const char* errorStr;
    tp_rsmi_errorString(result, &errorStr);
    return errorStr;
  }

  rsmi_status_t init_v2() {
    return tp_rsmi_init(static_cast<uint64_t>(0));
  }
  rsmi_status_t shutdown() {
    return tp_rsmi_shutdown();
  }
#endif
};

#undef TP_FORALL_NVML_SYMBOLS

} // namespace tensorpipe
