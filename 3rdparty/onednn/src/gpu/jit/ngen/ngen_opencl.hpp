/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef NGEN_OPENCL_HPP
#define NGEN_OPENCL_HPP

#include <CL/cl.h>

#include <sstream>

#ifndef NGEN_NEO_INTERFACE
#define NGEN_NEO_INTERFACE
#endif
#include "ngen.hpp"
#include "ngen_interface.hpp"

#include "npack/neo_packager.hpp"

namespace ngen {


// Exceptions.
class unsupported_opencl_runtime : public std::runtime_error {
public:
    unsupported_opencl_runtime() : std::runtime_error("Unsupported OpenCL runtime.") {}
};
class opencl_error : public std::runtime_error {
public:
    opencl_error(cl_int status_ = 0) : std::runtime_error("An OpenCL error occurred."), status(status_) {}
protected:
    cl_int status;
};

// OpenCL program generator class.
template <HW hw>
class OpenCLCodeGenerator : public BinaryCodeGenerator<hw>
{
public:
    inline std::vector<uint8_t> getBinary(cl_context context, cl_device_id device, const std::string &options = "-cl-std=CL2.0", const std::vector<uint8_t> &patches = std::vector<uint8_t>{});
    inline cl_kernel getKernel(cl_context context, cl_device_id device, const std::string &options = "-cl-std=CL2.0", const std::vector<uint8_t> &patches = std::vector<uint8_t>{});
    static inline HW detectHW(cl_context context, cl_device_id device);
    const std::string &getExternalName() const { return interface_.getExternalName(); }

protected:
    NEOInterfaceHandler interface_{hw};

    void externalName(const std::string &name)                           { interface_.externalName(name); }
    void requireBarrier()                                                { interface_.requireBarrier(); }
    void requireGRF(int grfs)                                            { interface_.requireGRF(grfs); }
    void requireLocalID(int dimensions)                                  { interface_.requireLocalID(dimensions); }
    void requireLocalSize()                                              { interface_.requireLocalSize(); }
    void requireNonuniformWGs()                                          { interface_.requireNonuniformWGs(); }
    void requireScratch(size_t bytes = 1)                                { interface_.requireScratch(bytes); }
    void requireSIMD(int simd_)                                          { interface_.requireSIMD(simd_); }
    void requireSLM(size_t bytes)                                        { interface_.requireSLM(bytes); }
    inline void requireType(DataType type)                               { interface_.requireType(type); }
    template <typename T> void requireType()                             { interface_.requireType<T>(); }

    void finalizeInterface()                                             { interface_.finalize(); }

    template <typename DT>
    void newArgument(std::string name)                                   { interface_.newArgument<DT>(name); }
    void newArgument(std::string name, DataType type,
                     ExternalArgumentType exttype = ExternalArgumentType::Scalar)
    {
        interface_.newArgument(name, type, exttype);
    }
    void newArgument(std::string name, ExternalArgumentType exttype)     { interface_.newArgument(name, exttype); }

    Subregister getArgument(const std::string &name) const               { return interface_.getArgument(name); }
    Subregister getArgumentIfExists(const std::string &name) const       { return interface_.getArgumentIfExists(name); }
    int getArgumentSurface(const std::string &name) const                { return interface_.getArgumentSurface(name); }
    GRF getLocalID(int dim) const                                        { return interface_.getLocalID(dim); }
    Subregister getLocalSize(int dim) const                              { return interface_.getLocalSize(dim); }

};

#define NGEN_FORWARD_OPENCL(hw) NGEN_FORWARD(hw) \
template <typename... Targs> void externalName(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::externalName(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireBarrier(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireBarrier(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireGRF(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireGRF(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireLocalID(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireLocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireLocalSize(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireLocalSize(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireNonuniformWGs(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireNonuniformWGs(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireScratch(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireScratch(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireSIMD(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireSIMD(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireSLM(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::requireSLM(std::forward<Targs>(args)...); } \
void requireType(ngen::DataType type) { ngen::OpenCLCodeGenerator<hw>::requireType(type); } \
template <typename DT = void> void requireType() { ngen::BinaryCodeGenerator<hw>::template requireType<DT>(); } \
template <typename... Targs> void finalizeInterface(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::finalizeInterface(std::forward<Targs>(args)...); } \
template <typename... Targs> void newArgument(Targs&&... args) { ngen::OpenCLCodeGenerator<hw>::newArgument(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::Subregister getArgument(Targs&&... args) { return ngen::OpenCLCodeGenerator<hw>::getArgument(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::Subregister getArgumentIfExists(Targs&&... args) { return ngen::OpenCLCodeGenerator<hw>::getArgumentIfExists(std::forward<Targs>(args)...); } \
template <typename... Targs> int getArgumentSurface(Targs&&... args) { return ngen::OpenCLCodeGenerator<hw>::getArgumentSurface(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::GRF getLocalID(Targs&&... args) { return ngen::OpenCLCodeGenerator<hw>::getLocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::Subregister getLocalSize(Targs&&... args) { return ngen::OpenCLCodeGenerator<hw>::getLocalSize(std::forward<Targs>(args)...); } \
NGEN_FORWARD_OPENCL_EXTRA

#define NGEN_FORWARD_OPENCL_EXTRA

namespace detail {

static inline void handleCL(cl_int result)
{
    if (result != CL_SUCCESS)
        throw opencl_error{result};
}

static inline std::vector<uint8_t> getOpenCLCProgramBinary(cl_context context, cl_device_id device, const char *src, const char *options)
{
    cl_int status;

    auto program = clCreateProgramWithSource(context, 1, &src, nullptr, &status);

    detail::handleCL(status);
    if (program == nullptr)
        throw opencl_error();

    detail::handleCL(clBuildProgram(program, 1, &device, options, nullptr, nullptr));

    size_t binarySize;
    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binarySize), &binarySize, nullptr));

    std::vector<uint8_t> binary(binarySize);
    const auto *binaryPtr = binary.data();
    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binaryPtr), &binaryPtr, nullptr));

    detail::handleCL(clReleaseProgram(program));

    return binary;
}

}; /* namespace detail */

template <HW hw>
std::vector<uint8_t> OpenCLCodeGenerator<hw>::getBinary(cl_context context, cl_device_id device, const std::string &options, const std::vector<uint8_t> &patches)
{
    std::ostringstream dummyCL;
    auto modOptions = options;

    interface_.generateDummyCL(dummyCL);
    auto dummyCLString = dummyCL.str();

    auto binary = detail::getOpenCLCProgramBinary(context, device, dummyCLString.c_str(), modOptions.c_str());

    npack::replaceKernel(binary, this->getCode(), patches);

    return binary;
}

template <HW hw>
cl_kernel OpenCLCodeGenerator<hw>::getKernel(cl_context context, cl_device_id device, const std::string &options, const std::vector<uint8_t> &patches)
{
    cl_int status;

    auto binary = getBinary(context, device, options, patches);

    const auto *binaryPtr = binary.data();
    size_t binarySize = binary.size();
    auto program = clCreateProgramWithBinary(context, 1, &device, &binarySize, &binaryPtr, nullptr, &status);
    detail::handleCL(status);
    if (program == nullptr)
        throw opencl_error();

    detail::handleCL(clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr));

    auto kernel = clCreateKernel(program, interface_.getExternalName().c_str(), &status);
    detail::handleCL(status);
    if (kernel == nullptr)
        throw opencl_error();

    detail::handleCL(clReleaseProgram(program));

    return kernel;
}

template <HW hw>
HW OpenCLCodeGenerator<hw>::detectHW(cl_context context, cl_device_id device)
{
    const char *dummyCL = "kernel void _(){}";
    const char *dummyOptions = "";

    auto binary = detail::getOpenCLCProgramBinary(context, device, dummyCL, dummyOptions);
    return npack::getBinaryArch(binary);
}

} /* namespace ngen */

#endif
