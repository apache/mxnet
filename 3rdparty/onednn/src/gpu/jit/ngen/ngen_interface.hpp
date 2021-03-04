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

#ifndef NGEN_INTERFACE_HPP
#define NGEN_INTERFACE_HPP


#include "ngen_core.hpp"

#ifdef NGEN_NEO_INTERFACE
#include <iostream>
#endif


namespace ngen {

enum class ExternalArgumentType { Scalar, GlobalPtr, LocalPtr, Hidden };


class InterfaceHandler
{
public:
    inline void externalName(const std::string &name)   { kernelName = name; }
    inline void requireSIMD(int simd_)                  { simd = simd_; }
    inline void requireLocalID(int dimensions)          { needLocalID = dimensions; }

    template <typename DT>
    inline void newArgument(std::string name)           { newArgument(name, getDataType<DT>()); }
    inline void newArgument(std::string name, DataType type, ExternalArgumentType exttype = ExternalArgumentType::Scalar);
    inline void newArgument(std::string name, ExternalArgumentType exttype);

    inline Subregister getArgument(const std::string &name) const;
    inline Subregister getArgumentIfExists(const std::string &name) const;
    inline int getArgumentSurface(const std::string &name) const;
    inline GRF getLocalID(int dim) const;

    const std::string &getExternalName() const          { return kernelName; }

protected:
    struct Assignment {
        std::string name;
        DataType type;
        ExternalArgumentType exttype;
        Subregister reg;
        int surface;
    };

    std::vector<Assignment> assignments;
    std::string kernelName = "default_kernel";
    int needLocalID = 0;
    int simd = 8;
};


// Exceptions.
#ifdef NGEN_SAFE
class unknown_argument_exception : public std::runtime_error {
public:
    unknown_argument_exception() : std::runtime_error("Argument not found") {}
};

class bad_argument_type_exception : public std::runtime_error {
public:
    bad_argument_type_exception() : std::runtime_error("Bad argument type") {}
};

class interface_not_finalized : public std::runtime_error {
public:
    interface_not_finalized() : std::runtime_error("Interface has not been finalized") {}
};
#endif

void InterfaceHandler::newArgument(std::string name, DataType type, ExternalArgumentType exttype)
{
    assignments.push_back({name, type, exttype, GRF(0).ud(0), -1});
}

void InterfaceHandler::newArgument(std::string name, ExternalArgumentType exttype)
{
    DataType type = DataType::invalid;

    switch (exttype) {
        case ExternalArgumentType::GlobalPtr: type = DataType::uq; break;
        case ExternalArgumentType::LocalPtr:  type = DataType::ud; break;
        default:
#ifdef NGEN_SAFE
            throw bad_argument_type_exception();
#else
        break;
#endif
    }

    newArgument(name, type, exttype);
}

Subregister InterfaceHandler::getArgumentIfExists(const std::string &name) const
{
    for (auto &assignment : assignments) {
        if (assignment.name == name)
            return assignment.reg;
    }

    return Subregister{};
}

Subregister InterfaceHandler::getArgument(const std::string &name) const
{
    Subregister arg = getArgumentIfExists(name);

#ifdef NGEN_SAFE
    if (arg.isInvalid())
        throw unknown_argument_exception();
#endif

    return arg;
}

int InterfaceHandler::getArgumentSurface(const std::string &name) const
{
    for (auto &assignment : assignments) {
        if (assignment.name == name) {
#ifdef NGEN_SAFE
            if (assignment.exttype != ExternalArgumentType::GlobalPtr)
                throw unknown_argument_exception();
#endif

            return assignment.surface;
        }
    }

#ifdef NGEN_SAFE
    throw unknown_argument_exception();
#else
    return 0x80;
#endif
}

GRF InterfaceHandler::getLocalID(int dim) const
{
#ifdef NGEN_SAFE
    if (dim > needLocalID) throw unknown_argument_exception();
#endif

    if (simd > 16)
        return GRF(1 + (dim << 1)).uw();
    else
        return GRF(1 + dim).uw();
}


#ifdef NGEN_NEO_INTERFACE

template <HW hw> class OpenCLCodeGenerator;

class NEOInterfaceHandler : public InterfaceHandler
{
    template <HW hw> friend class OpenCLCodeGenerator;
public:
    NEOInterfaceHandler(HW hw_) : hw(hw_)       {}

    void requireBarrier()                                { needBarrier = true; }
    void requireGRF(int grfs)                            { needGRF = grfs; }
    void requireNonuniformWGs()                          { needNonuniformWGs = true; }
    void requireLocalSize()                              { needLocalSize = true; }
    void requireScratch(size_t bytes = 1)                { scratchSize = bytes; }
    void requireSLM(size_t bytes)                        { slmSize = bytes; }
    inline void requireType(DataType type);
    template <typename T> void requireType()             { requireType(getDataType<T>()); }
    void requireWorkgroup(size_t x, size_t y, size_t z)  { wg[0] = x; wg[1] = y; wg[2] = z; }

    inline void finalize();

    inline Subregister getLocalSize(int dim) const;

    inline void generateDummyCL(std::ostream &stream) const;

#ifdef NGEN_ASM
    inline void dumpAssignments(std::ostream &stream) const;
#endif

protected:
    bool finalized = false;
    bool needBarrier = false;
    int32_t needGRF = 128;
    bool needLocalSize = false;
    bool needNonuniformWGs = false;
    bool needHalf = false;
    bool needDouble = false;
    size_t scratchSize = 0;
    size_t slmSize = 0;
    size_t wg[3] = {0, 0, 0};

    int crossthreadGRFs = 0;
    inline int getCrossthreadGRFs() const;
    inline GRF getCrossthreadBase() const;

    HW hw;
};

void NEOInterfaceHandler::requireType(DataType type)
{
    switch (type) {
        case DataType::hf: needHalf = true;   break;
        case DataType::df: needDouble = true; break;
        default: break;
    }
}

static inline const char *getCLDataType(DataType type)
{
    static const char *names[16] = {"uint", "int", "ushort", "short", "uchar", "char", "double", "float", "ulong", "long", "half", "ushort", "INVALID", "INVALID", "INVALID", "INVALID"};
    return names[static_cast<uint8_t>(type) & 0xF];
}

void NEOInterfaceHandler::generateDummyCL(std::ostream &stream) const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif

    if (needHalf)   stream << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    if (needDouble) stream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    if (wg[0] > 0 && wg[1] > 0 && wg[2] > 0)
        stream << "__attribute__((reqd_work_group_size(" << wg[0] << ',' << wg[1] << ',' << wg[2] << ")))\n";
    stream << "__attribute__((intel_reqd_sub_group_size(" << simd << ")))\n";
    stream << "kernel void " << kernelName << '(';

    bool firstArg = true;
    for (const auto &assignment : assignments) {
        if (assignment.exttype == ExternalArgumentType::Hidden) continue;

        if (!firstArg) stream << ", ";

        switch (assignment.exttype) {
            case ExternalArgumentType::GlobalPtr: stream << "global void *"; break;
            case ExternalArgumentType::LocalPtr: stream << "local void *"; break;
            case ExternalArgumentType::Scalar: stream << getCLDataType(assignment.type) << ' '; break;
            default: break;
        }

        stream << assignment.name;
        firstArg = false;
    }
    stream << ") {\n";
    stream << "    global volatile int *____;\n";
    if (hw == HW::Gen9)
        stream << "    volatile double *__df; *__df = 1.1 / *__df;\n";

    if (needLocalID)        stream << "    (void) ____[get_local_id(0)];\n";
    if (needLocalSize)      stream << "    (void) ____[get_enqueued_local_size(0)];\n";
    if (needBarrier)        stream << "    barrier(CLK_GLOBAL_MEM_FENCE);\n";
    if (scratchSize > 0)    stream << "    volatile char scratch[" << scratchSize << "] = {0};\n";
    if (slmSize > 0)        stream << "    volatile local char slm[" << slmSize << "]; slm[0]++;\n";

    stream << "}\n";
}

inline Subregister NEOInterfaceHandler::getLocalSize(int dim) const
{
    static const std::string localSizeArgs[3] = {"__local_size0", "__local_size1", "__local_size2"};
    return getArgument(localSizeArgs[dim]);
}

void NEOInterfaceHandler::finalize()
{
    // Make assignments, following NEO rules:
    //  - all inputs are naturally aligned
    //  - all sub-DWord inputs are DWord-aligned
    //  - first register is
    //      r3 (no local IDs)
    //      r5 (SIMD8/16, local IDs)
    //      r8 (SIMD32, local IDs)
    // [- assign local ptr arguments left-to-right? not checked]
    //  - assign global pointer arguments left-to-right
    //  - assign scalar arguments left-to-right
    //  - assign surface indices left-to-right for global pointers
    //  - no arguments can cross a GRF boundary. Arrays like work size count
    //     as 1 argument for this rule.

    static const std::string localSizeArgs[3] = {"__local_size0", "__local_size1", "__local_size2"};
    static const std::string scratchSizeArg = "__scratch_size";

    GRF base = getCrossthreadBase() + 1;
    int offset = 0;
    int nextSurface = 0;

    auto assignArgsOfType = [&](ExternalArgumentType exttype) {
        for (auto &assignment : assignments) {
            if (assignment.exttype != exttype) continue;

            auto bytes = getBytes(assignment.type);
            auto size = getDwords(assignment.type) << 2;

            if (assignment.name == localSizeArgs[0]) {
                // Move to next GRF if local size arguments won't fit in this one.
                if (offset > 0x20 - (3 * 4)) {
                    offset = 0;
                    base++;
                }
            }

            offset = (offset + size - 1) & -size;
            if (offset >= 0x20) {
                offset = 0;
                base++;
            }

            assignment.reg = base.sub(offset / bytes, assignment.type);

            if (assignment.exttype == ExternalArgumentType::GlobalPtr)
                assignment.surface = nextSurface++;
            else if (assignment.exttype == ExternalArgumentType::Scalar)
                requireType(assignment.type);

            offset += size;
        }
    };

    assignArgsOfType(ExternalArgumentType::LocalPtr);
    assignArgsOfType(ExternalArgumentType::GlobalPtr);
    assignArgsOfType(ExternalArgumentType::Scalar);

    // Add private memory size arguments.
    if (scratchSize > 0)
        newArgument(scratchSizeArg, DataType::uq, ExternalArgumentType::Hidden);

    // Add enqueued local size arguments.
    if (needLocalSize && needNonuniformWGs)
        for (int dim = 0; dim < 3; dim++)
            newArgument(localSizeArgs[dim], DataType::ud, ExternalArgumentType::Hidden);

    assignArgsOfType(ExternalArgumentType::Hidden);

    crossthreadGRFs = base.getBase() - getCrossthreadBase().getBase() + 1;

    // Manually add regular local size arguments.
    if (needLocalSize && !needNonuniformWGs)
        for (int dim = 0; dim < 3; dim++)
            assignments.push_back({localSizeArgs[dim], DataType::ud, ExternalArgumentType::Hidden,
                                   GRF(getCrossthreadBase()).ud(dim + 3), -1});

    finalized = true;
}

GRF NEOInterfaceHandler::getCrossthreadBase() const
{
    if (!needLocalID)
        return GRF(2);
    else
        return GRF((simd <= 16) ? 4 : 7);
}

int NEOInterfaceHandler::getCrossthreadGRFs() const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif
    return crossthreadGRFs;
}

#ifdef NGEN_ASM
void NEOInterfaceHandler::dumpAssignments(std::ostream &stream) const
{
    LabelManager manager;

    for (auto &assignment : assignments) {
        stream << "//  ";
        assignment.reg.outputText(stream, PrintDetail::sub, manager);
        stream << '\t' << assignment.name << std::endl;
    }
}
#endif

#endif

} /* namespace ngen */

#endif /* header guard */
