#!/bin/sh
#===============================================================================
# Copyright 2003-2016 Intel Corporation All Rights Reserved.
#
# The source code,  information  and material  ("Material") contained  herein is
# owned by Intel Corporation or its  suppliers or licensors,  and  title to such
# Material remains with Intel  Corporation or its  suppliers or  licensors.  The
# Material  contains  proprietary  information  of  Intel or  its suppliers  and
# licensors.  The Material is protected by  worldwide copyright  laws and treaty
# provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
# modified, published,  uploaded, posted, transmitted,  distributed or disclosed
# in any way without Intel's prior express written permission.  No license under
# any patent,  copyright or other  intellectual property rights  in the Material
# is granted to  or  conferred  upon  you,  either   expressly,  by implication,
# inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
# property rights must be express and approved by Intel in writing.
#
# Unless otherwise agreed by Intel in writing,  you may not remove or alter this
# notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
# suppliers or licensors in any way.
#===============================================================================

mkl_help() {
    echo ""
    echo "Syntax:"
    echo "  source $__mkl_tmp_SCRIPT_NAME <arch> [MKL_interface] [${__mkl_tmp_MOD_NAME}]"
    echo ""
    echo "   <arch> must be one of the following"
    echo "       ia32         : Setup for IA-32 architecture"
    echo "       intel64      : Setup for Intel(R) 64 architecture"
    echo "       mic          : Setup for Intel(R) Many Integrated Core Architecture"
    echo ""
    echo "   ${__mkl_tmp_MOD_NAME} (optional) - set path to Intel(R) MKL F95 modules"
    echo ""
    echo "   MKL_interface (optional) - Intel(R) MKL programming interface for intel64"
    echo "                              Not applicable without ${__mkl_tmp_MOD_NAME}"
    echo "       lp64         : 4 bytes integer (default)"
    echo "       ilp64        : 8 bytes integer"
    echo ""
    echo "If the arguments to the sourced script are ignored (consult docs for"
    echo "your shell) the alternative way to specify target is environment"
    echo "variables COMPILERVARS_ARCHITECTURE or MKLVARS_ARCHITECTURE to pass"
    echo "<arch> to the script, MKLVARS_INTERFACE to pass <MKL_interface> and"
    echo "MKLVARS_MOD to pass <${__mkl_tmp_MOD_NAME}>"
    echo ""
}

get_tbb_library_directory() {
    __tbb_tmp_lib_dir="gcc4.1"
    which gcc >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        __tbb_tmp_gcc_version_full=$(gcc --version | grep "gcc" | egrep -o " [0-9]+\.[0-9]+\.[0-9]+.*" | sed -e "s/^\ //")
        if [ $? -eq 0 ]; then
            __tbb_tmp_gcc_version=$(echo "${__tbb_tmp_gcc_version_full}" | egrep -o "^[0-9]+\.[0-9]+\.[0-9]+")
        fi
        case "${__tbb_tmp_gcc_version}" in
        4.[7-9]*|[5-9]* )
            __tbb_tmp_lib_dir="gcc4.7";;
        4.[4-6]* )
            __tbb_tmp_lib_dir="gcc4.4";;
        * )
            __tbb_tmp_lib_dir="gcc4.1";;
        esac
    fi
    echo ${__tbb_tmp_lib_dir}
}

if_mic() {
    __cmd=$1
    __arg1=$2
    __arg2=$3

    if [ "${__mkl_tmp_TARGET_ARCH}" = "mic" ]; then
        echo :`${__cmd} ${__arg1} ${__arg2}`
    fi
}

set_ld_library_path() {
    __tmp_target_arch_path=$1
    __tmp_ld_library_path="${__compiler_dir}/${__tmp_target_arch_path}:${__mkl_lib_dir}/${__tmp_target_arch_path}"

    __tmp_tbb_arch_path=$2
    __tmp_ld_library_path=${__tmp_tbb_arch_path:+"${__tmp_tbb_arch_path}:"}${__tmp_ld_library_path}

    echo "${__tmp_ld_library_path}"
}

set_library_path() {
    __tmp_target_arch_path=$1
    __tmp_tbb_arch_path=$2

    if [ "${__tmp_target_arch_path}" = "${__subdir_arch_ia32}" ]; then
        __tmp_library_path="${__compiler_dir}/${__tmp_target_arch_path}:${__mkl_lib_dir}/${__tmp_target_arch_path}"
        __tmp_library_path=${__tmp_tbb_arch_path:+"${__tmp_tbb_arch_path}:"}${__tmp_library_path}
    else
        __tmp_library_path="${__compiler_dir}/${__subdir_arch_intel64}:${__mkl_lib_dir}/${__subdir_arch_intel64}"
        __tmp_library_path=${__tmp_tbb_arch_path:+"${__tmp_tbb_arch_path}:"}${__tmp_library_path}
    fi

    echo "${__tmp_library_path}"
}

set_mic_ld_library_path() {
    __tmp_mic_ld_library_path="${__compiler_dir}/${__subdir_arch_mic}:${__mkl_lib_dir}/${__subdir_arch_mic}"
    __tmp_tbb_arch_path=$1

    __tmp_mic_ld_library_path=${__tmp_tbb_arch_path:+"${__tmp_tbb_arch_path}:"}${__tmp_mic_ld_library_path}

    echo "${__tmp_mic_ld_library_path}"
}

set_mic_library_path() {
    __tmp_mic_library_path="${__compiler_dir}/${__subdir_arch_mic}:${__mkl_lib_dir}/${__subdir_arch_mic}"
    __tmp_tbb_arch_path=$1

    __tmp_mic_library_path=${__tmp_tbb_arch_path:+"${__tmp_tbb_arch_path}:"}${__tmp_mic_library_path}

    echo "${__tmp_mic_library_path}"
}

set_nls_path() {
    __tmp_target_arch_path=$1
    echo "${__mkl_lib_dir}/${__tmp_target_arch_path}/locale/%l_%t/%N"
}

set_c_path() {
    __tmp_target_arch_path=$1
    __tmp_target_comp_model=$2
    echo "${CPRO_PATH}/mkl/include/${__tmp_target_arch_path}/${__tmp_target_comp_model}"
}

set_tbb_path() {
    __tmp_target_arch_path=$1

    __tmp_tbb_subdir="/$(get_tbb_library_directory)"
    if [ "${__tmp_target_arch_path}" = "${__subdir_arch_mic}" ]; then __tmp_tbb_subdir=""; fi

    __tmp_tbb_path=${__tbb_lib_dir}/${__tmp_target_arch_path}${__tmp_tbb_subdir}
    echo ${__tmp_tbb_path}
}

set_mkl_env() {
    CPRO_PATH=<INSTALLDIR>
    export MKLROOT=${CPRO_PATH}/mkl

    __mkl_tmp_SCRIPT_NAME="mklvars.sh"
    __mkl_tmp_MOD_NAME=mod

    __mkl_tmp_LP64_ILP64=
    __mkl_tmp_MOD=
    __mkl_tmp_TARGET_ARCH=
    __mkl_tmp_MKLVARS_VERBOSE=
    __mkl_tmp_BAD_SWITCH=

    if [ -z "$1" ] ; then
        if [ -n "$MKLVARS_ARCHITECTURE" ] ; then
            __mkl_tmp_TARGET_ARCH="$MKLVARS_ARCHITECTURE"
        elif [ -n "$COMPILERVARS_ARCHITECTURE" ] ; then
            __mkl_tmp_TARGET_ARCH="$COMPILERVARS_ARCHITECTURE"
        fi
        if [ "${__mkl_tmp_TARGET_ARCH}" != "ia32" -a "${__mkl_tmp_TARGET_ARCH}" != "intel64" -a "${__mkl_tmp_TARGET_ARCH}" != "mic" ] ; then
            __mkl_tmp_TARGET_ARCH=
        fi
        if [ -n "$MKLVARS_INTERFACE" ] ; then
            __mkl_tmp_LP64_ILP64="$MKLVARS_INTERFACE"
            if [ "${__mkl_tmp_LP64_ILP64}" != "lp64" -a "${__mkl_tmp_LP64_ILP64}" != "ilp64" ] ; then
                __mkl_tmp_LP64_ILP64=
            fi
        fi
        if [ -n "$MKLVARS_MOD" ] ; then
            __mkl_tmp_MOD="$MKLVARS_MOD"
        fi
        if [ -n "$MKLVARS_VERBOSE" ] ; then
            __mkl_tmp_MKLVARS_VERBOSE="$MKLVARS_VERBOSE"
        fi
    else
        while [ -n "$1" ]; do
           if   [ "$1" = "ia32" ]        ; then __mkl_tmp_TARGET_ARCH=ia32;
           elif [ "$1" = "intel64" ]     ; then __mkl_tmp_TARGET_ARCH=intel64;
           elif [ "$1" = "mic" ]         ; then __mkl_tmp_TARGET_ARCH=mic;
           elif [ "$1" = "lp64" ]        ; then __mkl_tmp_LP64_ILP64=lp64;
           elif [ "$1" = "ilp64" ]       ; then __mkl_tmp_LP64_ILP64=ilp64;
           elif [ "$1" = "${__mkl_tmp_MOD_NAME}" ] ; then __mkl_tmp_MOD=${__mkl_tmp_MOD_NAME};
           elif [ "$1" = "verbose" ]     ; then __mkl_tmp_MKLVARS_VERBOSE=verbose;
           else
               __mkl_tmp_BAD_SWITCH=$1
               break 10
           fi
           shift;
        done
    fi

    if [ -n "${__mkl_tmp_BAD_SWITCH}" ] ; then

        echo
        echo "ERROR: Unknown option '${__mkl_tmp_BAD_SWITCH}'"
        mkl_help

    else

        if [ -z "${__mkl_tmp_TARGET_ARCH}" ] ; then

            echo
            echo "ERROR: architecture is not defined. Accepted values: ia32, intel64, mic"
            mkl_help

        else
            __compiler_dir="${CPRO_PATH}/compiler/lib"
            __mkl_lib_dir="${MKLROOT}/lib"
            __tbb_lib_dir="${CPRO_PATH}/tbb/lib"
            __cpath="${MKLROOT}/include"

            __subdir_arch_ia32="ia32_lin"
            __subdir_arch_intel64="intel64_lin"
            __subdir_arch_mic="intel64_lin_mic"

            if   [ "${__mkl_tmp_TARGET_ARCH}" = "ia32" ];     then __target_arch_path="${__subdir_arch_ia32}";
            elif [ "${__mkl_tmp_TARGET_ARCH}" = "intel64" ];  then __target_arch_path="${__subdir_arch_intel64}";
            elif [ "${__mkl_tmp_TARGET_ARCH}" = "mic" ];      then __target_arch_path="${__subdir_arch_mic}";
            fi

            __tbb_path_arch=""
            __tbb_path_mic=""
            if [ -z "${TBBROOT}" ]; then
                if [ -d "${__tbb_lib_dir}" ]; then
                    if [ "${__target_arch_path}" = "${__subdir_arch_ia32}" ]; then
                        __tbb_path_arch=$(set_tbb_path ${__subdir_arch_ia32} )
                    else
                        __tbb_path_arch=$(set_tbb_path ${__subdir_arch_intel64} )
                        __tbb_path_mic=$(set_tbb_path ${__subdir_arch_mic} )
                    fi
                fi
            fi

            __ld_library_path=$(set_ld_library_path ${__target_arch_path} ${__tbb_path_arch})
            __ld_library_path=${__ld_library_path}$(if_mic set_ld_library_path ${__subdir_arch_intel64} ${__tbb_path_mic})
            if [ -d "/opt/intel/mic" ] && [ "${__mkl_tmp_TARGET_ARCH}" != "ia32" ] ; then
                __ld_library_path="/opt/intel/mic/coi/host-linux-release/lib:/opt/intel/mic/myo/lib":${__ld_library_path}
            fi

            __library_path=$(set_library_path ${__target_arch_path} ${__tbb_path_arch})

            if [ "${__mkl_tmp_TARGET_ARCH}" != "ia32" ]; then
                __mic_ld_library_path=$(set_mic_ld_library_path ${__tbb_path_mic})
                if [ -d "/opt/intel/mic" ] && [ "${__mkl_tmp_TARGET_ARCH}" != "ia32" ]; then
                    __mic_ld_library_path="/opt/intel/mic/coi/device-linux-release/lib:/opt/intel/mic/myo/lib":${__mic_ld_library_path}
                fi
                __mic_library_path=$(set_mic_library_path ${__tbb_path_mic})
            fi

            __nlspath=$(set_nls_path ${__target_arch_path})
            __nlspath=${__nlspath}$(if_mic set_nls_path ${__subdir_arch_intel64})

            if [ "${__mkl_tmp_MOD}" = "${__mkl_tmp_MOD_NAME}" ] ; then
                if [ "${__mkl_tmp_TARGET_ARCH}" = "ia32" ] ; then
                    __mkl_tmp_LP64_ILP64=
                else
                    if [ -z "$__mkl_tmp_LP64_ILP64" ] ; then
                        __mkl_tmp_LP64_ILP64=lp64
                    fi
                fi
                __cpath=$(set_c_path ${__target_arch_path} ${__mkl_tmp_LP64_ILP64}):${__cpath}
                __cpath=${__cpath}$(if_mic set_c_path ${__subdir_arch_intel64} ${__mkl_tmp_LP64_ILP64})
            fi

            export LD_LIBRARY_PATH="${__ld_library_path}${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}"
            export LIBRARY_PATH="${__library_path}${LIBRARY_PATH+:${LIBRARY_PATH}}"
            export MIC_LD_LIBRARY_PATH="${__mic_ld_library_path}${MIC_LD_LIBRARY_PATH+:${MIC_LD_LIBRARY_PATH}}"
            export MIC_LIBRARY_PATH="${__mic_library_path}${MIC_LIBRARY_PATH+:${MIC_LIBRARY_PATH}}"
            export NLSPATH="${__nlspath}${NLSPATH+:${NLSPATH}}"
            export CPATH="${__cpath}${CPATH+:${CPATH}}"

            if [ "${__mkl_tmp_MKLVARS_VERBOSE}" = "verbose" ] ; then
                echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
                echo LIBRARY_PATH=${LIBRARY_PATH}
                echo MIC_LD_LIBRARY_PATH=${MIC_LD_LIBRARY_PATH}
                echo MIC_LIBRARY_PATH=${MIC_LIBRARY_PATH}
                echo NLSPATH=${NLSPATH}
                echo CPATH=${CPATH}
            fi
        fi
    fi
}

set_mkl_env "$@"
