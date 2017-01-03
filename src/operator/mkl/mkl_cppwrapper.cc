/*******************************************************************************
* Copyright 2016 Intel Corporation
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
*
* \file mkl_cppwrapper.cc
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/



#include "mkl_cppwrapper.h"
#include <stdio.h>
#if MXNET_USE_MKL2017 == 1
#include "mkl_service.h"

int getMKLBuildDate() {
    static int build = 0;
    if (build == 0) {
        MKLVersion v;
        mkl_get_version(&v);
        build = atoi(v.Build);
        printf("MKL Build:%d\n", build);
    }
    return build;
}

bool enableMKLWarnGenerated() {
  return false;
}
#endif  // MSHADOW_USE_MKL2017
