/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

%typemap(in) (const char** in), (char** in)
{
    AV *tempav;
    I32 len;
    int i;
    SV  **tv;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    len = av_len(tempav) + 1;
    if(len!=0) 
    {
        $1 = (char **) safemalloc((len)*sizeof(char *));
        for (i = 0; i < len; i++) {
            tv = av_fetch(tempav, i, 0);
            $1[i] = (char *) SvPV_nolen(*tv);
        }
    }
    else
    {
       $1 = NULL;
    }
}
%typemap(freearg) (const char** in), (char** in)  {
    Safefree($1);
}

%typemap(in) (const char **keys, const char **vals), (char **keys, char **vals), (const char* const* keys, const char* const* vals)
{
    HV *temphv;
    char *key;
    SV *val;
    I32 len;
    int hash_len;
    int i = 0;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVHV)
    croak("Argument $argnum is not a hash.");
        temphv = (HV*)SvRV($input);
    hash_len = hv_iterinit(temphv);
    if(hash_len)
    {
        $1 = (char **)safemalloc(hash_len*sizeof(char *));
        $2 = (char **)safemalloc(hash_len*sizeof(char *));
        while ((val = hv_iternextsv(temphv, &key, &len)))
        {
            $1[i] = key;
            $2[i] = SvPV_nolen(val);
            ++i;
        }
    }
    else
    {
       $1 = NULL;
       $2 = NULL;
    }
}
%typemap(freearg) (const char **keys, const char **vals), (char **keys, char **vals) 
{
    Safefree($1);
    Safefree($2);
}

%typemap(in,numinputs=0) (const char **out) (char *temp)
{
    temp = NULL;
    $1 = &temp;
}

%typemap(argout) (const char **out)
{
    if(!result)
    {
        $result = newSVpv(*$1, 0);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in) (void **out_pdata) (void *temp)
{
    temp = NULL;
    $1 = &temp;
}

%typemap(argout) (void **out_pdata)
{
    if(!result)
    {
        $result = newSVpvn((char*)(*$1), SvIV(ST(1)));
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (int *out) (int temp), (bool *out) (bool temp)
{
    temp = 0;
    $1 = &temp;
}

%typemap(argout) (int *out), (bool *out)
{
    if(!result)
    {
        $result = newSViv(*$1);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (const int **out_stypes) (int* temp)
{
    temp = NULL;
    $1 = &temp;
}

%typemap(argout) (const int **out_stypes)
{
    if(av_len((AV*)SvRV(ST(3))) == -1 && !result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*arg4*sizeof(SV *));
        for (i = 0; i < *arg4 ; i++) {
            svs[i] = newSViv((*$1)[i]);
            sv_2mortal(svs[i]);
        }
        myav = av_make(*arg4, svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (nn_uint *out_size, const char ***out_array) (nn_uint temp_size, char** temp),
                         (mx_uint *out_size, const char ***out_array) (mx_uint temp_size, char** temp)
{
    $1 = &temp_size;
    *$1 = 0;
    $2 = &temp;
}

%typemap(argout) (nn_uint *out_size, const char ***out_array),
                 (mx_uint *out_size, const char ***out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = newSVpv((*$2)[i],0);
            sv_2mortal(svs[i]);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint *out_size, const char ***out_array2) (mx_uint temp_size, char** temp)
{
    $1 = &temp_size;
    *$1 = 0;
    $2 = &temp;
}

%typemap(argout) (mx_uint *out_size, const char ***out_array2)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *)*2);
        for (i = 0; i < *$1*2 ; i++) {
            svs[i] = newSVpv((*$2)[i],0);
            sv_2mortal(svs[i]);
        }
        myav = av_make(*$1*2,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in) (FunctionHandle in)
{
    int res;
    void **void_ptrptr = const_cast< void** >(&$1);
    res = SWIG_ConvertPtr($input,void_ptrptr, 0, 0);
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "FunctionHandle""'"); 
    }
}

%typemap(in) (AtomicSymbolCreator in)
{
    int res = SWIG_ConvertPtr($input,&$1, 0, 0);
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "AtomicSymbolCreator""'"); 
    }
}

%typemap(in) (const void *in), (void *in)
{
    $1 = (void *)SvPV_nolen($input);
}

%typemap(in) (const char *in)
{
    $1 = SvPV_nolen($input);
}

%typemap(in) (const mx_uint *in), (mx_uint *in)
{
    AV *tempav;
    int i;
    SV  **tv;
    int av_len;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    av_len = av_len(tempav) + 1;
    if(av_len)
    {
        $1 = (mx_uint *)safemalloc(av_len*sizeof(mx_uint));
        for (i = 0; i < av_len; i++) {
            tv = av_fetch(tempav, i, 0);
            $1[i] = (mx_uint)SvIV(*tv);
        }
    }
    else
    {
       $1 = NULL;
    }
}

%typemap(freearg) (const mx_uint *in), (mx_uint *in) {
    Safefree($1);
}

%typemap(in) (const int *in), (int *in)
{
    AV *tempav;
    int i;
    SV  **tv;
    int av_len; 
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    av_len = av_len(tempav) + 1;
    if(av_len)
    {
        $1 = (int *)safemalloc(av_len*sizeof(int));
        for (i = 0; i < av_len; i++) {
            tv = av_fetch(tempav, i, 0);
            $1[i] = (int)SvIV(*tv);
        }
    }
    else
    {
       $1 = NULL;
    }

}

%typemap(freearg) (const int *in), (int *in) {
    Safefree($1);
}

%typemap(in) (dim_t *in)
{
    AV *tempav;
    int i;
    SV  **tv;
    int av_len; 
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    av_len = av_len(tempav) + 1;
    if(av_len)
    {
        $1 = (dim_t *)safemalloc(av_len*sizeof(dim_t));
        for (i = 0; i < av_len; i++) {
            tv = av_fetch(tempav, i, 0);
            $1[i] = (dim_t)SvIV(*tv);
        }
    }
    else
    {
       $1 = NULL;
    }

}

%typemap(freearg) (dim_t *in) {
    Safefree($1);
}

%typemap(in) (NDArrayHandle* in), (SymbolHandle* in)
{
    AV *tempav;
    int i;
    SV  **tv;
    int res;
    int av_len;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    av_len = av_len(tempav) + 1;
    if(av_len)
    {
        $1 = ($1_type)safemalloc(av_len*sizeof($*1_type));
        for (i = 0; i < av_len; i++) {
            tv = av_fetch(tempav, i, 0);
            res = SWIG_ConvertPtr(*tv,SWIG_as_voidptrptr(&$1[i]), $*1_descriptor, 0);
            if (!SWIG_IsOK(res)) {
                SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "$*1_type""'"); 
            }
        }
    }
    else
    {
       $1 = NULL;
    }
}
%typemap(freearg) (NDArrayHandle* in), (SymbolHandle* in) {
    Safefree($1);
}

%typemap(in) (void** cuda_kernel_args)
{
    AV *tempav;
    int i;
    SV  **tv;
    int res;
    int av_len;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    av_len = av_len(tempav) + 1;
    if(av_len)
    {
        $1 = ($1_type)safemalloc(av_len*sizeof($*1_type));
        for (i = 0; i < av_len; i++) {
            tv = av_fetch(tempav, i, 0);
            res = SWIG_ConvertPtr(*tv,SWIG_as_voidptrptr(&$1[i]), SWIGTYPE_p_MXNDArray, 0);
            if (!SWIG_IsOK(res)) {
                $1[i] = (void*)SvPV_nolen(*tv);
            }
        }
    }
    else
    {
       $1 = NULL;
    }
}
%typemap(freearg) (void** cuda_kernel_args) {
    Safefree($1);
}

%typemap(in) (mx_float *in)
{
    AV *tempav;
    int i, len;
    SV  **tv;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    len = av_len(tempav) + 1;
    if(len)
    {
        $1 = (mx_float *)safemalloc(len*sizeof(mx_float));
        for (i = 0; i < len; i++) {
            tv = av_fetch(tempav, i, 0);
            $1[i] = (mx_float)SvNV(*tv);
        }
    }
    else
    {
       $1 = NULL;
    }
}

%typemap(freearg) (mx_float *in) {
    Safefree($1);
}

%typemap(in,numinputs=0) (NDArrayHandle *out) (NDArrayHandle temp),
                         (FunctionHandle* out) (FunctionHandle temp),
                         (SymbolHandle *out) (SymbolHandle temp),
                         (ExecutorHandle *out) (ExecutorHandle temp),
                         (DataIterHandle *out) (ExecutorHandle temp),
                         (KVStoreHandle *out) (KVStoreHandle temp),
                         (RecordIOHandle *out) (RecordIOHandle temp),
                         (RtcHandle *out) (RtcHandle temp),
                         (CachedOpHandle *out) (CachedOpHandle temp),
                         (CudaModuleHandle *out) (CudaModuleHandle temp),
                         (CudaKernelHandle *out) (CudaKernelHandle temp)
{
    $1 = &temp;
}
%typemap(argout) (NDArrayHandle *out), (FunctionHandle* out), (SymbolHandle *out), (ExecutorHandle *out), (DataIterHandle *out),
                 (KVStoreHandle *out), (RecordIOHandle *out), (RtcHandle *out) (RtcHandle temp), (CachedOpHandle *out) (CachedOpHandle temp),
                 (CudaModuleHandle *out) (CudaModuleHandle temp), (CudaKernelHandle *out) (CudaKernelHandle temp)

{
    if(!result)
    {
        $result =  SWIG_NewPointerObj(SWIG_as_voidptr(*$1), $*1_descriptor, 0); argvi++;
    }
}

%typemap(in) (mx_float **out_pdata) (mx_float *temp_pdata)
{
    $1 = &temp_pdata;
}
%typemap(argout) (mx_float **out_pdata)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int len;
        int i = 0;
        len = SvIV($input); 
        svs = (SV **)safemalloc(len*sizeof(SV *));
        for (i = 0; i < len ; i++) {
            svs[i] = newSVnv((*$1)[i]);
            sv_2mortal(svs[i]);
        }
        myav = av_make(len,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (char const **out_array, size_t *out_size) (char * temp, size_t temp_size)
{
    $2 = &temp_size;
    *$2 = 0;
    $1 = &temp;
}

%typemap(argout) (char const **out_array, size_t *out_size)
{
    if(!result)
    {
        $result = newSVpvn(*$1, *$2);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (size_t *out_size, char const **out_array) (size_t temp_size, char *temp)
{
    $1 = &temp_size;
    *$1 = 0;
    $2 = &temp;
}

%typemap(argout) (size_t *out_size, char const **out_array)
{
    if(!result)
    {
        $result = newSVpvn(*$2, *$1);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint *out_dim, const mx_uint **out_pdata) (mx_uint temp_dim, mx_uint *temp_pdata)
{
    $1 = &temp_dim;
    $2 = &temp_pdata;
}

%typemap(argout) (mx_uint *out_dim, const mx_uint **out_pdata)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = newSViv((*$2)[i]);
            sv_2mortal(svs[i]);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (uint64_t **out_index, uint64_t *out_size) (uint64_t *temp1, uint64_t temp2)
{
    $1 = &temp1;
    $2 = &temp2;
    *$2 = 0;
}

%typemap(argout) (uint64_t **out_index, uint64_t *out_size)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$2*sizeof(SV *));
        for (i = 0; i < *$2 ; i++) {
            svs[i] = newSViv((*$1)[i]);
            sv_2mortal(svs[i]);
        }
        myav = av_make(*$2,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint *out_size, FunctionHandle** out_array) (mx_uint temp_size, FunctionHandle* temp),
                         (mx_uint *out_size, AtomicSymbolCreator** out_array) (mx_uint temp_size, AtomicSymbolCreator* temp),
                         (mx_uint *out_size, DataIterCreator **out_array) (mx_uint temp_size, DataIterCreator* temp),
                         (mx_uint *out_size, NDArrayHandle** out_array) (mx_uint temp_size, NDArrayHandle* temp)
{
    $1 = &temp_size;
    *$1 = 0;
    $2 = &temp;
}

// many argouts needed because SWIG can't $**2_mangle
%typemap(argout) (mx_uint *out_size, AtomicSymbolCreator** out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXAtomicSymbolCreator, 0);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(argout) (mx_uint *out_size, FunctionHandle** out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXFunction, 0);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(argout) (mx_uint *out_size, DataIterCreator **out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXDataIterCreator, 0);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(argout) (mx_uint *out_size, NDArrayHandle** out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXNDArray, 0);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint* couple_out_size, NDArrayHandle** out_first_array, NDArrayHandle** out_second_array)
                         (mx_uint t, NDArrayHandle* t1, NDArrayHandle* t2)
{
    $1 = &t;
    *$1 = 0;
    $2 = &t1;
    $3 = &t2;
}

%typemap(argout) (mx_uint* couple_out_size, NDArrayHandle** out_first_array, NDArrayHandle** out_second_array)
{
    if(!result)
    {
        AV *container, *in_args, *arg_grads;
        int i;
        container = newAV();
        in_args = newAV();
        arg_grads = newAV();
        for (i = 0; i < *$1 ; i++) {
            av_push(in_args, SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXNDArray, 0)));
            av_push(arg_grads, SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$3)[i]), SWIGTYPE_p_MXNDArray, 0)));
        }
        av_push(container, newRV_noinc((SV*)in_args));
        av_push(container, newRV_noinc((SV*)arg_grads));
        $result = newRV_noinc((SV*)container);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (NDArrayHandle **out_grad) (NDArrayHandle* temp)
{
    int vars = SvIV(ST(3));
    if(vars)
    {
        $1 = &temp;
    }
    else
    {
        $1 = NULL;
    }
}


%typemap(argout) (NDArrayHandle** out_grad)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        int len = SvIV(ST(3));
        svs = (SV **)safemalloc(len*sizeof(SV *));
        for (i = 0; i < len ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$1)[i]), SWIGTYPE_p_MXNDArray, 0);
        }
        myav = av_make(len,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (int **out_stype) (int *temp)
{
    int vars = SvIV(ST(3));
    if(vars)
    {
        $1 = &temp;
    }
    else
    {
        $1 = NULL;
    }
}

%typemap(argout) (int** out_stype)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        int len = SvIV(ST(3));
        svs = (SV **)safemalloc(len*sizeof(SV *));
        for (i = 0; i < len ; i++) {
            svs[i] = newSViv((*$1)[i]);
        }
        myav = av_make(len,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in) (int *out_size, NDArrayHandle** out_array) (int temp, NDArrayHandle* temp_array)
{
    AV *tempav;
    int i;
    SV  **tv;
    int res;
    int av_len;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    av_len = av_len(tempav) + 1;
    temp_array = NULL;
    if(av_len)
    {
        temp_array = (void**)safemalloc(av_len*sizeof(void*));
        for (i = 0; i < av_len; i++) {
            tv = av_fetch(tempav, i, 0);
            res = SWIG_ConvertPtr(*tv,SWIG_as_voidptrptr(&(temp_array[i])), 0, 0);
            if (!SWIG_IsOK(res)) {
                SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "NDArray""'"); 
            }
        }
    }
    temp = av_len;
    $1 = &temp;
    $2 = &temp_array;
}

%typemap(freearg) (int *out_size, NDArrayHandle** out_array) {
    if(av_len((AV*)SvRV(ST(3))) > -1)
    {
        Safefree(*$2);
    }
}

%typemap(argout) (int *out_size, NDArrayHandle** out_array)
{
    SV **svs;
    int i = 0;
    if(av_len((AV*)SvRV(ST(3))) == -1)
    {
        if(!result)
        {
            AV *container = newAV();
            for (i = 0; i < *$1 ; i++) {
                av_push(container, SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXNDArray, 0)));
            }
            $result = newRV_noinc((SV*)container);
            sv_2mortal($result);
            argvi++;
        }
    }
}

%typemap(in,numinputs=0) (const char **name,
                          const char **description,
                          mx_uint *num_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions
                          ) 
                          (char *name_temp,
                           char *desc_temp,
                           mx_uint num_args_temp,
                           char **names_temp,
                           char **types_temp,
                           char **descs_temp
                           )
{
    $1 = &name_temp;
    $2 = &desc_temp;
    $3 = &num_args_temp;
    *$3 = 0;
    $4 = &names_temp;
    $5 = &types_temp;
    $6 = &descs_temp;
}

%typemap(argout) (const char **name,
                  const char **description,
                  mx_uint *num_args,
                  const char ***arg_names,
                  const char ***arg_type_infos,
                  const char ***arg_descriptions
                  )
{
    if(!result)
    {
        AV *container, *names, *types, *descs;
        int i;
        container = newAV();
        names = newAV();
        types = newAV();
        descs = newAV();
        if($1) av_push(container, newSVpv(*$1,0));
        if($2) av_push(container, newSVpv(*$2,0));
        if($3)
        {
            for (i = 0; i < *$3 ; i++) {
                av_push(names, newSVpv((*$4)[i],0));
                av_push(types, newSVpv((*$5)[i],0));
                av_push(descs, newSVpv((*$6)[i],0));
            }
        }
        av_push(container, newRV_noinc((SV*)names));
        av_push(container, newRV_noinc((SV*)types));
        av_push(container, newRV_noinc((SV*)descs));
        $result = newRV_noinc((SV*)container);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (const char **name,
                          const char **description,
                          mx_uint *num_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions,
                          const char **key_var_num_args
                          ) 
                          (char *name_temp, 
                           char *desc_temp, 
                           mx_uint num_args_temp, 
                           char **names_temp,
                           char **types_temp,
                           char **descs_temp,
                           char *key_temp
                           )
{
    $1 = &name_temp; 
    $2 = &desc_temp;
    $3 = &num_args_temp;
    *$3 = 0;
    $4 = &names_temp;
    $5 = &types_temp;
    $6 = &descs_temp;
    $7 = &key_temp;
}

%typemap(argout) (const char **name,
                  const char **description,
                  mx_uint *num_args,
                  const char ***arg_names,
                  const char ***arg_type_infos,
                  const char ***arg_descriptions,
                  const char **key_var_num_args
                  )
{
    if(!result)
    {
        AV *container, *names, *types, *descs;
        int i;
        container = newAV();
        names = newAV();
        types = newAV();
        descs = newAV();
        if($1) av_push(container, newSVpv(*$1,0));
        if($2) av_push(container, newSVpv(*$2,0));
        if($3)
        {
            for (i = 0; i < *$3 ; i++) {
                av_push(names, newSVpv((*$4)[i],0));
                av_push(types, newSVpv((*$5)[i],0));
                av_push(descs, newSVpv((*$6)[i],0));
            }
        }
        av_push(container, newRV_noinc((SV*)names));
        av_push(container, newRV_noinc((SV*)types));
        av_push(container, newRV_noinc((SV*)descs));
        if($7) av_push(container, newSVpv(*$7,0));
        $result = newRV_noinc((SV*)container);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint *out) (mx_uint temp), (size_t *out) (size_t temp)
{
    $1 = &temp;
    *$1 = 0;
}

%typemap(argout) (mx_uint *out), (size_t *out)
{
    if(!result)
    {
        $result = newSViv(*$1);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint *in_shape_size, const mx_uint **in_shape_ndim, const mx_uint ***in_shape_data) 
                         (mx_uint temp1, mx_uint *temp2, mx_uint **temp3),
                         (mx_uint *out_shape_size, const mx_uint **out_shape_ndim, const mx_uint ***out_shape_data) 
                         (mx_uint temp1, mx_uint *temp2, mx_uint **temp3),
                         (mx_uint *aux_shape_size, const mx_uint **aux_shape_ndim, const mx_uint ***aux_shape_data) 
                         (mx_uint temp1, mx_uint *temp2, mx_uint **temp3)
{
    $1 = &temp1;
    $2 = &temp2;
    $3 = &temp3;
    *$1 = 0;
}

%typemap(argout) (mx_uint *in_shape_size, const mx_uint **in_shape_ndim, const mx_uint ***in_shape_data),
                 (mx_uint *out_shape_size, const mx_uint **out_shape_ndim, const mx_uint ***out_shape_data),
                 (mx_uint *aux_shape_size, const mx_uint **aux_shape_ndim, const mx_uint ***aux_shape_data)
{
    if(!result && *arg15)
    {
        AV *container;
        AV *tmp;
        int i, j;
        container = newAV();
        for (i = 0; i < *$1 ; i++)
        {
            tmp = newAV();
            int len = (*$2)[i];
            for (j = 0; j < len ; j++)
            {
                av_push(tmp, newSViv((*$3)[i][j]));
            }
            av_push(container, newRV((SV*)tmp));
        }
        $result = newRV_noinc((SV*)container);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint *in_type_size, const int **in_type_data)
                         (mx_uint temp1, int *temp2),
                         (mx_uint *out_type_size, const int **out_type_data) 
                         (mx_uint temp1, int *temp2),
                         (mx_uint *aux_type_size, const int **aux_type_data) 
                         (mx_uint temp1, int *temp2)
{
    $1 = &temp1;
    $2 = &temp2;
    *$1 = 0;
}

%typemap(argout)  (mx_uint *in_type_size,  const int **in_type_data),
                  (mx_uint *out_type_size, const int **out_type_data),
                  (mx_uint *aux_type_size, const int **aux_type_data)

{
    if(!result && *arg11)
    {
        AV *container;
        int i;
        container = newAV();
        for (i = 0; i < *$1 ; i++) 
        {
            av_push(container, newSViv((*$2)[i]));
        }
        $result = newRV_noinc((SV*)container);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint* num_in_args,
                          NDArrayHandle** in_args,
                          NDArrayHandle** arg_grads)
                         (mx_uint temp1,
                         NDArrayHandle* temp2,
                         NDArrayHandle* temp3)
{
    $1 = &temp1;
    $2 = &temp2;
    $3 = &temp3;
    *$1 = 0;
}

%typemap(argout) (mx_uint* num_in_args,
                  NDArrayHandle** in_args,
                  NDArrayHandle** arg_grads)
{
    if(!result)
    {
        AV *container1 = newAV();
        AV *container2 = newAV();
        for (int i = 0; i < *$1 ; i++)
        {
            av_push(container1, SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXNDArray, 0)));
            av_push(container2, (*$3)[i] ? SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$3)[i]), SWIGTYPE_p_MXNDArray, 0)) : newSV(0));
        }
        $result = newRV_noinc((SV*)container1);
        sv_2mortal($result);
        argvi++;
        $result = newRV_noinc((SV*)container2);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (mx_uint* num_aux_states,
                          NDArrayHandle** aux_states)
                         (mx_uint temp1,
                         NDArrayHandle* temp2)
{
    $1 = &temp1;
    $2 = &temp2;
    *$1 = 0;
}

%typemap(argout) (mx_uint* num_aux_states,
                  NDArrayHandle** aux_states)
{
    if(!result)
    {
        AV *container  = newAV();
        for (int i = 0; i < *$1 ; i++)
        {
            av_push(container, SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_MXNDArray, 0)));
        }
        $result = newRV_noinc((SV*)container);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in) (int* shared_buffer_len,
              const char** shared_buffer_name_list,
              NDArrayHandle* shared_buffer_handle_list,
              const char*** updated_shared_buffer_name_list,
              NDArrayHandle** updated_shared_buffer_handle_list)
              (int temp1,
               char* temp2,
               NDArrayHandle temp3,
               char** temp4,
               NDArrayHandle* temp5)
{
    HV *temphv;
    char *key;
    SV *val;
    I32 len;
    int res;
    int i = 0;
    int hash_len;
    $1 = &temp1;
    $2 = &temp2;
    $3 = &temp3;
    $4 = &temp4;
    $5 = &temp5;
    if (!SvROK($input))
    {
        *$1 = -1;
        $2 = NULL;
        $3 = NULL;
    }
    else
    {
        if (SvTYPE(SvRV($input)) != SVt_PVHV)
            croak("Argument $argnum is not a hash.");
        temphv = (HV*)SvRV($input);
        *$1 = hv_iterinit(temphv);
        if(*$1)
        {
            $2 = (char**)safemalloc((*$1)*sizeof(char*));
            $3 = (void**)safemalloc((*$1)*sizeof(void*));
            while ((val = hv_iternextsv(temphv, &key, &len)))
            {
                $2[i] = key;
                res = SWIG_ConvertPtr(val,SWIG_as_voidptrptr(&($3[i])), 0, 0);
                if (!SWIG_IsOK(res)) {
                    SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "NDArray""'"); 
                }
                i++;
            }
        }
        else
        {
            $2 = NULL;
            $3 = NULL;
        }
    }
}

%typemap(freearg) (int* shared_buffer_len,
                   const char** shared_buffer_name_list,
                   NDArrayHandle* shared_buffer_handle_list,
                   const char*** updated_shared_buffer_name_list,
                   NDArrayHandle** updated_shared_buffer_handle_list)
{
    Safefree($2);
    Safefree($3);
}

%typemap(argout)  (int* shared_buffer_len,
                   const char** shared_buffer_name_list,
                   NDArrayHandle* shared_buffer_handle_list,
                   const char*** updated_shared_buffer_name_list,
                   NDArrayHandle** updated_shared_buffer_handle_list)

{
    if(!result)
    {
        HV* hash = newHV();
        for(int j = 0; j < *$1; j++)
        {
            hv_store(hash, (*$4)[j], strlen((*$4)[j]), SvREFCNT_inc(SWIG_NewPointerObj(SWIG_as_voidptr((*$5)[j]), SWIGTYPE_p_MXNDArray, 0)), 0);
        }
        $result = newRV_noinc((SV*)hash);
        sv_2mortal($result);
        argvi++;
    }
}


%typemap(in) (uint32_t x)
{
    union fbits u;
    u.f = SvNV($input);
    $1 = u.x;
}

%typemap(out) (uint16_t)
{
    $result = newSViv($1);
    sv_2mortal($result);
    argvi++;
}

%typemap(in) (uint16_t x)
{
    $1 = SvIV($input);
}

%typemap(out) (uint32_t)
{
    union fbits u;
    u.x = $1;
    $result = newSVnv(u.f);
    sv_2mortal($result);
    argvi++;
}

%typemap(in,numinputs=0) (MXKVStoreUpdater* updater)
{
    $1 = KVStore_callback;
}

%typemap(in,numinputs=0) (MXKVStoreStrUpdater* updater)
{
    $1 = KVStoreStr_callback;
}

%typemap(in,numinputs=0) (MXKVStoreServerController* controller)
{
    $1 = KVStoreServer_callback;
}

%typemap(in,numinputs=0) (ExecutorMonitorCallback callback)
{
    $1 = ExecutorMonitor_callback;
}

%typemap(in) (void* callback_handle)
{
    $1 = (void*)newSVsv($input);
}
