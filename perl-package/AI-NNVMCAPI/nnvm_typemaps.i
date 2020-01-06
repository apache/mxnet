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

%typemap(in) (const char **keys, const char **vals), (char **keys, char **vals)
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
}
%typemap(freearg) (const char **keys, const char **vals), (char **keys, char **vals) 
{
    Safefree($1);
    Safefree($2);
}

%typemap(in,numinputs=0) (const char **out) (char *temp)
{
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

%typemap(in,numinputs=0) (int *out) (int temp)
{
    $1 = &temp;
    *$1 = 0;
}

%typemap(argout) (int *out)
{
    if(!result)
    {
        $result = newSViv(*$1);
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

%typemap(in,numinputs=0) (nn_uint *half_of_out_size, const char ***out_array) (nn_uint temp_size, char **temp)
{
    $1 = &temp_size;
    *$1 = 0;
    $2 = &temp;
}
%typemap(argout) (nn_uint *half_of_out_size, const char ***out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < (*$1)*2 ; i++) {
            svs[i] = newSVpv((*$2)[i],0);
            sv_2mortal(svs[i]);
        };
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in) (SymbolHandle *in)
{
    AV *tempav;
    int i;
    SV  **tv;
    int res;
    int len;
    if (!SvROK($input))
        croak("Argument $argnum is not a reference.");
        if (SvTYPE(SvRV($input)) != SVt_PVAV)
        croak("Argument $argnum is not an array.");
        tempav = (AV*)SvRV($input);
    len = av_len(tempav) + 1;
    if(len)
    {
        $1 = ($1_type)safemalloc(len*sizeof($*1_type));
        for (i = 0; i < len; i++) {
            tv = av_fetch(tempav, i, 0);    
            res = SWIG_ConvertPtr(*tv,SWIG_as_voidptrptr(&$1[i]), 0, 0);
            if (!SWIG_IsOK(res)) {
                SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "$*1_type""'"); 
            }
        }
    }
}

%typemap(freearg) (SymbolHandle *in) {
    Safefree($1);
}


%typemap(in,numinputs=0) (const char **real_name,
                          const char **description,
                          nn_uint *num_doc_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions,
                          const char **return_type
                          ) 
                          (char *name_temp, 
                           char *desc_temp, 
                           nn_uint num_args_temp, 
                           char **names_temp,
                           char **types_temp,
                           char **descs_temp,
                           char *return_temp
                          )
{
    $1 = &name_temp; 
    $2 = &desc_temp;
    $3 = &num_args_temp; 
    $4 = &names_temp;
    $5 = &types_temp;
    $6 = &descs_temp;
    $7 = &return_temp;
}
%typemap(argout) (const char **real_name,
                  const char **description,
                  nn_uint *num_doc_args,
                  const char ***arg_names,
                  const char ***arg_type_infos,
                  const char ***arg_descriptions,
                  const char **return_type
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

%typemap(in,numinputs=0) (OpHandle *out) (OpHandle temp), 
                         (SymbolHandle *out) (SymbolHandle temp),
                         (GraphHandle *out) (GraphHandle temp)
{
    $1 = &temp;
}
%typemap(argout) (OpHandle *out) 
{
    if(!result)
    {
        $result =  SWIG_NewPointerObj(SWIG_as_voidptr(*$1), $*1_descriptor, 0); argvi++;
    }
}

%typemap(argout) (SymbolHandle *out), (GraphHandle *out) 
{
    if(!result)
    {
        $result =  SWIG_NewPointerObj(SWIG_as_voidptr(*$1), $*1_descriptor, 0); argvi++;
    }
}

%typemap(in,numinputs=0) (nn_uint *out_size, OpHandle** out_array) (nn_uint temp_num, OpHandle* temp)
{
    $1 = &temp_num;
    *$1 = 0;
    $2 = &temp;
}
%typemap(argout) (nn_uint *out_size, OpHandle** out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_NNOp, 0);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in,numinputs=0) (nn_uint *out_size, SymbolHandle** out_array) (nn_uint temp_num, SymbolHandle* temp)
{
    $1 = &temp_num;
    *$1 = 0;
    $2 = &temp;
}
%typemap(argout) (nn_uint *out_size, SymbolHandle** out_array)
{
    if(!result)
    {
        AV *myav;
        SV **svs;
        int i = 0;
        svs = (SV **)safemalloc(*$1*sizeof(SV *));
        for (i = 0; i < *$1 ; i++) {
            svs[i] = SWIG_NewPointerObj(SWIG_as_voidptr((*$2)[i]), SWIGTYPE_p_NNSymbol, 0);
        }
        myav = av_make(*$1,svs);
        Safefree(svs);
        $result = newRV_noinc((SV*)myav);
        sv_2mortal($result);
        argvi++;
    }
}

%typemap(in) (SymbolHandle in)
{
    int res = SWIG_ConvertPtr($input,&$1, 0, 0);
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum"" of type '" "SymbolHandle""'"); 
    }
}
