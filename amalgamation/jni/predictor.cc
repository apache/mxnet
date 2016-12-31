#include <jni.h>
#include "org_dmlc_mxnet_Predictor.h"

#include "../mxnet_predict-all.cc"

JNIEXPORT jlong JNICALL Java_org_dmlc_mxnet_Predictor_createPredictor
  (JNIEnv *env, jclass, jbyteArray jsymbol, jbyteArray jparams, jint devType, jint devId, jobjectArray jkeys, jobjectArray jshapes)
{
	jbyte* symbol = env->GetByteArrayElements(jsymbol, 0);
	jbyte* params = env->GetByteArrayElements(jparams, 0);
	jsize params_len = env->GetArrayLength(jparams);

	std::vector<std::pair<jstring, const char *>> track;
	std::vector<const char *> keys;
    for (int i=0; i<env->GetArrayLength(jkeys); i++) {
        jstring js = (jstring) env->GetObjectArrayElement(jkeys, i);
        const char *s = env->GetStringUTFChars(js, 0);
		keys.emplace_back(s);
		track.emplace_back(js, s);
    }

	std::vector<mx_uint> index{0};
	std::vector<mx_uint> shapes;
    for (int i=0; i<env->GetArrayLength(jshapes); i++) {
        jintArray jshape = (jintArray) env->GetObjectArrayElement(jshapes, i);
		jsize shape_len = env->GetArrayLength(jshape);
		jint *shape = env->GetIntArrayElements(jshape, 0);

		index.emplace_back(shape_len);
		for (int j=0; j<shape_len; ++j) shapes.emplace_back((mx_uint)shape[j]);
		env->ReleaseIntArrayElements(jshape, shape, 0);
    }

	PredictorHandle handle = 0;	
	if (MXPredCreate((const char *)symbol, (const char *)params, params_len, devType, devId, (mx_uint)keys.size(), &(keys[0]), &(index[0]), &(shapes[0]), &handle) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}

	env->ReleaseByteArrayElements(jsymbol, symbol, 0); 
	env->ReleaseByteArrayElements(jparams, params, 0); 
	for (auto& t: track) {
		env->ReleaseStringUTFChars(t.first, t.second);
	}

	return (jlong)handle;
}

JNIEXPORT void JNICALL Java_org_dmlc_mxnet_Predictor_nativeFree
  (JNIEnv *, jclass, jlong h)
{
	PredictorHandle handle = (PredictorHandle)h;	
	MXPredFree(handle);
}

JNIEXPORT jfloatArray JNICALL Java_org_dmlc_mxnet_Predictor_nativeGetOutput
  (JNIEnv *env, jclass, jlong h, jint index)
{
	PredictorHandle handle = (PredictorHandle)h;	

	mx_uint *shape = 0;
	mx_uint shape_len;
	if (MXPredGetOutputShape(handle, index, &shape, &shape_len) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}

	size_t size = 1;
	for (mx_uint i=0; i<shape_len; ++i) size *= shape[i];

	std::vector<float> data(size);
	if (MXPredGetOutput(handle, index, &(data[0]), size) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}
	
	jfloatArray joutput = env->NewFloatArray(size);
    jfloat *out = env->GetFloatArrayElements(joutput, NULL);

    for (int i=0; i<size; i++) out[i] = data[i];
    env->ReleaseFloatArrayElements(joutput, out, 0);

	return joutput;
}

JNIEXPORT void JNICALL Java_org_dmlc_mxnet_Predictor_nativeForward
  (JNIEnv *env, jclass, jlong h, jstring jkey, jfloatArray jinput)
{
	PredictorHandle handle = (PredictorHandle)h;	
	const char *key = env->GetStringUTFChars(jkey, 0);
	jfloat* input = env->GetFloatArrayElements(jinput, 0);
	jsize input_len = env->GetArrayLength(jinput);

	if (MXPredSetInput(handle, key, input, input_len) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}

	env->ReleaseStringUTFChars(jkey, key);
	env->ReleaseFloatArrayElements(jinput, input, 0);
	if (MXPredForward(handle) < 0) {
		jclass MxnetException = env->FindClass("org/dmlc/mxnet/MxnetException");
		env->ThrowNew(MxnetException, MXGetLastError());
	}
}


