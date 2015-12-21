#include <jni.h>

#ifndef MXNET_SCALA_JNI_HELPER_FUNC_H
#define MXNET_SCALA_JNI_HELPER_FUNC_H

jlong getLongField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  return env->GetLongField(obj, refFid);
}

jint getIntField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  return env->GetIntField(obj, refFid);
}

void setIntField(JNIEnv *env, jobject obj, jint value) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  env->SetIntField(obj, refFid, value);
}

#endif
