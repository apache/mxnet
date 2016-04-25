/*!
 *  Copyright (c) 2015 by Contributors
 * \file jni_helper_func.h
 * \brief Helper functions for operating JVM objects
 */
#include <jni.h>

#ifndef MXNET_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
#define MXNET_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_

jlong GetLongField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  jlong ret = env->GetLongField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

jint GetIntField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  jint ret = env->GetIntField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

void SetIntField(JNIEnv *env, jobject obj, jint value) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  env->SetIntField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void SetLongField(JNIEnv *env, jobject obj, jlong value) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  env->SetLongField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void SetStringField(JNIEnv *env, jobject obj, const char *value) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefString");
  jfieldID refFid = env->GetFieldID(refClass, "value", "Ljava/lang/String;");
  env->SetObjectField(obj, refFid, env->NewStringUTF(value));
  env->DeleteLocalRef(refClass);
}
#endif  // MXNET_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
