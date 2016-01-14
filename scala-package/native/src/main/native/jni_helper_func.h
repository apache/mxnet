#include <jni.h>

#ifndef MXNET_SCALA_JNI_HELPER_FUNC_H
#define MXNET_SCALA_JNI_HELPER_FUNC_H

// Define an env closure
// e.g. it can be used to implement java callback
typedef struct {
  JNIEnv *env;
  jobject obj;
} JNIClosure;

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

void setLongField(JNIEnv *env, jobject obj, jlong value) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  env->SetLongField(obj, refFid, value);
}

void setStringField(JNIEnv *env, jobject obj, const char *value) {
  jclass refClass = env->FindClass("ml/dmlc/mxnet/Base$RefString");
  jfieldID refFid = env->GetFieldID(refClass, "value", "Ljava/lang/String;");
  env->SetObjectField(obj, refFid, env->NewStringUTF(value));
}
#endif
