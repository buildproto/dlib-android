/*
 * jni_hogdetector.cpp using google-style
 *
 *  Created on: Oct 20, 2015
 *      Author: Tzutalin
 *
 *  Copyright (c) 2015 Tzutalin. All rights reserved.
 */
#include <android/bitmap.h>
#include <dlib/image_io.h>
#include <dlib/image_keypoint.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <glog/logging.h>
#include <jni.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif

// ========================================================
// JNI Mapping Methods
// ========================================================

#define DLIB_JNI_METHOD(METHOD_NAME) \
  Java_com_tzutalin_dlib_Surf_##METHOD_NAME

void JNIEXPORT DLIB_JNI_METHOD(jniNativeClassInit)(JNIEnv* _env, jclass _this) {
}

jint JNIEXPORTJNICALL
DLIB_JNI_METHOD(jniSurf)(JNIEnv* env, jobject thiz,
    jstring imgPath) {
  LOG(INFO) << "com_tzutalin_dlib_Surf jniSurf";
  const char* img_path = env->GetStringUTFChars(imgPath, 0);

  cv::Mat imageMat = cv::imread(std::string(img_path), 1);
  dlib::cv_image<dlib::bgr_pixel> img(imageMat);
  std::vector<dlib::surf_point> sp = dlib::get_surf_points(img);
  LOG(INFO) << "Dlib surf_point det size : " << sp.size();

  env->ReleaseStringUTFChars(imgPath, img_path);
  return sp.size();
}

jint JNIEXPORTJNICALL DLIB_JNI_METHOD(jniInit)(JNIEnv* env, jobject thiz) {
  return JNI_OK;
}

jint JNIEXPORTJNICALL DLIB_JNI_METHOD(jniDeInit)(JNIEnv* env, jobject thiz) {
  LOG(INFO) << "jniDeInit";
  return JNI_OK;
}

#ifdef __cplusplus
}
#endif
