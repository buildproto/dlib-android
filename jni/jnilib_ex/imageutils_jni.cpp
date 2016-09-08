/*
 * imageutils_jni.cpp using google-style
 *
 *  Created on: May 24, 2016
 *      Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */
#include <common/bitmap2mat.h>
#include <common/rgb2yuv.h>
#include <common/types.h>
#include <common/yuv2rgb.h>
#include <glog/logging.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace jnicommon;

#define IMAGEUTILS_METHOD(METHOD_NAME) \
  Java_com_tzutalin_dlibtest_ImageUtils_##METHOD_NAME  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL IMAGEUTILS_METHOD(convertYUV420SPToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray input, jintArray output, jint width,
    jint height, jboolean halfSize);

JNIEXPORT void JNICALL IMAGEUTILS_METHOD(convertYUV420ToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray y, jbyteArray u, jbyteArray v,
    jintArray output, jint width, jint height, jint y_row_stride,
    jint uv_row_stride, jint uv_pixel_stride, jboolean halfSize);

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(convertYUV420SPToRGB565)(JNIEnv* env, jclass clazz,
                                               jbyteArray input,
                                               jbyteArray output, jint width,
                                               jint height);

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(convertARGB8888ToYUV420SP)(JNIEnv* env, jclass clazz,
                                                 jintArray input,
                                                 jbyteArray output, jint width,
                                                 jint height);

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(convertRGB565ToYUV420SP)(JNIEnv* env, jclass clazz,
                                               jbyteArray input,
                                               jbyteArray output, jint width,
                                               jint height);

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(normalizeCapturedImage)(JNIEnv* env, jclass clazz,
                                              jobject input, jobject detectedPoints,
                                              jintArray output, jint width,
                                              jint height);

#ifdef __cplusplus
}
#endif

JNIEXPORT void JNICALL IMAGEUTILS_METHOD(convertYUV420SPToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray input, jintArray output, jint width,
    jint height, jboolean halfSize) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jint* const o = env->GetIntArrayElements(output, &outputCopy);

  if (halfSize) {
    ConvertYUV420SPToARGB8888HalfSize(reinterpret_cast<uint8*>(i),
                                      reinterpret_cast<uint32*>(o), width,
                                      height);
  } else {
    ConvertYUV420SPToARGB8888(reinterpret_cast<uint8*>(i),
                              reinterpret_cast<uint8*>(i) + width * height,
                              reinterpret_cast<uint32*>(o), width, height);
  }

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseIntArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL IMAGEUTILS_METHOD(convertYUV420ToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray y, jbyteArray u, jbyteArray v,
    jintArray output, jint width, jint height, jint y_row_stride,
    jint uv_row_stride, jint uv_pixel_stride, jboolean halfSize) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const y_buff = env->GetByteArrayElements(y, &inputCopy);
  jboolean outputCopy = JNI_FALSE;
  jint* const o = env->GetIntArrayElements(output, &outputCopy);

  if (halfSize) {
    ConvertYUV420SPToARGB8888HalfSize(reinterpret_cast<uint8*>(y_buff),
                                      reinterpret_cast<uint32*>(o), width,
                                      height);
  } else {
    jbyte* const u_buff = env->GetByteArrayElements(u, &inputCopy);
    jbyte* const v_buff = env->GetByteArrayElements(v, &inputCopy);

    ConvertYUV420ToARGB8888(
        reinterpret_cast<uint8*>(y_buff), reinterpret_cast<uint8*>(u_buff),
        reinterpret_cast<uint8*>(v_buff), reinterpret_cast<uint32*>(o), width,
        height, y_row_stride, uv_row_stride, uv_pixel_stride);

    env->ReleaseByteArrayElements(u, u_buff, JNI_ABORT);
    env->ReleaseByteArrayElements(v, v_buff, JNI_ABORT);
  }

  env->ReleaseByteArrayElements(y, y_buff, JNI_ABORT);
  env->ReleaseIntArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(convertYUV420SPToRGB565)(JNIEnv* env, jclass clazz,
                                               jbyteArray input,
                                               jbyteArray output, jint width,
                                               jint height) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertYUV420SPToRGB565(reinterpret_cast<uint8*>(i),
                          reinterpret_cast<uint16*>(o), width, height);

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(convertARGB8888ToYUV420SP)(JNIEnv* env, jclass clazz,
                                                 jintArray input,
                                                 jbyteArray output, jint width,
                                                 jint height) {
  jboolean inputCopy = JNI_FALSE;
  jint* const i = env->GetIntArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertARGB8888ToYUV420SP(reinterpret_cast<uint32*>(i),
                            reinterpret_cast<uint8*>(o), width, height);

  env->ReleaseIntArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(convertRGB565ToYUV420SP)(JNIEnv* env, jclass clazz,
                                               jbyteArray input,
                                               jbyteArray output, jint width,
                                               jint height) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertRGB565ToYUV420SP(reinterpret_cast<uint16*>(i),
                          reinterpret_cast<uint8*>(o), width, height);

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
    IMAGEUTILS_METHOD(normalizeCapturedImage)(JNIEnv* env, jclass clazz,
                                              jobject input, jobject detectedPoints,
                                              jintArray output, jint width,
                                              jint height) {
  /*
  jboolean inputCopy = JNI_FALSE;
  jbyte* const jniInput = env->GetByteArrayElements(input, &inputCopy);
  if (inputCopy == JNI_TRUE) {
    LOG(INFO) << "{normalizeCapturedImage} had to create a COPY of input image byte array";
  }
  jboolean outputCopy = JNI_FALSE;
  jbyte* const jniOutput = env->GetByteArrayElements(output, &outputCopy);
  if (outputCopy == JNI_TRUE) {
    LOG(INFO) << "{normalizeCapturedImage} had to create a COPY of output image byte array";
  }
  */
  jboolean outputCopy = JNI_FALSE;
  jint* const o = env->GetIntArrayElements(output, &outputCopy);
  if (outputCopy == JNI_TRUE) {
    LOG(INFO) << "{normalizeCapturedImage} had to create a COPY of output image int array";
  }
  uint32* outputBuffer = reinterpret_cast<uint32*>(o);
  
  cv::Mat rgbaMat;
  // cv::Mat brgMat; // for greyscale? maybe unnecessary?
  jnicommon::ConvertBitmapToRGBAMat(env, input, rgbaMat, true);
  LOG(INFO) << "{normalizeCapturedImage} Input bitmap converted to RGBA cv::Mat";

  jclass cArrayList = env->GetObjectClass(detectedPoints);
  LOG(INFO) << "{normalizeCapturedImage} got ArrayList jclass, look up <E get(int)> method next...";
  jmethodID method_ArrayList_get = env->GetMethodID(cArrayList, "get", "(I)Ljava/lang/Object;");
  LOG(INFO) << "{normalizeCapturedImage} got ArrayList get(int) jmethodID";

  jobject jCenterPoint = env->CallObjectMethod(detectedPoints, method_ArrayList_get, 30);
  LOG(INFO) << "{normalizeCapturedImage} got jCenterPoint jobject instance";

  jclass cPoint = env->GetObjectClass(jCenterPoint); // env->FindClass("android/graphics/Point");
  LOG(INFO) << "{normalizeCapturedImage} got jCenterPoint jclass";
  jfieldID field_Point_x = env->GetFieldID(cPoint, "x", "I");
  LOG(INFO) << "{normalizeCapturedImage} got cPoint x jfieldID";
  if (field_Point_x == NULL) {
    LOG(ERROR) << "{normalizeCapturedImage} Failed to get Java <int Point.x> fieldID";
    return;
  }
  jfieldID field_Point_y = env->GetFieldID(cPoint, "y", "I");
  LOG(INFO) << "{normalizeCapturedImage} got cPoint y jfieldID";
  if (field_Point_y == NULL) {
    LOG(ERROR) << "{normalizeCapturedImage} Failed to get Java <int Point.y> fieldID";
    return;
  }

  int center_x = env->GetIntField(jCenterPoint, field_Point_x);
  LOG(INFO) << "{normalizeCapturedImage} got jCenterPoint object x member variable int value";
  int center_y = env->GetIntField(jCenterPoint, field_Point_y);
  LOG(INFO) << "{normalizeCapturedImage} center of face: (" << center_x << ", " << center_y << ")";

  cv::Mat centeredMat = cv::Mat::zeros( rgbaMat.rows, rgbaMat.cols, rgbaMat.type() );
  LOG(INFO) << "{normalizeCapturedImage} initialized zeroed-out cv::Mat";
  cv::Mat centerWarpMat( 2, 3, CV_32FC1 );
  cv::Point2f src[1];
  src[0] = cv::Point2f(center_x, center_y);
  cv::Point2f dst[1];
  dst[0] = cv::Point2f((rgbaMat.rows*0.5), (rgbaMat.cols*0.5));
  centerWarpMat = cv::getAffineTransform( src, dst );
  LOG(INFO) << "{normalizeCapturedImage} got centering Affine transform";
  cv::warpAffine( rgbaMat, centeredMat, centerWarpMat, centeredMat.size() );
  LOG(INFO) << "{normalizeCapturedImage} performed Affine transform";

  /*
  std::vector<int> array;
  if (centeredMat.isContinuous()) {
    LOG(INFO) << "{normalizeCapturedImage} centered cv::Mat is continuous, assigning directly to array";
    array.assign(centeredMat.datastart, centeredMat.dataend);
  } else {
    LOG(INFO) << "{normalizeCapturedImage} flattening centered cv::Mat row by row";
    for (int i = 0; i < centeredMat.rows; ++i) {
      LOG(INFO) << "{normalizeCapturedImage} adding row " << i << " to array";
      array.insert(array.end(), centeredMat.ptr<int>(i), centeredMat.ptr<int>(i)+centeredMat.cols);
    }
  }

  LOG(INFO) << "{normalizeCapturedImage} copying array into pre-allocated output int[], " << array.size() << " bytes";
  memcpy(&array[0], output, array.size());
  */

  for (unsigned int pos = 0, len = width*height; pos < len; ++pos) {
      *outputBuffer++ = 8453889;
  }

  /*
  env->ReleaseByteArrayElements(input, jniInput, JNI_ABORT);
  env->ReleaseByteArrayElements(output, jniOutput, 0);
  */
  env->ReleaseIntArrayElements(output, o, 0);
}
