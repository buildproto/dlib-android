/*
 * hogdetector.h using google-style
 *
 *  Created on: May 24, 2016
 *      Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */

#pragma once

#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_loader/load_image.h>
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

class DLibHOGDetector {
 public:
  // Default svm path is /sdcard/person.svm if it exists
  DLibHOGDetector(std::string modelPath = "/sdcard/person.svm")
      : mModelPath(modelPath),
        mBeLoadedModel(false) {
  }

  inline int det(const cv::Mat& image) {
    typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
    dlib::object_detector<image_scanner_type> detector;
    if (mBeLoadedModel == false) {
      dlib::deserialize(mModelPath) >> detector;
      mBeLoadedModel = true;
    }

    dlib::cv_image<dlib::bgr_pixel> cimg(image);
    double thresh = 0.5;
    mRets = detector(cimg, thresh);
    return mRets.size();
  }

  inline std::vector<dlib::rectangle> getResult() {
    return mRets;
  }

 private:
  std::vector<dlib::rectangle> mRets;
  std::string mModelPath;
  bool mBeLoadedModel;
};

/*
 * DLib face detect and face feature extractor
 */
class DLibHOGFaceDetector {
 public:
  DLibHOGFaceDetector(std::string landmarkmodel = "")
      : mLandMarkModel(landmarkmodel),
        mBeLoadedLandmark(false) {
  }

  // The format of mat should be BGR
  inline int det(const cv::Mat& image) {
    LOG(INFO) << "com_tzutalin_dlib_PeopleDet go to det(mat)";
    if (!mLandMarkModel.empty() && mBeLoadedLandmark == false) {
      dlib::deserialize(mLandMarkModel) >> msp;
      LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
      mBeLoadedLandmark = true;
    }

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::cv_image<dlib::bgr_pixel> img(image);
    mRets = detector(img);
    LOG(INFO) << "Dlib HOG face det size : " << mRets.size();
    mFaceShapeMap.clear();

    if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
      for (unsigned long j = 0; j < mRets.size(); ++j) {
        dlib::full_object_detection shape = msp(img, mRets[j]);
        LOG(INFO) << "face index:" << j << "number of parts: " << shape.num_parts();
        mFaceShapeMap[j] = shape;
      }
    }

    return mRets.size();
  }

  inline std::unordered_map<int, dlib::full_object_detection>& getFaceShapeMap() {
    return mFaceShapeMap;
  }

  inline std::vector<dlib::rectangle> getResult() {
    return mRets;
  }

 private:
  std::vector<dlib::rectangle> mRets;
  std::string mLandMarkModel;
  dlib::shape_predictor msp;
  std::unordered_map<int, dlib::full_object_detection> mFaceShapeMap;
  bool mBeLoadedLandmark;
};
