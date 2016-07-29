/*
 * detector.h using google-style
 *
 *  Created on: May 24, 2016
 *	  Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */

#pragma once

#include <common/fileutils.h>
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

class OpencvHOGDetctor {
 public:
  OpencvHOGDetctor() {}

  inline int det(std::string path) {
    LOG(INFO) << "det path : " << path;
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if (src_img.empty()) return 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    std::vector<cv::Rect> found, found_filtered;
    hog.detectMultiScale(src_img, found, 0, cv::Size(8, 8), cv::Size(32, 32),
                         1.05, 2);
    size_t i, j;
    for (i = 0; i < found.size(); i++) {
      cv::Rect r = found[i];
      for (j = 0; j < found.size(); j++)
        if (j != i && (r & found[j]) == r) break;
      if (j == found.size()) found_filtered.push_back(r);
    }

    for (i = 0; i < found_filtered.size(); i++) {
      cv::Rect r = found_filtered[i];
      r.x += cvRound(r.width * 0.1);
      r.width = cvRound(r.width * 0.8);
      r.y += cvRound(r.height * 0.06);
      r.height = cvRound(r.height * 0.9);
      cv::rectangle(src_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
    }
    mResultMat = src_img;
    // cv::imwrite(path, mResultMat);
    LOG(INFO) << "det ends";
    mRets = found_filtered;
    return found_filtered.size();
  }

  inline cv::Mat& getResultMat() { return mResultMat; }

  inline std::vector<cv::Rect>& getResult() { return mRets; }

 private:
  cv::Mat mResultMat;
  std::vector<cv::Rect> mRets;
};

class DLibHOGDetector {
 private:
  typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
  dlib::object_detector<image_scanner_type> mObjectDetector;

  inline void init() {
    LOG(INFO) << "Model Path: " << mModelPath;
    if (jnicommon::fileExists(mModelPath)) {
      dlib::deserialize(mModelPath) >> mObjectDetector;
    } else {
      LOG(INFO) << "Not exist " << mModelPath;
    }
  }

 public:

  DLibHOGDetector(const std::string& modelPath = "/sdcard/person.svm")
      : mModelPath(modelPath) {
    init();
  }

  virtual inline int det(const std::string& path) {
    using namespace jnicommon;
    if (!fileExists(mModelPath) || !fileExists(path)) {
      LOG(WARNING) << "No modle path or input file path";
      return 0;
    }
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if (src_img.empty()) return 0;
    int img_width = src_img.cols;
    int img_height = src_img.rows;
    int im_size_min = MIN(img_width, img_height);
    int im_size_max = MAX(img_width, img_height);

    float scale = float(INPUT_IMG_MIN_SIZE) / float(im_size_min);
    if (scale * im_size_max > INPUT_IMG_MAX_SIZE) {
      scale = (float)INPUT_IMG_MAX_SIZE / (float)im_size_max;
    }

    if (scale != 1.0) {
      cv::Mat outputMat;
      cv::resize(src_img, outputMat,
                 cv::Size(img_width * scale, img_height * scale));
      src_img = outputMat;
    }

    // cv::resize(src_img, src_img, cv::Size(320, 240));
    dlib::cv_image<dlib::bgr_pixel> cimg(src_img);

    double thresh = 0.5;
    mRets = mObjectDetector(cimg, thresh);
    return mRets.size();
  }

  inline std::vector<dlib::rectangle> getResult() { return mRets; }

 protected:
  std::vector<dlib::rectangle> mRets;
  std::string mModelPath;
  const int INPUT_IMG_MAX_SIZE = 800;
  const int INPUT_IMG_MIN_SIZE = 600;
};

/*
 * DLib face detect and face feature extractor
 */
class DLibHOGFaceDetector : public DLibHOGDetector {
 private:
  std::string mLandMarkModel;
  dlib::shape_predictor msp;
  std::unordered_map<int, dlib::full_object_detection> mFaceShapeMap;
  dlib::frontal_face_detector mFaceDetector;

  inline void init() {
    LOG(INFO) << "Init mFaceDetector";
    mFaceDetector = dlib::get_frontal_face_detector();
  }

 public:
  DLibHOGFaceDetector() { init(); }

  DLibHOGFaceDetector(const std::string& landmarkmodel)
      : mLandMarkModel(landmarkmodel) {
    init();
    if (!mLandMarkModel.empty() && jnicommon::fileExists(mLandMarkModel)) {
      dlib::deserialize(mLandMarkModel) >> msp;
      LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
    }
  }

  virtual inline int det(const std::string& path) {
    LOG(INFO) << "Read path from " << path;
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    return det(src_img);
  }

  // The format of mat should be BGR
  virtual inline int det(const cv::Mat& image) {
    if (image.empty()) return 0;
    LOG(INFO) << "com_tzutalin_dlib_PeopleDet go to det(mat)";
    if (image.channels() == 4) {
      cv::cvtColor(image, image, CV_BGRA2BGR);
    } else if (image.channels() == 1) {
      cv::cvtColor(image, image, CV_GRAY2BGR);
    }
    // TODO : Convert to gray image to speed up detection
    // It's unnecessary to use color image for face/landmark detection
    dlib::cv_image<dlib::bgr_pixel> img(image);
    mRets = mFaceDetector(img);
    LOG(INFO) << "Dlib HOG face det size : " << mRets.size();
    mFaceShapeMap.clear();
    // Process shape
    if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
      for (unsigned long j = 0; j < mRets.size(); ++j) {
        dlib::full_object_detection shape = msp(img, mRets[j]);
        LOG(INFO) << "face index:" << j
                  << "number of parts: " << shape.num_parts();
        mFaceShapeMap[j] = shape;
      }
    }
    return mRets.size();
  }
/*
  // The format of mat should be BGR
  virtual inline int det(const cv::Mat& image, cv::Mat& transformedImage) {
	LOG(INFO) << "com_tzutalin_dlib_PeopleDet go to det(mat)";

	if (mLandMarkModel.empty()) {
		LOG(INFO) << "com_tzutalin_dlib_PeopleDet Landmark Model must be initialized. Bailing...";
		return 0;
	}

	LOG(INFO) << "com_tzutalin_dlib_PeopleDet calling dlib::get_frontal_face_detector";
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	LOG(INFO) << "com_tzutalin_dlib_PeopleDet got frontal face detector";
	dlib::cv_image<dlib::bgr_pixel> img(image);
	LOG(INFO) << "com_tzutalin_dlib_PeopleDet instantiated dv_image with passed in image";
  	std::vector<dlib::rectangle> firstPassRets;
	firstPassRets = detector(img);
	LOG(INFO) << "Dlib HOG face det size (first pass): " << firstPassRets.size();
	mFaceShapeMap.clear();

	// 1. Face outline
	// 2. Right eyebrow
	// 3. Left eyebrow
	// 4. Nose bridge
	// 5. (4a?) Nostrils
	// 6. Right Eye
	// 7. Left Eye
	// 8. Mouth

	cv::Mat centeredImage;
	dlib::cv_image<dlib::bgr_pixel> centeredImg;
	transformedImage = cv::Mat::zeros( image.rows, image.cols, image.type() );

	if (firstPassRets.size() == 0) {
		return 0;
	}

	for (unsigned long j = 0; j < firstPassRets.size(); ++j) {
		dlib::full_object_detection shape = msp(img, firstPassRets[j]);
		LOG(INFO) << "face index (input):" << j
				  << " number of parts: " << shape.num_parts();

		// New code to try to normalize with Affine Transformation
		// 1) Make the point of the nose always be in the exact center - point index 30
		// 2) Make the y position of the inner point of both eyes the same - right eye: 39, left eye: 42
		// Note: will need to convert the transformed image back for display (maybe)
		dlib::point tipOfNose = shape.part(30);
		dlib::point innerRightEye = shape.part(39);
		dlib::point innerLeftEye = shape.part(42);

		cv::Point2f srcTri[2];
	    // srcTri[0] = cv::Point2f( tipOfNose.x(), tipOfNose.y() );
		srcTri[0] = cv::Point2f( innerRightEye.x(), innerRightEye.y() );
		srcTri[1] = cv::Point2f( innerLeftEye.x(), innerLeftEye.y() );

		// Figure out destination for the transform the src point transformation
		cv::Size s = image.size();
		cv::Point2f dstTri[2];
	    // dstTri[0] = cv::Point2f( s.width*0.5, s.height*0.5 );
		dstTri[0] = cv::Point2f( innerRightEye.x(), innerRightEye.y() );
		dstTri[1] = cv::Point2f( innerLeftEye.x(), innerRightEye.y() );

		centeredImage = cv::Mat::zeros( image.rows, image.cols, image.type() );

		// cv::Mat center_warp_mat( 2, 3, CV_32FC1 );
		// center_warp_mat = cv::getAffineTransform( srcTri, dstTri );
		// cv::warpAffine( image, centeredImage, center_warp_mat, centeredImage.size() );

		if (innerRightEye.y() != innerLeftEye.y()) {
			cv::Point2f center = cv::Point2f( tipOfNose.x(), tipOfNose.y() );
			double angle = atan( static_cast<double>(innerLeftEye.y() - innerRightEye.y()) / static_cast<double>(innerLeftEye.x() - innerRightEye.x()) );
			LOG(INFO) << "com_tzutalin_dlib_PeopleDet need to rotate " << angle << " deg, horizontal distance: " << innerLeftEye.x() - innerRightEye.x();
			double scale = 1.0;

			cv::Mat rotation_mat( 2, 3, CV_32FC1 );
			rotation_mat = getRotationMatrix2D( center, angle, scale );
			cv::warpAffine( image, centeredImage, rotation_mat, centeredImage.size() );

			// TODO: lots of buffers - fix it
			LOG(INFO) << "com_tzutalin_dlib_PeopleDet instantiated dv_image with centered image";
			centeredImg = centeredImage;
			mRets = detector(centeredImg);
			LOG(INFO) << "Dlib HOG face det size (center pass): " << mRets.size();
		}
		else {
			LOG(INFO) << "eyes already aligned. Done.";
			mRets = firstPassRets;
		}
	}
	// ^ why the iteration when the model only returns one flattened shape?
	// 07-28 16:02:55.036 17647-24099/com.boompayments.boombox I/native: I/detector.h:181 face index:0number of parts: 68

	if (mRets.size() == 0) {
		return 0;
	}

	for (unsigned long j = 0; j < mRets.size(); ++j) {
		dlib::full_object_detection shape = msp(centeredImg, mRets[j]);
		LOG(INFO) << "face index (centered):" << j
				  << " number of parts: " << shape.num_parts();
		mFaceShapeMap[j] = shape;
	}

	transformedImage = centeredImage; // update the variable passed by ref to be the centered one
	return mRets.size();
*/
/********************************************************************
	cv::Mat centeredImage;
	dlib::cv_image<dlib::bgr_pixel> centeredImg;
 	std::vector<dlib::rectangle> centerPassRets;

	if (firstPassRets.size() == 0) {
		return 0;
	}

	for (unsigned long j = 0; j < firstPassRets.size(); ++j) {
		dlib::full_object_detection shape = msp(img, firstPassRets[j]);
		LOG(INFO) << "face index (input):" << j
				  << " number of parts: " << shape.num_parts();

		// New code to try to normalize with Affine Transformation
		// 1) Make the point of the nose always be in the exact center - point index 30
		// 2) Make the y position of the inner point of both eyes the same - right eye: 39, left eye: 42
		// Note: will need to convert the transformed image back for display (maybe)
		dlib::point tipOfNose = shape.part(30);
		cv::Point2f srcNose[1];
	    srcNose[0] = cv::Point2f( tipOfNose.x(), tipOfNose.y() );

		// Figure out destination for the transform the src point transformation
		cv::Size s = image.size();
		cv::Point2f dstNose[1];
	    dstNose[0] = cv::Point2f( s.width*0.5, s.height*0.5 );

		cv::Mat center_warp_mat( 2, 3, CV_32FC1 );
		center_warp_mat = cv::getAffineTransform( srcNose, dstNose );

		centeredImage = cv::Mat::zeros( image.rows, image.cols, image.type() );
		cv::warpAffine( image, centeredImage, center_warp_mat, centeredImage.size() );

		// TODO: lots of buffers - fix it
		LOG(INFO) << "com_tzutalin_dlib_PeopleDet instantiated dv_image with centered image";
		centeredImg = centeredImage;
		centerPassRets = detector(centeredImg);
		LOG(INFO) << "Dlib HOG face det size (center pass): " << centerPassRets.size();
	}
	// ^ why the iteration when the model only returns one flattened shape?
	// 07-28 16:02:55.036 17647-24099/com.boompayments.boombox I/native: I/detector.h:181 face index:0number of parts: 68

	if (centerPassRets.size() == 0) {
		return 0;
	}

	cv::Mat alignedImage;
	dlib::cv_image<dlib::bgr_pixel> alignedImg;

	for (unsigned long j = 0; j < centerPassRets.size(); ++j) {
		dlib::full_object_detection shape = msp(centeredImg, centerPassRets[j]);
		LOG(INFO) << "face index (centered):" << j
				  << " number of parts: " << shape.num_parts();

		dlib::point innerRightEye = shape.part(39);
		dlib::point innerLeftEye = shape.part(42);
		cv::Point2f srcEyes[2];
		srcEyes[0] = cv::Point2f( innerRightEye.x(), innerRightEye.y() );
		srcEyes[1] = cv::Point2f( innerLeftEye.x(), innerLeftEye.y() );

		cv::Point2f dstEyes[2];
		dstEyes[0] = cv::Point2f( innerRightEye.x(), innerRightEye.y() );
		dstEyes[1] = cv::Point2f( innerLeftEye.x(), innerRightEye.y() );

		cv::Mat align_eyes_warp_mat( 2, 3, CV_32FC1 );
		align_eyes_warp_mat = cv::getAffineTransform( srcEyes, dstEyes );

		alignedImage = cv::Mat::zeros( centeredImage.rows, centeredImage.cols, centeredImage.type() );
		cv::warpAffine( centeredImage, alignedImage, align_eyes_warp_mat, alignedImage.size() );

		// TODO: lots more buffers - fix it
		LOG(INFO) << "com_tzutalin_dlib_PeopleDet instantiated dv_image with aligned image";
		alignedImg = alignedImage;
		mRets = detector(alignedImg);
		LOG(INFO) << "Dlib HOG face det size (alignment pass): " << mRets.size();
	}

	if (mRets.size() == 0) {
		return 0;
	}

	for (unsigned long j = 0; j < mRets.size(); ++j) {
		dlib::full_object_detection shape = msp(alignedImg, mRets[j]);
		LOG(INFO) << "face index (aligned):" << j
				  << " number of parts: " << shape.num_parts();
		mFaceShapeMap[j] = shape;
	}

	image = alignedImage; // update the variable passed by ref to be the aligned one
	return mRets.size();
*****************************************************************/
/*
  }
  */

  std::unordered_map<int, dlib::full_object_detection>& getFaceShapeMap() {
	return mFaceShapeMap;
  }
};
