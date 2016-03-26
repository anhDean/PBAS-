#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include<set>

struct AmbiguousCandidate
{
	cv::Rect boundingBox;
	int counter;
	float accMovement; // accumulated movement
	int state;
};

const int DETECTOR_STATIC_OBJECT = 0;
const int DETECTOR_GHOST_OBJECT = 1;
const int DETECTOR_MOVING_OBJECT = 2;
const int DETECTOR_STATIC_BACKGROUND = 3;
const int DETECTOR_DYNAMIC_BACKGROUND = 4;
const int DETECTOR_UNCLASSIFIED_OBJECT = 5;


class MotionDetector
{
	private:
		static const int m_MIN_BLOBSIZE = 15*15;
		static const int m_MAX_STATIC_OBJ = 3;
		static const int m_MAX_MOVING_OBJ = 10;
		
		cv::Mat m_prev_grey;
		cv::Mat m_staticBg, m_staticObjectArea, m_dynamicBg, m_movingForeground;
		cv::Mat tmp_candidates;
		cv::Mat m_motionMap;
		std::vector<AmbiguousCandidate> m_staticObjectCandidates;


		bool evaluateStaticObjRect(cv::Rect& tmp_rect);
		bool isInMotion(cv::Rect boundingBox, float interSection = 0.15);

		//std::vector<AmbiguousCandidate> m_movingObjCandidates;
		//bool evaluateMovingObjRect(cv::Rect& tmp_rect);
		//void verifyMovingCandidates(const cv::Mat& segmentationMap);

	protected:

	public:

		void initialize(const cv::Mat& firstFrame);
		void calcOpticalFlow(const cv::Mat& input, cv::Mat& output);

		void updateMotionMap(const cv::Mat& input);
		void updateCandidates(const cv::Mat& segmentation);
		void verifyStaticCandidates(const cv::Mat& segMap);
		void classifyStaticCandidates(const cv::Mat& inputROI, const cv::Mat& bgModelROI, const cv::Mat& segROI);
		
		const cv::Mat& getMotionMap() const;
		const cv::Mat& getStaticObjectArea() const;
		const std::vector<AmbiguousCandidate>& getStaticObjects() const;
		//const std::vector<AmbiguousCandidate>& getMovingObjects() const;
		//void classifyMovingCandidates(const cv::Mat& segROI, const cv::Mat noiseMap);

		static cv::Mat opticalFlowToAbs(const cv::Mat& optical_flow);
		static std::vector<std::vector<cv::Point> > getContours(const cv::Mat& input);

		static bool containsBoundingBox(cv::Rect outer, cv::Rect inner);

		static void drawBoundingBoxes(const cv::Mat& input, cv::Mat& output, const std::vector<cv::Rect>& bb);
		static void drawBoundingBoxesClassified(const cv::Mat& input, cv::Mat& output, const std::vector<AmbiguousCandidate>& bb);
		static cv::Mat drawFlowField(const cv::Mat& input, const cv::Mat& flowMat);
		static cv::Mat drawContours(const cv::Mat& input);

};

