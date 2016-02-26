#pragma once
#include "VideoProcessor.h"
#include "FeatureTracker.h"
#include <string>
#include <ppl.h>
#include <windows.h>

class MainLoop
{
public:
	MainLoop(void);
	~MainLoop(void);
	void startVideoProcessing();


private:
	int m_nParams, m_nVideos;
	//own objectDetection
	std::vector<double> dist2Center, overlapP, speedChange, distanceForRect, minContourLength, maxContourLength, distanceForFeatures, distanceForMovement, resParam;
	//input / output videos
	std::vector<std::string> vidString, setOutputPath, setVidPath, setOutputVideo, baselineString;

	std::vector<std::string> setImageString(std::string base, int numberOfDigits);
	
	std::string createString();
	void defineVideoParamters();
	//pbas

	//graphcuts
	std::vector<double> newAlpha, newR, newBeta, newConstForeground, 	newConstBackground, labelThresh;
	std::vector<int> neighorThresh;

	//param string
	std::vector<std::string> outputBaseline;
	std::vector<std::vector<std::string>> imgString;
	
	//PBAS Parameters
	std::vector<int> newN, newTemporal, newRaute, lowerTimeUpdateRateBoundary, upperTimeUpdateRateBoundary;
	std::vector<double>	rThreshScale, rIncDecFac, increasingRateScale, decreasingRateScale;

};
