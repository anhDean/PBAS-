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
	std::string createString();
	void defineVideoParamters();
	std::vector<std::string> setImageString(std::string base, int numberOfDigits);
	//input / output videos
	std::vector<std::string> vidString,setOutputPath, setVidPath, setOutputVideo,baselineString;
	//homography
	std::vector<int> setScene;

	//own objectDetection
	std::vector<double> dist2Center,overlapP,speedChange, distanceForRect, minContourLength, maxContourLength, distanceForFeatures, distanceForMovement, resParam;
	
	//pbas


	//graphcuts
	std::vector<double> newAlpha, newR, newBeta, newConstForeground, 	newConstBackground, labelThresh;
	std::vector<int> neighorThresh;

	std::vector<bool> useGraphCuts;
	
	//param string
	std::vector<std::string> outputBaseline;
	int numberOfParams;

	std::vector<std::vector<std::string>> imgString;
	FeatureTracker* tracker;
	// Create instance
	VideoProcessor* processor;
	int numberOfVideos;


	//PBAS Parameters
	std::vector<int> newN, newTemporal, newRaute, lowerTimeUpdateRateBoundary, upperTimeUpdateRateBoundary;
	std::vector<double>	rThreshScale, rIncDecFac, increasingRateScale, decreasingRateScale;

};
