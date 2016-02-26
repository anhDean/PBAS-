#pragma once
#include<opencv2\opencv.hpp>
class PBASFeature
{
public:
	cv::Mat gradMag, gradAngle, pxIntensity;
	PBASFeature();
	PBASFeature(const PBASFeature &x);
	~PBASFeature();

	void PBASFeature::operator=(const PBASFeature& src);
	void PBASFeature::free();
	
};

