#pragma once
#include<opencv2\opencv.hpp>
class PBASFeature
{
public:
	cv::Mat gradMag, pxIntensity;
	PBASFeature();
	PBASFeature(const PBASFeature &x);
	~PBASFeature();

	double getGradMagnMean() const;
	PBASFeature& PBASFeature::operator=(const PBASFeature& src);
	void PBASFeature::free();
	
};

