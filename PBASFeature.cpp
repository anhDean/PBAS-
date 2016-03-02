#include "PBASFeature.h"


PBASFeature::PBASFeature(const PBASFeature &x)
{
	gradMag = x.gradMag.clone();
	pxIntensity = x.pxIntensity.clone();
}
PBASFeature& PBASFeature::operator=(const PBASFeature& src)
{
	if (this != &src)
	{
		this->free();
		gradMag = src.gradMag.clone();
		pxIntensity = src.pxIntensity.clone();
	}
	return *this;
}
PBASFeature::PBASFeature()
{
}

float BackgroundFeature::getDistance() const
{

}

PBASFeature::~PBASFeature()
{
	free();
}

void PBASFeature::free() {

	gradMag.release(); pxIntensity.release();
}

double PBASFeature::getGradMagnMean() const {

	CV_Assert(!gradMag.empty());
	return cv::mean(gradMag)[0];
}