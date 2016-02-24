#include "PBASFeature.h"


PBASFeature::PBASFeature(const PBASFeature &x)
{
	PBASFeature destination;
	this->gradMag = x.gradMag.clone();
	this->gradAngle= x.gradAngle.clone();
	this->pxIntensity = x.pxIntensity.clone();
}
void PBASFeature::operator=(const PBASFeature& src)
{
	this->gradMag = src.gradMag.clone();
	this->gradAngle = src.gradAngle.clone();
	this->pxIntensity = src.pxIntensity.clone();


}
PBASFeature::PBASFeature()
{
}

PBASFeature::~PBASFeature()
{
	free();
}

void PBASFeature::free() {

	this->gradMag.release(); this->gradAngle.release(); this->pxIntensity.release();
}