#pragma once
#include <opencv2/opencv.hpp>
#include "PBASFeature.h"
#include "LBSP.h"





class PBAS
{
public:
	PBAS(void);
	~PBAS(void);
	//PBAS::PBAS(int newN, double newR, int newParts, int newNrSubSampling, double a, double b, double cf, double cb, double rThrSc, double rIndDec, double incrTR, double decrTR, int lowerTB, int upperTB)
	bool process(cv::Mat *input, cv::Mat*);
	void initialization(int newN, double newR, int newParts, int newNrSubSampling, double a, double b, double cf, double cb, double rThrSc, double rIndDec, double incrTR, double decrTR, int lowerTB, int upperTB);
	void setConstForeground(double constF);
	void setConstBackground(double constB);
	void setAlpha(double alph);
	void setBeta(double bet);
	bool isMoving();
	double getR();
	cv::Mat* getTImage();
	cv::Mat* getRImage();


private:
	void checkValid(int *x, int *y);
	void createRandomNumberArray();
	void checkXY(cv::Point2i*);
	void getFeatures(PBASFeature& descriptor, cv::Mat* intImg);
	void updateThreshold();
	static void deallocMem(cv::Mat *mat);
	bool isMovement; // boolean to mark foreground/backg pixel
	double beta, alpha, constBackground, constForeground;
	
	//####################################################################################
	//N - Number: Defining the size of the background-history-model
	// number of samples per pixel
	int N;
	// background model
	std::vector<cv::Mat> backgroundModel;
	//####################################################################################
	//####################################################################################
	//R-Threshhold - Variables
	// radius of the sphere -> lower border boundary
	double R;
	// scale for the sphere threshhold to define pixel-based Thresholds
	double rThreshScale; // R_scale
	// increasing/decreasing factor for the r-Threshold based on the result of rTreshScale * meanMinDistBackground
	double rIncDecFac;	// R_inc/dec
	cv::Mat sumThreshBack, rThresh; //map of sum over dmin and of rThresholds
	float *sumArrayDistBack; //sum of dmin over moving average window (but noi average) minDistBackground ->pointer to arrays
	float *rData; //new pixel-based r-threshhold -> pointer to arrays
	//#####################################################################################
	//####################################################################################
	// Defining the number of background-model-images, which have a lowerDIstance to the current Image than defined by the R-Thresholds, that are necessary
	// to decide that this pixel is background
	int parts; // #min
	//#####################################################################################
	//####################################################################################
	// Initialize the background-model update rate 
	int nrSubsampling; // 1/T

	// scale that defines the increasing of the update rate of the background model, if the current pixel is background 
	//--> more frequently updates if pixel is background because, there shouln't be any change
	double increasingRateScale;
	// defining an upper value, that nrSubsampling can achieve, thus it doesn't reach to an infinite value, where actually no update is possible 
	// at all
	int upperTimeUpdateRateBoundary;	
	//holds update rate of current pixel
	cv::Mat tempCoeff; // 1/T(xi)
	float *tCoeff;
	// opposite scale to increasingRateScale, for decreasing the update rate of the background model, if the current pixel is foreground
	//--> Thesis: Our Foreground is a real moving object -> thus the background-model is good, so don't update it
	double decreasingRateScale;
	// defining a minimum value for nrSubsampling --> base value 2.0
	int lowerTimeUpdateRateBoundary;
	//#####################################################################################

	// background/foreground segmentation map -> result Map
	cv::Mat* segMap;

	// background and foreground identifiers
	int foreground, background; // identifiers if background or foreground

	int height, width;	// hieght, width of frame

	//random number generator
	cv::RNG randomGenerator;

	//pre - initialize the randomNumbers for better performance
	std::vector<int> randomSubSampling,	randomN, randomX, randomY,randomDist;
	int runs,countOfRandomNumb;

	uchar* data, *segRowData;
	float* dataBriefNorm, *dataBriefDir; // hold row of current image feature matrix
	uchar* dataBriefCol;
	uchar* dataStats;

	std::vector<uchar*> backgroundPt;
	std::vector<uchar*>backgroundPtBriefCol;
	std::vector<float*> backgroundPtBriefNorm, backgroundPtBriefDir;
	
	std::vector<PBASFeature> backGroundFeatures; // each feature at time t consists of 3 matrices -> vector for time vector for matrices

	PBASFeature temp, imgFeatures; // temp: hold temporary image features (3 matrices)

	//cv::ORB orb;
	//std::vector<cv::KeyPoint> keypoints1;
	//cv::Mat descr1;
	int xNeigh, yNeigh;	// x,y coordinate of neighbor
	float meanDistBack; // mean(d_min)
	double formerMaxNorm, formerMaxPixVal,formerMaxDir; // formerMaxNorm: average gardient magnitude of last frame

	//new background distance model
	std::vector<float*> distanceStatPtBack; // vector of dmin values
	std::vector<cv::Mat*> distanceStatisticBack; /*,distanceStatisticFore;*/	
	
	float formerDistanceFore,formerDistanceBack;
	cv::Mat* tempDist,* tempDistB;
	double setR;
	cv::Mat sobelX, sobelY; // matrices to perfom filtering -> get gradmagnitude

};
