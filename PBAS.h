#pragma once
#include <opencv2/opencv.hpp>
#include "PBASFeature.h"
#include "LBSP.h"


// log: removed R,

class PBAS
{

private:
	int N;
	int m_minHits; // #min
	double m_RScale; // R_scale
	double m_RIncDec;	// R_inc/dec
	double m_beta, m_alpha; // alpha, beta
	int m_defaultSubsampling; // T
	double m_subsamplingIncRate; // T_inc
	double m_subsamplingDecRate; // T_dec
	double m_defaultR; // defaultR and lower bound of decision threshold
	int m_samplingUpperBound;	 //T_Upper
	int m_samplingLowerBound;	// T_Lower

	// background model
	std::vector<PBASFeature> m_backgroundModel;
	//new background distance model
	std::vector<cv::Mat> m_minDistanceModel; // D model in paper holds dmin values over N

	cv::Mat m_sumMinDistMap, m_RMap, m_subSamplingMap; //map of sum over dmin and of rThresholds


	//pre - initialize the randomNumbers for better performance
	std::vector<int> randomSubSampling, randomN, randomX, randomY, randomDist;
	int runs;
	int height, width;	// hieght, width of frame



	// scale that defines the increasing of the update rate of the background model, if the current pixel is background 
	//--> more frequently updates if pixel is background because, there shouln't be any change

	// defining an upper value, that nrSubsampling can achieve, thus it doesn't reach to an infinite value, where actually no update is possible 
	// at all

	//holds update rate of current pixel
	
	//random number generator
	cv::RNG randomGenerator;

	// each feature at time t consists of 3 matrices -> vector for time vector for matrices
	double formerMaxNorm; // formerMaxNorm: average gardient magnitude of last frame
	
	static int pbasCounter;
	const static int NUM_RANDOMGENERATION = 100000;
	const static int FOREGROUND_VAL = 255;
	const static int BACKGROUND_VAL = 0;


	void checkValid(int &x, int &y);
	void createRandomNumberArray();
	void getFeatures(PBASFeature& descriptor, cv::Mat* intImg);
	void updateRThresholdXY(int x, int y, float avg_dmin);
	void updateSubsamplingXY(int x, int y, int bg_value, float avg_dmin);
	static void deallocMem(cv::Mat *mat);


public:
	PBAS(void);
	~PBAS(void);
	
	void initialization(int newN, double newR, int newParts, int newNrSubSampling, double a, double b, double rThrSc, double rIndDec, double incrTR, double decrTR, int lowerTB, int upperTB);
	
	bool process(cv::Mat *input, cv::Mat*);
	void setAlpha(double alph);
	void setBeta(double bet);

	const double& getAlpha() const;
	const double& getBeta() const;
	const cv::Mat& getTImg() const;
	const cv::Mat& getRImg() const;
};
