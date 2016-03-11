#pragma once
#include <opencv2/opencv.hpp>
#include "PBASFeature.h"
#include "LBSP.h"
#include <memory>
#include <mutex>


// log: removed R,

class PBAS
{

private:
	int m_N;
	int m_minHits;					// #min
	double m_RScale;				// R_scale
	double m_RIncDec;				// R_inc/dec
	double m_beta, m_alpha;			// alpha, beta
	int m_defaultSubsampling;		// T
	double m_subsamplingIncRate;	// T_inc
	double m_subsamplingDecRate;	// T_dec
	double m_defaultR;				// defaultR and lower bound of decision threshold
	int m_samplingUpperBound;		// T_Upper
	int m_samplingLowerBound;		// T_Lower
	int m_runs;						// iteration counter
	int m_height, m_width;			// height, width of frame
	// background model
	std::vector<PBASFeature> m_backgroundModel;
	//new background distance model
	std::vector<cv::Mat> m_minDistanceModel; // D model in paper holds dmin values over N
	
	cv::Mat m_sumMinDistMap, m_RMap, m_subSamplingMap; //map of sum over dmin and of rThresholds

	//random number generator
	cv::RNG randomGenerator;
	//pre - initialize the randomNumbers for better performance
	std::vector<int> randomSubSampling, randomN, randomX, randomY, randomDist;


	// each feature at time t consists of 3 matrices -> vector for time vector for matrices
	double formerMaxNorm; // formerMaxNorm: average gardient magnitude of last frame
	
	static int pbasCounter;
	const static int NUM_RANDOMGENERATION = 100000;


	void checkValid(int &x, int &y);
	void createRandomNumberArray();
	void getFeatures(PBASFeature& descriptor, cv::Mat* intImg, const cv::Mat* gradMag);
	void updateRThresholdXY(int x, int y, float avg_dmin);
	void updateSubsamplingXY(int x, int y, int bg_value, float avg_dmin);
	double calcDistanceXY(const PBASFeature& imgFeatures, int x, int y, int index) const;

public:
	PBAS(void);
	~PBAS(void);
	const static int FOREGROUND_VAL = 255;
	const static int BACKGROUND_VAL = 0;

	void initialization(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound);
	void reset();

	bool process(const cv::Mat *input, cv::Mat* output, const cv::Mat* gradMag, const cv::Mat& noiseMap);
	void setAlpha(double alph);
	void setBeta(double bet);

	static const int& getPBASCounter();
	const double& getAlpha() const;
	const double& getBeta() const;
	const cv::Mat& getTImg() const;
	const cv::Mat& getRImg() const;
	const cv::Mat& getSumMinDistMap() const; // measures "background dynamics"
	const int& getRuns() const;
	
};
