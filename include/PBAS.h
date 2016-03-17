#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

// log: removed R,
template <typename Descriptor>

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
	std::vector<std::vector<Descriptor>> m_backgroundModel; // outer vector N, inner vector matrix with features
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
	void updateRThresholdXY(int x, int y, float avg_dmin);
	void updateSubsamplingXY(int x, int y, int bg_value, float avg_dmin);

public:
	PBAS();
	~PBAS();
	const static int FOREGROUND_VAL = 255;
	const static int BACKGROUND_VAL = 0;

	void initialization(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound);
	void reset();

	bool process(const std::vector<Descriptor>& featureMap, int height, int width, cv::Mat& output);
	
	void setAlpha(double alph);
	void setBeta(double bet);

	static int getPBASCounter();
	double getAlpha() const;
	double getBeta() const;
	const cv::Mat& getTImg() const;
	const cv::Mat& getRImg() const;
	const cv::Mat& getSumMinDistMap() const; // measures "background dynamics"
	int getRuns() const;
	
};



template <typename Descriptor>
int PBAS<Descriptor>::pbasCounter = 0; // static member variable to track instances

template <typename Descriptor>
PBAS<Descriptor>::PBAS() : m_N(20), m_minHits(2), m_defaultSubsampling(16)
{
	//initialize background-model depending parameters
	//r-Thresh
	m_RScale = 6.0;
	m_RIncDec = 0.05;
	//T-thresh
	m_subsamplingIncRate = 0.001;
	m_subsamplingDecRate = 0.005;
	m_samplingLowerBound = 2;
	m_samplingUpperBound = 300;

	//initialize background-model independent parameters
	formerMaxNorm = 1.0;
	m_height = 0;
	m_width = 0;
	m_runs = 0;

	m_beta = 1.0;
	m_alpha = 1.0;

	++pbasCounter;
}

template <typename Descriptor>
PBAS<Descriptor>::~PBAS(void)
{
	if (pbasCounter > 0)
	{
		--pbasCounter;
	}
}

template <typename Descriptor>
void PBAS<Descriptor>::initialization(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound)
{
	m_N = N;			// N: number of past samples
	m_defaultR = defaultR;		// R: decision threshold
	m_minHits = minHits;
	m_defaultSubsampling = defaultSubsampling; // 


											   //r-Thresh
	m_RScale = RScale; // R_scale
	m_RIncDec = RIncDec; // = R_inc/dec
						 //T-thresh
	m_subsamplingIncRate = subsamplingIncRate;		// T_inc
	m_subsamplingDecRate = subsamplingDecRate;		// T_dec
	m_samplingLowerBound = samplingLowerBound;	// T_lower
	m_samplingUpperBound = samplingUpperBound;	// T_upper

	m_alpha = alpha;
	m_beta = beta;
	createRandomNumberArray(); // create random numbers beforehand for neighbor, background and distance update
}

template <typename Descriptor>
void PBAS<Descriptor>::createRandomNumberArray()
{
	randomN.clear();
	randomX.clear();
	randomY.clear();
	randomDist.clear();

	cv::theRNG().state = 0;

	//pre calculate random number 
	for (int l = 0; l < NUM_RANDOMGENERATION; l++)
	{
		randomN.push_back(cv::saturate_cast<int>(randomGenerator.uniform(0, m_N))); 		// for Background model position coordinate, upper bound is excluded
		randomX.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring X  coordinate, upper bound is excluded
		randomY.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring Y coordinate, upper bound is excluded
		randomDist.push_back(cv::saturate_cast<int>(randomGenerator.uniform(0, m_N))); // for Distance array posi  coordinate, upper bound is excluded
	}
}

template <typename Descriptor>
bool PBAS<Descriptor>::process(const std::vector<Descriptor>& featureMap, int height, int width, cv::Mat& output)
{
	
	m_height = height;
	m_width = width;

	int xNeigh, yNeigh;	// x,y coordinate of neighbor
	double formerDistanceBack, meanDistBack;
	cv::Mat segMap(m_height, m_width, CV_8U);


	if (m_runs < m_N)
		// if runs < N collect background features without updating the model
	{
		m_backgroundModel.push_back(featureMap);
		m_minDistanceModel.push_back(cv::Mat(segMap.size(), CV_32F)); // distanceStatisticBack: vector of Mat*, holds mean dist values for background

		if (m_runs == 0)
			// for the first run init R,T maps withdefault values
		{
			m_sumMinDistMap.create(segMap.size(), CV_32F);
			m_sumMinDistMap.setTo(cv::Scalar(0.0));

			m_RMap.create(segMap.size(), CV_32F);
			m_RMap.setTo(cv::Scalar(m_defaultSubsampling));

			m_subSamplingMap.create(segMap.size(), CV_32F);
			m_subSamplingMap.setTo(cv::Scalar(m_defaultR));
		}
		++m_runs;
	}

	double sumDist = 0.0;
	// variables to generate old average of gradient magnitude
	double maxNorm = 0.0;
	int glCounterFore = 0;

	int count;  // used for #min
	int index;  // index k = 1,...,N or k = 1,...,runs (runs != N)
	double dist; // distance measure
	double temp;
	double maxDist;
	float minDist; // arbritrary large number for minDist
	int entry;


	for (int y = 0; y < m_height; ++y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			count = 0;  // used for #min
			index = 0;  // index k = 1,...,N or k = 1,...,runs (runs != N)
			dist = 0.0; // distance measure
			temp = 0.0;
			maxDist = 0.0;
			minDist = 1000.0; // arbritrary large number for minDist
			entry = randomGenerator.uniform(5, NUM_RANDOMGENERATION - 5);


			// match observation with backgound model
			do
			{					
				dist = Descriptor::calcDistance(m_backgroundModel.at(index)[y * m_width + x], featureMap[y * m_width + x]);  // 1: L1 otherwise L2 norm

				if (dist < m_RMap.at<float>(y, x)) // match: smaller than pixel-depending threshold r
				{
					++count;
					if (minDist > dist)
						minDist = dist;
				}

				else
				{
					//maxNorm += norm;
					++glCounterFore;
				}
				++index;
			} while ((count < m_minHits) && (index < m_runs)); // count << #min && index < runs, max(runs) = N


			if (count >= m_minHits) // case background
			{
				//set pixel to background value
				segMap.at<uchar>(y, x) = BACKGROUND_VAL;
				if (m_runs < m_N)
				{
					formerDistanceBack = 0; // since no distance value will be replaces, nothing need to be buffered for moving avg calculation
					m_minDistanceModel.at(m_runs - 1).at<float>(y, x) = minDist;
					m_sumMinDistMap.at<float>(y, x) += m_minDistanceModel.at(m_runs - 1).at<float>(y, x);
				}
				//update model
				if (m_runs == m_N)
				{
					// Update current pixel
					// get random number between 0 and nrSubsampling-1
					int rand = 0;
					int updateCoeff = randomGenerator.uniform((int)0, (int)ceil(m_subSamplingMap.at<float>(y, x))); // 0 - 16
					if (updateCoeff < 1)// random subsampling, p(x) = 1/T
					{
						// replace randomly chosen sample
						rand = randomN.at(entry + 1); //randomGenerator.uniform((int)0,(int)N-1);
													  // replace background model at pixel y,x
						m_backgroundModel.at(rand)[y * m_width + x] = featureMap[y * m_width + x];
						// replace sum distance model at pixel y,x
						formerDistanceBack = m_minDistanceModel.at(randomDist.at(entry)).at<float>(y, x); // save old dmin
						m_minDistanceModel.at(randomDist.at(entry)).at<float>(y, x) = minDist; // replace old entry with new dmin
						m_sumMinDistMap.at<float>(y, x) += m_minDistanceModel.at(randomDist.at(entry)).at<float>(y, x) - formerDistanceBack; // calculate current sum of dmins
					}
					// Update neighboring background model
					updateCoeff = randomGenerator.uniform((int)0, (int)ceil(m_subSamplingMap.at<float>(y, x)));
					if (updateCoeff < 1)// random subsampling
					{
						//choose neighboring pixel randomly
						xNeigh = randomX.at(entry) + x;
						yNeigh = randomY.at(entry) + y;
						checkValid(xNeigh, yNeigh);

						// replace randomly chosen sample
						rand = randomN.at(entry - 1);
						m_backgroundModel.at(rand)[yNeigh * m_width + xNeigh] = featureMap[yNeigh * m_width + xNeigh];
					}
				}
			}
			else
			{
				segMap.at<uchar>(y, x) = FOREGROUND_VAL;

				// no model update when foreground!?
			}
			meanDistBack = m_sumMinDistMap.at<float>(y, x) / m_runs;
			// update learning rate  and decision threshold
			updateSubsamplingXY(x, y, segMap.at<uchar>(y, x), meanDistBack);
			updateRThresholdXY(x, y, meanDistBack);
		}
	}
	// calculate average gradient magnitude
	//double meanNorm = maxNorm / ((double)(glCounterFore + 1));
	//double meanNorm = abs(formerMaxNorm - imgFeatures.getGradMagnMean());
	//double meanNorm = abs(formerMaxNorm - Descriptor::calcMeanGradMagn(featureMap,m_height, m_width));
	//formerMaxNorm =  meanNorm; //TODO: old value 20 or 100
													 // write segmentation result to output
	segMap.copyTo(output);
	return true;
}

template <typename Descriptor>
void PBAS<Descriptor>::updateRThresholdXY(int x, int y, float avg_dmin) {
	//update R threshold of pixel x(j,i)
	//float meanDistBack = m_sumMinDistMap.at<float>(y, x) / runs; // mean(dmin) = sum(dmin)/runs

	if (m_RMap.at<float>(y, x) <= avg_dmin *m_RScale)
	{
		m_RMap.at<float>(y, x) *= (1 + m_RIncDec);
	}
	else
	{
		m_RMap.at<float>(y, x) *= (1 - m_RIncDec);

	}
	if (m_RMap.at<float>(y, x) < m_defaultR)
		m_RMap.at<float>(y, x) = m_defaultR;

}

template <typename Descriptor>
void PBAS<Descriptor>::updateSubsamplingXY(int x, int y, int seg_value, float avg_dmin) {

	//time update, adjust learning rate
	if (seg_value == BACKGROUND_VAL)
		m_subSamplingMap.at<float>(y, x) -= m_subsamplingDecRate / (avg_dmin + 1);
	else
		m_subSamplingMap.at<float>(y, x) += m_subsamplingIncRate / (avg_dmin + 1);
	// check for boundaries
	if (m_subSamplingMap.at<float>(y, x)  < m_samplingLowerBound)
		m_subSamplingMap.at<float>(y, x) = m_samplingLowerBound;
	else if (m_subSamplingMap.at<float>(y, x) > m_samplingUpperBound)
		m_subSamplingMap.at<float>(y, x) = m_samplingUpperBound;
}


template <typename Descriptor>
void PBAS<Descriptor>::setAlpha(double alph)
{
	m_alpha = alph;
}

template <typename Descriptor>
void PBAS<Descriptor>::setBeta(double bet)
{
	m_beta = bet;
}

template <typename Descriptor>
void PBAS<Descriptor>::checkValid(int &x, int &y)
{
	if (x < 0)
	{
		x = 0;
	}
	else if (x >= m_width)
	{
		x = m_width - 1;
	}

	if (y < 0)
	{
		y = 0;
	}
	else if (y >= m_height)
	{
		y = m_height - 1;
	}
}

template <typename Descriptor>
void PBAS<Descriptor>::reset()
{
	m_runs = 0;
	m_height = 0;
	m_width = 0;
	createRandomNumberArray();
	m_backgroundModel.clear();
	m_minDistanceModel.clear();
	m_sumMinDistMap.release();
	m_RMap.release();
	m_subSamplingMap.release();
}

template <typename Descriptor>
double PBAS<Descriptor>::getAlpha() const {
	return m_alpha;
}

template <typename Descriptor>
double PBAS<Descriptor>::PBAS::getBeta() const {
	return m_beta;
}

template <typename Descriptor>
int PBAS<Descriptor>::getPBASCounter()
{
	return pbasCounter;
}

template <typename Descriptor>
const cv::Mat& PBAS<Descriptor>::getSumMinDistMap() const
{
	return m_sumMinDistMap;
}
template <typename Descriptor>

int PBAS<Descriptor>::getRuns() const
{
	return m_runs;
}
template <typename Descriptor>

const cv::Mat& PBAS<Descriptor>::getTImg() const
{
	// Learning rate map
	return m_subSamplingMap;
}

template <typename Descriptor>
const cv::Mat& PBAS<Descriptor>::getRImg() const
{
	// treshold map
	return m_RMap;
}
