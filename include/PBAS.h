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
	int m_runs = 0;						// iteration counter
	int m_height, m_width;			// height, width of frame

	bool m_modelFilled = false;

	// background model
	std::vector<std::vector<Descriptor>> m_backgroundModel; // outer vector N, inner vector matrix with features
	std::vector<std::vector<float>> m_minDistanceModel; // D model in paper holds dmin values over N
	cv::Mat m_sumMinDistMap;

	cv::Mat m_initSet;
	std::vector<std::vector<Descriptor>> m_staticBackgroundModel; // outer vector N, inner vector matrix with features
	bool m_isInitialized = false;

	cv::Mat  m_RMap, m_subSamplingMap; //map of sum over dmin and of rThresholds
	cv::Mat  m_bgFilled;
	//random number generator
	cv::RNG m_randomGenerator;
	//pre - initialize the randomNumbers for better performance
	std::vector<int> m_randomN, m_randomX, m_randomY, m_randomDist;


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

	bool process(const std::vector<Descriptor>& featureMap, int height, int width, cv::Mat& output, const cv::Mat& opticalFlow);
	
	void setAlpha(double alph);
	void setBeta(double bet);

	static int getPBASCounter();
	double getAlpha() const;
	double getBeta() const;
	int getHeight() const;
	int getWidth() const;
	const cv::Mat& getTImg() const;
	const cv::Mat& getRImg() const;
	const std::vector<cv::Mat>& getMinDistModel() const;
	const cv::Mat& getSumMinDistMap() const; // measures "background dynamics"
	int getRuns() const;

	bool isInitialized() const;
	void subSamplingOffset(const cv::Mat& offsetMask);
	
	cv::Mat drawBgSample();
	
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
	m_randomN.clear();
	m_randomX.clear();
	m_randomY.clear();
	m_randomDist.clear();

	cv::theRNG().state = 0;

	//pre calculate random number 
	for (int l = 0; l < NUM_RANDOMGENERATION; l++)
	{
		m_randomN.push_back((int)(m_randomGenerator.uniform(0, m_N))); 		// for Background model position coordinate, upper bound is excluded
		m_randomX.push_back((int)(m_randomGenerator.uniform(-1, +2)));				// for neighboring X  coordinate, upper bound is excluded
		m_randomY.push_back((int)(m_randomGenerator.uniform(-1, +2)));				// for neighboring Y coordinate, upper bound is excluded
		m_randomDist.push_back((int)(m_randomGenerator.uniform(0, m_N))); // for Distance array posi  coordinate, upper bound is excluded
	}
}

template <typename Descriptor>
bool PBAS<Descriptor>::process(const std::vector<Descriptor>& featureMap, int height, int width, cv::Mat& output, const cv::Mat& opticalFlow)
{
	// variables to generate old average of gradient magnitude
	int xNeigh, yNeigh;	// x,y coordinate of neighbor
	float formerDistanceBack;
	int count;  // used for #min
	int index;  // index k = 1,...,N or k = 1,...,runs (runs != N)
	float dist; // distance measure
	float minDist; // arbritrary large number for minDist
	int entry;
	double sumDist;
	int modelSizeXY;
	
	bool runsFlag = (m_runs > 120);
	
	cv::Mat segMap(height, width, CV_8U);

	
	m_height = height;
	m_width = width;

	if (m_runs == 0) // initialize container
	{
		m_backgroundModel.resize(height * width);
		m_minDistanceModel.resize(height * width);
		m_staticBackgroundModel.resize(height * width);

		m_sumMinDistMap.create(segMap.size(), CV_32F);
		m_sumMinDistMap.setTo(cv::Scalar(0.0));

		m_RMap.create(segMap.size(), CV_32F);
		m_RMap.setTo(cv::Scalar(m_defaultR));

		m_subSamplingMap.create(segMap.size(), CV_32F);
		m_subSamplingMap.setTo(cv::Scalar(m_defaultSubsampling));

		m_bgFilled = cv::Mat::zeros(segMap.size(), CV_8U);
		m_initSet = cv::Mat::zeros(segMap.size(), CV_8U);
		
	}

	if (m_modelFilled)
	{
		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)
			{
				sumDist = 0.0;
				count = 0;  // used for #min
				index = 0;  // index k = 1,...,N or k = 1,...,runs (runs != N)
				dist = 0.0; // distance measure
				minDist = 1000.0; // arbritrary large number for minDist

				modelSizeXY = m_backgroundModel.at(y * m_width + x).size();
				do
				{
					dist = Descriptor::calcDistance(m_backgroundModel.at(y * m_width + x)[index], featureMap[y * m_width + x]);  // 1: L1 (faster) otherwise L2 norm

					if (dist < m_RMap.at<float>(y, x)) // match: smaller than pixel-depending threshold r
					{
						++count;
						if (dist < minDist)
							minDist = dist;
					}

					++index;

				} while ((count < m_minHits) && (index < modelSizeXY)); // count << #min && index < runs, max(runs) = N

				if (count >= m_minHits) // case background
				{
					//set pixel to background value
					segMap.at<uchar>(y, x) = BACKGROUND_VAL;

					entry = m_randomGenerator.uniform(5, NUM_RANDOMGENERATION - 5);
					// Update current pixel
					// get random number between 0 and nrSubsampling-1
					int rand = 0;
					int updateCoeff = m_randomGenerator.uniform(0, (int)ceil(m_subSamplingMap.at<float>(y, x))); // 0 - 16



					if (m_minDistanceModel.at(y * m_width + x).size() < m_N)
					{
						m_minDistanceModel.at(y * m_width + x).push_back(minDist);
					}

					if (updateCoeff < 1)// random subsampling, p(x) = 1/T
					{
						// replace randomly chosen sample
						rand = m_randomN.at(entry + 1);
						m_backgroundModel[y * m_width + x].at(rand)= featureMap[y * m_width + x];

						// replace sum distance model at pixel y,x
						rand = cv::saturate_cast<uchar>(m_randomGenerator.uniform(0, m_minDistanceModel.at(y * m_width + x).size() - 1));
						formerDistanceBack = m_minDistanceModel.at(y * m_width + x)[rand]; // save old dmin
						m_minDistanceModel.at(y * m_width + x)[rand] = minDist; // replace old entry with new dmin 
					}

					// Update neighboring background model
					updateCoeff = m_randomGenerator.uniform(0, (int)ceil(m_subSamplingMap.at<float>(y, x)));
					if (updateCoeff < 1)// random subsampling
					{
						//choose neighboring pixel randomly
						xNeigh = m_randomX.at(entry) + x;
						yNeigh = m_randomY.at(entry) + y;
						checkValid(xNeigh, yNeigh);

						// replace randomly chosen sample
						rand = m_randomN.at(entry - 1);
						m_backgroundModel.at(yNeigh * m_width + xNeigh)[rand] = featureMap[y * m_width + x];
					}

					m_sumMinDistMap.at<float>(y, x) = cv::saturate_cast<float>(minDist - formerDistanceBack + m_sumMinDistMap.at<float>(y, x));
				}

				else
				{
					segMap.at<uchar>(y, x) = FOREGROUND_VAL;
				}

				updateSubsamplingXY(x, y, segMap.at<uchar>(y, x), m_sumMinDistMap.at<float>(y, x) / m_minDistanceModel.at(y * m_width + x).size());
				updateRThresholdXY(x, y, m_sumMinDistMap.at<float>(y, x) / m_minDistanceModel.at(y * m_width + x).size());
				
				if (!m_isInitialized)
				{
					if ((opticalFlow.at<float>(y, x) == 0  && segMap.at<uchar>(y, x) == BACKGROUND_VAL) || (segMap.at<uchar>(y, x) == BACKGROUND_VAL && runsFlag) || (opticalFlow.at<float>(y, x) == 0 && runsFlag))
					{
						if (m_staticBackgroundModel.at(y * m_width + x).size() < m_N)
						{
							m_staticBackgroundModel.at(y * m_width + x).push_back(featureMap[y * m_width + x]);
						}
						else
						{
							m_initSet.at<uchar>(y, x) = 1;
						}
					}

				}
			
			
			}


		}



	}

	else
	{

		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)
			{
				sumDist = 0.0;
				count = 0;  // used for #min
				index = 0;  // index k = 1,...,N or k = 1,...,runs (runs != N)
				dist = 0.0; // distance measure
				minDist = 1000.0; // arbritrary large number for minDist
				//std::cout << m_backgroundModel.at(y * m_width + x).size() << std::endl;
				modelSizeXY = m_backgroundModel.at(y * m_width + x).size();


				if (m_backgroundModel.at(y * m_width + x).size() < m_N)
				{
					m_backgroundModel.at(y * m_width + x).push_back(featureMap[y * m_width + x]);
				}
				else
				{
					m_bgFilled.at<uchar>(y, x) = 1;
				}


				do
				{
					dist = Descriptor::calcDistance(m_backgroundModel.at(y * m_width + x)[index], featureMap[y * m_width + x]);  // 1: L1 (faster) otherwise L2 norm

					if (dist < m_RMap.at<float>(y, x)) // match: smaller than pixel-depending threshold r
					{
						++count;
						if (dist < minDist)
							minDist = dist;
					}

					++index;

				} while ((count < m_minHits) && (index < modelSizeXY)); // count << #min && index < runs, max(runs) = N


				if (count >= m_minHits) // case background
				{
					//set pixel to background value
					segMap.at<uchar>(y, x) = BACKGROUND_VAL;

					if (m_minDistanceModel.at(y * m_width + x).size() < m_N)
					{
						m_minDistanceModel.at(y * m_width + x).push_back(minDist);
					}

					m_sumMinDistMap.at<float>(y, x) = cv::saturate_cast<float>(minDist + m_sumMinDistMap.at<float>(y, x));
				}

				else
				{
					segMap.at<uchar>(y, x) = FOREGROUND_VAL;
				}


				updateSubsamplingXY(x, y, segMap.at<uchar>(y, x), m_sumMinDistMap.at<float>(y, x) / m_minDistanceModel.size());
				updateRThresholdXY(x, y, m_sumMinDistMap.at<float>(y, x) / m_minDistanceModel.size());
			
				// parallel initialization
				if (m_runs != 0)
				{
					if (!m_isInitialized)
					{
						
						if (opticalFlow.at<float>(y, x) == 0 && segMap.at<uchar>(y, x) == BACKGROUND_VAL)
						{
							if (m_staticBackgroundModel.at(y * m_width + x).size() < m_N)
							{
								m_staticBackgroundModel.at(y * m_width + x).push_back(featureMap[y * m_width + x]);
							}
							else
							{
								m_initSet.at<uchar>(y, x) = 1;
							}
						
						}

					}
				}
			
			}


		}

		if (cv::sum(m_bgFilled)[0] == m_height * m_width)
			m_modelFilled = true;
	}

	if (cv::sum(m_initSet)[0] == m_width * m_height && !m_isInitialized)
	{
		m_backgroundModel = m_staticBackgroundModel;
		m_isInitialized = true;
	}

	segMap.copyTo(output);
	++m_runs;
	return true;
}

template <typename Descriptor>
void PBAS<Descriptor>::updateRThresholdXY(int x, int y, float avg_dmin) {
	//update R threshold of pixel x(j,i)
	if (m_RMap.at<float>(y, x) <= avg_dmin * m_RScale)
	{
		m_RMap.at<float>(y, x) *= (1 + m_RIncDec);
	}
	else
	{
		m_RMap.at<float>(y, x) *= (1 - m_RIncDec);

	}
	if (m_RMap.at<float>(y, x) < m_defaultR)
		m_RMap.at<float>(y, x) = m_defaultR;
	
	if (m_RMap.at<float>(y, x) > 2.5 * m_defaultR)
		m_RMap.at<float>(y, x) = 2.5 * m_defaultR;

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
	m_modelFilled = false;
	m_isInitialized = false;
	m_bgFilled.release();
	m_staticBackgroundModel.clear();


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


template <typename Descriptor>
int PBAS<Descriptor>::getHeight() const
{
	return m_height;
}
template <typename Descriptor>
int PBAS<Descriptor>::getWidth() const
{
	return m_width;
}

template <typename Descriptor>
const std::vector<cv::Mat>& PBAS<Descriptor>::getMinDistModel() const
{
	return m_minDistanceModel;
}


template <typename Descriptor>
cv::Mat PBAS<Descriptor>::drawBgSample()
{
	cv::Mat b(m_height,m_width,CV_8U), g(m_height, m_width, CV_8U), r(m_height, m_width, CV_8U);
	std::vector<cv::Mat> channels;
	cv::Mat result;

	int idx;

	if (!m_modelFilled)
	{
		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)

			{
				idx = cv::saturate_cast<uchar>(m_randomGenerator.uniform(0, m_backgroundModel.at(y * m_width + x).size() - 1));
				b.at<uchar>(y, x) = m_backgroundModel.at(y * m_width + x).at(idx).m_B;
				g.at<uchar>(y, x) = m_backgroundModel.at(y * m_width + x).at(idx).m_G;
				r.at<uchar>(y, x) = m_backgroundModel.at(y * m_width + x).at(idx).m_R;
			}
		}
	}
	else
	{ 
		idx = cv::saturate_cast<uchar>(m_randomGenerator.uniform(0, m_N - 1));
		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)

			{
				
				b.at<uchar>(y, x) = m_backgroundModel.at(y * m_width + x).at(idx).m_B;
				g.at<uchar>(y, x) = m_backgroundModel.at(y * m_width + x).at(idx).m_G;
				r.at<uchar>(y, x) = m_backgroundModel.at(y * m_width + x).at(idx).m_R;
			}
		}
	}

	channels.push_back(b);
	channels.push_back(g);
	channels.push_back(r);

	cv::merge(channels, result);
	return result;

}

template <typename Descriptor>
bool PBAS<Descriptor>::isInitialized() const
{
	return m_isInitialized;
}

template <typename Descriptor>
void PBAS<Descriptor>::subSamplingOffset(const cv::Mat& offsetMask)
{
	for (int y = 0; y < m_subSamplingMap.rows; ++y)
	{
		for (int x = 0; x < m_subSamplingMap.cols; ++x)
		{
			m_subSamplingMap.at<float>(y,x) += offsetMask.at<float>(y,x);
			if (m_subSamplingMap.at<float>(y, x) <= 0)
				m_subSamplingMap.at<float>(y, x) = 0.001;
			else if (m_subSamplingMap.at<float>(y, x) > m_samplingUpperBound)
				m_subSamplingMap.at<float>(y, x) = m_samplingUpperBound;
		}
	}
}