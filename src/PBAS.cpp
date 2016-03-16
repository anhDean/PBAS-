#include "PBAS.h"
#include "LBSP.h"

int PBAS::pbasCounter = 0; // static member variable to track instances
PBAS::PBAS(void) : m_N(20), m_minHits(2), m_defaultSubsampling(16)
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
PBAS::~PBAS(void)
{
	if(pbasCounter > 0)
	{
		--pbasCounter;
	}		
}
void PBAS::initialization(int N, double defaultR, int minHits, int defaultSubsampling, double alpha, double beta, double RScale, double RIncDec, double subsamplingIncRate, double subsamplingDecRate, int samplingLowerBound, int samplingUpperBound)
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
	m_samplingLowerBound = samplingLowerBound; // T_lower
	m_samplingUpperBound = samplingUpperBound;	// T_upper

	m_alpha = alpha;
	m_beta = beta;
	createRandomNumberArray(); // create random numbers beforehand for neighbor, background and distance update
}
void PBAS::createRandomNumberArray()
{
	randomN.clear();
	randomX.clear();
	randomY.clear();
	randomDist.clear();
	
	cv::theRNG().state = 0;

	//pre calculate random number 
	for(int l = 0; l < NUM_RANDOMGENERATION; l++)
	{
		randomN.push_back(cv::saturate_cast<int>(randomGenerator.uniform(0, m_N))); 		// for Background model position coordinate, upper bound is excluded
		randomX.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring X  coordinate, upper bound is excluded
		randomY.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring Y coordinate, upper bound is excluded
		randomDist.push_back(cv::saturate_cast<int>(randomGenerator.uniform(0, m_N))); // for Distance array posi  coordinate, upper bound is excluded
	}
}

bool PBAS::process(const cv::Mat *input, cv::Mat* output, const cv::Mat& gradMag, const cv::Mat& noiseMap)
{
	PBASFeature imgFeatures; // temp: hold temporary image features (3 matrices)
	cv::Mat blurImage = input->clone();
	int xNeigh, yNeigh;	// x,y coordinate of neighbor
	double formerDistanceBack, meanDistBack;
	cv::Mat segMap(blurImage.rows, blurImage.cols, blurImage.type());
	
	m_height = input->rows;
	m_width = input->cols;
	assert(input->type() == CV_8UC1);
	
	getFeatures(imgFeatures, &blurImage, gradMag);
	if(m_runs < m_N)
		// if runs < N collect background features without updating the model
	{
		m_backgroundModel.push_back(imgFeatures);
		m_minDistanceModel.push_back(cv::Mat(blurImage.size(), CV_32F)); // distanceStatisticBack: vector of Mat*, holds mean dist values for background
		
		if(m_runs == 0)
		// for the first run init R,T maps withdefault values
		{		
			m_sumMinDistMap.create(blurImage.rows,blurImage.cols, CV_32F);
			m_sumMinDistMap.setTo(cv::Scalar(0.0));

			m_RMap.create(m_sumMinDistMap.rows, m_sumMinDistMap.cols, CV_32F);
			m_RMap.setTo(cv::Scalar(m_defaultSubsampling));

			m_subSamplingMap.create(m_sumMinDistMap.rows, m_sumMinDistMap.cols, CV_32F);
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


	for (int y=0; y < m_height; ++y) 
	{
		for(int x = 0; x < m_width; ++x) 
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
				dist = calcDistanceXY(imgFeatures, x, y, index);

				if(dist < m_RMap.at<float>(y, x)) // match: smaller than pixel-depending threshold r
				{
					++count;
					if(minDist > dist)
						minDist = dist;
				}
				else
				{	
					//maxNorm += norm;
					++glCounterFore; 
				}
				++index;
			}
			while((count < m_minHits) && (index < m_runs)); // count << #min && index < runs, max(runs) = N

			// case background
			if(count >= m_minHits)
			{
				//set pixel to background value
				segMap.at<uchar>(y,x) = BACKGROUND_VAL;
				if(m_runs < m_N)
				{
						formerDistanceBack = 0; // since no distance value will be replaces, nothing need to be buffered for moving avg calculation
						m_minDistanceModel.at(m_runs -1).at<float>(y, x) = minDist;
						m_sumMinDistMap.at<float>(y, x) += m_minDistanceModel.at(m_runs - 1).at<float>(y, x);
				}
				//update model
				if(m_runs == m_N)
				{
					// Update current pixel
					// get random number between 0 and nrSubsampling-1
					int rand = 0;
					int updateCoeff = randomGenerator.uniform((int)0, (int)ceil(m_subSamplingMap.at<float>(y, x))); // 0 - 16
					if(updateCoeff < 1)// random subsampling, p(x) = 1/T
					{
						// replace randomly chosen sample
						rand = randomN.at(entry+1); //randomGenerator.uniform((int)0,(int)N-1);
						// replace background model at pixel y,x
						m_backgroundModel.at(rand).gradMag.at<float>(y, x) = imgFeatures.gradMag.at<float>(y, x);
						m_backgroundModel.at(rand).pxIntensity.at<uchar>(y, x) = imgFeatures.pxIntensity.at<uchar>(y, x);
						// replace sum distance model at pixel y,x
						formerDistanceBack = m_minDistanceModel.at(randomDist.at(entry)).at<float>(y, x); // save old dmin
						m_minDistanceModel.at(randomDist.at(entry)).at<float>(y, x) = minDist; // replace old entry with new dmin
						m_sumMinDistMap.at<float>(y, x) += m_minDistanceModel.at(randomDist.at(entry)).at<float>(y, x) - formerDistanceBack; // calculate current sum of dmins
					}
					// Update neighboring background model
					updateCoeff = randomGenerator.uniform((int)0, (int)ceil(m_subSamplingMap.at<float>(y, x)));
					if(updateCoeff < 1)// random subsampling
					{
						//choose neighboring pixel randomly
						xNeigh = randomX.at(entry) + x;
						yNeigh = randomY.at(entry) + y;
						checkValid(xNeigh, yNeigh);

						// replace randomly chosen sample
						rand = randomN.at(entry-1); 
						(m_backgroundModel.at(rand)).gradMag.at<float>(yNeigh,xNeigh) = imgFeatures.gradMag.at<float>(yNeigh,xNeigh);
						(m_backgroundModel.at(rand)).pxIntensity.at<uchar>(yNeigh,xNeigh) = imgFeatures.pxIntensity.at<uchar>(yNeigh,xNeigh);
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
	double meanNorm = abs( formerMaxNorm - imgFeatures.getGradMagnMean());
	formerMaxNorm = (meanNorm > 20)? 20: meanNorm; //TODO: old value 20 or 100
	// write segmentation result to output
	segMap.copyTo(*output);
	return true;
}


void PBAS::updateRThresholdXY(int x, int y, float avg_dmin) {
	//update R threshold of pixel x(j,i)
	//float meanDistBack = m_sumMinDistMap.at<float>(y, x) / runs; // mean(dmin) = sum(dmin)/runs

	if (m_RMap.at<float>(y,x) <= avg_dmin *m_RScale)
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
void PBAS::updateSubsamplingXY(int x, int y, int seg_value, float avg_dmin) {

	//time update, adjust learning rate
	if (seg_value == BACKGROUND_VAL)
		m_subSamplingMap.at<float>(y, x) -= m_subsamplingIncRate / (avg_dmin + 1);
	else
		m_subSamplingMap.at<float>(y, x) += m_subsamplingDecRate / (avg_dmin + 1);
	// check for boundaries
	if (m_subSamplingMap.at<float>(y, x)  < m_samplingLowerBound)
		m_subSamplingMap.at<float>(y, x) = m_samplingLowerBound;
	else if (m_subSamplingMap.at<float>(y, x) > m_samplingUpperBound)
		m_subSamplingMap.at<float>(y, x) = m_samplingUpperBound;
}
void PBAS::getFeatures(PBASFeature& descriptor, cv::Mat* intImg, const cv::Mat& gradMag)
{
	// shared variable: gradMagnMap
	descriptor.gradMag = gradMag.clone();
	intImg->copyTo(descriptor.pxIntensity);
}

double PBAS::calcDistanceXY(const PBASFeature& imgFeatures, int x, int y, int index) const
{
	// index: position of sample in background model
	// imgFeatures: imgFeatures matrix in current frame
	double norm = abs((double)m_backgroundModel.at(index).gradMag.at<float>(y, x) - (double)imgFeatures.gradMag.at<float>(y, x));
	int pixVal = abs(m_backgroundModel.at(index).pxIntensity.at<uchar>(y, x) - imgFeatures.pxIntensity.at<uchar>(y, x));
	return (m_alpha*(norm / formerMaxNorm) + m_beta * pixVal);
}

void PBAS::setAlpha(double alph)
{
	m_alpha = alph;
}
void PBAS::setBeta(double bet)
{
	m_beta = bet;
}
void PBAS::checkValid(int &x, int &y)
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
void PBAS::reset()
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
const double& PBAS::getAlpha() const {
	return m_alpha;
}
const double& PBAS::getBeta() const {
	return m_beta;
}
const int& PBAS::getPBASCounter()
{
	return pbasCounter;

}
const cv::Mat& PBAS::getSumMinDistMap() const
{
	return m_sumMinDistMap;
}
const int& PBAS::getRuns() const
{
	return m_runs;
}
const cv::Mat& PBAS::getTImg() const
{
	// Learning rate map
	return m_subSamplingMap;
}
const cv::Mat& PBAS::getRImg() const
{
	// treshold map
	return m_RMap;
}
