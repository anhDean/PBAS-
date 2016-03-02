#include "PBAS.h"
#include "LBSP.h"

int PBAS::pbasCounter = 0; // static member variable to track instances

PBAS::PBAS(void) : N(20), m_minHits(2), m_defaultSubsampling(16)
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
	height = 0; 
	width = 0;
	runs = 0;
	
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

void PBAS::initialization(int newN, double newR, int newParts, int newNrSubSampling, double a, double b, double rThrSc, double rIndDec, double incrTR, double decrTR, int lowerTB, int upperTB)
{
	N = newN;			// N: number of past samples
	m_defaultR = newR;		// R: decision threshold
	m_minHits = newParts;
	m_defaultSubsampling = newNrSubSampling; // 


	//r-Thresh
	m_RScale = rThrSc; // R_scale
	m_RIncDec = rIndDec; // = R_inc/dec
	//T-thresh
	m_subsamplingIncRate = incrTR;		// T_inc
	m_subsamplingDecRate = decrTR;		// T_dec
	m_samplingLowerBound = lowerTB; // T_lower
	m_samplingUpperBound = upperTB;	// T_upper

	m_beta = b;
	m_alpha = a;
	createRandomNumberArray(); // create random numbers beforehand for neighbor, background and distance update
}

void PBAS::createRandomNumberArray()
{
	//pre calculate random number 
	for(int l = 0; l < NUM_RANDOMGENERATION; l++)
	{
		randomN.push_back(cv::saturate_cast<int>(randomGenerator.uniform((int)0,(int)N))); 		// for Background model position coordinate, upper bound is excluded
		randomX.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring X  coordinate, upper bound is excluded
		randomY.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring Y coordinate, upper bound is excluded
		randomDist.push_back(cv::saturate_cast<int>(randomGenerator.uniform((int)0, (int)N))); // for Distance array posi  coordinate, upper bound is excluded
	}
}

bool PBAS::process(cv::Mat* input, cv::Mat* output)
{
	PBASFeature imgFeatures; // temp: hold temporary image features (3 matrices)
	cv::Mat blurImage = input->clone();
	int xNeigh, yNeigh;	// x,y coordinate of neighbor
	float formerDistanceBack, meanDistBack;
	cv::Mat segMap(blurImage.rows, blurImage.cols, blurImage.type());
	
	height = input->rows;
	width = input->cols;
	assert(input->type() == CV_8UC1);
	//cv::Mat blurImage(input->rows, input->cols, CV_8UC1, input->data);
	
	
	if(runs < N)
		// if runs < N collect background features without updating the model
	{
		getFeatures(imgFeatures, &blurImage);
		m_backgroundModel.push_back(imgFeatures);
		//tempDistB = ; // create new matrix pointer (N in total)
		m_minDistanceModel.push_back(cv::Mat(blurImage.size(), CV_32FC1)); // distanceStatisticBack: vector of Mat*, holds mean dist values for background
		
		if(runs == 0)
		// for the first run init R,T maps withdefault values
		{	
			m_sumMinDistMap.create(blurImage.rows,blurImage.cols, CV_32FC1);
			m_sumMinDistMap.setTo(cv::Scalar(0.0));

			m_RMap.create(m_sumMinDistMap.rows, m_sumMinDistMap.cols, CV_32FC1);
			m_RMap.setTo(cv::Scalar(m_defaultSubsampling));

			m_subSamplingMap.create(m_sumMinDistMap.rows, m_sumMinDistMap.cols, CV_32FC1);
			m_subSamplingMap.setTo(cv::Scalar(m_defaultR));
		}
		++runs;
	}
	//calc features of current image
	getFeatures(imgFeatures, &blurImage);
	double sumDist = 0.0;
	// variables to generate old average of gradient magnitude
	double maxNorm = 0.0;
	int glCounterFore = 0; 

	// for all pixels do:
	for (int y=0; y<height; ++y) 
	// for each row, algorithm processes rows sequentially
	{
		// for each column
		for(int x = 0; x < width; ++x) 
		{
			// compare current pixel value to bachground model
			int count = 0;  // used for #min
			int index = 0;  // index k = 1,...,N or k = 1,...,runs (runs != N)
			double dist = 0.0; // distance measure
			double temp = 0.0;
			double maxDist = 0.0;
			double maxDistB = 0.0;
			double minDist = 1000.0; // arbritrary large number for minDist
			int entry = randomGenerator.uniform(5, NUM_RANDOMGENERATION-5);

			do
			{
				
				// distance calculation
				double norm = abs((double)m_backgroundModel.at(index).gradMag.at<float>(y, x) - (double)imgFeatures.gradMag.at<float>(y, x));
				int pixVal = abs(m_backgroundModel.at(index).pxIntensity.at<uchar>(y, x) - imgFeatures.pxIntensity.at<uchar>(y,x));
				dist = (m_alpha*(norm/formerMaxNorm) + m_beta * ((double) pixVal)); // beta: second weighting factor in distance function

				if(dist < m_RMap.at<float>(y, x)) // match: smaller than pixel-depending threshold r
				{
					++count;

					if(minDist > dist)
						minDist = dist;
				}
				else
				{
					maxNorm += norm;
					++glCounterFore; 
				}

				++index;
			}
			while((count < m_minHits) && (index < runs)); // count << #min && index < runs, max(runs) = N

			// is BACKGROUND
			if(count >= m_minHits)
			{
				//set pixel to background value
				segMap.at<uchar>(y,x) = BACKGROUND_VAL;
				
				if(runs < N)
				{
						formerDistanceBack = 0; // since no distance value will be replaces, nothing need to be buffered for moving avg calculation
						m_minDistanceModel.at(runs-1).at<float>(y, x) = minDist;
						m_sumMinDistMap.at<float>(y, x) += m_minDistanceModel.at(runs - 1).at<float>(y, x);
				}

				//update model
				if(runs == N)
				{

					// Update current pixel
					// get random number between 0 and nrSubsampling-1
					int rand = 0;
					int updateCoeff = randomGenerator.uniform((int)0, (int)ceil(m_subSamplingMap.at<float>(y, x))); // 0 - 16
					if(updateCoeff < 1)// random subsampling, p(x) = 1/T
					{
						// replace randomly chosen sample
						rand = randomN.at(entry+1); //randomGenerator.uniform((int)0,(int)N-1);
					
						m_backgroundModel.at(rand).gradMag.at<float>(y, x) = imgFeatures.gradMag.at<float>(y, x);
						m_backgroundModel.at(rand).pxIntensity.at<uchar>(y, x) = imgFeatures.pxIntensity.at<uchar>(y, x);

						
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
				//pixel is foreground
				segMap.at<uchar>(y, x) = FOREGROUND_VAL;
			}


			meanDistBack = m_sumMinDistMap.at<float>(y, x) / runs;
			updateRThresholdXY(x, y, meanDistBack);
			updateSubsamplingXY(x, y, segMap.at<uchar>(y, x),meanDistBack);
		}
	}
	// calculate average gradient magnitude
	//double meanNorm = maxNorm / ((double)(glCounterFore + 1));
	double meanNorm = imgFeatures.getGradMagnMean();
	formerMaxNorm = (meanNorm > 100) ?  100 : meanNorm; //TODO: old value 20
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

const cv::Mat& PBAS::getRImg() const
{
	// treshold map
	return m_RMap;

}

const cv::Mat& PBAS::getTImg() const 
{
	// Learning rate map
	return m_subSamplingMap;
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
	if(x < 0)
	{
		x = 0;
	}
	else if(x >= width)
	{
		x = width -1;
	}

	if(y < 0)
	{		
		y = 0;
	}
	else if(y >= height)
	{
		y = height - 1;
	}	
}

void PBAS::getFeatures(PBASFeature& descriptor, cv::Mat* intImg)
{
	cv::Mat sobelX, sobelY;
	// features: gradient magnitude and direction and pixel intensities	
	cv::Sobel(*intImg, sobelX, CV_32F, 1, 0, 3, 1, 0.0); // get gradient magnitude for dx
	cv::Sobel(*intImg, sobelY, CV_32F, 0, 1, 3, 1, 0.0); // get gradient magnitude for dy
	cv::cartToPolar(sobelX,sobelY, descriptor.gradMag, sobelY, true); // convert cartesian to polar coordinates
	intImg->copyTo(descriptor.pxIntensity);
	
}

const double& PBAS::getAlpha() const {
	return m_alpha;
}

const double& PBAS::getBeta() const {
	return m_beta;
}


inline void PBAS::deallocMem(cv::Mat * mat) { mat->release();  delete mat; }
