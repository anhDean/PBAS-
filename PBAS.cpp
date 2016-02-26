#include "PBAS.h"
#include "LBSP.h"

int pbasCounter = 0; // satic member variable to track instances

PBAS::PBAS(void) : N(20), R(20), parts(2), nrSubsampling(16), foreground(255), background(0)
{
	++pbasCounter;
	//initialize background-model depending parameters
	//r-Thresh
	rThreshScale = 6.0;
	rIncDecFac = 0.05;
	//T-thresh
	increasingRateScale = 0.001;
	decreasingRateScale = 0.005;
	lowerTimeUpdateRateBoundary = 2;
	upperTimeUpdateRateBoundary = 300;
	
	//initialize background-model independent parameters
	formerMaxNorm = 1.0;
	formerMaxDir = 1.0;
	formerMaxPixVal = 1.0;
	height = 0; 
	width = 0;
	runs = 0;
	countOfRandomNumb = 100000;
	isMovement = false;

	beta = 1.0;
	alpha = 1.0;
	constForeground = 2.0;
	constBackground = 1.0;

}

PBAS::~PBAS(void)
{
	if(pbasCounter > 0)
	{
		randomSubSampling.clear();
		randomN.clear();
		randomX.clear();
		randomY.clear();
		randomDist.clear();
		tempDistB->release();
		std::cout << " befor rel " << std::endl;
		for(int i = 0; i < N; ++i)
		{
			if(~distanceStatisticBack.at(i)->empty())
				distanceStatisticBack.at(i)->release();
		
			if(~backGroundFeatures.at(i).gradMag.empty())
				backGroundFeatures.at(i).gradMag.release();
		
			if(~backGroundFeatures.at(i).gradAngle.empty())
				backGroundFeatures.at(i).gradAngle.release();
		
			if(~backGroundFeatures.at(i).pxIntensity.empty())
				backGroundFeatures.at(i).pxIntensity.release();
		}
		std::cout << " after rel " << std::endl;
		std::for_each(distanceStatisticBack.begin(), distanceStatisticBack.begin(), deallocMem); // deallocate memory for each allocated mat object
		distanceStatisticBack.clear();
		backGroundFeatures.clear();
		sumThreshBack.release();
		rThresh.release();
		tempCoeff.release();
		--pbasCounter;
	}		
}

void PBAS::initialization(int newN, double newR, int newParts, int newNrSubSampling, double a, double b, double cf, double cb, double rThrSc, double rIndDec, double incrTR, double decrTR, int lowerTB, int upperTB)
{
	N = newN;			// N: number of past samples
	R = newR;			// R: decision threshold
	setR = newR;		// R: decision threshold
	parts = newParts;
	nrSubsampling = newNrSubSampling; // 


	//r-Thresh
	rThreshScale = rThrSc; // R_scale
	rIncDecFac = rIndDec; // = R_inc/dec
	//T-thresh
	increasingRateScale = incrTR;		// T_inc
	decreasingRateScale = decrTR;		// T_dec
	lowerTimeUpdateRateBoundary = lowerTB; // T_lower
	upperTimeUpdateRateBoundary = upperTB;	// T_upper

	beta = b;
	alpha = a;
	constBackground = cb;
	constForeground = cf;
	createRandomNumberArray(); // create random numbers beforehand for neighbor, background and distance update
}

void PBAS::createRandomNumberArray()
{
	//pre calculate random number 
	for(int l = 0; l < countOfRandomNumb; l++)
	{
		randomN.push_back(cv::saturate_cast<int>(randomGenerator.uniform((int)0,(int)N))); 		// for Background model position coordinate, upper bound is excluded
		randomX.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring X  coordinate, upper bound is excluded
		randomY.push_back(cv::saturate_cast<int>(randomGenerator.uniform(-1, +2)));				// for neighboring Y coordinate, upper bound is excluded
		randomDist.push_back(cv::saturate_cast<int>(randomGenerator.uniform((int)0, (int)N))); // for Distance array posi  coordinate, upper bound is excluded
	}
}

bool PBAS::process(cv::Mat* input, cv::Mat* output)
{
	height = input->rows;
	width = input->cols;
	isMovement = false;
	assert(input->type() == CV_8UC1);
	//cv::Mat blurImage(input->rows, input->cols, CV_8UC1, input->data);
	cv::Mat blurImage = input->clone();
	
	if(runs < N)
		// if runs < N collect background features without updating the model
	{
		
		getFeatures(temp, &blurImage);
		backGroundFeatures.push_back(temp);

		tempDistB = new cv::Mat(blurImage.size(), CV_32FC1);
		distanceStatisticBack.push_back(tempDistB); // distanceStatisticBack: vector of Mat*, holds mean dist values for background
		if(runs == 0)
		// for the first run init R,T maps withdefault values
		{	
			sumThreshBack.create(blurImage.rows,blurImage.cols, CV_32FC1);
			rThresh.create(sumThreshBack.rows, sumThreshBack.cols, CV_32FC1);
			tempCoeff.create(sumThreshBack.rows, sumThreshBack.cols, CV_32FC1);
			for(int rows = 0; rows <rThresh.rows; ++rows)
			{
				float* pt = rThresh.ptr<float>(rows);
				float* ptSumB = sumThreshBack.ptr<float>(rows);
				float* ptTemp = tempCoeff.ptr<float>(rows);
				for(int cols = 0; cols < rThresh.cols; ++cols)
				{
					pt[cols] = setR; // initialize with default R value
					ptTemp[cols] = nrSubsampling; // initialize with default T, e.g. 16
					ptSumB[cols] = 0.0;
				}
			}
		}

		++runs;
	}

	segMap = new cv::Mat(blurImage.rows,blurImage.cols,blurImage.type());
	//calc features of current image
	getFeatures(imgFeatures, &blurImage);
	double sumDist = 0.0;
	// variables to generate old average of gradient magnitude
	double maxNorm = 0.0;
	int glCounterFore = 0; 

	// for all pixels do:
	for (int j=0; j< height; ++j) 
	// for each row, algorithm processes rows sequentially
	{
		segRowData = segMap->ptr<uchar>(j); // j-th row of segmentation map
		// get current image features, rows
		dataBriefNorm = imgFeatures.gradMag.ptr<float>(j); // gradmagn
		dataBriefDir  = imgFeatures.gradAngle.ptr<float>(j);  // grad dir
		dataBriefCol  = imgFeatures.pxIntensity.ptr<uchar>(j); // pixel intensity

		sumArrayDistBack = sumThreshBack.ptr<float>(j); // sum(dmin)
		rData = rThresh.ptr<float>(j);
		tCoeff = tempCoeff.ptr<float>(j); // current pixel-specific learning rate


		// for each row clear row container for features
		if(!backgroundPtBriefDir.empty())  // std::vector<float*>
			backgroundPtBriefDir.clear();
		
		if(!backgroundPtBriefNorm.empty()) // std::vector<float*>
			backgroundPtBriefNorm.clear();
		
		if(!backgroundPtBriefCol.empty())  // std::vector<uchar*>
			backgroundPtBriefCol.clear();

		if(!distanceStatPtBack.empty())    // std::vector<float*>
			distanceStatPtBack.clear();

		for(int k = 0; k < runs; ++k)
		{
			// get history of feature values up to N
			backgroundPtBriefNorm.push_back(backGroundFeatures.at(k).gradMag.ptr<float>(j));
			backgroundPtBriefDir.push_back(backGroundFeatures.at(k).gradAngle.ptr<float>(j));
			backgroundPtBriefCol.push_back(backGroundFeatures.at(k).pxIntensity.ptr<uchar>(j));
			distanceStatPtBack.push_back(distanceStatisticBack.at(k)->ptr<float>(j));
		}

		// for each column
		for(int i = 0; i < width; ++i) 
		{
			// compare current pixel value to bachground model
			int count = 0;  // used for #min
			int index = 0;  // index k = 1,...,N or k = 1,...,runs (runs != N)
			double dist = 0.0; // distance measure
			double temp = 0.0;
			double maxDist = 0.0;
			double maxDistB = 0.0;
			double minDist = 1000.0; // arbritrary large number for minDist
			int entry = randomGenerator.uniform(5, countOfRandomNumb-5);

			do
			{
				// calculate distance metric
				double norm = abs((double)backgroundPtBriefNorm.at(index)[i] - (double)*dataBriefNorm);
				double pixVal = abs(((double)backgroundPtBriefCol.at(index)[i]) - ((double)*dataBriefCol));
				
				dist = ((double)alpha*(norm/formerMaxNorm) + beta*pixVal); // beta: second weighting factor in distance function

				if(dist < *rData) // match: smaller than pixel-depending threshold r
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
			while((count < parts) && (index < runs)); // count << #min && index < runs, max(runs) = N

			// is BACKGROUND
			if(count >= parts)
			{
				//set pixel to background value
				*segRowData = 0;
				
				if(runs < N)
				{
						formerDistanceBack = 0; // since no distance value will be replaces, nothing need to be buffered for moving avg calculation
						distanceStatPtBack.at(runs-1)[i] = minDist;
						*sumArrayDistBack = *sumArrayDistBack + distanceStatPtBack.at(runs-1)[i];
				}

				//update model
				if(runs == N)
				{

					// Update current pixel
					// get random number between 0 and nrSubsampling-1
					int rand = 0;
					int updateCoeff = randomGenerator.uniform((int)0, (int)ceil(*tCoeff)); // 0 - 16
					if(updateCoeff < 1)// random subsampling, p(x) = 1/T
					{
						// replace randomly chosen sample
						rand = randomN.at(entry+1); //randomGenerator.uniform((int)0,(int)N-1);

						backgroundPtBriefNorm.at(rand)[i] = (float)*dataBriefNorm;
						backgroundPtBriefDir.at(rand)[i] = (float)*dataBriefDir;
						backgroundPtBriefCol.at(rand)[i] = (uchar)*dataBriefCol;

						
						formerDistanceBack = distanceStatPtBack.at(randomDist.at(entry))[i]; // save old dmin
						distanceStatPtBack.at(randomDist.at(entry))[i] = minDist; // replace old entry with new dmin
						*sumArrayDistBack = *sumArrayDistBack + distanceStatPtBack.at(randomDist.at(entry))[i] - formerDistanceBack; // calculate current sum of dmins

					}

					// Update neighboring background model
					updateCoeff = randomGenerator.uniform((int)0, (int)ceil(*tCoeff));
					if(updateCoeff < 1)// random subsampling
					{
						//choose neighboring pixel randomly
						xNeigh = randomX.at(entry) + i;
						yNeigh = randomY.at(entry) + j;
						checkValid(&xNeigh, &yNeigh);

						// replace randomly chosen sample
						rand = randomN.at(entry-1); 
						(backGroundFeatures.at(rand)).gradMag.at<float>(yNeigh,xNeigh) = imgFeatures.gradMag.at<float>(yNeigh,xNeigh);
						(backGroundFeatures.at(rand)).gradAngle.at<float>(yNeigh,xNeigh) = imgFeatures.gradAngle.at<float>(yNeigh,xNeigh);
						(backGroundFeatures.at(rand)).pxIntensity.at<uchar>(yNeigh,xNeigh) = imgFeatures.pxIntensity.at<uchar>(yNeigh,xNeigh);
					}
				}


			
			}
			
			
			else
			{				
				//pixel is foreground
				*segRowData = 255;		
				isMovement = true; 
			}


			//update R threshold of pixel x(j,i) depending on mean(dmin)
			updateThreshold();
			
			//time update, adjust learning rate
			float tempT = *tCoeff;
			(*segRowData < 128)? tempT -= increasingRateScale / (meanDistBack + 1) : tempT += decreasingRateScale / (meanDistBack + 1);  
		
			if(tempT > lowerTimeUpdateRateBoundary && tempT < upperTimeUpdateRateBoundary )
				*tCoeff = tempT;

			++segRowData;
			++dataBriefNorm;
			++dataBriefDir;
			++dataBriefCol;
			++sumArrayDistBack;
			++rData;
			++tCoeff;
		}
	}
	// calculate average gradient magnitude
	double meanNorm = maxNorm / ((double)(glCounterFore + 1));

	formerMaxNorm = (meanNorm > 20) ? meanNorm : 20;
	// write segmentation result to output
	segMap->copyTo(*output);

	delete segMap;
	imgFeatures.free();
	blurImage.release();
	return true;
}


inline void PBAS::updateThreshold() {
	//update R threshold of pixel x(j,i)
	this->meanDistBack = float(*(this->sumArrayDistBack)) / (float)this->runs; // mean(dmin) = sum(dmin)/runs
	float tempR = *(this->rData); // current R(x_i)
	float incDec = this->rIncDecFac;
	if (tempR < (this->meanDistBack) *(this->rThreshScale))
	{
		tempR += tempR * this->rIncDecFac / 10.0;

	}
	else
	{
		tempR -= tempR *this->rIncDecFac / 10.0;
	}

	if (tempR >= this->setR)
		*(this->rData) = tempR;
	else
		*(this->rData) = this->setR;
}

cv::Mat* PBAS::getRImage()
{
	// treshold map
	return &rThresh;

}

cv::Mat* PBAS::getTImage()
{
	// Learning rate map
	return &tempCoeff;
}
double PBAS::getR()
{
	return R;
}

bool PBAS::isMoving()
{
	return isMovement;
}

void PBAS::setConstForeground(double constF)
{
	constForeground = constF;
}
void PBAS::setConstBackground(double constB)
{
	constBackground = constB;
}

void PBAS::setAlpha(double alph)
{
	alpha = alph;
}

void PBAS::setBeta(double bet)
{
	beta = bet;
}

void PBAS::checkXY(cv::Point2i *p0)
{
	if(p0->x < 0)
	{
		p0->x = 0;
	}
	else if(p0->x > width)
	{
		p0->x = width - 1;
	}

	if(p0->y < 0)
	{
		p0->y = 0;
	}
	else if(p0->y > height)
	{
		p0->y = height - 1;
	}
}

void PBAS::checkValid(int *x, int *y)
{
	if(*x < 0)
	{
		*x = 0;
	}
	else if(*x >= width)
	{
		*x = width -1;
	}

	if(*y < 0)
	{		
		*y = 0;
	}
	else if(*y >= height)
	{
		*y = height - 1;
	}	
}

void PBAS::getFeatures(PBASFeature& descriptor, cv::Mat* intImg)
{
	// features: gradient magnitude and direction and pixel intensities	
	cv::Sobel(*intImg,sobelX,CV_32F, 1, 0, 3, 1, 0.0); // get gradient magnitude for dx
	cv::Sobel(*intImg,sobelY,CV_32F, 0, 1, 3, 1, 0.0); // get gradient magnitude for dy
	
	PBASFeature tmp_feat;
	
	cv::cartToPolar(sobelX,sobelY, tmp_feat.gradMag, tmp_feat.gradAngle, true); // convert cartesian to polar coordinates
	
	//first  matrix: magnitude
	//second matrix: direction
	// third matrix: gray image, intensity
	intImg->copyTo(tmp_feat.pxIntensity);
	descriptor= tmp_feat; // deep copy
}

inline void PBAS::deallocMem(cv::Mat * mat) { delete mat;}
