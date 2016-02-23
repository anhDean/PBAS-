#include "stdafx.h"
#include "PBAS.h"

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
		
			if(~backGroundFeatures.at(i).at(0).empty())
				backGroundFeatures.at(i).at(0).release();
		
			if(~backGroundFeatures.at(i).at(1).empty())
				backGroundFeatures.at(i).at(1).release();
		
			if(~backGroundFeatures.at(i).at(2).empty())
				backGroundFeatures.at(i).at(2).release();
		}
		std::cout << " after rel " << std::endl;
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
	N = newN;
	R = newR;
	setR = newR;
	parts = newParts;
	nrSubsampling = newNrSubSampling;


	//r-Thresh
	rThreshScale = rThrSc;
	rIncDecFac = rIndDec;
	//T-thresh
	increasingRateScale = incrTR;
	decreasingRateScale = decrTR;
	lowerTimeUpdateRateBoundary = lowerTB;
	upperTimeUpdateRateBoundary = upperTB;

	beta = b;
	alpha = a;
	constBackground = cb;
	constForeground = cf;
	createRandomNumberArray();
}

void PBAS::createRandomNumberArray()
{
	//pre calculate random number 
	for(int l = 0; l < countOfRandomNumb; l++)
	{
		randomN.push_back((int)randomGenerator.uniform((int)0,(int)N));
		randomX.push_back((int)randomGenerator.uniform(-1, +2));
		randomY.push_back((int)randomGenerator.uniform(-1, +2));
		randomDist.push_back((int)randomGenerator.uniform((int)0, (int)N));
	}
}

bool PBAS::process(cv::Mat* input, cv::Mat* output)
{
	height = input->rows;
	width = input->cols;
	isMovement = false;

	cv::Mat blurImage(input->rows, input->cols, CV_8UC1, input->data);
	
	if(height < 10 || width < 10)
	{
		std::cout << "Error: Occurrence of different image size in PBAS. STOPPING " << std::endl;
		return false;
	}
	else if(height == 0 || width == 0)
	{
		std::cout << "Error: Width/height not set. Initialize first. width: "<< width << " height: " << height << std::endl;
		return false;
	}


	if(runs < N)
	{
		getFeatures(&temp, &blurImage);
		backGroundFeatures.push_back(temp);

		tempDistB = new cv::Mat(blurImage.size(), CV_32FC1);
		distanceStatisticBack.push_back(tempDistB);
		if(runs == 0)
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
					pt[cols] = setR;
					ptTemp[cols] = nrSubsampling;
					ptSumB[cols] = 0.0;
				}
			}
		}

		++runs;
	}

	segMap = new cv::Mat(blurImage.rows,blurImage.cols,blurImage.type());

	//calc features
	getFeatures(&imgFeatures, &blurImage);
	double sumDist = 0.0;
	double maxNorm = 0.0;
	int glCounterFore = 0;


	// for each pixel
	for (int j=0; j< height/*-5*/; ++j/*+=5*/) 
	{
		segData = segMap->ptr<uchar>(j);
		dataBriefNorm = imgFeatures.at(0).ptr<float>(j);
		dataBriefDir = imgFeatures.at(1).ptr<float>(j);
		dataBriefCol = imgFeatures.at(2).ptr<uchar>(j);
		sumArrayDistBack = sumThreshBack.ptr<float>(j);
		rData = rThresh.ptr<float>(j);
		tCoeff = tempCoeff.ptr<float>(j);

		if(!backgroundPtBriefDir.empty())
			backgroundPtBriefDir.clear();
		
		if(!backgroundPtBriefNorm.empty())
			backgroundPtBriefNorm.clear();
		
		if(!backgroundPtBriefCol.empty())
			backgroundPtBriefCol.clear();

		if(!distanceStatPtBack.empty())
			distanceStatPtBack.clear();

		for(int k = 0; k < runs; ++k)
		{
			backgroundPtBriefNorm.push_back(backGroundFeatures.at(k).at(0).ptr<float>(j));
			backgroundPtBriefDir.push_back(backGroundFeatures.at(k).at(1).ptr<float>(j));
			backgroundPtBriefCol.push_back(backGroundFeatures.at(k).at(2).ptr<uchar>(j));
			distanceStatPtBack.push_back(distanceStatisticBack.at(k)->ptr<float>(j));
		}


		for(int i = 0; i < width; ++i) 
		{
			// compare current pixel value to bachground model
			int count = 0; 
			int index = 0; 
			double dist = 0.0;
			double temp = 0.0;
			double maxDist = 0.0;
			double maxDistB = 0.0;
			double minDist = 1000.0;
			int entry = randomGenerator.uniform(5, countOfRandomNumb-5);

			do
			{
				double norm = abs((double)backgroundPtBriefNorm.at(index)[i] - (double)*dataBriefNorm);
				double pixVal = abs(((double)backgroundPtBriefCol.at(index)[i]) - ((double)*dataBriefCol));
				
				dist = ((double)alpha*(norm/formerMaxNorm) + beta*pixVal);

				if(dist < *rData)
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
			while((count < parts) && (index < runs));

			// is BACKGROUND
			if(count >= parts)
			{
				//set pixel to background
				*segData = 0;
				
				if(runs < N)
				{

						formerDistanceBack = 0;
						distanceStatPtBack.at(runs-1)[i] = minDist;
						*sumArrayDistBack = *sumArrayDistBack + distanceStatPtBack.at(runs-1)[i];
				}

				//update model
				if(runs == N)
				{

					// Update current pixel
					// get random number between 0 and nrSubsampling-1
					int rand = 0;
					int updateCoeff = randomGenerator.uniform((int)0, (int)ceil(*tCoeff));
					if(updateCoeff < 1)// random subsampling
					{
						// replace randomly chosen sample
						rand = randomN.at(entry+1); //randomGenerator.uniform((int)0,(int)N-1);

						backgroundPtBriefNorm.at(rand)[i] = (float)*dataBriefNorm;
						backgroundPtBriefDir.at(rand)[i] = (float)*dataBriefDir;
						backgroundPtBriefCol.at(rand)[i] = (uchar)*dataBriefCol;

						formerDistanceBack = distanceStatPtBack.at(randomDist.at(entry))[i];
						distanceStatPtBack.at(randomDist.at(entry))[i] = minDist;
						*sumArrayDistBack = *sumArrayDistBack + distanceStatPtBack.at(randomDist.at(entry))[i] - formerDistanceBack;

					}

					// Update neighboring background model
					updateCoeff = randomGenerator.uniform((int)0, (int)ceil(*tCoeff));
					if(updateCoeff < 1)// random subsampling
					{
						//choose neighboring pixel randomly
						xNeigh = randomX.at(entry)+i;
						yNeigh = randomY.at(entry)+j;
						checkValid(&xNeigh, &yNeigh);

						// replace randomly chosen sample
						rand = randomN.at(entry-1); 
						(backGroundFeatures.at(rand)).at(0).at<float>(yNeigh,xNeigh) = imgFeatures.at(0).at<float>(yNeigh,xNeigh);
						(backGroundFeatures.at(rand)).at(1).at<float>(yNeigh,xNeigh) = imgFeatures.at(1).at<float>(yNeigh,xNeigh);
						(backGroundFeatures.at(rand)).at(2).at<uchar>(yNeigh,xNeigh) = imgFeatures.at(2).at<uchar>(yNeigh,xNeigh);
					}
				}


			//update R threshould of pixel x(j,i)
			meanDistBack = (float)(*sumArrayDistBack) / (float)runs;
			float tempR = *rData;
			if(*rData < (meanDistBack)*rThreshScale)
			{
				tempR += *rData*rIncDecFac;

			}
			else 
			{
				tempR -= *rData*rIncDecFac;
			}

			if(tempR >= setR)
				*rData = tempR;	
			else
				*rData = setR;
			
			}
			else
			{
				
				//pixel is foreground
				*segData = 255;

				//update R threshold of pixel x(j,i)
				meanDistBack = (float)(*sumArrayDistBack) / (float)runs;
				float tempR = *rData;
				if(*rData < (meanDistBack)*rThreshScale)
				{
					tempR += *rData*rIncDecFac/10.0; 

				}
				else 
				{
					tempR -= *rData*rIncDecFac/10.0;
				}

				if(tempR >= setR)
					*rData = tempR;	
				else
					*rData = setR;
			
				isMovement = true; 
			}

			
			//time update
			float tempT = *tCoeff;
			if(*segData < 128)
			{
				tempT -= increasingRateScale/(meanDistBack+1);

			}
			else
			{
				tempT += decreasingRateScale/(meanDistBack+1); 
			}
			if(tempT > lowerTimeUpdateRateBoundary && tempT < upperTimeUpdateRateBoundary )
				*tCoeff = tempT;


			++segData;
			++dataBriefNorm;
			++dataBriefDir;
			++dataBriefCol;
			++sumArrayDistBack;
			++rData;
			++tCoeff;
		}
	}

	segMap->copyTo(*output);
	
	double meanNorm = maxNorm/((double)(glCounterFore+1));


	if(meanNorm > 20)
	{
		formerMaxNorm = meanNorm;
	}
	else
	{
		formerMaxNorm = 20;
	}

	delete segMap;
	imgFeatures.at(0).release();
	imgFeatures.at(1).release();
	imgFeatures.at(2).release();
	blurImage.release();
	return true;
}

cv::Mat* PBAS::getRImage()
{
	return &rThresh;

}

cv::Mat* PBAS::getTImage()
{
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

void PBAS::getFeatures(std::vector<cv::Mat>* descriptor, cv::Mat* intImg) 
{
	// features: gradient magnitude and direction
	if(!descriptor->empty())
		descriptor->clear();
	
	cv::Sobel(*intImg,sobelX,CV_32F,1,0, 3, 1, 0.0);
	cv::Sobel(*intImg,sobelY,CV_32F,0,1, 3, 1, 0.0);
	
	cv::Mat norm, dir;
	cv::Mat temp;
	
	cv::cartToPolar(sobelX,sobelY,norm,dir, true);
	
	
	
	//first entry: norm
	//second entry: direction
	descriptor->push_back(norm);
	descriptor->push_back(dir);
	
	intImg->copyTo(temp);
	descriptor->push_back(temp);
}
