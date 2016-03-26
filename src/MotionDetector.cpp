#include "MotionDetector.h"


void MotionDetector::initialize(const cv::Mat& firstFrame)
{
	CV_Assert((firstFrame.channels() == 3 || firstFrame.channels() == 1));
	if (firstFrame.channels() == 3)
		cv::cvtColor(firstFrame, m_prev_grey, CV_BGR2GRAY);
	else
		firstFrame.copyTo(m_prev_grey);

	m_staticObjectCandidates.clear();
	m_staticObjectArea = cv::Mat::zeros(firstFrame.size(), CV_8U);
	m_staticBg = cv::Mat::ones(firstFrame.size(), CV_8U);
	m_dynamicBg = cv::Mat::zeros(firstFrame.size(), CV_8U);
	m_movingForeground = cv::Mat::zeros(firstFrame.size(), CV_8U);
	tmp_candidates = cv::Mat::zeros(firstFrame.size(), CV_8U);
}
void MotionDetector::calcOpticalFlow(const cv::Mat& input, cv::Mat& output)
{

	CV_Assert(m_prev_grey.data != NULL && (input.channels() == 3 || input.channels() == 1));
	cv::Mat inputGray, flowUmat;
	std::vector<cv::Mat> flowchannels;
	if (input.channels() == 3)
		cv::cvtColor(input, inputGray, CV_BGR2GRAY);
	else
		input.copyTo(inputGray);

	cv::calcOpticalFlowFarneback(m_prev_grey, inputGray, output, 0.5, 3, 20, 5, 8, 1.2, 0);
	inputGray.copyTo(m_prev_grey);

}
cv::Mat MotionDetector::drawFlowField(const cv::Mat& input, const cv::Mat& flowMatXY)
{
	CV_Assert(flowMatXY.channels() == 2);
	cv::Mat original = input.clone();

	// By y += 5, x += 5 you can specify the grid 
	for (int y = 0; y < flowMatXY.rows; y += 5) {
		for (int x = 0; x < flowMatXY.cols; x += 5)
		{
			// get the flow from y, x position * 10 for better visibility
			const cv::Point2f flowatxy = flowMatXY.at<cv::Point2f>(y, x) * 2;
			// draw line at flow direction
			line(original, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), cv::Scalar(255, 255, 255));
			// draw initial point
			circle(original, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
		}
	}
	return original;
}
cv::Mat MotionDetector::opticalFlowToAbs(const cv::Mat& optical_flow)
{
	cv::Mat result;
	std::vector<cv::Mat> flowchannels;
	cv::split(optical_flow, flowchannels);
	cv::cartToPolar(flowchannels[0], flowchannels[1], result, flowchannels[0]);
	return result;

}
std::vector<std::vector<cv::Point> > MotionDetector::getContours(const cv::Mat& input)
{
	cv::Mat cp = input.clone();
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(cp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	return contours;
}
void MotionDetector::drawBoundingBoxes(const cv::Mat& input, cv::Mat& output, const std::vector<cv::Rect>& bb)
{
	cv::RNG rng(12345);

	//Draw contours
	std::vector<cv::Mat> channels;
	channels.push_back(input); channels.push_back(input); channels.push_back(input);
	cv::Mat drawing;
	cv::merge(channels, drawing);

	for (int i = 0; i< bb.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(drawing, bb[i].tl(), bb[i].br(), color, 0.5, 8, 0);
	}


	drawing.copyTo(output);

}
void MotionDetector::updateMotionMap(const cv::Mat& inputRGB)
{
	cv::Mat opticalFlow;
	calcOpticalFlow(inputRGB, opticalFlow);
	m_motionMap =  opticalFlowToAbs(opticalFlow);
	cv::GaussianBlur(m_motionMap, m_motionMap, cv::Size(3, 3), 2);
	cv::threshold(m_motionMap, m_motionMap, 0.25, 100, CV_THRESH_BINARY);
}
void MotionDetector::updateCandidates(const cv::Mat& segmentationMap)
{
	cv::Rect tmp_rect;
	std::vector<std::vector<cv::Point> > contours = getContours(segmentationMap);

	for (int i = 0; i< contours.size(); i++)
	{
		tmp_rect = cv::boundingRect(cv::Mat(contours[i]));
		evaluateStaticObjRect(tmp_rect);
	}
	verifyStaticCandidates(segmentationMap);
}
void MotionDetector::verifyStaticCandidates(const cv::Mat& segmentationMap)
{
	std::set<int> deleteIdx;
	std::vector<AmbiguousCandidate> copyCandidates;
	
	float tmp_fgSumPx;
	int tmp_counter;
	cv::Rect tmpBb;

	
	for (int i = 0; i < m_staticObjectCandidates.size(); ++i)
	{
		tmp_counter = m_staticObjectCandidates[i].counter + 1;
		tmpBb = m_staticObjectCandidates[i].boundingBox;
		tmp_fgSumPx = cv::sum(segmentationMap(tmpBb))[0] / 255.0;

		if  (tmp_fgSumPx /tmpBb.area() < 0.35 || (m_staticObjectCandidates[i].accMovement + cv::sum(m_motionMap(tmpBb))[0]) / tmp_counter > 0.55 * tmpBb.area())
		{		
			m_staticObjectArea(tmpBb) = 0;
		}

		else
		{
			++m_staticObjectCandidates[i].counter;
			m_staticObjectCandidates[i].accMovement += cv::sum(m_motionMap(tmpBb))[0];
			copyCandidates.push_back(m_staticObjectCandidates[i]);
			m_staticObjectArea(tmpBb) = 255;
		}
	}
	m_staticObjectCandidates = copyCandidates;
}
void MotionDetector::classifyStaticCandidates(const cv::Mat& input, const cv::Mat& bgModel, const cv::Mat& seg)
{
	int classificationCounter = 20;
	cv::Mat tmp_bgROI, tmp_segROI, tmp_inputROI;
	cv::Mat inputSegDiff, segBgDiff, inputBgDiff;
	cv::Mat segmentationInv, segAddedBg;

	int ghostSum, intermittentSum;
	float epsilon = 0.20;

	for (int i = 0; i < m_staticObjectCandidates.size(); ++i)
	{
		if (m_staticObjectCandidates[i].counter > classificationCounter)
		{
			tmp_inputROI = drawContours(input(m_staticObjectCandidates[i].boundingBox));
			tmp_segROI = drawContours(seg(m_staticObjectCandidates[i].boundingBox));
			tmp_bgROI = drawContours(bgModel(m_staticObjectCandidates[i].boundingBox));
			
			cv::matchTemplate(tmp_inputROI, tmp_segROI, inputSegDiff, CV_TM_CCORR_NORMED);
			cv::matchTemplate(tmp_bgROI, tmp_segROI, segBgDiff, CV_TM_CCORR_NORMED);


			if (segBgDiff.at<float>(0, 0) > inputSegDiff.at<float>(0, 0) + epsilon)
			{
				m_staticObjectCandidates[i].state = DETECTOR_GHOST_OBJECT;
			}

				

			else if (inputSegDiff.at<float>(0, 0) > segBgDiff.at<float>(0, 0) + epsilon)
			{
				m_staticObjectCandidates[i].state = DETECTOR_STATIC_OBJECT;
			}
				
			else
			{
				m_staticObjectCandidates[i].state = DETECTOR_UNCLASSIFIED_OBJECT;
			}
							
			
			/*
			cv::Mat canvas = cv::Mat(tmp_inputROI.rows, tmp_inputROI.cols + tmp_segROI.cols + tmp_bgROI.cols, CV_8U, cv::Scalar(255));
			tmp_bgROI.copyTo(canvas(cv::Range(0, tmp_bgROI.rows), cv::Range(0, tmp_bgROI.cols)));
			tmp_segROI.copyTo(canvas(cv::Range(0, tmp_segROI.rows), cv::Range(tmp_bgROI.cols, tmp_bgROI.cols + tmp_segROI.cols)));
			tmp_inputROI.copyTo(canvas(cv::Range(0, tmp_bgROI.rows), cv::Range(tmp_bgROI.cols + tmp_segROI.cols, tmp_bgROI.cols + tmp_segROI.cols + tmp_inputROI.cols)));
			cv::imshow("1,1) bg 1,2) seg 1,3)input 2,1) and(input,seg) 2,2) and(bg,seg)", canvas);

			std::cout << "0: static" << std::endl;
			std::cout << "1: ghost" << std::endl;
			std::cout << "5: unclassified" << std::endl;
			std::cout << "classification result: " << m_staticObjectCandidates[i].state << std::endl;
			std::cout << "corr(seg,bg): " << segBgDiff.at<float>(0, 0) << std::endl;
			std::cout << "corr(seg,input): " << inputSegDiff.at<float>(0, 0) << std::endl;
			cv::waitKey(20);
			*/


		

		}

	}

}

const cv::Mat& MotionDetector::getMotionMap() const
{
	return m_motionMap;
}
const cv::Mat& MotionDetector::getStaticObjectArea() const
{
	return m_staticObjectArea;
}
const std::vector<AmbiguousCandidate>& MotionDetector::getStaticObjects() const
{
	return m_staticObjectCandidates;
}


cv::Mat MotionDetector::drawContours(const cv::Mat& input)
{
	int channels = input.channels();
	cv::Mat input_gray;

	int thresholdLevel = 250;
	int linePx = 2;
	cv::Scalar color = cv::Scalar(255);

	CV_Assert(channels == 1 ||channels == 3);
	if (channels == 3)
	{
		cv::cvtColor(input, input_gray, CV_BGR2GRAY);
		cv::Canny(input_gray, input_gray, thresholdLevel, thresholdLevel * 2, 3);
	}
	else
	{
		cv::Canny(input, input_gray, thresholdLevel, thresholdLevel * 2, 3);
	}

	cv::Mat drawing = cv::Mat::zeros(input_gray.size(), CV_8U);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(input_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	for (int i = 0; i< contours.size(); ++i)
	{	
		cv::drawContours(drawing, contours, i, color, linePx, 8, hierarchy, 0, cv::Point());
	}

	return drawing;
}


void MotionDetector::drawBoundingBoxesClassified(const cv::Mat& input, cv::Mat& output, const std::vector<AmbiguousCandidate>& bb)
{

	//Draw contours
	std::vector<cv::Mat> channels;
	cv::Mat drawing;
	if (input.channels() == 1)
	{
		channels.push_back(input); channels.push_back(input); channels.push_back(input);
		cv::merge(channels, drawing);
	}

	else
		input.copyTo(drawing);
	
	
	

	cv::Scalar staticColor = cv::Scalar(0, 255, 0), ghostColor = cv::Scalar(255, 0, 0);
	cv::Scalar movingObjColor = cv::Scalar(0, 0, 255);
	cv::Scalar dynamicBgColor = cv::Scalar(255, 255, 0);
	for (int i = 0; i< bb.size(); i++)
	{
		if(bb[i].state == DETECTOR_GHOST_OBJECT)
			cv::rectangle(drawing, bb[i].boundingBox.tl(), bb[i].boundingBox.br(), ghostColor, 0.5, 8, 0);
		else if (bb[i].state == DETECTOR_STATIC_OBJECT)
			cv::rectangle(drawing, bb[i].boundingBox.tl(), bb[i].boundingBox.br(), staticColor, 0.5, 8, 0);
		else if (bb[i].state == DETECTOR_MOVING_OBJECT)
			cv::rectangle(drawing, bb[i].boundingBox.tl(), bb[i].boundingBox.br(), movingObjColor, 0.5, 8, 0);
		else if (bb[i].state == DETECTOR_DYNAMIC_BACKGROUND)
			cv::rectangle(drawing, bb[i].boundingBox.tl(), bb[i].boundingBox.br(), dynamicBgColor, 0.5, 8, 0);
	}
	drawing.copyTo(output);
}

bool MotionDetector::evaluateStaticObjRect(cv::Rect& tmp_rect)
{

	int tmp_area = tmp_rect.area();
	bool isChild = false;
	bool isParent = false;
	bool result = false;

	if (tmp_area > m_MIN_BLOBSIZE) // reject rects that are to small
	{
		if (!isInMotion(tmp_rect, 0.75)) // reject rects that are in motion
		{

			if (!(m_staticObjectCandidates.size() > m_MAX_STATIC_OBJ)) // reject rects after max count
			{
				AmbiguousCandidate tmp;
				tmp.boundingBox = tmp_rect;
				tmp.counter = 0;
				tmp.accMovement = 0;
				tmp.state = DETECTOR_UNCLASSIFIED_OBJECT;

				std::set<int> deleteIdx;
				int iteration = 0;


				for (int k = 0; k < m_staticObjectCandidates.size(); ++k)
				{
					if (containsBoundingBox(m_staticObjectCandidates[k].boundingBox, tmp_rect)) // is within
					{
						isChild = true;
						result = false;
						return result;
					}

					if (containsBoundingBox(tmp_rect, m_staticObjectCandidates[k].boundingBox)) // contains existing bounding box
					{
						if (isParent)
						{
							deleteIdx.insert(k);
						}
						else
						{
							isParent = true;
							result = true;
							m_staticObjectCandidates[k].boundingBox = tmp_rect;
						}
					}
				}

				for (auto it = deleteIdx.begin(); it != deleteIdx.end(); ++it)
				{
					m_staticObjectCandidates.erase(m_staticObjectCandidates.begin() + *it - iteration);
					++iteration;
				}

				if (!isParent && !isChild)
				{
					m_staticObjectCandidates.push_back(tmp);
					result = true;
				}
			}

		}

	}
	return result;

}



bool MotionDetector::isInMotion(cv::Rect boundingBox, float interSection)
{
	if (cv::sum(m_motionMap(boundingBox))[0] > interSection * boundingBox.area())
		return true;
	else
		return false;

}

bool MotionDetector::containsBoundingBox(cv::Rect outer, cv::Rect inner)
{
	float intersectionParam = 0.45;
	cv::Rect intersect = outer & inner;
	if (intersect .area() >= intersectionParam * inner.area())
		return true;
	else
		return false;
}










/* TODO: detection and classification of moving foreground objects
bool MotionDetector::evaluateMovingObjRect(cv::Rect& tmp_rect)
{

int tmp_area = tmp_rect.area();
bool isChild = false;
bool isParent = false;
bool result = false;

if (tmp_area > m_MIN_BLOBSIZE) // reject rects that are to small
{
if (isInMotion(tmp_rect, 0.6)) // reject rects that are in motion
{

if (!(m_movingObjCandidates.size() > m_MAX_MOVING_OBJ)) // reject rects after max count
{
AmbiguousCandidate tmp;
tmp.boundingBox = tmp_rect;
tmp.counter = 0;
tmp.accMovement = 0;
tmp.state = DETECTOR_UNCLASSIFIED_OBJECT;

std::set<int> deleteIdx;
int iteration = 0;


for (int k = 0; k < m_movingObjCandidates.size(); ++k)
{
if (containsBoundingBox(m_movingObjCandidates[k].boundingBox, tmp_rect)) // is within
{
isChild = true;
result = false;
return result;
}

if (containsBoundingBox(tmp_rect, m_movingObjCandidates[k].boundingBox)) // contains existing bounding box
{
if (isParent)
{
deleteIdx.insert(k);
}
else
{
isParent = true;
result = true;
m_movingObjCandidates[k].boundingBox = tmp_rect;
}
}
}

for (auto it = deleteIdx.begin(); it != deleteIdx.end(); ++it)
{
m_movingObjCandidates.erase(m_movingObjCandidates.begin() + *it - iteration);
++iteration;
}

if (!isParent && !isChild)
{
m_movingObjCandidates.push_back(tmp);
result = true;
}

}

}

}

return result;

}
void MotionDetector::classifyMovingCandidates(const cv::Mat& seg, const cv::Mat noiseMap)
{
float fgArea, noiseArea;
float noiseThreshold = 2.0;
int minFrameNum = 3;
int bbShrinkage = 4;


for (int i = 0; i < m_movingObjCandidates.size(); ++i)
{
if (isInMotion(m_movingObjCandidates[i].boundingBox, 0.65) && m_movingObjCandidates[i].counter > minFrameNum) // check if it is moving
{
fgArea = cv::mean(seg(m_movingObjCandidates[i].boundingBox))[0] / 255.0; // foreground between 0 - 1
if (fgArea > 0.6) // check if classified as foreground
{
cv::Rect innerBb(m_movingObjCandidates[i].boundingBox);
innerBb.height -= bbShrinkage;
innerBb.width -= bbShrinkage;
innerBb.x += bbShrinkage / 2;
innerBb.y += bbShrinkage / 2;

if (cv::mean(noiseMap(innerBb))[0] < noiseThreshold) // check noise within bounding box
m_movingObjCandidates[i].state = DETECTOR_MOVING_OBJECT;
else
m_movingObjCandidates[i].state = DETECTOR_DYNAMIC_BACKGROUND;
}
}
}

}


void MotionDetector::verifyMovingCandidates(const cv::Mat& segmentationMap)
{
	std::set<int> deleteIdx;
	std::vector<AmbiguousCandidate> copyCandidates;

	int tmp_fgSumPx;
	int tmp_counter;
	cv::Rect tmpBb;

	for (int i = 0; i < m_movingObjCandidates.size(); ++i)
	{
		tmp_counter = m_movingObjCandidates[i].counter + 1;
		tmpBb = m_movingObjCandidates[i].boundingBox;

		tmp_fgSumPx = cv::sum(segmentationMap(tmpBb))[0];

		if (tmp_fgSumPx / (255.0 * tmpBb.area()) > 0.65 && (m_movingObjCandidates[i].accMovement + cv::sum(m_motionMap(tmpBb))[0]) / tmp_counter > 0.85 * tmpBb.area())
		{
			++m_movingObjCandidates[i].counter;
			m_movingObjCandidates[i].accMovement += cv::sum(m_motionMap(tmpBb))[0];
			copyCandidates.push_back(m_movingObjCandidates[i]);
			tmp_candidates(tmpBb) = 255;
		}

		else
		{
			tmp_candidates(tmpBb) = 0;
		}
	}
	m_movingObjCandidates = copyCandidates;
}



const std::vector<AmbiguousCandidate>& MotionDetector::getMovingObjects() const
{
	return m_movingObjCandidates;
}
*/


