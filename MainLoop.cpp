#include "StdAfx.h"
#include "MainLoop.h"
#include <iomanip>

MainLoop::MainLoop(void)
{

}

MainLoop::~MainLoop(void)
{
}

void MainLoop::startVideoProcessing()
{
	//define Videos with scene and maybe different parameters
	defineVideoParamters();

	for(int actualParam = 0; actualParam < outputBaseline.size(); ++ actualParam)
	{
		std::cout << outputBaseline.at(actualParam) << std::endl;
		for(int actualVideo = 0; actualVideo < imgString.size()/outputBaseline.size(); ++actualVideo)
		{
			std::cout <<"paramSet: " << actualParam << "actualVideo: " << actualVideo << std::endl;
			std::string path;
			setOutputPath.at(actualVideo).clear();
			//create directories
			path = outputBaseline.at(actualParam);

			CreateDirectoryA(path.c_str(), NULL);
			path += "/";
			path += baselineString.at(actualVideo);
			CreateDirectoryA(path.c_str(), NULL);
			path += "/";
			path += setOutputVideo.at(actualVideo);
			if(!CreateDirectoryA(path.c_str(), NULL))
			{
				std::cerr << "Couldn't create directory" << std::endl;
			}

			path += setOutputPath.at(actualVideo);
			setOutputPath.at(actualVideo) = path;
			setOutputPath.at(actualVideo) += "/object_";
			if(!CreateDirectoryA(path.c_str(), NULL))
			{
				std::cerr << "Couldn't create directory" << std::endl;
			}
			path += "/bin";

			processor = new VideoProcessor;

			// Create feature tracker instance
			tracker = new FeatureTracker(processor,  resParam.at(actualVideo),
				newN.at(actualParam), newR.at(actualParam), newRaute.at(actualParam), newTemporal.at(actualParam), //const PBAS
				rThreshScale.at(actualParam), rIncDecFac.at(actualParam), increasingRateScale.at(actualParam), decreasingRateScale.at(actualParam), lowerTimeUpdateRateBoundary.at(actualParam), //const PBAS
				upperTimeUpdateRateBoundary.at(actualParam),//const PBAS
				newAlpha.at(actualParam), newBeta.at(actualParam), newConstForeground.at(actualVideo), newConstBackground.at(actualVideo)); //const graphCut


			//ChangeDetection
			processor->setInput(imgString.at(actualVideo));
			std::cout << "output path: " << path << std::endl;
			processor->setOutput(path, ".png", 6, 1);

			// Declare a window to display the video
			//processor->displayInput("Current Frame");
			//processor->displayOutput("Output Frame");

			// Play the video at the original frame rate
			//processor->setDelay((int)floor(double(1000.0)/double(processor->getFrameRate())));
			processor->setDelay(-1);

			// Set the frame processor callback function
			processor->setFrameProcessor(tracker);

			//processor->stopAtFrameNo(5);
			// Start the process*/
			processor->run();

			
			path.clear();
			delete tracker;
			delete processor;
		}
	}
}


void MainLoop::defineVideoParamters()
{
	numberOfParams = 16;
	numberOfVideos = 32; 
	
	//define the basic result directory, the algorithm creates the necessary subfolders for the changedetection database
	std::string result = "E:/PBAS/result_hp_version/";
	// create output directory
	CreateDirectoryA(result.c_str(), NULL);
	std::string temp;
	
	//define which videos should be processed [0-32]
	int startVideo = 0;
	int endVideo = 32;
	
	
	//define different parameters for the pbas algorithm
	for(int params = 0 ; params < 1 ; ++params)
	{
		if(params == 0)
		{
			//PBAS Const
			newN.push_back(35);
			newR.push_back(18);
			newRaute.push_back(2);
			newTemporal.push_back(18);
			newAlpha.push_back(10.0);
			newBeta.push_back(1.0);
			//r-Thresh
			rThreshScale.push_back(5.0);
			rIncDecFac.push_back(0.05);
			//T-thresh
			decreasingRateScale.push_back(0.05);
			increasingRateScale.push_back(1.0);
			lowerTimeUpdateRateBoundary.push_back(2);
			upperTimeUpdateRateBoundary.push_back(200);
			
			//automatically creating a string for the destination folder based on the chosen pbas parameters
			temp.clear();
			temp += result;
			temp +=	createString();
			temp += "";
			outputBaseline.push_back(temp);
		}
		


		for(int l = startVideo; l < endVideo; l++)
		{			

			if(l == 0)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/baseline/highway/input/in", 1703));
				setOutputVideo.push_back("highway");
				baselineString.push_back("baseline");
				setOutputPath.push_back("");


				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
			}
			else if(l == 1)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/baseline/office/input/in", 2051));
				setOutputVideo.push_back("office");
				baselineString.push_back("baseline");
				setOutputPath.push_back("");				


				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
			}
			else if(l == 2)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/baseline/pedestrians/input/in", 1100));
				setOutputVideo.push_back("pedestrians");
				baselineString.push_back("baseline");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);

				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
			}
			else if(l == 3)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/baseline/PETS2006/input/in", 1201));
				setOutputVideo.push_back("PETS2006");
				baselineString.push_back("baseline");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
			}
			//####################################
			//cameraJitter
			//###################################
			else if(l == 4)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/cameraJitter/badminton/input/in", 1152));
				setOutputVideo.push_back("badminton");
				baselineString.push_back("cameraJitter");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 5)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/cameraJitter/boulevard/input/in", 2502));
				setOutputVideo.push_back("boulevard");
				baselineString.push_back("cameraJitter");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 6)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/cameraJitter/sidewalk/input/in", 1202));
				setOutputVideo.push_back("sidewalk");
				baselineString.push_back("cameraJitter");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 7)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/cameraJitter/traffic/input/in", 1572));
				setOutputVideo.push_back("traffic");
				baselineString.push_back("cameraJitter");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			//###################################
			//dynamic background
			//##################################
			else if(l == 8)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/dynamicBackground/boats/input/in", 8001));
				setOutputVideo.push_back("boats");
				baselineString.push_back("dynamicBackground");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 9)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/dynamicBackground/canoe/input/in", 1191));
				setOutputVideo.push_back("canoe");
				baselineString.push_back("dynamicBackground");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 10)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/dynamicBackground/fall/input/in", 4002));
				setOutputVideo.push_back("fall");
				baselineString.push_back("dynamicBackground");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 11)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/dynamicBackground/fountain01/input/in", 1186));
				setOutputVideo.push_back("fountain01");
				baselineString.push_back("dynamicBackground");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 12)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/dynamicBackground/fountain02/input/in", 1501));
				setOutputVideo.push_back("fountain02");
				baselineString.push_back("dynamicBackground");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 13)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/dynamicBackground/overpass/input/in", 3002));
				setOutputVideo.push_back("overpass");
				baselineString.push_back("dynamicBackground");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			//#########################################
			//intermittentObjectMotion
			//#########################################
			else if(l == 14)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/intermittentObjectMotion/abandonedBox/input/in", 4502));
				setOutputVideo.push_back("abandonedBox");
				baselineString.push_back("intermittentObjectMotion");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 15)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/intermittentObjectMotion/parking/input/in", 2502));
				setOutputVideo.push_back("parking");
				baselineString.push_back("intermittentObjectMotion");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 16)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/intermittentObjectMotion/sofa/input/in", 2752));
				setOutputVideo.push_back("sofa");
				baselineString.push_back("intermittentObjectMotion");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 17)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/intermittentObjectMotion/streetLight/input/in", 3202));
				setOutputVideo.push_back("streetLight");
				baselineString.push_back("intermittentObjectMotion");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 18)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/intermittentObjectMotion/tramstop/input/in", 3202));
				setOutputVideo.push_back("tramstop");
				baselineString.push_back("intermittentObjectMotion");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 19)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/intermittentObjectMotion/winterDriveway/input/in", 2502));
				setOutputVideo.push_back("winterDriveway");
				baselineString.push_back("intermittentObjectMotion");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			//#########################################################################################################
			//shadow													
			else if(l == 20)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/shadow/backdoor/input/in", 2002));
				setOutputVideo.push_back("backdoor");
				baselineString.push_back("shadow");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 21)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/shadow/bungalows/input/in", 1702));
				setOutputVideo.push_back("bungalows");
				baselineString.push_back("shadow");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 22)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/shadow/busStation/input/in", 1252));
				setOutputVideo.push_back("busStation");
				baselineString.push_back("shadow");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 23)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/shadow/copyMachine/input/in", 3402));
				setOutputVideo.push_back("copyMachine");
				baselineString.push_back("shadow");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 24)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/shadow/cubicle/input/in", 7402));
				setOutputVideo.push_back("cubicle");
				baselineString.push_back("shadow");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 25)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/shadow/peopleInShade/input/in", 1201));
				setOutputVideo.push_back("peopleInShade");
				baselineString.push_back("shadow");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			//######################################################################################
			//thermal
			//######################################################################################
			else if(l == 26)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/thermal/corridor/input/in", 5402));
				setOutputVideo.push_back("corridor");
				baselineString.push_back("thermal");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 27)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/thermal/diningRoom/input/in", 3702));
				setOutputVideo.push_back("diningRoom");
				baselineString.push_back("thermal");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 28)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/thermal/lakeSide/input/in", 6502));
				setOutputVideo.push_back("lakeSide");
				baselineString.push_back("thermal");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 29)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/thermal/library/input/in", 4902));
				setOutputVideo.push_back("library");
				baselineString.push_back("thermal");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
			else if(l == 30)
			{
				//outputBaseline = "E:/PBAS/dataset/baseline/highway/";
				
				imgString.push_back(setImageString("E:/PBAS/dataset/thermal/park/input/in", 602));
				setOutputVideo.push_back("park");
				baselineString.push_back("thermal");
				setOutputPath.push_back("");

				resParam.push_back(1.0);

				//graphCuts Const
				useGraphCuts.push_back(false);
				newConstForeground.push_back(2.0);
				newConstBackground.push_back(1.0);
				
			}
		}
	}

}


std::string MainLoop::createString()
// creates string for output folder depending on set parameter
{
	std::string outputStringT;
	std::stringstream temp;
	
	//create string for outputfolder	
	temp << (int)newN.at(newN.size()-1);
	
	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp <<  (int)newR.at(newR.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp <<  (int)newRaute.at(newRaute.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp <<  (int)newTemporal.at(newTemporal.size()-1);
	
	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp <<  (float)rThreshScale.at(rThreshScale.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp <<  (float)rIncDecFac.at(rIncDecFac.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp <<  (float)decreasingRateScale.at(decreasingRateScale.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp << (float)increasingRateScale.at(increasingRateScale.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp << (int)lowerTimeUpdateRateBoundary.at(lowerTimeUpdateRateBoundary.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp << (int)upperTimeUpdateRateBoundary.at(upperTimeUpdateRateBoundary.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp << (int)newAlpha.at(newAlpha.size()-1);

	outputStringT += temp.str();
	temp.str("");
	outputStringT += "_";
	temp << (int)newBeta.at(newBeta.size()-1);

	outputStringT += temp.str();
	temp.str("");
	return outputStringT;
}

std::vector<std::string> MainLoop::setImageString(std::string base, int totalNumberOfImages)
{
	std::vector<std::string> tempString;
	std::string temp, ext = ".jpg";

	for(int i = 1; i < totalNumberOfImages; ++i)
	{
		std::stringstream nr;
		temp = base;

		nr << std::setw(6) << std::setfill('0') << i << std::setfill(' ');
		temp += nr.str() + ext;
		tempString.push_back(temp);
		temp.clear();
	}
	return tempString;
}