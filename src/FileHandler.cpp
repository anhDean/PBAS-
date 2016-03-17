#include "FileHandler.h"

FileHandler::FileHandler() : m_showProcessing(false), m_inputSuffix(".jpg"), m_inputPrefix("in"), m_outputSuffix(".png"), m_outputPrefix("bin")
{
}

FileHandler::FileHandler(const std::string& dataRt, const std::string& resultRt, int rt_depth, bool setupDirectories) : m_input_root(dataRt), m_output_root(resultRt), m_root_depth(rt_depth), m_inputFolders(getInputFoldernames()), m_outputFolders(m_inputFolders),
																													m_showProcessing(false), m_inputSuffix(".jpg"), m_inputPrefix("in"), m_outputSuffix(".png"), m_outputPrefix("bin")
																								
{
	setOutputFolders();
	if(setupDirectories)
		createDirectories();
}

void FileHandler::setOutputFolders()
{
	for (int i = 0; i < m_inputFolders.size(); ++i)
	{
		replaceSubstring(m_outputFolders[i], m_input_root, m_output_root);
		replaceSubstring(m_outputFolders[i], "\\input", "");
	}

}
void FileHandler::createDirectories() const
{
	// WARNING: DELETES ALL CONTENT OF ROOT OUTPUT DIRECTORY
	fs::remove_all(m_output_root);
	Sleep(30); // need short delay to remove all contents before creating new directories
	bool(*cr)(const fs::path&) = fs::create_directories; // function pointer to create directories function
	std::for_each(m_outputFolders.begin(), m_outputFolders.end(), cr);

}

std::vector<std::string> FileHandler::getInputFoldernames() const
{
	std::vector<std::string> folderNames;
	folderNames.push_back(m_input_root);

	for (int i = 0; i < m_root_depth; ++i)
	{
		folderNames = getNextPathname(folderNames);
	}

	for (int j = 0; j < folderNames.size(); ++j)
	{
		folderNames[j] += "\\input";
	}
	return folderNames;

}
std::vector<std::string> FileHandler::getNextPathname(std::vector<std::string> in) const
{
	std::vector<std::string> out;

	for (int i = 0; i < in.size(); ++i)
	{
		for (fs::directory_iterator it(in.at(i)), end; it != end; ++it)
		{
			if (fs::is_directory(it->path()))
			{
				out.push_back(it->path().string());
			}

		}
	}
	return out;
}
bool FileHandler::replaceSubstring(std::string& src, const std::string& old_str, const std::string& new_str)
{
	size_t start_pos = src.find(old_str);
	if (start_pos == std::string::npos)
		return false;
	src.replace(start_pos, old_str.length(), new_str);
	return true;

}



const size_t& FileHandler::getFolderCount() const
{
	return m_inputFolders.size();
}
const std::vector<std::string>& FileHandler::getInputFolders() const
{
	return m_inputFolders;
}
const std::vector<std::string>& FileHandler::getOutputFolders() const
{
	return m_outputFolders;
}



bool FileHandler::setInputSuffix(const std::string& newVal)
{
	if (newVal.find(".") != std::string::npos)
	{
		m_inputSuffix = newVal;
		return true;
	}
	else
		return false;
}
void FileHandler::setInputPrefix(const std::string& newVal)
{
	m_inputPrefix = newVal;
}

bool FileHandler::setOutputSuffix(const std::string& newVal)
{
	if (newVal.find(".") != std::string::npos)
	{
		m_outputSuffix = newVal;
		return true;
	}
	else
		return false;
}

void FileHandler::setOutputPrefix(const std::string& newVal)
{
	m_outputPrefix = newVal;
}


std::string FileHandler::getOutputFilename(const std::string& inputFilename)
{
	std::string outputFilename(inputFilename);
	replaceSubstring(outputFilename, m_inputPrefix, m_outputPrefix);
	replaceSubstring(outputFilename, m_inputSuffix, m_outputSuffix);
	return outputFilename;
}

void FileHandler::setDisplayFlag(bool flag) 
{
	m_showProcessing = flag;
}
const bool& FileHandler::getDisplayFlag()const
{
	return m_showProcessing;
}

bool FileHandler::createDirectory(const std::string dir) const
{
	// parent must exist, if not empty function does nothing
	fs::create_directory(dir);
	return true;
}




bool FileHandler::process_folder(std::string inputFolder, std::string outputFolder, FrameProcessor* processor)
{
	// get file name
	std::string tmp_inputFile, tmp_outputFile, windowName = outputFolder;
	cv::Mat input, output;
	cv::Mat tmp_bgDynamics, m, tmp_bgNoise, tmp_gradMagnMap;

	double minVal, maxVal, epsilon = 1e-6;

	for (fs::directory_iterator it(inputFolder), end; it != end; ++it)
	{

		if (fs::is_regular_file(it->path()) && (it->path().filename().string().find(m_inputSuffix) != std::string::npos))
		{
			tmp_inputFile = it->path().filename().string();
			tmp_outputFile = getOutputFilename(tmp_inputFile);
		}
		input = cv::imread(it->path().string(), CV_LOAD_IMAGE_COLOR);
		output.create(input.size(), CV_8U);
		processor->process(input, output);

		if (m_showProcessing)
		{			
			tmp_bgDynamics =  *(processor->getBackgroundDynamics()); // TODO: find a way to normalize matrix such that it can be visualized
			tmp_bgNoise = processor->getNoiseMap();
			tmp_gradMagnMap = processor->getGradMagnMap();

			cv::imshow(windowName, output);
			cv::imshow(windowName + " Input", input);
			cv::imshow("Background dynamics", tmp_bgDynamics);
			cv::imshow("Background Noise", tmp_bgNoise);
			cv::imshow("Gradient magnitude map", tmp_gradMagnMap);
			if (cv::waitKey(1) == 27)  // 27 = ESC
				break;
		}
		cv::imwrite(outputFolder + "\\" + tmp_outputFile, output);
	}
	processor->resetProcessor();
	input.release();
	output.release();
	tmp_bgNoise.release();
	tmp_bgDynamics.release();
	cv::destroyAllWindows();
	return true;
}

bool FileHandler::process_folder(int idx, FrameProcessor* processor)
{
	std::string inputFolder = m_inputFolders[idx];
	std::string outputFolder = m_outputFolders[idx];
	process_folder(inputFolder, outputFolder, processor);
	return true;
}
