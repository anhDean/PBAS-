#include "FileHandler.h"



FileHandler::FileHandler(const std::string& dataRt, const std::string& resultRt, int rt_depth) : m_input_root(dataRt), m_output_root(resultRt), m_root_depth(rt_depth), m_inputFolders(getInputFoldernames()),
																								m_outputFolders(m_inputFolders), m_inputSuffix(".jpg"), m_inputPrefix("in"), m_outputSuffix(".png"), m_outputPrefix("bin")
{
	setOutputFolders();
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

	fs::remove_all(m_output_root);
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



const int& FileHandler::getFolderCount() const
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


template<typename FrameProcessorClass>
bool FileHandler::process_folder(std::string inputFolder, std::string outputFolder, const FrameProcessorClass& processor) const
{
	// get file name
	std::string tmp_inputFile, tmp_outputFile;
	cv::Mat input, result;

	for (fs::directory_iterator it(inputFolder), end; it != end; ++it)
	{

		if (fs::is_regular_file(it->path()) && (it->path().filename().string().find(m_inputSuffix) != std::string::npos))
		{
			tmp_inputFile =  it->path().filename().string();
			tmp_outputFile = getOutputFilename(tmp_inputFile);
		}
		input = cv::imread(tmp_inputFile, CV_LOAD_IMAGE_COLOR);
		result.create(input.size(), CV_8U);
		processor->process(input, result);

		cv::imwrite(outputFolder + "\\" + tmp_outputFile, result);
	}

	return true;
}

template<typename FrameProcessorClass>
bool FileHandler::process_folder(int idx, const FrameProcessorClass& processor) const
{
	// get file name
	std::string tmp_inputFile, tmp_outputFile;
	cv::Mat input, result;

	for (fs::directory_iterator it(m_inputFolders[idx]), end; it != end; ++it)
	{

		if (fs::is_regular_file(it->path()) && (it->path().filename().string().find(m_inputSuffix) != std::string::npos))
		{
			tmp_inputFile = it->path().filename().string();
			tmp_outputFile = getOutputFilename(tmp_inputFile);
		}
		input = cv::imread(tmp_inputFile, CV_LOAD_IMAGE_COLOR);
		result.create(input.size(), CV_8U);
		processor->process(input, result);

		cv::imwrite(m_outputFolders[idx] + "\\" + tmp_outputFile, result);
	}

	return true;
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