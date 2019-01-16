// THIDFaceDeepFeat.cpp : Defines the exported functions for the DLL application.
//

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "glog/logging.h"
#include <boost/algorithm/string.hpp>
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include <boost/algorithm/string.hpp>
#include "autoarray.h"
#include "caffe/face_recognition/face_recognition.hpp"
#include "AlgorithmUtils.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

#ifndef _WIN32
#define _MAX_PATH 260
#endif

namespace
{
    char g_szFaceRecognitionSDKPath[_MAX_PATH] = { 0 };
}

int __stdcall InnerFaceRecognition(RecognitionHandle handle, const unsigned char *pNormImage, int batchSize, int channels,
    int imageHeight, int imageWidth, float *pFeatures)
{
    int nRet = 0;
    try
    {
        Net<float> *pCaffeNet = reinterpret_cast<Net<float> *>(handle);
        int length = batchSize * channels * imageHeight * imageWidth;
        AutoArray<float> normRealImage(length);
        for (int i = 0; i < batchSize; ++i)
        {
            for (int j = 0; j < channels; ++j)
            {
                for (int h = 0; h < imageHeight; ++h)
                {
                    for (int w = 0; w < imageWidth; ++w)
                    {
                        int index = i * channels * imageHeight * imageWidth + j * imageHeight * imageWidth + h * imageWidth + w;
                        int ori_idx = i * channels * imageHeight * imageWidth + h * imageWidth * channels + w * channels + j;
                        normRealImage[index] = (static_cast<float>(pNormImage[ori_idx]) - 127.5) * 0.0078125;
                    }
                    
                }
            }
        }

        std::vector<caffe::Blob<float>*> bottom_vec;
        bottom_vec.push_back(new caffe::Blob<float>);
        bottom_vec[0]->Reshape(batchSize, channels, imageHeight, imageWidth);
        bottom_vec[0]->set_cpu_data(normRealImage);

        float iter_loss;
        const vector<Blob<float>*>& result = pCaffeNet->Forward(bottom_vec, &iter_loss);

        for (int i = 0; i < result[0]->count(); ++i)
        {
            pFeatures[i] = result[0]->cpu_data()[i];
        }
    }
    catch (const std::bad_alloc &)
    {
        nRet = -2;
    }
    catch (const int &errCode)
    {
        nRet = errCode;
    }
    catch (...)
    {
        nRet = -3;
    }

    return nRet;
}


int __stdcall SetFaceRecognitionLibPath(const char *szLibPath)
{
	if (szLibPath == NULL)
		return -1;

#ifdef _WIN32
	strcpy_s(g_szFaceRecognitionSDKPath, _MAX_PATH, szLibPath);
#else
    strncpy(g_szFaceRecognitionSDKPath, szLibPath, _MAX_PATH);
#endif
	
	size_t len = strlen(g_szFaceRecognitionSDKPath);
	if (len != 0)
	{
	#ifdef _WIN32
		if (g_szFaceRecognitionSDKPath[len - 1] != '\\')
			strcat_s(g_szFaceRecognitionSDKPath, "\\");
	#else
	    if (g_szFaceRecognitionSDKPath[len - 1] != '/')
	        strncat(g_szFaceRecognitionSDKPath, "/", _MAX_PATH);
	#endif
	}

	return 0;
}

int __stdcall InitFaceRecognition(const char *szResName, int gpuID,
    RecognitionHandle *pHandle)
{
	if (pHandle == NULL)
		return -1;
	
	// initialize deep face network
	*pHandle = NULL;
	std::locale::global(std::locale(""));

	int retValue = 0;

#ifndef _WIN32	
	if (strlen(g_szFaceRecognitionSDKPath) == 0)
		strncpy(g_szFaceRecognitionSDKPath, "./", _MAX_PATH);
#endif

	try
	{
		std::string strDllPath;
		strDllPath = g_szFaceRecognitionSDKPath;
        strDllPath += szResName;

        std::fstream fileModel;
        fileModel.open(strDllPath.c_str(), std::fstream::in | std::fstream::binary);
        if (false == fileModel.is_open())
          return 1;
          
        fileModel.seekg(0, std::fstream::end);
        int dataSize = int(fileModel.tellg());
        fileModel.seekg(0, std::fstream::beg);
       
		//CMyFile fileModel(strDllPath.c_str(), CMyFile::modeRead);
		//int dataSize = static_cast<int>(fileModel.GetLength());
		AutoArray<char> encryptedData(dataSize);
		//fileModel.Read(encryptedData, dataSize);
		fileModel.read(encryptedData, dataSize);
		//fileModel.Close();
		fileModel.close();

		int *pBuffer = reinterpret_cast<int *>(encryptedData.begin());
		// encrypt data by shift left		
		int numOfData = dataSize / sizeof(pBuffer[0]);
		for (int i = 0; i < numOfData; ++i)
		{
			int tempData = pBuffer[i];
			pBuffer[i] = hzx::ror(static_cast<unsigned int>(tempData), 
                hzx::g_shiftBits);
		}

        const int modelnumber = pBuffer[0];
        std::vector<int> protoTxtLen, modelSize;
        for (int i = 0; i < modelnumber; ++i)
        {
            protoTxtLen.push_back(pBuffer[2 * i + 1]);
            modelSize.push_back(pBuffer[2 * i + 2]);
        }
        unsigned char *pDataBuf = reinterpret_cast<unsigned char *>(encryptedData.begin()) + sizeof(int) * (2 * modelnumber + 1);

		FLAGS_minloglevel = 2;	// INFO(=0)<WARNING(=1)<ERROR(=2)<FATAL(=3)

		// initialize network structure
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        if (gpuID < 0)
            Caffe::set_mode(Caffe::CPU);
        else {
            Caffe::set_mode(Caffe::GPU);
            Caffe::SetDevice(gpuID);
        }
#endif
		caffe::NetParameter net_param;
        caffe::NetParameter weight_param;

		retValue = caffe::ReatNetParamsFromBuffer(
            pDataBuf, protoTxtLen[0], &net_param);
        CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &net_param));
        retValue = caffe::ReatNetParamsFromBuffer(
            pDataBuf + protoTxtLen[0], modelSize[0], &weight_param);
        CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &weight_param));

        net_param.mutable_state()->set_phase(caffe::TEST);
		Net<float> *pCaffeNet = new Net<float>(net_param);

		// initialize network parameters		
		pCaffeNet->CopyTrainedLayersFrom(weight_param);
		*pHandle = reinterpret_cast<RecognitionHandle>(pCaffeNet);
	}
	catch (const std::bad_alloc &)
	{
		retValue = -2;
	}
	catch (const int &errCode)
	{
		retValue = errCode;
	}
	catch (...)
	{
		retValue = -3;
	}

	return retValue;
}

int __stdcall UninitFaceRecognition(RecognitionHandle handle)
{
	Net<float> *pCaffeNet = reinterpret_cast<Net<float> *>(handle);
	delete pCaffeNet;

	return 0;
}

int __stdcall GetFaceRecognitionSize(RecognitionHandle handle)
{
	Net<float> *pCaffeNet = reinterpret_cast<Net<float> *>(handle);
	const vector<Blob<float>*>& result = pCaffeNet->output_blobs();
	int len = result[0]->count() * sizeof(float);
	return len;
}
