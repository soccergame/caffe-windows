#include <fstream>
#include <exception>
#include <string>
#include <stack>
#include "caffe/caffe.hpp"
#include "autoarray.h"
#include "tclap\CmdLine.h"

namespace brc_sn
{
	// for imagenet, normalize image size is 224 * 224
	// const float g_scale = 224.0f / 128.0f;
	const float g_scale = 1.0f;// 224.0f / 128.0f;	// normal image resized to 128 * 128 , 
	const int g_shiftBits = 11;
	// rotate shift right by moves bits
	template<typename T> T ror(T x, unsigned int moves)
	{
		return (x >> moves) | (x << (sizeof(T) * 8 - moves));
	}

	// rotate shift left by moves bits
	template<typename T> T rol(T x, unsigned int moves)
	{
		return (x << moves) | (x >> (sizeof(T) * 8 - moves));
	}

	// const int normSize = static_cast<int>(144 * g_scale);		// normalize image size
	// const int eyeCenterY = static_cast<int>(48 * g_scale);		// eye center y coordinate in normalized image
	// const int distEyeCMouthC = static_cast<int>(48 * g_scale);	// distance between eye center to mouth center in normalized image
	int g_meanVal[3] = { 104, 117, 123 };  
}

int main(int argc, char *argv[])
{
    int retValue = 0;
    std::locale::global(std::locale(""));

    try
    {
        //TCLAP::CmdLine cmd(TCLAP_TEXT("Generate or extract caffe models.\n")  \
        //    TCLAP_TEXT("Copyright: Beijing SN Corp. Ltd.")	\
        //    TCLAP_TEXT("Author: He Zhixiang")	\
        //    TCLAP_TEXT("Data: Nov. 5, 2018"), TCLAP_TEXT(' '), TCLAP_TEXT("2.0"));

        //// Deployed prototxt file path
        //TCLAP::ValueArg<std::string> deployFile(TCLAP_TEXT(""), TCLAP_TEXT("DeployFile"),
        //    TCLAP_TEXT("class prefix to be added to each class name"),
        //    false, TCLAP_TEXT(""), TCLAP_TEXT("class prefix to be added to each class name"), cmd);
        //// Caffemodel file path
        //TCLAP::ValueArg<std::string> weightFile(TCLAP_TEXT(""), TCLAP_TEXT("WeightFile"),
        //    TCLAP_TEXT("Base path of all image sets"),
        //    true, TCLAP_TEXT(""),
        //    TCLAP_TEXT("Base path of all image sets needs to be removed in names"), cmd);

        std::string strNetPath = "D:/project/caffe-windows/Build/x64/Debug/age.bin";
        std::string strWeightPath = "D:/project/caffe-windows/Build/x64/Debug/ageWeight.bin";
        std::string szNetFile = "D:/project/caffe-windows/Build/x64/Debug/deploy_cw_net.prototxt";
        std::string szWeightFile = "D:/project/caffe-windows/Build/x64/Debug/deploy_cw_net.caffemodel";
        
  //      AutoArray<char> encryptedData;
  //      int widthByte = 0;
  //      std::fstream fp;
	 //   fp.open(strNetPath, std::fstream::in | std::fstream::binary);
  //      if (fp.is_open())
  //      {
  //      	fp.seekg(0, std::fstream::end);
	 //       int dataLen = int(fp.tellg());
  //    	    fp.seekg(0, std::fstream::beg);
  //    	
  //          widthByte = ((dataLen + 7) / 8) * 8;
  //    	    encryptedData.resize(widthByte);
  //    	    fp.read(encryptedData.begin(), widthByte);
  //      }
  //      
  //      fp.close();

  //      if (widthByte <= 0)
  //          throw 1;
  //      
  //      int *pBuffer = reinterpret_cast<int *>(encryptedData.begin());
	 //   // encrypt data by shift left		
	 //   int numOfData = widthByte / sizeof(pBuffer[0]);
	 //   for (int i = 0; i < numOfData; ++i)
	 //   {
	 //	    int tempData = pBuffer[i];
	 //	    pBuffer[i] = ror(static_cast<unsigned int>(tempData), g_shiftBits);
	 //   }
 
	 //const int protoTxtLen = pBuffer[0];
	 //const int modelSize = pBuffer[1];
	 //const unsigned char *pDataBuf = encryptedData + sizeof(int) * 2;
        if (false) {
            // ��ʼ������ṹ
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
            // �����������
            caffe::NetParameter param;
            // �Ӷ������ļ��ж�ȡdeploy�ļ�
            caffe::ReadProtoFromBinaryFileOrDie(strNetPath.c_str(), &param);
            // �Ӷ�����buffer�ж�ȡdeploy�ļ�
            //caffe::ReatNetParamsFromBuffer(pDataBuf, protoTxtLen, &param);
            // ���潫��Щ�����Գ�����deploy_net.prototxt��net.caffemodel����ʽ���ֳ�����
            // �Ѷ����ƶ�ȡ��deploy�ļ����ı�����ʽд����
            caffe::WriteProtoToTextFile(param, szNetFile);
            // �����������
            caffe::NetParameter net_param;
            // �Ӷ������ļ��ж�ȡcaffemodel��caffemodel����Ҳ��һ���������ļ�������ֱ�ӽ���׺bin
            // ��ΪcaffemodelҲû���κ�����
            caffe::ReadProtoFromBinaryFileOrDie(strWeightPath.c_str(), &net_param);
            // �Ӷ�����buffer�ж�ȡcaffemodel
            //caffe::ReatNetParamsFromBuffer(pDataBuf + protoTxtLen, modelSize, &net_param);
            // ��caffemodel�������
            caffe::WriteProtoToBinaryFile(net_param, szWeightFile);
        }
        else {
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
            caffe::NetParameter param;
            caffe::ReadProtoFromTextFileOrDie(szNetFile.c_str(), &param);
            std::cout << param.DebugString() << std::endl;
            caffe::NetParameter net_param;
            caffe::ReadProtoFromBinaryFileOrDie(strWeightPath.c_str(), &net_param);
            int num_net_layers = param.layer_size();
            int num_weight_layers = net_param.layer_size();

            std::cout << "net: " << num_net_layers << "; weight: " << num_weight_layers << std::endl;

            for (int j = 0; j < num_weight_layers; ++j) {
                const caffe::LayerParameter& source_layer = net_param.layer(j);
                std::cout << "layer type: " << source_layer.type() << std::endl;
                if (source_layer.type() == "Eltwise")
                {
                    std::cout << source_layer.DebugString() << std::endl;
                }
            }

            param.mutable_state()->set_phase(caffe::TEST);
            caffe::Net<float> *pCaffeNet = new caffe::Net<float>(param);
            pCaffeNet->CopyTrainedLayersFromBinaryProto(szWeightFile.c_str());
        }
        
        
        //// ��ôC��ζ�ȡdeploy��caffemodel�أ�
        //// ����ѡ����CPU����GPU
        //caffe::Caffe::set_mode(caffe::Caffe::CPU);
        //// �����������
        //caffe::NetParameter param;
        //
        //// �Ӷ������ļ��ж�ȡdeploy�ļ�  
        //// caffe::ReadProtoFromBinaryFileOrDie(strNetPath.c_str(), &param);
        //// ���Ҫ����ͨ��deploy.prototxt�ж�ȡ����ṹ������Ҫ��
        //caffe::ReadProtoFromTextFileOrDie(szNetFile.c_str(), &param);
        //// Ȼ�������TEST
        //param.mutable_state()->set_phase(caffe::TEST);
        //// ��������������ṹ
        //caffe::Net<float> *pCaffeNet = new caffe::Net<float>(param);
        //// ��ʼ������Ȩ�أ������Ϳ���ֱ�ӵ���������
        //pCaffeNet->CopyTrainedLayersFromBinaryProto(szWeightFile.c_str());
        //// �������ֱ�ӵ�������
    }
    catch (int errcode)
    {
        retValue = errcode;
    }
    catch (const std::bad_alloc &)
    {
        retValue = -1;
    }
    catch (...)
    {
        retValue = -2;
    }

    return retValue;
}