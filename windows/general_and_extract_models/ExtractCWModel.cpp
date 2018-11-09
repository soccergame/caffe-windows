#include <fstream>
#include <exception>
#include <string>
#include <stack>
#include "caffe/caffe.hpp"
#include "autoarray.h"
#include "tclap/CmdLine.h"
#include "opencv2/opencv.hpp"

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
        std::string szNetFile = "D:/project/caffe-windows/Build/x64/Debug/deploy_cw_age_gender_net.prototxt";
        std::string szWeightFile = "D:/project/caffe-windows/Build/x64/Debug/deploy_cw_age_gender_net.caffemodel";
        
        std::string image_path = "E:/工作资料/[2]图像部门/[4]图像数据/[1]人脸数据/ID_T_norm/sample_dy/sn04030481_norm1.jpg";
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
        bool need_extracted = false;
        bool need_debug = false;
        if (need_extracted) {
            // 初始化网络结构
            caffe::Caffe::set_mode(caffe::Caffe::CPU);
            // 定义网络参数
            caffe::NetParameter param;
            // 从二进制文件中读取deploy文件
            caffe::ReadProtoFromBinaryFileOrDie(strNetPath.c_str(), &param);
            // 从二进制buffer中读取deploy文件
            //caffe::ReatNetParamsFromBuffer(pDataBuf, protoTxtLen, &param);
            // 下面将这些参数以常见的deploy_net.prototxt和net.caffemodel的形式表现出来：
            // 把二进制读取的deploy文件以文本的形式写出来
            caffe::WriteProtoToTextFile(param, szNetFile);
            // 定义网络参数
            caffe::NetParameter net_param;
            // 从二进制文件中读取caffemodel，caffemodel本身也是一个二进制文件，所以直接将后缀bin
            // 改为caffemodel也没有任何问题
            caffe::ReadProtoFromBinaryFileOrDie(strWeightPath.c_str(), &net_param);
            // 从二进制buffer中读取caffemodel
            //caffe::ReatNetParamsFromBuffer(pDataBuf + protoTxtLen, modelSize, &net_param);
            // 将caffemodel保存出来
            caffe::WriteProtoToBinaryFile(net_param, szWeightFile);
        }
        else {
            cv::Mat ori_image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);

            caffe::Caffe::set_mode(caffe::Caffe::CPU);
            caffe::NetParameter param;
            caffe::ReadProtoFromTextFileOrDie(szNetFile.c_str(), &param);
            if (need_debug) {
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
            }
            param.mutable_state()->set_phase(caffe::TEST);
            caffe::Net<float> *pCaffeNet = new caffe::Net<float>(param);
            pCaffeNet->CopyTrainedLayersFromBinaryProto(szWeightFile.c_str());

            int length = ori_image.channels() * ori_image.rows * ori_image.cols;
            AutoArray<float> normRealImage(length);
            if (ori_image.channels() == 1)
            {
                for (int i = 0; i < length; ++i)
                    normRealImage[i] = static_cast<float>(ori_image.at<unsigned char>(i)) - 127.5f;
            }

            std::vector<caffe::Blob<float>*> bottom_vec;
            bottom_vec.push_back(new caffe::Blob<float>);
            bottom_vec[0]->Reshape(1, ori_image.channels(), ori_image.rows, ori_image.cols);
            bottom_vec[0]->set_cpu_data(normRealImage);

            float iter_loss;
            const vector<caffe::Blob<float>*>& result = pCaffeNet->Forward(bottom_vec, &iter_loss);

            int fea_len = result[0]->count();
            AutoArray<float> feature(fea_len);
            float max_score = -1.0f;
            float age = 0;
            for (int i = 0; i < fea_len-2; ++i)
            {
                feature[i] = result[0]->cpu_data()[i];
                if (max_score <= feature[i]) {
                    max_score = feature[i];
                    age = float(i + 1);
                }
            }
            feature[fea_len - 2] = result[0]->cpu_data()[fea_len - 2];
            feature[fea_len - 1] = result[0]->cpu_data()[fea_len - 1];
            if (feature[fea_len - 2] <= feature[fea_len - 1])
                printf("男人，年龄%0.1f岁\n", age);
            else
                printf("女人，年龄%0.1f岁\n", age);
            
        }
        
        
        //// 那么C如何读取deploy和caffemodel呢？
        //// 首先选择是CPU还是GPU
        //caffe::Caffe::set_mode(caffe::Caffe::CPU);
        //// 定义网络参数
        //caffe::NetParameter param;
        //
        //// 从二进制文件中读取deploy文件  
        //// caffe::ReadProtoFromBinaryFileOrDie(strNetPath.c_str(), &param);
        //// 如果要从普通的deploy.prototxt中读取网络结构，就需要：
        //caffe::ReadProtoFromTextFileOrDie(szNetFile.c_str(), &param);
        //// 然后表明是TEST
        //param.mutable_state()->set_phase(caffe::TEST);
        //// 接下来设置网络结构
        //caffe::Net<float> *pCaffeNet = new caffe::Net<float>(param);
        //// 初始化网络权重，这样就可以直接调用网络了
        //pCaffeNet->CopyTrainedLayersFromBinaryProto(szWeightFile.c_str());
        //// 后面可以直接调用网络
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