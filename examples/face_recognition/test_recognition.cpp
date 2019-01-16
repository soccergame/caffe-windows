#include "caffe/face_recognition/face_recognition.hpp"
#include "autoarray.h"
#include "TimeCount.h"
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h> 
#include <sys/stat.h> 
#endif
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define RECOGNITION_MODEL_FILE "libsnfr.so"

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cout << "usage:" << std::endl;
        std::cout << "feature_test <module_path> <base_dir> <file_list> <output_path> <gpuId>" 
            << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();

    std::string base_path = argv[2];
    std::string image_list = argv[3];
    int retValue = 0;
    try
    {
        int gpuId = atoi(argv[5]);

        // Initialize        
        retValue = SetFaceRecognitionLibPath(pModulePath);

        RecognitionHandle hFace;// , hAge, hSkin, hXiaci, hHappy;
        retValue |= InitFaceRecognition("libsnfr.so", gpuId, &hFace);
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue)
            throw retValue;

        int feaDim = GetFaceRecognitionSize(hFace) / sizeof(float);

        // Read images of input filepath
        std::vector<std::string> imgList;
        std::fstream fp;
        fp.open(image_list, std::fstream::in);
        if (fp.is_open()) {
            int class_num;
            fp >> class_num;
            for (int i = 0; i < class_num; ++i) {
                std::string class_name;
                fp >> class_name;
                int sample_num;
                fp >> sample_num;
                for (int j = 0; j < sample_num; ++j) {
                    std::string sample_name;
                    fp >> sample_name;
                    imgList.push_back(sample_name);
                }
            }
        }

        std::ofstream outfile(argv[4]);
        outfile << "{";
        bool flag = false;
        for (int i = 0; i < imgList.size(); i++) {
            std::string strImgName = base_path + imgList[i].c_str();
            cv::Mat oriImgData = cv::imread(strImgName, CV_LOAD_IMAGE_COLOR);
            if (oriImgData.empty())
                continue;

            cv::Mat imgData;
            cv::cvtColor(oriImgData(cv::Rect(56, 56, 112, 112)), imgData, cv::COLOR_BGR2RGB);
            AutoArray<float> feature(feaDim);
            retValue = InnerFaceRecognition(hFace, imgData.data, 1, imgData.channels(), imgData.rows,
                imgData.cols, feature.begin());
            if (0 != retValue)
                continue;

            if (flag) {
                outfile << ",";
            }
                

            outfile << "\"" << imgList[i] << "\"" << ":[";
            outfile << feature[0];
            for (int j = 1; j < feaDim; ++j)
                outfile << "," << feature[j];
            outfile << "]";

            flag = true;
        }

        UninitFaceRecognition(hFace);
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
