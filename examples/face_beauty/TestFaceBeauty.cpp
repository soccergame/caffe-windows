#include "caffe/face_beauty/face_beauty.h"
#include "caffe/face_detection/face_detection.hpp"
#include "caffe/common/autoarray.h"
#include "caffe/common/NormFaceImage.h"
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef _WIN32
#include<unistd.h> 
#include <dirent.h>
#endif

#define MTCNN_MODEL_FILE "/libsnfd.so"

using namespace FaceDetection;

int LoadModel(const char* pModulePath, MTCNN &fd_mtcnn)
{
    std::string strDllPath;
    strDllPath = pModulePath;
    strDllPath += MTCNN_MODEL_FILE;
    //mtcnn model load
    int ret = fd_mtcnn.init(strDllPath);

    return ret;
}

int savelandmark2file(const char *inputfilename, char *outputfilename, MTCNN fd_mtcnn)
{
    cv::Mat testImg = cv::imread(inputfilename);
    if (testImg.empty())
        return -1;
    cout << "before detect" << endl;
    //double t = (double)cv::getTickCount();
    vector<MTCNN::BoundingBox> res;
    res = fd_mtcnn.Detect(testImg, MTCNN::BGR, MTCNN::ORIENT_UP, 80, 0.6, 0.7, 0.7);
    //t = ((double)cv::getTickCount()-t) / cv::getTickFrequency();
    cout << "detected face NUM : " << res.size() << endl;
    //cout << "face detection:" << t*1000 << "ms" << endl;
    cout << "after detect" << endl;
    int maxRect_id = 0;
    float distance_max = -1e8;

    if (res.size() <1)
        return -1;

    for (size_t i = 0; i < res.size(); ++i)
    {
        // Calculate maximum faces
        float tmp_dis = (res[i].points_x[0] - res[i].points_x[4]) * (res[i].points_x[0] - res[i].points_x[4]) +
            (res[i].points_y[0] - res[i].points_y[4]) * (res[i].points_y[0] - res[i].points_y[4]);
        tmp_dis += (res[i].points_x[1] - res[i].points_x[3]) * (res[i].points_x[1] - res[i].points_x[3]) +
            (res[i].points_y[1] - res[i].points_y[3]) * (res[i].points_y[1] - res[i].points_y[3]);

        if (tmp_dis > distance_max)
        {
            maxRect_id = i;
            distance_max = tmp_dis;
        }
    }
    FILE *pFile = fopen(outputfilename, "w");
    if (NULL == pFile)
        return -1;
    int num = 5;
    fprintf(pFile, "%d\n", num);
    for (int i = 0; i<num; i++)
    {
        //printf("%f %f\n",res[maxRect_id].points_x[i],res[maxRect_id].points_y[i]);
        fprintf(pFile, "%f %f\n", res[maxRect_id].points_x[i], res[maxRect_id].points_y[i]);

    }
    fprintf(pFile, "%f %f %f %f\n", res[maxRect_id].x1, res[maxRect_id].y1, res[maxRect_id].x2, res[maxRect_id].y2);
    fclose(pFile);

    return 0;

    //cv::Mat alignedImg;
    //bool bAlign = fd_mtcnn.alignFace(testImg, alignedImg, res[maxRect_id]);
    //if(bAlign)
    //    imwrite("align.jpg",alignedImg);

    //cv::imshow("test", testImg);
    //cv::waitKey();
}

int save_max_rect_face(string inputfilename, MTCNN fd_mtcnn,
    MTCNN::BoundingBox &face_box)
{
    cv::Mat testImg = cv::imread(inputfilename, CV_LOAD_IMAGE_COLOR);
    if (testImg.empty())
        return -1;
    //cout << "before detect" << endl;
    //double t = (double)cv::getTickCount();
    vector<MTCNN::BoundingBox> res;
    res = fd_mtcnn.Detect(testImg, MTCNN::BGR, MTCNN::ORIENT_UP, 80, 0.6, 0.7, 0.7);
    //t = ((double)cv::getTickCount()-t) / cv::getTickFrequency();
    //cout << "detected face NUM : " << res.size() << endl;
    //cout << "face detection:" << t*1000 << "ms" << endl;
    //cout << "after detect" << endl;
    int maxRect_id = 0;
    float distance_max = -1e8;

    if (res.size() <1)
        return -1;

    for (size_t i = 0; i < res.size(); ++i)
    {
        // Calculate maximum faces
        float tmp_dis = (res[i].points_x[0] - res[i].points_x[4])*(res[i].points_x[0] - res[i].points_x[4]) +
            (res[i].points_y[0] - res[i].points_y[4])*(res[i].points_y[0] - res[i].points_y[4]);
        tmp_dis += (res[i].points_x[1] - res[i].points_x[3])*(res[i].points_x[1] - res[i].points_x[3]) +
            (res[i].points_y[1] - res[i].points_y[3])*(res[i].points_y[1] - res[i].points_y[3]);

        if (tmp_dis > distance_max)
        {
            maxRect_id = i;
            distance_max = tmp_dis;
        }
    }
    face_box = res[maxRect_id];

    return 0;
}

#ifndef _WIN32
void readFileList(const char* basePath, vector<string>& imgFiles)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir = opendir(basePath)) == NULL)
    {
        return;
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 ||
            strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8)//file 
        {
            int len = strlen(ptr->d_name);
            // jpg, jpeg, png, bmp
            if ((ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'p' && ptr->d_name[len - 3] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'e' && ptr->d_name[len - 3] == 'p' && ptr->d_name[len - 4] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'n' && ptr->d_name[len - 3] == 'p') || (ptr->d_name[len - 1] == 'p' && ptr->d_name[len - 2] == 'm' && ptr->d_name[len - 3] == 'b'))
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if (ptr->d_type == 10)/// link file
        {
            int len = strlen(ptr->d_name);
            if ((ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'p' && ptr->d_name[len - 3] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'e' && ptr->d_name[len - 3] == 'p' && ptr->d_name[len - 4] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'n' && ptr->d_name[len - 3] == 'p') || (ptr->d_name[len - 1] == 'p' && ptr->d_name[len - 2] == 'm' && ptr->d_name[len - 3] == 'b'))
            {
                memset(base, '\0', sizeof(base));
                strcpy(base, basePath);
                strcat(base, "/");
                strcat(base, ptr->d_name);
                imgFiles.push_back(base);
            }
        }
        else if (ptr->d_type == 4)//dir
        {
            memset(base, '\0', sizeof(base));
            strcpy(base, basePath);
            strcat(base, "/");
            strcat(base, ptr->d_name);
            readFileList(base, imgFiles);
        }
    }
    closedir(dir);
}
#endif

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "usage:" << std::endl;
        std::cout << "mtcnn_test <module_path> <filename> <gpuId>" << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();
        
    std::string strImgName = argv[2];
    int retValue = 0;
    float maxR = 0;
    int label = 0;
    
    try
    {
        int gpuId = atoi(argv[3]);

        // Initialize        
        retValue = SetDeepFeatLibPath(pModulePath);
        
        BeautyHandle hFace, hAge, hSkin, hXiaci, hHappy;
        retValue |= InitDeepFeat("libsnfb.so", gpuId, &hFace);
        //retValue |= InitDeepFeat("NNModel.dat", gpuId, &hAge);
        if (0 != retValue)
            throw retValue;

        MTCNN::BoundingBox face_box;

        MTCNN fd_mtcnn(gpuId);
        retValue = LoadModel(pModulePath, fd_mtcnn);
        if (0 != retValue) {
            UninitDeepFeat(hFace);
            //UninitDeepFeat(hAge);
            throw retValue;
        }  

        float NormPoints[10] = {
            95.0f, 100.0f,
            95.0f, 100.0f,
            150.0f, 150.0f,
            110.0f, 200.0f,
            190.0f, 200.0f,
        };
        float Weights[10] = {
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
        };
        ALGORITHMUTILS::CNormImageSimilarity affineNorm;
        affineNorm.Initialize(256, 256, 1.0, 300, NormPoints, Weights, 5);
        
        // Read Image
        //cv::Mat garyImgData = cv::imread(strImgName, CV_LOAD_IMAGE_GRAYSCALE);
#ifdef _WIN32
        cv::Mat oriImgData = cv::imread(strImgName, CV_LOAD_IMAGE_COLOR);
        // Face detection
        retValue = save_max_rect_face(strImgName, fd_mtcnn, face_box);
        if (0 != retValue)
            throw retValue;

        float feaPoints[10];
        for (int j = 0; j < 5; ++j) {
            feaPoints[2 * j] = face_box.points_x[j];
            feaPoints[2 * j + 1] = face_box.points_y[j];
        }
        // Choose eye corner and lip corner, suppose1cdhmopruw
        // The normalization method should be considered by some days
        // Use old normalization method
        std::vector<cv::Mat> mv;
        cv::split(oriImgData, mv);

        AutoArray<unsigned char> pNormImage5Pt(
            300 * 300 * oriImgData.channels());

        for (int j = 0; j < oriImgData.channels(); ++j) {
            int retValue = affineNorm.NormImage(mv[j].data, //pImage.begin() + j * testImg.rows * testImg.cols,
                oriImgData.cols, oriImgData.rows, feaPoints, 5,
                pNormImage5Pt + j * 256 * 256);
            if (retValue != 0)
                continue;

            //mv[j] = cv::Mat(300, 300, CV_8UC1, pNormImage5Pt + j * 300 * 300);
        }

        /*cv::Mat face_img;
        cv::merge(mv, face_img);*/

        /*nRetCode = NormalizeFace(oriImgData.data, oriImgData.cols, oriImgData.rows, oriImgData.channels(), FeaPoints, NormPoint,
            weight, 88, 300, 300, pDstImage);
        if (ERR_NONE != nRetCode)
        {
            std::cout << "Failed to normalize faces!" << std::endl;
            throw nRetCode;
        }*/
        
        //smart_ptr<unsigned char> pNormFace(300 * 300 * 3);
        //for (int p = 0; p < 300 * 300 * 3; ++p)
        //    pNormFace[p] = (unsigned char)(int(pDstImage[p]));

        //AutoArray<unsigned char> pCropNormFace(256 * 256 * 3);
        //AutoArray<unsigned char> pCropGrayNormFace(256 * 256);

        ////cv::Mat NormFaceImage(300, 300, CV_8UC3, pNormFace);
        //cv::Rect roi(22, 22, 256, 256);
        //cv::Mat CropImage = face_img(roi);
        //cv::Mat GrayCropImage;
        //cv::cvtColor(CropImage, GrayCropImage, cv::COLOR_BGR2GRAY);

        /*for (int h = 0; h < CropImage.rows; ++h) {
            const uchar* ptr = CropImage.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < CropImage.cols; ++w) {
                for (int c = 0; c < CropImage.channels(); ++c) {
                    int datum_index = (c * CropImage.rows + h) * CropImage.cols + w;
                    pCropNormFace[datum_index] = static_cast<char>(ptr[img_index++]);
                }
            }
        }*/

        /*for (int h = 0; h < GrayCropImage.rows; ++h) {
            const uchar* ptr = GrayCropImage.ptr<uchar>(h);
            int img_index = 0;
            for (int w = 0; w < GrayCropImage.cols; ++w) {
                for (int c = 0; c < GrayCropImage.channels(); ++c) {
                    int datum_index = (c * GrayCropImage.rows + h) * GrayCropImage.cols + w;
                    pCropGrayNormFace[datum_index] = static_cast<char>(ptr[img_index++]);
                }
            }
        }*/
        
        // 1�������
        int featDim = GetDeepFeatSize(hFace) / 4;
        AutoArray<float> pFeatures(featDim);
        retValue = InnerDeepFeat(hFace, pNormImage5Pt, 1, 3, 256, 256, pFeatures);

        float score = pFeatures[0] * 1.11f;
        if (score > 100.0f)
            score = 100.0f;
            
        std::cout << "Total score: " << score << std::endl;
#else
        char path_list[512] = { 0 };
        strcpy(path_list, strImgName.c_str());

        vector<string> imgList;
        readFileList(path_list, imgList);

        for (int l = 0; l < imgList.size(); l++)
        {
            retValue = save_max_rect_face(imgList[l], fd_mtcnn, face_box);
            if (0 != retValue)
                continue;

            cv::Mat oriImgData = cv::imread(imgList[l], CV_LOAD_IMAGE_COLOR);

            float feaPoints[10];
            for (int j = 0; j < 5; ++j) {
                feaPoints[2 * j] = face_box.points_x[j];
                feaPoints[2 * j + 1] = face_box.points_y[j];
            }

            std::vector<cv::Mat> mv;
            cv::split(oriImgData, mv);

            AutoArray<unsigned char> pNormImage5Pt(
                300 * 300 * oriImgData.channels());

            for (int j = 0; j < oriImgData.channels(); ++j) {
                int retValue = affineNorm.NormImage(mv[j].data, //pImage.begin() + j * testImg.rows * testImg.cols,
                    oriImgData.cols, oriImgData.rows, feaPoints, 5,
                    pNormImage5Pt + j * 256 * 256);
                if (retValue != 0)
                    continue;
            }

            int featDim = GetDeepFeatSize(hFace) / 4;
            AutoArray<float> pFeatures(featDim);
            retValue = InnerDeepFeat(hFace, pNormImage5Pt, 1, 3, 256, 256, pFeatures);

            float score = pFeatures[0] * 1.11f;
            if (score > 100.0f)
                score = 100.0f;

            std::cout << imgList[l] << " score: " << score << std::endl;
        }
#endif
        
        //// 2��覴�
        //featDim = GetDeepFeatSize(hXiaci) / 4;
        //pFeatures.reset(featDim);
        //nRetCode = InnerDeepFeat(hXiaci, pCropNormFace, 1, 3, 256, 256, pFeatures);

        //float maxR = -10000.0f;
        //int label = 15;
        //for (int j = 0; j < featDim; ++j)
        //{
        //    //std::cout << pFeatures[j] << " ";
        //    if (maxR < pFeatures[j])
        //    {
        //        maxR = pFeatures[j];
        //        label = j;
        //    }
        //}

        //if (0 == label)
        //    std::cout << "The flaws' number: " << "none!" << std::endl;
        //else if (1 == label)
        //    std::cout << "The flaws' number: " << "a little!" << std::endl;
        //else if (2 == label)
        //    std::cout << "The flaws' number: " << "small!" << std::endl;
        //else if (3 == label)
        //    std::cout << "The flaws' number: " << "a lot!" << std::endl;
        //else if (4 == label)
        //    std::cout << "The flaws' number: " << "very much!" << std::endl;

        //// 3������
        //featDim = GetDeepFeatSize(hHappy) / 4;
        //pFeatures.reset(featDim);
        //nRetCode = InnerDeepFeat(hHappy, pCropNormFace, 1, 3, 256, 256, pFeatures);

        //maxR = -10000.0f;
        //label = 15;
        //for (int j = 0; j < featDim; ++j)
        //{
        //    //std::cout << pFeatures[j] << " ";
        //    if (maxR < pFeatures[j])
        //    {
        //        maxR = pFeatures[j];
        //        label = j;
        //    }
        //}
        //// std::cout << std::endl;

        //if (0 == label)
        //    std::cout << "Angry!" << std::endl;
        //else if (1 == label)
        //    std::cout << "Unhappy!" << std::endl;
        //else if (2 == label)
        //    std::cout << "normal!" << std::endl;
        //else if (3 == label)
        //    std::cout << "happy!" << std::endl;
        //else if (4 == label)
        //    std::cout << "smile!" << std::endl;

        // 4������
        //featDim = GetDeepFeatSize(hAge) / 4;
        //pFeatures.resize(featDim);
        //retValue = InnerDeepFeat(hAge, pNormImage5Pt, 1, 3, 256, 256, pFeatures);
    
        //maxR = -10000.0f;
        //label = 15;
        //for (int j = 0; j < featDim; ++j)
        //{
        //    //std::cout << pFeatures[j] << " ";
        //    if (maxR < pFeatures[j])
        //    {
        //        maxR = pFeatures[j];
        //        label = j;
        //    }
        //}
        //if (0 == label)
        //    std::cout << "С��!" << std::endl;
        //else if (1 == label)
        //    std::cout << "����!" << std::endl;
        //else if (2 == label)
        //    std::cout << "����!" << std::endl;
        //else if (3 == label)
        //    std::cout << "����!" << std::endl;
        //else if (4 == label)
        //    std::cout << "����!" << std::endl;
        // std::cout << std::endl;

        //// 5����ɫ
        //featDim = GetDeepFeatSize(hSkin) / 4;
        //pFeatures.reset(featDim);
        //nRetCode = InnerDeepFeat(hSkin, pCropNormFace, 1, 3, 256, 256, pFeatures);

        //score = pFeatures[0] * 1.11f;
        //if (score > 100.0f)
        //    score = 100.0f;    
        //std::cout << "Skin score: " << score << std::endl;
        
        // Uninitialized
        //FaceDetectUninit();
        //FaceAlignmentUninit();
        UninitDeepFeat(hFace);
        //UninitDeepFeat(hSkin);
        //UninitDeepFeat(hXiaci);
        //UninitDeepFeat(hHappy);
        //UninitDeepFeat(hAge);
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

