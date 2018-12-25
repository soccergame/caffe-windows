#include "caffe/face_detection/face_detection.hpp"
#include "autoarray.h"
#include "NormFaceImage.h"
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

#if 0
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "usage:" << std::endl;
        std::cout << "mtcnn_test <module_path> <image_dir> <gpuId>" << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();

    std::string image_dir = argv[2];

    char path_list[512] = { 0 };
    strcpy(path_list, image_dir.c_str());

    int gpuId = atoi(argv[3]);

    MTCNN fd_mtcnn(gpuId);
    int res = LoadModel(pModulePath, fd_mtcnn);

    // Read images of input filepath
    vector<string> imgList;
    readFileList(path_list, imgList);

    for (int l = 0; l < imgList.size(); l++)
    {
        cout << imgList[l] << endl;
        char str[512] = { 0 };
        strcpy(str, imgList[l].c_str());

        char *p = strrchr(str, '.');
        *p = NULL;
        char name[512] = { 0 };
        sprintf(name, "%s.pts_5_1", str);

        res = savelandmark2file(imgList[l].c_str(), name, fd_mtcnn);
        if (0 == res)
            printf("%s ok!\n", imgList[l].c_str());
        else
            printf("%s error!\n", imgList[l].c_str());
    }
    return 0;
}
#endif

int main(int argc, char **argv) {
    if (argc != 12) {
        std::cout << "usage:" << std::endl;
        std::cout << "mtcnn_test <module_path> <base_dir> <file_list> <dst_image_dir> <gpuId>" 
            << " <norm_width> <norm_height> <norm_scale> <norm_size> <norm_type> <norm_method>" 
            << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();

    std::string image_dir = argv[2];
    std::string image_list = argv[3];
    std::string dst_image_dir = argv[4];

    int gpuId = atoi(argv[5]);
    int norm_width = atoi(argv[6]);
    int norm_height = atoi(argv[7]);
    float norm_scale = float(atof(argv[8]));
    int norm_size = atoi(argv[9]);
    int norm_type = atoi(argv[10]);
    int norm_method = atoi(argv[11]);

    hzx::CAffineNormImage affineNorm;
    hzx::CNormImage3pt normImage3pt;
    hzx::CNormImageSimilarity similarityNorm;

    hzx::InterpolateType type;
    if (2 == norm_type)
        type = hzx::InterpolateType::Area;
    else if (1 == norm_type)
        type = hzx::InterpolateType::Cubic;
    else
        type = hzx::InterpolateType::Bilinear;

    int ret = 0;
    if (2 == norm_method) {
        if (112 == norm_size) {
            ret = similarityNorm.Initialize(norm_width, norm_height, norm_scale,
                norm_size, hzx::g_NormPoints_112, 5, type);
        }
        else {
            ret = similarityNorm.Initialize(norm_width, norm_height, norm_scale,
                norm_size, hzx::g_NormPoints_128, 5, type);
        }
        
    } 
    else if (1 == norm_method) {
        if (112 == norm_size) {
            ret = normImage3pt.Initialize(norm_width, norm_height, norm_scale,
                norm_size, hzx::g_NormPoints_112, 5, type);
        }
        else {
            ret = normImage3pt.Initialize(norm_width, norm_height, norm_scale,
                norm_size, hzx::g_NormPoints_3pts_128, 5, type);
        }
    }
    else {
        if (112 == norm_size) {
            ret = affineNorm.Initialize(norm_width, norm_height, norm_scale,
                norm_size, hzx::g_NormPoints_112, 5, type);
        }
        else {
            ret = affineNorm.Initialize(norm_width, norm_height, norm_scale,
                norm_size, hzx::g_NormPoints_128, 5, type);
        }
    }

    MTCNN::BoundingBox face_box;

    MTCNN fd_mtcnn(gpuId);
    int res = LoadModel(pModulePath, fd_mtcnn);

    // Read images of input filepath
    std::vector<std::string> imgList;
    std::fstream fp;
    fp.open(image_list, std::fstream::in | std::fstream::binary);
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

    for (int l = 0; l < imgList.size(); l++)
    {
        //string src_image_name = imgList[l].substr(imgList[l].find_last_of('/'));
        string dst_image = dst_image_dir;// +(src_image_name.substr(0, src_image_name.find_last_of('.'))) + ".png";
        string name = imgList[l];
        size_t loc = name.find_first_of("/\\");
        while (loc != std::string::npos) {
            dst_image += "/";
            dst_image += name.substr(0, loc);
#ifdef _WIN32
            if(0 != access(dst_image.c_str(), 0))
                CreateDirectoryA(dst_image.c_str(), NULL);
#else
            if (0 != access(dst_image.c_str(), F_OK))
                mkdir(dst_image.c_str(), 0777);
#endif
            name = name.substr(loc + 1);
            loc = name.find_first_of("/\\");
        }
        dst_image += "/";
        dst_image += name;

#ifdef _WIN32
        if (0 == access(dst_image.c_str(), 0))
            continue;
#else
        if (0 == access(dst_image.c_str(), F_OK))
            continue;
#endif

        string src_image = image_dir;
        src_image += "/";
        src_image += imgList[l];
        res = save_max_rect_face(src_image, fd_mtcnn, face_box);
        /*cout << "src_image: " << src_image << endl;
        cout << "dst_image: " << dst_image << endl;*/
        if (0 == res) {
            float feaPoints[10];
            for (int j = 0; j < 5; ++j) {
                feaPoints[2 * j] = face_box.points_x[j];
                feaPoints[2 * j + 1] = face_box.points_y[j];
            }

            cv::Mat image = cv::imread(src_image, CV_LOAD_IMAGE_COLOR);
            std::vector<cv::Mat> mv;
            cv::split(image, mv);
            /*AutoArray<unsigned char> pImage(testImg.cols * testImg.rows * testImg.channels());
            for (int h = 0; h < testImg.rows; ++h) {
            for (int w = 0; w < testImg.cols; ++w) {
            for (int c = 0; c < testImg.channels(); ++c) {
            pImage[c * testImg.rows * testImg.cols + h * testImg.cols + w] =
            testImg.ptr<cv::Vec3b>(h)[w][c];
            }
            }
            }*/
            AutoArray<unsigned char> pNormImage5Pt(norm_width * norm_height * image.channels());

            if (2 == norm_method) {
                for (int j = 0; j < image.channels(); ++j) {
                    int retValue = similarityNorm.NormImage(mv[j].data, //pImage.begin() + j * testImg.rows * testImg.cols,
                        image.cols, image.rows, feaPoints, 5,
                        pNormImage5Pt + j * norm_width * norm_height);
                    if (retValue != 0)
                        continue;

                    mv[j] = cv::Mat(norm_height, norm_width, CV_8UC1, pNormImage5Pt + j * norm_width * norm_height);
                }
            }
            else if (1 == norm_method) {
                for (int j = 0; j < image.channels(); ++j) {
                    int retValue = normImage3pt.NormImage(mv[j].data, //pImage.begin() + j * testImg.rows * testImg.cols,
                        image.cols, image.rows, feaPoints, 5,
                        pNormImage5Pt + j * norm_width * norm_height);
                    if (retValue != 0)
                        continue;

                    mv[j] = cv::Mat(norm_height, norm_width, CV_8UC1, pNormImage5Pt + j * norm_width * norm_height);
                }
            }
            else
            {
                for (int j = 0; j < image.channels(); ++j) {
                    int retValue = affineNorm.NormImage(mv[j].data, //pImage.begin() + j * testImg.rows * testImg.cols,
                        image.cols, image.rows, feaPoints, 5,
                        pNormImage5Pt + j * norm_width * norm_height);
                    if (retValue != 0)
                        continue;

                    mv[j] = cv::Mat(norm_height, norm_width, CV_8UC1, pNormImage5Pt + j * norm_width * norm_height);
                }
            }
            

            cv::Mat face_img;
            cv::merge(mv, face_img);

            cv::imwrite(dst_image, face_img);

            if (((l + 1) % 1000) == 0) {
                std::cout << "Handled " << l + 1 << " images" << std::endl;
            }
                
        }
        else {
            cout << src_image << " Error!" << endl;
            continue;
        }     
    }
    return 0;
}
