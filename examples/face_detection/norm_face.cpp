#include "caffe/face_detection/face_detection.hpp"
#include "caffe/common/autoarray.h"
#include "caffe/common/NormFaceImage.h"
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

//bool get_rect_face(const cv::Mat &image, const cv::Rect &face_rect, cv::Mat &face_image, const int crop_size = 128);
const int g_normTemplateSize = 128;
static float g_NormPoints[10] = {
    33.5f, 32.0f,
    93.5f, 32.0f,
    63.5f, 63.5f,
    45.5f, 95.0f,
    81.5f, 95.0f,
};

//void readFileList(const char* basePath, vector<string>& imgFiles)
//{
//    DIR *dir;
//    struct dirent *ptr;
//    char base[1000];
//
//    if ((dir = opendir(basePath)) == NULL)
//    {
//        return;
//    }
//
//    while ((ptr = readdir(dir)) != NULL)
//    {
//        if (strcmp(ptr->d_name, ".") == 0 ||
//            strcmp(ptr->d_name, "..") == 0)
//            continue;
//        else if (ptr->d_type == 8)//file 
//        {
//            int len = strlen(ptr->d_name);
//            // jpg, jpeg, png, bmp
//            if ((ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'p' && ptr->d_name[len - 3] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'e' && ptr->d_name[len - 3] == 'p' && ptr->d_name[len - 4] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'n' && ptr->d_name[len - 3] == 'p') || (ptr->d_name[len - 1] == 'p' && ptr->d_name[len - 2] == 'm' && ptr->d_name[len - 3] == 'b'))
//            {
//                memset(base, '\0', sizeof(base));
//                strcpy(base, basePath);
//                strcat(base, "/");
//                strcat(base, ptr->d_name);
//                imgFiles.push_back(base);
//            }
//        }
//        else if (ptr->d_type == 10)/// link file
//        {
//            int len = strlen(ptr->d_name);
//            if ((ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'p' && ptr->d_name[len - 3] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'e' && ptr->d_name[len - 3] == 'p' && ptr->d_name[len - 4] == 'j') || (ptr->d_name[len - 1] == 'g' && ptr->d_name[len - 2] == 'n' && ptr->d_name[len - 3] == 'p') || (ptr->d_name[len - 1] == 'p' && ptr->d_name[len - 2] == 'm' && ptr->d_name[len - 3] == 'b'))
//            {
//                memset(base, '\0', sizeof(base));
//                strcpy(base, basePath);
//                strcat(base, "/");
//                strcat(base, ptr->d_name);
//                imgFiles.push_back(base);
//            }
//        }
//        else if (ptr->d_type == 4)//dir
//        {
//            memset(base, '\0', sizeof(base));
//            strcpy(base, basePath);
//            strcat(base, "/");
//            strcat(base, ptr->d_name);
//            readFileList(base, imgFiles);
//        }
//    }
//    closedir(dir);
//}

int LoadModel(const char* pModulePath, MTCNN &fd_mtcnn)
{
    std::string strDllPath;
    strDllPath = pModulePath;
    strDllPath += MTCNN_MODEL_FILE;
    //mtcnn model load
    int ret = fd_mtcnn.init(strDllPath);

    return ret;
}


//bool get_rect_face(const cv::Mat &image, const cv::Rect &face_rect, cv::Mat &face_image, const int crop_size)
//{
//
//    if ((image.empty()) || (face_rect.area() <= 0))
//    {
//        return false;
//    }
//    //get face image    
//    cv::Rect face_rect_fixed = face_rect & cv::Rect(0, 0, image.cols, image.rows); //与操作，防止越界。
//
//    int rect_xy_diff, margin;
//    rect_xy_diff = (face_rect_fixed.height - face_rect_fixed.width);
//    margin = abs(rect_xy_diff);
//
//    if (rect_xy_diff > 0) {
//        face_rect_fixed.y += margin / 2;
//        face_rect_fixed.height -= margin;
//    }
//    else {
//        face_rect_fixed.x += margin / 2;
//        face_rect_fixed.width -= margin;
//    }
//    cv::Mat tmp_img;
//    tmp_img = image(face_rect_fixed);
//    cv::resize(tmp_img, face_image, cv::Size(crop_size, crop_size));
//
//    return true;
//}

//vector <string> findfile(string path)
//{
//    DIR *dp;
//    struct dirent *dirp;
//    vector<std::string> filename;
//    if ((dp = opendir(path.c_str())) == NULL)
//        perror("open dir error");
//    while ((dirp = readdir(dp)) != NULL)
//    {
//        if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)    ///current dir OR parrent dir
//            continue;
//
//        const char *p = strrchr(string(dirp->d_name).c_str(), '.');
//        const char *q = p + 1;
//        if (strcmp(q, "jpg") == 0 || strcmp(q, "png") == 0)
//        {
//            filename.push_back(path + '/' + string(dirp->d_name));
//        }
//
//    }
//
//    for (int i = 0; i<filename.size(); i++)
//        cout << i << ":" << filename[i] << endl;
//    closedir(dp);
//    return filename;
//}

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
    if (argc != 8) {
        std::cout << "usage:" << std::endl;
        std::cout << "mtcnn_test <module_path> <base_dir> <file_list> <dst_image_dir> <gpuId>" << std::endl;
        return -1;
    }
    std::string modulePath = argv[1];
    const char* pModulePath = modulePath.c_str();

    std::string image_dir = argv[2];
    std::string image_list = argv[3];
    std::string dst_image_dir = argv[4];

    int gpuId = atoi(argv[5]);
    int norm_size = atoi(argv[6]);
    float norm_scale = float(atof(argv[7]));

    int eyeCenterY, distEyeCMouthC, distEyeC;
    eyeCenterY = int(0.5 * ((g_NormPoints[1] + g_NormPoints[3]) * norm_scale + norm_size - g_normTemplateSize * norm_scale) + 0.5);
    distEyeCMouthC = int(0.5 * (g_NormPoints[7] + g_NormPoints[9] - g_NormPoints[1] - g_NormPoints[3]) * norm_scale + 0.5);
    distEyeC = int((g_NormPoints[2] - g_NormPoints[0]) * norm_scale + 0.5);

    ALGORITHMUTILS::CAffineNormImage affineNorm;
    affineNorm.Initialize(norm_size, norm_size, norm_scale, 
        g_normTemplateSize, g_NormPoints, 5, ALGORITHMUTILS::Bilinear);

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
            CreateDirectoryA(dst_image.c_str(), NULL);
#else
            int iStatus = access(dst_image.c_str(), F_OK);
            if (iStatus != 0)
                mkdir(dst_image.c_str(), 0777);
#endif
            name = name.substr(loc + 1);
            loc = name.find_first_of("/\\");
        }
        dst_image += "/";
        dst_image += name;

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
            AutoArray<unsigned char> pNormImage5Pt(norm_size * norm_size * image.channels());

            for (int j = 0; j < image.channels(); ++j) {
                int retValue = affineNorm.NormImage(mv[j].data, //pImage.begin() + j * testImg.rows * testImg.cols,
                    image.cols, image.rows, feaPoints, 5,
                    pNormImage5Pt + j * norm_size * norm_size);
                if (retValue != 0)
                    continue;

                mv[j] = cv::Mat(norm_size, norm_size, CV_8UC1, pNormImage5Pt + j * norm_size * norm_size);
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
