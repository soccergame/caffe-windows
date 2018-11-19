#ifndef _FACE_DETECTION_HPP_
#define _FACE_DETECTION_HPP_

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

//#include <numeric>

using namespace caffe;
using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::sort;
//using std::iota;
//using namespace cv;
//#define CPU_ONLY
namespace FaceDetection{
    
    class MTCNN{
    public:
        struct FaceInfo{
        };
        enum COLOR_ORDER{
            GRAY,
            RGBA,
            RGB,
            BGRA,
            BGR
        };
        enum MODEL_VERSION{
            MODEL_V1,
            MODEL_V2
        };
        enum NMS_TYPE{
            MIN,
            UNION,
        };
        enum IMAGE_DIRECTION{
            ORIENT_LEFT,
            ORIENT_RIGHT,
            ORIENT_UP,
            ORIENT_DOWN,
        };
        struct BoundingBox{
            //rect two points
            float x1, y1;
            float x2, y2;
            //regression
            float dx1, dy1;
            float dx2, dy2;
            //cls
            float score;
            //inner points
            float points_x[5];
            float points_y[5];
        };
        
        struct CmpBoundingBox{
            bool operator() (const BoundingBox& b1, const BoundingBox& b2)
            {
                return b1.score > b2.score;
            }
        };
    private:
        boost::shared_ptr< Net<float> > P_Net;
        boost::shared_ptr< Net<float> > R_Net;
        boost::shared_ptr< Net<float> > O_Net;
        NetParameter P_param_;
        NetParameter R_param_;
        NetParameter O_param_;
        NetParameter P_weights_;
        NetParameter R_weights_;
        NetParameter O_weights_;
        
	//used by model 2 version
        boost::shared_ptr< Net<float> > L_Net;
        NetParameter L_param_;
        NetParameter L_weights_;
        double                           img_mean;
        double                           img_var;
        cv::Size                         input_geometry_;
        int                              num_channels_;
        MODEL_VERSION                    model_version;
        
    public:
	MTCNN(int GpuId = 0);
        int init(const std::string& pModelPath);
        vector< BoundingBox > Detect(const cv::Mat& img, const COLOR_ORDER color_order, const IMAGE_DIRECTION orient, int min_size = 20, float P_thres = 0.6, float R_thres = 0.7, float O_thres = 0.7, bool is_fast_resize = true, float scale_factor = 0.709);
        bool alignFace(const cv::Mat image, cv::Mat & alignedImg, BoundingBox box);
        cv::Size GetInputSize() { return input_geometry_; }
        int      GetInputChannel() { return num_channels_; }
        vector<int> GetInputShape() {
            Blob<float>* input_layer = P_Net->input_blobs()[0];
            return input_layer->shape();
        }

    private:
        void generateBoundingBox(const vector<float>& boxRegs, const vector<int>& box_shape,
                                 const vector<float>& cls, const vector<int>& cls_shape,
                                 float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                                );
        void filteroutBoundingBox(const vector<BoundingBox>& boxes, 
                                  const vector<float>& boxRegs, const vector<int>& box_shape,
                                  const vector<float>& cls, const vector<int>& cls_shape,
                                  const vector< float >& points, const vector< int >& points_shape,
                                  float threshold, vector<BoundingBox>& filterOutBoxes);
        void nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes);
        
        //void pad(vector<BoundingBox>& boxes, int imgW, int imgH);
        
        //vector<int> nms(vector<int> boxs, );
        vector<float> predict(const cv::Mat& img);
        void wrapInputLayer(boost::shared_ptr< Net<float> > net, std::vector<cv::Mat>* input_channels);
        void pyrDown(const vector<cv::Mat>& img_channels,float scale, std::vector<cv::Mat>* input_channels);
        void buildInputChannels(const std::vector<cv::Mat>& img_channels, const std::vector<BoundingBox>& boxes,
                                const cv::Size& target_size, std::vector<cv::Mat>* input_channels);
        
    };
}
#endif
