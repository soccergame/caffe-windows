#include "caffe/face_detection/face_detection.hpp" 
#include "caffe/common/autoarray.h"
#include "caffe/common/AlgorithmUtils.h"
#ifdef _WIN32
#include <io.h>
#define R_OK 4 /* Test for read permission. */
#define W_OK 2 /* Test for write permission. */
#define X_OK 1 /* Test for execute permission. */
#define F_OK 0 /* Test for existence. */
#else
#include <unistd.h>
#include <sys/types.h>
#endif

#define pts_num (5)

namespace FaceDetection{
    MTCNN::MTCNN(int GpuId)
    {
        this->model_version = MODEL_V1;
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        if (GpuId < 0)
            Caffe::set_mode(Caffe::CPU);
        else {
            Caffe::set_mode(Caffe::GPU);
            Caffe::SetDevice(GpuId);
        }     
#endif
    }

    int MTCNN::init(const std::string& pModelPath)
    {
        AutoArray<char> encryptedData;
        int dataSize = 0;
        std::fstream fp;
        fp.open(pModelPath, std::fstream::in | std::fstream::binary);
        if (fp.is_open())
        {
            fp.seekg(0, std::fstream::end);
            int dataLen = int(fp.tellg());
            fp.seekg(0, std::fstream::beg);

            dataSize = ((dataLen + 7) / 8) * 8;
            encryptedData.resize(dataSize);
            fp.read(encryptedData.begin(), dataSize);
        }
        fp.close();

        if (dataSize <= 0)
            return -1;

        int *pBuffer = reinterpret_cast<int *>(encryptedData.begin());
        // encrypt data by shift left		
        int numOfData = dataSize / sizeof(pBuffer[0]);
        for (int i = 0; i < numOfData; ++i)
        {
            int tempData = pBuffer[i];
            pBuffer[i] = brc_sn::ror(static_cast<unsigned int>(tempData),
                brc_sn::g_shiftBits);
        }

        const int modelnumber = pBuffer[0];
        std::vector<int> protoTxtLen, modelSize;
        for (int i = 0; i < modelnumber; ++i)
        {
            protoTxtLen.push_back(pBuffer[2 * i + 1]);
            modelSize.push_back(pBuffer[2 * i + 2]);
        }
        unsigned char *pDataBuf = reinterpret_cast<unsigned char *>(encryptedData.begin()) + sizeof(int) * (2 * modelnumber + 1);

        if (modelnumber != 3 && modelnumber != 4)
            return 1;

        int retValue = 0;
        // Load P_Net
        retValue = caffe::ReatNetParamsFromBuffer(
            pDataBuf, protoTxtLen[0], &P_param_);
        CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &P_param_));
        retValue = ReatNetParamsFromBuffer(
            pDataBuf + protoTxtLen[0], modelSize[0], &P_weights_);
        CHECK_EQ(retValue, 0) << "Read net parameters from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &P_weights_));
        pDataBuf += protoTxtLen[0] + modelSize[0];

        // Load R_Net
        retValue = caffe::ReatNetParamsFromBuffer(
            pDataBuf, protoTxtLen[1], &R_param_);
        CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &R_param_));
        retValue = ReatNetParamsFromBuffer(
            pDataBuf + protoTxtLen[1], modelSize[1], &R_weights_);
        CHECK_EQ(retValue, 0) << "Read net parameters from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &R_weights_));
        pDataBuf += protoTxtLen[1] + modelSize[1];

        // Load O_Net
        retValue = caffe::ReatNetParamsFromBuffer(
            pDataBuf, protoTxtLen[2], &O_param_);
        CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &O_param_));
        retValue = ReatNetParamsFromBuffer(
            pDataBuf + protoTxtLen[2], modelSize[2], &O_weights_);
        CHECK_EQ(retValue, 0) << "Read net parameters from buffer error, code: " << retValue;
        CHECK(caffe::UpgradeNetAsNeeded("<memory>", &O_weights_));
        pDataBuf += protoTxtLen[2] + modelSize[2];

        if (modelnumber == 4) {
            retValue = caffe::ReatNetParamsFromBuffer(
                pDataBuf, protoTxtLen[3], &L_param_);
            CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
            CHECK(caffe::UpgradeNetAsNeeded("<memory>", &L_param_));
            retValue = ReatNetParamsFromBuffer(
                pDataBuf + protoTxtLen[3], modelSize[3], &L_weights_);
            CHECK_EQ(retValue, 0) << "Read net parameters from buffer error, code: " << retValue;
            CHECK(caffe::UpgradeNetAsNeeded("<memory>", &L_weights_));
        }

        P_param_.mutable_state()->set_phase(TEST);
        P_Net.reset(new Net<float>(P_param_));
        P_Net->CopyTrainedLayersFrom(P_weights_);

        R_param_.mutable_state()->set_phase(TEST);
        R_Net.reset(new Net<float>(R_param_));
        R_Net->CopyTrainedLayersFrom(R_weights_);

        O_param_.mutable_state()->set_phase(TEST);
        O_Net.reset(new Net<float>(O_param_));
        O_Net->CopyTrainedLayersFrom(O_weights_);

        //L
        if (model_version == MODEL_V2) //MODEL_V2 add LNet
        {
            L_param_.mutable_state()->set_phase(TEST);
            L_Net.reset(new Net<float>(L_param_));
            L_Net->CopyTrainedLayersFrom(L_weights_);
        }

        Blob<float>* input_layer = P_Net->input_blobs()[0];
        num_channels_ = input_layer->channels();
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        //set img_mean
        img_mean = 127.5;
        //set img_var
        img_var = 0.0078125;

        return 0;
    }
    
    void MTCNN::wrapInputLayer(boost::shared_ptr< Net<float> > net, vector< cv::Mat >* input_channels)
    {
        Blob<float>* input_layer = net->input_blobs()[0];
        
        int width = input_layer->width();
        int height = input_layer->height();
        
        float* input_data = input_layer->mutable_cpu_data();
        for(int j = 0; j < input_layer->num(); j++)
        {
            for(int i = 0; i < input_layer->channels(); i ++)
            {
                cv::Mat channel(height, width, CV_32FC1, input_data);
                input_channels->push_back(channel);
                input_data += width * height;
            }
        }
    }

    void MTCNN::pyrDown(const vector<cv::Mat>& img_channels, float scale, vector< cv::Mat >* input_channels)
    {
        assert(img_channels.size() == input_channels->size());
        int hs = (*input_channels)[0].rows;
        int ws = (*input_channels)[0].cols;
        cv::Mat img_resized;
        for(int i = 0; i < img_channels.size(); i ++)
        {
            cv::resize(img_channels[i], (*input_channels)[i], cv::Size(ws, hs));
            //cv::imshow("test", (*input_channels)[i]);
            //cv::waitKey();
        }
    }

    void MTCNN::buildInputChannels(const vector< cv::Mat >& img_channels, const std::vector<BoundingBox>& boxes,
                                          const cv::Size& target_size, vector< cv::Mat >* input_channels)
    {
        assert(img_channels.size() * boxes.size() == input_channels->size() );
        cv::Rect img_rect(0, 0, img_channels[0].cols, img_channels[0].rows);
        for(int n = 0; n < boxes.size(); n++)
        {
            cv::Rect rect;
            rect.x = boxes[n].x1;
            rect.y = boxes[n].y1;
            rect.width = boxes[n].x2 - boxes[n].x1 + 1;
            rect.height = boxes[n].y2 - boxes[n].y1 + 1;
            cv::Rect cuted_rect = rect & img_rect;
            cv::Rect inner_rect(cuted_rect.x - rect.x, cuted_rect.y - rect.y, cuted_rect.width, cuted_rect.height);
            for(int c = 0; c < img_channels.size(); c++)
            {
                cv::Mat tmp(rect.height, rect.width, CV_32FC1, cv::Scalar(0.0));
                img_channels[c](cuted_rect).copyTo(tmp(inner_rect));
                cv::resize(tmp, (*input_channels)[n * img_channels.size() + c], target_size);
                //cv::imshow("show_tmp", (*input_channels)[n * img_channels.size() + c]);
                //cv::waitKey();
            }
        }
    }

    void MTCNN::generateBoundingBox(const vector<float>& boxRegs, const vector<int>& box_shape,
                                 const vector<float>& cls, const vector<int>& cls_shape,
                                 float scale, float threshold, vector<BoundingBox>& filterOutBoxes
                                )
    {
        //clear output element
        filterOutBoxes.clear();
        int stride = 2;
        int cellsize = 12;
        assert(box_shape.size() == cls_shape.size());
        assert(box_shape[3] == cls_shape[3] && box_shape[2] == cls_shape[2]);
        assert(box_shape[0] == 1 && cls_shape[0] == 1);
        assert(box_shape[1] == 4 && cls_shape[1] == 2);
        int w = box_shape[3];
        int h = box_shape[2];
        //int n = box_shape[0];
        for(int y = 0; y < h; y ++)
        {
            for(int x = 0; x < w; x ++)
            {
                float score =     cls[0 * 2 * w * h + 1 * w * h + w * y + x];
                if ( score >= threshold)
                {
                    BoundingBox box;
                    box.dx1 = boxRegs[0 * w * h + w * y + x];
                    box.dy1 = boxRegs[1 * w * h + w * y + x];
                    box.dx2 = boxRegs[2 * w * h + w * y + x];
                    box.dy2 = boxRegs[3 * w * h + w * y + x];
                    
                    box.x1 = std::floor( (stride * x + 1) / scale );
                    box.y1 = std::floor( (stride * y + 1) / scale );
                    box.x2 = std::floor( (stride * x + cellsize) / scale );  
                    box.y2 = std::floor( (stride * y + cellsize) / scale );
                    box.score = score;
                    //add elements
                    filterOutBoxes.push_back(box);
                    //filterOutRegs.push_back(reg);
                }
            }
        }
    }

    void MTCNN::filteroutBoundingBox(const vector< MTCNN::BoundingBox >& boxes, 
                                            const vector< float >& boxRegs, const vector< int >& box_shape, 
                                            const vector< float >& cls, const vector< int >& cls_shape, 
                                            const vector< float >& points, const vector< int >& points_shape,
                                            float threshold, vector< MTCNN::BoundingBox >& filterOutBoxes)
    {
        filterOutBoxes.clear();
        assert(box_shape.size() == cls_shape.size());
        assert(box_shape[0] == boxes.size() && cls_shape[0] == boxes.size());
        assert(box_shape[1] == 4 && cls_shape[1] == 2);
        if(points.size() > 0)
        {
            assert(points_shape[0] == boxes.size() && points_shape[1] == 10);
        }

        //int w = box_shape[3];
        //int h = box_shape[2];
        for(int i = 0; i < boxes.size(); i ++)
        {
            float score = cls[i * 2 + 1];
            if ( score > threshold )
            {
                BoundingBox box = boxes[i];
                float w = boxes[i].y2 - boxes[i].y1 + 1;
                float h = boxes[i].x2 - boxes[i].x1 + 1;
                if( points.size() > 0)
                {
                    for(int p = 0; p < 5; p ++)
                    {
                        box.points_x[p] = points[i * 10 + 5 + p] * w + boxes[i].x1 - 1;
                        box.points_y[p] = points[i * 10 + p] * h + boxes[i].y1 - 1;
                    }
                }
                box.dx1 = boxRegs[i * 4 + 0];
                box.dy1 = boxRegs[i * 4 + 1];
                box.dx2 = boxRegs[i * 4 + 2];
                box.dy2 = boxRegs[i * 4 + 3];
                
                //regression
                //w = box.x2 - box.x1 + 1;
                //h = box.y2 - box.y1 + 1;
                
                box.x1 = boxes[i].x1 + box.dy1 * w;
                box.y1 = boxes[i].y1 + box.dx1 * h;
                box.x2 = boxes[i].x2 + box.dy2 * w;
                box.y2 = boxes[i].y2 + box.dx2 * h;
                
                //rerec
                w = box.x2 - box.x1;
                h = box.y2 - box.y1;
                float l = std::max(w, h);
                box.x1 += (w - l) * 0.5;
                box.y1 += (h - l) * 0.5;
                box.x2 = box.x1 + l;
                box.y2 = box.y1 + l;
                box.score = score;
                
                filterOutBoxes.push_back(box);
            }
        }
    }

    //void MTCNN::pad(vector< MTCNN::BoundingBox >& boxes, int imgW, int imgH)
    //{
    //    
    //}

    void MTCNN::nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes)
    {
        filterOutBoxes.clear();
        if(boxes.size() == 0)
            return;
        //descending sort
        sort(boxes.begin(), boxes.end(), CmpBoundingBox() );
        vector<size_t> idx(boxes.size());
        //std::iota(idx.begin(), idx.end(), 0);//create index
        for(int i = 0; i < idx.size(); i++)
        { 
            idx[i] = i; 
        }
        while(idx.size() > 0)
        {
            int good_idx = idx[0];
            filterOutBoxes.push_back(boxes[good_idx]);
            //hypothesis : the closer the scores are similar
            vector<size_t> tmp = idx;
            idx.clear();
            for(int i = 1; i < tmp.size(); i++)
            {
                int tmp_i = tmp[i];
                float inter_x1 = std::max( boxes[good_idx].x1, boxes[tmp_i].x1 );
                float inter_y1 = std::max( boxes[good_idx].y1, boxes[tmp_i].y1 );
                float inter_x2 = std::min( boxes[good_idx].x2, boxes[tmp_i].x2 );
                float inter_y2 = std::min( boxes[good_idx].y2, boxes[tmp_i].y2 );
                 
                float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
                float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);
                
                float inter_area = w * h;
                float area_1 = (boxes[good_idx].x2 - boxes[good_idx].x1 + 1) * (boxes[good_idx].y2 - boxes[good_idx].y1 + 1);
                float area_2 = (boxes[i].x2 - boxes[i].x1 + 1) * (boxes[i].y2 - boxes[i].y1 + 1);
                float o = ( type == UNION ? (inter_area / (area_1 + area_2 - inter_area)) : (inter_area / std::min(area_1 , area_2)) );           
                if( o <= threshold )
                    idx.push_back(tmp_i);
            }
        }
    }

    //#define IMAGE_DEBUG
    /*
     *note: input channel order must be rgb
     */
    vector< MTCNN::BoundingBox > MTCNN::Detect(const cv::Mat& img, const COLOR_ORDER color_order, const IMAGE_DIRECTION orient, int min_size, float P_thres, float R_thres, float O_thres, bool is_fast_resize, float scale_factor)
    {
        /*change image format*/
        cv::Mat sample;
        if( img.channels() == 3 && num_channels_ == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
            if( color_order == RGBA)
                cv::cvtColor(img, sample, cv::COLOR_RGBA2RGB);
            else
                cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
        else if ( img.channels() == 1 && num_channels_ == 3 )
            cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
        else
            sample = img;
        cv::Mat sample_normalized;
        //convert to float and normalize
        sample.convertTo( sample_normalized, CV_32FC3, img_var, -img_mean * img_var);
    #ifdef IMAGE_DEBUG
        cv::imshow("before_rotate", sample_normalized);
        cv::waitKey();
    #endif
        if(orient == IMAGE_DIRECTION::ORIENT_UP)
            sample_normalized = sample_normalized.t();
        else if(orient == IMAGE_DIRECTION::ORIENT_DOWN){
            cv::flip(sample_normalized, sample_normalized, -1);
            sample_normalized = sample_normalized.t();
        }
        else if(orient == IMAGE_DIRECTION::ORIENT_LEFT)
        {
            cv::flip(sample_normalized, sample_normalized, -1);
        }
    #ifdef IMAGE_DEBUG
        cv::imshow("after_rotate", sample_normalized);
        cv::waitKey();
    #endif
        vector<float> points;
        const int img_H = sample_normalized.rows;
        const int img_W = sample_normalized.cols;
        int minl  = cv::min(img_H, img_W);
        //split the input image
        vector<cv::Mat> sample_norm_channels;
        cv::split(sample_normalized, sample_norm_channels);
        if(color_order == BGR || color_order == BGRA)
        {
            cv::Mat tmp = sample_norm_channels[0];
            sample_norm_channels[0] = sample_norm_channels[2];
            sample_norm_channels[2] = tmp;
        }
        //cout<<pyr_channels[2].at<float>(100,25)<<endl;
        //cv::addWeighted(sample_float, img_var, cv::Mat(), 0, -img_mean * img_var, sample_normalized);
        float m = 12.0 / min_size;
        minl *= m;
        vector<float> all_scales;
        float cur_scale = 1.0;
        while( minl >= 12.0 )
        {
            all_scales.push_back( m * cur_scale);
            cur_scale *= scale_factor;
            minl *= scale_factor;
        }
        /*stage 1: P_Net forward can get rectangle and regression */
        vector<BoundingBox> totalBoxes;
        for(int i = 0; i < all_scales.size(); i ++)
        {
            vector<cv::Mat> pyr_channels;
            cur_scale = all_scales[i];
            int hs = cvCeil(img_H * cur_scale);
            int ws = cvCeil(img_W * cur_scale);
            //对输入的形状进行变化
            Blob<float>* input_layer = P_Net->input_blobs()[0];
            input_layer->Reshape(1, num_channels_, hs, ws);
            //// forward dimension change to all layers
            P_Net->Reshape();
            //wrap input layers
            wrapInputLayer( P_Net, &pyr_channels);
            //对图像每个通道进行下采样
            pyrDown(sample_norm_channels, cur_scale, &pyr_channels);
            //P Net forward operation
            const vector<Blob<float>*> out = P_Net->Forward();
            /* copy the output layer to a vector*/
            Blob<float>* output_layer0 = out[0];
            vector<int> box_shape = output_layer0->shape();
            int output_size = box_shape[0] * box_shape[1] * box_shape[2] * box_shape[3];
            const float* begin0 = output_layer0->cpu_data();
            const float* end0 = output_size + begin0;
            vector<float> regs(begin0, end0);
            
            Blob<float>* output_layer1 = out[1];
            vector<int> cls_shape = output_layer1->shape();
            output_size = cls_shape[0] * cls_shape[1] * cls_shape[2] * cls_shape[3];
            const float* begin1 = output_layer1->cpu_data();
            const float* end1 = output_size + begin1;
            vector<float> cls(begin1, end1);
            vector<BoundingBox> filterOutBoxes;
            vector<BoundingBox> nmsOutBoxes;
            //vector<BoundingBox> filterOutRegs;
            generateBoundingBox(regs, box_shape, cls, cls_shape, cur_scale, P_thres, filterOutBoxes);
            nms_cpu(filterOutBoxes, 0.5, UNION, nmsOutBoxes);
            if(nmsOutBoxes.size() > 0)
                totalBoxes.insert(totalBoxes.end(), nmsOutBoxes.begin(), nmsOutBoxes.end());
        }
        //debug
    #ifdef IMAGE_DEBUG
        cv::Mat tmp = img.t();//for debug
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(255, 0, 0), 2);
        }
        cv::imshow("stage1_test", tmp);
        cv::waitKey();
    #endif
        //end debug
        //do global nms operator
        if (totalBoxes.size() > 0)
        {
            vector<BoundingBox> globalFilterBoxes;
            //cout<<totalBoxes.size()<<endl;
            nms_cpu(totalBoxes, 0.7, UNION, globalFilterBoxes);
            //cout<<globalFilterBoxes.size()<<endl;
            //totalBoxes = globalFilterBoxes;
            totalBoxes.clear();
            //cout<<totalBoxes.size()<<endl;
            for(int i = 0; i < globalFilterBoxes.size(); i ++)
            {
                float regw = globalFilterBoxes[i].y2 - globalFilterBoxes[i].y1 ;
                float regh = globalFilterBoxes[i].x2 - globalFilterBoxes[i].x1;
                BoundingBox box;
                float x1 = globalFilterBoxes[i].x1 + globalFilterBoxes[i].dy1 * regw;
                float y1 = globalFilterBoxes[i].y1 + globalFilterBoxes[i].dx1 * regh;
                float x2 = globalFilterBoxes[i].x2 + globalFilterBoxes[i].dy2 * regw;
                float y2 = globalFilterBoxes[i].y2 + globalFilterBoxes[i].dx2 * regh;
                float score = globalFilterBoxes[i].score;
                float h = y2 - y1;
                float w = x2 - x1;
                float l = std::max(h, w);
                x1 += (w - l) * 0.5;
                y1 += (h - l) * 0.5;
                x2 = x1 + l;
                y2 = y1 + l;
                box.x1 = x1, box.x2 = x2, box.y1 = y1, box.y2 = y2;
                totalBoxes.push_back(box);
            }
        }
    #ifdef IMAGE_DEBUG
        //debug
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("stage1_test", tmp);
        cv::waitKey();
        //end debug
    #endif
        //the second stage: R-Net
        
        if(totalBoxes.size() > 0)
        {
            vector<cv::Mat> n_channels;
            Blob<float>* input_layer = R_Net->input_blobs()[0];
            input_layer->Reshape(totalBoxes.size(), num_channels_, 24, 24);
            R_Net->Reshape();
            wrapInputLayer(R_Net, &n_channels);
            //fillout n_channels
            buildInputChannels(sample_norm_channels, totalBoxes, cv::Size(24,24), &n_channels);
            //R_Net forward
            R_Net->Forward();
            /*copy output layer to vector*/
            Blob<float>* output_layer0 = R_Net->output_blobs()[0];
            vector<int> box_shape = output_layer0->shape();
            int output_size = box_shape[0] * box_shape[1];
            const float* begin0 = output_layer0->cpu_data();
            const float* end0 = output_size + begin0;
            vector<float> regs(begin0, end0);
            
            Blob<float>* output_layer1 = R_Net->output_blobs()[1];
            vector<int> cls_shape = output_layer1->shape();
            output_size = cls_shape[0] * cls_shape[1];
            const float* begin1 = output_layer1->cpu_data();
            const float* end1 = output_size + begin1;
            vector<float> cls(begin1, end1);

            vector<BoundingBox> filterOutBoxes;
            filteroutBoundingBox(totalBoxes, regs, box_shape, cls, cls_shape, vector<float>(), vector<int>(), R_thres, filterOutBoxes);
            nms_cpu(filterOutBoxes, 0.7, UNION, totalBoxes);
        }
    #ifdef IMAGE_DEBUG
        //debug
        tmp = img.t();
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow("stage2_test", tmp);
        cv::waitKey();
        //end debug
    #endif
        // do third stage: O-Net
        if(totalBoxes.size() > 0)
        {
            vector<cv::Mat> n_channels;
            Blob<float>* input_layer = O_Net->input_blobs()[0];
            input_layer->Reshape(totalBoxes.size(), num_channels_, 48, 48);
            O_Net->Reshape();
            wrapInputLayer(O_Net, &n_channels);
            //fillout n_channels
            buildInputChannels(sample_norm_channels, totalBoxes, cv::Size(48,48), &n_channels);
            //O_Net forward
            O_Net->Forward();
            /*copy output layer to vector*/
            Blob<float>* output_layer0 = O_Net->output_blobs()[0];
            vector<int> box_shape = output_layer0->shape();
            int output_size = box_shape[0] * box_shape[1];
            const float* begin0 = output_layer0->cpu_data();
            const float* end0 = output_size + begin0;
            vector<float> regs(begin0, end0);
            
            Blob<float>* output_layer1 = O_Net->output_blobs()[1];
            vector<int> points_shape = output_layer1->shape();
            output_size = points_shape[0] * points_shape[1];
            const float* begin1 = output_layer1->cpu_data();
            const float* end1 = output_size + begin1;
            vector<float> points(begin1, end1);
            
            Blob<float>* output_layer2 = O_Net->output_blobs()[2];
            vector<int> cls_shape = output_layer2->shape();
            output_size = cls_shape[0] * cls_shape[1];
            const float* begin2 = output_layer2->cpu_data();
            const float* end2 = output_size + begin2;
            vector<float> cls(begin2, end2);
            
            vector<BoundingBox> filterOutBoxes;
            filteroutBoundingBox(totalBoxes, regs, box_shape, cls, cls_shape, points, points_shape, O_thres, filterOutBoxes);
            nms_cpu(filterOutBoxes, 0.7, MIN, totalBoxes);
        }
    #ifdef IMAGE_DEBUG
        //debug
        tmp = img.t();
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 255), 2);
            for(int i = 0; i < 5; i ++)
                cv::circle(tmp, cv::Point(totalBoxes[k].points_x[i], totalBoxes[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow("stage3_test", tmp);
        cv::waitKey();
        //end debug
    #endif
        //fouth stage : L_Net
        if(totalBoxes.size() > 0 && model_version == MODEL_V2)
        {
            cv::Rect img_rect(0, 0, img_W, img_H);
            vector<cv::Mat> n_channels;
            Blob<float>* input_layer = L_Net->input_blobs()[0];
            input_layer->Reshape(totalBoxes.size(), 15, 24, 24);
            L_Net->Reshape();
            wrapInputLayer(L_Net, &n_channels);
            //fillout n_channels
            //buildInputChannels(sample_norm_channels, totalBoxes, cv::Size(24,24), n_channels);
            vector<int> patchws;
            for(int i = 0; i < totalBoxes.size(); i++)
            {
                int patchw = std::max(totalBoxes[i].x2 - totalBoxes[i].x1, totalBoxes[i].y2 - totalBoxes[i].y1);
                patchw *= 0.25;
                patchw += patchw%2;
                patchws.push_back(patchw);
                for(int k = 0; k < 5; k++)
                {
                    cv::Rect rect;
                    rect.x = totalBoxes[i].points_x[k] - patchw * 0.5;
                    rect.y = totalBoxes[i].points_y[k] - patchw * 0.5;
                    rect.width = patchw;
                    rect.height = patchw;
                    cv::Rect cuted_rect = rect & img_rect;
                    cv::Rect inner_rect(cuted_rect.x - rect.x, cuted_rect.y - rect.y,
                        cuted_rect.width, cuted_rect.height);
                    for(int j = 0; j < num_channels_; j++)
                    {
                        cv::Mat tmp(rect.height, rect.width, CV_32F, cv::Scalar(0.0));
                        sample_norm_channels[k](cuted_rect).copyTo(tmp(inner_rect));
                        cv::resize(tmp, n_channels[i * 5 * num_channels_ + k * num_channels_ + j], cv::Size(24,24));
                    }
                }
            }
            //L_Net forward
            L_Net->Forward();
            //regression points
            //对x轴或y轴相对中心位移大于0.35的，采取不改变措施
            //对于位移较小的点，对其位移进行更新
            for(int k = 0; k < 5; k++)
            {
                //copy output layer to vector
                Blob<float>* output_layer = L_Net->output_blobs()[k];
                const float* begin = output_layer->cpu_data();
                const float* end = output_layer->count() + begin;
                vector<float> points(begin, end);
                for(int j = 0; j < totalBoxes.size(); j++)
                {
                    float out_x = points[j * 2 + 0];
                    float out_y = points[j * 2 + 1];
                    if( std::fabs( out_x - 0.5 ) <= 0.35 && std::fabs( out_y - 0.5 ) <= 0.35)
                    {
                        totalBoxes[j].points_x[k] += (-0.5 + out_x) * patchws[j];
                        totalBoxes[j].points_y[k] += (-0.5 + out_y) * patchws[j];
                    }
                }
            }
        }
    #ifdef IMAGE_DEBUG
        //debug
        tmp = img.t();
        for(int k = 0; k < totalBoxes.size(); k++)
        {
            cv::rectangle(tmp, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(0, 255, 255), 2);
            for(int i = 0; i < 5; i ++)
                cv::circle(tmp, cv::Point(totalBoxes[k].points_x[i], totalBoxes[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow("stage4_test", tmp);
        cv::waitKey();
        //end debug
    #endif
        for(int i = 0; i < totalBoxes.size(); i++)
        {
            if(orient == ORIENT_UP){
                std::swap(totalBoxes[i].x1, totalBoxes[i].y1);
                std::swap(totalBoxes[i].x2, totalBoxes[i].y2);
                for(int k = 0; k < 5; k++)
                {
                    std::swap(totalBoxes[i].points_x[k], totalBoxes[i].points_y[k]);
                }
            }
            else if(orient == ORIENT_DOWN){
                totalBoxes[i].x1 = img_W - totalBoxes[i].x1;
                totalBoxes[i].y1 = img_H - totalBoxes[i].y1;
                totalBoxes[i].x2 = img_W - totalBoxes[i].x2;
                totalBoxes[i].y2 = img_H - totalBoxes[i].y2;
                std::swap(totalBoxes[i].x1, totalBoxes[i].y1);
                std::swap(totalBoxes[i].x2, totalBoxes[i].y2);
                for(int k = 0; k < 5; k++)
                {
                    totalBoxes[i].points_x[k] = img_W - totalBoxes[i].points_x[k];
                    totalBoxes[i].points_y[k] = img_H - totalBoxes[i].points_y[k];
                    std::swap(totalBoxes[i].points_x[k], totalBoxes[i].points_y[k]);
                }
            }
            else if(orient == ORIENT_LEFT){
                totalBoxes[i].x1 = img_W - totalBoxes[i].x1;
                totalBoxes[i].y1 = img_H - totalBoxes[i].y1;
                totalBoxes[i].x2 = img_W - totalBoxes[i].x2;
                totalBoxes[i].y2 = img_H - totalBoxes[i].y2;
                for(int k = 0; k < 5; k++)
                {
                    totalBoxes[i].points_x[k] = img_W - totalBoxes[i].points_x[k];
                    totalBoxes[i].points_y[k] = img_H - totalBoxes[i].points_y[k];
                }
            }
        }
        return totalBoxes;
    }
	
	
	cv::Mat AlignTransformation(std::vector<cv::Point2f> ref_shape, std::vector<cv::Point2f>& src_shape) 
	{
		double X1 = 0, Y1 = 0, X2 = 0, Y2 = 0, Z = 0, C1 = 0, C2 = 0;
		double x1, y1, x2, y2;
		for (int k=0; k<5; k++) {
			x1 = ref_shape[k].x;
			y1 = ref_shape[k].y;
			x2 = src_shape[k].x;
			y2 = src_shape[k].y;

			Z += x2 * x2 + y2 * y2;
			X1 += x1;
			Y1 += y1;
			X2 += x2;
			Y2 += y2;
			C1 += x1 * x2 + y1 * y2;
			C2 += y1 * x2 - x1 * y2;
		}

		cv::Mat A = (cv::Mat_<double>(4,4) << X2, -Y2, 5, 0, Y2, X2, 0, 5, Z, 0, X2, Y2, 0, Z, -Y2, X2);
		cv::Mat B = (cv::Mat_<double>(4,1) << X1, Y1, C1, C2);
		cv::Mat soln;
		//cout << A << endl;
		//cout << B << endl;
		cv::solve(A, B, soln, cv::DECOMP_SVD);
		cv::Mat affine_mat = (cv::Mat_<double>(2,3) << soln.at<double>(0,0), -soln.at<double>(1,0), soln.at<double>(2,0), 
												soln.at<double>(1,0), soln.at<double>(0,0), soln.at<double>(3,0));
		//cout << affine_mat << endl;
		return affine_mat;
	}

	bool MTCNN::alignFace(const cv::Mat image, cv::Mat & alignedImg, BoundingBox box)
	{
		std::vector<cv::Point2f> m_dst_ref_shape;
		cv::Size input_geometry_ = cv::Size(112, 112);
		
		m_dst_ref_shape.push_back(cv::Point2f(30.2946f+8.0, 51.6963f));
		m_dst_ref_shape.push_back(cv::Point2f(65.5318f+8.0, 51.5014f));
		m_dst_ref_shape.push_back(cv::Point2f(48.0252f+8.0, 71.7366f));
		m_dst_ref_shape.push_back(cv::Point2f(33.5493f+8.0, 92.3655f));
		m_dst_ref_shape.push_back(cv::Point2f(62.7299f+8.0, 92.2041f));
		
			
		std::vector<cv::Point2f> src_shape;
		for (int i = 0; i < pts_num; i++) 
		{
			src_shape.push_back(cv::Point2f(box.points_x[i], box.points_y[i]));
			//cout << points[i].x << points[i].y << endl;
		}
			
		cv::Mat affine_mat = AlignTransformation(m_dst_ref_shape, src_shape);
		cv::warpAffine(image, alignedImg, affine_mat, input_geometry_);
		return true;
		
	}
}



