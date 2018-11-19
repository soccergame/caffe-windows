#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>
#include <random>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    inline void GetRadialXY(float x, float y, float cx, float cy, float k,
        float xscale, float yscale, float xshift, float yshift, float &rx, float &ry) {
        x = x * xscale + xshift;
        y = y * yscale + yshift;
        const float dx = x - cx;
        const float dy = y - cy;
        const float temp = k * (dx * dx + dy * dy);
        rx = x + dx * temp;
        ry = y + dy * temp;
    }

    inline float CalcShift(float x1, float x2, float cx, float k, float thresh = 1) {
        float x3 = x1 + (x2 - x1) * 0.5f;
        float res1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
        float res3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

        if (res1 > -thresh && res1 < thresh)
            return x1;
        if (res3 < 0) {
            return CalcShift(x3, x2, cx, k);
        }
        else {
            return CalcShift(x1, x3, cx, k);
        }
    }

    template<typename Dtype>
    void FishEyeDistort(const Dtype *data, int width, int height, int channels, float K, float centerX, float centerY, Dtype *dst_data) {
        float xshift = CalcShift(0, centerX - 1, centerX, K);
        float newcenterX = width - centerX;
        float xshift_2 = CalcShift(0, newcenterX - 1, newcenterX, K);

        float yshift = CalcShift(0, centerY - 1, centerY, K);
        float newcenterY = height - centerY;
        float yshift_2 = CalcShift(0, newcenterY - 1, newcenterY, K);
        float xscale = (width - xshift - xshift_2) / width;
        float yscale = (height - yshift - yshift_2) / height;

        //const int channels = 1;	
        const int width_step = width * channels;
        for (int j = 0; j < height; ++j) {
            const int j_mul_w = j * width_step;
            for (int i = 0; i < width; ++i) {
                float idx0, idx1;
                GetRadialXY((float)i, (float)j, centerX, centerY, K, xscale, yscale, xshift, yshift, idx1, idx0);
                if (idx0 < 0 || idx1 < 0 || idx0 > height - 1 || idx1 > width - 1) {
                    for (int k = 0; k < channels; ++k) {
                        dst_data[j_mul_w + i * channels + k] = 0;
                    }
                }
                else {
                    const int idx0_fl = static_cast<int>(floor(idx0));
                    const int idx0_cl = static_cast<int>(ceil(idx0));
                    const int idx1_fl = static_cast<int>(floor(idx1));
                    const int idx1_cl = static_cast<int>(ceil(idx1));

                    const float x = idx0 - idx0_fl;
                    const float y = idx1 - idx1_fl;

                    for (int k = 0; k < channels; ++k) {
                        const float s1 = data[idx0_fl * width_step + idx1_fl * channels + k];
                        const float s2 = data[idx0_fl * width_step + idx1_cl * channels + k];
                        const float s3 = data[idx0_cl * width_step + idx1_cl * channels + k];
                        const float s4 = data[idx0_cl * width_step + idx1_fl * channels + k];

                        const float res = (1 - x) * (s1 * (1 - y) + s2 * y) + x * (s3 * y + s4 * (1 - y));
                        dst_data[j_mul_w + i * channels + k] = static_cast<Dtype>(res);
                    }
                }
            }
        }
    }

#ifdef USE_OPENCV
    void FishEyeDistort(const cv::Mat &src, float K, float centerX, float centerY, cv::Mat &dst) {
        int width = src.cols;
        int height = src.rows;
        const int channels = src.channels();
        const uchar *data = src.data;
        uchar *dst_data = dst.data;
        FishEyeDistort(data, width, height, channels, K, centerX, centerY, dst_data);
    }
#endif  // USE_OPENCV

    template<typename Dtype>
    DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
        Phase phase)
        : param_(param), phase_(phase) {
        // resize images
        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        int new_height = param_.new_height();
        int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            new_height = new_size;
            new_width = new_size;
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
        }

        // check if we want to use mean_file
        if (param_.has_mean_file()) {
            CHECK_EQ(param_.mean_value_size(), 0) <<
                "Cannot specify mean_file and mean_value at the same time";
            const string& mean_file = param.mean_file();
            if (Caffe::root_solver()) {
                LOG(INFO) << "Loading mean file from: " << mean_file;
            }
            BlobProto blob_proto;
            ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
            data_mean_.FromProto(blob_proto);

            if (new_height > 0) {
                std::vector<int> shape = data_mean_.shape();
                shape[1] = new_height;
                shape[2] = new_width;
                data_mean_resized_.Reshape(shape);

                const Dtype* mean = data_mean_.cpu_data();
                Dtype * mean_resized = data_mean_resized_.mutable_cpu_data();
                ResizeDatum(data_mean_.width(), new_width, data_mean_.height(),
                    new_height, data_mean_.channels(), mean, mean_resized);
            }
        }
        // check if we want to use mean_value
        if (param_.mean_value_size() > 0) {
            CHECK(param_.has_mean_file() == false) <<
                "Cannot specify mean_file and mean_value at the same time";
            for (int c = 0; c < param_.mean_value_size(); ++c) {
                mean_values_.push_back(param_.mean_value(c));
            }
        }
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::DataAugmentation(const Datum& datum, Dtype *transformed_data) {
        const int input_channels = datum.channels();
        const int input_height = datum.height();
        const int input_width = datum.width();
        const int input_length = input_height * input_width;
        const int input_size = input_channels * input_length;
        const string& data = datum.data();
        const bool has_uint8 = data.size() > 0;

        // 无论如何，需先将Datum变量换为Dtype变量
        shared_ptr<Dtype> inter_data(new Dtype[input_size]);
        Dtype *interData = inter_data.get();// new Dtype[input_size];
        for (int index = 0; index < input_size; ++index) {
            if (has_uint8) {
                interData[index] = static_cast<Dtype>(static_cast<uint8_t>(data[index]));
            }
            else {
                interData[index] = datum.float_data(index);
            }
        }
        DataAugmentation(interData, transformed_data, 1, input_channels, input_height, input_width);

        //         if (interData)
        //             delete[] interData;
        //         interData = 0;
    }

    template <typename Dtype>
    void DataTransformer<Dtype>::ResizeDatum(int datum_width, int new_width, int datum_height, int new_height,
        int datum_channels, const Dtype *inter_data, Dtype *resize_data) const
    {
        float scale_x = static_cast<float>(datum_width) / new_width;
        float scale_y = static_cast<float>(datum_height) / new_height;

        for (int c = 0; c < datum_channels; ++c) {
            for (int y = 0; y < new_height; ++y) {
                float ori_y = scale_y * y;
                int uy = floor(ori_y);
                float cof_y = ori_y - uy;

                int offset = (c * new_height + y) * new_width;
                int ori_offset = (c * datum_height + uy) * datum_width;
                for (int x = 0; x < new_width; ++x) {
                    float ori_x = scale_x * x;
                    int ux = floor(ori_x);
                    float cof_x = ori_x - ux;
                    if (uy >= 0 && uy < datum_height - 1
                        && ux >= 0 && ux < datum_width - 1) {
                        float ans_x = (1.0f - cof_x) * inter_data[ori_offset + ux] +
                            cof_x * inter_data[ori_offset + ux + 1];
                        float ans_x_1 = (1.0f - cof_x) * inter_data[ori_offset + datum_width + ux] +
                            cof_x * inter_data[ori_offset + datum_width + ux + 1];
                        float ans = (1.0f - cof_y) * ans_x + cof_y * ans_x_1;
                        resize_data[offset + x] = ans;
                    }
                    else {
                        ux = std::max(0, ux);
                        ux = std::min(ux, datum_width - 1);
                        uy = std::max(0, uy);
                        uy = std::min(uy, datum_height - 1);
                        resize_data[offset + x] = inter_data[(c * datum_height + uy) * datum_width + ux];
                    }
                }
            }
        }
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::Resize(int crop_h, int crop_w, 
        int datum_channels, int &datum_height,
        int &datum_width, int &datum_length, shared_ptr<Dtype> &interData)
    {
        // resize images
        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        int new_height = param_.new_height();
        int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
            new_height = new_size;
            new_width = new_size;
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
        }
        else {
            CHECK_GE(datum_height, crop_h);
            CHECK_GE(datum_width, crop_w);
        }
        // resize image
        if (new_width > 0) {
            int new_length = new_width * new_height * datum_channels;
            shared_ptr<Dtype> resizeData(new Dtype[new_length]);
            Dtype *resize_data = resizeData.get();

            Dtype *inter_data = interData.get();
            ResizeDatum(datum_width, new_width, datum_height, new_height, datum_channels, inter_data, resize_data);

            // assign resized value to datum
            datum_height = new_height;
            datum_width = new_width;
            datum_length = new_length;
            interData = resizeData;

            if (param_.has_mean_file()) {
                CHECK_EQ(datum_channels, data_mean_resized_.channels());
                CHECK_EQ(datum_height, data_mean_resized_.height());
                CHECK_EQ(datum_width, data_mean_resized_.width());
            }
        }
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const Datum& datum,
        Dtype* transformed_data) {
        //const string& data = datum.data();
        const int datum_channels = datum.channels();
        int datum_height = datum.height();
        int datum_width = datum.width();
        int datum_length = datum_channels * datum_height * datum_width;

        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
            crop_w = datum_width;
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
            crop_h = datum_height;
        }
        const Dtype scale = param_.scale();
        const bool do_mirror = param_.mirror() && Rand(2);
        const bool has_mean_file = param_.has_mean_file();
        //const bool has_uint8 = data.size() > 0;
        const bool has_mean_values = mean_values_.size() > 0;
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width();
        const int margin = param_.margin();
        CHECK_GE(margin, 0) << "margin must be greater than or equal to 0";
        CHECK_LE(margin, (datum_height - crop_h) / 2) << "margin must be not greater than (datum_height - crop_size) / 2";
        CHECK_LE(margin, (datum_width - crop_w) / 2) << "margin must be not greater than (datum_width - crop_size) / 2";

        CHECK_GT(datum_channels, 0);
        CHECK_GE(datum_height, crop_h);
        CHECK_GE(datum_width, crop_w);
        CHECK_EQ(param_.patch_center_x_size(), param_.patch_center_y_size());

        shared_ptr<Dtype> interData(new Dtype[datum_length]);
        Dtype *inter_data = interData.get();
        DataAugmentation(datum, inter_data);

        Dtype* mean = NULL;
        if (has_mean_file) {
            CHECK_EQ(datum_channels, data_mean_.channels());
            CHECK_EQ(datum_height, data_mean_.height());
            CHECK_EQ(datum_width, data_mean_.width());
            mean = data_mean_.mutable_cpu_data();
        }
        if (has_mean_values) {
            CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
                "Specify either 1 mean_value or as many as channels: " << datum_channels;
            if (datum_channels > 1 && mean_values_.size() == 1) {
                // Replicate the mean_value for simplicity
                for (int c = 1; c < datum_channels; ++c) {
                    mean_values_.push_back(mean_values_[0]);
                }
            }
        }

        // resize image
        Resize(crop_h, crop_w, datum_channels, datum_height, 
            datum_width, datum_length, interData);
        inter_data = interData.get();
        const int new_height = param_.new_height();
        const int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0 || new_height > 0 || new_width > 0) {
            if (has_mean_file)
                mean = data_mean_resized_.mutable_cpu_data();
        }

        int height = datum_height;
        int width = datum_width;

        int h_off = 0;
        int w_off = 0;
        std::vector<int> patch_off_h;
        std::vector<int> patch_off_w;
        int patch_num = 0;

        if (crop_h || crop_w) {
            CHECK_GE(crop_h, patch_height);
            CHECK_GE(crop_w, patch_width);
            height = crop_h;
            width = crop_w;
            // We only do random crop when we do training.
            if (phase_ == TRAIN /*&& !param_.center_crop()*/) {
                h_off = Rand(datum_height - crop_h + 1 - 2 * margin) + margin;
                w_off = Rand(datum_width - crop_w + 1 - 2 * margin) + margin;
            }
            else {
                h_off = (datum_height - crop_h) / 2;
                w_off = (datum_width - crop_w) / 2;
            }
        }

        if (patch_height) {
            patch_num = param_.patch_center_x_size();
            height = patch_height;
            width = patch_width;
            for (int p = 0; p < patch_num; ++p) {
                patch_off_w.push_back(param_.patch_center_x(p) - patch_width / 2);
                patch_off_h.push_back(param_.patch_center_y(p) - patch_height / 2);
            }
        }
        else {
            patch_num = 1;
            patch_off_h.push_back(0);
            patch_off_w.push_back(0);
        }
#ifdef USE_ERASE
        const bool do_erase = param_.has_erase_ratio() & (UniformRand(0.0f, 1.0f) < param_.erase_ratio());
        int erase_x_min = width, erase_x_max = -1, erase_y_min = height, erase_y_max = -1;
        if (do_erase) {
            do {
                Dtype erase_scale = UniformRand(param_.scale_min(), param_.scale_max()); // std::uniform_real_distribution<float>(param_.scale_min(), param_.scale_max())(prnd_);
                int erase_width = (float)width * erase_scale;
                Dtype erase_aspect = UniformRand(param_.aspect_min(), param_.aspect_max()); // std::uniform_real_distribution<float>(param_.aspect_min(), param_.aspect_max())(prnd_);
                int erase_height = (float)erase_width * erase_aspect;
                erase_x_min = int(UniformRand(0, float(width)) + 0.5f); //std::uniform_int_distribution<int>(0, width)(prnd_);
                erase_y_min = int(UniformRand(0, float(height)) + 0.5f); //std::uniform_int_distribution<int>(0, height)(prnd_);
                erase_x_max = erase_x_min + erase_width - 1;
                erase_y_max = erase_y_min + erase_height - 1;
            } while (erase_x_min < 0 || erase_y_min < 0 || erase_x_max >= width || erase_y_max >= height);
        }
#endif
        Dtype datum_element;
        int top_index, data_index;
        for (int p = 0; p < patch_num; ++p) {
            for (int c = 0; c < datum_channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        if (patch_height) {
                            if (do_mirror)
                                data_index = (c * datum_height + h_off + patch_off_h[p] + h) * datum_width + (datum_width - 1 - (w_off + patch_off_w[p] + w));
                            else
                                data_index = (c * datum_height + h_off + patch_off_h[p] + h) * datum_width + w_off + patch_off_w[p] + w;

                            top_index = ((p*datum_channels + c) * height + h) * width + w;
                        }
                        else {
                            data_index = (c * datum_height + h_off + patch_off_h[p] + h) * datum_width + w_off + patch_off_w[p] + w;
                            if (do_mirror) {
                                top_index = ((p*datum_channels + c) * height + h) * width + (width - 1 - w);
                            }
                            else {
                                top_index = ((p*datum_channels + c) * height + h) * width + w;
                            }
                        }

                        if (data_index < 0 || data_index >= datum_length)
                            continue;
#ifdef USE_ERASE
                        // Fill datum_element into top
                        if (do_erase && w >= erase_x_min && w <= erase_x_max && h >= erase_y_min && h <= erase_y_max) {
                            datum_element = Rand(255);
                        }
                        else {
                            /*if (has_uint8) {
                                datum_element =
                                    static_cast<Dtype>(static_cast<uint8_t>(inter_data[data_index]));
                            }
                            else {
                                datum_element = inter_data[data_index];
                            }*/
                            datum_element = inter_data[data_index];
                        }
#else
                        datum_element = inter_data[data_index];
#endif

                        if (has_mean_file) {
                            transformed_data[top_index] =
                                (datum_element - mean[data_index]) * scale;
                        }
                        else {
                            if (has_mean_values) {
                                transformed_data[top_index] =
                                    (datum_element - mean_values_[c]) * scale;
                            }
                            else {
                                transformed_data[top_index] = datum_element * scale;
                            }
                        }
                    }
                }
            }
        }      
    }


    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const Datum& datum,
        Blob<Dtype>* transformed_blob) {
        // If datum is encoded, decode and transform the cv::image.
        if (datum.encoded()) {
#ifdef USE_OPENCV
            CHECK(!(param_.force_color() && param_.force_gray()))
                << "cannot set both force_color and force_gray";
            cv::Mat cv_img;
            if (param_.force_color() || param_.force_gray()) {
                // If force_color then decode in color otherwise decode in gray.
                cv_img = DecodeDatumToCVMat(datum, param_.force_color());
            }
            else {
                cv_img = DecodeDatumToCVMatNative(datum);
            }
            // Transform the cv::image into blob.
            return Transform(cv_img, transformed_blob);
#else
            LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
        }
        else {
            if (param_.force_color() || param_.force_gray()) {
                LOG(ERROR) << "force_color and force_gray only for encoded datum";
            }
        }

        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
            crop_w = datum_width;
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
            crop_h = datum_height;
        }
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width();

        // Check dimensions.
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        const int num = transformed_blob->num();

        CHECK_GE(channels, datum_channels);
        CHECK_LE(height, datum_height);
        CHECK_LE(width, datum_width);
        CHECK_GE(num, 1);
        CHECK_EQ(param_.patch_center_x_size(), param_.patch_center_y_size());

        if (crop_h || crop_w) {
            CHECK_GE(crop_h, height);
            CHECK_GE(crop_w, width);
            CHECK_GE(crop_h, patch_height);
            CHECK_GE(crop_w, patch_width);
        }
        else {
            CHECK_GE(datum_height, height);
            CHECK_GE(datum_width, width);
        }

        // resize images
        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        int new_height = param_.new_height();
        int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
            new_height = new_size;
            new_width = new_size;
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
        }

        if (new_height > 0)
        {
            CHECK_GE(new_height, height);
            CHECK_GE(new_width, width);
            CHECK_GE(new_height, patch_height);
            CHECK_GE(new_width, patch_width);
        }

        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        Transform(datum, transformed_data);
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
        Blob<Dtype>* transformed_blob) {
        const int datum_num = datum_vector.size();
        const int num = transformed_blob->num();
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();

        CHECK_GT(datum_num, 0) << "There is no datum to add";
        CHECK_LE(datum_num, num) <<
            "The size of datum_vector must be no greater than transformed_blob->num()";
        Blob<Dtype> uni_blob(1, channels, height, width);
        for (int item_id = 0; item_id < datum_num; ++item_id) {
            int offset = transformed_blob->offset(item_id);
            uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
            Transform(datum_vector[item_id], &uni_blob);
        }
    }

#ifdef USE_OPENCV
    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
        Blob<Dtype>* transformed_blob/*, bool transpose*/) {
        const int mat_num = mat_vector.size();
        const int num = transformed_blob->num();
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();

        CHECK_GT(mat_num, 0) << "There is no MAT to add";
        CHECK_EQ(mat_num, num) <<
            "The size of mat_vector must be equals to transformed_blob->num()";
        Blob<Dtype> uni_blob(1, channels, height, width);
        for (int item_id = 0; item_id < mat_num; ++item_id) {
            int offset = transformed_blob->offset(item_id);
            uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
            Transform(mat_vector[item_id], &uni_blob/*, transpose*/);
        }
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::DataAugmentation(const cv::Mat& cv_img, cv::Mat &transformed_mat)
    {
        const bool needs_scale_aug = param_.has_scale_factor();
        const bool needs_cover_aug = param_.has_cover_size();
        const bool needs_gaussian_aug = param_.has_gaussian_para();
        const bool needs_roll_aug = (param_.roll_angle() != 0);
        const bool needs_color_aug = param_.color_augmentation();
        const bool needs_adjustment_aug = param_.has_adjustment_para();
        const bool needs_fisheye_aug = param_.has_fisheye_param();

        transformed_mat = cv_img.clone();

        cv::Mat inter_mat = cv_img.clone();

        // fisheye distortion augmentation
        if (needs_fisheye_aug && Rand(2))
        {
            // minimum distortion ratio
            const float min_distort_ratio = param_.fisheye_param().min_distort_ratio();
            // maximum distortion ratio
            const float max_distort_ratio = param_.fisheye_param().max_distort_ratio();
            const float mul_factor = 100000.0f;

            const int shift_x = inter_mat.cols / 8;
            const int shift_y = inter_mat.rows / 8;
            const int center_x = param_.fisheye_param().center_x() + Rand(shift_x + 1) - shift_x / 2;
            const int center_y = param_.fisheye_param().center_y() + Rand(shift_y + 1) - shift_y / 2;

            CHECK_GE(min_distort_ratio, 0.00001f) << "distortion ratio must be larger than 0.00001!";
            CHECK_LE(max_distort_ratio, 0.0001f) << "distortion ratio must be smaller than 0.0001!";
            CHECK_LE(min_distort_ratio, max_distort_ratio) << "minimum distortion ratio must be less than maximum distortion ratio";

            float distort_ratio = Rand(int((max_distort_ratio - min_distort_ratio) *  mul_factor + 0.5) + 1) / mul_factor + min_distort_ratio;
            FishEyeDistort(inter_mat, distort_ratio, center_x, center_y, transformed_mat);
            inter_mat = transformed_mat.clone();
        }

        if (needs_color_aug && 3 == inter_mat.channels())
        {
            std::vector<cv::Mat> mv;
            cv::split(inter_mat, mv);
            const int color_mode = Rand(5);		// 0: color, 1: blue, 2: green, 3: red, 4: gray
            if (1 == color_mode)		// blue
            {
                mv[1] = mv[0];
                mv[2] = mv[0];
                cv::merge(mv, transformed_mat);
            }
            else if (2 == color_mode)	// green
            {
                mv[0] = mv[1];
                mv[2] = mv[1];
                cv::merge(mv, transformed_mat);
            }
            else if (3 == color_mode)	// red
            {
                mv[0] = mv[2];
                mv[1] = mv[2];
                cv::merge(mv, transformed_mat);
            }
            else if (4 == color_mode)  // gray
            {
                cv::cvtColor(inter_mat, mv[0], cv::COLOR_BGR2GRAY);
                mv[1] = mv[0];
                mv[2] = mv[0];
                cv::merge(mv, transformed_mat);
            }
            inter_mat = transformed_mat.clone();
        }

        double scale_factor = 1.0;
        double roll_angle = 0;
        bool flag = false;
        if (needs_scale_aug && Rand(2))
        {
            // Scale precision is 1e-2;
            const double max_scale_factor = param_.scale_factor().max_factor();
            const double min_scale_factor = param_.scale_factor().min_factor();

            CHECK_GT(min_scale_factor, 0) << "Scale factor must be bigger than 0!";
            CHECK_GT(max_scale_factor, 0) << "Scale factor must be bigger than 0!";

            if (1 != max_scale_factor || 1 != min_scale_factor)
            {
                scale_factor = 0.01 * Rand(int(max_scale_factor * 100.0 + 0.5) - int(min_scale_factor * 100.0 + 0.5) + 1) + min_scale_factor;
                flag = true;
            }
        }

        if (needs_roll_aug && Rand(2))
        {
            // roll precision is 1e-2;
            CHECK_GE(param_.roll_angle(), 0) << "Scale factor must be bigger than 0 or equal!";
            roll_angle = Rand(int(200 * param_.roll_angle() + 0.5) + 1) / 100.0 - param_.roll_angle();
            if (0 != roll_angle)
                flag = true;
        }

        if (flag)
        {
            cv::Point center(inter_mat.cols / 2, inter_mat.rows / 2);
            cv::Mat rot_mat(2, 3, CV_32FC1);
            rot_mat = cv::getRotationMatrix2D(center, roll_angle, scale_factor);
            cv::warpAffine(inter_mat, transformed_mat, rot_mat, inter_mat.size(), 1, 0, cv::Scalar(param_.filled_pixels()));
        }

        if (needs_gaussian_aug && Rand(2))
        {
            const int nchannels = transformed_mat.channels();
            float mean_value = param_.gaussian_para().mean_value();
            float variance_value = param_.gaussian_para().variance_value();
            Dtype gaussian_value;
            for (int i = 0; i < transformed_mat.rows; i++) {
                for (int j = 0; j < transformed_mat.cols; j++) {
                    gaussian_value = NormalRand(mean_value, variance_value);
                    for (int k = 0; k < nchannels; ++k)
                        transformed_mat.at<unsigned char>(i, j * nchannels + k) =
                        cv::saturate_cast<unsigned char>(Dtype(transformed_mat.at<unsigned char>(i, j * nchannels + k)) + gaussian_value);
                }
            }
        }

        if (needs_cover_aug && Rand(2))
        {
            const int max_cover_size = param_.cover_size().max_size();
            const int min_cover_size = param_.cover_size().min_size();
            const int cover_size = max_cover_size - min_cover_size + 1;
            const int block_w = Rand(cover_size) + min_cover_size;
            const int block_h = Rand(cover_size) + min_cover_size;
            const int block_pos_x = Rand(transformed_mat.cols - block_w + 1);
            const int block_pos_y = Rand(transformed_mat.rows - block_h + 1);

            transformed_mat(cv::Rect(block_pos_x, block_pos_y, block_w, block_h)).setTo(cv::Scalar(param_.filled_pixels(), param_.filled_pixels(), param_.filled_pixels()));
        }

        if (needs_adjustment_aug/* && Rand(2)*/)
        {
            const int nchannels = transformed_mat.channels();
            float max_brightness = param_.adjustment_para().max_brightness();
            float max_hue = param_.adjustment_para().max_hue();
            float min_saturation = param_.adjustment_para().min_saturation();
            float max_saturation = param_.adjustment_para().max_saturation();
            float min_contrast = param_.adjustment_para().min_contrast();
            float max_contrast = param_.adjustment_para().max_contrast();

            CHECK_GE(max_hue, 0) << "Max hue para must be bigger than 0!";
            CHECK_LE(max_hue, 180) << "Max hue para must be less than 180!";

            std::vector<int> adjustment_alg;
            if (0 != max_brightness)
                adjustment_alg.push_back(1);
            if (0 != max_hue)
                adjustment_alg.push_back(2);
            if (min_saturation < max_saturation && 0 <= max_saturation && 0 <= min_saturation)
                adjustment_alg.push_back(3);
            if (min_contrast < max_contrast && 0 <= max_contrast && 0 <= min_contrast)
                adjustment_alg.push_back(4);

            if (0 != adjustment_alg.size())
                shuffle(adjustment_alg.begin(), adjustment_alg.end());

            for (int i = 0; i < adjustment_alg.size(); ++i)
            {
                if (1 == adjustment_alg[i]) // 亮度随机扰动
                {
                    if (max_brightness > 255.0f)
                        max_brightness = 255.0f;
                    Dtype brightness_value = UniformRand(-max_brightness, max_brightness);
                    for (int i = 0; i < transformed_mat.rows; i++) {
                        for (int j = 0; j < transformed_mat.cols; j++) {
                            for (int k = 0; k < nchannels; ++k)
                                transformed_mat.at<unsigned char>(i, j * nchannels + k) =
                                cv::saturate_cast<unsigned char>(Dtype(transformed_mat.at<unsigned char>(i, j * nchannels + k)) + brightness_value);
                        }
                    }
                }

                if (2 == adjustment_alg[i])
                {
                    // 调整的色调范围在0~180度之间
                    if (max_hue > 180.0f)
                        max_hue = 180.0f;
                    Dtype hue_value = UniformRand(-max_hue, max_hue);
                    Dtype value;
                    cv::cvtColor(transformed_mat, inter_mat, CV_BGR2HSV);
                    for (int i = 0; i < inter_mat.rows; i++) {
                        for (int j = 0; j < inter_mat.cols; j++) {
                            // 仅改变H通道
                            value = Dtype(inter_mat.at<unsigned char>(i, j * nchannels)) + hue_value;
                            if (value > 180.0f)
                                value -= 180.0f;
                            else if (value < 0)
                                value += 180.0f;

                            inter_mat.at<unsigned char>(i, j * nchannels) = cv::saturate_cast<unsigned char>(value);

                        }
                    }
                    cv::cvtColor(inter_mat, transformed_mat, CV_HSV2BGR);
                }

                if (3 == adjustment_alg[i])
                {
                    if (max_saturation > 2.0f)
                        max_saturation = 2.0f;
                    Dtype saturation_value = UniformRand(min_saturation, max_saturation);
                    cv::cvtColor(transformed_mat, inter_mat, CV_BGR2HSV);
                    for (int i = 0; i < inter_mat.rows; i++) {
                        for (int j = 0; j < inter_mat.cols; j++) {
                            // 仅改变S通道
                            inter_mat.at<unsigned char>(i, j * nchannels + 1) =
                                cv::saturate_cast<unsigned char>(Dtype(inter_mat.at<unsigned char>(i, j * nchannels + 1)) * saturation_value);
                        }
                    }
                    cv::cvtColor(inter_mat, transformed_mat, CV_HSV2BGR);
                }

                if (4 == adjustment_alg[i])
                {
                    if (max_contrast > 2.0f)
                        max_contrast = 2.0f;
                    Dtype contrast_value = UniformRand(min_contrast, max_contrast);
                    for (int i = 0; i < transformed_mat.rows; i++) {
                        for (int j = 0; j < transformed_mat.cols; j++) {
                            for (int k = 0; k < nchannels; ++k)
                                transformed_mat.at<unsigned char>(i, j * nchannels + k) =
                                cv::saturate_cast<unsigned char>(Dtype(transformed_mat.at<unsigned char>(i, j * nchannels + k)) * contrast_value);
                        }
                    }
                }
            } // for (int i = 0; i < adjustment_alg.size(); ++i)
        } // if (needs_adjustment_aug && Rand(2))
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
        Blob<Dtype>* transformed_blob/*, bool transpose*/) {
        const int img_channels = cv_img.channels();
        int img_height = cv_img.rows;
        int img_width = cv_img.cols;
        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
            crop_w = img_width;
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
            crop_h = img_height;
        }

        // Check dimensions.
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        const int num = transformed_blob->num();

        CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

        CHECK_EQ(channels, img_channels);
        CHECK_LE(height, img_height);
        CHECK_LE(width, img_width);
        /*if (transpose) {
            CHECK_LE(height, img_width);
            CHECK_LE(width, img_height);
        }
        else {
            CHECK_LE(height, img_height);
            CHECK_LE(width, img_width);
        }*/
        CHECK_GE(num, 1);
        //if (transpose) {
        //  std::swap(height, width);
        //}

        //CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

        const Dtype scale = param_.scale();
        const bool do_mirror = param_.mirror() && Rand(2);
        const bool has_mean_file = param_.has_mean_file();
        const bool has_mean_values = mean_values_.size() > 0;
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width();
        const int margin = param_.margin();

        // resize images
        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        int new_height = param_.new_height();
        int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
            new_height = new_size;
            new_width = new_size;
            img_height = new_height;
            img_width = new_width;
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
            img_height = new_height;
            img_width = new_width;
        }
        else {
            CHECK_GE(img_height, crop_h);
            CHECK_GE(img_width, crop_w);
        }

        CHECK_GE(margin, 0) << "margin must be greater than or equal to 0";
        CHECK_LE(margin, (img_height - crop_h) / 2) << "margin must be not greater than (img_height - crop_size) / 2";
        CHECK_LE(margin, (img_width - crop_w) / 2) << "margin must be not greater than (img_width - crop_size) / 2";

        CHECK_GE(channels, img_channels);
        CHECK_GT(img_channels, 0);
        CHECK_GE(img_height, crop_h);
        CHECK_GE(img_width, crop_w);
        CHECK_GE(img_height, patch_height);
        CHECK_GE(img_width, patch_width);
        CHECK_EQ(param_.patch_center_x_size(), param_.patch_center_y_size());

        CHECK_LE(height, img_height);
        CHECK_LE(width, img_width);
        CHECK_GE(num, 1);

        cv::Mat trans_img;
        DataAugmentation(cv_img, trans_img);

        Dtype* mean = NULL;
        if (has_mean_file) {
            CHECK_EQ(img_channels, data_mean_.channels());
            CHECK_EQ(img_height, data_mean_.height());
            CHECK_EQ(img_width, data_mean_.width());
            mean = data_mean_.mutable_cpu_data();
        }
        if (has_mean_values) {
            CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
                "Specify either 1 mean_value or as many as channels: " << img_channels;
            if (img_channels > 1 && mean_values_.size() == 1) {
                // Replicate the mean_value for simplicity
                for (int c = 1; c < img_channels; ++c) {
                    mean_values_.push_back(mean_values_[0]);
                }
            }
        }

        // resize image
        cv::Mat resized_img;
        if (new_width > 0) {
            cv::resize(trans_img, resized_img, cv::Size(new_width, new_height));

            // assign resized value to datum
            trans_img = resized_img;

            if (has_mean_file) {
                CHECK_EQ(img_channels, data_mean_resized_.channels());
                CHECK_EQ(img_height, data_mean_resized_.height());
                CHECK_EQ(img_width, data_mean_resized_.width());
                mean = data_mean_resized_.mutable_cpu_data();
            }

            CHECK_GE(channels, img_channels);
            CHECK_GT(img_channels, 0);
            CHECK_GE(img_height, crop_h);
            CHECK_GE(img_width, crop_w);
            CHECK_GE(img_height, patch_height);
            CHECK_GE(img_width, patch_width);

            CHECK_LE(height, img_height);
            CHECK_LE(width, img_width);
        }

        int h_off = 0;
        int w_off = 0;
        //cv::Mat cv_cropped_img = cv_img;
        if (crop_h || crop_w) {
            CHECK_GE(crop_h, height);
            CHECK_GE(crop_w, width);
            // We only do random crop when we do training.
            if (phase_ == TRAIN /*&& !param_.center_crop()*/) {
                h_off = Rand(img_height - crop_h + 1 - 2 * margin) + margin;
                w_off = Rand(img_width - crop_w + 1 - 2 * margin) + margin;
            }
            else {
                h_off = (img_height - crop_h) / 2;
                w_off = (img_width - crop_w) / 2;
            }
            /*cv::Rect roi(w_off, h_off, crop_w, crop_h);
            cv_cropped_img = cv_img(roi);*/
        }
        /*else {
            if (transpose) {
                CHECK_EQ(img_width, height);
                CHECK_EQ(img_height, width);
            }
            else {
                CHECK_EQ(img_height, height);
                CHECK_EQ(img_width, width);
            }
        }*/

        //std::mt19937 prnd(time(NULL));
#ifdef USE_ERASE
        const bool do_erase = param_.has_erase_ratio() & (UniformRand(0.0f, 1.0f) < param_.erase_ratio());
        int erase_x_min = width, erase_x_max = -1, erase_y_min = height, erase_y_max = -1;
        if (do_erase) {
            do {
                Dtype erase_scale = UniformRand(param_.scale_min(), param_.scale_max()); // std::uniform_real_distribution<float>(param_.scale_min(), param_.scale_max())(prnd_);
                int erase_width = (float)width * erase_scale;
                Dtype erase_aspect = UniformRand(param_.aspect_min(), param_.aspect_max()); // std::uniform_real_distribution<float>(param_.aspect_min(), param_.aspect_max())(prnd_);
                int erase_height = (float)erase_width * erase_aspect;
                erase_x_min = int(UniformRand(0, float(width)) + 0.5f); //std::uniform_int_distribution<int>(0, width)(prnd_);
                erase_y_min = int(UniformRand(0, float(height)) + 0.5f); //std::uniform_int_distribution<int>(0, height)(prnd_);
                erase_x_max = erase_x_min + erase_width - 1;
                erase_y_max = erase_y_min + erase_height - 1;
            } while (erase_x_min < 0 || erase_y_min < 0 || erase_x_max >= width || erase_y_max >= height);
        }
#endif
        std::vector<cv::Mat> cv_patched_imgs;
        std::vector<unsigned char> cv_patched_flag;
        std::vector<int> patch_off_h;
        std::vector<int> patch_off_w;
        if (patch_height) {
            CHECK_EQ(height, patch_height);
            CHECK_EQ(width, patch_width);
            for (int p = 0; p < param_.patch_center_x_size(); ++p) {
                int patch_h_off = h_off + param_.patch_center_y(p) - patch_height / 2;
                int patch_w_off = w_off + param_.patch_center_x(p) - patch_width / 2;
                int patch_height = height;
                int patch_width = width;
                patch_off_h.push_back(patch_h_off);
                patch_off_w.push_back(patch_w_off);
                cv_patched_flag.push_back(0);
                if (patch_h_off < 0) {
                    patch_height = patch_height + patch_h_off;
                    patch_h_off = 0;
                    cv_patched_flag[0] = cv_patched_flag[0] + 1;
                }

                if (patch_w_off < 0) {
                    patch_width = patch_width + patch_w_off;
                    patch_w_off = 0;
                    cv_patched_flag[0] = cv_patched_flag[0] + 10;
                }
                if (patch_h_off + patch_height >= img_height) {
                    patch_height = img_height - patch_h_off;
                    cv_patched_flag[0] = cv_patched_flag[0] + 2;
                }
                if (patch_w_off + patch_width >= img_width) {
                    patch_width = img_width - patch_w_off;
                    cv_patched_flag[0] = cv_patched_flag[0] + 20;
                }
                cv::Rect roi(patch_w_off, patch_h_off, patch_width, patch_height);
                cv_patched_imgs.push_back(trans_img(roi));
                CHECK(cv_patched_imgs.at(p).data);
            }
        }
        else {
            cv::Rect roi(w_off, h_off, width, height);
            cv_patched_imgs.push_back(trans_img(roi));
            CHECK(cv_patched_imgs.at(0).data);
            cv_patched_flag.push_back(0);
            patch_off_h.push_back(0);
            patch_off_w.push_back(0);
        }


        //CHECK(cv_cropped_img.data);
        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        //bool is_float_data = cv_cropped_img.depth() == CV_32F;
        int top_index;
        int patch_num = cv_patched_imgs.size();
        CHECK_EQ(channels, img_channels * patch_num);
        //if (transpose) {
        //    for (int w = 0; w < width; ++w) {
        //        const uchar* ptr = cv_cropped_img.ptr<uchar>(w);
        //        const float* float_ptr = cv_cropped_img.ptr<float>(w);
        //        int img_index = 0;
        //        for (int h = 0; h < height; ++h) {
        //            for (int c = img_channels - 1; c >= 0; --c) {
        //                if (do_mirror) {
        //                    top_index = (c * height + h) * width + (width - 1 - w);
        //                }
        //                else {
        //                    top_index = (c * height + h) * width + w;
        //                }
        //                // int top_index = (c * height + h) * width + w;
        //                Dtype pixel;
        //                if (do_erase && w >= erase_x_min && w <= erase_x_max && h >= erase_y_min && h <= erase_y_max) {
        //                    pixel = Rand(255);
        //                }
        //                else {
        //                    pixel = static_cast<Dtype>(is_float_data ? float_ptr[img_index] : ptr[img_index]);
        //                }
        //                img_index++;
        //                if (has_mean_file) {
        //                    int mean_index;
        //                    mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
        //                    transformed_data[top_index] =
        //                        (pixel - mean[mean_index]) * scale;
        //                }
        //                else {
        //                    if (has_mean_values) {
        //                        transformed_data[top_index] =
        //                            (pixel - mean_values_[c]) * scale;
        //                    }
        //                    else {
        //                        transformed_data[top_index] = pixel * scale;
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}
        //else {
        for (int p = 0; p < patch_num; ++p) {
            for (int h = 0; h < height; ++h) {
                /*const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
                const float* float_ptr = cv_cropped_img.ptr<float>(h);*/
                const uchar* ptr = cv_patched_imgs.at(p).ptr<uchar>(h);
                int img_index = 0;
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < img_channels; ++c) {
                        if (do_mirror) {
                            top_index = ((p*img_channels + c) * height + h) * width + (width - 1 - w);
                            //top_index = (c * height + h) * width + (width - 1 - w);
                        }
                        else {
                            top_index = ((p*img_channels + c) * height + h) * width + w;
                            //top_index = (c * height + h) * width + w;
                        }
                        if (1 == cv_patched_flag.at(p) / 10 && w < (width - cv_patched_imgs.at(p).cols))
                            continue;
                        if (1 == (cv_patched_flag.at(p) % 10) && h < (height - cv_patched_imgs.at(p).rows))
                            continue;
                        if (2 == cv_patched_flag.at(p) / 10 && w >= cv_patched_imgs.at(p).cols)
                            continue;
                        if (2 == (cv_patched_flag.at(p) % 10) && h >= cv_patched_imgs.at(p).rows)
                            continue;
                        // int top_index = (c * height + h) * width + w;
#ifdef USE_ERASE
                        Dtype pixel;
                        if (do_erase && w >= erase_x_min && w <= erase_x_max && h >= erase_y_min && h <= erase_y_max) {
                            pixel = Rand(255);
                        }
                        else {
                            pixel = static_cast<Dtype>(ptr[img_index]);
                            //pixel = static_cast<Dtype>(is_float_data ? float_ptr[img_index] : ptr[img_index]);
                        }
                        img_index++;
#else
                        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
#endif
                        
                        if (has_mean_file) {
                            int mean_index;
                            mean_index = (c * img_height + patch_off_h[p] + h) * img_width + patch_off_w[p] + w;//(c * img_height + h_off + h) * img_width + w_off + w;
                            transformed_data[top_index] =
                                (pixel - mean[mean_index]) * scale;
                        }
                        else {
                            if (has_mean_values) {
                                transformed_data[top_index] =
                                    (pixel - mean_values_[c]) * scale;
                            }
                            else {
                                transformed_data[top_index] = pixel * scale;
                            }
                        }
                    }
                }
            }
        }
        //}
    }
#endif  // USE_OPENCV

    template<typename Dtype>
    void DataTransformer<Dtype>::DataAugmentation(const Dtype *input_data, Dtype *transformed_data,
        const int input_num, const int input_channels, const int input_height, const int input_width)
    {
        const int input_length = input_height * input_width;
        const int input_size = input_channels * input_length;
        const int input_count = input_num * input_size;

        const float filling_pixel = param_.filled_pixels();
        const bool needs_scale_aug = param_.has_scale_factor();
        const bool needs_cover_aug = param_.has_cover_size();
        const bool needs_gaussian_aug = param_.has_gaussian_para();
        const bool needs_roll_aug = (param_.roll_angle() != 0);
        const bool needs_color_aug = param_.color_augmentation();
        const bool needs_adjustment_aug = param_.has_adjustment_para();
        const float center_w = 0.5f * (input_width - 0);
        const float center_h = 0.5f * (input_height - 0);
        const float EPS = 0.000001f;
        const float PI = 3.1415926f;

        const bool needs_fisheye_aug = param_.has_fisheye_param();

        shared_ptr<Dtype> interData(new Dtype[input_count]);
        Dtype *inter_data = interData.get();// new Dtype[input_count];

        caffe_copy(input_count, input_data, transformed_data);
        caffe_copy(input_count, transformed_data, inter_data);

        // fisheye distortion augmentation
        if (needs_fisheye_aug && Rand(2)) {
            // minimum distortion ratio
            const float min_distort_ratio = param_.fisheye_param().min_distort_ratio();
            // maximum distortion ratio
            const float max_distort_ratio = param_.fisheye_param().max_distort_ratio();
            const float mul_factor = 100000.0f;

            const int shift_x = input_width / 8;
            const int shift_y = input_height / 8;
            const int center_x = param_.fisheye_param().center_x() + Rand(shift_x + 1) - shift_x / 2;
            const int center_y = param_.fisheye_param().center_y() + Rand(shift_y + 1) - shift_y / 2;

            CHECK_GE(min_distort_ratio, 0.00001f) << "distortion ratio must be larger than 0.00001!";
            CHECK_LE(max_distort_ratio, 0.0001f) << "distortion ratio must be smaller than 0.0001!";
            CHECK_LE(min_distort_ratio, max_distort_ratio) << "minimum distortion ratio must be less than maximum distortion ratio";

            float distort_ratio = Rand(int((max_distort_ratio - min_distort_ratio) *  mul_factor + 0.5) + 1) / mul_factor + min_distort_ratio;
            int numStep = 0;
            for (int n = 0; n < input_num; ++n) {
                for (int i = 0; i < input_channels; ++i) {
                    FishEyeDistort(inter_data + numStep + input_length * i, input_width, input_height, 1, distort_ratio,
                        center_x, center_y, transformed_data + numStep + input_length * i);
                }
                numStep += input_size;
            }
            caffe_copy(input_count, transformed_data, inter_data);
        }

        if (needs_color_aug && 3 == input_channels)
        {
            int data_index_b, data_index_g, data_index_r, top_index, numStep = 0, heightStep = 0;
            const int color_mode = Rand(5);	// 0: color, 1: blue, 2: green, 3: red, 4: gray

            for (int n = 0; n < input_num; ++n) {
                if (1 == color_mode) // Blue-channel
                {
                    caffe_copy(input_length, inter_data + numStep, transformed_data + numStep);
                    caffe_copy(input_length, inter_data + numStep, transformed_data + numStep + input_length);
                    caffe_copy(input_length, inter_data + numStep, transformed_data + numStep + input_length * 2);
                }
                else if (2 == color_mode) // Green-channel
                {
                    caffe_copy(input_length, inter_data + numStep + input_length, transformed_data + numStep);
                    caffe_copy(input_length, inter_data + numStep + input_length, transformed_data + numStep + input_length);
                    caffe_copy(input_length, inter_data + numStep + input_length, transformed_data + numStep + input_length * 2);
                }
                else if (3 == color_mode) // Red-channel
                {
                    caffe_copy(input_length, inter_data + numStep + input_length * 2, transformed_data + numStep);
                    caffe_copy(input_length, inter_data + numStep + input_length * 2, transformed_data + numStep + input_length);
                    caffe_copy(input_length, inter_data + numStep + input_length * 2, transformed_data + numStep + input_length * 2);
                }
                else if (4 == color_mode)	// gray image
                {	// gray image
                    heightStep = 0;
                    for (int h = 0; h < input_height; ++h) {
                        for (int w = 0; w < input_width; ++w) {
                            // 默认为BGR模式
                            data_index_b = numStep + heightStep + w;
                            data_index_g = numStep + input_length + heightStep + w;
                            data_index_r = numStep + 2 * input_length + heightStep + w;
                            top_index = numStep + heightStep + w;

                            transformed_data[top_index] = 0.299f*inter_data[data_index_r] + 0.587*inter_data[data_index_g] + 0.114*inter_data[data_index_b];
                            transformed_data[top_index + input_length] = transformed_data[top_index];
                            transformed_data[top_index + input_length * 2] = transformed_data[top_index];
                        }
                        heightStep = heightStep + input_width;
                    }
                }
                else // color
                {
                    // no-op
                }
                numStep += input_size;
            }
            caffe_copy(input_count, transformed_data, inter_data);
        }

        if (needs_scale_aug && 1 == Rand(2))
        {
            // Scale precision is 1e-2;
            const double max_scale_factor = param_.scale_factor().max_factor();
            const double min_scale_factor = param_.scale_factor().min_factor();

            CHECK_GT(min_scale_factor, 0) << "Scale factor must be bigger than 0!";
            CHECK_GT(max_scale_factor, 0) << "Scale factor must be bigger than 0!";
            CHECK_GE(max_scale_factor, min_scale_factor) << "Max factor should be bigger than min";

            if (1 != max_scale_factor || 1 != min_scale_factor)
            {
                double scale_factor = 0.01 * Rand(int(max_scale_factor * 100.0 + 0.5) - int(min_scale_factor * 100.0 + 0.5) + 1) + min_scale_factor;
                // 采用最近邻缩放,而不是双线性插值，缩放效果肯定不好，不知道会不会对训练产生不利的影响
                int data_index, new_h, new_w;
                int top_index = 0, numStep = 0, channelStep = 0;
                for (int n = 0; n < input_num; ++n) {
                    channelStep = 0;
                    for (int c = 0; c < input_channels; ++c) {
                        for (int h = 0; h < input_height; ++h) {
                            for (int w = 0; w < input_width; ++w) {
                                new_h = int((h - center_h)*scale_factor + center_h + 0.5f);
                                new_w = int((w - center_w)*scale_factor + center_w + 0.5f);
                                data_index = numStep + channelStep + new_h * input_width + new_w;

                                if (new_h < 0 || new_h >= input_height || new_w < 0 || new_w >= input_width)
                                    transformed_data[top_index] = filling_pixel;
                                else
                                    transformed_data[top_index] = inter_data[data_index];

                                ++top_index;
                            }
                        }
                        channelStep = channelStep + input_length;
                    }
                    numStep = numStep + input_size;
                }
                caffe_copy(input_count, transformed_data, inter_data);
            }
        }

        if (needs_roll_aug && 1 == Rand(2))
        {
            // roll precision is 1e-2;
            CHECK_GE(param_.roll_angle(), 0) << "Scale factor must be bigger than 0 or equal!";
            double roll_angle = Rand(int(200 * param_.roll_angle() + 0.5) + 1) / 100.0 - param_.roll_angle();
            roll_angle = roll_angle * PI / 180.0f;

            if (0 != roll_angle)
            {
                int data_index, new_h, new_w;
                int top_index = 0, numStep = 0, channelStep = 0;
                float radius, transformed_roll;
                for (int n = 0; n < input_num; ++n) {
                    channelStep = 0;
                    for (int c = 0; c < input_channels; ++c) {
                        for (int h = 0; h < input_height; ++h) {
                            for (int w = 0; w < input_width; ++w) {
                                // 计算转后的夹角和旋转半径长度
                                if (abs(w - center_w) <= EPS)
                                {
                                    transformed_roll = (h >= center_h ? (0.5f*PI) : (-0.5f*PI));
                                    radius = float(abs(h - center_h));
                                }
                                else if (abs(h - center_h) <= EPS)
                                {
                                    transformed_roll = (w >= center_w ? 0 : PI);
                                    radius = float(abs(w - center_w));
                                }
                                else
                                {
                                    transformed_roll = atan((h - center_h) / (w - center_w));
                                    if (h >= center_h && transformed_roll < 0)
                                        transformed_roll = PI + transformed_roll;

                                    if (h <= center_h && transformed_roll > 0)
                                        transformed_roll = transformed_roll - PI;

                                    radius = sqrt(float(h - center_h)*(h - center_h) + float(w - center_w)*(w - center_w));
                                }
                                // 计算转前的夹角
                                transformed_roll = transformed_roll + roll_angle;

                                new_h = int(center_h + radius*sin(transformed_roll) + 0.5f);
                                new_w = int(center_w + radius*cos(transformed_roll) + 0.5f);
                                data_index = numStep + channelStep + new_h * input_width + new_w;

                                if (new_h < 0 || new_h >= input_height || new_w < 0 || new_w >= input_width)
                                    transformed_data[top_index] = filling_pixel;
                                else
                                    transformed_data[top_index] = inter_data[data_index];

                                ++top_index;
                            }
                        }
                        channelStep = channelStep + input_length;
                    }
                    numStep = numStep + input_size;
                }
            }
        }

        if (needs_gaussian_aug && 1 == Rand(2))
        {
            float mean_value = param_.gaussian_para().mean_value();
            float variance_value = param_.gaussian_para().variance_value();
            Dtype gaussian_value;
            int index = 0, numStep = 0, heightStep = 0;
            for (int n = 0; n < input_num; ++n) {
                heightStep = 0;
                for (int h = 0; h < input_height; ++h) {
                    for (int w = 0; w < input_width; ++w) {
                        gaussian_value = NormalRand(mean_value, variance_value);
                        index = numStep + heightStep + w;
                        for (int c = 0; c < input_channels; ++c, index = index + input_length) {
                            transformed_data[index] = transformed_data[index] + gaussian_value;
                            if (0 > transformed_data[index])
                                transformed_data[index] = 0;
                            if (Dtype(255.0) < transformed_data[index])
                                transformed_data[index] = Dtype(255.0);
                        }
                    }
                    heightStep = heightStep + input_width;
                }
                numStep = numStep + input_size;
            }
        }

        if (needs_cover_aug && 1 == Rand(2))
        {
            const int max_cover_size = param_.cover_size().max_size();
            const int min_cover_size = param_.cover_size().min_size();
            const int cover_size = max_cover_size - min_cover_size + 1;
            CHECK_GT(cover_size, 0) << "Cover size must be bigger than 0!";
            CHECK_GE(min_cover_size, 0) << "Min cover size must be bigger than 0!";
            CHECK_GE(max_cover_size, 0) << "Max cover size must be bigger than 0!";
            if (1 < cover_size || 0 < min_cover_size)
            {
                const int block_w = Rand(cover_size) + min_cover_size;
                const int block_h = Rand(cover_size) + min_cover_size;
                const int block_pos_w = Rand(input_width - block_w);
                const int block_pos_h = Rand(input_height - block_h);

                int index = 0, numStep = 0, channelStep = 0, heightStep;
                for (int n = 0; n < input_num; ++n) {
                    channelStep = 0;
                    for (int c = 0; c < input_channels; ++c) {
                        heightStep = block_pos_h * input_width;
                        for (int h = block_pos_h; h < block_pos_h + block_h + 1; ++h) {
                            index = numStep + channelStep + heightStep + block_pos_w;
                            for (int w = block_pos_w; w < block_pos_w + block_w + 1; ++w, ++index) {
                                transformed_data[index] = filling_pixel;
                            }
                            heightStep = heightStep + input_width;
                        }
                        channelStep = channelStep + input_length;
                    }
                    numStep = numStep + input_size;
                }
            }
        }

        if (needs_adjustment_aug/* && Rand(2)*/)
        {
            float max_brightness = param_.adjustment_para().max_brightness();
            float max_hue = param_.adjustment_para().max_hue();
            float min_saturation = param_.adjustment_para().min_saturation();
            float max_saturation = param_.adjustment_para().max_saturation();
            float min_contrast = param_.adjustment_para().min_contrast();
            float max_contrast = param_.adjustment_para().max_contrast();

            CHECK_GE(max_hue, 0) << "Max hue para must be bigger than 0!";
            CHECK_LE(max_hue, 180) << "Max hue para must be less than 180!";

            std::vector<int> adjustment_alg;
            if (0 != max_brightness)
                adjustment_alg.push_back(1);
            if (0 != max_hue)
                adjustment_alg.push_back(2);
            if (min_saturation < max_saturation && 0 <= max_saturation && 0 <= min_saturation)
                adjustment_alg.push_back(3);
            if (min_contrast < max_contrast && 0 <= max_contrast && 0 <= min_contrast)
                adjustment_alg.push_back(4);

            if (0 != adjustment_alg.size())
                shuffle(adjustment_alg.begin(), adjustment_alg.end());

            for (int i = 0; i < adjustment_alg.size(); ++i)
            {
                if (1 == adjustment_alg[i]) // 亮度随机扰动
                {
                    if (max_brightness > 255.0f)
                        max_brightness = 255.0f;
                    Dtype brightness_value = UniformRand(-max_brightness, max_brightness);
                    int index = 0, numStep = 0, heightStep = 0;
                    for (int n = 0; n < input_num; ++n) {
                        heightStep = 0;
                        for (int h = 0; h < input_height; ++h) {
                            for (int w = 0; w < input_width; ++w) {
                                index = numStep + heightStep + w;
                                for (int c = 0; c < input_channels; ++c, index = index + input_length) {
                                    transformed_data[index] = transformed_data[index] + brightness_value;
                                    if (0 > transformed_data[index])
                                        transformed_data[index] = 0;
                                    if (Dtype(255.0) < transformed_data[index])
                                        transformed_data[index] = Dtype(255.0);
                                }
                            }
                            heightStep = heightStep + input_width;
                        }
                        numStep = numStep + input_size;
                    }
                }

                if (2 == adjustment_alg[i])
                {
                    // 调整的色调范围在0~180度之间
                    if (max_hue > 180.0f)
                        max_hue = 180.0f;
                    Dtype hue_value = UniformRand(-max_hue, max_hue);
                    Dtype V_max, V_min;
                    Dtype data_b, data_g, data_r, data_h, data_s, data_v;
                    // BGR to HSV
                    int numStep = 0, heightStep = 0;
                    for (int n = 0; n < input_num; ++n) {
                        heightStep = 0;
                        for (int h = 0; h < input_height; ++h) {
                            for (int w = 0; w < input_width; ++w) {
                                // 默认为BGR模式
                                data_b = transformed_data[numStep + heightStep + w] / 255.0;
                                data_g = transformed_data[numStep + input_length + heightStep + w] / 255.0;
                                data_r = transformed_data[numStep + 2 * input_length + heightStep + w] / 255.0;

                                V_max = (std::max)((std::max)(data_b, data_g), data_r);
                                V_min = (std::min)((std::min)(data_b, data_g), data_r);

                                // h-channel
                                if (data_b == V_max)
                                    data_h = 240.0 + 60 * (data_r - data_g) / (V_max - V_min);
                                else if (data_g == V_max)
                                    data_h = 120.0 + 60 * (data_b - data_r) / (V_max - V_min);
                                else
                                    data_h = 60 * (data_g - data_b) / (V_max - V_min);
                                // s-channel
                                if (0 != V_max)
                                    data_s = (V_max - V_min) / V_max;
                                else
                                    data_s = 0;
                                // v-channel
                                data_v = V_max;

                                // 仅改变H通道
                                data_h = data_h + 2.0 * hue_value;
                                if (data_h >= 360.0)
                                    data_h -= 360.0;
                                else if (data_h < 0)
                                    data_h += 360.0;
                                data_h = 2.0 * data_h;

                                // 重新变回BGR格式
                                if (0 == data_s)
                                {
                                    data_r = data_v;
                                    data_g = data_v;
                                    data_b = data_v;
                                }
                                else
                                {
                                    data_h = data_h / 60.0;
                                    int flag = int(data_h);
                                    Dtype f = data_h - flag;
                                    Dtype p = data_v * (1 - data_s);
                                    Dtype q = data_v * (1 - f*data_s);
                                    Dtype t = data_v * (1 - (1 - f)*data_s);
                                    switch (flag)
                                    {
                                    case 0:
                                        data_b = p;
                                        data_g = t;
                                        data_r = data_v;
                                        break;
                                    case 1:
                                        data_b = p;
                                        data_g = data_v;
                                        data_r = q;
                                        break;
                                    case 2:
                                        data_b = t;
                                        data_g = data_v;
                                        data_r = p;
                                        break;
                                    case 3:
                                        data_b = data_v;
                                        data_g = q;
                                        data_r = p;
                                        break;
                                    case 4:
                                        data_b = data_v;
                                        data_g = p;
                                        data_r = t;
                                        break;
                                    case 5:
                                        data_b = q;
                                        data_g = p;
                                        data_r = data_v;
                                        break;
                                    default:
                                        break;
                                    }
                                    if (data_b > 1.0)
                                        data_b = 1.0;
                                    if (data_r > 1.0)
                                        data_r = 1.0;
                                    if (data_g > 1.0)
                                        data_g = 1.0;
                                    transformed_data[numStep + heightStep + w] = data_b * 255.0;
                                    transformed_data[numStep + input_length + heightStep + w] = data_g * 255.0;
                                    transformed_data[numStep + 2 * input_length + heightStep + w] = data_r * 255.0;
                                }
                            }
                            heightStep = heightStep + input_width;
                        }
                        numStep = numStep + input_size;
                    }
                }

                if (3 == adjustment_alg[i])
                {
                    if (max_saturation > 2.0f)
                        max_saturation = 2.0f;
                    Dtype saturation_value = UniformRand(min_saturation, max_saturation);
                    Dtype V_max, V_min;
                    Dtype data_b, data_g, data_r, data_h, data_s, data_v;
                    // BGR to HSV
                    int numStep = 0, heightStep = 0;
                    for (int n = 0; n < input_num; ++n) {
                        heightStep = 0;
                        for (int h = 0; h < input_height; ++h) {
                            for (int w = 0; w < input_width; ++w) {
                                // 默认为BGR模式
                                data_b = transformed_data[numStep + heightStep + w] / 255.0;
                                data_g = transformed_data[numStep + input_length + heightStep + w] / 255.0;
                                data_r = transformed_data[numStep + 2 * input_length + heightStep + w] / 255.0;

                                V_max = (std::max)((std::max)(data_b, data_g), data_r);
                                V_min = (std::min)((std::min)(data_b, data_g), data_r);

                                // h-channel
                                if (data_b == V_max)
                                    data_h = 240.0 + 60 * (data_r - data_g) / (V_max - V_min);
                                else if (data_g == V_max)
                                    data_h = 120.0 + 60 * (data_b - data_r) / (V_max - V_min);
                                else
                                    data_h = 60 * (data_g - data_b) / (V_max - V_min);
                                // s-channel
                                if (0 != V_max)
                                    data_s = (V_max - V_min) / V_max;
                                else
                                    data_s = 0;
                                // v-channel
                                data_v = V_max;

                                // 仅改变S通道
                                data_s = data_s * saturation_value;
                                if (data_s < 0)
                                    data_s = 0;
                                if (data_s > 1.0)
                                    data_s = 1.0;

                                // 重新变回BGR格式
                                if (0 == data_s)
                                {
                                    data_r = data_v;
                                    data_g = data_v;
                                    data_b = data_v;
                                }
                                else
                                {
                                    data_h = data_h / 60.0;
                                    int flag = int(data_h);
                                    Dtype f = data_h - flag;
                                    Dtype p = data_v * (1 - data_s);
                                    Dtype q = data_v * (1 - f*data_s);
                                    Dtype t = data_v * (1 - (1 - f)*data_s);
                                    switch (flag)
                                    {
                                    case 0:
                                        data_b = p;
                                        data_g = t;
                                        data_r = data_v;
                                        break;
                                    case 1:
                                        data_b = p;
                                        data_g = data_v;
                                        data_r = q;
                                        break;
                                    case 2:
                                        data_b = t;
                                        data_g = data_v;
                                        data_r = p;
                                        break;
                                    case 3:
                                        data_b = data_v;
                                        data_g = q;
                                        data_r = p;
                                        break;
                                    case 4:
                                        data_b = data_v;
                                        data_g = p;
                                        data_r = t;
                                        break;
                                    case 5:
                                        data_b = q;
                                        data_g = p;
                                        data_r = data_v;
                                        break;
                                    default:
                                        break;
                                    }
                                    if (data_b > 1.0)
                                        data_b = 1.0;
                                    if (data_r > 1.0)
                                        data_r = 1.0;
                                    if (data_g > 1.0)
                                        data_g = 1.0;
                                    transformed_data[numStep + heightStep + w] = data_b * 255.0;
                                    transformed_data[numStep + input_length + heightStep + w] = data_g * 255.0;
                                    transformed_data[numStep + 2 * input_length + heightStep + w] = data_r * 255.0;
                                }
                            }
                            heightStep = heightStep + input_width;
                        }
                        numStep = numStep + input_size;
                    }
                }

                if (4 == adjustment_alg[i])
                {
                    if (max_contrast > 2.0f)
                        max_contrast = 2.0f;
                    Dtype contrast_value = UniformRand(min_contrast, max_contrast);
                    int index = 0, numStep = 0, heightStep = 0;
                    for (int n = 0; n < input_num; ++n) {
                        heightStep = 0;
                        for (int h = 0; h < input_height; ++h) {
                            for (int w = 0; w < input_width; ++w) {
                                index = numStep + heightStep + w;
                                for (int c = 0; c < input_channels; ++c, index = index + input_length) {
                                    transformed_data[index] = transformed_data[index] * contrast_value;
                                    if (0 > transformed_data[index])
                                        transformed_data[index] = 0;
                                    if (Dtype(255.0) < transformed_data[index])
                                        transformed_data[index] = Dtype(255.0);
                                }
                            }
                            heightStep = heightStep + input_width;
                        }
                        numStep = numStep + input_size;
                    }
                }
            } // for (int i = 0; i < adjustment_alg.size(); ++i)
        }

        //         if (inter_data)
        //             delete[] inter_data;
        //         inter_data = 0;
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::DataAugmentation(const Blob<Dtype> *input_blob,
        Blob<Dtype> *transformed_blob)
    {
        DataAugmentation(input_blob->cpu_data(), transformed_blob->mutable_cpu_data(),
            input_blob->num(), input_blob->channels(), input_blob->height(), input_blob->width());
    }

    template<typename Dtype>
    void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
        Blob<Dtype>* transformed_blob) {
        const int input_num = input_blob->num();
        const int input_channels = input_blob->channels();
        int input_height = input_blob->height();
        int input_width = input_blob->width();
        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
            crop_w = input_width;
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
            crop_h = input_height;
        }
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width();
        const int margin = param_.margin();
        CHECK_GE(margin, 0) << "margin must be greater than or equal to 0";
        CHECK_LE(margin, (input_height - crop_h) / 2) << "margin must be not greater than (input_height - crop_size) / 2";
        CHECK_LE(margin, (input_width - crop_w) / 2) << "margin must be not greater than (input_width - crop_size) / 2";

        shared_ptr< Blob<Dtype> > interBlob(new Blob<Dtype>(input_num, input_channels, input_height, input_width));
        Blob<Dtype> *inter_blob = interBlob.get();// new Blob<Dtype>(input_num, input_channels, input_height, input_width);
        DataAugmentation(input_blob, inter_blob);

        // resize images
        Resize(crop_h, crop_w, input_num, input_channels, 
            input_height, input_width, interBlob);
        inter_blob = interBlob.get();
        int new_height = param_.new_height();
        int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
            new_height = new_size;
            new_width = new_size;
        }

        CHECK_EQ(param_.patch_center_x_size(), param_.patch_center_y_size());
        // Initialize transformed_blob with the right shape.
        int h_off = 0;
        int w_off = 0;
        std::vector<int> patch_off_h;
        std::vector<int> patch_off_w;
        int height = input_height;
        int width = input_width;
        int input_length = input_channels * input_height * input_width;
        int patch_num = 0;
        if (crop_h || crop_w)
        {
            CHECK_GE(crop_h, patch_height);
            CHECK_GE(crop_w, patch_width);
            height = crop_h;
            width = crop_w;
            if (phase_ == TRAIN) {
                h_off = Rand(input_height - crop_h + 1 - 2 * margin) + margin;
                w_off = Rand(input_width - crop_w + 1 - 2 * margin) + margin;
            }
            else {
                h_off = (input_height - crop_h) / 2;
                w_off = (input_width - crop_w) / 2;
            }
        }
        if (patch_height)
        {
            patch_num = param_.patch_center_x_size();
            height = patch_height;
            width = patch_width;
            for (int p = 0; p < patch_num; ++p)
            {
                patch_off_w.push_back(param_.patch_center_x(p) - patch_width / 2);
                patch_off_h.push_back(param_.patch_center_y(p) - patch_height / 2);
            }
        }
        else
        {
            patch_num = 1;
            patch_off_h.push_back(0);
            patch_off_w.push_back(0);
        }

        //if (transformed_blob->count() == 0) {
        //    // Initialize transformed_blob with the right shape.
        //    if (crop_h && crop_w) {
        //        transformed_blob->Reshape(input_num, input_channels,
        //            height, crop_w);
        //    }
        //    else {
        //        transformed_blob->Reshape(input_num, input_channels,
        //            input_height, input_width);
        //    }
        //}

        if (transformed_blob->count() == 0) {
            transformed_blob->Reshape(input_num, input_channels * patch_num,
                height, width);
        }

        const int num = transformed_blob->num();
        const int channels = transformed_blob->channels();
        //const int height = transformed_blob->height();
        //const int width = transformed_blob->width();
        const int size = transformed_blob->count();

        CHECK_LE(input_num, num);
        CHECK_EQ(channels, input_channels * patch_num);
        //CHECK_EQ(input_channels, channels);
        CHECK_GE(input_height, height);
        CHECK_GE(input_width, width);


        const Dtype scale = param_.scale();
        const bool do_mirror = param_.mirror() && Rand(2);
        const bool has_mean_file = param_.has_mean_file();
        const bool has_mean_values = mean_values_.size() > 0;

        //int h_off = 0;
        //int w_off = 0;
        //if (crop_h && crop_w) {
        //    CHECK_EQ(crop_h, height);
        //    CHECK_EQ(crop_w, width);
        //    // We only do random crop when we do training.
        //    if (phase_ == TRAIN && !param_.center_crop()) {
        //        h_off = Rand(input_height - crop_h + 1);
        //        w_off = Rand(input_width - crop_w + 1);
        //    }
        //    else {
        //        h_off = (input_height - crop_h) / 2;
        //        w_off = (input_width - crop_w) / 2;
        //    }
        //}
        //else {
        //    CHECK_EQ(input_height, height);
        //    CHECK_EQ(input_width, width);
        //}

        //Dtype* input_data = input_blob->mutable_cpu_data();
        Dtype* input_data = inter_blob->mutable_cpu_data();
        if (has_mean_file) {
            if (new_width > 0) {
                CHECK_EQ(input_channels, data_mean_resized_.channels());
                CHECK_EQ(input_height, data_mean_resized_.height());
                CHECK_EQ(input_width, data_mean_resized_.width());
                for (int n = 0; n < input_num; ++n) {
                    int offset = inter_blob->offset(n);
                    caffe_sub(data_mean_resized_.count(), input_data + offset,
                        data_mean_resized_.cpu_data(), input_data + offset);
                }
            }
            else {
                CHECK_EQ(input_channels, data_mean_.channels());
                CHECK_EQ(input_height, data_mean_.height());
                CHECK_EQ(input_width, data_mean_.width());
                for (int n = 0; n < input_num; ++n) {
                    int offset = input_blob->offset(n);
                    caffe_sub(data_mean_.count(), input_data + offset,
                        data_mean_.cpu_data(), input_data + offset);
                }
            }  
        }

        if (has_mean_values) {
            CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
                "Specify either 1 mean_value or as many as channels: " << input_channels;
            if (mean_values_.size() == 1) {
                caffe_add_scalar(inter_blob->count(), -(mean_values_[0]), input_data);
                //caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
            }
            else {
                for (int n = 0; n < input_num; ++n) {
                    for (int c = 0; c < input_channels; ++c) {
                        int offset = inter_blob->offset(n, c);
                        //int offset = input_blob->offset(n, c);
                        caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
                            input_data + offset);
                    }
                }
            }
        }

        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        int top_index, data_index;
        /*for (int n = 0; n < input_num; ++n) {
            int top_index_n = n * channels;
            int data_index_n = n * channels;
            for (int c = 0; c < channels; ++c) {
                int top_index_c = (top_index_n + c) * height;
                int data_index_c = (data_index_n + c) * input_height + h_off;
                for (int h = 0; h < height; ++h) {
                    int top_index_h = (top_index_c + h) * width;
                    int data_index_h = (data_index_c + h) * input_width + w_off;
                    if (do_mirror) {
                        int top_index_w = top_index_h + width - 1;
                        for (int w = 0; w < width; ++w) {
                            transformed_data[top_index_w - w] = input_data[data_index_h + w];
                        }
                    }
                    else {
                        for (int w = 0; w < width; ++w) {
                            transformed_data[top_index_h + w] = input_data[data_index_h + w];
                        }
                    }
                }
            }
        }*/
        for (int n = 0; n < input_num; ++n) {
            for (int p = 0; p < patch_num; ++p) {
                for (int c = 0; c < input_channels; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            if (patch_height) {
                                if (do_mirror)
                                    data_index = ((n*input_channels + c) * input_height + h_off + patch_off_h[p] + h) * input_width + (input_width - 1 - (w_off + patch_off_w[p] + w));
                                else
                                    data_index = ((n*input_channels + c) * input_height + h_off + patch_off_h[p] + h) * input_width + w_off + patch_off_w[p] + w;
                                top_index = (((n*patch_num + p)*input_channels + c) * height + h) * width + w;
                            }
                            else {
                                data_index = ((n*input_channels + c) * input_height + h_off + patch_off_h[p] + h) * input_width + w_off + patch_off_w[p] + w;
                                if (do_mirror) {
                                    top_index = (((n*patch_num + p)*input_channels + c) * height + h) * width + (width - 1 - w);
                                }
                                else {
                                    top_index = (((n*patch_num + p)*input_channels + c) * height + h) * width + w;
                                }
                            }
                            if (data_index < 0 || data_index >= input_length)
                                continue;
                            transformed_data[top_index] = input_data[data_index];
                        }
                    }
                }
            }
        }
        if (scale != Dtype(1)) {
            DLOG(INFO) << "Scale: " << scale;
            caffe_scal(size, scale, transformed_data);
        }
    }

    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
        if (datum.encoded()) {
#ifdef USE_OPENCV
            CHECK(!(param_.force_color() && param_.force_gray()))
                << "cannot set both force_color and force_gray";
            cv::Mat cv_img;
            if (param_.force_color() || param_.force_gray()) {
                // If force_color then decode in color otherwise decode in gray.
                cv_img = DecodeDatumToCVMat(datum, param_.force_color());
            }
            else {
                cv_img = DecodeDatumToCVMatNative(datum);
            }
            // InferBlobShape using the cv::image.
            return InferBlobShape(cv_img);
#else
            LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
        }
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();
        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
            crop_w = datum_width;
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
            crop_h = datum_height;
        }
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width(); 

        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        const int new_height = param_.new_height();
        const int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
        }
        else {
            CHECK_GE(datum_height, crop_h);
            CHECK_GE(datum_width, crop_w);
        }

        // Check dimensions.
        CHECK_GT(datum_channels, 0);
        //CHECK_GE(datum_height, crop_h);
        //CHECK_GE(datum_width, crop_w);
        // Build BlobShape.
        vector<int> shape(4);
        shape[0] = 1;
        shape[1] = (patch_height) ? (datum_channels * param_.patch_center_x_size()) : datum_channels;
        shape[2] = (crop_h) ? crop_h : datum_height;
        shape[3] = (crop_w) ? crop_w : datum_width;
        shape[2] = (patch_height) ? patch_height : shape[2];
        shape[3] = (patch_width) ? patch_width : shape[3];
        /*shape[1] = datum_channels;
        shape[2] = (crop_h) ? crop_h : datum_height;
        shape[3] = (crop_w) ? crop_w : datum_width;*/
        return shape;
    }

    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(
        const vector<Datum> & datum_vector) {
        const int num = datum_vector.size();
        CHECK_GT(num, 0) << "There is no datum to in the vector";
        // Use first datum in the vector to InferBlobShape.
        vector<int> shape = InferBlobShape(datum_vector[0]);
        // Adjust num to the size of the vector.
        shape[0] = num;
        return shape;
    }

    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(
        const Blob<Dtype> *input_blob) {
        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
        }
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width();
        const int blob_channels = input_blob->channels();
        const int blob_height = input_blob->height();
        const int blob_width = input_blob->width();

        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        const int new_height = param_.new_height();
        const int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
        }
        else {
            CHECK_GE(blob_height, crop_h);
            CHECK_GE(blob_width, crop_w);
        }

        CHECK_GT(blob_channels, 0);
        vector<int> shape(4);
        shape[0] = input_blob->num();
        shape[1] = (patch_height) ? (blob_channels * param_.patch_center_x_size()) : blob_channels;
        shape[2] = (crop_h) ? crop_h : blob_height;
        shape[3] = (crop_w) ? crop_w : blob_width;
        shape[2] = (patch_height) ? patch_height : shape[2];
        shape[3] = (patch_width) ? patch_width : shape[3];
        return shape;
    }

#ifdef USE_OPENCV
    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
        int crop_h = 0;
        int crop_w = 0;
        if (param_.has_crop_size()) {
            crop_h = param_.crop_size();
            crop_w = param_.crop_size();
        }
        if (param_.has_crop_h()) {
            crop_h = param_.crop_h();
        }
        if (param_.has_crop_w()) {
            crop_w = param_.crop_w();
        }
        const int patch_height = param_.patch_height();
        const int patch_width = param_.patch_width();
        const int img_channels = cv_img.channels();
        const int img_height = cv_img.rows;
        const int img_width = cv_img.cols;

        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        const int new_height = param_.new_height();
        const int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
        }
        else {
            CHECK_GE(img_height, crop_h);
            CHECK_GE(img_width, crop_w);
        }

        // Check dimensions.
        CHECK_GT(img_channels, 0);
        //CHECK_GE(img_height, crop_h);
        //CHECK_GE(img_width, crop_w);
        // Build BlobShape.
        vector<int> shape(4);
        shape[0] = 1;
        shape[1] = (patch_height) ? (img_channels * param_.patch_center_x_size()) : img_channels;
        shape[2] = (crop_h) ? crop_h : img_height;
        shape[3] = (crop_w) ? crop_w : img_width;
        shape[2] = (patch_height) ? patch_height : shape[2];
        shape[3] = (patch_width) ? patch_width : shape[3];
        /*shape[1] = img_channels;
        shape[2] = (crop_h) ? crop_h : img_height;
        shape[3] = (crop_w) ? crop_w : img_width;*/
        return shape;
    }

    template<typename Dtype>
    vector<int> DataTransformer<Dtype>::InferBlobShape(
        const vector<cv::Mat> & mat_vector) {
        const int num = mat_vector.size();
        CHECK_GT(num, 0) << "There is no cv_img to in the vector";
        // Use first cv_img in the vector to InferBlobShape.
        vector<int> shape = InferBlobShape(mat_vector[0]);
        // Adjust num to the size of the vector.
        shape[0] = num;
        return shape;
    }
#endif  // USE_OPENCV

    template <typename Dtype>
    void DataTransformer<Dtype>::InitRand() {
        //printf("%d\n", int(param_.has_erase_ratio()));
        const bool needs_rand = param_.mirror() ||
            (phase_ == TRAIN && (param_.crop_size() || param_.crop_h() || param_.crop_w()) /*&& !param_.center_crop()*/) ||
            //(param_.has_erase_ratio()) ||
            (phase_ == TRAIN && param_.color_augmentation()) ||
            (phase_ == TRAIN && param_.has_gaussian_para()) ||
            (phase_ == TRAIN && param_.has_scale_factor()) ||
            (phase_ == TRAIN && param_.has_cover_size()) ||
            (phase_ == TRAIN && param_.has_adjustment_para()) ||
            (phase_ == TRAIN && param_.has_fisheye_param()) ||
            (phase_ == TRAIN && param_.roll_angle() != 0);
        if (needs_rand) {
            const unsigned int rng_seed = caffe_rng_rand();
            rng_.reset(new Caffe::RNG(rng_seed));
            //printf("hzx\n");
            //prnd_ = std::mt19937(time(NULL));
        }
        else {
            rng_.reset();
            //printf("lxw\n");
        }
    }

    template <typename Dtype>
    int DataTransformer<Dtype>::Rand(int n) {
        CHECK(rng_);
        CHECK_GT(n, 0);
        caffe::rng_t* rng =
            static_cast<caffe::rng_t*>(rng_->generator());
        return ((*rng)() % n);
    }

    template<typename Dtype>
    Dtype DataTransformer<Dtype>::NormalRand(float mean_value, float variance_value) {
        CHECK(rng_);
        caffe::rng_t* rng =
            static_cast<caffe::rng_t*>(rng_->generator());

        boost::normal_distribution<Dtype> d(mean_value, variance_value);
        return d((*rng));
    }

    template<typename Dtype>
    Dtype DataTransformer<Dtype>::UniformRand(float lower_value, float upper_value) {
        CHECK(rng_);
        caffe::rng_t* rng =
            static_cast<caffe::rng_t*>(rng_->generator());

        boost::uniform_real<Dtype> d(lower_value, upper_value);
        return d((*rng));
    }

    template <typename Dtype>
    void DataTransformer<Dtype>::Resize(int crop_h, int crop_w, 
        int input_num, int input_channels,
        int &input_height, int &input_width, 
        shared_ptr< Blob<Dtype> > &interBlob)
    {
        // resize images
        // It will also resize images if new_height or new_width are not zero.
        // new_size and new_height, new_width can't be set simultaneously.
        int new_height = param_.new_height();
        int new_width = param_.new_width();
        const int new_size = param_.new_size();
        if (new_size > 0) {
            CHECK_EQ(new_width, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_EQ(new_height, 0) <<
                "new_size and new_height, new_width can't be set simultaneously.";
            CHECK_LE(crop_w, new_size) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_size) <<
                "crop_size must be less than or equal to new_height";
            new_height = new_size;
            new_width = new_size;
        }
        else if (new_width > 0 || new_height > 0) {
            CHECK_GT(new_width, 0) <<
                "if you specify new_with, you must also specify new_height.";
            CHECK_GT(new_height, 0) <<
                "if you specify new_height, you must also specify new_width.";
            CHECK_LE(crop_w, new_width) <<
                "crop_size must be less than or equal to new_width";
            CHECK_LE(crop_h, new_height) <<
                "crop_size must be less than or equal to new_height";
        }
        else {
            CHECK_GE(input_height, crop_h);
            CHECK_GE(input_width, crop_w);
        }
        // resize image
        if (new_width > 0) {
            shared_ptr< Blob<Dtype> > resizeBlob(new Blob<Dtype>(input_num,
                input_channels, new_height, new_width));
            Blob<Dtype> *resize_blob = resizeBlob.get();
            Dtype *resize_data = resize_blob->mutable_cpu_data();

            const Blob<Dtype> *inter_blob = interBlob.get();
            const Dtype* input_data = inter_blob->cpu_data();
            for (int n = 0; n < input_num; ++n) {
                int offset = inter_blob->offset(n);
                int offset_resize = resize_blob->offset(n);
                ResizeDatum(input_width, new_width, input_height, new_height,
                    input_channels, input_data + offset, resize_data + offset_resize);
            }

            // assign resized value to datum
            input_height = new_height;
            input_width = new_width;
            interBlob = resizeBlob;

            if (param_.has_mean_file()) {
                CHECK_EQ(input_channels, data_mean_resized_.channels());
                CHECK_EQ(input_height, data_mean_resized_.height());
                CHECK_EQ(input_width, data_mean_resized_.width());
            }
        }
    }

    INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
