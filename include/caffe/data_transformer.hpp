#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>
#include <random>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    /**
    * @brief Applies common transformations to the input data, such as
    * scaling, mirroring, substracting the image mean...
    */
    template <typename Dtype>
    class DataTransformer {
    public:
        explicit DataTransformer(const TransformationParameter& param, Phase phase);
        virtual ~DataTransformer() {}

        /**
        * @brief Initialize the Random number generations if needed by the
        *    transformation.
        */
        void InitRand();

        /**
        * @brief Applies the transformation defined in the data layer's
        * transform_param block to the data.
        *
        * @param datum
        *    Datum containing the data to be transformed.
        * @param transformed_blob
        *    This is destination blob. It can be part of top blob's data if
        *    set_cpu_data() is used. See data_layer.cpp for an example.
        */
        void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

        /**
        * @brief Applies the transformation defined in the data layer's
        * transform_param block to a vector of Datum.
        *
        * @param datum_vector
        *    A vector of Datum containing the data to be transformed.
        * @param transformed_blob
        *    This is destination blob. It can be part of top blob's data if
        *    set_cpu_data() is used. See memory_layer.cpp for an example.
        */
        void Transform(const vector<Datum> & datum_vector,
            Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
        /**
        * @brief Applies the transformation defined in the data layer's
        * transform_param block to a vector of Mat.
        *
        * @param mat_vector
        *    A vector of Mat containing the data to be transformed.
        * @param transformed_blob
        *    This is destination blob. It can be part of top blob's data if
        *    set_cpu_data() is used. See memory_layer.cpp for an example.
        */
        void Transform(const vector<cv::Mat> & mat_vector,
            Blob<Dtype>* transformed_blob/*, bool transpose = false*/);

        /**
        * @brief Applies the transformation defined in the data layer's
        * transform_param block to a cv::Mat
        *
        * @param cv_img
        *    cv::Mat containing the data to be transformed.
        * @param transformed_blob
        *    This is destination blob. It can be part of top blob's data if
        *    set_cpu_data() is used. See image_data_layer.cpp for an example.
        */
        void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob/*, bool transpose = false*/);
#endif  // USE_OPENCV

        /**
        * @brief Applies the same transformation defined in the data layer's
        * transform_param block to all the num images in a input_blob.
        *
        * @param input_blob
        *    A Blob containing the data to be transformed. It applies the same
        *    transformation to all the num images in the blob.
        * @param transformed_blob
        *    This is destination blob, it will contain as many images as the
        *    input blob. It can be part of top blob's data.
        */
        void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

        /**
        * @brief Infers the shape of transformed_blob will have when
        *    the transformation is applied to the data.
        *
        * @param datum
        *    Datum containing the data to be transformed.
        */
        vector<int> InferBlobShape(const Datum& datum);
        /**
        * @brief Infers the shape of transformed_blob will have when
        *    the transformation is applied to the data.
        *    It uses the first element to infer the shape of the blob.
        *
        * @param datum_vector
        *    A vector of Datum containing the data to be transformed.
        */
        vector<int> InferBlobShape(const vector<Datum> & datum_vector);
        /**
        * @brief Infers the shape of transformed_blob will have when
        *    the transformation is applied to the data.
        *    It uses the first element to infer the shape of the blob.
        *
        * @param mat_vector
        *    A vector of Mat containing the data to be transformed.
        */
        vector<int> InferBlobShape(const Blob<Dtype> *input_blob);
#ifdef USE_OPENCV
        vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
        /**
        * @brief Infers the shape of transformed_blob will have when
        *    the transformation is applied to the data.
        *
        * @param cv_img
        *    cv::Mat containing the data to be transformed.
        */
        vector<int> InferBlobShape(const cv::Mat& cv_img);

        void DataAugmentation(const cv::Mat& cv_img, cv::Mat &transformed_mat);
#endif  // USE_OPENCV

    protected:
        /**
        * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
        *
        * @param n
        *    The upperbound (exclusive) value of the random number.
        * @return
        *    A uniformly random integer value from ({0, 1, ..., n-1}).
        */
        virtual int Rand(int n);

        virtual Dtype NormalRand(float mean_value, float variance_value);

        virtual Dtype UniformRand(float lower_value, float upper_value);

        void Transform(const Datum& datum, Dtype* transformed_data);

        //void VideoTransform(const VolumeDatum& datum, Dtype* transformed_data);

        void DataAugmentation(const Datum& datum, Dtype *transformed_data);

        void DataAugmentation(const Blob<Dtype> *input_blob, Blob<Dtype> *transformed_blob);

        void DataAugmentation(const Dtype *input_data, Dtype *transformed_data,
            const int input_num, const int input_channel, const int input_height, const int input_width);

        void ResizeDatum(int datum_width, int new_width, int datum_height, int new_height,
            int datum_channels, const Dtype *inter_data, Dtype *resize_data) const;

        void Resize(int crop_h, int crop_w, int input_num, int input_channels,
            int &input_height, int &input_width, shared_ptr< Blob<Dtype> > &interBlob);

        void Resize(int crop_h, int crop_w, int datum_channels, int &datum_height,
            int &datum_width, int &datum_length, shared_ptr<Dtype> &interData);

        //void Transform(const Datum& datum, Dtype* transformed_data);

        // Tranformation parameters
        TransformationParameter param_;


        shared_ptr<Caffe::RNG> rng_;
        //std::mt19937 prnd_;//caffe's RNG is difficult to use.
        Phase phase_;
        Blob<Dtype> data_mean_;
        Blob<Dtype> data_mean_resized_;
        vector<Dtype> mean_values_;
    };

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
