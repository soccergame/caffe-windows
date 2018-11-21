#ifndef CAFFE_NORMALIZE_LAYER_HPP_
#define CAFFE_NORMALIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    /**
    * @brief Normalizes the input to be 0 - 1 range.
    *
    * TODO(dox): thorough documentation for Forward, Backward, and proto params.
    */
    template <typename Dtype>
    class NormalizeLayer : public Layer<Dtype> {
    public:
        explicit NormalizeLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void Reshape(const vector< Blob<Dtype>* > &bottom,
            const vector< Blob<Dtype>* > &top);

        virtual inline const char* type() const { return "Normalize"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector< Blob<Dtype>* > &bottom,
            const vector< Blob<Dtype>* > &top);
        virtual void Forward_gpu(const vector< Blob<Dtype>* > &bottom,
            const vector< Blob<Dtype>* > &top);
        virtual void Backward_cpu(const vector< Blob<Dtype>* > &top,
            const vector<bool> &propagate_down, const vector< Blob<Dtype>* > &bottom);
        virtual void Backward_gpu(const vector< Blob<Dtype>* > &top,
            const vector<bool> &propagate_down, const vector< Blob<Dtype>* > &bottom);

        Blob<Dtype> squared_;
        static const float EPS;	// = 0.000001f;
    };
}  // namespace caffe

#endif  // CAFFE_NORMALIZE_LAYER_HPP_
