#ifndef CAFFE_TRANSFORM_DATA_LAYER_HPP_
#define CAFFE_TRANSFORM_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template <typename Dtype>
    class TransformDataLayer : public Layer<Dtype> {
    public:
        explicit TransformDataLayer(const LayerParameter& param)
            : Layer<Dtype>(param),
            transform_param_(param.transform_param()) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "TransformData"; }
        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }

        virtual void Forward_cpu(const vector< Blob<Dtype>* > &bottom,
            const vector< Blob<Dtype>* > &top);
        virtual void Backward_cpu(const vector< Blob<Dtype>* > &top,
            const vector<bool> &propagate_down, const vector< Blob<Dtype>* > &bottom) {}

    protected:
        TransformationParameter transform_param_;
        shared_ptr< DataTransformer<Dtype> > data_transformer_;
    };

}  // namespace caffe

#endif  // CAFFE_TRANSFORM_DATA_LAYER_HPP_
