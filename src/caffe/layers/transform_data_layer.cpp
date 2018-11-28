#include "caffe/layers/transform_data_layer.hpp"

// 该层不能在训练过程中使用，只能在封装SDK的过程中使用，因为训练过程中，对样本的随机镜像可能会导致取到的patch与我们想要得到的patch数据不一致
// 而在封装SDK的时候，输入数据是不会镜像操作的，因此取到的patch是我们所希望的patch

namespace caffe {
	template <typename Dtype>
	void TransformDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* > &bottom,
		const vector<Blob<Dtype>* > &top) {
		CHECK_EQ(bottom.size(), top.size()) << "The number of input blobs and output blobs must be the same!";
		this->data_transformer_.reset(
			new DataTransformer<Dtype>(this->transform_param_, this->phase_));
		this->data_transformer_->InitRand();
	}

	template <typename Dtype>
	void TransformDataLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>* > &bottom, const vector<Blob<Dtype>* > &top) {       
		for (int i = 0; i < bottom.size(); ++i) {
			vector<int> top_shape = this->data_transformer_->InferBlobShape(bottom[i]);
			top[i]->Reshape(top_shape);
		}      
	}

	template <typename Dtype>
	void TransformDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		for (int i = 0; i < bottom.size(); ++i) {
			this->data_transformer_->Transform(bottom[i], top[i]);
		}   
	}

	INSTANTIATE_CLASS(TransformDataLayer);
	REGISTER_LAYER_CLASS(TransformData);
}
