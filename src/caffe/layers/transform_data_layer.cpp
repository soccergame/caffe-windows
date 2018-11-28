#include "caffe/layers/transform_data_layer.hpp"

// �ò㲻����ѵ��������ʹ�ã�ֻ���ڷ�װSDK�Ĺ�����ʹ�ã���Ϊѵ�������У������������������ܻᵼ��ȡ����patch��������Ҫ�õ���patch���ݲ�һ��
// ���ڷ�װSDK��ʱ�����������ǲ��᾵������ģ����ȡ����patch��������ϣ����patch

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
