#ifndef CAFFE_RANKHARD_LOSS_LAYER_HPP_
#define CAFFE_RANKHARD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    /**
    * The input of the network is pairs of image patches. 
    * For each pair of patches, they are taken as the similar patches in the same video track. 
    * We use the label to specify whether the patches come from the same video, 
    * if they come from different videos they will have different labels (it does not matter what is the number, 
    * just need to be integer). In this way, 
    * we can get the third negative patch from other pairs with different labels.
    * In the loss, for each pair of patches, 
    * it will try to find the third negative patch in the same batch. 
    * There are two ways to do it, one is random selection, 
    * the other is hard negative mining.
    * neg_num means how many negative patches you want for each pair of patches, if it is 4, that means there are 4 triplets. 
    * pair_size = 2 just means inputs are pairs of patches. 
    * hard_ratio = 0.5 means half of the negative patches are hard examples, 
    * rand_ratio = 0.5 means half of the negative patches are randomly selected. 
    * For start, you can just set rand_ratio = 1 and hard_ratio = 0. 
    * The margin for contrastive loss needs to be designed for different tasks, 
    * trying to set margin = 0.5 or 0.1 might make a difference for other tasks.
    */
    template <typename Dtype>
    class RankHardLossLayer : public LossLayer<Dtype> {
    public:
        explicit RankHardLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), test_(false) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "RankHardLoss"; }

        virtual inline bool AllowForceBackward(const int bottom_index) const {
            return true;
        }

		shared_ptr<Caffe::RNG> GetRNGGenerator() const {
			return neg_sel_rng_;
		}		
		void SetTest(bool test) { test_ = test; }
    protected:
        /// @copydoc EuclideanLossLayer
        virtual void Forward_cpu(const vector< Blob<Dtype>* > &bottom,
            const vector< Blob<Dtype>* > &top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*> &bottom,
            const vector< Blob<Dtype>* > &top);

        virtual void Backward_cpu(const vector< Blob<Dtype>* > &top,
            const vector<bool> &propagate_down, const vector< Blob<Dtype>* > &bottom);
        virtual void Backward_gpu(const vector< Blob<Dtype>* > &top,
            const vector<bool> &propagate_down, const vector< Blob<Dtype>* > &bottom);

		void DistCosineGenerateTripletsCPU(const vector< Blob<Dtype>* > &bottom);
		void DistCosineGenerateTripletsGPU(const vector< Blob<Dtype>* > &bottom);

		void DistEuclideanTripletSelectCPU(const vector< Blob<Dtype>* > &bottom);    

		Blob<int> selected_triplets_;
		Blob<Dtype> triplet_losses_;
		int triplets_num_;		

		shared_ptr<Caffe::RNG> neg_sel_rng_;
		bool test_;
    };

}  // namespace caffe

#endif  // CAFFE_RANKHARD_LOSS_LAYER_HPP_
