/*
* test_rank_hard_loss_layer.cpp
*
* Created on: July 1, 2016
*     Author: THID@Hisign
*/

#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<cstring>
#include<vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rank_hard_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {	
	template <typename Dtype>
	bool CompareLoss(const pair<Dtype, int> &left, const pair<Dtype, int> &right) {
		return left.first > right.first ? true : false;
	}

	template <typename TypeParam>
	class RankHardLossLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		RankHardLossLayerTest()
			: blob_bottom_data_(new Blob<Dtype>(6, 8, 1, 1)),
			blob_bottom_y_(new Blob<Dtype>(6, 1, 1, 1)),
			blob_top_loss_(new Blob<Dtype>()){

// 			// [4, 2, 1, 3, 2, 1, 2, 2]
// 			blob_bottom_data_->mutable_cpu_data()[0] = Dtype(4);
// 			blob_bottom_data_->mutable_cpu_data()[1] = Dtype(2);
// 			blob_bottom_data_->mutable_cpu_data()[2] = Dtype(1);
// 			blob_bottom_data_->mutable_cpu_data()[3] = Dtype(3);
// 			blob_bottom_data_->mutable_cpu_data()[4] = Dtype(2);
// 			blob_bottom_data_->mutable_cpu_data()[5] = Dtype(1);
// 			blob_bottom_data_->mutable_cpu_data()[6] = Dtype(2);
// 			blob_bottom_data_->mutable_cpu_data()[7] = Dtype(2);
// 			blob_bottom_y_->mutable_cpu_data()[0] = Dtype(0);
// 			blob_bottom_y_->mutable_cpu_data()[1] = Dtype(0);
// 			blob_bottom_y_->mutable_cpu_data()[2] = Dtype(1);
// 			blob_bottom_y_->mutable_cpu_data()[3] = Dtype(1);

			// fill the values
			FillerParameter filler_param;
			filler_param.set_min(-1.0);
			filler_param.set_max(1.0);  // distances~=1.0 to test both sides ofmargin
			UniformFiller<Dtype>filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_data_);

			const int step = 2;
			for (int i = 0; i < blob_bottom_y_->count() / step; ++i) {
				for (int j = 0; j < step; ++j) {
					blob_bottom_y_->mutable_cpu_data()[i * step + j] = i;
				}				
			}
			blob_bottom_vec_.push_back(blob_bottom_y_);

			blob_top_vec_.push_back(blob_top_loss_);
		}
		virtual ~RankHardLossLayerTest() {
			delete blob_bottom_data_;			
			delete blob_bottom_y_;
			delete blob_top_loss_;
		}		

		Blob<Dtype>* const blob_bottom_data_;
		Blob<Dtype>* const blob_bottom_y_;
		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*>blob_bottom_vec_;
		vector<Blob<Dtype>*>blob_top_vec_;		
	};	

	TYPED_TEST_CASE(RankHardLossLayerTest, TestDtypesAndDevices);

	TYPED_TEST(RankHardLossLayerTest, TestEuclideanDistForward) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		RankHardLossParameter* rank_param_ptr = layer_param.mutable_rank_hard_loss_param();
		rank_param_ptr->set_dist_op(RankHardLossParameter_DistanceOp_Euclidean);
		rank_param_ptr->set_neg_num(400);
		rank_param_ptr->set_pair_size(2);
		rank_param_ptr->set_hard_ratio(0.5f);
		rank_param_ptr->set_margin(0.2f);		

		RankHardLossLayer<Dtype> layer(layer_param);		
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		caffe::rng_t* neg_rng =
			static_cast<caffe::rng_t*>(layer.GetRNGGenerator()->generator());

		// manually compute to compare
		const RankHardLossParameter rank_param = layer_param.rank_hard_loss_param();
		const int neg_num = rank_param.neg_num();
		const int pair_size = rank_param.pair_size();
		const float hard_ratio = rank_param.hard_ratio();
		const float margin = rank_param.margin();

		// hard_ratio + rand_ratio = 1, so changed the original code like this by hzx.
		const int hard_num = neg_num * hard_ratio;
		const int rand_num = neg_num - hard_num;

		const Dtype* bottom_data = this->blob_bottom_data_->cpu_data();
        const Dtype* label = this->blob_bottom_y_->cpu_data();
        //const int count = this->blob_bottom_data_->count();
        const int num = this->blob_bottom_data_->num();
        const int dim = this->blob_bottom_data_->count() / this->blob_bottom_data_->num();

		vector<pair<Dtype, int> > negpairs;	// pair: distance, sample index
		vector<Dtype> cache_loss;
		cache_loss.resize(num * neg_num * pair_size);
		vector<int> sid1;
		vector<int> sid2;

		Dtype total_loss = Dtype(0);
		int total_count = 0;

		vector<int> aShape = this->blob_bottom_data_->shape();
		aShape[0] = 1;
		Blob<Dtype> diff_ap_, diff_an_;
		diff_ap_.Reshape(aShape);
		diff_an_.Reshape(aShape);
		for (int i = 0; i < num; i += pair_size) {
			// calculate anchor - positive
			for (int j = 0; j < dim; ++j) {
				diff_ap_.mutable_cpu_data()[j] = bottom_data[i * dim + j] - bottom_data[(i + 1) * dim + j];
			}
			Dtype dist_sq_ap = Dtype(0);
			for (int j = 0; j < dim; ++j) {
				dist_sq_ap += diff_ap_.cpu_data()[j] * diff_ap_.cpu_data()[j];
			}

			negpairs.clear();
			sid1.clear();
			sid2.clear();
			for (int j = 0; j < num; ++j) {
				// 排除来自同一类别的样本
				if (label[j] == label[i])
					continue;

				// calculate anchor - negative
				for (int k = 0; k < dim; ++k) {
					diff_an_.mutable_cpu_data()[k] = bottom_data[i * dim + k] - bottom_data[j * dim + k];
				}				
				Dtype dist_sq_an = Dtype(0);
				for (int k = 0; k < dim; ++k) {
					dist_sq_an += diff_an_.cpu_data()[k] * diff_an_.cpu_data()[k];
				}				

				// loss
				Dtype loss = std::max(Dtype(0), margin + dist_sq_ap - dist_sq_an);

				// 排除分类正确的样本
 				if (loss == Dtype(0))
 					continue;

				for (int k = 0; k < dim; ++k) {
					diff_an_.mutable_cpu_data()[k] = bottom_data[(i + 1) * dim + k] - bottom_data[j * dim + k];
				}
				dist_sq_an = Dtype(0);
				for (int k = 0; k < dim; ++k) {
					dist_sq_an += diff_an_.cpu_data()[k] * diff_an_.cpu_data()[k];
				}				
				cache_loss[2 * j] = loss;
				cache_loss[2 * j + 1] = std::max(Dtype(0), margin + dist_sq_ap - dist_sq_an);

				negpairs.push_back(make_pair(loss, j));
			}

			// 如果三元组的负样本少于最小所需三元组数目，则将这些三元组全部作为已选择的三元组
			if (negpairs.size() <= neg_num) {
				for (int j = 0; j < static_cast<int>(negpairs.size()); ++j) {
					const int id = negpairs[j].second;
					total_loss += cache_loss[2 * id];
					total_loss += cache_loss[2 * id + 1];
					total_count += 2;
				}
			} else {
				// 按三元组的困难程度进行排列（即负样本与正样本对的距离比较近的）
				std::sort(negpairs.begin(), negpairs.end(), CompareLoss<Dtype>);

				// 将选择的三元组分为两个部分，一部分是达到期望数量的，一部分是多出来的
				// 期望数量的这一部分其实都是比较困难的三元组
				for (int j = 0; j < neg_num; ++j) {
					sid1.push_back(negpairs[j].second);
				}
				for (int j = neg_num; j < int(negpairs.size()); ++j) {
					sid2.push_back(negpairs[j].second);
				}
				shuffle(sid1.begin(), sid1.end(), neg_rng);

				// 从困难的三元组再随机挑选加入指定困难三元组数量的这一部分，也即：
				// 期望数量的这一部分又分为两个部分，困难三元组一部分，随机三元组一部分，并把随机的这一部分加入到多出来的那一组
				// 哪些困难三元组被放入困难三元组部分也是随机选择
				for (int j = 0; j < std::min(hard_num, (int)(sid1.size())); ++j) {
					const int id = sid1[j];
					total_loss += cache_loss[2 * id];
					total_loss += cache_loss[2 * id + 1];
					total_count += 2;					
				}
				for (int j = hard_num; j < int(sid1.size()); ++j) {
					sid2.push_back(sid1[j]);
				}

				// 多出来的那一组再次进行随机
				shuffle(sid2.begin(), sid2.end(), neg_rng);
				for (int j = 0; j < std::min(rand_num, (int)(sid2.size())); ++j) {
					const int id = sid2[j];
					total_loss += cache_loss[2 * id];
					total_loss += cache_loss[2 * id + 1];
					total_count += 2;
				}
			}
		}
		if (total_count > 0)
			total_loss /= (total_count * 2);

		EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], total_loss, 1e-6);
	}

	TYPED_TEST(RankHardLossLayerTest, TestEucildeanDistGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		RankHardLossParameter* rank_param_ptr = layer_param.mutable_rank_hard_loss_param();
		rank_param_ptr->set_dist_op(RankHardLossParameter_DistanceOp_Euclidean);
//		rank_param_ptr->set_dist_op(RankHardLossParameter_DistanceOp_Cosine);
		rank_param_ptr->set_neg_num(400);
		rank_param_ptr->set_pair_size(2);
		rank_param_ptr->set_hard_ratio(0.5f);
		rank_param_ptr->set_margin(0.2f);

		RankHardLossLayer<Dtype> layer(layer_param);
		layer.SetTest(true);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		// check the gradient
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
	}

	TYPED_TEST(RankHardLossLayerTest, TestCosineDistGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		RankHardLossParameter* rank_param_ptr = layer_param.mutable_rank_hard_loss_param();
		rank_param_ptr->set_dist_op(RankHardLossParameter_DistanceOp_Cosine);
		rank_param_ptr->set_neg_num(400);
		rank_param_ptr->set_pair_size(2);
		rank_param_ptr->set_hard_ratio(0.5f);
		rank_param_ptr->set_margin(0.2f);

		RankHardLossLayer<Dtype> layer(layer_param);		
		layer.SetTest(true);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		// check the gradient
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
	}
}  // namespace caffe
