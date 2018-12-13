#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rank_hard_loss_layer.hpp"

using std::max;

using namespace std;
using namespace cv;

namespace caffe {
	template <typename Dtype>
	bool CompareLoss(const pair<Dtype, int> &left, const pair<Dtype, int> &right) {
		return left.first > right.first ? true : false;
	}

	template <typename Dtype>
	void RankHardLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		const unsigned int neg_sel_seed = caffe_rng_rand();
		neg_sel_rng_.reset(new Caffe::RNG(neg_sel_seed));		
	}
    template <typename Dtype>
    void RankHardLossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::Reshape(bottom, top);
		const RankHardLossParameter rank_param = this->layer_param_.rank_hard_loss_param();
		const int neg_num = rank_param.neg_num();
		//const int pair_size = rank_param.pair_size();
		//const float hard_ratio = rank_param.hard_ratio();
		//const float margin = rank_param.margin();

		triplet_losses_.Reshape(bottom[0]->num(), neg_num, 2, 1);
		selected_triplets_.Reshape(bottom[0]->num(), neg_num, 2, 1);
    }

	template <typename Dtype>
	void RankHardLossLayer<Dtype>::DistEuclideanTripletSelectCPU(
		const vector< Blob<Dtype>* > &bottom) {
		const RankHardLossParameter rank_param = this->layer_param_.rank_hard_loss_param();
		const int neg_num = rank_param.neg_num();
		const int pair_size = rank_param.pair_size();
		const float hard_ratio = rank_param.hard_ratio();
		const float margin = rank_param.margin();

		// hard_ratio + rand_ratio = 1, so changed the original code like this by hzx.
		const int hard_num = neg_num * hard_ratio;
		const int rand_num = neg_num - hard_num;

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		//const int count = bottom[0]->count();
		const int num = bottom[0]->num();
		const int dim = bottom[0]->count() / bottom[0]->num();

		caffe::rng_t* neg_rng =
			static_cast<caffe::rng_t*>(neg_sel_rng_->generator());

		vector<pair<Dtype, int> > negpairs;	// pair: loss, sample index
		Blob<Dtype> cache_loss;
		cache_loss.Reshape(num, 2, 1, 1);

		vector<int> sid1;
		vector<int> sid2;		

		vector<int> aShape = bottom[0]->shape();
		aShape[0] = 1;
		Blob<Dtype> diff_ap_, diff_an_;
		diff_ap_.Reshape(aShape);
		diff_an_.Reshape(aShape);
		
		Dtype * triplet_loss = triplet_losses_.mutable_cpu_data();
		triplets_num_ = 0;
		for (int i = 0; i < num; i += pair_size) {
			for (int j = 1; j < pair_size; ++j) {
				CHECK(label[i] == label[i + j]) << "The labels within each pair must be the same.";
			}

			// calculate anchor - positive
			caffe_sub(dim,
				bottom_data + i * dim,				// anchor
				bottom_data + (i + 1) * dim,		// positive
				diff_ap_.mutable_cpu_data());		// a_i - p_i
			Dtype dist_sq_ap = caffe_cpu_dot(dim, diff_ap_.cpu_data(), diff_ap_.cpu_data());			

			negpairs.clear();
			sid1.clear();
			sid2.clear();
			for (int j = 0; j < num; ++j) {
				// 排除来自同一类别的样本
				if (label[j] == label[i])
					continue;

				// calculate anchor - negative
				caffe_sub(dim,
					bottom_data + i * dim,			// anchor
					bottom_data + j * dim,			// negative
					diff_an_.mutable_cpu_data());	// a_i - n_i
				Dtype dist_sq_an = caffe_cpu_dot(dim, diff_an_.cpu_data(), diff_an_.cpu_data());

				// loss
				Dtype loss = std::max(Dtype(0), margin + dist_sq_ap - dist_sq_an);

				// 排除分类正确的样本
 				if (!test_ && loss == Dtype(0))
 					continue;

				caffe_sub(dim, 
					bottom_data + (i + 1) * dim,	// anchor
					bottom_data + j * dim,			// negative
					diff_an_.mutable_cpu_data());	// a_i - n_i
				dist_sq_an = caffe_cpu_dot(dim, diff_an_.cpu_data(), diff_an_.cpu_data());
				cache_loss.mutable_cpu_data()[2 * j] = loss;
				cache_loss.mutable_cpu_data()[2 * j + 1] = std::max(Dtype(0), margin + dist_sq_ap - dist_sq_an);

				negpairs.push_back(make_pair(loss, j));
			}

			// 如果三元组的负样本少于最小所需三元组数目，则将这些三元组全部作为已选择的三元组
			if (negpairs.size() <= neg_num) {
				for (int j = 0; j < static_cast<int>(negpairs.size()); ++j) {
					const int id = negpairs[j].second;

					selected_triplets_.mutable_cpu_data()[triplets_num_] = i;
					triplet_loss[2 * triplets_num_] = cache_loss.cpu_data()[2 * id];

					selected_triplets_.mutable_cpu_diff()[triplets_num_] = id;
					triplet_loss[2 * triplets_num_ + 1] = cache_loss.cpu_data()[2 * id + 1];
					triplets_num_ += 1;
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
				for (int j = 0; j < min(hard_num, (int)(sid1.size())); ++j) {
					const int id = sid1[j];

					selected_triplets_.mutable_cpu_data()[triplets_num_] = i;
					triplet_loss[2 * triplets_num_] = cache_loss.cpu_data()[2 * id];

					selected_triplets_.mutable_cpu_diff()[triplets_num_] = id;
					triplet_loss[2 * triplets_num_ + 1] = cache_loss.cpu_data()[2 * id + 1];
					triplets_num_ += 1;
				}
				for (int j = hard_num; j < int(sid1.size()); ++j) {
					sid2.push_back(sid1[j]);
				}

				// 多出来的那一组再次进行随机
				shuffle(sid2.begin(), sid2.end(), neg_rng);
				for (int j = 0; j < min(rand_num, (int)(sid2.size())); ++j) {
					const int id = sid2[j];

					selected_triplets_.mutable_cpu_data()[triplets_num_] = i;
					triplet_loss[2 * triplets_num_] = cache_loss.cpu_data()[2 * id];

					selected_triplets_.mutable_cpu_diff()[triplets_num_] = id;
					triplet_loss[2 * triplets_num_ + 1] = cache_loss.cpu_data()[2 * id + 1];
					triplets_num_ += 1;
				}
			}
		}		
	}
	
    // 目前来看，可优化成为gpu运算的部分，都集中在这个函数中间的距离计算函数上
    template <typename Dtype>
	void RankHardLossLayer<Dtype>::DistCosineGenerateTripletsCPU(const vector<Blob<Dtype>*>& bottom) {
		const RankHardLossParameter rank_param = this->layer_param_.rank_hard_loss_param();
        const int neg_num = rank_param.neg_num();
        const int pair_size = rank_param.pair_size();
        const float hard_ratio = rank_param.hard_ratio();
        const float margin = rank_param.margin();

		caffe::rng_t* neg_rng =
			static_cast<caffe::rng_t*>(neg_sel_rng_->generator());

        // hard_ratio + rand_ratio = 1, so changed the original code like this by hzx.
        const int hard_num = neg_num * hard_ratio;
        const int rand_num = neg_num - hard_num;

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* label = bottom[1]->cpu_data();
        //const int count = bottom[0]->count();
        const int num = bottom[0]->num();
        const int dim = bottom[0]->count() / bottom[0]->num();
		Blob<Dtype> sample_dist;
		sample_dist.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
		Dtype* dis_data = sample_dist.mutable_cpu_data();
        

        // calculate distance, can be changed to matrix form
        caffe_cpu_gemm(CblasNoTrans, CblasTrans, num, num, dim, Dtype(-1), bottom_data, bottom_data, Dtype(0), dis_data);
        //for (int i = 0; i < num; ++i)
        //    dis_data[i * num + i] = Dtype(0);        

        // Select samples
        // Here change original float type to Dtype to avoid warning...by hzx
        vector<pair<Dtype, int> > negpairs;	// pair: distance, sample index
        vector<int> sid1;
        vector<int> sid2;

		Blob<Dtype> cache_loss;
		cache_loss.Reshape(num, 2, 1, 1);
		
		Dtype * triplet_loss = triplet_losses_.mutable_cpu_data();
		triplets_num_ = 0;
        for (int i = 0; i < num; i += pair_size) {
			for (int j = 1; j < pair_size; ++j) {
				CHECK(label[i] == label[i + j]) << "The labels within each pair must be the same.";
			}

			caffe_set(num * 2, Dtype(0), cache_loss.mutable_cpu_data());

            negpairs.clear();
            sid1.clear();
            sid2.clear();
            for (int j = 0; j < num; ++j) {
                // 排除来自同一类别的样本
                if (label[j] == label[i])
                    continue;
                Dtype tloss = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));

                // 排除分类正确的样本
				if (!test_ && tloss == 0)
                    continue;

				negpairs.push_back(make_pair(tloss, j));

				Dtype tloss2 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));
				cache_loss.mutable_cpu_data()[2 * j] = tloss;
				cache_loss.mutable_cpu_data()[2 * j + 1] = tloss2;
            }

            // 如果三元组的负样本少于最小所需三元组数目，则将这些三元组全部作为已选择的三元组
            if (negpairs.size() <= neg_num) {
                for (int j = 0; j < static_cast<int>(negpairs.size()); ++j) {
                    const int id = negpairs[j].second;                    
					
					selected_triplets_.mutable_cpu_data()[triplets_num_] = i;
					triplet_loss[2 * triplets_num_] = cache_loss.cpu_data()[2 * id];

					selected_triplets_.mutable_cpu_diff()[triplets_num_] = id;
					triplet_loss[2 * triplets_num_ + 1] = cache_loss.cpu_data()[2 * id + 1];
					triplets_num_ += 1;
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
				for (int j = 0; j < min(hard_num, (int)(sid1.size())); ++j) {
					const int id = sid1[j];
					
					selected_triplets_.mutable_cpu_data()[triplets_num_] = i;
					triplet_loss[2 * triplets_num_] = cache_loss.cpu_data()[2 * id];

					selected_triplets_.mutable_cpu_diff()[triplets_num_] = id;
					triplet_loss[2 * triplets_num_ + 1] = cache_loss.cpu_data()[2 * id + 1];
					triplets_num_ += 1;
				}
				for (int j = hard_num; j < int(sid1.size()); ++j) {
					sid2.push_back(sid1[j]);
				}

				// 多出来的那一组再次进行随机
				shuffle(sid2.begin(), sid2.end(), neg_rng);
				for (int j = 0; j < min(rand_num, (int)(sid2.size())); ++j) {
					const int id = sid2[j];
					
					selected_triplets_.mutable_cpu_data()[triplets_num_] = i;
					triplet_loss[2 * triplets_num_] = cache_loss.cpu_data()[2 * id];

					selected_triplets_.mutable_cpu_diff()[triplets_num_] = id;
					triplet_loss[2 * triplets_num_ + 1] = cache_loss.cpu_data()[2 * id + 1];
					triplets_num_ += 1;
				}
			}
		}
    }

    // Forward is only simple scalar computation, doesn't need to change to gpu
    template <typename Dtype>
    void RankHardLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const RankHardLossParameter rank_param = this->layer_param_.rank_hard_loss_param();
		const RankHardLossParameter::DistanceOp operation = rank_param.dist_op();

		Dtype loss = 0;
		int cnt = 0;
		
		triplets_num_ = 0;
		if (operation == RankHardLossParameter_DistanceOp_Euclidean) {
			DistEuclideanTripletSelectCPU(bottom);			
		}
		else if (operation == RankHardLossParameter_DistanceOp_Cosine) {
			DistCosineGenerateTripletsCPU(bottom);		// generate triplets online
		}

		const Dtype * triplet_loss = triplet_losses_.cpu_data();
		// fill loss		
		for (int index = 0; index < triplets_num_; ++index) {
			Dtype tloss1 = triplet_loss[cnt++];
			Dtype tloss2 = triplet_loss[cnt++];

			loss += tloss1 + tloss2;
		}
		if (cnt > 0)
			loss /= cnt;
		if (operation == RankHardLossParameter_DistanceOp_Euclidean)
			loss /= Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
    }

    template <typename Dtype>
    void RankHardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}

		const RankHardLossParameter rank_param = this->layer_param_.rank_hard_loss_param();
		const RankHardLossParameter::DistanceOp operation = rank_param.dist_op();

		const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        
		//const Dtype *top_diff = top[0]->mutable_cpu_diff();

		const int count = bottom[0]->count();
        //const int num = bottom[0]->num();
        const int dim = bottom[0]->count() / bottom[0]->num();      

		caffe_set(count, Dtype(0), bottom_diff);

		const Dtype * triplet_loss = triplet_losses_.cpu_data();

		int cnt = 0;
		if (operation == RankHardLossParameter_DistanceOp_Euclidean) {
			vector<int> aShape = bottom[0]->shape();
			aShape[0] = 1;
			Blob<Dtype> diff_ap_, diff_an_, diff_pn_;
			diff_ap_.Reshape(aShape);
			diff_an_.Reshape(aShape);
			diff_pn_.Reshape(aShape);

			for (int index = 0; index < triplets_num_; ++index) {
				const int i = selected_triplets_.cpu_data()[index];
				const int j = selected_triplets_.cpu_diff()[index];

				const Dtype* fori = bottom_data + i * dim;
				const Dtype* fpos = bottom_data + (i + 1) * dim;
				const Dtype* fneg = bottom_data + j * dim;

				Dtype* fori_diff = bottom_diff + i * dim;
				Dtype* fpos_diff = bottom_diff + (i + 1) * dim;
				Dtype* fneg_diff = bottom_diff + j * dim;

				caffe_sub(dim, fori, fpos, diff_ap_.mutable_cpu_data());		// a - p
				caffe_sub(dim, fpos, fneg, diff_pn_.mutable_cpu_data());		// p - n
				caffe_sub(dim, fori, fneg, diff_an_.mutable_cpu_data());		// a - n
				const Dtype tloss1 = triplet_loss[cnt++];
				if (tloss1 > Dtype(0)) {
					// triplet: fori, fpos, fneg
					caffe_sub(dim, fori_diff, diff_pn_.cpu_data(), fori_diff);	// a
					caffe_sub(dim, fpos_diff, diff_ap_.cpu_data(), fpos_diff);	// p
					caffe_add(dim, fneg_diff, diff_an_.cpu_data(), fneg_diff);	// n
				}

				const Dtype tloss2 = triplet_loss[cnt++];
				if (tloss2 > Dtype(0)) {
					// triplet: fpos, fori, fneg: a, p, n
					caffe_sub(dim, fpos_diff, diff_an_.cpu_data(), fpos_diff);	// a
					caffe_add(dim, fori_diff, diff_ap_.cpu_data(), fori_diff);	// p
					caffe_add(dim, fneg_diff, diff_pn_.cpu_data(), fneg_diff);	// n
				}
			}
		}
		else if (operation == RankHardLossParameter_DistanceOp_Cosine) {
			for (int index = 0; index < triplets_num_; ++index) {
				const int i = selected_triplets_.cpu_data()[index];
				const int j = selected_triplets_.cpu_diff()[index];

				const Dtype* fori = bottom_data + i * dim;
				const Dtype* fpos = bottom_data + (i + 1) * dim;
				const Dtype* fneg = bottom_data + j * dim;

				Dtype* fori_diff = bottom_diff + i * dim;
				Dtype* fpos_diff = bottom_diff + (i + 1) * dim;
				Dtype* fneg_diff = bottom_diff + j * dim;

				const Dtype tloss1 = triplet_loss[cnt++];
				const Dtype tloss2 = triplet_loss[cnt++];
				if (tloss1 > 0) {
					caffe_add(dim, fori_diff, fneg, fori_diff);
					caffe_sub(dim, fori_diff, fpos, fori_diff);
					caffe_sub(dim, fpos_diff, fori, fpos_diff);
					caffe_add(dim, fneg_diff, fori, fneg_diff);
				}
				if (tloss2 > 0) {
					caffe_sub(dim, fori_diff, fpos, fori_diff);
					caffe_add(dim, fpos_diff, fneg, fpos_diff);
					caffe_sub(dim, fpos_diff, fori, fpos_diff);
					caffe_add(dim, fneg_diff, fpos, fneg_diff);
				}
			}
		}

		if (cnt > 0)
		{
			const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(cnt);
			caffe_cpu_scale(count, alpha, bottom_diff, bottom_diff);
		}
    }

#ifdef CPU_ONLY
    STUB_GPU(RankHardLossLayer);
#endif

    INSTANTIATE_CLASS(RankHardLossLayer);
    REGISTER_LAYER_CLASS(RankHardLoss);
}  // namespace caffe
