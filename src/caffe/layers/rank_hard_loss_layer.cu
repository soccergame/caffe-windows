#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rank_hard_loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void RankHardLossLayer<Dtype>::DistCosineGenerateTripletsGPU(const vector<Blob<Dtype>*>& bottom) {
	    DistCosineGenerateTripletsCPU(bottom);
		//const RankHardLossParameter rank_param = this->layer_param_.rank_hard_loss_param();
        //const int neg_num = rank_param.neg_num();
        //const int pair_size = rank_param.pair_size();
        //const float hard_ratio = rank_param.hard_ratio();
        //const float margin = rank_param.margin();
		//
        //// hard_ratio + rand_ratio = 1, so changed the original code like this by hzx.
        //const int hard_num = neg_num * hard_ratio;
        //const int rand_num = neg_num - hard_num;
		//
        //const Dtype* bottom_data = bottom[0]->gpu_data();
        //const Dtype* label = bottom[1]->gpu_data();
        //const int count = bottom[0]->count();
        //const int num = bottom[0]->num();
        //const int dim = bottom[0]->count() / bottom[0]->num();
        //Dtype* dis_data = dis_.mutable_gpu_data();        
		//
        ////caffe_set(num * num, Dtype(0), dis_data);
		//
        //// calculate distance, can be changed to matrix form
        //caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, num, dim, Dtype(-1), bottom_data, bottom_data, Dtype(0), dis_data);
        //for (int i = 0; i < num; ++i)
        //    dis_data[i * num + i] = Dtype(0);
		//
        ///*for (int i = 0; i < num; ++i) {
		//	for (int j = i + 1; j < num; ++j) {
        //        const Dtype* fea1 = bottom_data + i * dim;
        //        const Dtype* fea2 = bottom_data + j * dim;
        //        Dtype ts = caffe_gpu_dot(dim, fea1, fea2);
        //        
        //        dis_data[i * num + j] = -ts;
        //        dis_data[j * num + i] = -ts;
        //    }
        //}*/
		//
        //// Select samples
        //// Here change original float type to Dtype to avoid warning...by hzx
        //vector<pair<Dtype, int> > negpairs;	// pair: distance, sample index
        //vector<int> sid1;
        //vector<int> sid2;
		//
		//vector<Dtype> cache_loss;
		//cache_loss.resize(num * 2, Dtype(0));
		//
		//selected_triplets_.clear();
		//triplet_losses_.clear();
        //for (int i = 0; i < num; i += pair_size) {
		//	for (int j = 1; j < pair_size; ++j) {
		//		CHECK(label[i] == label[i + j]) << "The labels within each pair must be the same.";
		//	}
		//
		//	caffe_gpu_set(num * 2, Dtype(0), &cache_loss[0]);
		//
        //    negpairs.clear();
        //    sid1.clear();
        //    sid2.clear();
        //    for (int j = 0; j < num; ++j) {
        //        // 排除来自同一类别的样本
        //        if (label[j] == label[i])
        //            continue;
        //        Dtype tloss = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
        //        // 排除分类正确的样本
        //        if (tloss == 0) 
        //            continue;
		//
		//		negpairs.push_back(make_pair(tloss, j));
		//
		//		Dtype tloss2 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));
		//		cache_loss[2 * j] = tloss;
		//		cache_loss[2 * j + 1] = tloss2;
        //    }
		//
        //    // 如果三元组的负样本少于最小所需三元组数目，则将这些三元组全部作为已选择的三元组
        //    if (negpairs.size() <= neg_num) {
        //        for (int j = 0; j < static_cast<int>(negpairs.size()); ++j) {
        //            const int id = negpairs[j].second;                    
		//			selected_triplets_.push_back(std::make_pair(i, id));
		//			triplet_losses_.push_back(cache_loss[2 * id]);
		//			triplet_losses_.push_back(cache_loss[2 * id + 1]);
        //        }                
		//	} else {
		//		// 按三元组的困难程度进行排列（即负样本与正样本对的距离比较近的）
		//		std::sort(negpairs.begin(), negpairs.end(), CompareLoss<Dtype>);
		//
		//		// 将选择的三元组分为两个部分，一部分是达到期望数量的，一部分是多出来的
		//		// 期望数量的这一部分其实都是比较困难的三元组
		//		for (int j = 0; j < neg_num; ++j) {
		//			sid1.push_back(negpairs[j].second);
		//		}
		//		for (int j = neg_num; j < int(negpairs.size()); ++j) {
		//			sid2.push_back(negpairs[j].second);
		//		}
		//		std::random_shuffle(sid1.begin(), sid1.end(), myrandom);
		//
		//		// 从困难的三元组再随机挑选加入指定困难三元组数量的这一部分，也即：
		//		// 期望数量的这一部分又分为两个部分，困难三元组一部分，随机三元组一部分，并把随机的这一部分加入到多出来的那一组
		//		// 哪些困难三元组被放入困难三元组部分也是随机选择
		//		for (int j = 0; j < min(hard_num, (int)(sid1.size())); ++j) {
		//			const int id = sid1[j];
		//			selected_triplets_.push_back(std::make_pair(i, id));
		//			triplet_losses_.push_back(cache_loss[2 * id]);
		//			triplet_losses_.push_back(cache_loss[2 * id + 1]);
		//		}
		//		for (int j = hard_num; j < int(sid1.size()); ++j) {
		//			sid2.push_back(sid1[j]);
		//		}
		//
		//		// 多出来的那一组再次进行随机
		//		std::random_shuffle(sid2.begin(), sid2.end(), myrandom);
		//		for (int j = 0; j < min(rand_num, (int)(sid2.size())); ++j) {
		//			const int id = sid2[j];
		//			selected_triplets_.push_back(std::make_pair(i, id));
		//			triplet_losses_.push_back(cache_loss[2 * id]);
		//			triplet_losses_.push_back(cache_loss[2 * id + 1]);
		//		}
		//	}
		//}
    }

	template <typename Dtype>
	void RankHardLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {
		Forward_cpu(bottom, top);
		//DistCosineGenerateTripletsGPU(bottom);
		//
		//const Dtype* dis_data = dis_.gpu_data();
	    //Dtype loss = 0;
		//int cnt = 0;
		//
		//// fill loss		
		//for (int index = 0; index < static_cast<int>(selected_triplets_.size()); ++index) {
		//	Dtype tloss1 = triplet_losses_[cnt++];
		//	Dtype tloss2 = triplet_losses_[cnt++];			
		//	
		//	loss += tloss1 + tloss2;			
		//}
		//
	    //loss = loss / cnt;
	    //top[0]->mutable_gpu_data()[0] = loss;
	}

	template <typename Dtype>
	void RankHardLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Backward_cpu(top, propagate_down, bottom);
	    //const Dtype* bottom_data = bottom[0]->gpu_data();
        //Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        //
		//const int count = bottom[0]->count();
        //const int num = bottom[0]->num();
        //const int dim = bottom[0]->count() / bottom[0]->num();
        //
        //const Dtype* dis_data = dis_.gpu_data();        
		//
		//caffe_gpu_set(count, Dtype(0), bottom_diff);
		//
		//int cnt = 0;
		//for (int index = 0; index < static_cast<int>(selected_triplets_.size()); ++index) {
		//	const int i = selected_triplets_[index].first;
		//	const int j = selected_triplets_[index].second;
		//
		//	const Dtype* fori = bottom_data + i * dim;
		//	const Dtype* fpos = bottom_data + (i + 1) * dim;
		//
		//	Dtype* fori_diff = bottom_diff + i * dim;
		//	Dtype* fpos_diff = bottom_diff + (i + 1) * dim;
		//
		//	Dtype tloss1 = triplet_losses_[cnt++];
		//	Dtype tloss2 = triplet_losses_[cnt++];
		//
		//	const Dtype* fneg = bottom_data + j * dim;
		//	Dtype* fneg_diff = bottom_diff + j * dim;
		//	if (tloss1 > 0) {
		//		caffe_gpu_add(dim, fori_diff, fneg, fori_diff);
		//		caffe_gpu_sub(dim, fori_diff, fpos, fori_diff);
		//		caffe_gpu_sub(dim, fpos_diff, fori, fpos_diff);
		//		caffe_gpu_add(dim, fneg_diff, fori, fneg_diff);
		//	}
		//	if (tloss2 > 0) {
		//		caffe_gpu_sub(dim, fori_diff, fpos, fori_diff);
		//		caffe_gpu_add(dim, fpos_diff, fneg, fpos_diff);
		//		caffe_gpu_sub(dim, fpos_diff, fori, fpos_diff);
		//		caffe_gpu_add(dim, fneg_diff, fpos, fneg_diff);
		//	}
		//}
		//caffe_gpu_scale(count, Dtype(1.0) / cnt, bottom_diff, bottom_diff);
	}

INSTANTIATE_LAYER_GPU_FUNCS(RankHardLossLayer);

}  // namespace caffe