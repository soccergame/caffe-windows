/*
* test_rank_hard_loss_layer.cpp
*
* Created on: July 1, 2016
*     Author: THID@Hisign
*/

#include<algorithm>
#include<cmath>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/label_specific_add_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

    template <typename TypeParam>
    class LabelSpecificAddLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;
    protected:
        LabelSpecificAddLayerTest()
            : blob_bottom_data_(new Blob<Dtype>(6, 3, 1, 1)),
            blob_bottom_y_(new Blob<Dtype>(6, 1, 1, 1)),
            blob_top_loss_(new Blob<Dtype>()) {

            // fill the values
            FillerParameter filler_param;
            filler_param.set_min(-1.0);
            filler_param.set_max(1.0);  // distances~=1.0 to test both sides ofmargin
            UniformFiller<Dtype> filler(filler_param);
            Blob<Dtype> *weight = new Blob<Dtype>(3, 8, 1, 1);
            Blob<Dtype> *fea = new Blob<Dtype>(6, 8, 1, 1);
            filler.Fill(weight);
            filler.Fill(fea);
            for (int i1 = 0; i1 < weight->num(); ++i1) {
                Dtype normsqr = 0;
                for (int i2 = 0; i2 < weight->channels(); ++i2) {
                    for (int i3 = 0; i3 < weight->height(); ++i3) {
                        for (int i4 = 0; i4 < weight->width(); ++i4) {
                            normsqr += weight->data_at(i1, i2, i3, i4) * weight->data_at(i1, i2, i3, i4);
                        }
                    }
                }
                normsqr = 1. / (sqrt(normsqr) + 1e-6);
                for (int i2 = 0; i2 < weight->channels(); ++i2) {
                    weight->mutable_cpu_data()[i1 * weight->channels() + i2] = weight->mutable_cpu_data()[i1 * weight->channels() + i2] * normsqr;
                }
            }

            for (int i1 = 0; i1 < fea->num(); ++i1) {
                Dtype normsqr = 0;
                for (int i2 = 0; i2 < fea->channels(); ++i2) {
                    for (int i3 = 0; i3 < fea->height(); ++i3) {
                        for (int i4 = 0; i4 < fea->width(); ++i4) {
                            normsqr += fea->data_at(i1, i2, i3, i4) * fea->data_at(i1, i2, i3, i4);
                        }
                    }
                }
                normsqr = 1. / (sqrt(normsqr) + 1e-6);
                for (int i2 = 0; i2 < fea->channels(); ++i2) {
                    fea->mutable_cpu_data()[i1 * fea->channels() + i2] = fea->mutable_cpu_data()[i1 * fea->channels() + i2] * normsqr;
                }
            }

            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 6, 8, 3, (Dtype)1.,
                fea->cpu_data(), weight->cpu_data(), (Dtype)0.,
                blob_bottom_data_->mutable_cpu_data());
            blob_bottom_vec_.push_back(blob_bottom_data_);

            for (int i = 0; i < blob_bottom_y_->count(); ++i) {
                blob_bottom_y_->mutable_cpu_data()[i] = i % 3;
            }
            blob_bottom_vec_.push_back(blob_bottom_y_);
            blob_top_vec_.push_back(blob_top_loss_);
        }
        virtual ~LabelSpecificAddLayerTest() {
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

    TYPED_TEST_CASE(LabelSpecificAddLayerTest, TestDtypesAndDevices);

    TYPED_TEST(LabelSpecificAddLayerTest, TestGradient) {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        LabelSpecificAddParameter* label_param_ptr = layer_param.mutable_label_specific_add_param();
        //		rank_param_ptr->set_dist_op(RankHardLossParameter_DistanceOp_Euclidean);
        //		rank_param_ptr->set_dist_op(RankHardLossParameter_DistanceOp_Cosine);
        label_param_ptr->set_bias(-0.25);

        LabelSpecificAddLayer<Dtype> layer(layer_param);
        //layer.SetTest(true);
        //layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        GradientChecker<Dtype> checker(1e-2, 1e-3);
        // check the gradient
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
    }

}  // namespace caffe
