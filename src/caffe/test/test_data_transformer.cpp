#ifdef USE_OPENCV
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe {

    void FillDatum(const int label, const int channels, const int height,
        const int width, const bool unique_pixels, Datum * datum) {
        datum->set_label(label);
        datum->set_channels(channels);
        datum->set_height(height);
        datum->set_width(width);
        int size = channels * height * width;
        std::string* data = datum->mutable_data();
        for (int j = 0; j < size; ++j) {
            int datum = unique_pixels ? j : label;
            data->push_back(static_cast<uint8_t>(datum));
        }
    }

    void FillPatchDatum(const int label, const int channels, const int height,
        const int width, Datum * datum) {
        datum->set_label(label);
        datum->set_channels(channels);
        datum->set_height(height);
        datum->set_width(width);
        int size = height * width;
        std::string* data = datum->mutable_data();
        for (int c = 0; c < channels; ++c) {
            for (int j = 0; j < size; ++j) {
                int datum = j * (c + 1);
                data->push_back(static_cast<uint8_t>(datum));
            }
        }
    }

    template <typename Dtype>
    class DataTransformTest : public ::testing::Test {
    protected:
        DataTransformTest()
            : seed_(1701),
            num_iter_(10) {}

        int NumSequenceMatches(const TransformationParameter transform_param,
            const Datum& datum, Phase phase) {
            // Get crop sequence with Caffe seed 1701.
            DataTransformer<Dtype> transformer(transform_param, phase);
            const int crop_size = transform_param.crop_size();
            Caffe::set_random_seed(seed_);
            transformer.InitRand();
            Blob<Dtype> blob(1, datum.channels(), datum.height(), datum.width());
            if (transform_param.crop_size() > 0) {
                blob.Reshape(1, datum.channels(), crop_size, crop_size);
            }

            vector<vector<Dtype> > crop_sequence;
            for (int iter = 0; iter < this->num_iter_; ++iter) {
                vector<Dtype> iter_crop_sequence;
                transformer.Transform(datum, &blob);
                for (int j = 0; j < blob.count(); ++j) {
                    iter_crop_sequence.push_back(blob.cpu_data()[j]);
                }
                crop_sequence.push_back(iter_crop_sequence);
            }
            // Check if the sequence differs from the previous
            int num_sequence_matches = 0;
            for (int iter = 0; iter < this->num_iter_; ++iter) {
                vector<Dtype> iter_crop_sequence = crop_sequence[iter];
                transformer.Transform(datum, &blob);
                for (int j = 0; j < blob.count(); ++j) {
                    num_sequence_matches += (crop_sequence[iter][j] == blob.cpu_data()[j]);
                }
            }
            return num_sequence_matches;
        }

        int seed_;
        int num_iter_;
    };

    TYPED_TEST_CASE(DataTransformTest, TestDtypes);

    TYPED_TEST(DataTransformTest, TestEmptyTransform) {
        TransformationParameter transform_param;
        const bool unique_pixels = false;  // all pixels the same equal to label
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        Blob<TypeParam> blob(1, channels, height, width);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        transformer.InitRand();
        transformer.Transform(datum, &blob);
        EXPECT_EQ(blob.num(), 1);
        EXPECT_EQ(blob.channels(), datum.channels());
        EXPECT_EQ(blob.height(), datum.height());
        EXPECT_EQ(blob.width(), datum.width());
        for (int j = 0; j < blob.count(); ++j) {
            EXPECT_EQ(blob.cpu_data()[j], label);
        }
    }

    TYPED_TEST(DataTransformTest, TestEmptyTransformUniquePixels) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        Blob<TypeParam> blob(1, 3, 4, 5);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        transformer.InitRand();
        transformer.Transform(datum, &blob);
        EXPECT_EQ(blob.num(), 1);
        EXPECT_EQ(blob.channels(), datum.channels());
        EXPECT_EQ(blob.height(), datum.height());
        EXPECT_EQ(blob.width(), datum.width());
        for (int j = 0; j < blob.count(); ++j) {
            EXPECT_EQ(blob.cpu_data()[j], j);
        }
    }

    TYPED_TEST(DataTransformTest, TestCropSize) {
        TransformationParameter transform_param;
        const bool unique_pixels = false;  // all pixels the same equal to label
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int crop_size = 2;

        transform_param.set_crop_size(crop_size);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        transformer.InitRand();
        Blob<TypeParam> blob(1, channels, crop_size, crop_size);
        for (int iter = 0; iter < this->num_iter_; ++iter) {
            transformer.Transform(datum, &blob);
            EXPECT_EQ(blob.num(), 1);
            EXPECT_EQ(blob.channels(), datum.channels());
            EXPECT_EQ(blob.height(), crop_size);
            EXPECT_EQ(blob.width(), crop_size);
            for (int j = 0; j < blob.count(); ++j) {
                EXPECT_EQ(blob.cpu_data()[j], label);
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestPatchSize) {
        TransformationParameter transform_param;
        const int label = 0;
        const int channels = 3;
        const int height = 6;
        const int width = 6;
        const int patch_height = 3;
        const int patch_width = 3;

        transform_param.add_patch_center_x(1);
        transform_param.add_patch_center_y(1);
        transform_param.add_patch_center_x(4);
        transform_param.add_patch_center_y(4);
        transform_param.set_patch_height(patch_height);
        transform_param.set_patch_width(patch_width);

        Datum datum;
        FillPatchDatum(label, channels, height, width, &datum);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        //transformer.InitRand();
        Blob<TypeParam> blob(1, 2 * channels, patch_height, patch_width);
        for (int iter = 0; iter < this->num_iter_; ++iter) {
            transformer.Transform(datum, &blob);
            EXPECT_EQ(blob.num(), 1);
            EXPECT_EQ(blob.channels(), 2 * datum.channels());
            EXPECT_EQ(blob.height(), patch_height);
            EXPECT_EQ(blob.width(), patch_width);
            for (int p = 0; p < 2; ++p) {
                for (int c = 0; c < channels; ++c) {
                    for (int h = 0; h < patch_height; ++h) {
                        for (int w = 0; w < patch_width; ++w) {
                            int blob_index = (p*channels + c) * patch_height * patch_width + h * patch_width + w;
                            int data_number = (h * width + w) * (c + 1);
                            EXPECT_EQ(blob.cpu_data()[blob_index], data_number + 21 * p * (c + 1));
                        }
                    }
                }
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestCropTrain) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int crop_size = 2;
        const int size = channels * crop_size * crop_size;

        transform_param.set_crop_size(crop_size);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
        EXPECT_LT(num_matches, size * this->num_iter_);
    }

    TYPED_TEST(DataTransformTest, TestCropTest) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int crop_size = 2;
        const int size = channels * crop_size * crop_size;

        transform_param.set_crop_size(crop_size);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
        EXPECT_EQ(num_matches, size * this->num_iter_);
    }

    TYPED_TEST(DataTransformTest, TestMirrorTrain) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int size = channels * height * width;

        transform_param.set_mirror(true);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        int num_matches = this->NumSequenceMatches(transform_param, datum, TRAIN);
        EXPECT_LT(num_matches, size * this->num_iter_);
    }

    TYPED_TEST(DataTransformTest, TestMirrorTest) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int size = channels * height * width;

        transform_param.set_mirror(true);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        int num_matches = this->NumSequenceMatches(transform_param, datum, TEST);
        EXPECT_LT(num_matches, size * this->num_iter_);
    }

    TYPED_TEST(DataTransformTest, TestCropMirrorTrain) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int crop_size = 2;

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        transform_param.set_crop_size(crop_size);
        int num_matches_crop = this->NumSequenceMatches(
            transform_param, datum, TRAIN);

        transform_param.set_mirror(true);
        int num_matches_crop_mirror =
            this->NumSequenceMatches(transform_param, datum, TRAIN);
        // When doing crop and mirror we expect less num_matches than just crop
        EXPECT_LE(num_matches_crop_mirror, num_matches_crop);
    }

    TYPED_TEST(DataTransformTest, TestCropMirrorTest) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int crop_size = 2;

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        transform_param.set_crop_size(crop_size);
        int num_matches_crop = this->NumSequenceMatches(transform_param, datum, TEST);

        transform_param.set_mirror(true);
        int num_matches_crop_mirror =
            this->NumSequenceMatches(transform_param, datum, TEST);
        // When doing crop and mirror we expect less num_matches than just crop
        EXPECT_LT(num_matches_crop_mirror, num_matches_crop);
    }

    TYPED_TEST(DataTransformTest, TestCropPatchTrain) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 9;
        const int width = 9;
        const int crop_size = 5;
        const int patch_height = 3;
        const int patch_width = 3;

        transform_param.add_patch_center_x(3);
        transform_param.add_patch_center_y(3);
        transform_param.set_crop_size(crop_size);
        transform_param.set_patch_height(patch_height);
        transform_param.set_patch_width(patch_width);

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);

        DataTransformer<TypeParam> transformer(transform_param, TRAIN);

        Blob<TypeParam> blob(1, channels, patch_height, patch_width);

        for (int iter = 0; iter < this->num_iter_; ++iter) {
            Caffe::set_random_seed(this->seed_);
            transformer.InitRand();
            transformer.Transform(datum, &blob);
            EXPECT_EQ(blob.num(), 1);
            EXPECT_EQ(blob.channels(), datum.channels());
            EXPECT_EQ(blob.height(), patch_height);
            EXPECT_EQ(blob.width(), patch_width);
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < patch_height; ++h) {
                    for (int w = 0; w < patch_width; ++w) {
                        int blob_index = c * patch_height * patch_width + h * patch_width + w;
                        EXPECT_EQ(blob.cpu_data()[blob_index] - blob.cpu_data()[0], c*height*width + h*width + w);
                    }
                }
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestCropPatchTest) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 9;
        const int width = 9;
        const int crop_size = 5;
        const int patch_height = 3;
        const int patch_width = 3;

        transform_param.add_patch_center_x(2);
        transform_param.add_patch_center_y(2);
        transform_param.set_crop_size(crop_size);
        transform_param.set_patch_height(patch_height);
        transform_param.set_patch_width(patch_width);

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);

        DataTransformer<TypeParam> transformer(transform_param, TEST);

        Blob<TypeParam> blob(1, channels, patch_height, patch_width);

        for (int iter = 0; iter < this->num_iter_; ++iter) {
            transformer.Transform(datum, &blob);
            EXPECT_EQ(blob.num(), 1);
            EXPECT_EQ(blob.channels(), datum.channels());
            EXPECT_EQ(blob.height(), patch_height);
            EXPECT_EQ(blob.width(), patch_width);
            EXPECT_EQ(blob.cpu_data()[0], 30);
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < patch_height; ++h) {
                    for (int w = 0; w < patch_width; ++w) {
                        int blob_index = c * patch_height * patch_width + h * patch_width + w;
                        EXPECT_EQ(blob.cpu_data()[blob_index] - blob.cpu_data()[0], c*height*width + h*width + w);
                    }
                }
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestCropMirrorPatchTrain) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 9;
        const int width = 9;
        const int crop_size = 5;
        const int patch_height = 3;
        const int patch_width = 3;

        transform_param.add_patch_center_x(2);
        transform_param.add_patch_center_y(2);
        transform_param.set_crop_size(crop_size);
        transform_param.set_patch_height(patch_height);
        transform_param.set_patch_width(patch_width);
        transform_param.set_mirror(true);

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);

        DataTransformer<TypeParam> transformer(transform_param, TRAIN);

        Blob<TypeParam> blob(1, channels, patch_height, patch_width);

        for (int iter = 0; iter < this->num_iter_; ++iter) {
            Caffe::set_random_seed(this->seed_);
            transformer.InitRand();
            transformer.Transform(datum, &blob);
            EXPECT_EQ(blob.num(), 1);
            EXPECT_EQ(blob.channels(), datum.channels());
            EXPECT_EQ(blob.height(), patch_height);
            EXPECT_EQ(blob.width(), patch_width);
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < patch_height; ++h) {
                    for (int w = 0; w < patch_width; ++w) {
                        int blob_index = c * patch_height * patch_width + h * patch_width + w;
                        EXPECT_EQ(blob.cpu_data()[blob_index] - blob.cpu_data()[2], c*height*width + h*width + (width - 1 - w) - 6);
                    }
                }
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestCropMirrorPatchTest) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 9;
        const int width = 9;
        const int crop_size = 5;
        const int patch_height = 3;
        const int patch_width = 3;

        transform_param.add_patch_center_x(3);
        transform_param.add_patch_center_y(2);
        transform_param.set_crop_size(crop_size);
        transform_param.set_patch_height(patch_height);
        transform_param.set_patch_width(patch_width);
        transform_param.set_mirror(true);

        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);

        DataTransformer<TypeParam> transformer(transform_param, TEST);

        Blob<TypeParam> blob(1, channels, patch_height, patch_width);

        for (int iter = 0; iter < this->num_iter_; ++iter) {
            Caffe::set_random_seed(this->seed_);
            transformer.InitRand();
            transformer.Transform(datum, &blob);
            EXPECT_EQ(blob.num(), 1);
            EXPECT_EQ(blob.channels(), datum.channels());
            EXPECT_EQ(blob.height(), patch_height);
            EXPECT_EQ(blob.width(), patch_width);
            EXPECT_EQ(blob.cpu_data()[0], 31);
            EXPECT_EQ(blob.cpu_data()[2], 29);
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < patch_height; ++h) {
                    for (int w = 0; w < patch_width; ++w) {
                        int blob_index = c * patch_height * patch_width + h * patch_width + w;
                        EXPECT_EQ(blob.cpu_data()[blob_index] - blob.cpu_data()[2], c*height*width + h*width + (width - 1 - w) - 6);
                    }
                }
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestMeanValue) {
        TransformationParameter transform_param;
        const bool unique_pixels = false;  // pixels are equal to label
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int mean_value = 2;

        transform_param.add_mean_value(mean_value);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        Blob<TypeParam> blob(1, channels, height, width);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        transformer.InitRand();
        transformer.Transform(datum, &blob);
        for (int j = 0; j < blob.count(); ++j) {
            EXPECT_EQ(blob.cpu_data()[j], label - mean_value);
        }
    }

    TYPED_TEST(DataTransformTest, TestMeanValues) {
        TransformationParameter transform_param;
        const bool unique_pixels = false;  // pixels are equal to label
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;

        transform_param.add_mean_value(0);
        transform_param.add_mean_value(1);
        transform_param.add_mean_value(2);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        Blob<TypeParam> blob(1, channels, height, width);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        transformer.InitRand();
        transformer.Transform(datum, &blob);
        for (int c = 0; c < channels; ++c) {
            for (int j = 0; j < height * width; ++j) {
                EXPECT_EQ(blob.cpu_data()[blob.offset(0, c) + j], label - c);
            }
        }
    }

    TYPED_TEST(DataTransformTest, TestMeanFile) {
        TransformationParameter transform_param;
        const bool unique_pixels = true;  // pixels are consecutive ints [0,size]
        const int label = 0;
        const int channels = 3;
        const int height = 4;
        const int width = 5;
        const int size = channels * height * width;

        // Create a mean file
        string mean_file;
        MakeTempFilename(&mean_file);
        BlobProto blob_mean;
        blob_mean.set_num(1);
        blob_mean.set_channels(channels);
        blob_mean.set_height(height);
        blob_mean.set_width(width);

        for (int j = 0; j < size; ++j) {
            blob_mean.add_data(j);
        }

        LOG(INFO) << "Using temporary mean_file " << mean_file;
        WriteProtoToBinaryFile(blob_mean, mean_file);

        transform_param.set_mean_file(mean_file);
        Datum datum;
        FillDatum(label, channels, height, width, unique_pixels, &datum);
        Blob<TypeParam> blob(1, channels, height, width);
        DataTransformer<TypeParam> transformer(transform_param, TEST);
        transformer.InitRand();
        transformer.Transform(datum, &blob);
        for (int j = 0; j < blob.count(); ++j) {
            EXPECT_EQ(blob.cpu_data()[j], 0);
        }
    }

    template <typename TypeParam>
    void BlobToCvMat(const Blob<TypeParam> &blob, cv::Mat &matImage)
    {
        const TypeParam *blob_top_data = blob.cpu_data();
        matImage = cv::Mat(blob.height(), blob.width(), CV_MAKETYPE(CV_8U, blob.channels()));
        int img_index = 0, data_index;
        unsigned char *data = matImage.data;
        for (int h = 0; h < blob.height(); ++h) {
            for (int w = 0; w < blob.width(); ++w) {
                for (int c = 0; c < blob.channels(); ++c) {
                    data_index = (c * blob.height() + h) * blob.width() + w;
                    data[img_index++] = (unsigned char)(blob_top_data[data_index]); // 默认为BGR模式
                }
            }
        }
    }

    TYPED_TEST(DataTransformTest, TrainResizeDatum) {
        string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
        Datum image;
        int label = 0;
        ReadImageToDatum(filename, 0, &image);
        TransformationParameter transform_param;
        int new_height = 160;
        int new_width = 120;
        transform_param.set_new_height(new_height);
        transform_param.set_new_width(new_width);
        int crop_size = 80;
        transform_param.set_crop_size(crop_size);
        Blob<TypeParam> blob(1, image.channels(), crop_size, crop_size);

        DataTransformer<TypeParam> transformer(transform_param, TRAIN);
        transformer.InitRand();
        transformer.Transform(image, &blob);

        cv::Mat matImage;
        BlobToCvMat(blob, matImage);

        filename = EXAMPLES_SOURCE_DIR "images/cat_resize_datum.jpg";
        cv::imwrite(filename, matImage);
    }

    template <typename TypeParam>
    void DatumToBlob(const Datum &datum, Blob<TypeParam> &blob)
    {
        blob.Reshape(1, datum.channels(), datum.height(), datum.width());
        const int input_channels = datum.channels();
        const int input_height = datum.height();
        const int input_width = datum.width();
        const int input_length = input_height * input_width;
        const int input_size = input_channels * input_length;
        const string& data = datum.data();
        const bool has_uint8 = data.size() > 0;

        TypeParam *interData = blob.mutable_cpu_data();
        for (int index = 0; index < input_size; ++index) {
            if (has_uint8) {
                interData[index] = static_cast<TypeParam>(static_cast<uint8_t>(data[index]));
            }
            else {
                interData[index] = datum.float_data(index);
            }
        }
    }

    TYPED_TEST(DataTransformTest, TrainResizeBlob) {
        string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
        Datum image;
        int label = 0;
        ReadImageToDatum(filename, label, &image);
        Blob<TypeParam> in_blob(1, image.channels(), image.height(), image.width());
        DatumToBlob(image, in_blob);

        TransformationParameter transform_param;
        int new_height = 160;
        int new_width = 120;
        transform_param.set_new_height(new_height);
        transform_param.set_new_width(new_width);
        int crop_size = 80;
        transform_param.set_crop_size(crop_size);
        Blob<TypeParam> blob(1, image.channels(), crop_size, crop_size);

        DataTransformer<TypeParam> transformer(transform_param, TRAIN);
        transformer.InitRand();
        transformer.Transform(&in_blob, &blob);

        cv::Mat matImage;
        BlobToCvMat(blob, matImage);

        filename = EXAMPLES_SOURCE_DIR "images/cat_resize_blob.jpg";
        cv::imwrite(filename, matImage);
    }

    TYPED_TEST(DataTransformTest, TrainAugmentation) {
        TransformationParameter transform_param;
        int new_height = 160;
        int new_width = 120;
        transform_param.set_new_height(new_height);
        transform_param.set_new_width(new_width);
        string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";

        /*Datum image;
        ReadImageToDatum(filename, 0, &image);
        Blob<TypeParam> blob(1, image.channels(), image.height(), image.width());*/
        cv::Mat image = cv::imread(filename);
        Blob<TypeParam> blob(1, image.channels(), new_height, new_width);
        //cv::Mat image_data;
        /*TransformationParameter_ScaleFactorParameter scale_factor;
        scale_factor.set_min_factor(0.8);
        scale_factor.set_max_factor(1.25);
        transform_param.set_allocated_scale_factor(&scale_factor);*/
        /*transform_param.set_roll_angle(15);*/
        /*TransformationParameter_GaussianAugParameter gaussian_para;
        gaussian_para.set_mean_value(0);
        gaussian_para.set_variance_value(26);
        transform_param.set_allocated_gaussian_para(&gaussian_para);
        TransformationParameter_CoverSizeParameter cover_size;
        cover_size.set_min_size(10);
        cover_size.set_max_size(50);
        transform_param.set_allocated_cover_size(&cover_size);*/
        /*TransformationParameter_AdjustmentsAugParameter adj_para;
        adj_para.set_min_saturation(0.5);
        adj_para.set_max_saturation(1.5);
        transform_param.set_allocated_adjustment_para(&adj_para);*/
        /*TransformationParameter_FishEyeParameter fe_para;
        fe_para.set_max_distort_ratio(0.00005);
        fe_para.set_min_distort_ratio(0.00001);
        fe_para.set_center_x(72);
        fe_para.set_center_y(72);
        transform_param.set_allocated_fisheye_param(&fe_para);*/
        DataTransformer<TypeParam> transformer(transform_param, TRAIN);
        transformer.InitRand();
        transformer.Transform(image, &blob);

        cv::Mat matImage;
        BlobToCvMat(blob, matImage);

        //filename_ = EXAMPLES_SOURCE_DIR;
        filename = EXAMPLES_SOURCE_DIR "images/cat_augmentation.jpg";
        cv::imwrite(filename, matImage);
    }

}  // namespace caffe
#endif  // USE_OPENCV
