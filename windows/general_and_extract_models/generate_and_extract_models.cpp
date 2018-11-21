#include <fstream>
#include <exception>
#include <string>
#include <stack>
#include "caffe/caffe.hpp"
#include "autoarray.h"
#include "AlgorithmUtils.h"
#include "MyString.h"
#include "MyFile.h"
#include "tclap/CmdLine.h"
#include "opencv2/opencv.hpp"

int _tmain(int argc, TCHAR *argv[])
{
    int retValue = 0;
    std::locale::global(std::locale(""));

    try
    {
        TCLAP::CmdLine cmd(TCLAP_TEXT("Generate or extract caffe models.\n")  \
            TCLAP_TEXT("Copyright: Beijing SN Corp. Ltd.")	\
            TCLAP_TEXT("Author: He Zhixiang")	\
            TCLAP_TEXT("Data: Nov. 5, 2018"), TCLAP_TEXT(' '), TCLAP_TEXT("2.0"));

        // Deployed prototxt file path
        TCLAP::MultiArg<tstring> modelPaths(TCLAP_TEXT("M"),
            TCLAP_TEXT("ModelPathsFile"),
            TCLAP_TEXT("File includes generate prototxt and caffemodel or dat needs to be extracted"),
            true, TCLAP_TEXT("Common use for generate"), cmd);

        TCLAP::ValueArg<tstring> resultDir(TCLAP_TEXT(""),
            TCLAP_TEXT("ResultDir"),
            TCLAP_TEXT("Directiry to store results"),
            false, TCLAP_TEXT(""),
            TCLAP_TEXT("result directory"), cmd);

        TCLAP::SwitchArg GenerateOrExtractedAugSwitch(_T("e"),
            _T("extracted"), _T("when specified, needs to extract models"),
            cmd, false);

        cmd.parse(argc, argv);

        bool is_extracted = GenerateOrExtractedAugSwitch.getValue();
        vector<tstring> model_paths = modelPaths.getValue();
        CMyString result_dir = resultDir.getValue();

        if (is_extracted) {
            CMyString net_path, weight_path;
            std::vector<CMyString> model_names;
            for (auto iter = model_paths.cbegin(); iter != model_paths.cend(); ++iter)
            {
                model_names.push_back(*iter);
            }

            for (int i = 0; i < model_names.size(); ++i) {
                CMyFile fileModel(model_names[i], CMyFile::modeRead);
                int dataSize = static_cast<int>(fileModel.GetLength());
                AutoArray<unsigned char> encryptedData(dataSize);
                fileModel.Read(encryptedData, dataSize);
                fileModel.Close();

                if (dataSize <= 0)
                    throw 1;

                int *pBuffer = reinterpret_cast<int *>(encryptedData.begin());
                // encrypt data by shift left		
                int numOfData = dataSize / sizeof(pBuffer[0]);
                for (int j = 0; j < numOfData; ++j)
                {
                    int tempData = pBuffer[j];
                    pBuffer[j] = brc_sn::ror(
                        static_cast<unsigned int>(tempData),
                        brc_sn::g_shiftBits);
                }
#ifndef OLD_VERSION
                const int modelnumber = pBuffer[0];
                std::vector<int> protoTxtLen, modelSize;
                for (int j = 0; j < modelnumber; ++j)
                {
                    protoTxtLen.push_back(pBuffer[2 * j + 1]);
                    modelSize.push_back(pBuffer[2 * j + 2]);
                }

                unsigned char *pDataBuf = reinterpret_cast<unsigned char*>(encryptedData.begin())
                    + sizeof(int) * (2 * modelnumber + 1);

                for (int j = 0; j < modelnumber; ++j) {
                    std::unique_ptr<caffe::NetParameter> nets(new caffe::NetParameter);
                    int retValue = caffe::ReatNetParamsFromBuffer(pDataBuf, protoTxtLen[j], nets.get());
                    CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
                    CHECK(caffe::UpgradeNetAsNeeded("<memory>", nets.get()));

                    std::unique_ptr<caffe::NetParameter> weights(new caffe::NetParameter);
                    retValue = ReatNetParamsFromBuffer(pDataBuf + protoTxtLen[j], modelSize[j], weights.get());
                    CHECK_EQ(retValue, 0) << "Read net parameters from buffer error, code: " << retValue;
                    CHECK(caffe::UpgradeNetAsNeeded("<memory>", weights.get()));

                    pDataBuf += protoTxtLen[j] + modelSize[j];

                    CMyString base_name = model_names[i].BaseName().SubType();
#ifdef _UNICODE
                    net_path.Format(_T("%ls/%ls_net_%d.prototxt"),
                        result_dir.c_str(), base_name.c_str(), j);
                    weight_path.Format(_T("%ls/%ls_weight_%d.caffemodel"),
                        result_dir.c_str(), base_name.c_str(), j);
#else
                    net_path.Format(_T("%s/%s_net_%d.prototxt"),
                        result_dir.c_str(), base_name.c_str(), j);
                    weight_path.Format(_T("%s/%s_weight_%d.caffemodel"),
                        result_dir.c_str(), base_name.c_str(), j);
#endif

                    caffe::WriteProtoToTextFile((*nets), net_path.ToString());
                    caffe::WriteProtoToBinaryFile((*weights), weight_path.ToString());
                }
#else
                const int protoTxtLen = pBuffer[0];
                const int modelSize = pBuffer[1];
                const unsigned char *pDataBuf =
                    reinterpret_cast<unsigned char *>(
                        encryptedData + sizeof(int) * 2);

                std::unique_ptr<caffe::NetParameter> nets(new caffe::NetParameter);
                int retValue = caffe::ReatNetParamsFromBuffer(pDataBuf, protoTxtLen, nets.get());
                CHECK_EQ(retValue, 0) << "Read net structure from buffer error, code: " << retValue;
                CHECK(caffe::UpgradeNetAsNeeded("<memory>", nets.get()));

                std::unique_ptr<caffe::NetParameter> weights(new caffe::NetParameter);
                retValue = ReatNetParamsFromBuffer(pDataBuf + protoTxtLen, modelSize, weights.get());
                CHECK_EQ(retValue, 0) << "Read net parameters from buffer error, code: " << retValue;
                CHECK(caffe::UpgradeNetAsNeeded("<memory>", weights.get()));

                CMyString base_name = model_names[i].BaseName().SubType();
#ifdef _UNICODE
                net_path.Format(_T("%ls/%ls_net.prototxt"),
                    result_dir.c_str(), base_name.c_str());
                weight_path.Format(_T("%ls/%ls_weight.caffemodel"),
                    result_dir.c_str(), base_name.c_str());
#else
                net_path.Format(_T("%s/%s_net.prototxt"),
                    result_dir.c_str(), base_name.c_str());
                weight_path.Format(_T("%s/%s_weight.caffemodel"),
                    result_dir.c_str(), base_name.c_str());
#endif

                caffe::WriteProtoToTextFile((*nets), net_path.ToString());
                caffe::WriteProtoToBinaryFile((*weights), weight_path.ToString());
#endif
            }
        }
        else {
            CMyString model_path;
            std::vector<CMyString> net_names, weight_names;
            for (auto iter = model_paths.cbegin(); iter != model_paths.cend(); iter = iter + 2)
            {
                net_names.push_back(*iter);
                weight_names.push_back(*(iter + 1));
            }

            const int ModelNumber = net_names.size();
            std::vector<int> solverLen, modelLen;
            for (int i = 0; i < ModelNumber; ++i)
            {
                // 生成临时文件的名字
                char szBuffer[10] = { 0 };
                snprintf(szBuffer, _countof(szBuffer), "%d", i);

                std::string  szTempSolver = "tempSolver_";
                szTempSolver += szBuffer;
                szTempSolver += ".dat";

                std::string szTempModel = "tempModel_";
                szTempModel += szBuffer;
                szTempModel += ".dat";

                // 初始化网络结构
                caffe::Caffe::set_mode(caffe::Caffe::CPU);
                caffe::NetParameter nets;
                caffe::ReadNetParamsFromTextFileOrDie(net_names[i].ToString(), &nets);
                nets.mutable_state()->set_phase(caffe::TEST);
                std::shared_ptr<caffe::Net<float>> pCaffeNet(new caffe::Net<float>(nets));
                pCaffeNet->CopyTrainedLayersFromBinaryProto(weight_names[i].ToString());

                // 写成二进制文件
                caffe::WriteProtoToBinaryFile(nets, szTempSolver);
                // 写成二进制model
                caffe::NetParameter weights;
                pCaffeNet->ToProto(&weights, false);
                caffe::WriteProtoToBinaryFile(weights, szTempModel);

                CMyFile fileSolver(szTempSolver.c_str(), CMyFile::modeRead);
                solverLen.push_back(fileSolver.GetLength());
                CMyFile fileModel(szTempModel.c_str(), CMyFile::modeRead);
                modelLen.push_back(fileModel.GetLength());
                fileSolver.Close();
                fileModel.Close();
            }

            // 开始写入成为一个文件
            int totalLen = 0;
            for (int i = 0; i < ModelNumber; ++i)
                totalLen += solverLen[i] + modelLen[i] + sizeof(int) * 2;
            totalLen += sizeof(int);

            totalLen = ((totalLen + 7) / 8) * 8;
            AutoArray<unsigned char> tempBuffer(totalLen);
            int *pBuffer = reinterpret_cast<int *>(tempBuffer.begin());
            pBuffer[0] = ModelNumber;
            for (int i = 0; i < ModelNumber; ++i)
            {
                pBuffer[2 * i + 1] = solverLen[i];
                pBuffer[2 * i + 2] = modelLen[i];
            }
            unsigned char *pDataPtr = tempBuffer + sizeof(int) * (2 * ModelNumber + 1);

            for (int i = 0; i < ModelNumber; ++i)
            {
                // 生成临时文件的名字
                char szBuffer[10] = { 0 };
                snprintf(szBuffer, _countof(szBuffer), "%d", i);

                std::string  szTempSolver = "tempSolver_";
                szTempSolver += szBuffer;
                szTempSolver += ".dat";

                std::string szTempModel = "tempModel_";
                szTempModel += szBuffer;
                szTempModel += ".dat";

                CMyFile fileSolver(szTempSolver.c_str(), CMyFile::modeRead);
                CMyFile fileModel(szTempModel.c_str(), CMyFile::modeRead);
                // read solver
                fileSolver.Read(pDataPtr, solverLen[i]);
                // read model
                fileModel.Read(pDataPtr + solverLen[i], modelLen[i]);
                fileSolver.Close();
                fileModel.Close();

                pDataPtr = pDataPtr + solverLen[i] + modelLen[i];

#ifdef WIN32
                DeleteFileA(szTempSolver.c_str());
                DeleteFileA(szTempModel.c_str());
#else
                remove(szTempSolver.c_str());
                remove(szTempModel.c_str());
#endif
            }

            // 加密	
            int numOfData = totalLen / sizeof(pBuffer[0]);
            for (int i = 0; i < numOfData; ++i)
            {
                int tempData = pBuffer[i];
                pBuffer[i] = brc_sn::rol(static_cast<unsigned int>(tempData), brc_sn::g_shiftBits);
            }

            // write encrypted model file	
#ifdef _UNICODE
            model_path.Format(_T("%ls/merge_bin.dat"), result_dir.c_str());
#else
            model_path.Format(_T("%s/merge_bin.dat"), result_dir.c_str());
#endif
            CMyFile fileResult(model_path, CMyFile::modeCreate | CMyFile::modeWrite);
            fileResult.Write(tempBuffer, totalLen);
            fileResult.Close();
        }
    }
    catch (int errcode)
    {
        retValue = errcode;
    }
    catch (const std::bad_alloc &)
    {
        retValue = -1;
    }
    catch (...)
    {
        retValue = -2;
    }

    return retValue;
}