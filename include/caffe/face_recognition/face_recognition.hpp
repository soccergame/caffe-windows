#pragma once

#ifndef _WIN32
#define __stdcall
#endif

#ifdef __cplusplus
extern "C"
{
#endif	//	__cplusplus

#ifndef _THIDFACERECOGHANDLE
#define _THIDFACERECOGHANDLE
    typedef void *RecognitionHandle;
#endif	//	_THIDFACEHANDLE

    /**
    *	\brief set feature extraction library path
    *		\param[in] szLibPath library path name
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall SetFaceRecognitionLibPath(const char *szLibPath);

    /**
    *	\brief initialize deep face feature extraction sdk
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall InitFaceRecognition(const char *szResName, int gpuId,
        RecognitionHandle *pHandle);


    /**
    *	\brief free deep face feature extraction sdk
    *	\return int error code defined in THIDErrorDef.h
    */
    int __stdcall UninitFaceRecognition(RecognitionHandle handle);

    /**
    *	\brief get deep face feature size in bytes
    *	\return int face feature size in bytes
    */
    int __stdcall GetFaceRecognitionSize(RecognitionHandle handle);

    int __stdcall InnerFaceRecognition(RecognitionHandle handle, const unsigned char *pNormImage, int batchSize, int channels,
        int imageHeight, int imageWidth, float *pFeatures);
#ifdef __cplusplus
}
#endif
