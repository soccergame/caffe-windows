#pragma once

#ifndef _WIN32
#define __stdcall
#endif

#ifdef __cplusplus
extern "C"
{
#endif	//	__cplusplus

#ifndef _THIDFACEHANDLE
#define _THIDFACEHANDLE
	typedef void *BeautyHandle;
#endif	//	_THIDFACEHANDLE

	/**
	 *	\brief set feature extraction library path
	 *		\param[in] szLibPath library path name
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall SetDeepFeatLibPath(const char *szLibPath);

	/**
	 *	\brief initialize deep face feature extraction sdk	 
	 *	\return int error code defined in THIDErrorDef.h	 
	 */
    int __stdcall InitDeepFeat(const char *szResName, int gpuId, BeautyHandle *pHandle);
	

	/**
	 *	\brief free deep face feature extraction sdk
	 *	\return int error code defined in THIDErrorDef.h
	 */
	int __stdcall UninitDeepFeat(BeautyHandle handle);

	/**
	 *	\brief get deep face feature size in bytes
	 *	\return int face feature size in bytes
	 */
    int __stdcall GetDeepFeatSize(BeautyHandle handle);

    int __stdcall InnerDeepFeat(BeautyHandle handle, const unsigned char *pNormImage, int batchSize, int channels,
        int imageHeight, int imageWidth, float *pFeatures);
#ifdef __cplusplus
}
#endif
