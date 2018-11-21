#pragma once

#include <string.h>
#include <sstream>
#include <memory>
#include <exception>
#include <new>
#include <cassert>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#include <sys/types.h>
#endif

#if defined(_ANDROID) || defined(_IOS)
#define fseeko64 fseek
#define ftello64	ftell
#define fopen64	fopen
#define ftruncate64 ftruncate
#endif

namespace brc_sn
{
    // for imagenet, normalize image size is 224 * 224
    // const float g_scale = 224.0f / 128.0f;
    //const float g_scale = 1.0f;// 224.0f / 128.0f;	// normal image resized to 128 * 128 , 
    const int g_shiftBits = 11;
    // rotate shift right by moves bits
    template<typename T> T ror(T x, unsigned int moves)
    {
        return (x >> moves) | (x << (sizeof(T) * 8 - moves));
    }

    // rotate shift left by moves bits
    template<typename T> T rol(T x, unsigned int moves)
    {
        return (x << moves) | (x >> (sizeof(T) * 8 - moves));
    }

    // const int normSize = static_cast<int>(144 * g_scale);		// normalize image size
    // const int eyeCenterY = static_cast<int>(48 * g_scale);		// eye center y coordinate in normalized image
    // const int distEyeCMouthC = static_cast<int>(48 * g_scale);	// distance between eye center to mouth center in normalized image
    // int g_meanVal[3] = { 104, 117, 123 };
}