#pragma once
#include <algorithm>
#include <vector>
#include <assert.h>

using namespace std;

namespace THID
{
    inline void GetAccuratePosePosition(const float *faceFeaturePoint,
        float *innerEyePoint, float *mouthLeft, float *mouthRight)
    {
        float eyeXAccurate = 0.0f, eyeYAccurate = 0.0f;
        int sum = 0;
        for (int i = 16; i < 24; ++i)
        {
            if (faceFeaturePoint[2*i] > 0 && faceFeaturePoint[2*i+1] > 0)
            {
                eyeXAccurate += faceFeaturePoint[2 * i];
                eyeYAccurate += faceFeaturePoint[2 * i + 1];
                ++sum;
            }
        }
        innerEyePoint[0] = eyeXAccurate / float(sum);
        innerEyePoint[1] = eyeYAccurate / float(sum);

        eyeXAccurate = 0.0f, eyeYAccurate = 0.0f;
        sum = 0;
        for (int i = 24; i < 32; ++i)
        {
            if (faceFeaturePoint[2*i] > 0 && faceFeaturePoint[2*i+1] > 0)
            {
                eyeXAccurate += faceFeaturePoint[2*i];
                eyeYAccurate += faceFeaturePoint[2*i+1];
                ++sum;
            }
        }
        innerEyePoint[2] = eyeXAccurate / float(sum);
        innerEyePoint[3] = eyeYAccurate / float(sum);

        mouthLeft[0] = faceFeaturePoint[90];
        mouthLeft[1] = faceFeaturePoint[91];
        mouthRight[0] = faceFeaturePoint[102];
        mouthRight[1] = faceFeaturePoint[103];
    }

    inline bool IsValidEyePoint(const float *pEyePoint, int nImgWidth, int nImgHeight)
    {
        return !(pEyePoint == NULL ||
            pEyePoint[0] < 0 || pEyePoint[0] >= nImgWidth ||
            pEyePoint[2] < 0 || pEyePoint[2] >= nImgWidth ||
            pEyePoint[1] < 0 || pEyePoint[1] >= nImgHeight ||
            pEyePoint[3] < 0 || pEyePoint[3] >= nImgHeight ||
            fabs(pEyePoint[0] - pEyePoint[2]) + fabs(pEyePoint[1] - pEyePoint[3]) < 5.0f);
    }

	inline unsigned char Interpolate(const unsigned char * pSrc, int width, int height, float curX, float curY,
		float halfScaleX, float halfScaleY,	float maxOverlapX, float maxOverlapY)
	{
		const float EPS = 0.00001f;

		// calculate start x position
		int startX = max(static_cast<int>(curX - halfScaleX - 0.5f) + 1, 0);
		// calculate end x position
		int endX = min(static_cast<int>(curX + halfScaleX + 0.5f), width - 1);
		// calculate start y position
		int startY = max(static_cast<int>(curY - halfScaleY - 0.5) + 1, 0);
		// calculate end y position
		int endY = min(static_cast<int>(curY + halfScaleY + 0.5), height - 1);

		// current weight
		float curWeight = 0;
		// weight sum
		float weightSum = 0;

//		float maxOverlapX = scaleX > 1 ? scaleX : 1;
//		float maxOverlapY = scaleY > 1 ? scaleY : 1;
		float curOverlapX = 0;
		float curOverlapY = 0;

		// weighted average
		float sumValue = 0;
		for (int y = startY; y <= endY; y++)
		{
			curOverlapY = halfScaleY + 0.5f - abs(y - curY);
			for (int x = startX; x <= endX; x++)
			{
				curOverlapX = halfScaleX + 0.5f - abs(x - curX);
				curWeight = curOverlapX * curOverlapY;

				sumValue += curWeight * pSrc[y * width + x];
				weightSum += curWeight;
			}
		}

		if (weightSum < EPS)
			return 0;

		sumValue /= weightSum;
		if (sumValue < 0)
			sumValue = 0;
		else if (sumValue > 255)
			sumValue = 255;
		return static_cast<unsigned char>(sumValue);
	}

	/** normalize image according the paper
	 * X. Wu, "Learning Robust Deep Face Representation," ArXiv e-prints, July 2015.
	 */
	//class CNormImage5pt
	//{
	//public:
	//	CNormImage5pt()
	//	{
	//		m_inited = false;
	//	}
	//	~CNormImage5pt()
	//	{
	//		ReleaseAll();
	//	}

	//	int Initialize(int normWidth, int normHeight, int eyeCenterY, int distEyeCenterMouthCenter)
	//	{
	//		if (m_inited)
	//			return 0;
	//		assert(normWidth > 0 && normHeight > 0);
	//		assert(eyeCenterY > 0 && eyeCenterY < normHeight);
	//		assert(distEyeCenterMouthCenter > 0);
	//		assert(eyeCenterY + distEyeCenterMouthCenter < normHeight);

	//		m_normWidth = normWidth;
	//		m_normHeight = normHeight;			
	//		m_eyeCenterY = eyeCenterY;
	//		m_distEyeCMouthC = distEyeCenterMouthCenter;

	//		m_normEyeC[0] = m_normWidth / 2.0f;
	//		m_normEyeC[1] = eyeCenterY;
	//		m_normMouthC[0] = m_normEyeC[0];
	//		m_normMouthC[1] = m_normEyeC[1] + m_distEyeCMouthC;

	//		m_inited = true;

	//		return 0;
	//	}

	//	int NormImage(const unsigned char *pRaw, int width, int height, const float *pFeaPoints,
	//		int numFeaPoints, unsigned char *pNormFace) const;
	//private:		
	//	void Affine(int normX, int normY, float scale, float cosAngle, float sinAngle,
	//		const float *oriEyeC, int &x, int &y) const
	//	{
	//		x = scale * (cosAngle * (normX - m_normEyeC[0]) - sinAngle * (normY - m_normEyeC[1])) + oriEyeC[0];
	//		y = scale * (sinAngle * (normX - m_normEyeC[0]) + cosAngle * (normY - m_normEyeC[1])) + oriEyeC[1];
	//	}		

	//	void ReleaseAll()
	//	{
	//		m_inited = false;
	//	}

	//	bool m_inited;
	//	int m_normWidth;
	//	int m_normHeight;
	//	int m_eyeCenterY;		// y coordinate of eye center
	//	int m_distEyeCMouthC;	// distance between eye center and mouth center
	//	float m_normEyeC[2];	// normalized image's eye center
	//	float m_normMouthC[2];	// normalized image's mouth center
	//};	

	// normWidth = normHeight = 148, eyeCenterY = 45, distEyeCMouthC = distEyeCenter = 56
	// normWidth = normHeight = 128, eyeCenterY = 35, distEyeCMouthC = distEyeCenter = 56
	// normWidth = normHeight = 160, eyeCenterY = 65, distEyeCMouthC = distEyeCenter = 50
	// normWidth = normHeight = 182, eyeCenterY = 76, distEyeCMouthC = distEyeCenter = 50
	class CNormImage3pt
	{
	public:
		CNormImage3pt()
		{
			m_inited = false;
		}
		~CNormImage3pt()
		{
			ReleaseAll();
		}
		
		int Initialize(int normWidth, int normHeight, int eyeCenterY, int distEyeCenterMouthCenter, int distEyeCenter)
		{
			m_normWidth = normWidth;
			m_normHeight = normHeight;
			m_eyeCenterY = eyeCenterY;
			m_distEyeCMouthC = distEyeCenterMouthCenter;		
			m_distEyeCenter = distEyeCenter;
			
			m_normMouthC[0] = float(m_normWidth) / 2.0f;
			m_normMouthC[1] = float(eyeCenterY + m_distEyeCMouthC);

			m_normLeftEye[0] = float(m_normWidth) / 2.0f - float(m_distEyeCenter) / 2.0f;
			m_normLeftEye[1] = float(m_eyeCenterY);

			m_normRightEye[0] = float(m_normWidth) / 2.0f + float(m_distEyeCenter) / 2.0f;
			m_normRightEye[1] = float(m_eyeCenterY);

			CalcAffineMatrix();

			m_inited = true;

			return 0;
		}

        int NormImage(const unsigned char *pRaw, int width, int height, const float *pFeaPoints,
            int numFeaPoints, unsigned char *pNormFace) const
        {
            {
                if (!m_inited)
                    return -1;
                assert(pRaw != nullptr && pFeaPoints != nullptr && pNormFace != nullptr);
                assert(width > 0 && height > 0);
                assert(numFeaPoints == 5 || numFeaPoints == 88);

                // get eye center & mouth center
                float eyePoint[4];
                float nose[2];
                float mouthLeft[2], mouthRight[2];
                if (numFeaPoints == 5) {
                    eyePoint[0] = pFeaPoints[0];
                    eyePoint[1] = pFeaPoints[1];
                    eyePoint[2] = pFeaPoints[2];
                    eyePoint[3] = pFeaPoints[3];
                    nose[0] = pFeaPoints[4];
                    nose[1] = pFeaPoints[5];
                    mouthLeft[0] = pFeaPoints[6];
                    mouthLeft[1] = pFeaPoints[7];
                    mouthRight[0] = pFeaPoints[8];
                    mouthRight[1] = pFeaPoints[9];
                }
                else {
                    GetAccuratePosePosition(pFeaPoints, eyePoint, mouthLeft, mouthRight);
                }

                //GetAccuratePosePosition(pFeaPoints, eyePoint, mouthLeft, mouthRight);

                // 		eyePoint.xleft = 94.545194f;
                // 		eyePoint.yleft = 113.953526f;
                // 		eyePoint.xright = 152.104925f;
                // 		eyePoint.yright = 103.559208f;
                // 		
                // 		mouthLeft.x = 110.336694f;
                // 		mouthLeft.y = 176.472624f;
                // 		mouthRight.x = 153.286533f;
                // 		mouthRight.y = 170.400433f;
                // check eye point validity
                if (!IsValidEyePoint(eyePoint, width, height))
                    return -2;

                float eyeCenter[2];	// eye center
                eyeCenter[0] = (eyePoint[0] + eyePoint[2]) / 2;
                eyeCenter[1] = (eyePoint[1] + eyePoint[3]) / 2;
                float mouthCenter[2];	// mouth center
                mouthCenter[0] = (mouthLeft[0] + mouthRight[0]) / 2;
                mouthCenter[1] = (mouthLeft[1] + mouthRight[1]) / 2;

                float xDiff = eyeCenter[0] - mouthCenter[0];
                float yDiff = eyeCenter[1] - mouthCenter[1];
                float oriMouthToEyeDist = sqrt(xDiff * xDiff + yDiff * yDiff);
                float scaleVertical = oriMouthToEyeDist / m_distEyeCMouthC;

                xDiff = eyePoint[0] - eyePoint[2];
                yDiff = eyePoint[1] - eyePoint[3];
                float eyeDist = sqrt(xDiff * xDiff + yDiff * yDiff);
                float scale = eyeDist / m_distEyeCenter;

                // vector B : [x0, y0, x1, y1, x2, y2]
                float vecB[s_Mat_Dim] = { eyePoint[0], eyePoint[1], eyePoint[2], eyePoint[3], mouthCenter[0], mouthCenter[1] };
                float vectorX[s_Mat_Dim] = { 0.0 };
                for (int i = 0; i < s_Mat_Dim; i++)
                {
                    for (int j = 0; j < s_Mat_Dim; j++)
                    {
                        vectorX[i] += m_affineMat[i * s_Mat_Dim + j] * vecB[j];
                    }

                }

                float halfScale_X, halfScale_Y, maxOverlap_X, maxOverlap_Y;
                float firstLineMul, secondLineMul, firstParam, secondParam;

                halfScale_X = scale / 2.0f;
                halfScale_Y = scaleVertical / 2.0f;
                maxOverlap_X = scale > 1.0f ? scale : 1.0f;
                maxOverlap_Y = scaleVertical > 1.0f ? scaleVertical : 1.0f;

                /*
                *  [x, y] = [vectorX[0] vector[1]; vector[2] vector[3]] * [i, j] + [vectorX[4] vector[5]]
                */
                const int AXIS_DIM = 2;
                firstLineMul = vectorX[AXIS_DIM * AXIS_DIM];
                secondLineMul = vectorX[AXIS_DIM * AXIS_DIM + 1];
                for (int j = 0; j < m_normHeight; ++j)
                {
                    firstParam = 0;
                    secondParam = 0;
                    for (int i = 0; i < m_normWidth; ++i)
                    {
                        float srcX = firstLineMul + firstParam;
                        float srcY = secondLineMul + secondParam;

                        firstParam += vectorX[0];
                        secondParam += vectorX[2];

                        pNormFace[j * m_normWidth + i] = Interpolate(pRaw, width, height, srcX, srcY, halfScale_X, halfScale_Y, maxOverlap_X, maxOverlap_Y);

                    }
                    firstLineMul += vectorX[1];
                    secondLineMul += vectorX[3];
                }

                return 0;
            }
        }

        int NormImageByEyeAndMouth(const unsigned char *pRaw, int width, int height, const float *pEyePoints,
            const float *pmouthCenter, unsigned char *pNormFace) const
        {
            if (!m_inited)
                return -1;
            assert(pRaw != nullptr && pEyePoints != nullptr && pNormFace != nullptr && pmouthCenter != nullptr);
            assert(width > 0 && height > 0);

            float eyeCenter[2];	// eye center
            eyeCenter[0] = (pEyePoints[0] + pEyePoints[2]) / 2;
            eyeCenter[1] = (pEyePoints[1] + pEyePoints[3]) / 2;

            float xDiff = eyeCenter[0] - pmouthCenter[0];
            float yDiff = eyeCenter[1] - pmouthCenter[1];
            float oriMouthToEyeDist = sqrt(xDiff * xDiff + yDiff * yDiff);
            float scaleVertical = oriMouthToEyeDist / m_distEyeCMouthC;

            xDiff = pEyePoints[0] - pEyePoints[2];
            yDiff = pEyePoints[1] - pEyePoints[3];
            float eyeDist = sqrt(xDiff * xDiff + yDiff * yDiff);
            float scale = eyeDist / m_distEyeCenter;

            // vector B : [x0, y0, x1, y1, x2, y2]
            float vecB[s_Mat_Dim] = { pEyePoints[0], pEyePoints[1], pEyePoints[2], pEyePoints[3], pmouthCenter[0], pmouthCenter[1] };
            float vectorX[s_Mat_Dim] = { 0.0 };
            for (int i = 0; i < s_Mat_Dim; i++)
            {
                for (int j = 0; j < s_Mat_Dim; j++)
                {
                    vectorX[i] += m_affineMat[i * s_Mat_Dim + j] * vecB[j];
                }

            }

            float halfScale_X, halfScale_Y, maxOverlap_X, maxOverlap_Y;
            float firstLineMul, secondLineMul, firstParam, secondParam;

            halfScale_X = scale / 2.0f;
            halfScale_Y = scaleVertical / 2.0f;
            maxOverlap_X = scale > 1.0f ? scale : 1.0f;
            maxOverlap_Y = scaleVertical > 1.0f ? scaleVertical : 1.0f;

            /*
            *  [x, y] = [vectorX[0] vector[1]; vector[2] vector[3]] * [i, j] + [vectorX[4] vector[5]]
            */
            const int AXIS_DIM = 2;
            firstLineMul = vectorX[AXIS_DIM * AXIS_DIM];
            secondLineMul = vectorX[AXIS_DIM * AXIS_DIM + 1];
            for (int j = 0; j < m_normHeight; ++j)
            {
                firstParam = 0;
                secondParam = 0;
                for (int i = 0; i < m_normWidth; ++i)
                {
                    float srcX = firstLineMul + firstParam;
                    float srcY = secondLineMul + secondParam;

                    firstParam += vectorX[0];
                    secondParam += vectorX[2];

                    pNormFace[j * m_normWidth + i] = Interpolate(pRaw, width, height, srcX, srcY, halfScale_X, halfScale_Y, maxOverlap_X, maxOverlap_Y);

                }
                firstLineMul += vectorX[1];
                secondLineMul += vectorX[3];
            }

            return 0;
        }
	private:
        void CalcAffineMatrix() {
            /**
            *Description: Initialize matrix A: [  x0', y0',  0,  0, 1, 0,
            0,  0, x0', y0', 0, 1,
            x1', y1', 0, 0, 1, 0,
            0,  0, x1, y1, 0, 1,
            x2', y2', 0, 0, 1, 0,
            0,  0, x2', y2', 0, 1,]
            */
            // 		float tempA[s_Mat_Dim * s_Mat_Dim] = { 0 };
            // 		tempA[0] = tempA[8] = m_normLeftEye.x;
            // 		tempA[1] = tempA[9] = m_normLeftEye.y;
            // 		tempA[4] = tempA[11] = 1.0;
            // 
            // 		tempA[12] = tempA[20] = m_normRightEye.x;
            // 		tempA[13] = tempA[21] = m_normRightEye.y;
            // 		tempA[16] = tempA[23] = 1.0;
            // 
            // 		tempA[24] = tempA[32] = m_normMouthC.x;
            // 		tempA[25] = tempA[33] = m_normMouthC.y;
            // 		tempA[28] = tempA[35] = 1.0;

            float x0 = m_normLeftEye[0];
            float y0 = m_normLeftEye[1];
            float x1 = m_normRightEye[0];
            float y1 = m_normRightEye[1];
            float x2 = m_normMouthC[0];
            float y2 = m_normMouthC[1];

            memset(m_affineMat, 0, sizeof(m_affineMat));
            float invMul = 1.0f / (x0*y1 - x1*y0 - x0*y2 + x2*y0 + x1*y2 - x2*y1);

            m_affineMat[0] = m_affineMat[13] = (y1 - y2) * invMul;
            m_affineMat[2] = m_affineMat[15] = -(y0 - y2) * invMul;
            m_affineMat[4] = m_affineMat[17] = (y0 - y1) * invMul;
            m_affineMat[6] = m_affineMat[19] = -(x1 - x2) * invMul;
            m_affineMat[8] = m_affineMat[21] = (x0 - x2) * invMul;
            m_affineMat[10] = m_affineMat[23] = -(x0 - x1) * invMul;
            m_affineMat[24] = m_affineMat[31] = (x1 * y2 - x2 * y1) * invMul;
            m_affineMat[26] = m_affineMat[33] = -(x0 * y2 - x2 * y0) * invMul;
            m_affineMat[28] = m_affineMat[35] = (x0 * y1 - x1 * y0) * invMul;
        }
		void ReleaseAll()
		{			
			m_inited = false;
		}

		bool m_inited;
		int m_normWidth;
		int m_normHeight;
		int m_eyeCenterY;
		int m_distEyeCMouthC;	// distance between eye center and mouth center
		int m_distEyeCenter;
        float m_normLeftEye[2];	// normalized image's left eye center
        float m_normRightEye[2];	// normalized image's right eye center
        float m_normMouthC[2];	// normalized image's mouth center

		static const int s_Mat_Dim = 6;
		float m_affineMat[s_Mat_Dim * s_Mat_Dim];
	};

	const int g_numPoints = 5;
	const int g_normSize = 256;
	// left_eye, right_eye, nose, left_mouse_corner and right_mouse_corner
	static float g_NormPoints[10] = {
		89.3095f, 72.9025f,
		169.3095f, 72.9025f,
		127.8949f, 127.0441f,
		96.8796f, 184.8907f,
		159.1065f, 184.7601f,
	};
	static float g_anotherNormPoints[10] = {
		30 * 2, 55 * 2,
		98 * 2, 55 * 2,
		63.5f * 2, 78 * 2,
		43 * 2, 100 * 2,
		85 * 2, 100 * 2,
	};
	enum InterpolateType
	{
		Bilinear = 0,
		Cubic = 1
	};

	// normalize to 160
	static float g_normPoints_160[10] = {
		55, 65,
		105, 65,
		80, 95,
		60, 115,
		100, 115,
	};
	// normalize image by affine transform
	class CAffineNormImage
	{
	public:
		CAffineNormImage()
		{
			m_inited = false;
		}
		~CAffineNormImage() 
		{
			ReleaseAll();
		}
        int Initialize(int normWidth, int normHeight, float scale = 1.0f, int normSize = g_normSize, const float *pNormPoints = g_NormPoints,
            int numPoints = 5, InterpolateType type = Bilinear) {
            assert(normWidth > 0);
            assert(normHeight > 0);
            assert(pNormPoints != nullptr);
            assert(numPoints > 0);
            assert(normWidth == normHeight);

            m_normWidth = normWidth;
            m_normHeight = normHeight;
            m_numPoints = numPoints;
            m_interpolateType = type;

            // float scale = static_cast<float>(normWidth) / g_normSize;

            m_normPoints.resize(m_numPoints * 2);
            for (int i = 0; i < m_numPoints; ++i)
            {
                m_normPoints[2 * i] = pNormPoints[2 * i] * scale + (normWidth - normSize * scale) / 2;
                m_normPoints[2 * i + 1] = pNormPoints[2 * i + 1] * scale + (normHeight - normSize * scale) / 2;
            }

            PreCompute();

            m_inited = true;
            return 0;
        }

        int NormImage(const unsigned char *pRaw, int width, int height, const float *pFeaPoints,
            int numFeaPoints, unsigned char *pNormFace) const
        {
            if (!m_inited)
                return -1;

            vector<float> imagePoints(2 * m_numPoints, 0);
            if (numFeaPoints == m_numPoints)		// the same number of points with input
            {
                for (int i = 0; i < m_numPoints; ++i)
                {
                    imagePoints[2 * i] = pFeaPoints[2 * i];
                    imagePoints[2 * i + 1] = pFeaPoints[2 * i + 1];
                }
            }
            else
            {
                // get eye center & mouth center
                float eyePoint[4];
                float mouthLeft[2], mouthRight[2];
                GetAccuratePosePosition(pFeaPoints, eyePoint, mouthLeft, mouthRight);
                // left_eye, right_eye, nose, left_mouse_corner and right_mouse_corner
                imagePoints[0] = eyePoint[0];
                imagePoints[0] = eyePoint[1];
                imagePoints[1] = eyePoint[2];
                imagePoints[1] = eyePoint[3];
                imagePoints[2] = (pFeaPoints[64] + pFeaPoints[78]) / 2.0f;
                imagePoints[2] = (pFeaPoints[65] + pFeaPoints[79]) / 2.0f;
                imagePoints[3] = mouthLeft[0];
                imagePoints[3] = mouthLeft[1];
                imagePoints[4] = mouthRight[0];
                imagePoints[4] = mouthRight[1];
            }

            float sumU = 0, sumV = 0, sumUXVY = 0, sumVX_UY = 0;
            for (int i = 0; i < m_numPoints; ++i)
            {
                sumU += imagePoints[2 * i];
                sumV += imagePoints[2 * i + 1];
                sumUXVY += imagePoints[2 * i] * m_normPoints[2 * i] + imagePoints[2 * i + 1] * m_normPoints[2 * i + 1];
                sumVX_UY += imagePoints[2 * i + 1] * m_normPoints[2 * i] - imagePoints[2 * i] * m_normPoints[2 * i + 1];
            }

            float scaleX = (-m_ndiva * sumUXVY + m_bdiva * sumU + m_cdiva * sumV) / m_raidus;
            float scaleY = (-m_ndiva * sumVX_UY - m_cdiva * sumU + m_bdiva * sumV) / m_raidus;
            float shiftX = (m_bdiva * sumUXVY - m_cdiva * sumVX_UY - sumU) / m_raidus;
            float shiftY = (m_cdiva * sumUXVY + m_bdiva * sumVX_UY - sumV) / m_raidus;

            float scale = sqrt(scaleX * scaleX + scaleY * scaleY);
            scale = 1.0f / scale;
            for (int j = 0; j < m_normHeight; ++j)
            {
                for (int i = 0; i < m_normWidth; ++i)
                {
                    float x, y;
                    Affine(i, j, scaleX, scaleY, shiftX, shiftY, x, y);
                    pNormFace[j * m_normWidth + i] = Interpolate(pRaw, width, height, x, y, scale);
                }
            }

            return 0;
        }

	private:
		inline void ReleaseAll() {	m_inited = false; }
        inline void PreCompute() {
            m_sumX = 0;
            m_sumY = 0;
            m_sumXXYY = 0;
            for (int i = 0; i < m_numPoints; ++i)
            {
                m_sumX += m_normPoints[2 * i];
                m_sumY += m_normPoints[2 * i + 1];
                m_sumXXYY += m_normPoints[2 * i] * m_normPoints[2 * i] + m_normPoints[2 * i + 1] * m_normPoints[2 * i + 1];
            }
            m_raidus = (m_sumX * m_sumX + m_sumY * m_sumY) / m_sumXXYY - m_numPoints;

            m_ndiva = m_numPoints / m_sumXXYY;
            m_bdiva = m_sumX / m_sumXXYY;
            m_cdiva = m_sumY / m_sumXXYY;
        }

		inline void Affine(int normX, int normY, float scaleX, float scaleY, float shiftX, float shiftY, float &x, float &y) const
		{
			x = normX * scaleX - normY * scaleY + shiftX;
			y = normX * scaleY + normY * scaleX + shiftY;
		}

		inline unsigned char Interpolate(const unsigned char * pSrc, int width, int height, float x, float y, float scale = 1.0f) const
		{
			float ans = 0;
			// bilinear interpolate
			if (m_interpolateType == Bilinear)
			{
				int uy = int(floor(y)), ux = int(floor(x));				
				if (uy >= 0 && uy < height - 1 && ux >= 0 && ux < width - 1)
				{
					int offset = uy * width + ux;
					float cof_x = y - uy;
					float cof_y = x - ux;
					ans = (1 - cof_y) * pSrc[offset] + cof_y * pSrc[offset + 1];
					ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * pSrc[offset + width]
						+ cof_y * pSrc[offset + width + 1]);
				}
			}
			else
			{
				// bicubic interpolate				
				if (y >= 0 && y < height && x >= 0 && x < width)
				{
					scale = (std::min)(scale, 1.0f);
					float kernel_width = (std::max)(8.0f, 4.0f / scale); // bicubic kernel width				
					std::vector<float> weights_x, weights_y;
					std::vector<int>  indices_x, indices_y;
					weights_x.reserve(20), indices_x.reserve(20);
					weights_y.reserve(20), indices_y.reserve(20);
					// get indices and weight along x axis
					for (int ux = int(ceil(y - kernel_width / 2.0f));
						ux <= int(floor(y + kernel_width / 2.0f)); ++ux)
					{
						indices_x.push_back((std::max)((std::min)(height - 1, ux), 0));
						weights_x.push_back(Cubic((y - ux) * scale));
					}
					// get indices and weight along y axis
					for (int uy = int(ceil(x - kernel_width / 2.0f));
						uy <= int(floor(x + kernel_width / 2.0f)); ++uy)
					{
						indices_y.push_back((std::max)((std::min)(width - 1, uy), 0));
						weights_y.push_back(Cubic((x - uy) * scale));
					}
					// normalize the weights
					int lx = int(weights_x.size()), ly = int(weights_y.size());
					Norm(&weights_x[0], lx);
					Norm(&weights_y[0], ly);
					float val = 0;					
					for (int i = 0; i < lx; ++i)
					{
						if (i == 0 || indices_x[i] != indices_x[i - 1])
						{
							val = 0;
							int offset = indices_x[i] * width;
							for (int j = 0; j < ly; ++j) {
								val += pSrc[offset + indices_y[j]] * weights_y[j];
							}
						}
						ans += val * weights_x[i];
					}
				}
			}

			unsigned char val = 0;
			if (ans < 0)
				ans = 0;
			if (ans > 255)
				ans = 255;
			val = static_cast<unsigned char>(ans);
			return val;
		}

		inline float Cubic(float x) const
		{
			float ax = fabs(x), ax2, ax3;
			ax2 = ax * ax;
			ax3 = ax2 * ax;
			if (ax <= 1)
				return 1.5f * ax3 - 2.5f * ax2 + 1;
			else if (ax <= 2)
				return -0.5f * ax3 + 2.5f * ax2 - 4 * ax + 2;

			return 0;
		}

		inline void Norm(float *pWeights, int num) const
		{
			float sum = 0;
			for (int i = 0; i < num; ++i)
				sum += pWeights[i];
			for (int i = 0; i < num; ++i)
				pWeights[i] /= sum;
		}

		int m_normWidth;
		int m_normHeight;
		bool m_inited;			// initialized or not
		int m_numPoints;		// number of points
		vector<float> m_normPoints;		// coordinate of normalized points
		InterpolateType m_interpolateType;

		float m_sumXXYY;	// sum(x * x + y * y)
		float m_sumX;		// sum(x)
		float m_sumY;		// sum(y)
		float m_raidus;		// r = (sumX * sumX + sumY * sumY) / sumXXYY - n

		float m_ndiva;		// n / sumXXYY
		float m_bdiva;		// sumX / sumXXYY
		float m_cdiva;		// sumY / sumXXYY		
	};

	/* norm_size = 160, margin = 32
	 * norm_size = 182, margin = 44
	 */
	//class CNormFaceRegion
	//{
	//public:
	//	CNormFaceRegion()
	//	{
	//		m_inited = false;
	//	}
	//	~CNormFaceRegion()
	//	{
	//		ReleaseAll();
	//	}

	//	int Initialize(int normWidth, int normHeight, int margin, InterpolateType type = Bilinear)
	//	{
	//		m_normWidth = normWidth;
	//		m_normHeight = normHeight;
	//		m_halfMargin = margin / 2;

	//		m_interpolateType = type;

	//		return THID_ERR_NONE;
	//	}

	//	int NormImage(const unsigned char *pRaw, int width, int height, const THIDFaceRect *pFaceRect, unsigned char *pNormFace) const
	//	{
	//		const int left = max(0, pFaceRect->left - m_halfMargin);
	//		const int top = max(0, pFaceRect->top - m_halfMargin);
	//		const int right = min(width, pFaceRect->right + m_halfMargin);
	//		const int bottom = min(height, pFaceRect->bottom + m_halfMargin);
	//		
	//		const float scaleX = static_cast<float>(right - left) / m_normWidth;
	//		const float scaleY = static_cast<float>(bottom - top) / m_normHeight;

	//		float scale = sqrt(scaleX * scaleX + scaleY * scaleY);
	//		scale = 1.0f / scale;
	//		for (int y = 0; y < m_normHeight; ++y)
	//		{
	//			float ori_y = scaleY * y + top;
	//			for (int x = 0; x < m_normWidth; ++x)
	//			{
	//				float ori_x = scaleX * x + left;
	//				pNormFace[y * m_normWidth + x] = Interpolate(pRaw, width, height, ori_x, ori_y, scale);
	//			}
	//		}
	//		return THID_ERR_NONE;
	//	}
	//private:
	//	inline void ReleaseAll() { m_inited = false; }

	//	inline unsigned char Interpolate(const unsigned char * pSrc, int width, int height, float x, float y, float scale = 1.0f) const
	//	{
	//		float ans = 0;
	//		// bilinear interpolate
	//		if (m_interpolateType == Bilinear)
	//		{
	//			int uy = floor(y), ux = floor(x);
	//			if (uy >= 0 && uy < height - 1 && ux >= 0 && ux < width - 1)
	//			{
	//				int offset = uy * width + ux;
	//				float cof_x = y - uy;
	//				float cof_y = x - ux;
	//				ans = (1 - cof_y) * pSrc[offset] + cof_y * pSrc[offset + 1];
	//				ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * pSrc[offset + width]
	//					+ cof_y * pSrc[offset + width + 1]);
	//			}
	//		}
	//		else
	//		{
	//			// bicubic interpolate				
	//			if (y >= 0 && y < height && x >= 0 && x < width)
	//			{
	//				scale = (std::min)(scale, 1.0f);
	//				float kernel_width = (std::max)(8.0f, 4.0f / scale); // bicubic kernel width				
	//				std::vector<float> weights_x, weights_y;
	//				std::vector<int>  indices_x, indices_y;
	//				weights_x.reserve(20), indices_x.reserve(20);
	//				weights_y.reserve(20), indices_y.reserve(20);
	//				// get indices and weight along x axis
	//				for (int ux = ceil(y - kernel_width / 2);
	//					ux <= floor(y + kernel_width / 2); ++ux)
	//				{
	//					indices_x.push_back((std::max)((std::min)(height - 1, ux), 0));
	//					weights_x.push_back(Cubic((y - ux) * scale));
	//				}
	//				// get indices and weight along y axis
	//				for (int uy = ceil(x - kernel_width / 2);
	//					uy <= floor(x + kernel_width / 2); ++uy)
	//				{
	//					indices_y.push_back((std::max)((std::min)(width - 1, uy), 0));
	//					weights_y.push_back(Cubic((x - uy) * scale));
	//				}
	//				// normalize the weights
	//				int lx = weights_x.size(), ly = weights_y.size();
	//				Norm(&weights_x[0], lx);
	//				Norm(&weights_y[0], ly);
	//				float val = 0;
	//				for (int i = 0; i < lx; ++i)
	//				{
	//					if (i == 0 || indices_x[i] != indices_x[i - 1])
	//					{
	//						val = 0;
	//						int offset = indices_x[i] * width;
	//						for (int j = 0; j < ly; ++j) {
	//							val += pSrc[offset + indices_y[j]] * weights_y[j];
	//						}
	//					}
	//					ans += val * weights_x[i];
	//				}
	//			}
	//		}

	//		unsigned char val = 0;
	//		if (ans < 0)
	//			ans = 0;
	//		if (ans > 255)
	//			ans = 255;
	//		val = static_cast<unsigned char>(ans);
	//		return val;
	//	}

	//	inline float Cubic(float x) const
	//	{
	//		float ax = fabs(x), ax2, ax3;
	//		ax2 = ax * ax;
	//		ax3 = ax2 * ax;
	//		if (ax <= 1)
	//			return 1.5f * ax3 - 2.5f * ax2 + 1;
	//		else if (ax <= 2)
	//			return -0.5f * ax3 + 2.5f * ax2 - 4 * ax + 2;

	//		return 0;
	//	}

	//	inline void Norm(float *pWeights, int num) const
	//	{
	//		float sum = 0;
	//		for (int i = 0; i < num; ++i)
	//			sum += pWeights[i];
	//		for (int i = 0; i < num; ++i)
	//			pWeights[i] /= sum;
	//	}

	//	int m_halfMargin;
	//	int m_normWidth;
	//	int m_normHeight;
	//	InterpolateType m_interpolateType;
	//	bool m_inited;		// initialized or not
	//};
}
