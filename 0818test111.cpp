#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;
using namespace std;

//Mat calcGrayHist(const Mat& img)
//{
//	//CV_Assert(img.type() == CV_8UC1);
//
//	Mat hist;
//	int channels[] = { 0 };
//	int dims = 1;
//	const int histSize[] = { 256 };
//	float graylevel[] = { 0, 256 };
//	const float* ranges[] = { graylevel };
//
//	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);
//
//	return hist;
//}


//int main()
//{
//	Mat img = imread("camera.bmp", IMREAD_GRAYSCALE);
//	Mat hist_ = calcGrayHist(img);
//
//	waitKey();
//	destroyAllWindows();
//	return 0;
//}



