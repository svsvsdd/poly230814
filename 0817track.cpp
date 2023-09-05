#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;
using namespace std;

//void on_level_change(int pos, void* userdata);
void on_brightness(int pos, void* userdata);


//int main()	//rgb Æ®·¢¹Ù
//{
//	Mat img = Mat::zeros(400, 400, CV_8UC3);
//	
//
//	namedWindow("image");
//	createTrackbar("B", "image", 0, 16, on_level_change, (void*)&img);
//	createTrackbar("G", "image", 0, 16, on_level_change, (void*)&img);
//	createTrackbar("R", "image", 0, 16, on_level_change, (void*)&img);
//
//	imshow("image", img);
//	
//
//	waitKey(0);
//	//destroyAllWindows();
//	return 0;
//}

void brightness2()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);
	//Mat dst(src.rows, src.cols, src.type());

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	namedWindow("dst");
	createTrackbar("Brightness", "dst", 0, 100, on_brightness, (void*)&src);
	on_brightness(0, (void*)&src);

	waitKey();
	destroyAllWindows();
}

//int main()
//{
//	brightness2();
//
//	waitKey();
//	destroyAllWindows();
//}





//void on_level_change(int pos, void* userdata)
//{
//	Mat img = *(Mat*)userdata;
//
//
//	int b = getTrackbarPos("B", "image");
//	int g = getTrackbarPos("G", "image");
//	int r = getTrackbarPos("R", "image");
//
//	img.setTo(Scalar(b * 16, g * 16, r * 16));
//	imshow("image", img);
//
//
//}




void on_brightness(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;
	Mat dst = src + pos;

	imshow("dst", dst);
}