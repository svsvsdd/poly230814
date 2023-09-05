#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;
using namespace std;



void camera_in()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "camera open failed" << endl;
		return;
	}

	cout << "frame width:" << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "frame height:" << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		inversed = ~frame;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(10) == 27)
			break;
	}
	destroyAllWindows();

}


void video_in()
{
	VideoCapture cap("stopwatch.avi");
	if (!cap.isOpened()) {
		cerr << "video open failed" << endl;
		return;
	}

	cout << "frame width:" << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "frame height:" << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;
	cout << "frame count:" << cvRound(cap.get(CAP_PROP_FRAME_COUNT)) << endl;

	double fps = cap.get(CAP_PROP_FPS);
	cout << "FPS:" << fps << endl;

	int delay = cvRound(1000 / fps);

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		inversed = ~frame;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(delay) == 27)
			break;
	}
	destroyAllWindows();
}


void camera_in_video_out()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "camera open failed" << endl;
		return;
	}

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	//double fps = cap.get(CAP_PROP_FPS);

	double fps = 30;

	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
	int delay = cvRound(1000 / fps);

	VideoWriter outputVideo("output.avi", fourcc, fps, Size(w, h));

	if (!outputVideo.isOpened()) {
		cout << "file open failed" << endl;
		return;
	}

	Mat frame, inversed;
	while (1) {
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;
		outputVideo << inversed;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(delay) == 27)
			break;

	}
	destroyAllWindows();


}



void drawLines()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));

	line(img, Point(50, 50), Point(200, 50), Scalar(0, 0, 255));
	line(img, Point(50, 100), Point(200, 100), Scalar(255, 0, 255), 3);
	line(img, Point(50, 150), Point(200, 150), Scalar(255, 0, 0), 10);

	line(img, Point(250, 50), Point(350, 100), Scalar(0, 0, 255), 1, LINE_4);
	line(img, Point(250, 70), Point(350, 120), Scalar(255, 0, 255), 1, LINE_8);
	line(img, Point(250, 90), Point(350, 140), Scalar(255, 0, 0), 1, LINE_AA);

	arrowedLine(img, Point(50, 200), Point(150, 200), Scalar(0, 0, 255), 1);
	arrowedLine(img, Point(50, 250), Point(350, 250), Scalar(255, 0, 255), 1);
	arrowedLine(img, Point(50, 300), Point(350, 300), Scalar(255, 0, 0), 1, LINE_8, 0, 0.05);

	drawMarker(img, Point(50, 350), Scalar(0, 0, 255), MARKER_CROSS);
	drawMarker(img, Point(100, 350), Scalar(0, 0, 255), MARKER_TILTED_CROSS);
	drawMarker(img, Point(150, 350), Scalar(0, 0, 255), MARKER_STAR);
	drawMarker(img, Point(200, 350), Scalar(0, 0, 255), MARKER_DIAMOND);
	drawMarker(img, Point(250, 350), Scalar(0, 0, 255), MARKER_SQUARE);
	drawMarker(img, Point(300, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_UP);
	drawMarker(img, Point(350, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_DOWN);

	imshow("img", img);

	waitKey(0);
	destroyAllWindows();
}



void drawPolys()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));

	rectangle(img, Rect(50, 50, 100, 50), Scalar(0, 0, 255), 2);
	rectangle(img, Rect(50, 150, 100, 50), Scalar(0, 0, 128), -1);


	circle(img, Point(300, 120), 30, Scalar(255, 255, 0), -1, LINE_AA);
	circle(img, Point(300, 120), 60, Scalar(255, 0, 0), 3, LINE_AA);


	ellipse(img, Point(120, 300), Size(60, 30), 20, 0, 270, Scalar(255, 255, 0), -1, LINE_AA);
	ellipse(img, Point(120, 300), Size(100, 50), 20, 0, 360, Scalar(0, 255, 0), 2, LINE_AA);

	vector<Point> pts;
	pts.push_back(Point(250, 250)); pts.push_back(Point(300, 250));
	pts.push_back(Point(300, 300)); pts.push_back(Point(350, 300));
	pts.push_back(Point(350, 350)); pts.push_back(Point(250, 350));
	polylines(img, pts, true, Scalar(255, 0, 0), 2);
	fillPoly(img, pts, Scalar(0, 255, 0));
	

	imshow("img", img);

	waitKey(0);
	destroyAllWindows();
}


void drawText1()
{
	Mat img(500, 800, CV_8UC3, Scalar(255, 255, 255));

	putText(img, "FONT_HERSHEY_COMPLEX | FONT_ITALIC", Point(20, 450), FONT_HERSHEY_COMPLEX | FONT_ITALIC, 1, Scalar(255, 0, 0));
	imshow("img", img);

	waitKey(0);
}


void drawText2()
{
	Mat img(500, 800, CV_8UC3, Scalar(255, 255, 255));

	String str = "Hello, OpenCV";
	int fontFace = FONT_HERSHEY_TRIPLEX;
	double fontScale = 2.0;
	int thickness = 1;

	Size textSize = getTextSize(str, fontFace, fontScale, thickness, 0);
	Size imgSize = img.size();


	int x = (imgSize.width - textSize.width) / 2;
	int y = (imgSize.height - textSize.height) / 2;
	int recx = x + textSize.width;
	int recy = y - textSize.height;


	rectangle(img, Point(x, y), Point(recx, recy), Scalar(0, 0, 255), 2);
	putText(img, str, Point(x, y), FONT_HERSHEY_TRIPLEX, 2, Scalar(255, 0, 0), 2);
	imshow("img", img);

	waitKey(0);
}




String filename = "mydata.json";

void writeData()
{
	String name = "Jane";
	int age = 10;
	Point pt1(100, 200);
	vector<int> scores = { 80,90,50 };
	Mat mat1 = (Mat_<float>(2, 2) << 1.0f, 1.5f, 2.0f, 3.2f);

	FileStorage fs(filename, FileStorage::WRITE);

	if (!fs.isOpened()) {
		cerr << "file open failed" << endl;
		return;
	}

	fs << "name" << name;
	fs << "age" << age;
	fs << "Point" << pt1;
	fs << "scores" << scores;
	fs << "data" << mat1;

	fs.release();

}


void readData()
{
	String name;
	int age;
	Point pt1;
	vector<int> scores;
	Mat mat1;

	FileStorage fs(filename, FileStorage::READ);

	if (!fs.isOpened()) {
		cerr << "file open failed" << endl;
		return;
	}

	fs["name"] >> name;
	fs["age"] >> age;
	fs["Point"] >> pt1;
	fs["scores"] >> scores;
	fs["data"] >> mat1;

	fs.release();

	cout << "name: " << name << endl;
	cout << "age: " << age << endl;
	cout << "Point: " << pt1 << endl;
	cout << "scores: " << Mat(scores).t() << endl;
	cout << "data:\n" << mat1 << endl;

}



void mask_setTo()
{
	Mat src = imread("lenna.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_smile.bmp", IMREAD_GRAYSCALE);

	if (src.empty() || mask.empty()) {
		cerr << "image load failed" << endl;
		return;
	}

	src.setTo(Scalar(0, 255, 255), mask);

	imshow("src", src);
	imshow("mask", mask);

}


void mask_copyTo()
{
	Mat src = imread("airplane.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_plane.bmp", IMREAD_GRAYSCALE);
	Mat dst = imread("field.bmp", IMREAD_COLOR);

	if (src.empty() || mask.empty() || dst.empty()) {
		cerr << "img load failed" << endl;
		return;
	}

	src.copyTo(dst, mask);

	imshow("dst", dst);

}

void time_inverse()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}


	Mat dst(src.rows, src.cols, src.type());

	TickMeter tm;
	tm.start();

	//for (int j = 0; j < src.rows; j++) {
	//	for (int i = 0; i < src.cols; i++) {
	//		dst.at<uchar>(j, i) = 255 - src.at<uchar>(j, i);	//하나하나 픽셀 계산. 8ms
	//		//dst.at<uchar>(j, i) = ~src.at<uchar>(j, i);
	//	}
	//}

	dst = ~src;	//opencv 함수 최적화 0.1ms

	tm.stop();
	cout << "img inverse took " << tm.getTimeMilli() << "ms." << endl;
}

void sum_mean_minMaxLoc()
{
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

	cout << "sum:" << (int)sum(img)[0] << endl;
	cout << "mean:" << (int)mean(img)[0] << endl;

	double minVal, maxVal;
	Point minPos, maxPos;
	minMaxLoc(img, &minVal, &maxVal, &minPos, &maxPos);

	cout << "minval:" << minVal << " at " << minPos << endl;
	cout << "maxval:" << maxVal << " at " << maxPos << endl;


}

void normalize()
{
	//Mat src = Mat_<float>({ 1,5 }, { -1.f,-0.5f,0.f,0.5f,1.f });
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);


	Mat dst;
	normalize(src, dst, 128, 255, NORM_MINMAX, CV_8UC1);

	/*cout << "src:" << src << endl;
	cout << "dst:" << dst << endl;*/

	imshow("src", src);
	imshow("dst", dst);
}

void brightness1()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);
	//Mat dst = src + 100;	//mat함수에서 오버플로우 발동 안하게 자동 적용(saturate)
	Mat dst(src.rows, src.cols, src.type());

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			dst.at<uchar>(j, i) = saturate_cast<uchar>(src.at<uchar>(j, i) + 100);	//하나하나 값에 100 더함
		}
	}


	imshow("src", src);
	imshow("dst", dst);
}


void contrast1()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);
	
	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	float s = 2.f;
	Mat dst = s * src;
	
	imshow("src", src);
	imshow("dst", dst);

}



void contrast2()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img load fail" << endl;
		return;
	}

	float alpha = -0.5;
	Mat dst = src + (src - 128) * alpha;


	imshow("src", src);
	imshow("dst", dst);

}

Mat calcGrayHist(const Mat& img)
{
	CV_Assert(img.type() == CV_8UC1);

	Mat hist;
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0,256 };
	const float* ranges[] = { graylevel };

	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);

	return hist;
}

//int main()
//{
//
//
//	Mat img = imread("camera.bmp", IMREAD_GRAYSCALE);
//	Mat hist = calcGrayHist(img);
//
//	// Print histogram values
//	for (int i = 0; i < hist.rows; i++) {
//		float binValue = hist.at<float>(i);
//		std::cout << "Bin " << i << ": " << binValue << std::endl;
//	}
//
//	waitKey();
//	destroyAllWindows();
//	return 0;
//}

Mat getGrayHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1,256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}
	return imgHist;
}



void histgoram_stretching()
{
	Mat src = imread("hawkes.bmp", IMREAD_GRAYSCALE);

	double gmin, gmax;
	minMaxLoc(src, &gmin, &gmax);
	
	Mat dst = (src - gmin) * 255 / (gmax - gmin);

	imshow("src", src);
	imshow("srchist", getGrayHistImage(calcGrayHist(src)));

	imshow("dst", dst);
	imshow("dsthist", getGrayHistImage(calcGrayHist(dst)));

}

void substract()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "camera open failed" << endl;
		return;
	}
	
	Mat model, model_gray, src, src_gray, stopimg, stopimg2, diff;
	while (true) {
		cap >> model;
		if (model.empty())
			break;
		cvtColor(model, model_gray, COLOR_BGR2GRAY);

		if (waitKey(10) == 's') {
			stopimg = model_gray.clone();
			//stopimg = model_gray;
			absdiff(stopimg, model_gray, diff);
		}

		/*cap >> src;
		if (src.empty())
			break;
		cvtColor(src, src_gray, COLOR_BGR2GRAY);*/

		/*if (waitKey(10) == 'i')
			stopimg2 = src_gray.clone();*/


		//Mat diff, diffimg;
		//absdiff(model_gray, src_gray, diff);


		if (!stopimg.empty() && !model_gray.empty() && stopimg.size() == model_gray.size()) {
			Mat diff;
			absdiff(stopimg, model_gray, diff);
			imshow("diff", diff);
		}
		//absdiff(stopimg, model_gray, diff);		//이대로 실행하면 s누르기 전에 stopimg가 존재하지 않기에 오류가 남.

		//absdiff(stopimg, stopimg2, diffimg);

		//absdiff(stopimg, stopimg2, diffimg);
		//diffimg = | stopimg - stopimg2 | ;

		imshow("model_gray", model_gray);
		//imshow("src_gray", src_gray);
		//imshow("diff", diff);
		//imshow("diffimg", diffimg);
		
		if (waitKey(10) == 27)
			break;
	}
}

void addsubmult() 
{
	Mat src1 = imread("lenna256.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("square.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "img load failed" << endl;
		return;
	}

	//imshow("src1", src1);
	//imshow("src2", src2);

	Mat dadd, dsub, dweight, ddiff;

	add(src1, src2, dadd);
	subtract(src1, src2, dsub);
	addWeighted(src1, 0.5, src2, 0.5, 0, dweight);
	absdiff(src1, src2, ddiff);

	/*imshow("dadd", dadd);
	imshow("dsub", dsub);
	imshow("dweight", dweight);
	imshow("ddiff", ddiff);*/


	Mat dbitand, dbitor, dbitxor, dbitnot;
	bitwise_and(src1, src2, dbitand);
	bitwise_or(src1, src2, dbitor);
	bitwise_xor(src1, src2, dbitxor);
	bitwise_not(src1, dbitnot);

	imshow("dbitand", dbitand);
	imshow("dbitor", dbitor);
	imshow("dbitxor", dbitxor);
	imshow("dbitnot", dbitnot);
}

void filter_embossing()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	float data[] = { -1,-1,0,-1,0,1,0,1,1 };
	Mat emboss(3, 3, CV_32FC1, data);

	Mat dst;
	filter2D(src, dst, -1, emboss, Point(-1, -1), 128);

	imshow("src", src);
	imshow("dst", dst);

}

void blurring_mean()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	imshow("src", src);

	Mat dst;
	for (int ksize = 3; ksize <= 7; ksize += 2) {
		blur(src, dst, Size(ksize, ksize));

		String desc = format("Mean: %dx%d", ksize, ksize);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

		imshow("dst", dst);	//이거 3개 다 출력되도록 수정할 것.
	
	}
}

void blurring_gaussian()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	imshow("src", src);
	
	Mat dst;


	/*int sigma = 5;
	GaussianBlur(src, dst, Size(), sigma);
	String desc = format("sigma: %d", sigma);
	putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	imshow("dst", dst);*/
	
	for (int sigma = 1; sigma <= 5; sigma ++) {
		GaussianBlur(src, dst, Size(), sigma);

		String desc = format("sigma: %d", sigma);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
		imshow("dst", dst);
		
		//sigma[];
		//for (int i = 1; i < 5; i++) {
		//	sigma[i]
		//		imshow("dst", dst);	//이거 3개 다 출력되도록 수정할 것.
		//}
	}

}


void unsharp_mask()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	imshow("src", src);

	for (int sigma = 1; sigma <= 5; sigma++) {
		Mat blurred;
		GaussianBlur(src, blurred, Size(), sigma);

		float alpha = 1.f;
		Mat dst = (1 + alpha) * src - alpha * blurred;


		String desc = format("sigma: %d", sigma);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
		//imshow("dst", dst);

		String windowName = "dst_sigma_" + to_string(sigma);	//이거로 창 여러개 만들 수 있음
		imshow(windowName, dst);
		//waitKey();
	}


}


void noise_gaussian()	//이거 다시 해볼것.
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	imshow("src", src);

	for (int stddev = 10; stddev <= 30; stddev += 10) {
		Mat noise(src.size(), CV_32SC1);
		randn(noise, 0, stddev);

		Mat dst = src + noise;
		//add(src, noise, dst);

		String desc = format("stddev: %d", stddev);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
		imshow("dst", dst);
		waitKey();

		//String windowName = "dst_stddev_" + to_string(stddev);	//이거로 창 여러개 만들 수 있음
		//imshow(windowName, dst);
		//waitKey();
	}

}


void filter_bilateral()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);
	Mat ori = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}
	
	Mat noise(src.size(), CV_32SC1);
	randn(noise, 0, 5);
	add(src, noise, src, Mat(), CV_8U);

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 5);

	Mat dst2;
	bilateralFilter(src, dst2, -1, 10, 5);

	imshow("ori", ori);
	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}



void filter_median()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	int num = (int)(src.total() * 0.1);
	for (int i = 0; i < num; i++) {
		int x = rand() % src.cols;
		int y = rand() % src.rows;
		src.at<uchar>(y, x) = (i % 2) * 255;
	}

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 1);

	Mat dst2;
	medianBlur(src, dst2, 3);

	Mat dst3;
	medianBlur(dst2, dst3, 3);

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
}


void affine_transform()
{
	Mat src = imread("tekapo.bmp");

	if (src.empty()) {
		cerr << "img olad fail" << endl;
		return;
	}

	Point2f srcPts[3], dstPts[3];
	srcPts[0] = Point2f(0, 0);
	srcPts[1] = Point2f(src.cols - 1, 0);
	srcPts[2] = Point2f(src.cols - 1, src.rows - 1);

	dstPts[0] = Point2f(50, 50);
	dstPts[1] = Point2f(src.cols - 100, 100);
	dstPts[2] = Point2f(src.cols - 50, src.rows - 50);

	Mat M = getAffineTransform(srcPts, dstPts);
	Mat dst;
	warpAffine(src, dst, M, Size());

	imshow("src", src);
	imshow("dst", dst);


}


//
//int main()
//{
//	//video_in();
//	//camera_in_video_out();
//	//rawLines();
//	//drawPolys();
//	//drawText2();
//	//mask_setTo();
//	//mask_copyTo();
//	//time_inverse();
//	//sum_mean_minMaxLoc();
//	//brightness1();
//	//contrast2();
//	//histgoram_stretching();
//	//substract();
//	//addsubmult();
//	//filter_embossing();
//	//blurring_mean();
//	//blurring_gaussian();
//
//	//unsharp_mask();
//	//noise_gaussian();
//	//filter_bilateral();
////	filter_median();
//	affine_transform();
//	waitKey();
//	destroyAllWindows();
//	return 0;
//}