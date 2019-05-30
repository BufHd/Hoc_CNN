#include "show_image.h"

// show channel out của layer 
// show: ma trận đầu ra của layer
// name: tên của layer
// height, width: kích thước đầu ra của layer
// channel: số channel ra của layer
void showC(Matrix show, std::string name, int height, int width,int channel)
{	
	
	for (int i = 0; i < channel; i++) {
		Matrix blockShow = show.block(i*height*width, 0, height*width, 1);

		Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> m_show(blockShow.data(), height, width);

		Matrix m_show2 = m_show;
		m_show2.transposeInPlace();

		cv::Mat showimg;
		cv::eigen2cv(m_show2, showimg);   // chuyển từ eigen sang opencv

		cv::Mat fir;
		showimg.convertTo(fir, CV_8UC1, 1, 0);

		std::string nameC = "";
		
		nameC = name+ " Channel"+ std::to_string(i);
		cv::namedWindow(nameC, cv::WINDOW_NORMAL);
		cv::imshow(nameC, fir);
	}
	
}

// show channel out của layer 
// show: ma trận đầu ra của layer
// height, width: kích thước đầu ra của layer
// channel: số channel ra của layer
void showC(Matrix show, int height, int width, int channel)
{

	for (int i = 0; i < channel; i++) {
		Matrix blockShow = show.block(i*height*width, 0, height*width, 1);

		Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> m_show(blockShow.data(), height, width);

		Matrix m_show2 = m_show;
		m_show2.transposeInPlace();

		cv::Mat showimg;
		cv::eigen2cv(m_show2, showimg);   // chuyển từ eigen sang opencv

		cv::Mat fir;
		showimg.convertTo(fir, CV_8UC1, 1, 0);

		std::string nameC = "";

		nameC = "Channel" + std::to_string(i+1);
		cv::imshow(nameC, fir);
		//cv::waitKey(1);
	}

}

// show image  
// img: Ma trận của ảnh ở dạng vector (size(),1)
// height, width: kích thước của ảnh
void showI(Matrix img, int height, int width)
{
	Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> m_show(img.data(), height, width);
	Matrix m_show2 = m_show;
	m_show2.transposeInPlace();

	cv::Mat showimg;
	cv::eigen2cv(m_show2, showimg);   // chuyển từ eigen sang opencv

	cv::Mat fir;
	showimg.convertTo(fir, CV_8UC1, 1, 0);
	//cv::namedWindow("image src", cv::WINDOW_NORMAL);
	cv::imshow("image src", fir);
	//cv::waitKey(1);
}


// show kernel của layer 
//  w: ma trận của layer ở dạng vector
// height, width: kích thước của kernel
// channel: số channel vào của layer
void showW(Matrix w, int height, int width, int channel)
{
	int n_w = w.cols();
	for (int j = 0; j < n_w; j++) {

		for (int i = 0; i < channel; i++) {
			Matrix blockShow = w.block(i*height*width, j, height*width, 1);

			Eigen::Map<Eigen::MatrixXd, Eigen::RowMajor> m_show(blockShow.data(), height, width);

			Matrix m_show2 = m_show;
			m_show2.transposeInPlace();

			m_show2 = m_show2 * 1000;
			cv::Mat showimg;
			cv::eigen2cv(m_show2, showimg);   // chuyển từ eigen sang opencv

			cv::Mat fir;
			showimg.convertTo(fir, CV_8UC1, 1, 0);

			std::string nameC = "";

			nameC = "W" + std::to_string(j+1);
			cv::namedWindow(nameC, cv::WINDOW_NORMAL);
			cv::imshow(nameC, fir);
			//cv::waitKey(1);
		}
	}
}
