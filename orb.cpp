#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <fstream>
#include <list>
#include <opencv/cv.h>
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Frame.h"
#include <experimental/filesystem>
#include <string>


using namespace std;
using namespace cv;
using namespace ORB_SLAM2;
namespace fs = std::experimental::filesystem;




// 计算两个特征点之间的距离
float getDistance(const KeyPoint& kp1, const KeyPoint& kp2)
{
    return std::sqrt(std::pow(kp1.pt.x - kp2.pt.x, 2) + std::pow(kp1.pt.y - kp2.pt.y, 2));
}

// 删除距离过近的特征点 返回删除特征点后的描述子匹配点对
std::vector<DMatch> removeDuplicates(const std::vector<KeyPoint>& keypoints, float minDistance , const vector<DMatch> matches)
{
    vector<DMatch> cut_matches;
    std::vector<KeyPoint> uniqueKeypoints;
    for (size_t i = 0; i < keypoints.size(); i++)
    {   int distance = 0;
        int points_num = 0;
        bool isDuplicate = false;
        for (size_t j = 0; j < keypoints.size(); j++)
        {
            distance += getDistance(keypoints[i], keypoints[j]);
            if(getDistance(keypoints[i], keypoints[j])<5)
              points_num++;
        }
        //cout<<"距离："<<distance<<endl;
        if(distance<minDistance)continue;
        if(points_num>5)continue;

        cut_matches.push_back(matches[i]);
        uniqueKeypoints.push_back(keypoints[i]);
    }
    cout<<"原本特征点数量："<<keypoints.size()<<endl;
    cout<<"剩余特征点数量："<<uniqueKeypoints.size()<<endl;
    return cut_matches;
}


bool find_feature_matches(const std::vector<cv::Mat>::iterator it1, const std::vector<cv::Mat>::iterator it2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches,
                          int img,
                          Mat K)
{
  
      //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 每一帧提取的特征点数 1000
    int nFeatures = 1000;
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = 1.2;
    // 尺度金字塔的层数 8
    int nLevels = 8;
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = 20;
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = 8;

    //1.检测角点位置,计算描述子
    ORBextractor orb_left(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    ORBextractor orb_right(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    
    orb_left(*it1,cv::Mat(),keypoints_1,descriptors_1);
    orb_right(*it2,cv::Mat(),keypoints_2,descriptors_2);

    //绘制角点图像
    Mat img_it1;
    Mat img_it2;
    drawKeypoints(*it1,keypoints_1,img_it1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    string filename_gtk = "gt_keypoint" + to_string(img) + ".png";
    imwrite(filename_gtk, img_it1);
    drawKeypoints(*it2,keypoints_2,img_it2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    string filename_NeRFk = "NeRF_keypoint" + to_string(img) + ".png";
    imwrite(filename_NeRFk, img_it2);

    
    //2.构建帧
    /**
     * @brief 为双目相机准备的构造函数
     * 
     * @param[in] imLeft            左目图像
     * @param[in] imRight           右目图像
     * @param[in] extractorLeft     左目图像特征点提取器句柄
     * @param[in] extractorRight    右目图像特征点提取器句柄
     * @param[in] K                 相机内参矩阵
     * @param[in] bf                相机基线长度和焦距的乘积 作者一开始给的就是40。参考https://blog.csdn.net/catpico/article/details/120688795
     * @param[in] thDepth           远点和近点的深度区分阈值 也是40
     *  
     */
    float bf = 40.0;
    float thDepth = 40.0;
    cv::Mat mImGray = *it1;
    cv::Mat imGrayRight = *it2;

    // step 2.1 ：将RGB图像转为灰度图像
    
    
    cvtColor(mImGray,mImGray,CV_RGB2GRAY);
    cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
    
    ORBextractor *mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    ORBextractor *mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
  

    ORB_SLAM2::Frame orb_frame(mImGray, imGrayRight, mpORBextractorLeft, mpORBextractorRight, K, bf, thDepth);



    //-- 1.2根据角点位置计算 BRIEF 描述子
    // descriptor->compute(*it1, keypoints_1, descriptors_1);
    // descriptor->compute(*it2, keypoints_2, descriptors_2);

   

    // //2.第二遍匹配，使用RANSAC 消除误匹配特征点

    //   //RANSAC 消除误匹配特征点 主要分为三个部分：
    //   //1）根据matches将特征点对齐,将坐标转换为float类型
    //   //2）使用求基础矩阵方法 findFundamentalMat,得到RansacStatus
    //   //3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除

    //   //2.1根据matches将特征点对齐,将坐标转换为float类型
    //   vector<KeyPoint> R_keypoints_1,R_keypoints_2;
    //   for (size_t i=0;i<ini_matches.size();i++)   
    //   {
    //       R_keypoints_1.push_back(keypoints_1[ini_matches[i].queryIdx]);
    //       R_keypoints_2.push_back(keypoints_2[ini_matches[i].trainIdx]);
    //       //这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点，
    //       //matches中存储了这些匹配点对的img01和img02的索引值
    //   }

    //   //坐标转换
    //   vector<Point2f>p01,p02;
    //   for (size_t i=0;i<ini_matches.size();i++)
    //   {
    //       p01.push_back(R_keypoints_1[i].pt);
    //       p02.push_back(R_keypoints_2[i].pt);
    //   }

    //   //2.2利用基础矩阵剔除误匹配点
    //   vector<KeyPoint> RR_keypoints_1,RR_keypoints_2;
    //   vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    //   int index=0;
    //   int ransacReprojThreshold = 15;  //拒绝阈值

    //   Mat H12;   //变换矩阵
    //   H12 = findHomography( Mat(p01), Mat(p02), CV_RANSAC, ransacReprojThreshold );
    //   Mat points1t;
    //   perspectiveTransform(Mat(p01), points1t, H12);

    //   for (size_t i=0;i<ini_matches.size();i++)
    //   {
        
    //       if( norm(p02[i] - points1t.at<Point2f>((int)i,0)) <= ransacReprojThreshold ) //给内点做标记
    //       {
    //           RR_keypoints_1.push_back(R_keypoints_1[i]);
    //           RR_keypoints_2.push_back(R_keypoints_2[i]);
    //           ini_matches[i].queryIdx=index;
    //           ini_matches[i].trainIdx=index;
    //           RR_matches.push_back(ini_matches[i]);
    //           index++;
    //       }
    //   }

    //   //2.3绘制ransac匹配点图片
    //   Mat img_RR_matches;
    //   drawMatches(*it1,RR_keypoints_1,*it2,RR_keypoints_2,RR_matches,img_RR_matches,Scalar(0,0,255));
    //   string filename_good = "image_good" + to_string(img) + ".png";
    //   imwrite(filename_good, img_RR_matches);

    //   //3.第三遍匹配，剔除太近的NeRF特征点
    //   //删除太近的NeRF特征点
    //   vector<DMatch> RR_cut_matches;//定义删除太近特征点后的描述子
    //   const float minDistance = 5000;  // 定义特征点最小距离
    //   RR_cut_matches = removeDuplicates(RR_keypoints_2, minDistance , RR_matches);
    //   if(RR_cut_matches.size()<5)return false;

    // vector<DMatch> orbgood_matches;
    // vector<cv::Point2f> vbPrevMatched;
    // vector<int> vnMatches12;
    // ORB_SLAM2::ORBmatcher matcher_orb(
    // 0.9,        //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
    // true);      //检查特征点的方向
    // // orbgood_matches = matcher_orb.SearchForInitialization(vbPrevMatched, vnMatches12,
    // // descriptors_1, descriptors_2, keypoints_1, keypoints_2, ini_matches );



    // Mat img_RR_cut_matches;
    // drawMatches(*it1,keypoints_1,*it2,keypoints_2,orbgood_matches,img_RR_cut_matches,Scalar(0,0,255));
    // string filename_good_cut = "image_good_cut" + to_string(img) + ".png";
    // imwrite(filename_good_cut, img_RR_cut_matches);

    // matches = orbgood_matches;
    return true;
  }





Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {
//   // 相机内参,Replica
//   Mat K = (Mat_<double>(3, 3) << 600.0, 0, 599.5, 0, 600.0, 339.5, 0, 0, 1);

  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
  cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

  //-- 计算本质矩阵
  Point2d principal_point(599.5, 339.5);  //相机光心
  double focal_length = 600;      //相机焦距
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
  cout << "essential_matrix is " << endl << essential_matrix << endl;

  //-- 计算单应矩阵
  //-- 但是本例中场景不是平面，单应矩阵意义不大
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, RANSAC, 3);
  cout << "homography_matrix is " << endl << homography_matrix << endl;

  //-- 从本质矩阵中恢复旋转和平移信息.
  // 此函数仅在Opencv3中提供
  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;

}


// 高斯模糊处理函数
void gaussianBlurFolder(const std::string& input_folder, const std::string& output_folder, int kernel_size)
{
    std::vector<std::string> files;
    for (auto& p : fs::directory_iterator(input_folder)) {
        if (p.path().extension() == ".jpg" || p.path().extension() == ".png") {
            files.push_back(p.path().string());
        }
    }

    for (auto& file : files) {
        Mat img = imread(file);
        Mat blur_img;
        if (img.empty()) {
            std::cout << "Could not read image: " << file << std::endl;
            continue;
        }
        GaussianBlur(img, blur_img, Size(kernel_size, kernel_size), 2, 2,BORDER_REFLECT_101);
        ////源图像;输出图像;高斯滤波器kernel大小，必须为正的奇数;高斯滤波在x方向的标准差;高斯滤波在y方向的标准差;边缘拓展点插值类型
        std::string output_file = output_folder + "/" + fs::path(file).filename().string();
        imwrite(output_file, blur_img);
    }
}



int main(int argc, char **argv) {
 
// 记录信息 打开文件，以输出模式写入
std::ofstream fout("output.txt");
// 检查文件是否成功打开
if (!fout.is_open()) 
{
  std::cerr << "Failed to open file!" << std::endl;
  return 1;
}

//高斯模糊处理gt图像
std::string input_folder = "../GT";
std::string output_folder = "../GTgauss";
gaussianBlurFolder(input_folder, output_folder, 7);


//读取gt图像 
std::string gt_folder_path = "../GTgauss/";

// 获取文件夹中所有的图像路径
std::vector<cv::String> gt_image_paths;
cv::glob(gt_folder_path, gt_image_paths);

// 读取所有的图像
std::vector<cv::Mat> gt_images;
for (const auto& path : gt_image_paths) 
{
  cv::Mat image = cv::imread(path);
  gt_images.push_back(image);
}



//读取NeRF图像 
std::string NeRF_folder_path = "../NeRF/";

// 获取文件夹中所有的图像路径
std::vector<cv::String> NeRF_image_paths;
cv::glob(NeRF_folder_path, NeRF_image_paths);

// 读取所有的图像
std::vector<cv::Mat> NeRF_images;
for (const auto& path : NeRF_image_paths) 
{
  cv::Mat image = cv::imread(path);
  NeRF_images.push_back(image);
}


std::vector<cv::Mat>::iterator it1 = gt_images.begin();
std::vector<cv::Mat>::iterator it2 = NeRF_images.begin();

int img=0;//用于输出图像的命名

// 相机内参,Replica
Mat K = (Mat_<double>(3, 3) << 600.0, 0, 599.5, 0, 600.0, 339.5, 0, 0, 1);


while (it1 < gt_images.end() && it2 < NeRF_images.end())
{


    // // 每一帧提取的特征点数 1000
    // int nFeatures = 1000;
    // // 图像建立金字塔时的变化尺度 1.2
    // float fScaleFactor = 1.2;
    // // 尺度金字塔的层数 8
    // int nLevels = 8;
    // // 提取fast特征点的默认阈值 20
    // int fIniThFAST = 20;
    // // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    // int fMinThFAST = 8;


  //-- 读取图像
  Mat img_1 = *it1;
  Mat img_2 = *it2;
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  if(find_feature_matches(it1, it2, keypoints_1, keypoints_2, matches, img, K))
  {
    // cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // //-- 估计两张图像间运动
    // Mat R, t;
    // pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // //-- 验证E=t^R*scale
    // Mat t_x =
    //     (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
    //     t.at<double>(2, 0), 0, -t.at<double>(0, 0),
    //     -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    // cout << "t^R=" << endl << t_x * R << endl;


    // fout << to_string(img) <<":"<< endl ;
    // fout << "R is " << endl << R << endl;
    // fout << "t is " << endl << t << endl;
    // fout << "t^R=" << endl << t_x * R << endl;

    // //-- 验证对极约束
    // Mat K = (Mat_<double>(3, 3) << 600.0, 0, 599.5, 0, 600.0, 339.5, 0, 0, 1);//replica
    // for (DMatch m: matches) 
    //     {
    //     Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    //     Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    //     Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    //     Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    //     Mat d = y2.t() * t_x * R * y1;
    //     cout << "epipolar constraint = " << d << endl;
    //     }
  }

  img++;
  it1++;
  it2++;	
}

 // 关闭文件
fout.close();
return 0;
}
