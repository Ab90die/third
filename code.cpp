#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

#include <opencv2/core.hpp>

#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string>
void getFiles(std::string path, std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}
int main() {
    int64 t1, t2, t3, t4;
    double t1kpt, t1des, t1match_bf, t1match_knn;
    double t2kpt, t2des, t2match_bf, t2match_knn;
    double t3kpt, t3des, t3match_bf, t3match_knn;
    int count;
    std::vector<std::string> Fname;
    std::string path;
    path = "/home/muhammad/下载/第四次考核/dataset_任务三/archive";
    getFiles(path,Fname);
    for(count = 0;count<Fname.size();count++) {
        std::cout << Fname[count] <<std::endl;



    // 1. 读取图片
    const cv::Mat image1 = cv::imread("/home/muhammad/下载/第四次考核/dataset_任务三/template/template_1.jpg", 0); //Load as grayscale
    const cv::Mat image2 = cv::imread("/home/muhammad/下载/第四次考核/dataset_任务三/template/template_2.jpg",0);
    const cv::Mat image3 = cv::imread("/home/muhammad/下载/第四次考核/dataset_任务三/template/template_3.jpg",0);

    const cv::Mat image4 = cv::imread(Fname[count], 0); //Load as grayscale

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::KeyPoint> keypoints3;
    std::vector<cv::KeyPoint> keypoints4;

    cv::Ptr<cv::SiftFeatureDetector> sift1 = cv::SiftFeatureDetector::create();
    cv::Ptr<cv::SiftFeatureDetector> sift2 = cv::SiftFeatureDetector::create();
    cv::Ptr<cv::SiftFeatureDetector> sift3 = cv::SiftFeatureDetector::create();
    // 2. 计算特征点
    t1 = cv::getTickCount();
    sift1->detect(image1, keypoints1);
    t2 = cv::getTickCount();
    sift2->detect(image2,keypoints2);
    t3 = cv::getTickCount();
    sift3->detect(image3,keypoints3);
    t4 = cv::getTickCount();
    t1kpt = 1000.0*(t4-t1) / cv::getTickFrequency();
    t2kpt = 1000.0*(t4-t2) / cv::getTickFrequency();
    t3kpt = 1000.0*(t4-t3) / cv::getTickFrequency();
    sift1->detect(image4, keypoints4);
    sift2->detect(image4, keypoints4);
    sift3->detect(image4, keypoints4);


    // 3. 计算特征描述符
    cv::Mat descriptors1, descriptors2, descriptors3, descriptors4;
    t1 = cv::getTickCount();
    sift1->compute(image1, keypoints1, descriptors1);
    t2 = cv::getTickCount();
    sift2->compute(image2, keypoints2, descriptors2);
    t3 = cv::getTickCount();
    sift3->compute(image3, keypoints3, descriptors3);
    t4 = cv::getTickCount();
    t1des = 1000.0*(t4-t1) / cv::getTickFrequency();
    t2des = 1000.0*(t4-t2) / cv::getTickFrequency();
    t3des = 1000.0*(t4-t3) / cv::getTickFrequency();
    sift1->compute(image4, keypoints4, descriptors4);
    sift2->compute(image4, keypoints4, descriptors4);
    sift3->compute(image4, keypoints4, descriptors4);


    // 4. 特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher1 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    cv::Ptr<cv::DescriptorMatcher> matcher2 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    cv::Ptr<cv::DescriptorMatcher> matcher3 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    // cv::BFMatcher matcher(cv::NORM_L2);

    // (1) 直接暴力匹配
    // std::vector<cv::DMatch> matches;
    // t1 = cv::getTickCount();
    // matcher1->match(descriptors1, descriptors4, matches);
    // t4 = cv::getTickCount();
    // t1match_bf = 1000.0*(t4-t1) / cv::getTickFrequency();
    // 画匹配图
    //cv::Mat img_matches_bf;
    //drawMatches(image1, keypoints1, image4, keypoints4, matches, img_matches_bf);
    //imshow("bf_matches", img_matches_bf);

    // (2) KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn1_matches;
    std::vector<std::vector<cv::DMatch> > knn2_matches;
    std::vector<std::vector<cv::DMatch> > knn3_matches;
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good1_matches;
    std::vector<cv::DMatch> good2_matches;
    std::vector<cv::DMatch> good3_matches;
    t1 = cv::getTickCount();
    matcher1->knnMatch( descriptors1, descriptors4, knn1_matches, 2);
    for (auto & knn_matche : knn1_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good1_matches.push_back(knn_matche[0]);
        }
    }
    t2 = cv::getTickCount();
    matcher2->knnMatch( descriptors2, descriptors4, knn2_matches, 2);
    for (auto & knn_matche : knn2_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good2_matches.push_back(knn_matche[0]);
        }
    }
    t3 = cv::getTickCount();
    matcher3->knnMatch( descriptors3, descriptors4, knn3_matches, 2);
    for (auto & knn_matche : knn3_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good3_matches.push_back(knn_matche[0]);
        }
    }

    t4 = cv::getTickCount();
    t1match_knn = 1000.0*(t4-t1) / cv::getTickFrequency();
    t2match_knn = 1000.0*(t4-t2) / cv::getTickFrequency();
    t3match_knn = 1000.0*(t4-t3) / cv::getTickFrequency();

    // 画匹配图
    cv::Mat img_matches_knn1;
    cv::Mat img_matches_knn2;
    cv::Mat img_matches_knn3;
    drawMatches( image1, keypoints1, image4,keypoints4, good1_matches, img_matches_knn1, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    drawMatches( image2, keypoints2, image4,keypoints4, good2_matches, img_matches_knn2, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    drawMatches( image3, keypoints3, image4,keypoints4, good3_matches, img_matches_knn3, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::resize(img_matches_knn1,img_matches_knn1,cv::Size(),0.5,0.5);
    cv::imshow("knn1_matches", img_matches_knn1);
    cv::resize(img_matches_knn2,img_matches_knn2,cv::Size(),0.5,0.5);
    cv::imshow("knn2_matches", img_matches_knn2);
    cv::resize(img_matches_knn3,img_matches_knn3,cv::Size(),0.5,0.5);
    cv::imshow("knn3_matches", img_matches_knn3);

    std::string P1 = "/home/muhammad/图片/考核三/";
    std::string P2,P6 ,P7, P8;
    P2 = std::to_string(count+1);
    P1 += P2;
    std::string P3 = ".1.jpg";
    std::string P4 = ".2.jpg";
    std::string P5 = ".3.jpg";
    P6 = P1;
    P7 = P1;
    P8 = P1;
    P6 += P3;
    P7 += P4;
    P8 += P5;
    cv::imwrite(P6, img_matches_knn1);
    cv::imwrite(P7, img_matches_knn2);
    cv::imwrite(P8, img_matches_knn3);
    cv::waitKey(1);
    }
    std::cout << "图1特征点1检测耗时(ms)：" << t1kpt << std::endl;
    std::cout << "图1特征描述符1耗时(ms)：" << t1des << std::endl;
    std::cout << "BF特征匹配1耗时(ms)：" << t1match_bf << std::endl;
    std::cout << "KNN-NNDR特征匹配1耗时(ms)：" << t1match_knn << std::endl;
    std::cout << "图1特征点2检测耗时(ms)：" << t2kpt << std::endl;
    std::cout << "图1特征描述符2耗时(ms)：" << t2des << std::endl;
    std::cout << "BF特征匹配2耗时(ms)：" << t2match_bf << std::endl;
    std::cout << "KNN-NNDR特征匹配2耗时(ms)：" << t2match_knn << std::endl;
    std::cout << "图1特征点3检测耗时(ms)：" << t3kpt << std::endl;
    std::cout << "图1特征描述符3耗时(ms)：" << t3des << std::endl;
    std::cout << "BF特征匹配3耗时(ms)：" << t3match_bf << std::endl;
    std::cout << "KNN-NNDR特征匹配3耗时(ms)：" << t3match_knn << std::endl;
    return 0;
}
