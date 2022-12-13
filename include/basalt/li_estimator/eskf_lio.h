/* 
This code is the implementation of our paper "R3LIVE: A Robust, Real-time, RGB-colored, 
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored, 
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package." 
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping." 
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry 
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for 
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision 
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
//#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <future>
#include "basalt/li_estimator/so3_math.h"
#include <ros/ros.h>
#include <Eigen/Core>
#include "basalt/li_estimator/common_lib.h"
#include "basalt/li_estimator/ikd_tree.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/eigen.hpp>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Vector3.h>

//#include <opencv2/highgui/highgui.hpp>
//#include <cv_bridge/cv_bridge.h>
#include <tbb/concurrent_queue.h>
#include <basalt/imu/imu_types.h>

#include <basalt/utils/time_utils.hpp>
#include <basalt/li_estimator/imu_processing.hpp>
#include <basalt/li_estimator/voxel_indices_grid.h>

namespace velodyne_ros {

struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  uint8_t intensity;
  uint16_t ring;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)
                                      (float, y, y)
                                      (float, z, z)
                                      (uint8_t, intensity, intensity)
                                      (uint16_t, ring, ring)
                                      (double, timestamp, timestamp)
)

// estimator inputs and output;
extern basalt::StatesGroup g_lio_state;
//extern std::shared_ptr<basalt::ImuProcess> g_imu_process;

namespace basalt {

#define INIT_TIME (0)
// #define LASER_POINT_COV (0.0015) // Ori
#define LASER_POINT_COV (0.00015)
#define NUM_MATCH_POINTS (5)

#define MAXN 360000
const int laserCloudWidth = 48;
const int laserCloudHeight = 48;
const int laserCloudDepth = 48;
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

struct LioVisualizationData {
  typedef std::shared_ptr<LioVisualizationData> Ptr;

  int64_t t_ns;
  Sophus::SE3d state;
  pcl::PointCloud<pcl::PointXYZI> curr_cloud;
  bool is_kf;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class EskfLioEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Ptr = std::shared_ptr<EskfLioEstimator>;
//    std::mutex  m_mutex_lio_process;
    std::shared_ptr<basalt::ImuProcess> m_imu_process;

    double m_maximum_pt_kdtree_dis = 1.0;
    double m_maximum_res_dis = 1.0;
    double m_planar_check_dis = 0.05;
    double m_lidar_imu_time_delay = 0;
    double m_long_rang_pt_dis = 500.0;
    bool m_if_publish_feature_map = true;
    int iterCount = 0;
    int NUM_MAX_ITERATIONS = 0;
    int FOV_RANGE = 4; // range of FOV = FOV_RANGE * cube_len
    int laserCloudCenWidth = 24;
    int laserCloudCenHeight = 24;
    int laserCloudCenDepth = 24;
    int laserCloudSelNum = 0;

    /// IMU relative variables
//    std::mutex mtx_buffer;
//    std::condition_variable sig_buffer;
    bool lidar_pushed = false;
    bool flg_exit = false;
    bool flg_reset = false;

    // Buffers for measurements
    double cube_len = 0.0;
    double lidar_end_time = 0.0;
    double last_timestamp_lidar = -1;
    double last_timestamp_imu = -1;
    double HALF_FOV_COS = 0.0;
    double FOV_DEG = 0.0;
    double res_mean_last = 0.05;
    double total_distance = 0.0;
    double frame_first_pt_time = 0;
    Eigen::Vector3d position_last = basalt::Zero3d;

  bool initialized;

    tbb::concurrent_bounded_queue<LidarData::Ptr> lidar_data_queue;
    tbb::concurrent_bounded_queue<ImuData<double>::Ptr> imu_data_queue;

    tbb::concurrent_bounded_queue<LioVisualizationData::Ptr>* out_vis_queue =
      nullptr;

//    std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
//    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_lio;

    //surf feature in map
    basalt::PointCloudXYZINormal::Ptr featsFromMap;
    basalt::PointCloudXYZINormal::Ptr cube_points_add;
    //all points
    basalt::PointCloudXYZINormal::Ptr laserCloudFullRes2;

    Eigen::Vector3f XAxisPoint_body; //(LIDAR_SP_LEN, 0.0, 0.0);
    Eigen::Vector3f XAxisPoint_world; //(LIDAR_SP_LEN, 0.0, 0.0);

    std::vector<basalt::BoxPointType> cub_needrm;
    std::vector<basalt::BoxPointType> cub_needad;

    basalt::PointCloudXYZINormal::Ptr featsArray[laserCloudNum];
    bool _last_inFOV[laserCloudNum];
    bool now_inFOV[laserCloudNum];
    bool cube_updated[laserCloudNum];
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor; //(new pcl::PointCloud<pcl::PointXYZI>());

    basalt::KD_TREE ikdtree;

//    ros::Publisher pubLaserCloudFullRes;
//    ros::Publisher pubLaserCloudFullRes1;
//    ros::Publisher pubLaserCloudMap;
//    ros::Publisher pubOdomAftMapped;
//    ros::Publisher pubPath;
    ros::Subscriber sub_lidar;
    ros::Subscriber sub_imu;

    bool dense_map_en, flg_EKF_inited = 0, flg_EKF_converged = 0;
    int frame_num = 0;
    double m_voxel_downsample_size_surf, filter_size_map_min, fov_deg, deltaT, deltaR, aver_time_consu = 0;
    double m_voxel_downsample_size_axis_z;
    geometry_msgs::PoseStamped msg_body_pose;
    geometry_msgs::PoseStamped odomAftMapped;
    pcl::VoxelGrid<basalt::PointType> downSizeFilterSurf;
    pcl::ApproximateVoxelGrid<basalt::PointType> downSizeAprxFilterSurf;
    pcl::VoxelGrid<basalt::PointType> downSizeFilterMap;

    pcl::VoxelIndicesGrid<basalt::PointType> downSizeFilter;

    double blind;
    int point_filter_num;

    /*** debug record ***/
    ros::NodeHandle m_ros_node_handle;

    int g_LiDAR_frame_index = 0;
    int m_lio_update_point_step = 1;

    std::shared_ptr<std::thread> processing_thread;

    void set_initial_state_cov(basalt::StatesGroup &stat);

//    typename ImuData<double>::Ptr popFromImuDataQueue();
    
//    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);

//    void lidar_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);

//    bool sync_packages(basalt::MeasureGroup &meas);

    EskfLioEstimator() {
      initialized = false;
      lidar_data_queue.set_capacity(10);
      imu_data_queue.set_capacity(300);
//        pubLaserCloudFullRes = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
//        pubLaserCloudFullRes1 = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/cloud_registered1", 100);
//        pubLaserCloudMap = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
//        pubOdomAftMapped = m_ros_node_handle.advertise<geometry_msgs::PoseStamped>("/aft_mapped_to_init", 10);
//        pubPath = m_ros_node_handle.advertise<visualization_msgs::Marker>("/path", 10);

        std::string IMU_topic, LiDAR_topic;
        LiDAR_topic = "/rslidar_points";
        IMU_topic = "/camera/imu";

        std::cout << "======= Summary of subscribed topics =======" << std::endl;
        std::cout << "IMU topic: " << IMU_topic << std::endl;
        std::cout << "=======        -End-                =======" << std::endl;

//        sub_imu = m_ros_node_handle.subscribe(IMU_topic.c_str(), 2000000, &EskfLioEstimator::imu_cbk, this, ros::TransportHints().tcpNoDelay());
//        sub_lidar = m_ros_node_handle.subscribe(LiDAR_topic.c_str(), 2000000, &EskfLioEstimator::lidar_cbk, this, ros::TransportHints().tcpNoDelay());

        if ( 1 )
        {
            dense_map_en = true;
            m_lidar_imu_time_delay = 0.0;
            NUM_MAX_ITERATIONS = 2;
            fov_deg = 360.0;
            m_voxel_downsample_size_surf = 0.3;
            m_voxel_downsample_size_axis_z = m_voxel_downsample_size_surf;
            filter_size_map_min = 0.30;
            cube_len = 10000000.0;
            m_maximum_pt_kdtree_dis = 0.5;
            m_maximum_res_dis = 0.3;
            m_planar_check_dis = 0.10;
            m_long_rang_pt_dis = 500.0;
            m_if_publish_feature_map = true;
            m_lio_update_point_step = 4;
        }
        featsFromMap = boost::make_shared<basalt::PointCloudXYZINormal>();
        cube_points_add = boost::make_shared<basalt::PointCloudXYZINormal>();
        laserCloudFullRes2 = boost::make_shared<basalt::PointCloudXYZINormal>();
        laserCloudFullResColor = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        XAxisPoint_body = Eigen::Vector3f(LIDAR_SP_LEN, 0.0, 0.0);
        XAxisPoint_world = Eigen::Vector3f(LIDAR_SP_LEN, 0.0, 0.0);

        downSizeFilterSurf.setLeafSize(m_voxel_downsample_size_surf, m_voxel_downsample_size_surf, m_voxel_downsample_size_axis_z);
        downSizeAprxFilterSurf.setLeafSize(m_voxel_downsample_size_surf, m_voxel_downsample_size_surf, m_voxel_downsample_size_axis_z);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
        blind = 0.1;
        point_filter_num = 1;

//        lio_thread = std::thread(std::bind(&EskfLioEstimator::service_LIO_update, this));
      processing_thread.reset(new std::thread(std::bind(&EskfLioEstimator::proc_func, this)));
    }
    ~EskfLioEstimator() {
      if (processing_thread) {
        processing_thread->join();
        processing_thread.reset();
      }
};

    //project lidar frame to world
    void pointBodyToWorld(basalt::PointType const *const pi, basalt::PointType *const po);

    template <typename T>
    void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po)
    {
        Eigen::Vector3d p_body(pi[0], pi[1], pi[2]);
        Eigen::Vector3d p_global(g_lio_state.rot_end * (basalt::Lidar_R_to_IMU * p_body + basalt::Lidar_offset_to_IMU) + g_lio_state.pos_end);
        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }
    void RGBpointBodyToWorld(basalt::PointType const *const pi, pcl::PointXYZI *const po);
    int get_cube_index(const int &i, const int &j, const int &k);
    void lasermap_fov_segment();

//    bool get_pointcloud_data_from_ros_message(sensor_msgs::PointCloud2::ConstPtr & msg, pcl::PointCloud<pcl::PointXYZINormal> & pcl_pc);
    int proc_func();
    void print_dash_board();
};
}