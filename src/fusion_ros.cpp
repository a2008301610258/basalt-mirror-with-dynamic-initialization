
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/li_estimator/eskf_lio.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/system_utils.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

/// VIO
basalt::Calibration<double> calib;  /// calibration
basalt::VioConfig vio_config; /// vio config
basalt::OpticalFlowBase::Ptr opt_flow_ptr;  /// optical flow
basalt::VioEstimatorBase::Ptr vio; /// vio

/// LIO
basalt::EskfLioEstimator::Ptr lio;  /// lio

basalt::StatesGroup g_lio_state; /// current state

/// For Vis
tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue;

tbb::concurrent_bounded_queue<basalt::LioVisualizationData::Ptr> out_lio_vis_queue;

/// Ros Publish
ros::Publisher traj_pub;
ros::Publisher pose_pub;
ros::Publisher left_cam_pub;
ros::Publisher right_cam_pub;
ros::Publisher window_pose_pub;
ros::Publisher points_pub;

ros::Publisher lio_pose_pub;
ros::Publisher lio_traj_pub;
ros::Publisher lio_cloud_pub;
ros::Publisher lio_kf_cloud_pub;

std::atomic<bool> terminate = false;

void callback_inertial(const sensor_msgs::Imu::ConstPtr &msg) {
//  std::cout << "callback_inertial" << std::endl;
  if (vio->finished || terminate) {
    return;
  }

  basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
  int64_t timestamp = msg->header.stamp.sec * 1.0e9 + msg->header.stamp.nsec;
  data->t_ns = timestamp;

  data->accel = basalt::ImuData<double>::Vec3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  data->gyro = basalt::ImuData<double>::Vec3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  vio->imu_data_queue.push(data);

  lio->imu_data_queue.push(data);
}

sensor_msgs::ImagePtr convert_image(basalt::ImageData img, Eigen::aligned_vector<Eigen::Vector4d> projections) {
  sensor_msgs::ImagePtr img_ptr(new sensor_msgs::Image);
  img_ptr->encoding = "mono8";
  img_ptr->width = img.img->w;
  img_ptr->height = img.img->h;

  img_ptr->data.resize(img_ptr->width * img_ptr->height);
  uint8_t *data_out = img_ptr->data.data();
  const uint16_t *data_in = img.img->ptr;

  for (size_t i = 0; i < img.img->size(); i++) {
    uint16_t val = data_in[i];
    val = val >> 8;
    data_out[i] = val;
  }
  return img_ptr;
}

basalt::ImageData get_image_data(const sensor_msgs::ImageConstPtr &img_msg) {
  basalt::ImageData id;
  id.img.reset(new basalt::ManagedImage<uint16_t>(img_msg->width, img_msg->height));

  if (!img_msg->header.frame_id.empty() &&
      std::isdigit(img_msg->header.frame_id[0])) {
    id.exposure = std::stol(img_msg->header.frame_id) * 1e-9;
  } else {
    id.exposure = -1;
  }

  if (img_msg->encoding == "mono8") {
    const uint8_t *data_in = img_msg->data.data();
    uint16_t *data_out = id.img->ptr;

    for (size_t i = 0; i < img_msg->data.size(); i++) {
      int val = data_in[i];
      val = val << 8;
      data_out[i] = val;
    }

  } else if (img_msg->encoding == "mono16") {
    std::memcpy(id.img->ptr, img_msg->data.data(), img_msg->data.size());
  } else {
    std::cerr << "Encoding " << img_msg->encoding << " is not supported."
              << std::endl;
    std::abort();
  }

  return id;
}

void callback_stereo(const sensor_msgs::ImageConstPtr &msg0, const sensor_msgs::ImageConstPtr &msg1, int cam_id0,
                     int cam_id1) {
//  std::cout << "callback_stereo" << std::endl;
  if (vio->finished || terminate) {
    return;
  }

  std::vector<basalt::ImageData> res(2);
  res[0] = get_image_data(msg0);
  res[1] = get_image_data(msg1);

  basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);
  int64_t timestamp = msg0->header.stamp.sec * 1.0e9 + msg0->header.stamp.nsec;
  data->t_ns = timestamp;
  data->img_data = res;

  opt_flow_ptr->input_queue.push(data);
}

void callback_lidar(const sensor_msgs::PointCloud2ConstPtr &msg) {
//  std::cout << "callback_lidar" << std::endl;
  const double blind = 0.1;
  const int point_filter_num = 1;

  basalt::LidarData::Ptr data(new basalt::LidarData);

  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);

//  ros::Time first_point_time(pl_orig.points[0].timestamp);

  data->lidar.clear();
  data->lidar.reserve(pl_orig.points.size());

  for (size_t i = 0; i < pl_orig.points.size(); i++) {
    if (std::isnan(pl_orig.points[i].x) || std::isnan(pl_orig.points[i].y) ||
        std::isnan(pl_orig.points[i].z)) {
      continue;
    }
    double range = std::sqrt(pl_orig.points[i].x * pl_orig.points[i].x +
                             pl_orig.points[i].y * pl_orig.points[i].y +
                             pl_orig.points[i].z * pl_orig.points[i].z);
    if (range < blind) {
      continue;
    }
    if (i % point_filter_num == 0) {
      Eigen::Vector3d pt_vec;
      basalt::PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = std::atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0) yaw_angle -= 360.0;
      if (yaw_angle <= -180.0) yaw_angle += 360.0;

      added_pt.curvature = (pl_orig.points[i].timestamp - pl_orig.points[0].timestamp) * 1000.0;  /// ms
      //            std::cout << "added_pt.curvature: " << added_pt.curvature << " " << (pl_orig.points[ i ].timestamp - pl_orig.points[ 0 ].timestamp) << std::endl;
//      pl_processed.points.push_back(added_pt);
      data->lidar.push_back(added_pt);
    }
  }

//  std::cout << "pl_processed size=" << data->lidar.points.size() << std::endl;

  data->lidar.height = 1;
  data->lidar.width = data->lidar.size();

  data->t_ns = pl_orig.points[0].timestamp * 1.0e9; //lidar_buffer.front()->header.stamp.toSec();
  int64_t lidar_end_time = data->t_ns + data->lidar.points.back().curvature / double(1000) * 1.0e9;
  data->t_ns_end = lidar_end_time;

  lio->lidar_data_queue.push(data);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}

void vio_vis_loop() {
  basalt::VioVisualizationData::Ptr vio_data;
  basalt::PoseVelBiasState<double>::Ptr state_data;

  Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

  while (true) {
    /// vio data
    out_vis_queue.pop(vio_data);
    if (vio_data.get()) {

      int left_cam_id = 0;
      int right_cam_id = 1;

      const auto& points0 = vio_data->projections[left_cam_id];
      const auto& img_data0 = vio_data->opt_flow_res->input_images->img_data.at(left_cam_id);

      const auto& points1 = vio_data->projections[right_cam_id];
      const auto& img_data1 = vio_data->opt_flow_res->input_images->img_data.at(right_cam_id);

      ///////////////////////////// left and right cam ////////////////////////////////
      sensor_msgs::ImagePtr image0Ptr = convert_image(img_data0, points0);
      image0Ptr->header.stamp = ros::Time::now();
      image0Ptr->header.frame_id = "/global";
      left_cam_pub.publish(image0Ptr);

      sensor_msgs::ImagePtr image1Ptr = convert_image(img_data1, points1);
      image1Ptr->header.stamp = ros::Time::now();
      image1Ptr->header.frame_id = "/global";
      right_cam_pub.publish(image1Ptr);

      ////////////////////////// pose within window ///////////////////////////////////

      ///////////////////////// points ////////////////////////////////////////
      sensor_msgs::PointCloud points_cloud;
      for (const auto &p : vio_data->points) {
        geometry_msgs::Point32 pt;
        pt.x = p.x();
        pt.y = p.y();
        pt.z = p.z();
        points_cloud.points.emplace_back(pt);
      }
      if (!points_cloud.points.empty()) {
        points_cloud.header.stamp = ros::Time::now();
        points_cloud.header.frame_id = "/global";
        points_pub.publish(points_cloud);
      }
    }

    /// state data
    out_state_queue.pop(state_data);

    if (state_data.get()) {
//      int64_t t_ns = state_data->t_ns;

      // std::cerr << "t_ns " << t_ns << std::endl;
      Sophus::SE3d T_w_i = state_data->T_w_i;
//      Eigen::Vector3d vel_w_i = state_data->vel_w_i;
//      Eigen::Vector3d bg = state_data->bias_gyro;
//      Eigen::Vector3d ba = state_data->bias_accel;

      ////////////////////// Pose ///////////////////////
      geometry_msgs::PoseStamped curr_pose;
      curr_pose.header.stamp = ros::Time::now();
      curr_pose.header.frame_id = "/global";
      curr_pose.pose.position.x = T_w_i.translation().x();
      curr_pose.pose.position.y = T_w_i.translation().y();
      curr_pose.pose.position.z = T_w_i.translation().z();
      curr_pose.pose.orientation.x = T_w_i.so3().unit_quaternion().x();
      curr_pose.pose.orientation.y = T_w_i.so3().unit_quaternion().y();
      curr_pose.pose.orientation.z = T_w_i.so3().unit_quaternion().z();
      curr_pose.pose.orientation.w = T_w_i.so3().unit_quaternion().w();

      pose_pub.publish(curr_pose);
      //////////////////////// Traj ///////////////////////

      vio_t_w_i.emplace_back(T_w_i.translation());

      if (vio_t_w_i.size() > 1) {
        visualization_msgs::Marker traj_marker;
        traj_marker.header.stamp = ros::Time::now();
        traj_marker.header.frame_id = "/global";
        traj_marker.ns = "trajectory";
        traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
        traj_marker.action = visualization_msgs::Marker::ADD;
        traj_marker.pose.orientation.w = 1.0;

        std_msgs::ColorRGBA color;
        color.r = 1.0; color.g = 1.0; color.b = 0.0; color.a = 1.0;

        traj_marker.color = color;
        traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.02;
        for (auto trans : vio_t_w_i) {
          geometry_msgs::Point point;
          point.x = trans.x();
          point.y = trans.y();
          point.z = trans.z();
          traj_marker.points.emplace_back(point);
        }
        traj_pub.publish(traj_marker);
      }
    }
  }
}

void lio_vis_loop() {
  basalt::LioVisualizationData::Ptr lio_data;

  Eigen::aligned_vector<Eigen::Vector3d> lio_t_w_i;

  while (true) {
    /// vio data
    out_lio_vis_queue.pop(lio_data);
    if (lio_data.get()) {
      Sophus::SE3d T_w_i = lio_data->state;
      double timestamp = lio_data->t_ns * 1.0e-9;

      ////////////////////// Cloud /////////////////////
      sensor_msgs::PointCloud2 laserCloud;
      pcl::toROSMsg(lio_data->curr_cloud, laserCloud);
      laserCloud.header.stamp.fromSec(timestamp);
      laserCloud.header.frame_id = "/global";
      lio_cloud_pub.publish(laserCloud);

      if (lio_data->is_kf) {
        lio_kf_cloud_pub.publish(laserCloud);
      }

      ////////////////////// Pose ///////////////////////
      geometry_msgs::PoseStamped curr_pose;
      curr_pose.header.stamp = ros::Time::now();
      curr_pose.header.frame_id = "/global";
      curr_pose.pose.position.x = T_w_i.translation().x();
      curr_pose.pose.position.y = T_w_i.translation().y();
      curr_pose.pose.position.z = T_w_i.translation().z();
      curr_pose.pose.orientation.x = T_w_i.so3().unit_quaternion().x();
      curr_pose.pose.orientation.y = T_w_i.so3().unit_quaternion().y();
      curr_pose.pose.orientation.z = T_w_i.so3().unit_quaternion().z();
      curr_pose.pose.orientation.w = T_w_i.so3().unit_quaternion().w();

      lio_pose_pub.publish(curr_pose);

      //////////////////////// Traj ///////////////////////
      if (lio_data->is_kf) {
        lio_t_w_i.emplace_back(T_w_i.translation());
      }

      if (lio_t_w_i.size() > 1) {
        visualization_msgs::Marker traj_marker;
        traj_marker.header.stamp = ros::Time::now();
        traj_marker.header.frame_id = "/global";
        traj_marker.ns = "trajectory";
        traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
        traj_marker.action = visualization_msgs::Marker::ADD;
        traj_marker.pose.orientation.w = 1.0;

        std_msgs::ColorRGBA color;
        color.r = 1.0; color.g = 1.0; color.b = 0.0; color.a = 1.0;

        traj_marker.color = color;
        traj_marker.scale.x = traj_marker.scale.y = traj_marker.scale.z = 0.02;
        for (auto trans : lio_t_w_i) {
          geometry_msgs::Point point;
          point.x = trans.x();
          point.y = trans.y();
          point.z = trans.z();
          traj_marker.points.emplace_back(point);
        }
        lio_traj_pub.publish(traj_marker);
      }
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vio_ros");

  /// parameters
  std::string cam_calib_path;
  std::string dataset_type;
  std::string config_path;
  int num_threads = 0;
  bool use_imu = true;
  bool use_double = true; //true;

  CLI::App app{"App description"};

  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")
      ->required();

  app.add_option("--config-path", config_path, "Path to config file.");

  app.add_option("--num-threads", num_threads, "Number of threads.");

  app.add_option("--use-imu", use_imu, "Use IMU.");
  app.add_option("--use-double", use_double, "Use double not float.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  std::cout << "cam_calib_path: " << cam_calib_path << std::endl;
  std::cout << "cam_calib_path: " << dataset_type << std::endl;
  std::cout << "cam_calib_path: " << config_path << std::endl;
  std::cout << "num_threads: " << num_threads << std::endl;

  /// global thread limit is in effect until global_control object is destroyed
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, num_threads);
  }

  /// vio_config
  if (!config_path.empty()) {
    vio_config.load(config_path);

    if (vio_config.vio_enforce_realtime) {
      vio_config.vio_enforce_realtime = false;
      std::cout
          << "The option vio_config.vio_enforce_realtime was enabled, "
             "but it should only be used with the live executables (supply "
             "images at a constant framerate). This executable runs on the "
             "datasets and processes images as fast as it can, so the option "
             "will be disabled. "
          << std::endl;
    }
  }

  /// calib
  load_data(cam_calib_path);

  /// opt_flow_ptr
  {
    opt_flow_ptr = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);
  }

  /// vio
  {
    vio = basalt::VioEstimatorFactory::getVioEstimator(
        vio_config, calib, basalt::constants::g, use_imu, use_double);

    ////
    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    opt_flow_ptr->output_queue = &vio->vision_data_queue;

    /// for visualization
    vio->out_vis_queue = &out_vis_queue;
    vio->out_state_queue = &out_state_queue;
  }

  /// lio
  {
    lio = std::make_shared<basalt::EskfLioEstimator>();
    lio->out_vis_queue = &out_lio_vis_queue;
  }

  std::string imu_topic = "/camera/imu";
  std::string cam_topic0 = "/camera/fisheye1/image_raw";
  std::string cam_topic1 = "/camera/fisheye2/image_raw";
  std::string lidar_topic = "/rslidar_points";

//  std::string imu_topic = "/imu0";
//  std::string cam_topic0 = "/cam0/image_raw";
//  std::string cam_topic1 = "/cam1/image_raw";
//  std::string lidar_topic = "/rslidar_points";

  ros::NodeHandle nh;
  ros::Subscriber sub_imu;
  sub_imu = nh.subscribe(imu_topic, 100, callback_inertial);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  auto image_sub0 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh, cam_topic0, 1);
  auto image_sub1 = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh, cam_topic1, 1);
  auto sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(sync_pol(5), *image_sub0, *image_sub1);
  sync->registerCallback(boost::bind(&callback_stereo, _1, _2, 0, 1));
  ros::Subscriber sub_lidar;
  sub_lidar = nh.subscribe(lidar_topic, 100, callback_lidar);

  traj_pub = nh.advertise<visualization_msgs::Marker>("/imu_traj", 2);
  pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/imu_pose", 2);
  left_cam_pub = nh.advertise<sensor_msgs::Image>("/left_cam", 2);
  right_cam_pub = nh.advertise<sensor_msgs::Image>("/right_cam", 2);
  window_pose_pub = nh.advertise<visualization_msgs::Marker>("/imu_window", 2);
  points_pub = nh.advertise<sensor_msgs::PointCloud>("/points_cloud", 2);

  lio_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/lio_imu_pose", 2);
  lio_traj_pub = nh.advertise<visualization_msgs::Marker>("/lio_imu_traj", 2);
  lio_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lio_cloud", 100);
  lio_kf_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/lio_kf_cloud", 100);

  std::thread vio_vis_thread(vio_vis_loop);
  std::thread lio_vis_thread(lio_vis_loop);

  ros::spin();

//  std::cout << "spin" << std::endl;
//  vio->maybe_join();
//  std::cout << "maybe_join" << std::endl;
//  vio->drain_input_queues();

  terminate = true;

  std::cout << "aaaa" << std::endl;
  vio_vis_thread.join();
  lio_vis_thread.join();
  std::cout << "bbbb" << std::endl;

  return 0;
}