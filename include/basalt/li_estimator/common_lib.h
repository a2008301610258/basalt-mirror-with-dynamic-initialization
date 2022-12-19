#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include "so3_math.h"
#include "states_group.h"
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>

#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <queue>
#include <deque>
#include <memory>

namespace basalt {
// #define DEBUG_PRINT
#define USE_ikdtree
#define ESTIMATE_GRAVITY 1
#define ENABLE_CAMERA_OBS 1
// #define USE_FOV_Checker

#define printf_line std::cout << __FILE__ << " " << __LINE__ << std::endl;

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)  // Gravity const in Hong Kong SAR, China

#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) std::vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())

#define DEBUG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "Log/" + name))
// using vins_estimator = fast_lio;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZINormal;

static const Eigen::Matrix3d Eye3d(Eigen::Matrix3d::Identity());
static const Eigen::Matrix3f Eye3f(Eigen::Matrix3f::Identity());
static const Eigen::Vector3d Zero3d(0, 0, 0);
static const Eigen::Vector3f Zero3f(0, 0, 0);
// Eigen::Vector3d Lidar_offset_to_IMU(0.05512, 0.02226, 0.0297); // Horizon
// static const Eigen::Vector3d Lidar_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia
static const Eigen::Vector3d Lidar_offset_to_IMU(0.207000, -0.006000,
                                                 -0.014000);  // Rs32
static const Eigen::Matrix3d Lidar_R_to_IMU =
    Eigen::Quaterniond(
        Eigen::AngleAxisd(1.000000 * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(0.080000 * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.340000 * M_PI / 180.0, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();

struct Pose6D {
  typedef double data_type;
  data_type offset_time;
  data_type rot[9];
  data_type acc[3];
  data_type vel[3];
  data_type pos[3];
  data_type gyr[3];
};

struct LidarData {
  using Ptr = std::shared_ptr<LidarData>;

  int64_t t_ns_beg;  ///< timestamp in nanoseconds
  int64_t t_ns_end;
  PointCloudXYZINormal lidar;
};

struct MeasureGroup  // Lidar data and imu dates for the curent process
{
  MeasureGroup() {
//    this->lidar.reset(new PointCloudXYZINormal());
  };
  double lidar_beg_time_ns;
  double lidar_end_time_ns;
  PointCloudXYZINormal lidar;
  std::deque<ImuTypeData> imu;
};


template <typename T>
T rad2deg(T radians) {
  return radians * 180.0 / PI_M;
}

template <typename T>
T deg2rad(T degrees) {
  return degrees * PI_M / 180.0;
}

template <typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a,
                const Eigen::Matrix<T, 3, 1> &g,
                const Eigen::Matrix<T, 3, 1> &v,
                const Eigen::Matrix<T, 3, 1> &p,
                const Eigen::Matrix<T, 3, 3> &R) {
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++) {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
  }
  // Eigen::Map<Eigen::Matrix3d>(rot_kp.rot, 3,3) = R;
//  return std::move(rot_kp);
  return rot_kp;
}

inline bool time_list(PointType &x, PointType &y) {
  return (x.curvature < y.curvature);
};

static void undistort_point_cloud(const MeasureGroup &meas,
                           const StatesGroup &_state_inout,
                           PointCloudXYZINormal &pcl_out) {

  StatesGroup state_inout = _state_inout;
  auto v_imu = meas.imu;
  const int64_t &imu_beg_time_ns = v_imu.front().tm_ns;
  const int64_t &imu_end_time_ns = v_imu.back().tm_ns;
  const int64_t &pcl_beg_time_ns = meas.lidar_beg_time_ns;
  const int64_t &pcl_end_time_ns = meas.lidar_end_time_ns;

  if (imu_beg_time_ns >= pcl_beg_time_ns) {
    ImuTypeData imu_data = v_imu.front();
    imu_data.tm_ns = pcl_beg_time_ns;
    v_imu.push_front(imu_data);
  }
  if (imu_end_time_ns <= pcl_end_time_ns) {
    ImuTypeData imu_data = v_imu.back();
    imu_data.tm_ns = pcl_end_time_ns;
    v_imu.push_back(imu_data);
  }

  /*** sort point clouds by offset time ***/
  pcl_out = meas.lidar;
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  std::cout << std::setprecision(19) << "[ IMU Process ]: Process lidar from " << pcl_beg_time_ns << " to "
            << pcl_end_time_ns << ", " << meas.lidar_end_time_ns << ", "
            << meas.imu.size() << " imu msgs from " << imu_beg_time_ns << " to "
            << imu_end_time_ns << ", last tim: " << state_inout.last_update_time_ns
            << ", dt: " << (pcl_end_time_ns - imu_end_time_ns) * double(1e-9) << std::endl;

  /*** Initialize IMU pose ***/
  std::vector<Pose6D> IMU_pose;

  /*** forward propagation at each imu point ***/
  Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end),
      pos_imu(state_inout.pos_end);
  Eigen::Matrix3d R_imu(state_inout.rot_end);

  double first_offs = 0.0;
  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));

  Eigen::Vector3d acc_s_last = R_imu * (v_imu.front().linear_acceleration - state_inout.bias_a) - gravity;
  Eigen::Vector3d angvel_last = v_imu.front().angular_velocity - state_inout.bias_g;
  IMU_pose.push_back(set_pose6d(first_offs, acc_s_last, angvel_last,
                                state_inout.vel_end, state_inout.pos_end,
                                state_inout.rot_end));

  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

//    if (tail.tm < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head.angular_velocity.x() + tail.angular_velocity.x()),
        0.5 * (head.angular_velocity.y() + tail.angular_velocity.y()),
        0.5 * (head.angular_velocity.z() + tail.angular_velocity.z());
    acc_avr << 0.5 *
               (head.linear_acceleration.x() + tail.linear_acceleration.x()),
        0.5 * (head.linear_acceleration.y() + tail.linear_acceleration.y()),
        0.5 * (head.linear_acceleration.z() + tail.linear_acceleration.z());

    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr - state_inout.bias_a;

    // dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
//    if (head.tm < last_lidar_end_time_) {
//      dt = tail.tm - last_lidar_end_time_;
//    } else {
      dt = (tail.tm_ns - head.tm_ns) * double(1e-9);
//    }

    /* covariance propagation */

    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRIX(acc_avr);

    /* propagation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr - gravity;

    /* propagation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
//    angvel_last = angvel_avr;
//    acc_s_last = acc_imu;
    double offs_t = (tail.tm_ns - pcl_beg_time_ns) * double(1e-9);
    // std::cout<<"acc "<<acc_imu.transpose()<<"vel "<<acc_imu.transpose()<<"vel "<<pos_imu.transpose()<<std::endl;
    IMU_pose.push_back(
        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
//  dt = pcl_end_time - imu_end_time;
////  BASALT_ASSERT(dt >= 0);
//  std::cout << "dt: " << dt << std::endl;
//  if (dt >= 0.0) {
//    angvel_avr = v_imu.back().angular_velocity;
//    acc_avr = v_imu.back().linear_acceleration;
//
//    angvel_avr -= state_inout.bias_g;
//    acc_avr = acc_avr - state_inout.bias_a;
//
//    Eigen::Matrix3d acc_avr_skew;
//    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);
//    acc_avr_skew << SKEW_SYM_MATRIX(acc_avr);
//
//    R_imu = R_imu * Exp_f;
//    acc_imu = R_imu * acc_avr - gravity;
//    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
//    vel_imu = vel_imu + acc_imu * dt;
//
//    double &&offs_t = pcl_end_time - pcl_beg_time;
//    IMU_pose.push_back(
//        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
//  }

  state_inout.vel_end = vel_imu;
  state_inout.rot_end = R_imu;
  state_inout.pos_end = pos_imu;

//  last_lidar_end_time_ = pcl_end_time;

  Eigen::Vector3d pos_liD_e =
      state_inout.pos_end + state_inout.rot_end * Lidar_offset_to_IMU;
  Eigen::Matrix3d R_liD_e = state_inout.rot_end * Lidar_R_to_IMU;

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMU_pose.end() - 1; it_kp != IMU_pose.begin(); it_kp--) {
    auto head = it_kp - 1;
    R_imu << MAT_FROM_ARRAY(head->rot);
    acc_imu << VEC_FROM_ARRAY(head->acc);
    // std::cout<<"head imu acc: "<<acc_imu.transpose()<<std::endl;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    angvel_avr << VEC_FROM_ARRAY(head->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      Eigen::Matrix3d R_i(R_imu * Exp(angvel_avr, dt));
      Eigen::Vector3d p_i(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt);
      Eigen::Matrix3d R_Li = R_i * Lidar_R_to_IMU;
      Eigen::Vector3d p_Li = R_i * Lidar_offset_to_IMU + p_i;

      Eigen::Matrix3d R_itoe_inG = R_liD_e.transpose() * R_Li;
      Eigen::Vector3d p_itoe_inG = R_liD_e.transpose() * (p_Li - pos_liD_e);

      Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      Eigen::Vector3d P_compensate = R_itoe_inG * P_i + p_itoe_inG;
      //            Eigen::Matrix3d R_etoi_inLocal = R_L0.transpose() * R_itoe_inG * R_L0; Eigen::Vector3d p_etoi_inLocal = R_L0.transpose() * (R_itoe_inG * pos_L0 + p_itoe_inG - pos_L0); Eigen::Vector3d P_compensate1 = R_etoi_inLocal * P_i + p_etoi_inLocal;

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
}
#endif
