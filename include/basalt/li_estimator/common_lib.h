#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include "so3_math.h"
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

#define DIM_OF_STATES (15)  // R p v bg ba
#define DIM_OF_PROC_N (12)  // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (0.0001)

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

template <typename T = double>
inline Eigen::Matrix<T, 3, 3> vec_to_hat(Eigen::Matrix<T, 3, 1> &omega) {
  Eigen::Matrix<T, 3, 3> res_mat_33;
  res_mat_33.setZero();
  res_mat_33(0, 1) = -omega(2);
  res_mat_33(1, 0) = omega(2);
  res_mat_33(0, 2) = omega(1);
  res_mat_33(2, 0) = -omega(1);
  res_mat_33(1, 2) = -omega(0);
  res_mat_33(2, 1) = omega(0);
  return res_mat_33;
}

inline Eigen::Matrix<double, 3, 3> Jl_so3(
    const Eigen::Matrix<double, 3, 1> &w) {
  double theta = w.norm();
  if (theta < 1e-6) {
    return Eigen::MatrixXd::Identity(3, 3);
  } else {
    Eigen::Matrix<double, 3, 1> a = w / theta;
    Eigen::Matrix<double, 3, 3> J =
        sin(theta) / theta * Eigen::MatrixXd::Identity(3, 3) +
        (1 - sin(theta) / theta) * a * a.transpose() +
        ((1 - cos(theta)) / theta) * vec_to_hat(a);
    return J;
  }
}

inline Eigen::Matrix<double, 3, 3> Jr_so3(
    const Eigen::Matrix<double, 3, 1> &w) {
  return Jl_so3(-w);
}

struct LidarData {
  using Ptr = std::shared_ptr<LidarData>;

  int64_t t_ns;  ///< timestamp in nanoseconds
  int64_t t_ns_end;
  PointCloudXYZINormal lidar;
};

struct ImuTypeData {
  ImuTypeData() {}
  ImuTypeData(double time, Eigen::Vector3d accel, Eigen::Vector3d gyro) {
    tm = time;
    linear_acceleration = accel;
    angular_velocity = gyro;
  }
  double tm;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};

struct MeasureGroup  // Lidar data and imu dates for the curent process
{
  MeasureGroup() {
//    this->lidar.reset(new PointCloudXYZINormal());
  };
  double lidar_beg_time;
  double lidar_end_time;
  PointCloudXYZINormal lidar;
  std::deque<ImuTypeData> imu;
};

struct StatesGroup {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<StatesGroup>;

  Eigen::Matrix3d rot_end;  // [0-2] the estimated attitude (rotation matrix) at the end lidar point
  Eigen::Vector3d pos_end;  // [3-5] the estimated position at the end lidar point (world frame)
  Eigen::Vector3d vel_end;  // [6-8] the estimated velocity at the end lidar point (world frame)
  Eigen::Vector3d bias_g;   // [9-11] gyroscope bias
  Eigen::Vector3d bias_a;   // [12-14] accelerator bias
  Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> cov;  // states covariance
  double last_update_time = 0;
  StatesGroup() {
    rot_end = Eigen::Matrix3d::Identity();
    pos_end = Eigen::Vector3d::Zero();
    vel_end = Eigen::Vector3d::Zero();
    bias_g = Eigen::Vector3d::Zero();
    bias_a = Eigen::Vector3d::Zero();

    cov = Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity() * INIT_COV;
    last_update_time = 0;
  }

  ~StatesGroup() {}

  StatesGroup operator+(
      const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add) {
    StatesGroup a = *this;
    //        a.rot_end = this->rot_end * Exp(state_add(0), state_add(1), state_add(2));
    a.rot_end = Exp(state_add(0), state_add(1), state_add(2)) * this->rot_end;
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);

    a.cov = this->cov;
    a.last_update_time = this->last_update_time;
    return a;
  }

  StatesGroup &operator+=(
      const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add) {
    //        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->rot_end =
        Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0)) * this->rot_end;
    this->pos_end += state_add.block<3, 1>(3, 0);
    this->vel_end += state_add.block<3, 1>(6, 0);
    this->bias_g += state_add.block<3, 1>(9, 0);
    this->bias_a += state_add.block<3, 1>(12, 0);
    return *this;
  }

  Eigen::Matrix<double, DIM_OF_STATES, 1> operator-(const StatesGroup &b) {
    Eigen::Matrix<double, DIM_OF_STATES, 1> a;
    //        Eigen::Matrix3d rotd(b.rot_end.transpose() * this->rot_end);
    Eigen::Matrix3d rotd(this->rot_end * b.rot_end.transpose());
    a.block<3, 1>(0, 0) = SO3_LOG(rotd);
    a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
    a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
    a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
    return a;
  }

  static void display(const StatesGroup &state,
                      std::string str = std::string("State: ")) {
    Eigen::Vector3d angle_axis = SO3_LOG(state.rot_end) * 57.3;
    printf("%s |", str.c_str());
    printf("[%.5f] | ", state.last_update_time);
    printf("(%.3f, %.3f, %.3f) | ", angle_axis(0), angle_axis(1),
           angle_axis(2));
    printf("(%.3f, %.3f, %.3f) | ", state.pos_end(0), state.pos_end(1),
           state.pos_end(2));
    printf("(%.3f, %.3f, %.3f) | ", state.vel_end(0), state.vel_end(1),
           state.vel_end(2));
    printf("(%.3f, %.3f, %.3f) | ", state.bias_g(0), state.bias_g(1),
           state.bias_g(2));
    printf("(%.3f, %.3f, %.3f) \r\n", state.bias_a(0), state.bias_a(1),
           state.bias_a(2));
  }
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
}
#endif
