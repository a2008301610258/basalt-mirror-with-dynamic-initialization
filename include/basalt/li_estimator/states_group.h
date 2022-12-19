//
// Created by zhihui on 12/13/22.
//

#ifndef BASALT_STATES_GROUP_H
#define BASALT_STATES_GROUP_H

#include <Eigen/Eigen>
#include <memory>
#include <deque>
#include <iomanip>
#include <basalt/utils/assert.h>

#include "so3_math.h"

namespace basalt {

#define DIM_OF_STATES (15)  // R p v bg ba
#define DIM_OF_PROC_N (12)  // Dimension of process noise (Let Dim(SO(3)) = 3)
#define INIT_COV (0.0001)

static const Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, 9.81);

struct ImuTypeData {
  ImuTypeData() {}
  ImuTypeData(int64_t time_ns, Eigen::Vector3d accel, Eigen::Vector3d gyro) {
    tm_ns = time_ns;
    linear_acceleration = accel;
    angular_velocity = gyro;
  }
  int64_t tm_ns;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
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
//  double last_update_time = 0;
  int64_t last_update_time_ns = 0;

  StatesGroup() {
    rot_end = Eigen::Matrix3d::Identity();
    pos_end = Eigen::Vector3d::Zero();
    vel_end = Eigen::Vector3d::Zero();
    bias_g = Eigen::Vector3d::Zero();
    bias_a = Eigen::Vector3d::Zero();

    cov = Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity() *
          INIT_COV;
//    last_update_time = 0;
    last_update_time_ns = 0;
  }

  ~StatesGroup() {}

  void init(Eigen::Matrix3d rot, Eigen::Vector3d pos, Eigen::Vector3d vel,
            Eigen::Vector3d bg, Eigen::Vector3d ba) {
    rot_end = rot;
    pos_end = pos;
    vel_end = vel;
    bias_g = bg;
    bias_a = ba;

    Eigen::Matrix3d R_GtoI = rot.transpose();

    cov = 1e-3 * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
    cov(2, 2) = 0.0;
    cov.block(3, 3, 3, 3).setZero();
    cov.block(6, 6, 3, 3) *= 2;
    cov.block(0, 0, 3, 3) = R_GtoI * cov.block(0, 0, 3, 3) * R_GtoI.transpose();
  }

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
//    a.last_update_time = this->last_update_time;
    a.last_update_time_ns = this->last_update_time_ns;
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
    printf("[%.5f] | ", state.last_update_time_ns);
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

class StateHelper {
 public:

  /**
   * eskf state propagation
   * */
  static StatesGroup eskf_state_propagate(const StatesGroup &state_in, std::deque<ImuTypeData> &v_imu, const int type) {
    static int64_t last_imu_time_ns = 0.;
    if (last_imu_time_ns > 0.) {
      std::cout << std::setprecision(19) << type << "-v_imu.front().tm_ns: " << v_imu.front().tm_ns << std::endl;
      std::cout << std::setprecision(19) << type << "-eskf_state_propagate(last_imu_time_ns): " << last_imu_time_ns << std::endl;
      double dt = (v_imu.front().tm_ns - last_imu_time_ns) * double(1e-9);
      std::cout << "imu_dt: " << dt << std::endl;
      BASALT_ASSERT(v_imu.front().tm_ns <= last_imu_time_ns);
      BASALT_ASSERT(last_imu_time_ns == state_in.last_update_time_ns);
    } else {
      std::cout << type << "-eskf_state_propagate is first" << std::endl;
    }

    last_imu_time_ns = v_imu.back().tm_ns;

    StatesGroup state_inout = state_in;
    Eigen::Vector3d acc_imu(0, 0, 0), angvel_avr(0, 0, 0), acc_avr(0, 0, 0),
        vel_imu(0, 0, 0), pos_imu(0, 0, 0);
    vel_imu = state_inout.vel_end;
    pos_imu = state_inout.pos_end;
    Eigen::Matrix3d R_imu(state_inout.rot_end);

    double dt = 0;

    int if_first_imu = 1;

    for (std::deque<ImuTypeData>::iterator it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++) {
      ImuTypeData head = *(it_imu);
      ImuTypeData tail = *(it_imu + 1);

      angvel_avr << 0.5 * (head.angular_velocity.x() + tail.angular_velocity.x()),
          0.5 * (head.angular_velocity.y() + tail.angular_velocity.y()),
          0.5 * (head.angular_velocity.z() + tail.angular_velocity.z());
      acc_avr << 0.5 * (head.linear_acceleration.x() + tail.linear_acceleration.x()),
          0.5 * (head.linear_acceleration.y() + tail.linear_acceleration.y()),
          0.5 * (head.linear_acceleration.z() + tail.linear_acceleration.z());

      angvel_avr -= state_inout.bias_g;

      acc_avr = acc_avr - state_inout.bias_a;

      if (tail.tm_ns < state_inout.last_update_time_ns) {
        continue;
      }

      if (if_first_imu) {
        if_first_imu = 0;
        dt = (tail.tm_ns - state_inout.last_update_time_ns) * double(1e-9);
      } else {
        dt = (tail.tm_ns - head.tm_ns) * double(1e-9);
      }
      if (dt > 0.05) {
        dt = 0.05;
      }

      /// Global reference
      Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);

      Eigen::Matrix<double, 3, 3> Jl = Jl_so3(angvel_avr * dt);

      Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
      F_x.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      F_x.block<3, 3>(0, 9) = -R_imu * Jl * dt;
      F_x.block<3, 3>(3, 0) = -0.5 * Skew<double>(R_imu * acc_avr) * dt * dt;
      F_x.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      F_x.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
      F_x.block<3, 3>(3, 12) = -0.5 * R_imu * dt * dt;
      F_x.block<3, 3>(6, 0) = -Skew<double>(R_imu * acc_avr) * dt;
      F_x.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
      F_x.block<3, 3>(6, 12) = -R_imu * dt;
      F_x.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
      F_x.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

      Eigen::MatrixXd F_w(Eigen::Matrix<double, DIM_OF_STATES, 12>::Zero());
      F_w.block<3, 3>(0, 0) = -R_imu * Jl * dt;
      F_w.block<3, 3>(3, 3) = -0.5 * R_imu * dt * dt;
      F_w.block<3, 3>(6, 3) = -R_imu * dt;
      F_w.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();   // Eye3d * dt;
      F_w.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();  // Eye3d * dt;

      /// covariance matrix of n_g, n_a, n_bg, n_ba
      const double accel_noise_std = 2.0000e-3; /// 0.00818;
      const double gyro_noise_std = 1.6968e-04; /// 0.00226;
      const double accel_bias_std = 3.0000e-03; /// 0.01;
      const double gyro_bias_std = 1.9393e-05; ///0.0007;

      Eigen::Matrix<double, 12, 12> Qw = Eigen::Matrix<double, 12, 12>::Zero();
      Qw.block(0, 0, 3, 3) = pow(gyro_noise_std, 2) / dt * Eigen::Matrix3d::Identity(); /// Gyroscope white noise (rad/s/sqrt(hz))
      Qw.block(3, 3, 3, 3) = pow(accel_noise_std, 2) / dt * Eigen::Matrix3d::Identity();  /// Accelerometer white noise (m/s^2/sqrt(hz))
      Qw.block(6, 6, 3, 3) = pow(gyro_bias_std, 2) * dt * Eigen::Matrix3d::Identity(); /// Gyroscope random walk (rad/s^2/sqrt(hz))
      Qw.block(9, 9, 3, 3) = pow(accel_bias_std, 2) * dt * Eigen::Matrix3d::Identity(); /// Accelerometer random walk (m/s^3/sqrt(hz))

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + F_w * Qw * F_w.transpose();

      //////////////////////////////////////////////////////////

      R_imu = R_imu * Exp_f;
      acc_imu = R_imu * acc_avr - gravity;
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
      vel_imu = vel_imu + acc_imu * dt;
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/

    state_inout.last_update_time_ns = v_imu.back().tm_ns;
    state_inout.vel_end = vel_imu;
    state_inout.rot_end = R_imu;
    state_inout.pos_end = pos_imu;

    std::cout << std::setprecision(19) << type << "-done_eskf_state_propagate(last_imu_time_ns): " << last_imu_time_ns << std::endl;

    return state_inout;
  }
};

}

#endif  // BASALT_STATES_GROUP_H
