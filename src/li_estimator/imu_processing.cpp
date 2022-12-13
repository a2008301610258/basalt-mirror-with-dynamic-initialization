#include <basalt/li_estimator/imu_processing.hpp>
#include <basalt/utils/assert.h>
#include <basalt/utils/time_utils.hpp>
namespace basalt {
#define COV_OMEGA_NOISE_DIAG 1e-1
#define COV_ACC_NOISE_DIAG 0.4
#define COV_GYRO_NOISE_DIAG 0.2

#define COV_BIAS_ACC_NOISE_DIAG 0.05
#define COV_BIAS_GYRO_NOISE_DIAG 0.1

#define COV_START_ACC_DIAG 1e-1
#define COV_START_GYRO_DIAG 1e-1
// #define COV_NOISE_EXT_I2C_R (0.0 * 1e-3)
// #define COV_NOISE_EXT_I2C_T (0.0 * 1e-3)
// #define COV_NOISE_EXT_I2C_Td (0.0 * 1e-3)

double g_lidar_star_tim = 0;
ImuProcess::ImuProcess()
    : b_first_frame_(true),
      imu_need_init_(true),
//      last_imu_(nullptr),
      start_timestamp_(-1) {
  Eigen::Quaterniond q(0, 1, 0, 0);
  Eigen::Vector3d t(0, 0, 0);
  init_iter_num = 1;
  cov_acc = Eigen::Vector3d(COV_START_ACC_DIAG, COV_START_ACC_DIAG,
                            COV_START_ACC_DIAG);
  cov_gyr = Eigen::Vector3d(COV_START_GYRO_DIAG, COV_START_GYRO_DIAG,
                            COV_START_GYRO_DIAG);
  mean_acc = Eigen::Vector3d(0, 0, -9.805);
  mean_gyr = Eigen::Vector3d(0, 0, 0);
  angvel_last = Zero3d;
  cov_proc_noise = Eigen::Matrix<double, DIM_OF_PROC_N, 1>::Zero();
  // Lidar_offset_to_IMU = Eigen::Vector3d(0.0, 0.0, -0.0);
  // fout.open(DEBUG_FILE_DIR("imu.txt"),std::ios::out);

  gravity = Eigen::Vector3d(0, 0, 9.805);
}

ImuProcess::~ImuProcess() { /**fout.close();*/
}

void ImuProcess::Reset() {
  ROS_WARN("Reset ImuProcess");
  angvel_last = Zero3d;
  cov_proc_noise = Eigen::Matrix<double, DIM_OF_PROC_N, 1>::Zero();

  cov_acc = Eigen::Vector3d(COV_START_ACC_DIAG, COV_START_ACC_DIAG,
                            COV_START_ACC_DIAG);
  cov_gyr = Eigen::Vector3d(COV_START_GYRO_DIAG, COV_START_GYRO_DIAG,
                            COV_START_GYRO_DIAG);
  mean_acc = Eigen::Vector3d(0, 0, -9.805);
  mean_gyr = Eigen::Vector3d(0, 0, 0);

  imu_need_init_ = true;
  b_first_frame_ = true;
  init_iter_num = 1;

//  last_imu_ = nullptr;

  // gyr_int_.Reset(-1, nullptr);
  start_timestamp_ = -1;
  v_imu_.clear();
  IMU_pose.clear();

  cur_pcl_un_.reset(new PointCloudXYZINormal());
}

void ImuProcess::IMU_Initial(const MeasureGroup &meas, StatesGroup &state_inout,
                             int &N) {
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  Eigen::Vector3d cur_acc, cur_gyr;

  if (b_first_frame_) {
    Reset();
    N = 1;
    b_first_frame_ = false;
  }

  Eigen::Vector3d a_avg = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_avg = Eigen::Vector3d::Zero();

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu.linear_acceleration;
    const auto &gyr_acc = imu.angular_velocity;
    cur_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
    cur_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();

    a_avg += cur_acc;
    w_avg += cur_gyr;
    N++;
  }
  // static initializer
  a_avg = a_avg / meas.imu.size();
  w_avg = w_avg / meas.imu.size();
  // Get z axis, which aligns with -g (z_in_G=0,0,1)
  Eigen::Vector3d z_axis = a_avg / a_avg.norm();

  // Create an x_axis
  Eigen::Vector3d e_1(1, 0, 0);

  // Make x_axis perpendicular to z
  Eigen::Vector3d x_axis = e_1 - z_axis * z_axis.transpose() *
                                     e_1;  /// x_axis is perpendicular to z_axis
  x_axis = x_axis / x_axis.norm();

  // Get z from the cross product of these two
  Eigen::Vector3d y_axis = Skew(z_axis) * x_axis;

  // From these axes get rotation
  Eigen::Matrix3d Ro;
  Ro.block(0, 0, 3, 1) = x_axis;
  Ro.block(0, 1, 3, 1) = y_axis;
  Ro.block(0, 2, 3, 1) = z_axis;

  // Create our state variables
  Eigen::Matrix3d R_GtoI = Ro;  /// global to imu frame (correct)
  Eigen::Matrix3d R_ItoG = R_GtoI.transpose();

  // Set our biases equal to our noise (subtract our gravity from accelerometer bias)
  Eigen::Vector3d bg = w_avg;
  Eigen::Vector3d ba = a_avg - R_ItoG.transpose() * gravity;

  state_inout.rot_end = R_ItoG;  // R_ItoG.transpose();  /// Local reference, the first Rotation must be I
  state_inout.bias_a = ba;
  state_inout.bias_g = bg;

  R_i0toG = state_inout.rot_end;
  p_i0inG = state_inout.pos_end;

  state_inout.cov = 1e-3 * Eigen::MatrixXd::Identity(state_inout.cov.rows(), state_inout.cov.cols());
  state_inout.cov( 2, 2) = 0.0;
  state_inout.cov.block( 3, 3, 3, 3 ).setZero();
  state_inout.cov.block( 6, 6, 3, 3 ) *= 2;
  state_inout.cov.block(0, 0, 3, 3) = R_GtoI * state_inout.cov.block(0, 0, 3, 3) * R_GtoI.transpose();
}

void ImuProcess::lic_state_propagate(const MeasureGroup &meas,
                                     StatesGroup &state_inout) {
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  // const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back().tm;
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  /*** sort point clouds by offset time ***/
  PointCloudXYZINormal pcl_out = meas.lidar;
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
//  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  double end_pose_dt = pcl_end_time - imu_end_time;
  BASALT_ASSERT(end_pose_dt >= 0.0);
  if (end_pose_dt > 0.0) {
    ImuTypeData imu_data;
    imu_data.tm = pcl_end_time;
    imu_data.linear_acceleration = v_imu.back().linear_acceleration;
    imu_data.angular_velocity = v_imu.back().angular_velocity;

    v_imu.push_back(imu_data);
  }
  state_inout = imu_preintegration(state_inout, v_imu);
  last_imu_ = meas.imu.back();
}

// Avoid abnormal state input
bool check_state(StatesGroup &state_inout) {
  bool is_fail = false;
  for (int idx = 0; idx < 3; idx++) {
    if (fabs(state_inout.vel_end(idx)) > 10) {
      is_fail = true;
      //            scope_color( ANSI_COLOR_RED_BG );
      for (int i = 0; i < 10; i++) {
        std::cout << __FILE__ << ", " << __LINE__ << ", check_state fail !!!! "
                  << state_inout.vel_end.transpose() << std::endl;
      }
      state_inout.vel_end(idx) = 0.0;
    }
  }
  return is_fail;
}

// Avoid abnormal state input
void check_in_out_state(const StatesGroup &state_in, StatesGroup &state_inout) {
  if ((state_in.pos_end - state_inout.pos_end).norm() > 1.0) {
    //        scope_color( ANSI_COLOR_RED_BG );
    for (int i = 0; i < 10; i++) {
      std::cout << __FILE__ << ", " << __LINE__
                << ", check_in_out_state fail !!!! "
                << state_in.pos_end.transpose() << " | "
                << state_inout.pos_end.transpose() << std::endl;
    }
    state_inout.pos_end = state_in.pos_end;
  }
}

//std::mutex g_imu_premutex;

StatesGroup ImuProcess::imu_preintegration(
    const StatesGroup &state_in, std::deque<ImuTypeData> &v_imu) {
//  std::unique_lock<std::mutex> lock(g_imu_premutex);
  StatesGroup state_inout = state_in;
//  if (check_state(state_inout)) {
//    state_inout.display(state_inout, "state_inout");
//    state_in.display(state_in, "state_in");
//  }
  Eigen::Vector3d acc_imu(0, 0, 0), angvel_avr(0, 0, 0), acc_avr(0, 0, 0),
      vel_imu(0, 0, 0), pos_imu(0, 0, 0);
  vel_imu = state_inout.vel_end;
  pos_imu = state_inout.pos_end;
  Eigen::Matrix3d R_imu(state_inout.rot_end);

  double          dt = 0;

  int if_first_imu = 1;
  // printf("IMU start_time = %.5f, end_time = %.5f, state_update_time = %.5f, start_delta = %.5f\r\n", v_imu.front()->header.stamp.toSec() -
  // g_lidar_star_tim,
  //        v_imu.back()->header.stamp.toSec() - g_lidar_star_tim,
  //        state_in.last_update_time - g_lidar_star_tim,
  //        state_in.last_update_time - v_imu.front()->header.stamp.toSec());

  for (std::deque<ImuTypeData>::iterator it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++) {
    // if(g_lidar_star_tim == 0 || state_inout.last_update_time == 0)
    // {
    //   return state_inout;
    // }
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

    if (tail.tm < state_inout.last_update_time) {
      continue;
    }

    if (if_first_imu) {
      if_first_imu = 0;
      dt = tail.tm - state_inout.last_update_time;
    } else {
      dt = tail.tm - head.tm;
    }
    if (dt > 0.05) {
      dt = 0.05;
    }

    /// Global reference
    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);

    Eigen::Matrix<double, 3, 3> Jl = Jl_so3(angvel_avr * dt);

    Eigen::MatrixXd F_x(Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
    F_x.block<3, 3>(0, 0) = Eye3d;
    F_x.block<3, 3>(0, 9) = -R_imu * Jl * dt;
    F_x.block<3, 3>(3, 0) = -0.5 * Skew<double>(R_imu * acc_avr) * dt * dt;
    F_x.block<3, 3>(3, 3) = Eye3d;
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    F_x.block<3, 3>(3, 12) = -0.5 * R_imu * dt * dt;
    F_x.block<3, 3>(6, 0) = -Skew<double>(R_imu * acc_avr) * dt;
    F_x.block<3, 3>(6, 6) = Eye3d;
    F_x.block<3, 3>(6, 12) = -R_imu * dt;
    F_x.block<3, 3>(9, 9) = Eye3d;
    F_x.block<3, 3>(12, 12) = Eye3d;

    Eigen::MatrixXd F_w(Eigen::Matrix<double, DIM_OF_STATES, 12>::Zero());
    F_w.block<3, 3>(0, 0) = -R_imu * Jl * dt;
    F_w.block<3, 3>(3, 3) = -0.5 * R_imu * dt * dt;
    F_w.block<3, 3>(6, 3) = -R_imu * dt;
    F_w.block<3, 3>(9, 6) = Eye3d;   // Eye3d * dt;
    F_w.block<3, 3>(12, 9) = Eye3d;  // Eye3d * dt;

    /// covariance matrix of n_g, n_a, n_bg, n_ba
    const double accel_noise_std = 2.0000e-3; /// 0.00818;
    const double gyro_noise_std = 1.6968e-04; /// 0.00226;
    const double accel_bias_std = 3.0000e-03; /// 0.01;
    const double gyro_bias_std = 1.9393e-05; ///0.0007;

    Eigen::Matrix<double, 12, 12> Qw = Eigen::Matrix<double, 12, 12>::Zero();
    Qw.block(0, 0, 3, 3) = pow(gyro_noise_std, 2) / dt * Eye3d; /// Gyroscope white noise (rad/s/sqrt(hz))
    Qw.block(3, 3, 3, 3) = pow(accel_noise_std, 2) / dt * Eye3d;  /// Accelerometer white noise (m/s^2/sqrt(hz))
    Qw.block(6, 6, 3, 3) = pow(gyro_bias_std, 2) * dt * Eye3d; /// Gyroscope random walk (rad/s^2/sqrt(hz))
    Qw.block(9, 9, 3, 3) = pow(accel_bias_std, 2) * dt * Eye3d; /// Accelerometer random walk (m/s^3/sqrt(hz))

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + F_w * Qw * F_w.transpose();

    //////////////////////////////////////////////////////////

    R_imu = R_imu * Exp_f;
    acc_imu = R_imu * acc_avr - gravity;
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    vel_imu = vel_imu + acc_imu * dt;
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/


  state_inout.last_update_time = v_imu.back().tm;
  // cout << "Last update time = " <<  state_inout.last_update_time - g_lidar_star_tim << endl;

  state_inout.vel_end = vel_imu;
  state_inout.rot_end = R_imu;
  state_inout.pos_end = pos_imu;

  return state_inout;
}

void ImuProcess::lic_point_cloud_undistort(const MeasureGroup &meas,
                                           const StatesGroup &_state_inout,
                                           PointCloudXYZINormal &pcl_out) {
  StatesGroup state_inout = _state_inout;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front().tm;
  const double &imu_end_time = v_imu.back().tm;
  const double &pcl_beg_time = meas.lidar_beg_time;
  /*** sort point clouds by offset time ***/
  pcl_out = meas.lidar;
  std::sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time =
      pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  std::cout << std::fixed << std::setprecision(6) << "[ IMU Process ]: Process lidar from " << pcl_beg_time << " to "
            << pcl_end_time << ", " << meas.lidar_end_time << ", "
            << meas.imu.size() << " imu msgs from " << imu_beg_time << " to "
            << imu_end_time << ", last tim: " << state_inout.last_update_time
            << ", dt: " << pcl_end_time - imu_end_time
            << ", offset: " << last_lidar_end_time_ - imu_beg_time << std::endl;

  /*** Initialize IMU pose ***/
  IMU_pose.clear();
  double first_offs = last_lidar_end_time_ - pcl_beg_time;
  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
  IMU_pose.push_back(set_pose6d(first_offs, acc_s_last, angvel_last,
                                state_inout.vel_end, state_inout.pos_end,
                                state_inout.rot_end));

  /*** forward propagation at each imu point ***/
  Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end),
      pos_imu(state_inout.pos_end);
  Eigen::Matrix3d R_imu(state_inout.rot_end);
  Eigen::MatrixXd F_x(
      Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity());
  Eigen::MatrixXd cov_w(
      Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Zero());
  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu != (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail.tm < last_lidar_end_time_) continue;

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
    if (head.tm < last_lidar_end_time_) {
      dt = tail.tm - last_lidar_end_time_;
    } else {
      dt = tail.tm - head.tm;
    }

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
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
    double &&offs_t = tail.tm - pcl_beg_time;
    // std::cout<<"acc "<<acc_imu.transpose()<<"vel "<<acc_imu.transpose()<<"vel "<<pos_imu.transpose()<<std::endl;
    IMU_pose.push_back(
        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  dt = pcl_end_time - imu_end_time;
  BASALT_ASSERT(dt >= 0);
  std::cout << "dt: " << dt << std::endl;
  if (dt >= 0.0) {
    angvel_avr = v_imu.back().angular_velocity;
    acc_avr = v_imu.back().linear_acceleration;

    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr - state_inout.bias_a;

    Eigen::Matrix3d acc_avr_skew;
    Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRIX(acc_avr);

    R_imu = R_imu * Exp_f;
    acc_imu = R_imu * acc_avr - gravity;
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
    vel_imu = vel_imu + acc_imu * dt;

    double &&offs_t = pcl_end_time - pcl_beg_time;
    IMU_pose.push_back(
        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  state_inout.vel_end = vel_imu;
  state_inout.rot_end = R_imu;
  state_inout.pos_end = pos_imu;

  last_lidar_end_time_ = pcl_end_time;

  Eigen::Vector3d pos_liD_e =
      state_inout.pos_end + state_inout.rot_end * Lidar_offset_to_IMU;
  Eigen::Matrix3d R_liD_e = state_inout.rot_end * Lidar_R_to_IMU;

  //    Eigen::Matrix3d R_L0 = _state_inout.rot_end * Lidar_R_to_IMU;
  //    Eigen::Vector3d pos_L0 = _state_inout.pos_end + _state_inout.rot_end * Lidar_offset_to_IMU; Eigen::Matrix3d R_L0 = R_i0toG * Lidar_R_to_IMU;
  //    Eigen::Vector3d pos_L0 = p_i0inG + R_i0toG * Lidar_offset_to_IMU;

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

bool ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZINormal::Ptr cur_pcl_un_) {

  if (meas.imu.empty()) {
    std::cout << "no imu data: " << meas.lidar_beg_time << ", " << meas.lidar_end_time << std::endl;
    return false;
  };

  if (imu_need_init_) {
    /// The very first lidar frame
    IMU_Initial(meas, stat, init_iter_num);

    imu_need_init_ = true;

    last_imu_ = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT) {
      imu_need_init_ = false;
      // std::cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<std::endl;
//      ROS_INFO(
//          "IMU Initials: Gravity: %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",
//          gravity[0], gravity[1], gravity[2], stat.bias_g[0], stat.bias_g[1],
//          stat.bias_g[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0],
//          cov_gyr[1], cov_gyr[2]);
      last_lidar_end_time_ = last_imu_.tm;

      return true;
    }

    return false;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)

  Timer t_undistort;
  lic_point_cloud_undistort(meas, stat, *cur_pcl_un_);
  std::cout << "undistort time: " << t_undistort.elapsed() * 1000.0 << " ms" << std::endl;

  Timer t_propagate;
  lic_state_propagate(meas, stat);
  std::cout << "propagate time: " << t_propagate.elapsed() * 1000.0 << " ms" << std::endl;

  last_imu_ = meas.imu.back();

  // std::cout<<"[ IMU Process ]: Time: "<<t3 - t1<<std::endl;
  return true;
}
}