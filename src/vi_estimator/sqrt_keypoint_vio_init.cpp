/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <basalt/vi_estimator/marg_helper.h>
#include <basalt/vi_estimator/sqrt_keypoint_vio_init.h>

#include <basalt/optimization/accumulator.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/system_utils.h>
#include <basalt/vi_estimator/sc_ba_base.h>
#include <basalt/utils/cast_utils.hpp>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

#include <basalt/linearization/linearization_base.hpp>

#include <fmt/format.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <chrono>

namespace basalt {

template <class Scalar_>
SqrtKeypointVioInitEstimator<Scalar_>::SqrtKeypointVioInitEstimator(
    const Eigen::Vector3d& g_, const basalt::Calibration<double>& calib_,
    const VioConfig& config_)
    : take_kf(true),
      frames_after_kf(0),
      g(g_.cast<Scalar>()),
      initialized(false),
      config(config_),
      lambda(config_.vio_lm_lambda_initial),
      min_lambda(config_.vio_lm_lambda_min),
      max_lambda(config_.vio_lm_lambda_max),
      lambda_vee(2),
      is_system_initialized(false) {
  obs_std_dev = Scalar(config.vio_obs_std_dev);
  huber_thresh = Scalar(config.vio_obs_huber_thresh);
  calib = calib_.cast<Scalar>();

  // Setup marginalization
  marg_data.is_sqrt = config.vio_sqrt_marg;
  marg_data.H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  marg_data.b.setZero(POSE_VEL_BIAS_SIZE);

  // Setup marginalization (vo)
  marg_vo_data.is_sqrt = config.vio_sqrt_marg;
  marg_vo_data.H.setZero(POSE_SIZE, POSE_SIZE);
  marg_vo_data.b.setZero(POSE_SIZE);

  // Version without prior
  nullspace_marg_data.is_sqrt = marg_data.is_sqrt;
  nullspace_marg_data.H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  nullspace_marg_data.b.setZero(POSE_VEL_BIAS_SIZE);

  // prior on pose (vo)
  if (marg_vo_data.is_sqrt) {
    marg_vo_data.H.diagonal().setConstant(
        std::sqrt(Scalar(config.vio_init_pose_weight)));
  } else {
    marg_vo_data.H.diagonal().setConstant(Scalar(config.vio_init_pose_weight));
  }

  if (marg_data.is_sqrt) {
    // prior on position
    marg_data.H.diagonal().template head<3>().setConstant(
        std::sqrt(Scalar(config.vio_init_pose_weight)));
    // prior on yaw
    marg_data.H(5, 5) = std::sqrt(Scalar(config.vio_init_pose_weight));

    // small prior to avoid jumps in bias
    marg_data.H.diagonal().template segment<3>(9).array() =
        std::sqrt(Scalar(config.vio_init_ba_weight));
    marg_data.H.diagonal().template segment<3>(12).array() =
        std::sqrt(Scalar(config.vio_init_bg_weight));
  } else {
    // prior on position
    marg_data.H.diagonal().template head<3>().setConstant(
        Scalar(config.vio_init_pose_weight));
    // prior on yaw
    marg_data.H(5, 5) = Scalar(config.vio_init_pose_weight);

    // small prior to avoid jumps in bias
    marg_data.H.diagonal().template segment<3>(9).array() =
        Scalar(config.vio_init_ba_weight);
    marg_data.H.diagonal().template segment<3>(12).array() =
        Scalar(config.vio_init_bg_weight);
  }

  std::cout << "marg_H (sqrt:" << marg_data.is_sqrt << ")\n"
            << marg_data.H << std::endl;

  gyro_bias_sqrt_weight = calib.gyro_bias_std.array().inverse();
  accel_bias_sqrt_weight = calib.accel_bias_std.array().inverse();

  max_states = config.vio_max_states;
  max_kfs = config.vio_max_kfs;

  opt_started = false;

  vision_data_queue.set_capacity(10);
  imu_data_queue.set_capacity(300);
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::initialize(
    int64_t t_ns, const Sophus::SE3d& T_w_i, const Eigen::Vector3d& vel_w_i,
    const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
  initialized = true;
  T_w_i_init = T_w_i.cast<Scalar>();

  last_state_t_ns = t_ns;
  imu_meas[t_ns] = IntegratedImuMeasurement<Scalar>(t_ns, bg.cast<Scalar>(),
                                                    ba.cast<Scalar>());
  frame_states[t_ns] = PoseVelBiasStateWithLin<Scalar>(
      t_ns, T_w_i_init, vel_w_i.cast<Scalar>(), bg.cast<Scalar>(),
      ba.cast<Scalar>(), true);

  marg_vo_data.order.abs_order_map[t_ns] = std::make_pair(0, POSE_SIZE);
  marg_vo_data.order.total_size = POSE_SIZE;
  marg_vo_data.order.items = 1;

  marg_data.order.abs_order_map[t_ns] = std::make_pair(0, POSE_VEL_BIAS_SIZE);
  marg_data.order.total_size = POSE_VEL_BIAS_SIZE;
  marg_data.order.items = 1;

  nullspace_marg_data.order = marg_data.order;

  initialize(bg, ba);
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::initialize(const Eigen::Vector3d& bg_,
                                                   const Eigen::Vector3d& ba_) {
  auto proc_func = [&, bg = bg_.cast<Scalar>(), ba = ba_.cast<Scalar>()] {
    OpticalFlowResult::Ptr prev_frame, curr_frame;
    typename IntegratedImuMeasurement<Scalar>::Ptr meas;

    const Vec3 accel_cov =
        calib.dicrete_time_accel_noise_std().array().square();
    const Vec3 gyro_cov = calib.dicrete_time_gyro_noise_std().array().square();

    typename ImuData<Scalar>::Ptr data = popFromImuDataQueue();
    BASALT_ASSERT_MSG(data, "first IMU measurment is nullptr");

    data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
    data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);

    while (true) {
      vision_data_queue.pop(curr_frame);

      if (config.vio_enforce_realtime) {
        // drop current frame if another frame is already in the queue.
        while (!vision_data_queue.empty()) vision_data_queue.pop(curr_frame);
      }

      if (!curr_frame.get()) {
        break;
      }

      // Correct camera time offset
      // curr_frame->t_ns += calib.cam_time_offset_ns;

      if (!initialized) {
        while (data->t_ns < curr_frame->t_ns) {
          data = popFromImuDataQueue();
          if (!data) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
          // std::cout << "Skipping IMU data.." << std::endl;
        }

        Vec3 vel_w_i_init;
        vel_w_i_init.setZero();

        /// init pose
        T_w_i_init.setQuaternion(Eigen::Quaternion<Scalar>::FromTwoVectors(data->accel, Vec3::UnitZ()));

        T_w_y_init.setQuaternion(Eigen::Quaternion<Scalar>::FromTwoVectors(data->accel, Vec3::UnitZ()));
        std::cout << "T_w_y_init: " << std::endl << T_w_y_init.so3().matrix() << std::endl;

//        T_w_i_init.setQuaternion(Eigen::Quaternion<Scalar>::Identity());

        last_state_t_ns = curr_frame->t_ns;
        imu_meas[last_state_t_ns] =
            IntegratedImuMeasurement<Scalar>(last_state_t_ns, bg, ba);
        frame_states[last_state_t_ns] = PoseVelBiasStateWithLin<Scalar>(
            last_state_t_ns, T_w_i_init, vel_w_i_init, bg, ba, true);

        marg_vo_data.order.abs_order_map[last_state_t_ns] =
            std::make_pair(0, POSE_SIZE);
        marg_vo_data.order.total_size = POSE_SIZE;
        marg_vo_data.order.items = 1;

//        marg_data.order.abs_order_map[last_state_t_ns] = std::make_pair(0, POSE_VEL_BIAS_SIZE);
//        marg_data.order.total_size = POSE_VEL_BIAS_SIZE;
//        marg_data.order.items = 1;

        std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
        std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;
        std::cout << "vel_w_i " << vel_w_i_init.transpose() << std::endl;

        if (config.vio_debug || config.vio_extended_logging) {
          logMargNullspace();
        }

        initialized = true;
      }

      if (prev_frame) {
        // preintegrate measurements

        auto last_state = frame_states.at(last_state_t_ns);

        meas.reset(new IntegratedImuMeasurement<Scalar>(
            prev_frame->t_ns, last_state.getState().bias_gyro,
            last_state.getState().bias_accel));

        BASALT_ASSERT_MSG(prev_frame->t_ns < curr_frame->t_ns,
                          "duplicate frame timestamps?! zero time delta leads "
                          "to invalid IMU integration.");

        while (data->t_ns <= prev_frame->t_ns) {
          data = popFromImuDataQueue();
          if (!data) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
        }

        while (data->t_ns <= curr_frame->t_ns) {
          meas->integrate(*data, accel_cov, gyro_cov);
          data = popFromImuDataQueue();
          if (!data) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
        }

        if (meas->get_start_t_ns() + meas->get_dt_ns() < curr_frame->t_ns) {
          if (!data.get()) break;
          int64_t tmp = data->t_ns;
          data->t_ns = curr_frame->t_ns;
          meas->integrate(*data, accel_cov, gyro_cov);
          data->t_ns = tmp;
        }
      }

      measure(curr_frame, meas);
      prev_frame = curr_frame;
    }

    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_marg_queue) out_marg_queue->push(nullptr);
    if (out_state_queue) out_state_queue->push(nullptr);

    finished = true;

    std::cout << "Finished VIOFilter " << std::endl;
  };

  processing_thread.reset(new std::thread(proc_func));
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::addIMUToQueue(
    const ImuData<double>::Ptr& data) {
  imu_data_queue.emplace(data);
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::addVisionToQueue(
    const OpticalFlowResult::Ptr& data) {
  vision_data_queue.push(data);
}

template <class Scalar_>
typename ImuData<Scalar_>::Ptr
SqrtKeypointVioInitEstimator<Scalar_>::popFromImuDataQueue() {
  ImuData<double>::Ptr data;
  imu_data_queue.pop(data);

  if constexpr (std::is_same_v<Scalar, double>) {
    return data;
  } else {
    typename ImuData<Scalar>::Ptr data2;
    if (data) {
      data2.reset(new ImuData<Scalar>);
      *data2 = data->cast<Scalar>();
    }
    return data2;
  }
}

template <class Scalar_>
bool SqrtKeypointVioInitEstimator<Scalar_>::measure(
    const OpticalFlowResult::Ptr& opt_flow_meas,
    const typename IntegratedImuMeasurement<Scalar>::Ptr& meas) {
  stats_sums_.add("frame_id", opt_flow_meas->t_ns).format("none");
  Timer t_total;

  if (meas.get()) {
    BASALT_ASSERT(frame_states[last_state_t_ns].getState().t_ns == meas->get_start_t_ns());
    BASALT_ASSERT(opt_flow_meas->t_ns == meas->get_dt_ns() + meas->get_start_t_ns());
    BASALT_ASSERT(meas->get_dt_ns() > 0);

    PoseVelBiasState<Scalar> next_state = frame_states.at(last_state_t_ns).getState();

    if (!is_system_initialized) {
      meas->predictState(frame_states.at(last_state_t_ns).getState(), g,next_state);
    }


    last_state_t_ns = opt_flow_meas->t_ns;
    next_state.t_ns = opt_flow_meas->t_ns;
    frame_states[last_state_t_ns] = PoseVelBiasStateWithLin<Scalar>(next_state);

    imu_meas[meas->get_start_t_ns()] = *meas;

    std::cout << "frame time: " << last_state_t_ns << " imu_meas time: " << meas->get_start_t_ns() << std::endl;
  }

  // save results
  prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

  // Make new residual for existing keypoints
  int connected0 = 0;  /// num tracked landmarks in cam0
  std::map<int64_t, int> num_points_connected; /// num tracked landmarks by host
  std::unordered_set<int> unconnected_obs0; /// new tracks in cam0
  /// num tracked landmarks' ids in cam0 and cam1
  std::vector<std::vector<int>> connected_obs0(opt_flow_meas->observations.size());

  for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
    TimeCamId tcid_target(opt_flow_meas->t_ns, i);

    for (const auto& kv_obs : opt_flow_meas->observations[i]) {
      int kpt_id = kv_obs.first;

      if (lmdb.landmarkExists(kpt_id)) {
        const TimeCamId& tcid_host = lmdb.getLandmark(kpt_id).host_kf_id;

        KeypointObservation<Scalar> kobs;
        kobs.kpt_id = kpt_id;
        kobs.pos = kv_obs.second.translation().cast<Scalar>();

        lmdb.addObservation(tcid_target, kobs);
        // obs[tcid_host][tcid_target].push_back(kobs);

        if (num_points_connected.count(tcid_host.frame_id) == 0) {
          num_points_connected[tcid_host.frame_id] = 0;
        }
        num_points_connected[tcid_host.frame_id]++;
        connected_obs0[i].emplace_back(kpt_id);

        if (i == 0) connected0++;
      } else {
        if (i == 0) {
          unconnected_obs0.emplace(kpt_id);
        }
      }
    }
  }

  if (Scalar(connected0) / (connected0 + unconnected_obs0.size()) <
          Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;

  if (config.vio_debug) {
    std::cout << "connected0 " << connected0 << " unconnected0 "
              << unconnected_obs0.size() << std::endl;
  }

  if (!is_system_initialized) {
    /// BA to optimize current frame's pose
    BundleAdjustmentBase<Scalar>::optimize_single_frame_pose(frame_states[last_state_t_ns], connected_obs0);
  }

  if (take_kf) {
    // Triangulate new points from one of the observations (with sufficient
    // baseline) and make keyframe for camera 0
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);

    TimeCamId tcidl(opt_flow_meas->t_ns, 0);

    int num_points_added = 0;
    for (int lm_id : unconnected_obs0) {
      // Find all observations
      std::map<TimeCamId, KeypointObservation<Scalar>> kp_obs;

      for (const auto& kv : prev_opt_flow_res) {
        for (size_t k = 0; k < kv.second->observations.size(); k++) {
          auto it = kv.second->observations[k].find(lm_id);
          if (it != kv.second->observations[k].end()) {
            TimeCamId tcido(kv.first, k);

            KeypointObservation<Scalar> kobs;
            kobs.kpt_id = lm_id;
            kobs.pos = it->second.translation().template cast<Scalar>();

            // obs[tcidl][tcido].push_back(kobs);
            kp_obs[tcido] = kobs;
          }
        }
      }

      // triangulate
      bool valid_kp = false;
      const Scalar min_triang_distance2 =
          Scalar(config.vio_min_triangulation_dist *
                 config.vio_min_triangulation_dist);
      for (const auto& kv_obs : kp_obs) {
        if (valid_kp) break;
        TimeCamId tcido = kv_obs.first;

        const Vec2 p0 = opt_flow_meas->observations.at(0)
                            .at(lm_id)
                            .translation()
                            .cast<Scalar>();
        const Vec2 p1 = prev_opt_flow_res[tcido.frame_id]
                            ->observations[tcido.cam_id]
                            .at(lm_id)
                            .translation()
                            .template cast<Scalar>();

        Vec4 p0_3d, p1_3d;
        bool valid1 = calib.intrinsics[0].unproject(p0, p0_3d);
        bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
        if (!valid1 || !valid2) continue;

        SE3 T_i0_i1 = getPoseStateWithLin(tcidl.frame_id).getPose().inverse() *
                      getPoseStateWithLin(tcido.frame_id).getPose();
        SE3 T_0_1 =
            calib.T_i_c[0].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

        if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

        Vec4 p0_triangulated = triangulate(p0_3d.template head<3>(),
                                           p1_3d.template head<3>(), T_0_1);

        if (p0_triangulated.array().isFinite().all() &&
            p0_triangulated[3] > 0 && p0_triangulated[3] < 3.0) {
          Keypoint<Scalar> kpt_pos;
          kpt_pos.host_kf_id = tcidl;
          kpt_pos.direction = StereographicParam<Scalar>::project(p0_triangulated);
          kpt_pos.inv_dist = p0_triangulated[3];
          lmdb.addLandmark(lm_id, kpt_pos);

          num_points_added++;
          valid_kp = true;
        }
      }

      if (valid_kp) {
        for (const auto& kv_obs : kp_obs) {
          lmdb.addObservation(kv_obs.first, kv_obs.second);
        }
      }
    }

    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }

  std::unordered_set<KeypointId> lost_landmaks;
  if (config.vio_marg_lost_landmarks) {
    for (const auto& kv : lmdb.getLandmarks()) {
      bool connected = false;
      for (size_t i = 0; i < opt_flow_meas->observations.size(); i++) {
        if (opt_flow_meas->observations[i].count(kv.first) > 0)
          connected = true;
      }
      if (!connected) {
        lost_landmaks.emplace(kv.first);
      }
    }
  }

  optimize_and_marg(num_points_connected, lost_landmaks);

  if (out_state_queue) {
    PoseVelBiasStateWithLin p = frame_states.at(last_state_t_ns);

    typename PoseVelBiasState<double>::Ptr data(
        new PoseVelBiasState<double>(p.getState().template cast<double>()));

    out_state_queue->push(data);
  }

  if (out_vis_queue) {
    VioVisualizationData::Ptr data(new VioVisualizationData);

    data->t_ns = last_state_t_ns;

    for (const auto& kv : frame_states) {
      data->states.emplace_back(
          kv.second.getState().T_w_i.template cast<double>());
    }

    for (const auto& kv : frame_poses) {
      data->frames.emplace_back(kv.second.getPose().template cast<double>());
    }

    get_current_points(data->points, data->point_ids);

    data->projections.resize(opt_flow_meas->observations.size());
    computeProjections(data->projections, last_state_t_ns);

    data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

    out_vis_queue->push(data);
  }

  last_processed_t_ns = last_state_t_ns;

  stats_sums_.add("measure", t_total.elapsed()).format("ms");

  return true;
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::logMargNullspace() {
  nullspace_marg_data.order = marg_data.order;
  if (config.vio_debug) {
    std::cout << "======== Marg nullspace ==========" << std::endl;
    stats_sums_.add("marg_ns", checkMargNullspace());
    std::cout << "=================================" << std::endl;
  } else {
    stats_sums_.add("marg_ns", checkMargNullspace());
  }
  stats_sums_.add("marg_ev", checkMargEigenvalues());
}

template <class Scalar_>
Eigen::VectorXd SqrtKeypointVioInitEstimator<Scalar_>::checkMargNullspace() const {
  return checkNullspace(nullspace_marg_data, frame_states, frame_poses,
                        config.vio_debug);
}

template <class Scalar_>
Eigen::VectorXd SqrtKeypointVioInitEstimator<Scalar_>::checkMargEigenvalues()
    const {
  return checkEigenvalues(nullspace_marg_data, false);
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::marginalize(
    const std::map<int64_t, int>& num_points_connected,
    const std::unordered_set<KeypointId>& lost_landmaks) {
  if (!opt_started) return;

  Timer t_total;

  if (frame_poses.size() > max_kfs || frame_states.size() >= max_states) {

    const int states_to_remove = frame_states.size() - max_states + 1;

    auto it = frame_states.cbegin();
    for (int i = 0; i < states_to_remove; i++) it++;
    int64_t last_state_to_marg = it->first;  /// the last state (the oldest one)

    AbsOrderMap aom;  /// Markov Blanket

    // remove all frame_poses that are not kfs
    std::set<int64_t> poses_to_marg;  /// frame_poses that need to marg (contains kfs_to_marg)
    for (const auto& kv : frame_poses) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

      if (kf_ids.count(kv.first) == 0) poses_to_marg.emplace(kv.first);

      // Check that we have the same order as marginalization
      BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_SIZE;
      aom.items++;
    }

    std::set<int64_t> states_to_marg_vel_bias; /// marg all frame_states's vel and bias that are kfs (include the [0, n - k] )
    std::set<int64_t> states_to_marg_all;  /// marg all frame_states that are not kfs (include the [0, n - k] )
    for (const auto& kv : frame_states) {
      if (kv.first > last_state_to_marg) break;

      if (kv.first != last_state_to_marg) {
        if (kf_ids.count(kv.first) > 0) {
          states_to_marg_vel_bias.emplace(kv.first);
        } else {
          states_to_marg_all.emplace(kv.first);
        }
      }

      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

      /// Check that we have the same order as marginalization
      if (aom.items < marg_data.order.abs_order_map.size())
        BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_VEL_BIAS_SIZE;
      aom.items++;
    }

    auto kf_ids_all = kf_ids;
    std::set<int64_t> kfs_to_marg;  /// kfs that need to marg
    /// states_to_marg_vel_bias is not empty, kf_ids.size() > max_kfs
    while (kf_ids.size() > max_kfs && !states_to_marg_vel_bias.empty()) {
      int64_t id_to_marg = -1;

      // starting from the oldest kf (and skipping the newest 2 kfs), try to
      // find a kf that has less than a small percentage of it's landmarks
      // tracked by the current frame
      if (kf_ids.size() > 2) {
        // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
        auto end_minus_2 = std::prev(kf_ids.end(), 2);

        for (auto it = kf_ids.begin(); it != end_minus_2; ++it) {
          if (num_points_connected.count(*it) == 0 ||
              (num_points_connected.at(*it) /
                   static_cast<float>(num_points_kf.at(*it)) <
               config.vio_kf_marg_feature_ratio)) {
            id_to_marg = *it;
            break;
          }
        }
      }

      // Note: This score function is taken from DSO, but it seems to mostly
      // marginalize the oldest keyframe. This may be due to the fact that
      // we don't have as long-lived landmarks, which may change if we ever
      // implement "rediscovering" of lost feature tracks by projecting
      // untracked landmarks into the localized frame.
      if (kf_ids.size() > 2 && id_to_marg < 0) {
        // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
        auto end_minus_2 = std::prev(kf_ids.end(), 2);

        int64_t last_kf = *kf_ids.crbegin();
        Scalar min_score = std::numeric_limits<Scalar>::max();
        int64_t min_score_id = -1;

        for (auto it1 = kf_ids.begin(); it1 != end_minus_2; ++it1) {
          // small distance to other keyframes --> higher score
          Scalar denom = 0;
          for (auto it2 = kf_ids.begin(); it2 != end_minus_2; ++it2) {
            denom += 1 / ((frame_poses.at(*it1).getPose().translation() -
                           frame_poses.at(*it2).getPose().translation())
                              .norm() +
                          Scalar(1e-5));
          }

          // small distance to latest kf --> lower score
          Scalar score =
              std::sqrt(
                  (frame_poses.at(*it1).getPose().translation() -
                   frame_states.at(last_kf).getState().T_w_i.translation())
                      .norm()) *
              denom;

          if (score < min_score) {
            min_score_id = *it1;
            min_score = score;
          }
        }

        id_to_marg = min_score_id;
      }

      // if no frame was selected, the logic above is faulty
      BASALT_ASSERT(id_to_marg >= 0);

      kfs_to_marg.emplace(id_to_marg);
      poses_to_marg.emplace(id_to_marg);

      kf_ids.erase(id_to_marg);  /// delete id_to_marg from kf_ids
    }

    //    std::cout << "marg order" << std::endl;
    //    aom.print_order();

    //    std::cout << "marg prior order" << std::endl;
    //    marg_order.print_order();

    if (config.vio_debug) {
      std::cout << "states_to_remove " << states_to_remove << std::endl;
      std::cout << "poses_to_marg.size() " << poses_to_marg.size() << std::endl;
      std::cout << "states_to_marg.size() " << states_to_marg_all.size()
                << std::endl;
      std::cout << "state_to_marg_vel_bias.size() "
                << states_to_marg_vel_bias.size() << std::endl;
      std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
    }

    Timer t_actual_marg;

    size_t asize = aom.total_size;

    bool is_lin_sqrt = isLinearizationSqrt(config.vio_linearization_type);

    MatX Q2Jp_or_H;
    VecX Q2r_or_b;

    {
      Timer t_linearize;

      typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
      lqr_options.lb_options.huber_parameter = huber_thresh;
      lqr_options.lb_options.obs_std_dev = obs_std_dev;
      lqr_options.linearization_type = config.vio_linearization_type;

      ImuLinData<Scalar> ild = {g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};

      for (const auto& kv : imu_meas) {
        int64_t start_t = kv.second.get_start_t_ns();
        int64_t end_t = kv.second.get_start_t_ns() + kv.second.get_dt_ns();

        if (aom.abs_order_map.count(start_t) == 0 ||
            aom.abs_order_map.count(end_t) == 0)
          continue;

        ild.imu_meas[kv.first] = &kv.second;
      }

      auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
          this, aom, lqr_options, &marg_data, &ild, &kfs_to_marg,
          &lost_landmaks, last_state_to_marg);

      lqr->linearizeProblem();
      lqr->performQR();

      if (is_lin_sqrt && marg_data.is_sqrt) {
        lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
      } else {
        lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);
      }

      stats_sums_.add("marg_linearize", t_linearize.elapsed()).format("ms");
    }

    //    KeypointVioEstimator::linearizeAbsIMU(
    //        aom, accum.getH(), accum.getB(), imu_error, bg_error, ba_error,
    //        frame_states, imu_meas, gyro_bias_weight, accel_bias_weight, g);
    //    linearizeMargPrior(marg_order, marg_sqrt_H, marg_sqrt_b, aom,
    //    accum.getH(),
    //                       accum.getB(), marg_prior_error);

    // Save marginalization prior
    if (out_marg_queue && !kfs_to_marg.empty()) {
      // int64_t kf_id = *kfs_to_marg.begin();

      {
        MargData::Ptr m(new MargData);
        m->aom = aom;

        if (is_lin_sqrt && marg_data.is_sqrt) {
          m->abs_H =
              (Q2Jp_or_H.transpose() * Q2Jp_or_H).template cast<double>();
          m->abs_b = (Q2Jp_or_H.transpose() * Q2r_or_b).template cast<double>();
        } else {
          m->abs_H = Q2Jp_or_H.template cast<double>();

          m->abs_b = Q2r_or_b.template cast<double>();
        }

        assign_cast_map_values(m->frame_poses, frame_poses);
        assign_cast_map_values(m->frame_states, frame_states);
        m->kfs_all = kf_ids_all;
        m->kfs_to_marg = kfs_to_marg;
        m->use_imu = true;

        for (int64_t t : m->kfs_all) {
          m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
        }

        out_marg_queue->push(m);
      }
    }

    std::set<int> idx_to_keep, idx_to_marg;
    for (const auto& kv : aom.abs_order_map) {
      if (kv.second.second == POSE_SIZE) {
        int start_idx = kv.second.first;
        if (poses_to_marg.count(kv.first) == 0) {
          for (size_t i = 0; i < POSE_SIZE; i++)
            idx_to_keep.emplace(start_idx + i);
        } else {
          for (size_t i = 0; i < POSE_SIZE; i++)
            idx_to_marg.emplace(start_idx + i);
        }
      } else {
        BASALT_ASSERT(kv.second.second == POSE_VEL_BIAS_SIZE);
        // state
        int start_idx = kv.second.first;
        if (states_to_marg_all.count(kv.first) > 0) {
          for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
            idx_to_marg.emplace(start_idx + i);
        } else if (states_to_marg_vel_bias.count(kv.first) > 0) {
          for (size_t i = 0; i < POSE_SIZE; i++)
            idx_to_keep.emplace(start_idx + i);
          for (size_t i = POSE_SIZE; i < POSE_VEL_BIAS_SIZE; i++)
            idx_to_marg.emplace(start_idx + i);
        } else {
          BASALT_ASSERT(kv.first == last_state_to_marg);
          for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
            idx_to_keep.emplace(start_idx + i);
        }
      }
    }

    if (config.vio_debug) {
      std::cout << "keeping " << idx_to_keep.size() << " marg "
                << idx_to_marg.size() << " total " << asize << std::endl;
      std::cout << "last_state_to_marg " << last_state_to_marg
                << " frame_poses " << frame_poses.size() << " frame_states "
                << frame_states.size() << std::endl;
    }

    if (config.vio_debug || config.vio_extended_logging) {
      MatX Q2Jp_or_H_nullspace;
      VecX Q2r_or_b_nullspace;

      typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
      lqr_options.lb_options.huber_parameter = huber_thresh;
      lqr_options.lb_options.obs_std_dev = obs_std_dev;
      lqr_options.linearization_type = config.vio_linearization_type;

      nullspace_marg_data.order = marg_data.order;

      ImuLinData<Scalar> ild = {
          g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};

      for (const auto& kv : imu_meas) {
        int64_t start_t = kv.second.get_start_t_ns();
        int64_t end_t = kv.second.get_start_t_ns() + kv.second.get_dt_ns();

        if (aom.abs_order_map.count(start_t) == 0 ||
            aom.abs_order_map.count(end_t) == 0)
          continue;

        ild.imu_meas[kv.first] = &kv.second;
      }

      auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
          this, aom, lqr_options, &nullspace_marg_data, &ild, &kfs_to_marg,
          &lost_landmaks, last_state_to_marg);

      lqr->linearizeProblem();
      lqr->performQR();

      if (is_lin_sqrt && marg_data.is_sqrt) {
        lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
      } else {
        lqr->get_dense_H_b(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
      }

      MatX nullspace_sqrt_H_new;
      VecX nullspace_sqrt_b_new;

      if (is_lin_sqrt && marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
            Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
            nullspace_sqrt_H_new, nullspace_sqrt_b_new);
      } else if (marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqToSqrt(
            Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
            nullspace_sqrt_H_new, nullspace_sqrt_b_new);
      } else {
        MargHelper<Scalar>::marginalizeHelperSqToSq(
            Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
            nullspace_sqrt_H_new, nullspace_sqrt_b_new);
      }

      nullspace_marg_data.H = nullspace_sqrt_H_new;
      nullspace_marg_data.b = nullspace_sqrt_b_new;
    }

    MatX marg_H_new;
    VecX marg_b_new;

    {
      Timer t;
      if (is_lin_sqrt && marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
            Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
            marg_b_new);
      } else if (marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqToSqrt(Q2Jp_or_H, Q2r_or_b,
                                                      idx_to_keep, idx_to_marg,
                                                      marg_H_new, marg_b_new);
      } else {
        MargHelper<Scalar>::marginalizeHelperSqToSq(Q2Jp_or_H, Q2r_or_b,
                                                    idx_to_keep, idx_to_marg,
                                                    marg_H_new, marg_b_new);
      }

      stats_sums_.add("marg_helper", t.elapsed()).format("ms");
    }

    {
      BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() ==
                    false);
      frame_states.at(last_state_to_marg).setLinTrue();
    }

    for (const int64_t id : states_to_marg_all) {
      frame_states.erase(id);
      imu_meas.erase(id);
      prev_opt_flow_res.erase(id);
    }

    for (const int64_t id : states_to_marg_vel_bias) {
      const PoseVelBiasStateWithLin<Scalar>& state = frame_states.at(id);
      PoseStateWithLin<Scalar> pose(state);

      frame_poses[id] = pose;
      frame_states.erase(id);
      imu_meas.erase(id);
    }

    for (const int64_t id : poses_to_marg) {
      frame_poses.erase(id);
      prev_opt_flow_res.erase(id);
    }

    lmdb.removeKeyframes(kfs_to_marg, poses_to_marg, states_to_marg_all);

    if (config.vio_marg_lost_landmarks) {
      for (const auto& lm_id : lost_landmaks) lmdb.removeLandmark(lm_id);
    }

    AbsOrderMap marg_order_new;

    for (const auto& kv : frame_poses) {
      marg_order_new.abs_order_map[kv.first] =
          std::make_pair(marg_order_new.total_size, POSE_SIZE);

      marg_order_new.total_size += POSE_SIZE;
      marg_order_new.items++;
    }

    {
      marg_order_new.abs_order_map[last_state_to_marg] =
          std::make_pair(marg_order_new.total_size, POSE_VEL_BIAS_SIZE);
      marg_order_new.total_size += POSE_VEL_BIAS_SIZE;
      marg_order_new.items++;
    }

    marg_data.H = marg_H_new;
    marg_data.b = marg_b_new;
    marg_data.order = marg_order_new;

    BASALT_ASSERT(size_t(marg_data.H.cols()) == marg_data.order.total_size);

    // Quadratic prior and "delta" of the current state to the original
    // linearization point give cost function
    //
    //    P(x) = 0.5 || J*(delta+x) + r ||^2.
    //
    // For marginalization this has been linearized at x=0 to give
    // linearization
    //
    //    P(x) = 0.5 || J*x + (J*delta + r) ||^2,
    //
    // with Jacobian J and residual J*delta + r.
    //
    // After marginalization, we recover the original form of the
    // prior. We are left with linearization (in sqrt form)
    //
    //    Pnew(x) = 0.5 || Jnew*x + res ||^2.
    //
    // To recover the original form with delta-independent r, we set
    //
    //    Pnew(x) = 0.5 || Jnew*(delta+x) + (res - Jnew*delta) ||^2,
    //
    // and thus rnew = (res - Jnew*delta).

    VecX delta;
    computeDelta(marg_data.order, delta);
    marg_data.b -= marg_data.H * delta;

    if (config.vio_debug || config.vio_extended_logging) {
      VecX delta;
      computeDelta(marg_data.order, delta);
      nullspace_marg_data.b -= nullspace_marg_data.H * delta;
    }

    stats_sums_.add("marg", t_actual_marg.elapsed()).format("ms");

    if (config.vio_debug) {
      std::cout << "marginalizaon done!!" << std::endl;
    }

    if (config.vio_debug || config.vio_extended_logging) {
      Timer t;
      logMargNullspace();
      stats_sums_.add("marg_log", t.elapsed()).format("ms");
    }

    //    std::cout << "new marg prior order" << std::endl;
    //    marg_order.print_order();
  }

  stats_sums_.add("marginalize", t_total.elapsed()).format("ms");
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::marginalize_imu() {
  /// check
  auto kf_beg = kf_ids.begin();
  auto frame_beg = frame_states.begin();
  for (; kf_beg != kf_ids.end(); ++kf_beg, ++frame_beg) {
    BASALT_ASSERT(frame_beg->first == *kf_beg);
  }

  const int states_to_remove = frame_states.size() - 2;
  auto it_beg = frame_states.cbegin();
  for (int i = 0; i < states_to_remove; i++) it_beg++;
  int64_t last_state_to_marg = it_beg->first;

  AbsOrderMap aom;  /// Markov Blanket

  std::set<int64_t> poses_to_marg;  // empty
  std::set<int64_t> kfs_to_marg;  // empty
  std::map<int64_t, int> num_points_connected; // empty
  std::unordered_set<KeypointId> lost_landmaks; //empty

  std::set<int64_t> states_to_marg_vel_bias;
  std::set<int64_t> states_to_marg_all; // empty
  for (const auto& kv : frame_states) {
    if (kv.first > last_state_to_marg) break;

    if (kv.first != last_state_to_marg) {
      states_to_marg_vel_bias.emplace(kv.first);
    }

    aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

    //TODO:
//    // Check that we have the same order as marginalization
//    if (aom.items < marg_data.order.abs_order_map.size())
//      BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) ==
//                    aom.abs_order_map.at(kv.first));

    aom.total_size += POSE_VEL_BIAS_SIZE;
    aom.items++;
  }

  std::cout << "states_to_marg_vel_bias size: " << states_to_marg_vel_bias.size() << std::endl;

  bool is_lin_sqrt = isLinearizationSqrt(config.vio_linearization_type);

  MatX Q2Jp_or_H;
  VecX Q2r_or_b;
  {
    typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
    lqr_options.lb_options.huber_parameter = huber_thresh;
    lqr_options.lb_options.obs_std_dev = obs_std_dev;
    lqr_options.linearization_type = config.vio_linearization_type;

    ImuLinData<Scalar> ild = {g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};

    for (const auto& kv : imu_meas) {
      int64_t start_t = kv.second.get_start_t_ns();
      int64_t end_t = kv.second.get_start_t_ns() + kv.second.get_dt_ns();

      if (aom.abs_order_map.count(start_t) == 0 ||
          aom.abs_order_map.count(end_t) == 0)
        continue;

      ild.imu_meas[kv.first] = &kv.second;
    }

    auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
        this, aom, lqr_options, &marg_data, &ild, &kfs_to_marg,
        &lost_landmaks, last_state_to_marg);

    /// linearize problem
    std::cout << "linearizeProblem" << std::endl;
    lqr->linearizeProblem();
    lqr->performQR();

    if (is_lin_sqrt && marg_data.is_sqrt) {
      lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
    } else {
      lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);
    }
  }

  /// get keep id and marg id
  std::cout << "get idx_to_keep and idx_to_marg" << std::endl;
  std::set<int> idx_to_keep, idx_to_marg;
  for (const auto& kv : aom.abs_order_map) {
    if (kv.second.second == POSE_SIZE) {
      int start_idx = kv.second.first;
      if (poses_to_marg.count(kv.first) == 0) {
        for (size_t i = 0; i < POSE_SIZE; i++)
          idx_to_keep.emplace(start_idx + i);
      } else {
        for (size_t i = 0; i < POSE_SIZE; i++)
          idx_to_marg.emplace(start_idx + i);
      }
    } else {
      BASALT_ASSERT(kv.second.second == POSE_VEL_BIAS_SIZE);
      // state
      int start_idx = kv.second.first;
      if (states_to_marg_all.count(kv.first) > 0) {
        for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
          idx_to_marg.emplace(start_idx + i);
      } else if (states_to_marg_vel_bias.count(kv.first) > 0) {
        for (size_t i = 0; i < POSE_SIZE; i++)
          idx_to_keep.emplace(start_idx + i);
        for (size_t i = POSE_SIZE; i < POSE_VEL_BIAS_SIZE; i++)
          idx_to_marg.emplace(start_idx + i);
      } else {
        BASALT_ASSERT(kv.first == last_state_to_marg);
        for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++)
          idx_to_keep.emplace(start_idx + i);
      }
    }
  }

  /// get marg H and b
  std::cout << " get marg H and b" << std::endl;
  MatX marg_H_new;
  VecX marg_b_new;
  {
    if (is_lin_sqrt && marg_data.is_sqrt) {
      MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
          Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
          marg_b_new);
    } else if (marg_data.is_sqrt) {
      MargHelper<Scalar>::marginalizeHelperSqToSqrt(Q2Jp_or_H, Q2r_or_b,
                                                    idx_to_keep, idx_to_marg,
                                                    marg_H_new, marg_b_new);
    } else {
      MargHelper<Scalar>::marginalizeHelperSqToSq(Q2Jp_or_H, Q2r_or_b,
                                                  idx_to_keep, idx_to_marg,
                                                  marg_H_new, marg_b_new);
    }
  }

  std::cout << " set lin true" << std::endl;
  {
    /// set lin true
//    BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() == false);
    for (auto it = frame_states.begin(); it != frame_states.end(); ++it) {
      if (it->first > last_state_to_marg) break;
      it->second.setLinTrue();
    }
  }

  for (const int64_t id : states_to_marg_vel_bias) {
    const PoseVelBiasStateWithLin<Scalar>& state = frame_states.at(id);
    PoseStateWithLin<Scalar> pose(state);

    frame_poses[id] = pose;
    frame_states.erase(id);
    imu_meas.erase(id);
  }

//  kf_ids.erase(last_state_to_marg);

  AbsOrderMap marg_order_new;

  for (const auto& kv : frame_poses) {
    marg_order_new.abs_order_map[kv.first] = std::make_pair(marg_order_new.total_size, POSE_SIZE);

    marg_order_new.total_size += POSE_SIZE;
    marg_order_new.items++;
  }

  {
    marg_order_new.abs_order_map[last_state_to_marg] = std::make_pair(marg_order_new.total_size, POSE_VEL_BIAS_SIZE);
    marg_order_new.total_size += POSE_VEL_BIAS_SIZE;
    marg_order_new.items++;
  }

  marg_data.H = marg_H_new;
  marg_data.b = marg_b_new;
  marg_data.order = marg_order_new;

  BASALT_ASSERT(size_t(marg_data.H.cols()) == marg_data.order.total_size);

  VecX delta;
  computeDelta(marg_data.order, delta);
  marg_data.b -= marg_data.H * delta;
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::optimize() {
  if (config.vio_debug) {
    std::cout << "=================================" << std::endl;
  }

  if (opt_started || frame_states.size() > 4) {
    opt_started = true;

    // harcoded configs
    // bool scale_Jp = config.vio_scale_jacobian && is_qr_solver();
    // bool scale_Jl = config.vio_scale_jacobian && is_qr_solver();

    // timing
    ExecutionStats stats;
    Timer timer_total;
    Timer timer_iteration;

//    std::cout << "marg_data" << std::endl;
//    std::cout << marg_data.is_sqrt << " H: " << marg_data.H.rows() << " " << marg_data.H.cols() << " " << marg_data.b.rows() << std::endl;
//    std::cout << "order: items=" << marg_data.order.items << " " << marg_data.order.total_size << std::endl;
//    std::cout << "abs_order_map: " << std::endl;
//    for (auto it = marg_data.order.abs_order_map.begin(); it != marg_data.order.abs_order_map.end(); ++it) {
//      std::cout << it->first << " " << it->second.first << " " << it->second.second;
//    }
//    std::cout << "-------" << std::endl;

    // construct order of states in linear system --> sort by ascending
    // timestamp
    AbsOrderMap aom;

    for (const auto& kv : frame_poses) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

      // Check that we have the same order as marginalization
      BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_SIZE;
      aom.items++;
    }

    for (const auto& kv : frame_states) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

      // Check that we have the same order as marginalization
      if (aom.items < marg_data.order.abs_order_map.size())
        BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_VEL_BIAS_SIZE;
      aom.items++;
    }

    // TODO: Check why we get better accuracy with old SC loop. Possible
    // culprits:
    // - different initial lambda (based on previous iteration)
    // - no landmark damping
    // - outlier removal after 4 iterations?
    lambda = Scalar(config.vio_lm_lambda_initial);

    // record stats
    stats.add("num_cams", this->frame_poses.size()).format("count");
    stats.add("num_lms", this->lmdb.numLandmarks()).format("count");
    stats.add("num_obs", this->lmdb.numObservations()).format("count");

    // setup landmark blocks
    typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
    lqr_options.lb_options.huber_parameter = huber_thresh;
    lqr_options.lb_options.obs_std_dev = obs_std_dev;
    lqr_options.linearization_type = config.vio_linearization_type;

    std::unique_ptr<LinearizationBase<Scalar, POSE_SIZE>> lqr;

    ImuLinData<Scalar> ild = {
        g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};
    for (const auto& kv : imu_meas) {
      ild.imu_meas[kv.first] = &kv.second;
    }

    {
      Timer t;
      lqr = LinearizationBase<Scalar, POSE_SIZE>::create(this, aom, lqr_options,
                                                         &marg_data, &ild);
      stats.add("allocateLMB", t.reset()).format("ms");
      lqr->log_problem_stats(stats);
    }

    bool terminated = false;
    bool converged = false;
    std::string message;

    int it = 0;
    int it_rejected = 0;
    for (; it <= config.vio_max_iterations && !terminated;) {
      if (it > 0) {
        timer_iteration.reset();
      }

      Scalar error_total = 0;
      VecX Jp_column_norm2;

      {
        // TODO: execution could be done staged

        Timer t;

        // linearize residuals
        bool numerically_valid;
        error_total = lqr->linearizeProblem(&numerically_valid);
        BASALT_ASSERT_STREAM(
            numerically_valid,
            "did not expect numerical failure during linearization");
        stats.add("linearizeProblem", t.reset()).format("ms");

        //        // compute pose jacobian norm squared for Jacobian scaling
        //        if (scale_Jp) {
        //          Jp_column_norm2 = lqr->getJp_diag2();
        //          stats.add("getJp_diag2", t.reset()).format("ms");
        //        }

        //        // scale landmark jacobians
        //        if (scale_Jl) {
        //          lqr->scaleJl_cols();
        //          stats.add("scaleJl_cols", t.reset()).format("ms");
        //        }

        // marginalize points in place
        lqr->performQR();
        stats.add("performQR", t.reset()).format("ms");
      }

      if (config.vio_debug) {
        // TODO: num_points debug output missing
        std::cout << "[LINEARIZE] Error: " << error_total << " num points "
                  << std::endl;
        std::cout << "Iteration " << it << " " << error_total << std::endl;
      }

      // compute pose jacobian scaling
      //      VecX jacobian_scaling;
      //      if (scale_Jp) {
      //        // TODO: what about jacobian scaling for SC solver?

      //        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
      //        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
      //        jacobian_scaling = (lqr_options.lb_options.jacobi_scaling_eps +
      //                            Jp_column_norm2.array().sqrt())
      //                               .inverse();
      //      }
      // if (config.vio_debug) {
      //   std::cout << "\t[INFO] Stage 1" << std::endl;
      //}

      // inner loop for backtracking in LM (still count as main iteration
      // though)
      for (int j = 0; it <= config.vio_max_iterations && !terminated; j++) {
        if (j > 0) {
          timer_iteration.reset();
          if (config.vio_debug) {
            std::cout << "Iteration " << it << ", backtracking" << std::endl;
          }
        }

        {
          // Timer t;

          // TODO: execution could be done staged

          //          // set (updated) damping for poses
          //          if (config.vio_lm_pose_damping_variant == 0) {
          //            lqr->setPoseDamping(lambda);
          //            stats.add("setPoseDamping", t.reset()).format("ms");
          //          }

          //          // scale landmark Jacobians only on the first inner
          //          iteration. if (scale_Jp && j == 0) {
          //            lqr->scaleJp_cols(jacobian_scaling);
          //            stats.add("scaleJp_cols", t.reset()).format("ms");
          //          }

          //          // set (updated) damping for landmarks
          //          if (config.vio_lm_landmark_damping_variant == 0) {
          //            lqr->setLandmarkDamping(lambda);
          //            stats.add("setLandmarkDamping", t.reset()).format("ms");
          //          }
        }

        // if (config.vio_debug) {
        //   std::cout << "\t[INFO] Stage 2 " << std::endl;
        // }

        VecX inc;
        {
          Timer t;

          // get dense reduced camera system
          MatX H;
          VecX b;

          lqr->get_dense_H_b(H, b);

          stats.add("get_dense_H_b", t.reset()).format("ms");

          int iter = 0;
          bool inc_valid = false;
          constexpr int max_num_iter = 3;

          while (iter < max_num_iter && !inc_valid) {
            VecX Hdiag_lambda = (H.diagonal() * lambda).cwiseMax(min_lambda);
            MatX H_copy = H;
            H_copy.diagonal() += Hdiag_lambda;

            Eigen::LDLT<Eigen::Ref<MatX>> ldlt(H_copy);
            inc = ldlt.solve(b);
            stats.add("solve", t.reset()).format("ms");

            if (!inc.array().isFinite().all()) {
              lambda = lambda_vee * lambda;
              lambda_vee *= vee_factor;
            } else {
              inc_valid = true;
            }
            iter++;
          }

          if (!inc_valid) {
            std::cerr << "Still invalid inc after " << max_num_iter
                      << " iterations." << std::endl;
          }
        }

        // backup state (then apply increment and check cost decrease)
        backup();

        // backsubstitute (with scaled pose increment)
        Scalar l_diff = 0;
        {
          // negate pose increment before point update
          inc = -inc;

          Timer t;
          l_diff = lqr->backSubstitute(inc);
          stats.add("backSubstitute", t.reset()).format("ms");
        }

        // undo jacobian scaling before applying increment to poses
        //        if (scale_Jp) {
        //          inc.array() *= jacobian_scaling.array();
        //        }

        // apply increment to poses
        for (auto& [frame_id, state] : frame_poses) {
          int idx = aom.abs_order_map.at(frame_id).first;
          state.applyInc(inc.template segment<POSE_SIZE>(idx));
        }

        for (auto& [frame_id, state] : frame_states) {
          int idx = aom.abs_order_map.at(frame_id).first;
          state.applyInc(inc.template segment<POSE_VEL_BIAS_SIZE>(idx));
        }

        // compute stepsize
        Scalar step_norminf = inc.array().abs().maxCoeff();

        // compute error update applying increment
        Scalar after_update_marg_prior_error = 0;
        Scalar after_update_vision_and_inertial_error = 0;

        {
          Timer t;
          computeError(after_update_vision_and_inertial_error);
          computeMargPriorError(marg_data, after_update_marg_prior_error);

          Scalar after_update_imu_error = 0, after_bg_error = 0,
                 after_ba_error = 0;
          ScBundleAdjustmentBase<Scalar>::computeImuError(
              aom, after_update_imu_error, after_bg_error, after_ba_error,
              frame_states, imu_meas, gyro_bias_sqrt_weight.array().square(),
              accel_bias_sqrt_weight.array().square(), g);

          after_update_vision_and_inertial_error +=
              after_update_imu_error + after_bg_error + after_ba_error;

          stats.add("computerError2", t.reset()).format("ms");
        }

        Scalar after_error_total = after_update_vision_and_inertial_error +
                                   after_update_marg_prior_error;

        // check cost decrease compared to quadratic model cost
        Scalar f_diff;
        bool step_is_valid = false;
        bool step_is_successful = false;
        Scalar relative_decrease = 0;
        {
          // compute actual cost decrease
          f_diff = error_total - after_error_total;

          relative_decrease = f_diff / l_diff;

          if (config.vio_debug) {
            std::cout << "\t[EVAL] error: {:.4e}, f_diff {:.4e} l_diff {:.4e} "
                         "step_quality {:.2e} step_size {:.2e}\n"_format(
                             after_error_total, f_diff, l_diff,
                             relative_decrease, step_norminf);
          }

          // TODO: consider to remove assert. For now we want to test if we
          // even run into the l_diff <= 0 case ever in practice
          // BASALT_ASSERT_STREAM(l_diff > 0, "l_diff " << l_diff);

          // l_diff <= 0 is a theoretical possibility if the model cost change
          // is tiny and becomes numerically negative (non-positive). It might
          // not occur since our linear systems are not that big (compared to
          // large scale BA for example) and we also abort optimization quite
          // early and usually don't have large damping (== tiny step size).
          step_is_valid = l_diff > 0;
          step_is_successful = step_is_valid && relative_decrease > 0;
        }

        double iteration_time = timer_iteration.elapsed();
        double cumulative_time = timer_total.elapsed();

        stats.add("iteration", iteration_time).format("ms");
        {
          basalt::MemoryInfo mi;
          if (get_memory_info(mi)) {
            stats.add("resident_memory", mi.resident_memory);
            stats.add("resident_memory_peak", mi.resident_memory_peak);
          }
        }

        if (step_is_successful) {
          BASALT_ASSERT(step_is_valid);

          if (config.vio_debug) {
            //          std::cout << "\t[ACCEPTED] lambda:" << lambda
            //                    << " Error: " << after_error_total <<
            //                    std::endl;

            std::cout << "\t[ACCEPTED] error: {:.4e}, lambda: {:.1e}, it_time: "
                         "{:.3f}s, total_time: {:.3f}s\n"
                         ""_format(after_error_total, lambda, iteration_time,
                                   cumulative_time);
          }

          lambda *= std::max<Scalar>(
              Scalar(1.0) / 3,
              1 - std::pow<Scalar>(2 * relative_decrease - 1, 3));
          lambda = std::max(min_lambda, lambda);

          lambda_vee = initial_vee;

          it++;

          // check function and parameter tolerance
          if ((f_diff > 0 && f_diff < Scalar(1e-6)) ||
              step_norminf < Scalar(1e-4)) {
            converged = true;
            terminated = true;
          }

          // stop inner lm loop
          break;
        } else {
          std::string reason = step_is_valid ? "REJECTED" : "INVALID";

          if (config.vio_debug) {
            //          std::cout << "\t[REJECTED] lambda:" << lambda
            //                    << " Error: " << after_error_total <<
            //                    std::endl;

            std::cout << "\t[{}] error: {}, lambda: {:.1e}, it_time:"
                         "{:.3f}s, total_time: {:.3f}s\n"
                         ""_format(reason, after_error_total, lambda,
                                   iteration_time, cumulative_time);
          }

          lambda = lambda_vee * lambda;
          lambda_vee *= vee_factor;

          //        lambda = std::max(min_lambda, lambda);
          //        lambda = std::min(max_lambda, lambda);

          restore();
          it++;
          it_rejected++;

          if (lambda > max_lambda) {
            terminated = true;
            message =
                "Solver did not converge and reached maximum damping lambda";
          }
        }
      }
    }

    stats.add("optimize", timer_total.elapsed()).format("ms");
    stats.add("num_it", it).format("count");
    stats.add("num_it_rejected", it_rejected).format("count");

    // TODO: call filterOutliers at least once (also for CG version)

    stats_all_.merge_all(stats);
    stats_sums_.merge_sums(stats);

    if (config.vio_debug) {
      if (!converged) {
        if (terminated) {
          std::cout << "Solver terminated early after {} iterations: {}"_format(
              it, message);
        } else {
          std::cout
              << "Solver did not converge after maximum number of {} iterations"_format(
                     it);
        }
      }

      stats.print();

      std::cout << "=================================" << std::endl;
    }
  }
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::marginalize_vo(
    const std::map<int64_t, int>& num_points_connected,
    const std::unordered_set<KeypointId>& lost_landmaks) {
//  BASALT_ASSERT(frame_states.empty());

  Timer t_total;

  if (true) {
    // Marginalize

    /// remove all frame_poses that are not kfs and not the current frame
    std::set<int64_t> non_kf_poses;
    for (const auto& kv : frame_poses) {
      if (kf_ids.count(kv.first) == 0 && kv.first != last_state_t_ns) {
        non_kf_poses.emplace(kv.first);
      }
    }

    for (int64_t id : non_kf_poses) {
      frame_poses.erase(id);
      lmdb.removeFrame(id);
      prev_opt_flow_res.erase(id);
    }

    auto kf_ids_all = kf_ids;
    std::set<FrameId> kfs_to_marg;   /// keyframes that need to marg
    while (kf_ids.size() > (max_kfs + max_states - 1)) {
      int64_t id_to_marg = -1;

      /// starting from the oldest kf (and skipping the newest 2 kfs), try to
      /// find a kf that has less than a small percentage of it's landmarks
      /// tracked by the current frame
      if (kf_ids.size() > 2) {
        // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
        auto end_minus_2 = std::prev(kf_ids.end(), 2);

        for (auto it = kf_ids.begin(); it != end_minus_2; ++it) {
          if (num_points_connected.count(*it) == 0 ||
              (num_points_connected.at(*it) / Scalar(num_points_kf.at(*it)) <
               Scalar(config.vio_kf_marg_feature_ratio))) {
            id_to_marg = *it;
            break;
          }
        }
      }

      // Note: This score function is taken from DSO, but it seems to mostly
      // marginalize the oldest keyframe. This may be due to the fact that
      // we don't have as long-lived landmarks, which may change if we ever
      // implement "rediscovering" of lost feature tracks by projecting
      // untracked landmarks into the localized frame.
      if (kf_ids.size() > 2 && id_to_marg < 0) {
        // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
        auto end_minus_2 = std::prev(kf_ids.end(), 2);

        int64_t last_kf = *kf_ids.crbegin();
        Scalar min_score = std::numeric_limits<Scalar>::max();
        int64_t min_score_id = -1;

        for (auto it1 = kf_ids.begin(); it1 != end_minus_2; ++it1) {
          // small distance to other keyframes --> higher score
          Scalar denom = 0;
          for (auto it2 = kf_ids.begin(); it2 != end_minus_2; ++it2) {
            denom += 1 / ((frame_poses.at(*it1).getPose().translation() -
                           frame_poses.at(*it2).getPose().translation())
                              .norm() +
                          Scalar(1e-5));
          }

          // small distance to latest kf --> lower score
          Scalar score =
              std::sqrt((frame_poses.at(*it1).getPose().translation() -
                         frame_poses.at(last_kf).getPose().translation())
                            .norm()) *
              denom;

          if (score < min_score) {
            min_score_id = *it1;
            min_score = score;
          }
        }

        id_to_marg = min_score_id;
      }

      // if no frame was selected, the logic above is faulty
      BASALT_ASSERT(id_to_marg >= 0);

      kfs_to_marg.emplace(id_to_marg);

      // Note: this looks like a leftover from VIO that is not doing anything in
      // VO -> we could check / compare / remove
      non_kf_poses.emplace(id_to_marg);

      kf_ids.erase(id_to_marg);
    }

    // Create AbsOrderMap entries that are in the marg prior or connected to the
    // keyframes that we marginalize
    // Create AbsOrderMap entries that are in the marg prior or connected to the
    // keyframes that we marginalize
    AbsOrderMap aom;
    {
      const auto& obs = lmdb.getObservations();

      aom.abs_order_map = marg_vo_data.order.abs_order_map;
      aom.total_size = marg_vo_data.order.total_size;
      aom.items = marg_vo_data.order.items;

      for (const auto& kv : frame_poses) {
        if (aom.abs_order_map.count(kv.first) == 0) {
          bool add_pose = false;

          for (const auto& [host, target_map] : obs) {
            // if one of the host frames that we marg out
            if (kfs_to_marg.count(host.frame_id) > 0) {
              for (const auto& [target, obs_map] : target_map) {
                // has observations in the frame also add it to marg prior
                if (target.frame_id == kv.first) {
                  add_pose = true;
                  break;
                }
              }
            }
            // Break if we already found one observation.
            if (add_pose) break;
          }

          if (add_pose) {
            aom.abs_order_map[kv.first] =
                std::make_pair(aom.total_size, POSE_SIZE);

            aom.total_size += POSE_SIZE;
            aom.items++;
          }
        }
      }

      // If marg lost landmakrs add corresponding frames to linearization
      if (config.vio_marg_lost_landmarks) {
        for (const auto& lm_id : lost_landmaks) {
          const auto& lm = lmdb.getLandmark(lm_id);
          if (aom.abs_order_map.count(lm.host_kf_id.frame_id) == 0) {
            aom.abs_order_map[lm.host_kf_id.frame_id] =
                std::make_pair(aom.total_size, POSE_SIZE);

            aom.total_size += POSE_SIZE;
            aom.items++;
          }

          for (const auto& [target, o] : lm.obs) {
            if (aom.abs_order_map.count(target.frame_id) == 0) {
              aom.abs_order_map[target.frame_id] =
                  std::make_pair(aom.total_size, POSE_SIZE);

              aom.total_size += POSE_SIZE;
              aom.items++;
            }
          }
        }
      }
    }

    //    std::cout << "marg order" << std::endl;
    //    aom.print_order();

    //    std::cout << "marg prior order" << std::endl;
    //    marg_order.print_order();

    if (config.vio_debug) {
      std::cout << "non_kf_poses.size() " << non_kf_poses.size() << std::endl;
      for (const auto& v : non_kf_poses) std::cout << v << ' ';
      std::cout << std::endl;

      std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
      for (const auto& v : kfs_to_marg) std::cout << v << ' ';
      std::cout << std::endl;

      std::cout << "last_state_t_ns " << last_state_t_ns << std::endl;

      std::cout << "frame_poses.size() " << frame_poses.size() << std::endl;
      for (const auto& v : frame_poses) std::cout << v.first << ' ';
      std::cout << std::endl;
    }

    // Remove unconnected frames
    if (!kfs_to_marg.empty()) {
      for (auto it = kfs_to_marg.cbegin(); it != kfs_to_marg.cend();) {
        if (aom.abs_order_map.count(*it) == 0) {
          frame_poses.erase(*it);
          prev_opt_flow_res.erase(*it);
          lmdb.removeKeyframes({*it}, {}, {});
          it = kfs_to_marg.erase(it);
        } else {
          it++;
        }
      }
    }

    std::cout << "kfs_to_marg: " << kfs_to_marg.size() << std::endl;
    BASALT_ASSERT(kfs_to_marg.size() == 0);

    if (!kfs_to_marg.empty()) {
      Timer t_actual_marg;

      // Marginalize only if last state is a keyframe
      BASALT_ASSERT(kf_ids_all.count(last_state_t_ns) > 0);

      size_t asize = aom.total_size;
      //      double marg_prior_error;

      //      DenseAccumulator accum;
      //      accum.reset(asize);

      bool is_lin_sqrt = isLinearizationSqrt(config.vio_linearization_type);

      MatX Q2Jp_or_H;
      VecX Q2r_or_b;

      {
        // Linearize points
        Timer t_linearize;

        typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
        lqr_options.lb_options.huber_parameter = huber_thresh;
        lqr_options.lb_options.obs_std_dev = obs_std_dev;
        lqr_options.linearization_type = config.vio_linearization_type;

        auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
            this, aom, lqr_options, &marg_vo_data, nullptr, &kfs_to_marg,
            &lost_landmaks);

        lqr->linearizeProblem();
        lqr->performQR();

        if (is_lin_sqrt && marg_vo_data.is_sqrt) {
          lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
        } else {
          lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);
        }

        stats_sums_.add("marg_linearize", t_linearize.elapsed()).format("ms");
      }

      // Save marginalization prior
      if (out_marg_queue && !kfs_to_marg.empty()) {
        // int64_t kf_id = *kfs_to_marg.begin();

        {
          MargData::Ptr m(new MargData);
          m->aom = aom;

          if (is_lin_sqrt && marg_vo_data.is_sqrt) {
            m->abs_H =
                (Q2Jp_or_H.transpose() * Q2Jp_or_H).template cast<double>();
            m->abs_b =
                (Q2Jp_or_H.transpose() * Q2r_or_b).template cast<double>();
          } else {
            m->abs_H = Q2Jp_or_H.template cast<double>();

            m->abs_b = Q2r_or_b.template cast<double>();
          }

          assign_cast_map_values(m->frame_poses, frame_poses);
          assign_cast_map_values(m->frame_states, frame_states);
          m->kfs_all = kf_ids_all;
          m->kfs_to_marg = kfs_to_marg;
          m->use_imu = false;

          for (int64_t t : m->kfs_all) {
            m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
          }

          out_marg_queue->push(m);
        }
      }

      std::set<int> idx_to_keep, idx_to_marg;
      for (const auto& kv : aom.abs_order_map) {
        if (kv.second.second == POSE_SIZE) {
          int start_idx = kv.second.first;
          if (kfs_to_marg.count(kv.first) == 0) {
            for (size_t i = 0; i < POSE_SIZE; i++)
              idx_to_keep.emplace(start_idx + i);
          } else {
            for (size_t i = 0; i < POSE_SIZE; i++)
              idx_to_marg.emplace(start_idx + i);
          }
        } else {
          BASALT_ASSERT(false);
        }
      }

      if (config.vio_debug) {
        std::cout << "keeping " << idx_to_keep.size() << " marg "
                  << idx_to_marg.size() << " total " << asize << std::endl;
        std::cout << " frame_poses " << frame_poses.size() << " frame_states "
                  << frame_states.size() << std::endl;
      }

      if (config.vio_debug || config.vio_extended_logging) {
        MatX Q2Jp_or_H_nullspace;
        VecX Q2r_or_b_nullspace;

        typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
        lqr_options.lb_options.huber_parameter = huber_thresh;
        lqr_options.lb_options.obs_std_dev = obs_std_dev;
        lqr_options.linearization_type = config.vio_linearization_type;

        nullspace_marg_data.order = marg_vo_data.order;

        auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
            this, aom, lqr_options, &nullspace_marg_data, nullptr, &kfs_to_marg,
            &lost_landmaks);

        lqr->linearizeProblem();
        lqr->performQR();

        if (is_lin_sqrt && marg_vo_data.is_sqrt) {
          lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
        } else {
          lqr->get_dense_H_b(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
        }

        MatX nullspace_sqrt_H_new;
        VecX nullspace_sqrt_b_new;

        if (is_lin_sqrt && marg_vo_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
              Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
              nullspace_sqrt_H_new, nullspace_sqrt_b_new);
        } else if (marg_vo_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqToSqrt(
              Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
              nullspace_sqrt_H_new, nullspace_sqrt_b_new);
        } else {
          MargHelper<Scalar>::marginalizeHelperSqToSq(
              Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
              nullspace_sqrt_H_new, nullspace_sqrt_b_new);
        }

        nullspace_marg_data.H = nullspace_sqrt_H_new;
        nullspace_marg_data.b = nullspace_sqrt_b_new;
      }

      MatX marg_sqrt_H_new;
      VecX marg_sqrt_b_new;

      {
        Timer t;
        if (is_lin_sqrt && marg_vo_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
              Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_sqrt_H_new,
              marg_sqrt_b_new);
        } else if (marg_vo_data.is_sqrt) {
          MargHelper<Scalar>::marginalizeHelperSqToSqrt(
              Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_sqrt_H_new,
              marg_sqrt_b_new);
        } else {
          MargHelper<Scalar>::marginalizeHelperSqToSq(
              Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_sqrt_H_new,
              marg_sqrt_b_new);
        }
        stats_sums_.add("marg_helper", t.elapsed()).format("ms");
      }

      for (auto& kv : frame_poses) {
        if (aom.abs_order_map.count(kv.first) > 0) {
          if (!kv.second.isLinearized()) kv.second.setLinTrue();
        }
      }

      for (const int64_t id : kfs_to_marg) {
        frame_poses.erase(id);
        prev_opt_flow_res.erase(id);
      }

      lmdb.removeKeyframes(kfs_to_marg, kfs_to_marg, kfs_to_marg);

      if (config.vio_marg_lost_landmarks) {
        for (const auto& lm_id : lost_landmaks) lmdb.removeLandmark(lm_id);
      }

      AbsOrderMap marg_order_new;

      for (const auto& kv : frame_poses) {
        if (aom.abs_order_map.count(kv.first) > 0) {
          marg_order_new.abs_order_map[kv.first] =
              std::make_pair(marg_order_new.total_size, POSE_SIZE);

          marg_order_new.total_size += POSE_SIZE;
          marg_order_new.items++;
        }
      }

      marg_vo_data.H = marg_sqrt_H_new;
      marg_vo_data.b = marg_sqrt_b_new;
      marg_vo_data.order = marg_order_new;

      BASALT_ASSERT(size_t(marg_vo_data.H.cols()) == marg_vo_data.order.total_size);

      // Quadratic prior and "delta" of the current state to the original
      // linearization point give cost function
      //
      //    P(x) = 0.5 || J*(delta+x) + r ||^2.
      //
      // For marginalization this has been linearized at x=0 to give
      // linearization
      //
      //    P(x) = 0.5 || J*x + (J*delta + r) ||^2,
      //
      // with Jacobian J and residual J*delta + r.
      //
      // After marginalization, we recover the original form of the
      // prior. We are left with linearization (in sqrt form)
      //
      //    Pnew(x) = 0.5 || Jnew*x + res ||^2.
      //
      // To recover the original form with delta-independent r, we set
      //
      //    Pnew(x) = 0.5 || Jnew*(delta+x) + (res - Jnew*delta) ||^2,
      //
      // and thus rnew = (res - Jnew*delta).

      VecX delta;
      computeDelta(marg_vo_data.order, delta);
      marg_vo_data.b -= marg_vo_data.H * delta;

      if (config.vio_debug || config.vio_extended_logging) {
        VecX delta;
        computeDelta(marg_vo_data.order, delta);
        nullspace_marg_data.b -= nullspace_marg_data.H * delta;
      }

      stats_sums_.add("marg_total", t_actual_marg.elapsed()).format("ms");

      if (config.vio_debug) {
        std::cout << "marginalizaon done!!" << std::endl;
      }

      if (config.vio_debug || config.vio_extended_logging) {
        Timer t;
        logMargNullspace();
        stats_sums_.add("marg_log", t.elapsed()).format("ms");
      }
    }
    std::cout << "marg_vo_data_H : " << marg_vo_data.H << std::endl;

    //    std::cout << "new marg prior order" << std::endl;
    //    marg_order.print_order();

    ///////////////////////// re imu-preintegration //////////////////////////////////
    Timer t_re_preintegration;
    /// re imu-preintergration
    /// delete imu meas out of sliding window
    auto frame_it_beg = frame_poses.begin();
    std::set<int64_t> imu_to_delete;

    for (auto it = imu_meas.begin(); it != imu_meas.end(); ++it) {
      if (it->first < frame_it_beg->first) {
        imu_to_delete.emplace(it->first);
      } else {
        break;
      }
    }
    for (const int64_t id : imu_to_delete) {
      imu_meas.erase(id);
    }
    ////////////////////////////////////
    std::set<std::pair<int64_t, int64_t>> pose_times;  /// <start_t, end_t>
    auto next_frame = frame_poses.begin();
    next_frame++;
    for (auto it = frame_poses.begin(); it != frame_poses.end() && next_frame != frame_poses.end(); ++it, ++next_frame) {
      pose_times.emplace(std::make_pair(it->first, next_frame->first));
    }

    const Vec3 accel_cov = calib.dicrete_time_accel_noise_std().array().square();
    const Vec3 gyro_cov = calib.dicrete_time_gyro_noise_std().array().square();

    for (auto pose_time : pose_times) {

      auto it_first_meas = imu_meas.find(pose_time.first);
      BASALT_ASSERT(it_first_meas != imu_meas.end());
//      auto it_second_meas = imu_meas.find(pose_time.second);
//      BASALT_ASSERT(it_second_meas != imu_meas.end());

//      std::set<std::pair<int64_t, IntegratedImuMeasurement<Scalar>>> imu_meas_all;
      std::vector<ImuData<Scalar>> imu_meas_all;
      for (auto it = imu_meas.begin(); it != imu_meas.end(); ++it) {
        if (it->first > pose_time.second) {
          break;
        }
        if (it->first >= pose_time.first && it->first < pose_time.second) {
          for (auto imu_data : it->second.get_imu_meas()) {
            imu_meas_all.emplace_back(imu_data);
          }
//          imu_meas_all.emplace(std::make_pair(it->first, it->second));
        }
      }
      BASALT_ASSERT(imu_meas_all.size() > 0);
      /// re-preintergration
      IntegratedImuMeasurement<Scalar> meas(pose_time.first, Vec3::Zero(), Vec3::Zero());
      size_t i = 0;
      auto imu_data = imu_meas_all.at(i);
      imu_data.accel = calib.calib_accel_bias.getCalibrated(imu_data.accel);
      imu_data.gyro = calib.calib_gyro_bias.getCalibrated(imu_data.gyro);

      while (imu_data.t_ns <= pose_time.first) {
        i++;
        if (i == imu_meas_all.size()) {
          break;
        }
        imu_data = imu_meas_all.at(i);
        imu_data.accel = calib.calib_accel_bias.getCalibrated(imu_data.accel);
        imu_data.gyro = calib.calib_gyro_bias.getCalibrated(imu_data.gyro);
      }

      while (imu_data.t_ns <= pose_time.second) {
        i++;
        meas.integrate(imu_data, accel_cov, gyro_cov);
        if (i == imu_meas_all.size()) {
          break;
        }
        imu_data = imu_meas_all.at(i);
        imu_data.accel = calib.calib_accel_bias.getCalibrated(imu_data.accel);
        imu_data.gyro = calib.calib_gyro_bias.getCalibrated(imu_data.gyro);
      }

      if (meas.get_start_t_ns() + meas.get_dt_ns() < pose_time.second) {
        int64_t tmp = imu_data.t_ns;
        imu_data.t_ns = pose_time.second;
        meas.integrate(imu_data, accel_cov, gyro_cov);
        imu_data.t_ns = tmp;
      }

      imu_meas[meas.get_start_t_ns()] = meas;

      std::cout << "imu_meas_all size: " << imu_meas_all.size() << " dt: " << (pose_time.second - pose_time.first) * 1.0e-9 << " "
                << (imu_meas_all.back().t_ns - imu_meas_all.front().t_ns) * 1.0e-9 << std::endl;

      std::cout << (imu_meas_all.front().t_ns - pose_time.first) * 1.0e-6 << " ms, " << (imu_meas_all.back().t_ns - pose_time.second) * 1.0e-6 << " ms" << std::endl;

      std::cout << pose_time.first << " " << pose_time.second << std::endl;
      std::cout << imu_meas_all.front().t_ns << " " << imu_meas_all.back().t_ns << std::endl;
      std::cout << "------" << std::endl;
    }

    std::set<int64_t> imu_to_delete2;
    for (auto it = imu_meas.begin(); it != imu_meas.end(); ++it) {
      bool keep = false;
      for (auto pose_time : pose_times) {
        if (it->first == pose_time.first) {
          keep = true;
          break;
        }
      }
      if (!keep) {
        imu_to_delete2.emplace(it->first);
      }
    }
    for (const int64_t id : imu_to_delete2) {
      imu_meas.erase(id);
    }
    std::cout << "time: " << t_re_preintegration.elapsed() * 1000.0 << " ms" << std::endl;
  }

  stats_sums_.add("marginalize", t_total.elapsed()).format("ms");
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::optimize_vo() {
  if (config.vio_debug) {
    std::cout << "=================================" << std::endl;
  }

  std::cout << "optimize_vo" << std::endl;

  // harcoded configs
  //  bool scale_Jp = config.vio_scale_jacobian && is_qr_solver();
  //  bool scale_Jl = config.vio_scale_jacobian && is_qr_solver();

  // timing
  ExecutionStats stats;
  Timer timer_total;
  Timer timer_iteration;

  // construct order of states in linear system --> sort by ascending
  // timestamp
  AbsOrderMap aom;
  aom.abs_order_map = marg_vo_data.order.abs_order_map;
  aom.total_size = marg_vo_data.order.total_size;
  aom.items = marg_vo_data.order.items;

  for (const auto& kv : frame_poses) {
    if (aom.abs_order_map.count(kv.first) == 0) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);
      aom.total_size += POSE_SIZE;
      aom.items++;
    }
  }

  // This is VO not VIO, so expect no IMU states
//  BASALT_ASSERT(frame_states.empty());

  // TODO: Check why we get better accuracy with old SC loop. Possible culprits:
  // - different initial lambda (based on previous iteration)
  // - no landmark damping
  // - outlier removal after 4 iterations?
  lambda = Scalar(config.vio_lm_lambda_initial);

  // record stats
  stats.add("num_cams", frame_poses.size()).format("count");
  stats.add("num_lms", lmdb.numLandmarks()).format("count");
  stats.add("num_obs", lmdb.numObservations()).format("count");

  // setup landmark blocks
  typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
  lqr_options.lb_options.huber_parameter = huber_thresh;
  lqr_options.lb_options.obs_std_dev = obs_std_dev;
  lqr_options.linearization_type = config.vio_linearization_type;
  std::unique_ptr<LinearizationBase<Scalar, POSE_SIZE>> lqr;

  {
    Timer t;
    lqr = LinearizationBase<Scalar, POSE_SIZE>::create(this, aom, lqr_options,
                                                       &marg_vo_data);
    stats.add("allocateLMB", t.reset()).format("ms");
    lqr->log_problem_stats(stats);
  }

  bool terminated = false;
  bool converged = false;
  std::string message;

  int it = 0;
  int it_rejected = 0;
  for (; it <= config.vio_max_iterations && !terminated;) {
    if (it > 0) {
      timer_iteration.reset();
    }

    Scalar error_total = 0;
    VecX Jp_column_norm2;

    // TODO: execution could be done staged

    Timer t;

    // linearize residuals
    bool numerically_valid;
    error_total = lqr->linearizeProblem(&numerically_valid);
    BASALT_ASSERT_STREAM(
        numerically_valid,
        "did not expect numerical failure during linearization");
    stats.add("linearizeProblem", t.reset()).format("ms");

    //      // compute pose jacobian norm squared for Jacobian scaling
    //      if (scale_Jp) {
    //        Jp_column_norm2 = lqr->getJp_diag2();
    //        stats.add("getJp_diag2", t.reset()).format("ms");
    //      }

    //      // scale landmark jacobians
    //      if (scale_Jl) {
    //        lqr->scaleJl_cols();
    //        stats.add("scaleJl_cols", t.reset()).format("ms");
    //      }

    // marginalize points in place
    lqr->performQR();
    stats.add("performQR", t.reset()).format("ms");

    if (config.vio_debug) {
      // TODO: num_points debug output missing
      std::cout << "[LINEARIZE] Error: " << error_total << " num points "
                << std::endl;
      std::cout << "Iteration " << it << " " << error_total << std::endl;
    }

    // inner loop for backtracking in LM (still count as main iteration though)
    for (int j = 0; it <= config.vio_max_iterations && !terminated; j++) {
      if (j > 0) {
        timer_iteration.reset();
        if (config.vio_debug) {
          std::cout << "Iteration " << it << ", backtracking" << std::endl;
        }
      }

      {
        //        Timer t;

        // TODO: execution could be done staged

        // set (updated) damping for poses
        //        if (config.vio_lm_pose_damping_variant == 0) {
        //          lqr->setPoseDamping(lambda);
        //          stats.add("setPoseDamping", t.reset()).format("ms");
        //        }

        //        // scale landmark Jacobians only on the first inner iteration.
        //        if (scale_Jp && j == 0) {
        //          lqr->scaleJp_cols(jacobian_scaling);
        //          stats.add("scaleJp_cols", t.reset()).format("ms");
        //        }

        //        // set (updated) damping for landmarks
        //        if (config.vio_lm_landmark_damping_variant == 0) {
        //          lqr->setLandmarkDamping(lambda);
        //          stats.add("setLandmarkDamping", t.reset()).format("ms");
        //        }
      }

      // if (config.vio_debug) {
      //   std::cout << "\t[INFO] Stage 2 " << std::endl;
      // }

      VecX inc;
      {
        Timer t;

        // get dense reduced camera system
        MatX H;
        VecX b;

        lqr->get_dense_H_b(H, b);

        stats.add("get_dense_H_b", t.reset()).format("ms");

        int iter = 0;
        bool inc_valid = false;
        constexpr int max_num_iter = 3;

        while (iter < max_num_iter && !inc_valid) {
          VecX Hdiag_lambda = (H.diagonal() * lambda).cwiseMax(min_lambda);
          MatX H_copy = H;
          H_copy.diagonal() += Hdiag_lambda;

          Eigen::LDLT<Eigen::Ref<MatX>> ldlt(H_copy);
          inc = ldlt.solve(b);
          stats.add("solve", t.reset()).format("ms");

          if (!inc.array().isFinite().all()) {
            lambda = lambda_vee * lambda;
            lambda_vee *= vee_factor;
          } else {
            inc_valid = true;
          }
          iter++;
        }

        if (!inc_valid) {
          std::cerr << "Still invalid inc after " << max_num_iter
                    << " iterations." << std::endl;
        }
      }

      // backup state (then apply increment and check cost decrease)
      backup();

      // TODO: directly invert pose increment when solving; change SC
      // `updatePoints` to recieve unnegated increment

      // backsubstitute (with scaled pose increment)
      Scalar l_diff = 0;
      {
        // negate pose increment before point update
        inc = -inc;

        Timer t;
        l_diff = lqr->backSubstitute(inc);
        stats.add("backSubstitute", t.reset()).format("ms");
      }

      // undo jacobian scaling before applying increment to poses
      //      if (scale_Jp) {
      //        inc.array() *= jacobian_scaling.array();
      //      }

      // apply increment to poses
      for (auto& [frame_id, state] : frame_poses) {
        int idx = aom.abs_order_map.at(frame_id).first;
        state.applyInc(inc.template segment<POSE_SIZE>(idx));
      }

      // compute stepsize
      Scalar step_norminf = inc.array().abs().maxCoeff();

      // this is VO not VIO
//      BASALT_ASSERT(frame_states.empty());

      // compute error update applying increment
      Scalar after_update_marg_prior_error = 0;
      Scalar after_update_vision_error = 0;

      {
        Timer t;
        computeError(after_update_vision_error);
        computeMargPriorError(marg_vo_data, after_update_marg_prior_error);
        stats.add("computerError2", t.reset()).format("ms");
      }

      Scalar after_error_total =
          after_update_vision_error + after_update_marg_prior_error;

      // check cost decrease compared to quadratic model cost
      Scalar f_diff;
      bool step_is_valid = false;
      bool step_is_successful = false;
      Scalar relative_decrease = 0;
      {
        // compute actual cost decrease
        f_diff = error_total - after_error_total;

        relative_decrease = f_diff / l_diff;

        if (config.vio_debug) {
          std::cout << "\t[EVAL] error: {:.4e}, f_diff {:.4e} l_diff {:.4e} "
                       "step_quality {:.2e} step_size {:.2e}\n"_format(
              after_error_total, f_diff, l_diff, relative_decrease,
              step_norminf);
        }

        // TODO: consider to remove assert. For now we want to test if we even
        // run into the l_diff <= 0 case ever in practice
        // BASALT_ASSERT_STREAM(l_diff > 0, "l_diff " << l_diff);

        // l_diff <= 0 is a theoretical possibility if the model cost change is
        // tiny and becomes numerically negative (non-positive). It might not
        // occur since our linear systems are not that big (compared to large
        // scale BA for example) and we also abort optimization quite early and
        // usually don't have large damping (== tiny step size).
        step_is_valid = l_diff > 0;
        step_is_successful = step_is_valid && relative_decrease > 0;
      }

      double iteration_time = timer_iteration.elapsed();
      double cumulative_time = timer_total.elapsed();

      stats.add("iteration", iteration_time).format("ms");
      {
        basalt::MemoryInfo mi;
        if (get_memory_info(mi)) {
          stats.add("resident_memory", mi.resident_memory);
          stats.add("resident_memory_peak", mi.resident_memory_peak);
        }
      }

      if (step_is_successful) {
        BASALT_ASSERT(step_is_valid);

        if (config.vio_debug) {
          //          std::cout << "\t[ACCEPTED] lambda:" << lambda
          //                    << " Error: " << after_error_total << std::endl;

          std::cout << "\t[ACCEPTED] error: {:.4e}, lambda: {:.1e}, it_time: "
                       "{:.3f}s, total_time: {:.3f}s\n"
                       ""_format(after_error_total, lambda, iteration_time,
                                 cumulative_time);
        }

        lambda *= std::max<Scalar>(
            Scalar(1.0) / 3,
            1 - std::pow<Scalar>(2 * relative_decrease - 1, 3));
        lambda = std::max(min_lambda, lambda);

        lambda_vee = initial_vee;

        it++;

        // check function and parameter tolerance
        if ((f_diff > 0 && f_diff < Scalar(1e-6)) ||
            step_norminf < Scalar(1e-4)) {
          converged = true;
          terminated = true;
        }

        // stop inner lm loop
        break;
      } else {
        std::string reason = step_is_valid ? "REJECTED" : "INVALID";

        if (config.vio_debug) {
          //          std::cout << "\t[REJECTED] lambda:" << lambda
          //                    << " Error: " << after_error_total << std::endl;

          std::cout << "\t[{}] error: {}, lambda: {:.1e}, it_time:"
                       "{:.3f}s, total_time: {:.3f}s\n"
                       ""_format(reason, after_error_total, lambda,
                                 iteration_time, cumulative_time);
        }

        lambda = lambda_vee * lambda;
        lambda_vee *= vee_factor;

        //        lambda = std::max(min_lambda, lambda);
        //        lambda = std::min(max_lambda, lambda);

        restore();
        it++;
        it_rejected++;

        if (lambda > max_lambda) {
          terminated = true;
          message =
              "Solver did not converge and reached maximum damping lambda";
        }
      }
    }
  }

  stats.add("optimize", timer_total.elapsed()).format("ms");
  stats.add("num_it", it).format("count");
  stats.add("num_it_rejected", it_rejected).format("count");

  // TODO: call filterOutliers at least once (also for CG version)

  stats_all_.merge_all(stats);
  stats_sums_.merge_sums(stats);

  if (config.vio_debug) {
    if (!converged) {
      if (terminated) {
        std::cout << "Solver terminated early after {} iterations: {}"_format(
            it, message);
      } else {
        std::cout
            << "Solver did not converge after maximum number of {} iterations"_format(
                it);
      }
    }

    stats.print();

    std::cout << "=================================" << std::endl;
  }
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::initialize_imu(Eigen::aligned_map<int64_t, Vec3>& vel, Vec3 &bg, Vec3 &ba) {

  /// parameters
  bg = Vec3::Zero();
  ba = Vec3::Zero();
  Eigen::Matrix3<Scalar> Rwg;  /// imu to world (2 Dof)
  Eigen::Vector3<Scalar> dirG;
  dirG.setZero();

  auto it_meas = imu_meas.begin();
  auto it_frame1 = frame_poses.begin();
  auto it_frame2 = frame_poses.begin();
  it_frame2++;
  for (; it_meas != imu_meas.end(); ++it_meas, ++it_frame1, ++it_frame2) {
    dirG -= it_frame2->second.getPose().so3().matrix() * it_meas->second.getDeltaState().vel_w_i;
    vel[it_frame2->first] = (it_frame2->second.getPose().translation() -
                             it_frame1->second.getPose().translation()) / (it_meas->second.get_dt_ns() * 1.0e-9);
//    vel[it_frame2->first] = Vec3::Zero();

    std::cout << "imu: " << it_meas->first << " size: " << it_meas->second.get_imu_meas().size() << " dt: " << it_meas->second.get_dt_ns() * 1.0e-9 << std::endl;
  }
  auto it_temp1 = frame_poses.begin();
  auto it_temp2 = vel.begin();
  vel[it_temp1->first] = it_temp2->second;

  dirG = dirG / dirG.norm();
  Vec3 gI(0.0, 0.0, -1.0);
  Vec3 v = gI.cross(dirG);
  const float nv = v.norm();
  const float cosg = gI.dot(dirG);
  const float ang = std::acos(cosg);
  Vec3 vzg = v * ang / nv;
  Rwg = Sophus::SO3<Scalar>::exp(vzg).matrix();
  Rwg = Eigen::Matrix3<Scalar>::Identity();

  std::cout << "vzg: " << std::endl << Sophus::SO3<Scalar>::exp(vzg).matrix() << std::endl;

  std::cout << "T_w_y_init: " << std::endl << T_w_y_init.so3().matrix() << std::endl;

//  Rwg = T_w_y_init.so3().matrix().inverse();

////  Rwg = Eigen::Matrix<Scalar, 3, 3>::Identity(); //test
  std::cout << "Rwg: " << Rwg << std::endl;
  std::cout << "vel size: " << vel.size() << std::endl;  /// size same with frame_poses
  for (auto it = vel.begin(); it != vel.end(); ++it) {
    std::cout << it->first << " " << it->second << std::endl;
  }

  /// Inertial-only Optimization
  const int num_iter = 4;
  const int num_para = 6 + 2 + 3 * vel.size();  /// Rwg(2dof) + bg + ba + vels
//  const int num_para = 6 + 3 * vel.size();  /// bg + ba + vels

  const Scalar prior_bg = 1e-10;
  const Scalar prior_ba = 1e-8;

  std::cout << "prior_bg: " << prior_bg << " prior_ba: " << prior_ba << std::endl;

  auto G_I = basalt::constants::g.cast<Scalar>();

  for (int iter = 0; iter < num_iter; iter++) {

    MatX H;
    VecX b;
    H.setZero(num_para, num_para);
    b.setZero(num_para);

    ///////////////////////// imu preintegration  /////////////////////////////

    int k = 0;
    auto it_pose0 = frame_poses.begin();
    auto it_pose1 = frame_poses.begin();
    it_pose1++;

    auto it_vel0 = vel.begin();
    auto it_vel1 = vel.begin();
    it_vel1++;
    for (auto it = imu_meas.begin(); it != imu_meas.end(); ++it, ++it_pose0, ++it_pose1, ++it_vel0, ++it_vel1) {

      BASALT_ASSERT(it_pose0->first == it_vel0->first);
      BASALT_ASSERT(it_pose1->first == it_vel1->first);
      BASALT_ASSERT(it->first == it_pose0->first);

      /// x (14 * 1): bg, ba, Rwg, v0, v1
      MatX H1;
      VecX b1;
      H1.setZero(14, 14);
      b1.setZero(14);

      MatX Jp1;
      Jp1.setZero(9, 14);
      VecX r1;
      r1.setZero(9);

      auto imu_m = it->second;

      Scalar dt = imu_m.get_dt_ns() * Scalar(1e-9);

      VecN bg_diff;
      VecN ba_diff;
      bg_diff = imu_m.get_d_state_d_bg() * (bg - imu_m.get_bias_gyro_lin());
      ba_diff = imu_m.get_d_state_d_ba() * (ba - imu_m.get_bias_accel_lin());

      BASALT_ASSERT(ba_diff.template segment<3>(3).isApproxToConstant(0));

      Vec3 g_G = Rwg * G_I;
      Eigen::Matrix<Scalar, 3, 3> R0_inv = it_pose0->second.getPose().so3().inverse().matrix();
      Vec3 tmp = R0_inv * (it_pose1->second.getPose().translation() - it_pose0->second.getPose().translation() -
                           it_vel0->second * dt - Scalar(0.5) * g_G * dt * dt);

      ///////////// residual //////////////
      Eigen::Matrix<Scalar, 9, 1> res;
      res.setZero();
      ///dp
      res.template segment<3>(0) =
          tmp - (imu_m.getDeltaState().T_w_i.translation() +
                 bg_diff.template segment<3>(0) + ba_diff.template segment<3>(0));
      ///dr
      res.template segment<3>(3) =
          (Sophus::SO3<Scalar>::exp(bg_diff.template segment<3>(3)) * imu_m.getDeltaState().T_w_i.so3() *
           it_pose1->second.getPose().so3().inverse() * it_pose0->second.getPose().so3())
              .log();
      Vec3 tmp2 = R0_inv * (it_vel1->second - it_vel0->second - g_G * dt);
      ///dv
      res.template segment<3>(6) =
          tmp2 - (imu_m.getDeltaState().vel_w_i + bg_diff.template segment<3>(6) +
                  ba_diff.template segment<3>(6));

      r1 = imu_m.get_sqrt_cov_inv() * res;

      std::cout << "dt: " << dt << " res- " << k << ": " << std::endl << res << std::endl;

      //////////// Jacobian ///////////
      Eigen::Matrix<Scalar, 9, 14> Jaco;
      Jaco.setZero();

      Eigen::Matrix<Scalar, 3, 2> d_Rwg_a12;
      d_Rwg_a12 << Scalar(1), Scalar(0),
                   Scalar(0), Scalar(1),
                   Scalar(0), Scalar(0);
      /// Jacobians of dp wrt x (bg, ba, Rwg, v0, v1)
      Jaco.template block<3, 3>(0, 0) = -imu_m.get_d_state_d_bg().template block<3, 3>(0, 0); /// J _dp_bg
      Jaco.template block<3, 3>(0, 3) = -imu_m.get_d_state_d_ba().template block<3, 3>(0, 0); /// J _dp_ba
      Jaco.template block<3, 2>(0, 6) = -0.5 * R0_inv * dt * dt * (-Sophus::SO3<Scalar>::hat(Rwg * G_I) * d_Rwg_a12); /// J_dp_Rwg
      Jaco.template block<3, 3>(0, 8) = -R0_inv * dt; /// J _dp_v0
      Jaco.template block<3, 3>(0, 11) = Eigen::Matrix<Scalar, 3, 3>::Zero(); /// J _dp_v1

      /// Jacobians of dr wrt x (bg, ba, Rwg, v0, v1)
      Eigen::Matrix<Scalar, 3, 3> inv_Jl;
      Sophus::leftJacobianInvSO3(res.template segment<3>(3), inv_Jl);
      Jaco.template block<3, 3>(3, 0) = inv_Jl * imu_m.get_d_state_d_bg().template block<3, 3>(3, 0); /// J _dr_bg
      Jaco.template block<3, 3>(3, 3) = Eigen::Matrix<Scalar, 3, 3>::Zero(); /// J _dr_ba
      Jaco.template block<3, 2>(3, 6) = Eigen::Matrix<Scalar, 3, 2>::Zero(); /// J_dr_Rwg
      Jaco.template block<3, 3>(3, 8) = Eigen::Matrix<Scalar, 3, 3>::Zero(); /// J _dr_v0
      Jaco.template block<3, 3>(3, 11) = Eigen::Matrix<Scalar, 3, 3>::Zero(); /// J _dr_v1

      /// Jacobians of dv wrt x (bg, ba, Rwg, v0, v1)
      Jaco.template block<3, 3>(6, 0) = -imu_m.get_d_state_d_bg().template block<3, 3>(6, 0); /// J _dv_bg
      Jaco.template block<3, 3>(6, 3) = -imu_m.get_d_state_d_ba().template block<3, 3>(6, 0); /// J _dv_ba
      Jaco.template block<3, 2>(6, 6) = -R0_inv * dt * (-Sophus::SO3<Scalar>::hat(Rwg * G_I) * d_Rwg_a12); /// J_dv_Rwg
      Jaco.template block<3, 3>(6, 8) = -R0_inv; /// J _dv_v0
      Jaco.template block<3, 3>(6, 11) = R0_inv; /// J _dv_v1

//      std::cout << "Jaco: " << std::endl << Jaco << std::endl;
      Jp1 = imu_m.get_sqrt_cov_inv() * Jaco;
      //////////
      H1 = Jp1.transpose() * Jp1;
      b1 = Jp1.transpose() * r1;

      H.template block<8, 8>(0, 0) += H1.template block<8, 8>(0, 0);
      H.template block<6, 6>(8 + 3 * k, 8 + 3 * k) += H1.template block<6, 6>(8, 8);
      H.template block<8, 6>(0, 8 + 3 * k) += H1.template block<8, 6>(0, 8);
      H.template block<6, 8>(8 + 3 * k, 0) += H1.template block<6, 8>(8, 0);

      b.template segment<8>(0) += b1.template segment<8>(0);
      b.template segment<6>(8 + 3 * k) += b1.template segment<6>(8);

//      std::cout << "H1: " << std::endl << H1 << std::endl;
//      std::cout << "b1: " << std::endl << b1 << std::endl;

      k++;
    }

    ////////////////////////// IMU bg bias prior /////////////////////////////////////
    {
      MatX Jp2;
      VecX r2;
      Jp2.setZero(3, 3);
      r2.setZero(3);

      Eigen::Vector3<Scalar> res_bg;
      res_bg = bg - Eigen::Vector3<Scalar>::Zero();
      std::cout << "res_bg_ba: " << res_bg << std::endl;

      /// d_bg (prior)
      r2 = Scalar(1.0) / sqrt(prior_bg) * res_bg;
      Jp2 = Scalar(1.0) / sqrt(prior_bg) * Eigen::Matrix<Scalar, 3, 3>::Identity();  /// wrt bg

      MatX H2;
      VecX b2;
      H2.setZero(3, 3);
      b2.setZero(3);

      H2 = Jp2.transpose() * Jp2;
      b2 = Jp2.transpose() * r2;

      H.template block<3, 3>(0, 0) += H2;
      b.template segment<3>(0) += b2;

//      std::cout << "H2: " << std::endl << H2 << std::endl;
//      std::cout << "b2: " << std::endl << b2 << std::endl;
    }

    ////////////////////////// IMU ba bias prior /////////////////////////////////////
    {
      MatX Jp3;
      VecX r3;
      Jp3.setZero(3, 3);
      r3.setZero(3);

      Eigen::Vector3<Scalar> res_ba;
      res_ba = ba - Eigen::Vector3<Scalar>::Zero();
      std::cout << "res_ba: " << res_ba << std::endl;

      /// d_ba (prior)
      r3 = Scalar(1.0) / sqrt(prior_ba) * res_ba;
      Jp3 = Scalar(1.0) / sqrt(prior_ba) * Eigen::Matrix<Scalar, 3, 3>::Identity();  /// wrt ba

      MatX H3;
      VecX b3;
      H3.setZero(3, 3);
      b3.setZero(3);

      H3 = Jp3.transpose() * Jp3;
      b3 = Jp3.transpose() * r3;

      H.template block<3, 3>(3, 3) += H3;
      b.template segment<3>(3) += b3;

//      std::cout << "H3: " << std::endl << H3 << std::endl;
//      std::cout << "b3: " << std::endl << b3 << std::endl;
    }
    ///////////////////////////////////////////

    /// Add small damping for GN
    constexpr Scalar lambda0 = 1e-6;
    VecX diag = H.diagonal();
    diag *= lambda0;
    H.diagonal().array() += diag.array().max(lambda0);

    /// LDLT to compute inc
    VecX inc = -H.ldlt().solve(b);

    /// apply inc
    bg += inc.template segment<3>(0);
    ba += inc.template segment<3>(3);
    Rwg = Sophus::SO3<Scalar>::exp(Eigen::Vector3<Scalar>(inc[6], inc[7], 0.0)).matrix() * Rwg;

    int kk = 0;
    for (auto it = vel.begin(); it != vel.end(); ++it) {
      it->second += inc.template segment<3>(8 + 3 * kk);
      kk++;
    }

    std::cout << inc.rows() << " inc: " << inc << std::endl;
    std::cout << "----------- iter: " << iter << " -----------" << std::endl;
  }

  std::cout << "Rwg: " << Rwg << std::endl;
  std::cout << "T_w_y_init.so3().matrix().inverse(): " << T_w_y_init.so3().matrix().inverse() << std::endl;
  std::cout << "bg: " << std::endl << bg << " ba: " << std::endl << ba << std::endl;
//  bg = Vec3::Zero();
//  ba = Vec3::Zero();

  /// apply rotation
  SE3 Tyw(Rwg.transpose(), Vec3::Zero());  /// world(the first camera's) to y (inertial)
  for (auto it = frame_poses.begin(); it != frame_poses.end(); ++it) {
    it->second.setTransformToWorld(Tyw);
  }
  for (auto it = vel.begin(); it != vel.end(); ++it) {
    it->second = Tyw.so3().matrix() * it->second;

    std::cout << "vel: " << it->second << std::endl;
  }

  /// re-intergrate
  //TODO:
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::optimize_and_marg(
    const std::map<int64_t, int>& num_points_connected,
    const std::unordered_set<KeypointId>& lost_landmaks) {

  if (!is_system_initialized) {
    /// convert frame_states to frame_poses
    for (auto it = frame_states.begin(); it != frame_states.end(); ++it) {
      PoseStateWithLin<Scalar> pose(it->second);
      frame_poses[it->first] = pose;
    }
    for (auto it = frame_states.begin(); it != frame_states.end();) {
      frame_states.erase(it++);
    }

    std::cout << "frame_poses size: " << frame_poses.size() << std::endl;
    std::cout << "frame_states size: " << frame_states.size() << std::endl;
    std::cout << "kf_ids: " << kf_ids.size() << std::endl;
    std::cout << "imu_meas: " << imu_meas.size() << std::endl;

    optimize_vo();

    size_t min_kf = max_kfs + max_states - 1;
    if (frame_poses.size() >= min_kf) {
      //TODO: Inertial-Only Optimization
      Timer t_init_imu;

      Eigen::aligned_map<int64_t, Vec3> vel;
      Vec3 bg = Vec3::Zero();
      Vec3 ba = Vec3::Zero();

      initialize_imu(vel, bg, ba);
      std::cout << "initialize_imu time: " << t_init_imu.elapsed() * 1000.0 << " ms" << std::endl;

      /// convert frame_poses to frame_states
      for (auto it = frame_poses.begin(); it != frame_poses.end(); ++it) {
        auto frame_pose = frame_poses[it->first];
        PoseVelBiasStateWithLin<Scalar> state(frame_pose.getT_ns(), frame_pose.getPose(), vel[it->first], bg, ba, false);
        frame_states[it->first] = state;
      }

      for (auto it = frame_poses.begin(); it != frame_poses.end();) {
        frame_poses.erase(it++);
      }

      /// set the prior
      {
        auto it_end = frame_states.end();
        it_end--;

//        auto it_end = frame_states.begin();
//        it_end++;

        AbsOrderMap aom;

        int num_frame = 0;
        for (auto it = frame_states.begin(); it != frame_states.end(); ++it) {
          if (it->first == it_end->first) {
            break;
          }
          it->second.setLinTrue();

          aom.abs_order_map[it->first] = std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);
          aom.total_size += POSE_VEL_BIAS_SIZE;
          aom.items++;
          num_frame++;
        }

        std::cout << "num_frame: " << num_frame << std::endl;
        const double init_pose_weight = 1.0e8;
        const double init_vel_weight = 1.0e2;
        const double init_ba_weight = 1.0e8;
        const double init_bg_weight = 1.0e8;

        marg_data.order = aom;
        marg_data.is_sqrt = config.vio_sqrt_marg;
        marg_data.H.setZero(POSE_VEL_BIAS_SIZE * num_frame, POSE_VEL_BIAS_SIZE * num_frame);
        marg_data.b.setZero(POSE_VEL_BIAS_SIZE * num_frame);
        for (int k = 0; k < num_frame; k++) {
          if (marg_data.is_sqrt) {
            // prior on position
            marg_data.H.diagonal().template segment<3>(0 + k * POSE_VEL_BIAS_SIZE).array() = std::sqrt(Scalar(init_pose_weight));
            // prior on yaw
//            marg_data.H(5 + k * POSE_VEL_BIAS_SIZE, 5 + k * POSE_VEL_BIAS_SIZE) = std::sqrt(Scalar(init_pose_weight));
            marg_data.H.diagonal().template segment<3>(3 + k * POSE_VEL_BIAS_SIZE).array() = std::sqrt(Scalar(init_pose_weight));
            marg_data.H.diagonal().template segment<3>(6 + k * POSE_VEL_BIAS_SIZE).array() = std::sqrt(Scalar(init_vel_weight));

            // small prior to avoid jumps in bias
            marg_data.H.diagonal().template segment<3>(9 + k * POSE_VEL_BIAS_SIZE).array() = std::sqrt(Scalar(init_ba_weight));
            marg_data.H.diagonal().template segment<3>(12 + k * POSE_VEL_BIAS_SIZE).array() = std::sqrt(Scalar(init_bg_weight));
          } else {
            // prior on position
            marg_data.H.diagonal().template segment<3>(0 + k * POSE_VEL_BIAS_SIZE).array() = Scalar(init_pose_weight);
            // prior on yaw
//            marg_data.H(5 + k * POSE_VEL_BIAS_SIZE, 5 + k * POSE_VEL_BIAS_SIZE) = Scalar(init_pose_weight);
            marg_data.H.diagonal().template segment<3>(3 + k * POSE_VEL_BIAS_SIZE).array() = Scalar(init_pose_weight);
            marg_data.H.diagonal().template segment<3>(6 + k * POSE_VEL_BIAS_SIZE).array() = Scalar(init_vel_weight);

            // small prior to avoid jumps in bias
            marg_data.H.diagonal().template segment<3>(9 + k * POSE_VEL_BIAS_SIZE).array() = Scalar(init_ba_weight);
            marg_data.H.diagonal().template segment<3>(12 + k * POSE_VEL_BIAS_SIZE).array() = Scalar(init_bg_weight);
          }
        }
//        std::cout << "marg_H: " << marg_data.H << std::endl;
//        std::cout << "marg_vo_H: " << marg_vo_data.H << std::endl;
      }
      /// Visual-Inertial Optimization
      std::cout << "before optimize" << std::endl;
      for (auto it = frame_states.begin(); it != frame_states.end(); ++it) {
        auto sr = it->second.getState().T_w_i.so3().matrix();
        auto sp = it->second.getState().T_w_i.translation();
        auto svel = it->second.getState().vel_w_i;
        auto sbg = it->second.getState().bias_gyro;
        auto sba = it->second.getState().bias_accel;
        std::cout << "t: " << it->first << std::endl;
        std::cout << sr << std::endl;
        std::cout << sp.x() << "," << sp.y() << "," << sp.z() <<
            "|" << svel.x() << "," << svel.y() << "," << svel.z() <<
            "|" << sbg.x() << "," << sbg.y() << "," << sbg.z() <<
            "|" << sba.x() << "," << sba.y() << "," << sba.z() << std::endl;
      }
      optimize();
      std::cout << "--------after optimize--------" << std::endl;
      for (auto it = frame_states.begin(); it != frame_states.end(); ++it) {
        auto sr = it->second.getState().T_w_i.so3().matrix();
        auto sp = it->second.getState().T_w_i.translation();
        auto svel = it->second.getState().vel_w_i;
        auto sbg = it->second.getState().bias_gyro;
        auto sba = it->second.getState().bias_accel;
        std::cout << "t: " << it->first << std::endl;
        std::cout << sr << std::endl;
        std::cout << sp.x() << "," << sp.y() << "," << sp.z() <<
                  "|" << svel.x() << "," << svel.y() << "," << svel.z() <<
                  "|" << sbg.x() << "," << sbg.y() << "," << sbg.z() <<
                  "|" << sba.x() << "," << sba.y() << "," << sba.z() << std::endl;
      }
      std::cout << "frame_poses size: " << frame_poses.size() << std::endl;
      std::cout << "frame_states size: " << frame_states.size() << std::endl;
      std::cout << "marginalize_imu" << std::endl;

      /// Marg the vel and bias
      marginalize_imu();

      is_system_initialized = true;
    } else {
      marginalize_vo(num_points_connected, lost_landmaks);
      std::cout << "marginalize_vo" << std::endl;
      std::cout << "frame_poses size: " << frame_poses.size() << std::endl;
      std::cout << "kf_ids: " << kf_ids.size() << std::endl;

      /// convert frame_poses to frame_states
      for (auto it = frame_poses.begin(); it != frame_poses.end(); ++it) {
        PoseVelBiasStateWithLin<Scalar> state(frame_poses[it->first], Vec3::Zero(), Vec3::Zero(), Vec3::Zero());
        frame_states[it->first] = state;
      }

      for (auto it = frame_poses.begin(); it != frame_poses.end();) {
        frame_poses.erase(it++);
      }
    }
  } else {
    std::cout << "optimize_and_marg" << std::endl;
    optimize();
    std::cout << "optimize" << std::endl;
    marginalize(num_points_connected, lost_landmaks);
    std::cout << "marginalize" << std::endl;
  }

  std::cout << "last_t: " << last_state_t_ns << std::endl;
  std::cout << "frame_poses size: " << frame_poses.size() << std::endl;
  std::cout << "frame_states size: " << frame_states.size() << std::endl;
  std::cout << "kf size: " << kf_ids.size() << std::endl;
  std::cout << "imu_meas size: " << imu_meas.size() << std::endl;
  std::cout << "prev_opt_flow_res size: " << prev_opt_flow_res.size() << std::endl;
  std::cout << "is_system_initialized: " << is_system_initialized << std::endl;

  std::cout << "================================" << std::endl;
}

template <class Scalar_>
void SqrtKeypointVioInitEstimator<Scalar_>::debug_finalize() {
  std::cout << "=== stats all ===\n";
  stats_all_.print();
  std::cout << "=== stats sums ===\n";
  stats_sums_.print();

  // save files
  stats_all_.save_json("stats_all.json");
  stats_sums_.save_json("stats_sums.json");
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class SqrtKeypointVioInitEstimator<double>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class SqrtKeypointVioInitEstimator<float>;
#endif

}  // namespace basalt
