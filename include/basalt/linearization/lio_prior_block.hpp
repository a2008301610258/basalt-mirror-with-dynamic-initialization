//
// Created by zhihui on 12/15/22.
//

#pragma once

#include <basalt/li_estimator/states_group.h>
#include <basalt/optimization/accumulator.h>
#include <basalt/utils/imu_types.h>

namespace basalt {

template <class Scalar_>
class LioPriorBlock {
 public:
  using Scalar = Scalar_;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  LioPriorBlock(const StatesGroup& meas, const AbsOrderMap& aom)
      : meas(meas), aom(aom){
    Jp.resize(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
    r.resize(POSE_VEL_BIAS_SIZE);
  }

  Scalar linearizeLioPrior(const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin<Scalar>>& frame_states) {
    Jp.setZero();
    r.setZero();

    Scalar w_s = 0.1;

    const int64_t tm_ns = meas.last_update_time_ns;

    PoseVelBiasStateWithLin<Scalar> state = frame_states.at(tm_ns);

    Eigen::Matrix<Scalar, 15, 15> cov_mat_temp = meas.cov.cast<Scalar>();
    /// cov [r,p,v,bg,ba] --> [p,r,v,bg,ba]
    cov_mat_temp.template block<3, 15>(0, 0).swap(cov_mat_temp.template block<3, 15>(3, 0));
    cov_mat_temp.template block<15, 3>(0, 0).swap(cov_mat_temp.template block<15, 3>(0, 3));

//    std::cout << "lio prior cov(p r v bg ba): " << std::endl;
//    std::cout << cov_mat_temp << std::endl;
//    std::cout << "-------------" << std::endl;
//    std::cout << cov_mat_temp * 0.01 << std::endl;

    Eigen::Matrix<Scalar, 15, 15> sqrt_cov_inv;
    sqrt_cov_inv.setIdentity();
    auto ldlt = cov_mat_temp.ldlt();

    sqrt_cov_inv = ldlt.transpositionsP() * sqrt_cov_inv;
    ldlt.matrixL().solveInPlace(sqrt_cov_inv);

    Eigen::Matrix<Scalar, 15, 1> D_inv_sqrt;
    for (size_t i = 0; i < 15; i++) {
      if (ldlt.vectorD()[i] < std::numeric_limits<Scalar>::min()) {
        D_inv_sqrt[i] = 0;
      } else {
        D_inv_sqrt[i] = Scalar(1.0) / sqrt(ldlt.vectorD()[i]);
      }
    }
    sqrt_cov_inv = D_inv_sqrt.asDiagonal() * sqrt_cov_inv;

    ///
    VecX res;
    res.setZero(POSE_VEL_BIAS_SIZE);

    Eigen::Matrix3<Scalar> state_rot = state.getStateLin().T_w_i.so3().matrix();
    Eigen::Vector3<Scalar> res_rot = Eigen::Vector3<Scalar>::Zero();

    res.template segment<3>(0) = (state.getStateLin().T_w_i.translation() - meas.pos_end.cast<Scalar>());  /// p

    res_rot = Sophus::SO3<Scalar>(state.getStateLin().T_w_i.so3().matrix() * meas.rot_end.transpose().cast<Scalar>()).log();
    res.template segment<3>(3) = res_rot;  /// r
    res.template segment<3>(6) = (state.getStateLin().vel_w_i - meas.vel_end.cast<Scalar>()); /// v
    res.template segment<3>(9) = (state.getStateLin().bias_gyro - meas.bias_g.cast<Scalar>());  /// bg
    res.template segment<3>(9) = (state.getStateLin().bias_accel - meas.bias_a.cast<Scalar>());  /// ba

    if (state.isLinearized()) {
      state_rot = state.getState().T_w_i.so3().matrix();
      res.template segment<3>(0) = (state.getState().T_w_i.translation() - meas.pos_end.cast<Scalar>());

      res_rot = Sophus::SO3<Scalar>(state.getState().T_w_i.so3().matrix() * meas.rot_end.transpose().cast<Scalar>()).log();
      res.template segment<3>(3) = res_rot;
      res.template segment<3>(6) = (state.getState().vel_w_i - meas.vel_end.cast<Scalar>());
      res.template segment<3>(9) = (state.getState().bias_gyro - meas.bias_g.cast<Scalar>());
      res.template segment<3>(9) = (state.getState().bias_accel - meas.bias_a.cast<Scalar>());
    }

    r = sqrt_cov_inv * w_s * res;

    /////

    MatX Jaco;
    Jaco.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);

    Eigen::Matrix3<Scalar> Jr_inv;
    Sophus::rightJacobianInvSO3(res_rot, Jr_inv);

    Jaco.template block<3, 3>(0, 0) = Eigen::Matrix3<Scalar>::Identity();
    Jaco.template block<3, 3>(3, 3) = Jr_inv * (state_rot * meas.rot_end.transpose().cast<Scalar>()).transpose();
    Jaco.template block<3, 3>(6, 6) = Eigen::Matrix3<Scalar>::Identity();
    Jaco.template block<3, 3>(9, 9) = Eigen::Matrix3<Scalar>::Identity();
    Jaco.template block<3, 3>(12, 12) = Eigen::Matrix3<Scalar>::Identity();

    Jp = sqrt_cov_inv * w_s * Jaco;

    Scalar error = Scalar(0.5) * (sqrt_cov_inv * w_s * res).squaredNorm();

    return error;
  }

  void add_dense_Q2Jp_Q2r(MatX& Q2Jp, VecX& Q2r, size_t row_start_idx) const {
    const int64_t tm_ns = meas.last_update_time_ns;
    const size_t idx = aom.abs_order_map.at(tm_ns).first;

    std::cout << "row_start_idx: " << row_start_idx << ", idx: " << idx << std::endl;

    Q2Jp.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(row_start_idx,
                                                                idx) +=
      Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>();

    std::cout << "Q2Jp done" << std::endl;

    Q2r.template segment<POSE_VEL_BIAS_SIZE>(row_start_idx) += r;
    std::cout << "Q2r done" << std::endl;
  }

  void add_dense_H_b(DenseAccumulator<Scalar>& accum) const {
    const int64_t tm_ns = meas.last_update_time_ns;
    const size_t idx = aom.abs_order_map.at(tm_ns).first;

    const MatX H = Jp.transpose() * Jp;
    const VecX b = Jp.transpose() * r;

    accum.template addH<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(
        idx, idx,
        H.template block<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>(0, 0));

    accum.template addB<POSE_VEL_BIAS_SIZE>(
        idx, b.template segment<POSE_VEL_BIAS_SIZE>(0));

  }

  void scaleJp_cols(const VecX& jacobian_scaling) {
    int64_t tm_ns = meas.last_update_time_ns;
    const size_t idx = aom.abs_order_map.at(tm_ns).first;

    Jp.template topLeftCorner<POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE>() *=
        jacobian_scaling.template segment<POSE_VEL_BIAS_SIZE>(idx)
            .asDiagonal();
  }

  void backSubstitute(const VecX& pose_inc, Scalar& l_diff) {
    int64_t tm_ns = meas.last_update_time_ns;
    const size_t idx = aom.abs_order_map.at(tm_ns).first;

    VecX pose_inc_reduced(POSE_VEL_BIAS_SIZE);
    pose_inc_reduced.template head<POSE_VEL_BIAS_SIZE>() =
        pose_inc.template segment<POSE_VEL_BIAS_SIZE>(idx);

    VecX Jinc = Jp * pose_inc_reduced;
    l_diff -= Jinc.transpose() * (Scalar(0.5) * Jinc + r);
  }

 protected:
  MatX Jp;
  VecX r;

  const StatesGroup meas;
  const AbsOrderMap& aom;
};
}