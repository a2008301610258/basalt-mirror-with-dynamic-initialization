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
#include <basalt/li_estimator/r3live.hpp>
//#include "tools/tools_timer.hpp"

using namespace basalt;

void R3LIVE::set_initial_state_cov( StatesGroup &state )
{
    // Set cov
//    scope_color( ANSI_COLOR_RED_BOLD );
    state.cov = state.cov.setIdentity() * INIT_COV;
    // state.cov.block(18, 18, 6 , 6 ) = state.cov.block(18, 18, 6 , 6 ) .setIdentity() * 0.1;
    // state.cov.block(24, 24, 5 , 5 ) = state.cov.block(24, 24, 5 , 5 ).setIdentity() * 0.001;
    state.cov.block( 0, 0, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5;   // R
    state.cov.block( 3, 3, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5;   // T
    state.cov.block( 6, 6, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5;   // vel
    state.cov.block( 9, 9, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-3;   // bias_g
    state.cov.block( 12, 12, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-1; // bias_a
//    state.cov.block( 15, 15, 3, 3 ) = Eigen::Matrix3d::Identity() * 1e-5; // Gravity
//    state.cov( 24, 24 ) = 0.00001;
//    state.cov.block( 18, 18, 6, 6 ) = state.cov.block( 18, 18, 6, 6 ).setIdentity() *  1e-3; // Extrinsic between camera and IMU.
//    state.cov.block( 25, 25, 4, 4 ) = state.cov.block( 25, 25, 4, 4 ).setIdentity() *  1e-3; // Camera intrinsic.
}

void R3LIVE::imu_cbk( const sensor_msgs::Imu::ConstPtr &msg_in )
{
    sensor_msgs::Imu::Ptr msg( new sensor_msgs::Imu( *msg_in ) );
    double                timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();
    if ( timestamp < last_timestamp_imu )
    {
        ROS_ERROR( "imu loop back, clear buffer" );
        imu_buffer_lio.clear();
        flg_reset = true;
    }
    last_timestamp_imu = timestamp;

    imu_buffer_lio.push_back( msg );
    // std::cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer_lio.size()<<std::endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void printf_field_name( sensor_msgs::PointCloud2::ConstPtr &msg )
{
    std::cout << "Input pointcloud field names: [" << msg->fields.size() << "]: ";
    for ( size_t i = 0; i < msg->fields.size(); i++ )
    {
      std::cout << msg->fields[ i ].name << ", ";
    }
    std::cout << std::endl;
}


bool R3LIVE::get_pointcloud_data_from_ros_message( sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud< pcl::PointXYZINormal > &pcl_pc )
{
    pcl::PointCloud< pcl::PointXYZI > res_pc;
    if ( msg->fields.size() < 3 )
    {
        std::cout << "Get pointcloud data from ros messages fail!!!" << std::endl;
        printf_field_name( msg );
        return false;
    }
    else
    {
        if ( ( msg->fields.size() == 8 ) && ( msg->fields[ 3 ].name == "intensity" ) &&
             ( msg->fields[ 4 ].name == "normal_x" ) ) // Input message type is pcl::PointXYZINormal
        {
            pcl::fromROSMsg( *msg, pcl_pc );
            return true;
        }
        else if ( ( msg->fields.size() == 4 ) && ( msg->fields[ 3 ].name == "rgb" ) )
        {
            double maximum_range = 5;
//            get_ros_parameter< double >( m_ros_node_handle, "iros_range", maximum_range, 5 );
            pcl::PointCloud< pcl::PointXYZRGB > pcl_rgb_pc;
            pcl::fromROSMsg( *msg, pcl_rgb_pc );
            double lidar_point_time = msg->header.stamp.toSec();
            int    pt_count = 0;
            pcl_pc.resize( pcl_rgb_pc.points.size() );
            for ( int i = 0; i < pcl_rgb_pc.size(); i++ )
            {
                pcl::PointXYZINormal temp_pt;
                temp_pt.x = pcl_rgb_pc.points[ i ].x;
                temp_pt.y = pcl_rgb_pc.points[ i ].y;
                temp_pt.z = pcl_rgb_pc.points[ i ].z;
                double frame_dis = sqrt( temp_pt.x * temp_pt.x + temp_pt.y * temp_pt.y + temp_pt.z * temp_pt.z );
                if ( frame_dis > maximum_range )
                {
                    continue;
                }
                temp_pt.intensity = ( pcl_rgb_pc.points[ i ].r + pcl_rgb_pc.points[ i ].g + pcl_rgb_pc.points[ i ].b ) / 3.0;
                temp_pt.curvature = 0;
                pcl_pc.points[ pt_count ] = temp_pt;
                pt_count++;
            }
            pcl_pc.points.resize( pt_count );
            return true;
        }
        else // TODO, can add by yourself
        {
          std::cout << "Get pointcloud data from ros messages fail!!! ";
//            scope_color( ANSI_COLOR_RED_BOLD );
            printf_field_name( msg );
            return false;
        }
    }
}

bool R3LIVE::sync_packages( MeasureGroup &meas )
{
    if ( lidar_buffer.empty() || imu_buffer_lio.empty() )
    {
        return false;
    }

    /*** push lidar frame ***/
    if ( !lidar_pushed )
    {
        meas.lidar.reset( new PointCloudXYZINormal() );
        if ( get_pointcloud_data_from_ros_message( lidar_buffer.front(), *( meas.lidar ) ) == false )
        {
            return false;
        }
        // pcl::fromROSMsg(*(lidar_buffer.front()), *(meas.lidar));
        meas.lidar_beg_time = lidar_buffer.front()->header.stamp.toSec();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
        meas.lidar_end_time = lidar_end_time;
        // printf("Input LiDAR time = %.3f, %.3f\n", meas.lidar_beg_time, meas.lidar_end_time);
        // printf_line_mem_MB;
        lidar_pushed = true;
    }

    if ( last_timestamp_imu < lidar_end_time )
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer_lio.front()->header.stamp.toSec();
    meas.imu.clear();
    while ( ( !imu_buffer_lio.empty() ) && ( imu_time < lidar_end_time ) )
    {
        imu_time = imu_buffer_lio.front()->header.stamp.toSec();
        // if ( imu_time > lidar_end_time + 0.02 )
        //     break;
        if ( imu_time > lidar_end_time)
            break;
        meas.imu.push_back( imu_buffer_lio.front() );
        imu_buffer_lio.pop_front();
    }

    lidar_buffer.pop_front();
    lidar_pushed = false;
    if (meas.imu.empty()){
        std::cout << "Empty imu data" << std::endl;
        return false;
    } 
    // std::cout<<"[IMU Sycned]: "<<imu_time<<" "<<lidar_end_time<<std::endl;
    return true;
}

// project lidar frame to world
void R3LIVE::pointBodyToWorld( PointType const *const pi, PointType *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_lio_state.rot_end * ( Lidar_R_to_IMU * p_body + Lidar_offset_to_IMU ) + g_lio_state.pos_end );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;
}

void R3LIVE::RGBpointBodyToWorld( PointType const *const pi, pcl::PointXYZI *const po )
{
    Eigen::Vector3d p_body( pi->x, pi->y, pi->z );
    Eigen::Vector3d p_global( g_lio_state.rot_end * ( Lidar_R_to_IMU * p_body + Lidar_offset_to_IMU ) + g_lio_state.pos_end );

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - std::floor( intensity );

    int reflection_map = intensity * 10000;
}

int R3LIVE::get_cube_index( const int &i, const int &j, const int &k )
{
    return ( i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k );
}

void R3LIVE::lasermap_fov_segment()
{
    pointBodyToWorld( XAxisPoint_body, XAxisPoint_world );
    int centerCubeI = int( ( g_lio_state.pos_end( 0 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenWidth;
    int centerCubeJ = int( ( g_lio_state.pos_end( 1 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenHeight;
    int centerCubeK = int( ( g_lio_state.pos_end( 2 ) + 0.5 * cube_len ) / cube_len ) + laserCloudCenDepth;
    if ( g_lio_state.pos_end( 0 ) + 0.5 * cube_len < 0 )
        centerCubeI--;
    if ( g_lio_state.pos_end( 1 ) + 0.5 * cube_len < 0 )
        centerCubeJ--;
    if ( g_lio_state.pos_end( 2 ) + 0.5 * cube_len < 0 )
        centerCubeK--;
    bool last_inFOV_flag = 0;
    int  cube_index = 0;
    cub_needrm.clear();
    cub_needad.clear();

    while ( centerCubeI < FOV_RANGE + 1 )
    {
        for ( int j = 0; j < laserCloudHeight; j++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int i = laserCloudWidth - 1;

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i - 1, j, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i - 1, j, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeI++;
        laserCloudCenWidth++;
    }

    while ( centerCubeI >= laserCloudWidth - ( FOV_RANGE + 1 ) )
    {
        for ( int j = 0; j < laserCloudHeight; j++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int i = 0;

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i + 1, j, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i + 1, j, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    while ( centerCubeJ < ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int j = laserCloudHeight - 1;

                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j - 1, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j - 1, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while ( centerCubeJ >= laserCloudHeight - ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int k = 0; k < laserCloudDepth; k++ )
            {
                int                       j = 0;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j + 1, k ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j + 1, k ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while ( centerCubeK < ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int j = 0; j < laserCloudHeight; j++ )
            {
                int                       k = laserCloudDepth - 1;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j, k - 1 ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j, k - 1 ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while ( centerCubeK >= laserCloudDepth - ( FOV_RANGE + 1 ) )
    {
        for ( int i = 0; i < laserCloudWidth; i++ )
        {
            for ( int j = 0; j < laserCloudHeight; j++ )
            {
                int                       k = 0;
                PointCloudXYZINormal::Ptr laserCloudCubeSurfPointer = featsArray[ get_cube_index( i, j, k ) ];
                last_inFOV_flag = _last_inFOV[ cube_index ];

                for ( ; i >= 1; i-- )
                {
                    featsArray[ get_cube_index( i, j, k ) ] = featsArray[ get_cube_index( i, j, k + 1 ) ];
                    _last_inFOV[ get_cube_index( i, j, k ) ] = _last_inFOV[ get_cube_index( i, j, k + 1 ) ];
                }

                featsArray[ get_cube_index( i, j, k ) ] = laserCloudCubeSurfPointer;
                _last_inFOV[ get_cube_index( i, j, k ) ] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeK--;
        laserCloudCenDepth--;
    }

    cube_points_add->clear();
    featsFromMap->clear();
    memset( now_inFOV, 0, sizeof( now_inFOV ) );

#ifdef USE_ikdtree
    if ( cub_needrm.size() > 0 )
        ikdtree.Delete_Point_Boxes( cub_needrm );
    // s_plot4.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if ( cub_needad.size() > 0 )
        ikdtree.Add_Point_Boxes( cub_needad );
    // s_plot5.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if ( cube_points_add->points.size() > 0 )
        ikdtree.Add_Points( cube_points_add->points, true );
#endif

}

void R3LIVE::lidar_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pcl::PointCloud< PointType > pl_processed;
  pcl::PointCloud< velodyne_ros::Point > pl_orig;
  pcl::fromROSMsg( *msg, pl_orig );

  ros::Time first_point_time(pl_orig.points[0].timestamp);

  pl_processed.clear();
  pl_processed.reserve( pl_orig.points.size() );
  for (size_t i = 0; i < pl_orig.points.size(); i++ )
  {
    if (std::isnan(pl_orig.points[ i ].x) || std::isnan(pl_orig.points[ i ].y) || std::isnan(pl_orig.points[ i ].z)) {
      continue;
    }
    double range = std::sqrt( pl_orig.points[ i ].x * pl_orig.points[ i ].x + pl_orig.points[ i ].y * pl_orig.points[ i ].y +
                              pl_orig.points[ i ].z * pl_orig.points[ i ].z );
    if ( range < blind )
    {
      continue;
    }
    if (i % point_filter_num == 0) {
      Eigen::Vector3d pt_vec;
      PointType       added_pt;
      added_pt.x = pl_orig.points[ i ].x;
      added_pt.y = pl_orig.points[ i ].y;
      added_pt.z = pl_orig.points[ i ].z;
      added_pt.intensity = pl_orig.points[ i ].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = std::atan2( added_pt.y, added_pt.x ) * 57.3;
      if ( yaw_angle >= 180.0 )
        yaw_angle -= 360.0;
      if ( yaw_angle <= -180.0 )
        yaw_angle += 360.0;

      added_pt.curvature = (pl_orig.points[ i ].timestamp - pl_orig.points[ 0 ].timestamp) * 1000.0;  /// ms
//            std::cout << "added_pt.curvature: " << added_pt.curvature << " " << (pl_orig.points[ i ].timestamp - pl_orig.points[ 0 ].timestamp) << std::endl;
      pl_processed.points.push_back( added_pt );
    }
  }

  std::cout << "pl_processed size=" << pl_processed.points.size() << std::endl;

  pl_processed.height = 1;
  pl_processed.width = pl_processed.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg( pl_processed, output );
  output.header.frame_id = "livox";
  output.header.stamp = first_point_time;

  sensor_msgs::PointCloud2::Ptr msg_in( new sensor_msgs::PointCloud2( output ) );
  msg_in->header.stamp = ros::Time( output.header.stamp.toSec() - m_lidar_imu_time_delay );
  mtx_buffer.lock();
  std::cout<<"got feature"<<std::endl;
  if ( msg->header.stamp.toSec() < last_timestamp_lidar )
  {
    ROS_ERROR( "lidar loop back, clear buffer" );
    lidar_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  lidar_buffer.push_back( msg_in );
  last_timestamp_lidar = msg_in->header.stamp.toSec();
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

int R3LIVE::service_LIO_update()
{
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "/world";
    /*** variables definition ***/
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    cv::Mat matA1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matD1( 1, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matV1( 3, 3, CV_32F, cv::Scalar::all( 0 ) );
    cv::Mat matP( 6, 6, CV_32F, cv::Scalar::all( 0 ) );

    PointCloudXYZINormal::Ptr feats_undistort( new PointCloudXYZINormal() );
    PointCloudXYZINormal::Ptr feats_down( new PointCloudXYZINormal() );
    PointCloudXYZINormal::Ptr laserCloudOri( new PointCloudXYZINormal() );
    PointCloudXYZINormal::Ptr coeffSel( new PointCloudXYZINormal() );

    /*** variables initialize ***/
    FOV_DEG = fov_deg + 10;
    HALF_FOV_COS = std::cos( ( fov_deg + 10.0 ) * 0.5 * PI_M / 180.0 );

    for ( int i = 0; i < laserCloudNum; i++ )
    {
        featsArray[ i ].reset( new PointCloudXYZINormal() );
    }

    std::shared_ptr< ImuProcess > p_imu( new ImuProcess() );
    m_imu_process = p_imu;
    //------------------------------------------------------------------------------------------------------
    ros::Rate rate( 5000 );
    bool      status = ros::ok();
    set_initial_state_cov( g_lio_state );
    while ( ros::ok() )
    {
        if ( flg_exit )
            break;
        ros::spinOnce();
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );

        {
            std::unique_lock<std::mutex> lk(mtx_buffer);
            sig_buffer.wait(lk, [&] {
                if (!flg_exit) {
                    return sync_packages(Measures);
                } else {
                    return true;
                }
            });
        }
        if (flg_exit)
            break;
        std::unique_lock< std::mutex > lock( m_mutex_lio_process );
        if ( 1 )
        {
            std::cout << "========================================" << std::endl;

            if ( flg_reset )
            {
                ROS_WARN( "reset when rosbag play back" );
                p_imu->Reset();
                flg_reset = false;
                continue;
            }
            g_LiDAR_frame_index++;

            p_imu->Process( Measures, g_lio_state, feats_undistort );

            StatesGroup state_propagate( g_lio_state );

            if ( feats_undistort->empty() || ( feats_undistort == NULL ) )
            {
                frame_first_pt_time = Measures.lidar_beg_time;
                std::cout << "not ready for odometry" << std::endl;
                continue;
            }

            if ( ( Measures.lidar_beg_time - frame_first_pt_time ) < INIT_TIME )
            {
                flg_EKF_inited = false;
                std::cout << "||||||||||Initiallizing LiDAR||||||||||" << std::endl;
            }
            else
            {
                flg_EKF_inited = true;
            }
            /*** Compute the euler angle ***/
            Eigen::Vector3d euler_cur = RotMtoEuler( g_lio_state.rot_end );
            lasermap_fov_segment();
            downSizeFilterSurf.setInputCloud( feats_undistort );
            downSizeFilterSurf.filter( *feats_down );

            /*** initialize the map kdtree ***/
            if ( ( feats_down->points.size() > 1 ) && ( ikdtree.Root_Node == nullptr ) )
            {
                // std::vector<PointType> points_init = feats_down->points;
                PointCloudXYZINormal::Ptr feats_down_updated( new PointCloudXYZINormal( *feats_down ) );
                for ( int i = 0; i < feats_down->points.size(); i++ )
                {
                    /* transform to world frame */
                    pointBodyToWorld( &( feats_down->points[ i ] ), &( feats_down_updated->points[ i ] ) );
                }

                ikdtree.set_downsample_param( filter_size_map_min );
                ikdtree.Build( feats_down_updated->points );
                continue;
            }
            std::cout << "ikdtree size: " << ikdtree.size() << std::endl;
            if ( ikdtree.Root_Node == nullptr )
            {
                std::cout << "~~~~~~~ Initialize Map iKD-Tree Failed! ~~~~~~~" << std::endl;
                continue;
            }
            int featsFromMapNum = ikdtree.size();
            int feats_down_size = feats_down->points.size();
            
            /*** ICP and iterated Kalman filter update ***/
            PointCloudXYZINormal::Ptr coeffSel_tmpt( new PointCloudXYZINormal( *feats_down ) );
            PointCloudXYZINormal::Ptr feats_down_updated( new PointCloudXYZINormal( *feats_down ) );
            std::vector< double >     res_last( feats_down_size, 1000.0 ); // initial

            if ( featsFromMapNum >= 5 )
            {
                if ( m_if_publish_feature_map )
                {
                    PointVector().swap( ikdtree.PCL_Storage );
                    ikdtree.flatten( ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD );
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;

                    sensor_msgs::PointCloud2 laserCloudMap;
                    pcl::toROSMsg( *featsFromMap, laserCloudMap );
                    laserCloudMap.header.stamp = ros::Time::now();
                    laserCloudMap.header.frame_id = "world";
                    pubLaserCloudMap.publish( laserCloudMap );
                }

                std::vector< bool >               point_selected_surf( feats_down_size, true );
                std::vector< std::vector< int > > pointSearchInd_surf( feats_down_size );
                std::vector< PointVector >        Nearest_Points( feats_down_size );

                int  rematch_num = 0;
                bool rematch_en = 0;
                flg_EKF_converged = 0;
                deltaR = 0.0;
                deltaT = 0.0;
                double maximum_pt_range = 0.0;
                for ( iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++ )
                {
                    laserCloudOri->clear();
                    coeffSel->clear();

                    /** closest surface search and residual computation **/
                    for ( int i = 0; i < feats_down_size; i += m_lio_update_point_step )
                    {
//                        double     search_start = omp_get_wtime();
                        PointType &pointOri_tmpt = feats_down->points[ i ];
                        double     ori_pt_dis =
                            sqrt( pointOri_tmpt.x * pointOri_tmpt.x + pointOri_tmpt.y * pointOri_tmpt.y + pointOri_tmpt.z * pointOri_tmpt.z );
                        maximum_pt_range = std::max( ori_pt_dis, maximum_pt_range );
                        PointType &pointSel_tmpt = feats_down_updated->points[ i ];

                        /* transform to world frame */
                        pointBodyToWorld( &pointOri_tmpt, &pointSel_tmpt );
                        std::vector< float > pointSearchSqDis_surf;

                        auto &points_near = Nearest_Points[ i ];

                        if ( iterCount == 0 || rematch_en )
                        {
                            point_selected_surf[ i ] = true;
                            /** Find the closest surfaces in the map **/
                            ikdtree.Nearest_Search( pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf );

                            float max_distance = pointSearchSqDis_surf[ NUM_MATCH_POINTS - 1 ];
                            //  max_distance to add residuals
                            // ANCHOR - Long range pt stragetry
                            if ( max_distance > m_maximum_pt_kdtree_dis )
                            {
                                point_selected_surf[ i ] = false;
                            }
                        }

                        if ( point_selected_surf[ i ] == false )
                            continue;
                        /// Ax=b compute (a,b,c,d) plane's parameters
                        Eigen::MatrixX<float> matA;
                        Eigen::MatrixX<float> matB;
                        matA.setZero(NUM_MATCH_POINTS, 3);
                        matB.setConstant(NUM_MATCH_POINTS, 1, -1.0);

                        for (int j = 0; j < NUM_MATCH_POINTS; j++ ) {
                          matA.row(j) << points_near[ j ].x, points_near[ j ].y, points_near[ j ].z;
                        }

                        Eigen::MatrixX<float> AtA = matA.transpose() * matA;
                        Eigen::MatrixX<float> Atb = matA.transpose() * matB;
                        Eigen::Vector3<float> matX = AtA.ldlt().solve(Atb);
//                        Eigen::Vector3<float> matX = matA.colPivHouseholderQr().solve(matB);

                        float pa = matX(0);
                        float pb = matX(1);
                        float pc = matX(2);
                        float pd = 1;

                        float ps = sqrt( pa * pa + pb * pb + pc * pc );
                        pa /= ps;
                        pb /= ps;
                        pc /= ps;
                        pd /= ps;

                        bool planeValid = true;
                        for ( int j = 0; j < NUM_MATCH_POINTS; j++ )
                        {
                            // ANCHOR -  Planar check
                            if ( fabs( pa * points_near[ j ].x + pb * points_near[ j ].y + pc * points_near[ j ].z + pd ) >
                                 m_planar_check_dis ) // Raw 0.05
                            {
                                // ANCHOR - Far distance pt processing
                                if ( ori_pt_dis < maximum_pt_range * 0.90 || ( ori_pt_dis < m_long_rang_pt_dis ) )
                                // if(1)
                                {
                                    planeValid = false;
                                    point_selected_surf[ i ] = false;
                                    break;
                                }
                            }
                        }

                        if ( planeValid )
                        {
                            float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                            float s = 1 - 0.9 * fabs( pd2 ) /
                                              sqrt( sqrt( pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y +
                                                          pointSel_tmpt.z * pointSel_tmpt.z ) );
                            // ANCHOR -  Point to plane distance
                            double acc_distance = ( ori_pt_dis < m_long_rang_pt_dis ) ? m_maximum_res_dis : 1.0;
                            if ( pd2 < acc_distance )
                            {
                                point_selected_surf[ i ] = true;
                                coeffSel_tmpt->points[ i ].x = pa;
                                coeffSel_tmpt->points[ i ].y = pb;
                                coeffSel_tmpt->points[ i ].z = pc;
                                coeffSel_tmpt->points[ i ].intensity = pd2;
                                res_last[ i ] = std::abs( pd2 );
                            }
                            else
                            {
                                point_selected_surf[ i ] = false;
                            }
                        }
                    }
                    double total_residual = 0.0;
                    laserCloudSelNum = 0;

                    for ( int i = 0; i < coeffSel_tmpt->points.size(); i++ )
                    {
                        if ( point_selected_surf[ i ] && ( res_last[ i ] <= 2.0 ) )
                        {
                            laserCloudOri->push_back( feats_down->points[ i ] );
                            coeffSel->push_back( coeffSel_tmpt->points[ i ] );
                            total_residual += res_last[ i ];
                            laserCloudSelNum++;
                        }
                    }
                    res_mean_last = total_residual / laserCloudSelNum;

                    std::cout << "iteCount=" <<iterCount << " laserCloudSelNum=" << laserCloudSelNum << std::endl;

                    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                    Eigen::MatrixXd Hsub( laserCloudSelNum, 6 );
                    Eigen::VectorXd meas_vec( laserCloudSelNum );
                    Hsub.setZero();

                    for ( int i = 0; i < laserCloudSelNum; i++ )
                    {
                        const PointType &laser_p = laserCloudOri->points[ i ];
                        Eigen::Vector3d  point_this( laser_p.x, laser_p.y, laser_p.z );
                        point_this = Lidar_R_to_IMU * point_this + Lidar_offset_to_IMU;
//                        Eigen::Matrix3d point_crossmat;
//                        point_crossmat << SKEW_SYM_MATRIX( point_this );

                        /*** get the normal vector of closest surface/corner ***/
                        const PointType &norm_p = coeffSel->points[ i ];
                        Eigen::Vector3d  norm_vec( norm_p.x, norm_p.y, norm_p.z );

                        /*** calculate the Measuremnt Jacobian matrix H ***/
//                        Eigen::Vector3d A( point_crossmat * g_lio_state.rot_end.transpose() * norm_vec );
                        Eigen::Vector3d point_w = g_lio_state.rot_end * point_this;
                        Eigen::Matrix3d skew_pt;
                        skew_pt << SKEW_SYM_MATRIX(point_w);
                        Eigen::Vector3d A(skew_pt * norm_vec);

                        Hsub.row( i ) << VEC_FROM_ARRAY( A ), norm_p.x, norm_p.y, norm_p.z;

                        /*** Measuremnt: distance to the closest surface/corner ***/
                        meas_vec( i ) = -norm_p.intensity;
                    }

                    Eigen::Vector3d                           rot_add, t_add, v_add, bg_add, ba_add, g_add;
                    Eigen::Matrix< double, DIM_OF_STATES, 1 > solution;
                    Eigen::MatrixXd                           K( DIM_OF_STATES, laserCloudSelNum );

                    /*** Iterative Kalman Filter Update ***/
                    if ( !flg_EKF_inited )
                    {
                        std::cout << "Run EKF init" << std::endl;
                        /*** only run in initialization period ***/
                        set_initial_state_cov( g_lio_state );
                    }
                    else
                    {
                        // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph" << ANSI_COLOR_RESET << endl;
                        auto &&Hsub_T = Hsub.transpose();
                        H_T_H.block< 6, 6 >( 0, 0 ) = Hsub_T * Hsub;
                        Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > &&K_1 =
                            ( H_T_H + ( g_lio_state.cov / LASER_POINT_COV ).inverse() ).inverse();
                        K = K_1.block< DIM_OF_STATES, 6 >( 0, 0 ) * Hsub_T;

//                        auto vec = state_propagate - g_lio_state;
                        Eigen::Matrix<double, DIM_OF_STATES, 1> vec;
                        Eigen::Matrix3d rotd = state_propagate.rot_end * g_lio_state.rot_end.transpose();
                        vec.block<3, 1>(0, 0) = SO3_LOG<double>(rotd);
                        vec.block<3, 1>(3, 0) = state_propagate.pos_end - g_lio_state.pos_end;
                        vec.block<3, 1>(6, 0) = state_propagate.vel_end- g_lio_state.vel_end;
                        vec.block<3, 1>(9, 0) = state_propagate.bias_g - g_lio_state.bias_g;
                        vec.block<3, 1>(12, 0) = state_propagate.bias_a - g_lio_state.bias_a;

                        solution = K * ( meas_vec - Hsub * vec.block< 6, 1 >( 0, 0 ) );
                        // double speed_delta = solution.block( 0, 6, 3, 1 ).norm();
                        // if(solution.block( 0, 6, 3, 1 ).norm() > 0.05 )
                        // {
                        //     solution.block( 0, 6, 3, 1 ) = solution.block( 0, 6, 3, 1 ) / speed_delta * 0.05;
                        // }

//                        g_lio_state = state_propagate + solution;
                        g_lio_state.rot_end = Exp<double>(solution(0), solution(1), solution(2)) * state_propagate.rot_end;
                        g_lio_state.pos_end = state_propagate.pos_end + solution.block<3, 1>(3, 0);
                        g_lio_state.vel_end = state_propagate.vel_end + solution.block<3, 1>(6, 0);
                        g_lio_state.bias_g = state_propagate.bias_g + solution.block<3, 1>(9, 0);
                        g_lio_state.bias_a = state_propagate.bias_a + solution.block<3, 1>(12, 0);

                        // cout << ANSI_COLOR_RED_BOLD << "Run EKF uph, vec = " << vec.head<9>().transpose() << ANSI_COLOR_RESET << endl;
                        rot_add = solution.block< 3, 1 >( 0, 0 );
                        t_add = solution.block< 3, 1 >( 3, 0 );
                        flg_EKF_converged = false;
                        if ( ( ( rot_add.norm() * 57.3 - deltaR ) < 0.01 ) && ( ( t_add.norm() * 100 - deltaT ) < 0.015 ) )
                        {
                            flg_EKF_converged = true;
                        }

                        deltaR = rot_add.norm() * 57.3;
                        deltaT = t_add.norm() * 100;
                    }

                    // printf_line;
                    g_lio_state.last_update_time = Measures.lidar_end_time;
                    euler_cur = RotMtoEuler( g_lio_state.rot_end );

                    /*** Rematch Judgement ***/
                    rematch_en = false;
                    if ( flg_EKF_converged || ( ( rematch_num == 0 ) && ( iterCount == ( NUM_MAX_ITERATIONS - 2 ) ) ) )
                    {
                        rematch_en = true;
                        rematch_num++;
                    }

                    /*** Convergence Judgements and Covariance Update ***/
                    // if (rematch_num >= 10 || (iterCount == NUM_MAX_ITERATIONS - 1))
                    if ( rematch_num >= 2 || ( iterCount == NUM_MAX_ITERATIONS - 1 ) ) // Fast lio ori version.
                    {
                        if ( flg_EKF_inited )
                        {
                            std::cout << "update cov" << std::endl;
                            /*** Covariance Update ***/
                            G.block< DIM_OF_STATES, 6 >( 0, 0 ) = K * Hsub;
                            g_lio_state.cov = ( I_STATE - G ) * g_lio_state.cov;
                            total_distance += ( g_lio_state.pos_end - position_last ).norm();
                            position_last = g_lio_state.pos_end;
                            std::cout << "update cov done" << std::endl;

                            // std::cout << "position: " << g_lio_state.pos_end.transpose() << " total distance: " << total_distance << std::endl;
                        }
                        break;
                    }
                }

                /*** add new frame points to map ikdtree ***/
                PointVector points_history;
                ikdtree.acquire_removed_points( points_history );

                memset( cube_updated, 0, sizeof( cube_updated ) );

                for ( int i = 0; i < points_history.size(); i++ )
                {
                    PointType &pointSel = points_history[ i ];

                    int cubeI = int( ( pointSel.x + 0.5 * cube_len ) / cube_len ) + laserCloudCenWidth;
                    int cubeJ = int( ( pointSel.y + 0.5 * cube_len ) / cube_len ) + laserCloudCenHeight;
                    int cubeK = int( ( pointSel.z + 0.5 * cube_len ) / cube_len ) + laserCloudCenDepth;

                    if ( pointSel.x + 0.5 * cube_len < 0 )
                        cubeI--;
                    if ( pointSel.y + 0.5 * cube_len < 0 )
                        cubeJ--;
                    if ( pointSel.z + 0.5 * cube_len < 0 )
                        cubeK--;

                    if ( cubeI >= 0 && cubeI < laserCloudWidth && cubeJ >= 0 && cubeJ < laserCloudHeight && cubeK >= 0 && cubeK < laserCloudDepth )
                    {
                        int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                        featsArray[ cubeInd ]->push_back( pointSel );
                    }
                }

                for ( int i = 0; i < feats_down_size; i++ )
                {
                    /* transform to world frame */
                    pointBodyToWorld( &( feats_down->points[ i ] ), &( feats_down_updated->points[ i ] ) );
                }
                ikdtree.Add_Points( feats_down_updated->points, true );

            }
//            std::cout <<"lio cost time: " << tim.toc()<< endl;

            /******* Publish current frame points in world coordinates:  *******/
            laserCloudFullRes2->clear();
            *laserCloudFullRes2 = dense_map_en ? ( *feats_undistort ) : ( *feats_down );

            int laserCloudFullResNum = laserCloudFullRes2->points.size();

            pcl::PointXYZI temp_point;
            laserCloudFullResColor->clear();
            {
                for ( int i = 0; i < laserCloudFullResNum; i++ )
                {
                    RGBpointBodyToWorld( &laserCloudFullRes2->points[ i ], &temp_point );
                    laserCloudFullResColor->push_back( temp_point );
                }
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg( *laserCloudFullResColor, laserCloudFullRes3 );
                // laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
                laserCloudFullRes3.header.stamp.fromSec( Measures.lidar_end_time );
                laserCloudFullRes3.header.frame_id = "world"; // world; camera_init
                pubLaserCloudFullRes.publish( laserCloudFullRes3 );

                static Eigen::Matrix3d last_rot = g_lio_state.rot_end;
                static Eigen::Vector3d last_pos = g_lio_state.pos_end;
                static std::vector<Eigen::Vector3d> trajectory;

                double delta_dis = (last_pos - g_lio_state.pos_end).norm();
                Eigen::Matrix3d delta_rot = last_rot.transpose() * g_lio_state.rot_end;
                Eigen::Vector3d delta_euler = RotMtoEuler(delta_rot);
                delta_euler *= 180.0 / M_PI;

                const double DELTA_DISTANCE = 0.2;
                const double DELTA_ANGLE = 5.0; // degree
                if (delta_dis > DELTA_DISTANCE || delta_euler.x() > DELTA_ANGLE || delta_euler.y() > DELTA_ANGLE || delta_euler.z() > DELTA_ANGLE) {
                    last_rot = g_lio_state.rot_end;
                    last_pos = g_lio_state.pos_end;
                    pubLaserCloudFullRes1.publish( laserCloudFullRes3 );

                    trajectory.emplace_back(last_pos);
                }

                if (trajectory.size() >= 2) {
                    visualization_msgs::Marker path_marker;
                    path_marker.header.frame_id = "world";
                    path_marker.header.stamp.fromSec( Measures.lidar_end_time );
                    path_marker.ns = "path";
                    path_marker.id = 1;
                    path_marker.type = visualization_msgs::Marker::LINE_STRIP;
                    path_marker.action = visualization_msgs::Marker::ADD;
                    path_marker.pose.orientation.w = 1.0;
                    std_msgs::ColorRGBA color;
                    color.a = 1.0;
                    color.r = 1.0;
                    color.g = 1.0;
                    color.b = 0.0;
                    path_marker.color = color;
                    path_marker.scale.x = 0.1;
                    path_marker.scale.y = 0.1;
                    for (auto tp : trajectory) {
                        geometry_msgs::Point point;
                        point.x = tp.x();
                        point.y = tp.y();
                        point.z = tp.z();
                        path_marker.points.push_back(point);
                    }
                    pubPath.publish(path_marker);
                }
            }

            /******* Publish Maps:  *******/
            sensor_msgs::PointCloud2 laserCloudMap;
            pcl::toROSMsg( *featsFromMap, laserCloudMap );
            laserCloudMap.header.stamp.fromSec( Measures.lidar_end_time ); // ros::Time().fromSec(last_timestamp_lidar);
            laserCloudMap.header.frame_id = "world";
            pubLaserCloudMap.publish( laserCloudMap );

            /******* Publish Odometry ******/
            geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw( euler_cur( 0 ), euler_cur( 1 ), euler_cur( 2 ) );
            odomAftMapped.header.frame_id = "world";
//            odomAftMapped.child_frame_id = "/aft_mapped";
            odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
            odomAftMapped.pose.orientation.x = geoQuat.x;
            odomAftMapped.pose.orientation.y = geoQuat.y;
            odomAftMapped.pose.orientation.z = geoQuat.z;
            odomAftMapped.pose.orientation.w = geoQuat.w;
            odomAftMapped.pose.position.x = g_lio_state.pos_end( 0 );
            odomAftMapped.pose.position.y = g_lio_state.pos_end( 1 );
            odomAftMapped.pose.position.z = g_lio_state.pos_end( 2 );

            pubOdomAftMapped.publish( odomAftMapped );

            static tf::TransformBroadcaster br;
            tf::Transform                   transform;
            tf::Quaternion                  q;
            transform.setOrigin(
                tf::Vector3( odomAftMapped.pose.position.x, odomAftMapped.pose.position.y, odomAftMapped.pose.position.z ) );
            q.setW( odomAftMapped.pose.orientation.w );
            q.setX( odomAftMapped.pose.orientation.x );
            q.setY( odomAftMapped.pose.orientation.y );
            q.setZ( odomAftMapped.pose.orientation.z );
            transform.setRotation( q );
            br.sendTransform( tf::StampedTransform( transform, ros::Time().fromSec( Measures.lidar_end_time ), "world", "/aft_mapped" ) );

            msg_body_pose.header.stamp = ros::Time::now();
            msg_body_pose.header.frame_id = "/camera_odom_frame";
            msg_body_pose.pose.position.x = g_lio_state.pos_end( 0 );
            msg_body_pose.pose.position.y = g_lio_state.pos_end( 1 );
            msg_body_pose.pose.position.z = g_lio_state.pos_end( 2 );
            msg_body_pose.pose.orientation.x = geoQuat.x;
            msg_body_pose.pose.orientation.y = geoQuat.y;
            msg_body_pose.pose.orientation.z = geoQuat.z;
            msg_body_pose.pose.orientation.w = geoQuat.w;

            /******* Publish Path ********/
//            msg_body_pose.header.frame_id = "world";
//            if ( frame_num > 10 )
//            {
//                path.poses.push_back( msg_body_pose );
//            }
//            pubPath.publish( path );

        }
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}
