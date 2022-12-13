//
// Created by zhihui on 3/22/22.
//

#ifndef SRC_VOXEL_INDICES_GRID_H
#define SRC_VOXEL_INDICES_GRID_H

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>

#include <pcl/filters/boost.h>
#include <pcl/filters/filter.h>
#include <map>
#include <unordered_set>

namespace pcl {

    /**
     * VoxelIndicesGrid
     * */
    template<typename PointT>
    class VoxelIndicesGrid : public VoxelGrid<PointT> {
    protected:
        using Filter<PointT>::filter_name_;
        using Filter<PointT>::getClassName;
        using Filter<PointT>::input_;
        using Filter<PointT>::indices_;

        typedef typename Filter<PointT>::PointCloud PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;
        typedef typename PointCloud::ConstPtr PointCloudConstPtr;
    public:
        typedef boost::shared_ptr<VoxelIndicesGrid<PointT> > Ptr;
        typedef boost::shared_ptr<const VoxelIndicesGrid<PointT> > ConstPtr;

        /** \brief Empty constructor. */
        VoxelIndicesGrid() : filteredIndices_(new std::vector<int>) {
            filter_name_ = "VoxelIndicesGrid";
        }

        /** \brief Destructor. */
        virtual ~VoxelIndicesGrid() {
        }

        IndicesPtr getFilteredIndices() {
            return filteredIndices_;
        }
    protected:
        /** \brief Point Indices of filtered points in input pointcloud */
        IndicesPtr filteredIndices_;

        /** \brief Downsample a Point Cloud using a voxelized grid approach
              * \param[out] output the resultant point cloud message
              */
        void
        applyFilter(PointCloud &output) {
            // Has the input dataset been set already?
            if (!input_) {
                PCL_WARN ("[pcl::%s::applyFilter] No input dataset given!\n", getClassName().c_str());
                output.width = output.height = 0;
                output.points.clear();
                return;
            }

            // Copy the header (and thus the frame_id) + allocate enough space for points
            output.height = 1;                    // downsampling breaks the organized structure
            output.is_dense = true;                 // we filter out invalid points

            Eigen::Vector4f min_p, max_p;
            // Get the minimum and maximum dimensions
            if (!this->filter_field_name_.empty()) // If we don't want to process the entire cloud...
                getMinMax3D<PointT>(input_,
                                    *indices_,
                                    this->filter_field_name_,
                                    static_cast<float> (this->filter_limit_min_),
                                    static_cast<float> (this->filter_limit_max_),
                                    min_p,
                                    max_p,
                                    this->filter_limit_negative_);
            else
                getMinMax3D<PointT>(*input_, *indices_, min_p, max_p);

            // Check that the leaf size is not too small, given the size of the data
            int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * this->inverse_leaf_size_[0]) + 1;
            int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * this->inverse_leaf_size_[1]) + 1;
            int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * this->inverse_leaf_size_[2]) + 1;

            if ((dx * dy * dz) > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
                PCL_WARN("[pcl::%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would overflow.",
                         getClassName().c_str());
                output = *input_;
                return;
            }

            // Compute the minimum and maximum bounding box values
            this->min_b_[0] = static_cast<int> (floor(min_p[0] * this->inverse_leaf_size_[0]));
            this->max_b_[0] = static_cast<int> (floor(max_p[0] * this->inverse_leaf_size_[0]));
            this->min_b_[1] = static_cast<int> (floor(min_p[1] * this->inverse_leaf_size_[1]));
            this->max_b_[1] = static_cast<int> (floor(max_p[1] * this->inverse_leaf_size_[1]));
            this->min_b_[2] = static_cast<int> (floor(min_p[2] * this->inverse_leaf_size_[2]));
            this->max_b_[2] = static_cast<int> (floor(max_p[2] * this->inverse_leaf_size_[2]));

            // Compute the number of divisions needed along all axis
            this->div_b_ = this->max_b_ - this->min_b_ + Eigen::Vector4i::Ones();
            this->div_b_[3] = 0;

            // Set up the division multiplier
            this->divb_mul_ = Eigen::Vector4i(1, this->div_b_[0], this->div_b_[0] * this->div_b_[1], 0);

            // Storage for mapping leaf and pointcloud indexes
            std::vector<cloud_point_index_idx> index_vector;
            index_vector.reserve(indices_->size());
            std::unordered_set<unsigned int> voxelGrid;

            // First pass: go over all points and insert them into the index_vector vector
            // with calculated idx. Points with the same idx value will contribute to the
            // same point of resulting CloudPoint
            filteredIndices_->clear();
            filteredIndices_->reserve(indices_->size());
            for (std::vector<int>::const_iterator it = indices_->begin(); it != indices_->end(); ++it) {
                if (!input_->is_dense)
                    // Check if the point is invalid
                    if (!pcl_isfinite (input_->points[*it].x) ||
                        !pcl_isfinite (input_->points[*it].y) ||
                        !pcl_isfinite (input_->points[*it].z))
                        continue;

                int ijk0 =
                        static_cast<int> (floor(input_->points[*it].x * this->inverse_leaf_size_[0])
                                          - static_cast<float> (this->min_b_[0]));
                int ijk1 =
                        static_cast<int> (floor(input_->points[*it].y * this->inverse_leaf_size_[1])
                                          - static_cast<float> (this->min_b_[1]));
                int ijk2 =
                        static_cast<int> (floor(input_->points[*it].z * this->inverse_leaf_size_[2])
                                          - static_cast<float> (this->min_b_[2]));

                // Compute the centroid leaf index
                int idx = ijk0 * this->divb_mul_[0] + ijk1 * this->divb_mul_[1] + ijk2 * this->divb_mul_[2];

                if(voxelGrid.find(static_cast<unsigned int> (idx)) == voxelGrid.end())
                {
                    // can insert
                    filteredIndices_->push_back(*it);
                    voxelGrid.insert(idx);
                }// else skip point

            }

            int total = filteredIndices_->size();
            output.points.resize(total);

            for(int i=0;i<total;++i)
            {
                output.points[i]= input_->points[(*filteredIndices_)[i]];
            }
            output.width = static_cast<uint32_t> (output.points.size());
        }
    };
}

#endif //SRC_VOXEL_INDICES_GRID_H
