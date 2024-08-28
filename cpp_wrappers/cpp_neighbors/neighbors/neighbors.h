
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>	

#include "../../cpp_utils/cloud/cloud.h"
#include "../../cpp_utils/nanoflann/nanoflann.hpp"

#include "../../cpp_utils/kiss_matcher/kdtree/kdtree_tbb.hpp"
#include "../../cpp_utils/kiss_matcher/points/point_cloud.hpp"

#include <set>
#include <cstdint>

using namespace std;

using MyKdTree = kiss_matcher::UnsafeKdTree<kiss_matcher::PointCloud>;

void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius);

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius);

void batch_nanoflann_neighbors(const vector<PointXYZ>& queries,
                               const vector<PointXYZ>& supports,
                               const vector<int>& q_batches,
                               const  vector<int>& s_batches,
                               vector<int>& neighbors_indices,
                               const float radius);

void batch_nanoflanntbb_neighbors(const vector<PointXYZ>& queries,
                               const vector<PointXYZ>& supports,
                               const vector<int>& q_batches,
                               const  vector<int>& s_batches,
                               vector<int>& neighbors_indices,
                               const float radius);

