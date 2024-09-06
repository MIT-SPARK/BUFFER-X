#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:

	// Elements
	// ********

	int order;
	int count;
	PointXYZ point;
	vector<float> features;
	vector<unordered_map<int, int>> labels;
	vector<int> input_point_indices;
	vector<int> sampled_point_neighboring_indices;

	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		order = -1; 
		count = 0; 
		point = PointXYZ();
    input_point_indices.reserve(10); // Empiricallly, 8 points are set to a single grid.
	}

  SampledData(const size_t max_neighbors)
	{
		order = -1; 
		count = 0;
		point = PointXYZ();
    input_point_indices.reserve(10);
    sampled_point_neighboring_indices.reserve(max_neighbors);
	}


	SampledData(const size_t fdim, const size_t ldim)
	{
		order = -1; 
		count = 0;
		point = PointXYZ();
    input_point_indices.reserve(10);

	  features = vector<float>(fdim);
	  labels = vector<unordered_map<int, int>>(ldim);
	}

	// Method Update
	void update_all(const PointXYZ& p, vector<float>::iterator f_begin, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_features(const PointXYZ& p, vector<float>::iterator f_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		return;
	}
	void update_classes(const PointXYZ& p, vector<int>::iterator l_begin)
	{
		count += 1;
		point += p;
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_points(const PointXYZ& p)
	{
		count += 1;
		point += p;
		return;
	}
  void update_points_and_indices(const PointXYZ& p, const int idx)
	{
		count += 1;
		point += p;
    input_point_indices.emplace_back(idx);
		return;
	}
};

void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

void grid_subsampling_while_searching(vector<PointXYZ>& original_points,
                                      vector<PointXYZ>& subsampled_points,
                                      vector<float>& original_features,
                                      vector<float>& subsampled_features,
                                      vector<int>& original_classes,
                                      vector<int>& subsampled_classes,
                                      unordered_map<size_t, SampledData>& data,
                                      float sampleDl, 
                                      int max_neighbors,
                                      int verbose);

void batch_grid_subsampling(const vector<PointXYZ>& original_points,
                            vector<PointXYZ>& subsampled_points,
                            const vector<float>& original_features,
                            vector<float>& subsampled_features,
                            const vector<int>& original_classes,
                            vector<int>& subsampled_classes,
                            const vector<int>& original_batches,
                            vector<int>& subsampled_batches,
                            float sampleDl,
                            int max_p);

void batch_grid_subsampling_and_searching(const vector<PointXYZ>& original_points,
                                          vector<PointXYZ>& subsampled_points,
                                          const vector<float>& original_features,
                                          vector<float>& subsampled_features,
                                          const vector<int>& original_classes,
                                          vector<int>& subsampled_classes,
                                          const vector<int>& original_batches,
                                          vector<int>& subsampled_batches,
                                          vector<int>& indices_for_upconv,
                                          float sampleDl,
                                          int max_p,
                                          int max_neighbors);


