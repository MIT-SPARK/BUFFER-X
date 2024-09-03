
#include "neighbors.h"


void brute_neighbors(vector<PointXYZ>& queries, vector<PointXYZ>& supports, vector<int>& neighbors_indices, float radius, int verbose)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	vector<vector<int>> tmp(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
			if ((p0 - p).sq_norm() < r2)
			{
				tmp[i0].push_back(i);
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
		    d2 = (p0 - p).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;


	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

	    // Loop only over the supports of current batch
	    vector<PointXYZ>::iterator p_it;
		int i = 0;
        for(p_it = supports.begin() + sum_sb; p_it < supports.begin() + sum_sb + s_batches[b]; p_it++ )
        {
		    d2 = (p0 - *p_it).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, sum_sb + i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	return;
}


void batch_nanoflann_neighbors(const vector<PointXYZ>& queries,
                               const vector<PointXYZ>& supports,
                               const vector<int>& q_batches,
                               const  vector<int>& s_batches,
                               vector<int>& neighbors_indices,
                               const float radius)
{

	// Initialize variables
	// ******************

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree_t;

    // Pointer to trees
    my_kd_tree_t* index;

    // Build KDTree for the first batch element
    current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
    index = new my_kd_tree_t(3, current_cloud, tree_params);
    index->buildIndex();


	// Search neigbors indices
	// ***********************

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

	for (auto& p0 : queries)
	{

	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;

	        // Change the points
	        current_cloud.pts.clear();
            current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

	        // Build KDTree of the current element of the batch
            delete index;
            index = new my_kd_tree_t(3, current_cloud, tree_params);
            index->buildIndex();
	    }

	    // Initial guess of neighbors size
        all_inds_dists[i0].reserve(max_count);

	    // Find neighbors
	    float query_pt[3] = { p0.x, p0.y, p0.z};
		size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

        // Update max count
        if (nMatches > max_count)
            max_count = nMatches;

        // Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	delete index;

	return;
}

void batch_nanoflanntbb_neighbors(const vector<PointXYZ>& queries,
                               const vector<PointXYZ>& supports,
                               const vector<int>& q_batches,
                               const  vector<int>& s_batches,
                               vector<int>& neighbors_indices,
                               const float radius)
{
	// Initialize variables
	// ******************
    int num_queries = queries.size();

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;

	// Counting vector
	int max_count = 0; 
	int empiricall_max_count = 400; // Sufficiently large number

    std::vector<uint32_t> vector_tmp;
    vector_tmp.reserve(empiricall_max_count);
    std::vector<std::vector<uint32_t>> all_inds(num_queries, vector_tmp);

    std::vector<int> q_batches_accum = {0};
    for (const auto& num_batch: q_batches) {
        int accum = q_batches_accum.back() + num_batch;
        q_batches_accum.push_back(accum);
    }

	// Nanoflann related variables
	// ***************************

    kiss_matcher::PointCloud cloud_q, cloud_s0, cloud_s1;

    cloud_q.points.resize(num_queries);
    for (size_t k = 0; k < num_queries; ++k)
    {
      const auto &p = queries[k];
      cloud_q.points[k] = Eigen::Vector4d(p.x, p.y, p.z, 1.0);
    }

    cloud_s0.points.resize(s_batches[0]);
    for (size_t i = 0; i < s_batches[0]; ++i)
    {
      const auto &p = supports[i];
      cloud_s0.points[i] = Eigen::Vector4d(p.x, p.y, p.z, 1.0);
    }

    cloud_s1.points.resize(s_batches[1]);
    for (size_t j = 0; j < s_batches[1]; ++j)
    {
      const auto &p = supports[s_batches[0] + j];
      cloud_s1.points[j] = Eigen::Vector4d(p.x, p.y, p.z, 1.0);
    }


    MyKdTree kdtree_s0(cloud_s0);
    MyKdTree kdtree_s1(cloud_s1);

	// Search neigbors indices
	// ***********************
    auto which_batch = [&](const int idx) {
        for (int i=1; i < q_batches_accum.size(); ++i) 
        {
            if (q_batches_accum[i-1] <= idx && idx < q_batches_accum[i]) 
            {
                return i - 1;
            }
        }
    };

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

    max_count = tbb::parallel_reduce(
        // range
        tbb::blocked_range<int>(0, num_queries),
        // identity
        0,
        // 1st lambda 
        [&](tbb::blocked_range<int> r, int max_num_neighbors) -> int {
        max_num_neighbors = 0;
        for (int i0 = r.begin(); i0 < r.end(); ++i0)
        {
            std::vector<std::pair<size_t, double> > indices_dists;
            indices_dists.reserve(1000);

            // Check if we changed batch
            if (which_batch(i0) % 2 == 0)
            {
                size_t num_results = kdtree_s0.radius_search(cloud_q.point(i0), r2, indices_dists);
                for (const auto& [idx, sqr_dist]: indices_dists) {
                    all_inds[i0].emplace_back(idx);
                }
                max_num_neighbors = max_num_neighbors > indices_dists.size() ? max_num_neighbors : indices_dists.size();

            } else {
                size_t num_results = kdtree_s1.radius_search(cloud_q.point(i0), r2, indices_dists);
                for (const auto& [idx, sqr_dist]: indices_dists) {
                    all_inds[i0].emplace_back(idx);
                }
                max_num_neighbors = max_num_neighbors > indices_dists.size() ? max_num_neighbors : indices_dists.size();

            }
        }
        return max_num_neighbors;
    }, 
    // 2nd lambda: parallel reduction
    [](const int a, const int b) -> int {

        return a > b? a : b;
    });

    // `supports.size()` is a dummy number
	neighbors_indices.resize(queries.size() * max_count, supports.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, num_queries), [&](tbb::blocked_range<int> r) {
        for (int i0 = r.begin(); i0 < r.end(); ++i0)
        {
            const auto& inds_dists = all_inds[i0];
            if (which_batch(i0) % 2 == 0) {
                for (int j = 0; j < max_count; j++)
                {
                    if (j < inds_dists.size())
                    {
                        neighbors_indices[i0 * max_count + j] = inds_dists[j];
                    }

                }
            } else {
                for (int j = 0; j < max_count; j++)
                {
                    if (j < inds_dists.size())
                    {
                        neighbors_indices[i0 * max_count + j] = inds_dists[j] + s_batches[0];
                    }

                }
            }
        }
    });

    return;
}
