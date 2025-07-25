#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "utils.hpp"

#define checkCUDA(call) {                                           \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
        exit(1);                                                    \
    }                                                               \
}

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {
    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
    float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

    if (depth_val <= 0 || depth_val > 6)
      continue;

    float diff = (depth_val - pt_cam_z) * sqrtf(1 + powf((pt_cam_x / pt_cam_z), 2) + powf((pt_cam_y / pt_cam_z), 2));
    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
  }
}

std::vector<std::string> getSubdirectories(const std::string& root_dir) {
  std::vector<std::string> subdirectories;
  DIR *dir;
  struct dirent *entry;
  struct stat info;

  if ((dir = opendir(root_dir.c_str())) != NULL) {
    while ((entry = readdir(dir)) != NULL) {
      std::string subdir_name = entry->d_name;
      std::string full_path = root_dir + "/" + subdir_name;

      if (subdir_name != "." && subdir_name != ".." && stat(full_path.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        subdirectories.push_back(full_path);
      }
    }
    closedir(dir);
  } else {
    std::cerr << "Error opening directory: " << root_dir << std::endl;
    exit(1);
  }

  return subdirectories;
}

void createDirectoryIfNotExists(const std::string& path) {
  if (access(path.c_str(), F_OK) != 0) {
    if (mkdir(path.c_str(), 0755) != 0) {
      std::cerr << "Failed to create directory: " << path << std::endl;
      exit(1);
    } else {
      std::cout << "Created directory: " << path << std::endl;
    }
  }
}

// Main function
int main(int argc, char * argv[]) {
  // Parent directory containing all subdirectories (0a7cc12c0e, 0a76e06478, etc.)
  std::string root_data_path = "/root/datasets/Scannetpp_iphone/test";

  std::vector<std::string> subdirectories = getSubdirectories(root_data_path);

  for (std::string& subdirectory : subdirectories) {
    subdirectory = subdirectory + "/iphone";
    std::string depth_path = subdirectory + "/depth";
    std::string pose_path = subdirectory + "/pose";
    std::string output_path = subdirectory + "/tsdf";
    createDirectoryIfNotExists(output_path);
    int num_frames = 50;
    int im_width = 256;
    int im_height = 192;
    float cam_K[3 * 3];
    float base2world[4 * 4];
    float cam2base[4 * 4];
    float cam2world[4 * 4];
    float depth_im[im_height * im_width];

    // Voxel grid parameters
    float voxel_grid_origin_x = -1.5f;
    float voxel_grid_origin_y = -1.5f;
    float voxel_grid_origin_z = 0.5f;
    float voxel_size = 0.006f;
    float trunc_margin = voxel_size * 5;
    int voxel_grid_dim_x = 500;
    int voxel_grid_dim_y = 500;
    int voxel_grid_dim_z = 500;

    int frame_count = 0;
    int batch_count = 0;

    int flag_last_frame = 0;

    // Allocate GPU memory for voxel grid and other parameters
    float * gpu_voxel_grid_TSDF;
    float * gpu_voxel_grid_weight;
    checkCUDA(cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float)));
    checkCUDA(cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float)));
    float * gpu_cam_K;
    float * gpu_cam2base;
    float * gpu_depth_im;
    checkCUDA(cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float)));
    checkCUDA(cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float)));
    checkCUDA(cudaMalloc(&gpu_depth_im, im_height * im_width * sizeof(float)));

    while (true) {
      // Reset camera pose and depth image for each batch
      int base_frame_idx = frame_count;
      std::ostringstream base_frame_prefix;
      base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;

      // Load base frame intrinsic and pose data
      std::string cam_K_file = subdirectory + "/intrinsic/frame_" + base_frame_prefix.str() + ".intrinsic.txt";
      std::string base2world_file = pose_path + "/frame_" + base_frame_prefix.str() + ".pose.txt";
      std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
      std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);
      std::vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
      std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

      // Invert base frame camera pose to get world-to-base frame transform
      float base2world_inv[16] = {0};
      invert_matrix(base2world, base2world_inv);

      // Initialize voxel grid
      float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
      float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
      for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
        voxel_grid_TSDF[i] = 1.0f;
      memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

      checkCUDA(cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

      for (int batch_frame_count = 0; batch_frame_count < num_frames; ++batch_frame_count) {
        int current_frame_idx = base_frame_idx + batch_frame_count;

        std::ostringstream curr_frame_prefix;
        curr_frame_prefix << std::setw(6) << std::setfill('0') << current_frame_idx;

        // Construct file paths for depth and camera pose
        std::string depth_im_file = depth_path + "/frame_" + curr_frame_prefix.str() + ".depth.png";
        std::string cam2world_file = pose_path + "/frame_" + curr_frame_prefix.str() + ".pose.txt";

        // Check if the file exists
        std::ifstream depth_file_check(depth_im_file);
        if (!depth_file_check.good()) {
          std::cout << "No more frames found. Stopping process." << std::endl;
          flag_last_frame = 1;
          delete[] voxel_grid_TSDF;
          delete[] voxel_grid_weight;
          break;
        }

        // Load depth and camera pose data
        ReadDepth(depth_im_file, im_height, im_width, depth_im);
        std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
        std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

        // Compute relative camera pose (camera-to-base frame)
        multiply_matrix(base2world_inv, cam2world, cam2base);

        // Copy data to GPU
        checkCUDA(cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice));
        checkCUDA(cudaMemcpy(gpu_depth_im, depth_im, im_height * im_width * sizeof(float), cudaMemcpyHostToDevice));

        std::cout << "Fusing: " << depth_im_file << std::endl;

        // Perform integration
        Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im,
                                                             im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                                             voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                                                             gpu_voxel_grid_TSDF, gpu_voxel_grid_weight);

        checkCUDA(cudaDeviceSynchronize());

        frame_count++;
      }
      if (flag_last_frame)
        break;
      // Save the current TSDF grid
      checkCUDA(cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost));
      checkCUDA(cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost));

      std::string output_file = output_path + "/cloud_bin_" + std::to_string(batch_count++) + ".ply";
      std::cout << "Saving surface point cloud (" << output_file << ")..." << std::endl;
      SaveVoxelGrid2SurfacePointCloud(output_file, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                      voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                      voxel_grid_TSDF, voxel_grid_weight, 0.2f, 1.0f);

      delete[] voxel_grid_TSDF;
      delete[] voxel_grid_weight;
    }

    // Clean up GPU memory for this subdirectory
    cudaFree(gpu_voxel_grid_TSDF);
    cudaFree(gpu_voxel_grid_weight);
    cudaFree(gpu_cam_K);
    cudaFree(gpu_cam2base);
    cudaFree(gpu_depth_im);
  }

  return 0;
}
