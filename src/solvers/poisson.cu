
template <typename T>
static __device__ void kernel(T* A, const T* w, const T* coordinate_dofs)
{
  // Quadrature rules
  static const T weights_421[1] = {0.1666666666666667};
  // Precomputed values of basis functions and precomputations
  // FE* dimensions: [permutation][entities][points][dofs]
  static const T FE8_C0_D100_Q421[4] = {-1.0, 1.0, 0.0, 0.0};
  static const T FE9_C1_D010_Q421[4] = {-1.0, 0.0, 1.0, 0.0};
  static const T FE9_C2_D001_Q421[4] = {-1.0, 0.0, 0.0, 1.0};
  // Quadrature loop independent computations for quadrature rule 421
  T J_c4 = 0.0;
  T J_c8 = 0.0;
  T J_c5 = 0.0;
  T J_c7 = 0.0;
  T J_c0 = 0.0;
  T J_c1 = 0.0;
  T J_c6 = 0.0;
  T J_c3 = 0.0;
  T J_c2 = 0.0;
  T w0_d100 = 0.0;
  T w0_d010 = 0.0;
  T w0_d001 = 0.0;
  for (int ic = 0; ic < 4; ++ic)
  {
    J_c4 += coordinate_dofs[ic * 3 + 1] * FE9_C1_D010_Q421[ic];
    J_c8 += coordinate_dofs[ic * 3 + 2] * FE9_C2_D001_Q421[ic];
    J_c5 += coordinate_dofs[ic * 3 + 1] * FE9_C2_D001_Q421[ic];
    J_c7 += coordinate_dofs[ic * 3 + 2] * FE9_C1_D010_Q421[ic];
    J_c0 += coordinate_dofs[ic * 3] * FE8_C0_D100_Q421[ic];
    J_c1 += coordinate_dofs[ic * 3] * FE9_C1_D010_Q421[ic];
    J_c6 += coordinate_dofs[ic * 3 + 2] * FE8_C0_D100_Q421[ic];
    J_c3 += coordinate_dofs[ic * 3 + 1] * FE8_C0_D100_Q421[ic];
    J_c2 += coordinate_dofs[ic * 3] * FE9_C2_D001_Q421[ic];
    w0_d100 += w[ic] * FE8_C0_D100_Q421[ic];
    w0_d010 += w[ic] * FE9_C1_D010_Q421[ic];
    w0_d001 += w[ic] * FE9_C2_D001_Q421[ic];
  }
  T sp_421[77];
  sp_421[0] = J_c4 * J_c8;
  sp_421[1] = J_c5 * J_c7;
  sp_421[2] = sp_421[0] + -1 * sp_421[1];
  sp_421[3] = J_c0 * sp_421[2];
  sp_421[4] = J_c5 * J_c6;
  sp_421[5] = J_c3 * J_c8;
  sp_421[6] = sp_421[4] + -1 * sp_421[5];
  sp_421[7] = J_c1 * sp_421[6];
  sp_421[8] = sp_421[3] + sp_421[7];
  sp_421[9] = J_c3 * J_c7;
  sp_421[10] = J_c4 * J_c6;
  sp_421[11] = sp_421[9] + -1 * sp_421[10];
  sp_421[12] = J_c2 * sp_421[11];
  sp_421[13] = sp_421[8] + sp_421[12];
  sp_421[14] = sp_421[2] / sp_421[13];
  sp_421[15] = J_c3 * (-1 * J_c8);
  sp_421[16] = sp_421[4] + sp_421[15];
  sp_421[17] = sp_421[16] / sp_421[13];
  sp_421[18] = sp_421[11] / sp_421[13];
  sp_421[19] = w0_d100 * sp_421[14];
  sp_421[20] = w0_d010 * sp_421[17];
  sp_421[21] = sp_421[19] + sp_421[20];
  sp_421[22] = w0_d001 * sp_421[18];
  sp_421[23] = sp_421[21] + sp_421[22];
  sp_421[24] = sp_421[23] * sp_421[14];
  sp_421[25] = sp_421[23] * sp_421[17];
  sp_421[26] = sp_421[23] * sp_421[18];
  sp_421[27] = J_c2 * J_c7;
  sp_421[28] = J_c8 * (-1 * J_c1);
  sp_421[29] = sp_421[27] + sp_421[28];
  sp_421[30] = sp_421[29] / sp_421[13];
  sp_421[31] = J_c0 * J_c8;
  sp_421[32] = J_c6 * (-1 * J_c2);
  sp_421[33] = sp_421[31] + sp_421[32];
  sp_421[34] = sp_421[33] / sp_421[13];
  sp_421[35] = J_c1 * J_c6;
  sp_421[36] = J_c0 * J_c7;
  sp_421[37] = sp_421[35] + -1 * sp_421[36];
  sp_421[38] = sp_421[37] / sp_421[13];
  sp_421[39] = w0_d100 * sp_421[30];
  sp_421[40] = w0_d010 * sp_421[34];
  sp_421[41] = sp_421[39] + sp_421[40];
  sp_421[42] = w0_d001 * sp_421[38];
  sp_421[43] = sp_421[41] + sp_421[42];
  sp_421[44] = sp_421[43] * sp_421[30];
  sp_421[45] = sp_421[43] * sp_421[34];
  sp_421[46] = sp_421[43] * sp_421[38];
  sp_421[47] = sp_421[44] + sp_421[24];
  sp_421[48] = sp_421[45] + sp_421[25];
  sp_421[49] = sp_421[26] + sp_421[46];
  sp_421[50] = J_c1 * J_c5;
  sp_421[51] = J_c2 * J_c4;
  sp_421[52] = sp_421[50] + -1 * sp_421[51];
  sp_421[53] = sp_421[52] / sp_421[13];
  sp_421[54] = J_c2 * J_c3;
  sp_421[55] = J_c0 * J_c5;
  sp_421[56] = sp_421[54] + -1 * sp_421[55];
  sp_421[57] = sp_421[56] / sp_421[13];
  sp_421[58] = J_c0 * J_c4;
  sp_421[59] = J_c1 * J_c3;
  sp_421[60] = sp_421[58] + -1 * sp_421[59];
  sp_421[61] = sp_421[60] / sp_421[13];
  sp_421[62] = w0_d100 * sp_421[53];
  sp_421[63] = w0_d010 * sp_421[57];
  sp_421[64] = sp_421[62] + sp_421[63];
  sp_421[65] = w0_d001 * sp_421[61];
  sp_421[66] = sp_421[64] + sp_421[65];
  sp_421[67] = sp_421[66] * sp_421[53];
  sp_421[68] = sp_421[66] * sp_421[57];
  sp_421[69] = sp_421[66] * sp_421[61];
  sp_421[70] = sp_421[47] + sp_421[67];
  sp_421[71] = sp_421[48] + sp_421[68];
  sp_421[72] = sp_421[49] + sp_421[69];
  sp_421[73] = fabs(sp_421[13]);
  sp_421[74] = sp_421[70] * sp_421[73];
  sp_421[75] = sp_421[71] * sp_421[73];
  sp_421[76] = sp_421[72] * sp_421[73];
  for (int iq = 0; iq < 1; ++iq)
  {
    const T fw0 = sp_421[74] * weights_421[iq];
    const T fw1 = sp_421[75] * weights_421[iq];
    const T fw2 = sp_421[76] * weights_421[iq];
    for (int i = 0; i < 4; ++i)
      A[i] += fw0 * FE8_C0_D100_Q421[i] + fw1 * FE9_C1_D010_Q421[i]
              + fw2 * FE9_C2_D001_Q421[i];
  }
}

template <typename T>
static __global__ void _poisson(const int N, T* A, const T* w,
                                const T* coordinate_dofs, int ndofs_cell)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    T* A_ = A + gid * ndofs_cell;
    const T* w_ = w + gid * ndofs_cell;
    const T* coords_ = coordinate_dofs + 12 * gid;
    kernel(A_, w_, coords_);
  }
}

template <typename T>
void poisson(const int N, T* A, const T* w, const T* coordinate_dofs,
             int ndofs_cell, int block_size)
{
  const int num_blocks = (N + block_size - 1) / block_size;
  dim3 dimBlock(block_size);
  dim3 dimGrid(num_blocks);
  _poisson<<<dimGrid, dimBlock>>>(N, A, w, coordinate_dofs, ndofs_cell);
  cudaDeviceSynchronize();
}

template void poisson<double>(int, double*, const double*, const double*, int,
                              int);