/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>
#include <limits>
#include "cuda_utils.h"

namespace ML {
namespace decisiontree {

/** All info pertaining to splitting a node */
template <typename DataT, typename IdxT>
struct Split {
  /** start with this as the initial gain */
  static constexpr DataT Min = std::numeric_limits<DataT>::min();

  /** threshold to compare in this node */
  DataT quesval;
  /** feature index */
  IdxT colid;
  /** best info gain on this node */
  DataT gain;
  /** number of samples in the left child */
  IdxT nLeft;

  DI init() {
    quesval = gain = Min;
    colid = -1;
    nLeft = 0;
  }

  DI Split<DataT, IdxT>& operator=(const Split<DataT, IdxT>& other) {
    quesval = other.quesval;
    colid = other.colid;
    gain = other.gain;
    nLeft = other.nLeft;
  }

  /** updates the current split if the input gain is better */
  DI void update(const Split<DataT, IdxT>& other) {
    if (other.gain > gain) *this = other;
  }

  /** reduce the split info in the warp. Best split will be with 0th lane */
  DI void warpReduce() {
    auto lane = MLCommon::laneId();
#pragma unroll
    for (int i = WarpSize / 2; i >= 1; i /= 2) {
      auto id = lane + i;
      auto qu = MLCommon::shfl(quesval, id);
      auto co = MLCommon::shfl(colid, id);
      auto ga = MLCommon::shfl(gain, id);
      auto nl = MLCommon::shfl(nLeft, id);
      Split<DataT, IdxT> tmp = {qu, co, ga, nl};
      update(tmp);
    }
  }
};  // end Split

template <typename T>
struct Pair {
  T x, y;
};

/** All info pertaining to a node in the decision tree */
template <typename DataT, typename LabelT, typename IdxT>
struct Node {
  typedef typename Node<DataT, LabelT, IdxT> NodeT;
  typedef typename Split<DataT, IdxT> SplitT;

  /** special value to represent the leaf node */
  static constexpr IdxT Leaf = static_cast<IdxT>(-1);

  /** prediction, if it is a leaf */
  LabelT prediction;
  /** parent gain */
  DataT parentGain;
  /** left child node id (right id is always +1) */
  IdxT left;
  /** range of sample rows belonging to this node */
  Pair<IdxT> range;
  /** depth of this node */
  IdxT depth;

  /**
   * @brief Makes this node as a leaf. Side effect of this is that it atomically
   *        updates the number of leaves counter
   * @param n_leaves number of leaves created in the tree so far
   * @note to be called only by one thread across all participating threadblocks
   */
  DI void makeLeaf(IdxT* n_leaves) {
    left = Leaf;
    atomicAdd(n_leaves, 1);
  }

  /**
   * @brief create left/right child nodes
   * @param n_nodes number of nodes created in current kernel launch
   * @param total_nodes total nodes created so far across all levels
   * @param nodes the list of nodes
   * @param splits split info for current node
   * @return the position of the left child node in the above list
   * @note to be called only by one thread across all participating threadblocks
   */
  DI IdxT makeChildNodes(IdxT* n_nodes, IdxT total_nodes, volatile NodeT* nodes,
                         const SplitT& split) {
    IdxT pos = atomicAdd(n_nodes, 2);
    left = total_nodes + pos;
    // left
    nodes[pos].parentGain = split.gain;
    nodes[pos].depth = depth + 1;
    nodes[pos].range = {range.x, split.nLeft};
    // right
    ++pos;
    nodes[pos].parentGain = split.gain;
    nodes[pos].depth = depth + 1;
    nodes[pos].range = {range.x + split.nLeft, range.y - split.nLeft};
    return pos;
  }
};  // end Node

/** host-memory-backed tree */
template <typename DataT, typename LabelT, typename IdxT>
struct SparseTree {
  typedef typename Node<DataT, LabelT, IdxT> NodeT;
  typedef typename Split<DataT, IdxT> SplitT;

  /** list of nodes (must be allocated using cudaMallocHost!) */
  NodeT* nodes;
  /** list of splits (must be allocated using cudaMallocHost!) */
  SplitT* splits;
  /** allocation size of this buffer (in number of Node elements) */
  IdxT len;
  /** number of nodes created so far */
  IdxT n_nodes;
  /** range of the currently worked upon nodes */
  IdxT start, end;

  /**
   * @brief assign the memory needed to store the tree nodes
   * @param _n pointer to a pinned memory region
   * @param _len max number of nodes this region can hold
   */
  void assignWorkspace(void* _n, IdxT _len) {
    nodes = reinterpret_cast<NodeT*>(_n);
    len = _len;
    splits = reinterpret_cast<SplitT*>(_n + sizeof(NodeT) * len);
  }

  /**
   * @brief Initialize the tree
   * @param nrows number of rows in the input dataset
   * @param rootGain initial gain metric
   */
  void init(IdxT nrows, DataT rootGain) {
    nodes = _n;
    len = _len;
    start = 0;
    end = n_nodes = 1;  // start with root node
    nodes[0].range = {0, nrows};
    nodes[0].parentGain = rootGain;
    nodes[0].depth = 0;
  }

  /** check whether any more nodes need to be processed or not */
  bool isOver() const { return end == n_nodes; }

  /**
   * @brief After the current batch is finished processing, update the range
   *        of nodes to be worked upon in the next batch
   * @param max_batch max number of nodes to be processed in a batch
   */
  void updateNodeRange(IdxT max_batch) {
    start = end;
    auto nodes_remaining = n_nodes - end;
    end = std::min(nodes_remaining, max_batch) + end;
  }
};

template <typename DataT, typename LabelT, typename IdxT>
struct Input {
  /** input unsampled dataset (assumed to be col-major) */
  DataT* data;
  /** input labels */
  LabelT* labels;
  /** total rows in the unsampled dataset */
  IdxT M;
  /** total cols in the unsampled dataset */
  IdxT N;
  /** number of classes (useful only in classification) */
  IdxT nclasses;
};

template <typename DataT, typename IdxT>
struct Params {
  /** minimum gain needed to split the current node */
  DataT min_gain;
  /** max depth */
  IdxT max_depth;
  /** max leaves */
  IdxT max_leaves;
  /** minimum samples in a node to consider for splitting */
  IdxT min_samples;
  /** number of quantile bins */
  IdxT nbins;
  /** row subsampling factor */
  float row_subsample;
  /** column subsampling factor */
  float col_subsample;
  /** max amount of nodes that can be processed in a given batch */
  IdxT max_batch_size;
  /** number of blocks used to parallelize column-wise computations */
  IdxT nBlksForCols;
  /** number of blocks used to parallelize row-wise computations */
  IdxT nBlksForRows;
  ///@todo: support metric enum
};

template <typename DataT, typename LabelT, typename IdxT>
__global__ void initialClassHistKernel(int* hist, const IdxT* rowids,
                                       Input<DataT, LabelT, IdxT> input,
                                       IdxT nrows) {
  extern __shared__ int* shist;
  for (IdxT i = threadIdx.x; i < input.nclasses; i += blockDim.x) shist = 0;
  __syncthreads();
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT stride = blockDim.x * gridDim.x;
  for (auto i = tid; i < nrows; i += stride) {
    auto row = rowids[i];
    auto label = input.labels[row];
    atomicAdd(shist + label, 1);
  }
  __syncthreads();
  for (IdxT i = threadIdx.x; i < input.nclasses; i += blockDim.x)
    atomicAdd(hist + i, shist[i]);
}


template <typename DataT, typename IdxT>
__global__ void initSplitKernel(Split<DataT, IdxT>* splits, IdxT batchSize) {
  IdxT tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < batchSize) splits[tid].init();
}

/**
 * @brief Helper method to have threadblocks signal completion to others
 * @param done_count location in global mem used to signal done
 * @param nBlks number of blocks involved with the done-handshake
 * @param master which block is supposed to be considered as master in this
 *               process of handshake.
 * @param smem shared mem used for 'am i last' signal propagation to all the
 *             threads in the block
 * @return true if the current threadblock is the last to arrive else false
 * @note This function should be entered by all threads in the block together.
 */
DI bool signalDone(volatile int* done_count, int nBlks, bool master,
                   void* smem) {
  if (nBlks == 1) return true;
  auto* sAmIlast = reinterpret_cast<int*>(smem);
  if (threadIdx.x == 0) {
    auto delta = master ? nBlks - 1 : -1;
    auto old = atomicAdd(done_count, delta);
    *sAmIlast = ((old + delta) == 0);
  }
  __syncthreads();
  return *sAmIlast;
}

/**
 * @brief Compute gain based on gini impurity metric
 * @param shist left/right class histograms for all bins (nbins x 2 x nclasses)
 *              After that, it also contains left/right sample counts for each
 *              of the bins (2 x nbins)
 * @param sbins quantiles for the current column (len = nbins)
 * @param parentGain parent node's best gain
 * @param sp will contain the per-thread best split so far
 * @param col current column
 * @param len total number of samples for the current node to be split
 * @param nbins number of bins
 * @param nclasses number of classes
 */
template <typename DataT, typename IdxT>
DI void giniInfoGain(const int* shist, const DataT* sbins, DataT parentGain,
                     SplitT& sp, IdxT col, IdxT len, IdxT nbins,
                     IdxT nclasses) {
  constexpr DataT One = DataT(1.0);
  DataT invlen = One / len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto gain = DataT(0.0);
    auto nLeft = shist[nbins * 2 * nclasses + i];
    auto invLeft = One / nLeft;
    auto invRight = One / shist[nbins * 2 * nclasses + nbins + i];
    for (IdxT j = 0; j < nclasses; ++j) {
      auto lval = DataT(shist[i * 2 * nclasses + j]);
      gain += lval * invLeft * lval;
      auto rval = DataT(shist[i * 2 * nclasses + nclasses + j]);
      gain += rval * invRight * rval;
    }
    gain = parentGain - (One - gain * invlen);
    Split<DataT, IdxT> tmp = {sbins[i], col, gain, nLeft};
    sp.update(tmp);
  }
}

/**
 * @brief Computes the best split across the threadblocks
 * @param s per-thread best split (thread0 will contain the best split)
 * @param smem shared mem
 * @param split current split to be updated
 * @param mutex location which provides exclusive access to node update
 * @note all threads in the block must enter this function together.
 */
template <typename DataT, typename IdxT>
DI void evalBestSplit(Split<DataT, IdxT>& s, void* smem,
                      Split<DataT, IdxT>* split, volatile int* mutex) {
  auto* sbest = reinterpret_cast<SplitT*>(smem);
  s.warpReduce();
  auto warp = threadIdx.x / MLCommon::WarpSize;
  auto nWarps = blockDim.x / MLCommon::WarpSize;
  auto lane = MLCommon::laneId();
  if (lane == 0) sbest[warp] = s;
  __syncthreads();
  if (lane < nWarps) s = sbest[lane];
  s.warpReduce();
  if (threadIdx.x == 0) {
    while(atomicCAS(mutex, 0, 1));
    split->update(s);
    __threadfence();
    *mutex = 0;
    __threadfence();
  }
  __syncthreads();
}

template <typename DataT, typename IdxT, int TPB>
DI bool leafBasedOnParams(IdxT myDepth, const Params<DataT, IdxT>& params,
                          const IdxT* n_leaves, IdxT nSamples, void* smem) {
  if (myDepth < params.max_depth) return false;
  if (nSamples >= params.min_samples) return false;
  if (*n_leaves < params.max_leaves) return false;
  return true;
}

/**
 * @brief Compute the prediction value for the current leaf node
 * @note to be called by only one block from all participating blocks
 */
///@todo: support for regression
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void computePrediction(const Pair<IdxT>& range, volatile IdxT* rowids,
                          const Input<DataT, LabelT, IdxT>& input,
                          volatile Node<DataT, LabelT, IdxT>* nodes,
                          volatile IdxT* n_leaves, void* smem) {
  typedef cub::KeyValuePair<int, int> KVP;
  typedef cub::BlockReduce<int, TPB> BlockReduceT;
  typedef typename BlockReduceT::TempStorage ReduceSMem;
  auto* shist = reinterpret_cast<int*>(smem);
  auto sreduce =
    *reinterpret_cast<ReduceSMem*>(smem + sizeof(int) * input.nclasses);
  auto tid = threadIdx.x;
  for (int i = tid; i < input.nclasses; i += blockDim.x) shist[i] = 0;
  __syncthreads();
  auto end = range.x + range.y;
  for (auto i = range.x + tid; i < end; i += blockDim.x) {
    auto label = input.labels[rowids[i]];
    atomicAdd(shist + label, 1);
  }
  __syncthreads();
  auto op = cub::ArgMax();
  KVP v = {-1, -1};
  for (int i = tid; i < input.nclasses; i += blockDim.x) {
    KVP tmp = {i, shist[i]};
    v = op(v, tmp);
  }
  v = BlockReduceT(sreduce).Reduce(v, op);
  __syncthreads();
  if (tid == 0) {
    nodes[0].makeLeaf(n_leaves);
    nodes[0].prediction = LabelT(v.key);
  }
}

/**
 * @brief Partition the samples to left/right nodes based on the best split
 * @return the position of the left child node in the nodes list. However, this
 *         value is valid only for threadIdx.x == 0.
 * @note this should be called by only one block from all participating blocks
 */
template <typename DataT, typename LabelT, typename IdxT, int TPB>
DI void partitionSamples(const Input<DataT, LabelT, IdxT>& input,
                         const Split<DataT, IdxT>* splits,
                         Node<DataT, LabelT, IdxT>* curr_nodes,
                         Node<DataT, LabelT, IdxT>* next_nodes,
                         volatile IdxT* rowids, volatile IdxT* n_nodes,
                         IdxT total_nodes, void* smem) {
  typedef cub::BlockScan<int, TPB> BlockScanT;
  typedef typename BlockScanT::TempStorage ScanSMem;
  // for scan
  size_t smemScanSize = sizeof(ScanSMem);
  auto temp1 = *reinterpret_cast<ScanSMem *>(smem);
  auto temp2 = *reinterpret_cast<ScanSMem *>(smem + smemScanSize);
  // for compaction
  size_t smemSize = sizeof(IdxT) * TPB;
  auto *lcomp = reinterpret_cast<IdxT *>(smem);
  auto *rcomp = reinterpret_cast<IdxT *>(smem + smemSize);
  auto nid = blockIdx.x;
  auto split = splits[nid];
  auto range = curr_nodes[nid].range;
  auto *col = input.data + split.colid * input.M;
  auto loffset = range.x, part = loffset + split.nLeft, roffset = part;
  auto end = range.x + range.y;
  int lflag = 0, rflag = 0, llen = 0, rlen = 0, minlen = 0;
  auto tid = threadIdx.x;
  while (loffset < part && roffset < end) {
    // find the samples in the left that belong to right and vice-versa
    auto loff = loffset + tid, roff = roffset + tid;
    if (llen == minlen)
      lflag = loff < part ? col[rowids[loff]] > split.quesval : 0;
    if (rlen == minlen)
      rflag = roff < end ? col[rowids[roff]] <= split.quesval : 0;
    // scan to compute the locations for each 'misfit' in the two partitions
    int lidx, ridx;
    BlockScanT(temp1).ExclusiveSum(lflag, lidx, llen);
    BlockScanT(temp2).ExclusiveSum(rflag, ridx, rlen);
    __syncthreads();
    minlen = llen < rlen ? llen : rlen;
    // compaction to figure out the right locations to swap
    if (lflag) lcomp[lidx] = lidx + loffset;
    if (rflag) rcomp[ridx] = ridx + roffset;
    __syncthreads();
    // reset the appropriate flags for the longer of the two
    if (lidx < minlen) lflag = 0;
    if (ridx < minlen) rflag = 0;
    if (llen == minlen) loffset += TPB;
    if (rlen == minlen) roffset += TPB;
    // swap the 'misfit's
    if (tid < minlen) {
      auto a = rowids[lcomp[tid]];
      auto b = rowids[rcomp[tid]];
      rowids[lcomp[tid]] = b;
      rowids[rcomp[tid]] = a;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    curr_nodes[nid].makeChildNodes(n_nodes, total_nodes, next_nodes,
                                   splits[nid]);
  }
}

template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void nodeSplitKernel(Params<DataT, IdxT> params,
                                Input<DataT, LabelT, IdxT> input,
                                volatile Node<DataT, LabelT, IdxT>* curr_nodes,
                                volatile Node<DataT, LabelT, IdxT>* next_nodes,
                                volatile IdxT* n_nodes, volatile IdxT* rowids,
                                const Split<DataT, IdxT>* splits,
                                volatile IdxT* n_leaves, IdxT total_nodes) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.x;
  auto node = nodes[nid];
  auto range = node.range;
  if (leafBasedOnParams<DataT, IdxT, TPB>(
        node.depth, params, n_leaves, range.y, smem) ||
      splits[nid].gain < params.min_gain) {
    computePrediction<DataT, LabelT, IdxT>(
      range, rowids, input, curr_nodes + nid, n_leaves, smem);
    return;
  }
  partitionSamples<DataT, LabelT, IdxT, TPB>(
    input, splits, curr_nodes, next_nodes, rowids, n_nodes, total_nodes, smem);
}

///@todo: support regression
template <typename DataT, typename LabelT, typename IdxT, int TPB>
__global__ void computeSplitKernel(int* hist, Params<DataT, IdxT> params,
                                   Input<DataT, LabelT, IdxT> input,
                                   const Node<DataT, LabelT, IdxT>* nodes,
                                   IdxT colStart, volatile int* done_count,
                                   volatile int* mutex, const IdxT* n_leaves,
                                   volatile IdxT* rowids,
                                   Split<DataT, IdxT>* splits) {
  extern __shared__ char smem[];
  IdxT nid = blockIdx.z;
  auto node = nodes[nid];
  auto range = node.range;
  if (leafBasedOnParams<DataT, IdxT, TPB>(
        node.depth, params, n_leaves, range.y, smem)) {
    return;
  }
  auto parentGain = node.parentGain;
  auto end = range.x + range.y;
  auto nbins = params.nbins;
  auto nclasses = input.nclasses;
  auto binSize = nbins * 2 * nclasses;
  auto len = binSize + 2 * nbins;
  auto *shist = reinterpret_cast<int*>(smem);
  auto *sbins = reinterpret_cast<DataT*>(smem + sizeof(int) * len);
  IdxT stride = blockDim.x * gridDim.x;
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto col = colids[colStart + blockIdx.y];
  if (col >= ncols) return;
  for (IdxT i = 0; i < len; i += blockDim.x) shist[i] = 0;
  for (IdxT b = 0; b < nbins; b += blockDim.x)
    sbins[b] = quantiles[col * nbins + b];
  __syncthreads();
  auto coloffset = col * input.M;
  // compute class histogram for all bins for all classes in shared mem
  for (auto i = range.x + tid; i < end; i += stride) {
    auto row = rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    for (IdxT b = 0; b < nbins; ++b) {
      auto isLeft = d <= sbins[b];  // no divergence
      auto offset = b * 2 * nclasses + isLeft * nclasses + label;
      atomicAdd(shist + offset, 1);  // class hist
      offset = binSize + isLeft * nbins + b;
      atomicAdd(shist + offset, 1);  // sample count
    }
  }
  __syncthreads();
  // update the corresponding global location
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    atomicAdd(hist + nid * len + i, shist[i]);
  }
  __syncthreads();
  // last threadblock will go ahead and compute the best split
  auto last = signalDone(done_count + nid * gridDim.y + blockIdx.y, gridDim.x,
                         blockIdx.x == 0, smem);
  if (!last) return;
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    shist[i] = hist[nid * len + i];
  }
  __syncthreads();
  Split<DataT, IdxT> sp;
  sp.init();
  ///@todo: support other metrics
  giniInfoGain<DataT, LabelT, IdxT>(shist, sbins, parentGain, sp, col, range.y,
                                    nbins, nclasses);
  evalBestSplit<DataT, IdxT>(sp, smem, splits + nid, mutex + nid);
}

/** internal state used to make progress on tree building */
template <typename DataT, typename LabelT, typename IdxT>
struct State {
  typedef typename Node<DataT, LabelT, IdxT> NodeT;
  typedef typename SparseTree<DataT, LabelT, IdxT> SpTreeT;
  typedef typename Split<DataT, IdxT> SplitT;

  /** number of sampled rows */
  IdxT nrows;
  /** number of sampled columns */
  IdxT ncols;
  /** max nodes that we can create */
  IdxT max_nodes;
  /** number of blocks used to parallelize column-wise computations */
  IdxT nBlksForCols;
  /** total number of histogram bins */
  IdxT nHistBins;
  /** DT params */
  Params<DataT, IdxT> params;
  /** training input */
  Input<DataT, LabelT, IdxT> input;
  /** will contain the final learned tree */
  SpTreeT tree;

  /** sampled row id's */
  volatile IdxT* rowids;
  /** sampled column id's */
  ///@todo: support for per-node col-subsampling
  IdxT* colids;
  /** number of nodes created in the current batch */
  IdxT* n_nodes;
  /** quantiles computed on the dataset (col-major) */
  DataT* quantiles;
  /** class histograms */
  int* hist;
  /** threadblock arrival count */
  volatile int* done_count;
  /** mutex array used for atomically updating best split */
  volatile int* mutex;
  /** number of leaves created so far */
  volatile IdxT* n_leaves;
  /** best splits for the current batch of nodes */
  SplitT* splits;
  /** current batch of nodes */
  NodeT* curr_nodes;
  /** next batch of nodes */
  NodeT* next_nodes;
  /** host copy of the number of new nodes in current branch */
  IdxT* h_n_nodes;
  /** host copy for initial histograms */
  int* h_hist;

  static constexpr bool isRegression() const {
    return std::is_same<DataT, LabelT>::value;
  }

  /**
   * @brief Computes workspace size needed for the current computation
   * @param d_wsize (in B) of the device workspace to be allocated
   * @param h_wsize (in B) of the host workspace to be allocated
   * @param p the input params
   * @param in the input data
   */
  void workspaceSize(size_t& d_wsize, size_t& h_wsize,
                     const Params<DataT, IdxT>& p,
                     const Input<DataT, LabelT, IdxT>& in) {
    ASSERT(!isRegression(), "Currently only classification is supported!");
    nrows = static_cast<IdxT>(p.row_subsample * in.M);
    ncols = static_cast<IdxT>(p.col_subsample * in.N);
    nBlksForCols = std::min(ncols, p.nBlksForCols);
    auto max_batch = params.max_batch_size;
    nHistBins = 2 * max_batch * (p.bins + 1) * nBlksForCols;
    // x3 just to be safe since we can't strictly adhere to max_leaves
    max_nodes = p.max_leaves * 3;
    params = p;
    input = in;
    d_wsize = static_cast<size_t>(0);
    d_wsize += sizeof(IdxT) * nrows;  // rowids
    d_wsize += sizeof(IdxT) * ncols;  // colids
    d_wsize += sizeof(IdxT);  // n_nodes
    d_wsize += sizeof(DataT) * p.bins * ncols;  // quantiles
    d_wsize += sizeof(int) * nHistBins;  // hist
    d_wsize += sizeof(int) * max_batch * nBlksForCols;  // done_count
    d_wsize += sizeof(int) * max_batch;  // mutex
    d_wsize += sizeof(IdxT);  // n_leaves
    d_wsize += sizeof(SplitT) * max_batch;  // splits
    d_wsize += sizeof(NodeT) * max_batch;  // curr_nodes
    d_wsize += sizeof(NodeT) * 2 * max_batch;  // next_nodes
    // all nodes in the tree
    h_wsize = sizeof(IdxT);  // h_n_nodes
    h_wsize += sizeof(int) * input.nclasses;  // h_hist
    h_wsize += (sizeof(NodeT) + sizeof(SplitT)) * max_nodes;
  }

  /**
   * @brief assign workspace to the current state
   * @param d_wspace device buffer allocated by the user for the workspace. Its
   *                 size should be atleast workspaceSize()
   * @param h_wspace pinned host buffer mainly needed to store the learned nodes
   * @param s cuda stream where to schedule work
   */
  void assignWorkspace(void* d_wspace, void* h_wspace) {
    auto max_batch = params.max_batch_size;
    // device
    rowids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT) * nrows;
    colids = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT) * ncols;
    n_nodes = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    quantiles = reinterpret_cast<DataT*>(d_wspace);
    d_wspace += sizeof(DataT) * p.bins * ncols;
    hist = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * nHistBins;
    done_count = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch * nBlksForCols;
    mutex = reinterpret_cast<int*>(d_wspace);
    d_wspace += sizeof(int) * max_batch;
    n_leaves = reinterpret_cast<IdxT*>(d_wspace);
    d_wspace += sizeof(IdxT);
    splits = reinterpret_cast<SplitT>(d_wspace);
    d_wspace += sizeof(SplitT) * max_batch;
    curr_nodes = reinterpret_cast<NodeT*>(d_wspace);
    d_wspace += sizeof(NodeT) * max_batch;
    next_nodes = reinterpret_cast<NodeT*>(d_wspace);
    // host
    h_n_nodes = reinterpret_cast<IdxT*>(h_wspace);
    h_wspace += sizeof(IdxT);
    h_hist = reinterpret_cast<int*>(h_wspace);
    h_wspace += sizeof(IdxT) * input.nclasses;
    tree.assignWorkspace(h_wspace, max_nodes);
  }

  /**
   * Main training method. This should only be called after the workspace has
   * been created and rowids and colids and such other input buffers have been
   * populated!
   */
  void train(cudaStream_t s) {
    init(s);
    while (!tree.isOver()) {
      auto new_nodes = doSplit(s);
      tree.n_nodes += new_nodes;
      tree.updateNodeRange(params.max_batch_size);
    }
  }

private:
  void init(cudaStream_t s) {
    CUDA_CHECK(cudaMemsetAsync(
                 done_count, 0, sizeof(int) * max_batch * nBlksForCols, s));
    CUDA_CHECK(cudaMemsetAsync(mutex, 0, sizeof(int) * max_batch, s));
    CUDA_CHECK(cudaMemsetAsync(n_leaves, 0, sizeof(IdxT), s));
    auto rootGain = initialMetric(s);
    tree.init(nrows, rootGain);
  }

  /**
   * Computes best split across all nodes in the current batch and splits the
   * nodes accordingly
   * @return the number of newly created nodes
   */
  IdxT doSplit(cudaStream_t s) {
    static constexpr int TPB = 256;
    static constexpr int TPB_SPLIT = 512;
    typedef cub::BlockScan<int, TPB> BlockScanT;
    typedef typename BlockScanT::TempStorage ScanSMem;
    auto nbins = params.nbins;
    auto nclasses = input.nclasses;
    auto binSize = nbins * 2 * nclasses;
    auto len = binSize + 2 * nbins;
    size_t smemSize = sizeof(int) * len + sizeof(DataT) * nbins;
    auto batchSize = tree.end - tree.start;
    auto nblks = MLCommon::ceildiv<int>(batchSize, TPB);
    // start fresh on the number of *new* nodes created in this batch
    CUDA_CHECK(cudaMemsetAsync(n_nodes, 0, sizeof(IdxT), s));
    initSplitKernel<DataT, IdxT><<<nblks, TPB, 0, s>>>(splits, batchSize);
    CUDA_CHECK(cudaGetLastError());
    // get the current set of nodes to be worked upon
    MLCommon::updateDevice(curr_nodes, tree.nodes + tree.start, batchSize, s);
    // iterate through a batch of columns (to reduce the memory pressure) and
    // compute the best split at the end
    dim3 grid(nBlksForRows, nBlksForCols, batchSize);
    for (IdxT c = 0; c < ncols; c += nBlksForCols) {
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * nHistBins, s));
      computeSplitKernel<DataT, LabelT, SplitT, TPB>
        <<<grid, TPB, smemSize, s>>>(
          hist, params, input, curr_nodes, c, done_count, mutex, n_leaves,
          rowids, splits);
      CUDA_CHECK(cudaGetLastError());
    }
    // create child nodes (or make the current ones leaf)
    smemSize = 2 * std::max(sizeof(ScanSMem), sizeof(IdxT) * TPB_SPLIT);
    nodeSplitKernel<DataT, LabelT, IdxT, TPB_SPLIT>
      <<<batchSize, TPB_SPLIT, smemSize, s>>>(
        params, input, curr_nodes, next_nodes, n_nodes, rowids, splits,
        n_leaves, tree.n_nodes);
    CUDA_CHECK(cudaGetLastError());
    // copy the best splits to host
    MLCommon::updateHost(tree.splits + tree.start, splits, batchSize, s);
    // copy the updated (due to leaf creation) and newly created child nodes
    MLCommon::updateHost(tree.nodes + tree.start, curr_nodes, batchSize, s);
    MLCommon::updateHost(h_n_nodes, n_nodes, 1, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    MLCommon::updateHost(tree.nodes + tree.n_nodes, next_nodes, *h_n_nodes, s);
    return *h_n_nodes;
  }

  /** computes the initial metric needed for root node split decision */
  DataT initialMetric(cudaStream_t s) {
    static constexpr int TPB = 256;
    static constexpr int NITEMS = 8;
    ///@todo: support for regression
    if (isRegression()) {
    } else {
      int nblks = ceildiv(nrows, TPB * NITEMS);
      size_t smemSize = sizeof(int) * input.nclasses;
      // reusing `hist` for initial bin computation only
      CUDA_CHECK(cudaMemsetAsync(hist, 0, sizeof(int) * input.nclasses, s));
      initialClassHistKernel<DataT, LabelT, IdxT>
        <<<nblks, TPB, smemSize, s>>>(hist, rowids, input, nrows);
      CUDA_CHECK(cudaGetLastError());
      MLCommon::updateHost(h_hist, hist, input.nclasses, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      // better to compute the initial metric (after class histograms) on CPU
      ///@todo: support other metrics
      auto out = DataT(1.0);
      auto invlen = out / DataT(nrows);
      for (IdxT i = 0; i < input.nclasses; ++i) {
        auto val = h_hist[i] * invlen;
        out -= val * val;
      }
    }
    return out;
  }
};  // end State

}  // end namespace decisiontree
}  // end namespace ML
