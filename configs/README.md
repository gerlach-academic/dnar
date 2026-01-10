Some general remarks on lr and graph sampling.

Most algorithms seem to learn fine with lr=1e-4, but mst, bfs, a_star seem to benefit from larger learning rates: 1e-3, especially a_star without a full tree (only up to goal) seems to need it (maybe even larger?)

We offer the er graph sampler just like the original paper, but for a_star we only use random geometric graphs (RGG) with 3NN connectivity as the general dataset and for planning we offer Grid (20% removed nodes) and default RGG (roadmap) graphs as well.
Sometimes RGG Graphs are called Gilbert Disc Model in literature. For Grid one can potentially tweak the removed node percentage, this is recommended up to ~40% after which graphs are likely disconnected.

Furthermore one may evaluate their models on other graph types and sizes, for that the correct config has to be used when calling eval.py, size can be changed in the cmdline args as well.
