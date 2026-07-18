Computational efficiency
====

To systematically and quantitatively evaluate the computational efficiency, hardware resource consumption, and runtime characteristics of all batch-removal algorithms involved in the benchmark, we established a standardized, unified profiling pipeline. This framework standardized the hardware operating environment, timing initialization rules and peak memory recording protocols across all compared methods, thereby reducing systematic biases caused by inconsistent experimental configurations. 

5.1 Computational Hardware Configuration
----

All model-training and data-processing tasks, except PRECAST, were executed with automatic device allocation controlled by PyTorch, where computation defaulted to an NVIDIA CUDA-enabled GPU (cuda:0) if available; otherwise, the CPU serves as the fallback computing unit. Real-time random access memory (RAM) consumption of the main Python process was tracked via the psutil library, which retrieves resident set size (RSS) memory usage of the active process and converts the unit from bytes to megabytes and gigabytes for standardized recording. All data preprocessing, model training, clustering and output writing steps ran in a single-threaded execution mode without parallel worker initialization. No explicit operating system parameters were configured, and all file I/O and CUDA memory management operations relied on the host server’s native Linux operating system environment.

5.2 Runtime Tracking
----

The runtime timer was initialized immediately before model instantiation and launched post garbage collection and CUDA cache clearance to eliminate transient memory overhead interference; the timestamp was captured again right after the completion of the full model.train() graph embedding training procedure. The total training duration was calculated as the raw time difference in seconds, which was further converted to minutes and hours for multi-scale efficiency comparison. Timing metrics only cover the core embedding training phase and exclude all preceding raw data loading, sample concatenation, normalization and subsampling steps, as well as subsequent clustering, metadata remapping and final H5AD/JSON file storage operations.

5.3 Peak Memory Recording Strategy
----

Baseline RAM consumption of the main process was captured immediately before training initialization using a custom memory query function wrapping psutil.Process().memory_info().rss, and the post-training memory footprint was sampled immediately after embedding generation finished. The net memory overhead consumed exclusively by model training was quantified by subtracting the pre-training baseline value from the post-training memory reading, with results simultaneously stored in megabytes and gigabytes within the benchmark log file. Automatic garbage collection and CUDA cache purging were manually triggered before baseline memory sampling to clear residual temporary tensors and intermediate expression matrices and avoid overestimating training memory usage.

All profiling values were packaged into a structured dictionary and exported as JSON output to standardize post-hoc statistical comparison of runtime and memory efficiency across multiple spatial integration algorithms.
