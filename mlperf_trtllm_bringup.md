# MLPerf TRT-LLM Bring-up

Qwen3.5-35B-A3B is used only as a smaller proxy for the Rubin bring up. It shares the same architecture as the MLPerf target model - Qwen3.5 397B.

## Pre-requisite
- All training data/SIFs have been moved to Rubin cluster.


## Branches

- **NeMo-RL:** [`hchings/RL:trtllm-agentic-swe-mlperf`](https://github.com/hchings/RL/tree/trtllm-agentic-swe-mlperf).
  — It layers TRTLLM SWE MR [#3130](https://github.com/NVIDIA-NeMo/RL/pull/3130) on Michal's MLPerf fork. Note that we'll formalize this branch into Nemo-RL repo side branch once branch base is aligned, which shouldn't be a blocker for Rubin bring up at this stage.
- **TensorRT-LLM:** `1.3.0rc21` plus [PR #16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642) for Qwen3.5 refit.
- **[Optimized repo](https://gitlab-master.nvidia.com/dl/mlperf/optimized):** `main` plus these tentative 35B scripts in [fork](https://gitlab-master.nvidia.com/erinh/optimized/-/tree/erinh/mlperf-trtllm-bringup-scripts?ref_type=heads):
  - [optimized/qwen35_397b_grpo/pytorch/config_GB200_4x4_t2g2_tp2pp1ep4gtp4_trtllm.sh](https://gitlab-master.nvidia.com/erinh/optimized/-/blob/erinh/mlperf-trtllm-bringup-scripts/qwen35_397b_grpo/pytorch/config_GB200_4x4_t2g2_tp2pp1ep4gtp4_trtllm.sh?ref_type=heads)
  - [optimized/qwen35_397b_grpo/pytorch/conf/grpo_qwen35_35b_a3b_swe_openhands_async_trtllm.yaml](https://gitlab-master.nvidia.com/erinh/optimized/-/blob/erinh/mlperf-trtllm-bringup-scripts/qwen35_397b_grpo/pytorch/conf/grpo_qwen35_35b_a3b_swe_openhands_async_trtllm.yaml?ref_type=heads)

Update the W&B project/name and all user-specific model, data, log, and
checkpoint paths before running.

## Environment

Please refer to this [Dockerfile](https://github.com/hchings/RL/blob/trtllm-agentic-swe-mlperf/docker/Dockerfile) to build an image. 
In [tools/build-custom-trtllm.sh L105](https://github.com/hchings/RL/blob/trtllm-agentic-swe-mlperf/tools/build-custom-trtllm.sh#L105) the Dockerfile used, change it to include Rubin. Since we've never built Nemo-RL + TRTLLM on Rubin, please review these build scripts and make HW-specific changes as needed.

The image includes TRT-LLM. For editable TRT-LLM, first build its C++ components
and package in a separate job because the build mutates the driver environment:

```bash
TRTLLM_SRC=/path/to/TensorRT-LLM bash tools/build-editable-trtllm.sh
```

Then set:

```bash
export NRL_TRTLLM_EDITABLE=/path/to/TensorRT-LLM
```

This requires the NeMo-RL editable-TRT-LLM support change and rebuilds the
TRT-LLM virtual environment from the prebuilt editable package.


Image for Blackwell on Lyris cluster:
- `master.nvidia.com:5005/shuyix/docker-images:nemo-rl-trtllm-20260722-aarch64-mlperf-cudnnfix`


## Run scripts

Please ensure all file paths used in [run_qwen35_35b_trtllm_4n.sh](https://github.com/hchings/RL/blob/trtllm-agentic-swe-mlperf/run_qwen35_35b_trtllm_4n.sh) match your cluster's paths. 

This script is only tested on 4 nodes X GB200/GB300. For Rubin, it might only need 2 nodes or less. 
Please adjust the parallel config in it correspondingly.

```
bash run_qwen35_35b_trtllm_4n.sh
```

## Repro validation & reference Wandbs
- Look out for any file not found / apptainer / Gym / r2egym errors in your output log. If you see any of this it means your image or uv has issues.
- You should see `reward/mean` at step 0 to be around `0.1` and trending upward as the training goes.
- Example runs for TRT-LLM TP4/TP8 and vLLM on Blackwell are below. See runs with `tp4-gb200-0722-erinh` in [wandb](https://wandb.ai/nvidia/grpo-dev-erinh/workspace?nw=nwusererinh).

---

Contact @Erin Ho @Shuyi Xiong @Chunwei Yan for questions/issues.

