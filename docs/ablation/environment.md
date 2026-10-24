# 本次 A100 原生环境

原 README 的 Docker 路径保持有效；本次远端没有 Docker socket 权限，因此使用独立 Python 环境。以下是已执行步骤的整理，不宣称在第二台干净机器上做过整套安装复验。

固定版本：Isaac Lab `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`、Newton `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3`、Python 3.12.14、PyTorch 2.11.0+cu128、TorchVision 0.26.0+cu128、Warp 1.17.0、A100 PCIe 40GB、驱动570.133.20。完整包清单随实验交付，不把可编辑安装路径误当成可移植的 requirements 文件。

在独立目录设置 `UV_PROJECT_ENVIRONMENT` 与 `UV_CACHE_DIR`，对固定 Isaac Lab 源码运行 `uv sync --frozen --inexact --extra importers --extra test --python 3.12`。本次 Newton 的远程依赖获取失败后，保留日志，用 `--no-install-package newton` 完成其余依赖，再从已校验的固定 Newton checkout 安装：

```bash
# 在 IsaacLab 项目外执行，避免再次应用其 tool.uv.sources Git 地址。
uv pip install --no-sources --python "$ENV/bin/python" --no-deps "$NEWTON_CHECKOUT"
```

Isaac Lab 锁文件默认 Torch CUDA13 构建不能在本机驱动上运行。本次只替换这个新隔离环境内的同版本 Torch 构建；使用官方 wheel 并验证 SHA，不更改驱动或共享环境：

```bash
uv pip install --no-sources --python "$ENV/bin/python" \
 'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997' \
 'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d'
```

数据固定到HF revision `ce1572906efe6157840e8474d5a0d7aa87481e74`。复用原 `tools/urdf_to_mjcf.py` 与 `tools/mjcf_to_usd.py` 转换资产，随后执行 `tools/test_env.py --num_envs 32`、`tools/test_reset.py`。运行前设置 `ISAACSIM_ASSET_ROOT` 到已准备的 Isaac 资产根目录；本次镜像中是仓库的 `data/isaac-assets`。必须检查实际资产完整性和原生测试，不能只验证文件存在。

原生实验命令使用该环境的 Python，并设置 `OMP_NUM_THREADS=4`、`TORCHINDUCTOR_COMPILE_THREADS=4`。campaign runner 自行设置 `CUDA_VISIBLE_DEVICES`、独立 Torch/Warp cache，保存源代码、资产、motion、包版本和GPU身份。进程首次编译时间计入成本；独立cache不代表重置了驱动/操作系统缓存。

`uv pip check` 的7项约束冲突来自该 Isaac Lab 提交在 pyproject.toml 中的显式覆盖：NumPy、MuJoCo、USD Exchange、USD Optimize 高于部分可选包声明上限。本次不为消除提示改变物理依赖；已验证资产转换和本实验链路，其他可选模块未被资格确认。
