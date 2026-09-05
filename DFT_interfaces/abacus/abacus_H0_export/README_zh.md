# ABACUS 3.11 H0Lite 单文件导出器

本目录是源码分发目录，只保留：

```text
README.md
README_zh.md
abacus-3.11.0-beta8-h0lite.patch
abacus-h0lite-v311_source.tar.gz
abacus-postprocess-v353_source.tar.gz
build_h0lite_single.sh
```

这里不保存编译好的 H0、运行库包、测试/比对脚本或完整 ABACUS 参考后端。
编译输出应放在仓库外部目录。

`abacus-h0lite-v311_source.tar.gz` 已包含应用修改后的 H0Lite 源码、所需
ABACUS 代码、独立 CMake 入口、构建脚本和许可证。下载这一个源码包即可
按下文编译，无需另行下载完整 ABACUS，也无需再次应用补丁。旧 3.5.3
源码包保留；旁边的 `.patch` 仅用于维护完整 ABACUS 开发源码树。

`abacus_h0` 是从 ABACUS 3.11 源码裁出的 H0/S0 计算程序。最终分发物只有
一个 x86_64 Linux ELF 文件，不依赖 Python、Conda、MPI、完整 ABACUS、
oneAPI 模块或随包动态库。

默认计算：

```text
H0 = T + Vnl
S0 = NAO overlap
SCF iterations = 0
diagonalizations = 0
```

加 `--with-vl` 后计算 `H0 = T + Vnl + Vl`。其中 `Vl` 是不依赖自洽电荷
密度的局域离子赝势矩阵，直接合并进 `h0rs1_nao.csr`，不会另写 Vl 文件。
Hartree、交换关联、DFT+U、EXX、DeePKS 等密度或电子态相关项始终不进入
H0。

## 输入要求

case 目录沿用普通 ABACUS LCAO SCF 输入：

```text
CASE/
├── INPUT
├── STRU
├── *.upf
└── *.orb
```

STRU、赝势和轨道也可由 INPUT 中的 `stru_file`、`pseudo_dir`、
`orbital_dir` 指定。当前严格支持 `basis_type=lcao`、`gamma_only=0`、标量
`nspin=1`、无 SOC 和守恒赝势。

程序使用 ABACUS 3.11 自身的 INPUT/STRU/UPF/轨道解析和数值积分代码。旧
3.10 INPUT 中已取消注册的 `out_interval` 会被删除，其余参数保持不变并
保存为 `INPUT.resolved`。`KPT` 不参与实空间 H0/S0 构造。CSR 精度来自
INPUT 的 `out_hsr`，旧参数 `out_mat_hs2` 仍按 3.11 别名规则解析。3.11
CSR 自带晶格、元素和坐标信息，图数据生成不需要另外读取 STRU。

## 单 case

```bash
cd /path/to/CASE
/path/to/abacus_h0
```

也可传入 case 路径：

```bash
/path/to/abacus_h0 /path/to/CASE
/path/to/abacus_h0 /path/to/CASE --with-vl
```

单 case 不需要指定核数：

- Slurm 内优先使用 `SLURM_CPUS_PER_TASK`；
- Slurm 外使用当前进程 CPU 亲和范围内的可用核数；
- 只有需要覆盖默认值时才传 `--cpus-per-task N`。

INPUT 中 `out_mat_h_vl 1` 也会启用 `--with-vl` 的相同模式。

## 单节点批量

`cases.txt` 每行一个 case。相对路径以列表文件所在目录为基准，空行和
`#` 注释会被忽略：

```text
0001
0002
/absolute/path/to/0003
```

一个 Slurm task 分到 32 核，同时计算 8 个 case、每个 case 4 核：

```bash
/path/to/abacus_h0 \
  --case-list /path/to/cases.txt \
  --tasks 8 \
  --cpus-per-task 4
```

- `--tasks` 是当前 `abacus_h0` 进程所在节点上同时运行的 case 数。
- `--cpus-per-task` 是每个 case 使用的 OpenMP 核数。
- CPU 总需求为 `tasks × cpus-per-task`，不得超过当前 Slurm task 的分配。
- 批量时若不传 `--cpus-per-task`，按 `可用核数 / tasks` 自动分配。

每个 case 在独立子进程中计算。父进程只加载一次程序并维护动态任务
队列，避免为每个 case 反复启动 Python、动态链接器和完整 ABACUS。

## Slurm 跨节点自动分片

程序不使用 MPI 通信，但读取 Slurm rank 自动切分同一份 case 列表。
推荐每个节点启动一个 Slurm task：

```bash
#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail
srun /path/to/abacus_h0 \
  --case-list /path/to/cases.txt \
  --tasks 8 \
  --cpus-per-task 4
```

每个 Slurm task 根据 `SLURM_PROCID/SLURM_NTASKS` 获得确定性分片，再在
本节点动态调度分片内的 case。Slurm job array 也可通过
`SLURM_ARRAY_TASK_ID/MIN/MAX/STEP` 自动分片。脱离 Slurm 时可显式指定：

```bash
/path/to/abacus_h0 --case-list cases.txt \
  --shard-index 0 --shard-count 2 --tasks 8 --cpus-per-task 4
```

## 输出、跳过和失败行为

输出与普通自洽 H 矩阵位于同一个目录：

```text
CASE/OUT.ABACUS/
├── h0rs1_nao.csr
├── s0r_nao.csr
├── get_H0-v311-strict.json
└── get_H0-v311-h0lite-provenance/<run-id>/
```

完成标记最后写入，记录输入、依赖、有效和解析后 INPUT、算符契约、核数、
分片及输出大小。默认不计算或验证哈希，也不预先完整解析 CSR；实际读取
或计算失败时再诊断相关文件。

同一模式下，完整标记和两个普通非空输出都存在时自动跳过。缺少必需输入、
已有未标记 H0/S0、标记后端或 Vl 模式不一致、必需输出为链接/空文件时
直接报错且不覆盖。批量模式会继续完成其他 case，末尾汇总失败并返回非零；
修复失败 case 后直接重跑，已完成 case 会跳过。

## 图数据生成

H0 与普通 SCF 的 H 必须同处 `CASE/OUT.ABACUS`。沿用 3.10 的图数据入口；
读取器自动区分 3.10/3.11，并对 3.11 采用 CSR 内嵌结构：

```bash
python DFT_interfaces/abacus/graph_data_gen_abacus.py --help
```

图生成直接按 3.10/3.11 文件名选择同目录的 H、H0 和 S0，不需要额外的
矩阵契约模块。H0Lite 自己仍保留完成标记，用于计算追溯和安全重跑。

## 直接部署

此前验证的单文件以 CentOS 7 / glibc 2.17 为基线构建。自行编译时，
目标服务器的 glibc 版本应不低于构建机；若要兼容 CentOS 7，需在
CentOS 7 或相同 glibc 基线的环境构建：

```bash
chmod +x abacus_h0
./abacus_h0 --version
./abacus_h0 --license
./abacus_h0 /path/to/CASE
```

它不是 ARM、macOS 或 Windows 原生程序。`--license` 输出嵌入 ELF 中的
ABACUS、GCC runtime 和 oneMKL 许可证及第三方声明。

## 从源码构建

环境要求：x86_64 Linux、GCC 9+（推荐 10.2+）、CMake 3.20+、GNU Make、
binutils、`file` 和 oneMKL 静态开发库（推荐 2023.2）。GCC 安装需包含
静态 `libstdc++`、`libgcc`、`libgomp.a`；`MKLROOT` 下需有 `include/mkl.h`、
`include/fftw/fftw3.h`，以及 `lib/intel64` 或 `lib` 中的
`libmkl_gf_lp64.a`、`libmkl_sequential.a`、`libmkl_core.a`。
这些编译工具和数学库由构建环境提供，不包含在源码包中；环境就绪后，
构建过程不访问网络。不需要 MPI、ELPA、Libxc、RapidJSON、Python 或 Conda。

本源码包已在 GCC 10.5.0、CMake 3.31.7、oneMKL 2023.2.0、glibc 2.28
环境从新解压目录完成编译，并运行通过默认 H0 和 `--with-vl` 两种模式。
生成文件仅动态依赖 glibc 基础组件，MKL/C++/OpenMP 运行库均静态链接。

1. 下载本目录中的 `abacus-h0lite-v311_source.tar.gz`，放到仓库外的工作
   目录并解压：

   ```bash
   tar -xzf abacus-h0lite-v311_source.tar.gz
   cd abacus-h0lite-v311_source
   ```

2. 加载编译环境并使 `MKLROOT` 有效。当前 mgt 集群使用：

   ```bash
   module load compiler/oneAPI/2024.2.1
   module load gcc/10.5.0
   export PATH=/data/src/abacus-lts-3.10/toolchain/install/cmake-3.31.7/bin:$PATH
   g++ --version
   ```

   oneAPI 模块提供 MKL，但不会替换系统的 `g++`，因此 GCC 模块也必须
   加载。上面的 CMake 路径是集群共享工具，可在计算节点使用。
   Jinsi 上对应的模块名称不同：

   ```bash
   module load compiler/gcc/10.2.0
   module load compiler/cmake/3.20.1
   module load compiler/oneapi/2023.2.0
   ```

   其他服务器按实际安装配置 `PATH` 和 `MKLROOT`，例如：

   ```bash
   export PATH=/path/to/gcc/bin:/path/to/cmake/bin:$PATH
   export MKLROOT=/path/to/oneapi/mkl/2023.2.0
   ```

3. 在已分配的计算资源上编译：

   ```bash
   BUILD_CPUS=8 bash ./build_h0lite_single.sh
   ./bin/abacus_h0 --version
   ./bin/abacus_h0 --help
   ```

默认构建目录为 `build-h0lite-single/`，可分发文件为 `bin/abacus_h0`。
Slurm 中优先使用 `SLURM_CPUS_PER_TASK` 作为编译并行数；Slurm 外使用
`BUILD_CPUS`（默认 1）。只需将生成的 `bin/abacus_h0` 复制到运行服务器。
需要自定义目录时仍支持原来的三个位置参数：

```bash
bash ./build_h0lite_single.sh . /path/to/build-h0lite /path/to/bin/abacus_h0
```

脚本会在配置前检查当前 GCC 版本，并将编译器绝对路径传给 CMake。
如果曾使用系统 GCC 8.5 配置，加载 GCC 10.5 后直接重跑即可；检测到
编译器改变时，旧 `CMakeCache.txt` 和 `CMakeFiles/` 会移入构建目录的
`compiler-cache-backup.XXXXXX/`，然后重新配置，无需手动删除源码或缓存。

源码来自 ABACUS 3.11.0-beta8 commit
`d88b719ea287e13b0e133eb57b8e16baa5361fa6`，已应用 H0Lite 修改。修改
`source/source_main/h0lite.cpp` 可调整数值调用；
`source/source_main/h0lite_frontend.cpp` 包含命令行和批量调度；原有模块
路径保持不变。编辑后再次运行同一构建命令即可增量编译。若新增 C++
编译单元，把它加入 `cmake/Sources.cmake` 对应列表。源码包只提供 H0Lite
构建目标，不用于构建完整 SCF 程序。

构建使用静态 oneMKL sequential，H0Lite 自身仍用 OpenMP；多个并发 case
不会再叠加 MKL 线程。最终 ELF 静态携带 C++、OpenMP 和数值运行库，只
动态依赖目标系统的基础 glibc 组件。生成的 `abacus_h0` 应安装或分发到
独立目录，不要复制回本源码目录。
