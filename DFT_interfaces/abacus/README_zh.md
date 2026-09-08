# ABACUS 图数据接口中文说明

当前入口用同一套命令处理 ABACUS 3.10 与 3.11 输出。修改集中在矩阵与
结构读取层，训练图的数据字段和 3.10 调用方式保持不变。

## 1. 目录约定

普通自洽 H 与 H0/S0 必须位于同一个 `OUT.ABACUS`：

```text
CASE/
├── INPUT
├── STRU
└── OUT.ABACUS/
    ├── running_scf.log
    ├── 普通 H(R)
    ├── H0(R)
    └── S0(R)
```

3.10 文件通常为：

```text
data-HR-sparse_SPIN0.csr
data-H0R-sparse_SPIN0.csr
data-S0R-sparse_SPIN0.csr
```

3.11 文件通常为：

```text
hrs1_nao.csr
h0rs1_nao.csr
s0r_nao.csr
```

3.10 CSR 没有完整结构头，图生成继续从 `running_scf.log` 读取结构。
3.11 HContainer CSR 内嵌晶格、元素和坐标，接口直接以 CSR 结构为准，并
核对 H、H0、S0 的结构一致性，不再要求单独读取 STRU。

图生成不要求额外的 provenance 标记模块，因此已有 3.10 数据可保持原
目录和调用方式不变。必需矩阵必须存在且非空；同一种矩阵同时出现 3.10
和 3.11 两个候选文件时会因歧义而拒绝读取。

## 2. ABACUS 3.11 普通 SCF 输出

用于图数据的 3.11 LCAO INPUT 至少应设置：

```text
calculation  scf
basis_type   lcao
gamma_only   0
nspin        1
out_hsr      1 8
```

`out_hsr` 的第二个值可按需要选 8 或 16 位。当前图接口读取文本 CSR，不
依赖 RapidJSON、CNPY 或可选 JSON/NPZ 导出，因此 RapidJSON CMake 目标
缺失不会阻塞本流程。

SCF 必须真正完成并收敛。接口会从 `running_scf.log` 检查 `Finish Time`、
收敛标记、最终总能和电子迭代数；只写出矩阵但未收敛的 case 不会接受。

## 3. 生成 H0/S0

推荐使用单文件 `abacus_h0`。默认定义与审计后的旧 3.5.3 H0 包一致：

```text
H0 = T + Vnl
S0 = NAO overlap
SCF = 0
diagonalization = 0
```

```bash
cd CASE
/path/to/abacus_h0
```

加局域赝势但仍不做 SCF：

```bash
/path/to/abacus_h0 CASE --with-vl
```

此时 `H0=T+Vnl+Vl`，Vl 直接并入 `h0rs1_nao.csr`，不单独写文件。

单 case 默认使用 Slurm 分配给当前 task 的核数，无需传核数。批量和
跨节点用法、输入契约、失败恢复、部署与源码构建见：

[abacus_H0_export/README_zh.md](abacus_H0_export/README_zh.md)

H0 导出目录只保留源码、旧 3.5.3 源码包和编译教程，不保存编译好的
`abacus_h0`。

## 4. 生成图数据

先查看完整参数：

```bash
conda run -n hamgnn-new --no-capture-output \
  python DFT_interfaces/abacus/graph_data_gen_abacus.py --help
```

单个或多个 case 使用同一入口：

```bash
conda run -n hamgnn-new --no-capture-output \
  python DFT_interfaces/abacus/graph_data_gen_abacus.py \
  --data-dirs /path/cases/0001 /path/cases/0002 \
  --graph-data-folder /path/graphs \
  --output-format lmdb \
  --num-processes 2 \
  --worker-threads 1 \
  --nao-max 27
```

主要参数：

- `--data-dirs`：case 目录；每个目录下必须有 INPUT 和统一的 OUT.ABACUS。
- `--graph-data-folder`：输出目录。
- `--output-format`：`lmdb`、`npz` 或 `both`。
- `--num-processes`：并行 worker 数；0 表示按当前 CPU 亲和范围自动选择。
- `--worker-threads`：每个 worker 内的线程数，默认 1，避免线程过量。
- `--nao-max`：原子轨道 padding 维度，可选 13、15、27、40。
- `--radius-scale`：H0 图边截断半径缩放。
- `--skip-dft-hamiltonian`：不读取普通自洽 H，将 H0 同时作为 H；
  仅用于不需要 DFT 标签的场景。
- `--overwrite`：新结果完整生成后再替换已有完整图输出。

默认 LMDB 输出为：

```text
GRAPH_DIR/
└── graph_data.lmdb/
    ├── data.mdb
    └── lock.mdb
```

图数量保存于 LMDB 的 `num_graphs` 键；启用 `--if-hamnet` 时，附加元数据
写入同一数据库的 `metadata_json` 键。只有全部 case 转换成功后才安装最终
输出，不另外写 `metadata.json` 或 `COMPLETE` 文件。

每个图的 `abacus_matrix_provenance` 保存实际 H、H0、S0 路径和各矩阵
检测到的 3.10/3.11 CSR 格式。程序不主动做哈希或全文件损坏扫描；CSR
在实际解析时发现问题才针对相关输入报错。

## 5. H 与 H0 的物理含义

默认 H0：

```text
T + Vnl
```

可选 H0：

```text
T + Vnl + Vl
```

普通自洽 H 还包含由收敛电子密度确定的 Hartree、交换关联以及 INPUT
启用的其他电子态相关项。训练时的差值含义取决于 H0 是否启用 Vl，数据
准备时必须明确区分，不能混用。

## 6. 常见错误

- 缺少 H0/S0：重新运行相应版本的 H0 导出程序，不要复制其他矩阵顶替。
- H/H0/S0 结构不一致：确认三者来自同一 case；3.11 以 CSR 内嵌结构为准。
- 找不到 UPF/轨道：检查 INPUT 的 `pseudo_dir`、`orbital_dir` 与 STRU 声明。
- 已有未标记 H0/S0：程序为防止覆盖来源不明结果会拒绝运行，应先人工
  确认其来源并移出目标目录。
- `torch_scatter` 等导入失败：使用项目的 `hamgnn-new` 环境，而不是基础
  Python 环境。
