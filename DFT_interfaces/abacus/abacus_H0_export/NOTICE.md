# H0Lite licensing and redistribution

H0Lite is a modified subset of ABACUS 3.11.0-beta8, not an official ABACUS
release. Upstream: https://github.com/deepmodeling/abacus-develop, commit
`d88b719ea287e13b0e133eb57b8e16baa5361fa6`.

Modification date: **2026-09-07**. Package revision: `h0lite-v311-simpson-buffered-20260907`.
Modifications by HamGNN H0Lite contributors (2004huwa); copyright (C) 2026 for
their additions. Original authors' copyright and license notices are retained.
ABACUS is licensed under LGPLv3; the new H0Lite code is LGPL-3.0-or-later.
Third-party code retains its own terms. This directory's notices do not
relicense the rest of HamGNN or the unchanged historical 3.5.3 source archive.

The changes add an H0/S0-only frontend, optional local ionic potential,
SCF-compatible input parsing, batch/Slurm scheduling, and standalone static
builds. `SOURCE_INFO.md` inside the archive lists modified upstream files.
The optional MemCheckDeluxe debug header is excluded from the trimmed package:
it is unused by H0Lite and its referenced complete license was unavailable.

Numerical integration uses the ABACUS 3.11 finite-grid Simpson implementation.
Changes add an explicitly selected H0 table mode for historical distance-grid,
cutoff and four-point interpolation conventions, and parallelize independent
direct-transform rows. The historical scalar diagonal-projector convention is
retained in the H0 frontend. Imported `ORB_gen_tables`, `ORB_table_phi` and
`ORB_table_beta` modules from the previous revision are no longer included.
The separate historical 3.5.3 archive remains unchanged and is not compiled
into H0Lite. Original 3.11 notices and full GPLv3/LGPLv3 texts are retained.

The 2026-09-07 changes additionally buffer CSR line breaks, avoid read-only
atom-pair copies in the existing 3.11 writer, and record separate H0/S0 write
timers. Numerical kernels and the CSR format/precision are unchanged.

## Source and binary distribution

- Keep `COPYING` (full GPLv3), `COPYING.LESSER` (full LGPLv3), copyright notices,
  and dated modification notices. LGPLv3 incorporates GPLv3; both are supplied.
- Runtime deployment needs only one `abacus_h0` ELF, but binary redistribution
  also requires the corresponding source and license materials. An appropriate
  way for a download distribution is to offer the **exact source used to build
  that binary**, including further modifications, source manifests, and build
  scripts, alongside it with equivalent access from the same download location.
  Keep the matching archive, build command/options, and compiler/oneMKL versions;
  provide installation information where required by the licenses.
- Do not restrict modification or rebuilding/relinking of the covered code,
  or reverse engineering for debugging those modifications. The complete
  H0Lite source and build scripts enable rebuilding this static executable;
  an unversioned upstream URL alone is not the matching source of a modified
  binary. Source need not be installed on the execution node.
- `abacus_h0 --license` embeds GPLv3, LGPLv3, project attribution/source access,
  the GCC Runtime Library Exception 3.1, the retained BLACS connector notice,
  and the actual linked oneMKL installation's license/third-party notices.
  This command is not a replacement for providing corresponding source.
- GCC runtimes are used under GPLv3 with the GCC Runtime Library Exception 3.1.
  oneMKL uses Intel's separate license and is not LGPL-relicensed. It and GCC
  are external build prerequisites, not libraries bundled in this source tar.
  The build requires the license files from the same oneMKL installation; it
  will not silently substitute notices from another version.

There is no warranty; consult the supplied license texts. These packaging
instructions address the identified distribution gaps, not every possible
downstream licensing scenario. Further modifications, proprietary combinations,
or restricted devices require their own assessment.

## 中文说明

“单文件”仅指运行时只需 `abacus_h0`，不表示可免除源码和许可证分发义务。
对外提供二进制下载时，应在同一下载位置等同提供该二进制实际对应的完整
修改源码、构建脚本和许可材料，注明编译环境及修改日期；不要限制用户对
相关代码的修改、重编译/重新链接，以及为调试修改而进行的逆向工程。
源码不必安装在计算节点上。若继续修改源码后分发，应更新修改说明和对应
源码包，不能仅指向原始 ABACUS 或不断变化的开发分支。

本次补齐 GPL/LGPL 全文和修改声明，保留原作者及第三方声明，移除未使用
且缺少完整许可材料的可选调试头文件；构建时从实际使用的 oneMKL 安装中
读取许可并嵌入 `--license`。历史 3.5.3 源码包保持不变，本说明不是对它的
重新授权或完整合规审计。
