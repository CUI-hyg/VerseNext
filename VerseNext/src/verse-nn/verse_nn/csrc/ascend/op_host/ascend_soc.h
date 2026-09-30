/*
 * 昇腾 SOC 名（算子 AddConfig 用）。
 *
 * ``OpDef::AICore().AddConfig(<soc>)`` 里的 soc 必须与算子工程的
 * ``ASCEND_COMPUTE_UNIT`` 一致，否则算子包安装后运行时找不到对应 soc 的
 * kernel 二进制。仓库里给一个默认值，``build.sh`` 会在拷贝到算子工程后
 * 用实际探测到的 SOC 覆盖本文件。
 */

#ifndef VERSE_ASCEND_SOC_H
#define VERSE_ASCEND_SOC_H

#ifndef VERSE_ASCEND_SOC
#define VERSE_ASCEND_SOC "ascend910b"
#endif

#endif  // VERSE_ASCEND_SOC_H
