<div align="center">

# 🧱 从零构建 AI

**分词器、自动微分、GPT、推理、向量数据库、RAG、Agent 和 MCP——每一样都从零写出来。**

小而可读、带测试的模块，在笔记本 CPU 上就能跑。不需要 API key，不需要 GPU，不需要下载模型。

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

---

大多数讲 LLM 系统原理的材料，要么停在一张示意图，要么第一步就是 `pip install` 要讲的那个东西。这里把每一块都写出来——几百行 Python，配上规定它必须做到什么的测试，和几秒钟就能跑完的演示——让你能读、能改、也能把它弄坏。

各章环环相扣：第 1 章的分词器喂给第 3 章的 GPT，GPT 的采样循环就是第 4 章，它的向量填进第 5 章的向量数据库，一直到最后一个通过你自己写的 Model Context Protocol 连接调用工具的 Agent。

## 章节

| | 章节 | 你会写出 | 状态 |
|---|---|---|---|
| 1 | [分词器](chapters/01-tokenizer/README.md) | 字节级 BPE：训练、编码、特殊 token | ✅ |
| 2 | 自动微分 | 数组上的反向模式自动求导 | 计划中 |
| 3 | GPT | 一个在笔记本上训练的 decoder-only Transformer | 计划中 |
| 4 | 推理 | temperature、top-k / top-p 采样、KV 缓存 | 计划中 |
| 5 | 向量数据库 | 先是精确搜索，再是 HNSW 索引，并测量召回率 | 计划中 |
| 6 | RAG | 切块、BM25、混合检索、评测 | 计划中 |
| 7 | Agent | 一个调用工具的循环，模型可以替换 | 计划中 |
| 8 | MCP | 一对基于 stdio 的 Model Context Protocol 服务端和客户端 | 计划中 |

## 运行

```bash
git clone <本仓库> && cd build-your-own-ai
uv sync                                        # Python 3.10+；唯一的依赖是 numpy
uv run pytest                                  # 所有章节的测试
uv run python chapters/01-tokenizer/run.py     # 某一章的演示
```

## 原则

- **从零开始，但不是玩具。** 每个模块都是在小规模上做真东西——像 GPT 分词器那样的字节级 BPE，而不是一张字符表——并说明生产系统在哪里不同。
- **哪里都能跑。** 唯一的依赖是 numpy。训练数据由 [`byoai/corpus.py`](byoai/corpus.py) 按种子生成，每台机器看到的文本都一样，不需要下载，也没有版权问题。
- **有测试。** 测试也是讲解的一部分：它们写明了每一块必须保证什么（无损往返、无法被注入的特殊 token、确定性的训练）。

## 许可证

[MIT](LICENSE)
