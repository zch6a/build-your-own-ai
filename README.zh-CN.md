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
| 2 | [自动微分](chapters/02-autograd/README.md) | 数组上的反向模式自动求导、Adam | ✅ |
| 3 | [GPT](chapters/03-gpt/README.md) | 一个在笔记本上两分钟训练完的 decoder-only Transformer | ✅ |
| 4 | [推理](chapters/04-inference/README.md) | KV 缓存（快 10 倍）、temperature、top-k / top-p、流式输出 | ✅ |
| 5 | [向量数据库](chapters/05-vectordb/README.md) | 先是精确搜索，再是 HNSW 索引，并测量召回率——以及向量里到底装了什么 | ✅ |
| 6 | [RAG](chapters/06-rag/README.md) | 切块、BM25、稠密（LSA）与混合检索、会引用出处或说“不知道”的回答——全部有测量 | ✅ |
| 7 | [Agent](chapters/07-agent/README.md) | 一个调用工具的循环：参数校验、错误回传、步数上限、沙箱工具；任何 OpenAI 兼容模型（包括 DeepSeek）都能直接接上 | ✅ |
| 8 | [MCP](chapters/08-mcp/README.md) | 一对基于 stdio 的 Model Context Protocol 服务端和客户端；Agent 的工具可以在 Claude Code 或任何 MCP 客户端里使用 | ✅ |

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
