# RAG 本地文档问答助手

这是一个适合大二学生做课程设计、竞赛作品和 AI/大模型实习面试展示的轻量级 RAG 项目。它支持本地上传 PDF/TXT 文档，完成文本清洗、切分、去重、FAISS 向量化存储，并通过 Streamlit 提供网页问答界面。

## 1. 技术栈

- Python 3.9+
- LangChain
- FAISS
- Streamlit
- PyPDF2
- 开源 Embedding 模型（默认：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`）
- CPU 本地运行，无需 GPU

## 2. 项目目录结构

```text
RAG-based Local Document Q&A Assistant/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── uploads/
│   └── vector_store/
└── rag_local_qa/
    ├── __init__.py
    ├── config.py
    ├── document_processor.py
    ├── vector_store.py
    └── qa_engine.py
```

## 3. 功能模块说明

### 3.1 文档上传

- 支持 PDF、TXT 批量上传
- 上传文件会保存到 `data/uploads/`

### 3.2 文档处理

- PDF 使用 `PyPDF2` 提取文本
- TXT 自动尝试 UTF-8 / GBK / GB18030 编码
- 自动清洗空白字符
- 自动删除重复行
- 使用 LangChain 的 `RecursiveCharacterTextSplitter` 进行文本切分
- 对重复 chunk 做哈希去重

### 3.3 向量存储

- 使用开源 Embedding 模型将 chunk 向量化
- 使用 FAISS 建立本地向量索引
- 索引持久化保存到 `data/vector_store/`

### 3.4 问答检索

问答流程如下：

1. 用户输入问题
2. 系统将问题向量化
3. 在 FAISS 中检索最相关文本块
4. 对检索结果做轻量级答案组织
5. 返回回答，并展示来源片段

### 3.5 前端界面

- 基于 Streamlit
- 支持上传文档、重建知识库、清空知识库
- 支持多轮问答
- 支持查看来源文档和检索到的 chunk

## 4. 启动步骤

### 4.1 创建虚拟环境

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

### 4.3 启动项目

```bash
streamlit run app.py
```

### 4.4 使用流程

1. 打开浏览器中的 Streamlit 页面
2. 在左侧上传一个或多个 PDF / TXT 文件
3. 点击“导入并重建知识库”
4. 等待向量索引构建完成
5. 在聊天框输入问题并查看答案

## 5. 可选配置

可以通过环境变量调整切分和检索参数：

```powershell
$env:CHUNK_SIZE=800
$env:CHUNK_OVERLAP=150
$env:TOP_K=5
$env:EMBEDDING_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
streamlit run app.py
```

## 6. 适合作为实习项目的亮点

- 模块化结构清晰，容易扩展为更完整的 RAG 系统
- 具备上传、预处理、向量化、检索、问答、界面展示完整闭环
- 本地运行，无需训练模型，无需 GPU，适合学生演示和面试讲解
- 后续可以继续扩展：
  - 接入本地 LLM（如 Ollama）
  - 支持 Markdown / Word / 网页解析
  - 增加多轮对话记忆
  - 增加重排序（Rerank）模块

## 7. 面试时可以怎么介绍

你可以这样介绍这个项目：

> 我独立实现了一个本地 RAG 文档问答助手，支持 PDF/TXT 批量上传、文本清洗切分、FAISS 向量检索和 Streamlit 可视化问答界面。项目重点在于把完整 RAG 流程跑通，并保证它可以在普通 CPU 环境下运行，便于快速展示和二次扩展。

如果你接下来还想继续升级，我建议优先加这三个点：

1. 接入本地大模型生成更自然的答案
2. 增加命中 chunk 高亮和引用定位
3. 增加“知识库管理”和“检索评估”模块
