# Elasticsearch 混合检索 RAG 系统实现

本项目实现了一个基于 Elasticsearch 的检索增强生成（RAG）系统，特点是结合了语义相似度和关键词匹配的混合检索能力。

## 主要特性

- 使用 LlamaIndex 进行文档处理和向量化
- 结合向量相似度和关键词匹配的混合搜索
- 可配置的语义搜索和关键词搜索权重
- 通过 SimpleDirectoryReader 支持多种文档格式
- 使用 Elasticsearch 进行高效的向量存储和检索
- 支持 JSON 格式数据的导入和处理

## 环境要求

- Python 3.8+
- Elasticsearch 8.x（本地运行或可访问的远程服务）
- 自定义大模型 API

## 安装步骤

1. 安装 Elasticsearch：
   - 从 [Elasticsearch 官网](https://www.elastic.co/downloads/elasticsearch) 下载并安装
   - 启动 Elasticsearch 服务

2. 创建 Python 虚拟环境：
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
创建 `.env` 文件并设置以下配置：
```
# Elasticsearch 配置
ELASTICSEARCH_URL=http://localhost:9200
INDEX_NAME=rag_documents

# 向量嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 自定义大模型 API 配置
LLM_API_URL=your_llm_api_url_here
LLM_API_KEY=your_llm_api_key_here
```

## 使用方法

1. 准备数据：
   将您的 JSON 格式数据文件命名为 `documents.json` 并放在项目根目录

2. 运行系统：
```bash
python rag_hybrid_search.py
```

## 工作原理

### 文档处理
- 从 JSON 文件加载文档数据
- 使用 Sentence Transformers 模型生成文档向量
- 将文档内容和向量存储在 Elasticsearch 中

### 混合检索
系统结合了两种搜索方式：
1. **向量相似度搜索**：使用文档和查询的向量余弦相似度
2. **关键词匹配**：使用 Elasticsearch 的标准文本匹配

最终得分计算公式：
```
最终得分 = α * 向量得分 + (1-α) * 关键词得分
```
其中 α 可配置（默认值：0.5）

## 自定义配置

可以自定义以下参数：
- `embedding_model`：更换向量嵌入模型（默认：'sentence-transformers/all-MiniLM-L6-v2'）
- `alpha`：调整向量搜索和关键词搜索的权重
- `k`：返回结果的数量

## 示例代码

```python
from rag_hybrid_search import ElasticsearchRAG

# 初始化 RAG 系统
rag = ElasticsearchRAG()

# 导入文档
rag.ingest_json_documents("documents.json")

# 执行混合搜索
results = rag.hybrid_search(
    query="RAG系统的主要优势是什么？",
    k=5,
    alpha=0.7  # 更偏重向量相似度搜索
)
```

## JSON 数据格式要求

您的 `documents.json` 文件应该遵循以下格式：
```json
[
    {
        "text": "文档内容",
        "metadata": {
            "title": "文档标题",
            "source": "文档来源",
            "date": "创建日期"
        }
    },
    // ... 更多文档
]
``` 