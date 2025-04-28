import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from elasticsearch import Elasticsearch
import logging
import sys
import time
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_es_index(es_client, index_name):
    """创建 Elasticsearch 索引"""
    if not es_client.indices.exists(index=index_name):
        index_config = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {"type": "object"}
                }
            }
        }
        es_client.indices.create(index=index_name, body=index_config)
        logger.info(f"Created index {index_name}")

class ElasticsearchRAG:
    def __init__(
        self,
        es_url: str = "https://localhost:9200",
        index_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        username: str = "elastic",
        password: str = None,
        verify_certs: bool = False
    ):
        self.es_url = es_url
        self.index_name = index_name
        self.embedding_model = embedding_model
        
        try:
            # Initialize Elasticsearch client
            es_config = {
                "hosts": [es_url],
                "verify_certs": verify_certs,
                "request_timeout": 30,
                "retry_on_timeout": True,
                "max_retries": 3,
            }
            
            if username and password:
                es_config["basic_auth"] = (username, password)
            
            self.es_client = Elasticsearch(**es_config)
            
            # Test connection
            if not self.es_client.ping():
                raise ConnectionError("Could not ping Elasticsearch server")
            logger.info("Successfully connected to Elasticsearch")
            
            # Create index if not exists
            create_es_index(self.es_client, self.index_name)
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model
            )
            
            # Initialize vector store
            self.vector_store = ElasticsearchStore(
                es_client=self.es_client,
                index_name=self.index_name,
                embedding_dim=384,
                distance_strategy="COSINE"
            )
            
            # Configure settings
            Settings.embed_model = self.embed_model
            Settings.llm = None
            
        except Exception as e:
            logger.error(f"Error initializing ElasticsearchRAG: {str(e)}")
            raise

    def _batch_process_documents(self, documents: List[Document], batch_size: int = 50):
        """批量处理文档"""
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {total_batches}")
            
            # 为每个文档生成向量
            batch_vectors = []
            for doc in batch:
                vector = self.embed_model.get_text_embedding(doc.text)
                batch_vectors.append({
                    "text": doc.text,
                    "vector": vector,
                    "metadata": doc.metadata
                })
            
            # 批量插入到 Elasticsearch
            operations = []
            for item in batch_vectors:
                operations.append({
                    "_index": self.index_name,
                    "_source": item
                })
            
            if operations:
                from elasticsearch.helpers import bulk
                success, failed = bulk(
                    self.es_client,
                    operations,
                    raise_on_error=False
                )
                logger.info(f"Indexed {success} documents in batch {i//batch_size + 1}")
                if failed:
                    logger.warning(f"Failed to index {len(failed)} documents")

    def ingest_json_documents(self, json_file: str) -> None:
        """导入文档到 Elasticsearch"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for item in data:
                text = item.get('HTML_txt', '')
                metadata = {k: v for k, v in item.items() if k != 'HTML_txt'}
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {json_file}")
            self._batch_process_documents(documents)
            logger.info("All documents successfully ingested into Elasticsearch")
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            raise

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """执行混合搜索"""
        try:
            query_vector = self.embed_model.get_text_embedding(query)
            
            search_query = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {
                            "match": {
                                "text": query
                            }
                        },
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                }
            }
            
            response = self.es_client.search(
                index=self.index_name,
                body=search_query
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"]["text"],
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {})
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise

    def verify_index_properties(self):
        """验证索引的属性，证明其为向量索引"""
        try:
            # 获取索引映射信息
            mapping = self.es_client.indices.get_mapping(index=self.index_name)
            properties = mapping[self.index_name]['mappings']['properties']
            
            print("\n=== 索引属性验证 ===")
            print("1. 验证向量字段存在性:")
            if 'vector' in properties:
                vector_props = properties['vector']
                print(f"  ✓ 存在向量字段 'vector'")
                print(f"  ✓ 向量类型: {vector_props['type']}")
                print(f"  ✓ 向量维度: {vector_props['dims']}")
                print(f"  ✓ 相似度度量: {vector_props['similarity']}")
                print(f"  ✓ 是否已索引: {vector_props['index']}")
            else:
                print("  × 未找到向量字段")
            
            print("\n2. 验证文本字段存在性:")
            if 'text' in properties:
                print(f"  ✓ 存在文本字段 'text'")
                print(f"  ✓ 字段类型: {properties['text']['type']}")
            else:
                print("  × 未找到文本字段")
            
            # 获取索引统计信息
            stats = self.es_client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            print(f"\n3. 索引统计信息:")
            print(f"  ✓ 总文档数: {doc_count}")
            
            return True
        except Exception as e:
            logger.error(f"验证索引属性时出错: {str(e)}")
            return False

    def test_vector_search(self, query_text: str, k: int = 5):
        """测试向量检索功能"""
        try:
            # 生成查询向量
            query_vector = self.embed_model.get_text_embedding(query_text)
            
            print("\n=== 向量检索测试 ===")
            print(f"查询文本: {query_text}")
            
            # 1. 纯向量检索
            vector_query = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
            
            vector_results = self.es_client.search(
                index=self.index_name,
                body=vector_query
            )
            
            print("\n1. 纯向量检索结果:")
            for i, hit in enumerate(vector_results['hits']['hits'], 1):
                print(f"\n结果 {i}:")
                print(f"相似度分数: {hit['_score']}")
                print(f"文本片段: {hit['_source']['text'][:200]}...")
            
            # 2. 混合检索（向量 + 关键词）
            hybrid_query = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {
                            "match": {"text": query_text}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
            
            hybrid_results = self.es_client.search(
                index=self.index_name,
                body=hybrid_query
            )
            
            print("\n2. 混合检索结果（向量 + 关键词）:")
            for i, hit in enumerate(hybrid_results['hits']['hits'], 1):
                print(f"\n结果 {i}:")
                print(f"混合相似度分数: {hit['_score']}")
                print(f"文本片段: {hit['_source']['text'][:200]}...")
            
        except Exception as e:
            logger.error(f"测试向量检索时出错: {str(e)}")
            raise

    def generate_query_vector(self, query_text: str):
        """生成查询向量并打印 ES 查询语句"""
        try:
            # 生成查询向量
            query_vector = self.embed_model.get_text_embedding(query_text)
            
            # 生成 ES 查询语句
            vector_query = {
                "size": 5,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
            
            print("\n=== ES 开发者控制台查询命令 ===")
            print("# 1. 查看索引映射")
            print("GET /rag_documents/_mapping")
            
            print("\n# 2. 执行向量检索")
            print("GET /rag_documents/_search")
            print(json.dumps(vector_query, indent=2, ensure_ascii=False))
            
            # 生成混合查询
            hybrid_query = {
                "size": 5,
                "query": {
                    "script_score": {
                        "query": {
                            "match": {"text": query_text}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
            
            print("\n# 3. 执行混合检索（向量 + 关键词）")
            print("GET /rag_documents/_search")
            print(json.dumps(hybrid_query, indent=2, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"生成查询向量时出错: {str(e)}")
            raise

    def generate_detailed_query(self, query_text: str):
        """生成详细的查询命令"""
        try:
            # 生成查询向量
            query_vector = self.embed_model.get_text_embedding(query_text)
            
            # 生成详细的查询
            detailed_query = {
                "size": 2,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "match_phrase": {  # 使用短语匹配而不是普通匹配
                                            "text": "黑龙江省高级人民法院"
                                        }
                                    }
                                ],
                                "should": [  # 增加可选匹配条件
                                    {
                                        "match": {
                                            "text": "执行裁定书"
                                        }
                                    }
                                ],
                                "minimum_should_match": 0  # 可选条件不是必须的
                            }
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                },
                "_source": ["text", "metadata"],
                "highlight": {  # 添加高亮显示
                    "fields": {
                        "text": {}
                    },
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"]
                }
            }
            
            print("\n=== ES 详细查询命令 ===")
            print("# 执行详细查询 (返回前2个最相关结果)")
            print("GET /rag_documents/_search")
            print(json.dumps(detailed_query, indent=2, ensure_ascii=False))
            
            # 实际执行查询
            print("\n=== 查询结果 ===")
            results = self.es_client.search(
                index=self.index_name,
                body=detailed_query
            )
            
            # 打印结果
            total_hits = results['hits']['total']['value']
            print(f"\n找到 {total_hits} 个匹配结果")
            
            for i, hit in enumerate(results['hits']['hits'], 1):
                print(f"\n结果 {i} (相关度得分: {hit['_score']:.2f}):")
                print("-" * 80)
                
                # 显示高亮结果如果有的话
                if 'highlight' in hit:
                    print("匹配片段:")
                    for fragment in hit['highlight']['text']:
                        print(f"...{fragment}...")
                    print()
                
                print(f"完整文本: {hit['_source']['text'][:500]}...")
                
                if 'metadata' in hit['_source']:
                    print("\n元数据:")
                    for key, value in hit['_source']['metadata'].items():
                        print(f"{key}: {value}")
                print("-" * 80)
            
            if total_hits == 0:
                print("\n提示: 未找到精确匹配的文档。您可以:")
                print("1. 检查索引中是否包含相关文档")
                print("2. 使用 GET /rag_documents/_search?q=黑龙江省高级人民法院 查看是否有相关内容")
                print("3. 检查文档的原始内容格式")
            
        except Exception as e:
            logger.error(f"生成详细查询时出错: {str(e)}")
            raise

    def simple_text_search(self, query_text: str, search_type="match", size=5):
        """
        执行简单的文本搜索，不使用向量
        
        Args:
            query_text: 搜索文本
            search_type: 搜索类型 (match, match_phrase, simple)
            size: 返回结果数量
        """
        try:
            # 构建查询
            if search_type == "simple":
                # 最简单的查询语法
                results = self.es_client.search(
                    index=self.index_name,
                    q=query_text,
                    size=size
                )
                
                # 打印可在控制台使用的命令
                print(f"\n=== 简单查询命令 ===")
                print(f"GET /{self.index_name}/_search?q={query_text}")
                
            elif search_type == "match_phrase":
                # 精确短语匹配
                query = {
                    "size": size,
                    "query": {
                        "match_phrase": {
                            "text": query_text
                        }
                    },
                    "highlight": {
                        "fields": {
                            "text": {}
                        }
                    }
                }
                
                results = self.es_client.search(
                    index=self.index_name,
                    body=query
                )
                
                # 打印可在控制台使用的命令
                print(f"\n=== 精确短语匹配命令 ===")
                print(f"GET /{self.index_name}/_search")
                print(json.dumps(query, indent=2, ensure_ascii=False))
                
            else:  # 默认 match
                # 标准全文搜索
                query = {
                    "size": size,
                    "query": {
                        "match": {
                            "text": query_text
                        }
                    },
                    "highlight": {
                        "fields": {
                            "text": {}
                        }
                    }
                }
                
                results = self.es_client.search(
                    index=self.index_name,
                    body=query
                )
                
                # 打印可在控制台使用的命令
                print(f"\n=== 全文搜索命令 ===")
                print(f"GET /{self.index_name}/_search")
                print(json.dumps(query, indent=2, ensure_ascii=False))
            
            # 打印结果
            total_hits = results['hits']['total']['value']
            print(f"\n=== 查询结果 ===")
            print(f"找到 {total_hits} 个匹配文档")
            
            for i, hit in enumerate(results['hits']['hits'], 1):
                print(f"\n结果 {i} (相关度: {hit['_score']:.2f}):")
                print("-" * 80)
                
                # 显示高亮结果
                if 'highlight' in hit:
                    print("匹配片段:")
                    for fragment in hit['highlight']['text']:
                        print(f"...{fragment}...")
                    print()
                
                # 显示文本内容
                print(f"文本内容: {hit['_source']['text'][:200]}...")
                print("-" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"执行简单搜索时出错: {str(e)}")
            raise

def main():
    # 从环境变量获取认证信息
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")
    if not es_password:
        print("ERROR: ELASTICSEARCH_PASSWORD environment variable not set")
        sys.exit(1)
    
    try:
        # 初始化 RAG 系统
        rag = ElasticsearchRAG(
            es_url="https://localhost:9200",
            username="elastic",
            password=es_password,
            verify_certs=False
        )
        
        # 执行简单文本搜索 (不使用向量)
        print("\n=== 不使用向量的简单文本搜索 ===")
        search_text = "黑龙江省高级人民法院"
        
        print("\n1. 标准全文搜索:")
        rag.simple_text_search(search_text, search_type="match", size=2)
        
        print("\n2. 精确短语匹配:")
        rag.simple_text_search(search_text, search_type="match_phrase", size=2)
        
        print("\n3. 最简单的查询语法:")
        rag.simple_text_search(search_text, search_type="simple", size=2)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 