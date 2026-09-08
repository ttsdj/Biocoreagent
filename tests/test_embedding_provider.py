"""语义嵌入双档 (C1) 的测试.

对应简历「语义嵌入」声明。真实处境：默认是**确定性的哈希嵌入**（离线、可复现），
只有显式设置 ``BIOCOREAGENT_EMBEDDING_BACKEND=sentence-transformers`` 并安装
sentence-transformers 后才启用真实语义模型。这个测试用 monkeypatch 把重量级
模型替换成桩，拿住「选型由环境变量决定」这个契约，不会在 CI 里联网加载模型。

断言：
  - 默认（未设环境变量）选中的是 dense_hash_fallback 确定性嵌入；
  - 哈希嵌入本身是确定性的、128 维以内归一化向量（真实数学性质，可复现）；
  - 设为 sentence-transformers 时 name 前缀为 ``sentence_transformers:<model>``。
"""

from __future__ import annotations

import math

from pico.features import postgres_memory


class _FakeVector:
    def __init__(self, data):
        self._data = list(data)

    def tolist(self):
        return list(self._data)


class _FakeSTModel:
    """替换 sentence-transformers 重模型的最小桩。"""

    def __init__(self, model_name):
        self.model_name = str(model_name)

    def encode(self, text, normalize_embeddings=True):  # noqa: ARG002
        return _FakeVector([0.1, 0.2, 0.3])


def _fake_loader(model_name):
    return _FakeSTModel(model_name)


# ---------------------------------------------------------------------------
# 默认（哈希）档
# ---------------------------------------------------------------------------


def test_default_provider_is_dense_hash_fallback(monkeypatch):
    monkeypatch.delenv("BIOCOREAGENT_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("BIOCOREAGENT_EMBEDDING_MODEL", raising=False)
    provider = postgres_memory.embedding_provider_from_environment()
    assert isinstance(provider, postgres_memory.EmbeddingProvider)
    assert provider.name == "dense_hash_fallback"
    # 不应是 sentence-transformers 档
    assert not isinstance(provider, postgres_memory.SentenceTransformerEmbeddingProvider)


def test_hash_embedding_is_deterministic_and_normalized():
    a = postgres_memory._hash_embedding("RNA-seq 差异表达 DESeq2")
    b = postgres_memory._hash_embedding("RNA-seq 差异表达 DESeq2")
    assert a == b, "同一文本的哈希嵌入必须确定（可复现）"

    # 归一化向量（欧氏范数 ≈ 1），且非空向量长度等于维度。
    norm = math.sqrt(sum(value * value for value in a))
    assert norm == 0.0 or abs(norm - 1.0) < 1e-6
    assert len(a) == 256  # 默认维度，确保 pgvector(256) 与之一致


def test_hash_embedding_differs_for_diff_text():
    a = postgres_memory._hash_embedding("差异表达")
    b = postgres_memory._hash_embedding("肿瘤突变负荷")
    assert a != b


# ---------------------------------------------------------------------------
# sentence-transformers 档（模型以桩替换，避免联网）
# ---------------------------------------------------------------------------


def test_sentence_transformer_name_prefix(monkeypatch):
    monkeypatch.setattr(
        postgres_memory, "_load_sentence_transformer", _fake_loader
    )
    provider = postgres_memory.SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
    assert provider.name == "sentence_transformers:all-MiniLM-L6-v2"
    assert provider.embed("hello") == [0.1, 0.2, 0.3]


def test_environment_selects_sentence_transformers(monkeypatch):
    monkeypatch.setattr(
        postgres_memory, "_load_sentence_transformer", _fake_loader
    )
    monkeypatch.setenv("BIOCOREAGENT_EMBEDDING_BACKEND", "sentence-transformers")
    monkeypatch.delenv("BIOCOREAGENT_EMBEDDING_MODEL", raising=False)
    provider = postgres_memory.embedding_provider_from_environment()
    assert isinstance(provider, postgres_memory.SentenceTransformerEmbeddingProvider)
    assert provider.name.startswith("sentence_transformers:")
    assert provider.name.endswith("all-MiniLM-L6-v2")
