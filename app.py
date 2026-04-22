"""Streamlit entrypoint for the local RAG document Q&A assistant."""

from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st

from rag_local_qa import DocumentProcessor, QAEngine, VectorStoreManager, get_config


st.set_page_config(
    page_title="RAG Local Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_services() -> Tuple:
    """Create shared app services once per Streamlit process."""
    config = get_config()
    config.ensure_directories()
    processor = DocumentProcessor(config)
    vector_manager = VectorStoreManager(config)
    qa_engine = QAEngine(config, vector_manager)
    return config, processor, vector_manager, qa_engine


def init_session_state() -> None:
    """Initialize session state keys used by the chat UI."""
    st.session_state.setdefault("chat_history", [])


def build_knowledge_base(processor: DocumentProcessor, vector_manager: VectorStoreManager) -> Dict[str, int]:
    """Load, clean, chunk, and index every file saved in the upload folder."""
    raw_documents = processor.load_all_documents()
    chunks = processor.split_and_deduplicate(raw_documents)

    if not chunks:
        vector_manager.clear()
        return {"documents": len(raw_documents), "chunks": 0}

    vector_manager.build_from_documents(chunks)
    return {"documents": len(raw_documents), "chunks": len(chunks)}


def reset_knowledge_base(processor: DocumentProcessor, vector_manager: VectorStoreManager) -> None:
    """Remove source files, vector index, and current chat memory."""
    processor.clear_uploaded_files()
    vector_manager.clear()
    st.session_state["chat_history"] = []


def render_sidebar(
    processor: DocumentProcessor,
    vector_manager: VectorStoreManager,
) -> None:
    """Render upload tools and knowledge-base status."""
    config, _, _, _ = get_services()

    with st.sidebar:
        st.header("知识库管理")
        st.caption("支持 PDF / TXT 批量上传，本地构建 FAISS 向量索引。")

        uploaded_files = st.file_uploader(
            "选择文档",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="第一次构建时会自动下载开源 Embedding 模型，可能需要几分钟。",
        )

        if st.button("导入并重建知识库", use_container_width=True):
            if not uploaded_files:
                st.warning("请先选择至少一个 PDF 或 TXT 文件。")
            else:
                with st.spinner("正在保存文件并构建知识库，请稍候..."):
                    saved_paths = processor.save_uploaded_files(uploaded_files)
                    stats = build_knowledge_base(processor, vector_manager)

                st.success(
                    f"已保存 {len(saved_paths)} 个文件，当前共索引 {stats['documents']} 份文档、{stats['chunks']} 个文本块。"
                )

        if st.button("重新扫描 uploads 并重建", use_container_width=True):
            with st.spinner("正在重新扫描本地文档并重建索引..."):
                stats = build_knowledge_base(processor, vector_manager)
            st.success(f"重建完成：{stats['documents']} 份文档，{stats['chunks']} 个文本块。")

        if st.button("清空知识库", use_container_width=True):
            reset_knowledge_base(processor, vector_manager)
            st.success("已清空上传文档、FAISS 索引和聊天记录。")

        saved_files = sorted(
            path.name
            for path in config.upload_dir.iterdir()
            if path.is_file() and path.suffix.lower() in config.allowed_extensions
        )
        indexed_chunks = vector_manager.get_indexed_chunk_count()

        st.divider()
        st.subheader("当前状态")
        st.write(f"已保存文档数：{len(saved_files)}")
        st.write(f"已索引文本块：{indexed_chunks}")
        st.write(f"Embedding 模型：`{config.embedding_model_name}`")

        if saved_files:
            st.markdown("**本地文档列表**")
            for file_name in saved_files:
                st.write(f"- {file_name}")


def render_chat(qa_engine: QAEngine) -> None:
    """Render the main chat area."""
    st.title("RAG 本地文档问答助手")
    st.caption("适合课程项目、实习面试展示和本地知识库问答的轻量级 RAG Demo。")

    if not get_services()[2].get_indexed_chunk_count():
        st.info("左侧先上传 PDF/TXT 并构建知识库，然后再开始提问。")

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("查看来源"):
                    for source in message["sources"]:
                        st.write(
                            f"[{source['rank']}] {source['source']} | distance={source['score']}\n\n{source['preview']}"
                        )

    question = st.chat_input("请输入你的问题，例如：这份文档的核心结论是什么？")

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答..."):
                result = qa_engine.ask(question)

            st.markdown(result.answer)

            if result.sources:
                with st.expander("查看来源"):
                    for source in result.sources:
                        st.write(
                            f"[{source['rank']}] {source['source']} | distance={source['score']}\n\n{source['preview']}"
                        )

            if result.retrieved_chunks:
                with st.expander("查看检索到的文本块"):
                    for chunk in result.retrieved_chunks:
                        st.write(
                            f"Top {chunk['rank']} | {chunk['source']} | chunk #{chunk['chunk_id']} | distance={chunk['score']}"
                        )
                        st.write(chunk["content"])
                        st.divider()

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )


def main() -> None:
    """Run the Streamlit app."""
    init_session_state()
    _, processor, vector_manager, qa_engine = get_services()
    render_sidebar(processor, vector_manager)
    render_chat(qa_engine)


if __name__ == "__main__":
    main()
