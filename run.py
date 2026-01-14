import asyncio
import json
from pathlib import Path

from langchain_core.messages import BaseMessage

from Agents.workflow import create_workflow_graph, create_initial_state
from RAG.rag import HybridRAGSystem


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """将对话与讨论过程序列化为可保存的结构"""
    serialized: list[dict] = []
    for m in messages or []:
        try:
            role = getattr(m, "type", m.__class__.__name__)
            content = getattr(m, "content", "")
            serialized.append(
                {
                    "role": role,
                    "content": content,
                }
            )
        except Exception as e:
            serialized.append(
                {
                    "role": "unknown",
                    "content": f"[Serialization error: {e}]",
                }
            )
    return serialized


def _format_report_to_txt(report: dict) -> str:
    """将报告字典格式化为易读的TXT文本"""
    lines = []
    
    # 标题
    lines.append("=" * 80)
    lines.append(report.get("title", "Experimental Design Report"))
    lines.append("=" * 80)
    lines.append("")
    
    # 摘要
    summary = report.get("summary", "")
    if summary:
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(summary)
        lines.append("")
    
    # 总体评分
    overall_score = report.get("overall_score", 0)
    lines.append(f"Overall Feasibility Score: {overall_score:.1f}/10")
    lines.append("")
    
    # 任务信息
    task_info = report.get("task_information", {})
    if task_info:
        lines.append("TASK INFORMATION")
        lines.append("-" * 80)
        lines.append(f"Description: {task_info.get('description', 'N/A')}")
        lines.append("")
        if task_info.get("background"):
            lines.append("Background:")
            lines.append(task_info["background"])
            lines.append("")
        if task_info.get("dataset_info"):
            lines.append("Dataset Information:")
            lines.append(task_info["dataset_info"])
            lines.append("")
    
    # 实验设计方案
    exp_design = report.get("experimental_design", {})
    if exp_design:
        lines.append("=" * 80)
        lines.append("EXPERIMENTAL DESIGN IMPLEMENTATION PLAN")
        lines.append("=" * 80)
        lines.append("")
        
        # 1. 数据使用计划
        data_plan = exp_design.get("1_data_usage_plan", {})
        if data_plan:
            lines.append("1. DATA USAGE PLAN")
            lines.append("-" * 80)
            for key, value in data_plan.items():
                if isinstance(value, str) and value.strip():
                    lines.append(f"\n{key.replace('_', ' ').title()}:")
                    lines.append(value)
                    lines.append("")
            lines.append("")
        
        # 2. 方法设计
        method_design = exp_design.get("2_method_design", {})
        if method_design:
            lines.append("2. METHOD DESIGN")
            lines.append("-" * 80)
            for key, value in method_design.items():
                if isinstance(value, str) and value.strip():
                    lines.append(f"\n{key.replace('_', ' ').title()}:")
                    lines.append(value)
                    lines.append("")
            lines.append("")
        
        # 3. 模型设计
        model_design = exp_design.get("3_model_design", {})
        if model_design:
            lines.append("3. MODEL DESIGN")
            lines.append("-" * 80)
            for key, value in model_design.items():
                if isinstance(value, str) and value.strip():
                    lines.append(f"\n{key.replace('_', ' ').title()}:")
                    lines.append(value)
                    lines.append("")
            lines.append("")
        
        # 4. 结果总结
        result_summary = exp_design.get("4_result_summary", {})
        if result_summary:
            lines.append("4. RESULT SUMMARY")
            lines.append("-" * 80)
            for key, value in result_summary.items():
                if isinstance(value, str) and value.strip():
                    lines.append(f"\n{key.replace('_', ' ').title()}:")
                    lines.append(value)
                    lines.append("")
            lines.append("")
    
    # 专家分析（重点展示实施方案）
    expert_analyses = report.get("expert_analyses", {})
    if expert_analyses:
        lines.append("=" * 80)
        lines.append("EXPERT IMPLEMENTATION PLANS")
        lines.append("=" * 80)
        lines.append("")
        
        role_names = {
            "data_management": "Data Management Expert",
            "methodology": "Methodology Expert",
            "model_architect": "Model Architect",
            "result_analyst": "Result Analyst"
        }
        
        for role, analysis in expert_analyses.items():
            role_name = role_names.get(role, role)
            score = analysis.get("score", 0)
            lines.append(f"{role_name.upper()} (Score: {score:.1f}/10)")
            lines.append("-" * 80)
            
            # 设计摘要
            design_summary = analysis.get("design_summary", "")
            if design_summary:
                lines.append("\nDesign Summary:")
                lines.append(design_summary)
                lines.append("")
            
            # 实施方案（重点）- 完整显示，不进行任何概括
            impl_plan = analysis.get("implementation_plan", {})
            full_metadata = analysis.get("full_metadata", {})
            
            # 如果 implementation_plan 为空，尝试从 full_metadata 中获取
            if (not impl_plan or (isinstance(impl_plan, dict) and len(impl_plan) == 0)) and full_metadata:
                impl_plan = full_metadata.get("detailed_design", {})
            
            if impl_plan and isinstance(impl_plan, dict) and len(impl_plan) > 0:
                lines.append("\nImplementation Plan (Complete, No Summarization):")
                lines.append("")
                for key, value in impl_plan.items():
                    if value:  # 只要value不为空就显示
                        lines.append(f"{key.replace('_', ' ').title()}:")
                        lines.append("-" * 60)
                        # 完整显示内容，保持原有格式，不进行任何修改
                        if isinstance(value, str):
                            # 保持代码块的原始格式
                            lines.append(value)
                        elif isinstance(value, dict):
                            # 如果是嵌套字典，使用JSON格式完整显示
                            import json
                            lines.append(json.dumps(value, ensure_ascii=False, indent=2))
                        elif isinstance(value, list):
                            # 如果是列表，完整显示
                            for item in value:
                                lines.append(f"  - {item}")
                        else:
                            lines.append(str(value))
                        lines.append("")
                        lines.append("")
            else:
                lines.append("\n⚠️ Implementation Plan: (Not available or empty)")
                lines.append("   This may indicate that the agent did not generate detailed_design.")
                if full_metadata:
                    lines.append(f"   Available metadata keys: {list(full_metadata.keys())}")
                lines.append("")
            
            # 建议
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                lines.append("Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    lines.append(f"  {i}. {rec}")
                lines.append("")
            
            # 检索到的知识条目（用于可解释性）
            retrieved_knowledge = full_metadata.get("retrieved_knowledge", []) if full_metadata else analysis.get("full_metadata", {}).get("retrieved_knowledge", [])
            if not retrieved_knowledge:
                # 尝试从implementation_plan的metadata中获取
                impl_meta = analysis.get("implementation_plan", {})
                if isinstance(impl_meta, dict):
                    retrieved_knowledge = impl_meta.get("retrieved_knowledge", [])
            
            if retrieved_knowledge:
                lines.append("Retrieved Knowledge Base Items (for Explainability):")
                lines.append("-" * 60)
                for i, kb_item in enumerate(retrieved_knowledge, 1):
                    kb_id = kb_item.get("id", "N/A")
                    kb_title = kb_item.get("title", "")
                    kb_content = kb_item.get("content", "")
                    kb_source = kb_item.get("source", "")
                    kb_score = kb_item.get("relevance_score", 0.0)
                    
                    lines.append(f"\n[{i}] Knowledge ID: {kb_id}")
                    if kb_title:
                        lines.append(f"    Title: {kb_title}")
                    if kb_source:
                        lines.append(f"    Source: {kb_source}")
                    lines.append(f"    Relevance Score: {kb_score:.4f}")
                    lines.append(f"    Content:")
                    # 显示内容（如果太长，适当截断但保留大部分）
                    if len(kb_content) > 1000:
                        lines.append(f"    {kb_content[:1000]}...")
                        lines.append(f"    ... (truncated, total length: {len(kb_content)} chars)")
                    else:
                        lines.append(f"    {kb_content}")
                    lines.append("")
            
            lines.append("")
    
    # 优先建议
    priority_recs = report.get("priority_recommendations", [])
    if priority_recs:
        lines.append("=" * 80)
        lines.append("PRIORITY RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")
        for i, rec in enumerate(priority_recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    return "\n".join(lines)


def _format_conversation_to_txt(messages: list[dict]) -> str:
    """将对话日志格式化为易读的TXT文本"""
    lines = []
    
    lines.append("=" * 80)
    lines.append("CONVERSATION & DISCUSSION LOG")
    lines.append("=" * 80)
    lines.append("")
    
    role_display = {
        "system": "SYSTEM",
        "human": "USER",
        "ai": "ASSISTANT",
        "tool": "TOOL"
    }
    
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        display_role = role_display.get(role.lower(), role.upper())
        
        lines.append(f"[Message {i}] {display_role}")
        lines.append("-" * 80)
        
        # 显示完整内容，不截断
        lines.append(content)
        
        lines.append("")
        lines.append("")
    
    return "\n".join(lines)


def main() -> None:
    rag = HybridRAGSystem()
    
    # 检查向量数据库是否已有数据，如果没有则加载知识库（包括核心论文）
    collection = rag.collections.get("shared_knowledge_base")
    if collection and collection.count() == 0:
        print("\n" + "="*60)
        print("📚 向量数据库为空，正在加载知识库（包括核心论文Methods部分）...")
        print("="*60)
        rag.load_knowledge_base(load_core_papers=True)
        print("="*60 + "\n")
    elif collection and collection.count() > 0:
        print(f"\n✓ 向量数据库已有 {collection.count()} 个文档，跳过加载")
        # 检查是否需要单独加载核心论文（如果之前没有加载过）
        # 通过检查是否有 doc_type="core_paper_methods" 的文档来判断
        try:
            sample_results = collection.peek(limit=100)
            has_core_papers = any(
                meta.get("doc_type") == "core_paper_methods" 
                for meta in (sample_results.get("metadatas", []) or [])
            )
            if not has_core_papers:
                print("⚠️ 检测到向量库中没有核心论文Methods部分，正在加载...")
                rag.load_core_papers()
        except Exception as e:
            print(f"⚠️ 检查核心论文时出错: {e}，尝试加载核心论文...")
            rag.load_core_papers()
    else:
        print("⚠️ 无法访问向量数据库集合，尝试加载知识库...")
        rag.load_knowledge_base(load_core_papers=True)
    
    app = create_workflow_graph(rag_system=rag)

    state = create_initial_state(rag_system=rag)
    # 使用异步接口运行包含异步节点的LangGraph应用，并适当提高递归上限，防止早期迭代触发递归限制
    result = asyncio.run(app.ainvoke(state, config={"recursion_limit": 40}))

    print("✓ Workflow finished")

    # 创建输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存最终报告（JSON和TXT格式）
    final_report = result.get("final_report")
    if final_report:
        title = final_report.get("title", "")
        print("✓ Final report title:", title)

        # 保存JSON格式
        report_json_path = output_dir / "final_report.json"
        with report_json_path.open("w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        print(f"✓ Final report (JSON) saved to: {report_json_path.resolve()}")

        # 保存TXT格式（易读）
        report_txt_path = output_dir / "final_report.txt"
        report_txt_content = _format_report_to_txt(final_report)
        with report_txt_path.open("w", encoding="utf-8") as f:
            f.write(report_txt_content)
        print(f"✓ Final report (TXT) saved to: {report_txt_path.resolve()}")

    # 2) 保存完整对话与讨论过程（会议记录，JSON和TXT格式）
    messages = result.get("messages", [])
    serialized_msgs = _serialize_messages(messages)
    
    # 保存JSON格式
    convo_json_path = output_dir / "conversation_log.json"
    with convo_json_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_msgs, f, ensure_ascii=False, indent=2)
    print(f"✓ Conversation log (JSON) saved to: {convo_json_path.resolve()}")
    
    # 保存TXT格式（易读）
    convo_txt_path = output_dir / "conversation_log.txt"
    convo_txt_content = _format_conversation_to_txt(serialized_msgs)
    with convo_txt_path.open("w", encoding="utf-8") as f:
        f.write(convo_txt_content)
    print(f"✓ Conversation log (TXT) saved to: {convo_txt_path.resolve()}")


if __name__ == "__main__":
    main()