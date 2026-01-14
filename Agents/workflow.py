"""
Agents/workflow.py
LangGraph工作流定义 - 多Agent协调系统
"""

from typing import Dict, Any, Literal, Optional
from pathlib import Path
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from Agents.state import REAgentState, CritiqueResult, OptimizationReport
from Agents.prompt import initialize_supervisor, initialize_agents_with_rag
from Agents.agent import Agent, summarize_messages
from RAG.rag import HybridRAGSystem


# ==================== 工作流节点定义 ====================

# 全局变量用于存储Supervisor、专家Agent和RAG系统实例（在工作流执行期间共享）
_workflow_supervisor: Agent = None
_workflow_expert_agents: Dict[str, Agent] = None
_workflow_rag_system: HybridRAGSystem = None


def set_workflow_components(supervisor: Agent = None, expert_agents: Dict[str, Agent] = None, rag_system: HybridRAGSystem = None):
    """设置工作流组件（Supervisor、专家Agent和RAG系统）"""
    global _workflow_supervisor, _workflow_expert_agents, _workflow_rag_system
    if supervisor is not None:
        _workflow_supervisor = supervisor
    if expert_agents is not None:
        _workflow_expert_agents = expert_agents
    if rag_system is not None:
        _workflow_rag_system = rag_system


def _check_scores_meet_requirements(state: REAgentState) -> bool:
    """
    检查评分是否满足严格的要求
    要求：平均分>=9.0，每个专家>=8.5
    """
    critiques = {
        "data_management": state.get("data_critique"),
        "methodology": state.get("methodology_critique"),
        "model_architect": state.get("model_critique"),
        "result_analyst": state.get("results_critique")
    }
    
    scores = []
    for role, critique in critiques.items():
        if critique:
            if isinstance(critique, dict):
                score = critique.get("score", 0)
            elif hasattr(critique, "score"):
                score = critique.score
            else:
                score = 0
            
            if score > 0:
                scores.append(score)
                # 检查每个专家是否>=8.5
                if score < 8.5:
                    print(f"  ✗ {role} 评分 {score:.1f} < 8.5 (不满足要求)")
                    return False
    
    if not scores:
        print("  ✗ 没有可用的评分")
        return False
    
    # 检查平均分是否>=9.0
    avg_score = sum(scores) / len(scores)
    if avg_score < 9.0:
        print(f"  ✗ 平均分 {avg_score:.1f} < 9.0 (不满足要求)")
        return False
    
    print(f"  ✓ 评分满足要求：平均分 {avg_score:.1f} >= 9.0，所有专家 >= 8.5")
    return True


def _create_status_summary(state: REAgentState) -> str:
    """创建当前状态摘要"""
    summary_parts = []
    
    # 任务信息状态
    task_description = state.get("task_description", "")
    background = state.get("background", "")
    dataset_info = state.get("dataset_info", "")
    
    if task_description:
        summary_parts.append(f"✓ 任务描述: {task_description[:150]}...")
    else:
        summary_parts.append("✗ 任务描述缺失")
    
    if background:
        summary_parts.append(f"✓ 背景要求: {background[:150]}...")
    else:
        summary_parts.append("✗ 背景要求缺失")
    
    if dataset_info:
        summary_parts.append(f"✓ 数据集信息: {dataset_info[:150]}...")
    else:
        summary_parts.append("✗ 数据集信息缺失")
    
    # 显示数据集统计信息（如果Supervisor已读取）
    dataset_stats = state.get("dataset_statistics", {})
    if dataset_stats:
        summary_parts.append("\n📊 数据集统计信息（已读取）:")
        for file_path, stats in dataset_stats.items():
            summary_parts.append(f"  文件: {file_path}")
            summary_parts.append(f"    - 行数（样本数）: {stats.get('num_rows', 'N/A')}")
            summary_parts.append(f"    - 列数（特征数）: {stats.get('num_cols', 'N/A')}")
            col_names = stats.get('column_names', [])
            if col_names:
                col_names_str = ', '.join(col_names[:10])  # 最多显示10个列名
                if len(col_names) > 10:
                    col_names_str += f" ... (共{len(col_names)}列)"
                summary_parts.append(f"    - 列名: {col_names_str}")
    
    # 检查是否有完整的任务信息
    has_complete_info = task_description and background and dataset_info
    if has_complete_info:
        summary_parts.append("\n✓ 任务信息完整，可以进行分析")
    else:
        summary_parts.append("\n✗ 任务信息不完整，需要更多信息")
    
    # 专家分析状态
    critiques = {
        "数据管理": state.get("data_critique"),
        "方法学": state.get("methodology_critique"),
        "模型架构": state.get("model_critique"),
        "结果分析": state.get("results_critique")
    }
    
    completed_analyses = sum(1 for c in critiques.values() if c is not None)
    summary_parts.append(f"\n专家分析: {completed_analyses}/4 完成")
    
    scores = []
    for name, critique in critiques.items():
        if critique:
            if isinstance(critique, dict):
                score = critique.get("score", 0)
            elif hasattr(critique, "score"):
                score = critique.score
            else:
                score = 0
            
            if score > 0:
                scores.append(score)
                # 显示评分和是否满足要求（>=8.5）
                status = "✓" if score >= 8.5 else "✗"
                summary_parts.append(f"  {status} {name}: {score:.1f}/10 {'(满足要求)' if score >= 8.5 else '(不满足要求，需要>=8.5)'}")
            else:
                summary_parts.append(f"  ✓ {name}: 已完成")
        else:
            summary_parts.append(f"  ✗ {name}: 待分析")
    
    # 显示平均分和评分要求
    if scores:
        avg_score = sum(scores) / len(scores)
        avg_status = "✓" if avg_score >= 9.0 else "✗"
        summary_parts.append(f"\n{avg_status} 平均分: {avg_score:.1f}/10 {'(满足要求)' if avg_score >= 9.0 else '(不满足要求，需要>=9.0)'}")
        summary_parts.append("评分要求: 平均分>=9.0，每个专家>=8.5")
    
    # 迭代状态
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 8)
    summary_parts.append(f"\n迭代: {iteration}/{max_iter}")
    
    # 报告状态
    if state.get("final_report"):
        summary_parts.append("✓ 最终报告已生成")
    
    return "\n".join(summary_parts)


def _parse_decision(response, state: Optional[REAgentState] = None) -> dict:
    """解析Supervisor的决策"""
    import json
    
    content = response.content
    
    # 尝试提取JSON
    try:
        # 如果响应中包含代码块
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()
        
        decision = json.loads(json_str)
        return decision
    
    except:
        # 降级：从响应中提取next_action
        if "next_action" in content.lower():
            for action in ["request_info", "analyze", "discuss", "iterate", "report", "end"]:
                if action in content.lower():
                    return {
                        "reasoning": "从响应中推断",
                        "next_action": action,
                        "tool_calls": []
                    }
        
        # 默认决策：检查任务信息是否完整
        if state:
            task_description = state.get("task_description", "")
            background = state.get("background", "")
            dataset_info = state.get("dataset_info", "")
            
            if not (task_description and background and dataset_info):
                default_action = "request_info"
            else:
                default_action = "analyze"
        else:
            default_action = "request_info"
        
        return {
            "reasoning": "无法解析决策，使用默认",
            "next_action": default_action,
            "tool_calls": []
        }


def _build_task_plan(state: REAgentState) -> Dict[str, Any]:
    """从状态中构建任务计划字典（用于Agent分析）"""
    task_plan = {
        "title": state.get("task_description", "Gene Regulatory Element Design Task"),
        "description": state.get("background", ""),
        "data_source": state.get("dataset_info", ""),
        "methodology": state.get("methodology", ""),
        "model_architecture": state.get("model_architecture", ""),
        "evaluation_metrics": state.get("evaluation_metrics", ""),
        "additional_info": state.get("additional_info", {})
    }
    
    # 添加数据集统计信息（如果Supervisor已读取）
    dataset_stats = state.get("dataset_statistics", {})
    if dataset_stats:
        task_plan["dataset_statistics"] = dataset_stats
    
    return task_plan


async def supervisor_node(state: REAgentState) -> REAgentState:
    """
    Supervisor节点 - 决策下一步行动
    
    根据当前状态，Supervisor决定下一步：
    - request_info: 需要更多任务信息
    - analyze: 调用专家Agent分析
    - discuss: 专家讨论
    - iterate: 迭代优化
    - report: 生成报告
    - end: 结束
    """
    global _workflow_supervisor, _workflow_rag_system
    
    print("\n" + "="*60)
    print("🔍 Supervisor 正在分析状态并决策...")
    print("="*60)
    
    # 获取或创建Supervisor实例
    if _workflow_supervisor is None:
        if _workflow_rag_system is None:
            _workflow_rag_system = HybridRAGSystem()
        _workflow_supervisor = initialize_supervisor(rag_system=_workflow_rag_system)
    
    # 检查是否需要质疑讨论结果
    after_discussion = state.get("after_discussion", False)
    if after_discussion:
        print("\n" + "="*60)
        print("🔍 Supervisor 正在质疑和评估讨论结果...")
        print("="*60)
        
        # 收集所有专家的分析结果用于质疑
        critiques = {
            "data_management": state.get("data_critique"),
            "methodology": state.get("methodology_critique"),
            "model_architect": state.get("model_critique"),
            "result_analyst": state.get("results_critique")
        }
        
        # 构建质疑上下文
        critique_summary = ""
        scores = []
        for role, critique in critiques.items():
            if critique:
                if isinstance(critique, dict):
                    score = critique.get("score", 0)
                    design_summary = critique.get("metadata", {}).get("design_summary", "")
                    detailed_design = critique.get("metadata", {}).get("detailed_design", {})
                    strengths = critique.get("strengths", [])
                    weaknesses = critique.get("weaknesses", [])
                    recommendations = critique.get("recommendations", [])
                elif hasattr(critique, "score"):
                    score = critique.score
                    design_summary = critique.metadata.get("design_summary", "") if hasattr(critique, "metadata") else ""
                    detailed_design = critique.metadata.get("detailed_design", {}) if hasattr(critique, "metadata") else {}
                    strengths = critique.strengths
                    weaknesses = critique.weaknesses
                    recommendations = critique.recommendations
                else:
                    continue
                
                role_name = {
                    "data_management": "数据管理专家",
                    "methodology": "方法学专家",
                    "model_architect": "模型架构师",
                    "result_analyst": "结果分析师"
                }.get(role, role)
                
                scores.append(score)
                critique_summary += f"\n【{role_name}】评分: {score:.1f}/10\n"
                if design_summary:
                    critique_summary += f"设计方案摘要: {design_summary[:300]}...\n"
                if strengths:
                    critique_summary += f"优点: {', '.join(strengths[:3])}\n"
                if weaknesses:
                    critique_summary += f"需要改进: {', '.join(weaknesses[:3])}\n"
                if recommendations:
                    critique_summary += f"建议: {', '.join(recommendations[:2])}\n"
                critique_summary += "\n"
        
        # 计算平均分
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 构建质疑提示
        critique_prompt = f"""You have just completed a discussion round with the expert agents. Now you need to CRITICALLY REVIEW and QUESTION their designs.

Current Expert Analysis Results:
{critique_summary}

Average Score: {avg_score:.1f}/10
Individual Scores: {', '.join([f'{s:.1f}' for s in scores])}

⚠️ YOUR CRITICAL REVIEW TASKS:
1. **Question the Design Quality**:
   - Are the design specifications comprehensive enough? Do they include detailed parameter values?
   - Are there missing critical parameters or hyperparameters?
   - Are the designs consistent with each other? (e.g., data preprocessing matches model input requirements)
   - Are the designs realistic and implementable?

2. **Question the Scores**:
   - Are the scores justified? Do they reflect the actual quality of the designs?
   - Are there designs that scored too high or too low?
   - Do the designs meet the strict requirements (detailed parameter specifications)?

3. **Identify Specific Issues**:
   - What specific aspects need improvement?
   - What parameters are missing or unclear?
   - What inconsistencies exist between different expert designs?

4. **Provide Critical Questions and Suggestions**:
   - Raise specific questions about each expert's design
   - Point out concrete issues that need to be addressed
   - Suggest specific improvements with parameter-level details

Please provide your critical review in JSON format:
{{
    "critical_questions": [
        "Specific question 1 about data management expert's design...",
        "Specific question 2 about methodology expert's design...",
        ...
    ],
    "identified_issues": [
        "Issue 1: Missing parameter X in model architecture...",
        "Issue 2: Inconsistency between data preprocessing and model input...",
        ...
    ],
    "score_evaluation": "Your evaluation of whether the scores are justified and whether they meet requirements",
    "recommendations": [
        "Specific recommendation 1 with parameter details...",
        "Specific recommendation 2 with parameter details...",
        ...
    ],
    "decision": "iterate" or "report",
    "reasoning": "Your reasoning for the decision based on the critical review"
}}

IMPORTANT:
- Be CRITICAL and SPECIFIC in your questions and issues
- Focus on missing parameters, inconsistencies, and design quality
- Only set decision = "report" if ALL requirements are met (avg >= 9.0, each >= 8.5, comprehensive designs)
- If any issues are found, set decision = "iterate" to continue refinement
"""
        
        # 调用Supervisor进行质疑
        critique_messages = [
            SystemMessage(content=_workflow_supervisor.prompt() + "\n\nYou are now critically reviewing the expert discussion results. Be thorough and specific in your critique."),
            HumanMessage(content=critique_prompt)
        ]
        
        critique_response = await _workflow_supervisor.llm_with_tools.ainvoke(critique_messages)
        critique_text = critique_response.content if critique_response.content else ""
        
        # 解析质疑结果
        import json
        import re
        critique_data = None
        
        # 尝试提取JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', critique_text, re.DOTALL)
        if json_match:
            try:
                critique_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        if critique_data is None:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', critique_text, re.DOTALL)
            if json_match:
                try:
                    critique_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # 打印质疑结果
        if critique_data:
            print("\n📋 Supervisor 质疑结果:")
            if critique_data.get("critical_questions"):
                print("  关键问题:")
                for i, q in enumerate(critique_data.get("critical_questions", [])[:5], 1):
                    print(f"    {i}. {q[:150]}...")
            if critique_data.get("identified_issues"):
                print("  发现的问题:")
                for i, issue in enumerate(critique_data.get("identified_issues", [])[:5], 1):
                    print(f"    {i}. {issue[:150]}...")
            if critique_data.get("score_evaluation"):
                print(f"  评分评估: {critique_data.get('score_evaluation', '')[:200]}...")
            
            # 将质疑结果添加到消息历史
            critique_message_content = f"""Supervisor Critical Review After Discussion:

Critical Questions:
{chr(10).join(f"- {q}" for q in critique_data.get("critical_questions", [])[:10])}

Identified Issues:
{chr(10).join(f"- {issue}" for issue in critique_data.get("identified_issues", [])[:10])}

Score Evaluation: {critique_data.get("score_evaluation", "")}

Recommendations:
{chr(10).join(f"- {rec}" for rec in critique_data.get("recommendations", [])[:10])}

Decision: {critique_data.get("decision", "iterate")}
Reasoning: {critique_data.get("reasoning", "")}
"""
            state["messages"].append(HumanMessage(content=critique_message_content))
            
            # 根据质疑结果决定下一步
            decision = critique_data.get("decision", "iterate")
            if decision == "report":
                # 检查评分是否真的满足要求
                if _check_scores_meet_requirements(state):
                    state["next_action"] = "report"
                else:
                    print("  ⚠️ Supervisor建议report，但评分不满足要求，强制iterate")
                    state["next_action"] = "iterate"
            else:
                state["next_action"] = "iterate"
        else:
            # 如果解析失败，使用评分检查决定
            print("  ⚠️ 质疑结果解析失败，使用评分检查决定下一步")
            if _check_scores_meet_requirements(state):
                state["next_action"] = "report"
            else:
                state["next_action"] = "iterate"
        
        # 清除标志
        del state["after_discussion"]
        
        print(f"\n✓ Supervisor质疑完成，将基于质疑结果进行决策")
        
        # 将质疑结果保存到状态中，供后续决策使用
        if critique_data:
            state["supervisor_critique"] = critique_data
        else:
            # 如果解析失败，创建一个基本的质疑结果
            state["supervisor_critique"] = {
                "critical_questions": [],
                "identified_issues": [],
                "score_evaluation": "评分检查中...",
                "recommendations": [],
                "decision": "iterate" if not _check_scores_meet_requirements(state) else "report",
                "reasoning": "基于评分要求进行决策"
            }
        
        # 继续执行正常的supervisor决策流程（质疑结果会影响决策）
    
    # 构建系统提示（包含工作流说明）
    base_prompt = _workflow_supervisor.prompt()
    workflow_details = """

Your Responsibilities:
1. **Read and Understand**:
   - First, use `read_file` tool to read the local task file `task/task_description.json`;
   - Then, extract the task name, goal, constraints, and data set path etc. from the task file;
   - **CRITICAL**: Use `read_file` tool to read the dataset file (CSV/TXT etc.) specified in the task file, and analyze its characteristics:
     * Count the number of rows (samples) and columns (features)
     * Identify all column names
     * Print these statistics clearly for reference
   - Based on the task information and dataset statistics, summarize the background and typical methods for this type of task.
2. **Plan and Assign**:
   - Based on the task JSON and dataset statistics, give a overall experimental plan;
   - Allocate the task to four expert agents (data_management, methodology, model_architect, result_analyst) in a structured and clear format, including the data/method/model/result metrics that each agent needs to focus on, and the detailed parameter specifications that need to be designed.
3. **Question and Evaluate**:
   - Identify the uncertainties or potential risks in the task, and propose questions and suggestions;
4. **Coordinate and Synthesize**:
   - Coordinate the four expert agents to work in parallel and iteratively discuss;
   - Collect their designs (including detailed parameter specifications and configuration details), and synthesize a complete pipeline report.

Workflow (three stages):
1. **Task Understanding and Dataset Analysis Stage (Supervisor-led)**
   - Use `read_file` to read `task/task_description.json`;
   - **MANDATORY**: Use `read_file` to read the dataset file specified in the task file (e.g., CSV file);
   - Analyze and report dataset statistics: number of rows, number of columns, column names;
   - Based on the task information and dataset characteristics, summarize the background and typical methods for this type of task;
   - Give an overall experimental plan;
   - Divide the plan into four sub-tasks, and allocate them to four expert agents.
2. **Expert Agent Design and Execution Stage**
   - data_management:
     - Use `read_file` to read the data set specified in the task (CSV/TXT etc.), and **CAREFULLY ANALYZE** each column's meaning, data type, and potential role:
       * **CRITICAL**: For each column, infer its meaning from column name, data patterns, and context (e.g., sequence columns, expression/activity columns, metadata columns, feature columns);
       * **CRITICAL**: Identify data types (sequences, numeric values, categorical labels, etc.) and their distributions;
       * **CRITICAL**: Determine which columns are input features, which are target variables, and which are metadata/auxiliary information;
       * **CRITICAL**: Analyze sequence length distribution if sequence columns exist (mean, median, range, outliers);
       * **CRITICAL**: Identify dataset type (MPRA/RNA-seq/ChIP-seq/ATAC-seq/etc.) based on column names and data characteristics;
     - CRITICALLY analyze dataset characteristics: dataset type, sequence length distribution, dataset size (number of samples), and data volume;
     - **CRITICAL**: Based on the inferred column meanings and dataset characteristics, select **REASONABLE and APPROPRIATE** data processing strategies:
       * For large datasets (>10K samples): focus on aggressive quality control, cleaning, filtering, and outlier removal;
       * For small datasets (<1K samples): prioritize data augmentation techniques (sequence augmentation, synthetic data generation);
       * For medium datasets (1K-10K): balance between quality control and augmentation;
       * **Tailor preprocessing steps to the specific data types and column meanings** (e.g., sequence-specific preprocessing for sequence columns, normalization strategies for expression columns);
     - Use `rag_search` to search the knowledge related to data preprocessing, division and enhancement strategies tailored to specific dataset types, column meanings, and sizes;
     - Give specific design specifications with detailed parameter values for data reading, preprocessing (tailored to dataset characteristics and column meanings), and division in the design.
   - methodology:
     - Use `rag_search` to search the knowledge related to the training method;
     - Design the corresponding training process with comprehensive parameter specifications and hyperparameter settings.
   - model_architect:
     - Analyze dataset size from dataset statistics to determine appropriate model complexity and parameter count;
     - Use `rag_search` to search the knowledge related to neural network architecture design, innovative architectures (attention, residual connections, etc.), and model complexity control;
     - Flexibly control model parameters and complexity based on data volume: smaller models for limited data, larger models for abundant data;
     - Design innovative, effective, and robust model structures with comprehensive parameter specifications and hyperparameter settings;
     - Provide clear rationale linking dataset characteristics to architectural choices.
   - result_analyst:
     - Use `rag_search` to search the knowledge related to evaluation metrics and statistical testing;
     - Design the corresponding evaluation scheme with comprehensive parameter specifications and hyperparameter settings.
3. **Iterative Discussion and Termination Decision (Supervisor-led)**
   - After the initial analysis, enter the expert discussion stage (discuss), and synthesize the opinions from different perspectives;
   - You need to raise questions and give suggestions to the expert's design, and decide whether to enter the next round of iteration (iterate);
   - When the design scheme with comprehensive parameter specifications is complete, generate the final experimental pipeline report and end (report -> end).

Final Report Structure:
The final report should focus on COMPREHENSIVE DESIGN SPECIFICATIONS with detailed parameter configurations, NOT on analysis of strengths/weaknesses.
The report should include four main sections with detailed design plans and parameter specifications:
1. **Data Usage Plan**: Complete design specifications for data source selection, preprocessing pipeline parameters, split strategy (ratios), augmentation methods with parameters, and quality control procedures
2. **Method Design**: Complete design specifications for training methodology including loss function parameters, optimization algorithm hyperparameters (learning rate, momentum, weight decay, etc.), regularization techniques with coefficients, and prior knowledge integration parameters
3. **Model Design**: Complete design specifications for neural network architecture including:
   - Layer-by-layer architecture with EXACT dimensions (input/output sizes, kernel sizes, stride, padding)
   - ALL hyperparameters: learning rate schedule, batch size, dropout rates, weight decay, optimizer parameters (Adam beta1/beta2, SGD momentum, etc.)
   - Activation functions and their parameters (LeakyReLU negative_slope, ELU alpha, etc.)
   - Regularization parameters (L1/L2 coefficients, dropout probabilities, batch norm momentum)
   - Initialization strategies and parameters (Xavier/Glorot parameters, He initialization parameters)
   - Training configuration (epochs, early stopping criteria, learning rate schedule parameters)
   - Model capacity (total parameter count, FLOPs estimation)
   - Interpretability mechanisms with their parameters
4. **Result Summary**: Complete design specifications for evaluation metrics with thresholds, validation strategy parameters, statistical testing procedures with significance levels, and biological validation methods

Each section MUST include:
- Detailed step-by-step design specifications
- COMPREHENSIVE parameter configurations and hyperparameter settings (this is CRITICAL)
- Specific parameter values, not just general descriptions
- Design rationale and justification for parameter choices
- Expected outcomes and how to interpret results

Decision rules (STRICT SCORING REQUIREMENTS):
- If task information is incomplete or missing, next_action = "request_info" (ask user for more details)
- If task information is complete and design is needed, next_action = "analyze"
- After initial analysis, next_action = "discuss"
- After discussion, check if scores meet requirements:
  * CRITICAL: Only set next_action = "report" if ALL of the following conditions are met:
    1. Average score across all expert agents >= 9.0/10
    2. Each individual expert agent score >= 8.5/10
    3. All agents have provided complete design plans with detailed parameter specifications
  * If scores do NOT meet these requirements, next_action = "iterate" (continue refinement)
- If completed and scores meet requirements, next_action = "end"

Available tools:
- read_file: Read content from local files (for reading task JSON and data files, especially CSV datasets)
- write_file: Write content to local files (for saving intermediate results or final report, if needed)

When using tools:
- Always use `read_file` to read the dataset file specified in the task description
- After reading a CSV dataset file, the system will automatically analyze and display:
  * Number of rows (samples)
  * Number of columns (features)
  * Column names
- Use these statistics to better understand the dataset and provide more accurate task assignments to expert agents

Please always think in a structured way and clearly state your reasoning for the decision.
"""
    system_prompt = base_prompt + workflow_details
    
    # 构建当前状态摘要
    status_summary = _create_status_summary(state)
    
    # 如果有Supervisor的质疑结果，添加到状态消息中
    supervisor_critique = state.get("supervisor_critique", {})
    critique_context = ""
    if supervisor_critique:
        critique_context = f"""

🔍 SUPERVISOR CRITICAL REVIEW (After Discussion):
Critical Questions:
{chr(10).join(f"- {q}" for q in supervisor_critique.get("critical_questions", [])[:5])}

Identified Issues:
{chr(10).join(f"- {issue}" for issue in supervisor_critique.get("identified_issues", [])[:5])}

Score Evaluation: {supervisor_critique.get("score_evaluation", "")}

Recommendations:
{chr(10).join(f"- {rec}" for rec in supervisor_critique.get("recommendations", [])[:5])}

You should consider these critical questions and issues when making your decision.
"""
        # 清除质疑结果（已使用）
        del state["supervisor_critique"]
    
    status_message = HumanMessage(content=f"""
Current state:
{status_summary}
{critique_context}

⚠️ STRICT SCORING REQUIREMENTS FOR REPORT GENERATION:
- You can ONLY set next_action = "report" if ALL of the following conditions are met:
  1. Average score across all expert agents >= 9.0/10
  2. Each individual expert agent score >= 8.5/10
  3. All agents have provided complete design plans with detailed parameter specifications
- If scores do NOT meet these requirements, you MUST set next_action = "iterate" to continue refinement

Please analyze the current state and decide the next action. Set the next_action field to one of the following:
- "request_info": Need more task information from user
- "analyze": Need to call expert agents for initial design
- "discuss": Need to facilitate discussion among expert agents to refine designs
- "iterate": Need more information or iterative optimization (USE THIS if scores < requirements)
- "report": Can generate a final report (ONLY if scores meet strict requirements above)
- "end": Work completed

Additionally, when you decide to call expert agents for design (next_action = "analyze" or "discuss" or "iterate"), you SHOULD provide a structured task plan for each expert agent.
Return it as a JSON object under the key "agent_task_plans", with the following structure:
{{
  "data_management": {{
    "title": "...",
    "description": "...",
    "task_prompt": "A detailed, specific task prompt for the data management expert. **CRITICAL REQUIREMENTS**:\n1. **Column Analysis**: When reading the dataset, CAREFULLY ANALYZE and INFER the meaning of each column:\n   - Examine column names, data patterns, and values to determine what each column represents (e.g., sequence data, expression/activity values, metadata, features);\n   - Identify data types (sequences, numeric, categorical) and their distributions;\n   - Determine which columns are input features, target variables, or auxiliary metadata;\n   - For sequence columns, analyze sequence length distribution (mean, median, range, outliers);\n   - Infer dataset type (MPRA/RNA-seq/ChIP-seq/ATAC-seq/etc.) based on column characteristics;\n2. **Dataset Characteristics Analysis**: Analyze dataset type, sequence length distribution, dataset size (number of samples), and data volume;\n3. **Reasonable Processing Strategy Selection**: Based on the inferred column meanings and dataset characteristics, select REASONABLE and APPROPRIATE data processing strategies:\n   - For large datasets (>10K samples): focus on aggressive quality control, cleaning, filtering, and outlier removal;\n   - For small datasets (<1K samples): prioritize data augmentation techniques (sequence augmentation, synthetic data generation);\n   - For medium datasets (1K-10K): balance between quality control and augmentation;\n   - **Tailor preprocessing steps to specific data types and column meanings** (e.g., sequence-specific preprocessing for sequence columns, normalization for expression columns);\n4. **Design Specifications**: Explain exactly what data to focus on, what parameter specifications to design for preprocessing tailored to dataset characteristics and column meanings. This should be comprehensive and tailored to the specific task, dataset characteristics, and inferred column meanings.",
  }},
  "methodology": {{
    "title": "...",
    "description": "...",
    "task_prompt": "A detailed, specific task prompt for the methodology expert, explaining exactly what training methodology to design, what loss functions to consider, what optimization strategies to use, etc.",
    ...
  }},
  "model_architect": {{
    "title": "...",
    "description": "...",
    "task_prompt": "A detailed, specific task prompt for the model architect expert. CRITICALLY: First analyze dataset size (number of samples) to determine appropriate model complexity and parameter count. For small datasets (<1K samples), design compact architectures with fewer parameters and strong regularization. For large datasets (>10K samples), design more expressive architectures. Encourage innovative, effective, and robust architecture designs (attention mechanisms, residual connections, multi-scale features, etc.). Explain exactly what architecture to design, what layer specifications to provide, what parameter estimations to make based on dataset size, etc. Provide clear rationale linking dataset characteristics to architectural choices.",
    ...
  }},
  "result_analyst": {{
    "title": "...",
    "description": "...",
    "task_prompt": "A detailed, specific task prompt for the result analyst, explaining exactly what evaluation metrics to use, what statistical tests to design, what validation strategy to implement, etc.",
    ...
  }}
}}

IMPORTANT: The "task_prompt" field for each agent should be:
- Detailed and specific to that agent's role
- Include concrete requirements and expectations
- Specify what parameter specifications need to be designed
- Reference the specific task context and dataset
- Be tailored to the current iteration and refinement needs

Please return the decision in JSON format (you can include optional fields as needed):
{{
    "reasoning": "Your reasoning process",
    "next_action": "request_info/analyze/discuss/iterate/report/end",
    "tool_calls": ["List of tools to call"],
    "agent_task_plans": {{
        "data_management": {{ ... }},
        "methodology": {{ ... }},
        "model_architect": {{ ... }},
        "result_analyst": {{ ... }}
    }}
}}
""")
    
    # 调用LLM（带工具）
    messages = state.get("messages", [])
    system_message = SystemMessage(content=system_prompt)
    
    # 消息历史管理：智能总结而不是简单截断
    # 估算 tokens（粗略估算：1 token ≈ 4 字符）
    total_chars = sum(len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0 for msg in messages)
    estimated_tokens = total_chars / 4
    
    MAX_MESSAGES = 50
    MAX_ESTIMATED_TOKENS = 200000  # 如果超过 200k tokens，进行总结
    
    if len(messages) > MAX_MESSAGES or estimated_tokens > MAX_ESTIMATED_TOKENS:
        if estimated_tokens > MAX_ESTIMATED_TOKENS:
            print(f"⚠️ 消息历史过长（估计 {estimated_tokens:.0f} tokens），进行智能总结...", flush=True)
            # 保留最近的 15 条消息，总结之前的消息
            recent_messages = messages[-15:]
            old_messages = messages[:-15]
            
            if old_messages:
                # 使用 LLM 总结旧消息
                try:
                    summary = await summarize_messages(old_messages, max_summary_length=2000)
                    # 创建总结消息
                    summary_message = HumanMessage(
                        content=f"[Previous Conversation Summary]\n{summary}\n\n[Continuing with recent messages...]"
                    )
                    # 用总结消息替换旧消息
                    messages = [summary_message] + recent_messages
                    print(f"  ✓ 消息历史已总结：{len(old_messages)} 条旧消息 -> 1 条总结消息", flush=True)
                except Exception as e:
                    print(f"  ⚠️ 消息总结失败: {e}，使用简单截断", flush=True)
                    # 如果总结失败，回退到简单截断
                    messages = messages[-MAX_MESSAGES:]
        else:
            print(f"⚠️ 消息历史过长 ({len(messages)}条)，截断至最近{MAX_MESSAGES}条", flush=True)
            # 保留最近的N条消息（优先保留非工具消息）
            recent_messages = messages[-MAX_MESSAGES:]
            messages = recent_messages
    
    # 构建消息列表（不直接修改state["messages"]，先处理工具调用）
    current_messages = [system_message] + messages + [status_message]
    
    # 处理工具调用（最多3轮）
    max_tool_iterations = 3
    tool_iteration = 0
    response = None
    new_messages_to_add = []  # 记录所有需要添加到state["messages"]的新消息
    
    while tool_iteration < max_tool_iterations:
        response = await _workflow_supervisor.llm_with_tools.ainvoke(current_messages)
        
        # 检查是否有工具调用
        has_tool_calls = False
        if hasattr(response, "tool_calls") and response.tool_calls:
            has_tool_calls = True
        elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
            has_tool_calls = True
        
        if not has_tool_calls:
            # 没有工具调用，这是最终响应
            break
        
        # 有工具调用，需要执行
        tool_iteration += 1
        current_messages.append(response)
        new_messages_to_add.append(response)
        
        # 执行工具调用
        from langchain_core.messages import ToolMessage
        tool_calls_list = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_list = response.tool_calls
        elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
            tool_calls_list = response.additional_kwargs["tool_calls"]
        
        for tool_call in tool_calls_list:
            # 处理不同格式的工具调用
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "") or tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("args", {}) or tool_call.get("function", {}).get("arguments", {})
                tool_call_id = tool_call.get("id", "") or tool_call.get("function", {}).get("id", "")
            else:
                tool_name = getattr(tool_call, "name", "")
                tool_args = getattr(tool_call, "args", {})
                tool_call_id = getattr(tool_call, "id", "")
            
            if isinstance(tool_args, str):
                import json
                try:
                    tool_args = json.loads(tool_args)
                except:
                    tool_args = {}
            
            # 查找并执行工具
            tool = next((t for t in _workflow_supervisor._tools if t.name == tool_name), None)
            if tool:
                try:
                    # 显示工具使用信息
                    print(f"  🔧 Supervisor 正在使用工具: {tool_name}", flush=True)
                    if tool_args:
                        print(f"     参数: {tool_args}", flush=True)
                    
                    if hasattr(tool, "_arun"):
                        result = await tool._arun(**tool_args)
                    else:
                        result = tool._run(**tool_args)
                    
                    # 如果读取了数据集文件，解析并打印统计信息
                    if tool_name == "read_file" and isinstance(result, dict) and result.get("success"):
                        file_data = result.get("data", {})
                        file_path = tool_args.get("file_path", "")
                        file_type = file_data.get("file_type", "text")
                        content = file_data.get("content", "")
                        
                        # 如果是CSV文件，解析统计信息
                        if (file_path.lower().endswith(".csv") or file_type == "csv") and content:
                            try:
                                import csv
                                import io
                                
                                # 处理不同格式的content
                                csv_content = content
                                if isinstance(content, dict):
                                    # 如果content是字典（JSON解析后的），尝试转换为字符串
                                    csv_content = str(content)
                                elif not isinstance(content, str):
                                    csv_content = str(content)
                                
                                # 解析CSV
                                csv_reader = csv.reader(io.StringIO(csv_content))
                                rows = list(csv_reader)
                                
                                if rows and len(rows) > 0:
                                    num_rows = len(rows) - 1  # 减去表头
                                    num_cols = len(rows[0]) if rows else 0
                                    column_names = rows[0] if rows else []
                                    
                                    print(f"\n  📊 数据集统计信息 ({file_path}):", flush=True)
                                    print(f"     ✓ 行数 (样本数): {num_rows}", flush=True)
                                    print(f"     ✓ 列数 (特征数): {num_cols}", flush=True)
                                    print(f"     ✓ 列名称 ({len(column_names)}个):", flush=True)
                                    for i, col_name in enumerate(column_names, 1):
                                        print(f"        {i}. {col_name}", flush=True)
                                    
                                    # 保存到state中
                                    if "dataset_statistics" not in state:
                                        state["dataset_statistics"] = {}
                                    state["dataset_statistics"][file_path] = {
                                        "num_rows": num_rows,
                                        "num_cols": num_cols,
                                        "column_names": column_names
                                    }
                                    
                                    # 对于大型CSV文件，只返回统计信息摘要，不返回完整内容
                                    # 这样可以避免消息历史过长
                                    MAX_CONTENT_LENGTH = 10000  # 10KB
                                    if len(csv_content) > MAX_CONTENT_LENGTH:
                                        # 只返回统计信息摘要
                                        summary_result = {
                                            "success": True,
                                            "data": {
                                                "file_path": file_path,
                                                "file_type": "csv",
                                                "summary": f"Large CSV file ({num_rows} rows, {num_cols} columns)",
                                                "dataset_statistics": {
                                                    "num_rows": num_rows,
                                                    "num_cols": num_cols,
                                                    "column_names": column_names
                                                },
                                                "size_bytes": len(csv_content),
                                                "line_count": len(rows),
                                                "note": "Full content not included due to size. Statistics extracted."
                                            }
                                        }
                                        result = summary_result
                                    else:
                                        # 小文件，保留完整内容，但添加统计信息
                                        if isinstance(result, dict) and "data" in result:
                                            result["data"]["dataset_statistics"] = {
                                                "num_rows": num_rows,
                                                "num_cols": num_cols,
                                                "column_names": column_names
                                            }
                            except Exception as e:
                                print(f"  ⚠️ 无法解析CSV文件 {file_path}: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                    
                    # 处理工具结果，对于过长的内容进行截断
                    MAX_TOOL_RESULT_LENGTH = 20000  # 20KB
                    if isinstance(result, dict):
                        result_data = result.get("data", {})
                        if isinstance(result_data, dict):
                            # 检查content字段
                            if "content" in result_data and isinstance(result_data["content"], str):
                                content = result_data["content"]
                                if len(content) > MAX_TOOL_RESULT_LENGTH:
                                    # 截断内容并添加提示
                                    truncated = content[:MAX_TOOL_RESULT_LENGTH]
                                    result_data["content"] = truncated + f"\n\n[内容已截断，原始长度: {len(content)} 字符]"
                                    result_data["truncated"] = True
                                    result_data["original_length"] = len(content)
                                    result["data"] = result_data
                    
                    result_content = str(result) if not isinstance(result, dict) else str(result.get("data", result))
                    # 如果结果内容仍然过长，直接截断
                    if len(result_content) > MAX_TOOL_RESULT_LENGTH:
                        result_content = result_content[:MAX_TOOL_RESULT_LENGTH] + f"\n\n[内容已截断，原始长度: {len(str(result))} 字符]"
                    
                    tool_msg = ToolMessage(
                        content=result_content,
                        tool_call_id=tool_call_id
                    )
                    current_messages.append(tool_msg)
                    new_messages_to_add.append(tool_msg)
                except Exception as e:
                    tool_msg = ToolMessage(
                        content=f"Tool execution failed: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                    current_messages.append(tool_msg)
                    new_messages_to_add.append(tool_msg)
            else:
                tool_msg = ToolMessage(
                    content=f"Tool '{tool_name}' not found",
                    tool_call_id=tool_call_id
                )
                current_messages.append(tool_msg)
                new_messages_to_add.append(tool_msg)
    
    # 更新消息历史（带截断保护）
    state["messages"].append(status_message)
    # 添加所有工具调用相关的消息和最终响应
    for msg in new_messages_to_add:
        state["messages"].append(msg)
    if response and response not in new_messages_to_add:
        state["messages"].append(response)
    
    # 再次检查消息历史长度，确保不超过限制
    MAX_MESSAGES_IN_STATE = 100  # state中保留更多消息用于日志，但LLM调用时只使用最近的
    if len(state["messages"]) > MAX_MESSAGES_IN_STATE:
        print(f"⚠️ 状态消息历史过长 ({len(state['messages'])}条)，截断至最近{MAX_MESSAGES_IN_STATE}条", flush=True)
        state["messages"] = state["messages"][-MAX_MESSAGES_IN_STATE:]
    
    # 解析决策
    decision = _parse_decision(response, state)
    proposed_action = decision.get("next_action", state.get("next_action", "request_info"))

    # 如果Supervisor提供了按角色划分的任务计划，写入state，供后续expert agent使用
    agent_plans = decision.get("agent_task_plans") or decision.get("agent_tasks")
    if isinstance(agent_plans, dict):
        state["agent_task_plans"] = agent_plans

    # 严格评分检查：如果Supervisor想进入report阶段，必须先检查评分是否满足要求
    if proposed_action == "report":
        scores_meet_requirements = _check_scores_meet_requirements(state)
        if not scores_meet_requirements:
            print("⚠️ 评分未达到要求，强制进入迭代阶段")
            print("   要求：平均分>=9.0，每个专家>=8.5")
            proposed_action = "iterate"
    
    state["next_action"] = proposed_action

    # 安全保护：如果迭代次数已达到上限，则强制进入报告阶段，避免无限 iterate 循环
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 8)
    if iteration >= max_iter and state["next_action"] in ("iterate", "analyze"):
        print(f"⚠️ 已达到最大迭代次数 ({max_iter})，强制进入报告阶段")
        state["next_action"] = "report"
    
    next_action = state.get("next_action", "request_info")
    print(f"✓ Supervisor决策: next_action = {next_action}")
    
    return state


async def request_info_node(state: REAgentState) -> REAgentState:
    """
    请求信息节点 - 当任务信息不完整时，请求用户提供更多信息
    """
    print("\n" + "="*60)
    print("📋 请求更多任务信息")
    print("="*60)
    
    # 检查缺失的信息
    missing_info = []
    if not state.get("task_description"):
        missing_info.append("任务描述 (task_description)")
    if not state.get("background"):
        missing_info.append("背景要求 (background)")
    if not state.get("dataset_info"):
        missing_info.append("数据集信息 (dataset_info)")
    
    # 添加用户消息，请求缺失信息
    request_message = HumanMessage(
        content=f"请提供以下缺失的信息：\n" + "\n".join(f"- {info}" for info in missing_info)
    )
    state["messages"].append(request_message)
    
    # 设置next_action为等待用户输入
    state["next_action"] = "waiting_user_input"
    
    print(f"⚠️ 需要用户提供以下信息：{', '.join(missing_info)}")
    
    return state


async def analyze_node(state: REAgentState) -> REAgentState:
    """
    分析节点 - 并行调用所有专家Agent进行分析
    """
    global _workflow_supervisor
    
    print("\n" + "="*60)
    print("🚀 开始并行分析 - 调用所有专家Agent")
    print("="*60)
    
    # 确保专家Agent已初始化
    global _workflow_expert_agents, _workflow_rag_system
    if _workflow_expert_agents is None:
        if _workflow_rag_system is None:
            _workflow_rag_system = HybridRAGSystem()
        _workflow_expert_agents = initialize_agents_with_rag(rag_system=_workflow_rag_system)
    
    # 获取Supervisor分配的按角色任务计划（如果有）
    agent_task_plans = state.get("agent_task_plans", {}) or {}

    # 并行调用所有专家Agent
    import asyncio
    tasks = {}
    for role, agent in _workflow_expert_agents.items():
        # 如果Supervisor为该角色提供了专门的task_plan，则优先使用
        role_plan = agent_task_plans.get(role)
        if isinstance(role_plan, dict):
            task_plan = role_plan
        else:
            # 回退：使用全局状态构建的通用任务计划
            task_plan = _build_task_plan(state)
        tasks[role] = agent.analyze(task_plan, state)
    
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    # 处理结果
    critiques = {}
    for (role, _), result in zip(tasks.items(), results):
        if isinstance(result, Exception):
            print(f"⚠️ Agent {role} 执行失败: {result}")
            critiques[role] = CritiqueResult(
                agent_role=role,
                score=0.0,
                strengths=[],
                weaknesses=[f"执行错误: {str(result)}"],
                recommendations=[],
                confidence=0.0,
                metadata={"error": str(result)}
            )
        else:
            critiques[role] = result
    
    # 更新状态中的分析结果
    state["data_critique"] = critiques.get("data_management")
    state["methodology_critique"] = critiques.get("methodology")
    state["model_critique"] = critiques.get("model_architect")
    state["results_critique"] = critiques.get("result_analyst")
    
    print("\n✓ 所有专家Agent分析完成")
    print(f"  - 数据管理: {critiques.get('data_management').score if critiques.get('data_management') else 'N/A'}/10")
    print(f"  - 方法学: {critiques.get('methodology').score if critiques.get('methodology') else 'N/A'}/10")
    print(f"  - 模型架构: {critiques.get('model_architect').score if critiques.get('model_architect') else 'N/A'}/10")
    print(f"  - 结果分析: {critiques.get('result_analyst').score if critiques.get('result_analyst') else 'N/A'}/10")
    
    # 分析完成后，进入讨论阶段（让Agent们讨论彼此的分析结果）
    state["next_action"] = "discuss"
    
    return state


async def discuss_node(state: REAgentState) -> REAgentState:
    """
    讨论节点 - 让四个专家Agent基于彼此的分析结果进行讨论
    
    每个Agent可以看到其他Agent的分析结果，并基于此进行补充或修正
    """
    global _workflow_expert_agents, _workflow_rag_system
    
    print("\n" + "="*60)
    print("💬 专家Agent讨论阶段")
    print("="*60)
    
    # 确保专家Agent已初始化
    if _workflow_expert_agents is None:
        if _workflow_rag_system is None:
            _workflow_rag_system = HybridRAGSystem()
        _workflow_expert_agents = initialize_agents_with_rag(rag_system=_workflow_rag_system)
    
    # 收集当前所有分析结果
    critiques = {
        "data_management": state.get("data_critique"),
        "methodology": state.get("methodology_critique"),
        "model_architect": state.get("model_critique"),
        "result_analyst": state.get("results_critique")
    }
    
    # 检查是否有分析结果
    if not any(critiques.values()):
        print("⚠️ 没有可讨论的分析结果，跳过讨论阶段")
        state["next_action"] = "analyze"
        return state
    
    # 构建讨论上下文（汇总所有Agent的分析结果）
    # 检查是否是迭代讨论（从iterate_node进入）
    is_iteration_discussion = state.get("iteration_summary") is not None
    iteration_summary = state.get("iteration_summary", "")
    
    discussion_context = ""
    if is_iteration_discussion:
        # 迭代讨论：包含迭代总结和建议
        discussion_context = f"{iteration_summary}\n\n"
        discussion_context += "=" * 60 + "\n"
        discussion_context += "Current Analysis Results from All Experts:\n"
        discussion_context += "=" * 60 + "\n\n"
    else:
        # 首次讨论：只包含分析结果
        discussion_context = "Below are the preliminary analysis results from all expert agents:\n\n"
    
    for role, critique in critiques.items():
        if critique:
            if isinstance(critique, dict):
                score = critique.get("score", 0)
                strengths = critique.get("strengths", [])
                weaknesses = critique.get("weaknesses", [])
                recommendations = critique.get("recommendations", [])
                design_summary = critique.get("metadata", {}).get("design_summary", "")
            elif hasattr(critique, "score"):
                score = critique.score
                strengths = critique.strengths
                weaknesses = critique.weaknesses
                recommendations = critique.recommendations
                design_summary = critique.metadata.get("design_summary", "") if hasattr(critique, "metadata") else ""
            else:
                continue
            
            role_name = {
                "data_management": "Data Management Expert",
                "methodology": "Methodology Expert",
                "model_architect": "Model Architect",
                "result_analyst": "Result Analyst"
            }.get(role, role)
            
            discussion_context += f"【{role_name}】Current Score: {score:.1f}/10\n"
            if design_summary:
                # 增大上下文窗口：显示更多内容（从200字符增加到800字符）
                summary_preview = design_summary[:800] if len(design_summary) > 800 else design_summary
                discussion_context += f"Design Summary: {summary_preview}"
                if len(design_summary) > 800:
                    discussion_context += "...\n"
                else:
                    discussion_context += "\n"
            if strengths:
                # 显示更多strengths（从3个增加到5个）
                discussion_context += f"Strengths: {', '.join(strengths[:5])}\n"
            if weaknesses:
                # 显示更多weaknesses（从3个增加到5个）
                discussion_context += f"Areas for Improvement: {', '.join(weaknesses[:5])}\n"
            if recommendations and not is_iteration_discussion:
                # 首次讨论时显示更多建议（从2个增加到5个）
                discussion_context += f"Recommendations: {', '.join(recommendations[:5])}\n"
            discussion_context += "\n"
    
    # 让每个Agent基于讨论上下文进行补充分析
    print("📝 各Agent正在基于讨论结果进行补充分析...")
    
    task_plan = _build_task_plan(state)
    updated_critiques = {}
    
    for role, agent in _workflow_expert_agents.items():
        current_critique = critiques.get(role)
        if not current_critique:
            continue
        
        print(f"  - {agent.title} 正在补充分析...")
        
        # 构建讨论提示
        if is_iteration_discussion:
            # 迭代讨论：强调基于建议更新设计方案和评分
            discussion_prompt = f"""⚠️ CRITICAL LANGUAGE REQUIREMENT:
- You MUST write ALL responses in English (EN). Do NOT use Chinese, Japanese, or any other language.
- All text in your response MUST be in English for international publication standards.

You are in an ITERATIVE DISCUSSION round. Based on the following context, please UPDATE your design and score:

{discussion_context}

⚠️ ITERATION REQUIREMENTS:
1. **Review the optimization recommendations** provided above - these are specific suggestions for improvement
2. **Update your design** based on these recommendations:
   - Improve your design specifications (detailed_design) to address the recommendations
   - Add missing parameter values, hyperparameter settings, and configuration details
   - For model architect: enhance architecture specifications with exact dimensions and all hyperparameters
   - Correct identified issues and add missing design details
3. **Update your score** based on the improvements:
   - If you have addressed the recommendations and improved your design with detailed parameters, you may increase your score
   - Score should reflect the CURRENT quality of your design after incorporating the recommendations
   - Only give score >= 8.5 if your design is truly comprehensive with detailed parameter specifications
4. **Provide updated recommendations** - new suggestions for further design improvement with specific parameter suggestions (if any)

Consider:
- Are the recommendations from other experts valid and applicable to your design?
- What specific parameter improvements can you make to your design based on these recommendations?
- How do these improvements affect the quality and completeness of your design specifications?

**Available Tools:**
- You CAN use `rag_search` to retrieve relevant knowledge from the knowledge base to support your design improvements
- You CAN use `read_file` to read dataset files or other relevant files if needed
- Use tools as needed to enhance your design specifications with accurate information

Please return the UPDATED analysis in JSON format:
{{
    "score": <0-10 float score, updated based on improvements>,
    "strengths": ["updated strength1", "updated strength2", ...],
    "weaknesses": ["remaining weakness1", "remaining weakness2", ...],
    "recommendations": ["new recommendation1", "new recommendation2", ...],
    "confidence": <0-1 float>,
    "discussion_notes": "Brief notes on what you improved based on the recommendations"
}}

IMPORTANT: Your score should reflect the CURRENT state of your design after incorporating the recommendations.
"""
        else:
            # 首次讨论：基于其他专家的意见进行补充
            discussion_prompt = f"""⚠️ CRITICAL LANGUAGE REQUIREMENT:
- You MUST write ALL responses in English (EN). Do NOT use Chinese, Japanese, or any other language.
- All text in your response MUST be in English for international publication standards.

Based on the following discussion context, please supplement or correct your analysis:

{discussion_context}

Consider:
1. Whether the opinions of other experts are consistent with your analysis?
2. Whether there are places that need to be supplemented or corrected?
3. Whether to agree with the suggestions of other experts?

**Available Tools:**
- You CAN use `rag_search` to retrieve relevant knowledge from the knowledge base to support your analysis
- You CAN use `read_file` to read dataset files or other relevant files if needed
- Use tools as needed to enhance your analysis with accurate information

Please return the updated analysis in JSON format:
{{
    "score": <0-10 float score>,
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "recommendations": ["recommendation1", "recommendation2", ...],
    "confidence": <0-1 float>,
    "discussion_notes": "Your discussion notes"
}}
"""
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=agent.prompt() + "\n\nYou are now participating in the expert discussion, and you can supplement or correct your analysis based on the opinions of other experts. You can use available tools (rag_search, read_file, etc.) to retrieve relevant knowledge or data to support your analysis."),
                HumanMessage(content=discussion_prompt)
            ]
            
            response = await agent.llm_with_tools.ainvoke(messages)
            
            # 处理工具调用（如果有）
            max_tool_iterations = 3
            tool_iteration = 0
            while tool_iteration < max_tool_iterations:
                has_tool_calls = False
                if hasattr(response, "tool_calls") and response.tool_calls:
                    has_tool_calls = True
                elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
                    has_tool_calls = True
                
                if not has_tool_calls:
                    break
                
                tool_iteration += 1
                messages.append(response)
                
                # 执行工具调用
                from langchain_core.messages import ToolMessage
                tool_calls_list = []
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_calls_list = response.tool_calls
                elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
                    tool_calls_list = response.additional_kwargs["tool_calls"]
                
                for tool_call in tool_calls_list:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "") or tool_call.get("function", {}).get("name", "")
                        tool_args = tool_call.get("args", {}) or tool_call.get("function", {}).get("arguments", {})
                        tool_call_id = tool_call.get("id", "") or tool_call.get("function", {}).get("id", "")
                    else:
                        tool_name = getattr(tool_call, "name", "")
                        tool_args = getattr(tool_call, "args", {})
                        tool_call_id = getattr(tool_call, "id", "")
                    
                    if isinstance(tool_args, str):
                        import json
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            tool_args = {}
                    
                    # 查找并执行工具
                    tool = next((t for t in agent._tools if t.name == tool_name), None)
                    if tool:
                        try:
                            if hasattr(tool, "_arun"):
                                result = await tool._arun(**tool_args)
                            else:
                                result = tool._run(**tool_args)
                            
                            result_content = str(result) if not isinstance(result, dict) else str(result.get("data", result))
                            messages.append(ToolMessage(
                                content=result_content,
                                tool_call_id=tool_call_id
                            ))
                        except Exception as e:
                            messages.append(ToolMessage(
                                content=f"Tool execution failed: {str(e)}",
                                tool_call_id=tool_call_id
                            ))
                
                # 继续调用LLM
                follow_up = HumanMessage(content="Please provide your updated analysis in JSON format as requested.")
                messages.append(follow_up)
                response = await agent.llm_with_tools.ainvoke(messages)
            
            response_text = response.content if response.content else ""
            
            # 解析响应（使用更健壮的解析方法）
            import json
            import re
            
            updated_data = None
            
            # 方法1: 尝试提取JSON代码块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    updated_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 方法2: 尝试提取普通JSON对象
            if updated_data is None:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        updated_data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # 方法3: 如果还是失败，尝试从文本中提取字段
            if updated_data is None:
                updated_data = {}
                # 获取原有值作为默认值
                old_score = current_critique.score if hasattr(current_critique, "score") else (current_critique.get("score", 5.0) if isinstance(current_critique, dict) else 5.0)
                old_strengths = current_critique.strengths if hasattr(current_critique, "strengths") else (current_critique.get("strengths", []) if isinstance(current_critique, dict) else [])
                old_weaknesses = current_critique.weaknesses if hasattr(current_critique, "weaknesses") else (current_critique.get("weaknesses", []) if isinstance(current_critique, dict) else [])
                old_recommendations = current_critique.recommendations if hasattr(current_critique, "recommendations") else (current_critique.get("recommendations", []) if isinstance(current_critique, dict) else [])
                old_confidence = current_critique.confidence if hasattr(current_critique, "confidence") else (current_critique.get("confidence", 0.5) if isinstance(current_critique, dict) else 0.5)
                
                # 尝试提取分数
                score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
                if score_match:
                    try:
                        updated_data["score"] = float(score_match.group(1))
                    except:
                        updated_data["score"] = old_score
                else:
                    updated_data["score"] = old_score
                
                # 尝试提取列表字段
                for key, old_value in [("strengths", old_strengths), ("weaknesses", old_weaknesses), ("recommendations", old_recommendations)]:
                    pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
                    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        items = re.findall(r'"([^"]+)"', match.group(1))
                        updated_data[key] = items if items else old_value
                    else:
                        updated_data[key] = old_value
                
                # 尝试提取confidence
                conf_match = re.search(r'"confidence"\s*:\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
                if conf_match:
                    try:
                        updated_data["confidence"] = float(conf_match.group(1))
                    except:
                        updated_data["confidence"] = old_confidence
                else:
                    updated_data["confidence"] = old_confidence
                
                # 尝试提取discussion_notes
                notes_match = re.search(r'"discussion_notes"\s*:\s*"([^"]+)"', response_text, re.DOTALL | re.IGNORECASE)
                if notes_match:
                    updated_data["discussion_notes"] = notes_match.group(1)
                else:
                    updated_data["discussion_notes"] = f"Discussion response: {response_text[:200]}" if response_text else "No response"
            
            # 如果仍然无法解析，使用原有分析结果
            if not updated_data or not isinstance(updated_data, dict) or not response_text:
                print(f"    ⚠️ {agent.title} JSON解析失败（响应为空或格式错误），保留原有分析结果")
                updated_critiques[role] = current_critique
                continue
            
            # 创建更新后的CritiqueResult
            # 重要：保留原始metadata中的所有信息（特别是detailed_design），只添加讨论相关的字段
            from Agents.state import CritiqueResult
            original_metadata = current_critique.metadata if hasattr(current_critique, "metadata") else {}
            if isinstance(current_critique, dict):
                original_metadata = current_critique.get("metadata", {})
            
            # 合并metadata：保留原始的所有字段，只更新讨论相关字段
            merged_metadata = original_metadata.copy()  # 先复制原始metadata
            merged_metadata.update({
                "discussion_notes": updated_data.get("discussion_notes", ""),
                "updated_after_discussion": True
            })
            # 确保detailed_design和design_summary不被覆盖
            if "detailed_design" not in merged_metadata or not merged_metadata.get("detailed_design"):
                # 如果detailed_design丢失了，尝试从原始metadata恢复
                if "detailed_design" in original_metadata:
                    merged_metadata["detailed_design"] = original_metadata["detailed_design"]
            
            updated_critique = CritiqueResult(
                agent_role=role,
                score=updated_data.get("score", current_critique.score if hasattr(current_critique, "score") else current_critique.get("score", 0)),
                strengths=updated_data.get("strengths", []),
                weaknesses=updated_data.get("weaknesses", []),
                recommendations=updated_data.get("recommendations", []),
                confidence=updated_data.get("confidence", 0.5),
                metadata=merged_metadata  # 使用合并后的metadata，保留所有原始信息
            )
            
            updated_critiques[role] = updated_critique
            print(f"    ✓ {agent.title} 补充分析完成")
            
        except Exception as e:
            print(f"    ⚠️ {agent.title} 讨论分析失败: {e}")
            # 保留原有分析结果
            updated_critiques[role] = current_critique
    
    # 更新状态中的分析结果
    if updated_critiques:
        state["data_critique"] = updated_critiques.get("data_management")
        state["methodology_critique"] = updated_critiques.get("methodology")
        state["model_critique"] = updated_critiques.get("model_architect")
        state["results_critique"] = updated_critiques.get("result_analyst")
        
        # 打印更新后的评分
        print(f"\n✓ 讨论完成，{len(updated_critiques)} 个Agent更新了分析结果")
        print("   更新后的评分：", end="")
        for role, critique in updated_critiques.items():
            if critique:
                score = critique.score if hasattr(critique, "score") else (critique.get("score", 0) if isinstance(critique, dict) else 0)
                role_short = {
                    "data_management": "数据",
                    "methodology": "方法",
                    "model_architect": "模型",
                    "result_analyst": "结果"
                }.get(role, role)
                print(f"{role_short}:{score:.1f} ", end="")
        print()
    
    # 清除迭代标记（如果存在），避免影响后续流程
    if "iteration_summary" in state:
        del state["iteration_summary"]
    if "recommendations_by_role" in state:
        del state["recommendations_by_role"]
    
    # 设置标志：讨论完成后需要Supervisor质疑和评估
    state["after_discussion"] = True
    
    # 讨论后，让Supervisor根据评分决定下一步
    # Supervisor会检查评分是否满足要求（平均分>=9，每个专家>=8.5）
    # 如果达标 → report，如果不达标 → iterate（继续迭代讨论）
    # 注意：这里不设置next_action，让supervisor节点自然处理
    # workflow图已经配置了discuss -> supervisor的边
    
    return state


async def iterate_node(state: REAgentState) -> REAgentState:
    """
    迭代节点 - 根据分析结果进行迭代优化
    
    收集所有专家的建议，然后触发讨论阶段，让Agent基于建议和上下文再次讨论并更新得分
    """
    print("\n" + "="*60)
    print("🔄 迭代优化阶段")
    print("="*60)
    
    # 每进入一次迭代节点，视为一轮新的优化迭代
    iteration_count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration_count
    max_iterations = state.get("max_iterations", 8)
    
    if iteration_count >= max_iterations:
        print(f"⚠️ 已达到最大迭代次数 ({max_iterations})，进入报告生成阶段")
        state["next_action"] = "report"
        return state
    
    # 收集所有建议（从所有专家的分析结果中提取）
    all_recommendations = []
    critiques = {
        "data_management": state.get("data_critique"),
        "methodology": state.get("methodology_critique"),
        "model_architect": state.get("model_critique"),
        "result_analyst": state.get("results_critique")
    }
    
    # 按角色组织建议，便于在讨论中引用
    recommendations_by_role = {}
    for role, critique in critiques.items():
        if critique:
            if isinstance(critique, dict):
                recommendations = critique.get("recommendations", [])
                score = critique.get("score", 0)
            elif hasattr(critique, "recommendations"):
                recommendations = critique.recommendations
                score = critique.score if hasattr(critique, "score") else 0
            else:
                recommendations = []
                score = 0
            
            if recommendations:
                role_name = {
                    "data_management": "数据管理专家",
                    "methodology": "方法学专家",
                    "model_architect": "模型架构师",
                    "result_analyst": "结果分析师"
                }.get(role, role)
                recommendations_by_role[role_name] = recommendations
                all_recommendations.extend(recommendations)
    
    # 构建迭代建议总结消息（包含所有建议和当前评分情况）
    iteration_summary = f"""🔄 第 {iteration_count} 轮迭代讨论

当前评分情况：
"""
    for role, critique in critiques.items():
        if critique:
            if isinstance(critique, dict):
                score = critique.get("score", 0)
            elif hasattr(critique, "score"):
                score = critique.score
            else:
                score = 0
            
            role_name = {
                "data_management": "数据管理专家",
                "methodology": "方法学专家",
                "model_architect": "模型架构师",
                "result_analyst": "结果分析师"
            }.get(role, role)
            iteration_summary += f"  - {role_name}: {score:.1f}/10\n"
    
    iteration_summary += f"\n汇总的优化建议（共 {len(all_recommendations)} 条）：\n\n"
    for i, rec in enumerate(all_recommendations[:15], 1):  # 最多显示15条
        iteration_summary += f"{i}. {rec}\n"
    
    if len(all_recommendations) > 15:
        iteration_summary += f"\n... 还有 {len(all_recommendations) - 15} 条建议未显示\n"
    
    iteration_summary += "\n请基于以上建议和所有专家的分析结果，进行讨论并更新您的设计方案和评分。"
    
    # 将迭代建议总结保存到状态中，供讨论节点使用
    state["iteration_summary"] = iteration_summary
    state["recommendations_by_role"] = recommendations_by_role
    
    # 添加迭代建议消息到消息历史
    iterate_message = HumanMessage(content=iteration_summary)
    state["messages"].append(iterate_message)
    
    # 直接触发讨论阶段（不是重新分析）
    state["next_action"] = "discuss"
    
    print(f"📝 收集到 {len(all_recommendations)} 条优化建议，准备进入讨论阶段...")
    print(f"   当前评分：", end="")
    for role, critique in critiques.items():
        if critique:
            score = critique.score if hasattr(critique, "score") else (critique.get("score", 0) if isinstance(critique, dict) else 0)
            role_short = {
                "data_management": "数据",
                "methodology": "方法",
                "model_architect": "模型",
                "result_analyst": "结果"
            }.get(role, role)
            print(f"{role_short}:{score:.1f} ", end="")
    print()
    
    return state


async def report_node(state: REAgentState) -> REAgentState:
    """
    报告节点 - 生成最终优化报告
    """
    print("\n" + "="*60)
    print("📊 生成最终优化报告")
    print("="*60)
    
    # 收集所有分析结果
    critiques = {
        "data_management": state.get("data_critique"),
        "methodology": state.get("methodology_critique"),
        "model_architect": state.get("model_critique"),
        "result_analyst": state.get("results_critique")
    }
    
    # 过滤掉None值
    critiques = {k: v for k, v in critiques.items() if v is not None}
    
    # 计算总体评分
    scores = []
    for critique in critiques.values():
        if critique:
            if isinstance(critique, dict):
                score = critique.get("score", 0)
            elif hasattr(critique, "score"):
                score = critique.score
            else:
                score = 0
            if score > 0:
                scores.append(score)
    
    overall_score = sum(scores) / len(scores) if scores else 0.0
    
    # 收集优先建议（重点关注可执行的实施方案建议）
    priority_recommendations = []
    for critique in critiques.values():
        if critique:
            if isinstance(critique, dict):
                recs = critique.get("recommendations", [])
            elif hasattr(critique, "recommendations"):
                recs = critique.recommendations
            else:
                recs = []
            
            # 收集所有建议（重点关注实施方案）
            if recs:
                priority_recommendations.extend(recs[:3])  # 每个Agent最多3条
    
    # 构建报告 - 生成详细的实验设计报告
    task_title = state.get("task_description", "Gene Regulatory Element Design Task")
    task_background = state.get("background", "")
    dataset_info = state.get("dataset_info", "")
    
    # 从各Agent的设计中提取详细信息（完整保留，不进行任何概括或删减）
    data_design = {}
    method_design = {}
    model_design = {}
    result_design = {}
    
    if critiques.get("data_management"):
        dm_critique = critiques["data_management"]
        dm_meta = dm_critique.metadata if hasattr(dm_critique, "metadata") else {}
        # 完整保留 detailed_design，不做任何修改
        data_design = dm_meta.get("detailed_design", {})
        # 如果 detailed_design 为空，尝试从其他字段获取
        if not data_design:
            print("⚠️ Data Management detailed_design is empty, checking metadata...")
            print(f"   Metadata keys: {list(dm_meta.keys())}")
    
    if critiques.get("methodology"):
        method_critique = critiques["methodology"]
        method_meta = method_critique.metadata if hasattr(method_critique, "metadata") else {}
        method_design = method_meta.get("detailed_design", {})
        if not method_design:
            print("⚠️ Methodology detailed_design is empty, checking metadata...")
            print(f"   Metadata keys: {list(method_meta.keys())}")
    
    if critiques.get("model_architect"):
        model_critique = critiques["model_architect"]
        model_meta = model_critique.metadata if hasattr(model_critique, "metadata") else {}
        model_design = model_meta.get("detailed_design", {})
        if not model_design:
            print("⚠️ Model Architect detailed_design is empty, checking metadata...")
            print(f"   Metadata keys: {list(model_meta.keys())}")
    
    if critiques.get("result_analyst"):
        result_critique = critiques["result_analyst"]
        result_meta = result_critique.metadata if hasattr(result_critique, "metadata") else {}
        result_design = result_meta.get("detailed_design", {})
        if not result_design:
            print("⚠️ Result Analyst detailed_design is empty, checking metadata...")
            print(f"   Metadata keys: {list(result_meta.keys())}")
    
    report = OptimizationReport(
        title=f"Experimental Design Report: {task_title}",
        summary=f"Based on the task background and data set information, after {state.get('iteration_count', 0)} rounds of expert design, a complete experimental design scheme is generated. The overall feasibility score: {overall_score:.1f}/10",
        critiques={
            role: critique for role, critique in critiques.items() if critique
        },
        overall_score=overall_score,
        priority_recommendations=priority_recommendations[:10],
        metadata={
            "iteration_count": state.get("iteration_count", 0),
            "task_description": task_title,
            "task_background": task_background,
            "dataset_info": dataset_info,
            "data_usage_plan": data_design,
            "method_design": method_design,
            "model_design": model_design,
            "result_summary": result_design
        }
    )
    
    # 将报告转换为字典格式存储（包含详细的实验设计）
    # 重点：突出实施方案和代码，减少优缺点分析
    state["final_report"] = {
        "title": report.title,
        "summary": report.summary,
        "overall_score": report.overall_score,
        "priority_recommendations": report.priority_recommendations,
        "task_information": {
            "description": task_title,
            "background": task_background,
            "dataset_info": dataset_info
        },
        "experimental_design": {
            "1_data_usage_plan": report.metadata.get("data_usage_plan", {}),
            "2_method_design": report.metadata.get("method_design", {}),
            "3_model_design": report.metadata.get("model_design", {}),
            "4_result_summary": report.metadata.get("result_summary", {})
        },
        "expert_analyses": {
            role: {
                "score": c.score if hasattr(c, "score") else c.get("score", 0),
                "design_summary": c.metadata.get("design_summary", "") if hasattr(c, "metadata") else "",
                # 完整保留 detailed_design，不进行任何概括或删减，直接拼接各专家的最终方案
                "implementation_plan": c.metadata.get("detailed_design", {}) if hasattr(c, "metadata") else {},
                "recommendations": c.recommendations if hasattr(c, "recommendations") else c.get("recommendations", []),
                # 检索到的知识条目（id + 内容），用于可解释性
                "retrieved_knowledge": c.metadata.get("retrieved_knowledge", []) if hasattr(c, "metadata") else [],
                # 同时保留完整的 metadata，确保不丢失任何信息
                "full_metadata": c.metadata if hasattr(c, "metadata") else {}
            }
            for role, c in report.critiques.items()
        },
        "metadata": report.metadata
    }
    
    # 添加报告消息（包含实验设计摘要，重点突出实施方案）
    report_content = f"""The experimental design report has been generated

{report.summary}

Experimental Design Implementation Plan:
1. Data Usage Plan: Contains {len(data_design)} design sections with detailed parameter specifications
2. Method Design: Contains {len(method_design)} design sections with detailed parameter specifications
3. Model Design: Contains {len(model_design)} design sections with comprehensive parameter configurations (layer dimensions, hyperparameters, etc.)
4. Result Summary: Contains {len(result_design)} design sections with detailed parameter specifications

Key Implementation Recommendations:
""" + "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(report.priority_recommendations))
    
    report_message = AIMessage(content=report_content)
    state["messages"].append(report_message)
    
    print(f"✓ The experimental design report has been generated, the overall score: {overall_score:.1f}/10")
    print(f"  - Number of priority recommendations: {len(priority_recommendations)}")
    
    # 报告生成后，进入结束阶段
    state["next_action"] = "end"
    
    return state


# ==================== 路由函数 ====================

def route_decision(state: REAgentState) -> Literal["request_info", "analyze", "discuss", "iterate", "report", "end"]:
    """
    根据next_action路由到下一个节点
    """
    next_action = state.get("next_action", "request_info")

    # 全局安全保护：超过最大迭代次数后，强制进入报告阶段，避免无限循环
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 8)
    if iteration >= max_iter and next_action in ("iterate", "analyze"):
        print(f"⚠️ 已达到最大迭代次数 ({max_iter})，路由强制切换为 report")
        next_action = "report"
    
    # 确保next_action是有效的
    valid_actions = ["request_info", "analyze", "discuss", "iterate", "report", "end"]
    if next_action not in valid_actions:
        print(f"⚠️ Invalid next_action: {next_action}, default using request_info")
        next_action = "request_info"
    
    return next_action


# ==================== 工作流图构建 ====================

def create_workflow_graph(rag_system: HybridRAGSystem = None, supervisor: Agent = None) -> StateGraph:
    """
    创建LangGraph工作流图
    
    Args:
        rag_system: RAG系统实例（可选）
        supervisor: Supervisor实例（可选，如果提供则使用，否则创建新实例）
    
    Returns:
        编译后的StateGraph应用
    """
    # 设置工作流组件
    if rag_system is None:
        rag_system = HybridRAGSystem()
    
    if supervisor is None:
        supervisor = initialize_supervisor(rag_system=rag_system)
    
    expert_agents = initialize_agents_with_rag(rag_system=rag_system)
    set_workflow_components(supervisor=supervisor, expert_agents=expert_agents, rag_system=rag_system)
    
    # 创建状态图
    workflow = StateGraph(REAgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("request_info", request_info_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("discuss", discuss_node)
    workflow.add_node("iterate", iterate_node)
    workflow.add_node("report", report_node)
    
    # 设置入口点
    workflow.set_entry_point("supervisor")
    
    # 添加条件边：从supervisor根据next_action路由
    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "request_info": "request_info",
            "analyze": "analyze",
            "discuss": "discuss",
            "iterate": "iterate",
            "report": "report",
            "end": END
        }
    )
    
    # 从request_info回到supervisor（等待用户输入后）
    workflow.add_edge("request_info", "supervisor")
    
    # 从analyze到discuss（分析完成后进入讨论）
    workflow.add_edge("analyze", "discuss")
    
    # 从discuss到supervisor（讨论后重新决策）
    workflow.add_edge("discuss", "supervisor")
    
    # 从iterate到discuss（迭代时直接进入讨论，让Agent基于建议更新设计）
    workflow.add_edge("iterate", "discuss")
    
    # 从report到END（报告生成后结束）
    workflow.add_edge("report", END)
    
    # 编译图
    app = workflow.compile()
    
    print("✓ LangGraph工作流图创建完成")
    print("  节点: supervisor -> [request_info/analyze -> discuss/iterate/report] -> end")
    
    return app


# ==================== 辅助函数 ====================

def create_initial_state(
    task_description: str = None,
    background: str = None,
    dataset_info: str = None,
    methodology: str = None,
    model_architecture: str = None,
    evaluation_metrics: str = None,
    rag_system: HybridRAGSystem = None,
    max_iterations: int = 8
) -> REAgentState:
    """
    创建初始状态
    
    Args:
        task_description: 任务描述
        background: 背景要求
        dataset_info: 数据集信息
        methodology: 方法描述（可选）
        model_architecture: 模型架构（可选）
        evaluation_metrics: 评估指标（可选）
        rag_system: RAG系统实例（可选）
        max_iterations: 最大迭代次数
    
    Returns:
        初始化的REAgentState
    """
    from langchain_core.messages import HumanMessage

    # 如果用户未显式提供任务信息，尝试从本地JSON读取默认任务描述
    additional_info: Dict[str, Any] = {}
    if not task_description and not background and not dataset_info:
        task_json_path = Path("task/task_description.json")
        try:
            if task_json_path.exists():
                with task_json_path.open("r", encoding="utf-8") as f:
                    task_json = json.load(f)
                additional_info["task_json"] = task_json

                # 从JSON中提取信息，填充到文本字段
                task_name = task_json.get("task_name", "")
                task_goal = task_json.get("task_goal", "")
                task_requirements = task_json.get("task_requirements", "")
                task_dataset = task_json.get("task_dataset", {})

                if not task_description:
                    # 任务描述偏向任务名称
                    task_description = task_name or task_goal
                if not background:
                    # 背景/目标与要求合并
                    background_parts = []
                    if task_goal:
                        background_parts.append(f"Goal: {task_goal}")
                    if task_requirements:
                        background_parts.append(f"Requirements: {task_requirements}")
                    background = "\n".join(background_parts) if background_parts else None
                if not dataset_info and task_dataset:
                    # 将数据集结构转成可读文本，方便Supervisor和Data Agent理解
                    ds_path = task_dataset.get("file_path", "")
                    ds_type = task_dataset.get("data_type", "")
                    ds_inputs = task_dataset.get("input_features", [])
                    ds_target = task_dataset.get("target_variable", "")
                    ds_constraint = task_dataset.get("key_constraint", "")
                    dataset_info = (
                        f"File path: {ds_path}; "
                        f"Data type: {ds_type}; "
                        f"Input features: {', '.join(ds_inputs) if ds_inputs else 'N/A'}; "
                        f"Target variable: {ds_target or 'N/A'}; "
                        f"Constraint: {ds_constraint or 'N/A'}"
                    )
        except Exception as e:
            # 失败时仅打印警告，不中断工作流
            print(f"⚠️ 读取默认任务文件 task/task_description.json 失败: {e}")

    # 创建初始消息（展示当前掌握的任务信息）
    initial_message = HumanMessage(
        content=(
            f"任务描述: {task_description or '待提供'}\n"
            f"背景要求: {background or '待提供'}\n"
            f"数据集信息: {dataset_info or '待提供'}"
        )
    )

    state: REAgentState = {
        "messages": [initial_message],
        "task_description": task_description,
        "background": background,
        "dataset_info": dataset_info,
        "methodology": methodology,
        "model_architecture": model_architecture,
        "evaluation_metrics": evaluation_metrics,
        "next_action": "request_info",
        "data_critique": None,
        "methodology_critique": None,
        "model_critique": None,
        "results_critique": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_report": None,
        "additional_info": additional_info or None,
    }

    # 如果提供了rag_system，设置工作流组件
    if rag_system:
        set_workflow_components(rag_system=rag_system)

    return state


# ==================== 导出 ====================

__all__ = [
    "create_workflow_graph",
    "create_initial_state",
    "set_workflow_components",
    "supervisor_node",
    "request_info_node",
    "analyze_node",
    "discuss_node",
    "iterate_node",
    "report_node",
    "route_decision"
]

