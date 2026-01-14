"""
Agents/state.py
定义Agent系统的状态和数据结构
"""

from typing import Dict, Any, List, Optional, TypedDict, Literal
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage


# ==================== Agent角色枚举 ====================

class AgentRole:
    """Agent角色定义"""
    DATA_MANAGEMENT = "data_management"
    METHODOLOGY = "methodology"
    MODEL_ARCHITECT = "model_architect"
    RESULT_ANALYST = "result_analyst"
    
    ALL_ROLES = [DATA_MANAGEMENT, METHODOLOGY, MODEL_ARCHITECT, RESULT_ANALYST]


# ==================== 状态定义 ====================

class REAgentState(TypedDict, total=False):
    """RE-Agent系统的状态"""
    messages: List[BaseMessage]
    
    # 任务信息（由用户提供）
    task_description: Optional[str]  # 任务描述
    background: Optional[str]  # 背景要求
    dataset_info: Optional[str]  # 数据集信息
    methodology: Optional[str]  # 方法描述（可选）
    model_architecture: Optional[str]  # 模型架构（可选）
    evaluation_metrics: Optional[str]  # 评估指标（可选）
    additional_info: Optional[Dict[str, Any]]  # 其他附加信息
    agent_task_plans: Optional[Dict[str, Dict[str, Any]]]  # 由Supervisor按角色分配的任务计划
    dataset_statistics: Optional[Dict[str, Dict[str, Any]]]  # 数据集统计信息（文件路径 -> {行数, 列数, 列名}）
    
    # 工作流控制
    next_action: Literal["request_info", "analyze", "discuss", "iterate", "report", "end"]
    
    # 专家分析结果
    data_critique: Optional[Dict[str, Any]]
    methodology_critique: Optional[Dict[str, Any]]
    model_critique: Optional[Dict[str, Any]]
    results_critique: Optional[Dict[str, Any]]
    
    # 迭代控制
    iteration_count: int
    max_iterations: int
    
    # 最终报告
    final_report: Optional[Dict[str, Any]]


# ==================== 分析结果模型 ====================

@dataclass
class CritiqueResult:
    """专家分析结果"""
    agent_role: str
    score: float  # 0-10分
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationReport:
    """优化报告"""
    title: str
    summary: str
    critiques: Dict[str, CritiqueResult]
    overall_score: float
    priority_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 工具注册表 ====================
# ==================== 工具基类 ====================

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


ToolResult = Dict[str, Any]


class BioAgentToolInput(BaseModel):
    """Bio-Agent工具的通用输入模型"""
    pass


class BioAgentTool(BaseTool):
    """
    Bio-Agent工具基类
    
    所有工具都应继承此类
    """
    
    def _run(self, *args, **kwargs) -> ToolResult:
        """同步执行（必须实现）"""
        raise NotImplementedError("工具必须实现_run方法")
    
    async def _arun(self, *args, **kwargs) -> ToolResult:
        """异步执行（可选）"""
        return self._run(*args, **kwargs)
    
    def format_success(self, data: Any, metadata: Dict[str, Any] = None) -> ToolResult:
        """格式化成功结果"""
        # 打印工具调用状态，便于在终端观察工具使用情况
        try:
            tool_name = getattr(self, "name", self.__class__.__name__)
            print(f"🛠️ Tool '{tool_name}' SUCCESS", flush=True)
        except Exception:
            # 日志失败不影响正常返回
            pass

        return {
            "status": "success",
            "success": True,
            "data": data,
            "error": None,
            "metadata": metadata or {}
        }
    
    def format_error(self, error: str, metadata: Dict[str, Any] = None) -> ToolResult:
        """格式化错误结果"""
        # 打印工具调用失败状态
        try:
            tool_name = getattr(self, "name", self.__class__.__name__)
            print(f"❌ Tool '{tool_name}' ERROR: {error}", flush=True)
        except Exception:
            pass

        return {
            "status": "error",
            "success": False,
            "data": None,
            "error": error,
            "metadata": metadata or {}
        }


# ==================== 工具1：RAG检索工具 ====================

from RAG.rag import HybridRAGSystem, AgentRAGInterface


class RAGSearchInput(BioAgentToolInput):
    """RAG检索工具输入"""
    query: str = Field(description="检索查询文本")
    agent_role: str = Field(
        default="data_management",
        description="智能体角色（data_management/methodology/model_architect/result_analyst），用于上下文理解"
    )
    top_k: int = Field(default=5, description="返回结果数量")


class RAGSearchTool(BioAgentTool):
    """RAG知识检索工具 - 从共享知识库检索相关文献和专业知识"""
    
    name: str = "rag_search"
    description: str = """
    从本地知识库检索基因调控元件设计相关的文献和专业知识。
    知识库包含：PubMed文献、arXiv预印本、bioRxiv预印本、PMC开放获取文章、GitHub代码库等。
    可以搜索：数据管理、训练方法、模型架构、评估指标等专业知识。
    """
    args_schema: type[BaseModel] = RAGSearchInput

    # 使用PrivateAttr存储底层RAG系统，避免pydantic字段校验错误
    _rag_system: HybridRAGSystem = PrivateAttr()

    def __init__(self, rag_system: Optional[HybridRAGSystem] = None, **data: Any):
        super().__init__(**data)
        self._rag_system = rag_system or HybridRAGSystem()
    
    async def _arun(
        self,
        query: str,
        agent_role: str = "data_management",
        top_k: int = 5
    ) -> ToolResult:
        """执行RAG检索"""
        try:
            # 创建Agent接口
            rag_interface = AgentRAGInterface(self._rag_system, agent_role)
            
            # 执行检索（现在只有共享库）
            results = await rag_interface.query(
                query=query,
                strategy="hybrid",  # 保留参数以兼容，但实际只使用共享库
                top_k=top_k
            )
            
            # 统计总结果数（现在只有shared_results）
            total_results = len(results.shared_results)
            
            # 格式化返回
            formatted_results = {
                "query": query,
                "results": [
                    {
                        "content": r.content,
                        "score": round(r.score, 4),
                        "metadata": {
                            "doc_id": r.metadata.get("doc_id", ""),
                            "title": r.metadata.get("title", ""),
                            "source": r.metadata.get("source", ""),
                            "authors": r.metadata.get("authors", ""),
                            "journal": r.metadata.get("journal", ""),
                            "date": r.metadata.get("date", ""),
                            "doi": r.metadata.get("doi", "")
                        }
                    }
                    for r in results.shared_results
                ],
                "total_results": total_results,
                "retrieval_time_ms": round(results.retrieval_time_ms, 2)
            }
            
            return self.format_success(
                data=formatted_results,
                metadata={
                    "agent_role": agent_role,
                    "top_k": top_k
                }
            )
        
        except Exception as e:
            return self.format_error(
                error=f"RAG检索失败: {str(e)}",
                metadata={"query": query, "agent_role": agent_role}
            )


# ==================== 工具2：文件读写工具 ====================

import json
from pathlib import Path


class FileReadInput(BioAgentToolInput):
    """文件读取工具输入"""
    file_path: str = Field(description="要读取的文件路径（相对或绝对路径）")
    encoding: str = Field(default="utf-8", description="文件编码")


class FileReadTool(BioAgentTool):
    """文件读取工具 - 读取文本文件或JSON文件"""
    
    name: str = "read_file"
    description: str = """
    读取文件内容。支持文本文件（.txt, .md等）和JSON文件（.json）。
    返回文件内容，如果是JSON文件则自动解析为字典。
    """
    args_schema: type[BaseModel] = FileReadInput
    
    async def _arun(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> ToolResult:
        """读取文件"""
        try:
            path = Path(file_path)
            project_root = Path(__file__).parent.parent  # Agents/state.py -> RE-Agent/
            
            # 首先尝试从 task_description.json 读取数据集路径
            task_desc_path = project_root / "task" / "task_description.json"
            dataset_path_from_json = None
            if task_desc_path.exists():
                try:
                    with open(task_desc_path, 'r', encoding='utf-8') as f:
                        task_desc = json.load(f)
                    dataset_path_from_json = task_desc.get("task_dataset", {}).get("file_path", "")
                    if dataset_path_from_json:
                        # 处理绝对路径或相对路径
                        if Path(dataset_path_from_json).is_absolute():
                            dataset_full_path = Path(dataset_path_from_json)
                        else:
                            dataset_full_path = project_root / dataset_path_from_json
                        
                        # 如果指定的文件路径不存在，但数据集路径存在，自动使用数据集路径
                        if (not path.exists() or not path.is_file()) and dataset_full_path.exists() and dataset_full_path.is_file():
                            print(f"  ℹ️ 文件 '{file_path}' 不存在，自动使用任务描述文件中的数据集路径: {dataset_path_from_json}", flush=True)
                            path = dataset_full_path
                except Exception as e:
                    # 如果读取 task_description.json 失败，继续原有逻辑
                    pass
            
            # 如果文件仍然不存在，尝试智能查找
            if not path.exists() or not path.is_file():
                # 如果路径是相对路径（不包含路径分隔符），尝试在项目根目录查找
                if not any(sep in file_path for sep in ['/', '\\']):
                    # 尝试直接在项目根目录
                    candidate = project_root / file_path
                    if candidate.exists() and candidate.is_file():
                        path = candidate
                    else:
                        # 尝试在 task/data/ 目录下递归查找
                        task_data_dir = project_root / "task" / "data"
                        if task_data_dir.exists():
                            found_files = list(task_data_dir.rglob(file_path))
                            if found_files:
                                path = found_files[0]  # 使用第一个找到的文件
                            else:
                                error_msg = f"文件不存在: {file_path}。已尝试在项目根目录和 task/data/ 目录下查找，未找到。"
                                if dataset_path_from_json:
                                    error_msg += f" 提示：任务描述文件中指定的数据集路径为: {dataset_path_from_json}"
                                else:
                                    error_msg += " 提示：请先使用 read_file 读取 task/task_description.json 获取正确的数据集文件路径。"
                                
                                return self.format_error(
                                    error=error_msg,
                                    metadata={"file_path": file_path, "searched_locations": [str(project_root), str(task_data_dir)]}
                                )
                else:
                    # 如果是相对路径但包含分隔符，尝试从项目根目录解析
                    if not path.is_absolute():
                        candidate = project_root / file_path
                        if candidate.exists() and candidate.is_file():
                            path = candidate
                        else:
                            return self.format_error(
                                error=f"文件不存在: {file_path}",
                                metadata={"file_path": file_path, "tried_path": str(candidate)}
                            )
                    else:
                        return self.format_error(
                            error=f"文件不存在: {file_path}",
                            metadata={"file_path": file_path}
                        )
            
            # 再次检查是否为文件
            if not path.is_file():
                return self.format_error(
                    error=f"路径不是文件: {file_path}",
                    metadata={"file_path": file_path, "resolved_path": str(path)}
                )
            
            # 读取文件（只读取前几行和统计信息，避免过大）
            MAX_PREVIEW_LINES = 10  # 最多显示前10行
            
            with open(path, 'r', encoding=encoding) as f:
                # 先读取所有行以获取统计信息
                all_lines = f.readlines()
                total_lines = len(all_lines)
                total_size = sum(len(line.encode(encoding)) for line in all_lines)
                
                # 只保留前几行作为预览
                preview_lines = all_lines[:MAX_PREVIEW_LINES]
                preview_content = ''.join(preview_lines)
            
            # 如果是JSON文件，尝试解析
            if path.suffix.lower() == '.json':
                try:
                    # 对于JSON文件，尝试解析完整内容以获取结构信息
                    full_content = ''.join(all_lines)
                    data = json.loads(full_content)
                    
                    # 如果是字典，只返回键和部分值预览
                    if isinstance(data, dict):
                        # 只返回前几个键值对作为预览
                        preview_data = dict(list(data.items())[:5])
                        if len(data) > 5:
                            preview_data["_note"] = f"... (and {len(data) - 5} more keys, total: {len(data)} keys)"
                    elif isinstance(data, list):
                        # 只返回前几个元素作为预览
                        preview_data = data[:5]
                        if len(data) > 5:
                            preview_data.append(f"... (and {len(data) - 5} more items, total: {len(data)} items)")
                    else:
                        preview_data = data
                    
                    return self.format_success(
                        data={
                            "file_path": str(path),
                            "file_type": "json",
                            "content": preview_data,
                            "preview_only": True,
                            "total_keys" if isinstance(data, dict) else "total_items": len(data) if isinstance(data, (dict, list)) else 1,
                            "size_bytes": total_size,
                            "line_count": total_lines
                        },
                        metadata={"encoding": encoding}
                    )
                except json.JSONDecodeError as e:
                    return self.format_error(
                        error=f"JSON解析失败: {str(e)}",
                        metadata={"file_path": file_path}
                    )
            
            # 判断文件类型
            file_type = "csv" if path.suffix.lower() == '.csv' else "text"
            
            # 构建预览内容说明
            preview_note = ""
            if total_lines > MAX_PREVIEW_LINES:
                preview_note = f"\n\n[Note: Showing first {MAX_PREVIEW_LINES} lines only. Total lines: {total_lines}]"
            
            return self.format_success(
                data={
                    "file_path": str(path),
                    "file_type": file_type,
                    "content": preview_content + preview_note,  # 只返回前几行预览
                    "preview_only": True,
                    "total_lines": total_lines,
                    "size_bytes": total_size,
                    "line_count": total_lines
                    },
                    metadata={"encoding": encoding}
                )
        
        except Exception as e:
            return self.format_error(
                error=f"读取文件失败: {str(e)}",
                metadata={"file_path": file_path}
            )


class FileWriteInput(BioAgentToolInput):
    """文件写入工具输入"""
    file_path: str = Field(description="要写入的文件路径（相对或绝对路径）")
    content: str = Field(description="要写入的内容（文本或JSON字符串）")
    encoding: str = Field(default="utf-8", description="文件编码")
    create_dirs: bool = Field(default=True, description="如果目录不存在，是否创建")


class FileWriteTool(BioAgentTool):
    """文件写入工具 - 写入文本文件或JSON文件"""
    
    name: str = "write_file"
    description: str = """
    写入文件内容。支持文本文件和JSON文件。
    如果content是JSON字符串，会自动格式化保存。
    如果目录不存在且create_dirs=True，会自动创建目录。
    """
    args_schema: type[BaseModel] = FileWriteInput
    
    async def _arun(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> ToolResult:
        """写入文件"""
        try:
            path = Path(file_path)
            
            # 创建目录（如果需要）
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果是JSON文件，尝试格式化
            if path.suffix.lower() == '.json':
                try:
                    # 尝试解析JSON以验证格式
                    data = json.loads(content)
                    # 格式化写入
                    with open(path, 'w', encoding=encoding) as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except json.JSONDecodeError as e:
                    return self.format_error(
                        error=f"JSON格式无效: {str(e)}",
                        metadata={"file_path": file_path}
                    )
            else:
                # 普通文本文件
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
            
            # 获取文件信息
            file_size = path.stat().st_size
            
            return self.format_success(
                data={
                    "file_path": str(path),
                    "file_size_bytes": file_size,
                    "message": "文件写入成功"
                },
                metadata={"encoding": encoding, "create_dirs": create_dirs}
            )
        
        except Exception as e:
            return self.format_error(
                error=f"写入文件失败: {str(e)}",
                metadata={"file_path": file_path}
            )


# ==================== 工具注册表 ====================

class ToolRegistry:
    """工具注册表 - 管理所有可用工具"""
    
    def __init__(self):
        self.tools: Dict[str, BioAgentTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        # RAG检索工具
        try:
            self.register_tool("rag_search", RAGSearchTool())
        except Exception as e:
            print(f"⚠️ 注册RAG工具失败: {e}")
        
        # 文件读取工具
        try:
            self.register_tool("read_file", FileReadTool())
        except Exception as e:
            print(f"⚠️ 注册文件读取工具失败: {e}")
        
        # 文件写入工具
        try:
            self.register_tool("write_file", FileWriteTool())
        except Exception as e:
            print(f"⚠️ 注册文件写入工具失败: {e}")
    
    def register_tool(self, name: str, tool: BioAgentTool):
        """注册工具"""
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> Optional[BioAgentTool]:
        """获取工具"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[BioAgentTool]:
        """获取所有工具列表（用于LangChain）"""
        return list(self.tools.values())


# 全局工具注册表实例
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取工具注册表单例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


# ==================== 导出 ====================

__all__ = [
    "AgentRole",
    "REAgentState",
    "CritiqueResult",
    "OptimizationReport",
    "BioAgentTool",
    "BioAgentToolInput",
    "ToolResult",
    "RAGSearchTool",
    "RAGSearchInput",
    "FileReadTool",
    "FileReadInput",
    "FileWriteTool",
    "FileWriteInput",
    "ToolRegistry",
    "get_tool_registry"
]
