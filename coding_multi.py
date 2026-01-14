"""
coding_multi.py
分段多轮代码生成 - 严格按照实验报告生成生产级代码
"""

import asyncio
import json
import re
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

from Agents.prompt import get_agent_by_role
from config.settings import get_settings


async def generate_code_from_report(report_path: str | None = None, output_dir: str = "code_generated_multi"):
    """
    根据实验方案报告分段生成完整代码
    
    Args:
        report_path: 实验方案报告JSON文件路径（默认使用 outputs/final_report.json）
        output_dir: 输出目录
    """
    print("="*80)
    print("🚀 Multi-Stage Code Generator - 分段代码生成")
    print("="*80)
    
    # 1. 读取报告文件（默认使用 outputs/final_report.json）
    if report_path is None:
        report_path = "outputs/final_report.json"
    
    print(f"\n📖 读取实验方案报告: {report_path}")
    report_path_obj = Path(report_path)
    if not report_path_obj.exists():
        raise FileNotFoundError(f"报告文件不存在: {report_path}")
    
    with open(report_path_obj, "r", encoding="utf-8") as f:
        full_report = json.load(f)
    
    # 只提取 expert_analyses 之前的内容
    report = {}
    for key in ["title", "summary", "priority_recommendations", "task_information", "experimental_design"]:
        if key in full_report:
            report[key] = full_report[key]
    
    print("✓ 报告文件读取成功（仅使用 expert_analyses 之前的内容）")
    
    # 2. 提取任务信息和实验方案
    task_info = report.get("task_information", {})
    exp_design = report.get("experimental_design", {})
    priority_recs = report.get("priority_recommendations", [])
    
    task_description = task_info.get("description", "")
    background = task_info.get("background", "")
    dataset_info = task_info.get("dataset_info", "")
    
    # 3. 初始化 Code Generator Agent
    print("\n🔧 初始化 Code Generator Agent...")
    code_agent = get_agent_by_role("code_generator")
    print("✓ Code Generator Agent 初始化完成")
    
    # 4. 准备输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 存储已生成的代码片段
    generated_code_parts = {}
    
    # ==================== 第一步：项目结构与配置 ====================
    print("\n" + "="*80)
    print("📦 Stage 1: Project Structure & Configuration")
    print("="*80)
    
    stage1_prompt = f"""You are generating PRODUCTION-GRADE code for a bioinformatics deep learning project. This is NOT a tutorial or demonstration - this is production code that will be used in real research.

⚠️ CRITICAL REQUIREMENTS:
- Generate PRODUCTION-GRADE code with ROBUST ERROR HANDLING
- STRICTLY FOLLOW the report specifications - DO NOT SIMPLIFY
- DO NOT SIMPLIFY any implementation details
- Include comprehensive error handling, logging, and validation
- All code must be production-ready and maintainable
- **MAKE DECISIVE CHOICES**: If the report provides multiple options or vague suggestions, you MUST make a clear decision and choose what you believe is the BEST solution. DO NOT leave choices ambiguous or use placeholder values. DO NOT be lazy - make specific, well-reasoned choices based on the context and best practices.

## Task Description
**Title**: {task_description}

**Background**: {background}

**Dataset Information**: {dataset_info}

## Experimental Design Report

### Data Usage Plan
{json.dumps(exp_design.get("1_data_usage_plan", {}), ensure_ascii=False, indent=2)}

### Method Design
{json.dumps(exp_design.get("2_method_design", {}), ensure_ascii=False, indent=2)}

### Model Design
{json.dumps(exp_design.get("3_model_design", {}), ensure_ascii=False, indent=2)}

### Result Summary
{json.dumps(exp_design.get("4_result_summary", {}), ensure_ascii=False, indent=2)}

## Priority Recommendations
{chr(10).join(f"- {rec}" for rec in priority_recs)}

## Stage 1 Task: Project Structure & Configuration

Generate ONLY the following files for Stage 1:

1. **config.py**: Complete configuration file with ALL hyperparameters from the experimental design
   - STRICTLY FOLLOW all parameter values specified in the report
   - Include ALL hyperparameters: learning rates, batch sizes, optimizer settings, regularization parameters, etc.
   - Add ROBUST ERROR HANDLING for configuration validation
   - Include type hints and comprehensive documentation
   - DO NOT SIMPLIFY - include all configuration details from the report

2. **requirements.txt**: Complete dependency list with version specifications
   - Include all necessary packages: PyTorch, NumPy, Pandas, scikit-learn, etc.
   - Specify version constraints based on compatibility requirements

3. **PROJECT_STRUCTURE.md**: Detailed project structure documentation
   - Explain the overall architecture
   - Document the purpose of each module
   - Include usage guidelines

4. **README.md**: Comprehensive project documentation
   - Project overview and purpose
   - Installation instructions
   - Usage guidelines
   - Configuration explanation

⚠️ REMEMBER:
- This is PRODUCTION-GRADE code, NOT a tutorial
- STRICTLY FOLLOW the report specifications - DO NOT SIMPLIFY
- Include ROBUST ERROR HANDLING throughout
- All code must be production-ready

Output format: Provide a STRICT, VALID JSON object with the following structure:

⚠️ CRITICAL JSON FORMAT REQUIREMENTS:
- You MUST return valid, parseable JSON that strictly conforms to JSON specification
- ALL string values (especially in the "code" field) MUST have control characters properly escaped:
  * Newlines: use \\n (not actual newline characters)
  * Tabs: use \\t (not actual tab characters)
  * Carriage returns: use \\r (not actual \\r characters)
  * Other control characters: use \\uXXXX Unicode escape sequences
- DO NOT include unescaped control characters in string values - they will cause JSON parsing to fail
- The "code" field contains Python code as a STRING - escape all special characters properly
- Use double quotes for all strings (not single quotes)
- Ensure all brackets, braces, and quotes are properly matched and escaped

Example of properly escaped code string:
"code": "def hello():\\n    print(\\\"Hello\\\")\\n    return True"

```json
{{
    "files": [
        {{
            "path": "config.py",
            "code": "...complete production code with ALL control characters properly escaped..."
        }},
        {{
            "path": "requirements.txt",
            "code": "..."
        }},
        {{
            "path": "PROJECT_STRUCTURE.md",
            "code": "..."
        }},
        {{
            "path": "README.md",
            "code": "..."
        }}
    ],
    "stage": 1,
    "description": "Brief description of what was generated"
}}
```

IMPORTANT: The JSON you return MUST be parseable by standard JSON parsers. Test your output mentally - if the "code" field contains Python code with newlines, they MUST be escaped as \\n, not actual newline characters.
"""
    
    stage1_code = await _generate_stage_code(code_agent, stage1_prompt, "Stage 1")
    generated_code_parts[1] = stage1_code
    
    # 保存 Stage 1 代码
    await _save_stage_files(output_path, stage1_code, 1)
    
    # ==================== 第二步：数据管道 ====================
    print("\n" + "="*80)
    print("📊 Stage 2: Data Pipeline")
    print("="*80)
    
    stage2_prompt = f"""You are generating PRODUCTION-GRADE code for a bioinformatics deep learning project. This is NOT a tutorial - this is production code.

⚠️ CRITICAL REQUIREMENTS:
- Generate PRODUCTION-GRADE code with ROBUST ERROR HANDLING
- STRICTLY FOLLOW the report specifications - DO NOT SIMPLIFY
- DO NOT SIMPLIFY any data preprocessing steps
- Include comprehensive error handling, data validation, and logging
- **MAKE DECISIVE CHOICES**: If the report provides multiple options or vague suggestions, you MUST make a clear decision and choose what you believe is the BEST solution. DO NOT leave choices ambiguous or use placeholder values. DO NOT be lazy - make specific, well-reasoned choices based on the context and best practices.

## Task Description
**Title**: {task_description}

**Background**: {background}

**Dataset Information**: {dataset_info}

## Experimental Design Report

### Data Usage Plan (CRITICAL - STRICTLY FOLLOW THIS)
{json.dumps(exp_design.get("1_data_usage_plan", {}), ensure_ascii=False, indent=2)}

### Method Design (for data augmentation details)
{json.dumps(exp_design.get("2_method_design", {}), ensure_ascii=False, indent=2)}

## Priority Recommendations
{chr(10).join(f"- {rec}" for rec in priority_recs)}

## Stage 2 Task: Data Pipeline

Generate ONLY the data pipeline code for Stage 2:

1. **dataset.py**: Complete data loading and preprocessing module
   - STRICTLY FOLLOW all data preprocessing specifications from the Data Usage Plan
   - Implement ALL preprocessing steps exactly as specified in the report
   - Include ROBUST ERROR HANDLING for data validation
   - Implement data loading, preprocessing, augmentation, and splitting
   - DO NOT SIMPLIFY - implement all data transformations exactly as specified
   - Include comprehensive logging and error handling
   - Add data quality checks and validation

2. **utils.py** (data-related utilities): Utility functions for data processing
   - One-hot encoding functions (if specified)
   - Data normalization functions
   - Data augmentation functions (if specified)
   - Data validation utilities
   - All utility functions with ROBUST ERROR HANDLING

⚠️ REMEMBER:
- This is PRODUCTION-GRADE code - STRICTLY FOLLOW the report specifications
- DO NOT SIMPLIFY any preprocessing steps
- Include ROBUST ERROR HANDLING for all data operations
- All data transformations must match the report exactly

Output format: Provide a STRICT, VALID JSON object:

⚠️ CRITICAL JSON FORMAT REQUIREMENTS:
- You MUST return valid, parseable JSON that strictly conforms to JSON specification
- ALL string values (especially in the "code" field) MUST have control characters properly escaped:
  * Newlines: use \\n (not actual newline characters)
  * Tabs: use \\t (not actual tab characters)
  * Carriage returns: use \\r (not actual \\r characters)
  * Other control characters: use \\uXXXX Unicode escape sequences
- DO NOT include unescaped control characters in string values - they will cause JSON parsing to fail
- The "code" field contains Python code as a STRING - escape all special characters properly
- Use double quotes for all strings (not single quotes)
- Ensure all brackets, braces, and quotes are properly matched and escaped

```json
{{
    "files": [
        {{
            "path": "dataset.py",
            "code": "...complete production code with ALL control characters properly escaped..."
        }},
        {{
            "path": "utils.py",
            "code": "...complete production code with ALL control characters properly escaped..."
        }}
    ],
    "stage": 2,
    "description": "Brief description"
}}
```

IMPORTANT: The JSON you return MUST be parseable by standard JSON parsers. All control characters in code strings MUST be escaped.
"""
    
    stage2_code = await _generate_stage_code(code_agent, stage2_prompt, "Stage 2")
    generated_code_parts[2] = stage2_code
    
    # 保存 Stage 2 代码
    await _save_stage_files(output_path, stage2_code, 2)
    
    # ==================== 第三步：模型架构 ====================
    print("\n" + "="*80)
    print("🏗️ Stage 3: Model Architecture")
    print("="*80)
    
    stage3_prompt = f"""You are generating PRODUCTION-GRADE code for a bioinformatics deep learning project. This is NOT a tutorial - this is production code.

⚠️ CRITICAL REQUIREMENTS:
- Generate PRODUCTION-GRADE code with ROBUST ERROR HANDLING
- STRICTLY FOLLOW the report specifications - DO NOT SIMPLIFY
- DO NOT SIMPLIFY any model architecture details
- Implement EXACT layer dimensions, kernel sizes, activation functions as specified
- **MAKE DECISIVE CHOICES**: If the report provides multiple options or vague suggestions, you MUST make a clear decision and choose what you believe is the BEST solution. DO NOT leave choices ambiguous or use placeholder values. DO NOT be lazy - make specific, well-reasoned choices based on the context and best practices.

## Task Description
**Title**: {task_description}

**Background**: {background}

**Dataset Information**: {dataset_info}

## Experimental Design Report

### Model Design (CRITICAL - STRICTLY FOLLOW THIS)
{json.dumps(exp_design.get("3_model_design", {}), ensure_ascii=False, indent=2)}

### Method Design (for regularization and optimization details)
{json.dumps(exp_design.get("2_method_design", {}), ensure_ascii=False, indent=2)}

## Priority Recommendations
{chr(10).join(f"- {rec}" for rec in priority_recs)}

## Stage 3 Task: Model Architecture

Generate ONLY the model architecture code for Stage 3:

1. **model.py**: Complete model architecture implementation
   - STRICTLY FOLLOW all architecture specifications from the Model Design section
   - Implement EXACT layer dimensions, kernel sizes, strides, padding as specified
   - Use EXACT activation functions, dropout rates, normalization layers as specified
   - Include ALL architectural components: attention mechanisms, residual connections, etc.
   - DO NOT SIMPLIFY - implement the architecture exactly as specified in the report
   - Include ROBUST ERROR HANDLING for model initialization and forward pass
   - Add comprehensive model validation and parameter checking
   - Include proper weight initialization as specified

⚠️ REMEMBER:
- This is PRODUCTION-GRADE code - STRICTLY FOLLOW the report specifications
- DO NOT SIMPLIFY any architecture details
- All layer dimensions and hyperparameters must match the report EXACTLY
- Include ROBUST ERROR HANDLING throughout

Output format: Provide a STRICT, VALID JSON object:

⚠️ CRITICAL JSON FORMAT REQUIREMENTS:
- You MUST return valid, parseable JSON that strictly conforms to JSON specification
- ALL string values (especially in the "code" field) MUST have control characters properly escaped:
  * Newlines: use \\n (not actual newline characters)
  * Tabs: use \\t (not actual tab characters)
  * Carriage returns: use \\r (not actual \\r characters)
  * Other control characters: use \\uXXXX Unicode escape sequences
- DO NOT include unescaped control characters in string values - they will cause JSON parsing to fail
- The "code" field contains Python code as a STRING - escape all special characters properly
- Use double quotes for all strings (not single quotes)
- Ensure all brackets, braces, and quotes are properly matched and escaped

```json
{{
    "files": [
        {{
            "path": "model.py",
            "code": "...complete production code with ALL control characters properly escaped..."
        }}
    ],
    "stage": 3,
    "description": "Brief description"
}}
```

IMPORTANT: The JSON you return MUST be parseable by standard JSON parsers. All control characters in code strings MUST be escaped.
"""
    
    stage3_code = await _generate_stage_code(code_agent, stage3_prompt, "Stage 3")
    generated_code_parts[3] = stage3_code
    
    # 保存 Stage 3 代码
    await _save_stage_files(output_path, stage3_code, 3)
    
    # ==================== 第四步：训练循环 ====================
    print("\n" + "="*80)
    print("🔄 Stage 4: Training Loop")
    print("="*80)
    
    stage4_prompt = f"""You are generating PRODUCTION-GRADE code for a bioinformatics deep learning project. This is NOT a tutorial - this is production code.

⚠️ CRITICAL REQUIREMENTS:
- Generate PRODUCTION-GRADE code with ROBUST ERROR HANDLING
- STRICTLY FOLLOW the report specifications - DO NOT SIMPLIFY
- DO NOT SIMPLIFY any training logic
- Implement EXACT loss functions, optimizers, schedulers as specified
- **MAKE DECISIVE CHOICES**: If the report provides multiple options or vague suggestions, you MUST make a clear decision and choose what you believe is the BEST solution. DO NOT leave choices ambiguous or use placeholder values. DO NOT be lazy - make specific, well-reasoned choices based on the context and best practices.

## Task Description
**Title**: {task_description}

**Background**: {background}

**Dataset Information**: {dataset_info}

## Experimental Design Report

### Method Design (CRITICAL - STRICTLY FOLLOW THIS)
{json.dumps(exp_design.get("2_method_design", {}), ensure_ascii=False, indent=2)}

### Model Design (for model-specific training details)
{json.dumps(exp_design.get("3_model_design", {}), ensure_ascii=False, indent=2)}

### Result Summary (for evaluation metrics)
{json.dumps(exp_design.get("4_result_summary", {}), ensure_ascii=False, indent=2)}

## Priority Recommendations
{chr(10).join(f"- {rec}" for rec in priority_recs)}

## Stage 4 Task: Training Loop

Generate ONLY the training code for Stage 4:

1. **train.py**: Complete training script
   - STRICTLY FOLLOW all training specifications from the Method Design section
   - Implement EXACT loss functions, optimizers, learning rate schedules as specified
   - Include ALL training features: early stopping, checkpointing, logging, etc.
   - DO NOT SIMPLIFY - implement all training logic exactly as specified
   - Include ROBUST ERROR HANDLING for training loop, data loading, model saving
   - Add comprehensive logging and progress tracking
   - Implement proper checkpoint management and resume functionality
   - Include gradient clipping, mixed precision training if specified

⚠️ REMEMBER:
- This is PRODUCTION-GRADE code - STRICTLY FOLLOW the report specifications
- DO NOT SIMPLIFY any training logic
- All hyperparameters and training procedures must match the report EXACTLY
- Include ROBUST ERROR HANDLING throughout

Output format: Provide a STRICT, VALID JSON object:

⚠️ CRITICAL JSON FORMAT REQUIREMENTS:
- You MUST return valid, parseable JSON that strictly conforms to JSON specification
- ALL string values (especially in the "code" field) MUST have control characters properly escaped:
  * Newlines: use \\n (not actual newline characters)
  * Tabs: use \\t (not actual tab characters)
  * Carriage returns: use \\r (not actual \\r characters)
  * Other control characters: use \\uXXXX Unicode escape sequences
- DO NOT include unescaped control characters in string values - they will cause JSON parsing to fail
- The "code" field contains Python code as a STRING - escape all special characters properly
- Use double quotes for all strings (not single quotes)
- Ensure all brackets, braces, and quotes are properly matched and escaped

```json
{{
    "files": [
        {{
            "path": "train.py",
            "code": "...complete production code with ALL control characters properly escaped..."
        }}
    ],
    "stage": 4,
    "description": "Brief description"
}}
```

IMPORTANT: The JSON you return MUST be parseable by standard JSON parsers. All control characters in code strings MUST be escaped.
"""
    
    stage4_code = await _generate_stage_code(code_agent, stage4_prompt, "Stage 4")
    generated_code_parts[4] = stage4_code
    
    # 保存 Stage 4 代码
    await _save_stage_files(output_path, stage4_code, 4)
    
    # ==================== 第五步：补充代码并检查接口 ====================
    print("\n" + "="*80)
    print("🔍 Stage 5: Additional Code & Interface Validation")
    print("="*80)
    
    # 读取已生成的所有代码文件（读取更多内容以便全面审核，但限制总长度）
    existing_files = {}
    MAX_FILE_PREVIEW = 3000  # 每个文件最多预览 3000 字符，平衡审核需求和 token 限制
    TOTAL_PREVIEW_LIMIT = 15000  # 所有文件预览总长度限制
    total_preview_chars = 0
    
    for stage_num in [1, 2, 3, 4]:
        stage_dir = output_path / f"stage_{stage_num}"
        if stage_dir.exists():
            for file_path in sorted(stage_dir.glob("*.py")):  # 排序以保证一致性
                if total_preview_chars >= TOTAL_PREVIEW_LIMIT:
                    print(f"   ⚠️ 达到总预览长度限制 ({TOTAL_PREVIEW_LIMIT} 字符)，跳过剩余文件")
                    break
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        full_content = f.read()
                        # 计算可用的预览长度
                        remaining_limit = TOTAL_PREVIEW_LIMIT - total_preview_chars
                        preview_length = min(MAX_FILE_PREVIEW, len(full_content), remaining_limit)
                        
                        if len(full_content) > preview_length:
                            # 优先显示文件开头（包含导入和主要函数定义）
                            preview_content = full_content[:preview_length]
                            existing_files[file_path.name] = preview_content + f"\n\n[... File truncated, total length: {len(full_content)} characters. Focus on function signatures, imports, class definitions, and key logic from the beginning ...]"
                        else:
                            existing_files[file_path.name] = full_content
                        
                        total_preview_chars += len(existing_files[file_path.name])
                except Exception as e:
                    print(f"   ⚠️ Failed to read {file_path.name}: {e}")
    
    stage5_prompt = f"""You are generating PRODUCTION-GRADE code for a bioinformatics deep learning project. This is NOT a tutorial - this is production code.

⚠️ CRITICAL REQUIREMENTS:
- Generate PRODUCTION-GRADE code with ROBUST ERROR HANDLING
- STRICTLY FOLLOW the report specifications - DO NOT SIMPLIFY
- DO NOT SIMPLIFY any implementation
- Check and fix ALL interface compatibility issues
- **MAKE DECISIVE CHOICES**: If the report provides multiple options or vague suggestions, you MUST make a clear decision and choose what you believe is the BEST solution. DO NOT leave choices ambiguous or use placeholder values. DO NOT be lazy - make specific, well-reasoned choices based on the context and best practices.
- **COMPREHENSIVE FILE REVIEW**: You MUST thoroughly read, understand, and analyze ALL existing code files before generating new code or making fixes. Avoid ALL conflicts, inconsistencies, and interface mismatches.

## Task Description
**Title**: {task_description}

**Background**: {background}

**Dataset Information**: {dataset_info}

## Experimental Design Report

### Result Summary (CRITICAL - STRICTLY FOLLOW THIS)
{json.dumps(exp_design.get("4_result_summary", {}), ensure_ascii=False, indent=2)}

### Method Design (for evaluation details)
{json.dumps(exp_design.get("2_method_design", {}), ensure_ascii=False, indent=2)}

## Priority Recommendations
{chr(10).join(f"- {rec}" for rec in priority_recs)}

## Existing Generated Code

⚠️ **CRITICAL: You MUST thoroughly read, understand, and analyze ALL of the following files before proceeding.**

The following files have already been generated in previous stages. You MUST:
1. Read and understand the COMPLETE structure of each file
2. Identify ALL function signatures, imports, and dependencies
3. Map out the data flow between files
4. Identify ALL potential conflicts and inconsistencies
5. Ensure your new code and fixes are fully compatible with ALL existing code

{json.dumps({k: v for k, v in existing_files.items()}, ensure_ascii=False, indent=2)}

## Stage 5 Task: Additional Code & Interface Validation

**CRITICAL: You MUST thoroughly read and review ALL existing code files before generating new code or making fixes.**

Generate the remaining code and fix ALL interface issues:

1. **Comprehensive File Review (MANDATORY FIRST STEP)**:
   - **READ and UNDERSTAND** ALL existing code files completely (dataset.py, model.py, train.py, config.py, utils.py)
   - **ANALYZE** each file's structure, function signatures, imports, and dependencies
   - **IDENTIFY** all potential conflicts, inconsistencies, and interface mismatches
   - **MAP OUT** the data flow between files (dataset → model → train → evaluate)
   - **VERIFY** all imports, function calls, and variable names are consistent across files
   - **CHECK** for naming conflicts, duplicate definitions, or missing dependencies
   - DO NOT proceed with code generation until you have fully understood ALL existing code

2. **evaluate.py**: Complete evaluation module
   - STRICTLY FOLLOW all evaluation specifications from the Result Summary section
   - Implement ALL evaluation metrics and statistical tests as specified
   - Include ROBUST ERROR HANDLING for evaluation
   - **ENSURE COMPLETE COMPATIBILITY**: The evaluate.py must work seamlessly with ALL existing files
   - Verify that evaluation inputs match train.py outputs exactly
   - Verify that evaluation can load models and datasets correctly
   - DO NOT SIMPLIFY - implement all evaluation logic exactly as specified

3. **Interface Fixes**: Review and fix ALL interface compatibility issues
   - **THOROUGHLY CHECK** that dataset.py output matches model.py input requirements (data shapes, types, formats)
   - **THOROUGHLY CHECK** that model.py output matches train.py requirements (forward pass output format, device handling)
   - **THOROUGHLY CHECK** that train.py output matches evaluate.py requirements (model checkpoints, predictions format)
   - **RESOLVE ALL CONFLICTS**: Fix ANY interface mismatches, naming conflicts, or inconsistencies in the existing code
   - **VERIFY CONSISTENCY**: Ensure all function signatures are compatible across files
   - **VALIDATE IMPORTS**: Check that all imports are correct and all dependencies are satisfied
   - **CHECK CONFIG USAGE**: Verify that config.py parameters are used consistently across all files
   - Add proper type hints and validation
   - Update any files that have conflicts or inconsistencies

4. **Additional Utilities**: Any remaining utility functions
   - Visualization functions
   - Additional helper functions
   - All with ROBUST ERROR HANDLING
   - **ENSURE NO CONFLICTS**: Verify that utility functions don't conflict with existing code

⚠️ REMEMBER:
- This is PRODUCTION-GRADE code - STRICTLY FOLLOW the report specifications
- DO NOT SIMPLIFY any implementation
- Check and fix ALL interface compatibility issues
- Include ROBUST ERROR HANDLING throughout

Output format: Provide a STRICT, VALID JSON object with updated files:

⚠️ CRITICAL JSON FORMAT REQUIREMENTS:
- You MUST return valid, parseable JSON that strictly conforms to JSON specification
- ALL string values (especially in the "code" field) MUST have control characters properly escaped:
  * Newlines: use \\n (not actual newline characters)
  * Tabs: use \\t (not actual tab characters)
  * Carriage returns: use \\r (not actual \\r characters)
  * Other control characters: use \\uXXXX Unicode escape sequences
- DO NOT include unescaped control characters in string values - they will cause JSON parsing to fail
- The "code" field contains Python code as a STRING - escape all special characters properly
- Use double quotes for all strings (not single quotes)
- Ensure all brackets, braces, and quotes are properly matched and escaped

```json
{{
    "files": [
        {{
            "path": "evaluate.py",
            "code": "...complete production code with ALL control characters properly escaped..."
        }},
        {{
            "path": "dataset.py",
            "code": "...updated code with interface fixes, ALL control characters properly escaped..."
        }},
        {{
            "path": "model.py",
            "code": "...updated code with interface fixes, ALL control characters properly escaped..."
        }},
        {{
            "path": "train.py",
            "code": "...updated code with interface fixes, ALL control characters properly escaped..."
        }}
    ],
    "stage": 5,
    "description": "Brief description of fixes and additions",
    "interface_fixes": ["List of interface issues fixed"]
}}
```

IMPORTANT: The JSON you return MUST be parseable by standard JSON parsers. All control characters in code strings MUST be escaped.
"""
    
    stage5_code = await _generate_stage_code(code_agent, stage5_prompt, "Stage 5")
    generated_code_parts[5] = stage5_code
    
    # 保存 Stage 5 代码（包括更新的文件）
    await _save_stage_files(output_path, stage5_code, 5)
    
    # ==================== 合并所有代码到最终目录 ====================
    print("\n" + "="*80)
    print("📦 Merging All Stages")
    print("="*80)
    
    await _merge_all_stages(output_path, generated_code_parts)
    
    # ==================== 删除前五轮的分别文件夹 ====================
    print("\n" + "="*80)
    print("🗑️ Cleaning Up Stage Directories")
    print("="*80)
    
    for stage_num in [1, 2, 3, 4, 5]:
        stage_dir = output_path / f"stage_{stage_num}"
        if stage_dir.exists():
            import shutil
            try:
                shutil.rmtree(stage_dir)
                print(f"   ✓ Deleted: stage_{stage_num}/")
            except Exception as e:
                print(f"   ⚠️ Failed to delete stage_{stage_num}/: {e}")
    
    # ==================== 完成 ====================
    print("\n" + "="*80)
    print("✅ Multi-Stage Code Generation Complete!")
    print("="*80)
    print(f"📁 Output directory: {output_path.resolve()}")
    
    # 列出最终生成的文件
    final_files = list((output_path / "final").glob("*")) if (output_path / "final").exists() else []
    if final_files:
        print(f"\n📄 Final generated files ({len(final_files)}):")
        for f in sorted(final_files):
            if f.is_file():
                size = f.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                print(f"   ✓ {f.name:<40} ({size_str})")


async def _generate_stage_code(code_agent, prompt: str, stage_name: str) -> dict:
    """生成单个阶段的代码"""
    print(f"\n💻 Generating {stage_name} code...")
    print("   (This may take several minutes, please wait...)")
    
    try:
        settings = get_settings()
        code_model = settings.code_model
        
        # 使用标准的 chat API
        # 注意：代码生成任务应该直接返回代码内容，不应该使用工具调用
        # 使用 llm 而不是 llm_with_tools 来避免工具调用
        messages = [
            SystemMessage(content=code_agent.prompt()),
            HumanMessage(content=prompt)
        ]
        
        try:
            # 代码生成直接使用 llm，不使用工具绑定
            # 代码生成任务应该直接返回代码内容，不应该有工具调用
            response = await code_agent.llm.ainvoke(messages)
            code_content = response.content if hasattr(response, 'content') and response.content else ""
            
            # 检查响应元数据，判断是否因为长度限制被截断
            finish_reason = None
            reasoning_tokens = 0
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                finish_reason = metadata.get('finish_reason')
                token_usage = metadata.get('tokeen_usage', {}) or metadata.get('token_usage', {})
                if isinstance(token_usage, dict):
                    completion_details = token_usage.get('completion_tokens_details', {})
                    if isinstance(completion_details, dict):
                        reasoning_tokens = completion_details.get('reasoning_tokens', 0)
            
            # 调试信息：检查响应对象结构
            if not code_content or not code_content.strip():
                print(f"   ⚠️ Empty or whitespace-only response detected. Response type: {type(response)}")
                if hasattr(response, 'content'):
                    print(f"   ⚠️ response.content type: {type(response.content)}, length: {len(str(response.content))}")
                if finish_reason == 'length':
                    print(f"   ⚠️ Response was truncated due to length limit (finish_reason: length)")
                    if reasoning_tokens > 0:
                        print(f"   ⚠️ Model used {reasoning_tokens} reasoning tokens, but content was empty")
                        print(f"   💡 Suggestion: The prompt may be too long or max_tokens too small.")
                        print(f"   💡 Consider: Reducing prompt size or increasing max_tokens limit.")
                if hasattr(response, '__dict__'):
                    print(f"   ⚠️ Response dict keys: {list(response.__dict__.keys())}")
                # 尝试获取更多调试信息
                if hasattr(response, 'additional_kwargs'):
                    print(f"   ⚠️ additional_kwargs keys: {list(response.additional_kwargs.keys()) if response.additional_kwargs else 'None'}")
                    # 检查是否有 refusal
                    if response.additional_kwargs and 'refusal' in response.additional_kwargs:
                        refusal_content = response.additional_kwargs.get('refusal', '')
                        print(f"   ⚠️ Model refusal detected: {refusal_content[:500] if refusal_content else 'No refusal message'}")
                        print(f"   💡 Suggestion: The prompt may be too long, contain problematic content, or request something the model refuses to do.")
        except Exception as e:
            if "chat model" in str(e).lower() or "not supported" in str(e).lower():
                print(f"   ⚠️ Model {code_model} doesn't support chat API, falling back...")
                from Agents.prompt import DEFAULT_MODEL
                code_agent.model = DEFAULT_MODEL
                code_agent._llm = None
                code_agent._llm_with_tools = None
                
                response = await code_agent.llm.ainvoke(messages)
                code_content = response.content if response.content else ""
                
                if not code_content:
                    print(f"   ⚠️ Empty response after fallback. Response type: {type(response)}")
                    if hasattr(response, '__dict__'):
                        print(f"   ⚠️ Response dict: {response.__dict__}")
                
                print(f"   ✓ Switched to model: {DEFAULT_MODEL}")
            else:
                raise
        
        # 检查响应是否为空
        if not code_content or not code_content.strip():
            # 获取 finish_reason 和 token 信息用于错误提示
            finish_reason = None
            reasoning_tokens = 0
            total_tokens = 0
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                finish_reason = metadata.get('finish_reason')
                token_usage = metadata.get('tokeen_usage', {}) or metadata.get('token_usage', {})
                if isinstance(token_usage, dict):
                    total_tokens = token_usage.get('total_tokens', 0)
                    completion_details = token_usage.get('completion_tokens_details', {})
                    if isinstance(completion_details, dict):
                        reasoning_tokens = completion_details.get('reasoning_tokens', 0)
            
            error_msg = f"   ❌ Empty response from API. Model: {code_model}"
            
            # 检查是否有 refusal
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                if 'refusal' in response.additional_kwargs:
                    refusal_content = response.additional_kwargs.get('refusal', '')
                    error_msg += f"\n   ⚠️ Model refusal detected: {str(refusal_content)[:200]}"
                    error_msg += f"\n   💡 Suggestion: The prompt may be too long or contain problematic content. Try reducing the prompt size."
            
            if finish_reason == 'length':
                error_msg += f"\n   ⚠️ Response was truncated due to length limit (finish_reason: length)"
                if reasoning_tokens > 0:
                    error_msg += f"\n   ⚠️ Model used {reasoning_tokens} reasoning tokens but content was empty"
                error_msg += f"\n   💡 Suggestion: Reduce prompt size or increase max_tokens limit"
            
            print(error_msg)
            
            # 保存调试信息
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_empty_response.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write(f"Model: {code_model}\n")
                f.write("="*80 + "\n")
                f.write("Empty response received from API.\n")
                f.write(f"Response length: {len(code_content)}\n")
                if finish_reason:
                    f.write(f"Finish reason: {finish_reason}\n")
                if reasoning_tokens > 0:
                    f.write(f"Reasoning tokens: {reasoning_tokens}\n")
                if total_tokens > 0:
                    f.write(f"Total tokens: {total_tokens}\n")
                f.write("\nResponse metadata:\n")
                if hasattr(response, 'response_metadata'):
                    import json
                    f.write(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))
            
            error_detail = f"Empty response from API for {stage_name}"
            if finish_reason == 'length':
                error_detail += " (truncated due to length limit)"
            error_detail += f". Check {debug_file} for details."
            raise ValueError(error_detail)
        
        # 解析 JSON 响应
        print(f"   📝 Response length: {len(code_content)} characters")
        code_data = _parse_json_response(code_content, stage_name)
        
        # 如果解析失败，保存原始响应用于调试
        if not code_data.get("files"):
            debug_file = Path("code_generated") / f"debug_{stage_name.lower().replace(' ', '_')}_response.txt"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Stage: {stage_name}\n")
                f.write("="*80 + "\n")
                f.write(code_content)
            print(f"   ⚠️ JSON parsing failed, saved raw response to: {debug_file}")
            print(f"   📄 First 500 chars of response:\n{code_content[:500]}")
        
        return code_data
        
    except Exception as e:
        print(f"\n❌ {stage_name} code generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def _fix_json_control_chars(json_str: str) -> str:
    """修复 JSON 字符串中的未转义控制字符
    
    在 JSON 字符串值中，控制字符（ASCII 0-31）必须被转义。
    此函数会识别字符串值内的未转义控制字符并正确转义它们。
    """
    result = []
    in_string = False
    escape_next = False
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        
        # 处理转义序列
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        # 处理反斜杠转义
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        # 处理字符串边界
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        # 在字符串内部处理控制字符
        if in_string:
            # 检查是否是控制字符（ASCII 0-31）
            if ord(char) < 32:
                # 常见的控制字符使用标准转义
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                else:
                    # 其他控制字符使用 Unicode 转义
                    result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            # 在字符串外部，直接添加
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def _parse_json_response(content: str, stage_name: str = "") -> dict:
    """解析 LLM 返回的 JSON 响应"""
    import json
    import re
    
    if not content or not content.strip():
        print(f"   ⚠️ Empty response content")
        return {"files": [], "stage": 0, "description": "Empty response"}
    
    # 方法1: 尝试提取 JSON 代码块（支持多行）
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            code_data = json.loads(json_str)
            print(f"   ✓ Successfully parsed JSON from code block")
            return code_data
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON code block parsing failed: {e}")
            # 尝试修复控制字符
            try:
                json_str_fixed = _fix_json_control_chars(json_str)
                code_data = json.loads(json_str_fixed)
                print(f"   ✓ Successfully parsed JSON after fixing control characters")
                return code_data
            except Exception as e2:
                print(f"   ⚠️ Control character fix failed: {e2}")
                pass
    
    # 方法2: 查找包含 "files" 的 JSON 对象（更精确的匹配）
    # 使用平衡括号匹配
    try:
        start_idx = content.find('{')
        if start_idx == -1:
            print(f"   ⚠️ No JSON object found (no opening brace)")
            return {"files": [], "stage": 0, "description": "No JSON object found"}
        
        # 找到匹配的结束括号
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if brace_count == 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx+1]
            try:
                code_data = json.loads(json_str)
                print(f"   ✓ Successfully parsed JSON using balanced brace matching")
                return code_data
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Balanced brace JSON parsing failed: {e}")
                # 尝试修复常见的 JSON 问题
                try:
                    # 先尝试修复控制字符
                    json_str_fixed = _fix_json_control_chars(json_str)
                    code_data = json.loads(json_str_fixed)
                    print(f"   ✓ Successfully parsed JSON after fixing control characters")
                    return code_data
                except Exception as e2:
                    try:
                        # 移除可能的注释
                        json_str_clean = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                        json_str_clean = re.sub(r'/\*.*?\*/', '', json_str_clean, flags=re.DOTALL)
                        json_str_clean = _fix_json_control_chars(json_str_clean)
                        code_data = json.loads(json_str_clean)
                        print(f"   ✓ Successfully parsed JSON after removing comments and fixing control characters")
                        return code_data
                    except:
                        pass
    except Exception as e:
        print(f"   ⚠️ Balanced brace matching failed: {e}")
    
    # 方法3: 尝试使用 json5（如果可用）
    try:
        import json5  # type: ignore
        # 找到第一个 { 到最后一个 }
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx+1]
            code_data = json5.loads(json_str)
            print(f"   ✓ Successfully parsed JSON using json5")
            return code_data
    except (ImportError, Exception) as e:
        pass
    
    # 方法4: 尝试提取所有可能的 JSON 对象
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
    for json_obj in json_objects:
        if '"files"' in json_obj or '"path"' in json_obj:
            try:
                code_data = json.loads(json_obj)
                if "files" in code_data:
                    print(f"   ✓ Successfully parsed JSON from pattern matching")
                    return code_data
            except:
                continue
    
    # 如果都失败，返回空结构并显示调试信息
    print(f"   ❌ All JSON parsing methods failed")
    print(f"   📄 Response preview (first 1000 chars):\n{content[:1000]}")
    return {"files": [], "stage": 0, "description": "Failed to parse response"}


async def _save_stage_files(output_path: Path, code_data: dict, stage_num: int):
    """保存单个阶段的代码文件"""
    stage_dir = output_path / f"stage_{stage_num}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    files = code_data.get("files", [])
    if not files:
        print(f"   ⚠️ No files found in {code_data.get('stage', stage_num)} response")
        return
    
    print(f"   📁 Saving {len(files)} files to stage_{stage_num}/")
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        
        file_path_str = (file_info.get("path") or 
                        file_info.get("file_path") or 
                        file_info.get("filename") or 
                        file_info.get("name") or "")
        
        if not file_path_str:
            continue
        
        file_path = stage_dir / file_path_str
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        code = (file_info.get("code") or 
               file_info.get("content") or 
               file_info.get("source") or 
               file_info.get("source_code") or "")
        
        if code:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"   ✓ Saved: {file_path.name} ({len(code)} chars)")
        else:
            print(f"   ⚠️ Skipped empty file: {file_path_str}")


async def _merge_all_stages(output_path: Path, generated_parts: dict):
    """合并所有阶段的代码到最终目录"""
    final_dir = output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📦 Merging all stages into final directory...")
    
    # 收集所有文件（Stage 5 的更新版本优先）
    all_files = {}
    
    # 先收集 Stage 1-4 的文件
    for stage_num in [1, 2, 3, 4]:
        stage_data = generated_parts.get(stage_num, {})
        files = stage_data.get("files", [])
        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            file_path = file_info.get("path") or file_info.get("file_path") or file_info.get("filename") or file_info.get("name") or ""
            if file_path:
                code = file_info.get("code") or file_info.get("content") or file_info.get("source") or file_info.get("source_code") or ""
                if code:
                    all_files[file_path] = code
    
    # Stage 5 的文件会覆盖之前的版本（如果有接口修复）
    stage5_data = generated_parts.get(5, {})
    files = stage5_data.get("files", [])
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        file_path = file_info.get("path") or file_info.get("file_path") or file_info.get("filename") or file_info.get("name") or ""
        if file_path:
            code = file_info.get("code") or file_info.get("content") or file_info.get("source") or file_info.get("source_code") or ""
            if code:
                all_files[file_path] = code  # 覆盖之前的版本
    
    # 保存所有文件到最终目录
    for file_path_str, code in all_files.items():
        file_path = final_dir / file_path_str
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
    
    print(f"   ✓ Merged {len(all_files)} files to final/")
    
    # 如果有接口修复信息，显示出来
    if stage5_data.get("interface_fixes"):
        print("\n🔧 Interface Fixes Applied:")
        for fix in stage5_data["interface_fixes"]:
            print(f"   - {fix}")


def main():
    """主函数"""
    import sys
    
    # 默认使用 outputs/final_report.json（在函数内部处理）
    report_path = None
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    
    output_dir = "code_generated_multi_legnet"
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    asyncio.run(generate_code_from_report(report_path, output_dir))


if __name__ == "__main__":
    main()
