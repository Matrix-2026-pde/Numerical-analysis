# Numerical Analysis Portfolio

本仓库记录了我在计算数学与数值分析方向的科研基础训练过程。所有算法均为从零手写实现（基于纯 Python + NumPy），并包含详细的数值实验、收敛性分析与 LaTeX 实验报告。

## 📂 目录结构

### 01_LU_Decomposition (线性方程组直接法)
- 实现了带部分主元（Partial Pivoting）的 LU 分解。
- **核心成果**：构造了良态但会产生巨大误差的矩阵，论证了“数值不稳定”与“矩阵病态”是两个独立的概念。附带收敛分析与完整 LaTeX 报告。

### 02_Jacobi_Iteration (线性方程组迭代法及加速)
- 实现了古典 Jacobi、Gauss-Seidel、SOR 以及 Anderson 加速迭代法。
- **核心成果**：数值扫描了 SOR 的最优松弛因子 ω，绘制了谱半径随 ω 变化的 V 型曲线，验证了 Young 定理。对比了不同加速方法的收敛效率（包含可视化图表）。

### 03_Academic_Tools (学术工具链)
- 基于 OpenAlex API 封装了学术文献检索工具，支持全文与标题精确检索，具备基础的文献调研与工程实现能力。

## 🤖 开发工作流：双模型协同与版本控制

在开发上述数值实验的过程中，我深度应用了现代 AI 工程化工作流，以提高科研效率并合理控制计算成本：

- **双模型协同架构 (Multi-Agent Workflow)**：
  - **规划模式 (Plan)**：使用 Claude API 进行数学算法的逻辑推导、代码架构设计以及数值稳定性分析。
  - **执行模式 (Act)**：使用 DeepSeek API 进行具体的 Python 代码实现、终端运行与错误调试。
  - **效果**：将需要深度推理的环节与高频率的代码编写环节分离，在保证算法严谨性的同时，大幅降低了 API Token 的消耗成本，实现了效率与经济效益的最优解。

- **版本控制与可复现性 (Git & GitHub)**：
  - 全程使用 Git 进行本地版本控制与时间片管理。每一个核心算法（如 LU 分解、SOR 迭代）均建立了独立的提交记录（Commit），确保科研成果的绝对可复现性与代码安全。

## 🛠️ 技术栈
- **编程语言**：Python (NumPy, Matplotlib)
- **文档排版**：LaTeX
- **开发工具**：VS Code (Cline, Copilot), Anaconda
- **版本控制**：Git, GitHub

---
*Powered by VS Code + Git + 双模型 AI 协同工作流 (Claude Plan + DeepSeek Act)*
