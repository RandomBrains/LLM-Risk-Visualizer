# LLM Risk Visualizer - 安装和运行指南

## 🚀 快速开始

### 1. Python环境要求
- Python 3.8 或更高版本
- pip 包管理器

### 2. 安装步骤

#### Windows系统：
```bash
# 如果Python未安装，请从Microsoft Store安装Python或访问python.org下载
# 验证Python安装
python --version

# 克隆或下载项目到本地目录
# cd 到项目目录

# 安装依赖包
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

#### Linux/macOS系统：
```bash
# 验证Python安装
python3 --version

# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# 安装依赖包
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

### 3. 访问应用
打开浏览器，访问：`http://localhost:8501`

## 🔧 故障排除

### 常见问题：

1. **Python未找到错误**
   - 确保Python已正确安装并添加到系统PATH
   - Windows用户可从Microsoft Store安装Python

2. **模块导入错误**
   - 确保所有依赖包已安装：`pip install -r requirements.txt`
   - 检查Python版本是否符合要求

3. **端口占用问题**
   - 使用不同端口运行：`streamlit run app.py --server.port 8502`

4. **数据文件缺失**
   - 应用使用内置的示例数据生成器，无需外部数据文件
   - 如需真实数据，请参考API集成文档

## 📦 Docker部署（可选）

如果您prefer使用Docker：

```bash
# 构建Docker镜像
docker build -t llm-risk-visualizer .

# 运行容器
docker run -p 8501:8501 llm-risk-visualizer
```

## 🔍 功能验证

安装完成后，您应该能够：
- ✅ 查看多模型风险对比仪表板
- ✅ 分析时间趋势数据
- ✅ 查看异常检测结果
- ✅ 导出风险数据报告
- ✅ 使用交互式筛选功能

## 📞 获取帮助

如果遇到问题：
1. 检查Python和依赖包版本
2. 查看终端错误信息
3. 确认所有文件完整下载
4. 参考项目README.md获取更多信息

---
**注意**: 本应用使用模拟数据进行演示。生产环境中请连接真实的LLM监控数据源。