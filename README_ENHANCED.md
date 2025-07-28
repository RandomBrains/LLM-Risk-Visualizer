# 🛡️ LLM Risk Visualizer Pro

A comprehensive, enterprise-grade dashboard for evaluating and visualizing multilingual Large Language Model (LLM) behavior, risks, and safety metrics with advanced authentication, API integration, and real-time monitoring.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Enhanced Features

### 🔐 Enterprise Security
- **Multi-User Authentication**: Secure login system with role-based access control
- **User Roles**: Admin, Analyst, and Viewer permissions with granular access control
- **Session Management**: JWT-based sessions with configurable timeout
- **Activity Logging**: Comprehensive audit trail of user actions

### 🔌 Real-Time API Integration
- **Live Data Fetching**: Connect to OpenAI, Anthropic, and Google APIs for real-time monitoring
- **Async Processing**: High-performance asynchronous data collection
- **API Health Monitoring**: Connection status and performance tracking
- **Fallback Mechanisms**: Graceful degradation when APIs are unavailable

### 🚨 Advanced Monitoring & Alerting
- **Real-Time Anomaly Detection**: Automated identification of unusual risk patterns
- **Multi-Channel Alerts**: Email, Slack, and webhook notifications
- **Configurable Rules**: Custom alert thresholds and conditions
- **Alert Management**: Acknowledgment and tracking system

### 💾 Enterprise Data Management
- **Database Integration**: SQLite, PostgreSQL support with connection pooling
- **Redis Caching**: High-performance data caching for improved response times
- **Data Export**: Multiple formats (CSV, JSON, Excel) with audit logging
- **Data Retention**: Configurable data lifecycle management

### Core Capabilities
- **Multi-Model Analysis**: Compare risk metrics across popular LLMs including GPT-4, Claude, Gemini, LLaMA-2, and more
- **Multilingual Support**: Analyze model behavior across 10+ languages including English, Chinese, Spanish, Arabic, and others
- **Comprehensive Risk Categories**: Track 6 key risk types:
  - Hallucination
  - Refusal
  - Bias
  - Toxicity
  - Privacy Leakage
  - Factual Errors

### Visualization Components
- **Interactive Dashboards**: 6 specialized tabs for different analysis perspectives
- **Real-time Filtering**: Dynamic model, language, and date range selection
- **Advanced Charts**:
  - Risk heatmaps
  - Time series trends
  - Anomaly detection
  - Cluster analysis
  - Model comparison radars
  - Distribution plots
  - Correlation matrices

### Data Analysis
- **Trend Analysis**: Identify improving or worsening risk patterns over time
- **Statistical Analytics**: Advanced correlation and regression analysis
- **Risk Scoring**: Weighted risk calculations with customizable thresholds
- **Predictive Insights**: Trend forecasting and risk prediction

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

#### Linux/macOS:
```bash
# Clone the repository
git clone https://github.com/yourusername/LLM-Risk-Visualizer.git
cd LLM-Risk-Visualizer

# Make script executable and run setup
chmod +x start.sh
./start.sh setup

# Start the application
./start.sh start
```

#### Windows:
```batch
REM Clone the repository
git clone https://github.com/yourusername/LLM-Risk-Visualizer.git
cd LLM-Risk-Visualizer

REM Run setup
start.bat setup

REM Start the application
start.bat start
```

### Option 2: Manual Setup

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Redis for caching
- (Optional) PostgreSQL for production database

#### Installation Steps

1. **Clone and navigate to the repository:**
```bash
git clone https://github.com/yourusername/LLM-Risk-Visualizer.git
cd LLM-Risk-Visualizer
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Setup environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the application:**
```bash
# Enhanced version (recommended)
streamlit run app_enhanced.py

# Or standard version
streamlit run app.py
```

6. **Access the application:**
   - Open your browser and navigate to `http://localhost:8501`
   - Default admin credentials: `admin` / `admin123` (change on first login)

### Option 3: Docker Deployment

#### Using Docker Compose (Recommended)
```bash
# Start all services (app + Redis + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Using Docker only
```bash
# Build the image
docker build -t llm-risk-visualizer .

# Run the container
docker run -p 8501:8501 llm-risk-visualizer
```

## 📊 Dashboard Overview

### 1. Executive Dashboard
- Real-time risk metrics and KPIs
- Interactive risk score heatmap across models and languages
- Risk distribution by category and severity
- Overall system health indicators

### 2. Trend Analysis
- Time series visualization of risk metrics
- Trend identification (improving/worsening patterns)
- Customizable time periods and granularity
- Comparative analysis across models and languages

### 3. Deep Analysis
- Advanced statistical analysis and correlations
- Model comparison radar charts
- Language-wise performance breakdown
- Custom risk scoring algorithms

### 4. Anomaly Detection
- Real-time anomaly identification with configurable sensitivity
- Anomaly timeline visualization and analysis
- Detailed anomaly reports with context
- Statistical significance indicators

### 5. Reports & Export
- Comprehensive risk summary matrices
- Data export in multiple formats (CSV, JSON, Excel)
- Automated report generation
- Export audit logging

### 6. User Profile & Admin Panel
- User account management and preferences
- Role-based access control administration
- API configuration and health monitoring
- System health and performance metrics

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Database Configuration
DATABASE_URL=sqlite:///risk_data.db

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT Secret (required for authentication)
JWT_SECRET_KEY=your-super-secret-jwt-key-here

# API Keys (optional)
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Email Configuration (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Alert Recipients
ALERT_EMAIL_RECIPIENTS=admin@company.com,security@company.com
```

### User Roles and Permissions

| Role | View Dashboard | Export Data | Manage Users | Configure APIs | View Logs |
|------|----------------|-------------|--------------|----------------|-----------|
| **Admin** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Analyst** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Viewer** | ✅ | ❌ | ❌ | ❌ | ❌ |

### API Integration

The system supports integration with major LLM providers:

- **OpenAI**: GPT-4, GPT-3.5 monitoring
- **Anthropic**: Claude safety metrics
- **Google**: Gemini performance data
- **Custom APIs**: Extensible connector framework

## 🛠️ Project Structure

```
LLM-Risk-Visualizer/
│
├── 📱 Applications
│   ├── app.py                 # Standard Streamlit application
│   └── app_enhanced.py        # Enhanced application with full features
│
├── 🔧 Core Modules
│   ├── config.py             # Configuration and constants
│   ├── data_processor.py     # Data processing and analysis logic
│   ├── visualizations.py     # Chart and visualization components
│   └── sample_data.py        # Sample data generation
│
├── 🔐 Security & Authentication
│   ├── auth.py               # User authentication and authorization
│   └── database.py           # Database management and operations
│
├── 🔌 Integration & Monitoring
│   ├── api.py                # API integration for live data
│   └── monitoring.py         # Monitoring and alerting system
│
├── 🚀 Deployment
│   ├── Dockerfile            # Docker containerization
│   ├── docker-compose.yml    # Multi-container orchestration
│   ├── start.sh             # Linux/macOS startup script
│   └── start.bat            # Windows startup script
│
├── 📋 Configuration
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example         # Environment variables template
│   └── INSTALLATION.md      # Detailed installation guide
│
└── 📚 Documentation
    ├── README.md            # This file
    └── LICENSE              # MIT license
```

## 📈 Data Format

The system expects risk data in the following standardized format:

```python
{
    "Date": datetime,           # Timestamp of the measurement
    "Model": str,              # LLM model name (e.g., "GPT-4")
    "Language": str,           # Language code or name
    "Risk_Category": str,      # Risk type (Hallucination, Bias, etc.)
    "Risk_Rate": float,        # Risk score (0.0 to 1.0)
    "Sample_Size": int,        # Number of samples analyzed
    "Confidence": float,       # Confidence interval (0.0 to 1.0)
    "Data_Source": str         # Source of the data (API, manual, etc.)
}
```

## 🔧 Advanced Configuration

### Database Configuration

#### SQLite (Default)
```python
DATABASE_URL = "sqlite:///risk_data.db"
```

#### PostgreSQL (Production)
```python
DATABASE_URL = "postgresql://username:password@localhost:5432/llm_risk_db"
```

### Monitoring Rules

Custom alert rules can be configured:

```python
AlertRule(
    id="custom_threshold",
    name="Custom Risk Threshold",
    description="Alert when risk exceeds custom threshold",
    condition={
        "metric": "risk_rate",
        "operator": ">",
        "threshold": 0.75
    },
    severity="high",
    cooldown_minutes=30,
    notification_methods=["email", "slack"]
)
```

### API Connectors

Extend the system with custom API connectors:

```python
class CustomAPIConnector(APIConnector):
    def fetch_risk_data(self, start_date, end_date, **kwargs):
        # Implement custom API logic
        return processed_data
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Install development dependencies**: `pip install -r requirements.txt`
4. **Make your changes** with proper tests
5. **Run tests**: `pytest tests/`
6. **Format code**: `black . && flake8`
7. **Commit changes**: `git commit -m 'Add some AmazingFeature'`
8. **Push to branch**: `git push origin feature/AmazingFeature`
9. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/ -v

# Format code
black .

# Check code quality
flake8 .
mypy .
```

## 📝 Use Cases

### Enterprise AI Safety Teams
- **Continuous Monitoring**: Real-time tracking of AI system behavior
- **Compliance Reporting**: Automated generation of safety compliance reports
- **Risk Trend Analysis**: Long-term pattern identification and mitigation

### Research Organizations
- **Multi-Model Studies**: Comparative analysis across different LLMs
- **Cross-Lingual Research**: Understanding model behavior across languages
- **Bias Detection**: Comprehensive bias analysis and measurement

### Product Teams
- **Model Selection**: Data-driven decisions on LLM selection
- **Performance Monitoring**: Ongoing assessment of deployed models
- **Quality Assurance**: Systematic testing and validation workflows

### Regulatory Compliance
- **Audit Trails**: Comprehensive logging for regulatory requirements
- **Risk Documentation**: Detailed risk assessment documentation
- **Incident Response**: Rapid identification and response to AI safety incidents

## 🔮 Roadmap

### Version 3.1 (Q2 2025)
- [ ] **Advanced Analytics**: Machine learning-based risk prediction
- [ ] **Custom Dashboards**: User-configurable dashboard layouts
- [ ] **API Rate Limiting**: Enhanced API management and throttling
- [ ] **Mobile App**: Native mobile application for monitoring

### Version 3.2 (Q3 2025)
- [ ] **Federated Learning**: Collaborative risk assessment across organizations
- [ ] **Advanced Visualizations**: 3D risk landscapes and network graphs
- [ ] **Integration APIs**: RESTful APIs for third-party integrations
- [ ] **Advanced Reporting**: Automated executive summaries and insights

### Version 4.0 (Q4 2025)
- [ ] **AI-Powered Insights**: Automated risk analysis and recommendations
- [ ] **Real-Time Collaboration**: Multi-user real-time analysis sessions
- [ ] **Enterprise SSO**: Integration with enterprise identity providers
- [ ] **Advanced ML Models**: Custom risk scoring algorithms

## 🚨 Security & Privacy

### Data Protection
- **Encryption**: All sensitive data encrypted at rest and in transit
- **Access Control**: Role-based access control with audit logging
- **Data Retention**: Configurable data retention policies
- **GDPR Compliance**: Privacy-by-design architecture

### Security Best Practices
- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Protection**: Parameterized queries and ORM usage
- **HTTPS Only**: TLS encryption for all communications

## 🐛 Troubleshooting

### Common Issues

#### Authentication Problems
```bash
# Reset admin password
python -c "from auth import AuthManager; AuthManager().db.create_user('admin', 'admin@example.com', 'newpassword123', 'admin')"
```

#### Database Issues
```bash
# Reset database
rm risk_data.db users.db
python -c "from database import DatabaseManager; from auth import DatabaseManager as AuthDB; DatabaseManager(); AuthDB()"
```

#### Port Conflicts
```bash
# Use different port
streamlit run app_enhanced.py --server.port 8502
```

#### Memory Issues
- Increase system memory or use Redis caching
- Reduce data retention period
- Implement data pagination

## 📞 Support & Community

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and questions
- **Documentation**: Comprehensive guides and tutorials
- **Email Support**: enterprise@yourcompany.com (for enterprise customers)

### Community Resources
- **Discord**: Real-time community chat
- **Blog**: Updates and best practices
- **YouTube**: Video tutorials and demos
- **LinkedIn**: Professional updates and insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Technologies
- **[Streamlit](https://streamlit.io/)**: Interactive web app framework
- **[Plotly](https://plotly.com/)**: Advanced data visualization
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[Redis](https://redis.io/)**: High-performance caching
- **[PostgreSQL](https://postgresql.org/)**: Enterprise database

### Contributors
- **Wolfgang Dremmler** - *Initial work and architecture*
- **Community Contributors** - *Feature enhancements and bug fixes*

### Special Thanks
- OpenAI, Anthropic, and Google for API access
- The open-source community for foundational libraries
- Beta testers and early adopters for valuable feedback

---

**Note**: This dashboard includes both sample data for demonstration and real API integration capabilities. For production deployment, ensure proper API keys and security configurations are in place.

---

**Version**: 3.0 | **Last Updated**: January 2025 | **Status**: Production Ready

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)