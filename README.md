# PRD Generator Assistant

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## 🎯 Overview

This Flask-based web application integrates with OpenAI's API to generate comprehensive Product Requirements Documents (PRDs). It utilizes multiple AI agents to analyze requirements and produce structured documentation.

## ✨ Features

- Multi-agent AI system for requirement analysis
- Interactive web interface
- Real-time document generation
- Markdown to Word document conversion
- Session management
- Progress tracking
- Error handling and logging

## 🔧 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Windows, macOS, or Linux operating system
- Modern web browser (Chrome, Firefox, Edge recommended)
- OpenAI API key with appropriate permissions

## 📥 Installation

1. **Clone the Repository**
   ```powershell
   git clone https://github.com/yourusername/openai-assistant-clean.git
   cd openai-assistant-clean
   ```

2. **Create Virtual Environment** (Recommended)
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Environment Setup**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ADMIN_USERNAME=your_username
   ADMIN_PASSWORD=your_password
   REDIS_HOST=localhost  # Optional
   REDIS_PORT=6379      # Optional
   ```

2. **Redis Configuration** (Optional)
   - Install Redis if you want to use it for session management
   - Update Redis configuration in `.env`

## 🚀 Running the Application

1. **Start the Application**
   ```powershell
   python .\new-prd-workflow.py
   ```

2. **Access the Web Interface**
   - Open your browser
   - Navigate to `http://localhost:7001`
   - Log in with your configured credentials

## 📖 Usage Guide

### Login Page
1. Enter your admin credentials
2. Click "Login" to access the system

### Page 1: Initial Input
1. Enter industry details
2. Provide sector information
3. Specify geography
4. Define project intent
5. List key features

### Page 2: Review & Edit
1. Review generated content
2. Edit if necessary
3. Proceed to requirements

### Page 3: Requirements Generation
1. Review high-level requirements
2. Click "Approve & Generate PRD Draft"
3. Wait for processing (progress indicator shown)

### Page 4: Final Document
1. Review all sections
2. Click "Download as Word" for final document
3. Save the generated PRD

## ❗ Troubleshooting

### Common Issues
1. **Connection Errors**
   ```
   Error: OpenAI API connection failed
   Solution: Check API key and internet connection
   ```

2. **Document Generation Failed**
   ```
   Error: Document generation failed
   Solution: Check browser console (F12) for detailed logs
   ```

3. **Session Expired**
   ```
   Error: Session expired
   Solution: Log in again
   ```

### Logging
- Application logs are stored in `app.log`
- Use browser DevTools (F12) for frontend issues
- Check terminal output for backend errors

## 💻 Development

### Project Structure
```
openai-assistant-clean/
├── templates/
│   ├── page0_login.html
│   ├── page1_input.html
│   ├── page2_agents.html
│   ├── page3_prompt_picker.html
│   └── page4_final_output.html
├── static/
│   └── js/
│       └── global.js
├── new-prd-workflow.py
├── requirements.txt
└── .env
```

### Running in Debug Mode
```powershell
$env:FLASK_ENV = "development"
python .\new-prd-workflow.py
```

### Testing
- Run unit tests: `python -m pytest tests/`
- Check code style: `python -m flake8`

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note**: Keep your API keys and credentials secure. Never commit sensitive information to version control.