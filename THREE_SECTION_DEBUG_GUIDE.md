# Three Section Server Debug Guide

## Server Stops When UI is Accessed - Troubleshooting

### Quick Diagnostic Steps

1. **Use the Safe Launcher**
   ```bash
   python launch_three_section.py
   ```
   This provides enhanced logging and error handling.

2. **Check Health Endpoints**
   ```bash
   # After starting server, test these URLs:
   http://localhost:5001/health
   http://localhost:5001/debug-info
   ```

3. **Run Troubleshooting Script**
   ```bash
   python troubleshoot_three_section.py
   ```

### Common Issues and Solutions

#### 1. **Missing Dependencies**
**Symptom**: ImportError when starting server
**Solution**:
```bash
pip install -r requirements.txt
```

#### 2. **Template Rendering Error**
**Symptom**: Server crashes when accessing `/three-section`
**Check**: 
- File exists: `templates/poc2_three_section_layout.html`
- No syntax errors in HTML/JavaScript

#### 3. **OpenAI API Issues**
**Symptom**: Server works but AI features fail
**Solution**:
```bash
set OPENAI_API_KEY=your_openai_api_key_here
```

#### 4. **Session/Database Issues**
**Symptom**: Server crashes on epic loading
**Check**: 
- Session configuration
- Database connectivity (if using ChromaDB)

#### 5. **JavaScript Errors**
**Symptom**: Page loads but functionality doesn't work
**Check**: Browser console for JavaScript errors

### Debug Process

#### Step 1: Start with Safe Launcher
```bash
python launch_three_section.py
```
Watch console output for immediate errors.

#### Step 2: Test Basic Endpoints
```bash
curl http://localhost:5001/health
```
Should return: `{"status": "ok", ...}`

#### Step 3: Test Debug Info
```bash
curl http://localhost:5001/debug-info
```
Check configuration and session status.

#### Step 4: Test Main UI
```bash
# In browser: http://localhost:5001/three-section
```
Check browser console (F12) for JavaScript errors.

#### Step 5: Check Logs
Look at these log files:
- `three_section_debug.log` (if using safe launcher)
- `app.log` (standard application log)
- Console output

### Error Patterns to Look For

#### Backend Crashes
```
Error: ImportError: No module named 'some_module'
```
**Fix**: Install missing dependencies

```
Error: TemplateNotFound: poc2_three_section_layout.html
```
**Fix**: Check template file exists in templates/ folder

#### Frontend Issues
```javascript
TypeError: Cannot read property 'style' of null
```
**Fix**: Check HTML element IDs match JavaScript

```
Failed to fetch
```
**Fix**: Check server is running and endpoints are accessible

### Advanced Debugging

#### Enable Debug Mode
```bash
set FLASK_DEBUG=true
python poc2_backend_processor_three_section.py
```

#### Check Network Requests
1. Open browser Developer Tools (F12)
2. Go to Network tab
3. Access the page
4. Look for failed requests (red entries)

#### Python Debug
Add this to troubleshoot specific functions:
```python
import traceback
try:
    # your code here
except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
```

### Safe Mode Testing

If server keeps crashing, test components individually:

#### 1. Test Backend Only
```python
# Create minimal test file
from flask import Flask
app = Flask(__name__)

@app.route('/test')
def test():
    return "Backend working"

if __name__ == '__main__':
    app.run(port=5001, debug=True)
```

#### 2. Test Template Rendering
```python
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/template-test')
def template_test():
    return render_template('poc2_three_section_layout.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
```

### Getting Help

When reporting issues, provide:

1. **Error messages** from console/logs
2. **Python version**: `python --version`
3. **Installed packages**: `pip list`
4. **Environment variables** (without API keys)
5. **Browser console errors** (F12 → Console)
6. **Network request failures** (F12 → Network)

### Quick Fixes

#### Reset Everything
```bash
# 1. Stop server (Ctrl+C)
# 2. Clear session data
rm -rf __pycache__/
# 3. Restart with safe launcher
python launch_three_section.py
```

#### Minimal Working Test
```bash
# Test if basic Flask works
python -c "from flask import Flask; app=Flask(__name__); app.run(port=5001, debug=True)"
```

#### Check Port Conflicts
```bash
# Windows
netstat -ano | findstr :5001

# Kill process if needed
taskkill /PID <process_id> /F
```

This guide should help identify and resolve the server crash issues!
