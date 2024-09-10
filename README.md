# employemntTracker

Here’s the guide formatted for Markdown so you can include it in your `README.md`:

```markdown
# Running a Flask Application

This guide will help you set up and run a Flask application on Windows.

## Prerequisites
- **Python Installed**: Ensure that Python is installed on your machine.

---

## Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/).
2. **Install Python**:
   - During installation, **check the option** "Add Python to PATH".
   - Choose to **install for all users**.
3. Verify the installation by opening a terminal (Command Prompt or PowerShell) and running the following command:
   ```bash
   python --version
   ```

---

## Step 2: Navigate to Your Project Directory

1. **Open Command Prompt or PowerShell**.
2. Navigate to your Flask project folder using the `cd` command:
   ```bash
   cd path\to\your\flask\project
   ```

---

## Step 3: Set Up a Virtual Environment

1. Create a virtual environment by running:
   ```bash
   python -m venv venv
   ```
   This will create a `venv` folder in your project directory containing the virtual environment.

2. **Activate the virtual environment**:
   ```bash
   venv\Scripts\activate
   ```
   You should see `(venv)` in your terminal, indicating that the virtual environment is activated.

---

## Step 4: Install Dependencies

1. Once the virtual environment is activated, install all the dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 5: Run the Flask Application

1. To start your Flask application, run:
   ```bash
   python run.py
   ```

2. Open your browser and visit:
   ```
   http://127.0.0.1:5000
   ```

---

## Additional Notes

- If you don’t have a `requirements.txt` file, you can create one with the installed dependencies by running:
  ```bash
  pip freeze > requirements.txt
  ```

- To **deactivate** the virtual environment, run:
  ```bash
  deactivate
  ```

---
 
