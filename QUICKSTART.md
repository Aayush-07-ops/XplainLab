# 🚀 Quick Start Guide - XplainAI

Get XplainAI up and running in 3 minutes!

## ⚡ Super Quick Setup

### For Windows (PowerShell)

```powershell
# Navigate to the project directory
cd C:\Users\ankus\xplainai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### For Mac/Linux (Terminal)

```bash
# Navigate to the project directory
cd ~/xplainai

# Install dependencies
pip3 install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🌐 Accessing the App

After running the command, Streamlit will:
1. Automatically open your browser
2. Navigate to `http://localhost:8501`

If the browser doesn't open automatically:
- Open your browser manually
- Go to: `http://localhost:8501`

## 🎯 First Time Using the App?

### Follow These 4 Simple Steps:

#### 1️⃣ Choose Dataset
Click on either:
- **🏦 Loan Approval** - for banking scenarios
- **🎓 Student Admission** - for education scenarios

#### 2️⃣ Select Algorithm
Pick an AI algorithm:
- **Decision Tree** - Easy to understand
- **Logistic Regression** - Fast & reliable
- **KNN** - Learns from similar cases
- **Random Forest** - Most accurate

#### 3️⃣ Enter Details
Fill in the form with your information:
- All fields have helpful tooltips
- Default values are provided
- Adjust as needed

#### 4️⃣ Get Results
Click "🔮 Get Prediction" and see:
- ✅ Approved or ❌ Rejected
- Why the decision was made
- What factors mattered most
- Interactive charts

## 🎨 Tips for Best Experience

### Experiment Freely
- Try different values to see how predictions change
- Compare different algorithms on the same data
- Switch between datasets to explore both scenarios

### Learning Mode
- Read the explanations carefully
- Check the feature importance charts
- Try to predict before clicking the button

### Navigation
- Use the **Back** buttons to go to previous steps
- Use **Try Again** to test different inputs
- Use **Start Over** to begin fresh

## 🛠️ Troubleshooting

### App Won't Start?

**Problem**: `streamlit: command not found`
```powershell
# Solution: Install streamlit
pip install streamlit
```

**Problem**: Module not found errors
```powershell
# Solution: Reinstall all dependencies
pip install -r requirements.txt
```

**Problem**: Port already in use
```powershell
# Solution: Use a different port
streamlit run app.py --server.port 8502
```

### Slow Performance?

The first time you select a dataset, models are trained. This takes a few seconds. After that, predictions are instant!

### Browser Issues?

- **Cache Problems**: Clear your browser cache
- **Wrong URL**: Make sure you're using `localhost:8501`
- **Firewall**: Check if your firewall is blocking the port

## 📱 Using on Mobile?

XplainAI is fully responsive!

1. Find your computer's IP address:
   ```powershell
   # Windows
   ipconfig
   
   # Mac/Linux
   ifconfig
   ```

2. On your phone's browser, go to:
   ```
   http://[YOUR_IP]:8501
   ```

## 🎓 Example Scenarios to Try

### Loan Approval

**Scenario 1: Strong Candidate**
- Credit Score: 750
- Annual Income: $80,000
- Loan Amount: $150,000
- Employment Years: 10
- Debt-to-Income: 0.25

**Scenario 2: Risky Candidate**
- Credit Score: 550
- Annual Income: $35,000
- Loan Amount: $200,000
- Employment Years: 1
- Debt-to-Income: 0.65

### Student Admission

**Scenario 1: Strong Applicant**
- GPA: 3.8
- Test Score: 1450
- Extracurriculars: 7
- Essay Score: 9.0

**Scenario 2: Average Applicant**
- GPA: 3.0
- Test Score: 1100
- Extracurriculars: 2
- Essay Score: 6.5

## 🔄 Stopping the App

To stop the Streamlit server:
- Press `Ctrl + C` in the terminal/PowerShell window
- Or just close the terminal window

## 🎉 You're Ready!

That's it! You now know everything you need to use XplainAI.

### Need More Help?

- Read the full `README.md` for detailed documentation
- Check the tooltips (ℹ️) in the app
- Experiment and explore!

---

**Happy Exploring! 🚀**

Made with ❤️ by XplainAI
