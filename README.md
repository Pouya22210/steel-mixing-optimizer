# 🔬 Steel Mixing Optimizer

A professional web-based application for finding optimal mixing ratios of steel powders to achieve target chemical compositions.

![Version](https://img.shields.io/badge/version-4.3-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- 🔐 **Secure Authentication** - Username/password protection
- 🧪 **Multiple Solutions** - View top N best steel combinations
- 🌡️ **MS Temperature** - Automatic martensite start temperature calculation
- 📊 **Interactive Visualizations** - Pie charts, bar charts, deviation analysis
- 📈 **Complete Analysis** - Shows all elements including non-target ones
- 💾 **Export Results** - Download results as CSV
- 🎯 **Target Optimization** - Specify composition targets and tolerances
- 🗑️ **Database Management** - Add/remove alloys from database
- 📱 **Responsive Design** - Works on desktop and mobile

## 🌡️ MS Temperature Formula

```
Ms (°C) = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al
```

Comprehensive formula accounting for all major alloying elements.

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/steel-mixing-optimizer.git
cd steel-mixing-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python steel_mixing_optimizer_v4.3_authenticated.py
```

4. Open your browser:
```
http://localhost:8050
```

5. Login with default credentials:
- Username: `admin` / Password: `steel2025`
- Username: `user` / Password: `password123`
- Username: `engineer` / Password: `metallurgy`

## 🌐 Deploy to Internet

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on:
- Deploying to Render (free)
- Setting up a custom domain
- Alternative hosting platforms
- Environment configuration

## 📖 How to Use

1. **Load Database**: Upload Excel file or use sample database
2. **Set Target**: Define target composition (e.g., C=0.51%, Cr=17.1%, Ni=6.4%)
3. **Set Tolerances**: Specify acceptable deviations for each element
4. **Configure**: Choose max combination size and number of solutions
5. **Optimize**: Click "Find Optimal Mix" button
6. **View Results**: See mixing ratios, MS temperatures, and complete analysis
7. **Export**: Download results as CSV

## 📊 Database Format

Your Excel file should have this structure:

| Name      | C    | Mn  | Cr   | Mo  | Ni   | W   | V   | Cu  | Co  | Al  | Si  |
|-----------|------|-----|------|-----|------|-----|-----|-----|-----|-----|-----|
| 440C-SLM  | 1.02 | 0.6 | 17.4 | 0.65| 0.1  | 0   | 0   | 0   | 0   | 0   | 0   |
| 316-SLM   | 0.01 | 1.5 | 16.8 | 2.5 | 12.7 | 0   | 0   | 0   | 0   | 0   | 0.7 |
| ...       | ...  | ... | ...  | ... | ...  | ... | ... | ... | ... | ... | ... |

- First column: Steel name
- Other columns: Element compositions in wt.%
- Use 0 for absent elements

## 🔐 Changing Login Credentials

Edit the `VALID_USERNAME_PASSWORD_PAIRS` dictionary in the Python file:

```python
VALID_USERNAME_PASSWORD_PAIRS = {
    'your_username': 'your_password',
    'another_user': 'another_pass',
}
```

## 🛠️ Technology Stack

- **Framework**: Dash (Plotly)
- **UI Components**: Dash Bootstrap Components
- **Authentication**: dash-auth
- **Data Processing**: Pandas, NumPy
- **Optimization**: SciPy
- **Visualization**: Plotly
- **Deployment**: Gunicorn

## 📁 Project Structure

```
steel-mixing-optimizer/
│
├── steel_mixing_optimizer_v4.3_authenticated.py  # Main application
├── requirements.txt                               # Python dependencies
├── Procfile                                       # Deployment config
├── .gitignore                                     # Git ignore rules
├── DEPLOYMENT_GUIDE.md                           # Deployment instructions
└── README.md                                      # This file
```

## 🔄 Version History

### v4.3 (Current)
- Added username/password authentication
- Comprehensive MS temperature formula
- Color-coded MS temperature badges
- Production-ready configuration

### v4.2
- Multiple solutions display
- Complete element breakdown
- Inline row selection for removal

### v4.1
- Alloy removal functionality
- Interactive database management

### v4.0
- Interactive visualizations
- Pie charts, bar charts, deviation analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- Assistant AI

## 🙏 Acknowledgments

- Andrews (1965) and other researchers for MS temperature formulas
- Dash/Plotly community for excellent documentation
- Users for feedback and feature requests

## 📧 Support

For issues or questions:
1. Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Review application logs
3. Test locally first
4. Open an issue on GitHub

## 🎯 Use Cases

- **Research**: Alloy design and optimization
- **Manufacturing**: Production planning
- **Education**: Teaching materials science
- **Industry**: Quality control and composition targeting

---

**Built with ❤️ for the metallurgy community**

⭐ Star this repository if you find it useful!
