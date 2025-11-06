# Machine-Learning-for-Drug-Discovery-Predicting-Molecular-Solubility

🔬 Machine Learning for Drug Discovery: Predicting Molecular Solubility

I’m excited to share my latest data science project, where I built a machine learning model to predict molecular solubility — a key property in pharmaceutical research and drug development!

🎯 Project Overview:
Developed a Linear Regression model that predicts compound solubility (logS) from molecular descriptors, helping accelerate the drug discovery process.

🛠️ Technical Implementation:

Dataset: Delaney Solubility dataset

Features: Molecular Lipophilicity (MolLogP), Molecular Weight (MolWt), Rotatable Bonds, Aromatic Proportion

Model: Linear Regression with train-test split

Evaluation Metrics: Mean Squared Error (MSE) and R² Score

📊 Key Results:
The model effectively captured the relationship between molecular properties and solubility, enabling quick predictions for new compounds.

For example, for a molecule with:

MolLogP = 2.5

MolWt = 150 g/mol

2 Rotatable Bonds

30% Aromaticity

👉 Predicted logS = -2.1 (moderately soluble)

Interpretation:
logS = -2.1 → Solubility = 10^(-2.1) = 0.00794 mol/L ≈ 7.94 mmol/L
💧 In simple terms: ~7.94 millimoles of this compound dissolve per liter of water at room temperature.

💡 Business Impact:

Reduces costly lab testing

Enables virtual screening of large compound libraries

Accelerates early-stage drug discovery

Demonstrates how AI + Chemistry = Faster Innovation
