# Drug-prediction-based-on-the-health-parameters-of-a-patient-using-Decision-tree

## Notebook Overview

This Jupyter notebook demonstrates how to build and evaluate a Decision Tree classifier to predict which drug (A, B, C, X, or Y) is most suitable for a patient based on their health parameters: Age, Sex, Blood Pressure, Cholesterol, and Na_to_K ratio.

### How to Run

1. Open `Decision-tree-classifier-drug-pred-v1.ipynb` in Jupyter Notebook or JupyterLab.
2. Run cells from top to bottom (or use "Run All").
3. The notebook includes pip install commands for all required dependencies.

### Dependencies

- numpy==2.2.0
- pandas==2.2.3
- scikit-learn
- matplotlib

### Dataset

The dataset is loaded from IBM's public cloud storage:
```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv
```

**Dataset Structure:**
- **200 samples** (patients)
- **6 columns:** Age, Sex, BP, Cholesterol, Na_to_K, Drug
- **Target variable:** Drug (5 classes: drugA, drugB, drugC, drugX, drugY)
- **No missing values**

### Key Steps

1. **Load & Explore** – Load dataset and perform basic analytics
2. **Preprocess** – Label-encode categorical features (Sex, BP, Cholesterol)
3. **Analyze** – Check correlations and class distribution
4. **Model** – Train DecisionTreeClassifier (entropy criterion, max_depth=4)
5. **Evaluate** – Compute accuracy and visualize the decision tree
6. **Interpret** – Extract and display human-readable decision rules
7. **Practice** – Retrain with max_depth=3 and compare performance

### Label Encoding Mapping

The notebook converts categorical values to numeric as follows:
- **Sex:** M → 1, F → 0
- **BP:** High → 0, Low → 1, Normal → 2
- **Cholesterol:** High → 0, Normal → 1

**Why encoding?** Decision trees need numeric input for continuous comparisons (e.g., "is BP > 0.5?").

### Expected Results

- **Accuracy (max_depth=4):** ~98.33% on the test set (59 out of 60 samples correct)
- **Top Features:** Na_to_K and BP are the most correlated with drug selection
- **Class Distribution:** Drug X and Drug Y have significantly more records than A, B, C

### What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that:
- Recursively splits data based on feature values to minimize impurity (using entropy or Gini).
- Creates a tree structure with internal nodes (splits), branches (conditions), and leaf nodes (predictions).
- Is interpretable: you can trace a path from root to leaf to understand the decision logic.

**Example decision rule extracted from the trained tree:**
```
If Na_to_K > 14.627:
    Predict Drug Y
Else if BP = High and Age ≤ 50.5:
    Predict Drug A
Else if BP = High and Age > 50.5:
    Predict Drug B
... (and so on)
```

### How the Model Works

1. **Train-Test Split:** Data is split 70% training, 30% testing (random_state=32 for reproducibility).
2. **Entropy Criterion:** The tree uses information entropy to choose the best feature at each split.
   - Entropy measures disorder; lower entropy = purer node (more homogeneous class labels).
3. **Max Depth:** Limited to 4 levels to balance accuracy and interpretability.
4. **Prediction:** For a new patient, the model traces down the tree based on their features and returns the predicted drug.

### Key Insights

- **Reducing tree depth** (e.g., from 4 to 3) decreases model complexity and overfitting risk, but may reduce accuracy if important patterns are lost.
- **Feature importance:** Na_to_K and BP dominate the decision logic.
- **Class imbalance:** Drug X and Y are overrepresented; this may bias the model slightly.
- **Results are reproducible** using random_state=32.

### Practice Exercises

The notebook includes practice cells to:

1. **Extract Decision Rules:**
   - Print textual tree representation using `export_text()`
   - Manually derive human-readable rules for each drug class
   
   Example output:
   ```
   Drug Y : Na_to_K > 14.627
   Drug A : Na_to_K ≤ 14.627, BP = High, Age ≤ 50.5
   Drug B : Na_to_K ≤ 14.627, BP = High, Age > 50.5
   Drug C : Na_to_K ≤ 14.627, BP = Low, Cholesterol = High
   Drug X : Na_to_K ≤ 14.627, BP = Normal, Cholesterol = High
   ```

2. **Compare Tree Depths:**
   - Retrain with max_depth=3 (shallower tree)
   - Compare accuracy: typically slightly lower but simpler rules
   - Demonstrates the bias-variance tradeoff

### Interpreting the Tree Visualization

The tree plot shows:
- **Internal nodes:** Feature split (e.g., "Na_to_K ≤ 14.627?")
- **Branches:** Yes (left) or No (right)
- **Leaf nodes:** Predicted drug class (color-coded)
- **Entropy/Samples:** Measure of purity and sample count at each node

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Package import errors | Run pip install cells at the top of the notebook |
| Different accuracy than expected | Check if random_state=32 is set in train_test_split |
| Tree visualization too small | Adjust figure size in plot_tree() call |
| Correlation values confusing | Ensure Drug_num is created (0-4 mapping) before corr() |

### References

- **Scikit-learn Decision Trees:** https://scikit-learn.org/stable/modules/tree.html
- **Information Entropy:** https://en.wikipedia.org/wiki/Entropy_(information_theory)
- **Decision Tree Hyperparameters:** max_depth, min_samples_split, criterion (entropy vs gini)

### Next Steps

- Try different max_depth values (1–10) and plot accuracy vs. depth curve
- Experiment with other criteria (gini vs entropy)
- Apply to other multiclass classification problems
- Combine multiple trees (Random Forest, Gradient Boosting) for better performance
