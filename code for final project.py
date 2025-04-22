import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import shap
import re
import matplotlib.pyplot as plt
import seaborn as sns
# Step 1: Load Dataset
df = pd.read_csv("part_openai_preference.csv")
df['prompt'] = df['prompt'].astype(str)
df = df[['prompt', 'weighted_results_image2_preference',
         'weighted_results_image2_coherence', 'weighted_results_image2_alignment']].dropna()
df['label'] = df['weighted_results_image2_preference'].astype(int)

# Step 2: Load NRC Emotion Dictionary
def load_nrc_emotion_dict(filepath):
    emotion_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, emotion, flag = line.strip().split('\t')
            if int(flag) == 1:
                if word not in emotion_dict:
                    emotion_dict[word] = set()
                emotion_dict[word].add(emotion)
    return emotion_dict

emotion_dict = load_nrc_emotion_dict('./NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

# Step 3: Color word list
color_words = {
    'red', 'blue', 'green', 'yellow', 'black', 'white',
    'purple', 'pink', 'orange', 'gray', 'brown', 'gold', 'silver'
}

# Step 4: Feature Functions
def count_emotion_words(prompt):
    tokens = prompt.lower().split()
    return sum(1 for token in tokens if token in emotion_dict)

def count_emotion_categories(prompt):
    tokens = prompt.lower().split()
    cat_set = set()
    for token in tokens:
        if token in emotion_dict:
            cat_set |= emotion_dict[token]
    return len(cat_set)

def detect_roles(prompt):
    patterns = [
        # who are you
        r"\byou are\b", r"\byou're\b", r"\byou become\b", r"\bpretend to be\b",
        # imagine
        r"\bimagine\b", r"\bsuppose\b", r"\benvision\b", r"\bthink of\b",
        # story
        r"\btell a story\b", r"\bdescribe\b", r"\bnarrate\b", r"\bpicture this\b",
        # supposition
        r"\bwhat if\b", r"\bhow would you\b", r"\bwould you rather\b", r"\bin a world where\b",
        # scenario
        r"\byou find yourself\b", r"\bin a situation where\b", r"\blet's say\b"
    ]

    return int(any(re.search(pat, prompt.lower()) for pat in patterns))


def detect_constraints(prompt):
    keywords = [
        "must include", "use only", "no more than", "limit to",
        "in exactly", "with only", "without", "strictly",
        "as a rule", "follow this", "in this format",
    ]
    return int(any(k in prompt.lower() for k in keywords))

def contains_color(prompt):
    tokens = prompt.lower().split()
    return int(any(word in color_words for word in tokens))

# Step 5: Apply Features
df['prompt_length'] = df['prompt'].apply(lambda x: len(x.split()))
df['emotion_word_count'] = df['prompt'].apply(count_emotion_words)
df['emotion_category_count'] = df['prompt'].apply(count_emotion_categories)
df['has_role_narrative'] = df['prompt'].apply(detect_roles)
df['has_constraint'] = df['prompt'].apply(detect_constraints)
df['contains_color'] = df['prompt'].apply(contains_color)

# Step 6: Define feature set
features = [
    'prompt_length', 'emotion_word_count', 'emotion_category_count',
    'has_role_narrative', 'has_constraint', 'contains_color',
    'weighted_results_image2_alignment', 'weighted_results_image2_coherence'
]

df.dropna(subset=features + ['label'], inplace=True)
eda_features = [
    'prompt_length', 'emotion_word_count', 'emotion_category_count',
    'has_role_narrative', 'has_constraint', 'contains_color',
    'weighted_results_image2_alignment', 'weighted_results_image2_coherence',
    'label'
]

eda_df = df[eda_features].copy()

# Heatmap of correlations
plt.figure(figsize=(8, 6))
corr = eda_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_heatmap_correlations.png")
plt.close()
# Step 7: Train model
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = logreg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: SHAP
explainer = shap.Explainer(logreg, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP - Logistic Regression with NRC Emotion + Color")
plt.tight_layout()
plt.savefig("shap_nrc_color.png")