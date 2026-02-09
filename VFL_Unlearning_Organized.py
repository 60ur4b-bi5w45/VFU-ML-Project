# %% [markdown]
# # Vertical Federated Unlearning (VFU) Project
# 
# A comprehensive framework for VFL with unlearning and verification attacks.
# 
# ## Table of Contents
# 1. Environment Setup
# 2. Dataset Loading  
# 3. VFL Data Partitioning
# 4. VFL Model Architecture
# 5. VFL Training Framework
# 6. Train Models
# 7. VFL Unlearning Methods
# 8. Membership Inference Attack
# 9. Run Unlearning Experiments
# 10. Advanced Verification Attacks

# %% [markdown]
# ---
# ## 1. Environment Setup

# %%
import numpy as np
import pandas as pd
import copy
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ---
# ## 2. Dataset Loading

# %% [markdown]
# ### 2.1 UCI Adult Dataset

# %%
def load_adult_dataset():
    """Load UCI Adult dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'sex',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    
    df = pd.read_csv(url, names=columns, sep=',\\s*', engine='python', na_values='?')
    df = df.dropna()
    df['income'] = (df['income'] == '>50K').astype(int)
    
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country']
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df.drop('income', axis=1), df['income']

X_adult, y_adult = load_adult_dataset()
print(f"UCI Adult: {X_adult.shape[0]} samples, {X_adult.shape[1]} features")

# %% [markdown]
# ### 2.2 German Credit Dataset

# %%
def load_german_credit_dataset():
    """Load German Credit dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
               'savings_status', 'employment', 'installment_rate', 'personal_status',
               'other_parties', 'residence_since', 'property_magnitude', 'age',
               'other_payment_plans', 'housing', 'existing_credits', 'job',
               'num_dependents', 'own_telephone', 'foreign_worker', 'class']
    
    df = pd.read_csv(url, names=columns, sep=' ')
    df['class'] = (df['class'] == 1).astype(int)
    
    categorical_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status',
                        'employment', 'personal_status', 'other_parties', 'property_magnitude',
                        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df.drop('class', axis=1), df['class']

X_german, y_german = load_german_credit_dataset()
print(f"German Credit: {X_german.shape[0]} samples, {X_german.shape[1]} features")

# %% [markdown]
# ### 2.3 MIMIC-III (Synthetic)

# %%
def load_mimic_iii_dataset():
    """Generate synthetic MIMIC-III data."""
    n_samples = 46000
    np.random.seed(SEED)
    
    X = pd.DataFrame({
        'age': np.random.normal(65, 15, n_samples).clip(18, 95),
        'heart_rate': np.random.normal(80, 20, n_samples).clip(40, 180),
        'blood_pressure_systolic': np.random.normal(120, 25, n_samples).clip(60, 200),
        'blood_pressure_diastolic': np.random.normal(80, 15, n_samples).clip(40, 120),
        'respiratory_rate': np.random.normal(18, 5, n_samples).clip(8, 40),
        'temperature': np.random.normal(37, 1, n_samples).clip(34, 42),
        'oxygen_saturation': np.random.normal(96, 3, n_samples).clip(70, 100),
        'glucose': np.random.normal(120, 40, n_samples).clip(40, 400),
        'creatinine': np.random.exponential(1.2, n_samples).clip(0.3, 10),
        'bun': np.random.normal(20, 10, n_samples).clip(5, 100),
        'sodium': np.random.normal(140, 5, n_samples).clip(120, 160),
        'potassium': np.random.normal(4, 0.5, n_samples).clip(2.5, 7),
        'hemoglobin': np.random.normal(12, 2, n_samples).clip(5, 18),
        'wbc': np.random.exponential(10, n_samples).clip(1, 50),
        'platelets': np.random.normal(250, 100, n_samples).clip(20, 600),
        'icu_los': np.random.exponential(5, n_samples).clip(0.5, 60),
        'gender': np.random.binomial(1, 0.55, n_samples),
        'ventilator': np.random.binomial(1, 0.3, n_samples),
        'vasopressor': np.random.binomial(1, 0.2, n_samples),
        'dialysis': np.random.binomial(1, 0.1, n_samples),
    })
    
    mortality_prob = (0.1 + 0.15 * (X['age'] > 70) + 0.2 * X['ventilator'] + 
                     0.15 * X['vasopressor'] + 0.1 * (X['creatinine'] > 2)).clip(0, 0.9)
    y = (np.random.random(n_samples) < mortality_prob).astype(int)
    
    return X, pd.Series(y)

X_mimic, y_mimic = load_mimic_iii_dataset()
print(f"MIMIC-III: {X_mimic.shape[0]} samples, {X_mimic.shape[1]} features")

# %% [markdown]
# ---
# ## 3. VFL Data Partitioning

# %%
class VFLDataPartitioner:
    """Partition data vertically for VFL."""
    
    def __init__(self, X, y, n_parties=2, label_party=0, random_state=42):
        self.X = X
        self.y = y
        self.n_parties = n_parties
        self.label_party = label_party
        self.random_state = random_state
        
    def partition_features(self, method='equal', custom_splits=None):
        np.random.seed(self.random_state)
        all_features = list(self.X.columns)
        
        if method == 'custom' and custom_splits:
            self.feature_splits = custom_splits
        else:
            np.random.shuffle(all_features)
            split_size = len(all_features) // self.n_parties
            self.feature_splits = []
            for i in range(self.n_parties):
                start = i * split_size
                end = None if i == self.n_parties - 1 else start + split_size
                self.feature_splits.append(all_features[start:end])
        
        self.party_data = []
        for i in range(self.n_parties):
            self.party_data.append({
                'X': self.X[self.feature_splits[i]],
                'y': self.y if i == self.label_party else None,
                'features': self.feature_splits[i],
                'is_active': i == self.label_party
            })
        return self.party_data
    
    def summary(self):
        print("VFL Partition Summary")
        print("=" * 40)
        for i, party in enumerate(self.party_data):
            ptype = "Active" if party['is_active'] else "Passive"
            print(f"Party {i} [{ptype}]: {len(party['features'])} features")

# %%
# Partition datasets
adult_parties = VFLDataPartitioner(X_adult, y_adult, 2, 0).partition_features('equal')
german_parties = VFLDataPartitioner(X_german, y_german, 2, 0).partition_features('equal')

hospital = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'respiratory_rate', 'oxygen_saturation', 'icu_los', 'ventilator', 'vasopressor', 'dialysis']
lab = ['glucose', 'creatinine', 'bun', 'sodium', 'potassium', 'hemoglobin', 'wbc', 'platelets', 'temperature']
demo = ['age', 'gender']
mimic_parties = VFLDataPartitioner(X_mimic, y_mimic, 3, 0).partition_features('custom', [hospital, lab, demo])

print("Adult parties:", len(adult_parties))
print("German parties:", len(german_parties))
print("MIMIC parties:", len(mimic_parties))

# %% [markdown]
# ---
# ## 4. VFL Model Architecture

# %%
class BottomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.network(x)

class TopModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

class VFLModel:
    def __init__(self, party_data, embedding_dim=32, hidden_dim=64, device='cpu'):
        self.party_data = party_data
        self.n_parties = len(party_data)
        self.device = device
        self.embedding_dim = embedding_dim
        
        self.bottom_models = []
        for party in party_data:
            model = BottomModel(len(party['features']), hidden_dim, embedding_dim).to(device)
            self.bottom_models.append(model)
        
        self.top_model = TopModel(embedding_dim * self.n_parties, hidden_dim, 1).to(device)
        
    def forward(self, X_parties):
        embeddings = [model(X) for model, X in zip(self.bottom_models, X_parties)]
        combined = torch.cat(embeddings, dim=1)
        return self.top_model(combined), embeddings
    
    def get_all_parameters(self):
        params = []
        for m in self.bottom_models:
            params.extend(list(m.parameters()))
        params.extend(list(self.top_model.parameters()))
        return params
    
    def state_dict(self):
        return {
            'top_model': self.top_model.state_dict(),
            'bottom_models': [m.state_dict() for m in self.bottom_models]
        }
    
    def load_state_dict(self, state):
        self.top_model.load_state_dict(state['top_model'])
        for i, s in enumerate(state['bottom_models']):
            self.bottom_models[i].load_state_dict(s)

# %% [markdown]
# ---
# ## 5. VFL Training Framework

# %%
class VFLTrainer:
    def __init__(self, vfl_model, learning_rate=0.001):
        self.model = vfl_model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.get_all_parameters(), lr=learning_rate)
        self.party_data = vfl_model.party_data
        
    def prepare_data(self, party_data, test_size=0.2):
        X_list = [party['X'].values.astype(np.float32) for party in party_data]
        y = next(p['y'].values.astype(np.float32) for p in party_data if p['is_active'])
        
        scalers, X_scaled = [], []
        for X in X_list:
            scaler = StandardScaler()
            X_scaled.append(scaler.fit_transform(X))
            scalers.append(scaler)
        
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        X_train = [torch.FloatTensor(X[train_idx]).to(self.model.device) for X in X_scaled]
        X_test = [torch.FloatTensor(X[test_idx]).to(self.model.device) for X in X_scaled]
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1).to(self.model.device)
        y_test = torch.FloatTensor(y[test_idx]).unsqueeze(1).to(self.model.device)
        
        return X_train, X_test, y_train, y_test, scalers
    
    def train_epoch(self, X_train, y_train, batch_size=256):
        n_samples = y_train.shape[0]
        indices = np.random.permutation(n_samples)
        total_loss, n_batches = 0, 0
        
        for m in self.model.bottom_models: m.train()
        self.model.top_model.train()
        
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:min(start + batch_size, n_samples)]
            X_batch = [X[batch_idx] for X in X_train]
            y_batch = y_train[batch_idx]
            
            self.optimizer.zero_grad()
            output, _ = self.model.forward(X_batch)
            loss = self.criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self, X_test, y_test):
        for m in self.model.bottom_models: m.eval()
        self.model.top_model.eval()
        
        with torch.no_grad():
            output, _ = self.model.forward(X_test)
            y_true = y_test.cpu().numpy()
            y_pred = (output > 0.5).float().cpu().numpy()
            y_prob = output.cpu().numpy()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob)
        }
    
    def train(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=256, verbose=True):
        for epoch in range(epochs):
            loss = self.train_epoch(X_train, y_train, batch_size)
            if verbose and (epoch + 1) % 10 == 0:
                metrics = self.evaluate(X_test, y_test)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {metrics['accuracy']:.4f} - AUC: {metrics['auc']:.4f}")

# %% [markdown]
# ---
# ## 6. Train Models

# %%
print("Training on UCI Adult...")
vfl_adult = VFLModel(adult_parties, embedding_dim=32, hidden_dim=64, device=device)
trainer_adult = VFLTrainer(vfl_adult, learning_rate=0.001)
X_train_adult, X_test_adult, y_train_adult, y_test_adult, _ = trainer_adult.prepare_data(adult_parties)
trainer_adult.train(X_train_adult, X_test_adult, y_train_adult, y_test_adult, epochs=50)

# %% [markdown]
# ---
# ## 7. VFL Unlearning Methods

# %%
class VFLUnlearner:
    def __init__(self, vfl_model, trainer, device='cpu'):
        self.model = vfl_model
        self.trainer = trainer
        self.device = device
        
    def gradient_ascent_unlearn(self, X_train, y_train, unlearn_indices, epochs=10, lr=0.001):
        print(f"Unlearning {len(unlearn_indices)} samples via Gradient Ascent...")
        
        X_unlearn = [X[unlearn_indices] for X in X_train]
        y_unlearn = y_train[unlearn_indices]
        
        optimizer = optim.Adam(self.model.get_all_parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for m in self.model.bottom_models: m.train()
        self.model.top_model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self.model.forward(X_unlearn)
            loss = criterion(output, y_unlearn)
            (-loss).backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    def fine_tune_unlearn(self, X_train, y_train, unlearn_indices, epochs=20, lr=0.0005):
        print(f"Unlearning {len(unlearn_indices)} samples via Fine-tuning...")
        
        remain_idx = np.setdiff1d(np.arange(y_train.shape[0]), unlearn_indices)
        X_remain = [X[remain_idx] for X in X_train]
        y_remain = y_train[remain_idx]
        
        optimizer = optim.Adam(self.model.get_all_parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            for m in self.model.bottom_models: m.train()
            self.model.top_model.train()
            
            indices = np.random.permutation(y_remain.shape[0])
            total_loss, n = 0, 0
            
            for start in range(0, y_remain.shape[0], 256):
                batch_idx = indices[start:min(start + 256, y_remain.shape[0])]
                X_batch = [X[batch_idx] for X in X_remain]
                y_batch = y_remain[batch_idx]
                
                optimizer.zero_grad()
                output, _ = self.model.forward(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/n:.4f}")

# %% [markdown]
# ---
# ## 8. Membership Inference Attack

# %%
class MembershipInferenceAttack:
    def __init__(self, vfl_model, device='cpu'):
        self.model = vfl_model
        self.device = device
        self.attack_model = None
        
    def extract_features(self, X_parties, y):
        for m in self.model.bottom_models: m.eval()
        self.model.top_model.eval()
        
        with torch.no_grad():
            output, _ = self.model.forward(X_parties)
            probs = output.squeeze().cpu().numpy()
            y_np = y.cpu().numpy().squeeze()
            
            confidence = np.abs(probs - 0.5)
            entropy = -probs * np.log(probs + 1e-10) - (1-probs) * np.log(1-probs + 1e-10)
            pred_correct = ((probs > 0.5) == y_np).astype(float)
            loss = -y_np * np.log(probs + 1e-10) - (1-y_np) * np.log(1-probs + 1e-10)
            
        return np.column_stack([probs, confidence, entropy, pred_correct, loss])
    
    def train_attack_model(self, X_train_parties, y_train, X_test_parties, y_test):
        train_features = self.extract_features(X_train_parties, y_train)
        test_features = self.extract_features(X_test_parties, y_test)
        
        X_attack = np.vstack([train_features, test_features])
        y_attack = np.array([1] * len(train_features) + [0] * len(test_features))
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_attack, y_attack, test_size=0.3, random_state=42)
        
        self.attack_model = LogisticRegression(max_iter=1000)
        self.attack_model.fit(X_tr, y_tr)
        
        return {
            'train_auc': roc_auc_score(y_tr, self.attack_model.predict_proba(X_tr)[:, 1]),
            'test_auc': roc_auc_score(y_te, self.attack_model.predict_proba(X_te)[:, 1])
        }
    
    def evaluate_unlearning(self, X_unlearn_parties, y_unlearn):
        features = self.extract_features(X_unlearn_parties, y_unlearn)
        preds = self.attack_model.predict(features)
        probs = self.attack_model.predict_proba(features)[:, 1]
        return {'member_rate': preds.mean(), 'avg_prob': probs.mean()}

# %% [markdown]
# ---
# ## 9. Run Unlearning Experiments

# %%
def run_unlearning_experiment(trainer, X_train, X_test, y_train, y_test, 
                              unlearn_ratio=0.05, method='gradient_ascent'):
    print("=" * 60)
    print(f"UNLEARNING EXPERIMENT ({method})")
    print("=" * 60)
    
    n_unlearn = int(y_train.shape[0] * unlearn_ratio)
    unlearn_idx = np.random.choice(y_train.shape[0], n_unlearn, replace=False)
    print(f"Samples to unlearn: {n_unlearn}")
    
    # MIA before
    mia = MembershipInferenceAttack(trainer.model, device)
    mia_results = mia.train_attack_model(X_train, y_train, X_test, y_test)
    print(f"MIA AUC (before): {mia_results['test_auc']:.4f}")
    
    X_unlearn = [X[unlearn_idx] for X in X_train]
    y_unlearn = y_train[unlearn_idx]
    
    before = mia.evaluate_unlearning(X_unlearn, y_unlearn)
    print(f"Member detection (before): {before['member_rate']*100:.1f}%")
    
    # Unlearn
    unlearner = VFLUnlearner(trainer.model, trainer, device)
    if method == 'gradient_ascent':
        unlearner.gradient_ascent_unlearn(X_train, y_train, unlearn_idx, epochs=10)
    else:
        unlearner.fine_tune_unlearn(X_train, y_train, unlearn_idx, epochs=20)
    
    # Evaluate after
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Model accuracy after: {metrics['accuracy']:.4f}")
    
    after = mia.evaluate_unlearning(X_unlearn, y_unlearn)
    print(f"Member detection (after): {after['member_rate']*100:.1f}%")
    print(f"Change: {(before['member_rate'] - after['member_rate'])*100:+.1f}%")

# %%
run_unlearning_experiment(trainer_adult, X_train_adult, X_test_adult, 
                          y_train_adult, y_test_adult, 0.05, 'gradient_ascent')

# %% [markdown]
# ---
# ## 10. Advanced Verification Attacks
# 
# ### 10.1 Shadow Model Attack

# %%
class ShadowModelAttack:
    def __init__(self, party_data, embedding_dim=32, hidden_dim=64, device='cpu'):
        self.party_data = party_data
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.shadow_models_retain = []
        self.shadow_models_full = []
        self.meta_classifier = None
        
    def _create_model(self):
        return VFLModel(self.party_data, self.embedding_dim, self.hidden_dim, self.device)
    
    def _extract_features(self, model):
        features = []
        for m in model.bottom_models:
            for p in m.parameters():
                if p.requires_grad:
                    d = p.data.cpu().numpy().flatten()
                    features.extend([np.mean(d), np.std(d), np.linalg.norm(d)])
        for p in model.top_model.parameters():
            if p.requires_grad:
                d = p.data.cpu().numpy().flatten()
                features.extend([np.mean(d), np.std(d), np.linalg.norm(d)])
        return np.array(features)
    
    def train_shadow_models(self, X_train, y_train, unlearn_idx, n_models=5, epochs=20):
        retain_idx = np.setdiff1d(np.arange(y_train.shape[0]), unlearn_idx)
        X_retain = [X[retain_idx] for X in X_train]
        y_retain = y_train[retain_idx]
        
        print(f"Training {n_models} shadow models...")
        
        for i in range(n_models):
            model = self._create_model()
            trainer = VFLTrainer(model, 0.001)
            for _ in range(epochs):
                trainer.train_epoch(X_retain, y_retain)
            self.shadow_models_retain.append(model)
            print(f"  Retain model {i+1} done")
        
        for i in range(n_models):
            model = self._create_model()
            trainer = VFLTrainer(model, 0.001)
            for _ in range(epochs):
                trainer.train_epoch(X_train, y_train)
            self.shadow_models_full.append(model)
            print(f"  Full model {i+1} done")
    
    def train_meta_classifier(self):
        feat_retain = [self._extract_features(m) for m in self.shadow_models_retain]
        feat_full = [self._extract_features(m) for m in self.shadow_models_full]
        
        X = np.nan_to_num(np.vstack([feat_retain, feat_full]))
        y = np.array([0]*len(feat_retain) + [1]*len(feat_full))
        
        self.meta_scaler = StandardScaler()
        X_scaled = self.meta_scaler.fit_transform(X)
        
        self.meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_classifier.fit(X_scaled, y)
        print(f"Meta-classifier accuracy: {accuracy_score(y, self.meta_classifier.predict(X_scaled)):.4f}")
    
    def attack_model(self, target):
        feat = np.nan_to_num(self._extract_features(target).reshape(1, -1))
        feat_scaled = self.meta_scaler.transform(feat)
        prob = self.meta_classifier.predict_proba(feat_scaled)[0][1]
        status = "GOOD" if prob < 0.5 else "POOR"
        print(f"P(full data): {prob:.4f} - {status}")
        return prob

# %%
# Shadow Model Attack demo
print("\\nShadow Model Attack Demo")
print("=" * 40)

n_unlearn = int(y_train_adult.shape[0] * 0.05)
unlearn_idx = np.random.choice(y_train_adult.shape[0], n_unlearn, replace=False)

shadow = ShadowModelAttack(adult_parties, device=device)
shadow.train_shadow_models(X_train_adult, y_train_adult, unlearn_idx, n_models=3, epochs=15)
shadow.train_meta_classifier()

print("\\nAttacking current model:")
shadow.attack_model(trainer_adult.model)

# %% [markdown]
# ---
# ## Summary
# 
# This notebook implements:
# 1. **VFL Framework**: Split learning with bottom/top models
# 2. **Unlearning**: Gradient Ascent (VFU-GA), Fine-tuning (VFU-FT)
# 3. **Verification**: Output MIA, Shadow Model Attack
# 
# **Success Criteria**:
# - MIA AUC ≈ 0.5
# - Shadow P(full) < 0.5

