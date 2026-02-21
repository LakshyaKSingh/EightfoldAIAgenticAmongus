import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """V7: Advanced Feature Engineering targeting specific fraud topologies"""
    df_clean = df.copy()
    
    # 1. Drop intentional noise columns
    noise_cols = [c for c in df_clean.columns if 'noise' in c]
    df_clean = df_clean.drop(columns=noise_cols, errors='ignore')
    
    # 2. Bot & Automation Signal (Fast + High Copy/Paste)
    df_clean['bot_signal'] = df_clean['copy_paste_ratio'] / (df_clean['time_since_last_app_hrs'] + 0.1)
    
    # 3. Application Spike (Sudden burst of activity)
    df_clean['app_spike_7d'] = df_clean['applications_7d'] / (df_clean['applications_30d'] + 0.1)
    
    # 4. Resume Inflation Signal (High claimed skills, low social proof)
    df_clean['inflation_signal'] = (df_clean['skills_count'] * df_clean['skills_to_exp_ratio']) / (df_clean['endorsements_count'] + 1)
    
    # 5. Career Instability (Long gaps, short tenures)
    df_clean['instability_score'] = df_clean['tenure_gap_months'] / (df_clean['avg_tenure_months'] + 0.1)
    
    # 6. Risk Aggregation (Max is often better than Mean for catching extreme anomalies)
    risk_cols = ['ip_risk_score', 'email_risk_score', 'institution_risk_score', 'company_risk_score']
    df_clean['max_risk'] = df_clean[risk_cols].max(axis=1)
    df_clean['mean_risk'] = df_clean[risk_cols].mean(axis=1)
    
    # 7. Core Intersections
    df_clean['trust_score'] = df_clean['endorsements_count'] / (df_clean['skills_count'] + 1)
    df_clean['login_danger_score'] = df_clean['is_new_device'] * (df_clean['failed_logins_24h'] + 1)
    df_clean['velocity_interaction'] = df_clean['applications_7d'] * df_clean['login_velocity_24h']
    
    return df_clean

def run_agent(df: pd.DataFrame, oracle, budget: int) -> np.ndarray:
    print("🚀 Initializing Pipeline: Advanced Feature Topology & Active Learning...")
    
    queries_used = 0
    labeled_data = {} 

    def query_oracle_safe(indices):
        nonlocal queries_used
        new_indices = [int(idx) for idx in indices if int(idx) not in labeled_data]
        remaining = budget - queries_used
        if remaining <= 0 or not new_indices:
            return
            
        indices_to_query = new_indices[:remaining]
        labels = oracle(indices_to_query)
        
        for idx, label in zip(indices_to_query, labels):
            labeled_data[idx] = label
        queries_used += len(indices_to_query)

    # --- 1. PREPARATION ---
    df_eng = engineer_features(df)
    X_raw = df_eng.drop(columns=['Label'], errors='ignore')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # --- 2. PHASE 1: CENTROID COLD START ---
    print("🔍 Phase 1: Representative Cold Start Exploration...")
    # Increased clusters to 20 to map the new, higher-dimensional feature space
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    cold_start_idx = []
    for i in range(20):
        cluster_points = np.where(clusters == i)[0]
        if len(cluster_points) > 0:
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_scaled[cluster_points] - center, axis=1)
            closest_idx = cluster_points[np.argmin(distances)]
            cold_start_idx.append(closest_idx)
            
    # Add 10 purely random points (Total = 30 initial queries)
    np.random.seed(42)
    cold_start_idx.extend(np.random.choice(len(df), size=10, replace=False))
    query_oracle_safe(cold_start_idx) 

    # --- 3. PHASE 2: ACTIVE LEARNING LOOP ---
    print("🧠 Phase 2: Active Learning Loop (Uncertainty Sampling)...")
    # Depth 5 allows the tree to understand the complex new features
    model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    
    while queries_used < budget:
        labeled_idx = list(labeled_data.keys())
        y_train = list(labeled_data.values())
        X_train = X_scaled[labeled_idx]
        
        if len(set(y_train)) < 2:
            unlabeled = np.setdiff1d(np.arange(len(df)), labeled_idx)
            query_oracle_safe(np.random.choice(unlabeled, size=10, replace=False))
            continue
            
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        unlabeled_idx = np.setdiff1d(np.arange(len(df)), labeled_idx)
        if len(unlabeled_idx) == 0: 
            break
            
        uncertainty = np.abs(probs[unlabeled_idx] - 0.5)
        batch_size = min(14, budget - queries_used) 
        most_uncertain_local_idx = np.argsort(uncertainty)[:batch_size]
        query_oracle_safe(unlabeled_idx[most_uncertain_local_idx])

    # --- 4. PHASE 3: BALANCED PSEUDO-LABELING ---
    print("⚖️ Phase 3: Class-Balanced Pseudo-Labeling...")
    labeled_idx = list(labeled_data.keys())
    y_train = list(labeled_data.values())
    X_train = X_scaled[labeled_idx]
    
    model.fit(X_train, y_train)
    final_probs = model.predict_proba(X_scaled)[:, 1]
    unlabeled_idx = np.setdiff1d(np.arange(len(df)), labeled_idx)
    
    # Confidence gates for pseudo-labels
    confident_1_idx = unlabeled_idx[final_probs[unlabeled_idx] > 0.80]
    confident_0_idx = unlabeled_idx[final_probs[unlabeled_idx] < 0.20]
    
    # Restored 3:1 ratio to protect Precision
    num_1s = len(confident_1_idx)
    num_0s_to_keep = min(len(confident_0_idx), num_1s * 3) 
    
    np.random.seed(42)
    if len(confident_0_idx) > num_0s_to_keep:
        confident_0_idx = np.random.choice(confident_0_idx, size=num_0s_to_keep, replace=False)
        
    pseudo_X = []
    pseudo_y = []
    for idx in confident_1_idx:
        pseudo_X.append(X_scaled[idx])
        pseudo_y.append(1)
    for idx in confident_0_idx:
        pseudo_X.append(X_scaled[idx])
        pseudo_y.append(0)

    # --- 5. PHASE 4: FINAL PREDICTION ---
    print("🏆 Phase 4: Final Prediction...")
    if pseudo_X:
        X_final_train = np.vstack((X_train, pseudo_X))
        y_final_train = y_train + pseudo_y
    else:
        X_final_train = X_train
        y_final_train = y_train
        
    final_model = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', random_state=42)
    
    if len(set(y_final_train)) >= 2:
        final_model.fit(X_final_train, y_final_train)
        
        raw_probs = final_model.predict_proba(X_scaled)[:, 1]
        
        # 0.35 is the sweet spot mathematically based on your previous runs
        predictions = (raw_probs > 0.35).astype(int)
    else:
        predictions = np.zeros(len(df))
        
    return predictions