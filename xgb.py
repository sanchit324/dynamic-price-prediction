import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

# Load and preprocess data
data = pd.read_csv("data_n4.csv")

# Convert to categorical type
data['start_pin'] = data['start_pin'].astype('category').cat.codes
data['destination_pin'] = data['destination_pin'].astype('category').cat.codes

# Time-based split
train = data[data['period'] < '2023-07-01']
test = data[data['period'] >= '2023-07-01']

# Feature engineering with time-aware validation
def create_features(df, train_set=None):
    # Route frequency (calculate from training data only)
    if train_set is not None:
        route_freq = train_set.groupby(['start_pin', 'destination_pin']).size().reset_index(name='route_freq')
        df = df.merge(route_freq, on=['start_pin', 'destination_pin'], how='left')
        df['route_freq'] = df['route_freq'].fillna(0)
    else:
        df['route_freq'] = 0  # Default for unseen routes
    
    # Other features
    df['distance_per_ton'] = df['travel_distance'] / (df['Quantity (In TON)'] + 1e-6)
    df['month'] = pd.to_datetime(df['period']).dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    return df

train = create_features(train)
test = create_features(test, train_set=train)

# Feature columns
num_features = ['travel_distance', 'Quantity (In TON)', 'nifty_infra_price',
                'distance_per_ton', 'route_freq', 'month_sin', 'month_cos']
cat_features = ['start_pin', 'destination_pin']
all_features = num_features + cat_features

# Split data
X_train, y_train = train[all_features], train['amount']
X_test, y_test = test[all_features], test['amount']

# %% [markdown]
# ## Approach 1: Native Categorical Support in XGBoost

# Convert categorical features to proper category type
X_train_native = X_train.copy()
X_test_native = X_test.copy()
for col in cat_features:
    X_train_native[col] = X_train_native[col].astype('category')
    X_test_native[col] = X_test_native[col].astype('category')

# Scale numerical features
scaler = StandardScaler()
X_train_native[num_features] = scaler.fit_transform(X_train_native[num_features])
X_test_native[num_features] = scaler.transform(X_test_native[num_features])

# Build and train XGBoost model
xgb_native = XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    enable_categorical=True,
    max_cat_to_onehot=5,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

xgb_native.fit(X_train_native, y_train,
               eval_set=[(X_test_native, y_test)],
               early_stopping_rounds=50,
               verbose=False)

# Evaluate
preds_native = xgb_native.predict(X_test_native)
print("Native Categorical Support Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds_native)):.2f}")
print(f"R²: {r2_score(y_test, preds_native):.4f}\n")

# %% [markdown]
# ## Approach 2: Embedding Layer + XGBoost

# Prepare embedding model
def create_embedding_model(n_start, n_dest, embedding_dim=16):
    start_input = Input(shape=(1,))
    dest_input = Input(shape=(1,))
    
    start_embed = Embedding(n_start + 1, embedding_dim)(start_input)
    dest_embed = Embedding(n_dest + 1, embedding_dim)(dest_input)
    
    start_flat = Flatten()(start_embed)
    dest_flat = Flatten()(dest_embed)
    
    merged = Concatenate()([start_flat, dest_flat])
    output = Dense(1)(merged)
    
    model = Model(inputs=[start_input, dest_input], outputs=output)
    model.compile(loss='mse', optimizer='adam')
    return model

# Get cardinalities
n_start = X_train['start_pin'].max() + 1
n_dest = X_train['destination_pin'].max() + 1

# Train embedding model
embed_model = create_embedding_model(n_start, n_dest)
embed_model.fit(
    [X_train['start_pin'], X_train['destination_pin']],
    y_train,
    epochs=20,
    batch_size=2048,
    validation_split=0.2,
    verbose=0
)

# Extract embeddings
start_embed = embed_model.layers[2].get_weights()[0]
dest_embed = embed_model.layers[3].get_weights()[0]

# Create embedding features
def create_embed_features(df):
    start_emb = start_embed[df['start_pin'].values]
    dest_emb = dest_embed[df['destination_pin'].values]
    return np.hstack([start_emb, dest_emb])

X_train_emb = create_embed_features(X_train)
X_test_emb = create_embed_features(X_test)

# Combine with numerical features
scaler_num = StandardScaler()
num_train = scaler_num.fit_transform(X_train[num_features])
num_test = scaler_num.transform(X_test[num_features])

X_train_combined = np.hstack([num_train, X_train_emb])
X_test_combined = np.hstack([num_test, X_test_emb])

# Train XGBoost on combined features
xgb_emb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

xgb_emb.fit(X_train_combined, y_train,
            eval_set=[(X_test_combined, y_test)],
            early_stopping_rounds=50,
            verbose=False)

# Evaluate
preds_emb = xgb_emb.predict(X_test_combined)
print("Embedding Layer Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds_emb)):.2f}")
print(f"R²: {r2_score(y_test, preds_emb):.4f}")

# Feature importance analysis
print("\nNative Model Feature Importance:")
print(pd.Series(xgb_native.feature_importances_, index=all_features).sort_values(ascending=False))

print("\nEmbedding Model Feature Importance (First 10 features):")
print(pd.Series(xgb_emb.feature_importances_, index=[f"f{i}" for i in range(X_train_combined.shape[1])]).head(10).sort_values(ascending=False))
# %%
