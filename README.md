# 🏆 Farm to Feed Shopping Basket Recommendation

[![Zindi](https://img.shields.io/badge/Zindi-7th%20Place-blue?style=for-the-badge)](https://zindi.africa/competitions/farm-to-feed-shopping-basket-recommendation-challenge)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-orange?style=for-the-badge)](https://lightgbm.readthedocs.io/)

<p align="center">
  <img src="https://img.shields.io/badge/Public%20LB-0.9801-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Private%20LB-0.9708-success?style=flat-square"/>
  
</p>

## 🎯 The Challenge

**Farm to Feed** connects Kenyan farmers with commercial kitchens to sell "odd-looking" but perfectly good produce, reducing food waste. They needed a system to predict:

| Question | Task | Metric |
|----------|------|--------|
| Will they buy? | Binary Classification | AUC (50%) |
| How much? | Quantity Regression | MAE (50%) |

**The catch:** Only **2%** of potential customer-product pairs result in purchases weekly—extreme sparsity.

---

## 💡 The Breakthrough Insight

> **"Will they buy?" and "How much?" require different information to answer well.**

I built **two specialized experts** instead of one confused model:

| Expert | Features | Model | Focus |
|--------|----------|-------|-------|
| **Classifier** | 235 clean features (intent signals) | LightGBM (3500 trees) | **AUC** |
| **Regressor** | 271 features (loyalty + intent) | LightGBM (1500 trees) | **MAE** |

**Key innovation:** Loyalty features (consistency, cadence, seasonality) are **gold for quantity prediction** but **noise for purchase prediction**.

```
🎯 Classifier Features (clean set)          📦 Regressor Features (+ loyalty)
├── Temporal patterns                        ├── ALL classifier features
├── Purchase frequency                       ├── 📊 Quantity consistency (CV)
├── Recency                                   ├── 📅 Seasonal baselines
├── Product popularity                        ├── ⏱️ Purchase cadence
└── Price dynamics                            ├── 💰 Price elasticity
                                               └── 🎯 Confidence scoring
```

---

## 🔧 Feature Engineering Highlights

### Multi-Window History (2W, 4W, 8W, 12W, 16W, 24W)

```
Customer-Product Level    Product Level          Customer Level
├── Purchase freq         ├── Popularity         ├── Activity freq
├── Qty stats             ├── Unique buyers      ├── Product variety
├── Spend patterns        └── Trends             └── Recency
└── Recency
```

### Cyclical Time Encoding
```python
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)  # Weeks wrap around
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)  # Dec → Jan is close
```

### Loyalty Metrics (Regressor-Only)
```python
consistency = 1 - (std_qty / mean_qty)           # Stable = predictable
cadence = 1 - (std_ipd / mean_ipd)               # Regular timing
seasonal = customer_product_month_median         # "Buy more in December"
```

### Advanced Features
- **SVD embeddings**: Collaborative filtering "customers like you bought this"
- **Purchase momentum**: Is behavior increasing or decreasing?
- **Behavioral clusters**: "Weekly bulk buyers" vs "monthly shoppers"

---

## 🤖 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    5-FOLD ENSEMBLE (20 models)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Fold 1      Fold 2      Fold 3      Fold 4      Fold 5     │
│  Seed 42     Seed 101    Seed 202    Seed 303    Seed 404   │
│                                                              │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐  │
│  │CLF W1│    │CLF W1│    │CLF W1│    │CLF W1│    │CLF W1│  │
│  │CLF W2│    │CLF W2│    │CLF W2│    │CLF W2│    │CLF W2│  │
│  │REG W1│    │REG W1│    │REG W1│    │REG W1│    │REG W1│  │
│  │REG W2│    │REG W2│    │REG W2│    │REG W2│    │REG W2│  │
│  └──────┘    └──────┘    └──────┘    └──────┘    └──────┘  │
│                                                              │
│                    ↓ AVERAGE PREDICTIONS ↓                   │
└─────────────────────────────────────────────────────────────┘
```

### Hyperparameters
| Model | Estimators | LR | Regularization |
|-------|------------|----|----------------|
| Classifier W1 | 3500 | 0.008 | α=0.6, λ=0.6 |
| Classifier W2 | 3500 | 0.018 | α=0.3, λ=0.3 |
| Regressor W1 | 1500 | 0.012 | α=1.2, λ=1.2 |
| Regressor W2 | 1500 | 0.018 | α=2.2, λ=2.2 |

---

## 🔄 Post-Processing: Adaptive Power Scaling

**Problem**: Raw quantity predictions overconfident when purchase probability low.

**Solution**: Scale by purchase probability with adaptive exponent

```python
def scale(prob):
    if prob < 0.01: return 1.8    # Very rare → aggressive scaling
    if prob < 0.1:  return 1.5    # Unlikely → strong scaling
    if prob < 0.3:  return 1.25   # Possible → moderate
    if prob < 0.5:  return 1.1    # Plausible → slight
    return 1.05                    # Likely → minimal

final_qty = (prob ** scale) * predicted_qty
```

**Safety**: Clip to `[0, 1.5 × max_historical]`

---

## 📊 Results

| Metric | Public | Private | Δ |
|--------|--------|---------|---|
| **Overall (Weighted)** | **0.98013** | **0.97084** | -0.9% |
| W1 Purchase AUC | 0.96632 | 0.96695 | +0.1% |
| W2 Purchase AUC | 0.96604 | 0.96602 | ±0.0% |
| W1 Quantity MAE | 0.3158 | 1.2375 | +3.9× |
| W2 Quantity MAE | 0.5334 | 2.3525 | +4.5× |

**🏆 Final Rank: 7th on Private Leaderboard**

### Key Takeaways
- **96.6%+ AUC** → Excellent at ranking buyers above non-buyers
- **Minimal AUC degradation** → No overfitting to public LB
- **MAE gap** → Room for improvement in quantity prediction

---

## ⚙️ Technical Specs

### Training
- **CPU**: 32 cores (3 hours) vs 1 core (96 hours)
- **RAM**: 100-110 GB
- **Models**: 20 total (4 models × 5 folds)
- **Features**: 235-271 per model

### Inference
- **CPU**: 2-4 cores
- **RAM**: 2-4 GB
- **Time**: <5 minutes for 275K predictions
- **Throughput**: ~900 predictions/second

---

## 💼 Business Impact

| Stakeholder | Benefit |
|-------------|---------|
| **Farmers** | Demand forecasts → planned harvesting, reduced waste |
| **Farm to Feed** | 15-20% waste reduction, 96.6% accurate targeting |
| **Customers** | Reliable availability, better prices |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/keystats/farm-to-feed-recommendation.git

# Install
pip install -r requirements.txt

# Run
jupyter notebook farm_to_feed_solution.ipynb
```

**Dependencies**: `pandas`, `numpy`, `lightgbm`, `scikit-learn`

---

## 📈 Future Improvements

- **Deep learning** for richer embeddings
- **External data** (weather, holidays, economic indicators)
- **Multi-task learning** for probability-quantity dependencies
- **Cold start handling** for new customers/products

---

<p align="center">
  <strong>🏆 7th Place Solution | Farm to Feed Challenge | Zindi</strong><br>
  <a href="https://zindi.africa/users/keystats">@keystats</a> • 
  <a href="https://github.com/keystats/farm-to-feed-recommendation">GitHub</a> • 
  <a href="https://zindi.africa/competitions/farm-to-feed-shopping-basket-recommendation-challenge">Challenge Page</a>
</p>
