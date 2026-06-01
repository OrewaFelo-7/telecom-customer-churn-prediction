# ============================================================
#  GCI World 2026 – Company A Telecom Dataset
#  Phase 1: Exploratory Data Analysis (EDA)
#  Author: [Your Omnicampus Name]
# ============================================================

# ── CELL 1: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded.")

# ── CELL 2: Mount Google Drive & Load Data ───────────────────
# If running on Google Colab, uncomment the next 3 lines:
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_PATH = '/content/drive/MyDrive/FinalAssignment/telecom/'

# If running locally:
DATA_PATH = './'   # <-- change to your folder path

client = pd.read_csv(DATA_PATH + 'Client.csv')
record = pd.read_csv(DATA_PATH + 'Record.csv')
df     = pd.merge(client, record, on='Customer_ID')

print(f"Client  : {client.shape[0]:,} rows × {client.shape[1]} columns")
print(f"Record  : {record.shape[0]:,} rows × {record.shape[1]} columns")
print(f"Merged  : {df.shape[0]:,} rows × {df.shape[1]} columns")


# ── CELL 3: Basic Data Overview ──────────────────────────────
print("\n=== DATA TYPES (first 10 columns) ===")
print(df.dtypes.head(10))

print("\n=== BASIC STATS (numeric, sample) ===")
print(df[['rev_Mean', 'mou_Mean', 'months', 'custcare_Mean', 'eqpdays']].describe().round(2))


# ── CELL 4: Target Variable – Churn Distribution ─────────────
print("\n=== CHURN VALUE COUNTS ===")
print(df['churn'].value_counts())
print(f"\nChurn Rate: {df['churn'].mean()*100:.2f}%")
print("NOTE: ~49.6% churn — this is at the TOP of the industry danger zone (20–50%)")


# ── CELL 5: Missing Values Analysis ─────────────────────────
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
print("\n=== MISSING VALUES (% per column) ===")
print(null_pct.round(2))
print("\nColumns with >20% missing (use with caution):")
print(null_pct[null_pct > 20].index.tolist())


# ── CELL 6: Revenue Stats ────────────────────────────────────
avg_arpu    = df['rev_Mean'].mean()
n_churners  = df['churn'].sum()
total       = len(df)

annual_risk = n_churners * avg_arpu * 12
savings     = annual_risk * 0.70 * 0.30

print("\n=== REVENUE AT RISK CALCULATION ===")
print(f"  Total customers        : {total:,}")
print(f"  Churners (30-60 days)  : {n_churners:,}  ({n_churners/total*100:.1f}%)")
print(f"  Avg monthly revenue    : ${avg_arpu:.2f}")
print(f"  Annual revenue at risk : ${annual_risk:,.0f}")
print(f"  Estimated savings      : ${savings:,.0f}")
print(f"  (Assuming 70% model recall, 30% retention rate from targeted offers)")


# ── CELL 7: Set Plot Style ───────────────────────────────────
CHURN_COLOR   = '#E24B4A'   # red    = churned
STAY_COLOR    = '#1D9E75'   # green  = stayed
NEUTRAL_COLOR = '#378ADD'   # blue   = neutral
BG            = '#FAFAF8'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor'  : BG,
    'font.family'     : 'DejaVu Sans',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'grid.color'      : '#CCCCCC',
})
print("Plot style set.")


# ── CELL 8: FIG 1 – Churn Distribution (Donut Chart) ─────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
ax.set_facecolor(BG)

sizes  = [50438, 49562]
colors = [STAY_COLOR, CHURN_COLOR]
wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                   wedgeprops=dict(width=0.55, edgecolor='white', linewidth=3))

ax.text(0,  0.12, '49.6%',    ha='center', va='center', fontsize=30,
        fontweight='bold', color=CHURN_COLOR)
ax.text(0, -0.22, 'Churn Rate', ha='center', va='center', fontsize=13, color='#555555')

patch_stay  = mpatches.Patch(color=STAY_COLOR,  label='Stayed  (50,438)')
patch_churn = mpatches.Patch(color=CHURN_COLOR, label='Churned (49,562)')
ax.legend(handles=[patch_stay, patch_churn], loc='lower center',
          bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=11, frameon=False)

ax.set_title('Company A: Nearly Half of Customers Churned',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('fig1_churn_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# SLIDE NOTE: Use this chart on "Problem Definition" slide.
# Key message: 49.6% churn is at the industry danger zone ceiling.


# ── CELL 9: FIG 2 – Churn Rate by Tenure ────────────────────
df['tenure_group'] = pd.cut(df['months'], bins=[0, 6, 12, 24, 36, 100],
                             labels=['0–6 mo', '7–12 mo', '13–24 mo', '25–36 mo', '37+ mo'])
tenure_data = df.groupby('tenure_group', observed=True)['churn'].mean().reset_index()
tenure_data['churn_pct'] = tenure_data['churn'] * 100

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)

bar_colors = [CHURN_COLOR if v > 50 else STAY_COLOR for v in tenure_data['churn_pct']]
bars = ax.bar(tenure_data['tenure_group'].astype(str), tenure_data['churn_pct'],
              color=bar_colors, edgecolor='white', linewidth=0.8, width=0.6)

for bar, val in zip(bars, tenure_data['churn_pct']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(50, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7, label='50% threshold')
ax.set_ylim(0, 68)
ax.set_xlabel('Tenure (months in service)', fontsize=11)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn Peaks for Mid-Tenure Customers (13–24 months)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig('fig2_churn_by_tenure.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# SLIDE NOTE: Key insight — the 13-24 month window is the DANGER ZONE.
# These are customers who are past the "honeymoon" phase but not yet loyal.
# This is when targeted retention should activate.


# ── CELL 10: FIG 3 – Churn Rate by Equipment Age ─────────────
df['equip_group'] = pd.cut(df['eqpdays'], bins=[0, 180, 365, 730, 99999],
                            labels=['< 6 mo', '6–12 mo', '1–2 yr', '2+ yr'])
equip_data = df.groupby('equip_group', observed=True)['churn'].mean().reset_index()
equip_data['churn_pct'] = equip_data['churn'] * 100

fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
ax.set_facecolor(BG)

bar_colors = [STAY_COLOR if v < 50 else CHURN_COLOR for v in equip_data['churn_pct']]
bars = ax.bar(equip_data['equip_group'].astype(str), equip_data['churn_pct'],
              color=bar_colors, edgecolor='white', linewidth=0.8, width=0.55)

for bar, val in zip(bars, equip_data['churn_pct']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(50, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)
ax.set_ylim(0, 68)
ax.set_xlabel('Age of Current Device (eqpdays)', fontsize=11)
ax.set_ylabel('Churn Rate (%)', fontsize=11)
ax.set_title('Older Devices = Higher Churn\n2+ Year Handsets Show 57.9% Churn Rate',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('fig3_churn_by_equipment_age.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# SLIDE NOTE: Strong business action from this insight:
# Proactively offer device upgrades to customers with 2+ year old equipment.


# ── CELL 11: FIG 4 – Usage Patterns (MOU & change_mou) ───────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)

# Left: MOU distribution
churn_0 = df[df['churn'] == 0]['mou_Mean'].dropna().clip(0, 2000)
churn_1 = df[df['churn'] == 1]['mou_Mean'].dropna().clip(0, 2000)
axes[0].hist(churn_0, bins=60, alpha=0.55, color=STAY_COLOR,  label='Stayed',  density=True)
axes[0].hist(churn_1, bins=60, alpha=0.55, color=CHURN_COLOR, label='Churned', density=True)
axes[0].axvline(churn_0.mean(), color=STAY_COLOR,  linewidth=2, linestyle='--',
                label=f'Stayed avg: {churn_0.mean():.0f}')
axes[0].axvline(churn_1.mean(), color=CHURN_COLOR, linewidth=2, linestyle='--',
                label=f'Churned avg: {churn_1.mean():.0f}')
axes[0].set_xlabel('Avg Monthly Minutes of Use', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Churners Use Fewer Minutes\n(543 vs 483 avg MOU)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, frameon=False)

# Right: change_mou
change_data = df.groupby('churn')['change_mou'].mean()
bars = axes[1].bar(['Stayed', 'Churned'], change_data.values,
                   color=[STAY_COLOR, CHURN_COLOR], edgecolor='white', linewidth=0.8, width=0.45)
for bar, val in zip(bars, change_data.values):
    offset = 2 if val >= 0 else -8
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
axes[1].axhline(0, color='#888888', linewidth=1, alpha=0.7)
axes[1].set_ylabel('Avg Change in MOU vs. Prior 3 Months', fontsize=10)
axes[1].set_title('Churners Show Steep Usage Drop\nBefore Leaving',
                  fontsize=12, fontweight='bold')

plt.suptitle('Usage Patterns: Churners Disengage Before Leaving',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig4_usage_patterns.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# SLIDE NOTE: change_mou is a LEADING INDICATOR of churn.
# A customer who suddenly reduces usage by 22+ minutes is at high risk.
# This is exactly the kind of signal a real-time alert system can catch.


# ── CELL 12: FIG 5 – Top Correlations with Churn ─────────────
num_df = df.select_dtypes(include='number')
corr   = num_df.corr()['churn'].drop('churn').abs().sort_values(ascending=False).head(12)
corr_df = corr.reset_index()
corr_df.columns = ['feature', 'abs_corr']

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(BG)

bar_colors = [CHURN_COLOR if v >= 0.08 else NEUTRAL_COLOR for v in corr_df['abs_corr']]
ax.barh(corr_df['feature'][::-1], corr_df['abs_corr'][::-1],
        color=bar_colors[::-1], edgecolor='white', linewidth=0.8, height=0.6)

for i, (val, feat) in enumerate(zip(corr_df['abs_corr'][::-1], corr_df['feature'][::-1])):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9, color='#444444')

ax.set_xlabel('Absolute Correlation with Churn', fontsize=11)
ax.set_title('Top Predictors of Churn\n(Equipment age & handset price are strongest signals)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 0.15)

patch_high = mpatches.Patch(color=CHURN_COLOR,   label='Strong signal (≥ 0.08)')
patch_low  = mpatches.Patch(color=NEUTRAL_COLOR, label='Moderate signal')
ax.legend(handles=[patch_high, patch_low], fontsize=10, frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig('fig5_top_correlations.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# SLIDE NOTE: eqpdays and hnd_price are the top raw correlates of churn.
# These will be key features in the ML model in Phase 3.


# ── CELL 13: FIG 6 – Revenue Impact Bar Chart ────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)

categories = ['Annual Revenue\nat Risk', 'Estimated Savings\n(70% recall, 30% retention)']
values     = [annual_risk / 1e6, savings / 1e6]
colors_rev = [CHURN_COLOR, STAY_COLOR]

bars = ax.bar(categories, values, color=colors_rev, edgecolor='white',
              linewidth=0.8, width=0.45)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'${val:.1f}M', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax.set_ylabel('USD (Millions)', fontsize=11)
ax.set_title(
    f'Revenue Impact: ${annual_risk/1e6:.1f}M at Risk — ${savings/1e6:.1f}M Recoverable\n'
    f'({n_churners:,} churners × ${avg_arpu:.0f} ARPU × 12 months)',
    fontsize=12, fontweight='bold'
)
ax.set_ylim(0, max(values) * 1.35)

plt.tight_layout()
plt.savefig('fig6_revenue_impact.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# SLIDE NOTE: This is your headline business number.
# Put it big on the "Business Case" slide.


# ── CELL 14: EDA SUMMARY ─────────────────────────────────────
print("\n" + "="*60)
print("  EDA SUMMARY — KEY FINDINGS")
print("="*60)
print(f"""
1. CHURN RATE: {df['churn'].mean()*100:.1f}% — near the worst in the industry.
   → This is NOT normal. Industry average is 20–25%.

2. DANGER ZONE: 13–24 months tenure shows 51.8% churn.
   → Mid-tenure customers are the highest priority for retention.

3. EQUIPMENT AGE: Customers with 2+ yr old devices churn at 57.9%.
   → Device upgrade offers should be a key retention action.

4. USAGE DROP SIGNAL: Churners show avg change_mou of -22.8 vs -5.3.
   → A sudden drop in usage is an early warning signal.

5. TOP CORRELATES: eqpdays (0.113), hnd_price (0.103), totmrc_Mean (0.069).
   → These are the strongest features for the ML model.

6. REVENUE AT RISK: ${annual_risk/1e6:.1f}M/year.
   → With a churn model at 70% recall + 30% retention rate,
     we can recover approximately ${savings/1e6:.1f}M/year.
""")
print("="*60)
print("Next step: Phase 2 — Feature Engineering + Model Building")
