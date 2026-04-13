import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


print("=== ЗАПУСК УЧАСТНИКА 2: Сравнение по годам обучения ===\n")

df = pd.read_csv('cleaned_all_semesters.csv')

TARGET = 'Q20'      # Общая оценка курса
YEAR_COL = 'Q2'


df = df[df['Semester'].str.contains('2021|2022|2023|2024|2025', na=False)].copy()

OUTPUT_DIR = Path('plots')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Данные загружены: {df.shape[0]:,} строк\n")


print("="*80)
print("ЭТАП 1: Средняя оценка курса (Q20) по годам обучения")

year_stats = df.groupby(YEAR_COL)[TARGET].agg(['mean', 'median', 'count', 'std']).round(3)
year_stats.index.name = 'Год обучения (Q2)'
year_stats.columns = ['Средняя оценка', 'Медиана', 'Кол-во ответов', 'Стд. отклонение']
print(year_stats)

# График
plt.figure(figsize=(10, 6))
sns.barplot(x=year_stats.index, y=year_stats['Средняя оценка'], palette="Blues_d")
plt.title('Средняя оценка курса (Q20) по годам обучения', fontsize=14)
plt.xlabel('Год обучения (Q2)')
plt.ylabel('Средняя оценка')
for i, v in enumerate(year_stats['Средняя оценка']):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=11, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig(OUTPUT_DIR / 'year_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("График сохранён → plots/year_barplot.png\n")


print("="*80)
print("ЭТАП 2: Сравнение по ключевым вопросам")

key_questions = ['Q20', 'Q27', 'Q22', 'Q24', 'Q23']

comparison = df.groupby(YEAR_COL)[key_questions].mean().round(3)
comparison.index.name = 'Год обучения (Q2)'
print(comparison)


print("\n" + "="*80)
print("ЭТАП 3: Построение модели Random Forest")

df_model = df.dropna(subset=[TARGET]).copy()
feature_cols = [col for col in df_model.columns if col.startswith('Q') and col not in [YEAR_COL, TARGET]]

X = df_model[feature_cols + [YEAR_COL]]
y = df_model[TARGET]

print(f"Строк для модели: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=400, max_depth=9, min_samples_split=5,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"MAE  = {mae:.3f}")
print(f"R²   = {r2:.3f}")

# Feature Importance
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nТоп-10 самых важных признаков:")
print(fi.head(10))

# График Feature Importance
top_fi = fi.head(8)
plt.figure(figsize=(11, 7))
sns.barplot(x=top_fi.values, y=top_fi.index, palette="Blues_d")
plt.title('Топ-8 самых важных факторов для оценки курса (Q20)', fontsize=14)
plt.xlabel('Важность признака')
for i, v in enumerate(top_fi.values):
    plt.text(v + 0.008, i, f"{v:.3f}", va='center', fontsize=11, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance_top8.png', dpi=300, bbox_inches='tight')
plt.close()
print("График Feature Importance сохранён → plots/feature_importance_top8.png")


print("\n" + "="*80)
print("ЭТАП 4: Анализ студентов 5-го года (магистранты)")

stats_5 = df[df[YEAR_COL] == 6][key_questions].mean().round(3)
stats_others = df[df[YEAR_COL] != 6][key_questions].mean().round(3)
diff = (stats_others - stats_5).round(3)

print("Сравнение 5-го года vs остальные:")
print(pd.DataFrame({'5-й год': stats_5, 'Остальные годы': stats_others, 'Разница': diff}))

print("\n" + "="*80)
print("ЭТАП 5: Сохранение всех результатов в Excel")

with pd.ExcelWriter('participant2_full_results.xlsx') as writer:
    year_stats.to_excel(writer, sheet_name='By_Year_Stats')
    comparison.to_excel(writer, sheet_name='Key_Questions_by_Year')
    fi.to_excel(writer, sheet_name='Feature_Importance')
    pd.DataFrame({'5-й год': stats_5, 'Остальные': stats_others, 'Разница': diff}).to_excel(writer, sheet_name='5th_Year_Analysis')

print("\n ВСЁ ГОТОВО!")

