import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 데이터 로드 (CSV 파일 경로에 맞게 수정)
data = pd.read_csv('s_22_3_z_3.csv')

# 특징과 라벨 분리
X = data.drop(columns=['label'])  # 'label' 컬럼을 제외한 나머지 컬럼이 특징
y = data['label']  # 'label' 컬럼이 라벨

# 8:2 비율로 학습 데이터와 테스트 데이터 분리 (랜덤 분할)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 초기화 (랜덤 포레스트 분류기 사용)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
