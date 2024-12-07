import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler  # StandardScaler 임포트
import pickle  # pickle 사용

# 데이터 로드 (CSV 파일 경로에 맞게 수정)
data = pd.read_csv('all_labelling.csv')

# 특징과 라벨 분리
X = data.drop(columns=['label'])  # 'label' 컬럼을 제외한 나머지 컬럼이 특징
y = data['label']  # 'label' 컬럼이 라벨

# 데이터 스케일링 (StandardScaler 사용)
scaler = StandardScaler()  # 스케일러 초기화
X_scaled = scaler.fit_transform(X)  # 데이터 스케일링

# 기기별 데이터 균등 분포를 위해 라벨별 그룹화
grouped_data = pd.DataFrame(X_scaled, columns=X.columns)
grouped_data['label'] = y
grouped_data = grouped_data.groupby('label')

X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

# 각 그룹에서 8:2로 나누기
for label, group in grouped_data:
    group_X = group.drop(columns=['label'])  # 특징 추출
    group_y = group['label']  # 라벨 추출
    
    X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
        group_X, group_y, test_size=0.2, random_state=42
    )
    X_train_list.append(X_train_group)
    X_test_list.append(X_test_group)
    y_train_list.append(y_train_group)
    y_test_list.append(y_test_group)

# 모든 그룹 데이터 합치기
X_train = pd.concat(X_train_list)
X_test = pd.concat(X_test_list)
y_train = pd.concat(y_train_list)
y_test = pd.concat(y_test_list)

# 모델 초기화 (랜덤 포레스트 분류기 사용)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

# 성능 평가 출력
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# 학습된 모델 저장 (pickle 사용)
with open('randomforest_moder.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("모델이 'randomforest_model.pkl' 파일로 저장되었습니다.")

# 스케일러 저장
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("스케일러가 'scaler.pkl' 파일로 저장되었습니다.")

# 예측 결과와 성능 평가 저장
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_csv('randomforest_predictions.csv', index=False)
print("예측 결과가 'randomforest_predictions.csv' 파일로 저장되었습니다.")

# 성능 평가 저장
classification_report_df = pd.DataFrame(classification_report_dict).transpose()
classification_report_df.to_csv('randomforest_classification_report.csv', index=True)
print("Classification Report가 'randomforest_classification_report.csv' 파일로 저장되었습니다.")
