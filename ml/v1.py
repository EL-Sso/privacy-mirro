import pickle
import pandas as pd

# 피클 파일로 저장된 랜덤 포레스트 모델 및 스케일러 불러오기
with open(r'C:\Users\lso\ml\randomforest_moder.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)  # 변수명을 명확히 변경

with open(r'C:\Users\lso\ml\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# 새로운 데이터 로드
csv_file_path = r'C:\Users\lso\ml\z\ZFlip.csv'  # 사용자가 제공한 CSV 파일 경로
new_data = pd.read_csv(csv_file_path)

# 기기 식별에 필요한 feature들만 사용 (전체 사용)
X_new = new_data

# 학습 시 사용한 스케일러로 새로운 데이터를 전처리
X_new_scaled = scaler.transform(X_new)

# 랜덤 포레스트 모델로 예측 수행
predictions = random_forest_model.predict(X_new_scaled)

# 전체 데이터의 예측 결과에서 가장 빈도가 높은 라벨을 선택
predicted_device_label = pd.Series(predictions).mode()[0]

print(f"새로운 데이터에 대해 예측된 기기 라벨: {predicted_device_label}")
