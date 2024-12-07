import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 모델

# 모델 및 스케일러 로드 함수
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded.")
    return model

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    print("Scaler loaded.")
    return scaler

# 모델 정확도 평가 함수
def evaluate_model_accuracy(model, scaler, new_data, labels):
    # 스케일링
    new_data_scaled = scaler.transform(new_data)

    # 예측 수행
    predictions = model.predict(new_data_scaled)

    # 정확도 계산
    accuracy = accuracy_score(labels, predictions)
    print(f"Model accuracy on new data: {accuracy * 100:.2f}%")
    return accuracy

# 모델 재훈련 함수
def retrain_model(model, new_data, labels):
    # 모델을 새로운 데이터로 재훈련
    model.fit(new_data, labels)
    print("Model retrained.")
    return model

# 모델 저장 함수
def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

# 데이터 예측 함수
def predict_device_label(model, scaler, new_data):
    # 스케일링
    new_data_scaled = scaler.transform(new_data)

    # 예측 수행
    predictions = model.predict(new_data_scaled)

    # 가장 빈도가 높은 라벨 선택
    predicted_label = pd.Series(predictions).mode()[0]
    print(f"Predicted device label: {predicted_label}")
    return predicted_label

# 실행
if __name__ == "__main__":
    # 모델 및 스케일러 경로
    model_path = r"C:\Users\lso\ml\randomforest_moder_new.pkl"
    scaler_path = r"C:\Users\lso\ml\scaler.pkl"
    new_data_path = r"C:\Users\lso\ml\s20.csv"  # 새로운 데이터 경로

    # 모델 및 스케일러 로드
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # 새로운 데이터 로드
    new_data = pd.read_csv(new_data_path)

    # 라벨 열이 있더라도 제거하여 전체 데이터를 특징(feature)으로 사용
    if "label" in new_data.columns:
        X_new = new_data.drop(columns=["label"])  # 'label' 열 제거
        labels_new = new_data["label"]  # 'label' 열을 새로운 라벨로 저장
    else:
        X_new = new_data
        labels_new = None  # 라벨이 없는 경우

    # 모델 정확도 평가
    if labels_new is not None:
        accuracy = evaluate_model_accuracy(model, scaler, X_new, labels_new)

        # 정확도가 일정 임계값 이하일 경우 모델 재훈련
        if accuracy < 0.8:  # 예시로 80% 이하의 정확도는 새로운 데이터로 간주
            print("Accuracy is low. Retraining the model with new data.")
            model = retrain_model(model, X_new, labels_new)
            save_model(model, model_path)  # 재훈련된 모델을 저장
        else:
            print("Model accuracy is acceptable.")
    else:
        print("No labels provided. Predictions will be made without accuracy evaluation.")

    # 예측 수행
    predict_device_label(model, scaler, X_new)
