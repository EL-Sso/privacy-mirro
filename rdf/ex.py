import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy.fft import fft


def load_and_extract_features(folder_path, file_filter=None):
    all_features = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and (file_filter is None or file_filter in file_name):
            print(f"Selected file: {file_name}")
            file_path = os.path.join(folder_path, file_name)
            try:
                data = pd.read_csv(file_path)
                print(f"Processing {file_name}, shape: {data.shape}")
                features = extract_features(data)
                if not features.empty:
                    all_features.append(features)
                else:
                    print(f"No valid features extracted from {file_name}")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        print(f"Final combined features shape: {combined_features.shape}")
        return combined_features
    else:
        print("No valid features extracted from any file.")
        return pd.DataFrame()


def extract_features(data):
    """
    센서 데이터로부터 시간 및 주파수 특징을 추출합니다.
    """
    features = {}
    for col in ['acceleration.x', 'acceleration.y', 'acceleration.z',
                'rotationRate.alpha', 'rotationRate.beta', 'rotationRate.gamma']:
        if col in data.columns:
            # 시간 영역 특징
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_max'] = data[col].max()
            features[f'{col}_min'] = data[col].min()
            features[f'{col}_var'] = data[col].var()

            # 주파수 영역 특징
            freq_features = extract_frequency_features(data[col])
            for k, v in freq_features.items():
                features[f'{col}_{k}'] = v
        else:
            print(f"Column {col} is missing in the data.")

    return pd.DataFrame([features]) if features else pd.DataFrame()


def extract_frequency_features(column_data):
    """
    주파수 영역에서 특징 추출: FFT를 활용하여 평균, 최대값, 최소값 등 계산.
    """
    try:
        # pandas Series를 numpy 배열로 변환
        column_data = column_data.dropna().to_numpy(dtype=np.float64)

        # FFT 수행
        fft_values = np.abs(fft(column_data))  # FFT 후 절댓값
        if len(fft_values) == 0:
            return {}

        # 주파수 영역 특징 계산
        freq_features = {
            'freq_mean': np.mean(fft_values),
            'freq_max': np.max(fft_values),
            'freq_min': np.min(fft_values),
            'freq_var': np.var(fft_values)
        }
        return freq_features
    except Exception as e:
        print(f"Error in FFT computation: {e}")
        return {}


def generate_device_fingerprint(data):
    """
    특징 데이터를 기반으로 디바이스 지문을 생성하고 시각화합니다.
    """
    # 결측값 대체
    imputer = SimpleImputer(strategy='mean')  # 평균으로 대체
    data_imputed = imputer.fit_transform(data)  # 결측값 처리

    # 데이터 스케일링
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_imputed)

    if scaled_data.shape[0] >= 2 and scaled_data.shape[1] >= 2:
        # PCA 실행
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        # 시각화
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.title('Device Fingerprint Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster')
        plt.show()

        return labels
    else:
        print("Insufficient data for PCA. Skipping PCA and clustering.")
        return None


def save_features_to_csv(features, output_file):
    try:
        # 경로가 디렉토리인지 확인하고 파일 경로 수정
        if not output_file.endswith('.csv'):
            output_file += '.csv'

        # 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir) and output_dir != '':
            os.makedirs(output_dir)
            print(f"Directory created: {output_dir}")

        # features가 DataFrame인지 확인
        if isinstance(features, pd.DataFrame) and not features.empty:
            # CSV로 저장
            features.to_csv(output_file, index=False)
            print(f"Features saved to {output_file}")
        else:
            print("No valid data to save to CSV.")
    except Exception as e:
        print(f"Error saving features to CSV: {e}")



def main():
    folder_path = r"C:\Users\lso\rdf\maze_s20\maze_s20_Processed\sineX"
    file_filter = "s20"  # 필터링할 파일명을 지정
    features = load_and_extract_features(folder_path, file_filter)

    if features.empty:
        print("No features were extracted. Exiting program.")
        return

    print("Extracted features:\n", features.head())
    labels = generate_device_fingerprint(features)
    if labels is not None:
        print("Generated device fingerprint labels:\n", labels)

    output_file = r"C:\Users\lso\fp\new_maze_sine_x\maze_s_20_features.csv"
    save_features_to_csv(features, output_file)


if __name__ == "__main__":
    main()
