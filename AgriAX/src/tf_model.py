import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

from tf_dataset import create_dataset

def build_baseline_model(input_shape=(224, 224, 3), num_classes=38):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # 초기 학습 시에는 베이스 모델의 가중치를 동결(Freeze)합니다.
    base_model.trainable = False

    # 프로젝트에 맞는 커스텀 분류층을 추가합니다.
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # 1. 경로 설정 및 데이터 파이프라인 호출
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data", "raw", "plantvillage_dataset", "color")

    print("데이터 파이프라인을 초기화합니다.")
    train_ds, val_ds, class_names = create_dataset(data_dir)
    num_classes = len(class_names)

    # 2. 베이스라인 모델 생성
    print("모델 구조를 빌드합니다.")
    model = build_baseline_model(num_classes=num_classes)
    model.summary()

    # 3. 모델 학습 (MVP 테스트를 위해 에포크는 3으로 제한)
    print("초기 모델 학습을 시작합니다.")
    epochs = 3
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 4. 학습된 모델 가중치 저장
    save_dir = os.path.join(current_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "baseline_model.keras")

    model.save(model_path)
    print(f"모델 학습 및 저장이 완료되었습니다. 저장 경로: {model_path}")
