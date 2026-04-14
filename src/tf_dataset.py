import tensorflow as tf
import os

def create_dataset(data_dir, batch_size=32, img_size=(224, 224)):

    print(f"데이터 로딩 경로: {data_dir}")

    # 1. 학습용(Training) 데이터 로드 (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical' # 다중 분류를 위한 원핫 인코딩
    )

    # 2. 검증용(Validation) 데이터 로드 (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    print(f"✅ 인식된 클래스 목록: {class_names}")

    # 3. 데이터 로딩 속도 최적화 (캐싱 및 프리패치)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    import os

    current_dir = os.getcwd()
    print(f"현재 실행 위치: {current_dir}")

    # 2. 절대 경로로 명확하게 타겟팅
    PLANTVILLAGE_DIR = os.path.join(current_dir, "data", "raw", "plantvillage_dataset", "color")
    print(f"파이썬이 찾고 있는 경로: {PLANTVILLAGE_DIR}")

    # 3. 폴더 존재 여부 확인 및 실행
    if os.path.exists(PLANTVILLAGE_DIR):
        print("폴더를 찾았습니다! 데이터 로딩을 시작합니다.\n")
        train_data, val_data, classes = create_dataset(PLANTVILLAGE_DIR)
        print("\n데이터 로더 테스트 성공")
    else:
        print("여전히 폴더를 찾을 수 없습니다. 경로가 미세하게 다를 수 있습니다.")
        print("현재 'data/raw/plantvillage' 폴더 안에는 이런 파일/폴더들이 있습니다:")
        try:
            check_path = os.path.join(current_dir, "data", "raw", "plantvillage")
            print(os.listdir(check_path))
        except Exception as e:
            print(f"상위 폴더조차 접근할 수 없습니다: {e}")
