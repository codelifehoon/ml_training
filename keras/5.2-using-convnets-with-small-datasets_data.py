import os, shutil


# 원본 데이터셋을 압축 해제한 디렉터리 경로
original_dataset_dir = './datasets/cats_and_dogs/train'

# 소규모 데이터셋을 저장할 디렉터리
base_dir = './datasets/cats_and_dogs_small'
if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(base_dir)   # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(base_dir)

def mkdir(base_dir,dir):
    new_dir = os.path.join(base_dir, dir)
    os.mkdir(new_dir)
    return new_dir

def copyfiles(fnames,orgdir,targetdir):
    for fname in fnames:
        src = os.path.join(orgdir, fname)
        dst = os.path.join(targetdir, fname)
        shutil.copyfile(src, dst)


# 훈련, 검증, 테스트 분할을 위한 디렉터리
train_dir = mkdir(base_dir, 'train')
validation_dir = mkdir(base_dir, 'validation')
test_dir = mkdir(base_dir, 'test')

# 훈련용 고양이 사진 디렉터리
train_cats_dir = mkdir(train_dir, 'cats')

# 훈련용 강아지 사진 디렉터리
train_dogs_dir = mkdir(train_dir, 'dogs')

# 검증용 고양이 사진 디렉터리
validation_cats_dir = mkdir(validation_dir, 'cats')

# 검증용 강아지 사진 디렉터리
validation_dogs_dir = mkdir(validation_dir, 'dogs')

# 테스트용 고양이 사진 디렉터리
test_cats_dir = mkdir(test_dir, 'cats')

# 테스트용 강아지 사진 디렉터리
test_dogs_dir = mkdir(test_dir, 'dogs')

# 처음 1,000개의 고양이 이미지를 train_cats_dir에 복사합니다
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
copyfiles(fnames,original_dataset_dir,train_cats_dir)


# 다음 500개 고양이 이미지를 validation_cats_dir에 복사합니다
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
copyfiles(fnames,original_dataset_dir,validation_cats_dir)


# 다음 500개 고양이 이미지를 test_cats_dir에 복사합니다
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
copyfiles(fnames,original_dataset_dir,test_cats_dir)


# 처음 1,000개의 강아지 이미지를 train_dogs_dir에 복사합니다
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
copyfiles(fnames,original_dataset_dir,train_dogs_dir)

# 다음 500개 강아지 이미지를 validation_dogs_dir에 복사합니다
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
copyfiles(fnames,original_dataset_dir,validation_dogs_dir)


# 다음 500개 강아지 이미지를 test_dogs_dir에 복사합니다
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
copyfiles(fnames,original_dataset_dir,test_dogs_dir)
“#모니터링_11pay”  채널에서 박장원님이  최근 11시 타임딜에  네퍼넬  대기 발생한다고  11P