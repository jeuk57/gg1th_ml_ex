# scikit-learn을 활용한 머신러닝 학습

# 가상환경 만들기
```
uv init --bare 
```

# 라이브러리 설치
```
uv add scikit-learn pandas numpy joblib matplotlib searborn
```

# jupyter notebook에 가상환경 추가하기
- ipykernel 설치
```
uv add ipykernel
```
- 가상환경 추가
```
uv run python -m ipykernel install --user --name .venv --display-name "ml_env"
```

# 시각화 한글 깨짐 문제해결 for WSL2 ubuntu
```
# 한글 폰트 설정
import platform

from matplotlib import rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Linux':
    rc('font', family = 'NanumGothic')  # 또는 '나눔고딕'
    print('Linux system... font set to NanumGothic')
elif platform.system() == 'Windows':
    rc('font', family = 'Malgun Gothic')   # 또는 '맑은 고딕'
    print('Windows system... font set to Malgun Gothic')
else:
    print('Unknown system... sorry~~~~')
```

# git 연결하기
- github 리모트 `gg1th_ml_ex` 레포지토리 만들기
- git연결하기
```
git init
git remote add origin "자신의 ssh url"
git branch -M main
git pull origin main
```
- git에 업로딩
```
git add README.md
git commit -m "README.md 파일 업로딩"
git push -u origin main
```
