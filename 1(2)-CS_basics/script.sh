
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi


# Conda 환셩 생성 및 활성화
## TODO
if ! conda info --envs | grep -q '^myenv'; then
    conda create -y -n myenv python=3.10
fi
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy

# Submission 폴더 파일 실행
cd submission 

for file in *.py; do
    ## TODO
    filename=$(basename "$file" .py)
    input_file="../input/${filename}_input"
    output_file="../output/${filename}_output"

    if [[ -f "$input_file" ]]; then
        python "$file" < "$input_file" > "$output_file"
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
cd ..
mypy ./submission/*.py > mypy_log.txt

# conda.yml 파일 생성
## TODO
conda env export > conda.yml


# 가상환경 비활성화
## TODO
conda deactivate

