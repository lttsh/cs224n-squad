model=$1
experiment=$2

python code/main.py --mode=official_eval --json_in_path=data/tiny-dev.json \
--ckpt_load_dir=experiments/$experiment/best_checkpoint --model=$model

 echo "Evaluating on tiny dev set.\n"
python code/evaluate.py data/tiny-dev.json predictions.json

echo "Evaluating on entier dev set.\n"
python code/main.py --mode=official_eval --json_in_path=data/dev-v1.1.json \
--ckpt_load_dir=experiments/$experiment/best_checkpoint --model=$model

python code/evaluate.py data/dev-v1.1.json predictions.json
