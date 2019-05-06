param(
    [string]$model = "seq2seq",
    [string]$task = "dailydialog"
)

$model_dir = checkpoints\$model
$model_file = $model_dir + "\" + $model

mkdir -p checkpoints
mkdir -p $model_dir

python $env:userprofile/ParlAI/parlai/scripts/train_model.py -m $model -t $task -bs 256 --model_file $model_file
