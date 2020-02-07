export PYTHONUNBUFFERED="True"
LOG="../log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python cpm_train.py --gpu 0 1 --train_dir ../dataset/LSP/lspet_dataset --val_dir ../dataset/LSP/lsp_dataset --config ../config/config.yml > $LOG
