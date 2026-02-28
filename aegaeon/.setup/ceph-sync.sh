#! /bin/bash

THIS_DIR=`realpath $(dirname $0)`
ROOT_DIR=`realpath $THIS_DIR/..`
REPO_NAME=$(basename $ROOT_DIR)

if [ -z "$1" ]; then
    echo "Usage: $0 <user>"
    exit 1
fi
USER_NAME=$1

SOURCE_DIR=$ROOT_DIR
TARGET_DIR="/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_$USER_NAME/$REPO_NAME"

# 首次全量同步
rsync -avh --exclude='.git' --exclude='.venv' --exclude='quick_model_loader' --exclude='profiles/**/*.json' "$SOURCE_DIR"/ "$TARGET_DIR"/

# 监听文件变化事件（创建、修改、删除、移动），检测到变化后等待键盘输入再同步
while inotifywait -r --exclude '/(\.git|\.venv|quick_model_loader)/' -e modify,create,delete,move "$SOURCE_DIR"; do
    echo "检测到文件变化，按回车键开始同步..."
    read -r
    rsync -avh --exclude='.git' --exclude='.venv' --exclude='quick_model_loader' --exclude='profiles/**/*.json' "$SOURCE_DIR"/ "$TARGET_DIR"/
done
