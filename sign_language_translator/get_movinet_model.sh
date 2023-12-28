# 在 bash 中執行此指令以下載對應的 MoViNet 模型檔案
# bash get_movinet_model.sh

movinet_id="a1"
mode="stream"

wget "https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_${movinet_id}_${mode}.tar.gz" -O "movinet_${movinet_id}_${mode}.tar.gz" -q
tar -xvf "movinet_${movinet_id}_${mode}.tar.gz"