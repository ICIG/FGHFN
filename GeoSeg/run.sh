#!/bin/bash

echo "请选择要运行的操作："
echo "1. Training - iSAID"
echo "2. Training - Potsdam"
echo "3. Training - Vaihingen"
echo "4. Training - LoveDA"
echo "5. Testing - iSAID"
echo "6. Testing - Vaihingen"
echo "7. Testing - Potsdam"
echo "8. Testing - LoveDA (输出RGB图像)"
echo "9. Testing - LoveDA (输出标签图像)"

# 读取用户选项
read -rp "请输入对应的数字选项: " choice

# 定义变量
config_path=""
output_path=""
additional_args=""

# 根据用户选择设置命令
case $choice in
  1)
    echo "选择了Training - iSAID"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    config_path="./config/isaid/$config"
    cmd="python train_supervision.py -c $config_path"
    ;;
  2)
    echo "选择了Training - Potsdam"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    config_path="./config/potsdam/$config"
    cmd="python train_supervision_dp.py -c $config_path"
    ;;
  3)
    echo "选择了Training - Vaihingen"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    config_path="./config/vaihingen/$config"
    cmd="python train_supervision_dp.py -c $config_path"
    ;;
  4)
    echo "选择了Training - LoveDA"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    config_path="./config/loveda/$config"
    cmd="python train_supervision_dp.py -c $config_path"
    ;;
  5)
    echo "选择了Testing - iSAID"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    read -rp "请输入输出目录名 (如：convlsrnet_isaid): " output
    output=${output:-convlsrnet_isaid}
    config_path="./config/isaid/$config"
    output_path="./fig_results/isaid/$output/"
    additional_args='-t "d4"'
    cmd="python test_isaid.py -c $config_path -o $output_path $additional_args"
    ;;
  6)
    echo "选择了Testing - Vaihingen"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    read -rp "请输入输出目录名 (如：convlsrnet_vaihingen): " output
    output=${output:-convlsrnet_vaihingen}
    config_path="./config/vaihingen/$config"
    output_path="./fig_results/$output/"
    additional_args='--rgb -t "d4"'
    cmd="python test_vaihingen.py -c $config_path -o $output_path $additional_args"
    ;;
  7)
    echo "选择了Testing - Potsdam"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    read -rp "请输入输出目录名 (如：convlsrnet_potsdam): " output
    output=${output:-convlsrnet_potsdam}
    config_path="./config/potsdam/$config"
    output_path="./fig_results/$output/"
    additional_args='--rgb -t "d4"'
    cmd="python test_potsdam.py -c $config_path -o $output_path $additional_args"
    ;;
  8)
    echo "选择了Testing - LoveDA (输出RGB图像)"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    read -rp "请输入输出目录名 (如：convlsrnet_loveda_rgb): " output
    output=${output:-convlsrnet_loveda_rgb}
    config_path="./config/loveda/$config"
    output_path="./fig_results/$output"
    additional_args='--rgb --val -t "d4"'
    cmd="python test_loveda.py -c $config_path -o $output_path $additional_args"
    ;;
  9)
    echo "选择了Testing - LoveDA (输出标签图像)"
    read -rp "请输入配置文件名 (如：convlsrnet.py): " config
    config=${config:-convlsrnet.py}
    read -rp "请输入输出目录名 (如：convlsrnet_loveda_onlinetest): " output
    output=${output:-convlsrnet_loveda_onlinetest}
    config_path="./config/loveda/$config"
    output_path="./fig_results/$output"
    additional_args='-t "d4"'
    cmd="python test_loveda.py -c $config_path -o $output_path $additional_args"
    ;;
  *)
    echo "无效选项，请重新运行脚本并输入有效数字。"
    exit 1
    ;;
esac

# 执行命令
echo "运行命令: $cmd"
eval "$cmd"

# 钉钉通知
webhook='https://oapi.dingtalk.com/robot/send?access_token=填入自己的Token'
text="命令已执行完毕！执行命令：$cmd"

# 发送钉钉通知
curl $webhook -H 'Content-Type: application/json' -d "
{
    'msgtype': 'text',
    'text': {
        'content': '$text',
    }
}"
