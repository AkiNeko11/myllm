@echo off
chcp 65001 >nul
echo ========================================
echo 角色语言风格微调 - 快速启动脚本
echo ========================================
echo.

:menu
echo 请选择操作：
echo 1. 安装依赖
echo 2. 处理数据
echo 3. 开始训练
echo 4. 测试模型
echo 5. 查看训练日志（TensorBoard）
echo 6. 退出
echo.
set /p choice=请输入选项 (1-6): 

if "%choice%"=="1" goto install
if "%choice%"=="2" goto prepare
if "%choice%"=="3" goto train
if "%choice%"=="4" goto inference
if "%choice%"=="5" goto tensorboard
if "%choice%"=="6" goto end

echo 无效选项，请重新选择
goto menu

:install
echo.
echo [1/3] 安装依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
echo ✓ 依赖安装完成！
pause
goto menu

:prepare
echo.
echo [2/3] 处理数据...
python scripts/prepare_data.py
echo.
echo ✓ 数据处理完成！
pause
goto menu

:train
echo.
echo [3/3] 开始训练...
echo 提示：训练过程中可以按Ctrl+C中断
python scripts/train.py
echo.
echo ✓ 训练完成！
pause
goto menu

:inference
echo.
echo 请输入模型路径（例如：checkpoints/角色名_Qwen-1.8B-Chat/final_model）
set /p model_path=模型路径: 
echo.
echo 启动交互式对话...
python scripts/inference.py --model_path %model_path% --load_in_4bit
pause
goto menu

:tensorboard
echo.
echo 启动TensorBoard...
echo 打开浏览器访问: http://localhost:6006
start http://localhost:6006
tensorboard --logdir logs
pause
goto menu

:end
echo.
echo 再见！
exit

