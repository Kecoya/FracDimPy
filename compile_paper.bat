@echo off
chcp 65001 >nul
REM JOSS论文编译脚本 - Windows版本
REM 使用方法: 双击compile_paper.bat或在命令行中运行

echo === JOSS Paper Compilation Script ===
echo 正在编译paper.md...

REM 检查Docker是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker未运行，请先启动Docker Desktop
    pause
    exit /b 1
)

REM 检查必要文件是否存在
if not exist "paper.md" (
    echo ❌ 未找到paper.md文件
    pause
    exit /b 1
)

if not exist "paper.bib" (
    echo ❌ 未找到paper.bib文件
    pause
    exit /b 1
)

REM 使用JOSS Docker镜像编译
echo 🐳 正在使用JOSS Docker镜像编译...
docker run --rm --volume "%CD%":/data --env JOURNAL=joss openjournals/inara

REM 检查编译结果
if exist "paper.pdf" (
    echo ✅ 编译成功！生成了paper.pdf文件
    echo 📄 PDF文件已生成在当前目录
) else (
    echo ❌ 编译失败，未生成paper.pdf文件
    pause
    exit /b 1
)

echo.
echo 编译完成！你可以打开paper.pdf查看结果。
pause