@echo off
setlocal
rem Runs cargo with the PATH this machine needs.
rem The GNU Rust toolchain links windows-sys through raw-dylib, which needs
rem MinGW dlltool and as. They come from MSYS2 (mingw-w64-ucrt-x86_64-binutils).
rem   build.bat                -> cargo build --release (CLI and GUI)
rem   build.bat test           -> cargo test
rem   build.bat platforms      -> type-check for Linux and macOS (no linking;
rem                               without SQLite, whose C source needs a
rem                               cross compiler: the index falls back to JSON
rem                               and there's no flat library; then again
rem                               with the graphics card code)
rem   build.bat run -- report C:\PixInsight\...
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%;C:\msys64\ucrt64\bin"
if "%~1"=="platforms" goto platforms
if not "%~1"=="" goto passthru

rem Release build. Now and then the final link fails with "ld returned 120
rem exit status" and no message (18, 19 and 20 Nov 2026). It always worked
rem the second time, and it couldn't be reproduced: not PATH, a locked or
rem running exe, memory, Defender or a crash. So: try once more, and note
rem each failure in target\link-failures.log (when, and what was running).
cargo build --release --workspace
if not errorlevel 1 exit /b 0
set "LOG=%~dp0target\link-failures.log"
>> "%LOG%" echo %date% %time%  build failed (see the console for the error); running:
tasklist /fo table /nh | findstr /i /c:"cargo.exe" /c:"rustc.exe" /c:"ld.exe" /c:"gcc.exe" /c:"crux" /c:"rust-analyzer" >> "%LOG%"
echo.
echo Build failed. If that was the occasional link error ("ld returned 120"),
echo a second try fixes it: trying once more. Noted in target\link-failures.log.
echo.
cargo build --release --workspace
if errorlevel 1 (
  >> "%LOG%" echo   retry failed too
  exit /b 1
)
>> "%LOG%" echo   retry succeeded
exit /b 0

:platforms
cargo check --workspace --all-targets --no-default-features --target x86_64-unknown-linux-gnu || exit /b 1
cargo check --workspace --all-targets --no-default-features --target aarch64-apple-darwin || exit /b 1
rem The graphics card code (wgpu: Vulkan on Linux, Metal on macOS) needs no
rem C compiler, so it's checked too.
cargo check --workspace --all-targets --no-default-features --features gpu --target x86_64-unknown-linux-gnu || exit /b 1
cargo check --workspace --all-targets --no-default-features --features gpu --target aarch64-apple-darwin || exit /b 1
exit /b 0

:passthru
cargo %*
