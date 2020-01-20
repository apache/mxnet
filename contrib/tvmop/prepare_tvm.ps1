# Downloads, compiles, installs LLVM to 3rdparty/tvm/build
# This script only works with LLVM 8 since newer LLVM has different directories

$llvm_version_name = "llvm-8.0.1.src"
$llvm_url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/llvm-8.0.1.src.tar.xz"

$tvm_build_dir = "..\3rdparty\tvm\build\"

If (!(test-path $tvm_build_dir)) {
    New-Item -ItemType Directory -Force -Path $tvm_build_dir
}

function Expand-Tar($tarFile, $dest) {
    if (-not (Get-Command Expand-7Zip -ErrorAction Ignore)) {
        Install-Package -Scope CurrentUser -Force 7Zip4PowerShell > $null
    }
    Expand-7Zip $tarFile $dest
}

Invoke-WebRequest $llvm_url -OutFile $tvm_build_dir"llvm.tar"
Expand-Tar $tvm_build_dir"llvm.tar" $tvm_build_dir"llvm.tar.xz"
Expand-Tar $tvm_build_dir"llvm.tar.xz\llvm.tar" $tvm_build_dir"llvm"

Move-Item -Path $tvm_build_dir"llvm\"$llvm_version_name -Destination $tvm_build_dir$llvm_version_name
Remove-Item -Path $tvm_build_dir"llvm.tar" -Force
Remove-Item -Path $tvm_build_dir"llvm.tar.xz" -Recurse
Remove-Item -Path $tvm_build_dir"llvm" -Recurse

New-Item -ItemType Directory -Force -Path $tvm_build_dir$llvm_version_name"\build\"
Set-Location -Path $tvm_build_dir$llvm_version_name"\build\"
& cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="/EHsc /Gy /MT" -DCMAKE_RC_COMPILER="rc" -DCMAKE_C_COMPILER="clang-cl" -DCMAKE_CXX_COMPILER="clang-cl" ..
ninja

Set-Location -Path "..\..\"
New-Item -ItemType Directory -Force -Path "llvm"
Move-Item $llvm_version_name"\build\bin" -Destination "llvm\bin"
Move-Item $llvm_version_name"\build\include" -Destination "llvm\include"
Copy-Item -Path $llvm_version_name"\include\llvm\*" -Destination "llvm\include\llvm\" -Recurse -Force
Copy-Item -Path $llvm_version_name"\include\llvm-c" -Destination "llvm\include\llvm-c" -Recurse -Force
Move-Item $llvm_version_name"\build\lib" -Destination "llvm\lib"
Remove-Item $llvm_version_name -Recurse

# use MSVC linker for MXNet compilation
$msvc_linker = (get-command link.exe).Path
$clang_linker = (get-command lld-link.exe).Path
If (!(test-path $clang_linker".backup")) {
    Rename-Item -Path $clang_linker -NewName "lld-link.exe.backup"
}
Copy-Item $msvc_linker -Destination $clang_linker

Set-Location -Path $tvm_build_dir"..\..\..\build"
