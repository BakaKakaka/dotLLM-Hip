<#
.SYNOPSIS
    Validates the HIP SDK / MSVC / GPU environment for DotLLM.Hip (work package H0.1).

.DESCRIPTION
    Checks, in order:
      1. HIP_PATH points at the SDK root (W14): include\hip\hip_runtime.h + bin\hipcc.exe.
      2. hipcc reports HIP 7.2.x (D4).
      3. amdhip64_7.dll, rocblas.dll and rocblas\library (with gfx1100/gfx1201 files) exist (W5, W7).
      4. HIP_VISIBLE_DEVICES is reported (W4).
      5. MSVC x64 tools and the Windows SDK (ucrt) are installed (W15).
      6. hipInfo.exe: lists devices, requires gfx1100, expects gfx1201, ignores iGPU/unsupported (D6, D9).
      7. Compiles native/hip/kernels/toolchain_smoke.hip for gfx1100 + gfx1201 with the exact
         DotLLM.Hip build flags and verifies the ISA (v_dot4_i32_iu8, v_wmma_f32_16x16x16_f16,
         .wavefront_size: 32).
      8. Runs the smoke kernels on every supported card via hipModuleLoadData/hipModuleLaunchKernel
         in a child process and verifies that exactly one amdhip64_* module is loaded (W5, W6).

    Exit code 0 = all checks passed (warnings allowed), 1 = at least one check failed.
    See docs/HIP.md and docs/HIP_PITFALLS.md.

.PARAMETER SkipSmoke
    Skip steps 7 and 8 (kernel compilation and GPU execution).

.PARAMETER SkipRun
    Compile and inspect the smoke kernel, but do not execute it on the GPU (step 8).

.PARAMETER WorkDir
    Scratch directory for the smoke build. Defaults to %TEMP%\dotllm-hip-env.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\check-hip-env.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipSmoke,
    [switch]$SkipRun,
    [string]$WorkDir = (Join-Path $env:TEMP 'dotllm-hip-env'),

    # Internal: child-process mode for step 8. Not intended to be called directly.
    [string]$RunSmokeChild,
    [string]$HipBin
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$RequiredHipMajor = 7
$RequiredHipMinor = 2
$SupportedArchs = @('gfx1100', 'gfx1201')
$PrimaryArch = 'gfx1100'
$KernelFlags = @('--genco', '-O3', '-mno-wavefrontsize64', '-std=c++17', '-Xclang', '-fno-cuda-host-device-constexpr')

# ---------------------------------------------------------------------------------------------
# Child mode (step 8): load the HIP runtime by full path, run the smoke kernels, check modules.
# ---------------------------------------------------------------------------------------------
$HipSmokeSource = @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

public static class DotLlmHipSmoke
{
    private const string Hip = "amdhip64_7.dll";
    private const uint LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100;
    private const uint LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000;
    private const int MemcpyHostToDevice = 1;
    private const int MemcpyDeviceToHost = 2;
    private const int Wave = 32;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryExW(string path, IntPtr file, uint flags);

    [DllImport(Hip)] private static extern int hipInit(uint flags);
    [DllImport(Hip)] private static extern int hipRuntimeGetVersion(out int version);
    [DllImport(Hip)] private static extern int hipDriverGetVersion(out int version);
    [DllImport(Hip)] private static extern int hipGetDeviceCount(out int count);
    [DllImport(Hip)] private static extern int hipSetDevice(int device);
    [DllImport(Hip)] private static extern int hipDeviceGetName(byte[] name, int len, int device);
    [DllImport(Hip)] private static extern int hipMalloc(out IntPtr devicePtr, UIntPtr size);
    [DllImport(Hip)] private static extern int hipFree(IntPtr devicePtr);
    [DllImport(Hip)] private static extern int hipMemcpy(IntPtr dst, IntPtr src, UIntPtr size, int kind);
    [DllImport(Hip)] private static extern int hipMemset(IntPtr dst, int value, UIntPtr size);
    [DllImport(Hip)] private static extern int hipDeviceSynchronize();
    [DllImport(Hip)] private static extern int hipModuleLoadData(out IntPtr module, IntPtr image);
    [DllImport(Hip)] private static extern int hipModuleGetFunction(out IntPtr function, IntPtr module, [MarshalAs(UnmanagedType.LPStr)] string name);
    [DllImport(Hip)] private static extern int hipModuleUnload(IntPtr module);
    [DllImport(Hip)] private static extern int hipModuleLaunchKernel(IntPtr function,
        uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ,
        uint sharedMemBytes, IntPtr stream, IntPtr kernelParams, IntPtr extra);
    [DllImport(Hip)] private static extern IntPtr hipGetErrorName(int error);

    private static void Check(int error, string what)
    {
        if (error != 0)
        {
            string name = Marshal.PtrToStringAnsi(hipGetErrorName(error));
            throw new InvalidOperationException(string.Format("{0} failed: {1} ({2})", what, name, error));
        }
    }

    /// Loads amdhip64_7.dll by full path (W5) and initializes the runtime. Returns runtime version.
    public static int Initialize(string hipBinDir)
    {
        string path = Path.Combine(hipBinDir, Hip);
        if (LoadLibraryExW(path, IntPtr.Zero, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) == IntPtr.Zero)
            throw new InvalidOperationException(string.Format("LoadLibraryEx('{0}') failed, Win32 error {1}", path, Marshal.GetLastWin32Error()));
        Check(hipInit(0), "hipInit");
        int version;
        Check(hipRuntimeGetVersion(out version), "hipRuntimeGetVersion");
        return version;
    }

    public static int DriverVersion()
    {
        int version;
        Check(hipDriverGetVersion(out version), "hipDriverGetVersion");
        return version;
    }

    public static int DeviceCount()
    {
        int count;
        Check(hipGetDeviceCount(out count), "hipGetDeviceCount");
        return count;
    }

    public static string DeviceName(int device)
    {
        byte[] buffer = new byte[256];
        Check(hipDeviceGetName(buffer, buffer.Length, device), "hipDeviceGetName");
        int len = Array.IndexOf(buffer, (byte)0);
        return Encoding.ASCII.GetString(buffer, 0, len < 0 ? buffer.Length : len);
    }

    private static void Launch(IntPtr function, params IntPtr[] args)
    {
        int n = args.Length;
        IntPtr values = Marshal.AllocHGlobal(IntPtr.Size * n);
        IntPtr pointers = Marshal.AllocHGlobal(IntPtr.Size * n);
        try
        {
            for (int i = 0; i < n; i++)
            {
                Marshal.WriteIntPtr(values, i * IntPtr.Size, args[i]);
                Marshal.WriteIntPtr(pointers, i * IntPtr.Size, IntPtr.Add(values, i * IntPtr.Size));
            }
            Check(hipModuleLaunchKernel(function, 1, 1, 1, Wave, 1, 1, 0, IntPtr.Zero, pointers, IntPtr.Zero), "hipModuleLaunchKernel");
            Check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        }
        finally
        {
            Marshal.FreeHGlobal(pointers);
            Marshal.FreeHGlobal(values);
        }
    }

    private static IntPtr Alloc(int bytes)
    {
        IntPtr devicePtr;
        Check(hipMalloc(out devicePtr, new UIntPtr((uint)bytes)), "hipMalloc");
        Check(hipMemset(devicePtr, 0, new UIntPtr((uint)bytes)), "hipMemset");
        return devicePtr;
    }

    private static IntPtr Upload(int[] data)
    {
        IntPtr devicePtr = Alloc(data.Length * 4);
        GCHandle h = GCHandle.Alloc(data, GCHandleType.Pinned);
        try { Check(hipMemcpy(devicePtr, h.AddrOfPinnedObject(), new UIntPtr((uint)(data.Length * 4)), MemcpyHostToDevice), "hipMemcpy H2D"); }
        finally { h.Free(); }
        return devicePtr;
    }

    private static IntPtr Upload(short[] data)
    {
        IntPtr devicePtr = Alloc(data.Length * 2);
        GCHandle h = GCHandle.Alloc(data, GCHandleType.Pinned);
        try { Check(hipMemcpy(devicePtr, h.AddrOfPinnedObject(), new UIntPtr((uint)(data.Length * 2)), MemcpyHostToDevice), "hipMemcpy H2D"); }
        finally { h.Free(); }
        return devicePtr;
    }

    private static void Download<T>(IntPtr devicePtr, T[] data, int elementSize) where T : struct
    {
        GCHandle h = GCHandle.Alloc(data, GCHandleType.Pinned);
        try { Check(hipMemcpy(h.AddrOfPinnedObject(), devicePtr, new UIntPtr((uint)(data.Length * elementSize)), MemcpyDeviceToHost), "hipMemcpy D2H"); }
        finally { h.Free(); }
    }

    private static int PackInt8x4(int b0, int b1, int b2, int b3)
    {
        return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    /// Runs smoke_dot4, smoke_wave and smoke_wmma on one device. Throws on any mismatch.
    public static string Run(int device, string hsacoPath)
    {
        Check(hipSetDevice(device), "hipSetDevice");
        byte[] image = File.ReadAllBytes(hsacoPath);
        IntPtr hostImage = Marshal.AllocHGlobal(image.Length);
        IntPtr module = IntPtr.Zero;
        IntPtr dA = IntPtr.Zero, dB = IntPtr.Zero, dOut = IntPtr.Zero;
        StringBuilder report = new StringBuilder();
        try
        {
            Marshal.Copy(image, 0, hostImage, image.Length);
            Check(hipModuleLoadData(out module, hostImage), "hipModuleLoadData");
            IntPtr fDot4, fWave, fWmma;
            Check(hipModuleGetFunction(out fDot4, module, "smoke_dot4"), "hipModuleGetFunction(smoke_dot4)");
            Check(hipModuleGetFunction(out fWave, module, "smoke_wave"), "hipModuleGetFunction(smoke_wave)");
            Check(hipModuleGetFunction(out fWmma, module, "smoke_wmma"), "hipModuleGetFunction(smoke_wmma)");

            // --- smoke_dot4: signed int8x4 dot product (v_dot4_i32_iu8) ---
            int[] a = new int[Wave], b = new int[Wave], expected = new int[Wave], got = new int[Wave];
            for (int i = 0; i < Wave; i++)
            {
                int a0 = i * 7 - 100, a1 = -i * 3 + 50, a2 = i - 16, a3 = 127 - i * 8;
                int b0 = -128 + i * 5, b1 = i * 2 - 31, b2 = 99 - i * 6, b3 = (i % 5) - 2;
                a[i] = PackInt8x4(a0, a1, a2, a3);
                b[i] = PackInt8x4(b0, b1, b2, b3);
                expected[i] = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }
            dA = Upload(a); dB = Upload(b); dOut = Alloc(Wave * 4);
            Launch(fDot4, dA, dB, dOut);
            Download(dOut, got, 4);
            for (int i = 0; i < Wave; i++)
                if (got[i] != expected[i])
                    throw new InvalidOperationException(string.Format("smoke_dot4 lane {0}: got {1}, expected {2}", i, got[i], expected[i]));
            report.Append("dot4 OK");
            hipFree(dA); hipFree(dB); hipFree(dOut); dA = dB = dOut = IntPtr.Zero;

            // --- smoke_wave: xor-butterfly with 64-bit mask + warpSize ---
            int[] wave = new int[2 * Wave];
            dOut = Alloc(wave.Length * 4);
            Launch(fWave, dOut);
            Download(dOut, wave, 4);
            for (int i = 0; i < Wave; i++)
            {
                if (wave[i] != 496)
                    throw new InvalidOperationException(string.Format("smoke_wave lane {0}: sum {1}, expected 496", i, wave[i]));
                if (wave[Wave + i] != 32)
                    throw new InvalidOperationException(string.Format("smoke_wave lane {0}: warpSize {1}, expected 32", i, wave[Wave + i]));
            }
            report.Append(", wave32/shfl OK");
            hipFree(dOut); dOut = IntPtr.Zero;

            // --- smoke_wmma: 16x16x16 f16 -> f32, A = 1.0, B = 2.0 => C = 32.0 everywhere ---
            short[] ha = new short[256], hb = new short[256];
            for (int i = 0; i < 256; i++) { ha[i] = 0x3C00; hb[i] = 0x4000; }
            float[] c = new float[256];
            dA = Upload(ha); dB = Upload(hb); dOut = Alloc(256 * 4);
            Launch(fWmma, dA, dB, dOut);
            Download(dOut, c, 4);
            for (int i = 0; i < 256; i++)
                if (c[i] != 32.0f)
                    throw new InvalidOperationException(string.Format("smoke_wmma element {0}: got {1}, expected 32", i, c[i]));
            report.Append(", wmma f16->f32 OK");
        }
        finally
        {
            if (dA != IntPtr.Zero) hipFree(dA);
            if (dB != IntPtr.Zero) hipFree(dB);
            if (dOut != IntPtr.Zero) hipFree(dOut);
            if (module != IntPtr.Zero) hipModuleUnload(module);
            Marshal.FreeHGlobal(hostImage);
        }
        return report.ToString();
    }
}
'@

if ($RunSmokeChild) {
    # $RunSmokeChild format: "ordinal=arch=hsacoPath;ordinal=arch=hsacoPath"
    $exit = 0
    try {
        Add-Type -TypeDefinition $HipSmokeSource -Language CSharp
        $rtVersion = [DotLlmHipSmoke]::Initialize($HipBin)
        $rtMajor = [int][math]::Floor($rtVersion / 10000000)
        $rtMinor = [int][math]::Floor(($rtVersion % 10000000) / 100000)
        $rtPatch = $rtVersion % 100000
        if ($rtMajor -ne $RequiredHipMajor) {
            Write-Output "FAIL|runtime|hipRuntimeGetVersion = $rtVersion (major $rtMajor), expected major $RequiredHipMajor (D4)"
            exit 1
        }
        Write-Output "PASS|runtime|HIP runtime $rtMajor.$rtMinor.$rtPatch, driver version $([DotLlmHipSmoke]::DriverVersion()), $([DotLlmHipSmoke]::DeviceCount()) visible device(s)"

        foreach ($spec in $RunSmokeChild.Split(';')) {
            $parts = $spec.Split('=', 3)
            $ordinal = [int]$parts[0]; $arch = $parts[1]; $hsaco = $parts[2]
            $name = [DotLlmHipSmoke]::DeviceName($ordinal)
            try {
                $result = [DotLlmHipSmoke]::Run($ordinal, $hsaco)
                Write-Output "PASS|device $ordinal|$name ($arch): $result"
            }
            catch {
                $ex = $_.Exception
                if ($null -ne $ex.InnerException) { $ex = $ex.InnerException }
                Write-Output "FAIL|device $ordinal|$name ($arch): $($ex.Message)"
                $exit = 1
            }
        }

        # W5/W6: exactly one HIP runtime module, and it must be the SDK's v7 DLL.
        $hipModules = @([System.Diagnostics.Process]::GetCurrentProcess().Modules |
            Where-Object { $_.ModuleName -like 'amdhip64_*' })
        $expectedPath = [System.IO.Path]::GetFullPath((Join-Path $HipBin 'amdhip64_7.dll'))
        if ($hipModules.Count -ne 1) {
            Write-Output ("FAIL|modules|{0} amdhip64_* modules loaded: {1}" -f $hipModules.Count, (($hipModules | ForEach-Object { $_.FileName }) -join ', '))
            $exit = 1
        }
        elseif (-not [string]::Equals($hipModules[0].FileName, $expectedPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            Write-Output "FAIL|modules|amdhip64 loaded from '$($hipModules[0].FileName)', expected '$expectedPath'"
            $exit = 1
        }
        else {
            Write-Output "PASS|modules|exactly one HIP runtime loaded: $($hipModules[0].FileName)"
        }
    }
    catch {
        Write-Output "FAIL|runtime|$($_.Exception.Message)"
        $exit = 1
    }
    exit $exit
}

# ---------------------------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------------------------
$script:Failures = 0
$script:Warnings = 0

function Write-Section([string]$title) {
    Write-Host ''
    Write-Host "== $title ==" -ForegroundColor Cyan
}
function Write-Pass([string]$msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-Info([string]$msg) { Write-Host "  [INFO] $msg" -ForegroundColor Gray }
function Write-Warn([string]$msg) { $script:Warnings++; Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { $script:Failures++; Write-Host "  [FAIL] $msg" -ForegroundColor Red }

# Quotes a single argument for CreateProcess (W12: HIP_PATH contains spaces).
function ConvertTo-ArgString([string[]]$arguments) {
    $quoted = foreach ($a in $arguments) {
        if ($a -eq '') { '""' }
        elseif ($a -match '[\s"]') { '"' + ($a -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' }
        else { $a }
    }
    return ($quoted -join ' ')
}

# Runs a native executable and captures stdout/stderr without PowerShell's stderr-as-error semantics.
function Invoke-Native([string]$exe, [string[]]$arguments) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = ConvertTo-ArgString $arguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $stdout = $p.StandardOutput.ReadToEndAsync()
    $stderr = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    return [pscustomobject]@{ ExitCode = $p.ExitCode; StdOut = $stdout.Result; StdErr = $stderr.Result }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Write-Host 'dotLLM HIP environment check (H0.1)' -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"

# ---------------------------------------------------------------------------------------------
# 1. HIP_PATH (W14)
# ---------------------------------------------------------------------------------------------
Write-Section '1. HIP_PATH'
$hipPath = $env:HIP_PATH
$hipBinDir = $null
if ([string]::IsNullOrWhiteSpace($hipPath)) {
    Write-Fail 'HIP_PATH is not set. Install HIP SDK 7.2.3 and set HIP_PATH to the SDK root, e.g. C:\Program Files\AMD\ROCm\7.2\ (restart the terminal/IDE afterwards).'
}
else {
    $hipRoot = $hipPath.TrimEnd('\', '/')
    Write-Info "HIP_PATH = $hipPath"
    if ((Split-Path -Leaf $hipRoot) -ieq 'bin') {
        Write-Fail "HIP_PATH points at the bin directory ('$hipPath'). It must be the SDK root, e.g. '$(Split-Path -Parent $hipRoot)\' (W14). Restart terminals/IDE after changing it."
    }
    $runtimeHeader = Join-Path $hipRoot 'include\hip\hip_runtime.h'
    $hipccPath = Join-Path $hipRoot 'bin\hipcc.exe'
    $rootOk = $true
    if (-not (Test-Path -LiteralPath $runtimeHeader)) { Write-Fail "Missing $runtimeHeader - HIP_PATH is not the SDK root (W14)."; $rootOk = $false }
    if (-not (Test-Path -LiteralPath $hipccPath)) { Write-Fail "Missing $hipccPath - HIP_PATH is not the SDK root (W14)."; $rootOk = $false }
    if ($rootOk) {
        Write-Pass 'HIP_PATH is the SDK root (include\hip\hip_runtime.h and bin\hipcc.exe found)'
        $hipBinDir = Join-Path $hipRoot 'bin'
    }
}

# ---------------------------------------------------------------------------------------------
# 2. hipcc version (D4)
# ---------------------------------------------------------------------------------------------
Write-Section '2. hipcc version'
$hipccVersionLine = $null
if ($hipBinDir) {
    $r = Invoke-Native (Join-Path $hipBinDir 'hipcc.exe') @('--version')
    $text = $r.StdOut + $r.StdErr
    $m = [regex]::Match($text, 'HIP version:\s*(\d+)\.(\d+)\.(\d+)\S*')
    if ($r.ExitCode -ne 0 -or -not $m.Success) {
        Write-Fail "hipcc --version failed (exit $($r.ExitCode)): $($text.Trim())"
    }
    else {
        $hipccVersionLine = $m.Value
        $clang = [regex]::Match($text, 'clang version\s*\S+').Value
        if ([int]$m.Groups[1].Value -eq $RequiredHipMajor -and [int]$m.Groups[2].Value -eq $RequiredHipMinor) {
            Write-Pass "$($m.Value) ($clang)"
        }
        else {
            Write-Fail "$($m.Value) - expected HIP $RequiredHipMajor.$RequiredHipMinor.x (D4: SDK 7.2.3 is locked)."
        }
    }
}
else { Write-Fail 'Skipped: HIP_PATH invalid.' }

# ---------------------------------------------------------------------------------------------
# 3. Runtime DLLs and rocBLAS library (W5, W7)
# ---------------------------------------------------------------------------------------------
Write-Section '3. Runtime libraries'
if ($hipBinDir) {
    foreach ($dll in @('amdhip64_7.dll', 'rocblas.dll')) {
        $p = Join-Path $hipBinDir $dll
        if (Test-Path -LiteralPath $p) {
            $ver = (Get-Item -LiteralPath $p).VersionInfo.FileVersion
            if ($ver) { Write-Pass "$dll ($ver)" } else { Write-Pass $dll }
        }
        else { Write-Fail "Missing $p" }
    }
    $rocblasLib = Join-Path $hipBinDir 'rocblas\library'
    if (Test-Path -LiteralPath $rocblasLib) {
        foreach ($arch in $SupportedArchs) {
            $n = @(Get-ChildItem -LiteralPath $rocblasLib -File | Where-Object { $_.Name -like "*$arch*" }).Count
            if ($n -gt 0) { Write-Pass "rocblas\library: $n files for $arch" }
            else { Write-Fail "rocblas\library has no files for $arch (W7)" }
        }
    }
    else { Write-Fail "Missing $rocblasLib - rocBLAS cannot load its Tensile kernels (W7)." }

    $driverV6 = Join-Path $env:SystemRoot 'System32\amdhip64_6.dll'
    if (Test-Path -LiteralPath $driverV6) {
        Write-Info 'System32\amdhip64_6.dll belongs to the Adrenalin driver - expected, never loaded by dotLLM, do NOT remove (W5).'
    }
}
else { Write-Fail 'Skipped: HIP_PATH invalid.' }

# ---------------------------------------------------------------------------------------------
# 4. HIP_VISIBLE_DEVICES (W4)
# ---------------------------------------------------------------------------------------------
Write-Section '4. HIP_VISIBLE_DEVICES'
$hvdProcess = $env:HIP_VISIBLE_DEVICES
$hvdUser = [Environment]::GetEnvironmentVariable('HIP_VISIBLE_DEVICES', 'User')
$hvdMachine = [Environment]::GetEnvironmentVariable('HIP_VISIBLE_DEVICES', 'Machine')
$fmt = { param($v) if ($null -eq $v -or $v -eq '') { '<not set>' } else { $v } }
Write-Info ("process = {0}, user = {1}, machine = {2}" -f (& $fmt $hvdProcess), (& $fmt $hvdUser), (& $fmt $hvdMachine))
Write-Info 'Ordinals depend on this variable. dotLLM selects devices by architecture (hip:gfx1100), never by hard-coded ordinal (W4).'

# ---------------------------------------------------------------------------------------------
# 5. MSVC + Windows SDK (W15)
# ---------------------------------------------------------------------------------------------
Write-Section '5. MSVC and Windows SDK'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path -LiteralPath $vswhere) {
    $r = Invoke-Native $vswhere @('-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
    $vsPath = $r.StdOut.Trim()
    if ($vsPath) {
        $vcVerFile = Join-Path $vsPath 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
        $vcVer = if (Test-Path -LiteralPath $vcVerFile) { (Get-Content -LiteralPath $vcVerFile -TotalCount 1).Trim() } else { 'unknown' }
        $vsVer = (Invoke-Native $vswhere @('-path', $vsPath, '-property', 'catalog_productDisplayVersion')).StdOut.Trim()
        Write-Pass "MSVC x64 tools $vcVer (Visual Studio $vsVer at $vsPath)"
    }
    else {
        Write-Fail 'No Visual Studio installation with MSVC x64 tools. Install the "Desktop development with C++" workload (W15).'
    }
}
else {
    Write-Fail "vswhere.exe not found ($vswhere). Install Visual Studio 2026 with ""Desktop development with C++"" (W15)."
}

$kitsRoot = $null
try {
    $kitsRoot = (Get-ItemProperty -Path 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots' -Name KitsRoot10 -ErrorAction Stop).KitsRoot10
} catch {
    try { $kitsRoot = (Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -Name KitsRoot10 -ErrorAction Stop).KitsRoot10 } catch { }
}
if ($kitsRoot) {
    $ucrt = @(Get-ChildItem -LiteralPath (Join-Path $kitsRoot 'Include') -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'ucrt\stdlib.h') } | ForEach-Object { $_.Name })
    if ($ucrt.Count -gt 0) { Write-Pass "Windows SDK ucrt: $($ucrt -join ', ')" }
    else { Write-Fail "Windows SDK found at $kitsRoot but no Include\<ver>\ucrt\stdlib.h (W15)." }
}
else { Write-Fail 'Windows 10/11 SDK not found (registry KitsRoot10). Install it with the C++ workload (W15).' }

# ---------------------------------------------------------------------------------------------
# 6. Devices via hipInfo.exe (D6, D9, W4, W11)
# ---------------------------------------------------------------------------------------------
Write-Section '6. Devices (hipInfo.exe)'
$devices = New-Object System.Collections.Generic.List[object]
if ($hipBinDir) {
    $hipInfo = Join-Path $hipBinDir 'hipInfo.exe'
    if (Test-Path -LiteralPath $hipInfo) {
        $r = Invoke-Native $hipInfo @()
        $current = $null
        foreach ($line in ($r.StdOut -split "`r?`n")) {
            # Device header has no colon: "device#                           0"
            $dm = [regex]::Match($line, '^\s*device#\s+(\d+)\s*$')
            if ($dm.Success) {
                $current = [ordered]@{ Ordinal = [int]$dm.Groups[1].Value }
                $devices.Add($current)
                continue
            }
            $m = [regex]::Match($line, '^\s*([A-Za-z0-9#_.]+):\s*(.*?)\s*$')
            if ($m.Success -and $null -ne $current -and -not $current.Contains($m.Groups[1].Value)) {
                $current[$m.Groups[1].Value] = $m.Groups[2].Value
            }
        }
        if ($devices.Count -eq 0) {
            Write-Fail "hipInfo.exe reported no devices (exit $($r.ExitCode)). $($r.StdErr.Trim())"
        }
        foreach ($d in $devices) {
            $arch = if ($d.Contains('gcnArchName')) { ($d['gcnArchName'] -split ':')[0] } else { '?' }
            $d['Arch'] = $arch
            $integrated = $d.Contains('isIntegrated') -and $d['isIntegrated'] -eq '1'
            $wgp = if ($d.Contains('multiProcessorCount')) { [int]$d['multiProcessorCount'] } else { 0 }
            $l2 = if ($d.Contains('l2CacheSize') -and $d['l2CacheSize'] -match '^\d+$') { '{0} MB' -f ([long]$d['l2CacheSize'] / 1MB) } else { '?' }
            $desc = "#{0} {1} | {2} | {3} | {4} WGP = {5} CU | L2 {6} | warpSize {7} | isIntegrated {8}" -f `
                $d['Ordinal'], $d['Name'], $arch, $d['totalGlobalMem'], $wgp, (2 * $wgp), $l2, $d['warpSize'], $d['isIntegrated']
            if ($integrated) {
                $d['Supported'] = $false
                Write-Warn "$desc -> IGNORED (integrated GPU, D6/D9)"
            }
            elseif ($SupportedArchs -notcontains $arch) {
                $d['Supported'] = $false
                Write-Warn "$desc -> IGNORED (unsupported architecture, D6)"
            }
            else {
                $d['Supported'] = $true
                Write-Pass "$desc -> supported"
            }
        }
        $supported = @($devices | Where-Object { $_['Supported'] })
        if (-not ($supported | Where-Object { $_['Arch'] -eq $PrimaryArch })) {
            Write-Fail "No $PrimaryArch device visible (primary target RX 7900 XTX). Check HIP_VISIBLE_DEVICES and the driver."
        }
        if (-not ($supported | Where-Object { $_['Arch'] -eq 'gfx1201' })) {
            Write-Warn 'No gfx1201 device visible (secondary target RX 9070 XT). gfx1201 correctness tests will be skipped.'
        }
    }
    else { Write-Fail "Missing $hipInfo" }
}
else { Write-Fail 'Skipped: HIP_PATH invalid.' }

# ---------------------------------------------------------------------------------------------
# 7. Smoke kernel: compile + ISA verification (W15, D7)
# ---------------------------------------------------------------------------------------------
$hsacoByArch = @{}
if ($SkipSmoke) {
    Write-Section '7. Smoke kernel compilation'
    Write-Info 'Skipped (-SkipSmoke).'
}
elseif (-not $hipBinDir) {
    Write-Section '7. Smoke kernel compilation'
    Write-Fail 'Skipped: HIP_PATH invalid.'
}
else {
    Write-Section '7. Smoke kernel compilation and ISA'
    $kernelDir = Join-Path $repoRoot 'native\hip\kernels'
    $smokeSource = Join-Path $kernelDir 'toolchain_smoke.hip'
    if (-not (Test-Path -LiteralPath $smokeSource)) {
        Write-Fail "Missing $smokeSource"
    }
    else {
        New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
        Write-Info "Flags: $($KernelFlags -join ' ')"
        foreach ($arch in $SupportedArchs) {
            $hsaco = Join-Path $WorkDir "toolchain_smoke.$arch.hsaco"
            $codeObject = Join-Path $WorkDir "toolchain_smoke.$arch.co"
            Remove-Item -LiteralPath $hsaco, $codeObject -ErrorAction SilentlyContinue

            # '-I' and the directory must be separate: hipcc.exe forwards a joined -I"dir with spaces" unquoted (W12).
            $hipccArgs = @($KernelFlags) + @("--offload-arch=$arch", '-I', $kernelDir, '-o', $hsaco, $smokeSource)
            $r = Invoke-Native (Join-Path $hipBinDir 'hipcc.exe') $hipccArgs
            if ($r.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $hsaco)) {
                $err = ($r.StdOut + $r.StdErr).Trim()
                $hint = ''
                if ($err -match 'cannot overload __host__ __device__') { $hint = ' Hint: -Xclang -fno-cuda-host-device-constexpr missing or STL changed (W15).' }
                elseif ($err -match 'STL1000') { $hint = ' Hint: MSVC STL requires a newer Clang; emergency flag -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH (W15).' }
                elseif ($err -match "cannot find|vcruntime|corecrt|'stdlib.h' file not found") { $hint = ' Hint: MSVC or Windows SDK (ucrt) not found by hipcc (W15).' }
                Write-Fail "hipcc $arch failed (exit $($r.ExitCode)).$hint`n$err"
                continue
            }
            $head = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($hsaco), 0, 24)
            $bundleNote = if ($head.StartsWith('__CLANG_OFFLOAD_BUNDLE__')) { 'clang offload bundle' } else { 'raw code object' }
            Write-Pass "hipcc --genco $arch -> $((Get-Item -LiteralPath $hsaco).Length) bytes ($bundleNote)"
            $hsacoByArch[$arch] = $hsaco

            $r = Invoke-Native (Join-Path $hipBinDir 'clang-offload-bundler.exe') @('--unbundle', '--type=o', "--targets=hipv4-amdgcn-amd-amdhsa--$arch", "--input=$hsaco", "--output=$codeObject")
            if ($r.ExitCode -ne 0) { Write-Fail "clang-offload-bundler $arch failed: $(($r.StdOut + $r.StdErr).Trim())"; continue }

            $disasm = (Invoke-Native (Join-Path $hipBinDir 'llvm-objdump.exe') @('-d', "--mcpu=$arch", $codeObject)).StdOut
            $notes = (Invoke-Native (Join-Path $hipBinDir 'llvm-readobj.exe') @('--notes', $codeObject)).StdOut
            foreach ($insn in @('v_dot4_i32_iu8', 'v_wmma_f32_16x16x16_f16')) {
                $hit = [regex]::Match($disasm, "(?m)^\s*($insn\b[^/\r\n]*)")
                if ($hit.Success) { Write-Pass "$arch ISA: $($hit.Groups[1].Value.Trim())" }
                else { Write-Fail "$arch ISA: '$insn' not found in disassembly" }
            }
            $waveSizes = @([regex]::Matches($notes, '\.wavefront_size:\s*(\d+)') | ForEach-Object { $_.Groups[1].Value })
            if ($waveSizes.Count -gt 0 -and -not ($waveSizes | Where-Object { $_ -ne '32' })) {
                Write-Pass "$arch metadata: .wavefront_size: 32 for all $($waveSizes.Count) kernels"
            }
            else { Write-Fail "$arch metadata: unexpected wavefront sizes [$($waveSizes -join ', ')] (expected 32)" }
        }
    }
}

# ---------------------------------------------------------------------------------------------
# 8. Smoke kernel execution on the GPUs + single-runtime check (W5, W6)
# ---------------------------------------------------------------------------------------------
Write-Section '8. Smoke kernel execution'
if ($SkipSmoke -or $SkipRun) {
    Write-Info 'Skipped (-SkipSmoke/-SkipRun).'
}
else {
    $specs = @()
    foreach ($d in @($devices | Where-Object { $_['Supported'] })) {
        if ($hsacoByArch.ContainsKey($d['Arch'])) { $specs += ('{0}={1}={2}' -f $d['Ordinal'], $d['Arch'], $hsacoByArch[$d['Arch']]) }
    }
    if ($specs.Count -eq 0) {
        Write-Fail 'No supported device with a compiled smoke kernel - nothing to run.'
    }
    else {
        # Child process: keeps the HIP runtime out of this shell and gives the module check a clean process.
        $hostExe = (Get-Process -Id $PID).Path
        $childArgs = @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
            '-RunSmokeChild', ($specs -join ';'), '-HipBin', $hipBinDir)
        $r = Invoke-Native $hostExe $childArgs
        $parsed = 0
        foreach ($line in ($r.StdOut -split "`r?`n")) {
            $parts = $line.Split('|', 3)
            if ($parts.Count -ne 3) { continue }
            $parsed++
            if ($parts[0] -eq 'PASS') { Write-Pass "$($parts[1]): $($parts[2])" } else { Write-Fail "$($parts[1]): $($parts[2])" }
        }
        if ($r.ExitCode -ne 0 -and $parsed -eq 0) {
            Write-Fail "Smoke child process failed (exit $($r.ExitCode)): $(($r.StdOut + $r.StdErr).Trim())"
        }
    }
}

# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------
Write-Host ''
if ($script:Failures -gt 0) {
    Write-Host "RESULT: FAIL ($($script:Failures) failure(s), $($script:Warnings) warning(s))" -ForegroundColor Red
    exit 1
}
Write-Host "RESULT: PASS ($($script:Warnings) warning(s))" -ForegroundColor Green
exit 0
