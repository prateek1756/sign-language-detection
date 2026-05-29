# create_colab_zip.ps1
# Run this script from the project root to create backend_colab.zip
# Upload that zip to Colab when prompted in Cell 3 of any training notebook.
#
# Usage (from project root):
#   .\notebooks\create_colab_zip.ps1

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir  = Join-Path $ProjectRoot "backend"
$OutputZip   = Join-Path $PSScriptRoot "backend_colab.zip"

# Remove old zip if exists
if (Test-Path $OutputZip) { Remove-Item $OutputZip }

# Files to include (exclude venv, data, models, logs, __pycache__)
$Include = @(
    "configs\training_config.py",
    "src\__init__.py",
    "src\model.py",
    "src\train.py",
    "src\evaluate.py",
    "src\preprocess.py",
    "src\collect_data.py",
    "src\dataset_split.py",
    "src\download_dataset.py",
    "src\download_models.py",
    "src\inference.py"
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($OutputZip, 'Create')

foreach ($rel in $Include) {
    $full = Join-Path $BackendDir $rel
    if (Test-Path $full) {
        # Entry path inside zip: backend/<rel>
        $entryName = "backend/" + $rel.Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $full, $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
        Write-Host "  Added: $entryName"
    } else {
        Write-Warning "  Missing: $full"
    }
}

$zip.Dispose()
Write-Host ""
Write-Host "Created: $OutputZip"
Write-Host "Upload this file to Colab in Cell 3 of any training notebook."
