param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$destination = Join-Path $ProjectRoot 'data\raw\lpc_generator'
$remote = 'https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git'

if (-not (Test-Path -LiteralPath $destination)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    git clone --depth 1 --filter=blob:none --sparse $remote $destination
}

if (-not (Test-Path -LiteralPath (Join-Path $destination '.git'))) {
    throw "La ruta existe, pero no es un clon Git válido: $destination"
}

git -C $destination config core.longpaths true
git -C $destination sparse-checkout set --no-cone `
    '/CREDITS.csv' `
    '/LICENSE' `
    '/README.md' `
    '/palette_definitions/' `
    '/sheet_definitions/' `
    '/spritesheets/**/idle.png' `
    '/spritesheets/**/idle/*.png'

$idleCount = (Get-ChildItem -LiteralPath (Join-Path $destination 'spritesheets') -Recurse -File -Filter '*.png' |
    Where-Object { $_.FullName -match '\\idle(\\|\.png$)' }).Count

Write-Host "LPC listo en: $destination"
Write-Host "Capas idle disponibles: $idleCount"
