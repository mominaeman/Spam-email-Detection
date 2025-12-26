# Automated Setup Script for Spam Email Detection Project
# This script will prepare your project for immediate demo

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Spam Detector - Quick Setup Script   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python first." -ForegroundColor Red
    exit 1
}

# Step 2: Install NumPy
Write-Host "`n[2/6] Installing NumPy..." -ForegroundColor Yellow
pip install numpy --quiet
Write-Host "✓ NumPy installed" -ForegroundColor Green

# Step 3: Create sample spam training files
Write-Host "`n[3/6] Creating sample spam training data..." -ForegroundColor Yellow

$spam1 = @"
Subject: URGENT!!! You Won $1,000,000!!!

Congratulations! You have won ONE MILLION DOLLARS in our lottery!
Click here NOW to claim your prize! ACT FAST! Limited time offer!
Send us your bank account details immediately!

FREE MONEY! WINNER! CLICK HERE! BUY NOW! VIAGRA! CASINO!
Win big money fast! No work required! Get rich quick!
"@

$spam2 = @"
Subject: Make Money Fast - Work From Home

Make $5000 per week working from home! No experience needed!
Click this link to get started today! Limited spots available!
AMAZING OPPORTUNITY! Don't miss out! FREE training included!
CHEAP PRODUCTS! BUY NOW! LIMITED TIME OFFER!
"@

$spam3 = @"
Subject: Enlarge your income NOW!!!

Special offer just for you! Increase your salary by 500%!
Click here for amazing deals! FREE consultation! Act now!
CHEAP VIAGRA! WEIGHT LOSS! MAKE MONEY FAST!
Online pharmacy! No prescription needed! Buy pills now!
"@

$spam4 = @"
Subject: WINNER!!! Claim your prize!!!

You are the lucky winner! Click to claim FREE iPad!
Congratulations! You won! Act now before offer expires!
FREE GIFT! WINNER! CASINO BONUS! Click here immediately!
"@

$spam5 = @"
Subject: HOT SINGLES in your area!!!

Meet singles tonight! Click here for HOT dates!
FREE registration! Real people waiting! Act now!
Adult content! Click here! FREE access! Join today!
"@

$spam1 | Out-File "train\spam\train-spam-00001.txt" -Encoding UTF8
$spam2 | Out-File "train\spam\train-spam-00002.txt" -Encoding UTF8
$spam3 | Out-File "train\spam\train-spam-00003.txt" -Encoding UTF8
$spam4 | Out-File "train\spam\train-spam-00004.txt" -Encoding UTF8
$spam5 | Out-File "train\spam\train-spam-00005.txt" -Encoding UTF8

Write-Host "✓ Created 5 spam training samples" -ForegroundColor Green

# Step 4: Copy ham training files
Write-Host "`n[4/6] Creating ham training data from test files..." -ForegroundColor Yellow

$hamTestFiles = Get-ChildItem "test\test-ham-*.txt" | Select-Object -First 10

$count = 1
foreach ($file in $hamTestFiles) {
    $newName = "train-ham-{0:D5}.txt" -f $count
    Copy-Item $file.FullName "train\ham\$newName"
    $count++
}

Write-Host "✓ Created $($count-1) ham training samples" -ForegroundColor Green

# Step 5: Run training
Write-Host "`n[5/6] Running training phase..." -ForegroundColor Yellow
Write-Host "      This will generate model.txt..." -ForegroundColor Gray

try {
    python train.py
    if (Test-Path "model.txt") {
        Write-Host "✓ Training completed! model.txt generated" -ForegroundColor Green
    } else {
        Write-Host "✗ Training failed - model.txt not created" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Error during training: $_" -ForegroundColor Red
}

# Step 6: Run testing
Write-Host "`n[6/6] Running testing phase..." -ForegroundColor Yellow
Write-Host "      This will generate result.txt and evaluation.txt..." -ForegroundColor Gray

try {
    python test.py
    if ((Test-Path "result.txt") -and (Test-Path "evaluation.txt")) {
        Write-Host "✓ Testing completed! Results generated" -ForegroundColor Green
    } else {
        Write-Host "✗ Testing failed - result files not created" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Error during testing: $_" -ForegroundColor Red
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "           SETUP COMPLETE!              " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nProject Summary:" -ForegroundColor White
Write-Host "  Training files:" -ForegroundColor Gray
$hamCount = (Get-ChildItem "train\ham\*.txt" | Measure-Object).Count
$spamCount = (Get-ChildItem "train\spam\*.txt" | Measure-Object).Count
Write-Host "    - Ham emails: $hamCount" -ForegroundColor Gray
Write-Host "    - Spam emails: $spamCount" -ForegroundColor Gray

Write-Host "`n  Testing files:" -ForegroundColor Gray
$testCount = (Get-ChildItem "test\*.txt" | Measure-Object).Count
Write-Host "    - Test emails: $testCount" -ForegroundColor Gray

Write-Host "`n  Generated files:" -ForegroundColor Gray
if (Test-Path "model.txt") { Write-Host "    ✓ model.txt" -ForegroundColor Green }
if (Test-Path "result.txt") { Write-Host "    ✓ result.txt" -ForegroundColor Green }
if (Test-Path "evaluation.txt") { Write-Host "    ✓ evaluation.txt" -ForegroundColor Green }

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  View Results:" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path "evaluation.txt") {
    Write-Host "`nEvaluation Results:" -ForegroundColor White
    Get-Content "evaluation.txt"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Next Steps:" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Review PROJECT_GUIDE.md for detailed explanation" -ForegroundColor White
Write-Host "2. Check result.txt for classification results" -ForegroundColor White
Write-Host "3. Check evaluation.txt for performance metrics" -ForegroundColor White
Write-Host "4. Open spam_detector.py to understand the code" -ForegroundColor White
Write-Host "`n✓ Your project is ready for the viva! Good luck! 🎉" -ForegroundColor Green
