# BARPATH: Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Replace Your Files

Download all the improved files and replace your originals:

```bash
# Backup your originals first
mkdir backup
cp 1_collect_data.py 2_analyze_data.py 5_critique_lift.py barpath.py backup/

# Replace with improved versions
# (Copy the downloaded files into your project directory)
```

**Files to replace:**
- ✅ `1_collect_data.py` → Improved data collection
- ✅ `2_analyze_data.py` → Better analysis with interpolation
- ✅ `5_critique_lift.py` → Comprehensive critique engine
- ✅ `barpath.py` → Enhanced conductor script
- ✅ `requirements.txt` → Complete dependencies

**Files that don't need changes:**
- ⚪ `3_generate_graphs.py` → Already functional
- ⚪ `4_render_video.py` → Works but could be improved later

### Step 2: Verify Dependencies

```bash
python barpath.py --check-deps
```

You should see:
```
🔍 Checking dependencies...
   ✅ cv2
   ✅ mediapipe
   ✅ ultralytics
   ✅ pandas
   ✅ numpy
   ✅ matplotlib
   ✅ scipy

✅ All dependencies available
```

If anything is missing:
```bash
pip install -r requirements.txt
```

### Step 3: Test with Your Video

```bash
python barpath.py \
    --input_video your_clean.mp4 \
    --model best.pt \
    --lift_type clean \
    --output_video output.mp4
```

### Step 4: Review Outputs

You should now see:

**1. Better Progress Reporting:**
```
======================================================================
                  STEP 1: COLLECTING RAW DATA
======================================================================

📹 Video Info:
   Resolution: 1920x1080
   FPS: 30.00
   Total Frames: 450
   Duration: 15.00 seconds

🤖 Initializing MediaPipe Pose...
🎯 Loading YOLO model...
   ✅ Model loaded successfully

🔄 Processing frames...
Collecting Data: 100%|███████████████| 450/450 [07:30<00:00,  1.0it/s]

======================================================================
                    📊 COLLECTION STATISTICS
======================================================================
Frames Processed: 450/450
Pose Detected:    442 (98.2%)
Barbell Detected: 389 (86.4%)
Optical Flow Active: 412 (91.6%)

✅ Raw data saved to: raw_data.pkl
   File size: 45.23 MB
```

**2. Data Quality Warnings:**
```
⚠️  WARNING: Barbell detection rate is low. Ensure barbell endcap is visible.
```

**3. Analysis Metrics:**
```
📈 Data quality metrics:
   Path smoothness (avg jerk): 234.7 px/s³
   Peak velocity: 456.2 px/s
   Peak acceleration: 123.4 px/s²
```

**4. Structured Critique:**
```
📋 TECHNICAL ANALYSIS RESULTS
=================================================================

⚠️  Found 3 technical concern(s):

1. 🔴 MAJOR: Early Arm Bend
   ├─ Issue: Arms began bending 0.15s before full extension
   ├─ Impact: Reduces power transfer from legs, limits bar height
   └─ Fix: Keep arms straight and relaxed. Think 'legs then arms'

2. 🔴 MAJOR: Incomplete Extension
   ├─ Issue: Knees only extended to 158.3° (target: >165°)
   ├─ Impact: Reduces upward force on bar, limits lift height
   └─ Fix: Focus on explosive triple extension (ankles, knees, hips)

3. 🟡 MODERATE: Inefficient Bar Path
   ├─ Issue: Bar deviated 12.3% of frame width horizontally
   ├─ Impact: Energy wasted on horizontal motion, harder to catch
   └─ Fix: Keep bar close to body. Focus on vertical pull

─────────────────────────────────────────────────────────────────
PRIORITY FOCUS AREAS:
  🔴 2 major issue(s) - Address these first
  🟡 1 moderate issue(s) - Work on these next
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "FileNotFoundError: 1_collect_data.py"
**Solution:** Make sure you're in the correct directory:
```bash
ls -la *.py
# Should see: 1_collect_data.py, 2_analyze_data.py, etc.
```

### Problem: Low barbell detection (<50%)
**Solutions:**
1. Check barbell endcap is visible in video
2. Improve lighting/contrast in recording
3. Verify YOLO model is trained correctly
4. Try recording from a better angle

### Problem: Low pose detection (<80%)
**Solutions:**
1. Ensure lifter's full body visible
2. Improve lighting
3. Reduce motion blur (faster shutter speed)
4. Check for occlusions

### Problem: Graphs are still spiky
**Check:**
```bash
# Smoothing enabled by default
python barpath.py ... # Should be smooth

# If you disabled it:
python barpath.py ... --no-smooth  # Don't do this
```

### Problem: "No technical issues detected" but there are problems
**Possible causes:**
1. Video doesn't show complete lift
2. Missing data (check barbell detection rate)
3. Wrong lift type specified
4. Video has multiple attempts (only single-attempt supported)

**Debug:**
```bash
# Run with verbose flag
python 5_critique_lift.py --input final_analysis.csv --lift_type clean --verbose

# Check the analysis CSV
# Look for NaN values in key columns
head -20 final_analysis.csv
```

---

## 📊 Understanding the Output

### Collection Statistics

```
Pose Detected:    442/450 (98.2%)  ✅ Excellent
Barbell Detected: 389/450 (86.4%)  ✅ Good
Optical Flow Active: 412/450 (91.6%)  ✅ Good
```

**What's good:**
- Pose detection >95%
- Barbell detection >70%
- Optical flow >85%

**What needs attention:**
- Pose detection <80% → Video quality issues
- Barbell detection <50% → Barbell not visible
- Optical flow <70% → Camera too shaky

### Critique Severity Levels

🔴 **MAJOR** - High impact issues
- Fix these first
- Significantly limiting performance
- Examples: Early arm bend, incomplete extension

🟡 **MODERATE** - Medium impact issues
- Address after major issues
- Affecting efficiency
- Examples: Bar path deviation, early foot contact

🟢 **MINOR** - Low impact issues
- Fine-tune after others
- Small optimizations
- Examples: (Currently none defined)

---

## 🎯 Next Steps

### 1. Test Thoroughly
Run on 5-10 different videos to validate improvements:
```bash
for video in videos/*.mp4; do
    python barpath.py --input_video "$video" --model best.pt --lift_type clean --no-video
done
```

### 2. Compare with Original
Keep one video and run both versions:
```bash
# Original version
python barpath_old.py --input test.mp4 --model best.pt --lift_type clean

# New version
python barpath.py --input test.mp4 --model best.pt --lift_type clean
```

Compare:
- Error rates (should be lower)
- Detection statistics (should be higher)
- Critique quality (should be more detailed)
- Graph smoothness (should be better)

### 3. Document Edge Cases
Note any videos that still don't work well:
- What went wrong?
- What detection rates did you get?
- What does the critique say?

This helps identify remaining issues.

### 4. Prepare for GitHub
When ready to publish:

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit: Improved barpath v1.0"

# Add model with git-lfs (models are large)
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add git-lfs for model files"

# Create repository on GitHub and push
git remote add origin https://github.com/yourusername/barpath.git
git push -u origin main
```

---

## 💡 Pro Tips

### Speed Up Processing

Skip video rendering for faster iteration:
```bash
python barpath.py ... --no-video
# Saves 60-80% of processing time
```

### Disable Smoothing for Raw Analysis

If you want unfiltered data:
```bash
python barpath.py ... --no-smooth
```

### Run Critique Only

If you already have analysis CSV:
```bash
python 5_critique_lift.py --input final_analysis.csv --lift_type clean
```

### Batch Processing

Process multiple videos:
```bash
#!/bin/bash
for video in videos/*.mp4; do
    basename=$(basename "$video" .mp4)
    python barpath.py \
        --input_video "$video" \
        --model best.pt \
        --lift_type clean \
        --output_video "outputs/${basename}_output.mp4" \
        --no-video  # Skip rendering for speed
done
```

---

## 📚 Documentation Files

You now have:

1. **README.md** - Comprehensive documentation
2. **IMPROVEMENTS_SUMMARY.md** - Detailed technical improvements
3. **BEFORE_AFTER_COMPARISON.md** - Code comparison examples
4. **QUICK_START.md** (this file) - Get running fast

Read them in this order:
1. QUICK_START.md (you are here) ← Start here
2. README.md ← Usage and setup
3. BEFORE_AFTER_COMPARISON.md ← Understand changes
4. IMPROVEMENTS_SUMMARY.md ← Deep technical details

---

## ✅ Success Checklist

After implementing, you should see:

- [x] Dependencies check passes
- [x] Progress bars show statistics
- [x] Data collection completes without KeyError
- [x] Analysis produces smooth graphs
- [x] Critique provides structured feedback
- [x] Error messages are informative
- [x] Detection rates are reported
- [x] Warnings appear for low quality data

If all checked, you're ready to move forward! 🎉

---

**Questions?** Review the documentation files or check the inline comments in the code.
