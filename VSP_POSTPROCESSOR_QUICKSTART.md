# VSP Post-Processor Quick Start

## TL;DR

Control VSP's image post-processing from Mediator's command line:

```bash
# Enable visual masking (black rectangles over detected objects)
python request.py --provider vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_mask

# Enable inpainting (remove detected objects)
python request.py --provider vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_edit

# Enable zoom (crop to detected region)
python request.py --provider vsp --max_tasks 10 --vsp_postproc --vsp_postproc_method zoom_in

# Disable (default)
python request.py --provider vsp --max_tasks 10
```

## Three Simple Flags

| Flag | Description | Values |
|------|-------------|--------|
| `--vsp_postproc` | Enable post-processing | (flag, no value) |
| `--vsp_postproc_backend` | Choose backend | `ask` (default), `sd` (future) |
| `--vsp_postproc_method` | Choose method | `visual_mask`, `visual_edit`, `zoom_in` |

## Methods Comparison

| Method | Effect | Use Case | Speed |
|--------|--------|----------|-------|
| `visual_mask` | Black rectangles | Testing detection | ⚡ Very fast |
| `visual_edit` | Inpainting (remove) | Content-aware removal | ⚡ Fast |
| `zoom_in` | Crop and zoom | Focus on object | ⚡ Very fast |

## Works With

- ✅ `--provider vsp`
- ✅ `--provider comt_vsp`
- ❌ Other providers (ignored)

## Example Commands

### Test with 5 tasks
```bash
python request.py --provider vsp --max_tasks 5 --vsp_postproc --vsp_postproc_method visual_mask
```

### Full eval run
```bash
python request.py --provider vsp --vsp_postproc --vsp_postproc_method visual_edit
```

### With CoMT
```bash
python request.py --provider comt_vsp --comt_sample_id "creation-10003" --max_tasks 10 --vsp_postproc --vsp_postproc_method visual_mask
```

### Specific categories
```bash
python request.py --provider vsp --categories "01-Illegal_Activity" --max_tasks 20 --vsp_postproc --vsp_postproc_method visual_edit
```

## Verify It Works

Check VSP debug logs:
```bash
grep "POST_PROCESSOR" output/job_*/details/vsp_*/*/*/output/vsp_debug.log
```

Expected: `[POST_PROCESSOR] ASK:visual_mask`

## "Before" Images (Automatic)

When post-processing is enabled, VSP automatically saves images **before** and **after** post-modification:

```
output/job_XXX/details/vsp_*/category/task_id/output/input/
├── image_0.jpg                          # Original input image
├── before_postproc_detection_*.png      # VSP-annotated (before post-processing)
└── <hash>.png                           # Post-modified image (after masking/inpainting/zoom)
```

This allows you to compare:
1. **Original image** → What the user provided
2. **Before images** → What VSP's vision tools detected/annotated
3. **Post-modified images** → What the LLM actually sees (after post-processing)

## Full Documentation

- `VSP_POSTPROCESSOR_USAGE.md` - Detailed usage guide
- `VSP_POSTPROCESSOR_INTEGRATION_SUMMARY.md` - Implementation details
