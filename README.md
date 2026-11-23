# FM-TransUNet and SteelOM2SEM

Research code for **Multimodal Large Model-Based Translation of Optical Microscopy Images into SEM-Like Images for GCr15 Steel**.

This repository provides the geometric OM–SEM registration tools, reviewed-pair dataset exporter, and supervised FM-TransUNet implementation. The training release uses the study backbone, reconstruction and spectral/morphology objectives, grouped sampling, and the published training budget. The command-line entry points wrap these methods in a portable workflow, independent of the original experiment runner.

The complete 769-pair dataset, source micrographs, pair-level cohort manifests and historical checkpoints are pending release. The [earlier 156-pair SteelOM2SEM release](https://github.com/YixiaoOneSmile/PI-TransUNet-SteelOM2SEM) has a different split and is useful for inspecting the file layout. It does not reproduce the expanded-study comparisons. The commercial generator experiments are outside this repository.

## Install

Use Python 3.11 or later and install a PyTorch build appropriate for the CPU or CUDA environment.

\`\`\`sh
python -m pip install -r requirements.txt
\`\`\`

Training with the study objective uses ImageNet VGG19 weights. Set \`FM_VGG19_WEIGHTS\` to a local official weight file or allow torchvision to download the weights when training starts. LPIPS evaluation loads its AlexNet weights. The lightweight smoke checks use neither download.

## 1. Geometric registration

Provide corresponding OM and SEM input images as 8-bit files. Coordinates and transforms refer to their pixel grids. For landmark-based registration, supply a JSON object containing ordered \`om\` and \`sem\` arrays of at least three non-collinear \`[x, y]\` coordinates. For a texture-based proposal, supply the nominal transform scale in SEM pixels per OM pixel.

\`\`\`sh
python -m fmtransunet.registration om.png sem.png --landmarks landmarks.json --output outputs/proposal
python -m fmtransunet.registration om.png sem.png --scale 1.5 --output outputs/texture-proposal
python -m fmtransunet.refinement om.png sem.png outputs/proposal/transform.json valid_mask.png outputs/refined
\`\`\`

The registration tool saves the OM-to-SEM matrix, aligned OM, support mask and review overlay. The refinement tool matches local texture in both directions, fits a bounded correction on accepted tiles, and evaluates it on separate tiles. Review the overlays and exclude the indentation, its deformation zone, the SEM footer and any other invalid area before exporting pairs. Vision assistance in the study supported code development and visual review. These tools calculate the transforms locally and require no model service.

## 2. Build registered patches and development folds

The exporter takes a JSON list of reviewed field pairs. Paths are relative to the manifest. The SEM is the target grid, the mask is an 8-bit image with nonzero values only in the valid overlap, and each patch origin is declared in SEM pixel coordinates. Use the same field-group identifier for acquisitions that depict overlapping material. Any file reused by multiple pairs must stay within a single group. Pixel pitch is in micrometres per native SEM pixel.

\`\`\`json
[
  {
    "pair_id": "example_01",
    "group": "field_group_01",
    "material": "spheroidized_annealed",
    "om": "om.png",
    "sem": "sem.png",
    "valid_mask": "valid_mask.png",
    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "accepted": true,
    "pitch_um": 0.134765625,
    "crop_xy": [[64, 64]]
  }
]
\`\`\`

Replace the example matrix and coordinates with the reviewed registration result. For a reviewed refined image, set \`aligned_om\` to the refined PNG and use its SEM-grid mask. The original OM path is still needed for provenance checks. Valid margins, groups and complementary views follow the study protocol. The supported material labels are \`spheroidized_annealed\`, \`quenched\` and \`tempered\`. Include at least three field groups per condition present.

\`\`\`sh
python -m fmtransunet.dataset reviewed_pairs.json data/expanded
\`\`\`

The exporter produces 256 × 256 paired PNGs and \`patches.json\` and \`folds.json\`. It preserves the original grayscale SEM pixel values in every crop. The 3-fold splits are for grouped development validation.

## 3. Supervised learning

\`\`\`sh
python -m fmtransunet.train data/expanded/patches.json --fold 0 --seed 42 --output outputs/parent
python -m fmtransunet.train data/expanded/patches.json --fold 0 --seed 42 \\
  --parent outputs/parent/best.pt --variant fm --output outputs/fm
\`\`\`

The parent uses 200 epochs with 24 updates per epoch. FM-TransUNet fine-tunes the final two decoder stages and output layer for 1,200 updates with frozen batch-normalization state. The four variants are \`fft\`, \`window\`, \`morphology\` and \`fm\`. The script reports field- and condition-balanced development metrics.

For the original study folds, use \`--paper-config configs/paper.json\` to select the original fold coefficients and supply the matching parent checkpoint. For new data, omit that flag to calibrate coefficients on training patches. Repository smoke checks exercise both training phases with synthetic data. Exact study-score reproduction requires the expanded data, the original cohort manifests, parent checkpoints and the original runtime settings.

## Checks

\`\`\`sh
python -m unittest discover -s tests -v
\`\`\`

The MIT license applies to the code in this repository.
