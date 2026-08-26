# PocketDVDNet iOS Demo

A minimal SwiftUI app that runs the trained 5-frame PocketDVDNet model on an
iPhone, showing a live side-by-side view: original camera feed on the left,
denoised output on the right.

## Requirements

- Xcode 16+ (tested with Xcode 26)
- An iPhone running iOS 16+ with Developer Mode enabled
  (Settings → Privacy & Security → Developer Mode)
- [xcodegen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
  — only needed if you modify `project.yml`

## Build and run

The Xcode project is committed, so you can open it directly:

```bash
open PocketDVDNet.xcodeproj
```

1. Select your iPhone as the run destination (connect via USB and trust the
   computer if prompted).
2. Set your signing team: Target → Signing & Capabilities → Team
   (a free personal team works; apps expire after 7 days).
3. Press Cmd+R to build and run.
4. Grant camera permission when prompted.

Alternatively, build and install from the command line:

```bash
xcodebuild -project PocketDVDNet.xcodeproj -scheme PocketDVDNet \
  -destination 'id=DEVICE_ID' -allowProvisioningUpdates build
xcrun devicectl device install app --device DEVICE_ID \
  ~/Library/Developer/Xcode/DerivedData/PocketDVDNet-*/Build/Products/Debug-iphoneos/PocketDVDNet.app
```

Find your device id with `xcrun devicectl list devices`.

## How it works

- `AVCaptureSession` delivers 1280×720 frames; each is scaled to 640×360 via
  CoreImage into a rotating pool of five pixel buffers.
- Once five frames are buffered, every new frame triggers an `MLModel`
  prediction; the output replaces the denoised pane.
- Portrait/landscape rotation is handled for both the preview layer and the
  model output.

## The model

The app bundles a pre-converted Core ML model at
`Resources/PocketDVDNet.mlpackage` (640×360, noise sigma 30). It was produced
from the PyTorch checkpoint by the top-level `convert_to_coreml.py`:

```bash
.venv/bin/python convert_to_coreml.py --width 640 --height 360 --noise-sigma 30
cp -R pocket_models/PocketDVDNet.mlpackage ios/Resources/
```

The noise sigma is baked into the graph, so rebuilding with a different sigma
or resolution requires re-running the conversion.

## Notes

- If `project.yml` is modified, regenerate the project with `xcodegen` (run in
  this directory).
- The iOS simulator's virtual camera pipes your Mac's webcam and can be flaky
  (err -12782); a real device is the reliable path.
