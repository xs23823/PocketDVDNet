import AVFoundation
import CoreImage
import CoreML
import UIKit

final class CameraController: NSObject, ObservableObject {
    @Published var denoisedImage: UIImage?
    @Published var fps: Double = 0
    @Published var errorMessage: String?

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "com.tomd.pocketdvdnet.camera")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let ciContext = CIContext()

    private let frameCount = 5
    private let modelWidth = 640
    private let modelHeight = 360

    private var model: MLModel?
    private weak var previewLayer: AVCaptureVideoPreviewLayer?
    private var scaledBuffers: [CVPixelBuffer] = []
    private var scaledIndex = 0
    private var history: [CVPixelBuffer] = []
    private var displayOrientation: CGImagePropertyOrientation = .right
    private var isDisplayPortrait = true
    private var isConfigured = false
    private var receivedFrameCount = 0
    private var fpsEMA: Double = 0
    private var lastPredictionAt: CFTimeInterval = 0

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    self?.configureAndStart()
                } else {
                    DispatchQueue.main.async { self?.errorMessage = "Camera access denied." }
                }
            }
        default:
            DispatchQueue.main.async { self.errorMessage = "Camera access denied. Enable it in Settings." }
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning {
                session.stopRunning()
            }
        }
    }

    func attachPreviewLayer(_ layer: AVCaptureVideoPreviewLayer) {
        previewLayer = layer
        applyPreviewOrientation()
    }

    func setDisplayOrientation(isPortrait: Bool) {
        isDisplayPortrait = isPortrait
        displayOrientation = isPortrait ? .right : .up
        applyPreviewOrientation()
    }

    private func applyPreviewOrientation() {
        let target: AVCaptureVideoOrientation = isDisplayPortrait ? .portrait : .landscapeRight
        if let connection = previewLayer?.connection, connection.isVideoOrientationSupported {
            connection.videoOrientation = target
        }
    }

    private func configureAndStart() {
        sessionQueue.async { [self] in
            if isConfigured {
                if !session.isRunning { session.startRunning() }
                return
            }

            guard
                let modelURL = Bundle.main.url(forResource: "PocketDVDNet", withExtension: "mlmodelc")
                    ?? Bundle.main.url(forResource: "PocketDVDNet", withExtension: "mlpackagec"),
                let loadedModel = try? MLModel(contentsOf: modelURL)
            else {
                DispatchQueue.main.async { self.errorMessage = "PocketDVDNet model is missing from the app bundle." }
                return
            }
            model = loadedModel

            guard
                let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                    ?? AVCaptureDevice.default(for: .video),
                let input = try? AVCaptureDeviceInput(device: device)
            else {
                DispatchQueue.main.async { self.errorMessage = "No camera is available on this device." }
                return
            }

            session.beginConfiguration()
            session.sessionPreset = .high

            for existingInput in session.inputs {
                session.removeInput(existingInput)
            }
            guard session.canAddInput(input) else {
                session.commitConfiguration()
                DispatchQueue.main.async { self.errorMessage = "Could not add camera input." }
                return
            }
            session.addInput(input)

            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            }
            if let connection = videoOutput.connection(with: .video),
               connection.isVideoOrientationSupported {
                connection.videoOrientation = .landscapeRight
            }

            session.commitConfiguration()
            isConfigured = true
            session.startRunning()

            if !session.isRunning {
                DispatchQueue.main.async { self.errorMessage = Self.cameraUnavailableMessage }
                return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
                guard let self, self.receivedFrameCount == 0 else { return }
                if self.session.isRunning {
                    self.sessionQueue.async { self.session.stopRunning() }
                }
                DispatchQueue.main.async { self.errorMessage = Self.cameraUnavailableMessage }
            }
        }
    }

    private static let cameraUnavailableMessage: String = """
        The simulator's virtual camera could not start (err -12782).

        Fix one of these, then relaunch:
        1. System Settings > Privacy & Security > Camera > allow Simulator.
        2. Or run on a real iPhone for the real camera pipeline.
        """

    private func scaleToModel(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        if scaledBuffers.isEmpty {
            var buffers: [CVPixelBuffer] = []
            for _ in 0..<frameCount {
                var buffer: CVPixelBuffer?
                CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    modelWidth,
                    modelHeight,
                    kCVPixelFormatType_32BGRA,
                    nil,
                    &buffer
                )
                if let buffer {
                    buffers.append(buffer)
                }
            }
            scaledBuffers = buffers
        }

        let target = scaledBuffers[scaledIndex % scaledBuffers.count]
        scaledIndex += 1

        let source = CIImage(cvPixelBuffer: pixelBuffer)
        let scaleX = CGFloat(modelWidth) / source.extent.width
        let scaleY = CGFloat(modelHeight) / source.extent.height
        let scaled = source.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        ciContext.render(scaled, to: target)
        return target
    }

    private func updateFPS() {
        let now = CACurrentMediaTime()
        if lastPredictionAt > 0 {
            let instant = 1.0 / max(now - lastPredictionAt, 1e-6)
            fpsEMA = fpsEMA == 0 ? instant : fpsEMA * 0.9 + instant * 0.1
        }
        lastPredictionAt = now
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard
            let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
            let model
        else { return }

        receivedFrameCount += 1

        history.append(scaleToModel(pixelBuffer))
        if history.count > frameCount {
            history.removeFirst(history.count - frameCount)
        }
        guard history.count == frameCount else { return }

        var features: [String: Any] = [:]
        for (index, buffer) in history.enumerated() {
            features["frame\(index)"] = MLFeatureValue(pixelBuffer: buffer)
        }

        guard
            let provider = try? MLDictionaryFeatureProvider(dictionary: features),
            let result = try? model.prediction(from: provider),
            let outputBuffer = result.featureValue(for: "denoised")?.imageBufferValue
        else { return }

        var outputImage = CIImage(cvPixelBuffer: outputBuffer)
        if displayOrientation != .up {
            outputImage = outputImage.oriented(displayOrientation)
        }
        guard let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else { return }

        updateFPS()
        let image = UIImage(cgImage: cgImage)
        let fps = fpsEMA
        DispatchQueue.main.async { [self] in
            denoisedImage = image
            self.fps = fps
        }
    }
}
