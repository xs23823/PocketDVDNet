import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let controller: CameraController

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }

        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.backgroundColor = .black
        controller.attachPreviewLayer(view.previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                if let message = camera.errorMessage {
                    Text(message)
                        .multilineTextAlignment(.center)
                        .padding()
                } else {
                    let paneWidth = (geometry.size.width - 2) / 2
                    HStack(spacing: 2) {
                        CameraPreviewView(session: camera.session, controller: camera)
                            .frame(width: paneWidth)
                            .clipped()

                        ZStack {
                            Color.black
                            if let image = camera.denoisedImage {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFill()
                            }
                        }
                        .frame(width: paneWidth)
                        .clipped()
                    }
                    .overlay(alignment: .top) {
                        HStack {
                            Text("Original")
                            Spacer()
                            Text("Denoised")
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.4))
                        .padding(.top, 4)
                    }
                    .overlay(alignment: .bottom) {
                        Text(String(format: "%.1f FPS", camera.fps))
                            .font(.footnote.monospacedDigit())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.black.opacity(0.4))
                            .clipShape(Capsule())
                            .padding(.bottom, 8)
                    }
                }
            }
            .ignoresSafeArea()
            .onAppear {
                camera.setDisplayOrientation(isPortrait: geometry.size.width < geometry.size.height)
                camera.start()
            }
            .onDisappear {
                camera.stop()
            }
            .onChange(of: geometry.size.width < geometry.size.height) { isPortrait in
                camera.setDisplayOrientation(isPortrait: isPortrait)
            }
        }
    }
}
