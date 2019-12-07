import Cocoa
typealias NSUIViewController = NSViewController

import MetalKit
import ModelIO

class ViewController: NSUIViewController {
    var renderer: Renderer!
    var cameraController: CameraController!
    
    var mtkView: MTKView {
        return view as! MTKView
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let device = MTLCreateSystemDefaultDevice()!
        mtkView.device = device
        mtkView.clearColor = MTLClearColorMake(0.3, 0.3, 0.3, 1.0)
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.sampleCount = 4
        
        renderer = Renderer(view: mtkView, device: device)
        mtkView.delegate = renderer

        cameraController = CameraController()
        renderer.viewMatrix = cameraController.viewMatrix

        mtkView.drawableSize.width = 256
        mtkView.drawableSize.height = 256

        mtkView.framebufferOnly = false
    }

    override func mouseDown(with event: NSEvent) {
        var point = view.convert(event.locationInWindow, from: nil)
        point.y = view.bounds.size.height - point.y
        cameraController.startedInteraction(at: point)
    }

    override func mouseDragged(with event: NSEvent) {
        var point = view.convert(event.locationInWindow, from: nil)
        point.y = view.bounds.size.height - point.y
        cameraController.dragged(to: point)
        renderer.viewMatrix = cameraController.viewMatrix
    }
}

