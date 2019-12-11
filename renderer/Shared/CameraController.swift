
import QuartzCore
import simd

class CameraController {
    var viewMatrix: float4x4 {
        return float4x4(translationBy: float3(0, 0, -radius)) *
               float4x4(rotationAbout: float3(1, 0, 0), by: altitude) *
               float4x4(rotationAbout: float3(0, 1, 0), by: azimuth)
        
    }
    
    //var radius: Float = 3
    // Mim: 1 Max: 25
    var radius: Float = 3
    var sensitivity: Float = 0.01
    let minAltitude: Float = -.pi / 4
    let maxAltitude: Float = .pi / 2
    
    private var altitude: Float = 0
    private var azimuth: Float = 0.5

    private var lastPoint: CGPoint = .zero
    
    func startedInteraction(at point: CGPoint) {
        lastPoint = point
    }
    
    func dragged(to point: CGPoint) {
        let deltaX = Float(lastPoint.x - point.x)
        let deltaY = Float(lastPoint.y - point.y)
        azimuth += -deltaX * sensitivity
        altitude += -deltaY * sensitivity
        altitude = min(max(minAltitude, altitude), maxAltitude)
        lastPoint = point
    }

    func moveCamera() {
        radius = (Float(arc4random_uniform(60)) + 20) / 10
        //radius = 2
        //print("generated radius: ", radius)
        azimuth = (Float(arc4random_uniform(3142)) + 1) / 1000
        //print("generated azimuth: ", azimuth)
        altitude = (Float(arc4random_uniform(3142)) + 1) / 1000

#if false
        //let deltaX = Float(2.0)
        //let deltaY = Float(2.0)

        /*if (azimuth > -6.28) {
           print("azimuth: %d count: %d", azimuth, captureCounter)
           azimuth += -deltaX * sensitivity
           captureCounter += 1
        }*/
        /*if (altitude > -7) {
            altitude += -deltaY * sensitivity
            print("altitude: %d count: %d", altitude, captureCounter)
            captureCounter += 1
        }*/

        //altitude = min(max(minAltitude, altitude), maxAltitude)
#endif
    }
}
