# src/webcam_basic.py
import cv2
import sys

def test_camera_access():
    """Test camera access with detailed diagnostics"""
    
    print("🔍 Testing camera access...")
    
    # Try different camera indices
    for camera_index in range(3):  # Test cameras 0, 1, 2
        print(f"Testing camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ Camera {camera_index} working:")
                print(f"   Resolution: {width}x{height}")
                print(f"   FPS: {fps}")
                
                # Show frame for 2 seconds
                cv2.imshow(f'Camera {camera_index} Test', frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                cap.release()
                return camera_index
            else:
                print(f"❌ Camera {camera_index} opened but can't read frames")
        else:
            print(f"❌ Can't open camera {camera_index}")
        
        cap.release()
    
    print("❌ No working cameras found!")
    return None

def basic_webcam_loop():
    """Basic webcam loop with proper error handling"""
    
    # Test camera first
    camera_index = test_camera_access()
    if camera_index is None:
        return
    
    print(f"\n🎥 Starting webcam with camera {camera_index}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Toggle recording info")
    print("  's' - Save current frame")
    
    cap = cv2.VideoCapture(camera_index)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    show_info = True
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add info overlay
        if show_info:
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'r' to toggle info", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Basic Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            show_info = not show_info
        elif key == ord('s'):
            filename = f"../outputs/frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved frame to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 Total frames processed: {frame_count}")

if __name__ == "__main__":
    basic_webcam_loop()