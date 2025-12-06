# final_setup_test.py
import os
import warnings

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_complete_setup():
    """Test the complete facial emotion detection setup"""
    print("🧪 COMPREHENSIVE SETUP TEST")
    print("=" * 50)
    
    success_count = 0
    total_tests = 7
    
    # Test 1: Basic imports
    print("\n1️⃣ Testing Basic Imports...")
    try:
        import numpy as np
        import cv2
        print(f"   ✅ NumPy: {np.__version__}")
        print(f"   ✅ OpenCV: {cv2.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False
    
    # Test 2: TensorFlow
    print("\n2️⃣ Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow: {tf.__version__}")
        # Test if GPU is available (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   🚀 GPU detected: {len(gpus)} device(s)")
        else:
            print("   💻 Using CPU (normal for M1 Mac)")
        success_count += 1
    except ImportError as e:
        print(f"   ❌ TensorFlow failed: {e}")
        return False
    
    # Test 3: MoviePy
    print("\n3️⃣ Testing MoviePy...")
    try:
        from moviepy.editor import VideoFileClip
        import moviepy
        print(f"   ✅ MoviePy: {moviepy.__version__}")
        success_count += 1
    except ImportError as e:
        print(f"   ❌ MoviePy failed: {e}")
        return False
    
    # Test 4: FER Library
    print("\n4️⃣ Testing FER Library...")
    try:
        from fer import FER
        print("   ✅ FER imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"   ❌ FER import failed: {e}")
        return False
    
    # Test 5: FER Initialization
    print("\n5️⃣ Testing FER Initialization...")
    try:
        detector = FER(mtcnn=True)
        print("   ✅ FER detector initialized")
        print("   📝 Note: First initialization downloads models (may take time)")
        success_count += 1
    except Exception as e:
        print(f"   ❌ FER initialization failed: {e}")
        return False
    
    # Test 6: Camera Access
    print("\n6️⃣ Testing Camera Access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera working - Resolution: {w}x{h}")
                success_count += 1
                
                # Keep frame for emotion test
                test_frame = frame.copy()
                cap.release()
            else:
                print("   ⚠️ Camera opened but can't read frames")
                cap.release()
                return False
        else:
            print("   ⚠️ Cannot access camera")
            return False
    except Exception as e:
        print(f"   ❌ Camera test failed: {e}")
        return False
    
    # Test 7: Complete Pipeline
    print("\n7️⃣ Testing Complete Emotion Detection Pipeline...")
    try:
        # Test emotion detection on captured frame
        emotions_result = detector.detect_emotions(test_frame)
        
        if emotions_result:
            print(f"   ✅ Detected {len(emotions_result)} face(s)")
            
            # Show details of first face
            face_data = emotions_result[0]
            emotions = face_data['emotions']
            box = face_data['box']
            
            # Find dominant emotion
            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]
            
            print(f"   🎭 Dominant emotion: {dominant} (confidence: {confidence:.3f})")
            print(f"   📊 Face location: {box}")
            
            # Show top 3 emotions
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            print("   🎯 Top 3 emotions:")
            for emotion, prob in sorted_emotions[:3]:
                print(f"      {emotion}: {prob:.3f}")
                
        else:
            print("   ⚠️ No faces detected (put your face in camera view)")
            print("   ✅ Pipeline working (just no face visible)")
        
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ Emotion detection pipeline failed: {e}")
        return False
    
    # Final Results
    print("\n" + "=" * 50)
    print(f"🎯 TEST RESULTS: {success_count}/{total_tests} PASSED")
    
    if success_count == total_tests:
        print("🎉 SETUP COMPLETE! You're ready to build the facial emotion detection system!")
        print("\n📋 Next Steps:")
        print("   1. Follow Phase 2 of the detailed implementation guide")
        print("   2. Create your first basic emotion detection script")
        print("   3. Build the real-time webcam integration")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before proceeding.")
        return False

def save_test_image():
    """Save a test image with emotion detection overlay"""
    try:
        import cv2
        from fer import FER
        
        detector = FER()
        cap = cv2.VideoCapture(0)
        
        print("📸 Taking test photo in 3 seconds...")
        print("Look at the camera!")
        
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"Photo in: {i}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow('Test Photo', frame)
            cv2.waitKey(1000)
        
        # Take final photo
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Detect emotions
            result = detector.detect_emotions(frame)
            
            if result:
                # Draw results
                for face in result:
                    x, y, w, h = face['box']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Get dominant emotion
                    emotions = face['emotions']
                    dominant = max(emotions, key=emotions.get)
                    confidence = emotions[dominant]
                    
                    # Add text
                    cv2.putText(frame, f"{dominant}: {confidence:.2f}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Save image
                filename = "outputs/setup_test_result.jpg"
                os.makedirs("outputs", exist_ok=True)
                cv2.imwrite(filename, frame)
                print(f"💾 Test result saved to: {filename}")
                
                # Show for 3 seconds
                cv2.imshow('Emotion Detection Result', frame)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                
            else:
                print("No faces detected in test photo")
        else:
            print("Failed to capture test photo")
            
    except Exception as e:
        print(f"Error saving test image: {e}")

if __name__ == "__main__":
    # Run main test
    setup_success = test_complete_setup()
    
    if setup_success:
        # Ask if user wants to save test image
        response = input("\n📸 Would you like to save a test image with emotion detection? (y/n): ").lower().strip()
        if response.startswith('y'):
            save_test_image()
        
        print("\n🚀 Ready to start Phase 2 of the implementation!")
    else:
        print("\n🔧 Please fix the setup issues first.")