# src/test_installation.py
import cv2
from fer import FER
import numpy as np
import matplotlib.pyplot as plt
import sys

def test_opencv():
    """Test OpenCV installation"""
    try:
        # Try to access camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("✅ OpenCV working - Camera accessible")
            return True
        else:
            print("⚠️  OpenCV installed but camera not accessible")
            return False
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_fer():
    """Test FER library"""
    try:
        detector = FER()
        print("✅ FER library working")
        return True
    except Exception as e:
        print(f"❌ FER test failed: {e}")
        return False

def test_other_libraries():
    """Test other required libraries"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ NumPy and Matplotlib working")
        return True
    except Exception as e:
        print(f"❌ Library test failed: {e}")
        return False

def main():
    print("🔍 Testing installation...")
    print(f"Python version: {sys.version}")
    
    opencv_ok = test_opencv()
    fer_ok = test_fer()
    other_ok = test_other_libraries()
    
    if opencv_ok and fer_ok and other_ok:
        print("\n🎉 All tests passed! Ready to proceed.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before continuing.")

if __name__ == "__main__":
    main()