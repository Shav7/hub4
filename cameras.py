"""Show every camera side by side with its OpenCV index; press the index digit to remember it.
Usage: python cameras.py        (q to quit without choosing)"""
import sys

sys.path.insert(0, "src")
import cv2

from vision import probe_cameras, remember_camera


def main() -> None:
    indices = probe_cameras()
    if not indices:
        sys.exit("no cameras found")
    captures = {i: cv2.VideoCapture(i) for i in indices}
    print(f"cameras: {indices}. Press the digit of the USB camera to remember it, q to quit.")
    try:
        while True:
            tiles = []
            for i, cap in captures.items():
                ok, frame = cap.read()
                tile = cv2.resize(frame, (640, 360)) if ok else cv2.UMat(360, 640, cv2.CV_8UC3).get()
                cv2.putText(tile, f"[{i}] press {i}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
                tiles.append(tile)
            cv2.imshow("hub4 cameras", cv2.hconcat(tiles))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if ord("0") <= key <= ord("9") and (key - ord("0")) in captures:
                chosen = key - ord("0")
                remember_camera(chosen, f"camera {chosen}")
                print(f"remembered camera {chosen}")
                break
    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
